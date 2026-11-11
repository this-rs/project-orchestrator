//! The Neo4j a test may WRITE to, and the cleanup it owes.
//!
//! `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` are also the variables of the
//! running service: a shell that loaded the service's `.env` would point a test
//! suite (and its `init_schema()`, and its writes) at the live database. A suite
//! that writes asks [`neo4j_target`] instead of reading the variables itself:
//!
//! - the host must be this machine (loopback);
//! - the live ports (7687 bolt, 7474 http) are refused;
//! - outside CI (`CI` unset) the only port accepted is the harness one, 17687
//!   (see `harness_neo4j.rs`). In CI the service container gets a random port
//!   on loopback, which no fixed number can name.
//!
//! A refusal is a PANIC with the reason, never a skip: a skipped suite is a green
//! run that tested nothing.
//!
//! [`Cleanup`] is the other half: the Cypher it is given runs when it is dropped,
//! panic or not, so a failing assertion does not leave test nodes behind.

use neo4rs::{query, Graph};

/// The port of the throwaway Neo4j of the test harness.
pub const HARNESS_PORT: u16 = 17687;
/// Ports of the live service: never a test target, whatever the environment says.
const LIVE_PORTS: [u16; 2] = [7687, 7474];

#[derive(Clone, Debug)]
pub struct Neo4jTarget {
    pub uri: String,
    pub user: String,
    pub password: String,
}

/// Host and port of a bolt URI (`bolt://host[:port][/...]`, port defaults to 7687).
fn host_and_port(uri: &str) -> Option<(String, u16)> {
    let rest = uri.split_once("://").map_or(uri, |(_, r)| r);
    let authority = rest.split('/').next()?;
    let authority = authority.rsplit('@').next()?;
    if let Some(v6) = authority.strip_prefix('[') {
        let (host, tail) = v6.split_once(']')?;
        let port = tail
            .strip_prefix(':')
            .map_or(Some(7687), |p| p.parse().ok())?;
        return Some((host.to_string(), port));
    }
    match authority.rsplit_once(':') {
        Some((host, port)) => Some((host.to_string(), port.parse().ok()?)),
        None => Some((authority.to_string(), 7687)),
    }
}

/// Why `uri` is not a Neo4j a test may write to; `Ok` if it is.
pub fn check_harness_uri(uri: &str, in_ci: bool) -> Result<(), String> {
    let (host, port) = host_and_port(uri)
        .ok_or_else(|| format!("{uri}: not a bolt URI with a host and a port"))?;
    if !matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1") {
        return Err(format!("{uri}: host {host} is not this machine"));
    }
    if LIVE_PORTS.contains(&port) {
        return Err(format!("{uri}: port {port} is the live service's"));
    }
    if !in_ci && port != HARNESS_PORT {
        return Err(format!(
            "{uri}: outside CI the only target is port {HARNESS_PORT}"
        ));
    }
    Ok(())
}

/// The target named by the environment, or the harness default; panics on a refusal.
pub fn neo4j_target() -> Neo4jTarget {
    target_for(
        std::env::var("NEO4J_URI").ok(),
        std::env::var_os("CI").is_some(),
    )
}

/// [`neo4j_target`] with the environment passed in (so a test can exercise the
/// refusal without touching the process environment other tests read).
pub fn target_for(uri: Option<String>, in_ci: bool) -> Neo4jTarget {
    let uri = uri.unwrap_or_else(|| format!("bolt://localhost:{HARNESS_PORT}"));
    if let Err(why) = check_harness_uri(&uri, in_ci) {
        panic!(
            "REFUSED to point a writing test at this Neo4j: {why}. Start a throwaway one: \
             docker run -d --rm --name po-int -p {HARNESS_PORT}:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5"
        );
    }
    Neo4jTarget {
        uri,
        user: std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into()),
        password: std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into()),
    }
}

/// Cypher run when dropped (each statement with one `$p` parameter), panic or not.
pub struct Cleanup {
    target: Neo4jTarget,
    statements: Vec<(&'static str, String)>,
}

impl Cleanup {
    pub fn new(target: &Neo4jTarget) -> Self {
        Self {
            target: target.clone(),
            statements: Vec::new(),
        }
    }

    /// Register a statement; register BEFORE creating what it removes.
    pub fn later(&mut self, cypher: &'static str, p: impl Into<String>) {
        self.statements.push((cypher, p.into()));
    }
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        let target = self.target.clone();
        let statements = std::mem::take(&mut self.statements);
        // Its own thread and runtime: `drop` may run while the test's runtime is
        // being torn down by a panic, and a pooled connection belongs to the
        // runtime that opened it.
        let _ = std::thread::spawn(move || {
            let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                return;
            };
            rt.block_on(async {
                let Ok(graph) = Graph::new(&target.uri, &target.user, &target.password).await
                else {
                    return;
                };
                for (cypher, p) in statements {
                    let _ = graph.run(query(cypher).param("p", p)).await;
                }
            });
        })
        .join();
    }
}
