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
//! WHAT THE ADDRESS RULE CANNOT SEE, and the probe that can. The address rule judges
//! the address, not the database behind it. Its exact limit: with `CI` exported, every
//! loopback port other than 7687/7474 is accepted — a copy of production on 27687 was
//! migrated that way (11/11) — and a live service bound to 17687
//! (`docker-compose.production.yml` takes `NEO4J_BOLT_PORT`) passes it without `CI`.
//! So [`neo4j_target`] also PROBES the database: a test database carries a
//! `(:__Harnais)` marker node, written by whoever provisions it (the CI workflow, the
//! `docker run` below); a reachable database without it is refused by name, `CI` or
//! not. A copy of production never has the marker. What the probe cannot see: a
//! marker someone wrote on a database that is not a test one.
//!
//! `MEILISEARCH_URL` / `MEILISEARCH_KEY` are the service's variables too, and
//! `AppState::new` rewrites the SETTINGS of the Meilisearch indexes it opens:
//! [`meili_target`] applies the same address rule (loopback, never 7700, outside CI
//! only the harness port 17700). It has no marker probe.
//!
//! [`Cleanup`] is the other half: the Cypher it is given runs when it is dropped,
//! panic or not, so a failing assertion does not leave test nodes behind.
//!
//! A throwaway Neo4j, with its marker:
//!   docker run -d --rm --name po-int -p 17687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5
//!   docker exec po-int cypher-shell -u neo4j -p testpassword "MERGE (:__Harnais {name: 'local'})"
//! A throwaway Meilisearch:
//!   docker run -d --rm --name po-meili -p 17700:7700 -e MEILI_MASTER_KEY=test-master-key getmeili/meilisearch:v1.34

use neo4rs::{query, Graph};

/// The port of the throwaway Neo4j of the test harness.
pub const HARNESS_PORT: u16 = 17687;
/// Ports of the live Neo4j: never a test target, whatever the environment says.
const LIVE_PORTS: [u16; 2] = [7687, 7474];
/// The port of the throwaway Meilisearch of the test harness, and the live one's.
pub const HARNESS_MEILI_PORT: u16 = 17700;
const LIVE_MEILI_PORT: u16 = 7700;
/// The label of the node a test database carries (see the module doc).
pub const HARNESS_MARKER: &str = "__Harnais";

#[derive(Clone, Debug)]
pub struct Neo4jTarget {
    pub uri: String,
    pub user: String,
    pub password: String,
}

/// Host and port of a URL (`scheme://host[:port][/...]`).
fn host_and_port(uri: &str, default_port: u16) -> Option<(String, u16)> {
    let rest = uri.split_once("://").map_or(uri, |(_, r)| r);
    let authority = rest.split('/').next()?;
    let authority = authority.rsplit('@').next()?;
    if let Some(v6) = authority.strip_prefix('[') {
        let (host, tail) = v6.split_once(']')?;
        let port = tail
            .strip_prefix(':')
            .map_or(Some(default_port), |p| p.parse().ok())?;
        return Some((host.to_string(), port));
    }
    match authority.rsplit_once(':') {
        Some((host, port)) => Some((host.to_string(), port.parse().ok()?)),
        None => Some((authority.to_string(), default_port)),
    }
}

/// Why `uri` is not a Neo4j a test may write to; `Ok` if it is.
pub fn check_harness_uri(uri: &str, in_ci: bool) -> Result<(), String> {
    check_address(uri, 7687, &LIVE_PORTS, HARNESS_PORT, in_ci)
}

/// Why `url` is not a Meilisearch a test may write to; `Ok` if it is.
pub fn check_harness_meili_url(url: &str, in_ci: bool) -> Result<(), String> {
    check_address(url, 80, &[LIVE_MEILI_PORT], HARNESS_MEILI_PORT, in_ci)
}

/// The address rule shared by both services: this machine, never a live port,
/// outside CI only the harness port.
fn check_address(
    addr: &str,
    default_port: u16,
    live: &[u16],
    harness: u16,
    in_ci: bool,
) -> Result<(), String> {
    if addr.trim().is_empty() {
        return Err("the address is unset/empty".to_string());
    }
    let (host, port) = host_and_port(addr, default_port)
        .ok_or_else(|| format!("{addr}: not a URL with a host and a port"))?;
    if !matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1") {
        return Err(format!("{addr}: host {host} is not this machine"));
    }
    if live.contains(&port) {
        return Err(format!("{addr}: port {port} is the live service's"));
    }
    if !in_ci && port != harness {
        return Err(format!(
            "{addr}: outside CI the only target is port {harness}"
        ));
    }
    Ok(())
}

/// The target named by the environment, or the harness default; panics on a refusal
/// of the address, or of a reachable database without the harness marker.
pub fn neo4j_target() -> Neo4jTarget {
    let target = target_for(
        std::env::var("NEO4J_URI").ok(),
        std::env::var_os("CI").is_some(),
    );
    assert_harness_marker(&target);
    target
}

/// The probe: a reachable database must carry a `(:__Harnais)` node. Unreachable is
/// not judged here (the suite skips or fails on its own). Its own thread and runtime:
/// the callers are synchronous, inside or outside a tokio runtime.
pub fn assert_harness_marker(target: &Neo4jTarget) {
    let t = target.clone();
    let markers = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()?;
        rt.block_on(async {
            let graph = Graph::new(&t.uri, &t.user, &t.password).await.ok()?;
            let q = query(&format!("MATCH (m:{HARNESS_MARKER}) RETURN count(m) AS c"));
            let mut rows = graph.execute(q).await.ok()?;
            rows.next().await.ok()??.get::<i64>("c").ok()
        })
    })
    .join()
    .ok()
    .flatten();
    if markers == Some(0) {
        panic!(
            "REFUSED to point a writing test at this Neo4j: {} answers but carries no \
             (:{HARNESS_MARKER}) node, so it is not a test database. On a throwaway one, run: \
             MERGE (:{HARNESS_MARKER} {{name: 'local'}})",
            target.uri
        );
    }
}

/// The Meilisearch a test may write to (`AppState::new` rewrites index settings):
/// the environment's, or the harness default; panics on a refusal.
pub fn meili_target() -> MeiliTarget {
    meili_target_for(
        std::env::var("MEILISEARCH_URL").ok(),
        std::env::var_os("CI").is_some(),
    )
}

/// [`meili_target`] with the environment passed in.
pub fn meili_target_for(url: Option<String>, in_ci: bool) -> MeiliTarget {
    let url = url.unwrap_or_else(|| format!("http://localhost:{HARNESS_MEILI_PORT}"));
    if let Err(why) = check_harness_meili_url(&url, in_ci) {
        panic!(
            "REFUSED to point a writing test at this Meilisearch: {why}. Start a throwaway one: \
             docker run -d --rm --name po-meili -p {HARNESS_MEILI_PORT}:7700 -e MEILI_MASTER_KEY=test-master-key getmeili/meilisearch:v1.34"
        );
    }
    MeiliTarget {
        url,
        key: std::env::var("MEILISEARCH_KEY").unwrap_or_else(|_| "test-master-key".into()),
    }
}

#[derive(Clone, Debug)]
pub struct MeiliTarget {
    pub url: String,
    pub key: String,
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
