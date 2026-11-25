//! Who owns the port we could not bind.
//!
//! A duplicate instance and a genuine failure deserve different exit codes.
//! launchd (`KeepAlive { SuccessfulExit = false }`) and systemd
//! (`Restart=on-failure`) relaunch a process that exits with a failure code,
//! so a duplicate that dies on `Address already in use` is relaunched forever
//! against a port it can never take. Measured 2026-11-25: `runs = 590` in one
//! day, one attempt every 5 s, `/tmp/orchestrator.log` at 1.0 GB of the same
//! four lines.
//!
//! The rule implemented here: if the port is demonstrably served by a live
//! orchestrator, the duplicate exits SUCCESSFULLY — the supervisor stops
//! relaunching it, the live instance keeps serving, and the message names what
//! to do. Anything else (a foreign process, a silent listener) keeps the
//! failure exit code: that is a real problem an operator must see.

use std::fmt;
use std::net::SocketAddr;
use std::time::Duration;

/// How long the probe may take. Bounded on purpose: it runs on the failure
/// path of a bind, i.e. during startup, and must never turn a fast failure
/// into a hang.
const PROBE_TIMEOUT: Duration = Duration::from_millis(400);

/// What answered on the port we could not bind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortOwner {
    /// A live orchestrator answered `/api/version`.
    Orchestrator { version: String },
    /// Something listens but does not answer like an orchestrator.
    Foreign,
    /// Nothing answered at all.
    Nobody,
}

/// The one bind failure that must not reach the supervisor as a failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortOwnedByOrchestrator {
    pub port: u16,
    pub version: String,
}

impl PortOwnedByOrchestrator {
    pub fn new(port: u16, version: impl Into<String>) -> Self {
        Self {
            port,
            version: version.into(),
        }
    }
}

impl fmt::Display for PortOwnedByOrchestrator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot bind 0.0.0.0:{}: a live Project Orchestrator (version {}) already serves this \
             port — this duplicate exits with status 0 so the supervisor (launchd KeepAlive, \
             systemd Restart=) stops relaunching it; the live instance keeps serving. To take \
             over: stop the live instance, then `launchctl kickstart -k gui/$(id -u)/land.ffs.orchestrator`.",
            self.port, self.version
        )
    }
}

impl std::error::Error for PortOwnedByOrchestrator {}

/// The exit code a startup failure must produce, when it is NOT the default
/// failure code.
///
/// `Some(0)` means: this is a duplicate, the supervisor must not relaunch it.
/// `None` means: keep the failure code — a foreign process or a real error must
/// stay visible to whoever supervises the server.
///
/// Kept here, and not inline in `main`, so the rule can be tested: the whole
/// point of this module is an exit code, which nothing else would catch.
pub fn successful_exit_code(error: &anyhow::Error) -> Option<i32> {
    error
        .downcast_ref::<PortOwnedByOrchestrator>()
        .map(|_duplicate| 0)
}

/// Ask the port who holds it. Never waits more than [`PROBE_TIMEOUT`].
pub async fn probe(port: u16) -> PortOwner {
    probe_addr(SocketAddr::from(([127, 0, 0, 1], port))).await
}

async fn probe_addr(addr: SocketAddr) -> PortOwner {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let exchange = async {
        let mut stream = tokio::net::TcpStream::connect(addr).await.ok()?;
        stream
            .write_all(b"GET /api/version HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .ok()?;
        let mut response = Vec::with_capacity(1024);
        stream.read_to_end(&mut response).await.ok()?;
        Some(response)
    };

    match tokio::time::timeout(PROBE_TIMEOUT, exchange).await {
        // It listens but never answered: a foreign process, not us.
        Err(_elapsed) => PortOwner::Foreign,
        // Connect refused, or the write failed: nothing is really there.
        Ok(None) => PortOwner::Nobody,
        Ok(Some(response)) => classify(&response),
    }
}

/// Decide what a raw HTTP response means.
///
/// A successful status is not enough, and neither is the PRESENCE of the two
/// keys `/api/version` carries: `{"version": null, "build": null}` has both, and
/// so does any service that happens to expose a `version` and a `build`. Taking
/// one of those for an orchestrator is the worst possible mistake here — the
/// duplicate would exit 0 (nothing restarts it) while a stranger holds the port,
/// so nobody serves. The SHAPE is required instead, as `VersionResponse` and
/// `VersionBuild` define it (`src/api/handlers.rs`): `version` a non-empty
/// string, `build` an object carrying at least one of the build markers. Only
/// one of the two markers is required, not both, so that a peer running an older
/// or newer build than this binary is still recognised.
fn classify(response: &[u8]) -> PortOwner {
    let text = String::from_utf8_lossy(response);
    let status_ok = text
        .lines()
        .next()
        .map(|line| line.contains(" 200"))
        .unwrap_or(false);
    if !status_ok {
        return PortOwner::Foreign;
    }
    let body = text.split("\r\n\r\n").nth(1).unwrap_or("");
    let value: serde_json::Value = match serde_json::from_str(body.trim()) {
        Ok(value) => value,
        Err(_) => return PortOwner::Foreign,
    };
    let version = value.get("version").and_then(|v| v.as_str()).unwrap_or("");
    let build_is_ours = value
        .get("build")
        .and_then(|build| build.as_object())
        .map(|build| {
            ["target", "profile"].iter().any(|key| {
                build
                    .get(*key)
                    .and_then(|v| v.as_str())
                    .is_some_and(|s| !s.is_empty())
            })
        })
        .unwrap_or(false);
    if version.is_empty() || !build_is_ours {
        return PortOwner::Foreign;
    }
    PortOwner::Orchestrator {
        version: version.to_string(),
    }
}

/// Test-only: answer `/api/version` like a live orchestrator on an ephemeral
/// port. Returns the port and the task that serves it (abort it to stop).
#[cfg(test)]
pub(crate) async fn spawn_fake_orchestrator(version: &str) -> (u16, tokio::task::JoinHandle<()>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Bind 0.0.0.0, like the real server: a listener on 127.0.0.1:port does NOT
    // collide with a later bind of 0.0.0.0:port on macOS once SO_REUSEADDR is
    // set (which tokio does), so the duplicate would sail past the bind and run
    // a full initialization instead of failing fast — the test would then be
    // testing nothing.
    let listener = tokio::net::TcpListener::bind("0.0.0.0:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let body = format!(
        r#"{{"version":"{version}","features":{{}},"build":{{"target":"test","profile":"test"}}}}"#
    );
    let handle = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let body = body.clone();
            tokio::spawn(async move {
                let mut request = [0u8; 1024];
                let _ = stream.read(&mut request).await;
                let response = format!(
                    "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            });
        }
    });
    (port, handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_live_orchestrator_is_recognised() {
        let (port, server) = spawn_fake_orchestrator("9.9.9").await;
        assert_eq!(
            probe(port).await,
            PortOwner::Orchestrator {
                version: "9.9.9".to_string()
            }
        );
        server.abort();
    }

    /// A listener that never answers must not be mistaken for an orchestrator,
    /// and must not make the probe hang: the existing bind-first regression
    /// test holds a bare `TcpListener` exactly like this.
    #[tokio::test]
    async fn a_silent_listener_is_foreign_and_bounded() {
        let holder = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = holder.local_addr().unwrap().port();
        let started = std::time::Instant::now();
        assert_eq!(probe(port).await, PortOwner::Foreign);
        assert!(
            started.elapsed() < PROBE_TIMEOUT * 3,
            "the probe must stay bounded, took {:?}",
            started.elapsed()
        );
        drop(holder);
    }

    #[tokio::test]
    async fn a_refused_connection_is_nobody() {
        // Port 1 needs privileges: nothing of ours listens there.
        assert_eq!(probe(1).await, PortOwner::Nobody);
    }

    #[test]
    fn classify_wants_a_version_and_a_build() {
        let po =
            b"HTTP/1.0 200 OK\r\n\r\n{\"version\":\"0.1.0\",\"build\":{\"profile\":\"release\"}}";
        assert_eq!(
            classify(po),
            PortOwner::Orchestrator {
                version: "0.1.0".to_string()
            }
        );

        // A successful answer that is not us.
        let other = b"HTTP/1.0 200 OK\r\n\r\n{\"name\":\"something-else\"}";
        assert_eq!(classify(other), PortOwner::Foreign);

        // A version string alone is not enough.
        let lookalike = b"HTTP/1.0 200 OK\r\n\r\n{\"version\":\"2.1\"}";
        assert_eq!(classify(lookalike), PortOwner::Foreign);

        // Both keys PRESENT but empty: the presence check this used to do would have taken
        // this for us, and the duplicate would have exited 0 against a stranger's port.
        let nulls = b"HTTP/1.0 200 OK\r\n\r\n{\"version\":null,\"build\":null}";
        assert_eq!(classify(nulls), PortOwner::Foreign);
        let empty_version =
            b"HTTP/1.0 200 OK\r\n\r\n{\"version\":\"\",\"build\":{\"profile\":\"release\"}}";
        assert_eq!(classify(empty_version), PortOwner::Foreign);
        // A `build` that carries none of our markers is somebody else's build object.
        let foreign_build =
            b"HTTP/1.0 200 OK\r\n\r\n{\"version\":\"1.0.0\",\"build\":{\"number\":42}}";
        assert_eq!(classify(foreign_build), PortOwner::Foreign);
        // One marker is enough: a peer built by an older or newer binary stays recognisable.
        let older_peer =
            b"HTTP/1.0 200 OK\r\n\r\n{\"version\":\"0.0.19\",\"build\":{\"target\":\"x86_64\"}}";
        assert_eq!(
            classify(older_peer),
            PortOwner::Orchestrator {
                version: "0.0.19".to_string()
            }
        );

        // Not a success status.
        let unauthorized = b"HTTP/1.0 401 Unauthorized\r\n\r\n{\"version\":\"0.1.0\",\"build\":{}}";
        assert_eq!(classify(unauthorized), PortOwner::Foreign);

        // Garbage.
        assert_eq!(classify(b"not http at all"), PortOwner::Foreign);
    }

    /// The rule the supervisor depends on: a duplicate exits 0, so
    /// `KeepAlive { SuccessfulExit = false }` stops relaunching it; anything
    /// else keeps the failure code and stays visible.
    #[test]
    fn a_duplicate_exits_successfully_and_everything_else_does_not() {
        let duplicate: anyhow::Error =
            anyhow::Error::new(PortOwnedByOrchestrator::new(8080, "0.1.0"));
        assert_eq!(successful_exit_code(&duplicate), Some(0));

        let foreign = anyhow::anyhow!("cannot bind 0.0.0.0:8080: Address already in use");
        assert_eq!(successful_exit_code(&foreign), None);

        let plain = anyhow::anyhow!("Neo4j is unreachable");
        assert_eq!(successful_exit_code(&plain), None);
    }

    #[test]
    fn the_message_says_what_to_do_and_still_says_cannot_bind() {
        let error = PortOwnedByOrchestrator::new(8080, "0.1.0");
        let text = error.to_string();
        assert!(text.contains("cannot bind"), "{text}");
        assert!(text.contains("8080"), "{text}");
        assert!(text.contains("0.1.0"), "{text}");
        assert!(text.contains("status 0"), "{text}");
        assert!(text.contains("kickstart"), "{text}");
    }
}
