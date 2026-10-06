//! Can this machine reach the services it is configured to use, and if not, why?
//!
//! With `infra_mode: external` Neo4j, Meilisearch and NATS live wherever the user put them,
//! often on another machine of the local network. The backend only reports "unreachable". The
//! reasons differ and so does the cure: nothing listening, a wrong address, a firewall, or, on
//! macOS 15, the system refusing the app access to the local network until the user allows it
//! under Privacy & Security > Local Network (the connection then fails with "No route to host").

use std::io;
use std::net::IpAddr;
use std::time::Duration;

use serde::Serialize;

/// How long one connection attempt gets. Long enough for a slow VPN, short enough for a splash.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// The result of trying to reach one service.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Probe {
    pub service: String,
    pub host: String,
    pub port: u16,
    pub ok: bool,
    /// `refused`, `unreachable`, `timeout`, `dns` or `other`; absent when it worked.
    pub kind: Option<String>,
    /// What to do about it, in words for the user; absent when it worked.
    pub hint: Option<String>,
}

fn default_port(scheme: &str) -> Option<u16> {
    match scheme {
        "bolt" | "bolt+s" | "bolt+ssc" | "neo4j" | "neo4j+s" | "neo4j+ssc" => Some(7687),
        "http" => Some(80),
        "https" => Some(443),
        "nats" | "tls" => Some(4222),
        _ => None,
    }
}

/// Splits what a user types into a host and a port: `bolt://host:7687`, `http://host:7700/path`,
/// `nas.local:7687` (no scheme), `192.168.1.5`, `[fd00::5]:7687`, `bolt://user:pw@host:7687`.
///
/// `url::Url::parse` is not used for this: it reads `nas.local:7687` as the *scheme* `nas.local`
/// with no host, which made the setup wizard test `localhost` whenever a remote host was typed by
/// name without a scheme.
pub(crate) fn split_host_port(raw: &str, default_port: Option<u16>) -> Option<(String, u16)> {
    let raw = raw.trim();
    let (scheme, rest) = match raw.split_once("://") {
        Some((scheme, rest)) => (Some(scheme.to_ascii_lowercase()), rest),
        None => (None, raw),
    };
    // Authority only: no path, query or fragment, and never the credentials.
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let authority = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    if authority.is_empty() {
        return None;
    }
    let (host, port) = if let Some(inner) = authority.strip_prefix('[') {
        // [v6] or [v6]:port
        let (host, after) = inner.split_once(']')?;
        let port = match after.strip_prefix(':') {
            Some(p) if !p.is_empty() => Some(p.parse::<u16>().ok()?),
            Some(_) => return None,
            None if after.is_empty() => None,
            None => return None,
        };
        (host.to_owned(), port)
    } else if authority.matches(':').count() == 1 {
        let (host, port) = authority.split_once(':')?;
        (host.to_owned(), Some(port.parse::<u16>().ok()?))
    } else if authority.contains(':') {
        return None; // a bare IPv6 address needs its brackets to carry a port
    } else {
        (authority.to_owned(), None)
    };
    if host.is_empty() {
        return None;
    }
    let port = port
        .or_else(|| scheme.as_deref().and_then(default_port_of))
        .or(default_port)?;
    Some((host, port))
}

fn default_port_of(scheme: &str) -> Option<u16> {
    default_port(scheme)
}

/// `bolt://host:7687`, `http://host:7700`, `nats://host` → host and port.
pub(crate) fn parse_endpoint(raw: &str) -> Option<(String, u16)> {
    split_host_port(raw, None)
}

/// A machine of the local network, as opposed to this one or the internet: a private or
/// link-local address, or a `.local` name.
pub(crate) fn is_lan_host(host: &str) -> bool {
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) => ip.is_private() || ip.is_link_local(),
        Ok(IpAddr::V6(ip)) => {
            let first = ip.segments()[0];
            (first & 0xfe00) == 0xfc00 || (first & 0xffc0) == 0xfe80 // ULA, link-local
        }
        Err(_) => host.to_ascii_lowercase().ends_with(".local"),
    }
}

/// Why a connection failed, and what to do about it.
pub(crate) fn classify(error: &io::Error, host: &str, port: u16) -> (&'static str, String) {
    let message = error.to_string().to_ascii_lowercase();
    if message.contains("lookup")
        || message.contains("resolve")
        || message.contains("nodename nor servname")
    {
        return (
            "dns",
            format!("Cannot resolve “{host}”. Check the address in the configuration."),
        );
    }
    match error.kind() {
        io::ErrorKind::ConnectionRefused => (
            "refused",
            format!("Nothing is listening on {host}:{port}. Is the service started?"),
        ),
        io::ErrorKind::HostUnreachable | io::ErrorKind::NetworkUnreachable => {
            if cfg!(target_os = "macos") && is_lan_host(host) {
                (
                    "unreachable",
                    "macOS may be blocking this app from the local network. Open System Settings → \
                     Privacy & Security → Local Network, enable Project Orchestrator, then relaunch it."
                        .to_owned(),
                )
            } else {
                (
                    "unreachable",
                    format!("{host} cannot be reached from this machine. Check the address and the network or VPN."),
                )
            }
        }
        io::ErrorKind::TimedOut => (
            "timeout",
            format!("No answer from {host}:{port} (a firewall, a VPN, or a wrong address)."),
        ),
        _ => (
            "other",
            format!("Could not connect to {host}:{port}: {error}"),
        ),
    }
}

async fn probe(service: &str, host: String, port: u16) -> Probe {
    let attempt = tokio::time::timeout(
        PROBE_TIMEOUT,
        tokio::net::TcpStream::connect((host.as_str(), port)),
    )
    .await;
    let failure = match attempt {
        Ok(Ok(_)) => None,
        Ok(Err(error)) => Some(error),
        Err(_) => Some(io::Error::from(io::ErrorKind::TimedOut)),
    };
    let (kind, hint) = match &failure {
        Some(error) => {
            let (kind, hint) = classify(error, &host, port);
            (Some(kind.to_owned()), Some(hint))
        }
        None => (None, None),
    };
    Probe {
        service: service.to_owned(),
        host,
        port,
        ok: failure.is_none(),
        kind,
        hint,
    }
}

/// What the setup wizard learns by testing one service.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ConnectionTest {
    pub service: String,
    pub host: String,
    pub port: u16,
    /// The service answered the way a real one does.
    pub ok: bool,
    /// What was checked beyond opening the port: `bolt-handshake`, `nats-banner`, `http-health`,
    /// or `tcp` when only the connection could be checked (TLS endpoints).
    pub verified_by: String,
    /// `refused`, `unreachable`, `timeout`, `dns`, `not_bolt`, `not_nats`, `http_status`, `other`.
    pub kind: Option<String>,
    pub hint: Option<String>,
}

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const REPLY_TIMEOUT: Duration = Duration::from_secs(3);

async fn open(host: &str, port: u16) -> Result<tokio::net::TcpStream, (String, String)> {
    let attempt = tokio::time::timeout(
        CONNECT_TIMEOUT,
        tokio::net::TcpStream::connect((host, port)),
    )
    .await;
    let error = match attempt {
        Ok(Ok(stream)) => return Ok(stream),
        Ok(Err(error)) => error,
        Err(_) => io::Error::from(io::ErrorKind::TimedOut),
    };
    let (kind, hint) = classify(&error, host, port);
    Err((kind.to_owned(), hint))
}

/// Bolt: the magic preamble and four protocol proposals; a real server answers with 4 bytes
/// (the version it picked, or zeros when it has none in common: still a Bolt server).
async fn bolt_handshake(stream: &mut tokio::net::TcpStream) -> bool {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut hello = vec![0x60, 0x60, 0xB0, 0x17];
    for version in [[0, 0, 4, 5], [0, 0, 0, 5], [0, 0, 4, 4], [0, 0, 0, 3]] {
        hello.extend_from_slice(&version);
    }
    if stream.write_all(&hello).await.is_err() {
        return false;
    }
    let mut reply = [0u8; 4];
    matches!(
        tokio::time::timeout(REPLY_TIMEOUT, stream.read_exact(&mut reply)).await,
        Ok(Ok(_))
    )
}

/// NATS speaks first: `INFO {...}\r\n`.
async fn nats_banner(stream: &mut tokio::net::TcpStream) -> bool {
    use tokio::io::AsyncReadExt;
    let mut buf = [0u8; 5];
    matches!(
        tokio::time::timeout(REPLY_TIMEOUT, stream.read_exact(&mut buf)).await,
        Ok(Ok(_))
    ) && &buf == b"INFO "
}

/// Meilisearch: `GET /health` answers 200 `{"status":"available"}`.
async fn meili_health(url: &str, host: &str, port: u16) -> Result<(), (String, String)> {
    let scheme = if url.trim().to_ascii_lowercase().starts_with("https://") {
        "https"
    } else {
        "http"
    };
    let target = format!("{scheme}://{host}:{port}/health");
    let client = reqwest::Client::builder()
        .timeout(CONNECT_TIMEOUT)
        .build()
        .map_err(|e| ("other".to_owned(), format!("HTTP client error: {e}")))?;
    match client.get(&target).send().await {
        Ok(resp) if resp.status().is_success() => Ok(()),
        Ok(resp) => Err((
            "http_status".to_owned(),
            format!(
                "{host}:{port} answered HTTP {}: is that Meilisearch, and is the URL right?",
                resp.status().as_u16()
            ),
        )),
        Err(error) => {
            let source = std::error::Error::source(&error)
                .and_then(|s| s.downcast_ref::<io::Error>())
                .map(|e| io::Error::new(e.kind(), e.to_string()));
            let io_error = source.unwrap_or_else(|| {
                if error.is_timeout() {
                    io::Error::from(io::ErrorKind::TimedOut)
                } else {
                    io::Error::other(error.to_string())
                }
            });
            let (kind, hint) = classify(&io_error, host, port);
            Err((kind.to_owned(), hint))
        }
    }
}

/// Tests one service the way the wizard needs: reachable AND really that service, with the
/// reason when it is not. `service` is `neo4j`, `meilisearch` or `nats`.
pub(crate) async fn test_service(service: &str, raw_url: &str) -> Result<ConnectionTest, String> {
    let default_port = match service {
        "neo4j" => 7687,
        "meilisearch" => 7700,
        "nats" => 4222,
        other => return Err(format!("Unknown service: {other}")),
    };
    let (host, port) = split_host_port(raw_url, Some(default_port)).ok_or_else(|| {
        format!(
            "“{}” is not an address (expected host, host:port or scheme://host:port)",
            raw_url.trim()
        )
    })?;
    let tls = raw_url.split("://").next().is_some_and(|scheme| {
        scheme.to_ascii_lowercase().ends_with("+s")
            || scheme.to_ascii_lowercase().ends_with("+ssc")
            || scheme.eq_ignore_ascii_case("tls")
            || scheme.eq_ignore_ascii_case("https")
    });
    let result = |ok: bool, verified_by: &str, failure: Option<(String, String)>| ConnectionTest {
        service: service.to_owned(),
        host: host.clone(),
        port,
        ok,
        verified_by: verified_by.to_owned(),
        kind: failure.as_ref().map(|f| f.0.clone()),
        hint: failure.map(|f| f.1),
    };
    if service == "meilisearch" {
        return Ok(match meili_health(raw_url, &host, port).await {
            Ok(()) => result(true, "http-health", None),
            Err(failure) => result(false, "http-health", Some(failure)),
        });
    }
    let mut stream = match open(&host, port).await {
        Ok(stream) => stream,
        Err(failure) => return Ok(result(false, "tcp", Some(failure))),
    };
    if tls {
        // The handshake is encrypted: only the connection can be checked here.
        return Ok(result(true, "tcp", None));
    }
    Ok(match service {
        "neo4j" if bolt_handshake(&mut stream).await => result(true, "bolt-handshake", None),
        "neo4j" => result(
            false,
            "bolt-handshake",
            Some(("not_bolt".to_owned(), format!("Something is listening on {host}:{port} but it does not speak Bolt: is that Neo4j's Bolt port (7687), not its HTTP port (7474)?"))),
        ),
        _ if nats_banner(&mut stream).await => result(true, "nats-banner", None),
        _ => result(
            false,
            "nats-banner",
            Some(("not_nats".to_owned(), format!("Something is listening on {host}:{port} but it is not a NATS server (no INFO banner)."))),
        ),
    })
}

/// Tries to open a connection to each configured service, in parallel.
pub(crate) async fn probe_endpoints(endpoints: Vec<(&'static str, String)>) -> Vec<Probe> {
    let mut tasks = Vec::new();
    for (service, raw) in endpoints {
        let Some((host, port)) = parse_endpoint(&raw) else {
            continue;
        };
        tasks.push(tokio::spawn(
            async move { probe(service, host, port).await },
        ));
    }
    let mut results = Vec::new();
    for task in tasks {
        if let Ok(result) = task.await {
            results.push(result);
        }
    }
    results
}

/// Tauri command: tests one service typed in the setup wizard. Reachable AND really that service,
/// with the reason when it is not (see [`ConnectionTest`]).
#[tauri::command]
pub async fn test_connection_detailed(
    service: String,
    url: String,
) -> Result<ConnectionTest, String> {
    test_service(&service, &url).await
}

/// The services of `config.yaml` as (name, endpoint): Neo4j, Meilisearch, and NATS when it is on.
/// Empty when there is no readable configuration yet.
fn configured_endpoints() -> Vec<(&'static str, String)> {
    let path = crate::setup::config_path();
    let Ok(contents) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(yaml) = serde_yaml::from_str::<project_orchestrator::YamlConfig>(&contents) else {
        return Vec::new();
    };
    let mut endpoints = vec![
        ("neo4j", yaml.neo4j.uri.clone()),
        ("meilisearch", yaml.meilisearch.url.clone()),
    ];
    if let Some(nats) = yaml.nats.url.clone() {
        endpoints.push(("nats", nats));
    }
    endpoints
}

/// Tauri command: reachability of the services of `config.yaml`, with a reason when one fails.
#[tauri::command]
pub async fn probe_services() -> Vec<Probe> {
    probe_endpoints(configured_endpoints()).await
}

/// Every service answers ITS OWN protocol (Bolt handshake, NATS banner, Meilisearch `/health`),
/// and there is at least one. An open port is not enough: something else may hold it.
pub(crate) async fn services_all_up(endpoints: Vec<(&'static str, String)>) -> bool {
    if endpoints.is_empty() {
        return false;
    }
    let tasks: Vec<_> = endpoints
        .into_iter()
        .map(|(service, raw)| {
            tokio::spawn(async move {
                test_service(service, &raw)
                    .await
                    .map(|test| test.ok)
                    .unwrap_or(false)
            })
        })
        .collect();
    for task in tasks {
        if !task.await.unwrap_or(false) {
            return false;
        }
    }
    true
}

/// [`services_all_up`] for the services of `config.yaml`.
///
/// What the app asks when Docker's API does not answer: Docker Desktop can have its control
/// socket stuck while the containers it started keep running and serving, and the containers
/// are what the app needs, not the socket.
pub(crate) async fn configured_services_up() -> bool {
    services_all_up(configured_endpoints()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_of_every_scheme_the_config_uses_give_a_host_and_a_port() {
        assert_eq!(
            parse_endpoint("bolt://localhost:7687"),
            Some(("localhost".into(), 7687))
        );
        assert_eq!(
            parse_endpoint("neo4j+s://db.example.com"),
            Some(("db.example.com".into(), 7687))
        );
        assert_eq!(
            parse_endpoint("http://192.168.1.20:7700"),
            Some(("192.168.1.20".into(), 7700))
        );
        assert_eq!(
            parse_endpoint("nats://nas.local"),
            Some(("nas.local".into(), 4222))
        );
        assert_eq!(
            parse_endpoint("http://[fd00::5]:7700"),
            Some(("fd00::5".into(), 7700))
        );
        assert_eq!(parse_endpoint("not a url"), None);
        assert_eq!(parse_endpoint("gopher://x"), None, "no port to guess");
    }

    #[test]
    fn the_local_network_is_told_from_this_machine_and_from_the_internet() {
        for lan in [
            "192.168.1.5",
            "10.0.0.2",
            "172.16.4.4",
            "169.254.1.1",
            "fd12::1",
            "fe80::1",
            "nas.local",
            "NAS.LOCAL",
        ] {
            assert!(is_lan_host(lan), "{lan}");
        }
        for other in [
            "127.0.0.1",
            "localhost",
            "::1",
            "8.8.8.8",
            "example.com",
            "172.32.0.1",
        ] {
            assert!(!is_lan_host(other), "{other}");
        }
    }

    #[test]
    fn each_failure_says_what_to_do() {
        let refused = io::Error::from(io::ErrorKind::ConnectionRefused);
        let (kind, hint) = classify(&refused, "localhost", 7687);
        assert_eq!(kind, "refused");
        assert!(
            hint.contains("localhost:7687") && hint.contains("started"),
            "{hint}"
        );

        let timeout = io::Error::from(io::ErrorKind::TimedOut);
        assert_eq!(classify(&timeout, "10.0.0.9", 7700).0, "timeout");

        let dns = io::Error::other(
            "failed to lookup address information: nodename nor servname provided",
        );
        let (kind, hint) = classify(&dns, "nope.invalid", 7687);
        assert_eq!(kind, "dns");
        assert!(hint.contains("nope.invalid"), "{hint}");
    }

    #[test]
    fn no_route_to_a_lan_host_points_at_the_macos_permission_only_on_macos() {
        let unreachable = io::Error::from(io::ErrorKind::HostUnreachable);
        let (kind, hint) = classify(&unreachable, "192.168.1.20", 7687);
        assert_eq!(kind, "unreachable");
        if cfg!(target_os = "macos") {
            assert!(hint.contains("Local Network"), "{hint}");
        } else {
            assert!(!hint.contains("Local Network"), "{hint}");
        }
        // A host outside the LAN is never blamed on that permission.
        let (_, hint) = classify(&unreachable, "93.184.216.34", 7687);
        assert!(!hint.contains("Local Network"), "{hint}");
    }

    #[tokio::test]
    async fn a_listening_port_is_ok_and_a_closed_one_is_refused_with_a_reason() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let open = listener.local_addr().unwrap().port();
        let closed = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        }; // dropped: nothing listens there now
        let results = probe_endpoints(vec![
            ("neo4j", format!("bolt://127.0.0.1:{open}")),
            ("meilisearch", format!("http://127.0.0.1:{closed}")),
            ("nats", "garbage".to_owned()),
        ])
        .await;
        assert_eq!(results.len(), 2, "an unparsable address is skipped");
        assert!(results[0].ok && results[0].hint.is_none());
        assert!(!results[1].ok);
        assert_eq!(results[1].kind.as_deref(), Some("refused"));
        assert!(results[1]
            .hint
            .as_deref()
            .unwrap()
            .contains(&closed.to_string()));
    }

    #[test]
    fn a_host_typed_without_a_scheme_keeps_its_host() {
        // url::Url::parse reads "nas.local:7687" as scheme "nas.local" and no host: the wizard then
        // tested localhost. Every shape a user types must keep the host they typed.
        for (raw, expected) in [
            ("nas.local:7687", ("nas.local", 7687)),
            ("db.internal:7700", ("db.internal", 7700)),
            ("192.168.1.20:7687", ("192.168.1.20", 7687)),
            ("192.168.1.20", ("192.168.1.20", 7687)),
            ("bolt://nas.local:7687", ("nas.local", 7687)),
            ("neo4j+s://db.example.com", ("db.example.com", 7687)),
            ("bolt://neo4j:secret@nas.local:7687", ("nas.local", 7687)),
            ("http://[fd00::5]:7700/health", ("fd00::5", 7700)),
            ("[fd00::5]:7687", ("fd00::5", 7687)),
            ("  localhost  ", ("localhost", 7687)),
        ] {
            assert_eq!(
                split_host_port(raw, Some(7687)),
                Some((expected.0.to_owned(), expected.1)),
                "{raw}"
            );
        }
        for bad in [
            "",
            "   ",
            "bolt://",
            ":7687",
            "host:notaport",
            "host:",
            "fd00::5",
            "[fd00::5",
        ] {
            assert_eq!(
                split_host_port(bad, Some(7687)),
                None,
                "{bad:?} is not an address"
            );
        }
    }

    async fn serve_once(reply: &'static [u8], speak_first: bool) -> u16 {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    if speak_first {
                        let _ = stream.write_all(reply).await;
                    } else {
                        let mut buf = [0u8; 64];
                        let _ = stream.read(&mut buf).await;
                        let _ = stream.write_all(reply).await;
                    }
                    let _ = stream.shutdown().await;
                });
            }
        });
        port
    }

    #[tokio::test]
    async fn services_are_up_only_when_every_one_answers_its_own_protocol() {
        let bolt = serve_once(&[0, 0, 4, 5], false).await;
        let nats = serve_once(b"INFO {\"server_id\":\"x\"}\r\n", true).await;
        let endpoints = |bolt_port: u16, nats_port: u16| {
            vec![
                ("neo4j", format!("bolt://127.0.0.1:{bolt_port}")),
                ("nats", format!("nats://127.0.0.1:{nats_port}")),
            ]
        };
        assert!(services_all_up(endpoints(bolt, nats)).await);

        // Something else on NATS's port (it speaks, but it is not NATS): not up.
        let impostor = serve_once(b"hello\r\n", true).await;
        assert!(!services_all_up(endpoints(bolt, impostor)).await);

        // A port nobody listens on: not up.
        let closed = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        assert!(!services_all_up(endpoints(bolt, closed)).await);

        // No service configured says nothing: never "up".
        assert!(!services_all_up(Vec::new()).await);
    }

    #[tokio::test]
    async fn neo4j_is_checked_by_its_handshake_not_by_an_open_port() {
        let bolt = serve_once(&[0, 0, 4, 5], false).await;
        let ok = test_service("neo4j", &format!("bolt://127.0.0.1:{bolt}"))
            .await
            .unwrap();
        assert!(
            ok.ok && ok.verified_by == "bolt-handshake" && ok.hint.is_none(),
            "{ok:?}"
        );

        // Something else on the port (an HTTP server, say) answers a handshake with text, or nothing.
        let http = serve_once(b"", false).await;
        let not_neo4j = test_service("neo4j", &format!("127.0.0.1:{http}"))
            .await
            .unwrap();
        assert!(!not_neo4j.ok, "{not_neo4j:?}");
        assert_eq!(not_neo4j.kind.as_deref(), Some("not_bolt"));
        assert!(not_neo4j.hint.as_deref().unwrap().contains("Bolt"));
    }

    #[tokio::test]
    async fn nats_is_checked_by_its_banner() {
        let nats = serve_once(b"INFO {\"server_id\":\"x\"}\r\n", true).await;
        let ok = test_service("nats", &format!("nats://127.0.0.1:{nats}"))
            .await
            .unwrap();
        assert!(ok.ok && ok.verified_by == "nats-banner", "{ok:?}");

        let other = serve_once(b"hello\r\n", true).await;
        let no = test_service("nats", &format!("127.0.0.1:{other}"))
            .await
            .unwrap();
        assert!(!no.ok);
        assert_eq!(no.kind.as_deref(), Some("not_nats"));
    }

    #[tokio::test]
    async fn meilisearch_is_checked_by_its_health_route_and_a_missing_scheme_is_http() {
        let up = serve_once(b"HTTP/1.1 200 OK\r\nContent-Length: 22\r\nContent-Type: application/json\r\n\r\n{\"status\":\"available\"}", false).await;
        let ok = test_service("meilisearch", &format!("127.0.0.1:{up}"))
            .await
            .unwrap();
        assert!(ok.ok && ok.verified_by == "http-health", "{ok:?}");

        let wrong = serve_once(
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n",
            false,
        )
        .await;
        let no = test_service("meilisearch", &format!("http://127.0.0.1:{wrong}"))
            .await
            .unwrap();
        assert!(!no.ok);
        assert_eq!(no.kind.as_deref(), Some("http_status"));
        assert!(no.hint.as_deref().unwrap().contains("404"));
    }

    #[tokio::test]
    async fn an_unreachable_service_says_why_and_a_bad_address_is_an_error() {
        let closed = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let refused = test_service("neo4j", &format!("bolt://127.0.0.1:{closed}"))
            .await
            .unwrap();
        assert!(!refused.ok);
        assert_eq!(refused.kind.as_deref(), Some("refused"));
        assert!(refused
            .hint
            .as_deref()
            .unwrap()
            .contains(&closed.to_string()));

        assert!(test_service("neo4j", "bolt://").await.is_err());
        assert!(test_service("redis", "localhost:6379").await.is_err());
    }

    #[tokio::test]
    async fn a_tls_endpoint_is_checked_by_connection_only_and_says_so() {
        let port = serve_once(b"", true).await;
        let t = test_service("neo4j", &format!("bolt+s://127.0.0.1:{port}"))
            .await
            .unwrap();
        assert!(t.ok);
        assert_eq!(
            t.verified_by, "tcp",
            "the handshake is encrypted: only the connection is checked"
        );
    }
}
