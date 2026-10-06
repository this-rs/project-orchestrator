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

/// `bolt://host:7687`, `http://host:7700`, `nats://host` → host and port.
pub(crate) fn parse_endpoint(raw: &str) -> Option<(String, u16)> {
    let url = url::Url::parse(raw.trim()).ok()?;
    let host = url.host_str()?.trim_matches(['[', ']']).to_owned();
    let port = url.port().or_else(|| default_port(url.scheme()))?;
    Some((host, port))
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

/// Tauri command: reachability of the services of `config.yaml`, with a reason when one fails.
#[tauri::command]
pub async fn probe_services() -> Vec<Probe> {
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
    probe_endpoints(endpoints).await
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
}
