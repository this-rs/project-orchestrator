//! Reading runtime configuration for edges that exist only at run time.
//!
//! Some of the most important edges in a system are invisible to both manifests
//! and compose files. A browser application depends on its API over HTTP: it
//! declares no package for it, and it is not a container in the compose file.
//!
//! Measured on this workspace: the frontend has 23 dependencies, none of them
//! infrastructure and none of them pointing at another project. Derived from
//! manifests alone, the single most important edge in the whole diagram —
//! frontend to backend — simply does not exist.
//!
//! Dev-server proxy configuration is where that edge is written down. It is
//! hand-maintained by whoever wires the app up, and it names a concrete target.

use std::path::Path;

/// An upstream a project talks to at run time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyUpstream {
    /// Scheme of the target, e.g. "http".
    pub scheme: String,
    /// Host of the target, e.g. "localhost".
    pub host: String,
    /// Port, when the target states one — the strongest signal for matching a
    /// target to the project that serves it.
    pub port: Option<u16>,
    /// Config file this was read from, relative to the project root.
    pub file: String,
    /// 1-based line, for provenance.
    pub line: u32,
}

/// Extract `target: '<url>'` entries from a dev-server config.
///
/// Scanned rather than parsed: these files are TypeScript modules with imports,
/// functions and conditionals, so there is no structured form to read without
/// executing them. A `target:` line is unambiguous enough, and the failure mode
/// of a miss is a missing edge rather than a wrong one.
pub fn parse_proxy_targets(raw: &str, file: &str) -> Vec<ProxyUpstream> {
    let mut out: Vec<ProxyUpstream> = Vec::new();

    for (idx, line) in raw.lines().enumerate() {
        // `target:` is matched anywhere in the line, not just at its start: the
        // inline form `'/api': { target: '...' }` is the common one. What keeps
        // this honest is the `://` requirement below — `build: { target: 'esnext' }`
        // names no upstream and is dropped.
        let mut cursor = line;
        while let Some(pos) = cursor.find("target:") {
            let after = &cursor[pos + "target:".len()..];
            cursor = after;

            let value = after
                .trim_start()
                .trim_start_matches(['\'', '"', '`'])
                .split(['\'', '"', '`', ',', '}'])
                .next()
                .unwrap_or_default()
                .trim();

            let Some((scheme, remainder)) = value.split_once("://") else {
                continue;
            };
            // Skip interpolated targets: `${env.API_URL}` names no concrete upstream.
            if remainder.contains("${") || remainder.is_empty() || scheme.contains(' ') {
                continue;
            }

            let authority = remainder.split('/').next().unwrap_or_default();
            let (host, port) = match authority.rsplit_once(':') {
                Some((h, p)) => (h.to_string(), p.parse::<u16>().ok()),
                None => (authority.to_string(), None),
            };
            if host.is_empty() {
                continue;
            }

            let upstream = ProxyUpstream {
                scheme: scheme.to_string(),
                host,
                port,
                file: file.to_string(),
                line: idx as u32 + 1,
            };

            // A config routes several prefixes (/api, /auth, /ws) to one upstream.
            // That is one edge, not three.
            if !out.iter().any(|u| {
                u.host == upstream.host && u.port == upstream.port && u.scheme == upstream.scheme
            }) {
                out.push(upstream);
            }
        }
    }

    out
}

/// Known dev-server config files, checked at the project root.
const CONFIG_FILES: [&str; 4] = [
    "vite.config.ts",
    "vite.config.js",
    "next.config.js",
    "webpack.config.js",
];

/// Read whatever runtime upstreams the project declares.
pub fn read_project_upstreams(root: &Path) -> Vec<ProxyUpstream> {
    let mut out = Vec::new();
    for name in CONFIG_FILES {
        let path = root.join(name);
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        out.extend(parse_proxy_targets(&raw, name));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shaped after the orchestrator frontend's own vite.config.ts, which routes
    /// three prefixes to one backend.
    const VITE: &str = r#"
export default defineConfig({
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/auth': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/ws': {
        target: 'http://localhost:8080',
        ws: true,
      },
    },
  },
})
"#;

    #[test]
    fn finds_the_edge_no_manifest_can_see() {
        let ups = parse_proxy_targets(VITE, "vite.config.ts");
        assert_eq!(ups.len(), 1);
        assert_eq!(ups[0].host, "localhost");
        assert_eq!(ups[0].port, Some(8080));
        assert_eq!(ups[0].scheme, "http");
    }

    #[test]
    fn several_prefixes_to_one_upstream_is_one_edge() {
        // /api, /auth and /ws all reach the same backend. Three edges would say
        // the frontend depends on three different things.
        assert_eq!(parse_proxy_targets(VITE, "vite.config.ts").len(), 1);
    }

    #[test]
    fn records_the_line_for_provenance() {
        let ups = parse_proxy_targets(VITE, "vite.config.ts");
        assert_eq!(ups[0].file, "vite.config.ts");
        assert_eq!(ups[0].line, 7);
    }

    #[test]
    fn distinct_upstreams_stay_distinct() {
        let raw = r#"
proxy: {
  '/api': { target: 'http://localhost:8080' },
  '/search': { target: 'http://localhost:7700' },
}
"#;
        let ups = parse_proxy_targets(raw, "vite.config.ts");
        assert_eq!(ups.len(), 2);
        assert_eq!(ups[1].port, Some(7700));
    }

    #[test]
    fn interpolated_target_names_no_upstream() {
        // `${process.env.API_URL}` resolves at run time. Guessing here would
        // invent an edge; better to draw none.
        let raw = "proxy: { '/api': { target: `http://${process.env.API_HOST}:8080` } }";
        assert!(parse_proxy_targets(raw, "vite.config.ts").is_empty());
    }

    #[test]
    fn target_without_a_port_is_still_an_upstream() {
        let raw = "proxy: { '/api': { target: 'https://api.example.com' } }";
        let ups = parse_proxy_targets(raw, "vite.config.ts");
        assert_eq!(ups.len(), 1);
        assert_eq!(ups[0].host, "api.example.com");
        assert_eq!(ups[0].port, None);
        assert_eq!(ups[0].scheme, "https");
    }

    #[test]
    fn ignores_unrelated_target_keys() {
        // `target` also names a build target in these files.
        let raw = "build: { target: 'esnext' }";
        assert!(parse_proxy_targets(raw, "vite.config.ts").is_empty());
    }

    #[test]
    fn accepts_double_quotes() {
        let raw = r#"proxy: { "/api": { target: "http://localhost:9000" } }"#;
        let ups = parse_proxy_targets(raw, "vite.config.ts");
        assert_eq!(ups.len(), 1);
        assert_eq!(ups[0].port, Some(9000));
    }
}
