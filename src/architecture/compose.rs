//! Reading `docker-compose.yml`.
//!
//! Where a compose file exists it is the best description of the deployed system
//! available anywhere in the repository, and it beats inferring infrastructure
//! from client libraries on every axis:
//!
//! - it names the service and pins its image, so "Neo4j 5.26" is exact rather than
//!   guessed from the presence of a `neo4rs` crate;
//! - `depends_on` states the edges outright, with no inference at all;
//! - the environment block carries connection URIs, so the wire protocol is read
//!   (`bolt://neo4j:7687` → Bolt) instead of assumed from the client package.
//!
//! The manifest catalogue remains the fallback for projects that ship no compose
//! file — most frontends, and libraries.

use serde::Deserialize;
use std::collections::BTreeMap;

use crate::neo4j::models::ComponentType;

#[derive(Debug, Deserialize)]
pub struct ComposeFile {
    #[serde(default)]
    pub services: BTreeMap<String, ComposeService>,
}

#[derive(Debug, Deserialize)]
pub struct ComposeService {
    #[serde(default)]
    pub image: Option<String>,
    /// Present when the service is built from this repository — that makes it the
    /// project's own component rather than a third-party one.
    #[serde(default)]
    pub build: Option<serde_yaml::Value>,
    #[serde(default)]
    pub environment: Option<EnvBlock>,
    #[serde(default)]
    pub depends_on: Option<DependsOn>,
}

/// `environment:` accepts both a `KEY=value` list and a mapping.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EnvBlock {
    List(Vec<String>),
    Map(BTreeMap<String, Option<String>>),
}

impl EnvBlock {
    fn values(&self) -> Vec<String> {
        match self {
            EnvBlock::List(items) => items
                .iter()
                .filter_map(|kv| kv.split_once('=').map(|(_, v)| v.to_string()))
                .collect(),
            EnvBlock::Map(map) => map.values().flatten().cloned().collect(),
        }
    }
}

/// `depends_on:` accepts both a short list and a long mapping with conditions.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum DependsOn {
    List(Vec<String>),
    Map(BTreeMap<String, serde_yaml::Value>),
}

impl DependsOn {
    fn names(&self) -> Vec<String> {
        match self {
            DependsOn::List(v) => v.clone(),
            DependsOn::Map(m) => m.keys().cloned().collect(),
        }
    }
}

/// One service as derived from a compose file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposedService {
    /// Service key in the compose file.
    pub name: String,
    /// Product name when the image is a recognised one, so this service and a
    /// client package importing the same product resolve to a single node.
    pub canonical_name: Option<&'static str>,
    pub component_type: ComponentType,
    /// Image reference, when the service is not built locally.
    pub image: Option<String>,
    /// True when built from this repository — i.e. this project itself.
    pub is_local_build: bool,
    /// Services it depends on, with the protocol read from the environment when
    /// a connection URI points at them.
    pub depends_on: Vec<ComposedEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposedEdge {
    pub target: String,
    pub protocol: Option<String>,
}

/// Classify a service from its image reference.
///
/// Matching is on the image name rather than an exact tag so that a version bump
/// does not silently drop a component off the diagram.
fn type_from_image(image: &str) -> ComponentType {
    let name = image
        .rsplit('/')
        .next()
        .unwrap_or(image)
        .split(':')
        .next()
        .unwrap_or(image)
        .to_lowercase();

    match name.as_str() {
        "neo4j" | "postgres" | "postgresql" | "mysql" | "mariadb" | "mongo" | "mongodb"
        | "clickhouse-server" | "qdrant" => ComponentType::Database,
        "redis" | "memcached" | "valkey" => ComponentType::Cache,
        "nats" | "rabbitmq" | "kafka" | "redpanda" => ComponentType::MessageQueue,
        "nginx" | "traefik" | "haproxy" | "caddy" | "envoy" => ComponentType::Gateway,
        "meilisearch" | "elasticsearch" | "opensearch" | "minio" => ComponentType::External,
        _ => ComponentType::Service,
    }
}

/// Pull `scheme://host` pairs out of environment values, so an edge can state the
/// protocol it actually travels over.
fn protocols_by_host(env: &Option<EnvBlock>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let Some(env) = env else { return out };

    for value in env.values() {
        let Some((scheme, rest)) = value.split_once("://") else {
            continue;
        };
        if scheme.is_empty() || scheme.contains(' ') {
            continue;
        }
        let host = rest
            .split(['/', ':'])
            .next()
            .unwrap_or_default()
            .to_string();
        if !host.is_empty() {
            out.entry(host).or_insert_with(|| scheme.to_uppercase());
        }
    }
    out
}

/// Parse a compose file into services and their edges.
pub fn parse_compose(raw: &str) -> anyhow::Result<Vec<ComposedService>> {
    let file: ComposeFile = serde_yaml::from_str(raw)?;

    let mut out: Vec<ComposedService> = file
        .services
        .into_iter()
        .map(|(name, svc)| {
            let is_local_build = svc.build.is_some();
            let component_type = if is_local_build {
                // Built here, so it is this project — the caller types it from the
                // project itself, which knows whether it is a frontend or a service.
                ComponentType::Service
            } else {
                svc.image
                    .as_deref()
                    .map(type_from_image)
                    .unwrap_or(ComponentType::Other)
            };

            let protocols = protocols_by_host(&svc.environment);
            let depends_on = svc
                .depends_on
                .as_ref()
                .map(DependsOn::names)
                .unwrap_or_default()
                .into_iter()
                .map(|target| ComposedEdge {
                    protocol: protocols.get(&target).cloned(),
                    target,
                })
                .collect();

            ComposedService {
                canonical_name: svc
                    .image
                    .as_deref()
                    .and_then(crate::architecture::catalogue::canonical_image_name),
                name,
                component_type,
                image: svc.image,
                is_local_build,
                depends_on,
            }
        })
        .collect();

    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shaped after the orchestrator's own compose file.
    const COMPOSE: &str = r#"
services:
  neo4j:
    image: neo4j:5.26.20-community
    ports: ["7687:7687"]
  meilisearch:
    image: getmeili/meilisearch:v1.34.2
  nats:
    image: nats:2.11-alpine
  orchestrator:
    build:
      context: .
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - MEILISEARCH_URL=http://meilisearch:7700
      - NATS_URL=nats://nats:4222
      - RUST_LOG=info
    depends_on:
      neo4j:
        condition: service_healthy
      meilisearch:
        condition: service_healthy
      nats:
        condition: service_healthy
"#;

    fn get<'a>(svcs: &'a [ComposedService], name: &str) -> &'a ComposedService {
        svcs.iter().find(|s| s.name == name).unwrap()
    }

    #[test]
    fn types_services_from_their_image() {
        let svcs = parse_compose(COMPOSE).unwrap();
        assert_eq!(get(&svcs, "neo4j").component_type, ComponentType::Database);
        assert_eq!(
            get(&svcs, "nats").component_type,
            ComponentType::MessageQueue
        );
        // Meilisearch is a third-party search engine, not one of our services.
        assert_eq!(
            get(&svcs, "meilisearch").component_type,
            ComponentType::External
        );
    }

    #[test]
    fn registry_prefix_and_tag_do_not_defeat_matching() {
        // "getmeili/meilisearch:v1.34.2" must classify the same as "meilisearch".
        assert_eq!(
            type_from_image("getmeili/meilisearch:v1.34.2"),
            ComponentType::External
        );
        assert_eq!(
            type_from_image("docker.io/library/redis:7-alpine"),
            ComponentType::Cache
        );
        assert_eq!(type_from_image("neo4j"), ComponentType::Database);
    }

    #[test]
    fn locally_built_service_is_flagged_as_this_project() {
        let svcs = parse_compose(COMPOSE).unwrap();
        assert!(get(&svcs, "orchestrator").is_local_build);
        assert!(!get(&svcs, "neo4j").is_local_build);
    }

    #[test]
    fn reads_protocol_from_the_connection_uri() {
        // The whole point of preferring compose over the crate catalogue: the
        // protocol is stated, not guessed from which client library is present.
        let svcs = parse_compose(COMPOSE).unwrap();
        let orch = get(&svcs, "orchestrator");
        let proto = |t: &str| {
            orch.depends_on
                .iter()
                .find(|e| e.target == t)
                .unwrap()
                .protocol
                .clone()
        };
        assert_eq!(proto("neo4j"), Some("BOLT".into()));
        assert_eq!(proto("meilisearch"), Some("HTTP".into()));
        assert_eq!(proto("nats"), Some("NATS".into()));
    }

    #[test]
    fn accepts_the_short_depends_on_form() {
        let raw = r#"
services:
  api:
    image: my/api
    depends_on: ["db"]
  db:
    image: postgres:16
"#;
        let svcs = parse_compose(raw).unwrap();
        let api = get(&svcs, "api");
        assert_eq!(api.depends_on.len(), 1);
        assert_eq!(api.depends_on[0].target, "db");
        // No URI in the environment, so no protocol is invented.
        assert_eq!(api.depends_on[0].protocol, None);
    }

    #[test]
    fn accepts_the_mapping_environment_form() {
        let raw = r#"
services:
  api:
    image: my/api
    environment:
      DATABASE_URL: postgres://db:5432/app
    depends_on: ["db"]
  db:
    image: postgres:16
"#;
        let svcs = parse_compose(raw).unwrap();
        assert_eq!(
            get(&svcs, "api").depends_on[0].protocol,
            Some("POSTGRES".into())
        );
    }

    #[test]
    fn unknown_image_is_a_service_not_a_guess() {
        assert_eq!(
            type_from_image("mycorp/billing-api:1.2"),
            ComponentType::Service
        );
    }

    #[test]
    fn recognised_images_carry_their_product_name() {
        // Without this the compose service `neo4j` and the crate `neo4rs` end
        // up as two separate databases on the diagram.
        let svcs = parse_compose(COMPOSE).unwrap();
        assert_eq!(get(&svcs, "neo4j").canonical_name, Some("Neo4j"));
        assert_eq!(
            get(&svcs, "meilisearch").canonical_name,
            Some("Meilisearch")
        );
        // A locally built service is this project, so it has no product name.
        assert_eq!(get(&svcs, "orchestrator").canonical_name, None);
    }

    #[test]
    fn service_without_depends_on_yields_no_edges() {
        let svcs = parse_compose(COMPOSE).unwrap();
        assert!(get(&svcs, "neo4j").depends_on.is_empty());
    }
}
