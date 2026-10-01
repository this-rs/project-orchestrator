//! Mapping packages to the infrastructure they imply.
//!
//! Neo4j, Meilisearch, NATS and every other datastore are absent from the code
//! graph: no file in the workspace *is* Neo4j. They only ever show up as a
//! declared dependency, so recognising `neo4rs` as "this project talks to Neo4j
//! over Bolt" is what makes infrastructure appear on the diagram at all.
//!
//! This table is the one piece of hand-maintained data in the derivation, and it
//! is deliberately generic: it describes ecosystems, not this workspace. Adding a
//! row benefits every project ever analysed.

use super::manifest::Ecosystem;
use crate::neo4j::models::ComponentType;

/// An infrastructure component implied by a client library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InfraService {
    /// Display name of the service, not of the client package.
    pub name: &'static str,
    pub component_type: ComponentType,
    /// Wire protocol, shown as the edge label.
    pub protocol: &'static str,
}

const fn svc(
    name: &'static str,
    component_type: ComponentType,
    protocol: &'static str,
) -> InfraService {
    InfraService {
        name,
        component_type,
        protocol,
    }
}

/// Cargo client crate → service.
fn cargo_catalogue(package: &str) -> Option<InfraService> {
    Some(match package {
        "neo4rs" => svc("Neo4j", ComponentType::Database, "Bolt"),
        "meilisearch-sdk" => svc("Meilisearch", ComponentType::External, "HTTP"),
        "async-nats" | "nats" => svc("NATS", ComponentType::MessageQueue, "NATS"),
        "redis" | "fred" | "deadpool-redis" => svc("Redis", ComponentType::Cache, "RESP"),
        "sqlx" | "diesel" | "tokio-postgres" | "postgres" | "deadpool-postgres" => {
            svc("PostgreSQL", ComponentType::Database, "SQL")
        }
        "mysql" | "mysql_async" => svc("MySQL", ComponentType::Database, "SQL"),
        "rusqlite" | "libsqlite3-sys" => svc("SQLite", ComponentType::Database, "embedded"),
        "mongodb" => svc("MongoDB", ComponentType::Database, "MongoDB wire"),
        "elasticsearch" => svc("Elasticsearch", ComponentType::External, "HTTP"),
        "rdkafka" => svc("Kafka", ComponentType::MessageQueue, "Kafka"),
        "lapin" | "amiquip" => svc("RabbitMQ", ComponentType::MessageQueue, "AMQP"),
        "aws-sdk-s3" | "rust-s3" => svc("S3", ComponentType::External, "HTTPS"),
        "qdrant-client" => svc("Qdrant", ComponentType::Database, "gRPC"),
        "clickhouse" => svc("ClickHouse", ComponentType::Database, "HTTP"),
        "opensearch" => svc("OpenSearch", ComponentType::External, "HTTP"),
        _ => return None,
    })
}

/// npm client package → service.
fn npm_catalogue(package: &str) -> Option<InfraService> {
    Some(match package {
        "neo4j-driver" => svc("Neo4j", ComponentType::Database, "Bolt"),
        "meilisearch" => svc("Meilisearch", ComponentType::External, "HTTP"),
        "nats" | "nats.ws" => svc("NATS", ComponentType::MessageQueue, "NATS"),
        "redis" | "ioredis" => svc("Redis", ComponentType::Cache, "RESP"),
        "pg" | "postgres" => svc("PostgreSQL", ComponentType::Database, "SQL"),
        "mysql" | "mysql2" => svc("MySQL", ComponentType::Database, "SQL"),
        "mongodb" | "mongoose" => svc("MongoDB", ComponentType::Database, "MongoDB wire"),
        "@elastic/elasticsearch" => svc("Elasticsearch", ComponentType::External, "HTTP"),
        "kafkajs" => svc("Kafka", ComponentType::MessageQueue, "Kafka"),
        "amqplib" => svc("RabbitMQ", ComponentType::MessageQueue, "AMQP"),
        "@aws-sdk/client-s3" => svc("S3", ComponentType::External, "HTTPS"),
        "better-sqlite3" => svc("SQLite", ComponentType::Database, "embedded"),
        _ => return None,
    })
}

/// Resolve a package to the infrastructure it implies, or `None` when the package
/// is ordinary library code rather than a client for something deployed.
pub fn lookup(ecosystem: Ecosystem, package: &str) -> Option<InfraService> {
    match ecosystem {
        Ecosystem::Cargo => cargo_catalogue(package),
        Ecosystem::Npm => npm_catalogue(package),
    }
}

/// Canonical display name for a container image, so a compose service and a
/// client package describe the *same* node.
///
/// Compose names its services in lower case (`neo4j`), the package catalogue uses
/// the product name (`Neo4j`). Without folding the two, a project that both runs
/// Neo4j in compose and imports `neo4rs` gets two database nodes, and the reader
/// has no way to tell they are one machine.
pub fn canonical_image_name(image: &str) -> Option<&'static str> {
    let name = image
        .rsplit('/')
        .next()
        .unwrap_or(image)
        .split(':')
        .next()
        .unwrap_or(image)
        .to_lowercase();

    Some(match name.as_str() {
        "neo4j" => "Neo4j",
        "meilisearch" => "Meilisearch",
        "nats" => "NATS",
        "redis" | "valkey" => "Redis",
        "postgres" | "postgresql" => "PostgreSQL",
        "mysql" | "mariadb" => "MySQL",
        "mongo" | "mongodb" => "MongoDB",
        "elasticsearch" => "Elasticsearch",
        "opensearch" => "OpenSearch",
        "rabbitmq" => "RabbitMQ",
        "kafka" | "redpanda" => "Kafka",
        "clickhouse-server" => "ClickHouse",
        "qdrant" => "Qdrant",
        "minio" => "S3",
        "memcached" => "Memcached",
        _ => return None,
    })
}

/// Web frameworks. A binary that serves HTTP is a service even when it also
/// parses command-line arguments — nearly every server does both, and taking
/// `clap` as proof of a CLI types the orchestrator backend as a command-line tool.
pub fn is_web_framework(package: &str) -> bool {
    matches!(
        package,
        "axum"
            | "actix-web"
            | "rocket"
            | "warp"
            | "tide"
            | "poem"
            | "salvo"
            | "hyper"
            | "tonic"
            | "express"
            | "fastify"
            | "koa"
            | "@nestjs/core"
    )
}

/// Frontend frameworks, used to type a project as `Frontend` rather than `Service`.
///
/// This matters more than it looks: `COMPONENT_TIER` in the frontend ranks nodes by
/// type, so a web app typed `Service` never reaches the top tier and the
/// edge → services → data reading collapses.
pub fn is_frontend_framework(package: &str) -> bool {
    matches!(
        package,
        "react"
            | "react-dom"
            | "vue"
            | "svelte"
            | "@angular/core"
            | "next"
            | "nuxt"
            | "solid-js"
            | "preact"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_client_crates_to_the_service_they_reach() {
        // The name recorded is the service, never the client package: the diagram
        // should read "Neo4j", not "neo4rs".
        let neo4j = lookup(Ecosystem::Cargo, "neo4rs").unwrap();
        assert_eq!(neo4j.name, "Neo4j");
        assert_eq!(neo4j.component_type, ComponentType::Database);
        assert_eq!(neo4j.protocol, "Bolt");
    }

    #[test]
    fn same_service_from_either_ecosystem_gets_one_identity() {
        // A Rust service and a TypeScript one both reaching Neo4j must land on the
        // same node, or the graph grows a duplicate per language.
        let from_cargo = lookup(Ecosystem::Cargo, "neo4rs").unwrap();
        let from_npm = lookup(Ecosystem::Npm, "neo4j-driver").unwrap();
        assert_eq!(from_cargo.name, from_npm.name);
        assert_eq!(from_cargo.component_type, from_npm.component_type);
    }

    #[test]
    fn ordinary_libraries_are_not_infrastructure() {
        assert!(lookup(Ecosystem::Cargo, "serde").is_none());
        assert!(lookup(Ecosystem::Cargo, "tokio").is_none());
        assert!(lookup(Ecosystem::Npm, "lodash").is_none());
        assert!(lookup(Ecosystem::Npm, "react").is_none());
    }

    #[test]
    fn ecosystems_do_not_leak_into_each_other() {
        // "pg" is a Postgres client on npm and nothing on crates.io.
        assert!(lookup(Ecosystem::Npm, "pg").is_some());
        assert!(lookup(Ecosystem::Cargo, "pg").is_none());
    }

    #[test]
    fn covers_this_workspace_stack() {
        // The three services the orchestrator actually runs against.
        for pkg in ["neo4rs", "meilisearch-sdk", "async-nats"] {
            assert!(
                lookup(Ecosystem::Cargo, pkg).is_some(),
                "{} must be recognised",
                pkg
            );
        }
    }

    #[test]
    fn compose_service_and_client_package_name_one_node() {
        // A project running Neo4j in compose and importing neo4rs must end up
        // with one database, not two.
        assert_eq!(canonical_image_name("neo4j:5.26-community"), Some("Neo4j"));
        assert_eq!(
            canonical_image_name("neo4j:5.26-community"),
            Some(lookup(Ecosystem::Cargo, "neo4rs").unwrap().name)
        );
        assert_eq!(
            canonical_image_name("getmeili/meilisearch:v1.34.2"),
            Some(lookup(Ecosystem::Cargo, "meilisearch-sdk").unwrap().name)
        );
    }

    #[test]
    fn unknown_image_has_no_canonical_name() {
        // An application image keeps its compose service name; inventing one
        // would merge unrelated services.
        assert_eq!(canonical_image_name("mycorp/billing-api:1.2"), None);
    }

    #[test]
    fn a_server_that_also_parses_arguments_is_not_a_cli() {
        // The orchestrator backend declares [[bin]] and depends on clap. It is
        // still a service, and typing it Cli puts it at the entry tier.
        assert!(is_web_framework("axum"));
        assert!(!is_web_framework("clap"));
    }

    #[test]
    fn detects_frontend_frameworks() {
        assert!(is_frontend_framework("react"));
        assert!(is_frontend_framework("svelte"));
        assert!(!is_frontend_framework("express"));
    }
}
