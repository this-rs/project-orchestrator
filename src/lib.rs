//! Project Orchestrator
//!
//! An AI agent orchestrator with:
//! - Neo4j knowledge graph for code structure and relationships
//! - Meilisearch for fast semantic search
//! - Tree-sitter for precise code parsing
//! - Plan management for coordinated multi-agent development

pub mod api;
pub mod meilisearch;
pub mod neo4j;
pub mod orchestrator;
pub mod parser;
pub mod plan;

use anyhow::Result;
use std::sync::Arc;

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub neo4j: Arc<neo4j::client::Neo4jClient>,
    pub meili: Arc<meilisearch::client::MeiliClient>,
    pub parser: Arc<parser::CodeParser>,
    pub config: Arc<Config>,
}

/// Application configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub neo4j_uri: String,
    pub neo4j_user: String,
    pub neo4j_password: String,
    pub meilisearch_url: String,
    pub meilisearch_key: String,
    pub workspace_path: String,
    pub server_port: u16,
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            neo4j_uri: std::env::var("NEO4J_URI")
                .unwrap_or_else(|_| "bolt://localhost:7687".into()),
            neo4j_user: std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into()),
            neo4j_password: std::env::var("NEO4J_PASSWORD")
                .unwrap_or_else(|_| "orchestrator123".into()),
            meilisearch_url: std::env::var("MEILISEARCH_URL")
                .unwrap_or_else(|_| "http://localhost:7700".into()),
            meilisearch_key: std::env::var("MEILISEARCH_KEY")
                .unwrap_or_else(|_| "orchestrator-meili-key-change-me".into()),
            workspace_path: std::env::var("WORKSPACE_PATH").unwrap_or_else(|_| ".".into()),
            server_port: std::env::var("SERVER_PORT")
                .unwrap_or_else(|_| "8080".into())
                .parse()
                .unwrap_or(8080),
        })
    }
}

impl AppState {
    /// Create new application state with all services initialized
    pub async fn new(config: Config) -> Result<Self> {
        let neo4j = Arc::new(
            neo4j::client::Neo4jClient::new(
                &config.neo4j_uri,
                &config.neo4j_user,
                &config.neo4j_password,
            )
            .await?,
        );

        let meili = Arc::new(
            meilisearch::client::MeiliClient::new(&config.meilisearch_url, &config.meilisearch_key)
                .await?,
        );

        let parser = Arc::new(parser::CodeParser::new()?);

        Ok(Self {
            neo4j,
            meili,
            parser,
            config: Arc::new(config),
        })
    }
}
