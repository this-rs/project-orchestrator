//! Neo4j client and models for the knowledge graph

mod analytics;
mod chat;
pub mod client;
mod code;
mod commit;
mod constraint;
mod decision;
mod feature_graph;
mod impl_graph_store;
mod milestone;
pub mod models;
mod note;
mod plan;
mod project;
mod release;
mod step;
mod task;
pub mod traits;
mod user;
mod workspace;

pub use client::Neo4jClient;
pub use models::*;
pub use traits::GraphStore;

#[cfg(test)]
pub(crate) mod mock;
