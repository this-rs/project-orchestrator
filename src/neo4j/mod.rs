//! Neo4j client and models for the knowledge graph

pub(crate) mod agent_execution;
mod analytics;
pub(crate) mod batch;
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
pub mod plan;
mod plan_run;
mod profile;
mod project;
mod protocol;
pub(crate) mod reasoning;
mod registry;
mod release;
mod skill;
mod step;
mod task;
mod topology;
pub mod traits;
mod trigger;
mod user;
mod workspace;

pub use agent_execution::{AgentExecutionNode, AgentExecutionStatus};
pub use client::Neo4jClient;
pub use models::*;
pub use traits::GraphStore;

#[cfg(test)]
pub(crate) mod mock;
