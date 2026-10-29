//! Neo4j client and models for the knowledge graph
use crate::neo4j::traits::*;

pub(crate) mod agent_execution;
mod alert;
mod analytics;
pub(crate) mod batch;
mod chat;
pub mod client;
mod code;
mod commit;
mod constraint;
pub mod data_migrations;
mod day_plan;
mod decision;
pub mod document;
mod environment;
mod event_trigger;
mod execution_place;
mod feature_graph;
pub mod graph_sampler;
mod impl_graph_store;
mod ingress;
mod lifecycle_hook;
mod mcp_federation;
mod milestone;
pub mod models;
mod neighborhood;
mod note;
mod persona;
pub mod plan;
mod plan_run;
mod profile;
mod project;
mod protocol;
pub(crate) mod reasoning;
mod registry;
mod release;
pub mod routing;
pub mod segment;
mod sharing;
mod skill;
mod step;
mod synapse_shadow;
mod task;
mod topology;
pub mod traits;
pub mod trajectory;
pub mod trajectory_migration;
pub mod trajectory_store;
mod trigger;
mod user;
mod workspace;

pub use agent_execution::{AgentExecutionNode, AgentExecutionStatus};
pub use client::Neo4jClient;
pub use document::{Document, DocumentChunk, DocumentChunkHit};
pub use models::*;
pub use segment::Neo4jSegmentStore;
pub use traits::{
    AnalyticsStore, CodeStore, CommitStore, DecisionStore, GraphStore, NoteStore, PlanStore,
    ProjectStore, ReleaseStore, StoreCapabilities, WorkspaceStore,
};

// The in-memory store. Unit tests get it through `cfg(test)`; the integration
// test targets (tests/store_conformance.rs runs the same checks against it and
// against Neo4j) get it through the `mock-store` feature, which the dev-dependency
// of the crate on itself enables for every test build and no binary.
#[cfg(any(test, feature = "mock-store"))]
pub mod mock;
