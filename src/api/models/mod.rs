//! API DTO models.
//!
//! Lightweight, frontend-shaped structures used to serialize REST responses.
//! Keep these decoupled from internal `neo4j::models` / `runner::state` types
//! so the API surface can evolve independently.

pub mod activity;

pub use activity::{
    ActivitySnapshot, ChatSessionSummary, PlanRunSummary, ProtocolRunSummary,
    SnapshotPlanRunStatus, SnapshotProtocolRunStatus,
};
