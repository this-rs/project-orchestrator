//! PlanRunner — Autonomous Plan Execution Engine
//!
//! Orchestrates the full lifecycle of plan execution:
//! trigger → spawn agent → guard → verify → enrich → next task
//!
//! # Architecture
//! ```text
//! PlanRunner (this module)
//! ├── models.rs    — RunnerConfig, TaskResult, RunnerEvent, TaskStateMachine
//! ├── state.rs     — RunnerState persisted in Neo4j for crash recovery
//! ├── runner.rs    — main execution loop (future)
//! ├── guard.rs     — AgentGuard: drift detection, hint injection (future)
//! ├── verifier.rs  — post-task verification: build, steps, git (future)
//! └── enricher.rs  — post-task knowledge capture (future)
//! ```

pub mod models;
pub mod state;

// Re-export key types for convenience
pub use models::{
    PlanRunStatus, RunSnapshot, RunnerConfig, RunnerEvent, TaskResult, TaskRunStatus,
    TaskStateMachine, TriggerSource,
};
pub use state::RunnerState;
