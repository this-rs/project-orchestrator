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
//! ├── runner.rs    — main execution loop + task dispatch
//! ├── guard.rs     — AgentGuard: drift detection, hint injection
//! ├── verifier.rs  — post-task verification: build, steps, git
//! ├── enricher.rs  — post-task knowledge capture (V1: git-based)
//! ├── trigger.rs   — TriggerEngine: evaluation + firing
//! └── providers/   — trigger providers (schedule, webhook, event)
//! ```

pub mod enricher;
pub mod guard;
pub mod models;
pub mod providers;
pub mod runner;
pub mod state;
pub mod trigger;
pub mod vector;
pub mod verifier;

// Re-export key types for convenience
pub use enricher::{EnrichResult, TaskEnricher};
pub use guard::{AgentGuard, ChatManagerHintSender, GuardConfig, GuardVerdict, HintSender};
pub use models::{
    PlanRunStatus, RunSnapshot, RunnerConfig, RunnerEvent, TaskResult, TaskRunStatus,
    TaskStateMachine, Trigger, TriggerFiring, TriggerSource, TriggerType,
};
pub use providers::TriggerProvider;
pub use runner::{PlanRunner, RunStatus, RUNNER_CANCEL, RUNNER_STATE};
pub use state::RunnerState;
pub use trigger::TriggerEngine;
pub use vector::{compare_vectors, predict_run, ComparisonResult, ExecutionVector, RunPrediction};
pub use verifier::{TaskVerifier, VerifyResult};
