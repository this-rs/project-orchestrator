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
pub mod feedback;
pub mod feedback_analyzer;
pub mod git;
pub mod guard;
pub mod lifecycle;
pub mod models;
pub mod persona;
pub mod prompt;
pub mod providers;
#[allow(clippy::module_inception)]
pub mod runner;
pub mod state;
pub mod trigger;
pub mod vector;
pub mod verifier;

// Re-export key types for convenience
pub use enricher::{EnrichResult, TaskEnricher};
pub use feedback::{
    contains_dissatisfaction, ManualCommitInfo, OverrideType, PostRunMessage,
    RunnerFeedbackCollector, UserOverride,
};
pub use feedback_analyzer::{
    AppliedProtocol, DetectedPattern, FeedbackAnalyzer, FeedbackPatternType, FeedbackReport,
    PatternSuggestion, ProtocolTrust, TrustStatus,
};
pub use git::{WorktreeCollector, WorktreeInfo, WorktreeResolution};
pub use guard::{AgentGuard, ChatManagerHintSender, GuardConfig, GuardVerdict, HintSender};
pub use lifecycle::{route_lifecycle_protocol, LifecycleRouteResult};
pub use models::{
    ActiveAgent, ActiveAgentSnapshot, CwdValidation, PlanRunStatus, RunSnapshot, RunnerConfig,
    RunnerEvent, TaskExecutionReport, TaskResult, TaskRunStatus, TaskStateMachine, Trigger,
    TriggerFiring, TriggerSource, TriggerType,
};
pub use persona::{
    activate_skills_for_task, complexity_directive, load_persona_stack, profile_task,
    record_persona_feedback, record_skill_feedback, Complexity, PersonaEntry, PersonaStack,
    PersonaTrigger, SkillActivationResult, TaskProfile,
};
pub use prompt::{
    build_runner_constraints, PromptBuilder, PromptSection, RunnerPromptContext, StructuredPrompt,
};
pub use providers::TriggerProvider;
pub use runner::{PlanRunner, RunStatus, RUNNER_CANCEL, RUNNER_STATE};
pub use state::RunnerState;
pub use trigger::TriggerEngine;
pub use vector::{
    compare_vectors, predict_run, predict_run_per_agent, AgentExecutionVector,
    AgentVectorCollector, ComparisonResult, ExecutionVector, RunPrediction,
};
pub use verifier::{TaskVerifier, VerifyResult};
