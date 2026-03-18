//! Neural Routing Runtime — orchestration layer.
//!
//! Brings together all neural-routing crates into a unified runtime:
//! - `CpuGuard` — circuit breaker for CPU protection
//! - `DualTrackRouter` — Policy Net + NN Router fallback with confidence calibration
//! - `InferenceEngine` — real-time Policy Net inference pipeline (<15ms)
//! - `DriftDetector` — reward distribution monitoring with Page-Hinkley & KS-test
//! - `ExplorationScheduler` — adaptive ε-greedy + Thompson Sampling
//! - `ContinualTrainer` — background fine-tuning with EWC and replay buffer
//! - Settings-based activation (runtime control, no feature flags)

pub mod collector;
pub mod confidence;
pub mod config;
pub mod continual;
pub mod cpu_guard;
pub mod drift;
pub mod dual_track;
pub mod exploration;
pub mod inference_engine;
pub mod reward;

pub use collector::{CollectorEvent, DecisionRecord, TrajectoryCollector};
pub use confidence::{PlattCalibrator, RolloutConfig};
pub use config::NeuralRoutingConfig;
pub use continual::{
    BufferEntry, ContinualConfig, ContinualTrainer, ReplayBuffer, TrainingRunResult,
};
pub use cpu_guard::CpuGuard;
pub use drift::{DriftAction, DriftConfig, DriftDetector, DriftEvent, DriftType};
pub use dual_track::DualTrackRouter;
pub use exploration::{ExplorationConfig, ExplorationDecision, ExplorationScheduler};
pub use inference_engine::{
    InferenceEngine, InferenceEngineConfig, InferenceError, InferenceResult, InferenceSource,
    PlannedAction,
};
pub use reward::{SessionRewardComputer, SessionSignals};

// Re-export core types so consumers only need neural-routing-runtime
pub use neural_routing_core::{
    create_reward_strategy, error as routing_error, DecisionContext, DecisionVectorBuilder,
    NNRoute, Neo4jTrajectoryStore, NodeFeatures, RewardConfig, RewardStrategy, Router, SessionMeta,
    ToolUsage, TouchedEntity, Trajectory, TrajectoryFilter, TrajectoryNode, TrajectoryStats,
    TrajectoryStore, SOURCE_EMBED_DIM, TOTAL_DIM,
};

// Re-export NN types needed by handlers
pub use neural_routing_nn::metrics::NNMetricsSnapshot;
