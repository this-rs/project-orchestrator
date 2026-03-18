//! Policy Network for neural route learning.
//!
//! Implements:
//! - Decision Transformer (~3M params, GPT-2 style) — primary policy
//! - CQL (Conservative Q-Learning) — offline RL fallback
//! - EWC (Elastic Weight Consolidation) — continual learning without catastrophic forgetting
//! - TrajectoryDataset — conversion from Neo4j trajectories to Decision Transformer tensors
//! - TrajectoryDataLoader — weighted sampling, temporal splits, batch iteration
//!
//! Built on candle (HuggingFace) — pure Rust, no Python dependency.
//! Always compiled (build full), activation controlled at runtime via settings.

pub mod ab_testing;
pub mod action_decoder;
pub mod benchmark;
pub mod codebook;
pub mod cql;
pub mod dataloader;
pub mod dataset;
pub mod evaluation;
pub mod ewc;
pub mod registry;
pub mod training;
pub mod transformer;

pub use ab_testing::{ABConfig, ABReport, ABRouter, Bucket, BucketMetrics, SequentialTester};
pub use action_decoder::{ActionDecoder, DecodeResult, DecodedAction};
pub use codebook::{ActionCodebook, CodebookBuilder, CodebookEntry};
pub use cql::CQLPolicy;
pub use dataloader::{TrajectoryDataLoader, TrajectoryDataLoaderConfig};
pub use dataset::{
    pad_and_batch, trajectories_to_tensors, trajectory_to_tensors, PolicyNormStats,
    TrajectoryBatch, TrajectoryTensors, ACTION_DIM, CONTEXT_DIM, QUERY_DIM, STATE_DIM,
};
pub use evaluation::{
    action_accuracy, importance_weighted_eval, pearson_correlation, shannon_entropy,
    spearman_correlation, EvaluationReport, McDropoutEstimator, OodDetector, OodResult,
};
pub use registry::{ModelRegistry, ModelStage, ModelVersion};
pub use training::TrainingConfig;
/// Re-export key types
pub use transformer::DecisionTransformer;
