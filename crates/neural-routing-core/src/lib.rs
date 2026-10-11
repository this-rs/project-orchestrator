//! Neural Routing Core — types, traits, and storage for trajectory-based route learning.
//!
//! This crate provides the foundational data structures shared by all neural-routing crates:
//! - `Trajectory` / `TrajectoryNode` / `DecisionVector` — the core data model
//! - `TrajectoryStore` trait — async CRUD + vector search abstraction
//! - `RewardStrategy` trait — pluggable credit assignment strategies
//! - `Neo4jTrajectoryStore` — concrete storage implementation

pub mod augmentation;
pub mod error;
pub mod mcts;
pub mod migration;
pub mod models;
pub mod proxy_model;
pub mod reward;
pub mod store;
pub mod traits;
pub mod validation;
pub mod vector_builder;

pub use augmentation::{run_augmentation, AugmentationConfig, AugmentationReport};
pub use error::NeuralRoutingError;
pub use mcts::{MctsConfig, MctsEngine};
pub use migration::{run_migration, MigrationConfig, MigrationReport};
pub use models::*;
pub use proxy_model::{GdsHeuristicProxy, ProxyModel};
pub use reward::*;
pub use store::Neo4jTrajectoryStore;
pub use traits::*;
pub use validation::*;
pub use vector_builder::{
    sentinel_vector, DecisionContext, DecisionVectorBuilder, NodeFeatures, ProjectionMatrix,
    SessionMeta, GRAPH_DIM, HISTORY_DIM, QUERY_DIM, SESSION_DIM, SOURCE_EMBED_DIM, TOOL_DIM,
    TOTAL_DIM,
};
