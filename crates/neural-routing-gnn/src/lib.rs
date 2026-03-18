//! GNN models for neural route learning.
//!
//! Implements R-GCN (Relational Graph Convolutional Network) and GraphSAGE
//! for encoding the knowledge graph into node embeddings used by the policy network.
//!
//! Built on candle (HuggingFace) — pure Rust, no Python dependency.
//! Always compiled (build full), activation controlled at runtime via settings.

pub mod encoder;
pub mod features;
pub mod graph_sage;
pub mod message_passing;
pub mod rgcn;
pub mod sampler;

/// Re-export key types
pub use encoder::{GNNArchitecture, GraphEncoder, GraphEncoderConfig};
pub use features::{
    NodeFeatureBuilder, NodeType, NormStats, ProjectionMatrix, RawNodeData, TOTAL_FEATURE_DIM,
};
pub use graph_sage::GraphSAGE;
pub use message_passing::{scatter_add, scatter_mean, MessagePassing};
pub use rgcn::RGCN;
pub use sampler::{
    export_to_pyg, GraphSampler, GraphSamplerConfig, PyGData, RelationType, SamplerError,
    SubGraph, SubGraphEdge, SubGraphNode,
};
