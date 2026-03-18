//! GNN Inference — CandleGnnEncoder with moka cache for production use.
//!
//! Provides:
//! - `GnnEncoder` trait — abstraction for graph encoding
//! - `CandleGnnEncoder` — loads safetensors checkpoint → forward pass → Vec<f32> 256d
//! - `CachedGnnEncoder` — wraps any GnnEncoder with moka concurrent LRU cache (1000 entries, 6h TTL)
//! - `GnnGraphStateBuilder` — replaces the mean-pooling GDS graph_state block (64d)
//!   with GNN embeddings projected to 64d

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

#[cfg(test)]
use crate::encoder::GNNArchitecture;
use crate::encoder::{GraphEncoder, GraphEncoderConfig};
use crate::features::{NodeFeatureBuilder, NormStats, RawNodeData, TOTAL_FEATURE_DIM};
use crate::sampler::{export_to_pyg, SubGraph};

// ---------------------------------------------------------------------------
// GnnEncoder trait
// ---------------------------------------------------------------------------

/// Error type for GNN encoding.
#[derive(Debug, thiserror::Error)]
pub enum GnnError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("No checkpoint loaded")]
    NoCheckpoint,

    #[error("Invalid graph: {0}")]
    InvalidGraph(String),
}

/// Trait for graph neural network encoders.
///
/// Produces fixed-dimension embeddings for nodes in a subgraph.
pub trait GnnEncoder: Send + Sync {
    /// Encode all nodes in a subgraph, returning [num_nodes × output_dim] embeddings.
    fn encode_subgraph(&self, subgraph: &SubGraph) -> Result<Vec<Vec<f32>>, GnnError>;

    /// Encode a single node by its ID in the subgraph.
    /// Returns the embedding for the target node.
    fn encode_node(&self, subgraph: &SubGraph, target_node_id: &str) -> Result<Vec<f32>, GnnError>;

    /// Get the output embedding dimension.
    fn output_dim(&self) -> usize;

    /// Whether this encoder supports inductive encoding of new nodes.
    fn is_inductive(&self) -> bool;
}

// ---------------------------------------------------------------------------
// CandleGnnEncoder — loads safetensors, runs forward pass
// ---------------------------------------------------------------------------

/// Configuration for the CandleGnnEncoder.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CandleGnnEncoderConfig {
    /// GNN encoder configuration.
    pub encoder_config: GraphEncoderConfig,
    /// Path to the safetensors checkpoint.
    pub checkpoint_path: Option<String>,
    /// Normalization stats (from training DataLoader).
    pub norm_stats: Option<NormStats>,
}

// Default derived: all fields implement Default (GraphEncoderConfig, Option<_>).

/// CandleGnnEncoder — production GNN encoder using candle.
///
/// Loads a safetensors checkpoint and runs forward passes on CPU.
/// Deterministic: same input → same output (no dropout at inference).
pub struct CandleGnnEncoder {
    encoder: GraphEncoder,
    varmap: VarMap,
    feature_builder: NodeFeatureBuilder,
    norm_stats: Option<NormStats>,
    config: CandleGnnEncoderConfig,
}

impl CandleGnnEncoder {
    /// Create a new encoder (no checkpoint loaded yet).
    pub fn new(config: CandleGnnEncoderConfig) -> Result<Self, GnnError> {
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let encoder = GraphEncoder::new(config.encoder_config.clone(), vb)?;

        // Load checkpoint if path is provided
        if let Some(ref path) = config.checkpoint_path {
            varmap.load(Path::new(path)).map_err(GnnError::Candle)?;
            info!("Loaded GNN checkpoint from {}", path);
        }

        Ok(Self {
            encoder,
            varmap,
            feature_builder: NodeFeatureBuilder::new(),
            norm_stats: config.norm_stats.clone(),
            config,
        })
    }

    /// Load a checkpoint from a safetensors file.
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<(), GnnError> {
        self.varmap.load(path).map_err(GnnError::Candle)?;
        info!("Loaded GNN checkpoint from {}", path.display());
        Ok(())
    }

    /// Set normalization stats (from training).
    pub fn set_norm_stats(&mut self, stats: NormStats) {
        self.norm_stats = Some(stats);
    }

    /// Build feature tensor from a subgraph.
    fn build_features(&self, subgraph: &SubGraph) -> CandleResult<Tensor> {
        let device = Device::Cpu;

        let raw_nodes: Vec<RawNodeData> = subgraph
            .nodes
            .iter()
            .map(|n| RawNodeData::from_properties(&n.labels, &n.properties))
            .collect();

        let features = if let Some(ref stats) = self.norm_stats {
            self.feature_builder
                .build_batch_with_stats(&raw_nodes, stats)
        } else {
            // Without norm stats, build without normalization
            raw_nodes
                .iter()
                .map(|n| self.feature_builder.build(n))
                .collect()
        };

        let num_nodes = features.len();
        let flat: Vec<f32> = features.into_iter().flatten().collect();
        Tensor::from_vec(flat, (num_nodes, TOTAL_FEATURE_DIM), &device)
    }
}

impl GnnEncoder for CandleGnnEncoder {
    fn encode_subgraph(&self, subgraph: &SubGraph) -> Result<Vec<Vec<f32>>, GnnError> {
        if subgraph.nodes.is_empty() {
            return Ok(vec![]);
        }

        let pyg = export_to_pyg(subgraph);
        let device = Device::Cpu;

        // Build node features
        let x = self.build_features(subgraph)?;

        // Build edge tensors
        let num_edges = pyg.num_edges;

        // If no edges, create empty edge tensors (GNN still applies self-transform)
        if num_edges == 0 {
            let edge_index = Tensor::zeros((2, 0), DType::I64, &device)?;
            let edge_type = Tensor::zeros(0, DType::U8, &device)?;
            let embeddings =
                self.encoder
                    .forward(&x, &edge_index, Some(&edge_type), pyg.num_nodes)?;
            return Ok(embeddings.to_vec2::<f32>()?);
        }

        let mut sources: Vec<i64> = Vec::with_capacity(num_edges);
        let mut targets: Vec<i64> = Vec::with_capacity(num_edges);
        for i in 0..num_edges {
            sources.push(pyg.edge_index[0][i] as i64);
            targets.push(pyg.edge_index[1][i] as i64);
        }
        let mut edge_flat = sources;
        edge_flat.extend(targets);
        let edge_index = Tensor::from_vec(edge_flat, (2, num_edges), &device)?;
        let edge_type = Tensor::from_vec(pyg.edge_type.clone(), num_edges, &device)?;

        // Forward pass
        let embeddings = self
            .encoder
            .forward(&x, &edge_index, Some(&edge_type), pyg.num_nodes)?;

        Ok(embeddings.to_vec2::<f32>()?)
    }

    fn encode_node(&self, subgraph: &SubGraph, target_node_id: &str) -> Result<Vec<f32>, GnnError> {
        let all_embeddings = self.encode_subgraph(subgraph)?;

        // Find the target node index
        let idx = subgraph
            .nodes
            .iter()
            .position(|n| n.app_id == target_node_id || n.element_id == target_node_id)
            .ok_or_else(|| {
                GnnError::InvalidGraph(format!("Node '{}' not found in subgraph", target_node_id))
            })?;

        Ok(all_embeddings[idx].clone())
    }

    fn output_dim(&self) -> usize {
        self.config.encoder_config.output_dim
    }

    fn is_inductive(&self) -> bool {
        self.encoder.is_inductive()
    }
}

// ---------------------------------------------------------------------------
// CachedGnnEncoder — moka concurrent LRU cache
// ---------------------------------------------------------------------------

/// Cached GNN encoder wrapping any GnnEncoder with a concurrent LRU cache.
///
/// Cache key: subgraph center_id + k_hops + node count hash.
/// Cache value: Vec<Vec<f32>> embeddings for all nodes.
///
/// - 1000 entries max
/// - TTL: 6 hours (GDS features may change)
/// - Thread-safe via moka's concurrent cache
pub struct CachedGnnEncoder {
    inner: Arc<dyn GnnEncoder>,
    cache: moka::sync::Cache<String, Vec<Vec<f32>>>,
}

impl CachedGnnEncoder {
    /// Create a new cached encoder with default settings (1000 entries, 6h TTL).
    pub fn new(inner: Arc<dyn GnnEncoder>) -> Self {
        let cache = moka::sync::Cache::builder()
            .max_capacity(1000)
            .time_to_live(Duration::from_secs(6 * 3600))
            .build();

        Self { inner, cache }
    }

    /// Create with custom capacity and TTL.
    pub fn with_config(inner: Arc<dyn GnnEncoder>, max_capacity: u64, ttl_secs: u64) -> Self {
        let cache = moka::sync::Cache::builder()
            .max_capacity(max_capacity)
            .time_to_live(Duration::from_secs(ttl_secs))
            .build();

        Self { inner, cache }
    }

    /// Invalidate a cache entry by node ID.
    pub fn invalidate(&self, node_id: &str) {
        // Invalidate all entries whose key contains this node_id
        // Since keys are center_id based, we invalidate matching center
        self.cache.invalidate(node_id);
        debug!("Invalidated cache for node: {}", node_id);
    }

    /// Invalidate all cache entries.
    pub fn invalidate_all(&self) {
        self.cache.invalidate_all();
        debug!("Invalidated all GNN cache entries");
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            entry_count: self.cache.entry_count(),
            weighted_size: self.cache.weighted_size(),
        }
    }

    /// Build cache key from subgraph properties.
    fn cache_key(subgraph: &SubGraph) -> String {
        format!(
            "{}:{}:{}:{}",
            subgraph.center_id,
            subgraph.k_hops,
            subgraph.nodes.len(),
            subgraph.edges.len()
        )
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub entry_count: u64,
    pub weighted_size: u64,
}

impl GnnEncoder for CachedGnnEncoder {
    fn encode_subgraph(&self, subgraph: &SubGraph) -> Result<Vec<Vec<f32>>, GnnError> {
        let key = Self::cache_key(subgraph);

        // Check cache
        if let Some(cached) = self.cache.get(&key) {
            debug!("GNN cache hit for {}", key);
            return Ok(cached);
        }

        // Cache miss: compute and store
        let embeddings = self.inner.encode_subgraph(subgraph)?;
        self.cache.insert(key, embeddings.clone());
        Ok(embeddings)
    }

    fn encode_node(&self, subgraph: &SubGraph, target_node_id: &str) -> Result<Vec<f32>, GnnError> {
        let all = self.encode_subgraph(subgraph)?;

        let idx = subgraph
            .nodes
            .iter()
            .position(|n| n.app_id == target_node_id || n.element_id == target_node_id)
            .ok_or_else(|| {
                GnnError::InvalidGraph(format!("Node '{}' not found in subgraph", target_node_id))
            })?;

        Ok(all[idx].clone())
    }

    fn output_dim(&self) -> usize {
        self.inner.output_dim()
    }

    fn is_inductive(&self) -> bool {
        self.inner.is_inductive()
    }
}

// ---------------------------------------------------------------------------
// GnnGraphStateBuilder — replaces mean-pooling GDS with GNN embeddings
// ---------------------------------------------------------------------------

/// Builds the graph_state block (64d) for DecisionVector using GNN embeddings.
///
/// Instead of mean-pooling 6d GDS features, this uses the full GNN embedding (256d)
/// projected to 64d via a linear projection (Achlioptas sparse random).
pub struct GnnGraphStateBuilder {
    /// 256d → 64d projection for graph state block.
    projection: crate::features::ProjectionMatrix,
}

impl GnnGraphStateBuilder {
    /// Create with deterministic projection.
    pub fn new() -> Self {
        // Use a different seed (73) to decorrelate from query/history projections
        Self {
            projection: crate::features::ProjectionMatrix::random(256, 64, 73),
        }
    }

    /// Build a 64d graph_state block from GNN embeddings of touched nodes.
    ///
    /// * `node_embeddings` — GNN embeddings (256d each) of recently touched nodes
    ///
    /// Returns a 64d vector suitable for the graph_state block in DecisionVector.
    pub fn build(&self, node_embeddings: &[Vec<f32>]) -> Vec<f32> {
        if node_embeddings.is_empty() {
            return vec![0.0f32; 64];
        }

        // Mean-pool GNN embeddings (256d)
        let dim = node_embeddings[0].len();
        let n = node_embeddings.len() as f32;
        let mut mean = vec![0.0f32; dim];
        for emb in node_embeddings {
            for (i, &v) in emb.iter().enumerate() {
                if i < dim {
                    mean[i] += v / n;
                }
            }
        }

        // Project 256d → 64d
        if dim == 256 {
            self.projection.project(&mean)
        } else {
            // Fallback: take first 64d or pad with zeros
            let mut result = vec![0.0f32; 64];
            for (i, &v) in mean.iter().enumerate().take(64) {
                result[i] = v;
            }
            result
        }
    }

    /// Build graph_state from an encoder and subgraph (convenience method).
    ///
    /// Encodes touched nodes, then projects to 64d.
    pub fn build_from_encoder(
        &self,
        encoder: &dyn GnnEncoder,
        subgraph: &SubGraph,
        touched_node_ids: &[String],
    ) -> Result<Vec<f32>, GnnError> {
        if touched_node_ids.is_empty() {
            return Ok(vec![0.0f32; 64]);
        }

        // Encode the full subgraph
        let all_embeddings = encoder.encode_subgraph(subgraph)?;

        // Extract embeddings for touched nodes only
        let mut touched_embeddings = Vec::new();
        for target_id in touched_node_ids {
            if let Some(idx) = subgraph
                .nodes
                .iter()
                .position(|n| n.app_id == *target_id || n.element_id == *target_id)
            {
                touched_embeddings.push(all_embeddings[idx].clone());
            }
        }

        Ok(self.build(&touched_embeddings))
    }
}

impl Default for GnnGraphStateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::{RelationType, SubGraphEdge, SubGraphNode};
    use std::collections::HashMap;

    fn make_test_subgraph(num_nodes: usize, num_edges: usize) -> SubGraph {
        let mut nodes = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let mut props = HashMap::new();
            props.insert("pagerank".to_string(), serde_json::json!(0.01 * i as f64));
            props.insert("degree_in".to_string(), serde_json::json!(i % 5));
            props.insert("energy".to_string(), serde_json::json!(0.5));
            nodes.push(SubGraphNode {
                element_id: format!("elem_{}", i),
                app_id: format!("app_{}", i),
                labels: vec!["File".to_string()],
                properties: props,
            });
        }

        let mut edges = Vec::new();
        for i in 0..num_edges {
            let s = i % num_nodes;
            let t = (i * 3 + 1) % num_nodes;
            if s == t {
                continue;
            }
            edges.push(SubGraphEdge {
                source_element_id: format!("elem_{}", s),
                target_element_id: format!("elem_{}", t),
                relation_type: RelationType::ALL[i % 8],
                weight: 1.0,
            });
        }

        SubGraph {
            center_id: "elem_0".to_string(),
            k_hops: 2,
            nodes,
            edges,
        }
    }

    #[test]
    fn test_candle_encoder_creation() -> Result<(), GnnError> {
        let config = CandleGnnEncoderConfig {
            encoder_config: GraphEncoderConfig {
                input_dim: TOTAL_FEATURE_DIM,
                hidden_dim: 64,
                output_dim: 64,
                num_layers: 2,
                num_relations: 8,
                num_bases: 4,
                dropout: 0.0,
                architecture: GNNArchitecture::GraphSAGE,
            },
            checkpoint_path: None,
            norm_stats: None,
        };

        let encoder = CandleGnnEncoder::new(config)?;
        assert_eq!(encoder.output_dim(), 64);
        assert!(encoder.is_inductive());
        Ok(())
    }

    #[test]
    fn test_encode_subgraph() -> Result<(), GnnError> {
        let config = CandleGnnEncoderConfig {
            encoder_config: GraphEncoderConfig {
                input_dim: TOTAL_FEATURE_DIM,
                hidden_dim: 64,
                output_dim: 32,
                num_layers: 2,
                num_relations: 8,
                num_bases: 4,
                dropout: 0.0,
                architecture: GNNArchitecture::GraphSAGE,
            },
            checkpoint_path: None,
            norm_stats: None,
        };

        let encoder = CandleGnnEncoder::new(config)?;
        let subgraph = make_test_subgraph(20, 40);
        let embeddings = encoder.encode_subgraph(&subgraph)?;

        assert_eq!(embeddings.len(), 20);
        assert_eq!(embeddings[0].len(), 32);

        // All embeddings should be finite
        for emb in &embeddings {
            for &v in emb {
                assert!(v.is_finite(), "Embedding value should be finite");
            }
        }

        Ok(())
    }

    #[test]
    fn test_encode_node() -> Result<(), GnnError> {
        let config = CandleGnnEncoderConfig {
            encoder_config: GraphEncoderConfig {
                input_dim: TOTAL_FEATURE_DIM,
                hidden_dim: 64,
                output_dim: 32,
                num_layers: 1,
                num_relations: 8,
                num_bases: 4,
                dropout: 0.0,
                architecture: GNNArchitecture::RGCN,
            },
            checkpoint_path: None,
            norm_stats: None,
        };

        let encoder = CandleGnnEncoder::new(config)?;
        let subgraph = make_test_subgraph(10, 20);

        let emb = encoder.encode_node(&subgraph, "app_3")?;
        assert_eq!(emb.len(), 32);

        // Non-existent node should error
        let err = encoder.encode_node(&subgraph, "nonexistent");
        assert!(err.is_err());

        Ok(())
    }

    #[test]
    fn test_encode_reproducible() -> Result<(), GnnError> {
        let config = CandleGnnEncoderConfig {
            encoder_config: GraphEncoderConfig {
                input_dim: TOTAL_FEATURE_DIM,
                hidden_dim: 32,
                output_dim: 16,
                num_layers: 1,
                num_relations: 8,
                num_bases: 4,
                dropout: 0.0,
                architecture: GNNArchitecture::GraphSAGE,
            },
            checkpoint_path: None,
            norm_stats: None,
        };

        let encoder = CandleGnnEncoder::new(config)?;
        let subgraph = make_test_subgraph(10, 15);

        let emb1 = encoder.encode_subgraph(&subgraph)?;
        let emb2 = encoder.encode_subgraph(&subgraph)?;

        // Same input → same output
        for (a, b) in emb1.iter().zip(emb2.iter()) {
            for (&va, &vb) in a.iter().zip(b.iter()) {
                assert!(
                    (va - vb).abs() < 1e-6,
                    "Embeddings should be reproducible: {} vs {}",
                    va,
                    vb
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_cached_encoder() -> Result<(), GnnError> {
        let config = CandleGnnEncoderConfig {
            encoder_config: GraphEncoderConfig {
                input_dim: TOTAL_FEATURE_DIM,
                hidden_dim: 32,
                output_dim: 16,
                num_layers: 1,
                num_relations: 8,
                num_bases: 4,
                dropout: 0.0,
                architecture: GNNArchitecture::GraphSAGE,
            },
            checkpoint_path: None,
            norm_stats: None,
        };

        let inner = Arc::new(CandleGnnEncoder::new(config)?);
        let cached = CachedGnnEncoder::new(inner);

        let subgraph = make_test_subgraph(10, 15);

        // First call — cache miss
        let emb1 = cached.encode_subgraph(&subgraph)?;
        // moka uses eventual consistency; run_pending_tasks ensures entry_count is up to date
        cached.cache.run_pending_tasks();
        let stats1 = cached.cache_stats();
        assert_eq!(stats1.entry_count, 1);

        // Second call — cache hit (same result)
        let emb2 = cached.encode_subgraph(&subgraph)?;
        assert_eq!(emb1.len(), emb2.len());
        for (a, b) in emb1.iter().zip(emb2.iter()) {
            assert_eq!(a, b, "Cached result should match");
        }

        // Invalidate
        cached.invalidate_all();
        cached.cache.run_pending_tasks();
        let stats2 = cached.cache_stats();
        assert_eq!(stats2.entry_count, 0);

        Ok(())
    }

    #[test]
    fn test_gnn_graph_state_builder() -> Result<(), GnnError> {
        let builder = GnnGraphStateBuilder::new();

        // Empty input
        let empty = builder.build(&[]);
        assert_eq!(empty.len(), 64);
        assert!(empty.iter().all(|&v| v == 0.0));

        // With 256d embeddings
        let embs = vec![vec![0.1f32; 256], vec![0.2f32; 256], vec![0.3f32; 256]];
        let block = builder.build(&embs);
        assert_eq!(block.len(), 64);

        // Should have non-zero values
        let has_nonzero = block.iter().any(|&v| v != 0.0);
        assert!(has_nonzero, "Graph state block should be non-zero");

        Ok(())
    }

    #[test]
    fn test_gnn_graph_state_from_encoder() -> Result<(), GnnError> {
        let config = CandleGnnEncoderConfig {
            encoder_config: GraphEncoderConfig {
                input_dim: TOTAL_FEATURE_DIM,
                hidden_dim: 64,
                output_dim: 256,
                num_layers: 2,
                num_relations: 8,
                num_bases: 4,
                dropout: 0.0,
                architecture: GNNArchitecture::GraphSAGE,
            },
            checkpoint_path: None,
            norm_stats: None,
        };

        let encoder = CandleGnnEncoder::new(config)?;
        let subgraph = make_test_subgraph(20, 40);
        let gs_builder = GnnGraphStateBuilder::new();

        let touched = vec![
            "app_0".to_string(),
            "app_3".to_string(),
            "app_7".to_string(),
        ];
        let block = gs_builder.build_from_encoder(&encoder, &subgraph, &touched)?;
        assert_eq!(block.len(), 64);

        let has_nonzero = block.iter().any(|&v| v != 0.0);
        assert!(has_nonzero, "Graph state from encoder should be non-zero");

        Ok(())
    }

    #[test]
    fn test_empty_subgraph() -> Result<(), GnnError> {
        let config = CandleGnnEncoderConfig::default();
        let encoder = CandleGnnEncoder::new(config)?;

        let subgraph = SubGraph {
            center_id: "none".to_string(),
            k_hops: 0,
            nodes: vec![],
            edges: vec![],
        };

        let embs = encoder.encode_subgraph(&subgraph)?;
        assert_eq!(embs.len(), 0);
        Ok(())
    }

    #[test]
    fn test_subgraph_no_edges() -> Result<(), GnnError> {
        let config = CandleGnnEncoderConfig {
            encoder_config: GraphEncoderConfig {
                input_dim: TOTAL_FEATURE_DIM,
                hidden_dim: 32,
                output_dim: 16,
                num_layers: 1,
                num_relations: 8,
                num_bases: 4,
                dropout: 0.0,
                architecture: GNNArchitecture::GraphSAGE,
            },
            checkpoint_path: None,
            norm_stats: None,
        };

        let encoder = CandleGnnEncoder::new(config)?;

        // Subgraph with nodes but no edges
        let subgraph = SubGraph {
            center_id: "elem_0".to_string(),
            k_hops: 1,
            nodes: vec![
                SubGraphNode {
                    element_id: "elem_0".to_string(),
                    app_id: "app_0".to_string(),
                    labels: vec!["File".to_string()],
                    properties: HashMap::new(),
                },
                SubGraphNode {
                    element_id: "elem_1".to_string(),
                    app_id: "app_1".to_string(),
                    labels: vec!["Function".to_string()],
                    properties: HashMap::new(),
                },
            ],
            edges: vec![],
        };

        let embs = encoder.encode_subgraph(&subgraph)?;
        assert_eq!(embs.len(), 2);
        assert_eq!(embs[0].len(), 16);
        Ok(())
    }

    #[test]
    fn test_encode_500_nodes_performance() -> Result<(), GnnError> {
        let config = CandleGnnEncoderConfig {
            encoder_config: GraphEncoderConfig {
                input_dim: TOTAL_FEATURE_DIM,
                hidden_dim: 128,
                output_dim: 256,
                num_layers: 3,
                num_relations: 8,
                num_bases: 4,
                dropout: 0.0,
                architecture: GNNArchitecture::GraphSAGE,
            },
            checkpoint_path: None,
            norm_stats: None,
        };

        let encoder = CandleGnnEncoder::new(config)?;
        let subgraph = make_test_subgraph(500, 1000);

        let start = std::time::Instant::now();
        let embeddings = encoder.encode_subgraph(&subgraph)?;
        let elapsed = start.elapsed();

        assert_eq!(embeddings.len(), 500);
        assert_eq!(embeddings[0].len(), 256);

        // Should complete in reasonable time on CPU
        // (relaxed: 5s for CI, typically <50ms on real hardware)
        assert!(
            elapsed.as_secs() < 5,
            "500 node encoding took too long: {:?}",
            elapsed
        );

        Ok(())
    }
}
