//! NodeFeatureBuilder — assembles fixed-dimension feature vectors for GNN training.
//!
//! Each node in the knowledge graph gets a feature vector composed of 5 blocks:
//! 1. **Voyage embedding projected** (128d): linear projection of 768d Voyage embeddings
//! 2. **GDS features** (12d): pagerank, betweenness, clustering coefficients, etc.
//! 3. **Type one-hot** (9d): File, Function, Struct, Note, Decision, Skill, Trajectory, TrajectoryNode, McpTool
//! 4. **Local metrics** (16d): co-change, synapse, energy, scar, staleness, etc.
//! 5. **Temporal features** (4d): age, recency, modification frequency
//!
//! Total: 169 dimensions per node.
//!
//! Missing features are imputed to 0.0 (simple imputation v1).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Feature dimensions
// ---------------------------------------------------------------------------

/// Projected Voyage embedding dimension.
pub const VOYAGE_DIM: usize = 128;
/// GDS structural features dimension.
pub const GDS_DIM: usize = 12;
/// Node type one-hot dimension.
pub const TYPE_DIM: usize = 9;
/// Local metrics dimension.
pub const LOCAL_DIM: usize = 16;
/// Temporal features dimension.
pub const TEMPORAL_DIM: usize = 4;

/// Total feature vector dimension per node.
pub const TOTAL_FEATURE_DIM: usize = VOYAGE_DIM + GDS_DIM + TYPE_DIM + LOCAL_DIM + TEMPORAL_DIM;

// ---------------------------------------------------------------------------
// Node type encoding
// ---------------------------------------------------------------------------

/// Node types supported by the GNN feature encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum NodeType {
    File = 0,
    Function = 1,
    Struct = 2,
    Note = 3,
    Decision = 4,
    Skill = 5,
    Trajectory = 6,
    TrajectoryNode = 7,
    McpTool = 8,
}

impl NodeType {
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "File" => Some(Self::File),
            "Function" => Some(Self::Function),
            "Struct" | "Trait" | "Enum" => Some(Self::Struct),
            "Note" => Some(Self::Note),
            "Decision" => Some(Self::Decision),
            "Skill" => Some(Self::Skill),
            "Trajectory" => Some(Self::Trajectory),
            "TrajectoryNode" => Some(Self::TrajectoryNode),
            "McpTool" => Some(Self::McpTool),
            _ => None,
        }
    }

    /// One-hot encoding as TYPE_DIM vector.
    pub fn one_hot(&self) -> [f32; TYPE_DIM] {
        let mut v = [0.0f32; TYPE_DIM];
        v[*self as usize] = 1.0;
        v
    }
}

// ---------------------------------------------------------------------------
// Raw node data (input to feature builder)
// ---------------------------------------------------------------------------

/// Raw data for a single node, extracted from Neo4j properties.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RawNodeData {
    /// Node type (from Neo4j labels).
    pub node_type: Option<String>,
    /// Voyage embedding (768d, if available).
    pub voyage_embedding: Option<Vec<f32>>,

    // GDS features (12d)
    pub pagerank: Option<f64>,
    pub betweenness: Option<f64>,
    pub clustering_coeff: Option<f64>,
    pub community_id: Option<i64>,
    pub degree_in: Option<i64>,
    pub degree_out: Option<i64>,
    pub hub_score: Option<f64>,
    pub authority_score: Option<f64>,
    pub triangle_count: Option<i64>,
    pub local_clustering: Option<f64>,
    pub closeness_centrality: Option<f64>,
    pub leiden_community: Option<i64>,

    // Local metrics (first 8 used, 8 reserved)
    pub co_change_count: Option<i64>,
    pub synapse_count: Option<i64>,
    pub energy: Option<f64>,
    pub scar_intensity: Option<f64>,
    pub staleness: Option<f64>,
    pub knowledge_density: Option<f64>,
    pub churn_score: Option<f64>,
    pub bridge_score: Option<f64>,

    // Temporal
    pub created_at_epoch: Option<f64>,
    pub last_accessed_epoch: Option<f64>,
    pub modification_count: Option<i64>,
}

impl RawNodeData {
    /// Extract RawNodeData from a SubGraphNode's properties map.
    pub fn from_properties(labels: &[String], props: &HashMap<String, serde_json::Value>) -> Self {
        let node_type = labels.first().cloned();

        Self {
            node_type,
            voyage_embedding: props
                .get("embedding")
                .and_then(|v| serde_json::from_value(v.clone()).ok()),
            pagerank: get_f64(props, "pagerank"),
            betweenness: get_f64(props, "betweenness"),
            clustering_coeff: get_f64(props, "clustering_coeff"),
            community_id: get_i64(props, "community_id"),
            degree_in: get_i64(props, "degree_in"),
            degree_out: get_i64(props, "degree_out"),
            hub_score: get_f64(props, "hub_score"),
            authority_score: get_f64(props, "authority_score"),
            triangle_count: get_i64(props, "triangle_count"),
            local_clustering: get_f64(props, "local_clustering"),
            closeness_centrality: get_f64(props, "closeness_centrality"),
            leiden_community: get_i64(props, "leiden_community"),
            co_change_count: get_i64(props, "co_change_count"),
            synapse_count: get_i64(props, "synapse_count"),
            energy: get_f64(props, "energy"),
            scar_intensity: get_f64(props, "scar_intensity"),
            staleness: get_f64(props, "staleness"),
            knowledge_density: get_f64(props, "knowledge_density"),
            churn_score: get_f64(props, "churn_score"),
            bridge_score: get_f64(props, "bridge_score"),
            created_at_epoch: get_f64(props, "created_at_epoch"),
            last_accessed_epoch: get_f64(props, "last_accessed_epoch"),
            modification_count: get_i64(props, "modification_count"),
        }
    }
}

fn get_f64(props: &HashMap<String, serde_json::Value>, key: &str) -> Option<f64> {
    props.get(key).and_then(|v| v.as_f64())
}

fn get_i64(props: &HashMap<String, serde_json::Value>, key: &str) -> Option<i64> {
    props.get(key).and_then(|v| v.as_i64())
}

// ---------------------------------------------------------------------------
// Projection matrix (Voyage 768d → 128d)
// ---------------------------------------------------------------------------

/// Random projection matrix for Voyage 768d → 128d.
///
/// In v1, uses a deterministic pseudo-random Gaussian projection.
/// Future: learned projection via autoencoder.
#[derive(Clone)]
pub struct ProjectionMatrix {
    /// [128 × 768] flattened row-major.
    pub weights: Vec<f32>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl ProjectionMatrix {
    /// Create a deterministic pseudo-random projection (Achlioptas sparse random projection).
    pub fn random(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let scale = 1.0 / (output_dim as f32).sqrt();
        let mut weights = Vec::with_capacity(input_dim * output_dim);

        // Simple deterministic pseudo-random: hash-based
        for i in 0..(input_dim * output_dim) {
            let h = simple_hash(seed, i as u64);
            // Sparse random projection: +1, 0, -1 with probabilities 1/6, 2/3, 1/6
            let val = match h % 6 {
                0 => scale,
                5 => -scale,
                _ => 0.0,
            };
            weights.push(val);
        }

        Self {
            weights,
            input_dim,
            output_dim,
        }
    }

    /// Project a single vector.
    pub fn project(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim);
        let mut output = vec![0.0f32; self.output_dim];
        for (row, out_val) in output.iter_mut().enumerate().take(self.output_dim) {
            let mut sum = 0.0f32;
            for (col, in_val) in input.iter().enumerate().take(self.input_dim) {
                sum += self.weights[row * self.input_dim + col] * in_val;
            }
            *out_val = sum;
        }
        output
    }
}

/// Simple deterministic hash (splitmix64-inspired).
///
/// Used for deterministic pseudo-random operations: projection matrix,
/// shuffle, negative sampling. Not cryptographic.
pub fn simple_hash(seed: u64, index: u64) -> u64 {
    let mut h = seed.wrapping_mul(6364136223846793005).wrapping_add(index);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

// ---------------------------------------------------------------------------
// Z-score normalization stats
// ---------------------------------------------------------------------------

/// Per-dimension normalization statistics for z-score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormStats {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub dim: usize,
}

impl NormStats {
    /// Compute normalization stats from a batch of feature vectors.
    pub fn from_batch(batch: &[Vec<f32>]) -> Self {
        if batch.is_empty() {
            return Self {
                mean: vec![],
                std: vec![],
                dim: 0,
            };
        }

        let dim = batch[0].len();
        let n = batch.len() as f64;
        let mut mean = vec![0.0f64; dim];
        let mut m2 = vec![0.0f64; dim];

        // Welford's online algorithm for numerical stability
        for (count, vec) in batch.iter().enumerate() {
            let count_f = (count + 1) as f64;
            for (d, &val) in vec.iter().enumerate() {
                let v = val as f64;
                let delta = v - mean[d];
                mean[d] += delta / count_f;
                let delta2 = v - mean[d];
                m2[d] += delta * delta2;
            }
        }

        let std: Vec<f64> = m2
            .iter()
            .map(|&v| {
                let variance = v / n.max(1.0);
                let s = variance.sqrt();
                if s < 1e-8 {
                    1.0
                } else {
                    s
                } // Avoid division by zero
            })
            .collect();

        Self { mean, std, dim }
    }

    /// Apply z-score normalization to a single vector.
    pub fn normalize(&self, features: &mut [f32]) {
        for (i, val) in features.iter_mut().enumerate() {
            if i < self.dim {
                *val = ((*val as f64 - self.mean[i]) / self.std[i]) as f32;
            }
        }
    }

    /// Apply z-score normalization to a batch.
    pub fn normalize_batch(&self, batch: &mut [Vec<f32>]) {
        for vec in batch.iter_mut() {
            self.normalize(vec);
        }
    }
}

// ---------------------------------------------------------------------------
// NodeFeatureBuilder
// ---------------------------------------------------------------------------

/// Builds fixed-dimension feature vectors for graph nodes.
pub struct NodeFeatureBuilder {
    projection: ProjectionMatrix,
    /// Current time epoch for temporal feature computation.
    now_epoch: f64,
}

impl NodeFeatureBuilder {
    pub fn new() -> Self {
        Self {
            projection: ProjectionMatrix::random(768, VOYAGE_DIM, 42),
            now_epoch: chrono::Utc::now().timestamp() as f64,
        }
    }

    pub fn with_projection(projection: ProjectionMatrix) -> Self {
        Self {
            projection,
            now_epoch: chrono::Utc::now().timestamp() as f64,
        }
    }

    /// Build feature vector for a single node.
    ///
    /// Returns a vector of TOTAL_FEATURE_DIM dimensions.
    /// Missing features are imputed to 0.0.
    pub fn build(&self, data: &RawNodeData) -> Vec<f32> {
        let mut features = Vec::with_capacity(TOTAL_FEATURE_DIM);

        // Block 1: Voyage embedding projected (128d)
        if let Some(ref emb) = data.voyage_embedding {
            if emb.len() == 768 {
                features.extend(self.projection.project(emb));
            } else {
                features.extend(vec![0.0f32; VOYAGE_DIM]);
            }
        } else {
            features.extend(vec![0.0f32; VOYAGE_DIM]);
        }

        // Block 2: GDS features (12d)
        features.push(data.pagerank.unwrap_or(0.0) as f32);
        features.push(data.betweenness.unwrap_or(0.0) as f32);
        features.push(data.clustering_coeff.unwrap_or(0.0) as f32);
        features.push(data.community_id.unwrap_or(0) as f32);
        features.push(data.degree_in.unwrap_or(0) as f32);
        features.push(data.degree_out.unwrap_or(0) as f32);
        features.push(data.hub_score.unwrap_or(0.0) as f32);
        features.push(data.authority_score.unwrap_or(0.0) as f32);
        features.push(data.triangle_count.unwrap_or(0) as f32);
        features.push(data.local_clustering.unwrap_or(0.0) as f32);
        features.push(data.closeness_centrality.unwrap_or(0.0) as f32);
        features.push(data.leiden_community.unwrap_or(0) as f32);

        // Block 3: Type one-hot (8d)
        let type_vec = data
            .node_type
            .as_deref()
            .and_then(NodeType::from_label)
            .map(|t| t.one_hot())
            .unwrap_or([0.0f32; TYPE_DIM]);
        features.extend(type_vec);

        // Block 4: Local metrics (16d = 8 used + 8 reserved zeros)
        features.push(data.co_change_count.unwrap_or(0) as f32);
        features.push(data.synapse_count.unwrap_or(0) as f32);
        features.push(data.energy.unwrap_or(0.0) as f32);
        features.push(data.scar_intensity.unwrap_or(0.0) as f32);
        features.push(data.staleness.unwrap_or(0.0) as f32);
        features.push(data.knowledge_density.unwrap_or(0.0) as f32);
        features.push(data.churn_score.unwrap_or(0.0) as f32);
        features.push(data.bridge_score.unwrap_or(0.0) as f32);
        features.extend(vec![0.0f32; 8]); // 8 reserved dims

        // Block 5: Temporal features (4d)
        let age_days = if let Some(created) = data.created_at_epoch {
            ((self.now_epoch - created) / 86400.0).max(0.0)
        } else {
            0.0
        };
        let last_accessed_days = if let Some(accessed) = data.last_accessed_epoch {
            ((self.now_epoch - accessed) / 86400.0).max(0.0)
        } else {
            0.0
        };
        let modification_freq = data.modification_count.unwrap_or(0) as f64 / age_days.max(1.0);
        let recency = 1.0 / (1.0 + last_accessed_days); // decay: recent = close to 1

        features.push((age_days + 1.0).ln() as f32); // log(age + 1)
        features.push((last_accessed_days + 1.0).ln() as f32); // log(last_accessed + 1)
        features.push(modification_freq as f32);
        features.push(recency as f32);

        debug_assert_eq!(features.len(), TOTAL_FEATURE_DIM);
        features
    }

    /// Build feature vectors for a batch of nodes.
    ///
    /// Returns (feature_matrix, norm_stats) where feature_matrix has
    /// shape [batch_size × TOTAL_FEATURE_DIM] and norm_stats contains
    /// the computed z-score statistics.
    pub fn build_batch(&self, nodes: &[RawNodeData]) -> (Vec<Vec<f32>>, NormStats) {
        let mut batch: Vec<Vec<f32>> = nodes.iter().map(|n| self.build(n)).collect();
        let stats = NormStats::from_batch(&batch);
        stats.normalize_batch(&mut batch);
        (batch, stats)
    }

    /// Build feature vectors with pre-existing normalization stats.
    pub fn build_batch_with_stats(
        &self,
        nodes: &[RawNodeData],
        stats: &NormStats,
    ) -> Vec<Vec<f32>> {
        let mut batch: Vec<Vec<f32>> = nodes.iter().map(|n| self.build(n)).collect();
        stats.normalize_batch(&mut batch);
        batch
    }
}

impl Default for NodeFeatureBuilder {
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

    fn make_empty_node() -> RawNodeData {
        RawNodeData::default()
    }

    fn make_full_node() -> RawNodeData {
        RawNodeData {
            node_type: Some("File".to_string()),
            voyage_embedding: Some(vec![0.1; 768]),
            pagerank: Some(0.05),
            betweenness: Some(0.3),
            clustering_coeff: Some(0.7),
            community_id: Some(2),
            degree_in: Some(5),
            degree_out: Some(3),
            hub_score: Some(0.1),
            authority_score: Some(0.2),
            triangle_count: Some(4),
            local_clustering: Some(0.6),
            closeness_centrality: Some(0.4),
            leiden_community: Some(1),
            co_change_count: Some(10),
            synapse_count: Some(5),
            energy: Some(0.8),
            scar_intensity: Some(0.1),
            staleness: Some(0.2),
            knowledge_density: Some(0.9),
            churn_score: Some(0.3),
            bridge_score: Some(0.5),
            created_at_epoch: Some(1700000000.0),
            last_accessed_epoch: Some(1710000000.0),
            modification_count: Some(50),
        }
    }

    #[test]
    fn test_total_dim_constant() {
        assert_eq!(TOTAL_FEATURE_DIM, 169);
    }

    #[test]
    fn test_build_empty_node_has_correct_dim() {
        let builder = NodeFeatureBuilder::new();
        let features = builder.build(&make_empty_node());
        assert_eq!(features.len(), TOTAL_FEATURE_DIM);
    }

    #[test]
    fn test_build_full_node_has_correct_dim() {
        let builder = NodeFeatureBuilder::new();
        let features = builder.build(&make_full_node());
        assert_eq!(features.len(), TOTAL_FEATURE_DIM);
    }

    #[test]
    fn test_missing_features_imputed_to_zero() {
        let builder = NodeFeatureBuilder::new();
        let features = builder.build(&make_empty_node());

        // Voyage block should be all zeros (no embedding)
        assert!(features[..VOYAGE_DIM].iter().all(|&v| v == 0.0));
        // GDS block should be all zeros
        let gds_start = VOYAGE_DIM;
        assert!(features[gds_start..gds_start + GDS_DIM]
            .iter()
            .all(|&v| v == 0.0));
        // Type block should be all zeros (unknown type)
        let type_start = gds_start + GDS_DIM;
        assert!(features[type_start..type_start + TYPE_DIM]
            .iter()
            .all(|&v| v == 0.0));
    }

    #[test]
    fn test_type_one_hot_encoding() {
        let builder = NodeFeatureBuilder::new();

        let mut node = make_empty_node();
        node.node_type = Some("File".to_string());
        let features = builder.build(&node);

        let type_start = VOYAGE_DIM + GDS_DIM;
        assert_eq!(features[type_start], 1.0); // File = index 0
        assert!(features[type_start + 1..type_start + TYPE_DIM]
            .iter()
            .all(|&v| v == 0.0));

        node.node_type = Some("Function".to_string());
        let features = builder.build(&node);
        assert_eq!(features[type_start], 0.0);
        assert_eq!(features[type_start + 1], 1.0); // Function = index 1
    }

    #[test]
    fn test_node_type_from_label() {
        assert_eq!(NodeType::from_label("File"), Some(NodeType::File));
        assert_eq!(NodeType::from_label("Function"), Some(NodeType::Function));
        assert_eq!(NodeType::from_label("Struct"), Some(NodeType::Struct));
        assert_eq!(NodeType::from_label("Trait"), Some(NodeType::Struct)); // maps to Struct
        assert_eq!(NodeType::from_label("Enum"), Some(NodeType::Struct)); // maps to Struct
        assert_eq!(NodeType::from_label("Note"), Some(NodeType::Note));
        assert_eq!(NodeType::from_label("Unknown"), None);
    }

    #[test]
    fn test_gds_features_populated() {
        let builder = NodeFeatureBuilder::new();
        let node = make_full_node();
        let features = builder.build(&node);

        let gds_start = VOYAGE_DIM;
        assert!((features[gds_start] - 0.05).abs() < 1e-5); // pagerank
        assert!((features[gds_start + 1] - 0.3).abs() < 1e-5); // betweenness
        assert!((features[gds_start + 4] - 5.0).abs() < 1e-5); // degree_in
    }

    #[test]
    fn test_temporal_features() {
        let now = chrono::Utc::now().timestamp() as f64;
        let mut builder = NodeFeatureBuilder::new();
        builder.now_epoch = now;

        let mut node = make_empty_node();
        node.created_at_epoch = Some(now - 86400.0 * 30.0); // 30 days ago
        node.last_accessed_epoch = Some(now - 86400.0 * 1.0); // 1 day ago
        node.modification_count = Some(60);

        let features = builder.build(&node);
        let temp_start = VOYAGE_DIM + GDS_DIM + TYPE_DIM + LOCAL_DIM;

        let age_log = features[temp_start]; // log(30 + 1)
        let recency = features[temp_start + 3]; // 1 / (1 + 1) = 0.5

        assert!(age_log > 3.0); // ln(31) ≈ 3.43
        assert!((recency - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_projection_deterministic() {
        let proj1 = ProjectionMatrix::random(768, 128, 42);
        let proj2 = ProjectionMatrix::random(768, 128, 42);
        assert_eq!(proj1.weights, proj2.weights);
    }

    #[test]
    fn test_projection_different_seeds() {
        let proj1 = ProjectionMatrix::random(768, 128, 42);
        let proj2 = ProjectionMatrix::random(768, 128, 99);
        assert_ne!(proj1.weights, proj2.weights);
    }

    #[test]
    fn test_projection_output_dim() {
        let proj = ProjectionMatrix::random(768, 128, 42);
        let input = vec![0.1f32; 768];
        let output = proj.project(&input);
        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_norm_stats_from_batch() {
        let batch = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![3.0, 4.0, 5.0],
            vec![5.0, 6.0, 7.0],
        ];
        let stats = NormStats::from_batch(&batch);

        assert_eq!(stats.dim, 3);
        // Mean should be [3.0, 4.0, 5.0]
        assert!((stats.mean[0] - 3.0).abs() < 1e-10);
        assert!((stats.mean[1] - 4.0).abs() < 1e-10);
        assert!((stats.mean[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_z_score_normalization() {
        let mut batch = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![3.0, 4.0, 5.0],
            vec![5.0, 6.0, 7.0],
        ];
        let stats = NormStats::from_batch(&batch);
        stats.normalize_batch(&mut batch);

        // After normalization, mean ≈ 0 and std ≈ 1
        let n = batch.len() as f32;
        for d in 0..3 {
            let col_mean: f32 = batch.iter().map(|v| v[d]).sum::<f32>() / n;
            assert!(
                col_mean.abs() < 1e-5,
                "Mean of dim {} should be ~0, got {}",
                d,
                col_mean
            );
        }
    }

    #[test]
    fn test_build_batch() {
        let builder = NodeFeatureBuilder::new();
        let nodes = vec![make_full_node(), make_empty_node(), make_full_node()];
        let (batch, stats) = builder.build_batch(&nodes);

        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].len(), TOTAL_FEATURE_DIM);
        assert_eq!(stats.dim, TOTAL_FEATURE_DIM);
    }

    #[test]
    fn test_build_batch_with_stats() {
        let builder = NodeFeatureBuilder::new();
        let train_nodes = vec![make_full_node(), make_empty_node(), make_full_node()];
        let (_, stats) = builder.build_batch(&train_nodes);

        // Apply same stats to new data
        let test_nodes = vec![make_full_node()];
        let test_batch = builder.build_batch_with_stats(&test_nodes, &stats);
        assert_eq!(test_batch.len(), 1);
        assert_eq!(test_batch[0].len(), TOTAL_FEATURE_DIM);
    }

    #[test]
    fn test_from_properties() {
        let labels = vec!["File".to_string()];
        let mut props = HashMap::new();
        props.insert("pagerank".to_string(), serde_json::json!(0.05));
        props.insert("degree_in".to_string(), serde_json::json!(5));
        props.insert("energy".to_string(), serde_json::json!(0.8));

        let data = RawNodeData::from_properties(&labels, &props);
        assert_eq!(data.node_type.as_deref(), Some("File"));
        assert!((data.pagerank.unwrap() - 0.05).abs() < 1e-10);
        assert_eq!(data.degree_in, Some(5));
        assert!((data.energy.unwrap() - 0.8).abs() < 1e-10);
        // Missing fields should be None
        assert!(data.betweenness.is_none());
        assert!(data.voyage_embedding.is_none());
    }

    #[test]
    fn test_norm_stats_empty_batch() {
        let stats = NormStats::from_batch(&[]);
        assert_eq!(stats.dim, 0);
        assert!(stats.mean.is_empty());
    }

    #[test]
    fn test_norm_stats_constant_dimension() {
        // If a dimension is constant, std should be 1.0 (not 0 to avoid div by zero)
        let batch = vec![vec![5.0f32, 1.0], vec![5.0, 2.0], vec![5.0, 3.0]];
        let stats = NormStats::from_batch(&batch);
        assert!(
            (stats.std[0] - 1.0).abs() < 1e-10,
            "Constant dim std should be 1.0"
        );
    }

    #[test]
    fn test_voyage_projection_nonzero() {
        let builder = NodeFeatureBuilder::new();
        let mut node = make_empty_node();
        node.voyage_embedding = Some(vec![0.1; 768]);
        let features = builder.build(&node);

        // Projected block should have some non-zero values
        let has_nonzero = features[..VOYAGE_DIM].iter().any(|&v| v != 0.0);
        assert!(has_nonzero, "Projected Voyage embedding should be non-zero");
    }

    #[test]
    fn test_voyage_wrong_dim_imputed() {
        let builder = NodeFeatureBuilder::new();
        let mut node = make_empty_node();
        node.voyage_embedding = Some(vec![0.1; 100]); // wrong dimension
        let features = builder.build(&node);

        // Should be imputed to zeros (wrong dim)
        assert!(features[..VOYAGE_DIM].iter().all(|&v| v == 0.0));
    }
}
