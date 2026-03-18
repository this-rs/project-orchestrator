//! DecisionVector builder — constructs 256d context embeddings at each decision point.
//!
//! The vector encodes the complete reasoning state:
//! - **query_embedding** (64d): projected from Voyage 768d via random orthogonal matrix
//! - **graph_state** (64d): mean-pooled GDS features of recently touched nodes
//! - **history** (64d): mean-pooled embeddings of N previous decisions in the trajectory
//! - **tool_context** (32d): features of the current tool/action
//! - **session_metadata** (32d): duration, decision count, cumulative reward, avg confidence
//!
//! Total: 64 + 64 + 64 + 32 + 32 = 256 dimensions, L2-normalized.

use serde::{Deserialize, Serialize};

// Dimension constants
pub const QUERY_DIM: usize = 64;
pub const GRAPH_DIM: usize = 64;
pub const HISTORY_DIM: usize = 64;
pub const TOOL_DIM: usize = 32;
pub const SESSION_DIM: usize = 32;
pub const TOTAL_DIM: usize = QUERY_DIM + GRAPH_DIM + HISTORY_DIM + TOOL_DIM + SESSION_DIM; // 256

/// Source embedding dimension (Voyage / fastembed).
pub const SOURCE_EMBED_DIM: usize = 768;

// ---------------------------------------------------------------------------
// Projection matrix — random orthogonal (deterministic seed)
// ---------------------------------------------------------------------------

/// A projection matrix that maps SOURCE_EMBED_DIM → target_dim.
///
/// Initialized with a deterministic pseudo-random orthogonal matrix
/// (Gaussian random + QR decomposition approximation via normalization).
#[derive(Debug, Clone)]
pub struct ProjectionMatrix {
    /// Row-major: [target_dim × source_dim].
    weights: Vec<f32>,
    source_dim: usize,
    target_dim: usize,
}

impl ProjectionMatrix {
    /// Create a new projection matrix with a deterministic seed.
    ///
    /// Uses a simple LCG PRNG seeded with `seed` to generate Gaussian-like
    /// random values, then normalizes each row to unit length (approximate
    /// orthogonality for high dimensions).
    pub fn new(source_dim: usize, target_dim: usize, seed: u64) -> Self {
        let mut weights = Vec::with_capacity(target_dim * source_dim);
        let mut rng_state = seed;

        for _ in 0..target_dim {
            let mut row = Vec::with_capacity(source_dim);
            let mut norm_sq = 0.0_f64;

            for _ in 0..source_dim {
                // Box-Muller approximation via LCG
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u1 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u2 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;

                // Box-Muller transform
                let u1_clamped = u1.clamp(1e-10, 1.0 - 1e-10);
                let z = (-2.0 * u1_clamped.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let val = z as f32;
                norm_sq += (val as f64) * (val as f64);
                row.push(val);
            }

            // Normalize row to unit length
            let norm = (norm_sq.sqrt()) as f32;
            if norm > 1e-10 {
                for v in &mut row {
                    *v /= norm;
                }
            }

            weights.extend(row);
        }

        Self {
            weights,
            source_dim,
            target_dim,
        }
    }

    /// Project a source vector to the target dimension.
    ///
    /// If `source` has fewer dimensions than expected, it's zero-padded.
    /// If it has more, the extra dimensions are ignored.
    pub fn project(&self, source: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.target_dim);

        for row in 0..self.target_dim {
            let row_offset = row * self.source_dim;
            let mut dot = 0.0_f32;
            let len = source.len().min(self.source_dim);
            for (j, &src_val) in source.iter().enumerate().take(len) {
                dot += self.weights[row_offset + j] * src_val;
            }
            result.push(dot);
        }

        result
    }
}

// ---------------------------------------------------------------------------
// GDS node features (input for graph_state block)
// ---------------------------------------------------------------------------

/// Features of a graph node (from Neo4j GDS or local cache).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeFeatures {
    /// PageRank score.
    pub pagerank: f32,
    /// Betweenness centrality.
    pub betweenness: f32,
    /// Community ID (Louvain).
    pub community_id: u32,
    /// Degree centrality.
    pub degree: f32,
    /// Churn score (how often the node changes).
    pub churn_score: f32,
    /// Knowledge density (notes/decisions linked).
    pub knowledge_density: f32,
}

impl NodeFeatures {
    /// Convert to a fixed-size feature vector (6d).
    fn to_vec(&self) -> [f32; 6] {
        [
            self.pagerank,
            self.betweenness,
            (self.community_id as f32) / 100.0, // normalized
            self.degree,
            self.churn_score,
            self.knowledge_density,
        ]
    }
}

// ---------------------------------------------------------------------------
// Tool features (input for tool_context block)
// ---------------------------------------------------------------------------

/// Known tool categories for one-hot encoding.
const TOOL_CATEGORIES: &[&str] = &[
    "code",
    "note",
    "plan",
    "task",
    "step",
    "decision",
    "commit",
    "protocol",
    "skill",
    "persona",
    "reasoning",
    "admin",
    "feature_graph",
    "chat",
    "milestone",
    "release",
    "constraint",
    "workspace",
    "resource",
    "component",
    "episode",
    "sharing",
    "analysis_profile",
    "trajectory",
];

// ---------------------------------------------------------------------------
// DecisionVectorBuilder
// ---------------------------------------------------------------------------

/// Builder that constructs a 256d DecisionVector from contextual inputs.
///
/// Reuse this builder across decisions in a session — the projection matrix
/// is expensive to construct but cheap to apply (<5ms per vector).
pub struct DecisionVectorBuilder {
    /// 768d → 64d projection for query embeddings.
    query_projection: ProjectionMatrix,
    /// 768d → 64d projection for history embeddings (same seed = same matrix is fine,
    /// but we use a different seed for decorrelation).
    history_projection: ProjectionMatrix,
}

impl DecisionVectorBuilder {
    /// Create a new builder with deterministic projection matrices.
    pub fn new() -> Self {
        Self {
            query_projection: ProjectionMatrix::new(SOURCE_EMBED_DIM, QUERY_DIM, 42),
            history_projection: ProjectionMatrix::new(SOURCE_EMBED_DIM, HISTORY_DIM, 137),
        }
    }

    /// Build a 256d DecisionVector from the given context.
    ///
    /// Returns a L2-normalized Vec<f32> of exactly 256 dimensions.
    pub fn build(&self, ctx: &DecisionContext) -> Vec<f32> {
        let mut vector = Vec::with_capacity(TOTAL_DIM);

        // Block 1: query_embedding (64d)
        let query_block = self.build_query_block(&ctx.query_embedding);
        vector.extend_from_slice(&query_block);

        // Block 2: graph_state (64d)
        let graph_block = self.build_graph_block(&ctx.touched_node_features);
        vector.extend_from_slice(&graph_block);

        // Block 3: history (64d)
        let history_block = self.build_history_block(&ctx.previous_embeddings);
        vector.extend_from_slice(&history_block);

        // Block 4: tool_context (32d)
        let tool_block = self.build_tool_block(&ctx.tool_name, &ctx.action_name, ctx.params_hash);
        vector.extend_from_slice(&tool_block);

        // Block 5: session_metadata (32d)
        let session_block = self.build_session_block(&ctx.session_meta);
        vector.extend_from_slice(&session_block);

        debug_assert_eq!(vector.len(), TOTAL_DIM);

        // L2 normalize
        l2_normalize(&mut vector);

        vector
    }

    /// Block 1: Project 768d query embedding → 64d.
    fn build_query_block(&self, query_embedding: &[f32]) -> Vec<f32> {
        if query_embedding.is_empty() {
            return sentinel_vector(QUERY_DIM, SENTINEL_SEED);
        }
        self.query_projection.project(query_embedding)
    }

    /// Block 2: Mean-pool GDS features of touched nodes → 64d.
    ///
    /// We expand each node's 6d features into a 64d space via tiling + position encoding.
    fn build_graph_block(&self, node_features: &[NodeFeatures]) -> Vec<f32> {
        if node_features.is_empty() {
            return sentinel_vector(GRAPH_DIM, SENTINEL_SEED.wrapping_add(1));
        }

        // Mean-pool raw features (6d)
        let n = node_features.len() as f32;
        let mut mean = [0.0_f32; 6];
        for nf in node_features {
            let fv = nf.to_vec();
            for (i, v) in fv.iter().enumerate() {
                mean[i] += v / n;
            }
        }

        // Expand 6d → 64d via tiling + sinusoidal position encoding
        let mut block = Vec::with_capacity(GRAPH_DIM);
        for i in 0..GRAPH_DIM {
            let feat_idx = i % 6;
            let position = i as f32 / GRAPH_DIM as f32;
            // Combine feature value with positional signal
            let val = if i < 6 {
                mean[feat_idx]
            } else if i % 2 == 0 {
                mean[feat_idx] * (position * std::f32::consts::PI).sin()
            } else {
                mean[feat_idx] * (position * std::f32::consts::PI).cos()
            };
            block.push(val);
        }

        // Also encode the count of nodes (as a signal of context richness)
        if block.len() > 1 {
            block[GRAPH_DIM - 1] = (node_features.len() as f32).ln().max(0.0) / 5.0;
        }

        block
    }

    /// Block 3: Mean-pool previous decision embeddings → 64d.
    ///
    /// If previous embeddings are 256d (full DecisionVectors), we project to 64d.
    /// If they're already 768d source embeddings, we use the history projection.
    fn build_history_block(&self, previous_embeddings: &[Vec<f32>]) -> Vec<f32> {
        if previous_embeddings.is_empty() {
            return sentinel_vector(HISTORY_DIM, SENTINEL_SEED.wrapping_add(2));
        }

        let mut mean = vec![0.0_f32; HISTORY_DIM];
        let n = previous_embeddings.len() as f32;

        for emb in previous_embeddings {
            let projected = if emb.len() == SOURCE_EMBED_DIM {
                // 768d source embedding → project to 64d
                self.history_projection.project(emb)
            } else if emb.len() >= TOTAL_DIM {
                // 256d full vector → take the first 64d (query block) as history signal
                emb[..HISTORY_DIM].to_vec()
            } else if emb.len() == HISTORY_DIM {
                emb.clone()
            } else {
                // Unknown size — zero-pad or truncate
                let mut v = emb.clone();
                v.resize(HISTORY_DIM, 0.0);
                v
            };

            for (i, v) in projected.iter().enumerate().take(HISTORY_DIM) {
                mean[i] += v / n;
            }
        }

        mean
    }

    /// Block 4: Tool context → 32d.
    ///
    /// One-hot for tool category (24d) + 8d for action hash features.
    fn build_tool_block(&self, tool_name: &str, action_name: &str, params_hash: u64) -> Vec<f32> {
        let mut block = vec![0.0_f32; TOOL_DIM];

        // One-hot encode tool category (first 24d)
        if let Some(idx) = TOOL_CATEGORIES.iter().position(|&t| t == tool_name) {
            if idx < 24 {
                block[idx] = 1.0;
            }
        }

        // Action name hash features (8d, spread across positions 24..32)
        let action_hash = simple_hash(action_name);
        for i in 0..8 {
            let bit = (action_hash >> (i * 8)) & 0xFF;
            block[24 + i] = (bit as f32) / 255.0;
        }

        // Mix in params_hash as a subtle signal on the last 4 dims
        for i in 0..4 {
            let byte = ((params_hash >> (i * 16)) & 0xFFFF) as f32 / 65535.0;
            block[28 + i] = block[28 + i] * 0.5 + byte * 0.5;
        }

        block
    }

    /// Block 5: Session metadata → 32d.
    fn build_session_block(&self, meta: &SessionMeta) -> Vec<f32> {
        let mut block = vec![0.0_f32; SESSION_DIM];

        // Encode core metrics (first 8d)
        block[0] = (meta.duration_ms as f32).ln().max(0.0) / 15.0; // log duration, normalized
        block[1] = (meta.decision_count as f32).ln().max(0.0) / 5.0; // log count
        block[2] = meta.cumulative_reward as f32; // already 0-1 typically
        block[3] = meta.avg_confidence as f32; // already 0-1
        block[4] = meta.unique_tools_used as f32 / 10.0; // normalized
        block[5] = meta.unique_entities_touched as f32 / 50.0; // normalized
        block[6] = if meta.decision_count > 0 {
            meta.duration_ms as f32 / meta.decision_count as f32 / 10000.0
        } else {
            0.0
        }; // avg ms per decision, normalized
        block[7] = meta.error_count as f32 / (meta.decision_count as f32 + 1.0); // error rate

        // Spread remaining dimensions with cross-features (interactions)
        for i in 8..SESSION_DIM {
            let idx_a = i % 8;
            let idx_b = (i + 3) % 8;
            block[i] = block[idx_a] * block[idx_b]; // interaction features
        }

        block
    }
}

impl Default for DecisionVectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Context inputs
// ---------------------------------------------------------------------------

/// All context needed to build a DecisionVector at a decision point.
#[derive(Debug, Clone)]
pub struct DecisionContext {
    /// The original query embedding (768d from Voyage/fastembed).
    pub query_embedding: Vec<f32>,
    /// GDS features of recently touched graph nodes.
    pub touched_node_features: Vec<NodeFeatures>,
    /// Previous decision embeddings in this trajectory (newest last).
    /// Can be 768d source embeddings or 256d full DecisionVectors.
    pub previous_embeddings: Vec<Vec<f32>>,
    /// Current MCP tool name (e.g., "code").
    pub tool_name: String,
    /// Current action name (e.g., "search").
    pub action_name: String,
    /// Hash of the action parameters (for distinctness, not reconstruction).
    pub params_hash: u64,
    /// Session-level metadata.
    pub session_meta: SessionMeta,
}

/// Session-level metadata for the session_metadata block.
#[derive(Debug, Clone, Default)]
pub struct SessionMeta {
    /// Total session duration so far in milliseconds.
    pub duration_ms: u64,
    /// Number of decisions made so far.
    pub decision_count: usize,
    /// Cumulative reward so far.
    pub cumulative_reward: f64,
    /// Average confidence across decisions so far.
    pub avg_confidence: f64,
    /// Number of unique tools used.
    pub unique_tools_used: usize,
    /// Number of unique entities touched.
    pub unique_entities_touched: usize,
    /// Number of errors/failures.
    pub error_count: usize,
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// L2 normalize a vector in-place.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Generate a deterministic sentinel vector for empty contexts.
///
/// Instead of returning a zero vector (which causes NaN in cosine similarity
/// and collapses all empty contexts to the same point), we generate a
/// low-magnitude deterministic pseudo-random vector with a given seed.
/// The result is L2-normalized so it sits on the unit sphere.
pub fn sentinel_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut rng_state = seed;
    for _ in 0..dim {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
        let z = (-2.0 * u.clamp(1e-10, 1.0 - 1e-10).ln()).sqrt()
            * (2.0 * std::f64::consts::PI * u2).cos();
        v.push(z as f32);
    }
    l2_normalize(&mut v);
    v
}

/// Seed used for the sentinel vector (deterministic, reproducible).
const SENTINEL_SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;

/// Simple deterministic hash for strings (FNV-1a).
pub fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_context() -> DecisionContext {
        DecisionContext {
            query_embedding: vec![0.1; SOURCE_EMBED_DIM],
            touched_node_features: vec![
                NodeFeatures {
                    pagerank: 0.05,
                    betweenness: 0.1,
                    community_id: 3,
                    degree: 0.2,
                    churn_score: 0.3,
                    knowledge_density: 0.8,
                },
                NodeFeatures {
                    pagerank: 0.15,
                    betweenness: 0.05,
                    community_id: 3,
                    degree: 0.4,
                    churn_score: 0.1,
                    knowledge_density: 0.6,
                },
            ],
            previous_embeddings: vec![vec![0.2; SOURCE_EMBED_DIM], vec![0.3; SOURCE_EMBED_DIM]],
            tool_name: "code".to_string(),
            action_name: "search".to_string(),
            params_hash: 0xDEADBEEF,
            session_meta: SessionMeta {
                duration_ms: 5000,
                decision_count: 5,
                cumulative_reward: 0.7,
                avg_confidence: 0.85,
                unique_tools_used: 3,
                unique_entities_touched: 12,
                error_count: 0,
            },
        }
    }

    #[test]
    fn test_vector_has_256_dimensions() {
        let builder = DecisionVectorBuilder::new();
        let ctx = make_context();
        let vector = builder.build(&ctx);
        assert_eq!(vector.len(), TOTAL_DIM);
        assert_eq!(vector.len(), 256);
    }

    #[test]
    fn test_vector_is_l2_normalized() {
        let builder = DecisionVectorBuilder::new();
        let ctx = make_context();
        let vector = builder.build(&ctx);

        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "L2 norm should be 1.0, got {}",
            norm
        );
    }

    #[test]
    fn test_similar_contexts_have_high_cosine_similarity() {
        let builder = DecisionVectorBuilder::new();

        // Two similar contexts (same query, slightly different graph features)
        let ctx1 = make_context();
        let mut ctx2 = make_context();
        ctx2.touched_node_features[0].pagerank = 0.06; // tiny change

        let v1 = builder.build(&ctx1);
        let v2 = builder.build(&ctx2);

        let cosine: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        assert!(
            cosine > 0.95,
            "Similar contexts should have cosine > 0.95, got {}",
            cosine
        );
    }

    #[test]
    fn test_different_tools_produce_different_vectors() {
        let builder = DecisionVectorBuilder::new();

        let ctx1 = make_context();
        let mut ctx2 = make_context();
        ctx2.tool_name = "note".to_string();
        ctx2.action_name = "get_context".to_string();

        let v1 = builder.build(&ctx1);
        let v2 = builder.build(&ctx2);

        let cosine: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        // Should be different but not orthogonal (they share query/graph/history)
        assert!(
            cosine < 0.99,
            "Different tools should produce different vectors, cosine={}",
            cosine
        );
        assert!(
            cosine > 0.5,
            "Vectors should still share structure, cosine={}",
            cosine
        );
    }

    #[test]
    fn test_empty_context_produces_valid_vector() {
        let builder = DecisionVectorBuilder::new();
        let ctx = DecisionContext {
            query_embedding: vec![],
            touched_node_features: vec![],
            previous_embeddings: vec![],
            tool_name: "unknown".to_string(),
            action_name: "unknown".to_string(),
            params_hash: 0,
            session_meta: SessionMeta::default(),
        };

        let vector = builder.build(&ctx);
        assert_eq!(vector.len(), 256);

        // With sentinel vectors, empty context should produce a non-zero, normalized vector
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Empty context should produce L2-normalized vector, got norm={}",
            norm
        );
        // Should NOT be all zeros (sentinel replaces zero vectors)
        let all_zero = vector.iter().all(|&x| x.abs() < 1e-10);
        assert!(
            !all_zero,
            "Empty context must not produce zero vector (sentinel expected)"
        );
    }

    #[test]
    fn test_sentinel_vector_is_deterministic() {
        let v1 = sentinel_vector(64, 42);
        let v2 = sentinel_vector(64, 42);
        assert_eq!(v1, v2, "Same seed must produce identical sentinels");
    }

    #[test]
    fn test_sentinel_vector_is_normalized() {
        let v = sentinel_vector(256, 0xCAFE);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Sentinel should be L2-normalized, got norm={}",
            norm
        );
    }

    #[test]
    fn test_sentinel_different_seeds_differ() {
        let v1 = sentinel_vector(64, 1);
        let v2 = sentinel_vector(64, 2);
        assert_ne!(v1, v2, "Different seeds must produce different sentinels");
    }

    #[test]
    fn test_projection_preserves_relative_distances() {
        let proj = ProjectionMatrix::new(768, 64, 42);

        // Three source vectors: a, b (similar), c (different)
        let a: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut b = a.clone();
        b[0] += 0.01; // tiny perturbation
        let c: Vec<f32> = (0..768).map(|i| (i as f32 * 0.03).cos()).collect();

        let pa = proj.project(&a);
        let pb = proj.project(&b);
        let pc = proj.project(&c);

        let cos_ab: f32 = pa.iter().zip(pb.iter()).map(|(x, y)| x * y).sum::<f32>()
            / (pa.iter().map(|x| x * x).sum::<f32>().sqrt()
                * pb.iter().map(|x| x * x).sum::<f32>().sqrt());
        let cos_ac: f32 = pa.iter().zip(pc.iter()).map(|(x, y)| x * y).sum::<f32>()
            / (pa.iter().map(|x| x * x).sum::<f32>().sqrt()
                * pc.iter().map(|x| x * x).sum::<f32>().sqrt());

        assert!(
            cos_ab > cos_ac,
            "Similar vectors should be closer: cos(a,b)={} vs cos(a,c)={}",
            cos_ab,
            cos_ac
        );
    }

    #[test]
    fn bench_build_latency_under_5ms() {
        let builder = DecisionVectorBuilder::new();
        let ctx = make_context();

        let mut durations = Vec::with_capacity(100);
        for _ in 0..100 {
            let start = std::time::Instant::now();
            let _v = builder.build(&ctx);
            durations.push(start.elapsed());
        }

        durations.sort();
        let p50 = durations[49];
        let p99 = durations[98];

        eprintln!("DecisionVector build: p50={:?}, p99={:?}", p50, p99);

        assert!(
            p99 < std::time::Duration::from_millis(5),
            "p99 build latency {:?} exceeds 5ms budget",
            p99
        );
    }

    #[test]
    fn test_deterministic_projection() {
        // Same seed should produce identical results
        let p1 = ProjectionMatrix::new(768, 64, 42);
        let p2 = ProjectionMatrix::new(768, 64, 42);
        let input: Vec<f32> = (0..768).map(|i| i as f32 * 0.001).collect();

        let r1 = p1.project(&input);
        let r2 = p2.project(&input);

        assert_eq!(r1, r2, "Same seed should produce identical projections");
    }
}
