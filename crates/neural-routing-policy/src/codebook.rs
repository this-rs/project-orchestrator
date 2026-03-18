//! Action Codebook — catalog of MCP actions with learned embeddings.
//!
//! The codebook maps each (tool, action) pair to a 256d embedding vector,
//! computed as the mean of historical DecisionVectors for that action.
//!
//! Structure:
//! - `CodebookEntry`: single action with embedding, frequency, avg reward
//! - `ActionCodebook`: the full codebook with nearest-neighbor lookup
//! - `CodebookBuilder`: incremental construction from trajectory data
//!
//! The codebook is the bridge between continuous policy outputs (256d vectors)
//! and discrete MCP actions (tool_name, action_name, param_template).

use serde::{Deserialize, Serialize};

use crate::dataset::ACTION_DIM;

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

/// A single entry in the action codebook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebookEntry {
    /// MCP tool name (e.g., "note", "task", "code").
    pub tool: String,
    /// MCP action name (e.g., "create", "search_semantic", "get").
    pub action: String,
    /// Optional parameter template (JSON-like hints for common params).
    pub param_template: Option<String>,
    /// Mean embedding vector (ACTION_DIM = 256d).
    pub embedding: Vec<f32>,
    /// Number of times this action was observed in training data.
    pub frequency: usize,
    /// Average reward when this action was taken.
    pub avg_reward: f32,
    /// L2 norm of the embedding (cached for fast cosine computation).
    pub norm: f32,
}

impl CodebookEntry {
    /// Create a new entry with a pre-computed embedding.
    pub fn new(
        tool: String,
        action: String,
        embedding: Vec<f32>,
        frequency: usize,
        avg_reward: f32,
    ) -> Self {
        let norm = l2_norm(&embedding);
        Self {
            tool,
            action,
            param_template: None,
            embedding,
            frequency,
            avg_reward,
            norm,
        }
    }

    /// Unique key for this action: "tool.action".
    pub fn key(&self) -> String {
        format!("{}.{}", self.tool, self.action)
    }

    /// Cosine similarity with a query vector.
    pub fn cosine_similarity(&self, query: &[f32], query_norm: f32) -> f32 {
        if self.norm < 1e-12 || query_norm < 1e-12 {
            return 0.0;
        }
        let dot: f32 = self
            .embedding
            .iter()
            .zip(query.iter())
            .map(|(a, b)| a * b)
            .sum();
        dot / (self.norm * query_norm)
    }
}

// ---------------------------------------------------------------------------
// Codebook
// ---------------------------------------------------------------------------

/// The action codebook: a catalog of all known MCP actions with embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCodebook {
    /// All codebook entries, sorted by frequency (most common first).
    pub entries: Vec<CodebookEntry>,
    /// OOD threshold: if best cosine similarity < threshold, the action is OOD.
    /// Calibrated on validation set.
    pub ood_threshold: f32,
}

impl ActionCodebook {
    /// Create an empty codebook.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            ood_threshold: 0.5, // default, should be calibrated
        }
    }

    /// Number of actions in the codebook.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the codebook is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add an entry to the codebook.
    pub fn add_entry(&mut self, entry: CodebookEntry) {
        self.entries.push(entry);
    }

    /// Find the nearest neighbor to a query vector.
    ///
    /// Returns `(entry_index, cosine_similarity)`.
    /// If the codebook is empty, returns `None`.
    pub fn nearest_neighbor(&self, query: &[f32]) -> Option<(usize, f32)> {
        if self.entries.is_empty() || query.len() != ACTION_DIM {
            return None;
        }

        let query_norm = l2_norm(query);
        if query_norm < 1e-12 {
            return Some((0, 0.0));
        }

        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for (i, entry) in self.entries.iter().enumerate() {
            let sim = entry.cosine_similarity(query, query_norm);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        Some((best_idx, best_sim))
    }

    /// Find top-K nearest neighbors.
    ///
    /// Returns Vec of `(entry_index, cosine_similarity)`, sorted by similarity descending.
    pub fn top_k(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.entries.is_empty() || query.len() != ACTION_DIM {
            return Vec::new();
        }

        let query_norm = l2_norm(query);
        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| (i, entry.cosine_similarity(query, query_norm)))
            .collect();

        // Sort descending by similarity
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Sort entries by frequency (most common first).
    pub fn sort_by_frequency(&mut self) {
        self.entries.sort_by(|a, b| b.frequency.cmp(&a.frequency));
    }

    /// Calibrate the OOD threshold from validation data.
    ///
    /// Strategy: find the threshold that achieves `target_recall` on known OOD samples.
    /// - `known_queries`: embeddings of valid actions (should match)
    /// - `ood_queries`: embeddings of invalid/random actions (should be rejected)
    /// - `target_recall`: desired OOD detection recall (e.g., 0.8)
    pub fn calibrate_ood_threshold(
        &mut self,
        known_queries: &[Vec<f32>],
        ood_queries: &[Vec<f32>],
        target_recall: f32,
    ) {
        if ood_queries.is_empty() {
            return;
        }

        // Compute similarities for OOD queries
        let mut ood_sims: Vec<f32> = ood_queries
            .iter()
            .filter_map(|q| self.nearest_neighbor(q).map(|(_, sim)| sim))
            .collect();

        ood_sims.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Find threshold that rejects target_recall fraction of OOD
        let reject_count = (ood_sims.len() as f32 * target_recall).ceil() as usize;
        let threshold_idx = reject_count.min(ood_sims.len()).saturating_sub(1);

        // The threshold should be above this percentile of OOD similarities
        let candidate = ood_sims.get(threshold_idx).copied().unwrap_or(0.5);

        // But also ensure we don't reject too many known queries
        let known_sims: Vec<f32> = known_queries
            .iter()
            .filter_map(|q| self.nearest_neighbor(q).map(|(_, sim)| sim))
            .collect();

        let known_rejected = known_sims.iter().filter(|&&s| s < candidate).count();
        let known_reject_rate = if known_sims.is_empty() {
            0.0
        } else {
            known_rejected as f32 / known_sims.len() as f32
        };

        // If we'd reject >20% of known queries, lower the threshold
        self.ood_threshold = if known_reject_rate > 0.2 {
            // Find a compromise: keep 80% of known queries
            let mut sorted_known = known_sims.clone();
            sorted_known.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let keep_idx = (sorted_known.len() as f32 * 0.2).ceil() as usize;
            sorted_known
                .get(keep_idx)
                .copied()
                .unwrap_or(candidate)
                .min(candidate)
        } else {
            candidate
        };
    }

    /// Save codebook to JSON.
    pub fn save_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::other(format!("JSON serialize: {e}")))?;
        std::fs::write(path, json)
    }

    /// Load codebook from JSON.
    pub fn load_json(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::other(format!("JSON deserialize: {e}")))
    }
}

impl Default for ActionCodebook {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Accumulator for building a codebook from observed action embeddings.
#[derive(Debug)]
struct ActionAccumulator {
    tool: String,
    action: String,
    embeddings_sum: Vec<f32>,
    reward_sum: f32,
    count: usize,
}

impl ActionAccumulator {
    fn new(tool: String, action: String) -> Self {
        Self {
            tool,
            action,
            embeddings_sum: vec![0.0; ACTION_DIM],
            reward_sum: 0.0,
            count: 0,
        }
    }

    fn add(&mut self, embedding: &[f32], reward: f32) {
        for (acc, val) in self.embeddings_sum.iter_mut().zip(embedding.iter()) {
            *acc += val;
        }
        self.reward_sum += reward;
        self.count += 1;
    }

    fn to_entry(&self) -> CodebookEntry {
        let count = self.count.max(1) as f32;
        let mean_embedding: Vec<f32> = self.embeddings_sum.iter().map(|v| v / count).collect();
        let avg_reward = self.reward_sum / count;

        CodebookEntry::new(
            self.tool.clone(),
            self.action.clone(),
            mean_embedding,
            self.count,
            avg_reward,
        )
    }
}

/// Builder for constructing an ActionCodebook incrementally.
pub struct CodebookBuilder {
    accumulators: std::collections::HashMap<String, ActionAccumulator>,
}

impl CodebookBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            accumulators: std::collections::HashMap::new(),
        }
    }

    /// Observe an action with its embedding and reward.
    pub fn observe(&mut self, tool: &str, action: &str, embedding: &[f32], reward: f32) {
        let key = format!("{tool}.{action}");
        let acc = self
            .accumulators
            .entry(key)
            .or_insert_with(|| ActionAccumulator::new(tool.to_string(), action.to_string()));
        acc.add(embedding, reward);
    }

    /// Build the codebook from all observations.
    pub fn build(self) -> ActionCodebook {
        let mut codebook = ActionCodebook::new();
        for (_, acc) in self.accumulators {
            codebook.add_entry(acc.to_entry());
        }
        codebook.sort_by_frequency();
        codebook
    }
}

impl Default for CodebookBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// L2 norm of a vector.
pub(crate) fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(seed: f32) -> Vec<f32> {
        (0..ACTION_DIM)
            .map(|i| ((i as f32 + seed) * 0.1).sin())
            .collect()
    }

    #[test]
    fn test_codebook_entry_cosine() {
        let emb = make_embedding(1.0);
        let entry = CodebookEntry::new("note".into(), "create".into(), emb.clone(), 10, 0.8);

        // Self-similarity should be 1.0
        let query_norm = l2_norm(&emb);
        let sim = entry.cosine_similarity(&emb, query_norm);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Self-similarity should be ~1.0, got {}",
            sim
        );

        // Different vector should have lower similarity
        let diff = make_embedding(100.0);
        let diff_norm = l2_norm(&diff);
        let sim2 = entry.cosine_similarity(&diff, diff_norm);
        assert!(sim2 < 1.0, "Different vectors should have sim < 1.0");
    }

    #[test]
    fn test_codebook_nearest_neighbor() {
        let mut codebook = ActionCodebook::new();

        // Add 3 entries with distinct embeddings
        for i in 0..3 {
            let emb = make_embedding(i as f32 * 50.0);
            codebook.add_entry(CodebookEntry::new(
                "tool".into(),
                format!("action_{i}"),
                emb,
                10 - i,
                0.5,
            ));
        }

        // Query with the exact embedding of action_1
        let query = make_embedding(50.0);
        let (idx, sim) = codebook.nearest_neighbor(&query).unwrap();
        assert_eq!(idx, 1, "Should find action_1");
        assert!((sim - 1.0).abs() < 1e-5, "Exact match should have sim ~1.0");
    }

    #[test]
    fn test_codebook_top_k() {
        let mut codebook = ActionCodebook::new();

        for i in 0..10 {
            let emb = make_embedding(i as f32 * 30.0);
            codebook.add_entry(CodebookEntry::new(
                "tool".into(),
                format!("action_{i}"),
                emb,
                1,
                0.5,
            ));
        }

        let query = make_embedding(0.0);
        let top3 = codebook.top_k(&query, 3);
        assert_eq!(top3.len(), 3);

        // First result should be the exact match (action_0)
        assert_eq!(top3[0].0, 0);
        assert!((top3[0].1 - 1.0).abs() < 1e-5);

        // Results should be in descending similarity order
        for w in top3.windows(2) {
            assert!(w[0].1 >= w[1].1, "Top-K should be sorted descending");
        }
    }

    #[test]
    fn test_codebook_builder() {
        let mut builder = CodebookBuilder::new();

        // Observe same action multiple times → mean embedding
        let emb1 = vec![1.0; ACTION_DIM];
        let emb2 = vec![3.0; ACTION_DIM];
        builder.observe("note", "create", &emb1, 0.8);
        builder.observe("note", "create", &emb2, 0.6);
        builder.observe("task", "update", &emb1, 0.9);

        let codebook = builder.build();
        assert_eq!(codebook.len(), 2);

        // Find note.create — its mean embedding should be [2.0; 256]
        let note_create = codebook
            .entries
            .iter()
            .find(|e| e.key() == "note.create")
            .unwrap();
        assert_eq!(note_create.frequency, 2);
        assert!((note_create.avg_reward - 0.7).abs() < 1e-5);
        assert!((note_create.embedding[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_ood_calibration() {
        let mut codebook = ActionCodebook::new();

        // Add entries
        for i in 0..5 {
            let emb = make_embedding(i as f32 * 10.0);
            codebook.add_entry(CodebookEntry::new(
                "tool".into(),
                format!("action_{i}"),
                emb,
                5,
                0.5,
            ));
        }

        // Known queries: same as codebook entries (high similarity)
        let known: Vec<Vec<f32>> = (0..5).map(|i| make_embedding(i as f32 * 10.0)).collect();

        // OOD queries: random/distant embeddings
        let ood: Vec<Vec<f32>> = (0..20).map(|i| make_embedding(i as f32 * 1000.0)).collect();

        codebook.calibrate_ood_threshold(&known, &ood, 0.8);

        // The threshold should be set such that most OOD are rejected
        assert!(
            codebook.ood_threshold > 0.0,
            "Threshold should be positive, got {}",
            codebook.ood_threshold
        );
        assert!(
            codebook.ood_threshold < 1.0,
            "Threshold should be < 1.0, got {}",
            codebook.ood_threshold
        );

        // Verify: known queries should mostly pass the threshold
        let known_pass = known
            .iter()
            .filter(|q| {
                codebook
                    .nearest_neighbor(q)
                    .map(|(_, sim)| sim >= codebook.ood_threshold)
                    .unwrap_or(false)
            })
            .count();
        assert!(
            known_pass >= 4,
            "At least 4/5 known queries should pass, got {}",
            known_pass
        );
    }

    #[test]
    fn test_empty_codebook() {
        let codebook = ActionCodebook::new();
        assert!(codebook.is_empty());
        assert_eq!(codebook.nearest_neighbor(&[0.0; ACTION_DIM]), None);
        assert!(codebook.top_k(&[0.0; ACTION_DIM], 5).is_empty());
    }

    #[test]
    fn test_sort_by_frequency() {
        let mut codebook = ActionCodebook::new();
        codebook.add_entry(CodebookEntry::new(
            "a".into(),
            "low".into(),
            vec![0.0; ACTION_DIM],
            1,
            0.0,
        ));
        codebook.add_entry(CodebookEntry::new(
            "b".into(),
            "high".into(),
            vec![0.0; ACTION_DIM],
            100,
            0.0,
        ));
        codebook.add_entry(CodebookEntry::new(
            "c".into(),
            "mid".into(),
            vec![0.0; ACTION_DIM],
            50,
            0.0,
        ));

        codebook.sort_by_frequency();
        assert_eq!(codebook.entries[0].frequency, 100);
        assert_eq!(codebook.entries[1].frequency, 50);
        assert_eq!(codebook.entries[2].frequency, 1);
    }
}
