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
use std::collections::HashMap;

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
///
/// Supports both **core entries** (static, from training) and **external entries**
/// (dynamic, from connected MCP servers). All search methods (nearest_neighbor,
/// top_k) operate across both sets transparently.
///
/// External entries are indexed by FQN for O(1) lookup and can be
/// registered/unregistered at runtime without affecting core entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCodebook {
    /// Core codebook entries (static, from training data).
    pub entries: Vec<CodebookEntry>,
    /// Dynamic external entries (from connected MCP servers).
    #[serde(default)]
    pub external_entries: Vec<CodebookEntry>,
    /// FQN → index into `external_entries` for O(1) lookup.
    #[serde(skip)]
    pub fqn_index: HashMap<String, usize>,
    /// OOD threshold: if best cosine similarity < threshold, the action is OOD.
    /// Calibrated on validation set.
    pub ood_threshold: f32,
}

/// Type alias for backward compatibility and clarity.
pub type DynamicCodebook = ActionCodebook;

impl ActionCodebook {
    /// Create an empty codebook.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            external_entries: Vec::new(),
            fqn_index: HashMap::new(),
            ood_threshold: 0.5, // default, should be calibrated
        }
    }

    /// Total number of actions (core + external).
    pub fn len(&self) -> usize {
        self.entries.len() + self.external_entries.len()
    }

    /// Number of core (internal) entries.
    pub fn core_len(&self) -> usize {
        self.entries.len()
    }

    /// Number of external entries.
    pub fn external_len(&self) -> usize {
        self.external_entries.len()
    }

    /// Whether the codebook is empty (no core or external entries).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.external_entries.is_empty()
    }

    /// Add a core entry to the codebook.
    pub fn add_entry(&mut self, entry: CodebookEntry) {
        self.entries.push(entry);
    }

    // ─── Dynamic external entry management ───────────────────────────────

    /// Register an external tool in the codebook.
    ///
    /// The tool is stored with `frequency=0` and `avg_reward=0.5` (neutral prior).
    /// The embedding should be 256d (ACTION_DIM). If the tool is already registered
    /// (same FQN), it will be replaced.
    ///
    /// # Arguments
    /// * `server_id` — MCP server identifier (used as `tool` field)
    /// * `tool_name` — tool name (used as `action` field)
    /// * `embedding` — 256d embedding vector
    pub fn register_external_tool(
        &mut self,
        server_id: &str,
        tool_name: &str,
        embedding: Vec<f32>,
    ) {
        let fqn = format!("{}::{}", server_id, tool_name);

        // Replace if already registered
        if let Some(&idx) = self.fqn_index.get(&fqn) {
            self.external_entries[idx] = CodebookEntry::new(
                server_id.to_string(),
                tool_name.to_string(),
                embedding,
                0,
                0.5,
            );
            return;
        }

        let idx = self.external_entries.len();
        self.external_entries.push(CodebookEntry::new(
            server_id.to_string(),
            tool_name.to_string(),
            embedding,
            0,
            0.5,
        ));
        self.fqn_index.insert(fqn, idx);
    }

    /// Unregister all external tools from a specific server.
    ///
    /// Removes all entries whose `tool` field matches `server_id`
    /// and rebuilds the FQN index.
    pub fn unregister_server(&mut self, server_id: &str) {
        self.external_entries
            .retain(|entry| entry.tool != server_id);
        self.rebuild_fqn_index();
    }

    /// Update an external tool's embedding and reward via running mean.
    ///
    /// Formula: `new_value = (old * (n-1) + observed) / n`
    ///
    /// Returns `true` if the tool was found and updated.
    pub fn update_from_trajectory(
        &mut self,
        fqn: &str,
        context_embedding: &[f32],
        reward: f32,
    ) -> bool {
        let idx = match self.fqn_index.get(fqn) {
            Some(&i) => i,
            None => return false,
        };

        let entry = &mut self.external_entries[idx];
        entry.frequency += 1;
        let n = entry.frequency as f32;

        // Running mean update for embedding
        if context_embedding.len() == entry.embedding.len() {
            for (old, new) in entry.embedding.iter_mut().zip(context_embedding.iter()) {
                *old = (*old * (n - 1.0) + *new) / n;
            }
        }

        // Running mean update for reward
        entry.avg_reward = (entry.avg_reward * (n - 1.0) + reward) / n;

        // Recalculate norm
        entry.norm = l2_norm(&entry.embedding);

        true
    }

    /// Get an external entry by FQN.
    pub fn get_external(&self, fqn: &str) -> Option<&CodebookEntry> {
        self.fqn_index
            .get(fqn)
            .and_then(|&idx| self.external_entries.get(idx))
    }

    /// Rebuild the FQN index after modifications to external_entries.
    fn rebuild_fqn_index(&mut self) {
        self.fqn_index.clear();
        for (i, entry) in self.external_entries.iter().enumerate() {
            let fqn = format!("{}::{}", entry.tool, entry.action);
            self.fqn_index.insert(fqn, i);
        }
    }

    /// Rebuild FQN index from deserialized data (call after load_json).
    pub fn rebuild_index(&mut self) {
        self.rebuild_fqn_index();
    }

    // ─── Search (across core + external) ─────────────────────────────────

    /// Iterate over all entries (core + external).
    fn all_entries(&self) -> impl Iterator<Item = (usize, &CodebookEntry, bool)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e, false))
            .chain(
                self.external_entries
                    .iter()
                    .enumerate()
                    .map(|(i, e)| (self.entries.len() + i, e, true)),
            )
    }

    /// Find the nearest neighbor to a query vector.
    ///
    /// Searches across both core and external entries.
    /// Returns `(entry_index, cosine_similarity)` where the index is into
    /// the combined [core..external] space. Use `get_entry()` to retrieve it.
    pub fn nearest_neighbor(&self, query: &[f32]) -> Option<(usize, f32)> {
        if self.is_empty() || query.len() != ACTION_DIM {
            return None;
        }

        let query_norm = l2_norm(query);
        if query_norm < 1e-12 {
            return Some((0, 0.0));
        }

        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for (i, entry, _) in self.all_entries() {
            let sim = entry.cosine_similarity(query, query_norm);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        Some((best_idx, best_sim))
    }

    /// Find top-K nearest neighbors across core + external entries.
    ///
    /// Returns Vec of `(entry_index, cosine_similarity)`, sorted by similarity descending.
    pub fn top_k(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.is_empty() || query.len() != ACTION_DIM {
            return Vec::new();
        }

        let query_norm = l2_norm(query);
        let mut scored: Vec<(usize, f32)> = self
            .all_entries()
            .map(|(i, entry, _)| (i, entry.cosine_similarity(query, query_norm)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Get an entry by combined index (core + external).
    pub fn get_entry(&self, index: usize) -> Option<&CodebookEntry> {
        if index < self.entries.len() {
            self.entries.get(index)
        } else {
            self.external_entries.get(index - self.entries.len())
        }
    }

    /// Check if an index refers to an external entry.
    pub fn is_external(&self, index: usize) -> bool {
        index >= self.entries.len()
    }

    /// Sort core entries by frequency (most common first).
    /// External entries are not sorted (they use FQN index).
    pub fn sort_by_frequency(&mut self) {
        self.entries.sort_by_key(|e| std::cmp::Reverse(e.frequency));
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

    // ── Dynamic external entry tests ──────────────────────────────────

    #[test]
    fn test_register_external_tool() {
        let mut codebook = ActionCodebook::new();
        let emb = make_embedding(42.0);
        codebook.register_external_tool("grafeo", "run_cypher", emb.clone());

        assert_eq!(codebook.external_len(), 1);
        assert_eq!(codebook.len(), 1); // total
        assert_eq!(codebook.core_len(), 0);

        // FQN index works
        let entry = codebook.get_external("grafeo::run_cypher").unwrap();
        assert_eq!(entry.tool, "grafeo");
        assert_eq!(entry.action, "run_cypher");
        assert_eq!(entry.frequency, 0);
        assert!((entry.avg_reward - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_register_multiple_tools() {
        let mut codebook = ActionCodebook::new();
        codebook.register_external_tool("s1", "get_data", make_embedding(1.0));
        codebook.register_external_tool("s1", "list_items", make_embedding(2.0));
        codebook.register_external_tool("s2", "search", make_embedding(3.0));

        assert_eq!(codebook.external_len(), 3);
        assert!(codebook.get_external("s1::get_data").is_some());
        assert!(codebook.get_external("s1::list_items").is_some());
        assert!(codebook.get_external("s2::search").is_some());
    }

    #[test]
    fn test_register_replaces_existing() {
        let mut codebook = ActionCodebook::new();
        let emb1 = make_embedding(1.0);
        let emb2 = make_embedding(2.0);

        codebook.register_external_tool("s1", "tool", emb1);
        assert_eq!(codebook.external_len(), 1);

        codebook.register_external_tool("s1", "tool", emb2.clone());
        assert_eq!(codebook.external_len(), 1); // Still 1, replaced

        let entry = codebook.get_external("s1::tool").unwrap();
        assert!((entry.embedding[0] - emb2[0]).abs() < 1e-5);
    }

    #[test]
    fn test_unregister_server() {
        let mut codebook = ActionCodebook::new();
        codebook.register_external_tool("s1", "a", make_embedding(1.0));
        codebook.register_external_tool("s1", "b", make_embedding(2.0));
        codebook.register_external_tool("s2", "c", make_embedding(3.0));

        assert_eq!(codebook.external_len(), 3);

        codebook.unregister_server("s1");

        assert_eq!(codebook.external_len(), 1);
        assert!(codebook.get_external("s1::a").is_none());
        assert!(codebook.get_external("s1::b").is_none());
        assert!(codebook.get_external("s2::c").is_some());
    }

    #[test]
    fn test_unregister_nonexistent_server() {
        let mut codebook = ActionCodebook::new();
        codebook.register_external_tool("s1", "a", make_embedding(1.0));

        codebook.unregister_server("nonexistent");
        assert_eq!(codebook.external_len(), 1); // Unchanged
    }

    #[test]
    fn test_nearest_neighbor_finds_external() {
        let mut codebook = ActionCodebook::new();

        // Add core entry with seed 0
        codebook.add_entry(CodebookEntry::new(
            "note".into(),
            "create".into(),
            make_embedding(0.0),
            10,
            0.8,
        ));

        // Add external entry with seed 50 (different from core)
        codebook.register_external_tool("ext", "query", make_embedding(50.0));

        // Query with exact external embedding
        let query = make_embedding(50.0);
        let (idx, sim) = codebook.nearest_neighbor(&query).unwrap();

        // Should match the external entry (index 1 = after 1 core entry)
        assert_eq!(idx, 1);
        assert!(codebook.is_external(idx));
        assert!((sim - 1.0).abs() < 1e-5);

        // Verify get_entry works
        let entry = codebook.get_entry(idx).unwrap();
        assert_eq!(entry.tool, "ext");
        assert_eq!(entry.action, "query");
    }

    #[test]
    fn test_top_k_mixed_core_and_external() {
        let mut codebook = ActionCodebook::new();

        // 3 core entries
        for i in 0..3 {
            codebook.add_entry(CodebookEntry::new(
                "core".into(),
                format!("action_{i}"),
                make_embedding(i as f32 * 50.0),
                10,
                0.5,
            ));
        }

        // 2 external entries
        codebook.register_external_tool("ext", "tool_a", make_embedding(150.0));
        codebook.register_external_tool("ext", "tool_b", make_embedding(200.0));

        assert_eq!(codebook.len(), 5);

        let top5 = codebook.top_k(&make_embedding(0.0), 5);
        assert_eq!(top5.len(), 5);

        // First should be exact match (core action_0)
        assert_eq!(top5[0].0, 0);
        assert!(!codebook.is_external(top5[0].0));
    }

    #[test]
    fn test_update_from_trajectory() {
        let mut codebook = ActionCodebook::new();
        let initial = vec![1.0; ACTION_DIM];
        codebook.register_external_tool("s1", "tool", initial);

        // Update 10 times with embedding [3.0; 256]
        let update_emb = vec![3.0; ACTION_DIM];
        for _ in 0..10 {
            assert!(codebook.update_from_trajectory("s1::tool", &update_emb, 0.9));
        }

        let entry = codebook.get_external("s1::tool").unwrap();
        assert_eq!(entry.frequency, 10);

        // After 10 updates with [3.0; 256], starting from freq=0:
        // First update: freq=1, emb = (1.0*0 + 3.0)/1 = 3.0 (initial discarded, freq was 0)
        // All subsequent: emb stays 3.0 (running mean of identical values)
        assert!(
            (entry.embedding[0] - 3.0).abs() < 0.01,
            "Expected ~3.0, got {}",
            entry.embedding[0]
        );

        // avg_reward: same pattern, freq=0 means initial 0.5 is discarded
        // 10 updates of 0.9 → converges to 0.9
        assert!(
            (entry.avg_reward - 0.9).abs() < 0.01,
            "Expected reward ~0.9, got {}",
            entry.avg_reward
        );
    }

    #[test]
    fn test_update_from_trajectory_unknown_fqn() {
        let mut codebook = ActionCodebook::new();
        assert!(!codebook.update_from_trajectory("nonexistent::tool", &[0.0; ACTION_DIM], 1.0));
    }

    #[test]
    fn test_backward_compat_empty_external() {
        // ActionCodebook with no external entries should behave exactly like before
        let mut codebook = ActionCodebook::new();
        assert!(codebook.external_entries.is_empty());
        assert_eq!(codebook.external_len(), 0);

        // Core operations unchanged
        codebook.add_entry(CodebookEntry::new(
            "note".into(),
            "create".into(),
            make_embedding(1.0),
            5,
            0.7,
        ));

        assert_eq!(codebook.len(), 1);
        assert_eq!(codebook.core_len(), 1);
        assert!(!codebook.is_empty());

        let (idx, _) = codebook.nearest_neighbor(&make_embedding(1.0)).unwrap();
        assert_eq!(idx, 0);
        assert!(!codebook.is_external(idx));
    }

    #[test]
    fn test_dynamic_codebook_type_alias() {
        // DynamicCodebook is just ActionCodebook
        let _codebook: DynamicCodebook = ActionCodebook::new();
    }

    #[test]
    fn test_get_entry_core_and_external() {
        let mut codebook = ActionCodebook::new();
        codebook.add_entry(CodebookEntry::new(
            "core".into(),
            "act".into(),
            make_embedding(0.0),
            1,
            0.5,
        ));
        codebook.register_external_tool("ext", "tool", make_embedding(1.0));

        // Index 0 = core
        let core = codebook.get_entry(0).unwrap();
        assert_eq!(core.tool, "core");
        assert!(!codebook.is_external(0));

        // Index 1 = external
        let ext = codebook.get_entry(1).unwrap();
        assert_eq!(ext.tool, "ext");
        assert!(codebook.is_external(1));

        // Index 2 = out of bounds
        assert!(codebook.get_entry(2).is_none());
    }
}
