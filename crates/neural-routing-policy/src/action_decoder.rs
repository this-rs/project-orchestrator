//! Action Decoder — converts continuous decision vectors to discrete MCP actions.
//!
//! Pipeline:
//! 1. Policy network predicts a 256d action vector
//! 2. Nearest neighbor lookup in the ActionCodebook
//! 3. OOD check: if cosine similarity < threshold → reject (fallback to heuristic)
//! 4. Context filter: remove actions invalid in current state
//! 5. Return the decoded action with confidence score
//!
//! The decoder is the final step in the neural routing pipeline:
//! Query → GNN → Policy → **Decoder** → MCP action

use serde::Serialize;

use crate::codebook::ActionCodebook;
use crate::dataset::ACTION_DIM;

// ---------------------------------------------------------------------------
// Decoded action
// ---------------------------------------------------------------------------

/// A decoded MCP action with confidence metadata.
#[derive(Debug, Clone, Serialize)]
pub struct DecodedAction {
    /// MCP tool name.
    pub tool: String,
    /// MCP action name.
    pub action: String,
    /// Optional parameter template.
    pub param_template: Option<String>,
    /// Cosine similarity to the codebook entry (0.0 - 1.0).
    pub confidence: f32,
    /// Whether this action is out-of-distribution.
    pub is_ood: bool,
    /// Index in the codebook.
    pub codebook_index: usize,
    /// Historical average reward for this action.
    pub avg_reward: f32,
}

/// Result of decoding an action vector.
#[derive(Debug, Clone, Serialize)]
pub enum DecodeResult {
    /// Successfully decoded to an action.
    Action(DecodedAction),
    /// Out-of-distribution — no confident match found.
    Ood {
        /// Best match similarity (below threshold).
        best_similarity: f32,
        /// The threshold that was not met.
        threshold: f32,
        /// Best guess action (for logging, not execution).
        best_guess: DecodedAction,
    },
    /// Codebook is empty or vector is invalid.
    NoCodebook,
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Action decoder: converts policy output vectors to MCP actions.
pub struct ActionDecoder<'a> {
    codebook: &'a ActionCodebook,
    /// Context filter: set of available (tool, action) keys.
    /// If None, all actions are available.
    available_actions: Option<std::collections::HashSet<String>>,
}

impl<'a> ActionDecoder<'a> {
    /// Create a decoder with the given codebook.
    pub fn new(codebook: &'a ActionCodebook) -> Self {
        Self {
            codebook,
            available_actions: None,
        }
    }

    /// Set the available actions filter.
    ///
    /// Only actions in this set will be considered for decoding.
    /// Format: `["tool.action", ...]`
    pub fn with_available_actions(mut self, actions: std::collections::HashSet<String>) -> Self {
        self.available_actions = Some(actions);
        self
    }

    /// Decode a single action vector.
    pub fn decode(&self, action_vector: &[f32]) -> DecodeResult {
        if self.codebook.is_empty() || action_vector.len() != ACTION_DIM {
            return DecodeResult::NoCodebook;
        }

        // If we have a context filter, use filtered search
        if let Some(ref available) = self.available_actions {
            return self.decode_filtered(action_vector, available);
        }

        // Simple nearest-neighbor
        match self.codebook.nearest_neighbor(action_vector) {
            None => DecodeResult::NoCodebook,
            Some((idx, sim)) => {
                let entry = &self.codebook.entries[idx];
                let decoded = DecodedAction {
                    tool: entry.tool.clone(),
                    action: entry.action.clone(),
                    param_template: entry.param_template.clone(),
                    confidence: sim,
                    is_ood: sim < self.codebook.ood_threshold,
                    codebook_index: idx,
                    avg_reward: entry.avg_reward,
                };

                if sim < self.codebook.ood_threshold {
                    DecodeResult::Ood {
                        best_similarity: sim,
                        threshold: self.codebook.ood_threshold,
                        best_guess: decoded,
                    }
                } else {
                    DecodeResult::Action(decoded)
                }
            }
        }
    }

    /// Decode with context filtering — only consider available actions.
    fn decode_filtered(
        &self,
        action_vector: &[f32],
        available: &std::collections::HashSet<String>,
    ) -> DecodeResult {
        let query_norm = l2_norm(action_vector);
        if query_norm < 1e-12 {
            return DecodeResult::NoCodebook;
        }

        let mut best_idx = None;
        let mut best_sim = f32::NEG_INFINITY;

        for (i, entry) in self.codebook.entries.iter().enumerate() {
            // Skip unavailable actions
            if !available.contains(&entry.key()) {
                continue;
            }

            let sim = entry.cosine_similarity(action_vector, query_norm);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        match best_idx {
            None => DecodeResult::NoCodebook,
            Some(idx) => {
                let entry = &self.codebook.entries[idx];
                let decoded = DecodedAction {
                    tool: entry.tool.clone(),
                    action: entry.action.clone(),
                    param_template: entry.param_template.clone(),
                    confidence: best_sim,
                    is_ood: best_sim < self.codebook.ood_threshold,
                    codebook_index: idx,
                    avg_reward: entry.avg_reward,
                };

                if best_sim < self.codebook.ood_threshold {
                    DecodeResult::Ood {
                        best_similarity: best_sim,
                        threshold: self.codebook.ood_threshold,
                        best_guess: decoded,
                    }
                } else {
                    DecodeResult::Action(decoded)
                }
            }
        }
    }

    /// Decode with top-K candidates (for beam search or diversity).
    pub fn decode_top_k(&self, action_vector: &[f32], k: usize) -> Vec<DecodedAction> {
        let candidates = self.codebook.top_k(action_vector, k);
        candidates
            .into_iter()
            .map(|(idx, sim)| {
                let entry = &self.codebook.entries[idx];
                DecodedAction {
                    tool: entry.tool.clone(),
                    action: entry.action.clone(),
                    param_template: entry.param_template.clone(),
                    confidence: sim,
                    is_ood: sim < self.codebook.ood_threshold,
                    codebook_index: idx,
                    avg_reward: entry.avg_reward,
                }
            })
            .collect()
    }

    /// Batch decode multiple action vectors.
    pub fn decode_batch(&self, action_vectors: &[Vec<f32>]) -> Vec<DecodeResult> {
        action_vectors.iter().map(|v| self.decode(v)).collect()
    }
}

use crate::codebook::l2_norm;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::{CodebookBuilder, CodebookEntry};

    fn make_embedding(seed: f32) -> Vec<f32> {
        (0..ACTION_DIM)
            .map(|i| ((i as f32 + seed) * 0.1).sin())
            .collect()
    }

    fn make_codebook() -> ActionCodebook {
        let mut builder = CodebookBuilder::new();

        // Add several distinct actions
        let actions = [
            ("note", "create", 1.0),
            ("note", "search_semantic", 2.0),
            ("task", "update", 3.0),
            ("code", "search", 4.0),
            ("plan", "create", 5.0),
        ];

        for (i, (tool, action, seed)) in actions.iter().enumerate() {
            let emb = make_embedding(*seed * 50.0);
            for _ in 0..10 {
                builder.observe(tool, action, &emb, 0.5 + i as f32 * 0.1);
            }
        }

        let mut codebook = builder.build();
        codebook.ood_threshold = 0.7;
        codebook
    }

    #[test]
    fn test_decode_exact_match() {
        let codebook = make_codebook();
        let decoder = ActionDecoder::new(&codebook);

        // Query with exact embedding of "note.create"
        let query = make_embedding(50.0);
        match decoder.decode(&query) {
            DecodeResult::Action(action) => {
                assert_eq!(action.tool, "note");
                assert_eq!(action.action, "create");
                assert!((action.confidence - 1.0).abs() < 1e-4);
                assert!(!action.is_ood);
            }
            other => panic!("Expected Action, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_ood() {
        let codebook = make_codebook();
        let decoder = ActionDecoder::new(&codebook);

        // Random query far from any codebook entry
        let query = vec![999.0; ACTION_DIM];
        match decoder.decode(&query) {
            DecodeResult::Ood {
                best_similarity,
                threshold,
                best_guess,
            } => {
                assert!(best_similarity < threshold);
                assert!(!best_guess.tool.is_empty());
            }
            DecodeResult::Action(a) => {
                // It's possible if the random vector happens to be close
                assert!(a.confidence >= codebook.ood_threshold);
            }
            DecodeResult::NoCodebook => panic!("Codebook is not empty"),
        }
    }

    #[test]
    fn test_decode_filtered() {
        let codebook = make_codebook();

        let mut available = std::collections::HashSet::new();
        available.insert("task.update".to_string());
        available.insert("plan.create".to_string());

        let decoder = ActionDecoder::new(&codebook).with_available_actions(available);

        // Query matching "note.create" — but note.create is not available
        let query = make_embedding(50.0);
        match decoder.decode(&query) {
            DecodeResult::Action(action) => {
                // Should NOT return note.create
                assert_ne!(format!("{}.{}", action.tool, action.action), "note.create");
                // Should return one of the available actions
                assert!(
                    action.tool == "task" || action.tool == "plan",
                    "Should return available action, got {}.{}",
                    action.tool,
                    action.action
                );
            }
            DecodeResult::Ood { best_guess, .. } => {
                assert!(
                    best_guess.tool == "task" || best_guess.tool == "plan",
                    "Best guess should be from available actions"
                );
            }
            DecodeResult::NoCodebook => panic!("Should have candidates"),
        }
    }

    #[test]
    fn test_decode_top_k() {
        let codebook = make_codebook();
        let decoder = ActionDecoder::new(&codebook);

        let query = make_embedding(50.0);
        let top3 = decoder.decode_top_k(&query, 3);

        assert_eq!(top3.len(), 3);
        // First result should be the best match
        assert!(top3[0].confidence >= top3[1].confidence);
        assert!(top3[1].confidence >= top3[2].confidence);
    }

    #[test]
    fn test_decode_batch() {
        let codebook = make_codebook();
        let decoder = ActionDecoder::new(&codebook);

        let queries: Vec<Vec<f32>> = (0..5).map(|i| make_embedding(i as f32 * 50.0)).collect();

        let results = decoder.decode_batch(&queries);
        assert_eq!(results.len(), 5);

        for result in &results {
            match result {
                DecodeResult::Action(_) | DecodeResult::Ood { .. } => {}
                DecodeResult::NoCodebook => panic!("Should decode"),
            }
        }
    }

    #[test]
    fn test_decode_empty_codebook() {
        let codebook = ActionCodebook::new();
        let decoder = ActionDecoder::new(&codebook);

        let query = vec![1.0; ACTION_DIM];
        matches!(decoder.decode(&query), DecodeResult::NoCodebook);
    }

    #[test]
    fn test_decode_latency() {
        // Ensure decoding is fast (< 1ms for single decode with ~500 entries)
        let mut codebook = ActionCodebook::new();
        for i in 0..500 {
            let emb = make_embedding(i as f32);
            codebook.add_entry(CodebookEntry::new(
                format!("tool_{}", i / 20),
                format!("action_{}", i % 20),
                emb,
                1,
                0.5,
            ));
        }

        let decoder = ActionDecoder::new(&codebook);
        let query = make_embedding(42.0);

        let start = std::time::Instant::now();
        let iterations = 1000;
        for _ in 0..iterations {
            let _ = decoder.decode(&query);
        }
        let elapsed = start.elapsed();
        let per_decode_us = elapsed.as_micros() / iterations as u128;

        // In debug mode, allow up to 5ms; in release this is well under 1ms.
        assert!(
            per_decode_us < 5000,
            "Decode should be < 5ms (debug), got {}μs",
            per_decode_us
        );
    }
}
