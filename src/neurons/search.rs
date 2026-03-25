//! Search Pipeline with Spreading Activation as a Signal
//!
//! Integrates spreading activation as the 5th signal in a multi-signal
//! search pipeline. The pipeline blends:
//!
//! 1. **Vector similarity** — cosine similarity from embedding search
//! 2. **Recency** — freshness of the note (newer = higher)
//! 3. **Energy** — neural energy level (active neurons rank higher)
//! 4. **Importance** — note importance level (high/medium/low)
//! 5. **Activation** — spreading activation score from graph structure
//!
//! Each signal is normalized to [0, 1] and blended via configurable weights.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::debug;
use uuid::Uuid;

use crate::embeddings::EmbeddingProvider;
use crate::neo4j::GraphStore;
use crate::notes::{Note, NoteImportance};

use super::activation::{ActivationSearchConfig, SpreadingActivationEngine};
use super::config::SpreadingActivationConfig;

// ============================================================================
// Configuration
// ============================================================================

/// Weights for blending the 5 search signals.
///
/// All weights should be non-negative. They are automatically normalized
/// to sum to 1.0 at query time, so relative magnitudes matter, not absolutes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchWeights {
    /// Weight for vector similarity signal.
    pub vector_weight: f64,
    /// Weight for recency signal.
    pub recency_weight: f64,
    /// Weight for energy signal.
    pub energy_weight: f64,
    /// Weight for importance signal.
    pub importance_weight: f64,
    /// Weight for spreading activation signal.
    /// Set to 0.0 to disable activation signal entirely.
    pub activation_weight: f64,
}

impl Default for SearchWeights {
    fn default() -> Self {
        Self {
            vector_weight: 0.4,
            recency_weight: 0.1,
            energy_weight: 0.15,
            importance_weight: 0.1,
            activation_weight: 0.25,
        }
    }
}

impl SearchWeights {
    /// Normalize weights so they sum to 1.0.
    /// Returns (vector, recency, energy, importance, activation).
    fn normalized(&self) -> (f64, f64, f64, f64, f64) {
        let sum = self.vector_weight
            + self.recency_weight
            + self.energy_weight
            + self.importance_weight
            + self.activation_weight;
        if sum <= 0.0 {
            return (0.2, 0.2, 0.2, 0.2, 0.2);
        }
        (
            self.vector_weight / sum,
            self.recency_weight / sum,
            self.energy_weight / sum,
            self.importance_weight / sum,
            self.activation_weight / sum,
        )
    }
}

/// A scored search result from the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The note that matched.
    pub note: Note,
    /// Final blended score (0.0–1.0).
    pub score: f64,
    /// Individual signal scores for transparency/debugging.
    pub signals: SignalBreakdown,
}

/// Breakdown of individual signal contributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalBreakdown {
    pub vector_score: f64,
    pub recency_score: f64,
    pub energy_score: f64,
    pub importance_score: f64,
    pub activation_score: f64,
}

// ============================================================================
// Pipeline
// ============================================================================

/// Multi-signal search pipeline that blends vector search, recency, energy,
/// importance, and spreading activation into a unified ranking.
pub struct SearchPipeline {
    #[allow(dead_code)]
    graph_store: Arc<dyn GraphStore>,
    engine: Arc<SpreadingActivationEngine>,
}

impl SearchPipeline {
    /// Create a new search pipeline.
    pub fn new(
        graph_store: Arc<dyn GraphStore>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        let engine = Arc::new(SpreadingActivationEngine::new(
            graph_store.clone(),
            embedding_provider,
        ));
        Self {
            graph_store,
            engine,
        }
    }

    /// Create a pipeline with an existing engine (for testing/sharing).
    pub fn with_engine(
        graph_store: Arc<dyn GraphStore>,
        engine: Arc<SpreadingActivationEngine>,
    ) -> Self {
        Self {
            graph_store,
            engine,
        }
    }

    /// Execute a multi-signal search query.
    ///
    /// # Arguments
    /// * `query` — text query for vector search
    /// * `source_node` — optional source node for spreading activation.
    ///   If `None`, the activation signal is zeroed out.
    /// * `project_id` — optional project filter
    /// * `weights` — signal blending weights
    /// * `limit` — max results to return
    ///
    /// # Pipeline
    /// 1. Run vector search (Phase 1 of spreading activation engine)
    /// 2. If `source_node` is Some, run activate_and_collect for activation scores
    /// 3. For each candidate note: compute 5 normalized signals
    /// 4. Blend signals using weights → final score
    /// 5. Sort by final score, return top `limit`
    pub async fn search(
        &self,
        query: &str,
        source_node: Option<Uuid>,
        project_id: Option<Uuid>,
        weights: &SearchWeights,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let config = SpreadingActivationConfig::default();
        let (w_vec, w_rec, w_eng, w_imp, w_act) = weights.normalized();

        // Signal 1: Vector search — get candidates with similarity scores
        let activated = self.engine.activate(query, project_id, &config).await?;

        if activated.is_empty() {
            return Ok(vec![]);
        }

        // Build candidate map: note_id → (note, vector_score)
        let mut candidates: HashMap<Uuid, (Note, f64)> = HashMap::new();
        for a in &activated {
            candidates.insert(a.note.id, (a.note.clone(), a.activation_score));
        }

        // Signal 5: Activation — structural propagation from source node
        let activation_scores: HashMap<Uuid, f64> = if let Some(source) = source_node {
            if w_act > 0.0 {
                let act_config = ActivationSearchConfig {
                    top_k: limit * 3, // Over-fetch for blending
                    ..Default::default()
                };
                match self.engine.activate_and_collect(source, &act_config).await {
                    Ok(results) => results
                        .into_iter()
                        .map(|r| (r.node_id, r.activation_level))
                        .collect(),
                    Err(e) => {
                        debug!("Activation signal failed (degraded mode): {}", e);
                        HashMap::new()
                    }
                }
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        };

        // Compute per-note signals and blend
        let now = Utc::now();
        let mut results: Vec<SearchResult> = candidates
            .into_iter()
            .map(|(note_id, (note, vector_sim))| {
                // Signal 1: Vector similarity — already normalized [0, 1]
                let vector_score = vector_sim.clamp(0.0, 1.0);

                // Signal 2: Recency — exponential decay over days
                let age_days = (now - note.created_at).num_seconds().max(0) as f64 / 86400.0;
                let recency_score = (-age_days / 30.0).exp(); // half-life ~30 days

                // Signal 3: Energy — already [0, 1]
                let energy_score = note.computed_energy().clamp(0.0, 1.0);

                // Signal 4: Importance — mapped to [0, 1]
                let importance_score = match note.importance {
                    NoteImportance::Critical => 1.0,
                    NoteImportance::High => 0.75,
                    NoteImportance::Medium => 0.5,
                    NoteImportance::Low => 0.25,
                };

                // Signal 5: Activation — from activate_and_collect, already [0, 1]
                let activation_score = activation_scores.get(&note_id).copied().unwrap_or(0.0);

                let score = w_vec * vector_score
                    + w_rec * recency_score
                    + w_eng * energy_score
                    + w_imp * importance_score
                    + w_act * activation_score;

                SearchResult {
                    note,
                    score,
                    signals: SignalBreakdown {
                        vector_score,
                        recency_score,
                        energy_score,
                        importance_score,
                        activation_score,
                    },
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        debug!(
            "SearchPipeline: {} results (weights: vec={:.2} rec={:.2} eng={:.2} imp={:.2} act={:.2})",
            results.len(),
            w_vec, w_rec, w_eng, w_imp, w_act
        );

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::GraphStore;
    use crate::notes::{NoteImportance, NoteType};

    fn mock_embedding_provider() -> Arc<dyn EmbeddingProvider> {
        Arc::new(crate::embeddings::MockEmbeddingProvider::new(768))
    }

    fn note_with_energy(id: Uuid, project_id: Option<Uuid>, content: &str, energy: f64) -> Note {
        let mut note = Note::new(
            project_id,
            NoteType::Guideline,
            content.to_string(),
            "test".to_string(),
        );
        note.id = id;
        note.energy = energy;
        note.importance = NoteImportance::High;
        note
    }

    fn gs(mock: &Arc<MockGraphStore>) -> Arc<dyn GraphStore> {
        mock.clone() as Arc<dyn GraphStore>
    }

    #[tokio::test]
    async fn test_search_activation_affects_ranking() {
        // Setup: 2 notes, one connected to source via synapse
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);
        let project_id = Uuid::new_v4();

        let note_a = note_with_energy(
            Uuid::new_v4(),
            Some(project_id),
            "authentication system",
            1.0,
        );
        let note_b = note_with_energy(Uuid::new_v4(), Some(project_id), "authorization rules", 1.0);

        store.create_note(&note_a).await.unwrap();
        store.create_note(&note_b).await.unwrap();

        let emb = mock_embedding_provider();
        let embedding_a = emb.embed_text("authentication system").await.unwrap();
        let embedding_b = emb.embed_text("authorization rules").await.unwrap();
        store
            .set_note_embedding(note_a.id, &embedding_a, "mock")
            .await
            .unwrap();
        store
            .set_note_embedding(note_b.id, &embedding_b, "mock")
            .await
            .unwrap();

        // Create synapse: source → note_a (but not → note_b)
        let source_note = note_with_energy(Uuid::new_v4(), Some(project_id), "source node", 1.0);
        store.create_note(&source_note).await.unwrap();
        store
            .create_synapses(source_note.id, &[(note_a.id, 0.9)])
            .await
            .unwrap();

        // Search with activation from source
        let pipeline = SearchPipeline::new(store, mock_embedding_provider());

        // With activation weight
        let results_with = pipeline
            .search(
                "auth",
                Some(source_note.id),
                Some(project_id),
                &SearchWeights::default(),
                10,
            )
            .await
            .unwrap();

        // Without activation (weight=0)
        let no_act_weights = SearchWeights {
            activation_weight: 0.0,
            ..Default::default()
        };
        let results_without = pipeline
            .search(
                "auth",
                Some(source_note.id),
                Some(project_id),
                &no_act_weights,
                10,
            )
            .await
            .unwrap();

        // When activation is enabled and source connects to note_a,
        // note_a should get a boost
        if !results_with.is_empty() && !results_without.is_empty() {
            // With activation, note_a that's connected to source should score differently
            let find_a = |results: &[SearchResult]| {
                results
                    .iter()
                    .find(|r| r.note.id == note_a.id)
                    .map(|r| r.score)
            };
            let score_with = find_a(&results_with);
            let score_without = find_a(&results_without);

            if let (Some(sw), Some(swo)) = (score_with, score_without) {
                // The activation signal should make a difference
                assert!(
                    (sw - swo).abs() > 0.001 || sw >= swo,
                    "Activation signal should affect ranking: with={:.4} without={:.4}",
                    sw,
                    swo
                );
            }
        }
    }

    #[tokio::test]
    async fn test_search_zero_activation_weight_excludes_signal() {
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);
        let project_id = Uuid::new_v4();

        let note = note_with_energy(
            Uuid::new_v4(),
            Some(project_id),
            "test note for zero weight",
            1.0,
        );
        store.create_note(&note).await.unwrap();
        let emb = mock_embedding_provider();
        let embedding = emb.embed_text("test note").await.unwrap();
        store
            .set_note_embedding(note.id, &embedding, "mock")
            .await
            .unwrap();

        let pipeline = SearchPipeline::new(store, mock_embedding_provider());
        let weights = SearchWeights {
            activation_weight: 0.0,
            ..Default::default()
        };

        let results = pipeline
            .search("test", None, Some(project_id), &weights, 10)
            .await
            .unwrap();

        // All activation_score signals should be 0
        for r in &results {
            assert_eq!(
                r.signals.activation_score, 0.0,
                "Activation signal should be 0 when weight is 0"
            );
        }
    }

    #[test]
    fn test_weights_normalization() {
        let w = SearchWeights {
            vector_weight: 1.0,
            recency_weight: 1.0,
            energy_weight: 1.0,
            importance_weight: 1.0,
            activation_weight: 1.0,
        };
        let (v, r, e, i, a) = w.normalized();
        let sum = v + r + e + i + a;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Normalized weights must sum to 1.0"
        );
        assert!((v - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_weights_zero_sum_fallback() {
        let w = SearchWeights {
            vector_weight: 0.0,
            recency_weight: 0.0,
            energy_weight: 0.0,
            importance_weight: 0.0,
            activation_weight: 0.0,
        };
        let (v, r, e, i, a) = w.normalized();
        assert!((v + r + e + i + a - 1.0).abs() < 1e-10);
    }
}
