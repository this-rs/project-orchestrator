//! Spreading Activation Engine
//!
//! Implements a 3-phase neural retrieval algorithm for knowledge notes:
//! 1. **Initial activation**: vector search → seed notes with cosine similarity scores
//! 2. **Spreading**: propagate activation through SYNAPSE edges (weighted, energy-gated)
//! 3. **Ranking**: merge direct + propagated activations, deduplicate, sort by score
//!
//! This is an **independent** retrieval mechanism that coexists with the existing
//! PageRank-weighted propagation in `get_context_notes`. Both systems run in
//! parallel during Phase 3 (dual-run comparison).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::debug;
use uuid::Uuid;

use crate::embeddings::EmbeddingProvider;
use crate::neo4j::GraphStore;
use crate::notes::Note;

use super::config::SpreadingActivationConfig;

// ============================================================================
// Result types
// ============================================================================

/// How a note was activated (direct vector match or propagated through synapses).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ActivationSource {
    /// Activated by direct vector similarity match.
    Direct,
    /// Activated by spreading through synapses from another note.
    Propagated {
        /// ID of the note that propagated the activation.
        via: Uuid,
        /// Number of hops from the nearest direct-match ancestor.
        hops: usize,
    },
}

/// A note that was activated by the spreading activation algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivatedNote {
    /// The note itself.
    pub note: Note,
    /// Final activation score (0.0 - 1.0+).
    /// For direct matches this is the cosine similarity.
    /// For propagated notes this is: parent_activation × synapse_weight × energy × decay^hops.
    pub activation_score: f64,
    /// How this note was activated.
    pub source: ActivationSource,
}

// ============================================================================
// Engine
// ============================================================================

/// Spreading Activation Engine.
///
/// Performs neural-style retrieval over the knowledge graph:
/// embed query → vector search → spread through synapses → rank results.
///
/// Designed for injection via `Arc<SpreadingActivationEngine>`.
pub struct SpreadingActivationEngine {
    graph_store: Arc<dyn GraphStore>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
}

impl SpreadingActivationEngine {
    /// Create a new engine with the given dependencies.
    pub fn new(
        graph_store: Arc<dyn GraphStore>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self {
            graph_store,
            embedding_provider,
        }
    }

    /// Run the full spreading activation algorithm for a query.
    ///
    /// # Arguments
    /// * `query` — natural language query to search for
    /// * `project_id` — optional project filter for data isolation
    /// * `config` — algorithm parameters (use `Default::default()` for sensible defaults)
    ///
    /// # Returns
    /// A list of activated notes sorted by activation score (descending),
    /// limited to `config.max_results`.
    pub async fn activate(
        &self,
        query: &str,
        project_id: Option<Uuid>,
        config: &SpreadingActivationConfig,
    ) -> Result<Vec<ActivatedNote>> {
        // Phase 1: Initial activation via vector search
        let embedding = self.embedding_provider.embed_text(query).await?;
        let seed_notes = self
            .graph_store
            .vector_search_notes(&embedding, config.initial_k, project_id, None)
            .await?;

        debug!(
            "Phase 1: vector search returned {} seed notes for query '{}'",
            seed_notes.len(),
            query
        );

        if seed_notes.is_empty() {
            return Ok(vec![]);
        }

        // Build activation map: note_id → (score, source, Note)
        let mut activations: HashMap<Uuid, (f64, ActivationSource, Note)> = HashMap::new();

        for (note, score) in &seed_notes {
            // Skip dead neurons
            if note.energy < config.min_energy {
                continue;
            }
            activations.insert(
                note.id,
                (*score, ActivationSource::Direct, note.clone()),
            );
        }

        // Phase 2: Spreading through synapses
        // BFS-style: process hop by hop
        let mut frontier: Vec<(Uuid, f64)> = activations
            .iter()
            .map(|(id, (score, _, _))| (*id, *score))
            .collect();

        let mut visited: HashSet<Uuid> = activations.keys().copied().collect();

        for hop in 0..config.max_hops {
            let mut next_frontier: Vec<(Uuid, f64)> = Vec::new();

            for (note_id, parent_activation) in &frontier {
                // Get synapses for this note
                let synapses = match self.graph_store.get_synapses(*note_id).await {
                    Ok(s) => s,
                    Err(e) => {
                        debug!("Failed to get synapses for {}: {}", note_id, e);
                        continue;
                    }
                };

                for (neighbor_id, synapse_weight) in synapses {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }

                    // Get neighbor note to check energy
                    let neighbor = match self.graph_store.get_note(neighbor_id).await {
                        Ok(Some(n)) => n,
                        _ => continue,
                    };

                    // Skip dead neurons
                    if neighbor.energy < config.min_energy {
                        continue;
                    }

                    // Calculate spread score
                    let spread_score = parent_activation
                        * synapse_weight
                        * neighbor.energy
                        * config.decay_per_hop;

                    // Skip if below threshold
                    if spread_score < config.min_activation {
                        continue;
                    }

                    // Insert or update with max score
                    let existing = activations.get(&neighbor_id);
                    let should_insert = match existing {
                        None => true,
                        Some((existing_score, _, _)) => spread_score > *existing_score,
                    };

                    if should_insert {
                        activations.insert(
                            neighbor_id,
                            (
                                spread_score,
                                ActivationSource::Propagated {
                                    via: *note_id,
                                    hops: hop + 1,
                                },
                                neighbor,
                            ),
                        );
                    }

                    visited.insert(neighbor_id);
                    next_frontier.push((neighbor_id, spread_score));
                }
            }

            debug!(
                "Phase 2: hop {} spread to {} new notes",
                hop + 1,
                next_frontier.len()
            );

            if next_frontier.is_empty() {
                break;
            }

            frontier = next_frontier;
        }

        // Phase 3: Ranking — sort by score desc, limit to max_results
        let mut results: Vec<ActivatedNote> = activations
            .into_values()
            .map(|(score, source, note)| ActivatedNote {
                note,
                activation_score: score,
                source,
            })
            .collect();

        results.sort_by(|a, b| {
            b.activation_score
                .partial_cmp(&a.activation_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(config.max_results);

        debug!(
            "Phase 3: returning {} activated notes (top score: {:.3})",
            results.len(),
            results.first().map(|r| r.activation_score).unwrap_or(0.0)
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

    /// Helper: create a mock provider that returns a fixed embedding
    fn mock_embedding_provider() -> Arc<dyn EmbeddingProvider> {
        Arc::new(crate::embeddings::MockEmbeddingProvider::new(768))
    }

    /// Helper: create a note with specific energy
    fn note_with_energy(
        id: Uuid,
        project_id: Option<Uuid>,
        content: &str,
        energy: f64,
    ) -> Note {
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

    /// Helper: get a trait-object reference for GraphStore methods.
    /// Needed because async_trait + Arc<ConcreteType> can't resolve method dispatch.
    fn gs(mock: &Arc<MockGraphStore>) -> Arc<dyn GraphStore> {
        mock.clone() as Arc<dyn GraphStore>
    }

    #[tokio::test]
    async fn test_empty_query_no_matches() {
        let mock = Arc::new(MockGraphStore::new());
        let engine = SpreadingActivationEngine::new(gs(&mock), mock_embedding_provider());

        let results = engine
            .activate("nonexistent topic", None, &SpreadingActivationConfig::default())
            .await
            .unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_direct_match_without_synapses() {
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);
        let project_id = Uuid::new_v4();

        let note_a = note_with_energy(Uuid::new_v4(), Some(project_id), "authentication system", 1.0);
        store.create_note(&note_a).await.unwrap();

        let embedding = mock_embedding_provider().embed_text("authentication system").await.unwrap();
        store.set_note_embedding(note_a.id, &embedding, "mock").await.unwrap();

        let engine = SpreadingActivationEngine::new(store, mock_embedding_provider());

        let results = engine
            .activate(
                "authentication system",
                Some(project_id),
                &SpreadingActivationConfig::default(),
            )
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert!(matches!(results[0].source, ActivationSource::Direct));
        assert!(results[0].activation_score > 0.0);
    }

    #[tokio::test]
    async fn test_spreading_through_synapses() {
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);
        let project_id = Uuid::new_v4();

        let note_a = note_with_energy(Uuid::new_v4(), Some(project_id), "authentication login", 1.0);
        let note_b = note_with_energy(Uuid::new_v4(), Some(project_id), "session management", 0.8);
        let note_c = note_with_energy(Uuid::new_v4(), Some(project_id), "token validation", 0.9);

        store.create_note(&note_a).await.unwrap();
        store.create_note(&note_b).await.unwrap();
        store.create_note(&note_c).await.unwrap();

        let embedding = mock_embedding_provider().embed_text("authentication login").await.unwrap();
        store.set_note_embedding(note_a.id, &embedding, "mock").await.unwrap();

        // Create synapses: A ↔ B (0.9), B ↔ C (0.8)
        store.create_synapses(note_a.id, &[(note_b.id, 0.9)]).await.unwrap();
        store.create_synapses(note_b.id, &[(note_c.id, 0.8)]).await.unwrap();

        let config = SpreadingActivationConfig {
            max_hops: 2,
            min_activation: 0.01, // Low threshold to catch propagated notes
            ..Default::default()
        };

        let engine = SpreadingActivationEngine::new(store, mock_embedding_provider());

        let results = engine
            .activate("authentication login", Some(project_id), &config)
            .await
            .unwrap();

        // Should have A (direct), B (1 hop), possibly C (2 hops)
        assert!(results.len() >= 2, "Expected at least 2 results, got {}", results.len());

        // A should be direct
        let a_result = results.iter().find(|r| r.note.id == note_a.id);
        assert!(a_result.is_some());
        assert!(matches!(a_result.unwrap().source, ActivationSource::Direct));

        // B should be propagated
        let b_result = results.iter().find(|r| r.note.id == note_b.id);
        assert!(b_result.is_some());
        assert!(matches!(
            b_result.unwrap().source,
            ActivationSource::Propagated { hops: 1, .. }
        ));
    }

    #[tokio::test]
    async fn test_dead_neuron_excluded() {
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);
        let project_id = Uuid::new_v4();

        let note_a = note_with_energy(Uuid::new_v4(), Some(project_id), "alive note", 1.0);
        let note_b = note_with_energy(Uuid::new_v4(), Some(project_id), "dead note", 0.0);

        store.create_note(&note_a).await.unwrap();
        store.create_note(&note_b).await.unwrap();

        let embedding = mock_embedding_provider().embed_text("alive note").await.unwrap();
        store.set_note_embedding(note_a.id, &embedding, "mock").await.unwrap();

        // Synapse A → B
        store.create_synapses(note_a.id, &[(note_b.id, 0.95)]).await.unwrap();

        let engine = SpreadingActivationEngine::new(store, mock_embedding_provider());

        let results = engine
            .activate("alive note", Some(project_id), &SpreadingActivationConfig::default())
            .await
            .unwrap();

        // B should NOT appear (energy 0.0 < min_energy 0.05)
        let b_result = results.iter().find(|r| r.note.id == note_b.id);
        assert!(b_result.is_none(), "Dead neuron should not appear in results");
    }

    #[tokio::test]
    async fn test_cycle_no_infinite_loop() {
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);
        let project_id = Uuid::new_v4();

        // A ↔ B (mutual synapses — cycle)
        let note_a = note_with_energy(Uuid::new_v4(), Some(project_id), "cycle A", 1.0);
        let note_b = note_with_energy(Uuid::new_v4(), Some(project_id), "cycle B", 1.0);

        store.create_note(&note_a).await.unwrap();
        store.create_note(&note_b).await.unwrap();

        let embedding = mock_embedding_provider().embed_text("cycle A").await.unwrap();
        store.set_note_embedding(note_a.id, &embedding, "mock").await.unwrap();

        // Bidirectional synapse (already created bidirectionally by create_synapses)
        store.create_synapses(note_a.id, &[(note_b.id, 0.9)]).await.unwrap();

        let engine = SpreadingActivationEngine::new(store, mock_embedding_provider());

        // This should NOT hang — visited set prevents revisiting
        let results = engine
            .activate(
                "cycle A",
                Some(project_id),
                &SpreadingActivationConfig {
                    max_hops: 5, // High hops to stress-test cycle handling
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Should complete without hanging
        assert!(!results.is_empty());
        // Both A and B should appear
        assert!(results.iter().any(|r| r.note.id == note_a.id));
        assert!(results.iter().any(|r| r.note.id == note_b.id));
    }

    #[tokio::test]
    async fn test_max_hops_respected() {
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);
        let project_id = Uuid::new_v4();

        // Chain: A → B → C → D (3 hops from A to D)
        let note_a = note_with_energy(Uuid::new_v4(), Some(project_id), "hop source", 1.0);
        let note_b = note_with_energy(Uuid::new_v4(), Some(project_id), "hop 1", 1.0);
        let note_c = note_with_energy(Uuid::new_v4(), Some(project_id), "hop 2", 1.0);
        let note_d = note_with_energy(Uuid::new_v4(), Some(project_id), "hop 3", 1.0);

        store.create_note(&note_a).await.unwrap();
        store.create_note(&note_b).await.unwrap();
        store.create_note(&note_c).await.unwrap();
        store.create_note(&note_d).await.unwrap();

        let embedding = mock_embedding_provider().embed_text("hop source").await.unwrap();
        store.set_note_embedding(note_a.id, &embedding, "mock").await.unwrap();

        store.create_synapses(note_a.id, &[(note_b.id, 0.9)]).await.unwrap();
        store.create_synapses(note_b.id, &[(note_c.id, 0.9)]).await.unwrap();
        store.create_synapses(note_c.id, &[(note_d.id, 0.9)]).await.unwrap();

        let config = SpreadingActivationConfig {
            max_hops: 2, // Only 2 hops allowed
            min_activation: 0.01,
            ..Default::default()
        };

        let engine = SpreadingActivationEngine::new(store, mock_embedding_provider());

        let results = engine
            .activate("hop source", Some(project_id), &config)
            .await
            .unwrap();

        // D is at 3 hops — should NOT appear
        let d_result = results.iter().find(|r| r.note.id == note_d.id);
        assert!(d_result.is_none(), "Note at 3 hops should be excluded with max_hops=2");

        // A (direct), B (1 hop), C (2 hops) should appear
        assert!(results.iter().any(|r| r.note.id == note_a.id));
        assert!(results.iter().any(|r| r.note.id == note_b.id));
        assert!(results.iter().any(|r| r.note.id == note_c.id));
    }

    #[tokio::test]
    async fn test_energy_influences_ranking() {
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);
        let project_id = Uuid::new_v4();

        // A is seed. B has high energy, C has low energy. Both at 1 hop.
        let note_a = note_with_energy(Uuid::new_v4(), Some(project_id), "source", 1.0);
        let note_b = note_with_energy(Uuid::new_v4(), Some(project_id), "high energy neighbor", 1.0);
        let note_c = note_with_energy(Uuid::new_v4(), Some(project_id), "low energy neighbor", 0.1);

        store.create_note(&note_a).await.unwrap();
        store.create_note(&note_b).await.unwrap();
        store.create_note(&note_c).await.unwrap();

        let embedding = mock_embedding_provider().embed_text("source").await.unwrap();
        store.set_note_embedding(note_a.id, &embedding, "mock").await.unwrap();

        // Same synapse weight for both
        store.create_synapses(note_a.id, &[(note_b.id, 0.9), (note_c.id, 0.9)]).await.unwrap();

        let config = SpreadingActivationConfig {
            min_activation: 0.01,
            ..Default::default()
        };

        let engine = SpreadingActivationEngine::new(store, mock_embedding_provider());

        let results = engine
            .activate("source", Some(project_id), &config)
            .await
            .unwrap();

        let b_result = results.iter().find(|r| r.note.id == note_b.id);
        let c_result = results.iter().find(|r| r.note.id == note_c.id);

        assert!(b_result.is_some() && c_result.is_some());
        assert!(
            b_result.unwrap().activation_score > c_result.unwrap().activation_score,
            "Higher energy note should have higher activation score"
        );
    }
}
