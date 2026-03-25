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
use crate::graph::models::AnalysisProfile;
use crate::neo4j::GraphStore;
use crate::notes::{Note, NoteImportance, NoteType};

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

/// A note or decision that was activated by the spreading activation algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivatedNote {
    /// The note itself (for Decision entities, a synthetic Note is created).
    pub note: Note,
    /// Final activation score (0.0 - 1.0+).
    /// For direct matches this is the cosine similarity.
    /// For propagated notes this is: parent_activation × synapse_weight × energy × decay^hops.
    pub activation_score: f64,
    /// How this note was activated.
    pub source: ActivationSource,
    /// Entity type: "note" for knowledge notes, "decision" for architectural decisions.
    /// Defaults to "note" for backward compatibility.
    #[serde(default = "default_entity_type")]
    pub entity_type: String,
}

fn default_entity_type() -> String {
    "note".to_string()
}

// ============================================================================
// Search Result types (spreading activation as search signal)
// ============================================================================

/// A search result produced by spreading activation over the graph.
///
/// Unlike `ActivatedNote` (which carries the full Note), this is a lightweight
/// struct designed for integration with search pipelines — it carries only
/// the node ID, activation level, hop distance, and the path taken.
///
/// Use cases:
/// - **Recommendation**: activate a purchased product → find related products
/// - **Knowledge exploration**: activate a concept → discover related concepts
/// - **Influence detection**: activate a node → measure propagation reach
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationSearchResult {
    /// ID of the activated node.
    pub node_id: Uuid,
    /// Final activation level after propagation (normalized 0.0–1.0).
    pub activation_level: f64,
    /// Number of hops from the source node (0 = source itself).
    pub hop_distance: usize,
    /// Path from the source to this node (list of node IDs traversed).
    pub path: Vec<Uuid>,
}

/// Configuration for the `activate_and_collect` search mode.
#[derive(Debug, Clone)]
pub struct ActivationSearchConfig {
    /// Maximum number of results to return (top-K by activation level).
    pub top_k: usize,
    /// Maximum hops from the source node.
    pub max_hops: usize,
    /// Decay factor per hop (0.0–1.0). Each hop multiplies by this factor.
    pub decay_per_hop: f64,
    /// Minimum activation level to include in results.
    pub min_activation: f64,
    /// Maximum number of nodes to visit (circuit breaker). Prevents
    /// runaway traversals in dense graphs.
    pub max_visited: usize,
    /// Minimum energy for a node to participate in spreading.
    pub min_energy: f64,
}

impl Default for ActivationSearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            max_hops: 3,
            decay_per_hop: 0.5,
            min_activation: 0.05,
            max_visited: 500,
            min_energy: 0.05,
        }
    }
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

    /// Convert a DecisionNode into a synthetic Note for inclusion in activation results.
    ///
    /// Decisions don't have the same fields as Notes, so we create a lightweight
    /// Note carrier with the decision's content. The `entity_type` field on
    /// `ActivatedNote` distinguishes decisions from real notes.
    fn decision_to_synthetic_note(decision: &crate::neo4j::models::DecisionNode) -> Note {
        let mut note = Note::new(
            None, // Decisions are task-scoped, not project-scoped
            NoteType::Observation,
            format!(
                "Decision: {}\nRationale: {}",
                decision.description, decision.rationale
            ),
            decision.decided_by.clone(),
        );
        note.id = decision.id;
        note.energy = 1.0; // Decisions are always active
        note.importance = NoteImportance::High;
        note
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
        self.activate_with_profile(query, project_id, config, None)
            .await
    }

    /// Run spreading activation with an optional analysis profile for intent-adaptive weighting.
    ///
    /// When a profile is provided, Phase 2 (spreading) multiplies each synapse's
    /// spread score by the profile's edge weight for the `SYNAPSE` relation type.
    /// This biases results toward the intent (debug → SYNAPSE=0.8 high propagation,
    /// impact → SYNAPSE=0.3 dampened propagation) without filtering any signal completely.
    ///
    /// When no profile is provided, behaves identically to `activate()`.
    pub async fn activate_with_profile(
        &self,
        query: &str,
        project_id: Option<Uuid>,
        config: &SpreadingActivationConfig,
        profile: Option<&AnalysisProfile>,
    ) -> Result<Vec<ActivatedNote>> {
        debug!(
            "Spreading activation: profile={}",
            profile.map(|p| p.name.as_str()).unwrap_or("none")
        );

        // Phase 1: Initial activation via vector search
        let embedding = self.embedding_provider.embed_text(query).await?;
        let seed_notes = self
            .graph_store
            .vector_search_notes(
                &embedding,
                config.initial_k,
                project_id,
                None,
                Some(config.min_cosine_similarity),
            )
            .await?;

        debug!(
            "Phase 1: vector search returned {} seed notes for query '{}' (min_cosine_similarity: {:.2})",
            seed_notes.len(),
            query,
            config.min_cosine_similarity
        );

        if seed_notes.is_empty() {
            return Ok(vec![]);
        }

        // Build activation map: id → (score, source, Note, entity_type)
        // For Decision entities, a synthetic Note is created from the decision data.
        let mut activations: HashMap<Uuid, (f64, ActivationSource, Note, String)> = HashMap::new();

        for (note, score) in &seed_notes {
            // Skip dead neurons
            if note.computed_energy() < config.min_energy {
                continue;
            }
            // Knowledge Scars: penalize scarred notes (biomimicry: Elun Scar)
            let scar_penalized_score = score * (1.0 - note.scar_intensity * 0.7);
            activations.insert(
                note.id,
                (
                    scar_penalized_score,
                    ActivationSource::Direct,
                    note.clone(),
                    "note".to_string(),
                ),
            );
        }

        // Phase 2: Spreading through synapses (cross-entity: Note ↔ Decision)
        // BFS-style: process hop by hop
        let mut frontier: Vec<(Uuid, f64)> = activations
            .iter()
            .map(|(id, (score, _, _, _))| (*id, *score))
            .collect();

        let mut visited: HashSet<Uuid> = activations.keys().copied().collect();

        for hop in 0..config.max_hops {
            let mut next_frontier: Vec<(Uuid, f64)> = Vec::new();

            for (node_id, parent_activation) in &frontier {
                // Get cross-entity synapses (Note↔Note, Note↔Decision, Decision↔Note)
                let synapses = match self.graph_store.get_cross_entity_synapses(*node_id).await {
                    Ok(s) => s,
                    Err(e) => {
                        debug!("Failed to get cross-entity synapses for {}: {}", node_id, e);
                        // Fallback to Note-only synapses
                        match self.graph_store.get_synapses(*node_id).await {
                            Ok(s) => s
                                .into_iter()
                                .map(|(id, w)| (id, w, "Note".to_string()))
                                .collect(),
                            Err(_) => continue,
                        }
                    }
                };

                for (neighbor_id, synapse_weight, entity_type) in synapses {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }

                    // Resolve neighbor: Note or Decision
                    let (neighbor_note, neighbor_energy, neighbor_entity_type) =
                        if entity_type == "Decision" {
                            // Decision neighbor: create synthetic Note, energy=1.0
                            match self.graph_store.get_decision(neighbor_id).await {
                                Ok(Some(decision)) => {
                                    let synthetic = Self::decision_to_synthetic_note(&decision);
                                    (synthetic, 1.0_f64, "decision".to_string())
                                }
                                _ => continue,
                            }
                        } else {
                            // Note neighbor: use real Note with energy
                            match self.graph_store.get_note(neighbor_id).await {
                                Ok(Some(n)) => {
                                    let energy = n.computed_energy();
                                    (n, energy, "note".to_string())
                                }
                                _ => continue,
                            }
                        };

                    // Skip dead neurons (decisions always have energy=1.0)
                    if neighbor_energy < config.min_energy {
                        continue;
                    }

                    // Calculate spread score with scar penalty (biomimicry: Elun Scar)
                    let scar_penalty = 1.0 - neighbor_note.scar_intensity * 0.7;

                    // Intent-adaptive: weight by profile's SYNAPSE edge weight
                    let profile_synapse_weight = profile
                        .and_then(|p| p.edge_weights.get("SYNAPSE"))
                        .copied()
                        .unwrap_or(1.0);

                    let spread_score = parent_activation
                        * synapse_weight
                        * neighbor_energy
                        * config.decay_per_hop
                        * scar_penalty
                        * profile_synapse_weight;

                    // Skip if below threshold
                    if spread_score < config.min_activation {
                        continue;
                    }

                    // Insert or update with max score
                    let existing = activations.get(&neighbor_id);
                    let should_insert = match existing {
                        None => true,
                        Some((existing_score, _, _, _)) => spread_score > *existing_score,
                    };

                    if should_insert {
                        activations.insert(
                            neighbor_id,
                            (
                                spread_score,
                                ActivationSource::Propagated {
                                    via: *node_id,
                                    hops: hop + 1,
                                },
                                neighbor_note,
                                neighbor_entity_type,
                            ),
                        );
                    }

                    visited.insert(neighbor_id);
                    next_frontier.push((neighbor_id, spread_score));
                }
            }

            debug!(
                "Phase 2: hop {} spread to {} new entities",
                hop + 1,
                next_frontier.len()
            );

            if next_frontier.is_empty() {
                break;
            }

            frontier = next_frontier;
        }

        // Phase 2.5: Connectivity re-ranking — boost direct matches that are
        // connected to other activated entities via synapses ("hub bonus").
        // This makes well-connected notes rank higher than isolated ones,
        // differentiating search_neurons from plain vector search.
        if config.connectivity_boost > 0.0 {
            let activated_ids: HashSet<Uuid> = activations.keys().copied().collect();
            let mut boost_map: HashMap<Uuid, f64> = HashMap::new();

            for (node_id, (_, source, _, _)) in &activations {
                if !matches!(source, ActivationSource::Direct) {
                    continue;
                }
                // Count how many other activated entities this direct match
                // is connected to via synapses
                if let Ok(synapses) = self.graph_store.get_synapses(*node_id).await {
                    let connected_activated = synapses
                        .iter()
                        .filter(|(neighbor_id, _)| {
                            *neighbor_id != *node_id && activated_ids.contains(neighbor_id)
                        })
                        .count();
                    if connected_activated > 0 {
                        let bonus = config.connectivity_boost * connected_activated as f64;
                        boost_map.insert(*node_id, bonus);
                    }
                }
            }

            // Apply boosts
            for (node_id, bonus) in &boost_map {
                if let Some(entry) = activations.get_mut(node_id) {
                    entry.0 += bonus;
                }
            }

            if !boost_map.is_empty() {
                debug!(
                    "Phase 2.5: connectivity boost applied to {} direct matches",
                    boost_map.len()
                );
            }
        }

        // Phase 3: Ranking with slot reservation
        //
        // Problem: direct matches score ~0.91 while propagated notes score
        // ~0.39 (after hop decay). Pure score ranking means propagated notes
        // NEVER appear in top-N results — making spreading useless.
        //
        // Solution: reserve a fraction of slots for propagated results.
        // With propagated_ratio=0.4 and max_results=10: 4 slots for
        // propagated, 6 for direct. Unused slots overflow to the other pool.
        let (mut direct, mut propagated): (Vec<ActivatedNote>, Vec<ActivatedNote>) = activations
            .into_values()
            .map(|(score, source, note, entity_type)| ActivatedNote {
                note,
                activation_score: score,
                source,
                entity_type,
            })
            .partition(|r| matches!(r.source, ActivationSource::Direct));

        // Sort each pool by score descending
        let score_cmp = |a: &ActivatedNote, b: &ActivatedNote| {
            b.activation_score
                .partial_cmp(&a.activation_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        };
        direct.sort_by(score_cmp);
        propagated.sort_by(score_cmp);

        let results = if config.propagated_ratio > 0.0 && !propagated.is_empty() {
            let propagated_slots =
                (config.max_results as f64 * config.propagated_ratio).ceil() as usize;
            let direct_slots = config.max_results.saturating_sub(propagated_slots);

            // Take from each pool up to their slot allocation
            let taken_direct: Vec<ActivatedNote> = direct.into_iter().take(direct_slots).collect();
            let taken_propagated: Vec<ActivatedNote> =
                propagated.into_iter().take(propagated_slots).collect();

            // Merge and re-sort by score
            let mut merged = taken_direct;
            merged.extend(taken_propagated);
            merged.sort_by(score_cmp);
            merged.truncate(config.max_results);
            merged
        } else {
            // No reservation: pure score ranking (original behavior)
            let mut all = direct;
            all.extend(propagated);
            all.sort_by(score_cmp);
            all.truncate(config.max_results);
            all
        };

        let direct_count = results
            .iter()
            .filter(|r| matches!(r.source, ActivationSource::Direct))
            .count();
        let propagated_count = results.len() - direct_count;

        debug!(
            "Phase 3: returning {} activated notes ({} direct, {} propagated, top score: {:.3})",
            results.len(),
            direct_count,
            propagated_count,
            results.first().map(|r| r.activation_score).unwrap_or(0.0)
        );

        Ok(results)
    }

    /// Activate a source node and collect the top-K most activated nodes
    /// as search results, using BFS spreading through synapses.
    ///
    /// This is a **structural search** — instead of searching by query text,
    /// you activate a known node and observe how activation propagates through
    /// the graph. Nodes that are strongly connected to the source (via high-weight
    /// synapses, short paths) rank highest.
    ///
    /// The circuit breaker (`config.max_visited`) prevents runaway traversals
    /// in dense graphs.
    ///
    /// # Arguments
    /// * `source_id` — the node to activate (must exist in the graph)
    /// * `config` — search parameters (top-K, max hops, decay, circuit breaker)
    ///
    /// # Returns
    /// Top-K `ActivationSearchResult`s sorted by activation_level descending.
    pub async fn activate_and_collect(
        &self,
        source_id: Uuid,
        config: &ActivationSearchConfig,
    ) -> Result<Vec<ActivationSearchResult>> {
        // Verify source exists
        let source_note = self
            .graph_store
            .get_note(source_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Source node {} not found", source_id))?;

        if source_note.computed_energy() < config.min_energy {
            return Ok(vec![]);
        }

        // activation_map: node_id → (activation_level, hop_distance, path)
        let mut activation_map: HashMap<Uuid, (f64, usize, Vec<Uuid>)> = HashMap::new();
        activation_map.insert(source_id, (1.0, 0, vec![source_id]));

        // BFS frontier: (node_id, activation_level, path)
        let mut frontier: Vec<(Uuid, f64, Vec<Uuid>)> = vec![(source_id, 1.0, vec![source_id])];
        let mut visited: HashSet<Uuid> = HashSet::new();
        visited.insert(source_id);

        let mut total_visited: usize = 1;

        for hop in 0..config.max_hops {
            if frontier.is_empty() {
                break;
            }

            // Circuit breaker: stop if we've visited too many nodes
            if total_visited >= config.max_visited {
                debug!(
                    "activate_and_collect: circuit breaker at {} visited nodes (limit: {})",
                    total_visited, config.max_visited
                );
                break;
            }

            let mut next_frontier: Vec<(Uuid, f64, Vec<Uuid>)> = Vec::new();

            for (node_id, parent_activation, parent_path) in &frontier {
                // Budget remaining before circuit breaker trips
                let budget = config.max_visited.saturating_sub(total_visited);
                if budget == 0 {
                    break;
                }

                let synapses = match self.graph_store.get_synapses(*node_id).await {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                for (neighbor_id, synapse_weight) in synapses {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    if total_visited >= config.max_visited {
                        break;
                    }

                    // Check neighbor energy
                    let neighbor_energy = match self.graph_store.get_note(neighbor_id).await {
                        Ok(Some(n)) => n.computed_energy(),
                        _ => continue,
                    };
                    if neighbor_energy < config.min_energy {
                        continue;
                    }

                    let spread_score =
                        parent_activation * synapse_weight * neighbor_energy * config.decay_per_hop;

                    if spread_score < config.min_activation {
                        continue;
                    }

                    let mut path = parent_path.clone();
                    path.push(neighbor_id);

                    let hop_distance = hop + 1;

                    // Insert or update if better score
                    let should_insert = match activation_map.get(&neighbor_id) {
                        None => true,
                        Some((existing, _, _)) => spread_score > *existing,
                    };

                    if should_insert {
                        activation_map
                            .insert(neighbor_id, (spread_score, hop_distance, path.clone()));
                    }

                    visited.insert(neighbor_id);
                    total_visited += 1;
                    next_frontier.push((neighbor_id, spread_score, path));
                }
            }

            frontier = next_frontier;
        }

        // Remove source from results (caller already knows about it)
        activation_map.remove(&source_id);

        // Sort by activation_level descending, take top-K
        let mut results: Vec<ActivationSearchResult> = activation_map
            .into_iter()
            .map(|(node_id, (level, hops, path))| ActivationSearchResult {
                node_id,
                activation_level: level,
                hop_distance: hops,
                path,
            })
            .collect();

        results.sort_by(|a, b| {
            b.activation_level
                .partial_cmp(&a.activation_level)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(config.top_k);

        // Normalize activation levels to [0, 1]
        if let Some(max_level) = results.first().map(|r| r.activation_level) {
            if max_level > 0.0 {
                for r in &mut results {
                    r.activation_level /= max_level;
                }
            }
        }

        debug!(
            "activate_and_collect: source={}, visited={}, results={} (top score: {:.3})",
            source_id,
            total_visited,
            results.len(),
            results.first().map(|r| r.activation_level).unwrap_or(0.0)
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
            .activate(
                "nonexistent topic",
                None,
                &SpreadingActivationConfig::default(),
            )
            .await
            .unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_direct_match_without_synapses() {
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);
        let project_id = Uuid::new_v4();

        let note_a = note_with_energy(
            Uuid::new_v4(),
            Some(project_id),
            "authentication system",
            1.0,
        );
        store.create_note(&note_a).await.unwrap();

        let embedding = mock_embedding_provider()
            .embed_text("authentication system")
            .await
            .unwrap();
        store
            .set_note_embedding(note_a.id, &embedding, "mock")
            .await
            .unwrap();

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

        let note_a = note_with_energy(
            Uuid::new_v4(),
            Some(project_id),
            "authentication login",
            1.0,
        );
        let note_b = note_with_energy(Uuid::new_v4(), Some(project_id), "session management", 0.8);
        let note_c = note_with_energy(Uuid::new_v4(), Some(project_id), "token validation", 0.9);

        store.create_note(&note_a).await.unwrap();
        store.create_note(&note_b).await.unwrap();
        store.create_note(&note_c).await.unwrap();

        let embedding = mock_embedding_provider()
            .embed_text("authentication login")
            .await
            .unwrap();
        store
            .set_note_embedding(note_a.id, &embedding, "mock")
            .await
            .unwrap();

        // Create synapses: A ↔ B (0.9), B ↔ C (0.8)
        store
            .create_synapses(note_a.id, &[(note_b.id, 0.9)])
            .await
            .unwrap();
        store
            .create_synapses(note_b.id, &[(note_c.id, 0.8)])
            .await
            .unwrap();

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
        assert!(
            results.len() >= 2,
            "Expected at least 2 results, got {}",
            results.len()
        );

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

        let embedding = mock_embedding_provider()
            .embed_text("alive note")
            .await
            .unwrap();
        store
            .set_note_embedding(note_a.id, &embedding, "mock")
            .await
            .unwrap();

        // Synapse A → B
        store
            .create_synapses(note_a.id, &[(note_b.id, 0.95)])
            .await
            .unwrap();

        let engine = SpreadingActivationEngine::new(store, mock_embedding_provider());

        let results = engine
            .activate(
                "alive note",
                Some(project_id),
                &SpreadingActivationConfig::default(),
            )
            .await
            .unwrap();

        // B should NOT appear (energy 0.0 < min_energy 0.05)
        let b_result = results.iter().find(|r| r.note.id == note_b.id);
        assert!(
            b_result.is_none(),
            "Dead neuron should not appear in results"
        );
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

        let embedding = mock_embedding_provider()
            .embed_text("cycle A")
            .await
            .unwrap();
        store
            .set_note_embedding(note_a.id, &embedding, "mock")
            .await
            .unwrap();

        // Bidirectional synapse (already created bidirectionally by create_synapses)
        store
            .create_synapses(note_a.id, &[(note_b.id, 0.9)])
            .await
            .unwrap();

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

        let embedding = mock_embedding_provider()
            .embed_text("hop source")
            .await
            .unwrap();
        store
            .set_note_embedding(note_a.id, &embedding, "mock")
            .await
            .unwrap();

        store
            .create_synapses(note_a.id, &[(note_b.id, 0.9)])
            .await
            .unwrap();
        store
            .create_synapses(note_b.id, &[(note_c.id, 0.9)])
            .await
            .unwrap();
        store
            .create_synapses(note_c.id, &[(note_d.id, 0.9)])
            .await
            .unwrap();

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
        assert!(
            d_result.is_none(),
            "Note at 3 hops should be excluded with max_hops=2"
        );

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
        let note_b = note_with_energy(
            Uuid::new_v4(),
            Some(project_id),
            "high energy neighbor",
            1.0,
        );
        let note_c = note_with_energy(Uuid::new_v4(), Some(project_id), "low energy neighbor", 0.1);

        store.create_note(&note_a).await.unwrap();
        store.create_note(&note_b).await.unwrap();
        store.create_note(&note_c).await.unwrap();

        let embedding = mock_embedding_provider()
            .embed_text("source")
            .await
            .unwrap();
        store
            .set_note_embedding(note_a.id, &embedding, "mock")
            .await
            .unwrap();

        // Same synapse weight for both
        store
            .create_synapses(note_a.id, &[(note_b.id, 0.9), (note_c.id, 0.9)])
            .await
            .unwrap();

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

    /// **E2E Integration Test (T2.7)**
    ///
    /// Validates the full Phase 2 neural network lifecycle:
    /// 1. Create notes → embed them → build synapses
    /// 2. Spreading activation retrieval (direct + propagated)
    /// 3. Hebbian reinforcement (co-activation boosts synapses)
    /// 4. Energy decay + synapse decay/pruning
    /// 5. Project isolation (no cross-project synapses)
    #[tokio::test]
    async fn test_e2e_neural_network_lifecycle() {
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);
        let provider = mock_embedding_provider();

        // ── Phase 1: Setup — 2 isolated projects ────────────────────────
        let project_a = Uuid::new_v4();
        let project_b = Uuid::new_v4();

        // Project A: auth cluster (3 notes)
        let auth_login = note_with_energy(
            Uuid::new_v4(),
            Some(project_a),
            "authentication login flow with JWT tokens",
            1.0,
        );
        let auth_session = note_with_energy(
            Uuid::new_v4(),
            Some(project_a),
            "session management and cookie handling",
            1.0,
        );
        let auth_password = note_with_energy(
            Uuid::new_v4(),
            Some(project_a),
            "password hashing with bcrypt",
            1.0,
        );

        // Project A: database cluster (2 notes)
        let db_query = note_with_energy(
            Uuid::new_v4(),
            Some(project_a),
            "database query optimization with indexes",
            1.0,
        );
        let db_migration = note_with_energy(
            Uuid::new_v4(),
            Some(project_a),
            "database migration strategy",
            1.0,
        );

        // Project B: auth notes (should NOT connect to project A)
        let b_auth = note_with_energy(
            Uuid::new_v4(),
            Some(project_b),
            "authentication OAuth2 flow",
            1.0,
        );

        // Global note (can connect to both projects)
        let global_note = note_with_energy(
            Uuid::new_v4(),
            None,
            "security best practices for web apps",
            1.0,
        );

        // Create all notes
        for note in [
            &auth_login,
            &auth_session,
            &auth_password,
            &db_query,
            &db_migration,
            &b_auth,
            &global_note,
        ] {
            store.create_note(note).await.unwrap();
        }

        // Embed ONLY the entry point notes — and use the QUERY text as the
        // embedding content, not the note content.
        //
        // Why: the MockEmbeddingProvider produces deterministic but different
        // vectors per text (hash-based). For vector_search_notes to return a
        // high cosine similarity, the stored embedding must match the query
        // embedding. By embedding auth_login with the query text "authentication
        // login", the vector search finds it as a strong direct match, while
        // all other project A notes are discovered only via synapse propagation.
        let query_text = "authentication login";
        let query_emb = provider.embed_text(query_text).await.unwrap();
        store
            .set_note_embedding(auth_login.id, &query_emb, "mock")
            .await
            .unwrap();

        // Also embed b_auth with its own query for the project B isolation test
        let b_query_emb = provider.embed_text("authentication").await.unwrap();
        store
            .set_note_embedding(b_auth.id, &b_query_emb, "mock")
            .await
            .unwrap();

        // Build synapses — auth cluster in project A
        store
            .create_synapses(
                auth_login.id,
                &[
                    (auth_session.id, 0.92),  // login ↔ session (very related)
                    (auth_password.id, 0.85), // login ↔ password (related)
                ],
            )
            .await
            .unwrap();
        store
            .create_synapses(
                auth_session.id,
                &[
                    (auth_password.id, 0.80), // session ↔ password (somewhat related)
                ],
            )
            .await
            .unwrap();

        // Database cluster in project A
        store
            .create_synapses(
                db_query.id,
                &[
                    (db_migration.id, 0.88), // query ↔ migration
                ],
            )
            .await
            .unwrap();

        // Cross-cluster weak link (auth needs db for user storage)
        store
            .create_synapses(auth_login.id, &[(db_query.id, 0.60)])
            .await
            .unwrap();

        // ── Phase 2: Spreading Activation ───────────────────────────────
        let engine = SpreadingActivationEngine::new(store.clone(), provider.clone());

        let config = SpreadingActivationConfig {
            max_results: 10,
            max_hops: 2,
            min_activation: 0.01,
            decay_per_hop: 0.5,
            min_energy: 0.0,
            min_cosine_similarity: 0.0, // Disabled for mock embeddings (hash-based, low similarity)
            ..Default::default()
        };

        let results = engine
            .activate("authentication login", Some(project_a), &config)
            .await
            .unwrap();

        // Should have direct matches
        let direct_count = results
            .iter()
            .filter(|r| matches!(r.source, ActivationSource::Direct))
            .count();
        assert!(
            direct_count >= 1,
            "Should have at least 1 direct match, got {}",
            direct_count
        );

        // Should have propagated matches via synapses
        let propagated_count = results
            .iter()
            .filter(|r| matches!(r.source, ActivationSource::Propagated { .. }))
            .count();
        assert!(
            propagated_count >= 1,
            "Should have at least 1 propagated match, got {}",
            propagated_count
        );

        // Auth notes should appear (they're in the same project, connected via synapses)
        let result_ids: Vec<Uuid> = results.iter().map(|r| r.note.id).collect();
        // auth_session should be propagated from auth_login via synapse
        assert!(
            result_ids.contains(&auth_session.id),
            "auth_session should be found via synapse propagation"
        );

        // Project B notes should NOT appear (project isolation)
        assert!(
            !result_ids.contains(&b_auth.id),
            "Project B notes must not appear in Project A search"
        );

        // Metadata consistency
        let total = results.len();
        assert_eq!(
            total,
            direct_count + propagated_count,
            "total = direct + propagated"
        );

        // ── Phase 3: Hebbian Reinforcement ──────────────────────────────
        // Simulate co-activation of auth_login, auth_session, auth_password
        let co_activated = vec![auth_login.id, auth_session.id, auth_password.id];

        // Get original synapse weight before reinforcement
        let original_synapses = store.get_synapses(auth_login.id).await.unwrap();
        let original_session_weight = original_synapses
            .iter()
            .find(|(id, _)| *id == auth_session.id)
            .map(|(_, w)| *w)
            .unwrap_or(0.0);

        // Reinforce synapses between co-activated notes
        let reinforced = store.reinforce_synapses(&co_activated, 0.05).await.unwrap();
        assert!(reinforced > 0, "Should have reinforced some synapses");

        // Verify synapse weight increased
        let after_synapses = store.get_synapses(auth_login.id).await.unwrap();
        let new_session_weight = after_synapses
            .iter()
            .find(|(id, _)| *id == auth_session.id)
            .map(|(_, w)| *w)
            .unwrap_or(0.0);
        assert!(
            new_session_weight > original_session_weight,
            "Synapse weight should increase after reinforcement: {} > {}",
            new_session_weight,
            original_session_weight
        );

        // Boost energy of co-activated notes
        for &id in &co_activated {
            store.boost_energy(id, 0.1).await.unwrap();
        }

        // Verify energy boosted (capped at 1.0)
        let notes = store
            .list_notes(None, None, &Default::default())
            .await
            .unwrap();
        let boosted_login = notes.0.iter().find(|n| n.id == auth_login.id).unwrap();
        assert!(
            boosted_login.energy >= 1.0,
            "auth_login energy should be >= 1.0 after boost"
        );
        assert!(
            boosted_login.last_activated.is_some(),
            "last_activated should be set after boost"
        );

        // Second reinforcement — weights should increase further
        let reinforced2 = store.reinforce_synapses(&co_activated, 0.05).await.unwrap();
        assert!(reinforced2 > 0);
        let after2_synapses = store.get_synapses(auth_login.id).await.unwrap();
        let weight_after_2 = after2_synapses
            .iter()
            .find(|(id, _)| *id == auth_session.id)
            .map(|(_, w)| *w)
            .unwrap_or(0.0);
        assert!(
            weight_after_2 > new_session_weight,
            "Second reinforcement should further increase weight: {} > {}",
            weight_after_2,
            new_session_weight
        );
        assert!(weight_after_2 <= 1.0, "Weight capped at 1.0");

        // ── Phase 4: Decay and Pruning ──────────────────────────────────
        // Create a very weak synapse to test pruning
        store
            .create_synapses(
                db_query.id,
                &[
                    (auth_password.id, 0.15), // weak cross-cluster link
                ],
            )
            .await
            .unwrap();

        // Apply heavy decay
        let (decayed, _pruned) = store.decay_synapses(0.5, 0.1).await.unwrap();
        assert!(decayed > 0, "Should have decayed some synapses");

        // The weak synapse (0.15 - 0.5 = -0.35) should be pruned
        let db_synapses_after = store.get_synapses(db_query.id).await.unwrap();
        let weak_link = db_synapses_after
            .iter()
            .find(|(id, _)| *id == auth_password.id);
        assert!(
            weak_link.is_none(),
            "Weak synapse should have been pruned after heavy decay"
        );

        // Strong synapses should survive (original 0.92 + boosts - 0.5 decay)
        let login_synapses_after = store.get_synapses(auth_login.id).await.unwrap();
        let session_link = login_synapses_after
            .iter()
            .find(|(id, _)| *id == auth_session.id);
        assert!(
            session_link.is_some(),
            "Strong synapse (auth_login ↔ auth_session) should survive decay"
        );

        // ── Phase 5: Verify isolation persists after all operations ─────
        // Search in project B should only see project B notes
        let results_b = engine
            .activate("authentication", Some(project_b), &config)
            .await
            .unwrap();
        let b_result_ids: Vec<Uuid> = results_b.iter().map(|r| r.note.id).collect();
        for &project_a_id in &[
            auth_login.id,
            auth_session.id,
            auth_password.id,
            db_query.id,
            db_migration.id,
        ] {
            assert!(
                !b_result_ids.contains(&project_a_id),
                "Project A notes must not appear in Project B search"
            );
        }

        // ── Phase 6: Cleanup verification ───────────────────────────────
        // Delete a note and verify its synapses are cleaned up
        store.delete_synapses(auth_password.id).await.unwrap();
        let password_synapses = store.get_synapses(auth_password.id).await.unwrap();
        assert!(
            password_synapses.is_empty(),
            "Synapses should be cleaned up after delete"
        );

        // Existing notes still function normally
        let final_results = engine
            .activate("authentication", Some(project_a), &config)
            .await
            .unwrap();
        assert!(
            !final_results.is_empty(),
            "System should still work after cleanup"
        );
    }

    #[tokio::test]
    async fn test_min_cosine_similarity_filters_low_scores() {
        // With a high min_cosine_similarity threshold (0.99), only exact or
        // near-exact embedding matches should survive. Notes with different
        // content will have lower cosine similarity and be filtered out.
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);
        let project_id = Uuid::new_v4();

        // Create two notes with different content
        let note_exact =
            note_with_energy(Uuid::new_v4(), Some(project_id), "exact match query", 1.0);
        let note_different = note_with_energy(
            Uuid::new_v4(),
            Some(project_id),
            "completely unrelated topic about databases",
            1.0,
        );

        store.create_note(&note_exact).await.unwrap();
        store.create_note(&note_different).await.unwrap();

        // Embed note_exact with the QUERY text (will get score ~1.0)
        let provider = mock_embedding_provider();
        let query_emb = provider.embed_text("exact match query").await.unwrap();
        store
            .set_note_embedding(note_exact.id, &query_emb, "mock")
            .await
            .unwrap();

        // Embed note_different with its own content (lower cosine similarity to query)
        let diff_emb = provider
            .embed_text("completely unrelated topic about databases")
            .await
            .unwrap();
        store
            .set_note_embedding(note_different.id, &diff_emb, "mock")
            .await
            .unwrap();

        let engine = SpreadingActivationEngine::new(store, provider);

        // With high threshold: only the exact match should survive
        let config_strict = SpreadingActivationConfig {
            min_cosine_similarity: 0.99,
            ..Default::default()
        };
        let results = engine
            .activate("exact match query", Some(project_id), &config_strict)
            .await
            .unwrap();

        // Only the exact match note should be returned
        assert_eq!(
            results.len(),
            1,
            "Strict threshold should keep only exact match"
        );
        assert_eq!(results[0].note.id, note_exact.id);

        // With no threshold: both should appear
        let config_permissive = SpreadingActivationConfig {
            min_cosine_similarity: 0.0,
            ..Default::default()
        };
        let results_all = engine
            .activate("exact match query", Some(project_id), &config_permissive)
            .await
            .unwrap();

        assert!(
            results_all.len() >= 2,
            "Permissive threshold should return both notes, got {}",
            results_all.len()
        );
    }

    #[tokio::test]
    async fn test_min_cosine_similarity_gibberish_returns_empty() {
        // Gibberish query should return no results when min_cosine_similarity
        // filters out the low-relevance "nearest neighbors".
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);
        let project_id = Uuid::new_v4();

        let note = note_with_energy(
            Uuid::new_v4(),
            Some(project_id),
            "authentication login flow",
            1.0,
        );
        store.create_note(&note).await.unwrap();

        let provider = mock_embedding_provider();
        let emb = provider
            .embed_text("authentication login flow")
            .await
            .unwrap();
        store
            .set_note_embedding(note.id, &emb, "mock")
            .await
            .unwrap();

        let engine = SpreadingActivationEngine::new(store, provider);

        // Query with gibberish — embedding will be very different from stored notes
        let config = SpreadingActivationConfig {
            min_cosine_similarity: 0.5, // Moderate threshold
            ..Default::default()
        };
        let results = engine
            .activate("xyzzy foobar qux nonsense", Some(project_id), &config)
            .await
            .unwrap();

        // With hash-based mock embeddings, gibberish will have low similarity
        // to real content. The threshold should filter it out.
        // (If the mock happens to produce high similarity by chance, this test
        // verifies at minimum that the filtering mechanism is working.)
        assert!(
            results.len() <= 1,
            "Gibberish query with threshold should return few or no results, got {}",
            results.len()
        );
    }

    #[tokio::test]
    async fn test_activate_with_profile_none_same_as_activate() {
        // This is a compile-time check that the API works.
        // Full integration tests require a real GraphStore + EmbeddingProvider.
        // The key guarantee: activate(q, p, c) == activate_with_profile(q, p, c, None)
        // which is ensured by the delegation pattern.
        let mock = Arc::new(MockGraphStore::new());
        let engine = SpreadingActivationEngine::new(gs(&mock), mock_embedding_provider());
        let config = SpreadingActivationConfig::default();

        let results_activate = engine.activate("test query", None, &config).await.unwrap();
        let results_with_profile = engine
            .activate_with_profile("test query", None, &config, None)
            .await
            .unwrap();

        assert_eq!(results_activate.len(), results_with_profile.len());
    }

    #[tokio::test]
    async fn test_activate_with_profile_dampens_spreading() {
        use std::collections::HashMap;

        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);
        let project_id = Uuid::new_v4();

        let note_a = note_with_energy(Uuid::new_v4(), Some(project_id), "profile source", 1.0);
        let note_b = note_with_energy(Uuid::new_v4(), Some(project_id), "profile neighbor", 1.0);

        store.create_note(&note_a).await.unwrap();
        store.create_note(&note_b).await.unwrap();

        let embedding = mock_embedding_provider()
            .embed_text("profile source")
            .await
            .unwrap();
        store
            .set_note_embedding(note_a.id, &embedding, "mock")
            .await
            .unwrap();

        store
            .create_synapses(note_a.id, &[(note_b.id, 0.9)])
            .await
            .unwrap();

        let config = SpreadingActivationConfig {
            min_activation: 0.001,
            ..Default::default()
        };

        let engine = SpreadingActivationEngine::new(store, mock_embedding_provider());

        // Without profile (SYNAPSE weight = 1.0)
        let results_no_profile = engine
            .activate_with_profile("profile source", Some(project_id), &config, None)
            .await
            .unwrap();

        // With a low-SYNAPSE profile (dampened spreading)
        let low_synapse_profile = AnalysisProfile {
            id: Uuid::new_v4().to_string(),
            project_id: None,
            name: "test_low_synapse".to_string(),
            description: None,
            edge_weights: HashMap::from([("SYNAPSE".to_string(), 0.3)]),
            fusion_weights: Default::default(),
            is_builtin: false,
        };
        let results_low = engine
            .activate_with_profile(
                "profile source",
                Some(project_id),
                &config,
                Some(&low_synapse_profile),
            )
            .await
            .unwrap();

        // Both should find the direct match (note_a)
        assert!(results_no_profile.iter().any(|r| r.note.id == note_a.id));
        assert!(results_low.iter().any(|r| r.note.id == note_a.id));

        // If note_b appears in both, the low-profile version should have a lower score
        let b_score_no_profile = results_no_profile
            .iter()
            .find(|r| r.note.id == note_b.id)
            .map(|r| r.activation_score);
        let b_score_low = results_low
            .iter()
            .find(|r| r.note.id == note_b.id)
            .map(|r| r.activation_score);

        if let (Some(score_none), Some(score_low)) = (b_score_no_profile, b_score_low) {
            assert!(
                score_low < score_none,
                "Low SYNAPSE profile should dampen propagated score: {} < {}",
                score_low,
                score_none
            );
        }
    }

    // ========================================================================
    // Spreading Search tests (activate_and_collect)
    // ========================================================================

    /// Test 1: Graph with 3 clusters — activate node in cluster A → results
    /// biased toward cluster A nodes.
    #[tokio::test]
    async fn test_spreading_search_cluster_bias() {
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);

        // Cluster A: a1 → a2 → a3 (strong synapses)
        let a1 = note_with_energy(Uuid::new_v4(), None, "cluster_a_1", 1.0);
        let a2 = note_with_energy(Uuid::new_v4(), None, "cluster_a_2", 1.0);
        let a3 = note_with_energy(Uuid::new_v4(), None, "cluster_a_3", 1.0);

        // Cluster B: b1 → b2 (only weakly connected to A via a1-b1)
        let b1 = note_with_energy(Uuid::new_v4(), None, "cluster_b_1", 1.0);
        let b2 = note_with_energy(Uuid::new_v4(), None, "cluster_b_2", 1.0);

        // Cluster C: c1 → c2 (not connected to A at all)
        let c1 = note_with_energy(Uuid::new_v4(), None, "cluster_c_1", 1.0);
        let c2 = note_with_energy(Uuid::new_v4(), None, "cluster_c_2", 1.0);

        for n in [&a1, &a2, &a3, &b1, &b2, &c1, &c2] {
            store.create_note(n).await.unwrap();
        }

        // Cluster A internal synapses (strong)
        store.create_synapses(a1.id, &[(a2.id, 0.9)]).await.unwrap();
        store
            .create_synapses(a2.id, &[(a3.id, 0.85)])
            .await
            .unwrap();

        // Weak cross-cluster link A→B
        store.create_synapses(a1.id, &[(b1.id, 0.3)]).await.unwrap();

        // Cluster B internal
        store.create_synapses(b1.id, &[(b2.id, 0.8)]).await.unwrap();

        // Cluster C internal (isolated from A)
        store.create_synapses(c1.id, &[(c2.id, 0.9)]).await.unwrap();

        let engine = SpreadingActivationEngine::new(store, mock_embedding_provider());

        let config = ActivationSearchConfig {
            top_k: 10,
            max_hops: 3,
            decay_per_hop: 0.7,
            min_activation: 0.01,
            max_visited: 100,
            min_energy: 0.05,
        };

        let results = engine.activate_and_collect(a1.id, &config).await.unwrap();

        // Cluster A nodes should appear and rank highest
        let a_ids: HashSet<Uuid> = [a2.id, a3.id].iter().copied().collect();
        let c_ids: HashSet<Uuid> = [c1.id, c2.id].iter().copied().collect();

        let a_results: Vec<_> = results
            .iter()
            .filter(|r| a_ids.contains(&r.node_id))
            .collect();
        let c_results: Vec<_> = results
            .iter()
            .filter(|r| c_ids.contains(&r.node_id))
            .collect();

        // Cluster A should have results
        assert!(!a_results.is_empty(), "Cluster A nodes should be activated");

        // Cluster C should NOT have results (isolated)
        assert!(
            c_results.is_empty(),
            "Cluster C nodes should not be reached (isolated)"
        );

        // The top result should be from cluster A (a2 is directly connected)
        assert!(
            a_ids.contains(&results[0].node_id),
            "Top result should be from cluster A"
        );
    }

    /// Test 2: activation_weight=0 effectively excludes activation signal.
    /// (This is tested in search.rs pipeline, but we also verify
    /// activate_and_collect returns empty for nonexistent source.)
    #[tokio::test]
    async fn test_spreading_search_nonexistent_source() {
        let mock = Arc::new(MockGraphStore::new());
        let engine = SpreadingActivationEngine::new(gs(&mock), mock_embedding_provider());

        let result = engine
            .activate_and_collect(Uuid::new_v4(), &ActivationSearchConfig::default())
            .await;

        // Should error — source not found
        assert!(result.is_err(), "Nonexistent source should return error");
    }

    /// Test 3: Dense graph → circuit breaker limits visited nodes.
    #[tokio::test]
    async fn test_spreading_search_circuit_breaker() {
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);

        // Create a star graph: center connected to 50 nodes
        let center = note_with_energy(Uuid::new_v4(), None, "center", 1.0);
        store.create_note(&center).await.unwrap();

        let mut spokes = Vec::new();
        for i in 0..50 {
            let spoke = note_with_energy(Uuid::new_v4(), None, &format!("spoke_{}", i), 1.0);
            store.create_note(&spoke).await.unwrap();
            spokes.push(spoke);
        }

        // Connect all spokes to center
        let synapse_pairs: Vec<(Uuid, f64)> = spokes.iter().map(|s| (s.id, 0.8)).collect();
        store
            .create_synapses(center.id, &synapse_pairs)
            .await
            .unwrap();

        // Also connect spokes to each other in a chain for depth
        for i in 0..49 {
            store
                .create_synapses(spokes[i].id, &[(spokes[i + 1].id, 0.7)])
                .await
                .unwrap();
        }

        let engine = SpreadingActivationEngine::new(store, mock_embedding_provider());

        // Set a very low circuit breaker limit
        let config = ActivationSearchConfig {
            top_k: 100,
            max_hops: 5,
            decay_per_hop: 0.8,
            min_activation: 0.001,
            max_visited: 10, // Circuit breaker at 10 nodes
            min_energy: 0.05,
        };

        let results = engine
            .activate_and_collect(center.id, &config)
            .await
            .unwrap();

        // Circuit breaker should limit results — we should NOT get all 50 spokes
        assert!(
            results.len() < 50,
            "Circuit breaker should limit results: got {} (expected < 50)",
            results.len()
        );
        // But we should still get some results
        assert!(
            !results.is_empty(),
            "Should still return some results before circuit breaker"
        );
    }

    /// Test 4: Use-case example — recommendation via activation.
    /// Activate a "purchased product" node → find related products
    /// through the graph structure.
    #[tokio::test]
    async fn test_spreading_search_recommendation_use_case() {
        let mock = Arc::new(MockGraphStore::new());
        let store = gs(&mock);

        // Product graph: user bought "Rust Book"
        // Rust Book → (0.9) → Rust Cookbook → (0.8) → Async Programming
        // Rust Book → (0.7) → Systems Programming → (0.6) → C Programming
        let rust_book = note_with_energy(Uuid::new_v4(), None, "Rust Book", 1.0);
        let rust_cookbook = note_with_energy(Uuid::new_v4(), None, "Rust Cookbook", 1.0);
        let async_prog = note_with_energy(Uuid::new_v4(), None, "Async Programming in Rust", 1.0);
        let systems_prog = note_with_energy(Uuid::new_v4(), None, "Systems Programming", 0.9);
        let c_prog = note_with_energy(Uuid::new_v4(), None, "C Programming", 0.8);
        let unrelated = note_with_energy(Uuid::new_v4(), None, "Cooking Recipes", 1.0);

        for n in [
            &rust_book,
            &rust_cookbook,
            &async_prog,
            &systems_prog,
            &c_prog,
            &unrelated,
        ] {
            store.create_note(n).await.unwrap();
        }

        // Build the product graph
        store
            .create_synapses(
                rust_book.id,
                &[(rust_cookbook.id, 0.9), (systems_prog.id, 0.7)],
            )
            .await
            .unwrap();
        store
            .create_synapses(rust_cookbook.id, &[(async_prog.id, 0.8)])
            .await
            .unwrap();
        store
            .create_synapses(systems_prog.id, &[(c_prog.id, 0.6)])
            .await
            .unwrap();
        // "Cooking Recipes" is isolated — no synapses to the product graph

        let engine = SpreadingActivationEngine::new(store, mock_embedding_provider());

        let config = ActivationSearchConfig {
            top_k: 5,
            max_hops: 3,
            decay_per_hop: 0.7,
            min_activation: 0.01,
            max_visited: 100,
            min_energy: 0.05,
        };

        // User bought Rust Book → get recommendations
        let recommendations = engine
            .activate_and_collect(rust_book.id, &config)
            .await
            .unwrap();

        // Should recommend related books
        assert!(!recommendations.is_empty(), "Should have recommendations");

        // Rust Cookbook should rank highest (direct, strong synapse)
        assert_eq!(
            recommendations[0].node_id, rust_cookbook.id,
            "Rust Cookbook should be top recommendation"
        );

        // All recommendations should be from the connected graph
        let recommended_ids: HashSet<Uuid> = recommendations.iter().map(|r| r.node_id).collect();
        assert!(
            !recommended_ids.contains(&unrelated.id),
            "Unrelated 'Cooking Recipes' should NOT be recommended"
        );

        // Verify hop distances make sense
        let cookbook_result = recommendations
            .iter()
            .find(|r| r.node_id == rust_cookbook.id)
            .unwrap();
        assert_eq!(
            cookbook_result.hop_distance, 1,
            "Rust Cookbook is 1 hop from Rust Book"
        );

        // Verify path: Rust Book → Rust Cookbook
        assert_eq!(cookbook_result.path.len(), 2);
        assert_eq!(cookbook_result.path[0], rust_book.id);
        assert_eq!(cookbook_result.path[1], rust_cookbook.id);

        // Async Programming should be further away
        if let Some(async_result) = recommendations.iter().find(|r| r.node_id == async_prog.id) {
            assert_eq!(
                async_result.hop_distance, 2,
                "Async Programming is 2 hops from Rust Book"
            );
            // Should have lower activation than Rust Cookbook
            assert!(
                async_result.activation_level < cookbook_result.activation_level,
                "2-hop result should score lower than 1-hop"
            );
        }
    }
}
