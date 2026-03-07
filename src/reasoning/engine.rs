//! ReasoningTree Engine
//!
//! Implements the 3-phase algorithm for building a ReasoningTree:
//! 1. **Activation**: Embed query → parallel vector search on notes + decisions → seed nodes
//! 2. **Propagation**: BFS through SYNAPSE, LINKED_TO, AFFECTS → multi-factor scoring → pruning
//! 3. **Cristallisation**: Transform activated subgraph into a decision tree with actions
//!
//! The engine reuses the existing embedding infrastructure (`EmbeddingProvider`)
//! and graph store (`GraphStore`) via dependency injection.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::embeddings::EmbeddingProvider;
use crate::neo4j::traits::GraphStore;
use crate::skills::models::SkillNode;

use super::cache::ReasoningTreeCache;
use super::models::{Action, EntitySource, ReasoningNode, ReasoningTree, ReasoningTreeConfig};

// ============================================================================
// Seed Node (Phase 1 output)
// ============================================================================

/// A seed node discovered during the activation phase.
///
/// Seeds are the starting points for graph propagation (Phase 2).
/// They come from vector search on notes, decisions, or from skill trigger matching.
#[derive(Debug, Clone)]
pub struct SeedNode {
    /// The type of entity
    pub entity_type: EntitySource,

    /// The entity identifier (UUID string for notes/decisions/skills, path for files)
    pub entity_id: String,

    /// Display label (content preview for notes, description for decisions, name for skills)
    pub label: String,

    /// Cosine similarity or trigger confidence score (0.0 - 1.0)
    pub score: f64,

    /// How this seed was discovered
    pub source: SeedSource,
}

/// How a seed node was discovered during Phase 1.
#[derive(Debug, Clone)]
pub enum SeedSource {
    /// Discovered via vector search on notes
    NoteVectorSearch,
    /// Discovered via vector search on decisions
    DecisionVectorSearch,
    /// Discovered via skill trigger pattern matching (fast-path)
    SkillTriggerMatch {
        /// The matched skill's name
        skill_name: String,
        /// The matched trigger's confidence threshold
        trigger_confidence: f64,
    },
}

// ============================================================================
// Activated Node (Phase 2 output — intermediate representation)
// ============================================================================

/// An entity activated during the propagation phase (Phase 2).
///
/// This is the intermediate representation between propagation and cristallisation.
/// Each activated node knows which seed it originated from and how it was reached.
#[derive(Debug, Clone)]
struct ActivatedEntity {
    /// The type of entity
    entity_type: EntitySource,

    /// The entity identifier
    entity_id: String,

    /// Display label
    label: String,

    /// Final relevance score after multi-factor scoring
    relevance: f64,

    /// Depth at which this entity was discovered (0 = seed, 1 = first hop, etc.)
    depth: usize,

    /// The seed node that activated this entity (for tree grouping in Phase 3)
    seed_id: String,

    /// Human-readable reasoning explaining why this node is relevant
    reasoning: String,

    /// Which relation type was traversed to reach this entity
    _relation: ActivationRelation,
}

/// How an entity was activated during propagation.
#[derive(Debug, Clone)]
enum ActivationRelation {
    /// Direct seed from Phase 1 (vector search or trigger match)
    Seed,
    /// Reached via SYNAPSE relation (note ↔ note/decision)
    Synapse { _weight: f64 },
    /// Reached via LINKED_TO relation (note → file/function/struct)
    LinkedTo,
    /// Reached via AFFECTS relation (decision → file/function)
    Affects,
}

// ============================================================================
// ReasoningTreeEngine
// ============================================================================

/// Engine for building ReasoningTrees from the knowledge graph.
///
/// Designed for injection via `Arc<ReasoningTreeEngine>`.
///
/// # Usage
///
/// ```rust,ignore
/// let engine = ReasoningTreeEngine::new(graph_store, embedding_provider, cache);
///
/// // Build a tree (checks cache first)
/// let tree = engine.build("how does the chat system work?", None, &config).await?;
///
/// // Provide feedback after using the tree
/// engine.feedback(tree.id, &followed_nodes, "success").await?;
/// ```
pub struct ReasoningTreeEngine {
    graph_store: Arc<dyn GraphStore>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    cache: ReasoningTreeCache,
}

impl ReasoningTreeEngine {
    /// Create a new engine with the given dependencies.
    pub fn new(
        graph_store: Arc<dyn GraphStore>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        cache: ReasoningTreeCache,
    ) -> Self {
        Self {
            graph_store,
            embedding_provider,
            cache,
        }
    }

    /// Build a ReasoningTree for a natural language request.
    ///
    /// Checks the cache first. If no cached tree exists (or it's expired),
    /// builds a new one through the 3-phase pipeline.
    ///
    /// # Arguments
    /// * `request` — natural language query
    /// * `project_id` — optional project scope for data isolation
    /// * `config` — algorithm parameters (use `Default::default()` for sensible defaults)
    pub async fn build(
        &self,
        request: &str,
        project_id: Option<Uuid>,
        config: &ReasoningTreeConfig,
    ) -> Result<ReasoningTree> {
        // Check cache
        if let Some(cached) = self.cache.get(request, project_id).await {
            debug!("ReasoningTree cache hit for request: '{}'", request);
            return Ok(cached);
        }

        let start = Instant::now();

        // Phase 1: Activation — vector search → seed nodes
        let (seeds, embedding) = self.phase1_activate(request, project_id, config).await?;

        debug!(
            "Phase 1 complete: {} seeds in {:?}",
            seeds.len(),
            start.elapsed()
        );

        if seeds.is_empty() {
            let mut tree = ReasoningTree::new(request, project_id);
            tree.request_embedding = Some(embedding);
            tree.build_time_ms = Some(start.elapsed().as_millis() as u64);
            self.cache.insert(tree.clone()).await;
            return Ok(tree);
        }

        // Phase 2: Propagation — BFS through graph relations
        let activated = self.phase2_propagate(&seeds, config).await?;

        debug!(
            "Phase 2 complete: {} activated entities in {:?}",
            activated.len(),
            start.elapsed()
        );

        // Phase 3: Cristallisation — transform to tree
        let tree = self
            .phase3_cristallize(
                request, project_id, embedding, &seeds, &activated, config, start,
            )
            .await;

        debug!(
            "Phase 3 complete: tree with {} nodes, confidence {:.2} in {:?}",
            tree.node_count,
            tree.confidence,
            start.elapsed()
        );

        // Cache the result
        self.cache.insert(tree.clone()).await;

        Ok(tree)
    }

    // ========================================================================
    // Cache management
    // ========================================================================

    /// Invalidate a cached reasoning tree by its UUID.
    ///
    /// Used after `reason_feedback` to ensure re-computation with updated
    /// neural scores on the next `build()` call.
    pub async fn invalidate_cache(&self, tree_id: Uuid) -> bool {
        self.cache.invalidate(tree_id).await
    }

    // ========================================================================
    // Phase 1: Activation
    // ========================================================================

    /// Phase 1: Activation — embed request, parallel vector search, merge seeds.
    ///
    /// Returns the top-K seed nodes and the request embedding (for reuse in Phase 2).
    async fn phase1_activate(
        &self,
        request: &str,
        project_id: Option<Uuid>,
        config: &ReasoningTreeConfig,
    ) -> Result<(Vec<SeedNode>, Vec<f32>)> {
        // Step 1: Embed the request
        let embedding = self.embedding_provider.embed_text(request).await?;

        // Step 2: Parallel vector search on notes + decisions
        let note_limit = config.max_seeds;
        let decision_limit = config.max_seeds / 2; // Decisions are less numerous

        let project_id_str = project_id.map(|p| p.to_string());

        let (note_results, decision_results) = tokio::join!(
            self.graph_store.vector_search_notes(
                &embedding,
                note_limit,
                project_id,
                None,      // no workspace filter
                Some(0.3), // min cosine similarity
            ),
            self.graph_store.search_decisions_by_vector(
                &embedding,
                decision_limit,
                project_id_str.as_deref(),
            ),
        );

        let mut seeds: Vec<SeedNode> = Vec::new();

        // Process note results
        if let Ok(notes) = note_results {
            for (note, score) in notes {
                if note.energy < 0.05 {
                    continue; // Skip dead neurons
                }

                let label = truncate_content(&note.content, 100);
                seeds.push(SeedNode {
                    entity_type: EntitySource::Note,
                    entity_id: note.id.to_string(),
                    label,
                    score,
                    source: SeedSource::NoteVectorSearch,
                });
            }
        }

        // Process decision results
        if let Ok(decisions) = decision_results {
            for (decision, score) in decisions {
                let label = truncate_content(&decision.description, 100);
                seeds.push(SeedNode {
                    entity_type: EntitySource::Decision,
                    entity_id: decision.id.to_string(),
                    label,
                    score,
                    source: SeedSource::DecisionVectorSearch,
                });
            }
        }

        // Step 3: Fast-path skill matching (if project scoped)
        if let Some(pid) = project_id {
            if let Ok(skills) = self.graph_store.get_skills_for_project(pid).await {
                for skill in &skills {
                    if let Some(seed) = check_skill_fast_path(skill, request) {
                        seeds.push(seed);
                    }
                }
            }
        }

        // Sort by score descending and take top-K
        seeds.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        seeds.truncate(config.max_seeds);

        Ok((seeds, embedding))
    }

    // ========================================================================
    // Phase 2: Propagation
    // ========================================================================

    /// Phase 2: Propagation — BFS through graph relations with multi-factor scoring.
    ///
    /// Starting from seed nodes, traverses:
    /// - **SYNAPSE** (note ↔ note/decision) — weighted by synapse.weight × energy
    /// - **LINKED_TO** (note → file/function/struct/trait) — via note anchors
    /// - **AFFECTS** (decision → file/function) — via decision-affects relations
    ///
    /// Each discovered entity is scored:
    /// `relevance = parent_score × relation_weight × energy × recency_decay`
    ///
    /// Nodes below `config.min_relevance` are pruned.
    async fn phase2_propagate(
        &self,
        seeds: &[SeedNode],
        config: &ReasoningTreeConfig,
    ) -> Result<Vec<ActivatedEntity>> {
        let mut activated: Vec<ActivatedEntity> = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        let mut total_nodes = 0usize;

        // Register seeds as depth-0 activated entities
        for seed in seeds {
            visited.insert(seed.entity_id.clone());
            activated.push(ActivatedEntity {
                entity_type: seed.entity_type,
                entity_id: seed.entity_id.clone(),
                label: seed.label.clone(),
                relevance: seed.score,
                depth: 0,
                seed_id: seed.entity_id.clone(),
                reasoning: format_seed_reasoning(seed),
                _relation: ActivationRelation::Seed,
            });
            total_nodes += 1;
        }

        // BFS through SYNAPSE relations (hop by hop, up to max_depth)
        // We use the existing get_cross_entity_synapses which handles Note↔Note and Note↔Decision
        let mut frontier: Vec<(String, f64, String, usize)> = seeds
            .iter()
            .filter(|s| matches!(s.entity_type, EntitySource::Note | EntitySource::Decision))
            .map(|s| (s.entity_id.clone(), s.score, s.entity_id.clone(), 0usize))
            .collect();

        for depth in 1..=config.max_depth {
            if frontier.is_empty() || total_nodes >= config.max_nodes {
                break;
            }

            let mut next_frontier: Vec<(String, f64, String, usize)> = Vec::new();

            for (entity_id, parent_score, seed_id, _) in &frontier {
                // Parse UUID — only note/decision entities have UUID IDs
                let node_uuid = match Uuid::parse_str(entity_id) {
                    Ok(id) => id,
                    Err(_) => continue,
                };

                // Traverse SYNAPSE relations (cross-entity: Note↔Note, Note↔Decision)
                let synapses = match self.graph_store.get_cross_entity_synapses(node_uuid).await {
                    Ok(s) => s,
                    Err(_) => {
                        // Fallback to note-only synapses
                        match self.graph_store.get_synapses(node_uuid).await {
                            Ok(s) => s
                                .into_iter()
                                .map(|(id, w)| (id, w, "Note".to_string()))
                                .collect(),
                            Err(_) => continue,
                        }
                    }
                };

                for (neighbor_id, synapse_weight, entity_type_str) in synapses {
                    let neighbor_id_str = neighbor_id.to_string();
                    if visited.contains(&neighbor_id_str) || total_nodes >= config.max_nodes {
                        continue;
                    }

                    // Resolve neighbor and compute score
                    let (label, energy, entity_type) = if entity_type_str == "Decision" {
                        match self.graph_store.get_decision(neighbor_id).await {
                            Ok(Some(d)) => (
                                truncate_content(&d.description, 80),
                                1.0,
                                EntitySource::Decision,
                            ),
                            _ => continue,
                        }
                    } else {
                        match self.graph_store.get_note(neighbor_id).await {
                            Ok(Some(n)) => {
                                if n.energy < 0.05 {
                                    continue; // Dead neuron
                                }
                                (
                                    truncate_content(&n.content, 80),
                                    n.energy,
                                    EntitySource::Note,
                                )
                            }
                            _ => continue,
                        }
                    };

                    // Multi-factor scoring:
                    // relevance = parent_score × synapse_weight × energy × decay
                    let decay = recency_decay(depth);
                    let relevance = parent_score * synapse_weight * energy * decay;

                    if relevance < config.min_relevance {
                        continue; // Prune weak branches
                    }

                    visited.insert(neighbor_id_str.clone());
                    activated.push(ActivatedEntity {
                        entity_type,
                        entity_id: neighbor_id_str.clone(),
                        label,
                        relevance,
                        depth,
                        seed_id: seed_id.clone(),
                        reasoning: format!(
                            "{} reached via SYNAPSE (weight: {:.2}, energy: {:.2}, depth: {})",
                            entity_type, synapse_weight, energy, depth
                        ),
                        _relation: ActivationRelation::Synapse {
                            _weight: synapse_weight,
                        },
                    });
                    total_nodes += 1;

                    next_frontier.push((neighbor_id_str, relevance, seed_id.clone(), depth));
                }
            }

            frontier = next_frontier;
        }

        // Discover LINKED_TO entities for note seeds (files, functions, structs)
        for seed in seeds.iter().filter(|s| s.entity_type == EntitySource::Note) {
            if total_nodes >= config.max_nodes {
                break;
            }

            if let Ok(uuid) = Uuid::parse_str(&seed.entity_id) {
                if let Ok(anchors) = self.graph_store.get_note_anchors(uuid).await {
                    for anchor in anchors {
                        if visited.contains(&anchor.entity_id) || total_nodes >= config.max_nodes {
                            continue;
                        }

                        let entity_type = entity_type_from_str(&anchor.entity_type.to_string());
                        let relevance = seed.score * 0.8; // Linked entities inherit 80% of parent score

                        if relevance < config.min_relevance {
                            continue;
                        }

                        visited.insert(anchor.entity_id.clone());
                        activated.push(ActivatedEntity {
                            entity_type,
                            entity_id: anchor.entity_id.clone(),
                            label: anchor.entity_id.clone(),
                            relevance,
                            depth: 1,
                            seed_id: seed.entity_id.clone(),
                            reasoning: format!(
                                "{} linked to note via LINKED_TO (inherited score: {:.2})",
                                entity_type, relevance
                            ),
                            _relation: ActivationRelation::LinkedTo,
                        });
                        total_nodes += 1;
                    }
                }
            }
        }

        // Discover AFFECTS entities for decision seeds
        for seed in seeds
            .iter()
            .filter(|s| s.entity_type == EntitySource::Decision)
        {
            if total_nodes >= config.max_nodes {
                break;
            }

            if let Ok(uuid) = Uuid::parse_str(&seed.entity_id) {
                if let Ok(affects) = self.graph_store.list_decision_affects(uuid).await {
                    for affect in affects {
                        if visited.contains(&affect.entity_id) || total_nodes >= config.max_nodes {
                            continue;
                        }

                        let entity_type = entity_type_from_str(&affect.entity_type);
                        let relevance = seed.score * 0.75; // Affected entities inherit 75% of decision score

                        if relevance < config.min_relevance {
                            continue;
                        }

                        visited.insert(affect.entity_id.clone());
                        activated.push(ActivatedEntity {
                            entity_type,
                            entity_id: affect.entity_id.clone(),
                            label: affect.entity_id.clone(),
                            relevance,
                            depth: 1,
                            seed_id: seed.entity_id.clone(),
                            reasoning: format!(
                                "{} affected by decision via AFFECTS (score: {:.2})",
                                entity_type, relevance
                            ),
                            _relation: ActivationRelation::Affects,
                        });
                        total_nodes += 1;
                    }
                }
            }
        }

        // Sort all activated entities by relevance descending
        activated.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Enforce max_nodes
        activated.truncate(config.max_nodes);

        Ok(activated)
    }

    // ========================================================================
    // Phase 3: Cristallisation
    // ========================================================================

    /// Phase 3: Cristallisation — transform activated entities into a ReasoningTree.
    ///
    /// Groups activated entities by their seed origin, creates hierarchical
    /// ReasoningNodes with children, and generates Action suggestions for leaf nodes.
    #[allow(clippy::too_many_arguments)]
    async fn phase3_cristallize(
        &self,
        request: &str,
        project_id: Option<Uuid>,
        embedding: Vec<f32>,
        seeds: &[SeedNode],
        activated: &[ActivatedEntity],
        config: &ReasoningTreeConfig,
        start: Instant,
    ) -> ReasoningTree {
        let mut tree = ReasoningTree::new(request, project_id);
        tree.request_embedding = Some(embedding);

        // Group activated entities by seed_id
        let mut by_seed: HashMap<String, Vec<&ActivatedEntity>> = HashMap::new();
        for entity in activated {
            by_seed
                .entry(entity.seed_id.clone())
                .or_default()
                .push(entity);
        }

        // Build root nodes (one per seed) with their children
        for seed in seeds {
            // Find all entities activated from this seed
            let children_entities: Vec<&&ActivatedEntity> = by_seed
                .get(&seed.entity_id)
                .map(|entities| {
                    entities
                        .iter()
                        .filter(|e| e.depth > 0) // Exclude seed itself
                        .collect()
                })
                .unwrap_or_default();

            // Create root node
            let mut root = ReasoningNode::new(
                seed.entity_type,
                &seed.entity_id,
                seed.score,
                format_seed_reasoning(seed),
            );
            root.label = Some(seed.label.clone());

            // Add action for root
            if config.include_actions {
                if let Some(action) = generate_action(seed.entity_type, &seed.entity_id, seed.score)
                {
                    root = root.with_action(action);
                }
            }

            // Add children (sorted by relevance, grouped by depth)
            let mut depth1_children: Vec<ReasoningNode> = Vec::new();

            for entity in &children_entities {
                let mut child_node = ReasoningNode::new(
                    entity.entity_type,
                    &entity.entity_id,
                    entity.relevance,
                    &entity.reasoning,
                )
                .with_depth(entity.depth);

                child_node.label = Some(entity.label.clone());

                // Generate action for children
                if config.include_actions {
                    if let Some(action) =
                        generate_action(entity.entity_type, &entity.entity_id, entity.relevance)
                    {
                        child_node = child_node.with_action(action);
                    }
                }

                depth1_children.push(child_node);
            }

            // Sort children by relevance descending
            depth1_children.sort_by(|a, b| {
                b.relevance
                    .partial_cmp(&a.relevance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for child in depth1_children {
                root.add_child(child);
            }

            tree.add_root(root);
        }

        tree.build_time_ms = Some(start.elapsed().as_millis() as u64);
        tree
    }

    /// Get a reference to the cache (for external invalidation).
    pub fn cache(&self) -> &ReasoningTreeCache {
        &self.cache
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Exponential recency decay factor based on depth.
///
/// `decay = exp(-0.3 × depth)` — each hop reduces signal by ~26%.
/// depth=0 → 1.0, depth=1 → 0.74, depth=2 → 0.55, depth=3 → 0.41, depth=4 → 0.30
fn recency_decay(depth: usize) -> f64 {
    (-0.3 * depth as f64).exp()
}

/// Check if a skill has a trigger pattern that matches the request (fast-path).
///
/// Only checks Regex triggers for fast matching. Semantic triggers are not
/// evaluated in the fast path to stay within the activation time budget.
fn check_skill_fast_path(skill: &SkillNode, request: &str) -> Option<SeedNode> {
    use crate::skills::models::TriggerType;

    // Only check active skills
    if skill.status != crate::skills::models::SkillStatus::Active {
        return None;
    }

    for trigger in &skill.trigger_patterns {
        match trigger.pattern_type {
            TriggerType::Regex => {
                if let Ok(re) = regex::RegexBuilder::new(&trigger.pattern_value)
                    .case_insensitive(true)
                    .size_limit(1 << 20) // 1MB
                    .build()
                {
                    if re.is_match(request) {
                        return Some(SeedNode {
                            entity_type: EntitySource::Skill,
                            entity_id: skill.id.to_string(),
                            label: skill.name.clone(),
                            score: trigger.confidence_threshold.max(0.8),
                            source: SeedSource::SkillTriggerMatch {
                                skill_name: skill.name.clone(),
                                trigger_confidence: trigger.confidence_threshold,
                            },
                        });
                    }
                }
            }
            TriggerType::Semantic => {
                // Semantic triggers require embedding comparison — skip in fast path
            }
            TriggerType::FileGlob | TriggerType::McpAction => {
                // FileGlob and McpAction triggers don't apply to natural language queries
            }
        }
    }

    None
}

/// Generate an MCP Action suggestion based on entity type and ID.
fn generate_action(entity_type: EntitySource, entity_id: &str, confidence: f64) -> Option<Action> {
    let (tool, action, params) = match entity_type {
        EntitySource::Note => ("note", "get", serde_json::json!({"note_id": entity_id})),
        EntitySource::Decision => (
            "decision",
            "get",
            serde_json::json!({"decision_id": entity_id}),
        ),
        EntitySource::Skill => (
            "skill",
            "activate",
            serde_json::json!({"skill_id": entity_id, "query": ""}),
        ),
        EntitySource::FeatureGraph => {
            ("feature_graph", "get", serde_json::json!({"id": entity_id}))
        }
        EntitySource::File => (
            "code",
            "get_file_symbols",
            serde_json::json!({"file_path": entity_id}),
        ),
        EntitySource::Function => (
            "code",
            "get_call_graph",
            serde_json::json!({"function": entity_id}),
        ),
        EntitySource::Struct | EntitySource::Trait => (
            "code",
            "find_references",
            serde_json::json!({"symbol": entity_id}),
        ),
    };

    Some(Action {
        tool: tool.to_string(),
        action: action.to_string(),
        params,
        confidence,
    })
}

/// Format a human-readable reasoning string for a seed node.
fn format_seed_reasoning(seed: &SeedNode) -> String {
    match &seed.source {
        SeedSource::NoteVectorSearch => {
            format!(
                "Note activated via vector search (cosine similarity: {:.2})",
                seed.score
            )
        }
        SeedSource::DecisionVectorSearch => {
            format!(
                "Decision activated via vector search (cosine similarity: {:.2})",
                seed.score
            )
        }
        SeedSource::SkillTriggerMatch {
            skill_name,
            trigger_confidence,
        } => {
            format!(
                "Skill '{}' matched via trigger pattern (confidence: {:.2})",
                skill_name, trigger_confidence
            )
        }
    }
}

/// Truncate content to a maximum length, adding "..." if truncated.
///
/// Handles UTF-8 correctly by finding a valid char boundary before slicing.
fn truncate_content(content: &str, max_len: usize) -> String {
    if content.len() <= max_len {
        content.to_string()
    } else {
        // Find a valid char boundary at or before max_len
        let mut end = max_len;
        while end > 0 && !content.is_char_boundary(end) {
            end -= 1;
        }
        let truncated = &content[..end];
        if let Some(last_space) = truncated.rfind(' ') {
            format!("{}...", &truncated[..last_space])
        } else {
            format!("{}...", truncated)
        }
    }
}

/// Map an entity type string from the graph to an EntitySource enum.
fn entity_type_from_str(s: &str) -> EntitySource {
    match s.to_lowercase().as_str() {
        "note" => EntitySource::Note,
        "decision" => EntitySource::Decision,
        "skill" => EntitySource::Skill,
        "feature_graph" => EntitySource::FeatureGraph,
        "file" => EntitySource::File,
        "function" => EntitySource::Function,
        "struct" | "class" => EntitySource::Struct,
        "trait" | "interface" => EntitySource::Trait,
        other => {
            warn!(entity_type = other, "Unknown entity type in reasoning graph, defaulting to File");
            EntitySource::File
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MockEmbeddingProvider;
    use crate::neo4j::mock::MockGraphStore;
    use std::time::Duration;

    fn make_engine() -> ReasoningTreeEngine {
        let graph_store = Arc::new(MockGraphStore::new());
        let embedding_provider = Arc::new(MockEmbeddingProvider::new(768));
        let cache = ReasoningTreeCache::with_config(10, Duration::from_secs(60));
        ReasoningTreeEngine::new(graph_store, embedding_provider, cache)
    }

    // ====================================================================
    // Helper function tests
    // ====================================================================

    #[test]
    fn test_truncate_content() {
        assert_eq!(truncate_content("short", 100), "short");
        assert_eq!(truncate_content("hello world foo bar", 11), "hello...");
        assert_eq!(truncate_content("nospaces", 4), "nosp...");
    }

    #[test]
    fn test_recency_decay() {
        let d0 = recency_decay(0);
        let d1 = recency_decay(1);
        let d2 = recency_decay(2);
        let d4 = recency_decay(4);

        // Depth 0 → no decay
        assert!((d0 - 1.0).abs() < f64::EPSILON);
        // Each hop reduces signal
        assert!(d1 < d0);
        assert!(d2 < d1);
        assert!(d4 < d2);
        // Depth 4 should still be above 0.25
        assert!(d4 > 0.25);
    }

    #[test]
    fn test_entity_type_from_str() {
        assert_eq!(entity_type_from_str("file"), EntitySource::File);
        assert_eq!(entity_type_from_str("File"), EntitySource::File);
        assert_eq!(entity_type_from_str("function"), EntitySource::Function);
        assert_eq!(entity_type_from_str("struct"), EntitySource::Struct);
        assert_eq!(entity_type_from_str("class"), EntitySource::Struct);
        assert_eq!(entity_type_from_str("trait"), EntitySource::Trait);
        assert_eq!(entity_type_from_str("interface"), EntitySource::Trait);
        assert_eq!(entity_type_from_str("unknown"), EntitySource::File); // fallback
    }

    #[test]
    fn test_generate_action_note() {
        let action = generate_action(EntitySource::Note, "abc-123", 0.9).unwrap();
        assert_eq!(action.tool, "note");
        assert_eq!(action.action, "get");
        assert_eq!(action.params["note_id"], "abc-123");
        assert!((action.confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_generate_action_file() {
        let action = generate_action(EntitySource::File, "src/main.rs", 0.7).unwrap();
        assert_eq!(action.tool, "code");
        assert_eq!(action.action, "get_file_symbols");
        assert_eq!(action.params["file_path"], "src/main.rs");
    }

    #[test]
    fn test_generate_action_decision() {
        let action = generate_action(EntitySource::Decision, "dec-456", 0.8).unwrap();
        assert_eq!(action.tool, "decision");
        assert_eq!(action.action, "get");
        assert_eq!(action.params["decision_id"], "dec-456");
    }

    #[test]
    fn test_generate_action_skill() {
        let action = generate_action(EntitySource::Skill, "skill-789", 0.85).unwrap();
        assert_eq!(action.tool, "skill");
        assert_eq!(action.action, "activate");
        assert_eq!(action.params["skill_id"], "skill-789");
    }

    #[test]
    fn test_format_seed_reasoning() {
        let seed = SeedNode {
            entity_type: EntitySource::Note,
            entity_id: "n1".to_string(),
            label: "test".to_string(),
            score: 0.85,
            source: SeedSource::NoteVectorSearch,
        };
        let reasoning = format_seed_reasoning(&seed);
        assert!(reasoning.contains("vector search"));
        assert!(reasoning.contains("0.85"));

        let seed2 = SeedNode {
            entity_type: EntitySource::Skill,
            entity_id: "s1".to_string(),
            label: "test skill".to_string(),
            score: 0.9,
            source: SeedSource::SkillTriggerMatch {
                skill_name: "wave-execution".to_string(),
                trigger_confidence: 0.8,
            },
        };
        let reasoning2 = format_seed_reasoning(&seed2);
        assert!(reasoning2.contains("wave-execution"));
        assert!(reasoning2.contains("trigger pattern"));
    }

    // ====================================================================
    // Engine integration tests (with MockGraphStore)
    // ====================================================================

    #[tokio::test]
    async fn test_build_tree_with_mock() {
        let engine = make_engine();
        let config = ReasoningTreeConfig::default();

        let tree = engine.build("test query", None, &config).await.unwrap();

        assert_eq!(tree.request, "test query");
        assert!(tree.request_embedding.is_some());
        assert!(tree.build_time_ms.is_some());
    }

    #[tokio::test]
    async fn test_build_tree_caching() {
        let engine = make_engine();
        let config = ReasoningTreeConfig::default();

        let tree1 = engine.build("cached query", None, &config).await.unwrap();
        let tree2 = engine.build("cached query", None, &config).await.unwrap();

        // Second call should return cached tree (same ID)
        assert_eq!(tree1.id, tree2.id);
    }

    #[tokio::test]
    async fn test_build_tree_different_queries() {
        let engine = make_engine();
        let config = ReasoningTreeConfig::default();

        let tree1 = engine.build("query A", None, &config).await.unwrap();
        let tree2 = engine.build("query B", None, &config).await.unwrap();

        // Different queries → different trees
        assert_ne!(tree1.id, tree2.id);
    }

    #[tokio::test]
    async fn test_cache_invalidation() {
        let engine = make_engine();
        let config = ReasoningTreeConfig::default();

        let tree = engine.build("to invalidate", None, &config).await.unwrap();
        let tree_id = tree.id;

        engine.cache().invalidate(tree_id).await;

        let tree2 = engine.build("to invalidate", None, &config).await.unwrap();

        // After invalidation, a new tree should be built (different ID)
        assert_ne!(tree_id, tree2.id);
    }

    #[tokio::test]
    async fn test_empty_seeds_produce_empty_tree() {
        let engine = make_engine();
        let config = ReasoningTreeConfig::default();

        // MockGraphStore returns no vector search results → empty tree
        let tree = engine.build("obscure query", None, &config).await.unwrap();

        assert_eq!(tree.node_count, 0);
        assert_eq!(tree.depth, 0);
        assert!((tree.confidence - 0.0).abs() < f64::EPSILON);
        assert!(tree.suggested_actions().is_empty());
    }

    // ====================================================================
    // Phase 2 unit tests (propagation logic)
    // ====================================================================

    #[tokio::test]
    async fn test_propagation_with_no_seeds() {
        let engine = make_engine();
        let config = ReasoningTreeConfig::default();

        let activated = engine.phase2_propagate(&[], &config).await.unwrap();
        assert!(activated.is_empty());
    }

    #[tokio::test]
    async fn test_propagation_respects_max_nodes() {
        let engine = make_engine();
        let config = ReasoningTreeConfig {
            max_nodes: 5,
            ..Default::default()
        };

        // Create many seeds
        let seeds: Vec<SeedNode> = (0..10)
            .map(|i| SeedNode {
                entity_type: EntitySource::Note,
                entity_id: Uuid::new_v4().to_string(),
                label: format!("seed {}", i),
                score: 0.9 - (i as f64 * 0.05),
                source: SeedSource::NoteVectorSearch,
            })
            .collect();

        let activated = engine.phase2_propagate(&seeds, &config).await.unwrap();
        assert!(activated.len() <= config.max_nodes);
    }

    // ====================================================================
    // Phase 3 unit tests (cristallisation logic)
    // ====================================================================

    #[test]
    fn test_cristallisation_groups_by_seed() {
        // This test verifies the grouping logic used in Phase 3
        let seed1_id = Uuid::new_v4().to_string();
        let seed2_id = Uuid::new_v4().to_string();

        let activated = vec![
            ActivatedEntity {
                entity_type: EntitySource::Note,
                entity_id: seed1_id.clone(),
                label: "Seed 1".to_string(),
                relevance: 0.9,
                depth: 0,
                seed_id: seed1_id.clone(),
                reasoning: "Root".to_string(),
                _relation: ActivationRelation::Seed,
            },
            ActivatedEntity {
                entity_type: EntitySource::File,
                entity_id: "src/main.rs".to_string(),
                label: "src/main.rs".to_string(),
                relevance: 0.7,
                depth: 1,
                seed_id: seed1_id.clone(),
                reasoning: "Linked file".to_string(),
                _relation: ActivationRelation::LinkedTo,
            },
            ActivatedEntity {
                entity_type: EntitySource::Note,
                entity_id: seed2_id.clone(),
                label: "Seed 2".to_string(),
                relevance: 0.8,
                depth: 0,
                seed_id: seed2_id.clone(),
                reasoning: "Root".to_string(),
                _relation: ActivationRelation::Seed,
            },
        ];

        // Group by seed
        let mut by_seed: HashMap<String, Vec<&ActivatedEntity>> = HashMap::new();
        for entity in &activated {
            by_seed
                .entry(entity.seed_id.clone())
                .or_default()
                .push(entity);
        }

        // Seed 1 should have 2 entities (itself + file child)
        assert_eq!(by_seed.get(&seed1_id).unwrap().len(), 2);
        // Seed 2 should have 1 entity (itself only)
        assert_eq!(by_seed.get(&seed2_id).unwrap().len(), 1);
    }
}
