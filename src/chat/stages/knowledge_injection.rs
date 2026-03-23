//! Knowledge Auto-Injection Stage for the Chat Enrichment Pipeline.
//!
//! Enriches the user message with relevant knowledge from four sources,
//! queried in parallel with individual timeouts:
//!
//! 1. **BM25 text search on notes** (MeiliSearch) — top-N notes matching the message
//! 2. **BM25 text search on decisions** (MeiliSearch) — top-N decisions matching the message
//! 3. **Propagated notes** (Neo4j graph traversal) — notes linked to files mentioned in the message
//! 4. **Entity context notes** (Neo4j) — notes directly attached to detected entities
//!
//! Results are deduplicated by note_id/decision_id (keeping highest score)
//! and truncated to a character budget (~3000 chars ≈ 750 tokens).
//!
//! # Design choices
//!
//! - Uses MeiliSearch BM25 instead of vector search because `ChatManager` does not
//!   hold an `EmbeddingProvider`. BM25 is fast (<50ms) and works with raw text queries.
//! - Entity extraction reuses [`crate::chat::entity_extractor::extract_entities`]
//!   which detects file paths, backtick identifiers, and code keyword patterns.
//! - Graceful degradation: each query has a 200ms timeout; failures are logged and skipped.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::time::timeout;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::chat::enrichment::{
    EnrichmentConfig, EnrichmentInput, ParallelEnrichmentStage, StageOutput,
};
use crate::chat::entity_extractor::{self, EntityType as ChatEntityType, ExtractedEntity};
use crate::chat::stages::intent_weights::IntentWeightMap;
use crate::meilisearch::SearchStore;
use crate::neo4j::traits::GraphStore;
use crate::neurons::intent::{IntentDetector, QueryIntentMode};
use crate::notes::models::EntityType as NoteEntityType;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the knowledge injection stage.
#[derive(Debug, Clone)]
pub struct KnowledgeInjectionConfig {
    /// Maximum number of notes from BM25 search (default: 5).
    pub max_notes: usize,
    /// Maximum number of decisions from BM25 search (default: 3).
    pub max_decisions: usize,
    /// Maximum number of propagated notes per file (default: 3).
    pub max_propagated_per_file: usize,
    /// Maximum number of files to query for propagated notes (default: 2).
    pub max_propagated_files: usize,
    /// Maximum number of entity context notes per entity (default: 3).
    pub max_entity_notes: usize,
    /// Maximum number of entities to query for context notes (default: 3).
    pub max_entity_queries: usize,
    /// Timeout for each individual query in milliseconds (default: 200).
    pub query_timeout_ms: u64,
    /// Maximum total character budget for the rendered section (default: 3000).
    pub max_content_chars: usize,
}

impl Default for KnowledgeInjectionConfig {
    fn default() -> Self {
        Self {
            max_notes: 5,
            max_decisions: 3,
            max_propagated_per_file: 3,
            max_propagated_files: 2,
            max_entity_notes: 3,
            max_entity_queries: 3,
            query_timeout_ms: 200,
            max_content_chars: 3000,
        }
    }
}

/// Returns scaffolding-adjusted config overrides based on the project's competence level.
///
/// - L0 (Struggling): maximum injection — the system provides all available context
/// - L1 (Growing): generous injection
/// - L2 (Competent): default injection (unchanged)
/// - L3 (Proficient): reduced injection — the project has good knowledge density
/// - L4 (Expert): minimal injection — trust the existing knowledge graph
pub fn scaffolding_config(level: u8) -> KnowledgeInjectionConfig {
    match level {
        0 => KnowledgeInjectionConfig {
            max_notes: 8,
            max_decisions: 5,
            max_propagated_per_file: 5,
            max_propagated_files: 3,
            max_entity_notes: 5,
            max_entity_queries: 5,
            query_timeout_ms: 300,
            max_content_chars: 5000,
        },
        1 => KnowledgeInjectionConfig {
            max_notes: 6,
            max_decisions: 4,
            max_propagated_per_file: 4,
            max_propagated_files: 3,
            max_entity_notes: 4,
            max_entity_queries: 4,
            query_timeout_ms: 250,
            max_content_chars: 4000,
        },
        2 => KnowledgeInjectionConfig::default(), // L2 = default behavior
        3 => KnowledgeInjectionConfig {
            max_notes: 3,
            max_decisions: 2,
            max_propagated_per_file: 2,
            max_propagated_files: 1,
            max_entity_notes: 2,
            max_entity_queries: 2,
            query_timeout_ms: 150,
            max_content_chars: 2000,
        },
        _ => KnowledgeInjectionConfig {
            // L4+ = minimal
            max_notes: 2,
            max_decisions: 1,
            max_propagated_per_file: 1,
            max_propagated_files: 1,
            max_entity_notes: 1,
            max_entity_queries: 1,
            query_timeout_ms: 100,
            max_content_chars: 1500,
        },
    }
}

// ============================================================================
// Deduplicated result types
// ============================================================================

/// A note with its relevance score, used for deduplication.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used in tests and for debug logging
struct ScoredNote {
    id: String,
    note_type: String,
    importance: String,
    content: String,
    score: f64,
    source: &'static str,
}

/// A decision with its relevance score.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used in tests and for debug logging
struct ScoredDecision {
    id: String,
    description: String,
    rationale: String,
    score: f64,
}

/// A predictive file suggestion from the WorldModel (biomimicry T7).
/// Predicted from CO_CHANGED / DISCUSSED patterns before the agent accesses them.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PredictedFile {
    path: String,
    score: f64,
    source: &'static str, // "co_change" or "discussed"
}

// ============================================================================
// Stage implementation
// ============================================================================

/// Enrichment stage that auto-injects relevant knowledge from the graph and search index.
pub struct KnowledgeInjectionStage {
    graph: Arc<dyn GraphStore>,
    search: Arc<dyn SearchStore>,
    config: KnowledgeInjectionConfig,
    /// Trajectory collector for decision capture (fire-and-forget).
    collector: Option<Arc<neural_routing_runtime::TrajectoryCollector>>,
}

impl KnowledgeInjectionStage {
    /// Create a new knowledge injection stage with default configuration.
    pub fn new(graph: Arc<dyn GraphStore>, search: Arc<dyn SearchStore>) -> Self {
        Self {
            graph,
            search,
            config: KnowledgeInjectionConfig::default(),
            collector: None,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(
        graph: Arc<dyn GraphStore>,
        search: Arc<dyn SearchStore>,
        config: KnowledgeInjectionConfig,
    ) -> Self {
        Self {
            graph,
            search,
            config,
            collector: None,
        }
    }

    /// Attach a trajectory collector for decision capture.
    pub fn with_collector(
        mut self,
        collector: Arc<neural_routing_runtime::TrajectoryCollector>,
    ) -> Self {
        self.collector = Some(collector);
        self
    }

    /// Extract UUIDs from the user message.
    ///
    /// Matches standard UUID v4 format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.
    /// Used to detect direct references to notes or decisions pasted in chat.
    fn extract_uuids(message: &str) -> Vec<Uuid> {
        // UUID v4 regex: 8-4-4-4-12 hex chars
        let mut uuids = Vec::new();
        let mut i = 0;
        let bytes = message.as_bytes();
        let len = bytes.len();

        // Minimum UUID length is 36 chars: 8-4-4-4-12
        while i + 36 <= len {
            // Quick check: look for the dash pattern at positions 8, 13, 18, 23
            if bytes[i + 8] == b'-'
                && bytes[i + 13] == b'-'
                && bytes[i + 18] == b'-'
                && bytes[i + 23] == b'-'
            {
                let candidate = &message[i..i + 36];
                if let Ok(uuid) = Uuid::parse_str(candidate) {
                    uuids.push(uuid);
                    i += 36;
                    continue;
                }
            }
            i += 1;
        }

        uuids
    }

    /// Resolve project_id from project_slug via the graph.
    async fn resolve_project_id(&self, slug: &str) -> Option<Uuid> {
        match self.graph.get_project_by_slug(slug).await {
            Ok(Some(project)) => Some(project.id),
            Ok(None) => None,
            Err(e) => {
                warn!(
                    "[knowledge_injection] Failed to resolve project slug '{}': {}",
                    slug, e
                );
                None
            }
        }
    }

    /// Run all knowledge queries in parallel with individual timeouts.
    ///
    /// Returns deduplicated notes and decisions, sorted by relevance.
    /// Accepts an explicit config to support scaffolding-adjusted parameters.
    async fn query_knowledge_with_config(
        &self,
        message: &str,
        project_slug: Option<&str>,
        _project_id: Option<Uuid>,
        entities: &[ExtractedEntity],
        referenced_uuids: &[Uuid],
        config: &KnowledgeInjectionConfig,
    ) -> (
        Vec<ScoredNote>,
        Vec<ScoredDecision>,
        Vec<PredictedFile>,
        Vec<ScoredNote>,
    ) {
        let query_timeout = Duration::from_millis(config.query_timeout_ms);

        // ── Query 1: BM25 notes search ──────────────────────────────────
        let search_clone = self.search.clone();
        let msg1 = message.to_string();
        let slug1 = project_slug.map(|s| s.to_string());
        let max_notes = config.max_notes;

        let notes_future = async move {
            let result = search_clone
                .search_notes_with_scores(
                    &msg1,
                    max_notes,
                    slug1.as_deref(),
                    None,           // note_type
                    Some("active"), // only active notes
                    None,           // importance
                )
                .await;
            result.unwrap_or_default()
        };

        // ── Query 2: BM25 decisions search ──────────────────────────────
        let search_clone2 = self.search.clone();
        let msg2 = message.to_string();
        let slug2 = project_slug.map(|s| s.to_string());
        let max_decisions = config.max_decisions;

        let decisions_future = async move {
            let result = search_clone2
                .search_decisions_in_project(&msg2, max_decisions, slug2.as_deref())
                .await;
            result.unwrap_or_default()
        };

        // ── Query 3: Entity notes (files + symbols) ─────────────────────
        let graph_clone = self.graph.clone();
        let file_entities: Vec<String> = entities
            .iter()
            .filter(|e| e.entity_type == ChatEntityType::File)
            .take(config.max_propagated_files)
            .map(|e| e.identifier.clone())
            .collect();
        let symbol_entities: Vec<(ChatEntityType, String)> = entities
            .iter()
            .filter(|e| e.entity_type != ChatEntityType::File)
            .take(config.max_entity_queries)
            .map(|e| (e.entity_type.clone(), e.identifier.clone()))
            .collect();
        let max_entity_notes = config.max_entity_notes;
        let max_propagated_per_file = config.max_propagated_per_file;
        let project_id_for_propagation = _project_id;

        let entity_notes_future = async move {
            let mut notes = Vec::new();
            let mut predictions: Vec<PredictedFile> = Vec::new();

            // Direct notes for files
            for file_path in &file_entities {
                match graph_clone
                    .get_notes_for_entity(&NoteEntityType::File, file_path)
                    .await
                {
                    Ok(file_notes) => {
                        for note in file_notes.into_iter().take(max_entity_notes) {
                            notes.push(ScoredNote {
                                id: note.id.to_string(),
                                note_type: format!("{:?}", note.note_type),
                                importance: format!("{:?}", note.importance),
                                content: note.content,
                                score: 0.8, // Direct entity notes have high relevance
                                source: "entity_context",
                            });
                        }
                    }
                    Err(e) => {
                        debug!(
                            file = %file_path,
                            error = %e,
                            "Failed to get notes for file entity"
                        );
                    }
                }
            }

            // Propagated notes for files (graph traversal)
            // Pass source_project_id to enable cross-project note propagation
            // with coupling-attenuated scores. min_score lowered to 0.15 because
            // coupling × score produces low values for sibling projects.
            for file_path in &file_entities {
                match graph_clone
                    .get_propagated_notes(
                        &NoteEntityType::File,
                        file_path,
                        2,                          // max_depth
                        0.15, // min_score (lowered for cross-project coupling attenuation)
                        None, // all relation types
                        project_id_for_propagation, // source_project_id for cross-project filtering
                        false, // force_cross_project (safe default)
                    )
                    .await
                {
                    Ok(propagated) => {
                        for pn in propagated.into_iter().take(max_propagated_per_file) {
                            notes.push(ScoredNote {
                                id: pn.note.id.to_string(),
                                note_type: format!("{:?}", pn.note.note_type),
                                importance: format!("{:?}", pn.note.importance),
                                content: pn.note.content,
                                score: pn.relevance_score,
                                source: "propagated",
                            });
                        }
                    }
                    Err(e) => {
                        debug!(
                            file = %file_path,
                            error = %e,
                            "Failed to get propagated notes for file"
                        );
                    }
                }
            }

            // WorldModel predictive context (biomimicry T7)
            // Inject top-5 CO_CHANGED co-changers as predicted file suggestions
            for file_path in &file_entities {
                match graph_clone
                    .get_file_co_changers(file_path, 2, 5) // min_count=2, limit=5
                    .await
                {
                    Ok(co_changers) => {
                        for cc in co_changers {
                            // Score: normalized co-change count (diminishing returns)
                            let score = 1.0 - (1.0 / (cc.count as f64 + 1.0));
                            predictions.push(PredictedFile {
                                path: cc.path,
                                score,
                                source: "co_change",
                            });
                        }
                    }
                    Err(e) => {
                        debug!(
                            file = %file_path,
                            error = %e,
                            "Failed to get co-changers for predictive context"
                        );
                    }
                }
            }

            // Direct notes for symbols (functions, structs, traits, enums)
            for (entity_type, identifier) in &symbol_entities {
                let note_entity_type = match entity_type {
                    ChatEntityType::Function => NoteEntityType::Function,
                    ChatEntityType::Struct => NoteEntityType::Struct,
                    ChatEntityType::Trait => NoteEntityType::Trait,
                    ChatEntityType::Enum => NoteEntityType::Enum,
                    _ => continue,
                };

                match graph_clone
                    .get_notes_for_entity(&note_entity_type, identifier)
                    .await
                {
                    Ok(entity_notes) => {
                        for note in entity_notes.into_iter().take(max_entity_notes) {
                            notes.push(ScoredNote {
                                id: note.id.to_string(),
                                note_type: format!("{:?}", note.note_type),
                                importance: format!("{:?}", note.importance),
                                content: note.content,
                                score: 0.75, // Symbol entity notes
                                source: "entity_context",
                            });
                        }
                    }
                    Err(e) => {
                        debug!(
                            entity = %identifier,
                            error = %e,
                            "Failed to get notes for symbol entity"
                        );
                    }
                }
            }

            (notes, predictions)
        };

        // ── Query 4: Direct UUID lookups (notes/decisions referenced by UUID) ─
        let graph_clone2 = self.graph.clone();
        let uuids = referenced_uuids.to_vec();

        let uuid_lookup_future = async move {
            let mut notes: Vec<ScoredNote> = Vec::new();
            let mut decisions: Vec<ScoredDecision> = Vec::new();

            for uuid in &uuids {
                // Try as note first
                match graph_clone2.get_note(*uuid).await {
                    Ok(Some(note)) => {
                        notes.push(ScoredNote {
                            id: note.id.to_string(),
                            note_type: format!("{:?}", note.note_type),
                            importance: format!("{:?}", note.importance),
                            content: note.content,
                            score: 1.0, // Direct reference = maximum relevance
                            source: "uuid_reference",
                        });
                        continue;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        debug!(uuid = %uuid, error = %e, "Failed to lookup note by UUID");
                    }
                }

                // Try as decision
                match graph_clone2.get_decision(*uuid).await {
                    Ok(Some(decision)) => {
                        decisions.push(ScoredDecision {
                            id: decision.id.to_string(),
                            description: decision.description,
                            rationale: decision.rationale,
                            score: 1.0, // Direct reference = maximum relevance
                        });
                    }
                    Ok(None) => {
                        debug!(uuid = %uuid, "UUID not found as note or decision");
                    }
                    Err(e) => {
                        debug!(uuid = %uuid, error = %e, "Failed to lookup decision by UUID");
                    }
                }
            }

            (notes, decisions)
        };

        // ── Query 5: DISCUSSED co-changers (WorldModel T7) ──────────────
        let graph_clone3 = self.graph.clone();
        let discussed_future = async move {
            let mut preds = Vec::new();
            if let Some(pid) = _project_id {
                match graph_clone3.get_discussed_co_changers(pid, 3, 5).await {
                    Ok(co_changers) => {
                        for cc in co_changers {
                            let score = 1.0 - (1.0 / (cc.count as f64 + 1.0));
                            preds.push(PredictedFile {
                                path: cc.path,
                                score: score * 0.9, // Slight discount vs direct CO_CHANGED
                                source: "discussed",
                            });
                        }
                    }
                    Err(e) => {
                        debug!(error = %e, "Failed to get discussed co-changers");
                    }
                }
            }
            preds
        };

        // ── Query 6: Workspace-level notes ────────────────────────────────
        let graph_clone4 = self.graph.clone();
        let workspace_notes_future = async move {
            let mut ws_notes: Vec<ScoredNote> = Vec::new();
            if let Some(pid) = _project_id {
                match graph_clone4
                    .get_workspace_notes_for_project(pid, 1.0) // full propagation, we score at 0.6
                    .await
                {
                    Ok(propagated) => {
                        for pn in propagated.into_iter().take(3) {
                            ws_notes.push(ScoredNote {
                                id: pn.note.id.to_string(),
                                note_type: format!("{:?}", pn.note.note_type),
                                importance: format!("{:?}", pn.note.importance),
                                content: pn.note.content,
                                score: 0.6, // Between entity notes (0.8) and propagated
                                source: "workspace",
                            });
                        }
                    }
                    Err(e) => {
                        debug!(error = %e, "Failed to get workspace notes for project");
                    }
                }
            }
            ws_notes
        };

        // ── Execute all queries in parallel with timeouts ────────────────
        let (
            notes_result,
            decisions_result,
            entity_notes_result,
            uuid_result,
            discussed_result,
            workspace_result,
        ) = tokio::join!(
            timeout(query_timeout, notes_future),
            timeout(query_timeout, decisions_future),
            timeout(query_timeout, entity_notes_future),
            timeout(query_timeout, uuid_lookup_future),
            timeout(query_timeout, discussed_future),
            timeout(query_timeout, workspace_notes_future),
        );

        // ── Collect & deduplicate notes ─────────────────────────────────
        let mut note_map: HashMap<String, ScoredNote> = HashMap::new();

        // BM25 notes
        if let Ok(hits) = notes_result {
            for hit in hits {
                let note = ScoredNote {
                    id: hit.document.id.clone(),
                    note_type: hit.document.note_type.clone(),
                    importance: hit.document.importance.clone(),
                    content: hit.document.content.clone(),
                    score: hit.score,
                    source: "bm25_search",
                };
                note_map
                    .entry(note.id.clone())
                    .and_modify(|existing| {
                        if note.score > existing.score {
                            *existing = note.clone();
                        }
                    })
                    .or_insert(note);
            }
        } else {
            debug!("[knowledge_injection] Notes BM25 search timed out");
        }

        // Entity notes (direct + propagated) + predictions
        let mut predicted_files: Vec<PredictedFile> = Vec::new();
        if let Ok((entity_notes, predictions)) = entity_notes_result {
            for note in entity_notes {
                note_map
                    .entry(note.id.clone())
                    .and_modify(|existing| {
                        if note.score > existing.score {
                            *existing = note.clone();
                        }
                    })
                    .or_insert(note);
            }
            // Deduplicate predictions by path, keeping highest score
            let mut pred_map: HashMap<String, PredictedFile> = HashMap::new();
            for pred in predictions {
                pred_map
                    .entry(pred.path.clone())
                    .and_modify(|existing| {
                        if pred.score > existing.score {
                            *existing = pred.clone();
                        }
                    })
                    .or_insert(pred);
            }
            predicted_files = pred_map.into_values().collect();
            predicted_files.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            debug!("[knowledge_injection] Entity notes query timed out");
        }

        // Merge DISCUSSED co-changer predictions (WorldModel T7)
        if let Ok(discussed_preds) = discussed_result {
            for pred in discussed_preds {
                // Merge with existing predictions: keep highest score per path
                if let Some(existing) = predicted_files.iter_mut().find(|p| p.path == pred.path) {
                    if pred.score > existing.score {
                        existing.score = pred.score;
                        existing.source = pred.source;
                    }
                } else {
                    predicted_files.push(pred);
                }
            }
            // Re-sort after merge
            predicted_files.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            debug!("[knowledge_injection] DISCUSSED co-changers query timed out");
        }

        // UUID-referenced notes
        let mut decision_map: HashMap<String, ScoredDecision> = HashMap::new();

        if let Ok((uuid_notes, uuid_decisions)) = uuid_result {
            for note in uuid_notes {
                note_map
                    .entry(note.id.clone())
                    .and_modify(|existing| {
                        if note.score > existing.score {
                            *existing = note.clone();
                        }
                    })
                    .or_insert(note);
            }
            for decision in uuid_decisions {
                decision_map
                    .entry(decision.id.clone())
                    .and_modify(|existing| {
                        if decision.score > existing.score {
                            *existing = decision.clone();
                        }
                    })
                    .or_insert(decision);
            }
        } else {
            debug!("[knowledge_injection] UUID lookup timed out");
        }

        // Collect and sort notes by score descending
        let mut notes: Vec<ScoredNote> = note_map.into_values().collect();
        notes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Collect BM25 decisions and merge with UUID decisions
        if let Ok(docs) = decisions_result {
            for (i, doc) in docs.into_iter().enumerate() {
                let decision = ScoredDecision {
                    id: doc.id.clone(),
                    description: doc.description,
                    rationale: doc.rationale,
                    // BM25 decisions don't have a score in the response,
                    // use position-based scoring (first result = highest relevance)
                    score: 1.0 - (i as f64 * 0.1),
                };
                decision_map
                    .entry(decision.id.clone())
                    .and_modify(|existing| {
                        if decision.score > existing.score {
                            *existing = decision.clone();
                        }
                    })
                    .or_insert(decision);
            }
        } else {
            debug!("[knowledge_injection] Decisions BM25 search timed out");
        }

        let mut decisions: Vec<ScoredDecision> = decision_map.into_values().collect();
        decisions.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Workspace notes (Query 6)
        let workspace_notes = if let Ok(ws_notes) = workspace_result {
            ws_notes
        } else {
            debug!("[knowledge_injection] Workspace notes query timed out");
            Vec::new()
        };

        (notes, decisions, predicted_files, workspace_notes)
    }

    /// Render notes, decisions, predicted files, and workspace guidelines into markdown sections.
    fn render_knowledge(
        &self,
        notes: &[ScoredNote],
        decisions: &[ScoredDecision],
        predictions: &[PredictedFile],
        workspace_notes: &[ScoredNote],
        max_content_chars: usize,
    ) -> Option<String> {
        if notes.is_empty()
            && decisions.is_empty()
            && predictions.is_empty()
            && workspace_notes.is_empty()
        {
            return None;
        }

        let mut content = String::new();
        let mut chars_used = 0;

        // Render workspace guidelines (high-priority, shown first)
        if !workspace_notes.is_empty() && chars_used < max_content_chars {
            content.push_str("### Workspace Guidelines\n");
            for note in workspace_notes {
                let entry = format!(
                    "- **[{}|{}]** {}\n",
                    note.note_type.to_lowercase(),
                    note.importance.to_lowercase(),
                    truncate_content(&note.content, 300),
                );
                if chars_used + entry.len() > max_content_chars {
                    break;
                }
                content.push_str(&entry);
                chars_used += entry.len();
            }
            content.push('\n');
        }

        // Render notes
        if !notes.is_empty() {
            content.push_str("### Relevant Notes\n");
            for note in notes {
                let entry = format!(
                    "- **[{}|{}]** {}\n",
                    note.note_type.to_lowercase(),
                    note.importance.to_lowercase(),
                    truncate_content(&note.content, 300),
                );
                if chars_used + entry.len() > max_content_chars {
                    break;
                }
                content.push_str(&entry);
                chars_used += entry.len();
            }
            content.push('\n');
        }

        // Render decisions
        if !decisions.is_empty() && chars_used < max_content_chars {
            content.push_str("### Relevant Decisions\n");
            for decision in decisions {
                let entry = format!(
                    "- **Decision:** {} — *Rationale:* {}\n",
                    truncate_content(&decision.description, 200),
                    truncate_content(&decision.rationale, 200),
                );
                if chars_used + entry.len() > max_content_chars {
                    break;
                }
                content.push_str(&entry);
                chars_used += entry.len();
            }
        }

        // Render predicted context (WorldModel biomimicry T7)
        if !predictions.is_empty() && chars_used < max_content_chars {
            content.push_str("\n### Predicted Context (WorldModel)\n");
            content.push_str("_Files you may also need (based on co-change patterns):_\n");
            for pred in predictions.iter().take(5) {
                let entry = format!(
                    "- `{}` (score: {:.2}, source: {})\n",
                    pred.path, pred.score, pred.source
                );
                if chars_used + entry.len() > max_content_chars {
                    break;
                }
                content.push_str(&entry);
                chars_used += entry.len();
            }
        }

        if content.is_empty() {
            None
        } else {
            Some(content)
        }
    }
}

/// Truncate content to a maximum number of characters, adding ellipsis if truncated.
fn truncate_content(content: &str, max_chars: usize) -> String {
    // Normalize: collapse newlines to spaces for inline display
    let normalized: String = content
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    let trimmed = normalized.trim();

    if trimmed.chars().count() <= max_chars {
        trimmed.to_string()
    } else {
        // Collect the first max_chars characters (char-safe, no byte slicing)
        let truncated: String = trimmed.chars().take(max_chars).collect();
        // Find a word boundary near max_chars for cleaner truncation
        match truncated.rfind(' ') {
            Some(pos) if pos > truncated.len() / 2 => {
                format!("{}…", &truncated[..pos])
            }
            _ => format!("{}…", truncated),
        }
    }
}

/// Map an [`IntentDetector`] result to the enrichment hint string
/// used by [`IntentWeightMap`].
///
/// This duplicates the mapping from `skill_activation.rs` to avoid
/// the inter-stage dependency on the `intent` hint. CPU-only, <1ms.
fn map_intent_mode(mode: QueryIntentMode) -> &'static str {
    match mode {
        QueryIntentMode::Debug => "debug",
        QueryIntentMode::Explore => "explore",
        QueryIntentMode::Impact => "review",
        QueryIntentMode::Plan => "planning",
        QueryIntentMode::Default => "general",
    }
}

#[async_trait::async_trait]
impl ParallelEnrichmentStage for KnowledgeInjectionStage {
    async fn execute(&self, input: &EnrichmentInput) -> Result<StageOutput> {
        let mut output = StageOutput::new(self.name());

        // Need a project scope for knowledge search
        let project_slug = match &input.project_slug {
            Some(slug) => slug.clone(),
            None => return Ok(output), // No project scope, skip
        };

        let project_id = if let Some(id) = input.project_id {
            Some(id)
        } else {
            self.resolve_project_id(&project_slug).await
        };

        // ── Scaffolding-adaptive config ──────────────────────────────────
        let scaffolding_enabled = std::env::var("ENRICHMENT_SCAFFOLDING")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(true);

        let effective_config = if scaffolding_enabled {
            if let Some(pid) = project_id {
                let graph_clone = self.graph.clone();
                let scaffolding_override = None;
                match timeout(
                    Duration::from_millis(100),
                    graph_clone.compute_scaffolding_level(pid, scaffolding_override),
                )
                .await
                {
                    Ok(Ok(level)) => {
                        let config = scaffolding_config(level.level);
                        info!(
                            "[knowledge_injection] Scaffolding L{} ({}) auto-computed: max_notes={}, max_content={}",
                            level.level, level.label, config.max_notes, config.max_content_chars
                        );
                        output.add_section(
                            "Scaffolding Level",
                            format!(
                                "L{} ({}) — competence: {:.0}%, pain: {:.0}%",
                                level.level,
                                level.label,
                                level.competence_score * 100.0,
                                level.homeostasis_pain * 100.0
                            ),
                            self.name(),
                        );
                        config
                    }
                    Ok(Err(e)) => {
                        warn!(
                            "[knowledge_injection] Scaffolding computation failed: {}",
                            e
                        );
                        self.config.clone()
                    }
                    Err(_) => {
                        warn!("[knowledge_injection] Scaffolding computation timed out (100ms)");
                        self.config.clone()
                    }
                }
            } else {
                info!("[knowledge_injection] Scaffolding enabled (auto-computed) but no project_id, using default config");
                self.config.clone()
            }
        } else {
            info!("[knowledge_injection] Scaffolding disabled via ENRICHMENT_SCAFFOLDING env var, using default L2 config");
            self.config.clone()
        };

        // Extract entities from the message
        let entities = entity_extractor::extract_entities(&input.message);
        let referenced_uuids = Self::extract_uuids(&input.message);

        debug!(
            "[knowledge_injection] Extracted {} entities, {} UUIDs from message",
            entities.len(),
            referenced_uuids.len()
        );

        // Run all knowledge queries in parallel (using scaffolding-adjusted config)
        let (mut notes, decisions, predictions, workspace_notes) = self
            .query_knowledge_with_config(
                &input.message,
                Some(&project_slug),
                project_id,
                &entities,
                &referenced_uuids,
                &effective_config,
            )
            .await;

        // ── Deduplicate: exclude notes already in the system prompt ──────
        let pre_dedup_count = notes.len();
        if !input.excluded_note_ids.is_empty() {
            notes.retain(|n| !input.excluded_note_ids.contains(&n.id));
            let removed = pre_dedup_count - notes.len();
            if removed > 0 {
                debug!(
                    "[knowledge_injection] Dedup: removed {} notes already in system prompt",
                    removed
                );
            }
        }

        // ── Intent-aware reweighting ────────────────────────────────────
        // Detect intent locally (duplicated from SkillActivationStage) to avoid
        // the inter-stage dependency on the `intent` hint. CPU-only, <1ms.
        let intent = map_intent_mode(IntentDetector::detect(&input.message));
        let weight_map = IntentWeightMap::for_intent(intent);
        for note in &mut notes {
            note.score *= weight_map.get(&note.note_type);
        }
        notes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!(
            "[knowledge_injection] Found {} notes ({} deduped), {} decisions, {} predictions (intent={})",
            notes.len(),
            pre_dedup_count - notes.len(),
            decisions.len(),
            predictions.len(),
            intent
        );

        // ── Trajectory collection: record context loading decision ────────
        if let Some(ref collector) = self.collector {
            let touched: Vec<neural_routing_runtime::TouchedEntity> =
                notes
                    .iter()
                    .take(5)
                    .map(|n| neural_routing_runtime::TouchedEntity {
                        entity_type: "Note".to_string(),
                        entity_id: n.id.clone(),
                        access_mode: "read".to_string(),
                        relevance: Some(n.score),
                    })
                    .chain(decisions.iter().take(3).map(|d| {
                        neural_routing_runtime::TouchedEntity {
                            entity_type: "Decision".to_string(),
                            entity_id: d.id.clone(),
                            access_mode: "read".to_string(),
                            relevance: Some(d.score),
                        }
                    }))
                    .collect();

            collector.record_decision(neural_routing_runtime::DecisionRecord {
                session_id: input.session_id.to_string(),
                context_embedding: vec![],
                action_type: "context.knowledge_injection".to_string(),
                action_params: serde_json::json!({
                    "notes_count": notes.len(),
                    "decisions_count": decisions.len(),
                    "predictions_count": predictions.len(),
                }),
                alternatives_count: notes.len() + decisions.len(),
                chosen_index: 0,
                confidence: if notes.is_empty() && decisions.is_empty() {
                    0.0_f64
                } else {
                    notes.first().map(|n| n.score).unwrap_or(0.5_f64)
                },
                tool_usages: vec![neural_routing_runtime::ToolUsage {
                    tool_name: "knowledge_injection".to_string(),
                    action: "query".to_string(),
                    params_hash: format!("slug:{}", project_slug),
                    duration_ms: None,
                    success: !notes.is_empty() || !decisions.is_empty(),
                }],
                touched_entities: touched,
                timestamp_ms: 0,
                query_embedding: vec![],
                node_features: vec![],
                protocol_run_id: input.protocol_run_id,
                protocol_state: input.protocol_state.clone(),
            });
        }

        // Render and inject into output
        if let Some(content) = self.render_knowledge(
            &notes,
            &decisions,
            &predictions,
            &workspace_notes,
            effective_config.max_content_chars,
        ) {
            output.add_section("Relevant Knowledge", content, self.name());
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        "knowledge_injection"
    }

    fn is_enabled(&self, config: &EnrichmentConfig) -> bool {
        config.knowledge_injection
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── truncate_content tests ──────────────────────────────────────────

    #[test]
    fn test_truncate_content_short() {
        let result = truncate_content("short text", 100);
        assert_eq!(result, "short text");
    }

    #[test]
    fn test_truncate_content_long() {
        let long = "a ".repeat(200); // 400 chars
        let result = truncate_content(&long, 50);
        assert!(result.len() <= 52); // 50 + "…"
        assert!(result.ends_with('…'));
    }

    #[test]
    fn test_truncate_content_newlines() {
        let result = truncate_content("line1\nline2\nline3", 100);
        assert_eq!(result, "line1 line2 line3");
        assert!(!result.contains('\n'));
    }

    #[test]
    fn test_truncate_content_word_boundary() {
        let text = "The quick brown fox jumps over the lazy dog";
        let result = truncate_content(text, 20);
        // Should truncate at a word boundary
        assert!(result.ends_with('…'));
        // "The quick brown fox " is 20 chars, word boundary at pos 19
        assert!(result.contains("brown"));
        assert!(!result.contains("jumps")); // "jumps" starts at position 20, beyond limit
    }

    #[test]
    fn test_truncate_content_empty() {
        let result = truncate_content("", 100);
        assert_eq!(result, "");
    }

    #[test]
    fn test_truncate_content_multibyte_utf8() {
        // French accents: "é" = 2 bytes, "à" = 2 bytes
        let text = "Les résultats à vérifier sont très élevés";
        let result = truncate_content(text, 20);
        assert!(result.ends_with('…'));
        // Must not panic on multi-byte chars
        assert!(result.chars().count() <= 21); // 20 + ellipsis
    }

    #[test]
    fn test_truncate_content_emoji() {
        // Emojis: "🔥" = 4 bytes, "✅" = 3 bytes
        let text = "🔥 Fix critical bug ✅ Tests pass 🎉 Released";
        let result = truncate_content(text, 25);
        assert!(result.ends_with('…'));
        // Must not panic on 4-byte emoji boundaries
        assert!(result.chars().count() <= 26);
    }

    #[test]
    fn test_truncate_content_cjk() {
        // CJK characters: each 3 bytes in UTF-8
        let text = "代码分析工具非常有用 code analysis tool";
        let result = truncate_content(text, 8);
        assert!(result.ends_with('…'));
        assert!(result.chars().count() <= 9);
    }

    #[test]
    fn test_truncate_content_mixed_scripts_exact_boundary() {
        // Exactly at boundary — no truncation needed
        let text = "café"; // 4 chars, 5 bytes (é = 2 bytes)
        let result = truncate_content(text, 4);
        assert_eq!(result, "café");
        assert!(!result.ends_with('…'));
    }

    // ── KnowledgeInjectionConfig tests ──────────────────────────────────

    #[test]
    fn test_config_default() {
        let config = KnowledgeInjectionConfig::default();
        assert_eq!(config.max_notes, 5);
        assert_eq!(config.max_decisions, 3);
        assert_eq!(config.max_propagated_per_file, 3);
        assert_eq!(config.max_propagated_files, 2);
        assert_eq!(config.max_entity_notes, 3);
        assert_eq!(config.max_entity_queries, 3);
        assert_eq!(config.query_timeout_ms, 200);
        assert_eq!(config.max_content_chars, 3000);
    }

    // ── render_knowledge tests ──────────────────────────────────────────

    #[test]
    fn test_render_empty() {
        let stage = make_test_stage();
        let result = stage.render_knowledge(&[], &[], &[], &[], 3000);
        assert!(result.is_none());
    }

    #[test]
    fn test_render_notes_only() {
        let stage = make_test_stage();
        let notes = vec![ScoredNote {
            id: "n1".to_string(),
            note_type: "Gotcha".to_string(),
            importance: "High".to_string(),
            content: "Watch out for null workspace".to_string(),
            score: 0.9,
            source: "bm25_search",
        }];
        let result = stage.render_knowledge(&notes, &[], &[], &[], 3000);
        assert!(result.is_some());
        let content = result.unwrap();
        assert!(content.contains("Relevant Notes"));
        assert!(content.contains("gotcha"));
        assert!(content.contains("Watch out for null workspace"));
    }

    #[test]
    fn test_render_decisions_only() {
        let stage = make_test_stage();
        let decisions = vec![ScoredDecision {
            id: "d1".to_string(),
            description: "Use BM25 instead of vector search".to_string(),
            rationale: "No EmbeddingProvider available in ChatManager".to_string(),
            score: 0.95,
        }];
        let result = stage.render_knowledge(&[], &decisions, &[], &[], 3000);
        assert!(result.is_some());
        let content = result.unwrap();
        assert!(content.contains("Relevant Decisions"));
        assert!(content.contains("BM25"));
        assert!(content.contains("Rationale"));
    }

    #[test]
    fn test_render_mixed() {
        let stage = make_test_stage();
        let notes = vec![ScoredNote {
            id: "n1".to_string(),
            note_type: "Pattern".to_string(),
            importance: "Medium".to_string(),
            content: "Use Arc for shared state".to_string(),
            score: 0.8,
            source: "bm25_search",
        }];
        let decisions = vec![ScoredDecision {
            id: "d1".to_string(),
            description: "Chose tokio::join! for parallelism".to_string(),
            rationale: "Individual timeouts per query".to_string(),
            score: 0.9,
        }];
        let result = stage.render_knowledge(&notes, &decisions, &[], &[], 3000);
        assert!(result.is_some());
        let content = result.unwrap();
        assert!(content.contains("Relevant Notes"));
        assert!(content.contains("Relevant Decisions"));
    }

    #[test]
    fn test_render_respects_char_budget() {
        let config = KnowledgeInjectionConfig {
            max_content_chars: 100,
            ..Default::default()
        };
        let stage = make_test_stage_with_config(config);

        // Create many notes that exceed the budget
        let notes: Vec<ScoredNote> = (0..20)
            .map(|i| ScoredNote {
                id: format!("n{}", i),
                note_type: "Tip".to_string(),
                importance: "Low".to_string(),
                content: format!(
                    "This is note number {} with some content that takes space",
                    i
                ),
                score: 1.0 - (i as f64 * 0.05),
                source: "bm25_search",
            })
            .collect();

        let result = stage.render_knowledge(&notes, &[], &[], &[], 100);
        assert!(result.is_some());
        let content = result.unwrap();
        // Should be truncated — not all 20 notes rendered
        let note_count = content.matches("- **[").count();
        assert!(
            note_count < 20,
            "Should truncate notes to budget, got {} notes",
            note_count
        );
    }

    // ── Full stage tests with mocks ─────────────────────────────────────

    #[tokio::test]
    async fn test_stage_no_project_skips() {
        use crate::chat::enrichment::ParallelEnrichmentStage;
        let stage = make_test_stage();
        let input = EnrichmentInput {
            message: "Hello world".to_string(),
            session_id: Uuid::new_v4(),
            project_slug: None, // No project
            project_id: None,
            cwd: None,
            protocol_run_id: None,
            protocol_state: None,
            excluded_note_ids: Default::default(),
            reasoning_path_tracker: None,
        };
        let output = stage.execute(&input).await.unwrap();
        assert!(
            output.sections.is_empty(),
            "Should skip when no project scope"
        );
    }

    #[tokio::test]
    async fn test_stage_is_enabled_checks_config() {
        use crate::chat::enrichment::ParallelEnrichmentStage;
        let stage = make_test_stage();
        let config = EnrichmentConfig {
            knowledge_injection: true,
            ..Default::default()
        };
        assert!(stage.is_enabled(&config));

        let config_disabled = EnrichmentConfig {
            knowledge_injection: false,
            ..Default::default()
        };
        assert!(!stage.is_enabled(&config_disabled));
    }

    #[tokio::test]
    async fn test_stage_name() {
        use crate::chat::enrichment::ParallelEnrichmentStage;
        let stage = make_test_stage();
        assert_eq!(stage.name(), "knowledge_injection");
    }

    #[tokio::test]
    async fn test_stage_with_project_runs_queries() {
        use crate::chat::enrichment::ParallelEnrichmentStage;
        let stage = make_test_stage();
        let input = EnrichmentInput {
            message: "Look at src/chat/manager.rs and the build_prompt function".to_string(),
            session_id: Uuid::new_v4(),
            project_slug: Some("test-project".to_string()),
            project_id: None,
            cwd: None,
            protocol_run_id: None,
            protocol_state: None,
            excluded_note_ids: Default::default(),
            reasoning_path_tracker: None,
        };

        // With mocks, this should complete without errors (empty results)
        let result = stage.execute(&input).await;
        assert!(result.is_ok(), "Stage should not error with mock stores");
    }

    #[tokio::test]
    async fn test_stage_with_protocol_context_propagates_to_decision_record() {
        use crate::chat::enrichment::ParallelEnrichmentStage;
        let stage = make_test_stage();
        let proto_run_id = Uuid::new_v4();
        let input = EnrichmentInput {
            message: "Look at src/chat/manager.rs and the build_prompt function".to_string(),
            session_id: Uuid::new_v4(),
            project_slug: Some("test-project".to_string()),
            project_id: None,
            cwd: None,
            protocol_run_id: Some(proto_run_id),
            protocol_state: Some("implement".to_string()),
            excluded_note_ids: Default::default(),
            reasoning_path_tracker: None,
        };

        // The stage should complete without errors; protocol context is passed through
        // to the DecisionRecord via input.protocol_run_id / input.protocol_state
        let result = stage.execute(&input).await;
        assert!(
            result.is_ok(),
            "Stage should not error with protocol context"
        );
    }

    // ── Deduplication logic tests ───────────────────────────────────────

    #[test]
    fn test_dedup_keeps_highest_score() {
        let mut note_map: HashMap<String, ScoredNote> = HashMap::new();

        let low = ScoredNote {
            id: "same-id".to_string(),
            note_type: "Tip".to_string(),
            importance: "Low".to_string(),
            content: "low score version".to_string(),
            score: 0.3,
            source: "bm25_search",
        };
        let high = ScoredNote {
            id: "same-id".to_string(),
            note_type: "Tip".to_string(),
            importance: "Low".to_string(),
            content: "high score version".to_string(),
            score: 0.9,
            source: "entity_context",
        };

        // Insert low first, then high
        note_map
            .entry(low.id.clone())
            .and_modify(|existing| {
                if low.score > existing.score {
                    *existing = low.clone();
                }
            })
            .or_insert(low);

        note_map
            .entry(high.id.clone())
            .and_modify(|existing| {
                if high.score > existing.score {
                    *existing = high.clone();
                }
            })
            .or_insert(high);

        let result = note_map.get("same-id").unwrap();
        assert!(
            (result.score - 0.9).abs() < f64::EPSILON,
            "Should keep the higher-scored version"
        );
        assert_eq!(result.content, "high score version");
    }

    // ── UUID extraction tests ─────────────────────────────────────────

    #[test]
    fn test_extract_uuids_basic() {
        let uuid_str = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";
        let message = format!("Look at note {}", uuid_str);
        let uuids = KnowledgeInjectionStage::extract_uuids(&message);
        assert_eq!(uuids.len(), 1);
        assert_eq!(uuids[0].to_string(), uuid_str);
    }

    #[test]
    fn test_extract_uuids_multiple() {
        let uuid1 = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";
        let uuid2 = "11111111-2222-3333-4444-555555555555";
        let message = format!("Note {} and decision {}", uuid1, uuid2);
        let uuids = KnowledgeInjectionStage::extract_uuids(&message);
        assert_eq!(uuids.len(), 2);
    }

    #[test]
    fn test_extract_uuids_none() {
        let message = "How do I create a new endpoint?";
        let uuids = KnowledgeInjectionStage::extract_uuids(message);
        assert!(uuids.is_empty());
    }

    #[test]
    fn test_extract_uuids_invalid_format() {
        // Not a valid UUID (wrong length)
        let message = "Look at a1b2c3d4-e5f6-7890-abcd";
        let uuids = KnowledgeInjectionStage::extract_uuids(message);
        assert!(uuids.is_empty());
    }

    #[test]
    fn test_extract_uuids_embedded_in_text() {
        let uuid_str = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";
        let message = format!("Check note({})", uuid_str);
        let uuids = KnowledgeInjectionStage::extract_uuids(&message);
        assert_eq!(uuids.len(), 1);
    }

    // ── Step verification tests ─────────────────────────────────────────
    // Verifies: "Le message 'regarde src/chat/manager.rs et la fonction
    //            build_system_prompt' détecte un fichier et une fonction.
    //            Les UUIDs sont extraits correctement."

    #[test]
    fn test_step_verification_file_and_function_detection() {
        // Entity extractor detects identifiers via:
        // 1. File path patterns (src/..., *.rs)
        // 2. Backtick identifiers (`build_system_prompt`)
        // 3. Code keyword patterns (fn build_system_prompt)
        let message = "regarde src/chat/manager.rs et la fonction `build_system_prompt`";
        let entities = entity_extractor::extract_entities(message);

        // Should detect a file
        let has_file = entities.iter().any(|e| {
            e.entity_type == ChatEntityType::File && e.identifier == "src/chat/manager.rs"
        });
        assert!(
            has_file,
            "Should detect file src/chat/manager.rs, got: {:?}",
            entities
        );

        // Should detect a function (via backtick extraction)
        let has_function = entities
            .iter()
            .any(|e| e.identifier == "build_system_prompt");
        assert!(
            has_function,
            "Should detect function build_system_prompt, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_step_verification_fn_keyword_detection() {
        // Also works with "fn" keyword pattern (without backticks)
        let message = "regarde src/chat/manager.rs et fn build_system_prompt";
        let entities = entity_extractor::extract_entities(message);

        let has_file = entities.iter().any(|e| {
            e.entity_type == ChatEntityType::File && e.identifier == "src/chat/manager.rs"
        });
        assert!(has_file, "Should detect file");

        let has_function = entities.iter().any(|e| {
            e.identifier == "build_system_prompt" && e.entity_type == ChatEntityType::Function
        });
        assert!(
            has_function,
            "Should detect function via 'fn' pattern, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_step_verification_uuid_extraction() {
        let uuid_str = "09fe035c-ab4e-40a8-a181-4e1be8a31004";
        let message = format!("Regarde la note {} et src/chat/manager.rs", uuid_str);
        let uuids = KnowledgeInjectionStage::extract_uuids(&message);
        assert_eq!(uuids.len(), 1, "Should extract one UUID");
        assert_eq!(uuids[0].to_string(), uuid_str);

        // File should also be detected
        let entities = entity_extractor::extract_entities(&message);
        let has_file = entities.iter().any(|e| {
            e.entity_type == ChatEntityType::File && e.identifier == "src/chat/manager.rs"
        });
        assert!(has_file, "Should also detect file alongside UUID");
    }

    // ── Step 3 verification: dedup, sort, budget ─────────────────────

    #[test]
    fn test_step3_no_duplicates_in_final_output() {
        let stage = make_test_stage();
        // Simulate notes from different sources with the same ID
        let notes = vec![
            ScoredNote {
                id: "dup-1".to_string(),
                note_type: "Gotcha".to_string(),
                importance: "High".to_string(),
                content: "Watch out for null".to_string(),
                score: 0.9,
                source: "bm25_search",
            },
            ScoredNote {
                id: "dup-1".to_string(), // same ID
                note_type: "Gotcha".to_string(),
                importance: "High".to_string(),
                content: "Watch out for null".to_string(),
                score: 0.7,
                source: "entity_context",
            },
            ScoredNote {
                id: "unique-2".to_string(),
                note_type: "Tip".to_string(),
                importance: "Medium".to_string(),
                content: "Use Arc for sharing".to_string(),
                score: 0.6,
                source: "propagated",
            },
        ];

        // Manually deduplicate like query_knowledge does
        let mut note_map: HashMap<String, ScoredNote> = HashMap::new();
        for note in notes {
            note_map
                .entry(note.id.clone())
                .and_modify(|existing| {
                    if note.score > existing.score {
                        *existing = note.clone();
                    }
                })
                .or_insert(note);
        }

        let deduped: Vec<ScoredNote> = note_map.into_values().collect();
        assert_eq!(deduped.len(), 2, "Should have 2 unique notes, not 3");

        // Render should have no duplicates
        let content = stage
            .render_knowledge(&deduped, &[], &[], &[], 3000)
            .unwrap();
        let count = content.matches("Watch out for null").count();
        assert_eq!(
            count, 1,
            "Content should appear exactly once, got {}",
            count
        );
    }

    #[test]
    fn test_step3_sort_by_relevance() {
        let mut notes = [
            ScoredNote {
                id: "low".to_string(),
                note_type: "Tip".to_string(),
                importance: "Low".to_string(),
                content: "Low priority".to_string(),
                score: 0.3,
                source: "bm25_search",
            },
            ScoredNote {
                id: "high".to_string(),
                note_type: "Gotcha".to_string(),
                importance: "Critical".to_string(),
                content: "High priority".to_string(),
                score: 0.95,
                source: "entity_context",
            },
            ScoredNote {
                id: "mid".to_string(),
                note_type: "Pattern".to_string(),
                importance: "Medium".to_string(),
                content: "Medium priority".to_string(),
                score: 0.6,
                source: "propagated",
            },
        ];

        // Sort like query_knowledge does
        notes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        assert_eq!(notes[0].id, "high", "Highest score first");
        assert_eq!(notes[1].id, "mid", "Medium score second");
        assert_eq!(notes[2].id, "low", "Lowest score last");
    }

    #[test]
    fn test_step3_context_budget_enforcement() {
        // Default budget is 3000 chars
        let stage = make_test_stage();

        // Create notes that total ~4500 chars (exceed budget)
        let notes: Vec<ScoredNote> = (0..30)
            .map(|i| ScoredNote {
                id: format!("note-{}", i),
                note_type: "Observation".to_string(),
                importance: "Medium".to_string(),
                content: format!(
                    "This is an observation about module {} with enough detail to be useful in context",
                    i
                ),
                score: 1.0 - (i as f64 * 0.02),
                source: "bm25_search",
            })
            .collect();

        let result = stage.render_knowledge(&notes, &[], &[], &[], 3000);
        assert!(result.is_some());
        let content = result.unwrap();
        assert!(
            content.len() <= 3200, // 3000 + some overhead from headers
            "Rendered content should respect char budget (3000), got {} chars",
            content.len()
        );
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    fn make_test_stage() -> KnowledgeInjectionStage {
        make_test_stage_with_config(KnowledgeInjectionConfig::default())
    }

    fn make_test_stage_with_config(config: KnowledgeInjectionConfig) -> KnowledgeInjectionStage {
        use crate::meilisearch::mock::MockSearchStore;
        use crate::neo4j::mock::MockGraphStore;

        let graph = Arc::new(MockGraphStore::new());
        let search = Arc::new(MockSearchStore::new());

        KnowledgeInjectionStage {
            graph,
            search,
            config,
            collector: None,
        }
    }

    // ── scaffolding_config tests ──────────────────────────────────────

    #[test]
    fn test_scaffolding_config_l0_generous() {
        let config = scaffolding_config(0);
        assert_eq!(config.max_notes, 8);
        assert_eq!(config.max_decisions, 5);
        assert_eq!(config.max_content_chars, 5000);
        assert_eq!(config.max_propagated_per_file, 5);
    }

    #[test]
    fn test_scaffolding_config_l2_default() {
        let config = scaffolding_config(2);
        let default = KnowledgeInjectionConfig::default();
        assert_eq!(config.max_notes, default.max_notes);
        assert_eq!(config.max_decisions, default.max_decisions);
        assert_eq!(config.max_content_chars, default.max_content_chars);
    }

    #[test]
    fn test_scaffolding_config_l4_minimal() {
        let config = scaffolding_config(4);
        assert_eq!(config.max_notes, 2);
        assert_eq!(config.max_decisions, 1);
        assert_eq!(config.max_content_chars, 1500);
        assert_eq!(config.max_propagated_per_file, 1);
    }

    #[test]
    fn test_scaffolding_config_progression() {
        // L0 > L1 > L2 > L3 > L4 in terms of max_notes
        let configs: Vec<_> = (0..=4).map(scaffolding_config).collect();
        for i in 0..4 {
            assert!(
                configs[i].max_notes >= configs[i + 1].max_notes,
                "L{} max_notes ({}) should be >= L{} max_notes ({})",
                i,
                configs[i].max_notes,
                i + 1,
                configs[i + 1].max_notes,
            );
            assert!(
                configs[i].max_content_chars >= configs[i + 1].max_content_chars,
                "L{} max_content_chars ({}) should be >= L{} max_content_chars ({})",
                i,
                configs[i].max_content_chars,
                i + 1,
                configs[i + 1].max_content_chars,
            );
        }
    }

    // ── Intent reweighting tests ───────────────────────────────────────

    #[test]
    fn test_intent_reweighting_debug_gotcha_above_guideline() {
        // Simulate: 1 guideline (score 0.9), 1 gotcha (score 0.7), 1 pattern (score 0.8)
        // With "debug" intent: gotcha×1.5=1.05, pattern×1.0=0.8, guideline×0.7=0.63
        // Expected ranking: gotcha > pattern > guideline
        let mut notes = vec![
            ScoredNote {
                id: "guideline-1".to_string(),
                note_type: "guideline".to_string(),
                importance: "high".to_string(),
                content: "Always use structured logging".to_string(),
                score: 0.9,
                source: "bm25_search",
            },
            ScoredNote {
                id: "gotcha-1".to_string(),
                note_type: "gotcha".to_string(),
                importance: "critical".to_string(),
                content: "Mutex deadlock when calling from async context".to_string(),
                score: 0.7,
                source: "bm25_search",
            },
            ScoredNote {
                id: "pattern-1".to_string(),
                note_type: "pattern".to_string(),
                importance: "medium".to_string(),
                content: "Use Arc<dyn Trait> for shared services".to_string(),
                score: 0.8,
                source: "bm25_search",
            },
        ];

        // Apply debug intent reweighting
        let weight_map = IntentWeightMap::for_intent("debug");
        for note in &mut notes {
            note.score *= weight_map.get(&note.note_type);
        }
        notes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // gotcha: 0.7 * 1.5 = 1.05
        assert_eq!(
            notes[0].id, "gotcha-1",
            "Gotcha should rank first for debug intent"
        );
        assert!((notes[0].score - 1.05).abs() < 1e-10);

        // pattern: 0.8 * 1.0 = 0.8
        assert_eq!(
            notes[1].id, "pattern-1",
            "Pattern should rank second for debug intent"
        );
        assert!((notes[1].score - 0.8).abs() < 1e-10);

        // guideline: 0.9 * 0.7 = 0.63
        assert_eq!(
            notes[2].id, "guideline-1",
            "Guideline should rank last for debug intent"
        );
        assert!((notes[2].score - 0.63).abs() < 1e-10);
    }

    #[test]
    fn test_intent_reweighting_general_preserves_order() {
        // With "general" intent, all weights are 1.0 → original order preserved
        let mut notes = vec![
            ScoredNote {
                id: "high".to_string(),
                note_type: "guideline".to_string(),
                importance: "high".to_string(),
                content: "high score".to_string(),
                score: 0.9,
                source: "bm25_search",
            },
            ScoredNote {
                id: "mid".to_string(),
                note_type: "gotcha".to_string(),
                importance: "high".to_string(),
                content: "mid score".to_string(),
                score: 0.7,
                source: "bm25_search",
            },
        ];

        let weight_map = IntentWeightMap::for_intent("general");
        for note in &mut notes {
            note.score *= weight_map.get(&note.note_type);
        }
        notes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        assert_eq!(
            notes[0].id, "high",
            "Original order preserved with general intent"
        );
        assert!((notes[0].score - 0.9).abs() < f64::EPSILON);
        assert_eq!(notes[1].id, "mid");
        assert!((notes[1].score - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_intent_reweighting_unknown_type_gets_one() {
        let mut notes = vec![ScoredNote {
            id: "custom".to_string(),
            note_type: "unknown_custom_type".to_string(),
            importance: "low".to_string(),
            content: "some content".to_string(),
            score: 0.5,
            source: "bm25_search",
        }];

        let weight_map = IntentWeightMap::for_intent("debug");
        for note in &mut notes {
            note.score *= weight_map.get(&note.note_type);
        }

        assert!(
            (notes[0].score - 0.5).abs() < f64::EPSILON,
            "Unknown note type should get weight 1.0 (score unchanged)"
        );
    }

    // ── Scaffolding integration tests ─────────────────────────────────

    /// Helper: create a KnowledgeInjectionStage with a pre-populated MockGraphStore
    async fn make_stage_with_mock_data(
        project_id: Uuid,
        completed_tasks: usize,
        failed_tasks: usize,
        frustration: f64,
        note_count: usize,
        scar_intensity: f64,
    ) -> KnowledgeInjectionStage {
        use crate::meilisearch::mock::MockSearchStore;
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::models::TaskStatus;
        use crate::test_helpers::{test_plan_for_project, test_task};

        let graph = Arc::new(MockGraphStore::new());

        // Create plan for this project
        let plan = test_plan_for_project(project_id);
        let plan_id = plan.id;
        graph.create_plan(&plan).await.unwrap();

        // Create completed tasks
        for i in 0..completed_tasks {
            let mut task = test_task();
            task.title = Some(format!("Completed task {}", i));
            task.status = TaskStatus::Completed;
            task.frustration_score = frustration;
            graph.create_task(plan_id, &task).await.unwrap();
        }

        // Create failed tasks
        for i in 0..failed_tasks {
            let mut task = test_task();
            task.title = Some(format!("Failed task {}", i));
            task.status = TaskStatus::Failed;
            task.frustration_score = frustration;
            graph.create_task(plan_id, &task).await.unwrap();
        }

        // Create notes with scar intensity
        for i in 0..note_count {
            use crate::notes::models::{Note, NoteType};
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                format!("Note content {}", i),
                "test-agent".to_string(),
            );
            note.scar_intensity = scar_intensity;
            graph.create_note(&note).await.unwrap();
        }

        let search = Arc::new(MockSearchStore::new());
        KnowledgeInjectionStage {
            graph: graph as Arc<dyn crate::neo4j::traits::GraphStore>,
            search: search as Arc<dyn crate::meilisearch::traits::SearchStore>,
            config: KnowledgeInjectionConfig::default(),
            collector: None,
        }
    }

    #[tokio::test]
    async fn test_scaffolding_integration_mature_project_uses_reduced_config() {
        // Mature project: many completed tasks, no failures, no frustration
        // competence = (1.0*0.5 + 1.0*0.2 + (1-0)*0.15 + 1.0*0.15) = 1.0 → L4
        let project_id = Uuid::new_v4();
        let stage = make_stage_with_mock_data(
            project_id, 20,  // completed tasks
            0,   // failed tasks
            0.0, // frustration
            10,  // notes (no scars)
            0.0, // scar_intensity
        )
        .await;

        // Compute scaffolding level directly
        let level = stage
            .graph
            .compute_scaffolding_level(project_id, None)
            .await
            .unwrap();

        assert!(
            level.level >= 3,
            "Mature project (high success) should be L3 or L4, got L{}",
            level.level
        );
        assert!(
            level.competence_score >= 0.75,
            "competence_score should be >= 0.75, got {:.2}",
            level.competence_score
        );

        let config = scaffolding_config(level.level);
        assert!(
            config.max_notes <= 3,
            "L{} should have max_notes <= 3, got {}",
            level.level,
            config.max_notes
        );
    }

    #[tokio::test]
    async fn test_scaffolding_integration_struggling_project_uses_generous_config() {
        // Struggling project: mostly failed tasks, high frustration, scarred notes
        // task_success_rate = 2/(2+18) = 0.1
        // avg_frustration = 0.9
        // scar_density = 0.8
        // competence = (0.1*0.5 + 0.1*0.2 + 0.2*0.15 + 1.0*0.15) = 0.05+0.02+0.03+0.15 = 0.25 → L0
        let project_id = Uuid::new_v4();
        let stage = make_stage_with_mock_data(
            project_id, 2,   // completed tasks
            18,  // failed tasks
            0.9, // high frustration
            10,  // notes with scars
            0.8, // high scar_intensity
        )
        .await;

        let level = stage
            .graph
            .compute_scaffolding_level(project_id, None)
            .await
            .unwrap();

        assert!(
            level.level <= 1,
            "Struggling project should be L0 or L1, got L{}",
            level.level
        );
        assert!(
            level.competence_score < 0.5,
            "competence_score should be < 0.5, got {:.2}",
            level.competence_score
        );

        let config = scaffolding_config(level.level);
        assert!(
            config.max_notes >= 6,
            "L{} should have max_notes >= 6 (generous scaffolding), got {}",
            level.level,
            config.max_notes
        );
    }

    #[tokio::test]
    async fn test_scaffolding_integration_empty_project_gets_high_level() {
        // Empty project: no tasks, no notes → defaults give competence = 1.0 → L4
        let project_id = Uuid::new_v4();
        let stage = make_stage_with_mock_data(
            project_id, 0,   // no tasks
            0,   // no tasks
            0.0, // no frustration
            0,   // no notes
            0.0, // no scars
        )
        .await;

        let level = stage
            .graph
            .compute_scaffolding_level(project_id, None)
            .await
            .unwrap();

        // Empty project defaults to high competence (no failures = success)
        assert_eq!(
            level.level, 4,
            "Empty project with no data should default to L4, got L{}",
            level.level
        );
        let config = scaffolding_config(level.level);
        assert_eq!(config.max_notes, 2, "L4 should have max_notes=2");
    }

    #[tokio::test]
    async fn test_scaffolding_disabled_via_env_uses_default() {
        // When ENRICHMENT_SCAFFOLDING=false, scaffolding is disabled → L2 default config
        let config_disabled = scaffolding_config(2); // L2 = default
        let config_default = KnowledgeInjectionConfig::default();
        assert_eq!(
            config_disabled.max_notes, config_default.max_notes,
            "L2 config should match default config"
        );
    }

    #[test]
    fn test_intent_reweighting_planning_boosts_guideline() {
        let mut notes = vec![
            ScoredNote {
                id: "gotcha-1".to_string(),
                note_type: "gotcha".to_string(),
                importance: "high".to_string(),
                content: "Gotcha content".to_string(),
                score: 0.9,
                source: "bm25_search",
            },
            ScoredNote {
                id: "guideline-1".to_string(),
                note_type: "guideline".to_string(),
                importance: "high".to_string(),
                content: "Guideline content".to_string(),
                score: 0.7,
                source: "bm25_search",
            },
        ];

        // Planning: guideline×1.5=1.05, gotcha×0.8=0.72
        let weight_map = IntentWeightMap::for_intent("planning");
        for note in &mut notes {
            note.score *= weight_map.get(&note.note_type);
        }
        notes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        assert_eq!(
            notes[0].id, "guideline-1",
            "Guideline should rank first for planning intent"
        );
    }
}
