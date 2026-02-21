//! Note Manager - CRUD operations for Knowledge Notes
//!
//! Provides high-level operations for creating, reading, updating, and deleting notes,
//! including linking notes to entities and managing note lifecycle.

use super::models::*;
use crate::embeddings::EmbeddingProvider;
use crate::events::{CrudAction, CrudEvent, EntityType as EventEntityType, EventEmitter};
use crate::meilisearch::indexes::NoteDocument;
use crate::meilisearch::SearchStore;
use crate::neo4j::GraphStore;
use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;

/// Configuration for automatic synapse creation between similar notes.
#[derive(Debug, Clone)]
pub struct SynapseConfig {
    /// Minimum cosine similarity score to create a synapse (default: 0.75)
    pub min_weight: f64,
    /// Maximum number of neighbors to connect per note (default: 10)
    pub max_neighbors: usize,
    /// Whether auto-synapse creation is enabled (default: true)
    pub enabled: bool,
}

impl Default for SynapseConfig {
    fn default() -> Self {
        Self {
            min_weight: 0.75,
            max_neighbors: 10,
            enabled: true,
        }
    }
}

/// Manager for Knowledge Notes operations
pub struct NoteManager {
    neo4j: Arc<dyn GraphStore>,
    meilisearch: Arc<dyn SearchStore>,
    event_emitter: Option<Arc<dyn EventEmitter>>,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    synapse_config: SynapseConfig,
}

impl NoteManager {
    /// Create a new NoteManager
    pub fn new(neo4j: Arc<dyn GraphStore>, meilisearch: Arc<dyn SearchStore>) -> Self {
        Self {
            neo4j,
            meilisearch,
            event_emitter: None,
            embedding_provider: None,
            synapse_config: SynapseConfig::default(),
        }
    }

    /// Create a new NoteManager with an event emitter
    pub fn with_event_emitter(
        neo4j: Arc<dyn GraphStore>,
        meilisearch: Arc<dyn SearchStore>,
        emitter: Arc<dyn EventEmitter>,
    ) -> Self {
        Self {
            neo4j,
            meilisearch,
            event_emitter: Some(emitter),
            embedding_provider: None,
            synapse_config: SynapseConfig::default(),
        }
    }

    /// Add an embedding provider to this NoteManager (builder pattern).
    ///
    /// When set, notes will be automatically embedded on creation and update.
    /// Embedding failures are logged but do not block the note operation.
    pub fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    /// Set synapse configuration (builder pattern).
    pub fn with_synapse_config(mut self, config: SynapseConfig) -> Self {
        self.synapse_config = config;
        self
    }

    /// Emit a CRUD event (no-op if event_emitter is None)
    fn emit(&self, event: crate::events::CrudEvent) {
        if let Some(emitter) = &self.event_emitter {
            emitter.emit(event);
        }
    }

    /// Generate and store an embedding for a note's content.
    ///
    /// This is a best-effort operation: if the embedding provider is not configured
    /// or the embedding fails, the note is still created/updated successfully.
    /// Embedding errors are logged at warn level but never propagated.
    async fn embed_note(&self, note_id: Uuid, content: &str) {
        let provider = match &self.embedding_provider {
            Some(p) => p,
            None => return,
        };

        match provider.embed_text(content).await {
            Ok(embedding) => {
                let model = provider.model_name().to_string();
                if let Err(e) = self
                    .neo4j
                    .set_note_embedding(note_id, &embedding, &model)
                    .await
                {
                    tracing::warn!(
                        note_id = %note_id,
                        error = %e,
                        "Failed to store note embedding"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    note_id = %note_id,
                    error = %e,
                    "Failed to generate note embedding"
                );
            }
        }
    }

    /// Automatically create synapses between a note and its semantically similar neighbors.
    ///
    /// This is a best-effort, fire-and-forget operation: it runs after the note is
    /// created/embedded. Failures are logged at warn level but never block the caller.
    ///
    /// Process:
    /// 1. Get the note's embedding from the provider
    /// 2. Vector search for K nearest neighbors (filtered by project_id)
    /// 3. Filter by min_weight threshold
    /// 4. Create bidirectional synapses via GraphStore
    fn spawn_auto_connect_synapses(
        &self,
        note_id: Uuid,
        content: &str,
        project_id: Option<Uuid>,
    ) {
        // Skip if no embedding provider or synapses disabled
        let provider = match &self.embedding_provider {
            Some(p) if self.synapse_config.enabled => p.clone(),
            _ => return,
        };

        let neo4j = self.neo4j.clone();
        let min_weight = self.synapse_config.min_weight;
        let max_neighbors = self.synapse_config.max_neighbors;
        let content = content.to_string();

        tokio::spawn(async move {
            // Step 1: Embed the note's content
            let embedding = match provider.embed_text(&content).await {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(
                        note_id = %note_id,
                        error = %e,
                        "Auto-synapse: failed to embed note content"
                    );
                    return;
                }
            };

            // Step 2: Vector search for nearest neighbors
            let candidates = match neo4j
                .vector_search_notes(&embedding, max_neighbors + 1, project_id, None)
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(
                        note_id = %note_id,
                        error = %e,
                        "Auto-synapse: vector search failed"
                    );
                    return;
                }
            };

            // Step 3: Filter — exclude self, apply min_weight, limit to max_neighbors
            let neighbors: Vec<(Uuid, f64)> = candidates
                .into_iter()
                .filter(|(note, score)| note.id != note_id && *score >= min_weight)
                .take(max_neighbors)
                .map(|(note, score)| (note.id, score))
                .collect();

            if neighbors.is_empty() {
                tracing::debug!(
                    note_id = %note_id,
                    "Auto-synapse: no neighbors above threshold {:.2}",
                    min_weight
                );
                return;
            }

            // Step 4: Create bidirectional synapses
            match neo4j.create_synapses(note_id, &neighbors).await {
                Ok(count) => {
                    tracing::debug!(
                        note_id = %note_id,
                        neighbors = neighbors.len(),
                        synapse_count = count,
                        "Auto-synapse: created synapses"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        note_id = %note_id,
                        error = %e,
                        "Auto-synapse: failed to create synapses"
                    );
                }
            }
        });
    }

    // ========================================================================
    // CRUD Operations
    // ========================================================================

    /// Create a new note
    pub async fn create_note(&self, input: CreateNoteRequest, created_by: &str) -> Result<Note> {
        let note = Note::new_full(
            input.project_id,
            input.note_type,
            input.importance.unwrap_or_default(),
            input.scope.unwrap_or(NoteScope::Project),
            input.content,
            input.tags.unwrap_or_default(),
            created_by.to_string(),
        );

        // Store in Neo4j
        self.neo4j.create_note(&note).await?;

        // Index in Meilisearch
        let doc = self.note_to_document(&note, None).await?;
        self.meilisearch.index_note(&doc).await?;

        // Add initial anchors if provided
        if let Some(anchors) = input.anchors {
            for anchor_req in anchors {
                self.neo4j
                    .link_note_to_entity(
                        note.id,
                        &anchor_req.entity_type,
                        &anchor_req.entity_id,
                        anchor_req.signature_hash.as_deref(),
                        anchor_req.body_hash.as_deref(),
                    )
                    .await?;
            }
        }

        // Generate embedding (best-effort, non-blocking for note creation)
        self.embed_note(note.id, &note.content).await;

        // Auto-connect synapses to semantically similar notes (fire-and-forget)
        self.spawn_auto_connect_synapses(note.id, &note.content, note.project_id);

        self.emit(
            CrudEvent::new(
                EventEntityType::Note,
                CrudAction::Created,
                note.id.to_string(),
            )
            .with_payload(serde_json::json!({"note_type": note.note_type.to_string()}))
            .with_project_id(note.project_id.map(|id| id.to_string()).unwrap_or_default()),
        );

        Ok(note)
    }

    /// Get a note by ID
    pub async fn get_note(&self, id: Uuid) -> Result<Option<Note>> {
        let note = self.neo4j.get_note(id).await?;

        // Load anchors if note exists
        if let Some(mut note) = note {
            note.anchors = self.neo4j.get_note_anchors(id).await?;
            Ok(Some(note))
        } else {
            Ok(None)
        }
    }

    /// Update a note
    pub async fn update_note(&self, id: Uuid, input: UpdateNoteRequest) -> Result<Option<Note>> {
        let content_changed = input.content.is_some();

        let updated = self
            .neo4j
            .update_note(
                id,
                input.content,
                input.importance,
                input.status,
                input.tags,
                None,
            )
            .await?;

        // Update Meilisearch index
        if let Some(ref note) = updated {
            let doc = self.note_to_document(note, None).await?;
            self.meilisearch.index_note(&doc).await?;

            // Re-embed if content changed (best-effort)
            if content_changed {
                self.embed_note(id, &note.content).await;

                // Content changed → delete old synapses and re-connect
                // delete_synapses is best-effort (non-critical for the update)
                if self.synapse_config.enabled {
                    if let Err(e) = self.neo4j.delete_synapses(id).await {
                        tracing::warn!(
                            note_id = %id,
                            error = %e,
                            "Failed to delete old synapses during update"
                        );
                    }
                    self.spawn_auto_connect_synapses(id, &note.content, note.project_id);
                }
            }

            self.emit(
                CrudEvent::new(EventEntityType::Note, CrudAction::Updated, id.to_string())
                    .with_project_id(note.project_id.map(|id| id.to_string()).unwrap_or_default()),
            );
        }

        Ok(updated)
    }

    /// Delete a note
    pub async fn delete_note(&self, id: Uuid) -> Result<bool> {
        // Delete from Neo4j (DETACH DELETE also removes SYNAPSE relationships)
        let deleted = self.neo4j.delete_note(id).await?;

        // Delete from Meilisearch
        if deleted {
            self.meilisearch.delete_note(&id.to_string()).await?;
            self.emit(CrudEvent::new(
                EventEntityType::Note,
                CrudAction::Deleted,
                id.to_string(),
            ));
        }

        Ok(deleted)
    }

    /// List notes with filters and pagination
    pub async fn list_notes(
        &self,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
        filters: &NoteFilters,
    ) -> Result<(Vec<Note>, usize)> {
        self.neo4j
            .list_notes(project_id, workspace_slug, filters)
            .await
    }

    /// List notes for a specific project
    pub async fn list_project_notes(
        &self,
        project_id: Uuid,
        filters: &NoteFilters,
    ) -> Result<(Vec<Note>, usize)> {
        self.neo4j.list_notes(Some(project_id), None, filters).await
    }

    // ========================================================================
    // Linking Operations
    // ========================================================================

    /// Link a note to an entity
    pub async fn link_note_to_entity(&self, note_id: Uuid, entity: &LinkNoteRequest) -> Result<()> {
        self.neo4j
            .link_note_to_entity(note_id, &entity.entity_type, &entity.entity_id, None, None)
            .await?;
        self.emit(
            CrudEvent::new(EventEntityType::Note, CrudAction::Linked, note_id.to_string())
                .with_payload(serde_json::json!({"entity_type": entity.entity_type.to_string(), "entity_id": &entity.entity_id})),
        );
        Ok(())
    }

    /// Link a note to an entity with semantic hashes
    pub async fn link_note_with_hashes(
        &self,
        note_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
        signature_hash: Option<&str>,
        body_hash: Option<&str>,
    ) -> Result<()> {
        self.neo4j
            .link_note_to_entity(note_id, entity_type, entity_id, signature_hash, body_hash)
            .await
    }

    /// Unlink a note from an entity
    pub async fn unlink_note_from_entity(
        &self,
        note_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> Result<()> {
        self.neo4j
            .unlink_note_from_entity(note_id, entity_type, entity_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Note,
                CrudAction::Unlinked,
                note_id.to_string(),
            )
            .with_payload(
                serde_json::json!({"entity_type": entity_type.to_string(), "entity_id": entity_id}),
            ),
        );
        Ok(())
    }

    /// Get all anchors for a note
    pub async fn get_note_anchors(&self, note_id: Uuid) -> Result<Vec<NoteAnchor>> {
        self.neo4j.get_note_anchors(note_id).await
    }

    // ========================================================================
    // Lifecycle Operations
    // ========================================================================

    /// Confirm a note is still valid
    pub async fn confirm_note(&self, note_id: Uuid, confirmed_by: &str) -> Result<Option<Note>> {
        let note = self.neo4j.confirm_note(note_id, confirmed_by).await?;

        // Update Meilisearch
        if let Some(ref note) = note {
            let doc = self.note_to_document(note, None).await?;
            self.meilisearch.index_note(&doc).await?;

            self.emit(
                CrudEvent::new(
                    EventEntityType::Note,
                    CrudAction::Updated,
                    note_id.to_string(),
                )
                .with_payload(serde_json::json!({"confirmed_by": confirmed_by}))
                .with_project_id(note.project_id.map(|id| id.to_string()).unwrap_or_default()),
            );
        }

        Ok(note)
    }

    /// Invalidate a note (mark as obsolete)
    pub async fn invalidate_note(
        &self,
        note_id: Uuid,
        reason: &str,
        invalidated_by: &str,
    ) -> Result<Option<Note>> {
        let updated = self
            .neo4j
            .update_note(note_id, None, None, Some(NoteStatus::Obsolete), None, None)
            .await?;

        // Update Meilisearch
        if updated.is_some() {
            self.meilisearch
                .update_note_status(&note_id.to_string(), "obsolete")
                .await?;
        }

        // Log the invalidation reason (could be stored as a change)
        tracing::info!(
            "Note {} invalidated by {}: {}",
            note_id,
            invalidated_by,
            reason
        );

        if updated.is_some() {
            self.emit(
                CrudEvent::new(
                    EventEntityType::Note,
                    CrudAction::Updated,
                    note_id.to_string(),
                )
                .with_payload(serde_json::json!({"status": "obsolete", "reason": reason})),
            );
        }

        Ok(updated)
    }

    /// Supersede an old note with a new one
    pub async fn supersede_note(
        &self,
        old_note_id: Uuid,
        new_note_input: CreateNoteRequest,
        created_by: &str,
    ) -> Result<Note> {
        // Create the new note
        let mut new_note = self.create_note(new_note_input, created_by).await?;
        new_note.supersedes = Some(old_note_id);

        // Mark the old note as superseded
        self.neo4j.supersede_note(old_note_id, new_note.id).await?;

        // Update old note in Meilisearch
        self.meilisearch
            .update_note_status(&old_note_id.to_string(), "archived")
            .await?;

        self.emit(
            CrudEvent::new(
                EventEntityType::Note,
                CrudAction::Updated,
                old_note_id.to_string(),
            )
            .with_payload(
                serde_json::json!({"status": "archived", "superseded_by": new_note.id.to_string()}),
            ),
        );

        Ok(new_note)
    }

    /// Get notes that need review
    pub async fn get_notes_needing_review(&self, project_id: Option<Uuid>) -> Result<Vec<Note>> {
        self.neo4j.get_notes_needing_review(project_id).await
    }

    /// Update staleness scores for all active notes
    pub async fn update_staleness_scores(&self) -> Result<usize> {
        self.neo4j.update_staleness_scores().await
    }

    /// Apply exponential energy decay to all active notes.
    ///
    /// Formula: `energy = energy × exp(-days_idle / half_life)`
    /// where `days_idle = (now - last_activated).days()`.
    ///
    /// Temporally idempotent: result depends only on elapsed time since
    /// `last_activated`, not on call frequency.
    pub async fn update_energy_scores(&self, half_life_days: f64) -> Result<usize> {
        self.neo4j.update_energy_scores(half_life_days).await
    }

    // ========================================================================
    // Search Operations
    // ========================================================================

    /// Search notes using semantic search
    pub async fn search_notes(
        &self,
        query: &str,
        filters: &NoteFilters,
    ) -> Result<Vec<NoteSearchHit>> {
        let project_slug = filters.search.as_deref(); // This is a simplification
        let note_type = filters
            .note_type
            .as_ref()
            .and_then(|v| v.first())
            .map(|t| t.to_string());
        let status = filters
            .status
            .as_ref()
            .and_then(|v| v.first())
            .map(|s| s.to_string());
        let importance = filters
            .importance
            .as_ref()
            .and_then(|v| v.first())
            .map(|i| i.to_string());

        let limit = filters.limit.unwrap_or(20) as usize;

        let hits = self
            .meilisearch
            .search_notes_with_scores(
                query,
                limit,
                project_slug,
                note_type.as_deref(),
                status.as_deref(),
                importance.as_deref(),
            )
            .await?;

        // Convert to NoteSearchHit
        let mut results = Vec::new();
        for hit in hits {
            // Get full note from Neo4j for complete data
            if let Ok(Some(note)) = self.neo4j.get_note(hit.document.id.parse()?).await {
                results.push(NoteSearchHit {
                    note,
                    score: hit.score,
                    highlights: None,
                });
            }
        }

        Ok(results)
    }

    /// Search notes using vector similarity (cosine) via Neo4j HNSW index.
    ///
    /// Embeds the query text using the configured embedding provider, then
    /// performs a vector similarity search against stored note embeddings.
    /// Falls back to BM25 text search (Meilisearch) if no embedding provider
    /// is configured.
    ///
    /// # Arguments
    /// * `query` - Natural language search query
    /// * `project_id` - Optional filter by project UUID
    /// * `workspace_slug` - Optional filter by workspace (includes all projects in workspace + global notes)
    /// * `limit` - Maximum number of results (default 20)
    pub async fn semantic_search_notes(
        &self,
        query: &str,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<NoteSearchHit>> {
        let limit = limit.unwrap_or(20);

        // If no embedding provider, fall back to BM25 text search
        let provider = match &self.embedding_provider {
            Some(p) => p,
            None => {
                tracing::warn!(
                    "semantic_search_notes: no embedding provider configured, falling back to BM25 text search"
                );
                let filters = NoteFilters {
                    limit: Some(limit as i64),
                    ..Default::default()
                };
                return self.search_notes(query, &filters).await;
            }
        };

        // Embed the query text
        let query_embedding = provider
            .embed_text(query)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to embed search query: {}", e))?;

        // Perform vector similarity search via Neo4j HNSW index
        let results = self
            .neo4j
            .vector_search_notes(&query_embedding, limit, project_id, workspace_slug)
            .await?;

        // Convert to NoteSearchHit
        Ok(results
            .into_iter()
            .map(|(note, score)| NoteSearchHit {
                note,
                score,
                highlights: None,
            })
            .collect())
    }

    // ========================================================================
    // Context Operations
    // ========================================================================

    /// Get notes for a specific entity (directly attached)
    pub async fn get_direct_notes(&self, entity_id: &str) -> Result<Vec<Note>> {
        // Try to parse as UUID first (for Task, Plan, etc.)
        if let Ok(uuid) = entity_id.parse::<Uuid>() {
            // Could be Task, Plan, or other UUID-based entity
            // For now, we'll search by entity_id across different types
            let mut all_notes = Vec::new();

            for entity_type in [EntityType::Task, EntityType::Plan, EntityType::Project] {
                let notes = self
                    .neo4j
                    .get_notes_for_entity(&entity_type, &uuid.to_string())
                    .await?;
                all_notes.extend(notes);
            }

            Ok(all_notes)
        } else {
            // Likely a file path
            self.neo4j
                .get_notes_for_entity(&EntityType::File, entity_id)
                .await
        }
    }

    /// Get propagated notes for an entity (via graph traversal)
    pub async fn get_propagated_notes(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
        max_depth: u32,
        min_score: f64,
    ) -> Result<Vec<PropagatedNote>> {
        self.neo4j
            .get_propagated_notes(entity_type, entity_id, max_depth, min_score)
            .await
    }

    /// Get contextual notes for an entity (direct + propagated)
    pub async fn get_context_notes(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
        max_depth: u32,
        min_score: f64,
    ) -> Result<NoteContextResponse> {
        // Get direct notes
        let direct_notes = self
            .neo4j
            .get_notes_for_entity(entity_type, entity_id)
            .await?;

        // Get propagated notes from graph traversal
        let mut propagated_notes = self
            .neo4j
            .get_propagated_notes(entity_type, entity_id, max_depth, min_score)
            .await?;

        // If entity is a Project, also get workspace-level notes
        // These propagate from the parent workspace with a decay factor
        if *entity_type == EntityType::Project {
            if let Ok(project_id) = entity_id.parse::<uuid::Uuid>() {
                const WORKSPACE_PROPAGATION_FACTOR: f64 = 0.8;
                let workspace_notes = self
                    .neo4j
                    .get_workspace_notes_for_project(project_id, WORKSPACE_PROPAGATION_FACTOR)
                    .await?;

                // Filter by min_score and add to propagated notes
                for note in workspace_notes {
                    if note.relevance_score >= min_score {
                        propagated_notes.push(note);
                    }
                }
            }
        }

        // Sort propagated notes by relevance score (descending)
        propagated_notes.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_count = direct_notes.len() + propagated_notes.len();

        Ok(NoteContextResponse {
            direct_notes,
            propagated_notes,
            total_count,
        })
    }

    // ========================================================================
    // History Operations
    // ========================================================================

    /// Get the change history for a note
    pub async fn get_note_history(&self, note_id: Uuid) -> Result<Vec<NoteChange>> {
        if let Some(note) = self.get_note(note_id).await? {
            Ok(note.changes)
        } else {
            Ok(vec![])
        }
    }

    // ========================================================================
    // Helper Functions
    // ========================================================================

    /// Convert a Note to a NoteDocument for Meilisearch indexing
    async fn note_to_document(
        &self,
        note: &Note,
        project_slug: Option<&str>,
    ) -> Result<NoteDocument> {
        // Get project slug if not provided
        let slug = if let Some(s) = project_slug {
            s.to_string()
        } else {
            // Try to get from project (if project_id is present)
            if let Some(pid) = note.project_id {
                if let Ok(Some(project)) = self.neo4j.get_project(pid).await {
                    project.slug
                } else {
                    String::new()
                }
            } else {
                String::new() // Global note, no project slug
            }
        };

        // Get anchor entity IDs
        let anchors = self
            .neo4j
            .get_note_anchors(note.id)
            .await
            .unwrap_or_default();
        let anchor_entities: Vec<String> = anchors
            .iter()
            .map(|a| format!("{}:{}", a.entity_type, a.entity_id))
            .collect();

        Ok(NoteDocument {
            id: note.id.to_string(),
            project_id: note.project_id.map(|id| id.to_string()).unwrap_or_default(),
            project_slug: slug,
            note_type: note.note_type.to_string(),
            status: note.status.to_string(),
            importance: note.importance.to_string(),
            scope_type: match &note.scope {
                NoteScope::Workspace => "workspace".to_string(),
                NoteScope::Project => "project".to_string(),
                NoteScope::Module(_) => "module".to_string(),
                NoteScope::File(_) => "file".to_string(),
                NoteScope::Function(_) => "function".to_string(),
                NoteScope::Struct(_) => "struct".to_string(),
                NoteScope::Trait(_) => "trait".to_string(),
            },
            scope_path: match &note.scope {
                NoteScope::Workspace | NoteScope::Project => String::new(),
                NoteScope::Module(p) | NoteScope::File(p) => p.clone(),
                NoteScope::Function(n) | NoteScope::Struct(n) | NoteScope::Trait(n) => n.clone(),
            },
            content: note.content.clone(),
            tags: note.tags.clone(),
            anchor_entities,
            created_at: note.created_at.timestamp(),
            created_by: note.created_by.clone(),
            staleness_score: note.staleness_score,
        })
    }

    // ========================================================================
    // Embedding Backfill
    // ========================================================================

    /// Backfill embeddings for all notes that don't have one yet.
    ///
    /// Processes notes in batches, generating embeddings and storing them.
    /// This operation is **idempotent**: re-running it only processes notes
    /// that still lack an embedding.
    ///
    /// Returns a `BackfillProgress` with the final counts.
    ///
    /// # Arguments
    /// * `batch_size` - Number of notes to process per batch (default: 50)
    /// * `cancel` - Optional cancellation flag; set to `true` to stop early
    pub async fn backfill_embeddings(
        &self,
        batch_size: usize,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<BackfillProgress> {
        let provider = match &self.embedding_provider {
            Some(p) => p,
            None => anyhow::bail!("No embedding provider configured"),
        };

        let batch_size = if batch_size == 0 { 50 } else { batch_size };

        // Get total count first
        let (_, total) = self.neo4j.list_notes_without_embedding(0, 0).await?;
        if total == 0 {
            tracing::info!("Backfill: all notes already have embeddings");
            return Ok(BackfillProgress {
                total,
                processed: 0,
                errors: 0,
                skipped: 0,
            });
        }

        tracing::info!("Backfill: {total} notes need embeddings");
        let mut processed = 0usize;
        let mut errors = 0usize;

        loop {
            // Check cancellation
            if let Some(flag) = cancel {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    tracing::info!("Backfill cancelled at {processed}/{total}");
                    break;
                }
            }

            // Fetch next batch (always offset 0 because processed notes disappear)
            let (batch, remaining) = self
                .neo4j
                .list_notes_without_embedding(batch_size, 0)
                .await?;
            if batch.is_empty() || remaining == 0 {
                break;
            }

            // Collect texts for batch embedding
            let texts: Vec<String> = batch.iter().map(|n| n.content.clone()).collect();
            let note_ids: Vec<Uuid> = batch.iter().map(|n| n.id).collect();

            match provider.embed_batch(&texts).await {
                Ok(embeddings) => {
                    let model = provider.model_name().to_string();
                    for (i, embedding) in embeddings.into_iter().enumerate() {
                        if let Err(e) = self
                            .neo4j
                            .set_note_embedding(note_ids[i], &embedding, &model)
                            .await
                        {
                            tracing::warn!(
                                note_id = %note_ids[i],
                                error = %e,
                                "Backfill: failed to store embedding"
                            );
                            errors += 1;
                        } else {
                            processed += 1;
                        }
                    }
                }
                Err(e) => {
                    // Batch embedding failed — try one by one as fallback
                    tracing::warn!(
                        error = %e,
                        batch_size = texts.len(),
                        "Backfill: batch embedding failed, falling back to individual"
                    );
                    for (i, text) in texts.iter().enumerate() {
                        match provider.embed_text(text).await {
                            Ok(embedding) => {
                                let model = provider.model_name().to_string();
                                if let Err(e) = self
                                    .neo4j
                                    .set_note_embedding(note_ids[i], &embedding, &model)
                                    .await
                                {
                                    tracing::warn!(
                                        note_id = %note_ids[i],
                                        error = %e,
                                        "Backfill: failed to store embedding"
                                    );
                                    errors += 1;
                                } else {
                                    processed += 1;
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    note_id = %note_ids[i],
                                    error = %e,
                                    "Backfill: failed to embed note, skipping"
                                );
                                errors += 1;
                            }
                        }
                    }
                }
            }

            tracing::info!("Backfill: {processed}/{total} notes embedded ({errors} errors)");
        }

        let skipped = total.saturating_sub(processed + errors);
        tracing::info!(
            "Backfill complete: {processed} processed, {errors} errors, {skipped} skipped"
        );

        Ok(BackfillProgress {
            total,
            processed,
            errors,
            skipped,
        })
    }

    /// Backfill SYNAPSE relationships for notes that have embeddings but no
    /// outgoing SYNAPSE edges yet.
    ///
    /// For each such note, retrieves its stored embedding, runs a vector
    /// similarity search to find nearest neighbours, filters by
    /// `min_similarity`, and creates bidirectional SYNAPSE relations.
    ///
    /// This is idempotent: notes that already have synapses are skipped
    /// (they don't appear in `list_notes_needing_synapses`). On re-run,
    /// only newly embedded notes or notes whose synapses were pruned are
    /// processed.
    ///
    /// Also calls `init_note_energy()` first to ensure all notes have an
    /// energy value.
    pub async fn backfill_synapses(
        &self,
        batch_size: usize,
        min_similarity: f64,
        max_neighbors: usize,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<SynapseBackfillProgress> {
        let batch_size = if batch_size == 0 { 50 } else { batch_size };
        let min_similarity = if min_similarity <= 0.0 { 0.75 } else { min_similarity };
        let max_neighbors = if max_neighbors == 0 { 10 } else { max_neighbors };

        // Phase 1: init energy on all notes that don't have it yet
        let energy_init = self.neo4j.init_note_energy().await?;
        if energy_init > 0 {
            tracing::info!("Synapse backfill: initialized energy on {energy_init} notes");
        }

        // Phase 2: get total count of notes needing synapses
        let (_, total) = self.neo4j.list_notes_needing_synapses(0, 0).await?;
        if total == 0 {
            tracing::info!("Synapse backfill: all notes with embeddings already have synapses");
            return Ok(SynapseBackfillProgress {
                total,
                processed: 0,
                synapses_created: 0,
                errors: 0,
                skipped: 0,
                energy_initialized: energy_init,
            });
        }

        tracing::info!("Synapse backfill: {total} notes need synapses");
        let mut processed = 0usize;
        let mut synapses_created = 0usize;
        let mut errors = 0usize;

        loop {
            // Check cancellation
            if let Some(flag) = cancel {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    tracing::info!("Synapse backfill cancelled at {processed}/{total}");
                    break;
                }
            }

            // Fetch next batch (always offset 0, processed notes disappear from results)
            let (batch, remaining) = self.neo4j.list_notes_needing_synapses(batch_size, 0).await?;
            if batch.is_empty() || remaining == 0 {
                break;
            }

            for note in &batch {
                // Check cancellation per-note for responsiveness
                if let Some(flag) = cancel {
                    if flag.load(std::sync::atomic::Ordering::Relaxed) {
                        tracing::info!("Synapse backfill cancelled at {processed}/{total}");
                        let skipped = total.saturating_sub(processed + errors);
                        return Ok(SynapseBackfillProgress {
                            total,
                            processed,
                            synapses_created,
                            errors,
                            skipped,
                            energy_initialized: energy_init,
                        });
                    }
                }

                // Get the stored embedding for this note
                let embedding = match self.neo4j.get_note_embedding(note.id).await {
                    Ok(Some(emb)) => emb,
                    Ok(None) => {
                        tracing::warn!(
                            note_id = %note.id,
                            "Synapse backfill: note listed as needing synapses but has no embedding, skipping"
                        );
                        errors += 1;
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!(
                            note_id = %note.id,
                            error = %e,
                            "Synapse backfill: failed to get embedding"
                        );
                        errors += 1;
                        continue;
                    }
                };

                // Vector search for nearest neighbours
                let neighbors = match self
                    .neo4j
                    .vector_search_notes(&embedding, max_neighbors + 1, note.project_id, None)
                    .await
                {
                    Ok(results) => results,
                    Err(e) => {
                        tracing::warn!(
                            note_id = %note.id,
                            error = %e,
                            "Synapse backfill: vector search failed"
                        );
                        errors += 1;
                        continue;
                    }
                };

                // Filter: exclude self, apply min_similarity threshold
                let synapse_targets: Vec<(Uuid, f64)> = neighbors
                    .into_iter()
                    .filter(|(n, score)| n.id != note.id && *score >= min_similarity)
                    .map(|(n, score)| (n.id, score))
                    .collect();

                if !synapse_targets.is_empty() {
                    match self.neo4j.create_synapses(note.id, &synapse_targets).await {
                        Ok(created) => {
                            synapses_created += created;
                        }
                        Err(e) => {
                            tracing::warn!(
                                note_id = %note.id,
                                error = %e,
                                "Synapse backfill: failed to create synapses"
                            );
                            errors += 1;
                            continue;
                        }
                    }
                }

                processed += 1;
            }

            tracing::info!(
                "Synapse backfill: {processed}/{total} notes processed, {synapses_created} synapses created ({errors} errors)"
            );
        }

        let skipped = total.saturating_sub(processed + errors);
        tracing::info!(
            "Synapse backfill complete: {processed} processed, {synapses_created} synapses, {errors} errors, {skipped} skipped"
        );

        Ok(SynapseBackfillProgress {
            total,
            processed,
            synapses_created,
            errors,
            skipped,
            energy_initialized: energy_init,
        })
    }
}

/// Progress report for the embedding backfill operation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BackfillProgress {
    /// Total number of notes that needed embedding at start
    pub total: usize,
    /// Number of notes successfully embedded
    pub processed: usize,
    /// Number of notes that failed embedding
    pub errors: usize,
    /// Number of notes skipped (cancelled before processing)
    pub skipped: usize,
}

/// Progress report for the synapse backfill operation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SynapseBackfillProgress {
    /// Total number of notes that needed synapses at start
    pub total: usize,
    /// Number of notes successfully processed
    pub processed: usize,
    /// Number of SYNAPSE relationships created
    pub synapses_created: usize,
    /// Number of notes that failed processing
    pub errors: usize,
    /// Number of notes skipped (cancelled before processing)
    pub skipped: usize,
    /// Number of notes that had energy initialized
    pub energy_initialized: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;

    /// Helper: build a NoteManager backed by mock stores, with a pre-seeded project.
    /// Returns (NoteManager, project_id).
    async fn create_note_manager() -> (NoteManager, Uuid) {
        let state = mock_app_state();
        let project = test_project();
        let project_id = project.id;
        // Seed the project so note_to_document can resolve the slug
        state.neo4j.create_project(&project).await.unwrap();
        let manager = NoteManager::new(state.neo4j.clone(), state.meili.clone());
        (manager, project_id)
    }

    /// Helper: build a CreateNoteRequest with minimal required fields.
    fn make_create_request(project_id: Uuid, content: &str) -> CreateNoteRequest {
        CreateNoteRequest {
            project_id: Some(project_id),
            note_type: NoteType::Guideline,
            content: content.to_string(),
            importance: Some(NoteImportance::High),
            scope: None,
            tags: Some(vec!["test".to_string()]),
            anchors: None,
            assertion_rule: None,
        }
    }

    // ====================================================================
    // Note CRUD
    // ====================================================================

    #[tokio::test]
    async fn test_create_note() {
        let (mgr, pid) = create_note_manager().await;
        let req = make_create_request(pid, "Always use Result for errors");

        let note = mgr.create_note(req, "agent-1").await.unwrap();

        assert_eq!(note.project_id, Some(pid));
        assert_eq!(note.note_type, NoteType::Guideline);
        assert_eq!(note.content, "Always use Result for errors");
        assert_eq!(note.importance, NoteImportance::High);
        assert_eq!(note.status, NoteStatus::Active);
        assert_eq!(note.created_by, "agent-1");

        // Verify stored in Neo4j
        let stored = mgr.get_note(note.id).await.unwrap();
        assert!(stored.is_some());
        assert_eq!(stored.unwrap().id, note.id);
    }

    #[tokio::test]
    async fn test_get_note() {
        let (mgr, pid) = create_note_manager().await;
        let req = make_create_request(pid, "Check all fields are returned");
        let created = mgr.create_note(req, "agent-1").await.unwrap();

        let fetched = mgr.get_note(created.id).await.unwrap().unwrap();

        assert_eq!(fetched.id, created.id);
        assert_eq!(fetched.content, "Check all fields are returned");
        assert_eq!(fetched.note_type, NoteType::Guideline);
        assert_eq!(fetched.importance, NoteImportance::High);
        assert_eq!(fetched.tags, vec!["test".to_string()]);
    }

    #[tokio::test]
    async fn test_get_note_not_found() {
        let (mgr, _pid) = create_note_manager().await;
        let non_existent_id = Uuid::new_v4();

        let result = mgr.get_note(non_existent_id).await.unwrap();

        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_update_note() {
        let (mgr, pid) = create_note_manager().await;
        let req = make_create_request(pid, "Original content");
        let created = mgr.create_note(req, "agent-1").await.unwrap();

        let update = UpdateNoteRequest {
            content: Some("Updated content".to_string()),
            importance: Some(NoteImportance::Critical),
            status: None,
            tags: None,
        };
        let updated = mgr.update_note(created.id, update).await.unwrap().unwrap();

        assert_eq!(updated.content, "Updated content");
        assert_eq!(updated.importance, NoteImportance::Critical);
        // Status should remain unchanged
        assert_eq!(updated.status, NoteStatus::Active);
    }

    #[tokio::test]
    async fn test_delete_note() {
        let (mgr, pid) = create_note_manager().await;
        let req = make_create_request(pid, "To be deleted");
        let created = mgr.create_note(req, "agent-1").await.unwrap();

        let deleted = mgr.delete_note(created.id).await.unwrap();
        assert!(deleted);

        // Verify removed from Neo4j
        let gone = mgr.get_note(created.id).await.unwrap();
        assert!(gone.is_none());

        // Deleting again returns false
        let deleted_again = mgr.delete_note(created.id).await.unwrap();
        assert!(!deleted_again);
    }

    // ====================================================================
    // Note Listing
    // ====================================================================

    #[tokio::test]
    async fn test_list_notes() {
        let (mgr, pid) = create_note_manager().await;

        // Create 3 notes
        for i in 0..3 {
            let req = make_create_request(pid, &format!("Note {}", i));
            mgr.create_note(req, "agent-1").await.unwrap();
        }

        let filters = NoteFilters {
            limit: Some(2),
            ..Default::default()
        };
        let (notes, total) = mgr.list_notes(None, None, &filters).await.unwrap();

        assert_eq!(total, 3);
        assert_eq!(notes.len(), 2); // limited to 2
    }

    #[tokio::test]
    async fn test_list_project_notes() {
        let state = mock_app_state();

        // Create two projects
        let project_a = test_project_named("Alpha");
        let project_b = test_project_named("Beta");
        state.neo4j.create_project(&project_a).await.unwrap();
        state.neo4j.create_project(&project_b).await.unwrap();

        let mgr = NoteManager::new(state.neo4j.clone(), state.meili.clone());

        // Create notes for each project
        let req_a = make_create_request(project_a.id, "Alpha note");
        let req_b = make_create_request(project_b.id, "Beta note");
        mgr.create_note(req_a, "agent-1").await.unwrap();
        mgr.create_note(req_b, "agent-1").await.unwrap();

        let filters = NoteFilters::default();
        let (notes, total) = mgr
            .list_project_notes(project_a.id, &filters)
            .await
            .unwrap();

        assert_eq!(total, 1);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].project_id, Some(project_a.id));
    }

    // ====================================================================
    // Linking
    // ====================================================================

    #[tokio::test]
    async fn test_link_note_to_entity() {
        let (mgr, pid) = create_note_manager().await;
        let req = make_create_request(pid, "Link me to a file");
        let note = mgr.create_note(req, "agent-1").await.unwrap();

        let link = LinkNoteRequest {
            entity_type: EntityType::File,
            entity_id: "src/main.rs".to_string(),
        };
        mgr.link_note_to_entity(note.id, &link).await.unwrap();

        let anchors = mgr.get_note_anchors(note.id).await.unwrap();
        assert_eq!(anchors.len(), 1);
        assert_eq!(anchors[0].entity_type, EntityType::File);
        assert_eq!(anchors[0].entity_id, "src/main.rs");
    }

    #[tokio::test]
    async fn test_unlink_note_from_entity() {
        let (mgr, pid) = create_note_manager().await;
        let req = make_create_request(pid, "Unlink me");
        let note = mgr.create_note(req, "agent-1").await.unwrap();

        // Link first
        let link = LinkNoteRequest {
            entity_type: EntityType::File,
            entity_id: "src/lib.rs".to_string(),
        };
        mgr.link_note_to_entity(note.id, &link).await.unwrap();

        // Verify linked
        let anchors = mgr.get_note_anchors(note.id).await.unwrap();
        assert_eq!(anchors.len(), 1);

        // Unlink
        mgr.unlink_note_from_entity(note.id, &EntityType::File, "src/lib.rs")
            .await
            .unwrap();

        let anchors = mgr.get_note_anchors(note.id).await.unwrap();
        assert!(anchors.is_empty());
    }

    // ====================================================================
    // Lifecycle
    // ====================================================================

    #[tokio::test]
    async fn test_confirm_note() {
        let (mgr, pid) = create_note_manager().await;
        let req = make_create_request(pid, "Confirm me");
        let note = mgr.create_note(req, "agent-1").await.unwrap();

        let confirmed = mgr
            .confirm_note(note.id, "reviewer")
            .await
            .unwrap()
            .unwrap();

        assert_eq!(confirmed.staleness_score, 0.0);
        assert_eq!(confirmed.last_confirmed_by, Some("reviewer".to_string()));
        assert!(confirmed.last_confirmed_at.is_some());
    }

    #[tokio::test]
    async fn test_invalidate_note() {
        let (mgr, pid) = create_note_manager().await;
        let req = make_create_request(pid, "Obsolete this note");
        let note = mgr.create_note(req, "agent-1").await.unwrap();

        let invalidated = mgr
            .invalidate_note(note.id, "No longer relevant", "reviewer")
            .await
            .unwrap()
            .unwrap();

        assert_eq!(invalidated.status, NoteStatus::Obsolete);
    }

    #[tokio::test]
    async fn test_supersede_note() {
        let (mgr, pid) = create_note_manager().await;
        let old_req = make_create_request(pid, "Old guideline");
        let old_note = mgr.create_note(old_req, "agent-1").await.unwrap();

        let new_req = make_create_request(pid, "New improved guideline");
        let new_note = mgr
            .supersede_note(old_note.id, new_req, "agent-2")
            .await
            .unwrap();

        assert_eq!(new_note.content, "New improved guideline");
        assert_eq!(new_note.supersedes, Some(old_note.id));

        // Old note should be marked as obsolete with superseded_by set
        let old_after = mgr.get_note(old_note.id).await.unwrap().unwrap();
        assert_eq!(old_after.status, NoteStatus::Obsolete);
        assert_eq!(old_after.superseded_by, Some(new_note.id));
    }

    #[tokio::test]
    async fn test_get_notes_needing_review() {
        let (mgr, pid) = create_note_manager().await;

        // Create a normal active note
        let req1 = make_create_request(pid, "Active note");
        mgr.create_note(req1, "agent-1").await.unwrap();

        // Create a note and then mark it as needing review
        let req2 = make_create_request(pid, "Stale note");
        let note2 = mgr.create_note(req2, "agent-1").await.unwrap();
        mgr.update_note(
            note2.id,
            UpdateNoteRequest {
                status: Some(NoteStatus::NeedsReview),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        // Create another note and mark it as stale
        let req3 = make_create_request(pid, "Also stale");
        let note3 = mgr.create_note(req3, "agent-1").await.unwrap();
        mgr.update_note(
            note3.id,
            UpdateNoteRequest {
                status: Some(NoteStatus::Stale),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let needing_review = mgr.get_notes_needing_review(Some(pid)).await.unwrap();

        assert_eq!(needing_review.len(), 2);
        for n in &needing_review {
            assert!(
                n.status == NoteStatus::NeedsReview || n.status == NoteStatus::Stale,
                "Expected needs_review or stale, got {:?}",
                n.status
            );
        }
    }

    // ====================================================================
    // Search
    // ====================================================================

    #[tokio::test]
    async fn test_search_notes() {
        let (mgr, pid) = create_note_manager().await;

        let req1 = make_create_request(pid, "Use async/await for all IO operations");
        mgr.create_note(req1, "agent-1").await.unwrap();

        let req2 = make_create_request(pid, "Prefer iterators over manual loops");
        mgr.create_note(req2, "agent-1").await.unwrap();

        let filters = NoteFilters::default();
        let hits = mgr.search_notes("async", &filters).await.unwrap();

        assert_eq!(hits.len(), 1);
        assert!(hits[0].note.content.contains("async"));
        assert!(hits[0].score > 0.0);
    }

    // ====================================================================
    // Context
    // ====================================================================

    #[tokio::test]
    async fn test_get_context_notes() {
        let (mgr, pid) = create_note_manager().await;

        // Create a note and link it to a file
        let req = make_create_request(pid, "Important context about main.rs");
        let note = mgr.create_note(req, "agent-1").await.unwrap();

        let link = LinkNoteRequest {
            entity_type: EntityType::File,
            entity_id: "src/main.rs".to_string(),
        };
        mgr.link_note_to_entity(note.id, &link).await.unwrap();

        let ctx = mgr
            .get_context_notes(&EntityType::File, "src/main.rs", 3, 0.0)
            .await
            .unwrap();

        assert_eq!(ctx.direct_notes.len(), 1);
        assert_eq!(ctx.direct_notes[0].id, note.id);
        // Propagated notes may be empty in the mock (no graph traversal)
        assert_eq!(
            ctx.total_count,
            ctx.direct_notes.len() + ctx.propagated_notes.len()
        );
    }

    // ====================================================================
    // Global notes (no project_id)
    // ====================================================================

    fn make_global_request(content: &str) -> CreateNoteRequest {
        CreateNoteRequest {
            project_id: None,
            note_type: NoteType::Guideline,
            content: content.to_string(),
            importance: Some(NoteImportance::High),
            scope: None,
            tags: Some(vec!["global".to_string()]),
            anchors: None,
            assertion_rule: None,
        }
    }

    #[tokio::test]
    async fn test_create_global_note() {
        let state = mock_app_state();
        let mgr = NoteManager::new(state.neo4j.clone(), state.meili.clone());

        let req = make_global_request("Always use kebab-case for slugs");
        let note = mgr.create_note(req, "agent-1").await.unwrap();

        assert!(note.project_id.is_none());
        assert_eq!(note.note_type, NoteType::Guideline);
        assert_eq!(note.content, "Always use kebab-case for slugs");
    }

    #[tokio::test]
    async fn test_list_notes_global_only() {
        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        let mgr = NoteManager::new(state.neo4j.clone(), state.meili.clone());

        // Create a project note and a global note
        let project_req = make_create_request(project.id, "Project-specific note");
        let global_req = make_global_request("Global team convention");
        mgr.create_note(project_req, "agent-1").await.unwrap();
        mgr.create_note(global_req, "agent-1").await.unwrap();

        // List all notes — should return both
        let all_filters = NoteFilters::default();
        let (all_notes, all_total) = mgr.list_notes(None, None, &all_filters).await.unwrap();
        assert_eq!(all_total, 2);
        assert_eq!(all_notes.len(), 2);

        // List global only — should return only the global one
        let global_filters = NoteFilters {
            global_only: Some(true),
            ..Default::default()
        };
        let (global_notes, global_total) =
            mgr.list_notes(None, None, &global_filters).await.unwrap();
        assert_eq!(global_total, 1);
        assert_eq!(global_notes.len(), 1);
        assert!(global_notes[0].project_id.is_none());
        assert_eq!(global_notes[0].content, "Global team convention");

        // List project-specific — should return only the project one
        let (project_notes, project_total) = mgr
            .list_notes(Some(project.id), None, &all_filters)
            .await
            .unwrap();
        assert_eq!(project_total, 1);
        assert_eq!(project_notes.len(), 1);
        assert_eq!(project_notes[0].project_id, Some(project.id));
    }

    // ====================================================================
    // Embedding integration
    // ====================================================================

    /// Helper: build a NoteManager backed by mock stores **with** a MockEmbeddingProvider.
    /// Returns (NoteManager, project_id, Arc<MockGraphStore>) so tests can inspect embeddings.
    async fn create_note_manager_with_embeddings(
    ) -> (NoteManager, Uuid, Arc<crate::neo4j::mock::MockGraphStore>) {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let meili = Arc::new(crate::meilisearch::mock::MockSearchStore::new());
        let project = test_project();
        let project_id = project.id;
        graph.create_project(&project).await.unwrap();

        let provider = Arc::new(crate::embeddings::MockEmbeddingProvider::new(768));

        let manager = NoteManager::new(graph.clone() as Arc<dyn crate::neo4j::GraphStore>, meili)
            .with_embedding_provider(provider);

        (manager, project_id, graph)
    }

    #[tokio::test]
    async fn test_create_note_generates_embedding() {
        let (mgr, pid, graph) = create_note_manager_with_embeddings().await;
        let req = make_create_request(pid, "Always handle errors with Result");

        let note = mgr.create_note(req, "agent-1").await.unwrap();

        // Verify embedding was stored in the graph
        let embeddings = graph.note_embeddings.read().await;
        let (embedding, model) = embeddings
            .get(&note.id)
            .expect("embedding should be stored");
        assert_eq!(embedding.len(), 768);
        assert_eq!(model, "mock-hash-embedding");
    }

    #[tokio::test]
    async fn test_update_note_content_regenerates_embedding() {
        let (mgr, pid, graph) = create_note_manager_with_embeddings().await;
        let req = make_create_request(pid, "Original content for embedding");
        let note = mgr.create_note(req, "agent-1").await.unwrap();

        // Grab original embedding
        let original_embedding = {
            let embeddings = graph.note_embeddings.read().await;
            embeddings.get(&note.id).unwrap().0.clone()
        };

        // Update content — should regenerate embedding
        let update = UpdateNoteRequest {
            content: Some("Completely different content for re-embedding".to_string()),
            importance: None,
            status: None,
            tags: None,
        };
        mgr.update_note(note.id, update).await.unwrap();

        // Verify embedding changed (different content → different hash → different vector)
        let embeddings = graph.note_embeddings.read().await;
        let (new_embedding, model) = embeddings.get(&note.id).unwrap();
        assert_eq!(new_embedding.len(), 768);
        assert_eq!(model, "mock-hash-embedding");
        assert_ne!(
            &original_embedding, new_embedding,
            "embedding should change when content changes"
        );
    }

    #[tokio::test]
    async fn test_update_note_without_content_keeps_embedding() {
        let (mgr, pid, graph) = create_note_manager_with_embeddings().await;
        let req = make_create_request(pid, "Stable content");
        let note = mgr.create_note(req, "agent-1").await.unwrap();

        // Grab original embedding
        let original_embedding = {
            let embeddings = graph.note_embeddings.read().await;
            embeddings.get(&note.id).unwrap().0.clone()
        };

        // Update only importance — should NOT regenerate embedding
        let update = UpdateNoteRequest {
            content: None,
            importance: Some(NoteImportance::Critical),
            status: None,
            tags: None,
        };
        mgr.update_note(note.id, update).await.unwrap();

        // Verify embedding is unchanged
        let embeddings = graph.note_embeddings.read().await;
        let (same_embedding, _) = embeddings.get(&note.id).unwrap();
        assert_eq!(
            &original_embedding, same_embedding,
            "embedding should not change when content is unchanged"
        );
    }

    #[tokio::test]
    async fn test_create_note_without_provider_no_embedding() {
        // Use the standard helper (no embedding provider)
        let (mgr, pid) = create_note_manager().await;
        let req = make_create_request(pid, "No embedding expected");

        let note = mgr.create_note(req, "agent-1").await.unwrap();

        // NoteManager without embedding provider — note should still be created
        let stored = mgr.get_note(note.id).await.unwrap();
        assert!(stored.is_some());
        // We can't directly check embeddings in the standard helper (no Arc<MockGraphStore> access),
        // but the fact that create_note succeeds without a provider is the key assertion.
    }

    #[tokio::test]
    async fn test_supersede_note_embeds_new_note() {
        let (mgr, pid, graph) = create_note_manager_with_embeddings().await;
        let old_req = make_create_request(pid, "Old guideline content");
        let old_note = mgr.create_note(old_req, "agent-1").await.unwrap();

        let new_req = make_create_request(pid, "New improved guideline content");
        let new_note = mgr
            .supersede_note(old_note.id, new_req, "agent-2")
            .await
            .unwrap();

        // Both old and new notes should have embeddings
        let embeddings = graph.note_embeddings.read().await;
        assert!(
            embeddings.contains_key(&old_note.id),
            "old note should have embedding"
        );
        assert!(
            embeddings.contains_key(&new_note.id),
            "new note should have embedding"
        );

        // Embeddings should be different (different content)
        let old_emb = &embeddings.get(&old_note.id).unwrap().0;
        let new_emb = &embeddings.get(&new_note.id).unwrap().0;
        assert_ne!(
            old_emb, new_emb,
            "different content should produce different embeddings"
        );
    }

    // ====================================================================
    // Backfill
    // ====================================================================

    /// Helper: create a NoteManager WITHOUT embedding provider, create notes,
    /// then return a new NoteManager WITH embedding provider for backfill testing.
    async fn create_notes_then_add_embeddings(
        count: usize,
    ) -> (
        NoteManager,
        Vec<Uuid>,
        Arc<crate::neo4j::mock::MockGraphStore>,
    ) {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let meili = Arc::new(crate::meilisearch::mock::MockSearchStore::new());
        let project = test_project();
        let project_id = project.id;
        graph.create_project(&project).await.unwrap();

        // Create notes WITHOUT embedding provider (simulates existing notes)
        let mgr_no_embed = NoteManager::new(
            graph.clone() as Arc<dyn crate::neo4j::GraphStore>,
            meili.clone(),
        );
        let mut note_ids = Vec::new();
        for i in 0..count {
            let req = make_create_request(project_id, &format!("Backfill note {i}"));
            let note = mgr_no_embed.create_note(req, "agent-1").await.unwrap();
            note_ids.push(note.id);
        }

        // Verify no embeddings exist
        let embeddings = graph.note_embeddings.read().await;
        assert_eq!(
            embeddings.len(),
            0,
            "no embeddings should exist before backfill"
        );
        drop(embeddings);

        // Now create a NoteManager WITH embedding provider
        let provider = Arc::new(crate::embeddings::MockEmbeddingProvider::new(768));
        let mgr = NoteManager::new(graph.clone() as Arc<dyn crate::neo4j::GraphStore>, meili)
            .with_embedding_provider(provider);

        (mgr, note_ids, graph)
    }

    #[tokio::test]
    async fn test_backfill_embeddings_all_notes() {
        let (mgr, note_ids, graph) = create_notes_then_add_embeddings(10).await;

        let progress = mgr.backfill_embeddings(50, None).await.unwrap();

        assert_eq!(progress.total, 10);
        assert_eq!(progress.processed, 10);
        assert_eq!(progress.errors, 0);

        // Verify all notes have embeddings
        let embeddings = graph.note_embeddings.read().await;
        for id in &note_ids {
            assert!(
                embeddings.contains_key(id),
                "note {id} should have embedding"
            );
        }
    }

    #[tokio::test]
    async fn test_backfill_is_idempotent() {
        let (mgr, _note_ids, _graph) = create_notes_then_add_embeddings(5).await;

        // First backfill
        let p1 = mgr.backfill_embeddings(50, None).await.unwrap();
        assert_eq!(p1.processed, 5);

        // Second backfill — should do nothing
        let p2 = mgr.backfill_embeddings(50, None).await.unwrap();
        assert_eq!(p2.total, 0);
        assert_eq!(p2.processed, 0);
    }

    #[tokio::test]
    async fn test_backfill_respects_batch_size() {
        let (mgr, _note_ids, graph) = create_notes_then_add_embeddings(7).await;

        // Backfill with batch size 3 — should still process all 7
        let progress = mgr.backfill_embeddings(3, None).await.unwrap();

        assert_eq!(progress.total, 7);
        assert_eq!(progress.processed, 7);

        let embeddings = graph.note_embeddings.read().await;
        assert_eq!(embeddings.len(), 7);
    }

    #[tokio::test]
    async fn test_backfill_cancellation() {
        let (mgr, _note_ids, graph) = create_notes_then_add_embeddings(10).await;

        // Set cancel flag immediately
        let cancel = std::sync::atomic::AtomicBool::new(true);
        let progress = mgr.backfill_embeddings(3, Some(&cancel)).await.unwrap();

        // Should have processed 0 notes (cancelled before first batch)
        assert_eq!(progress.processed, 0);
        let embeddings = graph.note_embeddings.read().await;
        assert_eq!(embeddings.len(), 0);
    }

    #[tokio::test]
    async fn test_backfill_without_provider_errors() {
        let (mgr, _pid) = create_note_manager().await;
        let result = mgr.backfill_embeddings(50, None).await;
        assert!(
            result.is_err(),
            "backfill should fail without embedding provider"
        );
    }

    #[tokio::test]
    async fn test_global_notes_still_work_with_project_notes() {
        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        let mgr = NoteManager::new(state.neo4j.clone(), state.meili.clone());

        // Create both types
        let proj_note = mgr
            .create_note(make_create_request(project.id, "Project rule"), "agent-1")
            .await
            .unwrap();
        let global_note = mgr
            .create_note(make_global_request("Global rule"), "agent-1")
            .await
            .unwrap();

        // Both should be retrievable by ID
        let fetched_proj = mgr.get_note(proj_note.id).await.unwrap().unwrap();
        let fetched_global = mgr.get_note(global_note.id).await.unwrap().unwrap();

        assert_eq!(fetched_proj.project_id, Some(project.id));
        assert!(fetched_global.project_id.is_none());

        // Both should be updatable
        let update = UpdateNoteRequest {
            content: Some("Updated global rule".to_string()),
            importance: None,
            status: None,
            tags: None,
        };
        let updated = mgr
            .update_note(global_note.id, update)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.content, "Updated global rule");
        assert!(updated.project_id.is_none());
    }

    // ====================================================================
    // Semantic (vector) search
    // ====================================================================

    #[tokio::test]
    async fn test_semantic_search_notes_returns_similar_notes() {
        let (mgr, pid, _graph) = create_note_manager_with_embeddings().await;

        // Create several notes with different content
        let req1 = make_create_request(pid, "Always handle errors with Result type in Rust");
        let note1 = mgr.create_note(req1, "agent-1").await.unwrap();

        let req2 = make_create_request(pid, "Error handling best practices and conventions");
        let note2 = mgr.create_note(req2, "agent-1").await.unwrap();

        let req3 = make_create_request(pid, "Database connection pooling and timeout settings");
        let _note3 = mgr.create_note(req3, "agent-1").await.unwrap();

        // Search for error-handling related notes
        let results = mgr
            .semantic_search_notes("how to handle errors", None, None, Some(10))
            .await
            .unwrap();

        // Should return results (mock provider uses deterministic hash-based embeddings)
        assert!(!results.is_empty(), "semantic search should return results");

        // Each result should have a finite score (mock cosine similarity can be near-zero
        // for random high-dimensional vectors, which is mathematically correct)
        for hit in &results {
            assert!(hit.score.is_finite(), "score should be finite");
        }

        // Verify the notes we created are in results
        let result_ids: Vec<Uuid> = results.iter().map(|h| h.note.id).collect();
        assert!(
            result_ids.contains(&note1.id) || result_ids.contains(&note2.id),
            "should find at least one of the error-handling notes"
        );
    }

    #[tokio::test]
    async fn test_semantic_search_notes_without_provider_falls_back() {
        let (mgr, pid) = create_note_manager().await;

        // Create a note
        let req = make_create_request(pid, "Test fallback content");
        mgr.create_note(req, "agent-1").await.unwrap();

        // Semantic search without provider should fall back to BM25 (no panic, no error)
        let results = mgr
            .semantic_search_notes("test", None, None, Some(10))
            .await;

        // Should not error — falls back gracefully to Meilisearch
        assert!(results.is_ok(), "should fall back to BM25 without error");
    }

    #[tokio::test]
    async fn test_semantic_search_notes_with_project_filter() {
        let (mgr, pid, _graph) = create_note_manager_with_embeddings().await;

        let req = make_create_request(pid, "Project-specific guideline");
        mgr.create_note(req, "agent-1").await.unwrap();

        // Search with project filter
        let results = mgr
            .semantic_search_notes("guideline", Some(pid), None, Some(10))
            .await
            .unwrap();

        // All results should belong to the project
        for hit in &results {
            assert_eq!(
                hit.note.project_id,
                Some(pid),
                "all results should belong to the filtered project"
            );
        }
    }
}
