//! Note Manager - CRUD operations for Knowledge Notes
//!
//! Provides high-level operations for creating, reading, updating, and deleting notes,
//! including linking notes to entities and managing note lifecycle.

use super::models::*;
use crate::events::{CrudAction, CrudEvent, EntityType as EventEntityType, EventBus};
use crate::meilisearch::indexes::NoteDocument;
use crate::meilisearch::SearchStore;
use crate::neo4j::GraphStore;
use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;

/// Manager for Knowledge Notes operations
pub struct NoteManager {
    neo4j: Arc<dyn GraphStore>,
    meilisearch: Arc<dyn SearchStore>,
    event_bus: Option<Arc<EventBus>>,
}

impl NoteManager {
    /// Create a new NoteManager
    pub fn new(neo4j: Arc<dyn GraphStore>, meilisearch: Arc<dyn SearchStore>) -> Self {
        Self {
            neo4j,
            meilisearch,
            event_bus: None,
        }
    }

    /// Create a new NoteManager with an event bus
    pub fn with_event_bus(
        neo4j: Arc<dyn GraphStore>,
        meilisearch: Arc<dyn SearchStore>,
        event_bus: Arc<EventBus>,
    ) -> Self {
        Self {
            neo4j,
            meilisearch,
            event_bus: Some(event_bus),
        }
    }

    /// Emit a CRUD event (no-op if event_bus is None)
    fn emit(&self, event: crate::events::CrudEvent) {
        if let Some(bus) = &self.event_bus {
            bus.emit(event);
        }
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

        self.emit(
            CrudEvent::new(
                EventEntityType::Note,
                CrudAction::Created,
                note.id.to_string(),
            )
            .with_payload(serde_json::json!({"note_type": note.note_type.to_string()}))
            .with_project_id(note.project_id.to_string()),
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

            self.emit(
                CrudEvent::new(EventEntityType::Note, CrudAction::Updated, id.to_string())
                    .with_project_id(note.project_id.to_string()),
            );
        }

        Ok(updated)
    }

    /// Delete a note
    pub async fn delete_note(&self, id: Uuid) -> Result<bool> {
        // Delete from Neo4j
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
        filters: &NoteFilters,
    ) -> Result<(Vec<Note>, usize)> {
        self.neo4j.list_notes(project_id, filters).await
    }

    /// List notes for a specific project
    pub async fn list_project_notes(
        &self,
        project_id: Uuid,
        filters: &NoteFilters,
    ) -> Result<(Vec<Note>, usize)> {
        self.neo4j.list_notes(Some(project_id), filters).await
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
                .with_project_id(note.project_id.to_string()),
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
            // Try to get from project
            if let Ok(Some(project)) = self.neo4j.get_project(note.project_id).await {
                project.slug
            } else {
                String::new()
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
            project_id: note.project_id.to_string(),
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
            project_id,
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

        assert_eq!(note.project_id, pid);
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
        let (notes, total) = mgr.list_notes(None, &filters).await.unwrap();

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
        assert_eq!(notes[0].project_id, project_a.id);
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
}
