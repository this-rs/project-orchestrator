//! Note Manager - CRUD operations for Knowledge Notes
//!
//! Provides high-level operations for creating, reading, updating, and deleting notes,
//! including linking notes to entities and managing note lifecycle.

use super::models::*;
use crate::meilisearch::{client::MeiliClient, indexes::NoteDocument};
use crate::neo4j::client::Neo4jClient;
use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;

/// Manager for Knowledge Notes operations
pub struct NoteManager {
    neo4j: Arc<Neo4jClient>,
    meilisearch: Arc<MeiliClient>,
}

impl NoteManager {
    /// Create a new NoteManager
    pub fn new(neo4j: Arc<Neo4jClient>, meilisearch: Arc<MeiliClient>) -> Self {
        Self { neo4j, meilisearch }
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
            .await
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
            .await
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
    #[test]
    fn test_note_manager_new() {
        // This is a compile-time test to ensure the struct is properly defined
        // Runtime tests would require mock implementations
    }
}
