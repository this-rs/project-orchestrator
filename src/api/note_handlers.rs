//! API handlers for Knowledge Notes

use super::handlers::{AppError, OrchestratorState};
use super::{PaginatedResponse, PaginationParams, SearchFilter};
use crate::notes::{
    CreateAnchorRequest, CreateNoteRequest, EntityType, LinkNoteRequest, Note, NoteContextResponse,
    NoteFilters, NoteImportance, NoteScope, NoteSearchHit, NoteStatus, NoteType, PropagatedNote,
    UpdateNoteRequest,
};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Query Parameters
// ============================================================================

/// Query parameters for listing notes
#[derive(Debug, Deserialize, Default)]
pub struct NotesListQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
    #[serde(flatten)]
    pub search_filter: SearchFilter,
    pub note_type: Option<String>,
    pub status: Option<String>,
    pub importance: Option<String>,
    pub project_id: Option<Uuid>,
    pub min_staleness: Option<f64>,
    pub max_staleness: Option<f64>,
    pub tags: Option<String>,
    pub global_only: Option<bool>,
}

impl NotesListQuery {
    /// Convert to NoteFilters
    pub fn to_note_filters(&self) -> NoteFilters {
        NoteFilters {
            note_type: self
                .note_type
                .as_ref()
                .and_then(|s| s.parse::<NoteType>().ok())
                .map(|t| vec![t]),
            status: self.status.as_ref().map(|s| {
                s.split(',')
                    .filter_map(|s| s.trim().parse::<NoteStatus>().ok())
                    .collect()
            }),
            importance: self
                .importance
                .as_ref()
                .and_then(|s| s.parse::<NoteImportance>().ok())
                .map(|i| vec![i]),
            min_staleness: self.min_staleness,
            max_staleness: self.max_staleness,
            tags: self
                .tags
                .as_ref()
                .map(|t| t.split(',').map(|s| s.trim().to_string()).collect()),
            search: self.search_filter.search.clone(),
            limit: Some(self.pagination.validated_limit() as i64),
            offset: Some(self.pagination.offset as i64),
            global_only: self.global_only,
            scope_type: None,
            sort_by: self.pagination.sort_by.clone(),
            sort_order: Some(self.pagination.sort_order.clone()),
        }
    }
}

/// Query parameters for searching notes
#[derive(Debug, Deserialize)]
pub struct NotesSearchQuery {
    pub q: String,
    pub project_slug: Option<String>,
    pub note_type: Option<String>,
    pub status: Option<String>,
    pub importance: Option<String>,
    pub limit: Option<usize>,
}

/// Query parameters for getting context notes
#[derive(Debug, Deserialize)]
pub struct ContextNotesQuery {
    pub entity_type: String,
    pub entity_id: String,
    pub max_depth: Option<u32>,
    pub min_score: Option<f64>,
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Request to create a note
#[derive(Debug, Deserialize)]
pub struct CreateNoteBody {
    pub project_id: Option<Uuid>,
    pub note_type: NoteType,
    pub content: String,
    pub importance: Option<NoteImportance>,
    pub scope: Option<NoteScope>,
    pub tags: Option<Vec<String>>,
    pub anchors: Option<Vec<CreateAnchorRequest>>,
    pub assertion_rule: Option<crate::notes::AssertionRule>,
}

/// Request to update a note
#[derive(Debug, Deserialize)]
pub struct UpdateNoteBody {
    pub content: Option<String>,
    pub importance: Option<NoteImportance>,
    pub status: Option<NoteStatus>,
    pub tags: Option<Vec<String>>,
}

/// Request to link a note to an entity
#[derive(Debug, Deserialize)]
pub struct LinkNoteBody {
    pub entity_type: EntityType,
    pub entity_id: String,
}

/// Response for staleness update
#[derive(Debug, Serialize)]
pub struct StalenessUpdateResponse {
    pub notes_updated: usize,
}

// ============================================================================
// Handlers
// ============================================================================

/// List notes with filters
pub async fn list_notes(
    State(state): State<OrchestratorState>,
    Query(query): Query<NotesListQuery>,
) -> Result<Json<PaginatedResponse<Note>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let filters = query.to_note_filters();
    let (notes, total) = state
        .orchestrator
        .note_manager()
        .list_notes(query.project_id, &filters)
        .await?;

    Ok(Json(PaginatedResponse::new(
        notes,
        total,
        query.pagination.validated_limit(),
        query.pagination.offset,
    )))
}

/// List notes for a specific project
pub async fn list_project_notes(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<Uuid>,
    Query(query): Query<NotesListQuery>,
) -> Result<Json<PaginatedResponse<Note>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let filters = query.to_note_filters();
    let (notes, total) = state
        .orchestrator
        .note_manager()
        .list_project_notes(project_id, &filters)
        .await?;

    Ok(Json(PaginatedResponse::new(
        notes,
        total,
        query.pagination.validated_limit(),
        query.pagination.offset,
    )))
}

/// Create a new note
pub async fn create_note(
    State(state): State<OrchestratorState>,
    Json(body): Json<CreateNoteBody>,
) -> Result<(StatusCode, Json<Note>), AppError> {
    let request = CreateNoteRequest {
        project_id: body.project_id,
        note_type: body.note_type,
        content: body.content,
        importance: body.importance,
        scope: body.scope,
        tags: body.tags,
        anchors: body.anchors,
        assertion_rule: body.assertion_rule,
    };

    let note = state
        .orchestrator
        .note_manager()
        .create_note(request, "api")
        .await?;

    Ok((StatusCode::CREATED, Json(note)))
}

/// Get a note by ID
pub async fn get_note(
    State(state): State<OrchestratorState>,
    Path(note_id): Path<Uuid>,
) -> Result<Json<Note>, AppError> {
    let note = state
        .orchestrator
        .note_manager()
        .get_note(note_id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Note {} not found", note_id)))?;

    Ok(Json(note))
}

/// Update a note
pub async fn update_note(
    State(state): State<OrchestratorState>,
    Path(note_id): Path<Uuid>,
    Json(body): Json<UpdateNoteBody>,
) -> Result<Json<Note>, AppError> {
    let request = UpdateNoteRequest {
        content: body.content,
        importance: body.importance,
        status: body.status,
        tags: body.tags,
    };

    let note = state
        .orchestrator
        .note_manager()
        .update_note(note_id, request)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Note {} not found", note_id)))?;

    Ok(Json(note))
}

/// Delete a note
pub async fn delete_note(
    State(state): State<OrchestratorState>,
    Path(note_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    let deleted = state
        .orchestrator
        .note_manager()
        .delete_note(note_id)
        .await?;

    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!("Note {} not found", note_id)))
    }
}

/// Search notes
pub async fn search_notes(
    State(state): State<OrchestratorState>,
    Query(query): Query<NotesSearchQuery>,
) -> Result<Json<Vec<NoteSearchHit>>, AppError> {
    let filters = NoteFilters {
        note_type: query
            .note_type
            .as_ref()
            .and_then(|s| s.parse::<NoteType>().ok())
            .map(|t| vec![t]),
        status: query.status.as_ref().map(|s| {
            s.split(',')
                .filter_map(|s| s.trim().parse::<NoteStatus>().ok())
                .collect()
        }),
        importance: query
            .importance
            .as_ref()
            .and_then(|s| s.parse::<NoteImportance>().ok())
            .map(|i| vec![i]),
        search: query.project_slug.clone(),
        limit: query.limit.map(|l| l as i64),
        ..Default::default()
    };

    let hits = state
        .orchestrator
        .note_manager()
        .search_notes(&query.q, &filters)
        .await?;

    Ok(Json(hits))
}

/// Link a note to an entity
pub async fn link_note_to_entity(
    State(state): State<OrchestratorState>,
    Path(note_id): Path<Uuid>,
    Json(body): Json<LinkNoteBody>,
) -> Result<StatusCode, AppError> {
    let request = LinkNoteRequest {
        entity_type: body.entity_type,
        entity_id: body.entity_id,
    };

    state
        .orchestrator
        .note_manager()
        .link_note_to_entity(note_id, &request)
        .await?;

    Ok(StatusCode::OK)
}

/// Unlink a note from an entity
pub async fn unlink_note_from_entity(
    State(state): State<OrchestratorState>,
    Path((note_id, entity_type, entity_id)): Path<(Uuid, String, String)>,
) -> Result<StatusCode, AppError> {
    let entity_type = entity_type
        .parse::<EntityType>()
        .map_err(|_| AppError::BadRequest(format!("Invalid entity type: {}", entity_type)))?;

    state
        .orchestrator
        .note_manager()
        .unlink_note_from_entity(note_id, &entity_type, &entity_id)
        .await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Confirm a note is still valid
pub async fn confirm_note(
    State(state): State<OrchestratorState>,
    Path(note_id): Path<Uuid>,
) -> Result<Json<Note>, AppError> {
    let note = state
        .orchestrator
        .note_manager()
        .confirm_note(note_id, "api")
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Note {} not found", note_id)))?;

    Ok(Json(note))
}

/// Invalidate a note
#[derive(Debug, Deserialize)]
pub struct InvalidateNoteBody {
    pub reason: String,
}

pub async fn invalidate_note(
    State(state): State<OrchestratorState>,
    Path(note_id): Path<Uuid>,
    Json(body): Json<InvalidateNoteBody>,
) -> Result<Json<Note>, AppError> {
    let note = state
        .orchestrator
        .note_manager()
        .invalidate_note(note_id, &body.reason, "api")
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Note {} not found", note_id)))?;

    Ok(Json(note))
}

/// Supersede a note with a new one
pub async fn supersede_note(
    State(state): State<OrchestratorState>,
    Path(old_note_id): Path<Uuid>,
    Json(body): Json<CreateNoteBody>,
) -> Result<(StatusCode, Json<Note>), AppError> {
    let request = CreateNoteRequest {
        project_id: body.project_id,
        note_type: body.note_type,
        content: body.content,
        importance: body.importance,
        scope: body.scope,
        tags: body.tags,
        anchors: body.anchors,
        assertion_rule: body.assertion_rule,
    };

    let new_note = state
        .orchestrator
        .note_manager()
        .supersede_note(old_note_id, request, "api")
        .await?;

    Ok((StatusCode::CREATED, Json(new_note)))
}

/// Get notes needing review
pub async fn get_notes_needing_review(
    State(state): State<OrchestratorState>,
    Query(query): Query<NotesListQuery>,
) -> Result<Json<Vec<Note>>, AppError> {
    let notes = state
        .orchestrator
        .note_manager()
        .get_notes_needing_review(query.project_id)
        .await?;

    Ok(Json(notes))
}

/// Update staleness scores for all notes
pub async fn update_staleness_scores(
    State(state): State<OrchestratorState>,
) -> Result<Json<StalenessUpdateResponse>, AppError> {
    let count = state
        .orchestrator
        .note_manager()
        .update_staleness_scores()
        .await?;

    Ok(Json(StalenessUpdateResponse {
        notes_updated: count,
    }))
}

/// Get contextual notes for an entity (direct + propagated)
pub async fn get_context_notes(
    State(state): State<OrchestratorState>,
    Query(query): Query<ContextNotesQuery>,
) -> Result<Json<NoteContextResponse>, AppError> {
    let entity_type = query
        .entity_type
        .parse::<EntityType>()
        .map_err(|_| AppError::BadRequest(format!("Invalid entity type: {}", query.entity_type)))?;

    let response = state
        .orchestrator
        .note_manager()
        .get_context_notes(
            &entity_type,
            &query.entity_id,
            query.max_depth.unwrap_or(3),
            query.min_score.unwrap_or(0.1),
        )
        .await?;

    Ok(Json(response))
}

/// Get propagated notes for an entity
pub async fn get_propagated_notes(
    State(state): State<OrchestratorState>,
    Query(query): Query<ContextNotesQuery>,
) -> Result<Json<Vec<PropagatedNote>>, AppError> {
    let entity_type = query
        .entity_type
        .parse::<EntityType>()
        .map_err(|_| AppError::BadRequest(format!("Invalid entity type: {}", query.entity_type)))?;

    let notes = state
        .orchestrator
        .note_manager()
        .get_propagated_notes(
            &entity_type,
            &query.entity_id,
            query.max_depth.unwrap_or(3),
            query.min_score.unwrap_or(0.1),
        )
        .await?;

    Ok(Json(notes))
}

/// Get notes attached to a specific entity
pub async fn get_entity_notes(
    State(state): State<OrchestratorState>,
    Path((entity_type, entity_id)): Path<(String, String)>,
) -> Result<Json<Vec<Note>>, AppError> {
    let entity_type = entity_type
        .parse::<EntityType>()
        .map_err(|_| AppError::BadRequest(format!("Invalid entity type: {}", entity_type)))?;

    let notes = state
        .orchestrator
        .neo4j()
        .get_notes_for_entity(&entity_type, &entity_id)
        .await?;

    Ok(Json(notes))
}
