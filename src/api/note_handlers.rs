//! API handlers for Knowledge Notes

use super::handlers::{AppError, OrchestratorState};
use super::{PaginatedResponse, PaginationParams, SearchFilter};
use crate::events::graph::GraphEvent;
use crate::graph::algorithms::add_thermal_noise;
use crate::notes::{
    BackfillProgress, CreateAnchorRequest, CreateNoteRequest, EntityType, LinkNoteRequest, Note,
    NoteContextResponse, NoteFilters, NoteImportance, NoteScope, NoteSearchHit, NoteStatus,
    NoteType, PropagatedNote, SynapseBackfillProgress, UpdateNoteRequest,
};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use tokio::sync::RwLock;
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
    /// Filter by workspace slug (notes of projects in this workspace)
    pub workspace_slug: Option<String>,
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
    /// Comma-separated list of relation types to traverse for propagation.
    /// E.g. "CONTAINS,IMPORTS,CALLS,CO_CHANGED,IMPLEMENTS_TRAIT"
    /// If absent, defaults to CONTAINS|IMPORTS|CALLS (backward compatible).
    pub relation_types: Option<String>,
    /// Source project UUID for cross-project coupling weighting.
    /// When set, notes from other projects are weighted by P2P coupling strength.
    pub source_project_id: Option<Uuid>,
    /// Force cross-project propagation even when coupling < 0.2
    pub force_cross_project: Option<bool>,
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
        .list_notes(query.project_id, query.workspace_slug.as_deref(), &filters)
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

    // Parse comma-separated relation_types into Vec<String>
    let relation_types: Option<Vec<String>> = query.relation_types.as_ref().map(|s| {
        s.split(',')
            .map(|r| r.trim().to_string())
            .filter(|r| !r.is_empty())
            .collect()
    });

    let notes = state
        .orchestrator
        .note_manager()
        .get_propagated_notes(
            &entity_type,
            &query.entity_id,
            query.max_depth.unwrap_or(3),
            query.min_score.unwrap_or(0.1),
            relation_types.as_deref(),
            query.source_project_id,
            query.force_cross_project.unwrap_or(false),
        )
        .await?;

    Ok(Json(notes))
}

/// Get unified context knowledge for an entity (notes + decisions + commits)
pub async fn get_context_knowledge(
    State(state): State<OrchestratorState>,
    Query(query): Query<ContextNotesQuery>,
) -> Result<Json<crate::notes::ContextKnowledge>, AppError> {
    let entity_type = query
        .entity_type
        .parse::<EntityType>()
        .map_err(|_| AppError::BadRequest(format!("Invalid entity type: {}", query.entity_type)))?;

    let result = state
        .orchestrator
        .note_manager()
        .get_context_knowledge(
            &entity_type,
            &query.entity_id,
            query.max_depth.unwrap_or(3),
            query.min_score.unwrap_or(0.1),
        )
        .await?;

    Ok(Json(result))
}

/// Get enriched propagated knowledge for an entity (notes + decisions + relation stats)
pub async fn get_propagated_knowledge(
    State(state): State<OrchestratorState>,
    Query(query): Query<ContextNotesQuery>,
) -> Result<Json<crate::notes::PropagatedKnowledge>, AppError> {
    let entity_type = query
        .entity_type
        .parse::<EntityType>()
        .map_err(|_| AppError::BadRequest(format!("Invalid entity type: {}", query.entity_type)))?;

    // Parse comma-separated relation_types into Vec<String>
    let relation_types: Option<Vec<String>> = query.relation_types.as_ref().map(|s| {
        s.split(',')
            .map(|r| r.trim().to_string())
            .filter(|r| !r.is_empty())
            .collect()
    });

    let result = state
        .orchestrator
        .note_manager()
        .get_propagated_knowledge(
            &entity_type,
            &query.entity_id,
            query.max_depth.unwrap_or(3),
            query.min_score.unwrap_or(0.1),
            relation_types.as_deref(),
        )
        .await?;

    Ok(Json(result))
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

// ============================================================================
// Embedding Backfill (Admin)
// ============================================================================

/// Status of the embedding backfill job
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum BackfillJobStatus {
    /// No backfill has been run
    Idle,
    /// Backfill is currently running
    Running,
    /// Backfill completed successfully
    Completed,
    /// Backfill failed with an error
    Failed,
    /// Backfill was cancelled by user
    Cancelled,
}

/// Full state of the embedding backfill job (returned by status endpoint)
#[derive(Debug, Clone, Serialize)]
pub struct BackfillJobState {
    pub status: BackfillJobStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<BackfillProgress>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Default for BackfillJobState {
    fn default() -> Self {
        Self {
            status: BackfillJobStatus::Idle,
            progress: None,
            started_at: None,
            finished_at: None,
            error: None,
        }
    }
}

/// Request body for starting a backfill
#[derive(Debug, Deserialize)]
pub struct StartBackfillBody {
    /// Number of notes per batch (default: 50)
    pub batch_size: Option<usize>,
}

/// Singleton state for the backfill job (only one can run at a time)
static BACKFILL_STATE: LazyLock<Arc<RwLock<BackfillJobState>>> =
    LazyLock::new(|| Arc::new(RwLock::new(BackfillJobState::default())));

/// Cancellation flag for the backfill job
static BACKFILL_CANCEL: LazyLock<Arc<AtomicBool>> =
    LazyLock::new(|| Arc::new(AtomicBool::new(false)));

/// POST /api/admin/backfill-embeddings — Start embedding backfill in background
///
/// Returns 202 Accepted immediately. The backfill runs asynchronously.
/// Use GET /api/admin/backfill-embeddings/status to monitor progress.
/// Returns 409 Conflict if a backfill is already running.
pub async fn start_backfill_embeddings(
    State(state): State<OrchestratorState>,
    body: Option<Json<StartBackfillBody>>,
) -> Result<(StatusCode, Json<BackfillJobState>), AppError> {
    let batch_size = body.and_then(|b| b.batch_size).unwrap_or(50);

    // Check if already running
    {
        let current = BACKFILL_STATE.read().await;
        if current.status == BackfillJobStatus::Running {
            return Err(AppError::Conflict(
                "A backfill job is already running. Use DELETE to cancel it first.".to_string(),
            ));
        }
    }

    // Reset cancel flag
    BACKFILL_CANCEL.store(false, Ordering::SeqCst);

    // Set state to Running
    let now = chrono::Utc::now().to_rfc3339();
    {
        let mut s = BACKFILL_STATE.write().await;
        *s = BackfillJobState {
            status: BackfillJobStatus::Running,
            progress: None,
            started_at: Some(now),
            finished_at: None,
            error: None,
        };
    }

    // Clone what we need for the background task
    let note_manager = state.orchestrator.note_manager().clone();
    let cancel_flag = BACKFILL_CANCEL.clone();
    let job_state = BACKFILL_STATE.clone();

    // Spawn background task
    tokio::spawn(async move {
        let result = note_manager
            .backfill_embeddings(batch_size, Some(&cancel_flag))
            .await;

        let finished_at = chrono::Utc::now().to_rfc3339();
        let mut s = job_state.write().await;

        match result {
            Ok(progress) => {
                let was_cancelled = cancel_flag.load(Ordering::SeqCst);
                s.status = if was_cancelled {
                    BackfillJobStatus::Cancelled
                } else {
                    BackfillJobStatus::Completed
                };
                s.progress = Some(progress);
                s.finished_at = Some(finished_at);
            }
            Err(e) => {
                tracing::error!("Backfill embeddings failed: {e:#}");
                s.status = BackfillJobStatus::Failed;
                s.error = Some(e.to_string());
                s.finished_at = Some(finished_at);
            }
        }
    });

    // Return 202 Accepted with current state
    let current = BACKFILL_STATE.read().await;
    Ok((StatusCode::ACCEPTED, Json(current.clone())))
}

/// GET /api/admin/backfill-embeddings/status — Get backfill job status
pub async fn get_backfill_embeddings_status(
    State(_state): State<OrchestratorState>,
) -> Result<Json<BackfillJobState>, AppError> {
    let current = BACKFILL_STATE.read().await;
    Ok(Json(current.clone()))
}

/// DELETE /api/admin/backfill-embeddings — Cancel a running backfill job
///
/// Sets the cancellation flag. The background task will stop after
/// finishing its current batch and update the status to Cancelled.
pub async fn cancel_backfill_embeddings(
    State(_state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let is_running = {
        let current = BACKFILL_STATE.read().await;
        current.status == BackfillJobStatus::Running
    };

    if !is_running {
        return Err(AppError::BadRequest(
            "No backfill job is currently running".to_string(),
        ));
    }

    BACKFILL_CANCEL.store(true, Ordering::SeqCst);

    Ok(Json(serde_json::json!({
        "message": "Cancellation requested. The job will stop after the current batch."
    })))
}

// ============================================================================
// Neural search (spreading activation)
// ============================================================================

/// Query parameters for neural search
#[derive(Debug, Deserialize)]
pub struct NeuronSearchQuery {
    pub query: String,
    pub project_slug: Option<String>,
    pub max_results: Option<usize>,
    pub max_hops: Option<usize>,
    pub min_score: Option<f64>,
}

/// Search notes using spreading activation (neural-style retrieval).
///
/// GET /api/notes/neurons/search?query=...&project_slug=...&max_results=10&max_hops=2&min_score=0.1
pub async fn search_neurons(
    State(state): State<OrchestratorState>,
    Query(query): Query<NeuronSearchQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let engine = state.orchestrator.activation_engine().ok_or_else(|| {
        AppError::BadRequest(
            "Spreading activation unavailable: no embedding provider configured".to_string(),
        )
    })?;

    // Resolve project_slug → project_id
    let project_id = if let Some(ref slug) = query.project_slug {
        let project = state
            .orchestrator
            .neo4j()
            .get_project_by_slug(slug)
            .await
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", slug)))?;
        Some(project.id)
    } else {
        None
    };

    let config = crate::neurons::SpreadingActivationConfig {
        max_results: query.max_results.unwrap_or(10),
        max_hops: query.max_hops.unwrap_or(2),
        min_activation: query.min_score.unwrap_or(0.1),
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let results = engine
        .activate(&query.query, project_id, &config)
        .await
        .map_err(AppError::Internal)?;
    let query_time_ms = start.elapsed().as_millis() as u64;

    let direct_matches = results
        .iter()
        .filter(|r| matches!(r.source, crate::neurons::ActivationSource::Direct))
        .count();
    let propagated_matches = results.len() - direct_matches;

    let results_json: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "id": r.note.id,
                "content": r.note.content,
                "note_type": r.note.note_type,
                "importance": r.note.importance,
                "activation_score": r.activation_score,
                "source": r.source,
                "energy": r.note.energy,
                "tags": r.note.tags,
                "project_id": r.note.project_id,
            })
        })
        .collect();

    // Emit activation results on Graph WebSocket as 3 progressive phases:
    //   phase="direct"      (immediate) — direct vector matches only
    //   phase="propagating"  (50ms later) — propagated notes + active edges
    //   phase="done"         (100ms later) — completion signal
    if let Some(pid) = project_id {
        use crate::events::graph::{
            ActivationResultPayload, GraphEvent, PropagatedNote,
        };

        let direct_ids: Vec<String> = results
            .iter()
            .filter(|r| matches!(r.source, crate::neurons::ActivationSource::Direct))
            .map(|r| r.note.id.to_string())
            .collect();

        let propagated: Vec<PropagatedNote> = results
            .iter()
            .filter_map(|r| {
                if let crate::neurons::ActivationSource::Propagated { via, .. } = &r.source {
                    Some(PropagatedNote {
                        id: r.note.id.to_string(),
                        via: Some(via.to_string()),
                        score: r.activation_score,
                    })
                } else {
                    None
                }
            })
            .collect();

        let mut direct_scores = std::collections::HashMap::new();
        let mut all_scores = std::collections::HashMap::new();
        for r in &results {
            let id = r.note.id.to_string();
            let score = r.activation_score;
            all_scores.insert(id.clone(), score);
            if matches!(r.source, crate::neurons::ActivationSource::Direct) {
                direct_scores.insert(id, score);
            }
        }

        // Build active edges: synapse pairs where both endpoints are activated
        let all_ids: std::collections::HashSet<String> =
            results.iter().map(|r| r.note.id.to_string()).collect();
        let active_edges: Vec<String> = results
            .iter()
            .filter_map(|r| {
                if let crate::neurons::ActivationSource::Propagated { via, .. } = &r.source {
                    let via_str = via.to_string();
                    if all_ids.contains(&via_str) {
                        Some(format!("{}-{}", via_str, r.note.id))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        let query_str = query.query.clone();
        let pid_str = pid.to_string();

        // Phase 1 (immediate): emit direct matches only
        let direct_payload = ActivationResultPayload {
            direct_ids: direct_ids.clone(),
            propagated: vec![],
            scores: direct_scores,
            active_edges: vec![],
            query: query_str.clone(),
            phase: Some("direct".to_string()),
        };
        state
            .event_bus
            .emit_graph(GraphEvent::activation_result(direct_payload, pid_str.clone()));

        // Phases 2 & 3 are fire-and-forget via tokio::spawn with delays
        let event_bus = state.event_bus.clone();
        tokio::spawn(async move {
            // Phase 2 (50ms later): emit propagated notes + active edges
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let propagating_payload = ActivationResultPayload {
                direct_ids: vec![],
                propagated,
                scores: all_scores,
                active_edges,
                query: query_str.clone(),
                phase: Some("propagating".to_string()),
            };
            event_bus.emit_graph(GraphEvent::activation_result(
                propagating_payload,
                pid_str.clone(),
            ));

            // Phase 3 (100ms after start): emit done signal
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let done_payload = ActivationResultPayload {
                direct_ids: vec![],
                propagated: vec![],
                scores: std::collections::HashMap::new(),
                active_edges: vec![],
                query: query_str,
                phase: Some("done".to_string()),
            };
            event_bus.emit_graph(GraphEvent::activation_result(done_payload, pid_str));
        });
    }

    Ok(Json(serde_json::json!({
        "results": results_json,
        "metadata": {
            "total_activated": results.len(),
            "direct_matches": direct_matches,
            "propagated_matches": propagated_matches,
            "query_time_ms": query_time_ms,
            "max_hops": config.max_hops,
            "min_score": config.min_activation,
        }
    })))
}

// ============================================================================
// Synapse backfill (async background job)
// ============================================================================

/// Request body for starting a synapse backfill
#[derive(Debug, Deserialize)]
pub struct StartSynapseBackfillBody {
    /// Notes per batch (default: 50)
    pub batch_size: Option<usize>,
    /// Minimum cosine similarity to create a synapse (default: 0.75)
    pub min_similarity: Option<f64>,
    /// Max synapses per note (default: 10)
    pub max_neighbors: Option<usize>,
}

/// Singleton state for the synapse backfill job
static SYNAPSE_BACKFILL_STATE: LazyLock<Arc<RwLock<SynapseBackfillJobState>>> =
    LazyLock::new(|| Arc::new(RwLock::new(SynapseBackfillJobState::default())));

/// Cancellation flag for the synapse backfill job
static SYNAPSE_BACKFILL_CANCEL: LazyLock<Arc<AtomicBool>> =
    LazyLock::new(|| Arc::new(AtomicBool::new(false)));

/// State of the synapse backfill job
#[derive(Debug, Clone, Serialize)]
pub struct SynapseBackfillJobState {
    pub status: BackfillJobStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<SynapseBackfillProgress>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Default for SynapseBackfillJobState {
    fn default() -> Self {
        Self {
            status: BackfillJobStatus::Idle,
            progress: None,
            started_at: None,
            finished_at: None,
            error: None,
        }
    }
}

/// POST /api/admin/backfill-synapses — Start synapse backfill in background
///
/// Returns 202 Accepted immediately. The backfill runs asynchronously.
/// Use GET /api/admin/backfill-synapses/status to monitor progress.
/// Returns 409 Conflict if a backfill is already running.
pub async fn start_backfill_synapses(
    State(state): State<OrchestratorState>,
    body: Option<Json<StartSynapseBackfillBody>>,
) -> Result<(StatusCode, Json<SynapseBackfillJobState>), AppError> {
    let (batch_size, min_similarity, max_neighbors) = match body {
        Some(Json(b)) => (
            b.batch_size.unwrap_or(50),
            b.min_similarity.unwrap_or(0.75),
            b.max_neighbors.unwrap_or(10),
        ),
        None => (50, 0.75, 10),
    };

    // Check if already running
    {
        let current = SYNAPSE_BACKFILL_STATE.read().await;
        if current.status == BackfillJobStatus::Running {
            return Err(AppError::Conflict(
                "A synapse backfill job is already running. Use DELETE to cancel it first."
                    .to_string(),
            ));
        }
    }

    // Reset cancel flag
    SYNAPSE_BACKFILL_CANCEL.store(false, Ordering::SeqCst);

    // Set state to Running
    let now = chrono::Utc::now().to_rfc3339();
    {
        let mut s = SYNAPSE_BACKFILL_STATE.write().await;
        *s = SynapseBackfillJobState {
            status: BackfillJobStatus::Running,
            progress: None,
            started_at: Some(now),
            finished_at: None,
            error: None,
        };
    }

    // Clone what we need for the background task
    let note_manager = state.orchestrator.note_manager().clone();
    let cancel_flag = SYNAPSE_BACKFILL_CANCEL.clone();
    let job_state = SYNAPSE_BACKFILL_STATE.clone();

    tokio::spawn(async move {
        let result = note_manager
            .backfill_synapses(
                batch_size,
                min_similarity,
                max_neighbors,
                Some(&cancel_flag),
            )
            .await;

        let finished_at = chrono::Utc::now().to_rfc3339();
        let mut s = job_state.write().await;

        match result {
            Ok(progress) => {
                let was_cancelled = cancel_flag.load(Ordering::SeqCst);
                s.status = if was_cancelled {
                    BackfillJobStatus::Cancelled
                } else {
                    BackfillJobStatus::Completed
                };
                s.progress = Some(progress);
                s.finished_at = Some(finished_at);
            }
            Err(e) => {
                tracing::error!("Backfill synapses failed: {e:#}");
                s.status = BackfillJobStatus::Failed;
                s.error = Some(e.to_string());
                s.finished_at = Some(finished_at);
            }
        }
    });

    // Return 202 with current state
    let current = SYNAPSE_BACKFILL_STATE.read().await;
    Ok((StatusCode::ACCEPTED, Json(current.clone())))
}

/// GET /api/admin/backfill-synapses/status — Get synapse backfill job status
pub async fn get_backfill_synapses_status(
    State(_state): State<OrchestratorState>,
) -> Result<Json<SynapseBackfillJobState>, AppError> {
    let current = SYNAPSE_BACKFILL_STATE.read().await;
    Ok(Json(current.clone()))
}

/// DELETE /api/admin/backfill-synapses — Cancel a running synapse backfill
pub async fn cancel_backfill_synapses(
    State(_state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let is_running = {
        let current = SYNAPSE_BACKFILL_STATE.read().await;
        current.status == BackfillJobStatus::Running
    };

    if !is_running {
        return Err(AppError::BadRequest(
            "No synapse backfill job is currently running".to_string(),
        ));
    }

    SYNAPSE_BACKFILL_CANCEL.store(true, Ordering::SeqCst);

    Ok(Json(serde_json::json!({
        "message": "Cancellation requested. The job will stop after the current note."
    })))
}

// ============================================================================
// Additional endpoints for MCP HTTP proxy parity
// ============================================================================

/// Query params for semantic note search
#[derive(Debug, Deserialize)]
pub struct SemanticSearchQuery {
    pub query: String,
    pub project_slug: Option<String>,
    pub workspace_slug: Option<String>,
    pub limit: Option<usize>,
    /// Minimum cosine similarity threshold (0.0 - 1.0).
    /// Results below this score are filtered out. Default: none (return all top-K).
    pub min_similarity: Option<f64>,
    /// Thermal noise temperature (0.0 - 1.0) for stochastic exploration.
    /// Inspired by Langevin dynamics: adds T × N(0, σ) Gaussian noise to scores.
    /// 0.0 = deterministic (default), 1.0 = maximum exploration.
    pub temperature: Option<f64>,
}

/// GET /api/notes/search-semantic — Vector-based semantic search
pub async fn search_notes_semantic(
    State(state): State<OrchestratorState>,
    Query(query): Query<SemanticSearchQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project_id = if let Some(ref slug) = query.project_slug {
        let project = state
            .orchestrator
            .neo4j()
            .get_project_by_slug(slug)
            .await
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", slug)))?;
        Some(project.id)
    } else {
        None
    };

    let mut hits = state
        .orchestrator
        .note_manager()
        .semantic_search_notes(
            &query.query,
            project_id,
            query.workspace_slug.as_deref(),
            query.limit,
            query.min_similarity,
        )
        .await
        .map_err(AppError::Internal)?;

    // Apply Langevin thermal noise for stochastic exploration
    if let Some(temperature) = query.temperature {
        if temperature > 0.0 {
            let mut scored: Vec<(NoteSearchHit, f64)> = hits
                .into_iter()
                .map(|h| {
                    let s = h.score;
                    (h, s)
                })
                .collect();
            add_thermal_noise(&mut scored, temperature);
            hits = scored
                .into_iter()
                .map(|(mut h, s)| {
                    h.score = s;
                    h
                })
                .collect();
        }
    }

    Ok(Json(serde_json::to_value(hits).unwrap_or_default()))
}

/// Request body for update_energy_scores
#[derive(Debug, Deserialize)]
pub struct UpdateEnergyBody {
    pub half_life: Option<f64>,
}

/// POST /api/notes/update-energy — Decay energy scores based on half-life
pub async fn update_energy_scores(
    State(state): State<OrchestratorState>,
    Json(body): Json<UpdateEnergyBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let half_life = body.half_life.unwrap_or(90.0);
    let count = state
        .orchestrator
        .note_manager()
        .update_energy_scores(half_life)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "notes_updated": count,
        "half_life_days": half_life
    })))
}

/// Request body for reinforce_neurons
#[derive(Debug, Deserialize)]
pub struct ReinforceNeuronsBody {
    pub note_ids: Vec<Uuid>,
    pub energy_boost: Option<f64>,
    pub synapse_boost: Option<f64>,
}

/// POST /api/notes/neurons/reinforce — Boost energy + reinforce synapses
pub async fn reinforce_neurons(
    State(state): State<OrchestratorState>,
    Json(body): Json<ReinforceNeuronsBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    if body.note_ids.len() < 2 {
        return Err(AppError::BadRequest(format!(
            "note_ids must contain at least 2 UUIDs (got {})",
            body.note_ids.len()
        )));
    }

    let energy_boost = body.energy_boost.unwrap_or(0.2);
    let synapse_boost = body.synapse_boost.unwrap_or(0.05);

    let neo4j = state.orchestrator.neo4j();
    let mut neurons_boosted = 0u64;
    for note_id in &body.note_ids {
        neo4j
            .boost_energy(*note_id, energy_boost)
            .await
            .map_err(AppError::Internal)?;
        neurons_boosted += 1;
    }

    let synapses_reinforced = neo4j
        .reinforce_synapses(&body.note_ids, synapse_boost)
        .await
        .map_err(AppError::Internal)?;

    // Emit graph events for reinforcement (fire-and-forget, best-effort per note)
    for note_id in &body.note_ids {
        // Resolve project_id for graph event scoping
        if let Ok(Some(note)) = neo4j.get_note(*note_id).await {
            if let Some(pid) = note.project_id {
                state.event_bus.emit_graph(GraphEvent::reinforcement(
                    note_id.to_string(),
                    energy_boost,
                    pid.to_string(),
                ));
            }
        }
    }

    Ok(Json(serde_json::json!({
        "neurons_boosted": neurons_boosted,
        "synapses_reinforced": synapses_reinforced,
        "energy_boost": energy_boost,
        "synapse_boost": synapse_boost,
    })))
}

/// Request body for decay_synapses
#[derive(Debug, Deserialize)]
pub struct DecaySynapsesBody {
    pub decay_amount: Option<f64>,
    pub prune_threshold: Option<f64>,
}

/// POST /api/notes/neurons/decay — Decay synapses and prune weak ones
pub async fn decay_synapses(
    State(state): State<OrchestratorState>,
    Json(body): Json<DecaySynapsesBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let decay_amount = body.decay_amount.unwrap_or(0.01);
    let prune_threshold = body.prune_threshold.unwrap_or(0.1);

    let (decayed, pruned) = state
        .orchestrator
        .neo4j()
        .decay_synapses(decay_amount, prune_threshold)
        .await
        .map_err(AppError::Internal)?;

    // Emit graph events for pruned synapses (batch signal for frontend)
    if pruned > 0 {
        // We don't have individual synapse IDs at this level, but the frontend
        // can use this event to trigger a full refresh of the neural layer.
        state.event_bus.emit_graph(
            GraphEvent::node(
                crate::events::graph::GraphEventType::CommunityChanged,
                crate::events::graph::GraphLayer::Neural,
                "decay_synapses",
                "global",
            )
            .with_delta(serde_json::json!({
                "synapses_pruned": pruned,
                "synapses_decayed": decayed,
                "decay_amount": decay_amount,
                "prune_threshold": prune_threshold,
            })),
        );
    }

    Ok(Json(serde_json::json!({
        "synapses_decayed": decayed,
        "synapses_pruned": pruned,
        "decay_amount": decay_amount,
        "prune_threshold": prune_threshold,
    })))
}

/// Request body for `POST /api/notes/neurons/heal-scars`.
#[derive(Debug, Deserialize)]
pub struct HealScarsBody {
    /// The UUID of the note or decision to heal.
    pub node_id: Uuid,
}

/// POST /api/notes/neurons/heal-scars — Reset scar_intensity to 0.0
///
/// Biomimicry: manual scar removal for notes/decisions that were incorrectly
/// penalized by negative reasoning feedback.
pub async fn heal_scars(
    State(state): State<OrchestratorState>,
    Json(body): Json<HealScarsBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let healed = state
        .orchestrator
        .neo4j()
        .heal_scars(body.node_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "healed": healed,
        "node_id": body.node_id.to_string(),
    })))
}

/// POST /api/notes/consolidate-memory — Batch memory consolidation
///
/// Biomimicry: Elun SleepSystem consolidation. Evaluates all non-consolidated
/// active notes for promotion (Ephemeral→Operational→Consolidated) and archives
/// stale ephemeral notes (>48h without reactivation).
pub async fn consolidate_memory(
    State(state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let (promoted, archived) = state
        .orchestrator
        .neo4j()
        .consolidate_memory()
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "promoted": promoted,
        "archived": archived,
        "total_processed": promoted + archived,
    })))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::events::EventBus;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_bearer_token};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;

    /// Build a test router with mock backends
    async fn test_app() -> axum::Router {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
        });
        create_router(state)
    }

    /// Create an authenticated GET request
    fn auth_get(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    /// Create an authenticated POST request with JSON body
    fn auth_post(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap()
    }

    /// Parse response body as JSON
    async fn body_json(resp: axum::http::Response<Body>) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    // ====================================================================
    // GET /api/notes — list notes (empty)
    // ====================================================================

    #[tokio::test]
    async fn test_list_notes_empty() {
        let app = test_app().await;
        let resp = app.oneshot(auth_get("/api/notes")).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["items"].as_array().unwrap().len(), 0);
        assert_eq!(json["total"], 0);
    }

    // ====================================================================
    // POST /api/notes + GET /api/notes/{id} — create and get
    // ====================================================================

    #[tokio::test]
    async fn test_create_and_get_note() {
        let app = test_app().await;

        // Create a note
        let create_body = serde_json::json!({
            "note_type": "guideline",
            "content": "Always use parameterized queries to prevent SQL injection.",
            "importance": "high",
            "tags": ["security", "sql"]
        });
        let resp = app
            .clone()
            .oneshot(auth_post("/api/notes", create_body))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::CREATED);
        let created = body_json(resp).await;
        let note_id = created["id"].as_str().unwrap();
        assert_eq!(
            created["content"],
            "Always use parameterized queries to prevent SQL injection."
        );
        assert_eq!(created["note_type"], "guideline");
        assert_eq!(created["importance"], "high");

        // Retrieve the note by ID
        let get_uri = format!("/api/notes/{}", note_id);
        let resp = app.oneshot(auth_get(&get_uri)).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let fetched = body_json(resp).await;
        assert_eq!(fetched["id"], note_id);
        assert_eq!(
            fetched["content"],
            "Always use parameterized queries to prevent SQL injection."
        );
    }

    // ====================================================================
    // GET /api/notes/{id} — not found
    // ====================================================================

    #[tokio::test]
    async fn test_get_note_not_found() {
        let app = test_app().await;
        let fake_id = uuid::Uuid::new_v4();
        let uri = format!("/api/notes/{}", fake_id);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ====================================================================
    // GET /api/entities/{entity_type}/{entity_id}/notes — invalid type
    // ====================================================================

    #[tokio::test]
    async fn test_get_entity_notes_invalid_type() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/entities/invalid_type/some-id/notes"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ====================================================================
    // GET /api/entities/{entity_type}/{entity_id}/notes — valid type
    // ====================================================================

    #[tokio::test]
    async fn test_get_entity_notes_valid_type() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/entities/file/src%2Fmain.rs/notes"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    // ====================================================================
    // GET /api/notes/context — missing required params
    // ====================================================================

    #[tokio::test]
    async fn test_get_context_notes_missing_params() {
        let app = test_app().await;
        // entity_type and entity_id are required query params
        let resp = app.oneshot(auth_get("/api/notes/context")).await.unwrap();

        // Missing required query parameters should result in 400
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ====================================================================
    // GET /api/notes/context — invalid entity type
    // ====================================================================

    #[tokio::test]
    async fn test_get_context_notes_invalid_entity_type() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/notes/context?entity_type=bogus&entity_id=abc",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ====================================================================
    // GET /api/notes/context — valid params (empty result)
    // ====================================================================

    #[tokio::test]
    async fn test_get_context_notes_valid() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/notes/context?entity_type=file&entity_id=src/main.rs",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        // NoteContextResponse has direct_notes and propagated_notes
        assert!(json["direct_notes"].is_array());
        assert!(json["propagated_notes"].is_array());
    }

    // ====================================================================
    // GET /api/notes/propagated — missing params
    // ====================================================================

    #[tokio::test]
    async fn test_get_propagated_notes_missing_params() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/notes/propagated"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ====================================================================
    // GET /api/notes/search — empty results
    // ====================================================================

    #[tokio::test]
    async fn test_search_notes_empty() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/notes/search?q=nonexistent"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    // ====================================================================
    // GET /api/notes/search — missing q param
    // ====================================================================

    #[tokio::test]
    async fn test_search_notes_missing_query() {
        let app = test_app().await;
        let resp = app.oneshot(auth_get("/api/notes/search")).await.unwrap();

        // 'q' is a required field in NotesSearchQuery
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ====================================================================
    // GET /api/notes/search-semantic — empty results
    // ====================================================================

    #[tokio::test]
    async fn test_search_notes_semantic_empty() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/notes/search-semantic?query=something+unusual",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        // semantic_search_notes falls back to BM25 when no embedding provider;
        // either way, result should be an array
        assert!(json.is_array());
    }

    // ====================================================================
    // POST /api/notes/{id}/links — link a note to an entity
    // ====================================================================

    #[tokio::test]
    async fn test_link_note_to_entity() {
        let app = test_app().await;

        // First create a note
        let create_body = serde_json::json!({
            "note_type": "gotcha",
            "content": "Watch out for race conditions in this module."
        });
        let resp = app
            .clone()
            .oneshot(auth_post("/api/notes", create_body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let created = body_json(resp).await;
        let note_id = created["id"].as_str().unwrap();

        // Link the note to a file entity
        let link_body = serde_json::json!({
            "entity_type": "file",
            "entity_id": "src/lib.rs"
        });
        let link_uri = format!("/api/notes/{}/links", note_id);
        let resp = app
            .clone()
            .oneshot(auth_post(&link_uri, link_body))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        // Verify the note appears when querying entity notes
        let resp = app
            .oneshot(auth_get("/api/entities/file/src%2Flib.rs/notes"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let notes = json.as_array().unwrap();
        assert!(
            notes.iter().any(|n| n["id"].as_str() == Some(note_id)),
            "Linked note should appear in entity notes"
        );
    }

    // ================================================================
    // Knowledge Fabric — Handler Tests
    // ================================================================

    /// Create an authenticated DELETE request
    fn auth_delete(uri: &str) -> Request<Body> {
        Request::builder()
            .method("DELETE")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    // ----------------------------------------------------------------
    // GET /api/notes/needs-review — notes needing review (empty)
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_notes_needing_review_empty() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/notes/needs-review"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_get_notes_needing_review_with_project_filter() {
        let app = test_app().await;
        let pid = uuid::Uuid::new_v4();
        let uri = format!("/api/notes/needs-review?project_id={}", pid);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
    }

    // ----------------------------------------------------------------
    // POST /api/notes/update-staleness — update staleness scores
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_update_staleness_scores() {
        let app = test_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/notes/update-staleness")
                    .header("authorization", test_bearer_token())
                    .header("content-type", "application/json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["notes_updated"].is_number());
    }

    // ----------------------------------------------------------------
    // GET /api/notes/context-knowledge — unified context knowledge
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_context_knowledge_valid() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/notes/context-knowledge?entity_type=file&entity_id=src/main.rs",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_context_knowledge_missing_params() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/notes/context-knowledge"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_get_context_knowledge_invalid_entity_type() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/notes/context-knowledge?entity_type=invalid&entity_id=abc",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_get_context_knowledge_with_optional_params() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/notes/context-knowledge?entity_type=file&entity_id=src/main.rs&max_depth=5&min_score=0.5",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ----------------------------------------------------------------
    // GET /api/notes/propagated-knowledge — enriched propagated knowledge
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_propagated_knowledge_valid() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/notes/propagated-knowledge?entity_type=file&entity_id=src/main.rs",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_propagated_knowledge_missing_params() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/notes/propagated-knowledge"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_get_propagated_knowledge_invalid_entity_type() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/notes/propagated-knowledge?entity_type=bogus&entity_id=abc",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_get_propagated_knowledge_with_relation_types() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/notes/propagated-knowledge?entity_type=file&entity_id=src/main.rs&relation_types=IMPORTS,CO_CHANGED",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ----------------------------------------------------------------
    // POST /api/notes/update-energy — Hebbian energy decay
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_update_energy_scores() {
        let app = test_app().await;
        let body = serde_json::json!({"half_life": 90.0});
        let resp = app
            .oneshot(auth_post("/api/notes/update-energy", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["notes_updated"].is_number());
        assert_eq!(json["half_life_days"], 90.0);
    }

    #[tokio::test]
    async fn test_update_energy_scores_default_half_life() {
        let app = test_app().await;
        let body = serde_json::json!({});
        let resp = app
            .oneshot(auth_post("/api/notes/update-energy", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["half_life_days"], 90.0);
    }

    // ----------------------------------------------------------------
    // POST /api/notes/neurons/reinforce — reinforce synapses
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_reinforce_neurons_valid() {
        let app = test_app().await;
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        let body = serde_json::json!({
            "note_ids": [id1, id2],
            "energy_boost": 0.3,
            "synapse_boost": 0.1
        });
        let resp = app
            .oneshot(auth_post("/api/notes/neurons/reinforce", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["neurons_boosted"], 2);
        assert!(json["synapses_reinforced"].is_number());
        assert_eq!(json["energy_boost"], 0.3);
        assert_eq!(json["synapse_boost"], 0.1);
    }

    #[tokio::test]
    async fn test_reinforce_neurons_defaults() {
        let app = test_app().await;
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        let body = serde_json::json!({
            "note_ids": [id1, id2]
        });
        let resp = app
            .oneshot(auth_post("/api/notes/neurons/reinforce", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["energy_boost"], 0.2);
        assert_eq!(json["synapse_boost"], 0.05);
    }

    #[tokio::test]
    async fn test_reinforce_neurons_too_few_ids() {
        let app = test_app().await;
        let id1 = uuid::Uuid::new_v4();
        let body = serde_json::json!({
            "note_ids": [id1]
        });
        let resp = app
            .oneshot(auth_post("/api/notes/neurons/reinforce", body))
            .await
            .unwrap();
        // Should return 400 because at least 2 note_ids required
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_reinforce_neurons_empty_ids() {
        let app = test_app().await;
        let body = serde_json::json!({
            "note_ids": []
        });
        let resp = app
            .oneshot(auth_post("/api/notes/neurons/reinforce", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ----------------------------------------------------------------
    // POST /api/notes/neurons/decay — decay synapses
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_decay_synapses_default_params() {
        let app = test_app().await;
        let body = serde_json::json!({});
        let resp = app
            .oneshot(auth_post("/api/notes/neurons/decay", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["synapses_decayed"].is_number());
        assert!(json["synapses_pruned"].is_number());
        assert_eq!(json["decay_amount"], 0.01);
        assert_eq!(json["prune_threshold"], 0.1);
    }

    #[tokio::test]
    async fn test_decay_synapses_custom_params() {
        let app = test_app().await;
        let body = serde_json::json!({
            "decay_amount": 0.05,
            "prune_threshold": 0.2
        });
        let resp = app
            .oneshot(auth_post("/api/notes/neurons/decay", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["decay_amount"], 0.05);
        assert_eq!(json["prune_threshold"], 0.2);
    }

    // ----------------------------------------------------------------
    // GET /api/notes/neurons/search — spreading activation search
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_search_neurons_with_query() {
        let app = test_app().await;
        // Local fastembed provider is initialized by default, so the
        // activation engine is present. With mock GraphStore the results
        // are empty but the request should succeed.
        let resp = app
            .oneshot(auth_get("/api/notes/neurons/search?query=authentication"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["results"].is_array());
        assert!(json["metadata"]["total_activated"].is_number());
    }

    #[tokio::test]
    async fn test_search_neurons_missing_query() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/notes/neurons/search"))
            .await
            .unwrap();
        // query is required
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ----------------------------------------------------------------
    // GET /api/admin/backfill-embeddings/status — embedding backfill status
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_backfill_embeddings_status() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/admin/backfill-embeddings/status"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        // Default state is "idle"
        assert_eq!(json["status"], "idle");
    }

    // ----------------------------------------------------------------
    // DELETE /api/admin/backfill-embeddings — cancel when not running
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_cancel_backfill_embeddings_when_idle() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_delete("/api/admin/backfill-embeddings"))
            .await
            .unwrap();
        // Should fail because no backfill is running
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ----------------------------------------------------------------
    // GET /api/admin/backfill-synapses/status — synapse backfill status
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_backfill_synapses_status() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/admin/backfill-synapses/status"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["status"], "idle");
    }

    // ----------------------------------------------------------------
    // DELETE /api/admin/backfill-synapses — cancel when not running
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_cancel_backfill_synapses_when_idle() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_delete("/api/admin/backfill-synapses"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ----------------------------------------------------------------
    // GET /api/notes/propagated — propagated notes with relation_types
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_propagated_notes_with_relation_types() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/notes/propagated?entity_type=file&entity_id=src/main.rs&relation_types=IMPORTS,CO_CHANGED",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
    }

    #[tokio::test]
    async fn test_get_propagated_notes_invalid_entity_type() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/notes/propagated?entity_type=invalid&entity_id=abc",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
