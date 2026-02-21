//! API handlers for Knowledge Notes

use super::handlers::{AppError, OrchestratorState};
use super::{PaginatedResponse, PaginationParams, SearchFilter};
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
