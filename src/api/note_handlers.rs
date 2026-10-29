//! API handlers for Knowledge Notes

use super::handlers::{AppError, OrchestratorState};
use super::{PaginatedResponse, PaginationParams, SearchFilter};
use crate::events::graph::GraphEvent;
use crate::graph::algorithms::add_thermal_noise;

use crate::neo4j::traits::*;
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
    /// Project scope of the propagation (project + global notes only). Cannot
    /// widen the scope: it must match the entity's owning project when that is
    /// resolvable, otherwise it is used as a single-project narrowing filter.
    pub source_project_id: Option<Uuid>,
    /// Deprecated and ignored: cross-project propagation is not available from the API.
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
    /// Optional protocol run ID for PRODUCED_DURING relation
    pub run_id: Option<Uuid>,
    /// Optional sharing consent persisted at creation (default `not_set`)
    #[serde(default)]
    pub sharing_consent: Option<crate::episodes::distill_models::SharingConsent>,
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

/// The author of a write made through the API: the agent of the calling chat session
/// when there is one (`crate::notes::provenance`), `fallback` otherwise — a person, a
/// script, or a caller that cannot be placed. Never refuses the write.
pub(crate) async fn caller_author(
    state: &OrchestratorState,
    claims: Option<&crate::auth::jwt::Claims>,
    headers: &axum::http::HeaderMap,
    fallback: &str,
) -> String {
    use crate::chat::envelope::{identify_caller, session_header, SpawnCaller};
    let caller = identify_caller(claims, session_header(headers), state.auth_config.is_some());
    let session = match caller {
        Ok(SpawnCaller::Agent { session_id, .. }) => Some(session_id),
        Ok(SpawnCaller::Human) | Err(_) => None,
    };
    crate::notes::provenance::author_of(state.orchestrator.neo4j(), session.as_deref(), fallback)
        .await
}

// ============================================================================
// Pinned notes: injected in every turn, whatever the message (knowledge_injection)
// ============================================================================

/// `?project=<slug>`: the project's pins; absent: the global pins (every session).
#[derive(Debug, Deserialize)]
pub struct PinnedNotesQuery {
    pub project: Option<String>,
}

/// `PUT /api/knowledge/pinned-notes` body: the whole list, in order.
#[derive(Debug, Deserialize)]
pub struct PinnedNotesBody {
    pub note_ids: Vec<Uuid>,
}

/// The pins of a scope.
#[derive(Debug, Serialize)]
pub struct PinnedNotesView {
    /// `global` or `project:<slug>`.
    pub scope: String,
    pub note_ids: Vec<Uuid>,
}

/// `GET /api/knowledge/pinned-notes[?project=<slug>]` — the pinned notes of a scope.
pub async fn get_pinned_notes(
    State(state): State<OrchestratorState>,
    Query(query): Query<PinnedNotesQuery>,
) -> Result<Json<PinnedNotesView>, AppError> {
    let project = query.project.as_deref();
    let note_ids = crate::chat::stages::knowledge_injection::load_pinned_notes(
        state.orchestrator.neo4j(),
        project,
    )
    .await
    .map_err(AppError::Internal)?;
    Ok(Json(PinnedNotesView {
        scope: crate::chat::stages::knowledge_injection::pinned_notes_scope(project),
        note_ids,
    }))
}

/// `PUT /api/knowledge/pinned-notes[?project=<slug>]` — replaces the pins of a scope
/// (an empty list removes them).
///
/// A pin reaches EVERY turn of every session of its scope: whoever pins writes into the
/// prompts of all of them. So a person only — never an agent, whose note pinned by
/// itself would be a standing injection (refused by the middleware too, see
/// `is_human_only_mutation`) — and the GLOBAL pins, which reach other people's chats,
/// the root account only. A project pin must be a note of that project or a global
/// note; a global pin must be a global note. At most `MAX_PINNED_NOTES`, each existing.
pub async fn put_pinned_notes(
    State(state): State<OrchestratorState>,
    axum::Extension(claims): axum::Extension<crate::auth::jwt::Claims>,
    Query(query): Query<PinnedNotesQuery>,
    Json(body): Json<PinnedNotesBody>,
) -> Result<Json<PinnedNotesView>, AppError> {
    use crate::chat::stages::knowledge_injection::{
        pinned_notes_scope, store_pinned_notes, MAX_PINNED_NOTES,
    };
    let graph = state.orchestrator.neo4j();
    let project = query.project.as_deref();
    let project_id = match project {
        Some(slug) => {
            // A person with their own session token: not an agent, an MCP token, a vault
            // token or the server's account (`may_keep_lasting_approvals`, the same rule).
            if state.auth_config.is_some() && !claims.may_keep_lasting_approvals() {
                return Err(AppError::Forbidden(
                    "pinned_notes_human_only: only a signed-in person pins notes".into(),
                ));
            }
            let p = graph
                .get_project_by_slug(slug)
                .await
                .map_err(AppError::Internal)?
                .ok_or_else(|| AppError::NotFound(format!("Project {slug} not found")))?;
            Some(p.id)
        }
        None => {
            super::provider_handlers::require_root(&state, &claims)?;
            None
        }
    };

    let mut seen = std::collections::HashSet::new();
    let ids: Vec<Uuid> = body
        .note_ids
        .into_iter()
        .filter(|id| seen.insert(*id))
        .collect();
    if ids.len() > MAX_PINNED_NOTES {
        return Err(AppError::BadRequest(format!(
            "pinned_notes_too_many: at most {MAX_PINNED_NOTES} notes"
        )));
    }
    for id in &ids {
        let note = graph
            .get_note(*id)
            .await
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::BadRequest(format!("pinned_note_unknown: {id}")))?;
        let fits = match (note.project_id, project_id) {
            (None, _) => true,
            (Some(owner), Some(scope)) => owner == scope,
            (Some(_), None) => false,
        };
        if !fits {
            return Err(AppError::BadRequest(format!(
                "pinned_note_out_of_scope: {id} belongs to another project{}",
                if project.is_none() {
                    " (a global pin must be a global note)"
                } else {
                    ""
                }
            )));
        }
    }
    store_pinned_notes(graph, project, &ids)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(PinnedNotesView {
        scope: pinned_notes_scope(project),
        note_ids: ids,
    }))
}

/// Create a new note
pub async fn create_note(
    State(state): State<OrchestratorState>,
    claims: Option<axum::Extension<crate::auth::jwt::Claims>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<CreateNoteBody>,
) -> Result<(StatusCode, Json<Note>), AppError> {
    let explicit_run_id = body.run_id;
    let project_id = body.project_id;
    let request = CreateNoteRequest {
        project_id,
        note_type: body.note_type,
        content: body.content,
        importance: body.importance,
        scope: body.scope,
        tags: body.tags,
        anchors: body.anchors,
        assertion_rule: body.assertion_rule,
        run_id: None, // run_id is handled at the handler level, not passed down
        sharing_consent: body.sharing_consent,
    };

    let note = state
        .orchestrator
        .note_manager()
        .create_note(
            request,
            &caller_author(&state, claims.as_ref().map(|c| &c.0), &headers, "api").await,
        )
        .await?;

    // Resolve run_id: explicit > auto-detect from active run
    let run_id = match explicit_run_id {
        Some(rid) => Some(rid),
        None => {
            // Auto-detect: find active run for the same project
            if let Some(pid) = project_id {
                match state
                    .orchestrator
                    .neo4j()
                    .find_active_run_for_project(pid)
                    .await
                {
                    Ok(active_run) => active_run,
                    Err(e) => {
                        tracing::debug!(
                            project_id = %pid,
                            error = %e,
                            "Failed to auto-detect active run for PRODUCED_DURING"
                        );
                        None
                    }
                }
            } else {
                None
            }
        }
    };

    // Create PRODUCED_DURING relation if we have a run_id
    if let Some(rid) = run_id {
        if let Err(e) = state
            .orchestrator
            .neo4j()
            .create_produced_during("Note", note.id, rid)
            .await
        {
            tracing::warn!(
                note_id = %note.id,
                run_id = %rid,
                error = %e,
                "Failed to create PRODUCED_DURING relation for note"
            );
        }
    }

    // Temporal correlation: if a ReasoningTree was recently built for this project,
    // persist it with a REASONING_FOR → Note link (the tree "caused" this note).
    if let Some(pid) = project_id {
        if let Some(engine) = state.orchestrator.reasoning_engine() {
            let max_age = std::time::Duration::from_secs(60);
            if let Some(tree) = engine.cache().get_recent_for_project(pid, max_age).await {
                match state
                    .orchestrator
                    .neo4j()
                    .persist_reasoning_tree(&tree, Some("Note"), Some(note.id))
                    .await
                {
                    Ok(persisted_id) => {
                        tracing::info!(
                            tree_id = %persisted_id,
                            note_id = %note.id,
                            "Persisted reasoning tree via temporal correlation (note created within 60s)"
                        );
                    }
                    Err(e) => {
                        tracing::debug!(
                            error = %e,
                            "Failed to persist reasoning tree via temporal correlation"
                        );
                    }
                }
            }
        }
    }

    // Growth hook: auto-link note to RELEVANT personas only (not all active ones).
    // Relevance = persona KNOWS files that overlap with files mentioned in note content.
    if let Some(pid) = project_id {
        let neo4j = state.orchestrator.neo4j_arc();
        let note_id = note.id;
        let note_content = note.content.clone();
        tokio::spawn(async move {
            let file_paths =
                crate::skills::activation::extract_file_paths_from_content(&note_content);
            match neo4j
                .find_relevant_personas_for_note(&file_paths, pid)
                .await
            {
                Ok(relevant) if !relevant.is_empty() => {
                    for (persona_id, avg_weight) in &relevant {
                        let link_weight = (*avg_weight * 0.8).max(0.3);
                        if let Err(e) = neo4j
                            .auto_link_note_to_persona(*persona_id, note_id, link_weight)
                            .await
                        {
                            tracing::debug!(
                                persona_id = %persona_id,
                                note_id = %note_id,
                                error = %e,
                                "Growth hook: auto_link_note failed (non-fatal)"
                            );
                        }
                    }
                    tracing::debug!(
                        note_id = %note_id,
                        count = relevant.len(),
                        "Growth hook: linked note to relevant personas"
                    );
                }
                Ok(_) => {
                    tracing::debug!(
                        note_id = %note_id,
                        file_paths = ?file_paths,
                        "Growth hook: no relevant personas found for note (skipped)"
                    );
                }
                Err(e) => {
                    tracing::debug!(
                        project_id = %pid,
                        error = %e,
                        "Growth hook: find_relevant_personas_for_note failed (non-fatal)"
                    );
                }
            }
        });
    }

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
    // For File entities, normalize absolute paths to relative using the project root.
    // This ensures note anchors are always project-agnostic for portability.
    let entity_id = if body.entity_type == EntityType::File && body.entity_id.starts_with('/') {
        // Try to load the note's project_id, then the project root_path
        if let Ok(Some(note)) = state.orchestrator.note_manager().get_note(note_id).await {
            if let Some(project_id) = note.project_id {
                if let Ok(Some(project)) = state.orchestrator.neo4j().get_project(project_id).await
                {
                    crate::utils::paths::relativize(&body.entity_id, &project.root_path)
                } else {
                    body.entity_id
                }
            } else {
                body.entity_id
            }
        } else {
            body.entity_id
        }
    } else {
        body.entity_id
    };

    let request = LinkNoteRequest {
        entity_type: body.entity_type,
        entity_id,
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
    claims: Option<axum::Extension<crate::auth::jwt::Claims>>,
    headers: axum::http::HeaderMap,
    Path(note_id): Path<Uuid>,
) -> Result<Json<Note>, AppError> {
    let note = state
        .orchestrator
        .note_manager()
        .confirm_note(
            note_id,
            &caller_author(&state, claims.as_ref().map(|c| &c.0), &headers, "api").await,
        )
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Note {} not found", note_id)))?;

    if confirmed_by_a_person(state.auth_config.is_some(), claims.as_ref().map(|c| &c.0)) {
        crate::neurons::synaptic_shadow::spawn_human_confirmation_reward(
            state.orchestrator.neo4j_arc(),
            note_id,
        );
    }

    Ok(Json(note))
}

/// Whether a confirmation is the « confirmed by a human » signal of the synaptic
/// layer (T8): a signed-in person — never an agent session, an MCP token, the
/// server's account or the anonymous user. Without authentication there is no
/// person to tell apart from an agent, so there is no such signal.
pub(crate) fn confirmed_by_a_person(
    auth_enabled: bool,
    claims: Option<&crate::auth::jwt::Claims>,
) -> bool {
    auth_enabled && claims.is_some_and(|c| c.is_human())
}

/// Invalidate a note
#[derive(Debug, Deserialize)]
pub struct InvalidateNoteBody {
    pub reason: String,
}

pub async fn invalidate_note(
    State(state): State<OrchestratorState>,
    Path(note_id): Path<Uuid>,
    claims: Option<axum::Extension<crate::auth::jwt::Claims>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<InvalidateNoteBody>,
) -> Result<Json<Note>, AppError> {
    let note = state
        .orchestrator
        .note_manager()
        .invalidate_note(
            note_id,
            &body.reason,
            &caller_author(&state, claims.as_ref().map(|c| &c.0), &headers, "api").await,
        )
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Note {} not found", note_id)))?;

    Ok(Json(note))
}

/// Supersede a note with a new one
pub async fn supersede_note(
    State(state): State<OrchestratorState>,
    Path(old_note_id): Path<Uuid>,
    claims: Option<axum::Extension<crate::auth::jwt::Claims>>,
    headers: axum::http::HeaderMap,
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
        run_id: None,
        sharing_consent: body.sharing_consent,
    };

    let new_note = state
        .orchestrator
        .note_manager()
        .supersede_note(
            old_note_id,
            request,
            &caller_author(&state, claims.as_ref().map(|c| &c.0), &headers, "api").await,
        )
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
        .get_notes_needing_review(
            query.project_id,
            super::handlers::normalize_slug(query.workspace_slug.as_deref()),
        )
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

/// Project scope of a propagation request. The HTTP API (and the MCP tools that
/// proxy it) has no caller identity, so `source_project_id` can only narrow:
/// the entity's owning project wins and a mismatching value is a 400;
/// `force_cross_project` is ignored (logged) because it would widen the scope.
async fn resolve_scope(
    state: &OrchestratorState,
    entity_type: &EntityType,
    query: &ContextNotesQuery,
) -> Result<Option<Uuid>, AppError> {
    if query.force_cross_project == Some(true) {
        tracing::warn!(
            entity_id = %query.entity_id,
            "force_cross_project ignored on the HTTP API: cross-project propagation is internal-only"
        );
    }
    state
        .orchestrator
        .note_manager()
        .resolve_propagation_scope(entity_type, &query.entity_id, query.source_project_id)
        .await
        .map_err(AppError::BadRequest)
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

    let scope_project = resolve_scope(&state, &entity_type, &query).await?;

    let notes = state
        .orchestrator
        .note_manager()
        .get_propagated_notes(
            &entity_type,
            &query.entity_id,
            query.max_depth.unwrap_or(3),
            query.min_score.unwrap_or(0.1),
            relation_types.as_deref(),
            scope_project,
            false, // never widened from the HTTP API (see resolve_scope)
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

    let scope_project = resolve_scope(&state, &entity_type, &query).await?;

    let result = state
        .orchestrator
        .note_manager()
        .get_propagated_knowledge(
            &entity_type,
            &query.entity_id,
            query.max_depth.unwrap_or(3),
            query.min_score.unwrap_or(0.1),
            relation_types.as_deref(),
            scope_project,
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
        use crate::events::graph::{ActivationResultPayload, GraphEvent, PropagatedNote};

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
        state.event_bus.emit_graph(GraphEvent::activation_result(
            direct_payload,
            pid_str.clone(),
        ));

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
                /* project_id */ None,
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
    pub project_id: Option<Uuid>,
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
    /// Intent-adaptive analysis profile name (e.g., "debug", "explore", "impact", "plan").
    /// When provided, weights the search results according to the profile's configuration.
    pub profile: Option<String>,
}

/// GET /api/notes/search-semantic — Vector-based semantic search
pub async fn search_notes_semantic(
    State(state): State<OrchestratorState>,
    Query(query): Query<SemanticSearchQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    // An explicit project_id wins over the slug shortcut. Without either, the
    // search stays global on purpose: the breadth across projects is a feature,
    // not an accident to fence off.
    let project_id = if let Some(pid) = query.project_id {
        Some(pid)
    } else if let Some(ref slug) = query.project_slug {
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
            query.profile.as_deref(),
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

/// All notes must exist and belong to one and the same known project: no
/// synapse (nor energy boost) across projects or with an unknown `project_id`.
/// `entries` holds `(note_id, project_id)` for notes that were found.
pub(crate) fn ensure_single_project(entries: &[(Uuid, Option<Uuid>)]) -> Result<Uuid, AppError> {
    let mut project: Option<Uuid> = None;
    for (id, pid) in entries {
        let pid = pid.ok_or_else(|| {
            AppError::BadRequest(format!("note {id} has no project_id: cannot be reinforced"))
        })?;
        match project {
            None => project = Some(pid),
            Some(p) if p == pid => {}
            Some(_) => {
                return Err(AppError::BadRequest(
                    "note_ids span several projects: reinforcement is per project".into(),
                ))
            }
        }
    }
    project.ok_or_else(|| AppError::BadRequest("note_ids is empty".into()))
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

    // Client-supplied ids: every note must exist and share one known project.
    let mut entries = Vec::with_capacity(body.note_ids.len());
    for note_id in &body.note_ids {
        let note = neo4j
            .get_note(*note_id)
            .await
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::NotFound(format!("Note {note_id} not found")))?;
        entries.push((*note_id, note.project_id));
    }
    ensure_single_project(&entries)?;

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
// Runner Feedback Analysis
// ============================================================================

/// Query parameters for analyze_runner_feedback
#[derive(Debug, Deserialize, Default)]
pub struct AnalyzeRunnerFeedbackQuery {
    pub project_id: Option<Uuid>,
}

/// POST /api/admin/analyze-runner-feedback — Analyze runner feedback patterns
///
/// Aggregates runner-feedback observation notes, detects recurring patterns,
/// and returns a structured report with frequencies and examples.
/// Part of the feedback→protocol loop.
pub async fn analyze_runner_feedback(
    State(state): State<OrchestratorState>,
    Query(query): Query<AnalyzeRunnerFeedbackQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let analyzer = crate::runner::FeedbackAnalyzer::new(state.orchestrator.neo4j_arc());
    let report = analyzer
        .analyze_runner_feedback(query.project_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::to_value(&report).unwrap_or_default()))
}

/// POST /api/admin/generate-feedback-rfcs — Generate RFC proposals from actionable patterns
///
/// Runs feedback analysis, then for each actionable pattern generates an RFC note
/// with the proposed protocol adjustment. Returns the RFC note IDs.
pub async fn generate_feedback_rfcs(
    State(state): State<OrchestratorState>,
    Json(body): Json<GenerateFeedbackRfcsBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let analyzer = crate::runner::FeedbackAnalyzer::new(state.orchestrator.neo4j_arc());
    let report = analyzer
        .analyze_runner_feedback(Some(body.project_id))
        .await
        .map_err(AppError::Internal)?;

    let rfc_ids = analyzer
        .generate_rfc_proposals(body.project_id, &report)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "actionable_patterns": report.actionable_count,
        "rfcs_created": rfc_ids.len(),
        "rfc_ids": rfc_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
    })))
}

#[derive(Debug, Deserialize)]
pub struct GenerateFeedbackRfcsBody {
    pub project_id: Uuid,
}

/// POST /api/admin/apply-feedback-rfc — Apply an accepted feedback RFC
///
/// Extracts the protocol spec from the accepted RFC, computes trust score,
/// and returns the AppliedProtocol ready for composition.
pub async fn apply_feedback_rfc(
    State(state): State<OrchestratorState>,
    Json(body): Json<ApplyFeedbackRfcBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let analyzer = crate::runner::FeedbackAnalyzer::new(state.orchestrator.neo4j_arc());
    let result = analyzer
        .apply_accepted_rfc(body.rfc_note_id, body.project_id)
        .await
        .map_err(AppError::Internal)?;

    match result {
        Some(applied) => Ok(Json(serde_json::to_value(&applied).unwrap_or_default())),
        None => Err(AppError::NotFound(
            "RFC note not found or not a valid runner-feedback RFC".to_string(),
        )),
    }
}

#[derive(Debug, Deserialize)]
pub struct ApplyFeedbackRfcBody {
    pub rfc_note_id: Uuid,
    pub project_id: Uuid,
}

// ============================================================================
// Tests
// ============================================================================

/// Body of `POST /api/notes/{note_id}/scope`.
#[derive(Debug, Deserialize)]
pub struct RescopeNoteBody {
    /// `null` (or omitted) makes the note global; a uuid gives it that project.
    pub project_id: Option<Uuid>,
}

/// Move a note between the global level and a project.
///
/// Writes both representations of a note's project — the property every read
/// filters on, and the `HAS_NOTE` edge `project.delete` follows. Doing only one
/// of them produces a note that looks attached but that the old project reclaims
/// on deletion, silently.
pub async fn rescope_note(
    State(state): State<OrchestratorState>,
    Path(note_id): Path<Uuid>,
    Json(body): Json<RescopeNoteBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    state
        .orchestrator
        .neo4j()
        .rescope_note(note_id, body.project_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "note_id": note_id,
        "project_id": body.project_id,
    })))
}

#[cfg(test)]
mod tests {
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::events::EventBus;
    use crate::neo4j::traits::*;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_bearer_token};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;

    /// Build a test router with mock backends
    async fn test_app() -> axum::Router {
        test_app_with_state(mock_app_state()).await
    }

    /// Seed `n` notes in the given project (None = unknown project) and return the app and the ids.
    async fn test_app_with_notes(
        projects: &[Option<uuid::Uuid>],
    ) -> (axum::Router, Vec<uuid::Uuid>) {
        use crate::neo4j::traits::GraphStore;
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let mut ids = Vec::new();
        for p in projects {
            let n = crate::notes::models::Note::new(
                *p,
                crate::notes::models::NoteType::Guideline,
                "n".into(),
                "t".into(),
            );
            graph.create_note(&n).await.unwrap();
            ids.push(n.id);
        }
        let state = crate::test_helpers::mock_app_state_with_graph(graph);
        (test_app_with_state(state).await, ids)
    }

    async fn test_app_with_state(app_state: crate::AppState) -> axum::Router {
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
            remote_mcp: crate::RemoteMcpConfig::default(),
            ingress: Default::default(),
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
            model_catalog: crate::chat::model_catalog::ModelCatalogCache::new(None),
            vault: crate::vault::VaultService::ephemeral(),
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
    /// A note's project lives in two places: the property every read filters on,
    /// and the `HAS_NOTE` edge `project.delete` follows. This endpoint moves
    /// both, so a note taken from a project is not reclaimed by it later.
    #[tokio::test]
    async fn rescoping_a_note_moves_it_between_the_global_level_and_a_project() {
        let project_id = uuid::Uuid::new_v4();
        let (app, notes) = test_app_with_notes(&[None]).await;
        let note_id = notes[0];

        let resp = app
            .clone()
            .oneshot(auth_post(
                &format!("/api/notes/{}/scope", note_id),
                serde_json::json!({ "project_id": project_id }),
            ))
            .await
            .unwrap();
        assert!(
            resp.status().is_success(),
            "attach failed: {}",
            resp.status()
        );
        assert_eq!(
            body_json(resp).await["project_id"],
            serde_json::json!(project_id)
        );

        let resp = app
            .oneshot(auth_post(
                &format!("/api/notes/{}/scope", note_id),
                serde_json::json!({ "project_id": null }),
            ))
            .await
            .unwrap();
        assert!(
            resp.status().is_success(),
            "detach failed: {}",
            resp.status()
        );
        assert!(
            body_json(resp).await["project_id"].is_null(),
            "null must make the note global again"
        );
    }

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

    /// A bound, live agent token for `session`.
    fn agent_bearer(session: uuid::Uuid) -> String {
        let claims = crate::auth::jwt::Claims::service_account("agent");
        let binding = crate::auth::jwt::AgentSessionBinding {
            session_id: session.to_string(),
            ceiling: None,
            tool_profile: None,
            third_party: false,
        };
        let (token, jti) = crate::auth::jwt::generate_session_token(
            &claims,
            Some(&binding),
            &crate::test_helpers::test_auth_config().jwt_secret,
            3600,
        )
        .unwrap();
        crate::auth::agent_tokens::register(&jti, Some(&session.to_string()));
        format!("Bearer {token}")
    }

    /// Provenance: a note a chat session writes names the agent behind it — the same
    /// agent for two sessions of one model, another for another model — and a note a
    /// person writes keeps `api`, as every note did before. Superseding stamps it too.
    #[tokio::test]
    async fn a_note_written_by_a_chat_session_names_its_agent() {
        use crate::neo4j::traits::GraphStore;
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let mut sessions = Vec::new();
        for model in ["claude-opus-5-5", "claude-opus-5-5", "deepseek-v4"] {
            let mut s = crate::test_helpers::test_chat_session(None);
            s.model = model.into();
            graph.create_chat_session(&s).await.unwrap();
            sessions.push(s.id);
        }
        let app = test_app_with_state(crate::test_helpers::mock_app_state_with_graph(
            graph.clone(),
        ))
        .await;
        let write = |uri: String, bearer: String| {
            let app = app.clone();
            async move {
                let resp = app
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri(uri)
                            .header("authorization", bearer)
                            .header("content-type", "application/json")
                            .body(Body::from(
                                serde_json::json!({"note_type": "tip", "content": "a tip"})
                                    .to_string(),
                            ))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(resp.status(), StatusCode::CREATED);
                body_json(resp).await
            }
        };

        let first = write("/api/notes".into(), agent_bearer(sessions[0])).await;
        let second = write("/api/notes".into(), agent_bearer(sessions[1])).await;
        let other = write("/api/notes".into(), agent_bearer(sessions[2])).await;
        let person = write("/api/notes".into(), test_bearer_token()).await;

        let expected = crate::notes::provenance::actor(
            crate::notes::provenance::agent_id_of_session(graph.as_ref(), sessions[0])
                .await
                .unwrap()
                .unwrap(),
        );
        assert_eq!(first["created_by"], expected.as_str());
        assert_eq!(
            second["created_by"], first["created_by"],
            "one model, one agent"
        );
        assert_ne!(
            other["created_by"], first["created_by"],
            "another model, another agent"
        );
        assert!(other["created_by"].as_str().unwrap().starts_with("agent:"));
        assert_eq!(
            person["created_by"], "api",
            "a person's note keeps its author"
        );

        let old = first["id"].as_str().unwrap();
        let replaced = write(
            format!("/api/notes/{old}/supersede"),
            agent_bearer(sessions[2]),
        )
        .await;
        assert_eq!(replaced["created_by"], other["created_by"]);
    }

    /// Pinning is a person's act: a project's pins are set by a signed-in person, checked
    /// note by note; an agent is refused (it would inject itself into every turn); the
    /// global pins, which reach everyone's chats, need the root account.
    #[tokio::test]
    async fn only_a_person_pins_notes_and_only_notes_of_the_scope() {
        use crate::neo4j::traits::GraphStore;
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let alpha = crate::test_helpers::test_project_named("alpha");
        graph.create_project(&alpha).await.unwrap();
        let beta = crate::test_helpers::test_project_named("beta");
        graph.create_project(&beta).await.unwrap();
        let mut ids = std::collections::HashMap::new();
        for (name, project) in [
            ("global", None),
            ("alpha", Some(alpha.id)),
            ("beta", Some(beta.id)),
        ] {
            let n = crate::notes::models::Note::new(
                project,
                crate::notes::models::NoteType::Guideline,
                name.into(),
                "t".into(),
            );
            graph.create_note(&n).await.unwrap();
            ids.insert(name, n.id);
        }
        let mut session = crate::test_helpers::test_chat_session(Some("alpha"));
        session.model = "claude-opus-5-5".into();
        graph.create_chat_session(&session).await.unwrap();
        let app = test_app_with_state(crate::test_helpers::mock_app_state_with_graph(
            graph.clone(),
        ))
        .await;
        let call =
            |method: &'static str, uri: &'static str, bearer: String, body: serde_json::Value| {
                let app = app.clone();
                async move {
                    let resp = app
                        .oneshot(
                            Request::builder()
                                .method(method)
                                .uri(uri)
                                .header("authorization", bearer)
                                .header("content-type", "application/json")
                                .body(Body::from(body.to_string()))
                                .unwrap(),
                        )
                        .await
                        .unwrap();
                    let status = resp.status();
                    (status, body_json(resp).await)
                }
            };
        let uri = "/api/knowledge/pinned-notes?project=alpha";

        let (status, view) = call(
            "PUT",
            uri,
            test_bearer_token(),
            serde_json::json!({"note_ids": [ids["alpha"], ids["global"], ids["alpha"]]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{view}");
        assert_eq!(view["scope"], "project:alpha");
        let expected = serde_json::json!([ids["alpha"], ids["global"]]);
        assert_eq!(view["note_ids"], expected, "deduplicated, in order");
        let (status, read) = call("GET", uri, test_bearer_token(), serde_json::json!(null)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(read["note_ids"], expected);

        // An agent cannot pin, not even in its own project.
        let (status, _) = call(
            "PUT",
            uri,
            agent_bearer(session.id),
            serde_json::json!({"note_ids": [ids["alpha"]]}),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);

        // A note of another project, an unknown note, too many notes: refused, nothing changed.
        for body in [
            serde_json::json!({"note_ids": [ids["beta"]]}),
            serde_json::json!({"note_ids": [uuid::Uuid::new_v4()]}),
            serde_json::json!({"note_ids": (0..11).map(|_| uuid::Uuid::new_v4()).collect::<Vec<_>>()}),
        ] {
            let (status, err) = call("PUT", uri, test_bearer_token(), body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{err}");
        }
        let (_, read) = call("GET", uri, test_bearer_token(), serde_json::json!(null)).await;
        assert_eq!(read["note_ids"], expected);

        // The global pins reach everyone's chats: a person who is not root is refused.
        let (status, _) = call(
            "PUT",
            "/api/knowledge/pinned-notes",
            test_bearer_token(),
            serde_json::json!({"note_ids": [ids["global"]]}),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);
    }

    /// The other writes an agent makes through the API name it too: an RFC, a decision,
    /// a confirmation and an invalidation. A person keeps the author each had before.
    #[tokio::test]
    async fn rfcs_decisions_confirmations_and_invalidations_name_the_agent() {
        use crate::neo4j::traits::GraphStore;
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let mut session = crate::test_helpers::test_chat_session(None);
        session.model = "claude-opus-5-5".into();
        graph.create_chat_session(&session).await.unwrap();
        let plan = crate::test_helpers::test_plan();
        graph.create_plan(&plan).await.unwrap();
        let task = crate::test_helpers::test_task();
        graph.create_task(plan.id, &task).await.unwrap();
        let agent = crate::notes::provenance::actor(
            crate::notes::provenance::agent_id_of_session(graph.as_ref(), session.id)
                .await
                .unwrap()
                .unwrap(),
        );
        let app = test_app_with_state(crate::test_helpers::mock_app_state_with_graph(
            graph.clone(),
        ))
        .await;
        let send = |uri: String, bearer: String, body: serde_json::Value| {
            let app = app.clone();
            async move {
                let resp = app
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri(uri)
                            .header("authorization", bearer)
                            .header("content-type", "application/json")
                            .body(Body::from(body.to_string()))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                let status = resp.status();
                (status, body_json(resp).await)
            }
        };
        let rfc_body = serde_json::json!({
            "title": "An RFC",
            "sections": [{"title": "Problem", "content": "p"}]
        });
        let decision_body = serde_json::json!({"description": "d", "rationale": "r"});

        let (status, rfc) = send(
            "/api/rfcs".into(),
            agent_bearer(session.id),
            rfc_body.clone(),
        )
        .await;
        assert!(status.is_success(), "{rfc}");
        assert_eq!(rfc["created_by"], agent.as_str(), "{rfc}");
        let (_, rfc) = send("/api/rfcs".into(), test_bearer_token(), rfc_body).await;
        assert_eq!(rfc["created_by"], "api");

        let uri = format!("/api/tasks/{}/decisions", task.id);
        let (status, decision) =
            send(uri.clone(), agent_bearer(session.id), decision_body.clone()).await;
        assert!(status.is_success(), "{decision}");
        assert_eq!(decision["decided_by"], agent.as_str(), "{decision}");
        let (_, decision) = send(uri, test_bearer_token(), decision_body).await;
        assert_eq!(
            decision["decided_by"], "agent",
            "a person's decision as before"
        );

        let (_, note) = send(
            "/api/notes".into(),
            test_bearer_token(),
            serde_json::json!({"note_type": "tip", "content": "t"}),
        )
        .await;
        let id = note["id"].as_str().unwrap().to_string();
        let (status, confirmed) = send(
            format!("/api/notes/{id}/confirm"),
            agent_bearer(session.id),
            serde_json::json!({}),
        )
        .await;
        assert!(status.is_success(), "{confirmed}");
        assert_eq!(
            confirmed["last_confirmed_by"],
            agent.as_str(),
            "{confirmed}"
        );
        let (status, invalidated) = send(
            format!("/api/notes/{id}/invalidate"),
            agent_bearer(session.id),
            serde_json::json!({"reason": "stale"}),
        )
        .await;
        assert!(status.is_success(), "{invalidated}");
        let changes = invalidated["changes"].to_string();
        assert!(
            changes.contains(&agent),
            "the invalidation is the agent's: {changes}"
        );
    }

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

    #[tokio::test]
    async fn test_get_propagated_notes_rejects_foreign_source_project() {
        let app = test_app().await;
        let (owner, other) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        let uri = format!(
            "/api/notes/propagated?entity_type=project&entity_id={owner}&source_project_id={other}"
        );
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
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
        let p = uuid::Uuid::new_v4();
        let (app, ids) = test_app_with_notes(&[Some(p), Some(p)]).await;
        let body = serde_json::json!({
            "note_ids": ids,
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
        assert_eq!(json["synapses_reinforced"], 2);
        assert_eq!(json["energy_boost"], 0.3);
        assert_eq!(json["synapse_boost"], 0.1);
    }

    #[tokio::test]
    async fn test_reinforce_neurons_defaults() {
        let p = uuid::Uuid::new_v4();
        let (app, ids) = test_app_with_notes(&[Some(p), Some(p)]).await;
        let body = serde_json::json!({
            "note_ids": ids
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
    async fn test_reinforce_neurons_cross_project_rejected() {
        let (app, ids) =
            test_app_with_notes(&[Some(uuid::Uuid::new_v4()), Some(uuid::Uuid::new_v4())]).await;
        let resp = app
            .oneshot(auth_post(
                "/api/notes/neurons/reinforce",
                serde_json::json!({ "note_ids": ids }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_reinforce_neurons_unknown_project_rejected() {
        let (app, ids) = test_app_with_notes(&[Some(uuid::Uuid::new_v4()), None]).await;
        let resp = app
            .oneshot(auth_post(
                "/api/notes/neurons/reinforce",
                serde_json::json!({ "note_ids": ids }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_reinforce_neurons_unknown_note_is_404() {
        let (app, mut ids) = test_app_with_notes(&[Some(uuid::Uuid::new_v4())]).await;
        ids.push(uuid::Uuid::new_v4());
        let resp = app
            .oneshot(auth_post(
                "/api/notes/neurons/reinforce",
                serde_json::json!({ "note_ids": ids }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
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
        // fastembed may be unavailable under the instrumented coverage job
        // (model init times out) → endpoint 5xx. Assert the happy path when
        // embeddings work; tolerate the env failure otherwise.
        if resp.status() == StatusCode::OK {
            let json = body_json(resp).await;
            assert!(json["results"].is_array());
            assert!(json["metadata"]["total_activated"].is_number());
        } else {
            assert!(
                resp.status().is_client_error() || resp.status().is_server_error(),
                "unexpected status: {}",
                resp.status()
            );
        }
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

#[cfg(test)]
mod human_confirmation_tests {
    use super::confirmed_by_a_person;
    use crate::auth::jwt::{Claims, TOKEN_TYPE_AGENT_SESSION, TOKEN_TYPE_MCP};
    use crate::neo4j::traits::*;

    fn person() -> Claims {
        Claims {
            sub: uuid::Uuid::new_v4().to_string(),
            email: "someone@example.org".into(),
            name: "Someone".into(),
            iat: 0,
            exp: i64::MAX,
            token_type: None,
            scope: None,
            jti: None,
        }
    }

    #[test]
    fn only_a_signed_in_person_gives_the_human_confirmation_signal() {
        assert!(confirmed_by_a_person(true, Some(&person())));
        // Without authentication nobody can be told apart from an agent.
        assert!(!confirmed_by_a_person(false, Some(&person())));
        assert!(!confirmed_by_a_person(true, None));
        assert!(!confirmed_by_a_person(true, Some(&Claims::anonymous())));
        for token_type in [TOKEN_TYPE_AGENT_SESSION, TOKEN_TYPE_MCP] {
            let agent = Claims {
                token_type: Some(token_type.to_string()),
                ..person()
            };
            assert!(!confirmed_by_a_person(true, Some(&agent)), "{token_type}");
        }
    }
}
