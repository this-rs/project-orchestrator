//! REST API handlers for RFCs (Request for Comments)
//!
//! RFCs are Notes with `note_type = "rfc"`. These handlers provide a
//! frontend-friendly API that wraps the underlying Note CRUD and Protocol
//! FSM transition machinery.
//!
//! The content field of an RFC note stores JSON: `{ "title": "...", "sections": [...] }`.
//! The RFC status is stored as a tag `rfc-status:<status>` on the note.
//! The protocol run link is stored as a tag `rfc-run:<uuid>` on the note.

use super::handlers::{AppError, OrchestratorState};
use super::PaginatedResponse;
use crate::neo4j::GraphStore;
use crate::notes::{
    CreateNoteRequest, Note, NoteFilters, NoteImportance, NoteType, UpdateNoteRequest,
};
use crate::protocol::{
    self, Protocol, ProtocolCategory, ProtocolState, ProtocolTransition, StateType,
};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use tracing::info;
use uuid::Uuid;

// ============================================================================
// Types
// ============================================================================

/// A single section within an RFC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RfcSection {
    pub title: String,
    pub content: String,
}

/// A trigger available from the current FSM state.
#[derive(Debug, Serialize)]
pub struct AvailableTransition {
    pub trigger: String,
    pub target_state: String,
    pub guard: Option<String>,
}

/// RFC response matching the frontend `Rfc` interface.
#[derive(Debug, Serialize)]
pub struct RfcResponse {
    pub id: String,
    pub title: String,
    pub status: String,
    pub importance: String,
    pub sections: Vec<RfcSection>,
    pub protocol_run_id: Option<String>,
    pub current_state: Option<String>,
    /// Available FSM transitions from the current state (populated on GET detail).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_transitions: Option<Vec<AvailableTransition>>,
    pub created_at: String,
    pub updated_at: Option<String>,
    pub created_by: Option<String>,
    pub tags: Vec<String>,
}

/// Stored JSON content inside the Note's `content` field.
#[derive(Debug, Serialize, Deserialize)]
struct RfcContent {
    title: String,
    sections: Vec<RfcSection>,
}

// ============================================================================
// Query / Request types
// ============================================================================

/// Query parameters for listing RFCs.
#[derive(Debug, Deserialize, Default)]
pub struct RfcListQuery {
    pub status: Option<String>,
    pub importance: Option<String>,
    pub project_id: Option<Uuid>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Request body for creating an RFC.
#[derive(Debug, Deserialize)]
pub struct CreateRfcBody {
    pub title: String,
    pub sections: Vec<RfcSection>,
    pub importance: Option<String>,
    pub tags: Option<Vec<String>>,
    pub project_id: Option<Uuid>,
}

/// Request body for updating an RFC.
#[derive(Debug, Deserialize)]
pub struct UpdateRfcBody {
    pub title: Option<String>,
    pub sections: Option<Vec<RfcSection>>,
    pub importance: Option<String>,
    pub tags: Option<Vec<String>>,
}

/// Request body for RFC lifecycle transition.
#[derive(Debug, Deserialize)]
pub struct TransitionRfcBody {
    pub action: String, // "propose" | "accept" | "reject" | "implement"
}

// ============================================================================
// Helpers
// ============================================================================

/// Extract the `rfc-status:<status>` tag value from a note's tags.
fn extract_rfc_status(tags: &[String]) -> String {
    tags.iter()
        .find(|t| t.starts_with("rfc-status:"))
        .map(|t| t.trim_start_matches("rfc-status:").to_string())
        .unwrap_or_else(|| "draft".to_string())
}

/// Extract the `rfc-run:<uuid>` tag value from a note's tags.
fn extract_run_id(tags: &[String]) -> Option<String> {
    tags.iter()
        .find(|t| t.starts_with("rfc-run:"))
        .map(|t| t.trim_start_matches("rfc-run:").to_string())
}

/// Try to extract a title from markdown content.
///
/// Looks for patterns like:
///   - `# Title`
///   - `## RFC — Title`
///   - `## RFC: Title`
///   - `## Title`
fn extract_title_from_markdown(content: &str) -> Option<String> {
    for line in content.lines() {
        let trimmed = line.trim();
        // Match lines starting with # or ##
        if let Some(rest) = trimmed
            .strip_prefix("## ")
            .or_else(|| trimmed.strip_prefix("# "))
        {
            // Strip optional "RFC —" or "RFC:" prefix
            let title = rest
                .strip_prefix("RFC")
                .and_then(|s| {
                    s.strip_prefix(" — ")
                        .or_else(|| s.strip_prefix(" - "))
                        .or_else(|| s.strip_prefix(": "))
                })
                .unwrap_or(rest)
                .trim();
            if !title.is_empty() {
                return Some(title.to_string());
            }
        }
    }
    None
}

/// Convert a Note into an RfcResponse.
fn note_to_rfc(note: &Note) -> RfcResponse {
    let (title, sections) = match serde_json::from_str::<RfcContent>(&note.content) {
        Ok(c) => (c.title, c.sections),
        Err(_) => {
            // Try extracting title from markdown headers before falling back
            let title = extract_title_from_markdown(&note.content)
                .unwrap_or_else(|| "Untitled RFC".to_string());
            (
                title,
                vec![RfcSection {
                    title: "Content".to_string(),
                    content: note.content.clone(),
                }],
            )
        }
    };

    let status = extract_rfc_status(&note.tags);
    let run_id = extract_run_id(&note.tags);

    // Filter out internal rfc-* tags from the user-visible tags list
    let user_tags: Vec<String> = note
        .tags
        .iter()
        .filter(|t| !t.starts_with("rfc-status:") && !t.starts_with("rfc-run:"))
        .cloned()
        .collect();

    RfcResponse {
        id: note.id.to_string(),
        title,
        status,
        importance: format!("{}", note.importance),
        sections,
        protocol_run_id: run_id,
        current_state: None,         // enriched later if needed
        available_transitions: None, // enriched later if needed
        created_at: note.created_at.to_rfc3339(),
        updated_at: None,
        created_by: Some(note.created_by.clone()),
        tags: user_tags,
    }
}

/// Build the JSON content string for a Note from title + sections.
fn rfc_content_json(title: &str, sections: &[RfcSection]) -> String {
    let content = RfcContent {
        title: title.to_string(),
        sections: sections.to_vec(),
    };
    serde_json::to_string(&content).unwrap_or_default()
}

/// Ensure the tag list contains a specific `rfc-status:<status>` tag,
/// replacing any existing one.
fn set_status_tag(tags: &mut Vec<String>, status: &str) {
    tags.retain(|t| !t.starts_with("rfc-status:"));
    tags.push(format!("rfc-status:{}", status));
}

// ============================================================================
// Bootstrap — auto-create rfc-lifecycle protocol if missing
// ============================================================================

/// Find the `rfc-lifecycle` protocol for a project, or auto-create it.
///
/// This makes the RFC system self-bootstrapping: a fresh project without any
/// protocol will get the full 9-state FSM created on first RFC transition.
///
/// States: draft → proposed → under_review → accepted → planning →
///         in_progress → implemented | rejected | superseded
///
/// 16 transitions covering the full lifecycle.
async fn find_or_create_rfc_lifecycle(
    neo4j: &dyn GraphStore,
    project_id: Uuid,
) -> Result<Protocol, AppError> {
    // 1. Try to find existing protocol
    let (protocols, _) = neo4j
        .list_protocols(project_id, None, 200, 0)
        .await
        .map_err(AppError::Internal)?;

    if let Some(existing) = protocols.iter().find(|p| p.name == "rfc-lifecycle") {
        return Ok(existing.clone());
    }

    // 2. Not found — bootstrap the full protocol
    info!(
        "Bootstrapping rfc-lifecycle protocol for project {}",
        project_id
    );

    // -- Create states --
    let protocol_id = Uuid::new_v4();

    let s_draft = ProtocolState {
        state_type: StateType::Start,
        ..ProtocolState::new(protocol_id, "draft")
    };
    let s_proposed = ProtocolState::new(protocol_id, "proposed");
    let s_under_review = ProtocolState::new(protocol_id, "under_review");
    let s_accepted = ProtocolState::new(protocol_id, "accepted");
    let s_planning = ProtocolState::new(protocol_id, "planning");
    let s_in_progress = ProtocolState::new(protocol_id, "in_progress");
    let s_implemented = ProtocolState::terminal(protocol_id, "implemented");
    let s_rejected = ProtocolState::terminal(protocol_id, "rejected");
    let s_superseded = ProtocolState::terminal(protocol_id, "superseded");

    let all_states = [
        &s_draft,
        &s_proposed,
        &s_under_review,
        &s_accepted,
        &s_planning,
        &s_in_progress,
        &s_implemented,
        &s_rejected,
        &s_superseded,
    ];

    // -- Create protocol node --
    let protocol = Protocol::new_full(
        project_id,
        "rfc-lifecycle",
        "RFC lifecycle FSM — auto-bootstrapped. 9 states, 16 transitions.",
        s_draft.id,
        vec![s_implemented.id, s_rejected.id, s_superseded.id],
        ProtocolCategory::Business,
    );
    // Override the id to match our pre-generated one
    let protocol = Protocol {
        id: protocol_id,
        ..protocol
    };

    neo4j
        .upsert_protocol(&protocol)
        .await
        .map_err(AppError::Internal)?;

    // -- Persist states --
    for s in &all_states {
        neo4j
            .upsert_protocol_state(s)
            .await
            .map_err(AppError::Internal)?;
    }

    // -- Create 16 transitions --
    // Helper to build & persist a transition
    let transitions_def: Vec<(Uuid, Uuid, &str)> = vec![
        // draft →
        (s_draft.id, s_proposed.id, "propose"),
        (s_draft.id, s_superseded.id, "supersede"),
        // proposed →
        (s_proposed.id, s_under_review.id, "submit_review"),
        (s_proposed.id, s_rejected.id, "reject"),
        (s_proposed.id, s_superseded.id, "supersede"),
        // under_review →
        (s_under_review.id, s_accepted.id, "accept"),
        (s_under_review.id, s_draft.id, "revise"),
        (s_under_review.id, s_rejected.id, "reject"),
        (s_under_review.id, s_superseded.id, "supersede"),
        // accepted →
        (s_accepted.id, s_planning.id, "start_planning"),
        (s_accepted.id, s_superseded.id, "supersede"),
        // planning →
        (s_planning.id, s_in_progress.id, "start_work"),
        (s_planning.id, s_superseded.id, "supersede"),
        // in_progress →
        (s_in_progress.id, s_implemented.id, "complete"),
        (s_in_progress.id, s_planning.id, "replan"),
        (s_in_progress.id, s_superseded.id, "supersede"),
    ];

    for (from, to, trigger) in &transitions_def {
        let t = ProtocolTransition::new(protocol_id, *from, *to, *trigger);
        neo4j
            .upsert_protocol_transition(&t)
            .await
            .map_err(AppError::Internal)?;
    }

    info!(
        "rfc-lifecycle protocol bootstrapped: {} with {} states, {} transitions",
        protocol_id,
        all_states.len(),
        transitions_def.len()
    );

    Ok(protocol)
}

// ============================================================================
// Handlers
// ============================================================================

/// List RFCs with optional filters.
///
/// GET /api/rfcs?status=&importance=&limit=&offset=
pub async fn list_rfcs(
    State(state): State<OrchestratorState>,
    Query(query): Query<RfcListQuery>,
) -> Result<Json<PaginatedResponse<RfcResponse>>, AppError> {
    let limit = query.limit.unwrap_or(50).min(200);
    let offset = query.offset.unwrap_or(0);

    // Build filters: always filter by note_type = rfc
    let mut tags_filter: Vec<String> = Vec::new();

    // If frontend filters by status, add the rfc-status tag filter
    if let Some(ref status) = query.status {
        if !status.is_empty() {
            tags_filter.push(format!("rfc-status:{}", status));
        }
    }

    let filters = NoteFilters {
        note_type: Some(vec![NoteType::Rfc]),
        importance: query
            .importance
            .as_ref()
            .and_then(|s| s.parse::<NoteImportance>().ok())
            .map(|i| vec![i]),
        tags: if tags_filter.is_empty() {
            None
        } else {
            Some(tags_filter)
        },
        limit: Some(limit as i64),
        offset: Some(offset as i64),
        ..Default::default()
    };

    let (notes, total) = state
        .orchestrator
        .note_manager()
        .list_notes(query.project_id, None, &filters)
        .await?;

    let mut items: Vec<RfcResponse> = notes.iter().map(note_to_rfc).collect();

    // Enrich current_state for each RFC that has a linked protocol run.
    // This lets the frontend show correct fallback transitions based on
    // the real FSM state instead of the simplified rfc-status.
    for rfc in items.iter_mut() {
        if let Some(ref run_id_str) = rfc.protocol_run_id {
            if let Ok(run_id) = run_id_str.parse::<Uuid>() {
                if let Ok(Some(run)) = state.orchestrator.neo4j().get_protocol_run(run_id).await {
                    if let Some(last_visit) = run.states_visited.last() {
                        rfc.current_state = Some(last_visit.state_name.clone());
                    }
                }
            }
        }
    }

    Ok(Json(PaginatedResponse::new(items, total, limit, offset)))
}

/// Get a single RFC by ID.
///
/// GET /api/rfcs/:rfc_id
pub async fn get_rfc(
    State(state): State<OrchestratorState>,
    Path(rfc_id): Path<Uuid>,
) -> Result<Json<RfcResponse>, AppError> {
    let note = state
        .orchestrator
        .note_manager()
        .get_note(rfc_id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("RFC {} not found", rfc_id)))?;

    if note.note_type != NoteType::Rfc {
        return Err(AppError::NotFound(format!("RFC {} not found", rfc_id)));
    }

    let mut rfc = note_to_rfc(&note);

    // Enrich with current_state from the protocol run if linked
    if let Some(ref run_id_str) = rfc.protocol_run_id {
        if let Ok(run_id) = run_id_str.parse::<Uuid>() {
            if let Ok(Some(run)) = state.orchestrator.neo4j().get_protocol_run(run_id).await {
                // The last entry in states_visited has the current state name
                if let Some(last_visit) = run.states_visited.last() {
                    rfc.current_state = Some(last_visit.state_name.clone());

                    // If the rfc-status tag is missing (e.g. server crash during transition),
                    // derive the status from the run's current state and backfill the tag.
                    let has_status_tag = note.tags.iter().any(|t| t.starts_with("rfc-status:"));
                    if !has_status_tag {
                        let derived_status = map_state_to_rfc_status(&last_visit.state_name);
                        rfc.status = derived_status.clone();

                        // Best-effort: backfill the missing tag so future reads are consistent
                        let mut tags = note.tags.clone();
                        set_status_tag(&mut tags, &derived_status);
                        let _ = state
                            .orchestrator
                            .note_manager()
                            .update_note(
                                rfc_id,
                                UpdateNoteRequest {
                                    content: None,
                                    importance: None,
                                    status: None,
                                    tags: Some(tags),
                                },
                            )
                            .await;
                    }
                }

                // Load available transitions from the current state
                if run.is_active() {
                    if let Ok(transitions) = state
                        .orchestrator
                        .neo4j()
                        .get_protocol_transitions(run.protocol_id)
                        .await
                    {
                        let states = state
                            .orchestrator
                            .neo4j()
                            .get_protocol_states(run.protocol_id)
                            .await
                            .unwrap_or_default();
                        let state_name_map: std::collections::HashMap<Uuid, String> =
                            states.iter().map(|s| (s.id, s.name.clone())).collect();

                        let available: Vec<AvailableTransition> = transitions
                            .iter()
                            .filter(|t| t.from_state == run.current_state)
                            .map(|t| AvailableTransition {
                                trigger: t.trigger.clone(),
                                target_state: state_name_map
                                    .get(&t.to_state)
                                    .cloned()
                                    .unwrap_or_else(|| t.to_state.to_string()),
                                guard: t.guard.clone(),
                            })
                            .collect();
                        rfc.available_transitions = Some(available);
                    }
                }
            }
        }
    }

    Ok(Json(rfc))
}

/// Create a new RFC.
///
/// POST /api/rfcs
pub async fn create_rfc(
    State(state): State<OrchestratorState>,
    Json(body): Json<CreateRfcBody>,
) -> Result<(StatusCode, Json<RfcResponse>), AppError> {
    let content = rfc_content_json(&body.title, &body.sections);

    let mut tags = body.tags.unwrap_or_default();
    set_status_tag(&mut tags, "draft");

    let importance = body
        .importance
        .as_ref()
        .and_then(|s| s.parse::<NoteImportance>().ok());

    let request = CreateNoteRequest {
        project_id: body.project_id,
        note_type: NoteType::Rfc,
        content,
        importance,
        scope: None,
        tags: Some(tags),
        anchors: None,
        assertion_rule: None,
        run_id: None,
    };

    let note = state
        .orchestrator
        .note_manager()
        .create_note(request, "api")
        .await?;

    let rfc = note_to_rfc(&note);
    Ok((StatusCode::CREATED, Json(rfc)))
}

/// Update an existing RFC.
///
/// PATCH /api/rfcs/:rfc_id
pub async fn update_rfc(
    State(state): State<OrchestratorState>,
    Path(rfc_id): Path<Uuid>,
    Json(body): Json<UpdateRfcBody>,
) -> Result<Json<RfcResponse>, AppError> {
    // Get existing note first to merge content
    let existing = state
        .orchestrator
        .note_manager()
        .get_note(rfc_id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("RFC {} not found", rfc_id)))?;

    if existing.note_type != NoteType::Rfc {
        return Err(AppError::NotFound(format!("RFC {} not found", rfc_id)));
    }

    // Merge content: update title/sections if provided
    let new_content = if body.title.is_some() || body.sections.is_some() {
        let existing_content = serde_json::from_str::<RfcContent>(&existing.content).ok();
        let title = body.title.unwrap_or_else(|| {
            existing_content
                .as_ref()
                .map(|c| c.title.clone())
                .unwrap_or_else(|| "Untitled RFC".to_string())
        });
        let sections = body
            .sections
            .unwrap_or_else(|| existing_content.map(|c| c.sections).unwrap_or_default());
        Some(rfc_content_json(&title, &sections))
    } else {
        None
    };

    // Merge tags: preserve rfc-status/rfc-run tags, update user tags if provided
    let new_tags = body.tags.map(|user_tags| {
        let mut tags: Vec<String> = existing
            .tags
            .iter()
            .filter(|t| t.starts_with("rfc-status:") || t.starts_with("rfc-run:"))
            .cloned()
            .collect();
        tags.extend(user_tags);
        tags
    });

    let importance = body
        .importance
        .as_ref()
        .and_then(|s| s.parse::<NoteImportance>().ok());

    let request = UpdateNoteRequest {
        content: new_content,
        importance,
        status: None,
        tags: new_tags,
    };

    let note = state
        .orchestrator
        .note_manager()
        .update_note(rfc_id, request)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("RFC {} not found", rfc_id)))?;

    Ok(Json(note_to_rfc(&note)))
}

/// Delete an RFC.
///
/// DELETE /api/rfcs/:rfc_id
pub async fn delete_rfc(
    State(state): State<OrchestratorState>,
    Path(rfc_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    // Verify it's actually an RFC before deleting
    let note = state
        .orchestrator
        .note_manager()
        .get_note(rfc_id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("RFC {} not found", rfc_id)))?;

    if note.note_type != NoteType::Rfc {
        return Err(AppError::NotFound(format!("RFC {} not found", rfc_id)));
    }

    let deleted = state
        .orchestrator
        .note_manager()
        .delete_note(rfc_id)
        .await?;

    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!("RFC {} not found", rfc_id)))
    }
}

/// Fire a lifecycle transition on an RFC.
///
/// POST /api/rfcs/:rfc_id/transition
///
/// Body: `{ "action": "propose" | "accept" | "reject" | "implement" }`
///
/// This finds the protocol run linked via `rfc-run:<uuid>` tag, fires the
/// transition on it, and updates the `rfc-status:<status>` tag accordingly.
pub async fn transition_rfc(
    State(state): State<OrchestratorState>,
    Path(rfc_id): Path<Uuid>,
    Json(body): Json<TransitionRfcBody>,
) -> Result<Json<RfcResponse>, AppError> {
    // 1. Get the note
    let note = state
        .orchestrator
        .note_manager()
        .get_note(rfc_id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("RFC {} not found", rfc_id)))?;

    if note.note_type != NoteType::Rfc {
        return Err(AppError::NotFound(format!("RFC {} not found", rfc_id)));
    }

    // 2. Extract or auto-create protocol run
    let run_id = if let Some(run_id_str) = extract_run_id(&note.tags) {
        let run_id = run_id_str.parse::<Uuid>().map_err(|_| {
            AppError::BadRequest(format!(
                "RFC {} has invalid rfc-run tag value: {}",
                rfc_id, run_id_str
            ))
        })?;

        // 3a. If the run is failed/cancelled, recover it to running before attempting transition.
        if let Ok(Some(mut run)) = state.orchestrator.neo4j().get_protocol_run(run_id).await {
            if run.status == protocol::RunStatus::Failed
                || run.status == protocol::RunStatus::Cancelled
            {
                run.status = protocol::RunStatus::Running;
                run.error = None;
                run.completed_at = None;
                let _ = state
                    .orchestrator
                    .neo4j()
                    .update_protocol_run(&mut run)
                    .await;
            }
        }

        run_id
    } else {
        // 3b. No run linked — auto-create one on the rfc-lifecycle protocol.
        //     If the protocol doesn't exist, bootstrap it automatically.
        let project_id = note.project_id.ok_or_else(|| {
            AppError::BadRequest(
                "RFC has no project_id — cannot find rfc-lifecycle protocol".to_string(),
            )
        })?;

        let lifecycle_protocol =
            find_or_create_rfc_lifecycle(state.orchestrator.neo4j(), project_id).await?;

        // Resolve the entry state
        let states = state
            .orchestrator
            .neo4j()
            .get_protocol_states(lifecycle_protocol.id)
            .await
            .map_err(AppError::Internal)?;

        let entry_state = states
            .iter()
            .find(|s| s.id == lifecycle_protocol.entry_state)
            .ok_or_else(|| {
                AppError::Internal(anyhow::anyhow!(
                    "Entry state not found for rfc-lifecycle protocol"
                ))
            })?;

        // Create a dedicated run for this RFC (bypass start_run concurrency check —
        // each RFC gets its own independent run on the shared protocol)
        let mut run =
            protocol::ProtocolRun::new(lifecycle_protocol.id, entry_state.id, &entry_state.name);
        run.triggered_by = "rfc-auto-create".to_string();

        state
            .orchestrator
            .neo4j()
            .create_protocol_run(&run)
            .await
            .map_err(AppError::Internal)?;

        // Link the run to the RFC via tag
        let mut tags = note.tags.clone();
        tags.push(format!("rfc-run:{}", run.id));
        let _ = state
            .orchestrator
            .note_manager()
            .update_note(
                rfc_id,
                UpdateNoteRequest {
                    content: None,
                    importance: None,
                    status: None,
                    tags: Some(tags),
                },
            )
            .await;

        run.id
    };

    // 4. Fire transition on the protocol run
    let result =
        protocol::engine::fire_transition(state.orchestrator.neo4j(), run_id, &body.action)
            .await
            .map_err(|e| {
                if e.to_string().contains("not found") {
                    AppError::NotFound(e.to_string())
                } else {
                    AppError::Internal(e)
                }
            })?;

    if !result.success {
        return Err(AppError::BadRequest(
            result
                .error
                .unwrap_or_else(|| "Transition failed".to_string()),
        ));
    }

    // 4. Map the new state name to an RFC status and update the tag.
    //    Re-read the note to get the latest tags (the auto-create path may have
    //    added an rfc-run:<uuid> tag since we first loaded `note`).
    let fresh_note = state
        .orchestrator
        .note_manager()
        .get_note(rfc_id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("RFC {} disappeared mid-transition", rfc_id)))?;
    let new_status = map_state_to_rfc_status(&result.current_state_name);
    let mut tags = fresh_note.tags.clone();
    set_status_tag(&mut tags, &new_status);

    let update_req = UpdateNoteRequest {
        content: None,
        importance: None,
        status: None,
        tags: Some(tags),
    };

    let updated_note = state
        .orchestrator
        .note_manager()
        .update_note(rfc_id, update_req)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("RFC {} not found after transition", rfc_id)))?;

    let mut rfc = note_to_rfc(&updated_note);
    rfc.current_state = Some(result.current_state_name);

    Ok(Json(rfc))
}

/// Map a protocol FSM state name to an RFC status string.
///
/// The RFC lifecycle protocol typically has states like:
/// draft → proposed → accepted → implemented (or rejected at any point).
fn map_state_to_rfc_status(state_name: &str) -> String {
    let lower = state_name.to_lowercase();
    match lower.as_str() {
        "draft" => "draft".to_string(),
        "proposed" | "in_review" | "review" => "proposed".to_string(),
        "accepted" | "approved" => "accepted".to_string(),
        "implemented" | "done" | "completed" => "implemented".to_string(),
        "rejected" | "declined" => "rejected".to_string(),
        // Fallback: use the state name as-is
        _ => lower,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::notes::{CreateNoteRequest, NoteImportance, NoteScope, NoteType};
    use crate::orchestrator::watcher::FileWatcher;
    use crate::orchestrator::Orchestrator;
    use crate::test_helpers::{mock_app_state, test_auth_config, test_bearer_token, test_project};
    use axum::{
        body::Body,
        http::{Request, StatusCode as AxumStatus},
    };
    use tower::ServiceExt;

    // ----------------------------------------------------------------
    // Test infrastructure
    // ----------------------------------------------------------------

    async fn mock_server_state() -> super::OrchestratorState {
        let app_state = mock_app_state();
        let orchestrator = std::sync::Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = std::sync::Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        std::sync::Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: std::sync::Arc::new(crate::events::HybridEmitter::new(std::sync::Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: std::sync::Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
        })
    }

    /// Build a mock server state with a pre-seeded project and return the project ID.
    async fn mock_server_with_project() -> (super::OrchestratorState, Uuid) {
        let app_state = mock_app_state();
        let project = test_project();
        let project_id = project.id;
        app_state.neo4j.create_project(&project).await.unwrap();

        let orchestrator = std::sync::Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = std::sync::Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = std::sync::Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: std::sync::Arc::new(crate::events::HybridEmitter::new(std::sync::Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: std::sync::Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
        });
        (state, project_id)
    }

    fn authed_get(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    fn authed_post(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .header("authorization", test_bearer_token())
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn authed_patch(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("PATCH")
            .uri(uri)
            .header("content-type", "application/json")
            .header("authorization", test_bearer_token())
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn authed_delete(uri: &str) -> Request<Body> {
        Request::builder()
            .method("DELETE")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    async fn body_json(resp: axum::http::Response<Body>) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// Create an RFC note directly in the mock store and return its ID.
    async fn seed_rfc(state: &super::OrchestratorState, project_id: Uuid) -> Uuid {
        let content = serde_json::json!({
            "title": "Test RFC",
            "sections": [
                {"title": "Problem", "content": "Something needs fixing"},
                {"title": "Proposed Solution", "content": "Fix it this way"}
            ]
        });
        let req = CreateNoteRequest {
            project_id: Some(project_id),
            note_type: NoteType::Rfc,
            content: content.to_string(),
            importance: Some(NoteImportance::High),
            scope: None,
            tags: Some(vec!["rfc-status:draft".to_string()]),
            anchors: None,
            assertion_rule: None,
            run_id: None,
        };
        let note = state
            .orchestrator
            .note_manager()
            .create_note(req, "test")
            .await
            .unwrap();
        note.id
    }

    // ----------------------------------------------------------------
    // Unit tests — pure functions
    // ----------------------------------------------------------------

    #[test]
    fn test_extract_rfc_status_present() {
        let tags = vec![
            "rfc".to_string(),
            "rfc-status:proposed".to_string(),
            "feature".to_string(),
        ];
        assert_eq!(extract_rfc_status(&tags), "proposed");
    }

    #[test]
    fn test_extract_rfc_status_missing() {
        let tags = vec!["rfc".to_string(), "feature".to_string()];
        assert_eq!(extract_rfc_status(&tags), "draft");
    }

    #[test]
    fn test_extract_run_id_present() {
        let run_id = Uuid::new_v4();
        let tags = vec![format!("rfc-run:{}", run_id)];
        assert_eq!(extract_run_id(&tags), Some(run_id.to_string()));
    }

    #[test]
    fn test_extract_run_id_missing() {
        let tags = vec!["rfc-status:draft".to_string()];
        assert_eq!(extract_run_id(&tags), None);
    }

    #[test]
    fn test_set_status_tag_replaces_existing() {
        let mut tags = vec!["rfc-status:draft".to_string(), "feature".to_string()];
        set_status_tag(&mut tags, "proposed");
        assert!(tags.contains(&"rfc-status:proposed".to_string()));
        assert!(!tags.contains(&"rfc-status:draft".to_string()));
        assert!(tags.contains(&"feature".to_string()));
    }

    #[test]
    fn test_set_status_tag_adds_if_missing() {
        let mut tags = vec!["feature".to_string()];
        set_status_tag(&mut tags, "draft");
        assert_eq!(tags.len(), 2);
        assert!(tags.contains(&"rfc-status:draft".to_string()));
    }

    #[test]
    fn test_map_state_to_rfc_status_known_states() {
        assert_eq!(map_state_to_rfc_status("draft"), "draft");
        assert_eq!(map_state_to_rfc_status("proposed"), "proposed");
        assert_eq!(map_state_to_rfc_status("in_review"), "proposed");
        assert_eq!(map_state_to_rfc_status("review"), "proposed");
        assert_eq!(map_state_to_rfc_status("accepted"), "accepted");
        assert_eq!(map_state_to_rfc_status("approved"), "accepted");
        assert_eq!(map_state_to_rfc_status("implemented"), "implemented");
        assert_eq!(map_state_to_rfc_status("done"), "implemented");
        assert_eq!(map_state_to_rfc_status("completed"), "implemented");
        assert_eq!(map_state_to_rfc_status("rejected"), "rejected");
        assert_eq!(map_state_to_rfc_status("declined"), "rejected");
    }

    #[test]
    fn test_map_state_to_rfc_status_case_insensitive() {
        assert_eq!(map_state_to_rfc_status("DRAFT"), "draft");
        assert_eq!(map_state_to_rfc_status("Proposed"), "proposed");
        assert_eq!(map_state_to_rfc_status("ACCEPTED"), "accepted");
    }

    #[test]
    fn test_map_state_to_rfc_status_fallback() {
        assert_eq!(map_state_to_rfc_status("under_review"), "under_review");
        assert_eq!(map_state_to_rfc_status("planning"), "planning");
        assert_eq!(map_state_to_rfc_status("in_progress"), "in_progress");
        assert_eq!(map_state_to_rfc_status("superseded"), "superseded");
    }

    #[test]
    fn test_extract_title_from_markdown_h1() {
        assert_eq!(
            extract_title_from_markdown("# My Title\nSome content"),
            Some("My Title".to_string())
        );
    }

    #[test]
    fn test_extract_title_from_markdown_h2() {
        assert_eq!(
            extract_title_from_markdown("## My Title\nSome content"),
            Some("My Title".to_string())
        );
    }

    #[test]
    fn test_extract_title_from_markdown_rfc_prefix() {
        assert_eq!(
            extract_title_from_markdown("## RFC — Auto Bootstrap\nContent"),
            Some("Auto Bootstrap".to_string())
        );
        assert_eq!(
            extract_title_from_markdown("## RFC: Auto Bootstrap\nContent"),
            Some("Auto Bootstrap".to_string())
        );
        assert_eq!(
            extract_title_from_markdown("## RFC - Auto Bootstrap\nContent"),
            Some("Auto Bootstrap".to_string())
        );
    }

    #[test]
    fn test_extract_title_from_markdown_no_heading() {
        assert_eq!(
            extract_title_from_markdown("Just some text\nNo heading here"),
            None
        );
    }

    #[test]
    fn test_note_to_rfc_json_content() {
        let content = serde_json::json!({
            "title": "My RFC",
            "sections": [{"title": "Problem", "content": "Something broke"}]
        });
        let note = crate::notes::Note::new_full(
            Some(Uuid::new_v4()),
            NoteType::Rfc,
            NoteImportance::High,
            NoteScope::Project,
            content.to_string(),
            vec!["rfc-status:draft".to_string()],
            "test".to_string(),
        );
        let rfc = note_to_rfc(&note);
        assert_eq!(rfc.title, "My RFC");
        assert_eq!(rfc.status, "draft");
        assert_eq!(rfc.importance, "high");
        assert_eq!(rfc.sections.len(), 1);
        assert_eq!(rfc.sections[0].title, "Problem");
    }

    #[test]
    fn test_note_to_rfc_markdown_content() {
        let note = crate::notes::Note::new_full(
            Some(Uuid::new_v4()),
            NoteType::Rfc,
            NoteImportance::Medium,
            NoteScope::Project,
            "## RFC — Auto Bootstrap\n\nSome content here".to_string(),
            vec!["rfc-status:proposed".to_string()],
            "test".to_string(),
        );
        let rfc = note_to_rfc(&note);
        assert_eq!(rfc.title, "Auto Bootstrap");
        assert_eq!(rfc.status, "proposed");
        assert_eq!(rfc.sections.len(), 1);
        assert_eq!(rfc.sections[0].title, "Content");
    }

    #[test]
    fn test_note_to_rfc_plain_text_fallback() {
        let note = crate::notes::Note::new_full(
            Some(Uuid::new_v4()),
            NoteType::Rfc,
            NoteImportance::Low,
            NoteScope::Project,
            "Just plain text, no heading".to_string(),
            vec!["rfc-status:draft".to_string()],
            "test".to_string(),
        );
        let rfc = note_to_rfc(&note);
        assert_eq!(rfc.title, "Untitled RFC");
    }

    #[test]
    fn test_note_to_rfc_filters_internal_tags() {
        let note = crate::notes::Note::new_full(
            Some(Uuid::new_v4()),
            NoteType::Rfc,
            NoteImportance::Medium,
            NoteScope::Project,
            "{}".to_string(),
            vec![
                "rfc-status:draft".to_string(),
                "rfc-run:some-uuid".to_string(),
                "user-tag".to_string(),
            ],
            "test".to_string(),
        );
        let rfc = note_to_rfc(&note);
        assert_eq!(rfc.tags, vec!["user-tag".to_string()]);
    }

    #[test]
    fn test_note_to_rfc_extracts_run_id() {
        let run_id = Uuid::new_v4();
        let note = crate::notes::Note::new_full(
            Some(Uuid::new_v4()),
            NoteType::Rfc,
            NoteImportance::Medium,
            NoteScope::Project,
            "{}".to_string(),
            vec![format!("rfc-run:{}", run_id)],
            "test".to_string(),
        );
        let rfc = note_to_rfc(&note);
        assert_eq!(rfc.protocol_run_id, Some(run_id.to_string()));
    }

    #[test]
    fn test_rfc_content_json_roundtrip() {
        let sections = vec![
            RfcSection {
                title: "Problem".to_string(),
                content: "Something".to_string(),
            },
            RfcSection {
                title: "Solution".to_string(),
                content: "Fix it".to_string(),
            },
        ];
        let json = rfc_content_json("My RFC", &sections);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["title"], "My RFC");
        assert_eq!(parsed["sections"].as_array().unwrap().len(), 2);
    }

    // ----------------------------------------------------------------
    // Handler integration tests (axum + mock backends)
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_list_rfcs_empty() {
        let state = mock_server_state().await;
        let app = create_router(state);

        let resp = app.oneshot(authed_get("/api/rfcs")).await.unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let json = body_json(resp).await;
        assert_eq!(json["total"], 0);
        assert_eq!(json["items"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_create_rfc() {
        let (state, project_id) = mock_server_with_project().await;
        let app = create_router(state);

        let body = serde_json::json!({
            "title": "New RFC",
            "sections": [{"title": "Problem", "content": "Need improvement"}],
            "importance": "high",
            "project_id": project_id.to_string(),
            "tags": ["backend"]
        });

        let resp = app.oneshot(authed_post("/api/rfcs", body)).await.unwrap();
        assert_eq!(resp.status(), AxumStatus::CREATED);

        let json = body_json(resp).await;
        assert_eq!(json["title"], "New RFC");
        assert_eq!(json["status"], "draft");
        assert_eq!(json["importance"], "high");
        assert_eq!(json["sections"].as_array().unwrap().len(), 1);
        assert!(json["id"].as_str().is_some());
        // Internal tags should be filtered
        let tags = json["tags"].as_array().unwrap();
        assert!(tags.iter().any(|t| t.as_str() == Some("backend")));
        assert!(!tags
            .iter()
            .any(|t| t.as_str().unwrap().starts_with("rfc-status:")));
    }

    #[tokio::test]
    async fn test_get_rfc() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;
        let app = create_router(state);

        let resp = app
            .oneshot(authed_get(&format!("/api/rfcs/{}", rfc_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let json = body_json(resp).await;
        assert_eq!(json["id"], rfc_id.to_string());
        assert_eq!(json["title"], "Test RFC");
        assert_eq!(json["status"], "draft");
        assert_eq!(json["sections"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_get_rfc_not_found() {
        let state = mock_server_state().await;
        let app = create_router(state);
        let fake_id = Uuid::new_v4();

        let resp = app
            .oneshot(authed_get(&format!("/api/rfcs/{}", fake_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_update_rfc_title() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;
        let app = create_router(state);

        let body = serde_json::json!({"title": "Updated RFC Title"});
        let resp = app
            .oneshot(authed_patch(&format!("/api/rfcs/{}", rfc_id), body))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let json = body_json(resp).await;
        assert_eq!(json["title"], "Updated RFC Title");
        // Sections should be preserved
        assert_eq!(json["sections"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_update_rfc_sections() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;
        let app = create_router(state);

        let body = serde_json::json!({
            "sections": [{"title": "New Section", "content": "New content"}]
        });
        let resp = app
            .oneshot(authed_patch(&format!("/api/rfcs/{}", rfc_id), body))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let json = body_json(resp).await;
        assert_eq!(json["sections"].as_array().unwrap().len(), 1);
        assert_eq!(json["sections"][0]["title"], "New Section");
        // Title should be preserved
        assert_eq!(json["title"], "Test RFC");
    }

    #[tokio::test]
    async fn test_update_rfc_importance() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;
        let app = create_router(state);

        let body = serde_json::json!({"importance": "critical"});
        let resp = app
            .oneshot(authed_patch(&format!("/api/rfcs/{}", rfc_id), body))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let json = body_json(resp).await;
        assert_eq!(json["importance"], "critical");
    }

    #[tokio::test]
    async fn test_update_rfc_not_found() {
        let state = mock_server_state().await;
        let app = create_router(state);
        let fake_id = Uuid::new_v4();

        let body = serde_json::json!({"title": "No such RFC"});
        let resp = app
            .oneshot(authed_patch(&format!("/api/rfcs/{}", fake_id), body))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_delete_rfc() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;
        let app = create_router(state);

        let resp = app
            .oneshot(authed_delete(&format!("/api/rfcs/{}", rfc_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_delete_rfc_not_found() {
        let state = mock_server_state().await;
        let app = create_router(state);
        let fake_id = Uuid::new_v4();

        let resp = app
            .oneshot(authed_delete(&format!("/api/rfcs/{}", fake_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_list_rfcs_returns_created() {
        let (state, project_id) = mock_server_with_project().await;
        let _rfc_id = seed_rfc(&state, project_id).await;
        let app = create_router(state);

        let resp = app.oneshot(authed_get("/api/rfcs")).await.unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let json = body_json(resp).await;
        assert_eq!(json["total"], 1);
        let items = json["items"].as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["title"], "Test RFC");
        assert_eq!(items[0]["status"], "draft");
    }

    #[tokio::test]
    async fn test_list_rfcs_with_status_filter() {
        let (state, project_id) = mock_server_with_project().await;
        // Create a draft RFC
        let _rfc_id = seed_rfc(&state, project_id).await;
        let app = create_router(state);

        // Filter by "proposed" — should return 0
        let resp = app
            .oneshot(authed_get("/api/rfcs?status=proposed"))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["total"], 0);
    }

    #[tokio::test]
    async fn test_list_rfcs_with_draft_filter() {
        let (state, project_id) = mock_server_with_project().await;
        let _rfc_id = seed_rfc(&state, project_id).await;
        let app = create_router(state);

        let resp = app
            .oneshot(authed_get("/api/rfcs?status=draft"))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["total"], 1);
    }

    #[tokio::test]
    async fn test_transition_rfc_bootstrap_and_propose() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;
        let app = create_router(state);

        // Transition: draft → proposed (triggers bootstrap of rfc-lifecycle protocol)
        let body = serde_json::json!({"action": "propose"});
        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", rfc_id),
                body,
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let json = body_json(resp).await;
        assert_eq!(json["current_state"], "proposed");
        assert_eq!(json["status"], "proposed");
        // Should have a protocol run linked
        assert!(json["protocol_run_id"].as_str().is_some());
    }

    #[tokio::test]
    async fn test_transition_rfc_full_lifecycle() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;

        // draft → proposed
        let app = create_router(state.clone());
        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", rfc_id),
                serde_json::json!({"action": "propose"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["current_state"], "proposed");

        // proposed → under_review
        let app = create_router(state.clone());
        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", rfc_id),
                serde_json::json!({"action": "submit_review"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["current_state"], "under_review");

        // under_review → accepted
        let app = create_router(state.clone());
        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", rfc_id),
                serde_json::json!({"action": "accept"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["current_state"], "accepted");

        // accepted → planning
        let app = create_router(state.clone());
        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", rfc_id),
                serde_json::json!({"action": "start_planning"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["current_state"], "planning");

        // planning → in_progress
        let app = create_router(state.clone());
        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", rfc_id),
                serde_json::json!({"action": "start_work"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["current_state"], "in_progress");

        // in_progress → implemented
        let app = create_router(state.clone());
        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", rfc_id),
                serde_json::json!({"action": "complete"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["current_state"], "implemented");
    }

    #[tokio::test]
    async fn test_transition_rfc_invalid_trigger() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;

        // First bootstrap with a valid transition
        let app = create_router(state.clone());
        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", rfc_id),
                serde_json::json!({"action": "propose"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        // Try invalid trigger from proposed state (accept is not valid from proposed)
        let app = create_router(state.clone());
        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", rfc_id),
                serde_json::json!({"action": "accept"}),
            ))
            .await
            .unwrap();
        // Should fail — accept is not valid from proposed
        assert_ne!(resp.status(), AxumStatus::OK);
    }

    #[tokio::test]
    async fn test_transition_rfc_reject_from_proposed() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;

        // draft → proposed
        let app = create_router(state.clone());
        app.oneshot(authed_post(
            &format!("/api/rfcs/{}/transition", rfc_id),
            serde_json::json!({"action": "propose"}),
        ))
        .await
        .unwrap();

        // proposed → rejected
        let app = create_router(state.clone());
        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", rfc_id),
                serde_json::json!({"action": "reject"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["current_state"], "rejected");
    }

    #[tokio::test]
    async fn test_transition_rfc_supersede_from_draft() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;

        // draft → superseded
        let app = create_router(state.clone());
        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", rfc_id),
                serde_json::json!({"action": "supersede"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["current_state"], "superseded");
    }

    #[tokio::test]
    async fn test_transition_rfc_revise_back_to_draft() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;

        // draft → proposed → under_review
        let app = create_router(state.clone());
        app.oneshot(authed_post(
            &format!("/api/rfcs/{}/transition", rfc_id),
            serde_json::json!({"action": "propose"}),
        ))
        .await
        .unwrap();
        let app = create_router(state.clone());
        app.oneshot(authed_post(
            &format!("/api/rfcs/{}/transition", rfc_id),
            serde_json::json!({"action": "submit_review"}),
        ))
        .await
        .unwrap();

        // under_review → draft (revise)
        let app = create_router(state.clone());
        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", rfc_id),
                serde_json::json!({"action": "revise"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["current_state"], "draft");
    }

    #[tokio::test]
    async fn test_transition_rfc_replan() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;

        // Move to in_progress: draft → proposed → under_review → accepted → planning → in_progress
        for action in &[
            "propose",
            "submit_review",
            "accept",
            "start_planning",
            "start_work",
        ] {
            let app = create_router(state.clone());
            let resp = app
                .oneshot(authed_post(
                    &format!("/api/rfcs/{}/transition", rfc_id),
                    serde_json::json!({"action": action}),
                ))
                .await
                .unwrap();
            assert_eq!(resp.status(), AxumStatus::OK);
        }

        // in_progress → planning (replan)
        let app = create_router(state.clone());
        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", rfc_id),
                serde_json::json!({"action": "replan"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["current_state"], "planning");
    }

    #[tokio::test]
    async fn test_transition_rfc_not_found() {
        let state = mock_server_state().await;
        let app = create_router(state);
        let fake_id = Uuid::new_v4();

        let resp = app
            .oneshot(authed_post(
                &format!("/api/rfcs/{}/transition", fake_id),
                serde_json::json!({"action": "propose"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_rfc_with_available_transitions() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;

        // Trigger a transition to create the protocol run
        let app = create_router(state.clone());
        app.oneshot(authed_post(
            &format!("/api/rfcs/{}/transition", rfc_id),
            serde_json::json!({"action": "propose"}),
        ))
        .await
        .unwrap();

        // Now GET should return available_transitions
        let app = create_router(state.clone());
        let resp = app
            .oneshot(authed_get(&format!("/api/rfcs/{}", rfc_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let json = body_json(resp).await;
        assert_eq!(json["current_state"], "proposed");
        let transitions = json["available_transitions"].as_array().unwrap();
        assert!(!transitions.is_empty());
        // From proposed, we should have submit_review, reject, supersede
        let triggers: Vec<&str> = transitions
            .iter()
            .map(|t| t["trigger"].as_str().unwrap())
            .collect();
        assert!(triggers.contains(&"submit_review"));
        assert!(triggers.contains(&"reject"));
        assert!(triggers.contains(&"supersede"));
    }

    #[tokio::test]
    async fn test_list_rfcs_enriches_current_state() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;

        // Transition to proposed
        let app = create_router(state.clone());
        app.oneshot(authed_post(
            &format!("/api/rfcs/{}/transition", rfc_id),
            serde_json::json!({"action": "propose"}),
        ))
        .await
        .unwrap();

        // List should show current_state
        let app = create_router(state.clone());
        let resp = app.oneshot(authed_get("/api/rfcs")).await.unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let json = body_json(resp).await;
        let items = json["items"].as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["current_state"], "proposed");
    }

    // ----------------------------------------------------------------
    // Bootstrap tests
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_bootstrap_creates_protocol() {
        let app_state = mock_app_state();
        let project = test_project();
        let project_id = project.id;
        app_state.neo4j.create_project(&project).await.unwrap();

        // Should have no protocols initially
        let (protocols, _) = app_state
            .neo4j
            .list_protocols(project_id, None, 100, 0)
            .await
            .unwrap();
        assert_eq!(protocols.len(), 0);

        // Bootstrap
        let protocol = find_or_create_rfc_lifecycle(app_state.neo4j.as_ref(), project_id)
            .await
            .unwrap();
        assert_eq!(protocol.name, "rfc-lifecycle");

        // Verify states
        let states = app_state
            .neo4j
            .get_protocol_states(protocol.id)
            .await
            .unwrap();
        assert_eq!(states.len(), 9);

        let state_names: Vec<&str> = states.iter().map(|s| s.name.as_str()).collect();
        assert!(state_names.contains(&"draft"));
        assert!(state_names.contains(&"proposed"));
        assert!(state_names.contains(&"under_review"));
        assert!(state_names.contains(&"accepted"));
        assert!(state_names.contains(&"planning"));
        assert!(state_names.contains(&"in_progress"));
        assert!(state_names.contains(&"implemented"));
        assert!(state_names.contains(&"rejected"));
        assert!(state_names.contains(&"superseded"));

        // Verify transitions
        let transitions = app_state
            .neo4j
            .get_protocol_transitions(protocol.id)
            .await
            .unwrap();
        assert_eq!(transitions.len(), 16);

        // Verify entry state is draft
        let entry = states
            .iter()
            .find(|s| s.id == protocol.entry_state)
            .unwrap();
        assert_eq!(entry.name, "draft");

        // Verify terminal states
        assert_eq!(protocol.terminal_states.len(), 3);
    }

    #[tokio::test]
    async fn test_bootstrap_idempotent() {
        let app_state = mock_app_state();
        let project = test_project();
        let project_id = project.id;
        app_state.neo4j.create_project(&project).await.unwrap();

        // Bootstrap twice
        let p1 = find_or_create_rfc_lifecycle(app_state.neo4j.as_ref(), project_id)
            .await
            .unwrap();
        let p2 = find_or_create_rfc_lifecycle(app_state.neo4j.as_ref(), project_id)
            .await
            .unwrap();

        // Should return the same protocol
        assert_eq!(p1.id, p2.id);

        // Should still have exactly 1 protocol
        let (protocols, _) = app_state
            .neo4j
            .list_protocols(project_id, None, 100, 0)
            .await
            .unwrap();
        assert_eq!(protocols.len(), 1);
    }

    #[tokio::test]
    async fn test_bootstrap_start_state_is_start_type() {
        let app_state = mock_app_state();
        let project = test_project();
        app_state.neo4j.create_project(&project).await.unwrap();

        let protocol = find_or_create_rfc_lifecycle(app_state.neo4j.as_ref(), project.id)
            .await
            .unwrap();
        let states = app_state
            .neo4j
            .get_protocol_states(protocol.id)
            .await
            .unwrap();

        let draft = states.iter().find(|s| s.name == "draft").unwrap();
        assert_eq!(draft.state_type, StateType::Start);
    }

    #[tokio::test]
    async fn test_bootstrap_terminal_states() {
        let app_state = mock_app_state();
        let project = test_project();
        app_state.neo4j.create_project(&project).await.unwrap();

        let protocol = find_or_create_rfc_lifecycle(app_state.neo4j.as_ref(), project.id)
            .await
            .unwrap();
        let states = app_state
            .neo4j
            .get_protocol_states(protocol.id)
            .await
            .unwrap();

        for name in &["implemented", "rejected", "superseded"] {
            let s = states.iter().find(|s| s.name == *name).unwrap();
            assert_eq!(
                s.state_type,
                StateType::Terminal,
                "{} should be terminal",
                name
            );
        }
    }

    // ----------------------------------------------------------------
    // Edge cases
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_create_rfc_without_project_id() {
        let state = mock_server_state().await;
        let app = create_router(state);

        let body = serde_json::json!({
            "title": "Orphan RFC",
            "sections": [{"title": "Problem", "content": "No project"}]
        });

        let resp = app.oneshot(authed_post("/api/rfcs", body)).await.unwrap();
        assert_eq!(resp.status(), AxumStatus::CREATED);

        let json = body_json(resp).await;
        assert_eq!(json["title"], "Orphan RFC");
    }

    #[tokio::test]
    async fn test_create_rfc_default_importance() {
        let state = mock_server_state().await;
        let app = create_router(state);

        let body = serde_json::json!({
            "title": "RFC No Importance",
            "sections": [{"title": "Problem", "content": "Default importance"}]
        });

        let resp = app.oneshot(authed_post("/api/rfcs", body)).await.unwrap();
        assert_eq!(resp.status(), AxumStatus::CREATED);

        let json = body_json(resp).await;
        // Default importance should be medium
        assert_eq!(json["importance"], "medium");
    }

    #[tokio::test]
    async fn test_update_rfc_preserves_status_tags() {
        let (state, project_id) = mock_server_with_project().await;
        let rfc_id = seed_rfc(&state, project_id).await;
        let app = create_router(state);

        // Update tags — internal rfc-status tag should be preserved
        let body = serde_json::json!({"tags": ["new-tag"]});
        let resp = app
            .oneshot(authed_patch(&format!("/api/rfcs/{}", rfc_id), body))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let json = body_json(resp).await;
        let tags = json["tags"].as_array().unwrap();
        assert!(tags.iter().any(|t| t.as_str() == Some("new-tag")));
        // Status should still be draft (preserved internally)
        assert_eq!(json["status"], "draft");
    }

    #[tokio::test]
    async fn test_list_rfcs_pagination() {
        let (state, project_id) = mock_server_with_project().await;

        // Create 3 RFCs
        for _ in 0..3 {
            seed_rfc(&state, project_id).await;
        }

        let app = create_router(state);

        // Request with limit=2
        let resp = app.oneshot(authed_get("/api/rfcs?limit=2")).await.unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let json = body_json(resp).await;
        assert_eq!(json["total"], 3);
        assert_eq!(json["items"].as_array().unwrap().len(), 2);
    }
}
