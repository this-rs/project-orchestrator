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
use crate::notes::{
    CreateNoteRequest, Note, NoteFilters, NoteImportance, NoteType, UpdateNoteRequest,
};
use crate::protocol;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
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
                let _ = state.orchestrator.neo4j().update_protocol_run(&run).await;
            }
        }

        run_id
    } else {
        // 3b. No run linked — auto-create one on the rfc-lifecycle protocol.
        //     Find the protocol by name in the note's project (or any project).
        let project_id = note.project_id.ok_or_else(|| {
            AppError::BadRequest(
                "RFC has no project_id — cannot find rfc-lifecycle protocol".to_string(),
            )
        })?;

        let (protocols, _) = state
            .orchestrator
            .neo4j()
            .list_protocols(project_id, None, 200, 0)
            .await
            .map_err(AppError::Internal)?;

        let lifecycle_protocol = protocols
            .iter()
            .find(|p| p.name == "rfc-lifecycle")
            .ok_or_else(|| {
                AppError::BadRequest(format!(
                    "No 'rfc-lifecycle' protocol found in project {}. \
                     Create one before transitioning RFCs.",
                    project_id
                ))
            })?;

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

    // 4. Map the new state name to an RFC status and update the tag
    let new_status = map_state_to_rfc_status(&result.current_state_name);
    let mut tags = note.tags.clone();
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
