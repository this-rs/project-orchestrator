//! API handlers for Sharing & Privacy (Privacy MVP)
//!
//! These handlers back the MCP `sharing` mega-tool (12 actions) and provide
//! the REST surface consumed by the frontend Sharing page.

use super::handlers::{AppError, OrchestratorState};
use crate::episodes::distill_models::{
    ConsentStats, PrivacyMode, SharingConsent, SharingEvent, SharingMode, SharingPolicy,
};
use crate::notes::NoteFilters;
use crate::reception::anchor::SignedTombstone;
use crate::sharing::consent_gate::run_consent_gate;
use axum::{
    extract::{Path, Query, State},
    Json,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Request / Response types
// ============================================================================

/// Response for GET /api/projects/{slug}/sharing
#[derive(Debug, Serialize)]
pub struct SharingStatusResponse {
    pub enabled: bool,
    pub policy: SharingPolicy,
}

/// Request body for PUT /api/projects/{slug}/sharing/policy
#[derive(Debug, Deserialize)]
pub struct PolicyUpdateRequest {
    pub mode: Option<SharingMode>,
    pub min_shareability_score: Option<f64>,
    #[serde(default)]
    pub type_overrides: Option<std::collections::HashMap<String, crate::episodes::distill_models::SharingAction>>,
}

/// Request body for PUT /api/notes/{note_id}/sharing/consent
#[derive(Debug, Deserialize)]
pub struct ConsentUpdateRequest {
    pub consent: SharingConsent,
}

/// Request body for POST /api/projects/{slug}/sharing/retract
#[derive(Debug, Deserialize)]
pub struct RetractRequest {
    pub note_id: Option<Uuid>,
    pub content_hash: Option<String>,
    pub reason: Option<String>,
}

/// Query params for GET /api/projects/{slug}/sharing/history
#[derive(Debug, Deserialize, Default)]
pub struct HistoryQuery {
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

/// Preview item
#[derive(Debug, Serialize)]
pub struct PreviewItem {
    pub note_id: String,
    pub note_type: String,
    pub consent: SharingConsent,
    pub shareability_score: f64,
    pub decision: String, // "allow" | "deny"
}

/// Suggestion item
#[derive(Debug, Serialize)]
pub struct SuggestionItem {
    pub note_id: String,
    pub note_type: String,
    pub shareability_score: f64,
    pub reason: String,
}

/// Last privacy report response
#[derive(Debug, Serialize)]
pub struct PrivacyReportResponse {
    pub stats: Option<ConsentStats>,
    pub generated_at: String,
}

// ============================================================================
// Helpers
// ============================================================================

/// Resolve a project slug to a project_id, returning AppError::NotFound if missing.
async fn resolve_project_id(state: &OrchestratorState, slug: &str) -> Result<Uuid, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(slug)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", slug)))?;
    Ok(project.id)
}

/// Load all notes for a project (up to 500).
async fn load_project_notes(
    state: &OrchestratorState,
    project_id: Uuid,
) -> Result<Vec<crate::notes::Note>, AppError> {
    let filters = NoteFilters {
        limit: Some(500),
        offset: Some(0),
        ..Default::default()
    };
    let (notes, _total) = state
        .orchestrator
        .neo4j()
        .list_notes(Some(project_id), None, &filters)
        .await
        .map_err(AppError::Internal)?;
    Ok(notes)
}

/// Load the sharing policy for a project, falling back to defaults.
async fn load_policy(state: &OrchestratorState, project_id: Uuid) -> Result<SharingPolicy, AppError> {
    let policy = state
        .orchestrator
        .neo4j()
        .get_sharing_policy(project_id)
        .await
        .map_err(AppError::Internal)?
        .unwrap_or_default();
    Ok(policy)
}

// ============================================================================
// Handlers
// ============================================================================

/// GET /api/projects/{slug}/sharing — global sharing status
pub async fn get_sharing_status(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<SharingStatusResponse>, AppError> {
    let project_id = resolve_project_id(&state, &slug).await?;
    let policy = load_policy(&state, project_id).await?;

    Ok(Json(SharingStatusResponse {
        enabled: policy.enabled,
        policy,
    }))
}

/// POST /api/projects/{slug}/sharing/enable
pub async fn enable_sharing(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<SharingStatusResponse>, AppError> {
    let project_id = resolve_project_id(&state, &slug).await?;
    let mut policy = load_policy(&state, project_id).await?;
    policy.enabled = true;

    state
        .orchestrator
        .neo4j()
        .update_sharing_policy(project_id, &policy)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(SharingStatusResponse {
        enabled: true,
        policy,
    }))
}

/// POST /api/projects/{slug}/sharing/disable
pub async fn disable_sharing(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<SharingStatusResponse>, AppError> {
    let project_id = resolve_project_id(&state, &slug).await?;
    let mut policy = load_policy(&state, project_id).await?;
    policy.enabled = false;

    state
        .orchestrator
        .neo4j()
        .update_sharing_policy(project_id, &policy)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(SharingStatusResponse {
        enabled: false,
        policy,
    }))
}

/// GET /api/projects/{slug}/sharing/policy
pub async fn get_sharing_policy(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<SharingPolicy>, AppError> {
    let project_id = resolve_project_id(&state, &slug).await?;
    let policy = load_policy(&state, project_id).await?;
    Ok(Json(policy))
}

/// PUT /api/projects/{slug}/sharing/policy — partial update (merge)
pub async fn set_sharing_policy(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Json(body): Json<PolicyUpdateRequest>,
) -> Result<Json<SharingPolicy>, AppError> {
    let project_id = resolve_project_id(&state, &slug).await?;
    let mut policy = load_policy(&state, project_id).await?;

    // Merge provided fields
    if let Some(mode) = body.mode {
        policy.mode = mode;
    }
    if let Some(score) = body.min_shareability_score {
        if !(0.0..=1.0).contains(&score) {
            return Err(AppError::BadRequest(
                "min_shareability_score must be between 0.0 and 1.0".into(),
            ));
        }
        policy.min_shareability_score = score;
    }
    if let Some(overrides) = body.type_overrides {
        policy.type_overrides = overrides;
    }

    state
        .orchestrator
        .neo4j()
        .update_sharing_policy(project_id, &policy)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(policy))
}

/// PUT /api/notes/{note_id}/sharing/consent
pub async fn set_sharing_consent(
    State(state): State<OrchestratorState>,
    Path(note_id): Path<Uuid>,
    Json(body): Json<ConsentUpdateRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    state
        .orchestrator
        .neo4j()
        .update_sharing_consent(note_id, &body.consent)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "note_id": note_id.to_string(),
        "consent": body.consent,
        "updated": true
    })))
}

/// GET /api/projects/{slug}/sharing/history — paginated audit trail
pub async fn get_sharing_history(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Query(params): Query<HistoryQuery>,
) -> Result<Json<Vec<SharingEvent>>, AppError> {
    let project_id = resolve_project_id(&state, &slug).await?;
    let limit = params.limit.unwrap_or(50).min(200);
    let offset = params.offset.unwrap_or(0);

    let events = state
        .orchestrator
        .neo4j()
        .list_sharing_events(project_id, limit, offset)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(events))
}

/// GET /api/projects/{slug}/sharing/preview — preview what would be shared
pub async fn preview_sharing(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<Vec<PreviewItem>>, AppError> {
    let project_id = resolve_project_id(&state, &slug).await?;
    let policy = load_policy(&state, project_id).await?;

    // Load all notes for the project
    let notes = load_project_notes(&state, project_id).await?;

    // Run consent gate
    let (_, decisions) = run_consent_gate(&notes, &policy);

    let items: Vec<PreviewItem> = decisions
        .into_iter()
        .map(|(note, decision)| {
            let score =
                crate::sharing::consent_gate::compute_shareability_score(note);
            PreviewItem {
                note_id: note.id.to_string(),
                note_type: format!("{:?}", note.note_type),
                consent: note.sharing_consent,
                shareability_score: score,
                decision: format!("{:?}", decision).to_lowercase(),
            }
        })
        .collect();

    Ok(Json(items))
}

/// GET /api/projects/{slug}/sharing/suggest — suggest notes for sharing
pub async fn suggest_sharing(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<Vec<SuggestionItem>>, AppError> {
    let project_id = resolve_project_id(&state, &slug).await?;
    let policy = load_policy(&state, project_id).await?;

    let notes = load_project_notes(&state, project_id).await?;

    // Suggest notes that score above threshold but don't yet have consent
    let mut suggestions = Vec::new();
    for note in &notes {
        let score = crate::sharing::consent_gate::compute_shareability_score(note);
        if score >= policy.min_shareability_score
            && note.sharing_consent == SharingConsent::NotSet
        {
            suggestions.push(SuggestionItem {
                note_id: note.id.to_string(),
                note_type: format!("{:?}", note.note_type),
                shareability_score: score,
                reason: format!(
                    "Score {:.2} >= threshold {:.2}, consent not yet set",
                    score, policy.min_shareability_score
                ),
            });
        }
    }

    // Sort by score descending
    suggestions.sort_by(|a, b| b.shareability_score.partial_cmp(&a.shareability_score).unwrap_or(std::cmp::Ordering::Equal));

    Ok(Json(suggestions))
}

/// POST /api/projects/{slug}/sharing/retract — retract a shared artifact
pub async fn retract_sharing(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Json(body): Json<RetractRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let _project_id = resolve_project_id(&state, &slug).await?;

    // Determine content_hash: either provided directly or computed from note_id
    let content_hash = if let Some(hash) = body.content_hash {
        hash
    } else if let Some(note_id) = body.note_id {
        // Use note_id as content hash for simplicity
        format!("note:{}", note_id)
    } else {
        return Err(AppError::BadRequest(
            "Either note_id or content_hash must be provided".into(),
        ));
    };

    // Build and persist tombstone
    let tombstone = SignedTombstone {
        content_hash: content_hash.clone(),
        issuer_did: state
            .identity
            .as_ref()
            .map(|id| id.did_key().to_string())
            .unwrap_or_else(|| "did:local:unknown".to_string()),
        signature_hex: "0".repeat(128), // placeholder — real signing requires InstanceIdentity
        issued_at: Utc::now(),
        reason: body.reason.clone(),
    };

    state
        .orchestrator
        .neo4j()
        .persist_tombstone(&tombstone)
        .await
        .map_err(AppError::Internal)?;

    // Create audit event
    let event = SharingEvent {
        id: Uuid::new_v4().to_string(),
        content_hash: content_hash.clone(),
        artifact_type: "note".into(),
        action: "retracted".into(),
        source_did: tombstone.issuer_did.clone(),
        target_did: "broadcast".into(),
        timestamp: Utc::now(),
        consent: SharingConsent::ExplicitDeny,
        privacy_mode: PrivacyMode::Standard,
        reason: body.reason,
    };

    state
        .orchestrator
        .neo4j()
        .create_sharing_event(&event)
        .await
        .map_err(AppError::Internal)?;

    // If note_id was provided, also set consent to ExplicitDeny
    if let Some(note_id) = body.note_id {
        let _ = state
            .orchestrator
            .neo4j()
            .update_sharing_consent(note_id, &SharingConsent::ExplicitDeny)
            .await;
    }

    Ok(Json(serde_json::json!({
        "retracted": true,
        "content_hash": content_hash,
        "tombstone_persisted": true,
        "event_recorded": true
    })))
}

/// GET /api/projects/{slug}/sharing/tombstones
pub async fn list_tombstones(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<Vec<SignedTombstone>>, AppError> {
    // Validate project exists
    let _project_id = resolve_project_id(&state, &slug).await?;

    let tombstones = state
        .orchestrator
        .neo4j()
        .list_tombstones()
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(tombstones))
}

/// GET /api/projects/{slug}/sharing/last-report — last privacy/consent report
pub async fn get_last_privacy_report(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<PrivacyReportResponse>, AppError> {
    let project_id = resolve_project_id(&state, &slug).await?;
    let policy = load_policy(&state, project_id).await?;

    let notes = load_project_notes(&state, project_id).await?;

    let (stats, _) = run_consent_gate(&notes, &policy);

    Ok(Json(PrivacyReportResponse {
        stats: Some(stats),
        generated_at: Utc::now().to_rfc3339(),
    }))
}
