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
    pub type_overrides:
        Option<std::collections::HashMap<String, crate::episodes::distill_models::SharingAction>>,
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
    pub content_preview: String,
    pub consent: SharingConsent,
    pub shareability_score: f64,
    pub decision: String, // "allow" | "deny"
}

/// Suggestion item
#[derive(Debug, Serialize)]
pub struct SuggestionItem {
    pub note_id: String,
    pub note_type: String,
    pub content_preview: String,
    pub shareability_score: f64,
    pub reason: String,
}

/// Truncate content to a short preview (first line, max 120 chars).
fn content_preview(content: &str) -> String {
    let first_line = content.lines().next().unwrap_or("");
    // Strip leading markdown headers
    let trimmed = first_line.trim_start_matches('#').trim();
    if trimmed.len() > 120 {
        format!("{}…", &trimmed[..120])
    } else {
        trimmed.to_string()
    }
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
async fn load_policy(
    state: &OrchestratorState,
    project_id: Uuid,
) -> Result<SharingPolicy, AppError> {
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
            let score = crate::sharing::consent_gate::compute_shareability_score(note);
            PreviewItem {
                note_id: note.id.to_string(),
                note_type: format!("{:?}", note.note_type),
                content_preview: content_preview(&note.content),
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
        if score >= policy.min_shareability_score && note.sharing_consent == SharingConsent::NotSet
        {
            suggestions.push(SuggestionItem {
                note_id: note.id.to_string(),
                note_type: format!("{:?}", note.note_type),
                content_preview: content_preview(&note.content),
                shareability_score: score,
                reason: format!(
                    "Score {:.2} >= threshold {:.2}, consent not yet set",
                    score, policy.min_shareability_score
                ),
            });
        }
    }

    // Sort by score descending
    suggestions.sort_by(|a, b| {
        b.shareability_score
            .partial_cmp(&a.shareability_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::events::EventBus;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_bearer_token, test_project};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;

    // ── Helpers ────────────────────────────────────────────────────────

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
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
        });
        create_router(state)
    }

    /// Build an app with a pre-seeded project so sharing endpoints find it.
    async fn test_app_with_project() -> (axum::Router, String) {
        let app_state = mock_app_state();
        let project = test_project();
        let slug = project.slug.clone();
        app_state.neo4j.create_project(&project).await.unwrap();
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
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
        });
        (create_router(state), slug)
    }

    fn auth_get(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    fn auth_post(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap()
    }

    fn auth_put(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("PUT")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap()
    }

    async fn body_json(resp: axum::http::Response<Body>) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    // ── Unit tests: content_preview ────────────────────────────────────

    #[test]
    fn test_content_preview_simple() {
        assert_eq!(content_preview("Hello world"), "Hello world");
    }

    #[test]
    fn test_content_preview_empty() {
        assert_eq!(content_preview(""), "");
    }

    #[test]
    fn test_content_preview_markdown_header() {
        assert_eq!(content_preview("# My Title"), "My Title");
        assert_eq!(content_preview("## Sub Title"), "Sub Title");
        assert_eq!(content_preview("### Deep Title"), "Deep Title");
    }

    #[test]
    fn test_content_preview_multiline() {
        assert_eq!(
            content_preview("First line\nSecond line\nThird line"),
            "First line"
        );
    }

    #[test]
    fn test_content_preview_truncation() {
        let long = "a".repeat(200);
        let result = content_preview(&long);
        // 120 chars + "…"
        assert_eq!(result.chars().count(), 121);
        assert!(result.ends_with('…'));
    }

    #[test]
    fn test_content_preview_markdown_header_long() {
        let long = format!("# {}", "b".repeat(200));
        let result = content_preview(&long);
        assert!(result.len() <= 124); // 120 ASCII + up to 3 bytes for …
        assert!(result.ends_with('…'));
        assert!(!result.starts_with('#'));
    }

    #[test]
    fn test_content_preview_whitespace_only() {
        assert_eq!(content_preview("   "), "");
        assert_eq!(content_preview("  \n  "), "");
    }

    // ── Unit tests: request deserialization ─────────────────────────────

    #[test]
    fn test_policy_update_request_deser() {
        let json = r#"{"mode":"auto","min_shareability_score":0.7}"#;
        let req: PolicyUpdateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            serde_json::to_value(req.mode).unwrap(),
            serde_json::json!("auto")
        );
        assert_eq!(req.min_shareability_score, Some(0.7));
        assert!(req.type_overrides.is_none());
    }

    #[test]
    fn test_policy_update_request_partial() {
        let json = r#"{"mode":"suggest"}"#;
        let req: PolicyUpdateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            serde_json::to_value(req.mode).unwrap(),
            serde_json::json!("suggest")
        );
        assert!(req.min_shareability_score.is_none());
    }

    #[test]
    fn test_consent_update_request_deser() {
        let json = r#"{"consent":"explicit_allow"}"#;
        let req: ConsentUpdateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.consent, SharingConsent::ExplicitAllow);
    }

    #[test]
    fn test_retract_request_with_note_id() {
        let id = Uuid::new_v4();
        let json = format!(r#"{{"note_id":"{}","reason":"GDPR request"}}"#, id);
        let req: RetractRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req.note_id, Some(id));
        assert!(req.content_hash.is_none());
        assert_eq!(req.reason.as_deref(), Some("GDPR request"));
    }

    #[test]
    fn test_retract_request_with_content_hash() {
        let json = r#"{"content_hash":"sha256:abc123"}"#;
        let req: RetractRequest = serde_json::from_str(json).unwrap();
        assert!(req.note_id.is_none());
        assert_eq!(req.content_hash.as_deref(), Some("sha256:abc123"));
    }

    #[test]
    fn test_history_query_defaults() {
        let q: HistoryQuery = serde_json::from_str("{}").unwrap();
        assert!(q.limit.is_none());
        assert!(q.offset.is_none());
    }

    // ── Unit tests: response serialization ──────────────────────────────

    #[test]
    fn test_preview_item_serialization() {
        let item = PreviewItem {
            note_id: "abc".into(),
            note_type: "Guideline".into(),
            content_preview: "Use parameterized queries".into(),
            consent: SharingConsent::NotSet,
            shareability_score: 0.85,
            decision: "allow".into(),
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["note_id"], "abc");
        assert_eq!(json["content_preview"], "Use parameterized queries");
        assert_eq!(json["shareability_score"], 0.85);
        assert_eq!(json["decision"], "allow");
    }

    #[test]
    fn test_suggestion_item_serialization() {
        let item = SuggestionItem {
            note_id: "def".into(),
            note_type: "Gotcha".into(),
            content_preview: "Watch out for NULL".into(),
            shareability_score: 0.6,
            reason: "Score 0.60 >= threshold 0.50".into(),
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["content_preview"], "Watch out for NULL");
        assert_eq!(json["reason"], "Score 0.60 >= threshold 0.50");
    }

    #[test]
    fn test_privacy_report_response_serialization() {
        let resp = PrivacyReportResponse {
            stats: Some(ConsentStats {
                consent_allowed: 10,
                consent_denied: 2,
                consent_pending: 5,
                denied_reasons: vec![],
            }),
            generated_at: "2026-03-15T00:00:00Z".into(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["stats"]["consent_allowed"], 10);
        assert_eq!(json["stats"]["consent_denied"], 2);
        assert_eq!(json["stats"]["consent_pending"], 5);
        assert_eq!(json["generated_at"], "2026-03-15T00:00:00Z");
    }

    #[test]
    fn test_status_response_serialization() {
        let resp = SharingStatusResponse {
            enabled: true,
            policy: SharingPolicy::default(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["enabled"], true);
        assert!(json["policy"].is_object());
    }

    // ── Integration tests: endpoints via mock router ────────────────────

    #[tokio::test]
    async fn test_get_status_unknown_project() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/projects/nonexistent/sharing"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_status_ok() {
        let (app, slug) = test_app_with_project().await;
        let uri = format!("/api/projects/{}/sharing", slug);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["enabled"], false); // default policy is disabled
        assert!(json["policy"].is_object());
    }

    #[tokio::test]
    async fn test_enable_disable_sharing() {
        let (app, slug) = test_app_with_project().await;

        // Enable
        let uri = format!("/api/projects/{}/sharing/enable", slug);
        let resp = app
            .clone()
            .oneshot(auth_post(&uri, serde_json::json!({})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["enabled"], true);

        // Disable
        let uri = format!("/api/projects/{}/sharing/disable", slug);
        let resp = app
            .oneshot(auth_post(&uri, serde_json::json!({})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["enabled"], false);
    }

    #[tokio::test]
    async fn test_get_policy() {
        let (app, slug) = test_app_with_project().await;
        let uri = format!("/api/projects/{}/sharing/policy", slug);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        // Default policy values
        assert_eq!(json["mode"], "manual");
        assert_eq!(json["l3_scan_enabled"], true);
    }

    #[tokio::test]
    async fn test_set_policy_valid() {
        let (app, slug) = test_app_with_project().await;
        let uri = format!("/api/projects/{}/sharing/policy", slug);
        let body = serde_json::json!({
            "mode": "auto",
            "min_shareability_score": 0.8
        });
        let resp = app.oneshot(auth_put(&uri, body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_set_policy_invalid_score() {
        let (app, slug) = test_app_with_project().await;
        let uri = format!("/api/projects/{}/sharing/policy", slug);
        let body = serde_json::json!({
            "min_shareability_score": 1.5
        });
        let resp = app.oneshot(auth_put(&uri, body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_set_policy_negative_score() {
        let (app, slug) = test_app_with_project().await;
        let uri = format!("/api/projects/{}/sharing/policy", slug);
        let body = serde_json::json!({
            "min_shareability_score": -0.1
        });
        let resp = app.oneshot(auth_put(&uri, body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_sharing_history_empty() {
        let (app, slug) = test_app_with_project().await;
        let uri = format!("/api/projects/{}/sharing/history", slug);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_sharing_preview_empty() {
        let (app, slug) = test_app_with_project().await;
        let uri = format!("/api/projects/{}/sharing/preview", slug);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_sharing_suggest_empty() {
        let (app, slug) = test_app_with_project().await;
        let uri = format!("/api/projects/{}/sharing/suggest", slug);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_tombstones_empty() {
        let (app, slug) = test_app_with_project().await;
        let uri = format!("/api/projects/{}/sharing/tombstones", slug);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_last_report_ok() {
        let (app, slug) = test_app_with_project().await;
        let uri = format!("/api/projects/{}/sharing/last-report", slug);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["stats"].is_object());
        assert!(json["generated_at"].is_string());
    }

    #[tokio::test]
    async fn test_retract_missing_params() {
        let (app, slug) = test_app_with_project().await;
        let uri = format!("/api/projects/{}/sharing/retract", slug);
        let body = serde_json::json!({}); // Neither note_id nor content_hash
        let resp = app.oneshot(auth_post(&uri, body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_retract_with_content_hash() {
        let (app, slug) = test_app_with_project().await;
        let uri = format!("/api/projects/{}/sharing/retract", slug);
        let body = serde_json::json!({
            "content_hash": "sha256:deadbeef",
            "reason": "User requested deletion"
        });
        let resp = app.oneshot(auth_post(&uri, body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["retracted"], true);
        assert_eq!(json["content_hash"], "sha256:deadbeef");
        assert_eq!(json["tombstone_persisted"], true);
        assert_eq!(json["event_recorded"], true);
    }

    #[tokio::test]
    async fn test_retract_with_note_id() {
        let (app, slug) = test_app_with_project().await;
        let note_id = Uuid::new_v4();
        let uri = format!("/api/projects/{}/sharing/retract", slug);
        let body = serde_json::json!({
            "note_id": note_id.to_string()
        });
        let resp = app.oneshot(auth_post(&uri, body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["retracted"], true);
        assert_eq!(json["content_hash"], format!("note:{}", note_id));
    }

    #[tokio::test]
    async fn test_set_consent() {
        let (app, _slug) = test_app_with_project().await;
        let note_id = Uuid::new_v4();
        let uri = format!("/api/notes/{}/sharing/consent", note_id);
        let body = serde_json::json!({
            "consent": "explicit_allow"
        });
        let resp = app.oneshot(auth_put(&uri, body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["note_id"], note_id.to_string());
        assert_eq!(json["updated"], true);
    }

    #[tokio::test]
    async fn test_sharing_requires_auth() {
        let app = test_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/projects/test/sharing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}
