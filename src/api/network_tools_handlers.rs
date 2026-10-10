//! Network tools of a native session (task P6): what a PROJECT consents to and which search
//! engines the server knows. The decisions are those of B40 (#596,
//! [`crate::chat::provider::nexus_tools`]); this module only lets a person write the documents
//! that module already reads, under the same keys, through [`GraphConsents`]' own reading. There
//! is no second consent mechanism.
//!
//! * `GET    /api/projects/{slug}/network-tools` — the project's consented origins, its browser
//!   authorisation, and every search engine with whether its origin is consented here.
//! * `PUT    /api/projects/{slug}/network-tools/origins` `{origin}` — consent to an origin
//!   (`tool_origin:<origin>`, scope `project:<slug>`). A URL is reduced to its origin.
//! * `DELETE /api/projects/{slug}/network-tools/origins?origin=<origin>` — revoke it.
//! * `PUT    /api/projects/{slug}/network-tools/browser` `{allowed}` — allow or deny the browser
//!   (`browser`, scope `project:<slug>`). Attaching the browser itself is N23: the authorisation
//!   is stored and read, the browser is attached only once its executable is configured.
//! * `GET    /api/chat/search-engines[?project=<slug>]` — the search engines (tool providers).
//! * `POST   /api/chat/search-engines` `{id, engine, base_url?, credential_ref}` — declare one
//!   (`search_provider:<id>`, scope `global`). The key is a vault REFERENCE; a body that carries
//!   a value is refused. The key is then granted with `POST /api/vault/grants` and the scope
//!   `{"kind":"provider","value":"tool:<id>"}` (the grant `Provider("tool:<id>")` of A26).
//! * `DELETE /api/chat/search-engines/{id}` — remove it, and revoke its provider grants.
//!
//! Every mutation is reserved to a human session: `auth::middleware` refuses an `agent_session`
//! token on these paths and each handler checks again. A refused body answers 400 with a typed
//! `code` and never echoes the value that was sent.
//!
//! [`GraphConsents`]: crate::chat::provider::nexus_tools::GraphConsents

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Extension, Json,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::handlers::{AppError, OrchestratorState};
use super::provider_handlers::require_human;
use crate::auth::jwt::Claims;
use crate::chat::provider::endpoint_guard::origin_of;
use crate::chat::provider::errors::OpenFailure;
use crate::chat::provider::nexus_tools::{
    BrowserAuthorization, SearchProviderRecord, ToolOriginConsent, BROWSER_KEY,
    SEARCH_PROVIDER_PREFIX, TOOL_ORIGIN_PREFIX, TOOL_PROVIDER_PREFIX,
};
use crate::chat::provider::settings::{project_scope, valid_id, GLOBAL};
use crate::vault::{GrantScope, SecretSelector};

/// Field names that would carry a secret VALUE. A body naming one is refused before anything
/// else is read: the key of an engine lives in the vault, the body names it.
const SECRET_FIELDS: &[&str] = &[
    "key", "api_key", "apikey", "token", "secret", "value", "password",
];
/// The engines `nexus-tools` knows (`--search-engine brave:<VAR>` / `searxng:<url>`).
const ENGINES: &[&str] = &["brave", "searxng"];
/// An origin or URL longer than this is not a consent anyone reads.
const MAX_URL_LEN: usize = 2048;

/// A refusal with a stable `code`: `{error, code, retryable: false}`.
fn refused(status: StatusCode, code: &'static str, message: impl Into<String>) -> AppError {
    AppError::Provider(Box::new(OpenFailure {
        status: status.as_u16(),
        code,
        message: message.into(),
        provider_id: None,
        action: None,
        retryable: false,
        retry_after_ms: None,
        fallbacks: None,
    }))
}

fn bad(code: &'static str, message: impl Into<String>) -> AppError {
    refused(StatusCode::BAD_REQUEST, code, message)
}

/// The project must exist: a consent to a project nobody can open is a typo kept forever.
async fn require_project(state: &OrchestratorState, slug: &str) -> Result<(), AppError> {
    match state
        .orchestrator
        .neo4j()
        .get_project_by_slug(slug)
        .await
        .map_err(AppError::Internal)?
    {
        Some(_) => Ok(()),
        None => Err(refused(
            StatusCode::NOT_FOUND,
            "project_not_found",
            "no project has this slug",
        )),
    }
}

/// The origin a network tool is judged on: `http` or `https`, a host, no credentials. A URL is
/// reduced to its origin (the consent is per origin, A28), exactly as the gate reduces the URL
/// of a call ([`origin_of`]).
pub(crate) fn tool_origin(raw: &str) -> Result<String, AppError> {
    let invalid = || {
        bad(
            "invalid_tool_origin",
            "an origin is http(s)://host[:port], without credentials",
        )
    };
    let raw = raw.trim();
    if raw.is_empty() || raw.len() > MAX_URL_LEN {
        return Err(invalid());
    }
    let parsed = url::Url::parse(raw).map_err(|_| invalid())?;
    if !matches!(parsed.scheme(), "http" | "https")
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return Err(invalid());
    }
    origin_of(raw).ok_or_else(invalid)
}

// ---------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------

async fn consented_origins(
    state: &OrchestratorState,
    slug: &str,
) -> Result<Vec<ToolOriginConsent>, AppError> {
    let documents = state
        .orchestrator
        .neo4j()
        .list_llm_settings(&project_scope(slug), TOOL_ORIGIN_PREFIX)
        .await
        .map_err(AppError::Internal)?;
    // The reading of `GraphConsents::any_origin_consented`: a document counts only if it
    // decodes and names the origin of its key.
    let mut origins: Vec<ToolOriginConsent> = documents
        .iter()
        .filter_map(|(key, raw)| {
            let consent = serde_json::from_str::<ToolOriginConsent>(raw).ok()?;
            (key.strip_prefix(TOOL_ORIGIN_PREFIX).unwrap_or(key) == consent.origin)
                .then_some(consent)
        })
        .collect();
    origins.sort_by(|a, b| a.origin.cmp(&b.origin));
    Ok(origins)
}

async fn browser_of(
    state: &OrchestratorState,
    slug: &str,
) -> Result<Option<BrowserAuthorization>, AppError> {
    Ok(state
        .orchestrator
        .neo4j()
        .get_llm_setting(&project_scope(slug), BROWSER_KEY)
        .await
        .map_err(AppError::Internal)?
        .and_then(|raw| serde_json::from_str(&raw).ok()))
}

async fn engines(state: &OrchestratorState) -> Result<Vec<SearchProviderRecord>, AppError> {
    let mut engines: Vec<SearchProviderRecord> = state
        .orchestrator
        .neo4j()
        .list_llm_settings(GLOBAL, SEARCH_PROVIDER_PREFIX)
        .await
        .map_err(AppError::Internal)?
        .iter()
        .filter_map(|(_, raw)| serde_json::from_str(raw).ok())
        .collect();
    engines.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(engines)
}

/// A search engine as shown: the stored record (a reference, never a key), the origin a project
/// consents to, the grant id, whether a live grant covers its key, and (for a project) whether
/// its origin is consented there.
#[derive(Debug, Serialize)]
pub struct SearchEngineView {
    #[serde(flatten)]
    record: SearchProviderRecord,
    origin: Option<String>,
    grant_id: String,
    key_granted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    origin_consented: Option<bool>,
}

fn engine_view(
    state: &OrchestratorState,
    record: SearchProviderRecord,
    consented: Option<&[ToolOriginConsent]>,
) -> SearchEngineView {
    let grant_id = record.grant_id();
    let origin = record.origin();
    let key_granted = match record.credential_ref.strip_prefix("vault:") {
        None => true,
        Some(name) => state.vault.grants(Utc::now()).iter().any(|g| {
            matches!(&g.scope, GrantScope::Provider(id) if *id == grant_id)
                && matches!(&g.secrets, SecretSelector::Names(names) if names.contains(name))
        }),
    };
    let origin_consented = consented.map(|consented| {
        origin
            .as_deref()
            .is_some_and(|o| consented.iter().any(|c| c.origin == o))
    });
    SearchEngineView {
        record,
        origin,
        grant_id,
        key_granted,
        origin_consented,
    }
}

/// The project's network-tool settings.
#[derive(Debug, Serialize)]
pub struct NetworkToolsView {
    project: String,
    origins: Vec<ToolOriginConsent>,
    browser: BrowserView,
    search_engines: Vec<SearchEngineView>,
}

/// The project's browser authorisation.
#[derive(Debug, Serialize)]
pub struct BrowserView {
    allowed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    authorized_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    authorized_at: Option<String>,
}

impl From<Option<BrowserAuthorization>> for BrowserView {
    fn from(value: Option<BrowserAuthorization>) -> Self {
        match value {
            Some(a) => Self {
                allowed: true,
                authorized_by: Some(a.authorized_by),
                authorized_at: Some(a.authorized_at),
            },
            None => Self {
                allowed: false,
                authorized_by: None,
                authorized_at: None,
            },
        }
    }
}

/// GET /api/projects/{slug}/network-tools
pub async fn get_network_tools(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<NetworkToolsView>, AppError> {
    require_project(&state, &slug).await?;
    let origins = consented_origins(&state, &slug).await?;
    let browser = browser_of(&state, &slug).await?.into();
    let search_engines = engines(&state)
        .await?
        .into_iter()
        .map(|record| engine_view(&state, record, Some(&origins)))
        .collect();
    Ok(Json(NetworkToolsView {
        project: slug,
        origins,
        browser,
        search_engines,
    }))
}

// ---------------------------------------------------------------------------
// Origins
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginBody {
    origin: String,
}

#[derive(Debug, Deserialize)]
pub struct OriginQuery {
    origin: Option<String>,
}

fn origin_body(body: &Value) -> Result<OriginBody, AppError> {
    serde_json::from_value(body.clone()).map_err(|_| {
        bad(
            "invalid_tool_origin",
            "the body is {\"origin\": \"https://host\"}",
        )
    })
}

/// PUT /api/projects/{slug}/network-tools/origins — consent (human only). Idempotent: consenting
/// again records who and when.
pub async fn put_origin(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(slug): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<ToolOriginConsent>, AppError> {
    require_human(&state, &claims)?;
    let origin = tool_origin(&origin_body(&body)?.origin)?;
    require_project(&state, &slug).await?;
    let consent = ToolOriginConsent {
        origin: origin.clone(),
        consented_by: claims.email.clone(),
        consented_at: Utc::now().to_rfc3339(),
    };
    let document = serde_json::to_string(&consent).map_err(|e| AppError::Internal(e.into()))?;
    state
        .orchestrator
        .neo4j()
        .put_llm_setting(
            &project_scope(&slug),
            &format!("{TOOL_ORIGIN_PREFIX}{origin}"),
            &document,
        )
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(consent))
}

/// DELETE /api/projects/{slug}/network-tools/origins?origin=… — revoke (human only). A running
/// session reads the consent at each call: its next request to that origin is refused.
pub async fn delete_origin(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(slug): Path<String>,
    Query(query): Query<OriginQuery>,
) -> Result<StatusCode, AppError> {
    require_human(&state, &claims)?;
    let origin = tool_origin(query.origin.as_deref().unwrap_or_default())?;
    let removed = state
        .orchestrator
        .neo4j()
        .delete_llm_setting(
            &project_scope(&slug),
            &format!("{TOOL_ORIGIN_PREFIX}{origin}"),
        )
        .await
        .map_err(AppError::Internal)?;
    if removed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(refused(
            StatusCode::NOT_FOUND,
            "tool_origin_not_found",
            "the project has no consent to this origin",
        ))
    }
}

// ---------------------------------------------------------------------------
// Browser
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BrowserBody {
    allowed: bool,
}

/// PUT /api/projects/{slug}/network-tools/browser — `{allowed}` (human only).
pub async fn put_browser(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(slug): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<BrowserView>, AppError> {
    require_human(&state, &claims)?;
    let body: BrowserBody = serde_json::from_value(body).map_err(|_| {
        bad(
            "invalid_browser_setting",
            "the body is {\"allowed\": true|false}",
        )
    })?;
    require_project(&state, &slug).await?;
    let graph = state.orchestrator.neo4j();
    let scope = project_scope(&slug);
    if body.allowed {
        let authorization = BrowserAuthorization {
            authorized_by: claims.email.clone(),
            authorized_at: Utc::now().to_rfc3339(),
        };
        let document =
            serde_json::to_string(&authorization).map_err(|e| AppError::Internal(e.into()))?;
        graph
            .put_llm_setting(&scope, BROWSER_KEY, &document)
            .await
            .map_err(AppError::Internal)?;
        Ok(Json(Some(authorization).into()))
    } else {
        graph
            .delete_llm_setting(&scope, BROWSER_KEY)
            .await
            .map_err(AppError::Internal)?;
        Ok(Json(None.into()))
    }
}

// ---------------------------------------------------------------------------
// Search engines
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EngineDraft {
    id: String,
    engine: String,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    credential_ref: Option<String>,
}

/// Checks a draft and makes the record. Never echoes a value of the body.
pub(crate) fn engine_record(body: &Value) -> Result<SearchProviderRecord, AppError> {
    let Some(fields) = body.as_object() else {
        return Err(bad("invalid_search_engine", "the body is a JSON object"));
    };
    if fields
        .keys()
        .any(|k| SECRET_FIELDS.contains(&k.to_ascii_lowercase().as_str()))
    {
        return Err(bad(
            "secret_value_refused",
            "a search engine names its key by a vault reference (`credential_ref: \"vault:<name>\"`), never by value",
        ));
    }
    let draft: EngineDraft = serde_json::from_value(body.clone()).map_err(|_| {
        bad(
            "invalid_search_engine",
            "the body is {id, engine, base_url?, credential_ref}",
        )
    })?;
    if !valid_id(&draft.id) {
        return Err(bad(
            "invalid_search_engine_id",
            "an id is 1 to 48 lowercase letters, digits or '-', starting with a letter or digit",
        ));
    }
    if !ENGINES.contains(&draft.engine.as_str()) {
        return Err(bad(
            "unknown_search_engine",
            "the engine is `brave` or `searxng`",
        ));
    }
    let credential_ref = draft.credential_ref.unwrap_or_else(|| "none".to_owned());
    let vault_name = credential_ref.strip_prefix("vault:");
    if credential_ref != "none"
        && vault_name.is_none_or(|n| crate::vault::store::validate_name(n).is_err())
    {
        return Err(bad(
            "invalid_credential_ref",
            "credential_ref is `vault:<name>` or `none`",
        ));
    }
    match draft.engine.as_str() {
        "brave" => {
            if draft.base_url.is_some() {
                return Err(bad(
                    "invalid_search_engine",
                    "brave has a fixed endpoint: no base_url",
                ));
            }
            if vault_name.is_none() {
                return Err(bad(
                    "credential_ref_required",
                    "brave needs its key: credential_ref `vault:<name>`",
                ));
            }
        }
        _ => {
            let url = draft.base_url.as_deref().unwrap_or_default();
            tool_origin(url).map_err(|_| {
                bad(
                    "invalid_search_engine_url",
                    "searxng needs base_url, an http(s) URL without credentials",
                )
            })?;
            if vault_name.is_some() {
                return Err(bad(
                    "invalid_credential_ref",
                    "searxng takes no key: credential_ref `none`",
                ));
            }
        }
    }
    Ok(SearchProviderRecord {
        id: draft.id,
        engine: draft.engine,
        base_url: draft.base_url.map(|u| u.trim().to_owned()),
        credential_ref,
    })
}

#[derive(Debug, Deserialize)]
pub struct EnginesQuery {
    project: Option<String>,
}

/// GET /api/chat/search-engines[?project=<slug>]
pub async fn list_search_engines(
    State(state): State<OrchestratorState>,
    Query(query): Query<EnginesQuery>,
) -> Result<Json<Vec<SearchEngineView>>, AppError> {
    let consented = match query.project.as_deref() {
        Some(slug) => Some(consented_origins(&state, slug).await?),
        None => None,
    };
    Ok(Json(
        engines(&state)
            .await?
            .into_iter()
            .map(|record| engine_view(&state, record, consented.as_deref()))
            .collect(),
    ))
}

/// POST /api/chat/search-engines — declare an engine (human only). 409 when the id exists.
pub async fn create_search_engine(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Json(body): Json<Value>,
) -> Result<(StatusCode, Json<SearchEngineView>), AppError> {
    require_human(&state, &claims)?;
    let record = engine_record(&body)?;
    let graph = state.orchestrator.neo4j();
    let key = format!("{SEARCH_PROVIDER_PREFIX}{}", record.id);
    if graph
        .get_llm_setting(GLOBAL, &key)
        .await
        .map_err(AppError::Internal)?
        .is_some()
    {
        return Err(refused(
            StatusCode::CONFLICT,
            "search_engine_exists",
            "a search engine already has this id",
        ));
    }
    let document = serde_json::to_string(&record).map_err(|e| AppError::Internal(e.into()))?;
    graph
        .put_llm_setting(GLOBAL, &key, &document)
        .await
        .map_err(AppError::Internal)?;
    Ok((StatusCode::CREATED, Json(engine_view(&state, record, None))))
}

/// DELETE /api/chat/search-engines/{id} — remove an engine and revoke the grants made to it
/// (human only): an engine declared later under the same id starts without a key.
pub async fn delete_search_engine(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
) -> Result<StatusCode, AppError> {
    require_human(&state, &claims)?;
    let removed = state
        .orchestrator
        .neo4j()
        .delete_llm_setting(GLOBAL, &format!("{SEARCH_PROVIDER_PREFIX}{id}"))
        .await
        .map_err(AppError::Internal)?;
    if !removed {
        return Err(refused(
            StatusCode::NOT_FOUND,
            "search_engine_not_found",
            "no search engine has this id",
        ));
    }
    let grant_id = format!("{TOOL_PROVIDER_PREFIX}{id}");
    let now = Utc::now();
    for grant in state.vault.grants(now) {
        if matches!(&grant.scope, GrantScope::Provider(p) if *p == grant_id) {
            state.vault.revoke(grant.id, now)?;
        }
    }
    Ok(StatusCode::NO_CONTENT)
}

/// The stored search engine a `Provider("tool:<id>")` grant is for, if `grant_id` names one.
pub(crate) async fn search_engine_for_grant(
    graph: &dyn crate::neo4j::GraphStore,
    grant_id: &str,
) -> anyhow::Result<Option<SearchProviderRecord>> {
    let Some(id) = grant_id.strip_prefix(TOOL_PROVIDER_PREFIX) else {
        return Ok(None);
    };
    Ok(graph
        .get_llm_setting(GLOBAL, &format!("{SEARCH_PROVIDER_PREFIX}{id}"))
        .await?
        .and_then(|raw| serde_json::from_str(&raw).ok()))
}

#[cfg(test)]
#[path = "network_tools_tests.rs"]
mod tests;
