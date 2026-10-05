//! Provider settings API: instances, project consent, pilot/executor roles,
//! model aliases and policy (decisions A15, A19, A24, A25, A28, A32).
//!
//! Every mutation is reserved to a human session: `auth::middleware` refuses an
//! `agent_session` token on these paths (A25) and each handler checks again, so
//! the rule survives a route being moved. No body carries a secret: instances
//! name a credential reference, and a body with any other field is refused.
//!
//! Instances are stored (Neo4j `LlmSetting` documents) and listed; they cannot
//! open a session yet: the resolver only knows the built-in instance until the
//! nexus registry and the native harness land, and the security gate (A32)
//! refuses to create one on a server running without authentication.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::handlers::{AppError, OrchestratorState};
use crate::auth::jwt::Claims;
use crate::chat::provider::endpoint_guard::{validate_endpoint, EndpointPolicy};
use crate::chat::provider::listing::{HealthEntry, ModelEntry, ProviderEntry};
use crate::chat::provider::resolver::CLAUDE_CODE;
use crate::chat::provider::settings::{
    self as st, ConsentRecord, InstanceDraft, InstancePatch, InstanceRecord, ModelAlias,
    ModelPolicy, RoleAssignments, SettingsError, ALIASES_KEY, CONSENT_PREFIX, GLOBAL,
    INSTANCE_PREFIX, POLICY_KEY, ROLES_KEY,
};
use crate::neo4j::GraphStore;
use std::sync::Arc;

fn require_human(state: &OrchestratorState, claims: &Claims) -> Result<(), AppError> {
    if state.auth_config.is_none() || claims.is_human() {
        Ok(())
    } else {
        Err(AppError::Forbidden(
            "provider settings can only be changed by a signed-in user".to_string(),
        ))
    }
}

/// The security gate (A32): a third-party instance is refused while session
/// tokens cannot be bound to a signing key, i.e. while authentication is off.
fn security_gate(state: &OrchestratorState) -> Result<(), AppError> {
    if state.auth_config.is_some() {
        Ok(())
    } else {
        Err(AppError::Conflict(
            "security_gate_closed: third-party providers need authentication enabled (bound session tokens)"
                .to_string(),
        ))
    }
}

fn map_settings_error(e: SettingsError) -> AppError {
    match e {
        SettingsError::Builtin => AppError::Forbidden(e.to_string()),
        SettingsError::UnknownInstance(_) => AppError::NotFound(e.to_string()),
        SettingsError::Invalid(_) | SettingsError::Endpoint(_) => {
            AppError::BadRequest(e.to_string())
        }
    }
}

fn graph(state: &OrchestratorState) -> Arc<dyn GraphStore> {
    state.orchestrator.neo4j_arc()
}

fn parse<T: serde::de::DeserializeOwned>(raw: &str) -> Option<T> {
    serde_json::from_str(raw).ok()
}

/// Every stored instance.
pub async fn load_instances(graph: &dyn GraphStore) -> Result<Vec<InstanceRecord>, AppError> {
    Ok(graph
        .list_llm_settings(GLOBAL, INSTANCE_PREFIX)
        .await
        .map_err(AppError::Internal)?
        .iter()
        .filter_map(|(_, v)| parse::<InstanceRecord>(v))
        .collect())
}

async fn load_aliases(graph: &dyn GraphStore) -> Result<Vec<ModelAlias>, AppError> {
    Ok(graph
        .get_llm_setting(GLOBAL, ALIASES_KEY)
        .await
        .map_err(AppError::Internal)?
        .and_then(|v| parse(&v))
        .unwrap_or_default())
}

async fn load_consents(graph: &dyn GraphStore, slug: &str) -> Result<Vec<ConsentRecord>, AppError> {
    Ok(graph
        .list_llm_settings(&st::project_scope(slug), CONSENT_PREFIX)
        .await
        .map_err(AppError::Internal)?
        .iter()
        .filter_map(|(_, v)| parse::<ConsentRecord>(v))
        .collect())
}

async fn instance_exists(graph: &dyn GraphStore, id: &str) -> Result<bool, AppError> {
    Ok(id == CLAUDE_CODE
        || graph
            .get_llm_setting(GLOBAL, &format!("{INSTANCE_PREFIX}{id}"))
            .await
            .map_err(AppError::Internal)?
            .is_some())
}

/// The stored instances as listing entries, consent evaluated for a project.
pub async fn stored_entries(
    graph: &dyn GraphStore,
    project_slug: Option<&str>,
) -> Result<Vec<ProviderEntry>, AppError> {
    let consents = match project_slug {
        Some(slug) => Some(load_consents(graph, slug).await?),
        None => None,
    };
    let aliases = load_aliases(graph).await?;
    Ok(load_instances(graph)
        .await?
        .into_iter()
        .map(|i| {
            let allowed = consents.as_ref().map(|cs| {
                cs.iter()
                    .any(|c| c.provider_id == i.id && c.origin == i.origin)
            });
            let models = i
                .default_model
                .iter()
                .map(|m| {
                    let alias = aliases
                        .iter()
                        .find(|a| a.provider == i.id && &a.model == m)
                        .map(|a| a.alias.clone());
                    ModelEntry::new(m.clone(), alias, &nexus_claude::agent::Capabilities::none())
                })
                .collect();
            ProviderEntry {
                id: i.id,
                kind: "openai_compatible",
                label: i.label,
                builtin: false,
                is_default: false,
                allowed_for_project: allowed,
                endpoint_origin: Some(i.origin),
                credential: i.credential_ref,
                health: HealthEntry::unknown(),
                models,
            }
        })
        .collect())
}

fn instance_view(i: &InstanceRecord) -> Value {
    json!({
        "id": i.id, "kind": i.kind, "preset": i.preset, "label": i.label,
        "base_url": i.base_url, "origin": i.origin, "default_model": i.default_model,
        "cost_source": i.cost_source, "credential_ref": i.credential_ref, "builtin": false,
    })
}

// ============================================================================
// Instances
// ============================================================================

/// POST /api/chat/providers/test — try a draft BEFORE saving it. Always 200:
/// the verdict is in the body.
pub async fn test_provider(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Json(mut draft): Json<InstanceDraft>,
) -> Result<Json<Value>, AppError> {
    require_human(&state, &claims)?;
    draft.id.get_or_insert_with(|| "draft".to_string());
    let policy = EndpointPolicy::default();
    let verdict =
        |state: &str, code: &str| json!({"ok": false, "health": {"state": state, "code": code}});
    let record = match st::record_from_draft(&draft, &policy) {
        Ok(r) => r,
        Err(SettingsError::Endpoint(e)) => return Ok(Json(verdict("unreachable", e.code()))),
        Err(e) => return Err(map_settings_error(e)),
    };
    if let Err(e) = validate_endpoint(&record.base_url, &policy).await {
        return Ok(Json(verdict("unreachable", e.code())));
    }
    let report = match crate::chat::provider::native_factory::probe_instance(
        &record,
        Some(state.vault.clone()),
        record.default_model.as_deref(),
    )
    .await
    {
        Ok(report) => report,
        // A credential problem (locked vault, no grant) is a verdict, not a 500.
        Err(e) => {
            let failure = crate::chat::provider::errors::open_failure(&e, Some(&record.id));
            return Ok(Json(verdict(
                if failure.code == "credentials_locked" || failure.code == "auth_required" {
                    "auth_required"
                } else {
                    "unknown"
                },
                failure.code,
            )));
        }
    };
    let health = HealthEntry::from_nexus(&report.health);
    let probe_failure = report
        .probe_error
        .as_ref()
        .map(|e| crate::chat::provider::errors::open_failure(e, Some(&record.id)).code);
    let tools_ok = report.tools == Some(true);
    let ok = health.state == "ok" && tools_ok;
    let mut body = json!({
        "ok": ok,
        "health": {
            "state": health.state,
            "code": health.code.or(probe_failure).or((report.tools == Some(false)).then_some("model_no_tools")),
        },
        "models": report.models.iter().map(|m| json!({ "id": m })).collect::<Vec<_>>(),
    });
    if report.tools.is_some() || report.context_window.is_some() {
        body["probe"] = json!({
            "tools": tools_ok,
            "context_window": report.context_window,
        });
    }
    Ok(Json(body))
}

/// POST /api/chat/providers — create an instance.
pub async fn create_provider(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Json(draft): Json<InstanceDraft>,
) -> Result<(StatusCode, Json<Value>), AppError> {
    require_human(&state, &claims)?;
    security_gate(&state)?;
    let policy = EndpointPolicy::default();
    let record = st::record_from_draft(&draft, &policy).map_err(map_settings_error)?;
    validate_endpoint(&record.base_url, &policy)
        .await
        .map_err(|e| AppError::BadRequest(format!("endpoint refused: {e}")))?;
    let graph = graph(&state);
    let key = format!("{INSTANCE_PREFIX}{}", record.id);
    if graph
        .get_llm_setting(GLOBAL, &key)
        .await
        .map_err(AppError::Internal)?
        .is_some()
    {
        return Err(AppError::Conflict(format!(
            "provider '{}' already exists",
            record.id
        )));
    }
    let body = serde_json::to_string(&record).map_err(|e| AppError::Internal(e.into()))?;
    graph
        .put_llm_setting(GLOBAL, &key, &body)
        .await
        .map_err(AppError::Internal)?;
    Ok((StatusCode::CREATED, Json(instance_view(&record))))
}

/// PUT|PATCH /api/chat/providers/{id} — change an instance (never its id).
pub async fn update_provider(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
    Json(patch): Json<InstancePatch>,
) -> Result<Json<Value>, AppError> {
    require_human(&state, &claims)?;
    if id == CLAUDE_CODE {
        return Err(map_settings_error(SettingsError::Builtin));
    }
    let graph = graph(&state);
    let key = format!("{INSTANCE_PREFIX}{id}");
    let old: InstanceRecord = graph
        .get_llm_setting(GLOBAL, &key)
        .await
        .map_err(AppError::Internal)?
        .and_then(|v| parse(&v))
        .ok_or_else(|| AppError::NotFound(format!("unknown provider instance '{id}'")))?;
    let policy = EndpointPolicy::default();
    let (next, origin_changed) =
        st::apply_patch(&old, &patch, &policy).map_err(map_settings_error)?;
    if patch.base_url.is_some() {
        validate_endpoint(&next.base_url, &policy)
            .await
            .map_err(|e| AppError::BadRequest(format!("endpoint refused: {e}")))?;
    }
    let body = serde_json::to_string(&next).map_err(|e| AppError::Internal(e.into()))?;
    graph
        .put_llm_setting(GLOBAL, &key, &body)
        .await
        .map_err(AppError::Internal)?;
    let mut view = instance_view(&next);
    // Consents are tied to the origin: a new origin leaves them stored but no
    // longer valid, and they are reported as such (A28).
    view["consents_invalidated"] = json!(origin_changed);
    Ok(Json(view))
}

/// DELETE /api/chat/providers/{id}.
pub async fn delete_provider(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
) -> Result<StatusCode, AppError> {
    require_human(&state, &claims)?;
    if id == CLAUDE_CODE {
        return Err(map_settings_error(SettingsError::Builtin));
    }
    let removed = graph(&state)
        .delete_llm_setting(GLOBAL, &format!("{INSTANCE_PREFIX}{id}"))
        .await
        .map_err(AppError::Internal)?;
    if removed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!(
            "unknown provider instance '{id}'"
        )))
    }
}

/// GET /api/chat/providers/{id}/status — health of one instance.
pub async fn provider_status(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<Json<HealthEntry>, AppError> {
    use nexus_claude::agent::AgentProvider;
    if id == CLAUDE_CODE {
        let provider = nexus_claude::providers::claude_code::ClaudeCodeProvider::new(
            nexus_claude::providers::claude_code::ClaudeCodeConfig::default(),
        );
        return Ok(Json(HealthEntry::from_nexus(&provider.health().await)));
    }
    if !instance_exists(graph(&state).as_ref(), &id).await? {
        return Err(AppError::NotFound(format!(
            "unknown provider instance '{id}'"
        )));
    }
    Ok(Json(HealthEntry::unknown()))
}

/// GET /api/chat/providers/{id}/models — models of one instance.
pub async fn provider_models(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, AppError> {
    let graph = graph(&state);
    if id == CLAUDE_CODE {
        let model = state
            .chat_manager
            .as_ref()
            .map(|m| m.resolve_model(None))
            .unwrap_or_else(|| crate::chat::ChatConfig::from_env().default_model);
        return Ok(Json(json!([{ "id": model }])));
    }
    let record: InstanceRecord = graph
        .get_llm_setting(GLOBAL, &format!("{INSTANCE_PREFIX}{id}"))
        .await
        .map_err(AppError::Internal)?
        .and_then(|v| parse(&v))
        .ok_or_else(|| AppError::NotFound(format!("unknown provider instance '{id}'")))?;
    Ok(Json(json!(record
        .default_model
        .iter()
        .map(|m| json!({ "id": m }))
        .collect::<Vec<_>>())))
}

// ============================================================================
// Consent
// ============================================================================

/// Body of `PUT .../llm-consent`: the instance and the origin being allowed.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsentBody {
    /// Instance (taken from the path when the route carries it).
    #[serde(default)]
    pub provider_id: Option<String>,
    /// Origin the human is allowing: it must be the instance's current origin.
    pub origin: String,
}

/// GET /api/projects/{slug}/llm-consent[s].
pub async fn list_consents(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<Vec<st::ConsentView>>, AppError> {
    let graph = graph(&state);
    let instances = load_instances(graph.as_ref()).await?;
    let rows = load_consents(graph.as_ref(), &slug)
        .await?
        .iter()
        .map(|c| {
            let current = instances
                .iter()
                .find(|i| i.id == c.provider_id)
                .map(|i| i.origin.as_str());
            st::consent_view(c, current)
        })
        .collect();
    Ok(Json(rows))
}

async fn put_consent(
    state: &OrchestratorState,
    claims: &Claims,
    slug: &str,
    provider_id: &str,
    origin: &str,
) -> Result<Json<st::ConsentView>, AppError> {
    require_human(state, claims)?;
    if provider_id == CLAUDE_CODE {
        return Err(AppError::BadRequest(
            "claude-code is always allowed: it needs no consent".to_string(),
        ));
    }
    let graph = graph(state);
    let instance: InstanceRecord = graph
        .get_llm_setting(GLOBAL, &format!("{INSTANCE_PREFIX}{provider_id}"))
        .await
        .map_err(AppError::Internal)?
        .and_then(|v| parse(&v))
        .ok_or_else(|| AppError::NotFound(format!("unknown provider instance '{provider_id}'")))?;
    // The human saw an origin on screen: it must still be the instance's.
    if instance.origin != origin {
        return Err(AppError::Conflict(
            "origin_mismatch: the instance no longer points at the origin you were shown"
                .to_string(),
        ));
    }
    let record = ConsentRecord {
        provider_id: provider_id.to_string(),
        origin: instance.origin.clone(),
        consented_by: claims.email.clone(),
        consented_at: chrono::Utc::now().to_rfc3339(),
    };
    let body = serde_json::to_string(&record).map_err(|e| AppError::Internal(e.into()))?;
    graph
        .put_llm_setting(
            &st::project_scope(slug),
            &format!("{CONSENT_PREFIX}{provider_id}"),
            &body,
        )
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(st::consent_view(&record, Some(&instance.origin))))
}

/// PUT /api/projects/{slug}/llm-consent — body `{ provider_id, origin }`.
pub async fn allow_consent(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(slug): Path<String>,
    Json(body): Json<ConsentBody>,
) -> Result<Json<st::ConsentView>, AppError> {
    let provider_id = body
        .provider_id
        .clone()
        .ok_or_else(|| AppError::BadRequest("provider_id is required".to_string()))?;
    put_consent(&state, &claims, &slug, &provider_id, &body.origin).await
}

/// PUT /api/projects/{slug}/llm-consents/{provider_id} — body `{ origin }`.
pub async fn allow_consent_for(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path((slug, provider_id)): Path<(String, String)>,
    Json(body): Json<ConsentBody>,
) -> Result<Json<st::ConsentView>, AppError> {
    put_consent(&state, &claims, &slug, &provider_id, &body.origin).await
}

/// DELETE /api/projects/{slug}/llm-consent[s]/{provider_id}.
pub async fn revoke_consent(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path((slug, provider_id)): Path<(String, String)>,
) -> Result<StatusCode, AppError> {
    require_human(&state, &claims)?;
    let removed = graph(&state)
        .delete_llm_setting(
            &st::project_scope(&slug),
            &format!("{CONSENT_PREFIX}{provider_id}"),
        )
        .await
        .map_err(AppError::Internal)?;
    if removed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound("no consent to revoke".to_string()))
    }
}

// ============================================================================
// Roles, aliases, policy
// ============================================================================

async fn read_roles(graph: &dyn GraphStore, scope: &str) -> Result<RoleAssignments, AppError> {
    Ok(graph
        .get_llm_setting(scope, ROLES_KEY)
        .await
        .map_err(AppError::Internal)?
        .and_then(|v| parse(&v))
        .unwrap_or_default())
}

async fn write_roles(
    state: &OrchestratorState,
    claims: &Claims,
    scope: &str,
    roles: RoleAssignments,
) -> Result<Json<RoleAssignments>, AppError> {
    require_human(state, claims)?;
    let graph = graph(state);
    let stored: Vec<String> = load_instances(graph.as_ref())
        .await?
        .into_iter()
        .map(|i| i.id)
        .collect();
    st::validate_roles(&roles, &|id| {
        id == CLAUDE_CODE || stored.iter().any(|s| s == id)
    })
    .map_err(map_settings_error)?;
    let body = serde_json::to_string(&roles).map_err(|e| AppError::Internal(e.into()))?;
    graph
        .put_llm_setting(scope, ROLES_KEY, &body)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(roles))
}

/// GET /api/chat/roles.
pub async fn get_roles(
    State(state): State<OrchestratorState>,
) -> Result<Json<RoleAssignments>, AppError> {
    Ok(Json(read_roles(graph(&state).as_ref(), GLOBAL).await?))
}

/// PUT /api/chat/roles.
pub async fn put_roles(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Json(roles): Json<RoleAssignments>,
) -> Result<Json<RoleAssignments>, AppError> {
    write_roles(&state, &claims, GLOBAL, roles).await
}

/// GET /api/projects/{slug}/llm-roles (an absent role inherits the global one).
pub async fn get_project_roles(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<RoleAssignments>, AppError> {
    Ok(Json(
        read_roles(graph(&state).as_ref(), &st::project_scope(&slug)).await?,
    ))
}

/// PUT /api/projects/{slug}/llm-roles.
pub async fn put_project_roles(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(slug): Path<String>,
    Json(roles): Json<RoleAssignments>,
) -> Result<Json<RoleAssignments>, AppError> {
    write_roles(&state, &claims, &st::project_scope(&slug), roles).await
}

/// GET /api/chat/model-aliases.
pub async fn get_aliases(
    State(state): State<OrchestratorState>,
) -> Result<Json<Vec<ModelAlias>>, AppError> {
    Ok(Json(load_aliases(graph(&state).as_ref()).await?))
}

/// PUT /api/chat/model-aliases.
pub async fn put_aliases(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Json(aliases): Json<Vec<ModelAlias>>,
) -> Result<Json<Vec<ModelAlias>>, AppError> {
    require_human(&state, &claims)?;
    let graph = graph(&state);
    let stored: Vec<String> = load_instances(graph.as_ref())
        .await?
        .into_iter()
        .map(|i| i.id)
        .collect();
    st::validate_aliases(&aliases, &|id| {
        id == CLAUDE_CODE || stored.iter().any(|s| s == id)
    })
    .map_err(map_settings_error)?;
    let body = serde_json::to_string(&aliases).map_err(|e| AppError::Internal(e.into()))?;
    graph
        .put_llm_setting(GLOBAL, ALIASES_KEY, &body)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(aliases))
}

/// GET /api/chat/model-policy (ships as `off`).
pub async fn get_policy(
    State(state): State<OrchestratorState>,
) -> Result<Json<ModelPolicy>, AppError> {
    Ok(Json(
        graph(&state)
            .get_llm_setting(GLOBAL, POLICY_KEY)
            .await
            .map_err(AppError::Internal)?
            .and_then(|v| parse(&v))
            .unwrap_or_default(),
    ))
}

/// PUT /api/chat/model-policy.
pub async fn put_policy(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Json(policy): Json<ModelPolicy>,
) -> Result<Json<ModelPolicy>, AppError> {
    require_human(&state, &claims)?;
    let graph = graph(&state);
    let aliases = load_aliases(graph.as_ref()).await?;
    st::validate_policy(&policy, &|a| aliases.iter().any(|x| x.alias == a))
        .map_err(map_settings_error)?;
    let body = serde_json::to_string(&policy).map_err(|e| AppError::Internal(e.into()))?;
    graph
        .put_llm_setting(GLOBAL, POLICY_KEY, &body)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(policy))
}

#[allow(dead_code)]
#[derive(Serialize)]
struct _Unused;
