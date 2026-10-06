//! Provider settings API: instances, project consent, pilot/executor roles,
//! model aliases and policy (decisions A15, A19, A24, A25, A28, A32).
//!
//! Every mutation is reserved to a human session: `auth::middleware` refuses an
//! `agent_session` token on these paths (A25) and each handler checks again, so
//! the rule survives a route being moved. No body carries a secret: instances
//! name a credential reference, and a body with any other field is refused.
//!
//! Instances are stored (Neo4j `LlmSetting` documents) and listed; a stored
//! instance opens sessions on the agent engine (`ChatManager::provider_for`),
//! once a project has consented to its origin and credential reference. The
//! security gate (A32) refuses to create one on a server running without
//! authentication.

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
use crate::chat::provider::listing::{HealthEntry, ModelEntry, ProviderEntry, RemoteEntry};
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
            let allowed = consents
                .as_ref()
                .map(|cs| cs.iter().any(|c| st::consent_holds(c, &i)));
            // What the provider of this instance declares, before any session: an
            // instance listed with EMPTY capabilities made the interface claim it
            // could not ask for approval ("policy only") when it can.
            let caps = crate::chat::provider::native_factory::declared_capabilities(&i);
            let models = i
                .default_model
                .iter()
                .map(|m| {
                    let alias = aliases
                        .iter()
                        .find(|a| a.provider == i.id && &a.model == m)
                        .map(|a| a.alias.clone());
                    ModelEntry::new(m.clone(), alias, &caps)
                })
                .collect();
            let remote = remote_entry(&i);
            ProviderEntry {
                id: i.id.clone(),
                kind: match i.kind.as_str() {
                    "codex" => "codex",
                    "acp" => "acp",
                    st::KIND_CLAUDE_CODE_REMOTE => "claude_code_remote",
                    _ => "openai_compatible",
                },
                label: i.label,
                builtin: false,
                is_default: false,
                allowed_for_project: allowed,
                endpoint_origin: (!st::is_process_kind(&i.kind)).then(|| i.origin.clone()),
                credential: i.credential_ref,
                health: HealthEntry::unknown(),
                models,
                capabilities: serde_json::to_value(&caps).ok(),
                remote,
            }
        })
        .collect())
}

fn instance_view(i: &InstanceRecord) -> Value {
    let mut view = json!({
        "id": i.id, "kind": i.kind, "preset": i.preset, "label": i.label,
        "base_url": i.base_url, "origin": i.origin, "default_model": i.default_model,
        "cost_source": i.cost_source, "credential_ref": i.credential_ref, "builtin": false,
    });
    // Additive: only a remote instance carries the machine fields. The pinned key
    // is public; its fingerprint is what a human compares. The SSH private key is
    // only ever referenced (`credential_ref`), never present.
    if i.kind == st::KIND_CLAUDE_CODE_REMOTE {
        view["host"] = json!(i.host);
        view["ssh_user"] = json!(i.ssh_user);
        view["ssh_port"] = json!(i.ssh_port.unwrap_or(22));
        view["host_key"] = json!(i.host_key);
        view["host_key_fingerprint"] = json!(i.host_key_fingerprint());
        view["remote_cwd"] = json!(i.remote_cwd);
        view["allow_trust"] = json!(i.allow_trust);
    }
    view
}

/// The machine of a remote instance, for the listing.
fn remote_entry(i: &InstanceRecord) -> Option<RemoteEntry> {
    (i.kind == st::KIND_CLAUDE_CODE_REMOTE).then(|| RemoteEntry {
        host: i.host.clone().unwrap_or_default(),
        ssh_user: i.ssh_user.clone(),
        ssh_port: i.ssh_port.unwrap_or(22),
        host_key_fingerprint: i.host_key_fingerprint(),
        remote_cwd: i.remote_cwd.clone(),
        allow_trust: i.allow_trust,
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
    let env_allow = st::env_credential_allowlist();
    let record = match st::record_from_draft(&draft, &policy, &env_allow) {
        Ok(r) => r,
        Err(SettingsError::Endpoint(e)) => return Ok(Json(verdict("unreachable", e.code()))),
        Err(e) => return Err(map_settings_error(e)),
    };
    if !st::is_process_kind(&record.kind) {
        if let Err(e) = validate_endpoint(&record.base_url, &policy).await {
            return Ok(Json(verdict("unreachable", e.code())));
        }
    }
    // A codex / acp instance is a local process: the test is its health check
    // (version, login), nothing is sent anywhere.
    if st::is_process_kind(&record.kind) {
        let provider = match crate::chat::provider::native_factory::build_native_provider(
            &record,
            Some(state.vault.clone()),
        ) {
            Ok(p) => p,
            Err(e) => {
                let failure = crate::chat::provider::errors::open_failure(&e, Some(&record.id));
                return Ok(Json(verdict("unknown", failure.code)));
            }
        };
        let nexus_health = provider.health().await;
        let health = HealthEntry::from_nexus(&nexus_health);
        let health = if record.kind == st::KIND_CLAUDE_CODE_REMOTE {
            health.with_remote_reason(&nexus_health)
        } else {
            health
        };
        let mut verdict = json!({
            "ok": health.state == "ok",
            "health": { "state": health.state, "code": health.code, "action": health.action },
            "models": [],
        });
        // Additive: only a remote machine explains itself.
        if let Some(reason) = health.reason {
            verdict["health"]["reason"] = json!(reason);
        }
        return Ok(Json(verdict));
    }
    // A test that carries a credential sends it to the endpoint: it is only done
    // for an instance that is ALREADY saved, at the origin it was saved with and
    // with the credential reference it was saved with. A draft with a key is
    // saved first (it cannot serve a session until a project consents anyway).
    if record.credential_ref != "none" {
        let saved = graph(&state)
            .get_llm_setting(GLOBAL, &format!("{INSTANCE_PREFIX}{}", record.id))
            .await
            .map_err(AppError::Internal)?
            .and_then(|raw| parse::<InstanceRecord>(&raw));
        let same = saved.is_some_and(|s| {
            s.origin == record.origin && s.credential_ref == record.credential_ref
        });
        if !same {
            return Ok(Json(verdict(
                "unknown",
                "credential_test_requires_saved_instance",
            )));
        }
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
    let record = st::record_from_draft(&draft, &policy, &st::env_credential_allowlist())
        .map_err(map_settings_error)?;
    if !st::is_process_kind(&record.kind) {
        validate_endpoint(&record.base_url, &policy)
            .await
            .map_err(|e| AppError::BadRequest(format!("endpoint refused: {e}")))?;
    }
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

/// GET /api/chat/providers/{id} — the stored detail of one instance, for the
/// edit form: the full `base_url` (path included), model, preset, cost source
/// and credential REFERENCE. Human only: the listing deliberately never shows a
/// path, and this route is the one that does. Never a secret value.
pub async fn get_provider(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
) -> Result<Json<Value>, AppError> {
    require_human(&state, &claims)?;
    if id == CLAUDE_CODE {
        return Err(map_settings_error(SettingsError::Builtin));
    }
    let record: InstanceRecord = graph(&state)
        .get_llm_setting(GLOBAL, &format!("{INSTANCE_PREFIX}{id}"))
        .await
        .map_err(AppError::Internal)?
        .and_then(|v| parse(&v))
        .ok_or_else(|| map_settings_error(SettingsError::UnknownInstance(id.clone())))?;
    Ok(Json(instance_view(&record)))
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
        st::apply_patch(&old, &patch, &policy, &st::env_credential_allowlist())
            .map_err(map_settings_error)?;
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
    let record = crate::chat::provider::store::instance(graph(&state).as_ref(), &id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("unknown provider instance '{id}'")))?;
    // The REAL health of the instance now (reachable? logged in? a locked vault is
    // `auth_required`), checked by the provider itself. Nothing is sent but the
    // provider's own health request.
    let health = match crate::chat::provider::native_factory::build_native_provider(
        &record,
        Some(state.vault.clone()),
    ) {
        Ok(provider) => {
            let nexus_health = provider.health().await;
            let entry = HealthEntry::from_nexus(&nexus_health);
            if record.kind == st::KIND_CLAUDE_CODE_REMOTE {
                entry.with_remote_reason(&nexus_health)
            } else {
                entry
            }
        }
        Err(_) => HealthEntry::unknown(),
    };
    Ok(Json(health))
}

// ============================================================================
// SSH host key discovery (claude_code_remote)
// ============================================================================

/// Body of `POST /api/chat/providers/ssh-host-key`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostKeyScanBody {
    /// Host name or address of the machine.
    pub host: String,
    /// ssh port (22 when absent).
    #[serde(default)]
    pub ssh_port: Option<u16>,
}

/// Total time `ssh-keyscan` may take (its own `-T 5` is per connection).
const KEYSCAN_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);
/// Output ceiling: a key line is a few hundred bytes.
const KEYSCAN_MAX_OUTPUT: u64 = 64 * 1024;

/// Why a scan failed; the texts are fixed (no raw tool output).
#[derive(Debug, PartialEq, Eq)]
pub enum KeyScanError {
    /// `ssh-keyscan` could not be started on this server.
    NotAvailable,
    /// No usable key came back (unreachable, refused, timed out, no ssh there).
    NoKey,
}

/// The arguments of `ssh-keyscan` for a host: explicit words, never a shell, and
/// `--` before the host so it can never be read as an option.
pub fn keyscan_args(host: &str, port: u16) -> Vec<String> {
    vec![
        "-T".into(),
        "5".into(),
        "-p".into(),
        port.to_string(),
        "-t".into(),
        "ed25519,ecdsa,rsa".into(),
        "--".into(),
        host.to_string(),
    ]
}

/// The first valid key of `ssh-keyscan` output (`<host> <type> <base64>` lines,
/// `#` comments): `("<type> <base64>", fingerprint)`.
pub fn parse_keyscan_output(output: &str) -> Option<(String, String)> {
    output.lines().find_map(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        let [_host, kind, blob] = words[..] else {
            return None;
        };
        let key = format!("{kind} {blob}");
        st::parse_host_key(&key).ok()?;
        let fingerprint = st::host_key_fingerprint(&key)?;
        Some((key, fingerprint))
    })
}

/// Runs `program` with `args` under the deadline and the output ceiling, and
/// parses the first key. The child dies with the future (`kill_on_drop`).
pub async fn run_keyscan(
    program: &str,
    args: &[String],
    deadline: std::time::Duration,
) -> Result<(String, String), KeyScanError> {
    use tokio::io::AsyncReadExt;
    let mut child = tokio::process::Command::new(program)
        .args(args)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|_| KeyScanError::NotAvailable)?;
    let stdout = child.stdout.take().ok_or(KeyScanError::NotAvailable)?;
    let read = async {
        let mut buf = Vec::new();
        let _ = stdout.take(KEYSCAN_MAX_OUTPUT).read_to_end(&mut buf).await;
        buf
    };
    let Ok(buf) = tokio::time::timeout(deadline, read).await else {
        let _ = child.start_kill();
        return Err(KeyScanError::NoKey);
    };
    // Past the ceiling or not, the child is not waited on: it is killed.
    let _ = child.start_kill();
    parse_keyscan_output(&String::from_utf8_lossy(&buf)).ok_or(KeyScanError::NoKey)
}

/// POST /api/chat/providers/ssh-host-key — read the public host key a machine
/// presents, and its fingerprint, so a HUMAN can confirm it before pinning it.
///
/// This is trust on first use made explicit: the key returned here is whatever
/// the network answered, so the UI must show the fingerprint and ask the person
/// to compare it with the machine's own (`ssh-keygen -lf` on the host). Nothing
/// is stored; the pinned key only enters through creating or patching an
/// instance. Same auth as the other provider-settings writes (a signed-in
/// person, authentication enabled): the server opens a connection to the host
/// the body names.
pub async fn scan_ssh_host_key(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Json(body): Json<HostKeyScanBody>,
) -> Result<Json<Value>, AppError> {
    require_human(&state, &claims)?;
    security_gate(&state)?;
    if !st::valid_ssh_word(&body.host) {
        return Err(AppError::BadRequest(
            "host: a plain name or address (letters, digits and . _ - : [ ] % @, not starting with '-')"
                .to_string(),
        ));
    }
    if body.ssh_port == Some(0) {
        return Err(AppError::BadRequest(
            "ssh_port: between 1 and 65535".to_string(),
        ));
    }
    let args = keyscan_args(&body.host, body.ssh_port.unwrap_or(22));
    match run_keyscan("ssh-keyscan", &args, KEYSCAN_DEADLINE).await {
        Ok((host_key, host_key_fingerprint)) => Ok(Json(json!({
            "host_key": host_key,
            "host_key_fingerprint": host_key_fingerprint,
        }))),
        Err(KeyScanError::NotAvailable) => Err(AppError::NotImplemented(
            "ssh-keyscan is not available on this server".to_string(),
        )),
        Err(KeyScanError::NoKey) => Err(AppError::BadRequest(
            "no host key could be read from that machine (unreachable, refused or too slow)"
                .to_string(),
        )),
    }
}

/// Query of `GET /api/chat/send-journal`.
#[derive(Debug, Deserialize)]
pub struct JournalQuery {
    /// Only the sendings of this project.
    pub project_slug: Option<String>,
    /// Most recent first, at most this many (default 100, max 500).
    pub limit: Option<usize>,
}

/// GET /api/chat/send-journal — who sent which project's content to which origin
/// (A37), most recent first. A person only: the journal is the user's own audit
/// trail, an agent has no business reading it. It never holds the content sent.
pub async fn send_journal(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    axum::extract::Query(query): axum::extract::Query<JournalQuery>,
) -> Result<Json<Value>, AppError> {
    require_human(&state, &claims)?;
    let limit = query.limit.unwrap_or(100).clamp(1, 500);
    let mut rows = graph(&state)
        .list_llm_settings("journal", "send:")
        .await
        .map_err(AppError::Internal)?;
    // Keys are `send:<epoch ms>:<session>`: newest last.
    rows.reverse();
    let entries: Vec<Value> = rows
        .iter()
        .filter_map(|(_, v)| serde_json::from_str::<Value>(v).ok())
        .filter(|e| match &query.project_slug {
            Some(p) => e["project"] == p.as_str(),
            None => true,
        })
        .take(limit)
        .collect();
    Ok(Json(json!({ "entries": entries })))
}

/// The stored default stays first and is kept even when the endpoint does not
/// list it (some gateways answer with a partial list); duplicates collapse.
fn merge_listed_models(mut listed: Vec<String>, default: Option<&str>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    listed.retain(|m| seen.insert(m.clone()));
    if let Some(d) = default {
        listed.retain(|m| m != d);
        listed.insert(0, d.to_string());
    }
    listed
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
    let stored: Vec<Value> = record
        .default_model
        .iter()
        .map(|m| json!({ "id": m }))
        .collect();
    // A process instance (codex, acp) chooses its own models: only the stored
    // default is known here.
    if st::is_process_kind(&record.kind) {
        return Ok(Json(json!(stored)));
    }
    // An OpenAI-compatible endpoint lists its models (`GET /models`). Ask it:
    // answering with the stored default alone left an instance saved without a
    // default with an empty picker, so no session could ever name a model.
    // A refusal or an unreachable endpoint falls back to the stored default; the
    // chat shows its own state card for the failure when a session is opened.
    let listed = match crate::chat::provider::native_factory::list_models(
        &record,
        Some(state.vault.clone()),
    )
    .await
    {
        Ok(ids) => ids,
        Err(e) => {
            tracing::warn!(provider = %record.id, error = %e, "model listing failed");
            Vec::new()
        }
    };
    if listed.is_empty() {
        return Ok(Json(json!(stored)));
    }
    let ids = merge_listed_models(listed, record.default_model.as_deref());
    Ok(Json(json!(ids
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
            let current = instances.iter().find(|i| i.id == c.provider_id);
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
        credential_ref: Some(instance.credential_ref.clone()),
        host_key_fingerprint: instance.host_key_fingerprint(),
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
    Ok(Json(st::consent_view(&record, Some(&instance))))
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

#[cfg(test)]
mod listed_models_tests {
    use super::merge_listed_models;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn the_endpoint_order_is_kept_without_a_default() {
        assert_eq!(
            merge_listed_models(v(&["deepseek-chat", "deepseek-reasoner"]), None),
            v(&["deepseek-chat", "deepseek-reasoner"])
        );
    }

    #[test]
    fn the_stored_default_comes_first_and_is_not_repeated() {
        assert_eq!(
            merge_listed_models(v(&["a", "b", "c"]), Some("b")),
            v(&["b", "a", "c"])
        );
    }

    #[test]
    fn a_default_the_endpoint_does_not_list_is_kept() {
        assert_eq!(merge_listed_models(v(&["a"]), Some("z")), v(&["z", "a"]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::time::{Duration, Instant};

    const KEY: &str =
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEXTCY3J636nEyMNqNrVj6HXnIXUnXL8sk4c5J6tek3N";
    const FP: &str = "SHA256:lP63ZdLutNnRU0/59cDaFw2mPoJzdasi0I3zFrtS3Ak";

    /// A fake `ssh-keyscan`: a shell script printing `body`.
    fn script(dir: &tempfile::TempDir, body: &str) -> String {
        let path = dir.path().join("fake-keyscan");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.display().to_string()
    }

    #[test]
    fn the_first_valid_key_line_is_parsed_and_noise_is_skipped() {
        let out = format!(
            "# build-1:22 SSH-2.0-OpenSSH\n\nbuild-1 ssh-rsa not*base64*key-at-all\nbuild-1 ssh-ed25519\nbuild-1 {KEY}\nbuild-1 ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC7\n"
        );
        let (key, fp) = parse_keyscan_output(&out).expect("a key");
        assert_eq!(key, KEY);
        assert_eq!(fp, FP);
        assert!(parse_keyscan_output("# only a comment\n").is_none());
        assert!(parse_keyscan_output("host ssh-ed25519 AAAA\n").is_none());
        assert!(parse_keyscan_output(
            "host ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEXTCY3J636n extra words\n"
        )
        .is_none());
    }

    #[test]
    fn the_host_always_follows_the_end_of_options_marker() {
        let args = keyscan_args("build-1.example.net", 2222);
        let end = args.iter().position(|a| a == "--").expect("a `--`");
        assert_eq!(args[end + 1], "build-1.example.net");
        assert_eq!(end + 2, args.len());
        assert_eq!(&args[..2], ["-T", "5"]);
        assert!(args.windows(2).any(|w| w == ["-p", "2222"]));
        assert!(args.windows(2).any(|w| w == ["-t", "ed25519,ecdsa,rsa"]));
    }

    #[tokio::test]
    async fn a_scan_returns_the_key_and_its_fingerprint() {
        let dir = tempfile::TempDir::new().unwrap();
        let program = script(&dir, &format!("printf 'h {KEY}\\n'"));
        let (key, fp) = run_keyscan(&program, &keyscan_args("h", 22), Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!((key.as_str(), fp.as_str()), (KEY, FP));
    }

    #[tokio::test]
    async fn a_scan_that_hangs_is_cut_at_the_deadline() {
        let dir = tempfile::TempDir::new().unwrap();
        let program = script(&dir, "sleep 30");
        let started = Instant::now();
        let r = run_keyscan(&program, &[], Duration::from_millis(300)).await;
        assert_eq!(r, Err(KeyScanError::NoKey));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn a_scan_that_never_stops_writing_is_capped() {
        let dir = tempfile::TempDir::new().unwrap();
        let program = script(&dir, "yes 'host ssh-ed25519 junk'");
        let started = Instant::now();
        let r = run_keyscan(&program, &[], Duration::from_secs(8)).await;
        assert_eq!(r, Err(KeyScanError::NoKey));
        assert!(
            started.elapsed() < Duration::from_secs(4),
            "the output ceiling must end the read, not the deadline"
        );
    }

    #[tokio::test]
    async fn a_missing_program_is_not_available() {
        let r = run_keyscan("/nonexistent/ssh-keyscan", &[], Duration::from_secs(1)).await;
        assert_eq!(r, Err(KeyScanError::NotAvailable));
    }
}
