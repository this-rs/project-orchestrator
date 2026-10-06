//! REST handlers for the secrets vault.
//!
//! User side (a person's token only — see [`require_human`]):
//! - `GET    /api/vault`                       status, secrets (names only), grants, pending requests
//! - `POST   /api/vault/init`                  `{passphrase, minutes?}`
//! - `POST   /api/vault/unlock`                `{passphrase, minutes?}`
//! - `POST   /api/vault/lock`
//! - `PUT    /api/vault/secrets/{name}`        `{value, description?}`
//! - `DELETE /api/vault/secrets/{name}`
//! - `POST   /api/vault/grants`                `{secrets, scope, minutes?, note?}`
//! - `DELETE /api/vault/grants/{id}`
//! - `POST   /api/vault/requests/{id}/answer`  answer an agent's request, optionally unlocking in the same call
//!
//! Agent side (vault token only — the session is read from its signature):
//! - `POST   /api/vault/agent/read`            `{name}` → the value, `text/plain`
//! - `POST   /api/vault/agent/requests`        `{name, reason}` → shows a secure input card in the chat
//!
//! No response except `agent/read` ever contains a value, and no request body
//! is logged: the handlers take bodies as typed structs whose secret fields are
//! wiped on drop and never formatted.

use axum::{
    extract::{Path, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use chrono::{Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use zeroize::Zeroizing;

use super::handlers::{AppError, OrchestratorState};
use crate::auth::jwt::{vault_token_session, Claims};
use crate::chat::types::ChatEvent;
use crate::vault::{
    Denied, Grant, GrantScope, RequestAnswer, SecretMeta, SecretRequest, SecretSelector,
    ServiceError, VaultError, VaultStatus,
};

const DEFAULT_UNLOCK_MINUTES: i64 = 60;
const DEFAULT_GRANT_MINUTES: i64 = 60;

// ============================================================================
// Errors and identity
// ============================================================================

impl From<ServiceError> for AppError {
    fn from(e: ServiceError) -> Self {
        let msg = e.to_string();
        match e {
            ServiceError::Vault(VaultError::NotInitialized) => AppError::Conflict(msg),
            ServiceError::Vault(VaultError::AlreadyInitialized) => AppError::Conflict(msg),
            ServiceError::Vault(VaultError::Locked) => AppError::Conflict(msg),
            // 403, not 401: clients read 401 as "your login expired" and log
            // the user out — a mistyped passphrase must not do that.
            ServiceError::Vault(VaultError::WrongPassphrase) => AppError::Forbidden(msg),
            ServiceError::Vault(VaultError::UnknownSecret) => AppError::NotFound(msg),
            ServiceError::Vault(
                VaultError::WeakPassphrase | VaultError::InvalidName | VaultError::ValueTooShort,
            ) => AppError::BadRequest(msg),
            ServiceError::Denied(Denied::UnknownSecret) => AppError::NotFound(msg),
            ServiceError::Denied(_) => AppError::Forbidden(msg),
            ServiceError::Throttled(_) => AppError::Forbidden(msg),
            ServiceError::UnknownRequest => AppError::NotFound(msg),
            ServiceError::ProofRequired => AppError::Forbidden(msg),
            _ => AppError::Internal(anyhow::anyhow!(msg)),
        }
    }
}

/// Management endpoints widen what agents can read. They must be called by a
/// person — not by an agent holding a session, vault or MCP token, which could
/// otherwise grant itself anything while the vault is open.
///
/// Without auth (`auth_config: None`) there are no people to tell apart; the
/// server is trusted as a whole, as everywhere else in that mode.
fn require_human(state: &OrchestratorState, claims: &Claims) -> Result<(), AppError> {
    if state.auth_config.is_none() || claims.is_human() {
        Ok(())
    } else {
        Err(AppError::Forbidden(
            "vault management requires a user session, not an agent token".to_string(),
        ))
    }
}

/// Header carrying the unlock proof (see `VaultService::check_proof`).
pub const PROOF_HEADER: &str = "x-vault-proof";

/// Changes that widen agent access (store, delete, grant, answer) need the
/// unlock proof on top of a human login: agents can forge logins, not proofs.
fn require_proof(state: &OrchestratorState, headers: &HeaderMap) -> Result<(), AppError> {
    let proof = headers.get(PROOF_HEADER).and_then(|v| v.to_str().ok());
    Ok(state.vault.check_proof(proof, Utc::now())?)
}

fn minutes(value: Option<i64>, default: i64) -> Duration {
    Duration::minutes(value.unwrap_or(default).max(1))
}

// ============================================================================
// User side
// ============================================================================

#[derive(Serialize)]
pub struct VaultOverview {
    #[serde(flatten)]
    pub status: VaultStatus,
    /// Set when the vault file exists but cannot be read.
    pub unavailable: Option<String>,
    pub secrets: Vec<SecretMeta>,
    pub grants: Vec<Grant>,
    pub requests: Vec<SecretRequest>,
}

pub async fn get_vault(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
) -> Result<Json<VaultOverview>, AppError> {
    require_human(&state, &claims)?;
    let v = &state.vault;
    let now = Utc::now();
    Ok(Json(VaultOverview {
        status: v.status(now),
        unavailable: v.unavailable_reason().map(str::to_string),
        secrets: v.list(),
        grants: v.grants(now),
        requests: v.pending_requests(now),
    }))
}

/// Passphrase body. No `Debug`: nothing can format it into a log.
#[derive(Deserialize)]
pub struct PassphraseBody {
    passphrase: Zeroizing<String>,
    minutes: Option<i64>,
}

/// Returned by init and unlock. `unlock_proof` is to be kept in memory by the
/// client (never stored) and sent as `x-vault-proof` on changes.
#[derive(Serialize)]
pub struct UnlockedResponse {
    pub unlocked_until: Option<chrono::DateTime<Utc>>,
    pub unlock_proof: String,
}

pub async fn init_vault(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Json(body): Json<PassphraseBody>,
) -> Result<(StatusCode, Json<UnlockedResponse>), AppError> {
    require_human(&state, &claims)?;
    let duration = minutes(body.minutes, DEFAULT_UNLOCK_MINUTES);
    let proof = state
        .vault
        .init(body.passphrase.to_string(), duration)
        .await?;
    Ok((
        StatusCode::CREATED,
        Json(UnlockedResponse {
            unlocked_until: state.vault.status(Utc::now()).unlocked_until,
            unlock_proof: proof,
        }),
    ))
}

pub async fn unlock_vault(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Json(body): Json<PassphraseBody>,
) -> Result<Json<UnlockedResponse>, AppError> {
    require_human(&state, &claims)?;
    let duration = minutes(body.minutes, DEFAULT_UNLOCK_MINUTES);
    let (until, proof) = state
        .vault
        .unlock(body.passphrase.to_string(), duration)
        .await?;
    Ok(Json(UnlockedResponse {
        unlocked_until: Some(until),
        unlock_proof: proof,
    }))
}

pub async fn lock_vault(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
) -> Result<StatusCode, AppError> {
    require_human(&state, &claims)?;
    state.vault.lock_now();
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
pub struct PutSecretBody {
    value: Zeroizing<String>,
    description: Option<String>,
}

pub async fn put_secret(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(body): Json<PutSecretBody>,
) -> Result<StatusCode, AppError> {
    require_human(&state, &claims)?;
    require_proof(&state, &headers)?;
    state
        .vault
        .put(&name, &body.value, body.description, Utc::now())?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn delete_secret(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<StatusCode, AppError> {
    require_human(&state, &claims)?;
    require_proof(&state, &headers)?;
    state.vault.delete(&name)?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
pub struct CreateGrantBody {
    pub secrets: SecretSelector,
    pub scope: GrantScope,
    pub minutes: Option<i64>,
    pub note: Option<String>,
}

pub async fn create_grant(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    headers: HeaderMap,
    Json(body): Json<CreateGrantBody>,
) -> Result<(StatusCode, Json<Grant>), AppError> {
    require_human(&state, &claims)?;
    require_proof(&state, &headers)?;
    // A grant to a provider instance lets the SERVER read that key to send it to
    // that instance's endpoint: it must name the key the instance is configured
    // with, on an instance that exists, and nothing else.
    if let GrantScope::Provider(instance) = &body.scope {
        validate_provider_grant(state.orchestrator.neo4j(), instance, &body.secrets).await?;
    }
    let grant = state.vault.grant(
        body.secrets,
        body.scope,
        minutes(body.minutes, DEFAULT_GRANT_MINUTES),
        body.note,
        Utc::now(),
    )?;
    Ok((StatusCode::CREATED, Json(grant)))
}

/// Validation of a `Provider(instance)` grant.
///
/// - the instance must be a stored one (`claude-code` has no key to grant);
/// - the selector must be `names`, never `all`;
/// - the names must be exactly what the instance's `credential_ref` points at
///   (`vault:<name>`): a grant cannot hand a provider another secret.
pub(crate) async fn validate_provider_grant(
    graph: &dyn crate::neo4j::GraphStore,
    instance: &str,
    secrets: &SecretSelector,
) -> Result<(), AppError> {
    let record = crate::chat::provider::store::instance(graph, instance)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| {
            AppError::NotFound(format!(
                "unknown provider instance '{instance}' (claude-code has no key to grant)"
            ))
        })?;
    let SecretSelector::Names(names) = secrets else {
        return Err(AppError::BadRequest(
            "a provider grant names its secret: `all` is not accepted".to_string(),
        ));
    };
    let allowed = record.credential_ref.strip_prefix("vault:");
    if names.is_empty() || names.iter().any(|n| Some(n.as_str()) != allowed) {
        return Err(AppError::BadRequest(format!(
            "instance '{instance}' reads the secret named in its credential_ref ({}); a grant to it can name only that",
            record.credential_ref
        )));
    }
    Ok(())
}

pub async fn revoke_grant(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    require_human(&state, &claims)?;
    if state.vault.revoke(id, Utc::now())? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound("no such grant".to_string()))
    }
}

/// How the user answers a request. `passphrase` unlocks the vault in the
/// same call — the "one card, one click" path when the vault was locked.
#[derive(Deserialize)]
pub struct AnswerBody {
    /// `provide` | `grant` | `decline`
    action: String,
    value: Option<Zeroizing<String>>,
    description: Option<String>,
    /// Defaults to the requesting session.
    scope: Option<GrantScope>,
    minutes: Option<i64>,
    passphrase: Option<Zeroizing<String>>,
    unlock_minutes: Option<i64>,
}

#[derive(Serialize)]
pub struct AnswerResponse {
    pub outcome: &'static str,
    pub grant: Option<Grant>,
    /// Set when the answer unlocked the vault (inline passphrase): the card's
    /// tab keeps it like any unlock proof.
    pub unlock_proof: Option<String>,
}

pub async fn answer_request(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    headers: HeaderMap,
    Path(id): Path<Uuid>,
    Json(body): Json<AnswerBody>,
) -> Result<Json<AnswerResponse>, AppError> {
    require_human(&state, &claims)?;
    // An agent's request is answered with a session/project/anywhere grant. A
    // grant to a provider instance is made on its own, validated (see
    // `validate_provider_grant`), never through an agent's request.
    if matches!(body.scope, Some(GrantScope::Provider(_))) {
        return Err(AppError::BadRequest(
            "a provider grant cannot answer an agent's request".to_string(),
        ));
    }
    let (answer, outcome) = match body.action.as_str() {
        "provide" => {
            let value = body
                .value
                .ok_or_else(|| AppError::BadRequest("`value` is required to provide".into()))?;
            (
                RequestAnswer::Provide {
                    value: value.to_string(),
                    description: body.description,
                },
                "provided",
            )
        }
        "grant" => (RequestAnswer::GrantExisting, "granted"),
        "decline" => (RequestAnswer::Decline, "declined"),
        other => {
            return Err(AppError::BadRequest(format!(
                "unknown action `{other}` (provide | grant | decline)"
            )))
        }
    };
    // Declining narrows nothing: no proof needed. Anything else widens agent
    // access: a passphrase typed on the card, or a proof from an earlier unlock.
    let mut new_proof = None;
    if !matches!(answer, RequestAnswer::Decline) {
        match body.passphrase {
            Some(pass) => {
                let d = minutes(body.unlock_minutes, DEFAULT_UNLOCK_MINUTES);
                let (_, proof) = state.vault.unlock(pass.to_string(), d).await?;
                new_proof = Some(proof);
            }
            None => require_proof(&state, &headers)?,
        }
    }
    let (request, grant) = state.vault.answer_request(
        id,
        answer,
        body.scope,
        minutes(body.minutes, DEFAULT_GRANT_MINUTES),
        Utc::now(),
    )?;
    emit_to_session(
        &state,
        &request.session_id,
        ChatEvent::SecretRequestResolved {
            id: request.id.to_string(),
            outcome: outcome.to_string(),
        },
    )
    .await;
    // Tell the agent, in its own conversation, that it can go on. Without
    // this it would sit waiting for a tool result that never comes.
    if let Some(cm) = &state.chat_manager {
        let hint = match outcome {
            "declined" => format!(
                "The user declined to provide the secret `{}`. Do not ask again for it in this task.",
                request.name
            ),
            _ => format!(
                "The secret `{0}` is now available to this session. Use it only inside a shell command so the value never reaches your context: `orchestrator secret exec -e VAR={0} -- cmd args` (preferred) or `orchestrator secret get {0} | cmd --password-stdin`. Never echo or print it.",
                request.name
            ),
        };
        let _ = cm.inject_hint(&request.session_id, &hint).await;
    }
    Ok(Json(AnswerResponse {
        outcome,
        grant,
        unlock_proof: new_proof,
    }))
}

// ============================================================================
// Agent side
// ============================================================================

/// The session a vault token speaks for, and its project (resolved here, from
/// the graph — never taken from the agent).
async fn agent_identity(
    state: &OrchestratorState,
    claims: &Claims,
) -> Result<(String, Option<String>), AppError> {
    let session = vault_token_session(claims)
        .ok_or_else(|| AppError::Forbidden("a vault token is required".to_string()))?
        .to_string();
    let project = match Uuid::parse_str(&session) {
        Ok(id) => state
            .orchestrator
            .neo4j()
            .get_chat_session(id)
            .await
            .ok()
            .flatten()
            .and_then(|s| s.project_slug),
        Err(_) => None,
    };
    Ok((session, project))
}

#[derive(Deserialize)]
pub struct AgentReadBody {
    name: String,
}

pub async fn agent_read(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Json(body): Json<AgentReadBody>,
) -> Result<Response, AppError> {
    let (session, project) = agent_identity(&state, &claims).await?;
    let value = state
        .vault
        .read_for_agent(&body.name, &session, project.as_deref(), Utc::now())?;
    tracing::info!(secret = %body.name, session = %session, "vault: secret delivered to agent");
    Ok((
        [
            (header::CONTENT_TYPE, "text/plain; charset=utf-8"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        value.to_string(),
    )
        .into_response())
}

#[derive(Deserialize)]
pub struct AgentRequestBody {
    name: String,
    reason: String,
}

#[derive(Serialize)]
pub struct AgentRequestResponse {
    pub request_id: Uuid,
    /// What the agent should do now, in plain words.
    pub next: String,
}

pub async fn agent_request(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Json(body): Json<AgentRequestBody>,
) -> Result<Json<AgentRequestResponse>, AppError> {
    let (session, project) = agent_identity(&state, &claims).await?;
    let now = Utc::now();
    // Already readable? Say so rather than bothering the user.
    if state
        .vault
        .read_for_agent_check(&body.name, &session, project.as_deref(), now)
    {
        return Ok(Json(AgentRequestResponse {
            request_id: Uuid::nil(),
            next: format!(
                "Already granted. Use it in a pipeline: `orchestrator secret get {} | …`.",
                body.name
            ),
        }));
    }
    let req =
        state
            .vault
            .open_request(&body.name, &body.reason, &session, project.as_deref(), now)?;
    emit_to_session(
        &state,
        &session,
        ChatEvent::SecretRequest {
            id: req.id.to_string(),
            name: req.name.clone(),
            reason: req.reason.clone(),
            exists: req.exists,
        },
    )
    .await;
    Ok(Json(AgentRequestResponse {
        request_id: req.id,
        next: "The user sees a secure input card in the chat. End your turn now and wait: \
               you will be told in this conversation when the secret is available or declined."
            .to_string(),
    }))
}

pub async fn agent_available(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
) -> Result<Json<crate::vault::service::AgentView>, AppError> {
    let (session, project) = agent_identity(&state, &claims).await?;
    Ok(Json(state.vault.available_to(
        &session,
        project.as_deref(),
        Utc::now(),
    )))
}

async fn emit_to_session(state: &OrchestratorState, session_id: &str, event: ChatEvent) {
    if let Some(cm) = &state.chat_manager {
        if let Ok(tx) = cm.get_events_tx(session_id).await {
            let _ = tx.send(event.clone());
        }
    }
    if let Some(nats) = &state.nats_emitter {
        nats.publish_chat_event(session_id, event);
    }
}

// ============================================================================
// Tests — the security properties, through the real router and middleware
// ============================================================================

#[cfg(test)]
mod tests {
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::auth::jwt::{decode_jwt, encode_jwt, generate_session_token, generate_vault_token};
    use crate::events::EventBus;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::mock_app_state;
    use crate::AuthConfig;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;
    use uuid::Uuid;

    const SECRET: &str = "test-secret-key-minimum-32-chars!!";

    /// These tests share `tracing` callsites (the handlers' log lines). Run
    /// one at a time so the log-capture test sees every event: a callsite hit
    /// concurrently by a sibling can cache "no interest" for its subscriber.
    static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    const VALUE: &str = "the-demo-passphrase";

    fn auth() -> AuthConfig {
        AuthConfig {
            jwt_secret: SECRET.to_string(),
            access_token_expiry_secs: 3600,
            refresh_token_expiry_secs: 604800,
            allowed_email_domain: None,
            allowed_emails: None,
            frontend_url: None,
            additional_origins: vec![],
            allow_registration: false,
            root_account: None,
            oidc: None,
            google_client_id: Some("test".to_string()),
            google_client_secret: Some("test".to_string()),
            google_redirect_uri: Some("http://localhost/callback".to_string()),
        }
    }

    async fn app() -> (Router, Arc<ServerState>) {
        let event_bus = Arc::new(crate::events::HybridEmitter::new(Arc::new(
            EventBus::default(),
        )));
        let orchestrator = Arc::new(
            Orchestrator::with_event_bus(mock_app_state(), event_bus.clone())
                .await
                .unwrap(),
        );
        let watcher = FileWatcher::new(orchestrator.clone());
        let state = Arc::new(ServerState {
            orchestrator,
            watcher: Arc::new(RwLock::new(watcher)),
            chat_manager: None,
            event_bus,
            nats_emitter: None,
            auth_config: Some(auth()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            remote_mcp: crate::RemoteMcpConfig::default(),
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
        (create_router(state.clone()), state)
    }

    struct Tokens {
        human: String,
        agent_session: String,
        vault_s1: String,
        vault_s2: String,
    }

    fn tokens() -> Tokens {
        let human = encode_jwt(Uuid::new_v4(), "t@example.com", "T", SECRET, 3600).unwrap();
        let claims = decode_jwt(&human, SECRET).unwrap();
        let (agent_session, jti) = generate_session_token(&claims, None, SECRET, 3600).unwrap();
        crate::auth::agent_tokens::register(&jti, None);
        Tokens {
            agent_session,
            vault_s1: generate_vault_token(&claims, "session-1", SECRET, 3600).unwrap(),
            vault_s2: generate_vault_token(&claims, "session-2", SECRET, 3600).unwrap(),
            human,
        }
    }

    async fn call(
        app: &Router,
        method: &str,
        path: &str,
        token: &str,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, String) {
        call_with_proof(app, method, path, token, None, body).await
    }

    async fn call_with_proof(
        app: &Router,
        method: &str,
        path: &str,
        token: &str,
        proof: Option<&str>,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, String) {
        let mut req = Request::builder()
            .method(method)
            .uri(path)
            .header("authorization", format!("Bearer {token}"));
        if let Some(p) = proof {
            req = req.header(super::PROOF_HEADER, p);
        }
        let body = match body {
            Some(b) => {
                req = req.header("content-type", "application/json");
                Body::from(b.to_string())
            }
            None => Body::empty(),
        };
        let resp = app.clone().oneshot(req.body(body).unwrap()).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Create the vault and store one secret; returns the unlock proof.
    async fn open_with_secret(app: &Router, t: &Tokens) -> String {
        let (s, body) = call(
            app,
            "POST",
            "/api/vault/init",
            &t.human,
            Some(serde_json::json!({"passphrase": "correct horse battery staple"})),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED);
        let proof = serde_json::from_str::<serde_json::Value>(&body).unwrap()["unlock_proof"]
            .as_str()
            .unwrap()
            .to_string();
        let (s, _) = call_with_proof(
            app,
            "PUT",
            "/api/vault/secrets/demo-secret",
            &t.human,
            Some(&proof),
            Some(serde_json::json!({"value": VALUE})),
        )
        .await;
        assert_eq!(s, StatusCode::NO_CONTENT);
        proof
    }

    /// Captures every log line, at every level, while the closure-driven
    /// scenario runs on this (current-thread) test runtime.
    #[derive(Clone, Default)]
    struct LogSink(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for LogSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogSink {
        type Writer = LogSink;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Whole flow — store, grant, agent read, agent request answered with a
    /// typed value — with TRACE logging on (request tracing included): no
    /// value, no passphrase and no unlock proof may appear in any log line.
    #[tokio::test]
    async fn no_value_passphrase_or_proof_ever_reaches_the_logs() {
        let _serial = SERIAL.lock().await;
        let sink = LogSink::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .with_writer(sink.clone())
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);
        // Callsites first hit by a parallel test may have cached "no interest"
        // before this subscriber existed; re-evaluate them now. The negative
        // control below fails loudly if capture still misses the handlers.
        tracing::callsite::rebuild_interest_cache();

        let (app, _) = app().await;
        let t = tokens();
        let proof = open_with_secret(&app, &t).await;
        let grant = serde_json::json!({
            "secrets": {"kind": "all"},
            "scope": {"kind": "session", "value": "session-1"},
        });
        let (s, _) = call_with_proof(
            &app,
            "POST",
            "/api/vault/grants",
            &t.human,
            Some(&proof),
            Some(grant),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED);
        let read = Some(serde_json::json!({"name": "demo-secret"}));
        let (s, _) = call(&app, "POST", "/api/vault/agent/read", &t.vault_s1, read).await;
        assert_eq!(s, StatusCode::OK);

        let ask = Some(serde_json::json!({"name": "gh-token", "reason": "release"}));
        let (_, body) = call(&app, "POST", "/api/vault/agent/requests", &t.vault_s1, ask).await;
        let id = serde_json::from_str::<serde_json::Value>(&body).unwrap()["request_id"]
            .as_str()
            .unwrap()
            .to_string();
        let answer = Some(serde_json::json!({
            "action": "provide",
            "value": "ghp_typed-on-the-card-42",
            "passphrase": "correct horse battery staple",
        }));
        let path = format!("/api/vault/requests/{id}/answer");
        let (s, _) = call(&app, "POST", &path, &t.human, answer).await;
        assert_eq!(s, StatusCode::OK);

        let logs = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
        // Negative control: the capture works and saw the requests.
        assert!(logs.contains("vault: secret delivered to agent"), "{logs}");
        for leaked in [
            VALUE,
            "ghp_typed-on-the-card-42",
            "correct horse battery staple",
            &proof,
        ] {
            assert!(!logs.contains(leaked), "leaked in logs: {leaked}");
        }
    }

    /// The attack the proof exists for: an agent reads the server config,
    /// forges a perfectly valid HUMAN login token, and tries to grant itself a
    /// secret while the vault is open. It has everything but the passphrase.
    #[tokio::test]
    async fn a_forged_login_without_the_unlock_proof_cannot_widen_access() {
        let _serial = SERIAL.lock().await;
        let (app, _) = app().await;
        let t = tokens();
        let real_proof = open_with_secret(&app, &t).await;
        let forged = encode_jwt(Uuid::new_v4(), "agent@evil", "Agent", SECRET, 3600).unwrap();
        let grant = || {
            Some(serde_json::json!({
                "secrets": {"kind": "all"},
                "scope": {"kind": "anywhere"},
            }))
        };
        let zeros = "00".repeat(32);
        for proof in [None, Some(zeros.as_str()), Some(&real_proof[..10])] {
            let (s, _) =
                call_with_proof(&app, "POST", "/api/vault/grants", &forged, proof, grant()).await;
            assert_eq!(s, StatusCode::FORBIDDEN, "proof {proof:?}");
        }
        let put = Some(serde_json::json!({"value": "attacker-chosen-value"}));
        let (s, _) = call(&app, "PUT", "/api/vault/secrets/demo-secret", &forged, put).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let (s, _) = call(
            &app,
            "DELETE",
            "/api/vault/secrets/demo-secret",
            &forged,
            None,
        )
        .await;
        assert_eq!(s, StatusCode::FORBIDDEN);

        // The real proof works — until the vault locks.
        let (s, _) = call_with_proof(
            &app,
            "POST",
            "/api/vault/grants",
            &t.human,
            Some(&real_proof),
            grant(),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED);
        let (s, _) = call(&app, "POST", "/api/vault/lock", &t.human, None).await;
        assert_eq!(s, StatusCode::NO_CONTENT);
        let (s, _) = call_with_proof(
            &app,
            "POST",
            "/api/vault/grants",
            &t.human,
            Some(&real_proof),
            grant(),
        )
        .await;
        assert_ne!(s, StatusCode::CREATED, "a proof dies with the unlock");
    }

    async fn store_instance(state: &Arc<ServerState>, credential_ref: &str) {
        let record = crate::chat::provider::settings::InstanceRecord {
            id: "deepseek".into(),
            kind: "openai_compatible".into(),
            preset: None,
            label: "DeepSeek".into(),
            base_url: "https://api.example.com/v1".into(),
            origin: "https://api.example.com".into(),
            default_model: Some("m".into()),
            cost_source: "unknown".into(),
            credential_ref: credential_ref.into(),
            ..Default::default()
        };
        state
            .orchestrator
            .neo4j()
            .put_llm_setting(
                "global",
                "instance:deepseek",
                &serde_json::to_string(&record).unwrap(),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn a_provider_grant_must_name_the_instance_key_on_an_existing_instance() {
        let _serial = SERIAL.lock().await;
        let (app, state) = app().await;
        let t = tokens();
        let proof = open_with_secret(&app, &t).await;
        store_instance(&state, "vault:demo-secret").await;
        let grant = |scope: serde_json::Value, secrets: serde_json::Value| {
            Some(serde_json::json!({"secrets": secrets, "scope": scope, "minutes": 60}))
        };
        let provider = |id: &str| serde_json::json!({"kind": "provider", "value": id});
        let names = |n: &[&str]| serde_json::json!({"kind": "names", "names": n});

        // The key the instance is configured with, on that instance: created.
        let (s, body) = call_with_proof(
            &app,
            "POST",
            "/api/vault/grants",
            &t.human,
            Some(&proof),
            grant(provider("deepseek"), names(&["demo-secret"])),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED, "{body}");
        // `all`: refused. Another secret: refused. An unknown instance: 404. claude-code: 404.
        for (scope, secrets, want) in [
            (
                provider("deepseek"),
                serde_json::json!({"kind": "all"}),
                StatusCode::BAD_REQUEST,
            ),
            (
                provider("deepseek"),
                names(&["something-else"]),
                StatusCode::BAD_REQUEST,
            ),
            (
                provider("deepseek"),
                names(&["demo-secret", "something-else"]),
                StatusCode::BAD_REQUEST,
            ),
            (provider("deepseek"), names(&[]), StatusCode::BAD_REQUEST),
            (
                provider("ghost"),
                names(&["demo-secret"]),
                StatusCode::NOT_FOUND,
            ),
            (
                provider("claude-code"),
                names(&["demo-secret"]),
                StatusCode::NOT_FOUND,
            ),
        ] {
            let (s, body) = call_with_proof(
                &app,
                "POST",
                "/api/vault/grants",
                &t.human,
                Some(&proof),
                grant(scope.clone(), secrets.clone()),
            )
            .await;
            assert_eq!(s, want, "{scope} {secrets}: {body}");
        }
        // Other scopes are untouched by the rule.
        let (s, _) = call_with_proof(
            &app,
            "POST",
            "/api/vault/grants",
            &t.human,
            Some(&proof),
            grant(
                serde_json::json!({"kind": "anywhere"}),
                serde_json::json!({"kind": "all"}),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED);
        // An instance with no vault credential has nothing to grant.
        store_instance(&state, "none").await;
        let (s, _) = call_with_proof(
            &app,
            "POST",
            "/api/vault/grants",
            &t.human,
            Some(&proof),
            grant(provider("deepseek"), names(&["demo-secret"])),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn an_agent_token_cannot_manage_the_vault() {
        let _serial = SERIAL.lock().await;
        let (app, _) = app().await;
        let t = tokens();
        let init = serde_json::json!({"passphrase": "correct horse battery staple"});
        let (s, _) = call(
            &app,
            "POST",
            "/api/vault/init",
            &t.agent_session,
            Some(init),
        )
        .await;
        assert_eq!(s, StatusCode::FORBIDDEN);

        open_with_secret(&app, &t).await;
        let grant = serde_json::json!({
            "secrets": {"kind": "all"},
            "scope": {"kind": "anywhere"},
        });
        let (s, _) = call(
            &app,
            "POST",
            "/api/vault/grants",
            &t.agent_session,
            Some(grant),
        )
        .await;
        assert_eq!(
            s,
            StatusCode::FORBIDDEN,
            "an agent must not grant itself secrets"
        );
        let (s, _) = call(&app, "GET", "/api/vault", &t.agent_session, None).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn a_vault_token_opens_the_agent_routes_and_nothing_else() {
        let _serial = SERIAL.lock().await;
        let (app, _) = app().await;
        let t = tokens();
        let (s, _) = call(&app, "GET", "/api/projects", &t.vault_s1, None).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let (s, _) = call(&app, "GET", "/api/vault", &t.vault_s1, None).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let (s, _) = call(&app, "GET", "/api/vault/agent/available", &t.vault_s1, None).await;
        assert_eq!(s, StatusCode::OK);
    }

    #[tokio::test]
    async fn an_agent_reads_a_value_only_under_a_grant_for_its_own_session() {
        let _serial = SERIAL.lock().await;
        let (app, state) = app().await;
        let t = tokens();
        let proof = open_with_secret(&app, &t).await;
        let read = || Some(serde_json::json!({"name": "demo-secret"}));

        let (s, body) = call(&app, "POST", "/api/vault/agent/read", &t.vault_s1, read()).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        assert!(!body.contains(VALUE));

        let grant = serde_json::json!({
            "secrets": {"kind": "names", "names": ["demo-secret"]},
            "scope": {"kind": "session", "value": "session-1"},
            "minutes": 30,
        });
        let (s, _) = call_with_proof(
            &app,
            "POST",
            "/api/vault/grants",
            &t.human,
            Some(&proof),
            Some(grant),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED);

        let (s, body) = call(&app, "POST", "/api/vault/agent/read", &t.vault_s1, read()).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(body, VALUE);
        // Delivered → masked from now on.
        assert_eq!(state.vault.masker().mask(VALUE), "[secret:demo-secret]");

        // The other session is still refused: the session comes from the
        // signature, not from anything the agent sends.
        let (s, body) = call(&app, "POST", "/api/vault/agent/read", &t.vault_s2, read()).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        assert!(!body.contains(VALUE));
    }

    #[tokio::test]
    async fn no_management_response_ever_carries_a_value() {
        let _serial = SERIAL.lock().await;
        let (app, _) = app().await;
        let t = tokens();
        open_with_secret(&app, &t).await;
        let (s, body) = call(&app, "GET", "/api/vault", &t.human, None).await;
        assert_eq!(s, StatusCode::OK);
        assert!(body.contains("demo-secret"));
        assert!(!body.contains(VALUE));
    }

    #[tokio::test]
    async fn a_request_card_can_be_answered_with_an_inline_unlock() {
        let _serial = SERIAL.lock().await;
        let (app, state) = app().await;
        let t = tokens();
        open_with_secret(&app, &t).await;
        state.vault.lock_now();

        let ask = serde_json::json!({"name": "gh-token", "reason": "push the release"});
        let (s, body) = call(
            &app,
            "POST",
            "/api/vault/agent/requests",
            &t.vault_s1,
            Some(ask),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{body}");
        let id = serde_json::from_str::<serde_json::Value>(&body).unwrap()["request_id"]
            .as_str()
            .unwrap()
            .to_string();

        // Wrong passphrase: refused, and the card stays answerable.
        let answer = |pass: &str| {
            Some(serde_json::json!({
                "action": "provide",
                "value": "ghp_0123456789abcdef",
                "passphrase": pass,
            }))
        };
        let path = format!("/api/vault/requests/{id}/answer");
        let (s, _) = call(&app, "POST", &path, &t.human, answer("wrong passphrase!!")).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let (s, body) = call(
            &app,
            "POST",
            &path,
            &t.human,
            answer("correct horse battery staple"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{body}");
        assert!(!body.contains("ghp_0123456789abcdef"));

        let read = Some(serde_json::json!({"name": "gh-token"}));
        let (s, body) = call(&app, "POST", "/api/vault/agent/read", &t.vault_s1, read).await;
        assert_eq!((s, body.as_str()), (StatusCode::OK, "ghp_0123456789abcdef"));
    }
}
