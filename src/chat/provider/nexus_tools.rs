//! `nexus-tools` attached to a native session (task B40; decisions A26, A28, A35).
//!
//! A native provider has no file, no shell and no web unless an MCP server brings them. That
//! server is `nexus-tools` (repository nexus, plan N17 to N27). This module decides, for one
//! session, whether it is attached, with which tools, with which keys, and what the project
//! agreed to.
//!
//! ## What is attached, and how
//!
//! * One process per session, named `nexus` (the name the harness gives its canonical tools
//!   `Read`, `Bash`, `WebFetch`… under), bounded at launch: `--cwd`/`--add-dir` are its whole
//!   file scope, `--tools` the only tools it serves.
//! * The session's tool profile travels in a **signed token** (`NEXUS_TOOLS_PROFILE`, format
//!   `v1.<claims>.<HMAC-SHA256>` of nexus-tools), checked by the server against a key made for
//!   this session alone (`NEXUS_TOOLS_KEY`). Both go in the child's **environment**: never on
//!   argv, and `McpServerSpec`'s `Debug` prints no environment value. The claims are the
//!   session id, the tools the policy can expose and an expiry; a tool outside the token is
//!   neither listed nor run by the server.
//! * The token is **bound to the session and revocable**: it is registered in
//!   [`crate::auth::agent_tokens`] and `revoke_session` (every close path calls it) revokes it.
//!   The server verifies a token by signature and expiry only, so the revocation is enforced on
//!   the host's side of the pipe: [`ToolAccessHooks`] refuses every `nexus` tool call of a
//!   session whose token is no longer live, and closing the session ends the process.
//!
//! ## What the project agreed to
//!
//! * Network tools (`WebFetch`, `WebSearch`, the browser's navigation) are subject to a consent
//!   of the PROJECT that is tied to the ORIGIN (scheme, host, non-default port) of the request
//!   (A28): stored under the project's scope as `tool_origin:<origin>`. A request to an origin
//!   without it is refused before the tool runs, with a typed [`ToolAccessError`].
//! * A search engine with a key is a **tool provider** `tool:<id>`. Its key is read from the
//!   vault under the grant `Provider("tool:<id>")` (A26): never under the session's id, never
//!   under the harness' or a model provider's. A locked vault or a missing grant drops that
//!   engine from the session (`WebSearch` disappears if none is left); no fallback.
//! * The browser (Obscura, N23) is attached only if its executable is found AND an operator
//!   authorised it for the project (`browser` document). Its stealth mode is never passed: this
//!   module takes no argument for the browser.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use chrono::{DateTime, Utc};
use hmac::{digest::KeyInit, Hmac, Mac};
use nexus_claude::agent::{
    CompactionInfo, HookVerdict, McpServerSpec, SessionHooks, ToolCallInfo, ToolPolicy,
    ToolResultInfo, TurnContext, TurnDirective,
};
use nexus_claude::providers::native::{
    nexus_tools_bound, BrowserTools, BROWSER_SERVER, NEXUS_TOOLS_CATALOG, NEXUS_TOOLS_SERVER,
};
use rand_core_06::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Sha256;

use super::endpoint_guard::origin_of;
use super::settings::project_scope;
use crate::neo4j::GraphStore;
use crate::vault::grants::Denied;
use crate::vault::service::ServiceError;
use crate::vault::VaultService;

/// Key prefix of a project's consent to an origin for the network tools (scope `project:<slug>`).
pub const TOOL_ORIGIN_PREFIX: &str = "tool_origin:";
/// Key of the project's authorisation of the browser (scope `project:<slug>`).
pub const BROWSER_KEY: &str = "browser";
/// Key prefix of the search providers (scope `global`).
pub const SEARCH_PROVIDER_PREFIX: &str = "search_provider:";
/// Prefix of the identifier a tool provider is granted under: it cannot be mistaken for a model
/// provider instance or a session.
pub const TOOL_PROVIDER_PREFIX: &str = "tool:";
/// Origin of the Brave Search API, the one engine with a fixed endpoint.
pub const BRAVE_ORIGIN: &str = "https://api.search.brave.com";

/// Why a tool call or a launch is refused. `code` is stable; no message carries a URL beyond its
/// origin, a token or a key.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolAccessError {
    /// The project did not consent to this origin.
    #[error("the project did not consent to the origin {origin}")]
    OriginNotConsented {
        /// Normalised origin.
        origin: String,
    },
    /// The URL is not one a network tool can be judged on.
    #[error("the URL cannot be judged: it has no origin")]
    InvalidUrl,
    /// No search engine of this session has a consented origin.
    #[error("no search engine of this session has a consented origin")]
    SearchNotConsented,
    /// The browser is not authorised for the project.
    #[error("the browser is not authorised for this project")]
    BrowserNotAuthorized,
    /// The session's tool token was revoked (the session is closed or was cut off).
    #[error("the tool token of this session was revoked")]
    TokenRevoked,
    /// The vault is locked: the search key cannot be read now.
    #[error("the vault is locked: the key of tool provider {provider} cannot be read")]
    CredentialsLocked {
        /// Tool provider id.
        provider: String,
    },
    /// The vault holds no key granted to this tool provider.
    #[error("no key is granted to tool provider {provider}")]
    CredentialRequired {
        /// Tool provider id.
        provider: String,
    },
    /// The consent could not be read: refused (never an allow on an ignorance).
    #[error("the project's consents cannot be read")]
    ConsentUnavailable,
}

impl ToolAccessError {
    /// Stable code, for the wire and the logs.
    pub fn code(&self) -> &'static str {
        match self {
            Self::OriginNotConsented { .. } => "tool_origin_not_allowed",
            Self::InvalidUrl => "tool_url_invalid",
            Self::SearchNotConsented => "tool_search_not_allowed",
            Self::BrowserNotAuthorized => "browser_not_authorized",
            Self::TokenRevoked => "tool_token_revoked",
            Self::CredentialsLocked { .. } => "credentials_locked",
            Self::CredentialRequired { .. } => "tool_credential_required",
            Self::ConsentUnavailable => "tool_consent_unavailable",
        }
    }

    /// What the model reads: the code, then the reason. A model-readable refusal is the only
    /// typed channel a `before_tool` hook has (`HookVerdict::Deny { reason }`).
    pub fn reason(&self) -> String {
        format!("{}: {self}", self.code())
    }
}

// ---------------------------------------------------------------------------
// Stored documents
// ---------------------------------------------------------------------------

/// A project's consent to an origin, for the network tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOriginConsent {
    /// Normalised origin consented to.
    pub origin: String,
    /// Who consented (a human login).
    pub consented_by: String,
    /// RFC 3339 time.
    pub consented_at: String,
}

/// An operator's authorisation of the browser for a project.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrowserAuthorization {
    /// Who authorised it (a human login).
    pub authorized_by: String,
    /// RFC 3339 time.
    pub authorized_at: String,
}

/// A search engine, a tool provider. It stores a credential REFERENCE, never a key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchProviderRecord {
    /// Identifier. The grant is `Provider("tool:<id>")`.
    pub id: String,
    /// `brave` (keyed API) or `searxng` (self-hosted, `base_url`).
    pub engine: String,
    /// The SearXNG instance, for `searxng`.
    #[serde(default)]
    pub base_url: Option<String>,
    /// `vault:<name>` or `none`.
    #[serde(default = "no_credential")]
    pub credential_ref: String,
}

fn no_credential() -> String {
    "none".to_owned()
}

impl SearchProviderRecord {
    /// The identifier the vault grant follows.
    pub fn grant_id(&self) -> String {
        format!("{TOOL_PROVIDER_PREFIX}{}", self.id)
    }

    /// The origin the engine is reached at: what a project consents to.
    pub fn origin(&self) -> Option<String> {
        match self.engine.as_str() {
            "brave" => Some(BRAVE_ORIGIN.to_owned()),
            "searxng" => self.base_url.as_deref().and_then(origin_of),
            _ => None,
        }
    }

    fn vault_name(&self) -> Option<&str> {
        self.credential_ref.strip_prefix("vault:")
    }
}

// ---------------------------------------------------------------------------
// Consent reads
// ---------------------------------------------------------------------------

/// What the project agreed to, read when asked (a consent can be revoked mid-session).
#[async_trait]
pub trait ToolConsents: Send + Sync {
    /// Whether the project consented to `origin`.
    async fn origin_consented(&self, project: &str, origin: &str) -> Result<bool, ToolAccessError>;
    /// Whether the project consented to at least one origin: without one, every `WebFetch`
    /// would be refused, so the tool is not offered at all.
    async fn any_origin_consented(&self, project: &str) -> Result<bool, ToolAccessError>;
    /// Whether the browser is authorised for the project.
    async fn browser_authorized(&self, project: &str) -> Result<bool, ToolAccessError>;
    /// The search providers declared on the server.
    async fn search_providers(&self) -> Vec<SearchProviderRecord>;
}

/// [`ToolConsents`] over the graph's settings documents.
pub struct GraphConsents(pub Arc<dyn GraphStore>);

#[async_trait]
impl ToolConsents for GraphConsents {
    async fn origin_consented(&self, project: &str, origin: &str) -> Result<bool, ToolAccessError> {
        let raw = self
            .0
            .get_llm_setting(
                &project_scope(project),
                &format!("{TOOL_ORIGIN_PREFIX}{origin}"),
            )
            .await
            .map_err(|_| ToolAccessError::ConsentUnavailable)?;
        // A document that does not decode, or that names another origin, is no consent.
        Ok(raw
            .and_then(|raw| serde_json::from_str::<ToolOriginConsent>(&raw).ok())
            .is_some_and(|c| c.origin == origin))
    }

    async fn any_origin_consented(&self, project: &str) -> Result<bool, ToolAccessError> {
        let documents = self
            .0
            .list_llm_settings(&project_scope(project), TOOL_ORIGIN_PREFIX)
            .await
            .map_err(|_| ToolAccessError::ConsentUnavailable)?;
        // The same reading as [`Self::origin_consented`]: a document counts only if it decodes
        // and names the origin of its key.
        Ok(documents.iter().any(|(key, raw)| {
            serde_json::from_str::<ToolOriginConsent>(raw)
                .is_ok_and(|c| key.strip_prefix(TOOL_ORIGIN_PREFIX).unwrap_or(key) == c.origin)
        }))
    }

    async fn browser_authorized(&self, project: &str) -> Result<bool, ToolAccessError> {
        let raw = self
            .0
            .get_llm_setting(&project_scope(project), BROWSER_KEY)
            .await
            .map_err(|_| ToolAccessError::ConsentUnavailable)?;
        Ok(raw.is_some_and(|raw| serde_json::from_str::<BrowserAuthorization>(&raw).is_ok()))
    }

    async fn search_providers(&self) -> Vec<SearchProviderRecord> {
        self.0
            .list_llm_settings(super::settings::GLOBAL, SEARCH_PROVIDER_PREFIX)
            .await
            .unwrap_or_default()
            .iter()
            .filter_map(|(_, raw)| serde_json::from_str(raw).ok())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The gate: which call is refused
// ---------------------------------------------------------------------------

/// Tools of the browser that load a page: their destination is an origin to consent to.
const BROWSER_NAVIGATION: &[&str] = &["browser_navigate", "browser_tab_new"];

fn browser_tool_of(name: &str) -> Option<&str> {
    name.strip_prefix(&format!("mcp__{BROWSER_SERVER}__"))
}

/// Judges one tool call against what the project agreed to. `Ok` means "not this gate's
/// business or allowed"; the harness' own policy still applies after.
///
/// `search_origins` are the origins of the engines this session was launched with.
pub async fn check_call(
    consents: &dyn ToolConsents,
    project: Option<&str>,
    search_origins: &[String],
    call: &ToolCallInfo,
) -> Result<(), ToolAccessError> {
    let canonical = call.canonical.as_deref();
    let browser_tool = browser_tool_of(&call.name);
    if canonical == Some("WebFetch") {
        let url = call.input.get("url").and_then(Value::as_str);
        let mut needed =
            vec![origin_of(url.unwrap_or_default()).ok_or(ToolAccessError::InvalidUrl)?];
        // nexus-tools upgrades `http://` to `https://` before it connects: the request really
        // goes to the secure origin, which must be consented to as well.
        if needed[0].starts_with("http://") {
            needed.push(needed[0].replacen("http://", "https://", 1));
        }
        return require_origins(consents, project, &needed).await;
    }
    if canonical == Some("WebSearch") {
        let project = project.ok_or(ToolAccessError::SearchNotConsented)?;
        for origin in search_origins {
            if consents.origin_consented(project, origin).await? {
                return Ok(());
            }
        }
        return Err(ToolAccessError::SearchNotConsented);
    }
    if let Some(tool) = browser_tool {
        let project = project.ok_or(ToolAccessError::BrowserNotAuthorized)?;
        if !consents.browser_authorized(project).await? {
            return Err(ToolAccessError::BrowserNotAuthorized);
        }
        if BROWSER_NAVIGATION.contains(&tool) {
            let url = call.input.get("url").and_then(Value::as_str);
            let origin = origin_of(url.unwrap_or_default()).ok_or(ToolAccessError::InvalidUrl)?;
            return require_origins(consents, Some(project), &[origin]).await;
        }
    }
    Ok(())
}

async fn require_origins(
    consents: &dyn ToolConsents,
    project: Option<&str>,
    origins: &[String],
) -> Result<(), ToolAccessError> {
    for origin in origins {
        let allowed = match project {
            Some(project) => consents.origin_consented(project, origin).await?,
            // No project, no consent (A28): the same default as a model endpoint.
            None => false,
        };
        if !allowed {
            return Err(ToolAccessError::OriginNotConsented {
                origin: origin.clone(),
            });
        }
    }
    Ok(())
}

/// The `before_tool` gate of a native session: the revocation of its token, then the consent of
/// its project, then (if both pass) the session's other hooks.
pub struct ToolAccessHooks {
    inner: Option<Arc<dyn SessionHooks>>,
    consents: Arc<dyn ToolConsents>,
    session_id: String,
    project: Option<String>,
    search_origins: Vec<String>,
}

impl ToolAccessHooks {
    /// A gate over `inner` (the knowledge-graph hooks, when the session has them).
    pub fn new(
        inner: Option<Arc<dyn SessionHooks>>,
        consents: Arc<dyn ToolConsents>,
        session_id: impl Into<String>,
        project: Option<String>,
        search_origins: Vec<String>,
    ) -> Self {
        Self {
            inner,
            consents,
            session_id: session_id.into(),
            project,
            search_origins,
        }
    }

    fn is_nexus_tool(call: &ToolCallInfo) -> bool {
        call.name
            .starts_with(&format!("mcp__{NEXUS_TOOLS_SERVER}__"))
    }
}

#[async_trait]
impl SessionHooks for ToolAccessHooks {
    async fn before_tool(&self, call: &ToolCallInfo) -> HookVerdict {
        if Self::is_nexus_tool(call) && !crate::auth::agent_tokens::tools_live(&self.session_id) {
            return HookVerdict::Deny {
                reason: ToolAccessError::TokenRevoked.reason(),
            };
        }
        if let Err(refusal) = check_call(
            self.consents.as_ref(),
            self.project.as_deref(),
            &self.search_origins,
            call,
        )
        .await
        {
            tracing::warn!(
                session_id = %self.session_id,
                code = refusal.code(),
                "network tool refused"
            );
            return HookVerdict::Deny {
                reason: refusal.reason(),
            };
        }
        match &self.inner {
            Some(inner) => inner.before_tool(call).await,
            None => HookVerdict::Continue,
        }
    }

    async fn after_tool(&self, result: &ToolResultInfo) -> Option<String> {
        match &self.inner {
            Some(inner) => inner.after_tool(result).await,
            None => None,
        }
    }

    async fn before_compaction(&self, info: &CompactionInfo) -> Option<String> {
        match &self.inner {
            Some(inner) => inner.before_compaction(info).await,
            None => None,
        }
    }

    async fn before_turn(&self, ctx: &TurnContext) -> TurnDirective {
        match &self.inner {
            Some(inner) => inner.before_turn(ctx).await,
            None => TurnDirective::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// The signed profile token
// ---------------------------------------------------------------------------

/// The key and the token of one session. Neither is ever printed.
pub struct SessionToolsToken {
    /// `NEXUS_TOOLS_KEY`: 64 hexadecimal characters (32 random bytes), made for this session.
    key: String,
    /// `NEXUS_TOOLS_PROFILE`.
    token: String,
}

impl std::fmt::Debug for SessionToolsToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SessionToolsToken(<redacted>)")
    }
}

#[derive(Serialize)]
struct Claims<'a> {
    sid: &'a str,
    tools: &'a [String],
    exp: u64,
}

/// Signs the token of a profile with `key` (the bytes of the key string, as nexus-tools reads
/// them): `v1.<base64url(claims)>.<base64url(HMAC-SHA256(key, "v1.<base64url(claims)>"))>`.
pub fn sign_profile(key: &str, session_id: &str, tools: &[String], exp: u64) -> String {
    let body = serde_json::to_vec(&Claims {
        sid: session_id,
        tools,
        exp,
    })
    .unwrap_or_default();
    let signed = format!("v1.{}", URL_SAFE_NO_PAD.encode(body));
    let mut mac = Hmac::<Sha256>::new_from_slice(key.as_bytes())
        .unwrap_or_else(|_| unreachable!("HMAC accepts any key length"));
    mac.update(signed.as_bytes());
    format!(
        "{signed}.{}",
        URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes())
    )
}

impl SessionToolsToken {
    /// A fresh key and the token it signs.
    pub fn mint(session_id: &str, tools: &[String], exp: u64) -> Self {
        let mut bytes = [0u8; 32];
        rand_core_06::OsRng.fill_bytes(&mut bytes);
        let key = hex::encode(bytes);
        let token = sign_profile(&key, session_id, tools, exp);
        Self { key, token }
    }

    /// The environment of the server process.
    fn environment(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("NEXUS_TOOLS_KEY".to_owned(), self.key.clone()),
            ("NEXUS_TOOLS_PROFILE".to_owned(), self.token.clone()),
        ])
    }
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

/// Where the tools of one session come from.
#[derive(Debug, Clone)]
pub struct NexusToolsConfig {
    /// The `nexus-tools` executable, by its absolute path: what
    /// [`super::native_factory::runnable_nexus_tools`] resolved at this opening.
    pub program: PathBuf,
    /// The browser, when its executable was found. Present does not mean authorised.
    pub browser: Option<BrowserTools>,
}

/// The `nexus-tools` tools a session's policy gives it (the profile signed into its token).
///
/// Everything the policy does not remove: a `deny` without argument, and plan mode (read-only
/// tools only, unless the session can leave it: `nexus_tools_bound`). An `allow` list narrows the
/// profile ONLY if it names some `nexus-tools` tool; one that names only other servers' tools
/// (the default) says nothing about these. `ceiling` is the most the mode can be raised to.
pub fn profile_for(policy: &ToolPolicy, ceiling: Option<&ToolPolicy>) -> Vec<String> {
    let names_a_nexus_tool = policy.allow.iter().any(|pattern| {
        NEXUS_TOOLS_CATALOG.iter().any(|(tool, _)| {
            pattern.matches_closed(tool, None)
                || pattern.matches_closed(&format!("mcp__{NEXUS_TOOLS_SERVER}__{tool}"), None)
        })
    });
    let mut effective = policy.clone();
    if !names_a_nexus_tool {
        effective.allow.clear();
    }
    nexus_tools_bound(&effective, ceiling, true)
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// What a session needs to be given its tools.
pub struct AttachInput<'a> {
    /// Session id (the token's `sid`).
    pub session_id: &'a str,
    /// Working directory: the file scope.
    pub cwd: &'a Path,
    /// Extra directories of the file scope.
    pub extra_dirs: &'a [PathBuf],
    /// The session's neutral tool policy.
    pub policy: &'a ToolPolicy,
    /// The ceiling of its mode, when it has one.
    pub ceiling: Option<&'a ToolPolicy>,
    /// The project, derived by the server (A28).
    pub project: Option<&'a str>,
    /// Seconds the token lives.
    pub ttl_secs: u64,
    /// Now.
    pub now: DateTime<Utc>,
}

/// The servers to add to the session, and what the gate needs to know about them.
#[derive(Debug, Default)]
pub struct Attachment {
    /// MCP servers by name (`nexus`, and `browser` when it is offered).
    pub servers: BTreeMap<String, McpServerSpec>,
    /// Origins of the engines the server was launched with.
    pub search_origins: Vec<String>,
    /// The tools of the token, canonical names.
    pub tools: Vec<String>,
    /// Engines left out, and why (no key granted, vault locked, origin not consented).
    pub skipped_engines: Vec<(String, ToolAccessError)>,
}

/// Resolves the key of a search provider for THAT provider: the vault is asked under the grant
/// `Provider("tool:<id>")` and under nothing else. `None`: the engine needs no key.
pub fn resolve_search_key(
    vault: Option<&VaultService>,
    provider: &SearchProviderRecord,
    now: DateTime<Utc>,
) -> Result<Option<zeroize::Zeroizing<String>>, ToolAccessError> {
    let Some(name) = provider.vault_name() else {
        return Ok(None);
    };
    let grant_id = provider.grant_id();
    let vault = vault.ok_or_else(|| ToolAccessError::CredentialsLocked {
        provider: grant_id.clone(),
    })?;
    match vault.read_for_provider(name, &grant_id, now) {
        Ok(value) => Ok(Some(value)),
        Err(ServiceError::Denied(Denied::NoGrant | Denied::UnknownSecret)) => {
            Err(ToolAccessError::CredentialRequired { provider: grant_id })
        }
        // Locked, throttled, unreadable: the key cannot be read now. Never "no key".
        Err(_) => Err(ToolAccessError::CredentialsLocked { provider: grant_id }),
    }
}

/// Decides and builds what a native session is given. Returns an empty [`Attachment`] when the
/// policy exposes no `nexus-tools` tool.
pub async fn attach(
    config: &NexusToolsConfig,
    consents: &dyn ToolConsents,
    vault: Option<&VaultService>,
    input: AttachInput<'_>,
) -> Attachment {
    let mut attachment = Attachment::default();
    // 1. The profile: what the session's policy can expose (the harness' own rule).
    let mut tools = profile_for(input.policy, input.ceiling);
    // 2. The engines: consented origin AND a key granted to the provider. Env var names are
    //    the only thing on argv; the value goes in the environment.
    let mut engine_args: Vec<String> = Vec::new();
    let mut secrets: BTreeMap<String, String> = BTreeMap::new();
    if let Some(project) = input.project {
        for (index, provider) in consents.search_providers().await.into_iter().enumerate() {
            let Some(origin) = provider.origin() else {
                continue;
            };
            match consents.origin_consented(project, &origin).await {
                Ok(true) => {}
                Ok(false) => {
                    attachment.skipped_engines.push((
                        provider.id.clone(),
                        ToolAccessError::OriginNotConsented { origin },
                    ));
                    continue;
                }
                Err(refusal) => {
                    attachment
                        .skipped_engines
                        .push((provider.id.clone(), refusal));
                    continue;
                }
            }
            match (
                provider.engine.as_str(),
                resolve_search_key(vault, &provider, input.now),
            ) {
                ("brave", Ok(Some(key))) => {
                    let variable = format!("NEXUS_SEARCH_KEY_{index}");
                    engine_args.push("--search-engine".to_owned());
                    engine_args.push(format!("brave:{variable}"));
                    secrets.insert(variable, key.as_str().to_owned());
                }
                // A keyed engine without a key is no engine.
                ("brave", Ok(None)) => attachment.skipped_engines.push((
                    provider.id.clone(),
                    ToolAccessError::CredentialRequired {
                        provider: provider.grant_id(),
                    },
                )),
                ("searxng", Ok(_)) => {
                    if let Some(url) = provider.base_url.as_deref() {
                        engine_args.push("--search-engine".to_owned());
                        engine_args.push(format!("searxng:{url}"));
                    }
                }
                (_, Err(refusal)) => attachment
                    .skipped_engines
                    .push((provider.id.clone(), refusal)),
                _ => continue,
            }
            attachment.search_origins.push(origin);
        }
    }
    // Without an engine the server has no `WebSearch`: do not name it in the token either.
    if engine_args.is_empty() {
        tools.retain(|tool| tool != "WebSearch");
        attachment.search_origins.clear();
    }
    // `WebFetch` is offered only to a project that consented to some origin (each call is then
    // judged on ITS origin by [`ToolAccessHooks`]). Without a project, or with an unreadable
    // store, it is not offered: never an allow on an ignorance.
    let fetch_consented = match input.project {
        Some(project) => consents
            .any_origin_consented(project)
            .await
            .unwrap_or(false),
        None => false,
    };
    if !fetch_consented {
        tools.retain(|tool| tool != "WebFetch");
    }
    if tools.is_empty() {
        return attachment;
    }
    // 3. The token: signed, bound to the session, expiring, revocable.
    let exp = u64::try_from(input.now.timestamp())
        .unwrap_or(0)
        .saturating_add(input.ttl_secs);
    let token = SessionToolsToken::mint(input.session_id, &tools, exp);
    crate::auth::agent_tokens::register_tools(input.session_id);
    let mut args = vec!["--cwd".to_owned(), input.cwd.display().to_string()];
    for dir in input.extra_dirs {
        args.push("--add-dir".to_owned());
        args.push(dir.display().to_string());
    }
    args.push("--tools".to_owned());
    args.push(tools.join(","));
    args.extend(engine_args);
    let mut env = token.environment();
    env.extend(secrets);
    attachment.servers.insert(
        NEXUS_TOOLS_SERVER.to_owned(),
        McpServerSpec::Stdio {
            command: config.program.display().to_string(),
            args,
            env,
        },
    );
    attachment.tools = tools;
    // 4. The browser: detected AND authorised for the project. No argument is ever given to it.
    if let (Some(browser), Some(project)) = (&config.browser, input.project) {
        if browser.is_installed() && consents.browser_authorized(project).await.unwrap_or(false) {
            attachment
                .servers
                .insert(BROWSER_SERVER.to_owned(), browser.server());
        }
    }
    attachment
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::vault::grants::{GrantScope, SecretSelector};
    use chrono::Duration;
    use nexus_claude::agent::{PolicyMode, ToolCategory};
    use serde_json::json;

    const PASS: &str = "correct horse battery staple";
    const SECRET: &str = "BSA-secret-value-never-printed-0042";

    fn call(name: &str, canonical: Option<&str>, input: Value) -> ToolCallInfo {
        ToolCallInfo {
            id: None,
            name: name.to_owned(),
            canonical: canonical.map(str::to_owned),
            category: ToolCategory::Web,
            input,
        }
    }

    fn fetch(url: &str) -> ToolCallInfo {
        call(
            "mcp__nexus__WebFetch",
            Some("WebFetch"),
            json!({ "url": url }),
        )
    }

    async fn store_consent(graph: &MockGraphStore, project: &str, origin: &str) {
        graph
            .put_llm_setting(
                &project_scope(project),
                &format!("{TOOL_ORIGIN_PREFIX}{origin}"),
                &serde_json::to_string(&ToolOriginConsent {
                    origin: origin.into(),
                    consented_by: "me@example.com".into(),
                    consented_at: "2026-10-08T10:00:00Z".into(),
                })
                .unwrap(),
            )
            .await
            .unwrap();
    }

    async fn authorize_browser(graph: &MockGraphStore, project: &str) {
        graph
            .put_llm_setting(
                &project_scope(project),
                BROWSER_KEY,
                &serde_json::to_string(&BrowserAuthorization {
                    authorized_by: "me@example.com".into(),
                    authorized_at: "2026-10-08T10:00:00Z".into(),
                })
                .unwrap(),
            )
            .await
            .unwrap();
    }

    async fn store_provider(graph: &MockGraphStore, record: &SearchProviderRecord) {
        graph
            .put_llm_setting(
                super::super::settings::GLOBAL,
                &format!("{SEARCH_PROVIDER_PREFIX}{}", record.id),
                &serde_json::to_string(record).unwrap(),
            )
            .await
            .unwrap();
    }

    fn brave(id: &str) -> SearchProviderRecord {
        SearchProviderRecord {
            id: id.into(),
            engine: "brave".into(),
            base_url: None,
            credential_ref: "vault:brave-key".into(),
        }
    }

    async fn vault_with_key(grant_to: Option<&str>) -> Arc<VaultService> {
        let vault = VaultService::ephemeral();
        vault.init(PASS.into(), Duration::hours(1)).await.unwrap();
        let now = Utc::now();
        vault.put("brave-key", SECRET, None, now).unwrap();
        if let Some(id) = grant_to {
            vault
                .grant(
                    SecretSelector::Names(["brave-key".to_string()].into()),
                    GrantScope::Provider(id.to_string()),
                    Duration::hours(1),
                    None,
                    now,
                )
                .unwrap();
        }
        vault
    }

    fn config() -> NexusToolsConfig {
        NexusToolsConfig {
            program: PathBuf::from("/opt/nexus/nexus-tools"),
            browser: None,
        }
    }

    fn input<'a>(
        session: &'a str,
        policy: &'a ToolPolicy,
        project: Option<&'a str>,
    ) -> AttachInput<'a> {
        AttachInput {
            session_id: session,
            cwd: Path::new("/work/app"),
            extra_dirs: &[],
            policy,
            ceiling: None,
            project,
            ttl_secs: 3600,
            now: Utc::now(),
        }
    }

    fn ask() -> ToolPolicy {
        ToolPolicy::new(PolicyMode::Ask)
    }

    fn stdio(
        attachment: &Attachment,
        name: &str,
    ) -> (String, Vec<String>, BTreeMap<String, String>) {
        match attachment.servers.get(name) {
            Some(McpServerSpec::Stdio { command, args, env }) => {
                (command.clone(), args.clone(), env.clone())
            }
            other => panic!("expected a stdio server `{name}`, got {other:?}"),
        }
    }

    // ---- (3) the consent of the project, by origin -------------------------------------

    #[tokio::test]
    async fn webfetch_to_an_origin_without_consent_is_refused_with_a_typed_error() {
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph.clone());
        let refusal = check_call(
            &consents,
            Some("proj"),
            &[],
            &fetch("https://docs.rs/x?q=1"),
        )
        .await
        .unwrap_err();
        assert_eq!(
            refusal,
            ToolAccessError::OriginNotConsented {
                origin: "https://docs.rs".into()
            }
        );
        assert_eq!(refusal.code(), "tool_origin_not_allowed");
        assert!(refusal.reason().starts_with("tool_origin_not_allowed:"));
        assert!(
            !refusal.reason().contains("q=1"),
            "no path or query in a refusal"
        );
    }

    #[tokio::test]
    async fn a_consent_is_per_origin_and_per_project() {
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph.clone());
        store_consent(&graph, "proj", "https://docs.rs").await;
        // The consented origin passes, whatever the path (and the default port).
        for url in ["https://docs.rs/a", "HTTPS://Docs.RS:443/b?x=1"] {
            check_call(&consents, Some("proj"), &[], &fetch(url))
                .await
                .unwrap();
        }
        // Another host, a sub-domain, another port, another scheme, another project: refused.
        for url in [
            "https://evil.docs.rs/",
            "https://docs.rs.evil.com/",
            "https://docs.rs:8443/",
            "http://docs.rs/",
        ] {
            assert!(
                check_call(&consents, Some("proj"), &[], &fetch(url))
                    .await
                    .is_err(),
                "{url}"
            );
        }
        assert!(
            check_call(&consents, Some("other"), &[], &fetch("https://docs.rs/"))
                .await
                .is_err()
        );
        // No project, no consent.
        assert!(check_call(&consents, None, &[], &fetch("https://docs.rs/"))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn an_http_fetch_needs_the_secure_origin_it_is_upgraded_to() {
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph.clone());
        store_consent(&graph, "proj", "http://docs.rs").await;
        // nexus-tools connects to https://docs.rs: consenting to the plain origin is not enough.
        let refusal = check_call(&consents, Some("proj"), &[], &fetch("http://docs.rs/"))
            .await
            .unwrap_err();
        assert_eq!(
            refusal,
            ToolAccessError::OriginNotConsented {
                origin: "https://docs.rs".into()
            }
        );
        store_consent(&graph, "proj", "https://docs.rs").await;
        check_call(&consents, Some("proj"), &[], &fetch("http://docs.rs/"))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn a_url_without_an_origin_is_refused_and_a_forged_document_is_no_consent() {
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph.clone());
        for bad in ["", "not a url", "file:///etc/passwd"] {
            let refusal = check_call(&consents, Some("proj"), &[], &fetch(bad)).await;
            assert!(refusal.is_err(), "{bad:?}");
        }
        // A document stored under an origin's key that names another origin consents to nothing.
        graph
            .put_llm_setting(
                &project_scope("proj"),
                &format!("{TOOL_ORIGIN_PREFIX}https://docs.rs"),
                &serde_json::to_string(&ToolOriginConsent {
                    origin: "https://evil.example".into(),
                    consented_by: "x".into(),
                    consented_at: "t".into(),
                })
                .unwrap(),
            )
            .await
            .unwrap();
        graph
            .put_llm_setting(
                &project_scope("proj"),
                "tool_origin:https://garbage.example",
                "{",
            )
            .await
            .unwrap();
        for url in ["https://docs.rs/", "https://garbage.example/"] {
            assert!(check_call(&consents, Some("proj"), &[], &fetch(url))
                .await
                .is_err());
        }
    }

    #[tokio::test]
    async fn an_unreadable_store_refuses_instead_of_allowing() {
        struct Broken;
        #[async_trait]
        impl ToolConsents for Broken {
            async fn origin_consented(&self, _: &str, _: &str) -> Result<bool, ToolAccessError> {
                Err(ToolAccessError::ConsentUnavailable)
            }
            async fn any_origin_consented(&self, _: &str) -> Result<bool, ToolAccessError> {
                Err(ToolAccessError::ConsentUnavailable)
            }
            async fn browser_authorized(&self, _: &str) -> Result<bool, ToolAccessError> {
                Err(ToolAccessError::ConsentUnavailable)
            }
            async fn search_providers(&self) -> Vec<SearchProviderRecord> {
                Vec::new()
            }
        }
        assert_eq!(
            check_call(&Broken, Some("p"), &[], &fetch("https://docs.rs/")).await,
            Err(ToolAccessError::ConsentUnavailable)
        );
    }

    #[tokio::test]
    async fn websearch_needs_a_consented_engine_origin() {
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph.clone());
        let search = call(
            "mcp__nexus__WebSearch",
            Some("WebSearch"),
            json!({"query": "rust"}),
        );
        let origins = vec![BRAVE_ORIGIN.to_owned()];
        assert_eq!(
            check_call(&consents, Some("proj"), &origins, &search).await,
            Err(ToolAccessError::SearchNotConsented)
        );
        assert_eq!(
            check_call(&consents, Some("proj"), &[], &search).await,
            Err(ToolAccessError::SearchNotConsented)
        );
        store_consent(&graph, "proj", BRAVE_ORIGIN).await;
        check_call(&consents, Some("proj"), &origins, &search)
            .await
            .unwrap();
        assert_eq!(
            check_call(&consents, None, &origins, &search).await,
            Err(ToolAccessError::SearchNotConsented)
        );
    }

    #[tokio::test]
    async fn another_servers_tool_named_webfetch_is_not_judged_here() {
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph);
        // No canonical name: it is not nexus-tools' WebFetch (the harness gives that name to
        // the `nexus` server only).
        let other = call(
            "mcp__other__WebFetch",
            None,
            json!({"url": "https://x.example/"}),
        );
        check_call(&consents, Some("p"), &[], &other).await.unwrap();
    }

    // ---- the hook: refused before the tool runs ---------------------------------------

    struct CountingInner(std::sync::atomic::AtomicUsize);

    #[async_trait]
    impl SessionHooks for CountingInner {
        async fn before_tool(&self, _: &ToolCallInfo) -> HookVerdict {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            HookVerdict::AddContext("inner".into())
        }
    }

    #[tokio::test]
    async fn the_hook_denies_before_the_inner_hooks_and_passes_the_rest_on() {
        let graph = Arc::new(MockGraphStore::new());
        crate::auth::agent_tokens::register_tools("hook-s1");
        let inner = Arc::new(CountingInner(Default::default()));
        let hooks = ToolAccessHooks::new(
            Some(inner.clone()),
            Arc::new(GraphConsents(graph.clone())),
            "hook-s1",
            Some("proj".into()),
            Vec::new(),
        );
        let verdict = hooks.before_tool(&fetch("https://docs.rs/")).await;
        let HookVerdict::Deny { reason } = verdict else {
            panic!("a denial was expected, got {verdict:?}");
        };
        assert!(reason.starts_with("tool_origin_not_allowed:"));
        assert_eq!(inner.0.load(std::sync::atomic::Ordering::SeqCst), 0);
        store_consent(&graph, "proj", "https://docs.rs").await;
        assert_eq!(
            hooks.before_tool(&fetch("https://docs.rs/")).await,
            HookVerdict::AddContext("inner".into())
        );
        crate::auth::agent_tokens::revoke_session("hook-s1");
    }

    // ---- (1) the token: signed, bound, revocable, never on argv -------------------------

    #[tokio::test]
    async fn a_revoked_token_stops_every_nexus_tool_call_of_the_session() {
        let graph = Arc::new(MockGraphStore::new());
        crate::auth::agent_tokens::register_tools("rev-s1");
        let hooks = ToolAccessHooks::new(
            None,
            Arc::new(GraphConsents(graph)),
            "rev-s1",
            Some("proj".into()),
            Vec::new(),
        );
        let read = call(
            "mcp__nexus__Read",
            Some("Read"),
            json!({"file_path": "/work/a"}),
        );
        assert_eq!(hooks.before_tool(&read).await, HookVerdict::Continue);
        // Closing the session (every close path calls this) revokes the token.
        crate::auth::agent_tokens::revoke_session("rev-s1");
        let HookVerdict::Deny { reason } = hooks.before_tool(&read).await else {
            panic!("a revoked session must not run a nexus tool");
        };
        assert!(reason.starts_with("tool_token_revoked:"));
        // Another server's tools are not this token's business.
        let po = call("mcp__project-orchestrator__task", None, json!({}));
        assert_eq!(hooks.before_tool(&po).await, HookVerdict::Continue);
    }

    #[test]
    fn the_token_is_the_one_the_real_server_accepts_and_it_names_exactly_the_profile() {
        let tools: Vec<String> = vec!["Read".into(), "Grep".into()];
        let exp = u64::try_from(Utc::now().timestamp()).unwrap() + 600;
        let minted = SessionToolsToken::mint("sess-7", &tools, exp);
        let key = nexus_tools::SigningKey::new(minted.key.clone().into_bytes()).unwrap();
        let now = u64::try_from(Utc::now().timestamp()).unwrap();
        let profile = nexus_tools::verify(&key, &minted.token, now).expect("a valid token");
        assert_eq!(profile.session_id, "sess-7", "bound to the session");
        assert!(profile.allows("Read") && profile.allows("Grep"));
        assert!(!profile.allows("Bash") && !profile.allows("Write"));
        // Expired: refused by the real verifier.
        assert!(nexus_tools::verify(&key, &minted.token, exp + 1).is_err());
        // Another session's key does not verify it.
        let other = nexus_tools::SigningKey::new(
            SessionToolsToken::mint("x", &tools, exp).key.into_bytes(),
        )
        .unwrap();
        assert!(nexus_tools::verify(&other, &minted.token, now).is_err());
        // A tampered claim set does not verify.
        let forged = sign_profile("0".repeat(64).as_str(), "sess-7", &["Bash".into()], exp);
        assert!(nexus_tools::verify(&key, &forged, now).is_err());
    }

    #[tokio::test]
    async fn a_tool_outside_the_profile_is_neither_listed_nor_run_by_the_server() {
        use nexus_tools::{Server, Session, ToolRegistry};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
        let tools: Vec<String> = vec!["echo".into()];
        let exp = u64::try_from(Utc::now().timestamp()).unwrap() + 600;
        let minted = SessionToolsToken::mint("sess-8", &tools, exp);
        let key = nexus_tools::SigningKey::new(minted.key.clone().into_bytes()).unwrap();
        let profile = nexus_tools::verify(
            &key,
            &minted.token,
            u64::try_from(Utc::now().timestamp()).unwrap(),
        )
        .unwrap();
        // Two tools that count how many times they really ran.
        struct Counting(&'static str, Arc<AtomicUsize>);
        #[async_trait]
        impl nexus_tools::Tool for Counting {
            fn name(&self) -> &str {
                self.0
            }
            fn description(&self) -> &str {
                "counts its runs"
            }
            fn input_schema(&self) -> Value {
                json!({"type": "object"})
            }
            fn annotations(&self) -> nexus_tools::Annotations {
                nexus_tools::Annotations::default()
            }
            async fn call(
                &self,
                _: &nexus_tools::CallContext,
                _: Value,
            ) -> nexus_tools::ToolResult {
                self.1.fetch_add(1, Ordering::SeqCst);
                nexus_tools::ToolResult::ok("ran")
            }
        }
        let (allowed_runs, forbidden_runs) =
            (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let registry = ToolRegistry::new()
            .with(Counting("echo", allowed_runs.clone()))
            .with(Counting("write", forbidden_runs.clone()));
        let (mut to_server, server_in) = tokio::io::duplex(1 << 16);
        let (server_out, from_server) = tokio::io::duplex(1 << 16);
        let served = tokio::spawn(nexus_tools::serve_lines(
            Arc::new(Server::new(registry)),
            Session::new(profile),
            tokio::io::BufReader::new(server_in),
            server_out,
        ));
        let mut from_server = tokio::io::BufReader::new(from_server);
        let mut answers = Vec::new();
        for (id, method, params) in [
            (1, "tools/list", json!({})),
            (2, "tools/call", json!({"name": "write", "arguments": {}})),
            (3, "tools/call", json!({"name": "echo", "arguments": {}})),
        ] {
            let request = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
            to_server
                .write_all((request.to_string() + "\n").as_bytes())
                .await
                .unwrap();
            let mut answer = String::new();
            tokio::time::timeout(
                std::time::Duration::from_secs(10),
                from_server.read_line(&mut answer),
            )
            .await
            .expect("an answer in time")
            .unwrap();
            answers.push(answer);
        }
        drop(to_server);
        let _ = served.await;
        let text = answers.join("");
        assert!(
            text.contains("\"echo\""),
            "the profile's tool is listed: {text}"
        );
        assert!(
            !answers[0].contains("\"name\":\"write\""),
            "not listed: {}",
            answers[0]
        );
        assert!(
            answers[1].contains("error"),
            "the direct call is refused: {}",
            answers[1]
        );
        assert_eq!(
            allowed_runs.load(Ordering::SeqCst),
            1,
            "the profile's tool ran"
        );
        assert_eq!(
            forbidden_runs.load(Ordering::SeqCst),
            0,
            "a tool outside the profile never runs, even when called directly"
        );
    }

    #[tokio::test]
    async fn attach_gives_the_server_a_signed_profile_in_the_environment_and_never_on_argv() {
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph);
        let policy =
            ToolPolicy::from_patterns(PolicyMode::Ask, &["Read", "Grep"], &[] as &[&str]).unwrap();
        let attachment = attach(
            &config(),
            &consents,
            None,
            input("att-s1", &policy, Some("proj")),
        )
        .await;
        let (command, args, env) = stdio(&attachment, "nexus");
        assert_eq!(command, "/opt/nexus/nexus-tools");
        assert_eq!(attachment.tools, ["Read", "Grep"]);
        let tools_at = args.iter().position(|a| a == "--tools").unwrap();
        assert_eq!(args[tools_at + 1], "Read,Grep");
        assert!(
            !args.contains(&"--trust-harness".to_owned()),
            "a signed profile, not trust"
        );
        assert!(!args.contains(&"--unrestricted".to_owned()));
        let (key, token) = (&env["NEXUS_TOOLS_KEY"], &env["NEXUS_TOOLS_PROFILE"]);
        assert!(key.len() >= 32 && token.starts_with("v1."));
        for arg in &args {
            assert!(!arg.contains(token.as_str()) && !arg.contains(key.as_str()));
        }
        // The real verifier reads the same profile out of it.
        let verifier = nexus_tools::SigningKey::new(key.clone().into_bytes()).unwrap();
        let profile = nexus_tools::verify(
            &verifier,
            token,
            u64::try_from(Utc::now().timestamp()).unwrap(),
        )
        .unwrap();
        assert_eq!(profile.session_id, "att-s1");
        assert!(profile.allows("Read") && !profile.allows("Bash"));
        // The token is registered as live for the session, and no journal line has it.
        assert!(crate::auth::agent_tokens::tools_live("att-s1"));
        let printed = format!("{attachment:?}");
        assert!(!printed.contains(token.as_str()) && !printed.contains(key.as_str()));
        crate::auth::agent_tokens::revoke_session("att-s1");
        assert!(!crate::auth::agent_tokens::tools_live("att-s1"));
    }

    /// WebFetch is offered only to a project that consented to some origin: without one every
    /// call would be refused. A forged document (key and origin that differ) is no consent, an
    /// unreadable store offers nothing, and a session without a project never gets it.
    #[tokio::test]
    async fn webfetch_is_offered_only_once_the_project_consented_to_an_origin() {
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph.clone());
        let policy = ask();
        let offered = |attachment: &Attachment| attachment.tools.contains(&"WebFetch".to_owned());
        let before = attach(
            &config(),
            &consents,
            None,
            input("wf-1", &policy, Some("proj")),
        )
        .await;
        assert!(!offered(&before) && before.tools.contains(&"Read".to_owned()));
        graph
            .put_llm_setting(
                &project_scope("proj"),
                &format!("{TOOL_ORIGIN_PREFIX}https://docs.rs"),
                &serde_json::to_string(&ToolOriginConsent {
                    origin: "https://evil.example".into(),
                    consented_by: "me@example.com".into(),
                    consented_at: "2026-10-08T10:00:00Z".into(),
                })
                .unwrap(),
            )
            .await
            .unwrap();
        let forged = attach(
            &config(),
            &consents,
            None,
            input("wf-2", &policy, Some("proj")),
        )
        .await;
        assert!(!offered(&forged), "a forged document is no consent");
        store_consent(&graph, "proj", "https://docs.rs").await;
        let after = attach(
            &config(),
            &consents,
            None,
            input("wf-3", &policy, Some("proj")),
        )
        .await;
        assert!(offered(&after));
        let elsewhere = attach(
            &config(),
            &consents,
            None,
            input("wf-4", &policy, Some("other")),
        )
        .await;
        let no_project = attach(&config(), &consents, None, input("wf-5", &policy, None)).await;
        assert!(!offered(&elsewhere) && !offered(&no_project));
        for sid in ["wf-1", "wf-2", "wf-3", "wf-4", "wf-5"] {
            crate::auth::agent_tokens::revoke_session(sid);
        }
    }

    #[tokio::test]
    async fn two_sessions_get_two_keys_and_a_policy_that_exposes_nothing_gets_no_server() {
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph);
        let policy = ask();
        let a = attach(&config(), &consents, None, input("keys-a", &policy, None)).await;
        let b = attach(&config(), &consents, None, input("keys-b", &policy, None)).await;
        assert_ne!(
            stdio(&a, "nexus").2["NEXUS_TOOLS_KEY"],
            stdio(&b, "nexus").2["NEXUS_TOOLS_KEY"]
        );
        crate::auth::agent_tokens::revoke_session("keys-a");
        crate::auth::agent_tokens::revoke_session("keys-b");
        // A policy that denies every tool leaves nothing to attach.
        let deny_all: Vec<&str> = NEXUS_TOOLS_CATALOG.iter().map(|(tool, _)| *tool).collect();
        let none = ToolPolicy::from_patterns(PolicyMode::Ask, &[] as &[&str], &deny_all).unwrap();
        let empty = attach(&config(), &consents, None, input("keys-c", &none, None)).await;
        assert!(empty.servers.is_empty() && empty.tools.is_empty());
        assert!(
            !crate::auth::agent_tokens::tools_live("keys-c"),
            "nothing minted"
        );
    }

    #[test]
    fn the_default_allow_list_of_the_backend_does_not_hide_the_nexus_tools_but_a_named_one_narrows()
    {
        let all: Vec<String> = NEXUS_TOOLS_CATALOG
            .iter()
            .map(|(t, _)| (*t).to_owned())
            .collect();
        let none: &[&str] = &[];
        // The default permission config allows only the project-orchestrator tools.
        let default =
            ToolPolicy::from_patterns(PolicyMode::Ask, &["mcp__project-orchestrator__*"], none)
                .unwrap();
        assert_eq!(
            profile_for(&default, None),
            all,
            "an allow list that names none of them says nothing about them"
        );
        // An allow list that names some narrows the profile to those.
        let named = ToolPolicy::from_patterns(
            PolicyMode::Ask,
            &["Read", "mcp__nexus__Grep", "Bash(git *)"],
            none,
        )
        .unwrap();
        assert_eq!(profile_for(&named, None), ["Read", "Grep", "Bash"]);
        // A deny without argument removes the tool, a deny with one does not.
        let denied = ToolPolicy::from_patterns(
            PolicyMode::Ask,
            none,
            &["Bash", "WebFetch(domain:evil.example)"],
        )
        .unwrap();
        let profile = profile_for(&denied, None);
        assert!(!profile.contains(&"Bash".to_owned()) && profile.contains(&"WebFetch".to_owned()));
        // Plan mode with nothing above it: the process may need to leave plan, so the profile
        // keeps the edits (the harness refuses them while in plan mode).
        let plan = ToolPolicy::from_patterns(PolicyMode::PlanOnly, none, none).unwrap();
        assert_eq!(profile_for(&plan, None), all);
        // And the harness, under the setting the factory applies, really offers the tool.
        use nexus_claude::providers::native::{ToolEntry, ToolRegistry};
        let bash = ToolEntry::mcp("nexus", "Bash", String::new(), json!({}), false);
        assert!(ToolRegistry::is_exposed(
            &bash,
            &default,
            super::super::native_factory::STRICT_TOOL_EXPOSURE
        ));
        assert!(
            !ToolRegistry::is_exposed(&bash, &default, true),
            "the strict default is what would have hidden it"
        );
    }

    // ---- (4) the key of a search engine, for the tool provider -----------------------------

    #[tokio::test]
    async fn the_search_key_is_read_for_the_tool_provider_never_for_the_session_or_the_harness() {
        let provider = brave("brave-main");
        // Granted to the session, to the harness' model-provider instance, to a lookalike: no key.
        for wrong in ["srch-s1", "native-local", "brave-main", "tool:other"] {
            let vault = vault_with_key(Some(wrong)).await;
            assert_eq!(
                resolve_search_key(Some(&vault), &provider, Utc::now()).err(),
                Some(ToolAccessError::CredentialRequired {
                    provider: "tool:brave-main".into()
                }),
                "a grant to {wrong}"
            );
        }
        // Granted to the tool provider's id: resolved.
        let vault = vault_with_key(Some("tool:brave-main")).await;
        let key = resolve_search_key(Some(&vault), &provider, Utc::now())
            .unwrap()
            .unwrap();
        assert_eq!(key.as_str(), SECRET);
        // A locked vault is a typed error, not "no key".
        vault.lock_now();
        assert_eq!(
            resolve_search_key(Some(&vault), &provider, Utc::now()).err(),
            Some(ToolAccessError::CredentialsLocked {
                provider: "tool:brave-main".into()
            })
        );
        assert!(matches!(
            resolve_search_key(None, &provider, Utc::now()),
            Err(ToolAccessError::CredentialsLocked { .. })
        ));
    }

    #[tokio::test]
    async fn an_engine_is_launched_with_its_key_in_the_environment_only() {
        let graph = Arc::new(MockGraphStore::new());
        store_provider(&graph, &brave("brave-main")).await;
        store_consent(&graph, "proj", BRAVE_ORIGIN).await;
        let consents = GraphConsents(graph);
        let vault = vault_with_key(Some("tool:brave-main")).await;
        let policy = ask();
        let attachment = attach(
            &config(),
            &consents,
            Some(&vault),
            input("srch-s2", &policy, Some("proj")),
        )
        .await;
        let (_, args, env) = stdio(&attachment, "nexus");
        let at = args.iter().position(|a| a == "--search-engine").unwrap();
        let variable = args[at + 1].strip_prefix("brave:").unwrap();
        assert_eq!(
            env[variable], SECRET,
            "the value travels in the environment"
        );
        assert!(args.iter().all(|a| !a.contains(SECRET)), "never on argv");
        assert!(!format!("{attachment:?}").contains(SECRET));
        assert!(attachment.tools.contains(&"WebSearch".to_owned()));
        assert_eq!(attachment.search_origins, [BRAVE_ORIGIN]);
        crate::auth::agent_tokens::revoke_session("srch-s2");
    }

    #[tokio::test]
    async fn an_engine_without_a_grant_or_consent_is_left_out_and_websearch_goes_with_it() {
        let graph = Arc::new(MockGraphStore::new());
        store_provider(&graph, &brave("brave-main")).await;
        let consents = GraphConsents(graph.clone());
        let policy = ask();
        // Grant to the session id only; origin consented: no key for the tool provider.
        store_consent(&graph, "proj", BRAVE_ORIGIN).await;
        let vault = vault_with_key(Some("srch-s3")).await;
        let attachment = attach(
            &config(),
            &consents,
            Some(&vault),
            input("srch-s3", &policy, Some("proj")),
        )
        .await;
        assert_eq!(attachment.skipped_engines.len(), 1);
        assert_eq!(
            attachment.skipped_engines[0].1.code(),
            "tool_credential_required"
        );
        assert!(!attachment.tools.contains(&"WebSearch".to_owned()));
        let (_, args, env) = stdio(&attachment, "nexus");
        assert!(!args.contains(&"--search-engine".to_owned()));
        assert!(env.keys().all(|k| !k.starts_with("NEXUS_SEARCH_KEY")));
        crate::auth::agent_tokens::revoke_session("srch-s3");
        // Key granted properly but the project never consented to the engine's origin.
        let vault = vault_with_key(Some("tool:brave-main")).await;
        let fresh = Arc::new(MockGraphStore::new());
        store_provider(&fresh, &brave("brave-main")).await;
        let attachment = attach(
            &config(),
            &GraphConsents(fresh),
            Some(&vault),
            input("srch-s4", &policy, Some("proj")),
        )
        .await;
        assert_eq!(
            attachment.skipped_engines[0].1.code(),
            "tool_origin_not_allowed"
        );
        assert!(!attachment.tools.contains(&"WebSearch".to_owned()));
        crate::auth::agent_tokens::revoke_session("srch-s4");
    }

    // ---- (5) the browser -----------------------------------------------------------------

    fn fake_obscura() -> (tempfile::TempDir, BrowserTools) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("obscura");
        std::fs::write(&path, "#!/bin/sh\n").unwrap();
        (dir, BrowserTools::new(path))
    }

    #[tokio::test]
    async fn the_browser_is_offered_only_if_detected_and_authorised_for_the_project() {
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph.clone());
        let (_dir, installed) = fake_obscura();
        let policy = ask();
        let with = |browser: Option<BrowserTools>| NexusToolsConfig {
            program: PathBuf::from("/opt/nexus/nexus-tools"),
            browser,
        };
        // Detected, not authorised.
        let a = attach(
            &with(Some(installed.clone())),
            &consents,
            None,
            input("br-1", &policy, Some("proj")),
        )
        .await;
        assert!(!a.servers.contains_key("browser"));
        // Authorised, not detected (no executable configured; or one that is not there).
        authorize_browser(&graph, "proj").await;
        let b = attach(
            &with(None),
            &consents,
            None,
            input("br-2", &policy, Some("proj")),
        )
        .await;
        assert!(!b.servers.contains_key("browser"));
        let gone = BrowserTools::new("/nonexistent/obscura");
        let b2 = attach(
            &with(Some(gone)),
            &consents,
            None,
            input("br-2b", &policy, Some("proj")),
        )
        .await;
        assert!(!b2.servers.contains_key("browser"));
        // Authorised for another project only.
        let c = attach(
            &with(Some(installed.clone())),
            &consents,
            None,
            input("br-3", &policy, Some("other")),
        )
        .await;
        assert!(!c.servers.contains_key("browser"));
        // No project at all.
        let d = attach(
            &with(Some(installed.clone())),
            &consents,
            None,
            input("br-4", &policy, None),
        )
        .await;
        assert!(!d.servers.contains_key("browser"));
        // Detected AND authorised: offered, and never in stealth mode.
        let e = attach(
            &with(Some(installed)),
            &consents,
            None,
            input("br-5", &policy, Some("proj")),
        )
        .await;
        let (_, args, env) = stdio(&e, "browser");
        assert_eq!(args, ["mcp"]);
        assert!(args.iter().all(|a| !a.contains("stealth")));
        assert!(env
            .get("OBSCURA_ALLOW_PRIVATE_NETWORK")
            .is_some_and(|v| v == "0"));
        assert!(!env.keys().any(|k| k.to_lowercase().contains("stealth")));
        for sid in ["br-1", "br-2", "br-2b", "br-3", "br-4", "br-5"] {
            crate::auth::agent_tokens::revoke_session(sid);
        }
    }

    #[tokio::test]
    async fn a_browser_call_is_refused_when_the_project_withdrew_the_authorisation() {
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph.clone());
        let snapshot = call("mcp__browser__browser_snapshot", None, json!({}));
        assert_eq!(
            check_call(&consents, Some("proj"), &[], &snapshot).await,
            Err(ToolAccessError::BrowserNotAuthorized)
        );
        authorize_browser(&graph, "proj").await;
        check_call(&consents, Some("proj"), &[], &snapshot)
            .await
            .unwrap();
        // Authorised, but navigating is a request to an origin: consent applies too.
        let navigate = call(
            "mcp__browser__browser_navigate",
            None,
            json!({"url": "https://news.example/a"}),
        );
        assert_eq!(
            check_call(&consents, Some("proj"), &[], &navigate).await,
            Err(ToolAccessError::OriginNotConsented {
                origin: "https://news.example".into()
            })
        );
        store_consent(&graph, "proj", "https://news.example").await;
        check_call(&consents, Some("proj"), &[], &navigate)
            .await
            .unwrap();
    }

    // ---- misc ----------------------------------------------------------------------------

    #[test]
    fn the_stored_documents_round_trip_and_the_provider_origin_follows_its_engine() {
        let searx = SearchProviderRecord {
            id: "home".into(),
            engine: "searxng".into(),
            base_url: Some("https://Search.Home.example:8443/".into()),
            credential_ref: "none".into(),
        };
        assert_eq!(
            searx.origin().as_deref(),
            Some("https://search.home.example:8443")
        );
        assert_eq!(brave("b").origin().as_deref(), Some(BRAVE_ORIGIN));
        let unknown = SearchProviderRecord {
            engine: "mystery".into(),
            ..brave("m")
        };
        assert_eq!(unknown.origin(), None);
        let decoded: SearchProviderRecord =
            serde_json::from_str(r#"{"id":"x","engine":"searxng","base_url":"http://s"}"#).unwrap();
        assert_eq!(decoded.credential_ref, "none");
    }
}
