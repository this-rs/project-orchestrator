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
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::collections::{BTreeMap, BTreeSet};
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

use super::endpoint_guard::{origin_of, validate_endpoint, EndpointPolicy};
use super::settings::project_scope;
#[allow(unused_imports)]
use crate::neo4j::traits::*;
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
/// Origin Obscura's keyless `serp` reaches (DuckDuckGo's HTML results page): what a project
/// consents to for the `obscura` engine.
pub const OBSCURA_SEARCH_ORIGIN: &str = "https://html.duckduckgo.com";
/// Id of the **built-in** `obscura` engine (decision 1e9d7501): when the server's nexus-tools
/// embeds Obscura, a keyless search engine exists without anyone declaring it. It is not stored;
/// an engine declared with the `obscura` engine replaces it. Each project still consents to its
/// origin ([`OBSCURA_SEARCH_ORIGIN`]) like to any other: the first search is refused with that
/// origin named until the project allows it.
pub const BUILTIN_OBSCURA_ID: &str = "obscura-builtin";

/// The built-in engine's record (see [`BUILTIN_OBSCURA_ID`]).
pub fn builtin_obscura_engine() -> SearchProviderRecord {
    SearchProviderRecord {
        id: BUILTIN_OBSCURA_ID.to_owned(),
        engine: "obscura".to_owned(),
        base_url: None,
        credential_ref: "none".to_owned(),
    }
}

/// Whether the `nexus-tools` at `program` embeds Obscura: its `--capabilities` line says
/// `"obscura": true`. A build without the feature, an older one without the flag, one that fails or does not
/// answer in 5 s, or one that prints something else: no. The answer is kept per path and file
/// modification time, so a replaced executable is asked again.
pub async fn embeds_obscura(program: &Path) -> bool {
    use std::sync::Mutex;
    use std::time::SystemTime;
    static SEEN: Mutex<BTreeMap<PathBuf, (Option<SystemTime>, bool)>> = Mutex::new(BTreeMap::new());
    let modified = std::fs::metadata(program).and_then(|m| m.modified()).ok();
    if let Some((at, answer)) = SEEN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(program)
    {
        if *at == modified {
            return *answer;
        }
    }
    let mut command = tokio::process::Command::new(program);
    command
        .arg("--capabilities")
        .env_clear()
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    let answer =
        match tokio::time::timeout(std::time::Duration::from_secs(5), command.output()).await {
            Ok(Ok(output)) if output.status.success() => {
                serde_json::from_slice::<Value>(&output.stdout)
                    .ok()
                    .and_then(|caps| caps.get("obscura").and_then(Value::as_bool))
                    .unwrap_or(false)
            }
            _ => false,
        };
    SEEN.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(program.to_path_buf(), (modified, answer));
    answer
}

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
    /// The consent could not be written: the answer `always` is not kept.
    #[error("the project's consent to {origin} cannot be written")]
    ConsentNotWritable {
        /// The origin the consent was about.
        origin: String,
    },
    /// The engine's executable is not configured or not found on this server.
    #[error("the executable of search engine {engine} is not installed on this server")]
    EngineNotInstalled {
        /// Engine name (`obscura`).
        engine: String,
    },
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
            Self::ConsentNotWritable { .. } => "tool_consent_not_writable",
            Self::EngineNotInstalled { .. } => "tool_engine_not_installed",
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

/// Records the project's consent to `origin` — the ONE durable place a destination is consented
/// to, whoever the person answered through (N23b).
///
/// This is the very document `PUT /api/projects/{slug}/network-tools/origins` writes, under the
/// very key [`TOOL_ORIGIN_PREFIX`] + origin, and the only writer of it: the settings route and an
/// `always` answered in the chat both come here. There is no second store, so a consent given in
/// the chat is one the route reads back, revokes, and lists — and one a later session's gate finds
/// through [`ToolConsents::origin_consented`].
///
/// `origin` must already be normalised (`scheme://host:port`) and `consented_by` names the PERSON
/// who answered: the HTTP route takes it from the human-only session it refuses agents on, the
/// answer path from the session's owner. Never a model, never a tool.
pub async fn record_origin_consent(
    graph: &dyn GraphStore,
    project: &str,
    origin: &str,
    consented_by: &str,
) -> Result<ToolOriginConsent, ToolAccessError> {
    let consent = ToolOriginConsent {
        origin: origin.to_owned(),
        consented_by: consented_by.to_owned(),
        consented_at: chrono::Utc::now().to_rfc3339(),
    };
    let document =
        serde_json::to_string(&consent).map_err(|_| ToolAccessError::ConsentNotWritable {
            origin: origin.to_owned(),
        })?;
    graph
        .put_llm_setting(
            &project_scope(project),
            &format!("{TOOL_ORIGIN_PREFIX}{origin}"),
            &document,
        )
        .await
        .map_err(|_| ToolAccessError::ConsentNotWritable {
            origin: origin.to_owned(),
        })?;
    Ok(consent)
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
    /// `brave` (keyed API), `searxng` (self-hosted, `base_url`) or `obscura` (keyless, its
    /// executable configured on the server: [`NexusToolsConfig::obscura`]).
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
            "obscura" => Some(OBSCURA_SEARCH_ORIGIN.to_owned()),
            "searxng" => self.base_url.as_deref().and_then(origin_of),
            _ => None,
        }
    }

    fn vault_name(&self) -> Option<&str> {
        self.credential_ref.strip_prefix("vault:")
    }
}

/// Key of the order a person chose for the search engines (scope `global`), written by
/// `PUT /api/chat/search-engines`. `nexus-tools` tries its engines in the order of its
/// `--search-engine` flags, so this order is the order a `WebSearch` queries them in.
pub const SEARCH_ENGINE_ORDER_KEY: &str = "search_engine_order";

/// The order a person chose for the search engines: ids, first tried first.
///
/// An engine the list does not name comes after those it names, in the order it had without the
/// list (by id): no document — every installation before this one — is today's order, and an
/// engine declared after the choice goes last. An id the list names that no engine has (one
/// removed since) is ignored.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchEngineOrder {
    /// Engine ids, first tried first.
    pub order: Vec<String>,
    /// Who chose it (a human login).
    #[serde(default)]
    pub ordered_by: String,
    /// RFC 3339 time.
    #[serde(default)]
    pub ordered_at: String,
}

/// The stored order, if any. An unreadable or undecodable document is no order: the engines keep
/// their order by id, the order they had before anyone chose one.
pub async fn read_search_engine_order(graph: &dyn GraphStore) -> Vec<String> {
    graph
        .get_llm_setting(super::settings::GLOBAL, SEARCH_ENGINE_ORDER_KEY)
        .await
        .ok()
        .flatten()
        .and_then(|raw| serde_json::from_str::<SearchEngineOrder>(&raw).ok())
        .map(|o| o.order)
        .unwrap_or_default()
}

/// Puts `engines` in the chosen `order`. A STABLE sort on the rank alone: the engines the order
/// does not name keep the order they came in (by id), after the named ones.
pub fn order_search_providers(engines: &mut [SearchProviderRecord], order: &[String]) {
    if order.is_empty() {
        return;
    }
    engines.sort_by_key(|e| {
        order
            .iter()
            .position(|id| *id == e.id)
            .unwrap_or(usize::MAX)
    });
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
    /// The search providers declared on the server, first tried first.
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

    /// In the order a person chose ([`SEARCH_ENGINE_ORDER_KEY`]); without one, by id (the order
    /// the store lists them in): the order of the `--search-engine` flags, so of the queries.
    async fn search_providers(&self) -> Vec<SearchProviderRecord> {
        let mut engines: Vec<SearchProviderRecord> = self
            .0
            .list_llm_settings(super::settings::GLOBAL, SEARCH_PROVIDER_PREFIX)
            .await
            .unwrap_or_default()
            .iter()
            .filter_map(|(_, raw)| serde_json::from_str(raw).ok())
            .collect();
        order_search_providers(&mut engines, &read_search_engine_order(&*self.0).await);
        engines
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
    check_call_with(consents, project, search_origins, &BTreeSet::new(), call).await
}

/// The origin nexus-tools really connects to: it upgrades `http://` to `https://` first.
fn reached(origin: &str) -> String {
    origin.replacen("http://", "https://", 1)
}

/// [`check_call`] where `let_through` holds origins the caller already decided to allow for this
/// call (Rock'n roll, [`ToolAccessHooks::with_trust`]), written as nexus-tools reaches them.
pub async fn check_call_with(
    consents: &dyn ToolConsents,
    project: Option<&str>,
    search_origins: &[String],
    let_through: &BTreeSet<String>,
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
        return require_origins(consents, project, let_through, &needed).await;
    }
    if canonical == Some("WebSearch") {
        // EVERY engine of the session must be allowed, not one of them: nexus-tools tries its
        // engines in order and falls back to the next (`search/mod.rs`), so the engine that
        // answers is not the one a check can name. An engine left unconsented would be reached
        // through the fallback of one that is. The refusal names the first one to allow (the
        // built-in engine's, before any consent), as a fetch to a new origin does.
        if search_origins.is_empty() {
            return Err(ToolAccessError::SearchNotConsented);
        }
        return require_origins(consents, project, let_through, search_origins).await;
    }
    if let Some(tool) = browser_tool {
        let project = project.ok_or(ToolAccessError::BrowserNotAuthorized)?;
        if !consents.browser_authorized(project).await? {
            return Err(ToolAccessError::BrowserNotAuthorized);
        }
        if BROWSER_NAVIGATION.contains(&tool) {
            let url = call.input.get("url").and_then(Value::as_str);
            let origin = origin_of(url.unwrap_or_default()).ok_or(ToolAccessError::InvalidUrl)?;
            return require_origins(consents, Some(project), let_through, &[origin]).await;
        }
    }
    Ok(())
}

async fn require_origins(
    consents: &dyn ToolConsents,
    project: Option<&str>,
    let_through: &BTreeSet<String>,
    origins: &[String],
) -> Result<(), ToolAccessError> {
    for origin in origins {
        if let_through.contains(&reached(origin)) {
            continue;
        }
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

/// Whether the session is in `trust` (Rock'n roll) RIGHT NOW. Read at each call: a person can
/// take the session out of trust, or back in, while it runs.
pub type TrustProbe = Arc<dyn Fn() -> bool + Send + Sync>;
/// Tells the person something the session did on its own. Resolves to whether the text was
/// handed to the session's event stream: a notice that was lost is sent again next time, and the
/// call waits for it, so the notice comes before the tool's own events.
pub type TrustNotice =
    Arc<dyn Fn(String) -> futures::future::BoxFuture<'static, bool> + Send + Sync>;

/// The origins a call names, as the gate judges them (N23b): the page a fetch loads or a browser
/// opens. The SUBJECT of a consent question, and what an `always` answer records.
///
/// Asserted by the harness: computed from the CALL, by this one function, on both sides of the
/// question (the gate that asks and the answer path that records). Never read from what the
/// model says about what it wants — a model does not formulate its own request for access.
///
/// Empty for a call that names no destination: a call without a usable `url`, or a search, whose
/// engines come from the server's configuration — a search is not left unanswered for that
/// ([`search_destinations`]: the question is then about the engines themselves).
///
/// nexus-tools upgrades `http://` to `https://` before it connects and the gate judges both
/// forms, so both are returned: a consent that covered only one of them would ask again for the
/// other the moment it was given.
pub fn destinations_of(name: &str, canonical: Option<&str>, input: &Value) -> Vec<String> {
    let names_a_page = canonical == Some("WebFetch")
        || browser_tool_of(name).is_some_and(|tool| BROWSER_NAVIGATION.contains(&tool));
    if !names_a_page {
        return Vec::new();
    }
    let Some(url) = input.get("url").and_then(Value::as_str) else {
        return Vec::new();
    };
    let Some(origin) = origin_of(url) else {
        return Vec::new();
    };
    let secure = reached(&origin);
    if secure == origin {
        vec![origin]
    } else {
        vec![origin, secure]
    }
}

/// The origins a consent question about this call is about: the endpoint nexus-tools really
/// connects to ([`destinations_of`], `https://`), which is also the subject the harness remembers
/// a `session` answer against (`tool\u{0}subject`). `None`: the call names no destination.
pub fn consent_subject(name: &str, canonical: Option<&str>, input: &Value) -> Option<String> {
    let first = destinations_of(name, canonical, input).into_iter().next()?;
    Some(reached(&first))
}

/// The origins a SEARCH would reach: every engine of the session, as nexus-tools reaches them
/// (`https://`), deduplicated.
///
/// A search names no destination of its own — its engines come from the server's configuration —
/// so [`destinations_of`] is empty for it and, without this, its refusal stayed a flat
/// `tool_origin_not_allowed` a person could never answer. But there IS something to decide: this
/// project, this search engine. And something to decide about every engine at once, because
/// nexus-tools tries them in order and falls back to the next, so any of them may be reached.
/// That is why the question names them all and an `always` answer records them all: a consent
/// covering only the first would be asked again the moment the second engine answered.
///
/// Asserted by the harness, never by the model: the list is the one the session was launched
/// with (`ToolAccessHooks::search_origins`, the gate's own), read back per session
/// ([`remember_search_origins`]).
pub fn search_destinations(search_origins: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(search_origins.len());
    for origin in search_origins {
        let origin = reached(origin);
        if !out.contains(&origin) {
            out.push(origin);
        }
    }
    out
}

/// The search origins of the sessions alive in this process, by session id: what the gate checks
/// and what a question about a search is about, kept where both can read it (the gate lives in
/// the hooks, the question is decorated in the runtime).
///
/// Written when a session attaches its `nexus-tools` ([`remember_search_origins`]), forgotten
/// when it closes ([`forget_search_origins`]). A session with no entry has no engine: a search it
/// attempted was refused before any engine ran.
static SESSION_SEARCH_ORIGINS: std::sync::OnceLock<
    std::sync::RwLock<std::collections::HashMap<String, Vec<String>>>,
> = std::sync::OnceLock::new();

fn session_search_origins_registry(
) -> &'static std::sync::RwLock<std::collections::HashMap<String, Vec<String>>> {
    SESSION_SEARCH_ORIGINS.get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()))
}

/// Records the engines this session was launched with.
pub fn remember_search_origins(session_id: &str, search_origins: &[String]) {
    session_search_origins_registry()
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(session_id.to_string(), search_origins.to_vec());
}

/// The engines this session was launched with (`[]`: none, or a session that closed).
pub fn search_origins_of(session_id: &str) -> Vec<String> {
    session_search_origins_registry()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(session_id)
        .cloned()
        .unwrap_or_default()
}

/// Forgets them when the session closes.
pub fn forget_search_origins(session_id: &str) {
    session_search_origins_registry()
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .remove(session_id);
}

/// Whether Rock'n roll opens the network of a session: the session's access is normal AND a
/// PERSON put it in trust (and it has not left it). Never the mode alone: the runner, a protocol
/// executor and a delegated task run in `bypassPermissions` with nobody in front of them, and a
/// read-only session has no `Bash` (its denial wins over trust), so an open network would be a
/// channel for what it reads. The same two conditions as the lift of the file scope
/// ([`lifts_file_scope`]).
pub fn trust_opens_network(access: super::policy::SessionAccess, chosen_by_person: bool) -> bool {
    access == super::policy::SessionAccess::Normal && chosen_by_person
}

/// The posture an origin must pass for Rock'n roll to reach it without a consent of the project:
/// the strict one of the endpoint guard (https only; nothing private, link-local, metadata or
/// reserved), with no relaxation. nexus-tools' own fetch refuses the same classes.
fn trust_origin_policy() -> EndpointPolicy {
    EndpointPolicy {
        allow_insecure_loopback: false,
        ..EndpointPolicy::default()
    }
}

/// Whether `origin` names the machine itself: `localhost` (or a name under it) or a loopback
/// literal, IPv4-mapped IPv6 included. The endpoint guard accepts these on purpose, for a local
/// inference server a person configured; a fetch an agent chose is another matter.
fn names_the_machine_itself(origin: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(origin) else {
        return true;
    };
    let Some(host) = url.host_str() else {
        return true;
    };
    let host = host
        .trim_matches(|c| c == '[' || c == ']')
        .trim_end_matches('.')
        .to_ascii_lowercase();
    host == "localhost"
        || host.ends_with(".localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.to_canonical().is_loopback())
}

/// The `before_tool` gate of a native session: the revocation of its token, then the consent of
/// its project, then (if both pass) the session's other hooks.
///
/// In Rock'n roll no consent is asked for: a PUBLIC origin is allowed on the spot and the person
/// is told ([`Self::with_trust`]); who counts as being in Rock'n roll is [`trust_opens_network`].
/// Nothing is written to the project's consent document, which stays an act of a person, and
/// the permission is the session's, for as long as it stays in trust.
pub struct ToolAccessHooks {
    inner: Option<Arc<dyn SessionHooks>>,
    consents: Arc<dyn ToolConsents>,
    session_id: String,
    project: Option<String>,
    search_origins: Vec<String>,
    /// Whether the session is in trust now; `None`: never (the default).
    trust: Option<TrustProbe>,
    /// Where the person is told what trust allowed.
    notice: Option<TrustNotice>,
    /// The origins already announced: said once per origin, not at every call.
    announced: std::sync::Mutex<std::collections::BTreeSet<String>>,
    /// The directories the session was launched with (cwd, then extra ones): what its file
    /// tools are held to when a session launched `--unscoped` left trust.
    scope_dirs: Option<(PathBuf, Vec<PathBuf>)>,
    /// Whether a PERSON is in front of this session ([`Self::with_person_in_front`]): the only
    /// condition under which a destination the project has not consented to becomes a QUESTION
    /// instead of a refusal (N23b).
    ///
    /// `false`, the default, and the truth for every session nobody is in front of — an agent
    /// token, a runner, `delegate_task`, a protocol executor, a sub-agent: those refuse, exactly
    /// as before, and no question is even emitted. So a model can never be the one that answers
    /// its own request for network access.
    person_in_front: bool,
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
            trust: None,
            notice: None,
            announced: Default::default(),
            scope_dirs: None,
            person_in_front: false,
        }
    }

    /// Lets a session in trust reach public origins without the project's consent, telling the
    /// person through `notice`. `in_trust` is read at each call.
    #[must_use]
    pub fn with_trust(mut self, in_trust: TrustProbe, notice: TrustNotice) -> Self {
        self.trust = Some(in_trust);
        self.notice = Some(notice);
        self
    }

    /// States that a PERSON is in front of this session (N23b): a destination the project has not
    /// consented to becomes a QUESTION, through the harness's own permission channel
    /// (`HookVerdict::RequireConsent`), instead of a refusal.
    ///
    /// The caller decides from what it knows of the session — a signed-in person, no runner, no
    /// parent session — and never from anything the model says. Without this call the gate refuses
    /// as it always did, so forgetting it fails closed.
    #[must_use]
    pub fn with_person_in_front(mut self, person_in_front: bool) -> Self {
        self.person_in_front = person_in_front;
        self
    }

    /// What the gate asks a person about `refusal`, or the refusal to keep (N23b).
    ///
    /// A question only when ALL of these hold: the refusal is a destination without the project's
    /// consent, the call NAMES that destination ([`destinations_of`], or for a search the engines
    /// of the session — [`search_destinations`]), and a person is in front of the session.
    /// Everything else keeps the flat refusal of before.
    fn consent_question(&self, refusal: &ToolAccessError, named: &[String]) -> Option<HookVerdict> {
        let ToolAccessError::OriginNotConsented { origin } = refusal else {
            return None;
        };
        if !self.person_in_front || !named.iter().any(|destination| destination == origin) {
            return None;
        }
        Some(HookVerdict::RequireConsent {
            subject: reached(origin),
            reason: refusal.reason(),
        })
    }

    /// What trust makes of a refusal: the origin to let through when the session is in Rock'n roll
    /// and the refusal is an origin without consent; `None` keeps the refusal. The origin is the
    /// one nexus-tools REACHES (`http://` is upgraded to `https://`), which is also the one the
    /// person is told about.
    ///
    /// Trust spares the person a question; it does not open the machine's own network. A public
    /// origin only: the endpoint guard's strict posture (https, nothing private, link-local,
    /// metadata or reserved) and no loopback literal or `localhost`, the same classes
    /// nexus-tools' own fetch refuses (`web/ssrf.rs`). A broader reach would announce an access
    /// the tool then refuses.
    async fn allowed_by_trust(&self, refusal: &ToolAccessError) -> Option<String> {
        let in_trust = self.trust.as_ref()?;
        if !in_trust() {
            return None;
        }
        let ToolAccessError::OriginNotConsented { origin } = refusal else {
            return None;
        };
        let reached = reached(origin);
        if names_the_machine_itself(&reached) {
            return None;
        }
        if let Err(why) = validate_endpoint(&reached, &trust_origin_policy()).await {
            tracing::warn!(
                session_id = %self.session_id,
                refusal = ?why,
                "Rock'n roll does not cover this origin: refused"
            );
            return None;
        }
        Some(reached)
    }

    /// Tells the person `origin` was let through, once per origin, and traces it on the server.
    /// The origin counts as told only once the notice was delivered: a notice that was lost is
    /// sent again at the next call.
    async fn announce(&self, origin: &str) {
        tracing::warn!(
            session_id = %self.session_id,
            origin,
            "network origin opened by Rock'n roll (no project consent)"
        );
        let Some(notice) = &self.notice else {
            return;
        };
        let first = self
            .announced
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(origin.to_owned());
        if !first {
            return;
        }
        let delivered = notice(format!(
            "Rock'n roll: network access to {origin} was allowed automatically for this \
             session. It is not saved as a consent of the project."
        ))
        .await;
        if !delivered {
            self.announced
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(origin);
        }
    }

    /// The session's directories, to confine its file tools once it leaves trust.
    #[must_use]
    pub fn with_scope_dirs(mut self, cwd: PathBuf, extra_dirs: Vec<PathBuf>) -> Self {
        self.scope_dirs = Some((cwd, extra_dirs));
        self
    }

    /// Why a file tool call of a session launched `--unscoped` that LEFT trust is refused: its
    /// path resolves outside the session's directories. Resolved by nexus-tools' own resolver
    /// (`files::Scope`: symlinks and `..` followed component by component), so the backend and
    /// the server agree on what "inside" means. `None`: allowed (not a file tool, no path given:
    /// the working directory, or inside). Fails closed when the directories are unknown or gone.
    /// `Bash` is not judged here: in a mode other than trust each command is asked for.
    /// Known limit: a `cd` the shell made while in trust survives (nexus-tools keeps the shell's
    /// directory across calls, and an unscoped scope accepted it), so the next commands start
    /// there; outside trust each of them is asked for, and the file tools are judged by the path
    /// they name, whatever the shell's directory. The session's temporary directory (command
    /// outputs) is one of its directories ([`session_temp_dir`]).
    fn outside_scope(&self, call: &ToolCallInfo) -> Option<String> {
        let tool = call
            .canonical
            .as_deref()
            .or_else(|| call.name.rsplit("__").next())
            .unwrap_or_default();
        let key = match tool {
            "Read" | "Write" | "Edit" => "file_path",
            "NotebookEdit" => "notebook_path",
            "Glob" | "Grep" => "path",
            _ => return None,
        };
        let given = call
            .input
            .get(key)
            .and_then(Value::as_str)
            .filter(|path| !path.trim().is_empty())?;
        let refused = || {
            Some(format!(
                "this session left trust mode: its file tools are held to its own directories \
                 again, and `{given}` is outside them"
            ))
        };
        let Some((cwd, extra)) = &self.scope_dirs else {
            return refused();
        };
        match nexus_tools::files::Scope::new(cwd, extra.iter().cloned()) {
            Ok(scope) if scope.resolve(given).is_ok() => None,
            _ => refused(),
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
        // Launched `--unscoped`, then out of trust: the server still enforces no root, so the
        // backend does, until the process is relaunched (decision fce98b8a).
        if Self::is_nexus_tool(call)
            && crate::auth::agent_tokens::must_confine_unscoped(&self.session_id)
        {
            if let Some(reason) = self.outside_scope(call) {
                tracing::warn!(
                    session_id = %self.session_id,
                    tool = %call.name,
                    "file tool outside the session's directories refused: the session left trust"
                );
                return HookVerdict::Deny { reason };
            }
        }
        // Rock'n roll opens the origins that lack a consent one by one (a fetch to `http://X`
        // needs `https://X` too, a search needs every engine of the session): each is checked
        // again with the ones already let through, and each is announced. The loop ends: it adds
        // a new origin at every turn or returns.
        // The destinations THIS call names, computed once, before any I/O: what a person may be
        // asked about, and what an `always` answer will record (N23b). A search names none of its
        // own — what it reaches is the engines of the session, so its question is about those.
        let named = {
            let named = destinations_of(&call.name, call.canonical.as_deref(), &call.input);
            if named.is_empty() && call.canonical.as_deref() == Some("WebSearch") {
                search_destinations(&self.search_origins)
            } else {
                named
            }
        };
        let mut let_through = BTreeSet::new();
        loop {
            match check_call_with(
                self.consents.as_ref(),
                self.project.as_deref(),
                &self.search_origins,
                &let_through,
                call,
            )
            .await
            {
                Ok(()) => break,
                Err(refusal) => match self.allowed_by_trust(&refusal).await {
                    Some(origin) if let_through.insert(origin.clone()) => {
                        self.announce(&origin).await;
                    }
                    _ => {
                        // N23b: the hook does not decide in a person's place. A destination the
                        // project has not consented to becomes a QUESTION — the harness asks
                        // through its `permission_ask` channel and remembers the answer against
                        // THIS origin. Every other case (no person in front, a refusal that names
                        // no destination) keeps the flat refusal of before.
                        if let Some(question) = self.consent_question(&refusal, &named) {
                            return question;
                        }
                        tracing::warn!(
                            session_id = %self.session_id,
                            code = refusal.code(),
                            person_in_front = self.person_in_front,
                            "network tool refused"
                        );
                        return HookVerdict::Deny {
                            reason: refusal.reason(),
                        };
                    }
                },
            }
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

    async fn compaction_directive(
        &self,
        info: &CompactionInfo,
    ) -> nexus_claude::agent::CompactionDirective {
        match &self.inner {
            Some(inner) => inner.compaction_directive(info).await,
            None => nexus_claude::agent::CompactionDirective::none(),
        }
    }

    async fn before_turn(&self, ctx: &TurnContext) -> TurnDirective {
        match &self.inner {
            Some(inner) => inner.before_turn(ctx).await,
            None => TurnDirective::default(),
        }
    }

    // Claude Code's own events (contract v7, ISO1) belong to the inner hooks: a wrapper that
    // dropped them would hide the graph's answers on the agent engine.
    async fn session_start(
        &self,
        info: &nexus_claude::agent::SessionStartInfo,
    ) -> nexus_claude::agent::HookReply {
        match &self.inner {
            Some(inner) => inner.session_start(info).await,
            None => nexus_claude::agent::HookReply::none(),
        }
    }

    async fn user_prompt_submit(
        &self,
        prompt: &nexus_claude::agent::PromptInfo,
    ) -> nexus_claude::agent::HookReply {
        match &self.inner {
            Some(inner) => inner.user_prompt_submit(prompt).await,
            None => nexus_claude::agent::HookReply::none(),
        }
    }

    async fn before_stop(
        &self,
        info: &nexus_claude::agent::StopInfo,
    ) -> nexus_claude::agent::HookReply {
        match &self.inner {
            Some(inner) => inner.before_stop(info).await,
            None => nexus_claude::agent::HookReply::none(),
        }
    }

    async fn before_subagent_stop(
        &self,
        info: &nexus_claude::agent::SubagentStopInfo,
    ) -> nexus_claude::agent::HookReply {
        match &self.inner {
            Some(inner) => inner.before_subagent_stop(info).await,
            None => nexus_claude::agent::HookReply::none(),
        }
    }

    async fn notification(&self, info: &nexus_claude::agent::NotificationInfo) {
        if let Some(inner) = &self.inner {
            inner.notification(info).await;
        }
    }

    // The sub-agents of the session (P18) are the inner hooks' to run: the gate only judges
    // tool calls, and the child session is held to its own gate.
    fn runs_subagents(&self) -> bool {
        self.inner
            .as_ref()
            .is_some_and(|inner| inner.runs_subagents())
    }

    async fn run_subagent(
        &self,
        request: &nexus_claude::agent::SubagentRequest,
    ) -> Result<String, String> {
        match &self.inner {
            Some(inner) => inner.run_subagent(request).await,
            None => Err("this session runs no sub-agent".to_owned()),
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
    /// nexus-tools accepts `--unscoped` only with this claim (nexus #93). Left out when false,
    /// like nexus-tools does, so the token of a scoped session is unchanged.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    unscoped: bool,
}

/// Signs the token of a profile with `key` (the bytes of the key string, as nexus-tools reads
/// them): `v1.<base64url(claims)>.<base64url(HMAC-SHA256(key, "v1.<base64url(claims)>"))>`.
pub fn sign_profile(key: &str, session_id: &str, tools: &[String], exp: u64) -> String {
    sign_profile_claims(key, session_id, tools, exp, false)
}

/// [`sign_profile`], with the `unscoped` claim when `unscoped` (the session's file scope is
/// lifted: [`attach`]).
pub fn sign_profile_claims(
    key: &str,
    session_id: &str,
    tools: &[String],
    exp: u64,
    unscoped: bool,
) -> String {
    let body = serde_json::to_vec(&Claims {
        sid: session_id,
        tools,
        exp,
        unscoped,
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
        Self::mint_with(session_id, tools, exp, false)
    }

    /// [`Self::mint`], the token carrying the `unscoped` claim when `unscoped`.
    pub fn mint_with(session_id: &str, tools: &[String], exp: u64, unscoped: bool) -> Self {
        let mut bytes = [0u8; 32];
        rand_core_06::OsRng.fill_bytes(&mut bytes);
        let key = hex::encode(bytes);
        let token = sign_profile_claims(&key, session_id, tools, exp, unscoped);
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
    /// Obscura (`dsh-obscura-nav`), for the keyless `obscura` search engine: the server's
    /// configuration (`chat.obscura_path` / `OBSCURA_BIN`; else, only when `program` does not
    /// embed Obscura, found on the `PATH`: `crate::chat::config::obscura_engine_program`), never a
    /// request's. `None`: the embedded engine when `embedded_obscura`, else a declared `obscura`
    /// engine is skipped (`tool_engine_not_installed`).
    pub obscura: Option<PathBuf>,
    /// `program` embeds Obscura (nexus-tools built with its `obscura` feature, what
    /// [`embeds_obscura`] reads from `--capabilities`): a declared `obscura` engine runs in that
    /// process when no external executable is configured, and the **built-in** `obscura` engine
    /// ([`builtin_obscura_engine`]) is offered when no engine was declared.
    pub embedded_obscura: bool,
    /// Places no file tool of the session may reach, scoped or not (`--deny-path`, nexus #93):
    /// [`host_deny_paths`] in production.
    pub deny_paths: Vec<PathBuf>,
    /// The user's home, when the server knows it ([`HostPlaces::home`]): the native shell's
    /// `git`/`gh` then mirror the host's credentials (decision 2238b1e4) — `GIT_CONFIG_GLOBAL`
    /// for the commit identity and the `gh auth git-credential` helper, `GH_TOKEN` for the
    /// token — while the shell keeps its own HOME. `None`: no mirroring (tests, unknown home).
    pub host_home: Option<PathBuf>,
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
    /// The session's access: a read-only session never has its file scope lifted.
    pub access: super::policy::SessionAccess,
    /// A PERSON lifted this session's file scope ([`crate::neo4j::models::ChatSessionNode::file_scope_lifted`],
    /// decision fce98b8a). Not the decision itself: [`attach`] lifts the scope only when the
    /// policy is in trust AND the access is normal AND this is set ([`lifts_file_scope`]).
    pub lifted_by_person: bool,
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
    /// The server was launched `--unscoped`: its roots are not enforced (the deny list is).
    pub unscoped: bool,
    /// The temporary directory of this session's `nexus-tools` (`TMPDIR`, [`session_temp_dir`]):
    /// its command outputs live below it, so it is part of the session's directories.
    pub temp_dir: Option<PathBuf>,
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

/// The temporary directory of one session's `nexus-tools`: `<tmp>/po-nexus-tools/<session>`,
/// created private (0700). `None` when it cannot be made: the server then uses the host's.
pub fn session_temp_dir(session_id: &str) -> Option<PathBuf> {
    let name: String = session_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if name.is_empty() {
        return None;
    }
    let base = std::env::temp_dir().join("po-nexus-tools");
    let dir = base.join(name);
    std::fs::create_dir_all(&dir).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for private in [&base, &dir] {
            std::fs::set_permissions(private, std::fs::Permissions::from_mode(0o700)).ok()?;
        }
    }
    std::fs::canonicalize(dir).ok()
}

/// The token of the host's `gh` login, read once per attach so a native session's `git`/`gh`
/// authenticate exactly like the host does (decision 2238b1e4). `HOME=<home> gh auth token` is the
/// only pointer that works on macOS: the token lives in the Keychain, resolved through the real
/// home, so `GH_CONFIG_DIR` alone answers 401. The child gets a clean environment (a stray
/// `GH_TOKEN`/`GH_CONFIG_DIR` of the server must not decide how the host's login is read).
///
/// Best-effort: no `gh` on the PATH, not logged in, a timeout of 5 s, or an empty answer → `None`,
/// and the session simply has no `GH_TOKEN`. The value is never logged.
async fn host_gh_token(home: &Path) -> Option<String> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut command = tokio::process::Command::new("gh");
    command
        .args(["auth", "token"])
        .env_clear()
        .env("PATH", path)
        .env("HOME", home)
        .env("GH_PROMPT_DISABLED", "1")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    let output = tokio::time::timeout(std::time::Duration::from_secs(5), command.output())
        .await
        .ok()?
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let token = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    (!token.is_empty()).then_some(token)
}

/// Whether a session's file scope is lifted (`nexus-tools --unscoped`): its policy is in trust
/// (`trust` and `bypassPermissions` alike: the neutral mode), its access is normal (a read-only
/// session never reads the whole disk), AND a person chose it ([`AttachInput::lifted_by_person`],
/// decision fce98b8a). The runner, `delegate_task` and protocol runs are in trust too, but
/// nobody chose the lift for them: they stay scoped.
pub fn lifts_file_scope(
    policy: &ToolPolicy,
    access: super::policy::SessionAccess,
    lifted_by_person: bool,
) -> bool {
    policy.mode == nexus_claude::agent::PolicyMode::Trust
        && access == super::policy::SessionAccess::Normal
        && lifted_by_person
}

/// The application directory of project-orchestrator (`config.yaml`, `vault.json`,
/// `identity.key`, documents, transcripts), as the server finds it.
fn app_dir() -> Option<PathBuf> {
    let config = dirs::config_dir()?;
    Some(if cfg!(target_os = "windows") {
        config.join("ProjectOrchestrator")
    } else {
        config.join("project-orchestrator")
    })
}

/// Where the host keeps what [`deny_paths_for`] protects. [`HostPlaces::detect`] in production.
#[derive(Debug, Clone, Default)]
pub struct HostPlaces {
    /// The user's home directory.
    pub home: Option<PathBuf>,
    /// The application directory of project-orchestrator ([`app_dir`]).
    pub app: Option<PathBuf>,
    /// The vault store.
    pub vault: Option<PathBuf>,
    /// The configuration file the server actually loaded (`--config`, the app directory or the
    /// working directory), made absolute.
    pub config: Option<PathBuf>,
    /// `$CLAUDE_CONFIG_DIR`, when set: Claude Code's configuration moved out of `~/.claude`.
    pub claude_config_dir: Option<PathBuf>,
}

impl HostPlaces {
    /// The places of this host. `config` is the file the server loaded (relative to its working
    /// directory when it was found there).
    pub fn detect(config: Option<&Path>) -> Self {
        let absolute = |path: &Path| {
            if path.is_absolute() {
                Some(path.to_path_buf())
            } else {
                std::env::current_dir().ok().map(|cwd| cwd.join(path))
            }
        };
        Self {
            home: dirs::home_dir(),
            app: app_dir(),
            vault: Some(crate::vault::Vault::default_path()),
            config: config.and_then(absolute),
            claude_config_dir: std::env::var_os("CLAUDE_CONFIG_DIR")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from),
        }
    }
}

/// Home entries that hold credentials of the user's other tools: no file tool reads them.
const HOME_SECRETS: &[&str] = &[
    ".ssh",
    ".claude",
    ".claude.json",
    ".project-orchestrator",
    ".config/gh",
    ".aws",
    ".netrc",
    ".docker/config.json",
    ".kube",
    ".gnupg",
    ".codex",
];

/// The places no file tool of any session may reach (`--deny-path`, nexus #93), scoped or not:
/// - in the home: `.ssh`, `.claude` and `.claude.json` (`mcp.json` holds the server's token),
///   the old `~/.project-orchestrator`, and the credentials of other tools (`gh`, AWS, `.netrc`,
///   Docker, Kubernetes, GnuPG, Codex);
/// - `$CLAUDE_CONFIG_DIR` when set;
/// - the vault store, the configuration file the server loaded and the `.env` next to it;
/// - every entry of the application directory except `chat-neutral`, where sessions without a
///   project WORK.
///
/// The secret files are named even when absent, so one created later is still denied. Bash is
/// a shell, not a sandbox: this list confines the file tools only.
pub fn deny_paths_for(places: &HostPlaces) -> Vec<PathBuf> {
    let mut denied = Vec::new();
    if let Some(home) = &places.home {
        denied.extend(HOME_SECRETS.iter().map(|entry| home.join(entry)));
    }
    denied.extend(places.claude_config_dir.clone());
    denied.extend(places.vault.clone());
    if let Some(config) = &places.config {
        denied.push(config.clone());
        if let Some(dir) = config.parent() {
            denied.push(dir.join(".env"));
        }
    }
    if let Some(app) = &places.app {
        for name in [
            "config.yaml",
            "vault.json",
            "vault-grants.json",
            "identity.key",
            "documents",
            "native-transcripts",
        ] {
            denied.push(app.join(name));
        }
        if let Ok(entries) = std::fs::read_dir(app) {
            for entry in entries.flatten() {
                if entry.file_name() != crate::chat::neutral_place::DIR_NAME {
                    denied.push(entry.path());
                }
            }
        }
    }
    denied.retain(|path| path.is_absolute());
    let mut seen = std::collections::BTreeSet::new();
    denied.retain(|path| seen.insert(path.clone()));
    denied
}

/// [`deny_paths_for`] the places of this host, `config` being the file the server loaded.
pub fn host_deny_paths(config: Option<&Path>) -> Vec<PathBuf> {
    deny_paths_for(&HostPlaces::detect(config))
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
                // In trust the engine stays: the origin is judged at each call, where trust
                // lets a public one through (and a session taken out of trust is held to the
                // project's consent again, the same as the built-in engine below).
                Ok(false) if input.policy.mode == nexus_claude::agent::PolicyMode::Trust => {}
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
                // Keyless: the executable is the server's, one argv element after the flag; without
                // one, the copy embedded in nexus-tools (`obscura` alone).
                ("obscura", Ok(_)) => match config.obscura.as_deref() {
                    Some(program) => {
                        engine_args.push("--search-engine".to_owned());
                        engine_args.push(format!("obscura:{}", program.display()));
                    }
                    None if config.embedded_obscura => {
                        engine_args.push("--search-engine".to_owned());
                        engine_args.push("obscura".to_owned());
                    }
                    None => {
                        attachment.skipped_engines.push((
                            provider.id.clone(),
                            ToolAccessError::EngineNotInstalled {
                                engine: "obscura".to_owned(),
                            },
                        ));
                        continue;
                    }
                },
                (_, Err(refusal)) => attachment
                    .skipped_engines
                    .push((provider.id.clone(), refusal)),
                _ => continue,
            }
            attachment.search_origins.push(origin);
        }
        // The built-in engine (decision 1e9d7501): nexus-tools embeds Obscura and no engine was
        // attached, so search is keyless by default. It is attached even when the project has not
        // consented to its origin yet: the consent is judged at each call ([`check_call`]), and the
        // first search is refused with the origin named, as a fetch to a new origin is. Never next
        // to another engine: nexus-tools would fall back to it, to an origin nobody chose.
        if engine_args.is_empty() && config.embedded_obscura {
            engine_args.push("--search-engine".to_owned());
            engine_args.push("obscura".to_owned());
            attachment
                .search_origins
                .push(OBSCURA_SEARCH_ORIGIN.to_owned());
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
    // The file scope: lifted only for trust + normal access + a person's explicit choice. Then
    // BOTH the flag and the signed claim go to the server (nexus-tools refuses one without the
    // other); the deny list goes in every case.
    let unscoped = lifts_file_scope(input.policy, input.access, input.lifted_by_person);
    let token = SessionToolsToken::mint_with(input.session_id, &tools, exp, unscoped);
    crate::auth::agent_tokens::register_tools(input.session_id);
    crate::auth::agent_tokens::set_launched_unscoped(input.session_id, unscoped);
    let mut args = vec!["--cwd".to_owned(), input.cwd.display().to_string()];
    for dir in input.extra_dirs {
        args.push("--add-dir".to_owned());
        args.push(dir.display().to_string());
    }
    for denied in &config.deny_paths {
        args.push("--deny-path".to_owned());
        args.push(denied.display().to_string());
    }
    if unscoped {
        tracing::info!(
            session_id = input.session_id,
            file_scope = "unscoped",
            deny_paths = config.deny_paths.len(),
            "nexus-tools launched with its file scope lifted (trust, chosen by a person): roots \
             not enforced, deny list kept"
        );
        args.push("--unscoped".to_owned());
    }
    attachment.unscoped = unscoped;
    args.push("--tools".to_owned());
    args.push(tools.join(","));
    args.extend(engine_args);
    let mut env = token.environment();
    // A temporary directory of its own: `nexus-tools` keeps the outputs of its commands in
    // `<tmp>/nexus-tools-<pid>`, which the hooks then know to be this session's (and only it).
    if let Some(tmp) = session_temp_dir(input.session_id) {
        for variable in ["TMPDIR", "TMP", "TEMP"] {
            env.insert(variable.to_owned(), tmp.display().to_string());
        }
        attachment.temp_dir = Some(tmp);
    }
    env.extend(secrets);
    // The host's git/gh credential context (decision 2238b1e4): the sandboxed `git`/`gh` of a
    // native session then authenticate exactly like the host does, while the shell keeps its own
    // HOME. `GIT_CONFIG_GLOBAL` gives the commit identity and the `gh auth git-credential` helper;
    // `GH_TOKEN` is the host's `gh` login, asked once here (best-effort). Without `gh`, without a
    // login, or on a timeout, the session simply has no GH_TOKEN and push fails as before —
    // `git`/`gh` stay working for everything that needs no credential.
    if let Some(home) = &config.host_home {
        let gitconfig = home.join(".gitconfig");
        if gitconfig.is_file() {
            env.insert(
                "NEXUS_TOOLS_GIT_CONFIG_GLOBAL".to_owned(),
                gitconfig.display().to_string(),
            );
        }
        if let Some(token) = host_gh_token(home).await {
            env.insert("NEXUS_TOOLS_GH_TOKEN".to_owned(), token);
        }
    }
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
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
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
            obscura: None,
            embedded_obscura: false,
            deny_paths: vec![PathBuf::from("/home/me/.ssh")],
            host_home: None,
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
            access: super::super::policy::SessionAccess::Normal,
            lifted_by_person: false,
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

    /// A known host home makes the session's `git` read the host's config (decision 2238b1e4):
    /// identity and the `gh auth git-credential` helper, without moving the shell into the home.
    #[tokio::test]
    async fn a_known_host_home_points_the_session_git_at_the_host_config() {
        let home = tempfile::tempdir().unwrap();
        let gitconfig = home.path().join(".gitconfig");
        std::fs::write(&gitconfig, "[user]\n\tname = H\n").unwrap();
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph.clone());
        let policy = ask();
        let cfg = NexusToolsConfig {
            host_home: Some(home.path().to_path_buf()),
            ..config()
        };
        let a = attach(&cfg, &consents, None, input("git-home", &policy, None)).await;
        let env = stdio(&a, NEXUS_TOOLS_SERVER).2;
        assert_eq!(
            env.get("NEXUS_TOOLS_GIT_CONFIG_GLOBAL").map(String::as_str),
            Some(gitconfig.to_str().unwrap())
        );
        // No `gh` login under this home, so no token — and never an empty variable.
        assert!(!env.contains_key("NEXUS_TOOLS_GH_TOKEN"));
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
            // The refusal names the origin to allow (the same as a fetch to a new origin).
            Err(ToolAccessError::OriginNotConsented {
                origin: BRAVE_ORIGIN.to_owned()
            })
        );
        assert_eq!(
            check_call(&consents, Some("proj"), &[], &search).await,
            Err(ToolAccessError::SearchNotConsented)
        );
        store_consent(&graph, "proj", BRAVE_ORIGIN).await;
        check_call(&consents, Some("proj"), &origins, &search)
            .await
            .unwrap();
        // No project, no consent (A28), exactly as for a fetch: the refusal names the origin to
        // allow instead of the generic code, and Rock'n roll can open it like any other.
        assert_eq!(
            check_call(&consents, None, &origins, &search).await,
            Err(ToolAccessError::OriginNotConsented {
                origin: BRAVE_ORIGIN.to_owned()
            })
        );
        // An engine-less session still has nothing to allow.
        assert_eq!(
            check_call(&consents, None, &[], &search).await,
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

    /// Inner hooks that answer Claude Code's own events (contract v7).
    struct LifecycleInner;

    #[async_trait]
    impl SessionHooks for LifecycleInner {
        async fn user_prompt_submit(
            &self,
            _: &nexus_claude::agent::PromptInfo,
        ) -> nexus_claude::agent::HookReply {
            nexus_claude::agent::HookReply::context("graph context")
        }
        async fn before_stop(
            &self,
            _: &nexus_claude::agent::StopInfo,
        ) -> nexus_claude::agent::HookReply {
            nexus_claude::agent::HookReply::block("not yet")
        }
    }

    #[tokio::test]
    async fn the_gate_passes_claude_codes_own_events_on_to_the_inner_hooks() {
        let graph = Arc::new(MockGraphStore::new());
        let hooks = ToolAccessHooks::new(
            Some(Arc::new(LifecycleInner)),
            Arc::new(GraphConsents(graph)),
            "hook-v7",
            Some("proj".into()),
            Vec::new(),
        );
        let prompt = hooks
            .user_prompt_submit(&nexus_claude::agent::PromptInfo::new("go"))
            .await;
        assert_eq!(prompt.additional_context.as_deref(), Some("graph context"));
        let stop = hooks
            .before_stop(&nexus_claude::agent::StopInfo::new(false, None))
            .await;
        assert_eq!(stop.block.as_deref(), Some("not yet"));
    }

    // ---- (0a-bis) N23b: a destination without consent is a QUESTION, when a person is there --

    /// The gate of a person's own session: the shape in which the harness asks.
    fn person_gate(session: &str, graph: &Arc<MockGraphStore>) -> ToolAccessHooks {
        crate::auth::agent_tokens::register_tools(session);
        ToolAccessHooks::new(
            None,
            Arc::new(GraphConsents(graph.clone())),
            session,
            Some("proj".into()),
            Vec::new(),
        )
        .with_person_in_front(true)
    }

    /// The gate of a session nobody is in front of (an agent token, a runner, a delegation, a
    /// protocol executor): `with_person_in_front` is not called, which is the default.
    fn agent_gate(session: &str, graph: &Arc<MockGraphStore>) -> ToolAccessHooks {
        crate::auth::agent_tokens::register_tools(session);
        ToolAccessHooks::new(
            None,
            Arc::new(GraphConsents(graph.clone())),
            session,
            Some("proj".into()),
            Vec::new(),
        )
    }

    #[tokio::test]
    async fn an_unknown_origin_is_a_question_for_a_person_and_a_refusal_for_an_agent() {
        let graph = Arc::new(MockGraphStore::new());
        // A person in front: the hook does not decide in their place, it names the SUBJECT.
        let asked = person_gate("n23b-person", &graph)
            .before_tool(&fetch("https://docs.rs/x"))
            .await;
        let HookVerdict::RequireConsent { subject, reason } = asked else {
            panic!("a consent question was expected, got {asked:?}");
        };
        assert_eq!(subject, "https://docs.rs", "scheme+host+port, never a path");
        assert!(reason.starts_with("tool_origin_not_allowed:"), "{reason}");
        // Nobody in front: refused exactly as before, and no question is raised at all.
        let refused = agent_gate("n23b-agent", &graph)
            .before_tool(&fetch("https://docs.rs/x"))
            .await;
        let HookVerdict::Deny { reason } = refused else {
            panic!("a denial was expected for an agent session, got {refused:?}");
        };
        assert!(reason.starts_with("tool_origin_not_allowed:"));
        crate::auth::agent_tokens::revoke_session("n23b-person");
        crate::auth::agent_tokens::revoke_session("n23b-agent");
    }

    #[tokio::test]
    async fn a_call_that_names_no_destination_stays_a_flat_refusal_even_for_a_person() {
        let graph = Arc::new(MockGraphStore::new());
        // No engine attached at all: there is nothing to reach, so nothing to consent to, and the
        // refusal is the flat `tool_search_not_allowed` of before.
        let search = call(
            "mcp__nexus__WebSearch",
            Some("WebSearch"),
            json!({"query": "rust"}),
        );
        assert!(destinations_of(&search.name, Some("WebSearch"), &search.input).is_empty());
        let verdict = person_gate("n23b-search", &graph)
            .before_tool(&search)
            .await;
        assert!(
            matches!(verdict, HookVerdict::Deny { .. }),
            "no engine, so no question: {verdict:?}"
        );
        crate::auth::agent_tokens::revoke_session("n23b-search");
    }

    /// A search names no page of its own, but it does reach something: the engines of the
    /// session. Their origins ARE the question, all of them (nexus-tools falls back from one to
    /// the next), so a person can answer it in the chat instead of going to the settings page.
    #[tokio::test]
    async fn a_search_asks_about_the_engines_it_would_reach() {
        let graph = Arc::new(MockGraphStore::new());
        crate::auth::agent_tokens::register_tools("n23b-search-engines");
        let gate = ToolAccessHooks::new(
            None,
            Arc::new(GraphConsents(graph.clone())),
            "n23b-search-engines",
            Some("proj".into()),
            vec![
                "https://html.duckduckgo.com".to_owned(),
                "https://api.search.brave.com".to_owned(),
            ],
        )
        .with_person_in_front(true);
        let verdict = gate.before_tool(&web_search()).await;
        let HookVerdict::RequireConsent { subject, reason } = verdict else {
            panic!("a consent question was expected, got {verdict:?}");
        };
        // The subject is the one the refusal named — the first engine, as the gate tries them.
        assert_eq!(subject, "https://html.duckduckgo.com");
        assert!(reason.starts_with("tool_origin_not_allowed:"), "{reason}");
        // Every engine is what an `always` answer would record, and they are the engines really
        // reached: `https://`, deduplicated.
        assert_eq!(
            search_destinations(&[
                "http://html.duckduckgo.com".to_owned(),
                "https://html.duckduckgo.com".to_owned(),
                "https://api.search.brave.com".to_owned(),
            ]),
            vec![
                "https://html.duckduckgo.com".to_owned(),
                "https://api.search.brave.com".to_owned(),
            ]
        );
        crate::auth::agent_tokens::revoke_session("n23b-search-engines");
    }

    #[tokio::test]
    async fn the_subject_is_the_endpoint_nexus_tools_reaches() {
        let graph = Arc::new(MockGraphStore::new());
        // nexus-tools upgrades `http://` before it connects, and the gate judges BOTH forms: the
        // question is about the secure one, and both are what an `always` records.
        assert_eq!(
            destinations_of(
                "mcp__nexus__WebFetch",
                Some("WebFetch"),
                &json!({"url": "http://docs.rs/x"})
            ),
            vec!["http://docs.rs".to_owned(), "https://docs.rs".to_owned()]
        );
        let verdict = person_gate("n23b-http", &graph)
            .before_tool(&fetch("http://docs.rs/x"))
            .await;
        let HookVerdict::RequireConsent { subject, .. } = verdict else {
            panic!("a consent question was expected, got {verdict:?}");
        };
        assert_eq!(subject, "https://docs.rs");
        crate::auth::agent_tokens::revoke_session("n23b-http");
    }

    #[tokio::test]
    async fn an_always_answer_records_the_same_document_the_settings_route_writes() {
        let graph = Arc::new(MockGraphStore::new());
        let fetch_call = fetch("https://docs.rs/x");
        // What the answer path does for `always`: the SAME writer, the SAME store, the SAME key
        // as `PUT /api/projects/{slug}/network-tools/origins`, once per destination the gate judges.
        for origin in destinations_of(
            &fetch_call.name,
            fetch_call.canonical.as_deref(),
            &fetch_call.input,
        ) {
            record_origin_consent(graph.as_ref(), "proj", &origin, "me@example.com")
                .await
                .unwrap();
        }
        // The store itself, not a mock of it: the document the settings route reads back.
        let stored = graph
            .get_llm_setting(
                &project_scope("proj"),
                &format!("{TOOL_ORIGIN_PREFIX}https://docs.rs"),
            )
            .await
            .unwrap()
            .expect("the consent document");
        let consent: ToolOriginConsent = serde_json::from_str(&stored).unwrap();
        assert_eq!(consent.origin, "https://docs.rs");
        assert_eq!(consent.consented_by, "me@example.com");
        // ...and the gate of a LATER session finds it: no question, the call runs.
        assert_eq!(
            person_gate("n23b-always", &graph)
                .before_tool(&fetch_call)
                .await,
            HookVerdict::Continue
        );
        crate::auth::agent_tokens::revoke_session("n23b-always");
    }

    #[tokio::test]
    async fn a_session_answer_writes_no_consent_and_covers_one_origin_only() {
        let graph = Arc::new(MockGraphStore::new());
        let a = fetch("https://docs.rs/x");
        let b = fetch("https://evil.example/");
        // A `session` answer is the harness's own memory (`tool\u{0}subject`), never a durable
        // consent: the project's store still holds nothing.
        assert!(graph
            .get_llm_setting(
                &project_scope("proj"),
                &format!("{TOOL_ORIGIN_PREFIX}https://docs.rs")
            )
            .await
            .unwrap()
            .is_none());
        // Two origins are two questions and two subjects: approving one says nothing about the
        // other, and the second is not even among the destinations the first covered.
        let subject_a = consent_subject(&a.name, a.canonical.as_deref(), &a.input).unwrap();
        let subject_b = consent_subject(&b.name, b.canonical.as_deref(), &b.input).unwrap();
        assert_ne!(subject_a, subject_b);
        assert!(!destinations_of(&b.name, b.canonical.as_deref(), &b.input).contains(&subject_a));
    }

    // ---- (0b) trust (Rock'n roll): public origins without a consent, and the person is told ----

    /// Public IP literals: the guard needs no DNS to judge them, so these tests stay offline.
    const PUBLIC: &str = "https://93.184.216.34";

    type Said = Arc<std::sync::Mutex<Vec<String>>>;

    fn trust_hooks(
        session: &str,
        in_trust: Arc<std::sync::atomic::AtomicBool>,
    ) -> (ToolAccessHooks, Said) {
        crate::auth::agent_tokens::register_tools(session);
        let said: Said = Arc::default();
        let sink = said.clone();
        let hooks = ToolAccessHooks::new(
            None,
            Arc::new(GraphConsents(Arc::new(MockGraphStore::new()))),
            session,
            Some("proj".into()),
            Vec::new(),
        )
        .with_trust(
            Arc::new(move || in_trust.load(std::sync::atomic::Ordering::SeqCst)),
            Arc::new(move |text: String| {
                sink.lock().unwrap().push(text);
                Box::pin(async { true }) as futures::future::BoxFuture<'static, bool>
            }),
        );
        (hooks, said)
    }

    fn flag(on: bool) -> Arc<std::sync::atomic::AtomicBool> {
        Arc::new(std::sync::atomic::AtomicBool::new(on))
    }

    #[tokio::test]
    async fn a_session_in_trust_reaches_a_public_origin_without_consent_and_says_so_once() {
        let (hooks, said) = trust_hooks("trust-s1", flag(true));
        assert_eq!(
            hooks.before_tool(&fetch(&format!("{PUBLIC}/a"))).await,
            HookVerdict::Continue
        );
        assert_eq!(
            hooks.before_tool(&fetch(&format!("{PUBLIC}/b"))).await,
            HookVerdict::Continue
        );
        {
            let said = said.lock().unwrap();
            assert_eq!(said.len(), 1, "said once per origin, not at every call");
            assert!(said[0].contains(PUBLIC), "{}", said[0]);
            assert!(said[0].contains("not saved"), "it is not a stored consent");
        }
        assert_eq!(
            hooks.before_tool(&fetch("https://1.1.1.1/")).await,
            HookVerdict::Continue
        );
        assert_eq!(
            said.lock().unwrap().len(),
            2,
            "another origin is said again"
        );
        crate::auth::agent_tokens::revoke_session("trust-s1");
    }

    #[tokio::test]
    async fn rock_n_roll_does_not_open_the_machines_own_network() {
        // nexus-tools' own fetch refuses these classes (`web/ssrf.rs`); a gate that announced
        // "allowed automatically" for them would announce an access the tool then refuses.
        let (hooks, said) = trust_hooks("trust-s2", flag(true));
        for url in [
            "https://127.0.0.1/",
            "https://[::1]/",
            "https://localhost/",
            "https://api.localhost/",
            "https://[::ffff:127.0.0.1]/",
            "https://169.254.169.254/latest/meta-data/",
            "https://10.0.0.5/",
            "https://192.168.1.1/admin",
            "https://0.0.0.0/",
            "http://127.0.0.1:8080/api/projects",
        ] {
            let verdict = hooks.before_tool(&fetch(url)).await;
            let HookVerdict::Deny { reason } = verdict else {
                panic!("{url} must stay refused in Rock'n roll, got {verdict:?}");
            };
            assert!(
                reason.starts_with("tool_origin_not_allowed:"),
                "{url}: {reason}"
            );
        }
        assert!(
            said.lock().unwrap().is_empty(),
            "nothing allowed, nothing said"
        );
        crate::auth::agent_tokens::revoke_session("trust-s2");
    }

    #[tokio::test]
    async fn a_session_out_of_trust_is_refused_as_before() {
        let (hooks, said) = trust_hooks("trust-s3", flag(false));
        let verdict = hooks.before_tool(&fetch(&format!("{PUBLIC}/"))).await;
        let HookVerdict::Deny { reason } = verdict else {
            panic!("a denial was expected, got {verdict:?}");
        };
        assert!(reason.starts_with("tool_origin_not_allowed:"));
        assert!(said.lock().unwrap().is_empty());
        crate::auth::agent_tokens::revoke_session("trust-s3");
    }

    #[tokio::test]
    async fn leaving_trust_takes_the_permission_back_even_for_an_origin_already_announced() {
        let mode = flag(true);
        let (hooks, said) = trust_hooks("trust-s4", mode.clone());
        let url = format!("{PUBLIC}/");
        assert_eq!(hooks.before_tool(&fetch(&url)).await, HookVerdict::Continue);
        mode.store(false, std::sync::atomic::Ordering::SeqCst);
        assert!(
            matches!(
                hooks.before_tool(&fetch(&url)).await,
                HookVerdict::Deny { .. }
            ),
            "announcing an origin is not a grant"
        );
        mode.store(true, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(hooks.before_tool(&fetch(&url)).await, HookVerdict::Continue);
        assert_eq!(
            said.lock().unwrap().len(),
            1,
            "back in trust: not announced twice"
        );
        crate::auth::agent_tokens::revoke_session("trust-s4");
    }

    #[tokio::test]
    async fn an_http_origin_is_judged_and_announced_as_the_https_origin_nexus_tools_reaches() {
        let (hooks, said) = trust_hooks("trust-s5", flag(true));
        assert_eq!(
            hooks.before_tool(&fetch("http://93.184.216.34/")).await,
            HookVerdict::Continue
        );
        {
            let said = said.lock().unwrap();
            assert_eq!(said.len(), 1, "one notice, not one per scheme: {said:?}");
            assert!(said[0].contains("https://93.184.216.34"), "{}", said[0]);
            assert!(
                !said[0].contains("http://"),
                "the origin reached: {}",
                said[0]
            );
        }
        assert!(matches!(
            hooks.before_tool(&fetch("http://10.0.0.5/")).await,
            HookVerdict::Deny { .. }
        ));
        // A request that names no origin at all stays refused.
        assert!(matches!(
            hooks.before_tool(&fetch("not a url")).await,
            HookVerdict::Deny { .. }
        ));
        crate::auth::agent_tokens::revoke_session("trust-s5");
    }

    /// A search with `engines` attached, the project consenting to `consented` of them.
    async fn search_hooks(
        session: &str,
        engines: &[&str],
        consented: &[&str],
        in_trust: Arc<std::sync::atomic::AtomicBool>,
    ) -> (ToolAccessHooks, Said) {
        let graph = Arc::new(MockGraphStore::new());
        for origin in consented {
            store_consent(&graph, "proj", origin).await;
        }
        crate::auth::agent_tokens::register_tools(session);
        let said: Said = Arc::default();
        let sink = said.clone();
        let hooks = ToolAccessHooks::new(
            None,
            Arc::new(GraphConsents(graph)),
            session,
            Some("proj".into()),
            engines.iter().map(|e| (*e).to_owned()).collect(),
        )
        .with_trust(
            Arc::new(move || in_trust.load(std::sync::atomic::Ordering::SeqCst)),
            Arc::new(move |text: String| {
                sink.lock().unwrap().push(text);
                Box::pin(async { true }) as futures::future::BoxFuture<'static, bool>
            }),
        );
        (hooks, said)
    }

    fn web_search() -> ToolCallInfo {
        call(
            "mcp__nexus__WebSearch",
            Some("WebSearch"),
            json!({"query": "antibes"}),
        )
    }

    #[tokio::test]
    async fn a_search_needs_every_engine_of_the_session_allowed_not_one_of_them() {
        const A: &str = "https://93.184.216.34";
        const B: &str = "https://1.1.1.1";
        // nexus-tools falls back from one engine to the next: the project consents to A only, so
        // B would be reached through A's fallback. Outside Rock'n roll that is refused, naming B.
        let mode = flag(false);
        let (hooks, said) = search_hooks("trust-s7", &[A, B], &[A], mode.clone()).await;
        let HookVerdict::Deny { reason } = hooks.before_tool(&web_search()).await else {
            panic!("a denial was expected");
        };
        assert!(reason.starts_with("tool_origin_not_allowed:"), "{reason}");
        assert!(reason.contains(B) && !reason.contains(A), "{reason}");
        // In Rock'n roll the engine without a consent is opened and announced, and ONLY it.
        mode.store(true, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            hooks.before_tool(&web_search()).await,
            HookVerdict::Continue
        );
        {
            let said = said.lock().unwrap();
            assert_eq!(said.len(), 1, "{said:?}");
            assert!(said[0].contains(B), "{}", said[0]);
        }
        // Out again: the allowance was the mode's, B is refused once more.
        mode.store(false, std::sync::atomic::Ordering::SeqCst);
        assert!(matches!(
            hooks.before_tool(&web_search()).await,
            HookVerdict::Deny { .. }
        ));
        crate::auth::agent_tokens::revoke_session("trust-s7");
    }

    #[tokio::test]
    async fn every_engine_without_a_consent_is_announced_in_rock_n_roll() {
        let (hooks, said) = search_hooks(
            "trust-s8",
            &["https://93.184.216.34", "https://1.1.1.1"],
            &[],
            flag(true),
        )
        .await;
        assert_eq!(
            hooks.before_tool(&web_search()).await,
            HookVerdict::Continue
        );
        let said = said.lock().unwrap();
        assert_eq!(said.len(), 2, "one notice per engine: {said:?}");
        drop(said);
        crate::auth::agent_tokens::revoke_session("trust-s8");
    }

    #[tokio::test]
    async fn a_notice_that_was_lost_is_sent_again_at_the_next_call() {
        crate::auth::agent_tokens::register_tools("trust-s9");
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = attempts.clone();
        let hooks = ToolAccessHooks::new(
            None,
            Arc::new(GraphConsents(Arc::new(MockGraphStore::new()))),
            "trust-s9",
            Some("proj".into()),
            Vec::new(),
        )
        .with_trust(
            Arc::new(|| true),
            Arc::new(move |_text: String| {
                // The first delivery fails (the session's stream was not there), the next works.
                let first = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0;
                Box::pin(async move { !first }) as futures::future::BoxFuture<'static, bool>
            }),
        );
        let url = format!("{PUBLIC}/");
        assert_eq!(hooks.before_tool(&fetch(&url)).await, HookVerdict::Continue);
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(hooks.before_tool(&fetch(&url)).await, HookVerdict::Continue);
        assert_eq!(
            attempts.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "the lost notice is sent again"
        );
        assert_eq!(hooks.before_tool(&fetch(&url)).await, HookVerdict::Continue);
        assert_eq!(
            attempts.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "and once delivered it is not repeated"
        );
        crate::auth::agent_tokens::revoke_session("trust-s9");
    }

    #[test]
    fn rock_n_roll_opens_the_network_only_for_a_normal_session_a_person_put_in_trust() {
        use crate::chat::provider::policy::SessionAccess;
        // A person chose trust on a normal session: open.
        assert!(trust_opens_network(SessionAccess::Normal, true));
        // The runner, a protocol executor, a delegated task and a session nobody owns run in
        // `bypassPermissions` with nobody to have chosen it: closed.
        assert!(!trust_opens_network(SessionAccess::Normal, false));
        // A read-only session has no Bash (its denial wins over trust): an open network would be
        // its channel out, even when a person chose trust.
        assert!(!trust_opens_network(SessionAccess::ReadOnly, true));
        assert!(!trust_opens_network(SessionAccess::ReadOnly, false));
    }

    #[tokio::test]
    async fn a_session_without_the_trust_wiring_keeps_the_old_refusal() {
        crate::auth::agent_tokens::register_tools("trust-s6");
        let hooks = ToolAccessHooks::new(
            None,
            Arc::new(GraphConsents(Arc::new(MockGraphStore::new()))),
            "trust-s6",
            Some("proj".into()),
            Vec::new(),
        );
        assert!(matches!(
            hooks.before_tool(&fetch(&format!("{PUBLIC}/"))).await,
            HookVerdict::Deny { .. }
        ));
        crate::auth::agent_tokens::revoke_session("trust-s6");
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

    /// The claims a signed profile token carries, read without the key (the body is not secret).
    fn token_claims(token: &str) -> serde_json::Value {
        let body = token.split('.').nth(1).unwrap();
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(body).unwrap()).unwrap()
    }

    /// The file scope is lifted (decision fce98b8a) only when the policy is in trust, the access
    /// is normal AND a person chose it. Then BOTH the flag and the signed claim go to the server
    /// (nexus-tools refuses one without the other); the deny list goes in every case. The
    /// runner, `delegate_task` and protocol runs are in trust without a person's choice: scoped.
    #[tokio::test]
    async fn the_file_scope_is_lifted_only_for_trust_normal_access_and_a_persons_choice() {
        use super::super::policy::SessionAccess::{Normal, ReadOnly};
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph);
        let cases = [
            (PolicyMode::Trust, Normal, true, true),
            (PolicyMode::Trust, Normal, false, false),
            (PolicyMode::Trust, ReadOnly, true, false),
            (PolicyMode::Ask, Normal, true, false),
            (PolicyMode::AutoEdits, Normal, true, false),
            (PolicyMode::PlanOnly, Normal, true, false),
        ];
        for (index, (mode, access, lifted_by_person, expected)) in cases.into_iter().enumerate() {
            let sid = format!("att-lift-{index}");
            let policy =
                ToolPolicy::from_patterns(mode, &["Read", "Grep"], &[] as &[&str]).unwrap();
            let mut attach_input = input(&sid, &policy, Some("proj"));
            attach_input.access = access;
            attach_input.lifted_by_person = lifted_by_person;
            let attachment = attach(&config(), &consents, None, attach_input).await;
            let case = format!("{mode:?} {access:?} lifted={lifted_by_person}");
            let (_, args, env) = stdio(&attachment, "nexus");
            assert_eq!(
                args.contains(&"--unscoped".to_owned()),
                expected,
                "{case}: {args:?}"
            );
            assert_eq!(attachment.unscoped, expected, "{case}");
            let claims = token_claims(&env["NEXUS_TOOLS_PROFILE"]);
            assert_eq!(
                claims.get("unscoped") == Some(&json!(true)),
                expected,
                "{case}: {claims}"
            );
            if !expected {
                assert!(
                    claims.get("unscoped").is_none(),
                    "{case}: a scoped token is unchanged"
                );
            }
            assert_eq!(
                crate::auth::agent_tokens::is_scope_lifted(&sid),
                expected,
                "{case}"
            );
            // The deny list, in every case.
            let at = args.iter().position(|a| a == "--deny-path").unwrap();
            assert_eq!(args[at + 1], "/home/me/.ssh", "{case}");
            // A private temporary directory of its own (its command outputs live below it).
            let tmp = attachment
                .temp_dir
                .clone()
                .expect("a session temporary directory");
            assert_eq!(env["TMPDIR"], tmp.display().to_string(), "{case}");
            assert!(tmp.ends_with(&sid), "{case}: {tmp:?}");
            crate::auth::agent_tokens::revoke_session(&sid);
        }
    }

    /// `bypassPermissions` (the stored spelling) is the same trust as `trust`.
    #[tokio::test]
    async fn bypass_permissions_lifts_the_file_scope_like_trust() {
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph);
        for (index, mode) in ["bypassPermissions", "trust"].into_iter().enumerate() {
            let pair = super::super::policy::parse_mode(mode).unwrap();
            let policy =
                ToolPolicy::from_patterns(pair.neutral, &["Read"], &[] as &[&str]).unwrap();
            let sid = format!("att-bypass-{index}");
            let mut attach_input = input(&sid, &policy, Some("proj"));
            attach_input.lifted_by_person = true;
            let attachment = attach(&config(), &consents, None, attach_input).await;
            assert!(attachment.unscoped, "{mode}");
            crate::auth::agent_tokens::revoke_session(&sid);
        }
    }

    /// A session launched `--unscoped` that LEFT trust is held to its directories by its hooks
    /// (the process keeps its roots unenforced until relaunched), with nexus-tools' own resolver.
    #[tokio::test]
    async fn a_session_launched_unscoped_is_confined_by_its_hooks_once_it_leaves_trust() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("in.txt"), "in").unwrap();
        // The session's temporary directory: a background command's output file lives below it.
        let session_tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(session_tmp.path().join("nexus-tools-42")).unwrap();
        std::fs::write(session_tmp.path().join("nexus-tools-42/task.out"), "done").unwrap();
        std::fs::write(outside.path().join("out.txt"), "out").unwrap();
        let sid = "hook-confine-1";
        let hooks = ToolAccessHooks::new(
            None,
            Arc::new(GraphConsents(Arc::new(MockGraphStore::new()))),
            sid,
            Some("proj".into()),
            Vec::new(),
        )
        .with_scope_dirs(
            root.path().to_path_buf(),
            vec![session_tmp.path().to_path_buf()],
        );
        crate::auth::agent_tokens::register_tools(sid);
        let read = |path: &Path| {
            call(
                "mcp__nexus__Read",
                Some("Read"),
                json!({"file_path": path.display().to_string()}),
            )
        };
        let denied = |verdict: HookVerdict| matches!(verdict, HookVerdict::Deny { .. });
        // Launched scoped: nothing for the hooks to do (the server confines).
        assert!(!denied(
            hooks
                .before_tool(&read(&outside.path().join("out.txt")))
                .await
        ));
        // Launched unscoped, in trust: the lift holds, outside is allowed.
        crate::auth::agent_tokens::set_launched_unscoped(sid, true);
        assert!(!denied(
            hooks
                .before_tool(&read(&outside.path().join("out.txt")))
                .await
        ));
        // Left trust: outside refused for every file tool, inside still allowed.
        crate::auth::agent_tokens::set_scope_lift_live(sid, false);
        let out = outside.path().join("out.txt").display().to_string();
        for (name, canonical, input) in [
            ("mcp__nexus__Read", "Read", json!({"file_path": out})),
            (
                "mcp__nexus__Write",
                "Write",
                json!({"file_path": out, "content": "x"}),
            ),
            (
                "mcp__nexus__Edit",
                "Edit",
                json!({"file_path": out, "old_string": "o", "new_string": "x"}),
            ),
            (
                "mcp__nexus__NotebookEdit",
                "NotebookEdit",
                json!({"notebook_path": out, "new_source": "x"}),
            ),
            (
                "mcp__nexus__Grep",
                "Grep",
                json!({"pattern": "o", "path": outside.path().display().to_string()}),
            ),
            (
                "mcp__nexus__Glob",
                "Glob",
                json!({"pattern": "*", "path": outside.path().display().to_string()}),
            ),
            // `..` resolved like the server does.
            (
                "mcp__nexus__Read",
                "Read",
                json!({"file_path": format!("{}/../{}/out.txt", root.path().display(), outside.path().file_name().unwrap().to_string_lossy())}),
            ),
        ] {
            assert!(
                denied(
                    hooks
                        .before_tool(&call(name, Some(canonical), input.clone()))
                        .await
                ),
                "{name} {input}"
            );
        }
        assert!(!denied(
            hooks.before_tool(&read(&root.path().join("in.txt"))).await
        ));
        assert!(
            !denied(
                hooks
                    .before_tool(&read(&session_tmp.path().join("nexus-tools-42/task.out")))
                    .await
            ),
            "the output of a background command stays readable"
        );
        assert!(!denied(
            hooks
                .before_tool(&call(
                    "mcp__nexus__Glob",
                    Some("Glob"),
                    json!({"pattern": "*"})
                ))
                .await
        ));
        // Back in trust by a person: lifted again.
        crate::auth::agent_tokens::set_scope_lift_live(sid, true);
        assert!(!denied(
            hooks
                .before_tool(&read(&outside.path().join("out.txt")))
                .await
        ));
        crate::auth::agent_tokens::revoke_session(sid);
    }

    /// The deny list names the secrets of the host and of project-orchestrator (the config the
    /// server loaded and its `.env`, `$CLAUDE_CONFIG_DIR`, other tools' credentials) and never
    /// the neutral directory where sessions without a project work. Built from temporary
    /// places: the test reads nothing of the real machine.
    #[test]
    fn the_deny_list_names_the_secrets_and_spares_the_neutral_directory() {
        let home = tempfile::tempdir().unwrap();
        let app = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let neutral = app.path().join(crate::chat::neutral_place::DIR_NAME);
        std::fs::create_dir_all(&neutral).unwrap();
        std::fs::write(app.path().join("some-future-store.db"), "x").unwrap();
        let config = elsewhere.path().join("config.yaml");
        let claude_dir = elsewhere.path().join("claude-config");
        let places = HostPlaces {
            home: Some(home.path().to_path_buf()),
            app: Some(app.path().to_path_buf()),
            vault: Some(app.path().join("vault.json")),
            config: Some(config.clone()),
            claude_config_dir: Some(claude_dir.clone()),
        };
        let denied = deny_paths_for(&places);
        let mut expected: Vec<PathBuf> = HOME_SECRETS
            .iter()
            .map(|entry| home.path().join(entry))
            .collect();
        expected.extend([
            claude_dir,
            config,
            elsewhere.path().join(".env"),
            app.path().join("vault.json"),
            app.path().join("identity.key"),
            // An entry of the app directory nobody named yet.
            app.path().join("some-future-store.db"),
        ]);
        for path in &expected {
            assert!(denied.contains(path), "{path:?} not in {denied:?}");
        }
        for secret in [
            ".claude.json",
            ".aws",
            ".netrc",
            ".config/gh",
            ".docker/config.json",
            ".kube",
            ".gnupg",
            ".codex",
        ] {
            assert!(denied.contains(&home.path().join(secret)), "{secret}");
        }
        assert!(
            !denied.iter().any(|d| neutral.starts_with(d)),
            "the neutral directory is denied: {denied:?}"
        );
        assert!(denied.iter().all(|d| d.is_absolute()));
        let unique: std::collections::BTreeSet<_> = denied.iter().collect();
        assert_eq!(unique.len(), denied.len(), "no path twice");
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
        // `TaskStop` rides with `Bash` (nexus #83): the harness stops a background task
        // with it (`cancel_tools(task)`); the model is offered it only if the policy says so.
        assert_eq!(
            profile_for(&named, None),
            ["Read", "Grep", "Bash", "TaskStop"]
        );
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

    #[tokio::test]
    async fn a_session_in_trust_keeps_an_engine_whose_origin_the_project_never_consented_to() {
        let graph = Arc::new(MockGraphStore::new());
        store_provider(&graph, &brave("brave-main")).await;
        let vault = vault_with_key(Some("tool:brave-main")).await;
        let trust = ToolPolicy::new(PolicyMode::Trust);
        let attachment = attach(
            &config(),
            &GraphConsents(graph),
            Some(&vault),
            input("srch-s4t", &trust, Some("proj")),
        )
        .await;
        assert!(
            attachment.skipped_engines.is_empty(),
            "{:?}",
            attachment.skipped_engines
        );
        assert!(attachment.tools.contains(&"WebSearch".to_owned()));
        assert_eq!(attachment.search_origins, [BRAVE_ORIGIN]);
        crate::auth::agent_tokens::revoke_session("srch-s4t");
    }

    // ---- (4b) the built-in `obscura` engine (decision 1e9d7501) -------------------------------

    fn embedded() -> NexusToolsConfig {
        NexusToolsConfig {
            embedded_obscura: true,
            ..config()
        }
    }

    #[tokio::test]
    async fn with_obscura_embedded_search_is_keyless_by_default_and_its_first_call_names_the_origin(
    ) {
        let graph = Arc::new(MockGraphStore::new());
        let consents = GraphConsents(graph.clone());
        let policy = ask();
        // Nothing declared, nothing consented: the built-in engine is attached all the same.
        let attachment = attach(
            &embedded(),
            &consents,
            None,
            input("bi-s1", &policy, Some("proj")),
        )
        .await;
        let (_, args, _) = stdio(&attachment, "nexus");
        let at = args.iter().position(|a| a == "--search-engine").unwrap();
        assert_eq!(
            args[at + 1],
            "obscura",
            "the engine embedded in nexus-tools, no path"
        );
        assert!(attachment.tools.contains(&"WebSearch".to_owned()));
        assert_eq!(attachment.search_origins, [OBSCURA_SEARCH_ORIGIN]);
        // The project has not consented: the first search is refused with the origin to allow.
        let search = call(
            "mcp__nexus__WebSearch",
            Some("WebSearch"),
            json!({"query": "rust"}),
        );
        assert_eq!(
            check_call(&consents, Some("proj"), &attachment.search_origins, &search).await,
            Err(ToolAccessError::OriginNotConsented {
                origin: OBSCURA_SEARCH_ORIGIN.to_owned()
            })
        );
        store_consent(&graph, "proj", OBSCURA_SEARCH_ORIGIN).await;
        check_call(&consents, Some("proj"), &attachment.search_origins, &search)
            .await
            .unwrap();
        crate::auth::agent_tokens::revoke_session("bi-s1");
        // No project, no consent possible: no built-in engine either.
        let none = attach(&embedded(), &consents, None, input("bi-s2", &policy, None)).await;
        assert!(!none.tools.contains(&"WebSearch".to_owned()));
        crate::auth::agent_tokens::revoke_session("bi-s2");
        // A nexus-tools that does not embed Obscura: no default engine, as before.
        let plain = attach(
            &config(),
            &consents,
            None,
            input("bi-s3", &policy, Some("proj")),
        )
        .await;
        assert!(!plain.tools.contains(&"WebSearch".to_owned()));
        crate::auth::agent_tokens::revoke_session("bi-s3");
    }

    #[tokio::test]
    async fn the_built_in_engine_never_sits_behind_a_declared_one() {
        let graph = Arc::new(MockGraphStore::new());
        store_provider(&graph, &brave("brave-main")).await;
        store_consent(&graph, "proj", BRAVE_ORIGIN).await;
        let vault = vault_with_key(Some("tool:brave-main")).await;
        let policy = ask();
        let attachment = attach(
            &embedded(),
            &GraphConsents(graph),
            Some(&vault),
            input("bi-s4", &policy, Some("proj")),
        )
        .await;
        let (_, args, _) = stdio(&attachment, "nexus");
        // One engine only: nexus-tools would otherwise fall back to an origin nobody chose.
        assert_eq!(args.iter().filter(|a| *a == "--search-engine").count(), 1);
        assert!(args.iter().all(|a| a != "obscura"));
        assert_eq!(attachment.search_origins, [BRAVE_ORIGIN]);
        crate::auth::agent_tokens::revoke_session("bi-s4");
    }

    fn searxng(id: &str, url: &str) -> SearchProviderRecord {
        SearchProviderRecord {
            id: id.into(),
            engine: "searxng".into(),
            base_url: Some(url.into()),
            credential_ref: "none".into(),
        }
    }

    /// The `--search-engine` values, in argv order: the order nexus-tools tries them in.
    fn engine_flags(args: &[String]) -> Vec<String> {
        args.windows(2)
            .filter(|w| w[0] == "--search-engine")
            .map(|w| w[1].clone())
            .collect()
    }

    /// The engine flags and search origins a native session of `proj` is attached with.
    async fn engines_of(graph: &Arc<MockGraphStore>, sid: &str) -> (Vec<String>, Vec<String>) {
        let policy = ask();
        let attachment = attach(
            &config(),
            &GraphConsents(graph.clone()),
            None,
            input(sid, &policy, Some("proj")),
        )
        .await;
        let (_, args, _) = stdio(&attachment, "nexus");
        crate::auth::agent_tokens::revoke_session(sid);
        (engine_flags(&args), attachment.search_origins)
    }

    #[tokio::test]
    async fn search_engines_reach_nexus_tools_in_the_chosen_order_and_by_id_without_one() {
        let graph = Arc::new(MockGraphStore::new());
        for record in [
            searxng("a-sx", "https://a.example"),
            searxng("b-sx", "https://b.example"),
            searxng("c-sx", "https://c.example"),
        ] {
            store_provider(&graph, &record).await;
        }
        for origin in [
            "https://a.example",
            "https://b.example",
            "https://c.example",
        ] {
            store_consent(&graph, "proj", origin).await;
        }
        // No order stored (every installation so far): by id, as before.
        let (before, _) = engines_of(&graph, "ord-s1").await;
        assert_eq!(
            before,
            [
                "searxng:https://a.example",
                "searxng:https://b.example",
                "searxng:https://c.example"
            ]
        );
        // A chosen order: the named engines first, in that order; the unnamed one after them;
        // an id no engine has is ignored.
        let order = SearchEngineOrder {
            order: vec!["c-sx".into(), "gone".into(), "a-sx".into()],
            ordered_by: "me@example.com".into(),
            ordered_at: "2026-10-18T10:00:00Z".into(),
        };
        graph
            .put_llm_setting(
                super::super::settings::GLOBAL,
                SEARCH_ENGINE_ORDER_KEY,
                &serde_json::to_string(&order).unwrap(),
            )
            .await
            .unwrap();
        let (after, origins) = engines_of(&graph, "ord-s2").await;
        assert_eq!(
            after,
            [
                "searxng:https://c.example",
                "searxng:https://a.example",
                "searxng:https://b.example"
            ]
        );
        assert_eq!(
            origins,
            [
                "https://c.example",
                "https://a.example",
                "https://b.example"
            ]
        );
        // A document that does not decode is no order: by id again, never a refusal.
        graph
            .put_llm_setting(
                super::super::settings::GLOBAL,
                SEARCH_ENGINE_ORDER_KEY,
                "not json",
            )
            .await
            .unwrap();
        let (garbled, _) = engines_of(&graph, "ord-s3").await;
        assert_eq!(garbled, before);
    }

    #[tokio::test]
    async fn a_declared_obscura_engine_runs_embedded_when_no_executable_is_configured() {
        let graph = Arc::new(MockGraphStore::new());
        let declared = SearchProviderRecord {
            id: "ddg".into(),
            engine: "obscura".into(),
            base_url: None,
            credential_ref: "none".into(),
        };
        store_provider(&graph, &declared).await;
        store_consent(&graph, "proj", OBSCURA_SEARCH_ORIGIN).await;
        let policy = ask();
        let attachment = attach(
            &embedded(),
            &GraphConsents(graph),
            None,
            input("bi-s5", &policy, Some("proj")),
        )
        .await;
        let (_, args, _) = stdio(&attachment, "nexus");
        let at = args.iter().position(|a| a == "--search-engine").unwrap();
        assert_eq!(args[at + 1], "obscura");
        assert_eq!(args.iter().filter(|a| *a == "--search-engine").count(), 1);
        assert!(attachment.skipped_engines.is_empty());
        crate::auth::agent_tokens::revoke_session("bi-s5");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn embedding_is_read_from_the_capabilities_of_nexus_tools() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, body: &str| {
            let path = dir.path().join(name);
            std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            path
        };
        let with = write(
            "with",
            r#"[ "$1" = --capabilities ] && echo '{"version":"0.5.1","tls":true,"obscura":true}'"#,
        );
        let without = write(
            "without",
            r#"echo '{"version":"0.5.1","tls":true,"obscura":false}'"#,
        );
        // An older nexus-tools: "unknown argument", exit 2.
        let older = write("older", "echo 'unknown argument' >&2; exit 2");
        assert!(embeds_obscura(&with).await);
        assert!(!embeds_obscura(&without).await);
        assert!(!embeds_obscura(&older).await);
        assert!(!embeds_obscura(&dir.path().join("missing")).await);
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
            obscura: None,
            embedded_obscura: false,
            deny_paths: Vec::new(),
            host_home: None,
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
