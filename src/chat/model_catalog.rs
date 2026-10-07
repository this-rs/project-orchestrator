//! Live Claude model catalog — merges Anthropic's Models API with local UI curation.
//!
//! Historically, the list of selectable Claude models was hardcoded in two
//! independent places (this backend's model list and the frontend's model
//! selector), which meant every model rename or addition required a manual
//! code change + release in both. This module fetches
//! `GET https://api.anthropic.com/v1/models` (when an API key is configured)
//! and caches the result in memory with a long TTL, so new/renamed models
//! show up automatically.
//!
//! Design goals:
//! - **Never block a request on network I/O.** Reads always return whatever
//!   is currently cached; a stale cache triggers a background refresh
//!   (stale-while-revalidate) rather than making the caller wait.
//! - **Never look broken.** If no API key is configured, or the Anthropic API
//!   is unreachable, we fall back to a small static list — the model
//!   selector always has *something* sensible to show.
//! - **Don't hammer the API.** The cache TTL is deliberately long (hours, not
//!   minutes) — a new model appearing a few hours late is fine.
//! - **No API key required.** When no `ANTHROPIC_API_KEY` is configured, the
//!   catalog borrows the OAuth token Claude Code itself logged in with (see
//!   `read_claude_code_oauth_token`). The Models API accepts it, and every
//!   install that can chat at all has one.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::events::{EntityType, EventEmitter};
use crate::neo4j::models::{AlertNode, AlertSeverity};
use crate::neo4j::GraphStore;

/// `alert_type` under which a newly released model is recorded. Doubles as
/// the durable "already announced" marker — see `announce_new_models`.
pub const MODEL_ADDED_ALERT: &str = "model_added";

/// How long a cached catalog is considered fresh before a background refresh
/// is triggered.
const CACHE_TTL: Duration = Duration::from_secs(12 * 60 * 60); // 12h

/// How long to wait before retrying after a FAILED refresh (expired login,
/// network down, Keychain locked). A failure must not freeze the catalog for a
/// full `CACHE_TTL`: the user would not see a model released meanwhile.
const RETRY_BACKOFF: Duration = Duration::from_secs(2 * 60);

/// HTTP timeout for the Anthropic Models API call itself.
const FETCH_TIMEOUT: Duration = Duration::from_secs(10);

const ANTHROPIC_MODELS_URL: &str = "https://api.anthropic.com/v1/models";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Defensive cap on pagination — the live catalog is small (dozens of
/// models at most); this just prevents a buggy/malicious `has_more` loop.
const MAX_PAGES: u8 = 5;

/// Beta header that makes the Anthropic API accept a Claude.ai OAuth access
/// token (`Authorization: Bearer`) instead of an `x-api-key`.
const OAUTH_BETA: &str = "oauth-2025-04-20";

/// macOS Keychain service under which Claude Code stores its login.
const KEYCHAIN_SERVICE: &str = "Claude Code-credentials";

/// Upper bound on the `security` call — a Keychain access prompt must never
/// wedge the background refresh.
const KEYCHAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// A single model entry, shaped for direct consumption by the frontend's
/// model selector.
///
/// Deliberately carries **semantic** tokens (`family`, `version`, `tier`)
/// rather than presentation values. An earlier revision shipped a raw
/// Tailwind class (`dot_color`) over the wire; that only ever worked because
/// the frontend happened to hardcode the same strings in a scanned source
/// file. Tailwind v4 runs with no config and no safelist here, so a class
/// that exists *only* in this Rust file is purged from the production bundle
/// — silently, with no build error and no failing test. Colors are now the
/// frontend's business; this module never names one.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ModelDefinition {
    /// Official Anthropic API model ID (e.g. "claude-sonnet-4-6")
    pub id: String,
    /// Model family, lowercase: "opus" | "sonnet" | "haiku" | "fable" | "other"
    pub family: String,
    /// Version as displayed, e.g. "5.5", "4.6". Empty when undeterminable.
    pub version: String,
    /// "current" (in the active lineup) | "legacy" (still served, superseded)
    pub tier: String,
    /// Short display label for compact UI (e.g. "Sonnet 4.6")
    pub short_label: String,
    /// Full marketing name (e.g. "Claude Sonnet 4.6")
    pub full_label: String,
    /// One-line description for selection cards
    pub description: String,
}

pub const TIER_CURRENT: &str = "current";
pub const TIER_LEGACY: &str = "legacy";

/// Known families, longest-match-first is irrelevant here (no overlap).
const KNOWN_FAMILIES: &[&str] = &["opus", "sonnet", "haiku", "fable", "mythos"];

/// Manual curation, in preferred display order.
/// `(id, family, version, tier, description)`.
///
/// `short_label` is derived as `"{Family} {version}"` and `full_label` as
/// `"Claude {short_label}"` — every model follows that pattern, so neither
/// is worth a tuple field. Adding a model is one line, and it carries no
/// presentation decision.
///
/// Models absent from this table (including brand-new ones the live API
/// returns) still show up — see `resolve_model` — with a derived label and
/// no description, until someone curates them.
const CURATED_ORDER: &[(&str, &str, &str, &str, &str)] = &[
    (
        "claude-opus-5-5",
        "opus",
        "5.5",
        TIER_CURRENT,
        "Recommended default — long-running agentic coding & knowledge work",
    ),
    (
        "claude-fable-5-1",
        "fable",
        "5.1",
        TIER_CURRENT,
        "Most capable — demanding reasoning & long-horizon agentic work",
    ),
    (
        "claude-sonnet-5-5",
        "sonnet",
        "5.5",
        TIER_CURRENT,
        "Best balance of speed and intelligence",
    ),
    (
        "claude-haiku-5-5",
        "haiku",
        "5.5",
        TIER_CURRENT,
        "Fastest — near-frontier intelligence",
    ),
    (
        "claude-haiku-4-5",
        "haiku",
        "4.5",
        TIER_LEGACY,
        "Legacy — superseded by Haiku 5.5",
    ),
    (
        "claude-opus-5",
        "opus",
        "5",
        TIER_LEGACY,
        "Legacy — superseded by Opus 5.5",
    ),
    (
        "claude-fable-5",
        "fable",
        "5",
        TIER_LEGACY,
        "Legacy — superseded by Fable 5.1",
    ),
    (
        "claude-sonnet-5",
        "sonnet",
        "5",
        TIER_LEGACY,
        "Legacy — superseded by Sonnet 5.5",
    ),
    (
        "claude-opus-4-8",
        "opus",
        "4.8",
        TIER_LEGACY,
        "Legacy Opus — complex reasoning",
    ),
    (
        "claude-opus-4-7",
        "opus",
        "4.7",
        TIER_LEGACY,
        "Legacy Opus — complex reasoning",
    ),
    (
        "claude-opus-4-6",
        "opus",
        "4.6",
        TIER_LEGACY,
        "Legacy Opus — complex reasoning",
    ),
    (
        "claude-sonnet-4-6",
        "sonnet",
        "4.6",
        TIER_LEGACY,
        "Legacy Sonnet — fast & capable",
    ),
];

/// Capitalize a family slug for display: "opus" -> "Opus".
fn family_label(family: &str) -> String {
    let mut chars = family.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// `("opus", "5.5")` -> `"Opus 5.5"`. Falls back to the family alone when
/// no version could be determined.
fn compose_short_label(family: &str, version: &str) -> String {
    let fam = family_label(family);
    if version.is_empty() {
        fam
    } else {
        format!("{fam} {version}")
    }
}

fn build_definition(
    id: &str,
    family: &str,
    version: &str,
    tier: &str,
    description: &str,
    full_label_override: Option<&str>,
) -> ModelDefinition {
    let short_label = compose_short_label(family, version);
    ModelDefinition {
        id: id.to_string(),
        family: family.to_string(),
        version: version.to_string(),
        tier: tier.to_string(),
        full_label: full_label_override
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Claude {short_label}")),
        short_label,
        description: description.to_string(),
    }
}

fn curated_lookup(id: &str) -> Option<ModelDefinition> {
    CURATED_ORDER.iter().find(|(cid, ..)| *cid == id).map(
        |(id, family, version, tier, description)| {
            build_definition(id, family, version, tier, description, None)
        },
    )
}

/// Split an unknown model ID into `(family, version)`.
///
/// `"claude-opus-4-9"` -> `("opus", "4.9")`; `"claude-foo-bar-7"` ->
/// `("other", "7")`. Trailing numeric segments are joined with dots.
fn derive_family_version(id: &str) -> (String, String) {
    let id = canonical_id(id);
    let without_prefix = id.strip_prefix("claude-").unwrap_or(id);
    let parts: Vec<&str> = without_prefix.split('-').collect();

    let mut num_parts: Vec<&str> = Vec::new();
    for part in parts.iter().rev() {
        if !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()) {
            num_parts.push(part);
        } else {
            break;
        }
    }
    num_parts.reverse();
    let version = num_parts.join(".");

    let family = KNOWN_FAMILIES
        .iter()
        .find(|f| parts.iter().any(|p| p.eq_ignore_ascii_case(f)))
        .map(|f| f.to_string())
        .unwrap_or_else(|| "other".to_string());

    (family, version)
}

/// Strip a trailing snapshot date: `"claude-haiku-4-5-20251001"` ->
/// `"claude-haiku-4-5"`. The Models API lists some models only under their
/// dated snapshot ID; without this they would neither match their curated
/// alias (showing up twice) nor derive a sane version (`"4.5.20251001"`).
fn canonical_id(id: &str) -> &str {
    match id.rsplit_once('-') {
        Some((head, tail)) if tail.len() == 8 && tail.chars().all(|c| c.is_ascii_digit()) => head,
        _ => id,
    }
}

/// `"5.5"` -> `[5, 5]`, for numeric (not lexicographic) version ordering.
fn version_key(version: &str) -> Vec<u32> {
    version.split('.').filter_map(|p| p.parse().ok()).collect()
}

/// Build a full `ModelDefinition` for a model ID the live API returned,
/// preferring curated data and falling back to derived heuristics.
///
/// An uncurated model is assumed `current`: the live Models API only lists
/// models that are actually available, and anything we have not curated yet
/// is far more likely to be newly released than retired.
fn resolve_model(id: &str, api_display_name: Option<&str>) -> ModelDefinition {
    if let Some(curated) = curated_lookup(id) {
        return curated;
    }
    let (family, version) = derive_family_version(id);
    build_definition(id, &family, &version, TIER_CURRENT, "", api_display_name)
}

/// The static list used when no API key is configured, or the live fetch
/// fails and nothing has ever been cached yet. Order matches `CURATED_ORDER`.
fn static_fallback_models() -> Vec<ModelDefinition> {
    CURATED_ORDER
        .iter()
        .map(|(id, ..)| curated_lookup(id).expect("id comes from CURATED_ORDER itself"))
        .collect()
}

#[derive(Debug, Deserialize)]
struct AnthropicModelsResponse {
    data: Vec<AnthropicModelEntry>,
    has_more: bool,
    #[serde(default)]
    last_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicModelEntry {
    id: String,
    #[serde(default)]
    display_name: Option<String>,
}

/// Where the live fetch gets its credential from.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CredentialSource {
    /// Never fetch — serve the static fallback. Used by tests and by the
    /// throwaway caches in handler test fixtures.
    None,
    /// An explicitly configured Anthropic API key.
    ApiKey(String),
    /// Reuse Claude Code's own login (read fresh on every refresh, since the
    /// CLI rotates the access token).
    ClaudeCodeLogin,
}

/// A credential resolved for one refresh.
enum Credential {
    ApiKey(String),
    OAuth(String),
}

/// Extract a still-valid access token from Claude Code's credentials JSON
/// (`{"claudeAiOauth": {"accessToken", "expiresAt" (ms), ...}}`).
///
/// An expired token is rejected rather than refreshed: refreshing would
/// rotate the refresh token behind the CLI's back and could log it out. The
/// CLI refreshes on its next use, and the next catalog refresh picks it up.
fn parse_claude_code_credentials(raw: &str, now_ms: i64) -> Option<String> {
    let json: serde_json::Value = serde_json::from_str(raw).ok()?;
    let oauth = json.get("claudeAiOauth")?;
    let token = oauth.get("accessToken")?.as_str()?.trim();
    if token.is_empty() {
        return None;
    }
    if let Some(expires_at) = oauth.get("expiresAt").and_then(|v| v.as_i64()) {
        if expires_at <= now_ms {
            return None;
        }
    }
    Some(token.to_string())
}

/// Find the OAuth access token Claude Code is logged in with.
///
/// Lookup order: `CLAUDE_CODE_OAUTH_TOKEN` (long-lived token from
/// `claude setup-token`), then the macOS Keychain, then
/// `$CLAUDE_CONFIG_DIR/.credentials.json` (default `~/.claude`, which is
/// where the CLI stores it on Linux).
async fn read_claude_code_oauth_token() -> Option<String> {
    if let Ok(token) = std::env::var("CLAUDE_CODE_OAUTH_TOKEN") {
        if !token.trim().is_empty() {
            return Some(token.trim().to_string());
        }
    }

    let now_ms = chrono::Utc::now().timestamp_millis();

    if cfg!(target_os = "macos") {
        let lookup = tokio::process::Command::new("security")
            .args(["find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"])
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true)
            .output();
        match tokio::time::timeout(KEYCHAIN_TIMEOUT, lookup).await {
            Ok(Ok(out)) if out.status.success() => {
                let raw = String::from_utf8_lossy(&out.stdout);
                if let Some(token) = parse_claude_code_credentials(&raw, now_ms) {
                    return Some(token);
                }
            }
            Ok(_) => {}
            Err(_) => tracing::warn!("Timed out reading Claude Code login from the Keychain"),
        }
    }

    let config_dir = std::env::var_os("CLAUDE_CONFIG_DIR")
        .map(std::path::PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".claude")))?;
    let raw = tokio::fs::read_to_string(config_dir.join(".credentials.json"))
        .await
        .ok()?;
    parse_claude_code_credentials(&raw, now_ms)
}

struct CacheState {
    models: Vec<ModelDefinition>,
    fetched_at: Instant,
    refreshing: bool,
    /// The last refresh failed: staleness is then measured against
    /// `RETRY_BACKOFF` instead of `CACHE_TTL`.
    last_failed: bool,
}

impl CacheState {
    fn is_stale(&self) -> bool {
        let ttl = if self.last_failed {
            RETRY_BACKOFF
        } else {
            CACHE_TTL
        };
        self.fetched_at.elapsed() >= ttl
    }
}

/// Handles needed to announce a newly released model. Optional so the cache
/// stays constructible in tests and in any context without a graph.
struct Notifier {
    emitter: Arc<dyn EventEmitter>,
    graph: Arc<dyn GraphStore>,
}

/// Shared, lazily-refreshed cache of the Claude model catalog.
pub struct ModelCatalogCache {
    inner: RwLock<CacheState>,
    http: reqwest::Client,
    credentials: CredentialSource,
    notifier: Option<Notifier>,
}

impl ModelCatalogCache {
    /// Construct a new cache. Does **not** make any network calls — the
    /// first call to `get_models()` seeds it with the static fallback and
    /// kicks off a background refresh if an API key is configured.
    pub fn new(api_key: Option<String>) -> Arc<Self> {
        let credentials = match api_key {
            Some(key) if !key.trim().is_empty() => CredentialSource::ApiKey(key),
            _ => CredentialSource::None,
        };
        Self::with_credentials(credentials)
    }

    fn with_credentials(credentials: CredentialSource) -> Arc<Self> {
        Arc::new(Self {
            inner: RwLock::new(CacheState {
                models: static_fallback_models(),
                // Force the first `get_models()` call to treat this as stale
                // so a real fetch is scheduled immediately when a key exists.
                fetched_at: Instant::now() - CACHE_TTL - Duration::from_secs(1),
                refreshing: false,
                last_failed: false,
            }),
            http: reqwest::Client::builder()
                .timeout(FETCH_TIMEOUT)
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            credentials,
            notifier: None,
        })
    }

    /// Production constructor: like `new`, plus the handles required to
    /// announce a model that appears in the live catalog for the first time.
    ///
    /// Without an API key, falls back to Claude Code's own login rather than
    /// to the static list — so a subscription-only install still sees new
    /// models without any configuration.
    pub fn new_with_notifier(
        api_key: Option<String>,
        emitter: Arc<dyn EventEmitter>,
        graph: Arc<dyn GraphStore>,
    ) -> Arc<Self> {
        let credentials = match api_key {
            Some(key) if !key.trim().is_empty() => CredentialSource::ApiKey(key),
            _ => CredentialSource::ClaudeCodeLogin,
        };
        Self::new_with_notifier_and_credentials(credentials, emitter, graph)
    }

    fn new_with_notifier_and_credentials(
        credentials: CredentialSource,
        emitter: Arc<dyn EventEmitter>,
        graph: Arc<dyn GraphStore>,
    ) -> Arc<Self> {
        let mut cache = Arc::try_unwrap(Self::with_credentials(credentials))
            .unwrap_or_else(|_| unreachable!("freshly created Arc is unique"));
        cache.notifier = Some(Notifier { emitter, graph });
        Arc::new(cache)
    }

    /// Return the current catalog, triggering a background refresh if the
    /// cache is stale (or empty) and no refresh is already in flight. Never
    /// blocks on network I/O — always returns immediately with whatever is
    /// cached (which is at minimum the static fallback list).
    pub async fn get_models(self: &Arc<Self>) -> Vec<ModelDefinition> {
        let needs_refresh = {
            let state = self.inner.read().await;
            !state.refreshing && state.is_stale()
        };

        if needs_refresh && self.credentials != CredentialSource::None {
            let mut state = self.inner.write().await;
            // Re-check under the write lock — another task may have started
            // the refresh between our read and this write.
            if !state.refreshing && state.is_stale() {
                state.refreshing = true;
                let this = Arc::clone(self);
                tokio::spawn(async move {
                    this.refresh().await;
                });
            }
        }

        self.inner.read().await.models.clone()
    }

    /// User-requested refresh ("Actualiser"). Never waits for the network:
    /// the fetch runs in a background task and the call returns at once.
    /// Returns `false` when nothing was started (no credentials, or a refresh
    /// is already in flight — the caller's wish is being served either way).
    /// A new model is announced over `/ws/events`; callers may also just
    /// re-read `get_models()` a few seconds later.
    pub async fn request_refresh(self: &Arc<Self>) -> bool {
        if self.credentials == CredentialSource::None {
            return false;
        }
        let mut state = self.inner.write().await;
        if state.refreshing {
            return false;
        }
        state.refreshing = true;
        drop(state);
        let this = Arc::clone(self);
        tokio::spawn(async move {
            this.refresh().await;
        });
        true
    }

    /// Whether a refresh is currently in flight.
    pub async fn is_refreshing(&self) -> bool {
        self.inner.read().await.refreshing
    }

    /// Run one refresh attempt to completion (the background task body).
    /// Falls back silently to the existing cache on any failure.
    async fn refresh(self: &Arc<Self>) {
        let result = self.fetch_live_catalog().await;

        // Announcing touches Neo4j, so it happens after the write lock is
        // released — holding it across that I/O would stall every reader.
        let mut announce: Option<Vec<ModelDefinition>> = None;
        let mut state = self.inner.write().await;
        match result {
            Ok(models) if !models.is_empty() => {
                tracing::info!(
                    count = models.len(),
                    "Refreshed Claude model catalog from Anthropic Models API"
                );
                announce = Some(models.clone());
                state.models = models;
                state.fetched_at = Instant::now();
                state.last_failed = false;
            }
            Ok(_) => {
                tracing::warn!(
                    "Anthropic Models API returned an empty catalog — keeping previous list"
                );
                // Still bump fetched_at so we don't hammer the API every request.
                state.fetched_at = Instant::now();
                state.last_failed = false;
            }
            Err(err) => {
                tracing::warn!(error = %err, "Failed to refresh Claude model catalog — keeping previous list");
                // Retry after the short backoff, not after 12h.
                state.fetched_at = Instant::now();
                state.last_failed = true;
            }
        }
        state.refreshing = false;
        drop(state);

        if let Some(models) = announce {
            self.announce_new_models(&models).await;
        }
    }

    /// Record and broadcast models seen in the live catalog for the first time.
    ///
    /// The durable "already announced" marker is an `Alert` node keyed by
    /// `dedup_key`, one per model, for the life of the graph. That is what
    /// makes this restart-safe: the in-memory cache is seeded from the static
    /// fallback on every boot, so diffing against it would re-announce every
    /// uncurated model each time the process restarts.
    ///
    /// The very first run establishes a baseline silently — with nothing
    /// recorded yet, every model would otherwise look new and the user would
    /// be buried in toasts for models that have existed for months.
    async fn announce_new_models(self: &Arc<Self>, models: &[ModelDefinition]) {
        let Some(notifier) = self.notifier.as_ref() else {
            return;
        };

        let known = match Self::known_model_keys(&notifier.graph).await {
            Ok(k) => k,
            Err(e) => {
                tracing::warn!(error = %e, "Could not read announced-model markers; skipping announcements");
                return;
            }
        };
        let baseline = known.is_empty();

        for model in models {
            let key = AlertNode::make_dedup_key(MODEL_ADDED_ALERT, None, &model.id);
            if known.contains(&key) {
                continue;
            }

            let alert = AlertNode::new_for_subject(
                MODEL_ADDED_ALERT.to_string(),
                AlertSeverity::Info,
                format!("{} is now available", model.full_label),
                None,
                &model.id,
            );

            if let Err(e) = notifier.graph.create_alert(&alert).await {
                tracing::warn!(model = %model.id, error = %e, "Failed to record new-model alert");
                continue;
            }

            if baseline {
                continue;
            }

            tracing::info!(model = %model.id, "New Claude model available");
            // `project_id: None` is what makes this app-wide: the WS filter
            // lets project-less events through whatever project a client
            // subscribed to.
            notifier.emitter.emit_created(
                EntityType::Alert,
                &alert.id.to_string(),
                serde_json::json!({
                    "alert_type": MODEL_ADDED_ALERT,
                    "model_id": model.id,
                    "full_label": model.full_label,
                    "family": model.family,
                    "version": model.version,
                }),
                None,
            );
        }

        if baseline {
            tracing::info!(
                count = models.len(),
                "Recorded model-catalog baseline; future additions will be announced"
            );
        }
    }

    /// Every `dedup_key` already recorded for an announced model.
    ///
    /// Paged rather than fetched in one shot: the alert store is shared with
    /// the heartbeat checks, so its size is not ours to assume. Capped so a
    /// pathological store cannot turn a background refresh into a long scan.
    async fn known_model_keys(graph: &Arc<dyn GraphStore>) -> anyhow::Result<HashSet<String>> {
        const PAGE: usize = 500;
        const MAX_PAGES: usize = 20;

        let mut keys = HashSet::new();
        for page in 0..MAX_PAGES {
            let (alerts, _total) = graph.list_alerts(None, None, PAGE, page * PAGE).await?;
            let fetched = alerts.len();
            keys.extend(
                alerts
                    .into_iter()
                    .filter(|a| a.alert_type == MODEL_ADDED_ALERT)
                    .map(|a| a.dedup_key),
            );
            if fetched < PAGE {
                break;
            }
        }
        Ok(keys)
    }

    async fn resolve_credential(&self) -> anyhow::Result<Credential> {
        match &self.credentials {
            CredentialSource::None => anyhow::bail!("no credential configured"),
            CredentialSource::ApiKey(key) => Ok(Credential::ApiKey(key.clone())),
            CredentialSource::ClaudeCodeLogin => read_claude_code_oauth_token()
                .await
                .map(Credential::OAuth)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "no ANTHROPIC_API_KEY and no valid Claude Code login found \
                         (run `claude` once to log in, or set anthropic.api_key)"
                    )
                }),
        }
    }

    async fn fetch_live_catalog(&self) -> anyhow::Result<Vec<ModelDefinition>> {
        let credential = self.resolve_credential().await?;

        let mut entries: Vec<AnthropicModelEntry> = Vec::new();
        let mut after_id: Option<String> = None;

        for _ in 0..MAX_PAGES {
            let mut req = self
                .http
                .get(ANTHROPIC_MODELS_URL)
                .header("anthropic-version", ANTHROPIC_VERSION);
            req = match &credential {
                Credential::ApiKey(key) => req.header("x-api-key", key),
                Credential::OAuth(token) => {
                    req.bearer_auth(token).header("anthropic-beta", OAUTH_BETA)
                }
            };
            if let Some(cursor) = &after_id {
                req = req.query(&[("after_id", cursor.as_str())]);
            }

            let resp = req.send().await?.error_for_status()?;
            let page: AnthropicModelsResponse = resp.json().await?;
            let has_more = page.has_more;
            let last_id = page.last_id.clone();
            entries.extend(page.data);

            if !has_more || last_id.is_none() {
                break;
            }
            after_id = last_id;
        }

        Ok(merge_catalog(&entries))
    }
}

/// Merge the live Models API listing with local curation.
///
/// Curated models first (in our preferred order), then anything the live
/// API knows about that we haven't curated yet, in API order.
///
/// Curated entries are ALWAYS included, even when absent from the Models
/// API response. Chat goes through the Claude Code CLI (its own
/// auth/routing), which can serve models the first-party Models API doesn't
/// list — e.g. `claude-fable-5` works via the CLI while being absent from
/// the API. Gating curation on API presence silently dropped such models
/// from the selector. Deactivating a model is an explicit act (remove its
/// CURATED_ORDER entry) rather than an implicit side effect of an API
/// listing change.
///
/// An uncurated model is `current` only if nothing newer of its family is
/// in the list: the API also lists older snapshots nobody has curated
/// (e.g. `claude-sonnet-4-5-20250929`), which must not be advertised as
/// part of the active lineup.
fn merge_catalog(entries: &[AnthropicModelEntry]) -> Vec<ModelDefinition> {
    let mut seen = HashSet::new();
    let mut models: Vec<ModelDefinition> = Vec::new();

    for (id, ..) in CURATED_ORDER {
        let api_entry = entries.iter().find(|e| canonical_id(&e.id) == *id);
        models.push(resolve_model(
            id,
            api_entry.and_then(|e| e.display_name.as_deref()),
        ));
        seen.insert(id.to_string());
    }

    let curated_count = models.len();
    for entry in entries {
        let canonical = canonical_id(&entry.id);
        if !seen.insert(canonical.to_string()) {
            continue;
        }
        models.push(resolve_model(&entry.id, entry.display_name.as_deref()));
    }

    let newest = |family: &str| {
        models
            .iter()
            .filter(|m| m.family == family)
            .map(|m| version_key(&m.version))
            .max()
            .unwrap_or_default()
    };
    let demote: Vec<usize> = (curated_count..models.len())
        .filter(|&i| {
            let m = &models[i];
            m.family != "other" && version_key(&m.version) < newest(&m.family)
        })
        .collect();
    for i in demote {
        models[i].tier = TIER_LEGACY.to_string();
    }

    models
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_family_version_known_families() {
        assert_eq!(
            derive_family_version("claude-opus-4-9"),
            ("opus".into(), "4.9".into())
        );
        assert_eq!(
            derive_family_version("claude-haiku-5"),
            ("haiku".into(), "5".into())
        );
        assert_eq!(
            derive_family_version("claude-fable-5-1"),
            ("fable".into(), "5.1".into())
        );
    }

    #[test]
    fn test_derive_family_version_unknown_family() {
        assert_eq!(
            derive_family_version("claude-foo-bar-7"),
            ("other".into(), "7".into())
        );
    }

    #[test]
    fn test_derive_family_version_no_trailing_digits() {
        // No version segment at all — must not panic, and must not swallow
        // an interior number as if it were the version.
        assert_eq!(
            derive_family_version("claude-opus-next"),
            ("opus".into(), "".into())
        );
    }

    #[test]
    fn test_compose_short_label() {
        assert_eq!(compose_short_label("opus", "5.5"), "Opus 5.5");
        assert_eq!(compose_short_label("sonnet", "4.6"), "Sonnet 4.6");
        // Versionless model degrades to the family alone rather than
        // rendering a trailing space.
        assert_eq!(compose_short_label("opus", ""), "Opus");
    }

    #[test]
    fn test_definitions_carry_no_presentation_values() {
        // Guards the Tailwind purge trap: a color class shipped from here
        // would be absent from every file Tailwind scans, and would be
        // stripped from the production bundle with no error anywhere.
        let json = serde_json::to_string(&static_fallback_models()).unwrap();
        assert!(
            !json.contains("bg-"),
            "presentation class leaked into the API payload"
        );
    }

    #[test]
    fn test_every_curated_entry_is_self_consistent() {
        for (id, family, version, tier, _) in CURATED_ORDER {
            assert!(
                KNOWN_FAMILIES.contains(family),
                "{id}: unknown family {family}"
            );
            assert!(
                *tier == TIER_CURRENT || *tier == TIER_LEGACY,
                "{id}: bad tier {tier}"
            );
            // The curated family/version must agree with what the ID itself
            // says, so a typo in the table cannot silently mislabel a model.
            let (derived_family, derived_version) = derive_family_version(id);
            assert_eq!(derived_family, *family, "{id}: family disagrees with ID");
            assert_eq!(derived_version, *version, "{id}: version disagrees with ID");
        }
    }

    #[test]
    fn test_curated_lookup_fable_and_sonnet_both_resolve() {
        // Fable 5 was briefly renamed away when Sonnet 5 launched, then
        // reintroduced as its own selectable entry alongside Sonnet 5 —
        // both IDs must resolve independently via curation.
        let fable = curated_lookup("claude-fable-5");
        assert!(fable.is_some());
        assert_eq!(fable.unwrap().short_label, "Fable 5");
        assert!(curated_lookup("claude-sonnet-5").is_some());
        assert!(curated_lookup("claude-sonnet-5-5").is_some());
    }

    #[test]
    fn test_static_fallback_models_nonempty_and_ordered() {
        let models = static_fallback_models();
        assert!(!models.is_empty());
        assert_eq!(models[0].id, "claude-opus-5-5");
        assert_eq!(models[1].id, "claude-fable-5-1");
        assert_eq!(models[2].id, "claude-sonnet-5-5");
    }

    #[test]
    fn test_resolve_model_prefers_curated_over_api_display_name() {
        let m = resolve_model("claude-sonnet-5", Some("Some Other Name"));
        assert_eq!(m.full_label, "Claude Sonnet 5");
        assert_eq!(m.short_label, "Sonnet 5");
    }

    #[test]
    fn test_resolve_model_falls_back_for_unknown_id() {
        let m = resolve_model("claude-new-hotness-9", Some("Claude New Hotness 9"));
        assert_eq!(m.full_label, "Claude New Hotness 9");
        assert_eq!(m.family, "other");
        assert_eq!(m.version, "9");
        assert_eq!(m.tier, TIER_CURRENT);
        assert_eq!(m.description, "");
    }

    #[test]
    fn test_canonical_id_strips_snapshot_date_only() {
        assert_eq!(
            canonical_id("claude-haiku-4-5-20251001"),
            "claude-haiku-4-5"
        );
        assert_eq!(canonical_id("claude-sonnet-5-5"), "claude-sonnet-5-5");
        // A 7-digit tail is not a date.
        assert_eq!(canonical_id("claude-x-1234567"), "claude-x-1234567");
        assert_eq!(
            derive_family_version("claude-opus-4-5-20251101"),
            ("opus".into(), "4.5".into())
        );
    }

    fn entry(id: &str, name: &str) -> AnthropicModelEntry {
        AnthropicModelEntry {
            id: id.into(),
            display_name: Some(name.into()),
        }
    }

    #[test]
    fn test_merge_catalog_matches_dated_ids_to_curated_aliases() {
        let models = merge_catalog(&[
            entry("claude-sonnet-5-5", "Claude Sonnet 5.5"),
            entry("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
        ]);
        let haikus: Vec<_> = models
            .iter()
            .filter(|m| m.family == "haiku" && m.version == "4.5")
            .collect();
        assert_eq!(
            haikus.len(),
            1,
            "dated snapshot must not duplicate the curated alias"
        );
        assert_eq!(haikus[0].id, "claude-haiku-4-5");
    }

    #[test]
    fn test_merge_catalog_uncurated_models_tiered_by_family_recency() {
        let models = merge_catalog(&[
            entry("claude-sonnet-6", "Claude Sonnet 6"),
            entry("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5"),
        ]);
        let find = |id: &str| models.iter().find(|m| m.id == id).unwrap();
        // Newest of its family -> advertised as current.
        assert_eq!(find("claude-sonnet-6").tier, TIER_CURRENT);
        assert_eq!(find("claude-sonnet-6").version, "6");
        // An old uncurated snapshot -> legacy, with a clean version.
        let old = find("claude-sonnet-4-5-20250929");
        assert_eq!(old.tier, TIER_LEGACY);
        assert_eq!(old.short_label, "Sonnet 4.5");
    }

    #[test]
    fn test_merge_catalog_keeps_curated_tiers() {
        // Curation stays authoritative even when an uncurated newer model
        // exists; only uncurated entries are auto-tiered.
        let models = merge_catalog(&[entry("claude-opus-6", "Claude Opus 6")]);
        let opus55 = models.iter().find(|m| m.id == "claude-opus-5-5").unwrap();
        assert_eq!(opus55.tier, TIER_CURRENT);
    }

    #[test]
    fn test_version_key_is_numeric() {
        assert!(version_key("4.10") > version_key("4.9"));
        assert!(version_key("5") < version_key("5.5"));
    }

    #[test]
    fn test_parse_claude_code_credentials() {
        let now = 1_000_000;
        let valid = r#"{"claudeAiOauth":{"accessToken":"tok","expiresAt":2000000}}"#;
        assert_eq!(
            parse_claude_code_credentials(valid, now),
            Some("tok".into())
        );

        let expired = r#"{"claudeAiOauth":{"accessToken":"tok","expiresAt":999999}}"#;
        assert_eq!(parse_claude_code_credentials(expired, now), None);

        let no_expiry = r#"{"claudeAiOauth":{"accessToken":"tok"}}"#;
        assert_eq!(
            parse_claude_code_credentials(no_expiry, now),
            Some("tok".into())
        );

        assert_eq!(parse_claude_code_credentials(r#"{"other":{}}"#, now), None);
        assert_eq!(parse_claude_code_credentials("not json", now), None);
        let empty = r#"{"claudeAiOauth":{"accessToken":"  "}}"#;
        assert_eq!(parse_claude_code_credentials(empty, now), None);
    }

    #[test]
    fn test_production_constructor_uses_claude_code_login_without_key() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let emitter = Arc::new(RecordingEmitter::default());
        let cache = ModelCatalogCache::new_with_notifier(None, emitter.clone(), graph.clone());
        assert_eq!(cache.credentials, CredentialSource::ClaudeCodeLogin);

        let cache = ModelCatalogCache::new_with_notifier(Some("k".into()), emitter, graph);
        assert_eq!(cache.credentials, CredentialSource::ApiKey("k".into()));

        // The plain constructor (test fixtures) must stay offline.
        assert_eq!(
            ModelCatalogCache::new(None).credentials,
            CredentialSource::None
        );
    }

    /// Records every CrudEvent it is handed, so a test can assert on what
    /// actually reached the bus.
    #[derive(Default)]
    struct RecordingEmitter {
        events: std::sync::Mutex<Vec<crate::events::CrudEvent>>,
    }

    impl crate::events::EventEmitter for RecordingEmitter {
        fn emit(&self, event: crate::events::CrudEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    impl RecordingEmitter {
        fn model_ids(&self) -> Vec<String> {
            self.events
                .lock()
                .unwrap()
                .iter()
                .filter(|e| {
                    e.payload.get("alert_type").and_then(|v| v.as_str()) == Some(MODEL_ADDED_ALERT)
                })
                .filter_map(|e| {
                    e.payload
                        .get("model_id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                })
                .collect()
        }
    }

    fn model(id: &str) -> ModelDefinition {
        resolve_model(id, None)
    }

    async fn cache_with(
        graph: Arc<dyn GraphStore>,
        emitter: Arc<RecordingEmitter>,
    ) -> Arc<ModelCatalogCache> {
        ModelCatalogCache::new_with_notifier_and_credentials(CredentialSource::None, emitter, graph)
    }

    #[tokio::test]
    async fn test_first_run_records_baseline_without_announcing() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let emitter = Arc::new(RecordingEmitter::default());
        let cache = cache_with(graph.clone(), emitter.clone()).await;

        cache
            .announce_new_models(&[model("claude-opus-5-5"), model("claude-sonnet-5")])
            .await;

        // Nothing announced: with no markers recorded, every model would look
        // new and the user would be buried in toasts for old models.
        assert!(
            emitter.model_ids().is_empty(),
            "baseline run must not announce anything"
        );
        // But the markers ARE persisted, so the next run has a reference point.
        let (alerts, _) = graph.list_alerts(None, None, 100, 0).await.unwrap();
        assert_eq!(alerts.len(), 2);
    }

    #[tokio::test]
    async fn test_only_genuinely_new_models_are_announced() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let emitter = Arc::new(RecordingEmitter::default());
        let cache = cache_with(graph.clone(), emitter.clone()).await;

        // Establish the baseline.
        cache
            .announce_new_models(&[model("claude-opus-5-5"), model("claude-sonnet-5")])
            .await;
        assert!(emitter.model_ids().is_empty());

        // A later refresh turns up one extra model.
        cache
            .announce_new_models(&[
                model("claude-opus-5-5"),
                model("claude-sonnet-5"),
                model("claude-opus-6"),
            ])
            .await;

        assert_eq!(emitter.model_ids(), vec!["claude-opus-6".to_string()]);
    }

    #[tokio::test]
    async fn test_restart_does_not_reannounce() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let catalog = vec![model("claude-opus-5-5"), model("claude-sonnet-5")];

        // First process: baseline.
        let e1 = Arc::new(RecordingEmitter::default());
        cache_with(graph.clone(), e1.clone())
            .await
            .announce_new_models(&catalog)
            .await;

        // Second process: fresh cache, same graph. The in-memory cache is
        // reseeded from the static fallback on every boot, so this is exactly
        // the case that would spam on every restart if the markers were not
        // durable.
        let e2 = Arc::new(RecordingEmitter::default());
        cache_with(graph.clone(), e2.clone())
            .await
            .announce_new_models(&catalog)
            .await;

        assert!(
            e2.model_ids().is_empty(),
            "a restart must not re-announce known models"
        );
    }

    #[tokio::test]
    async fn test_announcement_is_app_wide() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let emitter = Arc::new(RecordingEmitter::default());
        let cache = cache_with(graph.clone(), emitter.clone()).await;

        cache.announce_new_models(&[model("claude-opus-5-5")]).await;
        cache
            .announce_new_models(&[model("claude-opus-5-5"), model("claude-opus-6")])
            .await;

        let events = emitter.events.lock().unwrap();
        let announced = events
            .iter()
            .find(|e| {
                e.payload.get("alert_type").and_then(|v| v.as_str()) == Some(MODEL_ADDED_ALERT)
            })
            .expect("one announcement expected");
        // project_id: None is what makes the WS filter deliver this to every
        // client regardless of which project they subscribed to.
        assert!(
            announced.project_id.is_none(),
            "a model announcement must not be scoped to a project"
        );
    }

    #[tokio::test]
    async fn test_request_refresh_without_credentials_starts_nothing() {
        let cache = ModelCatalogCache::new(None);
        assert!(!cache.request_refresh().await);
        assert!(!cache.is_refreshing().await);
    }

    #[tokio::test]
    async fn test_failed_refresh_retries_after_short_backoff_not_ttl() {
        let cache = ModelCatalogCache::with_credentials(CredentialSource::ApiKey("k".into()));
        {
            let mut st = cache.inner.write().await;
            st.fetched_at = Instant::now() - RETRY_BACKOFF - Duration::from_secs(1);
            st.last_failed = true;
            assert!(st.is_stale(), "a failure is retried after the backoff");
            st.last_failed = false;
            assert!(!st.is_stale(), "a success stays fresh for the whole TTL");
        }
    }

    #[test]
    fn test_haiku_5_5_is_current_and_4_5_is_legacy() {
        let m = static_fallback_models();
        let h55 = m
            .iter()
            .find(|x| x.id == "claude-haiku-5-5")
            .expect("haiku 5.5");
        assert_eq!(h55.tier, TIER_CURRENT);
        let h45 = m
            .iter()
            .find(|x| x.id == "claude-haiku-4-5")
            .expect("haiku 4.5");
        assert_eq!(h45.tier, TIER_LEGACY);
    }

    #[tokio::test]
    async fn test_get_models_without_api_key_returns_static_fallback() {
        let cache = ModelCatalogCache::new(None);
        let models = cache.get_models().await;
        assert_eq!(models, static_fallback_models());
        // No API key — must never attempt a refresh (would flip `refreshing`
        // to true and stay there since fetch_live_catalog short-circuits an
        // error, but we assert the simpler, load-bearing invariant: no crash,
        // stable fallback content).
        let models_again = cache.get_models().await;
        assert_eq!(models_again, models);
    }
}
