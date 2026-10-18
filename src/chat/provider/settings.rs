//! Settings of the provider harness: instances, consent, roles, aliases and
//! model policy (decisions A15, A19, A24, A25, A28, A32).
//!
//! Pure: parsing, validation and the shapes stored as JSON documents (one per
//! scope and key, see `GraphStore::get_llm_setting`). Nothing here reads a
//! secret: an instance carries a credential REFERENCE, never a value, and a
//! body that names a secret field is refused outright.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::endpoint_guard::{origin_of, validate_url, EndpointPolicy, EndpointRefusal};
use super::resolver::CLAUDE_CODE;

/// Setting scope of server-wide documents.
pub const GLOBAL: &str = "global";
/// Key prefix of instance documents.
pub const INSTANCE_PREFIX: &str = "instance:";
/// Key prefix of consent documents (scope `project:<slug>`, or [`NO_PROJECT_SCOPE`]).
pub const CONSENT_PREFIX: &str = "consent:";
/// Scope of the consents that hold for the sessions outside any project (B3): given by a
/// person through `PUT /api/chat/llm-consents/{provider_id}`, bound to the instance's origin
/// and credential reference exactly like a project's consent (A28). A project never inherits
/// them: a project consents for itself.
pub const NO_PROJECT_SCOPE: &str = "no_project";
/// Key of the role assignment document.
pub const ROLES_KEY: &str = "roles";
/// Key of the alias table.
pub const ALIASES_KEY: &str = "model_aliases";
/// Key of the model policy.
pub const POLICY_KEY: &str = "model_policy";

/// Scope of a project's documents.
pub fn project_scope(slug: &str) -> String {
    format!("project:{slug}")
}

/// Scope of the consents that apply to a session: its project's, or the no-project scope.
pub fn consent_scope(project_slug: Option<&str>) -> String {
    match project_slug {
        Some(slug) => project_scope(slug),
        None => NO_PROJECT_SCOPE.to_string(),
    }
}

/// Why a body was refused. Messages never echo a value the caller sent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettingsError {
    /// The body is not acceptable; the text says which rule.
    Invalid(String),
    /// The endpoint guard refused the URL.
    Endpoint(EndpointRefusal),
    /// The built-in instance cannot be changed.
    Builtin,
    /// The instance does not exist.
    UnknownInstance(String),
}

impl std::fmt::Display for SettingsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(why) => f.write_str(why),
            Self::Endpoint(r) => write!(f, "endpoint refused: {r}"),
            Self::Builtin => f.write_str("the built-in claude-code instance cannot be changed"),
            Self::UnknownInstance(id) => write!(f, "unknown provider instance '{id}'"),
        }
    }
}

impl std::error::Error for SettingsError {}

fn invalid(why: &str) -> SettingsError {
    SettingsError::Invalid(why.to_string())
}

/// Where a credential lives. The value is never held here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CredentialSource {
    /// No credential.
    None,
    /// Read from the vault under that name.
    Vault(String),
    /// Read from that environment variable of the server.
    Env(String),
}

/// Environment variable naming the server variables an instance may use as a
/// credential (`env:<VAR>`): a comma-separated list. EMPTY BY DEFAULT: with no
/// list, `env:` references are refused, so a provider cannot be pointed at a
/// variable of the server that nobody declared (deny by default, not a
/// block-list of the secrets we thought of).
pub const ENV_CREDENTIALS_VAR: &str = "CHAT_PROVIDER_ENV_CREDENTIALS";

/// The declared list of `env:` credential variables.
pub fn env_credential_allowlist() -> Vec<String> {
    parse_allowlist(&std::env::var(ENV_CREDENTIALS_VAR).unwrap_or_default())
}

/// Parses a comma-separated list of variable names.
pub fn parse_allowlist(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .map(str::to_string)
        .collect()
}

/// Parses `none`, `vault:<name>` or `env:<VAR>`. `env:` is accepted only for a
/// variable declared in `env_allow` (and never one of the server's own secrets,
/// even if declared).
pub fn parse_credential_ref(
    raw: &str,
    env_allow: &[String],
) -> Result<CredentialSource, SettingsError> {
    let raw = raw.trim();
    if raw == "none" || raw.is_empty() {
        return Ok(CredentialSource::None);
    }
    let name_ok = |n: &str| {
        !n.is_empty()
            && n.len() <= 128
            && n.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/'))
    };
    if let Some(name) = raw.strip_prefix("vault:") {
        return if name_ok(name) {
            Ok(CredentialSource::Vault(name.to_string()))
        } else {
            Err(invalid("credential_ref: malformed vault name"))
        };
    }
    if let Some(var) = raw.strip_prefix("env:") {
        if !name_ok(var) {
            return Err(invalid("credential_ref: malformed variable name"));
        }
        if crate::chat::manager::SERVER_ONLY_SECRETS.contains(&var)
            || !env_allow.iter().any(|a| a == var)
        {
            return Err(invalid(
                "credential_ref: that variable is not declared for provider credentials \
                 (CHAT_PROVIDER_ENV_CREDENTIALS); use vault:<name>",
            ));
        }
        return Ok(CredentialSource::Env(var.to_string()));
    }
    // A bare value (a pasted key) is exactly what must never be stored.
    Err(invalid(
        "credential_ref must be `none`, `vault:<name>` or `env:<VAR>`: never a secret value",
    ))
}

/// Body of `POST /chat/providers[/test]`. Unknown fields (an `api_key`, a
/// `token`...) are refused: no secret is accepted in these bodies.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstanceDraft {
    /// Instance identifier (slug).
    pub id: Option<String>,
    /// Kind: `openai_compatible` (default), `anthropic`, `codex`, `acp` or `claude_code_remote`.
    pub kind: Option<String>,
    /// Preset (deepseek, vllm, ollama, llama_server, nim), informational.
    #[serde(default)]
    pub preset: Option<String>,
    /// Display label.
    #[serde(default)]
    pub label: Option<String>,
    /// Endpoint base URL.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Default model.
    #[serde(default)]
    pub default_model: Option<String>,
    /// Where the cost comes from.
    #[serde(default)]
    pub cost_source: Option<String>,
    /// Credential reference.
    #[serde(default)]
    pub credential_ref: Option<String>,
    /// `claude_code_remote`: host name or address of the machine.
    #[serde(default)]
    pub host: Option<String>,
    /// `claude_code_remote`: remote user (the ssh default when absent).
    #[serde(default)]
    pub ssh_user: Option<String>,
    /// `claude_code_remote`: ssh port (22 when absent).
    #[serde(default)]
    pub ssh_port: Option<u16>,
    /// `claude_code_remote`: the PINNED public key of the host, `<type> <base64>`.
    #[serde(default)]
    pub host_key: Option<String>,
    /// `claude_code_remote`: working directory ON THE REMOTE machine.
    #[serde(default)]
    pub remote_cwd: Option<String>,
    /// `claude_code_remote`: allow the no-confirmation mode on that machine.
    #[serde(default)]
    pub allow_trust: Option<bool>,
    /// OPTIONAL limits of the instance (RL1). Absent or empty = no limit at all:
    /// the behaviour is then exactly what it was before this field existed.
    #[serde(default)]
    pub rate_limit: Vec<RateLimit>,
}

/// Body of `PUT|PATCH /chat/providers/{id}`: the id and the kind never change.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstancePatch {
    /// Preset.
    #[serde(default)]
    pub preset: Option<String>,
    /// Display label.
    #[serde(default)]
    pub label: Option<String>,
    /// Endpoint base URL.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Default model.
    #[serde(default)]
    pub default_model: Option<String>,
    /// Cost source.
    #[serde(default)]
    pub cost_source: Option<String>,
    /// Credential reference.
    #[serde(default)]
    pub credential_ref: Option<String>,
    /// `claude_code_remote`: host name or address of the machine.
    #[serde(default)]
    pub host: Option<String>,
    /// `claude_code_remote`: remote user (the ssh default when absent).
    #[serde(default)]
    pub ssh_user: Option<String>,
    /// `claude_code_remote`: ssh port (22 when absent).
    #[serde(default)]
    pub ssh_port: Option<u16>,
    /// `claude_code_remote`: the PINNED public key of the host, `<type> <base64>`.
    #[serde(default)]
    pub host_key: Option<String>,
    /// `claude_code_remote`: working directory ON THE REMOTE machine.
    #[serde(default)]
    pub remote_cwd: Option<String>,
    /// `claude_code_remote`: allow the no-confirmation mode on that machine.
    #[serde(default)]
    pub allow_trust: Option<bool>,
    /// OPTIONAL limits (RL1). `None` = leave the stored list untouched; `Some([])`
    /// = remove every limit. The field is never a secret.
    #[serde(default)]
    pub rate_limit: Option<Vec<RateLimit>>,
}

/// A stored provider instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct InstanceRecord {
    /// Identifier.
    pub id: String,
    /// Kind.
    pub kind: String,
    /// Preset.
    #[serde(default)]
    pub preset: Option<String>,
    /// Label.
    pub label: String,
    /// Base URL (no credential in it: refused at validation).
    pub base_url: String,
    /// Normalised origin; what a consent is tied to.
    pub origin: String,
    /// Default model.
    #[serde(default)]
    pub default_model: Option<String>,
    /// Cost source.
    pub cost_source: String,
    /// Credential reference.
    pub credential_ref: String,
    /// `claude_code_remote`: host of the machine.
    #[serde(default)]
    pub host: Option<String>,
    /// `claude_code_remote`: remote user.
    #[serde(default)]
    pub ssh_user: Option<String>,
    /// `claude_code_remote`: ssh port.
    #[serde(default)]
    pub ssh_port: Option<u16>,
    /// `claude_code_remote`: the pinned public key, `<type> <base64>`.
    #[serde(default)]
    pub host_key: Option<String>,
    /// `claude_code_remote`: working directory on the remote machine.
    #[serde(default)]
    pub remote_cwd: Option<String>,
    /// `claude_code_remote`: the no-confirmation mode is allowed on that machine.
    #[serde(default)]
    pub allow_trust: bool,
    /// The configured limits (RL1). Empty = no limit: the instance behaves exactly
    /// as it did before this field existed. A record stored before the field
    /// existed reads back with an empty list, without any migration.
    #[serde(default)]
    pub rate_limit: Vec<RateLimit>,
}

impl InstanceRecord {
    /// OpenSSH fingerprint (`SHA256:...`) of the pinned host key, when there is one.
    pub fn host_key_fingerprint(&self) -> Option<String> {
        self.host_key.as_deref().and_then(host_key_fingerprint)
    }
}

/// The kind of an instance that runs Claude Code on another machine over SSH.
pub const KIND_CLAUDE_CODE_REMOTE: &str = "claude_code_remote";

/// The kind of an instance that runs the NATIVE harness over the Anthropic Messages API
/// (CTX5): Anthropic's own API by default, or a gateway that speaks it. Its key is a
/// vault reference, never an `env:` variable nor a value.
pub const KIND_ANTHROPIC: &str = "anthropic";

/// Base URL of an `anthropic` instance that names none: Anthropic's own API.
pub const ANTHROPIC_API_BASE: &str = "https://api.anthropic.com";

// ============================================================================
// Rate limits (RL1)
// ============================================================================
//
// An instance may carry an OPTIONAL list of limits: a number of requests or
// tokens per window, or a budget in US dollars per calendar window (day, week,
// month). Each limit applies to the whole instance, to one credential REFERENCE
// or to one model.
//
// Two families, and they are not the same question (see the plan): a RATE limit
// answers "how much may go out per window", a BUDGET limit answers "how much may
// this cost per day/week/month". A budget window is a CALENDAR window, so it
// carries its time zone: midnight is not the same instant everywhere.
//
// Nothing here holds a secret. A credential scope names a credential REFERENCE
// (`vault:<name>`, `env:<VAR>`) and the validation runs it through the very same
// parser the instance uses, so a bare key (`sk-...`) is refused like anywhere
// else. Absence of the field is not a default limit: it means NO limit, and the
// behaviour stays exactly what it was before the field existed.

/// Upper bound on how many limits one instance may carry. A bound exists so a
/// body cannot make the settings document unbounded.
pub const MAX_RATE_LIMITS: usize = 32;

/// What a limit counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RateLimitKind {
    /// Requests sent to the provider.
    Requests,
    /// Tokens consumed.
    Tokens,
    /// Money spent, in US dollars.
    BudgetUsd,
}

/// A unit window of a limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RateLimitUnit {
    /// A rolling minute, aligned on the clock's minute.
    Minute,
    /// An hour.
    Hour,
    /// A calendar day (needs a time zone).
    Day,
    /// A calendar week (needs a time zone).
    Week,
    /// A calendar month (needs a time zone).
    Month,
}

impl RateLimitUnit {
    /// Whether the window is a CALENDAR window: its boundaries are midnight in a
    /// time zone, so the limit must say which one.
    pub fn is_calendar(self) -> bool {
        matches!(self, Self::Day | Self::Week | Self::Month)
    }

    /// The word the interface shows ("minute", "hour"...).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Minute => "minute",
            Self::Hour => "hour",
            Self::Day => "day",
            Self::Week => "week",
            Self::Month => "month",
        }
    }
}

/// The object form of a rolling window: `{"rolling": 3600}`, in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RollingWindow {
    /// Window length in seconds.
    pub rolling: u64,
}

/// A limit's window: a unit window (`"minute"`) or a rolling one
/// (`{"rolling": 3600}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RateLimitWindow {
    /// A named window.
    Unit(RateLimitUnit),
    /// A window that slides.
    Rolling(RollingWindow),
}

impl RateLimitWindow {
    /// The unit, when the window is a named one.
    pub fn unit(self) -> Option<RateLimitUnit> {
        match self {
            Self::Unit(u) => Some(u),
            Self::Rolling(_) => None,
        }
    }

    /// Whether the window is a CALENDAR one (day, week, month).
    pub fn is_calendar(self) -> bool {
        self.unit().is_some_and(RateLimitUnit::is_calendar)
    }
}

/// The `"instance"` scope, as a bare string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InstanceScope {
    /// The whole instance.
    Instance,
}

/// `{"credential": "vault:<name>"}`: one credential REFERENCE, never a value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CredentialScope {
    /// The credential reference (`vault:<name>`, `env:<VAR>`).
    pub credential: String,
}

/// `{"model": "<id>"}`: one model of the instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelScope {
    /// Model identifier, as the instance knows it.
    pub model: String,
}

/// Whom a limit applies to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RateLimitScope {
    /// The whole instance (`"instance"`).
    Instance(InstanceScope),
    /// One credential reference.
    Credential(CredentialScope),
    /// One model.
    Model(ModelScope),
}

impl RateLimitScope {
    /// The credential REFERENCE this scope names, if any. Never a value.
    pub fn credential(&self) -> Option<&str> {
        match self {
            Self::Credential(CredentialScope { credential }) => Some(credential),
            _ => None,
        }
    }

    /// A short label for a status line or a reason ("instance", "vault:deepseek",
    /// "model deepseek-chat"). Never a secret: it echoes a reference or an id.
    pub fn label(&self) -> String {
        match self {
            Self::Instance(_) => "instance".to_string(),
            Self::Credential(CredentialScope { credential }) => credential.clone(),
            Self::Model(ModelScope { model }) => format!("model {model}"),
        }
    }
}

/// One configured limit on an instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateLimit {
    /// What it counts.
    pub kind: RateLimitKind,
    /// Over which window.
    pub window: RateLimitWindow,
    /// Time zone of a calendar window: `"utc"` or an IANA name. Absent (null) for
    /// `minute`, `hour` and rolling windows.
    #[serde(default)]
    pub calendar: Option<String>,
    /// The ceiling: strictly positive, and a whole number for requests and tokens.
    pub max: f64,
    /// Whom it applies to.
    pub scope: RateLimitScope,
}

/// Whether a time zone is known: `utc` in any case, or an IANA zone name.
pub fn known_timezone(raw: &str) -> bool {
    let name = raw.trim();
    if name.eq_ignore_ascii_case("utc") {
        return true;
    }
    name.parse::<chrono_tz::Tz>().is_ok()
}

/// Validates ONE limit. Messages say which rule, never echo a value.
fn validate_rate_limit(limit: &RateLimit, env_allow: &[String]) -> Result<(), SettingsError> {
    if !limit.max.is_finite() || limit.max <= 0.0 {
        return Err(invalid("rate_limit.max: strictly positive"));
    }
    match limit.kind {
        // A COUNT is a whole number: half a request means nothing.
        RateLimitKind::Requests | RateLimitKind::Tokens => {
            if limit.max.fract() != 0.0 {
                return Err(invalid(
                    "rate_limit.max: a whole number of requests or tokens",
                ));
            }
        }
        // A budget is money: a fraction of a dollar is fine.
        RateLimitKind::BudgetUsd => {
            let unit = limit.window.unit();
            if !matches!(
                unit,
                Some(RateLimitUnit::Day) | Some(RateLimitUnit::Week) | Some(RateLimitUnit::Month)
            ) {
                return Err(invalid(
                    "rate_limit.window: a budget is per day, week or month",
                ));
            }
        }
    }
    match (limit.calendar.as_deref(), limit.window.is_calendar()) {
        (Some(tz), true) => {
            if !known_timezone(tz) {
                return Err(invalid(
                    "rate_limit.calendar: utc or an IANA time-zone name",
                ));
            }
        }
        (None, true) => {
            return Err(invalid(
                "rate_limit.calendar: a day, week or month window needs its time zone",
            ));
        }
        (Some(_), false) => {
            return Err(invalid(
                "rate_limit.calendar: only a day, week or month window has one",
            ));
        }
        (None, false) => {}
    }
    if let RateLimitWindow::Rolling(RollingWindow { rolling }) = limit.window {
        if rolling == 0 {
            return Err(invalid("rate_limit.window.rolling: at least one second"));
        }
    }
    match &limit.scope {
        RateLimitScope::Instance(_) => {}
        RateLimitScope::Credential(CredentialScope { credential }) => {
            if credential.trim().is_empty() || credential.trim() == "none" {
                return Err(invalid(
                    "rate_limit.scope.credential: names a credential reference",
                ));
            }
            // The same parser the instance uses: `env:` must be declared, and a
            // bare key is refused — a limit never stores a secret.
            parse_credential_ref(credential, env_allow)?;
        }
        RateLimitScope::Model(ModelScope { model }) => {
            let model = model.trim();
            if model.is_empty() || model.len() > 200 || model.contains(['\0', '\n', '\r']) {
                return Err(invalid("rate_limit.scope.model: a model identifier"));
            }
        }
    }
    Ok(())
}

/// Validates a whole list: each limit, the ceiling on how many, and no duplicate
/// (kind + window + scope).
pub fn validate_rate_limits(
    limits: &[RateLimit],
    env_allow: &[String],
) -> Result<(), SettingsError> {
    if limits.len() > MAX_RATE_LIMITS {
        return Err(invalid("rate_limit: at most 32 limits per instance"));
    }
    for (i, limit) in limits.iter().enumerate() {
        validate_rate_limit(limit, env_allow)?;
        let duplicate = limits[..i].iter().any(|seen| {
            seen.kind == limit.kind && seen.window == limit.window && seen.scope == limit.scope
        });
        if duplicate {
            return Err(invalid(
                "rate_limit: two limits share the same kind, window and scope",
            ));
        }
    }
    Ok(())
}

// ----------------------------------------------------------------------------
// The state of those limits (RL7)
// ----------------------------------------------------------------------------
//
// `GET /chat/providers/{id}/limits` answers one entry per configured limit, with
// its current state. Nothing measures anything yet: RL2 (counters) and RL4
// (breaker) are not built, so every limit reports `unknown` with null numbers.
// What is settled here is the SHAPE — adding the numbers later will not change a
// key. No field of it is ever a secret.

/// The current state of one limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RateLimitStatus {
    /// Under the limit. Nothing to report.
    Ok,
    /// OUR limit is reached (or the provider confirmed a quota): nothing is sent
    /// until `resets_at`.
    Limited,
    /// The provider is overloaded or intermittent while our own count is under
    /// the limit: retry later, the key is NOT exhausted. RL3 draws that line.
    Overloaded,
    /// Nothing is measured yet. Never confused with `limited`.
    Unknown,
}

/// One limit and its current state, as the status route answers it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateLimitState {
    /// What the limit counts.
    pub kind: RateLimitKind,
    /// Its window.
    pub window: RateLimitWindow,
    /// Time zone of a calendar window (null otherwise).
    pub calendar: Option<String>,
    /// Its ceiling.
    pub max: f64,
    /// Whom it applies to.
    pub scope: RateLimitScope,
    /// The credential REFERENCE this limit is scoped to, when it is scoped to one
    /// (else null). A reference, never a value.
    pub credential: Option<String>,
    /// How much was used in the current window. Null while nothing is counted.
    pub used: Option<f64>,
    /// What is left. Null while nothing is counted.
    pub remaining: Option<f64>,
    /// When the window (or the quota) resets, RFC 3339. Null when unknown.
    pub resets_at: Option<String>,
    /// `ok | limited | overloaded | unknown`.
    pub status: RateLimitStatus,
    /// Since when the current state holds, RFC 3339. Null when none holds.
    pub since: Option<String>,
    /// When the next attempt may be made, RFC 3339. Null when none is planned.
    pub next_retry_at: Option<String>,
}

/// The whole answer of `GET /chat/providers/{id}/limits`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateLimitReport {
    /// The instance the limits belong to.
    pub provider_id: String,
    /// The worst state among the limits (`ok` when there is no limit at all).
    pub status: RateLimitStatus,
    /// One entry per configured limit.
    pub limits: Vec<RateLimitState>,
}

/// Builds the CURRENT state of an instance's limits.
///
/// No counter exists yet (RL2) and no breaker either (RL4), so every limit is
/// `unknown` with null usage, and an instance without limits is `ok` (nothing can
/// limit it). RL2/RL4 fill `used`, `remaining`, `resets_at` and the status without
/// touching a single key of this shape.
pub fn rate_limit_report(record: &InstanceRecord) -> RateLimitReport {
    let limits: Vec<RateLimitState> = record
        .rate_limit
        .iter()
        .map(|limit| RateLimitState {
            kind: limit.kind,
            window: limit.window,
            calendar: limit.calendar.clone(),
            max: limit.max,
            scope: limit.scope.clone(),
            credential: limit.scope.credential().map(str::to_string),
            used: None,
            remaining: None,
            resets_at: None,
            status: RateLimitStatus::Unknown,
            since: None,
            next_retry_at: None,
        })
        .collect();
    RateLimitReport {
        provider_id: record.id.clone(),
        // Nothing configured = nothing that can be limited.
        status: if limits.is_empty() {
            RateLimitStatus::Ok
        } else {
            RateLimitStatus::Unknown
        },
        limits,
    }
}

/// Wire name of the event that announces a change of a provider's limit state.
pub const RATE_LIMIT_EVENT: &str = "provider_rate_limit";

/// The event that announces a change of a provider's limit state (RL7).
///
/// SHAPE ONLY: nothing emits it yet — RL2 (counters) and RL4 (breaker) will — but
/// the shape is settled here and in `tests/fixtures/rate_limit/event.json`, so the
/// interface and the chat contract can be built against it. It carries no secret:
/// `credential` and `scope` name a credential REFERENCE.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderRateLimitEvent {
    /// Event name on the wire (`provider_rate_limit`).
    #[serde(rename = "type")]
    pub event_type: String,
    /// The instance whose state changed.
    pub provider_id: String,
    /// The credential REFERENCE the changed limit is scoped to, when it is scoped
    /// to one. Never a value.
    pub credential: Option<String>,
    /// Whom the changed limit applies to.
    pub scope: RateLimitScope,
    /// What it counts.
    pub kind: RateLimitKind,
    /// Its window.
    pub window: RateLimitWindow,
    /// The new state.
    pub status: RateLimitStatus,
    /// `quota` when the limit (ours or the provider's) is reached, `overload` for
    /// an intermittent provider: this is the distinction RL3 exists to make.
    #[serde(default)]
    pub reason: Option<String>,
    /// When the limit resets, RFC 3339.
    #[serde(default)]
    pub resets_at: Option<String>,
    /// When the next attempt may be made, RFC 3339.
    #[serde(default)]
    pub next_retry_at: Option<String>,
}

/// OpenSSH fingerprint of a `<type> <base64>` public key: `SHA256:` and the
/// unpadded base64 of the SHA-256 of the decoded key blob (what `ssh-keygen -lf`
/// prints). `None` when the text is not a well-formed key.
pub fn host_key_fingerprint(host_key: &str) -> Option<String> {
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD};
    use base64::Engine;
    use sha2::{Digest, Sha256};
    let (_, blob) = parse_host_key(host_key).ok()?;
    let raw = STANDARD.decode(blob).ok()?;
    Some(format!(
        "SHA256:{}",
        STANDARD_NO_PAD.encode(Sha256::digest(raw))
    ))
}

/// Splits `<type> <base64>`: exactly two words, a key type and a base64 blob.
pub fn parse_host_key(raw: &str) -> Result<(&str, &str), SettingsError> {
    let mut parts = raw.split_whitespace();
    let (kind, blob, rest) = (parts.next(), parts.next(), parts.next());
    match (kind, blob, rest) {
        (Some(k), Some(b), None)
            if (k.starts_with("ssh-") || k.starts_with("ecdsa-") || k.starts_with("sk-"))
                && k.len() <= 64
                && b.len() >= 16
                && b.len() <= 8192
                && b.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"+/=".contains(&c)) =>
        {
            Ok((k, b))
        }
        _ => Err(invalid(
            "host_key: exactly '<type> <base64>' (the key is pinned, never learned)",
        )),
    }
}

/// A host or user name, same rule as the transport: letters, digits and
/// `. _ - : @ [ ] %`, never starting with `-` (ssh would read an option).
pub fn valid_ssh_word(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 253
        && !value.starts_with('-')
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-:@[]%".contains(&c))
}

fn valid_remote_cwd(cwd: &str) -> bool {
    !cwd.trim().is_empty() && cwd.len() <= 1024 && !cwd.contains(['\0', '\n', '\r'])
}

/// Identity a consent of the remote kind is tied to: where the session's
/// content goes, `ssh:[user@]host:port`.
pub fn ssh_origin(host: &str, user: Option<&str>, port: Option<u16>) -> String {
    format!(
        "ssh:{}{}:{}",
        user.map(|u| format!("{u}@")).unwrap_or_default(),
        host,
        port.unwrap_or(22)
    )
}

/// Slug of a label for an instance id: lowercase letters, digits and dashes.
fn slugify(label: &str) -> String {
    let mut out = String::new();
    for c in label.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_string()
}

const COST_SOURCES: [&str; 5] = ["reported", "priced", "free", "subscription", "unknown"];

pub(crate) fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 48
        && id
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

fn check_url(url: &str, policy: &EndpointPolicy) -> Result<String, SettingsError> {
    validate_url(url, policy).map_err(SettingsError::Endpoint)?;
    origin_of(url).ok_or_else(|| invalid("base_url: not a URL"))
}

/// Environment variable declaring the ACP agents an instance may launch: a
/// JSON object, `{"opencode": ["opencode", "acp"]}`. EMPTY BY DEFAULT: an API
/// body never carries a command line (that would be remote code execution as
/// the server's user); it names a declared one by `preset`. An entry may also be
/// `{"argv": [...], "per_session_mcp": false}`: the agent takes no MCP server per
/// session (see [`AcpCommand`]).
pub const ACP_COMMANDS_VAR: &str = "CHAT_PROVIDER_ACP_COMMANDS";

/// One declared ACP agent: the command that starts it and, when the operator says
/// so, whether it takes MCP servers per session.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(untagged)]
enum AcpCommandDecl {
    Argv(Vec<String>),
    Full(AcpCommandFull),
}

/// The object form of a declared ACP agent.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AcpCommandFull {
    argv: Vec<String>,
    #[serde(default)]
    per_session_mcp: Option<bool>,
}

/// A declared ACP agent ([`ACP_COMMANDS_VAR`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcpCommand {
    /// Program and arguments.
    pub argv: Vec<String>,
    /// What the operator declared: `Some(false)` for an agent that takes no MCP
    /// server per session (an OpenClaw launched through a wrapper, an older OpenClaw
    /// that IGNORES them without a word). `None`: [`acp_command_carries_mcp`] decides.
    pub per_session_mcp: Option<bool>,
}

impl AcpCommand {
    /// Whether the agent takes MCP servers per session: the declaration, else the
    /// name of a known agent among the arguments.
    pub fn carries_mcp(&self) -> bool {
        self.per_session_mcp
            .unwrap_or_else(|| acp_command_carries_mcp(&self.argv))
    }
}

/// The declared ACP commands.
pub fn acp_commands() -> std::collections::BTreeMap<String, AcpCommand> {
    parse_acp_commands(&std::env::var(ACP_COMMANDS_VAR).unwrap_or_default())
}

/// Parses the declaration: a malformed whole declares nothing, a malformed entry (an
/// unknown field, a misspelt `per_session_mcp`) is left out, never half-read.
pub fn parse_acp_commands(raw: &str) -> std::collections::BTreeMap<String, AcpCommand> {
    serde_json::from_str::<std::collections::BTreeMap<String, serde_json::Value>>(raw)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(name, value)| match serde_json::from_value::<AcpCommandDecl>(value) {
            Ok(decl) => Some((name, decl)),
            Err(error) => {
                tracing::warn!(
                    agent = %name,
                    %error,
                    "{ACP_COMMANDS_VAR}: malformed entry left out (its instances answer that the agent is not declared)"
                );
                None
            }
        })
        .map(|(name, decl)| {
            let command = match decl {
                AcpCommandDecl::Argv(argv) => AcpCommand {
                    argv,
                    per_session_mcp: None,
                },
                AcpCommandDecl::Full(AcpCommandFull {
                    argv,
                    per_session_mcp,
                }) => AcpCommand {
                    argv,
                    per_session_mcp,
                },
            };
            (name, command)
        })
        .filter(|(name, command)| {
            valid_id(name) && !command.argv.is_empty() && !command.argv[0].trim().is_empty()
        })
        .collect()
}

/// ACP agents known to take no MCP server per session, by the name of their program.
/// `openclaw acp` refuses `mcpServers` at `session/new` / `session/load` with an error
/// (2026.9.x: "ACP bridge mode does not support per-session MCP servers"; older
/// versions ignored them in silence). Its MCP servers are configured on the OpenClaw
/// gateway (`openclaw mcp set`).
const ACP_AGENTS_WITHOUT_PER_SESSION_MCP: &[&str] = &["openclaw"];

/// Launchers that run the program named after them (their options skipped): `npx`,
/// `bunx`, `pnpx`, `npm exec|x`, `pnpm dlx|exec`, `yarn dlx|exec`, `env`, `node` (its
/// script). Those with a subcommand ([`LAUNCHER_SUBCOMMANDS`]) run a program only
/// through it.
const ACP_LAUNCHERS: &[&str] = &["npx", "bunx", "pnpx", "npm", "pnpm", "yarn", "env", "node"];

/// The launchers that run a program only through a subcommand, and those subcommands
/// (`npm install openclaw`, `pnpm add openclaw` run nothing named after them).
const LAUNCHER_SUBCOMMANDS: &[(&str, &[&str])] = &[
    ("npm", &["exec", "x"]),
    ("pnpm", &["dlx", "exec"]),
    ("yarn", &["dlx", "exec"]),
];

/// Options of those launchers that take the NEXT argument as their value
/// (`npx -p <pkg>`, `env -u <var>`, `node -r <module>`, `pnpm -C <dir>`,
/// `npm exec --registry <url>`): that value is not a program.
const LAUNCHER_OPTIONS_WITH_VALUE: &[&str] = &[
    "-p",
    "--package",
    "-u",
    "--unset",
    "-C",
    "--chdir",
    "-r",
    "--require",
    "--import",
    "--loader",
    "--experimental-loader",
    "--env-file",
    "--filter",
    "--dir",
    "--cwd",
    "--prefix",
    "--workspace",
    "--registry",
];

/// How deep a command line given to a launcher ([`command_option`]) is read, nested
/// in another one.
const LAUNCHER_COMMAND_DEPTH: usize = 4;

/// A command line split into words as a shell would for its first words: on white
/// space, single and double quotes grouping (and removed). Nothing else is
/// interpreted.
fn command_words(line: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    for c in line.chars() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => word.push(c),
            None if c == '\'' || c == '"' => {
                quote = Some(c);
                in_word = true;
            }
            None if c.is_whitespace() => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            None => {
                word.push(c);
                in_word = true;
            }
        }
    }
    if in_word {
        words.push(word);
    }
    words
}

/// The name a program position designates: its file name (both separators: a
/// declaration may name a Windows path on any host), a package version removed
/// (`openclaw@2026.9.9`), at most one extension removed (`openclaw.mjs`,
/// `OpenClaw.cmd`), lowercased. `None` for a URL.
fn program_name(arg: &str) -> Option<String> {
    if arg.contains("://") {
        return None;
    }
    let file = arg.rsplit(['/', '\\']).next().unwrap_or(arg);
    let file = match file.char_indices().skip(1).find(|&(_, c)| c == '@') {
        Some((at, _)) => &file[..at],
        None => file,
    };
    let name = file.rsplit_once('.').map_or(file, |(stem, _)| stem);
    Some(name.to_ascii_lowercase())
}

/// Whether `arg` is the option of `launcher` that hands it a COMMAND LINE to run (its
/// first word is a program): `Some(None)` when the line is the NEXT
/// argument (`npx -c "…"`, `env -S "…"`), `Some(Some(line))` when it is glued to it
/// (`npx --call=…`, `env --split-string=…`, `env -S…`). `env` reads `-S`, `node` none
/// (its `-c` checks a script), the package launchers `-c`.
fn command_option<'a>(launcher: &str, arg: &'a str) -> Option<Option<&'a str>> {
    let (short, long) = match launcher {
        "env" => ("-S", "--split-string"),
        "node" => return None,
        _ => ("-c", "--call"),
    };
    if arg == short || arg == long {
        return Some(None);
    }
    if let Some(line) = arg
        .strip_prefix(long)
        .and_then(|rest| rest.strip_prefix('='))
    {
        return Some(Some(line));
    }
    if launcher == "env" {
        if let Some(line) = arg.strip_prefix(short).filter(|line| !line.is_empty()) {
            return Some(Some(line));
        }
    }
    None
}

/// The programs a command runs, as far as its arguments tell: `argv[0]`, and, while
/// that program is a known launcher ([`ACP_LAUNCHERS`]), the first argument after it
/// (after its subcommand, [`LAUNCHER_SUBCOMMANDS`]) that is neither an option, an
/// option's value, nor an `env` assignment (`npx --yes openclaw`, `env FOO=1
/// openclaw`, `pnpm -C <dir> dlx openclaw`, `npm exec -- openclaw`, `node
/// …/openclaw.mjs`, `env npx openclaw`), and the first word of a command line given to
/// it (`npx -c "openclaw acp"`, `env -S "openclaw acp"`). Nothing else: an option's
/// value, a path or a URL among the arguments is never taken for the program.
fn acp_programs(argv: &[String]) -> Vec<String> {
    let mut programs = Vec::new();
    collect_programs(argv, 0, &mut programs);
    programs
}

/// [`acp_programs`] of `argv`, a command line `depth` levels deep in another one.
fn collect_programs(argv: &[String], depth: usize, programs: &mut Vec<String>) {
    let mut at = 0;
    while let Some(arg) = argv.get(at) {
        let Some(name) = program_name(arg) else {
            break;
        };
        let launcher = ACP_LAUNCHERS.contains(&name.as_str());
        programs.push(name.clone());
        if !launcher {
            break;
        }
        at += 1;
        // `pnpm -c exec "…"`: the positional after the subcommand is a command line.
        let mut shell = false;
        if let Some((_, subcommands)) = LAUNCHER_SUBCOMMANDS.iter().find(|(l, _)| *l == name) {
            loop {
                match argv.get(at).map(String::as_str) {
                    Some("-c" | "--shell-mode") => {
                        shell = true;
                        at += 1;
                    }
                    Some(option) if LAUNCHER_OPTIONS_WITH_VALUE.contains(&option) => at += 2,
                    Some(option) if option.starts_with('-') => at += 1,
                    _ => break,
                }
            }
            match argv.get(at) {
                Some(sub) if subcommands.contains(&sub.as_str()) => at += 1,
                // Any other command runs no program named after it.
                _ => break,
            }
        }
        while let Some(next) = argv.get(at) {
            if let Some(glued) = command_option(&name, next) {
                let (line, after) = match glued {
                    Some(line) => (Some(line), at + 1),
                    None => (argv.get(at + 1).map(String::as_str), at + 2),
                };
                if depth >= LAUNCHER_COMMAND_DEPTH {
                    return;
                }
                let words = command_words(line.unwrap_or_default());
                if name == "env" {
                    // `env -S "A=1 openclaw"` is `env A=1 openclaw`: the words are env's
                    // own arguments, followed by the ones after them.
                    let mut spliced = vec![name.clone()];
                    spliced.extend(words);
                    spliced.extend(argv.iter().skip(after).cloned());
                    collect_programs(&spliced, depth + 1, programs);
                    return;
                }
                shell_programs(&words, depth, programs);
                at = after;
            } else if LAUNCHER_OPTIONS_WITH_VALUE.contains(&next.as_str()) {
                at += 2;
            } else if next.starts_with('-') || (name == "env" && next.contains('=')) {
                at += 1;
            } else if shell {
                if depth < LAUNCHER_COMMAND_DEPTH {
                    shell_programs(&command_words(next), depth, programs);
                }
                return;
            } else {
                break;
            }
        }
    }
}

/// The programs of a shell command line (`npx -c "…"`): its first word after the
/// variable assignments (`OC_DEBUG=1 openclaw acp`).
fn shell_programs(words: &[String], depth: usize, programs: &mut Vec<String>) {
    let assignment = |word: &String| {
        word.split_once('=').is_some_and(|(name, _)| {
            name.chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
    };
    let start = words.iter().take_while(|word| assignment(word)).count();
    collect_programs(&words[start..], depth + 1, programs);
}

/// Whether the ACP agent this command starts takes MCP servers per session, as far
/// as its arguments tell: `false` when a PROGRAM it runs ([`acp_programs`]: `argv[0]`,
/// or what a known launcher runs) is a known agent that refuses them
/// ([`ACP_AGENTS_WITHOUT_PER_SESSION_MCP`]) — `openclaw`, `/usr/bin/openclaw`,
/// `npx openclaw acp`, `npm exec openclaw`, `yarn dlx openclaw`, `pnpm -C <dir> dlx
/// openclaw`, `npx -c "openclaw acp"`, `env -S "openclaw acp"`, `node
/// …/openclaw.mjs`, `env openclaw`, `OpenClaw.cmd`. An
/// option value (`--config ~/.openclaw/openclaw.json`, `--profile openclaw`) or a URL
/// (`wss://gw/openclaw`) is not a program. A convenience: the declaration's
/// `per_session_mcp` wins over it.
pub fn acp_command_carries_mcp(argv: &[String]) -> bool {
    !acp_programs(argv)
        .iter()
        .any(|name| ACP_AGENTS_WITHOUT_PER_SESSION_MCP.contains(&name.as_str()))
}

/// Kinds that run a local process instead of calling an endpoint: they have no
/// URL, their consent is tied to a process identity.
pub fn is_process_kind(kind: &str) -> bool {
    matches!(kind, "codex" | "acp" | KIND_CLAUDE_CODE_REMOTE)
}

/// Identity a consent of a process kind is tied to (it plays the role of the origin).
pub fn process_origin(kind: &str, preset: Option<&str>) -> String {
    match (kind, preset) {
        ("acp", Some(name)) => format!("process:acp:{name}"),
        _ => format!("process:{kind}"),
    }
}

/// Validates a draft into a record (syntax only; the DNS check is async and
/// done by the caller through `validate_endpoint`).
pub fn record_from_draft(
    draft: &InstanceDraft,
    policy: &EndpointPolicy,
    env_allow: &[String],
) -> Result<InstanceRecord, SettingsError> {
    let kind = draft
        .kind
        .clone()
        .unwrap_or_else(|| "openai_compatible".into());
    let remote = kind == KIND_CLAUDE_CODE_REMOTE;
    // A remote instance has no free-form id: `claude-code@<slug of the label>`.
    // `claude-code` alone stays the built-in local instance and is refused for
    // every user instance, whatever its kind.
    let id = if remote {
        if draft.id.as_deref() == Some(CLAUDE_CODE) {
            return Err(SettingsError::Builtin);
        }
        let name = slugify(draft.label.as_deref().unwrap_or_default());
        if name.is_empty() {
            return Err(invalid(
                "label: a name is required (it makes the instance id)",
            ));
        }
        let name: String = name.chars().take(36).collect();
        format!("{CLAUDE_CODE}@{}", name.trim_end_matches('-'))
    } else {
        draft.id.clone().ok_or_else(|| invalid("id is required"))?
    };
    if id == CLAUDE_CODE {
        return Err(SettingsError::Builtin);
    }
    if !remote && !valid_id(&id) {
        return Err(invalid("id: lowercase letters, digits and dashes only"));
    }
    if !matches!(
        kind.as_str(),
        "openai_compatible" | "codex" | "acp" | KIND_CLAUDE_CODE_REMOTE | KIND_ANTHROPIC
    ) {
        return Err(invalid(
            "kind: openai_compatible, anthropic, codex, acp or claude_code_remote",
        ));
    }
    if kind == KIND_ANTHROPIC && draft.preset.is_some() {
        return Err(invalid(
            "preset: an anthropic instance has no dialect preset (it speaks the Messages API)",
        ));
    }
    if !remote && draft_has_remote_fields(draft) {
        return Err(invalid(
            "host, ssh_user, ssh_port, host_key, remote_cwd and allow_trust belong to claude_code_remote",
        ));
    }
    let (base_url, origin) = if remote {
        if draft
            .base_url
            .as_deref()
            .is_some_and(|u| !u.trim().is_empty())
        {
            return Err(invalid(
                "base_url: not used by a claude_code_remote instance",
            ));
        }
        let host = draft.host.as_deref().unwrap_or_default();
        if !valid_ssh_word(host) {
            return Err(invalid(
                "host: a plain name or address (letters, digits and . _ - : [ ] % @, not starting with '-')",
            ));
        }
        if draft
            .ssh_user
            .as_deref()
            .is_some_and(|u| !valid_ssh_word(u))
        {
            return Err(invalid("ssh_user: a plain user name"));
        }
        if draft.ssh_port == Some(0) {
            return Err(invalid("ssh_port: between 1 and 65535"));
        }
        parse_host_key(draft.host_key.as_deref().unwrap_or_default())?;
        if host_key_fingerprint(draft.host_key.as_deref().unwrap_or_default()).is_none() {
            return Err(invalid("host_key: the key is not valid base64"));
        }
        if !draft.remote_cwd.as_deref().is_some_and(valid_remote_cwd) {
            return Err(invalid(
                "remote_cwd: the working directory on the remote machine is required",
            ));
        }
        (
            String::new(),
            ssh_origin(host, draft.ssh_user.as_deref(), draft.ssh_port),
        )
    } else if is_process_kind(&kind) {
        // A process instance has no URL, and an API body never carries a command.
        if draft
            .base_url
            .as_deref()
            .is_some_and(|u| !u.trim().is_empty())
        {
            return Err(invalid("base_url: not used by a codex or acp instance"));
        }
        if kind == "acp" {
            let name = draft
                .preset
                .as_deref()
                .ok_or_else(|| invalid("preset: names the ACP agent declared on the server"))?;
            if !acp_commands().contains_key(name) {
                return Err(invalid(
                    "preset: that ACP agent is not declared on this server (CHAT_PROVIDER_ACP_COMMANDS)",
                ));
            }
        }
        (
            String::new(),
            process_origin(&kind, draft.preset.as_deref()),
        )
    } else {
        let base_url = draft
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .or_else(|| (kind == KIND_ANTHROPIC).then(|| ANTHROPIC_API_BASE.to_string()))
            .ok_or_else(|| invalid("base_url is required"))?;
        let origin = check_url(&base_url, policy)?;
        (base_url, origin)
    };
    let credential_ref = draft
        .credential_ref
        .clone()
        .unwrap_or_else(|| "none".into());
    if remote {
        // The SSH private key lives in the vault: never an `env:` variable of the
        // server, never a value, never nothing (a remote machine needs a key).
        require_vault_ref(&credential_ref)?;
    } else if kind == KIND_ANTHROPIC {
        require_vault_api_key(&credential_ref)?;
    } else {
        parse_credential_ref(&credential_ref, env_allow)?;
    }
    if kind == "acp" && credential_ref != "none" {
        return Err(invalid("credential_ref: an ACP agent holds its own login"));
    }
    let cost_source = draft
        .cost_source
        .clone()
        .unwrap_or_else(|| "unknown".into());
    if !COST_SOURCES.contains(&cost_source.as_str()) {
        return Err(invalid(
            "cost_source: reported, priced, free, subscription or unknown",
        ));
    }
    // RL1: the limits are optional, but when they are given they are checked
    // here — an instance never stores a limit it would refuse to apply.
    validate_rate_limits(&draft.rate_limit, env_allow)?;
    Ok(InstanceRecord {
        label: draft
            .label
            .clone()
            .filter(|l| !l.trim().is_empty())
            .unwrap_or_else(|| id.clone()),
        id,
        kind,
        preset: draft.preset.clone(),
        base_url,
        origin,
        default_model: draft.default_model.clone().filter(|m| !m.trim().is_empty()),
        cost_source,
        credential_ref,
        host: remote.then(|| draft.host.clone()).flatten(),
        ssh_user: remote.then(|| draft.ssh_user.clone()).flatten(),
        ssh_port: remote.then_some(draft.ssh_port).flatten(),
        host_key: remote
            .then(|| draft.host_key.as_deref().map(normalize_host_key))
            .flatten(),
        remote_cwd: remote.then(|| draft.remote_cwd.clone()).flatten(),
        allow_trust: remote && draft.allow_trust.unwrap_or(false),
        rate_limit: draft.rate_limit.clone(),
    })
}

fn draft_has_remote_fields(d: &InstanceDraft) -> bool {
    d.host.is_some()
        || d.ssh_user.is_some()
        || d.ssh_port.is_some()
        || d.host_key.is_some()
        || d.remote_cwd.is_some()
        || d.allow_trust.is_some()
}

fn patch_has_remote_fields(p: &InstancePatch) -> bool {
    p.host.is_some()
        || p.ssh_user.is_some()
        || p.ssh_port.is_some()
        || p.host_key.is_some()
        || p.remote_cwd.is_some()
        || p.allow_trust.is_some()
}

/// One space between the two words, nothing around them.
fn normalize_host_key(raw: &str) -> String {
    raw.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The key of an `anthropic` instance: `vault:<name>`, nothing else (CTX5).
fn require_vault_api_key(raw: &str) -> Result<(), SettingsError> {
    match parse_credential_ref(raw, &[])? {
        CredentialSource::Vault(_) => Ok(()),
        _ => Err(invalid(
            "credential_ref: an anthropic instance needs `vault:<name>` (the API key, stored in the vault)",
        )),
    }
}

/// The credential of a remote instance: `vault:<name>`, nothing else.
fn require_vault_ref(raw: &str) -> Result<(), SettingsError> {
    match parse_credential_ref(raw, &[])? {
        CredentialSource::Vault(_) => Ok(()),
        _ => Err(invalid(
            "credential_ref: a remote instance needs `vault:<name>` (the SSH private key, stored in the vault)",
        )),
    }
}

/// Applies a patch to a record; returns the new record and whether the ORIGIN
/// changed (every consent given for the old origin then stops holding).
pub fn apply_patch(
    old: &InstanceRecord,
    patch: &InstancePatch,
    policy: &EndpointPolicy,
    env_allow: &[String],
) -> Result<(InstanceRecord, bool), SettingsError> {
    let mut next = old.clone();
    if let Some(label) = patch.label.as_ref().filter(|l| !l.trim().is_empty()) {
        next.label = label.clone();
    }
    if patch.preset.is_some() {
        if old.kind == KIND_ANTHROPIC {
            return Err(invalid(
                "preset: an anthropic instance has no dialect preset (it speaks the Messages API)",
            ));
        }
        next.preset = patch.preset.clone();
    }
    if patch.base_url.is_some() && is_process_kind(&old.kind) {
        return Err(invalid("base_url: not used by a process instance"));
    }
    let remote = old.kind == KIND_CLAUDE_CODE_REMOTE;
    if !remote && patch_has_remote_fields(patch) {
        return Err(invalid(
            "host, ssh_user, ssh_port, host_key, remote_cwd and allow_trust belong to claude_code_remote",
        ));
    }
    if remote {
        if let Some(host) = patch.host.as_ref() {
            if !valid_ssh_word(host) {
                return Err(invalid("host: a plain name or address"));
            }
            next.host = Some(host.clone());
        }
        if let Some(user) = patch.ssh_user.as_ref() {
            // An empty string clears the user (the ssh default).
            if user.is_empty() {
                next.ssh_user = None;
            } else if valid_ssh_word(user) {
                next.ssh_user = Some(user.clone());
            } else {
                return Err(invalid("ssh_user: a plain user name"));
            }
        }
        if let Some(port) = patch.ssh_port {
            if port == 0 {
                return Err(invalid("ssh_port: between 1 and 65535"));
            }
            next.ssh_port = Some(port);
        }
        if let Some(key) = patch.host_key.as_ref() {
            parse_host_key(key)?;
            if host_key_fingerprint(key).is_none() {
                return Err(invalid("host_key: the key is not valid base64"));
            }
            next.host_key = Some(normalize_host_key(key));
        }
        if let Some(cwd) = patch.remote_cwd.as_ref() {
            if !valid_remote_cwd(cwd) {
                return Err(invalid("remote_cwd: a directory on the remote machine"));
            }
            next.remote_cwd = Some(cwd.clone());
        }
        if let Some(trust) = patch.allow_trust {
            next.allow_trust = trust;
        }
        next.origin = ssh_origin(
            next.host.as_deref().unwrap_or_default(),
            next.ssh_user.as_deref(),
            next.ssh_port,
        );
    }
    if let Some(url) = patch.base_url.as_ref() {
        next.origin = check_url(url, policy)?;
        next.base_url = url.clone();
    }
    if patch.default_model.is_some() {
        next.default_model = patch.default_model.clone().filter(|m| !m.trim().is_empty());
    }
    if let Some(cost) = patch.cost_source.as_ref() {
        if !COST_SOURCES.contains(&cost.as_str()) {
            return Err(invalid(
                "cost_source: reported, priced, free, subscription or unknown",
            ));
        }
        next.cost_source = cost.clone();
    }
    if let Some(cred) = patch.credential_ref.as_ref() {
        if remote {
            require_vault_ref(cred)?;
        } else if old.kind == KIND_ANTHROPIC {
            require_vault_api_key(cred)?;
        } else {
            parse_credential_ref(cred, env_allow)?;
        }
        next.credential_ref = cred.clone();
    }
    // RL1: `Some([])` clears every limit, `None` leaves them untouched. A limit
    // never changes where content goes or with what, so it touches no consent.
    if let Some(limits) = patch.rate_limit.as_ref() {
        validate_rate_limits(limits, env_allow)?;
        next.rate_limit = limits.clone();
    }
    // A consent is tied to the origin AND to the credential reference: changing
    // either sends the project's content somewhere or with something it did not
    // agree to.
    // For a remote machine the pinned host key is part of who receives the
    // content: a new key is a new machine.
    let origin_changed = next.origin != old.origin
        || next.credential_ref != old.credential_ref
        || next.host_key != old.host_key;
    Ok((next, origin_changed))
}

/// A stored consent: who allowed which origin for a project, and when.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ConsentRecord {
    /// Instance.
    pub provider_id: String,
    /// Origin consented to.
    pub origin: String,
    /// Who consented (a human login).
    pub consented_by: String,
    /// RFC 3339 time.
    pub consented_at: String,
    /// Credential reference the consent was given for. A consent recorded
    /// before this field existed has none and holds only for an instance with
    /// no credential.
    #[serde(default)]
    pub credential_ref: Option<String>,
    /// Fingerprint of the pinned host key the consent was given for (a remote
    /// machine only): the same host name with another key is another machine.
    #[serde(default)]
    pub host_key_fingerprint: Option<String>,
}

/// Whether a consent still holds for the instance as it is now: same origin
/// AND same credential reference (A28), and for a remote machine the same
/// pinned host key.
pub fn consent_holds(consent: &ConsentRecord, instance: &InstanceRecord) -> bool {
    consent.provider_id == instance.id
        && consent.origin == instance.origin
        && consent.credential_ref.as_deref().unwrap_or("none") == instance.credential_ref
        && consent.host_key_fingerprint == instance.host_key_fingerprint()
}

/// A consent row as the API answers it: `valid` is false when the instance's
/// origin is no longer the one consented to.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ConsentView {
    /// Instance.
    pub provider_id: String,
    /// Origin consented to.
    pub origin: String,
    /// Who.
    pub consented_by: String,
    /// When.
    pub consented_at: String,
    /// Credential reference consented to (`none` when none was recorded).
    pub credential_ref: String,
    /// The consent still holds for the instance's current origin and credential.
    pub valid: bool,
}

/// Evaluates a consent against the instance as it is now (A28); `None` = the
/// instance no longer exists.
pub fn consent_view(record: &ConsentRecord, instance: Option<&InstanceRecord>) -> ConsentView {
    ConsentView {
        provider_id: record.provider_id.clone(),
        origin: record.origin.clone(),
        consented_by: record.consented_by.clone(),
        consented_at: record.consented_at.clone(),
        credential_ref: record
            .credential_ref
            .clone()
            .unwrap_or_else(|| "none".to_string()),
        valid: instance.is_some_and(|i| consent_holds(record, i)),
    }
}

/// Target of a role.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RoleTarget {
    /// Instance.
    pub provider: String,
    /// Model, when pinned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Alias, when the role goes through one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
}

/// `pilot` and `executor`; an absent role inherits (or means "single provider").
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RoleAssignments {
    /// Sessions opened by a human.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pilot: Option<RoleTarget>,
    /// Runner, delegation, protocols, one-shot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor: Option<RoleTarget>,
}

/// Refuses a role pointing at an instance that does not exist.
pub fn validate_roles(
    roles: &RoleAssignments,
    exists: &dyn Fn(&str) -> bool,
) -> Result<(), SettingsError> {
    for target in [&roles.pilot, &roles.executor].into_iter().flatten() {
        if !exists(&target.provider) {
            return Err(SettingsError::UnknownInstance(target.provider.clone()));
        }
        if target.model.is_some() && target.alias.is_some() {
            return Err(invalid("a role names a model or an alias, not both"));
        }
    }
    Ok(())
}

/// Alias to instance and model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelAlias {
    /// Alias name.
    pub alias: String,
    /// Instance.
    pub provider: String,
    /// Model.
    pub model: String,
}

/// Validates an alias table: unique, well-formed names, existing instances.
pub fn validate_aliases(
    aliases: &[ModelAlias],
    exists: &dyn Fn(&str) -> bool,
) -> Result<(), SettingsError> {
    let mut seen = std::collections::BTreeSet::new();
    for a in aliases {
        if !valid_id(&a.alias) {
            return Err(invalid("alias: lowercase letters, digits and dashes only"));
        }
        if !seen.insert(a.alias.as_str()) {
            return Err(invalid("alias: duplicated name"));
        }
        if a.model.trim().is_empty() {
            return Err(invalid("alias: a model is required"));
        }
        if !exists(&a.provider) {
            return Err(SettingsError::UnknownInstance(a.provider.clone()));
        }
    }
    Ok(())
}

/// Rule roles of the model policy (A19).
pub const POLICY_RULE_ROLES: [&str; 7] = [
    "chat",
    "runner.simple",
    "runner.complex",
    "runner.creative",
    "runner.retry",
    "utility.feature_graph",
    "utility.compaction",
];

/// Caps of the model policy.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PolicyCaps {
    /// USD per task.
    #[serde(default)]
    pub per_task_usd: Option<f64>,
    /// USD per run.
    #[serde(default)]
    pub per_run_usd: Option<f64>,
    /// Tokens per task.
    #[serde(default)]
    pub per_task_tokens: Option<u64>,
    /// Tokens per run.
    #[serde(default)]
    pub per_run_tokens: Option<u64>,
}

/// The model policy. Shipped as `off`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelPolicy {
    /// `off`, `shadow` or `enforce`.
    pub mode: String,
    /// Rule role to alias.
    #[serde(default)]
    pub rules: BTreeMap<String, String>,
    /// Aliases tried in order.
    #[serde(default)]
    pub fallback: Vec<String>,
    /// Caps.
    #[serde(default)]
    pub caps: PolicyCaps,
}

impl Default for ModelPolicy {
    fn default() -> Self {
        Self {
            mode: "off".into(),
            rules: BTreeMap::new(),
            fallback: Vec::new(),
            caps: PolicyCaps::default(),
        }
    }
}

/// Validates a policy against the alias table.
pub fn validate_policy(
    policy: &ModelPolicy,
    alias_known: &dyn Fn(&str) -> bool,
) -> Result<(), SettingsError> {
    if !matches!(policy.mode.as_str(), "off" | "shadow" | "enforce") {
        return Err(invalid("mode: off, shadow or enforce"));
    }
    for (role, alias) in &policy.rules {
        if !POLICY_RULE_ROLES.contains(&role.as_str()) {
            return Err(invalid("rules: unknown rule role"));
        }
        if !alias_known(alias) {
            return Err(invalid("rules: an alias is not defined"));
        }
    }
    for alias in &policy.fallback {
        if !alias_known(alias) {
            return Err(invalid("fallback: an alias is not defined"));
        }
    }
    let c = &policy.caps;
    if [c.per_task_usd, c.per_run_usd]
        .into_iter()
        .flatten()
        .any(|v| !v.is_finite() || v < 0.0)
    {
        return Err(invalid("caps: amounts must be positive numbers"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CTX5: an `anthropic` instance runs the native harness on the Messages API. Its base
    /// URL defaults to Anthropic's own API; its key is a VAULT reference and nothing else;
    /// it takes no dialect preset — at creation and at every patch.
    #[test]
    fn an_anthropic_instance_defaults_to_the_api_and_takes_its_key_from_the_vault_only() {
        let policy = EndpointPolicy::default();
        let anthropic = InstanceDraft {
            id: Some("claude-api".into()),
            kind: Some(KIND_ANTHROPIC.into()),
            label: Some("Claude API".into()),
            default_model: Some("claude-opus-5-5".into()),
            credential_ref: Some("vault:anthropic".into()),
            ..Default::default()
        };
        let r = record_from_draft(&anthropic, &policy, &[]).expect("a valid anthropic instance");
        assert_eq!(
            (r.kind.as_str(), r.base_url.as_str(), r.origin.as_str()),
            (KIND_ANTHROPIC, ANTHROPIC_API_BASE, ANTHROPIC_API_BASE)
        );
        assert!(
            !is_process_kind(&r.kind),
            "an endpoint kind: its consent follows its origin"
        );
        let gateway = InstanceDraft {
            base_url: Some("https://gateway.example.com/v1".into()),
            ..anthropic.clone()
        };
        assert_eq!(
            record_from_draft(&gateway, &policy, &[]).unwrap().origin,
            "https://gateway.example.com"
        );
        let env = ["ANTHROPIC_API_KEY".to_string()];
        for bad in ["none", "env:ANTHROPIC_API_KEY", "sk-ant-pasted-value"] {
            let draft = InstanceDraft {
                credential_ref: Some(bad.into()),
                ..anthropic.clone()
            };
            let err = record_from_draft(&draft, &policy, &env).expect_err(bad);
            assert!(!format!("{err:?}{err}").contains("sk-ant"), "{bad}");
        }
        let preset = InstanceDraft {
            preset: Some("deepseek".into()),
            ..anthropic.clone()
        };
        assert!(record_from_draft(&preset, &policy, &[]).is_err());
        // A patch keeps the rule: the key stays in the vault, no preset appears.
        let to_env = InstancePatch {
            credential_ref: Some("env:ANTHROPIC_API_KEY".into()),
            ..Default::default()
        };
        assert!(apply_patch(&r, &to_env, &policy, &env).is_err());
        let to_preset = InstancePatch {
            preset: Some("deepseek".into()),
            ..Default::default()
        };
        assert!(apply_patch(&r, &to_preset, &policy, &[]).is_err());
        let other_vault = InstancePatch {
            credential_ref: Some("vault:anthropic-2".into()),
            ..Default::default()
        };
        let (next, changed) = apply_patch(&r, &other_vault, &policy, &[]).unwrap();
        assert!(changed, "another key is another consent");
        assert_eq!(next.credential_ref, "vault:anthropic-2");
    }

    fn draft() -> InstanceDraft {
        InstanceDraft {
            id: Some("deepseek".into()),
            kind: Some("openai_compatible".into()),
            preset: Some("deepseek".into()),
            label: Some("DeepSeek".into()),
            base_url: Some("https://api.deepseek.com/v1".into()),
            default_model: Some("deepseek-chat".into()),
            cost_source: Some("priced".into()),
            credential_ref: Some("vault:deepseek".into()),
            ..Default::default()
        }
    }

    #[test]
    fn a_valid_draft_becomes_a_record_with_its_origin() {
        let r = record_from_draft(&draft(), &EndpointPolicy::default(), &[]).unwrap();
        assert_eq!(r.origin, "https://api.deepseek.com");
        assert_eq!(r.credential_ref, "vault:deepseek");
    }

    #[test]
    fn a_body_naming_a_secret_field_is_refused() {
        let body = serde_json::json!({"id": "x", "base_url": "https://a.example.com", "api_key": "sk-123"});
        assert!(serde_json::from_value::<InstanceDraft>(body).is_err());
        let body = serde_json::json!({"credential_ref": "sk-live-abcdef"});
        let patch: InstancePatch = serde_json::from_value(body).unwrap();
        let old = record_from_draft(&draft(), &EndpointPolicy::default(), &[]).unwrap();
        let err = apply_patch(&old, &patch, &EndpointPolicy::default(), &[]).unwrap_err();
        assert!(
            !err.to_string().contains("sk-live"),
            "never echoes the value"
        );
    }

    #[test]
    fn credential_refs_are_references_only() {
        let allow = vec!["DEEPSEEK_API_KEY".to_string(), "NEO4J_PASSWORD".to_string()];
        assert_eq!(
            parse_credential_ref("none", &[]).unwrap(),
            CredentialSource::None
        );
        assert_eq!(
            parse_credential_ref("vault:deepseek", &[]).unwrap(),
            CredentialSource::Vault("deepseek".into())
        );
        assert_eq!(
            parse_credential_ref("env:DEEPSEEK_API_KEY", &allow).unwrap(),
            CredentialSource::Env("DEEPSEEK_API_KEY".into())
        );
        assert!(parse_credential_ref("sk-abcdef", &allow).is_err());
        assert!(
            parse_credential_ref("env:NEO4J_PASSWORD", &allow).is_err(),
            "a server secret stays refused even when someone declared it"
        );
        assert!(parse_credential_ref("vault:", &allow).is_err());
    }

    #[test]
    fn env_credentials_are_refused_unless_declared() {
        // Deny by default: an undeclared variable is refused whatever its name.
        for var in [
            "env:HOME",
            "env:AWS_SECRET_ACCESS_KEY",
            "env:ANYTHING_ELSE",
            "env:DEEPSEEK_API_KEY",
        ] {
            assert!(parse_credential_ref(var, &[]).is_err(), "{var}");
        }
        let declared = parse_allowlist(" DEEPSEEK_API_KEY , ,OTHER_KEY ");
        assert_eq!(declared, vec!["DEEPSEEK_API_KEY", "OTHER_KEY"]);
        assert!(parse_credential_ref("env:DEEPSEEK_API_KEY", &declared).is_ok());
        assert!(parse_credential_ref("env:HOME", &declared).is_err());
        // The message never echoes a value, only says what to do.
        let msg = parse_credential_ref("env:HOME", &declared)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("CHAT_PROVIDER_ENV_CREDENTIALS"));
    }

    #[test]
    fn the_builtin_id_and_bad_urls_are_refused() {
        let mut d = draft();
        d.id = Some("claude-code".into());
        assert_eq!(
            record_from_draft(&d, &EndpointPolicy::default(), &[]).unwrap_err(),
            SettingsError::Builtin
        );
        let mut d = draft();
        d.base_url = Some("http://example.com/v1".into());
        assert!(matches!(
            record_from_draft(&d, &EndpointPolicy::default(), &[]).unwrap_err(),
            SettingsError::Endpoint(_)
        ));
        let mut d = draft();
        d.base_url = Some("https://user:pw@api.example.com".into());
        assert!(record_from_draft(&d, &EndpointPolicy::default(), &[]).is_err());
        let mut d = draft();
        d.kind = Some("codex".into());
        assert!(record_from_draft(&d, &EndpointPolicy::default(), &[]).is_err());
    }

    #[test]
    fn changing_the_url_changes_the_origin_and_invalidates_consent() {
        let old = record_from_draft(&draft(), &EndpointPolicy::default(), &[]).unwrap();
        let consent = ConsentRecord {
            provider_id: old.id.clone(),
            origin: old.origin.clone(),
            consented_by: "me@example.com".into(),
            consented_at: "2026-10-05T10:00:00Z".into(),
            credential_ref: Some(old.credential_ref.clone()),
            ..Default::default()
        };
        assert!(consent_view(&consent, Some(&old)).valid);
        let patch = InstancePatch {
            preset: None,
            label: None,
            base_url: Some("https://other.example.com/v1".into()),
            default_model: None,
            cost_source: None,
            credential_ref: None,
            ..Default::default()
        };
        let (next, changed) = apply_patch(&old, &patch, &EndpointPolicy::default(), &[]).unwrap();
        assert!(changed);
        assert!(!consent_view(&consent, Some(&next)).valid);
        assert!(
            !consent_view(&consent, None).valid,
            "a deleted instance holds no consent"
        );
        // Same origin, other path: the consent holds.
        let patch = InstancePatch {
            base_url: Some("https://api.deepseek.com/v2".into()),
            ..patch
        };
        let (_, changed) = apply_patch(&old, &patch, &EndpointPolicy::default(), &[]).unwrap();
        assert!(!changed);
    }

    #[test]
    fn roles_must_point_at_existing_instances_and_not_both_model_and_alias() {
        let exists = |id: &str| id == "claude-code" || id == "deepseek";
        let ok = RoleAssignments {
            pilot: Some(RoleTarget {
                provider: "claude-code".into(),
                model: None,
                alias: Some("default".into()),
            }),
            executor: None,
        };
        validate_roles(&ok, &exists).unwrap();
        let bad = RoleAssignments {
            pilot: Some(RoleTarget {
                provider: "ghost".into(),
                model: None,
                alias: None,
            }),
            executor: None,
        };
        assert_eq!(
            validate_roles(&bad, &exists).unwrap_err(),
            SettingsError::UnknownInstance("ghost".into())
        );
        let both = RoleAssignments {
            pilot: Some(RoleTarget {
                provider: "deepseek".into(),
                model: Some("m".into()),
                alias: Some("fast".into()),
            }),
            executor: None,
        };
        assert!(validate_roles(&both, &exists).is_err());
        assert_eq!(
            RoleAssignments::default(),
            serde_json::from_str("{}").unwrap()
        );
    }

    #[test]
    fn aliases_are_unique_and_point_at_instances() {
        let exists = |id: &str| id == "deepseek";
        let a = |alias: &str| ModelAlias {
            alias: alias.into(),
            provider: "deepseek".into(),
            model: "m".into(),
        };
        validate_aliases(&[a("fast"), a("deep")], &exists).unwrap();
        assert!(validate_aliases(&[a("fast"), a("fast")], &exists).is_err());
        assert!(validate_aliases(&[a("Fast")], &exists).is_err());
        let ghost = ModelAlias {
            alias: "x".into(),
            provider: "ghost".into(),
            model: "m".into(),
        };
        assert!(validate_aliases(&[ghost], &exists).is_err());
    }

    #[test]
    fn the_policy_ships_off_and_is_validated_against_the_aliases() {
        let p = ModelPolicy::default();
        assert_eq!(p.mode, "off");
        let known = |a: &str| a == "fast" || a == "deep";
        validate_policy(&p, &known).unwrap();
        let mut q = ModelPolicy {
            mode: "shadow".into(),
            ..Default::default()
        };
        q.rules.insert("runner.simple".into(), "fast".into());
        q.fallback = vec!["deep".into()];
        validate_policy(&q, &known).unwrap();
        q.rules.insert("runner.weird".into(), "fast".into());
        assert!(validate_policy(&q, &known).is_err());
        let bad_mode = ModelPolicy {
            mode: "auto".into(),
            ..Default::default()
        };
        assert!(validate_policy(&bad_mode, &known).is_err());
        let neg = ModelPolicy {
            caps: PolicyCaps {
                per_run_usd: Some(-1.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(validate_policy(&neg, &known).is_err());
        let unknown_alias = ModelPolicy {
            fallback: vec!["ghost".into()],
            ..Default::default()
        };
        assert!(validate_policy(&unknown_alias, &known).is_err());
    }

    #[test]
    fn changing_the_credential_reference_invalidates_the_consent_too() {
        let old = record_from_draft(&draft(), &EndpointPolicy::default(), &[]).unwrap();
        let consent = ConsentRecord {
            provider_id: old.id.clone(),
            origin: old.origin.clone(),
            consented_by: "me".into(),
            consented_at: "t".into(),
            credential_ref: Some(old.credential_ref.clone()),
            host_key_fingerprint: old.host_key_fingerprint(),
        };
        assert!(consent_holds(&consent, &old));
        let patch = InstancePatch {
            preset: None,
            label: None,
            base_url: None,
            default_model: None,
            cost_source: None,
            credential_ref: Some("vault:another-key".into()),
            ..Default::default()
        };
        let (next, changed) = apply_patch(&old, &patch, &EndpointPolicy::default(), &[]).unwrap();
        assert!(changed, "the API reports the consent as invalidated");
        assert!(
            !consent_holds(&consent, &next),
            "same origin, other key: no longer consented"
        );
        // A consent recorded before the field existed holds only without credential.
        let legacy = ConsentRecord {
            credential_ref: None,
            ..consent
        };
        assert!(!consent_holds(&legacy, &old));
        let mut keyless = old.clone();
        keyless.credential_ref = "none".into();
        assert!(consent_holds(&legacy, &keyless));
    }

    #[test]
    fn codex_and_acp_instances_have_a_process_identity_and_never_a_command_line() {
        let codex = InstanceDraft {
            id: Some("codex".into()),
            kind: Some("codex".into()),
            preset: None,
            label: None,
            base_url: None,
            default_model: None,
            cost_source: None,
            credential_ref: None,
            ..Default::default()
        };
        let r = record_from_draft(&codex, &EndpointPolicy::default(), &[]).unwrap();
        assert_eq!(
            (r.kind.as_str(), r.origin.as_str(), r.base_url.as_str()),
            ("codex", "process:codex", "")
        );
        // A URL makes no sense for it.
        let mut with_url = codex.clone();
        with_url.base_url = Some("https://example.com".into());
        assert!(record_from_draft(&with_url, &EndpointPolicy::default(), &[]).is_err());
        // ACP: only a command DECLARED on the server can be named; the body has no command field.
        let declared = parse_acp_commands(
            r#"{"opencode": ["opencode", "acp"], "bad name": ["x"], "empty": []}"#,
        );
        assert_eq!(declared.keys().collect::<Vec<_>>(), vec!["opencode"]);
        assert!(parse_acp_commands("not json").is_empty());
        let argv = |parts: &[&str]| parts.iter().map(|p| p.to_string()).collect::<Vec<_>>();
        // The declaration wins over the name, both ways.
        let declared = parse_acp_commands(
            r#"{"old": {"argv": ["my-bridge", "acp"], "per_session_mcp": false},
                "named": ["openclaw", "acp"],
                "forced": {"argv": ["openclaw", "acp"], "per_session_mcp": true},
                "plain": {"argv": ["opencode", "acp"]},
                "typo": {"argv": ["x"], "per_session_mpc": false}}"#,
        );
        assert_eq!(
            declared.keys().collect::<Vec<_>>(),
            vec!["forced", "named", "old", "plain"],
            "an unknown field declares nothing"
        );
        assert!(!declared["old"].carries_mcp());
        assert!(!declared["named"].carries_mcp());
        assert!(declared["forced"].carries_mcp());
        assert!(declared["plain"].carries_mcp());
        assert_eq!(declared["old"].argv, argv(&["my-bridge", "acp"]));
        assert!(serde_json::from_value::<InstanceDraft>(
            serde_json::json!({"id": "a", "kind": "acp", "command": ["sh", "-c", "evil"]})
        )
        .is_err());
        let acp = InstanceDraft {
            id: Some("oc".into()),
            kind: Some("acp".into()),
            preset: Some("opencode".into()),
            ..codex.clone()
        };
        // Nothing is declared in the test environment: refused.
        assert!(record_from_draft(&acp, &EndpointPolicy::default(), &[]).is_err());
        assert_eq!(
            process_origin("acp", Some("opencode")),
            "process:acp:opencode"
        );
        assert!(
            is_process_kind("acp")
                && is_process_kind("codex")
                && !is_process_kind("openai_compatible")
        );
    }

    /// OpenClaw's ACP bridge takes no MCP server per session: found by name at the
    /// PROGRAM positions only (`argv[0]`, what a known launcher runs), never in an
    /// option's value, a path given as an argument, or a URL.
    #[test]
    fn the_openclaw_name_is_read_at_program_positions_only() {
        let argv = |parts: &[&str]| parts.iter().map(|p| p.to_string()).collect::<Vec<_>>();
        for carries in [
            &["opencode", "acp"][..],
            &["node", "/opt/oc/openclaw.config.mjs", "acp"],
            &["agent", "--url", "wss://openclaw.example/ws"],
            // A non-OpenClaw agent whose arguments merely name OpenClaw keeps its tools.
            &["my-agent", "--config", "~/.openclaw/openclaw.json"],
            &["my-agent", "--token-file", "/etc/po/openclaw.token"],
            &["my-agent", "--url", "wss://gw/openclaw"],
            &["my-agent", "--profile", "openclaw"],
            &["my-agent", "acp", "openclaw"],
            &["npx", "my-agent", "--profile", "openclaw"],
            &["env", "PROFILE=openclaw", "my-agent", "openclaw"],
            &["npx", "-p", "openclaw", "my-agent"],
            &["pnpm", "install", "openclaw"],
            &["wss://gw/openclaw"],
        ] {
            assert!(acp_command_carries_mcp(&argv(carries)), "{carries:?}");
        }
        for refuses in [
            &["openclaw", "acp", "--token-file", "/run/oc"][..],
            &["/usr/local/bin/openclaw", "acp"],
            &[r"C:\tools\OpenClaw.cmd", "acp"],
            &["npx", "openclaw", "acp"],
            &["npx", "--yes", "openclaw@2026.9.9", "acp"],
            &["npx", "-p", "openclaw", "openclaw", "acp"],
            &["pnpm", "dlx", "openclaw", "acp"],
            &["pnpm", "--silent", "exec", "openclaw", "acp"],
            &["bunx", "openclaw", "acp"],
            &["node", "/opt/oc/dist/openclaw.mjs", "acp"],
            &[
                "node",
                "--require",
                "/opt/trace.js",
                "/opt/oc/openclaw.mjs",
                "acp",
            ],
            &["env", "openclaw", "acp"],
            &[
                "/usr/bin/env",
                "-u",
                "HOME",
                "OC_DEBUG=1",
                "openclaw",
                "acp",
            ],
            &["env", "npx", "openclaw", "acp"],
        ] {
            assert!(!acp_command_carries_mcp(&argv(refuses)), "{refuses:?}");
        }
        assert!(acp_command_carries_mcp(&[]));
    }

    /// The launchers of round 2 are read again (review of #682): `npm exec|x`, `pnpx`,
    /// `yarn dlx`, `pnpm` with an option that takes a value before its subcommand, and
    /// a command line handed to the launcher (`npx -c`, `env -S`) run OpenClaw — while
    /// an option's value, a path or a URL that names it still does not.
    #[test]
    fn openclaw_run_by_any_launcher_is_detected_and_a_named_argument_still_is_not() {
        let argv = |parts: &[&str]| parts.iter().map(|p| p.to_string()).collect::<Vec<_>>();
        for refuses in [
            &["npm", "exec", "openclaw", "acp"][..],
            &["npm", "x", "openclaw", "acp"],
            &["npm", "exec", "--yes", "--", "openclaw", "acp"],
            // Review of #684, finding 4: `--registry <url>` takes a value.
            &[
                "npm",
                "exec",
                "--registry",
                "https://registry.example",
                "openclaw",
                "acp",
            ],
            &[
                "npx",
                "--registry",
                "https://registry.example",
                "openclaw",
                "acp",
            ],
            &[
                "npm",
                "--prefix",
                "/srv/oc",
                "exec",
                "--package",
                "openclaw@2026.9.9",
                "--",
                "openclaw",
                "acp",
            ],
            &["pnpx", "openclaw", "acp"],
            &["yarn", "dlx", "openclaw", "acp"],
            &[
                "yarn", "--cwd", "/srv/oc", "dlx", "-p", "openclaw", "openclaw", "acp",
            ],
            &["pnpm", "-C", "/srv/oc", "dlx", "openclaw", "acp"],
            &[
                "pnpm", "--dir", "/srv/oc", "--silent", "exec", "openclaw", "acp",
            ],
            &["pnpm", "--filter", "gw", "exec", "openclaw", "acp"],
            &["pnpm", "dlx", "-c", "openclaw acp"],
            &["pnpm", "-c", "exec", "openclaw acp --token-file /run/oc"],
            &["env", "-S", "openclaw acp"],
            &["env", "--split-string=openclaw acp"],
            &["env", "-Sopenclaw acp"],
            &["/usr/bin/env", "-i", "-S", "OC_DEBUG=1 openclaw acp"],
            &["env", "-S", "-u HOME", "openclaw", "acp"],
            &["npx", "-c", "OC_DEBUG=1 openclaw acp"],
            &["npx", "-c", "openclaw acp"],
            &["npx", "--call", "'openclaw' acp"],
            &["npx", "--call=openclaw acp"],
            &["npm", "exec", "-c", "openclaw acp"],
            &["env", "-S", "npx -c 'openclaw acp'"],
        ] {
            assert!(!acp_command_carries_mcp(&argv(refuses)), "{refuses:?}");
        }
        for carries in [
            // The false positives of round 2 stay what they are.
            &["my-agent", "--config", "~/.openclaw/openclaw.json"][..],
            &["my-agent", "--token-file", "/etc/po/openclaw.token"],
            &["my-agent", "--url", "wss://gw/openclaw"],
            &["my-agent", "--profile", "openclaw"],
            // A launcher command that runs nothing named after it.
            &["npm", "install", "openclaw"],
            &["npm", "--prefix", "/srv/openclaw", "start"],
            &["npx", "--registry", "https://openclaw.example", "my-agent"],
            &["yarn", "add", "openclaw"],
            &["pnpm", "-C", "/srv/openclaw", "dlx", "my-agent"],
            &[
                "npm",
                "exec",
                "--package",
                "openclaw",
                "--",
                "my-agent",
                "--profile",
                "openclaw",
            ],
            // A command line whose program is another one.
            &["npx", "-c", "my-agent --profile openclaw"],
            &["env", "-S", "my-agent --config ~/.openclaw/openclaw.json"],
            &["env", "-S", "\"wss://gw/openclaw\" acp"],
            &["pnpm", "-c", "exec", "my-agent openclaw"],
            // `node -c` checks a script: it does not run a command line.
            &["node", "-c", "/opt/oc/my-agent.mjs"],
        ] {
            assert!(acp_command_carries_mcp(&argv(carries)), "{carries:?}");
        }
        // Nested command lines stop at a depth, without panic or loop.
        let mut line = "openclaw acp".to_string();
        for _ in 0..10 {
            line = format!("npx -c \"{}\"", line.replace('"', "'"));
        }
        let _ = acp_command_carries_mcp(&argv(&["env", "-S", &line]));
        assert_eq!(
            command_words(r#"a  "b c" 'd "e"' f"g"h"#),
            vec!["a", "b c", "d \"e\"", "fgh"]
        );
    }

    // ── claude_code_remote ────────────────────────────────────────────────

    /// A real ed25519 host key and its `ssh-keygen -lf` fingerprint.
    const KEY: &str =
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEXTCY3J636nEyMNqNrVj6HXnIXUnXL8sk4c5J6tek3N";
    const KEY_FP: &str = "SHA256:lP63ZdLutNnRU0/59cDaFw2mPoJzdasi0I3zFrtS3Ak";
    const OTHER_KEY: &str =
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl";

    fn remote_draft() -> InstanceDraft {
        InstanceDraft {
            kind: Some(KIND_CLAUDE_CODE_REMOTE.into()),
            label: Some("Build box 1".into()),
            host: Some("build-1.example.net".into()),
            ssh_user: Some("deploy".into()),
            ssh_port: Some(2222),
            host_key: Some(KEY.into()),
            remote_cwd: Some("/srv/work".into()),
            credential_ref: Some("vault:ssh-build-1".into()),
            default_model: Some("sonnet".into()),
            ..Default::default()
        }
    }

    fn remote_record() -> InstanceRecord {
        record_from_draft(&remote_draft(), &EndpointPolicy::default(), &[]).unwrap()
    }

    fn refused(d: InstanceDraft) -> String {
        match record_from_draft(&d, &EndpointPolicy::default(), &[]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected a refusal"),
        }
    }

    #[test]
    fn a_remote_draft_becomes_a_record_with_a_forced_id_and_an_ssh_origin() {
        let r = remote_record();
        assert_eq!(r.id, "claude-code@build-box-1");
        assert_eq!(r.kind, "claude_code_remote");
        assert_eq!(r.origin, "ssh:deploy@build-1.example.net:2222");
        assert!(r.base_url.is_empty());
        assert!(is_process_kind(&r.kind));
        assert_eq!(r.host_key_fingerprint().as_deref(), Some(KEY_FP));
        assert!(!r.allow_trust);
        // The id a caller sent is not honoured (only the label makes it).
        let mut d = remote_draft();
        d.id = Some("anything".into());
        assert_eq!(
            record_from_draft(&d, &EndpointPolicy::default(), &[])
                .unwrap()
                .id,
            "claude-code@build-box-1"
        );
    }

    #[test]
    fn the_origin_defaults_to_port_22_and_no_user() {
        let mut d = remote_draft();
        d.ssh_user = None;
        d.ssh_port = None;
        let r = record_from_draft(&d, &EndpointPolicy::default(), &[]).unwrap();
        assert_eq!(r.origin, "ssh:build-1.example.net:22");
    }

    #[test]
    fn the_fingerprint_is_the_openssh_one() {
        assert_eq!(host_key_fingerprint(KEY).as_deref(), Some(KEY_FP));
        assert_eq!(
            host_key_fingerprint("ssh-ed25519 not*base64*at-all-xx"),
            None
        );
        assert_eq!(host_key_fingerprint("garbage"), None);
    }

    #[test]
    fn the_reserved_id_stays_reserved_for_every_kind() {
        let mut d = remote_draft();
        d.id = Some("claude-code".into());
        assert!(matches!(
            record_from_draft(&d, &EndpointPolicy::default(), &[]),
            Err(SettingsError::Builtin)
        ));
        let mut local = draft();
        local.id = Some("claude-code".into());
        assert!(matches!(
            record_from_draft(&local, &EndpointPolicy::default(), &[]),
            Err(SettingsError::Builtin)
        ));
        // A label that makes no slug cannot make an id.
        let mut d = remote_draft();
        d.label = Some("!!!".into());
        assert!(refused(d).contains("label"));
    }

    #[test]
    fn every_bad_remote_draft_is_refused() {
        for bad_host in [
            "",
            "-oProxyCommand=evil",
            "host name",
            "host;rm",
            "host\nx",
            "a$(b)",
            "host'x",
        ] {
            let mut d = remote_draft();
            d.host = Some(bad_host.into());
            assert!(refused(d).contains("host"), "{bad_host:?}");
        }
        let mut d = remote_draft();
        d.host = None;
        assert!(refused(d).contains("host"));
        let mut d = remote_draft();
        d.ssh_user = Some("-lroot".into());
        assert!(refused(d).contains("ssh_user"));
        let mut d = remote_draft();
        d.ssh_port = Some(0);
        assert!(refused(d).contains("ssh_port"));
        for bad_key in [
            "",
            "ssh-ed25519",
            "ssh-ed25519 AAAA",
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEXTCY3J636nEyMNqNrVj6HXnIXUnXL8sk4c5J6tek3N extra",
            "rsa AAAAC3NzaC1lZDI1NTE5AAAAIEXTCY3J636nEyMNqNrVj6HXnIXUnXL8sk4c5J6tek3N",
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEXTCY3J636nEyMNq$(id)",
        ] {
            let mut d = remote_draft();
            d.host_key = Some(bad_key.into());
            assert!(refused(d).contains("host_key"), "{bad_key:?}");
        }
        let mut d = remote_draft();
        d.host_key = None;
        assert!(refused(d).contains("host_key"));
        let mut d = remote_draft();
        d.remote_cwd = None;
        assert!(refused(d).contains("remote_cwd"));
        let mut d = remote_draft();
        d.base_url = Some("https://example.com".into());
        assert!(refused(d).contains("base_url"));
    }

    #[test]
    fn a_remote_credential_is_a_vault_reference_and_nothing_else() {
        for bad in [
            None,
            Some("none"),
            Some(""),
            Some("env:HOME"),
            Some("-----BEGIN OPENSSH PRIVATE KEY-----"),
            Some("vault:bad name"),
        ] {
            let mut d = remote_draft();
            d.credential_ref = bad.map(str::to_string);
            assert!(refused(d).contains("credential_ref"), "{bad:?}");
        }
        // Even an env var the operator declared is not accepted for this kind.
        let mut d = remote_draft();
        d.credential_ref = Some("env:MY_SSH".into());
        assert!(
            record_from_draft(&d, &EndpointPolicy::default(), &["MY_SSH".to_string()]).is_err()
        );
    }

    #[test]
    fn remote_fields_on_another_kind_are_refused() {
        let mut d = draft();
        d.host = Some("example.net".into());
        assert!(refused(d).contains("claude_code_remote"));
        let patch = InstancePatch {
            host: Some("example.net".into()),
            ..Default::default()
        };
        let old = record_from_draft(&draft(), &EndpointPolicy::default(), &[]).unwrap();
        assert!(apply_patch(&old, &patch, &EndpointPolicy::default(), &[]).is_err());
    }

    #[test]
    fn a_change_of_machine_revokes_consent() {
        let old = remote_record();
        let consent = ConsentRecord {
            provider_id: old.id.clone(),
            origin: old.origin.clone(),
            consented_by: "me".into(),
            consented_at: "t".into(),
            credential_ref: Some(old.credential_ref.clone()),
            host_key_fingerprint: old.host_key_fingerprint(),
        };
        assert!(consent_holds(&consent, &old));
        let patches = [
            InstancePatch {
                host: Some("other.example.net".into()),
                ..Default::default()
            },
            InstancePatch {
                ssh_port: Some(22),
                ..Default::default()
            },
            InstancePatch {
                ssh_user: Some("root".into()),
                ..Default::default()
            },
            InstancePatch {
                host_key: Some(OTHER_KEY.into()),
                ..Default::default()
            },
        ];
        for patch in patches {
            let (next, changed) =
                apply_patch(&old, &patch, &EndpointPolicy::default(), &[]).unwrap();
            assert!(changed, "{patch:?}");
            assert!(!consent_holds(&consent, &next), "{patch:?}");
        }
        // A working-directory or label change is not a new destination.
        let patch = InstancePatch {
            remote_cwd: Some("/other".into()),
            label: Some("renamed".into()),
            ..Default::default()
        };
        let (next, changed) = apply_patch(&old, &patch, &EndpointPolicy::default(), &[]).unwrap();
        assert!(!changed && consent_holds(&consent, &next));
        // The patch keeps the same rules as the draft.
        for bad in [
            InstancePatch {
                host: Some("-oProxyCommand=x".into()),
                ..Default::default()
            },
            InstancePatch {
                ssh_port: Some(0),
                ..Default::default()
            },
            InstancePatch {
                host_key: Some("ssh-ed25519 AAAA".into()),
                ..Default::default()
            },
            InstancePatch {
                credential_ref: Some("env:HOME".into()),
                ..Default::default()
            },
            InstancePatch {
                credential_ref: Some("none".into()),
                ..Default::default()
            },
            InstancePatch {
                base_url: Some("https://x.example.com".into()),
                ..Default::default()
            },
        ] {
            assert!(
                apply_patch(&old, &bad, &EndpointPolicy::default(), &[]).is_err(),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn an_old_record_without_the_remote_fields_still_reads() {
        let raw = r#"{"id":"x","kind":"codex","label":"x","base_url":"","origin":"process:codex","cost_source":"unknown","credential_ref":"none"}"#;
        let r: InstanceRecord = serde_json::from_str(raw).unwrap();
        assert!(r.host.is_none() && !r.allow_trust);
        assert_eq!(r.host_key_fingerprint(), None);
    }

    // ========================================================================
    // RL1 — rate limits
    // ========================================================================

    fn fixture(name: &str) -> serde_json::Value {
        let path = format!(
            "{}/tests/fixtures/rate_limit/{name}",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    fn golden_limits() -> Vec<RateLimit> {
        serde_json::from_value(fixture("limits.json")["limits"].clone()).expect("the golden limits")
    }

    fn instance_scope() -> RateLimitScope {
        RateLimitScope::Instance(InstanceScope::Instance)
    }

    fn limit(
        kind: RateLimitKind,
        window: RateLimitWindow,
        calendar: Option<&str>,
        max: f64,
        scope: RateLimitScope,
    ) -> RateLimit {
        RateLimit {
            kind,
            window,
            calendar: calendar.map(str::to_string),
            max,
            scope,
        }
    }

    fn unit(u: RateLimitUnit) -> RateLimitWindow {
        RateLimitWindow::Unit(u)
    }

    fn with_limits(limits: Vec<RateLimit>) -> InstanceRecord {
        let mut d = draft();
        d.rate_limit = limits;
        record_from_draft(&d, &EndpointPolicy::default(), &[]).unwrap()
    }

    /// An instance with NO limit is not "an instance with a default limit": it is
    /// exactly the instance of before this field existed.
    #[test]
    fn an_instance_without_any_limit_is_strictly_unchanged() {
        let record = record_from_draft(&draft(), &EndpointPolicy::default(), &[]).unwrap();
        assert!(record.rate_limit.is_empty());

        // A patch that never mentions the field leaves the list alone.
        let (next, changed) = apply_patch(
            &record,
            &InstancePatch::default(),
            &EndpointPolicy::default(),
            &[],
        )
        .unwrap();
        assert!(next.rate_limit.is_empty());
        assert!(!changed);

        // A record stored BEFORE the field existed reads back with an empty list:
        // no migration, and nothing to limit.
        let old = r#"{"id":"x","kind":"openai_compatible","label":"x","base_url":"https://a.example.com","origin":"https://a.example.com","cost_source":"unknown","credential_ref":"none"}"#;
        let read: InstanceRecord = serde_json::from_str(old).unwrap();
        assert!(read.rate_limit.is_empty(), "no migration, no limit");

        let report = rate_limit_report(&read);
        assert!(report.limits.is_empty());
        assert_eq!(
            report.status,
            RateLimitStatus::Ok,
            "nothing configured = nothing that can be limited"
        );
    }

    /// The three fixtures are the contract with the frontend: the first field of
    /// each is the version, like the refs and attention fixtures.
    #[test]
    fn every_rate_limit_fixture_declares_the_contract_version_first() {
        for name in ["limits.json", "report.json", "event.json"] {
            let path = format!(
                "{}/tests/fixtures/rate_limit/{name}",
                env!("CARGO_MANIFEST_DIR")
            );
            let text = std::fs::read_to_string(&path).unwrap();
            let first = text.lines().nth(1).unwrap().trim();
            assert!(
                first.starts_with("\"_contract_version\": 1"),
                "{name}: {first}"
            );
        }
    }

    /// The fixture is valid, and serializing the parsed limits gives it back
    /// exactly: a renamed key or a dropped `calendar` makes this red.
    #[test]
    fn the_golden_limits_fixture_is_valid_and_round_trips() {
        let v = fixture("limits.json");
        let limits = golden_limits();
        validate_rate_limits(&limits, &[]).expect("the golden limits must be valid");
        assert_eq!(
            serde_json::to_value(&limits).unwrap(),
            v["limits"].clone(),
            "the wire shape is pinned by the fixture"
        );
        // The five cases the fixture exists for.
        assert_eq!(limits.len(), 5);
        assert!(limits
            .iter()
            .any(|l| l.window == RateLimitWindow::Rolling(RollingWindow { rolling: 60 })));
        assert!(limits
            .iter()
            .any(|l| l.scope.credential() == Some("vault:nvidia-nim")));
        assert!(limits
            .iter()
            .any(|l| l.calendar.as_deref() == Some("Europe/Paris")));
    }

    /// The status route's shape, byte for byte.
    #[test]
    fn the_report_shape_is_the_golden_one() {
        let record = with_limits(golden_limits());
        let report = serde_json::to_value(rate_limit_report(&record)).unwrap();
        assert_eq!(report, fixture("report.json")["report"].clone());
    }

    /// The event shape, byte for byte, and it parses back.
    #[test]
    fn the_event_shape_is_the_golden_one_and_parses_back() {
        let v = fixture("event.json");
        let event: ProviderRateLimitEvent =
            serde_json::from_value(v["event"].clone()).expect("the golden event");
        assert_eq!(event.event_type, RATE_LIMIT_EVENT);
        assert_eq!(event.status, RateLimitStatus::Limited);
        assert_eq!(event.reason.as_deref(), Some("quota"));
        assert_eq!(
            event.credential.as_deref(),
            Some("vault:deepseek"),
            "a reference, never a value"
        );
        assert_eq!(serde_json::to_value(&event).unwrap(), v["event"].clone());
    }

    /// Every rule the interface must mirror, refused with its own reason.
    #[test]
    fn an_invalid_limit_is_refused_with_a_precise_reason() {
        let cases: Vec<(&str, RateLimit)> = vec![
            (
                "zero",
                limit(
                    RateLimitKind::Requests,
                    unit(RateLimitUnit::Minute),
                    None,
                    0.0,
                    instance_scope(),
                ),
            ),
            (
                "negative",
                limit(
                    RateLimitKind::Requests,
                    unit(RateLimitUnit::Minute),
                    None,
                    -1.0,
                    instance_scope(),
                ),
            ),
            (
                "half a request",
                limit(
                    RateLimitKind::Requests,
                    unit(RateLimitUnit::Minute),
                    None,
                    1.5,
                    instance_scope(),
                ),
            ),
            (
                "half a token",
                limit(
                    RateLimitKind::Tokens,
                    unit(RateLimitUnit::Hour),
                    None,
                    0.5,
                    instance_scope(),
                ),
            ),
            (
                "a budget per minute",
                limit(
                    RateLimitKind::BudgetUsd,
                    unit(RateLimitUnit::Minute),
                    None,
                    10.0,
                    instance_scope(),
                ),
            ),
            (
                "a budget over a rolling window",
                limit(
                    RateLimitKind::BudgetUsd,
                    RateLimitWindow::Rolling(RollingWindow { rolling: 3600 }),
                    None,
                    10.0,
                    instance_scope(),
                ),
            ),
            (
                "a calendar window without a time zone",
                limit(
                    RateLimitKind::Requests,
                    unit(RateLimitUnit::Day),
                    None,
                    100.0,
                    instance_scope(),
                ),
            ),
            (
                "a time zone on a window that has none",
                limit(
                    RateLimitKind::Requests,
                    unit(RateLimitUnit::Minute),
                    Some("Europe/Paris"),
                    100.0,
                    instance_scope(),
                ),
            ),
            (
                "an unknown time zone",
                limit(
                    RateLimitKind::Requests,
                    unit(RateLimitUnit::Day),
                    Some("Mars/Olympus"),
                    100.0,
                    instance_scope(),
                ),
            ),
            (
                "a rolling window of zero seconds",
                limit(
                    RateLimitKind::Requests,
                    RateLimitWindow::Rolling(RollingWindow { rolling: 0 }),
                    None,
                    3.0,
                    instance_scope(),
                ),
            ),
            (
                "an empty credential scope",
                limit(
                    RateLimitKind::Requests,
                    unit(RateLimitUnit::Minute),
                    None,
                    3.0,
                    RateLimitScope::Credential(CredentialScope {
                        credential: String::new(),
                    }),
                ),
            ),
            (
                "a credential scope that names no credential",
                limit(
                    RateLimitKind::Requests,
                    unit(RateLimitUnit::Minute),
                    None,
                    3.0,
                    RateLimitScope::Credential(CredentialScope {
                        credential: "none".into(),
                    }),
                ),
            ),
            (
                "an empty model scope",
                limit(
                    RateLimitKind::Requests,
                    unit(RateLimitUnit::Minute),
                    None,
                    3.0,
                    RateLimitScope::Model(ModelScope { model: "  ".into() }),
                ),
            ),
        ];
        for (why, l) in &cases {
            let err = validate_rate_limits(std::slice::from_ref(l), &[])
                .unwrap_err()
                .to_string();
            assert!(!err.is_empty(), "{why} must be refused with a reason");
            assert!(
                err.contains("rate_limit") || err.contains("credential_ref"),
                "{why}: {err}"
            );
        }
        // Two precise messages, so the interface can say the same thing.
        let missing_tz = validate_rate_limits(
            &[limit(
                RateLimitKind::Requests,
                unit(RateLimitUnit::Day),
                None,
                100.0,
                instance_scope(),
            )],
            &[],
        )
        .unwrap_err()
        .to_string();
        assert!(missing_tz.contains("time zone"), "{missing_tz}");
        let bad_tz = validate_rate_limits(
            &[limit(
                RateLimitKind::Requests,
                unit(RateLimitUnit::Day),
                Some("Mars/Olympus"),
                100.0,
                instance_scope(),
            )],
            &[],
        )
        .unwrap_err()
        .to_string();
        assert!(bad_tz.contains("IANA"), "{bad_tz}");

        // And the valid neighbours are accepted: the rules draw a line, not a net.
        validate_rate_limits(
            &[
                limit(
                    RateLimitKind::Requests,
                    unit(RateLimitUnit::Minute),
                    None,
                    40.0,
                    instance_scope(),
                ),
                limit(
                    RateLimitKind::Tokens,
                    unit(RateLimitUnit::Day),
                    Some("UTC"),
                    1000.0,
                    instance_scope(),
                ),
                limit(
                    RateLimitKind::BudgetUsd,
                    unit(RateLimitUnit::Week),
                    Some("utc"),
                    5.5,
                    instance_scope(),
                ),
                limit(
                    RateLimitKind::Requests,
                    RateLimitWindow::Rolling(RollingWindow { rolling: 60 }),
                    None,
                    20.0,
                    RateLimitScope::Model(ModelScope {
                        model: "deepseek-chat".into(),
                    }),
                ),
            ],
            &[],
        )
        .expect("valid limits are accepted");
    }

    /// A limit never stores a secret: a credential scope goes through the very
    /// same parser as an instance's `credential_ref`.
    #[test]
    fn a_limit_never_stores_a_secret() {
        let bare_key = limit(
            RateLimitKind::Requests,
            unit(RateLimitUnit::Minute),
            None,
            5.0,
            RateLimitScope::Credential(CredentialScope {
                credential: "sk-live-abcdef".into(),
            }),
        );
        let err = validate_rate_limits(&[bare_key], &[])
            .unwrap_err()
            .to_string();
        assert!(
            !err.contains("sk-live"),
            "the reason never echoes what was sent: {err}"
        );

        // An `env:` reference must be declared, exactly like an instance's.
        let env = limit(
            RateLimitKind::Requests,
            unit(RateLimitUnit::Minute),
            None,
            5.0,
            RateLimitScope::Credential(CredentialScope {
                credential: "env:DEEPSEEK_API_KEY".into(),
            }),
        );
        assert!(
            validate_rate_limits(std::slice::from_ref(&env), &[]).is_err(),
            "an undeclared variable is refused (deny by default)"
        );
        validate_rate_limits(&[env], &["DEEPSEEK_API_KEY".to_string()])
            .expect("a declared variable is accepted");
    }

    /// Kind + window + scope is the identity of a limit.
    #[test]
    fn two_limits_with_the_same_kind_window_and_scope_are_refused() {
        let a = limit(
            RateLimitKind::Requests,
            unit(RateLimitUnit::Minute),
            None,
            40.0,
            instance_scope(),
        );
        let mut b = a.clone();
        b.max = 60.0;
        let err = validate_rate_limits(&[a.clone(), b], &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("rate_limit"), "{err}");

        // Another scope, another window, another kind: all fine.
        let other_scope = limit(
            RateLimitKind::Requests,
            unit(RateLimitUnit::Minute),
            None,
            40.0,
            RateLimitScope::Credential(CredentialScope {
                credential: "vault:x".into(),
            }),
        );
        validate_rate_limits(&[a.clone(), other_scope], &[]).expect("another scope");
        validate_rate_limits(
            &[
                a.clone(),
                limit(
                    RateLimitKind::Requests,
                    unit(RateLimitUnit::Hour),
                    None,
                    40.0,
                    instance_scope(),
                ),
            ],
            &[],
        )
        .expect("another window");
        let mut other_kind = a.clone();
        other_kind.kind = RateLimitKind::Tokens;
        validate_rate_limits(&[a, other_kind], &[]).expect("another kind");
    }

    #[test]
    fn too_many_limits_are_refused() {
        let one = limit(
            RateLimitKind::Requests,
            unit(RateLimitUnit::Minute),
            None,
            1.0,
            instance_scope(),
        );
        let many = vec![one; MAX_RATE_LIMITS + 1];
        assert!(validate_rate_limits(&many, &[]).is_err());
    }

    /// A limit does not move the content, nor the key: no consent is touched.
    #[test]
    fn a_patch_replaces_or_clears_the_limits_without_touching_a_consent() {
        let record = with_limits(vec![limit(
            RateLimitKind::Requests,
            unit(RateLimitUnit::Minute),
            None,
            40.0,
            instance_scope(),
        )]);
        let consent = ConsentRecord {
            provider_id: record.id.clone(),
            origin: record.origin.clone(),
            consented_by: "me@example.com".into(),
            consented_at: "2026-10-08T10:00:00Z".into(),
            credential_ref: Some(record.credential_ref.clone()),
            ..Default::default()
        };

        let patch = InstancePatch {
            rate_limit: Some(vec![limit(
                RateLimitKind::Requests,
                unit(RateLimitUnit::Hour),
                None,
                400.0,
                instance_scope(),
            )]),
            ..Default::default()
        };
        let (next, changed) =
            apply_patch(&record, &patch, &EndpointPolicy::default(), &[]).unwrap();
        assert_eq!(next.rate_limit.len(), 1);
        assert_eq!(next.rate_limit[0].window, unit(RateLimitUnit::Hour));
        assert!(!changed, "a limit moves neither the content nor the key");
        assert!(consent_view(&consent, Some(&next)).valid);

        // `Some([])` clears every limit.
        let patch = InstancePatch {
            rate_limit: Some(vec![]),
            ..Default::default()
        };
        let (next, _) = apply_patch(&next, &patch, &EndpointPolicy::default(), &[]).unwrap();
        assert!(next.rate_limit.is_empty());

        // An invalid list is refused and nothing is written.
        let patch = InstancePatch {
            rate_limit: Some(vec![limit(
                RateLimitKind::Requests,
                unit(RateLimitUnit::Minute),
                None,
                0.0,
                instance_scope(),
            )]),
            ..Default::default()
        };
        assert!(apply_patch(&record, &patch, &EndpointPolicy::default(), &[]).is_err());
    }

    /// Unknown fields are refused on the limit itself, like everywhere else in a
    /// settings body: a field nobody knows is not silently stored.
    #[test]
    fn a_limit_with_an_unknown_field_is_refused() {
        let body = serde_json::json!({"rate_limit": [
            {"kind": "requests", "window": "minute", "max": 10.0, "scope": "instance", "api_key": "sk-1"}
        ]});
        assert!(serde_json::from_value::<InstanceDraft>(body).is_err());
        let body = serde_json::json!({"kind": "requests", "window": "minute", "max": 10.0, "scope": "instance", "secret": "x"});
        assert!(serde_json::from_value::<RateLimit>(body).is_err());
        // A malformed window or scope is refused too, rather than defaulted.
        let body = serde_json::json!({"kind": "requests", "window": "fortnight", "max": 10.0, "scope": "instance"});
        assert!(serde_json::from_value::<RateLimit>(body).is_err());
        let body = serde_json::json!({"kind": "requests", "window": "minute", "max": 10.0, "scope": {"credential": "vault:x", "extra": 1}});
        assert!(serde_json::from_value::<RateLimit>(body).is_err());
    }

    /// Neither the settings document, nor the status answer, nor the event ever
    /// carries a secret: only the credential REFERENCE appears.
    #[test]
    fn neither_the_report_nor_the_event_echoes_a_secret() {
        let record = with_limits(vec![limit(
            RateLimitKind::Requests,
            unit(RateLimitUnit::Minute),
            None,
            40.0,
            RateLimitScope::Credential(CredentialScope {
                credential: "vault:deepseek".into(),
            }),
        )]);
        let document = serde_json::to_string(&record).unwrap();
        assert!(document.contains("vault:deepseek"), "the reference is kept");
        assert!(!document.contains("sk-"), "never a value");

        let report = serde_json::to_string(&rate_limit_report(&record)).unwrap();
        assert!(report.contains("vault:deepseek"));
        assert!(!report.contains("sk-"));

        let event = ProviderRateLimitEvent {
            event_type: RATE_LIMIT_EVENT.into(),
            provider_id: record.id.clone(),
            credential: Some("vault:deepseek".into()),
            scope: RateLimitScope::Credential(CredentialScope {
                credential: "vault:deepseek".into(),
            }),
            kind: RateLimitKind::Requests,
            window: unit(RateLimitUnit::Minute),
            status: RateLimitStatus::Limited,
            reason: Some("quota".into()),
            resets_at: Some("2026-10-08T12:34:00Z".into()),
            next_retry_at: None,
        };
        let wire = serde_json::to_string(&event).unwrap();
        assert!(wire.contains("vault:deepseek") && !wire.contains("sk-"));
        assert!(wire.contains("\"type\":\"provider_rate_limit\""));
    }

    #[test]
    fn a_known_timezone_is_utc_in_any_case_or_an_iana_name() {
        for good in [
            "utc",
            "UTC",
            "Utc",
            "Europe/Paris",
            "America/New_York",
            "Asia/Tokyo",
        ] {
            assert!(known_timezone(good), "{good}");
        }
        for bad in ["Mars/Olympus", "GMT+2", "fr", "europe/paris"] {
            assert!(!known_timezone(bad), "{bad}");
        }
    }
}
