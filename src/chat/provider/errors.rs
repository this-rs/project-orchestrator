//! Typed opening errors (decisions A10, A29).
//!
//! Maps a nexus [`ProviderError`] or a [`ResolveError`] to an HTTP status, a
//! stable `code` and a short user-facing message. The message is written here,
//! not copied from the source error: the free-text fields (`detail`, `program`,
//! `model`) are never echoed raw, so no file path, URL or credential can reach
//! the client. The one exception is the cause of an unreachable endpoint, which
//! is echoed **only** when `safe_cause` proves it is a short, URL- and
//! credential-free line (ERR-1: the cause is data, but a leak is not).

use nexus_claude::agent::ProviderError;
use serde::Serialize;

use super::resolver::ResolveError;

/// What the API answers when a session cannot be opened.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpenFailure {
    /// HTTP status. Not part of the JSON body.
    #[serde(skip)]
    pub status: u16,
    /// Stable error code; the frontend picks its error card from it.
    pub code: &'static str,
    /// Short user-facing message, free of internal detail.
    #[serde(rename = "error")]
    pub message: String,
    /// Provider instance concerned, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_id: Option<String>,
    /// Suggested action (for `auth_required`: the login command to run).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// Whether trying again later, unchanged, can succeed.
    pub retryable: bool,
    /// Delay asked by the provider before retrying, in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
    /// Models the conversation can open on instead, without the vault (Auto
    /// refused on a locked vault). Empty: nothing to offer but unlocking.
    /// `Some` (maybe empty) only for that refusal: the client tells it from a
    /// named instance's locked credential by the field's presence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallbacks: Option<Vec<FallbackModel>>,
    /// `endpoint_not_allowed` only: where the consent is missing, so the client knows
    /// which consent a person gives — `project` (`PUT /api/projects/{slug}/llm-consents/
    /// {provider_id}`) or `no_project` (`PUT /api/chat/llm-consents/{provider_id}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consent_scope: Option<&'static str>,
    /// `provider_not_chosen` only: the instances the request (or the pilot role) can name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Vec<String>>,
}

/// An error raised while opening (or resuming) a session ON a known provider instance:
/// the HTTP layer reports THAT instance, never the one it would assume (B6: a locked vault
/// or a missing executable of a third-party instance was shown as Claude Code's). Displays
/// as the error it carries, whose causes stay in the chain.
#[derive(Debug)]
pub struct OnProvider {
    /// The instance the opening was on.
    pub provider_id: String,
    source: anyhow::Error,
}

impl std::fmt::Display for OnProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.source, f)
    }
}

impl std::error::Error for OnProvider {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(AsRef::<dyn std::error::Error + Send + Sync>::as_ref(
            &self.source,
        ))
    }
}

/// Tags `err` with the instance it was raised on. An error already tagged keeps its tag
/// (the innermost opening knows best).
pub fn on_provider(provider_id: &str, err: anyhow::Error) -> anyhow::Error {
    if err.chain().any(|c| c.is::<OnProvider>()) {
        return err;
    }
    anyhow::Error::new(OnProvider {
        provider_id: provider_id.to_string(),
        source: err,
    })
}

/// A model offered instead when Auto cannot route with the vault locked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FallbackModel {
    /// Provider instance id.
    pub provider_id: String,
    /// Model id on that instance.
    pub model: String,
}

/// Auto refused because the vault is locked and holds keys of candidates it
/// would have routed to. Answers `credentials_locked` (423) with `fallbacks`:
/// the person unlocks, or picks one of the models that need no vault.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the vault is locked: Auto cannot route to the providers whose key it holds")]
pub struct AutoVaultLocked {
    /// Accessible models, best first.
    pub fallbacks: Vec<FallbackModel>,
}

impl OpenFailure {
    /// JSON body: `{ "error", "code", "provider_id"?, "action"?, "retryable", "retry_after_ms"? }`.
    pub fn to_json(&self) -> serde_json::Value {
        let mut body = serde_json::Map::new();
        body.insert(
            "error".to_string(),
            serde_json::Value::String(self.message.clone()),
        );
        body.insert(
            "code".to_string(),
            serde_json::Value::String(self.code.to_string()),
        );
        if let Some(provider_id) = &self.provider_id {
            body.insert(
                "provider_id".to_string(),
                serde_json::Value::String(provider_id.clone()),
            );
        }
        if let Some(action) = &self.action {
            body.insert(
                "action".to_string(),
                serde_json::Value::String(action.clone()),
            );
        }
        body.insert(
            "retryable".to_string(),
            serde_json::Value::Bool(self.retryable),
        );
        if let Some(retry_after_ms) = self.retry_after_ms {
            body.insert(
                "retry_after_ms".to_string(),
                serde_json::Value::from(retry_after_ms),
            );
        }
        if let Some(fallbacks) = &self.fallbacks {
            body.insert(
                "fallbacks".to_string(),
                serde_json::to_value(fallbacks).unwrap_or_default(),
            );
        }
        if let Some(scope) = self.consent_scope {
            body.insert(
                "consent_scope".to_string(),
                serde_json::Value::String(scope.to_string()),
            );
        }
        if let Some(candidates) = &self.candidates {
            body.insert(
                "candidates".to_string(),
                serde_json::to_value(candidates).unwrap_or_default(),
            );
        }
        serde_json::Value::Object(body)
    }
}

/// Whether a capability name is a plain identifier, safe to show.
fn is_plain_identifier(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= 64
        && text
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Longest `detail` this module will echo back to a client.
const MAX_CAUSE_BYTES: usize = 200;

/// A `detail` from a provider error, but only when it is plainly a short cause
/// and carries no URL, no host, no path and no credential-shaped token.
///
/// The detail is redacted upstream by nexus (`agent::credentials::redact_with`),
/// which masks registered secrets, `Bearer`/`Basic` tokens and known fields and
/// truncates to 512 bytes. It does **not** strip an arbitrary URL or a file path,
/// so echoing it verbatim could still hand a client a host, a path or a stray
/// key. This gate is the second lock: a detail that fails any test below is
/// dropped (the generic message stands) rather than sanitised — a half-masked
/// URL is still a URL.
///
/// `None` means "not safe to show".
fn safe_cause(detail: &str) -> Option<&str> {
    let cause = detail.trim();
    if cause.is_empty() || cause.len() > MAX_CAUSE_BYTES {
        return None;
    }
    // A scheme, an absolute path or userinfo is never part of a cause we show.
    if cause.contains("://") || cause.contains('/') || cause.contains('\\') {
        return None;
    }
    // A control character (CR, LF, ESC) could forge a log line or a header.
    if cause.chars().any(char::is_control) {
        return None;
    }
    // One credential-shaped token in the whole detail disqualifies it: the line
    // as a whole must be quotable, not merely free of the exact secret.
    if cause
        .split(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | '(' | ')' | '[' | ']'))
        .any(looks_like_a_secret)
    {
        return None;
    }
    Some(cause)
}

/// Whether one whitespace-delimited token is shaped like a credential. Errs on
/// the side of caution: a false positive just hides a cause, a false negative
/// leaks a key.
fn looks_like_a_secret(token: &str) -> bool {
    let token = token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.');
    if token.len() < 12 {
        return false;
    }
    let lower = token.to_ascii_lowercase();
    for prefix in [
        "sk-",
        "sk_",
        "pk-",
        "pk_",
        "ghp_",
        "gho_",
        "ghs_",
        "github_pat_",
        "xoxb-",
        "xoxp-",
        "aiza",
        "akia",
        "bearer",
        "token=",
        "api_key=",
        "apikey=",
        "password=",
        "secret=",
    ] {
        if lower.starts_with(prefix) {
            return true;
        }
    }
    // A long run mixing letters and digits and no separator is key-shaped.
    let has_alpha = token.chars().any(|c| c.is_ascii_alphabetic());
    let has_digit = token.chars().any(|c| c.is_ascii_digit());
    has_alpha && has_digit && token.len() >= 24
}

/// The message of an unreachable endpoint: the generic line, the instance when
/// one is known, and the masked cause when the detail survives `safe_cause`.
/// Never the raw detail, never a URL or a credential.
fn unreachable_message(provider_id: Option<&str>, detail: &str) -> String {
    let mut message = String::from("The provider endpoint could not be reached");
    if let Some(id) = provider_id {
        message.push_str(&format!(" ({id})"));
    }
    if let Some(cause) = safe_cause(detail) {
        message.push_str(&format!(": {cause}"));
    }
    message.push('.');
    message
}

/// The masked cause of a provider error, when the error names one and it survives
/// [`safe_cause`] — the very gate the user-facing message goes through, so an
/// outage event can never carry more than that message already shows.
///
/// `None` when the error names no cause, or when the one it names is not safe to
/// show: an outage is still an outage without it.
pub fn cause_of(err: &ProviderError) -> Option<String> {
    match err {
        ProviderError::EndpointUnreachable { detail } => safe_cause(detail).map(str::to_owned),
        _ => None,
    }
}

/// Maps a provider error met while opening (or resuming) a session.
///
/// `unauthorized` answers 502, not 401: a 401 would log the user out of the
/// frontend, while the rejected credential is the provider's, not the user's.
pub fn open_failure(err: &ProviderError, provider_id: Option<&str>) -> OpenFailure {
    let mut action: Option<String> = None;
    let mut retry_after_ms: Option<u64> = None;
    let (status, code, message): (u16, &'static str, String) = match err {
        ProviderError::CliNotFound { .. } => (
            424,
            err.kind(),
            "The provider's command-line tool is not installed on the server.".to_string(),
        ),
        ProviderError::AuthRequired { login_hint } => {
            action = login_hint.clone();
            (
                424,
                err.kind(),
                "The provider requires a login before it can be used.".to_string(),
            )
        }
        ProviderError::CredentialsLocked => (
            423,
            err.kind(),
            "The credential store is locked. Unlock it and try again.".to_string(),
        ),
        ProviderError::Unauthorized => (
            502,
            err.kind(),
            "The provider rejected the configured credential.".to_string(),
        ),
        ProviderError::EndpointUnreachable { detail } => {
            (502, err.kind(), unreachable_message(provider_id, detail))
        }
        ProviderError::ModelNoTools { .. } => (
            422,
            err.kind(),
            "The selected model does not support tool calls.".to_string(),
        ),
        ProviderError::ContextTooSmall { needed, available } => {
            let message = match (needed, available) {
                (Some(needed), Some(available)) => format!(
                    "The model's context window is too small: {needed} tokens needed, {available} available."
                ),
                (Some(needed), None) => {
                    format!("The model's context window is too small: {needed} tokens needed.")
                }
                (None, Some(available)) => format!(
                    "The model's context window is too small: {available} tokens available."
                ),
                (None, None) => "The model's context window is too small.".to_string(),
            };
            (422, err.kind(), message)
        }
        ProviderError::RateLimited {
            retry_after_ms: delay,
        } => {
            retry_after_ms = *delay;
            (
                429,
                err.kind(),
                "The provider is rate limiting requests. Try again later.".to_string(),
            )
        }
        ProviderError::Overloaded => (
            503,
            err.kind(),
            "The provider is temporarily overloaded. Try again later.".to_string(),
        ),
        ProviderError::Timeout { .. } => (
            504,
            err.kind(),
            "The provider did not answer in time.".to_string(),
        ),
        ProviderError::ProcessExited { .. } => (
            502,
            err.kind(),
            "The provider process stopped unexpectedly.".to_string(),
        ),
        ProviderError::Protocol { .. } => (
            502,
            err.kind(),
            "The provider sent an unexpected answer.".to_string(),
        ),
        ProviderError::Unsupported { capability } => {
            let message = if is_plain_identifier(capability) {
                format!("The provider does not support this capability: {capability}.")
            } else {
                "The provider does not support this capability.".to_string()
            };
            (422, err.kind(), message)
        }
        ProviderError::TurnInProgress => (
            409,
            err.kind(),
            "A turn is already in progress on this session.".to_string(),
        ),
        ProviderError::InvalidRequest { .. } => (
            400,
            err.kind(),
            "The provider refused the request as invalid.".to_string(),
        ),
        ProviderError::Closed => (410, err.kind(), "The session is closed.".to_string()),
        // `ProviderError` is non-exhaustive: a variant added by nexus maps here
        // until this table learns it.
        _ => (
            502,
            "provider_error",
            "The provider failed to open the session.".to_string(),
        ),
    };
    OpenFailure {
        status,
        code,
        message,
        provider_id: provider_id.map(|id| id.to_string()),
        action,
        retryable: err.retryable(),
        retry_after_ms,
        fallbacks: None,
        consent_scope: None,
        candidates: None,
    }
}

/// Maps a resolution error (no session was even attempted).
/// Turn an SDK error raised while starting the Claude Code CLI into the typed
/// provider error it stands for, when it stands for one.
pub fn provider_error_from_sdk(err: &nexus_claude::SdkError) -> Option<ProviderError> {
    match err {
        nexus_claude::SdkError::CliNotFound { .. } => Some(ProviderError::CliNotFound {
            program: "claude".to_string(),
        }),
        nexus_claude::SdkError::Timeout { seconds } => Some(ProviderError::Timeout {
            after_ms: seconds.saturating_mul(1000),
        }),
        _ => None,
    }
}

/// Wrap an SDK error raised while opening a session so that the HTTP layer can
/// still tell what it was: the typed [`ProviderError`] travels inside the
/// `anyhow` chain, under the same human-readable message as before.
pub fn sdk_open_error(context: &str, err: nexus_claude::SdkError) -> anyhow::Error {
    let message = format!("{context}: {err}");
    match provider_error_from_sdk(&err) {
        Some(typed) => anyhow::Error::new(typed).context(message),
        None => anyhow::anyhow!(message),
    }
}

/// The code a session refused for lack of a slot answers with (HTTP 429).
pub const SESSION_LIMIT_REACHED: &str = "session_limit_reached";

/// The session table is full: `active` live sessions on both engines, `limit` allowed. A
/// capacity limit, not a server fault (issue #876): the caller can wait, close a session or
/// ask a person. The message keeps the historical wording (`Maximum number of active
/// sessions reached (N)`), which callers and tests match on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("Maximum number of active sessions reached ({limit})")]
pub struct SessionLimitReached {
    pub active: usize,
    pub limit: usize,
}

impl SessionLimitReached {
    /// What the API answers: 429, retryable (a slot frees when a session ends or idles out).
    pub fn failure(&self) -> OpenFailure {
        OpenFailure {
            status: 429,
            code: SESSION_LIMIT_REACHED,
            message: format!(
                "{} ({} active): close a session or wait for one to finish",
                self, self.active
            ),
            provider_id: None,
            action: Some("close_a_session".to_string()),
            retryable: true,
            retry_after_ms: Some(30_000),
            fallbacks: None,
            consent_scope: None,
            candidates: None,
        }
    }
}

/// The typed answer for an error returned by `create_session` /
/// `resume_session`, when the error chain carries a typed cause. `None` means
/// "not an opening failure we can name" — the caller answers 500.
pub fn classify_open_error(err: &anyhow::Error, provider_id: Option<&str>) -> Option<OpenFailure> {
    // The instance the error was raised on, when the opening knew it, wins over the
    // caller's guess (B6).
    let tagged = err
        .chain()
        .find_map(|cause| cause.downcast_ref::<OnProvider>())
        .map(|tag| tag.provider_id.as_str());
    let provider_id = tagged.or(provider_id);
    err.chain().find_map(|cause| {
        if cause.is::<crate::chat::mirror::MirrorReadOnly>() {
            return Some(crate::chat::mirror::MirrorReadOnly::failure());
        }
        // B36-RACE: a delegation refused because its parent was cancelled meanwhile.
        if let Some(refused) =
            cause.downcast_ref::<crate::chat::cancelled_parents::ParentCancelled>()
        {
            return Some(refused.failure());
        }
        if let Some(full) = cause.downcast_ref::<SessionLimitReached>() {
            return Some(full.failure());
        }
        if let Some(locked) = cause.downcast_ref::<AutoVaultLocked>() {
            let mut failure = open_failure(&ProviderError::CredentialsLocked, None);
            failure.fallbacks = Some(locked.fallbacks.clone());
            Some(failure)
        } else if let Some(typed) = cause.downcast_ref::<ProviderError>() {
            Some(open_failure(typed, provider_id))
        } else {
            cause.downcast_ref::<ResolveError>().map(resolve_failure)
        }
    })
}

pub fn resolve_failure(err: &ResolveError) -> OpenFailure {
    let (status, message, provider_id, retryable): (u16, String, Option<String>, bool) = match err {
        ResolveError::ProviderConflict { session, .. } => (
            409,
            "This session is bound to another provider; a session never changes provider."
                .to_string(),
            Some(session.clone()),
            false,
        ),
        ResolveError::UnknownProvider(id) => (
            404,
            "This provider is not configured.".to_string(),
            Some(id.clone()),
            false,
        ),
        ResolveError::NotAllowed(id) => (
            403,
            "This provider is not allowed for this project: a person allows it in the \
             project's provider settings."
                .to_string(),
            Some(id.clone()),
            false,
        ),
        ResolveError::EndpointRefused(id) => (
            403,
            "This provider's endpoint is refused by the server's endpoint guard (scheme, \
             private address or unresolvable host): change the instance's URL; no consent \
             changes that."
                .to_string(),
            Some(id.clone()),
            false,
        ),
        ResolveError::NotAllowedOutsideProject(id) => (
            403,
            "This provider is not allowed for chats outside a project: a person allows it \
             for those in the provider settings, or the chat opens in a project that allows it."
                .to_string(),
            Some(id.clone()),
            false,
        ),
        ResolveError::Unavailable { provider_id, .. } => (
            503,
            "This provider is currently unavailable.".to_string(),
            Some(provider_id.clone()),
            true,
        ),
        ResolveError::EngineUnavailable { provider_id } => (
            409,
            "This session was opened on the agent engine (CHAT_PROVIDER_PATH=agent), which is \
             switched off on this server. Switch it back on to resume it; the session was not \
             changed."
                .to_string(),
            Some(provider_id.clone()),
            false,
        ),
        ResolveError::NoProvider => (
            409,
            "No provider is usable: Claude Code is not installed on the server and no other \
             provider is configured and allowed here. Install the Claude Code CLI, or \
             configure a provider."
                .to_string(),
            None,
            false,
        ),
        ResolveError::ProviderNotChosen { .. } => (
            409,
            "Claude Code is not installed on the server and several providers are usable \
             here: choose one for this chat, or set the default (pilot) provider."
                .to_string(),
            None,
            false,
        ),
    };
    OpenFailure {
        status,
        code: err.code(),
        message,
        provider_id,
        action: None,
        retryable,
        retry_after_ms: None,
        fallbacks: None,
        consent_scope: match err {
            ResolveError::NotAllowed(_) => Some("project"),
            ResolveError::NotAllowedOutsideProject(_) => Some("no_project"),
            _ => None,
        },
        candidates: match err {
            ResolveError::ProviderNotChosen { candidates } => Some(candidates.clone()),
            _ => None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::super::resolver::Role;
    use super::*;

    // ── SDK errors raised while starting the CLI ───────────────────────────

    #[test]
    fn a_missing_cli_is_cli_not_found_424_not_a_mute_500() {
        let sdk = nexus_claude::SdkError::CliNotFound {
            searched_paths: "/Users/x/.config/claude\n/usr/local/bin".to_string(),
        };
        let err = sdk_open_error("Failed to create InteractiveClient", sdk);
        // The log message is unchanged…
        assert!(err
            .to_string()
            .starts_with("Failed to create InteractiveClient: "));
        // …and the HTTP layer can still name the failure.
        let failure = classify_open_error(&err, Some("claude-code")).expect("typed failure");
        assert_eq!(failure.status, 424);
        assert_eq!(failure.code, "cli_not_found");
        assert_eq!(failure.provider_id.as_deref(), Some("claude-code"));
        let body = failure.to_json().to_string();
        assert!(!body.contains("/Users/x"), "searched paths leaked: {body}");
    }

    #[test]
    fn a_full_session_table_is_a_429_with_a_stable_code() {
        let err = anyhow::Error::new(SessionLimitReached {
            active: 10,
            limit: 10,
        })
        .context("Failed to create session");
        let failure = classify_open_error(&err, None).expect("typed, not a bare 500");
        assert_eq!(failure.status, 429);
        assert_eq!(failure.code, SESSION_LIMIT_REACHED);
        assert!(failure.retryable);
        let body = failure.to_json();
        assert_eq!(body["code"], "session_limit_reached");
        let message = body["error"].as_str().unwrap();
        assert!(message.contains("Maximum number of active sessions reached (10)"));
        assert!(message.contains("10 active"));
    }

    #[test]
    fn an_untyped_error_is_not_classified() {
        let sdk = nexus_claude::SdkError::ConnectionError("pipe closed".to_string());
        let err = sdk_open_error("Failed to connect InteractiveClient", sdk);
        assert!(classify_open_error(&err, None).is_none());
        assert!(classify_open_error(&anyhow::anyhow!("anything else"), None).is_none());
    }

    #[test]
    fn a_typed_cause_is_found_under_added_context() {
        let err = anyhow::Error::new(ProviderError::CredentialsLocked)
            .context("opening session")
            .context("create_session failed");
        let failure = classify_open_error(&err, Some("deepseek")).expect("typed failure");
        assert_eq!((failure.status, failure.code), (423, "credentials_locked"));
    }

    const SECRET: &str = "sk-secret-123";
    const PATH: &str = "/Users/x/.config";

    fn leaky() -> String {
        format!("GET https://user:{SECRET}@host/v1 failed, see {PATH}/provider.json")
    }

    /// Checks status and code, and that nothing internal reaches the body.
    fn assert_mapped(err: ProviderError, status: u16, code: &str) -> OpenFailure {
        let failure = open_failure(&err, Some("deepseek-prod"));
        assert_eq!(failure.status, status, "{code}");
        assert_eq!(failure.code, code);
        assert_eq!(failure.provider_id.as_deref(), Some("deepseek-prod"));
        assert_eq!(failure.retryable, err.retryable(), "{code}");
        let body = failure.to_json();
        assert_eq!(body["code"], serde_json::json!(code));
        assert_eq!(body["error"], serde_json::json!(failure.message.clone()));
        assert_eq!(body["provider_id"], serde_json::json!("deepseek-prod"));
        assert!(body.get("status").is_none());
        let text = body.to_string();
        assert!(!text.contains(SECRET), "{code}: secret leaked in {text}");
        assert!(!text.contains(PATH), "{code}: path leaked in {text}");
        assert!(!text.contains("https://"), "{code}: URL leaked in {text}");
        // The derived serialisation is the same body.
        assert_eq!(serde_json::to_value(&failure).unwrap(), body);
        failure
    }

    #[test]
    fn cli_not_found_is_424_without_the_program_path() {
        let failure = assert_mapped(
            ProviderError::CliNotFound {
                program: format!("{PATH}/bin/claude --key {SECRET}"),
            },
            424,
            "cli_not_found",
        );
        assert!(!failure.retryable);
        assert_eq!(failure.action, None);
    }

    #[test]
    fn auth_required_is_424_and_carries_the_login_hint_as_action() {
        let failure = assert_mapped(
            ProviderError::AuthRequired {
                login_hint: Some("claude login".to_string()),
            },
            424,
            "auth_required",
        );
        assert_eq!(failure.action.as_deref(), Some("claude login"));
        assert_eq!(
            failure.to_json()["action"],
            serde_json::json!("claude login")
        );

        let bare = assert_mapped(
            ProviderError::AuthRequired { login_hint: None },
            424,
            "auth_required",
        );
        assert_eq!(bare.action, None);
        assert!(bare.to_json().get("action").is_none());
    }

    #[test]
    fn credentials_locked_is_423() {
        assert_mapped(ProviderError::CredentialsLocked, 423, "credentials_locked");
    }

    #[test]
    fn unauthorized_is_502_never_401() {
        let failure = assert_mapped(ProviderError::Unauthorized, 502, "unauthorized");
        assert_ne!(failure.status, 401);
    }

    #[test]
    fn endpoint_unreachable_is_502_retryable_and_names_the_instance_only() {
        // `leaky()` carries a URL, a userinfo secret and a path: none may reach
        // the message, so the cause is dropped whole — the instance is named,
        // the raw detail never is.
        let failure = assert_mapped(
            ProviderError::EndpointUnreachable { detail: leaky() },
            502,
            "endpoint_unreachable",
        );
        assert!(failure.retryable);
        assert_eq!(
            failure.message,
            "The provider endpoint could not be reached (deepseek-prod)."
        );
    }

    #[test]
    fn endpoint_unreachable_names_the_masked_cause() {
        // A plain cause — the shape nexus produces once redaction has run — is
        // echoed, so a user can tell a DNS failure from a refused connection.
        let failure = open_failure(
            &ProviderError::EndpointUnreachable {
                detail: "connection failed: Connection refused (os error 61)".to_string(),
            },
            Some("deepseek-prod"),
        );
        assert_eq!(
            (failure.status, failure.code),
            (502, "endpoint_unreachable")
        );
        assert!(
            failure.message.contains("connection failed"),
            "the cause must be named, got {:?}",
            failure.message
        );
        assert!(failure.message.contains("deepseek-prod"));
    }

    #[test]
    fn endpoint_unreachable_never_shows_a_secret_in_the_detail() {
        // A secret-shaped token with no URL and no path: the detail is not
        // "safe" either, so it is dropped rather than trimmed around.
        const API_KEY: &str = "sk-live-9f3a2b7c1d4e6f8a0b2c4d6e";
        let failure = assert_mapped(
            ProviderError::EndpointUnreachable {
                detail: format!("authentication failed for key {API_KEY}"),
            },
            502,
            "endpoint_unreachable",
        );
        let body = failure.to_json().to_string();
        assert!(!failure.message.contains(API_KEY), "secret leaked: {body}");
        assert!(!body.contains(API_KEY), "secret leaked: {body}");
        assert_eq!(
            failure.message,
            "The provider endpoint could not be reached (deepseek-prod)."
        );
    }

    #[test]
    fn safe_cause_keeps_a_plain_cause_and_drops_anything_leaky() {
        assert_eq!(
            safe_cause("dns resolution failed for the api host"),
            Some("dns resolution failed for the api host")
        );
        // A scheme, a path, a control character or a key-shaped token each drop it.
        assert_eq!(safe_cause("GET https://api.example.com/v1 failed"), None);
        assert_eq!(safe_cause("see /Users/x/.config/provider.json"), None);
        assert_eq!(safe_cause("boom\nsecond line"), None);
        assert_eq!(safe_cause("Bearer sk-abcdef0123456789abcdef"), None);
        assert_eq!(safe_cause(""), None);
    }

    #[test]
    fn cause_of_names_a_plain_cause_and_drops_anything_leaky() {
        // The outage event takes its `cause` from here, so it can never carry more
        // than the message the user already reads.
        assert_eq!(
            cause_of(&ProviderError::EndpointUnreachable {
                detail: "connection failed: Connection refused (os error 61)".to_string(),
            })
            .as_deref(),
            Some("connection failed: Connection refused (os error 61)")
        );
        assert_eq!(
            cause_of(&ProviderError::EndpointUnreachable {
                detail: "GET https://api.example.com/v1 failed".to_string(),
            }),
            None,
            "a URL is not a cause we show"
        );
        // An error that names no endpoint names no cause either.
        assert_eq!(cause_of(&ProviderError::Overloaded), None);
    }

    #[test]
    fn model_no_tools_is_422() {
        assert_mapped(
            ProviderError::ModelNoTools { model: leaky() },
            422,
            "model_no_tools",
        );
    }

    #[test]
    fn context_too_small_is_422_with_the_figures_when_known() {
        let failure = assert_mapped(
            ProviderError::ContextTooSmall {
                needed: Some(12000),
                available: Some(8192),
            },
            422,
            "context_too_small",
        );
        assert!(failure.message.contains("12000"));
        assert!(failure.message.contains("8192"));

        let bare = assert_mapped(
            ProviderError::ContextTooSmall {
                needed: None,
                available: None,
            },
            422,
            "context_too_small",
        );
        assert_eq!(bare.message, "The model's context window is too small.");
    }

    #[test]
    fn rate_limited_is_429_and_keeps_the_delay() {
        let failure = assert_mapped(
            ProviderError::RateLimited {
                retry_after_ms: Some(1500),
            },
            429,
            "rate_limited",
        );
        assert!(failure.retryable);
        assert_eq!(failure.retry_after_ms, Some(1500));
        let body = failure.to_json();
        assert_eq!(body["retryable"], serde_json::json!(true));
        assert_eq!(body["retry_after_ms"], serde_json::json!(1500));

        let bare = assert_mapped(
            ProviderError::RateLimited {
                retry_after_ms: None,
            },
            429,
            "rate_limited",
        );
        assert_eq!(bare.retry_after_ms, None);
        assert!(bare.to_json().get("retry_after_ms").is_none());
    }

    #[test]
    fn overloaded_is_503_and_retryable() {
        let failure = assert_mapped(ProviderError::Overloaded, 503, "overloaded");
        assert!(failure.retryable);
    }

    #[test]
    fn timeout_is_504_and_retryable() {
        let failure = assert_mapped(ProviderError::Timeout { after_ms: 30000 }, 504, "timeout");
        assert!(failure.retryable);
    }

    #[test]
    fn process_exited_is_502() {
        let failure = assert_mapped(
            ProviderError::ProcessExited { code: Some(137) },
            502,
            "process_exited",
        );
        assert!(!failure.retryable);
    }

    #[test]
    fn protocol_is_502_generic() {
        let failure = assert_mapped(ProviderError::Protocol { detail: leaky() }, 502, "protocol");
        assert_eq!(failure.message, "The provider sent an unexpected answer.");
    }

    #[test]
    fn unsupported_is_422_and_names_only_a_plain_capability() {
        let named = assert_mapped(
            ProviderError::Unsupported {
                capability: "set_model_live".to_string(),
            },
            422,
            "unsupported",
        );
        assert!(named.message.contains("set_model_live"));

        let hidden = assert_mapped(
            ProviderError::Unsupported {
                capability: leaky(),
            },
            422,
            "unsupported",
        );
        assert_eq!(
            hidden.message,
            "The provider does not support this capability."
        );
    }

    #[test]
    fn turn_in_progress_is_409() {
        assert_mapped(ProviderError::TurnInProgress, 409, "turn_in_progress");
    }

    #[test]
    fn invalid_request_is_400_without_the_detail() {
        assert_mapped(
            ProviderError::InvalidRequest { detail: leaky() },
            400,
            "invalid_request",
        );
    }

    #[test]
    fn closed_is_410() {
        assert_mapped(ProviderError::Closed, 410, "closed");
    }

    #[test]
    fn provider_id_is_omitted_when_unknown() {
        let failure = open_failure(&ProviderError::Overloaded, None);
        assert_eq!(failure.provider_id, None);
        let body = failure.to_json();
        assert!(body.get("provider_id").is_none());
        assert_eq!(serde_json::to_value(&failure).unwrap(), body);
    }

    fn assert_resolved(
        err: ResolveError,
        status: u16,
        code: &str,
        provider_id: Option<&str>,
    ) -> OpenFailure {
        let failure = resolve_failure(&err);
        assert_eq!(failure.status, status, "{code}");
        assert_eq!(failure.code, code);
        assert_eq!(failure.provider_id.as_deref(), provider_id);
        assert_eq!(failure.action, None);
        assert_eq!(failure.retry_after_ms, None);
        let body = failure.to_json();
        assert_eq!(body["code"], serde_json::json!(code));
        assert_eq!(serde_json::to_value(&failure).unwrap(), body);
        failure
    }

    #[test]
    fn provider_conflict_is_409() {
        let failure = assert_resolved(
            ResolveError::ProviderConflict {
                session: "claude-code".to_string(),
                requested: "deepseek-prod".to_string(),
            },
            409,
            "provider_conflict",
            Some("claude-code"),
        );
        assert!(!failure.retryable);
    }

    #[test]
    fn provider_unknown_is_404() {
        assert_resolved(
            ResolveError::UnknownProvider("ghost".to_string()),
            404,
            "provider_unknown",
            Some("ghost"),
        );
    }

    #[test]
    fn endpoint_not_allowed_is_403() {
        assert_resolved(
            ResolveError::NotAllowed("deepseek-prod".to_string()),
            403,
            "endpoint_not_allowed",
            Some("deepseek-prod"),
        );
    }

    #[test]
    fn a_refused_consent_says_which_consent_is_missing() {
        let in_project = resolve_failure(&ResolveError::NotAllowed("local".into()));
        assert_eq!(in_project.to_json()["consent_scope"], "project");
        let outside = assert_resolved(
            ResolveError::NotAllowedOutsideProject("local".into()),
            403,
            "endpoint_not_allowed",
            Some("local"),
        );
        let body = outside.to_json();
        assert_eq!(body["consent_scope"], "no_project");
        assert!(body["error"]
            .as_str()
            .unwrap()
            .contains("outside a project"));
        // A refusal of the endpoint guard asks for no consent: none would fix it.
        let guard = assert_resolved(
            ResolveError::EndpointRefused("local".into()),
            403,
            "endpoint_not_allowed",
            Some("local"),
        );
        assert_eq!(guard.consent_scope, None);
        assert!(guard.to_json().get("consent_scope").is_none());
        // Any other refusal carries neither field.
        let other = resolve_failure(&ResolveError::NoProvider).to_json();
        assert!(other.get("consent_scope").is_none() && other.get("candidates").is_none());
    }

    #[test]
    fn several_usable_providers_without_claude_code_are_a_409_listing_them() {
        let failure = assert_resolved(
            ResolveError::ProviderNotChosen {
                candidates: vec!["a-cloud".into(), "b-local".into()],
            },
            409,
            "provider_not_chosen",
            None,
        );
        assert_eq!(
            failure.to_json()["candidates"],
            serde_json::json!(["a-cloud", "b-local"])
        );
    }

    #[test]
    fn an_error_tagged_with_its_instance_reports_that_instance_not_the_callers_guess() {
        // B6: the caller (the HTTP handler) assumed claude-code; the opening was on `local`.
        let err = on_provider(
            "local",
            anyhow::Error::new(ProviderError::CredentialsLocked).context("opening"),
        );
        let failure = classify_open_error(&err, Some("claude-code")).unwrap();
        assert_eq!((failure.status, failure.code), (423, "credentials_locked"));
        assert_eq!(failure.provider_id.as_deref(), Some("local"));
        // The tag changes neither the text nor what the chain carries.
        assert_eq!(err.to_string(), "opening");
        assert!(err.chain().any(|c| matches!(
            c.downcast_ref::<ProviderError>(),
            Some(ProviderError::CredentialsLocked)
        )));
        // The innermost tag wins; a resolver refusal keeps its own instance.
        let twice = on_provider("outer", err);
        assert_eq!(
            classify_open_error(&twice, None)
                .unwrap()
                .provider_id
                .as_deref(),
            Some("local")
        );
        let refused = on_provider(
            "local",
            anyhow::Error::new(ResolveError::UnknownProvider("ghost".into())),
        );
        assert_eq!(
            classify_open_error(&refused, None)
                .unwrap()
                .provider_id
                .as_deref(),
            Some("ghost")
        );
        // Untagged: the caller's guess, as before.
        let plain = anyhow::Error::new(ProviderError::CredentialsLocked);
        assert_eq!(
            classify_open_error(&plain, Some("claude-code"))
                .unwrap()
                .provider_id
                .as_deref(),
            Some("claude-code")
        );
    }

    #[test]
    fn provider_unavailable_is_503_and_retryable() {
        let failure = assert_resolved(
            ResolveError::Unavailable {
                provider_id: "deepseek-prod".to_string(),
                role: Role::Pilot,
            },
            503,
            "provider_unavailable",
            Some("deepseek-prod"),
        );
        assert!(failure.retryable);
    }

    #[test]
    fn no_provider_is_409_without_a_provider_id() {
        let failure = assert_resolved(ResolveError::NoProvider, 409, "no_provider", None);
        assert!(!failure.retryable);
        assert!(failure.to_json().get("provider_id").is_none());
    }

    #[test]
    fn a_switched_off_engine_is_a_typed_409_that_names_the_flag() {
        let f = resolve_failure(&ResolveError::EngineUnavailable {
            provider_id: "claude-code".into(),
        });
        assert_eq!((f.status, f.code), (409, "engine_unavailable"));
        assert!(f.message.contains("CHAT_PROVIDER_PATH"));
        assert_eq!(f.provider_id.as_deref(), Some("claude-code"));
        assert!(!f.retryable);
    }

    // ── the error-code table of the docs is complete ────────────────────────

    const DOC: &str = include_str!("../../../docs/api/provider-errors.md");

    fn documented(code: &str) -> bool {
        DOC.contains(&format!("`{code}`"))
    }

    #[test]
    fn every_code_of_an_opening_failure_is_in_the_docs_table() {
        use nexus_claude::agent::ProviderError as E;
        // Every variant nexus knows today, plus the catch-all of `open_failure`.
        // (`ProviderError` is non-exhaustive: a new variant lands in `provider_error`.)
        let errors = [
            E::CliNotFound {
                program: "x".into(),
            },
            E::AuthRequired { login_hint: None },
            E::CredentialsLocked,
            E::Unauthorized,
            E::EndpointUnreachable { detail: "x".into() },
            E::ModelNoTools { model: "x".into() },
            E::ContextTooSmall {
                needed: None,
                available: None,
            },
            E::RateLimited {
                retry_after_ms: None,
            },
            E::Overloaded,
            E::Timeout { after_ms: 1 },
            E::ProcessExited { code: None },
            E::Protocol { detail: "x".into() },
            E::unsupported("x"),
            E::TurnInProgress,
            E::invalid("x"),
            E::Closed,
        ];
        let mut codes: Vec<&'static str> =
            errors.iter().map(|e| open_failure(e, None).code).collect();
        codes.push("provider_error");
        for r in [
            ResolveError::ProviderConflict {
                session: "a".into(),
                requested: "b".into(),
            },
            ResolveError::UnknownProvider("a".into()),
            ResolveError::NotAllowed("a".into()),
            ResolveError::Unavailable {
                provider_id: "a".into(),
                role: Role::Pilot,
            },
            ResolveError::NoProvider,
            ResolveError::NotAllowedOutsideProject("a".into()),
            ResolveError::EndpointRefused("a".into()),
            ResolveError::ProviderNotChosen {
                candidates: vec!["a".into(), "b".into()],
            },
            ResolveError::EngineUnavailable {
                provider_id: "a".into(),
            },
        ] {
            codes.push(resolve_failure(&r).code);
        }
        for code in codes {
            assert!(
                documented(code),
                "`{code}` is emitted but missing from docs/api/provider-errors.md"
            );
        }
    }

    #[test]
    fn every_code_literal_of_the_handlers_is_in_the_docs_table() {
        // Codes the settings / vault / chat handlers and the middleware put at the
        // start of an error text (`"security_gate_closed: ..."`), plus the envelope codes.
        let sources = [
            (
                "provider_handlers.rs",
                include_str!("../../api/provider_handlers.rs"),
            ),
            (
                "vault_handlers.rs",
                include_str!("../../api/vault_handlers.rs"),
            ),
            (
                "chat_handlers.rs",
                include_str!("../../api/chat_handlers.rs"),
            ),
            ("middleware.rs", include_str!("../../auth/middleware.rs")),
        ];
        let literal = regex::Regex::new(r#""([a-z]+(?:_[a-z]+)+): "#).unwrap();
        let mut found = Vec::new();
        for (name, text) in sources {
            // Only the production part of the file.
            let production = text.split("#[cfg(test)]").next().unwrap_or(text);
            for c in literal.captures_iter(production) {
                found.push((name, c[1].to_string()));
            }
        }
        assert!(
            found.iter().any(|(_, c)| c == "security_gate_closed"),
            "the scan finds the known code"
        );
        for (name, code) in &found {
            assert!(
                documented(code),
                "`{code}` ({name}) is emitted but missing from docs/api/provider-errors.md"
            );
        }
        // The verdict codes of POST /providers/test.
        for code in [
            "credential_test_requires_saved_instance",
            "credentials_locked",
            "model_no_tools",
        ] {
            assert!(documented(code), "`{code}`");
        }
        // The envelope codes are documented by suffix under `envelope_*`.
        let envelope = include_str!("../envelope.rs");
        let enve = regex::Regex::new(r#""envelope_([a-z_]+)""#).unwrap();
        let production = envelope.split("#[cfg(test)]").next().unwrap_or(envelope);
        let mut n = 0;
        for c in enve.captures_iter(production) {
            n += 1;
            assert!(
                DOC.contains(&c[1]),
                "envelope code `envelope_{}` missing from the docs",
                &c[1]
            );
        }
        assert!(n >= 9, "the nine envelope codes were scanned");
    }

    #[test]
    fn the_retired_probe_code_is_not_emitted_anywhere() {
        // `probe_unavailable` was a placeholder before the real probe existed.
        for text in [
            include_str!("../../api/provider_handlers.rs"),
            include_str!("../../chat/manager.rs"),
            include_str!("native_factory.rs"),
        ] {
            let production = text.split("#[cfg(test)]").next().unwrap_or(text);
            assert!(!production.contains("probe_unavailable"));
        }
    }
}

#[cfg(test)]
mod auto_vault_locked_tests {
    use super::*;

    #[test]
    fn auto_on_a_locked_vault_is_credentials_locked_with_its_fallbacks() {
        let err = anyhow::Error::new(AutoVaultLocked {
            fallbacks: vec![FallbackModel {
                provider_id: "ollama".into(),
                model: "qwen3".into(),
            }],
        });
        let failure = classify_open_error(&err, Some("claude-code")).expect("typed");
        assert_eq!(failure.status, 423);
        assert_eq!(failure.code, "credentials_locked");
        // Auto chose no instance: none is named.
        assert_eq!(failure.provider_id, None);
        let body = failure.to_json();
        assert_eq!(
            body["fallbacks"],
            serde_json::json!([{ "provider_id": "ollama", "model": "qwen3" }])
        );
        assert_eq!(serde_json::to_value(&failure).unwrap(), body);
    }

    #[test]
    fn no_fallback_still_says_it_was_auto() {
        let err = anyhow::Error::new(AutoVaultLocked { fallbacks: vec![] });
        let body = classify_open_error(&err, None).expect("typed").to_json();
        assert_eq!(body["code"], serde_json::json!("credentials_locked"));
        assert_eq!(body["fallbacks"], serde_json::json!([]));
        // Any other opening error carries no such field.
        let named = open_failure(&ProviderError::CredentialsLocked, Some("openai")).to_json();
        assert!(named.get("fallbacks").is_none());
    }
}
