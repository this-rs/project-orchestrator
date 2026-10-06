//! Typed opening errors (decisions A10, A29).
//!
//! Maps a nexus [`ProviderError`] or a [`ResolveError`] to an HTTP status, a
//! stable `code` and a short user-facing message. The message is written here:
//! the free-text fields of the source error (`detail`, `program`, `model`) are
//! never copied, so no file path, URL or credential can reach the client.

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
        ProviderError::EndpointUnreachable { .. } => (
            502,
            err.kind(),
            "The provider endpoint could not be reached.".to_string(),
        ),
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

/// The typed answer for an error returned by `create_session` /
/// `resume_session`, when the error chain carries a typed cause. `None` means
/// "not an opening failure we can name" — the caller answers 500.
pub fn classify_open_error(err: &anyhow::Error, provider_id: Option<&str>) -> Option<OpenFailure> {
    err.chain().find_map(|cause| {
        if let Some(typed) = cause.downcast_ref::<ProviderError>() {
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
            "This provider is not allowed for this project.".to_string(),
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
            "No provider is configured. Configure one to start a session.".to_string(),
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
    fn endpoint_unreachable_is_502_generic_and_retryable() {
        let failure = assert_mapped(
            ProviderError::EndpointUnreachable { detail: leaky() },
            502,
            "endpoint_unreachable",
        );
        assert!(failure.retryable);
        assert_eq!(
            failure.message,
            "The provider endpoint could not be reached."
        );
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
