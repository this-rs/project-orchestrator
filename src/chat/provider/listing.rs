//! Body of `GET /api/chat/providers` (decision A42): the instances, their
//! health and the capabilities of each model, known BEFORE a session exists.
//!
//! The shape is fixed by `docs/api/chat-contract/provider-additions.json`.
//! Pure: the handler gathers the facts (health, catalogue), this module only
//! shapes them, so a secret can never reach the body: an instance carries a
//! credential REFERENCE and the origin of its endpoint, nothing else.

use nexus_claude::agent::{Capabilities, HealthStatus, ProviderError, ProviderHealth};
use serde::Serialize;
use serde_json::Value;

use super::resolver::CLAUDE_CODE;

/// One model of an instance and what it can do.
#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    /// Model identifier.
    pub id: String,
    /// Alias (`fast`, `default`, `deep`, `utility`) that points at it, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    /// Capabilities of THIS model (A4).
    pub capabilities: Value,
}

impl ModelEntry {
    /// A model entry from nexus `Capabilities`.
    pub fn new(id: impl Into<String>, alias: Option<String>, caps: &Capabilities) -> Self {
        Self {
            id: id.into(),
            alias,
            capabilities: serde_json::to_value(caps).unwrap_or(Value::Null),
        }
    }
}

/// Health of an instance, in the vocabulary of the frontend.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct HealthEntry {
    /// `ok | auth_required | unreachable | cli_not_found | unknown`.
    pub state: &'static str,
    /// Error code (`cli_not_found`, `auth_required`, ...) when not ok.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<&'static str>,
    /// Command a human should run (never run by PO, A27).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// RFC 3339 time of the check.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checked_at: Option<String>,
    /// A fixed readable sentence saying why the instance cannot be used (set for
    /// a remote machine only: "the host key does not match the pinned key"...).
    /// Never the raw output of a client, never a path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl HealthEntry {
    /// Health nobody measured yet.
    pub fn unknown() -> Self {
        Self {
            state: "unknown",
            code: None,
            action: None,
            checked_at: None,
            reason: None,
        }
    }

    /// The reason a REMOTE machine is unavailable, from what nexus reported.
    /// Nexus answers fixed sentences; anything that looks like a path is dropped
    /// anyway (defence in depth: an identity file path must never leave).
    pub fn with_remote_reason(mut self, health: &ProviderHealth) -> Self {
        if health.status != HealthStatus::Unavailable {
            return self;
        }
        let text = match &health.error {
            Some(ProviderError::EndpointUnreachable { detail }) => detail.clone(),
            Some(ProviderError::CliNotFound { program }) => {
                format!("the CLI is not installed on the machine ({program})")
            }
            Some(ProviderError::CredentialsLocked) => {
                "the vault is locked: the SSH key cannot be read".to_string()
            }
            Some(ProviderError::AuthRequired { .. }) => {
                "the SSH key is not granted to this instance in the vault".to_string()
            }
            _ => return self,
        };
        self.reason = Some(if text.contains(['/', '\\']) || text.len() > 300 {
            "the machine cannot be used".to_string()
        } else {
            text
        });
        self
    }

    /// From the health nexus reports.
    pub fn from_nexus(health: &ProviderHealth) -> Self {
        let checked_at = (health.checked_at_ms > 0)
            .then(|| chrono::DateTime::from_timestamp_millis(health.checked_at_ms as i64))
            .flatten()
            .map(|t| t.to_rfc3339());
        let (state, code) = match (&health.status, &health.error) {
            (HealthStatus::Ok, _) => ("ok", None),
            (HealthStatus::Degraded, _) => ("ok", None),
            (_, Some(ProviderError::CliNotFound { .. })) => {
                ("cli_not_found", Some("cli_not_found"))
            }
            (_, Some(ProviderError::AuthRequired { .. })) => {
                ("auth_required", Some("auth_required"))
            }
            (_, Some(ProviderError::Unauthorized)) => ("auth_required", Some("unauthorized")),
            (_, Some(ProviderError::CredentialsLocked)) => {
                ("auth_required", Some("credentials_locked"))
            }
            (_, Some(ProviderError::EndpointUnreachable { .. })) => {
                ("unreachable", Some("endpoint_unreachable"))
            }
            _ => ("unknown", None),
        };
        Self {
            state,
            code,
            action: health.login_hint.clone(),
            checked_at,
            reason: None,
        }
    }
}

/// One provider instance of the listing.
#[derive(Debug, Clone, Serialize)]
pub struct ProviderEntry {
    /// Instance identifier.
    pub id: String,
    /// `claude_code | openai_compatible | codex | acp | claude_code_remote`.
    pub kind: &'static str,
    /// Display label.
    pub label: String,
    /// Built into PO (not removable).
    pub builtin: bool,
    /// Is the configured default.
    pub is_default: bool,
    /// Consent given for the asked project; `null` when no project was asked.
    pub allowed_for_project: Option<bool>,
    /// Origin (scheme, host, port) of the endpoint, never a path or a credential.
    pub endpoint_origin: Option<String>,
    /// Credential REFERENCE (`vault:<name>`, `env:<VAR>`, `none`), never the value.
    pub credential: String,
    /// Health.
    pub health: HealthEntry,
    /// Models and their capabilities.
    pub models: Vec<ModelEntry>,
    /// What the instance does whatever the model (permission prompts, sandbox,
    /// live model switch...), as its provider declares it. Used by the interface
    /// when a model carries none, and for an instance that lists no model yet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Value>,
    /// Where a `claude_code_remote` instance runs; absent for every other kind.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote: Option<RemoteEntry>,
}

/// The machine of a `claude_code_remote` instance, as the listing shows it:
/// the pinned key's FINGERPRINT (public), never the key file or its vault name.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RemoteEntry {
    /// Host name or address.
    pub host: String,
    /// Remote user, when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssh_user: Option<String>,
    /// ssh port (22 when none was set).
    pub ssh_port: u16,
    /// `SHA256:...` fingerprint of the pinned host key.
    pub host_key_fingerprint: Option<String>,
    /// Working directory on the machine.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_cwd: Option<String>,
    /// The no-confirmation mode is allowed on that machine.
    pub allow_trust: bool,
}

/// The listing.
#[derive(Debug, Clone, Serialize)]
pub struct ProviderListing {
    /// Identifier of the default instance, or `null`.
    pub default_provider: Option<String>,
    /// The instances, the built-in one first.
    pub providers: Vec<ProviderEntry>,
}

/// The built-in Claude Code instance. Always consented (it is the historical
/// path) and always listed first.
pub fn builtin_claude_code(
    health: HealthEntry,
    models: Vec<ModelEntry>,
    asked_project: bool,
) -> ProviderEntry {
    ProviderEntry {
        id: CLAUDE_CODE.to_string(),
        kind: "claude_code",
        label: "Claude Code".to_string(),
        builtin: true,
        is_default: false,
        allowed_for_project: asked_project.then_some(true),
        endpoint_origin: None,
        credential: "none".to_string(),
        health,
        models,
        capabilities: None,
        remote: None,
    }
}

/// Assembles the listing; the default is the first instance when none is
/// configured (Claude Code, A16: "claude-code when healthy").
pub fn assemble(
    mut providers: Vec<ProviderEntry>,
    configured_default: Option<&str>,
) -> ProviderListing {
    let default = configured_default
        .filter(|d| providers.iter().any(|p| p.id == *d))
        .map(str::to_string)
        .or_else(|| {
            providers
                .iter()
                .find(|p| p.id == CLAUDE_CODE)
                .map(|p| p.id.clone())
        });
    for p in &mut providers {
        p.is_default = default.as_deref() == Some(p.id.as_str());
    }
    ProviderListing {
        default_provider: default,
        providers,
    }
}

pub use super::endpoint_guard::origin_of;

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> Capabilities {
        Capabilities::none()
    }

    #[test]
    fn the_builtin_instance_is_the_default_when_nothing_is_configured() {
        let l = assemble(
            vec![builtin_claude_code(HealthEntry::unknown(), vec![], false)],
            None,
        );
        assert_eq!(l.default_provider.as_deref(), Some("claude-code"));
        assert!(l.providers[0].is_default);
        assert_eq!(l.providers[0].allowed_for_project, None);
    }

    #[test]
    fn a_configured_default_that_does_not_exist_is_ignored() {
        let l = assemble(
            vec![builtin_claude_code(HealthEntry::unknown(), vec![], true)],
            Some("ghost"),
        );
        assert_eq!(l.default_provider.as_deref(), Some("claude-code"));
        assert_eq!(l.providers[0].allowed_for_project, Some(true));
    }

    #[test]
    fn health_maps_to_the_frontend_states() {
        let ok = HealthEntry::from_nexus(&ProviderHealth::ok(Some("2.1".into())));
        assert_eq!((ok.state, ok.code), ("ok", None));
        let missing =
            HealthEntry::from_nexus(&ProviderHealth::unavailable(ProviderError::CliNotFound {
                program: "/secret/path/claude".into(),
            }));
        assert_eq!(
            (missing.state, missing.code),
            ("cli_not_found", Some("cli_not_found"))
        );
        let body = serde_json::to_string(&missing).unwrap();
        assert!(!body.contains("/secret/path"), "{body}");
        let auth =
            HealthEntry::from_nexus(&ProviderHealth::unavailable(ProviderError::AuthRequired {
                login_hint: Some("claude login".into()),
            }));
        assert_eq!(auth.state, "auth_required");
    }

    #[test]
    fn an_endpoint_origin_never_carries_userinfo_path_or_query() {
        assert_eq!(
            origin_of("https://user:hunter2@api.example.com:8443/v1/chat?key=abc").as_deref(),
            Some("https://api.example.com:8443")
        );
        assert_eq!(
            origin_of("http://localhost:8080/v1").as_deref(),
            Some("http://localhost:8080")
        );
        assert_eq!(origin_of("not a url"), None);
    }

    #[test]
    fn the_listing_body_has_the_documented_field_names() {
        let entry = builtin_claude_code(
            HealthEntry::unknown(),
            vec![ModelEntry::new(
                "claude-sonnet-5",
                Some("default".into()),
                &caps(),
            )],
            true,
        );
        let v = serde_json::to_value(assemble(vec![entry], None)).unwrap();
        let p = &v["providers"][0];
        for k in [
            "id",
            "kind",
            "label",
            "builtin",
            "is_default",
            "allowed_for_project",
            "endpoint_origin",
            "credential",
            "health",
            "models",
        ] {
            assert!(p.get(k).is_some(), "missing {k}");
        }
        assert_eq!(p["models"][0]["alias"], "default");
        assert!(p["models"][0]["capabilities"].is_object());
        assert_eq!(v["default_provider"], "claude-code");
    }

    #[test]
    fn a_remote_reason_is_a_fixed_sentence_and_a_path_never_leaves() {
        let unreachable = |detail: &str| {
            let h = ProviderHealth::unavailable(ProviderError::unreachable(detail));
            HealthEntry::from_nexus(&h).with_remote_reason(&h)
        };
        let ok = unreachable("deploy@box:22: the host key does not match the pinned key");
        assert_eq!(
            ok.reason.as_deref(),
            Some("deploy@box:22: the host key does not match the pinned key")
        );
        // Whatever a lower layer puts in the text, a path is replaced.
        let leaked = unreachable("identity file /tmp/nexus-1234/id_ssh must be owner-only");
        assert_eq!(leaked.reason.as_deref(), Some("the machine cannot be used"));
        let long = unreachable(&"x".repeat(400));
        assert_eq!(long.reason.as_deref(), Some("the machine cannot be used"));
        // A healthy instance has no reason, and no other kind gets one from the plain mapping.
        let h = ProviderHealth::ok(None);
        assert!(HealthEntry::from_nexus(&h)
            .with_remote_reason(&h)
            .reason
            .is_none());
        assert!(
            HealthEntry::from_nexus(&ProviderHealth::unavailable(ProviderError::unreachable(
                "x"
            )))
            .reason
            .is_none()
        );
        let json = serde_json::to_value(HealthEntry::unknown()).unwrap();
        assert!(json.get("reason").is_none(), "additive: absent unless set");
    }
}
