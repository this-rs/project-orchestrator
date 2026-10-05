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
/// Key prefix of consent documents (scope `project:<slug>`).
pub const CONSENT_PREFIX: &str = "consent:";
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstanceDraft {
    /// Instance identifier (slug).
    pub id: Option<String>,
    /// Kind; only `openai_compatible` can be created today.
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
}

/// Body of `PUT|PATCH /chat/providers/{id}`: the id and the kind never change.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

/// A stored provider instance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
}

const COST_SOURCES: [&str; 5] = ["reported", "priced", "free", "subscription", "unknown"];

fn valid_id(id: &str) -> bool {
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

/// Validates a draft into a record (syntax only; the DNS check is async and
/// done by the caller through `validate_endpoint`).
pub fn record_from_draft(
    draft: &InstanceDraft,
    policy: &EndpointPolicy,
    env_allow: &[String],
) -> Result<InstanceRecord, SettingsError> {
    let id = draft.id.clone().ok_or_else(|| invalid("id is required"))?;
    if id == CLAUDE_CODE {
        return Err(SettingsError::Builtin);
    }
    if !valid_id(&id) {
        return Err(invalid("id: lowercase letters, digits and dashes only"));
    }
    let kind = draft
        .kind
        .clone()
        .unwrap_or_else(|| "openai_compatible".into());
    if kind != "openai_compatible" {
        return Err(invalid(
            "kind: only openai_compatible instances can be created",
        ));
    }
    let base_url = draft
        .base_url
        .clone()
        .filter(|u| !u.trim().is_empty())
        .ok_or_else(|| invalid("base_url is required"))?;
    let origin = check_url(&base_url, policy)?;
    let credential_ref = draft
        .credential_ref
        .clone()
        .unwrap_or_else(|| "none".into());
    parse_credential_ref(&credential_ref, env_allow)?;
    let cost_source = draft
        .cost_source
        .clone()
        .unwrap_or_else(|| "unknown".into());
    if !COST_SOURCES.contains(&cost_source.as_str()) {
        return Err(invalid(
            "cost_source: reported, priced, free, subscription or unknown",
        ));
    }
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
    })
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
        next.preset = patch.preset.clone();
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
        parse_credential_ref(cred, env_allow)?;
        next.credential_ref = cred.clone();
    }
    // A consent is tied to the origin AND to the credential reference: changing
    // either sends the project's content somewhere or with something it did not
    // agree to.
    let origin_changed = next.origin != old.origin || next.credential_ref != old.credential_ref;
    Ok((next, origin_changed))
}

/// A stored consent: who allowed which origin for a project, and when.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
}

/// Whether a consent still holds for the instance as it is now: same origin
/// AND same credential reference (A28).
pub fn consent_holds(consent: &ConsentRecord, instance: &InstanceRecord) -> bool {
    consent.provider_id == instance.id
        && consent.origin == instance.origin
        && consent.credential_ref.as_deref().unwrap_or("none") == instance.credential_ref
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
        };
        assert!(consent_view(&consent, Some(&old)).valid);
        let patch = InstancePatch {
            preset: None,
            label: None,
            base_url: Some("https://other.example.com/v1".into()),
            default_model: None,
            cost_source: None,
            credential_ref: None,
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
        };
        assert!(consent_holds(&consent, &old));
        let patch = InstancePatch {
            preset: None,
            label: None,
            base_url: None,
            default_model: None,
            cost_source: None,
            credential_ref: Some("vault:another-key".into()),
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
}
