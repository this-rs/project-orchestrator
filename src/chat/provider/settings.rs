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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstanceDraft {
    /// Instance identifier (slug).
    pub id: Option<String>,
    /// Kind: `openai_compatible` (default), `codex` or `acp`.
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
}

impl InstanceRecord {
    /// OpenSSH fingerprint (`SHA256:...`) of the pinned host key, when there is one.
    pub fn host_key_fingerprint(&self) -> Option<String> {
        self.host_key.as_deref().and_then(host_key_fingerprint)
    }
}

/// The kind of an instance that runs Claude Code on another machine over SSH.
pub const KIND_CLAUDE_CODE_REMOTE: &str = "claude_code_remote";

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

/// Environment variable declaring the ACP agents an instance may launch: a
/// JSON object, `{"opencode": ["opencode", "acp"]}`. EMPTY BY DEFAULT: an API
/// body never carries a command line (that would be remote code execution as
/// the server's user); it names a declared one by `preset`.
pub const ACP_COMMANDS_VAR: &str = "CHAT_PROVIDER_ACP_COMMANDS";

/// The declared ACP commands.
pub fn acp_commands() -> std::collections::BTreeMap<String, Vec<String>> {
    parse_acp_commands(&std::env::var(ACP_COMMANDS_VAR).unwrap_or_default())
}

/// Parses the declaration; anything malformed declares nothing.
pub fn parse_acp_commands(raw: &str) -> std::collections::BTreeMap<String, Vec<String>> {
    serde_json::from_str::<std::collections::BTreeMap<String, Vec<String>>>(raw)
        .unwrap_or_default()
        .into_iter()
        .filter(|(name, argv)| valid_id(name) && !argv.is_empty() && !argv[0].trim().is_empty())
        .collect()
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
        "openai_compatible" | "codex" | "acp" | KIND_CLAUDE_CODE_REMOTE
    ) {
        return Err(invalid(
            "kind: openai_compatible, codex, acp or claude_code_remote",
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
        } else {
            parse_credential_ref(cred, env_allow)?;
        }
        next.credential_ref = cred.clone();
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
}
