//! Builds the native harness (nexus `NativeProvider` on an OpenAI-compatible
//! endpoint) for a STORED instance.
//!
//! Nothing secret is read here: the endpoint receives a credential REFERENCE
//! and a resolver; the key is read per request (`read_for_provider` under a
//! grant for that instance) and never kept. No fallback exists: a locked vault
//! is `credentials_locked`, a missing grant `auth_required`.

use std::str::FromStr;
use std::sync::Arc;

use nexus_claude::agent::{
    AgentProvider, CostBasis, CredentialRef, CredentialResolver, EnvCredentialResolver,
    ProviderError,
};
use nexus_claude::model::{EndpointQuirks, OpenAiEndpoint, OpenAiEndpointConfig};
use nexus_claude::providers::native::{NativeConfig, NativeProvider};

use super::credentials::VaultCredentialResolver;
use super::settings::InstanceRecord;
use crate::vault::VaultService;

/// What a connection test found out about an instance (A30): can it be
/// reached, which models it lists, can the chosen model call a tool, and how
/// large its context window is.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeReport {
    /// Health as nexus reports it.
    pub health: nexus_claude::agent::ProviderHealth,
    /// Models the endpoint lists.
    pub models: Vec<String>,
    /// Model that was probed, if any.
    pub model: Option<String>,
    /// Whether that model called a tool.
    pub tools: Option<bool>,
    /// Its context window, when known.
    pub context_window: Option<u64>,
    /// Why the tool probe failed (typed), when it did.
    pub probe_error: Option<ProviderError>,
}

/// Connection test of an instance, run BEFORE it is saved or used: health, the
/// model listing, and a real tool-call probe of one model. Reads nothing but
/// the credential reference's secret, per request, through the resolver.
pub async fn probe_instance(
    record: &InstanceRecord,
    vault: Option<Arc<VaultService>>,
    model: Option<&str>,
) -> Result<ProbeReport, ProviderError> {
    let provider = build_native(record, vault)?;
    let health = provider.health().await;
    if health.status == nexus_claude::agent::HealthStatus::Unavailable {
        return Ok(ProbeReport {
            health,
            models: Vec::new(),
            model: None,
            tools: None,
            context_window: None,
            probe_error: None,
        });
    }
    let models: Vec<String> = provider
        .catalog()
        .await
        .map(|c| c.into_iter().map(|m| m.id).collect())
        .unwrap_or_default();
    let chosen = model
        .map(str::to_string)
        .or_else(|| record.default_model.clone())
        .or_else(|| models.first().cloned());
    let (mut tools, mut context_window, mut probe_error) = (None, None, None);
    if let Some(m) = chosen.as_deref() {
        match provider.refresh_capabilities(m).await {
            Ok(caps) => {
                tools = Some(caps.tools);
                context_window = caps.context_window.map(|w| w.value);
            }
            Err(e) => probe_error = Some(e),
        }
    }
    Ok(ProbeReport {
        health,
        models,
        model: chosen,
        tools,
        context_window,
        probe_error,
    })
}

/// Models an OpenAI-compatible instance lists (`GET /models`), in the order the
/// endpoint gives them. Nothing is stored; the key is read for this request only.
pub async fn list_models(
    record: &InstanceRecord,
    vault: Option<Arc<VaultService>>,
) -> Result<Vec<String>, ProviderError> {
    let provider = build_native(record, vault)?;
    Ok(provider
        .catalog()
        .await?
        .into_iter()
        .map(|m| m.id)
        .collect())
}

/// Capabilities an instance declares without touching the network: its provider
/// is built from the stored record and asked (`capabilities` is static, and
/// `tools` stays false until a probe has run). A remote Claude Code, or a record
/// that does not build, declares none.
pub fn declared_capabilities(record: &InstanceRecord) -> nexus_claude::agent::Capabilities {
    if record.kind == super::settings::KIND_CLAUDE_CODE_REMOTE {
        return nexus_claude::agent::Capabilities::none();
    }
    build_native_provider(record, None)
        .map(|p| p.capabilities(record.default_model.as_deref()))
        .unwrap_or_else(|_| nexus_claude::agent::Capabilities::none())
}

fn build_native(
    record: &InstanceRecord,
    vault: Option<Arc<VaultService>>,
) -> Result<Arc<NativeProvider>, ProviderError> {
    // A stored record is checked again here: an `env:` reference must name a
    // variable declared for provider credentials, whenever it was stored.
    super::settings::parse_credential_ref(
        &record.credential_ref,
        &super::settings::env_credential_allowlist(),
    )
    .map_err(|_| ProviderError::invalid("credential reference not allowed"))?;
    let credential = CredentialRef::from_str(&record.credential_ref)?;
    let mut endpoint = OpenAiEndpointConfig::new(record.id.clone(), record.base_url.clone());
    endpoint.credential = credential;
    endpoint.quirks = record
        .preset
        .as_deref()
        .and_then(EndpointQuirks::preset)
        .unwrap_or_else(EndpointQuirks::generic);
    // A private-range endpoint needs an operator decision that this API does
    // not offer: the strict default stays.
    endpoint.allow_private_network = false;

    let resolver: Arc<dyn CredentialResolver> = match vault {
        Some(vault) => Arc::new(VaultCredentialResolver::new(vault)),
        None => Arc::new(EnvCredentialResolver),
    };
    let model_endpoint = Arc::new(OpenAiEndpoint::new(endpoint, resolver));

    let mut config = NativeConfig::new(record.id.clone());
    config.default_model = record.default_model.clone();
    // Only "free" is known without a price table; the rest stays `unknown`
    // (no amount) until a price is configured (A21: never an invented price).
    if record.cost_source == "free" {
        config.cost_basis = CostBasis::Free;
    }
    Ok(Arc::new(NativeProvider::new(config, model_endpoint)))
}

/// The provider of a stored instance, by kind: the native harness over an
/// OpenAI-compatible endpoint, Codex (`app-server`) or a declared ACP agent.
pub fn build_native_provider(
    record: &InstanceRecord,
    vault: Option<Arc<VaultService>>,
) -> Result<Arc<dyn AgentProvider>, ProviderError> {
    build_provider_with_handle(record, vault).map(|(provider, _)| provider)
}

/// A provider and, for a native record, its concrete harness.
pub type BuiltProvider = (Arc<dyn AgentProvider>, Option<Arc<NativeProvider>>);

/// Same, also returning the concrete native harness when the record is one: only
/// it can run a capability probe (the trait has no such method).
pub fn build_provider_with_handle(
    record: &InstanceRecord,
    vault: Option<Arc<VaultService>>,
) -> Result<BuiltProvider, ProviderError> {
    match record.kind.as_str() {
        "codex" => build_codex(record, vault).map(|p| (p, None)),
        "acp" => build_acp(record).map(|p| (p, None)),
        super::settings::KIND_CLAUDE_CODE_REMOTE => {
            build_remote_claude(record, vault).map(|p| (p, None))
        }
        _ => {
            let native = build_native(record, vault)?;
            Ok((Arc::clone(&native) as Arc<dyn AgentProvider>, Some(native)))
        }
    }
}

fn cost_basis_of(record: &InstanceRecord) -> CostBasis {
    // Only "free" is known without a price table; the rest stays `unknown`.
    if record.cost_source == "free" {
        CostBasis::Free
    } else {
        CostBasis::Unknown
    }
}

/// Codex: the program is `codex` found on the server's PATH (never a path from an
/// API body), with its own persistent home per instance.
fn build_codex(
    record: &InstanceRecord,
    vault: Option<Arc<VaultService>>,
) -> Result<Arc<dyn AgentProvider>, ProviderError> {
    use nexus_claude::providers::codex::{CodexConfig, CodexProvider};
    super::settings::parse_credential_ref(
        &record.credential_ref,
        &super::settings::env_credential_allowlist(),
    )
    .map_err(|_| ProviderError::invalid("credential reference not allowed"))?;
    let mut config = CodexConfig::new(record.id.clone());
    config.default_model = record.default_model.clone();
    config.cost_basis = cost_basis_of(record);
    config.credential = CredentialRef::from_str(&record.credential_ref)?;
    let resolver: Arc<dyn CredentialResolver> = match vault {
        Some(vault) => Arc::new(VaultCredentialResolver::new(vault)),
        None => Arc::new(EnvCredentialResolver),
    };
    Ok(Arc::new(CodexProvider::with_resolver(config, resolver)))
}

/// ACP: the command is the one DECLARED on the server under the instance's
/// preset; the agent holds its own login (no credential).
fn build_acp(record: &InstanceRecord) -> Result<Arc<dyn AgentProvider>, ProviderError> {
    use nexus_claude::providers::acp::{AcpConfig, AcpProvider};
    let name = record
        .preset
        .as_deref()
        .ok_or_else(|| ProviderError::invalid("an ACP instance names a declared agent"))?;
    let command = super::settings::acp_commands()
        .remove(name)
        .ok_or_else(|| ProviderError::invalid("that ACP agent is not declared on this server"))?;
    let mut config = AcpConfig::new(record.id.clone(), command);
    config.default_model = record.default_model.clone();
    config.cost_basis = cost_basis_of(record);
    config.validate()?;
    Ok(Arc::new(AcpProvider::new(config)))
}

/// Claude Code on another machine over SSH (nexus `ClaudeCodeConfig::remote`).
///
/// The record names a vault entry holding the SSH PRIVATE key. Nothing is read
/// here: the key is read, and written to an owner-only file, when a session (or a
/// health probe) needs it. See [`RemoteClaudeProvider`] for the lifetime.
fn build_remote_claude(
    record: &InstanceRecord,
    vault: Option<Arc<VaultService>>,
) -> Result<Arc<dyn AgentProvider>, ProviderError> {
    use nexus_claude::providers::claude_code::ClaudeCodeConfig;
    use nexus_claude::transport::RemoteHost;
    // The stored record is checked again: `vault:<name>` only, whenever it was stored.
    let key_name = match super::settings::parse_credential_ref(&record.credential_ref, &[]) {
        Ok(super::settings::CredentialSource::Vault(name)) => name,
        _ => {
            return Err(ProviderError::invalid(
                "a remote instance needs a vault credential",
            ))
        }
    };
    let (Some(host), Some(host_key)) = (record.host.clone(), record.host_key.clone()) else {
        return Err(ProviderError::invalid(
            "a remote instance needs a host and a host key",
        ));
    };
    let mut remote = RemoteHost::new(host, host_key);
    remote.user = record.ssh_user.clone();
    remote.port = record.ssh_port;
    remote.cwd = record.remote_cwd.clone();
    remote.allow_trust = record.allow_trust;
    remote
        .validate()
        .map_err(|e| ProviderError::invalid(e.to_string()))?;
    let mut config = ClaudeCodeConfig::default();
    config.id = record.id.clone();
    config.default_model = record.default_model.clone();
    config.cost_basis = match record.cost_source.as_str() {
        "free" => CostBasis::Free,
        "subscription" => CostBasis::Subscription,
        "reported" => CostBasis::Reported,
        _ => CostBasis::Unknown,
    };
    config.remote = Some(remote);
    Ok(Arc::new(RemoteClaudeProvider {
        config,
        key_name,
        vault,
    }))
}

/// A Claude Code instance on another machine.
///
/// Lifetime of the SSH key, chosen on purpose: the provider is cached per record
/// (`ChatManager::provider_for`) for as long as the server runs, so a key file
/// owned by the provider would sit on disk for the whole uptime, and survive a
/// vault lock. Instead the key is read from the vault, under the grant
/// `Provider(<instance id>)`, at the moment it is needed (`health`, `open`,
/// `resume`), written to an owner-only `0600` file in a fresh `0700` directory
/// (nexus `SecretFile`), and the file is owned by what uses it: the probe, or
/// the session ([`KeyedSession`]). It is removed when that is dropped. A locked
/// vault or a missing grant is a typed error at that moment; there is no
/// fallback to another provider or to the local CLI.
struct RemoteClaudeProvider {
    /// Config of the instance; `remote.identity_file` is always `None` here.
    config: nexus_claude::providers::claude_code::ClaudeCodeConfig,
    /// Name of the vault entry holding the private key.
    key_name: String,
    vault: Option<Arc<VaultService>>,
}

impl RemoteClaudeProvider {
    /// A provider whose `remote.identity_file` points at a fresh file holding the
    /// key, and the file itself (keep it alive as long as the connection needs it).
    fn with_key(
        &self,
    ) -> Result<
        (
            nexus_claude::providers::claude_code::ClaudeCodeProvider,
            nexus_claude::transport::SecretFile,
        ),
        ProviderError,
    > {
        use nexus_claude::providers::claude_code::ClaudeCodeProvider;
        let vault = self.vault.clone().ok_or_else(|| {
            ProviderError::invalid("the vault is not available: the SSH key lives there")
        })?;
        let secret = VaultCredentialResolver::new(vault)
            .read_vault(&self.config.id, &self.key_name)?
            .ok_or(ProviderError::CredentialsLocked)?;
        let mut pem = secret.expose().to_string();
        // OpenSSH refuses a private key file without its final newline.
        if !pem.ends_with('\n') {
            pem.push('\n');
        }
        let file = nexus_claude::transport::SecretFile::create("id_ssh", pem.as_bytes())
            .map_err(|_| ProviderError::invalid("the SSH key file could not be prepared"))?;
        let mut config = self.config.clone();
        if let Some(remote) = config.remote.as_mut() {
            remote.identity_file = Some(file.path().to_path_buf());
        }
        Ok((ClaudeCodeProvider::new(config), file))
    }
}

#[async_trait::async_trait]
impl AgentProvider for RemoteClaudeProvider {
    fn id(&self) -> &str {
        &self.config.id
    }
    fn kind(&self) -> nexus_claude::agent::ProviderKind {
        nexus_claude::agent::ProviderKind::ClaudeCode
    }
    async fn health(&self) -> nexus_claude::agent::ProviderHealth {
        match self.with_key() {
            Ok((provider, _key)) => provider.health().await,
            Err(e) => nexus_claude::agent::ProviderHealth::unavailable(e),
        }
    }
    async fn catalog(&self) -> Result<Vec<nexus_claude::agent::ModelInfo>, ProviderError> {
        Ok(self.config.models.clone())
    }
    fn capabilities(&self, model: Option<&str>) -> nexus_claude::agent::Capabilities {
        self.config.capabilities(model)
    }
    async fn open(
        &self,
        spec: nexus_claude::agent::SessionSpec,
    ) -> Result<Arc<dyn nexus_claude::agent::AgentSession>, ProviderError> {
        let (provider, key) = self.with_key()?;
        let inner = provider.open(spec).await?;
        Ok(Arc::new(KeyedSession { inner, _key: key }))
    }
    async fn resume(
        &self,
        spec: nexus_claude::agent::SessionSpec,
        token: nexus_claude::agent::ResumeToken,
    ) -> Result<Arc<dyn nexus_claude::agent::AgentSession>, ProviderError> {
        let (provider, key) = self.with_key()?;
        let inner = provider.resume(spec, token).await?;
        Ok(Arc::new(KeyedSession { inner, _key: key }))
    }
}

/// A session that owns the key file its connection was made with: the file goes
/// when the session is dropped, never earlier (a reconnection would need it).
struct KeyedSession {
    inner: Arc<dyn nexus_claude::agent::AgentSession>,
    _key: nexus_claude::transport::SecretFile,
}

#[async_trait::async_trait]
impl nexus_claude::agent::AgentSession for KeyedSession {
    fn capabilities(&self) -> &nexus_claude::agent::Capabilities {
        self.inner.capabilities()
    }
    fn resume_token(&self) -> Option<nexus_claude::agent::ResumeToken> {
        self.inner.resume_token()
    }
    async fn send_turn(
        &self,
        input: nexus_claude::agent::TurnInput,
    ) -> Result<nexus_claude::agent::EventStream, ProviderError> {
        self.inner.send_turn(input).await
    }
    async fn answer_permission(
        &self,
        request_id: &str,
        decision: nexus_claude::agent::PermissionDecision,
    ) -> Result<(), ProviderError> {
        self.inner.answer_permission(request_id, decision).await
    }
    async fn answer_question(
        &self,
        question_id: &str,
        answer: nexus_claude::agent::QuestionAnswer,
    ) -> Result<(), ProviderError> {
        self.inner.answer_question(question_id, answer).await
    }
    async fn interrupt(
        &self,
        scope: nexus_claude::agent::InterruptScope,
    ) -> Result<nexus_claude::agent::InterruptOutcome, ProviderError> {
        self.inner.interrupt(scope).await
    }
    async fn cancel_tools(
        &self,
        scope: nexus_claude::agent::CancelScope,
    ) -> Result<nexus_claude::agent::CancelOutcome, ProviderError> {
        self.inner.cancel_tools(scope).await
    }
    async fn set_model(&self, model: &str) -> Result<(), ProviderError> {
        self.inner.set_model(model).await
    }
    async fn set_policy_mode(
        &self,
        mode: nexus_claude::agent::PolicyMode,
        native: Option<&str>,
    ) -> Result<(), ProviderError> {
        self.inner.set_policy_mode(mode, native).await
    }
    fn out_of_band(&self) -> Option<nexus_claude::agent::EventStream> {
        self.inner.out_of_band()
    }
    async fn close(&self) -> Result<(), ProviderError> {
        self.inner.close().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(credential_ref: &str, preset: Option<&str>) -> InstanceRecord {
        InstanceRecord {
            id: "local".into(),
            kind: "openai_compatible".into(),
            preset: preset.map(str::to_string),
            label: "Local".into(),
            base_url: "http://127.0.0.1:9/v1".into(),
            origin: "http://127.0.0.1:9".into(),
            default_model: Some("m".into()),
            cost_source: "free".into(),
            credential_ref: credential_ref.into(),
            ..Default::default()
        }
    }

    #[test]
    fn a_stored_instance_becomes_a_native_provider() {
        let p = build_native_provider(&record("none", Some("llama_server")), None).unwrap();
        assert_eq!(p.id(), "local");
        assert_eq!(p.kind(), nexus_claude::agent::ProviderKind::Native);
    }

    #[test]
    fn a_native_instance_declares_permission_prompts_and_no_sandbox_before_any_session() {
        // Listed with EMPTY capabilities, the interface said such an instance
        // "cannot pause a tool call to ask you" (policy only) before the first message.
        let caps = declared_capabilities(&record("none", Some("deepseek")));
        assert!(caps.interactive_permissions);
        assert!(caps.set_model_live);
        assert_eq!(caps.sandbox, nexus_claude::agent::SandboxLevel::None);
        // Nothing is claimed about tools before a probe has run.
        assert!(!caps.tools);
    }

    #[test]
    fn a_record_that_does_not_build_declares_nothing() {
        let caps = declared_capabilities(&record("sk-live-abcdef", None));
        assert_eq!(caps, nexus_claude::agent::Capabilities::none());
    }

    #[test]
    fn a_pasted_key_is_refused_without_being_echoed() {
        let err = build_native_provider(&record("sk-live-abcdef", None), None)
            .err()
            .expect("refused");
        assert!(!format!("{err:?}{err}").contains("sk-live"));
    }

    #[test]
    fn codex_and_acp_instances_are_built_without_any_command_from_the_record() {
        let mut codex = record("none", None);
        codex.kind = "codex".into();
        codex.base_url = String::new();
        codex.origin = "process:codex".into();
        let p = build_native_provider(&codex, None).unwrap();
        assert_eq!(p.kind(), nexus_claude::agent::ProviderKind::Codex);
        // ACP with nothing declared on the server: refused, there is no command to run.
        let mut acp = record("none", Some("opencode"));
        acp.kind = "acp".into();
        let err = build_native_provider(&acp, None).err().expect("refused");
        assert_eq!(err.kind(), "invalid_request");
    }

    // ── list_models ───────────────────────────────────────────────────────

    async fn endpoint_listing(body: serde_json::Value, status: u16) -> wiremock::MockServer {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .mount(&server)
            .await;
        server
    }

    #[tokio::test]
    async fn list_models_returns_the_ids_the_endpoint_gives_in_its_order() {
        let server = endpoint_listing(
            serde_json::json!({"object": "list", "data": [{"id": "zeta"}, {"id": "alpha"}]}),
            200,
        )
        .await;
        let mut r = record("none", Some("llama_server"));
        r.base_url = format!("{}/v1", server.uri());
        let ids = list_models(&r, None).await.expect("a listing");
        assert_eq!(ids, ["zeta", "alpha"]);
    }

    #[tokio::test]
    async fn list_models_reports_an_endpoint_that_refuses() {
        let server = endpoint_listing(serde_json::json!({"error": "no"}), 500).await;
        let mut r = record("none", Some("llama_server"));
        r.base_url = format!("{}/v1", server.uri());
        assert!(list_models(&r, None).await.is_err());
    }

    #[tokio::test]
    async fn list_models_refuses_an_instance_it_cannot_build_without_echoing_a_pasted_key() {
        let err = list_models(&record("sk-live-abcdef", None), None)
            .await
            .expect_err("refused");
        assert!(!format!("{err:?}{err}").contains("sk-live"));
    }

    // ── claude_code_remote ────────────────────────────────────────────────

    const KEY: &str =
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEXTCY3J636nEyMNqNrVj6HXnIXUnXL8sk4c5J6tek3N";
    const PEM: &str = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA-test-only\n-----END OPENSSH PRIVATE KEY-----";

    fn remote_record(host: &str, port: u16) -> InstanceRecord {
        InstanceRecord {
            id: "claude-code@box".into(),
            kind: "claude_code_remote".into(),
            label: "Box".into(),
            origin: format!("ssh:{host}:{port}"),
            cost_source: "subscription".into(),
            credential_ref: "vault:ssh-box".into(),
            host: Some(host.into()),
            ssh_port: Some(port),
            host_key: Some(KEY.into()),
            remote_cwd: Some("/srv/work".into()),
            ..Default::default()
        }
    }

    async fn vault_with_key(grant_to: Option<&str>) -> Arc<VaultService> {
        use crate::vault::grants::{GrantScope, SecretSelector};
        let vault = VaultService::ephemeral();
        vault
            .init(
                "correct horse battery staple".into(),
                chrono::Duration::hours(1),
            )
            .await
            .unwrap();
        let now = chrono::Utc::now();
        vault.put("ssh-box", PEM, None, now).unwrap();
        if let Some(instance) = grant_to {
            vault
                .grant(
                    SecretSelector::Names(["ssh-box".to_string()].into()),
                    GrantScope::Provider(instance.to_string()),
                    chrono::Duration::hours(1),
                    None,
                    now,
                )
                .unwrap();
        }
        vault
    }

    fn remote_provider(vault: Option<Arc<VaultService>>) -> Arc<dyn AgentProvider> {
        build_native_provider(&remote_record("127.0.0.1", 1), vault).unwrap()
    }

    #[test]
    fn a_remote_instance_is_a_claude_code_provider_without_per_session_mcp() {
        let p = remote_provider(None);
        assert_eq!(p.id(), "claude-code@box");
        assert_eq!(p.kind(), nexus_claude::agent::ProviderKind::ClaudeCode);
        let caps = p.capabilities(None);
        assert!(!caps.per_session_mcp && !caps.tool_cancel);
    }

    #[test]
    fn a_remote_record_without_a_vault_credential_is_refused() {
        for bad in ["none", "env:HOME", "sk-pasted-secret"] {
            let mut r = remote_record("127.0.0.1", 1);
            r.credential_ref = bad.into();
            let err = build_native_provider(&r, None).err().expect("refused");
            assert_eq!(err.kind(), "invalid_request", "{bad}");
            assert!(!format!("{err:?}").contains("sk-pasted"));
        }
        let mut r = remote_record("-oProxyCommand=evil", 1);
        r.host = Some("-oProxyCommand=evil".into());
        assert!(build_native_provider(&r, None).is_err());
    }

    #[tokio::test]
    async fn the_key_file_is_owner_only_and_gone_with_its_owner() {
        use std::os::unix::fs::PermissionsExt;
        let vault = vault_with_key(Some("claude-code@box")).await;
        let provider = RemoteClaudeProvider {
            config: {
                let mut c = nexus_claude::providers::claude_code::ClaudeCodeConfig::default();
                c.id = "claude-code@box".into();
                c.remote = Some(nexus_claude::transport::RemoteHost::new("127.0.0.1", KEY));
                c
            },
            key_name: "ssh-box".into(),
            vault: Some(vault),
        };
        let (_p, file) = provider.with_key().unwrap();
        let path = file.path().to_path_buf();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let dir_mode = std::fs::metadata(path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(dir_mode, 0o700);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("-----BEGIN") && text.ends_with('\n'));
        drop(file);
        assert!(!path.exists(), "the key file outlived its owner");
        assert!(!path.parent().unwrap().exists());
    }

    #[tokio::test]
    async fn the_key_is_read_under_a_grant_and_a_missing_one_is_typed_never_a_fallback() {
        // No grant: auth_required, naming the instance.
        let health = remote_provider(Some(vault_with_key(None).await))
            .health()
            .await;
        assert_eq!(
            health.status,
            nexus_claude::agent::HealthStatus::Unavailable
        );
        assert_eq!(
            health.error.as_ref().map(|e| e.kind()),
            Some("auth_required")
        );
        // Granted to ANOTHER instance: same.
        let health = remote_provider(Some(vault_with_key(Some("claude-code@other")).await))
            .health()
            .await;
        assert_eq!(
            health.error.as_ref().map(|e| e.kind()),
            Some("auth_required")
        );
        // Locked vault.
        let vault = vault_with_key(Some("claude-code@box")).await;
        vault.lock_now();
        let health = remote_provider(Some(vault)).health().await;
        assert_eq!(
            health.error.as_ref().map(|e| e.kind()),
            Some("credentials_locked")
        );
        // No vault service at all: refused, not "no key".
        let err = remote_provider(None)
            .open(nexus_claude::agent::SessionSpec::new("/srv/work"))
            .await
            .err()
            .expect("refused");
        assert_eq!(err.kind(), "invalid_request");
    }

    #[tokio::test]
    async fn an_unreachable_machine_is_unavailable_with_a_readable_reason_and_no_path() {
        // Port 1 on loopback: refused at once (or no ssh client: also unavailable).
        let vault = vault_with_key(Some("claude-code@box")).await;
        let health = remote_provider(Some(vault)).health().await;
        assert_eq!(
            health.status,
            nexus_claude::agent::HealthStatus::Unavailable
        );
        let entry = crate::chat::provider::listing::HealthEntry::from_nexus(&health)
            .with_remote_reason(&health);
        let body = serde_json::to_string(&entry).unwrap();
        assert!(entry.reason.is_some(), "{body}");
        for leak in [
            "nexus-",
            "id_ssh",
            "/tmp",
            "/var/",
            "PRIVATE KEY",
            "known_hosts",
        ] {
            assert!(!body.contains(leak), "{leak} in {body}");
        }
        let failure = crate::chat::provider::errors::open_failure(
            health.error.as_ref().unwrap(),
            Some("claude-code@box"),
        );
        let body = format!("{failure:?}");
        for leak in ["nexus-", "id_ssh", "PRIVATE KEY", "known_hosts"] {
            assert!(!body.contains(leak), "{leak} in {body}");
        }
    }
}
