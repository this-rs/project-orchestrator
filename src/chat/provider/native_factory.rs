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
    match record.kind.as_str() {
        "codex" => build_codex(record, vault),
        "acp" => build_acp(record),
        _ => Ok(build_native(record, vault)? as Arc<dyn AgentProvider>),
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
        }
    }

    #[test]
    fn a_stored_instance_becomes_a_native_provider() {
        let p = build_native_provider(&record("none", Some("llama_server")), None).unwrap();
        assert_eq!(p.id(), "local");
        assert_eq!(p.kind(), nexus_claude::agent::ProviderKind::Native);
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
}
