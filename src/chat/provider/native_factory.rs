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

/// The native provider of a stored instance.
pub fn build_native_provider(
    record: &InstanceRecord,
    vault: Option<Arc<VaultService>>,
) -> Result<Arc<dyn AgentProvider>, ProviderError> {
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
}
