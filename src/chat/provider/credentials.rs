//! Provider credentials — the backend's [`CredentialResolver`] (decisions A11, A26).
//!
//! A provider instance never stores a key: it stores a REFERENCE (`vault:<name>`,
//! `env:<VAR>` or `none`). Nexus asks the host to resolve it at the last moment;
//! this is the host's answer.
//!
//! * `vault:<name>` — read through [`VaultService::read_for_provider`], under a
//!   grant of scope `Provider(<instance>)`. A locked vault is
//!   `credentials_locked`: the session is refused, and the caller must NOT fall
//!   back to another provider (the project may have chosen this endpoint for
//!   confidentiality).
//! * `env:<VAR>` — read from the server's environment, and registered with the
//!   masker like a vault value. The server's own secrets cannot be named.
//! * `none` — no credential.
//!
//! No value ever appears in an error: errors name the instance and what to do.

use std::sync::Arc;

use async_trait::async_trait;
use nexus_claude::agent::{CredentialRef, CredentialResolver, ProviderError, Secret};

use crate::vault::grants::Denied;
use crate::vault::service::ServiceError;
use crate::vault::VaultService;

/// Resolves provider credential references against the vault and the
/// server's environment.
#[derive(Clone)]
pub struct VaultCredentialResolver {
    vault: Arc<VaultService>,
}

impl VaultCredentialResolver {
    pub fn new(vault: Arc<VaultService>) -> Self {
        Self { vault }
    }

    fn read_vault(&self, instance: &str, name: &str) -> Result<Option<Secret>, ProviderError> {
        match self
            .vault
            .read_for_provider(name, instance, chrono::Utc::now())
        {
            Ok(value) => Ok(Some(Secret::new(value.as_str()))),
            // Locked: say so, and nothing else. No fallback is the caller's duty.
            Err(ServiceError::Denied(Denied::VaultLocked)) => Err(ProviderError::CredentialsLocked),
            // Unlocked, but this instance has no right to the secret (or it does
            // not exist): a person must act; tell them what.
            Err(ServiceError::Denied(Denied::NoGrant | Denied::UnknownSecret)) => {
                Err(ProviderError::AuthRequired {
                    login_hint: Some(format!(
                        "store the provider key in the vault and grant it to provider instance '{instance}'"
                    )),
                })
            }
            // Vault unavailable, throttled, storage error: the credential cannot
            // be read now. Never a silent "no credential".
            Err(_) => Err(ProviderError::CredentialsLocked),
        }
    }

    fn read_env(&self, var: &str) -> Result<Option<Secret>, ProviderError> {
        // An instance config pointing at the server's own secrets would send
        // the signing key or a database password to an endpoint.
        if crate::chat::manager::SERVER_ONLY_SECRETS.contains(&var) {
            return Err(ProviderError::InvalidRequest {
                detail: "this server variable cannot be used as a provider credential".to_string(),
            });
        }
        match std::env::var(var) {
            Ok(value) if !value.is_empty() => {
                // Mask it like a vault value: an endpoint may echo it back.
                self.vault.masker().register(&format!("env:{var}"), &value);
                Ok(Some(Secret::new(value)))
            }
            _ => Err(ProviderError::AuthRequired {
                login_hint: Some(format!("set the environment variable {var} on the server")),
            }),
        }
    }
}

#[async_trait]
impl CredentialResolver for VaultCredentialResolver {
    async fn resolve(
        &self,
        instance: &str,
        reference: &CredentialRef,
    ) -> Result<Option<Secret>, ProviderError> {
        match reference {
            CredentialRef::None => Ok(None),
            CredentialRef::Vault(name) => self.read_vault(instance, name),
            CredentialRef::Env(var) => self.read_env(var),
            // A reference kind this backend does not know: refuse, never "no credential".
            _ => Err(ProviderError::InvalidRequest {
                detail: "unsupported credential reference kind".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::grants::{GrantScope, SecretSelector};
    use chrono::{Duration, Utc};

    const PASS: &str = "correct horse battery staple";

    async fn resolver_with_key(
        grant_to: Option<&str>,
    ) -> (VaultCredentialResolver, Arc<VaultService>) {
        let vault = VaultService::ephemeral();
        vault.init(PASS.into(), Duration::hours(1)).await.unwrap();
        let now = Utc::now();
        vault
            .put("endpoint-key", "sk-resolver-value-4321", None, now)
            .unwrap();
        if let Some(instance) = grant_to {
            vault
                .grant(
                    SecretSelector::Names(["endpoint-key".to_string()].into()),
                    GrantScope::Provider(instance.to_string()),
                    Duration::hours(1),
                    None,
                    now,
                )
                .unwrap();
        }
        (VaultCredentialResolver::new(vault.clone()), vault)
    }

    /// `Secret` has no `Debug` (by design), so `unwrap_err` cannot be used.
    fn err_of(result: Result<Option<Secret>, ProviderError>) -> ProviderError {
        match result {
            Err(e) => e,
            Ok(_) => panic!("expected an error, got a credential"),
        }
    }

    fn vault_ref() -> CredentialRef {
        CredentialRef::Vault("endpoint-key".to_string())
    }

    #[tokio::test]
    async fn a_granted_vault_reference_resolves_and_is_masked() {
        let (resolver, vault) = resolver_with_key(Some("local-llm")).await;
        let secret = resolver
            .resolve("local-llm", &vault_ref())
            .await
            .unwrap()
            .expect("a credential");
        assert_eq!(secret.expose(), "sk-resolver-value-4321");
        assert!(!vault
            .masker()
            .mask("got sk-resolver-value-4321")
            .contains("sk-resolver-value-4321"));
    }

    #[tokio::test]
    async fn a_locked_vault_is_credentials_locked_never_no_credential() {
        let (resolver, vault) = resolver_with_key(Some("local-llm")).await;
        vault.lock_now();
        let err = err_of(resolver.resolve("local-llm", &vault_ref()).await);
        assert_eq!(err, ProviderError::CredentialsLocked);
        assert_eq!(err.kind(), "credentials_locked");
        assert!(
            !err.retryable(),
            "a locked vault must not be retried blindly"
        );
    }

    #[tokio::test]
    async fn an_ungranted_instance_is_told_what_to_do_without_any_value() {
        let (resolver, _vault) = resolver_with_key(Some("local-llm")).await;
        let err = err_of(resolver.resolve("another-instance", &vault_ref()).await);
        let ProviderError::AuthRequired { login_hint } = &err else {
            panic!("expected auth_required, got {err:?}");
        };
        let hint = login_hint.clone().unwrap_or_default();
        assert!(hint.contains("another-instance"));
        assert!(!format!("{err:?} {err}").contains("sk-resolver-value-4321"));
    }

    #[tokio::test]
    async fn none_is_no_credential() {
        let (resolver, _vault) = resolver_with_key(None).await;
        assert!(resolver
            .resolve("local-llm", &CredentialRef::None)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn the_servers_own_secrets_cannot_be_used_as_a_provider_credential() {
        let (resolver, _vault) = resolver_with_key(None).await;
        for var in ["PO_JWT_SECRET", "NEO4J_PASSWORD"] {
            let err = err_of(
                resolver
                    .resolve("local-llm", &CredentialRef::Env(var.to_string()))
                    .await,
            );
            assert_eq!(err.kind(), "invalid_request", "{var}");
        }
    }

    #[tokio::test]
    async fn an_unset_env_reference_is_auth_required() {
        let (resolver, _vault) = resolver_with_key(None).await;
        let err = err_of(
            resolver
                .resolve(
                    "local-llm",
                    &CredentialRef::Env("PO_TEST_PROVIDER_KEY_THAT_IS_NEVER_SET".to_string()),
                )
                .await,
        );
        assert_eq!(err.kind(), "auth_required");
    }
}
