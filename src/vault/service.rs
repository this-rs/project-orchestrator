//! The vault as the server sees it: one object owning the encrypted store, the
//! grant book, the output masker and the requests agents are waiting on.
//!
//! Every path that hands out a value goes through [`VaultService::read_for_agent`]
//! (an agent, under a grant) — and every value handed out is registered with the
//! masker first, so it is masked before the caller can possibly print it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Duration, Utc};
use serde::Serialize;
use uuid::Uuid;

use super::crypto::KdfParams;
use super::grants::{
    authorize, AccessRequest, Denied, Grant, GrantBook, GrantScope, SecretSelector,
};
use super::mask::SharedMasker;
use super::store::{validate_name, SecretMeta, Vault, VaultError, VaultStatus};

/// Pending requests older than this are dropped: the agent has long given up.
pub const REQUEST_TTL: Duration = Duration::minutes(30);

/// Unlock proofs kept at once (one per tab that unlocked).
const MAX_PROOFS: usize = 8;

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Wrong passphrases in a row before unlock attempts are slowed down.
const FREE_ATTEMPTS: u32 = 3;
/// Ceiling of the delay imposed after repeated failures.
const MAX_BACKOFF_SECS: i64 = 300;

/// An agent asked the user for a secret it does not have (MCP `vault.request_secret`).
#[derive(Debug, Clone, Serialize)]
pub struct SecretRequest {
    pub id: Uuid,
    /// The name the value will be stored under.
    pub name: String,
    /// Why the agent needs it, in the agent's words — shown on the card.
    pub reason: String,
    pub session_id: String,
    pub project_slug: Option<String>,
    /// Whether a secret with this name already exists (the user may then just
    /// grant it instead of typing it again).
    pub exists: bool,
    pub created_at: DateTime<Utc>,
}

/// The vault as an agent may see it.
#[derive(Debug, Clone, Serialize)]
pub struct AgentView {
    /// Granted secrets are readable only while this is true.
    pub unlocked: bool,
    pub secrets: Vec<AgentSecretView>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentSecretView {
    pub name: String,
    pub description: Option<String>,
    /// An active grant covers this session.
    pub granted: bool,
}

/// How the user answered a request.
#[derive(Debug, Clone)]
pub enum RequestAnswer {
    /// Typed a value: stored under the requested name, then granted.
    Provide {
        value: String,
        description: Option<String>,
    },
    /// The secret already exists: just grant it.
    GrantExisting,
    Decline,
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum ServiceError {
    #[error(transparent)]
    Vault(#[from] VaultError),
    #[error("{0}")]
    Denied(#[from] Denied),
    #[error("too many wrong passphrases — retry in {0} s")]
    Throttled(i64),
    #[error("no such pending request (it may have expired or been answered)")]
    UnknownRequest,
    #[error("this change needs the vault passphrase: unlock the vault from this page first")]
    ProofRequired,
    #[error("the vault is unavailable: {0}")]
    Unavailable(String),
    #[error("vault storage error: {0}")]
    Io(String),
}

impl From<std::io::Error> for ServiceError {
    fn from(e: std::io::Error) -> Self {
        ServiceError::Io(e.to_string())
    }
}

#[derive(Debug, Default)]
struct Throttle {
    failures: u32,
    retry_at: Option<DateTime<Utc>>,
}

impl Throttle {
    fn check(&self, now: DateTime<Utc>) -> Result<(), ServiceError> {
        match self.retry_at {
            Some(at) if now < at => Err(ServiceError::Throttled((at - now).num_seconds().max(1))),
            _ => Ok(()),
        }
    }

    fn failed(&mut self, now: DateTime<Utc>) {
        self.failures += 1;
        if self.failures >= FREE_ATTEMPTS {
            let exp = (self.failures - FREE_ATTEMPTS).min(8);
            let secs = (1i64 << exp).min(MAX_BACKOFF_SECS);
            self.retry_at = Some(now + Duration::seconds(secs));
        }
    }

    fn succeeded(&mut self) {
        *self = Self::default();
    }
}

#[derive(Debug)]
pub struct VaultService {
    vault: Arc<Mutex<Vault>>,
    grants: Mutex<GrantBook>,
    masker: SharedMasker,
    requests: Mutex<HashMap<Uuid, SecretRequest>>,
    throttle: Mutex<Throttle>,
    kdf: KdfParams,
    /// Unlock proofs: random tokens handed to whoever typed the passphrase
    /// (init/unlock), required to widen access. See [`VaultService::check_proof`].
    proofs: Mutex<Vec<zeroize::Zeroizing<String>>>,
    /// Set when the vault file could not be loaded: every operation that would
    /// write or open refuses, instead of silently starting a fresh vault that
    /// would shadow (and later overwrite) the unreadable one.
    unavailable: Option<String>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Tests init ephemeral vaults a lot: use cheap parameters there. Outside
/// tests an ephemeral vault is only the refusing stand-in of
/// [`VaultService::unavailable`], which never derives a key.
fn ephemeral_kdf() -> KdfParams {
    #[cfg(test)]
    {
        KdfParams::insecure_for_tests()
    }
    #[cfg(not(test))]
    {
        KdfParams::default()
    }
}

impl VaultService {
    /// The vault stored at `path` (grants beside it), locked. Values it hands
    /// out are registered with `masker` — [`super::mask::global`] in the server.
    pub fn open(
        path: impl Into<PathBuf>,
        kdf: KdfParams,
        masker: SharedMasker,
    ) -> Result<Self, ServiceError> {
        let path = path.into();
        let grants = GrantBook::load(GrantBook::default_path_beside(&path))?;
        Ok(Self {
            vault: Arc::new(Mutex::new(Vault::load(path)?)),
            grants: Mutex::new(grants),
            masker,
            requests: Mutex::new(HashMap::new()),
            throttle: Mutex::new(Throttle::default()),
            kdf,
            proofs: Mutex::new(Vec::new()),
            unavailable: None,
        })
    }

    /// A vault in a fresh temporary location — for tests and for the many
    /// `ServerState` literals in test modules. Nothing is written until init.
    pub fn ephemeral() -> Arc<Self> {
        let path = std::env::temp_dir()
            .join(format!("po-vault-ephemeral-{}", Uuid::new_v4()))
            .join("vault.json");
        Arc::new(
            Self::open(path, ephemeral_kdf(), SharedMasker::default())
                .expect("a vault in a fresh temp path cannot fail to load"),
        )
    }

    /// A service that refuses everything, reporting `reason`.
    pub fn unavailable(reason: String) -> Arc<Self> {
        let mut svc = Arc::try_unwrap(Self::ephemeral()).expect("fresh Arc");
        svc.unavailable = Some(reason);
        Arc::new(svc)
    }

    /// Why the vault cannot be used, if it cannot.
    pub fn unavailable_reason(&self) -> Option<&str> {
        self.unavailable.as_deref()
    }

    fn usable(&self) -> Result<(), ServiceError> {
        match &self.unavailable {
            Some(r) => Err(ServiceError::Unavailable(r.clone())),
            None => Ok(()),
        }
    }

    pub fn masker(&self) -> &SharedMasker {
        &self.masker
    }

    pub fn status(&self, now: DateTime<Utc>) -> VaultStatus {
        lock(&self.vault).status(now)
    }

    pub fn list(&self) -> Vec<SecretMeta> {
        lock(&self.vault).list()
    }

    pub fn contains(&self, name: &str) -> bool {
        lock(&self.vault).contains(name)
    }

    pub fn is_unlocked(&self, now: DateTime<Utc>) -> bool {
        lock(&self.vault).is_unlocked(now)
    }

    /// Key derivation is deliberately slow (tens to hundreds of ms): run it off
    /// the async runtime.
    /// Returns an unlock proof (see [`Self::check_proof`]).
    pub async fn init(
        &self,
        passphrase: String,
        duration: Duration,
    ) -> Result<String, ServiceError> {
        self.usable()?;
        let vault = self.vault.clone();
        let kdf = self.kdf;
        tokio::task::spawn_blocking(move || {
            let passphrase = zeroize::Zeroizing::new(passphrase);
            lock(&vault).init(&passphrase, kdf, duration, Utc::now())
        })
        .await
        .map_err(|e| ServiceError::Io(e.to_string()))??;
        Ok(self.issue_proof())
    }

    /// Wrong passphrases are counted; past a few, attempts are refused for an
    /// increasing delay — a UI or an agent must not be able to brute-force it.
    pub async fn unlock(
        &self,
        passphrase: String,
        duration: Duration,
    ) -> Result<(DateTime<Utc>, String), ServiceError> {
        self.usable()?;
        lock(&self.throttle).check(Utc::now())?;
        let vault = self.vault.clone();
        let result = tokio::task::spawn_blocking(move || {
            let passphrase = zeroize::Zeroizing::new(passphrase);
            lock(&vault).unlock(&passphrase, duration, Utc::now())
        })
        .await
        .map_err(|e| ServiceError::Io(e.to_string()))?;
        let mut throttle = lock(&self.throttle);
        match result {
            Ok(until) => {
                throttle.succeeded();
                drop(throttle);
                Ok((until, self.issue_proof()))
            }
            Err(VaultError::WrongPassphrase) => {
                throttle.failed(Utc::now());
                Err(VaultError::WrongPassphrase.into())
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn lock_now(&self) {
        lock(&self.vault).lock();
        lock(&self.proofs).clear();
    }

    fn issue_proof(&self) -> String {
        use chacha20poly1305::aead::rand_core::RngCore;
        let mut bytes = [0u8; 32];
        chacha20poly1305::aead::OsRng.fill_bytes(&mut bytes);
        let proof = hex::encode(bytes);
        let mut proofs = lock(&self.proofs);
        proofs.push(zeroize::Zeroizing::new(proof.clone()));
        // A few tabs may each have unlocked; older proofs beyond that go.
        let excess = proofs.len().saturating_sub(MAX_PROOFS);
        proofs.drain(..excess);
        proof
    }

    /// Operations that WIDEN what agents can read (store a secret, grant one,
    /// answer a request) require proof that the caller typed the passphrase
    /// during the current unlock — not merely a valid login token.
    ///
    /// Why: agents run as the same OS user as the server and can read its
    /// config, signing key included, so they can forge any login token. They
    /// cannot produce this proof: it is random, issued only in answer to the
    /// passphrase, held in memory on both sides, and void once the vault locks.
    pub fn check_proof(&self, proof: Option<&str>, now: DateTime<Utc>) -> Result<(), ServiceError> {
        if !self.is_unlocked(now) {
            lock(&self.proofs).clear();
            return Err(VaultError::Locked.into());
        }
        let Some(proof) = proof else {
            return Err(ServiceError::ProofRequired);
        };
        let proofs = lock(&self.proofs);
        // Constant-time comparison: the check must not leak a prefix.
        let ok = proofs.iter().fold(false, |acc, p| {
            acc | constant_time_eq(p.as_bytes(), proof.as_bytes())
        });
        if ok {
            Ok(())
        } else {
            Err(ServiceError::ProofRequired)
        }
    }

    pub fn put(
        &self,
        name: &str,
        value: &str,
        description: Option<String>,
        now: DateTime<Utc>,
    ) -> Result<(), ServiceError> {
        lock(&self.vault).put(name, value, description, now)?;
        // A replaced value must be masked under its new form too. The old form
        // stays masked only if it was delivered before; `register` overwrites it,
        // which is acceptable: the old value is no longer the secret.
        if self.was_delivered(name) {
            self.masker.register(name, value);
        }
        Ok(())
    }

    fn was_delivered(&self, name: &str) -> bool {
        self.masker.knows(name)
    }

    pub fn delete(&self, name: &str) -> Result<(), ServiceError> {
        lock(&self.vault).delete(name)?;
        self.masker.forget(name);
        Ok(())
    }

    pub fn grants(&self, now: DateTime<Utc>) -> Vec<Grant> {
        lock(&self.grants).list(now)
    }

    pub fn grant(
        &self,
        secrets: SecretSelector,
        scope: GrantScope,
        duration: Duration,
        note: Option<String>,
        now: DateTime<Utc>,
    ) -> Result<Grant, ServiceError> {
        if let SecretSelector::Names(names) = &secrets {
            for n in names {
                validate_name(n)?;
            }
        }
        Ok(lock(&self.grants).create(secrets, scope, duration, note, now)?)
    }

    pub fn revoke(&self, id: Uuid, now: DateTime<Utc>) -> Result<bool, ServiceError> {
        Ok(lock(&self.grants).revoke(id, now)?)
    }

    /// The one path by which a value leaves the vault. The caller identity
    /// (`session_id`) must come from a signed token, never from a header or
    /// body the agent controls.
    pub fn read_for_agent(
        &self,
        name: &str,
        session_id: &str,
        project_slug: Option<&str>,
        now: DateTime<Utc>,
    ) -> Result<zeroize::Zeroizing<String>, ServiceError> {
        let mut vault = lock(&self.vault);
        let unlocked = vault.is_unlocked(now);
        let exists = vault.contains(name);
        let req = AccessRequest {
            secret: name,
            session_id,
            project_slug,
        };
        authorize(unlocked, exists, &lock(&self.grants), &req, now)?;
        let value = vault.get(name, now)?;
        // Masked BEFORE it is returned: from here on the agent may print it.
        self.masker.register(name, &value);
        Ok(value)
    }

    /// Whether `read_for_agent` would succeed — without delivering anything.
    pub fn read_for_agent_check(
        &self,
        name: &str,
        session_id: &str,
        project_slug: Option<&str>,
        now: DateTime<Utc>,
    ) -> bool {
        let mut vault = lock(&self.vault);
        let unlocked = vault.is_unlocked(now);
        let exists = vault.contains(name);
        let req = AccessRequest {
            secret: name,
            session_id,
            project_slug,
        };
        authorize(unlocked, exists, &lock(&self.grants), &req, now).is_ok()
    }

    /// What a session may know: the vault state and, per secret, its name and
    /// whether this session can read it now. Never a value.
    pub fn available_to(
        &self,
        session_id: &str,
        project_slug: Option<&str>,
        now: DateTime<Utc>,
    ) -> AgentView {
        let mut vault = lock(&self.vault);
        let unlocked = vault.is_unlocked(now);
        let grants = lock(&self.grants);
        let secrets = vault
            .list()
            .into_iter()
            .map(|m| {
                let req = AccessRequest {
                    secret: &m.name,
                    session_id,
                    project_slug,
                };
                AgentSecretView {
                    granted: grants.covering(&req, now).is_some(),
                    name: m.name,
                    description: m.description,
                }
            })
            .collect();
        AgentView { unlocked, secrets }
    }

    // ── Requests from agents ─────────────────────────────────────────────

    pub fn open_request(
        &self,
        name: &str,
        reason: &str,
        session_id: &str,
        project_slug: Option<&str>,
        now: DateTime<Utc>,
    ) -> Result<SecretRequest, ServiceError> {
        validate_name(name)?;
        let mut requests = lock(&self.requests);
        requests.retain(|_, r| now - r.created_at < REQUEST_TTL);
        // The same agent asking twice for the same thing gets the same card.
        if let Some(existing) = requests
            .values()
            .find(|r| r.session_id == session_id && r.name == name)
        {
            return Ok(existing.clone());
        }
        let request = SecretRequest {
            id: Uuid::new_v4(),
            name: name.to_string(),
            reason: reason.chars().take(500).collect(),
            session_id: session_id.to_string(),
            project_slug: project_slug.map(str::to_string),
            exists: self.contains(name),
            created_at: now,
        };
        requests.insert(request.id, request.clone());
        Ok(request)
    }

    pub fn pending_requests(&self, now: DateTime<Utc>) -> Vec<SecretRequest> {
        let mut requests = lock(&self.requests);
        requests.retain(|_, r| now - r.created_at < REQUEST_TTL);
        let mut out: Vec<SecretRequest> = requests.values().cloned().collect();
        out.sort_by_key(|r| r.created_at);
        out
    }

    /// Answer a request. On `Provide`/`GrantExisting` the secret is granted to
    /// `scope` (defaulting to the requesting session) for `duration`. The
    /// request is consumed only once the answer took effect, so a failure (the
    /// vault is locked, the value is too short) leaves the card answerable.
    pub fn answer_request(
        &self,
        id: Uuid,
        answer: RequestAnswer,
        scope: Option<GrantScope>,
        duration: Duration,
        now: DateTime<Utc>,
    ) -> Result<(SecretRequest, Option<Grant>), ServiceError> {
        let request = lock(&self.requests)
            .get(&id)
            .filter(|r| now - r.created_at < REQUEST_TTL)
            .cloned()
            .ok_or(ServiceError::UnknownRequest)?;
        let grant = match answer {
            RequestAnswer::Decline => None,
            RequestAnswer::Provide { value, description } => {
                let value = zeroize::Zeroizing::new(value);
                self.put(&request.name, &value, description, now)?;
                Some(self.grant_for(&request, scope, duration, now)?)
            }
            RequestAnswer::GrantExisting => {
                if !self.contains(&request.name) {
                    return Err(VaultError::UnknownSecret.into());
                }
                Some(self.grant_for(&request, scope, duration, now)?)
            }
        };
        lock(&self.requests).remove(&id);
        Ok((request, grant))
    }

    fn grant_for(
        &self,
        request: &SecretRequest,
        scope: Option<GrantScope>,
        duration: Duration,
        now: DateTime<Utc>,
    ) -> Result<Grant, ServiceError> {
        let scope = scope.unwrap_or_else(|| GrantScope::Session(request.session_id.clone()));
        let names = SecretSelector::Names([request.name.clone()].into_iter().collect());
        self.grant(
            names,
            scope,
            duration,
            Some(format!("requested: {}", request.reason)),
            now,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PASS: &str = "correct horse battery staple";

    async fn open_vault() -> Arc<VaultService> {
        let svc = VaultService::ephemeral();
        svc.init(PASS.into(), Duration::hours(1)).await.unwrap();
        svc
    }

    #[tokio::test]
    async fn an_agent_reads_only_under_a_grant_and_the_value_is_then_masked() {
        let svc = open_vault().await;
        let now = Utc::now();
        svc.put("mermaid", "the-mermaid-passphrase", None, now)
            .unwrap();

        assert_eq!(
            svc.read_for_agent("mermaid", "s1", None, now).unwrap_err(),
            ServiceError::Denied(Denied::NoGrant)
        );
        // Nothing delivered yet → nothing to mask.
        assert_eq!(
            svc.masker().mask("the-mermaid-passphrase"),
            "the-mermaid-passphrase"
        );

        svc.grant(
            SecretSelector::Names(["mermaid".to_string()].into()),
            GrantScope::Session("s1".into()),
            Duration::hours(1),
            None,
            now,
        )
        .unwrap();
        let v = svc.read_for_agent("mermaid", "s1", None, now).unwrap();
        assert_eq!(v.as_str(), "the-mermaid-passphrase");
        assert_eq!(
            svc.masker().mask("x the-mermaid-passphrase"),
            "x [secret:mermaid]"
        );
        // Another session is still refused.
        assert!(svc.read_for_agent("mermaid", "s2", None, now).is_err());
    }

    #[tokio::test]
    async fn a_locked_vault_refuses_even_granted_reads() {
        let svc = open_vault().await;
        let now = Utc::now();
        svc.put("k", "value-12345", None, now).unwrap();
        svc.grant(
            SecretSelector::All,
            GrantScope::Anywhere,
            Duration::hours(1),
            None,
            now,
        )
        .unwrap();
        svc.lock_now();
        assert_eq!(
            svc.read_for_agent("k", "s1", None, now).unwrap_err(),
            ServiceError::Denied(Denied::VaultLocked)
        );
    }

    #[tokio::test]
    async fn repeated_wrong_passphrases_are_throttled() {
        let svc = open_vault().await;
        svc.lock_now();
        for _ in 0..FREE_ATTEMPTS {
            let e = svc
                .unlock("wrong passphrase!!".into(), Duration::hours(1))
                .await;
            assert_eq!(
                e.unwrap_err(),
                ServiceError::Vault(VaultError::WrongPassphrase)
            );
        }
        // Even the right passphrase waits: the throttle does not reveal whether
        // the next guess would have been right.
        let e = svc.unlock(PASS.into(), Duration::hours(1)).await;
        assert!(matches!(e, Err(ServiceError::Throttled(_))), "{e:?}");
    }

    #[tokio::test]
    async fn answering_a_request_stores_grants_and_consumes_it() {
        let svc = open_vault().await;
        let now = Utc::now();
        let req = svc
            .open_request("mermaid", "publish diagrams", "s1", Some("po"), now)
            .unwrap();
        assert!(!req.exists);
        // Asking twice yields the same card.
        let again = svc
            .open_request("mermaid", "publish diagrams", "s1", Some("po"), now)
            .unwrap();
        assert_eq!(req.id, again.id);

        let (_, grant) = svc
            .answer_request(
                req.id,
                RequestAnswer::Provide {
                    value: "the-mermaid-passphrase".into(),
                    description: None,
                },
                None,
                Duration::hours(2),
                now,
            )
            .unwrap();
        assert_eq!(grant.unwrap().scope, GrantScope::Session("s1".into()));
        assert!(svc.pending_requests(now).is_empty());
        assert!(svc.read_for_agent("mermaid", "s1", Some("po"), now).is_ok());
    }

    #[tokio::test]
    async fn a_failed_answer_leaves_the_request_open() {
        let svc = open_vault().await;
        let now = Utc::now();
        let req = svc.open_request("k", "why", "s1", None, now).unwrap();
        svc.lock_now();
        let e = svc.answer_request(
            req.id,
            RequestAnswer::Provide {
                value: "value-12345".into(),
                description: None,
            },
            None,
            Duration::hours(1),
            now,
        );
        assert_eq!(e.unwrap_err(), ServiceError::Vault(VaultError::Locked));
        assert_eq!(svc.pending_requests(now).len(), 1);
    }

    #[tokio::test]
    async fn requests_expire() {
        let svc = open_vault().await;
        let now = Utc::now();
        let req = svc.open_request("k", "why", "s1", None, now).unwrap();
        let later = now + REQUEST_TTL + Duration::seconds(1);
        assert!(svc.pending_requests(later).is_empty());
        assert_eq!(
            svc.answer_request(
                req.id,
                RequestAnswer::Decline,
                None,
                Duration::hours(1),
                later
            )
            .unwrap_err(),
            ServiceError::UnknownRequest
        );
    }
}
