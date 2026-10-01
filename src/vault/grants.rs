//! Grants: which secrets an agent may use, where, and until when.
//!
//! This is what makes the vault non-blocking. Instead of the user typing a
//! secret every time an agent needs it, they grant once — "these secrets, for
//! this project (or this session, or anywhere), until this time" — and agents
//! within that scope use them without asking.
//!
//! Two independent conditions must BOTH hold for an agent to read a secret:
//! the vault is open (the key is in memory — bounded by the unlock duration),
//! and a grant covers the request (bounded by scope and its own expiry). The
//! user's "unlock for a period, a scope, or both" is exactly the choice of which
//! of the two bounds to tighten.
//!
//! Grants hold no secret material, so they persist while the vault is locked;
//! they are merely inert until it is opened again.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Upper bound on a grant's lifetime. The unlock duration already bounds actual
/// use; this only stops a grant from quietly outliving the reason it was made.
pub const MAX_GRANT: Duration = Duration::days(30);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum GrantScope {
    /// One chat session.
    Session(String),
    /// Every session working on this project.
    Project(String),
    /// Every session. The widest grant; the unlock duration is then the only
    /// bound, so the UI must say so.
    Anywhere,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "names", rename_all = "snake_case")]
pub enum SecretSelector {
    All,
    Names(BTreeSet<String>),
}

impl SecretSelector {
    fn includes(&self, name: &str) -> bool {
        match self {
            Self::All => true,
            Self::Names(names) => names.contains(name),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grant {
    pub id: Uuid,
    pub secrets: SecretSelector,
    pub scope: GrantScope,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    /// Why it was granted, in the user's words — shown when reviewing grants.
    #[serde(default)]
    pub note: Option<String>,
}

/// Who is asking for what.
#[derive(Debug, Clone, Copy)]
pub struct AccessRequest<'a> {
    pub secret: &'a str,
    pub session_id: &'a str,
    /// Project the session works on, when it has one.
    pub project_slug: Option<&'a str>,
}

/// Why access was refused — precise, because "denied" alone leaves the agent
/// (and the user reading its message) guessing what to do next.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Denied {
    #[error("the vault is locked — the user must unlock it")]
    VaultLocked,
    #[error("no such secret in the vault")]
    UnknownSecret,
    #[error("no grant covers this secret for this session — ask the user (vault request_secret)")]
    NoGrant,
}

impl Grant {
    fn covers(&self, req: &AccessRequest<'_>, now: DateTime<Utc>) -> bool {
        if now >= self.expires_at || !self.secrets.includes(req.secret) {
            return false;
        }
        match &self.scope {
            GrantScope::Anywhere => true,
            GrantScope::Session(id) => id == req.session_id,
            GrantScope::Project(slug) => req.project_slug == Some(slug.as_str()),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct GrantFile {
    grants: Vec<Grant>,
}

#[derive(Debug)]
pub struct GrantBook {
    path: PathBuf,
    grants: Vec<Grant>,
}

impl GrantBook {
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, std::io::Error> {
        let path = path.into();
        let grants = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice::<GrantFile>(&bytes)
                .map(|f| f.grants)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e),
        };
        Ok(Self { path, grants })
    }

    pub fn default_path_beside(vault_path: &Path) -> PathBuf {
        vault_path.with_file_name("vault-grants.json")
    }

    /// Active grants only; expired ones are dropped from view (and from disk on
    /// the next write).
    pub fn list(&self, now: DateTime<Utc>) -> Vec<Grant> {
        self.grants
            .iter()
            .filter(|g| now < g.expires_at)
            .cloned()
            .collect()
    }

    pub fn create(
        &mut self,
        secrets: SecretSelector,
        scope: GrantScope,
        duration: Duration,
        note: Option<String>,
        now: DateTime<Utc>,
    ) -> Result<Grant, std::io::Error> {
        let duration = duration.clamp(Duration::minutes(1), MAX_GRANT);
        let grant = Grant {
            id: Uuid::new_v4(),
            secrets,
            scope,
            created_at: now,
            expires_at: now + duration,
            note,
        };
        let mut next: Vec<Grant> = self.list(now);
        next.push(grant.clone());
        self.persist(next)?;
        Ok(grant)
    }

    /// Returns whether a grant was removed.
    pub fn revoke(&mut self, id: Uuid, now: DateTime<Utc>) -> Result<bool, std::io::Error> {
        let before = self.grants.len();
        let next: Vec<Grant> = self.list(now).into_iter().filter(|g| g.id != id).collect();
        let removed = next.len() < before && self.grants.iter().any(|g| g.id == id);
        self.persist(next)?;
        Ok(removed)
    }

    /// Does some active grant cover this request? Checks the grant only — the
    /// caller must separately check that the vault is open and the secret exists
    /// (see [`authorize`]).
    pub fn covering(&self, req: &AccessRequest<'_>, now: DateTime<Utc>) -> Option<&Grant> {
        self.grants.iter().find(|g| g.covers(req, now))
    }

    fn persist(&mut self, grants: Vec<Grant>) -> Result<(), std::io::Error> {
        let bytes = serde_json::to_vec_pretty(&GrantFile {
            grants: grants.clone(),
        })
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        write_private(&self.path, &bytes)?;
        self.grants = grants;
        Ok(())
    }
}

/// The single decision point for an agent reading a secret. Order matters for
/// the message the agent gets: a locked vault is reported first, because no
/// grant can help until the user opens it.
pub fn authorize(
    vault_unlocked: bool,
    secret_exists: bool,
    book: &GrantBook,
    req: &AccessRequest<'_>,
    now: DateTime<Utc>,
) -> Result<Uuid, Denied> {
    if !vault_unlocked {
        return Err(Denied::VaultLocked);
    }
    if !secret_exists {
        return Err(Denied::UnknownSecret);
    }
    book.covering(req, now).map(|g| g.id).ok_or(Denied::NoGrant)
}

/// Same atomic, owner-only write as the vault file: temp file created 0600,
/// flushed, renamed over the original.
fn write_private(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    use std::io::Write;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t0() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-10-01T09:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    fn book() -> (tempfile::TempDir, GrantBook) {
        let dir = tempfile::tempdir().unwrap();
        let b = GrantBook::load(dir.path().join("vault-grants.json")).unwrap();
        (dir, b)
    }

    fn req<'a>(secret: &'a str, session: &'a str, project: Option<&'a str>) -> AccessRequest<'a> {
        AccessRequest {
            secret,
            session_id: session,
            project_slug: project,
        }
    }

    fn names(ns: &[&str]) -> SecretSelector {
        SecretSelector::Names(ns.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn a_project_grant_covers_its_sessions_and_nothing_else() {
        let (_d, mut b) = book();
        b.create(
            names(&["mermaid"]),
            GrantScope::Project("po".into()),
            Duration::hours(2),
            None,
            t0(),
        )
        .unwrap();
        assert!(b
            .covering(&req("mermaid", "s1", Some("po")), t0())
            .is_some());
        assert!(b
            .covering(&req("mermaid", "s2", Some("po")), t0())
            .is_some());
        assert!(b
            .covering(&req("mermaid", "s1", Some("obrain")), t0())
            .is_none());
        assert!(b.covering(&req("mermaid", "s1", None), t0()).is_none());
        assert!(b.covering(&req("github", "s1", Some("po")), t0()).is_none());
    }

    #[test]
    fn a_session_grant_covers_that_session_only() {
        let (_d, mut b) = book();
        b.create(
            SecretSelector::All,
            GrantScope::Session("s1".into()),
            Duration::hours(1),
            None,
            t0(),
        )
        .unwrap();
        assert!(b
            .covering(&req("anything", "s1", Some("po")), t0())
            .is_some());
        assert!(b
            .covering(&req("anything", "s2", Some("po")), t0())
            .is_none());
    }

    #[test]
    fn an_expired_grant_covers_nothing() {
        let (_d, mut b) = book();
        b.create(
            SecretSelector::All,
            GrantScope::Anywhere,
            Duration::minutes(30),
            None,
            t0(),
        )
        .unwrap();
        let later = t0() + Duration::minutes(30);
        assert!(b.covering(&req("x", "s", None), later).is_none());
        assert!(b.list(later).is_empty());
    }

    #[test]
    fn revocation_is_immediate() {
        let (_d, mut b) = book();
        let g = b
            .create(
                SecretSelector::All,
                GrantScope::Anywhere,
                Duration::hours(1),
                None,
                t0(),
            )
            .unwrap();
        assert!(b.revoke(g.id, t0()).unwrap());
        assert!(b.covering(&req("x", "s", None), t0()).is_none());
        assert!(
            !b.revoke(g.id, t0()).unwrap(),
            "revoking twice reports nothing removed"
        );
    }

    #[test]
    fn a_locked_vault_denies_even_with_a_valid_grant() {
        let (_d, mut b) = book();
        b.create(
            SecretSelector::All,
            GrantScope::Anywhere,
            Duration::hours(1),
            None,
            t0(),
        )
        .unwrap();
        let r = req("mermaid", "s1", Some("po"));
        assert_eq!(
            authorize(false, true, &b, &r, t0()),
            Err(Denied::VaultLocked)
        );
        assert!(authorize(true, true, &b, &r, t0()).is_ok());
    }

    #[test]
    fn the_denial_says_what_to_do_next() {
        let (_d, b) = book();
        let r = req("mermaid", "s1", Some("po"));
        assert_eq!(
            authorize(true, false, &b, &r, t0()),
            Err(Denied::UnknownSecret)
        );
        assert_eq!(authorize(true, true, &b, &r, t0()), Err(Denied::NoGrant));
        assert!(Denied::NoGrant.to_string().contains("request_secret"));
    }

    #[test]
    fn grants_survive_a_reload() {
        let (d, mut b) = book();
        b.create(
            names(&["mermaid"]),
            GrantScope::Project("po".into()),
            Duration::hours(2),
            Some("mermaid diagrams".into()),
            t0(),
        )
        .unwrap();
        let reloaded = GrantBook::load(d.path().join("vault-grants.json")).unwrap();
        assert_eq!(reloaded.list(t0()).len(), 1);
        assert!(reloaded
            .covering(&req("mermaid", "s9", Some("po")), t0())
            .is_some());
    }

    #[test]
    fn grant_duration_is_bounded() {
        let (_d, mut b) = book();
        let g = b
            .create(
                SecretSelector::All,
                GrantScope::Anywhere,
                Duration::days(365),
                None,
                t0(),
            )
            .unwrap();
        assert_eq!(g.expires_at, t0() + MAX_GRANT);
    }

    #[test]
    fn scope_serialises_in_snake_case() {
        // Known trap in this codebase: enums reaching the UI in PascalCase.
        let json = serde_json::to_string(&GrantScope::Project("po".into())).unwrap();
        assert_eq!(json, r#"{"kind":"project","value":"po"}"#);
    }
}
