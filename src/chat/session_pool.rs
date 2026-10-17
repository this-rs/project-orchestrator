//! A pool of warm delegation sessions (CACHE-POOL, step 2).
//!
//! # Why a pool at all
//!
//! Measured on 17/10/2026, three Claude Code sessions opened with the same model, the same
//! prompt and the same working directory, read from their persisted `result` events:
//!
//! | opening | `cache_creation` | `cache_read` |
//! |---|---|---|
//! | session 1, turn 1 | 54 055 | 0 |
//! | session 2, turn 1 | 30 458 | 23 544 |
//! | session 3, turn 1 | 30 504 | 23 544 |
//! | session 3, turn 2 (SAME session) | 9 413 | 48 645 |
//!
//! The stable prefix is shared across sessions, but about 30 500 tokens are WRITTEN AGAIN
//! at every new session, while the next turn of a session already open writes only 9 413.
//! The cost of a delegation is its OPENING, not its work: a delegation that opens a fresh
//! child session pays that rewriting before doing anything.
//!
//! # What is pooled, and what a reused session may never carry
//!
//! A [`WarmSession`] is a delegation session that was just closed and whose prefix is
//! still in the provider's prompt cache. The next delegation of the same conversation, on
//! the same engine and the same model, RESUMES it instead of opening a new one.
//!
//! A reused session is only a warm PREFIX. It never lends its envelope, its consent, its
//! tool policy or its project to the delegation that resumes it: the delegation keeps its
//! own, computed by the opening path exactly as for a fresh session ([`super::subagents`]
//! resolves the envelope before it asks the pool). The pool's own rule is therefore the
//! opposite of an inheritance: an entry is eligible ONLY when every one of those facts is
//! IDENTICAL to what the new delegation needs. Anything else — another project, another
//! working directory, another access, another permission mode, another tool policy, a
//! changed consent, another person, another conversation — makes the entry INELIGIBLE. It
//! is not adapted: it is left alone (or dropped once expired) and a fresh session is opened.
//!
//! Refusing on a mismatch rather than adapting is the whole security of the design: a
//! session resumed across projects, or after the person's approvals changed, would carry a
//! consent nobody granted for this delegation, which is a security bug and not a
//! performance win.
//!
//! # Why the conversation is part of the key
//!
//! A resumed session replays its own transcript to the model: a warm session reused by
//! ANOTHER conversation would hand one conversation's prompts and answers to the next one,
//! possibly of another person. The key therefore names the parent conversation, its person
//! and its project, and the (engine, model) pair the task asks for — the same engine and
//! model give the same warm handle, which is what the measurement needs.
//!
//! # Failure mode
//!
//! The pool is an OPTIMISATION. An empty pool, a disabled pool, an expired entry, a
//! contaminated one, or a resume that the provider refuses all end the same way: a fresh
//! session, exactly as before this module existed. A delegation never fails for want of a
//! pool entry.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::provider::policy::SessionAccess;

/// How long a closed delegation session stays worth resuming. Below the provider's usual
/// prompt-cache lifetime (5 minutes): past it the prefix is gone and a resume would pay the
/// old transcript (a whole extra conversation in the context) for no cache hit at all.
pub const DEFAULT_TTL: Duration = Duration::from_secs(240);

/// How many warm sessions one server keeps. Delegations of one conversation are few and
/// short-lived; the cap only bounds the memory of the registry.
pub const DEFAULT_MAX_ENTRIES: usize = 8;

/// What identifies the warm session a delegation wants: the engine and model the task
/// names, and the conversation it belongs to. Two `Agent` calls that agree on all of it may
/// share a warm handle; disagreeing on any of it is another warm session entirely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmKey {
    /// The provider instance the child opens on (`claude-code`, `codex`, an ACP agent...).
    pub provider: String,
    /// The model of the child session.
    pub model: String,
    /// The conversation whose `Agent` call opened the warm session. A resumed session
    /// replays its transcript: it is never lent to another conversation.
    pub parent_session_id: String,
    /// The person the child belongs to.
    pub owner: Option<String>,
    /// The project the child runs in.
    pub project: Option<String>,
}

/// The facts a warm session was opened with, every one of which the next delegation must
/// match EXACTLY. They are never inherited: a mismatch means "open a fresh session".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmFacts {
    /// Working directory the child runs in.
    pub cwd: String,
    /// `normal` / `read_only`.
    pub access: SessionAccess,
    /// The permission mode the child was opened with.
    pub permission_mode: Option<String>,
    /// The extra directories the child may see.
    pub add_dirs: Vec<String>,
    /// The workspace whose roots were added.
    pub workspace: Option<String>,
    /// Fingerprint of the person's lasting approvals ([`super::lasting_rules`]): a
    /// delegation whose consent changed never resumes a session opened under the old one.
    pub consent: String,
    /// Fingerprint of the child's tool policy (its tool-name policy and the read-only MCP
    /// declarations that hold for its project).
    pub tool_policy: String,
}

impl WarmFacts {
    /// The first fact that differs from `other`, named as a [`Refusal`], or `None` when the
    /// two agree on everything. Named so a log line says WHY a warm session was passed over.
    pub fn mismatch(&self, other: &WarmFacts) -> Option<Refusal> {
        if self.cwd != other.cwd {
            return Some(Refusal::AnotherDirectory);
        }
        if self.access != other.access {
            return Some(Refusal::AnotherAccess);
        }
        if self.permission_mode != other.permission_mode {
            return Some(Refusal::AnotherPermissionMode);
        }
        if self.add_dirs != other.add_dirs {
            return Some(Refusal::AnotherDirectories);
        }
        if self.workspace != other.workspace {
            return Some(Refusal::AnotherWorkspace);
        }
        if self.tool_policy != other.tool_policy {
            return Some(Refusal::AnotherToolPolicy);
        }
        if self.consent != other.consent {
            return Some(Refusal::ConsentChanged);
        }
        None
    }
}

/// Why a warm session was not handed to a delegation. Every variant means the same thing
/// operationally — a fresh session is opened — but the name is what makes a refusal
/// observable (a log line, a test).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// Past [`SessionPool::ttl`]: the provider's cache is gone.
    Expired,
    /// Another engine instance.
    AnotherEngine,
    /// Another model.
    AnotherModel,
    /// Another conversation.
    AnotherConversation,
    /// Another person.
    AnotherPerson,
    /// Another project.
    AnotherProject,
    /// Another working directory.
    AnotherDirectory,
    /// Another access (`normal` / `read_only`).
    AnotherAccess,
    /// Another permission mode.
    AnotherPermissionMode,
    /// Other extra directories.
    AnotherDirectories,
    /// Another workspace.
    AnotherWorkspace,
    /// Another tool policy.
    AnotherToolPolicy,
    /// The person's lasting approvals changed since the warm session was opened.
    ConsentChanged,
}

impl Refusal {
    /// The stable name used in logs and asserted in tests.
    pub fn as_str(self) -> &'static str {
        match self {
            Refusal::Expired => "expired",
            Refusal::AnotherEngine => "another engine",
            Refusal::AnotherModel => "another model",
            Refusal::AnotherConversation => "another conversation",
            Refusal::AnotherPerson => "another person",
            Refusal::AnotherProject => "another project",
            Refusal::AnotherDirectory => "another directory",
            Refusal::AnotherAccess => "another access",
            Refusal::AnotherPermissionMode => "another permission mode",
            Refusal::AnotherDirectories => "another directories",
            Refusal::AnotherWorkspace => "another workspace",
            Refusal::AnotherToolPolicy => "another tool policy",
            Refusal::ConsentChanged => "consent changed",
        }
    }
}

/// A closed delegation session worth resuming, and everything it was opened with.
#[derive(Debug, Clone)]
pub struct WarmSession {
    /// The session to resume.
    pub session_id: String,
    /// What a delegation must agree on to claim it.
    pub key: WarmKey,
    /// How it was opened: never inherited, only compared.
    pub facts: WarmFacts,
    /// When it was opened, and when it was last given back to the pool.
    pub opened_at: Instant,
    pub last_used: Instant,
    /// How many delegations have run in this session so far (1 for the opening one). A
    /// resumed session carries the earlier turns: this is what a log line shows.
    pub uses: u32,
}

impl WarmSession {
    /// Whether the entry may be handed to a delegation needing `key`/`facts` at `now`.
    /// `Err` names the first reason it may not.
    pub fn eligibility(
        &self,
        key: &WarmKey,
        facts: &WarmFacts,
        now: Instant,
        ttl: Duration,
    ) -> Result<(), Refusal> {
        if now.saturating_duration_since(self.last_used) > ttl {
            return Err(Refusal::Expired);
        }
        if key.provider != self.key.provider {
            return Err(Refusal::AnotherEngine);
        }
        if key.model != self.key.model {
            return Err(Refusal::AnotherModel);
        }
        if key.parent_session_id != self.key.parent_session_id {
            return Err(Refusal::AnotherConversation);
        }
        if key.owner != self.key.owner {
            return Err(Refusal::AnotherPerson);
        }
        if key.project != self.key.project {
            return Err(Refusal::AnotherProject);
        }
        facts.mismatch(&self.facts).map_or(Ok(()), Err)
    }
}

/// What a claim found: a warm session to resume, or the reasons the pool had none.
#[derive(Debug, Default)]
pub struct Claim {
    /// Taken OUT of the pool: two delegations running at once never resume one session.
    pub held: Option<WarmSession>,
    /// Why each entry the pool held was passed over (empty when the pool was empty).
    pub refused: Vec<(String, Refusal)>,
}

impl Claim {
    /// A one-line account of the claim, for a debug log.
    pub fn summary(&self) -> String {
        match &self.held {
            Some(warm) => format!(
                "warm session {} ({} use(s), opened {}s ago)",
                warm.session_id,
                warm.uses,
                warm.last_used.elapsed().as_secs()
            ),
            None if self.refused.is_empty() => "no warm session".to_owned(),
            None => {
                let reasons: Vec<String> = self
                    .refused
                    .iter()
                    .map(|(id, why)| format!("{id} ({})", why.as_str()))
                    .collect();
                format!("no eligible warm session: {}", reasons.join(", "))
            }
        }
    }
}

/// The registry itself: a small, bounded list of warm sessions.
#[derive(Debug)]
pub struct SessionPool {
    entries: Mutex<Vec<WarmSession>>,
    max_entries: usize,
    ttl: Duration,
}

impl Default for SessionPool {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES, DEFAULT_TTL)
    }
}

impl SessionPool {
    /// A pool of at most `max_entries` warm sessions, each claimable for `ttl`. A
    /// `max_entries` of 0 is a pool that never holds anything ([`Self::disabled`]).
    pub fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            max_entries,
            ttl,
        }
    }

    /// The pool that reuses nothing: the delegation path then opens a fresh session every
    /// time, exactly as before this module existed. `PO_WARM_POOL=off` builds this one.
    pub fn disabled() -> Self {
        Self::new(0, Duration::ZERO)
    }

    /// The pool the configuration asks for: `PO_WARM_POOL=off` disables it, and
    /// `PO_WARM_POOL_TTL_SECS` moves the lifetime (never above `DEFAULT_TTL`, which is
    /// already below the provider's cache lifetime).
    pub fn from_env() -> Self {
        let off = std::env::var("PO_WARM_POOL").is_ok_and(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "off" || v == "0" || v == "false" || v == "no"
        });
        if off {
            return Self::disabled();
        }
        let ttl = std::env::var("PO_WARM_POOL_TTL_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(DEFAULT_TTL, |s| {
                Duration::from_secs(s.min(DEFAULT_TTL.as_secs()))
            });
        Self::new(DEFAULT_MAX_ENTRIES, ttl)
    }

    /// How long an entry stays claimable.
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// How many warm sessions the pool holds (tests, observability).
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether the pool holds nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The session ids the pool holds, oldest first (observability, tests).
    pub fn held(&self) -> Vec<String> {
        self.lock().iter().map(|w| w.session_id.clone()).collect()
    }

    /// Takes the warm session `key`/`facts` may reuse, if the pool holds one. The entry is
    /// REMOVED: a claimed session is never handed to two delegations, and a reuse that
    /// fails forgets it ([`Self::abandon`]) rather than returning it. Expired entries met on
    /// the way are dropped.
    pub fn claim(&self, key: &WarmKey, facts: &WarmFacts, now: Instant) -> Claim {
        let mut entries = self.lock();
        let mut claim = Claim::default();
        let mut chosen: Option<usize> = None;
        let mut expired: Vec<usize> = Vec::new();
        for (index, warm) in entries.iter().enumerate() {
            match warm.eligibility(key, facts, now, self.ttl) {
                Ok(()) => {
                    if chosen.is_none() {
                        chosen = Some(index);
                    }
                }
                Err(Refusal::Expired) => expired.push(index),
                Err(why) => claim.refused.push((warm.session_id.clone(), why)),
            }
        }
        for index in expired.into_iter().rev() {
            entries.remove(index);
        }
        if let Some(index) = chosen {
            claim.held = Some(entries.remove(index));
        }
        claim
    }

    /// Gives a session back, warm, once the delegation that ran in it is over: the next
    /// delegation of the same conversation may resume it. An entry for the same session id
    /// is replaced (the freshest facts win), the oldest entries are dropped past the cap,
    /// and nothing is kept when the pool is disabled.
    pub fn release(&self, mut warm: WarmSession, now: Instant) {
        if self.max_entries == 0 {
            return;
        }
        warm.last_used = now;
        let mut entries = self.lock();
        entries.retain(|held| held.session_id != warm.session_id);
        entries.push(warm);
        while entries.len() > self.max_entries {
            entries.remove(0);
        }
    }

    /// Drops the entry of `session_id`: the session must not be resumed again (its reuse
    /// failed, or it closed for a reason that makes its prefix worthless).
    pub fn abandon(&self, session_id: &str) {
        self.lock().retain(|held| held.session_id != session_id);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<WarmSession>> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// A stable 64-bit fingerprint of `parts`, so two openings can be compared without keeping
/// what they were built from (FNV-1a: no dependency, same answer in every process).
pub fn fingerprint<'a>(parts: impl IntoIterator<Item = Option<&'a str>>) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for byte in part.unwrap_or("\u{0}").as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        // A separator no field can contain keeps ["ab", "c"] from reading like ["a", "bc"].
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(provider: &str, model: &str) -> WarmKey {
        WarmKey {
            provider: provider.into(),
            model: model.into(),
            parent_session_id: "parent".into(),
            owner: Some("owner".into()),
            project: Some("proj".into()),
        }
    }

    fn facts() -> WarmFacts {
        WarmFacts {
            cwd: "/work".into(),
            access: SessionAccess::Normal,
            permission_mode: Some("default".into()),
            add_dirs: vec![],
            workspace: None,
            consent: fingerprint([Some("rule-1")]),
            tool_policy: fingerprint([Some("normal"), Some("default")]),
        }
    }

    fn warm(id: &str, key: WarmKey, facts: WarmFacts, now: Instant) -> WarmSession {
        WarmSession {
            session_id: id.into(),
            key,
            facts,
            opened_at: now,
            last_used: now,
            uses: 1,
        }
    }

    /// CACHE-POOL criterion 1, the MECHANISM: two delegations of the same conversation on
    /// the same engine and model get the SAME warm session handle; the second is a resume,
    /// not an opening. Red without the pool: each delegation would carry its own id.
    #[test]
    fn the_same_engine_and_model_gets_the_same_warm_handle() {
        let pool = SessionPool::default();
        let now = Instant::now();
        pool.release(warm("s1", key("claude-code", "opus"), facts(), now), now);
        let first = pool.claim(&key("claude-code", "opus"), &facts(), now);
        assert_eq!(
            first.held.as_ref().map(|w| w.session_id.as_str()),
            Some("s1")
        );
        assert!(first.refused.is_empty());
        // Taken out of the pool while a delegation runs in it: a second concurrent
        // delegation must NOT be handed the same session.
        let second = pool.claim(&key("claude-code", "opus"), &facts(), now);
        assert!(second.held.is_none(), "a claimed session is not lent twice");
        // Given back warm, the same handle serves the next delegation.
        pool.release(first.held.unwrap(), now);
        let third = pool.claim(&key("claude-code", "opus"), &facts(), now);
        assert_eq!(
            third.held.as_ref().map(|w| w.session_id.as_str()),
            Some("s1")
        );
    }

    /// CACHE-POOL criterion 3, another PROJECT: a session opened in one project is never
    /// resumed for a delegation of another. Red if the project check is dropped from
    /// [`WarmSession::eligibility`].
    #[test]
    fn another_project_is_never_reused() {
        let pool = SessionPool::default();
        let now = Instant::now();
        pool.release(warm("s1", key("claude-code", "opus"), facts(), now), now);
        let mut other = key("claude-code", "opus");
        other.project = Some("autre".into());
        let claim = pool.claim(&other, &facts(), now);
        assert!(claim.held.is_none(), "a warm session crosses no project");
        assert_eq!(
            claim.refused,
            vec![("s1".to_owned(), Refusal::AnotherProject)]
        );
        // The entry is left alone for the delegations of ITS project.
        assert_eq!(pool.held(), vec!["s1".to_owned()]);
    }

    /// CACHE-POOL criterion 3, another TOOL POLICY: a session opened under one tool policy
    /// is never resumed under another. Red if the tool-policy fact is not compared.
    #[test]
    fn another_tool_policy_is_never_reused() {
        let pool = SessionPool::default();
        let now = Instant::now();
        pool.release(warm("s1", key("claude-code", "opus"), facts(), now), now);
        let mut read_only_session = facts();
        read_only_session.tool_policy = fingerprint([Some("read_only"), Some("default")]);
        let claim = pool.claim(&key("claude-code", "opus"), &read_only_session, now);
        assert!(claim.held.is_none());
        assert_eq!(
            claim.refused,
            vec![("s1".to_owned(), Refusal::AnotherToolPolicy)]
        );
    }

    /// CACHE-POOL criterion 2, consent: a person approving (or revoking) a lasting rule
    /// between two delegations makes the warm session ineligible — the session would
    /// otherwise carry decisions taken before the change. Red if `consent` is not compared.
    #[test]
    fn a_changed_consent_is_never_reused() {
        let pool = SessionPool::default();
        let now = Instant::now();
        pool.release(warm("s1", key("claude-code", "opus"), facts(), now), now);
        let mut after = facts();
        after.consent = fingerprint([Some("rule-1"), Some("rule-2")]);
        let claim = pool.claim(&key("claude-code", "opus"), &after, now);
        assert!(claim.held.is_none(), "a changed consent is not inherited");
        assert_eq!(
            claim.refused,
            vec![("s1".to_owned(), Refusal::ConsentChanged)]
        );
    }

    /// CACHE-POOL criterion 2, another PERSON: a warm session of one person's conversation
    /// is never resumed for another's. Red if the owner is not compared.
    #[test]
    fn another_person_is_never_reused() {
        let pool = SessionPool::default();
        let now = Instant::now();
        pool.release(warm("s1", key("claude-code", "opus"), facts(), now), now);
        let mut other = key("claude-code", "opus");
        other.owner = Some("quelqu_un_d_autre".into());
        let claim = pool.claim(&other, &facts(), now);
        assert!(claim.held.is_none());
        assert_eq!(
            claim.refused,
            vec![("s1".to_owned(), Refusal::AnotherPerson)]
        );
    }

    /// CACHE-POOL criterion 2, another CONVERSATION: a warm session replays its transcript,
    /// so it is never lent to a conversation that did not open it.
    #[test]
    fn another_conversation_is_never_reused() {
        let pool = SessionPool::default();
        let now = Instant::now();
        pool.release(warm("s1", key("claude-code", "opus"), facts(), now), now);
        let mut other = key("claude-code", "opus");
        other.parent_session_id = "une autre conversation".into();
        let claim = pool.claim(&other, &facts(), now);
        assert!(claim.held.is_none());
        assert_eq!(
            claim.refused,
            vec![("s1".to_owned(), Refusal::AnotherConversation)]
        );
    }

    /// CACHE-POOL criterion 3, EXPIRY: past the TTL the provider's cache is gone, so the
    /// entry is dropped and a fresh session is opened. Red if `Expired` is not checked.
    #[test]
    fn an_expired_warm_session_is_dropped_and_never_reused() {
        let pool = SessionPool::new(4, Duration::from_secs(60));
        let opened = Instant::now();
        pool.release(
            warm("s1", key("claude-code", "opus"), facts(), opened),
            opened,
        );
        let later = opened + Duration::from_secs(61);
        let claim = pool.claim(&key("claude-code", "opus"), &facts(), later);
        assert!(claim.held.is_none(), "an expired session is not resumed");
        assert!(pool.is_empty(), "and it is dropped, not kept");
    }

    /// Another engine, another model: the pool is keyed on the pair, so a delegation routed
    /// to another engine opens its own session (the same engine and model are what the
    /// measurement shares).
    #[test]
    fn another_engine_or_model_opens_a_fresh_session() {
        let pool = SessionPool::default();
        let now = Instant::now();
        pool.release(warm("s1", key("claude-code", "opus"), facts(), now), now);
        let engine = pool.claim(&key("codex", "opus"), &facts(), now);
        assert_eq!(
            engine.refused,
            vec![("s1".to_owned(), Refusal::AnotherEngine)]
        );
        let model = pool.claim(&key("claude-code", "haiku"), &facts(), now);
        assert_eq!(
            model.refused,
            vec![("s1".to_owned(), Refusal::AnotherModel)]
        );
    }

    /// Every fact of the opening is compared, one by one: none of them is silently
    /// inherited by the delegation that resumes the session.
    #[test]
    fn every_opening_fact_must_match() {
        let base = facts();
        let cases: Vec<(&str, WarmFacts, Refusal)> = vec![
            (
                "cwd",
                WarmFacts {
                    cwd: "/ailleurs".into(),
                    ..base.clone()
                },
                Refusal::AnotherDirectory,
            ),
            (
                "access",
                WarmFacts {
                    access: SessionAccess::ReadOnly,
                    ..base.clone()
                },
                Refusal::AnotherAccess,
            ),
            (
                "mode",
                WarmFacts {
                    permission_mode: Some("acceptEdits".into()),
                    ..base.clone()
                },
                Refusal::AnotherPermissionMode,
            ),
            (
                "add_dirs",
                WarmFacts {
                    add_dirs: vec!["/autre".into()],
                    ..base.clone()
                },
                Refusal::AnotherDirectories,
            ),
            (
                "workspace",
                WarmFacts {
                    workspace: Some("ws".into()),
                    ..base.clone()
                },
                Refusal::AnotherWorkspace,
            ),
        ];
        for (name, facts, expected) in cases {
            assert_eq!(base.mismatch(&facts), Some(expected), "{name}");
        }
        assert_eq!(base.mismatch(&base), None);
    }

    /// The pool is bounded and a session given back twice is held once (a delegation that
    /// ran twice in the same session must not fill the pool with stale copies).
    #[test]
    fn the_pool_is_bounded_and_keeps_one_entry_per_session() {
        let now = Instant::now();
        let pool = SessionPool::new(2, Duration::from_secs(60));
        for id in ["s1", "s2", "s3"] {
            pool.release(warm(id, key("claude-code", "opus"), facts(), now), now);
        }
        assert_eq!(pool.held(), vec!["s2".to_owned(), "s3".to_owned()]);
        pool.release(warm("s3", key("claude-code", "opus"), facts(), now), now);
        assert_eq!(pool.held(), vec!["s2".to_owned(), "s3".to_owned()]);
    }

    /// A disabled pool keeps nothing: the delegation path then behaves exactly as before.
    #[test]
    fn a_disabled_pool_never_holds_anything() {
        let pool = SessionPool::disabled();
        let now = Instant::now();
        pool.release(warm("s1", key("claude-code", "opus"), facts(), now), now);
        assert!(pool.is_empty());
        assert!(pool
            .claim(&key("claude-code", "opus"), &facts(), now)
            .held
            .is_none());
    }

    /// An abandoned session (its reuse failed) is not offered again.
    #[test]
    fn an_abandoned_session_is_forgotten() {
        let pool = SessionPool::default();
        let now = Instant::now();
        pool.release(warm("s1", key("claude-code", "opus"), facts(), now), now);
        pool.abandon("s1");
        assert!(pool.is_empty());
    }

    /// The fingerprint separates fields (a reordering, or a missing field, changes it).
    #[test]
    fn the_fingerprint_separates_its_fields() {
        assert_eq!(
            fingerprint([Some("a"), Some("b")]),
            fingerprint([Some("a"), Some("b")])
        );
        assert_ne!(
            fingerprint([Some("ab"), Some("")]),
            fingerprint([Some("a"), Some("b")])
        );
        assert_ne!(
            fingerprint([Some("a"), None]),
            fingerprint([Some("a"), Some("")])
        );
    }
}
