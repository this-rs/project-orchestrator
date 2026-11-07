//! RL5 — the ONE door a call goes through before it costs anything.
//!
//! Everything that reaches a provider — a chat turn, a plan-runner agent, a
//! `delegate_task` child, a sub-agent's own session — passes here first, and a
//! long job is not STARTED on an instance that is already limited. One door, so
//! there is one place to read to know why nothing went out.
//!
//! [`Gate::admit`] does two things, in this order, for one call:
//!
//! 0. it READS the quota registry
//!    ([`QuotaRegistry`](crate::chat::provider::quota::QuotaRegistry), when one is
//!    attached): an instance (or a model of it) the provider said is exhausted is
//!    refused, with the reset the registry holds. The gate never WRITES that
//!    registry — it is the single source of truth for "exhausted until X", fed by
//!    the session that saw the refusal — and never opens a breaker on a quota;
//! 1. it consults the breaker (RL4) — an open OVERLOAD pause refuses immediately,
//!    with the instant it ends, and **no counter is touched**;
//! 2. it reserves in the ledger (RL2) — the reservation is all-or-nothing over
//!    every applicable limit, under one lock, so two simultaneous calls cannot
//!    both slip past a ceiling of one.
//!
//! (Read the numbering as the order: quota registry, breaker, counters.)
//!
//! A refusal is a NAMED answer, never an exception: [`Refusal`] carries the kind
//! (a spent quota or an overload — the distinction RL3 draws), the scope as a
//! LABEL, and `retry_at`/`reset_at` as instants. Turning it into an HTTP code or
//! an event belongs to the caller; this module never talks to a provider, opens a
//! session, or reads a secret.
//!
//! # The breaker's subject
//!
//! RL4 keys its machine by (instance, credential REFERENCE) — the very pair the
//! doc of [`Key`] settles. So ONE admission has ONE breaker key:
//! `Key::new(instance, ask.credential_ref)`. Two limits of the same instance used
//! with the same credential therefore share one pause; a call on another
//! credential of the same instance is not stopped by it. An instance- or
//! model-scoped limit keys the instance without a credential (the pair RL4 names
//! when "the instance has no per-key limit"), which is what such a limit pauses.
//!
//! # What the gate refuses to do
//!
//! * It answers ONE instance at a time and never turns a lone available provider
//!   into "no candidate" without a name: every refusal carries the instance, the
//!   kind and the instant it ends.
//! * It never SUBSTITUTES a choice. An explicit provider named by a person or an
//!   agent that is limited gets the refusal — filtering candidates is the
//!   cognitive router's job, and only for a choice IT made.
//! * With NO limit configured, the path is exactly what it was: [`Gate::admit`]
//!   returns a granted, empty reservation without reading the clock, taking a
//!   lock or touching a breaker, and the settle of that reservation does nothing.
//!   A build with no limits pays nothing for this module.
//! * A retry is a new call: the caller asks [`Gate::admit`] again, which reserves
//!   again. Retries therefore CONSUME the counting, as they must.
//!
//! # Settling
//!
//! [`Gate::settle`] gives back what was not consumed and feeds the outcome back
//! into the breaker: an OVERLOAD classified by RL3 opens the pause, and a call that
//! reached the provider closes it. A spent quota does not touch the breaker (the
//! quota registry holds it). A work that was already running when the limit is hit settles
//! normally, then the NEXT admission is refused: the caller stops cleanly, it
//! does not loop.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};

use super::breaker::{Breaker, Decision, Jitter, Key, Policy, Signal, SystemJitter, Transition};
use super::classify::{rfc3339, Classification, OurStanding};
use super::ledger::{Ask, Clock, Denial, Grant, Ledger, Note, Outcome, Reservation, SystemClock};
use super::store::{LedgerStore, MemoryStore};
use crate::chat::provider::quota::{Exhaustion, QuotaRegistry};
use crate::chat::provider::settings::{
    InstanceRecord, RateLimit, RateLimitReport, RateLimitStatus,
};

/// Why a call was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalKind {
    /// A quota is spent: ours (a configured limit) or the provider's own word, as
    /// the quota registry holds it.
    /// Nothing goes out until `reset_at`.
    Quota,
    /// The provider is overloaded (or said nothing readable, which RL3 treats the
    /// same way): the credential is NOT spent, the pause is short.
    Overload,
}

impl RefusalKind {
    /// The word a human, a log line or an event reads.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quota => "quota",
            Self::Overload => "overload",
        }
    }
}

/// A refused call: what refused, and when it makes room again.
///
/// Every field is safe to log or to answer over HTTP: a scope is a LABEL (the
/// word `instance`, a credential REFERENCE, `model <id>`), never a value.
#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    /// The instance that refused.
    pub instance: String,
    /// A spent quota or an overload.
    pub kind: RefusalKind,
    /// The scope that refused, as a LABEL. Never a secret.
    pub scope: String,
    /// When the next attempt may be made, when known.
    pub retry_at: Option<DateTime<Utc>>,
    /// When the quota resets, when this is an exhaustion that knows it.
    pub reset_at: Option<DateTime<Utc>>,
    /// A sentence a human can act on.
    pub message: String,
}

impl Refusal {
    /// How long to wait, in seconds, when an instant is known. What a `Retry-After`
    /// header or an event carries.
    pub fn retry_after_secs(&self, now: DateTime<Utc>) -> Option<u64> {
        self.retry_at
            .map(|at| (at - now).num_seconds().max(0) as u64)
    }

    /// `retry_at` as RL1's report writes instants, when there is one.
    pub fn retry_at_rfc3339(&self) -> Option<String> {
        self.retry_at.map(rfc3339)
    }

    /// `reset_at` as RL1's report writes instants, when there is one.
    pub fn reset_at_rfc3339(&self) -> Option<String> {
        self.reset_at.map(rfc3339)
    }

    /// Whether this refusal says a resource is spent. `false` for an overload and
    /// for an unreadable refusal: nothing may be excluded from a pool because of
    /// one, which is the whole point of the distinction.
    pub fn marks_exhausted(&self) -> bool {
        self.kind == RefusalKind::Quota
    }
}

/// The answer of [`Gate::admit`].
#[derive(Debug, Clone, PartialEq)]
pub enum Admission {
    /// The call may go out. Settle it with [`Gate::settle`]. `probe` is true when
    /// this call is the ONE half-open probe of its breaker: it alone settles a
    /// half-open pause.
    Admitted(Box<Grant>, bool),
    /// The call must not go out.
    Refused(Box<Refusal>),
}

impl Admission {
    /// Whether the call may go out.
    pub fn is_admitted(&self) -> bool {
        matches!(self, Self::Admitted(..))
    }

    /// Whether this admitted call is the half-open probe.
    pub fn is_probe(&self) -> bool {
        matches!(self, Self::Admitted(_, true))
    }

    /// The refusal, when there is one.
    pub fn refusal(&self) -> Option<&Refusal> {
        match self {
            Self::Refused(refusal) => Some(refusal),
            Self::Admitted(..) => None,
        }
    }

    /// The grant, when there is one.
    pub fn grant(self) -> Option<Grant> {
        match self {
            Self::Admitted(grant, _) => Some(*grant),
            Self::Refused(_) => None,
        }
    }
}

/// What a settlement leaves behind: what to warn about, and the pauses it opened.
///
/// A caller that must stop a running job reads [`Self::opened`] (the transition
/// that opened a pause); a caller that only logs reads [`Self::notes`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Aftermath {
    /// What the settlement has to say (an unpriced budget, an exceeded ceiling).
    pub notes: Vec<Note>,
    /// The pauses this settlement opened or closed, oldest first.
    pub transitions: Vec<Transition>,
}

impl Aftermath {
    /// Whether nothing at all happened (the no-limit path).
    pub fn is_empty(&self) -> bool {
        self.notes.is_empty() && self.transitions.is_empty()
    }

    /// The first transition that OPENED a pause, with the instant it ends.
    pub fn opened(&self) -> Option<&Transition> {
        self.transitions
            .iter()
            .find(|t| t.until.is_some() && t.to != super::breaker::Phase::Closed)
    }
}

/// The admission gate: RL2's counters and RL4's breakers behind one door, reading
/// the quota registry.
///
/// Thread-safe: the breakers live under one lock and the ledger under its own,
/// and the two are never held at the same time. Holds no secret and does no I/O
/// beyond the ledger's own store.
#[derive(Debug)]
pub struct Gate {
    ledger: Ledger,
    clock: Arc<dyn Clock>,
    jitter: Arc<dyn Jitter>,
    policy: Policy,
    breakers: Mutex<HashMap<Key, Breaker>>,
    quota: Option<Arc<QuotaRegistry>>,
}

impl Gate {
    /// A gate counting in `store`, pausing with the REAL jitter and `policy`.
    pub fn new(clock: Arc<dyn Clock>, store: Arc<dyn LedgerStore>, policy: Policy) -> Self {
        Self::with_jitter(clock, store, policy, Arc::new(SystemJitter::new()))
    }

    /// [`Self::new`] with the default breaker policy.
    pub fn from_store(clock: Arc<dyn Clock>, store: Arc<dyn LedgerStore>) -> Self {
        Self::new(clock, store, Policy::default())
    }

    /// [`Self::new`] with the jitter injected: a test pins the width of a pause
    /// instead of drawing it.
    pub fn with_jitter(
        clock: Arc<dyn Clock>,
        store: Arc<dyn LedgerStore>,
        policy: Policy,
        jitter: Arc<dyn Jitter>,
    ) -> Self {
        Self {
            ledger: Ledger::new(Arc::clone(&clock), store),
            clock,
            jitter,
            policy,
            breakers: Mutex::new(HashMap::new()),
            quota: None,
        }
    }

    /// Attaches the quota registry the gate READS. The same registry the router and
    /// the sessions share: two readers of one fact cannot disagree.
    pub fn with_quota(mut self, quota: Arc<QuotaRegistry>) -> Self {
        self.quota = Some(quota);
        self
    }

    /// A gate for a process that configured nothing: the real clock, a store that
    /// forgets at exit, the default policy. Identical in behaviour to [`Self::new`]
    /// — it exists so every construction site has an honest default.
    pub fn unlimited() -> Self {
        Self::from_store(Arc::new(SystemClock), Arc::new(MemoryStore::new()))
    }

    /// The counters, for a caller that wants the measured numbers alone.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// The breaker policy in force.
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// The instant this gate reads — the ONE clock of every decision it takes, so
    /// a caller that renders a `retry_after` measures it against the same now.
    pub fn now(&self) -> DateTime<Utc> {
        self.clock.now()
    }

    /// The decision, taken BEFORE a call goes out.
    pub fn admit(&self, instance: &str, limits: &[RateLimit], ask: &Ask) -> Admission {
        if let Some(held) = self.quota_hold(instance, ask) {
            return Admission::Refused(Box::new(refusal_of_quota(instance, &held)));
        }
        if limits.is_empty() {
            // No limit configured: the path is exactly what it was. No clock read,
            // no lock, no breaker — nothing here can cost anything.
            return Admission::Admitted(
                Box::new(Grant {
                    instance: instance.to_string(),
                    takes: Vec::new(),
                }),
                false,
            );
        }
        let key = key_for(instance, ask);
        let probe = {
            let mut breakers = self.lock_breakers();
            let breaker = breakers
                .entry(key.clone())
                .or_insert_with(|| self.build_breaker());
            // ONE `attempt` per admission: a half-open breaker grants its probe to
            // the first caller and to nobody else, so a burst of simultaneous calls
            // cannot fire a burst of probes.
            let decision = breaker.attempt(&key);
            if !decision.is_allowed() {
                return Admission::Refused(Box::new(refusal_of_decision(
                    instance, &key, &decision,
                )));
            }
            decision.is_probe()
        };
        match self.ledger.try_reserve(instance, limits, ask) {
            Reservation::Granted(grant) => Admission::Admitted(Box::new(grant), probe),
            Reservation::Denied(denial) => {
                if probe {
                    // The probe was granted but never leaves: hand it back, or the
                    // breaker would wait out its whole timeout for a call that
                    // does not exist.
                    if let Some(breaker) = self.lock_breakers().get(&key) {
                        breaker.release_probe(&key);
                    }
                }
                Admission::Refused(Box::new(refusal_of_denial(instance, &denial)))
            }
        }
    }

    /// [`Self::admit`] for a stored instance: the limits and the credential
    /// reference are read from the record, so a call site is one line and cannot
    /// disagree with the configuration about which credential it is using.
    pub fn admit_record(&self, record: &InstanceRecord, model: Option<&str>) -> Admission {
        self.admit(&record.id, &record.rate_limit, &ask_for(record, model))
    }

    /// What the call really consumed, and what its outcome says about the pause.
    ///
    /// `classification` is RL3's reading of the refusal, when there was one: it is
    /// what opens the breaker, with the kind that must not blur. It is `None` for a
    /// call that concluded (success, or a failure that said nothing about limits) —
    /// a call that reached the provider closes the pause.
    pub fn settle(
        &self,
        instance: &str,
        limits: &[RateLimit],
        ask: &Ask,
        admission: Admission,
        outcome: Outcome,
        classification: Option<&Classification>,
    ) -> Aftermath {
        let Admission::Admitted(grant, probe) = admission else {
            // A refusal never reserved anything: there is nothing to give back.
            return Aftermath::default();
        };
        if limits.is_empty() {
            return Aftermath::default();
        }
        let notes = self.ledger.settle(Reservation::Granted(*grant), outcome);
        let key = key_for(instance, ask);
        let signal = match classification {
            // `None` for a spent quota: the registry holds it, not a breaker.
            Some(classification) => Signal::of(&classification.verdict),
            // A call that was really sent and came back says the endpoint answers:
            // it is what closes a half-open breaker. A call that never went out
            // changes nothing about the provider.
            None => match outcome {
                Outcome::Consumed { .. } => Some(Signal::Success),
                Outcome::NeverSent => None,
            },
        };
        let mut transitions = Vec::new();
        if let Some(signal) = signal {
            let mut breakers = self.lock_breakers();
            let breaker = breakers
                .entry(key.clone())
                .or_insert_with(|| self.build_breaker());
            if let Some(transition) = breaker.record(&key, signal, probe) {
                transitions.push(transition);
            }
        }
        Aftermath { notes, transitions }
    }

    /// [`Self::settle`] for a stored instance.
    pub fn settle_record(
        &self,
        record: &InstanceRecord,
        model: Option<&str>,
        admission: Admission,
        outcome: Outcome,
        classification: Option<&Classification>,
    ) -> Aftermath {
        self.settle(
            &record.id,
            &record.rate_limit,
            &ask_for(record, model),
            admission,
            outcome,
            classification,
        )
    }

    /// Where our own counting stands for the call about to be classified (RL3's
    /// `Signals::ours`), without reserving anything.
    pub fn standing(&self, instance: &str, limits: &[RateLimit], ask: &Ask) -> OurStanding {
        self.ledger.standing(instance, limits, ask)
    }

    /// The observable state of one subject, as RL7's route answers it. `None` when
    /// no breaker was ever built for that pair (nothing paused, nothing to say).
    pub fn breaker_state(
        &self,
        instance: &str,
        credential: Option<&str>,
    ) -> Option<super::breaker::State> {
        let key = Key::new(instance, credential.map(str::to_string)).ok()?;
        let breakers = self.lock_breakers();
        breakers.get(&key).map(|breaker| breaker.state(&key))
    }

    /// The pauses observed so far, oldest first (RL4's bounded trace).
    pub fn transitions(&self) -> Vec<Transition> {
        let breakers = self.lock_breakers();
        let mut all: Vec<Transition> = breakers.values().flat_map(|s| s.trace()).collect();
        all.sort_by(|a, b| a.at.cmp(&b.at));
        all
    }

    /// The state of one instance's limits: RL1's shape, RL2's numbers, and RL4's
    /// pause. What `GET /chat/providers/{id}/limits` answers.
    pub fn report(&self, record: &InstanceRecord) -> RateLimitReport {
        let mut report = self.ledger.report(record);
        {
            let breakers = self.lock_breakers();
            // The breaker is keyed by the CALL's credential reference. A limit that is
            // not scoped to a credential (instance, model) says nothing about it, so
            // the row falls back to what the record carries — the key the pause was
            // actually taken under.
            let call_credential = ask_for(record, None).credential_ref;
            for state in report.limits.iter_mut() {
                let Ok(key) = Key::new(
                    record.id.clone(),
                    state.credential.clone().or_else(|| call_credential.clone()),
                ) else {
                    continue;
                };
                let Some(breaker) = breakers.get(&key) else {
                    continue;
                };
                let observed = breaker.state(&key);
                if observed.status == RateLimitStatus::Overloaded {
                    state.status = observed.status;
                    if let Some(since) = observed.since {
                        state.since = Some(rfc3339(since));
                    }
                    state.next_retry_at = observed.until.map(rfc3339);
                }
            }
        }
        if let Some(held) = self.quota.as_ref().and_then(|q| q.exhausted(&record.id)) {
            let resets_at = instant_of_ms(held.resets_at_ms).map(rfc3339);
            for state in report.limits.iter_mut() {
                state.status = RateLimitStatus::Limited;
                if held.reset_known {
                    state.resets_at = resets_at.clone();
                }
                state.next_retry_at = resets_at.clone();
            }
        }
        report.status = if report.limits.is_empty() {
            // Nothing configured = nothing that can be limited.
            RateLimitStatus::Ok
        } else {
            report
                .limits
                .iter()
                .map(|state| state.status)
                .fold(RateLimitStatus::Ok, worse_status)
        };
        report
    }

    /// The hold the quota registry has on this call, if any: the whole instance, or
    /// the model asked for.
    fn quota_hold(&self, instance: &str, ask: &Ask) -> Option<Exhaustion> {
        let quota = self.quota.as_ref()?;
        quota.exhausted(instance).or_else(|| {
            ask.model
                .as_deref()
                .and_then(|model| quota.model_exhausted(instance, model))
        })
    }

    fn lock_breakers(&self) -> std::sync::MutexGuard<'_, HashMap<Key, Breaker>> {
        self.breakers.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn build_breaker(&self) -> Breaker {
        Breaker::new(
            Arc::clone(&self.clock),
            Arc::clone(&self.jitter),
            self.policy.clone(),
        )
    }
}

/// The ONE breaker key of an admission: the pair RL4 keys its machine by.
///
/// A credential that is not a REFERENCE (`vault:`/`env:`) is never made a key: the
/// subject falls back to the bare instance, so a stray value in a record can neither
/// reach a log line nor split one instance's pause in two.
pub fn key_for(instance: &str, ask: &Ask) -> Key {
    Key::new(instance, ask.credential_ref.clone()).unwrap_or_else(|_| Key::for_instance(instance))
}

/// What a stored instance tells the ledger about the call about to be made.
///
/// The credential is named by its REFERENCE (`vault:<nom>`, `env:<VAR>`) — what
/// the record already carries — and an empty reference means the instance has
/// none, not "the empty credential".
pub fn ask_for(record: &InstanceRecord, model: Option<&str>) -> Ask {
    Ask {
        credential_ref: Some(record.credential_ref.clone()).filter(|r| !r.trim().is_empty()),
        model: model.map(str::to_string),
        tokens: None,
        cost_usd: None,
    }
}

/// The instant an epoch-ms registry value stands for.
fn instant_of_ms(ms: u64) -> Option<DateTime<Utc>> {
    DateTime::<Utc>::from_timestamp_millis(i64::try_from(ms).ok()?)
}

/// The refusal the quota registry's hold answers: a spent quota, dated when the
/// registry knows the reset (otherwise only the fallback cooldown's end).
fn refusal_of_quota(instance: &str, held: &Exhaustion) -> Refusal {
    let resets = instant_of_ms(held.resets_at_ms);
    let scope = held
        .window
        .clone()
        .unwrap_or_else(|| "instance".to_string());
    Refusal {
        instance: instance.to_string(),
        kind: RefusalKind::Quota,
        scope: scope.clone(),
        retry_at: resets,
        reset_at: resets.filter(|_| held.reset_known),
        message: match resets {
            Some(at) => format!(
                "limit reached for {instance} ({scope}): next attempt at {}",
                rfc3339(at)
            ),
            None => {
                format!("limit reached for {instance} ({scope}): nothing goes out until it resets")
            }
        },
    }
}

/// The refusal an open breaker answers, with the pause it is waiting on.
fn refusal_of_decision(instance: &str, key: &Key, decision: &Decision) -> Refusal {
    let retry_at = decision.until();
    let scope = scope_label(key);
    let message = match retry_at {
        Some(at) => format!(
            "the provider is overloaded for {instance} ({scope}): retry at {}; the key is not \
             exhausted",
            rfc3339(at)
        ),
        None => format!(
            "a probe is already in flight for {instance} ({scope}): the key is not exhausted"
        ),
    };
    Refusal {
        instance: instance.to_string(),
        // An overload, and a half-open probe in flight: never an exhaustion.
        kind: RefusalKind::Overload,
        scope,
        retry_at,
        reset_at: None,
        message,
    }
}

/// The refusal the ledger's own counting answers: always a spent quota.
fn refusal_of_denial(instance: &str, denial: &Denial) -> Refusal {
    Refusal {
        instance: instance.to_string(),
        kind: RefusalKind::Quota,
        scope: denial.scope.clone(),
        retry_at: denial.retry_at,
        reset_at: denial.retry_at,
        message: match denial.retry_at {
            Some(at) => format!(
                "{instance}: {}; next attempt at {}",
                denial.message(),
                rfc3339(at)
            ),
            None => format!(
                "{instance}: {}; nothing goes out until it resets",
                denial.message()
            ),
        },
    }
}

/// A breaker key's scope, as a label a human reads. No field of it is a value.
fn scope_label(key: &Key) -> String {
    key.credential().unwrap_or("instance").to_string()
}

/// The worst of two states, so a report names the one a reader must act on.
fn worse_status(left: RateLimitStatus, right: RateLimitStatus) -> RateLimitStatus {
    use RateLimitStatus::*;
    match (left, right) {
        (Limited, _) | (_, Limited) => Limited,
        (Overloaded, _) | (_, Overloaded) => Overloaded,
        (Unknown, _) | (_, Unknown) => Unknown,
        _ => Ok,
    }
}

#[cfg(test)]
mod tests {
    use super::super::breaker::{FixedJitter, Phase};
    use super::super::classify::from_provider_error;
    use super::super::ledger::ManualClock;
    use super::*;
    use crate::chat::provider::settings::{
        CredentialScope, InstanceScope, RateLimitKind, RateLimitScope, RateLimitUnit,
        RateLimitWindow,
    };
    use std::time::Duration;

    fn clock() -> Arc<ManualClock> {
        Arc::new(ManualClock::at("2026-10-08T12:00:00Z"))
    }

    /// A policy a test can predict: no jitter, a short overload backoff.
    fn policy() -> Policy {
        Policy {
            overload_threshold: 2,
            base_backoff: Duration::from_secs(10),
            max_backoff: Duration::from_secs(60),
            jitter_ratio: 0.0,
            max_retries: 3,
            probe_timeout: Duration::from_secs(30),
            min_cooldown: Duration::from_millis(100),
        }
    }

    fn gate(clock: Arc<ManualClock>) -> Gate {
        Gate::with_jitter(
            clock,
            Arc::new(MemoryStore::new()),
            policy(),
            Arc::new(FixedJitter::new(0.0)),
        )
    }

    fn requests(max: f64) -> RateLimit {
        RateLimit {
            kind: RateLimitKind::Requests,
            window: RateLimitWindow::Unit(RateLimitUnit::Minute),
            calendar: None,
            max,
            scope: RateLimitScope::Instance(InstanceScope::Instance),
        }
    }

    fn on_another_credential(credential: &str, max: f64) -> RateLimit {
        RateLimit {
            scope: RateLimitScope::Credential(CredentialScope {
                credential: credential.to_string(),
            }),
            ..requests(max)
        }
    }

    fn ask() -> Ask {
        Ask {
            credential_ref: Some("vault:deepseek".to_string()),
            model: Some("deepseek-chat".to_string()),
            ..Ask::default()
        }
    }

    fn record(limits: Vec<RateLimit>) -> InstanceRecord {
        InstanceRecord {
            id: "deepseek".into(),
            credential_ref: "vault:deepseek".into(),
            rate_limit: limits,
            ..InstanceRecord::default()
        }
    }

    fn spent_quota_at() -> Classification {
        // Our own ceiling is what explains the refusal: an exhaustion that does not
        // know when it ends (the provider said nothing — it is OUR limit).
        from_provider_error(
            "rate_limited",
            None,
            OurStanding::AtOrAbove,
            "deepseek",
            Utc::now(),
        )
    }

    fn overload() -> Classification {
        from_provider_error(
            "overloaded",
            Some(5_000),
            OurStanding::Under,
            "deepseek",
            Utc::now(),
        )
    }

    #[test]
    fn with_no_limit_nothing_is_read_and_nothing_is_paused() {
        let clock = clock();
        let gate = gate(Arc::clone(&clock));
        let admission = gate.admit("deepseek", &[], &ask());
        assert!(admission.is_admitted());
        assert!(admission.grant().unwrap().takes.is_empty());
        // The breaker was never even built: the path is the one it was before.
        assert!(gate
            .breaker_state("deepseek", Some("vault:deepseek"))
            .is_none());
        // And the clock was not consulted: a gate whose clock never moved still
        // reports the very instant it started at.
        assert_eq!(gate.now(), clock.now());
    }

    #[test]
    fn a_reservation_is_granted_until_the_ceiling_refuses_it() {
        let gate = gate(clock());
        let limits = vec![requests(2.0)];
        assert!(gate.admit("deepseek", &limits, &ask()).is_admitted());
        assert!(gate.admit("deepseek", &limits, &ask()).is_admitted());
        let third = gate.admit("deepseek", &limits, &ask());
        assert!(
            !third.is_admitted(),
            "the ceiling of two must refuse the third"
        );
        let refusal = third.refusal().unwrap();
        assert_eq!(refusal.kind, RefusalKind::Quota);
        assert!(refusal.marks_exhausted());
        assert!(
            refusal.retry_at.is_some(),
            "a refusal must say when it makes room again"
        );
        assert_eq!(refusal.reset_at, refusal.retry_at);
    }

    #[test]
    fn a_denial_takes_nothing_so_the_window_alone_frees_room() {
        let clock = clock();
        let gate = gate(Arc::clone(&clock));
        let limits = vec![requests(1.0)];
        assert!(gate.admit("deepseek", &limits, &ask()).is_admitted());
        assert!(!gate.admit("deepseek", &limits, &ask()).is_admitted());
        clock.advance(Duration::from_secs(61));
        // The window turned over: exactly what it was before the refusal.
        assert!(gate.admit("deepseek", &limits, &ask()).is_admitted());
    }

    #[test]
    fn a_retry_consumes_the_counting_exactly_once() {
        let gate = gate(clock());
        let limits = vec![requests(2.0)];
        let first = gate.admit("deepseek", &limits, &ask());
        let notes = gate.settle(
            "deepseek",
            &limits,
            &ask(),
            first,
            Outcome::Consumed {
                tokens: None,
                cost_usd: None,
            },
            None,
        );
        assert!(notes.is_empty());
        assert_eq!(
            gate.report(&record(limits.clone())).limits[0].used,
            Some(1.0),
            "one call, one unit: settling does not count it twice"
        );
    }

    #[test]
    fn a_call_that_never_went_out_gives_everything_back() {
        let gate = gate(clock());
        let limits = vec![requests(2.0)];
        let admission = gate.admit("deepseek", &limits, &ask());
        let after = gate.settle(
            "deepseek",
            &limits,
            &ask(),
            admission,
            Outcome::NeverSent,
            None,
        );
        assert!(after.is_empty(), "nothing consumed, nothing to report");
        assert_eq!(gate.report(&record(limits)).limits[0].used, Some(0.0));
    }

    #[test]
    fn a_spent_quota_at_settle_never_opens_a_breaker() {
        // MUTATION: mapping a quota verdict to an overload signal in `Signal::of`
        // (so settle opens a pause on it) turns the last two assertions red.
        let gate = gate(clock());
        let limits = vec![requests(5.0)];
        for _ in 0..3 {
            let admission = gate.admit("deepseek", &limits, &ask());
            let after = gate.settle(
                "deepseek",
                &limits,
                &ask(),
                admission,
                Outcome::Consumed {
                    tokens: None,
                    cost_usd: None,
                },
                Some(&spent_quota_at()),
            );
            assert!(after.transitions.is_empty(), "a quota moves no breaker");
        }
        assert!(
            !gate
                .breaker_state("deepseek", Some("vault:deepseek"))
                .is_some_and(|state| state.is_open()),
            "the registry holds a spent quota, not a breaker"
        );
        assert!(gate.admit("deepseek", &limits, &ask()).is_admitted());
    }

    #[test]
    fn an_overload_pauses_without_ever_marking_the_key_spent() {
        let gate = gate(clock());
        let limits = vec![requests(5.0)];
        for _ in 0..2 {
            let admission = gate.admit("deepseek", &limits, &ask());
            gate.settle(
                "deepseek",
                &limits,
                &ask(),
                admission,
                Outcome::Consumed {
                    tokens: None,
                    cost_usd: None,
                },
                Some(&overload()),
            );
        }
        let state = gate
            .breaker_state("deepseek", Some("vault:deepseek"))
            .unwrap();
        assert_eq!(state.phase, Phase::Open);
        assert_eq!(state.status, RateLimitStatus::Overloaded);
        let refusal = gate.admit("deepseek", &limits, &ask());
        let refusal = refusal.refusal().unwrap();
        assert_eq!(refusal.kind, RefusalKind::Overload);
        assert!(!refusal.marks_exhausted());
    }

    #[test]
    fn a_success_that_reaches_the_provider_closes_the_pause() {
        let clock = clock();
        let gate = gate(Arc::clone(&clock));
        let limits = vec![requests(5.0)];
        for _ in 0..2 {
            let admission = gate.admit("deepseek", &limits, &ask());
            gate.settle(
                "deepseek",
                &limits,
                &ask(),
                admission,
                Outcome::Consumed {
                    tokens: None,
                    cost_usd: None,
                },
                Some(&overload()),
            );
        }
        let state = gate
            .breaker_state("deepseek", Some("vault:deepseek"))
            .unwrap();
        assert!(state.is_open());
        // The pause is over: the next caller is the ONE probe.
        let until = state.until.unwrap();
        clock.advance(Duration::from_secs(
            (until - clock.now()).num_seconds() as u64 + 1,
        ));
        // The provider answered: nothing about an overload holds any more.
        let admission = gate.admit("deepseek", &limits, &ask());
        assert!(
            admission.is_admitted(),
            "the pause is over: the probe goes out"
        );
        let after = gate.settle(
            "deepseek",
            &limits,
            &ask(),
            admission,
            Outcome::Consumed {
                tokens: None,
                cost_usd: None,
            },
            None,
        );
        assert!(after.transitions.iter().any(|t| t.to == Phase::Closed));
        assert!(!gate
            .breaker_state("deepseek", Some("vault:deepseek"))
            .unwrap()
            .is_open());
    }

    #[test]
    fn only_one_call_gets_the_probe_when_a_pause_ends() {
        let clock = clock();
        let gate = gate(Arc::clone(&clock));
        let limits = vec![requests(50.0)];
        for _ in 0..2 {
            let admission = gate.admit("deepseek", &limits, &ask());
            gate.settle(
                "deepseek",
                &limits,
                &ask(),
                admission,
                Outcome::Consumed {
                    tokens: None,
                    cost_usd: None,
                },
                Some(&overload()),
            );
        }
        let until = gate
            .breaker_state("deepseek", Some("vault:deepseek"))
            .unwrap()
            .until
            .unwrap();
        clock.advance(Duration::from_secs(
            (until - clock.now()).num_seconds() as u64 + 1,
        ));
        assert!(gate.admit("deepseek", &limits, &ask()).is_admitted());
        let second = gate.admit("deepseek", &limits, &ask());
        assert!(
            !second.is_admitted(),
            "a half-open breaker lets exactly one probe through"
        );
        assert!(second.refusal().unwrap().retry_at.is_none());
    }

    fn consumed() -> Outcome {
        Outcome::Consumed {
            tokens: None,
            cost_usd: None,
        }
    }

    #[test]
    fn only_the_probe_settles_a_half_open_pause() {
        // MUTATION: settling every admission as the probe (`probe` forced true in
        // `settle`) lets the straggler close the pause before the probe is back.
        let clock = clock();
        let gate = gate(Arc::clone(&clock));
        let limits = vec![requests(100.0)];
        let straggler = gate.admit("deepseek", &limits, &ask());
        for _ in 0..2 {
            let admission = gate.admit("deepseek", &limits, &ask());
            gate.settle(
                "deepseek",
                &limits,
                &ask(),
                admission,
                consumed(),
                Some(&overload()),
            );
        }
        let state = gate
            .breaker_state("deepseek", Some("vault:deepseek"))
            .unwrap();
        clock.advance(Duration::from_secs(
            (state.until.unwrap() - clock.now()).num_seconds() as u64 + 1,
        ));
        let probe = gate.admit("deepseek", &limits, &ask());
        assert!(probe.is_probe());
        assert!(!straggler.is_probe());
        // The call admitted BEFORE the pause comes back now: it changes nothing.
        let after = gate.settle("deepseek", &limits, &ask(), straggler, consumed(), None);
        assert!(after.transitions.is_empty());
        assert!(gate
            .breaker_state("deepseek", Some("vault:deepseek"))
            .unwrap()
            .is_open());
        // The probe settles it.
        let after = gate.settle("deepseek", &limits, &ask(), probe, consumed(), None);
        assert!(after.transitions.iter().any(|t| t.to == Phase::Closed));
    }

    #[test]
    fn a_probe_the_counters_refuse_is_handed_back() {
        // MUTATION: dropping the `release_probe` call leaves the breaker half-open
        // with a probe that never left.
        let clock = clock();
        let gate = gate(Arc::clone(&clock));
        // Two per minute: two calls fill it, and the pause (20 s) is too short to
        // refill a whole unit.
        let limits = vec![requests(2.0)];
        for _ in 0..2 {
            let admission = gate.admit("deepseek", &limits, &ask());
            gate.settle(
                "deepseek",
                &limits,
                &ask(),
                admission,
                consumed(),
                Some(&overload()),
            );
        }
        let until = gate
            .breaker_state("deepseek", Some("vault:deepseek"))
            .unwrap()
            .until
            .unwrap();
        clock.advance(Duration::from_secs(
            (until - clock.now()).num_seconds() as u64 + 1,
        ));
        let refused = gate.admit("deepseek", &limits, &ask());
        assert_eq!(refused.refusal().unwrap().kind, RefusalKind::Quota);
        assert_eq!(
            gate.breaker_state("deepseek", Some("vault:deepseek"))
                .unwrap()
                .phase,
            Phase::Open,
            "the probe was handed back, not left out with nobody holding it"
        );
    }

    #[test]
    fn a_stray_value_in_a_record_never_becomes_a_key() {
        // MUTATION: building the key without the fallback would panic or key the
        // value; here the bare instance is the subject.
        let record = InstanceRecord {
            credential_ref: "sk-not-a-reference".into(),
            ..record(vec![requests(5.0)])
        };
        let ask = ask_for(&record, None);
        assert_eq!(key_for("deepseek", &ask), Key::for_instance("deepseek"));
        let gate = gate(clock());
        let report = gate.report(&record);
        assert_eq!(report.limits.len(), 1, "the report still answers");
        assert!(gate
            .breaker_state("deepseek", Some("sk-not-a-reference"))
            .is_none());
    }

    #[test]
    fn a_limit_scoped_to_another_credential_never_pauses_this_one() {
        let gate = gate(clock());
        let limits = vec![on_another_credential("vault:other", 1.0)];
        let admission = gate.admit("deepseek", &limits, &ask());
        assert!(admission.is_admitted());
        assert!(
            admission.grant().unwrap().takes.is_empty(),
            "a limit scoped to another credential does not apply to this call"
        );
        assert!(gate
            .breaker_state("deepseek", Some("vault:other"))
            .is_none());
    }

    #[test]
    fn the_report_fills_rl7s_shape_from_the_counters_and_the_pause() {
        let gate = gate(clock());
        let limits = vec![requests(5.0)];
        let before = gate.report(&record(limits.clone()));
        assert_eq!(before.limits[0].status, RateLimitStatus::Ok);
        assert_eq!(before.limits[0].used, Some(0.0));
        assert_eq!(before.limits[0].remaining, Some(5.0));

        for _ in 0..2 {
            let admission = gate.admit("deepseek", &limits, &ask());
            gate.settle(
                "deepseek",
                &limits,
                &ask(),
                admission,
                Outcome::Consumed {
                    tokens: None,
                    cost_usd: None,
                },
                Some(&overload()),
            );
        }
        let after = gate.report(&record(limits));
        assert_eq!(after.status, RateLimitStatus::Overloaded);
        assert_eq!(after.limits[0].status, RateLimitStatus::Overloaded);
        assert!(
            after.limits[0].next_retry_at.is_some(),
            "an open pause says when the next attempt may be made"
        );
        assert!(after.limits[0].since.is_some());
        assert_eq!(after.limits[0].used, Some(2.0));
    }

    fn registry(clock: &Arc<ManualClock>) -> Arc<QuotaRegistry> {
        Arc::new(QuotaRegistry::new().with_clock(clock.clone()))
    }

    #[test]
    fn an_instance_the_registry_holds_exhausted_is_refused_with_a_name_and_its_reset() {
        // MUTATION: dropping the `quota_hold` check at the top of `admit` turns
        // every assertion here red (the call is admitted).
        let clock = clock();
        let quota = registry(&clock);
        let gate = gate(Arc::clone(&clock)).with_quota(quota.clone());
        let reset_ms = quota.now_ms() + 3_600_000;
        quota.mark_rate_limited("deepseek", Some(reset_ms), None);

        // With NO configured limit: the registry alone names the refusal.
        let admission = gate.admit("deepseek", &[], &ask());
        let refusal = admission.refusal().expect("refused").clone();
        assert_eq!(refusal.kind, RefusalKind::Quota);
        assert_eq!(refusal.instance, "deepseek");
        assert!(refusal.marks_exhausted());
        assert_eq!(refusal.scope, "instance");
        assert_eq!(refusal.reset_at, instant_of_ms(reset_ms));
        assert_eq!(refusal.retry_at, refusal.reset_at);
        assert!(refusal.message.contains("deepseek"));

        // The reset passes: the instance is back without anyone doing anything.
        clock.advance(Duration::from_secs(3_601));
        assert!(gate.admit("deepseek", &[], &ask()).is_admitted());
    }

    #[test]
    fn a_hold_the_registry_could_not_date_is_a_refusal_without_a_reset() {
        let clock = clock();
        let quota = registry(&clock);
        let gate = gate(Arc::clone(&clock)).with_quota(quota.clone());
        quota.mark_out_of_credit("deepseek");
        let admission = gate.admit("deepseek", &[], &ask());
        let refusal = admission.refusal().expect("refused");
        assert_eq!(refusal.kind, RefusalKind::Quota);
        assert_eq!(
            refusal.reset_at, None,
            "the fallback cooldown is not a reset"
        );
        assert!(refusal.retry_at.is_some(), "but it does say when to retry");
    }

    #[test]
    fn a_model_hold_refuses_that_model_only() {
        let clock = clock();
        let quota = registry(&clock);
        let gate = gate(Arc::clone(&clock)).with_quota(quota.clone());
        quota.mark_model_rate_limited(
            "deepseek",
            "deepseek-reasoner",
            Some(quota.now_ms() + 600_000),
            None,
        );
        let held = Ask {
            model: Some("deepseek-reasoner".to_string()),
            ..ask()
        };
        assert!(!gate.admit("deepseek", &[], &held).is_admitted());
        assert!(
            gate.admit("deepseek", &[], &ask()).is_admitted(),
            "the instance's other models stay eligible"
        );
    }

    #[test]
    fn another_instance_and_a_gate_without_a_registry_are_never_refused() {
        // The mono-provider rule: a lone available provider is not turned into a
        // refusal by somebody else's hold, nor by the absence of a registry.
        let clock = clock();
        let quota = registry(&clock);
        quota.mark_rate_limited("other", Some(quota.now_ms() + 600_000), None);
        let with = gate(Arc::clone(&clock)).with_quota(quota);
        assert!(with.admit("deepseek", &[], &ask()).is_admitted());
        let without = gate(clock);
        assert!(without.admit("deepseek", &[], &ask()).is_admitted());
    }

    #[test]
    fn the_gate_reads_the_registry_and_never_writes_it() {
        let clock = clock();
        let quota = registry(&clock);
        let gate = gate(Arc::clone(&clock)).with_quota(quota.clone());
        let limits = vec![requests(5.0)];
        let admission = gate.admit("deepseek", &limits, &ask());
        gate.settle(
            "deepseek",
            &limits,
            &ask(),
            admission,
            Outcome::Consumed {
                tokens: None,
                cost_usd: None,
            },
            Some(&spent_quota_at()),
        );
        assert!(
            quota.exhausted("deepseek").is_none(),
            "marking an account exhausted is the session's job, not the gate's"
        );
    }

    #[test]
    fn the_report_says_limited_with_the_registrys_reset() {
        let clock = clock();
        let quota = registry(&clock);
        let gate = gate(Arc::clone(&clock)).with_quota(quota.clone());
        let limits = vec![requests(5.0)];
        let reset_ms = quota.now_ms() + 600_000;
        quota.mark_rate_limited("deepseek", Some(reset_ms), None);
        let report = gate.report(&record(limits.clone()));
        assert_eq!(report.status, RateLimitStatus::Limited);
        assert_eq!(
            report.limits[0].resets_at,
            instant_of_ms(reset_ms).map(rfc3339)
        );
        // An undated hold still says limited, without inventing a reset.
        quota.clear("deepseek");
        quota.mark_out_of_credit("deepseek");
        let undated = gate.report(&record(limits));
        assert_eq!(undated.limits[0].status, RateLimitStatus::Limited);
        assert_eq!(undated.limits[0].resets_at, None);
        assert!(undated.limits[0].next_retry_at.is_some());
    }

    #[test]
    fn an_instance_with_no_limit_stays_ok_even_after_another_one_paused() {
        let gate = gate(clock());
        let limits = vec![requests(5.0)];
        let admission = gate.admit("deepseek", &limits, &ask());
        gate.settle(
            "deepseek",
            &limits,
            &ask(),
            admission,
            Outcome::Consumed {
                tokens: None,
                cost_usd: None,
            },
            Some(&overload()),
        );
        let report = gate.report(&record(Vec::new()));
        assert_eq!(report.status, RateLimitStatus::Ok);
        assert!(report.limits.is_empty());
    }

    #[test]
    fn the_ask_names_the_credential_reference_and_never_a_value() {
        let record = record(vec![requests(1.0)]);
        let ask = ask_for(&record, Some("deepseek-chat"));
        assert_eq!(ask.credential_ref.as_deref(), Some("vault:deepseek"));
        assert_eq!(ask.model.as_deref(), Some("deepseek-chat"));

        let anonymous = InstanceRecord {
            credential_ref: String::new(),
            ..record
        };
        assert_eq!(
            ask_for(&anonymous, None).credential_ref,
            None,
            "an empty reference is no credential, not the empty credential"
        );
    }

    #[test]
    fn the_breaker_key_is_the_instance_and_the_credential_reference() {
        assert_eq!(
            key_for("deepseek", &ask()),
            Key::new("deepseek", Some("vault:deepseek".to_string())).unwrap()
        );
        assert_eq!(
            key_for("deepseek", &Ask::default()),
            Key::for_instance("deepseek")
        );
    }

    #[test]
    fn a_refusal_never_carries_a_secret_and_says_what_a_caller_needs() {
        let gate = gate(clock());
        let limits = vec![requests(1.0)];
        assert!(gate.admit("deepseek", &limits, &ask()).is_admitted());
        let refusal = gate.admit("deepseek", &limits, &ask());
        let refusal = refusal.refusal().unwrap().clone();
        assert!(refusal.message.contains("deepseek"));
        assert!(!refusal.message.contains("sk-"));
        assert_eq!(refusal.scope, "instance");
        let now = gate.now();
        assert!(
            refusal.retry_after_secs(now).unwrap() <= 60,
            "a minute window never asks for more than a minute"
        );
        assert!(refusal.retry_at_rfc3339().is_some());
        assert!(refusal.reset_at_rfc3339().is_some());
    }

    #[test]
    fn settling_a_refusal_has_nothing_to_give_back() {
        let gate = gate(clock());
        let limits = vec![requests(1.0)];
        assert!(gate.admit("deepseek", &limits, &ask()).is_admitted());
        let refused = gate.admit("deepseek", &limits, &ask());
        let after = gate.settle(
            "deepseek",
            &limits,
            &ask(),
            refused,
            Outcome::Consumed {
                tokens: None,
                cost_usd: None,
            },
            Some(&spent_quota_at()),
        );
        assert!(after.is_empty());
        assert_eq!(
            gate.report(&record(limits)).limits[0].used,
            Some(1.0),
            "settling a refusal never moves a counter"
        );
    }
}
