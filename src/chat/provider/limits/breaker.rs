//! RL4 — the breaker and the cooldown, one state machine per (instance, key).
//!
//! RL3 reads a refusal and answers "spent quota" or "overload". RL2 counts. This
//! module holds the third thing the pair needs: **what to do next**, and for how
//! long. It is a PURE state machine — no I/O, no event bus, no secret, no clock of
//! its own (the instant comes from an injected [`Clock`], exactly as the ledger's)
//! — so its whole behaviour is a truth table a test can drive.
//!
//! # The three phases
//!
//! ```text
//!            overload threshold reached, or a quota is spent
//!   Closed  ────────────────────────────────────────────────▶  Open
//!      ▲                                                        │
//!      │                                       cooldown elapsed │
//!      │  the one probe succeeded                               ▼
//!      └──────────────────────────────────────────────────  HalfOpen
//! ```
//!
//! * **Closed** — calls go out. Consecutive overloads are counted; a quota opens
//!   immediately.
//! * **Open** — every call is refused until `until`. The instant a new attempt may
//!   be made is [`Decision::Refuse::until`], so a caller never guesses.
//! * **HalfOpen** — EXACTLY ONE probe is in flight. Every further `attempt` is
//!   refused until that probe is settled by `record`, or abandoned after
//!   `probe_timeout` (a cancelled turn whose outcome never comes back must not
//!   wedge the breaker forever).
//!
//! # The two cooldowns, which are not the same thing
//!
//! * **A spent quota** ([`Signal::Quota`]) opens until `reset_at` **when the
//!   provider said when** — never one second earlier, because "the reset is at
//!   18:04" is the provider's own statement about its own resource. With no known
//!   reset it opens for `quota_unknown_cooldown`, a bounded guess, never "forever"
//!   and never "one attempt to see".
//! * **An overload** ([`Signal::Overload`]) opens for a SHORT, exponential backoff
//!   with jitter, capped at `max_backoff`. It NEVER marks the key as spent: our
//!   own ceiling is not what refused us, so nothing about the credential changed.
//!   This is the sentence the whole module exists to make true.
//!
//! A late `Success` reported while `Open` on a quota is DROPPED: the credential is
//! spent whatever one straggler call did. The same success while `Open` on an
//! overload closes the breaker — there, a call that got through is the proof we
//! were waiting for.
//!
//! # Retries consume the count — delegated, not reimplemented
//!
//! This module keeps ONE counter that is policy, not consumption: how many retries
//! this episode has already spent (`max_retries`, which also saturates the
//! exponent of the backoff). It counts no request, no token, no dollar — that is
//! [`super::ledger`]'s job and stays there. A caller that sends an attempt this
//! module allowed still goes through [`Ledger::try_reserve`](super::ledger::Ledger)
//! first, so a retry is counted exactly once, by the ledger. RL5 is the slice that
//! wires that call site; this module only answers the question.
//!
//! # What a caller sees
//!
//! Every state change is a [`Transition`], appended to a bounded trace readable
//! with [`Breaker::trace`]. [`State`] is what RL7's route answers: the phase, RL1's
//! [`RateLimitStatus`], since when, and when the next attempt may be made. Both
//! carry the instance id and the credential REFERENCE (`vault:<nom>`,
//! `env:<VAR>`) — never a credential value, because no field of this module can
//! hold one.
//!
//! # Boundary with the other breakers
//!
//! `src/chat/provider/circuit_breaker.rs` is a generic HEALTH breaker: it counts
//! whatever failure the caller calls a failure, over a whole instance, and knows
//! nothing about quotas. This one is rate-limit specific, keyed by
//! (instance, credential REFERENCE), and its two cooldowns are the two RL3 reads.
//! They are not merged: that one answers "is this instance healthy", this one
//! answers "may this key be spent right now, and when may it be tried again".

use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};

use super::classify::{rfc3339, Verdict};
use super::ledger::Clock;
use crate::chat::provider::settings::RateLimitStatus;

/// How many transitions the breaker remembers for observation. A diagnostic ring,
/// not a ledger: the oldest is dropped, never the newest.
const TRACE_CAP: usize = 512;

// ============================================================================
// The subject of the machine
// ============================================================================

/// One breakable subject: an instance, and the credential REFERENCE it is used
/// with (`vault:<nom>`, `env:<VAR>`, or none when the instance has no per-key
/// limit). A reference, never a value — that is the only reason this type is safe
/// to key a map by, log, or answer over HTTP.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Key {
    /// The provider instance (`nim`, `deepseek`, an alias…).
    pub instance: String,
    /// The credential REFERENCE, when the breaker is per key.
    pub credential: Option<String>,
}

impl Key {
    /// A key for one credential of one instance.
    pub fn new(instance: impl Into<String>, credential: Option<String>) -> Self {
        Self {
            instance: instance.into(),
            credential,
        }
    }

    /// A key for an instance used without a named credential.
    pub fn for_instance(instance: impl Into<String>) -> Self {
        Self::new(instance, None)
    }
}

/// The phase of the machine. The name RL10's diagram and RL7's status draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Calls go out.
    Closed,
    /// Refused until the cooldown ends.
    Open,
    /// Exactly one probe is in flight.
    HalfOpen,
}

// ============================================================================
// What a call reported, and what the breaker answers
// ============================================================================

/// What one finished call reported, as RL3 read it (or a plain success).
///
/// [`Signal::of`] is the ONE place the RL3 vocabulary is translated, so a change
/// in [`Verdict`] is one local edit and no rule is duplicated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// The call went through.
    Success,
    /// The provider (or our own configured ceiling) states the quota is spent.
    Quota {
        /// When it resets, when the provider said when.
        reset_at: Option<DateTime<Utc>>,
    },
    /// Overloaded, throttling, or unreadable: retry with a bounded backoff, and
    /// never mark the key as spent. `hint` is the provider's `Retry-After`, a
    /// floor that is never shortened.
    Overload {
        /// The delay the provider asked for, when it asked for one.
        hint: Option<Duration>,
    },
}

impl Signal {
    /// The signal an RL3 reading stands for. A refusal this module cannot read is
    /// an overload, never an exhaustion — the contradiction rule of RL3, kept on
    /// this side of the boundary too.
    pub fn of(verdict: &Verdict) -> Self {
        match verdict {
            Verdict::QuotaExhausted { reset_at } => Self::Quota {
                reset_at: *reset_at,
            },
            Verdict::Overloaded { retry_after } => Self::Overload { hint: *retry_after },
            Verdict::Unknown => Self::Overload { hint: None },
        }
    }
}

/// What the breaker answers when asked whether a call may go out.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// Send it. `probe` is true when this is the single half-open probe, so a
    /// caller can tell a normal call from the one that will decide the phase.
    Allow {
        /// Whether this is the one half-open probe.
        probe: bool,
    },
    /// Do not send it. `until` is the earliest instant a new attempt may be
    /// granted (null when a probe is in flight and its outcome is what we wait for).
    Refuse {
        /// When a new attempt may be granted, if it is known.
        until: Option<DateTime<Utc>>,
        /// RL1's status for the answer: `limited` (quota) or `overloaded`.
        status: RateLimitStatus,
    },
}

impl Decision {
    /// Whether a call may go out.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Decision::Allow { .. })
    }

    /// Whether this allowed call is the single half-open probe.
    pub fn is_probe(&self) -> bool {
        matches!(self, Decision::Allow { probe: true })
    }

    /// When a refused call may be tried again, if the breaker knows.
    pub fn until(&self) -> Option<DateTime<Utc>> {
        match self {
            Decision::Allow { .. } => None,
            Decision::Refuse { until, .. } => *until,
        }
    }

    /// RL1's status for this decision (`ok` when allowed).
    pub fn status(&self) -> RateLimitStatus {
        match self {
            Decision::Allow { .. } => RateLimitStatus::Ok,
            Decision::Refuse { status, .. } => *status,
        }
    }
}

// ============================================================================
// Observations
// ============================================================================

/// Why a transition happened. A closed vocabulary, so a log line or a WS event
/// can be written without ever quoting a header, a body or a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// A quota is spent: open until `until`.
    Quota,
    /// The provider is overloaded: a short cooldown, the key NOT marked spent.
    Overload,
    /// The one half-open probe was granted.
    ProbeGranted,
    /// The probe succeeded: closed again.
    Recovered,
    /// The probe never settled: counted as a failure.
    ProbeAbandoned,
}

/// One state change, for the log and for RL7's `provider_rate_limit` event.
/// Carries the credential REFERENCE, never a value.
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    /// The instance whose breaker moved.
    pub instance: String,
    /// The credential REFERENCE, when the breaker is per key.
    pub credential: Option<String>,
    /// The phase left.
    pub from: Phase,
    /// The phase entered.
    pub to: Phase,
    /// When it happened.
    pub at: DateTime<Utc>,
    /// RL1's status after the move.
    pub status: RateLimitStatus,
    /// Why it happened.
    pub reason: Reason,
    /// The cooldown's end, when the move opened one.
    pub until: Option<DateTime<Utc>>,
}

impl Transition {
    fn new(
        key: &Key,
        from: Phase,
        to: Phase,
        at: DateTime<Utc>,
        status: RateLimitStatus,
        reason: Reason,
        until: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            instance: key.instance.clone(),
            credential: key.credential.clone(),
            from,
            to,
            at,
            status,
            reason,
            until,
        }
    }

    /// `until` as RL1's report writes it, RFC 3339, when there is one.
    pub fn until_rfc3339(&self) -> Option<String> {
        self.until.map(rfc3339)
    }

    /// `at` as RL1's report writes it, RFC 3339.
    pub fn at_rfc3339(&self) -> String {
        rfc3339(self.at)
    }
}

/// The observable state of one subject — the shape RL7 answers with, filled by
/// this machine instead of left `unknown`. No field can hold a secret: the only
/// credential-shaped one is the REFERENCE the caller keyed us by.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    /// The instance.
    pub instance: String,
    /// The credential REFERENCE, never a value.
    pub credential: Option<String>,
    /// Which phase the machine is in.
    pub phase: Phase,
    /// RL1's status: `ok`, `limited` (quota) or `overloaded`.
    pub status: RateLimitStatus,
    /// Since when the current phase holds.
    pub since: Option<DateTime<Utc>>,
    /// The current cooldown's end, when one is running.
    pub until: Option<DateTime<Utc>>,
    /// When the next attempt may be made (the cooldown's end).
    pub next_retry_at: Option<DateTime<Utc>>,
    /// Whether the key is MARKED spent. Only a quota ever sets this.
    pub exhausted: bool,
    /// Consecutive overloads observed while closed.
    pub overloads: u32,
    /// Retries spent since the last success, bounded by [`Policy::max_retries`].
    pub retries: u32,
}

impl State {
    /// Whether the key is currently refused.
    pub fn is_open(&self) -> bool {
        self.phase != Phase::Closed
    }
}

// ============================================================================
// Policy and jitter
// ============================================================================

/// The knobs of the machine. Defaults are the ones the plan's RL4 describes: a
/// short overload cooldown, a threshold of three consecutive overloads, and a
/// bounded retry count.
#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    /// Consecutive overloads, while closed, that open the breaker.
    pub overload_threshold: u32,
    /// The first overload cooldown, before the exponent and the jitter.
    pub base_backoff: Duration,
    /// The ceiling of an overload cooldown. Our backoff never grows past it.
    pub max_backoff: Duration,
    /// The fraction of the cooldown that jitter may remove, in `[0, 1]`. Zero
    /// means a fixed backoff; the default spreads retries so they do not all hit
    /// the provider on the same second (a thundering herd re-saturates the window
    /// instantly).
    pub jitter_ratio: f64,
    /// Retries spent per episode before the exponent saturates. Bounds the rate of
    /// retries, never the possibility of recovering: a probe is still granted
    /// after each cooldown.
    pub max_retries: u32,
    /// The cooldown of a quota the provider did NOT date. A bounded guess: neither
    /// "retry at once" nor "never again".
    pub quota_unknown_cooldown: Duration,
    /// How long the one half-open probe may stay unanswered before it is counted
    /// as a failure.
    pub probe_timeout: Duration,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            overload_threshold: 3,
            base_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(60),
            jitter_ratio: 0.5,
            max_retries: 8,
            quota_unknown_cooldown: Duration::from_secs(300),
            probe_timeout: Duration::from_secs(30),
        }
    }
}

/// Where the jitter comes from. Injected so a test is a fixture, not a coin toss.
pub trait Jitter: Send + Sync + Debug {
    /// A fraction in `[0, 1)`.
    fn frac(&self) -> f64;
}

/// A deterministic source: the same seed replays the same sequence. Tests use it
/// so a jittered cooldown is still an exact assertion.
#[derive(Debug)]
pub struct SeededJitter {
    state: Mutex<u64>,
}

impl SeededJitter {
    /// A generator seeded by `seed` (SplitMix64, as a `u64`).
    pub fn new(seed: u64) -> Self {
        Self {
            state: Mutex::new(seed ^ 0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&self) -> u64 {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl Jitter for SeededJitter {
    fn frac(&self) -> f64 {
        // 53 significant bits: exactly representable in an f64.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A jitter that always returns the same fraction. For tests, and for a
/// deployment that deliberately wants a fixed backoff (`jitter_ratio` zero makes
/// the value irrelevant anyway).
#[derive(Debug, Clone, Copy)]
pub struct FixedJitter {
    frac: f64,
}

impl FixedJitter {
    /// A jitter pinned to `frac`, clamped to `[0, 1]`.
    pub fn new(frac: f64) -> Self {
        Self {
            frac: if frac.is_finite() {
                frac.clamp(0.0, 1.0)
            } else {
                0.0
            },
        }
    }
}

impl Jitter for FixedJitter {
    fn frac(&self) -> f64 {
        self.frac
    }
}

/// The jitter of a running process: seeded from the wall clock, the pid and the
/// address of a stack value, then SplitMix64.
#[derive(Debug)]
pub struct SystemJitter {
    inner: SeededJitter,
}

impl SystemJitter {
    /// A generator seeded from the process itself.
    pub fn new() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let here = (&nanos as *const u64) as usize as u64;
        Self {
            inner: SeededJitter::new(nanos ^ (u64::from(std::process::id()) << 32) ^ here),
        }
    }
}

impl Default for SystemJitter {
    fn default() -> Self {
        Self::new()
    }
}

impl Jitter for SystemJitter {
    fn frac(&self) -> f64 {
        self.inner.frac()
    }
}

// ============================================================================
// The machine
// ============================================================================

/// One subject's live state.
#[derive(Debug, Clone)]
struct Entry {
    phase: Phase,
    /// The cooldown's end while open.
    until: Option<DateTime<Utc>>,
    /// Whether the key is MARKED spent. Only `Signal::Quota` ever sets it.
    exhausted: bool,
    /// Consecutive overloads while closed.
    overloads: u32,
    /// Retries spent since the last success.
    retries: u32,
    /// When the in-flight half-open probe was granted.
    probe_since: Option<DateTime<Utc>>,
    /// Since when the current phase holds.
    since: Option<DateTime<Utc>>,
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            phase: Phase::Closed,
            until: None,
            exhausted: false,
            overloads: 0,
            retries: 0,
            probe_since: None,
            since: None,
        }
    }
}

impl Entry {
    fn status(&self) -> RateLimitStatus {
        match self.phase {
            Phase::Closed => RateLimitStatus::Ok,
            Phase::Open if self.exhausted => RateLimitStatus::Limited,
            Phase::Open => RateLimitStatus::Overloaded,
            Phase::HalfOpen => RateLimitStatus::Overloaded,
        }
    }

    /// Reaches `Closed` and forgets the episode. The exhausted mark can only be
    /// cleared here: a quota is only over when a probe got through.
    fn close(&mut self, now: DateTime<Utc>) {
        self.phase = Phase::Closed;
        self.until = None;
        self.exhausted = false;
        self.overloads = 0;
        self.retries = 0;
        self.probe_since = None;
        self.since = Some(now);
    }

    /// Reaches `Open` for `until`. Deliberately blind to `exhausted`: the overload
    /// path must not be able to touch that mark, in either direction.
    fn open(&mut self, until: DateTime<Utc>, now: DateTime<Utc>) {
        self.phase = Phase::Open;
        self.until = Some(until);
        self.probe_since = None;
        self.since = Some(now);
    }
}

/// The breaker: every subject's state, and the decision to send or not.
///
/// Thread-safe (one map behind a mutex), like the ledger. It holds no secret and
/// performs no I/O.
#[derive(Debug)]
pub struct Breaker {
    clock: Arc<dyn Clock>,
    jitter: Arc<dyn Jitter>,
    policy: Policy,
    entries: Mutex<HashMap<Key, Entry>>,
    trace: Mutex<VecDeque<Transition>>,
}

impl Breaker {
    /// A breaker with an explicit clock, jitter source and policy.
    pub fn new(clock: Arc<dyn Clock>, jitter: Arc<dyn Jitter>, policy: Policy) -> Self {
        Self {
            clock,
            jitter,
            policy,
            entries: Mutex::new(HashMap::new()),
            trace: Mutex::new(VecDeque::new()),
        }
    }

    /// A breaker with the default policy.
    pub fn with_defaults(clock: Arc<dyn Clock>, jitter: Arc<dyn Jitter>) -> Self {
        Self::new(clock, jitter, Policy::default())
    }

    /// The policy in force.
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// May a call go out right now? Moving out of `Open` grants the one probe.
    ///
    /// The instant is read from the injected clock; nothing here sleeps.
    pub fn attempt(&self, key: &Key) -> Decision {
        let now = self.clock.now();
        let policy = &self.policy;
        let mut moved = None;
        let decision = {
            let mut entries = self.lock_entries();
            let e = entries.entry(key.clone()).or_default();
            match e.phase {
                Phase::Closed => Decision::Allow { probe: false },
                Phase::Open => {
                    let ready = e.until.map(|until| now >= until).unwrap_or(true);
                    if ready {
                        // The cooldown has ended: the quota it held has lapsed, and
                        // one probe — never a burst — may now go out.
                        e.phase = Phase::HalfOpen;
                        e.exhausted = false;
                        e.probe_since = Some(now);
                        moved = Some(Transition::new(
                            key,
                            Phase::Open,
                            Phase::HalfOpen,
                            now,
                            RateLimitStatus::Overloaded,
                            Reason::ProbeGranted,
                            None,
                        ));
                        Decision::Allow { probe: true }
                    } else {
                        Decision::Refuse {
                            until: e.until,
                            status: e.status(),
                        }
                    }
                }
                Phase::HalfOpen => match e.probe_since {
                    // The probe was never settled (a cancelled turn, a dropped
                    // future): count it as a failure instead of waiting forever.
                    Some(since) if elapsed(since, now) >= policy.probe_timeout => {
                        e.retries = e.retries.saturating_add(1).min(policy.max_retries.max(1));
                        let delay = overload_delay(policy, e.retries, None, self.jitter.frac());
                        let until = now + chrono_d(delay);
                        e.open(until, now);
                        let status = e.status();
                        moved = Some(Transition::new(
                            key,
                            Phase::HalfOpen,
                            Phase::Open,
                            now,
                            status,
                            Reason::ProbeAbandoned,
                            Some(until),
                        ));
                        Decision::Refuse {
                            until: Some(until),
                            status,
                        }
                    }
                    // One probe at a time: nothing else goes out while it is out.
                    Some(_) => Decision::Refuse {
                        until: None,
                        status: e.status(),
                    },
                    None => {
                        e.probe_since = Some(now);
                        moved = Some(Transition::new(
                            key,
                            Phase::HalfOpen,
                            Phase::HalfOpen,
                            now,
                            RateLimitStatus::Overloaded,
                            Reason::ProbeGranted,
                            None,
                        ));
                        Decision::Allow { probe: true }
                    }
                },
            }
        };
        if let Some(t) = moved {
            self.push_trace(t);
        }
        decision
    }

    /// Tells the breaker how the last call ended. Returns the transition it caused,
    /// when it caused one.
    pub fn record(&self, key: &Key, signal: Signal) -> Option<Transition> {
        let now = self.clock.now();
        let policy = &self.policy;
        let transition = {
            let mut entries = self.lock_entries();
            let e = entries.entry(key.clone()).or_default();
            let from = e.phase;
            match signal {
                Signal::Success => {
                    // A straggler call that got through does not refill a quota the
                    // provider itself declared spent: the mark stands until its reset.
                    if from == Phase::Open && e.exhausted {
                        return None;
                    }
                    e.close(now);
                    if from == Phase::Closed {
                        None
                    } else {
                        Some(Transition::new(
                            key,
                            from,
                            Phase::Closed,
                            now,
                            RateLimitStatus::Ok,
                            Reason::Recovered,
                            None,
                        ))
                    }
                }
                Signal::Quota { reset_at } => {
                    // The provider's own instant, to the second: never earlier.
                    let until =
                        reset_at.unwrap_or_else(|| now + chrono_d(policy.quota_unknown_cooldown));
                    // A spent quota is a different cause than an overload: the
                    // retry run it interrupts is over, not paused.
                    e.overloads = 0;
                    e.retries = 0;
                    e.exhausted = true;
                    e.open(until, now);
                    Some(Transition::new(
                        key,
                        from,
                        Phase::Open,
                        now,
                        RateLimitStatus::Limited,
                        Reason::Quota,
                        Some(until),
                    ))
                }
                Signal::Overload { hint } => {
                    e.overloads = e.overloads.saturating_add(1);
                    e.retries = e.retries.saturating_add(1).min(policy.max_retries.max(1));
                    let opens =
                        e.overloads >= policy.overload_threshold.max(1) || from != Phase::Closed;
                    if opens {
                        let delay = overload_delay(policy, e.retries, hint, self.jitter.frac());
                        let until = now + chrono_d(delay);
                        e.open(until, now);
                        Some(Transition::new(
                            key,
                            from,
                            Phase::Open,
                            now,
                            RateLimitStatus::Overloaded,
                            Reason::Overload,
                            Some(until),
                        ))
                    } else {
                        None
                    }
                }
            }
        };
        if let Some(t) = &transition {
            self.push_trace(t.clone());
        }
        transition
    }

    /// The observable state of one subject, as RL7 answers it.
    pub fn state(&self, key: &Key) -> State {
        let entries = self.lock_entries();
        let (phase, status, since, until, exhausted, overloads, retries) = match entries.get(key) {
            Some(e) => (
                e.phase,
                e.status(),
                e.since,
                e.until,
                e.exhausted,
                e.overloads,
                e.retries,
            ),
            None => (Phase::Closed, RateLimitStatus::Ok, None, None, false, 0, 0),
        };
        State {
            instance: key.instance.clone(),
            credential: key.credential.clone(),
            phase,
            status,
            since,
            until,
            next_retry_at: if phase == Phase::Open { until } else { None },
            exhausted,
            overloads,
            retries,
        }
    }

    /// The transitions observed so far, oldest first (bounded by [`TRACE_CAP`]).
    /// This is the "event/trace" a decision can be explained with.
    pub fn trace(&self) -> Vec<Transition> {
        self.lock_trace().iter().cloned().collect()
    }

    /// Takes the observed transitions, leaving the trace empty.
    pub fn take_trace(&self) -> Vec<Transition> {
        self.lock_trace().drain(..).collect()
    }

    fn lock_entries(&self) -> std::sync::MutexGuard<'_, HashMap<Key, Entry>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn lock_trace(&self) -> std::sync::MutexGuard<'_, VecDeque<Transition>> {
        self.trace.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn push_trace(&self, transition: Transition) {
        let mut trace = self.lock_trace();
        if trace.len() >= TRACE_CAP {
            trace.pop_front();
        }
        trace.push_back(transition);
    }
}

// ============================================================================
// The arithmetic
// ============================================================================

/// The jittered, capped cooldown of an overload.
///
/// `retries` saturates at [`Policy::max_retries`], so the exponent saturates with
/// it; the ceiling [`Policy::max_backoff`] then bounds the result whatever the
/// retry count. The jitter keeps `jitter_ratio` of the cooldown free, so two keys
/// limited the same second do not come back the same second. A provider's
/// `Retry-After` is a FLOOR: it is never shortened by our own ceiling.
fn overload_delay(policy: &Policy, retries: u32, hint: Option<Duration>, frac: f64) -> Duration {
    let steps = retries.max(1).min(policy.max_retries.max(1));
    let factor = 2u32.saturating_pow(steps - 1);
    let raw = policy.base_backoff.saturating_mul(factor);
    let capped = raw.min(policy.max_backoff);
    let ratio = policy.jitter_ratio.clamp(0.0, 1.0);
    let hi = capped.as_secs_f64();
    let lo = hi * (1.0 - ratio);
    let drawn = lo + frac.clamp(0.0, 1.0) * (hi - lo);
    let ours = Duration::from_secs_f64(if drawn.is_finite() {
        drawn.max(0.0)
    } else {
        0.0
    });
    match hint {
        Some(hint) if hint > ours => hint,
        _ => ours,
    }
}

/// A `std::time::Duration` as chrono's, for the instants we compute. A duration
/// too large for chrono (centuries) becomes zero rather than panicking.
fn chrono_d(d: Duration) -> chrono::Duration {
    chrono::Duration::from_std(d).unwrap_or_default()
}

/// How much time passed between two instants, clamped to zero.
fn elapsed(from: DateTime<Utc>, now: DateTime<Utc>) -> Duration {
    (now - from).to_std().unwrap_or_default()
}

// ============================================================================
// Tests: the transition table, a manual clock, and one red per rule
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::limits::ledger::ManualClock;

    fn at(text: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(text)
            .unwrap()
            .with_timezone(&Utc)
    }

    /// A breaker whose jitter is OFF, so every cooldown is an exact instant.
    fn exact() -> Fixture {
        fixture_with(Policy {
            jitter_ratio: 0.0,
            ..Policy::default()
        })
    }

    fn fixture_with(policy: Policy) -> Fixture {
        let clock = Arc::new(ManualClock::at("2026-10-16T12:00:00Z"));
        let breaker = Breaker::new(clock.clone(), Arc::new(FixedJitter::new(0.5)), policy);
        Fixture { breaker, clock }
    }

    fn key() -> Key {
        Key::new("nim", Some("vault:nim-prod".to_string()))
    }

    struct Fixture {
        breaker: Breaker,
        clock: Arc<ManualClock>,
    }

    impl Fixture {
        fn open_by_overload(&self, times: u32) {
            for _ in 0..times {
                self.breaker.record(&key(), Signal::Overload { hint: None });
            }
        }
    }

    // ---- Closed: the table's first column ------------------------------------

    #[test]
    fn a_closed_breaker_allows_and_never_refuses() {
        let f = exact();
        let first = f.breaker.attempt(&key());
        assert!(first.is_allowed());
        assert!(!first.is_probe(), "a normal call is not a probe");
        assert_eq!(first.status(), RateLimitStatus::Ok);
        assert_eq!(f.breaker.state(&key()).phase, Phase::Closed);
        assert!(
            f.breaker.trace().is_empty(),
            "nothing moved, nothing to report"
        );
    }

    #[test]
    fn one_overload_below_the_threshold_keeps_it_closed() {
        let f = exact();
        assert!(
            f.breaker
                .record(&key(), Signal::Overload { hint: None })
                .is_none(),
            "one overload is not yet a reason to stop"
        );
        assert!(f.breaker.attempt(&key()).is_allowed());
        let st = f.breaker.state(&key());
        assert_eq!(st.phase, Phase::Closed);
        assert_eq!(st.overloads, 1);
        assert_eq!(st.status, RateLimitStatus::Ok);
    }

    #[test]
    fn the_threshold_moves_closed_to_open_on_an_overload() {
        let f = exact();
        f.open_by_overload(2);
        assert!(!f.breaker.state(&key()).is_open(), "still closed at two");
        let t = f
            .breaker
            .record(&key(), Signal::Overload { hint: None })
            .expect("the third consecutive overload opens");
        assert_eq!((t.from, t.to), (Phase::Closed, Phase::Open));
        assert_eq!(t.reason, Reason::Overload);
        assert_eq!(t.status, RateLimitStatus::Overloaded);
        // retries = 3 -> 1s * 2^2 = 4s, no jitter.
        assert_eq!(t.until, Some(at("2026-10-16T12:00:04Z")));
        let d = f.breaker.attempt(&key());
        assert!(!d.is_allowed());
        assert_eq!(d.status(), RateLimitStatus::Overloaded);
        assert_eq!(d.until(), Some(at("2026-10-16T12:00:04Z")));
    }

    #[test]
    fn a_quota_moves_closed_to_open_until_the_reset() {
        let f = exact();
        let reset = at("2026-10-16T12:05:00Z");
        let t = f
            .breaker
            .record(
                &key(),
                Signal::Quota {
                    reset_at: Some(reset),
                },
            )
            .unwrap();
        assert_eq!((t.from, t.to), (Phase::Closed, Phase::Open));
        assert_eq!(t.reason, Reason::Quota);
        assert_eq!(t.status, RateLimitStatus::Limited);
        assert_eq!(t.until, Some(reset));
        let st = f.breaker.state(&key());
        assert!(st.exhausted, "a quota marks the key spent");
        assert_eq!(st.status, RateLimitStatus::Limited);
        assert_eq!(st.next_retry_at, Some(reset));
    }

    #[test]
    fn a_success_while_closed_is_a_no_op() {
        let f = exact();
        assert!(f.breaker.record(&key(), Signal::Success).is_none());
        assert!(f.breaker.trace().is_empty());
        assert_eq!(f.breaker.state(&key()).status, RateLimitStatus::Ok);
    }

    #[test]
    fn a_quota_without_a_known_reset_uses_a_bounded_default() {
        let f = exact();
        let t = f
            .breaker
            .record(&key(), Signal::Quota { reset_at: None })
            .unwrap();
        let waited = (t.until.unwrap() - t.at).to_std().unwrap();
        assert_eq!(waited, Policy::default().quota_unknown_cooldown);
        assert!(f.breaker.attempt(&key()).until().is_some());
    }

    // ---- Open: the table's second column -------------------------------------

    #[test]
    fn the_quota_cooldown_ends_at_reset_at_and_not_before() {
        // MUTATION: computing the cooldown from `now` instead of `reset_at` turns
        // the first assertion red; dropping the comparison turns the second red.
        let f = exact();
        let reset = at("2026-10-16T12:05:00Z");
        f.breaker.record(
            &key(),
            Signal::Quota {
                reset_at: Some(reset),
            },
        );

        f.clock.advance(Duration::from_secs(299)); // reset - 1s
        let early = f.breaker.attempt(&key());
        assert!(
            !early.is_allowed(),
            "the key may not reopen one second before its own reset"
        );
        assert_eq!(early.until(), Some(reset), "and it says exactly until when");
        assert_eq!(
            f.breaker.trace().len(),
            1,
            "no probe may have been granted before the reset"
        );

        f.clock.advance(Duration::from_secs(1)); // exactly the reset
        let on_time = f.breaker.attempt(&key());
        assert!(on_time.is_allowed());
        assert!(on_time.is_probe());
        assert_eq!(f.breaker.state(&key()).phase, Phase::HalfOpen);
        assert!(!f.breaker.state(&key()).exhausted, "the reset has passed");
    }

    #[test]
    fn the_overload_cooldown_is_short_then_grants_one_probe() {
        let f = exact();
        f.open_by_overload(3); // opened at 12:00:00 for 4s
        f.clock.advance(Duration::from_secs(3));
        assert!(!f.breaker.attempt(&key()).is_allowed());
        f.clock.advance(Duration::from_secs(1));
        assert!(f.breaker.attempt(&key()).is_probe());
    }

    #[test]
    fn a_success_reported_while_open_on_a_quota_does_not_reopen_it() {
        let f = exact();
        let reset = at("2026-10-16T12:05:00Z");
        f.breaker.record(
            &key(),
            Signal::Quota {
                reset_at: Some(reset),
            },
        );
        assert!(
            f.breaker.record(&key(), Signal::Success).is_none(),
            "a straggler call does not refill a quota the provider declared spent"
        );
        let st = f.breaker.state(&key());
        assert_eq!(st.phase, Phase::Open);
        assert!(st.exhausted);
        assert!(!f.breaker.attempt(&key()).is_allowed());
    }

    #[test]
    fn a_success_reported_while_open_on_an_overload_closes_it() {
        let f = exact();
        f.open_by_overload(3);
        let t = f.breaker.record(&key(), Signal::Success).unwrap();
        assert_eq!((t.from, t.to), (Phase::Open, Phase::Closed));
        assert_eq!(t.reason, Reason::Recovered);
        assert_eq!(f.breaker.state(&key()).status, RateLimitStatus::Ok);
        assert!(f.breaker.attempt(&key()).is_allowed());
    }

    // ---- HalfOpen: the table's third column ----------------------------------

    #[test]
    fn at_most_one_probe_is_granted_in_half_open() {
        // MUTATION: dropping the "probe in flight" guard lets a burst through and
        // turns the count of granted probes red (the retry storm).
        let f = exact();
        f.open_by_overload(3);
        f.clock.advance(Duration::from_secs(4));
        assert!(
            f.breaker.attempt(&key()).is_probe(),
            "the first call is the probe"
        );
        for i in 0..10 {
            let d = f.breaker.attempt(&key());
            assert!(
                !d.is_allowed(),
                "call {i} of the burst went through the half-open door"
            );
            assert_eq!(d.status(), RateLimitStatus::Overloaded);
        }
        let granted = f
            .breaker
            .trace()
            .iter()
            .filter(|t| t.reason == Reason::ProbeGranted)
            .count();
        assert_eq!(granted, 1, "exactly one probe, never a burst");
    }

    #[test]
    fn a_successful_probe_closes_the_breaker() {
        let f = exact();
        f.open_by_overload(3);
        f.clock.advance(Duration::from_secs(4));
        assert!(f.breaker.attempt(&key()).is_probe());
        let t = f.breaker.record(&key(), Signal::Success).unwrap();
        assert_eq!((t.from, t.to), (Phase::HalfOpen, Phase::Closed));
        assert_eq!(t.reason, Reason::Recovered);
        let st = f.breaker.state(&key());
        assert_eq!(st.phase, Phase::Closed);
        assert_eq!(st.retries, 0, "a success forgets the episode");
        assert!(!f.breaker.attempt(&key()).is_probe());
    }

    #[test]
    fn a_failed_probe_reopens_for_longer() {
        let f = exact();
        f.open_by_overload(3); // 12:00:04
        f.clock.advance(Duration::from_secs(4));
        assert!(f.breaker.attempt(&key()).is_probe());
        let t = f
            .breaker
            .record(&key(), Signal::Overload { hint: None })
            .unwrap();
        assert_eq!((t.from, t.to), (Phase::HalfOpen, Phase::Open));
        assert_eq!(t.until, Some(at("2026-10-16T12:00:12Z")), "4s then 8s");
    }

    #[test]
    fn a_quota_reported_by_the_probe_reopens_until_the_new_reset() {
        let f = exact();
        f.open_by_overload(3);
        f.clock.advance(Duration::from_secs(4));
        assert!(f.breaker.attempt(&key()).is_probe());
        let reset = at("2026-10-16T13:00:00Z");
        let t = f
            .breaker
            .record(
                &key(),
                Signal::Quota {
                    reset_at: Some(reset),
                },
            )
            .unwrap();
        assert_eq!(t.reason, Reason::Quota);
        assert_eq!(t.until, Some(reset));
        assert!(f.breaker.state(&key()).exhausted);
    }

    #[test]
    fn an_abandoned_probe_reopens_after_the_probe_timeout() {
        let f = exact();
        f.open_by_overload(3);
        f.clock.advance(Duration::from_secs(4));
        assert!(f.breaker.attempt(&key()).is_probe());

        f.clock.advance(Duration::from_secs(29));
        let waiting = f.breaker.attempt(&key());
        assert!(!waiting.is_allowed());
        assert_eq!(waiting.until(), None, "we wait for the probe's outcome");
        assert!(
            f.breaker
                .trace()
                .iter()
                .all(|t| t.reason != Reason::ProbeAbandoned),
            "29s is not yet a timeout"
        );

        f.clock.advance(Duration::from_secs(1)); // == probe_timeout (30s)
        assert!(!f.breaker.attempt(&key()).is_allowed());
        assert_eq!(
            f.breaker
                .trace()
                .iter()
                .filter(|t| t.reason == Reason::ProbeAbandoned)
                .count(),
            1
        );

        // and the breaker is not wedged: after the fresh cooldown, a new probe.
        f.clock.advance(Duration::from_secs(60));
        assert!(f.breaker.attempt(&key()).is_probe());
    }

    // ---- The four named rules, each with its mutation ------------------------

    #[test]
    fn an_overload_never_marks_the_key_exhausted() {
        // MUTATION: setting `exhausted` in EITHER overload arm — the one that
        // follows a refusal, or the one that follows an abandoned probe — turns
        // this red.
        let f = exact();
        f.open_by_overload(3);
        let st = f.breaker.state(&key());
        assert_eq!(st.phase, Phase::Open);
        assert_eq!(st.status, RateLimitStatus::Overloaded);
        assert!(!st.exhausted, "an overload is never an exhausted key");

        // ...not through a failed probe either: retries 3 -> 4 -> an 8s cooldown.
        f.clock.advance(Duration::from_secs(4));
        assert!(f.breaker.attempt(&key()).is_probe());
        f.breaker.record(&key(), Signal::Overload { hint: None });
        let st = f.breaker.state(&key());
        assert!(!st.exhausted);
        assert_eq!(st.status, RateLimitStatus::Overloaded);

        // ...nor through an ABANDONED probe, the other overload path that opens.
        f.clock.advance(Duration::from_secs(8)); // the 8s cooldown ends
        assert!(f.breaker.attempt(&key()).is_probe());
        f.clock.advance(Duration::from_secs(30)); // == probe_timeout
        assert!(!f.breaker.attempt(&key()).is_allowed());
        let st = f.breaker.state(&key());
        assert!(!st.exhausted, "an abandoned probe is still an overload");
        assert_eq!(st.status, RateLimitStatus::Overloaded);
    }

    #[test]
    fn the_backoff_is_capped_at_the_ceiling() {
        // MUTATION: dropping the `min(max_backoff)` turns this red.
        let f = exact();
        f.open_by_overload(8);
        let delays: Vec<Duration> = f
            .breaker
            .trace()
            .iter()
            .map(|t| (t.until.unwrap() - t.at).to_std().unwrap())
            .collect();
        // retries 3..8 -> 1s * 2^(n-1) = 4, 8, 16, 32, 64, 128 -> capped at 60.
        assert_eq!(
            delays,
            vec![
                Duration::from_secs(4),
                Duration::from_secs(8),
                Duration::from_secs(16),
                Duration::from_secs(32),
                Duration::from_secs(60),
                Duration::from_secs(60),
            ],
            "the cooldown grows, then stops at the ceiling"
        );
        for d in delays {
            assert!(d <= f.breaker.policy().max_backoff);
        }
    }

    #[test]
    fn a_providers_retry_after_is_a_floor_and_is_never_shortened() {
        let f = exact();
        let hint = Duration::from_secs(120); // beyond our own ceiling
        for _ in 0..3 {
            f.breaker
                .record(&key(), Signal::Overload { hint: Some(hint) });
        }
        let st = f.breaker.state(&key());
        let waited = (st.until.unwrap() - st.since.unwrap()).to_std().unwrap();
        assert!(
            waited >= hint,
            "the provider's word is a floor, got {waited:?}"
        );
    }

    // ---- Jitter, observation, isolation --------------------------------------

    #[test]
    fn jitter_spreads_the_overload_cooldown() {
        let clock = Arc::new(ManualClock::at("2026-10-16T12:00:00Z"));
        let breaker = Breaker::with_defaults(clock, Arc::new(SeededJitter::new(42)));
        let a = Key::for_instance("nim-a");
        let b = Key::for_instance("nim-b");
        for k in [&a, &b] {
            for _ in 0..3 {
                breaker.record(k, Signal::Overload { hint: None });
            }
        }
        let (sa, sb) = (breaker.state(&a), breaker.state(&b));
        assert_ne!(
            sa.until, sb.until,
            "two keys limited the same second must not return the same second"
        );
        for st in [sa, sb] {
            let waited = (st.until.unwrap() - st.since.unwrap()).to_std().unwrap();
            assert!(waited <= Policy::default().max_backoff);
        }
    }

    #[test]
    fn every_state_change_is_observable_and_names_only_a_reference() {
        let f = exact();
        f.breaker.record(
            &key(),
            Signal::Quota {
                reset_at: Some(at("2026-10-16T12:05:00Z")),
            },
        );
        f.clock.advance(Duration::from_secs(300));
        assert!(f.breaker.attempt(&key()).is_probe());
        f.breaker.record(&key(), Signal::Success);

        let reasons: Vec<Reason> = f.breaker.trace().iter().map(|t| t.reason).collect();
        assert_eq!(
            reasons,
            vec![Reason::Quota, Reason::ProbeGranted, Reason::Recovered]
        );
        let observed = f.breaker.trace();
        assert_eq!(observed[0].at_rfc3339(), "2026-10-16T12:00:00Z");
        assert_eq!(observed[1].at_rfc3339(), "2026-10-16T12:05:00Z");
        assert_eq!(
            observed[0].until_rfc3339().as_deref(),
            Some("2026-10-16T12:05:00Z")
        );
        for t in observed {
            assert_eq!(t.instance, "nim");
            assert_eq!(
                t.credential.as_deref(),
                Some("vault:nim-prod"),
                "the credential REFERENCE, never a value"
            );
        }
        let taken = f.breaker.take_trace();
        assert_eq!(taken.len(), 3);
        assert!(f.breaker.trace().is_empty(), "take_trace drains");
    }

    #[test]
    fn the_trace_is_bounded() {
        let f = exact();
        for _ in 0..(TRACE_CAP + 40) {
            f.breaker.record(&key(), Signal::Overload { hint: None });
        }
        assert_eq!(f.breaker.trace().len(), TRACE_CAP);
    }

    #[test]
    fn two_credentials_of_one_instance_are_independent() {
        let f = exact();
        let a = Key::new("nim", Some("vault:key-a".to_string()));
        let b = Key::new("nim", Some("vault:key-b".to_string()));
        f.breaker.record(
            &a,
            Signal::Quota {
                reset_at: Some(at("2026-10-16T12:30:00Z")),
            },
        );
        assert!(!f.breaker.attempt(&a).is_allowed());
        assert!(
            f.breaker.attempt(&b).is_allowed(),
            "the other key of the same instance is untouched"
        );
        assert_eq!(
            f.breaker.state(&b).credential.as_deref(),
            Some("vault:key-b")
        );
    }

    #[test]
    fn an_rl3_reading_maps_to_the_right_signal() {
        assert_eq!(
            Signal::of(&Verdict::QuotaExhausted { reset_at: None }),
            Signal::Quota { reset_at: None }
        );
        let hint = Some(Duration::from_secs(2));
        assert_eq!(
            Signal::of(&Verdict::Overloaded { retry_after: hint }),
            Signal::Overload { hint }
        );
        assert_eq!(
            Signal::of(&Verdict::Unknown),
            Signal::Overload { hint: None },
            "an unreadable refusal is an overload, never an exhaustion"
        );
    }
}
