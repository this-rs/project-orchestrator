//! RL4 — the breaker: overload backoff and the single probe, one state machine per
//! (instance, key).
//!
//! RL3 reads a refusal and answers "spent quota" or "overload". RL2 counts. This
//! module holds what to do about an **overload**, and for how long. It is a PURE
//! state machine — no I/O, no event bus, no secret, no clock of its own (the
//! instant comes from an injected [`Clock`], exactly as the ledger's) — so its
//! whole behaviour is a truth table a test can drive.
//!
//! # A spent quota is NOT this module's business
//!
//! [`QuotaRegistry`](crate::chat::provider::quota::QuotaRegistry) is the single
//! source of truth for "this account is exhausted until X": it is fed by the
//! provider's own report and by quota refusals, and the router excludes on it. An
//! exhausted account is healthy — it answers, and says no — so it never goes
//! through a breaker. This module therefore has NO quota signal: [`Signal::of`]
//! maps a quota verdict to `None`, and nothing here can open on one, mark a key
//! spent, or date a reset. (Two clocks for the same fact would disagree.)
//!
//! # The three phases
//!
//! ```text
//!            overload threshold reached
//!   Closed  ───────────────────────────▶  Open
//!      ▲                                    │
//!      │                   cooldown elapsed │
//!      │  the one probe succeeded           ▼
//!      └──────────────────────────────  HalfOpen
//! ```
//!
//! * **Closed** — calls go out. Consecutive overloads are counted.
//! * **Open** — every call is refused until `until`. The instant a new attempt may
//!   be made is [`Decision::Refuse::until`], so a caller never guesses.
//! * **HalfOpen** — EXACTLY ONE probe is in flight. Every further `attempt` is
//!   refused until that probe is settled by `record`, or abandoned after
//!   `probe_timeout` (a cancelled turn whose outcome never comes back must not
//!   wedge the breaker forever).
//!
//! # Stragglers and the probe
//!
//! [`Breaker::record`] takes `probe`: whether the call being settled WAS the
//! half-open probe. A call admitted before the pause opened and settled after is a
//! straggler and changes nothing — a late success must not close a pause no probe
//! tested, and a late overload must not push the end of a pause that has one. Only
//! the probe settles a half-open breaker.
//!
//! # Untrusted numbers
//!
//! A provider's `Retry-After` is capped once ([`cap_hint`](super::classify::cap_hint),
//! a day), every instant is computed saturating, and an overload never opens for
//! less than [`Policy::min_cooldown`]. No `Duration` input panics.
//!
//! An **overload** ([`Signal::Overload`]) opens for a SHORT, exponential backoff
//! with jitter, capped at `max_backoff`; a provider's `Retry-After` is a floor that
//! is never shortened. It never marks anything as spent: our own ceiling is not
//! what refused us, so nothing about the credential changed.
//!
//! # Retries consume the count — delegated, not reimplemented
//!
//! This module keeps ONE counter that is policy, not consumption: how many retries
//! this episode has already spent (`max_retries`, which also saturates the
//! exponent of the backoff). It counts no request, no token, no dollar — that is
//! [`super::ledger`]'s job and stays there.
//!
//! # What a caller sees
//!
//! Every state change is a [`Transition`], appended to a bounded trace readable
//! with [`Breaker::trace`]. [`State`] carries the phase, RL1's
//! [`RateLimitStatus`], since when, and when the next attempt may be made. Both
//! carry the instance id and the credential REFERENCE (`vault:<nom>`,
//! `env:<VAR>`) — never a credential value, because no field of this module can
//! hold one.
//!
//! # Boundary with the other breakers
//!
//! `src/chat/provider/circuit_breaker.rs` is a generic HEALTH breaker over a whole
//! instance. This one is overload-specific and keyed by (instance, credential
//! REFERENCE). They are not merged.

use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};

use super::classify::{after, cap_hint, rfc3339, Verdict};
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
/// limit).
///
/// The reference is VALIDATED at construction ([`Key::new`] refuses anything that
/// is not `vault:`/`env:` followed by a name), the fields are private, and `Debug`
/// and `Display` print only the instance and that reference: a raw secret cannot be
/// put in a key, so a key is safe to key a map by, log, or answer over HTTP.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Key {
    instance: String,
    credential: Option<String>,
}

/// A credential that is not a REFERENCE. It carries nothing of the rejected value:
/// an error that echoed a secret would leak it as surely as the key would have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidReference;

impl std::fmt::Display for InvalidReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a credential must be a reference (vault:<name> or env:<VAR>), never a value")
    }
}

impl std::error::Error for InvalidReference {}

/// Whether `raw` is `vault:<name>` or `env:<VAR>` with a well-formed name.
fn is_reference(raw: &str) -> bool {
    let Some(name) = raw
        .strip_prefix("vault:")
        .or_else(|| raw.strip_prefix("env:"))
    else {
        return false;
    };
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/'))
}

impl Key {
    /// A key for one credential of one instance. The credential must be a
    /// REFERENCE; anything else is refused.
    pub fn new(
        instance: impl Into<String>,
        credential: Option<String>,
    ) -> Result<Self, InvalidReference> {
        if credential.as_deref().is_some_and(|c| !is_reference(c)) {
            return Err(InvalidReference);
        }
        Ok(Self {
            instance: instance.into(),
            credential,
        })
    }

    /// A key for an instance used without a named credential.
    pub fn for_instance(instance: impl Into<String>) -> Self {
        Self {
            instance: instance.into(),
            credential: None,
        }
    }

    /// The provider instance (`nim`, `deepseek`, an alias…).
    pub fn instance(&self) -> &str {
        &self.instance
    }

    /// The credential REFERENCE, when the breaker is per key.
    pub fn credential(&self) -> Option<&str> {
        self.credential.as_deref()
    }
}

impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.credential {
            Some(reference) => write!(f, "{}/{}", self.instance, reference),
            None => f.write_str(&self.instance),
        }
    }
}

impl Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Key({self})")
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
    /// Overloaded, throttling, or unreadable: retry with a bounded backoff. `hint` is the provider's `Retry-After`, a
    /// floor that is never shortened.
    Overload {
        /// The delay the provider asked for, when it asked for one.
        hint: Option<Duration>,
    },
}

impl Signal {
    /// The signal an RL3 reading stands for, or `None` for a spent quota, which
    /// belongs to the quota registry and never to a breaker. A refusal this module
    /// cannot read is an overload, never an exhaustion.
    pub fn of(verdict: &Verdict) -> Option<Self> {
        match verdict {
            Verdict::QuotaExhausted { .. } => None,
            Verdict::Overloaded { retry_after } => Some(Self::Overload { hint: *retry_after }),
            Verdict::Unknown => Some(Self::Overload { hint: None }),
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
        /// RL1's status for the answer: always `overloaded`.
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
    /// The provider is overloaded: a short cooldown.
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
            instance: key.instance().to_string(),
            credential: key.credential().map(str::to_string),
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
    /// RL1's status: `ok` or `overloaded`.
    pub status: RateLimitStatus,
    /// Since when the current phase holds.
    pub since: Option<DateTime<Utc>>,
    /// The current cooldown's end, when one is running.
    pub until: Option<DateTime<Utc>>,
    /// When the next attempt may be made (the cooldown's end).
    pub next_retry_at: Option<DateTime<Utc>>,
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
    /// How long the one half-open probe may stay unanswered before it is counted
    /// as a failure.
    pub probe_timeout: Duration,
    /// The shortest cooldown an overload can open for. Jitter at ratio 1, a zero
    /// `base_backoff` or a NaN ratio would otherwise open a pause that has already
    /// ended, which is no pause (and a retry storm).
    pub min_cooldown: Duration,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            overload_threshold: 3,
            base_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(60),
            jitter_ratio: 0.5,
            max_retries: 8,
            probe_timeout: Duration::from_secs(30),
            min_cooldown: Duration::from_millis(500),
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

/// The phase of a subject, with what only that phase has: an open pause has its
/// end, a half-open one has the instant its probe left. A phase cannot exist
/// without its data, so there is no half-open state without a probe to wait for.
#[derive(Debug, Clone, Copy)]
enum Mode {
    Closed { overloads: u32 },
    Open { until: DateTime<Utc> },
    HalfOpen { probe_since: DateTime<Utc> },
}

/// One subject's live state.
#[derive(Debug, Clone)]
struct Entry {
    mode: Mode,
    /// Retries spent since the last success.
    retries: u32,
    /// Since when the current phase holds.
    since: Option<DateTime<Utc>>,
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            mode: Mode::Closed { overloads: 0 },
            retries: 0,
            since: None,
        }
    }
}

impl Entry {
    fn phase(&self) -> Phase {
        match self.mode {
            Mode::Closed { .. } => Phase::Closed,
            Mode::Open { .. } => Phase::Open,
            Mode::HalfOpen { .. } => Phase::HalfOpen,
        }
    }

    fn status(&self) -> RateLimitStatus {
        match self.mode {
            Mode::Closed { .. } => RateLimitStatus::Ok,
            Mode::Open { .. } | Mode::HalfOpen { .. } => RateLimitStatus::Overloaded,
        }
    }

    /// Reaches `Closed` and forgets the episode.
    fn close(&mut self, now: DateTime<Utc>) {
        self.mode = Mode::Closed { overloads: 0 };
        self.retries = 0;
        self.since = Some(now);
    }

    /// Reaches `Open` for `until`.
    fn open(&mut self, until: DateTime<Utc>, now: DateTime<Utc>) {
        self.mode = Mode::Open { until };
        self.since = Some(now);
    }

    fn bump_retries(&mut self, policy: &Policy) {
        self.retries = self
            .retries
            .saturating_add(1)
            .min(policy.max_retries.max(1));
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
    /// The instant is read from the injected clock; nothing here sleeps. A caller
    /// whose [`Decision::is_probe`] is true MUST settle with `record(.., probe: true)`.
    pub fn attempt(&self, key: &Key) -> Decision {
        let now = self.clock.now();
        let policy = &self.policy;
        let mut moved = None;
        let decision = {
            let mut entries = self.lock_entries();
            let e = entries.entry(key.clone()).or_default();
            match e.mode {
                Mode::Closed { .. } => Decision::Allow { probe: false },
                Mode::Open { until } if now >= until => {
                    // The cooldown has ended: one probe — never a burst — may now
                    // go out.
                    e.mode = Mode::HalfOpen { probe_since: now };
                    e.since = Some(now);
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
                }
                Mode::Open { until } => Decision::Refuse {
                    until: Some(until),
                    status: RateLimitStatus::Overloaded,
                },
                Mode::HalfOpen { probe_since }
                    if elapsed(probe_since, now) >= policy.probe_timeout =>
                {
                    // The probe was never settled (a cancelled turn, a dropped
                    // future): count it as a failure instead of waiting forever.
                    e.bump_retries(policy);
                    let delay = overload_delay(policy, e.retries, None, self.jitter.frac());
                    let until = after(now, delay);
                    e.open(until, now);
                    moved = Some(Transition::new(
                        key,
                        Phase::HalfOpen,
                        Phase::Open,
                        now,
                        RateLimitStatus::Overloaded,
                        Reason::ProbeAbandoned,
                        Some(until),
                    ));
                    Decision::Refuse {
                        until: Some(until),
                        status: RateLimitStatus::Overloaded,
                    }
                }
                // One probe at a time: nothing else goes out while it is out.
                Mode::HalfOpen { .. } => Decision::Refuse {
                    until: None,
                    status: RateLimitStatus::Overloaded,
                },
            }
        };
        if let Some(t) = moved {
            self.push_trace(t);
        }
        decision
    }

    /// Tells the breaker how a call ended. `probe` says whether that call WAS the
    /// half-open probe ([`Decision::is_probe`] when it was admitted).
    ///
    /// A record that no longer describes the present is ignored: a call admitted
    /// before the breaker opened that finishes while it is `Open`, or while a probe
    /// is out, is a straggler — a late success must not close a pause no probe has
    /// tested (and orphan the probe in flight), and a late overload must not push
    /// the end of a pause that already has one. Only the probe settles a half-open
    /// breaker. Returns the transition the record caused, when it caused one.
    pub fn record(&self, key: &Key, signal: Signal, probe: bool) -> Option<Transition> {
        let now = self.clock.now();
        let policy = &self.policy;
        let transition = {
            let mut entries = self.lock_entries();
            let e = entries.entry(key.clone()).or_default();
            match (e.mode, signal) {
                (Mode::Closed { .. }, Signal::Success) => {
                    e.close(now);
                    None
                }
                (Mode::Closed { overloads }, Signal::Overload { hint }) => {
                    let overloads = overloads.saturating_add(1);
                    e.mode = Mode::Closed { overloads };
                    e.bump_retries(policy);
                    if overloads >= policy.overload_threshold.max(1) {
                        let delay = overload_delay(policy, e.retries, hint, self.jitter.frac());
                        let until = after(now, delay);
                        e.open(until, now);
                        Some(Transition::new(
                            key,
                            Phase::Closed,
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
                // A straggler: not the probe, not about this pause.
                (Mode::Open { .. }, _) => None,
                (Mode::HalfOpen { .. }, _) if !probe => None,
                (Mode::HalfOpen { .. }, Signal::Success) => {
                    e.close(now);
                    Some(Transition::new(
                        key,
                        Phase::HalfOpen,
                        Phase::Closed,
                        now,
                        RateLimitStatus::Ok,
                        Reason::Recovered,
                        None,
                    ))
                }
                (Mode::HalfOpen { .. }, Signal::Overload { hint }) => {
                    e.bump_retries(policy);
                    let delay = overload_delay(policy, e.retries, hint, self.jitter.frac());
                    let until = after(now, delay);
                    e.open(until, now);
                    Some(Transition::new(
                        key,
                        Phase::HalfOpen,
                        Phase::Open,
                        now,
                        RateLimitStatus::Overloaded,
                        Reason::Overload,
                        Some(until),
                    ))
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
        let entry = entries.get(key).cloned().unwrap_or_default();
        let (until, overloads) = match entry.mode {
            Mode::Closed { overloads } => (None, overloads),
            Mode::Open { until } => (Some(until), 0),
            Mode::HalfOpen { .. } => (None, 0),
        };
        State {
            instance: key.instance().to_string(),
            credential: key.credential().map(str::to_string),
            phase: entry.phase(),
            status: entry.status(),
            since: entry.since,
            until,
            next_retry_at: until,
            overloads,
            retries: entry.retries,
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
/// it; the ceiling [`Policy::max_backoff`] then bounds our own backoff whatever the
/// retry count. The jitter keeps `jitter_ratio` of the cooldown free, so two keys
/// limited the same second do not come back the same second; the result is never
/// below [`Policy::min_cooldown`]. A provider's `Retry-After` is a FLOOR: it is
/// never shortened by our own ceiling — but it is first bounded by
/// [`cap_hint`](super::classify::cap_hint), because a header is untrusted input.
/// No input makes this panic: every conversion saturates.
fn overload_delay(policy: &Policy, retries: u32, hint: Option<Duration>, frac: f64) -> Duration {
    let steps = retries.max(1).min(policy.max_retries.max(1));
    let factor = 2u32.saturating_pow(steps - 1);
    let raw = policy.base_backoff.saturating_mul(factor);
    let capped = raw.min(policy.max_backoff);
    let ratio = policy.jitter_ratio.clamp(0.0, 1.0);
    let hi = capped.as_secs_f64();
    let lo = hi * (1.0 - ratio);
    let drawn = lo + frac.clamp(0.0, 1.0) * (hi - lo);
    let drawn = if drawn.is_finite() {
        drawn.max(0.0)
    } else {
        0.0
    };
    let ours = Duration::try_from_secs_f64(drawn)
        .unwrap_or(Duration::MAX)
        .max(policy.min_cooldown);
    match hint.map(cap_hint) {
        Some(hint) if hint > ours => hint,
        _ => ours,
    }
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
        Key::new("nim", Some("vault:nim-prod".to_string())).unwrap()
    }

    struct Fixture {
        breaker: Breaker,
        clock: Arc<ManualClock>,
    }

    impl Fixture {
        fn open_by_overload(&self, times: u32) {
            for _ in 0..times {
                self.breaker
                    .record(&key(), Signal::Overload { hint: None }, false);
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
                .record(&key(), Signal::Overload { hint: None }, false)
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
            .record(&key(), Signal::Overload { hint: None }, false)
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
    fn a_success_while_closed_is_a_no_op() {
        let f = exact();
        assert!(f.breaker.record(&key(), Signal::Success, false).is_none());
        assert!(f.breaker.trace().is_empty());
        assert_eq!(f.breaker.state(&key()).status, RateLimitStatus::Ok);
    }

    // ---- Open: the table's second column -------------------------------------

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
    fn a_straggler_reported_while_open_changes_nothing() {
        // MUTATION: closing on a `Success` while `Open` (the old arm) turns the
        // first block red; re-opening on an `Overload` while `Open` turns the
        // second red (it would push `until` and bump `retries`).
        let f = exact();
        f.open_by_overload(3); // open until 12:00:04
        let before = f.breaker.state(&key());
        assert!(
            f.breaker.record(&key(), Signal::Success, false).is_none(),
            "a call admitted before the pause says nothing about the pause"
        );
        assert_eq!(f.breaker.state(&key()), before, "still open, same end");
        assert!(!f.breaker.attempt(&key()).is_allowed());

        f.clock.advance(Duration::from_secs(1));
        assert!(f
            .breaker
            .record(&key(), Signal::Overload { hint: None }, false)
            .is_none());
        let after = f.breaker.state(&key());
        assert_eq!(
            after.until, before.until,
            "the end of the pause did not move"
        );
        assert_eq!(after.retries, before.retries, "no retry was spent");
        assert_eq!(f.breaker.trace().len(), 1, "no transition either");
    }

    #[test]
    fn a_straggler_reported_while_a_probe_is_out_does_not_settle_it() {
        // MUTATION: ignoring `probe` in the half-open arms lets the straggler
        // close the breaker and orphan the probe in flight.
        let f = exact();
        f.open_by_overload(3);
        f.clock.advance(Duration::from_secs(4));
        assert!(f.breaker.attempt(&key()).is_probe());
        assert!(f.breaker.record(&key(), Signal::Success, false).is_none());
        assert!(f
            .breaker
            .record(&key(), Signal::Overload { hint: None }, false)
            .is_none());
        assert_eq!(f.breaker.state(&key()).phase, Phase::HalfOpen);
        // The probe itself still settles it.
        let t = f.breaker.record(&key(), Signal::Success, true).unwrap();
        assert_eq!(t.to, Phase::Closed);
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
        let t = f.breaker.record(&key(), Signal::Success, true).unwrap();
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
            .record(&key(), Signal::Overload { hint: None }, true)
            .unwrap();
        assert_eq!((t.from, t.to), (Phase::HalfOpen, Phase::Open));
        assert_eq!(t.until, Some(at("2026-10-16T12:00:12Z")), "4s then 8s");
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
    fn an_overload_is_never_reported_as_a_limit() {
        // MUTATION: reporting `Limited` from `Entry::status` for an open phase
        // turns this red.
        let f = exact();
        f.open_by_overload(3);
        let st = f.breaker.state(&key());
        assert_eq!(st.phase, Phase::Open);
        assert_eq!(st.status, RateLimitStatus::Overloaded);

        // ...not through a failed probe either: retries 3 -> 4 -> an 8s cooldown.
        f.clock.advance(Duration::from_secs(4));
        assert!(f.breaker.attempt(&key()).is_probe());
        f.breaker
            .record(&key(), Signal::Overload { hint: None }, true);
        let st = f.breaker.state(&key());
        assert_eq!(st.status, RateLimitStatus::Overloaded);

        // ...nor through an ABANDONED probe, the other overload path that opens.
        f.clock.advance(Duration::from_secs(8)); // the 8s cooldown ends
        assert!(f.breaker.attempt(&key()).is_probe());
        f.clock.advance(Duration::from_secs(30)); // == probe_timeout
        assert!(!f.breaker.attempt(&key()).is_allowed());
        let st = f.breaker.state(&key());
        assert_eq!(st.status, RateLimitStatus::Overloaded);
    }

    #[test]
    fn the_backoff_is_capped_at_the_ceiling() {
        // MUTATION: dropping the `min(max_backoff)` turns this red.
        let f = exact();
        f.open_by_overload(3);
        // Every failed probe reopens for twice as long, until the ceiling.
        for _ in 0..5 {
            let until = f.breaker.state(&key()).until.unwrap();
            f.clock.advance((until - f.clock.now()).to_std().unwrap());
            assert!(f.breaker.attempt(&key()).is_probe());
            f.breaker
                .record(&key(), Signal::Overload { hint: None }, true);
        }
        let delays: Vec<Duration> = f
            .breaker
            .trace()
            .iter()
            .filter(|t| t.reason == Reason::Overload)
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
    }

    #[test]
    fn a_providers_retry_after_is_a_floor_and_is_never_shortened() {
        let f = exact();
        let hint = Duration::from_secs(120); // beyond our own ceiling
        for _ in 0..3 {
            f.breaker
                .record(&key(), Signal::Overload { hint: Some(hint) }, false);
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
                breaker.record(k, Signal::Overload { hint: None }, false);
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
        f.open_by_overload(3); // open for 4s
        f.clock.advance(Duration::from_secs(4));
        assert!(f.breaker.attempt(&key()).is_probe());
        f.breaker.record(&key(), Signal::Success, true);

        let reasons: Vec<Reason> = f.breaker.trace().iter().map(|t| t.reason).collect();
        assert_eq!(
            reasons,
            vec![Reason::Overload, Reason::ProbeGranted, Reason::Recovered]
        );
        let observed = f.breaker.trace();
        assert_eq!(observed[0].at_rfc3339(), "2026-10-16T12:00:00Z");
        assert_eq!(observed[1].at_rfc3339(), "2026-10-16T12:00:04Z");
        assert_eq!(
            observed[0].until_rfc3339().as_deref(),
            Some("2026-10-16T12:00:04Z")
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
    fn two_credentials_of_one_instance_are_independent() {
        let f = exact();
        let a = Key::new("nim", Some("vault:key-a".to_string())).unwrap();
        let b = Key::new("nim", Some("vault:key-b".to_string())).unwrap();
        for _ in 0..3 {
            f.breaker.record(&a, Signal::Overload { hint: None }, false);
        }
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
    fn the_trace_is_bounded() {
        let f = exact();
        f.open_by_overload(3);
        for _ in 0..(TRACE_CAP + 40) {
            let until = f.breaker.state(&key()).until.unwrap();
            f.clock.advance((until - f.clock.now()).to_std().unwrap());
            assert!(f.breaker.attempt(&key()).is_probe());
            f.breaker
                .record(&key(), Signal::Overload { hint: None }, true);
        }
        assert_eq!(f.breaker.trace().len(), TRACE_CAP);
    }

    #[test]
    fn the_jitter_sources_stay_in_the_unit_interval() {
        let system = SystemJitter::default();
        let draws: Vec<f64> = (0..32).map(|_| system.frac()).collect();
        assert!(draws.iter().all(|f| (0.0..1.0).contains(f)));
        assert!(
            draws.windows(2).any(|w| w[0] != w[1]),
            "a live source is not a constant"
        );
        assert!(SystemJitter::new().frac() < 1.0);
        let seeded = SeededJitter::new(7);
        let replay = SeededJitter::new(7);
        assert_eq!(seeded.frac(), replay.frac(), "the same seed replays");
        assert_eq!(FixedJitter::new(2.0).frac(), 1.0, "clamped");
        assert_eq!(FixedJitter::new(f64::NAN).frac(), 0.0, "not finite");
    }

    #[test]
    fn a_degenerate_policy_never_opens_a_zero_cooldown() {
        // MUTATION: dropping `.max(policy.min_cooldown)` makes each of these zero.
        let floor = Policy::default().min_cooldown;
        assert!(floor > Duration::ZERO);
        for policy in [
            // Jitter takes the whole cooldown away on a zero draw.
            Policy {
                jitter_ratio: 1.0,
                ..Policy::default()
            },
            Policy {
                base_backoff: Duration::ZERO,
                ..Policy::default()
            },
            Policy {
                jitter_ratio: f64::NAN,
                ..Policy::default()
            },
        ] {
            assert_eq!(overload_delay(&policy, 1, None, 0.0), floor, "{policy:?}");
        }
        // And through the machine: the pause really ends after the floor, not at once.
        let f = fixture_with(Policy {
            jitter_ratio: 1.0,
            overload_threshold: 1,
            ..Policy::default()
        });
        f.breaker
            .record(&key(), Signal::Overload { hint: None }, false);
        assert!(!f.breaker.attempt(&key()).is_allowed(), "not open-and-over");
    }

    #[test]
    fn no_hint_and_no_ceiling_can_overflow_or_zero_a_cooldown() {
        // MUTATION: removing `cap_hint` from `overload_delay` lets 1e15 s through
        // unbounded (the date then saturates, but the hint is no longer capped);
        // reverting `after` to `now + chrono_d(..)` panics or yields zero here.
        use crate::chat::provider::limits::classify::MAX_RETRY_AFTER;
        let policy = Policy::default();
        for hint in [
            Duration::from_secs(1_000_000_000_000_000),
            Duration::from_secs(u64::MAX),
            Duration::MAX,
        ] {
            let delay = overload_delay(&policy, 1, Some(hint), 0.5);
            assert_eq!(delay, MAX_RETRY_AFTER, "the hint is capped, never dropped");
        }
        // A ceiling near the end of time does not panic `from_secs_f64`.
        let huge = Policy {
            base_backoff: Duration::MAX,
            max_backoff: Duration::MAX,
            jitter_ratio: 0.0,
            ..Policy::default()
        };
        assert!(overload_delay(&huge, 8, None, 1.0) > Duration::from_secs(3600));
        // And a pause that long still has an end, in the far future, not "now".
        let f = fixture_with(Policy {
            overload_threshold: 1,
            ..huge
        });
        let t = f
            .breaker
            .record(
                &key(),
                Signal::Overload {
                    hint: Some(Duration::MAX),
                },
                false,
            )
            .unwrap();
        assert!(t.until.unwrap() > f.clock.now() + chrono::Duration::hours(23));
        assert!(!f.breaker.attempt(&key()).is_allowed());
    }

    #[test]
    fn a_key_only_ever_holds_a_reference() {
        // MUTATION: dropping the `is_reference` check in `Key::new` makes the
        // secret-shaped values below construct a key.
        for secret in [
            "sk-live-abcdef0123456789",
            "vault:",
            "env:",
            "vault:has space",
            "plain-value",
            "Bearer abc",
        ] {
            assert_eq!(
                Key::new("nim", Some(secret.to_string())),
                Err(InvalidReference),
                "{secret}"
            );
        }
        let key = Key::new("nim", Some("env:NIM_KEY".to_string())).unwrap();
        assert_eq!(key.credential(), Some("env:NIM_KEY"));
        assert_eq!(format!("{key}"), "nim/env:NIM_KEY");
        assert_eq!(format!("{key:?}"), "Key(nim/env:NIM_KEY)");
        assert_eq!(Key::for_instance("nim").credential(), None);
        assert_eq!(format!("{:?}", Key::for_instance("nim")), "Key(nim)");
        // The error never echoes the rejected value.
        let message = InvalidReference.to_string();
        assert!(!message.contains("sk-"));
        let _: &dyn std::error::Error = &InvalidReference;
    }

    #[test]
    fn decisions_and_transitions_answer_for_both_outcomes() {
        let f = exact();
        let allowed = f.breaker.attempt(&key());
        assert_eq!(allowed.until(), None);
        let unseen = f.breaker.state(&Key::for_instance("never-seen"));
        assert_eq!(
            (unseen.phase, unseen.status),
            (Phase::Closed, RateLimitStatus::Ok)
        );
        assert!(!unseen.is_open());
        f.open_by_overload(3);
        let refused = f.breaker.attempt(&key());
        assert_eq!(refused.status(), RateLimitStatus::Overloaded);
        assert!(!refused.is_probe());
        assert_eq!(f.breaker.policy().overload_threshold, 3);
        let breaker = Breaker::with_defaults(f.clock.clone(), Arc::new(FixedJitter::new(0.0)));
        assert_eq!(breaker.policy(), &Policy::default());
    }

    #[test]
    fn an_rl3_reading_maps_to_the_right_signal() {
        // MUTATION: mapping `QuotaExhausted` to an overload signal (so that a
        // spent quota opens a breaker) turns the first assertion red.
        assert_eq!(
            Signal::of(&Verdict::QuotaExhausted { reset_at: None }),
            None,
            "a spent quota is the quota registry's, never a breaker's"
        );
        let hint = Some(Duration::from_secs(2));
        assert_eq!(
            Signal::of(&Verdict::Overloaded { retry_after: hint }),
            Some(Signal::Overload { hint })
        );
        assert_eq!(
            Signal::of(&Verdict::Unknown),
            Some(Signal::Overload { hint: None }),
            "an unreadable refusal is an overload, never an exhaustion"
        );
    }
}
