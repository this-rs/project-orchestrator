//! Provider outage detection: from a stream of call outcomes, decide when an
//! instance is **down** — a service interruption to persist and attribute —
//! rather than merely having blipped.
//!
//! # What this is, and what it is not
//!
//! A single failed call is a blip: the transient retry (`ChatEvent::Retrying`)
//! already covers it, unchanged. An *outage* is a run of failures that does not
//! clear on its own, and it is the event the user needs explained ("DeepSeek is
//! down, not your message"). This module owns exactly that decision, and nothing
//! else:
//!
//! - **Two triggers, ORed.** The instance is declared down as soon as EITHER
//!   `failure_threshold` consecutive failures are seen (default 3), OR the run
//!   of failures has lasted `degrade_window` without a success in between
//!   (default 30 s). The count catches a provider that fails fast; the window
//!   catches one that fails slowly, where a handful of timeouts can span half a
//!   minute before three of them line up in a row.
//! - **A blip is not an outage.** Any success resets the run and returns the
//!   instance to *up*. Two failures, a success, two more failures: never down.
//! - **One notice per transition.** Going down emits one notice; every further
//!   failure while down emits nothing; the first success emits one notice that
//!   the instance is back. Down and up are attributable (instance, masked cause,
//!   start of the outage, duration).
//!
//! # Ownership vs the other "instance unavailable" mechanisms
//!
//! Reading the tree (16/10/2026) there are already several health mechanisms,
//! each on a different object, and this detector is another — so the boundary
//! matters:
//!
//! - [`super::circuit_breaker::CircuitBreaker`] decides **whether to attempt** a
//!   call (control): consecutive failures open a circuit, a cooldown then a
//!   single half-open probe close it. Its output is "may I call?".
//! - `cognitive::candidates::HealthCache` decides the **current health** fed to
//!   `ModelFacts.healthy`, from an *active* probe cached for 60 s.
//! - This detector decides **what to announce** (observability): it produces a
//!   persisted, attributable event and no control signal at all.
//!
//! Those are three different outputs. They may all be fed from the same outcome
//! (the caller maps one `ProviderError` to one [`Outcome`] in one place), but
//! none of them double-counts another's decision: the breaker never emits an
//! event, this detector never refuses a call, and `HealthCache` never reads a
//! production outcome. In particular this detector does **not** write
//! `ModelFacts.healthy` — the warning in [`super::circuit_breaker`] about a
//! second writer there still holds.
//!
//! # Where the notice goes (channel choice)
//!
//! The task pointed at the conversation-segment model
//! (`cognitive::segment::{Dimension::Harness, BoundaryReason}`). That model is
//! the right *domain* home — a provider outage is a change on the harness axis —
//! but not the right *channel*: a [`super::cognitive::segment::Boundary`] is a
//! per-conversation, per-turn record (`at_turn`, and `from`/`to` segment ids
//! derived from fingerprints) whose whole point is byte-for-byte replayability
//! from a turn stream. An outage has no turn and no fingerprint, and a
//! boundary is keyed by conversation, not by provider.
//!
//! So the notice is carried by the **persisted session event stream** — the same
//! [`ChatEvent`] channel ERR-1 used to carry a typed cause on `Result`, and the
//! one `ProviderSwitched`/`ConversationRelayed` already use for provider-level
//! facts. [`ChatEvent::ProviderOutage`] is an additive variant of that existing
//! channel: no second channel is created, and a client that does not know it
//! ignores it. Wiring is a deliberate follow-up — a self-contained primitive
//! first, its wiring next, matching how [`super::circuit_breaker`] was landed —
//! because the two call sites it needs are the two transient-retry loops that
//! already emit [`ChatEvent::Retrying`] (`agent_runtime.rs` for the agent
//! engine, `manager.rs` for the legacy one), which are hot and in flight. The
//! move is the same in both: hold one [`ProviderOutage`] per `provider_id` beside
//! the retry state, feed each *final* call's [`Outcome`] to
//! [`ProviderOutage::record`], and emit `notice.to_chat_event(chrono::Utc::now())`
//! whenever it returns one.

use std::time::{Duration, Instant};

use super::circuit_breaker::Outcome;
use crate::chat::types::ChatEvent;

/// Consecutive failures that declare an outage, when no success interrupts them.
pub const DEFAULT_FAILURE_THRESHOLD: u32 = 3;

/// How long a run of failures may last before it is an outage on its own,
/// however few failures it holds.
pub const DEFAULT_DEGRADE_WINDOW: Duration = Duration::from_secs(30);

/// When an instance counts as down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutagePolicy {
    /// Consecutive failures that declare an outage (at least 1; 0 is read as 1).
    pub failure_threshold: u32,
    /// A run of failures this long declares an outage on its own.
    pub degrade_window: Duration,
}

impl Default for OutagePolicy {
    fn default() -> Self {
        Self {
            failure_threshold: DEFAULT_FAILURE_THRESHOLD,
            degrade_window: DEFAULT_DEGRADE_WINDOW,
        }
    }
}

impl OutagePolicy {
    /// A checked policy.
    #[must_use]
    pub fn new(failure_threshold: u32, degrade_window: Duration) -> Self {
        Self {
            failure_threshold,
            degrade_window,
        }
    }
}

/// Which way the state moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutagePhase {
    /// The instance was declared out of service.
    Down,
    /// The instance answered again.
    Up,
}

impl OutagePhase {
    /// Stable wire name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Down => "down",
            Self::Up => "up",
        }
    }
}

/// A transition of the outage state, ready to be attributed and persisted.
///
/// Times are monotonic ([`Instant`]); [`OutageNotice::to_chat_event`] turns them
/// into the wall-clock fields of the persisted [`ChatEvent`]. Keeping the
/// machine on `Instant` makes it deterministic and testable without a clock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutageNotice {
    /// The instance this is about.
    pub provider_id: String,
    /// `down` or `up`.
    pub phase: OutagePhase,
    /// Consecutive failures observed when the outage was declared (0 on `up`).
    pub failures: u32,
    /// When the outage began: the first failure of the run that became an outage.
    pub since: Instant,
    /// When this transition was observed.
    pub at: Instant,
    /// Short, masked cause of the outage. Present on `down`, and carried on `up`
    /// (the outage the instance recovered from); never a URL, a path or a
    /// credential (the caller takes it from `errors::safe_cause`).
    pub cause: Option<String>,
    /// How long the outage lasted (`up` only).
    pub duration: Option<Duration>,
}

impl OutageNotice {
    /// The persisted, attributable event this notice becomes.
    ///
    /// `now_wall` is the wall clock at the moment the notice is emitted; the
    /// start of the outage is placed that far back from it, off the monotonic
    /// clocks, so the record is exact without this module holding a wall clock.
    #[must_use]
    pub fn to_chat_event(&self, now_wall: chrono::DateTime<chrono::Utc>) -> ChatEvent {
        let elapsed = self.at.saturating_duration_since(self.since);
        let back = chrono::Duration::from_std(elapsed).unwrap_or_default();
        ChatEvent::ProviderOutage {
            provider_id: self.provider_id.clone(),
            phase: self.phase.as_str().to_string(),
            cause: self.cause.clone(),
            failures: self.failures,
            started_at: now_wall - back,
            duration_ms: self.duration.map(|d| d.as_millis() as u64),
        }
    }
}

/// One instance's outage state.
#[derive(Debug, Clone)]
struct DownState {
    since: Instant,
    cause: Option<String>,
}

/// The outage state machine of one provider instance.
///
/// Not thread-safe (no internal lock): wrap it in a `Mutex`, or one per instance
/// behind a map keyed by `provider_id`, exactly like [`super::circuit_breaker::CircuitBreaker`].
#[derive(Debug, Clone)]
pub struct ProviderOutage {
    provider_id: String,
    policy: OutagePolicy,
    consecutive_failures: u32,
    /// Start of the current run of failures; cleared by any success.
    failure_run_start: Option<Instant>,
    /// `Some` while the instance is down.
    down: Option<DownState>,
}

impl ProviderOutage {
    /// A detector for one instance, starting up and healthy.
    #[must_use]
    pub fn new(provider_id: impl Into<String>, policy: OutagePolicy) -> Self {
        Self {
            provider_id: provider_id.into(),
            policy,
            consecutive_failures: 0,
            failure_run_start: None,
            down: None,
        }
    }

    /// Whether the instance is currently declared down.
    #[must_use]
    pub fn is_down(&self) -> bool {
        self.down.is_some()
    }

    /// Consecutive failures seen so far in the current run.
    #[must_use]
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }

    /// Records the outcome of one call and returns the notice **only** when this
    /// outcome flipped the state. Feed it one [`Outcome`] per *final* call — the
    /// caller maps a `ProviderError` through `ProviderError::retryable()` once the
    /// transient retry is spent; a retried-and-then-succeeded call is one `Success`.
    ///
    /// `cause` is the masked cause of the failure (ignored on success); pass what
    /// `errors::safe_cause` accepted, never the raw provider text.
    pub fn record(
        &mut self,
        outcome: Outcome,
        now: Instant,
        cause: Option<&str>,
    ) -> Option<OutageNotice> {
        match outcome {
            Outcome::Success => {
                self.consecutive_failures = 0;
                self.failure_run_start = None;
                let down = self.down.take()?;
                Some(OutageNotice {
                    provider_id: self.provider_id.clone(),
                    phase: OutagePhase::Up,
                    failures: 0,
                    since: down.since,
                    at: now,
                    // The cause of the outage that just ended travels with the
                    // recovery, so the `up` event is self-contained.
                    cause: down.cause,
                    duration: Some(now.saturating_duration_since(down.since)),
                })
            }
            Outcome::Failure => {
                self.consecutive_failures = self.consecutive_failures.saturating_add(1);
                let since = *self.failure_run_start.get_or_insert(now);
                if self.down.is_some() {
                    // Already down: this failure is part of the same outage.
                    return None;
                }
                let by_count = self.consecutive_failures >= self.policy.failure_threshold.max(1);
                let by_window = now.saturating_duration_since(since) >= self.policy.degrade_window;
                if !by_count && !by_window {
                    return None;
                }
                let cause = cause.map(str::to_owned);
                self.down = Some(DownState {
                    since,
                    cause: cause.clone(),
                });
                Some(OutageNotice {
                    provider_id: self.provider_id.clone(),
                    phase: OutagePhase::Down,
                    failures: self.consecutive_failures,
                    since,
                    at: now,
                    cause,
                    duration: None,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(base: Instant, secs: u64) -> Instant {
        base + Duration::from_secs(secs)
    }

    #[test]
    fn fewer_than_threshold_failures_stay_up() {
        let base = Instant::now();
        let mut outage = ProviderOutage::new("deepseek", OutagePolicy::default());
        assert!(outage
            .record(Outcome::Failure, base, Some("boom"))
            .is_none());
        assert!(outage
            .record(Outcome::Failure, at(base, 1), Some("boom"))
            .is_none());
        assert!(!outage.is_down());
    }

    #[test]
    fn three_consecutive_failures_flip_down_once() {
        let base = Instant::now();
        let mut outage = ProviderOutage::new("deepseek", OutagePolicy::default());
        assert!(outage
            .record(Outcome::Failure, base, Some("conn refused"))
            .is_none());
        assert!(outage
            .record(Outcome::Failure, at(base, 1), Some("conn refused"))
            .is_none());
        let notice = outage
            .record(Outcome::Failure, at(base, 2), Some("conn refused"))
            .expect("the third consecutive failure is the outage");
        assert_eq!(notice.phase, OutagePhase::Down);
        assert_eq!(notice.provider_id, "deepseek");
        assert_eq!(notice.failures, 3);
        assert_eq!(notice.since, base, "the outage began at the first failure");
        assert_eq!(notice.cause.as_deref(), Some("conn refused"));
        assert!(outage.is_down());

        // A fourth failure is part of the same outage: no second notice.
        assert!(outage
            .record(Outcome::Failure, at(base, 3), Some("conn refused"))
            .is_none());
    }

    #[test]
    fn a_success_ends_the_outage_and_reports_its_length() {
        let base = Instant::now();
        let mut outage = ProviderOutage::new("deepseek", OutagePolicy::default());
        for i in 0..3 {
            outage.record(Outcome::Failure, at(base, i), Some("conn refused"));
        }
        assert!(outage.is_down());
        let notice = outage
            .record(Outcome::Success, at(base, 12), None)
            .expect("the first success ends the outage");
        assert_eq!(notice.phase, OutagePhase::Up);
        assert_eq!(notice.since, base);
        assert_eq!(notice.duration, Some(Duration::from_secs(12)));
        assert!(!outage.is_down());
        // A second success is not a transition.
        assert!(outage
            .record(Outcome::Success, at(base, 13), None)
            .is_none());
    }

    #[test]
    fn a_blip_is_not_an_outage() {
        let base = Instant::now();
        let mut outage = ProviderOutage::new("deepseek", OutagePolicy::default());
        outage.record(Outcome::Failure, base, Some("x"));
        outage.record(Outcome::Failure, at(base, 1), Some("x"));
        outage.record(Outcome::Success, at(base, 2), None);
        // The count restarted: two more failures are still short of three.
        assert!(outage
            .record(Outcome::Failure, at(base, 3), Some("x"))
            .is_none());
        assert!(outage
            .record(Outcome::Failure, at(base, 4), Some("x"))
            .is_none());
        assert!(!outage.is_down());
    }

    #[test]
    fn the_degrade_window_flips_down_below_the_threshold() {
        let base = Instant::now();
        let policy = OutagePolicy::new(5, Duration::from_secs(30));
        let mut outage = ProviderOutage::new("slowpoke", policy);
        // Two failures, 31 s apart: the count is not reached, the window is.
        assert!(outage
            .record(Outcome::Failure, base, Some("gateway timeout"))
            .is_none());
        let notice = outage
            .record(Outcome::Failure, at(base, 31), Some("gateway timeout"))
            .expect("a 31-second run of failures is an outage on its own");
        assert_eq!(notice.phase, OutagePhase::Down);
        assert_eq!(notice.failures, 2);
    }

    #[test]
    fn the_notice_becomes_a_persisted_attributable_event() {
        let base = Instant::now();
        let mut outage = ProviderOutage::new("deepseek-prod", OutagePolicy::default());
        for i in 0..3 {
            outage.record(Outcome::Failure, at(base, i), Some("connection refused"));
        }
        let notice = outage
            .record(Outcome::Success, at(base, 5), None)
            .expect("up");
        let wall = chrono::Utc::now();
        let event = notice.to_chat_event(wall);

        let ChatEvent::ProviderOutage {
            provider_id,
            phase,
            cause,
            failures,
            started_at,
            duration_ms,
        } = &event
        else {
            panic!("expected a provider_outage event, got {event:?}");
        };
        assert_eq!(provider_id, "deepseek-prod");
        assert_eq!(phase, "up");
        assert_eq!(*duration_ms, Some(5_000));
        assert_eq!(*failures, 0);
        assert_eq!(cause.as_deref(), Some("connection refused"));
        assert_eq!(*started_at, wall - chrono::Duration::seconds(5));
    }

    #[test]
    fn dropping_a_leaky_cause_does_not_change_the_transition() {
        // The caller hands `None` when the detail was not safe: the outage is
        // still declared on the third failure, just without a cause.
        let base = Instant::now();
        let mut outage = ProviderOutage::new("deepseek", OutagePolicy::default());
        assert!(outage.record(Outcome::Failure, base, None).is_none());
        assert!(outage.record(Outcome::Failure, at(base, 1), None).is_none());
        let notice = outage
            .record(Outcome::Failure, at(base, 2), None)
            .expect("the third failure declares the outage, cause or not");
        assert_eq!(notice.phase, OutagePhase::Down);
        assert!(notice.cause.is_none());
    }
}
