//! Per-instance circuit breaker: turns a stream of call outcomes into a
//! live "is this instance healthy right now" signal.
//!
//! This module is intentionally standalone: it has no dependency on
//! [`super::catalog`], [`super::resolver`], [`super::cognitive`] or
//! [`super::errors`]. It knows nothing about providers, HTTP status codes or
//! `nexus_claude::agent::ProviderError` — the caller decides what counts as
//! a [`Outcome::Failure`] (see `ProviderError::retryable()` in nexus, or
//! `OpenFailure.retryable` in [`super::errors`]) and feeds it in.
//!
//! Wiring this breaker's output into `ModelFacts.healthy`
//! (`cognitive::candidates`) or into `StoreCatalog::with_unhealthy`
//! (`catalog`) is a deliberate follow-up, not done here: at the time this
//! module was written, both `catalog.rs` and `cognitive/candidates.rs` had
//! several worktrees actively modifying them (routing-decision work,
//! review-fix passes); adding this new, self-contained file avoids adding a
//! merge surface to that in-flight work. See decision on the plan
//! "Résilience & routage multi-provider inspirés OmniRoute".
//!
//! # Design
//!
//! Three states, like a standard (Hystrix-style) breaker:
//!
//! - **Closed**: calls are attempted; consecutive failures are counted.
//!   `failure_threshold` consecutive failures opens the circuit. Any
//!   success resets the counter to zero.
//! - **Open**: calls are refused ([`CircuitBreaker::can_attempt`] returns
//!   `false`) until `cooldown` has elapsed since opening. Once elapsed, the
//!   *next* [`CircuitBreaker::can_attempt`] call transitions to `HalfOpen`
//!   and returns `true` exactly once — this is the single probe attempt.
//! - **HalfOpen**: exactly one call is in flight (the probe). A
//!   [`Outcome::Success`] closes the circuit (back to `Closed` with the
//!   counter at zero). A [`Outcome::Failure`] reopens it with a fresh
//!   cooldown window. If neither is reported before `probe_timeout`
//!   elapses, the probe is treated as abandoned — see below.
//!
//! ## The gap this fixes (found auditing OmniRoute's `adaptiveCircuit.ts`)
//!
//! OmniRoute's breaker closes on *any* success it is told about, even one
//! reported while the circuit is `Open` and no probe was ever granted (a
//! call started before the circuit opened, or a caller that never checks
//! `can_attempt`, reporting its outcome late). That lets a circuit close
//! without ever having passed through a real probe.
//!
//! This implementation refuses that: [`CircuitBreaker::record`] only acts
//! on a `Success` when the state is `HalfOpen` (reported in response to the
//! one attempt `can_attempt` granted) or `Closed` (where it is a no-op
//! beyond resetting the counter, since the circuit was never open). A
//! `Success` reported while `Open` is dropped — see
//! `a_late_success_reported_while_open_does_not_close_the_circuit` below.
//!
//! ## The abandoned-probe gap (found in review of PR #753)
//!
//! Granting a probe (`can_attempt` returning `true` from `Open`) moves the
//! breaker to `HalfOpen`, which refuses every subsequent `can_attempt`
//! unconditionally — on the assumption that `record` will eventually be
//! called with the probe's outcome. If the probe's call is cancelled or its
//! task is dropped before that (a cancelled turn, a dropped future) with no
//! `record` call ever made, the breaker was wedged in `HalfOpen` forever:
//! no call could ever be attempted again.
//!
//! `probe_timeout` fixes this: `can_attempt`, when called while `HalfOpen`,
//! first checks whether the in-flight probe has been waiting longer than
//! `probe_timeout`. If so, the probe is treated as an implicit `Failure`
//! (the circuit reopens with a fresh cooldown measured from now) before the
//! normal `Open`-state probe-granting logic runs — so a caller that keeps
//! polling `can_attempt` is eventually granted a new probe instead of
//! waiting forever on one that will never resolve.
//!
//! # Ownership vs the other "instance unavailable" mechanisms
//!
//! Review of PR #753 rightly asked which mechanism owns model health, so
//! the same failure is not counted twice. Reading the tree (12/10/2026),
//! there are three pre-existing ones, each on a *different* object, and
//! this breaker is a fourth. The boundary that keeps them from
//! double-counting is the **kind of call whose outcome each one listens
//! to** — not the error itself:
//!
//! - `nexus-tools/src/search/resilience.rs::Breaker` — a near-identical
//!   3-state breaker, but for **web search engines** (WebSearch / Obscura),
//!   in another crate and another layer. Unrelated to LLM provider health;
//!   do not confuse the two `Breaker` types, do not merge them.
//! - `cognitive::decider::ProbeLog` — the **capabilities** of an
//!   (instance, model): does it call tools, is it usable at all. Feeds
//!   `supports_tools` / the "capability unknown" rejection, which is
//!   orthogonal to health (`RejectReason::NoTools` ≠
//!   `RejectReason::Unhealthy`). Listens only to *capability probes* fired
//!   when a candidate pool is built.
//! - `cognitive::candidates::HealthCache` — the **current** health of an
//!   instance, from an *active* probe cached for a 60 s window. This is
//!   today's source of `ModelFacts.healthy`. Listens only to its own
//!   synthetic probe.
//!
//! This breaker owns a fourth thing: health observed **passively**, from
//! the outcomes of *real production calls*, with anti-flapping (cooldown +
//! single half-open probe) that a per-window active cache cannot give. So
//! when it is wired (OR-3bis) it must NOT become a second writer of
//! `ModelFacts.healthy` beside `HealthCache`; it sits *in front* of it — an
//! `Open` breaker short-circuits `healthy` to `false` without an active
//! probe, otherwise `HealthCache` decides as it does today. Each failure is
//! then counted once: production outcomes feed this breaker, the synthetic
//! health probe feeds `HealthCache`, capability probes feed `ProbeLog`.
//! `force_open_until` is the one door the QUOTA task writes through, so a
//! known reset time also flows through this single short-circuit rather
//! than a parallel one.
use crate::neo4j::traits::*;

use std::time::{Duration, Instant};

/// The three states of the breaker, exposed for observability/logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Calls are attempted; `failures` consecutive failures have been seen.
    Closed,
    /// Calls are refused until the cooldown elapses.
    Open,
    /// One probe call is in flight; its outcome decides the next state.
    HalfOpen,
}

/// The outcome of one call, as judged by the caller.
///
/// This breaker does not interpret HTTP status codes or provider error
/// types itself — see the module doc. The caller maps whatever error
/// taxonomy it has (e.g. `ProviderError::retryable()` in nexus) to one of
/// these two variants before calling [`CircuitBreaker::record`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The call succeeded (or failed in a way that should not count against
    /// the instance, e.g. a client error unrelated to provider health).
    Success,
    /// The call failed in a way that should count against the instance's
    /// health (timeout, 5xx, connection refused, rate limited, ...).
    Failure,
}

/// A circuit breaker for one provider instance.
///
/// Not thread-safe by itself (no internal lock): wrap it in a `Mutex` (or
/// one per instance behind a `DashMap`) at the call site, matching how
/// `catalog::StoreCatalog` already keys per-instance state.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    cooldown: Duration,
    /// How long a granted probe (`HalfOpen`) is given to report its outcome
    /// before it is treated as abandoned. See the module doc.
    probe_timeout: Duration,
    state: CircuitState,
    consecutive_failures: u32,
    /// Set when `state == Open`: the instant at or after which the next
    /// `can_attempt` call is allowed to grant a probe.
    next_probe_at: Option<Instant>,
    /// Set when `state == HalfOpen`: the instant after which the in-flight
    /// probe is considered abandoned.
    probe_deadline: Option<Instant>,
}

impl CircuitBreaker {
    /// Creates a new breaker, starting `Closed` with a zero failure count.
    ///
    /// `failure_threshold` must be at least 1 (a threshold of 0 would open
    /// on the very first recorded failure with no chance to accumulate,
    /// which is a degenerate but not unreasonable choice — it is not
    /// rejected here, just documented).
    ///
    /// `probe_timeout` bounds how long a granted probe may stay in flight
    /// before it is treated as abandoned (see the module doc); it is a
    /// required, explicit parameter rather than a default, since a value
    /// that is too long re-creates the deadlock this fixes in slow motion,
    /// and a value that is too short can abandon a probe that was still
    /// genuinely in flight, inviting a stampede of near-simultaneous
    /// probes. Callers should size it to their own call timeout, not to
    /// `cooldown`.
    #[must_use]
    pub fn new(failure_threshold: u32, cooldown: Duration, probe_timeout: Duration) -> Self {
        Self {
            failure_threshold,
            cooldown,
            probe_timeout,
            state: CircuitState::Closed,
            consecutive_failures: 0,
            next_probe_at: None,
            probe_deadline: None,
        }
    }

    /// The current state, for logging/metrics. Does not mutate.
    #[must_use]
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// When this breaker will next allow an attempt, if it is currently
    /// `Open`. `None` when `Closed` (every attempt is allowed) or
    /// `HalfOpen` (one probe is already in flight; see [`Self::can_attempt`]
    /// and the probe-timeout behaviour in the module doc).
    ///
    /// Exposed so an external, better-informed signal (a provider's
    /// `Retry-After` header, a Claude Code `rate_limit_event`, a Codex
    /// reset timestamp) can read — and, via [`Self::force_open_until`],
    /// drive — the same cooldown clock instead of a second, parallel
    /// quota mechanism duplicating this one.
    #[must_use]
    pub fn open_until(&self) -> Option<Instant> {
        match self.state {
            CircuitState::Open => self.next_probe_at,
            CircuitState::Closed | CircuitState::HalfOpen => None,
        }
    }

    /// Forces the circuit `Open` until `until`, regardless of the current
    /// failure count or state. Intended for an external signal that already
    /// knows the retry time — a provider's `Retry-After` header, or a
    /// quota mechanism reading a Claude Code `rate_limit_event` / Codex
    /// reset timestamp — so that mechanism drives this breaker's cooldown
    /// clock directly instead of needing its own, separate one.
    pub fn force_open_until(&mut self, until: Instant) {
        self.state = CircuitState::Open;
        self.next_probe_at = Some(until);
        self.probe_deadline = None;
    }

    /// Whether a call may be attempted right now.
    ///
    /// This is the method that can transition `Open` -> `HalfOpen`: it is
    /// mutating (`&mut self`) specifically so that granting the single
    /// probe attempt is an atomic, observable step from the breaker's own
    /// point of view — not two racy reads (“is cooldown over” then “mark
    /// probe in flight”) split across the call site.
    ///
    /// It is also the method that detects an abandoned probe: see the
    /// module doc's "abandoned-probe gap" section.
    pub fn can_attempt(&mut self, now: Instant) -> bool {
        if self.state == CircuitState::HalfOpen && self.probe_abandoned(now) {
            // The in-flight probe never reported; treat it as a failure so
            // the breaker does not stay wedged in `HalfOpen` forever, then
            // fall through to the normal `Open`-state logic below.
            self.open(now);
        }

        match self.state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => false, // a (still-live) probe is in flight
            CircuitState::Open => self.try_grant_probe(now),
        }
    }

    fn probe_abandoned(&self, now: Instant) -> bool {
        self.probe_deadline.is_some_and(|deadline| now >= deadline)
    }

    fn try_grant_probe(&mut self, now: Instant) -> bool {
        let due = self.next_probe_at.is_some_and(|at| now >= at);
        if due {
            self.state = CircuitState::HalfOpen;
            self.next_probe_at = None;
            self.probe_deadline = Some(now + self.probe_timeout);
            true
        } else {
            false
        }
    }

    /// Records the outcome of a call made because `can_attempt` returned
    /// `true`. Calling this after `can_attempt` returned `false` is a
    /// caller bug (the call should not have been made); this method
    /// tolerates it by being a no-op on points that would otherwise close
    /// an unprobed `Open` circuit — see the module doc "gap" section.
    ///
    /// Calling this *late* for a probe whose `probe_timeout` has already
    /// elapsed — whether or not [`Self::can_attempt`] was polled in
    /// between to observe that — is also tolerated as a no-op: this method
    /// runs the same abandonment check `can_attempt` does before matching
    /// on the outcome, so a straggler result for a probe the breaker would
    /// already consider abandoned cannot retroactively change its state
    /// (in particular, cannot close the circuit on a late `Success`) just
    /// because nobody happened to call `can_attempt` in between.
    pub fn record(&mut self, outcome: Outcome, now: Instant) {
        if self.state == CircuitState::HalfOpen && self.probe_abandoned(now) {
            self.open(now);
        }

        match (self.state, outcome) {
            (CircuitState::Closed, Outcome::Success) => {
                self.consecutive_failures = 0;
            }
            (CircuitState::Closed, Outcome::Failure) => {
                self.consecutive_failures += 1;
                if self.consecutive_failures >= self.failure_threshold.max(1) {
                    self.open(now);
                }
            }
            (CircuitState::HalfOpen, Outcome::Success) => {
                self.state = CircuitState::Closed;
                self.consecutive_failures = 0;
                self.probe_deadline = None;
            }
            (CircuitState::HalfOpen, Outcome::Failure) => {
                self.open(now);
            }
            (CircuitState::Open, _) => {
                // Deliberately ignored: no outcome reported while the
                // circuit is Open (i.e. not the granted probe, or a
                // straggler for a probe already reopened as abandoned)
                // may change its state. A late Success must not close it
                // (the OmniRoute gap); a late Failure is already reflected
                // in the circuit being open and must not reset the
                // cooldown clock, which would let a slow caller keep it
                // open indefinitely.
            }
        }
    }

    fn open(&mut self, now: Instant) {
        self.state = CircuitState::Open;
        self.next_probe_at = Some(now + self.cooldown);
        self.probe_deadline = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::*;

    const COOLDOWN: Duration = Duration::from_secs(60);
    const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

    fn breaker() -> CircuitBreaker {
        CircuitBreaker::new(3, COOLDOWN, PROBE_TIMEOUT)
    }

    #[test]
    fn starts_closed_with_no_failures() {
        let b = breaker();
        assert_eq!(b.state(), CircuitState::Closed);
    }

    #[test]
    fn fewer_than_threshold_failures_stay_closed() {
        let mut b = breaker();
        let t0 = Instant::now();
        b.record(Outcome::Failure, t0);
        b.record(Outcome::Failure, t0);
        assert_eq!(b.state(), CircuitState::Closed);
    }

    #[test]
    fn threshold_consecutive_failures_opens_the_circuit() {
        let mut b = breaker();
        let t0 = Instant::now();
        b.record(Outcome::Failure, t0);
        b.record(Outcome::Failure, t0);
        b.record(Outcome::Failure, t0);
        assert_eq!(b.state(), CircuitState::Open);
    }

    #[test]
    fn a_success_resets_the_consecutive_failure_count() {
        let mut b = breaker();
        let t0 = Instant::now();
        b.record(Outcome::Failure, t0);
        b.record(Outcome::Failure, t0);
        b.record(Outcome::Success, t0);
        // Two more failures should not be enough to open (count was reset).
        b.record(Outcome::Failure, t0);
        b.record(Outcome::Failure, t0);
        assert_eq!(b.state(), CircuitState::Closed);
    }

    #[test]
    fn can_attempt_refuses_calls_while_open_and_within_cooldown() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        assert_eq!(b.state(), CircuitState::Open);
        assert!(!b.can_attempt(t0 + Duration::from_secs(30)));
        // Refusing a probe must not itself change the state.
        assert_eq!(b.state(), CircuitState::Open);
    }

    #[test]
    fn can_attempt_grants_exactly_one_probe_after_cooldown() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        let after_cooldown = t0 + Duration::from_secs(61);
        assert!(b.can_attempt(after_cooldown));
        assert_eq!(b.state(), CircuitState::HalfOpen);
        // A second concurrent caller must not get a second probe while the
        // first one is still within its probe_timeout.
        assert!(!b.can_attempt(after_cooldown + Duration::from_secs(1)));
    }

    #[test]
    fn a_successful_probe_closes_the_circuit() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        let probe_at = t0 + Duration::from_secs(61);
        assert!(b.can_attempt(probe_at));
        b.record(Outcome::Success, probe_at);
        assert_eq!(b.state(), CircuitState::Closed);
    }

    #[test]
    fn a_failed_probe_reopens_with_a_fresh_cooldown() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        let probe_at = t0 + Duration::from_secs(61);
        assert!(b.can_attempt(probe_at));
        b.record(Outcome::Failure, probe_at);
        assert_eq!(b.state(), CircuitState::Open);
        // The fresh cooldown is measured from the failed probe, not from
        // the original opening: right after the failed probe, still closed
        // to new attempts.
        assert!(!b.can_attempt(probe_at + Duration::from_secs(1)));
        assert!(b.can_attempt(probe_at + Duration::from_secs(61)));
    }

    /// The OmniRoute gap this breaker deliberately avoids: see module doc.
    #[test]
    fn a_late_success_reported_while_open_does_not_close_the_circuit() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        assert_eq!(b.state(), CircuitState::Open);
        // No call to can_attempt here: this success is reported by some
        // straggler call that was in flight before the circuit opened, or
        // by a caller that skipped the can_attempt check entirely.
        b.record(Outcome::Success, t0 + Duration::from_secs(1));
        assert_eq!(
            b.state(),
            CircuitState::Open,
            "a success not granted via can_attempt must not close an Open circuit"
        );
    }

    /// Symmetric case: a late failure reported while Open must not reset
    /// the cooldown clock (which would let a slow, already-accounted-for
    /// failure keep extending the outage).
    #[test]
    fn a_late_failure_reported_while_open_does_not_extend_the_cooldown() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        // A straggler failure arrives mid-cooldown.
        b.record(Outcome::Failure, t0 + Duration::from_secs(30));
        // The original cooldown (relative to t0) must still be the one in
        // effect: it elapses at t0 + 60s, not later.
        assert!(b.can_attempt(t0 + Duration::from_secs(61)));
    }

    #[test]
    fn a_threshold_of_zero_is_treated_as_one() {
        let mut b = CircuitBreaker::new(0, COOLDOWN, PROBE_TIMEOUT);
        let t0 = Instant::now();
        b.record(Outcome::Failure, t0);
        assert_eq!(b.state(), CircuitState::Open);
    }

    /// The abandoned-probe gap fixed after review of PR #753: see module
    /// doc. A probe that is granted and then never reported (the caller's
    /// turn was cancelled, its task dropped) must not wedge the breaker in
    /// `HalfOpen` forever.
    #[test]
    fn an_abandoned_probe_times_out_and_the_circuit_reopens() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        let probe_at = t0 + Duration::from_secs(61);
        assert!(b.can_attempt(probe_at));
        assert_eq!(b.state(), CircuitState::HalfOpen);

        // The caller never calls record() again. Polling can_attempt
        // before probe_timeout elapses must not grant a second probe
        // (the first might still be genuinely in flight).
        let still_within_probe_timeout = probe_at + Duration::from_secs(5);
        assert!(!b.can_attempt(still_within_probe_timeout));
        assert_eq!(b.state(), CircuitState::HalfOpen);

        // Once probe_timeout has elapsed, the abandoned probe is treated
        // as a failure: the circuit reopens with a fresh cooldown measured
        // from now, rather than staying wedged in HalfOpen.
        let probe_timed_out = probe_at + PROBE_TIMEOUT + Duration::from_secs(1);
        assert!(!b.can_attempt(probe_timed_out));
        assert_eq!(b.state(), CircuitState::Open);

        // The fresh cooldown runs from the timeout, not from the original
        // opening.
        assert!(!b.can_attempt(probe_timed_out + Duration::from_secs(1)));
        assert!(b.can_attempt(probe_timed_out + COOLDOWN + Duration::from_secs(1)));
        assert_eq!(b.state(), CircuitState::HalfOpen);
    }

    /// A straggler outcome for a probe already reopened as abandoned must
    /// not retroactively change the state the breaker has moved on to.
    #[test]
    fn a_straggler_outcome_after_probe_abandonment_is_ignored() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        let probe_at = t0 + Duration::from_secs(61);
        assert!(b.can_attempt(probe_at));

        let probe_timed_out = probe_at + PROBE_TIMEOUT + Duration::from_secs(1);
        // can_attempt observes the abandonment and reopens.
        assert!(!b.can_attempt(probe_timed_out));
        assert_eq!(b.state(), CircuitState::Open);

        // The original (abandoned) probe's call finally returns, long
        // after the breaker moved on — reporting Success must not close
        // an Open circuit that no longer corresponds to that probe.
        b.record(Outcome::Success, probe_timed_out + Duration::from_secs(1));
        assert_eq!(b.state(), CircuitState::Open);
    }

    /// Same abandonment, but `record` is the first thing called after the
    /// deadline — nobody polled `can_attempt` in between. The breaker must
    /// still refuse to let a stale probe's outcome close the circuit.
    #[test]
    fn a_late_success_via_record_alone_after_the_deadline_does_not_close_the_circuit() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        let probe_at = t0 + Duration::from_secs(61);
        assert!(b.can_attempt(probe_at));
        assert_eq!(b.state(), CircuitState::HalfOpen);

        // No further can_attempt call: record() alone must detect the
        // abandonment.
        let probe_timed_out = probe_at + PROBE_TIMEOUT + Duration::from_secs(1);
        b.record(Outcome::Success, probe_timed_out);
        assert_eq!(
            b.state(),
            CircuitState::Open,
            "a stale probe's late success must not close the circuit, even \
             when record() (not can_attempt) is what first notices the \
             probe timed out"
        );
    }

    #[test]
    fn open_until_reports_the_cooldown_deadline_only_while_open() {
        let mut b = breaker();
        let t0 = Instant::now();
        assert_eq!(b.open_until(), None, "Closed has no cooldown deadline");

        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        assert_eq!(b.open_until(), Some(t0 + COOLDOWN));

        let probe_at = t0 + Duration::from_secs(61);
        assert!(b.can_attempt(probe_at));
        assert_eq!(
            b.open_until(),
            None,
            "HalfOpen has no cooldown deadline (it has a probe deadline instead)"
        );
    }

    #[test]
    fn force_open_until_overrides_state_and_cooldown_unconditionally() {
        let mut b = breaker();
        let t0 = Instant::now();
        // Closed, no failures recorded at all: an external signal (e.g. a
        // quota mechanism reading a provider's Retry-After) can still
        // force the circuit open until a time of its choosing.
        let until = t0 + Duration::from_secs(120);
        b.force_open_until(until);
        assert_eq!(b.state(), CircuitState::Open);
        assert_eq!(b.open_until(), Some(until));
        assert!(!b.can_attempt(until - Duration::from_secs(1)));
        assert!(b.can_attempt(until));
    }
}
