//! Provider outage handling, both halves: from a stream of call outcomes, decide
//! when an instance is **down** — a service interruption to persist and attribute
//! rather than merely having blipped — and decide what that state means for the
//! turn in flight: **held**, not closed, and resumed when the instance answers.
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
//!   persisted, attributable event. [`TurnHold`], which wraps it, is the *control*
//!   that this state implies — hold a turn whose retry budget is spent for a down
//!   instance, resume it when the instance answers — and nothing else: it never
//!   refuses a call, and never writes `ModelFacts.healthy`.
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
//! ignores it.
//!
//! Both engines are wired, each feeding the same detector where it can:
//!
//! - the **agent engine** (`agent_runtime.rs`) from the typed `ProviderError` it
//!   already has: one [`TurnHold`] per session beside its `RetryConfig`; a *final*
//!   call's [`Outcome`] is fed in one place, a `down`/`up` transition emits
//!   `notice.to_chat_event(chrono::Utc::now())`, and a failure whose ordinary retry
//!   budget is spent is **held** — probed at the hold's cadence, bounded by
//!   `max_hold` — instead of closing the turn;
//! - the **legacy Claude Code loop** (`manager.rs`) from the provider *text* it is
//!   given (`classify_api_error`), through [`outcome_of_legacy_failure`] — the one
//!   place that maps that taxonomy to [`Outcome`]. Its hold reuses [`TurnHold`] and
//!   `hold_on` exactly as the agent engine does, and re-sends the turn only where the
//!   ordinary retry already would (a turn that showed nothing, or text but no tool):
//!   an executed tool is never replayed.
use crate::neo4j::traits::*;

use std::time::{Duration, Instant};

use super::circuit_breaker::Outcome;
use crate::chat::types::{classify_api_error, is_out_of_credit_message, ApiErrorKind, ChatEvent};

/// Consecutive failures that declare an outage, when no success interrupts them.
pub const DEFAULT_FAILURE_THRESHOLD: u32 = 3;

/// How long a run of failures may last before it is an outage on its own,
/// however few failures it holds.
pub const DEFAULT_DEGRADE_WINDOW: Duration = Duration::from_secs(30);

/// `CHAT_PROVIDER_OUTAGE_FAILURES`: consecutive failures that declare an outage.
pub const ENV_FAILURE_THRESHOLD: &str = "CHAT_PROVIDER_OUTAGE_FAILURES";

/// `CHAT_PROVIDER_OUTAGE_WINDOW_SECS`: a run of failures this long is an outage.
pub const ENV_DEGRADE_WINDOW_SECS: &str = "CHAT_PROVIDER_OUTAGE_WINDOW_SECS";

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

    /// The policy the operator configured ([`ENV_FAILURE_THRESHOLD`],
    /// [`ENV_DEGRADE_WINDOW_SECS`]), else [`Default`]. An unreadable value keeps the
    /// default rather than refusing to serve chat.
    #[must_use]
    pub fn from_env() -> Self {
        let default = Self::default();
        Self {
            // A 0 is read as 1 by `ProviderOutage::record` — an outage on the first
            // failure — which is a usable kill switch: detect at once, or never.
            failure_threshold: env_u64(ENV_FAILURE_THRESHOLD)
                .map(|v| v as u32)
                .unwrap_or(default.failure_threshold),
            degrade_window: env_u64(ENV_DEGRADE_WINDOW_SECS)
                .map(Duration::from_secs)
                .unwrap_or(default.degrade_window),
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

    /// Replaces the policy (a configuration change; the run and the state stand).
    pub fn set_policy(&mut self, policy: OutagePolicy) {
        self.policy = policy;
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

/// Longest a turn is held for a down instance before the failure stands: long
/// enough to ride out a provider restart, short enough that a turn never hangs for
/// good.
pub const DEFAULT_MAX_HOLD: Duration = Duration::from_secs(15 * 60);

/// How often a held turn probes the instance again.
pub const DEFAULT_PROBE_INTERVAL: Duration = Duration::from_secs(5);

/// How long a turn may be held for a down instance, and how often it is probed.
///
/// The hold is what keeps a *down* provider from closing a turn. The ordinary retry
/// budget covers a blip; once it is spent the failure would otherwise end the turn,
/// and the user would read "the provider endpoint could not be reached" for a
/// service that is merely away. While the instance is known down the turn is held
/// instead — probed every `probe_interval` (each probe is one more call, whose
/// outcome is what tells the detector the instance is back) — and bounded by
/// `max_hold`, so an instance that never comes back still ends the turn on the real
/// cause rather than holding it for good.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoldPolicy {
    /// Longest a turn waits for the instance to answer again, in total.
    pub max_hold: Duration,
    /// Delay between two probes of a held turn.
    pub probe_interval: Duration,
}

impl Default for HoldPolicy {
    fn default() -> Self {
        Self {
            max_hold: DEFAULT_MAX_HOLD,
            probe_interval: DEFAULT_PROBE_INTERVAL,
        }
    }
}

impl HoldPolicy {
    /// A checked policy.
    #[must_use]
    pub fn new(max_hold: Duration, probe_interval: Duration) -> Self {
        Self {
            max_hold,
            probe_interval,
        }
    }

    /// The policy the operator configured (`CHAT_PROVIDER_HOLD_MAX_SECS`,
    /// `CHAT_PROVIDER_HOLD_PROBE_MS`), else [`Default`]. `CHAT_PROVIDER_HOLD_MAX_SECS=0`
    /// is the kill switch: a spent retry budget is never held, as before this feature.
    /// The probe interval has a floor of 50 ms — a held turn must not spin.
    #[must_use]
    pub fn from_env() -> Self {
        let default = Self::default();
        Self {
            max_hold: env_u64("CHAT_PROVIDER_HOLD_MAX_SECS")
                .map(Duration::from_secs)
                .unwrap_or(default.max_hold),
            probe_interval: env_u64("CHAT_PROVIDER_HOLD_PROBE_MS")
                .map(|ms| Duration::from_millis(ms.max(50)))
                .unwrap_or(default.probe_interval),
        }
    }
}

/// A `u64` from the environment, or `None` when the variable is unset or is not a
/// number. Never refuses to start over a typo: the caller falls back to its default.
fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

/// The **control** half of outage handling, on top of the detection of
/// [`ProviderOutage`]: it decides whether a turn whose retry budget is spent is
/// **held** (the instance is down) or closed.
///
/// The two halves share one state machine, hence one value and one lock: they must
/// see the same sequence of outcomes. The detector declares the instance down, and
/// the hold is exactly what that state means for the turn in flight — feed one
/// without the other and a hold outlives its outage, or an outage closes a turn it
/// should have held.
///
/// **No tool is ever replayed.** The hold only extends a retry the caller had
/// already allowed on a turn that showed nothing (`retryable_failure`), or one that
/// showed text but called no tool (`mid_generation_retryable`): a turn that already
/// ran a tool never reaches the hold, exactly as it never reaches the ordinary
/// retry. The guarantee of the restart recovery holds here too — an executed tool
/// is never run twice.
#[derive(Debug, Clone)]
pub struct TurnHold {
    outage: ProviderOutage,
    hold: HoldPolicy,
    /// When the current hold began — the start of the outage that declared it.
    /// Cleared when the instance is up again, or when the budget is given up.
    held_since: Option<Instant>,
}

impl TurnHold {
    /// A hold for one instance, with the default policies.
    #[must_use]
    pub fn new(provider_id: impl Into<String>) -> Self {
        Self::with_policies(provider_id, OutagePolicy::default(), HoldPolicy::default())
    }

    /// A hold with explicit policies.
    #[must_use]
    pub fn with_policies(
        provider_id: impl Into<String>,
        outage: OutagePolicy,
        hold: HoldPolicy,
    ) -> Self {
        Self {
            outage: ProviderOutage::new(provider_id, outage),
            hold,
            held_since: None,
        }
    }

    /// Replaces both policies (a configuration change; the state stands).
    pub fn set_policies(&mut self, outage: OutagePolicy, hold: HoldPolicy) {
        self.outage.set_policy(outage);
        self.hold = hold;
    }

    /// Whether the instance is currently declared down.
    #[must_use]
    pub fn is_down(&self) -> bool {
        self.outage.is_down()
    }

    /// Whether a turn is being held right now.
    #[must_use]
    pub fn is_held(&self) -> bool {
        self.held_since.is_some()
    }

    /// Consecutive failures seen so far in the current run.
    #[must_use]
    pub fn consecutive_failures(&self) -> u32 {
        self.outage.consecutive_failures()
    }

    /// Records one final call outcome and returns the notice **only** on a
    /// transition (see [`ProviderOutage::record`]). The hold follows the state: it
    /// begins with the outage and ends with the recovery.
    pub fn record(
        &mut self,
        outcome: Outcome,
        now: Instant,
        cause: Option<&str>,
    ) -> Option<OutageNotice> {
        let notice = self.outage.record(outcome, now, cause);
        match (notice.as_ref(), outcome) {
            // The instance just went down: the hold begins at the outage.
            (Some(notice), Outcome::Failure) => self.held_since = Some(notice.since),
            // It answered again: the hold is over, and so is the outage.
            (Some(_), Outcome::Success) => self.held_since = None,
            _ => {}
        }
        notice
    }

    /// Whether a turn whose retry budget is spent may go on: `Some(wait)` while the
    /// instance is down and the hold budget still allows — wait `wait`, then probe
    /// the instance again (send the turn once more, or read the next attempt's
    /// stream). `None` when the failure must stand: the instance is not down, or
    /// `max_hold` is spent.
    pub fn hold_on(&mut self, now: Instant) -> Option<Duration> {
        let since = self.held_since.filter(|_| self.outage.is_down())?;
        if now.saturating_duration_since(since) >= self.hold.max_hold {
            // The budget is spent: the failure stands. The state is cleared, so a
            // later outage is held on its own clock.
            self.held_since = None;
            return None;
        }
        Some(self.hold.probe_interval)
    }

    /// How long the current hold has lasted (`None`: not held).
    #[must_use]
    pub fn held_for(&self, now: Instant) -> Option<Duration> {
        self.held_since
            .map(|since| now.saturating_duration_since(since))
    }
}

/// The **legacy** engine's bridge to [`Outcome`].
///
/// The Claude Code loop (`manager.rs`) reads its failures from provider *text*
/// (`crate::chat::types::classify_api_error`), not from a typed `ProviderError`,
/// so it cannot hand a result to [`TurnHold`] / [`ProviderOutage`] the way the
/// agent engine does. This is the one place that maps that text taxonomy to the
/// health [`Outcome`] they consume, so the legacy loop feeds the **same**
/// detector as the native one instead of growing a second state machine.
///
/// `None` means "this is not a health failure": the instance answered, with a
/// refusal or a bad request of its own. That is what `ProviderError::retryable()`
/// decides on the typed path — a client error does not count against the
/// instance. A 402 (out of credit) is excluded on purpose: it is the cognitive
/// router's signal, not an outage.
///
/// The cause returned alongside is our own short token, never the raw provider
/// text, so an outage notice can never carry more than the user-facing message
/// already shows (the guarantee `errors::cause_of` gives on the typed path).
pub fn outcome_of_legacy_failure(text: &str) -> Option<(Outcome, &'static str)> {
    // A 402 (out of credit) is a refusal, not a health failure: the router cools
    // the instance down from the text the CLI prints, and the detector must not
    // also read it as an outage.
    if is_out_of_credit_message(text) {
        return None;
    }
    match classify_api_error(text) {
        ApiErrorKind::Retryable(reason) => {
            return Some((Outcome::Failure, retry_reason_token(&reason)));
        }
        ApiErrorKind::NonRetryable(_) => {}
    }
    // `classify_api_error` matches "timed out" case-sensitively; a client that
    // says "Timeout waiting for response after 30 seconds" is transient too.
    let lower = text.to_ascii_lowercase();
    if lower.contains("timeout") || lower.contains("timed out") {
        return Some((Outcome::Failure, LEGACY_TIMEOUT));
    }
    if UNREACHABLE_TEXT.iter().any(|token| lower.contains(token)) {
        return Some((Outcome::Failure, LEGACY_UNREACHABLE));
    }
    None
}

/// The token reported for a retryable failure, from `classify_api_error`'s own
/// vocabulary. An unknown reason falls back to a generic token rather than
/// echoing the raw text.
fn retry_reason_token(reason: &str) -> &'static str {
    match reason {
        "api_error" => LEGACY_API_ERROR,
        "overloaded_error" => LEGACY_OVERLOADED,
        "rate_limit_error" => LEGACY_RATE_LIMIT,
        "http_5xx" => LEGACY_HTTP_5XX,
        "timeout" => LEGACY_TIMEOUT,
        _ => LEGACY_UNKNOWN,
    }
}

/// Substrings that mean the endpoint could not be reached at all: the
/// network-level refusals a client prints when it cannot open a connection to the
/// API (a Bun/Node `fetch` failure, a DNS failure, a refused socket). Deliberately
/// narrow — a *locally* dead CLI ("Failed to connect to Claude CLI") is the resume
/// path's business, not an outage of the provider instance.
const UNREACHABLE_TEXT: &[&str] = &[
    "unable to connect to api",
    "connection refused",
    "econnrefused",
    "connection reset",
    "no route to host",
    "network is unreachable",
    "temporary failure in name resolution",
    "name or service not known",
    "nodename nor servname",
    "getaddrinfo",
    "fetch failed",
];

/// Cause tokens of the legacy taxonomy: short, stable, ours.
const LEGACY_API_ERROR: &str = "api_error";
const LEGACY_OVERLOADED: &str = "overloaded_error";
const LEGACY_RATE_LIMIT: &str = "rate_limit_error";
const LEGACY_HTTP_5XX: &str = "http_5xx";
const LEGACY_TIMEOUT: &str = "timeout";
const LEGACY_UNREACHABLE: &str = "endpoint_unreachable";
const LEGACY_UNKNOWN: &str = "provider_error";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::*;

    fn at(base: Instant, secs: u64) -> Instant {
        base + Duration::from_secs(secs)
    }

    /// The legacy loop's health failures: 5xx, overloaded, rate limit and timeout
    /// all count against the instance (they are `classify_api_error`'s retryable
    /// set), each with its own short token.
    #[test]
    fn a_legacy_retryable_failure_counts_against_the_instance() {
        let cases: &[(&str, &str)] = &[
            (
                r#"API Error: 500 {"type":"error","error":{"type":"api_error","message":"Internal server error"}}"#,
                "api_error",
            ),
            (
                r#"API Error: 529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
                "overloaded_error",
            ),
            (
                r#"API Error: 429 {"type":"error","error":{"type":"rate_limit_error","message":"Rate limited"}}"#,
                "rate_limit_error",
            ),
            ("API Error: 502 Bad Gateway", "http_5xx"),
            ("API Error: 503 Service Unavailable", "http_5xx"),
            ("Request timed out", "timeout"),
            // The CLI reports a stalled request this way; the case-sensitive
            // "timed out" of `classify_api_error` alone would miss it.
            ("Timeout waiting for response after 30 seconds", "timeout"),
        ];
        for (text, token) in cases {
            assert_eq!(
                outcome_of_legacy_failure(text),
                Some((Outcome::Failure, *token)),
                "text: {text}"
            );
        }
    }

    /// An endpoint that cannot be reached is a health failure however the client
    /// spells it (the CLI's own line, or a Bun/Node `fetch` failure).
    #[test]
    fn an_unreachable_endpoint_counts_against_the_instance() {
        let cases = [
            "API Error: Unable to connect to API (ConnectionRefused)",
            "Error: connect ECONNREFUSED 127.0.0.1:443",
            "TypeError: fetch failed",
            "getaddrinfo ENOTFOUND api.anthropic.com",
            "connect: network is unreachable",
        ];
        for text in cases {
            assert_eq!(
                outcome_of_legacy_failure(text),
                Some((Outcome::Failure, "endpoint_unreachable")),
                "text: {text}"
            );
        }
    }

    /// What the instance *answered* is not its health: a client error and a 402
    /// must not feed the detector (the router owns the 402).
    #[test]
    fn an_answered_client_error_is_not_a_health_failure() {
        let cases = [
            r#"API Error: 400 {"type":"error","error":{"type":"invalid_request_error","message":"Bad request"}}"#,
            r#"API Error: 401 {"type":"error","error":{"type":"authentication_error","message":"Unauthorized"}}"#,
            r#"API Error: 402 {"type":"error","error":{"type":"invalid_request_error","message":"Insufficient credits"}}"#,
            // A locally dead CLI is the resume path's business, not an outage.
            "Failed to connect to Claude CLI: No such file or directory",
            "stream ended before a Result",
        ];
        for text in cases {
            assert_eq!(outcome_of_legacy_failure(text), None, "text: {text}");
        }
    }

    /// The cause is our token, never the provider's raw text.
    #[test]
    fn a_legacy_cause_is_our_token_not_the_raw_text() {
        let (_, cause) = outcome_of_legacy_failure("API Error: 502 Bad Gateway").unwrap();
        assert_eq!(cause, "http_5xx");
        assert!(!cause.contains("API Error"), "leaked the raw text: {cause}");
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

    /// A hold with a threshold/window easy to reason about in a test.
    fn hold_of(threshold: u32, max_hold_secs: u64, probe_ms: u64) -> TurnHold {
        TurnHold::with_policies(
            "deepseek",
            OutagePolicy::new(threshold, Duration::from_secs(30)),
            HoldPolicy::new(
                Duration::from_secs(max_hold_secs),
                Duration::from_millis(probe_ms),
            ),
        )
    }

    #[test]
    fn a_turn_past_its_retry_budget_is_held_for_a_down_instance() {
        let base = Instant::now();
        let mut hold = hold_of(1, 60, 10);
        // One failure declares the instance down: the hold begins with the outage.
        assert!(hold
            .record(Outcome::Failure, base, Some("conn refused"))
            .is_some());
        assert!(hold.is_down());
        assert!(hold.is_held());
        assert_eq!(hold.hold_on(base), Some(Duration::from_millis(10)));
        // A further failure of the same outage is not a transition, and the hold
        // still stands: the turn goes on being probed.
        assert!(hold
            .record(Outcome::Failure, at(base, 1), Some("conn refused"))
            .is_none());
        assert_eq!(hold.hold_on(at(base, 1)), Some(Duration::from_millis(10)));
    }

    #[test]
    fn the_hold_is_bounded_and_gives_up_when_its_budget_is_spent() {
        let base = Instant::now();
        let mut hold = hold_of(1, 5, 10);
        hold.record(Outcome::Failure, base, None);
        assert_eq!(hold.hold_on(at(base, 4)), Some(Duration::from_millis(10)));
        // Past `max_hold`: the failure stands, and the hold is over.
        assert_eq!(hold.hold_on(at(base, 5)), None);
        assert!(!hold.is_held());
        // The instance is still down, but a hold that gave up is not restarted on
        // its own: the failure it produced stands.
        assert_eq!(hold.hold_on(at(base, 6)), None);
    }

    #[test]
    fn the_recovery_ends_the_hold_and_carries_the_outage() {
        let base = Instant::now();
        let mut hold = hold_of(1, 60, 10);
        hold.record(Outcome::Failure, base, Some("conn refused"));
        let notice = hold
            .record(Outcome::Success, at(base, 12), None)
            .expect("the instance answered again");
        assert_eq!(notice.phase, OutagePhase::Up);
        assert_eq!(notice.duration, Some(Duration::from_secs(12)));
        assert_eq!(notice.cause.as_deref(), Some("conn refused"));
        assert!(!hold.is_down());
        assert!(!hold.is_held());
        assert_eq!(hold.hold_on(at(base, 12)), None);
    }

    #[test]
    fn an_up_instance_is_never_held() {
        let base = Instant::now();
        let mut hold = hold_of(3, 60, 10);
        assert!(hold.record(Outcome::Failure, base, None).is_none());
        assert!(!hold.is_held(), "two failures short of the threshold");
        assert_eq!(hold.hold_on(base), None);
    }

    #[test]
    fn a_blip_does_not_start_a_hold() {
        let base = Instant::now();
        let mut hold = hold_of(3, 60, 10);
        hold.record(Outcome::Failure, base, None);
        hold.record(Outcome::Failure, at(base, 1), None);
        hold.record(Outcome::Success, at(base, 2), None);
        assert!(!hold.is_held());
        hold.record(Outcome::Failure, at(base, 3), None);
        assert!(!hold.is_held(), "the count restarted: still short of three");
    }

    #[test]
    fn the_hold_reports_how_long_it_has_lasted() {
        let base = Instant::now();
        let mut hold = hold_of(1, 60, 10);
        assert_eq!(hold.held_for(base), None);
        hold.record(Outcome::Failure, base, None);
        assert_eq!(hold.held_for(at(base, 7)), Some(Duration::from_secs(7)));
    }
}
