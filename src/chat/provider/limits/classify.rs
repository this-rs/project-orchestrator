//! RL3 — the pure reading of a provider's refusal.
//!
//! A refusal is never taken at face value. The same HTTP 429 means "your quota is
//! spent" for one endpoint and "I am having a bad minute" for another, and the two
//! demand opposite reactions: the first must stop sending until `resets_at`, the
//! second must retry with a backoff and may switch provider. Telling them apart is
//! this module's whole job, and it is a PURE function: status, headers, error code,
//! the provider's [`super::profiles`] row, and OUR OWN counting, in; one
//! [`Classification`] out. No clock of its own (the instant is passed in), no I/O,
//! no logging, nothing to mock.
//!
//! # The rules, in order, and why that order
//!
//! 1. **The provider states its quota is spent** — a zero-remaining header or a
//!    quota/billing code — is an exhaustion, whatever our own count says. It is the
//!    provider's own statement about its own resource; our ledger cannot argue with
//!    it.
//! 2. **The provider says it is overloaded** (`overloaded`, `engine_overloaded`…)
//!    is an overload, even while our own ceiling is reached: the provider's
//!    statement is more specific than our inference.
//! 3. **Our own ceiling explains the refusal** — a limiting status while our count
//!    is at the configured limit — is our exhaustion: nothing may go out until the
//!    limit resets. This is the only case where OUR number, not the provider's word,
//!    decides.
//! 4. **A "slow down" while our count is near (not at) its limit** is our limit too:
//!    the header confirms what our counting was already showing.
//! 5. **A limiting status with no quota evidence and our count under the limit** is
//!    an overload. This is the contradiction rule: being refused repeatedly while
//!    our own counting is far from its ceiling cannot mean the credential is spent —
//!    it means the provider is throttling or having a bad minute, so the key must
//!    NOT be marked exhausted (that would stop sending long after the provider
//!    recovered).
//! 6. **Anything else is `Unknown`, and an `Unknown` is NEVER a quota.** A caller
//!    treats it as an overload. A case we cannot read must not cost the user a
//!    credential: blocking on a guess is worse than one retry too many.
//!
//! # The one place the written brief and the fixtures disagree, and what was done
//!
//! RL3's brief lists `Retry-After` among the *quota* signals, while RL1's golden
//! event fixture names `retry after` as the OVERLOAD reason, and rule 5 above
//! demands the opposite of a blanket reading. Implemented reading: `Retry-After`
//! is a WEAK limit signal — it confirms our own limit when our count is near or at
//! its ceiling (rule 4), and it is the backoff hint of an overload otherwise
//! (rules 5-6). A bare `Retry-After` with no configured limit is therefore an
//! overload, never an exhaustion. **This is a choice, not a measurement**: RL9's
//! real captures are what will settle it, and both readings are named here so the
//! contradiction is not rediscovered as a bug.
//!
//! # What this module does NOT decide
//!
//! It does not decide *whether* to send (that is the ledger's reservation), and it
//! never reads a message: the prose of an error body is written by the provider, so
//! it is neither trusted nor echoed. Only the machine-readable `code`/`type` is
//! matched, against a table, and only a code the table knows is ever repeated in
//! the reason.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::time::Duration;

use chrono::{DateTime, SecondsFormat, Utc};

use super::profiles::{self, Evidence, ProviderProfile};
use crate::chat::provider::settings::{RateLimitState, RateLimitStatus};

/// Above this fraction of the configured ceiling our own counting is "near" the
/// limit (see [`OurStanding`]). Named here so the one heuristic in the file is
/// visible and testable instead of buried in a comparison.
pub const NEAR_RATIO: f64 = 0.8;

/// A value at or above this is an absolute instant (epoch seconds); below it, it
/// is a number of seconds to wait. Both forms exist in the wild for the same
/// `x-ratelimit-reset` header, and 1e9 seconds is 31 years — no reset is that far
/// away, so the split is not ambiguous in practice.
const EPOCH_FLOOR: i64 = 1_000_000_000;

/// The header a provider uses to ask for a delay.
const RETRY_AFTER: &str = "retry-after";

/// Where OUR counting stands against the configured limit.
///
/// The classifier never queries the ledger: it receives this, so its rules stay
/// pure and its tests need no counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OurStanding {
    /// No limit is configured for this instance: our counting cannot explain a
    /// refusal.
    NoLimit,
    /// A limit is configured but nothing was measured yet (fresh process, a count
    /// that was never persisted for a short window).
    Unknown,
    /// Measured, below the limit.
    Under,
    /// Measured, at or above [`NEAR_RATIO`] of the limit but not at it.
    Near,
    /// Measured, at or above the limit: nothing may go out.
    AtOrAbove,
}

impl OurStanding {
    /// The standing of a usage against a ceiling. Both `None` (not measured, no
    /// limit) are told apart: an unmeasured limit is not the absence of one.
    pub fn of(used: Option<f64>, max: Option<f64>) -> Self {
        let Some(max) = max else {
            return Self::NoLimit;
        };
        if max <= 0.0 {
            return Self::NoLimit;
        }
        // A limit is configured but nothing was measured: that is not the absence
        // of a limit, and it must not become a quota by accident.
        let Some(used) = used else {
            return Self::Unknown;
        };
        if !used.is_finite() {
            return Self::Unknown;
        }
        let ratio = used / max;
        if ratio >= 1.0 {
            Self::AtOrAbove
        } else if ratio >= NEAR_RATIO {
            Self::Near
        } else {
            Self::Under
        }
    }

    /// Whether our own counting can explain a refusal at all.
    fn blames_our_limit(self) -> bool {
        matches!(self, Self::Near | Self::AtOrAbove)
    }

    /// The words a human reason uses.
    fn words(self) -> &'static str {
        match self {
            Self::Under => "under its limit",
            Self::Near => "near, but not at, its limit",
            _ => "not measurable",
        }
    }
}

/// The machine-readable part of an error body: `code` and `type`, never the prose.
#[derive(Debug, Clone, Copy, Default)]
pub struct ErrorBody<'a> {
    /// `error.code` (OpenAI-compatible), `code`, or an equivalent field.
    pub code: Option<&'a str>,
    /// `error.type` / `type`.
    pub kind: Option<&'a str>,
}

/// Everything the classifier may look at. Built with [`Signals::new`] and the
/// chaining setters; nothing here is produced by a model, and no field is logged
/// verbatim.
#[derive(Debug, Clone, Copy)]
pub struct Signals<'a> {
    /// The endpoint family (see [`profiles::preset_for_base_url`]).
    pub preset: &'a str,
    /// The HTTP status of the refusal.
    pub http_status: u16,
    /// The response headers, as `(name, value)` pairs. Only the names a profile
    /// knows are ever read, and only their parsed NUMBERS reach the reason.
    pub headers: &'a [(&'a str, &'a str)],
    /// The machine-readable fields of the body.
    pub body: ErrorBody<'a>,
    /// Where our own counting stands.
    pub ours: OurStanding,
    /// How many times in a row the provider refused. Only ever used in the human
    /// reason: the verdict does not depend on it.
    pub consecutive: u32,
}

impl<'a> Signals<'a> {
    /// The minimum: a status, a preset, no headers, no body, no measurement, and
    /// this refusal counted as the first.
    pub fn new(preset: &'a str, http_status: u16) -> Self {
        Self {
            preset,
            http_status,
            headers: &[],
            body: ErrorBody::default(),
            ours: OurStanding::NoLimit,
            consecutive: 1,
        }
    }

    /// Attaches the response headers.
    pub fn headers(mut self, headers: &'a [(&'a str, &'a str)]) -> Self {
        self.headers = headers;
        self
    }

    /// Attaches the machine-readable fields of the body.
    pub fn body(mut self, body: ErrorBody<'a>) -> Self {
        self.body = body;
        self
    }

    /// States where our own counting stands.
    pub fn ours(mut self, ours: OurStanding) -> Self {
        self.ours = ours;
        self
    }

    /// States how many refusals in a row this is.
    pub fn consecutive(mut self, consecutive: u32) -> Self {
        self.consecutive = consecutive;
        self
    }
}

/// What the refusal is.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// The quota (the provider's, or our configured limit) is spent: nothing is
    /// sent until `reset_at`, when the provider said when.
    QuotaExhausted {
        /// When the quota resets, when the response carried it.
        reset_at: Option<DateTime<Utc>>,
    },
    /// The provider is overloaded or throttling us while our own counting is not
    /// the cause: retry after `retry_after`, and do NOT mark the key as spent.
    Overloaded {
        /// The delay the provider asked for, or its profile's default.
        retry_after: Option<Duration>,
    },
    /// Nothing readable. Callers treat it as an overload (never as an exhaustion).
    Unknown,
}

/// What carried the verdict: the provider's own word, or our own counting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// The provider stated its quota is spent (zero remaining, quota code).
    ProviderQuota,
    /// Our configured limit is what the refusal is explained by.
    OurQuota,
    /// An overload: the provider's word, or a refusal our counting cannot explain.
    Overload,
    /// Nothing usable.
    NoSignal,
}

/// The reading of one refusal.
#[derive(Debug, Clone, PartialEq)]
pub struct Classification {
    /// What the refusal is.
    pub verdict: Verdict,
    /// What carried it.
    pub signal: Signal,
    /// How solid the provider's profile is: an `Unproven` profile is a guess, and
    /// the caller may want to say so in a log line.
    pub evidence: Evidence,
    /// A short human sentence. Never contains a header value that was not parsed
    /// into a number, and never the body prose.
    pub reason: String,
}

impl Classification {
    /// The status RL1's report and event carry for this reading.
    pub fn status(&self) -> RateLimitStatus {
        match self.verdict {
            Verdict::QuotaExhausted { .. } => RateLimitStatus::Limited,
            Verdict::Overloaded { .. } => RateLimitStatus::Overloaded,
            Verdict::Unknown => RateLimitStatus::Unknown,
        }
    }

    /// The `reason` token of `provider_rate_limit` (RL1's event vocabulary):
    /// `quota`, `retry after` or `unknown`.
    pub fn event_reason(&self) -> &'static str {
        match self.verdict {
            Verdict::QuotaExhausted { .. } => "quota",
            Verdict::Overloaded { .. } => "retry after",
            Verdict::Unknown => "unknown",
        }
    }

    /// Whether this reading marks the credential as spent. The ONE question a
    /// caller must not get wrong.
    pub fn is_quota(&self) -> bool {
        matches!(self.verdict, Verdict::QuotaExhausted { .. })
    }

    /// When the quota resets, if this is an exhaustion that knows it.
    pub fn reset_at(&self) -> Option<DateTime<Utc>> {
        match self.verdict {
            Verdict::QuotaExhausted { reset_at } => reset_at,
            _ => None,
        }
    }

    /// When the next attempt may be made: the reset instant for an exhaustion, the
    /// backoff from `now` for an overload, nothing for an unknown.
    pub fn next_attempt_at(&self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        match self.verdict {
            Verdict::QuotaExhausted { reset_at } => reset_at,
            Verdict::Overloaded { retry_after } => {
                retry_after.map(|d| now + chrono::Duration::from_std(d).unwrap_or_default())
            }
            Verdict::Unknown => None,
        }
    }
}

/// Reads one refusal. `now` is the only time input: nothing here calls a clock.
pub fn classify(signals: &Signals<'_>, now: DateTime<Utc>) -> Classification {
    let profile = profiles::profile(signals.preset);
    let retry_after =
        value_of(signals.headers, RETRY_AFTER).and_then(|raw| parse_retry_after(raw, now));
    let remaining = remaining_min(signals.headers, profile);
    let reset_at = reset_instant(signals.headers, profile, now);
    let code = code_of(&signals.body);
    let limiting = is_limiting(signals.http_status);

    // 1. The provider states its OWN quota is spent. Its statement is about its
    //    own resource: our counting cannot argue with it, whatever it says.
    if remaining == Some(0.0) {
        return quota(
            Signal::ProviderQuota,
            "the provider reports no quota left".to_string(),
            reset_at,
            profile,
        );
    }
    if let Some(code) = code.as_deref().filter(|c| profile.quota_codes.contains(c)) {
        return quota(
            Signal::ProviderQuota,
            format!("the provider answered the quota code '{code}'"),
            reset_at,
            profile,
        );
    }
    // 2. The provider states it is overloaded. More specific than our inference,
    //    so it wins even while our own ceiling is reached.
    if let Some(code) = code
        .as_deref()
        .filter(|c| profile.overload_codes.contains(c))
    {
        return overload(
            format!("the provider reports an overload ('{code}')"),
            retry_after,
            profile,
        );
    }
    // 3. Our own ceiling explains a limiting refusal — at the limit always, near it
    //    only when the provider also asks us to slow down (rules 3 and 4 of the
    //    module documentation).
    if limiting && signals.ours.blames_our_limit() {
        let at_the_limit = signals.ours == OurStanding::AtOrAbove;
        if at_the_limit || retry_after.is_some() {
            let reason = if at_the_limit {
                "our own counting is at the configured limit".to_string()
            } else {
                format!(
                    "the provider asks us to slow down while our own counting is {}",
                    signals.ours.words()
                )
            };
            return quota(
                Signal::OurQuota,
                reason,
                reset_at.or_else(|| retry_after.map(|delay| after(now, delay))),
                profile,
            );
        }
    }
    // 4. A limiting status our own counting cannot explain: an overload. A refusal
    //    under our ceiling can never mean the credential is spent.
    if limiting {
        return overload(contradiction_reason(signals), retry_after, profile);
    }
    // 5. A delay with no limiting status is still not an exhaustion.
    if retry_after.is_some() {
        return overload(
            format!(
                "the provider asked us to slow down (HTTP {})",
                signals.http_status
            ),
            retry_after,
            profile,
        );
    }
    // 6. Nothing readable. Never an exhaustion: blocking on a guess costs more
    //    than one attempt too many.
    unknown(
        format!("no readable limit signal (HTTP {})", signals.http_status),
        profile,
    )
}

/// The instant `delay` after `now`.
fn after(now: DateTime<Utc>, delay: Duration) -> DateTime<Utc> {
    now + chrono::Duration::from_std(delay).unwrap_or_default()
}

/// The reason of a refusal our own counting cannot explain: the contradiction
/// rule, spelled out so a log reader knows the key was NOT marked spent.
fn contradiction_reason(signals: &Signals<'_>) -> String {
    format!(
        "the provider refused (HTTP {}) {} time(s) in a row while our own counting is {}: \
         an overload, not an exhausted credential",
        signals.http_status,
        signals.consecutive,
        signals.ours.words()
    )
}

/// The compatible reading of a `ProviderError` kind (nexus) when the wire signals
/// are no longer available.
///
/// nexus maps every HTTP 429 to `rate_limited` and 503/529 to `overloaded`
/// (`model/openai.rs::classify_http`), so a `rate_limited` alone does NOT say
/// whether a quota is spent — it says "the endpoint refused", which is exactly the
/// ambiguous case rule 6 protects: without a stronger signal and with our counting
/// under its limit, it is read as an overload, never as an exhaustion.
pub fn from_provider_error(
    kind: &str,
    retry_after_ms: Option<u64>,
    ours: OurStanding,
    preset: &str,
    now: DateTime<Utc>,
) -> Classification {
    let profile = profiles::profile(preset);
    let retry_after = retry_after_ms.map(Duration::from_millis);
    match kind {
        "rate_limited" => {
            if ours == OurStanding::AtOrAbove {
                quota(
                    Signal::OurQuota,
                    "our own counting is at the configured limit".to_string(),
                    retry_after.map(|delay| after(now, delay)),
                    profile,
                )
            } else {
                overload(
                    "the provider refused (rate_limited) without any quota proof: not an \
                     exhausted credential"
                        .to_string(),
                    retry_after,
                    profile,
                )
            }
        }
        "overloaded" => overload(
            "the provider is reported as overloaded".to_string(),
            retry_after,
            profile,
        ),
        _ => unknown(format!("'{kind}' is not a rate-limit signal"), profile),
    }
}

/// Fills the live state of a limit from a reading (RL1's shape, RL2/RL3's numbers).
pub fn apply_verdict(
    state: &mut RateLimitState,
    classification: &Classification,
    now: DateTime<Utc>,
) {
    state.status = classification.status();
    if classification.verdict != Verdict::Unknown {
        state.since = Some(rfc3339(now));
    }
    if let Some(reset) = classification.reset_at() {
        state.resets_at = Some(rfc3339(reset));
    }
    state.next_retry_at = classification.next_attempt_at(now).map(rfc3339);
}

/// `2026-10-08T12:34:00Z`: what RL1's report carries.
pub(crate) fn rfc3339(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn quota(
    signal: Signal,
    reason: String,
    reset_at: Option<DateTime<Utc>>,
    profile: &ProviderProfile,
) -> Classification {
    Classification {
        verdict: Verdict::QuotaExhausted { reset_at },
        signal,
        evidence: profile.evidence,
        reason,
    }
}

fn overload(
    reason: String,
    retry_after: Option<Duration>,
    profile: &ProviderProfile,
) -> Classification {
    let retry_after = retry_after.or(Some(profile.default_backoff));
    Classification {
        verdict: Verdict::Overloaded { retry_after },
        signal: Signal::Overload,
        evidence: profile.evidence,
        reason,
    }
}

fn unknown(reason: String, profile: &ProviderProfile) -> Classification {
    Classification {
        verdict: Verdict::Unknown,
        signal: Signal::NoSignal,
        evidence: profile.evidence,
        reason,
    }
}

/// Whether a status can be a refusal of the "too much / not now" family.
fn is_limiting(status: u16) -> bool {
    status == 429 || status >= 500
}

/// The value of a header, case-insensitively.
fn value_of<'a>(headers: &[(&'a str, &'a str)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.trim().eq_ignore_ascii_case(name))
        .map(|(_, value)| *value)
}

/// Whether a header name is a prefix or a prefix plus a `-<variant>` suffix.
fn matches_prefix(name: &str, prefix: &str) -> bool {
    let name = name.trim().to_ascii_lowercase();
    name == prefix
        || name
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with('-'))
}

/// The smallest value among the remaining-quota headers a profile knows, so that
/// any zero anywhere is seen.
fn remaining_min(headers: &[(&str, &str)], profile: &ProviderProfile) -> Option<f64> {
    headers
        .iter()
        .filter(|(name, _)| {
            profile
                .remaining_prefixes
                .iter()
                .any(|prefix| matches_prefix(name, prefix))
        })
        .filter_map(|(_, value)| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .fold(None::<f64>, |acc, value| {
            Some(acc.map_or(value, |a| a.min(value)))
        })
}

/// The first reset header a profile knows, parsed.
fn reset_instant(
    headers: &[(&str, &str)],
    profile: &ProviderProfile,
    now: DateTime<Utc>,
) -> Option<DateTime<Utc>> {
    headers
        .iter()
        .find(|(name, _)| {
            profile
                .reset_prefixes
                .iter()
                .any(|prefix| matches_prefix(name, prefix))
        })
        .and_then(|(_, value)| parse_reset(value, now))
}

/// The `code` (or `type`) of an error body, lowercased.
fn code_of(body: &ErrorBody<'_>) -> Option<String> {
    body.code
        .or(body.kind)
        .map(|raw| raw.trim().to_ascii_lowercase())
        .filter(|code| !code.is_empty())
}

/// Seconds (or a date) of a `Retry-After` header.
fn parse_retry_after(raw: &str, now: DateTime<Utc>) -> Option<Duration> {
    let value = raw.trim();
    if let Ok(seconds) = value.parse::<f64>() {
        return (seconds.is_finite() && seconds >= 0.0)
            .then(|| Duration::from_millis((seconds * 1000.0).round() as u64));
    }
    let when = DateTime::parse_from_rfc2822(value).ok()?;
    let delta = when.with_timezone(&Utc) - now;
    Some(Duration::from_millis(delta.num_milliseconds().max(0) as u64))
}

/// A `x-ratelimit-reset` value: epoch seconds, seconds to wait, or a Go duration.
fn parse_reset(raw: &str, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let value = raw.trim();
    if let Ok(number) = value.parse::<i64>() {
        return if number >= EPOCH_FLOOR {
            DateTime::from_timestamp(number, 0)
        } else {
            Some(now + chrono::Duration::seconds(number.max(0)))
        };
    }
    let delay = parse_go_duration(value)?;
    Some(now + chrono::Duration::from_std(delay).ok()?)
}

/// A Go-style duration (`500ms`, `1s`, `6m0s`, `1h2m3s`), as OpenAI-compatible
/// endpoints write in `x-ratelimit-reset-requests`.
fn parse_go_duration(raw: &str) -> Option<Duration> {
    let text = raw.trim();
    if text.is_empty() {
        return None;
    }
    let bytes = text.as_bytes();
    let mut index = 0usize;
    let mut total_ms = 0.0f64;
    while index < bytes.len() {
        let number_start = index;
        while index < bytes.len() && (bytes[index].is_ascii_digit() || bytes[index] == b'.') {
            index += 1;
        }
        if index == number_start {
            return None;
        }
        let amount: f64 = text[number_start..index].parse().ok()?;
        let unit_start = index;
        while index < bytes.len() && !bytes[index].is_ascii_digit() && bytes[index] != b'.' {
            index += 1;
        }
        if index == unit_start {
            return None;
        }
        total_ms += match &text[unit_start..index] {
            "ns" => amount / 1_000_000.0,
            "us" | "µs" => amount / 1_000.0,
            "ms" => amount,
            "s" => amount * 1_000.0,
            "m" => amount * 60_000.0,
            "h" => amount * 3_600_000.0,
            _ => return None,
        };
    }
    (total_ms.is_finite() && total_ms >= 0.0)
        .then(|| Duration::from_millis(total_ms.round() as u64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::settings::{
        InstanceScope, RateLimit, RateLimitKind, RateLimitScope, RateLimitUnit, RateLimitWindow,
    };
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    fn at(text: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(text)
            .unwrap()
            .with_timezone(&Utc)
    }

    fn noon() -> DateTime<Utc> {
        at("2026-10-08T12:00:00Z")
    }

    fn openai() -> &'static str {
        profiles::OPENAI_COMPATIBLE
    }

    fn empty_state() -> RateLimitState {
        RateLimitState {
            kind: RateLimitKind::Requests,
            window: RateLimitWindow::Unit(RateLimitUnit::Minute),
            calendar: None,
            max: 40.0,
            scope: RateLimitScope::Instance(InstanceScope::Instance),
            credential: None,
            used: None,
            remaining: None,
            resets_at: None,
            status: RateLimitStatus::Unknown,
            since: None,
            next_retry_at: None,
        }
    }

    fn unused_limit() -> RateLimit {
        RateLimit {
            kind: RateLimitKind::Requests,
            window: RateLimitWindow::Unit(RateLimitUnit::Minute),
            calendar: None,
            max: 40.0,
            scope: RateLimitScope::Instance(InstanceScope::Instance),
        }
    }

    // ---- rule 1: the provider's own statement beats our counting -------------

    #[test]
    fn a_zero_remaining_header_is_the_providers_own_quota() {
        let headers = [("X-RateLimit-Remaining", "0")];
        let signals = Signals::new(openai(), 429)
            .headers(&headers)
            .ours(OurStanding::Under);
        let read = classify(&signals, noon());
        assert!(read.is_quota(), "{}", read.reason);
        assert_eq!(read.signal, Signal::ProviderQuota);
        assert_eq!(read.status(), RateLimitStatus::Limited);
    }

    #[test]
    fn any_zero_among_the_prefixed_remaining_headers_counts() {
        // `x-ratelimit-remaining-requests` is the same statement as the bare name,
        // and the SMALLEST of them is what the endpoint is bound by.
        let headers = [
            ("x-ratelimit-remaining-tokens", "9000"),
            ("x-ratelimit-remaining-requests", "0"),
        ];
        let read = classify(&Signals::new(openai(), 429).headers(&headers), noon());
        assert!(read.is_quota());
        // And a non-zero remaining header alone is NOT a quota (it is rule 5's case).
        let headers = [("x-ratelimit-remaining", "12")];
        let read = classify(&Signals::new(openai(), 429).headers(&headers), noon());
        assert!(!read.is_quota(), "{}", read.reason);
    }

    #[test]
    fn a_quota_code_is_the_providers_own_quota() {
        let body = ErrorBody {
            code: Some("insufficient_quota"),
            kind: Some("error"),
        };
        let read = classify(&Signals::new(openai(), 429).body(body), noon());
        assert!(read.is_quota(), "{}", read.reason);
        assert_eq!(read.signal, Signal::ProviderQuota);
        assert!(
            read.reason.contains("insufficient_quota"),
            "{}",
            read.reason
        );
    }

    #[test]
    fn a_billing_code_is_a_quota_even_on_a_status_that_is_not_429() {
        let body = ErrorBody {
            code: Some("insufficient_balance"),
            kind: None,
        };
        let read = classify(&Signals::new("deepseek", 402).body(body), noon());
        assert!(read.is_quota(), "{}", read.reason);
    }

    #[test]
    fn the_providers_own_quota_verb_wins_whatever_our_own_counting_says() {
        // Our ledger says there is room; the provider says the key is spent. The
        // provider is talking about its own resource: it wins.
        let headers = [("x-ratelimit-remaining", "0")];
        for ours in [
            OurStanding::NoLimit,
            OurStanding::Unknown,
            OurStanding::Under,
            OurStanding::Near,
            OurStanding::AtOrAbove,
        ] {
            let read = classify(
                &Signals::new(openai(), 429).headers(&headers).ours(ours),
                noon(),
            );
            assert!(read.is_quota(), "{ours:?}: {}", read.reason);
        }
    }

    // ---- rule 2: the provider's own overload statement -----------------------

    #[test]
    fn an_overload_code_wins_over_our_own_counting() {
        let body = ErrorBody {
            code: Some("engine_overloaded"),
            kind: None,
        };
        // Even while our own ceiling is reached, an explicit overload statement is
        // read for what it says: the provider, not our limit, is the cause.
        let read = classify(
            &Signals::new("deepseek", 429)
                .body(body)
                .ours(OurStanding::AtOrAbove),
            noon(),
        );
        assert!(!read.is_quota(), "{}", read.reason);
        assert_eq!(read.status(), RateLimitStatus::Overloaded);
    }

    // ---- rule 3: our own ceiling --------------------------------------------

    #[test]
    fn our_own_limit_reached_is_our_exhaustion() {
        let read = classify(
            &Signals::new(openai(), 429).ours(OurStanding::AtOrAbove),
            noon(),
        );
        assert!(read.is_quota(), "{}", read.reason);
        assert_eq!(read.signal, Signal::OurQuota);
        assert!(read.reason.contains("configured limit"), "{}", read.reason);
    }

    #[test]
    fn a_five_hundred_while_at_our_limit_is_read_the_same_way() {
        let read = classify(
            &Signals::new(openai(), 502).ours(OurStanding::AtOrAbove),
            noon(),
        );
        assert!(read.is_quota(), "{}", read.reason);
    }

    // ---- rules 4-5: the contradiction rule and the weak signal ---------------

    #[test]
    fn a_refusal_while_near_but_not_at_our_limit_without_a_delay_is_an_overload() {
        // Near the limit, but the provider says nothing more: a weak signal alone
        // does not make an exhaustion (rule 4 needs the delay).
        let read = classify(&Signals::new(openai(), 429).ours(OurStanding::Near), noon());
        assert!(!read.is_quota(), "{}", read.reason);
        assert!(
            read.reason.contains("near, but not at, its limit"),
            "{}",
            read.reason
        );
    }

    #[test]
    fn a_refusal_while_under_our_limit_is_an_overload_not_an_exhaustion() {
        let read = classify(
            &Signals::new(openai(), 429)
                .ours(OurStanding::Under)
                .consecutive(3),
            noon(),
        );
        assert!(!read.is_quota(), "{}", read.reason);
        assert_eq!(read.signal, Signal::Overload);
        assert_eq!(read.status(), RateLimitStatus::Overloaded);
        assert!(read.reason.contains('3'), "{}", read.reason);
        assert!(read.reason.contains("under its limit"), "{}", read.reason);
    }

    #[test]
    fn a_refusal_with_no_measured_counting_is_an_overload_too() {
        for ours in [OurStanding::NoLimit, OurStanding::Unknown] {
            let read = classify(&Signals::new(openai(), 429).ours(ours), noon());
            assert!(!read.is_quota(), "{ours:?}: {}", read.reason);
            assert!(read.reason.contains("not measurable"), "{}", read.reason);
        }
    }

    #[test]
    fn retry_after_while_near_our_limit_is_our_quota() {
        // The weak limit signal, confirmed by our own counting: the provider is
        // telling us to slow down and we are nearly at our own ceiling.
        let headers = [("Retry-After", "30")];
        let read = classify(
            &Signals::new(openai(), 429)
                .headers(&headers)
                .ours(OurStanding::Near),
            noon(),
        );
        assert!(read.is_quota(), "{}", read.reason);
        assert_eq!(read.signal, Signal::OurQuota);
        assert_eq!(
            read.reset_at(),
            Some(noon() + chrono::Duration::seconds(30))
        );
    }

    #[test]
    fn retry_after_while_under_our_limit_is_an_overload_and_not_our_quota() {
        // THE point of the contradiction rule: the same header, the opposite
        // verdict, because our own counting does not explain the refusal.
        let headers = [("Retry-After", "30")];
        let read = classify(
            &Signals::new(openai(), 429)
                .headers(&headers)
                .ours(OurStanding::Under),
            noon(),
        );
        assert!(!read.is_quota(), "{}", read.reason);
        assert_eq!(
            read.verdict,
            Verdict::Overloaded {
                retry_after: Some(Duration::from_secs(30))
            }
        );
    }

    #[test]
    fn retry_after_seconds_become_the_backoff_hint() {
        let headers = [("retry-after", "2.5")];
        let read = classify(&Signals::new(openai(), 429).headers(&headers), noon());
        assert_eq!(
            read.verdict,
            Verdict::Overloaded {
                retry_after: Some(Duration::from_millis(2500))
            }
        );
    }

    #[test]
    fn retry_after_as_a_date_is_read_against_the_clock() {
        let headers = [("Retry-After", "Thu, 08 Oct 2026 12:00:30 GMT")];
        let read = classify(&Signals::new(openai(), 429).headers(&headers), noon());
        assert_eq!(
            read.verdict,
            Verdict::Overloaded {
                retry_after: Some(Duration::from_secs(30))
            }
        );
        // A date already in the past asks for no wait at all, never for a negative one.
        let headers = [("Retry-After", "Thu, 08 Oct 2026 11:59:00 GMT")];
        let read = classify(&Signals::new(openai(), 429).headers(&headers), noon());
        assert_eq!(
            read.verdict,
            Verdict::Overloaded {
                retry_after: Some(Duration::ZERO)
            }
        );
    }

    #[test]
    fn an_unreadable_retry_after_is_ignored_rather_than_guessed() {
        let headers = [("Retry-After", "soon")];
        let read = classify(&Signals::new(openai(), 429).headers(&headers), noon());
        assert_eq!(
            read.verdict,
            Verdict::Overloaded {
                retry_after: Some(Duration::from_secs(5))
            },
            "the profile's default backoff, not a guess at the header"
        );
    }

    // ---- rule 6: the safe floor ---------------------------------------------

    #[test]
    fn a_503_with_no_signal_is_an_overload() {
        let read = classify(&Signals::new(openai(), 503), noon());
        assert_eq!(read.status(), RateLimitStatus::Overloaded);
        assert_eq!(read.signal, Signal::Overload);
        assert!(!read.is_quota());
    }

    #[test]
    fn an_unreadable_answer_is_unknown_and_never_a_quota() {
        let read = classify(&Signals::new(openai(), 418), noon());
        assert_eq!(read.verdict, Verdict::Unknown);
        assert_eq!(read.status(), RateLimitStatus::Unknown);
        assert_eq!(read.signal, Signal::NoSignal);
        assert!(!read.is_quota());
        assert_eq!(read.event_reason(), "unknown");
        assert_eq!(read.next_attempt_at(noon()), None);
    }

    #[test]
    fn a_plain_client_error_is_unknown() {
        let body = ErrorBody {
            code: Some("invalid_request"),
            kind: None,
        };
        let read = classify(&Signals::new(openai(), 400).body(body), noon());
        assert_eq!(read.verdict, Verdict::Unknown);
    }

    #[test]
    fn a_code_the_table_does_not_know_is_neither_a_quota_nor_echoed() {
        let body = ErrorBody {
            code: Some("we_are_sorry_comrade"),
            kind: None,
        };
        let read = classify(&Signals::new(openai(), 429).body(body), noon());
        assert!(!read.is_quota(), "{}", read.reason);
        assert_eq!(read.status(), RateLimitStatus::Overloaded);
        assert!(
            !read.reason.contains("comrade"),
            "a code outside the table is never repeated: {}",
            read.reason
        );
    }

    #[test]
    fn a_retry_after_with_no_limiting_status_is_still_only_an_overload() {
        let headers = [("Retry-After", "10")];
        let read = classify(&Signals::new(openai(), 200).headers(&headers), noon());
        assert!(!read.is_quota(), "{}", read.reason);
    }

    // ---- the reason never leaks ---------------------------------------------

    #[test]
    fn the_reason_never_carries_a_header_value_that_was_not_parsed() {
        let headers = [
            ("Authorization", "Bearer sk-live-secret"),
            ("X-Api-Key", "sk-another-secret"),
            ("X-Request-Id", "req-sk-42"),
            ("Retry-After", "3"),
        ];
        let read = classify(&Signals::new(openai(), 429).headers(&headers), noon());
        assert!(!read.reason.contains("sk-"), "{}", read.reason);
        assert!(!read.reason.contains("Bearer"), "{}", read.reason);
        // And nothing else in the reading carries them either.
        let rendered = format!("{read:?}");
        assert!(!rendered.contains("sk-"), "{rendered}");
    }

    // ---- the helpers, on their own ------------------------------------------

    #[test]
    fn standing_is_computed_from_usage_and_ceiling() {
        assert_eq!(OurStanding::of(None, None), OurStanding::NoLimit);
        assert_eq!(OurStanding::of(Some(1.0), None), OurStanding::NoLimit);
        assert_eq!(OurStanding::of(Some(1.0), Some(0.0)), OurStanding::NoLimit);
        assert_eq!(OurStanding::of(None, Some(40.0)), OurStanding::Unknown);
        assert_eq!(
            OurStanding::of(Some(f64::NAN), Some(40.0)),
            OurStanding::Unknown
        );
        assert_eq!(OurStanding::of(Some(0.0), Some(40.0)), OurStanding::Under);
        assert_eq!(OurStanding::of(Some(31.9), Some(40.0)), OurStanding::Under);
        assert_eq!(OurStanding::of(Some(32.0), Some(40.0)), OurStanding::Near);
        assert_eq!(
            OurStanding::of(Some(40.0), Some(40.0)),
            OurStanding::AtOrAbove
        );
        assert_eq!(
            OurStanding::of(Some(41.0), Some(40.0)),
            OurStanding::AtOrAbove
        );
    }

    #[test]
    fn the_reset_header_gives_the_instant_a_quota_ends() {
        // Epoch seconds: an absolute instant.
        let headers = [
            ("x-ratelimit-remaining", "0"),
            ("x-ratelimit-reset", "1786000000"),
        ];
        let read = classify(&Signals::new(openai(), 429).headers(&headers), noon());
        assert_eq!(read.reset_at(), DateTime::from_timestamp(1786000000, 0));
        // A small number is a wait, not an instant.
        let headers = [("x-ratelimit-remaining", "0"), ("x-ratelimit-reset", "90")];
        let read = classify(&Signals::new(openai(), 429).headers(&headers), noon());
        assert_eq!(
            read.reset_at(),
            Some(noon() + chrono::Duration::seconds(90))
        );
        // A Go duration, the shape OpenAI-compatible endpoints write.
        let headers = [
            ("x-ratelimit-remaining", "0"),
            ("x-ratelimit-reset-requests", "6m0s"),
        ];
        let read = classify(&Signals::new(openai(), 429).headers(&headers), noon());
        assert_eq!(
            read.reset_at(),
            Some(noon() + chrono::Duration::seconds(360))
        );
        // An unreadable value leaves the instant unknown rather than invented.
        let headers = [
            ("x-ratelimit-remaining", "0"),
            ("x-ratelimit-reset", "whenever"),
        ];
        let read = classify(&Signals::new(openai(), 429).headers(&headers), noon());
        assert_eq!(read.reset_at(), None);
        assert!(read.is_quota());
    }

    #[test]
    fn durations_of_every_unit_a_go_client_writes_are_read() {
        let cases = [
            ("0s", 0u64),
            ("500ms", 500),
            ("1s", 1000),
            ("6m0s", 360_000),
            ("1h2m3s", 3_723_000),
            ("250us", 0),
            ("1000000ns", 1),
        ];
        for (text, expected) in cases {
            assert_eq!(
                parse_go_duration(text),
                Some(Duration::from_millis(expected)),
                "{text}"
            );
        }
        for broken in ["", "s", "12", "1x", "1s2", "  "] {
            assert_eq!(parse_go_duration(broken), None, "{broken:?}");
        }
    }

    // ---- the mapping onto RL1's shapes --------------------------------------

    #[test]
    fn the_verdict_maps_onto_the_status_of_the_report() {
        let quota = classify(
            &Signals::new(openai(), 429).body(ErrorBody {
                code: Some("insufficient_quota"),
                kind: None,
            }),
            noon(),
        );
        assert_eq!(quota.status(), RateLimitStatus::Limited);
        assert_eq!(quota.event_reason(), "quota");
        let overload = classify(&Signals::new(openai(), 503), noon());
        assert_eq!(overload.status(), RateLimitStatus::Overloaded);
        assert_eq!(overload.event_reason(), "retry after");
    }

    #[test]
    fn applying_a_verdict_fills_the_state_without_changing_a_key() {
        let mut state = empty_state();
        let headers = [("x-ratelimit-remaining", "0"), ("x-ratelimit-reset", "90")];
        let read = classify(&Signals::new(openai(), 429).headers(&headers), noon());
        apply_verdict(&mut state, &read, noon());
        assert_eq!(state.status, RateLimitStatus::Limited);
        assert_eq!(state.since.as_deref(), Some("2026-10-08T12:00:00Z"));
        assert_eq!(state.resets_at.as_deref(), Some("2026-10-08T12:01:30Z"));
        assert_eq!(state.next_retry_at.as_deref(), Some("2026-10-08T12:01:30Z"));
        // An overload sets its own next attempt from the backoff.
        let mut state = empty_state();
        let headers = [("Retry-After", "30")];
        let read = classify(&Signals::new(openai(), 429).headers(&headers), noon());
        apply_verdict(&mut state, &read, noon());
        assert_eq!(state.status, RateLimitStatus::Overloaded);
        assert_eq!(state.next_retry_at.as_deref(), Some("2026-10-08T12:00:30Z"));
        // An unknown leaves the state as it was rather than dating a non-event.
        let mut state = empty_state();
        let read = classify(&Signals::new(openai(), 418), noon());
        apply_verdict(&mut state, &read, noon());
        assert_eq!(state.status, RateLimitStatus::Unknown);
        assert_eq!(state.since, None);
        assert_eq!(state.next_retry_at, None);
    }

    #[test]
    fn next_attempt_at_says_when_to_try_again() {
        let exhausted = Verdict::QuotaExhausted {
            reset_at: Some(noon() + chrono::Duration::seconds(60)),
        };
        let read = Classification {
            verdict: exhausted,
            signal: Signal::ProviderQuota,
            evidence: Evidence::Documented,
            reason: String::new(),
        };
        assert_eq!(
            read.next_attempt_at(noon()),
            Some(noon() + chrono::Duration::seconds(60))
        );
        let read = Classification {
            verdict: Verdict::Overloaded { retry_after: None },
            signal: Signal::Overload,
            evidence: Evidence::Documented,
            reason: String::new(),
        };
        assert_eq!(read.next_attempt_at(noon()), None);
    }

    #[test]
    fn the_reading_carries_the_evidence_grade_of_the_profile() {
        let read = classify(&Signals::new(profiles::NIM, 429), noon());
        assert_eq!(read.evidence, Evidence::Unproven);
        let read = classify(&Signals::new(openai(), 429), noon());
        assert_eq!(read.evidence, Evidence::Documented);
    }

    #[test]
    fn a_profile_that_does_not_exist_is_the_generic_one_not_a_panic() {
        let read = classify(&Signals::new("who-knows", 503), noon());
        assert_eq!(read.evidence, Evidence::Documented);
        assert_eq!(read.status(), RateLimitStatus::Overloaded);
    }

    // ---- the fallback used while nexus carries no wire signals ---------------

    #[test]
    fn a_provider_error_kind_without_a_quota_never_invents_one() {
        // nexus maps EVERY 429 to `rate_limited`, quota or not: on its own it
        // cannot be an exhaustion.
        for ours in [
            OurStanding::NoLimit,
            OurStanding::Unknown,
            OurStanding::Under,
            OurStanding::Near,
        ] {
            let read = from_provider_error("rate_limited", Some(30_000), ours, openai(), noon());
            assert!(!read.is_quota(), "{ours:?}: {}", read.reason);
            assert_eq!(
                read.verdict,
                Verdict::Overloaded {
                    retry_after: Some(Duration::from_secs(30))
                }
            );
        }
        // …unless OUR own counting is at its ceiling: then the refusal is explained.
        let read = from_provider_error(
            "rate_limited",
            Some(30_000),
            OurStanding::AtOrAbove,
            openai(),
            noon(),
        );
        assert!(read.is_quota(), "{}", read.reason);
        assert_eq!(read.signal, Signal::OurQuota);
    }

    #[test]
    fn a_provider_error_that_says_overloaded_is_an_overload() {
        let read =
            from_provider_error("overloaded", None, OurStanding::AtOrAbove, openai(), noon());
        assert!(!read.is_quota(), "{}", read.reason);
        assert_eq!(
            read.verdict,
            Verdict::Overloaded {
                retry_after: Some(Duration::from_secs(5))
            },
            "no delay from the provider: the profile's default"
        );
    }

    #[test]
    fn a_provider_error_of_another_kind_is_unknown() {
        let read = from_provider_error(
            "endpoint_unreachable",
            None,
            OurStanding::Under,
            openai(),
            noon(),
        );
        assert_eq!(read.verdict, Verdict::Unknown);
        assert!(!read.is_quota());
    }

    // ---- the builders -------------------------------------------------------

    #[test]
    fn the_builders_of_signals_default_to_the_safest_reading() {
        let read = classify(&Signals::new(openai(), 429), noon());
        assert!(!read.is_quota(), "{}", read.reason);
        let headers = [("x-ratelimit-remaining", "0")];
        let built = Signals::new(openai(), 429)
            .headers(&headers)
            .body(ErrorBody {
                code: None,
                kind: None,
            })
            .ours(OurStanding::Under)
            .consecutive(7);
        assert_eq!(built.http_status, 429);
        assert_eq!(built.ours, OurStanding::Under);
        assert_eq!(built.consecutive, 7);
        assert_eq!(built.body.code, None);
        assert_eq!(built.headers.len(), 1);
    }

    #[test]
    fn a_limit_is_carried_by_the_signals_without_being_read_here() {
        // The classifier never queries the ledger: the STANDING is what arrives,
        // so a configured limit is only ever visible through it.
        let standing = OurStanding::of(Some(39.0), Some(40.0));
        assert_eq!(standing, OurStanding::Near);
        assert_eq!(unused_limit().max, 40.0);
    }
}
