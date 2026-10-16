//! RL2 — the counters: how much went out, and may this go out right now.
//!
//! A configured limit (RL1) is a statement about a window: "40 requests per
//! minute", "200 000 tokens per hour", "10 USD per day". This module turns that
//! statement into a live number, and — the part that matters — into a DECISION
//! taken BEFORE a call goes out: [`Ledger::try_reserve`] is all-or-nothing over
//! every limit that applies to a request, under one lock, so two simultaneous
//! calls can never both slip past a ceiling of one. After the answer,
//! [`Ledger::settle`] reconciles the estimate with what was really consumed.
//!
//! # Which counter for which window
//!
//! * **rolling** (`{"rolling": 60}`) → a SLIDING window: the exact reading. Every
//!   take is kept with its instant and ages out of the window one by one;
//! * **a unit window** (`"minute"`, `"hour"`) → a TOKEN BUCKET of capacity `max`,
//!   refilled continuously over the window. A unit window measured exactly would
//!   hand back the whole allowance at each boundary (a burst at :00 of everything
//!   plus whatever the new window allows); the bucket smooths exactly that;
//! * **a calendar window** (`"day"`, `"week"`, `"month"`, always with its time
//!   zone) → a CUMUL over the period, and the only one that is PERSISTED: a
//!   restart at noon must not hand back a fresh day of budget.
//!
//! # What is measured, and what is only counted
//!
//! The cost of a call is the one the provider reports, i.e. the amount the
//! existing cost path already knows (`src/chat/cost.rs`). When a model's cost is
//! UNKNOWN, a budget limit does NOT block: the call is counted as a request
//! against that budget's window, a [`Note::UnpricedBudget`] is returned so the
//! caller can warn, and nothing is refused. Blocking on a price we do not know
//! would refuse work for a reason nobody could read — and the whole point of a
//! budget is the money, not the calls.
//!
//! The same honest limit applies to tokens: an estimate given before the call is
//! enforced exactly; with no estimate the counter enforces what is already known
//! and the true usage is added at settlement.
//!
//! # The clock
//!
//! [`Clock`] is injected: no test sleeps, and no rule depends on the machine's
//! wall clock. [`ManualClock`] is what the tests drive; [`SystemClock`] is what
//! production uses.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Datelike, NaiveDate, TimeZone as _, Utc};
use chrono_tz::Tz;

use super::classify::{rfc3339, OurStanding};
use super::store::LedgerStore;
use crate::chat::provider::settings::{
    rate_limit_report, InstanceRecord, RateLimit, RateLimitKind, RateLimitReport, RateLimitScope,
    RateLimitState, RateLimitStatus, RateLimitUnit, RateLimitWindow, RollingWindow,
};

/// Tolerance of the floating-point comparisons of counts and dollars.
const EPS: f64 = 1e-9;

// ============================================================================
// The clock
// ============================================================================

/// Where the ledger reads the time. Injected so that a window boundary is a test
/// fixture, not a machine's local hour.
pub trait Clock: Send + Sync + std::fmt::Debug {
    /// The current instant, in UTC.
    fn now(&self) -> DateTime<Utc>;
}

/// The real clock.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// A clock a test drives by hand: [`ManualClock::advance`] is the only thing that
/// makes time pass.
///
/// Panics at construction on a malformed instant: a fixture that cannot be read
/// would silently test the wrong moment.
#[derive(Debug)]
pub struct ManualClock {
    at: Mutex<DateTime<Utc>>,
}

impl ManualClock {
    /// A clock stopped at an RFC 3339 instant.
    pub fn at(instant: &str) -> Self {
        let at = DateTime::parse_from_rfc3339(instant)
            .expect("an RFC 3339 instant")
            .with_timezone(&Utc);
        Self { at: Mutex::new(at) }
    }

    /// A clock stopped at an instant already parsed.
    pub fn from_instant(instant: DateTime<Utc>) -> Self {
        Self {
            at: Mutex::new(instant),
        }
    }

    /// Moves the clock forward. The only way time passes for a manual clock.
    pub fn advance(&self, by: Duration) {
        let mut at = self.at.lock().unwrap_or_else(|e| e.into_inner());
        *at += chrono::Duration::from_std(by).unwrap_or_default();
    }
}

impl Clock for ManualClock {
    fn now(&self) -> DateTime<Utc> {
        *self.at.lock().unwrap_or_else(|e| e.into_inner())
    }
}

// ============================================================================
// The counters
// ============================================================================

/// A period of the calendar, in a time zone. Only these three are calendar
/// windows: a minute or an hour is a duration, however it is spelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalendarUnit {
    /// Midnight to midnight.
    Day,
    /// Monday to Monday (ISO weeks).
    Week,
    /// The first of the month to the first of the next.
    Month,
}

impl CalendarUnit {
    /// The calendar unit of a configured window, when it is one.
    pub fn of(unit: RateLimitUnit) -> Option<Self> {
        match unit {
            RateLimitUnit::Day => Some(Self::Day),
            RateLimitUnit::Week => Some(Self::Week),
            RateLimitUnit::Month => Some(Self::Month),
            RateLimitUnit::Minute | RateLimitUnit::Hour => None,
        }
    }

    /// The first day of the period `date` falls in.
    fn period_of(self, date: NaiveDate) -> NaiveDate {
        match self {
            Self::Day => date,
            Self::Week => {
                date - chrono::Duration::days(date.weekday().num_days_from_monday() as i64)
            }
            Self::Month => date.with_day(1).unwrap_or(date),
        }
    }

    /// The first day of the period after the one `date` falls in.
    fn next_period_of(self, date: NaiveDate) -> NaiveDate {
        match self {
            Self::Day => date + chrono::Duration::days(1),
            Self::Week => date + chrono::Duration::days(7),
            Self::Month => {
                let (year, month) = if date.month() == 12 {
                    (date.year() + 1, 1)
                } else {
                    (date.year(), date.month() + 1)
                };
                NaiveDate::from_ymd_opt(year, month, 1).unwrap_or(date)
            }
        }
    }
}

/// Midnight of a local date, in UTC. A date in a DST gap (a zone whose clocks skip
/// local midnight) falls back to the instant read as UTC: a boundary off by an
/// hour is a smaller lie than a boundary that panics.
fn midnight(tz: Tz, date: NaiveDate) -> DateTime<Utc> {
    let naive = date.and_time(chrono::NaiveTime::MIN);
    tz.from_local_datetime(&naive)
        .earliest()
        .map(|local| local.with_timezone(&Utc))
        .unwrap_or_else(|| naive.and_utc())
}

/// The start of the calendar period containing `now`.
fn period_start(unit: CalendarUnit, tz: Tz, now: DateTime<Utc>) -> DateTime<Utc> {
    midnight(tz, unit.period_of(now.with_timezone(&tz).date_naive()))
}

/// The start of the calendar period after the one containing `now`.
fn next_period_start(unit: CalendarUnit, tz: Tz, now: DateTime<Utc>) -> DateTime<Utc> {
    // The period the instant falls in FIRST, then the next one: taking the next
    // period OF THE DAY (a Thursday) would land a week window on the wrong Monday.
    let date = unit.period_of(now.with_timezone(&tz).date_naive());
    midnight(tz, unit.next_period_of(date))
}

/// The time zone of a limit: `"utc"` or an IANA name (RL1 validated it), defaulting
/// to UTC for a limit that carries none.
fn tz_of(calendar: Option<&str>) -> Tz {
    calendar
        .and_then(|name| name.trim().parse::<Tz>().ok())
        .unwrap_or(chrono_tz::UTC)
}

/// How long a window lasts in seconds. A calendar window's LENGTH is not what
/// bounds it (its boundaries are); this is only used for a bucket's refill rate.
fn window_secs(window: RateLimitWindow) -> u64 {
    match window {
        RateLimitWindow::Unit(RateLimitUnit::Minute) => 60,
        RateLimitWindow::Unit(RateLimitUnit::Hour) => 3_600,
        RateLimitWindow::Unit(RateLimitUnit::Day) => 86_400,
        RateLimitWindow::Unit(RateLimitUnit::Week) => 604_800,
        RateLimitWindow::Unit(RateLimitUnit::Month) => 2_592_000,
        RateLimitWindow::Rolling(RollingWindow { rolling }) => rolling,
    }
}

/// A sliding window: every take is kept with its instant and ages out exactly one
/// window later. The exact reading of a rolling limit.
#[derive(Debug, Clone, PartialEq)]
pub struct SlidingWindow {
    window: Duration,
    max: f64,
    takes: VecDeque<(DateTime<Utc>, f64)>,
}

impl SlidingWindow {
    fn new(window: u64, max: f64) -> Self {
        Self {
            window: Duration::from_secs(window),
            max,
            takes: VecDeque::new(),
        }
    }

    fn span(&self) -> chrono::Duration {
        chrono::Duration::from_std(self.window).unwrap_or_default()
    }

    fn used(&self, now: DateTime<Utc>) -> f64 {
        let start = now - self.span();
        self.takes
            .iter()
            .filter(|(at, _)| *at > start)
            .map(|(_, amount)| *amount)
            .sum()
    }

    /// Forgets the takes that have left the window: they no longer count, and
    /// keeping them would grow without bound.
    fn prune(&mut self, now: DateTime<Utc>) {
        let start = now - self.span();
        while self.takes.front().is_some_and(|(at, _)| *at <= start) {
            self.takes.pop_front();
        }
    }

    fn take_unchecked(&mut self, now: DateTime<Utc>, amount: f64) {
        self.prune(now);
        self.takes.push_back((now, amount));
    }

    /// Gives back the most recent takes, newest first: what a settlement releases
    /// is the take it just made.
    fn release(&mut self, now: DateTime<Utc>, amount: f64) {
        self.prune(now);
        let mut left = amount;
        while left > EPS {
            let Some((at, taken)) = self.takes.pop_back() else {
                break;
            };
            if taken > left {
                self.takes.push_back((at, taken - left));
                left = 0.0;
            } else {
                left -= taken;
            }
        }
    }

    /// When the window starts emptying: the instant its oldest take leaves. `None`
    /// when nothing was taken (there is nothing to wait for).
    fn resets_at(&self, _now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        self.takes.front().map(|(at, _)| *at + self.span())
    }
}

/// A token bucket: a capacity refilled continuously over the window. What a unit
/// window (`"minute"`, `"hour"`) is turned into.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenBucket {
    max: f64,
    rate: f64,
    tokens: f64,
    last: DateTime<Utc>,
}

impl TokenBucket {
    fn new(max: f64, window: u64, now: DateTime<Utc>) -> Self {
        let rate = if window == 0 {
            0.0
        } else {
            max / window as f64
        };
        Self {
            max,
            rate,
            tokens: max,
            last: now,
        }
    }

    /// The tokens available at `now`, without writing anything: reading a counter
    /// must not change it.
    fn refilled(&self, now: DateTime<Utc>) -> f64 {
        let elapsed = (now - self.last).num_milliseconds().max(0) as f64 / 1000.0;
        (self.tokens + elapsed * self.rate).min(self.max)
    }

    fn used(&self, now: DateTime<Utc>) -> f64 {
        self.max - self.refilled(now)
    }

    fn take_unchecked(&mut self, now: DateTime<Utc>, amount: f64) {
        let tokens = self.refilled(now);
        self.last = now;
        // The bucket may go NEGATIVE. Clamping at zero would hide a settlement that
        // cost more than the ceiling allowed — the very thing the caller must be
        // told about (`Note::Exceeded`), and what keeps the limit `limited` until it
        // refills instead of silently reading as full.
        self.tokens = tokens - amount;
    }

    fn release(&mut self, now: DateTime<Utc>, amount: f64) {
        let tokens = self.refilled(now);
        self.last = now;
        self.tokens = (tokens + amount).min(self.max);
    }

    /// When the bucket is full again — a bucket is refilled continuously, so this
    /// is the instant the pieces taken so far have all come back.
    fn resets_at(&self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        let missing = self.max - self.refilled(now);
        if missing <= EPS || self.rate <= 0.0 {
            return None;
        }
        let seconds = missing / self.rate;
        Some(now + chrono::Duration::milliseconds((seconds * 1000.0).ceil() as i64))
    }
}

/// A cumul over a calendar period, in its time zone. The only counter that is
/// persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct CalendarWindow {
    unit: CalendarUnit,
    tz: Tz,
    max: f64,
    start: DateTime<Utc>,
    used: f64,
    unpriced: u64,
}

impl CalendarWindow {
    fn new(unit: CalendarUnit, tz: Tz, max: f64, now: DateTime<Utc>) -> Self {
        Self {
            unit,
            tz,
            max,
            start: period_start(unit, tz, now),
            used: 0.0,
            unpriced: 0,
        }
    }

    /// Starts a new period when the calendar has moved on. A cumul from a passed
    /// period is not "carried over": the budget was for that day, that week, that
    /// month.
    fn roll(&mut self, now: DateTime<Utc>) {
        let start = period_start(self.unit, self.tz, now);
        if start != self.start {
            self.start = start;
            self.used = 0.0;
            self.unpriced = 0;
        }
    }

    fn used(&self, now: DateTime<Utc>) -> f64 {
        if period_start(self.unit, self.tz, now) != self.start {
            0.0
        } else {
            self.used
        }
    }

    fn take_unchecked(&mut self, now: DateTime<Utc>, amount: f64) {
        self.roll(now);
        self.used += amount;
    }

    fn release(&mut self, now: DateTime<Utc>, amount: f64) {
        self.roll(now);
        self.used = (self.used - amount).max(0.0);
    }

    /// Counts a call whose price we do not know. It never blocks: see the module
    /// documentation.
    fn add_unpriced(&mut self, now: DateTime<Utc>) -> u64 {
        self.roll(now);
        self.unpriced += 1;
        self.unpriced
    }

    /// The documented form: when the period ends.
    fn resets_at(&self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        Some(next_period_start(self.unit, self.tz, now))
    }

    /// What survives the process: the period, the cumul, and how many unpriced
    /// calls it counted.
    fn persisted(&self) -> serde_json::Value {
        serde_json::json!({
            "start": crate::chat::provider::limits::classify::rfc3339(self.start),
            "used": self.used,
            "unpriced": self.unpriced,
        })
    }

    /// Reads back what [`Self::persisted`] wrote. A document that does not read
    /// leaves the counter as it is: a corrupt cumul never becomes a false allowance.
    fn restore(&mut self, value: &serde_json::Value) {
        let start = value
            .get("start")
            .and_then(|raw| raw.as_str())
            .and_then(|raw| DateTime::parse_from_rfc3339(raw).ok())
            .map(|at| at.with_timezone(&Utc));
        let used = value.get("used").and_then(serde_json::Value::as_f64);
        let unpriced = value.get("unpriced").and_then(serde_json::Value::as_u64);
        if let Some(start) = start {
            self.start = start;
        }
        if let Some(used) = used {
            self.used = used;
        }
        if let Some(unpriced) = unpriced {
            self.unpriced = unpriced;
        }
    }
}

/// The counter of one limit: the algorithm its window calls for.
#[derive(Debug, Clone, PartialEq)]
pub enum Counter {
    /// A rolling window.
    Sliding(SlidingWindow),
    /// A unit window.
    Bucket(TokenBucket),
    /// A calendar window.
    Calendar(CalendarWindow),
}

impl Counter {
    /// The counter the window calls for (see the module documentation).
    fn new(limit: &RateLimit, now: DateTime<Utc>) -> Self {
        match limit.window {
            RateLimitWindow::Rolling(_) => {
                Counter::Sliding(SlidingWindow::new(window_secs(limit.window), limit.max))
            }
            RateLimitWindow::Unit(unit) => match CalendarUnit::of(unit) {
                Some(calendar) => Counter::Calendar(CalendarWindow::new(
                    calendar,
                    tz_of(limit.calendar.as_deref()),
                    limit.max,
                    now,
                )),
                None => {
                    Counter::Bucket(TokenBucket::new(limit.max, window_secs(limit.window), now))
                }
            },
        }
    }

    /// What the counter counts now.
    pub fn used(&self, now: DateTime<Utc>) -> f64 {
        match self {
            Counter::Sliding(window) => window.used(now),
            Counter::Bucket(bucket) => bucket.used(now),
            Counter::Calendar(window) => window.used(now),
        }
    }

    /// Whether `amount` may be taken without passing the ceiling, without taking
    /// anything. An amount of zero (an unknown price, an unestimated count) is
    /// admitted only while the counter is NOT already full.
    fn admits(&self, used: f64, amount: f64) -> bool {
        if amount > EPS {
            used + amount <= self.max() + EPS
        } else {
            used < self.max() - EPS
        }
    }

    /// Takes without recording. Only for putting things right — the true cost of
    /// a call that already happened.
    fn take_unchecked(&mut self, now: DateTime<Utc>, amount: f64) {
        match self {
            Counter::Sliding(window) => window.take_unchecked(now, amount),
            Counter::Bucket(bucket) => bucket.take_unchecked(now, amount),
            Counter::Calendar(window) => window.take_unchecked(now, amount),
        }
    }

    /// Gives a take back.
    fn release(&mut self, now: DateTime<Utc>, amount: f64) {
        match self {
            Counter::Sliding(window) => window.release(now, amount),
            Counter::Bucket(bucket) => bucket.release(now, amount),
            Counter::Calendar(window) => window.release(now, amount),
        }
    }

    /// Counts an unpriced call (calendar windows only: the others count what they
    /// take and have no notion of a price).
    fn add_unpriced(&mut self, now: DateTime<Utc>) -> u64 {
        if let Counter::Calendar(window) = self {
            window.add_unpriced(now)
        } else {
            0
        }
    }

    /// When the ceiling next makes room.
    fn resets_at(&self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        match self {
            Counter::Sliding(window) => window.resets_at(now),
            Counter::Bucket(bucket) => bucket.resets_at(now),
            Counter::Calendar(window) => window.resets_at(now),
        }
    }

    /// The ceiling this counter enforces.
    fn max(&self) -> f64 {
        match self {
            Counter::Sliding(window) => window.max,
            Counter::Bucket(bucket) => bucket.max,
            Counter::Calendar(window) => window.max,
        }
    }

    /// What to persist, when the counter outlives the process.
    fn persisted(&self) -> Option<serde_json::Value> {
        match self {
            Counter::Calendar(window) => Some(window.persisted()),
            _ => None,
        }
    }

    /// Reads back what [`Self::persisted`] wrote.
    fn restore(&mut self, value: &serde_json::Value) {
        if let Counter::Calendar(window) = self {
            window.restore(value);
        }
    }
}

// ============================================================================
// The ledger
// ============================================================================

/// What a caller knows about the call it is about to make.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Ask {
    /// The credential REFERENCE the call will use (`vault:<name>`, `env:<VAR>`),
    /// never a value: limits scoped to a credential are matched on it.
    pub credential_ref: Option<String>,
    /// The model the call will use.
    pub model: Option<String>,
    /// The tokens the caller estimates, when it can.
    pub tokens: Option<f64>,
    /// The dollars the caller expects, when it knows them.
    pub cost_usd: Option<f64>,
}

/// One limit's reservation: the key it was taken under, the limit, the amount.
#[derive(Debug, Clone, PartialEq)]
pub struct Take {
    /// The ledger key of the limit.
    pub key: String,
    /// The configured limit.
    pub limit: RateLimit,
    /// What was reserved for it.
    pub amount: f64,
}

/// A granted reservation: everything that was reserved for one call.
#[derive(Debug, Clone, PartialEq)]
pub struct Grant {
    /// The instance the call goes to.
    pub instance: String,
    /// One entry per applicable limit (an unpriced budget is in here with 0).
    pub takes: Vec<Take>,
}

/// A refused reservation: the limit that refused, and when it makes room.
#[derive(Debug, Clone, PartialEq)]
pub struct Denial {
    /// What the refusing limit counts.
    pub kind: RateLimitKind,
    /// Its window, as a human reads it.
    pub window: String,
    /// Whom it applies to.
    pub scope: String,
    /// Its ceiling.
    pub max: f64,
    /// When the next attempt may be made, when known.
    pub retry_at: Option<DateTime<Utc>>,
}

impl Denial {
    /// A line a human can read. A credential is named by its REFERENCE.
    pub fn message(&self) -> String {
        format!(
            "limit reached: {} {} per {} ({})",
            self.max,
            kind_word(self.kind),
            self.window,
            self.scope
        )
    }
}

/// The answer of [`Ledger::try_reserve`].
#[derive(Debug, Clone, PartialEq)]
pub enum Reservation {
    /// The call may go out. Settle it.
    Granted(Grant),
    /// The call must not go out.
    Denied(Denial),
}

impl Reservation {
    /// Whether the call may go out.
    pub fn is_granted(&self) -> bool {
        matches!(self, Self::Granted(_))
    }

    /// The refusal, when there is one.
    pub fn denial(&self) -> Option<&Denial> {
        match self {
            Self::Denied(denial) => Some(denial),
            Self::Granted(_) => None,
        }
    }
}

/// What the call did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Outcome {
    /// It reached the provider: here is what it consumed (`None` = the provider
    /// did not say).
    Consumed {
        /// Tokens really consumed, when the provider reported them.
        tokens: Option<f64>,
        /// Dollars really spent, when the cost path measured them.
        cost_usd: Option<f64>,
    },
    /// It never went out (or failed before consuming anything): give everything back.
    NeverSent,
}

/// What the caller should know after settling.
#[derive(Debug, Clone, PartialEq)]
pub enum Note {
    /// A budget limit could not be priced: the call was counted against it as a
    /// request, and did not block. The caller warns; nothing is refused.
    UnpricedBudget {
        /// Whom the limit applies to.
        scope: String,
        /// Its window, as a human reads it.
        window: String,
        /// How many unpriced calls this period now holds.
        unpriced: u64,
    },
    /// The settlement pushed a counter above its ceiling (the call really cost more
    /// than was reserved): nothing else goes out until it resets.
    Exceeded {
        /// Whom the limit applies to.
        scope: String,
        /// What it counts now.
        used: f64,
        /// Its ceiling.
        max: f64,
    },
}

impl Note {
    /// A line a human (or a log) can read. Never a secret: a scope labels a
    /// reference or a model id.
    pub fn message(&self) -> String {
        match self {
            Self::UnpricedBudget {
                scope,
                window,
                unpriced,
            } => format!(
                "budget limit ({scope}, per {window}): {unpriced} call(s) counted as requests, \
                 their model has no known cost — not blocking"
            ),
            Self::Exceeded { scope, used, max } => {
                format!("limit exceeded ({scope}): {used} of {max}, nothing else goes out until it resets")
            }
        }
    }
}

/// One limit's live counter, plus what the last refusal said about it.
#[derive(Debug, Clone, PartialEq)]
struct Tracked {
    counter: Counter,
    since: Option<DateTime<Utc>>,
    next_retry_at: Option<DateTime<Utc>>,
}

/// The ledger: every instance's counters, and the atomic decision to send or not.
#[derive(Debug)]
pub struct Ledger {
    clock: Arc<dyn Clock>,
    store: Arc<dyn LedgerStore>,
    inner: Mutex<HashMap<String, Tracked>>,
}

impl Ledger {
    /// A ledger reading `clock`, remembering its long cumuls in `store`.
    pub fn new(clock: Arc<dyn Clock>, store: Arc<dyn LedgerStore>) -> Self {
        Self {
            clock,
            store,
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// The decision, taken BEFORE a call goes out: EVERY applicable limit is
    /// checked and reserved under one lock, and if one refuses, everything already
    /// reserved for this call is given back. Two simultaneous calls therefore
    /// cannot both slip past a ceiling of one.
    pub fn try_reserve(&self, instance: &str, limits: &[RateLimit], ask: &Ask) -> Reservation {
        let now = self.clock.now();
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut takes: Vec<Take> = Vec::new();
        for limit in limits.iter().filter(|l| applies(&l.scope, ask)) {
            let key = key_of(instance, limit);
            // A budget limit whose cost is unknown does not block: it is counted
            // as a request at settlement (see the module documentation).
            let amount = match limit.kind {
                RateLimitKind::Requests => 1.0,
                RateLimitKind::Tokens => ask.tokens.unwrap_or(0.0),
                RateLimitKind::BudgetUsd => match ask.cost_usd {
                    Some(cost) => cost,
                    None => {
                        // The counter must exist BEFORE the settlement records the
                        // unpriced call against it, and so that a restart finds the
                        // very cumul it left.
                        self.ensure(&mut guard, instance, limit, now);
                        takes.push(Take {
                            key,
                            limit: limit.clone(),
                            amount: 0.0,
                        });
                        continue;
                    }
                },
            };
            let tracked = self.ensure(&mut guard, instance, limit, now);
            let used = tracked.counter.used(now);
            if tracked.counter.admits(used, amount) {
                tracked.counter.take_unchecked(now, amount);
                let retry_at = tracked.counter.resets_at(now);
                tracked.next_retry_at = retry_at;
                self.persist(&key, &tracked.counter);
                takes.push(Take {
                    key,
                    limit: limit.clone(),
                    amount,
                });
            } else {
                let denial = Denial {
                    kind: limit.kind,
                    window: window_label(limit.window),
                    scope: limit.scope.label(),
                    max: limit.max,
                    retry_at: tracked.counter.resets_at(now),
                };
                // All or nothing: what the limits already reserved for THIS call
                // goes back before the refusal is answered.
                for take in &takes {
                    if let Some(tracked) = guard.get_mut(&take.key) {
                        tracked.counter.release(now, take.amount);
                    }
                }
                return Reservation::Denied(denial);
            }
        }
        Reservation::Granted(Grant {
            instance: instance.to_string(),
            takes,
        })
    }

    /// What the call really consumed. A call that never went out gives everything
    /// back; a cost the provider did not report on an unpriced model is counted as
    /// a request and reported in a [`Note`], never as a refusal.
    pub fn settle(&self, reservation: Reservation, outcome: Outcome) -> Vec<Note> {
        let Reservation::Granted(grant) = reservation else {
            return Vec::new();
        };
        let now = self.clock.now();
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut notes = Vec::new();
        for take in grant.takes {
            let Some(tracked) = guard.get_mut(&take.key) else {
                continue;
            };
            match outcome {
                Outcome::NeverSent => tracked.counter.release(now, take.amount),
                Outcome::Consumed { tokens, cost_usd } => match take.limit.kind {
                    RateLimitKind::Requests => {}
                    RateLimitKind::Tokens => {
                        adjust(
                            &mut tracked.counter,
                            now,
                            take.amount,
                            tokens.unwrap_or(0.0),
                        );
                    }
                    RateLimitKind::BudgetUsd => match cost_usd {
                        Some(cost) => adjust(&mut tracked.counter, now, take.amount, cost),
                        None => {
                            let unpriced = tracked.counter.add_unpriced(now);
                            notes.push(Note::UnpricedBudget {
                                scope: take.limit.scope.label(),
                                window: window_label(take.limit.window),
                                unpriced,
                            });
                        }
                    },
                },
            }
            let used = tracked.counter.used(now);
            if used > take.limit.max + EPS {
                tracked.since = Some(now);
                tracked.next_retry_at = tracked.counter.resets_at(now);
                notes.push(Note::Exceeded {
                    scope: take.limit.scope.label(),
                    used,
                    max: take.limit.max,
                });
            }
            self.persist(&take.key, &tracked.counter);
        }
        notes
    }

    /// Where our own counting stands for the call about to be classified: the
    /// WORST of the applicable limits, because one ceiling reached is enough to
    /// explain a refusal.
    pub fn standing(&self, instance: &str, limits: &[RateLimit], ask: &Ask) -> OurStanding {
        let now = self.clock.now();
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut standing = OurStanding::NoLimit;
        for limit in limits.iter().filter(|l| applies(&l.scope, ask)) {
            let tracked = self.ensure(&mut guard, instance, limit, now);
            let used = tracked.counter.used(now);
            standing = worst(standing, OurStanding::of(Some(used), Some(limit.max)));
        }
        standing
    }

    /// RL1's report shape, with RL2's numbers in it.
    pub fn report(&self, record: &InstanceRecord) -> RateLimitReport {
        let now = self.clock.now();
        let mut report = rate_limit_report(record);
        {
            let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            for (state, limit) in report.limits.iter_mut().zip(record.rate_limit.iter()) {
                let tracked = self.ensure(&mut guard, &record.id, limit, now);
                fill_state(state, tracked, limit, now);
            }
        }
        report.status = report
            .limits
            .iter()
            .map(|state| state.status)
            .fold(RateLimitStatus::Ok, worse_status);
        report
    }

    /// The live counter of a limit, restored from the store on first use.
    fn ensure<'g>(
        &self,
        guard: &'g mut HashMap<String, Tracked>,
        instance: &str,
        limit: &RateLimit,
        now: DateTime<Utc>,
    ) -> &'g mut Tracked {
        let key = key_of(instance, limit);
        guard.entry(key.clone()).or_insert_with(|| {
            let mut counter = Counter::new(limit, now);
            if let Some(document) = self.store.load(&key) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&document) {
                    counter.restore(&value);
                }
            }
            Tracked {
                counter,
                since: None,
                next_retry_at: None,
            }
        })
    }

    /// Writes the counters that must outlive the process.
    fn persist(&self, key: &str, counter: &Counter) {
        if let Some(document) = counter.persisted() {
            self.store.save(key, &document.to_string());
        }
    }
}

/// Whether a limit applies to the call being made.
fn applies(scope: &RateLimitScope, ask: &Ask) -> bool {
    match scope {
        RateLimitScope::Instance(_) => true,
        RateLimitScope::Credential(credential) => {
            ask.credential_ref.as_deref() == Some(credential.credential.as_str())
        }
        RateLimitScope::Model(model) => ask.model.as_deref() == Some(model.model.as_str()),
    }
}

/// The stable key of one configured limit on one instance. Derived from the
/// configuration itself, so a restart finds the very counter it left.
fn key_of(instance: &str, limit: &RateLimit) -> String {
    format!(
        "{instance}|{}",
        serde_json::to_string(limit).unwrap_or_default()
    )
}

/// Puts the true consumption in place of the estimate, in either direction.
fn adjust(counter: &mut Counter, now: DateTime<Utc>, reserved: f64, actual: f64) {
    if actual > reserved {
        counter.take_unchecked(now, actual - reserved);
    } else {
        counter.release(now, reserved - actual);
    }
}

/// The word a human reads for what a limit counts.
fn kind_word(kind: RateLimitKind) -> &'static str {
    match kind {
        RateLimitKind::Requests => "requests",
        RateLimitKind::Tokens => "tokens",
        RateLimitKind::BudgetUsd => "USD",
    }
}

/// A window as a human reads it.
fn window_label(window: RateLimitWindow) -> String {
    match window {
        RateLimitWindow::Unit(unit) => unit.as_str().to_string(),
        RateLimitWindow::Rolling(RollingWindow { rolling }) => format!("rolling {rolling}s"),
    }
}

/// The worst of two standings: one ceiling reached is enough to explain a refusal.
fn worst(left: OurStanding, right: OurStanding) -> OurStanding {
    if rank(right) > rank(left) {
        right
    } else {
        left
    }
}

fn rank(standing: OurStanding) -> u8 {
    match standing {
        OurStanding::NoLimit => 0,
        OurStanding::Unknown => 1,
        OurStanding::Under => 2,
        OurStanding::Near => 3,
        OurStanding::AtOrAbove => 4,
    }
}

/// The worst status of a set of limits.
fn worse_status(left: RateLimitStatus, right: RateLimitStatus) -> RateLimitStatus {
    if status_rank(right) > status_rank(left) {
        right
    } else {
        left
    }
}

fn status_rank(status: RateLimitStatus) -> u8 {
    match status {
        RateLimitStatus::Ok => 0,
        RateLimitStatus::Unknown => 1,
        RateLimitStatus::Overloaded => 2,
        RateLimitStatus::Limited => 3,
    }
}

/// Fills RL1's state with what RL2 measured. No key changes: only the numbers and
/// the status that RL7 left as `unknown`.
fn fill_state(
    state: &mut RateLimitState,
    tracked: &Tracked,
    limit: &RateLimit,
    now: DateTime<Utc>,
) {
    let used = tracked.counter.used(now);
    state.used = Some(used);
    state.remaining = Some((limit.max - used).max(0.0));
    state.resets_at = tracked.counter.resets_at(now).map(rfc3339);
    state.status = if used >= limit.max - EPS {
        RateLimitStatus::Limited
    } else {
        RateLimitStatus::Ok
    };
    state.since = tracked.since.map(rfc3339);
    state.next_retry_at = tracked.next_retry_at.map(rfc3339);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::limits::store::MemoryStore;
    use crate::chat::provider::settings::{
        CredentialScope, InstanceScope, ModelScope, RateLimitWindow,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    const T0: &str = "2026-10-08T12:00:00Z";

    fn instant(text: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(text)
            .unwrap()
            .with_timezone(&Utc)
    }

    fn ledger(store: Arc<dyn LedgerStore>) -> (Arc<ManualClock>, Ledger) {
        let clock = Arc::new(ManualClock::at(T0));
        let ledger = Ledger::new(clock.clone(), store);
        (clock, ledger)
    }

    fn instance_scope() -> RateLimitScope {
        RateLimitScope::Instance(InstanceScope::Instance)
    }

    fn rolling(secs: u64) -> RateLimitWindow {
        RateLimitWindow::Rolling(RollingWindow { rolling: secs })
    }

    fn unit(window: RateLimitUnit) -> RateLimitWindow {
        RateLimitWindow::Unit(window)
    }

    fn counted(kind: RateLimitKind, window: RateLimitWindow, max: f64) -> RateLimit {
        RateLimit {
            kind,
            window,
            calendar: None,
            max,
            scope: instance_scope(),
        }
    }

    fn requests(window: RateLimitWindow, max: f64) -> RateLimit {
        counted(RateLimitKind::Requests, window, max)
    }

    fn tokens(window: RateLimitWindow, max: f64) -> RateLimit {
        counted(RateLimitKind::Tokens, window, max)
    }

    fn budget(calendar: &str, window: RateLimitUnit, max: f64) -> RateLimit {
        RateLimit {
            kind: RateLimitKind::BudgetUsd,
            window: unit(window),
            calendar: Some(calendar.to_string()),
            max,
            scope: instance_scope(),
        }
    }

    fn record(limits: Vec<RateLimit>) -> InstanceRecord {
        InstanceRecord {
            id: "deepseek".into(),
            rate_limit: limits,
            ..Default::default()
        }
    }

    // ---- requests: the exact ceiling, and the concurrency that must not pass it

    #[test]
    fn a_requests_limit_admits_exactly_max_then_denies() {
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let limits = vec![requests(rolling(60), 2.0)];
        let ask = Ask::default();
        assert!(ledger.try_reserve("nim", &limits, &ask).is_granted());
        assert!(ledger.try_reserve("nim", &limits, &ask).is_granted());
        let refused = ledger.try_reserve("nim", &limits, &ask);
        assert!(!refused.is_granted());
        let denial = refused.denial().expect("a denial");
        assert_eq!(denial.kind, RateLimitKind::Requests);
        assert_eq!(denial.max, 2.0);
        assert_eq!(denial.scope, "instance");
        assert_eq!(denial.window, "rolling 60s");
        assert_eq!(
            denial.retry_at,
            Some(instant(T0) + chrono::Duration::seconds(60))
        );
        assert_eq!(
            denial.message(),
            "limit reached: 2 requests per rolling 60s (instance)"
        );
    }

    #[test]
    fn two_simultaneous_reservations_never_exceed_the_limit() {
        // The point of reserving BEFORE sending, under one lock: whatever the
        // interleaving, ten is ten.
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let ledger = Arc::new(ledger);
        let limits = vec![requests(rolling(60), 10.0)];
        let granted = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let ledger = Arc::clone(&ledger);
            let limits = limits.clone();
            let granted = Arc::clone(&granted);
            handles.push(std::thread::spawn(move || {
                for _ in 0..5 {
                    if ledger
                        .try_reserve("nim", &limits, &Ask::default())
                        .is_granted()
                    {
                        granted.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().expect("thread");
        }
        assert_eq!(granted.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn a_call_that_never_went_out_gives_its_slot_back() {
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let limits = vec![requests(rolling(60), 1.0)];
        let ask = Ask::default();
        let grant = ledger.try_reserve("nim", &limits, &ask);
        assert!(grant.is_granted());
        assert!(!ledger.try_reserve("nim", &limits, &ask).is_granted());
        assert!(
            ledger.settle(grant, Outcome::NeverSent).is_empty(),
            "a call that never went out has nothing to warn about"
        );
        // The slot is back.
        assert!(ledger.try_reserve("nim", &limits, &ask).is_granted());
    }

    #[test]
    fn settling_a_denial_is_a_no_op() {
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let limits = vec![requests(rolling(60), 1.0)];
        let ask = Ask::default();
        let _ = ledger.try_reserve("nim", &limits, &ask);
        let refused = ledger.try_reserve("nim", &limits, &ask);
        assert!(refused.denial().is_some());
        assert!(ledger
            .settle(
                refused,
                Outcome::Consumed {
                    tokens: None,
                    cost_usd: None
                }
            )
            .is_empty());
    }

    // ---- the windows ---------------------------------------------------------

    #[test]
    fn a_rolling_window_frees_its_slot_when_it_slides() {
        let (clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let limits = vec![requests(rolling(60), 2.0)];
        let ask = Ask::default();
        assert!(ledger.try_reserve("nim", &limits, &ask).is_granted());
        assert!(ledger.try_reserve("nim", &limits, &ask).is_granted());
        clock.advance(Duration::from_secs(30));
        assert!(!ledger.try_reserve("nim", &limits, &ask).is_granted());
        // The first take is 60 s old: it leaves the window and hands its slot back.
        clock.advance(Duration::from_secs(31));
        assert!(ledger.try_reserve("nim", &limits, &ask).is_granted());
    }

    #[test]
    fn a_unit_window_is_a_bucket_that_refills_continuously() {
        let (clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let limits = vec![requests(unit(RateLimitUnit::Minute), 60.0)];
        let ask = Ask::default();
        for _ in 0..60 {
            assert!(ledger.try_reserve("nim", &limits, &ask).is_granted());
        }
        assert!(!ledger.try_reserve("nim", &limits, &ask).is_granted());
        // One token a second: ten seconds give ten more.
        clock.advance(Duration::from_secs(10));
        for _ in 0..10 {
            assert!(ledger.try_reserve("nim", &limits, &ask).is_granted());
        }
        assert!(!ledger.try_reserve("nim", &limits, &ask).is_granted());
    }

    #[test]
    fn a_token_estimate_is_settled_to_the_actual_usage() {
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let limits = vec![tokens(unit(RateLimitUnit::Hour), 1000.0)];
        let ask = Ask {
            tokens: Some(400.0),
            ..Default::default()
        };
        let grant = ledger.try_reserve("deepseek", &limits, &ask);
        assert!(grant.is_granted());
        // We reserved 400 and used 150: the difference comes back.
        let notes = ledger.settle(
            grant,
            Outcome::Consumed {
                tokens: Some(150.0),
                cost_usd: None,
            },
        );
        assert!(notes.is_empty(), "{notes:?}");
        let report = ledger.report(&record(limits));
        assert_eq!(report.limits[0].used, Some(150.0));
        assert_eq!(report.limits[0].remaining, Some(850.0));
    }

    #[test]
    fn a_settlement_that_cost_more_than_reserved_says_so() {
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let limits = vec![tokens(unit(RateLimitUnit::Hour), 1000.0)];
        let ask = Ask {
            tokens: Some(100.0),
            ..Default::default()
        };
        let grant = ledger.try_reserve("deepseek", &limits, &ask);
        // The provider reports 1 200: the call already happened, the counter says so.
        let notes = ledger.settle(
            grant,
            Outcome::Consumed {
                tokens: Some(1200.0),
                cost_usd: None,
            },
        );
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(matches!(notes[0], Note::Exceeded { .. }));
        assert!(
            notes[0].message().contains("1200"),
            "{}",
            notes[0].message()
        );
        let report = ledger.report(&record(limits));
        assert_eq!(report.limits[0].used, Some(1200.0));
        assert_eq!(report.limits[0].remaining, Some(0.0));
        assert_eq!(report.limits[0].status, RateLimitStatus::Limited);
    }

    // ---- the budget ----------------------------------------------------------

    #[test]
    fn a_calendar_budget_counts_the_cost_the_cost_path_measured() {
        let (clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let limits = vec![budget("utc", RateLimitUnit::Day, 10.0)];
        let ask = Ask {
            cost_usd: Some(3.0),
            ..Default::default()
        };
        let grant = ledger.try_reserve("deepseek", &limits, &ask);
        assert!(grant.is_granted());
        let notes = ledger.settle(
            grant,
            Outcome::Consumed {
                tokens: None,
                cost_usd: Some(2.5),
            },
        );
        assert!(notes.is_empty(), "{notes:?}");
        let report = ledger.report(&record(limits.clone()));
        assert_eq!(report.limits[0].used, Some(2.5));
        assert_eq!(report.limits[0].remaining, Some(7.5));
        // Midnight UTC ends the day: 2026-10-09T00:00:00Z.
        assert_eq!(
            report.limits[0].resets_at.as_deref(),
            Some("2026-10-09T00:00:00Z")
        );
        let _ = clock;
    }

    #[test]
    fn a_budget_cumul_from_yesterday_is_not_carried_into_today() {
        let (clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let limits = vec![budget("utc", RateLimitUnit::Day, 10.0)];
        let ask = Ask {
            cost_usd: Some(9.0),
            ..Default::default()
        };
        assert!(ledger.try_reserve("deepseek", &limits, &ask).is_granted());
        assert!(!ledger.try_reserve("deepseek", &limits, &ask).is_granted());
        clock.advance(Duration::from_secs(13 * 3600));
        let report = ledger.report(&record(limits.clone()));
        assert_eq!(report.limits[0].used, Some(0.0), "the day rolled in UTC");
        assert!(ledger.try_reserve("deepseek", &limits, &ask).is_granted());
    }

    #[test]
    fn a_calendar_budget_resets_at_midnight_in_its_time_zone() {
        // 22:30 UTC is already 00:30 the next day in Paris: the Paris day has
        // rolled while the UTC day has not. That is why the limit carries a zone.
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let paris = vec![budget("Europe/Paris", RateLimitUnit::Day, 10.0)];
        let utc = vec![budget("utc", RateLimitUnit::Day, 10.0)];
        let report = ledger.report(&record(paris));
        // Midnight Paris (UTC+2 until late October) is 22:00 UTC.
        assert_eq!(
            report.limits[0].resets_at.as_deref(),
            Some("2026-10-08T22:00:00Z")
        );
        assert!(
            report.limits[0].used.is_some(),
            "a fresh counter reads zero, not unknown"
        );
        let report = ledger.report(&record(utc));
        assert_eq!(
            report.limits[0].resets_at.as_deref(),
            Some("2026-10-09T00:00:00Z")
        );
    }

    #[test]
    fn a_week_is_iso_and_a_month_crossing_december_is_the_next_year() {
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        // 2026-10-08 is a Thursday: the ISO week ends on Monday 2026-10-12.
        let week = vec![budget("utc", RateLimitUnit::Week, 10.0)];
        let report = ledger.report(&record(week));
        assert_eq!(
            report.limits[0].resets_at.as_deref(),
            Some("2026-10-12T00:00:00Z")
        );
        // A month window on 31 December ends on 1 January of the next year.
        let clock = Arc::new(ManualClock::at("2026-12-31T23:00:00Z"));
        let ledger = Ledger::new(clock, Arc::new(MemoryStore::new()));
        let month = vec![budget("utc", RateLimitUnit::Month, 10.0)];
        let report = ledger.report(&record(month));
        assert_eq!(
            report.limits[0].resets_at.as_deref(),
            Some("2027-01-01T00:00:00Z")
        );
    }

    #[test]
    fn an_unpriced_budget_counts_a_request_and_never_blocks() {
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let limits = vec![budget("utc", RateLimitUnit::Day, 10.0)];
        // No price for this model: the caller does not know what the call will cost.
        let ask = Ask::default();
        let grant = ledger.try_reserve("deepseek", &limits, &ask);
        assert!(grant.is_granted(), "an unknown price never blocks the work");
        let notes = ledger.settle(
            grant,
            Outcome::Consumed {
                tokens: None,
                cost_usd: None,
            },
        );
        assert_eq!(notes.len(), 1, "{notes:?}");
        let Note::UnpricedBudget {
            ref scope,
            ref window,
            unpriced,
        } = notes[0]
        else {
            panic!("expected an unpriced-budget note: {notes:?}");
        };
        assert_eq!(scope, "instance");
        assert_eq!(window, "day");
        assert_eq!(unpriced, 1);
        assert!(
            notes[0].message().contains("no known cost"),
            "{}",
            notes[0].message()
        );
        assert!(
            notes[0].message().contains("not blocking"),
            "{}",
            notes[0].message()
        );
        // The budget did not spend anything, and it still does not block.
        let report = ledger.report(&record(limits.clone()));
        assert_eq!(report.limits[0].used, Some(0.0));
        assert_eq!(report.limits[0].remaining, Some(10.0));
        assert!(ledger.try_reserve("deepseek", &limits, &ask).is_granted());
    }

    // ---- persistence ---------------------------------------------------------

    #[test]
    fn a_long_cumul_survives_a_restart_and_a_short_window_does_not() {
        let store: Arc<dyn LedgerStore> = Arc::new(MemoryStore::new());
        let (clock, ledger) = ledger(Arc::clone(&store));
        let limits = vec![
            budget("utc", RateLimitUnit::Day, 10.0),
            requests(rolling(60), 5.0),
        ];
        let ask = Ask {
            cost_usd: Some(4.0),
            ..Default::default()
        };
        let grant = ledger.try_reserve("deepseek", &limits, &ask);
        assert!(grant.is_granted());
        let _ = ledger.settle(
            grant,
            Outcome::Consumed {
                tokens: None,
                cost_usd: Some(4.0),
            },
        );
        drop(ledger);
        // A new process, the same store: the day's money is still spent.
        let restarted = Ledger::new(clock, Arc::clone(&store));
        let report = restarted.report(&record(limits));
        assert_eq!(report.limits[0].used, Some(4.0), "the budget survived");
        assert_eq!(
            report.limits[1].used,
            Some(0.0),
            "the rolling window did not"
        );
    }

    #[test]
    fn a_corrupt_document_leaves_the_counter_as_it_is() {
        let store: Arc<dyn LedgerStore> = Arc::new(MemoryStore::new());
        let limits = vec![budget("utc", RateLimitUnit::Day, 10.0)];
        let key = key_of("deepseek", &limits[0]);
        store.save(&key, "{\"start\": \"whenever\", \"used\": \"lots\"}");
        let (_clock, ledger) = ledger(Arc::clone(&store));
        let report = ledger.report(&record(limits));
        assert_eq!(report.limits[0].used, Some(0.0));
        // A document that is not JSON at all is ignored the same way.
        let limits = vec![budget("utc", RateLimitUnit::Day, 10.0)];
        let key = key_of("deepseek", &limits[0]);
        store.save(&key, "not json");
        let report = ledger.report(&record(limits));
        assert_eq!(report.limits[0].used, Some(0.0));
    }

    #[test]
    fn the_file_store_keeps_a_budget_across_processes() {
        let dir = std::env::temp_dir().join(format!("po-limits-ledger-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch");
        let path = dir.join("limits.json");
        let limits = vec![budget("utc", RateLimitUnit::Day, 10.0)];
        {
            let store: Arc<dyn LedgerStore> =
                Arc::new(super::super::store::FileStore::open(&path).expect("open"));
            let (clock, ledger) = ledger(store);
            let ask = Ask {
                cost_usd: Some(6.0),
                ..Default::default()
            };
            let grant = ledger.try_reserve("deepseek", &limits, &ask);
            let _ = ledger.settle(
                grant,
                Outcome::Consumed {
                    tokens: None,
                    cost_usd: Some(6.0),
                },
            );
            let _ = clock;
        }
        // Reopened from disk, in a fresh ledger.
        let store: Arc<dyn LedgerStore> =
            Arc::new(super::super::store::FileStore::open(&path).expect("reopen"));
        let (_clock, ledger) = ledger(store);
        let report = ledger.report(&record(limits));
        assert_eq!(report.limits[0].used, Some(6.0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- scope ---------------------------------------------------------------

    #[test]
    fn a_credential_scoped_limit_only_applies_to_its_credential() {
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let limits = vec![RateLimit {
            kind: RateLimitKind::Requests,
            window: rolling(60),
            calendar: None,
            max: 1.0,
            scope: RateLimitScope::Credential(CredentialScope {
                credential: "vault:nim".into(),
            }),
        }];
        // Another credential of the same instance is not bound by this limit.
        let other = Ask {
            credential_ref: Some("vault:other".into()),
            ..Default::default()
        };
        assert!(ledger.try_reserve("nim", &limits, &other).is_granted());
        assert!(ledger.try_reserve("nim", &limits, &other).is_granted());
        // The named one is.
        let named = Ask {
            credential_ref: Some("vault:nim".into()),
            ..Default::default()
        };
        assert!(ledger.try_reserve("nim", &limits, &named).is_granted());
        assert!(!ledger.try_reserve("nim", &limits, &named).is_granted());
    }

    #[test]
    fn a_model_scoped_limit_only_applies_to_its_model() {
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let limits = vec![RateLimit {
            kind: RateLimitKind::Requests,
            window: rolling(60),
            calendar: None,
            max: 1.0,
            scope: RateLimitScope::Model(ModelScope {
                model: "deepseek-reasoner".into(),
            }),
        }];
        let other = Ask {
            model: Some("deepseek-chat".into()),
            ..Default::default()
        };
        assert!(ledger.try_reserve("deepseek", &limits, &other).is_granted());
        let named = Ask {
            model: Some("deepseek-reasoner".into()),
            ..Default::default()
        };
        assert!(ledger.try_reserve("deepseek", &limits, &named).is_granted());
        assert!(!ledger.try_reserve("deepseek", &limits, &named).is_granted());
    }

    #[test]
    fn an_instance_without_limits_admits_everything() {
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let ask = Ask::default();
        let grant = ledger.try_reserve("nim", &[], &ask);
        assert!(grant.is_granted());
        match &grant {
            Reservation::Granted(g) => {
                assert!(g.takes.is_empty());
                assert_eq!(g.instance, "nim");
            }
            Reservation::Denied(_) => panic!("no limit, nothing to refuse"),
        }
        assert!(ledger.settle(grant, Outcome::NeverSent).is_empty());
        assert_eq!(ledger.standing("nim", &[], &ask), OurStanding::NoLimit);
        let report = ledger.report(&record(vec![]));
        assert_eq!(report.status, RateLimitStatus::Ok);
        assert!(report.limits.is_empty());
    }

    // ---- the bridge to RL3 ---------------------------------------------------

    #[test]
    fn standing_tells_the_classifier_where_we_are() {
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let limits = vec![requests(rolling(60), 10.0)];
        let ask = Ask::default();
        assert_eq!(ledger.standing("nim", &limits, &ask), OurStanding::Under);
        for _ in 0..8 {
            assert!(ledger.try_reserve("nim", &limits, &ask).is_granted());
        }
        assert_eq!(ledger.standing("nim", &limits, &ask), OurStanding::Near);
        for _ in 0..2 {
            assert!(ledger.try_reserve("nim", &limits, &ask).is_granted());
        }
        assert_eq!(
            ledger.standing("nim", &limits, &ask),
            OurStanding::AtOrAbove
        );
    }

    #[test]
    fn one_ceiling_reached_is_what_explains_a_refusal() {
        assert_eq!(
            worst(OurStanding::NoLimit, OurStanding::Under),
            OurStanding::Under
        );
        assert_eq!(
            worst(OurStanding::Under, OurStanding::NoLimit),
            OurStanding::Under
        );
        assert_eq!(
            worst(OurStanding::Unknown, OurStanding::NoLimit),
            OurStanding::Unknown
        );
        assert_eq!(
            worst(OurStanding::Near, OurStanding::Unknown),
            OurStanding::Near
        );
        assert_eq!(
            worst(OurStanding::AtOrAbove, OurStanding::Near),
            OurStanding::AtOrAbove
        );
        assert_eq!(
            worst(OurStanding::Near, OurStanding::AtOrAbove),
            OurStanding::AtOrAbove
        );
    }

    #[test]
    fn the_status_of_a_report_is_the_worst_one_of_its_limits() {
        use RateLimitStatus::{Limited, Ok, Overloaded, Unknown};
        assert_eq!(worse_status(Ok, Ok), Ok);
        assert_eq!(worse_status(Ok, Unknown), Unknown);
        assert_eq!(worse_status(Unknown, Ok), Unknown);
        assert_eq!(worse_status(Ok, Overloaded), Overloaded);
        assert_eq!(worse_status(Overloaded, Unknown), Overloaded);
        assert_eq!(worse_status(Ok, Limited), Limited);
        assert_eq!(worse_status(Limited, Overloaded), Limited);
    }

    // ---- the clock -----------------------------------------------------------

    #[test]
    fn the_clock_is_injectable_and_the_real_one_reads_the_wall_clock() {
        let clock = ManualClock::at(T0);
        assert_eq!(clock.now(), instant(T0));
        clock.advance(Duration::from_secs(30));
        assert_eq!(clock.now(), instant(T0) + chrono::Duration::seconds(30));
        let parsed = ManualClock::from_instant(instant(T0));
        assert_eq!(parsed.now(), instant(T0));
        let real = SystemClock.now();
        assert!(
            (Utc::now() - real).num_seconds().abs() < 5,
            "the real clock is now"
        );
    }

    // ---- the helpers, on their own ------------------------------------------

    #[test]
    fn a_window_has_a_length_in_seconds_for_the_counters_that_need_one() {
        assert_eq!(window_secs(unit(RateLimitUnit::Minute)), 60);
        assert_eq!(window_secs(unit(RateLimitUnit::Hour)), 3_600);
        assert_eq!(window_secs(unit(RateLimitUnit::Day)), 86_400);
        assert_eq!(window_secs(unit(RateLimitUnit::Week)), 604_800);
        assert_eq!(window_secs(unit(RateLimitUnit::Month)), 2_592_000);
        assert_eq!(window_secs(rolling(90)), 90);
    }

    #[test]
    fn only_a_day_a_week_or_a_month_is_a_calendar_window() {
        assert_eq!(
            CalendarUnit::of(RateLimitUnit::Day),
            Some(CalendarUnit::Day)
        );
        assert_eq!(
            CalendarUnit::of(RateLimitUnit::Week),
            Some(CalendarUnit::Week)
        );
        assert_eq!(
            CalendarUnit::of(RateLimitUnit::Month),
            Some(CalendarUnit::Month)
        );
        assert_eq!(CalendarUnit::of(RateLimitUnit::Minute), None);
        assert_eq!(CalendarUnit::of(RateLimitUnit::Hour), None);
    }

    #[test]
    fn a_limit_with_no_zone_or_an_unknown_one_reads_in_utc() {
        assert_eq!(tz_of(None), chrono_tz::UTC);
        assert_eq!(tz_of(Some("utc")), chrono_tz::UTC);
        assert_eq!(tz_of(Some(" Europe/Paris ")), chrono_tz::Europe::Paris);
        assert_eq!(tz_of(Some("Mars/Olympus")), chrono_tz::UTC);
    }

    #[test]
    fn a_denial_names_what_it_counts_and_never_a_value() {
        let ask = Ask::default();
        for (kind, word) in [
            (RateLimitKind::Requests, "requests"),
            (RateLimitKind::Tokens, "tokens"),
            (RateLimitKind::BudgetUsd, "USD"),
        ] {
            assert_eq!(kind_word(kind), word);
            let denial = Denial {
                kind,
                window: window_label(unit(RateLimitUnit::Hour)),
                scope: RateLimitScope::Credential(CredentialScope {
                    credential: "vault:nim".into(),
                })
                .label(),
                max: 20.0,
                retry_at: None,
            };
            let message = denial.message();
            assert!(message.contains(word), "{message}");
            assert!(message.contains("per hour"), "{message}");
            assert!(message.contains("vault:nim"), "{message}");
            assert!(!message.contains("sk-"), "{message}");
        }
        assert_eq!(window_label(rolling(60)), "rolling 60s");
        assert_eq!(window_label(unit(RateLimitUnit::Minute)), "minute");
        assert!(applies(&instance_scope(), &ask));
        assert!(!applies(
            &RateLimitScope::Credential(CredentialScope {
                credential: "vault:nim".into()
            }),
            &ask
        ));
        assert!(!applies(
            &RateLimitScope::Model(ModelScope { model: "m".into() }),
            &ask
        ));
    }

    #[test]
    fn a_key_is_derived_from_the_configuration_so_a_restart_finds_it_again() {
        let limit = requests(rolling(60), 5.0);
        assert_eq!(key_of("nim", &limit), key_of("nim", &limit));
        assert_ne!(key_of("nim", &limit), key_of("deepseek", &limit));
        assert_ne!(
            key_of("nim", &limit),
            key_of("nim", &requests(rolling(61), 5.0))
        );
        assert!(key_of("nim", &limit).starts_with("nim|"));
    }

    #[test]
    fn counting_an_unpriced_call_only_moves_a_calendar_cumul() {
        let now = instant(T0);
        let mut bucket = Counter::new(&requests(unit(RateLimitUnit::Minute), 10.0), now);
        assert_eq!(bucket.add_unpriced(now), 0);
        assert!(bucket.persisted().is_none());
        let mut window = Counter::new(&budget("utc", RateLimitUnit::Day, 10.0), now);
        assert_eq!(window.add_unpriced(now), 1);
        assert_eq!(window.add_unpriced(now), 2);
        assert!(window.persisted().is_some());
        // Restoring a counter that has nothing to restore is a no-op, not a panic.
        let document = serde_json::json!({"start": "2026-10-08T00:00:00Z"});
        bucket.restore(&document);
        window.restore(&document);
        assert_eq!(window.used(now), 0.0);
    }

    #[test]
    fn the_counter_of_a_window_is_the_algorithm_that_window_calls_for() {
        let now = instant(T0);
        assert!(matches!(
            Counter::new(&requests(rolling(60), 5.0), now),
            Counter::Sliding(_)
        ));
        assert!(matches!(
            Counter::new(&requests(unit(RateLimitUnit::Hour), 5.0), now),
            Counter::Bucket(_)
        ));
        assert!(matches!(
            Counter::new(&budget("utc", RateLimitUnit::Week, 5.0), now),
            Counter::Calendar(_)
        ));
    }

    #[test]
    fn a_settlement_moves_a_counter_in_both_directions() {
        let now = instant(T0);
        let mut counter = Counter::new(&requests(rolling(60), 10.0), now);
        counter.take_unchecked(now, 4.0);
        assert_eq!(counter.used(now), 4.0);
        adjust(&mut counter, now, 4.0, 7.0);
        assert_eq!(counter.used(now), 7.0, "an underestimate is added");
        adjust(&mut counter, now, 7.0, 2.0);
        assert_eq!(counter.used(now), 2.0, "an overestimate comes back");
        counter.release(now, 99.0);
        assert_eq!(counter.used(now), 0.0, "a release never goes below zero");
        assert_eq!(counter.max(), 10.0);
    }

    #[test]
    fn a_window_says_when_it_makes_room_again() {
        let now = instant(T0);
        let mut window = Counter::new(&requests(rolling(60), 2.0), now);
        assert_eq!(
            window.resets_at(now),
            None,
            "nothing taken, nothing to wait for"
        );
        window.take_unchecked(now, 1.0);
        assert_eq!(
            window.resets_at(now),
            Some(now + chrono::Duration::seconds(60))
        );
        let mut bucket = Counter::new(&requests(unit(RateLimitUnit::Minute), 60.0), now);
        assert_eq!(
            bucket.resets_at(now),
            None,
            "a full bucket waits for nothing"
        );
        bucket.take_unchecked(now, 30.0);
        assert_eq!(
            bucket.resets_at(now),
            Some(now + chrono::Duration::seconds(30))
        );
    }

    #[test]
    fn a_report_without_a_ledger_is_still_the_rl7_shape() {
        let limits = vec![requests(rolling(60), 5.0)];
        let plain = rate_limit_report(&record(limits.clone()));
        assert_eq!(plain.limits[0].used, None);
        let (_clock, ledger) = ledger(Arc::new(MemoryStore::new()));
        let measured = ledger.report(&record(limits));
        assert_eq!(measured.limits[0].used, Some(0.0));
        assert_eq!(measured.limits[0].remaining, Some(5.0));
        assert_eq!(measured.limits[0].since, None);
        assert_eq!(measured.limits[0].next_retry_at, None);
        assert_eq!(measured.status, RateLimitStatus::Ok);
    }
}
