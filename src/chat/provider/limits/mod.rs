//! Optional per-instance rate limits: the COUNTERS (RL2) and the CLASSIFIER (RL3).
//!
//! RL1 settled the configuration (`rate_limit` on an instance) and RL7 the shape
//! of `GET /chat/providers/{id}/limits`. This module is what makes the numbers in
//! that shape true, and what tells an overload from an exhausted quota.
//!
//! * [`ledger`] — RL2: rolling windows, token buckets and calendar cumuls, an
//!   injectable clock, an atomic reservation taken BEFORE a call goes out, and the
//!   settlement that follows it. It answers "how much is left, and may this go
//!   out right now";
//! * [`classify`] — RL3: the pure reading of a provider's refusal (status,
//!   headers, body code) against our own counting. It answers "is this an
//!   exhausted quota or an overload";
//! * [`profiles`] — the DATA of RL3: which headers and codes each endpoint family
//!   uses, and how solid that knowledge is;
//! * [`store`] — where a budget's calendar cumul survives a restart.
//!
//! # What is NOT here
//!
//! Nothing in this module talks to a provider, opens a session, emits an event or
//! reads a secret: it is the decision, kept pure and testable (see the coverage
//! gate on the pure files). The call site that reserves before a turn and settles
//! after it belongs to the chat path, and lands with the slice that wires it —
//! this module exposes exactly what that slice needs ([`Ledger::try_reserve`],
//! [`Ledger::settle`], [`Ledger::standing`]) without touching it today.
use crate::neo4j::traits::*;

pub mod classify;
pub mod ledger;
pub mod profiles;
pub mod store;

pub use classify::{
    apply_verdict, classify, from_provider_error, Classification, ErrorBody, OurStanding, Signal,
    Signals, Verdict, NEAR_RATIO,
};
pub use ledger::{Ask, Clock, Ledger, ManualClock, Note, Outcome, Reservation, SystemClock};
pub use profiles::{preset_for_base_url, profile, Evidence, ProviderProfile};
pub use store::{FileStore, LedgerStore, MemoryStore};

use super::settings::{rate_limit_report, InstanceRecord, RateLimitReport};

/// The state of an instance's limits: RL1's shape, filled by RL2's counters.
///
/// Without a ledger the RL7 answer is kept exactly as it was (`unknown`, null
/// numbers): this function is additive, so the route can be moved over to the
/// measured state without changing a key of the wire shape.
pub fn report(record: &InstanceRecord, ledger: Option<&Ledger>) -> RateLimitReport {
    match ledger {
        Some(ledger) => ledger.report(record),
        None => rate_limit_report(record),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::settings::{
        InstanceRecord, RateLimitScope, RateLimitStatus, RateLimitUnit, RateLimitWindow,
    };
    use crate::neo4j::traits::*;

    fn record(limits: Vec<crate::chat::provider::settings::RateLimit>) -> InstanceRecord {
        InstanceRecord {
            id: "deepseek".into(),
            rate_limit: limits,
            ..Default::default()
        }
    }

    fn one_limit() -> crate::chat::provider::settings::RateLimit {
        use crate::chat::provider::settings::{RateLimit, RateLimitKind};
        RateLimit {
            kind: RateLimitKind::Requests,
            window: RateLimitWindow::Unit(RateLimitUnit::Minute),
            calendar: None,
            max: 2.0,
            scope: RateLimitScope::Instance(
                crate::chat::provider::settings::InstanceScope::Instance,
            ),
        }
    }

    #[test]
    fn without_a_ledger_the_shape_is_the_one_rl7_settled() {
        let report = report(&record(vec![one_limit()]), None);
        assert_eq!(report.status, RateLimitStatus::Unknown);
        assert_eq!(report.limits.len(), 1);
        assert_eq!(report.limits[0].used, None);
        assert_eq!(report.limits[0].resets_at, None);
    }

    #[test]
    fn with_a_ledger_the_same_shape_carries_the_measured_numbers() {
        let ledger = Ledger::new(
            std::sync::Arc::new(ManualClock::at("2026-10-08T12:00:00Z")),
            std::sync::Arc::new(MemoryStore::new()),
        );
        let limits = vec![one_limit()];
        let ask = Ask::default();
        assert!(ledger.try_reserve("deepseek", &limits, &ask).is_granted());
        let report = report(&record(limits), Some(&ledger));
        assert_eq!(report.status, RateLimitStatus::Ok);
        assert_eq!(report.limits[0].used, Some(1.0));
        assert_eq!(report.limits[0].remaining, Some(1.0));
        assert!(report.limits[0].resets_at.is_some());
    }
}
