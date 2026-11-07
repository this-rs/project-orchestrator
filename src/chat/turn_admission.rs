//! The admission of ONE turn on a provider instance (RL5, on the send path).
//!
//! A turn of an agent session reserves at the [`Gate`] before it leaves, and settles
//! when it ends. This module is the glue between the session and the gate; the
//! decision itself is the gate's (`provider::limits::admission`).
//!
//! * The gate READS the quota registry and counts our own configured limits; it
//!   never writes quota state. A refusal is turned into the typed
//!   [`ProviderError`] the provider would have answered itself
//!   (`RateLimited` with the instant it ends, `Overloaded`), so every consumer of a
//!   send error — the fallback, the interface — already knows what to do with it.
//! * One instance at a time: this never chooses another provider and never turns a
//!   lone available one into "no candidate" — the refusal names the instance, the
//!   kind and the instant it ends.
//! * It fails OPEN on what it cannot read: an instance record that cannot be loaded
//!   means no configured limit, never a blocked turn.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::sync::{Arc, Mutex};

use nexus_claude::agent::ProviderError;

use super::provider::limits::{
    ask_for, from_provider_error, Admission, Ask, Gate, Outcome, Refusal, RefusalKind,
};
use super::provider::settings::{InstanceRecord, RateLimit};
use crate::neo4j::GraphStore;

/// What a turn holds between its admission and its settlement.
struct Held {
    limits: Vec<RateLimit>,
    ask: Ask,
    preset: &'static str,
    admission: Admission,
}

/// The gate of one session, and the admission its turn in flight holds.
pub(crate) struct TurnAdmission {
    gate: Arc<Gate>,
    held: Mutex<Option<Held>>,
}

impl std::fmt::Debug for TurnAdmission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurnAdmission").finish_non_exhaustive()
    }
}

impl TurnAdmission {
    pub(crate) fn new(gate: Arc<Gate>) -> Self {
        Self {
            gate,
            held: Mutex::new(None),
        }
    }

    /// Admits the next turn of `provider_id`, or answers the typed refusal.
    pub(crate) async fn admit(
        &self,
        graph: &dyn GraphStore,
        provider_id: &str,
    ) -> Result<(), ProviderError> {
        let record = super::provider::store::instance(graph, provider_id)
            .await
            .ok()
            .flatten();
        self.admit_record(provider_id, record.as_ref())
    }

    /// [`Self::admit`] once the record is read (`None`: no stored instance, no
    /// configured limit — the quota registry is still consulted).
    pub(crate) fn admit_record(
        &self,
        provider_id: &str,
        record: Option<&InstanceRecord>,
    ) -> Result<(), ProviderError> {
        let (limits, ask, preset) = match record {
            Some(record) => (
                record.rate_limit.clone(),
                ask_for(record, None),
                super::provider::limits::preset_for_base_url(&record.base_url),
            ),
            None => (Vec::new(), Ask::default(), "openai_compatible"),
        };
        let admission = self.gate.admit(provider_id, &limits, &ask);
        if let Some(refusal) = admission.refusal() {
            return Err(refusal_error(refusal, self.gate.now()));
        }
        *self.lock() = Some(Held {
            limits,
            ask,
            preset,
            admission,
        });
        Ok(())
    }

    /// Settles the turn admitted last: `error_code` is the typed cause the turn ended
    /// on (ERR-1), when it ended on one. A turn that never left gives its
    /// reservation back.
    pub(crate) fn settle(&self, provider_id: &str, left: bool, error_code: Option<&str>) {
        let Some(held) = self.lock().take() else {
            return;
        };
        let classification = error_code
            .filter(|code| matches!(*code, "rate_limited" | "overloaded"))
            .map(|code| {
                from_provider_error(
                    code,
                    None,
                    self.gate.standing(provider_id, &held.limits, &held.ask),
                    held.preset,
                    self.gate.now(),
                )
            });
        let outcome = if left {
            Outcome::Consumed {
                tokens: None,
                cost_usd: None,
            }
        } else {
            Outcome::NeverSent
        };
        self.gate.settle(
            provider_id,
            &held.limits,
            &held.ask,
            held.admission,
            outcome,
            classification.as_ref(),
        );
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Held>> {
        self.held.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The error a refusal stands for: the one the provider would have answered.
fn refusal_error(refusal: &Refusal, now: chrono::DateTime<chrono::Utc>) -> ProviderError {
    tracing::info!(
        instance = %refusal.instance,
        kind = refusal.kind.as_str(),
        scope = %refusal.scope,
        retry_at = ?refusal.retry_at_rfc3339(),
        "turn refused at the admission gate: {}",
        refusal.message
    );
    match refusal.kind {
        RefusalKind::Quota => ProviderError::RateLimited {
            retry_after_ms: refusal
                .retry_after_secs(now)
                .map(|secs| secs.saturating_mul(1000)),
            resets_at_ms: refusal
                .reset_at
                .and_then(|at| u64::try_from(at.timestamp_millis()).ok()),
        },
        RefusalKind::Overload => ProviderError::Overloaded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::limits::{ManualClock, MemoryStore, Policy};
    use crate::chat::provider::quota::QuotaRegistry;
    use crate::chat::provider::settings::{
        InstanceScope, RateLimitKind, RateLimitScope, RateLimitUnit, RateLimitWindow,
    };
    use std::time::Duration;

    fn requests(max: f64) -> RateLimit {
        RateLimit {
            kind: RateLimitKind::Requests,
            window: RateLimitWindow::Unit(RateLimitUnit::Minute),
            calendar: None,
            max,
            scope: RateLimitScope::Instance(InstanceScope::Instance),
        }
    }

    fn record(max: f64) -> InstanceRecord {
        InstanceRecord {
            id: "deepseek".into(),
            credential_ref: "vault:deepseek".into(),
            rate_limit: vec![requests(max)],
            ..InstanceRecord::default()
        }
    }

    fn admission(
        quota: Option<Arc<QuotaRegistry>>,
    ) -> (TurnAdmission, Arc<ManualClock>, Arc<Gate>) {
        let clock = Arc::new(ManualClock::at("2026-10-08T12:00:00Z"));
        let mut gate = Gate::new(
            clock.clone(),
            Arc::new(MemoryStore::new()),
            Policy {
                overload_threshold: 1,
                base_backoff: Duration::from_secs(10),
                jitter_ratio: 0.0,
                ..Policy::default()
            },
        );
        if let Some(quota) = quota {
            gate = gate.with_quota(quota);
        }
        let gate = Arc::new(gate);
        (TurnAdmission::new(gate.clone()), clock, gate)
    }

    #[test]
    fn a_configured_ceiling_refuses_the_next_turn_with_a_typed_error() {
        let (turns, _clock, _gate) = admission(None);
        let record = record(1.0);
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
        let refused = turns.admit_record("deepseek", Some(&record)).unwrap_err();
        match refused {
            ProviderError::RateLimited {
                retry_after_ms,
                resets_at_ms,
            } => {
                assert!(retry_after_ms.is_some_and(|ms| ms <= 60_000));
                assert!(resets_at_ms.is_some(), "the refusal says when it ends");
            }
            other => panic!("expected a rate limit, got {other:?}"),
        }
    }

    #[test]
    fn a_turn_that_never_left_gives_its_reservation_back() {
        let (turns, _clock, _gate) = admission(None);
        let record = record(1.0);
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
        turns.settle("deepseek", false, None);
        assert!(
            turns.admit_record("deepseek", Some(&record)).is_ok(),
            "the slot came back: the first turn was never sent"
        );
    }

    #[test]
    fn a_settled_turn_keeps_its_unit() {
        let (turns, _clock, _gate) = admission(None);
        let record = record(1.0);
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
        turns.settle("deepseek", true, None);
        assert!(turns.admit_record("deepseek", Some(&record)).is_err());
        // Settling with nothing held is a no-op.
        turns.settle("deepseek", true, None);
    }

    #[test]
    fn an_overload_pauses_the_instance_and_the_refusal_says_so() {
        let (turns, _clock, _gate) = admission(None);
        let record = record(10.0);
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
        turns.settle("deepseek", true, Some("overloaded"));
        let refused = turns.admit_record("deepseek", Some(&record)).unwrap_err();
        assert!(matches!(refused, ProviderError::Overloaded));
    }

    #[test]
    fn a_rate_limited_turn_under_our_ceiling_is_an_overload_not_a_quota() {
        let (turns, _clock, _gate) = admission(None);
        let record = record(10.0);
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
        turns.settle("deepseek", true, Some("rate_limited"));
        let refused = turns.admit_record("deepseek", Some(&record)).unwrap_err();
        assert!(
            matches!(refused, ProviderError::Overloaded),
            "no quota proof: the key is not exhausted"
        );
    }

    #[test]
    fn the_quota_registry_refuses_an_instance_with_no_record_and_never_blocks_another() {
        let clock = Arc::new(ManualClock::at("2026-10-08T12:00:00Z"));
        let quota = Arc::new(QuotaRegistry::new().with_clock(clock.clone()));
        let (turns, _c, _g) = {
            let gate = Arc::new(
                Gate::new(
                    clock.clone(),
                    Arc::new(MemoryStore::new()),
                    Policy::default(),
                )
                .with_quota(quota.clone()),
            );
            (TurnAdmission::new(gate.clone()), clock.clone(), gate)
        };
        quota.mark_rate_limited("claude_code", Some(quota.now_ms() + 600_000), None);
        match turns.admit_record("claude_code", None).unwrap_err() {
            ProviderError::RateLimited { resets_at_ms, .. } => {
                assert_eq!(resets_at_ms, Some(quota.now_ms() + 600_000));
            }
            other => panic!("expected a rate limit, got {other:?}"),
        }
        // A lone other provider is untouched by that hold.
        assert!(turns.admit_record("local", None).is_ok());
        // And the gate did not write the registry.
        assert!(quota.exhausted("local").is_none());
    }

    #[tokio::test]
    async fn the_record_is_read_from_the_graph_and_a_missing_one_fails_open() {
        use crate::neo4j::mock::MockGraphStore;
        let graph = MockGraphStore::new();
        let (turns, _clock, _gate) = admission(None);
        // No stored instance: no configured limit, the turn goes.
        assert!(turns.admit(&graph, "ghost").await.is_ok());
        turns.settle("ghost", true, None);
    }
}
