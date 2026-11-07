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
//! * The retries INSIDE a turn (`play` re-sends through `send_turn` directly) do not
//!   pass the gate again: only the turn's final state is settled. One admission, one
//!   settlement, per turn.
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
    ask_for, from_provider_error, Admission, Ask, Gate, OurStanding, Outcome, Refusal, RefusalKind,
};
use super::provider::settings::{InstanceRecord, RateLimit};
use crate::neo4j::GraphStore;

/// How a turn ended, as far as the gate needs to know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ending<'a> {
    /// The provider answered, or refused, with this typed cause (ERR-1).
    /// `quota_refusal` is the session's own reading that the ACCOUNT refused.
    Answered {
        error_code: Option<&'a str>,
        quota_refusal: bool,
    },
    /// The person stopped it, or its task was aborted: no verdict on the provider.
    Interrupted,
    /// It never left (the send failed).
    NeverSent,
}

/// Settles the turn as interrupted when dropped unsettled.
pub(crate) struct TurnGuard<'a> {
    turns: &'a TurnAdmission,
    provider_id: &'a str,
    armed: bool,
}

impl TurnGuard<'_> {
    /// Settles the turn with `ending` and disarms the guard.
    pub(crate) fn settle(mut self, ending: Ending<'_>) {
        self.armed = false;
        self.turns.settle(self.provider_id, ending);
    }
}

impl Drop for TurnGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.turns.settle(self.provider_id, Ending::Interrupted);
        }
    }
}

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
        // A turn admitted earlier and never settled (its task died before the guard
        // could run) is settled FIRST — before the gate is asked again — or its probe
        // would still be out and this very admission refused.
        if let Some(stale) = self.lock().take() {
            self.settle_held(provider_id, stale, Ending::Interrupted);
        }
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

    /// Settles the turn admitted last, however it ended (see [`Ending`]).
    pub(crate) fn settle(&self, provider_id: &str, ending: Ending<'_>) {
        let Some(held) = self.lock().take() else {
            return;
        };
        self.settle_held(provider_id, held, ending);
    }

    fn settle_held(&self, provider_id: &str, held: Held, ending: Ending<'_>) {
        let consumed = Outcome::Consumed {
            tokens: None,
            cost_usd: None,
        };
        match ending {
            // A turn that never left gives its reservation back; one that was cut
            // short keeps its unit. NEITHER says anything about the provider, and a
            // probe among them is handed back, not counted as a recovery.
            Ending::NeverSent => {
                self.gate.settle_unread(
                    provider_id,
                    &held.limits,
                    &held.ask,
                    held.admission,
                    Outcome::NeverSent,
                );
            }
            Ending::Interrupted => {
                self.gate.settle_unread(
                    provider_id,
                    &held.limits,
                    &held.ask,
                    held.admission,
                    consumed,
                );
            }
            Ending::Answered {
                error_code,
                quota_refusal,
            } => {
                // A refusal of the ACCOUNT (a 429, a 402) is the quota registry's
                // business — the session marks it — and never an overload: it is
                // read as a spent quota, which the breaker ignores.
                let kind = if quota_refusal || error_code == Some("rate_limited") {
                    Some("rate_limited")
                } else {
                    error_code.filter(|code| *code == "overloaded")
                };
                let classification = kind.map(|kind| {
                    let ours = if kind == "rate_limited" {
                        OurStanding::AtOrAbove
                    } else {
                        self.gate.standing(provider_id, &held.limits, &held.ask)
                    };
                    from_provider_error(kind, None, ours, held.preset, self.gate.now())
                });
                self.gate.settle(
                    provider_id,
                    &held.limits,
                    &held.ask,
                    held.admission,
                    consumed,
                    classification.as_ref(),
                );
            }
        }
    }

    /// A guard for the turn in flight: if the task playing it is aborted (a
    /// watchdog, a closed session) before it settles, dropping the guard settles it
    /// as [`Ending::Interrupted`] — the probe is handed back instead of staying out.
    pub(crate) fn guard<'a>(&'a self, provider_id: &'a str) -> TurnGuard<'a> {
        TurnGuard {
            turns: self,
            provider_id,
            armed: true,
        }
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

    fn answered(error_code: Option<&'static str>, quota_refusal: bool) -> Ending<'static> {
        Ending::Answered {
            error_code,
            quota_refusal,
        }
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
        turns.settle("deepseek", Ending::NeverSent);
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
        turns.settle("deepseek", answered(None, false));
        assert!(turns.admit_record("deepseek", Some(&record)).is_err());
        // Settling with nothing held is a no-op.
        turns.settle("deepseek", answered(None, false));
    }

    #[test]
    fn an_overload_pauses_the_instance_and_the_refusal_says_so() {
        let (turns, _clock, _gate) = admission(None);
        let record = record(10.0);
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
        turns.settle("deepseek", answered(Some("overloaded"), false));
        let refused = turns.admit_record("deepseek", Some(&record)).unwrap_err();
        assert!(matches!(refused, ProviderError::Overloaded));
    }

    #[test]
    fn a_refusal_of_the_account_never_opens_the_overload_breaker() {
        // MUTATION: reading a quota refusal as an overload (kind forced to
        // "overloaded") opens the pause and the second admission is refused. The
        // account's refusal is the quota registry's, never the breaker's.
        let (turns, _clock, gate) = admission(None);
        let record = record(10.0);
        for quota_refusal in [true, false] {
            assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
            let code = if quota_refusal {
                None
            } else {
                Some("rate_limited")
            };
            turns.settle("deepseek", answered(code, quota_refusal));
            assert!(
                turns.admit_record("deepseek", Some(&record)).is_ok(),
                "a 429 (quota_refusal={quota_refusal}) is not an overload"
            );
            turns.settle("deepseek", Ending::NeverSent);
        }
        assert!(gate
            .breaker_state("deepseek", Some("vault:deepseek"))
            .is_none_or(|state| !state.is_open()));
    }

    /// An open pause whose time has passed: the next admission is the probe.
    fn paused_then_ready() -> (TurnAdmission, Arc<ManualClock>, Arc<Gate>, InstanceRecord) {
        let (turns, clock, gate) = admission(None);
        let record = record(10.0);
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
        turns.settle("deepseek", answered(Some("overloaded"), false));
        clock.advance(Duration::from_secs(11));
        (turns, clock, gate, record)
    }

    fn phase(gate: &Gate) -> crate::chat::provider::limits::Phase {
        gate.breaker_state("deepseek", Some("vault:deepseek"))
            .unwrap()
            .phase
    }

    #[test]
    fn a_probe_whose_send_failed_is_handed_back() {
        // MUTATION: not releasing the probe when nothing is recorded leaves the
        // breaker half-open: the second admission is refused as "a probe is already
        // in flight".
        let (turns, _clock, gate, record) = paused_then_ready();
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
        assert_eq!(phase(&gate), crate::chat::provider::limits::Phase::HalfOpen);
        turns.settle("deepseek", Ending::NeverSent);
        assert!(
            turns.admit_record("deepseek", Some(&record)).is_ok(),
            "the next turn gets the probe again"
        );
    }

    #[test]
    fn a_stopped_probe_is_neither_a_recovery_nor_a_failure() {
        // MUTATION: settling an interrupted probe as an answered turn CLOSES the
        // breaker without the provider having answered.
        use crate::chat::provider::limits::Phase;
        let (turns, _clock, gate, record) = paused_then_ready();
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
        turns.settle("deepseek", Ending::Interrupted);
        assert_eq!(phase(&gate), Phase::Open, "handed back, not closed");
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
    }

    #[test]
    fn a_probe_refused_for_the_account_is_handed_back() {
        // MUTATION: a quota verdict that releases nothing leaves the probe out.
        let (turns, _clock, _gate, record) = paused_then_ready();
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
        turns.settle("deepseek", answered(Some("rate_limited"), true));
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
    }

    #[test]
    fn a_dropped_guard_settles_the_turn_as_interrupted() {
        // MUTATION: a `Drop` that settles nothing (a task aborted mid-turn) leaves
        // the probe out for the whole timeout.
        let (turns, _clock, _gate, record) = paused_then_ready();
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
        let guard = turns.guard("deepseek");
        drop(guard);
        assert!(
            turns.lock().is_none(),
            "the drop settled the held admission"
        );
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
        // A guard that settled first does not settle twice.
        turns.guard("deepseek").settle(Ending::NeverSent);
        assert!(turns.lock().is_none());
    }

    #[test]
    fn a_turn_admitted_over_an_unsettled_one_settles_the_stale_one() {
        // MUTATION: overwriting the held admission without settling it leaks the
        // probe: the third admission is refused.
        let (turns, _clock, _gate, record) = paused_then_ready();
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok()); // the probe
                                                                        // Its task died; the session admits the next turn over it.
        assert!(turns.admit_record("deepseek", Some(&record)).is_ok());
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
        turns.settle("ghost", answered(None, false));
    }
}
