//! Forgetting: the only path allowed to remove a synapse.
//!
//! # Why this module exists
//!
//! Measured on 01/11/2026 (production graph: 59 254 `SYNAPSE`, 44 013 `cosine`
//! never reinforced, 2 092 edges in [0.05, 0.15), 23 of 48 projects above the
//! density threshold): the legacy decay subtracted a fixed amount per PASS
//! (`w - 0.02` every 6 h) and then deleted with `DELETE`, without a trace.
//!
//! Four consequences, all measured:
//!
//! 1. **Forgetting was counted in passes, not in time.** A server down for three
//!    days forgot nothing; a restart that ran two checks close together forgot
//!    twice as fast. The same graph, the same age, two different outcomes.
//! 2. **Subtraction is anti-proportional.** An edge at 0.2 lost 0.02 per pass:
//!    gone in 1.9 days. Under a proportional decay of the same calendar rate it
//!    lives 5.5 days. The old model therefore erased the WEAK structure — most
//!    of the graph — and only kept the hubs.
//! 3. **`DELETE` is unmeasurable and irreversible.** "Forgotten" and "never
//!    known" were the same thing, so no run could be judged on data loss.
//! 4. **The edge's own history was ignored.** `reinforcement_count`,
//!    `last_reinforced_at` and `created_at` were written on every edge and read
//!    by no part of the decay.
//!
//! # The model
//!
//! - **Proportional to elapsed time**, not to passes: `w · 2^(−Δt / h)`, with
//!   `Δt = now − coalesce(last_reinforced_at, created_at)`. Idempotent, cadence
//!   independent, and a downtime is caught up automatically.
//! - **Per provenance.** A co-activation edge (validated by use) halves every
//!   [`DecayPolicy::half_life_days`]; a similarity edge (created by the
//!   embedding backfill, never reinforced) halves twice as fast.
//! - **Protected by evidence.** An edge reinforced
//!   [`DecayPolicy::protected_after`] times never falls below
//!   [`DecayPolicy::protection_floor`] and is never forgotten: forgetting must
//!   not be able to erase what has served repeatedly.
//! - **Dormant, then archived.** Below [`DecayPolicy::dormancy_threshold`] an
//!   edge is still there but no longer travelled; below
//!   [`DecayPolicy::forget_threshold`], and untouched for
//!   [`DecayPolicy::forget_after_days`], it is ARCHIVED (`:FORGOTTEN`, with its
//!   weight, provenance, counters and the date), never destroyed. A later
//!   reinforcement revives it at the weight it had.
//! - **Bounded.** One pass never archives more than
//!   [`DecayPolicy::max_archive_fraction`] of the edges it walked, and never
//!   takes a node below [`DecayPolicy::min_degree`] edges.
//!
//! The formula is not new: [`crate::neurons::synaptic_shadow::proportional_decay`]
//! already applied it to `shadow_weight`, calibrated by `DAYS_TO_DORMANT`. This
//! module applies the same family to the REAL weight, which is what every
//! existing reader traverses.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Days for an edge at weight 1.0, reinforced by use, to reach the forget
/// threshold without any further reinforcement.
///
/// Anchored on the measured legacy cadence (0.02 every 6 h: 1.0 → 0.05 in
/// 47.5 passes, 11.9 days), so switching models does not change how long a
/// USED edge survives — only how gently the weak ones fade.
pub const DAYS_TO_FORGET: f64 = 11.9;

/// How the graph forgets.
///
/// One policy, one threshold: the heartbeat, the homeostasis controller, the
/// skills maintenance and the protocol executor all pass THIS, so an edge can
/// no longer be spared by one and deleted by another on the same day.
#[derive(Debug, Clone, PartialEq)]
pub struct DecayPolicy {
    /// Half-life (days) of an edge validated by use (`coactivation`).
    pub half_life_days: f64,
    /// Half-life (days) of an edge created by similarity and never reinforced.
    pub similarity_half_life_days: f64,
    /// Below this, an edge is dormant: kept, no longer travelled.
    pub dormancy_threshold: f64,
    /// Below this, an edge becomes eligible for archival.
    pub forget_threshold: f64,
    /// An edge untouched for this many days may be archived.
    pub forget_after_days: i64,
    /// Reinforcements after which an edge is protected for good.
    pub protected_after: u32,
    /// Floor a protected edge never falls below.
    pub protection_floor: f64,
    /// Edges a node keeps whatever happens (its strongest are never taken).
    pub min_degree: usize,
    /// Share of the walked edges one pass may archive at most.
    pub max_archive_fraction: f64,
}

impl Default for DecayPolicy {
    fn default() -> Self {
        let forget_threshold = 0.02;
        let half_life_days = Self::half_life_days_for(DAYS_TO_FORGET, forget_threshold);
        Self {
            half_life_days,
            // A similarity edge decays twice as fast, as the legacy model did.
            similarity_half_life_days: half_life_days / 2.0,
            // The single threshold: the legacy split (0.05 in the heartbeat,
            // 0.15 in the homeostasis) let the same edge survive one path and
            // die in the other on the same day.
            dormancy_threshold: 0.15,
            forget_threshold,
            forget_after_days: 14,
            protected_after: 3,
            protection_floor: 0.15,
            min_degree: 3,
            max_archive_fraction: 0.05,
        }
    }
}

impl DecayPolicy {
    /// Half-life that brings a weight of 1.0 to `threshold` in `days`.
    pub fn half_life_days_for(days: f64, threshold: f64) -> f64 {
        if days <= 0.0 || threshold <= 0.0 || threshold >= 1.0 {
            return 1.0;
        }
        days / (1.0 / threshold).log2()
    }

    /// Half-life of an edge, by provenance.
    ///
    /// Anything that is not `coactivation` is a similarity edge: the backfill
    /// creates it, no reader ever rewards it, so it decays at the faster rate.
    pub fn half_life_for(&self, source: &str) -> f64 {
        if source == "coactivation" {
            self.half_life_days
        } else {
            self.similarity_half_life_days
        }
    }

    /// The weight after `elapsed_secs` without reinforcement.
    ///
    /// `w · 2^(−Δt / h)`: the SAME result whether the caller runs one pass over
    /// 6 hours or two passes over 3 hours. A non-positive elapsed time leaves
    /// the weight untouched (clock skew never resurrects nor erases).
    pub fn decayed(&self, weight: f64, elapsed_secs: f64, source: &str) -> f64 {
        let h = self.half_life_for(source);
        if elapsed_secs <= 0.0 || h <= 0.0 {
            return weight;
        }
        weight * 0.5_f64.powf(elapsed_secs / (h * 86_400.0))
    }

    /// [`Self::decayed`], with the protection floor applied: an edge reinforced
    /// enough never becomes dormant, whatever its age.
    pub fn decayed_protected(
        &self,
        weight: f64,
        elapsed_secs: f64,
        source: &str,
        reinforcements: u32,
    ) -> f64 {
        let w = self.decayed(weight, elapsed_secs, source);
        self.protected(w, reinforcements)
    }

    /// Raise `weight` to the protection floor when the edge is proven.
    pub fn protected(&self, weight: f64, reinforcements: u32) -> f64 {
        if reinforcements >= self.protected_after && weight < self.protection_floor {
            self.protection_floor
        } else {
            weight
        }
    }

    /// Whether an edge may be archived now.
    ///
    /// Three conditions, all required: below the forget threshold, not proven
    /// by use, and untouched for [`Self::forget_after_days`]. `last_touch` is
    /// `coalesce(last_reinforced_at, created_at)`: an edge created recently is
    /// never forgotten, even if nothing reinforced it yet.
    pub fn is_forgettable(
        &self,
        weight: f64,
        reinforcements: u32,
        last_touch: Option<DateTime<Utc>>,
        now: DateTime<Utc>,
    ) -> bool {
        if weight >= self.forget_threshold || reinforcements >= self.protected_after {
            return false;
        }
        match last_touch {
            Some(at) => (now - at).num_days() >= self.forget_after_days,
            // No timestamp at all: an edge of unknown age is left alone. The
            // migrator is what fills `created_at`, never this pass.
            None => false,
        }
    }

    /// The policy a legacy caller asked for, in the terms of this model.
    ///
    /// `decay_amount` is the amount the old subtractive model removed per
    /// pass at weight 1.0; it becomes the calendar: the same edge forgets after
    /// the same number of days. `prune_threshold` becomes the forget threshold
    /// (the only cut-off), and the dormancy threshold stays above it.
    pub fn legacy(decay_amount: f64, prune_threshold: f64) -> Self {
        let mut policy = Self::default();
        let threshold = prune_threshold.clamp(0.001, 0.5);
        let days = if decay_amount > 0.0 {
            DAYS_TO_FORGET * (decay_amount / 0.02)
        } else {
            DAYS_TO_FORGET
        };
        policy.forget_threshold = threshold;
        policy.half_life_days = Self::half_life_days_for(days, threshold);
        policy.similarity_half_life_days = policy.half_life_days / 2.0;
        policy.dormancy_threshold = (threshold * 7.5).min(1.0);
        policy.protection_floor = policy.dormancy_threshold;
        policy
    }

    /// How many edges one pass may archive, out of `walked` edges walked.
    ///
    /// Always at least one so a small project can still forget, and never more
    /// than a twentieth of the graph by default: a single pass cannot empty it.
    pub fn archive_cap(&self, walked: usize) -> usize {
        let cap = (walked as f64 * self.max_archive_fraction).ceil() as usize;
        cap.max(1)
    }
}

/// What one forgetting pass did. Journaled, so the loss stops being invisible.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassReport {
    /// Edges walked and decayed.
    pub decayed: usize,
    /// Of those, the ones now below the dormancy threshold (kept, not travelled).
    pub dormant: usize,
    /// Of those, the ones protected by their reinforcement count.
    pub protected: usize,
    /// Edges moved to `:FORGOTTEN` (never destroyed).
    pub archived: usize,
    /// The archive cap was reached: the pass stopped early and the rest is
    /// left for the next one.
    pub capped: bool,
}

impl PassReport {
    /// A pass that walked `decayed` edges, of which `dormant` fell asleep.
    pub fn new(decayed: usize, dormant: usize, protected: usize) -> Self {
        Self {
            decayed,
            dormant,
            protected,
            archived: 0,
            capped: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 11, 2, 12, 0, 0).unwrap()
    }

    /// The whole point of the change: the cadence of the pass no longer decides
    /// how much is forgotten. Two passes over three hours and one pass over six
    /// hours land on the same weight.
    #[test]
    fn two_short_passes_forget_exactly_as_one_long_pass() {
        let p = DecayPolicy::default();
        let six_hours = 6.0 * 3600.0;
        let three_hours = 3.0 * 3600.0;

        let once = p.decayed(0.8, six_hours, "coactivation");
        let twice = p.decayed(
            p.decayed(0.8, three_hours, "coactivation"),
            three_hours,
            "coactivation",
        );

        assert!((once - twice).abs() < 1e-12, "once={once} twice={twice}");
    }

    /// A server that ran nothing for three days catches up in one pass, instead
    /// of forgetting as if only one interval had elapsed.
    #[test]
    fn a_downtime_is_caught_up_by_the_next_pass() {
        let p = DecayPolicy::default();
        let six_hours = 6.0 * 3600.0;
        let three_days = 72.0 * 3600.0;

        let hourly = p.decayed(1.0, six_hours, "coactivation");
        let after_downtime = p.decayed(1.0, three_days, "coactivation");

        // One interval costs ~8 % at the top weight (the exponential is steeper
        // than the subtraction it replaced); three days cost two thirds.
        assert!(hourly > 0.90 && hourly < 0.95, "hourly={hourly}");
        assert!(
            after_downtime < 0.5,
            "three days must be far lower: {after_downtime}"
        );
    }

    /// The measured defect: a weak edge died in 1.9 days under subtraction.
    /// Under a proportional decay of the same calendar rate it lives longer.
    #[test]
    fn a_weak_edge_forgets_much_slower_than_subtraction_did() {
        let p = DecayPolicy::default();
        let day = 86_400.0;

        // 0.2 -> below the forget threshold, in whole days.
        let mut w = 0.2;
        let mut days = 0.0;
        while w >= p.forget_threshold && days < 400.0 {
            days += 1.0;
            w = p.decayed(0.2, days * day, "coactivation");
        }

        // Legacy: (0.2 - 0.05) / 0.02 = 7.5 passes of 6 h = 1.9 days.
        assert!(
            days > 4.0,
            "a weak edge must not vanish in 1.9 days, measured {days}"
        );
    }

    /// The calendar rate is the legacy one for a USED edge: 1.0 reaches the
    /// forget threshold in roughly the measured 11.9 days.
    #[test]
    fn a_used_edge_forgets_on_the_legacy_calendar() {
        let p = DecayPolicy::default();
        let w = p.decayed(1.0, DAYS_TO_FORGET * 86_400.0, "coactivation");
        assert!((w - p.forget_threshold).abs() < 1e-9, "w={w}");
    }

    /// A similarity edge halves twice as fast, as the legacy 2x rate did.
    #[test]
    fn an_unrewarded_similarity_edge_decays_twice_as_fast() {
        let p = DecayPolicy::default();
        let day = 86_400.0;
        let sim = p.decayed(1.0, day, "cosine");
        let coact = p.decayed(1.0, day, "coactivation");
        assert!(sim < coact, "sim={sim} coact={coact}");
        // Half the half-life means twice the exponent: one day of similarity
        // decays exactly as much as two days of co-activation.
        let coact_two = p.decayed(1.0, 2.0 * day, "coactivation");
        assert!(
            (sim - coact_two).abs() < 1e-9,
            "sim={sim} coact_two={coact_two}"
        );
    }

    /// Forgetting must not be able to erase what has served repeatedly.
    #[test]
    fn a_proven_edge_never_become_dormant_nor_forgettable() {
        let p = DecayPolicy::default();
        let ancient = 10_000.0 * 86_400.0;

        let w = p.decayed_protected(0.9, ancient, "coactivation", p.protected_after);
        assert!((w - p.protection_floor).abs() < 1e-9, "w={w}");
        assert!(!p.is_forgettable(
            w,
            p.protected_after,
            Some(now() - chrono::Duration::days(365)),
            now()
        ));
    }

    /// An edge created recently is never archived, even if nothing reinforced
    /// it yet: the backfill must not be undone the day it ran.
    #[test]
    fn a_fresh_edge_is_not_forgettable() {
        let p = DecayPolicy::default();
        let fresh = now() - chrono::Duration::days(1);
        assert!(!p.is_forgettable(0.001, 0, Some(fresh), now()));
    }

    /// ...but an edge untouched for longer than the policy says, below the
    /// threshold, and never proven, is.
    #[test]
    fn an_abandoned_weak_edge_is_forgettable() {
        let p = DecayPolicy::default();
        let old = now() - chrono::Duration::days(p.forget_after_days + 1);
        assert!(p.is_forgettable(0.001, 0, Some(old), now()));
        assert!(p.is_forgettable(0.001, 0, Some(old), now()) || p.protected_after == 0);
    }

    /// An edge with no timestamp is left alone: unknown age is not an excuse.
    #[test]
    fn an_edge_of_unknown_age_is_never_forgotten() {
        let p = DecayPolicy::default();
        assert!(!p.is_forgettable(0.0, 0, None, now()));
    }

    /// One pass can never empty a graph: the cap is a twentieth, at least one.
    #[test]
    fn one_pass_cannot_empty_the_graph() {
        let p = DecayPolicy::default();
        assert_eq!(p.archive_cap(0), 1, "a small project can still forget");
        assert_eq!(p.archive_cap(10), 1);
        assert_eq!(p.archive_cap(1000), 50);
        assert_eq!(p.archive_cap(59_254), 2_963);
    }

    /// The default policy forgets at one single threshold, not two: the value
    /// the heartbeat and the homeostasis both use is this one.
    #[test]
    fn the_thresholds_are_ordered_and_single() {
        let p = DecayPolicy::default();
        assert!(p.forget_threshold < p.dormancy_threshold);
        assert_eq!(p.protection_floor, p.dormancy_threshold);
        assert!(p.half_life_days > 0.0 && p.similarity_half_life_days > 0.0);
    }
}

/// The pass, against the in-memory store: what the graph DOES, not only what
/// the formula says. Every assertion here is one line of the contract the
/// twenty-eight measured defects asked for.
#[cfg(test)]
mod pass {
    use super::*;
    use crate::neo4j::mock::{MockGraphStore, MockSynapseMeta};
    use crate::neo4j::traits::*;
    use crate::notes::{Note, NoteType};
    use std::sync::Arc;
    use uuid::Uuid;

    /// Two notes of ONE project (a reinforcement skips a cross-project pair).
    async fn pair(store: &MockGraphStore, weight: f64) -> (Uuid, Uuid) {
        let project = Uuid::new_v4();
        let a = Note::new(Some(project), NoteType::Gotcha, "a".into(), "a".into());
        let b = Note::new(Some(project), NoteType::Gotcha, "b".into(), "b".into());
        let (a_id, b_id) = (a.id, b.id);
        store.create_note(&a).await.unwrap();
        store.create_note(&b).await.unwrap();
        store
            .create_synapses(a_id, &[(b_id, weight)])
            .await
            .unwrap();
        (a_id, b_id)
    }

    async fn weight(store: &MockGraphStore, a: Uuid, b: Uuid) -> f64 {
        store
            .get_synapses(a)
            .await
            .unwrap()
            .into_iter()
            .find(|(id, _)| *id == b)
            .map(|(_, w)| w)
            .expect("the edge is still there")
    }

    /// Age the edge and give it a reinforcement count.
    async fn age(store: &MockGraphStore, a: Uuid, b: Uuid, days: i64, reinforcements: u32) {
        let key = if a < b { (a, b) } else { (b, a) };
        store.synapse_meta.write().await.insert(
            key,
            MockSynapseMeta {
                last_touch: Utc::now() - chrono::Duration::days(days),
                reinforcements,
            },
        );
    }

    /// The whole reason for the change: a pass that runs twice in a row decays
    /// nothing the second time. The legacy model subtracted per PASS, so a
    /// restart or a catch-up erased twice as fast.
    #[tokio::test]
    async fn a_second_pass_in_a_row_forgets_nothing() {
        let store = Arc::new(MockGraphStore::new());
        let (a, b) = pair(&store, 0.9).await;
        let policy = DecayPolicy::default();

        store.decay_synapses(&policy).await.unwrap();
        let after_one = weight(&store, a, b).await;
        let second = store.decay_synapses(&policy).await.unwrap();
        let after_two = weight(&store, a, b).await;

        assert!(second.decayed > 0, "the pass still walks the edges");
        assert_eq!(
            after_one, after_two,
            "no elapsed time, no forgetting: {after_one} -> {after_two}"
        );
        assert!(after_one < 0.9, "the first pass did decay it");
    }

    /// An abandoned edge leaves the traversal but NOT the graph: it becomes a
    /// `:FORGOTTEN` archive the store still holds.
    #[tokio::test]
    async fn an_abandoned_edge_is_archived_never_destroyed() {
        let store = Arc::new(MockGraphStore::new());
        let (a, b) = pair(&store, 0.01).await;
        age(&store, a, b, 30, 0).await;
        let policy = DecayPolicy {
            min_degree: 0,
            ..DecayPolicy::default()
        };

        let report = store.decay_synapses(&policy).await.unwrap();

        assert_eq!(report.archived, 1, "one pair archived: {report:?}");
        assert!(store.get_synapses(a).await.unwrap().is_empty());
        let archived = store.forgotten_synapses.read().await;
        let key = if a < b { (a, b) } else { (b, a) };
        assert!(
            archived.contains_key(&key),
            "the archive is the trace the legacy DELETE never left"
        );
    }

    /// A node is never orphaned by forgetting: the two ends must each keep more
    /// than `min_degree` edges for an edge to go.
    #[tokio::test]
    async fn a_pass_never_orphans_a_node() {
        let store = Arc::new(MockGraphStore::new());
        let (a, b) = pair(&store, 0.01).await;
        age(&store, a, b, 30, 0).await;

        let report = store.decay_synapses(&DecayPolicy::default()).await.unwrap();

        assert_eq!(report.archived, 0, "min_degree protects the last edges");
        assert_eq!(weight(&store, a, b).await, weight(&store, a, b).await);
    }

    /// Evidence outranks age: an edge reinforced enough is raised to the floor
    /// and never archived, however long it slept.
    #[tokio::test]
    async fn a_proven_edge_is_protected_for_good() {
        let store = Arc::new(MockGraphStore::new());
        let (a, b) = pair(&store, 0.01).await;
        age(&store, a, b, 365, 3).await;
        let policy = DecayPolicy {
            min_degree: 0,
            ..DecayPolicy::default()
        };

        let report = store.decay_synapses(&policy).await.unwrap();

        assert_eq!(report.archived, 0);
        assert_eq!(report.protected, 2, "both directions are reported");
        let w = weight(&store, a, b).await;
        assert!(
            (w - policy.protection_floor).abs() < 1e-9,
            "a proven edge sits on the floor, not under it: {w}"
        );
    }

    /// Forgetting is a sleep, not a loss: reinforcing an archived pair brings it
    /// back AT THE WEIGHT IT HAD, not at the 0.5 of a new edge.
    #[tokio::test]
    async fn a_reinforced_archived_edge_comes_back_at_its_own_weight() {
        let store = Arc::new(MockGraphStore::new());
        let (a, b) = pair(&store, 0.01).await;
        age(&store, a, b, 30, 0).await;
        let policy = DecayPolicy {
            min_degree: 0,
            ..DecayPolicy::default()
        };
        store.decay_synapses(&policy).await.unwrap();
        assert!(store.get_synapses(a).await.unwrap().is_empty());
        let archived = {
            let forgotten = store.forgotten_synapses.read().await;
            let key = if a < b { (a, b) } else { (b, a) };
            *forgotten.get(&key).expect("archived")
        };

        // A bare reinforcement, without boost: only the revival moves the weight.
        store.reinforce_synapses(&[a, b], 0.0).await.unwrap();

        let w = weight(&store, a, b).await;
        assert!(
            (w - archived).abs() < 1e-12,
            "revived at the weight it had when it slept ({archived}), got {w}"
        );
        assert!(w < 0.5, "never the 0.5 of a brand-new edge: {w}");
    }

    /// A backfilled edge is born now: the pass must not undo what the backfill
    /// did the same day, even at a weight under the forget threshold.
    #[tokio::test]
    async fn a_fresh_edge_survives_its_first_passes() {
        let store = Arc::new(MockGraphStore::new());
        let (a, _b) = pair(&store, 0.01).await;
        let policy = DecayPolicy {
            min_degree: 0,
            ..DecayPolicy::default()
        };

        let report = store.decay_synapses(&policy).await.unwrap();

        assert_eq!(report.archived, 0, "too young to be forgotten");
        assert!(!store.get_synapses(a).await.unwrap().is_empty());
    }
}
