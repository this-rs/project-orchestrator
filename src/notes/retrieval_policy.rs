//! Retrieval policy — OUT of the production path.
//!
//! This module READS the ledger ([`super::retrieval_ledger`]) and renders a
//! dated assessment against pre-registered criteria. It sets **nothing**: it
//! writes no ledger entry, changes no index, confidence, routing or budget
//! parameter, and exposes no setter. Two tests hold that line: one reads this
//! file's own text, the other compares the ledger counters before and after a
//! report.
//!
//! Why the separation: a policy that tuned production from its own outputs
//! would confirm itself. Here the output is a report a human reads; the tuning,
//! if it ever happens, is another PR on an isolated instance.
//!
//! The criteria are **pre-registered** ([`CRITERIA`], dated): they were written
//! before the first measurement, so the threshold cannot be chosen after seeing
//! the number.
//!
//! Two scopes, and the report says which is which:
//! - the counters (`Snapshot`) cover everything since the process started: they
//!   feed the hit rate and the cost;
//! - the bounded journal (`WINDOW` events) is a recent sample: it feeds the
//!   per-trigger comparison, the timing, the calibration and the drift. Its
//!   [`Window::evicted`] says how much of it was dropped — a non-zero value
//!   means "recent", never "complete".

use super::retrieval_ledger::{RetrievalEvent, Snapshot, Trigger};

/// How many events of the bounded journal are read.
pub const WINDOW: usize = 256;

/// Smallest arm worth comparing: below this, a rate is an anecdote.
pub const MIN_ARM: usize = 10;

/// Smallest number of (score, outcome) pairs worth a calibration.
pub const MIN_PAIRS: usize = 10;

/// When the criteria below were written down, before the first measurement.
pub const REGISTERED_ON: &str = "2026-11-01";

/// The six criteria, pre-registered. Each is a number, not a wish: a measure
/// either beats it, misses it, or cannot be computed — and "cannot be computed"
/// is published as such, never as a pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Criteria {
    /// Share of settled retrievals that had at least one item used.
    pub useful_hit_rate_min: f64,
    /// Median delay, in ms, from a retrieval the agent was BLOCKED on to the
    /// next retrieval of the same session that served something.
    pub time_to_correction_median_ms_max: f64,
    /// Gap, in hit rate, against the fixed reference arm (explicit requests).
    pub regret_max: f64,
    /// Brier score of the index score read as a claimed probability of use.
    pub brier_max: f64,
    /// Drift between the two halves of the sample that counts as a change.
    pub change_delta: f64,
    /// Index calls per retrieval that served something.
    pub cost_per_useful_max: f64,
}

/// The registered values. Changing one of them is a decision, not an edit.
pub const CRITERIA: Criteria = Criteria {
    useful_hit_rate_min: 0.30,
    time_to_correction_median_ms_max: 60_000.0,
    regret_max: 0.10,
    brier_max: 0.25,
    change_delta: 0.15,
    cost_per_useful_max: 4.0,
};

/// The outcome of one criterion.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Met {
        measured: f64,
        threshold: f64,
    },
    Missed {
        measured: f64,
        threshold: f64,
    },
    /// No denominator, no pairs, no blocked event: the measure does not exist
    /// yet. It is NOT a pass, and NOT a zero.
    NotMeasurable {
        reason: &'static str,
    },
}

impl Verdict {
    /// For criteria where a bigger number is better.
    fn at_least(measured: f64, threshold: f64) -> Self {
        if measured >= threshold {
            Verdict::Met {
                measured,
                threshold,
            }
        } else {
            Verdict::Missed {
                measured,
                threshold,
            }
        }
    }

    /// For criteria where a smaller number is better.
    fn at_most(measured: f64, threshold: f64) -> Self {
        if measured <= threshold {
            Verdict::Met {
                measured,
                threshold,
            }
        } else {
            Verdict::Missed {
                measured,
                threshold,
            }
        }
    }

    pub fn measured(&self) -> Option<f64> {
        match self {
            Verdict::Met { measured, .. } | Verdict::Missed { measured, .. } => Some(*measured),
            Verdict::NotMeasurable { .. } => None,
        }
    }

    pub fn is_missed(&self) -> bool {
        matches!(self, Verdict::Missed { .. })
    }
}

/// One trigger's arm: how many settled retrievals it caused, and how many of
/// them served something.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arm {
    pub settled: usize,
    pub useful: usize,
    pub rate: f64,
}

/// What the journal sample could and could not see.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Window {
    /// Events in the sample.
    pub seen: usize,
    /// Of those, the ones whose outcome is known (`usage` set, not abandoned).
    pub settled: usize,
    /// Events pushed out of the bounded journal since the process started: the
    /// sample is recent, not complete.
    pub evicted: u64,
    /// Settled retrievals where the agent was blocked for lack of content.
    pub blocked_total: usize,
    /// Of those, the ones followed by a retrieval that served something.
    pub blocked_corrected: usize,
}

/// Drift between the two halves of the sample. Informational: it does not gate,
/// it warns that the thing being measured moved.
#[derive(Debug, Clone, PartialEq)]
pub struct Change {
    pub verdict: Verdict,
    pub first_half_rate: Option<f64>,
    pub second_half_rate: Option<f64>,
    pub delta: Option<f64>,
    pub detected: bool,
}

/// The report: six criteria, and the sample they were read on.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub window: Window,
    /// Criterion 1 — does memory get used at all?
    pub correction: Verdict,
    /// Criterion 2 — how long between "blocked" and "served"?
    pub time_to_correction: Verdict,
    /// Criterion 3 — how far is each trigger from the fixed reference arm?
    pub regret: Verdict,
    /// Per-trigger detail behind `regret`.
    pub per_trigger: Vec<(&'static str, Arm)>,
    /// Criterion 4 — is the index score a calibrated probability?
    pub calibration: Verdict,
    /// Criterion 5 — did the thing being measured move?
    pub change: Change,
    /// Criterion 6 — what does it cost per retrieval that served something?
    pub cost: Verdict,
}

/// The live report: current counters, current journal sample.
pub fn report() -> Report {
    evaluate(
        &super::retrieval_ledger::snapshot(),
        &super::retrieval_ledger::recent(WINDOW),
    )
}

/// The pure core: a report from a snapshot and a sample. No global is touched,
/// which is what makes it testable without a running server.
pub fn evaluate(snapshot: &Snapshot, events: &[RetrievalEvent]) -> Report {
    let settled: Vec<&RetrievalEvent> = events
        .iter()
        .filter(|e| e.usage.is_some() && !e.abandoned)
        .collect();

    let blocked: Vec<&&RetrievalEvent> = settled
        .iter()
        .filter(|e| e.usage.as_ref().and_then(|u| u.blocked.as_ref()).is_some())
        .collect();

    let window = Window {
        seen: events.len(),
        settled: settled.len(),
        evicted: snapshot.evicted,
        blocked_total: blocked.len(),
        blocked_corrected: 0,
    };

    let mut report = Report {
        window,
        correction: hit_rate(snapshot),
        time_to_correction: Verdict::NotMeasurable {
            reason: "no blocked retrieval in the sample",
        },
        regret: Verdict::NotMeasurable {
            reason: "no arm to compare",
        },
        per_trigger: Vec::new(),
        calibration: calibration(&settled),
        change: Change {
            verdict: Verdict::NotMeasurable {
                reason: "not enough settled retrievals",
            },
            first_half_rate: None,
            second_half_rate: None,
            delta: None,
            detected: false,
        },
        cost: cost_per_useful(snapshot),
    };

    let (timing, corrected) = time_to_correction(&settled);
    report.window.blocked_corrected = corrected;
    report.time_to_correction = timing;

    let (arms, regret) = regret_against_reference(&settled);
    report.per_trigger = arms;
    report.regret = regret;

    report.change = change(&settled);
    report
}

/// Criterion 1 — share of settled retrievals that had at least one item used.
///
/// Read on the counters, not on the journal: eviction must not shrink it.
fn hit_rate(snapshot: &Snapshot) -> Verdict {
    let denominator = snapshot.used + snapshot.unused;
    if denominator == 0 {
        return Verdict::NotMeasurable {
            reason: "no settled retrieval since start",
        };
    }
    let rate = snapshot.used as f64 / denominator as f64;
    Verdict::at_least(rate, CRITERIA.useful_hit_rate_min)
}

/// Criterion 6 — index calls per retrieval that served something.
///
/// "Useful" is the measured fact (an item was cited at the end of the turn),
/// not an attributed benefit: benefit is rarely filled in, and a cost computed
/// on an empty set would be a fiction.
fn cost_per_useful(snapshot: &Snapshot) -> Verdict {
    if snapshot.used == 0 {
        return Verdict::NotMeasurable {
            reason: "no retrieval has served anything yet",
        };
    }
    let cost = snapshot.calls as f64 / snapshot.used as f64;
    Verdict::at_most(cost, CRITERIA.cost_per_useful_max)
}

fn used_ids(e: &RetrievalEvent) -> &[String] {
    e.usage
        .as_ref()
        .map(|u| u.used_ids.as_slice())
        .unwrap_or(&[])
}

/// Criterion 2 — median delay from a retrieval the agent was blocked on to the
/// next retrieval of the same session that served something.
///
/// Returns the verdict and how many blocked retrievals were corrected.
fn time_to_correction(settled: &[&RetrievalEvent]) -> (Verdict, usize) {
    let mut by_session: std::collections::BTreeMap<uuid::Uuid, Vec<&RetrievalEvent>> =
        std::collections::BTreeMap::new();
    for e in settled {
        if let Some(session) = e.session_id {
            by_session.entry(session).or_default().push(e);
        }
    }

    let mut delays = Vec::new();
    let mut blocked_total = 0usize;
    for events in by_session.values_mut() {
        events.sort_by_key(|e| e.id);
        for (i, e) in events.iter().enumerate() {
            let blocked = e.usage.as_ref().and_then(|u| u.blocked.as_ref()).is_some();
            if !blocked {
                continue;
            }
            blocked_total += 1;
            let corrected = events[i + 1..]
                .iter()
                .find(|later| !used_ids(later).is_empty());
            if let Some(later) = corrected {
                if let (Ok(a), Ok(b)) = (
                    chrono::DateTime::parse_from_rfc3339(&e.at),
                    chrono::DateTime::parse_from_rfc3339(&later.at),
                ) {
                    let ms = (b - a).num_milliseconds().max(0) as f64;
                    delays.push(ms);
                }
            }
        }
    }

    if blocked_total == 0 {
        return (
            Verdict::NotMeasurable {
                reason: "no blocked retrieval in the sample",
            },
            0,
        );
    }
    let corrected = delays.len();
    if corrected == 0 {
        return (
            Verdict::NotMeasurable {
                reason: "blocked retrievals were never followed by one that served",
            },
            0,
        );
    }
    match median(delays) {
        Some(m) => (
            Verdict::at_most(m, CRITERIA.time_to_correction_median_ms_max),
            corrected,
        ),
        None => (
            Verdict::NotMeasurable {
                reason: "no usable timestamp on both sides",
            },
            0,
        ),
    }
}

/// Criterion 3 — gap between the fixed reference arm (explicit requests) and
/// the other triggers. Positive regret means the arm did worse than asking
/// explicitly; negative means it did better.
fn regret_against_reference(settled: &[&RetrievalEvent]) -> (Vec<(&'static str, Arm)>, Verdict) {
    let mut buckets: std::collections::BTreeMap<&'static str, (usize, usize)> =
        std::collections::BTreeMap::new();
    for e in settled {
        let entry = buckets.entry(e.trigger.as_str()).or_insert((0, 0));
        entry.0 += 1;
        if !used_ids(e).is_empty() {
            entry.1 += 1;
        }
    }
    let arms: Vec<(&'static str, Arm)> = buckets
        .into_iter()
        .map(|(name, (settled, useful))| {
            (
                name,
                Arm {
                    settled,
                    useful,
                    rate: useful as f64 / settled as f64,
                },
            )
        })
        .collect();

    let reference = arms
        .iter()
        .find(|(name, _)| *name == Trigger::Explicit.as_str());
    let Some((_, reference)) = reference else {
        return (
            arms,
            Verdict::NotMeasurable {
                reason: "the reference arm (explicit) has no settled retrieval",
            },
        );
    };
    if reference.settled < MIN_ARM {
        return (
            arms,
            Verdict::NotMeasurable {
                reason: "the reference arm (explicit) has fewer than MIN_ARM retrievals",
            },
        );
    }

    let mut worst: Option<(&'static str, f64)> = None;
    for (name, arm) in &arms {
        if *name == Trigger::Explicit.as_str() || arm.settled < MIN_ARM {
            continue;
        }
        let regret = reference.rate - arm.rate;
        if worst.is_none_or(|(_, w)| regret > w) {
            worst = Some((name, regret));
        }
    }
    match worst {
        Some((_, regret)) => (arms, Verdict::at_most(regret, CRITERIA.regret_max)),
        None => (
            arms,
            Verdict::NotMeasurable {
                reason: "no other trigger has MIN_ARM settled retrievals",
            },
        ),
    }
}

/// Criterion 4 — Brier score of the index score, read as the probability that
/// the item it ranks will actually be used.
///
/// The assumption is stated, not hidden: an index score is not calibrated by
/// construction, and measuring it is exactly the point. Scores are clamped to
/// [0, 1]; a cosine can be negative or above one.
fn calibration(settled: &[&RetrievalEvent]) -> Verdict {
    let mut sum = 0.0;
    let mut pairs = 0usize;
    for e in settled {
        let used = used_ids(e);
        for item in &e.results {
            let Some(score) = item.score else { continue };
            let predicted = score.clamp(0.0, 1.0);
            let observed = if used.contains(&item.id) { 1.0 } else { 0.0 };
            sum += (predicted - observed) * (predicted - observed);
            pairs += 1;
        }
    }
    if pairs < MIN_PAIRS {
        return Verdict::NotMeasurable {
            reason: "fewer than MIN_PAIRS (score, outcome) pairs in the sample",
        };
    }
    Verdict::at_most(sum / pairs as f64, CRITERIA.brier_max)
}

/// Criterion 5 — drift between the two halves of the sample.
///
/// Informational: crossing the delta does not mean "bad", it means the thing
/// being measured moved and a single average no longer describes it.
fn change(settled: &[&RetrievalEvent]) -> Change {
    let mut ordered: Vec<&&RetrievalEvent> = settled.iter().collect();
    ordered.sort_by_key(|e| e.id);
    if ordered.len() < 2 * MIN_ARM {
        return Change {
            verdict: Verdict::NotMeasurable {
                reason: "fewer than 2*MIN_ARM settled retrievals",
            },
            first_half_rate: None,
            second_half_rate: None,
            delta: None,
            detected: false,
        };
    }
    let mid = ordered.len() / 2;
    let rate = |slice: &[&&RetrievalEvent]| {
        let useful = slice.iter().filter(|e| !used_ids(e).is_empty()).count();
        useful as f64 / slice.len() as f64
    };
    let first = rate(&ordered[..mid]);
    let second = rate(&ordered[mid..]);
    let delta = (second - first).abs();
    Change {
        verdict: Verdict::at_most(delta, CRITERIA.change_delta),
        first_half_rate: Some(first),
        second_half_rate: Some(second),
        delta: Some(delta),
        detected: delta > CRITERIA.change_delta,
    }
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    Some(if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notes::retrieval_ledger::{Benefit, RetrievedItem, Usage};

    fn at(minute: u32) -> String {
        format!("2026-11-01T10:{minute:02}:00+00:00")
    }

    fn event(id: u64, trigger: Trigger, minute: u32) -> RetrievalEvent {
        RetrievalEvent {
            id,
            at: at(minute),
            session_id: Some(uuid::Uuid::nil()),
            source: crate::notes::retrieval_ledger::Source::Semantic,
            trigger,
            query: "q".to_string(),
            query_chars: 1,
            results: vec![RetrievedItem::new("n1", "note", Some(0.9))],
            ok: true,
            latency_ms: 10,
            usage: None,
            abandoned: false,
        }
    }

    fn settled(mut e: RetrievalEvent, used: &[&str], blocked: Option<&str>) -> RetrievalEvent {
        e.usage = Some(Usage {
            used_ids: used.iter().map(|s| s.to_string()).collect(),
            blocked: blocked.map(|b| b.to_string()),
            benefit: Some(Benefit::Yes),
            ..Default::default()
        });
        e
    }

    #[test]
    fn an_empty_ledger_measures_nothing_and_never_passes() {
        let report = evaluate(&Snapshot::default(), &[]);
        for (name, verdict) in [
            ("correction", &report.correction),
            ("time_to_correction", &report.time_to_correction),
            ("regret", &report.regret),
            ("calibration", &report.calibration),
            ("cost", &report.cost),
        ] {
            assert!(
                matches!(verdict, Verdict::NotMeasurable { .. }),
                "{name} must not pass on an empty ledger, got {verdict:?}"
            );
        }
        assert!(!report.change.detected);
    }

    #[test]
    fn a_ratio_without_denominator_is_not_a_zero() {
        // One call, never settled: the cost has no denominator.
        let snapshot = Snapshot {
            calls: 1,
            ..Default::default()
        };
        let report = evaluate(&snapshot, &[]);
        assert_eq!(
            report.cost,
            Verdict::NotMeasurable {
                reason: "no retrieval has served anything yet"
            }
        );
    }

    #[test]
    fn the_hit_rate_comes_from_the_counters_so_eviction_cannot_shrink_it() {
        let snapshot = Snapshot {
            used: 3,
            unused: 1,
            evicted: 500,
            ..Default::default()
        };
        let report = evaluate(&snapshot, &[]);
        assert_eq!(
            report.correction,
            Verdict::Met {
                measured: 0.75,
                threshold: CRITERIA.useful_hit_rate_min
            }
        );
        assert_eq!(report.window.evicted, 500);
    }

    #[test]
    fn a_blocked_retrieval_is_timed_until_the_next_one_that_served() {
        let blocked = settled(event(1, Trigger::Explicit, 0), &[], Some("missing note"));
        let served = settled(event(2, Trigger::Explicit, 2), &["n1"], None);
        let (verdict, corrected) = time_to_correction(&[&blocked, &served]);
        assert_eq!(corrected, 1);
        assert_eq!(
            verdict,
            Verdict::Missed {
                measured: 120_000.0,
                threshold: CRITERIA.time_to_correction_median_ms_max
            },
            "two minutes is over the registered minute: the verdict must say missed"
        );
    }

    #[test]
    fn a_blocked_retrieval_never_corrected_is_not_a_pass() {
        let blocked = settled(event(1, Trigger::Explicit, 0), &[], Some("missing note"));
        let (verdict, corrected) = time_to_correction(&[&blocked]);
        assert_eq!(corrected, 0);
        assert!(matches!(verdict, Verdict::NotMeasurable { .. }));
    }

    #[test]
    fn regret_is_measured_against_the_explicit_arm() {
        let mut events = Vec::new();
        // Reference arm: 10 explicit retrievals, 8 served.
        for i in 0..10u64 {
            let used: &[&str] = if i < 8 { &["n1"] } else { &[] };
            events.push(settled(event(i + 1, Trigger::Explicit, 0), used, None));
        }
        // Injection arm: 10 retrievals, none served -> regret 0.8 > 0.10.
        for i in 0..10u64 {
            events.push(settled(event(i + 100, Trigger::Injection, 5), &[], None));
        }
        let refs: Vec<&RetrievalEvent> = events.iter().collect();
        let (arms, verdict) = regret_against_reference(&refs);
        assert_eq!(arms.len(), 2);
        assert!(
            verdict.is_missed(),
            "an arm that never serves is a regret of 0.8, got {verdict:?}"
        );
    }

    #[test]
    fn a_confidence_of_one_on_an_unused_item_costs_a_full_brier_point() {
        let mut e = settled(event(1, Trigger::Explicit, 0), &[], None);
        e.results = (0..MIN_PAIRS)
            .map(|i| RetrievedItem::new(format!("n{i}"), "note", Some(1.0)))
            .collect();
        let verdict = calibration(&[&e]);
        assert_eq!(
            verdict,
            Verdict::Missed {
                measured: 1.0,
                threshold: CRITERIA.brier_max
            }
        );
    }

    #[test]
    fn change_is_detected_only_above_the_registered_delta() {
        let mut events = Vec::new();
        // First half: 5/10 served. Second half: 10/10 served. Delta 0.5.
        for i in 0..20u64 {
            let used: &[&str] = if (5..10).contains(&i) { &[] } else { &["n1"] };
            events.push(settled(event(i + 1, Trigger::Explicit, 0), used, None));
        }
        let refs: Vec<&RetrievalEvent> = events.iter().collect();
        let report = change(&refs);
        assert!(report.detected);
        assert_eq!(report.first_half_rate, Some(0.5));
        assert_eq!(report.second_half_rate, Some(1.0));
    }

    #[test]
    fn a_report_leaves_the_ledger_untouched() {
        let _guard = crate::notes::retrieval_ledger::serial_guard();
        let before = crate::notes::retrieval_ledger::snapshot();
        let journal_before = crate::notes::retrieval_ledger::recent(WINDOW).len();
        let _ = report();
        let after = crate::notes::retrieval_ledger::snapshot();
        let journal_after = crate::notes::retrieval_ledger::recent(WINDOW).len();
        assert_eq!(before, after, "the policy wrote to the ledger counters");
        assert_eq!(
            journal_before, journal_after,
            "the policy wrote to the journal"
        );
    }

    /// The policy must name no writer. Checked on the module's own text, the same
    /// way the router classification is checked against `routes.rs`: a comment
    /// cannot drift away from the code it describes.
    #[test]
    fn the_policy_names_no_write_path_to_production() {
        let src = include_str!("retrieval_policy.rs");
        let production = src.split("#[cfg(test)]").next().unwrap_or(src);
        for forbidden in [
            "retrieval_ledger::record",
            "retrieval_ledger::settle_turn",
            "retrieval_ledger::outcome",
            "retrieval_ledger::set_",
            "retrieval_ledger::begin_turn",
        ] {
            assert!(
                !production.contains(forbidden),
                "the policy calls {forbidden}: it is no longer read-only"
            );
        }
    }

    /// Un critère qui rate doit RENDRE SA MESURE : « manqué » sans chiffre ne dit
    /// pas de combien, et c'est ce chiffre qui décide de la correction.
    #[test]
    fn un_critere_sous_la_barre_est_manque_et_dit_sa_mesure() {
        // 1 récupération servie sur 4 réglées : 0.25, sous les 0.30 enregistrés.
        let snapshot = Snapshot {
            used: 1,
            unused: 3,
            ..Default::default()
        };
        let report = evaluate(&snapshot, &[]);
        assert!(
            report.correction.is_missed(),
            "0.25 est sous la barre : {report:?}"
        );
        assert_eq!(report.correction.measured(), Some(0.25));
        assert_eq!(
            Verdict::NotMeasurable { reason: "rien" }.measured(),
            None,
            "un critère non mesurable n'a pas de valeur, pas zéro"
        );
    }

    /// La médiane d'un échantillon vide n'existe pas (ce n'est pas zéro), et
    /// celle d'un nombre pair de valeurs est la moyenne des deux du milieu.
    #[test]
    fn une_mediane_sans_valeur_n_existe_pas_et_celle_d_un_pair_est_une_moyenne() {
        assert_eq!(median(Vec::new()), None, "pas de médiane sans valeur");
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(vec![3.0, 1.0, 2.0]), Some(2.0));
    }

    /// Un bras de référence trop maigre ne mesure rien : comparer un regret à
    /// cinq récupérations explicites serait un chiffre inventé.
    #[test]
    fn un_regret_sans_bras_de_reference_garni_n_est_pas_mesurable() {
        let mut events = Vec::new();
        for i in 0..(MIN_ARM / 2) as u64 {
            events.push(settled(event(i + 1, Trigger::Explicit, 0), &["n1"], None));
        }
        for i in 0..MIN_ARM as u64 {
            events.push(settled(event(i + 100, Trigger::Injection, 5), &[], None));
        }
        let refs: Vec<&RetrievalEvent> = events.iter().collect();
        let (_, verdict) = regret_against_reference(&refs);
        assert!(
            matches!(verdict, Verdict::NotMeasurable { .. }),
            "moins de MIN_ARM récupérations explicites : {verdict:?}"
        );
    }

    /// Le bras de référence est garni mais aucun autre motif ne l'est : il n'y a
    /// rien à comparer, et ce n'est pas un regret nul.
    #[test]
    fn un_regret_sans_autre_bras_garni_n_est_pas_mesurable() {
        let mut events = Vec::new();
        for i in 0..MIN_ARM as u64 {
            events.push(settled(event(i + 1, Trigger::Explicit, 0), &["n1"], None));
        }
        for i in 0..(MIN_ARM / 2) as u64 {
            events.push(settled(event(i + 100, Trigger::Injection, 5), &[], None));
        }
        let refs: Vec<&RetrievalEvent> = events.iter().collect();
        let (_, verdict) = regret_against_reference(&refs);
        assert!(
            matches!(verdict, Verdict::NotMeasurable { .. }),
            "aucun autre motif n'atteint MIN_ARM : {verdict:?}"
        );
    }
}
