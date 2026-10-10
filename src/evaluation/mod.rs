//! Offline evaluation bench for the backend classifiers.
//!
//! Five classifiers decide things the backend acts on: which tool groups a
//! message needs, the intent of a message, the class of a chat turn, which
//! skill a message or file activates, and which event trigger fires. Until now
//! none of them had an accuracy number. This module measures each one against a
//! hand-labelled fixture (`tests/fixtures/classifiers/*.json`) and against the
//! majority-class baseline, which is the score a classifier that ignores its
//! input would get.
//!
//! The bench is offline, deterministic and fast: no network, no Neo4j, no clock
//! in the predictions. Its figures are published in the shadow routing report
//! (`classifiers` field) and by the heartbeat check `classifier_bench`.
//!
//! Language invariance is measured where the fixture pairs a French case and an
//! English case under the same `pair` id: the share of pairs on which the
//! classifier gives the same answer (the same metamorphic idea as
//! `tests/routing_metamorphic.rs`, but on labelled data).

pub mod classifiers;

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, Context, Result};
use neural_routing_gnn::benchmark::metrics::f1_score;
use serde::{Deserialize, Serialize};

/// One labelled case of a fixture.
#[derive(Debug, Clone, Deserialize)]
pub struct Case {
    pub id: String,
    /// `fr` or `en`.
    pub lang: String,
    /// Cases sharing a pair id are the same meaning in two languages.
    #[serde(default)]
    pub pair: Option<String>,
    /// Classifier-specific input, read by its adapter.
    pub input: serde_json::Value,
    /// Hand-written gold label.
    pub label: String,
}

/// A versioned fixture: one classifier, its catalog (when it needs one), its cases.
#[derive(Debug, Clone, Deserialize)]
pub struct Fixture {
    pub classifier: String,
    pub version: u32,
    #[serde(default)]
    pub catalog: serde_json::Value,
    pub cases: Vec<Case>,
}

/// A classifier under test. `name` is the fixture's `classifier` key.
pub trait Classifier {
    fn name(&self) -> &'static str;
    fn predict(&self, fixture: &Fixture, case: &Case) -> Result<String>;
}

/// Scores of one classifier on one fixture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassifierScore {
    pub classifier: String,
    pub cases: usize,
    pub fr_cases: usize,
    pub en_cases: usize,
    pub accuracy: f64,
    /// Accuracy of always predicting the most frequent gold label.
    pub baseline_accuracy: f64,
    pub baseline_label: String,
    pub macro_f1: f64,
    /// Share of FR/EN pairs answered identically. `None` without pairs.
    pub lang_agreement: Option<f64>,
    pub lang_pairs: usize,
    /// `accuracy >= baseline_accuracy`.
    pub above_baseline: bool,
}

/// Most frequent label and its share. Ties go to the smallest label, so the
/// baseline is deterministic.
pub fn majority_baseline(labels: &[&str]) -> Option<(String, f64)> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for label in labels {
        *counts.entry(label).or_default() += 1;
    }
    let (label, n) = counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(a.0)))?;
    Some((label.to_string(), n as f64 / labels.len() as f64))
}

/// Share of predictions equal to the gold label. `0.0` on an empty set.
pub fn accuracy(predictions: &[&str], gold: &[&str]) -> f64 {
    if gold.is_empty() {
        return 0.0;
    }
    let hits = predictions.iter().zip(gold).filter(|(p, g)| p == g).count();
    hits as f64 / gold.len() as f64
}

/// Macro-averaged F1 over every label seen in the gold or the predictions,
/// one-vs-rest, through the shared `f1_score` of the GNN benchmark.
pub fn macro_f1(predictions: &[&str], gold: &[&str]) -> f64 {
    let labels: BTreeSet<&str> = predictions.iter().chain(gold).copied().collect();
    if labels.is_empty() {
        return 0.0;
    }
    let total: f64 = labels
        .iter()
        .map(|label| {
            let pairs: Vec<(bool, bool)> = predictions
                .iter()
                .zip(gold)
                .map(|(p, g)| (p == label, g == label))
                .collect();
            f1_score(&pairs).2
        })
        .sum();
    total / labels.len() as f64
}

/// Share of FR/EN pairs (same `pair` id, one case per language) on which the
/// predictions agree. `None` when the fixture has no pairs.
pub fn lang_agreement(cases: &[Case], predictions: &[&str]) -> (Option<f64>, usize) {
    let mut by_pair: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (case, prediction) in cases.iter().zip(predictions) {
        if let Some(pair) = case.pair.as_deref() {
            by_pair.entry(pair).or_default().push(prediction);
        }
    }
    let pairs: Vec<&Vec<&str>> = by_pair.values().filter(|v| v.len() == 2).collect();
    if pairs.is_empty() {
        return (None, 0);
    }
    let agreed = pairs.iter().filter(|v| v[0] == v[1]).count();
    (Some(agreed as f64 / pairs.len() as f64), pairs.len())
}

/// Scores one classifier on one fixture.
pub fn score(fixture: &Fixture, classifier: &dyn Classifier) -> Result<ClassifierScore> {
    let predictions: Vec<String> = fixture
        .cases
        .iter()
        .map(|case| classifier.predict(fixture, case))
        .collect::<Result<_>>()
        .with_context(|| format!("classifier `{}`", classifier.name()))?;
    let preds: Vec<&str> = predictions.iter().map(String::as_str).collect();
    let gold: Vec<&str> = fixture.cases.iter().map(|c| c.label.as_str()).collect();
    let (baseline_label, baseline_accuracy) = majority_baseline(&gold)
        .ok_or_else(|| anyhow!("fixture `{}` has no cases", fixture.classifier))?;
    let acc = accuracy(&preds, &gold);
    let (lang_agreement, lang_pairs) = lang_agreement(&fixture.cases, &preds);
    Ok(ClassifierScore {
        classifier: fixture.classifier.clone(),
        cases: fixture.cases.len(),
        fr_cases: fixture.cases.iter().filter(|c| c.lang == "fr").count(),
        en_cases: fixture.cases.iter().filter(|c| c.lang == "en").count(),
        accuracy: acc,
        baseline_accuracy,
        baseline_label,
        macro_f1: macro_f1(&preds, &gold),
        lang_agreement,
        lang_pairs,
        above_baseline: acc >= baseline_accuracy,
    })
}

/// Scores every fixture with the classifier of the same name.
pub fn run(fixtures: &[Fixture]) -> Result<Vec<ClassifierScore>> {
    let all = classifiers::all();
    fixtures
        .iter()
        .map(|fixture| {
            let classifier = all
                .iter()
                .find(|c| c.name() == fixture.classifier)
                .ok_or_else(|| anyhow!("no classifier named `{}`", fixture.classifier))?;
            score(fixture, classifier.as_ref())
        })
        .collect()
}

/// The fixtures versioned in `tests/fixtures/classifiers/`, embedded at build time.
pub fn embedded_fixtures() -> Result<Vec<Fixture>> {
    const SOURCES: &[(&str, &str)] = &[
        (
            "tool_groups",
            include_str!("../../tests/fixtures/classifiers/tool_groups.json"),
        ),
        (
            "intent",
            include_str!("../../tests/fixtures/classifiers/intent.json"),
        ),
        (
            "task_class",
            include_str!("../../tests/fixtures/classifiers/task_class.json"),
        ),
        (
            "skills",
            include_str!("../../tests/fixtures/classifiers/skills.json"),
        ),
        (
            "triggers",
            include_str!("../../tests/fixtures/classifiers/triggers.json"),
        ),
    ];
    SOURCES
        .iter()
        .map(|(name, json)| {
            serde_json::from_str::<Fixture>(json)
                .with_context(|| format!("embedded fixture `{name}`"))
        })
        .collect()
}

/// Scores of the five classifiers on the embedded fixtures.
pub fn run_embedded() -> Result<Vec<ClassifierScore>> {
    run(&embedded_fixtures()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(id: &str, lang: &str, pair: Option<&str>, label: &str) -> Case {
        Case {
            id: id.into(),
            lang: lang.into(),
            pair: pair.map(Into::into),
            input: serde_json::Value::Null,
            label: label.into(),
        }
    }

    #[test]
    fn baseline_is_the_frequency_of_the_most_common_label() {
        let (label, share) = majority_baseline(&["a", "b", "b", "b", "c"]).unwrap();
        assert_eq!(label, "b");
        assert!((share - 0.6).abs() < 1e-12);
    }

    #[test]
    fn baseline_ties_go_to_the_smallest_label() {
        let (label, _) = majority_baseline(&["y", "x", "y", "x"]).unwrap();
        assert_eq!(label, "x");
    }

    #[test]
    fn baseline_of_nothing_is_none() {
        assert!(majority_baseline(&[]).is_none());
    }

    #[test]
    fn a_perfect_prediction_has_accuracy_one_and_macro_f1_one() {
        let gold = ["a", "b", "a"];
        assert_eq!(accuracy(&gold, &gold), 1.0);
        assert!((macro_f1(&gold, &gold) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn accuracy_counts_exact_matches_only() {
        assert!((accuracy(&["a", "a", "b"], &["a", "b", "b"]) - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn lang_agreement_uses_only_complete_pairs() {
        let cases = [
            case("1-fr", "fr", Some("1"), "a"),
            case("1-en", "en", Some("1"), "a"),
            case("2-fr", "fr", Some("2"), "b"),
            case("2-en", "en", Some("2"), "b"),
            case("solo", "fr", None, "a"),
        ];
        let (agreement, pairs) = lang_agreement(&cases, &["a", "a", "b", "a", "a"]);
        assert_eq!(pairs, 2);
        assert_eq!(agreement, Some(0.5));
    }

    #[test]
    fn lang_agreement_without_pairs_is_none() {
        let cases = [case("x", "en", None, "a")];
        assert_eq!(lang_agreement(&cases, &["a"]), (None, 0));
    }

    #[test]
    fn the_embedded_fixtures_parse_and_name_the_five_classifiers() {
        let fixtures = embedded_fixtures().unwrap();
        let names: BTreeSet<&str> = fixtures.iter().map(|f| f.classifier.as_str()).collect();
        assert_eq!(
            names,
            BTreeSet::from(["intent", "skills", "task_class", "tool_groups", "triggers"])
        );
    }
}
