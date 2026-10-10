//! Metamorphic invariance of the text classifiers, offline.
//!
//! Three equivalences over the labelled bench fixtures (`tests/fixtures/classifiers`,
//! the same adapters as `src/evaluation/classifiers.rs`):
//!  - `lang`       — a French message and its English pair get the same answer;
//!  - `noise`      — punctuation noise on a message changes nothing;
//!  - `paraphrase` — a hand-written rewording in the same language changes nothing.
//!
//! Each classifier has its own agreement threshold (`THRESHOLDS`), asserted in CI.
//! Deterministic: no network, no embedding (`IntentDetector`, `TaskSignature`,
//! `evaluate_skill_match` are lexical).

mod common;

use common::metamorphic::{label_similarity, punctuation_noise, Report};
use project_orchestrator::evaluation::{classifiers, embedded_fixtures, Case, Classifier, Fixture};
use serde_json::Value;

/// Rewordings written before the first measurement, not tuned against it.
/// Each pair is one message and a paraphrase in the same language.
const PARAPHRASES: &[(&str, &str)] = &[
    (
        "pourquoi le paiement échoue ?",
        "pourquoi le paiement ne passe pas ?",
    ),
    (
        "why does the payment fail?",
        "why doesn't the payment go through?",
    ),
    ("il y a un bug dans le cache", "le cache se comporte mal"),
    ("there is a bug in the cache", "the cache misbehaves"),
    (
        "ça plante au démarrage, aide-moi",
        "ça ne démarre plus, aide-moi",
    ),
    (
        "it crashes at startup, help me",
        "it won't start anymore, help me",
    ),
    (
        "comment fonctionne le cache ?",
        "peux-tu m'expliquer le cache ?",
    ),
    ("how does the cache work?", "can you explain the cache?"),
    (
        "on doit ajouter une fonctionnalité de cache",
        "il faut créer un cache",
    ),
    ("we need to implement a cache", "we should set up a cache"),
    (
        "quel est l'impact de cette modification ?",
        "quel est l'impact de ce changement ?",
    ),
    (
        "what is the impact of this change?",
        "what would this change affect?",
    ),
];

/// Minimum agreement per classifier and per invariant, asserted in CI.
///
/// `lang` and `noise` are set at the measured value after the keyword fix: the
/// fixture pairs and punctuation variants all agree except the residual pair
/// noted on `task_class`. `paraphrase` values are floors (non-regression),
/// not targets: the paraphrase list was written to probe, and its violations
/// are lexical gaps that stay visible in the test output.
struct Threshold {
    classifier: &'static str,
    lang: f64,
    noise: f64,
    paraphrase: f64,
}

const THRESHOLDS: &[Threshold] = &[
    // lang: 16/16 after the fix.
    Threshold {
        classifier: "intent",
        lang: 1.0,
        noise: 1.0,
        paraphrase: 0.5,
    },
    // lang: 15/16. Residual: "comment les triggers sont-ils évalués ?" (explore)
    // vs "how are triggers evaluated?" (general): "how are" is too broad to add
    // (it matches "how are you").
    Threshold {
        classifier: "task_class",
        lang: 0.9375,
        noise: 1.0,
        paraphrase: 0.5,
    },
    // lang: 16/16 on the fixture, whose skill regexes are bilingual by design.
    Threshold {
        classifier: "skills",
        lang: 1.0,
        noise: 1.0,
        paraphrase: 0.58,
    },
];

fn message(case: &Case) -> &str {
    case.input
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or_default()
}

/// The same case with another message; every other input field is kept.
fn with_message(case: &Case, message: &str) -> Case {
    let mut varied = case.clone();
    varied.input["message"] = Value::String(message.to_string());
    varied
}

/// A case with only a message, for texts that come from no fixture.
fn bare_case(message: &str) -> Case {
    Case {
        id: "paraphrase".to_string(),
        lang: "en".to_string(),
        pair: None,
        input: serde_json::json!({ "message": message }),
        label: String::new(),
    }
}

fn predict(classifier: &dyn Classifier, fixture: &Fixture, case: &Case) -> String {
    classifier
        .predict(fixture, case)
        .unwrap_or_else(|e| panic!("{} failed on {:?}: {e:#}", classifier.name(), message(case)))
}

/// Cases of a fixture grouped by pair id: `(french, english)`.
fn pairs(fixture: &Fixture) -> Vec<(&Case, &Case)> {
    let mut out = Vec::new();
    for fr in fixture.cases.iter().filter(|c| c.lang == "fr") {
        let Some(pair) = fr.pair.as_deref() else {
            continue;
        };
        if let Some(en) = fixture
            .cases
            .iter()
            .find(|c| c.lang == "en" && c.pair.as_deref() == Some(pair))
        {
            out.push((fr, en));
        }
    }
    out
}

fn check(name: &str) {
    let threshold = THRESHOLDS
        .iter()
        .find(|t| t.classifier == name)
        .unwrap_or_else(|| panic!("no threshold for `{name}`"));
    let fixtures = embedded_fixtures().expect("the embedded fixtures parse");
    let fixture = fixtures
        .iter()
        .find(|f| f.classifier == name)
        .unwrap_or_else(|| panic!("no fixture for `{name}`"));
    let all = classifiers::all();
    let classifier = all
        .iter()
        .find(|c| c.name() == name)
        .unwrap_or_else(|| panic!("no classifier `{name}`"))
        .as_ref();

    let mut lang = Report::new("lang (fr <-> en)");
    for (fr, en) in pairs(fixture) {
        let a = predict(classifier, fixture, fr);
        let b = predict(classifier, fixture, en);
        lang.observe(
            &format!("\"{}\"  vs  \"{}\"", message(fr), message(en)),
            &a,
            &b,
            label_similarity(&a, &b),
        );
    }

    let mut noise = Report::new("noise (punctuation)");
    for case in &fixture.cases {
        let base = predict(classifier, fixture, case);
        for varied in punctuation_noise(message(case)) {
            let b = predict(classifier, fixture, &with_message(case, &varied));
            noise.observe(
                &format!("\"{}\"  ->  \"{varied}\"", message(case)),
                &base,
                &b,
                label_similarity(&base, &b),
            );
        }
    }

    let mut paraphrase = Report::new("paraphrase (same language)");
    for (original, reworded) in PARAPHRASES {
        let a = predict(classifier, fixture, &bare_case(original));
        let b = predict(classifier, fixture, &bare_case(reworded));
        paraphrase.observe(
            &format!("\"{original}\"  ->  \"{reworded}\""),
            &a,
            &b,
            label_similarity(&a, &b),
        );
    }

    println!("\n=== classifier `{name}` ===");
    lang.print();
    noise.print();
    paraphrase.print();

    for (report, min) in [
        (&lang, threshold.lang),
        (&noise, threshold.noise),
        (&paraphrase, threshold.paraphrase),
    ] {
        assert!(
            report.agreement() + 1e-12 >= min,
            "`{name}` {} agreement {:.1}% < threshold {:.1}%:\n{}",
            report.kind(),
            report.agreement() * 100.0,
            min * 100.0,
            report.violations().join("\n"),
        );
    }
}

#[test]
fn intent_is_invariant_to_language_noise_and_paraphrase() {
    check("intent");
}

#[test]
fn task_class_is_invariant_to_language_noise_and_paraphrase() {
    check("task_class");
}

#[test]
fn skills_are_invariant_to_language_noise_and_paraphrase() {
    check("skills");
}
