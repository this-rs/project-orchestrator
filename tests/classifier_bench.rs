//! Offline bench of the five backend classifiers (`project_orchestrator::evaluation`).
//!
//! Each classifier is scored on its hand-labelled fixture in
//! `tests/fixtures/classifiers/` and compared with the majority-class baseline.
//! The bench is offline and deterministic: no network, no Neo4j.
//!
//! Run it alone with `cargo test --test classifier_bench -- --nocapture` to see
//! the table.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;

use project_orchestrator::evaluation::{self, ClassifierScore, Fixture};
use sha2::{Digest, Sha256};

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/classifiers");

fn print_table(scores: &[ClassifierScore], elapsed: std::time::Duration) {
    println!(
        "\n{:<12} {:>6} {:>7} {:>9} {:>9} {:>9} {:>9} {:>6}",
        "classifier", "cases", "fr/en", "accuracy", "baseline", "macro-F1", "FR<->EN", "pairs"
    );
    for s in scores {
        let lang = s
            .lang_agreement
            .map_or_else(|| "n/a".to_string(), |a| format!("{:.1}%", a * 100.0));
        println!(
            "{:<12} {:>6} {:>3}/{:<3} {:>8.1}% {:>8.1}% {:>9.3} {:>9} {:>6}   (baseline label: {})",
            s.classifier,
            s.cases,
            s.fr_cases,
            s.en_cases,
            s.accuracy * 100.0,
            s.baseline_accuracy * 100.0,
            s.macro_f1,
            lang,
            s.lang_pairs,
            s.baseline_label,
        );
    }
    println!("bench wall time: {elapsed:?}");
}

#[test]
fn every_classifier_is_scored_against_the_majority_baseline() {
    let started = Instant::now();
    let scores = evaluation::run_embedded().expect("the embedded fixtures score");
    let elapsed = started.elapsed();
    print_table(&scores, elapsed);

    let names: BTreeSet<&str> = scores.iter().map(|s| s.classifier.as_str()).collect();
    assert_eq!(
        names,
        BTreeSet::from(["intent", "skills", "task_class", "tool_groups", "triggers"]),
        "the bench must list the five classifiers"
    );
    assert!(
        elapsed.as_secs_f64() < 5.0,
        "the bench must run in under 5 s, took {elapsed:?}"
    );

    let under: Vec<String> = scores
        .iter()
        .filter(|s| !s.above_baseline)
        .map(|s| {
            format!(
                "{} ({:.1}% < baseline {:.1}%)",
                s.classifier,
                s.accuracy * 100.0,
                s.baseline_accuracy * 100.0
            )
        })
        .collect();
    assert!(under.is_empty(), "under the majority baseline: {under:?}");
}

#[test]
fn a_classifier_under_its_baseline_fails_the_bench() {
    // Degraded fixture: every gold label of the intent classifier is replaced by
    // the next class (cyclic, in sorted order). That is a bijection on the
    // classes, so the class counts and the baseline do not change, while every
    // gold label becomes wrong for the classifier.
    let mut fixtures: Vec<Fixture> = evaluation::embedded_fixtures().expect("fixtures parse");
    for fixture in fixtures.iter_mut().filter(|f| f.classifier == "intent") {
        let classes: Vec<String> = fixture
            .cases
            .iter()
            .map(|c| c.label.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        for case in &mut fixture.cases {
            let at = classes
                .iter()
                .position(|c| *c == case.label)
                .expect("label is a class");
            case.label = classes[(at + 1) % classes.len()].clone();
        }
    }

    let scores = evaluation::run(&fixtures).expect("the degraded fixture scores");
    let intent = scores
        .iter()
        .find(|s| s.classifier == "intent")
        .expect("intent is scored");
    assert!(
        !intent.above_baseline,
        "degraded intent labels must put it under the baseline: accuracy {:.3}, baseline {:.3}",
        intent.accuracy, intent.baseline_accuracy
    );
}

#[test]
fn each_fixture_has_at_least_30_cases_and_10_french_ones() {
    for fixture in evaluation::embedded_fixtures().expect("fixtures parse") {
        let fr = fixture.cases.iter().filter(|c| c.lang == "fr").count();
        assert!(
            fixture.cases.len() >= 30,
            "{}: {} cases",
            fixture.classifier,
            fixture.cases.len()
        );
        assert!(fr >= 10, "{}: {} French cases", fixture.classifier, fr);
        let ids: BTreeSet<&str> = fixture.cases.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(
            ids.len(),
            fixture.cases.len(),
            "{}: duplicate ids",
            fixture.classifier
        );
    }
}

#[test]
fn the_fixtures_match_their_sha256_manifest() {
    let dir = Path::new(FIXTURE_DIR);
    let manifest = std::fs::read_to_string(dir.join("MANIFEST.sha256")).expect("MANIFEST.sha256");
    let mut listed = BTreeSet::new();
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let (expected, name) = line.split_once("  ").expect("`<sha256>  <file>` lines");
        let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        let actual: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(actual, expected, "{name} differs from the manifest");
        listed.insert(name.to_string());
    }

    let on_disk: BTreeSet<String> = std::fs::read_dir(dir)
        .expect("fixture dir")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|name| name.ends_with(".json"))
        .collect();
    assert_eq!(listed, on_disk, "every fixture must be in the manifest");
}
