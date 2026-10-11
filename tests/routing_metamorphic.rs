//! Metamorphic invariance harness for `RoutingProvider`.
//!
//! Ported from Laya's `research/eval/metamorphic.py` (Convai Innovations, Apache-2.0).
//!
//! ## Why this exists
//!
//! We have no gold labels for "which tool groups did this message actually need".
//! Accuracy is therefore unmeasurable. Invariance is not: two inputs that mean the
//! same thing must produce the same routing decision. A violation is a bug even
//! though we never learn what the right answer was.
//!
//! Three invariants:
//!  - `lang`  — a message and its translation must select the same tool groups.
//!  - `noise` — appending a semantically empty sentence must change nothing.
//!  - `level` — raising the scaffolding level may only shrink the selection.

use std::collections::BTreeSet;

use project_orchestrator::chat::routing::{HeuristicRouter, RoutingContext, RoutingProvider};

// ============================================================================
// Harness
// ============================================================================

fn decide(router: &dyn RoutingProvider, msg: &str, level: u8) -> BTreeSet<String> {
    let ctx = RoutingContext {
        scaffolding_level: level,
        user_message: msg.to_string(),
        ..Default::default()
    };
    router
        .route(&ctx)
        .tool_groups
        .iter()
        .map(|g| format!("{:?}", g))
        .collect()
}

fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 1.0;
    }
    a.intersection(b).count() as f64 / union as f64
}

fn fmt(set: &BTreeSet<String>) -> String {
    let v: Vec<&str> = set.iter().map(|s| s.as_str()).collect();
    v.join("+")
}

struct Report {
    kind: &'static str,
    n: usize,
    agree: usize,
    jaccard_sum: f64,
    violations: Vec<String>,
}

impl Report {
    fn new(kind: &'static str) -> Self {
        Self {
            kind,
            n: 0,
            agree: 0,
            jaccard_sum: 0.0,
            violations: Vec::new(),
        }
    }

    fn observe(&mut self, label: &str, a: &BTreeSet<String>, b: &BTreeSet<String>) {
        self.n += 1;
        self.jaccard_sum += jaccard(a, b);
        if a == b {
            self.agree += 1;
        } else {
            let lost: Vec<&str> = a.difference(b).map(|s| s.as_str()).collect();
            let gained: Vec<&str> = b.difference(a).map(|s| s.as_str()).collect();
            self.violations.push(format!(
                "    {label}\n      A = {}\n      B = {}\n      lost: [{}]  gained: [{}]",
                fmt(a),
                fmt(b),
                lost.join(", "),
                gained.join(", "),
            ));
        }
    }

    fn print(&self) {
        let rate = self.agree as f64 / self.n.max(1) as f64;
        println!(
            "\n── invariant `{}` ───────────────────────────────────────────\n\
             cases: {}   agreement: {}/{} = {:.0}%   mean jaccard: {:.3}",
            self.kind,
            self.n,
            self.agree,
            self.n,
            rate * 100.0,
            self.jaccard_sum / self.n.max(1) as f64
        );
        if self.violations.is_empty() {
            println!("  no violations");
        } else {
            println!("  VIOLATIONS ({}):", self.violations.len());
            for v in &self.violations {
                println!("{v}");
            }
        }
    }

    fn agreement(&self) -> f64 {
        self.agree as f64 / self.n.max(1) as f64
    }
}

// ============================================================================
// Paired corpus — (french, english) meaning the same thing
// ============================================================================

const PAIRS: &[(&str, &str, &str)] = &[
    // group under test, FR, EN
    (
        "code",
        "cherche la fonction build_system_prompt",
        "search for the function build_system_prompt",
    ),
    (
        "code",
        "quelles sont les dépendances de ce fichier ?",
        "what are the dependencies of this file?",
    ),
    (
        "code",
        "montre-moi le graphe d'appels",
        "show me the call graph",
    ),
    (
        "code",
        "quel est l'impact de cette modification ?",
        "what is the impact of this change?",
    ),
    (
        "structural",
        "quels sont les points chauds du projet ?",
        "what are the hotspots of the project?",
    ),
    (
        "structural",
        "montre-moi la santé du graphe",
        "show me the graph health",
    ),
    (
        "structural",
        "il faut lancer une synchronisation",
        "we need to run a sync",
    ),
    (
        "structural",
        "l'énergie des synapses a l'air basse",
        "the synapse energy looks low",
    ),
    (
        "behavioral",
        "où en est le protocole en cours ?",
        "what is the state of the running protocol?",
    ),
    (
        "behavioral",
        "quelles compétences sont disponibles ?",
        "which skills are available?",
    ),
    (
        "behavioral",
        "raconte-moi l'épisode précédent",
        "tell me about the previous episode",
    ),
    (
        "behavioral",
        "décris la machine à états",
        "describe the state machine",
    ),
    (
        "workspace",
        "montre-moi la topologie de l'espace de travail",
        "show me the workspace topology",
    ),
    (
        "workspace",
        "quels composants dépendent de celui-ci ?",
        "which components depend on this one?",
    ),
    (
        "workspace",
        "on a besoin d'une ressource partagée",
        "we need a shared resource",
    ),
    (
        "workspace",
        "c'est un changement inter-projets",
        "this is a cross-project change",
    ),
    (
        "collab",
        "reprends la discussion précédente",
        "resume the previous chat",
    ),
    (
        "collab",
        "partage ce résultat avec l'équipe",
        "share this result with the team",
    ),
    (
        "collab",
        "montre-moi l'arbre de raisonnement",
        "show me the reasoning tree",
    ),
    (
        "collab",
        "liste les échanges ouverts",
        "list the open sessions",
    ),
    // ── Added AFTER the fix, deliberately not tuned for it ──────────────
    (
        "code",
        "quels fichiers ont changé ensemble ?",
        "which files changed together?",
    ),
    (
        "behavioral",
        "montre-moi les personas actifs",
        "show me the active personas",
    ),
    (
        "workspace",
        "la ressource partagée est obsolète",
        "the shared resource is stale",
    ),
    (
        "collab",
        "trace la trajectoire de cette session",
        "trace the trajectory of this session",
    ),
    (
        "none",
        "il y a une régression dans le parseur",
        "there is a regression in the parser",
    ),
    (
        "none",
        "combien de tâches restent dans le plan ?",
        "how many tasks are left in the plan?",
    ),
];

/// Sentences that carry no routing intent and must not change any decision.
const NOISE: &[&str] = &[
    " Merci d'avance.",
    " On en reparle à la prochaine session.",
    " C'est urgent.",
];

// ============================================================================
// Invariant 1 — language
// ============================================================================

#[test]
fn metamorphic_language_invariance() {
    let router = HeuristicRouter;
    let mut report = Report::new("lang (fr <-> en)");

    for (grp, fr, en) in PAIRS {
        let a = decide(&router, fr, 4);
        let b = decide(&router, en, 4);
        report.observe(&format!("[{grp}] \"{fr}\"  vs  \"{en}\""), &a, &b);
    }

    report.print();
    assert!(
        report.agreement() >= 1.0,
        "language invariance violated: {:.0}% agreement",
        report.agreement() * 100.0
    );
}

// ============================================================================
// Invariant 2 — irrelevant noise
// ============================================================================

#[test]
fn metamorphic_noise_invariance() {
    let router = HeuristicRouter;
    let mut report = Report::new("noise (append empty sentence)");

    for (grp, fr, _en) in PAIRS {
        let base = decide(&router, fr, 4);
        for noise in NOISE {
            let noised = format!("{fr}{noise}");
            let b = decide(&router, &noised, 4);
            report.observe(
                &format!("[{grp}] +\"{}\" on \"{fr}\"", noise.trim()),
                &base,
                &b,
            );
        }
    }

    report.print();
    assert!(
        report.agreement() >= 1.0,
        "noise invariance violated: {:.0}% agreement",
        report.agreement() * 100.0
    );
}

// ============================================================================
// Invariant 3 — scaffolding monotonicity
// ============================================================================

#[test]
fn metamorphic_level_monotonicity() {
    let router = HeuristicRouter;
    let mut n = 0usize;
    let mut bad: Vec<String> = Vec::new();

    for (grp, fr, en) in PAIRS {
        for msg in [fr, en] {
            for (hi, lo) in [(1u8, 0u8), (2, 1), (3, 2), (4, 3)] {
                n += 1;
                let a = decide(&router, msg, hi);
                let b = decide(&router, msg, lo);
                // higher level must be a subset of lower level
                if !a.is_subset(&b) {
                    let extra: Vec<&str> = a.difference(&b).map(|s| s.as_str()).collect();
                    bad.push(format!(
                        "    [{grp}] \"{msg}\"  L{hi} ⊄ L{lo}   extra at L{hi}: [{}]",
                        extra.join(", ")
                    ));
                }
            }
        }
    }

    println!(
        "\n── invariant `level (monotonic shrink)` ─────────────────────\n\
         cases: {n}   violations: {}",
        bad.len()
    );
    for b in &bad {
        println!("{b}");
    }
    if bad.is_empty() {
        println!("  no violations");
    }
    assert!(
        bad.is_empty(),
        "scaffolding monotonicity violated in {} cases",
        bad.len()
    );
}
