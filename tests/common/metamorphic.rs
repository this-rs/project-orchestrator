//! Metamorphic harness shared by the offline classifier tests.
//!
//! Ported from Laya's `research/eval/metamorphic.py` (Convai Innovations, Apache-2.0).
//!
//! We rarely have gold labels for what a classifier should answer. Invariance
//! needs none: two inputs that mean the same thing must get the same answer,
//! and a violation is a bug even when the right answer is unknown.
//!
//! The harness only compares outcomes. Each test chooses its equivalence
//! (translation, paraphrase, noise, permutation, monotonicity) and renders its
//! outcomes as strings; `Report` counts agreement and mean similarity.
//!
//! Nothing here touches the network or an embedding model.

use std::collections::BTreeSet;

/// Jaccard similarity of two sets. Two empty sets are identical.
pub fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 1.0;
    }
    a.intersection(b).count() as f64 / union as f64
}

/// Similarity of two single labels: `1.0` when equal, `0.0` otherwise.
pub fn label_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        1.0
    } else {
        0.0
    }
}

/// Renders a set as `a+b+c`, sorted, `-` when empty.
pub fn fmt_set(set: &BTreeSet<String>) -> String {
    if set.is_empty() {
        return "-".to_string();
    }
    set.iter().map(String::as_str).collect::<Vec<_>>().join("+")
}

/// Counts equivalence checks: how many agree exactly, the mean similarity, and
/// one line per violation.
pub struct Report {
    kind: &'static str,
    n: usize,
    agree: usize,
    similarity_sum: f64,
    violations: Vec<String>,
}

impl Report {
    pub fn new(kind: &'static str) -> Self {
        Self {
            kind,
            n: 0,
            agree: 0,
            similarity_sum: 0.0,
            violations: Vec::new(),
        }
    }

    /// Records one check. `a` and `b` are the rendered outcomes; `similarity`
    /// is in `[0, 1]` (Jaccard for sets, `label_similarity` for labels).
    pub fn observe(&mut self, label: &str, a: &str, b: &str, similarity: f64) {
        self.n += 1;
        self.similarity_sum += similarity;
        if a == b {
            self.agree += 1;
        } else {
            self.violations
                .push(format!("    {label}\n      A = {a}\n      B = {b}"));
        }
    }

    pub fn print(&self) {
        println!(
            "\n── invariant `{}` ───────────────────────────────────────────\n\
             cases: {}   agreement: {}/{} = {:.1}%   mean similarity: {:.3}",
            self.kind,
            self.n,
            self.agree,
            self.n,
            self.agreement() * 100.0,
            self.similarity_sum / self.n.max(1) as f64
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

    /// Share of checks that agree exactly. `0.0` with no check.
    pub fn agreement(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.agree as f64 / self.n as f64
    }

    pub fn kind(&self) -> &'static str {
        self.kind
    }

    pub fn violations(&self) -> &[String] {
        &self.violations
    }
}

/// Sentences that carry no intent and must not change a decision.
pub const NOISE: &[&str] = &[
    " Merci d'avance.",
    " On en reparle à la prochaine session.",
    " C'est urgent.",
];

/// Surface punctuation noise: the same message with its punctuation changed.
/// Each variant keeps every word and only moves or repeats the marks.
pub fn punctuation_noise(message: &str) -> Vec<String> {
    vec![
        format!("{message} !!!"),
        format!("{message}..."),
        message.replace('?', "").replace('!', "").trim().to_string(),
    ]
}
