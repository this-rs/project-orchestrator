//! Startup migration gate — the pieces PO itself owns for the source → target
//! storage-backend switch.
//!
//! The data copy (reading Neo4j, writing obrain) is the migrator's job and
//! lives outside this repository (the spike-migrateur). What PO owns, and what
//! this module provides, is the *guarding* and the *decision*:
//!
//! - a process-wide **read-only gate** ([`set_read_only`] / [`is_read_only`]):
//!   while a migration is in progress every write route (HTTP and MCP) must
//!   refuse with a named answer, so the source is never modified through PO;
//! - the **per-type verification** ([`PerTypeCounts`] + [`verify_by_type`]): the
//!   switch is only allowed once the source and the target agree, type by type
//!   (nodes and links);
//! - the **decision** ([`decide`]): flip to the target, or stay on the source
//!   with a named reason (rollback demanded by configuration, or a per-type
//!   mismatch).
//!
//! Everything here is pure and backend-agnostic: it is tested against in-memory
//! counts, never against a live store.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

/// The stable `code` a write refused during migration answers with, in the same
/// shape as the other named refusals of the API (`{ "code": … }`).
pub const READ_ONLY_CODE: &str = "migration_read_only";

/// The HTTP status of a write refused during migration: the client was not at
/// fault, the server is mid-switch (503 Service Unavailable).
pub const READ_ONLY_STATUS: u16 = 503;

// ── Process-wide read-only gate ────────────────────────────────────────────
//
// One gate per process: the switch happens once, at startup, before any route
// is served. A single `AtomicBool` (not a per-`ServerState` field) keeps the
// gate observable from the router middleware, the MCP dispatch and the startup
// orchestrator without threading a new field through every state constructor.

static READ_ONLY: AtomicBool = AtomicBool::new(false);

/// Enter (`true`) or leave (`false`) the read-only migration state.
pub fn set_read_only(value: bool) {
    READ_ONLY.store(value, Ordering::SeqCst);
}

/// Whether the process is currently read-only: every write route must refuse
/// with [`READ_ONLY_CODE`].
pub fn is_read_only() -> bool {
    READ_ONLY.load(Ordering::SeqCst)
}

// ── Per-type verification ───────────────────────────────────────────────────

/// Node and relationship counts by type, from one backend. The migrator fills
/// one for the source and one for the target; [`verify_by_type`] compares them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PerTypeCounts {
    /// Node count by label.
    pub nodes: BTreeMap<String, usize>,
    /// Relationship count by relationship type.
    pub relationships: BTreeMap<String, usize>,
}

impl PerTypeCounts {
    /// True when every label and every relationship type carries a count on
    /// both sides (i.e. no side is empty). Useful as a cheap guard against
    /// comparing a populated source to a target that was never written.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.relationships.is_empty()
    }
}

/// One per-type difference between the source and the target. A non-empty
/// result of [`verify_by_type`] means the switch must NOT happen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeMismatch {
    /// `"node"` or `"relationship"`.
    pub kind: &'static str,
    /// The label / relationship type that differs.
    pub type_name: String,
    /// Count on the source.
    pub source: usize,
    /// Count on the target.
    pub target: usize,
}

impl fmt::Display for TypeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} '{}' — source {} vs target {}",
            self.kind, self.type_name, self.source, self.target
        )
    }
}

/// Compare the source and the target per-type counts. The result is empty if
/// and only if every label and every relationship type agrees on both sides.
/// A label/type present on only one side is a mismatch (missing = count 0).
pub fn verify_by_type(source: &PerTypeCounts, target: &PerTypeCounts) -> Vec<TypeMismatch> {
    let mut mismatches = Vec::new();

    for (name, &src) in &source.nodes {
        match target.nodes.get(name) {
            Some(&tgt) if tgt == src => {}
            Some(&tgt) => mismatches.push(TypeMismatch {
                kind: "node",
                type_name: name.clone(),
                source: src,
                target: tgt,
            }),
            None => mismatches.push(TypeMismatch {
                kind: "node",
                type_name: name.clone(),
                source: src,
                target: 0,
            }),
        }
    }
    for (name, &tgt) in &target.nodes {
        if !source.nodes.contains_key(name) {
            mismatches.push(TypeMismatch {
                kind: "node",
                type_name: name.clone(),
                source: 0,
                target: tgt,
            });
        }
    }

    for (name, &src) in &source.relationships {
        match target.relationships.get(name) {
            Some(&tgt) if tgt == src => {}
            Some(&tgt) => mismatches.push(TypeMismatch {
                kind: "relationship",
                type_name: name.clone(),
                source: src,
                target: tgt,
            }),
            None => mismatches.push(TypeMismatch {
                kind: "relationship",
                type_name: name.clone(),
                source: src,
                target: 0,
            }),
        }
    }
    for (name, &tgt) in &target.relationships {
        if !source.relationships.contains_key(name) {
            mismatches.push(TypeMismatch {
                kind: "relationship",
                type_name: name.clone(),
                source: 0,
                target: tgt,
            });
        }
    }

    mismatches
}

// ── The flip decision ───────────────────────────────────────────────────────

/// What the startup does with a backend-switch request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationDecision {
    /// Switch to the target: the source was never modified, and (when asked)
    /// the per-type verification passed.
    Flip {
        /// Source backend name (e.g. `"default"`).
        source: String,
        /// Target backend name (e.g. `"obrain"`).
        target: String,
    },
    /// Stay on the source, and why.
    StayOnSource {
        source: String,
        target: String,
        /// A human-readable reason (rollback, or the per-type mismatches).
        reason: String,
    },
}

/// Decide the switch from the configuration and the verification result.
///
/// - `rollback` (a configuration variable) always stays on the source.
/// - `verify_by_type` + a non-empty [`verify_by_type`] result stays on the
///   source, naming every mismatch.
/// - Otherwise the switch is allowed.
pub fn decide(
    source: &str,
    target: &str,
    rollback: bool,
    verify_by_type: bool,
    mismatches: &[TypeMismatch],
) -> MigrationDecision {
    if rollback {
        return MigrationDecision::StayOnSource {
            source: source.to_string(),
            target: target.to_string(),
            reason: "rollback demanded by configuration — staying on the source".to_string(),
        };
    }
    if verify_by_type && !mismatches.is_empty() {
        return MigrationDecision::StayOnSource {
            source: source.to_string(),
            target: target.to_string(),
            reason: format!(
                "per-type verification failed ({} mismatch): {}",
                mismatches.len(),
                summarize(mismatches)
            ),
        };
    }
    MigrationDecision::Flip {
        source: source.to_string(),
        target: target.to_string(),
    }
}

/// A short, bounded summary of the mismatches for the logged reason.
fn summarize(mismatches: &[TypeMismatch]) -> String {
    const MAX: usize = 5;
    let mut parts: Vec<String> = mismatches.iter().take(MAX).map(|m| m.to_string()).collect();
    if mismatches.len() > MAX {
        parts.push(format!("… and {} more", mismatches.len() - MAX));
    }
    parts.join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(nodes: &[(&str, usize)], rels: &[(&str, usize)]) -> PerTypeCounts {
        PerTypeCounts {
            nodes: nodes.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            relationships: rels.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    #[test]
    fn identical_counts_verify_clean() {
        let a = counts(&[("Project", 3), ("Note", 7)], &[("LINKED_TO", 4)]);
        let b = a.clone();
        assert!(verify_by_type(&a, &b).is_empty());
    }

    #[test]
    fn a_count_difference_is_named_per_type() {
        let a = counts(&[("Project", 3)], &[("LINKED_TO", 4)]);
        let b = counts(&[("Project", 2)], &[("LINKED_TO", 4)]);
        let m = verify_by_type(&a, &b);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].kind, "node");
        assert_eq!(m[0].type_name, "Project");
        assert_eq!(m[0].source, 3);
        assert_eq!(m[0].target, 2);
    }

    #[test]
    fn a_type_missing_on_one_side_is_a_mismatch() {
        let a = counts(&[("Project", 3)], &[("LINKED_TO", 4)]);
        let b = counts(
            &[("Project", 3), ("Note", 1)],
            &[("LINKED_TO", 4), ("DEPENDS_ON", 2)],
        );
        let m = verify_by_type(&a, &b);
        assert_eq!(m.len(), 2, "{m:?}");
        assert!(m
            .iter()
            .any(|x| x.kind == "node" && x.type_name == "Note" && x.source == 0));
        assert!(m
            .iter()
            .any(|x| x.kind == "relationship" && x.type_name == "DEPENDS_ON" && x.source == 0));
    }

    #[test]
    fn empty_counts_are_not_empty_guard() {
        assert!(PerTypeCounts::default().is_empty());
        assert!(!counts(&[("Project", 1)], &[]).is_empty());
    }

    #[test]
    fn decide_flips_when_verification_passes_and_no_rollback() {
        let d = decide("default", "obrain", false, true, &[]);
        assert_eq!(
            d,
            MigrationDecision::Flip {
                source: "default".into(),
                target: "obrain".into(),
            }
        );
    }

    #[test]
    fn decide_stays_on_source_on_a_mismatch_and_names_it() {
        let m = vec![TypeMismatch {
            kind: "node",
            type_name: "Project".into(),
            source: 3,
            target: 2,
        }];
        match decide("default", "obrain", false, true, &m) {
            MigrationDecision::StayOnSource { source, reason, .. } => {
                assert_eq!(source, "default");
                assert!(
                    reason.contains("Project"),
                    "reason must name the type: {reason}"
                );
                assert!(
                    reason.contains("3"),
                    "reason must name the source count: {reason}"
                );
            }
            other => panic!("expected StayOnSource, got {other:?}"),
        }
    }

    #[test]
    fn decide_skips_verification_when_disabled_but_never_skips_rollback() {
        let m = vec![TypeMismatch {
            kind: "node",
            type_name: "Project".into(),
            source: 3,
            target: 2,
        }];
        assert!(matches!(
            decide("default", "obrain", false, false, &m),
            MigrationDecision::Flip { .. }
        ));
        assert!(matches!(
            decide("default", "obrain", true, true, &[]),
            MigrationDecision::StayOnSource { .. }
        ));
    }
}
