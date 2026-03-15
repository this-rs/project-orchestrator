//! Consent gate — filters artifacts before distillation based on
//! project sharing policy and per-note consent.
//!
//! This is the privacy checkpoint that ensures only explicitly
//! allowed content enters the distillation pipeline.

use crate::episodes::distill_models::{
    ConsentStats, DenialDetail, DenialReason, SharingAction, SharingConsent, SharingMode,
    SharingPolicy,
};
use crate::notes::models::Note;

/// Outcome of the consent gate for a single note.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsentDecision {
    /// Allowed to proceed into distillation.
    Allow,
    /// Denied with a reason.
    Deny(DenialReason),
    /// Requires human review (suggest mode).
    PendingReview,
}

/// Evaluate the consent gate for a batch of notes against a project policy.
///
/// Returns [`ConsentStats`] summarising allowed/denied/pending counts,
/// and a list of (note, decision) pairs for downstream processing.
pub fn run_consent_gate<'a>(
    notes: &'a [Note],
    policy: &SharingPolicy,
) -> (ConsentStats, Vec<(&'a Note, ConsentDecision)>) {
    let mut stats = ConsentStats::default();
    let mut decisions = Vec::with_capacity(notes.len());

    // If sharing is globally disabled, deny everything
    if !policy.enabled {
        for note in notes {
            let detail = DenialDetail {
                artifact_id: note.id.to_string(),
                artifact_type: "note".to_string(),
                reason: DenialReason::SharingDisabled,
            };
            stats.consent_denied += 1;
            stats.denied_reasons.push(detail);
            decisions.push((note, ConsentDecision::Deny(DenialReason::SharingDisabled)));
        }
        return (stats, decisions);
    }

    for note in notes {
        let decision = evaluate_note(note, policy);
        match &decision {
            ConsentDecision::Allow => stats.consent_allowed += 1,
            ConsentDecision::Deny(reason) => {
                stats.consent_denied += 1;
                stats.denied_reasons.push(DenialDetail {
                    artifact_id: note.id.to_string(),
                    artifact_type: "note".to_string(),
                    reason: reason.clone(),
                });
            }
            ConsentDecision::PendingReview => stats.consent_pending += 1,
        }
        decisions.push((note, decision));
    }

    (stats, decisions)
}

/// Evaluate a single note against the sharing policy.
fn evaluate_note(note: &Note, policy: &SharingPolicy) -> ConsentDecision {
    // 1. Explicit per-note consent overrides everything
    match note.sharing_consent {
        SharingConsent::ExplicitAllow => return ConsentDecision::Allow,
        SharingConsent::ExplicitDeny => return ConsentDecision::Deny(DenialReason::ExplicitDeny),
        SharingConsent::PolicyAuto | SharingConsent::NotSet => {
            // Fall through to policy evaluation
        }
    }

    // 2. Check type-level overrides
    let note_type_str = note.note_type.to_string();
    if let Some(action) = policy.type_overrides.get(&note_type_str) {
        match action {
            SharingAction::Never => {
                return ConsentDecision::Deny(DenialReason::TypeNeverPolicy);
            }
            SharingAction::Review => {
                return ConsentDecision::PendingReview;
            }
            SharingAction::Auto => {
                // Fall through to score check
            }
        }
    }

    // 3. Evaluate based on global sharing mode
    match policy.mode {
        SharingMode::Manual => ConsentDecision::Deny(DenialReason::ManualRequired),
        SharingMode::Suggest => ConsentDecision::PendingReview,
        SharingMode::Auto => {
            // Auto mode: check shareability score against threshold
            let score = compute_shareability_score(note);
            if score >= policy.min_shareability_score {
                ConsentDecision::Allow
            } else {
                ConsentDecision::Deny(DenialReason::InsufficientScore)
            }
        }
    }
}

/// Compute a shareability score for a note (0.0 - 1.0).
///
/// Heuristic based on:
/// - Content length (longer = more context = higher score)
/// - Tag count (more tags = better categorized)
/// - Importance level
/// - Note type (guidelines and patterns are more shareable)
pub fn compute_shareability_score(note: &Note) -> f64 {
    let mut score = 0.0;

    // Content length factor (caps at 500 chars)
    let content_len = note.content.len() as f64;
    score += (content_len / 500.0).min(1.0) * 0.3;

    // Tag count factor (caps at 5 tags)
    let tag_count = note.tags.len() as f64;
    score += (tag_count / 5.0).min(1.0) * 0.2;

    // Importance factor
    use crate::notes::models::NoteImportance;
    score += match note.importance {
        NoteImportance::Critical => 0.3,
        NoteImportance::High => 0.25,
        NoteImportance::Medium => 0.15,
        NoteImportance::Low => 0.05,
    };

    // Note type bonus (patterns and guidelines are more universally useful)
    use crate::notes::models::NoteType;
    score += match note.note_type {
        NoteType::Guideline => 0.2,
        NoteType::Pattern => 0.2,
        NoteType::Gotcha => 0.15,
        NoteType::Tip => 0.15,
        NoteType::Observation => 0.1,
        NoteType::Rfc => 0.1,
        NoteType::Context => 0.05,
        NoteType::Assertion => 0.1,
    };

    score.clamp(0.0, 1.0)
}

// ============================================================================
// Pipeline integration
// ============================================================================

/// Run the consent gate and populate the report's consent_stats.
///
/// This is the integration point for the distillation pipeline:
/// it filters notes based on the project's sharing policy, writes
/// statistics into the `AnonymizationReport`, and returns only the
/// notes that are allowed to proceed.
pub fn apply_consent_gate_to_report<'a>(
    notes: &'a [Note],
    policy: &SharingPolicy,
    report: &mut crate::episodes::distill_models::AnonymizationReport,
) -> Vec<&'a Note> {
    let (stats, decisions) = run_consent_gate(notes, policy);
    report.consent_stats = Some(stats);

    decisions
        .into_iter()
        .filter_map(|(note, decision)| {
            if decision == ConsentDecision::Allow {
                Some(note)
            } else {
                None
            }
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::distill_models::{SharingAction, SharingConsent, SharingMode};
    use crate::notes::models::Note;
    use std::collections::HashMap;

    fn make_test_note(note_type: &str, consent: SharingConsent) -> Note {
        use crate::notes::models::NoteType;
        use std::str::FromStr;
        let nt = NoteType::from_str(note_type).unwrap_or(NoteType::Observation);
        let mut note = Note::new(
            None,
            nt,
            format!(
                "Test content for {} note with enough text to have a decent score",
                note_type
            ),
            "test-agent".to_string(),
        );
        note.tags = vec!["rust".to_string(), "testing".to_string()];
        note.sharing_consent = consent;
        note
    }

    fn auto_policy() -> SharingPolicy {
        SharingPolicy {
            mode: SharingMode::Auto,
            type_overrides: HashMap::new(),
            l3_scan_enabled: true,
            min_shareability_score: 0.5,
            enabled: true,
        }
    }

    #[test]
    fn test_disabled_policy_denies_all() {
        let policy = SharingPolicy {
            enabled: false,
            ..auto_policy()
        };
        let notes = vec![make_test_note("guideline", SharingConsent::NotSet)];
        let (stats, decisions) = run_consent_gate(&notes, &policy);
        assert_eq!(stats.consent_denied, 1);
        assert_eq!(stats.consent_allowed, 0);
        assert_eq!(
            decisions[0].1,
            ConsentDecision::Deny(DenialReason::SharingDisabled)
        );
    }

    #[test]
    fn test_explicit_allow_overrides_policy() {
        let policy = SharingPolicy {
            mode: SharingMode::Manual,
            enabled: true,
            ..auto_policy()
        };
        let notes = vec![make_test_note("guideline", SharingConsent::ExplicitAllow)];
        let (stats, _) = run_consent_gate(&notes, &policy);
        assert_eq!(stats.consent_allowed, 1);
    }

    #[test]
    fn test_explicit_deny_overrides_auto() {
        let policy = auto_policy();
        let notes = vec![make_test_note("guideline", SharingConsent::ExplicitDeny)];
        let (stats, decisions) = run_consent_gate(&notes, &policy);
        assert_eq!(stats.consent_denied, 1);
        assert_eq!(
            decisions[0].1,
            ConsentDecision::Deny(DenialReason::ExplicitDeny)
        );
    }

    #[test]
    fn test_type_never_policy() {
        let mut policy = auto_policy();
        policy
            .type_overrides
            .insert("gotcha".to_string(), SharingAction::Never);
        let notes = vec![make_test_note("gotcha", SharingConsent::NotSet)];
        let (stats, decisions) = run_consent_gate(&notes, &policy);
        assert_eq!(stats.consent_denied, 1);
        assert_eq!(
            decisions[0].1,
            ConsentDecision::Deny(DenialReason::TypeNeverPolicy)
        );
    }

    #[test]
    fn test_manual_mode_denies() {
        let policy = SharingPolicy {
            mode: SharingMode::Manual,
            enabled: true,
            ..auto_policy()
        };
        let notes = vec![make_test_note("guideline", SharingConsent::NotSet)];
        let (stats, decisions) = run_consent_gate(&notes, &policy);
        assert_eq!(stats.consent_denied, 1);
        assert_eq!(
            decisions[0].1,
            ConsentDecision::Deny(DenialReason::ManualRequired)
        );
    }

    #[test]
    fn test_suggest_mode_pending() {
        let policy = SharingPolicy {
            mode: SharingMode::Suggest,
            enabled: true,
            ..auto_policy()
        };
        let notes = vec![make_test_note("guideline", SharingConsent::NotSet)];
        let (stats, decisions) = run_consent_gate(&notes, &policy);
        assert_eq!(stats.consent_pending, 1);
        assert_eq!(decisions[0].1, ConsentDecision::PendingReview);
    }

    #[test]
    fn test_auto_mode_allows_high_score() {
        let policy = SharingPolicy {
            min_shareability_score: 0.3,
            ..auto_policy()
        };
        let notes = vec![make_test_note("guideline", SharingConsent::NotSet)];
        let (stats, _) = run_consent_gate(&notes, &policy);
        assert_eq!(stats.consent_allowed, 1);
    }

    #[test]
    fn test_auto_mode_denies_low_score() {
        let policy = SharingPolicy {
            min_shareability_score: 0.99,
            ..auto_policy()
        };
        let notes = vec![make_test_note("context", SharingConsent::NotSet)];
        let (stats, decisions) = run_consent_gate(&notes, &policy);
        assert_eq!(stats.consent_denied, 1);
        assert_eq!(
            decisions[0].1,
            ConsentDecision::Deny(DenialReason::InsufficientScore)
        );
    }

    #[test]
    fn test_shareability_score_range() {
        let note = make_test_note("guideline", SharingConsent::NotSet);
        let score = compute_shareability_score(&note);
        assert!(score >= 0.0 && score <= 1.0, "Score {} out of range", score);
    }

    #[test]
    fn test_mixed_batch() {
        let mut policy = auto_policy();
        policy.min_shareability_score = 0.3; // low threshold so pattern note passes
        let notes = vec![
            make_test_note("guideline", SharingConsent::ExplicitAllow),
            make_test_note("gotcha", SharingConsent::ExplicitDeny),
            make_test_note("pattern", SharingConsent::NotSet),
        ];
        let (stats, _) = run_consent_gate(&notes, &policy);
        assert_eq!(stats.consent_allowed, 2); // explicit allow + auto allow
        assert_eq!(stats.consent_denied, 1); // explicit deny
    }
}
