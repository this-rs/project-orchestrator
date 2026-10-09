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
use std::collections::HashMap;
use uuid::Uuid;

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

/// Evaluate a single note against the sharing policy (distillation path).
fn evaluate_note(note: &Note, policy: &SharingPolicy) -> ConsentDecision {
    // 1. Explicit per-note consent overrides everything
    match note.sharing_consent {
        SharingConsent::ExplicitAllow => return ConsentDecision::Allow,
        SharingConsent::ExplicitDeny => return ConsentDecision::Deny(DenialReason::ExplicitDeny),
        SharingConsent::PolicyAuto | SharingConsent::NotSet => {
            // Fall through to policy evaluation
        }
    }
    decide_by_policy(
        Some(&note.note_type.to_string()),
        Some(compute_shareability_score(note)),
        policy,
    )
}

/// The policy half of the decision, shared by the distillation gate
/// ([`evaluate_note`]) and the read predicate ([`may_read`]): type overrides,
/// then the sharing mode, then the score threshold in `Auto`.
///
/// `type_key` is the artifact type as spelled in `SharingPolicy::type_overrides`
/// (`None` for artifacts without a type: code nodes). `score` is the
/// shareability score (`None` when it cannot be computed: code nodes); in
/// `Auto` mode an unscorable artifact is refused (`InsufficientScore`) rather
/// than guessed.
pub(crate) fn decide_by_policy(
    type_key: Option<&str>,
    score: Option<f64>,
    policy: &SharingPolicy,
) -> ConsentDecision {
    // Type-level overrides
    if let Some(action) = type_key.and_then(|t| policy.type_overrides.get(t)) {
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

    // Global sharing mode
    match policy.mode {
        SharingMode::Manual => ConsentDecision::Deny(DenialReason::ManualRequired),
        SharingMode::Suggest => ConsentDecision::PendingReview,
        SharingMode::Auto => match score {
            Some(s) if s >= policy.min_shareability_score => ConsentDecision::Allow,
            _ => ConsentDecision::Deny(DenialReason::InsufficientScore),
        },
    }
}

// ============================================================================
// Read predicate: the single consent rule for chat reads and propagation
// ============================================================================
//
// Consent protects SHARING BEYOND THE PROJECT. It never conditions the read of
// a note by a conversation of its own project ("do not share" != "do not use
// locally").
//
// Decision table (first matching row wins)
//
// | scope        | owner known | consent                | owner sharing          | verdict |
// |--------------|-------------|------------------------|------------------------|---------|
// | SameProject  | -           | any (incl. Deny)       | any                    | Allow   |
// | CrossProject | no          | -                      | -                      | Deny(OwnerUnknown) + warn |
// | CrossProject | yes         | ExplicitAllow          | any                    | Allow   |
// | CrossProject | yes         | ExplicitDeny           | any                    | Deny(ExplicitDeny) |
// | CrossProject | yes         | NotSet / PolicyAuto(*) | Unreadable             | Deny(PolicyUnreadable) + warn |
// | CrossProject | yes         | NotSet / PolicyAuto(*) | NoPolicy               | Deny(NoPolicy) |
// | CrossProject | yes         | NotSet / PolicyAuto(*) | Policy{enabled=false}  | Deny(SharingDisabled) |
// | CrossProject | yes         | NotSet / PolicyAuto(*) | Policy{enabled=true}   | [`decide_by_policy`]: Allow only if the mode/type/score allow; PendingReview and Manual are NOT allowed |
//
// (*) `PolicyAuto` is treated like `NotSet`, exactly as `evaluate_note` does.
//
// Code nodes (File, Function, Component, Feature...) have no consent of their
// own: they are read with `consent = NotSet` and no type/score, so they
// inherit the owner project's policy (in `Auto` mode, no score exists, so they
// are refused). Global notes (no project_id) are outside this predicate: the
// rule of `PropagationScope` (#614) admits them always, they carry
// cross-cutting guidelines.

/// Relation between the reader (the conversation's project) and the owner of
/// the artifact being read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadScope {
    /// The reader is in the owner's project.
    SameProject,
    /// The artifact crosses a project boundary (or is out of scope).
    CrossProject,
}

impl ReadScope {
    /// Classify a read. `None` when the artifact is global (`owner = None`):
    /// global notes are not subject to consent (see module notes above).
    /// An unknown reader against an owned artifact is `CrossProject`.
    pub fn between(reader: Option<Uuid>, owner: Option<Uuid>) -> Option<Self> {
        match (reader, owner) {
            (_, None) => None,
            (Some(r), Some(o)) if r == o => Some(Self::SameProject),
            _ => Some(Self::CrossProject),
        }
    }
}

/// What is known about the sharing policy of the artifact's owner project.
#[derive(Debug, Clone, Copy)]
pub enum OwnerSharing<'a> {
    /// The owner project is not known.
    OwnerUnknown,
    /// The policy could not be read (store error, corrupt value).
    Unreadable,
    /// The owner has no policy (never configured).
    NoPolicy,
    /// The owner's policy.
    Policy(&'a SharingPolicy),
}

/// Why a cross-project read was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadDenial {
    OwnerUnknown,
    PolicyUnreadable,
    NoPolicy,
    /// The artifact (or note) says ExplicitDeny, or the policy refused it.
    Policy(DenialReason),
    /// Policy needs a human review: not an approval, so not readable.
    PendingReview,
}

/// Verdict of [`may_read`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadVerdict {
    Allow,
    Deny(ReadDenial),
}

impl ReadVerdict {
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow)
    }
}

/// Facts about the artifact that the policy needs. `NONE` for code nodes.
#[derive(Debug, Clone, Copy, Default)]
pub struct ArtifactFacts<'a> {
    /// Type as spelled in `SharingPolicy::type_overrides`.
    pub type_key: Option<&'a str>,
    /// Shareability score, when computable.
    pub score: Option<f64>,
}

impl ArtifactFacts<'static> {
    pub const NONE: ArtifactFacts<'static> = ArtifactFacts {
        type_key: None,
        score: None,
    };
}

/// THE consent predicate for reads (pure; see the decision table above).
/// Missing information always refuses, and logs a `warn`.
pub fn may_read(
    scope: ReadScope,
    consent: SharingConsent,
    owner: OwnerSharing<'_>,
    facts: ArtifactFacts<'_>,
) -> ReadVerdict {
    if scope == ReadScope::SameProject {
        return ReadVerdict::Allow;
    }
    if matches!(owner, OwnerSharing::OwnerUnknown) {
        tracing::warn!("consent: cross-project read refused, owner project unknown");
        return ReadVerdict::Deny(ReadDenial::OwnerUnknown);
    }
    match consent {
        SharingConsent::ExplicitAllow => ReadVerdict::Allow,
        SharingConsent::ExplicitDeny => {
            ReadVerdict::Deny(ReadDenial::Policy(DenialReason::ExplicitDeny))
        }
        SharingConsent::NotSet | SharingConsent::PolicyAuto => match owner {
            OwnerSharing::OwnerUnknown => unreachable!("handled above"),
            OwnerSharing::Unreadable => {
                tracing::warn!("consent: cross-project read refused, owner policy unreadable");
                ReadVerdict::Deny(ReadDenial::PolicyUnreadable)
            }
            OwnerSharing::NoPolicy => ReadVerdict::Deny(ReadDenial::NoPolicy),
            OwnerSharing::Policy(p) if !p.enabled => {
                ReadVerdict::Deny(ReadDenial::Policy(DenialReason::SharingDisabled))
            }
            OwnerSharing::Policy(p) => match decide_by_policy(facts.type_key, facts.score, p) {
                ConsentDecision::Allow => ReadVerdict::Allow,
                ConsentDecision::Deny(r) => ReadVerdict::Deny(ReadDenial::Policy(r)),
                ConsentDecision::PendingReview => ReadVerdict::Deny(ReadDenial::PendingReview),
            },
        },
    }
}

/// [`may_read`] for a note.
pub fn may_read_note(scope: ReadScope, note: &Note, owner: OwnerSharing<'_>) -> ReadVerdict {
    let type_key = note.note_type.to_string();
    // The score is only needed on the cross-project policy path.
    let facts = if scope == ReadScope::CrossProject {
        ArtifactFacts {
            type_key: Some(&type_key),
            score: Some(compute_shareability_score(note)),
        }
    } else {
        ArtifactFacts::default()
    };
    may_read(scope, note.sharing_consent, owner, facts)
}

/// [`may_read`] for a code node (File, Function, Component, Feature...): no
/// consent of its own, it inherits the owner project's policy.
pub fn may_read_code_node(scope: ReadScope, owner: OwnerSharing<'_>) -> ReadVerdict {
    may_read(scope, SharingConsent::NotSet, owner, ArtifactFacts::NONE)
}

/// Sharing situation of an owner project, as read from the store.
#[derive(Debug, Clone)]
pub enum OwnerPolicyState {
    Unreadable,
    NoPolicy,
    Policy(SharingPolicy),
}

impl OwnerPolicyState {
    pub fn as_owner_sharing(&self) -> OwnerSharing<'_> {
        match self {
            Self::Unreadable => OwnerSharing::Unreadable,
            Self::NoPolicy => OwnerSharing::NoPolicy,
            Self::Policy(p) => OwnerSharing::Policy(p),
        }
    }
}

/// Per-call cache of owner policies in front of the store, with the glue that
/// turns "reader project + artifact owner" into a [`ReadVerdict`]. This is the
/// entry point for propagation filters and for the anchor resolver (T3).
/// Never permissive: a store error is `Unreadable` (logged `warn`).
pub struct ConsentReader<'a> {
    store: &'a dyn crate::neo4j::GraphStore,
    reader: Option<Uuid>,
    cache: HashMap<Uuid, OwnerPolicyState>,
}

impl<'a> ConsentReader<'a> {
    pub fn new(store: &'a dyn crate::neo4j::GraphStore, reader: Option<Uuid>) -> Self {
        Self {
            store,
            reader,
            cache: HashMap::new(),
        }
    }

    async fn owner_state(&mut self, owner: Uuid) -> &OwnerPolicyState {
        if !self.cache.contains_key(&owner) {
            let state = match self.store.get_sharing_policy(owner).await {
                Ok(Some(p)) => OwnerPolicyState::Policy(p),
                Ok(None) => OwnerPolicyState::NoPolicy,
                Err(e) => {
                    tracing::warn!(
                        owner_project = %owner, error = %e,
                        "consent: sharing policy unreadable; cross-project reads refused"
                    );
                    OwnerPolicyState::Unreadable
                }
            };
            self.cache.insert(owner, state);
        }
        &self.cache[&owner]
    }

    /// Verdict for a note. Global notes (no project) are always `Allow` (#614).
    pub async fn note(&mut self, note: &Note) -> ReadVerdict {
        let Some(scope) = ReadScope::between(self.reader, note.project_id) else {
            return ReadVerdict::Allow;
        };
        match (scope, note.project_id) {
            (ReadScope::SameProject, _) => may_read_note(scope, note, OwnerSharing::NoPolicy),
            (_, Some(owner)) => {
                let state = self.owner_state(owner).await.clone();
                may_read_note(scope, note, state.as_owner_sharing())
            }
            (_, None) => may_read_note(scope, note, OwnerSharing::OwnerUnknown),
        }
    }

    /// Verdict for a code node owned by `owner` (`None` = owner unknown: refused).
    pub async fn code_node(&mut self, owner: Option<Uuid>) -> ReadVerdict {
        let Some(owner) = owner else {
            return may_read_code_node(ReadScope::CrossProject, OwnerSharing::OwnerUnknown);
        };
        match ReadScope::between(self.reader, Some(owner)) {
            Some(ReadScope::SameProject) => {
                may_read_code_node(ReadScope::SameProject, OwnerSharing::NoPolicy)
            }
            _ => {
                let state = self.owner_state(owner).await.clone();
                may_read_code_node(ReadScope::CrossProject, state.as_owner_sharing())
            }
        }
    }

    /// Verdict for a decision owned by `owner`.
    pub async fn decision(&mut self, consent: SharingConsent, owner: Option<Uuid>) -> ReadVerdict {
        let Some(owner) = owner else {
            return may_read(
                ReadScope::CrossProject,
                consent,
                OwnerSharing::OwnerUnknown,
                ArtifactFacts::NONE,
            );
        };
        match ReadScope::between(self.reader, Some(owner)) {
            Some(ReadScope::SameProject) => may_read(
                ReadScope::SameProject,
                consent,
                OwnerSharing::NoPolicy,
                ArtifactFacts::NONE,
            ),
            _ => {
                let state = self.owner_state(owner).await.clone();
                may_read(
                    ReadScope::CrossProject,
                    consent,
                    state.as_owner_sharing(),
                    ArtifactFacts::NONE,
                )
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
        assert!((0.0..=1.0).contains(&score), "Score {} out of range", score);
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

    // ------------------------------------------------------------------
    // Read predicate: the whole decision table
    // ------------------------------------------------------------------

    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::GraphStore;

    fn policy(enabled: bool, mode: SharingMode) -> SharingPolicy {
        SharingPolicy {
            mode,
            enabled,
            min_shareability_score: 0.5,
            ..SharingPolicy::default()
        }
    }

    const ALL_CONSENTS: [SharingConsent; 4] = [
        SharingConsent::NotSet,
        SharingConsent::ExplicitAllow,
        SharingConsent::ExplicitDeny,
        SharingConsent::PolicyAuto,
    ];

    #[test]
    fn read_same_project_is_always_allowed() {
        let p = policy(false, SharingMode::Manual);
        let owners = [
            OwnerSharing::OwnerUnknown,
            OwnerSharing::Unreadable,
            OwnerSharing::NoPolicy,
            OwnerSharing::Policy(&p),
        ];
        for c in ALL_CONSENTS {
            for o in owners {
                assert_eq!(
                    may_read(ReadScope::SameProject, c, o, ArtifactFacts::NONE),
                    ReadVerdict::Allow,
                    "{c:?} / {o:?}"
                );
            }
        }
    }

    #[test]
    fn read_cross_owner_unknown_is_refused_whatever_the_consent() {
        for c in ALL_CONSENTS {
            assert_eq!(
                may_read(
                    ReadScope::CrossProject,
                    c,
                    OwnerSharing::OwnerUnknown,
                    ArtifactFacts::NONE
                ),
                ReadVerdict::Deny(ReadDenial::OwnerUnknown)
            );
        }
    }

    #[test]
    fn read_cross_explicit_consent_beats_the_owner_policy() {
        let disabled = policy(false, SharingMode::Manual);
        let owners = [
            OwnerSharing::Unreadable,
            OwnerSharing::NoPolicy,
            OwnerSharing::Policy(&disabled),
        ];
        for o in owners {
            assert!(may_read(
                ReadScope::CrossProject,
                SharingConsent::ExplicitAllow,
                o,
                ArtifactFacts::NONE
            )
            .is_allow());
            assert_eq!(
                may_read(
                    ReadScope::CrossProject,
                    SharingConsent::ExplicitDeny,
                    o,
                    ArtifactFacts::NONE
                ),
                ReadVerdict::Deny(ReadDenial::Policy(DenialReason::ExplicitDeny))
            );
        }
        // Even an enabled auto policy cannot override a deny.
        let auto = policy(true, SharingMode::Auto);
        assert!(!may_read(
            ReadScope::CrossProject,
            SharingConsent::ExplicitDeny,
            OwnerSharing::Policy(&auto),
            ArtifactFacts {
                type_key: None,
                score: Some(1.0)
            }
        )
        .is_allow());
    }

    #[test]
    fn read_cross_notset_follows_the_owner_policy() {
        let facts = |s| ArtifactFacts {
            type_key: Some("tip"),
            score: Some(s),
        };
        for c in [SharingConsent::NotSet, SharingConsent::PolicyAuto] {
            let go = |o, f| may_read(ReadScope::CrossProject, c, o, f);
            assert_eq!(
                go(OwnerSharing::Unreadable, facts(1.0)),
                ReadVerdict::Deny(ReadDenial::PolicyUnreadable)
            );
            assert_eq!(
                go(OwnerSharing::NoPolicy, facts(1.0)),
                ReadVerdict::Deny(ReadDenial::NoPolicy)
            );
            let off = policy(false, SharingMode::Auto);
            assert_eq!(
                go(OwnerSharing::Policy(&off), facts(1.0)),
                ReadVerdict::Deny(ReadDenial::Policy(DenialReason::SharingDisabled))
            );
            let manual = policy(true, SharingMode::Manual);
            assert_eq!(
                go(OwnerSharing::Policy(&manual), facts(1.0)),
                ReadVerdict::Deny(ReadDenial::Policy(DenialReason::ManualRequired))
            );
            let suggest = policy(true, SharingMode::Suggest);
            assert_eq!(
                go(OwnerSharing::Policy(&suggest), facts(1.0)),
                ReadVerdict::Deny(ReadDenial::PendingReview)
            );
            let auto = policy(true, SharingMode::Auto);
            assert!(go(OwnerSharing::Policy(&auto), facts(0.5)).is_allow());
            assert_eq!(
                go(OwnerSharing::Policy(&auto), facts(0.49)),
                ReadVerdict::Deny(ReadDenial::Policy(DenialReason::InsufficientScore))
            );
            // No score (code node): refused, not guessed.
            assert_eq!(
                go(OwnerSharing::Policy(&auto), ArtifactFacts::NONE),
                ReadVerdict::Deny(ReadDenial::Policy(DenialReason::InsufficientScore))
            );
            // Type overrides apply as in the distillation gate.
            let mut never = policy(true, SharingMode::Auto);
            never
                .type_overrides
                .insert("tip".into(), SharingAction::Never);
            assert_eq!(
                go(OwnerSharing::Policy(&never), facts(1.0)),
                ReadVerdict::Deny(ReadDenial::Policy(DenialReason::TypeNeverPolicy))
            );
            let mut review = policy(true, SharingMode::Auto);
            review
                .type_overrides
                .insert("tip".into(), SharingAction::Review);
            assert_eq!(
                go(OwnerSharing::Policy(&review), facts(1.0)),
                ReadVerdict::Deny(ReadDenial::PendingReview)
            );
        }
    }

    #[test]
    fn read_scope_between_classifies_global_same_and_cross() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        assert_eq!(ReadScope::between(Some(a), None), None);
        assert_eq!(ReadScope::between(None, None), None);
        assert_eq!(
            ReadScope::between(Some(a), Some(a)),
            Some(ReadScope::SameProject)
        );
        assert_eq!(
            ReadScope::between(Some(a), Some(b)),
            Some(ReadScope::CrossProject)
        );
        assert_eq!(
            ReadScope::between(None, Some(b)),
            Some(ReadScope::CrossProject)
        );
    }

    #[test]
    fn code_nodes_inherit_the_owner_policy() {
        let manual = policy(true, SharingMode::Manual);
        assert!(may_read_code_node(ReadScope::SameProject, OwnerSharing::NoPolicy).is_allow());
        assert!(!may_read_code_node(ReadScope::CrossProject, OwnerSharing::NoPolicy).is_allow());
        assert!(
            !may_read_code_node(ReadScope::CrossProject, OwnerSharing::Policy(&manual)).is_allow()
        );
        assert!(
            !may_read_code_node(ReadScope::CrossProject, OwnerSharing::OwnerUnknown).is_allow()
        );
    }

    #[test]
    fn distillation_gate_is_unchanged_by_the_shared_policy_half() {
        // ExplicitAllow passes, ExplicitDeny blocks, NotSet follows the mode.
        let manual = policy(true, SharingMode::Manual);
        let notes = vec![
            make_test_note("tip", SharingConsent::ExplicitAllow),
            make_test_note("tip", SharingConsent::ExplicitDeny),
            make_test_note("tip", SharingConsent::NotSet),
        ];
        let (_, d) = run_consent_gate(&notes, &manual);
        assert_eq!(d[0].1, ConsentDecision::Allow);
        assert_eq!(d[1].1, ConsentDecision::Deny(DenialReason::ExplicitDeny));
        assert_eq!(d[2].1, ConsentDecision::Deny(DenialReason::ManualRequired));
    }

    // ------------------------------------------------------------------
    // ConsentReader + mock propagation (CrossProject path)
    // ------------------------------------------------------------------

    async fn owned_note(
        store: &MockGraphStore,
        project: Option<Uuid>,
        consent: SharingConsent,
        file: &str,
    ) -> Uuid {
        use crate::notes::models::{EntityType, NoteType};
        let mut n = Note::new(project, NoteType::Tip, "x".repeat(300), "t".into());
        n.sharing_consent = consent;
        store.create_note(&n).await.unwrap();
        store
            .link_note_to_entity(n.id, &EntityType::File, file, None, None)
            .await
            .unwrap();
        n.id
    }

    #[tokio::test]
    async fn propagation_cross_project_applies_consent_and_project_stays_closed() {
        use crate::notes::models::EntityType;
        let store = MockGraphStore::new();
        let me = Uuid::new_v4();
        let other = Uuid::new_v4();
        let f = "/f.rs";

        let local_deny = owned_note(&store, Some(me), SharingConsent::ExplicitDeny, f).await;
        let global = owned_note(&store, None, SharingConsent::ExplicitDeny, f).await;
        let allow = owned_note(&store, Some(other), SharingConsent::ExplicitAllow, f).await;
        let deny = owned_note(&store, Some(other), SharingConsent::ExplicitDeny, f).await;
        let notset = owned_note(&store, Some(other), SharingConsent::NotSet, f).await;

        let ids = |v: Vec<crate::notes::PropagatedNote>| {
            let mut s: Vec<Uuid> = v.into_iter().map(|p| p.note.id).collect();
            s.sort();
            s
        };
        let sorted = |mut v: Vec<Uuid>| {
            v.sort();
            v
        };

        // NotSet + no policy: refused. Same-project deny and global stay readable.
        let got = store
            .get_propagated_notes(&EntityType::File, f, 2, 0.0, None, Some(me), true)
            .await
            .unwrap();
        assert_eq!(ids(got), sorted(vec![local_deny, global, allow]));

        // Owner opts in (enabled + auto, low threshold): NotSet becomes readable,
        // ExplicitDeny never does.
        store
            .update_sharing_policy(
                other,
                &SharingPolicy {
                    min_shareability_score: 0.3,
                    ..policy(true, SharingMode::Auto)
                },
            )
            .await
            .unwrap();
        let got = store
            .get_propagated_notes(&EntityType::File, f, 2, 0.0, None, Some(me), true)
            .await
            .unwrap();
        let got = ids(got);
        assert!(got.contains(&notset));
        assert!(!got.contains(&deny));

        // `Project` scope never admits another project, consent or not.
        let got = store
            .get_propagated_notes(&EntityType::File, f, 2, 0.0, None, Some(me), false)
            .await
            .unwrap();
        assert_eq!(ids(got), sorted(vec![local_deny, global]));
    }

    #[tokio::test]
    async fn consent_reader_decisions_and_code_nodes() {
        let store = MockGraphStore::new();
        let me = Uuid::new_v4();
        let other = Uuid::new_v4();
        let mut r = ConsentReader::new(&store, Some(me));

        assert!(r.code_node(Some(me)).await.is_allow());
        assert!(!r.code_node(Some(other)).await.is_allow());
        assert!(!r.code_node(None).await.is_allow());
        assert!(r
            .decision(SharingConsent::ExplicitDeny, Some(me))
            .await
            .is_allow());
        assert!(!r
            .decision(SharingConsent::NotSet, Some(other))
            .await
            .is_allow());
        assert!(r
            .decision(SharingConsent::ExplicitAllow, Some(other))
            .await
            .is_allow());
    }

    #[tokio::test]
    async fn count_notes_by_consent_measures_the_population() {
        let store = MockGraphStore::new();
        let p = Uuid::new_v4();
        let q = Uuid::new_v4();
        for (proj, c) in [
            (Some(p), SharingConsent::NotSet),
            (Some(p), SharingConsent::NotSet),
            (Some(p), SharingConsent::ExplicitAllow),
            (Some(q), SharingConsent::ExplicitDeny),
            (None, SharingConsent::PolicyAuto),
        ] {
            owned_note(&store, proj, c, "/c.rs").await;
        }
        let all = store.count_notes_by_consent(None).await.unwrap();
        assert_eq!(
            (
                all.not_set,
                all.explicit_allow,
                all.explicit_deny,
                all.policy_auto
            ),
            (2, 1, 1, 1)
        );
        assert_eq!(all.total(), 5);
        let only_p = store.count_notes_by_consent(Some(p)).await.unwrap();
        assert_eq!(
            (only_p.not_set, only_p.explicit_allow, only_p.total()),
            (2, 1, 3)
        );
    }

    #[test]
    fn consent_db_str_roundtrip_and_default() {
        for c in ALL_CONSENTS {
            assert_eq!(SharingConsent::from_db_str(c.as_db_str()), c);
            assert_eq!(
                serde_json::to_string(&c).unwrap(),
                format!("\"{}\"", c.as_db_str())
            );
        }
        assert_eq!(SharingConsent::from_db_str(""), SharingConsent::NotSet);
        assert_eq!(SharingConsent::from_db_str("junk"), SharingConsent::NotSet);
    }

    #[test]
    fn decision_node_without_consent_deserializes_as_notset() {
        let d = crate::test_helpers::test_decision("d", "r");
        let mut v = serde_json::to_value(&d).unwrap();
        v.as_object_mut().unwrap().remove("sharing_consent");
        let back: crate::neo4j::models::DecisionNode = serde_json::from_value(v).unwrap();
        assert_eq!(back.sharing_consent, SharingConsent::NotSet);
    }
}
