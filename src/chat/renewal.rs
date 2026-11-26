//! Cognitive renewal of a logical PO session across technical provider sessions.
//!
//! A PO conversation must survive the closing of the technical session that
//! carries it (a Claude Code CLI session, a native thread, a Codex thread...).
//! The provider owns that session and its history; PO owns the *logical* one.
//! Renewal is therefore not a compaction: nothing is rewritten in place, the
//! technical session is closed at a safe point and a new one is opened from a
//! versioned cognitive checkpoint.
//!
//! Four distinct objects, never to be confused:
//! - the **logical session** ([`LogicalSessionId`]): stable, provider-agnostic;
//! - the **[`CognitiveCheckpoint`]**: the working memory worth carrying over;
//! - the **audit transcript**: what actually happened, never rewritten;
//! - the **technical session**: the provider's, disposable.
//!
//! Everything here is pure and testable without a provider: deciding whether
//! renewal is *safe*, what a checkpoint carries, and what a reconstruction
//! drops. Wiring it into the chat manager is deliberately a separate step.

use serde::{Deserialize, Serialize};

/// Identity of a logical session. Never a provider identifier: a CLI session id
/// or a `thread_id` dies with its technical session, this one does not.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LogicalSessionId(pub String);

/// How many technical sessions this logical session has already consumed.
///
/// Called an *epoch* so that an audit can say "third renewal" without knowing
/// which provider carried each one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub struct CognitiveEpoch(pub u32);

/// Why a renewal is refused. A boolean would hide which guard fired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenewalRefused {
    /// A tool call is still in flight: closing now would lose its result.
    ToolInFlight,
    /// A write may or may not have landed: the state is unknown.
    UncertainWrite,
    /// The caller asked for an epoch that is not the next one.
    EpochMismatch {
        expected: CognitiveEpoch,
        asked: CognitiveEpoch,
    },
}

impl std::fmt::Display for RenewalRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ToolInFlight => write!(f, "a tool call is in flight"),
            Self::UncertainWrite => write!(f, "a write has an unknown outcome"),
            Self::EpochMismatch { expected, asked } => {
                write!(f, "epoch {} asked while {} is next", asked.0, expected.0)
            }
        }
    }
}

/// A decision worth carrying past a renewal, short enough to survive a budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CarriedDecision {
    pub what: String,
    pub why: String,
}

/// The working memory of a logical session at one point in time.
///
/// Everything in it is *selected* by PO: the transcript already holds the whole
/// history, this holds what a fresh session needs to go on working.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CognitiveCheckpoint {
    /// Bumped on every write; a checkpoint is never edited in place.
    pub version: u32,
    pub epoch: CognitiveEpoch,
    pub objective: String,
    pub constraints: Vec<String>,
    pub decisions: Vec<CarriedDecision>,
    pub open_hypotheses: Vec<String>,
    /// Paths, hashes or handles: the proof itself lives in Obrain or on disk.
    pub evidence: Vec<String>,
    /// Operations started and not confirmed finished.
    pub pending_operations: Vec<String>,
}

impl CognitiveCheckpoint {
    /// A checkpoint that claims to be new but has no objective carries nothing:
    /// refuse it rather than hand an empty memory to the next session.
    pub fn is_usable(&self) -> bool {
        !self.objective.trim().is_empty()
    }

    /// Next version of this checkpoint, for an append-only history.
    pub fn next_version(&self, epoch: CognitiveEpoch) -> Self {
        let mut next = self.clone();
        next.version += 1;
        next.epoch = epoch;
        next
    }
}

/// Decides whether the technical session may be closed now.
///
/// Both guards are conservative: an unknown write counts as unsafe.
pub fn safe_point(tool_in_flight: bool, uncertain_write: bool) -> Result<(), RenewalRefused> {
    if tool_in_flight {
        return Err(RenewalRefused::ToolInFlight);
    }
    if uncertain_write {
        return Err(RenewalRefused::UncertainWrite);
    }
    Ok(())
}

/// What a new technical session is seeded with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reconstructed {
    /// Kept items, in the order they must be presented.
    pub kept: Vec<String>,
    /// Items dropped to fit the budget, named so nothing vanishes silently.
    pub omitted: Vec<String>,
}

/// Rebuilds a working context from a checkpoint under a token budget.
///
/// The **objective** and the **constraints** are mandatory: they are always
/// kept, whatever the budget, because a fresh session that no longer knows
/// what it is working toward nor under which rules it runs has lost the logical
/// session itself. The budget governs only the optional memory — decisions,
/// pending operations, evidence, open hypotheses — ordered by decreasing
/// necessity. Nothing optional is dropped without being listed in
/// [`Reconstructed::omitted`]; the mandatory items are never listed there.
pub fn reconstruct(
    checkpoint: &CognitiveCheckpoint,
    budget_tokens: usize,
    cost_of: impl Fn(&str) -> usize,
) -> Reconstructed {
    // Mandatory context: never gated by the budget, never omitted.
    let mut kept: Vec<String> = Vec::new();
    kept.push(checkpoint.objective.clone());
    kept.extend(checkpoint.constraints.iter().cloned());

    // Optional context, gated by the budget and named when dropped.
    let mut candidates: Vec<String> = Vec::new();
    candidates.extend(
        checkpoint
            .decisions
            .iter()
            .map(|d| format!("decision: {} — {}", d.what, d.why)),
    );
    candidates.extend(
        checkpoint
            .pending_operations
            .iter()
            .map(|op| format!("pending: {op}")),
    );
    candidates.extend(checkpoint.evidence.iter().map(|e| format!("evidence: {e}")));
    candidates.extend(
        checkpoint
            .open_hypotheses
            .iter()
            .map(|h| format!("hypothesis: {h}")),
    );

    let mut omitted = Vec::new();
    let mut used = 0usize;
    for item in candidates {
        let cost = cost_of(&item);
        if used + cost <= budget_tokens {
            used += cost;
            kept.push(item);
        } else {
            omitted.push(item);
        }
    }
    Reconstructed { kept, omitted }
}

/// One renewal, kept for audit: who was closed, who replaced it, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenewalAudit {
    pub logical_session: LogicalSessionId,
    pub from_epoch: CognitiveEpoch,
    pub to_epoch: CognitiveEpoch,
    /// Provider session that was closed (audit only, never used as identity).
    pub closed_technical_session: Option<String>,
    pub reason: String,
    pub checkpoint_version: u32,
    pub omitted_items: usize,
}

/// Operations the session has started, so a resumed session neither redoes a
/// finished one nor replays one whose outcome is unknown.
///
/// Keys are caller-chosen and stable (for example `commit:<sha>` or
/// `deploy:<env>:<version>`). Three states:
/// - **done**: the operation finished and its result is recorded;
/// - **uncertain**: it started but the outcome is unknown (the write may or may
///   not have landed) — it must be *reconciled*, never automatically replayed;
/// - anything else: never started.
///
/// A ledger that only remembers `done` is not enough: an external operation can
/// succeed before its state is persisted, and a naive "redo everything not
/// done" would then execute it twice. The uncertain set makes that window
/// explicit and forces a reconciliation before resuming.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdempotencyLedger {
    done: std::collections::BTreeSet<String>,
    uncertain: std::collections::BTreeSet<String>,
}

impl IdempotencyLedger {
    /// Record a finished operation.
    pub fn record(&mut self, key: impl Into<String>) {
        let key = key.into();
        self.done.insert(key.clone());
        self.uncertain.remove(&key);
    }

    /// Record that an operation started but its outcome is unknown.
    pub fn mark_uncertain(&mut self, key: impl Into<String>) {
        let key = key.into();
        if !self.done.contains(&key) {
            self.uncertain.insert(key);
        }
    }

    /// Reconcile an uncertain operation: it either did land (`succeeded`) and
    /// moves to done, or did not, and is removed so it may be retried.
    pub fn reconcile(&mut self, key: &str, succeeded: bool) {
        self.uncertain.remove(key);
        if succeeded {
            self.done.insert(key.to_string());
        }
    }

    pub fn already_done(&self, key: &str) -> bool {
        self.done.contains(key)
    }

    pub fn is_uncertain(&self, key: &str) -> bool {
        self.uncertain.contains(key)
    }

    /// Operations whose outcome is unknown and must be reconciled before a
    /// resumed session may treat them as not-done.
    pub fn uncertain_keys(&self) -> Vec<&str> {
        self.uncertain.iter().map(String::as_str).collect()
    }

    pub fn len(&self) -> usize {
        self.done.len()
    }

    pub fn is_empty(&self) -> bool {
        self.done.is_empty() && self.uncertain.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint() -> CognitiveCheckpoint {
        CognitiveCheckpoint {
            version: 3,
            epoch: CognitiveEpoch(1),
            objective: "reduce the context cost of long sessions".into(),
            constraints: vec!["never rewrite the audit transcript".into()],
            decisions: vec![CarriedDecision {
                what: "digests replace raw tool output".into(),
                why: "measured 66-82% of context growth".into(),
            }],
            open_hypotheses: vec!["artifacts are injected by PO".into()],
            evidence: vec!["note:7b75636f".into()],
            pending_operations: vec![],
        }
    }

    #[test]
    fn a_renewal_stops_on_a_tool_in_flight() {
        assert_eq!(safe_point(true, false), Err(RenewalRefused::ToolInFlight));
        assert_eq!(safe_point(false, true), Err(RenewalRefused::UncertainWrite));
        assert_eq!(safe_point(false, false), Ok(()));
    }

    #[test]
    fn a_checkpoint_without_an_objective_is_not_usable() {
        let mut c = checkpoint();
        assert!(c.is_usable());
        c.objective = "   ".into();
        assert!(!c.is_usable(), "une memoire vide ne doit pas passer");
    }

    #[test]
    fn a_reconstruction_under_budget_names_what_it_drops() {
        let c = checkpoint();
        let full = reconstruct(&c, 10_000, |s| s.len() / 4);
        assert!(full.omitted.is_empty());
        assert_eq!(full.kept.len(), 5);

        let tight = reconstruct(&c, 20, |s| s.len() / 4);
        assert!(!tight.omitted.is_empty(), "un budget serre doit omettre");
        assert_eq!(
            tight.kept.len() + tight.omitted.len(),
            5,
            "rien ne disparait sans etre nomme"
        );
    }

    #[test]
    fn the_objective_is_kept_before_anything_else() {
        let c = checkpoint();
        let tight = reconstruct(&c, 12, |s| s.len() / 4);
        assert_eq!(
            tight.kept.first().map(String::as_str),
            Some(c.objective.as_str())
        );
    }

    #[test]
    fn the_next_version_bumps_and_moves_the_epoch() {
        let next = checkpoint().next_version(CognitiveEpoch(2));
        assert_eq!(next.version, 4);
        assert_eq!(next.epoch, CognitiveEpoch(2));
    }

    #[test]
    fn a_resumed_session_does_not_redo_a_recorded_operation() {
        let mut ledger = IdempotencyLedger::default();
        assert!(!ledger.already_done("commit:abc"));
        ledger.record("commit:abc");
        assert!(ledger.already_done("commit:abc"));
        assert_eq!(ledger.len(), 1);
    }

    #[test]
    fn the_checkpoint_survives_a_round_trip_and_keeps_its_version() {
        let c = checkpoint();
        let back: CognitiveCheckpoint =
            serde_json::from_str(&serde_json::to_string(&c).expect("serialise")).expect("relit");
        assert_eq!(back, c);
        assert_eq!(back.version, 3);
    }

    #[test]
    fn every_refusal_is_named_and_the_ledger_reports_its_emptiness() {
        assert_eq!(
            RenewalRefused::EpochMismatch {
                expected: CognitiveEpoch(2),
                asked: CognitiveEpoch(1)
            }
            .to_string(),
            "epoch 1 asked while 2 is next"
        );
        assert_eq!(
            RenewalRefused::ToolInFlight.to_string(),
            "a tool call is in flight"
        );
        assert_eq!(
            RenewalRefused::UncertainWrite.to_string(),
            "a write has an unknown outcome"
        );

        let mut ledger = IdempotencyLedger::default();
        assert!(ledger.is_empty());
        ledger.record("x");
        assert!(!ledger.is_empty());
    }

    #[test]
    fn the_objective_and_constraints_survive_any_budget() {
        let c = checkpoint();
        let r = reconstruct(&c, 0, |s| s.len() / 4);
        assert!(r.kept.iter().any(|k| k == &c.objective), "objectif omis");
        assert!(
            r.kept.iter().any(|k| k == &c.constraints[0]),
            "contrainte omise"
        );
        assert!(
            !r.omitted
                .iter()
                .any(|k| k == &c.objective || k == &c.constraints[0]),
            "un item obligatoire figure dans les omis"
        );
    }

    #[test]
    fn pending_operations_are_carried_and_named() {
        let mut c = checkpoint();
        c.pending_operations = vec!["deploy:prod:1".into()];
        let r = reconstruct(&c, 10_000, |s| s.len() / 4);
        assert!(r.kept.iter().any(|k| k == "pending: deploy:prod:1"));
    }

    #[test]
    fn an_uncertain_operation_is_reconciled_not_replayed() {
        let mut ledger = IdempotencyLedger::default();
        ledger.mark_uncertain("deploy:prod:1");
        assert!(ledger.is_uncertain("deploy:prod:1"));
        assert!(!ledger.already_done("deploy:prod:1"));
        assert_eq!(ledger.uncertain_keys(), vec!["deploy:prod:1"]);

        ledger.reconcile("deploy:prod:1", true);
        assert!(ledger.already_done("deploy:prod:1"));
        assert!(!ledger.is_uncertain("deploy:prod:1"));
        assert!(ledger.uncertain_keys().is_empty());
    }

    #[test]
    fn a_failed_uncertain_operation_can_be_retried() {
        let mut ledger = IdempotencyLedger::default();
        ledger.mark_uncertain("write:file:x");
        ledger.reconcile("write:file:x", false);
        assert!(!ledger.already_done("write:file:x"));
        assert!(!ledger.is_uncertain("write:file:x"));
    }

    #[test]
    fn recording_clears_the_uncertain_state() {
        let mut ledger = IdempotencyLedger::default();
        ledger.mark_uncertain("commit:abc");
        ledger.record("commit:abc");
        assert!(ledger.already_done("commit:abc"));
        assert!(!ledger.is_uncertain("commit:abc"));
    }
}
