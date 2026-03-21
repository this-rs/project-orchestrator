//! # Protocol Evolution — Adaptive Mutation of Lifecycle Protocols
//!
//! When the learning loop detects recurring patterns (e.g., frequent test failures,
//! regression-prone tasks), this module proposes and applies mutations to the
//! project's lifecycle protocols so future runs benefit automatically.
//!
//! ## Mutation Rules
//!
//! | PatternType            | Gate keyword | Mutation                              |
//! |------------------------|-------------|---------------------------------------|
//! | FrequentGateFailure    | "test"      | Add `run_tests` state after implement |
//! | FrequentGateFailure    | "clippy"    | Add `lint_check` state after implement|
//! | FrequentGateFailure    | "check"     | Add `cargo_check` state after implement|
//! | RegressionProne        | —           | Add `review` state before done        |
//! | SuccessPattern         | "test_first"| Add `test_first` state before implement|
//!
//! ## Guardrails
//!
//! - Max 3 mutations per cycle
//! - Confidence threshold: 0.8
//! - Each mutation is idempotent (skipped if state already exists)
//! - Each mutation creates a Decision for traceability

use std::sync::Arc;

use anyhow::Result;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;
use crate::pipeline::feedback::{DetectedPattern, PatternType};
use crate::protocol::models::{ProtocolState, ProtocolTransition};

// ─── Constants ──────────────────────────────────────────────────────────────

/// Maximum mutations applied per learning cycle.
const MAX_MUTATIONS_PER_CYCLE: usize = 3;

/// Minimum confidence to trigger a protocol mutation.
const MIN_CONFIDENCE: f64 = 0.8;

// ─── Mutation Rules ─────────────────────────────────────────────────────────

/// A concrete mutation to apply to a protocol.
#[derive(Debug, Clone)]
pub struct MutationRule {
    /// Name of the new state to insert.
    pub state_name: String,
    /// Description for the new state.
    pub state_description: String,
    /// Where to insert: the new state goes AFTER this state.
    /// If None, inserted before the first terminal state.
    pub insert_after: Option<String>,
    /// Trigger name for the transition INTO the new state.
    pub trigger_in: String,
    /// Trigger name for the transition OUT of the new state.
    pub trigger_out: String,
    /// Human-readable rationale for this mutation.
    pub rationale: String,
}

/// Result of a single mutation attempt.
#[derive(Debug, Clone)]
pub struct MutationResult {
    /// The rule that was applied (or skipped).
    pub rule: MutationRule,
    /// Whether the mutation was applied (false = skipped, e.g., state already exists).
    pub applied: bool,
    /// Reason for skipping (if not applied).
    pub skip_reason: Option<String>,
    /// Decision ID created for traceability (if applied).
    pub decision_id: Option<Uuid>,
}

/// Result of a full evolution cycle.
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    /// Protocol that was mutated.
    pub protocol_id: Uuid,
    /// Individual mutation results.
    pub mutations: Vec<MutationResult>,
    /// How many were actually applied.
    pub applied_count: usize,
    /// How many were skipped (already existed or over limit).
    pub skipped_count: usize,
}

// ─── Pattern → Mutation mapping ─────────────────────────────────────────────

/// Derive mutation rules from a detected pattern.
///
/// Returns `None` if the pattern doesn't map to any known mutation.
pub fn pattern_to_mutations(pattern: &DetectedPattern) -> Vec<MutationRule> {
    match &pattern.pattern_type {
        PatternType::FrequentGateFailure => {
            // Check related_gates for keywords
            let gates_lower: Vec<String> = pattern
                .related_gates
                .iter()
                .map(|g| g.to_lowercase())
                .collect();

            let mut rules = Vec::new();

            if gates_lower.iter().any(|g| g.contains("test")) {
                rules.push(MutationRule {
                    state_name: "run_tests".to_string(),
                    state_description: "Auto-added: run tests after implementation (learned from frequent test failures)".to_string(),
                    insert_after: Some("implement".to_string()),
                    trigger_in: "implementation_done".to_string(),
                    trigger_out: "tests_passed".to_string(),
                    rationale: format!(
                        "Frequent test gate failures detected ({} occurrences, {:.0}% confidence). \
                         Adding mandatory test step to catch failures earlier.",
                        pattern.frequency,
                        pattern.confidence * 100.0
                    ),
                });
            }

            if gates_lower
                .iter()
                .any(|g| g.contains("clippy") || g.contains("lint"))
            {
                rules.push(MutationRule {
                    state_name: "lint_check".to_string(),
                    state_description: "Auto-added: run linter after implementation (learned from frequent lint failures)".to_string(),
                    insert_after: Some("implement".to_string()),
                    trigger_in: "implementation_done".to_string(),
                    trigger_out: "lint_passed".to_string(),
                    rationale: format!(
                        "Frequent lint/clippy failures detected ({} occurrences, {:.0}% confidence). \
                         Adding mandatory lint step.",
                        pattern.frequency,
                        pattern.confidence * 100.0
                    ),
                });
            }

            if gates_lower
                .iter()
                .any(|g| g.contains("check") || g.contains("compile"))
            {
                rules.push(MutationRule {
                    state_name: "cargo_check".to_string(),
                    state_description: "Auto-added: compilation check after implementation (learned from frequent build failures)".to_string(),
                    insert_after: Some("implement".to_string()),
                    trigger_in: "implementation_done".to_string(),
                    trigger_out: "check_passed".to_string(),
                    rationale: format!(
                        "Frequent compilation failures detected ({} occurrences, {:.0}% confidence). \
                         Adding mandatory check step.",
                        pattern.frequency,
                        pattern.confidence * 100.0
                    ),
                });
            }

            rules
        }

        PatternType::RegressionProne => {
            vec![MutationRule {
                state_name: "review".to_string(),
                state_description: "Auto-added: review step before completion (learned from regression-prone tasks)".to_string(),
                insert_after: None, // Will be inserted before the first terminal state
                trigger_in: "ready_for_review".to_string(),
                trigger_out: "review_approved".to_string(),
                rationale: format!(
                    "Regression-prone pattern detected ({} occurrences, {:.0}% confidence). \
                     Adding review step to catch regressions before completion.",
                    pattern.frequency,
                    pattern.confidence * 100.0
                ),
            }]
        }

        PatternType::SuccessPattern => {
            let desc_lower = pattern.description.to_lowercase();
            if desc_lower.contains("test_first") || desc_lower.contains("test first") {
                vec![MutationRule {
                    state_name: "test_first".to_string(),
                    state_description: "Auto-added: write tests before implementation (learned from successful test-first pattern)".to_string(),
                    insert_after: Some("plan".to_string()),
                    trigger_in: "planning_done".to_string(),
                    trigger_out: "tests_written".to_string(),
                    rationale: format!(
                        "Test-first success pattern detected ({} occurrences, {:.0}% confidence). \
                         Reinforcing by making test-first a mandatory step.",
                        pattern.frequency,
                        pattern.confidence * 100.0
                    ),
                }]
            } else {
                vec![]
            }
        }

        // These pattern types don't trigger protocol mutations
        PatternType::EffectiveRetry | PatternType::CommonRootCause => vec![],
    }
}

// ─── Protocol Evolver ───────────────────────────────────────────────────────

/// Applies learned mutations to a protocol FSM.
pub struct ProtocolEvolver {
    graph: Arc<dyn GraphStore>,
}

impl ProtocolEvolver {
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }

    /// Evolve a protocol based on detected patterns.
    ///
    /// Applies up to `MAX_MUTATIONS_PER_CYCLE` mutations, skipping patterns
    /// below `MIN_CONFIDENCE` and states that already exist (idempotent).
    ///
    /// `task_id` is used for creating Decision records (traceability).
    pub async fn evolve(
        &self,
        protocol_id: Uuid,
        patterns: &[DetectedPattern],
        task_id: Option<Uuid>,
    ) -> Result<EvolutionResult> {
        info!(
            protocol_id = %protocol_id,
            pattern_count = patterns.len(),
            "Starting protocol evolution cycle"
        );

        // Load current protocol states
        let existing_states = self.graph.get_protocol_states(protocol_id).await?;
        let existing_state_names: Vec<String> =
            existing_states.iter().map(|s| s.name.clone()).collect();

        let mut result = EvolutionResult {
            protocol_id,
            mutations: Vec::new(),
            applied_count: 0,
            skipped_count: 0,
        };

        // Collect all candidate mutations
        let mut candidate_rules: Vec<(MutationRule, &DetectedPattern)> = Vec::new();
        for pattern in patterns {
            // Filter by confidence
            if pattern.confidence < MIN_CONFIDENCE {
                debug!(
                    pattern_id = %pattern.id,
                    confidence = pattern.confidence,
                    "Skipping pattern below confidence threshold ({MIN_CONFIDENCE})"
                );
                continue;
            }

            let rules = pattern_to_mutations(pattern);
            for rule in rules {
                candidate_rules.push((rule, pattern));
            }
        }

        // Apply mutations (up to limit)
        for (rule, _pattern) in candidate_rules {
            if result.applied_count >= MAX_MUTATIONS_PER_CYCLE {
                result.mutations.push(MutationResult {
                    rule: rule.clone(),
                    applied: false,
                    skip_reason: Some(format!(
                        "Max mutations per cycle reached ({MAX_MUTATIONS_PER_CYCLE})"
                    )),
                    decision_id: None,
                });
                result.skipped_count += 1;
                continue;
            }

            // Idempotence check
            if existing_state_names.contains(&rule.state_name) {
                debug!(
                    state = %rule.state_name,
                    "State already exists in protocol — skipping"
                );
                result.mutations.push(MutationResult {
                    rule: rule.clone(),
                    applied: false,
                    skip_reason: Some(format!(
                        "State '{}' already exists in protocol",
                        rule.state_name
                    )),
                    decision_id: None,
                });
                result.skipped_count += 1;
                continue;
            }

            // Apply the mutation
            match self
                .apply_mutation(protocol_id, &rule, &existing_states, task_id)
                .await
            {
                Ok(decision_id) => {
                    info!(
                        state = %rule.state_name,
                        protocol_id = %protocol_id,
                        "Protocol mutation applied successfully"
                    );
                    result.mutations.push(MutationResult {
                        rule,
                        applied: true,
                        skip_reason: None,
                        decision_id,
                    });
                    result.applied_count += 1;
                }
                Err(e) => {
                    warn!(
                        state = %rule.state_name,
                        error = %e,
                        "Failed to apply protocol mutation"
                    );
                    result.mutations.push(MutationResult {
                        rule: rule.clone(),
                        applied: false,
                        skip_reason: Some(format!("Error: {e}")),
                        decision_id: None,
                    });
                    result.skipped_count += 1;
                }
            }
        }

        info!(
            protocol_id = %protocol_id,
            applied = result.applied_count,
            skipped = result.skipped_count,
            "Protocol evolution cycle complete"
        );

        Ok(result)
    }

    /// Apply a single mutation to a protocol.
    ///
    /// 1. Create the new intermediate state
    /// 2. Find the insertion point (after `insert_after` or before first terminal)
    /// 3. Rewire transitions: predecessor → new_state → successor
    /// 4. Create a Decision for traceability
    async fn apply_mutation(
        &self,
        protocol_id: Uuid,
        rule: &MutationRule,
        existing_states: &[ProtocolState],
        task_id: Option<Uuid>,
    ) -> Result<Option<Uuid>> {
        // 1. Create the new state
        let mut new_state = ProtocolState::new(protocol_id, &rule.state_name);
        new_state.description = rule.state_description.clone();
        self.graph.upsert_protocol_state(&new_state).await?;

        // 2. Find insertion point
        let (predecessor_id, successor_id) =
            self.find_insertion_point(existing_states, rule.insert_after.as_deref())?;

        // 3. Create transitions
        // Predecessor → new state
        let t_in =
            ProtocolTransition::new(protocol_id, predecessor_id, new_state.id, &rule.trigger_in);
        self.graph.upsert_protocol_transition(&t_in).await?;

        // New state → successor
        let t_out =
            ProtocolTransition::new(protocol_id, new_state.id, successor_id, &rule.trigger_out);
        self.graph.upsert_protocol_transition(&t_out).await?;

        // 4. Create Decision for traceability (if we have a task_id)
        let decision_id = if let Some(tid) = task_id {
            let decision = crate::neo4j::models::DecisionNode {
                id: Uuid::new_v4(),
                description: format!(
                    "Auto-evolved protocol: added '{}' state to protocol {}",
                    rule.state_name, protocol_id
                ),
                rationale: rule.rationale.clone(),
                alternatives: vec![
                    "No mutation — keep protocol as-is".to_string(),
                    "Manual protocol edit by developer".to_string(),
                ],
                chosen_option: Some("auto-evolved".to_string()),
                status: crate::neo4j::models::DecisionStatus::Accepted,
                decided_by: "learning-loop".to_string(),
                decided_at: chrono::Utc::now(),
                embedding: None,
                embedding_model: None,
                scar_intensity: 0.0,
            };
            self.graph.create_decision(tid, &decision).await?;
            Some(decision.id)
        } else {
            None
        };

        Ok(decision_id)
    }

    /// Find where to insert a new state in the protocol FSM.
    ///
    /// Returns `(predecessor_id, successor_id)`.
    fn find_insertion_point(
        &self,
        states: &[ProtocolState],
        insert_after: Option<&str>,
    ) -> Result<(Uuid, Uuid)> {
        use crate::protocol::models::StateType;

        if let Some(after_name) = insert_after {
            // Find the named state
            let predecessor = states
                .iter()
                .find(|s| s.name == after_name)
                .ok_or_else(|| {
                    anyhow::anyhow!("Cannot find state '{}' to insert after", after_name)
                })?;

            // Successor = first terminal state, or the state that typically follows
            let successor = states
                .iter()
                .find(|s| s.state_type == StateType::Terminal)
                .or_else(|| states.last())
                .ok_or_else(|| anyhow::anyhow!("Protocol has no states to use as successor"))?;

            Ok((predecessor.id, successor.id))
        } else {
            // Insert before the first terminal state
            let terminal = states
                .iter()
                .find(|s| s.state_type == StateType::Terminal)
                .ok_or_else(|| anyhow::anyhow!("Protocol has no terminal state"))?;

            // Predecessor = last non-terminal state
            let predecessor = states
                .iter()
                .rev()
                .find(|s| s.state_type != StateType::Terminal)
                .ok_or_else(|| {
                    anyhow::anyhow!("Protocol has no non-terminal states to insert before")
                })?;

            Ok((predecessor.id, terminal.id))
        }
    }
}

// ─── Auto-revert nocive mutations ───────────────────────────────────────────

/// Result of an auto-revert check.
#[derive(Debug, Clone)]
pub struct RevertResult {
    /// Protocol checked.
    pub protocol_id: Uuid,
    /// States that were reverted (removed).
    pub reverted_states: Vec<String>,
    /// Notes created explaining the revert.
    pub gotcha_note_ids: Vec<Uuid>,
}

/// Threshold: if failure rate increases by more than this after a mutation, revert it.
const FAILURE_RATE_INCREASE_THRESHOLD: f64 = 0.20;

/// Minimum runs to consider for comparison (need enough data).
const MIN_RUNS_FOR_COMPARISON: usize = 3;

impl ProtocolEvolver {
    /// Check if any auto-evolved states in a protocol are making things worse.
    ///
    /// Compares failure rate of recent runs. If a protocol with auto-evolved states
    /// has a failure rate > 20% higher than before, revert the mutation.
    ///
    /// `project_id` is used to create gotcha notes on revert.
    pub async fn check_and_revert(
        &self,
        protocol_id: Uuid,
        project_id: Uuid,
    ) -> Result<RevertResult> {
        let mut result = RevertResult {
            protocol_id,
            reverted_states: Vec::new(),
            gotcha_note_ids: Vec::new(),
        };

        // Get all runs for this protocol, ordered by creation
        let (runs, _) = self
            .graph
            .list_protocol_runs(protocol_id, None, 50, 0)
            .await?;

        if runs.len() < MIN_RUNS_FOR_COMPARISON * 2 {
            debug!(
                protocol_id = %protocol_id,
                run_count = runs.len(),
                "Not enough runs for auto-revert comparison (need {})",
                MIN_RUNS_FOR_COMPARISON * 2
            );
            return Ok(result);
        }

        // Split runs into two halves: older (before) and newer (after)
        let midpoint = runs.len() / 2;
        let older_runs = &runs[midpoint..]; // list_protocol_runs returns newest first
        let newer_runs = &runs[..midpoint];

        let older_failure_rate = compute_failure_rate(older_runs);
        let newer_failure_rate = compute_failure_rate(newer_runs);

        let increase = newer_failure_rate - older_failure_rate;

        if increase <= FAILURE_RATE_INCREASE_THRESHOLD {
            debug!(
                protocol_id = %protocol_id,
                older_rate = format!("{:.0}%", older_failure_rate * 100.0),
                newer_rate = format!("{:.0}%", newer_failure_rate * 100.0),
                "Failure rate not significantly worse — no revert needed"
            );
            return Ok(result);
        }

        warn!(
            protocol_id = %protocol_id,
            older_rate = format!("{:.0}%", older_failure_rate * 100.0),
            newer_rate = format!("{:.0}%", newer_failure_rate * 100.0),
            increase = format!("{:.0}%", increase * 100.0),
            "Failure rate increased significantly — reverting auto-evolved states"
        );

        // Find and remove auto-evolved states
        let states = self.graph.get_protocol_states(protocol_id).await?;
        for state in &states {
            if state.description.starts_with("Auto-added:") {
                // Remove transitions pointing to/from this state
                let transitions = self.graph.get_protocol_transitions(protocol_id).await?;
                for t in &transitions {
                    if t.from_state == state.id || t.to_state == state.id {
                        self.graph.delete_protocol_transition(t.id).await?;
                    }
                }

                // Remove the state itself
                self.graph.delete_protocol_state(state.id).await?;

                info!(
                    state = %state.name,
                    protocol_id = %protocol_id,
                    "Reverted auto-evolved state"
                );

                // Create a gotcha note
                let mut note = crate::notes::models::Note::new(
                    Some(project_id),
                    crate::notes::models::NoteType::Gotcha,
                    format!(
                        "Auto-revert: removed '{}' state from protocol. \
                         Failure rate increased from {:.0}% to {:.0}% (+{:.0}%) after this mutation was applied. \
                         Original description: {}",
                        state.name,
                        older_failure_rate * 100.0,
                        newer_failure_rate * 100.0,
                        increase * 100.0,
                        state.description,
                    ),
                    "learning-loop".to_string(),
                );
                note.importance = crate::notes::models::NoteImportance::High;
                note.tags = vec!["auto-learned".to_string(), "auto-revert".to_string()];
                note.memory_horizon = crate::notes::models::MemoryHorizon::Ephemeral;
                note.scar_intensity = 0.8;

                if self.graph.create_note(&note).await.is_ok() {
                    result.gotcha_note_ids.push(note.id);
                }

                result.reverted_states.push(state.name.clone());
            }
        }

        Ok(result)
    }
}

/// Compute the failure rate from a slice of protocol runs.
fn compute_failure_rate(runs: &[crate::protocol::models::ProtocolRun]) -> f64 {
    if runs.is_empty() {
        return 0.0;
    }
    let failed = runs
        .iter()
        .filter(|r| r.status == crate::protocol::models::RunStatus::Failed)
        .count();
    failed as f64 / runs.len() as f64
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pattern(
        pattern_type: PatternType,
        confidence: f64,
        related_gates: Vec<&str>,
        description: &str,
    ) -> DetectedPattern {
        DetectedPattern {
            id: format!("test-{}", Uuid::new_v4()),
            pattern_type,
            description: description.to_string(),
            frequency: 5,
            confidence,
            tech_stacks: vec!["rust".to_string()],
            related_gates: related_gates.into_iter().map(String::from).collect(),
            recommendation: "test recommendation".to_string(),
            affected_files: vec![],
        }
    }

    #[test]
    fn frequent_test_failure_maps_to_run_tests() {
        let pattern = make_pattern(
            PatternType::FrequentGateFailure,
            0.9,
            vec!["cargo test"],
            "Tests fail frequently",
        );
        let rules = pattern_to_mutations(&pattern);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].state_name, "run_tests");
        assert_eq!(rules[0].insert_after, Some("implement".to_string()));
    }

    #[test]
    fn frequent_clippy_failure_maps_to_lint_check() {
        let pattern = make_pattern(
            PatternType::FrequentGateFailure,
            0.9,
            vec!["cargo clippy"],
            "Clippy fails frequently",
        );
        let rules = pattern_to_mutations(&pattern);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].state_name, "lint_check");
    }

    #[test]
    fn regression_prone_maps_to_review() {
        let pattern = make_pattern(
            PatternType::RegressionProne,
            0.85,
            vec![],
            "Tasks cause regressions",
        );
        let rules = pattern_to_mutations(&pattern);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].state_name, "review");
        assert_eq!(rules[0].insert_after, None); // before terminal
    }

    #[test]
    fn success_pattern_test_first() {
        let pattern = make_pattern(
            PatternType::SuccessPattern,
            0.9,
            vec![],
            "test_first approach leads to success",
        );
        let rules = pattern_to_mutations(&pattern);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].state_name, "test_first");
    }

    #[test]
    fn effective_retry_has_no_mutation() {
        let pattern = make_pattern(
            PatternType::EffectiveRetry,
            0.95,
            vec!["build"],
            "Retrying build works",
        );
        let rules = pattern_to_mutations(&pattern);
        assert!(rules.is_empty());
    }

    #[test]
    fn common_root_cause_has_no_mutation() {
        let pattern = make_pattern(
            PatternType::CommonRootCause,
            0.9,
            vec![],
            "Missing import is common cause",
        );
        let rules = pattern_to_mutations(&pattern);
        assert!(rules.is_empty());
    }

    #[test]
    fn multiple_gates_produce_multiple_rules() {
        let pattern = make_pattern(
            PatternType::FrequentGateFailure,
            0.9,
            vec!["cargo test", "cargo clippy"],
            "Both test and clippy fail",
        );
        let rules = pattern_to_mutations(&pattern);
        assert_eq!(rules.len(), 2);
        let names: Vec<&str> = rules.iter().map(|r| r.state_name.as_str()).collect();
        assert!(names.contains(&"run_tests"));
        assert!(names.contains(&"lint_check"));
    }

    #[test]
    fn low_confidence_unrelated_success_no_mutation() {
        let pattern = make_pattern(
            PatternType::SuccessPattern,
            0.9,
            vec![],
            "fast iteration is good",
        );
        let rules = pattern_to_mutations(&pattern);
        assert!(rules.is_empty()); // no "test_first" keyword
    }

    // ── Async tests with MockGraphStore ──────────────────────────────────

    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::GraphStore;
    use crate::protocol::models::{Protocol, ProtocolState};

    /// Helper: set up a mock store with a protocol and start/implement/done states.
    /// Returns (store, protocol_id, start_state, implement_state, done_state).
    async fn setup_protocol() -> (
        Arc<MockGraphStore>,
        Uuid,
        ProtocolState,
        ProtocolState,
        ProtocolState,
    ) {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        // Use a temporary id to build the protocol, then fix protocol_id on states
        let tmp_start = ProtocolState::start(Uuid::nil(), "start");
        let protocol = Protocol::new(project_id, "test-proto", tmp_start.id);
        let protocol_id = protocol.id;

        let start_state = ProtocolState::start(protocol_id, "start");
        let implement_state = ProtocolState::new(protocol_id, "implement");
        let done_state = ProtocolState::terminal(protocol_id, "done");

        store.upsert_protocol(&protocol).await.unwrap();
        store.upsert_protocol_state(&start_state).await.unwrap();
        store.upsert_protocol_state(&implement_state).await.unwrap();
        store.upsert_protocol_state(&done_state).await.unwrap();

        (
            Arc::new(store),
            protocol_id,
            start_state,
            implement_state,
            done_state,
        )
    }

    #[tokio::test]
    async fn evolve_creates_state_for_high_confidence_test_gate() {
        let (store, protocol_id, _, _, _) = setup_protocol().await;
        let evolver = ProtocolEvolver::new(store.clone());

        let patterns = vec![make_pattern(
            PatternType::FrequentGateFailure,
            0.9,
            vec!["test"],
            "Tests fail frequently",
        )];

        let result = evolver.evolve(protocol_id, &patterns, None).await.unwrap();

        assert_eq!(result.applied_count, 1);
        assert_eq!(result.skipped_count, 0);
        assert_eq!(result.mutations.len(), 1);
        assert!(result.mutations[0].applied);
        assert_eq!(result.mutations[0].rule.state_name, "run_tests");

        // Verify the state was actually created in the store
        let states = store.get_protocol_states(protocol_id).await.unwrap();
        let names: Vec<&str> = states.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"run_tests"));
    }

    #[tokio::test]
    async fn evolve_skips_patterns_below_min_confidence() {
        let (store, protocol_id, _, _, _) = setup_protocol().await;
        let evolver = ProtocolEvolver::new(store.clone());

        let patterns = vec![make_pattern(
            PatternType::FrequentGateFailure,
            0.5, // below MIN_CONFIDENCE (0.8)
            vec!["test"],
            "Low confidence pattern",
        )];

        let result = evolver.evolve(protocol_id, &patterns, None).await.unwrap();

        assert_eq!(result.applied_count, 0);
        assert_eq!(result.mutations.len(), 0); // filtered out before mutation stage

        // Verify no new state was created
        let states = store.get_protocol_states(protocol_id).await.unwrap();
        let names: Vec<&str> = states.iter().map(|s| s.name.as_str()).collect();
        assert!(!names.contains(&"run_tests"));
    }

    #[tokio::test]
    async fn evolve_is_idempotent_skips_existing_states() {
        let (store, protocol_id, _, _, _) = setup_protocol().await;
        let evolver = ProtocolEvolver::new(store.clone());

        let patterns = vec![make_pattern(
            PatternType::FrequentGateFailure,
            0.9,
            vec!["test"],
            "Tests fail frequently",
        )];

        // First evolution — should apply
        let result1 = evolver.evolve(protocol_id, &patterns, None).await.unwrap();
        assert_eq!(result1.applied_count, 1);

        // Second evolution — should skip (state already exists)
        let result2 = evolver.evolve(protocol_id, &patterns, None).await.unwrap();
        assert_eq!(result2.applied_count, 0);
        assert_eq!(result2.skipped_count, 1);
        assert!(result2.mutations[0]
            .skip_reason
            .as_ref()
            .unwrap()
            .contains("already exists"));
    }

    #[tokio::test]
    async fn evolve_respects_max_mutations_per_cycle() {
        let (store, protocol_id, _, _, _) = setup_protocol().await;
        let evolver = ProtocolEvolver::new(store.clone());

        // Create 4 patterns that each produce a distinct mutation rule.
        // FrequentGateFailure with "test", "clippy", "check" = 3 rules from one pattern,
        // plus a RegressionProne = 1 more rule. Total = 4, but max is 3.
        let patterns = vec![
            make_pattern(
                PatternType::FrequentGateFailure,
                0.95,
                vec!["test", "clippy", "check"],
                "Multiple gate failures",
            ),
            make_pattern(
                PatternType::RegressionProne,
                0.9,
                vec![],
                "Regression prone tasks",
            ),
        ];

        let result = evolver.evolve(protocol_id, &patterns, None).await.unwrap();

        assert_eq!(result.applied_count, 3); // MAX_MUTATIONS_PER_CYCLE
        assert!(result.skipped_count >= 1); // at least one skipped
                                            // The skipped one should mention the limit
        let skipped = result.mutations.iter().find(|m| !m.applied).unwrap();
        assert!(skipped
            .skip_reason
            .as_ref()
            .unwrap()
            .contains("Max mutations per cycle"));
    }

    #[tokio::test]
    async fn check_and_revert_returns_empty_when_not_enough_runs() {
        let (store, protocol_id, _, _, _) = setup_protocol().await;
        let evolver = ProtocolEvolver::new(store.clone());
        let project_id = Uuid::new_v4();

        // No runs at all — should return empty
        let result = evolver
            .check_and_revert(protocol_id, project_id)
            .await
            .unwrap();

        assert!(result.reverted_states.is_empty());
        assert!(result.gotcha_note_ids.is_empty());
    }

    #[tokio::test]
    async fn check_and_revert_no_revert_when_failure_rate_stable() {
        use crate::protocol::models::{ProtocolRun, RunStatus};

        let (store, protocol_id, start_state, _, _) = setup_protocol().await;
        let evolver = ProtocolEvolver::new(store.clone());
        let project_id = Uuid::new_v4();

        // Create 6 runs (>= MIN_RUNS_FOR_COMPARISON * 2 = 6), all completed (0% failure)
        for _ in 0..6 {
            let mut run = ProtocolRun::new(protocol_id, start_state.id, "start");
            run.status = RunStatus::Completed;
            store.create_protocol_run(&run).await.unwrap();
        }

        let result = evolver
            .check_and_revert(protocol_id, project_id)
            .await
            .unwrap();

        assert!(result.reverted_states.is_empty());
        assert!(result.gotcha_note_ids.is_empty());
    }
}
