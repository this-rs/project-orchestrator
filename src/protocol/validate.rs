//! Protocol structural validation
//!
//! Provides a pure validation function for protocol FSM structure that can be used:
//! - At compose time (from `compose_protocol` handler)
//! - At validation time (against stored protocols)
//!
//! # PYSR-4: Structural validation rules
//!
//! 1. **Duplicate transitions**: Two transitions with the same `(from_state, trigger)` are
//!    non-deterministic and rejected.
//! 2. **Terminal state required**: At least one state must be terminal.
//! 3. **Reachability**: All states must be reachable from the start state (BFS on graph).
//!
//! The validator handles cycles (reprises, OnFailureStrategy) correctly via BFS.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use super::StateType;

// ============================================================================
// Violation types
// ============================================================================

/// A structural violation found in a protocol.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProtocolViolation {
    /// Two transitions share the same (from_state, trigger), making the FSM non-deterministic.
    DuplicateTransition { from_state: String, trigger: String },
    /// No terminal state exists in the protocol.
    NoTerminalState,
    /// A state is not reachable from the start state.
    UnreachableState { state: String },
}

impl std::fmt::Display for ProtocolViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateTransition {
                from_state,
                trigger,
            } => {
                write!(
                    f,
                    "duplicate transition: from '{}' with trigger '{}' appears multiple times (non-deterministic)",
                    from_state, trigger
                )
            }
            Self::NoTerminalState => {
                write!(f, "protocol must have at least one terminal state")
            }
            Self::UnreachableState { state } => {
                write!(f, "state '{}' is unreachable from the start state", state)
            }
        }
    }
}

// ============================================================================
// Input types (generic over name representation)
// ============================================================================

/// A state for validation purposes.
/// Only the name and type are needed for structural validation.
#[derive(Debug, Clone)]
pub struct ValidatableState<'a> {
    pub name: &'a str,
    pub state_type: StateType,
}

/// A transition for validation purposes.
/// Only the from/to state names and trigger are needed.
#[derive(Debug, Clone)]
pub struct ValidatableTransition<'a> {
    pub from_state: &'a str,
    pub to_state: &'a str,
    pub trigger: &'a str,
}

// ============================================================================
// Validation function
// ============================================================================

/// Validate the structural integrity of a protocol FSM.
///
/// # Arguments
/// - `states`: The states in the protocol (name + type)
/// - `transitions`: The transitions in the protocol (from + to + trigger)
///
/// # Returns
/// A list of violations found. Empty list means the protocol is valid.
///
/// # Notes
/// - Handles cycles correctly (BFS on a directed graph)
/// - Does NOT validate state existence (caller should do that first)
/// - Does NOT validate transition type constraints (from terminal, to start) — that's
///   done separately by `validate_transition_state_types`
pub fn validate_protocol_structure<'a>(
    states: &[ValidatableState<'a>],
    transitions: &[ValidatableTransition<'a>],
) -> Vec<ProtocolViolation> {
    let mut violations = Vec::new();

    // 1. Check for duplicate transitions (same from_state + trigger = non-deterministic)
    {
        let mut seen: HashSet<(&str, &str)> = HashSet::new();
        for t in transitions {
            let key = (t.from_state, t.trigger);
            if !seen.insert(key) {
                violations.push(ProtocolViolation::DuplicateTransition {
                    from_state: t.from_state.to_string(),
                    trigger: t.trigger.to_string(),
                });
            }
        }
    }

    // 2. Check that at least one terminal state exists
    let has_terminal = states.iter().any(|s| s.state_type == StateType::Terminal);
    if !has_terminal {
        violations.push(ProtocolViolation::NoTerminalState);
    }

    // 3. Check that all states are reachable from start
    if !states.is_empty() {
        // Find start state, or fall back to first state
        let start_name = states
            .iter()
            .find(|s| s.state_type == StateType::Start)
            .map(|s| s.name)
            .or_else(|| states.first().map(|s| s.name));

        if let Some(start) = start_name {
            // Build adjacency: from_state -> [to_states]
            let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
            for t in transitions {
                adj.entry(t.from_state).or_default().push(t.to_state);
            }

            // BFS from start
            let mut reachable: HashSet<&str> = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);
            reachable.insert(start);

            while let Some(current) = queue.pop_front() {
                if let Some(neighbors) = adj.get(current) {
                    for &next in neighbors {
                        if reachable.insert(next) {
                            queue.push_back(next);
                        }
                    }
                }
            }

            // Check all states are reachable
            for state in states {
                if !reachable.contains(state.name) {
                    violations.push(ProtocolViolation::UnreachableState {
                        state: state.name.to_string(),
                    });
                }
            }
        }
    }

    violations
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    fn state(name: &str, state_type: StateType) -> ValidatableState<'_> {
        ValidatableState { name, state_type }
    }

    fn transition<'a>(from: &'a str, to: &'a str, trigger: &'a str) -> ValidatableTransition<'a> {
        ValidatableTransition {
            from_state: from,
            to_state: to,
            trigger,
        }
    }

    #[test]
    fn valid_linear_protocol() {
        let states = vec![
            state("start", StateType::Start),
            state("work", StateType::Intermediate),
            state("done", StateType::Terminal),
        ];
        let transitions = vec![
            transition("start", "work", "begin"),
            transition("work", "done", "finish"),
        ];

        let violations = validate_protocol_structure(&states, &transitions);
        assert!(
            violations.is_empty(),
            "expected no violations: {:?}",
            violations
        );
    }

    #[test]
    fn valid_protocol_with_cycle() {
        // Cycles are allowed (e.g., retry loops)
        let states = vec![
            state("start", StateType::Start),
            state("work", StateType::Intermediate),
            state("done", StateType::Terminal),
        ];
        let transitions = vec![
            transition("start", "work", "begin"),
            transition("work", "work", "retry"), // cycle back to self
            transition("work", "done", "success"),
        ];

        let violations = validate_protocol_structure(&states, &transitions);
        assert!(
            violations.is_empty(),
            "cycles should be allowed: {:?}",
            violations
        );
    }

    #[test]
    fn duplicate_transition_detected() {
        let states = vec![
            state("start", StateType::Start),
            state("a", StateType::Intermediate),
            state("b", StateType::Intermediate),
            state("done", StateType::Terminal),
        ];
        let transitions = vec![
            transition("start", "a", "go"), // dup
            transition("start", "b", "go"), // dup
            transition("a", "done", "finish"),
            transition("b", "done", "finish"),
        ];

        let violations = validate_protocol_structure(&states, &transitions);
        assert_eq!(violations.len(), 1);
        assert!(matches!(
            &violations[0],
            ProtocolViolation::DuplicateTransition { from_state, trigger }
            if from_state == "start" && trigger == "go"
        ));
    }

    #[test]
    fn no_terminal_state_detected() {
        let states = vec![
            state("start", StateType::Start),
            state("work", StateType::Intermediate),
        ];
        let transitions = vec![transition("start", "work", "begin")];

        let violations = validate_protocol_structure(&states, &transitions);
        assert_eq!(violations.len(), 1);
        assert!(matches!(violations[0], ProtocolViolation::NoTerminalState));
    }

    #[test]
    fn unreachable_state_detected() {
        let states = vec![
            state("start", StateType::Start),
            state("done", StateType::Terminal),
            state("orphan", StateType::Intermediate), // unreachable
        ];
        let transitions = vec![transition("start", "done", "finish")];

        let violations = validate_protocol_structure(&states, &transitions);
        assert_eq!(violations.len(), 1);
        assert!(matches!(
            &violations[0],
            ProtocolViolation::UnreachableState { state } if state == "orphan"
        ));
    }

    #[test]
    fn multiple_violations_detected() {
        // No terminal AND unreachable state
        let states = vec![
            state("start", StateType::Start),
            state("work", StateType::Intermediate),
            state("orphan", StateType::Intermediate), // unreachable
        ];
        let transitions = vec![transition("start", "work", "begin")];

        let violations = validate_protocol_structure(&states, &transitions);
        assert_eq!(violations.len(), 2);

        let has_no_terminal = violations
            .iter()
            .any(|v| matches!(v, ProtocolViolation::NoTerminalState));
        let has_unreachable = violations.iter().any(
            |v| matches!(v, ProtocolViolation::UnreachableState { state } if state == "orphan"),
        );

        assert!(has_no_terminal, "should detect no terminal state");
        assert!(has_unreachable, "should detect unreachable state");
    }

    #[test]
    fn empty_protocol_no_violations() {
        // Edge case: no states, no transitions
        let violations = validate_protocol_structure(&[], &[]);
        // No terminal state is the only violation (empty is technically valid structurally
        // but has no terminal)
        assert_eq!(violations.len(), 1);
        assert!(matches!(violations[0], ProtocolViolation::NoTerminalState));
    }

    #[test]
    fn single_terminal_state_valid() {
        // Minimal valid protocol: just a terminal state
        let states = vec![state("done", StateType::Terminal)];
        let transitions = vec![];

        let violations = validate_protocol_structure(&states, &transitions);
        assert!(violations.is_empty());
    }
}

/// Tests against real Neo4j graph — run with `cargo test --lib protocol::validate::real_graph -- --ignored`
#[cfg(test)]
mod real_graph {
    use super::*;
    #[allow(unused_imports)]

    /// Validate ALL stored protocols against the real Neo4j graph.
    ///
    /// PYSR-4 criterion 4: prove that existing protocols pass structural validation.
    ///
    /// Run manually with:
    /// ```sh
    /// PARITY_NEO4J_URI=bolt://localhost:7687 \
    /// PARITY_NEO4J_USER=neo4j \
    /// PARITY_NEO4J_PASSWORD=<password> \
    /// cargo test --lib protocol::validate::real_graph::all_stored_protocols_pass -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore]
    async fn all_stored_protocols_pass() {
        use crate::neo4j::client::Neo4jClient;

        // Read Neo4j connection from environment (never print credentials)
        let required = std::env::var("HARNESS_NEO4J_REQUIRED").is_ok();
        let Ok(uri) = std::env::var("PARITY_NEO4J_URI") else {
            assert!(
                !required,
                "HARNESS_NEO4J_REQUIRED is set but PARITY_NEO4J_URI is not"
            );
            eprintln!("Skipping: PARITY_NEO4J_URI not set");
            return;
        };
        let user = std::env::var("PARITY_NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
        let pass = std::env::var("PARITY_NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into());

        // Connect with timeout
        let connect = Neo4jClient::new(&uri, &user, &pass);
        let client = match tokio::time::timeout(std::time::Duration::from_secs(30), connect).await {
            Ok(Ok(c)) => c,
            _ => {
                assert!(
                    !required,
                    "HARNESS_NEO4J_REQUIRED is set but no Neo4j answers at {uri}"
                );
                eprintln!("Skipping: no Neo4j at {uri}");
                return;
            }
        };

        // List all projects
        let projects = client.list_projects().await.expect("list_projects");
        eprintln!("Found {} projects", projects.len());

        let mut total_protocols = 0;
        let mut invalid_protocols: Vec<(String, uuid::Uuid, uuid::Uuid, Vec<ProtocolViolation>)> =
            Vec::new();

        for project in &projects {
            // List protocols for this project (limit 1000 should be enough)
            let (protocols, _) = client
                .list_protocols(Some(project.id), None, 1000, 0)
                .await
                .expect("list_protocols");

            for protocol in protocols {
                total_protocols += 1;

                // Get states and transitions
                let states = client
                    .get_protocol_states(protocol.id)
                    .await
                    .expect("get_protocol_states");
                let transitions = client
                    .get_protocol_transitions(protocol.id)
                    .await
                    .expect("get_protocol_transitions");

                // Convert to validatable types
                let validatable_states: Vec<ValidatableState<'_>> = states
                    .iter()
                    .map(|s| ValidatableState {
                        name: s.name.as_str(),
                        state_type: s.state_type,
                    })
                    .collect();

                let validatable_transitions: Vec<ValidatableTransition<'_>> = transitions
                    .iter()
                    .map(|t| {
                        let from_name = states
                            .iter()
                            .find(|s| s.id == t.from_state)
                            .map(|s| s.name.as_str())
                            .unwrap_or("?");
                        let to_name = states
                            .iter()
                            .find(|s| s.id == t.to_state)
                            .map(|s| s.name.as_str())
                            .unwrap_or("?");
                        ValidatableTransition {
                            from_state: from_name,
                            to_state: to_name,
                            trigger: t.trigger.as_str(),
                        }
                    })
                    .collect();

                // Validate
                let violations =
                    validate_protocol_structure(&validatable_states, &validatable_transitions);

                if !violations.is_empty() {
                    invalid_protocols.push((
                        protocol.name.clone(),
                        protocol.id,
                        project.id,
                        violations,
                    ));
                }
            }
        }

        // Print results
        eprintln!("\n=== PYSR-4 Protocol Validation Results ===");
        eprintln!("Total protocols validated: {}", total_protocols);
        eprintln!(
            "Valid: {}, Invalid: {}",
            total_protocols - invalid_protocols.len(),
            invalid_protocols.len()
        );

        if invalid_protocols.is_empty() {
            eprintln!(
                "\n✓ All {}/{} protocols are structurally valid",
                total_protocols, total_protocols
            );
        } else {
            eprintln!("\n✗ Invalid protocols:");
            for (name, id, project_id, violations) in &invalid_protocols {
                eprintln!("  - {} (id: {}, project: {})", name, id, project_id);
                for v in violations {
                    eprintln!("    • {}", v);
                }
            }
            panic!(
                "{} of {} protocols have structural violations",
                invalid_protocols.len(),
                total_protocols
            );
        }
    }
}
