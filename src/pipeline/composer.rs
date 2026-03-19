//! # Protocol Composer — Dynamic pipeline generation from Plans
//!
//! Analyzes a Plan (tasks, waves, constraints, affected_files) and generates
//! a hierarchical Protocol for automated verification.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Detected technology stack.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TechStack {
    Rust,
    TypeScript,
    JavaScript,
    Python,
    Go,
    Java,
    Unknown,
}

/// A composed pipeline specification (not yet a Protocol — a blueprint).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSpec {
    pub plan_id: Uuid,
    pub name: String,
    pub tech_stacks: Vec<TechStack>,
    pub stages: Vec<PipelineStage>,
    pub final_gates: Vec<GateSpec>,
}

/// A single stage in the pipeline, corresponding to a plan wave.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    pub wave_number: usize,
    pub task_ids: Vec<Uuid>,
    pub pre_gates: Vec<GateSpec>,
    pub post_gates: Vec<GateSpec>,
}

/// Specification for a quality gate within a pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSpec {
    pub gate_type: String,
    pub params: HashMap<String, serde_json::Value>,
}

/// Constraint from a Plan that affects gate selection.
#[derive(Debug, Clone)]
pub struct PlanConstraint {
    /// Constraint type: "performance", "security", "style", "compatibility".
    pub constraint_type: String,
    /// Human-readable description.
    pub description: String,
    /// Severity: "must", "should", "nice_to_have".
    pub severity: String,
}

/// Wave from `get_waves()`.
#[derive(Debug, Clone)]
pub struct PlanWave {
    pub wave_number: usize,
    pub task_ids: Vec<Uuid>,
    pub affected_files: Vec<String>,
}

// ---------------------------------------------------------------------------
// Protocol output types (for protocol(action: "compose"))
// ---------------------------------------------------------------------------

/// A state in the generated protocol state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolState {
    pub name: String,
    /// One of: "start", "intermediate", "terminal".
    pub state_type: String,
    pub description: String,
    pub action: Option<String>,
}

/// A transition in the generated protocol state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolTransition {
    pub from_state: String,
    pub to_state: String,
    pub trigger: String,
    pub guard: Option<String>,
}

// ---------------------------------------------------------------------------
// 1. detect_tech_stack
// ---------------------------------------------------------------------------

/// Detect technology stacks from file extensions in `affected_files`.
///
/// Returns a deduplicated list of [`TechStack`] values.
pub fn detect_tech_stack(affected_files: &[String]) -> Vec<TechStack> {
    let mut stacks = HashSet::new();

    for file in affected_files {
        let ext = file.rsplit('.').next().unwrap_or("");
        match ext {
            "rs" => {
                stacks.insert(TechStack::Rust);
            }
            "ts" | "tsx" => {
                stacks.insert(TechStack::TypeScript);
            }
            "js" | "jsx" => {
                stacks.insert(TechStack::JavaScript);
            }
            "py" => {
                stacks.insert(TechStack::Python);
            }
            "go" => {
                stacks.insert(TechStack::Go);
            }
            "java" => {
                stacks.insert(TechStack::Java);
            }
            _ => {}
        }
    }

    let mut result: Vec<TechStack> = stacks.into_iter().collect();
    // Sort for deterministic output.
    result.sort_by_key(|t| match t {
        TechStack::Rust => 0,
        TechStack::TypeScript => 1,
        TechStack::JavaScript => 2,
        TechStack::Python => 3,
        TechStack::Go => 4,
        TechStack::Java => 5,
        TechStack::Unknown => 6,
    });
    result
}

// ---------------------------------------------------------------------------
// 2. select_gates
// ---------------------------------------------------------------------------

/// Select quality gates based on plan constraints and detected tech stacks.
///
/// Rules:
/// - Rust → `cargo-check` + `cargo-test`
/// - TypeScript → `npm-typecheck`
/// - `constraint_type == "performance"` with `severity == "must"` → coverage gate (80%)
/// - `constraint_type == "performance"` with `severity == "should"` → coverage gate (60%)
/// - Always adds `pr-checks` if there are any other gates.
pub fn select_gates(constraints: &[PlanConstraint], tech_stacks: &[TechStack]) -> Vec<GateSpec> {
    let mut gates = Vec::new();

    // Tech-stack-specific gates.
    for stack in tech_stacks {
        match stack {
            TechStack::Rust => {
                gates.push(GateSpec {
                    gate_type: "cargo-check".into(),
                    params: HashMap::new(),
                });
                gates.push(GateSpec {
                    gate_type: "cargo-test".into(),
                    params: HashMap::new(),
                });
            }
            TechStack::TypeScript => {
                gates.push(GateSpec {
                    gate_type: "npm-typecheck".into(),
                    params: HashMap::new(),
                });
            }
            _ => {}
        }
    }

    // Constraint-driven gates.
    for constraint in constraints {
        if constraint.constraint_type == "performance" {
            let threshold: f64 = match constraint.severity.as_str() {
                "must" => 80.0,
                "should" => 60.0,
                _ => continue,
            };
            let mut params = HashMap::new();
            params.insert("threshold".into(), serde_json::json!(threshold));
            gates.push(GateSpec {
                gate_type: "coverage".into(),
                params,
            });
        }
    }

    // Always add pr-checks if there are any gates.
    if !gates.is_empty() {
        gates.push(GateSpec {
            gate_type: "pr-checks".into(),
            params: HashMap::new(),
        });
    }

    gates
}

// ---------------------------------------------------------------------------
// 3. compose_pipeline
// ---------------------------------------------------------------------------

/// Compose a full pipeline specification from plan data.
///
/// - Detects tech stack from `affected_files`
/// - Selects gates based on constraints and tech stack
/// - Creates stages from waves with appropriate pre/post gates
/// - Adds final gates at the end
pub fn compose_pipeline(
    plan_id: Uuid,
    plan_name: &str,
    waves: &[PlanWave],
    constraints: &[PlanConstraint],
    affected_files: &[String],
) -> PipelineSpec {
    let tech_stacks = detect_tech_stack(affected_files);
    let all_gates = select_gates(constraints, &tech_stacks);

    // Split gates into per-stage post-gates and final gates.
    // Per-stage post-gates: build/typecheck gates (fast feedback).
    // Final gates: coverage + pr-checks (run once at the end).
    let mut stage_post_gates = Vec::new();
    let mut final_gates = Vec::new();

    for gate in &all_gates {
        match gate.gate_type.as_str() {
            "cargo-check" | "cargo-test" | "npm-typecheck" => {
                stage_post_gates.push(gate.clone());
            }
            _ => {
                final_gates.push(gate.clone());
            }
        }
    }

    // Build stages from waves.
    let stages: Vec<PipelineStage> = waves
        .iter()
        .map(|wave| PipelineStage {
            wave_number: wave.wave_number,
            task_ids: wave.task_ids.clone(),
            pre_gates: Vec::new(),
            post_gates: stage_post_gates.clone(),
        })
        .collect();

    PipelineSpec {
        plan_id,
        name: format!("pipeline-{plan_name}"),
        tech_stacks,
        stages,
        final_gates,
    }
}

// ---------------------------------------------------------------------------
// 4. PipelineSpec → Protocol conversion
// ---------------------------------------------------------------------------

impl PipelineSpec {
    /// Convert this pipeline spec into protocol states.
    ///
    /// Structure:
    /// `init` (start) → wave-1 → gate-1-{type} → wave-2 → gate-2-{type} → ... → final-gate-{type} → done (terminal)
    pub fn to_protocol_states(&self) -> Vec<ProtocolState> {
        let mut states = Vec::new();

        // Start state.
        states.push(ProtocolState {
            name: "init".into(),
            state_type: "start".into(),
            description: format!("Initialize pipeline: {}", self.name),
            action: None,
        });

        // Wave and post-gate states.
        for stage in &self.stages {
            let wave_name = format!("wave-{}", stage.wave_number);
            states.push(ProtocolState {
                name: wave_name,
                state_type: "intermediate".into(),
                description: format!(
                    "Execute wave {} ({} tasks)",
                    stage.wave_number,
                    stage.task_ids.len()
                ),
                action: Some("execute_wave".into()),
            });

            for gate in &stage.post_gates {
                let gate_name = format!("gate-{}-{}", stage.wave_number, gate.gate_type);
                states.push(ProtocolState {
                    name: gate_name,
                    state_type: "intermediate".into(),
                    description: format!("Run {} after wave {}", gate.gate_type, stage.wave_number),
                    action: Some(format!("run_gate:{}", gate.gate_type)),
                });
            }
        }

        // Final gate states.
        for gate in &self.final_gates {
            let gate_name = format!("final-gate-{}", gate.gate_type);
            states.push(ProtocolState {
                name: gate_name,
                state_type: "intermediate".into(),
                description: format!("Final gate: {}", gate.gate_type),
                action: Some(format!("run_gate:{}", gate.gate_type)),
            });
        }

        // Terminal state.
        states.push(ProtocolState {
            name: "done".into(),
            state_type: "terminal".into(),
            description: "Pipeline complete".into(),
            action: None,
        });

        // Failure terminal state.
        states.push(ProtocolState {
            name: "failed".into(),
            state_type: "terminal".into(),
            description: "Pipeline failed — a gate did not pass".into(),
            action: None,
        });

        states
    }

    /// Convert this pipeline spec into protocol transitions.
    ///
    /// Transitions chain states sequentially:
    /// - `wave_complete` transitions from wave to its first post-gate (or next wave)
    /// - `gate_passed` transitions to the next gate or wave
    /// - `gate_failed` transitions to the `failed` terminal state
    pub fn to_protocol_transitions(&self) -> Vec<ProtocolTransition> {
        let states = self.to_protocol_states();
        let mut transitions = Vec::new();

        // Collect only actionable states (skip "failed" terminal).
        let flow_states: Vec<&ProtocolState> =
            states.iter().filter(|s| s.name != "failed").collect();

        for i in 0..flow_states.len().saturating_sub(1) {
            let from = &flow_states[i];
            let to = &flow_states[i + 1];

            let trigger = if from.name == "init" {
                "pipeline_started".into()
            } else if from.action.as_deref() == Some("execute_wave") {
                "wave_complete".into()
            } else if from
                .action
                .as_deref()
                .is_some_and(|a| a.starts_with("run_gate:"))
            {
                "gate_passed".into()
            } else {
                "next".into()
            };

            // Happy-path transition.
            transitions.push(ProtocolTransition {
                from_state: from.name.clone(),
                to_state: to.name.clone(),
                trigger,
                guard: None,
            });

            // Failure transition for gate states.
            if from
                .action
                .as_deref()
                .is_some_and(|a| a.starts_with("run_gate:"))
            {
                transitions.push(ProtocolTransition {
                    from_state: from.name.clone(),
                    to_state: "failed".into(),
                    trigger: "gate_failed".into(),
                    guard: None,
                });
            }
        }

        transitions
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- detect_tech_stack ---------------------------------------------------

    #[test]
    fn detect_rust_files() {
        let files = vec!["src/main.rs".into(), "src/lib.rs".into()];
        let stacks = detect_tech_stack(&files);
        assert_eq!(stacks, vec![TechStack::Rust]);
    }

    #[test]
    fn detect_typescript_files() {
        let files = vec!["app/page.tsx".into(), "lib/utils.ts".into()];
        let stacks = detect_tech_stack(&files);
        assert_eq!(stacks, vec![TechStack::TypeScript]);
    }

    #[test]
    fn detect_mixed_stacks() {
        let files = vec![
            "src/main.rs".into(),
            "frontend/app.tsx".into(),
            "scripts/deploy.py".into(),
        ];
        let stacks = detect_tech_stack(&files);
        assert!(stacks.contains(&TechStack::Rust));
        assert!(stacks.contains(&TechStack::TypeScript));
        assert!(stacks.contains(&TechStack::Python));
        assert_eq!(stacks.len(), 3);
    }

    #[test]
    fn detect_empty_files() {
        let stacks = detect_tech_stack(&[]);
        assert!(stacks.is_empty());
    }

    #[test]
    fn detect_unknown_extensions_ignored() {
        let files = vec!["README.md".into(), "Makefile".into(), "data.csv".into()];
        let stacks = detect_tech_stack(&files);
        assert!(stacks.is_empty());
    }

    #[test]
    fn detect_deduplicates() {
        let files = vec!["a.rs".into(), "b.rs".into(), "c.rs".into()];
        let stacks = detect_tech_stack(&files);
        assert_eq!(stacks, vec![TechStack::Rust]);
    }

    #[test]
    fn detect_javascript_and_go() {
        let files = vec!["index.js".into(), "component.jsx".into(), "main.go".into()];
        let stacks = detect_tech_stack(&files);
        assert!(stacks.contains(&TechStack::JavaScript));
        assert!(stacks.contains(&TechStack::Go));
        assert_eq!(stacks.len(), 2);
    }

    #[test]
    fn detect_java() {
        let files = vec!["Main.java".into()];
        let stacks = detect_tech_stack(&files);
        assert_eq!(stacks, vec![TechStack::Java]);
    }

    // -- select_gates --------------------------------------------------------

    #[test]
    fn select_gates_rust_stack() {
        let gates = select_gates(&[], &[TechStack::Rust]);
        let types: Vec<&str> = gates.iter().map(|g| g.gate_type.as_str()).collect();
        assert!(types.contains(&"cargo-check"));
        assert!(types.contains(&"cargo-test"));
        assert!(types.contains(&"pr-checks"));
    }

    #[test]
    fn select_gates_typescript_stack() {
        let gates = select_gates(&[], &[TechStack::TypeScript]);
        let types: Vec<&str> = gates.iter().map(|g| g.gate_type.as_str()).collect();
        assert!(types.contains(&"npm-typecheck"));
        assert!(types.contains(&"pr-checks"));
        assert!(!types.contains(&"cargo-check"));
    }

    #[test]
    fn select_gates_performance_must() {
        let constraints = vec![PlanConstraint {
            constraint_type: "performance".into(),
            description: "High coverage required".into(),
            severity: "must".into(),
        }];
        let gates = select_gates(&constraints, &[TechStack::Rust]);
        let coverage_gate = gates.iter().find(|g| g.gate_type == "coverage");
        assert!(coverage_gate.is_some());
        let threshold = coverage_gate.unwrap().params.get("threshold").unwrap();
        assert_eq!(threshold, &serde_json::json!(80.0));
    }

    #[test]
    fn select_gates_performance_should() {
        let constraints = vec![PlanConstraint {
            constraint_type: "performance".into(),
            description: "Moderate coverage".into(),
            severity: "should".into(),
        }];
        let gates = select_gates(&constraints, &[TechStack::Rust]);
        let coverage_gate = gates.iter().find(|g| g.gate_type == "coverage");
        assert!(coverage_gate.is_some());
        let threshold = coverage_gate.unwrap().params.get("threshold").unwrap();
        assert_eq!(threshold, &serde_json::json!(60.0));
    }

    #[test]
    fn select_gates_empty_returns_empty() {
        let gates = select_gates(&[], &[]);
        assert!(gates.is_empty());
    }

    #[test]
    fn select_gates_no_pr_checks_when_no_other_gates() {
        let gates = select_gates(&[], &[TechStack::Python]);
        // Python has no specific gates, so no pr-checks either.
        assert!(gates.is_empty());
    }

    #[test]
    fn select_gates_nice_to_have_skipped() {
        let constraints = vec![PlanConstraint {
            constraint_type: "performance".into(),
            description: "Optional coverage".into(),
            severity: "nice_to_have".into(),
        }];
        // Only Python (no built-in gates) + nice_to_have → no gates.
        let gates = select_gates(&constraints, &[TechStack::Python]);
        assert!(gates.is_empty());
    }

    // -- compose_pipeline ----------------------------------------------------

    #[test]
    fn compose_pipeline_basic() {
        let plan_id = Uuid::new_v4();
        let waves = vec![
            PlanWave {
                wave_number: 1,
                task_ids: vec![Uuid::new_v4()],
                affected_files: vec!["src/lib.rs".into()],
            },
            PlanWave {
                wave_number: 2,
                task_ids: vec![Uuid::new_v4(), Uuid::new_v4()],
                affected_files: vec!["src/main.rs".into()],
            },
        ];
        let affected_files = vec!["src/lib.rs".into(), "src/main.rs".into()];

        let spec = compose_pipeline(plan_id, "test-plan", &waves, &[], &affected_files);

        assert_eq!(spec.plan_id, plan_id);
        assert_eq!(spec.name, "pipeline-test-plan");
        assert_eq!(spec.tech_stacks, vec![TechStack::Rust]);
        assert_eq!(spec.stages.len(), 2);
        assert_eq!(spec.stages[0].wave_number, 1);
        assert_eq!(spec.stages[1].wave_number, 2);

        // Each stage should have cargo-check + cargo-test as post-gates.
        for stage in &spec.stages {
            let types: Vec<&str> = stage
                .post_gates
                .iter()
                .map(|g| g.gate_type.as_str())
                .collect();
            assert!(types.contains(&"cargo-check"));
            assert!(types.contains(&"cargo-test"));
        }

        // Final gates should include pr-checks.
        let final_types: Vec<&str> = spec
            .final_gates
            .iter()
            .map(|g| g.gate_type.as_str())
            .collect();
        assert!(final_types.contains(&"pr-checks"));
    }

    #[test]
    fn compose_pipeline_with_coverage_constraint() {
        let plan_id = Uuid::new_v4();
        let waves = vec![PlanWave {
            wave_number: 1,
            task_ids: vec![Uuid::new_v4()],
            affected_files: vec!["src/lib.rs".into()],
        }];
        let constraints = vec![PlanConstraint {
            constraint_type: "performance".into(),
            description: "Must have coverage".into(),
            severity: "must".into(),
        }];
        let affected_files = vec!["src/lib.rs".into()];

        let spec = compose_pipeline(plan_id, "cov-plan", &waves, &constraints, &affected_files);

        let final_types: Vec<&str> = spec
            .final_gates
            .iter()
            .map(|g| g.gate_type.as_str())
            .collect();
        assert!(final_types.contains(&"coverage"));
        assert!(final_types.contains(&"pr-checks"));
    }

    #[test]
    fn compose_pipeline_empty_waves() {
        let plan_id = Uuid::new_v4();
        let spec = compose_pipeline(plan_id, "empty", &[], &[], &[]);

        assert!(spec.stages.is_empty());
        assert!(spec.final_gates.is_empty());
        assert!(spec.tech_stacks.is_empty());
    }

    // -- to_protocol_states --------------------------------------------------

    #[test]
    fn protocol_states_has_init_and_done() {
        let plan_id = Uuid::new_v4();
        let waves = vec![PlanWave {
            wave_number: 1,
            task_ids: vec![Uuid::new_v4()],
            affected_files: vec!["src/lib.rs".into()],
        }];
        let affected_files = vec!["src/lib.rs".into()];
        let spec = compose_pipeline(plan_id, "proto", &waves, &[], &affected_files);

        let states = spec.to_protocol_states();

        let first = &states[0];
        assert_eq!(first.name, "init");
        assert_eq!(first.state_type, "start");

        let done = states.iter().find(|s| s.name == "done").unwrap();
        assert_eq!(done.state_type, "terminal");

        let failed = states.iter().find(|s| s.name == "failed").unwrap();
        assert_eq!(failed.state_type, "terminal");
    }

    #[test]
    fn protocol_states_wave_actions() {
        let plan_id = Uuid::new_v4();
        let waves = vec![
            PlanWave {
                wave_number: 1,
                task_ids: vec![Uuid::new_v4()],
                affected_files: vec!["a.rs".into()],
            },
            PlanWave {
                wave_number: 2,
                task_ids: vec![Uuid::new_v4()],
                affected_files: vec!["b.rs".into()],
            },
        ];
        let affected_files = vec!["a.rs".into(), "b.rs".into()];
        let spec = compose_pipeline(plan_id, "wave-test", &waves, &[], &affected_files);

        let states = spec.to_protocol_states();

        let wave_states: Vec<&ProtocolState> = states
            .iter()
            .filter(|s| s.action.as_deref() == Some("execute_wave"))
            .collect();
        assert_eq!(wave_states.len(), 2);
        assert_eq!(wave_states[0].name, "wave-1");
        assert_eq!(wave_states[1].name, "wave-2");
    }

    #[test]
    fn protocol_states_gate_actions() {
        let plan_id = Uuid::new_v4();
        let waves = vec![PlanWave {
            wave_number: 1,
            task_ids: vec![Uuid::new_v4()],
            affected_files: vec!["src/lib.rs".into()],
        }];
        let affected_files = vec!["src/lib.rs".into()];
        let spec = compose_pipeline(plan_id, "gate-test", &waves, &[], &affected_files);

        let states = spec.to_protocol_states();

        let gate_states: Vec<&ProtocolState> = states
            .iter()
            .filter(|s| {
                s.action
                    .as_deref()
                    .is_some_and(|a| a.starts_with("run_gate:"))
            })
            .collect();

        // Should have cargo-check + cargo-test post-gates + pr-checks final gate.
        assert!(gate_states.len() >= 3);
    }

    // -- to_protocol_transitions ---------------------------------------------

    #[test]
    fn protocol_transitions_chain_correctly() {
        let plan_id = Uuid::new_v4();
        let waves = vec![PlanWave {
            wave_number: 1,
            task_ids: vec![Uuid::new_v4()],
            affected_files: vec!["src/lib.rs".into()],
        }];
        let affected_files = vec!["src/lib.rs".into()];
        let spec = compose_pipeline(plan_id, "trans-test", &waves, &[], &affected_files);

        let transitions = spec.to_protocol_transitions();

        // First transition should be from init.
        let first = &transitions[0];
        assert_eq!(first.from_state, "init");
        assert_eq!(first.trigger, "pipeline_started");

        // Should have failure transitions for gate states.
        let fail_transitions: Vec<&ProtocolTransition> = transitions
            .iter()
            .filter(|t| t.trigger == "gate_failed")
            .collect();
        assert!(!fail_transitions.is_empty());
        for ft in &fail_transitions {
            assert_eq!(ft.to_state, "failed");
        }

        // Should end at "done".
        let to_done: Vec<&ProtocolTransition> = transitions
            .iter()
            .filter(|t| t.to_state == "done")
            .collect();
        assert_eq!(to_done.len(), 1);
    }

    #[test]
    fn protocol_transitions_wave_complete_trigger() {
        let plan_id = Uuid::new_v4();
        let waves = vec![PlanWave {
            wave_number: 1,
            task_ids: vec![Uuid::new_v4()],
            affected_files: vec!["src/lib.rs".into()],
        }];
        let affected_files = vec!["src/lib.rs".into()];
        let spec = compose_pipeline(plan_id, "trigger-test", &waves, &[], &affected_files);

        let transitions = spec.to_protocol_transitions();

        let wave_transitions: Vec<&ProtocolTransition> = transitions
            .iter()
            .filter(|t| t.trigger == "wave_complete")
            .collect();
        assert!(!wave_transitions.is_empty());
        assert_eq!(wave_transitions[0].from_state, "wave-1");
    }

    #[test]
    fn protocol_empty_pipeline() {
        let plan_id = Uuid::new_v4();
        let spec = compose_pipeline(plan_id, "empty", &[], &[], &[]);

        let states = spec.to_protocol_states();
        // Should still have init, done, failed.
        assert_eq!(states.len(), 3);

        let transitions = spec.to_protocol_transitions();
        // init → done.
        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].from_state, "init");
        assert_eq!(transitions[0].to_state, "done");
    }
}
