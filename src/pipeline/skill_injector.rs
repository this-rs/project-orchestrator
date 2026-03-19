//! # Skill Injector — Skills modify generated pipelines
//!
//! When the composer generates a pipeline, the SkillInjector consults
//! active skills and applies their modifications to the pipeline spec.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::composer::{GateSpec, PipelineSpec, ProtocolState};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A modification that a skill wants to apply to a pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillInjection {
    /// Which skill suggested this.
    pub skill_id: Uuid,
    /// Skill name for tracing.
    pub skill_name: String,
    /// What kind of modification.
    pub modification: InjectionType,
    /// Human-readable rationale.
    pub rationale: String,
    /// Confidence score [0, 1] — higher = more certain the injection helps.
    pub confidence: f64,
}

/// The kind of modification a skill wants to apply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InjectionType {
    /// Add a new state to the pipeline protocol.
    AddState {
        state: ProtocolState,
        /// Insert after this state name.
        after_state: String,
        /// Transition trigger from the preceding state.
        trigger: String,
    },
    /// Modify parameters of an existing quality gate.
    ModifyGateParams {
        gate_type: String,
        param_overrides: HashMap<String, serde_json::Value>,
    },
    /// Add a guard condition on a transition.
    AddGuard {
        from_state: String,
        to_state: String,
        guard: String,
    },
    /// Add a pre-gate to a specific wave stage.
    AddPreGate { wave_number: usize, gate: GateSpec },
    /// Add a post-gate to a specific wave stage.
    AddPostGate { wave_number: usize, gate: GateSpec },
    /// Override the coverage threshold.
    OverrideCoverageThreshold { new_threshold: f64 },
}

/// Result of applying injections to a pipeline spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionResult {
    /// How many injections were applied.
    pub applied: usize,
    /// How many were skipped (low confidence, conflicting, etc.).
    pub skipped: usize,
    /// Trace of what was applied.
    pub trace: Vec<InjectionTrace>,
}

/// Trace entry for a single injection attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionTrace {
    pub skill_id: Uuid,
    pub skill_name: String,
    pub modification_type: String,
    pub rationale: String,
    pub applied: bool,
    /// Reason if the injection was skipped.
    pub reason: Option<String>,
}

/// Simplified skill data for the injector (avoids coupling to the full Skill struct).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillContext {
    pub skill_id: Uuid,
    pub name: String,
    pub tags: Vec<String>,
    pub trigger_patterns: Vec<String>,
    pub context_template: Option<String>,
    /// Notes associated with the skill.
    pub notes: Vec<SkillNote>,
}

/// A note attached to a skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillNote {
    pub note_type: String,
    pub content: String,
    pub importance: String,
}

// ---------------------------------------------------------------------------
// SkillInjector
// ---------------------------------------------------------------------------

/// The Skill Injector — consults active skills and modifies pipeline specs.
pub struct SkillInjector {
    /// Minimum confidence threshold to apply an injection.
    min_confidence: f64,
}

impl SkillInjector {
    /// Create a new `SkillInjector` with the given minimum confidence threshold.
    ///
    /// Injections with confidence below this value will be skipped.
    pub fn new(min_confidence: f64) -> Self {
        Self { min_confidence }
    }

    // -----------------------------------------------------------------------
    // analyze_skills
    // -----------------------------------------------------------------------

    /// Analyze active skills and derive pipeline injections from their metadata.
    ///
    /// Heuristics:
    /// - Tag `"coverage"` + a note mentioning a threshold → `OverrideCoverageThreshold`
    /// - Tag `"testing"` + note mentioning "write tests first" → `AddPreGate` (cargo-test)
    /// - Tag `"security"` → `AddGuard` on merge transitions
    /// - `context_template` containing a JSON injection directive → parsed directly
    pub fn analyze_skills(
        &self,
        skills: &[SkillContext],
        _spec: &PipelineSpec,
    ) -> Vec<SkillInjection> {
        let mut injections = Vec::new();

        for skill in skills {
            // --- coverage tag -------------------------------------------------
            if skill.tags.iter().any(|t| t == "coverage") {
                if let Some(threshold) = Self::extract_coverage_threshold(skill) {
                    injections.push(SkillInjection {
                        skill_id: skill.skill_id,
                        skill_name: skill.name.clone(),
                        modification: InjectionType::OverrideCoverageThreshold {
                            new_threshold: threshold,
                        },
                        rationale: format!(
                            "Skill '{}' requests coverage threshold {:.0}%",
                            skill.name, threshold
                        ),
                        confidence: 0.8,
                    });
                }
            }

            // --- testing tag --------------------------------------------------
            if skill.tags.iter().any(|t| t == "testing") {
                let mentions_tests_first = skill.notes.iter().any(|n| {
                    let lower = n.content.to_lowercase();
                    lower.contains("write tests first") || lower.contains("test-first")
                });
                if mentions_tests_first {
                    injections.push(SkillInjection {
                        skill_id: skill.skill_id,
                        skill_name: skill.name.clone(),
                        modification: InjectionType::AddPreGate {
                            wave_number: 1,
                            gate: GateSpec {
                                gate_type: "cargo-test".into(),
                                params: HashMap::new(),
                            },
                        },
                        rationale: format!(
                            "Skill '{}' enforces test-first: add cargo-test pre-gate on wave 1",
                            skill.name
                        ),
                        confidence: 0.7,
                    });
                }
            }

            // --- security tag -------------------------------------------------
            if skill.tags.iter().any(|t| t == "security") {
                injections.push(SkillInjection {
                    skill_id: skill.skill_id,
                    skill_name: skill.name.clone(),
                    modification: InjectionType::AddGuard {
                        from_state: "done".into(),
                        to_state: "done".into(),
                        guard: "security_review_passed".into(),
                    },
                    rationale: format!(
                        "Skill '{}' requires security review before completion",
                        skill.name
                    ),
                    confidence: 0.9,
                });
            }

            // --- context_template JSON directives -----------------------------
            if let Some(ref template) = skill.context_template {
                if let Some(directive_injections) =
                    Self::parse_context_template(skill, template)
                {
                    injections.extend(directive_injections);
                }
            }
        }

        injections
    }

    // -----------------------------------------------------------------------
    // apply_injections
    // -----------------------------------------------------------------------

    /// Apply a set of injections to a mutable pipeline spec.
    ///
    /// Injections whose confidence is below `min_confidence` are skipped.
    pub fn apply_injections(
        &self,
        spec: &mut PipelineSpec,
        injections: &[SkillInjection],
    ) -> InjectionResult {
        let mut applied = 0usize;
        let mut skipped = 0usize;
        let mut trace = Vec::new();

        for injection in injections {
            let mod_type = Self::modification_type_name(&injection.modification);

            // Confidence gate.
            if injection.confidence < self.min_confidence {
                skipped += 1;
                trace.push(InjectionTrace {
                    skill_id: injection.skill_id,
                    skill_name: injection.skill_name.clone(),
                    modification_type: mod_type,
                    rationale: injection.rationale.clone(),
                    applied: false,
                    reason: Some(format!(
                        "confidence {:.2} < threshold {:.2}",
                        injection.confidence, self.min_confidence
                    )),
                });
                continue;
            }

            let result = self.apply_single(spec, injection);

            match result {
                Ok(()) => {
                    applied += 1;
                    trace.push(InjectionTrace {
                        skill_id: injection.skill_id,
                        skill_name: injection.skill_name.clone(),
                        modification_type: mod_type,
                        rationale: injection.rationale.clone(),
                        applied: true,
                        reason: None,
                    });
                }
                Err(reason) => {
                    skipped += 1;
                    trace.push(InjectionTrace {
                        skill_id: injection.skill_id,
                        skill_name: injection.skill_name.clone(),
                        modification_type: mod_type,
                        rationale: injection.rationale.clone(),
                        applied: false,
                        reason: Some(reason),
                    });
                }
            }
        }

        InjectionResult {
            applied,
            skipped,
            trace,
        }
    }

    // -----------------------------------------------------------------------
    // inject (convenience)
    // -----------------------------------------------------------------------

    /// Convenience: analyze skills then apply the resulting injections.
    pub fn inject(
        &self,
        skills: &[SkillContext],
        spec: &mut PipelineSpec,
    ) -> InjectionResult {
        let injections = self.analyze_skills(skills, spec);
        self.apply_injections(spec, &injections)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Apply a single injection to the pipeline spec.
    fn apply_single(
        &self,
        spec: &mut PipelineSpec,
        injection: &SkillInjection,
    ) -> Result<(), String> {
        match &injection.modification {
            InjectionType::AddState {
                state,
                after_state,
                trigger,
            } => {
                // We operate on the protocol-level representation stored in the
                // spec by generating protocol states, inserting, then storing
                // the new state as an extra stage. Because PipelineSpec doesn't
                // directly hold protocol states we work on the stage list and
                // store an extra metadata stage. For simplicity the injector
                // adds the state to a synthetic wave 0 pre-gate list if the
                // after_state is "init", otherwise it appends to final_gates.
                // The real protocol conversion (`to_protocol_states`) is
                // extended later; for now we record the intent.
                let _ = (state, after_state, trigger);

                // Verify after_state exists in the protocol state names.
                let protocol_states = spec.to_protocol_states();
                if !protocol_states.iter().any(|s| s.name == *after_state) {
                    return Err(format!(
                        "after_state '{}' not found in protocol states",
                        after_state
                    ));
                }

                // Store as a final gate with a special marker type so that
                // downstream protocol generation can pick it up.
                spec.final_gates.push(GateSpec {
                    gate_type: format!("injected-state:{}", state.name),
                    params: {
                        let mut p = HashMap::new();
                        p.insert(
                            "after_state".into(),
                            serde_json::Value::String(after_state.clone()),
                        );
                        p.insert(
                            "trigger".into(),
                            serde_json::Value::String(trigger.clone()),
                        );
                        p.insert(
                            "description".into(),
                            serde_json::Value::String(state.description.clone()),
                        );
                        p
                    },
                });
                Ok(())
            }

            InjectionType::ModifyGateParams {
                gate_type,
                param_overrides,
            } => {
                let mut found = false;
                // Search all stages and final_gates.
                for stage in &mut spec.stages {
                    for gate in stage
                        .pre_gates
                        .iter_mut()
                        .chain(stage.post_gates.iter_mut())
                    {
                        if gate.gate_type == *gate_type {
                            for (k, v) in param_overrides {
                                gate.params.insert(k.clone(), v.clone());
                            }
                            found = true;
                        }
                    }
                }
                for gate in &mut spec.final_gates {
                    if gate.gate_type == *gate_type {
                        for (k, v) in param_overrides {
                            gate.params.insert(k.clone(), v.clone());
                        }
                        found = true;
                    }
                }
                if found {
                    Ok(())
                } else {
                    Err(format!("no gate of type '{}' found in pipeline", gate_type))
                }
            }

            InjectionType::AddGuard {
                from_state,
                to_state,
                guard,
            } => {
                // We record the guard intent. The pipeline spec itself doesn't
                // store transitions directly — they are derived. So we store
                // the guard as a special final gate that the protocol
                // converter can consult.
                spec.final_gates.push(GateSpec {
                    gate_type: "injected-guard".into(),
                    params: {
                        let mut p = HashMap::new();
                        p.insert(
                            "from_state".into(),
                            serde_json::Value::String(from_state.clone()),
                        );
                        p.insert(
                            "to_state".into(),
                            serde_json::Value::String(to_state.clone()),
                        );
                        p.insert(
                            "guard".into(),
                            serde_json::Value::String(guard.clone()),
                        );
                        p
                    },
                });
                Ok(())
            }

            InjectionType::AddPreGate { wave_number, gate } => {
                if let Some(stage) = spec
                    .stages
                    .iter_mut()
                    .find(|s| s.wave_number == *wave_number)
                {
                    stage.pre_gates.push(gate.clone());
                    Ok(())
                } else {
                    Err(format!("wave {} not found in pipeline", wave_number))
                }
            }

            InjectionType::AddPostGate { wave_number, gate } => {
                if let Some(stage) = spec
                    .stages
                    .iter_mut()
                    .find(|s| s.wave_number == *wave_number)
                {
                    stage.post_gates.push(gate.clone());
                    Ok(())
                } else {
                    Err(format!("wave {} not found in pipeline", wave_number))
                }
            }

            InjectionType::OverrideCoverageThreshold { new_threshold } => {
                let mut found = false;
                for gate in &mut spec.final_gates {
                    if gate.gate_type == "coverage" {
                        gate.params.insert(
                            "threshold".into(),
                            serde_json::json!(new_threshold),
                        );
                        found = true;
                    }
                }
                for stage in &mut spec.stages {
                    for gate in stage
                        .pre_gates
                        .iter_mut()
                        .chain(stage.post_gates.iter_mut())
                    {
                        if gate.gate_type == "coverage" {
                            gate.params.insert(
                                "threshold".into(),
                                serde_json::json!(new_threshold),
                            );
                            found = true;
                        }
                    }
                }
                if found {
                    Ok(())
                } else {
                    Err("no coverage gate found in pipeline to override".into())
                }
            }
        }
    }

    /// Extract a coverage threshold from a skill's notes.
    ///
    /// Looks for notes containing patterns like "90%", "threshold 85", etc.
    fn extract_coverage_threshold(skill: &SkillContext) -> Option<f64> {
        for note in &skill.notes {
            let content = &note.content;
            // Look for patterns like "90%", "threshold: 85", "coverage: 75%"
            if let Some(val) = Self::extract_percentage(content) {
                return Some(val);
            }
        }
        None
    }

    /// Extract a percentage value from text (e.g. "90%", "threshold 85").
    fn extract_percentage(text: &str) -> Option<f64> {
        // Simple pattern: find a number followed by optional '%'.
        let lower = text.to_lowercase();
        for word in lower.split_whitespace() {
            let cleaned = word.trim_end_matches('%').trim_end_matches(',').trim_end_matches('.');
            if let Ok(val) = cleaned.parse::<f64>() {
                // Only accept values that look like percentages (0–100).
                if (0.0..=100.0).contains(&val) {
                    return Some(val);
                }
            }
        }
        None
    }

    /// Parse a context_template for JSON injection directives.
    ///
    /// Expected format:
    /// ```json
    /// {
    ///   "injections": [
    ///     {
    ///       "type": "add_post_gate",
    ///       "wave_number": 1,
    ///       "gate_type": "lint",
    ///       "params": {},
    ///       "confidence": 0.8,
    ///       "rationale": "Always lint after implementation"
    ///     }
    ///   ]
    /// }
    /// ```
    fn parse_context_template(
        skill: &SkillContext,
        template: &str,
    ) -> Option<Vec<SkillInjection>> {
        let parsed: serde_json::Value = serde_json::from_str(template).ok()?;
        let directives = parsed.get("injections")?.as_array()?;

        let mut injections = Vec::new();
        for directive in directives {
            let injection_type = directive.get("type")?.as_str()?;
            let confidence = directive
                .get("confidence")
                .and_then(|c| c.as_f64())
                .unwrap_or(0.6);
            let rationale = directive
                .get("rationale")
                .and_then(|r| r.as_str())
                .unwrap_or("from context_template directive")
                .to_string();

            let modification = match injection_type {
                "add_pre_gate" => {
                    let wave_number =
                        directive.get("wave_number")?.as_u64()? as usize;
                    let gate_type =
                        directive.get("gate_type")?.as_str()?.to_string();
                    let params: HashMap<String, serde_json::Value> = directive
                        .get("params")
                        .and_then(|p| serde_json::from_value(p.clone()).ok())
                        .unwrap_or_default();
                    InjectionType::AddPreGate {
                        wave_number,
                        gate: GateSpec { gate_type, params },
                    }
                }
                "add_post_gate" => {
                    let wave_number =
                        directive.get("wave_number")?.as_u64()? as usize;
                    let gate_type =
                        directive.get("gate_type")?.as_str()?.to_string();
                    let params: HashMap<String, serde_json::Value> = directive
                        .get("params")
                        .and_then(|p| serde_json::from_value(p.clone()).ok())
                        .unwrap_or_default();
                    InjectionType::AddPostGate {
                        wave_number,
                        gate: GateSpec { gate_type, params },
                    }
                }
                "modify_gate_params" => {
                    let gate_type =
                        directive.get("gate_type")?.as_str()?.to_string();
                    let param_overrides: HashMap<String, serde_json::Value> =
                        directive
                            .get("params")
                            .and_then(|p| serde_json::from_value(p.clone()).ok())
                            .unwrap_or_default();
                    InjectionType::ModifyGateParams {
                        gate_type,
                        param_overrides,
                    }
                }
                "add_guard" => {
                    let from_state =
                        directive.get("from_state")?.as_str()?.to_string();
                    let to_state =
                        directive.get("to_state")?.as_str()?.to_string();
                    let guard =
                        directive.get("guard")?.as_str()?.to_string();
                    InjectionType::AddGuard {
                        from_state,
                        to_state,
                        guard,
                    }
                }
                "override_coverage_threshold" => {
                    let new_threshold =
                        directive.get("new_threshold")?.as_f64()?;
                    InjectionType::OverrideCoverageThreshold { new_threshold }
                }
                _ => continue,
            };

            injections.push(SkillInjection {
                skill_id: skill.skill_id,
                skill_name: skill.name.clone(),
                modification,
                rationale,
                confidence,
            });
        }

        if injections.is_empty() {
            None
        } else {
            Some(injections)
        }
    }

    /// Return a human-readable name for an injection type.
    fn modification_type_name(modification: &InjectionType) -> String {
        match modification {
            InjectionType::AddState { .. } => "AddState".into(),
            InjectionType::ModifyGateParams { .. } => "ModifyGateParams".into(),
            InjectionType::AddGuard { .. } => "AddGuard".into(),
            InjectionType::AddPreGate { .. } => "AddPreGate".into(),
            InjectionType::AddPostGate { .. } => "AddPostGate".into(),
            InjectionType::OverrideCoverageThreshold { .. } => {
                "OverrideCoverageThreshold".into()
            }
        }
    }
}

impl Default for SkillInjector {
    fn default() -> Self {
        Self::new(0.5)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::composer::{compose_pipeline, PlanConstraint, PlanWave};

    /// Helper: build a basic Rust pipeline spec with a coverage gate.
    fn make_rust_spec_with_coverage() -> PipelineSpec {
        let plan_id = Uuid::new_v4();
        let waves = vec![
            PlanWave {
                wave_number: 1,
                task_ids: vec![Uuid::new_v4()],
                affected_files: vec!["src/lib.rs".into()],
            },
            PlanWave {
                wave_number: 2,
                task_ids: vec![Uuid::new_v4()],
                affected_files: vec!["src/main.rs".into()],
            },
        ];
        let constraints = vec![PlanConstraint {
            constraint_type: "performance".into(),
            description: "Must have coverage".into(),
            severity: "must".into(),
        }];
        let affected = vec!["src/lib.rs".into(), "src/main.rs".into()];
        compose_pipeline(plan_id, "test", &waves, &constraints, &affected)
    }

    /// Helper: build a basic Rust pipeline spec without coverage.
    fn make_rust_spec() -> PipelineSpec {
        let plan_id = Uuid::new_v4();
        let waves = vec![PlanWave {
            wave_number: 1,
            task_ids: vec![Uuid::new_v4()],
            affected_files: vec!["src/lib.rs".into()],
        }];
        let affected = vec!["src/lib.rs".into()];
        compose_pipeline(plan_id, "test", &waves, &[], &affected)
    }

    fn make_skill(name: &str, tags: Vec<&str>, notes: Vec<(&str, &str)>) -> SkillContext {
        SkillContext {
            skill_id: Uuid::new_v4(),
            name: name.into(),
            tags: tags.into_iter().map(String::from).collect(),
            trigger_patterns: vec![],
            context_template: None,
            notes: notes
                .into_iter()
                .map(|(ntype, content)| SkillNote {
                    note_type: ntype.into(),
                    content: content.into(),
                    importance: "high".into(),
                })
                .collect(),
        }
    }

    // -- analyze_skills --------------------------------------------------------

    #[test]
    fn analyze_coverage_skill() {
        let injector = SkillInjector::default();
        let spec = make_rust_spec_with_coverage();
        let skills = vec![make_skill(
            "strict-coverage",
            vec!["coverage"],
            vec![("config", "minimum coverage threshold 90%")],
        )];

        let injections = injector.analyze_skills(&skills, &spec);
        assert_eq!(injections.len(), 1);
        match &injections[0].modification {
            InjectionType::OverrideCoverageThreshold { new_threshold } => {
                assert!((new_threshold - 90.0).abs() < f64::EPSILON);
            }
            other => panic!("expected OverrideCoverageThreshold, got {:?}", other),
        }
    }

    #[test]
    fn analyze_testing_skill() {
        let injector = SkillInjector::default();
        let spec = make_rust_spec();
        let skills = vec![make_skill(
            "tdd",
            vec!["testing"],
            vec![("practice", "Always write tests first before implementation")],
        )];

        let injections = injector.analyze_skills(&skills, &spec);
        assert_eq!(injections.len(), 1);
        match &injections[0].modification {
            InjectionType::AddPreGate { wave_number, gate } => {
                assert_eq!(*wave_number, 1);
                assert_eq!(gate.gate_type, "cargo-test");
            }
            other => panic!("expected AddPreGate, got {:?}", other),
        }
    }

    #[test]
    fn analyze_security_skill() {
        let injector = SkillInjector::default();
        let spec = make_rust_spec();
        let skills = vec![make_skill("sec-review", vec!["security"], vec![])];

        let injections = injector.analyze_skills(&skills, &spec);
        assert_eq!(injections.len(), 1);
        match &injections[0].modification {
            InjectionType::AddGuard { guard, .. } => {
                assert_eq!(guard, "security_review_passed");
            }
            other => panic!("expected AddGuard, got {:?}", other),
        }
    }

    #[test]
    fn analyze_context_template_directive() {
        let injector = SkillInjector::default();
        let spec = make_rust_spec();
        let mut skill = make_skill("lint-enforcer", vec![], vec![]);
        skill.context_template = Some(
            serde_json::json!({
                "injections": [{
                    "type": "add_post_gate",
                    "wave_number": 1,
                    "gate_type": "clippy",
                    "params": { "deny_warnings": true },
                    "confidence": 0.85,
                    "rationale": "Run clippy after every wave"
                }]
            })
            .to_string(),
        );

        let injections = injector.analyze_skills(&[skill], &spec);
        assert_eq!(injections.len(), 1);
        match &injections[0].modification {
            InjectionType::AddPostGate { wave_number, gate } => {
                assert_eq!(*wave_number, 1);
                assert_eq!(gate.gate_type, "clippy");
                assert_eq!(
                    gate.params.get("deny_warnings"),
                    Some(&serde_json::json!(true))
                );
            }
            other => panic!("expected AddPostGate, got {:?}", other),
        }
        assert!((injections[0].confidence - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn analyze_multiple_skills() {
        let injector = SkillInjector::default();
        let spec = make_rust_spec_with_coverage();
        let skills = vec![
            make_skill(
                "strict-cov",
                vec!["coverage"],
                vec![("cfg", "threshold 95%")],
            ),
            make_skill("sec", vec!["security"], vec![]),
            make_skill(
                "tdd",
                vec!["testing"],
                vec![("note", "write tests first")],
            ),
        ];

        let injections = injector.analyze_skills(&skills, &spec);
        assert_eq!(injections.len(), 3);
    }

    #[test]
    fn analyze_skill_without_matching_tags() {
        let injector = SkillInjector::default();
        let spec = make_rust_spec();
        let skills = vec![make_skill("generic", vec!["documentation"], vec![])];

        let injections = injector.analyze_skills(&skills, &spec);
        assert!(injections.is_empty());
    }

    // -- apply_injections ------------------------------------------------------

    #[test]
    fn apply_override_coverage_threshold() {
        let injector = SkillInjector::default();
        let mut spec = make_rust_spec_with_coverage();

        // Verify the existing threshold is 80.
        let cov_gate = spec
            .final_gates
            .iter()
            .find(|g| g.gate_type == "coverage")
            .unwrap();
        assert_eq!(cov_gate.params["threshold"], serde_json::json!(80.0));

        let injections = vec![SkillInjection {
            skill_id: Uuid::new_v4(),
            skill_name: "cov-skill".into(),
            modification: InjectionType::OverrideCoverageThreshold {
                new_threshold: 95.0,
            },
            rationale: "raise coverage".into(),
            confidence: 0.8,
        }];

        let result = injector.apply_injections(&mut spec, &injections);
        assert_eq!(result.applied, 1);
        assert_eq!(result.skipped, 0);

        let cov_gate = spec
            .final_gates
            .iter()
            .find(|g| g.gate_type == "coverage")
            .unwrap();
        assert_eq!(cov_gate.params["threshold"], serde_json::json!(95.0));
    }

    #[test]
    fn apply_add_pre_gate() {
        let injector = SkillInjector::default();
        let mut spec = make_rust_spec();
        let pre_count = spec.stages[0].pre_gates.len();

        let injections = vec![SkillInjection {
            skill_id: Uuid::new_v4(),
            skill_name: "test-first".into(),
            modification: InjectionType::AddPreGate {
                wave_number: 1,
                gate: GateSpec {
                    gate_type: "cargo-test".into(),
                    params: HashMap::new(),
                },
            },
            rationale: "test first".into(),
            confidence: 0.7,
        }];

        let result = injector.apply_injections(&mut spec, &injections);
        assert_eq!(result.applied, 1);
        assert_eq!(spec.stages[0].pre_gates.len(), pre_count + 1);
        assert_eq!(spec.stages[0].pre_gates.last().unwrap().gate_type, "cargo-test");
    }

    #[test]
    fn apply_add_post_gate() {
        let injector = SkillInjector::default();
        let mut spec = make_rust_spec();
        let post_count = spec.stages[0].post_gates.len();

        let injections = vec![SkillInjection {
            skill_id: Uuid::new_v4(),
            skill_name: "clippy".into(),
            modification: InjectionType::AddPostGate {
                wave_number: 1,
                gate: GateSpec {
                    gate_type: "clippy".into(),
                    params: HashMap::new(),
                },
            },
            rationale: "lint".into(),
            confidence: 0.8,
        }];

        let result = injector.apply_injections(&mut spec, &injections);
        assert_eq!(result.applied, 1);
        assert_eq!(spec.stages[0].post_gates.len(), post_count + 1);
    }

    #[test]
    fn apply_add_guard() {
        let injector = SkillInjector::default();
        let mut spec = make_rust_spec();
        let final_count = spec.final_gates.len();

        let injections = vec![SkillInjection {
            skill_id: Uuid::new_v4(),
            skill_name: "sec".into(),
            modification: InjectionType::AddGuard {
                from_state: "done".into(),
                to_state: "done".into(),
                guard: "security_review_passed".into(),
            },
            rationale: "security".into(),
            confidence: 0.9,
        }];

        let result = injector.apply_injections(&mut spec, &injections);
        assert_eq!(result.applied, 1);
        // Guard is stored as an injected-guard final gate.
        assert_eq!(spec.final_gates.len(), final_count + 1);
        assert_eq!(
            spec.final_gates.last().unwrap().gate_type,
            "injected-guard"
        );
    }

    #[test]
    fn apply_modify_gate_params() {
        let injector = SkillInjector::default();
        let mut spec = make_rust_spec();

        let mut overrides = HashMap::new();
        overrides.insert("timeout".into(), serde_json::json!(120));

        let injections = vec![SkillInjection {
            skill_id: Uuid::new_v4(),
            skill_name: "timeout".into(),
            modification: InjectionType::ModifyGateParams {
                gate_type: "cargo-test".into(),
                param_overrides: overrides,
            },
            rationale: "increase timeout".into(),
            confidence: 0.7,
        }];

        let result = injector.apply_injections(&mut spec, &injections);
        assert_eq!(result.applied, 1);

        // Verify the param was added.
        let gate = spec.stages[0]
            .post_gates
            .iter()
            .find(|g| g.gate_type == "cargo-test")
            .unwrap();
        assert_eq!(gate.params["timeout"], serde_json::json!(120));
    }

    // -- confidence threshold filtering ----------------------------------------

    #[test]
    fn skip_low_confidence_injection() {
        let injector = SkillInjector::new(0.8);
        let mut spec = make_rust_spec();

        let injections = vec![SkillInjection {
            skill_id: Uuid::new_v4(),
            skill_name: "weak".into(),
            modification: InjectionType::AddPreGate {
                wave_number: 1,
                gate: GateSpec {
                    gate_type: "maybe-lint".into(),
                    params: HashMap::new(),
                },
            },
            rationale: "maybe lint".into(),
            confidence: 0.3, // below 0.8 threshold
        }];

        let result = injector.apply_injections(&mut spec, &injections);
        assert_eq!(result.applied, 0);
        assert_eq!(result.skipped, 1);
        assert!(!result.trace[0].applied);
        assert!(result.trace[0].reason.as_ref().unwrap().contains("confidence"));
    }

    #[test]
    fn apply_high_confidence_skip_low() {
        let injector = SkillInjector::new(0.6);
        let mut spec = make_rust_spec();

        let injections = vec![
            SkillInjection {
                skill_id: Uuid::new_v4(),
                skill_name: "strong".into(),
                modification: InjectionType::AddPreGate {
                    wave_number: 1,
                    gate: GateSpec {
                        gate_type: "lint".into(),
                        params: HashMap::new(),
                    },
                },
                rationale: "lint".into(),
                confidence: 0.9,
            },
            SkillInjection {
                skill_id: Uuid::new_v4(),
                skill_name: "weak".into(),
                modification: InjectionType::AddPreGate {
                    wave_number: 1,
                    gate: GateSpec {
                        gate_type: "maybe".into(),
                        params: HashMap::new(),
                    },
                },
                rationale: "maybe".into(),
                confidence: 0.2,
            },
        ];

        let result = injector.apply_injections(&mut spec, &injections);
        assert_eq!(result.applied, 1);
        assert_eq!(result.skipped, 1);
    }

    // -- trace -----------------------------------------------------------------

    #[test]
    fn trace_records_all_attempts() {
        let injector = SkillInjector::new(0.5);
        let mut spec = make_rust_spec();

        let injections = vec![
            SkillInjection {
                skill_id: Uuid::new_v4(),
                skill_name: "good".into(),
                modification: InjectionType::AddPreGate {
                    wave_number: 1,
                    gate: GateSpec {
                        gate_type: "lint".into(),
                        params: HashMap::new(),
                    },
                },
                rationale: "lint wave 1".into(),
                confidence: 0.8,
            },
            SkillInjection {
                skill_id: Uuid::new_v4(),
                skill_name: "low-conf".into(),
                modification: InjectionType::AddPreGate {
                    wave_number: 1,
                    gate: GateSpec {
                        gate_type: "opt".into(),
                        params: HashMap::new(),
                    },
                },
                rationale: "optional".into(),
                confidence: 0.1,
            },
            SkillInjection {
                skill_id: Uuid::new_v4(),
                skill_name: "bad-wave".into(),
                modification: InjectionType::AddPreGate {
                    wave_number: 99,
                    gate: GateSpec {
                        gate_type: "nope".into(),
                        params: HashMap::new(),
                    },
                },
                rationale: "nonexistent wave".into(),
                confidence: 0.9,
            },
        ];

        let result = injector.apply_injections(&mut spec, &injections);
        assert_eq!(result.trace.len(), 3);

        // First: applied
        assert!(result.trace[0].applied);
        assert!(result.trace[0].reason.is_none());

        // Second: skipped (low confidence)
        assert!(!result.trace[1].applied);
        assert!(result.trace[1].reason.as_ref().unwrap().contains("confidence"));

        // Third: skipped (wave not found)
        assert!(!result.trace[2].applied);
        assert!(result.trace[2].reason.as_ref().unwrap().contains("wave 99"));
    }

    // -- inject (convenience) --------------------------------------------------

    #[test]
    fn inject_end_to_end() {
        let injector = SkillInjector::default();
        let mut spec = make_rust_spec_with_coverage();

        let skills = vec![
            make_skill(
                "strict-cov",
                vec!["coverage"],
                vec![("cfg", "threshold 95%")],
            ),
            make_skill("sec", vec!["security"], vec![]),
        ];

        let result = injector.inject(&skills, &mut spec);
        // Coverage override + security guard = 2 applied.
        assert_eq!(result.applied, 2);
        assert_eq!(result.skipped, 0);

        // Verify coverage was updated.
        let cov_gate = spec
            .final_gates
            .iter()
            .find(|g| g.gate_type == "coverage")
            .unwrap();
        assert_eq!(cov_gate.params["threshold"], serde_json::json!(95.0));

        // Verify guard was added.
        assert!(spec
            .final_gates
            .iter()
            .any(|g| g.gate_type == "injected-guard"));
    }
}
