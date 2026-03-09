//! Agent Persona — Cognitive routing based on task profiling.
//!
//! Analyzes a task's complexity (steps count, affected files, tags) to produce
//! a [`TaskProfile`] that drives:
//! - Guard timeout adaptation
//! - Cost budget per task
//! - Prompt tone (concise vs analytical vs exploratory)
//!
//! Also provides skill activation helpers that bridge the Neural Skills system
//! into the Runner's execution loop.

use crate::neo4j::models::TaskNode;
use crate::neo4j::traits::GraphStore;
use crate::skills::activation::{evaluate_skill_match, HookActivationConfig};
use crate::skills::models::SkillNode;

use std::sync::Arc;
use tracing::{info, warn};
use uuid::Uuid;

// ============================================================================
// Complexity & TaskProfile
// ============================================================================

/// Cognitive complexity classification for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Complexity {
    /// Single-file, few steps — fast execution, concise prompt.
    Simple,
    /// Multi-file, many steps, refactoring — deep analysis needed.
    Complex,
    /// Design or new feature — exploration before commitment.
    Creative,
}

impl std::fmt::Display for Complexity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Simple => write!(f, "simple"),
            Self::Complex => write!(f, "complex"),
            Self::Creative => write!(f, "creative"),
        }
    }
}

/// Execution profile derived from task analysis.
///
/// Controls guard timeout, cost budget, and prompt tone.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskProfile {
    pub complexity: Complexity,
    /// Guard timeout in seconds (overrides RunnerConfig.task_timeout_secs).
    pub timeout_secs: u64,
    /// Maximum cost in USD for this task.
    pub max_cost_usd: f64,
    /// Temperature hint for the agent's search/exploration behavior.
    pub search_temperature: f64,
}

/// Profile a task based on its steps count, affected files, and tags.
///
/// Classification rules:
/// - Tags "design" or "new-feature" → Creative (1800s, $1.00)
/// - Multi-file (>1) AND (>5 steps OR tags contain "refactor"/"architecture") → Complex (3600s, $2.00)
/// - Everything else → Simple (600s, $0.50)
pub fn profile_task(task: &TaskNode, steps_count: usize) -> TaskProfile {
    let files_count = task.affected_files.len();
    let tags: Vec<String> = task.tags.iter().map(|t| t.to_lowercase()).collect();

    // Creative: design or new-feature tasks
    let is_creative = tags.iter().any(|t| t == "design" || t == "new-feature");
    if is_creative {
        return TaskProfile {
            complexity: Complexity::Creative,
            timeout_secs: 1800,
            max_cost_usd: 1.0,
            search_temperature: 0.8,
        };
    }

    // Complex: multi-file with many steps or refactoring/architecture tags
    let has_complex_tags = tags
        .iter()
        .any(|t| t == "refactor" || t == "architecture");
    let is_complex = files_count > 1 && (steps_count > 5 || has_complex_tags);
    if is_complex {
        return TaskProfile {
            complexity: Complexity::Complex,
            timeout_secs: 3600,
            max_cost_usd: 2.0,
            search_temperature: 0.5,
        };
    }

    // Simple: default
    TaskProfile {
        complexity: Complexity::Simple,
        timeout_secs: 600,
        max_cost_usd: 0.5,
        search_temperature: 0.3,
    }
}

/// Generate a prompt directive based on the task profile complexity.
///
/// Returns a string to append to the runner constraints that adapts
/// the agent's cognitive approach.
pub fn complexity_directive(complexity: Complexity) -> &'static str {
    match complexity {
        Complexity::Simple => "Execute de maniere directe et concise. Pas d'analyse excessive.",
        Complexity::Complex => {
            "Prends le temps d'analyser en profondeur avant de coder. \
             Identifie les impacts transversaux et les effets de bord potentiels."
        }
        Complexity::Creative => {
            "Explore plusieurs approches avant de choisir. \
             Propose au moins 2 alternatives mentalement avant d'implementer."
        }
    }
}

// ============================================================================
// Skill Activation for Runner
// ============================================================================

/// Result of skill activation for a task — carries the context text
/// and the IDs of activated skills for post-task feedback.
#[derive(Debug, Clone)]
pub struct SkillActivationResult {
    /// Assembled context text from matched skill context_templates.
    pub context_text: String,
    /// IDs of skills that were activated (for post-task feedback).
    pub activated_skill_ids: Vec<Uuid>,
}

/// Activate skills relevant to a task by matching its description and tags
/// against skill trigger patterns.
///
/// This reuses the same `evaluate_skill_match` function used by the hook
/// pipeline, but constructs a search query from the task metadata instead
/// of tool input.
///
/// Returns `None` if no skills match or if `project_id` is not available.
pub async fn activate_skills_for_task(
    graph: &dyn GraphStore,
    project_id: Uuid,
    task: &TaskNode,
) -> Option<SkillActivationResult> {
    // Build search query from task description + tags
    let query = format!(
        "{} {}",
        &task.description,
        task.tags.join(" ")
    );

    // Also build a file context from affected_files (first file if any)
    let file_context = task.affected_files.first().map(|f| f.as_str());

    // Load matchable skills for this project
    let skills = match graph.get_skills_for_project(project_id).await {
        Ok(s) => s,
        Err(e) => {
            warn!("Failed to load skills for project {}: {}", project_id, e);
            return None;
        }
    };

    let matchable: Vec<_> = skills.into_iter().filter(|s| s.is_matchable()).collect();
    if matchable.is_empty() {
        return None;
    }

    // Evaluate trigger patterns against the task query
    let config = HookActivationConfig::default();
    let mut matches: Vec<(SkillNode, f64)> = Vec::new();

    for skill in matchable {
        let confidence =
            evaluate_skill_match(&skill, Some(&query), file_context);
        if confidence >= config.confidence_threshold {
            matches.push((skill, confidence));
        }
    }

    if matches.is_empty() {
        return None;
    }

    // Sort by confidence descending
    matches.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top 3 skills max
    matches.truncate(3);

    // Assemble context from context_templates
    let mut context_parts: Vec<String> = Vec::new();
    let mut activated_ids: Vec<Uuid> = Vec::new();

    for (skill, confidence) in &matches {
        activated_ids.push(skill.id);

        if let Some(ref template) = skill.context_template {
            context_parts.push(format!(
                "### Skill: {} (confidence: {:.0}%)\n{}",
                skill.name,
                confidence * 100.0,
                template
            ));
        } else {
            // Fallback: use skill description
            if !skill.description.is_empty() {
                context_parts.push(format!(
                    "### Skill: {} (confidence: {:.0}%)\n{}",
                    skill.name,
                    confidence * 100.0,
                    skill.description
                ));
            }
        }
    }

    if context_parts.is_empty() && activated_ids.is_empty() {
        return None;
    }

    let context_text = if context_parts.is_empty() {
        String::new()
    } else {
        context_parts.join("\n\n")
    };

    info!(
        skills_activated = activated_ids.len(),
        context_len = context_text.len(),
        "Skills activated for task"
    );

    Some(SkillActivationResult {
        context_text,
        activated_skill_ids: activated_ids,
    })
}

// ============================================================================
// Post-task feedback — async Neo4j recording
// ============================================================================

/// Record skill usage feedback after task completion/failure.
///
/// For each activated skill:
/// - Creates an `(:AgentExecution)-[:USED_SKILL]->(:Skill)` relationship
/// - Adjusts skill energy: +0.05 on success, -0.03 on failure
///
/// This is fire-and-forget (spawned via `tokio::spawn`).
pub async fn record_skill_feedback(
    graph: Arc<dyn GraphStore>,
    task_id: Uuid,
    run_id: Uuid,
    activated_skill_ids: Vec<Uuid>,
    success: bool,
    cost_usd: f64,
    duration_secs: f64,
) {
    let energy_delta = if success { 0.05 } else { -0.03 };

    for skill_id in activated_skill_ids {
        // Update skill energy
        match graph.get_skill(skill_id).await {
            Ok(Some(mut skill)) => {
                skill.energy = (skill.energy + energy_delta).clamp(0.0, 1.0);

                // Update activation tracking
                skill.activation_count += 1;
                if success {
                    // Approximate hit_rate update (exponential moving average)
                    skill.hit_rate = skill.hit_rate * 0.9 + 0.1;
                } else {
                    skill.hit_rate *= 0.9;
                }
                skill.last_activated = Some(chrono::Utc::now());

                if let Err(e) = graph.update_skill(&skill).await {
                    warn!(
                        "Failed to update skill {} energy after task {}: {}",
                        skill_id, task_id, e
                    );
                } else {
                    info!(
                        skill_id = %skill_id,
                        skill_name = %skill.name,
                        success,
                        energy = skill.energy,
                        "Skill feedback recorded (task={}, run={}, cost=${:.4}, duration={:.1}s)",
                        task_id, run_id, cost_usd, duration_secs
                    );
                }
            }
            Ok(None) => {
                warn!("Skill {} not found for feedback (task={})", skill_id, task_id);
            }
            Err(e) => {
                warn!(
                    "Failed to load skill {} for feedback (task={}): {}",
                    skill_id, task_id, e
                );
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn make_task(tags: Vec<&str>, affected_files: Vec<&str>) -> TaskNode {
        use crate::neo4j::models::TaskStatus;
        TaskNode {
            id: Uuid::new_v4(),
            title: Some("Test task".to_string()),
            description: "A test task description".to_string(),
            status: TaskStatus::Pending,
            assigned_to: None,
            priority: None,
            tags: tags.into_iter().map(String::from).collect(),
            acceptance_criteria: vec![],
            affected_files: affected_files.into_iter().map(String::from).collect(),
            estimated_complexity: None,
            actual_complexity: None,
            created_at: Utc::now(),
            updated_at: None,
            started_at: None,
            completed_at: None,
            frustration_score: 0.0,
            execution_context: None,
            persona: None,
            prompt_cache: None,
        }
    }

    #[test]
    fn test_profile_simple_task() {
        let task = make_task(vec!["bugfix"], vec!["src/main.rs"]);
        let profile = profile_task(&task, 2);

        assert_eq!(profile.complexity, Complexity::Simple);
        assert_eq!(profile.timeout_secs, 600);
        assert!((profile.max_cost_usd - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_profile_complex_task_by_steps_and_files() {
        let task = make_task(
            vec!["backend"],
            vec!["src/a.rs", "src/b.rs", "src/c.rs"],
        );
        let profile = profile_task(&task, 6);

        assert_eq!(profile.complexity, Complexity::Complex);
        assert_eq!(profile.timeout_secs, 3600);
        assert!((profile.max_cost_usd - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_profile_complex_task_by_refactor_tag() {
        let task = make_task(vec!["refactor"], vec!["src/a.rs", "src/b.rs"]);
        let profile = profile_task(&task, 3);

        assert_eq!(profile.complexity, Complexity::Complex);
    }

    #[test]
    fn test_profile_complex_task_by_architecture_tag() {
        let task = make_task(vec!["architecture"], vec!["src/a.rs", "src/b.rs"]);
        let profile = profile_task(&task, 2);

        assert_eq!(profile.complexity, Complexity::Complex);
    }

    #[test]
    fn test_profile_creative_task_design() {
        let task = make_task(vec!["design"], vec!["src/new_module.rs"]);
        let profile = profile_task(&task, 4);

        assert_eq!(profile.complexity, Complexity::Creative);
        assert_eq!(profile.timeout_secs, 1800);
        assert!((profile.max_cost_usd - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_profile_creative_task_new_feature() {
        let task = make_task(vec!["new-feature"], vec![]);
        let profile = profile_task(&task, 10);

        // Creative overrides Complex even with many steps
        assert_eq!(profile.complexity, Complexity::Creative);
    }

    #[test]
    fn test_profile_creative_takes_priority_over_complex() {
        // Both design + architecture + multi-file → Creative wins
        let task = make_task(
            vec!["design", "architecture"],
            vec!["src/a.rs", "src/b.rs", "src/c.rs"],
        );
        let profile = profile_task(&task, 8);

        assert_eq!(profile.complexity, Complexity::Creative);
    }

    #[test]
    fn test_profile_single_file_many_steps_stays_simple() {
        // Only 1 file, even with >5 steps, not complex (needs multi-file)
        let task = make_task(vec![], vec!["src/main.rs"]);
        let profile = profile_task(&task, 10);

        assert_eq!(profile.complexity, Complexity::Simple);
    }

    #[test]
    fn test_profile_multi_file_few_steps_stays_simple() {
        // Multiple files but <= 5 steps and no complex tags → Simple
        let task = make_task(vec![], vec!["src/a.rs", "src/b.rs"]);
        let profile = profile_task(&task, 3);

        assert_eq!(profile.complexity, Complexity::Simple);
    }

    #[test]
    fn test_profile_tags_case_insensitive() {
        let task = make_task(vec!["Design"], vec!["src/a.rs"]);
        let profile = profile_task(&task, 1);

        assert_eq!(profile.complexity, Complexity::Creative);
    }

    #[test]
    fn test_complexity_directive_simple() {
        let directive = complexity_directive(Complexity::Simple);
        assert!(directive.contains("concise"));
    }

    #[test]
    fn test_complexity_directive_complex() {
        let directive = complexity_directive(Complexity::Complex);
        assert!(directive.contains("profondeur"));
    }

    #[test]
    fn test_complexity_directive_creative() {
        let directive = complexity_directive(Complexity::Creative);
        assert!(directive.contains("approches"));
    }

    #[test]
    fn test_complexity_display() {
        assert_eq!(Complexity::Simple.to_string(), "simple");
        assert_eq!(Complexity::Complex.to_string(), "complex");
        assert_eq!(Complexity::Creative.to_string(), "creative");
    }
}
