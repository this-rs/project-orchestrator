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

use crate::neo4j::models::{PersonaSubgraph, TaskNode};
use crate::neo4j::traits::GraphStore;
use crate::skills::activation::{evaluate_skill_match, HookActivationConfig};
use crate::skills::models::SkillNode;

use chrono::{DateTime, Utc};
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

// ============================================================================
// PersonaStack — Weighted persona context for Runner prompt injection
// ============================================================================

/// How a persona was activated (determines initial weight and decay behavior).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersonaTrigger {
    /// Explicitly assigned to a task via `task.persona` field.
    TaskAssign,
    /// Matched by file affinity (file in `affected_files` has KNOWS relation).
    FileMatch,
    /// Hinted by a step's `persona` field.
    StepHint,
    /// Inherited from a parent persona via EXTENDS chain.
    Inherited,
    /// Matched by community affinity (same Louvain cluster).
    CommunityMatch,
}

impl std::fmt::Display for PersonaTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TaskAssign => write!(f, "task_assign"),
            Self::FileMatch => write!(f, "file_match"),
            Self::StepHint => write!(f, "step_hint"),
            Self::Inherited => write!(f, "inherited"),
            Self::CommunityMatch => write!(f, "community_match"),
        }
    }
}

/// A single persona entry in the stack, with its activation metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersonaEntry {
    /// The persona ID.
    pub persona_id: Uuid,
    /// Persona name (cached for prompt rendering).
    pub persona_name: String,
    /// Relevance weight [0,1] — determines prompt budget share.
    pub weight: f64,
    /// When this persona was activated.
    pub activated_at: DateTime<Utc>,
    /// How it was triggered (affects decay rate).
    pub trigger: PersonaTrigger,
    /// Cached subgraph for prompt rendering (loaded once on activation).
    pub cached_subgraph: PersonaSubgraph,
}

/// Weighted stack of personas for a task execution.
///
/// The primary persona (highest weight) gets ~70% of the persona token budget.
/// Remaining personas share the rest proportionally.
/// Entries decay over time and are evicted below a threshold.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PersonaStack {
    /// Active persona entries, sorted by weight descending.
    entries: Vec<PersonaEntry>,
    /// Maximum token budget for persona context in the prompt.
    pub max_context_tokens: usize,
}

impl PersonaStack {
    /// Create a new empty PersonaStack with a default token budget.
    pub fn new(max_context_tokens: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_context_tokens,
        }
    }

    /// Push a persona entry onto the stack, maintaining weight-descending order.
    pub fn push(&mut self, entry: PersonaEntry) {
        // Don't add duplicates — update weight if already present
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|e| e.persona_id == entry.persona_id)
        {
            existing.weight = existing.weight.max(entry.weight);
            existing.trigger = entry.trigger;
        } else {
            self.entries.push(entry);
        }
        self.entries.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get the primary (highest-weight) persona, if any.
    pub fn get_primary(&self) -> Option<&PersonaEntry> {
        self.entries.first()
    }

    /// Get all entries (sorted by weight descending).
    pub fn entries(&self) -> &[PersonaEntry] {
        &self.entries
    }

    /// Number of personas in the stack.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Decay all entry weights by elapsed time.
    ///
    /// Decay rate depends on trigger type:
    /// - TaskAssign: very slow decay (0.01/min) — explicitly assigned
    /// - FileMatch/StepHint: moderate decay (0.03/min)
    /// - Inherited/CommunityMatch: fast decay (0.05/min)
    pub fn decay_all(&mut self, elapsed_secs: f64) {
        let elapsed_mins = elapsed_secs / 60.0;
        for entry in &mut self.entries {
            let rate = match entry.trigger {
                PersonaTrigger::TaskAssign => 0.01,
                PersonaTrigger::FileMatch | PersonaTrigger::StepHint => 0.03,
                PersonaTrigger::Inherited | PersonaTrigger::CommunityMatch => 0.05,
            };
            entry.weight = (entry.weight - rate * elapsed_mins).max(0.0);
        }
    }

    /// Evict entries below the given weight threshold.
    pub fn evict_below(&mut self, threshold: f64) {
        self.entries.retain(|e| e.weight >= threshold);
    }

    /// Render the persona stack into a prompt string for injection.
    ///
    /// Budget allocation:
    /// - Primary persona: 70% of budget
    /// - Remaining personas: 30% shared proportionally by weight
    ///
    /// Each persona section includes:
    /// - Name, trigger, weight
    /// - System prompt override (if any, from PersonaNode via subgraph)
    /// - Top files/functions it KNOWS (by weight)
    /// - Top notes it USES
    /// - Top decisions it USES
    pub fn render_for_prompt(&self, token_budget: usize) -> String {
        if self.entries.is_empty() {
            return String::new();
        }

        let mut sections: Vec<String> = Vec::new();

        // Primary persona gets 70% of budget
        let primary_budget = (token_budget as f64 * 0.7) as usize;
        let secondary_budget = token_budget.saturating_sub(primary_budget);

        if let Some(primary) = self.entries.first() {
            sections.push(Self::render_entry(primary, primary_budget, true));
        }

        // Remaining personas share 30%
        let rest = &self.entries[1..];
        if !rest.is_empty() {
            let total_weight: f64 = rest.iter().map(|e| e.weight).sum();
            for entry in rest {
                let share = if total_weight > 0.0 {
                    (entry.weight / total_weight * secondary_budget as f64) as usize
                } else {
                    secondary_budget / rest.len()
                };
                if share > 50 {
                    // Don't render tiny sections
                    sections.push(Self::render_entry(entry, share, false));
                }
            }
        }

        sections.join("\n")
    }

    /// Render a single persona entry as a markdown section.
    fn render_entry(entry: &PersonaEntry, budget_chars: usize, is_primary: bool) -> String {
        let mut out = String::new();
        let label = if is_primary {
            "Primary Persona"
        } else {
            "Secondary Persona"
        };

        out.push_str(&format!(
            "### {} — {} (weight: {:.2}, trigger: {})\n",
            label, entry.persona_name, entry.weight, entry.trigger
        ));

        let sub = &entry.cached_subgraph;

        // Files it knows (top by weight, limited by budget)
        if !sub.files.is_empty() {
            out.push_str("**Known files:**\n");
            let max_files = (budget_chars / 100).clamp(3, 15);
            for rel in sub.files.iter().take(max_files) {
                out.push_str(&format!("- `{}` (w:{:.2})\n", rel.entity_id, rel.weight));
            }
        }

        // Functions it knows
        if !sub.functions.is_empty() {
            out.push_str("**Known functions:**\n");
            let max_fns = (budget_chars / 120).clamp(2, 10);
            for rel in sub.functions.iter().take(max_fns) {
                out.push_str(&format!("- `{}` (w:{:.2})\n", rel.entity_id, rel.weight));
            }
        }

        // Notes it uses (show content snippets)
        if !sub.notes.is_empty() {
            out.push_str("**Knowledge notes:**\n");
            let max_notes = (budget_chars / 200).clamp(2, 8);
            for rel in sub.notes.iter().take(max_notes) {
                out.push_str(&format!(
                    "- [note:{}] (w:{:.2})\n",
                    rel.entity_id, rel.weight
                ));
            }
        }

        // Decisions it uses
        if !sub.decisions.is_empty() {
            out.push_str("**Decisions:**\n");
            let max_decisions = (budget_chars / 200).clamp(1, 5);
            for rel in sub.decisions.iter().take(max_decisions) {
                out.push_str(&format!(
                    "- [decision:{}] (w:{:.2})\n",
                    rel.entity_id, rel.weight
                ));
            }
        }

        // Truncate if over budget
        if out.len() > budget_chars {
            out.truncate(budget_chars.saturating_sub(4));
            out.push_str("...\n");
        }

        out
    }
}

/// Load a PersonaStack for a task, resolving persona from multiple sources.
///
/// Resolution order:
/// 1. `task.persona` field (JSON string containing persona_id) → TaskAssign
/// 2. File affinity: match `affected_files` against personas via `find_personas_for_file` → FileMatch
/// 3. Step hints: parse `step.persona` fields → StepHint (included as metadata)
/// 4. Inherited: follow EXTENDS chain of primary persona → Inherited
///
/// Returns None if no personas are found.
pub async fn load_persona_stack(
    graph: &dyn GraphStore,
    task: &TaskNode,
    project_id: Uuid,
    steps: &[crate::neo4j::models::StepNode],
) -> Option<PersonaStack> {
    let mut stack = PersonaStack::new(4000); // ~4000 chars budget
    let now = Utc::now();

    // 1. Task-level persona assignment
    if let Some(ref persona_json) = task.persona {
        if let Ok(persona_id) = persona_json.trim_matches('"').parse::<Uuid>() {
            match load_and_push(
                graph,
                persona_id,
                PersonaTrigger::TaskAssign,
                1.0,
                now,
                &mut stack,
            )
            .await
            {
                Ok(true) => {
                    info!(persona_id = %persona_id, "Persona loaded from task assignment");
                }
                Ok(false) => {
                    warn!(persona_id = %persona_id, "Task-assigned persona not found");
                }
                Err(e) => {
                    warn!(persona_id = %persona_id, error = %e, "Failed to load task persona");
                }
            }
        }
    }

    // 2. File affinity — find personas that KNOW the affected files
    if stack.is_empty() {
        // Only do file matching if no explicit persona was assigned
        for file_path in &task.affected_files {
            match graph.find_personas_for_file(file_path, project_id).await {
                Ok(matches) => {
                    for (persona, weight) in matches.into_iter().take(2) {
                        // Cap file-match weight at 0.8 (lower than explicit assignment)
                        let w = (weight * 0.8).min(0.8);
                        let _ = load_and_push(
                            graph,
                            persona.id,
                            PersonaTrigger::FileMatch,
                            w,
                            now,
                            &mut stack,
                        )
                        .await;
                    }
                }
                Err(e) => {
                    debug!(file_path = %file_path, error = %e, "Failed to find personas for file");
                }
            }
            // Stop after finding at least one persona
            if !stack.is_empty() {
                break;
            }
        }
    }

    // 3. Step-level persona hints (added as metadata, lower weight)
    for step in steps {
        if let Some(ref hint_json) = step.persona {
            if let Ok(hint_id) = hint_json.trim_matches('"').parse::<Uuid>() {
                // Don't duplicate the primary persona
                if stack.entries().iter().any(|e| e.persona_id == hint_id) {
                    continue;
                }
                let _ = load_and_push(
                    graph,
                    hint_id,
                    PersonaTrigger::StepHint,
                    0.5,
                    now,
                    &mut stack,
                )
                .await;
            }
        }
    }

    // 4. Inherited — follow EXTENDS chain of primary persona
    if let Some(primary) = stack.get_primary().cloned() {
        let parents = primary.cached_subgraph.parents.clone();
        for (i, parent_rel) in parents.iter().enumerate() {
            // Weight decreases with distance: 0.4, 0.2, 0.1...
            let w = 0.4 / (i as f64 + 1.0);
            if w < 0.05 {
                break;
            }
            if let Ok(parent_id) = parent_rel.entity_id.parse::<Uuid>() {
                let _ = load_and_push(
                    graph,
                    parent_id,
                    PersonaTrigger::Inherited,
                    w,
                    now,
                    &mut stack,
                )
                .await;
            }
        }
    }

    if stack.is_empty() {
        None
    } else {
        info!(
            persona_count = stack.len(),
            primary = ?stack.get_primary().map(|p| &p.persona_name),
            "PersonaStack loaded for task"
        );
        Some(stack)
    }
}

/// Helper: load a persona's subgraph and push it onto the stack.
/// Returns Ok(true) if loaded successfully, Ok(false) if not found.
async fn load_and_push(
    graph: &dyn GraphStore,
    persona_id: Uuid,
    trigger: PersonaTrigger,
    weight: f64,
    activated_at: DateTime<Utc>,
    stack: &mut PersonaStack,
) -> Result<bool, anyhow::Error> {
    // Check if already in stack
    if stack.entries().iter().any(|e| e.persona_id == persona_id) {
        return Ok(true);
    }

    let subgraph = graph.get_persona_subgraph(persona_id).await?;
    stack.push(PersonaEntry {
        persona_id,
        persona_name: subgraph.persona_name.clone(),
        weight,
        activated_at,
        trigger,
        cached_subgraph: subgraph,
    });
    Ok(true)
}

/// Record persona feedback after task completion/failure.
///
/// Adjusts the persona's energy: +0.05 on success, -0.03 on failure.
/// Also updates activation_count and success_rate.
pub async fn record_persona_feedback(
    graph: Arc<dyn GraphStore>,
    persona_ids: Vec<Uuid>,
    success: bool,
) {
    let energy_delta = if success { 0.05 } else { -0.03 };

    for persona_id in persona_ids {
        match graph.get_persona(persona_id).await {
            Ok(Some(mut persona)) => {
                persona.energy = (persona.energy + energy_delta).clamp(0.0, 1.0);
                persona.activation_count += 1;
                if success {
                    // Exponential moving average for success_rate
                    persona.success_rate = persona.success_rate * 0.9 + 0.1;
                } else {
                    persona.success_rate *= 0.9;
                }
                persona.last_activated = Some(Utc::now());

                if let Err(e) = graph.update_persona(&persona).await {
                    warn!(
                        persona_id = %persona_id,
                        error = %e,
                        "Failed to update persona feedback"
                    );
                } else {
                    info!(
                        persona_id = %persona_id,
                        persona_name = %persona.name,
                        success,
                        energy = persona.energy,
                        "Persona feedback recorded"
                    );
                }
            }
            Ok(None) => {
                warn!(persona_id = %persona_id, "Persona not found for feedback");
            }
            Err(e) => {
                warn!(
                    persona_id = %persona_id,
                    error = %e,
                    "Failed to load persona for feedback"
                );
            }
        }
    }
}

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
    let has_complex_tags = tags.iter().any(|t| t == "refactor" || t == "architecture");
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
    let query = format!("{} {}", &task.description, task.tags.join(" "));

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
        let confidence = evaluate_skill_match(&skill, Some(&query), file_context);
        if confidence >= config.confidence_threshold {
            matches.push((skill, confidence));
        }
    }

    if matches.is_empty() {
        return None;
    }

    // Sort by confidence descending
    matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
                warn!(
                    "Skill {} not found for feedback (task={})",
                    skill_id, task_id
                );
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
    use crate::neo4j::models::{PersonaSubgraphStats, PersonaWeightedRelation};
    use chrono::Utc;
    use uuid::Uuid;

    fn make_subgraph(name: &str) -> PersonaSubgraph {
        PersonaSubgraph {
            persona_id: Uuid::new_v4(),
            persona_name: name.to_string(),
            files: vec![
                PersonaWeightedRelation {
                    entity_type: "file".to_string(),
                    entity_id: "src/main.rs".to_string(),
                    weight: 0.9,
                },
                PersonaWeightedRelation {
                    entity_type: "file".to_string(),
                    entity_id: "src/lib.rs".to_string(),
                    weight: 0.7,
                },
            ],
            functions: vec![PersonaWeightedRelation {
                entity_type: "function".to_string(),
                entity_id: "handle_request".to_string(),
                weight: 0.8,
            }],
            notes: vec![PersonaWeightedRelation {
                entity_type: "note".to_string(),
                entity_id: Uuid::new_v4().to_string(),
                weight: 0.6,
            }],
            decisions: vec![],
            skill_ids: vec![],
            protocol_ids: vec![],
            feature_graph_id: None,
            parent_ids: vec![],
            stats: PersonaSubgraphStats {
                total_entities: 4,
                coverage_score: 0.3,
                freshness: 0.8,
            },
        }
    }

    fn make_entry(name: &str, weight: f64, trigger: PersonaTrigger) -> PersonaEntry {
        PersonaEntry {
            persona_id: Uuid::new_v4(),
            persona_name: name.to_string(),
            weight,
            activated_at: Utc::now(),
            trigger,
            cached_subgraph: make_subgraph(name),
        }
    }

    // ========================================================================
    // PersonaStack tests
    // ========================================================================

    #[test]
    fn test_persona_stack_empty() {
        let stack = PersonaStack::new(4000);
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
        assert!(stack.get_primary().is_none());
        assert!(stack.render_for_prompt(4000).is_empty());
    }

    #[test]
    fn test_persona_stack_push_and_order() {
        let mut stack = PersonaStack::new(4000);
        stack.push(make_entry("low", 0.3, PersonaTrigger::FileMatch));
        stack.push(make_entry("high", 0.9, PersonaTrigger::TaskAssign));
        stack.push(make_entry("mid", 0.5, PersonaTrigger::StepHint));

        assert_eq!(stack.len(), 3);
        assert_eq!(stack.get_primary().unwrap().persona_name, "high");
        assert_eq!(stack.entries()[1].persona_name, "mid");
        assert_eq!(stack.entries()[2].persona_name, "low");
    }

    #[test]
    fn test_persona_stack_dedup() {
        let mut stack = PersonaStack::new(4000);
        let entry1 = make_entry("expert", 0.5, PersonaTrigger::FileMatch);
        let id = entry1.persona_id;
        stack.push(entry1);

        // Push same persona with higher weight — should update, not duplicate
        let mut entry2 = make_entry("expert", 0.9, PersonaTrigger::TaskAssign);
        entry2.persona_id = id;
        stack.push(entry2);

        assert_eq!(stack.len(), 1);
        assert_eq!(stack.get_primary().unwrap().weight, 0.9);
        assert_eq!(
            stack.get_primary().unwrap().trigger,
            PersonaTrigger::TaskAssign
        );
    }

    #[test]
    fn test_persona_stack_decay() {
        let mut stack = PersonaStack::new(4000);
        stack.push(make_entry("assigned", 1.0, PersonaTrigger::TaskAssign));
        stack.push(make_entry("inherited", 0.5, PersonaTrigger::Inherited));

        // Decay for 10 minutes
        stack.decay_all(600.0);

        // TaskAssign decays slowly: 1.0 - 0.01*10 = 0.9
        assert!((stack.entries()[0].weight - 0.9).abs() < 0.01);
        // Inherited decays fast: 0.5 - 0.05*10 = 0.0
        assert!(stack.entries()[1].weight < 0.01);
    }

    #[test]
    fn test_persona_stack_evict() {
        let mut stack = PersonaStack::new(4000);
        stack.push(make_entry("strong", 0.8, PersonaTrigger::TaskAssign));
        stack.push(make_entry("weak", 0.1, PersonaTrigger::Inherited));

        stack.evict_below(0.2);
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.get_primary().unwrap().persona_name, "strong");
    }

    #[test]
    fn test_persona_stack_render() {
        let mut stack = PersonaStack::new(4000);
        stack.push(make_entry("neo4j-expert", 0.9, PersonaTrigger::TaskAssign));
        stack.push(make_entry(
            "rust-specialist",
            0.4,
            PersonaTrigger::FileMatch,
        ));

        let rendered = stack.render_for_prompt(4000);
        assert!(rendered.contains("Primary Persona"));
        assert!(rendered.contains("neo4j-expert"));
        assert!(rendered.contains("Secondary Persona"));
        assert!(rendered.contains("rust-specialist"));
        assert!(rendered.contains("src/main.rs"));
        assert!(rendered.contains("handle_request"));
    }

    #[test]
    fn test_persona_stack_render_single() {
        let mut stack = PersonaStack::new(4000);
        stack.push(make_entry("solo", 1.0, PersonaTrigger::TaskAssign));

        let rendered = stack.render_for_prompt(4000);
        assert!(rendered.contains("Primary Persona"));
        assert!(rendered.contains("solo"));
        assert!(!rendered.contains("Secondary Persona"));
    }

    #[test]
    fn test_persona_trigger_display() {
        assert_eq!(PersonaTrigger::TaskAssign.to_string(), "task_assign");
        assert_eq!(PersonaTrigger::FileMatch.to_string(), "file_match");
        assert_eq!(PersonaTrigger::StepHint.to_string(), "step_hint");
        assert_eq!(PersonaTrigger::Inherited.to_string(), "inherited");
        assert_eq!(
            PersonaTrigger::CommunityMatch.to_string(),
            "community_match"
        );
    }

    // ========================================================================
    // Complexity & TaskProfile tests (existing)
    // ========================================================================

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
        let task = make_task(vec!["backend"], vec!["src/a.rs", "src/b.rs", "src/c.rs"]);
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
