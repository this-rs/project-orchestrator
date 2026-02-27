//! Skill Activation for Hook Pipeline
//!
//! Implements the hot-path activation flow for Claude Code hooks:
//! 1. Extract pattern/file context from tool input
//! 2. Match against skill trigger patterns (Regex + FileGlob, no Semantic in hot path)
//! 3. Pick top skill(s)
//! 4. Load member notes and decisions
//! 5. Assemble context text within token budget
//! 6. Return HookActivateResponse
//!
//! Performance budget: < 150ms for the complete pipeline.
//!
//! # Design decisions
//!
//! - **No embeddings in hot path**: Regex/FileGlob matching costs <1ms vs 20-50ms for embedding.
//!   Semantic matching is reserved for the MCP `skill(action: "activate")` tool.
//! - **Local trigger evaluation**: Skills are loaded from DB then matched in Rust,
//!   avoiding per-trigger Neo4j round-trips.
//! - **Context budget**: 3200 chars max (~800 tokens) to prevent additionalContext flooding.

use regex::Regex;
use uuid::Uuid;

use std::sync::Arc;

use crate::neo4j::models::DecisionNode;
use crate::neo4j::traits::GraphStore;
use crate::neurons::AutoReinforcementConfig;
use crate::notes::models::Note;
use crate::skills::hook_extractor::{extract_file_context, extract_pattern};
use crate::skills::models::{HookActivateResponse, SkillNode, TriggerType};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the hook activation pipeline.
#[derive(Debug, Clone)]
pub struct HookActivationConfig {
    /// Minimum confidence score for a trigger match to be considered valid.
    /// Skills below this threshold are ignored.
    pub confidence_threshold: f64,
    /// Maximum characters for the assembled context text.
    /// Budget: 3200 chars (~800 tokens) for hooks.
    /// The additionalContext is injected at EVERY tool call, so keeping it small
    /// is critical (20 calls/session x 3200 chars = 64K chars worst case).
    pub max_context_chars: usize,
    /// Minimum note energy to include in context.
    /// Notes below this energy are considered "dead" and excluded.
    pub min_note_energy: f64,
    /// If the top-2 skills have confidence difference < this value,
    /// merge their contexts (skills are too close to distinguish).
    pub merge_threshold: f64,
}

impl Default for HookActivationConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            max_context_chars: 3200,
            min_note_energy: 0.1,
            merge_threshold: 0.1,
        }
    }
}

/// Result of the activation pipeline, containing both the response
/// to send back and the note IDs needed for async reinforcement.
///
/// This is NOT serialized directly — the handler extracts `response` for
/// the HTTP body and `activated_note_ids` for the reinforcement spawn.
pub struct HookActivationOutcome {
    /// The response to return to the hook caller.
    pub response: HookActivateResponse,
    /// IDs of notes included in the context — used for Hebbian reinforcement.
    pub activated_note_ids: Vec<Uuid>,
}

// ============================================================================
// Main activation function
// ============================================================================

/// Activate skills for a Claude Code hook call.
///
/// This is the main entry point for the hook activation pipeline.
/// It extracts patterns from the tool input, matches them against
/// skill triggers, and assembles context text for injection.
///
/// Returns `None` if no skill matches above the confidence threshold,
/// or if no pattern can be extracted from the tool input.
///
/// # Performance
///
/// Target: < 150ms total. The bottleneck is the DB call to load skills
/// and their members. Trigger matching is done locally in <1ms.
pub async fn activate_for_hook(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    tool_name: &str,
    tool_input: &serde_json::Value,
    config: &HookActivationConfig,
) -> anyhow::Result<Option<HookActivationOutcome>> {
    // 1. Extract pattern and file context from tool input
    let pattern = extract_pattern(tool_name, tool_input);
    let file_context = extract_file_context(tool_name, tool_input);

    // Nothing to match against → skip
    if pattern.is_none() && file_context.is_none() {
        return Ok(None);
    }

    // 2. Load matchable skills for this project
    let skills = graph_store.get_skills_for_project(project_id).await?;
    let matchable: Vec<_> = skills.into_iter().filter(|s| s.is_matchable()).collect();

    if matchable.is_empty() {
        return Ok(None);
    }

    // 3. Evaluate trigger patterns locally
    let mut matches: Vec<(SkillNode, f64)> = Vec::new();
    for skill in matchable {
        let confidence =
            evaluate_skill_match(&skill, pattern.as_deref(), file_context.as_deref());
        if confidence >= config.confidence_threshold {
            matches.push((skill, confidence));
        }
    }

    if matches.is_empty() {
        return Ok(None);
    }

    // Sort by confidence descending
    matches.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 4. Pick top skill (or merge top-2 if confidence is very close)
    let should_merge = matches.len() >= 2
        && (matches[0].1 - matches[1].1).abs() < config.merge_threshold;

    if should_merge {
        // Merge top-2 skills
        let (skill1, conf1) = matches.remove(0);
        let (skill2, _conf2) = matches.remove(0);

        let (notes1, decisions1) = graph_store.get_skill_members(skill1.id).await?;
        let (notes2, decisions2) = graph_store.get_skill_members(skill2.id).await?;

        // Merge notes, dedup by id
        let mut all_notes = notes1;
        for note in notes2 {
            if !all_notes.iter().any(|n| n.id == note.id) {
                all_notes.push(note);
            }
        }

        // Merge decisions, dedup by id
        let mut all_decisions = decisions1;
        for dec in decisions2 {
            if !all_decisions.iter().any(|d| d.id == dec.id) {
                all_decisions.push(dec);
            }
        }

        // Filter by energy
        let active_notes: Vec<_> = all_notes
            .into_iter()
            .filter(|n| n.energy >= config.min_note_energy)
            .collect();

        let merged_name = format!("{} + {}", skill1.name, skill2.name);
        let context = assemble_context(
            &merged_name,
            &active_notes,
            &all_decisions,
            config.max_context_chars,
        );

        let activated_note_ids: Vec<Uuid> = active_notes.iter().map(|n| n.id).collect();

        Ok(Some(HookActivationOutcome {
            response: HookActivateResponse {
                context,
                skill_name: merged_name,
                skill_id: skill1.id, // Use primary skill's ID
                confidence: conf1,
                notes_count: active_notes.len(),
                decisions_count: all_decisions.len(),
            },
            activated_note_ids,
        }))
    } else {
        // Single top skill
        let (skill, confidence) = matches.remove(0);
        let (notes, decisions) = graph_store.get_skill_members(skill.id).await?;

        let active_notes: Vec<_> = notes
            .into_iter()
            .filter(|n| n.energy >= config.min_note_energy)
            .collect();

        let context = assemble_context(
            &skill.name,
            &active_notes,
            &decisions,
            config.max_context_chars,
        );

        let activated_note_ids: Vec<Uuid> = active_notes.iter().map(|n| n.id).collect();

        Ok(Some(HookActivationOutcome {
            response: HookActivateResponse {
                context,
                skill_name: skill.name.clone(),
                skill_id: skill.id,
                confidence,
                notes_count: active_notes.len(),
                decisions_count: decisions.len(),
            },
            activated_note_ids,
        }))
    }
}

// ============================================================================
// Trigger matching (local, no DB calls)
// ============================================================================

/// Evaluate how well a skill's triggers match the given pattern and file context.
///
/// Returns the highest confidence score across all reliable triggers.
/// Semantic triggers are skipped in the hot path (per architectural decision).
pub fn evaluate_skill_match(
    skill: &SkillNode,
    pattern: Option<&str>,
    file_context: Option<&str>,
) -> f64 {
    let mut max_confidence = 0.0_f64;

    for trigger in skill.reliable_triggers() {
        let confidence = match trigger.pattern_type {
            TriggerType::Regex => {
                if let Some(pat) = pattern {
                    match_regex_trigger(&trigger.pattern_value, pat)
                } else {
                    0.0
                }
            }
            TriggerType::FileGlob => {
                // FileGlob can match against file_context (primary)
                // or pattern (fallback, e.g., Read tool returns file_path as pattern)
                let target = file_context.or(pattern);
                if let Some(file) = target {
                    match_file_glob_trigger(&trigger.pattern_value, file)
                } else {
                    0.0
                }
            }
            TriggerType::Semantic => {
                // Semantic matching skipped in hot path
                // (FastEmbed ~20-50ms = 10-25% of 200ms budget)
                0.0
            }
        };

        max_confidence = max_confidence.max(confidence);
    }

    max_confidence
}

/// Match a regex trigger pattern against an input string.
///
/// Returns a confidence score (0.0 or 1.0):
/// - 1.0 if the regex matches the input
/// - 0.0 if no match or regex compilation fails
fn match_regex_trigger(trigger_pattern: &str, input: &str) -> f64 {
    match Regex::new(trigger_pattern) {
        Ok(re) => {
            if re.is_match(input) {
                1.0
            } else {
                0.0
            }
        }
        Err(_) => 0.0,
    }
}

/// Match a file glob trigger pattern against a file path.
///
/// Returns a confidence score:
/// - 1.0 if the glob matches
/// - 0.0 if no match or invalid glob
fn match_file_glob_trigger(trigger_pattern: &str, file_path: &str) -> f64 {
    match glob::Pattern::new(trigger_pattern) {
        Ok(pat) => {
            if pat.matches(file_path) {
                1.0
            } else {
                0.0
            }
        }
        Err(_) => 0.0,
    }
}

// ============================================================================
// Context assembly
// ============================================================================

/// Maximum chars reserved for decisions section.
/// With max 2 decisions at ~100 chars each + header = ~250 chars.
const DECISIONS_BUDGET: usize = 300;

/// Assemble context text dynamically from notes and decisions.
///
/// Produces a compact Markdown context optimized for hook injection:
/// - Header: `## \u{1f4a1} {skill_name}` (~20 tokens)
/// - Notes: sorted by importance then energy, truncated to 150 chars each
/// - Decisions: max 2, description + chosen option at 100 chars each
/// - Budget: strict enforcement, never exceeds max_chars
///
/// The context_template from Plan 2 is NOT used here — this assembles
/// fresh content at every activation for maximum relevance.
pub fn assemble_context(
    skill_name: &str,
    notes: &[Note],
    decisions: &[DecisionNode],
    max_chars: usize,
) -> String {
    let header = format!("## \u{1f4a1} {}\n", skill_name);
    let mut context = header;

    // Reserve budget for decisions if any exist
    let notes_budget = if !decisions.is_empty() {
        max_chars.saturating_sub(DECISIONS_BUDGET)
    } else {
        max_chars
    };

    // Sort notes by importance (Critical first) then energy
    let mut sorted_notes: Vec<&Note> = notes.iter().collect();
    sorted_notes.sort_by(|a, b| {
        let imp_ord = b
            .importance
            .weight()
            .partial_cmp(&a.importance.weight())
            .unwrap_or(std::cmp::Ordering::Equal);
        if imp_ord != std::cmp::Ordering::Equal {
            return imp_ord;
        }
        b.energy
            .partial_cmp(&a.energy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Add notes one by one until budget is exhausted
    let mut notes_included = 0;
    for note in &sorted_notes {
        let emoji = note_type_emoji(&note.note_type.to_string());
        let importance_badge = importance_badge(&note.importance.to_string());
        let content = truncate_content(&note.content, 150);
        let line = format!("- {}{}{}\n", emoji, importance_badge, content);

        if context.len() + line.len() > notes_budget {
            break;
        }
        context.push_str(&line);
        notes_included += 1;
    }

    // Show how many notes were included vs total
    let omitted = sorted_notes.len().saturating_sub(notes_included);
    if omitted > 0 {
        let omit_line = format!("_(+{} more notes)_\n", omitted);
        if context.len() + omit_line.len() <= notes_budget {
            context.push_str(&omit_line);
        }
    }

    // Add decisions (max 2, within reserved budget)
    if !decisions.is_empty() {
        for decision in decisions.iter().take(2) {
            let chosen = decision
                .chosen_option
                .as_deref()
                .unwrap_or("(pending)");
            let line = format!(
                "- \u{1f3af} **{}**: {}\n",
                truncate_content(&decision.description, 60),
                truncate_content(chosen, 100),
            );

            if context.len() + line.len() > max_chars {
                break;
            }
            context.push_str(&line);
        }
    }

    // Final hard budget enforcement
    if context.len() > max_chars {
        context.truncate(max_chars.saturating_sub(3));
        context.push_str("...");
    }

    context
}

// ============================================================================
// Hebbian reinforcement (async, fire-and-forget)
// ============================================================================

/// Spawn async Hebbian reinforcement after a successful hook activation.
///
/// This is fire-and-forget: errors are logged but never propagated.
/// The function returns immediately (< 1ms) — all DB work runs in the
/// background via `tokio::spawn`.
///
/// Reinforcement actions:
/// 1. Boost energy of each activated note by `hook_energy_boost`
/// 2. Reinforce synapses between all co-activated notes by `hook_synapse_boost`
pub fn spawn_hook_reinforcement(
    graph_store: Arc<dyn GraphStore>,
    activated_note_ids: Vec<Uuid>,
    config: AutoReinforcementConfig,
) {
    if !config.enabled || activated_note_ids.is_empty() {
        return;
    }

    tokio::spawn(async move {
        if let Err(e) =
            reinforce_hook_activation(&*graph_store, &activated_note_ids, &config).await
        {
            tracing::warn!(
                notes_count = activated_note_ids.len(),
                "Hook Hebbian reinforcement failed: {}",
                e
            );
        }
    });
}

/// Perform Hebbian reinforcement for activated notes.
///
/// This is the actual DB work — called inside a `tokio::spawn` by
/// `spawn_hook_reinforcement`. NOT intended to be called directly
/// from request handlers.
async fn reinforce_hook_activation(
    graph_store: &dyn GraphStore,
    note_ids: &[Uuid],
    config: &AutoReinforcementConfig,
) -> anyhow::Result<()> {
    // 1. Boost energy for each activated note
    for &note_id in note_ids {
        graph_store
            .boost_energy(note_id, config.hook_energy_boost)
            .await?;
    }

    // 2. Reinforce synapses between co-activated notes
    if note_ids.len() >= 2 {
        let reinforced = graph_store
            .reinforce_synapses(note_ids, config.hook_synapse_boost)
            .await?;
        tracing::debug!(
            notes = note_ids.len(),
            synapses = reinforced,
            "Hook Hebbian reinforcement completed"
        );
    }

    Ok(())
}

/// Get emoji prefix for a note type.
fn note_type_emoji(note_type: &str) -> &'static str {
    match note_type {
        "gotcha" => "\u{26a0}\u{fe0f} ",      // ⚠️
        "guideline" => "\u{1f4cb} ",           // 📋
        "pattern" => "\u{1f504} ",             // 🔄
        "tip" => "\u{1f4a1} ",                 // 💡
        "context" => "\u{1f4dd} ",             // 📝
        "observation" => "\u{1f50d} ",         // 🔍
        "assertion" => "\u{2705} ",            // ✅
        _ => "",
    }
}

/// Get importance badge.
fn importance_badge(importance: &str) -> &'static str {
    match importance {
        "critical" => "\u{1f534} ", // 🔴
        "high" => "\u{1f7e0} ",     // 🟠
        _ => "",
    }
}

/// Truncate content to max_chars, taking only the first line or truncating.
fn truncate_content(content: &str, max_chars: usize) -> String {
    // Take first line
    let first_line = content.lines().next().unwrap_or(content);
    let clean = first_line.trim();

    if clean.len() <= max_chars {
        clean.to_string()
    } else {
        format!("{}...", &clean[..max_chars.saturating_sub(3)])
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notes::models::{NoteImportance, NoteScope, NoteStatus, NoteType};
    use crate::skills::models::{SkillNode, SkillTrigger};
    use chrono::Utc;

    fn make_test_note(id: Uuid, content: &str, note_type: NoteType, importance: NoteImportance, energy: f64) -> Note {
        Note {
            id,
            project_id: Some(Uuid::new_v4()),
            note_type,
            status: NoteStatus::Active,
            importance,
            scope: NoteScope::Project,
            content: content.to_string(),
            tags: vec![],
            anchors: vec![],
            created_at: Utc::now(),
            created_by: "test".to_string(),
            last_confirmed_at: None,
            last_confirmed_by: None,
            staleness_score: 0.0,
            energy,
            last_activated: None,
            supersedes: None,
            superseded_by: None,
            changes: vec![],
            assertion_rule: None,
            last_assertion_result: None,
        }
    }

    fn make_test_decision(id: Uuid, description: &str, chosen: &str) -> DecisionNode {
        DecisionNode {
            id,
            description: description.to_string(),
            rationale: "test rationale".to_string(),
            alternatives: vec![],
            chosen_option: Some(chosen.to_string()),
            decided_by: "test".to_string(),
            decided_at: Utc::now(),
            status: crate::neo4j::models::DecisionStatus::Accepted,
            embedding: None,
            embedding_model: None,
        }
    }

    // --- Trigger matching ---

    #[test]
    fn test_match_regex_trigger_matches() {
        assert!((match_regex_trigger("neo4j|cypher", "neo4j_client") - 1.0).abs() < f64::EPSILON);
        assert!((match_regex_trigger("neo4j|cypher", "cypher_query") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_match_regex_trigger_no_match() {
        assert!((match_regex_trigger("neo4j|cypher", "api_handler") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_match_regex_trigger_invalid_regex() {
        assert!((match_regex_trigger("[invalid", "test") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_match_file_glob_trigger_matches() {
        assert!((match_file_glob_trigger("src/neo4j/**", "src/neo4j/client.rs") - 1.0).abs() < f64::EPSILON);
        assert!((match_file_glob_trigger("src/neo4j/*", "src/neo4j/client.rs") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_match_file_glob_trigger_no_match() {
        assert!((match_file_glob_trigger("src/neo4j/**", "src/api/handlers.rs") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_match_file_glob_trigger_invalid_glob() {
        assert!((match_file_glob_trigger("[invalid", "test") - 0.0).abs() < f64::EPSILON);
    }

    // --- Skill match evaluation ---

    #[test]
    fn test_evaluate_skill_match_regex_hit() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Neo4j");
        skill.trigger_patterns = vec![SkillTrigger::regex("neo4j|cypher|UNWIND", 0.6)];

        let confidence = evaluate_skill_match(&skill, Some("neo4j_client"), None);
        assert!((confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_skill_match_regex_miss() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Neo4j");
        skill.trigger_patterns = vec![SkillTrigger::regex("neo4j|cypher", 0.6)];

        let confidence = evaluate_skill_match(&skill, Some("api_handler"), None);
        assert!((confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_skill_match_file_glob_hit() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Neo4j");
        skill.trigger_patterns = vec![SkillTrigger::file_glob("src/neo4j/**", 0.8)];

        let confidence = evaluate_skill_match(&skill, None, Some("src/neo4j/client.rs"));
        assert!((confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_skill_match_file_glob_with_pattern_fallback() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Neo4j");
        skill.trigger_patterns = vec![SkillTrigger::file_glob("src/neo4j/**", 0.8)];

        // file_context is None, but pattern is a file path (from Read tool)
        let confidence = evaluate_skill_match(&skill, Some("src/neo4j/client.rs"), None);
        assert!((confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_skill_match_multiple_triggers_best_wins() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Neo4j");
        skill.trigger_patterns = vec![
            SkillTrigger::regex("neo4j|cypher", 0.6),
            SkillTrigger::file_glob("src/api/**", 0.8), // won't match
        ];

        let confidence = evaluate_skill_match(&skill, Some("neo4j_client"), Some("src/skills/test.rs"));
        assert!((confidence - 1.0).abs() < f64::EPSILON); // regex matched
    }

    #[test]
    fn test_evaluate_skill_match_no_pattern_no_file() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Neo4j");
        skill.trigger_patterns = vec![SkillTrigger::regex("neo4j", 0.6)];

        let confidence = evaluate_skill_match(&skill, None, None);
        assert!((confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_skill_match_unreliable_trigger_skipped() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Neo4j");
        let mut trigger = SkillTrigger::regex(".*", 0.5); // matches everything
        trigger.quality_score = Some(0.1); // but unreliable
        skill.trigger_patterns = vec![trigger];

        let confidence = evaluate_skill_match(&skill, Some("anything"), None);
        assert!((confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_skill_match_semantic_skipped() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Neo4j");
        skill.trigger_patterns = vec![SkillTrigger::semantic("[0.1, 0.2]", 0.7)];

        let confidence = evaluate_skill_match(&skill, Some("neo4j"), None);
        assert!((confidence - 0.0).abs() < f64::EPSILON); // semantic skipped in hot path
    }

    #[test]
    fn test_evaluate_skill_match_dormant_not_matchable() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Neo4j");
        skill.status = crate::skills::models::SkillStatus::Dormant;
        skill.trigger_patterns = vec![SkillTrigger::regex("neo4j", 0.5)];

        // The function itself doesn't filter by status — that's done by the caller
        // using `is_matchable()`. But let's confirm the trigger still evaluates.
        let confidence = evaluate_skill_match(&skill, Some("neo4j"), None);
        assert!((confidence - 1.0).abs() < f64::EPSILON);
    }

    // --- Context assembly ---

    #[test]
    fn test_assemble_context_basic() {
        let notes = vec![
            make_test_note(Uuid::new_v4(), "Always use UNWIND for batch operations", NoteType::Guideline, NoteImportance::High, 0.8),
            make_test_note(Uuid::new_v4(), "Connection pool leak if not closed", NoteType::Gotcha, NoteImportance::Critical, 0.9),
        ];

        let decisions = vec![
            make_test_decision(Uuid::new_v4(), "Use Neo4j 5.x driver", "neo4j-rust-driver 0.8"),
        ];

        let context = assemble_context("Neo4j Performance", &notes, &decisions, 3200);

        assert!(context.starts_with("## \u{1f4a1} Neo4j Performance"));
        assert!(context.contains("UNWIND"));
        assert!(context.contains("Connection pool"));
        assert!(context.contains("Neo4j 5.x driver"));
        assert!(context.len() <= 3200);
    }

    #[test]
    fn test_assemble_context_sorted_by_importance() {
        let notes = vec![
            make_test_note(Uuid::new_v4(), "Low importance note", NoteType::Tip, NoteImportance::Low, 0.5),
            make_test_note(Uuid::new_v4(), "Critical gotcha note", NoteType::Gotcha, NoteImportance::Critical, 0.9),
            make_test_note(Uuid::new_v4(), "Medium pattern note", NoteType::Pattern, NoteImportance::Medium, 0.7),
        ];

        let context = assemble_context("Test", &notes, &[], 3200);

        // Critical should appear before Low
        let critical_pos = context.find("Critical gotcha").unwrap();
        let low_pos = context.find("Low importance").unwrap();
        assert!(critical_pos < low_pos);
    }

    #[test]
    fn test_assemble_context_budget_enforcement() {
        let mut notes = Vec::new();
        for i in 0..100 {
            notes.push(make_test_note(
                Uuid::new_v4(),
                &format!("This is a long note number {} with lots of content to fill the budget quickly and trigger truncation before all notes are included", i),
                NoteType::Context,
                NoteImportance::Medium,
                0.5,
            ));
        }

        let context = assemble_context("Huge Skill", &notes, &[], 2000);
        assert!(context.len() <= 2000);
    }

    #[test]
    fn test_assemble_context_empty_notes() {
        let context = assemble_context("Empty Skill", &[], &[], 3200);
        assert!(context.starts_with("## \u{1f4a1} Empty Skill"));
        assert!(!context.contains("Decisions"));
    }

    #[test]
    fn test_assemble_context_no_decisions() {
        let notes = vec![
            make_test_note(Uuid::new_v4(), "A note", NoteType::Tip, NoteImportance::Medium, 0.5),
        ];
        let context = assemble_context("Test", &notes, &[], 3200);
        assert!(!context.contains("Decisions"));
    }

    #[test]
    fn test_assemble_context_with_emojis() {
        let notes = vec![
            make_test_note(Uuid::new_v4(), "Watch out for this", NoteType::Gotcha, NoteImportance::Critical, 0.9),
            make_test_note(Uuid::new_v4(), "Follow this rule", NoteType::Guideline, NoteImportance::High, 0.8),
            make_test_note(Uuid::new_v4(), "Common pattern here", NoteType::Pattern, NoteImportance::Medium, 0.7),
            make_test_note(Uuid::new_v4(), "Helpful tip", NoteType::Tip, NoteImportance::Low, 0.5),
        ];

        let context = assemble_context("Test", &notes, &[], 3200);

        // Check emojis are present (using unicode escapes)
        assert!(context.contains("\u{26a0}\u{fe0f}")); // ⚠️ for gotcha
        assert!(context.contains("\u{1f4cb}")); // 📋 for guideline
        assert!(context.contains("\u{1f504}")); // 🔄 for pattern
        assert!(context.contains("\u{1f4a1}")); // 💡 for tip
    }

    // --- Truncation ---

    #[test]
    fn test_truncate_content_short() {
        assert_eq!(truncate_content("short text", 100), "short text");
    }

    #[test]
    fn test_truncate_content_long() {
        let long = "a".repeat(300);
        let result = truncate_content(&long, 100);
        assert!(result.len() <= 100);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_truncate_content_multiline() {
        let multiline = "First line\nSecond line\nThird line";
        assert_eq!(truncate_content(multiline, 100), "First line");
    }

    // --- Helper tests ---

    #[test]
    fn test_note_type_emoji() {
        assert_eq!(note_type_emoji("gotcha"), "\u{26a0}\u{fe0f} ");
        assert_eq!(note_type_emoji("guideline"), "\u{1f4cb} ");
        assert_eq!(note_type_emoji("pattern"), "\u{1f504} ");
        assert_eq!(note_type_emoji("tip"), "\u{1f4a1} ");
        assert_eq!(note_type_emoji("unknown"), "");
    }

    #[test]
    fn test_importance_badge() {
        assert_eq!(importance_badge("critical"), "\u{1f534} ");
        assert_eq!(importance_badge("high"), "\u{1f7e0} ");
        assert_eq!(importance_badge("medium"), "");
        assert_eq!(importance_badge("low"), "");
    }

    // --- Budget acceptance criteria ---

    #[test]
    fn test_assemble_context_20_notes_within_budget() {
        // Acceptance criteria: "skill avec 20 notes → contexte de 790 tokens (tronqué correctement)"
        // 800 tokens ≈ 3200 chars max
        let mut notes = Vec::new();
        let importances = [
            NoteImportance::Critical,
            NoteImportance::High,
            NoteImportance::Medium,
            NoteImportance::Low,
        ];
        let note_types = [
            NoteType::Gotcha,
            NoteType::Guideline,
            NoteType::Pattern,
            NoteType::Tip,
        ];

        for i in 0..20 {
            notes.push(make_test_note(
                Uuid::new_v4(),
                &format!(
                    "Note {} with substantial content that simulates a real knowledge note with useful information about the codebase patterns and conventions",
                    i
                ),
                note_types[i % 4].clone(),
                importances[i % 4].clone(),
                0.9 - (i as f64 * 0.03), // decreasing energy
            ));
        }

        let decisions = vec![
            make_test_decision(Uuid::new_v4(), "Use async Neo4j driver for all database operations", "neo4j-rust-driver 0.8 with async support"),
            make_test_decision(Uuid::new_v4(), "Implement caching with TTL-based invalidation", "30-second TTL per skill activation"),
            make_test_decision(Uuid::new_v4(), "This third decision should be excluded", "Only 2 decisions max"),
        ];

        let context = assemble_context("Neo4j Expertise", &notes, &decisions, 3200);

        // Strict budget enforcement
        assert!(
            context.len() <= 3200,
            "Context length {} exceeds 3200 char budget",
            context.len()
        );

        // Header present
        assert!(context.starts_with("## \u{1f4a1} Neo4j Expertise\n"));

        // Critical notes should appear first (they have highest importance weight)
        let critical_pos = context.find("Note 0 with substantial");
        let low_pos = context.find("Note 3 with substantial");
        if let (Some(c), Some(l)) = (critical_pos, low_pos) {
            assert!(c < l, "Critical notes should appear before Low notes");
        }

        // Not all 20 notes should fit — omitted indicator should be present
        assert!(
            context.contains("more notes)_"),
            "Should show omitted notes count when not all fit"
        );

        // Max 2 decisions
        assert!(
            !context.contains("This third decision"),
            "Only 2 decisions max should be included"
        );

        // At least some decisions are present
        assert!(
            context.contains("Neo4j driver") || context.contains("caching"),
            "At least one decision should be included"
        );
    }

    #[test]
    fn test_assemble_context_30_notes_budget_enforcement() {
        // Step 2 verification: "30 notes → contexte exactement ≤ 3200 chars"
        let mut notes = Vec::new();
        for i in 0..30 {
            notes.push(make_test_note(
                Uuid::new_v4(),
                &format!(
                    "Knowledge note number {} describing an important pattern or convention in the codebase that developers need to know about",
                    i
                ),
                NoteType::Context,
                NoteImportance::Medium,
                0.5,
            ));
        }

        let decisions = vec![
            make_test_decision(Uuid::new_v4(), "Architecture decision for the module", "chosen approach"),
        ];

        let context = assemble_context("Large Skill", &notes, &decisions, 3200);

        assert!(
            context.len() <= 3200,
            "Context length {} exceeds 3200 char budget with 30 notes",
            context.len()
        );

        // Should include at least 5 notes (each note line is ~180 chars, 5 = ~900)
        let note_count = context.matches("Knowledge note number").count();
        assert!(
            note_count >= 5,
            "Should include at least 5 notes, got {}",
            note_count
        );
        assert!(
            note_count < 30,
            "Should not include all 30 notes, got {}",
            note_count
        );

        // Omitted indicator present
        assert!(context.contains("more notes)_"));
    }

    #[test]
    fn test_assemble_context_decisions_budget_reserved() {
        // Ensure decisions budget (300 chars) is properly reserved
        let mut notes = Vec::new();
        for i in 0..50 {
            notes.push(make_test_note(
                Uuid::new_v4(),
                &format!("Note content {} filling the budget", i),
                NoteType::Tip,
                NoteImportance::Medium,
                0.5,
            ));
        }

        let decisions = vec![
            make_test_decision(Uuid::new_v4(), "Important decision", "chosen option value"),
        ];

        // With decisions present, notes should leave room
        let context_with_decisions = assemble_context("Test", &notes, &decisions, 3200);
        let context_without_decisions = assemble_context("Test", &notes, &[], 3200);

        // Context without decisions should have more notes since no budget reservation
        let notes_with = context_with_decisions
            .matches("Note content")
            .count();
        let notes_without = context_without_decisions
            .matches("Note content")
            .count();

        assert!(
            notes_without >= notes_with,
            "Without decisions reservation ({} notes), should include >= notes than with reservation ({} notes)",
            notes_without,
            notes_with,
        );

        // Both within budget
        assert!(context_with_decisions.len() <= 3200);
        assert!(context_without_decisions.len() <= 3200);
    }

    // --- Config defaults ---

    #[test]
    fn test_hook_activation_config_defaults() {
        let config = HookActivationConfig::default();
        assert!((config.confidence_threshold - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.max_context_chars, 3200);
        assert!((config.min_note_energy - 0.1).abs() < f64::EPSILON);
        assert!((config.merge_threshold - 0.1).abs() < f64::EPSILON);
    }
}
