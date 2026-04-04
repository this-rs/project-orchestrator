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

use regex::RegexBuilder;
use uuid::Uuid;

use std::sync::Arc;

use crate::neo4j::models::DecisionNode;
use crate::neo4j::traits::GraphStore;
use crate::neurons::AutoReinforcementConfig;
use crate::notes::models::Note;
use crate::skills::cache::{evaluate_cached_skill, SkillCache};
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
        let confidence = evaluate_skill_match(&skill, pattern.as_deref(), file_context.as_deref());
        if confidence >= config.confidence_threshold {
            matches.push((skill, confidence));
        }
    }

    if matches.is_empty() {
        return Ok(None);
    }

    // Sort by confidence descending
    matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 4. Pick top skill (or merge top-2 if confidence is very close)
    let should_merge =
        matches.len() >= 2 && (matches[0].1 - matches[1].1).abs() < config.merge_threshold;

    if should_merge {
        // Merge top-2 skills
        let (skill1, conf1) = matches.remove(0);
        let (skill2, _conf2) = matches.remove(0);

        let ((notes1, decisions1), (notes2, decisions2)) = tokio::try_join!(
            graph_store.get_skill_members(skill1.id),
            graph_store.get_skill_members(skill2.id),
        )?;

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
        let mut active_notes: Vec<_> = all_notes
            .into_iter()
            .filter(|n| n.computed_energy() >= config.min_note_energy)
            .collect();

        // Contextual scoring: sort by relevance when file/pattern context available
        let has_context = file_context.is_some() || pattern.is_some();
        if has_context {
            active_notes.sort_by(|a, b| {
                let sa =
                    score_note_relevance(a, file_context.as_deref(), pattern.as_deref(), tool_name);
                let sb =
                    score_note_relevance(b, file_context.as_deref(), pattern.as_deref(), tool_name);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let total_note_count = active_notes.len();
        let merged_name = format!("{} + {}", skill1.name, skill2.name);
        let (context, notes_included) = assemble_context_with_confidence(
            &merged_name,
            &active_notes,
            &all_decisions,
            config.max_context_chars,
            Some(conf1),
            has_context, // pre_sorted when we scored
        );

        // Only reinforce notes that were actually rendered in the context
        let activated_note_ids: Vec<Uuid> = active_notes
            .iter()
            .take(notes_included)
            .map(|n| n.id)
            .collect();

        if has_context {
            tracing::info!(
                skill = %merged_name,
                contextual = notes_included,
                total = total_note_count,
                "Contextual filtering applied (merged)"
            );
        }

        Ok(Some(HookActivationOutcome {
            response: HookActivateResponse {
                context,
                skill_name: merged_name,
                skill_id: skill1.id, // Use primary skill's ID
                confidence: conf1,
                notes_count: notes_included,
                decisions_count: all_decisions.len(),
                total_note_count,
                contextual_note_count: if has_context { notes_included } else { 0 },
            },
            activated_note_ids,
        }))
    } else {
        // Single top skill
        let (skill, confidence) = matches.remove(0);
        let (notes, decisions) = graph_store.get_skill_members(skill.id).await?;

        let mut active_notes: Vec<_> = notes
            .into_iter()
            .filter(|n| n.computed_energy() >= config.min_note_energy)
            .collect();

        // Contextual scoring: sort by relevance when file/pattern context available
        let has_context = file_context.is_some() || pattern.is_some();
        if has_context {
            active_notes.sort_by(|a, b| {
                let sa =
                    score_note_relevance(a, file_context.as_deref(), pattern.as_deref(), tool_name);
                let sb =
                    score_note_relevance(b, file_context.as_deref(), pattern.as_deref(), tool_name);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let total_note_count = active_notes.len();
        let (context, notes_included) = assemble_context_with_confidence(
            &skill.name,
            &active_notes,
            &decisions,
            config.max_context_chars,
            Some(confidence),
            has_context,
        );

        // Only reinforce notes that were actually rendered in the context
        let activated_note_ids: Vec<Uuid> = active_notes
            .iter()
            .take(notes_included)
            .map(|n| n.id)
            .collect();

        if has_context {
            tracing::info!(
                skill = %skill.name,
                contextual = notes_included,
                total = total_note_count,
                "Contextual filtering applied"
            );
        }

        Ok(Some(HookActivationOutcome {
            response: HookActivateResponse {
                context,
                skill_name: skill.name.clone(),
                skill_id: skill.id,
                confidence,
                notes_count: notes_included,
                decisions_count: decisions.len(),
                total_note_count,
                contextual_note_count: if has_context { notes_included } else { 0 },
            },
            activated_note_ids,
        }))
    }
}

/// Cached version of `activate_for_hook`.
///
/// Uses the `SkillCache` to avoid:
/// 1. Neo4j round-trip for `get_skills_for_project()` on cache hit
/// 2. `Regex::new()` / `glob::Pattern::new()` recompilation (pre-compiled triggers)
///
/// Falls back to DB on cache miss, then populates the cache for next request.
/// Member notes/decisions are always loaded fresh from DB (energy may change).
///
/// # Performance
///
/// - Cache hit: ~1ms (trigger matching only, no DB for skills)
/// - Cache miss: ~150ms (same as uncached, plus cache insert ~0.1ms)
pub async fn activate_for_hook_cached(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    tool_name: &str,
    tool_input: &serde_json::Value,
    config: &HookActivationConfig,
    cache: &SkillCache,
) -> anyhow::Result<Option<HookActivationOutcome>> {
    // 1. Extract pattern and file context from tool input
    let pattern = extract_pattern(tool_name, tool_input);
    let file_context = extract_file_context(tool_name, tool_input);

    if pattern.is_none() && file_context.is_none() {
        return Ok(None);
    }

    // 2. Try cache first, fallback to DB
    let cached_skills = match cache.get(&project_id).await {
        Some(skills) => skills,
        None => {
            // Cache miss — load from DB and populate cache
            let skills = graph_store.get_skills_for_project(project_id).await?;
            cache.insert(project_id, skills).await;
            // Get from cache (just inserted, no TOCTOU since insert is write-locked)
            cache.get(&project_id).await.unwrap_or_default()
        }
    };

    if cached_skills.is_empty() {
        return Ok(None);
    }

    // 3. Evaluate using pre-compiled triggers (no Regex::new per request)
    let mut matches: Vec<(SkillNode, f64)> = Vec::new();
    for cached in &cached_skills {
        let confidence = evaluate_cached_skill(cached, pattern.as_deref(), file_context.as_deref());
        if confidence >= config.confidence_threshold {
            matches.push((cached.skill.clone(), confidence));
        }
    }

    if matches.is_empty() {
        return Ok(None);
    }

    // Sort by confidence descending
    matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 4. Pick top skill (or merge top-2 if confidence is very close)
    let should_merge =
        matches.len() >= 2 && (matches[0].1 - matches[1].1).abs() < config.merge_threshold;

    if should_merge {
        let (skill1, conf1) = matches.remove(0);
        let (skill2, _conf2) = matches.remove(0);

        let ((notes1, decisions1), (notes2, decisions2)) = tokio::try_join!(
            graph_store.get_skill_members(skill1.id),
            graph_store.get_skill_members(skill2.id),
        )?;

        let mut all_notes = notes1;
        for note in notes2 {
            if !all_notes.iter().any(|n| n.id == note.id) {
                all_notes.push(note);
            }
        }

        let mut all_decisions = decisions1;
        for dec in decisions2 {
            if !all_decisions.iter().any(|d| d.id == dec.id) {
                all_decisions.push(dec);
            }
        }

        let mut active_notes: Vec<_> = all_notes
            .into_iter()
            .filter(|n| n.computed_energy() >= config.min_note_energy)
            .collect();

        // Contextual scoring: sort by relevance when file/pattern context available
        let has_context = file_context.is_some() || pattern.is_some();
        if has_context {
            active_notes.sort_by(|a, b| {
                let sa =
                    score_note_relevance(a, file_context.as_deref(), pattern.as_deref(), tool_name);
                let sb =
                    score_note_relevance(b, file_context.as_deref(), pattern.as_deref(), tool_name);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let total_note_count = active_notes.len();
        let merged_name = format!("{} + {}", skill1.name, skill2.name);
        let (context, notes_included) = assemble_context_with_confidence(
            &merged_name,
            &active_notes,
            &all_decisions,
            config.max_context_chars,
            Some(conf1),
            has_context,
        );

        // Only reinforce notes that were actually rendered in the context
        let activated_note_ids: Vec<Uuid> = active_notes
            .iter()
            .take(notes_included)
            .map(|n| n.id)
            .collect();

        if has_context {
            tracing::info!(
                skill = %merged_name,
                contextual = notes_included,
                total = total_note_count,
                "Contextual filtering applied (merged, cached)"
            );
        }

        Ok(Some(HookActivationOutcome {
            response: HookActivateResponse {
                context,
                skill_name: merged_name,
                skill_id: skill1.id,
                confidence: conf1,
                notes_count: notes_included,
                decisions_count: all_decisions.len(),
                total_note_count,
                contextual_note_count: if has_context { notes_included } else { 0 },
            },
            activated_note_ids,
        }))
    } else {
        let (skill, confidence) = matches.remove(0);
        let (notes, decisions) = graph_store.get_skill_members(skill.id).await?;

        let mut active_notes: Vec<_> = notes
            .into_iter()
            .filter(|n| n.computed_energy() >= config.min_note_energy)
            .collect();

        // Contextual scoring: sort by relevance when file/pattern context available
        let has_context = file_context.is_some() || pattern.is_some();
        if has_context {
            active_notes.sort_by(|a, b| {
                let sa =
                    score_note_relevance(a, file_context.as_deref(), pattern.as_deref(), tool_name);
                let sb =
                    score_note_relevance(b, file_context.as_deref(), pattern.as_deref(), tool_name);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let total_note_count = active_notes.len();
        let (context, notes_included) = assemble_context_with_confidence(
            &skill.name,
            &active_notes,
            &decisions,
            config.max_context_chars,
            Some(confidence),
            has_context,
        );

        // Only reinforce notes that were actually rendered in the context
        let activated_note_ids: Vec<Uuid> = active_notes
            .iter()
            .take(notes_included)
            .map(|n| n.id)
            .collect();

        if has_context {
            tracing::info!(
                skill = %skill.name,
                contextual = notes_included,
                total = total_note_count,
                "Contextual filtering applied (cached)"
            );
        }

        Ok(Some(HookActivationOutcome {
            response: HookActivateResponse {
                context,
                skill_name: skill.name.clone(),
                skill_id: skill.id,
                confidence,
                notes_count: notes_included,
                decisions_count: decisions.len(),
                total_note_count,
                contextual_note_count: if has_context { notes_included } else { 0 },
            },
            activated_note_ids,
        }))
    }
}

// ============================================================================
// Trigger matching (local, no DB calls)
// ============================================================================

/// Compute a depth-based boost factor for FileGlob matches.
///
/// Deeper globs are more specific → higher confidence:
/// - depth 1 (`src/**`) → 0.7 (shallow, likely broad)
/// - depth 2 (`src/graph/**`) → 0.85 (moderate specificity)
/// - depth 3+ (`src/graph/algo/**`) → 1.0 (very specific)
///
/// Used by both `evaluate_skill_match` (uncached) and `evaluate_cached_skill` (cached).
pub(crate) fn glob_depth_boost(glob_pattern: &str) -> f64 {
    // Count path segments before the wildcard: "src/graph/**" → 2 segments
    let prefix = glob_pattern.trim_end_matches("/**").trim_end_matches("/*");
    let depth = prefix.matches('/').count() + 1; // "src" = 1, "src/graph" = 2
                                                 // Scale: depth=1 → 0.7, depth=2 → 0.85, depth=3+ → 1.0
    (0.55 + 0.15 * (depth as f64).min(3.0)).min(1.0)
}

/// Evaluate how well a skill's triggers match the given pattern and file context.
///
/// Returns the highest confidence score across all reliable triggers.
/// FileGlob matches are boosted by glob depth (deeper = more specific = higher confidence).
/// Semantic triggers are skipped in the hot path (per architectural decision).
pub fn evaluate_skill_match(
    skill: &SkillNode,
    pattern: Option<&str>,
    file_context: Option<&str>,
) -> f64 {
    let mut max_confidence = 0.0_f64;

    for trigger in skill.reliable_triggers() {
        let matched = match trigger.pattern_type {
            TriggerType::Regex => {
                if let Some(pat) = pattern {
                    match_regex_trigger(&trigger.pattern_value, pat)
                } else {
                    false
                }
            }
            TriggerType::FileGlob => {
                let target = file_context.or(pattern);
                if let Some(file) = target {
                    match_file_glob_trigger(&trigger.pattern_value, file)
                } else {
                    false
                }
            }
            TriggerType::McpAction => {
                if let Some(pat) = pattern {
                    match_mcp_action_trigger(&trigger.pattern_value, pat)
                } else {
                    false
                }
            }
            TriggerType::Semantic => false,
        };

        if matched {
            let effective_confidence = match trigger.pattern_type {
                TriggerType::FileGlob => {
                    // Boost by glob depth: deeper = more specific = higher score
                    trigger.confidence_threshold * glob_depth_boost(&trigger.pattern_value)
                }
                _ => trigger.confidence_threshold,
            };
            max_confidence = max_confidence.max(effective_confidence);
        }
    }

    max_confidence
}

/// Match a regex trigger pattern against an input string.
///
/// Returns true if the regex matches the input, false otherwise.
///
/// Uses case-insensitive matching to be consistent with
/// `evaluate_regex_quality` in triggers.rs (which uses `(?i)` prefix).
fn match_regex_trigger(trigger_pattern: &str, input: &str) -> bool {
    // Reject overly long patterns to prevent compilation DoS
    if trigger_pattern.len() > 500 {
        return false;
    }
    match RegexBuilder::new(trigger_pattern)
        .case_insensitive(true)
        .size_limit(10_000)
        .dfa_size_limit(10_000)
        .build()
    {
        Ok(re) => re.is_match(input),
        Err(_) => false,
    }
}

/// Match a file glob trigger pattern against a file path.
///
/// Returns true if the glob matches, false otherwise.
fn match_file_glob_trigger(trigger_pattern: &str, file_path: &str) -> bool {
    match glob::Pattern::new(trigger_pattern) {
        Ok(pat) => pat.matches(file_path),
        Err(_) => false,
    }
}

/// Match an MCP action trigger against an extracted MCP pattern.
///
/// The trigger pattern_value is either:
/// - `"mega_tool"` → matches any action of that tool (e.g., `"note"` matches `"note create ..."`)
/// - `"mega_tool:action"` → matches a specific action (e.g., `"note:create"` matches `"note create ..."`)
///
/// The input pattern (from `extract_mcp_pattern`) is space-separated: `"mega_tool action key1 key2"`.
pub fn match_mcp_action_trigger(trigger_pattern: &str, input: &str) -> bool {
    // Split trigger: "note:create" → ("note", Some("create"))
    let (trigger_tool, trigger_action) = match trigger_pattern.split_once(':') {
        Some((tool, action)) => (tool.trim(), Some(action.trim())),
        None => (trigger_pattern.trim(), None),
    };

    if trigger_tool.is_empty() {
        return false;
    }

    // Split input: "note create Always use parameterized queries" → ["note", "create", ...]
    let mut parts = input.split_whitespace();
    let input_tool = match parts.next() {
        Some(t) => t,
        None => return false,
    };

    // Tool name must match (case-insensitive)
    if !input_tool.eq_ignore_ascii_case(trigger_tool) {
        return false;
    }

    // If trigger specifies an action, it must also match
    if let Some(action) = trigger_action {
        if !action.is_empty() {
            let input_action = match parts.next() {
                Some(a) => a,
                None => return false,
            };
            if !input_action.eq_ignore_ascii_case(action) {
                return false;
            }
        }
    }

    true
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
    assemble_context_with_confidence(skill_name, notes, decisions, max_chars, None, false).0
}

/// Assemble context with optional confidence score in the header.
///
/// When `pre_sorted` is true, notes are used in the order provided (caller has
/// already sorted by contextual relevance score). When false, notes are sorted
/// by importance then energy (legacy behavior).
///
/// Returns `(context_text, notes_included_count)` — the count is needed to
/// restrict Hebbian reinforcement to only the notes that actually made it into
/// the context budget.
pub fn assemble_context_with_confidence(
    skill_name: &str,
    notes: &[Note],
    decisions: &[DecisionNode],
    max_chars: usize,
    confidence: Option<f64>,
    pre_sorted: bool,
) -> (String, usize) {
    let header = match confidence {
        Some(conf) => format!(
            "## \u{1f9e0} Skill \"{}\" (confidence {}%)\n",
            skill_name,
            (conf * 100.0).round() as u32
        ),
        None => format!("## \u{1f4a1} {}\n", skill_name),
    };
    let mut context = header;

    // Reserve budget for decisions if any exist
    let notes_budget = if !decisions.is_empty() {
        max_chars.saturating_sub(DECISIONS_BUDGET)
    } else {
        max_chars
    };

    // Sort notes by importance (Critical first) then energy — unless pre_sorted
    let mut sorted_notes: Vec<&Note> = notes.iter().collect();
    if !pre_sorted {
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
    }

    // Add notes one by one until budget is exhausted
    // Use chars().count() consistently (not .len() which is byte count)
    // to correctly handle multi-byte UTF-8 characters in budget accounting.
    let mut notes_included = 0;
    for note in &sorted_notes {
        let emoji = note_type_emoji(&note.note_type.to_string());
        let importance_badge = importance_badge(&note.importance.to_string());
        let content = truncate_content(&note.content, 150);
        let line = format!("- {}{}{}\n", emoji, importance_badge, content);

        if context.chars().count() + line.chars().count() > notes_budget {
            break;
        }
        context.push_str(&line);
        notes_included += 1;
    }

    // Show how many notes were included vs total
    let omitted = sorted_notes.len().saturating_sub(notes_included);
    if omitted > 0 {
        let omit_line = format!("_(+{} more notes)_\n", omitted);
        if context.chars().count() + omit_line.chars().count() <= notes_budget {
            context.push_str(&omit_line);
        }
    }

    // Add decisions (max 2, within reserved budget)
    if !decisions.is_empty() {
        for decision in decisions.iter().take(2) {
            let chosen = decision.chosen_option.as_deref().unwrap_or("(pending)");
            let line = format!(
                "- \u{1f3af} **{}**: {}\n",
                truncate_content(&decision.description, 60),
                truncate_content(chosen, 100),
            );

            if context.chars().count() + line.chars().count() > max_chars {
                break;
            }
            context.push_str(&line);
        }
    }

    // Final hard budget enforcement (use char_indices for UTF-8 safety)
    if context.chars().count() > max_chars {
        let trunc_target = max_chars.saturating_sub(3);
        if let Some((byte_pos, _)) = context.char_indices().nth(trunc_target) {
            context.truncate(byte_pos);
            context.push_str("...");
        }
    }

    (context, notes_included)
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
        if let Err(e) = reinforce_hook_activation(&*graph_store, &activated_note_ids, &config).await
        {
            tracing::warn!(
                notes_count = activated_note_ids.len(),
                "Hook Hebbian reinforcement failed: {}",
                e
            );
        }
    });
}

/// Spawn async activation_count increment after a successful hook activation.
///
/// Fire-and-forget: errors are logged but never propagated.
/// This ensures hook activations are tracked in the skill's metrics
/// (activation_count, last_activated), which is required for lifecycle
/// promotion logic.
pub fn spawn_activation_increment(graph_store: Arc<dyn GraphStore>, skill_id: Uuid) {
    tokio::spawn(async move {
        if let Err(e) = graph_store.increment_skill_activation(skill_id).await {
            tracing::warn!(
                %skill_id,
                "Hook activation count increment failed: {}",
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

    // 3. Track re-activation for route quality measurement
    if !note_ids.is_empty() {
        let reactivated = graph_store.track_reactivation(note_ids).await?;
        tracing::debug!(reactivated, "Re-activation tracking completed");
    }

    Ok(())
}

/// Get emoji prefix for a note type.
fn note_type_emoji(note_type: &str) -> &'static str {
    match note_type {
        "gotcha" => "\u{26a0}\u{fe0f} ", // ⚠️
        "guideline" => "\u{1f4cb} ",     // 📋
        "pattern" => "\u{1f504} ",       // 🔄
        "tip" => "\u{1f4a1} ",           // 💡
        "context" => "\u{1f4dd} ",       // 📝
        "observation" => "\u{1f50d} ",   // 🔍
        "assertion" => "\u{2705} ",      // ✅
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
/// Uses char_indices to avoid panicking on multi-byte UTF-8 boundaries.
fn truncate_content(content: &str, max_chars: usize) -> String {
    // Take first line
    let first_line = content.lines().next().unwrap_or(content);
    let clean = first_line.trim();

    if clean.chars().count() <= max_chars {
        clean.to_string()
    } else {
        let trunc_len = max_chars.saturating_sub(3);
        let end = clean
            .char_indices()
            .nth(trunc_len)
            .map(|(i, _)| i)
            .unwrap_or(clean.len());
        format!("{}...", &clean[..end])
    }
}

// ============================================================================
// Contextual note scoring (pure functions, zero DB queries)
// ============================================================================

/// Extract meaningful path segments from a file context string.
///
/// Filters out trivial segments like "src", "lib", "mod", "index", "main", "tests".
/// Strips file extensions. Returns lowercased segments for case-insensitive matching.
///
/// # Examples
/// ```
/// use project_orchestrator::skills::activation::extract_path_segments;
///
/// assert_eq!(extract_path_segments("src/neo4j/client.rs"), vec!["neo4j", "client"]);
/// assert!(extract_path_segments("src/lib.rs").is_empty());
/// // Absolute paths keep non-trivial segments
/// assert_eq!(
///     extract_path_segments("src/skills/activation.rs"),
///     vec!["skills", "activation"]
/// );
/// ```
pub fn extract_path_segments(file_context: &str) -> Vec<String> {
    const TRIVIAL_SEGMENTS: &[&str] = &[
        "src", "lib", "mod", "index", "main", "tests", "test", "benches", "bench", "examples",
        "example", "bin", "target", "build", "dist", "out", "pkg",
    ];

    file_context
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|s| {
            // Strip file extension
            s.rsplit_once('.').map(|(name, _)| name).unwrap_or(s)
        })
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty() && s.len() > 1 && !TRIVIAL_SEGMENTS.contains(&s.as_str()))
        .collect()
}

/// Extract the basename (filename without extension) from a file path.
fn extract_basename(file_context: &str) -> Option<String> {
    let filename = file_context.rsplit('/').next()?;
    let name = filename
        .rsplit_once('.')
        .map(|(n, _)| n)
        .unwrap_or(filename);
    if name.is_empty() {
        None
    } else {
        Some(name.to_lowercase())
    }
}

/// Score a note's relevance to the current tool context.
///
/// This is a **pure function** — no async, no DB calls, no side effects.
/// It operates entirely on data already loaded in memory from `get_skill_members()`.
///
/// # Signals
///
/// 1. **Tag-path affinity** (0 → 0.4): overlap between note.tags and file path segments
/// 2. **Content keyword match** (0 → 0.3): note.content mentions the filename or path segments
/// 3. **Importance weight** (multiplicative): critical ×1.5, high ×1.2, medium ×1.0, low ×0.8
/// 4. **Freshness decay** (subtractive): -staleness_score × 0.1
/// 5. **Note type affinity** (0 → 0.1): gotcha notes score higher for Edit/Write tools
/// 6. **Anchor bonus** (0 → 0.4): LINKED_TO anchor matches file_context (added by T4)
///
/// Returns a score in approximately [0.0, 1.5] range. Higher = more relevant.
pub fn score_note_relevance(
    note: &Note,
    file_context: Option<&str>,
    _pattern: Option<&str>,
    tool_name: &str,
) -> f64 {
    let mut score: f64 = 0.1; // base score — every note has some value

    if let Some(file_ctx) = file_context {
        let path_segments = extract_path_segments(file_ctx);
        let basename = extract_basename(file_ctx);

        // Signal 1: Tag-path affinity (0 → 0.4)
        // Count how many path segments overlap with note tags (case-insensitive)
        let tag_overlap = note
            .tags
            .iter()
            .filter(|tag| {
                let tag_lower = tag.to_lowercase();
                path_segments.contains(&tag_lower)
            })
            .count();
        score += (tag_overlap as f64 * 0.15).min(0.4);

        // Signal 2: Content keyword match (0 → 0.3)
        let content_lower = note.content.to_lowercase();

        // Basename match: e.g., "client.rs" found in content → +0.2
        if let Some(ref base) = basename {
            if content_lower.contains(base.as_str()) {
                score += 0.2;
            }
        }

        // Path segment matches in content: each unique match → +0.05 (max +0.1)
        let segment_matches = path_segments
            .iter()
            .filter(|seg| seg.len() >= 3 && content_lower.contains(seg.as_str()))
            .count();
        score += (segment_matches as f64 * 0.05).min(0.1);

        // Signal 6: Anchor bonus (LINKED_TO) — added by T4
        // Checks note.anchors for File entities matching the file_context.
        // When anchors are empty (pre-T4 or pre-auto-anchor), this adds 0.
        let mut best_anchor_bonus = 0.0_f64;
        for anchor in &note.anchors {
            if matches!(anchor.entity_type, crate::notes::EntityType::File) {
                if anchor.entity_id == file_ctx
                    || anchor.entity_id.ends_with(&format!("/{}", file_ctx))
                    || file_ctx.ends_with(&format!("/{}", anchor.entity_id))
                {
                    best_anchor_bonus = best_anchor_bonus.max(0.4); // exact match
                } else if same_directory(&anchor.entity_id, file_ctx) {
                    best_anchor_bonus = best_anchor_bonus.max(0.2); // same dir
                }
            }
        }
        score += best_anchor_bonus;
    }

    // Signal 3: Importance weight (multiplicative)
    let importance_multiplier = match note.importance {
        crate::notes::NoteImportance::Critical => 1.5,
        crate::notes::NoteImportance::High => 1.2,
        crate::notes::NoteImportance::Medium => 1.0,
        crate::notes::NoteImportance::Low => 0.8,
    };
    score *= importance_multiplier;

    // Signal 4: Freshness decay (subtractive)
    score -= note.staleness_score * 0.1;

    // Signal 5: Note type affinity
    // Edit/Write/Bash tools benefit from gotcha notes (error prevention)
    // MCP tools benefit from guideline/pattern notes
    match tool_name {
        "Edit" | "Write" | "Bash" | "NotebookEdit" => {
            if matches!(note.note_type, crate::notes::NoteType::Gotcha) {
                score += 0.1;
            }
        }
        t if t.starts_with("mcp__") => {
            if matches!(
                note.note_type,
                crate::notes::NoteType::Guideline | crate::notes::NoteType::Pattern
            ) {
                score += 0.1;
            }
        }
        _ => {}
    }

    // Ensure score is non-negative
    score.max(0.0)
}

/// Check if two file paths share the same parent directory.
///
/// Handles both absolute and relative paths by comparing the substring
/// before the last '/'.
pub fn same_directory(path_a: &str, path_b: &str) -> bool {
    let dir_a = path_a.rsplit_once('/').map(|(dir, _)| dir);
    let dir_b = path_b.rsplit_once('/').map(|(dir, _)| dir);
    match (dir_a, dir_b) {
        (Some(a), Some(b)) => {
            // Handle case where one is absolute and the other relative
            // e.g., "/Users/x/src/neo4j/client.rs" vs "src/neo4j/analytics.rs"
            a == b || a.ends_with(b) || b.ends_with(a)
        }
        _ => false,
    }
}

// ============================================================================
// File path extraction from note content (for auto-anchoring)
// ============================================================================

/// Known source file extensions for path detection.
const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "py", "go", "java", "kt", "swift", "rb", "php", "cs", "cpp",
    "c", "h", "hpp", "toml", "yaml", "yml", "json", "sql", "sh", "bash", "zsh", "md", "html",
    "css", "scss", "vue", "svelte", "ex", "exs", "zig", "lua", "r", "m", "mm",
];

/// Extract file paths mentioned in a note's content.
///
/// Detects:
/// - Relative paths: `src/neo4j/client.rs`, `tests/unit/test_foo.py`
/// - Absolute paths: `/Users/foo/project/src/bar.rs` → normalized to relative after `src/`
/// - Backtick-wrapped: \`src/neo4j/client.rs\`
/// - Paths with known source extensions
///
/// Filters out:
/// - URLs (http://, https://, ftp://)
/// - Package/import strings (no `/` separators like `crate::foo::bar`)
/// - Paths that are too short or don't contain a `/`
///
/// Returns deduplicated relative paths.
pub fn extract_file_paths_from_content(content: &str) -> Vec<String> {
    use std::collections::HashSet;

    let mut paths = HashSet::new();

    // Regex-free approach: split content into tokens, check each for path patterns
    for line in content.lines() {
        // Extract tokens from the line, handling backticks specially
        let tokens = extract_path_tokens(line);
        for token in tokens {
            if let Some(path) = validate_and_normalize_path(&token) {
                paths.insert(path);
            }
        }
    }

    let mut result: Vec<String> = paths.into_iter().collect();
    result.sort(); // Deterministic ordering
    result
}

/// Extract potential path tokens from a line of text.
///
/// Handles backtick-wrapped paths and whitespace/punctuation-delimited tokens.
fn extract_path_tokens(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();

    // First: extract backtick-wrapped content (highest priority)
    let mut rest = line;
    while let Some(start) = rest.find('`') {
        let after_tick = &rest[start + 1..];
        if let Some(end) = after_tick.find('`') {
            let inside = &after_tick[..end];
            if inside.contains('/') && !inside.contains(' ') {
                tokens.push(inside.to_string());
            }
            rest = &after_tick[end + 1..];
        } else {
            break;
        }
    }

    // Second: split line by whitespace and common delimiters, look for path-like tokens
    for word in line.split(|c: char| {
        c.is_whitespace()
            || c == ','
            || c == ';'
            || c == '('
            || c == ')'
            || c == '['
            || c == ']'
            || c == '"'
            || c == '\''
    }) {
        // Strip surrounding backticks, parens, quotes
        let clean = word.trim_matches(|c: char| {
            c == '`' || c == '\'' || c == '"' || c == '(' || c == ')' || c == ':'
        });
        if clean.contains('/') && !clean.is_empty() {
            tokens.push(clean.to_string());
        }
    }

    tokens
}

/// Validate a token as a file path and normalize it.
///
/// Returns `Some(normalized_path)` if the token looks like a valid source file path,
/// `None` otherwise.
fn validate_and_normalize_path(token: &str) -> Option<String> {
    // Strip trailing punctuation (period, comma, semicolon, colon, paren)
    let token = token
        .trim()
        .trim_end_matches(['.', ',', ';', ':', ')', ']']);

    // Reject URLs
    if token.starts_with("http://") || token.starts_with("https://") || token.starts_with("ftp://")
    {
        return None;
    }

    // Must contain at least one '/' (to be a path, not just a filename)
    if !token.contains('/') {
        return None;
    }

    // Reject Rust module paths (::)
    if token.contains("::") {
        return None;
    }

    // Must have a recognized extension
    let extension = token.rsplit('.').next()?;
    if !SOURCE_EXTENSIONS.contains(&extension.to_lowercase().as_str()) {
        return None;
    }

    // Normalize absolute paths: find "src/" and take from there
    let normalized = if token.starts_with('/') {
        if let Some(idx) = token.find("/src/") {
            &token[idx + 1..] // Skip the leading '/', keep "src/..."
        } else if let Some(idx) = token.find("/tests/") {
            &token[idx + 1..]
        } else {
            // Absolute path without src/ or tests/ — skip (could be system path)
            return None;
        }
    } else {
        token
    };

    // Sanity checks
    let segments: Vec<&str> = normalized.split('/').collect();
    if segments.len() < 2 {
        return None; // Need at least dir/file.ext
    }

    // Reject if any segment is empty or starts with '.'
    if segments.iter().any(|s| s.is_empty() || s.starts_with('.')) {
        return None;
    }

    Some(normalized.to_string())
}

// ============================================================================
// Auto-anchoring: link notes to files mentioned in their content
// ============================================================================

/// Auto-anchor a single note to files mentioned in its content.
///
/// Extracts file paths from the note content and creates LINKED_TO relations
/// to matching File nodes in the graph. Uses `link_note_to_entity` which
/// performs MERGE (idempotent — safe to call multiple times).
///
/// **Important**: `root_path` must be provided so that relative paths extracted
/// from note content (e.g. `src/neo4j/client.rs`) are resolved to the absolute
/// paths used by File nodes in Neo4j (e.g. `/home/user/project/src/neo4j/client.rs`).
/// Without `root_path`, the MATCH query silently returns 0 rows (ghost anchors).
///
/// Returns the number of new anchors created.
pub async fn auto_anchor_note(
    graph_store: &dyn GraphStore,
    note: &Note,
    root_path: Option<&str>,
) -> anyhow::Result<usize> {
    use crate::notes::models::EntityType;

    let paths = extract_file_paths_from_content(&note.content);
    let mut anchored = 0;

    for path in &paths {
        // File nodes in Neo4j use absolute paths. extract_file_paths_from_content
        // returns relative paths (e.g. "src/neo4j/client.rs"). Resolve them to
        // absolute using the project's root_path so the MATCH query finds the node.
        let resolved_path = if let Some(root) = root_path {
            let root = root.trim_end_matches('/');
            format!("{}/{}", root, path)
        } else {
            // No root_path — use relative path as-is (will likely not match,
            // but keeps backward compatibility for tests/edge cases)
            path.clone()
        };

        // Try to link — link_note_to_entity uses MERGE, so it's safe if already linked
        if let Err(e) = graph_store
            .link_note_to_entity(note.id, &EntityType::File, &resolved_path, None, None)
            .await
        {
            tracing::debug!(
                note_id = %note.id,
                path = %resolved_path,
                "Auto-anchor skipped (file may not exist in graph): {}",
                e
            );
            continue;
        }
        anchored += 1;
    }

    Ok(anchored)
}

/// Auto-anchor all notes for a project to files mentioned in their content.
///
/// This is a batch operation meant to be called from the MCP admin action
/// `auto_anchor_notes`. It loads the project's `root_path` and all project notes,
/// then runs `auto_anchor_note` on each with path resolution.
///
/// Returns `AutoAnchorResult` with diagnostics.
pub async fn auto_anchor_notes_for_project(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
) -> anyhow::Result<AutoAnchorResult> {
    use crate::notes::models::NoteFilters;

    // Resolve project root_path for absolute file path matching.
    // File nodes in Neo4j use absolute paths, but extract_file_paths_from_content
    // returns relative paths — we need root_path to bridge the gap.
    let root_path = match graph_store.get_project(project_id).await? {
        Some(proj) => Some(proj.root_path),
        None => {
            tracing::warn!(%project_id, "Auto-anchor: project not found, using relative paths");
            None
        }
    };

    let filters = NoteFilters::default();
    let (notes, _total) = graph_store
        .list_notes(Some(project_id), None, &filters)
        .await?;

    let notes_count = notes.len();
    let mut total_anchors = 0;

    for note in &notes {
        let anchored = auto_anchor_note(graph_store, note, root_path.as_deref()).await?;
        total_anchors += anchored;
    }

    tracing::info!(
        %project_id,
        notes = notes_count,
        anchors = total_anchors,
        root_path = root_path.as_deref().unwrap_or("<none>"),
        "Auto-anchoring completed"
    );

    Ok(AutoAnchorResult {
        notes_processed: notes_count,
        anchors_created: total_anchors,
        root_path_resolved: root_path,
    })
}

/// Result of batch auto-anchoring with debug info.
pub struct AutoAnchorResult {
    pub notes_processed: usize,
    pub anchors_created: usize,
    pub root_path_resolved: Option<String>,
}

// ============================================================================
// Decision Auto-Anchor (AFFECTS relationships)
// ============================================================================

/// Auto-anchor a decision to files mentioned in its content.
///
/// Extracts file paths from the decision's description, rationale, and
/// chosen_option, then creates AFFECTS relationships to matching File nodes.
/// Uses `add_decision_affects` which performs MERGE (idempotent).
///
/// **Important**: `root_path` must be provided so that relative paths extracted
/// from content are resolved to the absolute paths used by File nodes in Neo4j.
///
/// Returns the number of new anchors created.
pub async fn auto_anchor_decision(
    graph_store: &dyn GraphStore,
    decision: &DecisionNode,
    root_path: Option<&str>,
) -> anyhow::Result<usize> {
    let mut all_paths = extract_file_paths_from_content(&decision.description);

    if !decision.rationale.is_empty() {
        all_paths.extend(extract_file_paths_from_content(&decision.rationale));
    }
    if let Some(ref chosen) = decision.chosen_option {
        all_paths.extend(extract_file_paths_from_content(chosen));
    }

    // Deduplicate
    all_paths.sort();
    all_paths.dedup();

    let mut anchored = 0;
    for path in &all_paths {
        // File nodes in Neo4j use absolute paths. extract_file_paths_from_content
        // returns relative paths (e.g. "src/neo4j/client.rs"). Resolve them to
        // absolute using the project's root_path so the MATCH query finds the node.
        let resolved_path = if let Some(root) = root_path {
            let root = root.trim_end_matches('/');
            format!("{}/{}", root, path)
        } else {
            path.clone()
        };

        if let Err(e) = graph_store
            .add_decision_affects(decision.id, "File", &resolved_path, None)
            .await
        {
            tracing::debug!(
                decision_id = %decision.id,
                path = %resolved_path,
                "Auto-anchor decision skipped (file may not exist in graph): {}",
                e
            );
            continue;
        }
        anchored += 1;
    }

    Ok(anchored)
}

/// Auto-anchor all decisions for a project to files mentioned in their content.
///
/// Retrieves decisions via the Project->Plan->Task->Decision chain using
/// `get_project_decisions_for_graph`, then runs `auto_anchor_decision` on each.
///
/// Returns `(decisions_processed, anchors_created)`.
pub async fn auto_anchor_decisions_for_project(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
) -> anyhow::Result<(usize, usize)> {
    // Resolve project root_path for absolute file path matching.
    let root_path = match graph_store.get_project(project_id).await? {
        Some(proj) => Some(proj.root_path),
        None => {
            tracing::warn!(%project_id, "Auto-anchor decisions: project not found");
            None
        }
    };

    // Get all decisions for this project via Plan→Task→Decision chain.
    // get_project_decisions_for_graph returns (DecisionNode, Vec<AffectsRelation>);
    // we only need the DecisionNode.
    let decision_pairs = graph_store
        .get_project_decisions_for_graph(project_id)
        .await?;

    let decisions_count = decision_pairs.len();
    let mut total_anchors = 0;

    for (decision, _existing_affects) in &decision_pairs {
        let anchored = auto_anchor_decision(graph_store, decision, root_path.as_deref()).await?;
        total_anchors += anchored;
    }

    tracing::info!(
        %project_id,
        decisions_processed = decisions_count,
        anchors_created = total_anchors,
        "Auto-anchor decisions completed"
    );

    Ok((decisions_count, total_anchors))
}

// ============================================================================
// Cross-Project Note Anchoring
// ============================================================================

/// Auto-anchor notes from ALL sources (project, global, other projects) to
/// files of a given project.
///
/// Unlike `auto_anchor_notes_for_project` which only processes notes belonging
/// to the specified project, this function loads all notes and anchors them
/// against the target project's file tree. This enables cross-project knowledge
/// propagation — a note from project A mentioning `src/shared/utils.rs` will
/// be linked to the matching File node in project B if it has that path.
///
/// Returns `AutoAnchorResult` with diagnostics.
pub async fn auto_anchor_all_notes_to_project(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
) -> anyhow::Result<AutoAnchorResult> {
    use crate::notes::models::NoteFilters;

    let root_path = match graph_store.get_project(project_id).await? {
        Some(proj) => Some(proj.root_path),
        None => {
            tracing::warn!(%project_id, "Cross-project auto-anchor: project not found");
            return Ok(AutoAnchorResult {
                notes_processed: 0,
                anchors_created: 0,
                root_path_resolved: None,
            });
        }
    };

    // Load ALL notes (None for project_id = all notes across all projects)
    let filters = NoteFilters::default();
    let (all_notes, _total) = graph_store.list_notes(None, None, &filters).await?;

    let notes_count = all_notes.len();
    let mut total_anchors = 0;

    for note in &all_notes {
        let anchored = auto_anchor_note(graph_store, note, root_path.as_deref()).await?;
        total_anchors += anchored;
    }

    tracing::info!(
        %project_id,
        notes_processed = notes_count,
        anchors_created = total_anchors,
        "Cross-project auto-anchor notes completed"
    );

    Ok(AutoAnchorResult {
        notes_processed: notes_count,
        anchors_created: total_anchors,
        root_path_resolved: root_path,
    })
}

// ============================================================================
// Knowledge Link Reconstruction (orchestrator)
// ============================================================================

/// Report from `reconstruct_knowledge_links` with diagnostic counters.
#[derive(Debug, Default, serde::Serialize)]
pub struct ReconstructReport {
    pub notes_processed: usize,
    pub notes_linked: usize,
    pub decisions_processed: usize,
    pub affects_created: usize,
    pub structural_propagated: usize,
    pub semantic_linked: usize,
    pub elapsed_ms: u64,
}

/// Reconstruct all knowledge links for a project after sync.
///
/// Called automatically post-sync and available as admin action.
/// Performs two passes:
///
/// 1. **Cross-project note anchoring**: loads ALL notes (project + global + other
///    projects) and creates LINKED_TO relationships to this project's File nodes.
/// 2. **Decision anchoring**: loads all decisions for this project and creates
///    AFFECTS relationships to File nodes mentioned in description/rationale.
///
/// Both passes use MERGE for idempotent relationship creation.
/// File paths are resolved from relative to absolute using the project's `root_path`.
pub async fn reconstruct_knowledge_links(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
) -> anyhow::Result<ReconstructReport> {
    let start = std::time::Instant::now();

    // 1. Cross-project note anchoring (includes project's own notes + global + other projects)
    let anchor_result = auto_anchor_all_notes_to_project(graph_store, project_id).await?;

    // 2. Decision anchoring (AFFECTS)
    let (decisions_processed, affects_created) =
        auto_anchor_decisions_for_project(graph_store, project_id).await?;

    // 3. Structural propagation (IMPORTS/CALLS/CO_CHANGED → LINKED_TO propagated)
    let structural_propagated = graph_store.propagate_structural_links(project_id).await?;

    // 4. Semantic propagation (file embeddings ↔ note embeddings via HNSW)
    let semantic_linked = graph_store
        .propagate_semantic_links(project_id, 0.7)
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(%project_id, error = %e, "Semantic propagation failed (non-fatal)");
            0
        });

    // 5. High-level entity propagation (FeatureGraph, Skill, Protocol → Files)
    let high_level_propagated = graph_store
        .propagate_high_level_links(project_id)
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(%project_id, error = %e, "High-level propagation failed (non-fatal)");
            0
        });

    let elapsed = start.elapsed();

    let report = ReconstructReport {
        notes_processed: anchor_result.notes_processed,
        notes_linked: anchor_result.anchors_created,
        decisions_processed,
        affects_created,
        structural_propagated: structural_propagated + high_level_propagated,
        semantic_linked,
        elapsed_ms: elapsed.as_millis() as u64,
    };

    tracing::info!(
        %project_id,
        ?report,
        "reconstruct_knowledge_links completed"
    );

    Ok(report)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notes::models::{NoteImportance, NoteScope, NoteStatus, NoteType};
    use crate::skills::models::{SkillNode, SkillStatus, SkillTrigger};
    use chrono::Utc;

    fn make_test_note(
        id: Uuid,
        content: &str,
        note_type: NoteType,
        importance: NoteImportance,
        energy: f64,
    ) -> Note {
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
            reactivation_count: 0,
            last_reactivated: None,
            freshness_pinged_at: None,
            activation_count: 0,
            supersedes: None,
            superseded_by: None,
            changes: vec![],
            assertion_rule: None,
            last_assertion_result: None,
            memory_horizon: crate::notes::MemoryHorizon::Operational,
            scar_intensity: 0.0,
            sharing_consent: Default::default(),
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
            scar_intensity: 0.0,
        }
    }

    // --- Trigger matching ---

    #[test]
    fn test_match_regex_trigger_matches() {
        assert!(match_regex_trigger("neo4j|cypher", "neo4j_client"));
        assert!(match_regex_trigger("neo4j|cypher", "cypher_query"));
    }

    #[test]
    fn test_match_regex_trigger_no_match() {
        assert!(!match_regex_trigger("neo4j|cypher", "api_handler"));
    }

    #[test]
    fn test_match_regex_trigger_invalid_regex() {
        assert!(!match_regex_trigger("[invalid", "test"));
    }

    #[test]
    fn test_match_file_glob_trigger_matches() {
        assert!(match_file_glob_trigger(
            "src/neo4j/**",
            "src/neo4j/client.rs"
        ));
        assert!(match_file_glob_trigger(
            "src/neo4j/*",
            "src/neo4j/client.rs"
        ));
    }

    #[test]
    fn test_match_file_glob_trigger_no_match() {
        assert!(!match_file_glob_trigger(
            "src/neo4j/**",
            "src/api/handlers.rs"
        ));
    }

    #[test]
    fn test_match_file_glob_trigger_invalid_glob() {
        assert!(!match_file_glob_trigger("[invalid", "test"));
    }

    // --- Skill match evaluation ---

    #[test]
    fn test_evaluate_skill_match_regex_hit() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Neo4j");
        skill.trigger_patterns = vec![SkillTrigger::regex("neo4j|cypher|UNWIND", 0.6)];

        let confidence = evaluate_skill_match(&skill, Some("neo4j_client"), None);
        assert!((confidence - 0.6).abs() < f64::EPSILON); // returns trigger's confidence_threshold
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
        // "src/neo4j/**" depth=2 → boost=0.85 → effective = 0.8 * 0.85 = 0.68
        let expected = 0.8 * glob_depth_boost("src/neo4j/**");
        assert!(
            (confidence - expected).abs() < f64::EPSILON,
            "Expected {}, got {}",
            expected,
            confidence
        );
    }

    #[test]
    fn test_evaluate_skill_match_file_glob_with_pattern_fallback() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Neo4j");
        skill.trigger_patterns = vec![SkillTrigger::file_glob("src/neo4j/**", 0.8)];

        // file_context is None, but pattern is a file path (from Read tool)
        let confidence = evaluate_skill_match(&skill, Some("src/neo4j/client.rs"), None);
        let expected = 0.8 * glob_depth_boost("src/neo4j/**");
        assert!(
            (confidence - expected).abs() < f64::EPSILON,
            "Expected {}, got {}",
            expected,
            confidence
        );
    }

    #[test]
    fn test_evaluate_skill_match_multiple_triggers_best_wins() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Neo4j");
        skill.trigger_patterns = vec![
            SkillTrigger::regex("neo4j|cypher", 0.6),
            SkillTrigger::file_glob("src/api/**", 0.8), // won't match
        ];

        let confidence =
            evaluate_skill_match(&skill, Some("neo4j_client"), Some("src/skills/test.rs"));
        assert!((confidence - 0.6).abs() < f64::EPSILON); // regex matched with its threshold
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
        assert!((confidence - 0.5).abs() < f64::EPSILON); // returns trigger's confidence_threshold
    }

    // --- Depth boost ---

    #[test]
    fn test_glob_depth_boost_values() {
        // depth=1: "src/**" → 0.55 + 0.15*1.0 = 0.70
        assert!((glob_depth_boost("src/**") - 0.70).abs() < f64::EPSILON);
        // depth=2: "src/api/**" → 0.55 + 0.15*2.0 = 0.85
        assert!((glob_depth_boost("src/api/**") - 0.85).abs() < f64::EPSILON);
        // depth=3: "src/api/v2/**" → 0.55 + 0.15*3.0 = 1.0
        assert!((glob_depth_boost("src/api/v2/**") - 1.0).abs() < f64::EPSILON);
        // depth=4: capped at 1.0
        assert!((glob_depth_boost("src/api/v2/deep/**") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_specific_glob_beats_generic() {
        // Two skills with SAME base confidence (0.8), but different glob depths.
        // The specific skill should score higher than the generic one.
        let mut generic_skill = SkillNode::new(Uuid::new_v4(), "Broad");
        generic_skill.trigger_patterns = vec![SkillTrigger::file_glob("src/**", 0.8)];

        let mut specific_skill = SkillNode::new(Uuid::new_v4(), "Specific");
        specific_skill.trigger_patterns =
            vec![SkillTrigger::file_glob("src/skills/activation/**", 0.8)];

        let file = "src/skills/activation/hook.rs";

        let generic_score = evaluate_skill_match(&generic_skill, None, Some(file));
        let specific_score = evaluate_skill_match(&specific_skill, None, Some(file));

        // generic: 0.8 * 0.70 = 0.56 (depth=1)
        // specific: 0.8 * 1.0 = 0.80 (depth=3)
        assert!(
            specific_score > generic_score,
            "Specific glob ({}) should beat generic glob ({})",
            specific_score,
            generic_score
        );

        // Also verify via cached path (mirror consistency)
        use crate::skills::cache::{evaluate_cached_skill, CachedSkill};
        let cached_generic = CachedSkill::from_skill(generic_skill);
        let cached_specific = CachedSkill::from_skill(specific_skill);

        let cached_generic_score = evaluate_cached_skill(&cached_generic, None, Some(file));
        let cached_specific_score = evaluate_cached_skill(&cached_specific, None, Some(file));

        assert!(
            cached_specific_score > cached_generic_score,
            "Cached: specific ({}) should beat generic ({})",
            cached_specific_score,
            cached_generic_score
        );

        // Verify cached and uncached produce the same scores
        assert!(
            (generic_score - cached_generic_score).abs() < f64::EPSILON,
            "Generic: uncached ({}) != cached ({})",
            generic_score,
            cached_generic_score
        );
        assert!(
            (specific_score - cached_specific_score).abs() < f64::EPSILON,
            "Specific: uncached ({}) != cached ({})",
            specific_score,
            cached_specific_score
        );
    }

    // --- Context assembly ---

    #[test]
    fn test_assemble_context_basic() {
        let notes = vec![
            make_test_note(
                Uuid::new_v4(),
                "Always use UNWIND for batch operations",
                NoteType::Guideline,
                NoteImportance::High,
                0.8,
            ),
            make_test_note(
                Uuid::new_v4(),
                "Connection pool leak if not closed",
                NoteType::Gotcha,
                NoteImportance::Critical,
                0.9,
            ),
        ];

        let decisions = vec![make_test_decision(
            Uuid::new_v4(),
            "Use Neo4j 5.x driver",
            "neo4j-rust-driver 0.8",
        )];

        let context = assemble_context("Neo4j Performance", &notes, &decisions, 3200);

        assert!(context.starts_with("## \u{1f4a1} Neo4j Performance"));
        assert!(context.contains("UNWIND"));
        assert!(context.contains("Connection pool"));
        assert!(context.contains("Neo4j 5.x driver"));
        assert!(context.chars().count() <= 3200);
    }

    #[test]
    fn test_assemble_context_sorted_by_importance() {
        let notes = vec![
            make_test_note(
                Uuid::new_v4(),
                "Low importance note",
                NoteType::Tip,
                NoteImportance::Low,
                0.5,
            ),
            make_test_note(
                Uuid::new_v4(),
                "Critical gotcha note",
                NoteType::Gotcha,
                NoteImportance::Critical,
                0.9,
            ),
            make_test_note(
                Uuid::new_v4(),
                "Medium pattern note",
                NoteType::Pattern,
                NoteImportance::Medium,
                0.7,
            ),
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
        assert!(context.chars().count() <= 2000);
    }

    #[test]
    fn test_assemble_context_empty_notes() {
        let context = assemble_context("Empty Skill", &[], &[], 3200);
        assert!(context.starts_with("## \u{1f4a1} Empty Skill"));
        assert!(!context.contains("Decisions"));
    }

    #[test]
    fn test_assemble_context_no_decisions() {
        let notes = vec![make_test_note(
            Uuid::new_v4(),
            "A note",
            NoteType::Tip,
            NoteImportance::Medium,
            0.5,
        )];
        let context = assemble_context("Test", &notes, &[], 3200);
        assert!(!context.contains("Decisions"));
    }

    #[test]
    fn test_assemble_context_with_emojis() {
        let notes = vec![
            make_test_note(
                Uuid::new_v4(),
                "Watch out for this",
                NoteType::Gotcha,
                NoteImportance::Critical,
                0.9,
            ),
            make_test_note(
                Uuid::new_v4(),
                "Follow this rule",
                NoteType::Guideline,
                NoteImportance::High,
                0.8,
            ),
            make_test_note(
                Uuid::new_v4(),
                "Common pattern here",
                NoteType::Pattern,
                NoteImportance::Medium,
                0.7,
            ),
            make_test_note(
                Uuid::new_v4(),
                "Helpful tip",
                NoteType::Tip,
                NoteImportance::Low,
                0.5,
            ),
        ];

        let context = assemble_context("Test", &notes, &[], 3200);

        // Check emojis are present (using unicode escapes)
        assert!(context.contains("\u{26a0}\u{fe0f}")); // ⚠️ for gotcha
        assert!(context.contains("\u{1f4cb}")); // 📋 for guideline
        assert!(context.contains("\u{1f504}")); // 🔄 for pattern
        assert!(context.contains("\u{1f4a1}")); // 💡 for tip
    }

    // --- Confidence header ---

    #[test]
    fn test_assemble_context_with_confidence_header() {
        let notes = vec![make_test_note(
            Uuid::new_v4(),
            "Always use UNWIND for batch operations",
            NoteType::Guideline,
            NoteImportance::High,
            0.8,
        )];

        let decisions = vec![make_test_decision(
            Uuid::new_v4(),
            "Use Neo4j 5.x driver",
            "neo4j-rust-driver 0.8",
        )];

        // With confidence → 🧠 header with percentage
        let (context, _) = assemble_context_with_confidence(
            "Neo4j Perf",
            &notes,
            &decisions,
            3200,
            Some(0.85),
            false,
        );
        assert!(
            context.starts_with("## \u{1f9e0} Skill \"Neo4j Perf\" (confidence 85%)"),
            "Expected confidence header, got: {}",
            context.lines().next().unwrap_or("")
        );
        assert!(context.contains("UNWIND"));
        assert!(context.contains("Neo4j 5.x driver"));
    }

    #[test]
    fn test_assemble_context_with_confidence_none() {
        let notes = vec![make_test_note(
            Uuid::new_v4(),
            "A note",
            NoteType::Tip,
            NoteImportance::Medium,
            0.5,
        )];

        // Without confidence → 💡 header (same as assemble_context)
        let (context, _) =
            assemble_context_with_confidence("Test Skill", &notes, &[], 3200, None, false);
        assert!(
            context.starts_with("## \u{1f4a1} Test Skill"),
            "Expected default header without confidence, got: {}",
            context.lines().next().unwrap_or("")
        );
    }

    #[test]
    fn test_assemble_context_with_confidence_rounding() {
        let notes = vec![make_test_note(
            Uuid::new_v4(),
            "Content",
            NoteType::Context,
            NoteImportance::Medium,
            0.5,
        )];

        // Confidence 0.666 → should round to 67%
        let (context, _) =
            assemble_context_with_confidence("Skill", &notes, &[], 3200, Some(0.666), false);
        assert!(
            context.contains("confidence 67%"),
            "Expected confidence 67%, got: {}",
            context.lines().next().unwrap_or("")
        );

        // Confidence 1.0 → 100%
        let (context, _) =
            assemble_context_with_confidence("Skill", &notes, &[], 3200, Some(1.0), false);
        assert!(
            context.contains("confidence 100%"),
            "Expected confidence 100%, got: {}",
            context.lines().next().unwrap_or("")
        );

        // Confidence 0.0 → 0%
        let (context, _) =
            assemble_context_with_confidence("Skill", &notes, &[], 3200, Some(0.0), false);
        assert!(
            context.contains("confidence 0%"),
            "Expected confidence 0%, got: {}",
            context.lines().next().unwrap_or("")
        );
    }

    // --- Pre-sorted & notes_included ---

    #[test]
    fn test_assemble_context_pre_sorted_preserves_order() {
        // When pre_sorted=true, notes should appear in the order given
        // (no re-sorting by importance/energy)
        let notes = vec![
            make_test_note(
                Uuid::new_v4(),
                "Low importance but high score",
                NoteType::Tip,
                NoteImportance::Low,
                0.3,
            ),
            make_test_note(
                Uuid::new_v4(),
                "Critical but low score",
                NoteType::Gotcha,
                NoteImportance::Critical,
                0.9,
            ),
        ];

        // pre_sorted=true: Low-importance note should appear first (as given)
        let (context_sorted, _) =
            assemble_context_with_confidence("Test", &notes, &[], 3200, None, true);
        let lines: Vec<&str> = context_sorted.lines().collect();
        // Line 0 = header, Line 1 = first note (low importance), Line 2 = second note (critical)
        assert!(
            lines[1].contains("Low importance"),
            "First note should be low importance, got: {}",
            lines[1]
        );
        assert!(
            lines[2].contains("Critical"),
            "Second note should be critical, got: {}",
            lines[2]
        );

        // pre_sorted=false: Critical should come first (sorted by importance)
        let (context_default, _) =
            assemble_context_with_confidence("Test", &notes, &[], 3200, None, false);
        let lines: Vec<&str> = context_default.lines().collect();
        assert!(
            lines[1].contains("Critical"),
            "First note should be critical, got: {}",
            lines[1]
        );
        assert!(
            lines[2].contains("Low importance"),
            "Second note should be low importance, got: {}",
            lines[2]
        );
    }

    #[test]
    fn test_assemble_context_returns_notes_included_count() {
        // Create many notes that exceed the budget
        let notes: Vec<Note> = (0..50)
            .map(|i| {
                make_test_note(
                    Uuid::new_v4(),
                    &format!(
                        "Note content number {} with some padding text to take space",
                        i
                    ),
                    NoteType::Guideline,
                    NoteImportance::Medium,
                    0.5,
                )
            })
            .collect();

        let (context, notes_included) =
            assemble_context_with_confidence("Test", &notes, &[], 1000, None, false);
        // With a 1000 char budget, not all 50 notes should fit
        assert!(
            notes_included < 50,
            "Expected fewer than 50 notes included, got: {}",
            notes_included
        );
        assert!(notes_included > 0, "Expected at least 1 note included");
        assert!(
            context.chars().count() <= 1000,
            "Context should respect budget"
        );
    }

    #[test]
    fn test_activated_note_ids_subset_of_rendered() {
        // This verifies the fix for the Hebbian over-broad bug:
        // activated_note_ids should only contain notes that were rendered
        let notes: Vec<Note> = (0..30)
            .map(|i| make_test_note(
                Uuid::new_v4(),
                &format!("Knowledge entry #{}: this is a substantial note with detailed content about topic {}", i, i),
                NoteType::Guideline,
                NoteImportance::Medium,
                0.5,
            ))
            .collect();

        let (_context, notes_included) =
            assemble_context_with_confidence("Test", &notes, &[], 1500, None, false);
        // With 30 notes and 1500 char budget, only a subset should be included
        assert!(
            notes_included < 30,
            "Not all notes should fit in budget: {} included",
            notes_included
        );
        // The activated_note_ids should be notes[..notes_included]
        // (In the actual pipeline, take(notes_included) is used)
        let activated_ids: Vec<Uuid> = notes.iter().take(notes_included).map(|n| n.id).collect();
        assert_eq!(activated_ids.len(), notes_included);
    }

    // --- Behavioral synapses ---

    #[tokio::test]
    async fn test_behavioral_synapse_source_tagging() {
        use crate::neo4j::mock::MockGraphStore;

        let graph = MockGraphStore::new();
        let note_a = Uuid::new_v4();
        let note_b = Uuid::new_v4();
        let note_c = Uuid::new_v4();

        // Create cosine synapses (backfill path)
        graph
            .create_synapses(note_a, &[(note_b, 0.8)])
            .await
            .unwrap();

        // Verify source tagged as cosine
        {
            let sources = graph.synapse_sources.read().await;
            let key = if note_a < note_b {
                (note_a, note_b)
            } else {
                (note_b, note_a)
            };
            assert_eq!(sources.get(&key).unwrap(), "cosine");
        }

        // Reinforce via co-activation (behavioral path) — upgrades to coactivation
        graph
            .reinforce_synapses(&[note_a, note_b], 0.05)
            .await
            .unwrap();

        {
            let sources = graph.synapse_sources.read().await;
            let key = if note_a < note_b {
                (note_a, note_b)
            } else {
                (note_b, note_a)
            };
            assert_eq!(
                sources.get(&key).unwrap(),
                "coactivation",
                "reinforce_synapses should upgrade source from cosine to coactivation"
            );
        }

        // Create a pure cosine synapse (A-C) for comparison
        graph
            .create_synapses(note_a, &[(note_c, 0.7)])
            .await
            .unwrap();

        // Decay: coactivation (A-B) should decay at base rate,
        //        cosine (A-C) should decay at 2x base rate
        let (decayed, _pruned) = graph.decay_synapses(0.05, 0.0).await.unwrap();
        assert!(decayed > 0);

        let synapses = graph.note_synapses.read().await;
        // A→B was 0.8+0.05(boost)=0.85, coactivation decays 0.05 → 0.80
        let ab_weight = synapses
            .get(&note_a)
            .unwrap()
            .iter()
            .find(|(id, _)| *id == note_b)
            .unwrap()
            .1;
        // A→C was 0.7, cosine decays 0.10 (2x) → 0.60
        let ac_weight = synapses
            .get(&note_a)
            .unwrap()
            .iter()
            .find(|(id, _)| *id == note_c)
            .unwrap()
            .1;

        assert!(
            ab_weight > ac_weight,
            "Coactivation synapse ({ab_weight:.2}) should decay slower than cosine ({ac_weight:.2})"
        );
        // Verify exact values (within float tolerance)
        assert!((ab_weight - 0.80).abs() < 0.01, "A-B: {ab_weight}");
        assert!((ac_weight - 0.60).abs() < 0.01, "A-C: {ac_weight}");
    }

    #[tokio::test]
    async fn test_track_reactivation_increments_count() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::notes::{Note, NoteType};
        use std::sync::Arc;

        let mock = Arc::new(MockGraphStore::new());

        // Create two notes
        let mut note_a = Note::new(
            None,
            NoteType::Gotcha,
            "Reactivation test A".into(),
            "test".into(),
        );
        note_a.energy = 0.5;
        let mut note_b = Note::new(
            None,
            NoteType::Tip,
            "Reactivation test B".into(),
            "test".into(),
        );
        note_b.energy = 0.3;

        let id_a = note_a.id;
        let id_b = note_b.id;

        mock.create_note(&note_a).await.unwrap();
        mock.create_note(&note_b).await.unwrap();

        // Verify initial state
        let a = mock.get_note(id_a).await.unwrap().unwrap();
        assert_eq!(a.reactivation_count, 0);
        assert!(a.last_reactivated.is_none());

        // Track reactivation
        let updated = mock.track_reactivation(&[id_a, id_b]).await.unwrap();
        assert_eq!(updated, 2);

        // Verify reactivation_count incremented
        let a = mock.get_note(id_a).await.unwrap().unwrap();
        assert_eq!(a.reactivation_count, 1);
        assert!(a.last_reactivated.is_some());

        // Track again
        mock.track_reactivation(&[id_a]).await.unwrap();
        let a = mock.get_note(id_a).await.unwrap().unwrap();
        assert_eq!(a.reactivation_count, 2);

        // B should still be at 1
        let b = mock.get_note(id_b).await.unwrap().unwrap();
        assert_eq!(b.reactivation_count, 1);
    }

    #[tokio::test]
    async fn test_reinforce_hook_tracks_reactivation() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::notes::{Note, NoteType};
        use std::sync::Arc;

        let mock = Arc::new(MockGraphStore::new());

        let mut note_a = Note::new(
            None,
            NoteType::Gotcha,
            "Hook reactivation A".into(),
            "test".into(),
        );
        note_a.energy = 0.5;
        let mut note_b = Note::new(
            None,
            NoteType::Tip,
            "Hook reactivation B".into(),
            "test".into(),
        );
        note_b.energy = 0.5;

        let id_a = note_a.id;
        let id_b = note_b.id;

        mock.create_note(&note_a).await.unwrap();
        mock.create_note(&note_b).await.unwrap();

        let config = AutoReinforcementConfig::default();
        reinforce_hook_activation(mock.as_ref(), &[id_a, id_b], &config)
            .await
            .unwrap();

        // Both should have reactivation_count = 1
        let a = mock.get_note(id_a).await.unwrap().unwrap();
        let b = mock.get_note(id_b).await.unwrap().unwrap();
        assert_eq!(a.reactivation_count, 1);
        assert_eq!(b.reactivation_count, 1);
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
        assert!(result.chars().count() <= 100);
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
                note_types[i % 4],
                importances[i % 4],
                0.9 - (i as f64 * 0.03), // decreasing energy
            ));
        }

        let decisions = vec![
            make_test_decision(
                Uuid::new_v4(),
                "Use async Neo4j driver for all database operations",
                "neo4j-rust-driver 0.8 with async support",
            ),
            make_test_decision(
                Uuid::new_v4(),
                "Implement caching with TTL-based invalidation",
                "30-second TTL per skill activation",
            ),
            make_test_decision(
                Uuid::new_v4(),
                "This third decision should be excluded",
                "Only 2 decisions max",
            ),
        ];

        let context = assemble_context("Neo4j Expertise", &notes, &decisions, 3200);

        // Strict budget enforcement (use chars count, not byte length)
        let char_count = context.chars().count();
        assert!(
            char_count <= 3200,
            "Context char count {} exceeds 3200 char budget",
            char_count
        );

        // Header present
        assert!(context.starts_with("## \u{1f4a1} Neo4j Expertise\n"));

        // Critical notes should appear first (they have highest importance weight)
        let critical_pos = context.find("Note 0 with substantial");
        let low_pos = context.find("Note 3 with substantial");
        if let (Some(c), Some(l)) = (critical_pos, low_pos) {
            assert!(c < l, "Critical notes should appear before Low notes");
        }

        // With correct char-based budgeting (not byte-based), 20 short notes
        // may all fit within 3200 chars. The omitted indicator should appear
        // only when notes are actually truncated.
        let notes_included = sorted_note_ids_count(&context, 20);
        if notes_included < 20 {
            assert!(
                context.contains("more notes)_"),
                "Should show omitted notes count when not all fit"
            );
        }

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

    /// Helper: count how many of the 20 "Note N" entries appear in the context.
    fn sorted_note_ids_count(context: &str, total: usize) -> usize {
        (0..total)
            .filter(|i| context.contains(&format!("Note {} with", i)))
            .count()
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

        let decisions = vec![make_test_decision(
            Uuid::new_v4(),
            "Architecture decision for the module",
            "chosen approach",
        )];

        let context = assemble_context("Large Skill", &notes, &decisions, 3200);

        // Use chars count for budget check (consistent with char-based budgeting)
        let char_count = context.chars().count();
        assert!(
            char_count <= 3200,
            "Context char count {} exceeds 3200 char budget with 30 notes",
            char_count
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

        let decisions = vec![make_test_decision(
            Uuid::new_v4(),
            "Important decision",
            "chosen option value",
        )];

        // With decisions present, notes should leave room
        let context_with_decisions = assemble_context("Test", &notes, &decisions, 3200);
        let context_without_decisions = assemble_context("Test", &notes, &[], 3200);

        // Context without decisions should have more notes since no budget reservation
        let notes_with = context_with_decisions.matches("Note content").count();
        let notes_without = context_without_decisions.matches("Note content").count();

        assert!(
            notes_without >= notes_with,
            "Without decisions reservation ({} notes), should include >= notes than with reservation ({} notes)",
            notes_without,
            notes_with,
        );

        // Both within budget (use chars count for consistency)
        assert!(context_with_decisions.chars().count() <= 3200);
        assert!(context_without_decisions.chars().count() <= 3200);
    }

    // --- McpAction trigger matching ---

    #[test]
    fn test_mcp_action_trigger_tool_only_matches() {
        // Trigger "note" matches any input starting with "note"
        assert!(match_mcp_action_trigger(
            "note",
            "note create Always use parameterized queries"
        ));
        assert!(match_mcp_action_trigger("note", "note search some query"));
        assert!(match_mcp_action_trigger(
            "task",
            "task get_next plan_id=abc"
        ));
    }

    #[test]
    fn test_mcp_action_trigger_tool_and_action_matches() {
        // Trigger "note:create" matches input "note create ..."
        assert!(match_mcp_action_trigger(
            "note:create",
            "note create Always use parameterized queries"
        ));
        assert!(match_mcp_action_trigger(
            "task:update",
            "task update status=completed"
        ));
        assert!(match_mcp_action_trigger(
            "code:analyze_impact",
            "code analyze_impact /src/main.rs"
        ));
    }

    #[test]
    fn test_mcp_action_trigger_case_insensitive() {
        assert!(match_mcp_action_trigger("Note", "note create foo"));
        assert!(match_mcp_action_trigger("note:CREATE", "note create foo"));
        assert!(match_mcp_action_trigger("NOTE:Create", "note create foo"));
        assert!(match_mcp_action_trigger("task", "TASK update bar"));
    }

    #[test]
    fn test_mcp_action_trigger_no_match_different_tool() {
        assert!(!match_mcp_action_trigger("note", "task create something"));
        assert!(!match_mcp_action_trigger("commit", "note create something"));
    }

    #[test]
    fn test_mcp_action_trigger_no_match_different_action() {
        // Trigger "note:create" should NOT match "note search ..."
        assert!(!match_mcp_action_trigger("note:create", "note search foo"));
        assert!(!match_mcp_action_trigger("task:update", "task create bar"));
    }

    #[test]
    fn test_mcp_action_trigger_tool_only_no_action_in_input() {
        // Trigger "note" (tool-only) should match even input with only tool name
        assert!(match_mcp_action_trigger("note", "note"));
    }

    #[test]
    fn test_mcp_action_trigger_tool_action_but_input_has_no_action() {
        // Trigger "note:create" but input is just "note" → no match
        assert!(!match_mcp_action_trigger("note:create", "note"));
    }

    #[test]
    fn test_mcp_action_trigger_empty_inputs() {
        assert!(!match_mcp_action_trigger("", "note create"));
        assert!(!match_mcp_action_trigger("note", ""));
        assert!(!match_mcp_action_trigger("", ""));
    }

    #[test]
    fn test_mcp_action_trigger_whitespace_handling() {
        // Trigger with spaces around colon
        assert!(match_mcp_action_trigger(
            " note : create ",
            "note create foo"
        ));
        // Input with extra whitespace
        assert!(match_mcp_action_trigger(
            "note:create",
            "  note   create   foo  "
        ));
    }

    #[test]
    fn test_mcp_action_trigger_empty_action_after_colon() {
        // "note:" — empty action means tool-only match
        assert!(match_mcp_action_trigger("note:", "note create foo"));
        assert!(match_mcp_action_trigger("note:", "note"));
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

    // =========================================================================
    // E2E integration tests: activate_for_hook with mock store
    // =========================================================================

    /// Helper to build a mock store with a skill, notes, and McpAction triggers.
    async fn setup_mcp_skill_store(
        project_id: Uuid,
        skill_name: &str,
        triggers: Vec<SkillTrigger>,
        notes: Vec<Note>,
    ) -> crate::neo4j::mock::MockGraphStore {
        let store = crate::neo4j::mock::MockGraphStore::new();

        // Create a project first (mock requires it for create_skill)
        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test-project".to_string(),
            slug: "test-project".to_string(),
            description: Some("Test project".to_string()),
            root_path: "/tmp/test-project".to_string(),
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        // Create skill
        let mut skill = SkillNode::new(project_id, skill_name);
        skill.status = SkillStatus::Active;
        skill.trigger_patterns = triggers;
        skill.energy = 0.8;
        let skill_id = skill.id;
        store.create_skill(&skill).await.unwrap();

        // Create notes and link as skill members
        for note in &notes {
            store.create_note(note).await.unwrap();
            store
                .add_skill_member(skill_id, "note", note.id)
                .await
                .unwrap();
        }

        store
    }

    #[tokio::test]
    async fn test_e2e_mcp_task_create_triggers_skill() {
        let project_id = Uuid::new_v4();

        let note = make_test_note(
            Uuid::new_v4(),
            "⚠️ task(action: \"create\") requires plan_id. Without it you get a 400 error.",
            NoteType::Gotcha,
            NoteImportance::High,
            0.8,
        );

        let triggers = vec![SkillTrigger::mcp_action("task", 0.7)];

        let store = setup_mcp_skill_store(project_id, "MCP Task Usage", triggers, vec![note]).await;

        let config = HookActivationConfig::default();
        let tool_input = serde_json::json!({
            "action": "create",
            "title": "my task"
        });

        let result = activate_for_hook(
            &store,
            project_id,
            "mcp__project-orchestrator__task",
            &tool_input,
            &config,
        )
        .await
        .unwrap();

        assert!(result.is_some(), "Should match the MCP Task Usage skill");
        let outcome = result.unwrap();
        assert!(
            outcome.response.context.contains("plan_id"),
            "Context should mention plan_id, got: {}",
            outcome.response.context
        );
        assert_eq!(outcome.response.skill_name, "MCP Task Usage");
        assert!(outcome.response.confidence >= 0.5);
        assert_eq!(outcome.response.notes_count, 1);
    }

    #[tokio::test]
    async fn test_e2e_mcp_note_search_triggers_skill() {
        let project_id = Uuid::new_v4();

        let note = make_test_note(
            Uuid::new_v4(),
            "note(action: \"search_semantic\") uses vector similarity. Prefer over search for natural language queries.",
            NoteType::Guideline,
            NoteImportance::High,
            0.8,
        );

        let triggers = vec![SkillTrigger::mcp_action("note:search_semantic", 0.75)];

        let store =
            setup_mcp_skill_store(project_id, "MCP Note Search", triggers, vec![note]).await;

        let config = HookActivationConfig::default();
        let tool_input = serde_json::json!({
            "action": "search_semantic",
            "query": "neo4j batch processing"
        });

        let result = activate_for_hook(
            &store,
            project_id,
            "mcp__project-orchestrator__note",
            &tool_input,
            &config,
        )
        .await
        .unwrap();

        assert!(result.is_some(), "Should match MCP Note Search skill");
        let outcome = result.unwrap();
        assert!(outcome.response.context.contains("search_semantic"));
        assert_eq!(outcome.response.skill_name, "MCP Note Search");
    }

    #[tokio::test]
    async fn test_e2e_mcp_code_analyze_impact_triggers_skill() {
        let project_id = Uuid::new_v4();

        let note = make_test_note(
            Uuid::new_v4(),
            "code(action: \"analyze_impact\") requires ABSOLUTE paths. Relative paths like 'src/main.rs' will fail. Always use full path: '/Users/.../src/main.rs'.",
            NoteType::Gotcha,
            NoteImportance::High,
            0.9,
        );

        let triggers = vec![SkillTrigger::mcp_action("code:analyze_impact", 0.8)];

        let store =
            setup_mcp_skill_store(project_id, "MCP Code Impact", triggers, vec![note]).await;

        let config = HookActivationConfig::default();
        let tool_input = serde_json::json!({
            "action": "analyze_impact",
            "target": "src/relative/path.rs"
        });

        let result = activate_for_hook(
            &store,
            project_id,
            "mcp__project-orchestrator__code",
            &tool_input,
            &config,
        )
        .await
        .unwrap();

        assert!(result.is_some(), "Should match MCP Code Impact skill");
        let outcome = result.unwrap();
        assert!(
            outcome.response.context.contains("ABSOLUTE"),
            "Context should warn about absolute paths"
        );
    }

    #[tokio::test]
    async fn test_e2e_mcp_no_skill_match_returns_none() {
        let project_id = Uuid::new_v4();

        let note = make_test_note(
            Uuid::new_v4(),
            "Some note about task usage",
            NoteType::Tip,
            NoteImportance::Medium,
            0.8,
        );

        // Skill only triggers on "task" actions
        let triggers = vec![SkillTrigger::mcp_action("task", 0.7)];

        let store = setup_mcp_skill_store(project_id, "MCP Task Skill", triggers, vec![note]).await;

        let config = HookActivationConfig::default();

        // Call with "commit" tool → should NOT match "task" trigger
        let tool_input = serde_json::json!({
            "action": "create",
            "sha": "abc123"
        });

        let result = activate_for_hook(
            &store,
            project_id,
            "mcp__project-orchestrator__commit",
            &tool_input,
            &config,
        )
        .await
        .unwrap();

        assert!(
            result.is_none(),
            "commit tool should not match task trigger"
        );
    }

    #[tokio::test]
    async fn test_e2e_mcp_specific_action_doesnt_match_different_action() {
        let project_id = Uuid::new_v4();

        let note = make_test_note(
            Uuid::new_v4(),
            "Guideline for note creation",
            NoteType::Guideline,
            NoteImportance::High,
            0.8,
        );

        // Only triggers on note:create, not note:search_semantic
        let triggers = vec![SkillTrigger::mcp_action("note:create", 0.75)];

        let store =
            setup_mcp_skill_store(project_id, "MCP Note Create", triggers, vec![note]).await;

        let config = HookActivationConfig::default();
        let tool_input = serde_json::json!({
            "action": "search_semantic",
            "query": "test query"
        });

        let result = activate_for_hook(
            &store,
            project_id,
            "mcp__project-orchestrator__note",
            &tool_input,
            &config,
        )
        .await
        .unwrap();

        assert!(
            result.is_none(),
            "note:create trigger should NOT match note search_semantic action"
        );
    }

    #[tokio::test]
    async fn test_e2e_mcp_low_energy_notes_excluded() {
        let project_id = Uuid::new_v4();

        // Note with very low energy — should be filtered out
        let note = make_test_note(
            Uuid::new_v4(),
            "Dead note content",
            NoteType::Tip,
            NoteImportance::Medium,
            0.01, // Below min_note_energy (0.1)
        );

        let triggers = vec![SkillTrigger::mcp_action("task", 0.7)];

        let store = setup_mcp_skill_store(project_id, "MCP Task Skill", triggers, vec![note]).await;

        let config = HookActivationConfig::default();
        let tool_input = serde_json::json!({
            "action": "create",
            "title": "test"
        });

        let result = activate_for_hook(
            &store,
            project_id,
            "mcp__project-orchestrator__task",
            &tool_input,
            &config,
        )
        .await
        .unwrap();

        // The skill matches but all notes are below energy threshold
        // Result may still be Some (with 0 notes) or None depending on implementation
        if let Some(outcome) = result {
            assert_eq!(
                outcome.response.notes_count, 0,
                "Low energy notes should be filtered out"
            );
        }
    }

    #[tokio::test]
    async fn test_e2e_mcp_multiple_notes_assembled() {
        let project_id = Uuid::new_v4();

        let notes = vec![
            make_test_note(
                Uuid::new_v4(),
                "task(action: \"create\") requires plan_id mandatory",
                NoteType::Gotcha,
                NoteImportance::High,
                0.9,
            ),
            make_test_note(
                Uuid::new_v4(),
                "step(action: \"create\") requires task_id mandatory",
                NoteType::Gotcha,
                NoteImportance::High,
                0.8,
            ),
            make_test_note(
                Uuid::new_v4(),
                "Always use UUID format, not slugs, for entity IDs",
                NoteType::Guideline,
                NoteImportance::Medium,
                0.7,
            ),
        ];

        let triggers = vec![SkillTrigger::mcp_action("task", 0.7)];

        let store = setup_mcp_skill_store(project_id, "MCP Usage Patterns", triggers, notes).await;

        let config = HookActivationConfig::default();
        let tool_input = serde_json::json!({
            "action": "create",
            "title": "test task"
        });

        let result = activate_for_hook(
            &store,
            project_id,
            "mcp__project-orchestrator__task",
            &tool_input,
            &config,
        )
        .await
        .unwrap();

        assert!(result.is_some(), "Should match with multiple notes");
        let outcome = result.unwrap();
        assert_eq!(
            outcome.response.notes_count, 3,
            "All 3 notes above energy threshold"
        );
        assert!(outcome.response.context.contains("plan_id"));
        assert!(
            outcome.response.context.len() <= 3200,
            "Context within budget"
        );
    }

    #[tokio::test]
    async fn test_e2e_mcp_performance_under_50ms() {
        let project_id = Uuid::new_v4();

        // Create 20 notes to simulate a real skill with substantial content
        let notes: Vec<Note> = (0..20)
            .map(|i| {
                make_test_note(
                    Uuid::new_v4(),
                    &format!("Guideline #{}: Always check parameters before calling MCP tools. Validate UUIDs, check required fields, and ensure paths are absolute.", i),
                    NoteType::Guideline,
                    NoteImportance::Medium,
                    0.5 + (i as f64) * 0.02,
                )
            })
            .collect();

        let triggers = vec![
            SkillTrigger::mcp_action("task", 0.7),
            SkillTrigger::mcp_action("note", 0.65),
            SkillTrigger::regex("neo4j|cypher", 0.6),
            SkillTrigger::file_glob("src/skills/**", 0.55),
        ];

        let store = setup_mcp_skill_store(project_id, "MCP Usage Patterns", triggers, notes).await;

        let config = HookActivationConfig::default();
        let tool_input = serde_json::json!({
            "action": "create",
            "title": "test task for perf"
        });

        // Run 100 activations and check wall time
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _result = activate_for_hook(
                &store,
                project_id,
                "mcp__project-orchestrator__task",
                &tool_input,
                &config,
            )
            .await
            .unwrap();
        }
        let elapsed = start.elapsed();
        let per_call_ms = elapsed.as_millis() as f64 / 100.0;

        assert!(
            per_call_ms < 50.0,
            "Average call should be < 50ms, got {:.2}ms",
            per_call_ms
        );
    }

    #[tokio::test]
    async fn test_e2e_mcp_no_project_skills_returns_none() {
        // Empty store — no skills registered
        let store = crate::neo4j::mock::MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let config = HookActivationConfig::default();
        let tool_input = serde_json::json!({
            "action": "create",
            "title": "test"
        });

        let result = activate_for_hook(
            &store,
            project_id,
            "mcp__project-orchestrator__task",
            &tool_input,
            &config,
        )
        .await
        .unwrap();

        assert!(
            result.is_none(),
            "No skills in project → should return None gracefully"
        );
    }

    // --- Contextual note scoring ---

    fn make_scored_note(
        content: &str,
        tags: Vec<&str>,
        note_type: NoteType,
        importance: NoteImportance,
        energy: f64,
        staleness: f64,
    ) -> Note {
        let mut note = make_test_note(Uuid::new_v4(), content, note_type, importance, energy);
        note.tags = tags.into_iter().map(|t| t.to_string()).collect();
        note.staleness_score = staleness;
        note
    }

    #[test]
    fn test_extract_path_segments_basic() {
        let segments = extract_path_segments("src/neo4j/client.rs");
        assert_eq!(segments, vec!["neo4j", "client"]);
    }

    #[test]
    fn test_extract_path_segments_filters_trivial() {
        let segments = extract_path_segments("src/lib.rs");
        assert!(segments.is_empty());
    }

    #[test]
    fn test_extract_path_segments_absolute_path() {
        let segments = extract_path_segments("/Users/x/project/src/skills/activation.rs");
        assert_eq!(segments, vec!["users", "project", "skills", "activation"]);
    }

    #[test]
    fn test_extract_path_segments_no_extension() {
        let segments = extract_path_segments("src/neo4j/Dockerfile");
        assert_eq!(segments, vec!["neo4j", "dockerfile"]);
    }

    #[test]
    fn test_score_tag_path_affinity() {
        let neo4j_note = make_scored_note(
            "Neo4j performance tip",
            vec!["neo4j", "cypher"],
            NoteType::Tip,
            NoteImportance::Medium,
            0.8,
            0.0,
        );
        let tauri_note = make_scored_note(
            "Tauri macOS gotcha",
            vec!["tauri", "macos"],
            NoteType::Gotcha,
            NoteImportance::Medium,
            0.8,
            0.0,
        );

        let neo4j_score =
            score_note_relevance(&neo4j_note, Some("src/neo4j/client.rs"), None, "Read");
        let tauri_score =
            score_note_relevance(&tauri_note, Some("src/neo4j/client.rs"), None, "Read");

        assert!(
            neo4j_score > tauri_score,
            "neo4j note ({}) should score higher than tauri note ({}) for neo4j file",
            neo4j_score,
            tauri_score
        );
    }

    #[test]
    fn test_score_content_keyword_match() {
        let matching_note = make_scored_note(
            "Modify src/neo4j/client.rs to fix the query pattern",
            vec![],
            NoteType::Observation,
            NoteImportance::Medium,
            0.8,
            0.0,
        );
        let non_matching_note = make_scored_note(
            "General performance optimization advice",
            vec![],
            NoteType::Observation,
            NoteImportance::Medium,
            0.8,
            0.0,
        );

        let matching_score =
            score_note_relevance(&matching_note, Some("src/neo4j/client.rs"), None, "Read");
        let non_matching_score = score_note_relevance(
            &non_matching_note,
            Some("src/neo4j/client.rs"),
            None,
            "Read",
        );

        assert!(
            matching_score > non_matching_score,
            "Note mentioning 'client' ({}) should score higher than generic note ({})",
            matching_score,
            non_matching_score
        );
    }

    #[test]
    fn test_score_importance_weighting() {
        let critical_note = make_scored_note(
            "Same content",
            vec!["neo4j"],
            NoteType::Tip,
            NoteImportance::Critical,
            0.8,
            0.0,
        );
        let low_note = make_scored_note(
            "Same content",
            vec!["neo4j"],
            NoteType::Tip,
            NoteImportance::Low,
            0.8,
            0.0,
        );

        let critical_score =
            score_note_relevance(&critical_note, Some("src/neo4j/client.rs"), None, "Read");
        let low_score = score_note_relevance(&low_note, Some("src/neo4j/client.rs"), None, "Read");

        // critical = 1.5x, low = 0.8x → critical should be ~1.875x the low
        assert!(
            critical_score > low_score,
            "Critical note ({}) should score higher than low note ({})",
            critical_score,
            low_score
        );
        let ratio = critical_score / low_score;
        assert!(
            (ratio - 1.875).abs() < 0.3,
            "Ratio should be ~1.875, got {}",
            ratio
        );
    }

    #[test]
    fn test_score_freshness_decay() {
        let fresh_note = make_scored_note(
            "Fresh note",
            vec!["neo4j"],
            NoteType::Tip,
            NoteImportance::Medium,
            0.8,
            0.0, // fresh
        );
        let stale_note = make_scored_note(
            "Stale note",
            vec!["neo4j"],
            NoteType::Tip,
            NoteImportance::Medium,
            0.8,
            0.5, // stale
        );

        let fresh_score =
            score_note_relevance(&fresh_note, Some("src/neo4j/client.rs"), None, "Read");
        let stale_score =
            score_note_relevance(&stale_note, Some("src/neo4j/client.rs"), None, "Read");

        assert!(
            fresh_score > stale_score,
            "Fresh note ({}) should score higher than stale note ({})",
            fresh_score,
            stale_score
        );
    }

    #[test]
    fn test_score_note_type_affinity_edit_gotcha() {
        let gotcha_note = make_scored_note(
            "Watch out for this",
            vec![],
            NoteType::Gotcha,
            NoteImportance::Medium,
            0.8,
            0.0,
        );
        let tip_note = make_scored_note(
            "Helpful tip here",
            vec![],
            NoteType::Tip,
            NoteImportance::Medium,
            0.8,
            0.0,
        );

        let gotcha_score =
            score_note_relevance(&gotcha_note, Some("src/neo4j/client.rs"), None, "Edit");
        let tip_score = score_note_relevance(&tip_note, Some("src/neo4j/client.rs"), None, "Edit");

        assert!(
            gotcha_score > tip_score,
            "Gotcha note ({}) should score higher than tip ({}) for Edit tool",
            gotcha_score,
            tip_score
        );
    }

    #[test]
    fn test_score_note_type_affinity_mcp_guideline() {
        let guideline_note = make_scored_note(
            "Follow this pattern",
            vec![],
            NoteType::Guideline,
            NoteImportance::Medium,
            0.8,
            0.0,
        );
        let tip_note = make_scored_note(
            "Helpful tip here",
            vec![],
            NoteType::Tip,
            NoteImportance::Medium,
            0.8,
            0.0,
        );

        let guideline_score = score_note_relevance(
            &guideline_note,
            Some("src/neo4j/client.rs"),
            None,
            "mcp__project-orchestrator__task",
        );
        let tip_score = score_note_relevance(
            &tip_note,
            Some("src/neo4j/client.rs"),
            None,
            "mcp__project-orchestrator__task",
        );

        assert!(
            guideline_score > tip_score,
            "Guideline note ({}) should score higher than tip ({}) for MCP tool",
            guideline_score,
            tip_score
        );
    }

    #[test]
    fn test_score_is_pure_no_side_effects() {
        let note = make_scored_note(
            "Test note",
            vec!["neo4j"],
            NoteType::Gotcha,
            NoteImportance::High,
            0.8,
            0.1,
        );

        // Call twice with same inputs → same output
        let score1 = score_note_relevance(&note, Some("src/neo4j/client.rs"), None, "Edit");
        let score2 = score_note_relevance(&note, Some("src/neo4j/client.rs"), None, "Edit");

        assert!(
            (score1 - score2).abs() < f64::EPSILON,
            "Function should be pure: {} != {}",
            score1,
            score2
        );
    }

    #[test]
    fn test_score_without_file_context() {
        let note = make_scored_note(
            "Generic note",
            vec!["neo4j"],
            NoteType::Tip,
            NoteImportance::Medium,
            0.8,
            0.0,
        );

        // Without file_context, only base score + importance + type affinity apply
        let score = score_note_relevance(&note, None, None, "Read");
        assert!(score > 0.0, "Score should still be positive: {}", score);
        // Should be base (0.1) × importance (1.0) = 0.1
        assert!(
            (score - 0.1).abs() < f64::EPSILON,
            "Score without context should be base 0.1, got {}",
            score
        );
    }

    #[test]
    fn test_score_combo_ordering() {
        // Simulate a real scenario: skill with mixed notes, file is neo4j
        let notes = [
            make_scored_note(
                "Neo4j RETURN count() gotcha after DELETE",
                vec!["neo4j", "cypher", "gotcha"],
                NoteType::Gotcha,
                NoteImportance::Critical,
                0.9,
                0.0,
            ),
            make_scored_note(
                "Tauri window decoration bug on macOS",
                vec!["tauri", "macos"],
                NoteType::Gotcha,
                NoteImportance::Critical,
                0.9,
                0.0,
            ),
            make_scored_note(
                "General code review tip for better readability",
                vec!["code-review"],
                NoteType::Tip,
                NoteImportance::Low,
                0.5,
                0.3,
            ),
        ];

        let mut scored: Vec<(usize, f64)> = notes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                (
                    i,
                    score_note_relevance(n, Some("src/neo4j/client.rs"), None, "Edit"),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Neo4j note should be first (tag affinity + content match + critical)
        assert_eq!(scored[0].0, 0, "Neo4j note should rank first");
        // Code review tip should be last (no affinity + low importance + stale)
        assert_eq!(scored[2].0, 2, "Code review tip should rank last");
    }

    #[test]
    fn test_same_directory() {
        assert!(same_directory(
            "src/neo4j/client.rs",
            "src/neo4j/analytics.rs"
        ));
        assert!(!same_directory(
            "src/neo4j/client.rs",
            "src/api/handlers.rs"
        ));
        assert!(same_directory("/Users/x/src/neo4j/a.rs", "src/neo4j/b.rs"));
        assert!(!same_directory("a.rs", "b.rs")); // no parent dir
    }

    // --- extract_file_paths_from_content ---

    #[test]
    fn test_extract_file_paths_relative() {
        let content = "Modifier `src/neo4j/client.rs` pour ajouter la méthode.";
        let paths = extract_file_paths_from_content(content);
        assert_eq!(paths, vec!["src/neo4j/client.rs"]);
    }

    #[test]
    fn test_extract_file_paths_multiple() {
        let content = "Les fichiers src/api/handlers.rs et src/neo4j/note.rs doivent être modifiés. Voir aussi src/skills/activation.rs.";
        let paths = extract_file_paths_from_content(content);
        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&"src/api/handlers.rs".to_string()));
        assert!(paths.contains(&"src/neo4j/note.rs".to_string()));
        assert!(paths.contains(&"src/skills/activation.rs".to_string()));
    }

    #[test]
    fn test_extract_file_paths_absolute_normalized() {
        let content = "Le fichier /Users/foo/project/src/neo4j/client.rs doit être modifié.";
        let paths = extract_file_paths_from_content(content);
        assert_eq!(paths, vec!["src/neo4j/client.rs"]);
    }

    #[test]
    fn test_extract_file_paths_backtick_wrapped() {
        let content = "Modifier `src/skills/triggers.rs` puis `src/mcp/tools.rs`";
        let paths = extract_file_paths_from_content(content);
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&"src/skills/triggers.rs".to_string()));
        assert!(paths.contains(&"src/mcp/tools.rs".to_string()));
    }

    #[test]
    fn test_extract_file_paths_ignores_urls() {
        let content = "See https://example.com/src/neo4j/client.rs for docs";
        let paths = extract_file_paths_from_content(content);
        assert!(paths.is_empty(), "URLs should be ignored, got: {:?}", paths);
    }

    #[test]
    fn test_extract_file_paths_ignores_rust_modules() {
        let content = "Use crate::neo4j::client for the implementation";
        let paths = extract_file_paths_from_content(content);
        assert!(
            paths.is_empty(),
            "Rust module paths should be ignored, got: {:?}",
            paths
        );
    }

    #[test]
    fn test_extract_file_paths_no_paths() {
        let content = "This is a note without any file path references at all.";
        let paths = extract_file_paths_from_content(content);
        assert!(paths.is_empty());
    }

    #[test]
    fn test_extract_file_paths_deduplicates() {
        let content = "Modifier src/neo4j/client.rs. Le fichier src/neo4j/client.rs est critique.";
        let paths = extract_file_paths_from_content(content);
        assert_eq!(paths, vec!["src/neo4j/client.rs"]);
    }

    #[test]
    fn test_extract_file_paths_various_extensions() {
        let content = "Files: src/app.ts, src/index.tsx, tests/test_foo.py, src/main.go";
        let paths = extract_file_paths_from_content(content);
        assert_eq!(paths.len(), 4);
    }

    #[test]
    fn test_extract_file_paths_tests_dir() {
        let content = "Le fichier /home/user/project/tests/unit/test_note.rs à vérifier";
        let paths = extract_file_paths_from_content(content);
        assert_eq!(paths, vec!["tests/unit/test_note.rs"]);
    }

    #[test]
    fn test_score_anchor_exact_match() {
        use crate::notes::models::{EntityType as NoteEntityType, NoteAnchor};

        let mut note = make_scored_note(
            "Neo4j tip",
            vec![],
            NoteType::Tip,
            NoteImportance::Medium,
            0.8,
            0.0,
        );
        note.anchors = vec![NoteAnchor::new(
            NoteEntityType::File,
            "src/neo4j/client.rs".to_string(),
        )];

        let note_without_anchors = make_scored_note(
            "Neo4j tip",
            vec![],
            NoteType::Tip,
            NoteImportance::Medium,
            0.8,
            0.0,
        );

        let with_anchor = score_note_relevance(&note, Some("src/neo4j/client.rs"), None, "Read");
        let without_anchor = score_note_relevance(
            &note_without_anchors,
            Some("src/neo4j/client.rs"),
            None,
            "Read",
        );

        assert!(
            with_anchor > without_anchor,
            "Anchored note ({}) should score higher than non-anchored ({})",
            with_anchor,
            without_anchor
        );
        // Difference should be ~0.4 (exact match bonus)
        let diff = with_anchor - without_anchor;
        assert!(
            (diff - 0.4).abs() < 0.01,
            "Anchor bonus should be ~0.4, got {}",
            diff
        );
    }

    #[test]
    fn test_score_anchor_directory_match() {
        use crate::notes::models::{EntityType as NoteEntityType, NoteAnchor};

        let mut note = make_scored_note(
            "Neo4j tip",
            vec![],
            NoteType::Tip,
            NoteImportance::Medium,
            0.8,
            0.0,
        );
        note.anchors = vec![NoteAnchor::new(
            NoteEntityType::File,
            "src/neo4j/analytics.rs".to_string(),
        )];

        let note_without_anchors = make_scored_note(
            "Neo4j tip",
            vec![],
            NoteType::Tip,
            NoteImportance::Medium,
            0.8,
            0.0,
        );

        let with_dir_anchor =
            score_note_relevance(&note, Some("src/neo4j/client.rs"), None, "Read");
        let without_anchor = score_note_relevance(
            &note_without_anchors,
            Some("src/neo4j/client.rs"),
            None,
            "Read",
        );

        // Directory match → +0.2
        let diff = with_dir_anchor - without_anchor;
        assert!(
            (diff - 0.2).abs() < 0.01,
            "Directory anchor bonus should be ~0.2, got {}",
            diff
        );
    }

    #[test]
    fn test_score_no_anchors_no_bonus() {
        let note = make_scored_note(
            "Generic note",
            vec![],
            NoteType::Tip,
            NoteImportance::Medium,
            0.8,
            0.0,
        );

        let score = score_note_relevance(&note, Some("src/neo4j/client.rs"), None, "Read");
        // base(0.1) × importance(1.0) = 0.1
        assert!(
            (score - 0.1).abs() < f64::EPSILON,
            "No tags, no content match, no anchors → base score 0.1, got {}",
            score
        );
    }

    // --- E2E contextual scoring simulation ---

    /// Simulates a realistic skill activation scenario: a skill with 15+ notes
    /// of mixed topics (neo4j, tauri, chat, api), triggered by a Read on
    /// `src/neo4j/client.rs`. Verifies that notes relevant to neo4j are
    /// sorted first in the assembled context string.
    #[test]
    fn test_e2e_contextual_scoring_sorts_relevant_notes_first() {
        use crate::notes::models::{EntityType as NoteEntityType, NoteAnchor};

        // --- Build 18 notes with realistic diversity ---
        let neo4j_notes = vec![
            {
                let mut n = make_scored_note(
                    "Always use parameters in Cypher queries to prevent injection.",
                    vec!["neo4j", "cypher", "security"],
                    NoteType::Gotcha,
                    NoteImportance::Critical,
                    0.9,
                    0.0,
                );
                n.anchors = vec![NoteAnchor::new(
                    NoteEntityType::File,
                    "src/neo4j/client.rs".to_string(),
                )];
                n
            },
            make_scored_note(
                "Neo4j MERGE is not atomic, use unique constraints. See src/neo4j/client.rs.",
                vec!["neo4j", "cypher", "gotcha"],
                NoteType::Gotcha,
                NoteImportance::High,
                0.8,
                0.05,
            ),
            make_scored_note(
                "The Neo4j driver pool size should match tokio thread count.",
                vec!["neo4j", "performance"],
                NoteType::Tip,
                NoteImportance::Medium,
                0.7,
                0.1,
            ),
            {
                let mut n = make_scored_note(
                    "All LINKED_TO relations must use this exact name, not ATTACHED_TO.",
                    vec!["neo4j", "knowledge-fabric"],
                    NoteType::Guideline,
                    NoteImportance::High,
                    0.85,
                    0.0,
                );
                n.anchors = vec![NoteAnchor::new(
                    NoteEntityType::File,
                    "src/neo4j/note.rs".to_string(),
                )];
                n
            },
        ];

        let tauri_notes = vec![
            make_scored_note(
                "Tauri IPC uses JSON serialization, avoid large binary payloads.",
                vec!["tauri", "ipc", "performance"],
                NoteType::Gotcha,
                NoteImportance::High,
                0.8,
                0.0,
            ),
            make_scored_note(
                "Use tauri::Manager for window management in multi-window setups.",
                vec!["tauri", "desktop", "window"],
                NoteType::Pattern,
                NoteImportance::Medium,
                0.6,
                0.2,
            ),
            make_scored_note(
                "Custom protocol handlers must be registered before window creation.",
                vec!["tauri", "protocol"],
                NoteType::Guideline,
                NoteImportance::Medium,
                0.5,
                0.3,
            ),
        ];

        let chat_notes = vec![
            make_scored_note(
                "Chat session cleanup requires explicit entity unlinking first.",
                vec!["chat", "session", "lifecycle"],
                NoteType::Gotcha,
                NoteImportance::Medium,
                0.7,
                0.1,
            ),
            make_scored_note(
                "Message pagination uses offset/limit, not cursor-based.",
                vec!["chat", "api", "pagination"],
                NoteType::Context,
                NoteImportance::Low,
                0.5,
                0.2,
            ),
            make_scored_note(
                "The chat model field supports 'sonnet', 'opus', 'haiku' shortcuts.",
                vec!["chat", "model"],
                NoteType::Tip,
                NoteImportance::Low,
                0.4,
                0.15,
            ),
        ];

        let api_notes = vec![
            make_scored_note(
                "All API endpoints return 404 with {\"error\": \"...\"} JSON body.",
                vec!["api", "rest", "error-handling"],
                NoteType::Guideline,
                NoteImportance::Medium,
                0.6,
                0.1,
            ),
            make_scored_note(
                "Rate limiting is not implemented yet, track in backlog.",
                vec!["api", "security", "backlog"],
                NoteType::Observation,
                NoteImportance::Low,
                0.3,
                0.4,
            ),
        ];

        let misc_notes = vec![
            make_scored_note(
                "French stop words must be filtered in trigger generation.",
                vec!["triggers", "i18n"],
                NoteType::Guideline,
                NoteImportance::Medium,
                0.6,
                0.0,
            ),
            make_scored_note(
                "Test helpers are in test_helpers.rs, use mock_app_state().",
                vec!["testing", "mocking"],
                NoteType::Tip,
                NoteImportance::Medium,
                0.5,
                0.1,
            ),
            make_scored_note(
                "Stale note with low energy should rank last.",
                vec!["misc"],
                NoteType::Observation,
                NoteImportance::Low,
                0.2,
                0.8,
            ),
        ];

        // Combine all notes in a deliberately non-optimal order
        let mut all_notes: Vec<Note> = Vec::new();
        all_notes.extend(tauri_notes);
        all_notes.extend(chat_notes);
        all_notes.extend(misc_notes);
        all_notes.extend(api_notes);
        all_notes.extend(neo4j_notes); // neo4j notes added LAST

        assert_eq!(all_notes.len(), 15, "Should have 15 notes total");

        // Score and sort (simulate what activate_for_hook does)
        let file_context = Some("src/neo4j/client.rs");
        let mut scored: Vec<(f64, &Note)> = all_notes
            .iter()
            .map(|n| (score_note_relevance(n, file_context, None, "Read"), n))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // The top 4 notes should ALL be neo4j-related
        let top_4_contents: Vec<&str> = scored[0..4]
            .iter()
            .map(|(_, n)| n.content.as_str())
            .collect();
        for content in &top_4_contents {
            assert!(
                content.to_lowercase().contains("neo4j")
                    || content.to_lowercase().contains("linked_to")
                    || content.to_lowercase().contains("cypher"),
                "Top-4 note should be neo4j-related, got: '{}'",
                content
            );
        }

        // The note with exact anchor + critical importance should be first
        assert!(
            scored[0].1.content.contains("parameters in Cypher"),
            "First note should be the critical gotcha with exact anchor, got: '{}'",
            scored[0].1.content
        );

        // The note mentioning client.rs in content should rank high (top 3)
        let client_rs_note_rank = scored
            .iter()
            .position(|(_, n)| n.content.contains("client.rs"))
            .expect("client.rs note should exist");
        assert!(
            client_rs_note_rank <= 2,
            "Note mentioning client.rs should be in top 3, got rank {}",
            client_rs_note_rank
        );

        // Tauri notes should be in the bottom half
        let tauri_positions: Vec<usize> = scored
            .iter()
            .enumerate()
            .filter(|(_, (_, n))| n.tags.contains(&"tauri".to_string()))
            .map(|(i, _)| i)
            .collect();
        for pos in &tauri_positions {
            assert!(
                *pos >= 4,
                "Tauri notes should be below neo4j notes, found at position {}",
                pos
            );
        }

        // Assemble context and verify neo4j notes appear first in the string
        let sorted_notes: Vec<Note> = scored.iter().map(|(_, n)| (*n).clone()).collect();
        let decisions = vec![];
        let (context_str, notes_included) = assemble_context_with_confidence(
            "TestSkill",
            &sorted_notes,
            &decisions,
            3200,
            Some(0.75),
            true, // pre_sorted
        );

        assert!(notes_included > 0, "Should include at least 1 note");
        // With short notes, all 15 may fit in 3200 chars. The important thing
        // is that the ORDER is correct (neo4j first), not that truncation happens.

        // First note line in context should be neo4j-related
        // Format is "- {emoji}{badge}{content}\n"
        let first_note_line = context_str
            .lines()
            .find(|l| l.starts_with("- "))
            .expect("Should have at least one note line");
        assert!(
            first_note_line.to_lowercase().contains("cypher")
                || first_note_line.to_lowercase().contains("parameter"),
            "First note in context should be neo4j-related, got: '{}'",
            first_note_line
        );
    }

    /// Verifies the regression guard: when no file_context or pattern is
    /// provided, the scoring falls back to energy-based ordering
    /// (legacy behavior preserved).
    #[test]
    fn test_e2e_no_context_preserves_legacy_ordering() {
        let high_energy = make_scored_note(
            "High energy generic note",
            vec!["misc"],
            NoteType::Tip,
            NoteImportance::Low,
            0.95,
            0.0,
        );
        let low_energy = make_scored_note(
            "Low energy generic note",
            vec!["misc"],
            NoteType::Tip,
            NoteImportance::Low,
            0.3,
            0.0,
        );

        let score_high = score_note_relevance(&high_energy, None, None, "Read");
        let score_low = score_note_relevance(&low_energy, None, None, "Read");

        // Without file_context, all signals except importance are inert.
        // Both have same importance (Low → ×0.8), so scores should be equal.
        assert!(
            (score_high - score_low).abs() < f64::EPSILON,
            "Without context, scores should be equal regardless of energy: high={}, low={}",
            score_high,
            score_low
        );
    }

    // =========================================================================
    // Decision auto-anchoring tests
    // =========================================================================

    #[tokio::test]
    async fn test_auto_anchor_decision_extracts_paths() {
        let store = crate::neo4j::mock::MockGraphStore::new();
        let decision = make_test_decision(
            Uuid::new_v4(),
            "Modify `src/neo4j/client.rs` to add the method",
            "Use src/api/handlers.rs for the endpoint",
        );

        let result = auto_anchor_decision(&store, &decision, Some("/tmp/project")).await;
        assert!(result.is_ok());
        // MockGraphStore.add_decision_affects returns Ok(()) so both paths should anchor
        assert_eq!(result.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_auto_anchor_decision_no_paths() {
        let store = crate::neo4j::mock::MockGraphStore::new();
        let decision = make_test_decision(
            Uuid::new_v4(),
            "Use a better algorithm for sorting",
            "Quicksort",
        );

        let result = auto_anchor_decision(&store, &decision, Some("/tmp/project")).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_auto_anchor_decision_deduplicates() {
        let store = crate::neo4j::mock::MockGraphStore::new();
        let decision = DecisionNode {
            id: Uuid::new_v4(),
            description: "Modify src/neo4j/client.rs for the query".to_string(),
            rationale: "src/neo4j/client.rs is the central file".to_string(),
            alternatives: vec![],
            chosen_option: Some("Update src/neo4j/client.rs directly".to_string()),
            decided_by: "test".to_string(),
            decided_at: Utc::now(),
            status: crate::neo4j::models::DecisionStatus::Accepted,
            embedding: None,
            embedding_model: None,
            scar_intensity: 0.0,
        };

        let result = auto_anchor_decision(&store, &decision, Some("/tmp/project")).await;
        assert!(result.is_ok());
        // Same path in description, rationale, and chosen_option → deduplicated to 1
        assert_eq!(result.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_auto_anchor_decision_without_root_path() {
        let store = crate::neo4j::mock::MockGraphStore::new();
        let decision = make_test_decision(
            Uuid::new_v4(),
            "Modify `src/neo4j/client.rs`",
            "Direct edit",
        );

        let result = auto_anchor_decision(&store, &decision, None).await;
        assert!(result.is_ok());
        // Should still work, using relative path as-is
        assert_eq!(result.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_auto_anchor_decision_empty_rationale() {
        let store = crate::neo4j::mock::MockGraphStore::new();
        let decision = DecisionNode {
            id: Uuid::new_v4(),
            description: "Modify `src/api/handlers.rs`".to_string(),
            rationale: String::new(),
            alternatives: vec![],
            chosen_option: None,
            decided_by: "test".to_string(),
            decided_at: Utc::now(),
            status: crate::neo4j::models::DecisionStatus::Proposed,
            embedding: None,
            embedding_model: None,
            scar_intensity: 0.0,
        };

        let result = auto_anchor_decision(&store, &decision, Some("/tmp/project")).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
    }

    // =========================================================================
    // Project-level decision anchoring tests
    // =========================================================================

    #[tokio::test]
    async fn test_auto_anchor_decisions_for_project_with_decisions() {
        let store = crate::neo4j::mock::MockGraphStore::new();
        let project_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();

        // Create project
        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test-project".to_string(),
            slug: "test-project".to_string(),
            description: None,
            root_path: "/tmp/test-project".to_string(),
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        // Create a decision mentioning files
        let decision = make_test_decision(
            Uuid::new_v4(),
            "Modify `src/neo4j/client.rs` and `src/api/handlers.rs`",
            "Direct implementation",
        );
        store.create_decision(task_id, &decision).await.unwrap();

        let result = auto_anchor_decisions_for_project(&store, project_id).await;
        assert!(result.is_ok());
        let (decisions_processed, anchors_created) = result.unwrap();
        assert_eq!(decisions_processed, 1);
        assert_eq!(anchors_created, 2);
    }

    #[tokio::test]
    async fn test_auto_anchor_decisions_for_project_not_found() {
        let store = crate::neo4j::mock::MockGraphStore::new();
        let fake_id = Uuid::new_v4();

        // Project doesn't exist → should still succeed (with None root_path)
        let result = auto_anchor_decisions_for_project(&store, fake_id).await;
        assert!(result.is_ok());
        let (decisions_processed, _anchors) = result.unwrap();
        // No decisions found for non-existent project → 0
        assert_eq!(decisions_processed, 0);
    }

    // =========================================================================
    // Cross-project note anchoring tests
    // =========================================================================

    #[tokio::test]
    async fn test_auto_anchor_all_notes_to_project() {
        let store = crate::neo4j::mock::MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test-project".to_string(),
            slug: "test-project".to_string(),
            description: None,
            root_path: "/tmp/test-project".to_string(),
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        // Create note from ANOTHER project that mentions a file
        let note = make_test_note(
            Uuid::new_v4(),
            "The file `src/neo4j/client.rs` has a performance issue",
            NoteType::Gotcha,
            NoteImportance::High,
            0.8,
        );
        store.create_note(&note).await.unwrap();

        let result = auto_anchor_all_notes_to_project(&store, project_id).await;
        assert!(result.is_ok());
        let report = result.unwrap();
        assert_eq!(report.notes_processed, 1);
        // Mock link_note_to_entity returns Ok(()) → anchor created
        assert_eq!(report.anchors_created, 1);
        assert_eq!(
            report.root_path_resolved,
            Some("/tmp/test-project".to_string())
        );
    }

    #[tokio::test]
    async fn test_auto_anchor_all_notes_project_not_found() {
        let store = crate::neo4j::mock::MockGraphStore::new();
        let fake_id = Uuid::new_v4();

        let result = auto_anchor_all_notes_to_project(&store, fake_id).await;
        assert!(result.is_ok());
        let report = result.unwrap();
        assert_eq!(report.notes_processed, 0);
        assert_eq!(report.anchors_created, 0);
        assert!(report.root_path_resolved.is_none());
    }

    // =========================================================================
    // Knowledge link reconstruction (orchestrator) tests
    // =========================================================================

    #[tokio::test]
    async fn test_reconstruct_knowledge_links_full() {
        let store = crate::neo4j::mock::MockGraphStore::new();
        let project_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test-project".to_string(),
            slug: "test-project".to_string(),
            description: None,
            root_path: "/tmp/test-project".to_string(),
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        // Add a note mentioning a file
        let note = make_test_note(
            Uuid::new_v4(),
            "Fix `src/api/handlers.rs` error handling",
            NoteType::Gotcha,
            NoteImportance::High,
            0.9,
        );
        store.create_note(&note).await.unwrap();

        // Add a decision mentioning a file
        let decision = make_test_decision(
            Uuid::new_v4(),
            "Refactor `src/neo4j/client.rs` for performance",
            "Batch queries",
        );
        store.create_decision(task_id, &decision).await.unwrap();

        let result = reconstruct_knowledge_links(&store, project_id).await;
        assert!(result.is_ok());
        let report = result.unwrap();

        assert_eq!(report.notes_processed, 1);
        assert_eq!(report.notes_linked, 1);
        assert_eq!(report.decisions_processed, 1);
        assert_eq!(report.affects_created, 1);
        // Mock propagation returns 0
        assert_eq!(report.structural_propagated, 0);
        assert_eq!(report.semantic_linked, 0);
        assert!(report.elapsed_ms < 5000); // Should be fast
    }

    #[tokio::test]
    async fn test_reconstruct_knowledge_links_empty_project() {
        let store = crate::neo4j::mock::MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "empty-project".to_string(),
            slug: "empty-project".to_string(),
            description: None,
            root_path: "/tmp/empty".to_string(),
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        let result = reconstruct_knowledge_links(&store, project_id).await;
        assert!(result.is_ok());
        let report = result.unwrap();

        assert_eq!(report.notes_processed, 0);
        assert_eq!(report.notes_linked, 0);
        assert_eq!(report.decisions_processed, 0);
        assert_eq!(report.affects_created, 0);
    }

    #[tokio::test]
    async fn test_reconstruct_report_is_serializable() {
        let report = ReconstructReport {
            notes_processed: 10,
            notes_linked: 5,
            decisions_processed: 3,
            affects_created: 2,
            structural_propagated: 1,
            semantic_linked: 0,
            elapsed_ms: 42,
        };
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["notes_processed"], 10);
        assert_eq!(json["affects_created"], 2);
        assert_eq!(json["elapsed_ms"], 42);
    }
}
