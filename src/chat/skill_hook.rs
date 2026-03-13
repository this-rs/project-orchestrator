//! SkillActivationHook — in-process PreToolUse hook for Neural Skills
//!
//! Intercepts every PreToolUse event from the Claude Code CLI, resolves the
//! project from the tool context (file path or cwd), activates matching skills,
//! and returns the skill context as `additionalContext` in the hook response.
//!
//! This replaces the old external hook flow (.po-config + pre-tool-use.cjs)
//! with a zero-config, in-process solution that is:
//! - ~100x faster (~1-5ms vs ~300ms HTTP round-trip)
//! - Zero-config (no .po-config, no settings.json, no hook scripts)
//! - Lock-free (runs inside the SDK's hook dispatch, no Mutex contention)
//!
//! # Error Policy
//!
//! **NEVER block tool execution.** Any error → passthrough (continue_: true).
//! Skills are a nice-to-have context injection, not a gating mechanism.

use crate::api::hook_handlers::skill_cache;
use crate::neo4j::traits::GraphStore;
use crate::neurons::AutoReinforcementConfig;
use crate::skills::activation::{
    activate_for_hook_cached, spawn_activation_increment, spawn_hook_reinforcement,
    HookActivationConfig,
};
use crate::skills::hook_extractor::{
    enrich_redirect_with_context_card, extract_file_context, generate_redirect_suggestion,
};
use crate::skills::project_resolver::resolve_project_from_context;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// In-memory cache entry for persona file index.
/// Maps file_path → Vec<(persona_id, persona_name, weight)>.
struct PersonaFileIndex {
    entries: HashMap<String, Vec<(Uuid, String, f64)>>,
    loaded_at: std::time::Instant,
}

/// TTL for persona file index cache (2 minutes).
const PERSONA_INDEX_TTL: std::time::Duration = std::time::Duration::from_secs(120);

/// In-process hook callback that activates Neural Skills on every PreToolUse event.
///
/// Registered in `create_session()` and `resume_session()` alongside `CompactionNotifier`.
/// Uses the same `SkillCache` and activation pipeline as the REST endpoint.
/// Also matches file access against persona KNOWS relations for persona context injection.
pub(crate) struct SkillActivationHook {
    /// Access to Neo4j graph store for project resolution + skill loading.
    graph_store: Arc<dyn GraphStore>,
    /// Per-project persona file index cache: project_id → PersonaFileIndex.
    persona_index: RwLock<HashMap<Uuid, PersonaFileIndex>>,
}

impl SkillActivationHook {
    pub fn new(graph_store: Arc<dyn GraphStore>) -> Self {
        Self {
            graph_store,
            persona_index: RwLock::new(HashMap::new()),
        }
    }

    /// Passthrough response — continue without injecting context.
    fn passthrough() -> nexus_claude::HookJSONOutput {
        nexus_claude::HookJSONOutput::Sync(nexus_claude::SyncHookJSONOutput {
            continue_: Some(true),
            ..Default::default()
        })
    }

    /// Match a file path against the persona file index.
    /// Returns the best matching persona (highest weight) if found.
    async fn match_persona_for_file(
        &self,
        project_id: Uuid,
        file_path: &str,
    ) -> Option<(Uuid, String, f64)> {
        // Check cache first (read lock)
        {
            let cache = self.persona_index.read().await;
            if let Some(index) = cache.get(&project_id) {
                if index.loaded_at.elapsed() < PERSONA_INDEX_TTL {
                    return index
                        .entries
                        .get(file_path)
                        .and_then(|v| v.first().cloned());
                }
            }
        }

        // Cache miss or expired — load from Neo4j
        match self
            .graph_store
            .find_personas_for_file(file_path, project_id)
            .await
        {
            Ok(matches) if !matches.is_empty() => {
                let first = matches.first().map(|(p, w)| (p.id, p.name.clone(), *w));

                // Populate cache (write lock) — just for this file, not full index
                // Full index load would be done on first access per project
                {
                    let mut cache = self.persona_index.write().await;
                    let index = cache.entry(project_id).or_insert_with(|| PersonaFileIndex {
                        entries: HashMap::new(),
                        loaded_at: std::time::Instant::now(),
                    });
                    index.entries.insert(
                        file_path.to_string(),
                        matches
                            .iter()
                            .map(|(p, w)| (p.id, p.name.clone(), *w))
                            .collect(),
                    );
                }

                first
            }
            Ok(_) => None,
            Err(e) => {
                debug!(
                    file_path = %file_path,
                    error = %e,
                    "Persona file match failed"
                );
                None
            }
        }
    }

    /// Build persona context string for injection into additionalContext.
    async fn build_persona_context(
        &self,
        persona_id: Uuid,
        persona_name: &str,
        weight: f64,
    ) -> Option<String> {
        let subgraph = match self.graph_store.get_persona_subgraph(persona_id).await {
            Ok(sg) => sg,
            Err(e) => {
                debug!(
                    persona_id = %persona_id,
                    error = %e,
                    "Failed to load persona subgraph for hook"
                );
                return None;
            }
        };

        let mut ctx = format!(
            "## 🎭 Persona: {} (energy: {:.2}, weight: {:.2})\n",
            persona_name,
            // We don't have energy in subgraph, use a placeholder from stats
            subgraph.stats.freshness,
            weight
        );

        // Top notes (by weight, max 5)
        if !subgraph.notes.is_empty() {
            ctx.push_str("**Knowledge notes:**\n");
            for rel in subgraph.notes.iter().take(5) {
                ctx.push_str(&format!(
                    "- [note:{}] (w:{:.2})\n",
                    rel.entity_id, rel.weight
                ));
            }
        }

        // Top files (by weight, max 5)
        if !subgraph.files.is_empty() {
            ctx.push_str("**Known files:**\n");
            for rel in subgraph.files.iter().take(5) {
                ctx.push_str(&format!("- `{}` (w:{:.2})\n", rel.entity_id, rel.weight));
            }
        }

        // Truncate to ~2000 chars for additionalContext budget
        if ctx.len() > 2000 {
            ctx.truncate(1996);
            ctx.push_str("...\n");
        }

        Some(ctx)
    }
}

#[async_trait::async_trait]
impl nexus_claude::HookCallback for SkillActivationHook {
    async fn execute(
        &self,
        input: &nexus_claude::HookInput,
        _tool_use_id: Option<&str>,
        _context: &nexus_claude::HookContext,
    ) -> std::result::Result<nexus_claude::HookJSONOutput, nexus_claude::SdkError> {
        // 1. Only process PreToolUse events — passthrough everything else
        let pre_tool = match input {
            nexus_claude::HookInput::PreToolUse(pre) => pre,
            _ => return Ok(Self::passthrough()),
        };

        // 2. Resolve project from tool context (file path → project, or cwd → project)
        let project_id = match resolve_project_from_context(
            &*self.graph_store,
            &pre_tool.tool_name,
            &pre_tool.tool_input,
            &pre_tool.cwd,
        )
        .await
        {
            Ok(Some(id)) => id,
            Ok(None) => {
                debug!(
                    tool = %pre_tool.tool_name,
                    cwd = %pre_tool.cwd,
                    "Skill hook: no project matched"
                );
                return Ok(Self::passthrough());
            }
            Err(e) => {
                warn!(
                    tool = %pre_tool.tool_name,
                    "Skill hook: project resolution failed: {}", e
                );
                return Ok(Self::passthrough());
            }
        };

        // 3. Activate skills via the cached pipeline (same as REST endpoint)
        let config = HookActivationConfig::default();
        let cache = skill_cache();
        let skill_outcome = activate_for_hook_cached(
            &*self.graph_store,
            project_id,
            &pre_tool.tool_name,
            &pre_tool.tool_input,
            &config,
            cache,
        )
        .await;

        // 3b. Persona matching: on Read/Edit/Write, match file against persona KNOWS
        let file_path = extract_file_context(&pre_tool.tool_name, &pre_tool.tool_input);
        let persona_context = if let Some(ref fp) = file_path {
            match self.match_persona_for_file(project_id, fp).await {
                Some((pid, pname, weight)) => {
                    info!(
                        persona_name = %pname,
                        weight = weight,
                        file_path = %fp,
                        "Persona matched via file access"
                    );
                    self.build_persona_context(pid, &pname, weight).await
                }
                None => None,
            }
        } else {
            None
        };

        // If neither skill nor persona matched → passthrough
        let outcome = match skill_outcome {
            Ok(Some(outcome)) => Some(outcome),
            Ok(None) => {
                if persona_context.is_none() {
                    debug!(
                        tool = %pre_tool.tool_name,
                        "Skill hook: no skill or persona matched"
                    );
                    return Ok(Self::passthrough());
                }
                None
            }
            Err(e) => {
                warn!(
                    tool = %pre_tool.tool_name,
                    "Skill hook: activation failed: {}", e
                );
                if persona_context.is_none() {
                    return Ok(Self::passthrough());
                }
                None
            }
        };

        // 4. Hebbian reinforcement (fire-and-forget, async background)
        if let Some(ref outcome) = outcome {
            let reinforcement_config = AutoReinforcementConfig::default();
            spawn_hook_reinforcement(
                self.graph_store.clone(),
                outcome.activated_note_ids.clone(),
                reinforcement_config,
            );

            // 4b. Increment activation_count (fire-and-forget)
            spawn_activation_increment(self.graph_store.clone(), outcome.response.skill_id);
        }

        // 5. Build combined context: skill context + persona context + redirect suggestion
        let mut combined_context = outcome
            .as_ref()
            .map(|o| o.response.context.clone())
            .unwrap_or_default();

        // 5a. Inject persona context (after skill context)
        if let Some(ref pc) = persona_context {
            if !combined_context.is_empty() {
                combined_context.push_str("\n\n");
            }
            combined_context.push_str(pc);
        }

        // 5b. Generate redirect suggestion for Grep/Bash tools
        if let Some(suggestion) =
            generate_redirect_suggestion(&pre_tool.tool_name, &pre_tool.tool_input)
        {
            // Try to enrich with ContextCard (best-effort, don't block on failure)
            let redirect_fp = extract_file_context(&pre_tool.tool_name, &pre_tool.tool_input);
            let enriched = if let Some(ref fp) = redirect_fp {
                let project_id_str = project_id.to_string();
                match self.graph_store.get_context_card(fp, &project_id_str).await {
                    Ok(Some(card)) => enrich_redirect_with_context_card(suggestion, &card),
                    _ => crate::skills::hook_extractor::EnrichedRedirectSuggestion {
                        suggestion,
                        context_warnings: vec![],
                    },
                }
            } else {
                crate::skills::hook_extractor::EnrichedRedirectSuggestion {
                    suggestion,
                    context_warnings: vec![],
                }
            };

            // Append redirect suggestion to context (never overwrite skill context)
            combined_context.push_str("\n\n");
            combined_context.push_str(&enriched.to_string());

            debug!(
                tool = %pre_tool.tool_name,
                mcp_tool = %enriched.suggestion.mcp_tool,
                "Redirect suggestion injected"
            );
        }

        let skill_name = outcome
            .as_ref()
            .map(|o| o.response.skill_name.as_str())
            .unwrap_or("none");
        let confidence = outcome
            .as_ref()
            .map(|o| o.response.confidence)
            .unwrap_or(0.0);
        let has_persona = persona_context.is_some();

        info!(
            skill_name = %skill_name,
            confidence = confidence,
            has_persona = has_persona,
            tool = %pre_tool.tool_name,
            "Hook activated via PreToolUse"
        );

        Ok(nexus_claude::HookJSONOutput::Sync(
            nexus_claude::SyncHookJSONOutput {
                continue_: Some(true),
                hook_specific_output: Some(nexus_claude::HookSpecificOutput::PreToolUse(
                    nexus_claude::PreToolUseHookSpecificOutput {
                        permission_decision: None,
                        permission_decision_reason: None,
                        updated_input: None,
                        additional_context: Some(combined_context),
                    },
                )),
                ..Default::default()
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use nexus_claude::{HookCallback, HookContext, HookInput};

    /// Helper to create a minimal HookContext for tests
    fn test_context() -> HookContext {
        HookContext { signal: None }
    }

    #[tokio::test]
    async fn test_non_pre_tool_use_input_returns_passthrough() {
        let mock_store = Arc::new(MockGraphStore::new());
        let hook = SkillActivationHook::new(mock_store);

        // PreCompact is not PreToolUse → should passthrough
        let input = HookInput::PreCompact(nexus_claude::PreCompactHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/transcript".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            trigger: "context_full".to_string(),
            custom_instructions: None,
        });

        let result = hook.execute(&input, None, &test_context()).await.unwrap();
        match result {
            nexus_claude::HookJSONOutput::Sync(sync) => {
                assert_eq!(sync.continue_, Some(true));
                assert!(sync.hook_specific_output.is_none());
            }
            _ => panic!("Expected Sync output"),
        }
    }

    #[tokio::test]
    async fn test_pre_tool_use_no_project_match_returns_passthrough() {
        let mock_store = Arc::new(MockGraphStore::new());
        let hook = SkillActivationHook::new(mock_store);

        let input = HookInput::PreToolUse(nexus_claude::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/transcript".to_string(),
            cwd: "/completely/unknown/path".to_string(),
            permission_mode: None,
            tool_name: "Read".to_string(),
            tool_input: serde_json::json!({"file_path": "/completely/unknown/file.rs"}),
        });

        let result = hook.execute(&input, None, &test_context()).await.unwrap();
        match result {
            nexus_claude::HookJSONOutput::Sync(sync) => {
                assert_eq!(sync.continue_, Some(true));
                // No project matched → no additional context
                assert!(sync.hook_specific_output.is_none());
            }
            _ => panic!("Expected Sync output"),
        }
    }

    #[test]
    fn test_passthrough_has_continue_true() {
        let output = SkillActivationHook::passthrough();
        match output {
            nexus_claude::HookJSONOutput::Sync(sync) => {
                assert_eq!(sync.continue_, Some(true));
                assert!(sync.hook_specific_output.is_none());
                assert!(sync.reason.is_none());
            }
            _ => panic!("Expected Sync output"),
        }
    }
}
