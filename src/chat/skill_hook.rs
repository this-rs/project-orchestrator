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
    activate_for_hook_cached, spawn_hook_reinforcement, HookActivationConfig,
};
use crate::skills::project_resolver::resolve_project_from_context;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// In-process hook callback that activates Neural Skills on every PreToolUse event.
///
/// Registered in `create_session()` and `resume_session()` alongside `CompactionNotifier`.
/// Uses the same `SkillCache` and activation pipeline as the REST endpoint.
pub(crate) struct SkillActivationHook {
    /// Access to Neo4j graph store for project resolution + skill loading.
    graph_store: Arc<dyn GraphStore>,
}

impl SkillActivationHook {
    pub fn new(graph_store: Arc<dyn GraphStore>) -> Self {
        Self { graph_store }
    }

    /// Passthrough response — continue without injecting context.
    fn passthrough() -> nexus_claude::HookJSONOutput {
        nexus_claude::HookJSONOutput::Sync(nexus_claude::SyncHookJSONOutput {
            continue_: Some(true),
            ..Default::default()
        })
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
        let outcome = match activate_for_hook_cached(
            &*self.graph_store,
            project_id,
            &pre_tool.tool_name,
            &pre_tool.tool_input,
            &config,
            cache,
        )
        .await
        {
            Ok(Some(outcome)) => outcome,
            Ok(None) => {
                debug!(
                    tool = %pre_tool.tool_name,
                    "Skill hook: no skill matched"
                );
                return Ok(Self::passthrough());
            }
            Err(e) => {
                warn!(
                    tool = %pre_tool.tool_name,
                    "Skill hook: activation failed: {}", e
                );
                return Ok(Self::passthrough());
            }
        };

        // 4. Hebbian reinforcement (fire-and-forget, async background)
        let reinforcement_config = AutoReinforcementConfig::default();
        spawn_hook_reinforcement(
            self.graph_store.clone(),
            outcome.activated_note_ids,
            reinforcement_config,
        );

        // 5. Return skill context as additionalContext
        info!(
            skill_name = %outcome.response.skill_name,
            confidence = outcome.response.confidence,
            tool = %pre_tool.tool_name,
            "Skill activated via PreToolUse hook"
        );

        Ok(nexus_claude::HookJSONOutput::Sync(
            nexus_claude::SyncHookJSONOutput {
                continue_: Some(true),
                hook_specific_output: Some(nexus_claude::HookSpecificOutput::PreToolUse(
                    nexus_claude::PreToolUseHookSpecificOutput {
                        permission_decision: None,
                        permission_decision_reason: None,
                        updated_input: None,
                        additional_context: Some(outcome.response.context),
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
