//! AgentExecutor — executes business protocol actions via Claude agent.
//!
//! Two execution paths:
//! - **Fast path**: if the action is parseable as an MCP-style call (e.g., `admin(...)`),
//!   delegates to the [`SystemExecutor`] logic (no LLM, no cost).
//! - **Slow path**: spawns a Claude agent session via [`ChatManager`], sends the
//!   composed prompt, streams the response, and extracts the trigger to fire.
//!
//! Trigger extraction convention:
//! - The agent outputs `TRIGGER: <trigger_name>` at the end of its response, OR
//! - The agent calls `protocol(action: "transition", ...)` which is intercepted.
//!
//! If neither is found, falls back to outgoing transition heuristics.

use super::system::SystemExecutor;
use super::{ExecutionResult, Executor};
use crate::chat::manager::ChatManager;
use crate::chat::types::{ChatEvent, ChatRequest};
use crate::neo4j::traits::GraphStore;
use crate::protocol::{Protocol, ProtocolRun, ProtocolState, ProtocolTransition};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

/// Default timeout for agent sessions (30 minutes).
const DEFAULT_AGENT_TIMEOUT_SECS: u64 = 1800;

/// Executes business protocol state actions.
///
/// Fast path: MCP-style actions are delegated to [`SystemExecutor`] (no LLM).
/// Slow path: spawns a Claude agent session, sends the prompt, and extracts the trigger.
pub struct AgentExecutor {
    system: SystemExecutor,
    /// ChatManager for spawning Claude agent sessions (slow path).
    /// None = fallback to stub behavior (V1 compatibility / tests).
    chat_manager: Option<Arc<ChatManager>>,
    /// Cancellation token for graceful shutdown.
    cancel: CancellationToken,
}

impl AgentExecutor {
    /// Create a new AgentExecutor with ChatManager for LLM execution.
    pub fn new(chat_manager: Option<Arc<ChatManager>>, cancel: CancellationToken) -> Self {
        Self {
            system: SystemExecutor::new(),
            chat_manager,
            cancel,
        }
    }

    /// Create a stub AgentExecutor without ChatManager (for tests / V1 compat).
    pub fn new_stub() -> Self {
        Self {
            system: SystemExecutor::new(),
            chat_manager: None,
            cancel: CancellationToken::new(),
        }
    }
}

impl Default for AgentExecutor {
    fn default() -> Self {
        Self::new_stub()
    }
}

/// Check if an action string looks like an MCP-style call: `tool(method, ...)`.
fn is_mcp_style(action: &str) -> bool {
    let trimmed = action.trim();
    // Check for pattern: word(...)  or  word(...) + word(...)
    trimmed.contains('(') && trimmed.contains(')')
}

/// Build the agent prompt from state context, run context, and outgoing transitions.
///
/// The prompt is structured as:
/// 1. Role and context (protocol name, current state, run history)
/// 2. State action description + prompt_fragment
/// 3. Available tools whitelist / forbidden actions
/// 4. Possible triggers (outgoing transitions) for the agent to choose from
/// 5. Output convention: end with `TRIGGER: <name>`
fn build_agent_prompt(
    state: &ProtocolState,
    run: &ProtocolRun,
    protocol: &Protocol,
    transitions: &[ProtocolTransition],
    project_root: Option<&str>,
) -> String {
    let mut prompt = String::with_capacity(2048);

    // Section 1: Context
    prompt.push_str(&format!(
        "# Protocol Execution — {} / State: {}\n\n",
        protocol.name, state.name
    ));

    if !protocol.description.is_empty() {
        prompt.push_str(&format!(
            "**Protocol purpose:** {}\n\n",
            protocol.description
        ));
    }

    if !state.description.is_empty() {
        prompt.push_str(&format!("**State purpose:** {}\n\n", state.description));
    }

    // Run context
    prompt.push_str("## Run Context\n\n");
    prompt.push_str(&format!("- **Run ID:** {}\n", run.id));
    if let Some(plan_id) = run.plan_id {
        prompt.push_str(&format!("- **Plan ID:** {}\n", plan_id));
    }
    if let Some(task_id) = run.task_id {
        prompt.push_str(&format!("- **Task ID:** {}\n", task_id));
    }
    if let Some(ref root) = project_root {
        prompt.push_str(&format!("- **Project root:** {}\n", root));
    }

    // States visited history
    if !run.states_visited.is_empty() {
        prompt.push_str("\n**States visited so far:**\n");
        for visit in &run.states_visited {
            let trigger_info = visit
                .trigger
                .as_deref()
                .map(|t| format!(" (trigger: `{}`)", t))
                .unwrap_or_default();
            prompt.push_str(&format!("- {}{}\n", visit.state_name, trigger_info));
        }
        prompt.push('\n');
    }

    // Section 2: Action
    prompt.push_str("## Action Required\n\n");
    if let Some(ref action) = state.action {
        prompt.push_str(&format!("{}\n\n", action));
    }

    // Prompt fragment (state-specific instructions)
    if let Some(ref fragment) = state.prompt_fragment {
        prompt.push_str(&format!("### Additional Instructions\n\n{}\n\n", fragment));
    }

    // Section 3: Tool constraints
    if let Some(ref tools) = state.available_tools {
        if !tools.is_empty() {
            prompt.push_str("## Allowed Tools\n\n");
            prompt.push_str("You may ONLY use the following tools in this state:\n");
            for tool in tools {
                prompt.push_str(&format!("- `{}`\n", tool));
            }
            prompt.push('\n');
        }
    }

    if let Some(ref forbidden) = state.forbidden_actions {
        if !forbidden.is_empty() {
            prompt.push_str("## Forbidden Actions\n\n");
            prompt.push_str("You MUST NOT perform any of the following:\n");
            for action in forbidden {
                prompt.push_str(&format!("- {}\n", action));
            }
            prompt.push('\n');
        }
    }

    // Section 4: Possible triggers
    let outgoing: Vec<_> = transitions
        .iter()
        .filter(|t| t.from_state == state.id)
        .collect();

    if !outgoing.is_empty() {
        prompt.push_str("## Possible Triggers\n\n");
        prompt.push_str(
            "When you have completed the action, you MUST indicate which trigger to fire.\n",
        );
        prompt.push_str("Choose ONE of the following:\n\n");
        for t in &outgoing {
            let guard_info = t
                .guard
                .as_deref()
                .map(|g| format!(" *(guard: {})*", g))
                .unwrap_or_default();
            prompt.push_str(&format!("- `{}`{}\n", t.trigger, guard_info));
        }
        prompt.push('\n');
    }

    // Section 5: Output convention
    prompt.push_str("## Output Convention\n\n");
    prompt.push_str(
        "When you are done, end your response with exactly:\n\n```\nTRIGGER: <trigger_name>\n```\n\n",
    );
    prompt.push_str("where `<trigger_name>` is one of the triggers listed above.\n");

    prompt
}

/// Extract a trigger from the agent's response text.
///
/// Looks for:
/// 1. `TRIGGER: <name>` pattern (case-insensitive)
/// 2. `protocol(action: "transition", ... trigger: "<name>")` MCP call
///
/// Returns None if no trigger is found.
fn extract_trigger_from_text(text: &str) -> Option<String> {
    // Pattern 1: TRIGGER: <name> (last occurrence wins)
    for line in text.lines().rev() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed
            .strip_prefix("TRIGGER:")
            .or_else(|| trimmed.strip_prefix("trigger:"))
            .or_else(|| trimmed.strip_prefix("Trigger:"))
        {
            let trigger = rest.trim().trim_matches('`').trim();
            if !trigger.is_empty() {
                return Some(trigger.to_string());
            }
        }
    }

    // Pattern 2: protocol(action: "transition" ... trigger: "xxx")
    // Simple regex-free extraction
    if let Some(pos) = text.rfind("\"transition\"") {
        let after = &text[pos..];
        // Look for trigger field
        if let Some(trig_pos) = after.find("\"trigger\"") {
            let after_trigger = &after[trig_pos + 9..]; // skip "trigger"
                                                        // Find the value: skip whitespace, colon, whitespace, quote
            let value_start = after_trigger.find('"');
            if let Some(vs) = value_start {
                let rest = &after_trigger[vs + 1..];
                if let Some(ve) = rest.find('"') {
                    let trigger = &rest[..ve];
                    if !trigger.is_empty() {
                        return Some(trigger.to_string());
                    }
                }
            }
        }
    }

    None
}

/// Extract trigger from a ToolUse event (protocol transition call intercepted).
fn extract_trigger_from_tool_use(tool: &str, input: &serde_json::Value) -> Option<String> {
    if tool != "mcp__project-orchestrator__protocol" {
        return None;
    }
    let action = input.get("action").and_then(|v| v.as_str())?;
    if action != "transition" {
        return None;
    }
    let trigger = input.get("trigger").and_then(|v| v.as_str())?;
    Some(trigger.to_string())
}

/// Determine the trigger from outgoing transitions (fallback when agent doesn't specify one).
fn determine_trigger_from_transitions(
    transitions: &[ProtocolTransition],
    state: &ProtocolState,
) -> String {
    let outgoing: Vec<_> = transitions
        .iter()
        .filter(|t| t.from_state == state.id)
        .collect();

    match outgoing.len() {
        0 => "done".to_string(),
        1 => outgoing[0].trigger.clone(),
        _ => {
            // Prefer "done" or "success" triggers
            if let Some(t) = outgoing.iter().find(|t| t.trigger == "done") {
                t.trigger.clone()
            } else if let Some(t) = outgoing.iter().find(|t| t.trigger == "success") {
                t.trigger.clone()
            } else {
                outgoing[0].trigger.clone()
            }
        }
    }
}

/// Execute the slow path: spawn a Claude agent session, send the prompt, monitor events,
/// and extract the trigger from the agent's response.
#[allow(clippy::too_many_arguments)]
async fn execute_via_agent(
    chat_manager: &ChatManager,
    prompt: String,
    run: &ProtocolRun,
    protocol: &Protocol,
    state: &ProtocolState,
    transitions: &[ProtocolTransition],
    project_root: Option<&str>,
    cancel: &CancellationToken,
) -> Result<ExecutionResult> {
    // 1. Create a chat session
    let cwd = project_root.unwrap_or(".");
    let request = ChatRequest {
        message: String::new(), // sent separately via send_message
        session_id: None,
        cwd: cwd.to_string(),
        project_slug: None,
        model: None,
        permission_mode: Some("bypassPermissions".to_string()),
        add_dirs: None,
        workspace_slug: None,
        user_claims: None,
        spawned_by: Some(
            serde_json::json!({
                "type": "protocol_runner",
                "run_id": run.id.to_string(),
                "protocol_id": protocol.id.to_string(),
                "state_name": state.name,
            })
            .to_string(),
        ),
        task_context: Some(format!(
            "Protocol '{}' — executing state '{}'",
            protocol.name, state.name
        )),
        scaffolding_override: None,
    };

    let session = chat_manager
        .create_session(&request)
        .await
        .context("Failed to create agent session for protocol state")?;

    let session_id = session.session_id.clone();

    info!(
        run_id = %run.id,
        state_name = %state.name,
        session_id = %session_id,
        "AgentExecutor: spawned Claude session"
    );

    // 2. Subscribe to events before sending the message
    let mut event_rx = chat_manager
        .subscribe(&session_id)
        .await
        .context("Failed to subscribe to agent session events")?;

    // 3. Send the prompt
    chat_manager
        .send_message(&session_id, &prompt)
        .await
        .context("Failed to send prompt to agent session")?;

    // 4. Monitor events and collect response
    let timeout_secs = state
        .state_timeout_secs
        .unwrap_or(DEFAULT_AGENT_TIMEOUT_SECS);
    let timeout = std::time::Duration::from_secs(timeout_secs);
    let start = std::time::Instant::now();

    let mut accumulated_text = String::new();
    let mut detected_trigger: Option<String> = None;
    let mut notes = Vec::new();
    let mut session_cost_usd: Option<f64> = None;

    loop {
        let remaining = timeout.saturating_sub(start.elapsed());
        if remaining.is_zero() {
            warn!(
                run_id = %run.id,
                state_name = %state.name,
                "AgentExecutor: session timed out after {}s",
                timeout_secs
            );
            notes.push(format!("Agent session timed out after {}s", timeout_secs));
            break;
        }

        let event = tokio::select! {
            _ = cancel.cancelled() => {
                info!(
                    run_id = %run.id,
                    state_name = %state.name,
                    "AgentExecutor: cancelled — closing session"
                );
                notes.push("Agent session cancelled".to_string());
                break;
            }
            result = tokio::time::timeout(remaining, event_rx.recv()) => {
                match result {
                    Ok(Ok(event)) => event,
                    Ok(Err(tokio::sync::broadcast::error::RecvError::Closed)) => {
                        debug!(run_id = %run.id, "AgentExecutor: event channel closed");
                        break;
                    }
                    Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(n))) => {
                        warn!(run_id = %run.id, "AgentExecutor: lagged {n} events");
                        continue;
                    }
                    Err(_) => {
                        warn!(run_id = %run.id, "AgentExecutor: timed out waiting for events");
                        notes.push(format!("Agent session timed out after {}s", timeout_secs));
                        break;
                    }
                }
            }
        };

        match event {
            ChatEvent::AssistantText { content, .. } => {
                accumulated_text.push_str(&content);
            }
            ChatEvent::ToolUse {
                ref tool,
                ref input,
                ..
            } => {
                // Check if the agent is calling protocol(transition) — intercept
                if let Some(trigger) = extract_trigger_from_tool_use(tool, input) {
                    info!(
                        run_id = %run.id,
                        state_name = %state.name,
                        trigger = %trigger,
                        "AgentExecutor: intercepted protocol transition call"
                    );
                    detected_trigger = Some(trigger);
                }
            }
            ChatEvent::Result {
                cost_usd,
                duration_ms,
                subtype,
                ..
            } => {
                session_cost_usd = cost_usd;
                notes.push(format!(
                    "Agent session completed ({}ms, cost: ${:.4}, subtype: {})",
                    duration_ms,
                    cost_usd.unwrap_or(0.0),
                    subtype
                ));
                // Session complete — exit the event loop
                break;
            }
            _ => {
                // Ignore other events (Thinking, ToolResult, etc.)
            }
        }
    }

    // 5. Close the session (cleanup)
    if let Err(e) = chat_manager.close_session(&session_id).await {
        warn!(
            run_id = %run.id,
            session_id = %session_id,
            "AgentExecutor: failed to close session: {}",
            e
        );
    }

    // 6. Extract trigger
    let trigger = detected_trigger
        .or_else(|| extract_trigger_from_text(&accumulated_text))
        .unwrap_or_else(|| {
            warn!(
                run_id = %run.id,
                state_name = %state.name,
                "AgentExecutor: no trigger found in agent response — falling back to heuristic"
            );
            determine_trigger_from_transitions(transitions, state)
        });

    if let Some(cost) = session_cost_usd {
        notes.push(format!("Session cost: ${:.4}", cost));
    }

    info!(
        run_id = %run.id,
        state_name = %state.name,
        trigger = %trigger,
        text_len = accumulated_text.len(),
        "AgentExecutor: execution complete"
    );

    Ok(ExecutionResult {
        trigger,
        notes,
        should_retry: false,
    })
}

#[async_trait]
impl Executor for AgentExecutor {
    async fn execute_state(
        &self,
        state: &ProtocolState,
        run: &ProtocolRun,
        protocol: &Protocol,
        store: &dyn GraphStore,
    ) -> Result<ExecutionResult> {
        let action_str = match &state.action {
            Some(a) if !a.is_empty() => a.clone(),
            _ => {
                debug!(
                    state_name = %state.name,
                    run_id = %run.id,
                    "AgentExecutor: no action defined — auto-advancing"
                );
                // Delegate to system executor for trigger determination
                return self.system.execute_state(state, run, protocol, store).await;
            }
        };

        // Fast path: MCP-style action → delegate to SystemExecutor (no LLM cost)
        if is_mcp_style(&action_str) {
            debug!(
                state_name = %state.name,
                action = %action_str,
                run_id = %run.id,
                "AgentExecutor: MCP-style action — delegating to SystemExecutor"
            );
            return self.system.execute_state(state, run, protocol, store).await;
        }

        // Slow path: LLM execution via ChatManager
        let chat_manager = match &self.chat_manager {
            Some(cm) => cm,
            None => {
                // No ChatManager → stub behavior (V1 compat / tests)
                info!(
                    state_name = %state.name,
                    action = %action_str,
                    run_id = %run.id,
                    "AgentExecutor: no ChatManager available — stubbing LLM execution"
                );

                let transitions = store.get_protocol_transitions(run.protocol_id).await?;
                let trigger = determine_trigger_from_transitions(&transitions, state);

                let mut notes = vec![format!(
                    "Stub: action '{}' requires LLM but no ChatManager configured",
                    action_str
                )];
                if let Some(ref pf) = state.prompt_fragment {
                    notes.push(format!("Prompt fragment: {}", pf));
                }

                return Ok(ExecutionResult {
                    trigger,
                    notes,
                    should_retry: false,
                });
            }
        };

        // Load transitions and project root for prompt building
        let transitions = store.get_protocol_transitions(run.protocol_id).await?;
        let project = store.get_project(protocol.project_id).await?;
        let project_root = project.as_ref().map(|p| p.root_path.as_str());

        // Build the agent prompt
        let prompt = build_agent_prompt(state, run, protocol, &transitions, project_root);

        info!(
            state_name = %state.name,
            action = %action_str,
            run_id = %run.id,
            prompt_len = prompt.len(),
            "AgentExecutor: spawning Claude agent for LLM execution"
        );

        // Execute via agent
        execute_via_agent(
            chat_manager,
            prompt,
            run,
            protocol,
            state,
            &transitions,
            project_root,
            &self.cancel,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_mcp_style() {
        assert!(is_mcp_style("admin(update_staleness_scores)"));
        assert!(is_mcp_style("admin(a) + admin(b)"));
        assert!(is_mcp_style("note(list, project_id)"));
        assert!(!is_mcp_style("Run all tests"));
        assert!(!is_mcp_style("analyze code quality"));
        assert!(!is_mcp_style(""));
    }

    #[test]
    fn test_extract_trigger_from_text_basic() {
        let text = "I've completed the analysis.\n\nTRIGGER: review_complete";
        assert_eq!(
            extract_trigger_from_text(text),
            Some("review_complete".to_string())
        );
    }

    #[test]
    fn test_extract_trigger_from_text_with_backticks() {
        let text = "Done.\n\nTRIGGER: `success`";
        assert_eq!(extract_trigger_from_text(text), Some("success".to_string()));
    }

    #[test]
    fn test_extract_trigger_from_text_lowercase() {
        let text = "trigger: done";
        assert_eq!(extract_trigger_from_text(text), Some("done".to_string()));
    }

    #[test]
    fn test_extract_trigger_from_text_last_wins() {
        let text = "TRIGGER: first\nSome more text\nTRIGGER: second";
        assert_eq!(extract_trigger_from_text(text), Some("second".to_string()));
    }

    #[test]
    fn test_extract_trigger_from_text_none() {
        let text = "I've completed the task successfully.";
        assert_eq!(extract_trigger_from_text(text), None);
    }

    #[test]
    fn test_extract_trigger_from_text_protocol_transition() {
        let text = r#"I'm calling protocol transition now.
{"action": "transition", "run_id": "abc", "trigger": "tests_pass"}"#;
        assert_eq!(
            extract_trigger_from_text(text),
            Some("tests_pass".to_string())
        );
    }

    #[test]
    fn test_extract_trigger_from_tool_use_protocol() {
        let input = serde_json::json!({
            "action": "transition",
            "run_id": "abc-123",
            "trigger": "approved"
        });
        assert_eq!(
            extract_trigger_from_tool_use("mcp__project-orchestrator__protocol", &input),
            Some("approved".to_string())
        );
    }

    #[test]
    fn test_extract_trigger_from_tool_use_wrong_tool() {
        let input = serde_json::json!({
            "action": "transition",
            "trigger": "approved"
        });
        assert_eq!(
            extract_trigger_from_tool_use("mcp__other__tool", &input),
            None
        );
    }

    #[test]
    fn test_extract_trigger_from_tool_use_wrong_action() {
        let input = serde_json::json!({
            "action": "get_run",
            "run_id": "abc"
        });
        assert_eq!(
            extract_trigger_from_tool_use("mcp__project-orchestrator__protocol", &input),
            None
        );
    }

    #[test]
    fn test_determine_trigger_fallback_single() {
        let state_id = uuid::Uuid::new_v4();
        let protocol_id = uuid::Uuid::new_v4();
        let state = ProtocolState::new(protocol_id, "test");
        let transitions = vec![ProtocolTransition::new(
            protocol_id,
            state_id,
            uuid::Uuid::new_v4(),
            "next",
        )];
        // Use state_id in the transition's from_state by creating properly
        let mut state_with_id = state;
        state_with_id.id = state_id;

        let trigger = determine_trigger_from_transitions(&transitions, &state_with_id);
        assert_eq!(trigger, "next");
    }

    #[test]
    fn test_determine_trigger_fallback_prefers_done() {
        let state_id = uuid::Uuid::new_v4();
        let protocol_id = uuid::Uuid::new_v4();
        let mut state = ProtocolState::new(protocol_id, "test");
        state.id = state_id;

        let transitions = vec![
            ProtocolTransition::new(protocol_id, state_id, uuid::Uuid::new_v4(), "fail"),
            ProtocolTransition::new(protocol_id, state_id, uuid::Uuid::new_v4(), "done"),
            ProtocolTransition::new(protocol_id, state_id, uuid::Uuid::new_v4(), "retry"),
        ];

        let trigger = determine_trigger_from_transitions(&transitions, &state);
        assert_eq!(trigger, "done");
    }

    #[test]
    fn test_determine_trigger_fallback_no_outgoing() {
        let state = ProtocolState::new(uuid::Uuid::new_v4(), "terminal");
        let transitions = vec![];

        let trigger = determine_trigger_from_transitions(&transitions, &state);
        assert_eq!(trigger, "done");
    }

    #[test]
    fn test_build_agent_prompt_contains_sections() {
        let protocol_id = uuid::Uuid::new_v4();
        let state_id = uuid::Uuid::new_v4();

        let protocol = Protocol::new(uuid::Uuid::new_v4(), "Test Protocol", state_id);

        let mut state = ProtocolState::new(protocol_id, "review");
        state.id = state_id;
        state.description = "Review code changes".to_string();
        state.action = Some("Analyze the diff and provide feedback".to_string());
        state.prompt_fragment = Some("Focus on security issues".to_string());
        state.available_tools = Some(vec!["Read".to_string(), "Grep".to_string()]);
        state.forbidden_actions = Some(vec!["Do not modify files".to_string()]);

        let run = ProtocolRun::new(protocol_id, state_id, "review");

        let transitions = vec![
            ProtocolTransition::new(protocol_id, state_id, uuid::Uuid::new_v4(), "approved"),
            ProtocolTransition::new(protocol_id, state_id, uuid::Uuid::new_v4(), "rejected"),
        ];

        let prompt =
            build_agent_prompt(&state, &run, &protocol, &transitions, Some("/tmp/project"));

        // Verify all sections are present
        assert!(prompt.contains("Test Protocol"));
        assert!(prompt.contains("State: review"));
        assert!(prompt.contains("Review code changes"));
        assert!(prompt.contains("Analyze the diff and provide feedback"));
        assert!(prompt.contains("Focus on security issues"));
        assert!(prompt.contains("`Read`"));
        assert!(prompt.contains("`Grep`"));
        assert!(prompt.contains("Do not modify files"));
        assert!(prompt.contains("`approved`"));
        assert!(prompt.contains("`rejected`"));
        assert!(prompt.contains("TRIGGER:"));
        assert!(prompt.contains("/tmp/project"));
    }

    #[test]
    fn test_build_agent_prompt_no_action_no_description() {
        let protocol_id = uuid::Uuid::new_v4();
        let state_id = uuid::Uuid::new_v4();
        let protocol = Protocol::new(uuid::Uuid::new_v4(), "Proto", state_id);
        let mut state = ProtocolState::new(protocol_id, "idle");
        state.id = state_id;
        state.description = String::new();
        state.action = None;
        let run = ProtocolRun::new(protocol_id, state_id, "idle");
        let prompt = build_agent_prompt(&state, &run, &protocol, &[], None);
        // Should still contain the header and output convention
        assert!(prompt.contains("Proto"));
        assert!(prompt.contains("State: idle"));
        assert!(prompt.contains("TRIGGER:"));
        // No "State purpose" section
        assert!(!prompt.contains("State purpose"));
    }

    #[test]
    fn test_build_agent_prompt_with_states_visited() {
        use crate::protocol::StateVisit;
        let protocol_id = uuid::Uuid::new_v4();
        let state_id = uuid::Uuid::new_v4();
        let protocol = Protocol::new(uuid::Uuid::new_v4(), "P", state_id);
        let mut state = ProtocolState::new(protocol_id, "s2");
        state.id = state_id;
        let mut run = ProtocolRun::new(protocol_id, state_id, "s2");
        run.states_visited = vec![
            StateVisit {
                state_id: uuid::Uuid::new_v4(),
                state_name: "s1".to_string(),
                entered_at: chrono::Utc::now(),
                exited_at: None,
                duration_ms: None,
                trigger: Some("start".to_string()),
                progress_snapshot: None,
            },
            StateVisit {
                state_id: uuid::Uuid::new_v4(),
                state_name: "s2".to_string(),
                entered_at: chrono::Utc::now(),
                exited_at: None,
                duration_ms: None,
                trigger: None,
                progress_snapshot: None,
            },
        ];
        let prompt = build_agent_prompt(&state, &run, &protocol, &[], None);
        assert!(prompt.contains("States visited so far"));
        assert!(prompt.contains("s1"));
        assert!(prompt.contains("(trigger: `start`)"));
    }

    #[test]
    fn test_build_agent_prompt_no_transitions() {
        let protocol_id = uuid::Uuid::new_v4();
        let state_id = uuid::Uuid::new_v4();
        let protocol = Protocol::new(uuid::Uuid::new_v4(), "P", state_id);
        let mut state = ProtocolState::new(protocol_id, "final");
        state.id = state_id;
        let run = ProtocolRun::new(protocol_id, state_id, "final");
        let prompt = build_agent_prompt(&state, &run, &protocol, &[], None);
        assert!(!prompt.contains("Possible Triggers"));
    }

    #[test]
    fn test_build_agent_prompt_with_guards() {
        let protocol_id = uuid::Uuid::new_v4();
        let state_id = uuid::Uuid::new_v4();
        let protocol = Protocol::new(uuid::Uuid::new_v4(), "P", state_id);
        let mut state = ProtocolState::new(protocol_id, "gate");
        state.id = state_id;
        let run = ProtocolRun::new(protocol_id, state_id, "gate");
        let mut t = ProtocolTransition::new(protocol_id, state_id, uuid::Uuid::new_v4(), "pass");
        t.guard = Some("all_tests_green".to_string());
        let prompt = build_agent_prompt(&state, &run, &protocol, &[t], None);
        assert!(prompt.contains("Possible Triggers"));
        assert!(prompt.contains("`pass`"));
        assert!(prompt.contains("guard: all_tests_green"));
    }

    #[test]
    fn test_build_agent_prompt_no_project_root() {
        let protocol_id = uuid::Uuid::new_v4();
        let state_id = uuid::Uuid::new_v4();
        let protocol = Protocol::new(uuid::Uuid::new_v4(), "P", state_id);
        let mut state = ProtocolState::new(protocol_id, "s");
        state.id = state_id;
        let run = ProtocolRun::new(protocol_id, state_id, "s");
        let prompt = build_agent_prompt(&state, &run, &protocol, &[], None);
        assert!(!prompt.contains("Project root"));
    }

    #[test]
    fn test_determine_trigger_prefers_success_over_first() {
        let state_id = uuid::Uuid::new_v4();
        let protocol_id = uuid::Uuid::new_v4();
        let mut state = ProtocolState::new(protocol_id, "test");
        state.id = state_id;
        let transitions = vec![
            ProtocolTransition::new(protocol_id, state_id, uuid::Uuid::new_v4(), "fail"),
            ProtocolTransition::new(protocol_id, state_id, uuid::Uuid::new_v4(), "success"),
            ProtocolTransition::new(protocol_id, state_id, uuid::Uuid::new_v4(), "retry"),
        ];
        let trigger = determine_trigger_from_transitions(&transitions, &state);
        assert_eq!(trigger, "success");
    }

    #[test]
    fn test_determine_trigger_uses_first_when_no_preferred() {
        let state_id = uuid::Uuid::new_v4();
        let protocol_id = uuid::Uuid::new_v4();
        let mut state = ProtocolState::new(protocol_id, "test");
        state.id = state_id;
        let transitions = vec![
            ProtocolTransition::new(protocol_id, state_id, uuid::Uuid::new_v4(), "alpha"),
            ProtocolTransition::new(protocol_id, state_id, uuid::Uuid::new_v4(), "beta"),
        ];
        let trigger = determine_trigger_from_transitions(&transitions, &state);
        assert_eq!(trigger, "alpha");
    }

    #[test]
    fn test_extract_trigger_from_text_empty() {
        assert_eq!(extract_trigger_from_text(""), None);
    }

    #[test]
    fn test_extract_trigger_from_text_titlecase() {
        let text = "All done.\nTrigger: completed";
        assert_eq!(
            extract_trigger_from_text(text),
            Some("completed".to_string())
        );
    }

    #[test]
    fn test_extract_trigger_from_text_empty_trigger_value() {
        // "TRIGGER: " with nothing after → should return None
        let text = "TRIGGER:   ";
        assert_eq!(extract_trigger_from_text(text), None);
    }

    #[test]
    fn test_agent_executor_stub_creation() {
        let executor = AgentExecutor::new_stub();
        assert!(executor.chat_manager.is_none());
    }

    #[test]
    fn test_agent_executor_default() {
        let executor = AgentExecutor::default();
        assert!(executor.chat_manager.is_none());
    }

    #[test]
    fn test_is_mcp_style_edge_cases() {
        // Parentheses but no tool name pattern — still matches since we just check for ( and )
        assert!(is_mcp_style("()"));
        assert!(is_mcp_style("  (something)  "));
        // Only opening paren
        assert!(!is_mcp_style("foo("));
        // Only closing paren
        assert!(!is_mcp_style(")bar"));
    }
}
