//! Neutral hooks: who plays each knowledge-graph hook of a session (B14).
//!
//! ## Why
//!
//! The knowledge graph enters an agent's context through three hooks: skill
//! activation before a tool (`SkillActivationHook`), redirect advice after a noisy
//! tool (`PostToolUseRedirectHook`) and the compaction guidance
//! (`CompactionNotifier`), with the injection ledger (`hook_ledger`) beside them.
//! Claude Code and the native harness call them in their own loop
//! (`HookSupport::InProtocol`, the table of `ChatManager::graph_hook_table`). A
//! provider that runs no hook (Codex, ACP, a remote Claude Code given no local
//! project) used to get nothing at those points, without a word.
//!
//! [`HookPlan`] says, per hook point, which door the session uses — the provider's
//! loop, or the host around the turn — from the session's capabilities. The host
//! door is [`HookFallback`]:
//!
//! | Hook | Provider door | Host door (fallback) |
//! |---|---|---|
//! | `SkillActivationHook` | `PreToolUse` in the loop | the skill context of the turn (`stages::skill_activation`, every turn, every provider); the per-tool injection lost is written in the ledger |
//! | `PostToolUseRedirectHook` | `PostToolUse` in the loop | the same callback over each `tool_result` seen, after the turn; its advice is announced as a `system_hint` and carried in front of the next turn |
//! | `CompactionNotifier` | `PreCompact` in the loop (the native harness: PO's own compaction calls it) | the provider's compaction signal (`compaction_started` / `compact_boundary`) resets the ledger; the context held for after a compaction is the end of turn's (`HeldContext`) |
//!
//! The ledger is the same type on both doors: provider-neutral, reset on the
//! compaction signal whichever door brings it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use nexus_claude::agent::{Capabilities, HookSupport, ProviderKind};
use serde_json::Value;

use crate::chat::hook_ledger::{HookLedger, HookPoint};
use crate::chat::types::ChatEvent;

/// Who plays a hook point of a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Door {
    /// The provider calls it in its loop (`SessionSpec::hooks`).
    Provider,
    /// The host plays its part around the turn ([`HookFallback`]).
    Host,
}

/// The door of each hook point of a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HookPlan {
    pub before_tool: Door,
    pub after_tool: Door,
    pub before_compaction: Door,
}

/// Whether the host hands the graph's hook table to a session of `kind`
/// (`SessionSpec::hooks`). A remote session has no local project to resolve a file
/// against: it is given none.
pub(crate) fn provider_takes_hooks(kind: ProviderKind, local: bool) -> bool {
    local && matches!(kind, ProviderKind::Native | ProviderKind::ClaudeCode)
}

impl HookPlan {
    /// The plan of a session of `kind` whose frozen capabilities are `caps`. The
    /// provider's door needs both: the host handed it the table, and the provider
    /// says it calls it (`HookSupport::InProtocol`; `None` and `Command` ignore it).
    pub(crate) fn for_session(kind: ProviderKind, caps: &Capabilities, local: bool) -> Self {
        let door = if provider_takes_hooks(kind, local) && caps.hooks == HookSupport::InProtocol {
            Door::Provider
        } else {
            Door::Host
        };
        Self {
            before_tool: door,
            after_tool: door,
            before_compaction: door,
        }
    }

    /// Some hook point needs the host's fallback.
    pub(crate) fn needs_fallback(&self) -> bool {
        [self.before_tool, self.after_tool, self.before_compaction].contains(&Door::Host)
    }
}

/// Tool results kept for the after-turn pass, at most (the latest ones): a turn of
/// hundreds of calls is not replayed whole through the post-tool hook.
const MAX_RESULTS_PER_TURN: usize = 32;

/// A tool call seen on the wire: its name (the stable alias when the adapter gives
/// one, the vocabulary the hooks are written in) and its input.
type Call = (String, Value);

#[derive(Default)]
struct FallbackState {
    /// Calls announced, by id, waiting for their result.
    open: HashMap<String, Call>,
    /// Calls that ended with a result during the turn, with that result.
    finished: Vec<(Call, Value)>,
    /// Advice waiting for the next turn.
    carried: Vec<String>,
    /// A compaction was announced started, its boundary not seen yet.
    compacting: bool,
}

/// The host's door to the graph hooks of a session whose provider runs none
/// (see the module).
pub(crate) struct HookFallback {
    /// The post-tool callback (`PostToolUseRedirectHook` in production).
    after_tool: Arc<dyn nexus_claude::HookCallback>,
    ledger: Arc<HookLedger>,
    session_id: String,
    cwd: String,
    state: Mutex<FallbackState>,
}

impl HookFallback {
    pub(crate) fn new(
        after_tool: Arc<dyn nexus_claude::HookCallback>,
        ledger: Arc<HookLedger>,
        session_id: impl Into<String>,
        cwd: impl Into<String>,
    ) -> Self {
        Self {
            after_tool,
            ledger,
            session_id: session_id.into(),
            cwd: cwd.into(),
            state: Mutex::new(FallbackState::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, FallbackState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Sees each event of the session as it is emitted.
    pub(crate) fn observe(&self, event: &ChatEvent) {
        match event {
            ChatEvent::ToolUse {
                id,
                tool,
                input,
                canonical,
                parent_tool_use_id,
                ..
            } => {
                let name = canonical.clone().unwrap_or_else(|| tool.clone());
                // One tool call: the ledger's clock (what was advised goes stale
                // after `REINJECT_AFTER_CALLS` calls), as the PreToolUse hook's.
                self.ledger.begin_call();
                // The skill context this call would have been given before it ran
                // never reaches it: said, not silent.
                let total = self.ledger.record_lost(HookPoint::BeforeTool, &name);
                if total == 1 {
                    tracing::info!(
                        session_id = %self.session_id,
                        tool = %name,
                        "the provider runs no hook: the per-tool skill context is lost, \
                         the skill context arrives once per turn (hook_ledger)"
                    );
                } else {
                    tracing::debug!(
                        session_id = %self.session_id,
                        tool = %name,
                        lost = total,
                        "per-tool injection lost (no hook in the provider)"
                    );
                }
                // A sub-agent's call is not advised on: its advice would land in
                // front of the PARENT's next turn.
                if parent_tool_use_id.is_none() {
                    self.lock().open.insert(id.clone(), (name, input.clone()));
                }
            }
            ChatEvent::ToolUseInputResolved { id, input, .. } => {
                if let Some(call) = self.lock().open.get_mut(id) {
                    call.1 = input.clone();
                }
            }
            ChatEvent::ToolResult {
                id,
                result,
                is_error,
                ..
            } => {
                let mut state = self.lock();
                if let Some(call) = state.open.remove(id) {
                    if !is_error {
                        state.finished.push((call, result.clone()));
                        let excess = state.finished.len().saturating_sub(MAX_RESULTS_PER_TURN);
                        state.finished.drain(..excess);
                    }
                }
            }
            ChatEvent::ToolCancelled { id, .. } => {
                self.lock().open.remove(id);
            }
            // The provider's compaction signal: what was injected is no longer in
            // context. The guidance the provider would have been given before it
            // (`CompactionNotifier`) could not be: written once per compaction, on
            // its start (a provider that only says it ended, on its boundary).
            ChatEvent::CompactionStarted { trigger } => {
                self.ledger.reset();
                self.ledger
                    .record_lost(HookPoint::BeforeCompaction, trigger);
                self.lock().compacting = true;
            }
            ChatEvent::CompactBoundary { trigger, .. } => {
                self.ledger.reset();
                if !std::mem::take(&mut self.lock().compacting) {
                    self.ledger
                        .record_lost(HookPoint::BeforeCompaction, trigger);
                }
            }
            _ => {}
        }
    }

    /// After a turn: plays the post-tool hook over each tool result of the turn.
    /// Returns the advice it gave (announced as system hints by the caller), which
    /// is also carried in front of the next turn ([`Self::carried`]).
    pub(crate) async fn after_turn(&self) -> Vec<String> {
        let finished = std::mem::take(&mut self.lock().finished);
        let mut advice = Vec::new();
        for ((tool, input), result) in finished {
            let hook_input =
                nexus_claude::HookInput::PostToolUse(nexus_claude::PostToolUseHookInput {
                    session_id: self.session_id.clone(),
                    transcript_path: String::new(),
                    cwd: self.cwd.clone(),
                    permission_mode: None,
                    tool_name: tool.clone(),
                    tool_input: input,
                    tool_response: result,
                    agent_id: None,
                    agent_type: None,
                });
            let ctx = nexus_claude::HookContext { signal: None };
            match self.after_tool.execute(&hook_input, None, &ctx).await {
                Ok(output) => {
                    if let Some(text) = additional_context(output) {
                        advice.push(text);
                    }
                }
                // A hook advises, it never takes the turn down.
                Err(e) => tracing::warn!(
                    session_id = %self.session_id,
                    tool = %tool,
                    error = %e,
                    "post-tool hook (host fallback) failed"
                ),
            }
        }
        self.lock().carried.extend(advice.iter().cloned());
        // What the session missed so far, readable in the log at each turn's end.
        let lost = self.ledger.lost();
        if !lost.is_empty() {
            tracing::debug!(
                session_id = %self.session_id,
                lost = ?lost,
                "injections the provider could not take (hook_ledger)"
            );
        }
        advice
    }

    /// The advice of the previous turns: what goes in front of the next turn. Only
    /// read: [`Self::settle`] drops it once a turn carrying it was answered, as the
    /// held context (`TurnOutcome::answered`).
    pub(crate) fn carried(&self) -> Option<String> {
        let state = self.lock();
        (!state.carried.is_empty()).then(|| state.carried.join("\n\n"))
    }

    /// The end of a turn: an answered turn delivered what was carried in front of
    /// it, which is dropped; a turn refused, failed or stopped keeps it for the next.
    /// Called BEFORE [`Self::after_turn`] adds the advice of this turn's tools.
    pub(crate) fn settle(&self, answered: bool) {
        if answered {
            self.lock().carried.clear();
        }
    }
}

/// The `additionalContext` of a post-tool answer.
fn additional_context(output: nexus_claude::HookJSONOutput) -> Option<String> {
    match output {
        nexus_claude::HookJSONOutput::Sync(sync) => match sync.hook_specific_output {
            Some(nexus_claude::HookSpecificOutput::PostToolUse(post)) => post
                .additional_context
                .filter(|text| !text.trim().is_empty()),
            _ => None,
        },
        _ => None,
    }
}

/// Answers a fixed advice after a `Grep`, nothing otherwise (tests).
#[cfg(test)]
pub(crate) struct TestAdvisor;

#[cfg(test)]
#[async_trait::async_trait]
impl nexus_claude::HookCallback for TestAdvisor {
    async fn execute(
        &self,
        input: &nexus_claude::HookInput,
        _tool_use_id: Option<&str>,
        _context: &nexus_claude::HookContext,
    ) -> Result<nexus_claude::HookJSONOutput, nexus_claude::SdkError> {
        let context = match input {
            nexus_claude::HookInput::PostToolUse(post) if post.tool_name == "Grep" => {
                Some(format!(
                    "use find_references ({})",
                    post.tool_response.as_str().unwrap()
                ))
            }
            _ => None,
        };
        Ok(nexus_claude::HookJSONOutput::Sync(
            nexus_claude::SyncHookJSONOutput {
                continue_: Some(true),
                hook_specific_output: context.map(|c| {
                    nexus_claude::HookSpecificOutput::PostToolUse(
                        nexus_claude::PostToolUseHookSpecificOutput {
                            additional_context: Some(c),
                        },
                    )
                }),
                ..Default::default()
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(hooks: HookSupport) -> Capabilities {
        let mut caps = Capabilities::none();
        caps.hooks = hooks;
        caps
    }

    #[test]
    fn the_provider_plays_the_hooks_only_when_it_is_handed_them_and_calls_them() {
        let in_protocol = caps(HookSupport::InProtocol);
        for kind in [ProviderKind::Native, ProviderKind::ClaudeCode] {
            let plan = HookPlan::for_session(kind, &in_protocol, true);
            assert_eq!(plan.before_tool, Door::Provider, "{kind:?}");
            assert!(!plan.needs_fallback(), "{kind:?}");
            // A remote session is handed no table.
            assert!(HookPlan::for_session(kind, &in_protocol, false).needs_fallback());
        }
        // A provider that says it runs no hook falls back, whatever its kind.
        for support in [HookSupport::None, HookSupport::Command] {
            let plan = HookPlan::for_session(ProviderKind::Native, &caps(support), true);
            assert_eq!(
                (plan.before_tool, plan.after_tool, plan.before_compaction),
                (Door::Host, Door::Host, Door::Host)
            );
        }
        for kind in [ProviderKind::Codex, ProviderKind::Acp] {
            assert!(HookPlan::for_session(kind, &in_protocol, true).needs_fallback());
        }
    }

    fn tool_use(id: &str, tool: &str, canonical: Option<&str>) -> ChatEvent {
        ChatEvent::ToolUse {
            id: id.into(),
            tool: tool.into(),
            input: serde_json::json!({}),
            parent_tool_use_id: None,
            category: None,
            canonical: canonical.map(str::to_string),
        }
    }

    fn tool_result(id: &str, result: &str, is_error: bool) -> ChatEvent {
        ChatEvent::ToolResult {
            id: id.into(),
            result: Value::String(result.into()),
            is_error,
            parent_tool_use_id: None,
        }
    }

    #[tokio::test]
    async fn the_fallback_advises_after_the_turn_carries_it_once_and_writes_the_loss() {
        let ledger = Arc::new(HookLedger::new());
        let fallback = HookFallback::new(Arc::new(TestAdvisor), Arc::clone(&ledger), "s", "/tmp");
        // The provider spells it `grep`, its alias is the hooks' vocabulary.
        fallback.observe(&tool_use("t1", "grep", Some("Grep")));
        fallback.observe(&tool_result("t1", "30 lines", false));
        fallback.observe(&tool_use("t2", "read_file", None));
        fallback.observe(&tool_result("t2", "body", false));
        // A failed call is not advised on.
        fallback.observe(&tool_use("t3", "Grep", None));
        fallback.observe(&tool_result("t3", "boom", true));

        assert_eq!(
            fallback.after_turn().await,
            vec!["use find_references (30 lines)".to_string()]
        );
        let advice = Some("use find_references (30 lines)");
        assert_eq!(fallback.carried().as_deref(), advice);
        fallback.settle(false);
        assert_eq!(
            fallback.carried().as_deref(),
            advice,
            "kept by a turn not answered"
        );
        fallback.settle(true);
        assert_eq!(fallback.carried(), None, "carried once");
        assert!(fallback.after_turn().await.is_empty(), "played once");
        // A sub-agent's Grep: its loss is written, its advice is not given.
        let mut sub = tool_use("t4", "Grep", None);
        if let ChatEvent::ToolUse {
            parent_tool_use_id, ..
        } = &mut sub
        {
            *parent_tool_use_id = Some("agent-1".into());
        }
        fallback.observe(&sub);
        fallback.observe(&tool_result("t4", "30 lines", false));
        assert!(
            fallback.after_turn().await.is_empty(),
            "no advice for a sub-agent"
        );
        assert_eq!(
            ledger.lost(),
            vec![
                (HookPoint::BeforeTool, "Grep".to_string(), 3),
                (HookPoint::BeforeTool, "read_file".to_string(), 1),
            ]
        );
    }

    #[test]
    fn the_compaction_signal_resets_the_ledger() {
        use crate::chat::hook_ledger::InjectionKey;
        for signal in [
            ChatEvent::CompactionStarted {
                trigger: "auto".into(),
            },
            ChatEvent::CompactBoundary {
                trigger: "auto".into(),
                pre_tokens: None,
            },
        ] {
            let ledger = Arc::new(HookLedger::new());
            let fallback = HookFallback::new(Arc::new(TestAdvisor), Arc::clone(&ledger), "s", "/");
            let key = InjectionKey::Redirect("find_references".into());
            ledger.begin_call();
            ledger.record(key.clone());
            assert!(!ledger.is_fresh(&key));
            fallback.observe(&signal);
            assert!(ledger.is_fresh(&key), "{signal:?}");
            assert_eq!(
                ledger.lost(),
                vec![(HookPoint::BeforeCompaction, "auto".to_string(), 1)],
                "{signal:?}"
            );
        }
        // Started then its boundary: one compaction, one guidance lost.
        let ledger = Arc::new(HookLedger::new());
        let fallback = HookFallback::new(Arc::new(TestAdvisor), Arc::clone(&ledger), "s", "/");
        fallback.observe(&ChatEvent::CompactionStarted {
            trigger: "manual".into(),
        });
        fallback.observe(&ChatEvent::CompactBoundary {
            trigger: "manual".into(),
            pre_tokens: Some(10),
        });
        assert_eq!(
            ledger.lost(),
            vec![(HookPoint::BeforeCompaction, "manual".to_string(), 1)]
        );
    }
}
