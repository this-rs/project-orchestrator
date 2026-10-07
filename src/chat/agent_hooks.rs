//! The knowledge-graph hooks of the Claude Code engine, served to any provider.
//!
//! ## Why
//!
//! The Claude Code engine builds its hooks once per session as a table
//! `event name → [HookMatcher]` (skill activation before a tool, redirect advice
//! after a noisy one, the compaction notifier and the injection ledger reset at
//! compaction). Those hooks are where the project's graph enters the agent's
//! context while it works. The agent engine used to start every session with no
//! hook at all, so a native session never saw a skill, a persona, a note or the
//! compaction guidance, whatever the model behind it.
//!
//! [`GraphSessionHooks`] is the provider-neutral face of the SAME table: it
//! implements the contract's [`SessionHooks`] on top of the very callbacks the
//! Claude path registers. There is one logic, two doors.
//!
//! ## Mapping
//!
//! | `SessionHooks` | Claude event | What comes back |
//! |---|---|---|
//! | `before_tool` | `PreToolUse` | `additionalContext` → `AddContext`; `decision: block` or `permissionDecision: deny` → `Deny`; `updatedInput` → `ReplaceInput` |
//! | `after_tool` | `PostToolUse` | `additionalContext` |
//! | `before_compaction` | `PreCompact` | the callbacks' `reason` (where the notifier puts the instructions it built) |
//!
//! The tool is named as Claude Code names it (`canonical`, else the provider's
//! name) and its input is given as is: that is the vocabulary the hooks were
//! written in (`Read`, `Edit`, `Bash`… with `file_path`, `command`…).
//!
//! A hook that fails is skipped (logged): a hook observes and advises, it never
//! takes the turn down.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use nexus_claude::agent::{
    CompactionInfo, HookVerdict, SessionHooks, ToolCallInfo, ToolResultInfo,
};
use nexus_claude::{
    HookContext, HookInput, HookJSONOutput, HookMatcher, PostToolUseHookInput, PreCompactHookInput,
    PreToolUseHookInput,
};
use serde_json::Value;
use tracing::warn;

/// The table the Claude path hands to its CLI: event name → matchers.
pub(crate) type HookTable = HashMap<String, Vec<HookMatcher>>;

pub(crate) const PRE_TOOL_USE: &str = "PreToolUse";
pub(crate) const POST_TOOL_USE: &str = "PostToolUse";
pub(crate) const PRE_COMPACT: &str = "PreCompact";

/// [`SessionHooks`] over a [`HookTable`].
pub(crate) struct GraphSessionHooks {
    table: HookTable,
    session_id: String,
    cwd: String,
    permission_mode: Option<String>,
}

impl GraphSessionHooks {
    pub(crate) fn new(
        table: HookTable,
        session_id: impl Into<String>,
        cwd: impl Into<String>,
        permission_mode: Option<String>,
    ) -> Self {
        Self {
            table,
            session_id: session_id.into(),
            cwd: cwd.into(),
            permission_mode,
        }
    }

    /// The callbacks registered for `event` whose matcher accepts `tool`.
    fn callbacks<'a>(
        &'a self,
        event: &str,
        tool: Option<&'a str>,
    ) -> impl Iterator<Item = &'a Arc<dyn nexus_claude::HookCallback>> + 'a {
        self.table
            .get(event)
            .into_iter()
            .flatten()
            .filter(move |matcher| matches(matcher.matcher.as_ref(), tool))
            .flat_map(|matcher| matcher.hooks.iter())
    }

    async fn run(
        &self,
        hook: &Arc<dyn nexus_claude::HookCallback>,
        input: &HookInput,
        tool_use_id: Option<&str>,
        event: &str,
    ) -> Option<HookJSONOutput> {
        let context = HookContext { signal: None };
        match hook.execute(input, tool_use_id, &context).await {
            Ok(output) => Some(output),
            Err(error) => {
                warn!(session_id = %self.session_id, event, error = %error, "graph hook failed (skipped)");
                None
            }
        }
    }
}

/// Claude's matcher: absent or empty matches everything; a string is a list of
/// tool names separated by `|` (`*` matches everything).
fn matches(matcher: Option<&Value>, tool: Option<&str>) -> bool {
    let Some(pattern) = matcher.and_then(Value::as_str).filter(|p| !p.is_empty()) else {
        return true;
    };
    let Some(tool) = tool else {
        return true;
    };
    pattern
        .split('|')
        .any(|name| name == "*" || name.trim() == tool)
}

/// What a callback said, flattened.
#[derive(Default)]
struct Said {
    context: Option<String>,
    deny: Option<String>,
    replaced: Option<Value>,
    reason: Option<String>,
}

fn read(output: HookJSONOutput) -> Said {
    use nexus_claude::HookSpecificOutput;
    let HookJSONOutput::Sync(sync) = output else {
        return Said::default();
    };
    let mut said = Said {
        reason: sync.reason.clone(),
        ..Said::default()
    };
    if sync.decision.as_deref() == Some("block") {
        said.deny = Some(
            sync.reason
                .clone()
                .unwrap_or_else(|| "blocked by a hook".to_owned()),
        );
    }
    match sync.hook_specific_output {
        Some(HookSpecificOutput::PreToolUse(pre)) => {
            if pre.permission_decision.as_deref() == Some("deny") {
                said.deny = Some(
                    pre.permission_decision_reason
                        .unwrap_or_else(|| "denied by a hook".to_owned()),
                );
            }
            said.replaced = pre.updated_input;
            said.context = pre.additional_context;
        }
        Some(HookSpecificOutput::PostToolUse(post)) => said.context = post.additional_context,
        _ => {}
    }
    said
}

/// Texts joined by a blank line, empty ones dropped.
fn joined(parts: Vec<String>) -> Option<String> {
    let parts: Vec<String> = parts.into_iter().filter(|p| !p.trim().is_empty()).collect();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

fn tool_name(call: &ToolCallInfo) -> String {
    call.canonical.clone().unwrap_or_else(|| call.name.clone())
}

#[async_trait]
impl SessionHooks for GraphSessionHooks {
    async fn before_tool(&self, call: &ToolCallInfo) -> HookVerdict {
        let name = tool_name(call);
        let input = HookInput::PreToolUse(PreToolUseHookInput {
            session_id: self.session_id.clone(),
            transcript_path: String::new(),
            cwd: self.cwd.clone(),
            permission_mode: self.permission_mode.clone(),
            tool_name: name.clone(),
            tool_input: call.input.clone(),
            agent_id: None,
            agent_type: None,
        });
        let mut contexts = Vec::new();
        let mut replaced = None;
        for hook in self.callbacks(PRE_TOOL_USE, Some(&name)) {
            let Some(output) = self
                .run(hook, &input, call.id.as_deref(), PRE_TOOL_USE)
                .await
            else {
                continue;
            };
            let said = read(output);
            if let Some(reason) = said.deny {
                return HookVerdict::Deny { reason };
            }
            if said.replaced.is_some() {
                replaced = said.replaced;
            }
            contexts.extend(said.context);
        }
        // A replaced input and added context cannot both travel in one verdict: the
        // replacement wins (it changes what runs), the context is not lost when
        // there is no replacement.
        match (replaced, joined(contexts)) {
            (Some(input), _) => HookVerdict::ReplaceInput(input),
            (None, Some(text)) => HookVerdict::AddContext(text),
            (None, None) => HookVerdict::Continue,
        }
    }

    async fn after_tool(&self, result: &ToolResultInfo) -> Option<String> {
        let name = tool_name(&result.call);
        let input = HookInput::PostToolUse(PostToolUseHookInput {
            session_id: self.session_id.clone(),
            transcript_path: String::new(),
            cwd: self.cwd.clone(),
            permission_mode: self.permission_mode.clone(),
            tool_name: name.clone(),
            tool_input: result.call.input.clone(),
            tool_response: result.output.clone(),
            agent_id: None,
            agent_type: None,
        });
        let mut contexts = Vec::new();
        for hook in self.callbacks(POST_TOOL_USE, Some(&name)) {
            if let Some(output) = self
                .run(hook, &input, result.call.id.as_deref(), POST_TOOL_USE)
                .await
            {
                contexts.extend(read(output).context);
            }
        }
        joined(contexts)
    }

    async fn before_compaction(&self, info: &CompactionInfo) -> Option<String> {
        let input = HookInput::PreCompact(PreCompactHookInput {
            session_id: self.session_id.clone(),
            transcript_path: String::new(),
            cwd: self.cwd.clone(),
            permission_mode: self.permission_mode.clone(),
            trigger: info.trigger.clone(),
            custom_instructions: info.custom_instructions.clone(),
        });
        let mut reasons = Vec::new();
        for hook in self.callbacks(PRE_COMPACT, None) {
            if let Some(output) = self.run(hook, &input, None, PRE_COMPACT).await {
                reasons.extend(read(output).reason);
            }
        }
        joined(reasons)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_claude::agent::ToolCategory;
    use nexus_claude::{
        HookCallback, HookSpecificOutput, PostToolUseHookSpecificOutput,
        PreToolUseHookSpecificOutput, SyncHookJSONOutput,
    };
    use serde_json::json;
    use std::sync::Mutex;

    /// Answers with a fixed output and remembers what it was given.
    struct Scripted {
        output: Mutex<Option<HookJSONOutput>>,
        seen: Mutex<Vec<HookInput>>,
        fail: bool,
    }

    impl Scripted {
        fn answering(output: SyncHookJSONOutput) -> Arc<Self> {
            Arc::new(Self {
                output: Mutex::new(Some(HookJSONOutput::Sync(output))),
                seen: Mutex::new(Vec::new()),
                fail: false,
            })
        }

        fn failing() -> Arc<Self> {
            Arc::new(Self {
                output: Mutex::new(None),
                seen: Mutex::new(Vec::new()),
                fail: true,
            })
        }
    }

    #[async_trait]
    impl HookCallback for Scripted {
        async fn execute(
            &self,
            input: &HookInput,
            _tool_use_id: Option<&str>,
            _context: &HookContext,
        ) -> Result<HookJSONOutput, nexus_claude::SdkError> {
            self.seen.lock().unwrap().push(input.clone());
            if self.fail {
                return Err(nexus_claude::SdkError::InvalidState {
                    message: "boom".into(),
                });
            }
            Ok(self
                .output
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| HookJSONOutput::Sync(SyncHookJSONOutput::default())))
        }
    }

    fn table(event: &str, matcher: Option<Value>, hooks: Vec<Arc<dyn HookCallback>>) -> HookTable {
        let mut table = HookTable::new();
        table.insert(event.to_owned(), vec![HookMatcher { matcher, hooks }]);
        table
    }

    fn hooks(table: HookTable) -> GraphSessionHooks {
        GraphSessionHooks::new(table, "s1", "/work/app", Some("default".into()))
    }

    fn call(name: &str, canonical: Option<&str>, input: Value) -> ToolCallInfo {
        ToolCallInfo {
            id: Some("c1".into()),
            name: name.into(),
            canonical: canonical.map(str::to_owned),
            category: ToolCategory::Mcp,
            input,
        }
    }

    fn pre_context(text: &str) -> SyncHookJSONOutput {
        SyncHookJSONOutput {
            continue_: Some(true),
            hook_specific_output: Some(HookSpecificOutput::PreToolUse(
                PreToolUseHookSpecificOutput {
                    permission_decision: None,
                    permission_decision_reason: None,
                    updated_input: None,
                    additional_context: Some(text.into()),
                },
            )),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn before_tool_gives_the_hook_the_claude_vocabulary_and_returns_its_context() {
        let hook = Scripted::answering(pre_context("SKILL: use find_references"));
        let hooks = hooks(table(PRE_TOOL_USE, None, vec![hook.clone()]));
        let verdict = hooks
            .before_tool(&call(
                "mcp__nexus__read",
                Some("Read"),
                json!({"file_path": "src/a.rs"}),
            ))
            .await;
        assert_eq!(
            verdict,
            HookVerdict::AddContext("SKILL: use find_references".into())
        );
        let seen = hook.seen.lock().unwrap();
        let HookInput::PreToolUse(pre) = &seen[0] else {
            panic!("a PreToolUse input is expected");
        };
        // Named as Claude Code names it, input untouched, session facts filled in.
        assert_eq!(pre.tool_name, "Read");
        assert_eq!(pre.tool_input, json!({"file_path": "src/a.rs"}));
        assert_eq!(
            (pre.session_id.as_str(), pre.cwd.as_str()),
            ("s1", "/work/app")
        );
    }

    #[tokio::test]
    async fn a_tool_without_canonical_name_is_given_under_its_own_name() {
        let hook = Scripted::answering(pre_context("x"));
        let hooks = hooks(table(PRE_TOOL_USE, None, vec![hook.clone()]));
        hooks
            .before_tool(&call("mcp__project-orchestrator__note", None, json!({})))
            .await;
        let seen = hook.seen.lock().unwrap();
        let HookInput::PreToolUse(pre) = &seen[0] else {
            panic!()
        };
        assert_eq!(pre.tool_name, "mcp__project-orchestrator__note");
    }

    #[tokio::test]
    async fn a_blocking_or_denying_hook_denies_with_its_reason() {
        let block = Scripted::answering(SyncHookJSONOutput {
            decision: Some("block".into()),
            reason: Some("hot bridge".into()),
            ..Default::default()
        });
        let verdict = hooks(table(PRE_TOOL_USE, None, vec![block]))
            .before_tool(&call("t", None, json!({})))
            .await;
        assert_eq!(
            verdict,
            HookVerdict::Deny {
                reason: "hot bridge".into()
            }
        );

        let deny = Scripted::answering(SyncHookJSONOutput {
            hook_specific_output: Some(HookSpecificOutput::PreToolUse(
                PreToolUseHookSpecificOutput {
                    permission_decision: Some("deny".into()),
                    permission_decision_reason: Some("not here".into()),
                    updated_input: None,
                    additional_context: None,
                },
            )),
            ..Default::default()
        });
        let verdict = hooks(table(PRE_TOOL_USE, None, vec![deny]))
            .before_tool(&call("t", None, json!({})))
            .await;
        assert_eq!(
            verdict,
            HookVerdict::Deny {
                reason: "not here".into()
            }
        );
    }

    #[tokio::test]
    async fn an_updated_input_replaces_the_input() {
        let rewrite = Scripted::answering(SyncHookJSONOutput {
            hook_specific_output: Some(HookSpecificOutput::PreToolUse(
                PreToolUseHookSpecificOutput {
                    permission_decision: None,
                    permission_decision_reason: None,
                    updated_input: Some(json!({"command": "ls"})),
                    additional_context: None,
                },
            )),
            ..Default::default()
        });
        let verdict = hooks(table(PRE_TOOL_USE, None, vec![rewrite]))
            .before_tool(&call("Bash", Some("Bash"), json!({"command": "ls -R /"})))
            .await;
        assert_eq!(verdict, HookVerdict::ReplaceInput(json!({"command": "ls"})));
    }

    #[tokio::test]
    async fn contexts_of_several_hooks_are_joined_and_a_failing_hook_is_skipped() {
        let a = Scripted::answering(pre_context("ONE"));
        let b = Scripted::answering(pre_context("TWO"));
        let verdict = hooks(table(PRE_TOOL_USE, None, vec![a, Scripted::failing(), b]))
            .before_tool(&call("t", None, json!({})))
            .await;
        assert_eq!(verdict, HookVerdict::AddContext("ONE\n\nTWO".into()));
    }

    #[tokio::test]
    async fn without_a_registered_hook_the_call_goes_on() {
        let hooks = hooks(HookTable::new());
        assert_eq!(
            hooks.before_tool(&call("t", None, json!({}))).await,
            HookVerdict::Continue
        );
        assert_eq!(
            hooks
                .after_tool(&ToolResultInfo {
                    call: call("t", None, json!({})),
                    output: json!("x"),
                    is_error: false,
                })
                .await,
            None
        );
        assert_eq!(
            hooks
                .before_compaction(&CompactionInfo {
                    trigger: "auto".into(),
                    custom_instructions: None,
                })
                .await,
            None
        );
    }

    #[tokio::test]
    async fn a_matcher_restricts_a_hook_to_its_tools() {
        let hook = Scripted::answering(pre_context("ONLY BASH"));
        let hooks = hooks(table(
            PRE_TOOL_USE,
            Some(json!("Bash|Grep")),
            vec![hook.clone()],
        ));
        assert_eq!(
            hooks.before_tool(&call("x", Some("Read"), json!({}))).await,
            HookVerdict::Continue
        );
        assert!(hook.seen.lock().unwrap().is_empty());
        assert_eq!(
            hooks.before_tool(&call("x", Some("Grep"), json!({}))).await,
            HookVerdict::AddContext("ONLY BASH".into())
        );
    }

    #[tokio::test]
    async fn after_tool_hands_over_the_response_and_returns_the_context() {
        let hook = Scripted::answering(SyncHookJSONOutput {
            hook_specific_output: Some(HookSpecificOutput::PostToolUse(
                PostToolUseHookSpecificOutput {
                    additional_context: Some("use find_references instead".into()),
                },
            )),
            ..Default::default()
        });
        let hooks = hooks(table(POST_TOOL_USE, None, vec![hook.clone()]));
        let said = hooks
            .after_tool(&ToolResultInfo {
                call: call("mcp__nexus__grep", Some("Grep"), json!({"pattern": "foo"})),
                output: json!("60 matches"),
                is_error: false,
            })
            .await;
        assert_eq!(said.as_deref(), Some("use find_references instead"));
        let seen = hook.seen.lock().unwrap();
        let HookInput::PostToolUse(post) = &seen[0] else {
            panic!("a PostToolUse input is expected");
        };
        assert_eq!(post.tool_name, "Grep");
        assert_eq!(post.tool_response, json!("60 matches"));
    }

    #[tokio::test]
    async fn before_compaction_returns_the_reason_the_notifier_builds() {
        let notifier = Scripted::answering(SyncHookJSONOutput {
            continue_: Some(true),
            reason: Some("Preserve: decision on the hot bridge".into()),
            ..Default::default()
        });
        let reset = Scripted::answering(SyncHookJSONOutput {
            continue_: Some(true),
            ..Default::default()
        });
        let hooks = hooks(table(
            PRE_COMPACT,
            None,
            vec![notifier.clone(), reset.clone()],
        ));
        let said = hooks
            .before_compaction(&CompactionInfo {
                trigger: "auto".into(),
                custom_instructions: None,
            })
            .await;
        assert_eq!(
            said.as_deref(),
            Some("Preserve: decision on the hot bridge")
        );
        // Both ran: the ledger reset must see every compaction.
        assert_eq!(notifier.seen.lock().unwrap().len(), 1);
        assert_eq!(reset.seen.lock().unwrap().len(), 1);
        let seen = notifier.seen.lock().unwrap();
        let HookInput::PreCompact(pre) = &seen[0] else {
            panic!("a PreCompact input is expected");
        };
        assert_eq!(pre.trigger, "auto");
    }
}
