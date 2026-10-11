//! When each tool call really ran, as the engine saw it.
//!
//! ## Why
//!
//! A `tool_use` frame is stamped when the model ANNOUNCES the call (often before
//! its input has finished streaming) and a `tool_result` when the result comes
//! back. The span between the two also holds the time the user took to answer a
//! permission, and, for calls the engine runs one after another, the time the
//! previous calls ran. A trace drawn from those two times lies on exactly the
//! calls worth looking at.
//!
//! ## What is measured
//!
//! One [`ToolClock`] per session ([`ToolClock::for_session`]), fed where each
//! engine produces what it knows, never from a relay:
//!
//! - the **PreToolUse callbacks** of the shared hook table (`graph_hook_table`),
//!   wrapped by [`clock_the_table`]: the moment the engine takes the call up (the
//!   Claude Code CLI, or the agent engine through `GraphSessionHooks::before_tool`).
//!   On both engines it runs BEFORE the permission is asked;
//! - the **chat events**, observed where the engine emits them (the legacy stream
//!   loop and its interrupt cleanup, `AgentSessionHandle::emit`): `tool_use`
//!   (announced), `permission_request` (matched to its call by `tool_use_id`) and
//!   `ask_user_question` (the engine waits for the user), `tool_result` /
//!   `tool_cancelled` (the call is over);
//! - the user's **answer to a permission**, noted by [`ToolClock::decided`] BEFORE
//!   it is handed to the CLI or the provider (a fast tool's result cannot overtake
//!   it), and taken back by [`ToolClock::undecided`] when the hand-over fails —
//!   only the mark that answer placed ([`DecisionMark`]): a second answer to the
//!   same request (a double click, two tabs) that the engine refuses does not
//!   erase the first.
//!
//! When the call is over, the clock answers one [`ChatEvent::ToolTiming`], which
//! the engine stores and relays right after the result, in the same order.
//! Times: seconds since the epoch, the milliseconds as the fraction.
//!
//! `run_started_at` is when the tool itself started running, and only a time the
//! engine saw: the permission's answer when one was asked and allowed, else the
//! take-up. It is absent rather than guessed: a denied permission (the tool never
//! ran), a permission never answered (a cancellation, a question answered by the
//! result itself), an engine that runs no host hook (remote cwd, Codex, ACP
//! sessions of the agent engine). `incomplete` says the clock may have missed a
//! wait (a permission request that named no call): such a call has no
//! `run_started_at` either, its take-up may hold the user's wait.
//!
//! The turn's `result` forgets the calls left open, except the calls of a
//! sub-agent (`parent_tool_use_id`): a background `Task` outlives the turn that
//! launched it, its calls are timed when they end. A sub-agent call that never
//! ends is kept until the session's clock goes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use chrono::{DateTime, Utc};

use super::types::ChatEvent;

/// Seconds since the epoch, the milliseconds as the fraction (the chat wire's unit).
fn seconds(at: DateTime<Utc>) -> f64 {
    crate::api::ws_chat_handler::wire_seconds(at)
}

#[derive(Default)]
struct Call {
    parent: Option<String>,
    called_at: Option<DateTime<Utc>>,
    started_at: Option<DateTime<Utc>>,
    permission_requested_at: Option<DateTime<Utc>>,
    /// The answer: when, and whether the tool may run.
    permission_resolved: Option<(DateTime<Utc>, bool)>,
    /// The [`ToolClock::decided`] that placed `permission_resolved`, if one did.
    resolved_by: Option<DecisionMark>,
    /// A question to the user: its answer is the result, the call never "runs".
    question: bool,
    incomplete: bool,
}

#[derive(Default)]
struct State {
    calls: HashMap<String, Call>,
    /// Permission (or question) request id → tool call id, while the call lasts.
    requests: HashMap<String, String>,
    /// The last [`DecisionMark`] handed out.
    marks: u64,
}

/// What one [`ToolClock::decided`] placed on the clock, so that
/// [`ToolClock::undecided`] takes back that answer and no other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecisionMark(u64);

/// The times of the tool calls of one session, until they are over.
#[derive(Default)]
pub(crate) struct ToolClock {
    state: Mutex<State>,
}

fn registry() -> &'static Mutex<HashMap<String, Weak<ToolClock>>> {
    static CLOCKS: OnceLock<Mutex<HashMap<String, Weak<ToolClock>>>> = OnceLock::new();
    CLOCKS.get_or_init(Default::default)
}

impl ToolClock {
    /// The clock of a session: every door of one session shares it. It lives as
    /// long as one of them holds it.
    pub(crate) fn for_session(session_id: &str) -> Arc<ToolClock> {
        let mut clocks = registry().lock().unwrap_or_else(|e| e.into_inner());
        clocks.retain(|_, clock| clock.strong_count() > 0);
        if let Some(clock) = clocks.get(session_id).and_then(Weak::upgrade) {
            return clock;
        }
        let clock = Arc::new(ToolClock::default());
        clocks.insert(session_id.to_string(), Arc::downgrade(&clock));
        clock
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The engine takes the call up (PreToolUse). The first take-up counts.
    pub(crate) fn taken_up(&self, id: &str, at: DateTime<Utc>) {
        self.state()
            .calls
            .entry(id.to_string())
            .or_default()
            .started_at
            .get_or_insert(at);
    }

    /// The user answered `request_id`, at `at`; call it BEFORE the answer is handed
    /// to the engine. The first answer counts: the mark is returned only when this
    /// answer is the one the clock keeps, for [`ToolClock::undecided`].
    pub(crate) fn decided(
        &self,
        request_id: &str,
        allow: bool,
        at: DateTime<Utc>,
    ) -> Option<DecisionMark> {
        let mut state = self.state();
        let Some(call_id) = state.requests.get(request_id).cloned() else {
            tracing::debug!(
                request_id,
                "tool clock: an answer to a request it does not know"
            );
            return None;
        };
        state.marks += 1;
        let mark = DecisionMark(state.marks);
        let call = state.calls.get_mut(&call_id)?;
        if call.permission_resolved.is_some() {
            return None;
        }
        call.permission_resolved = Some((at, allow));
        call.resolved_by = Some(mark);
        Some(mark)
    }

    /// The answer `mark` placed on `request_id` could not be handed over: it is
    /// pending again. Any other answer (an earlier one, one the engine took) stays.
    pub(crate) fn undecided(&self, request_id: &str, mark: Option<DecisionMark>) {
        let Some(mark) = mark else { return };
        let mut state = self.state();
        let Some(call_id) = state.requests.get(request_id).cloned() else {
            tracing::debug!(
                request_id,
                "tool clock: a failed answer to a request it does not know"
            );
            return;
        };
        if let Some(call) = state.calls.get_mut(&call_id) {
            if call.resolved_by == Some(mark) {
                call.permission_resolved = None;
                call.resolved_by = None;
            }
        }
    }

    /// Note what `event` says about a call; when it ends one, that call's timing.
    pub(crate) fn observe(&self, event: &ChatEvent, at: DateTime<Utc>) -> Option<ChatEvent> {
        let mut state = self.state();
        match event {
            ChatEvent::ToolUse {
                id,
                parent_tool_use_id,
                ..
            } => {
                let call = state.calls.entry(id.clone()).or_default();
                call.called_at.get_or_insert(at);
                if call.parent.is_none() {
                    call.parent = parent_tool_use_id.clone();
                }
                None
            }
            ChatEvent::PermissionRequest {
                id, tool_use_id, ..
            } => {
                match tool_use_id {
                    Some(call_id) => {
                        let call = state.calls.entry(call_id.clone()).or_default();
                        call.permission_requested_at.get_or_insert(at);
                        state.requests.insert(id.clone(), call_id.clone());
                    }
                    // A request that names no call: any open call may have waited.
                    None => state.calls.values_mut().for_each(|c| c.incomplete = true),
                }
                None
            }
            ChatEvent::AskUserQuestion {
                id, tool_call_id, ..
            } if !tool_call_id.is_empty() => {
                let call = state.calls.entry(tool_call_id.clone()).or_default();
                call.permission_requested_at.get_or_insert(at);
                call.question = true;
                state.requests.insert(id.clone(), tool_call_id.clone());
                None
            }
            ChatEvent::PermissionDecision { id, allow, .. } => {
                if let Some(call_id) = state.requests.get(id).cloned() {
                    if let Some(call) = state.calls.get_mut(&call_id) {
                        call.permission_resolved.get_or_insert((at, *allow));
                    }
                }
                None
            }
            ChatEvent::ToolResult {
                id,
                parent_tool_use_id,
                ..
            } => end(&mut state, id, parent_tool_use_id, false, at),
            ChatEvent::ToolCancelled {
                id,
                parent_tool_use_id,
            } => end(&mut state, id, parent_tool_use_id, true, at),
            // The turn is over: a call without a result will not get one, except
            // the calls of a sub-agent, which may outlive the turn (background Task).
            ChatEvent::Result { .. } => {
                let State {
                    calls, requests, ..
                } = &mut *state;
                calls.retain(|_, call| call.parent.is_some());
                requests.retain(|_, call_id| calls.contains_key(call_id));
                None
            }
            _ => None,
        }
    }
}

/// The timing of the call `id`, which is over (once: the call is taken out).
fn end(
    state: &mut State,
    id: &str,
    parent: &Option<String>,
    cancelled: bool,
    at: DateTime<Utc>,
) -> Option<ChatEvent> {
    let call = state.calls.remove(id)?;
    state.requests.retain(|_, call_id| call_id != id);
    let asked = call.permission_requested_at.is_some();
    let run_started_at = match (asked, call.question, call.permission_resolved) {
        // A wait the clock may have missed: no time it can vouch for.
        _ if call.incomplete => None,
        (_, true, _) => None,
        (true, false, Some((resolved, true))) => Some(resolved),
        (true, false, _) => None,
        (false, false, _) => call.started_at,
    };
    Some(ChatEvent::ToolTiming {
        id: id.to_string(),
        called_at: call.called_at.map(seconds),
        started_at: call.started_at.map(seconds),
        permission_requested_at: call.permission_requested_at.map(seconds),
        permission_resolved_at: call.permission_resolved.map(|(at, _)| seconds(at)),
        permission_outcome: call
            .permission_resolved
            .map(|(_, allow)| if allow { "allowed" } else { "denied" }.to_string()),
        run_started_at: run_started_at.map(seconds),
        ended_at: seconds(at),
        cancelled,
        incomplete: call.incomplete,
        parent_tool_use_id: parent.clone().or(call.parent),
    })
}

/// A PreToolUse callback that tells the clock when the engine takes a call up,
/// then answers what the callback it wraps answers (or nothing, alone).
///
/// It WRAPS the table's PreToolUse callbacks rather than standing beside them: an
/// engine that calls only some of a table's callbacks (the scripted CLI of the
/// tests calls the first) still reaches the clock, and the first take-up counts.
pub(crate) struct ToolClockHook {
    clock: Arc<ToolClock>,
    inner: Option<Arc<dyn nexus_claude::HookCallback>>,
}

/// Puts the session's clock on every PreToolUse callback of `table`, or adds it
/// alone when the table has none for every tool (a runner session).
pub(crate) fn clock_the_table(table: &mut super::agent_hooks::HookTable, clock: Arc<ToolClock>) {
    let matchers = table.entry("PreToolUse".to_string()).or_default();
    for matcher in matchers.iter_mut() {
        for hook in matcher.hooks.iter_mut() {
            *hook = Arc::new(ToolClockHook {
                clock: Arc::clone(&clock),
                inner: Some(Arc::clone(hook)),
            });
        }
    }
    if matchers
        .iter()
        .all(|m| m.matcher.is_some() || m.hooks.is_empty())
    {
        matchers.push(nexus_claude::HookMatcher {
            matcher: None,
            hooks: vec![Arc::new(ToolClockHook { clock, inner: None })],
        });
    }
}

#[async_trait::async_trait]
impl nexus_claude::HookCallback for ToolClockHook {
    async fn execute(
        &self,
        input: &nexus_claude::HookInput,
        tool_use_id: Option<&str>,
        context: &nexus_claude::HookContext,
    ) -> std::result::Result<nexus_claude::HookJSONOutput, nexus_claude::SdkError> {
        if let (nexus_claude::HookInput::PreToolUse(_), Some(id)) = (input, tool_use_id) {
            self.clock.taken_up(id, Utc::now());
        }
        match &self.inner {
            Some(inner) => inner.execute(input, tool_use_id, context).await,
            None => Ok(nexus_claude::HookJSONOutput::Sync(
                nexus_claude::SyncHookJSONOutput {
                    continue_: Some(true),
                    ..Default::default()
                },
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use uuid::Uuid;

    fn at(ms: i64) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(ms).unwrap()
    }

    fn tool_use(id: &str) -> ChatEvent {
        ChatEvent::ToolUse {
            id: id.into(),
            tool: "Bash".into(),
            input: json!({}),
            parent_tool_use_id: None,
            category: None,
            canonical: None,
        }
    }

    fn asks(request: &str, call: Option<&str>) -> ChatEvent {
        ChatEvent::PermissionRequest {
            id: request.into(),
            tool: "Bash".into(),
            input: json!({"command": "ls"}),
            parent_tool_use_id: None,
            category: None,
            canonical: None,
            tool_use_id: call.map(str::to_string),
        }
    }

    fn decision(request: &str, allow: bool) -> ChatEvent {
        ChatEvent::PermissionDecision {
            id: request.into(),
            allow,
            scope: None,
            rule: None,
        }
    }

    fn result(id: &str) -> ChatEvent {
        ChatEvent::ToolResult {
            id: id.into(),
            result: json!("ok"),
            is_error: false,
            parent_tool_use_id: None,
        }
    }

    fn cancelled(id: &str) -> ChatEvent {
        ChatEvent::ToolCancelled {
            id: id.into(),
            parent_tool_use_id: None,
        }
    }

    fn fields(event: ChatEvent) -> Value {
        assert_eq!(event.event_type(), "tool_timing");
        serde_json::to_value(event).unwrap()
    }

    #[test]
    fn a_call_without_permission_runs_from_its_take_up_to_its_result() {
        let clock = ToolClock::default();
        assert!(clock.observe(&tool_use("t1"), at(1_000)).is_none());
        clock.taken_up("t1", at(1_250));
        let t = fields(clock.observe(&result("t1"), at(1_900)).unwrap());
        assert_eq!(t["id"], "t1");
        assert_eq!(t["called_at"], 1.0);
        assert_eq!(t["started_at"], 1.25);
        assert_eq!(t["run_started_at"], 1.25);
        assert_eq!(t["ended_at"], 1.9);
        assert!(t.get("permission_requested_at").is_none(), "{t}");
        assert!(
            t.get("cancelled").is_none() && t.get("incomplete").is_none(),
            "{t}"
        );
    }

    #[test]
    fn the_wait_for_an_allowed_permission_is_not_run_time() {
        let clock = ToolClock::default();
        clock.observe(&tool_use("t1"), at(1_000));
        clock.taken_up("t1", at(1_100));
        clock.observe(&asks("req-1", Some("t1")), at(1_200));
        clock.observe(&decision("req-1", true), at(9_200));
        let t = fields(clock.observe(&result("t1"), at(9_500)).unwrap());
        assert_eq!(t["started_at"], 1.1);
        assert_eq!(t["permission_requested_at"], 1.2);
        assert_eq!(t["permission_resolved_at"], 9.2);
        assert_eq!(t["permission_outcome"], "allowed");
        assert_eq!(t["run_started_at"], 9.2);
        assert_eq!(t["ended_at"], 9.5);
    }

    #[test]
    fn a_denied_permission_is_not_a_run() {
        let clock = ToolClock::default();
        clock.observe(&tool_use("t1"), at(1_000));
        clock.taken_up("t1", at(1_100));
        clock.observe(&asks("req-1", Some("t1")), at(1_200));
        clock.observe(&decision("req-1", false), at(3_000));
        // The engine answers the denial with an error result.
        let t = fields(clock.observe(&result("t1"), at(3_010)).unwrap());
        assert_eq!(t["permission_outcome"], "denied");
        assert_eq!(t["permission_resolved_at"], 3.0);
        assert!(t.get("run_started_at").is_none(), "{t}");
    }

    #[test]
    fn without_a_take_up_the_run_start_is_absent_not_the_announcement() {
        // An engine that runs no host hook (remote cwd, Codex, ACP): only the
        // announcement and the result are known.
        let clock = ToolClock::default();
        clock.observe(&tool_use("t1"), at(1_000));
        let t = fields(clock.observe(&result("t1"), at(5_000)).unwrap());
        assert_eq!(t["called_at"], 1.0);
        assert!(t.get("started_at").is_none(), "{t}");
        assert!(t.get("run_started_at").is_none(), "{t}");
    }

    #[test]
    fn a_permission_goes_to_the_call_it_names() {
        let clock = ToolClock::default();
        // Two calls of the same tool with the same input: only the id tells them apart.
        clock.observe(&tool_use("a"), at(1_000));
        clock.observe(&tool_use("b"), at(1_001));
        clock.taken_up("a", at(1_002));
        clock.taken_up("b", at(1_003));
        clock.observe(&asks("r", Some("a")), at(1_500));
        let b = fields(clock.observe(&result("b"), at(2_000)).unwrap());
        assert!(b.get("permission_requested_at").is_none(), "{b}");
        assert_eq!(b["run_started_at"], 1.003);
        let a = fields(clock.observe(&result("a"), at(2_100)).unwrap());
        assert_eq!(a["permission_requested_at"], 1.5);
    }

    #[test]
    fn a_request_naming_no_call_marks_the_open_calls_incomplete() {
        let clock = ToolClock::default();
        clock.observe(&tool_use("a"), at(1_000));
        clock.taken_up("a", at(1_050));
        clock.observe(&asks("r", None), at(1_100));
        let a = fields(clock.observe(&result("a"), at(2_000)).unwrap());
        assert_eq!(a["incomplete"], true);
        assert_eq!(a["started_at"], 1.05);
        // The take-up may hold the user's wait: absent rather than estimated.
        assert!(a.get("run_started_at").is_none(), "{a}");
    }

    #[test]
    fn an_answer_noted_before_the_hand_over_survives_a_result_that_overtakes_its_event() {
        let clock = ToolClock::default();
        clock.observe(&tool_use("t1"), at(1_000));
        clock.observe(&asks("req-1", Some("t1")), at(1_100));
        clock.decided("req-1", true, at(2_000));
        // The tool was fast: its result comes before the decision event.
        let t = fields(clock.observe(&result("t1"), at(2_050)).unwrap());
        assert_eq!(t["permission_resolved_at"], 2.0);
        assert_eq!(t["run_started_at"], 2.0);
        // The late decision event changes nothing (the call is over).
        assert!(clock.observe(&decision("req-1", true), at(2_100)).is_none());
    }

    #[test]
    fn an_answer_that_could_not_be_handed_over_is_pending_again() {
        let clock = ToolClock::default();
        clock.observe(&tool_use("t1"), at(1_000));
        clock.observe(&asks("req-1", Some("t1")), at(1_100));
        let mark = clock.decided("req-1", true, at(2_000));
        assert!(mark.is_some());
        clock.undecided("req-1", mark);
        clock.decided("req-1", true, at(3_000));
        let t = fields(clock.observe(&result("t1"), at(3_500)).unwrap());
        assert_eq!(t["permission_resolved_at"], 3.0);
    }

    #[test]
    fn a_second_answer_the_engine_refuses_does_not_erase_the_first() {
        let clock = ToolClock::default();
        clock.observe(&tool_use("t1"), at(1_000));
        clock.taken_up("t1", at(1_050));
        clock.observe(&asks("req-1", Some("t1")), at(1_100));
        let first = clock.decided("req-1", true, at(2_000));
        // A double click / a second tab: the engine refuses the second answer.
        let second = clock.decided("req-1", false, at(2_300));
        assert!(second.is_none(), "the first answer counts");
        clock.undecided("req-1", second);
        let t = fields(clock.observe(&result("t1"), at(2_500)).unwrap());
        assert!(first.is_some());
        assert_eq!(t["permission_resolved_at"], 2.0, "{t}");
        assert_eq!(t["permission_outcome"], "allowed", "{t}");
        assert_eq!(t["run_started_at"], 2.0, "{t}");
    }

    #[test]
    fn an_answer_to_an_unknown_request_marks_nothing() {
        let clock = ToolClock::default();
        assert!(clock.decided("nope", true, at(1)).is_none());
        clock.undecided("nope", None);
    }

    #[test]
    fn a_sub_agent_call_outlives_the_turn_that_launched_it() {
        let clock = ToolClock::default();
        clock.observe(
            &ChatEvent::ToolUse {
                id: "sub".into(),
                tool: "Bash".into(),
                input: json!({}),
                parent_tool_use_id: Some("task".into()),
                category: None,
                canonical: None,
            },
            at(1_000),
        );
        clock.taken_up("sub", at(1_010));
        clock.observe(&asks("req-s", Some("sub")), at(1_020));
        clock.observe(&tool_use("top"), at(1_100));
        clock.observe(
            &serde_json::from_value::<ChatEvent>(json!({
                "type": "result", "session_id": "s", "duration_ms": 0,
                "subtype": "success", "is_error": false
            }))
            .unwrap(),
            at(2_000),
        );
        assert!(
            clock.observe(&result("top"), at(2_100)).is_none(),
            "cleared"
        );
        assert!(
            clock.decided("req-s", true, at(3_000)).is_some(),
            "its request kept"
        );
        let t = fields(clock.observe(&result("sub"), at(4_000)).unwrap());
        assert_eq!(t["parent_tool_use_id"], "task", "{t}");
        assert_eq!(t["called_at"], 1.0, "{t}");
        assert_eq!(t["run_started_at"], 3.0, "{t}");
    }

    #[test]
    fn an_interrupt_during_the_permission_wait_is_a_cancelled_call_that_never_ran() {
        let clock = ToolClock::default();
        clock.observe(&tool_use("t1"), at(1_000));
        clock.taken_up("t1", at(1_050));
        clock.observe(&asks("req-1", Some("t1")), at(1_100));
        let t = fields(clock.observe(&cancelled("t1"), at(4_000)).unwrap());
        assert_eq!(t["cancelled"], true);
        assert_eq!(t["ended_at"], 4.0);
        assert!(t.get("permission_resolved_at").is_none(), "{t}");
        assert!(t.get("run_started_at").is_none(), "{t}");
    }

    #[test]
    fn a_question_answered_by_the_result_has_no_run_start() {
        let clock = ToolClock::default();
        clock.observe(&tool_use("q"), at(1_000));
        clock.observe(
            &ChatEvent::AskUserQuestion {
                id: "req".into(),
                tool_call_id: "q".into(),
                questions: json!([]),
                input: json!({}),
                parent_tool_use_id: None,
                synthetic: None,
            },
            at(1_100),
        );
        clock.observe(&decision("req", true), at(1_101));
        let t = fields(clock.observe(&result("q"), at(5_000)).unwrap());
        assert_eq!(t["permission_requested_at"], 1.1);
        assert!(t.get("run_started_at").is_none(), "{t}");
    }

    #[test]
    fn one_timing_per_call_and_none_after_the_end_of_a_turn() {
        let clock = ToolClock::default();
        assert!(clock.observe(&result("never-seen"), at(1)).is_none());
        clock.observe(&tool_use("t"), at(2));
        assert!(clock.observe(&result("t"), at(3)).is_some());
        assert!(
            clock.observe(&result("t"), at(4)).is_none(),
            "a second result"
        );
        assert!(
            clock.observe(&cancelled("t"), at(5)).is_none(),
            "a late cancel"
        );
        clock.observe(&tool_use("u"), at(6));
        clock.observe(
            &serde_json::from_value::<ChatEvent>(json!({
                "type": "result", "session_id": "s", "duration_ms": 0,
                "subtype": "success", "is_error": false
            }))
            .unwrap(),
            at(7),
        );
        assert!(clock.observe(&result("u"), at(8)).is_none());
    }

    #[test]
    fn every_door_of_a_session_shares_one_clock() {
        let sid = Uuid::new_v4().to_string();
        let first = ToolClock::for_session(&sid);
        assert!(Arc::ptr_eq(&first, &ToolClock::for_session(&sid)));
        let other = ToolClock::for_session(&Uuid::new_v4().to_string());
        assert!(!Arc::ptr_eq(&first, &other));
    }

    struct Says(&'static str);

    #[async_trait::async_trait]
    impl nexus_claude::HookCallback for Says {
        async fn execute(
            &self,
            _input: &nexus_claude::HookInput,
            _tool_use_id: Option<&str>,
            _context: &nexus_claude::HookContext,
        ) -> std::result::Result<nexus_claude::HookJSONOutput, nexus_claude::SdkError> {
            Ok(nexus_claude::HookJSONOutput::Sync(
                nexus_claude::SyncHookJSONOutput {
                    reason: Some(self.0.to_string()),
                    ..Default::default()
                },
            ))
        }
    }

    fn pre_tool_use(tool: &str) -> nexus_claude::HookInput {
        nexus_claude::HookInput::PreToolUse(nexus_claude::PreToolUseHookInput {
            session_id: "s".into(),
            transcript_path: String::new(),
            cwd: "/w".into(),
            permission_mode: None,
            tool_name: tool.into(),
            tool_input: json!({"command": "ls"}),
            agent_id: None,
            agent_type: None,
        })
    }

    #[tokio::test]
    async fn the_first_pre_tool_use_callback_clocks_the_call_and_still_answers() {
        let clock = Arc::new(ToolClock::default());
        let mut table = super::super::agent_hooks::HookTable::new();
        table.insert(
            "PreToolUse".into(),
            vec![nexus_claude::HookMatcher {
                matcher: None,
                hooks: vec![Arc::new(Says("skill context"))],
            }],
        );
        clock_the_table(&mut table, Arc::clone(&clock));
        assert_eq!(table["PreToolUse"].len(), 1, "wrapped, not added");
        let first = &table["PreToolUse"][0].hooks[0];
        let ctx = nexus_claude::HookContext { signal: None };
        let out = first
            .execute(&pre_tool_use("Bash"), Some("t1"), &ctx)
            .await
            .unwrap();
        let nexus_claude::HookJSONOutput::Sync(sync) = out else {
            panic!("a sync answer")
        };
        assert_eq!(sync.reason.as_deref(), Some("skill context"));
        let t = fields(clock.observe(&result("t1"), Utc::now()).unwrap());
        assert!(t["started_at"].is_f64(), "{t}");
    }

    #[tokio::test]
    async fn a_table_without_pre_tool_use_gets_the_clock_alone() {
        let clock = Arc::new(ToolClock::default());
        let mut table = super::super::agent_hooks::HookTable::new();
        clock_the_table(&mut table, Arc::clone(&clock));
        let ctx = nexus_claude::HookContext { signal: None };
        table["PreToolUse"][0].hooks[0]
            .execute(&pre_tool_use("Read"), Some("t2"), &ctx)
            .await
            .unwrap();
        let t = fields(clock.observe(&result("t2"), Utc::now()).unwrap());
        assert!(t["started_at"].is_f64(), "{t}");
    }

    /// Legacy engine: the answer is on the clock once `send_permission_response`
    /// returns, BEFORE the decision event goes anywhere (the CLI already has it).
    #[tokio::test]
    async fn a_legacy_permission_answer_is_on_the_clock_when_it_reaches_the_cli() {
        let state = crate::test_helpers::mock_app_state();
        let manager = crate::chat::manager::ChatManager::new_without_memory(
            state.neo4j,
            state.meili,
            crate::chat::manager::test_support::chat_config(),
        );
        let sid = Uuid::new_v4().to_string();
        let client = nexus_claude::InteractiveClient::new(nexus_claude::ClaudeCodeOptions {
            model: Some("test".into()),
            cli_path: Some("/nonexistent/claude".into()),
            ..Default::default()
        })
        .unwrap();
        let (mut stdin_rx, _queue) =
            crate::chat::manager::test_support::insert_session_with_client(
                &manager,
                &sid,
                client,
                false,
                &["req-1"],
            )
            .await
            .unwrap();
        let clock = ToolClock::for_session(&sid);
        clock.observe(&tool_use("t1"), Utc::now());
        clock.observe(&asks("req-1", Some("t1")), Utc::now());

        manager
            .send_permission_response(&sid, "req-1", true)
            .await
            .unwrap();
        let written = stdin_rx.try_recv().expect("the CLI got the answer");
        assert!(written.contains("req-1"), "{written}");
        let t = fields(clock.observe(&result("t1"), Utc::now()).unwrap());
        assert_eq!(t["permission_outcome"], "allowed", "{t}");
        assert!(t["run_started_at"].is_f64(), "{t}");
    }
}
