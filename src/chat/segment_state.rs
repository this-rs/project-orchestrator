//! What a session is doing in its CURRENT segment, deduced — never asked of a model.
//!
//! `SessionActivity` used to say `streaming: true` and nothing else: a session parked for
//! 4 h 32 in a sub-agent call that never came back (session `2c7b913b`, 30/10/2026) looked
//! exactly like one writing an answer. This module is the PURE MODEL that tells those
//! cases apart. It takes the tail of the session's typed events ([`TailEvent`], built from
//! the `ChatEvent`s it emitted), what the live manager knows right now ([`Activity`]) and a
//! clock value, and returns one [`SessionState`] with the [`Reason`] that decided it: which
//! rule, which event, and how old that event was.
//!
//! * **Pure.** No provider, graph, network or model is reached from here; the caller
//!   records the tail (`chat::segment_tap`) and passes the clock in. The same tail, the
//!   same activity and the same clock give the same verdict, on any machine: integer
//!   milliseconds only, no wall clock read inside, rules applied in one fixed order.
//! * **No evidence is not `Done`.** `Done` releases the session (`segment_tap`, the
//!   release sweep), so it needs a positive proof: a turn that ENDED (`result`) after the
//!   last input, cleanly, with nothing addressed to the human, no watch, no queue, nothing
//!   planned. A tail that starts mid-conversation (a tap installed after a restart, a lag)
//!   proves nothing and gives `Reporting` — the session stays live.
//! * **`Reporting`, `Watching` and `WaitingOnPerson` never give `Done`.** A turn that
//!   ended on text addressed to the human is a report: the human has not read it yet. A
//!   live monitor or background command is a watch. A permission, a question, a secret or
//!   a queued message is a person the session waits on.
//!
//! # Rules, first match wins
//!
//! 1. a permission is pending, a question (`ask_user_question`) or a secret request is
//!    open, or the last event says a person must act → [`SessionState::WaitingOnPerson`];
//! 2. streaming:
//!    * no event for [`STALLED_AFTER_MS`] or more → [`SessionState::Stalled`];
//!    * a tool call has no result yet → [`SessionState::Working`];
//!    * the last event is text being written → [`SessionState::Writing`];
//!    * otherwise → [`SessionState::Working`];
//! 3. not streaming:
//!    * a message is queued → [`SessionState::WaitingOnPerson`];
//!    * a turn is planned (`auto_continue`, `retrying` after the last end) →
//!      [`SessionState::Working`];
//!    * a monitor or background command runs → [`SessionState::Watching`];
//!    * the last turn has no proven end, failed, was interrupted, or ended on text
//!      addressed to the human → [`SessionState::Reporting`];
//!    * it ended less than [`SETTLE_MS`] ago → [`SessionState::Working`] (its end-of-turn
//!      work may still plan a turn);
//!    * otherwise → [`SessionState::Done`].
//!
//! # The `Stalled` threshold: 30 minutes, and why
//!
//! `Stalled` is a LABEL, it closes nothing and ends nothing (the watchdogs that act are
//! `start_lost_tool_result_watchdog` and the agent runtime's tail watchdog). A false
//! `Stalled` on a legitimately long call is therefore a wrong word on screen, a missed one
//! is the incident; the threshold is chosen to make the first impossible for every call
//! this harness bounds, and the second short:
//!
//! * the longest bounded foreground call is a `Bash` at its 10-minute ceiling (600 s), the
//!   longest bounded PO MCP hop is 300 s (`LOST_TOOL_RESULT_GRACE_MCP` = 600 s already
//!   calls one twice that late "lost"): 30 min is three times the longest of them;
//! * 30 min (1 800 s) is the harness's own convention for "silent = dead": the idle
//!   session timeout (`chat.session_timeout_secs`) and the background-task death detector
//!   (`BACKGROUND_TASK_IDLE_DEATH_SECS`) both use it;
//! * it stays under the 1 h `LOST_TOOL_RESULT_GRACE_COMMAND`, so the label shows before
//!   the command watchdog acts, and the 2c7b913b replay (40 min of silence) is `Stalled`.
//!
//! The trade-off, stated: a call with no bound of its own that legitimately says nothing
//! for more than 30 min — a sub-agent (`Agent`) whose CHILD streams in its own session
//! while the parent's tail stays silent, a hand-written `sleep 3600` — is labelled
//! `Stalled` while it is in fact working. The label is honest about what it measures (no
//! event for that long), and the child's own row still says `Working`. A permission
//! pending is never `Stalled`: waiting on a person is rule 1.
//!
//! # The settle window: 30 seconds
//!
//! A turn's end is followed by its end-of-turn work (`post_stream`), which may itself plan
//! a turn (an auto-continue after a turn limit, an objective reminder). `Done` therefore
//! waits [`SETTLE_MS`] = 30 s (`post_stream::POST_STREAM_STEP_BUDGET`, the budget of one
//! such step) after the `result` before it is given; inside it the session is `Working`.
//! This is not a closing timer: past it, every other condition of `Done` must still hold.

use std::collections::{BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::chat::types::ChatEvent;

/// No event for this long while streaming is [`SessionState::Stalled`] (see the module
/// doc for the choice).
pub const STALLED_AFTER_MS: u64 = 30 * 60 * 1_000;

/// How long after a turn's end `Done` waits for its end-of-turn work (module doc).
pub const SETTLE_MS: u64 = 30 * 1_000;

/// Most events a [`Tail`] keeps. Consecutive text deltas are folded into one entry, so
/// this is 128 meaningful events, not 128 tokens.
pub const TAIL_CAP: usize = 128;

/// What a session is doing in its current segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    /// A turn runs: a tool is in flight, the model thinks, a turn is planned.
    Working,
    /// Text is being written now.
    Writing,
    /// The turn ended on something addressed to the human, nothing planned.
    Reporting,
    /// A monitor or a background command runs between turns.
    Watching,
    /// A permission, a question, a secret or a queued message waits on a person.
    WaitingOnPerson,
    /// The turn ended, its output went to whoever spawned the session (or there was
    /// none), no watch, no queue, nothing planned: the mission is over.
    Done,
    /// Streaming, but no event for [`STALLED_AFTER_MS`] or more.
    Stalled,
}

impl SessionState {
    /// Stable wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Working => "working",
            Self::Writing => "writing",
            Self::Reporting => "reporting",
            Self::Watching => "watching",
            Self::WaitingOnPerson => "waiting_on_person",
            Self::Done => "done",
            Self::Stalled => "stalled",
        }
    }
}

/// Which rule decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    PermissionPending,
    QuestionOpen,
    SecretRequested,
    PersonMustAct,
    NoEventFor,
    ToolInFlight,
    TextStreaming,
    TurnRunning,
    MessageQueued,
    TurnPlanned,
    BackgroundRunning,
    NoProvenEnd,
    TurnFailed,
    TurnInterrupted,
    AnsweredHuman,
    TurnSettling,
    TurnCompleted,
}

/// Why the state is what it is: the rule, the event that decided (its wire name, when
/// one did) and that event's age at the clock value given.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reason {
    pub rule: Rule,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    pub age_ms: u64,
}

/// The verdict: one state and its reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    pub state: SessionState,
    pub reason: Reason,
}

/// Who the session's output is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Addressee {
    /// A person reads it (a root conversation). The default: unknown is a person, so
    /// an unknown session is never `Done` for want of a reader.
    #[default]
    Human,
    /// The session was spawned (runner, delegation, sub-conversation...): its answer
    /// goes back to its spawner, not to a person.
    Spawner,
}

/// What the live manager knows of the session right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Activity {
    pub streaming: bool,
    pub pending_permissions: usize,
    pub monitors: usize,
    pub bash_tasks: usize,
    pub queued_messages: usize,
    pub addressee: Addressee,
}

/// How a turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnEnd {
    Completed,
    Failed,
    Interrupted,
}

/// What one event says, for this model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Signal {
    /// A message that starts a turn (the user's, a team message).
    Input,
    /// Text of the answer; `top_level` is false for a sub-agent's (sidechain) text.
    Text {
        top_level: bool,
    },
    /// A streaming text delta.
    Delta {
        top_level: bool,
    },
    /// The session works without saying anything to anyone (thinking, a timing...).
    Progress,
    ToolStart {
        id: String,
    },
    ToolEnd {
        id: String,
    },
    QuestionAsked {
        id: String,
    },
    SecretAsked {
        id: String,
    },
    SecretResolved {
        id: String,
    },
    /// The harness says a person must act (`provider_notice`, `action_required`).
    PersonMustAct,
    /// A turn ended (`result`), with or without a final text.
    Ended {
        how: TurnEnd,
        text: bool,
    },
    /// An error addressed to the human outside a `result`.
    Failure,
    /// Another turn is on its way (`auto_continue`, `retrying`).
    Planned,
}

/// One entry of the tail: when (milliseconds since the epoch), what (the event's wire
/// name) and what it says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailEvent {
    pub at_ms: u64,
    pub event: &'static str,
    pub signal: Signal,
}

/// What an event says for this model, `None` for an event that says nothing about what
/// the session does (a queue snapshot, a mode change, a background-task table...).
pub fn signal_of(event: &ChatEvent) -> Option<Signal> {
    Some(match event {
        ChatEvent::UserMessage { .. } | ChatEvent::TeamMessage { .. } => Signal::Input,
        ChatEvent::AssistantText {
            parent_tool_use_id, ..
        } => Signal::Text {
            top_level: parent_tool_use_id.is_none(),
        },
        ChatEvent::StreamDelta {
            parent_tool_use_id, ..
        } => Signal::Delta {
            top_level: parent_tool_use_id.is_none(),
        },
        ChatEvent::ToolUse { id, .. } => Signal::ToolStart { id: id.clone() },
        ChatEvent::ToolResult { id, .. } | ChatEvent::ToolCancelled { id, .. } => {
            Signal::ToolEnd { id: id.clone() }
        }
        ChatEvent::AskUserQuestion { tool_call_id, .. } => Signal::QuestionAsked {
            id: tool_call_id.clone(),
        },
        ChatEvent::SecretRequest { id, .. } => Signal::SecretAsked { id: id.clone() },
        ChatEvent::SecretRequestResolved { id, .. } => Signal::SecretResolved { id: id.clone() },
        ChatEvent::ProviderNotice { severity, .. } if severity == "action_required" => {
            Signal::PersonMustAct
        }
        ChatEvent::Result {
            is_error,
            stop_reason,
            subtype,
            result_text,
            ..
        } => {
            // The two spellings the engines use: a `stop_reason` on the result, or a
            // `subtype` that names the interruption (`error_interrupted`).
            let how = if stop_reason.as_deref() == Some("interrupted")
                || subtype.contains("interrupted")
            {
                TurnEnd::Interrupted
            } else if *is_error || subtype.starts_with("error") {
                TurnEnd::Failed
            } else {
                TurnEnd::Completed
            };
            Signal::Ended {
                how,
                text: result_text.as_deref().is_some_and(|t| !t.trim().is_empty()),
            }
        }
        ChatEvent::Error {
            parent_tool_use_id: None,
            ..
        }
        | ChatEvent::SessionError { .. } => Signal::Failure,
        ChatEvent::AutoContinue { .. } | ChatEvent::Retrying { .. } => Signal::Planned,
        ChatEvent::Thinking { .. }
        | ChatEvent::ToolTiming { .. }
        | ChatEvent::ToolUseInputResolved { .. }
        | ChatEvent::BackgroundOutput { .. }
        | ChatEvent::CompactionStarted { .. }
        | ChatEvent::CompactBoundary { .. }
        | ChatEvent::ProviderNotice { .. }
        | ChatEvent::ProviderOutage { .. }
        | ChatEvent::PermissionRequest { .. }
        | ChatEvent::PermissionDecision { .. }
        | ChatEvent::Error { .. } => Signal::Progress,
        _ => return None,
    })
}

/// A bounded tail of a session's events, as the model reads it.
#[derive(Debug, Clone, Default)]
pub struct Tail {
    events: VecDeque<TailEvent>,
    /// When recording started: the age of an empty tail.
    observed_since_ms: u64,
}

impl Tail {
    /// An empty tail, recording from `now_ms`.
    pub fn starting_at(now_ms: u64) -> Self {
        Self {
            events: VecDeque::new(),
            observed_since_ms: now_ms,
        }
    }

    /// Records one event seen at `at_ms`. An event that says nothing is skipped; a delta
    /// that follows a delta of the same level only moves that entry's time.
    pub fn record(&mut self, event: &ChatEvent, at_ms: u64) {
        let Some(signal) = signal_of(event) else {
            return;
        };
        if let (Signal::Delta { top_level }, Some(last)) = (&signal, self.events.back_mut()) {
            if last.signal
                == (Signal::Delta {
                    top_level: *top_level,
                })
            {
                last.at_ms = last.at_ms.max(at_ms);
                return;
            }
        }
        self.events.push_back(TailEvent {
            at_ms,
            event: event.event_type(),
            signal,
        });
        self.trim();
    }

    /// Keeps the tail at [`TAIL_CAP`], evicting the oldest entries — but never the most
    /// recent [`Signal::Input`]: it is the proof a turn opened, and a turn longer than the
    /// cap would otherwise lose it (and `classify` would answer `NoProvenEnd` for want of
    /// it). When that Input is the front entry, the entry just after it is dropped instead.
    fn trim(&mut self) {
        while self.events.len() > TAIL_CAP {
            if self.events.iter().rposition(|e| e.signal == Signal::Input) == Some(0) {
                let _ = self.events.remove(1);
            } else {
                self.events.pop_front();
            }
        }
    }

    /// Puts in front of what was recorded the events that came BEFORE recording started
    /// (the stored thread, oldest first, with their times): those at or after
    /// `observed_since_ms` are left out, since the live recording has them. The cap keeps
    /// the newest.
    pub fn seed<'a>(&mut self, older: impl IntoIterator<Item = (&'a ChatEvent, u64)>) {
        let mut seeded = Tail::starting_at(self.observed_since_ms);
        for (event, at_ms) in older {
            if at_ms < self.observed_since_ms {
                seeded.record(event, at_ms);
            }
        }
        seeded.events.extend(self.events.drain(..));
        seeded.trim();
        self.events = seeded.events;
    }

    /// The events kept, oldest first.
    pub fn events(&self) -> Vec<TailEvent> {
        self.events.iter().cloned().collect()
    }

    /// When recording started.
    pub fn observed_since_ms(&self) -> u64 {
        self.observed_since_ms
    }

    /// [`classify`] on this tail.
    pub fn classify(&self, activity: &Activity, now_ms: u64) -> Verdict {
        let events: Vec<TailEvent> = self.events();
        classify(&events, self.observed_since_ms, activity, now_ms)
    }
}

fn verdict(
    state: SessionState,
    rule: Rule,
    at: Option<&TailEvent>,
    since: u64,
    now: u64,
) -> Verdict {
    Verdict {
        state,
        reason: Reason {
            rule,
            event: at.map(|e| e.event.to_string()),
            age_ms: now.saturating_sub(at.map_or(since, |e| e.at_ms)),
        },
    }
}

/// The state of a session (rules in the module doc). `tail` is oldest first;
/// `observed_since_ms` is when it started being recorded (the age of an empty tail).
pub fn classify(
    tail: &[TailEvent],
    observed_since_ms: u64,
    activity: &Activity,
    now_ms: u64,
) -> Verdict {
    let since = observed_since_ms;
    let last = tail.last();
    let v = |state, rule, at: Option<&TailEvent>| verdict(state, rule, at, since, now_ms);

    // 1. A person is waited on.
    if activity.pending_permissions > 0 {
        return v(SessionState::WaitingOnPerson, Rule::PermissionPending, last);
    }
    let mut ended_tools = BTreeSet::new();
    let mut resolved_secrets = BTreeSet::new();
    let mut open_tool: Option<&TailEvent> = None;
    // Walk back: a question or secret is open when nothing later closes it.
    for e in tail.iter().rev() {
        match &e.signal {
            Signal::ToolEnd { id } => {
                ended_tools.insert(id.as_str());
            }
            Signal::SecretResolved { id } => {
                resolved_secrets.insert(id.as_str());
            }
            Signal::QuestionAsked { id } if !ended_tools.contains(id.as_str()) => {
                return v(SessionState::WaitingOnPerson, Rule::QuestionOpen, Some(e));
            }
            Signal::SecretAsked { id } if !resolved_secrets.contains(id.as_str()) => {
                return v(
                    SessionState::WaitingOnPerson,
                    Rule::SecretRequested,
                    Some(e),
                );
            }
            Signal::ToolStart { id }
                if open_tool.is_none() && !ended_tools.contains(id.as_str()) =>
            {
                open_tool = Some(e);
            }
            // An interrupted turn closes what it had open: a question it asked never gets its
            // `tool_result` (measured 10/11/2026: none, and no `tool_cancelled`), so the end
            // of the turn is the only witness that nobody is waited on any more.
            Signal::Ended {
                how: TurnEnd::Interrupted,
                ..
            } => break,
            // A question answered as a user turn (a provider without native questions)
            // is closed by that input; nothing before an input is still open.
            Signal::Input => break,
            _ => {}
        }
    }
    if let Some(e) = last.filter(|e| e.signal == Signal::PersonMustAct) {
        return v(SessionState::WaitingOnPerson, Rule::PersonMustAct, Some(e));
    }

    // 2. A turn is streaming.
    if activity.streaming {
        let silent_for = now_ms.saturating_sub(last.map_or(since, |e| e.at_ms));
        if silent_for >= STALLED_AFTER_MS {
            return v(SessionState::Stalled, Rule::NoEventFor, last);
        }
        if let Some(e) = open_tool {
            return v(SessionState::Working, Rule::ToolInFlight, Some(e));
        }
        if let Some(e) =
            last.filter(|e| matches!(e.signal, Signal::Delta { .. } | Signal::Text { .. }))
        {
            return v(SessionState::Writing, Rule::TextStreaming, Some(e));
        }
        return v(SessionState::Working, Rule::TurnRunning, last);
    }

    // 3. Between turns.
    if activity.queued_messages > 0 {
        return v(SessionState::WaitingOnPerson, Rule::MessageQueued, last);
    }
    // The last turn: everything after the last input.
    let start = tail
        .iter()
        .rposition(|e| e.signal == Signal::Input)
        .map_or(0, |i| i + 1);
    let turn = &tail[start..];
    // The last end of the turn, with its index and what it says.
    let end = turn
        .iter()
        .enumerate()
        .rev()
        .find_map(|(i, e)| match e.signal {
            Signal::Ended { how, text } => Some((i, e, how, text)),
            _ => None,
        });
    if let Some(e) = turn[end.map_or(0, |(i, ..)| i + 1)..]
        .iter()
        .rev()
        .find(|e| e.signal == Signal::Planned)
    {
        return v(SessionState::Working, Rule::TurnPlanned, Some(e));
    }
    if activity.monitors + activity.bash_tasks > 0 {
        return v(SessionState::Watching, Rule::BackgroundRunning, last);
    }
    // No input in the tail means it may start mid-turn: an end is only proven when
    // the tail holds the input that opened its turn.
    let input_seen = start > 0;
    let Some((_, end_event, how, text)) = end.filter(|_| input_seen) else {
        return v(SessionState::Reporting, Rule::NoProvenEnd, last);
    };
    if let Some(e) = turn.iter().rev().find(|e| e.signal == Signal::Failure) {
        return v(SessionState::Reporting, Rule::TurnFailed, Some(e));
    }
    match how {
        TurnEnd::Failed => return v(SessionState::Reporting, Rule::TurnFailed, Some(end_event)),
        TurnEnd::Interrupted => {
            return v(
                SessionState::Reporting,
                Rule::TurnInterrupted,
                Some(end_event),
            )
        }
        TurnEnd::Completed => {}
    }
    if activity.addressee == Addressee::Human {
        let human_text = text
            || turn.iter().any(|e| {
                matches!(
                    e.signal,
                    Signal::Text { top_level: true } | Signal::Delta { top_level: true }
                )
            });
        if human_text {
            return v(
                SessionState::Reporting,
                Rule::AnsweredHuman,
                Some(end_event),
            );
        }
    }
    if now_ms.saturating_sub(end_event.at_ms) < SETTLE_MS {
        return v(SessionState::Working, Rule::TurnSettling, Some(end_event));
    }
    v(SessionState::Done, Rule::TurnCompleted, Some(end_event))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: u64 = 60 * 1_000;
    /// An arbitrary clock origin: the model never reads a wall clock.
    const T0: u64 = 1_761_800_000_000;

    fn ev(at_ms: u64, event: &'static str, signal: Signal) -> TailEvent {
        TailEvent {
            at_ms,
            event,
            signal,
        }
    }
    fn input(at: u64) -> TailEvent {
        ev(at, "user_message", Signal::Input)
    }
    fn tool(at: u64, id: &str) -> TailEvent {
        ev(at, "tool_use", Signal::ToolStart { id: id.into() })
    }
    fn result(at: u64, id: &str) -> TailEvent {
        ev(at, "tool_result", Signal::ToolEnd { id: id.into() })
    }
    fn timing(at: u64) -> TailEvent {
        ev(at, "tool_timing", Signal::Progress)
    }
    fn text(at: u64) -> TailEvent {
        ev(at, "assistant_text", Signal::Text { top_level: true })
    }
    fn delta(at: u64) -> TailEvent {
        ev(at, "stream_delta", Signal::Delta { top_level: true })
    }
    fn end(at: u64, how: TurnEnd, text: bool) -> TailEvent {
        ev(at, "result", Signal::Ended { how, text })
    }
    fn streaming() -> Activity {
        Activity {
            streaming: true,
            ..Default::default()
        }
    }
    fn idle() -> Activity {
        Activity::default()
    }
    fn spawned() -> Activity {
        Activity {
            addressee: Addressee::Spawner,
            ..Default::default()
        }
    }
    fn state(tail: &[TailEvent], activity: Activity, now: u64) -> SessionState {
        classify(tail, T0, &activity, now).state
    }

    /// One row per state at least: tail, activity, clock → state and rule.
    #[test]
    fn the_table_of_states() {
        let tool_running = vec![input(T0), tool(T0 + 1_000, "a")];
        let writing = vec![input(T0), delta(T0 + 1_000)];
        let reported = vec![
            input(T0),
            text(T0 + 1_000),
            end(T0 + 2_000, TurnEnd::Completed, true),
        ];
        let silent_end = vec![
            input(T0),
            tool(T0 + 1_000, "a"),
            result(T0 + 2_000, "a"),
            end(T0 + 3_000, TurnEnd::Completed, false),
        ];
        let question = vec![
            input(T0),
            ev(
                T0 + 1_000,
                "ask_user_question",
                Signal::QuestionAsked { id: "q".into() },
            ),
        ];
        let secret = vec![
            input(T0),
            ev(
                T0 + 1_000,
                "secret_request",
                Signal::SecretAsked { id: "s".into() },
            ),
        ];
        let notice = vec![
            input(T0),
            ev(T0 + 1_000, "provider_notice", Signal::PersonMustAct),
        ];
        let planned = vec![
            input(T0),
            end(T0 + 1_000, TurnEnd::Completed, false),
            ev(T0 + 2_000, "auto_continue", Signal::Planned),
        ];
        let failed = vec![input(T0), end(T0 + 1_000, TurnEnd::Failed, false)];
        let interrupted = vec![input(T0), end(T0 + 1_000, TurnEnd::Interrupted, false)];
        let watching = Activity {
            monitors: 1,
            ..idle()
        };
        let bash = Activity {
            bash_tasks: 1,
            ..spawned()
        };
        let queued = Activity {
            queued_messages: 1,
            ..spawned()
        };
        let permission = Activity {
            pending_permissions: 1,
            ..streaming()
        };
        let late = T0 + 10 * MIN;
        let thinking = vec![input(T0), ev(T0 + 1, "thinking", Signal::Progress)];

        /// name, tail, activity, clock → state, rule.
        type Row<'a> = (&'a str, &'a [TailEvent], Activity, u64, SessionState, Rule);
        let rows: Vec<Row> = vec![
            (
                "tool in flight",
                &tool_running,
                streaming(),
                T0 + 5_000,
                SessionState::Working,
                Rule::ToolInFlight,
            ),
            (
                "thinking",
                &thinking,
                streaming(),
                T0 + 5_000,
                SessionState::Working,
                Rule::TurnRunning,
            ),
            (
                "text being written",
                &writing,
                streaming(),
                T0 + 5_000,
                SessionState::Writing,
                Rule::TextStreaming,
            ),
            (
                "silent 30 min while streaming",
                &tool_running,
                streaming(),
                T0 + 1_000 + STALLED_AFTER_MS,
                SessionState::Stalled,
                Rule::NoEventFor,
            ),
            (
                "a long Bash at its 10-min ceiling",
                &tool_running,
                streaming(),
                T0 + 1_000 + 10 * MIN,
                SessionState::Working,
                Rule::ToolInFlight,
            ),
            (
                "report to the human",
                &reported,
                idle(),
                late,
                SessionState::Reporting,
                Rule::AnsweredHuman,
            ),
            (
                "watch between turns",
                &reported,
                watching,
                late,
                SessionState::Watching,
                Rule::BackgroundRunning,
            ),
            (
                "background bash, spawned",
                &silent_end,
                bash,
                late,
                SessionState::Watching,
                Rule::BackgroundRunning,
            ),
            (
                "permission pending",
                &tool_running,
                permission,
                T0 + 2 * STALLED_AFTER_MS,
                SessionState::WaitingOnPerson,
                Rule::PermissionPending,
            ),
            (
                "question open",
                &question,
                streaming(),
                late,
                SessionState::WaitingOnPerson,
                Rule::QuestionOpen,
            ),
            (
                "secret requested",
                &secret,
                streaming(),
                late,
                SessionState::WaitingOnPerson,
                Rule::SecretRequested,
            ),
            (
                "vault locked",
                &notice,
                streaming(),
                late,
                SessionState::WaitingOnPerson,
                Rule::PersonMustAct,
            ),
            (
                "message queued",
                &silent_end,
                queued,
                late,
                SessionState::WaitingOnPerson,
                Rule::MessageQueued,
            ),
            (
                "auto-continue planned",
                &planned,
                spawned(),
                late,
                SessionState::Working,
                Rule::TurnPlanned,
            ),
            (
                "failed turn",
                &failed,
                spawned(),
                late,
                SessionState::Reporting,
                Rule::TurnFailed,
            ),
            (
                "interrupted turn",
                &interrupted,
                spawned(),
                late,
                SessionState::Reporting,
                Rule::TurnInterrupted,
            ),
            (
                "no evidence",
                &[],
                idle(),
                late,
                SessionState::Reporting,
                Rule::NoProvenEnd,
            ),
            (
                "end without its input",
                &silent_end[1..],
                spawned(),
                late,
                SessionState::Reporting,
                Rule::NoProvenEnd,
            ),
            (
                "just ended",
                &silent_end,
                spawned(),
                T0 + 3_000 + SETTLE_MS - 1,
                SessionState::Working,
                Rule::TurnSettling,
            ),
            (
                "silent end, root session",
                &silent_end,
                idle(),
                late,
                SessionState::Done,
                Rule::TurnCompleted,
            ),
            (
                "spawned child answered its spawner",
                &reported,
                spawned(),
                late,
                SessionState::Done,
                Rule::TurnCompleted,
            ),
        ];
        for (name, tail, activity, now, want, rule) in rows {
            let got = classify(tail, T0, &activity, now);
            assert_eq!(
                (got.state, got.reason.rule),
                (want, rule),
                "{name}: {got:?}"
            );
        }
    }

    /// The incident of session 2c7b913b (30/10/2026): four `Agent` calls, the last event a
    /// `tool_timing`, one call without its result, `streaming: true`, and 40 minutes of
    /// silence. Not `Working`: `Stalled`, decided by the `tool_timing` that was last seen.
    #[test]
    fn the_2c7b913b_replay_is_stalled_not_working() {
        let t = |s: u64| T0 + s * 1_000;
        let tail = vec![
            input(t(0)),
            tool(t(1), "call_01"),
            tool(t(1), "call_02"),
            tool(t(1), "call_03"),
            tool(t(1), "call_04"),
            result(t(60), "call_01"),
            timing(t(60)),
            result(t(90), "call_02"),
            timing(t(90)),
            result(t(120), "call_03"),
            timing(t(120)),
        ];
        let now = t(120) + 40 * MIN;
        let got = classify(&tail, T0, &streaming(), now);
        assert_eq!(got.state, SessionState::Stalled, "{got:?}");
        assert_eq!(got.state.as_str(), "stalled");
        assert_eq!(got.reason.rule, Rule::NoEventFor);
        assert_eq!(got.reason.event.as_deref(), Some("tool_timing"));
        assert_eq!(got.reason.age_ms, 40 * MIN);
        // Ten minutes in, the same tail is a call in flight, named.
        let early = classify(&tail, T0, &streaming(), t(120) + 10 * MIN);
        assert_eq!(early.state, SessionState::Working);
        assert_eq!(early.reason.rule, Rule::ToolInFlight);
    }

    /// Reporting, Watching and WaitingOnPerson never give Done, whatever the clock says.
    #[test]
    fn report_watch_and_wait_never_give_done() {
        let reported = vec![
            input(T0),
            text(T0 + 1),
            end(T0 + 2, TurnEnd::Completed, true),
        ];
        let silent = vec![input(T0), end(T0 + 2, TurnEnd::Completed, false)];
        for later in [
            SETTLE_MS,
            10 * MIN,
            STALLED_AFTER_MS * 10,
            365 * 24 * 60 * MIN,
        ] {
            let now = T0 + later;
            assert_eq!(state(&reported, idle(), now), SessionState::Reporting);
            for activity in [
                Activity {
                    monitors: 1,
                    ..spawned()
                },
                Activity {
                    bash_tasks: 2,
                    ..spawned()
                },
                Activity {
                    queued_messages: 1,
                    ..spawned()
                },
                Activity {
                    pending_permissions: 1,
                    ..spawned()
                },
            ] {
                assert_ne!(
                    state(&silent, activity, now),
                    SessionState::Done,
                    "{activity:?}"
                );
            }
        }
    }

    /// A new message after Done flips the state back: the segment starts again.
    #[test]
    fn a_new_input_leaves_done() {
        let mut tail = vec![input(T0), end(T0 + 1, TurnEnd::Completed, false)];
        assert_eq!(state(&tail, spawned(), T0 + MIN), SessionState::Done);
        tail.push(input(T0 + 2 * MIN));
        assert_eq!(
            state(&tail, streaming(), T0 + 2 * MIN),
            SessionState::Working
        );
        assert_eq!(
            state(&tail, spawned(), T0 + 3 * MIN),
            SessionState::Reporting
        );
    }

    /// Same tail, same activity, same clock: the same verdict, byte for byte.
    #[test]
    fn replayable() {
        let tail = vec![
            input(T0),
            tool(T0 + 5, "a"),
            result(T0 + 9, "a"),
            text(T0 + 12),
        ];
        let a = classify(&tail, T0, &streaming(), T0 + 99);
        let b = classify(&tail.clone(), T0, &streaming(), T0 + 99);
        assert_eq!(a, b);
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::json!({"state": "writing", "reason": {"rule": "text_streaming", "event": "assistant_text", "age_ms": 87}})
        );
    }

    /// The tail folds the token deltas, keeps its cap, and skips what says nothing.
    #[test]
    fn the_tail_folds_deltas_and_stays_bounded() {
        let mut tail = Tail::starting_at(T0);
        let d = |s: &str| ChatEvent::StreamDelta {
            text: s.into(),
            parent_tool_use_id: None,
        };
        tail.record(
            &ChatEvent::UserMessage {
                content: "go".into(),
            },
            T0 + 1,
        );
        for i in 0..500 {
            tail.record(&d("x"), T0 + 10 + i);
        }
        tail.record(&ChatEvent::StreamingStatus { is_streaming: true }, T0 + 999);
        let events = tail.events();
        assert_eq!(events.len(), 2, "{events:?}");
        assert_eq!(events[1].at_ms, T0 + 509);
        for i in 0..(TAIL_CAP as u64 * 2) {
            tail.record(
                &ChatEvent::Thinking {
                    content: "t".into(),
                    parent_tool_use_id: None,
                },
                T0 + 1_000 + i,
            );
        }
        assert_eq!(tail.events().len(), TAIL_CAP);
    }

    /// The cap is tested above (it bounds the queue); this tests its CONSEQUENCE.
    /// A turn longer than the cap evicts the `Input` that opened it — the proof a
    /// turn began — and `classify` then answers `Reporting/NoProvenEnd` for want of
    /// it, so a long delegated child is never released as `Done` (only the 30-min
    /// expiration frees it). The opening `Input` must survive the cap.
    #[test]
    fn a_turn_longer_than_the_cap_still_releases_as_done() {
        let mut tail = Tail::starting_at(T0);
        tail.record(&ChatEvent::UserMessage { content: "go".into() }, T0);
        // 66 tool calls: each is a ToolStart + a ToolEnd, 132 entries — past TAIL_CAP.
        for i in 0..66 {
            let id = format!("call_{i}");
            tail.record(
                &ChatEvent::ToolUse {
                    id: id.clone(),
                    tool: "Bash".into(),
                    input: serde_json::Value::Null,
                    parent_tool_use_id: None,
                    category: None,
                    canonical: None,
                },
                T0 + 1 + i,
            );
            tail.record(
                &ChatEvent::ToolResult {
                    id,
                    result: serde_json::Value::Null,
                    is_error: false,
                    parent_tool_use_id: None,
                },
                T0 + 2 + i,
            );
        }
        tail.record(
            &ChatEvent::Result {
                session_id: "s".into(),
                duration_ms: 0,
                cost_usd: None,
                subtype: "success".into(),
                is_error: false,
                num_turns: None,
                result_text: None,
                cost: None,
                usage: None,
                model: None,
                stop_reason: None,
                error_code: None,
                error_message: None,
            },
            T0 + 200,
        );

        let got = tail.classify(&spawned(), T0 + 200 + SETTLE_MS);
        assert_eq!(got.state, SessionState::Done, "{got:?}");
    }

    #[test]
    fn every_state_has_its_wire_name() {
        for (state, name) in [
            (SessionState::Working, "working"),
            (SessionState::Writing, "writing"),
            (SessionState::Reporting, "reporting"),
            (SessionState::Watching, "watching"),
            (SessionState::WaitingOnPerson, "waiting_on_person"),
            (SessionState::Done, "done"),
            (SessionState::Stalled, "stalled"),
        ] {
            assert_eq!(state.as_str(), name);
            assert_eq!(serde_json::to_value(state).unwrap(), name);
        }
    }

    /// The other events of the engines, each on the signal the rules read.
    #[test]
    fn the_remaining_events_map_to_their_signals() {
        use crate::chat::types::ChatEvent as E;
        let when = chrono::Utc::now();
        let s = |e: &E| signal_of(e);
        assert_eq!(
            s(&E::ToolResult {
                id: "a".into(),
                result: serde_json::Value::Null,
                is_error: false,
                parent_tool_use_id: None,
            }),
            Some(Signal::ToolEnd { id: "a".into() })
        );
        assert_eq!(
            s(&E::ToolCancelled {
                id: "a".into(),
                parent_tool_use_id: None,
            }),
            Some(Signal::ToolEnd { id: "a".into() })
        );
        assert_eq!(
            s(&E::ToolUse {
                id: "a".into(),
                tool: "Bash".into(),
                input: serde_json::Value::Null,
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            }),
            Some(Signal::ToolStart { id: "a".into() })
        );
        assert_eq!(
            s(&E::AskUserQuestion {
                id: "r".into(),
                tool_call_id: "q".into(),
                questions: serde_json::Value::Null,
                input: serde_json::Value::Null,
                parent_tool_use_id: None,
                synthetic: None,
            }),
            Some(Signal::QuestionAsked { id: "q".into() })
        );
        assert_eq!(
            s(&E::SecretRequest {
                id: "k".into(),
                name: "token".into(),
                reason: "deploy".into(),
                exists: false,
            }),
            Some(Signal::SecretAsked { id: "k".into() })
        );
        assert_eq!(
            s(&E::SecretRequestResolved {
                id: "k".into(),
                outcome: "provided".into(),
            }),
            Some(Signal::SecretResolved { id: "k".into() })
        );
        let notice = |severity: &str| E::ProviderNotice {
            kind: "credentials_locked".into(),
            code: None,
            message: "unlock the vault".into(),
            severity: severity.into(),
            data: None,
        };
        assert_eq!(s(&notice("action_required")), Some(Signal::PersonMustAct));
        assert_eq!(s(&notice("latency")), Some(Signal::Progress));
        let error = |parent: Option<&str>| E::Error {
            message: "boom".into(),
            parent_tool_use_id: parent.map(str::to_string),
            code: None,
            reason: None,
            index: None,
            request_id: None,
        };
        assert_eq!(s(&error(None)), Some(Signal::Failure));
        assert_eq!(s(&error(Some("sub"))), Some(Signal::Progress));
        assert_eq!(
            s(&E::SessionError {
                reason: "subprocess_exited".into(),
                message: "gone".into(),
                received_at: when,
            }),
            Some(Signal::Failure)
        );
        assert_eq!(
            s(&E::AutoContinue {
                session_id: "s".into(),
                delay_ms: 5,
            }),
            Some(Signal::Planned)
        );
        assert_eq!(
            s(&E::Retrying {
                attempt: 1,
                max_attempts: 3,
                delay_ms: 5,
                error_message: "503".into(),
            }),
            Some(Signal::Planned)
        );
        assert_eq!(
            s(&E::CompactionStarted {
                trigger: "auto".into(),
            }),
            Some(Signal::Progress)
        );
    }

    /// Seeding puts the stored thread in front of the live recording, keeps the cap, and
    /// leaves out what the recording already has.
    #[test]
    fn a_seeded_tail_keeps_the_older_events_before_the_recording() {
        let mut tail = Tail::starting_at(T0 + 100);
        tail.record(
            &ChatEvent::UserMessage {
                content: "now".into(),
            },
            T0 + 100,
        );
        let older: Vec<(ChatEvent, u64)> = (0..(TAIL_CAP as u64 + 10))
            .map(|i| {
                (
                    ChatEvent::Thinking {
                        content: "t".into(),
                        parent_tool_use_id: None,
                    },
                    T0 - 1_000 + i,
                )
            })
            .chain(std::iter::once((
                ChatEvent::UserMessage {
                    content: "already recorded".into(),
                },
                T0 + 200,
            )))
            .collect();
        tail.seed(older.iter().map(|(e, at)| (e, *at)));
        assert_eq!(tail.observed_since_ms(), T0 + 100);
        let events = tail.events();
        assert_eq!(events.len(), TAIL_CAP, "the cap keeps the newest");
        assert_eq!(events.last().unwrap().signal, Signal::Input);
        assert_eq!(events.last().unwrap().at_ms, T0 + 100);
    }

    /// A question the interrupted turn left open is no longer waited on. Measured 10/11/2026
    /// (session cadfa92b): an `AskUserQuestion` interrupted arrives as `result{interrupted}`
    /// plus a `background_output` — NEVER as a `tool_result`, and that day no `tool_cancelled`
    /// at all (0 lines, 1 line naming the tool_call_id). Closed only on a `ToolEnd` of its id,
    /// the question stayed open for ever and the session read as `waiting_on_person`.
    #[test]
    fn an_interrupted_turn_closes_the_question_it_left_open() {
        let interrupted = vec![
            input(T0),
            ev(
                T0 + 1,
                "ask_user_question",
                Signal::QuestionAsked { id: "q".into() },
            ),
            end(T0 + 2, TurnEnd::Interrupted, false),
        ];
        let got = classify(&interrupted, T0, &idle(), T0 + 3);
        assert_ne!(
            got.reason.rule,
            Rule::QuestionOpen,
            "the turn that asked it was interrupted: nobody is waited on any more ({:?})",
            got
        );
    }

    /// A question or a secret closed by its answer is no longer waited on.
    #[test]
    fn an_answered_question_or_secret_is_not_waited_on() {
        let question_answered = vec![
            input(T0),
            ev(
                T0 + 1,
                "ask_user_question",
                Signal::QuestionAsked { id: "q".into() },
            ),
            result(T0 + 2, "q"),
        ];
        let secret_resolved = vec![
            input(T0),
            ev(
                T0 + 1,
                "secret_request",
                Signal::SecretAsked { id: "k".into() },
            ),
            ev(
                T0 + 2,
                "secret_request_resolved",
                Signal::SecretResolved { id: "k".into() },
            ),
        ];
        assert_eq!(
            classify(&question_answered, T0, &streaming(), T0 + 3)
                .reason
                .rule,
            Rule::TurnRunning
        );
        assert_eq!(
            classify(&secret_resolved, T0, &streaming(), T0 + 3)
                .reason
                .rule,
            Rule::TurnRunning
        );
    }

    /// A failure inside the turn is reported even when the turn's end reads as completed,
    /// and a top-level delta is text for the person when the end carries none.
    #[test]
    fn a_failure_in_the_turn_is_a_report_and_a_delta_is_text_for_a_person() {
        let failed_then_completed = vec![
            input(T0),
            ev(T0 + 1, "error", Signal::Failure),
            end(T0 + 2, TurnEnd::Completed, false),
        ];
        let got = classify(&failed_then_completed, T0, &idle(), T0 + 10 * MIN);
        assert_eq!(
            (got.state, got.reason.rule),
            (SessionState::Reporting, Rule::TurnFailed)
        );
        assert_eq!(got.reason.event.as_deref(), Some("error"));

        let streamed_answer = vec![
            input(T0),
            delta(T0 + 1),
            end(T0 + 2, TurnEnd::Completed, false),
        ];
        let got = classify(&streamed_answer, T0, &idle(), T0 + 10 * MIN);
        assert_eq!(
            (got.state, got.reason.rule),
            (SessionState::Reporting, Rule::AnsweredHuman)
        );
    }

    /// A tail whose only end is an earlier turn's: nothing of this turn is proven.
    #[test]
    fn an_end_before_the_last_input_proves_nothing() {
        let old_turn = vec![
            input(T0),
            end(T0 + 1, TurnEnd::Completed, false),
            input(T0 + 2),
        ];
        let got = classify(&old_turn, T0, &idle(), T0 + 10 * MIN);
        assert_eq!(got.reason.rule, Rule::NoProvenEnd);
        assert_eq!(got.state, SessionState::Reporting);
    }

    /// The events as the engines emit them map to the signals the rules read.
    #[test]
    fn events_map_to_signals() {
        let result = |stop: Option<&str>, is_error: bool, text: Option<&str>| ChatEvent::Result {
            session_id: "s".into(),
            duration_ms: 1,
            cost_usd: None,
            subtype: if is_error {
                "error_during_execution".into()
            } else {
                "success".into()
            },
            is_error,
            num_turns: None,
            result_text: text.map(str::to_string),
            cost: None,
            usage: None,
            model: None,
            stop_reason: stop.map(str::to_string),
            error_code: None,
            error_message: None,
        };
        assert_eq!(
            signal_of(&result(None, false, Some("done"))),
            Some(Signal::Ended {
                how: TurnEnd::Completed,
                text: true
            })
        );
        assert_eq!(
            signal_of(&result(Some("interrupted"), false, None)),
            Some(Signal::Ended {
                how: TurnEnd::Interrupted,
                text: false
            })
        );
        assert_eq!(
            signal_of(&result(None, true, None)),
            Some(Signal::Ended {
                how: TurnEnd::Failed,
                text: false
            })
        );
        assert_eq!(
            signal_of(&ChatEvent::AssistantText {
                content: "x".into(),
                parent_tool_use_id: Some("p".into())
            }),
            Some(Signal::Text { top_level: false })
        );
        assert_eq!(
            signal_of(&ChatEvent::StreamingStatus {
                is_streaming: false
            }),
            None
        );
        assert_eq!(
            signal_of(&ChatEvent::PendingQueue { messages: vec![] }),
            None
        );
    }
}
