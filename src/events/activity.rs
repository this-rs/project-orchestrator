//! Unified `ActivityEvent` envelope for the Live Activity Hub (`/ws/activity`).
//!
//! This module defines the **single contract** shared between the backend event
//! pipeline and the frontend Activity Hub UI. It wraps every event type relevant
//! to the live visualization of in-flight work:
//!
//! - [`RunnerEvent`] — plan-runner lifecycle (PlanStarted, TaskCompleted, etc.)
//! - [`ChatEvent`] — filtered subset of LLM session events (ToolUse, Result, …)
//! - [`CrudEvent`] — filtered to {Plan, Task, ProtocolRun, ChatSession}
//! - Protocol FSM progress — current state + optional [`ProgressSnapshot`]
//!
//! ## Discriminated union
//!
//! Serialized with `#[serde(tag = "kind", rename_all = "snake_case")]` so the
//! TypeScript side gets a clean discriminated union:
//!
//! ```ts
//! type ActivityEvent =
//!   | { kind: "runner";            seq: number; ... }
//!   | { kind: "chat";              seq: number; ... }
//!   | { kind: "crud";              seq: number; ... }
//!   | { kind: "protocol_progress"; seq: number; ... };
//! ```
//!
//! ## Sequence numbers
//!
//! Every event carries a monotonic `seq` produced by the [`EventBus`] AtomicU64
//! counter. Clients use `seq` to detect gaps after a WebSocket reconnect and
//! request replay via `GET /api/activity/snapshot`.
//!
//! ## Filtering
//!
//! [`ActivityEvent::try_from_chat`] and [`ActivityEvent::try_from_crud`] return
//! `Option<Self>` — they return `None` for variants that are noise in the
//! Activity Hub (e.g. `ChatEvent::StreamDelta`, CRUD on `Note`).
//!
//! [`EventBus`]: super::bus::EventBus

use crate::chat::types::ChatEvent;
use crate::events::types::{CrudAction, CrudEvent, EntityType};
use crate::protocol::models::RunStatus as ProtocolRunStatus;
use crate::runner::models::RunnerEvent;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// ActivityEvent — unified envelope
// ============================================================================

/// Unified event envelope for the Live Activity Hub WebSocket.
///
/// Every variant carries:
/// - `seq`       — monotonic sequence number (per [`EventBus`])
/// - `timestamp` — ISO-8601 emission time
/// - a payload   — the original typed event
///
/// Variants are tagged with `kind` (snake_case) for clean TS discriminated unions.
///
/// [`EventBus`]: super::bus::EventBus
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActivityEvent {
    /// Plan-runner lifecycle event (always relevant — no filtering).
    Runner {
        seq: u64,
        timestamp: String,
        /// Run identifier (mirrored from the inner event for client-side indexing).
        run_id: Uuid,
        /// The original [`RunnerEvent`].
        event: RunnerEvent,
    },

    /// Filtered subset of [`ChatEvent`] relevant to the Activity Hub.
    ///
    /// Construct via [`ActivityEvent::try_from_chat`] — variants that are not
    /// relevant (e.g. StreamDelta, PermissionRequest) yield `None` there.
    Chat {
        seq: u64,
        timestamp: String,
        /// Chat session identifier (string — sessions use string ids).
        session_id: String,
        /// Optional task identifier when this session was spawned by the runner.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_id: Option<Uuid>,
        /// Optional run identifier when this session belongs to a plan run.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        run_id: Option<Uuid>,
        /// The original [`ChatEvent`] (filtering happens at construction time).
        event: ChatEvent,
    },

    /// Filtered CRUD event — only `{Plan, Task, ProtocolRun, ChatSession}`.
    ///
    /// Construct via [`ActivityEvent::try_from_crud`] — other entity types
    /// yield `None`.
    Crud {
        seq: u64,
        timestamp: String,
        entity_type: EntityType,
        action: CrudAction,
        entity_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        project_id: Option<String>,
        #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
        payload: serde_json::Value,
    },

    /// Protocol FSM progress (current state + optional sub-action progress).
    ///
    /// Emitted whenever a protocol run transitions to a new state, reports
    /// progress mid-state via `report_progress()`, or reaches a terminal state.
    ProtocolProgress {
        seq: u64,
        timestamp: String,
        run_id: Uuid,
        protocol_id: Uuid,
        /// Current FSM state identifier.
        current_state: Uuid,
        /// Human-readable name of the current state (denormalized for UI).
        state_name: String,
        /// Run status: running / completed / failed / cancelled.
        status: ProtocolRunStatus,
        /// Optional in-state progress snapshot.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        progress: Option<ProtocolProgressSnapshot>,
    },
}

// ============================================================================
// ProtocolProgressSnapshot — flattened progress payload
// ============================================================================

/// Lightweight progress snapshot embedded in [`ActivityEvent::ProtocolProgress`].
///
/// Mirrors `protocol::models::ProgressSnapshot` but is defined here so that the
/// Activity Hub contract remains self-contained (the frontend imports a single
/// schema).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProtocolProgressSnapshot {
    /// Current sub-action being executed (e.g., "backfill_synapses").
    pub sub_action: String,
    /// Number of sub-actions processed so far.
    pub processed: usize,
    /// Total number of sub-actions.
    pub total: usize,
    /// Elapsed time in milliseconds.
    pub elapsed_ms: u64,
}

impl From<&crate::protocol::models::ProgressSnapshot> for ProtocolProgressSnapshot {
    fn from(p: &crate::protocol::models::ProgressSnapshot) -> Self {
        Self {
            sub_action: p.sub_action.clone(),
            processed: p.processed,
            total: p.total,
            elapsed_ms: p.elapsed_ms,
        }
    }
}

// ============================================================================
// ProtocolProgress — input struct for the From impl
// ============================================================================

/// Input struct used by [`ActivityEvent::from_protocol_progress`] (and the
/// `From<ProtocolProgress>` impl) when emitting a protocol progress event.
///
/// This is a constructor-only helper — callers fill it from the live
/// `ProtocolRun` / `ProgressSnapshot` they hold, and the `From` impl wraps it
/// in [`ActivityEvent::ProtocolProgress`].
#[derive(Debug, Clone)]
pub struct ProtocolProgress {
    pub seq: u64,
    pub timestamp: String,
    pub run_id: Uuid,
    pub protocol_id: Uuid,
    pub current_state: Uuid,
    pub state_name: String,
    pub status: ProtocolRunStatus,
    pub progress: Option<ProtocolProgressSnapshot>,
}

// ============================================================================
// Conversions
// ============================================================================

impl ActivityEvent {
    /// Returns the monotonic sequence number of this event.
    pub fn seq(&self) -> u64 {
        match self {
            Self::Runner { seq, .. }
            | Self::Chat { seq, .. }
            | Self::Crud { seq, .. }
            | Self::ProtocolProgress { seq, .. } => *seq,
        }
    }

    /// Returns the `kind` discriminator string (for logging and metrics).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Runner { .. } => "runner",
            Self::Chat { .. } => "chat",
            Self::Crud { .. } => "crud",
            Self::ProtocolProgress { .. } => "protocol_progress",
        }
    }

    /// Wrap a [`RunnerEvent`] in an [`ActivityEvent::Runner`].
    ///
    /// RunnerEvents are always relevant — no filtering. Extracts `run_id` from
    /// the inner event so the client can index by run without re-parsing.
    pub fn from_runner(seq: u64, event: RunnerEvent) -> Self {
        let run_id = runner_event_run_id(&event);
        Self::Runner {
            seq,
            timestamp: now_iso(),
            run_id,
            event,
        }
    }

    /// Try to wrap a [`ChatEvent`] in an [`ActivityEvent::Chat`].
    ///
    /// Returns `None` for variants that are noise in the Activity Hub (e.g.
    /// `StreamDelta`, `PermissionRequest`, `UserMessage`). Variants kept are
    /// the ones useful for live visualization of agent work — see
    /// [`is_chat_event_relevant`].
    pub fn try_from_chat(
        seq: u64,
        session_id: impl Into<String>,
        task_id: Option<Uuid>,
        run_id: Option<Uuid>,
        event: ChatEvent,
    ) -> Option<Self> {
        if !is_chat_event_relevant(&event) {
            return None;
        }
        Some(Self::Chat {
            seq,
            timestamp: now_iso(),
            session_id: session_id.into(),
            task_id,
            run_id,
            event,
        })
    }

    /// Try to wrap a [`CrudEvent`] in an [`ActivityEvent::Crud`].
    ///
    /// Returns `None` unless the event's `entity_type` is one of:
    /// `Plan`, `Task`, `ProtocolRun`, `ChatSession`.
    pub fn try_from_crud(seq: u64, event: CrudEvent) -> Option<Self> {
        if !is_crud_entity_relevant(&event.entity_type) {
            return None;
        }
        Some(Self::Crud {
            seq,
            timestamp: event.timestamp,
            entity_type: event.entity_type,
            action: event.action,
            entity_id: event.entity_id,
            project_id: event.project_id,
            payload: event.payload,
        })
    }

    /// Wrap a [`ProtocolProgress`] in an [`ActivityEvent::ProtocolProgress`].
    pub fn from_protocol_progress(p: ProtocolProgress) -> Self {
        Self::ProtocolProgress {
            seq: p.seq,
            timestamp: p.timestamp,
            run_id: p.run_id,
            protocol_id: p.protocol_id,
            current_state: p.current_state,
            state_name: p.state_name,
            status: p.status,
            progress: p.progress,
        }
    }
}

// ----------------------------------------------------------------------------
// From impls
// ----------------------------------------------------------------------------

/// Wraps a `RunnerEvent` with `seq = 0` and a freshly-generated timestamp.
///
/// Prefer [`ActivityEvent::from_runner`] when you have a sequence number from
/// the [`EventBus`](super::bus::EventBus); this `From` impl exists for ergonomics
/// in tests and ad-hoc emitters.
impl From<RunnerEvent> for ActivityEvent {
    fn from(event: RunnerEvent) -> Self {
        Self::from_runner(0, event)
    }
}

/// Wraps a `ProtocolProgress` (already carries seq + timestamp).
impl From<ProtocolProgress> for ActivityEvent {
    fn from(p: ProtocolProgress) -> Self {
        Self::from_protocol_progress(p)
    }
}

// ----------------------------------------------------------------------------
// Filtering predicates (pub(crate) for unit-testability)
// ----------------------------------------------------------------------------

/// Returns true when a `ChatEvent` is relevant to the Activity Hub.
///
/// Activity Hub focuses on **agent work signals**, not interactive UX events:
/// - **Kept**: AssistantText, Thinking, ToolUse, ToolResult, ToolCancelled,
///   Result, Error, BackgroundOutput, ActiveTasksUpdate, ToolsCancelled,
///   SessionError, AutoContinue, Retrying, SystemInit, CompactBoundary,
///   CompactionStarted.
/// - **Dropped**: UserMessage, SystemHint, ToolUseInputResolved,
///   PermissionRequest, AskUserQuestion, InputRequest, StreamDelta,
///   StreamingStatus, PermissionDecision, PermissionModeChanged,
///   ModelChanged, CompactionRecovery, AutoContinueStateChanged.
pub fn is_chat_event_relevant(event: &ChatEvent) -> bool {
    matches!(
        event,
        ChatEvent::AssistantText { .. }
            | ChatEvent::Thinking { .. }
            | ChatEvent::ToolUse { .. }
            | ChatEvent::ToolResult { .. }
            | ChatEvent::ToolCancelled { .. }
            | ChatEvent::Result { .. }
            | ChatEvent::Error { .. }
            | ChatEvent::BackgroundOutput { .. }
            | ChatEvent::ActiveTasksUpdate { .. }
            | ChatEvent::ToolsCancelled { .. }
            | ChatEvent::SessionError { .. }
            | ChatEvent::AutoContinue { .. }
            | ChatEvent::Retrying { .. }
            | ChatEvent::SystemInit { .. }
            | ChatEvent::CompactBoundary { .. }
            | ChatEvent::CompactionStarted { .. }
    )
}

/// Returns true when an `EntityType` is relevant to the Activity Hub CRUD pane.
pub fn is_crud_entity_relevant(entity_type: &EntityType) -> bool {
    matches!(
        entity_type,
        EntityType::Plan | EntityType::Task | EntityType::ProtocolRun | EntityType::ChatSession
    )
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Extract `run_id` from any [`RunnerEvent`] variant.
fn runner_event_run_id(event: &RunnerEvent) -> Uuid {
    match event {
        RunnerEvent::PlanStarted { run_id, .. }
        | RunnerEvent::WaveStarted { run_id, .. }
        | RunnerEvent::TaskStarted { run_id, .. }
        | RunnerEvent::TaskCompleted { run_id, .. }
        | RunnerEvent::TaskFailed { run_id, .. }
        | RunnerEvent::TaskTimeout { run_id, .. }
        | RunnerEvent::WaveCompleted { run_id, .. }
        | RunnerEvent::PlanCompleted { run_id, .. }
        | RunnerEvent::TaskCompletedWithoutSteps { run_id, .. }
        | RunnerEvent::CwdMismatch { run_id, .. }
        | RunnerEvent::TaskSpawningTimeout { run_id, .. }
        | RunnerEvent::RunnerError { run_id, .. }
        | RunnerEvent::BudgetExceeded { run_id, .. }
        | RunnerEvent::WorktreeRecovery { run_id, .. }
        | RunnerEvent::LifecycleTransition { run_id, .. } => *run_id,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::types::{CrudAction, CrudEvent, EntityType};

    fn sample_runner_event() -> RunnerEvent {
        RunnerEvent::PlanStarted {
            run_id: Uuid::nil(),
            plan_id: Uuid::nil(),
            plan_title: "Demo".into(),
            total_tasks: 1,
            total_waves: 1,
            prediction: None,
        }
    }

    #[test]
    fn test_kind_strings_match_discriminator() {
        let r = ActivityEvent::from_runner(0, sample_runner_event());
        assert_eq!(r.kind(), "runner");

        let c = ActivityEvent::try_from_chat(
            0,
            "sess-1",
            None,
            None,
            ChatEvent::Thinking {
                content: "thinking".into(),
                parent_tool_use_id: None,
            },
        )
        .unwrap();
        assert_eq!(c.kind(), "chat");

        let crud_evt = CrudEvent::new(EntityType::Plan, CrudAction::Updated, "plan-1");
        let cr = ActivityEvent::try_from_crud(0, crud_evt).unwrap();
        assert_eq!(cr.kind(), "crud");

        let pp = ActivityEvent::from_protocol_progress(ProtocolProgress {
            seq: 0,
            timestamp: now_iso(),
            run_id: Uuid::nil(),
            protocol_id: Uuid::nil(),
            current_state: Uuid::nil(),
            state_name: "S".into(),
            status: ProtocolRunStatus::Running,
            progress: None,
        });
        assert_eq!(pp.kind(), "protocol_progress");
    }

    #[test]
    fn test_runner_event_serializes_with_kind_tag() {
        let evt = ActivityEvent::from_runner(42, sample_runner_event());
        let json = serde_json::to_string(&evt).unwrap();
        assert!(json.contains("\"kind\":\"runner\""));
        assert!(json.contains("\"seq\":42"));
        assert!(json.contains("\"event\""));
    }

    #[test]
    fn test_round_trip_runner_event() {
        let evt = ActivityEvent::from_runner(7, sample_runner_event());
        let json = serde_json::to_string(&evt).unwrap();
        let back: ActivityEvent = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2, "round-trip JSON must be identical");
        assert_eq!(back.seq(), 7);
        assert_eq!(back.kind(), "runner");
    }

    #[test]
    fn test_round_trip_chat_event() {
        let evt = ActivityEvent::try_from_chat(
            12,
            "sess-xyz",
            Some(Uuid::nil()),
            Some(Uuid::nil()),
            ChatEvent::ToolUse {
                id: "tu-1".into(),
                tool: "Edit".into(),
                input: serde_json::json!({"path": "/tmp/a"}),
                parent_tool_use_id: None,
            },
        )
        .unwrap();
        let json = serde_json::to_string(&evt).unwrap();
        let back: ActivityEvent = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn test_round_trip_crud_event() {
        let crud = CrudEvent::new(EntityType::Task, CrudAction::StatusChanged, "task-1")
            .with_payload(serde_json::json!({"old": "pending", "new": "running"}))
            .with_project_id("proj-1");
        let evt = ActivityEvent::try_from_crud(99, crud).unwrap();
        let json = serde_json::to_string(&evt).unwrap();
        let back: ActivityEvent = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn test_round_trip_protocol_progress() {
        let evt = ActivityEvent::from_protocol_progress(ProtocolProgress {
            seq: 3,
            timestamp: now_iso(),
            run_id: Uuid::nil(),
            protocol_id: Uuid::nil(),
            current_state: Uuid::nil(),
            state_name: "Backfilling".into(),
            status: ProtocolRunStatus::Running,
            progress: Some(ProtocolProgressSnapshot {
                sub_action: "backfill_synapses".into(),
                processed: 12,
                total: 100,
                elapsed_ms: 4242,
            }),
        });
        let json = serde_json::to_string(&evt).unwrap();
        let back: ActivityEvent = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
        assert!(json.contains("\"kind\":\"protocol_progress\""));
        assert!(json.contains("\"sub_action\":\"backfill_synapses\""));
    }

    #[test]
    fn test_filter_drops_stream_delta() {
        let ev = ChatEvent::StreamDelta {
            text: "hello".into(),
            parent_tool_use_id: None,
        };
        assert!(!is_chat_event_relevant(&ev));
        assert!(
            ActivityEvent::try_from_chat(0, "sess-1", None, None, ev).is_none(),
            "StreamDelta must be filtered out — it's a high-frequency noise event"
        );
    }

    #[test]
    fn test_filter_drops_user_message() {
        let ev = ChatEvent::UserMessage {
            content: "hi".into(),
        };
        assert!(!is_chat_event_relevant(&ev));
        assert!(ActivityEvent::try_from_chat(0, "s", None, None, ev).is_none());
    }

    #[test]
    fn test_filter_drops_permission_request() {
        let ev = ChatEvent::PermissionRequest {
            id: "p1".into(),
            tool: "Bash".into(),
            input: serde_json::Value::Null,
            parent_tool_use_id: None,
        };
        assert!(!is_chat_event_relevant(&ev));
        assert!(ActivityEvent::try_from_chat(0, "s", None, None, ev).is_none());
    }

    #[test]
    fn test_filter_keeps_result() {
        let ev = ChatEvent::Result {
            session_id: "s".into(),
            duration_ms: 1000,
            cost_usd: Some(0.01),
            subtype: "success".into(),
            is_error: false,
            num_turns: Some(2),
            result_text: None,
        };
        assert!(is_chat_event_relevant(&ev));
        assert!(ActivityEvent::try_from_chat(0, "s", None, None, ev).is_some());
    }

    #[test]
    fn test_crud_filter_keeps_plan_task_protocolrun_chatsession() {
        for et in [
            EntityType::Plan,
            EntityType::Task,
            EntityType::ProtocolRun,
            EntityType::ChatSession,
        ] {
            assert!(is_crud_entity_relevant(&et), "{:?} must be kept", et);
            let evt = CrudEvent::new(et.clone(), CrudAction::Updated, "e");
            assert!(ActivityEvent::try_from_crud(0, evt).is_some());
        }
    }

    #[test]
    fn test_crud_filter_drops_unrelated_entities() {
        for et in [
            EntityType::Note,
            EntityType::Decision,
            EntityType::Skill,
            EntityType::Persona,
            EntityType::Project,
        ] {
            assert!(!is_crud_entity_relevant(&et), "{:?} must be dropped", et);
            let evt = CrudEvent::new(et, CrudAction::Updated, "e");
            assert!(ActivityEvent::try_from_crud(0, evt).is_none());
        }
    }

    #[test]
    fn test_runner_event_extracts_run_id() {
        let run_id = Uuid::new_v4();
        let evt = ActivityEvent::from_runner(
            0,
            RunnerEvent::TaskStarted {
                run_id,
                task_id: Uuid::nil(),
                task_title: "T1".into(),
                wave_number: 1,
            },
        );
        if let ActivityEvent::Runner {
            run_id: extracted, ..
        } = evt
        {
            assert_eq!(extracted, run_id);
        } else {
            panic!("expected Runner variant");
        }
    }

    #[test]
    fn test_from_runner_event_trait() {
        let runner_ev = sample_runner_event();
        let activity: ActivityEvent = runner_ev.into();
        assert_eq!(activity.kind(), "runner");
        assert_eq!(activity.seq(), 0);
    }

    #[test]
    fn test_from_protocol_progress_trait() {
        let pp = ProtocolProgress {
            seq: 5,
            timestamp: now_iso(),
            run_id: Uuid::nil(),
            protocol_id: Uuid::nil(),
            current_state: Uuid::nil(),
            state_name: "Init".into(),
            status: ProtocolRunStatus::Running,
            progress: None,
        };
        let activity: ActivityEvent = pp.into();
        assert_eq!(activity.kind(), "protocol_progress");
        assert_eq!(activity.seq(), 5);
    }
}
