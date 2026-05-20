//! DTOs for the **Live Activity Hub** REST snapshot endpoint.
//!
//! `GET /api/activity/snapshot?project_id=<uuid>` returns the initial state
//! the frontend needs to render the Activity Hub _before_ subscribing to the
//! `/ws/activity` delta stream. The contract is intentionally minimal:
//!
//! - **`plan_runs`** — every `PlanRun` currently `running` for the project,
//!   with their current wave + counters
//! - **`protocol_runs`** — every `ProtocolRun` currently `running` for the
//!   project (via `INSTANCE_OF` to a `Protocol` belonging to the project)
//! - **`chat_sessions`** — recently active `ChatSession` nodes for the
//!   project (ordered by `updated_at` desc, capped)
//! - **`last_event_seq`** — current head of the global event sequence so the
//!   client can pass `lastEventSeq` to `/ws/activity` and replay any deltas
//!   that fired _between_ the REST read and the WS subscribe
//!
//! Snapshots are **eventually consistent** with the WS stream: a delta may
//! arrive on `/ws/activity` for a state already reflected in the snapshot —
//! the client should treat snapshot rows as authoritative for `seq <=
//! last_event_seq` and switch to deltas for everything strictly newer.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// PlanRun summary (in-progress only)
// ============================================================================

/// Status of a [`PlanRunSummary`] as exposed on the snapshot endpoint.
///
/// Mirrors a subset of [`crate::runner::PlanRunStatus`] — only values that
/// can appear for *active* runs are kept. We keep the type opaque to the
/// REST surface so future internal additions don't leak through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotPlanRunStatus {
    Running,
}

/// Lightweight projection of a `PlanRun` for the snapshot endpoint.
///
/// Field selection rules:
/// - **No raw `state_json`** — clients should subscribe to `/ws/activity`
///   for live mutation; the snapshot only carries enough to draw the row.
/// - **Counters as `usize`** so the JSON shape matches what the live
///   `RunnerEvent::WaveStarted` / `TaskCompleted` deltas send.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanRunSummary {
    /// Unique identifier of the run.
    pub run_id: Uuid,
    /// Plan being executed.
    pub plan_id: Uuid,
    /// Plan title — pre-joined so the frontend doesn't need a second roundtrip.
    pub plan_title: String,
    /// Total task count expected for the run.
    pub total_tasks: usize,
    /// Index of the wave currently executing (1-based).
    pub current_wave: usize,
    /// Number of completed tasks.
    pub completed_tasks: usize,
    /// Number of failed tasks (does not include still-pending ones).
    pub failed_tasks: usize,
    /// Run status — restricted to active values on the snapshot.
    pub status: SnapshotPlanRunStatus,
    /// Cost accumulated so far (USD).
    pub cost_usd: f64,
    /// When the run started.
    pub started_at: DateTime<Utc>,
    /// Currently-executing task id if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_task_id: Option<Uuid>,
    /// Currently-executing task title if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_task_title: Option<String>,
    /// Git branch the runner is operating on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_branch: Option<String>,
}

// ============================================================================
// ProtocolRun summary (running only)
// ============================================================================

/// Status restricted to active values on the snapshot endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotProtocolRunStatus {
    Running,
}

/// Lightweight projection of a `ProtocolRun` for the snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProtocolRunSummary {
    /// Run identifier.
    pub id: Uuid,
    /// Protocol the run instantiates.
    pub protocol_id: Uuid,
    /// Protocol name — pre-joined to avoid an extra fetch.
    pub protocol_name: String,
    /// Current FSM state id.
    pub current_state: Uuid,
    /// Human-readable name of the current state.
    pub state_name: String,
    /// Status — restricted to `running` on the snapshot.
    pub status: SnapshotProtocolRunStatus,
    /// Number of state transitions performed so far.
    pub states_visited: usize,
    /// When the run started.
    pub started_at: DateTime<Utc>,
    /// Optional plan context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_id: Option<Uuid>,
    /// Optional task context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<Uuid>,
    /// Nesting depth (0 = root, 1 = child of root, …).
    pub depth: u32,
}

// ============================================================================
// ChatSession summary (recently active)
// ============================================================================

/// Lightweight projection of a `ChatSession` for the snapshot.
///
/// "Active" is defined as "has been updated recently" — the snapshot endpoint
/// caps the list (by `updated_at desc`) so a project with thousands of old
/// sessions doesn't blow up the payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatSessionSummary {
    /// Session identifier.
    pub id: Uuid,
    /// Optional title (auto-generated or user-provided).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Model used.
    pub model: String,
    /// Number of messages exchanged.
    pub message_count: i64,
    /// Last update timestamp.
    pub updated_at: DateTime<Utc>,
    /// Total cost in USD (if tracked).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_cost_usd: Option<f64>,
    /// Free-form preview string (first user message, truncated).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview: Option<String>,
    /// Project slug the session belongs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_slug: Option<String>,
}

// ============================================================================
// Top-level snapshot envelope
// ============================================================================

/// Full snapshot returned by `GET /api/activity/snapshot`.
///
/// Fields are always present (empty arrays when nothing matches). `last_event_seq`
/// is `0` on a freshly-started process — the client should still pass it back as
/// the `lastEventSeq` query param when opening `/ws/activity` to enable replay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActivitySnapshot {
    /// All `PlanRun`s currently running for the requested project.
    pub plan_runs: Vec<PlanRunSummary>,
    /// All `ProtocolRun`s currently running for the requested project.
    pub protocol_runs: Vec<ProtocolRunSummary>,
    /// Recently active `ChatSession`s for the project (capped).
    pub chat_sessions: Vec<ChatSessionSummary>,
    /// Head of the global event sequence at snapshot time.
    pub last_event_seq: u64,
}

impl Default for ActivitySnapshot {
    fn default() -> Self {
        Self {
            plan_runs: Vec::new(),
            protocol_runs: Vec::new(),
            chat_sessions: Vec::new(),
            last_event_seq: 0,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn sample_plan_run() -> PlanRunSummary {
        PlanRunSummary {
            run_id: Uuid::nil(),
            plan_id: Uuid::nil(),
            plan_title: "Bootstrap project".into(),
            total_tasks: 7,
            current_wave: 2,
            completed_tasks: 3,
            failed_tasks: 0,
            status: SnapshotPlanRunStatus::Running,
            cost_usd: 0.42,
            started_at: Utc.with_ymd_and_hms(2026, 5, 21, 9, 30, 0).unwrap(),
            current_task_id: Some(Uuid::nil()),
            current_task_title: Some("Add /ws/activity multiplexer".into()),
            git_branch: Some("runner/test".into()),
        }
    }

    fn sample_protocol_run() -> ProtocolRunSummary {
        ProtocolRunSummary {
            id: Uuid::nil(),
            protocol_id: Uuid::nil(),
            protocol_name: "Code Review".into(),
            current_state: Uuid::nil(),
            state_name: "InReview".into(),
            status: SnapshotProtocolRunStatus::Running,
            states_visited: 2,
            started_at: Utc.with_ymd_and_hms(2026, 5, 21, 9, 35, 0).unwrap(),
            plan_id: None,
            task_id: None,
            depth: 0,
        }
    }

    fn sample_chat_session() -> ChatSessionSummary {
        ChatSessionSummary {
            id: Uuid::nil(),
            title: Some("Planning the next sprint".into()),
            model: "claude-sonnet".into(),
            message_count: 12,
            updated_at: Utc.with_ymd_and_hms(2026, 5, 21, 9, 40, 0).unwrap(),
            total_cost_usd: Some(0.18),
            preview: Some("Can you summarize the open tasks?".into()),
            project_slug: Some("project-orchestrator".into()),
        }
    }

    #[test]
    fn snapshot_default_is_empty() {
        let snap = ActivitySnapshot::default();
        assert!(snap.plan_runs.is_empty());
        assert!(snap.protocol_runs.is_empty());
        assert!(snap.chat_sessions.is_empty());
        assert_eq!(snap.last_event_seq, 0);
    }

    #[test]
    fn snapshot_serializes_to_json_with_stable_keys() {
        let snap = ActivitySnapshot {
            plan_runs: vec![sample_plan_run()],
            protocol_runs: vec![sample_protocol_run()],
            chat_sessions: vec![sample_chat_session()],
            last_event_seq: 42,
        };
        let json = serde_json::to_value(&snap).unwrap();

        // Top-level shape
        assert!(json.get("plan_runs").is_some());
        assert!(json.get("protocol_runs").is_some());
        assert!(json.get("chat_sessions").is_some());
        assert_eq!(json["last_event_seq"], 42);

        // PlanRun shape
        let pr = &json["plan_runs"][0];
        assert_eq!(pr["plan_title"], "Bootstrap project");
        assert_eq!(pr["status"], "running");
        assert_eq!(pr["current_wave"], 2);

        // ProtocolRun shape
        let prr = &json["protocol_runs"][0];
        assert_eq!(prr["protocol_name"], "Code Review");
        assert_eq!(prr["state_name"], "InReview");
        assert_eq!(prr["status"], "running");

        // ChatSession shape
        let cs = &json["chat_sessions"][0];
        assert_eq!(cs["model"], "claude-sonnet");
        assert_eq!(cs["message_count"], 12);
    }

    #[test]
    fn snapshot_roundtrip_json() {
        let snap = ActivitySnapshot {
            plan_runs: vec![sample_plan_run()],
            protocol_runs: vec![sample_protocol_run()],
            chat_sessions: vec![sample_chat_session()],
            last_event_seq: 7,
        };
        let json = serde_json::to_string(&snap).unwrap();
        let parsed: ActivitySnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, snap);
    }

    #[test]
    fn plan_run_status_serializes_snake_case() {
        let json = serde_json::to_string(&SnapshotPlanRunStatus::Running).unwrap();
        assert_eq!(json, "\"running\"");
    }

    #[test]
    fn protocol_run_status_serializes_snake_case() {
        let json = serde_json::to_string(&SnapshotProtocolRunStatus::Running).unwrap();
        assert_eq!(json, "\"running\"");
    }

    #[test]
    fn optional_fields_are_omitted_when_none() {
        let mut session = sample_chat_session();
        session.title = None;
        session.total_cost_usd = None;
        session.preview = None;
        session.project_slug = None;
        let json = serde_json::to_string(&session).unwrap();
        assert!(!json.contains("title"));
        assert!(!json.contains("total_cost_usd"));
        assert!(!json.contains("preview"));
        assert!(!json.contains("project_slug"));
    }
}
