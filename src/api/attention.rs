//! Contract of `GET /api/attention` — the cross-workspace "Today" cockpit.
//!
//! This module holds **only the wire DTOs** (no aggregation logic, no
//! handler): it is the contract the frontend builds against while the
//! aggregator is written (plan "Today -> cockpit transversal", task 0.2).
//!
//! Rules of the contract:
//! - every struct and enum is `snake_case` on the wire, explicitly
//!   (`#[serde(rename_all = "snake_case")]`) — no PascalCase leaks;
//! - unknown fields are rejected on deserialization (`deny_unknown_fields`),
//!   so a field added on one side only fails the shared-fixture tests;
//! - `Option` fields are always emitted (`null`), never skipped: the shape of
//!   the JSON does not depend on the data;
//! - every list is sorted by **waiting age, oldest first** (not by priority);
//!   [`AttentionResponse::sort_by_age`] enforces it;
//! - timestamps are RFC 3339 UTC; `age_secs` is computed server-side against
//!   `generated_at`, so clients never need a synchronised clock.
//!
//! The shared fixtures live in `tests/fixtures/attention/*.json` and are
//! byte-identical copies of the frontend's
//! `src/services/__fixtures__/attention/*.json` (see `MANIFEST.sha256`).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A workspace as shown in lanes and referenced by threads / requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct WorkspaceRef {
    pub id: Uuid,
    pub slug: String,
    pub name: String,
}

/// The four bands of the cockpit. A thread sits in exactly one band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Band {
    /// A LIVE agent is stopped on the user (permission / question).
    Waiting,
    /// Active run, streaming sessions, running protocols.
    Running,
    /// Failed / over budget run, blocked task, orphan request.
    Stuck,
    /// RFCs, decisions, notes to review, alerts — nobody is blocked.
    Thinking,
}

/// Status of one dot in a thread's mini-graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WavePointStatus {
    Done,
    Running,
    Pending,
    /// A live agent is waiting on the user for this task.
    Waiting,
    Failed,
    Blocked,
}

/// Status of a plan run, as far as the cockpit needs to tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
    BudgetExceeded,
    Cancelled,
}

/// Why a thread is in the `stuck` band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StuckReason {
    Failed,
    BudgetExceeded,
    TaskBlocked,
    SessionError,
    /// A request whose CLI died (see `orphans`).
    OrphanRequest,
}

/// Kind of an actionable request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    Permission,
    Question,
}

/// Whether the plan runner is free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunnerStatus {
    Free,
    Busy,
}

/// Kind of an item of the "thinking" band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingKind {
    Rfc,
    Decision,
    NoteReview,
    Alert,
}

/// Which mechanism attaches a chat session to a thread. A session may be
/// attached by several mechanisms at once (one link per mechanism); a session
/// with none is reported in `unattached`, never dropped nor filed at random.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkVia {
    /// `(:ChatSession)-[:SPAWNED_BY_RUN {run_id, plan_id, task_id}]->(:PlanRun)`,
    /// written by `ChatManager::create_session` for every runner session (a
    /// runner has no parent session, so the parent-based `SPAWNED_BY` cannot
    /// carry it). Carries `run_id` and `plan_id` (and `task_id` if known).
    RunnerRun,
    /// `ChatSession.spawned_by` JSON written by the runner
    /// (`{"type":"runner","run_id","plan_id"}`).
    SpawnedByJson,
    /// Explicit session -> task association. Carries `task_id`.
    TaskAssociation,
    /// Explicit session -> plan association. Carries `plan_id`.
    PlanAssociation,
}

/// Liveness of a chat session's CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    /// The CLI is running.
    Live,
    /// The CLI is stopped (a pending request is then an orphan).
    Dead,
}

/// One dot of a wave.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct WavePoint {
    pub task_id: Uuid,
    pub status: WavePointStatus,
}

/// One wave of the embedded summary (avoids N calls to `/plans/{id}/waves`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct WaveSummaryDto {
    /// 1-indexed wave number.
    pub wave_number: u32,
    pub points: Vec<WavePoint>,
}

/// Reference to a plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct PlanRef {
    pub id: Uuid,
    pub title: String,
}

/// Reference to a task (used to NAME blocked tasks, never just count them).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct TaskRef {
    pub id: Uuid,
    pub title: String,
}

/// The plan run attached to a thread.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct RunRef {
    pub id: Uuid,
    pub status: RunStatus,
    pub started_at: DateTime<Utc>,
    pub duration_secs: u64,
    /// Cost so far; updated in place by clients, never interpolated.
    pub cost_usd: f64,
}

/// What "Resume" would do — announced BEFORE the click: the runner skips
/// done AND blocked tasks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct ResumePreview {
    pub done_count: u32,
    /// Blocked tasks the runner will skip — named, to push to unblock first.
    pub skipped_blocked: Vec<TaskRef>,
    pub rerun_count: u32,
}

/// One link between a session and a thread, with its provenance kept.
///
/// `run_id` / `task_id` / `plan_id` are those of the mechanism (always
/// emitted, `null` when the mechanism does not carry them). For a resumed
/// run, an old session keeps the OLD `run_id` while the thread shows the new
/// run: the thread stays the one of the same plan.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct SessionLink {
    pub via: LinkVia,
    pub run_id: Option<Uuid>,
    pub task_id: Option<Uuid>,
    pub plan_id: Option<Uuid>,
}

/// A chat session of a thread, with every link that attaches it. `links` is
/// de-duplicated (one entry per distinct link) and never empty.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct ThreadSession {
    pub id: Uuid,
    pub title: String,
    pub state: SessionState,
    pub links: Vec<SessionLink>,
}

/// The unit of the cockpit: a thread of work (plan + run + sessions).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct Thread {
    pub id: Uuid,
    pub title: String,
    /// Slug of the workspace lane (matches `lanes[].slug`).
    pub workspace: String,
    pub band: Band,
    /// Set when `band == stuck`.
    pub stuck_reason: Option<StuckReason>,
    pub plan: Option<PlanRef>,
    pub run: Option<RunRef>,
    /// Chat sessions linked to the thread (same order as `sessions`).
    pub session_ids: Vec<Uuid>,
    /// The same sessions, each with the links (provenance) that attach it.
    pub sessions: Vec<ThreadSession>,
    /// Since when the thread has been in its band.
    pub since: DateTime<Utc>,
    /// Waiting age in seconds at `generated_at`.
    pub age_secs: u64,
    pub waves: Vec<WaveSummaryDto>,
    /// Blocked tasks of the plan (named).
    pub blocked_tasks: Vec<TaskRef>,
    /// Set when the thread can be resumed.
    pub resume: Option<ResumePreview>,
}

impl Thread {
    /// Canonical form of the sessions: links sorted (mechanism order, then
    /// ids) and de-duplicated, `session_ids` kept in step with `sessions`.
    /// The order of the sessions themselves is the aggregator's (creation).
    pub fn sort_sessions(&mut self) {
        for s in &mut self.sessions {
            s.links.sort();
            s.links.dedup();
        }
        self.session_ids = self.sessions.iter().map(|s| s.id).collect();
    }
}

/// An option offered by a question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct QuestionOption {
    pub label: String,
    pub description: Option<String>,
}

/// An actionable request: a LIVE agent is stopped on the user.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct WaitingRequest {
    /// Permission request id, or id of the question event.
    pub request_id: String,
    pub kind: RequestKind,
    pub session_id: Uuid,
    pub thread_id: Option<Uuid>,
    pub workspace: String,
    /// Tool asking for permission (`kind == permission`).
    pub tool_name: Option<String>,
    /// EXACT text: the command to authorize or the question — never truncated.
    pub text: String,
    /// Offered answers (`kind == question`); empty otherwise.
    pub options: Vec<QuestionOption>,
    /// Sequence number of the stored event (ordering / answered-after check).
    pub seq: u64,
    pub requested_at: DateTime<Utc>,
    pub age_secs: u64,
}

/// A request whose CLI is dead: same shape as [`WaitingRequest`], plus since
/// when the CLI is stopped. Never offers "Allow" — only "Continue".
///
/// (Fields are repeated rather than `flatten`ed: `deny_unknown_fields` does
/// not work through `flatten`.)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct OrphanRequest {
    pub request_id: String,
    pub kind: RequestKind,
    pub session_id: Uuid,
    pub thread_id: Option<Uuid>,
    pub workspace: String,
    pub tool_name: Option<String>,
    pub text: String,
    pub options: Vec<QuestionOption>,
    pub seq: u64,
    pub requested_at: DateTime<Utc>,
    pub age_secs: u64,
    /// Since when the CLI is stopped; `null` when unknown.
    pub cli_stopped_at: Option<DateTime<Utc>>,
}

/// What occupies the (single) plan runner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct RunnerOccupant {
    pub plan_id: Uuid,
    pub plan_title: String,
    pub run_id: Uuid,
    pub workspace: String,
    pub since: DateTime<Utc>,
}

/// State of the plan runner: one active plan run per server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct RunnerState {
    pub status: RunnerStatus,
    /// `Some` iff `status == busy`.
    pub busy_with: Option<RunnerOccupant>,
}

/// An item of the "thinking" band (RFC, decision, note to review, alert).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct ThinkingItem {
    pub id: String,
    pub kind: ThinkingKind,
    pub title: String,
    /// Workspace slug when the item belongs to one.
    pub workspace: Option<String>,
    /// Entity status as stored (`proposed`, `needs_review`, ...).
    pub status: String,
    pub thread_id: Option<Uuid>,
    pub since: DateTime<Utc>,
    pub age_secs: u64,
}

/// A chat session with NO link to any thread (free conversation). It is never
/// ignored: it comes out in its workspace lane with its pending requests, so
/// a question or permission it is waiting on cannot get lost. Its pending
/// requests are listed here only (not repeated in `waiting` / `orphans`);
/// `state` tells whether the CLI is alive (answer) or dead (resume).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct UnattachedSession {
    pub id: Uuid,
    /// Slug of the lane (matches `lanes[].slug`).
    pub workspace_slug: String,
    pub title: String,
    pub state: SessionState,
    /// Pending requests (`thread_id` is always `null`); empty when none.
    pub pending: Vec<WaitingRequest>,
    /// Since when the session waits (oldest pending request, else last activity).
    pub since: DateTime<Utc>,
    pub age_secs: u64,
}

/// Full response of `GET /api/attention`: everything the page displays,
/// without a second call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct AttentionResponse {
    pub generated_at: DateTime<Utc>,
    /// Workspace lanes (all workspaces that have something to show).
    pub lanes: Vec<WorkspaceRef>,
    pub threads: Vec<Thread>,
    /// Actionable requests (live session) — band 1.
    pub waiting: Vec<WaitingRequest>,
    /// Requests whose CLI is dead — band 3.
    pub orphans: Vec<OrphanRequest>,
    pub runner: RunnerState,
    pub thinking: Vec<ThinkingItem>,
    /// Sessions without any link, grouped by lane — never dropped.
    pub unattached: Vec<UnattachedSession>,
}

impl AttentionResponse {
    /// Sort every list by waiting age, oldest first (ties broken by id so the
    /// order is deterministic). Lanes are sorted by name.
    pub fn sort_by_age(&mut self) {
        self.lanes
            .sort_by(|a, b| a.name.cmp(&b.name).then(a.id.cmp(&b.id)));
        self.threads
            .sort_by(|a, b| b.age_secs.cmp(&a.age_secs).then(a.id.cmp(&b.id)));
        self.waiting.sort_by(|a, b| {
            b.age_secs
                .cmp(&a.age_secs)
                .then_with(|| a.request_id.cmp(&b.request_id))
        });
        self.orphans.sort_by(|a, b| {
            b.age_secs
                .cmp(&a.age_secs)
                .then_with(|| a.request_id.cmp(&b.request_id))
        });
        self.thinking
            .sort_by(|a, b| b.age_secs.cmp(&a.age_secs).then_with(|| a.id.cmp(&b.id)));
        for t in &mut self.threads {
            t.sort_sessions();
        }
        // Unattached: grouped by lane (lane order = sorted by name), then
        // oldest first, ties broken by id. Unknown lane slug sorts last.
        let lane_ix = |slug: &str| {
            self.lanes
                .iter()
                .position(|l| l.slug == slug)
                .unwrap_or(usize::MAX)
        };
        let mut un = std::mem::take(&mut self.unattached);
        un.sort_by(|a, b| {
            lane_ix(&a.workspace_slug)
                .cmp(&lane_ix(&b.workspace_slug))
                .then(b.age_secs.cmp(&a.age_secs))
                .then(a.id.cmp(&b.id))
        });
        self.unattached = un;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::path::PathBuf;

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/attention")
    }

    /// The data sets (file stems): the 7 original ones, then the 3 of the
    /// session-link amendment.
    const DATASETS: [&str; 10] = [
        "empty",
        "one_band",
        "four_bands",
        "runner_busy",
        "orphan",
        "blocked_task",
        "forty_threads",
        "multi_link",
        "unattached_waiting",
        "resumed_run",
    ];

    fn ser<T: Serialize>(v: T) -> Value {
        serde_json::to_value(v).unwrap()
    }

    // ---- one serialization test per enum: the wire form is snake_case ----

    #[test]
    fn band_serializes_snake_case() {
        let all = [Band::Waiting, Band::Running, Band::Stuck, Band::Thinking];
        assert_eq!(ser(all), json!(["waiting", "running", "stuck", "thinking"]));
    }

    #[test]
    fn wave_point_status_serializes_snake_case() {
        use WavePointStatus::*;
        let got = ser([Done, Running, Pending, Waiting, Failed, Blocked]);
        assert_eq!(
            got,
            json!(["done", "running", "pending", "waiting", "failed", "blocked"])
        );
    }

    #[test]
    fn run_status_serializes_snake_case() {
        use RunStatus::*;
        let got = ser([Running, Completed, Failed, BudgetExceeded, Cancelled]);
        assert_eq!(
            got,
            json!([
                "running",
                "completed",
                "failed",
                "budget_exceeded",
                "cancelled"
            ])
        );
    }

    #[test]
    fn stuck_reason_serializes_snake_case() {
        use StuckReason::*;
        let got = ser([
            Failed,
            BudgetExceeded,
            TaskBlocked,
            SessionError,
            OrphanRequest,
        ]);
        assert_eq!(
            got,
            json!([
                "failed",
                "budget_exceeded",
                "task_blocked",
                "session_error",
                "orphan_request"
            ])
        );
    }

    #[test]
    fn request_kind_serializes_snake_case() {
        assert_eq!(
            ser([RequestKind::Permission, RequestKind::Question]),
            json!(["permission", "question"])
        );
    }

    #[test]
    fn link_via_serializes_snake_case() {
        use LinkVia::*;
        assert_eq!(
            ser([RunnerRun, SpawnedByJson, TaskAssociation, PlanAssociation]),
            json!([
                "runner_run",
                "spawned_by_json",
                "task_association",
                "plan_association"
            ])
        );
    }

    #[test]
    fn session_state_serializes_snake_case() {
        assert_eq!(
            ser([SessionState::Live, SessionState::Dead]),
            json!(["live", "dead"])
        );
    }

    #[test]
    fn runner_status_serializes_snake_case() {
        assert_eq!(
            ser([RunnerStatus::Free, RunnerStatus::Busy]),
            json!(["free", "busy"])
        );
    }

    #[test]
    fn thinking_kind_serializes_snake_case() {
        use ThinkingKind::*;
        assert_eq!(
            ser([Rfc, Decision, NoteReview, Alert]),
            json!(["rfc", "decision", "note_review", "alert"])
        );
    }

    /// The enum vocabulary is shared with the frontend through `enums.json`.
    #[test]
    fn enums_match_shared_vocabulary() {
        use WavePointStatus as W;
        let shared: Value = serde_json::from_str(
            &std::fs::read_to_string(fixtures_dir().join("enums.json")).unwrap(),
        )
        .unwrap();
        let expected = json!({
            "band": ser([Band::Waiting, Band::Running, Band::Stuck, Band::Thinking]),
            "wave_point_status": ser([W::Done, W::Running, W::Pending, W::Waiting, W::Failed, W::Blocked]),
            "run_status": ser([RunStatus::Running, RunStatus::Completed, RunStatus::Failed, RunStatus::BudgetExceeded, RunStatus::Cancelled]),
            "stuck_reason": ser([StuckReason::Failed, StuckReason::BudgetExceeded, StuckReason::TaskBlocked, StuckReason::SessionError, StuckReason::OrphanRequest]),
            "request_kind": ser([RequestKind::Permission, RequestKind::Question]),
            "runner_status": ser([RunnerStatus::Free, RunnerStatus::Busy]),
            "thinking_kind": ser([ThinkingKind::Rfc, ThinkingKind::Decision, ThinkingKind::NoteReview, ThinkingKind::Alert]),
            "link_via": ser([LinkVia::RunnerRun, LinkVia::SpawnedByJson, LinkVia::TaskAssociation, LinkVia::PlanAssociation]),
            "session_state": ser([SessionState::Live, SessionState::Dead]),
        });
        assert_eq!(shared, expected, "enums.json diverges from the Rust enums");
    }

    #[test]
    fn struct_fields_are_snake_case_and_options_always_emitted() {
        let r = RunnerState {
            status: RunnerStatus::Free,
            busy_with: None,
        };
        assert_eq!(ser(&r), json!({"status": "free", "busy_with": null}));
        let resp = AttentionResponse {
            generated_at: "2026-10-01T08:00:00Z".parse().unwrap(),
            lanes: vec![],
            threads: vec![],
            waiting: vec![],
            orphans: vec![],
            runner: r,
            thinking: vec![],
            unattached: vec![],
        };
        let v = ser(&resp);
        for key in v.as_object().unwrap().keys() {
            assert!(
                key.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "non snake_case key {key}"
            );
        }
        assert_eq!(v["generated_at"], json!("2026-10-01T08:00:00Z"));
    }

    #[test]
    fn unknown_field_is_rejected() {
        let bad = json!({"status": "free", "busy_with": null, "extra": 1});
        assert!(serde_json::from_value::<RunnerState>(bad).is_err());
    }

    #[test]
    fn link_fields_are_always_emitted_and_unknown_ones_rejected() {
        let l = SessionLink {
            via: LinkVia::TaskAssociation,
            run_id: None,
            task_id: Some(Uuid::nil()),
            plan_id: None,
        };
        assert_eq!(
            ser(&l),
            json!({"via": "task_association", "run_id": null,
                   "task_id": "00000000-0000-0000-0000-000000000000", "plan_id": null})
        );
        let bad = json!({"via": "runner_run", "run_id": null, "task_id": null,
                         "plan_id": null, "extra": 1});
        assert!(serde_json::from_value::<SessionLink>(bad).is_err());
        assert!(serde_json::from_value::<LinkVia>(json!("RunnerRun")).is_err());
        assert!(serde_json::from_value::<SessionState>(json!("Live")).is_err());
    }

    #[test]
    fn pascal_case_enum_is_rejected() {
        assert!(serde_json::from_value::<Band>(json!("Waiting")).is_err());
        assert!(serde_json::from_value::<RunStatus>(json!("BudgetExceeded")).is_err());
    }

    // ---- shared fixtures: deserialize + exact round-trip ----

    fn load(name: &str) -> (String, AttentionResponse) {
        let raw = std::fs::read_to_string(fixtures_dir().join(format!("{name}.json")))
            .unwrap_or_else(|e| panic!("fixture {name}: {e}"));
        let parsed: AttentionResponse = serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("fixture {name} does not deserialize: {e}"));
        (raw, parsed)
    }

    #[test]
    fn every_fixture_deserializes_and_round_trips_exactly() {
        for name in DATASETS {
            let (raw, parsed) = load(name);
            let original: Value = serde_json::from_str(&raw).unwrap();
            assert_eq!(
                serde_json::to_value(&parsed).unwrap(),
                original,
                "fixture {name}: Rust output differs from the shared JSON"
            );
        }
    }

    #[test]
    fn fixtures_are_sorted_by_waiting_age() {
        for name in DATASETS {
            let (_, parsed) = load(name);
            let mut sorted = parsed.clone();
            sorted.sort_by_age();
            assert_eq!(
                parsed, sorted,
                "fixture {name} is not in canonical (age desc) order"
            );
        }
    }

    #[test]
    fn fixtures_cover_the_required_scenarios() {
        let (_, empty) = load("empty");
        assert!(empty.lanes.is_empty() && empty.threads.is_empty() && empty.waiting.is_empty());
        assert_eq!(empty.runner.status, RunnerStatus::Free);

        let (_, one) = load("one_band");
        let bands: std::collections::HashSet<_> = one.threads.iter().map(|t| t.band).collect();
        assert_eq!(bands.len(), 1);

        let (_, four) = load("four_bands");
        let bands: std::collections::HashSet<_> = four.threads.iter().map(|t| t.band).collect();
        assert_eq!(bands.len(), 4);
        assert!(!four.waiting.is_empty() && !four.thinking.is_empty());

        let (_, busy) = load("runner_busy");
        assert_eq!(busy.runner.status, RunnerStatus::Busy);
        assert!(busy.runner.busy_with.is_some());

        let (_, orphan) = load("orphan");
        assert!(!orphan.orphans.is_empty());
        assert!(orphan.waiting.is_empty());

        let (_, blocked) = load("blocked_task");
        assert!(blocked.threads.iter().any(
            |t| !t.blocked_tasks.is_empty() && t.stuck_reason == Some(StuckReason::TaskBlocked)
        ));

        let (_, forty) = load("forty_threads");
        assert_eq!(forty.threads.len(), 40);

        // amendment: session links / unattached sessions
        let (_, multi) = load("multi_link");
        let counts: Vec<usize> = multi.threads[0]
            .sessions
            .iter()
            .map(|s| s.links.len())
            .collect();
        assert!(counts.iter().any(|&n| n >= 2), "a session linked twice");
        let (_, un) = load("unattached_waiting");
        assert!(un
            .unattached
            .iter()
            .any(|s| s.state == SessionState::Live && !s.pending.is_empty()));
        assert!(un.unattached.iter().any(|s| s.state == SessionState::Dead));
        let (_, res) = load("resumed_run");
        let t = &res.threads[0];
        let new_run = t.run.as_ref().unwrap().id;
        let run_ids: std::collections::HashSet<_> = t
            .sessions
            .iter()
            .flat_map(|s| s.links.iter().filter_map(|l| l.run_id))
            .collect();
        assert!(run_ids.contains(&new_run) && run_ids.len() == 2);
        let plan_ids: std::collections::HashSet<_> = t
            .sessions
            .iter()
            .flat_map(|s| s.links.iter().filter_map(|l| l.plan_id))
            .collect();
        assert_eq!(plan_ids.len(), 1, "same plan before and after the resume");
    }

    #[test]
    fn fixtures_are_internally_consistent() {
        for name in DATASETS {
            let (_, r) = load(name);
            let slugs: Vec<&str> = r.lanes.iter().map(|l| l.slug.as_str()).collect();
            for t in &r.threads {
                assert!(
                    slugs.contains(&t.workspace.as_str()),
                    "{name}: unknown lane {}",
                    t.workspace
                );
                assert_eq!(
                    t.stuck_reason.is_some(),
                    t.band == Band::Stuck,
                    "{name}: stuck_reason iff stuck"
                );
            }
            assert_eq!(
                r.runner.busy_with.is_some(),
                r.runner.status == RunnerStatus::Busy,
                "{name}: busy_with iff busy"
            );
        }
    }

    /// Session-link rules (note 07909b4a): every session of a thread carries
    /// at least one link, de-duplicated, with the id its mechanism needs; a
    /// session without link is in `unattached` only, with its own requests.
    #[test]
    fn session_links_are_consistent() {
        for name in DATASETS {
            let (_, r) = load(name);
            let mut attached = std::collections::HashSet::new();
            for t in &r.threads {
                let ids: Vec<Uuid> = t.sessions.iter().map(|s| s.id).collect();
                assert_eq!(ids, t.session_ids, "{name}: session_ids == sessions[].id");
                for s in &t.sessions {
                    assert!(attached.insert(s.id), "{name}: session in two threads");
                    assert!(!s.links.is_empty(), "{name}: attached session without link");
                    let uniq: std::collections::HashSet<_> = s.links.iter().collect();
                    assert_eq!(uniq.len(), s.links.len(), "{name}: duplicated link");
                    for l in &s.links {
                        let ok = match l.via {
                            LinkVia::RunnerRun => l.run_id.is_some(),
                            LinkVia::SpawnedByJson => l.run_id.is_some() || l.plan_id.is_some(),
                            LinkVia::TaskAssociation => l.task_id.is_some(),
                            LinkVia::PlanAssociation => l.plan_id.is_some(),
                        };
                        assert!(ok, "{name}: link {:?} lacks its id", l.via);
                    }
                }
            }
            let slugs: Vec<&str> = r.lanes.iter().map(|l| l.slug.as_str()).collect();
            for u in &r.unattached {
                assert!(!attached.contains(&u.id), "{name}: unattached AND attached");
                assert!(slugs.contains(&u.workspace_slug.as_str()), "{name}: lane");
                for p in &u.pending {
                    assert_eq!(p.session_id, u.id, "{name}: pending of another session");
                    assert_eq!(p.thread_id, None, "{name}: unattached request has a thread");
                    assert_eq!(p.workspace, u.workspace_slug);
                }
            }
            // a request attached to a thread points at one of its sessions
            for w in &r.waiting {
                if let Some(tid) = w.thread_id {
                    let t = r.threads.iter().find(|t| t.id == tid).expect("thread");
                    assert!(
                        t.session_ids.contains(&w.session_id),
                        "{name}: {}",
                        w.request_id
                    );
                }
            }
            for w in &r.orphans {
                if let Some(tid) = w.thread_id {
                    let t = r.threads.iter().find(|t| t.id == tid).expect("thread");
                    assert!(
                        t.session_ids.contains(&w.session_id),
                        "{name}: {}",
                        w.request_id
                    );
                }
            }
        }
    }

    #[test]
    fn sort_by_age_dedups_links_and_groups_unattached_by_lane() {
        let (_, mut r) = load("multi_link");
        let s = &mut r.threads[0].sessions[0];
        let dup = s.links[0].clone();
        s.links.push(dup);
        s.links.reverse();
        r.sort_by_age();
        assert_eq!(r.threads[0].sessions[0].links.len(), 3);
        assert_eq!(r.threads[0].sessions[0].links[0].via, LinkVia::RunnerRun);

        let (_, mut u) = load("unattached_waiting");
        u.unattached.reverse();
        u.sort_by_age();
        let lanes: Vec<&str> = u
            .unattached
            .iter()
            .map(|x| x.workspace_slug.as_str())
            .collect();
        assert_eq!(
            lanes,
            ["acme-freelance", "project-orchestrator", "studio-site"]
        );
    }

    // ---- byte-identity of the two copies of the fixtures ----

    fn digest_listing(dir: &std::path::Path) -> String {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.ends_with(".json"))
            .collect();
        names.sort();
        names
            .iter()
            .map(|n| {
                let bytes = std::fs::read(dir.join(n)).unwrap();
                format!("{}  {}\n", hex::encode(Sha256::digest(&bytes)), n)
            })
            .collect()
    }

    /// `MANIFEST.sha256` is copied identically in both repos. A fixture that
    /// is edited here without refreshing the manifest (and copying both to the
    /// frontend) fails here; the frontend runs the same check on its side.
    ///
    /// Refresh: `(cd tests/fixtures/attention && shasum -a 256 *.json > MANIFEST.sha256)`
    /// then copy the `.json` files and the manifest verbatim to the frontend.
    #[test]
    fn fixtures_match_manifest() {
        let dir = fixtures_dir();
        let manifest = std::fs::read_to_string(dir.join("MANIFEST.sha256")).unwrap();
        assert_eq!(
            digest_listing(&dir),
            manifest,
            "fixtures changed without refreshing MANIFEST.sha256 (and the frontend copy)"
        );
    }

    /// When the frontend checkout is reachable (sibling `frontend/` directory,
    /// or `ATTENTION_FRONTEND_FIXTURES`), the two copies must be byte-equal.
    #[test]
    fn fixtures_are_byte_identical_to_frontend_copy_when_reachable() {
        let candidate = std::env::var_os("ATTENTION_FRONTEND_FIXTURES")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../frontend/src/services/__fixtures__/attention")
            });
        if !candidate.is_dir() {
            eprintln!("frontend fixtures not reachable at {candidate:?}: skipped");
            return;
        }
        let mine = fixtures_dir();
        assert_eq!(digest_listing(&mine), digest_listing(&candidate));
        assert_eq!(
            std::fs::read(mine.join("MANIFEST.sha256")).unwrap(),
            std::fs::read(candidate.join("MANIFEST.sha256")).unwrap()
        );
    }
}
