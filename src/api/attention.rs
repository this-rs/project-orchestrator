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
    /// Chat sessions linked to the thread.
    pub session_ids: Vec<Uuid>,
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

    /// The 7 mandatory data sets (file stems).
    const DATASETS: [&str; 7] = [
        "empty",
        "one_band",
        "four_bands",
        "runner_busy",
        "orphan",
        "blocked_task",
        "forty_threads",
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
