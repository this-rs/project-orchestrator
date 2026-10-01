//! Aggregator behind `GET /api/attention`: ONE request, every workspace.
//!
//! It assembles the sources the cockpit needs into the contract of
//! [`crate::api::attention`]:
//!
//! | source                         | feeds                                   |
//! |--------------------------------|-----------------------------------------|
//! | plans, plan runs               | threads (band 2 / 3), `runner`          |
//! | blocked tasks, task graph      | `blocked_tasks`, `waves`, `resume`      |
//! | chat sessions + links + events | `waiting` / `orphans`, `unattached`     |
//! | protocol runs (running)        | threads (band 2)                        |
//! | RFCs, decisions, notes, alerts | `thinking` (band 4)                     |
//!
//! Placement (a thread is in EXACTLY ONE band, first match wins):
//! 1. `waiting`  - a LIVE session of the plan has a pending request;
//! 2. `running`  - the latest run is running, or a session of the plan streams;
//! 3. `stuck`    - latest run failed / over budget (plan still open), else a
//!    request orphaned by a dead CLI, else a blocked task (plan open).
//!
//! A request of a dead session is never in band 1. RFCs, decisions, notes and
//! alerts never make a thread: they are `thinking` items.
//!
//! Number of store reads is CONSTANT: the sources are read in one concurrent
//! round, then links + events (grouped by session ids), then the task graph
//! (grouped by plan ids) - whatever the number of plans or sessions. A source
//! that fails does not fail the response: it is reported in `source_errors`
//! with the bands it degrades, the others are served.

use std::collections::{BTreeMap, HashMap, HashSet};

use axum::extract::{Query, State};
use axum::Json;
use chrono::{DateTime, Duration, Utc};
use serde::Deserialize;
use uuid::Uuid;

use super::attention::*;
use super::handlers::{normalize_slug, AppError, OrchestratorState};
use crate::chat::attachment::{
    attach_sessions, unattached_sessions, AttachedSession, Attachment, UNASSIGNED_LANE,
};
use crate::chat::attention::{derive_attention, SessionAttentionInput};
use crate::neo4j::models::{
    ChatEventRecord, ChatSessionNode, DecisionStatus, PlanNode, PlanStatus, PlansTaskGraph,
    TaskNode, TaskStatus,
};
use crate::neo4j::traits::GraphStore;
use crate::notes::{Note, NoteFilters, NoteStatus, NoteType};
use crate::runner::eligibility::{resume_breakdown, runner_occupancy};
use crate::runner::models::PlanRunStatus;
use crate::runner::RunnerState as PlanRunState;

/// Sessions idle for longer than this (and not live) are out of the cockpit:
/// a question asked months ago to a CLI that is long gone is not actionable.
pub const ATTENTION_WINDOW_DAYS: i64 = 30;

const PLAN_LIMIT: usize = 5000;
const RUN_LIMIT: i64 = 500;
const SESSION_LIMIT: usize = 2000;
const BLOCKED_LIMIT: usize = 2000;
const THINKING_LIMIT: usize = 200;
const PROTOCOL_RUN_LIMIT: usize = 200;

/// Everything the aggregator needs besides the store: the in-memory facts and
/// the clock, supplied by the caller so the function stays testable.
#[derive(Debug, Clone)]
pub struct AttentionParams {
    pub now: DateTime<Utc>,
    /// Exact lane filter (already normalised: never blank).
    pub workspace_slug: Option<String>,
    /// Sessions whose CLI is alive (in memory).
    pub live: HashSet<Uuid>,
    /// Sessions currently streaming (in memory).
    pub streaming: HashSet<Uuid>,
    /// Per live session, the permission `request_id`s its CLI still holds
    /// (in memory). Decides whether a permission followed by a user message
    /// is still actionable.
    pub pending_permissions: HashMap<Uuid, HashSet<String>>,
    /// Snapshot of the global runner slot.
    pub runner: Option<PlanRunState>,
}

#[derive(Default)]
struct Errors(Vec<SourceError>);

impl Errors {
    fn take<T>(&mut self, r: anyhow::Result<T>, source: &str, bands: &[Band]) -> Option<T> {
        match r {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!(source, error = %e, "attention source failed");
                self.0.push(SourceError {
                    source: source.to_string(),
                    bands: bands.to_vec(),
                    message: e.to_string(),
                });
                None
            }
        }
    }
}

fn secs_since(now: DateTime<Utc>, since: DateTime<Utc>) -> u64 {
    u64::try_from((now - since).num_seconds()).unwrap_or(0)
}

fn run_status(s: PlanRunStatus) -> RunStatus {
    match s {
        PlanRunStatus::Running => RunStatus::Running,
        PlanRunStatus::Completed => RunStatus::Completed,
        PlanRunStatus::CompletedWithErrors | PlanRunStatus::Failed => RunStatus::Failed,
        PlanRunStatus::Cancelled => RunStatus::Cancelled,
        PlanRunStatus::BudgetExceeded => RunStatus::BudgetExceeded,
    }
}

fn plan_is_open(s: &PlanStatus) -> bool {
    matches!(s, PlanStatus::Approved | PlanStatus::InProgress)
}

fn task_title(t: &TaskNode) -> String {
    t.title.clone().unwrap_or_else(|| "untitled".to_string())
}

fn point_status(s: &TaskStatus, waiting: bool) -> WavePointStatus {
    match s {
        TaskStatus::Completed => WavePointStatus::Done,
        _ if waiting => WavePointStatus::Waiting,
        TaskStatus::InProgress => WavePointStatus::Running,
        TaskStatus::Pending => WavePointStatus::Pending,
        TaskStatus::Failed => WavePointStatus::Failed,
        TaskStatus::Blocked => WavePointStatus::Blocked,
    }
}

/// Dependency levels of a plan, as the mini-graph needs them (Kahn levels:
/// wave 1 = no dependency). File-conflict splitting of the runner is not
/// replayed here. A cycle does not fail: the tasks caught in it close the
/// summary in one last wave. `edges` are `(from, to)`: `from` depends on `to`.
pub fn summarize_waves(
    tasks: &[&TaskNode],
    edges: &[(Uuid, Uuid)],
    waiting_tasks: &HashSet<Uuid>,
) -> Vec<WaveSummaryDto> {
    let by_id: HashMap<Uuid, &TaskNode> = tasks.iter().map(|t| (t.id, *t)).collect();
    let mut deps: HashMap<Uuid, HashSet<Uuid>> =
        by_id.keys().map(|k| (*k, HashSet::new())).collect();
    for (from, to) in edges {
        if by_id.contains_key(from) && by_id.contains_key(to) && from != to {
            deps.entry(*from).or_default().insert(*to);
        }
    }
    let mut placed: HashSet<Uuid> = HashSet::new();
    let mut waves: Vec<Vec<&TaskNode>> = Vec::new();
    loop {
        let level: Vec<&TaskNode> = by_id
            .values()
            .copied()
            .filter(|t| !placed.contains(&t.id) && deps[&t.id].iter().all(|d| placed.contains(d)))
            .collect();
        if level.is_empty() {
            break;
        }
        placed.extend(level.iter().map(|t| t.id));
        waves.push(level);
    }
    let rest: Vec<&TaskNode> = by_id
        .values()
        .copied()
        .filter(|t| !placed.contains(&t.id))
        .collect();
    if !rest.is_empty() {
        waves.push(rest);
    }
    waves
        .into_iter()
        .enumerate()
        .map(|(i, mut w)| {
            w.sort_by(|a, b| {
                b.priority
                    .unwrap_or(0)
                    .cmp(&a.priority.unwrap_or(0))
                    .then_with(|| task_title(a).cmp(&task_title(b)))
                    .then(a.id.cmp(&b.id))
            });
            WaveSummaryDto {
                wave_number: (i + 1) as u32,
                points: w
                    .iter()
                    .map(|t| WavePoint {
                        task_id: t.id,
                        status: point_status(&t.status, waiting_tasks.contains(&t.id)),
                    })
                    .collect(),
            }
        })
        .collect()
}

/// What decided the band of a plan thread.
struct Draft {
    plan_id: Uuid,
    band: Band,
    stuck_reason: Option<StuckReason>,
    since: DateTime<Utc>,
}

fn oldest(it: impl Iterator<Item = DateTime<Utc>>) -> Option<DateTime<Utc>> {
    it.min()
}

fn note_title(content: &str) -> String {
    let line = content
        .lines()
        .map(|l| l.trim().trim_start_matches('#').trim())
        .find(|l| !l.is_empty())
        .unwrap_or("Untitled");
    line.chars().take(160).collect()
}

fn rfc_title(note: &Note) -> String {
    #[derive(Deserialize)]
    struct C {
        title: String,
    }
    serde_json::from_str::<C>(&note.content)
        .map(|c| c.title)
        .unwrap_or_else(|_| note_title(&note.content))
}

fn rfc_status(note: &Note) -> Option<&str> {
    note.tags.iter().find_map(|t| t.strip_prefix("rfc-status:"))
}

/// Open RFCs (proposed + under review). `list_notes` combines `tags` with AND
/// and an RFC carries exactly one `rfc-status:*` tag, so one query per status
/// and a de-duplicated union (never one query with both tags).
async fn list_open_rfcs(
    graph: &dyn GraphStore,
    ws: Option<&str>,
) -> anyhow::Result<(Vec<Note>, usize)> {
    let filters = |status: &str| NoteFilters {
        note_type: Some(vec![NoteType::Rfc]),
        tags: Some(vec![format!("rfc-status:{status}")]),
        limit: Some(THINKING_LIMIT as i64),
        ..Default::default()
    };
    let (fa, fb) = (filters("proposed"), filters("under_review"));
    let (a, b) = tokio::join!(
        graph.list_notes(None, ws, &fa),
        graph.list_notes(None, ws, &fb)
    );
    let (mut notes, _) = a?;
    let (more, _) = b?;
    let seen: HashSet<Uuid> = notes.iter().map(|n| n.id).collect();
    notes.extend(more.into_iter().filter(|n| !seen.contains(&n.id)));
    let total = notes.len();
    Ok((notes, total))
}

/// Build the whole response. Never fails: a failing source is reported in
/// `source_errors`.
pub async fn build_attention(graph: &dyn GraphStore, p: &AttentionParams) -> AttentionResponse {
    let now = p.now;
    let ws = p.workspace_slug.as_deref();
    let mut errs = Errors::default();

    // ---- round 1: every independent source, concurrently ----
    let (
        workspaces,
        project_rows,
        plans,
        runs,
        blocked,
        sessions,
        proto,
        decisions,
        rfcs,
        notes,
        alerts,
    ) = tokio::join!(
        graph.list_workspaces(),
        graph.list_project_workspace_rows(),
        graph.list_plans_filtered(None, ws, None, None, None, None, PLAN_LIMIT, 0, None, "desc"),
        graph.list_all_plan_runs(RUN_LIMIT, 0, None, ws),
        graph.list_all_tasks_filtered(
            None,
            None,
            ws,
            Some(vec!["blocked".to_string()]),
            None,
            None,
            None,
            None,
            BLOCKED_LIMIT,
            0,
            None,
            "desc",
        ),
        // NOT narrowed by the store: runner sessions carry neither a
        // workspace nor a project slug, so a store-side filter would drop
        // them. The lane is resolved below (plan -> project -> workspace).
        graph.list_chat_sessions(None, None, SESSION_LIMIT, 0, true),
        graph.list_all_protocol_runs(
            Some(crate::protocol::RunStatus::Running),
            None,
            ws,
            PROTOCOL_RUN_LIMIT,
            0,
        ),
        graph.list_decisions_by_status(DecisionStatus::Proposed, None, ws, THINKING_LIMIT, 0),
        list_open_rfcs(graph, ws),
        graph.get_notes_needing_review(None, ws),
        graph.list_alerts(None, ws, THINKING_LIMIT, 0),
    );
    let all = [Band::Waiting, Band::Running, Band::Stuck, Band::Thinking];
    let thread_bands = [Band::Waiting, Band::Running, Band::Stuck];
    let workspaces = errs
        .take(workspaces, "workspaces", &all)
        .unwrap_or_default();
    let project_rows = errs
        .take(project_rows, "project_workspaces", &all)
        .unwrap_or_default();
    let plans: HashMap<Uuid, PlanNode> = errs
        .take(plans, "plans", &thread_bands)
        .map(|(v, _)| v.into_iter().map(|p| (p.id, p)).collect())
        .unwrap_or_default();
    let runs = errs
        .take(runs, "plan_runs", &[Band::Running, Band::Stuck])
        .unwrap_or_default();
    let blocked: Vec<(Uuid, TaskNode)> = errs
        .take(blocked, "blocked_tasks", &[Band::Stuck])
        .map(|(v, _)| v.into_iter().map(|t| (t.plan_id, t.task)).collect())
        .unwrap_or_default();
    let sessions = errs.take(
        sessions,
        "sessions",
        &[Band::Waiting, Band::Running, Band::Stuck],
    );
    let proto = errs
        .take(proto, "protocol_runs", &[Band::Running])
        .map(|(v, _)| v)
        .unwrap_or_default();
    let decisions = errs
        .take(decisions, "decisions", &[Band::Thinking])
        .map(|(v, _)| v)
        .unwrap_or_default();
    let rfcs = errs
        .take(rfcs, "rfcs", &[Band::Thinking])
        .map(|(v, _)| v)
        .unwrap_or_default();
    let notes = errs
        .take(notes, "notes_needing_review", &[Band::Thinking])
        .unwrap_or_default();
    let alerts = errs
        .take(alerts, "alerts", &[Band::Thinking])
        .map(|(v, _)| v)
        .unwrap_or_default();

    // ---- lane lookups ----
    let lane_refs: HashMap<String, WorkspaceRef> = workspaces
        .iter()
        .map(|w| {
            (
                w.slug.clone(),
                WorkspaceRef {
                    id: w.id,
                    slug: w.slug.clone(),
                    name: w.name.clone(),
                },
            )
        })
        .collect();
    let project_lane: HashMap<Uuid, String> = project_rows
        .iter()
        .map(|r| (r.project_id, r.workspace_slug.clone()))
        .collect();
    let project_slug_lane: HashMap<String, String> = project_rows
        .iter()
        .map(|r| (r.project_slug.clone(), r.workspace_slug.clone()))
        .collect();
    let lane_of_project =
        |pid: Option<Uuid>| -> Option<String> { pid.and_then(|p| project_lane.get(&p).cloned()) };

    // Latest run per plan (the store returns them newest first).
    let mut latest_run: HashMap<Uuid, &PlanRunState> = HashMap::new();
    for r in &runs {
        latest_run.entry(r.plan_id).or_insert(r);
    }
    let plan_lane = |plan_id: Uuid| -> String {
        plans
            .get(&plan_id)
            .and_then(|pl| lane_of_project(pl.project_id))
            .or_else(|| {
                latest_run
                    .get(&plan_id)
                    .and_then(|r| lane_of_project(r.project_id))
            })
            .unwrap_or_else(|| UNASSIGNED_LANE.to_string())
    };
    let plan_title = |plan_id: Uuid| -> String {
        plans
            .get(&plan_id)
            .map(|pl| pl.title.clone())
            .unwrap_or_else(|| format!("Plan {}", &plan_id.to_string()[..8]))
    };

    // ---- round 2: links + events of the sessions in the window ----
    let window_start = now - Duration::days(ATTENTION_WINDOW_DAYS);
    let sessions: Vec<ChatSessionNode> = sessions
        .map(|(v, _)| v)
        .unwrap_or_default()
        .into_iter()
        .filter(|s| p.live.contains(&s.id) || s.updated_at >= window_start)
        .collect();
    let mut attachment = Attachment::default();
    let mut events: Vec<ChatEventRecord> = Vec::new();
    if !sessions.is_empty() {
        let ids: Vec<Uuid> = sessions.iter().map(|s| s.id).collect();
        let (att, evs) = tokio::join!(
            attach_sessions(graph, &sessions),
            graph.get_attention_events(&ids)
        );
        // Without the links every session would look free: serve none.
        if let Some(a) = errs.take(
            att,
            "session_links",
            &[Band::Waiting, Band::Running, Band::Stuck],
        ) {
            attachment = a;
            events = errs
                .take(evs, "session_events", &[Band::Waiting, Band::Stuck])
                .unwrap_or_default();
        }
    }

    // Sessions by thread, requests derived once for all of them.
    let thread_sessions: &BTreeMap<Uuid, Vec<AttachedSession>> = &attachment.by_plan;
    let thread_session_ids: HashSet<Uuid> = thread_sessions
        .values()
        .flatten()
        .map(|a| a.session.id)
        .collect();
    let (thread_events, free_events): (Vec<_>, Vec<_>) = events
        .into_iter()
        .partition(|e| thread_session_ids.contains(&e.session_id));
    let inputs: Vec<SessionAttentionInput> = thread_sessions
        .iter()
        .flat_map(|(plan_id, atts)| {
            let lane = plan_lane(*plan_id);
            let live = &p.live;
            atts.iter().map(move |a| SessionAttentionInput {
                session_id: a.session.id,
                workspace: lane.clone(),
                thread_id: Some(*plan_id),
                alive: live.contains(&a.session.id),
                pending_in_memory: p
                    .pending_permissions
                    .get(&a.session.id)
                    .cloned()
                    .unwrap_or_default(),
            })
        })
        .collect();
    let derived = derive_attention(&inputs, thread_events, now);

    // ---- blocked tasks by plan (named) ----
    let mut blocked_by_plan: HashMap<Uuid, Vec<&TaskNode>> = HashMap::new();
    for (plan_id, t) in &blocked {
        blocked_by_plan.entry(*plan_id).or_default().push(t);
    }

    // ---- placement of the plan threads ----
    let mut candidates: HashSet<Uuid> = HashSet::new();
    candidates.extend(derived.waiting.iter().filter_map(|w| w.thread_id));
    candidates.extend(derived.orphans.iter().filter_map(|w| w.thread_id));
    candidates.extend(derived.errors.iter().filter_map(|e| e.thread_id));
    candidates.extend(
        latest_run
            .iter()
            .filter(|(_, r)| r.status == PlanRunStatus::Running)
            .map(|(id, _)| *id),
    );
    for (plan_id, atts) in thread_sessions {
        if atts.iter().any(|a| p.streaming.contains(&a.session.id)) {
            candidates.insert(*plan_id);
        }
    }
    for (plan_id, r) in &latest_run {
        let failed = matches!(
            r.status,
            PlanRunStatus::Failed
                | PlanRunStatus::CompletedWithErrors
                | PlanRunStatus::BudgetExceeded
        );
        if failed
            && plans
                .get(plan_id)
                .is_some_and(|pl| plan_is_open(&pl.status))
        {
            candidates.insert(*plan_id);
        }
    }
    for plan_id in blocked_by_plan.keys() {
        if plans
            .get(plan_id)
            .is_some_and(|pl| plan_is_open(&pl.status))
        {
            candidates.insert(*plan_id);
        }
    }

    let mut drafts: Vec<Draft> = Vec::new();
    for plan_id in &candidates {
        let waiting_since = oldest(
            derived
                .waiting
                .iter()
                .filter(|w| w.thread_id == Some(*plan_id))
                .map(|w| w.requested_at),
        );
        let orphan_since = oldest(
            derived
                .orphans
                .iter()
                .filter(|w| w.thread_id == Some(*plan_id))
                .map(|w| w.requested_at),
        );
        let error_since = oldest(
            derived
                .errors
                .iter()
                .filter(|e| e.thread_id == Some(*plan_id))
                .map(|e| e.occurred_at),
        );
        let run = latest_run.get(plan_id).copied();
        let streaming_since = oldest(
            thread_sessions
                .get(plan_id)
                .into_iter()
                .flatten()
                .filter(|a| p.streaming.contains(&a.session.id))
                .map(|a| a.session.updated_at),
        );
        let open = plans
            .get(plan_id)
            .is_some_and(|pl| plan_is_open(&pl.status));
        let draft = if let Some(since) = waiting_since {
            Draft {
                plan_id: *plan_id,
                band: Band::Waiting,
                stuck_reason: None,
                since,
            }
        } else if run.is_some_and(|r| r.status == PlanRunStatus::Running)
            || streaming_since.is_some()
        {
            let since = run
                .filter(|r| r.status == PlanRunStatus::Running)
                .map(|r| r.started_at)
                .or(streaming_since)
                .unwrap_or(now);
            Draft {
                plan_id: *plan_id,
                band: Band::Running,
                stuck_reason: None,
                since,
            }
        } else if let Some((reason, since)) = run
            .filter(|_| open)
            .and_then(|r| match r.status {
                PlanRunStatus::BudgetExceeded => Some((StuckReason::BudgetExceeded, r)),
                PlanRunStatus::Failed | PlanRunStatus::CompletedWithErrors => {
                    Some((StuckReason::Failed, r))
                }
                _ => None,
            })
            .map(|(reason, r)| (reason, r.completed_at.unwrap_or(r.started_at)))
        {
            Draft {
                plan_id: *plan_id,
                band: Band::Stuck,
                stuck_reason: Some(reason),
                since,
            }
        } else if let Some(since) = orphan_since {
            Draft {
                plan_id: *plan_id,
                band: Band::Stuck,
                stuck_reason: Some(StuckReason::OrphanRequest),
                since,
            }
        } else if let Some(since) = error_since {
            Draft {
                plan_id: *plan_id,
                band: Band::Stuck,
                stuck_reason: Some(StuckReason::SessionError),
                since,
            }
        } else if let Some(since) = open
            .then(|| {
                oldest(
                    blocked_by_plan
                        .get(plan_id)
                        .into_iter()
                        .flatten()
                        .map(|t| t.updated_at.unwrap_or(t.created_at)),
                )
            })
            .flatten()
        {
            Draft {
                plan_id: *plan_id,
                band: Band::Stuck,
                stuck_reason: Some(StuckReason::TaskBlocked),
                since,
            }
        } else {
            continue;
        };
        drafts.push(draft);
    }

    // ---- round 3: the task graph of the threads, grouped ----
    let mut draft_ids: Vec<Uuid> = drafts.iter().map(|d| d.plan_id).collect();
    draft_ids.sort();
    let task_graph: Option<PlansTaskGraph> = if draft_ids.is_empty() {
        Some(PlansTaskGraph::default())
    } else {
        errs.take(
            graph.get_plans_task_graph(&draft_ids).await,
            "task_graph",
            &thread_bands,
        )
    };
    let mut tasks_by_plan: HashMap<Uuid, Vec<&TaskNode>> = HashMap::new();
    let mut plan_of_task: HashMap<Uuid, Uuid> = HashMap::new();
    let mut edges_by_plan: HashMap<Uuid, Vec<(Uuid, Uuid)>> = HashMap::new();
    if let Some(g) = &task_graph {
        for (pid, t) in &g.tasks {
            tasks_by_plan.entry(*pid).or_default().push(t);
            plan_of_task.insert(t.id, *pid);
        }
        for (from, to) in &g.edges {
            if let Some(pid) = plan_of_task.get(from) {
                edges_by_plan.entry(*pid).or_default().push((*from, *to));
            }
        }
    }

    // ---- assemble the threads ----
    let mut threads: Vec<Thread> = Vec::new();
    for d in &drafts {
        let plan = plans.get(&d.plan_id);
        let run = latest_run.get(&d.plan_id).copied();
        let atts: &[AttachedSession] = thread_sessions
            .get(&d.plan_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        // Tasks a live agent is waiting on (the session carries the task id).
        let live_waiting_sessions: HashSet<Uuid> = derived
            .waiting
            .iter()
            .filter(|w| w.thread_id == Some(d.plan_id))
            .map(|w| w.session_id)
            .collect();
        let waiting_tasks: HashSet<Uuid> = atts
            .iter()
            .filter(|a| live_waiting_sessions.contains(&a.session.id))
            .flat_map(|a| a.links.iter().filter_map(|l| l.task_id))
            .collect();
        let tasks = tasks_by_plan.get(&d.plan_id).cloned().unwrap_or_default();
        let waves = summarize_waves(
            &tasks,
            edges_by_plan
                .get(&d.plan_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]),
            &waiting_tasks,
        );

        let resumable = run.is_some_and(|r| {
            matches!(
                r.status,
                PlanRunStatus::Failed
                    | PlanRunStatus::CompletedWithErrors
                    | PlanRunStatus::BudgetExceeded
                    | PlanRunStatus::Cancelled
            )
        }) && plan.is_some_and(|pl| plan_is_open(&pl.status));
        let resume = (resumable && task_graph.is_some() && !tasks.is_empty()).then(|| {
            let wave_tasks: Vec<crate::neo4j::plan::WaveTask> = tasks
                .iter()
                .map(|t| crate::neo4j::plan::WaveTask {
                    id: t.id,
                    title: t.title.clone(),
                    status: t.status.clone(),
                    priority: t.priority,
                    affected_files: Vec::new(),
                    depends_on: Vec::new(),
                })
                .collect();
            resume_breakdown(&wave_tasks).to_preview()
        });

        let mut blocked_tasks: Vec<TaskRef> = blocked_by_plan
            .get(&d.plan_id)
            .into_iter()
            .flatten()
            .map(|t| TaskRef {
                id: t.id,
                title: task_title(t),
            })
            .collect();
        blocked_tasks.sort_by(|a, b| a.title.cmp(&b.title).then(a.id.cmp(&b.id)));

        threads.push(Thread {
            id: d.plan_id,
            title: plan_title(d.plan_id),
            workspace: plan_lane(d.plan_id),
            band: d.band,
            stuck_reason: d.stuck_reason,
            plan: Some(PlanRef {
                id: d.plan_id,
                title: plan_title(d.plan_id),
            }),
            run: run.map(|r| RunRef {
                id: r.run_id,
                status: run_status(r.status),
                started_at: r.started_at,
                duration_secs: secs_since(r.completed_at.unwrap_or(now), r.started_at),
                cost_usd: r.cost_usd,
            }),
            session_ids: Vec::new(),
            sessions: atts
                .iter()
                .map(|a| ThreadSession {
                    id: a.session.id,
                    title: a.session.title.clone().unwrap_or_default(),
                    state: if p.live.contains(&a.session.id) {
                        SessionState::Live
                    } else {
                        SessionState::Dead
                    },
                    links: a.links.clone(),
                })
                .collect(),
            since: d.since,
            age_secs: secs_since(now, d.since),
            waves,
            blocked_tasks,
            resume,
        });
    }

    // Running protocol runs that are not already the lifecycle of a plan thread.
    let thread_plan_ids: HashSet<Uuid> = threads.iter().map(|t| t.id).collect();
    for run in &proto {
        if run
            .plan_id
            .is_some_and(|pid| thread_plan_ids.contains(&pid))
        {
            continue;
        }
        let lane = run
            .plan_id
            .map(plan_lane)
            .filter(|l| l != UNASSIGNED_LANE)
            .or_else(|| ws.map(str::to_string))
            .unwrap_or_else(|| UNASSIGNED_LANE.to_string());
        threads.push(Thread {
            id: run.id,
            title: match run.plan_id {
                Some(pid) => format!("Protocol run - {}", plan_title(pid)),
                None => format!("Protocol run {}", &run.id.to_string()[..8]),
            },
            workspace: lane,
            band: Band::Running,
            stuck_reason: None,
            plan: run
                .plan_id
                .filter(|pid| plans.contains_key(pid))
                .map(|pid| PlanRef {
                    id: pid,
                    title: plan_title(pid),
                }),
            run: None,
            session_ids: Vec::new(),
            sessions: Vec::new(),
            since: run.started_at,
            age_secs: secs_since(now, run.started_at),
            waves: Vec::new(),
            blocked_tasks: Vec::new(),
            resume: None,
        });
    }

    // ---- requests of threads ----
    let waiting = derived.waiting;
    let orphans = derived.orphans;

    // ---- free sessions (no link, or links that resolve to no plan) ----
    let mut free: Vec<ChatSessionNode> = attachment.unattached.clone();
    free.extend(attachment.unresolved.iter().map(|a| a.session.clone()));
    let unattached = unattached_sessions(
        &free,
        free_events,
        &p.live,
        &p.pending_permissions,
        &project_slug_lane,
        now,
    );

    // ---- thinking ----
    let mut thinking: Vec<ThinkingItem> = Vec::new();
    let item = |id: String,
                kind,
                title: String,
                workspace: Option<String>,
                status: &str,
                since: DateTime<Utc>| {
        ThinkingItem {
            id,
            kind,
            title,
            workspace,
            status: status.to_string(),
            thread_id: None,
            since,
            age_secs: secs_since(now, since),
        }
    };
    for n in &rfcs {
        if n.note_type == NoteType::Rfc
            && matches!(rfc_status(n), Some("proposed" | "under_review"))
        {
            thinking.push(item(
                n.id.to_string(),
                ThinkingKind::Rfc,
                rfc_title(n),
                lane_of_project(n.project_id),
                rfc_status(n).unwrap_or("proposed"),
                n.created_at,
            ));
        }
    }
    for d in &decisions {
        if d.decision.status == DecisionStatus::Proposed {
            thinking.push(item(
                d.decision.id.to_string(),
                ThinkingKind::Decision,
                note_title(&d.decision.description),
                lane_of_project(d.project_id),
                "proposed",
                d.decision.decided_at,
            ));
        }
    }
    for n in &notes {
        if n.status == NoteStatus::NeedsReview && n.note_type != NoteType::Rfc {
            thinking.push(item(
                n.id.to_string(),
                ThinkingKind::NoteReview,
                note_title(&n.content),
                lane_of_project(n.project_id),
                "needs_review",
                n.last_confirmed_at.unwrap_or(n.created_at),
            ));
        }
    }
    for a in &alerts {
        if !a.acknowledged {
            thinking.push(item(
                a.id.to_string(),
                ThinkingKind::Alert,
                note_title(&a.message),
                lane_of_project(a.project_id),
                "unacknowledged",
                a.created_at,
            ));
        }
    }

    // ---- runner (a server-wide fact: not narrowed by the lane filter) ----
    let runner = {
        let active = p
            .runner
            .as_ref()
            .filter(|r| r.status == PlanRunStatus::Running);
        match active {
            Some(r) => runner_occupancy(Some(r), &plan_title(r.plan_id), &plan_lane(r.plan_id)),
            None => runner_occupancy(None, "", ""),
        }
    };

    // ---- exact lane filter, whatever the store pushed down ----
    if let Some(slug) = ws {
        threads.retain(|t| t.workspace == slug);
    }
    let (waiting, orphans): (Vec<_>, Vec<_>) = match ws {
        Some(slug) => (
            waiting
                .into_iter()
                .filter(|w| w.workspace == slug)
                .collect(),
            orphans
                .into_iter()
                .filter(|w| w.workspace == slug)
                .collect(),
        ),
        None => (waiting, orphans),
    };
    let mut unattached = unattached;
    if let Some(slug) = ws {
        unattached.retain(|u| u.workspace_slug == slug);
        thinking.retain(|t| t.workspace.as_deref() == Some(slug));
    }

    // ---- lanes: every workspace something is shown in ----
    let mut slugs: HashSet<String> = HashSet::new();
    slugs.extend(threads.iter().map(|t| t.workspace.clone()));
    slugs.extend(waiting.iter().map(|w| w.workspace.clone()));
    slugs.extend(orphans.iter().map(|w| w.workspace.clone()));
    slugs.extend(unattached.iter().map(|u| u.workspace_slug.clone()));
    slugs.extend(thinking.iter().filter_map(|t| t.workspace.clone()));
    let lanes: Vec<WorkspaceRef> = slugs
        .into_iter()
        .map(|slug| {
            lane_refs
                .get(&slug)
                .cloned()
                .unwrap_or_else(|| WorkspaceRef {
                    id: Uuid::nil(),
                    name: if slug == UNASSIGNED_LANE {
                        "Unassigned".to_string()
                    } else {
                        slug.clone()
                    },
                    slug,
                })
        })
        .collect();

    let mut response = AttentionResponse {
        generated_at: now,
        lanes,
        threads,
        waiting,
        orphans,
        runner,
        thinking,
        unattached,
        source_errors: errs.0,
    };
    response.sort_by_age();
    response
}

/// Query of `GET /api/attention`.
#[derive(Debug, Deserialize, Default)]
pub struct AttentionQuery {
    /// Exact lane filter; empty or blank means every workspace.
    pub workspace_slug: Option<String>,
}

/// `GET /api/attention?workspace_slug=` - the cockpit, in one request.
pub async fn get_attention(
    State(state): State<OrchestratorState>,
    Query(q): Query<AttentionQuery>,
) -> Result<Json<AttentionResponse>, AppError> {
    let snap = match state.chat_manager.as_ref() {
        Some(cm) => cm.live_session_snapshot().await,
        None => Default::default(),
    };
    let runner = crate::runner::RUNNER_STATE.read().await.clone();
    let params = AttentionParams {
        now: Utc::now(),
        workspace_slug: normalize_slug(q.workspace_slug.as_deref()).map(str::to_string),
        live: snap.live,
        streaming: snap.streaming,
        pending_permissions: snap.pending_permissions,
        runner,
    };
    Ok(Json(
        build_attention(state.orchestrator.neo4j(), &params).await,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::types::ChatEvent;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::models::{AlertNode, AlertSeverity, SessionLinkKind, SessionLinkRow};
    use crate::notes::{NoteImportance, NoteScope};
    use crate::protocol::{Protocol, ProtocolRun};
    use crate::runner::models::TriggerSource;
    use crate::test_helpers::{
        test_chat_session, test_decision, test_plan, test_project_named, test_task_titled,
        test_workspace,
    };
    use chrono::TimeZone;
    use serde_json::json;

    fn now() -> DateTime<Utc> {
        Utc.timestamp_opt(1_800_000_000, 0).unwrap()
    }

    fn ago(secs: i64) -> DateTime<Utc> {
        now() - Duration::seconds(secs)
    }

    /// Two-lane world on the mock store. Every `add_*` returns the new id.
    struct World {
        g: MockGraphStore,
    }

    struct Lane {
        slug: String,
        project: Uuid,
        project_slug: String,
    }

    impl World {
        fn new() -> Self {
            Self {
                g: MockGraphStore::new(),
            }
        }

        async fn lane(&self, slug: &str) -> Lane {
            let mut ws = test_workspace();
            ws.slug = slug.into();
            ws.name = format!("Lane {slug}");
            self.g.create_workspace(&ws).await.unwrap();
            let project = test_project_named(&format!("proj-{slug}"));
            self.g.create_project(&project).await.unwrap();
            self.g
                .add_project_to_workspace(ws.id, project.id)
                .await
                .unwrap();
            Lane {
                slug: slug.into(),
                project: project.id,
                project_slug: project.slug,
            }
        }

        async fn plan(&self, lane: &Lane, title: &str, status: PlanStatus) -> Uuid {
            let mut plan = test_plan();
            plan.title = title.into();
            plan.status = status;
            self.g.create_plan(&plan).await.unwrap();
            self.g
                .link_plan_to_project(plan.id, lane.project)
                .await
                .unwrap();
            plan.id
        }

        async fn task(&self, plan: Uuid, title: &str, status: TaskStatus) -> Uuid {
            let mut t = test_task_titled(title);
            t.status = status;
            t.updated_at = Some(ago(7200));
            self.g.create_task(plan, &t).await.unwrap();
            t.id
        }

        async fn run(&self, plan: Uuid, lane: &Lane, status: PlanRunStatus, started: i64) -> Uuid {
            let mut r = PlanRunState::new(Uuid::new_v4(), plan, 3, TriggerSource::Manual);
            r.status = status;
            r.started_at = ago(started);
            r.completed_at = (status != PlanRunStatus::Running).then(|| ago(started - 60));
            r.project_id = Some(lane.project);
            r.cost_usd = 1.5;
            self.g.create_plan_run(&r).await.unwrap();
            r.run_id
        }

        /// A runner session of `plan`, last active `idle` seconds ago.
        async fn runner_session(&self, plan: Uuid, run: Uuid, idle: i64) -> Uuid {
            let mut s = test_chat_session(None);
            s.title = Some("runner".into());
            s.created_at = ago(idle + 10);
            s.updated_at = ago(idle);
            s.spawned_by =
                Some(json!({"type": "runner", "run_id": run, "plan_id": plan}).to_string());
            self.g.create_chat_session(&s).await.unwrap();
            s.id
        }

        /// A free session (no link) in `lane`.
        async fn free_session(&self, lane: &Lane, idle: i64) -> Uuid {
            let mut s = test_chat_session(Some(&lane.project_slug));
            s.workspace_slug = Some(lane.slug.clone());
            s.created_at = ago(idle + 10);
            s.updated_at = ago(idle);
            self.g.create_chat_session(&s).await.unwrap();
            s.id
        }

        async fn ask_permission(&self, session: Uuid, id: &str, age: i64) {
            let ev = ChatEvent::PermissionRequest {
                id: id.into(),
                tool: "Bash".into(),
                input: json!({"command": "cargo publish"}),
                parent_tool_use_id: None,
            };
            self.store(session, 1, ev, age).await;
        }

        async fn ask_question(&self, session: Uuid, id: &str, age: i64) {
            let ev = ChatEvent::AskUserQuestion {
                id: id.into(),
                tool_call_id: "tc".into(),
                questions: json!([{"question": "Which database?", "options": [{"label": "pg"}]}]),
                input: json!({}),
                parent_tool_use_id: None,
            };
            self.store(session, 2, ev, age).await;
        }

        async fn store(&self, session: Uuid, seq: i64, ev: ChatEvent, age: i64) {
            let rec = ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: session,
                seq,
                event_type: ev.event_type().to_string(),
                data: serde_json::to_string(&ev).unwrap(),
                created_at: ago(age),
            };
            self.g.store_chat_events(session, vec![rec]).await.unwrap();
        }

        async fn rfc(&self, lane: &Lane, title: &str, status: &str) -> Uuid {
            let n = Note::new_full(
                Some(lane.project),
                NoteType::Rfc,
                NoteImportance::Medium,
                NoteScope::Project,
                json!({"title": title, "sections": []}).to_string(),
                vec![format!("rfc-status:{status}")],
                "t".into(),
            );
            self.g.create_note(&n).await.unwrap();
            n.id
        }

        async fn alert(&self, lane: &Lane, msg: &str, acknowledged: bool) {
            let a = AlertNode {
                id: Uuid::new_v4(),
                alert_type: "git_drift".into(),
                severity: AlertSeverity::Warning,
                message: msg.into(),
                project_id: Some(lane.project),
                acknowledged,
                acknowledged_by: None,
                acknowledged_at: None,
                created_at: ago(500),
                dedup_key: format!("git_drift:{}:{msg}", lane.project),
                occurrence_count: 1,
                first_seen: None,
                last_seen: None,
                priority: 0.5,
            };
            self.g.create_alert(&a).await.unwrap();
        }

        async fn run_attention(
            &self,
            live: &[Uuid],
            streaming: &[Uuid],
            ws: Option<&str>,
        ) -> AttentionResponse {
            self.run_attention_mem(live, streaming, &[], ws).await
        }

        /// `held`: (session, request_id) permissions the live CLIs hold in memory.
        async fn run_attention_mem(
            &self,
            live: &[Uuid],
            streaming: &[Uuid],
            held: &[(Uuid, &str)],
            ws: Option<&str>,
        ) -> AttentionResponse {
            let mut pending_permissions: HashMap<Uuid, HashSet<String>> = HashMap::new();
            for (sid, id) in held {
                pending_permissions
                    .entry(*sid)
                    .or_default()
                    .insert(id.to_string());
            }
            let params = AttentionParams {
                now: now(),
                workspace_slug: ws.map(str::to_string),
                live: live.iter().copied().collect(),
                streaming: streaming.iter().copied().collect(),
                pending_permissions,
                runner: None,
            };
            let r = build_attention(&self.g, &params).await;
            // Every response must hold the contract and survive the wire.
            assert_eq!(contract_violations(&r), Vec::<String>::new());
            let wire = serde_json::to_value(&r).unwrap();
            assert_eq!(
                serde_json::from_value::<AttentionResponse>(wire).unwrap(),
                r
            );
            r
        }
    }

    fn thread(r: &AttentionResponse, id: Uuid) -> &Thread {
        r.threads
            .iter()
            .find(|t| t.id == id)
            .expect("thread present")
    }

    #[tokio::test]
    async fn each_thread_sits_in_one_band_following_the_placement_rules() {
        let w = World::new();
        let acme = w.lane("acme").await;

        // waiting wins over a failed run: ONE thread, in band 1.
        let p_wait = w.plan(&acme, "Waiting plan", PlanStatus::InProgress).await;
        let r_wait = w.run(p_wait, &acme, PlanRunStatus::Failed, 4000).await;
        let s_wait = w.runner_session(p_wait, r_wait, 30).await;
        w.ask_permission(s_wait, "perm-1", 300).await;

        let p_run = w.plan(&acme, "Running plan", PlanStatus::InProgress).await;
        w.run(p_run, &acme, PlanRunStatus::Running, 900).await;

        let p_fail = w.plan(&acme, "Failed plan", PlanStatus::InProgress).await;
        w.run(p_fail, &acme, PlanRunStatus::BudgetExceeded, 8000)
            .await;

        let p_block = w.plan(&acme, "Blocked plan", PlanStatus::InProgress).await;
        w.task(p_block, "needs creds", TaskStatus::Blocked).await;

        // a closed plan with a failed run is history, not a thread
        let p_done = w.plan(&acme, "Done plan", PlanStatus::Completed).await;
        w.run(p_done, &acme, PlanRunStatus::Failed, 9000).await;

        // a dead session with a pending request: orphan, never band 1
        let p_orphan = w.plan(&acme, "Orphan plan", PlanStatus::InProgress).await;
        let r_orphan = w.run(p_orphan, &acme, PlanRunStatus::Completed, 7000).await;
        let s_dead = w.runner_session(p_orphan, r_orphan, 600).await;
        w.ask_question(s_dead, "q-1", 650).await;

        // an RFC to review: thinking only
        let rfc = w.rfc(&acme, "Adopt SQLite", "proposed").await;

        let r = w.run_attention(&[s_wait], &[], None).await;

        assert_eq!(thread(&r, p_wait).band, Band::Waiting);
        assert_eq!(thread(&r, p_run).band, Band::Running);
        let t = thread(&r, p_fail);
        assert_eq!(
            (t.band, t.stuck_reason),
            (Band::Stuck, Some(StuckReason::BudgetExceeded))
        );
        let t = thread(&r, p_block);
        assert_eq!(
            (t.band, t.stuck_reason),
            (Band::Stuck, Some(StuckReason::TaskBlocked))
        );
        assert_eq!(t.blocked_tasks.len(), 1);
        assert_eq!(t.blocked_tasks[0].title, "needs creds");
        let t = thread(&r, p_orphan);
        assert_eq!(
            (t.band, t.stuck_reason),
            (Band::Stuck, Some(StuckReason::OrphanRequest))
        );
        assert!(
            r.threads.iter().all(|t| t.id != p_done),
            "closed plan is not a thread"
        );

        // each thread once
        let ids: HashSet<Uuid> = r.threads.iter().map(|t| t.id).collect();
        assert_eq!(ids.len(), r.threads.len());
        assert_eq!(r.threads.len(), 5);

        // a request of a dead session is never in band 1
        assert!(r.waiting.iter().all(|x| x.session_id != s_dead));
        assert_eq!(r.orphans.len(), 1);
        assert_eq!(r.orphans[0].session_id, s_dead);
        assert_eq!(r.waiting.len(), 1);
        assert_eq!(r.waiting[0].text, "cargo publish");
        assert_eq!(r.waiting[0].thread_id, Some(p_wait));

        // an RFC is never in band 1: it is a thinking item and makes no thread
        assert!(r
            .thinking
            .iter()
            .any(|i| i.id == rfc.to_string() && i.kind == ThinkingKind::Rfc));
        assert!(r
            .threads
            .iter()
            .all(|t| t.id.to_string() != rfc.to_string()));
        assert!(r.waiting.iter().all(|x| x.request_id != rfc.to_string()));
    }

    #[tokio::test]
    async fn proposed_and_under_review_rfcs_are_both_listed() {
        // `list_notes` ANDs the tags (as does the mock): one query per status.
        let w = World::new();
        let acme = w.lane("acme").await;
        let a = w.rfc(&acme, "A", "proposed").await;
        let b = w.rfc(&acme, "B", "under_review").await;
        let c = w.rfc(&acme, "C", "accepted").await;
        let r = w.run_attention(&[], &[], None).await;
        let ids: HashSet<String> = r.thinking.iter().map(|i| i.id.clone()).collect();
        assert!(ids.contains(&a.to_string()), "proposed RFC missing");
        assert!(ids.contains(&b.to_string()), "under_review RFC missing");
        assert!(!ids.contains(&c.to_string()), "accepted RFC is not open");
    }

    #[tokio::test]
    async fn permission_asked_before_a_resume_is_not_actionable() {
        let w = World::new();
        let acme = w.lane("acme").await;
        let p = w.plan(&acme, "Resumed plan", PlanStatus::InProgress).await;
        let run = w.run(p, &acme, PlanRunStatus::Completed, 5000).await;
        let s = w.runner_session(p, run, 5).await;
        w.ask_permission(s, "old", 300).await; // seq 1
                                               // resume_session sends a user_message (seq 3 > 1)
        w.store(
            s,
            3,
            ChatEvent::UserMessage {
                content: "Continue.".into(),
            },
            200,
        )
        .await;
        let r = w.run_attention(&[s], &[], None).await;
        assert!(r.waiting.is_empty(), "pre-resume permission is stale");
        assert!(r.threads.iter().all(|t| t.id != p));
    }

    #[tokio::test]
    async fn live_cli_still_holding_a_permission_keeps_it_actionable_after_a_user_message() {
        let w = World::new();
        let acme = w.lane("acme").await;
        let p = w.plan(&acme, "Typing plan", PlanStatus::InProgress).await;
        let run = w.run(p, &acme, PlanRunStatus::Completed, 5000).await;
        let s = w.runner_session(p, run, 5).await;
        w.ask_permission(s, "p-live", 300).await; // seq 1
        w.store(
            s,
            3,
            ChatEvent::UserMessage {
                content: "any news?".into(),
            },
            200,
        )
        .await;
        let r = w.run_attention_mem(&[s], &[], &[(s, "p-live")], None).await;
        assert_eq!(
            r.waiting.len(),
            1,
            "memory says the requester is still there"
        );
        assert_eq!(r.waiting[0].request_id, "p-live");
        // same events, CLI no longer holds it (resume): stale
        let r = w.run_attention_mem(&[s], &[], &[(s, "other")], None).await;
        assert!(r.waiting.is_empty());
    }

    /// Production path: the death is written by `emit_subprocess_death`
    /// (the function the OOB listener calls), read back by the aggregator's
    /// grouped read and derived into a stuck thread dated by the death.
    #[tokio::test]
    async fn subprocess_death_written_by_the_listener_makes_the_thread_stuck() {
        let w = World::new();
        let acme = w.lane("acme").await;
        let p = w.plan(&acme, "Dying plan", PlanStatus::InProgress).await;
        let run = w.run(p, &acme, PlanRunStatus::Completed, 7000).await;
        let s = w.runner_session(p, run, 100).await;
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let next_seq = std::sync::atomic::AtomicI64::new(1);
        crate::chat::oob_listener::emit_subprocess_death(
            &s.to_string(),
            Some(s),
            &tx,
            &next_seq,
            &w.g,
        )
        .await;
        let r = w.run_attention(&[], &[], None).await;
        let t = thread(&r, p);
        assert_eq!(
            (t.band, t.stuck_reason),
            (Band::Stuck, Some(StuckReason::SessionError))
        );
    }

    #[tokio::test]
    async fn orphan_always_carries_a_cli_stopped_at() {
        let w = World::new();
        let acme = w.lane("acme").await;
        let p = w.plan(&acme, "Orphan plan", PlanStatus::InProgress).await;
        let run = w.run(p, &acme, PlanRunStatus::Completed, 7000).await;
        let s = w.runner_session(p, run, 600).await;
        w.ask_question(s, "q-1", 650).await;
        let r = w.run_attention(&[], &[], None).await;
        assert_eq!(r.orphans.len(), 1);
        assert_eq!(r.orphans[0].cli_stopped_at, Some(ago(650)));
    }

    #[tokio::test]
    async fn cli_dead_on_session_error_is_stuck_session_error() {
        let w = World::new();
        let acme = w.lane("acme").await;
        let p = w.plan(&acme, "Crashed plan", PlanStatus::InProgress).await;
        let run = w.run(p, &acme, PlanRunStatus::Completed, 7000).await;
        let s = w.runner_session(p, run, 100).await;
        w.store(
            s,
            4,
            ChatEvent::SessionError {
                reason: "subprocess_exited".into(),
                message: "gone".into(),
                received_at: ago(120),
            },
            120,
        )
        .await;
        let r = w.run_attention(&[], &[], None).await;
        let t = thread(&r, p);
        assert_eq!(
            (t.band, t.stuck_reason),
            (Band::Stuck, Some(StuckReason::SessionError))
        );
        // restarted since (later user_message): no longer stuck
        w.store(
            s,
            5,
            ChatEvent::UserMessage {
                content: "go".into(),
            },
            50,
        )
        .await;
        let r = w.run_attention(&[], &[], None).await;
        assert!(r.threads.iter().all(|t| t.id != p));
    }

    #[tokio::test]
    async fn streaming_session_puts_its_thread_in_running() {
        let w = World::new();
        let acme = w.lane("acme").await;
        let p = w.plan(&acme, "Quiet plan", PlanStatus::InProgress).await;
        let run = w.run(p, &acme, PlanRunStatus::Completed, 5000).await;
        let s = w.runner_session(p, run, 5).await;
        let r = w.run_attention(&[s], &[s], None).await;
        assert_eq!(thread(&r, p).band, Band::Running);
        let r = w.run_attention(&[s], &[], None).await;
        assert!(
            r.threads.is_empty(),
            "idle live session alone is not attention"
        );
    }

    #[tokio::test]
    async fn everything_is_sorted_by_waiting_age_oldest_first() {
        let w = World::new();
        let acme = w.lane("acme").await;
        let mut sessions = Vec::new();
        for (title, age) in [("young", 100), ("oldest", 9000), ("middle", 2000)] {
            let p = w.plan(&acme, title, PlanStatus::InProgress).await;
            let run = w.run(p, &acme, PlanRunStatus::Running, 20_000).await;
            let s = w.runner_session(p, run, 5).await;
            w.ask_permission(s, &format!("perm-{title}"), age).await;
            sessions.push(s);
        }
        w.alert(&acme, "newer", false).await;
        w.rfc(&acme, "RFC", "proposed").await;
        let r = w.run_attention(&sessions, &[], None).await;
        let titles: Vec<&str> = r.threads.iter().map(|t| t.title.as_str()).collect();
        assert_eq!(titles, ["oldest", "middle", "young"]);
        let ages: Vec<u64> = r.waiting.iter().map(|x| x.age_secs).collect();
        assert_eq!(ages, [9000, 2000, 100]);
        let ages: Vec<u64> = r.threads.iter().map(|t| t.age_secs).collect();
        assert!(ages.windows(2).all(|p| p[0] >= p[1]));
        let ages: Vec<u64> = r.thinking.iter().map(|t| t.age_secs).collect();
        assert!(ages.windows(2).all(|p| p[0] >= p[1]));
    }

    async fn big_world(plans: usize) -> (World, Vec<Uuid>) {
        let w = World::new();
        let acme = w.lane("acme").await;
        let studio = w.lane("studio").await;
        let mut live = Vec::new();
        for i in 0..plans {
            let lane = if i % 2 == 0 { &acme } else { &studio };
            let p = w
                .plan(lane, &format!("plan {i}"), PlanStatus::InProgress)
                .await;
            let a = w.task(p, "a", TaskStatus::Completed).await;
            let b = w.task(p, "b", TaskStatus::Pending).await;
            w.g.add_task_dependency(b, a).await.unwrap();
            w.task(p, "c", TaskStatus::Blocked).await;
            let run = w.run(p, lane, PlanRunStatus::Running, 600).await;
            let s = w.runner_session(p, run, 5).await;
            w.ask_permission(s, &format!("perm-{i}"), 100 + i as i64)
                .await;
            live.push(s);
            w.free_session(lane, 50).await;
        }
        (w, live)
    }

    #[tokio::test]
    async fn store_reads_are_constant_whatever_the_number_of_threads() {
        let (small, live_small) = big_world(3).await;
        let (large, live_large) = big_world(40).await;
        let rs = small.run_attention(&live_small, &[], None).await;
        let rl = large.run_attention(&live_large, &[], None).await;
        assert_eq!(rs.threads.len(), 3);
        assert_eq!(rl.threads.len(), 40);
        assert_eq!(rl.unattached.len(), 40);

        let (reads_small, reads_large) = (small.g.reads(), large.g.reads());
        assert_eq!(
            reads_small.len(),
            reads_large.len(),
            "no N+1: {reads_large:?}"
        );
        let mut names = reads_large.clone();
        // open RFCs: one query per status (2), by design, whatever N is.
        assert_eq!(
            names.iter().copied().filter(|n| *n == "list_notes").count(),
            2
        );
        names.retain(|n| *n != "list_notes");
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(
            names.len(),
            before,
            "each source is read once: {reads_large:?}"
        );
        // grouped link read: one call, never one per session
        assert_eq!(
            large
                .g
                .session_link_reads
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        // the embedded wave summaries come from the one grouped graph read
        assert!(reads_large.contains(&"get_plans_task_graph"));
        assert!(rl.threads.iter().all(|t| !t.waves.is_empty()));
    }

    #[tokio::test]
    async fn workspace_filter_is_exact_on_every_source() {
        let w = World::new();
        let acme = w.lane("acme").await;
        let studio = w.lane("studio").await;
        let mut live = Vec::new();
        for lane in [&acme, &studio] {
            let p = w
                .plan(lane, &format!("plan {}", lane.slug), PlanStatus::InProgress)
                .await;
            let run = w.run(p, lane, PlanRunStatus::Running, 600).await;
            let s = w.runner_session(p, run, 5).await;
            w.ask_permission(s, "perm", 100).await;
            live.push(s);
            let free = w.free_session(lane, 40).await;
            w.ask_question(free, "free-q", 90).await;
            live.push(free);
            w.rfc(lane, &format!("rfc {}", lane.slug), "proposed").await;
            w.alert(lane, &format!("alert {}", lane.slug), false).await;
        }
        let all = w.run_attention(&live, &[], None).await;
        assert_eq!(all.lanes.len(), 2);
        assert_eq!(all.threads.len(), 2);

        for slug in ["acme", "studio"] {
            let r = w.run_attention(&live, &[], Some(slug)).await;
            assert_eq!(r.lanes.len(), 1);
            assert_eq!(r.lanes[0].slug, slug);
            assert!(r.threads.iter().all(|t| t.workspace == slug));
            assert_eq!(r.threads.len(), 1);
            assert!(r.waiting.iter().all(|x| x.workspace == slug));
            assert_eq!(r.waiting.len(), 1);
            assert!(r.unattached.iter().all(|u| u.workspace_slug == slug));
            assert_eq!(r.unattached.len(), 1);
            assert_eq!(r.thinking.len(), 2);
            assert!(r
                .thinking
                .iter()
                .all(|t| t.workspace.as_deref() == Some(slug)));
        }
        let none = w.run_attention(&live, &[], Some("nowhere")).await;
        assert!(none.threads.is_empty() && none.waiting.is_empty() && none.lanes.is_empty());
        assert!(none.unattached.is_empty() && none.thinking.is_empty());
    }

    #[tokio::test]
    async fn a_failing_source_degrades_its_bands_and_spares_the_others() {
        let w = World::new();
        let acme = w.lane("acme").await;
        let p = w.plan(&acme, "Waiting plan", PlanStatus::InProgress).await;
        let run = w.run(p, &acme, PlanRunStatus::Running, 600).await;
        let s = w.runner_session(p, run, 5).await;
        w.ask_permission(s, "perm", 100).await;
        w.rfc(&acme, "RFC", "proposed").await;

        w.g.fail_reads.lock().unwrap().insert("list_all_plan_runs");
        let r = w.run_attention(&[s], &[], None).await;
        assert_eq!(r.source_errors.len(), 1);
        assert_eq!(r.source_errors[0].source, "plan_runs");
        assert_eq!(r.source_errors[0].bands, [Band::Running, Band::Stuck]);
        // band 1 and band 4 are still served
        assert_eq!(thread(&r, p).band, Band::Waiting);
        assert_eq!(r.waiting.len(), 1);
        assert_eq!(r.thinking.len(), 1);

        // and a failing thinking source leaves the threads alone
        w.g.fail_reads.lock().unwrap().clear();
        w.g.fail_reads.lock().unwrap().insert("list_alerts");
        let r = w.run_attention(&[s], &[], None).await;
        assert_eq!(r.source_errors.len(), 1);
        assert_eq!(r.source_errors[0].bands, [Band::Thinking]);
        assert_eq!(r.threads.len(), 1);
        assert_eq!(r.thinking.len(), 1);

        // everything failing still answers (200, empty, all errors listed)
        w.g.fail_reads.lock().unwrap().clear();
        w.g.fail_reads.lock().unwrap().insert("list_chat_sessions");
        w.g.fail_reads
            .lock()
            .unwrap()
            .insert("get_plans_task_graph");
        let r = w.run_attention(&[s], &[], None).await;
        let failed: Vec<&str> = r.source_errors.iter().map(|e| e.source.as_str()).collect();
        assert!(
            failed.contains(&"sessions") && failed.contains(&"task_graph"),
            "{failed:?}"
        );
        assert!(r.waiting.is_empty());
        assert!(
            !r.threads.is_empty(),
            "the running thread survives without sessions"
        );
    }

    #[tokio::test]
    async fn free_and_unresolved_sessions_are_never_lost() {
        let w = World::new();
        let acme = w.lane("acme").await;
        let free = w.free_session(&acme, 60).await;
        w.ask_question(free, "q-free", 120).await;
        // linked to a task that no longer exists: resolves to no plan
        let mut ghost = test_chat_session(None);
        ghost.workspace_slug = Some("acme".into());
        ghost.updated_at = ago(30);
        ghost.created_at = ago(40);
        w.g.create_chat_session(&ghost).await.unwrap();
        w.g.session_link_rows.write().await.push(SessionLinkRow {
            session_id: ghost.id,
            kind: SessionLinkKind::TaskAssociation,
            run_id: None,
            task_id: Some(Uuid::new_v4()),
            plan_id: None,
            thread_plan_id: None,
        });
        let r = w.run_attention(&[free], &[], None).await;
        assert!(r.threads.is_empty());
        let ids: HashSet<Uuid> = r.unattached.iter().map(|u| u.id).collect();
        assert_eq!(ids, [free, ghost.id].into_iter().collect());
        let u = r.unattached.iter().find(|u| u.id == free).unwrap();
        assert_eq!((u.state, u.pending.len()), (SessionState::Live, 1));
        assert_eq!(u.pending[0].text, "Which database?");
        // its request is listed there only
        assert!(r.waiting.is_empty() && r.orphans.is_empty());
    }

    #[tokio::test]
    async fn waves_resume_and_runner_are_embedded_without_per_plan_calls() {
        let w = World::new();
        let acme = w.lane("acme").await;
        let p = w.plan(&acme, "Plan", PlanStatus::InProgress).await;
        let a = w.task(p, "a", TaskStatus::Completed).await;
        let b = w.task(p, "b", TaskStatus::Failed).await;
        let c = w.task(p, "c", TaskStatus::Blocked).await;
        let d = w.task(p, "d", TaskStatus::Pending).await;
        w.g.add_task_dependency(b, a).await.unwrap();
        w.g.add_task_dependency(d, b).await.unwrap();
        let run = w.run(p, &acme, PlanRunStatus::Failed, 3000).await;
        let r = w.run_attention(&[], &[], None).await;
        let t = thread(&r, p);
        assert_eq!(t.band, Band::Stuck);
        assert_eq!(t.stuck_reason, Some(StuckReason::Failed));
        assert_eq!(t.run.as_ref().unwrap().id, run);
        assert_eq!(t.run.as_ref().unwrap().status, RunStatus::Failed);
        assert_eq!(t.run.as_ref().unwrap().duration_secs, 60);
        let numbers: Vec<u32> = t.waves.iter().map(|w| w.wave_number).collect();
        assert_eq!(numbers, [1, 2, 3]);
        let status = |id: Uuid| {
            t.waves
                .iter()
                .flat_map(|w| &w.points)
                .find(|p| p.task_id == id)
                .map(|p| p.status)
        };
        assert_eq!(status(a), Some(WavePointStatus::Done));
        assert_eq!(status(b), Some(WavePointStatus::Failed));
        assert_eq!(status(c), Some(WavePointStatus::Blocked));
        assert_eq!(status(d), Some(WavePointStatus::Pending));
        let resume = t.resume.as_ref().unwrap();
        assert_eq!(resume.done_count, 1);
        assert_eq!(resume.skipped_blocked.len(), 1);
        assert_eq!(resume.rerun_count, 2);

        // the runner busy with this plan
        let mut running = PlanRunState::new(Uuid::new_v4(), p, 4, TriggerSource::Manual);
        running.status = PlanRunStatus::Running;
        let params = AttentionParams {
            now: now(),
            workspace_slug: None,
            live: HashSet::new(),
            streaming: HashSet::new(),
            pending_permissions: HashMap::new(),
            runner: Some(running.clone()),
        };
        let r = build_attention(&w.g, &params).await;
        assert_eq!(r.runner.status, RunnerStatus::Busy);
        let o = r.runner.busy_with.unwrap();
        assert_eq!(
            (o.plan_id, o.plan_title.as_str(), o.workspace.as_str()),
            (p, "Plan", "acme")
        );
    }

    #[tokio::test]
    async fn live_waiting_task_is_marked_waiting_in_its_wave() {
        let w = World::new();
        let acme = w.lane("acme").await;
        let p = w.plan(&acme, "Plan", PlanStatus::InProgress).await;
        let t1 = w.task(p, "t1", TaskStatus::InProgress).await;
        let run = w.run(p, &acme, PlanRunStatus::Running, 900).await;
        let s = w.runner_session(p, run, 5).await;
        w.ask_permission(s, "perm", 100).await;
        w.g.session_link_rows.write().await.push(SessionLinkRow {
            session_id: s,
            kind: SessionLinkKind::RunRelation,
            run_id: Some(run),
            task_id: Some(t1),
            plan_id: Some(p),
            thread_plan_id: Some(p),
        });
        let r = w.run_attention(&[s], &[], None).await;
        let t = thread(&r, p);
        assert_eq!(t.waves[0].points[0].status, WavePointStatus::Waiting);
        assert_eq!(
            t.sessions[0].links.len(),
            2,
            "JSON + run relation, both kept"
        );
    }

    #[tokio::test]
    async fn thinking_items_cover_rfc_decision_note_and_alert() {
        let w = World::new();
        let acme = w.lane("acme").await;
        w.rfc(&acme, "Under review", "under_review").await;
        w.rfc(&acme, "Just a draft", "draft").await;
        w.alert(&acme, "ack'd", true).await;
        w.alert(&acme, "open alert", false).await;
        let mut stale = Note::new_full(
            Some(acme.project),
            NoteType::Tip,
            NoteImportance::Medium,
            NoteScope::Project,
            "# Stale tip\nbody".into(),
            vec![],
            "t".into(),
        );
        stale.status = NoteStatus::NeedsReview;
        w.g.create_note(&stale).await.unwrap();
        let plan = w.plan(&acme, "P", PlanStatus::Draft).await;
        let task = w.task(plan, "t", TaskStatus::Pending).await;
        let mut dec = test_decision("Use Postgres", "r");
        dec.status = DecisionStatus::Proposed;
        w.g.create_decision(task, &dec).await.unwrap();

        let r = w.run_attention(&[], &[], None).await;
        let kinds: Vec<(ThinkingKind, String)> = r
            .thinking
            .iter()
            .map(|t| (t.kind, t.title.clone()))
            .collect();
        for expected in [
            (ThinkingKind::Rfc, "Under review"),
            (ThinkingKind::Alert, "open alert"),
            (ThinkingKind::NoteReview, "Stale tip"),
            (ThinkingKind::Decision, "Use Postgres"),
        ] {
            assert!(
                kinds.contains(&(expected.0, expected.1.to_string())),
                "{kinds:?}"
            );
        }
        assert_eq!(
            r.thinking.len(),
            4,
            "draft RFC and acknowledged alert are out: {kinds:?}"
        );
        assert!(r
            .thinking
            .iter()
            .all(|t| t.workspace.as_deref() == Some("acme")));
        assert!(r.threads.is_empty());
    }

    #[tokio::test]
    async fn running_protocol_run_is_a_running_thread_unless_a_plan_thread_covers_it() {
        let w = World::new();
        let acme = w.lane("acme").await;
        let proto = Protocol::new(Uuid::nil(), "p", Uuid::new_v4());
        let mut proto = proto;
        proto.project_id = acme.project;
        w.g.upsert_protocol(&proto).await.unwrap();
        let run = ProtocolRun::new(proto.id, proto.entry_state, "start");
        w.g.create_protocol_run(&run).await.unwrap();
        let r = w.run_attention(&[], &[], None).await;
        let t = thread(&r, run.id);
        assert_eq!(t.band, Band::Running);
        assert!(t.plan.is_none());
        assert_eq!(
            t.workspace, UNASSIGNED_LANE,
            "no plan -> no project -> unassigned lane"
        );
        assert!(r.lanes.iter().any(|l| l.slug == UNASSIGNED_LANE));
    }

    #[test]
    fn wave_summary_survives_a_cycle_and_orders_by_dependency() {
        let t = |title: &str, st: TaskStatus| {
            let mut x = test_task_titled(title);
            x.status = st;
            x
        };
        let (a, b, c) = (
            t("a", TaskStatus::Pending),
            t("b", TaskStatus::Pending),
            t("c", TaskStatus::Pending),
        );
        let refs = [&a, &b, &c];
        // a <- b <- c  and a self-consistent cycle must not hang
        let waves = summarize_waves(&refs, &[(b.id, a.id), (c.id, b.id)], &HashSet::new());
        assert_eq!(waves.len(), 3);
        let cyc = summarize_waves(&refs, &[(a.id, b.id), (b.id, a.id)], &HashSet::new());
        assert_eq!(cyc.len(), 2);
        assert_eq!(cyc[1].points.len(), 2, "the cycle closes the summary");
        assert!(summarize_waves(&[], &[], &HashSet::new()).is_empty());
    }
}
