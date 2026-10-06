//! Single source of truth for "will the runner execute this task?".
//!
//! The runner's wave loop AND the resume preview shown by the cockpit both
//! call [`task_eligibility`]. A second copy of the rule would diverge at the
//! first change of the runner and the "Resume" button would lie.
//!
//! Rule: completed tasks are skipped, blocked tasks are skipped (with a
//! warning), everything else (pending, in_progress, failed) is (re)run.

use crate::api::attention::{
    ResumePreview, RunnerOccupant, RunnerState as AttentionRunnerState, RunnerStatus, TaskRef,
};
use crate::neo4j::models::TaskStatus;
use crate::neo4j::plan::WaveTask;
use crate::runner::models::PlanRunStatus;
use crate::runner::state::RunnerState;
use uuid::Uuid;

/// What the runner does with a task on (re)start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eligibility {
    /// Will be executed (pending, in_progress, or failed to retry).
    Run,
    /// Already done: skipped.
    SkipDone,
    /// Blocked: skipped until someone unblocks it.
    SkipBlocked,
    /// Interrupted: its owner disappeared mid-way. NEVER relaunched on the runner's own
    /// initiative: whatever made it look current may be out of date, so somebody resumes it, or
    /// does not. This is the rule that keeps a restart from re-running stale work.
    SkipInterrupted,
}

impl Eligibility {
    pub fn is_run(self) -> bool {
        self == Eligibility::Run
    }
}

/// The one eligibility decision.
pub fn task_eligibility(status: &TaskStatus) -> Eligibility {
    match status {
        TaskStatus::Completed => Eligibility::SkipDone,
        TaskStatus::Blocked => Eligibility::SkipBlocked,
        TaskStatus::Interrupted => Eligibility::SkipInterrupted,
        TaskStatus::Pending | TaskStatus::InProgress | TaskStatus::Failed => Eligibility::Run,
    }
}

/// Full breakdown of a resume, before the click.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResumeBreakdown {
    pub done: u32,
    /// Blocked tasks that will be skipped, named.
    pub blocked: Vec<TaskRef>,
    /// Failed tasks that will be retried.
    pub failed_to_retry: u32,
    /// Never-finished tasks (pending / in_progress) that will run.
    pub todo: u32,
    /// Interrupted tasks that will be skipped, named. Not in the `ResumePreview` contract yet
    /// (it is strict and shared with the frontend): the preview counts them nowhere, so they are
    /// neither re-run nor presented as done; naming them to the user comes with the contract change.
    pub interrupted: Vec<TaskRef>,
}

impl ResumeBreakdown {
    /// Contract shape: `rerun_count` = failed retried + still to do.
    pub fn to_preview(&self) -> ResumePreview {
        ResumePreview {
            done_count: self.done,
            skipped_blocked: self.blocked.clone(),
            rerun_count: self.failed_to_retry + self.todo,
        }
    }
}

/// Classify tasks with the runner's own rule.
pub fn resume_breakdown<'a>(tasks: impl IntoIterator<Item = &'a WaveTask>) -> ResumeBreakdown {
    let mut b = ResumeBreakdown::default();
    for t in tasks {
        match task_eligibility(&t.status) {
            Eligibility::SkipDone => b.done += 1,
            Eligibility::SkipBlocked => b.blocked.push(TaskRef {
                id: t.id,
                title: t.title.clone().unwrap_or_else(|| "untitled".to_string()),
            }),
            Eligibility::SkipInterrupted => b.interrupted.push(TaskRef {
                id: t.id,
                title: t.title.clone().unwrap_or_else(|| "untitled".to_string()),
            }),
            Eligibility::Run => {
                if t.status == TaskStatus::Failed {
                    b.failed_to_retry += 1;
                } else {
                    b.todo += 1;
                }
            }
        }
    }
    b
}

/// Runner state for the cockpit: free, or busy with `{plan_id, title}`.
/// A finished run kept in the global slot does not occupy the runner.
pub fn runner_occupancy(
    active: Option<&RunnerState>,
    plan_title: &str,
    workspace: &str,
) -> AttentionRunnerState {
    match active {
        Some(s) if s.status == PlanRunStatus::Running => AttentionRunnerState {
            status: RunnerStatus::Busy,
            busy_with: Some(RunnerOccupant {
                plan_id: s.plan_id,
                plan_title: plan_title.to_string(),
                run_id: s.run_id,
                workspace: workspace.to_string(),
                since: s.started_at,
            }),
        },
        _ => AttentionRunnerState {
            status: RunnerStatus::Free,
            busy_with: None,
        },
    }
}

/// Plan id of the run currently occupying the runner, if any (to look up its title).
pub async fn busy_plan_id() -> Option<Uuid> {
    let g = crate::runner::runner::RUNNER_STATE.read().await;
    g.as_ref()
        .filter(|s| s.status == PlanRunStatus::Running)
        .map(|s| s.plan_id)
}

/// Current runner state, reading the global slot and the plan title via `graph`.
pub async fn current_runner_state(
    graph: &dyn crate::neo4j::traits::GraphStore,
    workspace: &str,
) -> AttentionRunnerState {
    let snapshot = crate::runner::runner::RUNNER_STATE.read().await.clone();
    let title = match snapshot
        .as_ref()
        .filter(|s| s.status == PlanRunStatus::Running)
    {
        Some(s) => graph
            .get_plan(s.plan_id)
            .await
            .ok()
            .flatten()
            .map(|p| p.title)
            .unwrap_or_else(|| s.plan_id.to_string()),
        None => String::new(),
    };
    runner_occupancy(snapshot.as_ref(), &title, workspace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::plan::Wave;
    use crate::runner::models::TriggerSource;

    fn wt(status: TaskStatus, title: &str) -> WaveTask {
        WaveTask {
            id: Uuid::new_v4(),
            title: Some(title.into()),
            status,
            priority: None,
            affected_files: vec![],
            depends_on: vec![],
        }
    }

    fn plan() -> Vec<WaveTask> {
        vec![
            wt(TaskStatus::Completed, "a"),
            wt(TaskStatus::Completed, "b"),
            wt(TaskStatus::Blocked, "needs creds"),
            wt(TaskStatus::Failed, "flaky"),
            wt(TaskStatus::Pending, "todo1"),
            wt(TaskStatus::InProgress, "todo2"),
        ]
    }

    #[test]
    fn eligibility_rule() {
        assert_eq!(
            task_eligibility(&TaskStatus::Completed),
            Eligibility::SkipDone
        );
        assert_eq!(
            task_eligibility(&TaskStatus::Blocked),
            Eligibility::SkipBlocked
        );
        for s in [
            TaskStatus::Pending,
            TaskStatus::InProgress,
            TaskStatus::Failed,
        ] {
            assert!(task_eligibility(&s).is_run(), "{s:?}");
        }
    }

    #[test]
    fn preview_announces_blocked_skip_and_runner_skips_it() {
        let tasks = plan();
        let b = resume_breakdown(&tasks);
        assert_eq!(b.done, 2);
        assert_eq!(b.blocked.len(), 1);
        assert_eq!(b.blocked[0].title, "needs creds");
        assert_eq!(b.failed_to_retry, 1);
        assert_eq!(b.todo, 2);
        let p = b.to_preview();
        assert_eq!((p.done_count, p.rerun_count), (2, 3));

        // The runner's wave filter uses the very same predicate: the tasks it
        // would run are exactly the preview's rerun set, the blocked one is not in.
        let wave = Wave {
            wave_number: 1,
            task_count: tasks.len(),
            tasks: tasks.clone(),
            split_from_conflicts: false,
        };
        let run = crate::runner::runner::eligible_wave_tasks(&wave);
        assert_eq!(run.len() as u32, p.rerun_count);
        assert!(run.iter().all(|t| t.id != b.blocked[0].id));
    }

    #[test]
    fn an_interrupted_task_is_never_run_on_the_runners_own_initiative() {
        // The point of the status: a restart must not re-run work whose owner disappeared and
        // that may since have been done, abandoned or overtaken. `InProgress` IS re-run by the
        // rule above; `Interrupted` is what stale in-progress work is turned into.
        assert_eq!(
            task_eligibility(&TaskStatus::Interrupted),
            Eligibility::SkipInterrupted
        );
        assert!(!task_eligibility(&TaskStatus::Interrupted).is_run());
        assert!(task_eligibility(&TaskStatus::InProgress).is_run());
    }

    #[test]
    fn interrupted_work_is_neither_done_nor_to_rerun_in_the_preview_and_the_runner_skips_it() {
        let mut tasks = plan();
        let before = resume_breakdown(&tasks).to_preview();
        tasks.push(wt(TaskStatus::Interrupted, "left over"));

        let b = resume_breakdown(&tasks);
        assert_eq!(b.interrupted.len(), 1);
        assert_eq!(b.interrupted[0].title, "left over");
        assert_eq!(
            b.to_preview(),
            before,
            "it changes neither the done nor the rerun count"
        );

        // The runner's own wave filter drops it: the tasks it runs are still the rerun set.
        let wave = Wave {
            wave_number: 1,
            task_count: tasks.len(),
            tasks: tasks.clone(),
            split_from_conflicts: false,
        };
        let run = crate::runner::runner::eligible_wave_tasks(&wave);
        assert_eq!(run.len() as u32, before.rerun_count);
        assert!(run.iter().all(|t| t.id != b.interrupted[0].id));
    }

    #[test]
    fn runner_free_or_busy() {
        let free = runner_occupancy(None, "", "ws");
        assert_eq!(free.status, RunnerStatus::Free);
        assert!(free.busy_with.is_none());

        let mut s = RunnerState::new(Uuid::new_v4(), Uuid::new_v4(), 3, TriggerSource::Manual);
        let busy = runner_occupancy(Some(&s), "My plan", "ws");
        assert_eq!(busy.status, RunnerStatus::Busy);
        let o = busy.busy_with.unwrap();
        assert_eq!((o.plan_id, o.plan_title.as_str()), (s.plan_id, "My plan"));

        s.status = PlanRunStatus::Completed;
        assert_eq!(
            runner_occupancy(Some(&s), "x", "ws").status,
            RunnerStatus::Free
        );
    }
}
