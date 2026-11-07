//! Typed transition guards (plan fdf67efc, task G1).
//!
//! A guard is a CLOSED expression the harness evaluates against the graph and the run when a
//! transition is about to fire. It is never free text read by a model, and it is never satisfied
//! by something the agent merely said: every predicate looks at recorded state (the visited
//! states of the run, the linked task and its steps, the child runs).
//!
//! The legacy `ProtocolTransition::guard` string is a descriptive LABEL ("all tasks completed")
//! posed by seeds and skill imports. It is never evaluated and never rejected, so protocols
//! already stored keep their behaviour. A transition that wants to be guarded sets `guard_expr`.
//!
//! Fail closed: a predicate that cannot be evaluated because its input is missing (no linked
//! task, no steps, no child run) is a FAILURE with the reason, never a pass.

use std::future::Future;
use std::pin::Pin;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::models::{ProtocolRun, RunStatus};
use crate::neo4j::models::{StepStatus, TaskStatus};
use crate::neo4j::traits::GraphStore;

/// Deepest nesting of `all` / `any` / `not` accepted at creation.
pub const MAX_GUARD_DEPTH: usize = 4;
/// Most leaf predicates in one guard accepted at creation.
pub const MAX_GUARD_LEAVES: usize = 16;

/// A closed predicate over the graph and the run. No variant evaluates text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Guard {
    /// The run has entered the state named `state` at least once.
    StateVisited { state: String },
    /// The run has entered the state named `state` at most `max` times (loop bound).
    VisitsAtMost { state: String, max: u32 },
    /// The task linked to the run has this status.
    TaskStatusIs { status: TaskStatus },
    /// Every step of the linked task is completed or skipped (and there is at least one).
    StepsAllDone,
    /// The run has child runs and every one of them completed.
    ChildrenAllCompleted,
    /// Every sub-guard holds.
    All { guards: Vec<Guard> },
    /// At least one sub-guard holds.
    Any { guards: Vec<Guard> },
    /// The sub-guard does not hold.
    Not { guard: Box<Guard> },
}

/// Why a predicate did not hold: which one, what was seen, what was required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GuardFailure {
    pub predicate: String,
    pub observed: String,
    pub expected: String,
}

impl std::fmt::Display for GuardFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (observed: {}; expected: {})",
            self.predicate, self.observed, self.expected
        )
    }
}

/// The outcome of evaluating a guard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardVerdict {
    Pass,
    Fail(Vec<GuardFailure>),
}

impl GuardVerdict {
    pub fn is_pass(&self) -> bool {
        matches!(self, GuardVerdict::Pass)
    }

    /// One line naming every failed predicate; empty for a pass.
    pub fn reason(&self) -> String {
        match self {
            GuardVerdict::Pass => String::new(),
            GuardVerdict::Fail(failures) => failures
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join("; "),
        }
    }

    fn fail(
        predicate: impl Into<String>,
        observed: impl Into<String>,
        expected: impl Into<String>,
    ) -> Self {
        GuardVerdict::Fail(vec![GuardFailure {
            predicate: predicate.into(),
            observed: observed.into(),
            expected: expected.into(),
        }])
    }
}

type VerdictFuture<'a> = Pin<Box<dyn Future<Output = Result<GuardVerdict>> + Send + 'a>>;

impl Guard {
    /// Reject a guard that is empty, too deep or too large. Called at creation so a guard that
    /// cannot be evaluated sensibly never reaches the store.
    pub fn validate(&self) -> std::result::Result<(), String> {
        let mut leaves = 0usize;
        self.validate_inner(1, &mut leaves)?;
        if leaves > MAX_GUARD_LEAVES {
            return Err(format!(
                "guard has {leaves} predicates; the limit is {MAX_GUARD_LEAVES}"
            ));
        }
        Ok(())
    }

    fn validate_inner(&self, depth: usize, leaves: &mut usize) -> std::result::Result<(), String> {
        if depth > MAX_GUARD_DEPTH {
            return Err(format!("guard nests deeper than {MAX_GUARD_DEPTH} levels"));
        }
        match self {
            Guard::StateVisited { state } | Guard::VisitsAtMost { state, .. } => {
                *leaves += 1;
                if state.trim().is_empty() {
                    return Err("guard names an empty state".into());
                }
                Ok(())
            }
            Guard::TaskStatusIs { .. } | Guard::StepsAllDone | Guard::ChildrenAllCompleted => {
                *leaves += 1;
                Ok(())
            }
            Guard::All { guards } | Guard::Any { guards } => {
                if guards.is_empty() {
                    return Err("`all` / `any` guard has no sub-guard".into());
                }
                guards
                    .iter()
                    .try_for_each(|g| g.validate_inner(depth + 1, leaves))
            }
            Guard::Not { guard } => guard.validate_inner(depth + 1, leaves),
        }
    }

    /// Short human description, used in prompts and in refusals.
    pub fn describe(&self) -> String {
        match self {
            Guard::StateVisited { state } => format!("state `{state}` visited"),
            Guard::VisitsAtMost { state, max } => {
                format!("state `{state}` visited at most {max} times")
            }
            Guard::TaskStatusIs { status } => format!("linked task is {status:?}").to_lowercase(),
            Guard::StepsAllDone => "all steps of the linked task done".into(),
            Guard::ChildrenAllCompleted => "all child runs completed".into(),
            Guard::All { guards } => format!(
                "all of [{}]",
                guards
                    .iter()
                    .map(Guard::describe)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Guard::Any { guards } => format!(
                "any of [{}]",
                guards
                    .iter()
                    .map(Guard::describe)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Guard::Not { guard } => format!("not ({})", guard.describe()),
        }
    }

    /// Evaluate against the recorded state. No LLM, a bounded number of store reads.
    pub fn evaluate<'a>(
        &'a self,
        store: &'a dyn GraphStore,
        run: &'a ProtocolRun,
    ) -> VerdictFuture<'a> {
        Box::pin(async move {
            match self {
                Guard::StateVisited { state } => {
                    let visited = run.states_visited.iter().any(|v| &v.state_name == state);
                    if visited {
                        return Ok(GuardVerdict::Pass);
                    }
                    let seen: Vec<&str> = run
                        .states_visited
                        .iter()
                        .map(|v| v.state_name.as_str())
                        .collect();
                    Ok(GuardVerdict::fail(
                        self.describe(),
                        format!("visited [{}]", seen.join(", ")),
                        format!("`{state}` among the visited states"),
                    ))
                }
                Guard::VisitsAtMost { state, max } => {
                    let count = run
                        .states_visited
                        .iter()
                        .filter(|v| &v.state_name == state)
                        .count();
                    if count as u64 <= u64::from(*max) {
                        return Ok(GuardVerdict::Pass);
                    }
                    Ok(GuardVerdict::fail(
                        self.describe(),
                        format!("{count} visits"),
                        format!("at most {max}"),
                    ))
                }
                Guard::TaskStatusIs { status } => {
                    let Some(task_id) = run.task_id else {
                        return Ok(GuardVerdict::fail(
                            self.describe(),
                            "no task linked to the run",
                            "a linked task",
                        ));
                    };
                    match store.get_task(task_id).await? {
                        Some(task) if &task.status == status => Ok(GuardVerdict::Pass),
                        Some(task) => Ok(GuardVerdict::fail(
                            self.describe(),
                            format!("task is {:?}", task.status).to_lowercase(),
                            format!("{status:?}").to_lowercase(),
                        )),
                        None => Ok(GuardVerdict::fail(
                            self.describe(),
                            format!("linked task {task_id} not found"),
                            "an existing task",
                        )),
                    }
                }
                Guard::StepsAllDone => {
                    let Some(task_id) = run.task_id else {
                        return Ok(GuardVerdict::fail(
                            self.describe(),
                            "no task linked to the run",
                            "a linked task",
                        ));
                    };
                    let steps = store.get_task_steps(task_id).await?;
                    if steps.is_empty() {
                        return Ok(GuardVerdict::fail(
                            self.describe(),
                            "the task has no steps",
                            "at least one step, all done",
                        ));
                    }
                    let open = steps
                        .iter()
                        .filter(|s| {
                            !matches!(s.status, StepStatus::Completed | StepStatus::Skipped)
                        })
                        .count();
                    if open == 0 {
                        Ok(GuardVerdict::Pass)
                    } else {
                        Ok(GuardVerdict::fail(
                            self.describe(),
                            format!("{open} of {} steps not done", steps.len()),
                            "0 steps open",
                        ))
                    }
                }
                Guard::ChildrenAllCompleted => {
                    let children = store.list_child_runs(run.id).await?;
                    if children.is_empty() {
                        return Ok(GuardVerdict::fail(
                            self.describe(),
                            "the run has no child run",
                            "at least one child run, all completed",
                        ));
                    }
                    let not_done = children
                        .iter()
                        .filter(|c| c.status != RunStatus::Completed)
                        .count();
                    if not_done == 0 {
                        Ok(GuardVerdict::Pass)
                    } else {
                        Ok(GuardVerdict::fail(
                            self.describe(),
                            format!("{not_done} of {} child runs not completed", children.len()),
                            "all completed",
                        ))
                    }
                }
                Guard::All { guards } => {
                    let mut failures = Vec::new();
                    for g in guards {
                        if let GuardVerdict::Fail(mut f) = g.evaluate(store, run).await? {
                            failures.append(&mut f);
                        }
                    }
                    Ok(if failures.is_empty() {
                        GuardVerdict::Pass
                    } else {
                        GuardVerdict::Fail(failures)
                    })
                }
                Guard::Any { guards } => {
                    let mut failures = Vec::new();
                    for g in guards {
                        match g.evaluate(store, run).await? {
                            GuardVerdict::Pass => return Ok(GuardVerdict::Pass),
                            GuardVerdict::Fail(mut f) => failures.append(&mut f),
                        }
                    }
                    Ok(GuardVerdict::Fail(failures))
                }
                Guard::Not { guard } => match guard.evaluate(store, run).await? {
                    GuardVerdict::Pass => Ok(GuardVerdict::fail(
                        self.describe(),
                        format!("`{}` holds", guard.describe()),
                        "it must not hold",
                    )),
                    GuardVerdict::Fail(_) => Ok(GuardVerdict::Pass),
                },
            }
        })
    }
}

/// Parse a guard from the JSON a client sent, validating it. The error names what to fix.
pub fn parse_guard(value: &serde_json::Value) -> std::result::Result<Guard, String> {
    let guard: Guard = serde_json::from_value(value.clone()).map_err(|e| {
        format!(
            "guard_expr is not a valid guard ({e}). Accepted kinds: state_visited, visits_at_most, \
             task_status_is, steps_all_done, children_all_completed, all, any, not"
        )
    })?;
    guard.validate()?;
    Ok(guard)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use uuid::Uuid;

    fn run_visiting(names: &[&str]) -> ProtocolRun {
        let first = names.first().copied().unwrap_or("Start");
        let mut run = ProtocolRun::new(Uuid::new_v4(), Uuid::new_v4(), first);
        for name in names.iter().skip(1) {
            run.visit_state(Uuid::new_v4(), *name, "go");
        }
        run
    }

    fn visited(state: &str) -> Guard {
        Guard::StateVisited {
            state: state.into(),
        }
    }

    #[test]
    fn a_typed_guard_survives_a_json_round_trip() {
        let guard = Guard::All {
            guards: vec![
                visited("Review"),
                Guard::Not {
                    guard: Box::new(Guard::VisitsAtMost {
                        state: "Retry".into(),
                        max: 0,
                    }),
                },
                Guard::StepsAllDone,
            ],
        };
        let json = serde_json::to_value(&guard).unwrap();
        assert_eq!(json["kind"], "all");
        assert_eq!(parse_guard(&json).unwrap(), guard);
    }

    #[test]
    fn an_unknown_kind_is_refused_with_the_accepted_kinds() {
        let err =
            parse_guard(&serde_json::json!({ "kind": "coverage_above", "min": 80 })).unwrap_err();
        assert!(
            err.contains("state_visited"),
            "message must list accepted kinds: {err}"
        );
        assert!(err.contains("children_all_completed"), "{err}");
    }

    #[test]
    fn free_text_is_not_a_guard() {
        assert!(parse_guard(&serde_json::json!("coverage > 80%")).is_err());
    }

    #[test]
    fn empty_combinators_and_empty_state_names_are_refused() {
        assert!(Guard::All { guards: vec![] }.validate().is_err());
        assert!(Guard::Any { guards: vec![] }.validate().is_err());
        assert!(visited("  ").validate().is_err());
    }

    #[test]
    fn depth_and_size_are_bounded() {
        let mut deep = visited("A");
        for _ in 0..MAX_GUARD_DEPTH {
            deep = Guard::Not {
                guard: Box::new(deep),
            };
        }
        assert!(
            deep.validate().is_err(),
            "nesting beyond the limit must be refused"
        );

        let wide = Guard::All {
            guards: (0..=MAX_GUARD_LEAVES)
                .map(|i| visited(&format!("S{i}")))
                .collect(),
        };
        assert!(
            wide.validate().is_err(),
            "more predicates than the limit must be refused"
        );

        let ok = Guard::All {
            guards: (0..MAX_GUARD_LEAVES)
                .map(|i| visited(&format!("S{i}")))
                .collect(),
        };
        assert!(ok.validate().is_ok());
    }

    #[tokio::test]
    async fn state_visited_passes_and_fails_with_the_observed_states() {
        let store = MockGraphStore::new();
        let run = run_visiting(&["Start", "Review"]);
        assert!(visited("Review")
            .evaluate(&store, &run)
            .await
            .unwrap()
            .is_pass());

        let verdict = visited("Deploy").evaluate(&store, &run).await.unwrap();
        assert!(!verdict.is_pass());
        let reason = verdict.reason();
        assert!(reason.contains("Deploy"), "{reason}");
        assert!(
            reason.contains("Start, Review"),
            "observed states must be named: {reason}"
        );
    }

    #[tokio::test]
    async fn visits_at_most_bounds_a_loop() {
        let store = MockGraphStore::new();
        let run = run_visiting(&["Start", "Retry", "Retry", "Retry"]);
        let guard = |max| Guard::VisitsAtMost {
            state: "Retry".into(),
            max,
        };
        assert!(guard(3).evaluate(&store, &run).await.unwrap().is_pass());
        let verdict = guard(2).evaluate(&store, &run).await.unwrap();
        assert!(!verdict.is_pass());
        assert!(
            verdict.reason().contains("3 visits"),
            "{}",
            verdict.reason()
        );
    }

    #[tokio::test]
    async fn task_predicates_fail_closed_without_a_linked_task() {
        let store = MockGraphStore::new();
        let run = run_visiting(&["Start"]);
        for guard in [
            Guard::TaskStatusIs {
                status: TaskStatus::Completed,
            },
            Guard::StepsAllDone,
        ] {
            let verdict = guard.evaluate(&store, &run).await.unwrap();
            assert!(!verdict.is_pass(), "{guard:?} must not pass without a task");
            assert!(
                verdict.reason().contains("no task linked"),
                "{}",
                verdict.reason()
            );
        }
    }

    #[tokio::test]
    async fn children_all_completed_fails_when_there_is_no_child() {
        let store = MockGraphStore::new();
        let run = run_visiting(&["Start"]);
        let verdict = Guard::ChildrenAllCompleted
            .evaluate(&store, &run)
            .await
            .unwrap();
        assert!(!verdict.is_pass());
        assert!(
            verdict.reason().contains("no child run"),
            "{}",
            verdict.reason()
        );
    }

    #[tokio::test]
    async fn combinators_report_every_failed_leaf() {
        let store = MockGraphStore::new();
        let run = run_visiting(&["Start"]);

        let all = Guard::All {
            guards: vec![visited("Start"), visited("A"), visited("B")],
        };
        let reason = all.evaluate(&store, &run).await.unwrap().reason();
        assert!(reason.contains("`A`") && reason.contains("`B`"), "{reason}");
        assert!(
            !reason.contains("`Start`"),
            "a passing leaf is not a failure: {reason}"
        );

        let any = Guard::Any {
            guards: vec![visited("A"), visited("Start")],
        };
        assert!(any.evaluate(&store, &run).await.unwrap().is_pass());
        let none = Guard::Any {
            guards: vec![visited("A"), visited("B")],
        };
        assert!(!none.evaluate(&store, &run).await.unwrap().is_pass());

        let not_visited = Guard::Not {
            guard: Box::new(visited("A")),
        };
        assert!(not_visited.evaluate(&store, &run).await.unwrap().is_pass());
        let not_start = Guard::Not {
            guard: Box::new(visited("Start")),
        };
        assert!(!not_start.evaluate(&store, &run).await.unwrap().is_pass());
    }
    #[test]
    fn every_guard_describes_itself_for_the_refusal_and_the_prompt() {
        let all = Guard::All {
            guards: vec![
                visited("A"),
                Guard::VisitsAtMost {
                    state: "B".into(),
                    max: 2,
                },
                Guard::TaskStatusIs {
                    status: TaskStatus::Completed,
                },
            ],
        };
        let d = all.describe();
        assert!(d.starts_with("all of ["), "{d}");
        assert!(d.contains("`B` visited at most 2 times"), "{d}");
        assert!(d.contains("linked task is completed"), "{d}");

        let any = Guard::Any {
            guards: vec![Guard::StepsAllDone, Guard::ChildrenAllCompleted],
        };
        let d = any.describe();
        assert!(d.contains("any of [all steps"), "{d}");
        assert!(d.contains("all child runs completed"), "{d}");

        let not = Guard::Not {
            guard: Box::new(visited("A")),
        };
        assert_eq!(not.describe(), "not (state `A` visited)");
    }

    #[tokio::test]
    async fn task_status_and_steps_are_read_from_the_linked_task() {
        let store = MockGraphStore::new();
        let mut run = run_visiting(&["Start"]);

        let mut task = crate::test_helpers::test_task();
        task.status = TaskStatus::InProgress;
        let task_id = task.id;
        store.tasks.write().await.insert(task_id, task);
        run.task_id = Some(task_id);

        let completed = Guard::TaskStatusIs {
            status: TaskStatus::Completed,
        };
        let verdict = completed.evaluate(&store, &run).await.unwrap();
        assert!(!verdict.is_pass());
        assert!(
            verdict.reason().contains("task is inprogress"),
            "{}",
            verdict.reason()
        );
        let in_progress = Guard::TaskStatusIs {
            status: TaskStatus::InProgress,
        };
        assert!(in_progress.evaluate(&store, &run).await.unwrap().is_pass());

        // A task id that points to nothing fails closed.
        let mut dangling = run_visiting(&["Start"]);
        dangling.task_id = Some(Uuid::new_v4());
        let verdict = completed.evaluate(&store, &dangling).await.unwrap();
        assert!(
            verdict.reason().contains("not found"),
            "{}",
            verdict.reason()
        );

        // A task with no step fails closed, then with one open step, then all done.
        let verdict = Guard::StepsAllDone.evaluate(&store, &run).await.unwrap();
        assert!(
            verdict.reason().contains("no steps"),
            "{}",
            verdict.reason()
        );

        let mut done = crate::test_helpers::test_step(0, "done");
        done.status = StepStatus::Completed;
        let mut open = crate::test_helpers::test_step(1, "open");
        open.status = StepStatus::Pending;
        let ids = vec![done.id, open.id];
        store.steps.write().await.insert(done.id, done);
        store.steps.write().await.insert(open.id, open.clone());
        store.task_steps.write().await.insert(task_id, ids);
        let verdict = Guard::StepsAllDone.evaluate(&store, &run).await.unwrap();
        assert!(
            verdict.reason().contains("1 of 2 steps not done"),
            "{}",
            verdict.reason()
        );

        open.status = StepStatus::Skipped;
        store.steps.write().await.insert(open.id, open);
        assert!(Guard::StepsAllDone
            .evaluate(&store, &run)
            .await
            .unwrap()
            .is_pass());
    }

    #[tokio::test]
    async fn children_all_completed_needs_every_child_completed() {
        let store = MockGraphStore::new();
        let run = run_visiting(&["Start"]);
        let mut child = run_visiting(&["Start"]);
        child.parent_run_id = Some(run.id);
        child.status = RunStatus::Running;
        store
            .protocol_runs
            .write()
            .await
            .insert(child.id, child.clone());

        let verdict = Guard::ChildrenAllCompleted
            .evaluate(&store, &run)
            .await
            .unwrap();
        assert!(
            verdict.reason().contains("1 of 1 child runs not completed"),
            "{}",
            verdict.reason()
        );

        child.status = RunStatus::Completed;
        store.protocol_runs.write().await.insert(child.id, child);
        assert!(Guard::ChildrenAllCompleted
            .evaluate(&store, &run)
            .await
            .unwrap()
            .is_pass());
    }

    #[tokio::test]
    async fn not_refuses_when_the_inner_guard_holds_and_a_loop_bound_refuses_when_exceeded() {
        let store = MockGraphStore::new();
        let run = run_visiting(&["Start", "Review", "Review"]);

        let not = Guard::Not {
            guard: Box::new(visited("Review")),
        };
        let verdict = not.evaluate(&store, &run).await.unwrap();
        assert!(!verdict.is_pass());
        assert!(
            verdict.reason().contains("it must not hold"),
            "{}",
            verdict.reason()
        );

        let bound = Guard::VisitsAtMost {
            state: "Review".into(),
            max: 1,
        };
        let verdict = bound.evaluate(&store, &run).await.unwrap();
        assert!(
            verdict.reason().contains("2 visits"),
            "{}",
            verdict.reason()
        );
    }
}
