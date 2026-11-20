use super::{
    background_task_is_idle_dead, idle_verdict, process_alive, IdleFacts, IdleLimits, IdleVerdict,
    IDLE_HARD_CAP_FACTOR, PERSON_WAIT_CAP,
};
#[allow(unused_imports)]
use crate::neo4j::traits::*;
use std::time::Duration;

const T: Duration = Duration::from_secs(1800);

fn mins(m: i64) -> chrono::Duration {
    chrono::Duration::minutes(m)
}

#[test]
fn a_task_that_emitted_recently_is_alive_whatever_else_is_known() {
    for process in [None, Some(true), Some(false)] {
        for cli in [false, true] {
            assert!(!background_task_is_idle_dead(
                mins(29),
                mins(30),
                process,
                cli
            ));
        }
    }
}

/// The failure this guards: a build that writes to a file says nothing for
/// 40 minutes. Its process runs → it is not dead.
#[test]
fn a_silent_task_whose_process_runs_is_not_dead() {
    assert!(!background_task_is_idle_dead(
        mins(40),
        mins(30),
        Some(true),
        false
    ));
    // The process is the stronger evidence: it wins over a CLI report
    // that is empty (or that never came).
    assert!(!background_task_is_idle_dead(
        mins(600),
        mins(30),
        Some(true),
        false
    ));
}

#[test]
fn a_silent_task_whose_process_is_gone_is_dead() {
    assert!(background_task_is_idle_dead(
        mins(30),
        mins(30),
        Some(false),
        false
    ));
    // Even if the CLI still reports work: that is some other task.
    assert!(background_task_is_idle_dead(
        mins(31),
        mins(30),
        Some(false),
        true
    ));
}

/// The pid claim often fails ("no new PID in diff"): then the CLI's own
/// report is the only evidence.
#[test]
fn a_silent_task_without_a_pid_follows_the_cli_report() {
    assert!(!background_task_is_idle_dead(
        mins(40),
        mins(30),
        None,
        true
    ));
    // Nothing says it runs: the previous behaviour, unchanged.
    assert!(background_task_is_idle_dead(
        mins(40),
        mins(30),
        None,
        false
    ));
}

#[test]
fn process_alive_tells_a_running_process_from_a_dead_one() {
    assert_eq!(process_alive(None), None);

    #[cfg(unix)]
    {
        assert_eq!(process_alive(Some(std::process::id())), Some(true));
        // kill(0, ..) would address our whole process group.
        assert_eq!(process_alive(Some(0)), Some(false));
        assert_eq!(process_alive(Some(u32::MAX)), Some(false));

        // A child that has exited and been reaped no longer exists.
        let mut child = std::process::Command::new("true")
            .spawn()
            .expect("spawn `true`");
        let pid = child.id();
        child.wait().expect("wait for `true`");
        assert_eq!(process_alive(Some(pid)), Some(false));

        // A child that has exited but NOT been reaped is a zombie: it is dead, and
        // the probe must say so (kill(0) alone still succeeds on it).
        let mut zombie = std::process::Command::new("true")
            .spawn()
            .expect("spawn `true`");
        let zombie_pid = zombie.id();
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(process_alive(Some(zombie_pid)), Some(false));
        zombie.wait().expect("reap the zombie");
    }
}

/// [`idle_verdict`] for the idle cleanup (`patience` = the timeout), nobody waited on.
fn session_is_expired(
    idle: Duration,
    timeout: Duration,
    is_streaming: bool,
    background_tasks: usize,
) -> bool {
    verdict(
        IdleFacts {
            idle,
            is_streaming,
            background_tasks,
            ..IdleFacts::default()
        },
        timeout,
    ) == IdleVerdict::Expired
}

fn verdict(facts: IdleFacts, timeout: Duration) -> IdleVerdict {
    idle_verdict(
        facts,
        IdleLimits {
            patience: timeout,
            timeout,
            person_wait_cap: PERSON_WAIT_CAP,
        },
    )
}

/// A running turn waiting on a permission, idle for `idle`.
fn waiting(idle: Duration, a_person_can_answer: bool) -> IdleFacts {
    IdleFacts {
        idle,
        is_streaming: true,
        background_tasks: 0,
        waiting_on_a_person: true,
        a_person_can_answer,
    }
}

/// What the cleanup feeds `session_is_expired`: the larger of this
/// server's tracking and the CLI's report. With the tracking emptied (the
/// old purge) but the CLI reporting one task, the session is kept.
#[test]
fn idle_session_is_kept_while_the_cli_reports_background_work() {
    let tracked = 0usize;
    let cli_reported = 1usize;
    assert!(!session_is_expired(
        T * 2,
        T,
        false,
        tracked.max(cli_reported)
    ));
    // Without the CLI's report, the emptied tracking alone closes it.
    assert!(session_is_expired(T * 2, T, false, tracked));
}

#[test]
fn active_session_is_kept() {
    assert!(!session_is_expired(Duration::from_secs(60), T, false, 0));
}

#[test]
fn idle_session_without_work_expires() {
    assert!(session_is_expired(T + Duration::from_secs(1), T, false, 0));
}

#[test]
fn idle_session_with_background_work_is_kept() {
    // The failure mode: agent's turn over, sub-agents still running.
    assert!(!session_is_expired(T * 2, T, false, 3));
}

#[test]
fn streaming_session_is_kept() {
    assert!(!session_is_expired(T * 2, T, true, 0));
}

#[test]
fn stuck_background_work_is_bounded_by_the_hard_cap() {
    assert!(session_is_expired(T * 9, T, true, 1));
}

/// The decision: waiting on a person PROLONGS a session — past the ordinary delay and
/// past the hard cap — and never makes it eternal: past the person wait cap it closes.
#[test]
fn a_session_waiting_on_a_person_is_kept_past_the_hard_cap_and_closed_past_the_wait_cap() {
    let past_hard_cap = T * (IDLE_HARD_CAP_FACTOR + 1);
    assert!(past_hard_cap < PERSON_WAIT_CAP);
    assert_eq!(
        verdict(waiting(past_hard_cap, true), T),
        IdleVerdict::WaitingOnAPerson
    );
    assert_eq!(
        verdict(waiting(PERSON_WAIT_CAP, true), T),
        IdleVerdict::WaitingOnAPerson,
        "up to the cap included"
    );
    assert_eq!(
        verdict(waiting(PERSON_WAIT_CAP + Duration::from_secs(1), true), T),
        IdleVerdict::Expired,
        "past the person wait cap, the session is closed"
    );
}

/// A spawned session (a delegated task, a sub-agent, a runner) is never answered by a
/// person: waiting protects it from nothing, the ordinary rules apply — kept while its
/// turn runs, closed at the hard cap.
#[test]
fn a_spawned_session_waiting_is_judged_by_the_ordinary_rules() {
    assert_eq!(verdict(waiting(T * 2, false), T), IdleVerdict::Working);
    assert_eq!(
        verdict(waiting(T * (IDLE_HARD_CAP_FACTOR + 1), false), T),
        IdleVerdict::Expired
    );
}

/// The wait cap prolongs, never shortens: configured below the hard cap, the hard cap
/// applies to a waiting session as to any running one.
#[test]
fn a_wait_cap_below_the_hard_cap_never_shortens_a_session() {
    let limits = IdleLimits {
        patience: T,
        timeout: T,
        person_wait_cap: T * 2,
    };
    assert_eq!(
        idle_verdict(waiting(T * 3, true), limits),
        IdleVerdict::WaitingOnAPerson
    );
    assert_eq!(
        idle_verdict(waiting(T * (IDLE_HARD_CAP_FACTOR + 1), true), limits),
        IdleVerdict::Expired
    );
}

/// `reclaim_idle_slot` judges with the same rule, only a shorter patience: a quiet
/// session goes past the grace, a working one or one waiting on a person is kept.
#[test]
fn the_reclaim_patience_closes_only_what_the_cleanup_would() {
    let grace = Duration::from_secs(120);
    let limits = IdleLimits {
        patience: grace,
        timeout: T,
        person_wait_cap: PERSON_WAIT_CAP,
    };
    let quiet = IdleFacts {
        idle: grace * 2,
        ..IdleFacts::default()
    };
    assert_eq!(idle_verdict(quiet, limits), IdleVerdict::Expired);
    let working = IdleFacts {
        is_streaming: true,
        ..quiet
    };
    assert_eq!(idle_verdict(working, limits), IdleVerdict::Working);
    assert_eq!(
        idle_verdict(waiting(grace * 2, true), limits),
        IdleVerdict::WaitingOnAPerson
    );
    assert_eq!(
        idle_verdict(
            IdleFacts {
                idle: grace / 2,
                ..quiet
            },
            limits
        ),
        IdleVerdict::Recent
    );
}
