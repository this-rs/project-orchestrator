use super::{background_task_is_idle_dead, process_alive, session_is_expired};
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
        tracked.max(cli_reported),
        false // waiting_on_input
    ));
    // Without the CLI's report, the emptied tracking alone closes it.
    assert!(session_is_expired(T * 2, T, false, tracked, false));
}

#[test]
fn active_session_is_kept() {
    assert!(!session_is_expired(
        Duration::from_secs(60),
        T,
        false,
        0,
        false
    ));
}

#[test]
fn idle_session_without_work_expires() {
    assert!(session_is_expired(
        T + Duration::from_secs(1),
        T,
        false,
        0,
        false
    ));
}

#[test]
fn idle_session_with_background_work_is_kept() {
    // The failure mode: agent's turn over, sub-agents still running.
    assert!(!session_is_expired(T * 2, T, false, 3, false));
}

#[test]
fn streaming_session_is_kept() {
    assert!(!session_is_expired(T * 2, T, true, 0, false));
}

#[test]
fn stuck_background_work_is_bounded_by_the_hard_cap() {
    assert!(session_is_expired(T * 9, T, true, 1, false));
}

#[test]
fn idle_session_waiting_on_input_is_not_expired_even_after_hard_cap() {
    // D2 FIX: session waiting for user input should not expire after hard cap
    assert!(!session_is_expired(T * 9, T, false, 0, true));
}
