
use super::is_runner_spawned;
#[test]
fn detects_runner_sessions() {
    assert!(is_runner_spawned(Some(
        r#"{"type":"runner","run_id":"x","plan_id":"y"}"#
    )));
}

#[test]
fn user_and_malformed_sessions_are_not_runners() {
    assert!(!is_runner_spawned(None));
    assert!(!is_runner_spawned(Some("")));
    assert!(!is_runner_spawned(Some(
        r#"{"type":"user","note":"runner"}"#
    )));
    assert!(!is_runner_spawned(Some("runner")));
}
