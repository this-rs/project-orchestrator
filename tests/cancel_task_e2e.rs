//! End-to-end test for `POST /api/chat/sessions/{id}/cancel-task/{task_id}`.
//!
//! T10 of plan 754a1379 (Background tasks tracking & granular control).
//!
//! ## Why `#[ignore]` by default
//!
//! These tests require:
//! - A real Anthropic API key (`ANTHROPIC_API_KEY` env var)
//! - The Claude Code CLI installed locally
//! - Neo4j and Meilisearch reachable (or mocked at the AppState level)
//! - Network access to the Anthropic API
//!
//! Same dependency set as `cancel_tools_e2e.rs` (T6 of plan 28e9afe3).
//! Running them in CI requires secret management and adds significant
//! flakiness from real LLM behaviour. Tests are gated behind `#[ignore]`
//! so they're listed but not auto-run. Invoke with:
//!
//! ```sh
//! cargo test --test cancel_task_e2e -- --ignored
//! ```
//!
//! ## Coverage relative to existing unit tests
//!
//! Most of the cancel_task contract is already validated at unit-test
//! level in `chat::manager::tests::test_cancel_task_*` and
//! `chat::manager::tests::test_track_background_task_*`:
//!
//! - INSERT on `ToolUse Monitor` → entry materialises in the map
//!   (test_track_background_task_inserts_monitor_on_first_pass +
//!   test_track_background_task_updates_description_on_second_pass).
//! - INSERT filter for Bash run_in_background=true (vs synchronous
//!   Bash, vs other tools) — full truth table covered.
//! - cancel_task marks `pending_removal_at` and broadcasts
//!   `ActiveTasksUpdate` (test_cancel_task_marks_for_removal_and_broadcasts).
//! - Idempotent cancel on unknown id / unknown session
//!   (test_cancel_task_unknown_*).
//! - Rate cap 30 / 5 min (test_cancel_task_rate_cap_enforced — uses
//!   a tightened cap of 2 / 60s for fast assertion).
//! - Grace-period purge after pending_removal_at expires
//!   (test_tick_purge_removes_stale_pending_entries_and_broadcasts).
//! - Idle-based death detection
//!   (test_tick_purge_marks_idle_entries_as_pending_removal).
//! - Lazy crash recovery from orphan correlation_id
//!   (test_recovery_inserts_orphan_*, test_recovery_skips_*,
//!   test_recovery_touches_last_seen_at_*).
//!
//! What's left for E2E is the integration layer:
//! - REST endpoint hits the right path and returns the expected JSON.
//! - WebSocket actually broadcasts `ActiveTasksUpdate`.
//! - Two simultaneous Monitors both end up in the map, cancelling one
//!   doesn't corrupt the other (isolation property).
//! - Crash recovery works on a real subprocess — kill -KILL the PO
//!   server during a Monitor, restart, watch the recovered entry
//!   appear via the next BackgroundOutput tick.
//!
//! ## Manual run scenario
//!
//! 1. Start a PO server: `cargo run --release` (with full config + API key).
//! 2. Open the frontend on `localhost:3000`, create a chat session.
//! 3. Ask the agent: *"Monitor `tail -F /tmp/test.log` with timeout 60s.
//!    Then in parallel start a Bash run_in_background `sleep 60`."*
//! 4. Verify the toolbar pill shows `👀 1 Monitor • ⚙ 1 Bash` — these
//!    are the two tracked entries (assertion target: REST GET
//!    `/api/chat/sessions/{id}/background-tasks` returns 2 entries).
//! 5. Click Stop on the Monitor pill (frontend POSTs cancel-task).
//! 6. Verify within ~6s (5s grace + poll latency):
//!    - The Monitor pill disappears from the toolbar.
//!    - The Bash pill REMAINS visible (isolation: cancel_task didn't
//!      affect the other tracked entry).
//!    - WS observed an `ActiveTasksUpdate` carrying only the Bash entry.
//! 7. `kill -9 $(pgrep -f 'po-server')` to crash PO mid-Monitor.
//! 8. Restart the server.
//! 9. Append a new line to `/tmp/test.log`. The next tick fires a
//!    BackgroundOutput, which trips the lazy recovery path: the
//!    Monitor pill reappears via `ActiveTasksUpdate`. The recovered
//!    entry has `description: "(recovered after restart)"`.
//!
//! ## Implementation note
//!
//! Implementing these as `#[tokio::test]` requires a reusable
//! test-server fixture (PO doesn't currently bundle one — same gap
//! as cancel_tools_e2e.rs noted by decision `8336f2f3`). When that
//! fixture lands, the stubs below become real tests with assertion
//! coverage on the manual scenario.

#![cfg(unix)]

#[cfg(test)]
mod tests {
    /// Cancel one Monitor among two — assert isolation. The other
    /// Monitor stays tracked, the cancelled one disappears after the
    /// grace period (5s).
    #[test]
    #[ignore = "requires real Claude CLI + ANTHROPIC_API_KEY + test-server fixture (TODO)"]
    fn test_cancel_task_isolates_to_target_id() {
        eprintln!(
            "[T10 stub] Manual scenario:\n\
             1. Start two Monitors (different files). Verify GET background-tasks → 2 entries.\n\
             2. POST cancel-task/{{id_of_monitor_A}}.\n\
             3. Assert: WS receives ActiveTasksUpdate with both still present (Monitor A\n\
                marked pending_removal_at).\n\
             4. Wait grace (5s + poll). Assert: WS receives ActiveTasksUpdate with only B.\n\
             5. GET background-tasks → 1 entry, id == B."
        );
    }

    /// Lazy crash recovery — kill PO mid-Monitor, restart, verify the
    /// next tick brings the entry back. The recovered description is
    /// the placeholder, not the original.
    #[test]
    #[ignore = "requires real Claude CLI + ANTHROPIC_API_KEY + test-server fixture (TODO)"]
    fn test_lazy_recovery_after_server_restart() {
        let scenario = "[T10 stub] Manual scenario:\n\
             1. Start a Monitor on /tmp/test.log. Verify entry in map (id = X).\n\
             2. kill -9 $(pgrep -f po-server).\n\
             3. Restart server. GET background-tasks → empty (map reset).\n\
             4. Append a line to /tmp/test.log → Monitor emits BackgroundOutput.\n\
             5. WS receives ActiveTasksUpdate carrying a single entry with\n\
                id == X and description == '(recovered after restart)'.";
        eprintln!("{}", scenario);
    }

    /// Rate cap path — spam cancel-task POST and observe `capped: true`
    /// in at least one response body.
    #[test]
    #[ignore = "requires real Claude CLI + ANTHROPIC_API_KEY + test-server fixture (TODO)"]
    fn test_cancel_task_rate_cap_returns_capped_true() {
        eprintln!(
            "[T10 stub] Spam 31 POST cancel-task/{{any_id}} within 5 min, \
             expect at least one response body with capped:true."
        );
    }

    /// Bash with run_in_background=true is tracked (and cancellable),
    /// synchronous Bash is NOT.
    #[test]
    #[ignore = "requires real Claude CLI + ANTHROPIC_API_KEY + test-server fixture (TODO)"]
    fn test_bash_run_in_background_tracked_synchronous_not() {
        eprintln!(
            "[T10 stub] Trigger one Bash run_in_background:true and one synchronous Bash. \
             GET background-tasks must return 1 entry (the bg one) only."
        );
    }

    // ====================================================================
    // T7 of plan fc35b25e — V2 PID-targeted kill — 2-Monitor isolation
    // ====================================================================

    /// V2 isolation flagship — start two Monitors on different log files,
    /// cancel one, and verify *only* that one's subprocess actually dies
    /// (the other is untouched). This is the test that proves V2 over V1:
    /// V1 would have left both `tail -F` running regardless of cancel.
    ///
    /// Asserts:
    /// 1. After ~1.5 s post-ToolUse, both entries have `pid: Some(...)`
    ///    populated (T2 async claim landed).
    /// 2. `cancel_task(id_A)` returns `killed_pids` containing the
    ///    Monitor A subtree's PIDs (T4 wires kill_subtree).
    /// 3. `pgrep -f "tail.*test_a.log"` returns empty after cancel
    ///    (subprocess actually dead).
    /// 4. `pgrep -f "tail.*test_b.log"` still matches (Monitor B
    ///    untouched — true isolation).
    #[test]
    #[ignore = "requires real Claude CLI + ANTHROPIC_API_KEY + test-server fixture (TODO)"]
    fn test_v2_isolation_2_monitors_cancel_one() {
        let scenario = "[T7 plan fc35b25e — V2 isolation manual scenario]\n\
             \n\
             Setup:\n\
             - Two empty log files: /tmp/po_test_a.log, /tmp/po_test_b.log.\n\
             - PO server running with the V2 cancel_task code (commit efb47a8).\n\
             \n\
             Steps:\n\
             1. POST /api/chat/sessions {...} → session_id.\n\
             2. WS connect.\n\
             3. WS send: \"Use Monitor on /tmp/po_test_a.log with timeout 5min.\"\n\
                Wait for ToolUse event id_A → wait 1.5s for the async PID claim (T2).\n\
             4. WS send: \"Now use Monitor on /tmp/po_test_b.log with timeout 5min.\"\n\
                Wait for ToolUse event id_B → wait 1.5s.\n\
             5. GET /api/chat/sessions/{session_id}/background-tasks.\n\
                Assert: 2 entries, both with pid=Some(<u32>) (different PIDs).\n\
                Capture pid_a, pid_b.\n\
             6. POST /api/chat/sessions/{session_id}/cancel-task/{id_A}.\n\
                Assert: response.killed_pids contains pid_a (and possibly\n\
                its descendants — the inner sleep / shell of `tail -F`).\n\
                Assert: response.capped == false.\n\
             7. Wait ~200 ms for SIGINT delivery + reaping.\n\
                Run `pgrep -f \"tail.*po_test_a.log\"` from the test host.\n\
                Assert: empty output (Monitor A's tail is dead).\n\
             8. Run `pgrep -f \"tail.*po_test_b.log\"`.\n\
                Assert: returns pid_b (Monitor B still alive — isolation\n\
                property holds).\n\
             9. WS observed an ActiveTasksUpdate where Monitor A is\n\
                marked pending_removal_at and Monitor B is unchanged.\n\
             10. After 5 s grace, WS observed an ActiveTasksUpdate where\n\
                 Monitor A entry is gone (purged by T12) and Monitor B\n\
                 still present.\n\
             \n\
             Cleanup:\n\
             - POST cancel-task/{id_B} to stop Monitor B.\n\
             - rm /tmp/po_test_a.log /tmp/po_test_b.log.\n\
             \n\
             Why V2 wins over V1 here:\n\
             - V1: response.killed_pids would have been [] (no PID stored).\n\
             - V1: Monitor A's tail subprocess would still match pgrep.\n\
             - V1: only the toolbar pill would change (cosmetic cancel).\n\
             V2 makes the kill actually happen, scoped to the one task.";
        eprintln!("{scenario}");
    }
}
