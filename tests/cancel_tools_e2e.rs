//! End-to-end test for `POST /api/chat/sessions/{id}/cancel-tools`.
//!
//! T6 of plan 28e9afe3 (User-initiated tool subprocess cancellation).
//!
//! ## Why `#[ignore]` by default
//!
//! These tests require:
//! - A real Anthropic API key (`ANTHROPIC_API_KEY` env var)
//! - The Claude Code CLI installed locally
//! - Neo4j and Meilisearch reachable (or mocked at the AppState level)
//! - Network access to the Anthropic API
//!
//! Running them in CI requires secret management and adds significant
//! flakiness from real LLM behaviour. Tests are gated behind
//! `#[ignore]` so they're listed but not auto-run. Invoke with:
//!
//! ```sh
//! cargo test --test cancel_tools_e2e -- --ignored
//! ```
//!
//! ## Coverage relative to the empirical gate
//!
//! The fundamental "cancel preserves the turn" invariant was already
//! validated **by static source analysis** of the Claude Code CLI in
//! T1 (decision d2bf0e7b on plan 28e9afe3 — see
//! `bridgeMessaging.ts:362` → `QueryEngine.ts:1158` and
//! `BashTool.tsx:881`). These E2E tests are SLA / regression checks:
//! - Latency cancel→PID gone < 500 ms
//! - The turn does keep going (a follow-up message succeeds)
//! - The cancel ChatEvent is broadcast on the session feed
//!
//! ## Manual run scenario
//!
//! 1. Start a PO server with full config: `cargo run --release`
//! 2. Export your API key: `export ANTHROPIC_API_KEY=sk-ant-...`
//! 3. `cargo test --test cancel_tools_e2e test_cancel_kills_sleep_keeps_turn -- --ignored --nocapture`
//! 4. Observe stderr for the latency report.

#![cfg(unix)]

#[cfg(test)]
mod tests {
    /// **Manual scenario** — to flesh out when adding a real-CLI test
    /// harness to PO. The structural plan is:
    ///
    /// ```text
    /// 1. POST /api/chat/sessions { ... }                        → session_id
    /// 2. WS /ws/chat/{session_id} (subscribe to events)
    /// 3. WS send: { "type": "user_message", "content":
    ///      "Run `bash -c 'sleep 30 && echo done'`. \
    ///       Use a foreground Bash tool. Don't use run_in_background." }
    /// 4. Wait for ChatEvent::ToolUse { tool: "Bash", input: ... }
    ///    on the WS — confirms the agent has actually invoked Bash.
    /// 5. pgrep -f 'sleep 30' → assert at least one PID exists
    /// 6. Note `t0 = Instant::now()`
    /// 7. POST /api/chat/sessions/{session_id}/cancel-tools
    ///    Expect 200 with body { cli_pid: Some(...), killed_pids: [...], capped: false }
    /// 8. Poll pgrep -f 'sleep 30' until empty → record `t1`
    ///    Latency = t1 - t0; assert < 500 ms
    /// 9. Receive on WS:
    ///    - ChatEvent::ToolsCancelled { killed_count >= 1, requested_by: "user" }
    ///    - ChatEvent::ToolResult { is_error: true, ... } for the Bash tool
    /// 10. WS send: { "type": "user_message", "content": "ok continue" }
    /// 11. Receive ChatEvent::AssistantText (any) → confirms turn continues
    /// 12. Cleanup: WS send Interrupt + DELETE /api/chat/sessions/{id}
    /// ```
    ///
    /// Implementation depends on a reusable test-server fixture that
    /// PO doesn't currently have. Tracked as a follow-up — when it
    /// lands, this stub becomes a real `#[tokio::test] #[ignore]`.
    #[test]
    #[ignore = "requires real Claude CLI + ANTHROPIC_API_KEY + test-server fixture (TODO)"]
    fn test_cancel_kills_sleep_keeps_turn() {
        eprintln!(
            "[T6 stub] Manual scenario documented in module doc-comment. \
             Real impl pending PO test-server fixture."
        );
    }

    /// Companion test for the rate cap path: 11 rapid POSTs should
    /// see at least one capped:true response. Same fixture dependency
    /// as above.
    #[test]
    #[ignore = "requires real Claude CLI + ANTHROPIC_API_KEY + test-server fixture (TODO)"]
    fn test_cancel_rate_cap_returns_capped_true() {
        eprintln!(
            "[T6 stub] Spam 11 POST cancel-tools in <60s, expect at least one \
             response body with capped:true and killed_pids:[]."
        );
    }
}
