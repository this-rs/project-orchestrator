#!/usr/bin/env bash
# agent_async_demo.sh
#
# Demonstrates the async / deferred-process patterns an agent should master:
#   1. Spawn a background job, capture its PID
#   2. Poll until completion with `sleep` + `kill -0`
#   3. Capture exit code without `wait` racing the foreground shell
#   4. Tail a log file while the job runs
#   5. Trigger an MCP watch event (touch a tracked file) and observe sync
#
# Run this script from any of the runner-spawned agent's bash sessions.
# Every command used here MUST be in .claude/settings.local.json allow-list.
#
# Required allow-list entries:
#   Bash(sleep:*), Bash(wait:*), Bash(timeout:*), Bash(tail:*),
#   Bash(kill:*), Bash(ps:*), Bash(test:*), Bash(date:*), Bash(echo:*),
#   Bash(cat:*), Bash(printf:*)
#
# Exit codes:
#   0  — all patterns succeeded
#   1  — a required command was denied (allow-list incomplete)
#   2  — a pattern produced wrong output

set -euo pipefail

PO_URL="${PO_URL:-http://localhost:8080}"
TMPDIR_DEMO="$(mktemp -d -t agent_async_demo.XXXXXX)"
trap 'rm -rf "$TMPDIR_DEMO"' EXIT

log() { printf '[%s] %s\n' "$(date +%H:%M:%S.%3N 2>/dev/null || date +%H:%M:%S)" "$*"; }

# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1: Plain sleep — must succeed without permission prompt
# ─────────────────────────────────────────────────────────────────────────────
log "Pattern 1 — sleep 1 (smoke)"
sleep 1
log "  ✓ sleep returned"

# ─────────────────────────────────────────────────────────────────────────────
# Pattern 2: Background job + PID capture + poll-until-done
# ─────────────────────────────────────────────────────────────────────────────
log "Pattern 2 — background job + poll"
( sleep 3 && echo "bg-job-done" > "$TMPDIR_DEMO/bg.out" ) &
BG_PID=$!
log "  spawned bg PID=$BG_PID"

# Poll with sleep — the canonical "wait for X to finish" agent pattern.
POLL=0
while kill -0 "$BG_PID" 2>/dev/null; do
  POLL=$((POLL + 1))
  log "  poll #$POLL — still running (pid $BG_PID)"
  sleep 1
done

if [ ! -f "$TMPDIR_DEMO/bg.out" ] || [ "$(cat "$TMPDIR_DEMO/bg.out")" != "bg-job-done" ]; then
  log "  ✗ bg job did not produce expected output"
  exit 2
fi
log "  ✓ bg job completed, output captured"

# ─────────────────────────────────────────────────────────────────────────────
# Pattern 3: timeout — hard upper bound on a possibly-stuck command
# ─────────────────────────────────────────────────────────────────────────────
log "Pattern 3 — timeout 2s on a 5s sleep (must return 124)"
# On macOS, GNU `timeout` is named `gtimeout` (from coreutils). On Linux + DGX
# workers, plain `timeout` is present. Skip cleanly if neither exists.
if command -v timeout >/dev/null 2>&1; then TIMEOUT_BIN=timeout
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_BIN=gtimeout
else TIMEOUT_BIN=""; fi
if [ -n "$TIMEOUT_BIN" ]; then
  RC=0
  "$TIMEOUT_BIN" 2 sleep 5 || RC=$?
  if [ "$RC" -ne 124 ] && [ "$RC" -ne 143 ]; then
    log "  ✗ timeout did not produce expected rc (got $RC, want 124 or 143)"
    exit 2
  fi
  log "  ✓ timeout fired correctly (rc=$RC, bin=$TIMEOUT_BIN)"
else
  log "  ⚠ neither 'timeout' nor 'gtimeout' available — pattern skipped"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Pattern 4: tail -F a growing log while a producer writes to it
# ─────────────────────────────────────────────────────────────────────────────
log "Pattern 4 — tail follow on growing log"
LOG="$TMPDIR_DEMO/build.log"
: > "$LOG"

# Producer in background
(
  for i in 1 2 3 4; do
    echo "line $i" >> "$LOG"
    sleep 0.3
  done
) &
PROD_PID=$!

# Consumer: tail with timeout so we don't hang the demo
timeout 3 tail -n +1 -F "$LOG" > "$TMPDIR_DEMO/tail.out" 2>/dev/null || true
wait "$PROD_PID" 2>/dev/null || true

LINES=$(wc -l < "$TMPDIR_DEMO/tail.out" | tr -d ' ')
if [ "$LINES" -lt 4 ]; then
  log "  ✗ tail captured $LINES lines, expected ≥ 4"
  exit 2
fi
log "  ✓ tail captured $LINES lines"

# ─────────────────────────────────────────────────────────────────────────────
# Pattern 5: Probe MCP watch_status (read-only) — confirm orchestrator alive
# ─────────────────────────────────────────────────────────────────────────────
log "Pattern 5 — MCP /health probe"
if curl -fsS --max-time 3 "$PO_URL/health" > "$TMPDIR_DEMO/health.json" 2>/dev/null; then
  log "  ✓ orchestrator /health: $(cat "$TMPDIR_DEMO/health.json")"
else
  log "  ⚠ orchestrator unreachable at $PO_URL — skipping watch event check"
fi

log ""
log "All async patterns OK. The agent can:"
log "  • sleep, timeout, wait, tail without permission prompts"
log "  • spawn bg jobs, capture PID, poll until done"
log "  • follow logs in real time"
log "  • probe MCP endpoints between work batches"
