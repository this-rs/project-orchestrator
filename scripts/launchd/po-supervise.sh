#!/bin/bash
# Supervisor entry point for the Project Orchestrator server.
#
# Its only job: keep the log launchd redirects stdout/stderr to from growing
# without bound, then become the server (`exec`, so no extra PID sits between
# launchd and the orchestrator and signals reach it directly).
#
# Why this exists: the plist redirects stdout/stderr to /tmp/orchestrator.log,
# and launchd never rotates it. A crash loop therefore inflates it with one full
# boot's worth of output per attempt — measured 2026-11-25, 1.0 GB of the same
# four lines. The file is trimmed with `cat >`, not `mv`: launchd holds the file
# open in append mode, so renaming it would make every later write land in an
# unlinked inode and the new file would stay empty.
#
# Usage (from the plist): po-supervise.sh <orchestrator-binary> [args...]
set -u

LOG="${PO_SUPERVISOR_LOG:-/tmp/orchestrator.log}"
KEEP_BYTES="${PO_SUPERVISOR_LOG_KEEP_BYTES:-5000000}"

if [ -f "$LOG" ]; then
  size=$(stat -f%z "$LOG" 2>/dev/null || stat -c%s "$LOG" 2>/dev/null || echo 0)
  if [ "${size:-0}" -gt "$KEEP_BYTES" ]; then
    tmp=$(mktemp "${TMPDIR:-/tmp}/po-supervisor-log.XXXXXX") || tmp=""
    if [ -n "$tmp" ]; then
      tail -c "$KEEP_BYTES" "$LOG" > "$tmp" 2>/dev/null && cat "$tmp" > "$LOG"
      rm -f "$tmp"
      printf '[%s] po-supervise: %s trimmed from %s to <=%s bytes\n' \
        "$(date -u +%FT%TZ)" "$LOG" "$size" "$KEEP_BYTES" >> "$LOG"
    fi
  fi
fi

if [ "$#" -eq 0 ]; then
  echo "po-supervise: no command given (expected: <orchestrator-binary> [args...])" >&2
  exit 64
fi

exec "$@"
