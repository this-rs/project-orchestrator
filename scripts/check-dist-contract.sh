#!/bin/sh
# Fails when the embedded frontend (dist/) lacks a WebSocket frame the backend relies on.
# Why: backend #525 (server-side message queue) shipped in v0.0.16 while dist/ was still a
# frontend build from before the matching frontend change, so the feature silently did not work.
# Each token below is a frame/event name the backend handles or emits; the served bundle must
# mention it. Add a token here when a backend feature needs a new frontend counterpart.
set -eu
DIST="${1:-dist}"
TOKENS="queue_op pending_queue"
status=0
for t in $TOKENS; do
  if ! grep -rqF -- "$t" "$DIST/assets"; then
    echo "dist/ never mentions '$t': the embedded frontend predates the backend feature that uses it." >&2
    echo "  Rebuild the frontend from main and redeploy dist/ (chore(dist): deploy frontend main <sha>)." >&2
    status=1
  fi
done
[ "$status" -eq 0 ] && echo "dist/ mentions every required frame: $TOKENS"
exit $status
