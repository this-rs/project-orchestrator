#!/bin/bash
# Generate agent context
# Usage: ./context.sh --task <task_id> --plan <plan_id> [--depth <n>]

set -e

ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://localhost:8080}"

TASK_ID=""
PLAN_ID=""
DEPTH=2
OUTPUT="prompt"

while [[ $# -gt 0 ]]; do
  case $1 in
    --task) TASK_ID="$2"; shift 2 ;;
    --plan) PLAN_ID="$2"; shift 2 ;;
    --depth) DEPTH="$2"; shift 2 ;;
    --json) OUTPUT="json"; shift ;;
    --prompt) OUTPUT="prompt"; shift ;;
    *) shift ;;
  esac
done

if [[ -z "$TASK_ID" || -z "$PLAN_ID" ]]; then
  echo "Usage: $0 --task <task_id> --plan <plan_id> [--depth <n>] [--json|--prompt]"
  exit 1
fi

case "$OUTPUT" in
  json)
    curl -s "$ORCHESTRATOR_URL/api/plans/$PLAN_ID/tasks/$TASK_ID/context" | jq .
    ;;
  prompt)
    curl -s "$ORCHESTRATOR_URL/api/plans/$PLAN_ID/tasks/$TASK_ID/prompt" | jq -r '.prompt'
    ;;
esac
