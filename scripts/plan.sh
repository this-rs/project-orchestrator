#!/bin/bash
# Plan management for orchestrator
# Usage: ./plan.sh <command> [args...]

set -e

ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://localhost:8080}"

case "$1" in
  # List plans
  list)
    curl -s "$ORCHESTRATOR_URL/api/plans" | jq -r '.[] | "\(.id)\t\(.status)\t\(.priority)\t\(.title)"' | column -t -s $'\t'
    ;;

  # Show plan details
  show)
    PLAN_ID=$2
    curl -s "$ORCHESTRATOR_URL/api/plans/$PLAN_ID" | jq .
    ;;

  # Create plan
  create)
    shift
    TITLE=""
    DESC=""
    PRIORITY=0

    while [[ $# -gt 0 ]]; do
      case $1 in
        --title) TITLE="$2"; shift 2 ;;
        --desc) DESC="$2"; shift 2 ;;
        --priority) PRIORITY="$2"; shift 2 ;;
        *) shift ;;
      esac
    done

    curl -s -X POST "$ORCHESTRATOR_URL/api/plans" \
      -H "Content-Type: application/json" \
      -d "{\"title\":\"$TITLE\",\"description\":\"$DESC\",\"priority\":$PRIORITY}" | jq .
    ;;

  # Add task
  add-task)
    shift
    PLAN_ID=""
    DESC=""
    DEPENDS=""

    while [[ $# -gt 0 ]]; do
      case $1 in
        --plan) PLAN_ID="$2"; shift 2 ;;
        --desc) DESC="$2"; shift 2 ;;
        --depends) DEPENDS="$2"; shift 2 ;;
        *) shift ;;
      esac
    done

    # Build depends_on array
    DEPS_JSON="null"
    if [[ -n "$DEPENDS" ]]; then
      DEPS_JSON=$(echo "$DEPENDS" | tr ',' '\n' | jq -R . | jq -s .)
    fi

    curl -s -X POST "$ORCHESTRATOR_URL/api/plans/$PLAN_ID/tasks" \
      -H "Content-Type: application/json" \
      -d "{\"description\":\"$DESC\",\"depends_on\":$DEPS_JSON}" | jq .
    ;;

  # Update task status
  update-task)
    TASK_ID=$2
    shift 2
    STATUS=""
    AGENT=""

    while [[ $# -gt 0 ]]; do
      case $1 in
        --status) STATUS="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        *) shift ;;
      esac
    done

    BODY="{}"
    [[ -n "$STATUS" ]] && BODY=$(echo "$BODY" | jq ".status = \"$STATUS\"")
    [[ -n "$AGENT" ]] && BODY=$(echo "$BODY" | jq ".assigned_to = \"$AGENT\"")

    curl -s -X PATCH "$ORCHESTRATOR_URL/api/tasks/$TASK_ID" \
      -H "Content-Type: application/json" \
      -d "$BODY"

    echo "Task $TASK_ID updated"
    ;;

  # Get next task
  next-task)
    PLAN_ID=$2
    curl -s "$ORCHESTRATOR_URL/api/plans/$PLAN_ID/next-task" | jq .
    ;;

  # Add decision
  add-decision)
    shift
    TASK_ID=""
    DESC=""
    RATIONALE=""

    while [[ $# -gt 0 ]]; do
      case $1 in
        --task) TASK_ID="$2"; shift 2 ;;
        --desc) DESC="$2"; shift 2 ;;
        --rationale) RATIONALE="$2"; shift 2 ;;
        *) shift ;;
      esac
    done

    curl -s -X POST "$ORCHESTRATOR_URL/api/tasks/$TASK_ID/decisions" \
      -H "Content-Type: application/json" \
      -d "{\"description\":\"$DESC\",\"rationale\":\"$RATIONALE\"}" | jq .
    ;;

  # Link files to task
  link-files)
    TASK_ID=$2
    shift 2
    FILES=$(printf '%s\n' "$@" | jq -R . | jq -s .)

    curl -s -X POST "$ORCHESTRATOR_URL/api/tasks/$TASK_ID/files" \
      -H "Content-Type: application/json" \
      -d "{\"files\":$FILES}"

    echo "Files linked to task $TASK_ID"
    ;;

  # Update plan status
  status)
    PLAN_ID=$2
    STATUS=$3

    curl -s -X PATCH "$ORCHESTRATOR_URL/api/plans/$PLAN_ID" \
      -H "Content-Type: application/json" \
      -d "{\"status\":\"$STATUS\"}"

    echo "Plan $PLAN_ID status updated to $STATUS"
    ;;

  *)
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  list                                    - List all plans"
    echo "  show <plan_id>                          - Show plan details"
    echo "  create --title <t> --desc <d>           - Create a plan"
    echo "  add-task --plan <id> --desc <d>         - Add task to plan"
    echo "  update-task <id> --status <s>           - Update task"
    echo "  next-task <plan_id>                     - Get next available task"
    echo "  add-decision --task <id> --desc <d>     - Add decision"
    echo "  link-files <task_id> <file1> <file2>    - Link files to task"
    echo "  status <plan_id> <status>               - Update plan status"
    exit 1
    ;;
esac
