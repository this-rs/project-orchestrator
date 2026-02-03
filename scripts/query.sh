#!/bin/bash
# Query interface for orchestrator
# Usage: ./query.sh <command> [args...]

set -e

ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://localhost:8080}"

case "$1" in
  # Neo4j/Graph queries via API
  context)
    PLAN_ID=$2
    TASK_ID=$3
    curl -s "$ORCHESTRATOR_URL/api/plans/$PLAN_ID/tasks/$TASK_ID/context" | jq .
    ;;

  prompt)
    PLAN_ID=$2
    TASK_ID=$3
    curl -s "$ORCHESTRATOR_URL/api/plans/$PLAN_ID/tasks/$TASK_ID/prompt" | jq -r '.prompt'
    ;;

  # Search
  search)
    INDEX=$2
    QUERY=$3
    LIMIT=${4:-10}

    case "$INDEX" in
      code)
        curl -s "$ORCHESTRATOR_URL/api/search/code?q=$QUERY&limit=$LIMIT" | jq .
        ;;
      decisions)
        curl -s "$ORCHESTRATOR_URL/api/decisions/search?q=$QUERY&limit=$LIMIT" | jq .
        ;;
      *)
        echo "Unknown index: $INDEX"
        exit 1
        ;;
    esac
    ;;

  # Plan operations
  plans)
    curl -s "$ORCHESTRATOR_URL/api/plans" | jq .
    ;;

  plan)
    PLAN_ID=$2
    curl -s "$ORCHESTRATOR_URL/api/plans/$PLAN_ID" | jq .
    ;;

  next-task)
    PLAN_ID=$2
    curl -s "$ORCHESTRATOR_URL/api/plans/$PLAN_ID/next-task" | jq .
    ;;

  # Task operations
  task)
    TASK_ID=$2
    curl -s "$ORCHESTRATOR_URL/api/tasks/$TASK_ID" | jq .
    ;;

  # Health check
  health)
    curl -s "$ORCHESTRATOR_URL/health" | jq .
    ;;

  *)
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  context <plan_id> <task_id>  - Get task context"
    echo "  prompt <plan_id> <task_id>   - Get task prompt"
    echo "  search <index> <query>       - Search code or decisions"
    echo "  plans                        - List all plans"
    echo "  plan <plan_id>               - Get plan details"
    echo "  next-task <plan_id>          - Get next available task"
    echo "  task <task_id>               - Get task details"
    echo "  health                       - Health check"
    exit 1
    ;;
esac
