#!/usr/bin/env bash
# Seed script: Create the "rfc-lifecycle" protocol via compose (one-shot)
# Requires: PO backend running on localhost:6600
# Usage: PO_TOKEN=<jwt> ./scripts/seed-rfc-lifecycle-protocol.sh [PROJECT_ID]
# Env: PO_URL (default: http://localhost:6600), PO_TOKEN (JWT bearer token, optional for no-auth mode)
#
# FSM: 9 states modeling the RFC lifecycle
#
#   draft ──propose──► proposed ──submit_review──► under_review
#     │                   │                           │    │
#     │                   │reject                     │    │reject
#     │                   ▼                      accept    ▼
#     │               rejected◄───────────────────┤   rejected
#     │                                           ▼
#     │                                       accepted ──start_planning──► planning
#     │                                                                      │
#     │  supersede (from any non-terminal)                          start_work│
#     └──────────────────────► superseded                                    ▼
#                                                                       in_progress
#                                                                            │
#                                                                    complete│
#                                                                            ▼
#                                                                      implemented
#
# Terminal states: implemented, rejected, superseded

set -euo pipefail

BASE_URL="${PO_URL:-http://localhost:6600}"
PROJECT_ID="${1:-00333b5f-2d0a-4467-9c98-155e55d2b7e5}"

# Auth header (optional — omit PO_TOKEN for no-auth mode)
AUTH_HEADER=""
if [ -n "${PO_TOKEN:-}" ]; then
  AUTH_HEADER="Authorization: Bearer $PO_TOKEN"
fi

# Helper: curl with optional auth
po_curl() {
  if [ -n "$AUTH_HEADER" ]; then
    curl "$@" -H "$AUTH_HEADER"
  else
    curl "$@"
  fi
}

echo "=== Creating rfc-lifecycle protocol (via compose) ==="
echo "Project: $PROJECT_ID"
echo "API:     $BASE_URL"
echo ""

RESPONSE=$(po_curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/protocols/compose" \
  -H "Content-Type: application/json" \
  -d "{
    \"project_id\": \"$PROJECT_ID\",
    \"name\": \"rfc-lifecycle\",
    \"description\": \"RFC lifecycle protocol: manages the progression of RFC notes through draft, review, acceptance, planning, implementation, rejection, and supersession. Composed with Note (type: rfc) — no dedicated entity needed.\",
    \"category\": \"business\",
    \"relevance_vector\": {
      \"phase\": 0.25,
      \"structure\": 0.6,
      \"domain\": 0.5,
      \"resource\": 0.5,
      \"lifecycle\": 0.3
    },
    \"states\": [
      {
        \"name\": \"draft\",
        \"state_type\": \"start\",
        \"description\": \"Initial RFC draft. Author is writing the proposal, gathering context, exploring alternatives.\",
        \"action\": \"create_rfc_note\"
      },
      {
        \"name\": \"proposed\",
        \"state_type\": \"intermediate\",
        \"description\": \"RFC submitted for consideration. Awaiting review assignment.\",
        \"action\": \"notify_stakeholders\"
      },
      {
        \"name\": \"under_review\",
        \"state_type\": \"intermediate\",
        \"description\": \"RFC is actively being reviewed. Feedback is being collected and addressed.\",
        \"action\": \"collect_feedback\"
      },
      {
        \"name\": \"accepted\",
        \"state_type\": \"intermediate\",
        \"description\": \"RFC has been accepted. Ready for planning and implementation.\",
        \"action\": \"record_decision\"
      },
      {
        \"name\": \"planning\",
        \"state_type\": \"intermediate\",
        \"description\": \"Implementation plan is being created. Tasks, steps, and dependencies are being defined.\",
        \"action\": \"create_implementation_plan\"
      },
      {
        \"name\": \"in_progress\",
        \"state_type\": \"intermediate\",
        \"description\": \"Implementation is underway. Tasks are being executed according to the plan.\",
        \"action\": \"track_implementation\"
      },
      {
        \"name\": \"implemented\",
        \"state_type\": \"terminal\",
        \"description\": \"RFC has been fully implemented. All acceptance criteria met.\",
        \"action\": \"close_rfc\"
      },
      {
        \"name\": \"rejected\",
        \"state_type\": \"terminal\",
        \"description\": \"RFC has been rejected during review. Rationale documented.\",
        \"action\": \"document_rejection\"
      },
      {
        \"name\": \"superseded\",
        \"state_type\": \"terminal\",
        \"description\": \"RFC has been superseded by a newer RFC. Link to successor maintained.\",
        \"action\": \"link_successor\"
      }
    ],
    \"transitions\": [
      {
        \"from_state\": \"draft\",
        \"to_state\": \"proposed\",
        \"trigger\": \"propose\",
        \"guard\": \"content_not_empty AND has_title\"
      },
      {
        \"from_state\": \"proposed\",
        \"to_state\": \"under_review\",
        \"trigger\": \"submit_review\"
      },
      {
        \"from_state\": \"under_review\",
        \"to_state\": \"accepted\",
        \"trigger\": \"accept\",
        \"guard\": \"no_blocking_objections\"
      },
      {
        \"from_state\": \"under_review\",
        \"to_state\": \"rejected\",
        \"trigger\": \"reject\"
      },
      {
        \"from_state\": \"proposed\",
        \"to_state\": \"rejected\",
        \"trigger\": \"reject\"
      },
      {
        \"from_state\": \"accepted\",
        \"to_state\": \"planning\",
        \"trigger\": \"start_planning\"
      },
      {
        \"from_state\": \"planning\",
        \"to_state\": \"in_progress\",
        \"trigger\": \"start_work\",
        \"guard\": \"plan_has_tasks\"
      },
      {
        \"from_state\": \"in_progress\",
        \"to_state\": \"implemented\",
        \"trigger\": \"complete\",
        \"guard\": \"all_tasks_completed\"
      },
      {
        \"from_state\": \"draft\",
        \"to_state\": \"superseded\",
        \"trigger\": \"supersede\"
      },
      {
        \"from_state\": \"proposed\",
        \"to_state\": \"superseded\",
        \"trigger\": \"supersede\"
      },
      {
        \"from_state\": \"under_review\",
        \"to_state\": \"superseded\",
        \"trigger\": \"supersede\"
      },
      {
        \"from_state\": \"accepted\",
        \"to_state\": \"superseded\",
        \"trigger\": \"supersede\"
      },
      {
        \"from_state\": \"planning\",
        \"to_state\": \"superseded\",
        \"trigger\": \"supersede\"
      },
      {
        \"from_state\": \"in_progress\",
        \"to_state\": \"superseded\",
        \"trigger\": \"supersede\"
      },
      {
        \"from_state\": \"under_review\",
        \"to_state\": \"draft\",
        \"trigger\": \"revise\",
        \"guard\": \"has_feedback\"
      },
      {
        \"from_state\": \"in_progress\",
        \"to_state\": \"planning\",
        \"trigger\": \"replan\",
        \"guard\": \"scope_changed\"
      }
    ],
    \"triggers\": [
      {
        \"pattern_type\": \"regex\",
        \"pattern_value\": \"(?i)\\\\brfc\\\\b\",
        \"confidence_threshold\": 0.7
      },
      {
        \"pattern_type\": \"semantic\",
        \"pattern_value\": \"create a proposal or RFC for a new feature or architecture change\",
        \"confidence_threshold\": 0.6
      },
      {
        \"pattern_type\": \"regex\",
        \"pattern_value\": \"(?i)rfc.lifecycle\",
        \"confidence_threshold\": 0.9
      }
    ]
  }")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -ne 201 ] && [ "$HTTP_CODE" -ne 200 ]; then
  echo "ERROR: HTTP $HTTP_CODE"
  echo "$BODY" | jq . 2>/dev/null || echo "$BODY"
  exit 1
fi

PROTOCOL_ID=$(echo "$BODY" | jq -r '.protocol_id')
SKILL_ID=$(echo "$BODY" | jq -r '.skill_id')
STATES_CREATED=$(echo "$BODY" | jq -r '.states_created')
TRANSITIONS_CREATED=$(echo "$BODY" | jq -r '.transitions_created')

echo "✓ Protocol created: $PROTOCOL_ID"
echo "  Skill:       $SKILL_ID"
echo "  States:      $STATES_CREATED"
echo "  Transitions: $TRANSITIONS_CREATED"

# ── Verify: GET the protocol ────────────────────────────────────────
echo ""
echo "=== Verifying protocol ==="
PROTOCOL_DATA=$(po_curl -s "$BASE_URL/api/protocols/$PROTOCOL_ID")
echo "Protocol name: $(echo "$PROTOCOL_DATA" | jq -r '.name')"
echo "States: $(echo "$PROTOCOL_DATA" | jq -r '.states | length')"
echo "Transitions: $(echo "$PROTOCOL_DATA" | jq -r '.transitions | length')"
echo ""

# ── Test: simulate with RFC-matching context ────────────────────────
echo "=== Simulating activation (RFC context) ==="
SIM_RESPONSE=$(po_curl -s -X POST "$BASE_URL/api/protocols/simulate" \
  -H "Content-Type: application/json" \
  -d "{
    \"protocol_id\": \"$PROTOCOL_ID\",
    \"context\": {
      \"phase\": 0.25,
      \"structure\": 0.6,
      \"domain\": 0.5,
      \"resource\": 0.5,
      \"lifecycle\": 0.3
    }
  }")

SIM_SCORE=$(echo "$SIM_RESPONSE" | jq -r '.score')
SIM_ACTIVATE=$(echo "$SIM_RESPONSE" | jq -r '.would_activate')
echo "Score: $SIM_SCORE (would_activate: $SIM_ACTIVATE)"
echo "Dimensions: $(echo "$SIM_RESPONSE" | jq -c '.dimensions')"
echo ""

# ── Test: start a run and fire transitions ──────────────────────────
echo "=== Test run: draft → proposed → under_review → accepted ==="

# Start run
RUN=$(po_curl -s -X POST "$BASE_URL/api/protocols/$PROTOCOL_ID/runs" \
  -H "Content-Type: application/json" \
  -d "{}")
RUN_ID=$(echo "$RUN" | jq -r '.id')
CURRENT_STATE=$(echo "$RUN" | jq -r '.current_state')
echo "✓ Run started: $RUN_ID (state: $CURRENT_STATE)"

# Transition: draft → proposed
T1=$(po_curl -s -X POST "$BASE_URL/api/protocols/runs/$RUN_ID/transition" \
  -H "Content-Type: application/json" \
  -d "{\"trigger\": \"propose\"}")
CURRENT_STATE=$(echo "$T1" | jq -r '.current_state')
echo "✓ propose → $CURRENT_STATE"

# Transition: proposed → under_review
T2=$(po_curl -s -X POST "$BASE_URL/api/protocols/runs/$RUN_ID/transition" \
  -H "Content-Type: application/json" \
  -d "{\"trigger\": \"submit_review\"}")
CURRENT_STATE=$(echo "$T2" | jq -r '.current_state')
echo "✓ submit_review → $CURRENT_STATE"

# Transition: under_review → accepted
T3=$(po_curl -s -X POST "$BASE_URL/api/protocols/runs/$RUN_ID/transition" \
  -H "Content-Type: application/json" \
  -d "{\"trigger\": \"accept\"}")
CURRENT_STATE=$(echo "$T3" | jq -r '.current_state')
echo "✓ accept → $CURRENT_STATE"

# Verify run state
FINAL_RUN=$(po_curl -s "$BASE_URL/api/protocols/runs/$RUN_ID")
STATES_VISITED=$(echo "$FINAL_RUN" | jq -r '.states_visited | length')
echo ""
echo "Run status: $(echo "$FINAL_RUN" | jq -r '.status')"
echo "States visited: $STATES_VISITED"
echo "History: $(echo "$FINAL_RUN" | jq -r '[.states_visited[].state_name] | join(" → ")')"

echo ""
echo "=== Done ==="
echo "Protocol ID: $PROTOCOL_ID"
echo "Skill ID:    $SKILL_ID"
echo "States: draft → proposed → under_review → accepted → planning → in_progress → implemented"
echo "Terminal states: implemented, rejected, superseded"
echo "Reverse transitions: revise (under_review → draft), replan (in_progress → planning)"
echo ""
echo "Inspect: curl $BASE_URL/api/protocols/$PROTOCOL_ID | jq"
echo "Run:     curl $BASE_URL/api/protocols/runs/$RUN_ID | jq"
