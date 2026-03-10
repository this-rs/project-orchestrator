#!/usr/bin/env bash
# Seed script: Create RFC foundational Knowledge Fabric notes
# Requires: PO backend running on localhost:6600
# Usage: PO_TOKEN=<jwt> ./scripts/seed-rfc-knowledge.sh [PROJECT_ID]
# Env: PO_URL (default: http://localhost:6600), PO_TOKEN (JWT bearer token, optional for no-auth mode)
#
# Creates 3 foundational notes for RFC management:
#   1. RFC Conventions (guideline) — when/how to create RFCs, required sections
#   2. RFC Lifecycle Pitfalls (gotcha) — common mistakes to avoid
#   3. RFC Auto-Detection Pattern (pattern) — how observation_detector triggers RFC creation
#
# Each note is linked to relevant source files:
#   - src/chat/observation_detector.rs (auto-detection logic)
#   - src/protocol/models.rs (protocol FSM models)
#   - src/notes/models.rs (note types including Rfc)

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

# Helper: check HTTP response code and exit on error
check_response() {
  local label="$1"
  local response="$2"
  local http_code
  local body

  http_code=$(echo "$response" | tail -1)
  body=$(echo "$response" | sed '$d')

  if [ "$http_code" -lt 200 ] || [ "$http_code" -ge 300 ]; then
    echo "ERROR ($label): HTTP $http_code"
    echo "$body" | jq . 2>/dev/null || echo "$body"
    exit 1
  fi

  echo "$body"
}

# Helper: link a note to a file entity
link_note_to_file() {
  local note_id="$1"
  local file_path="$2"

  local link_response
  link_response=$(po_curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/notes/$note_id/links" \
    -H "Content-Type: application/json" \
    -d "{
      \"entity_type\": \"File\",
      \"entity_id\": \"$file_path\"
    }")

  local link_code
  link_code=$(echo "$link_response" | tail -1)
  if [ "$link_code" -ge 200 ] && [ "$link_code" -lt 300 ]; then
    echo "    Linked to $file_path"
  else
    echo "    Warning: could not link to $file_path (HTTP $link_code)"
  fi
}

echo "=== Seeding RFC Knowledge Fabric Notes ==="
echo "Project: $PROJECT_ID"
echo "API:     $BASE_URL"
echo ""

# ── Note 1: RFC Conventions (guideline) ────────────────────────────
echo "--- Note 1: RFC Conventions (guideline) ---"
RESPONSE=$(po_curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/notes" \
  -H "Content-Type: application/json" \
  -d "{
    \"project_id\": \"$PROJECT_ID\",
    \"note_type\": \"Guideline\",
    \"content\": \"## RFC Conventions\\n\\n### When to create an RFC\\n- Cross-cutting architectural changes\\n- New system-wide patterns\\n- Breaking API changes\\n- Technology choices\\n\\n### Required Sections\\n1. **Problem**: What problem does this solve?\\n2. **Proposed Solution**: Detailed technical approach\\n3. **Alternatives**: Other approaches considered with pros/cons\\n4. **Impact**: Files, systems, and teams affected\\n\\n### Lifecycle\\ndraft > proposed > under_review > accepted > planning > in_progress > implemented\\nTerminal states: implemented, rejected, superseded\\n\\n### Rules\\n- Never skip under_review (proposed > accepted is forbidden — needs review)\\n- Always link RFC to affected files via LINKED_TO\\n- Auto-detected RFCs start as draft\\n- Advance via protocol transitions: propose, submit_review, accept, start_planning, start_work, complete\\n- Before creating: search_semantic with threshold 0.85 to avoid duplicates\",
    \"importance\": \"High\",
    \"tags\": [\"rfc\", \"convention\", \"guideline\"]
  }")

BODY=$(check_response "note 1" "$RESPONSE")
NOTE1_ID=$(echo "$BODY" | jq -r '.id')
echo "  Created: $NOTE1_ID"

# Link to relevant files
link_note_to_file "$NOTE1_ID" "src/notes/models.rs"
link_note_to_file "$NOTE1_ID" "src/protocol/models.rs"
echo ""

# ── Note 2: RFC Lifecycle Pitfalls (gotcha) ────────────────────────
echo "--- Note 2: RFC Lifecycle Pitfalls (gotcha) ---"
RESPONSE=$(po_curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/notes" \
  -H "Content-Type: application/json" \
  -d "{
    \"project_id\": \"$PROJECT_ID\",
    \"note_type\": \"Gotcha\",
    \"content\": \"## RFC Lifecycle Pitfalls\\n\\n1. **Never skip under_review**: An RFC must go through proposed > under_review where stakeholders can review. Auto-accepting (proposed > accepted) is an anti-pattern — use submit_review first.\\n2. **Always link to plans**: When an RFC is accepted, create implementation plans and link them. Orphan accepted RFCs never get implemented.\\n3. **Deduplication**: Before creating a new RFC, search_semantic with threshold 0.85 to avoid duplicates. Two RFCs addressing the same problem cause confusion.\\n4. **Auto-detection threshold**: The RfcAccumulator requires 2+ consecutive architectural messages. Single mentions are not enough to trigger auto-creation.\\n5. **RFC is not a Decision**: An RFC is a proposal. Decisions are the outcome. An accepted RFC should generate a Decision via decision(action: \\\"add\\\"). Do not conflate the two.\\n6. **Supersession links**: When superseding an RFC, always link the new RFC to the old one. Orphan superseded RFCs lose their context trail.\\n7. **Guard conditions**: Transitions with guards (content_not_empty, no_blocking_objections, plan_has_tasks, all_tasks_completed) are enforced — do not bypass them.\",
    \"importance\": \"High\",
    \"tags\": [\"rfc\", \"gotcha\", \"lifecycle\"]
  }")

BODY=$(check_response "note 2" "$RESPONSE")
NOTE2_ID=$(echo "$BODY" | jq -r '.id')
echo "  Created: $NOTE2_ID"

# Link to relevant files
link_note_to_file "$NOTE2_ID" "src/protocol/models.rs"
link_note_to_file "$NOTE2_ID" "src/chat/observation_detector.rs"
echo ""

# ── Note 3: RFC Auto-Detection Pattern (pattern) ──────────────────
echo "--- Note 3: RFC Auto-Detection Pattern (pattern) ---"
RESPONSE=$(po_curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/notes" \
  -H "Content-Type: application/json" \
  -d "{
    \"project_id\": \"$PROJECT_ID\",
    \"note_type\": \"Pattern\",
    \"content\": \"## RFC Auto-Detection Pattern\\n\\nThe observation_detector uses ObservationCategory::Rfc with weighted patterns:\\n\\n### Direct patterns (confidence >= 0.7)\\n- \\\"RFC:\\\" or \\\"RFC —\\\" prefix\\n- \\\"I propose\\\" / \\\"we should consider\\\" / \\\"architecture question\\\"\\n- \\\"proposition architecturale\\\" / \\\"proposition technique\\\"\\n\\n### Indirect patterns (confidence >= 0.5)\\n- \\\"on devrait\\\" / \\\"il faudrait\\\" / \\\"il manque un\\\"\\n- \\\"we should\\\" / \\\"we need to rethink\\\" / \\\"how would you structure\\\"\\n\\n### Accumulation\\nRfcAccumulator tracks consecutive RFC observations. At threshold (2+), triggers auto-creation of RFC draft note with Problem/Proposed Solution sections pre-filled from conversation context.\\n\\n### Deduplication\\nBefore creation: search_semantic on existing RFC notes. Skip if similarity > 0.85.\\n\\n### Integration with rfc-lifecycle protocol\\nWhen an RFC note is created (manually or auto-detected), a protocol run is started in draft state. The skill context_template provides available transitions to the agent.\",
    \"importance\": \"Medium\",
    \"tags\": [\"rfc\", \"pattern\", \"auto-detection\", \"observation-detector\"]
  }")

BODY=$(check_response "note 3" "$RESPONSE")
NOTE3_ID=$(echo "$BODY" | jq -r '.id')
echo "  Created: $NOTE3_ID"

# Link to relevant files
link_note_to_file "$NOTE3_ID" "src/chat/observation_detector.rs"
link_note_to_file "$NOTE3_ID" "src/notes/models.rs"
echo ""

# ── Summary ────────────────────────────────────────────────────────
echo "=== Done ==="
echo "Notes created:"
echo "  1. RFC Conventions (guideline): $NOTE1_ID"
echo "  2. RFC Lifecycle Pitfalls (gotcha): $NOTE2_ID"
echo "  3. RFC Auto-Detection Pattern (pattern): $NOTE3_ID"
echo ""
echo "Inspect:"
echo "  curl $BASE_URL/api/notes/$NOTE1_ID | jq"
echo "  curl $BASE_URL/api/notes/$NOTE2_ID | jq"
echo "  curl $BASE_URL/api/notes/$NOTE3_ID | jq"
echo ""
echo "List all RFC notes:"
echo "  curl '$BASE_URL/api/notes?project_id=$PROJECT_ID&tags=rfc' | jq"
