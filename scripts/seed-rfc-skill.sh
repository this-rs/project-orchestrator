#!/usr/bin/env bash
# Seed script: Create the "rfc-management" Neural Skill
# Requires: PO backend running on localhost:6600
# Usage: PO_TOKEN=<jwt> ./scripts/seed-rfc-skill.sh [PROJECT_ID]
# Env: PO_URL (default: http://localhost:6600), PO_TOKEN (JWT bearer token, optional for no-auth mode)
#
# This skill manages the RFC lifecycle:
#   - Auto-detects architectural discussions via trigger patterns
#   - Creates RFC notes with required sections (Problem, Proposed Solution, Alternatives, Impact)
#   - Drives RFC state via the rfc-lifecycle protocol (draft -> proposed -> accepted -> implemented)
#
# Trigger patterns:
#   - Regex: RFC keyword, architectural discussion markers (FR/EN)
#   - Semantic: proposal/architecture change intent
#   - McpAction: note:create (for RFC-typed notes)

set -euo pipefail

BASE_URL="${PO_URL:-http://localhost:6600}"
PROJECT_ID="${1:-00333b5f-2d0a-4467-9c98-155e55d2b7e5}"

# Auth header (optional — omit PO_TOKEN for no-auth mode)
AUTH_HEADER=""
if [ -n "${PO_TOKEN:-}" ]; then
  AUTH_HEADER="Authorization: Bearer $PO_TOKEN"
fi

# Helper: curl with optional auth, returns body + HTTP code
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

echo "=== Creating rfc-management Neural Skill ==="
echo "Project: $PROJECT_ID"
echo "API:     $BASE_URL"
echo ""

# ── 1. Create the skill ────────────────────────────────────────────
RESPONSE=$(po_curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/skills" \
  -H "Content-Type: application/json" \
  -d "{
    \"project_id\": \"$PROJECT_ID\",
    \"name\": \"rfc-management\",
    \"description\": \"Manages RFC lifecycle: auto-detection of architectural discussions, RFC note creation, protocol-backed state machine for RFC lifecycle (draft > proposed > accepted > implemented | rejected). Integrates with the rfc-lifecycle protocol and Knowledge Fabric notes.\",
    \"tags\": [\"rfc\", \"architecture\", \"protocol\", \"knowledge\"],
    \"trigger_patterns\": [
      {
        \"pattern_type\": \"Regex\",
        \"pattern_value\": \"(?i)\\\\bRFC\\\\b|\\\\bproposition architecturale\\\\b|\\\\bdesign decision\\\\b\",
        \"confidence_threshold\": 0.7
      },
      {
        \"pattern_type\": \"Regex\",
        \"pattern_value\": \"(?i)il faudrait|we should consider|architecture question|how would you structure\",
        \"confidence_threshold\": 0.5
      },
      {
        \"pattern_type\": \"Semantic\",
        \"pattern_value\": \"create a proposal or RFC for a new architectural pattern, design decision, or cross-cutting change\",
        \"confidence_threshold\": 0.6
      },
      {
        \"pattern_type\": \"McpAction\",
        \"pattern_value\": \"note:create\",
        \"confidence_threshold\": 0.5
      }
    ],
    \"context_template\": \"## RFC Management Context\\n\\nActive RFCs: {{list_rfcs}}\\n\\nRFC Lifecycle: draft > proposed > under_review > accepted > planning > in_progress > implemented\\nTerminal states: implemented, rejected, superseded\\n\\nActions: list_rfcs, advance_rfc, get_rfc_status\\n\\nConventions:\\n- Required sections: Problem, Proposed Solution, Alternatives, Impact\\n- Link to affected files via LINKED_TO\\n- Advance via protocol transitions (propose, submit_review, accept, start_planning, start_work, complete)\\n- Never skip under_review (needs stakeholder review)\\n- Search semantic before creating to avoid duplicates (threshold 0.85)\"
  }")

BODY=$(check_response "create skill" "$RESPONSE")
SKILL_ID=$(echo "$BODY" | jq -r '.id')
echo "Skill created: $SKILL_ID"
echo "  Name: $(echo "$BODY" | jq -r '.name')"
echo "  Triggers: $(echo "$BODY" | jq -r '.trigger_patterns | length')"
echo ""

# ── 2. Link to rfc-lifecycle protocol (if exists) ──────────────────
echo "=== Linking to rfc-lifecycle protocol ==="
PROTO_RESPONSE=$(po_curl -s "$BASE_URL/api/protocols?project_id=$PROJECT_ID&search=rfc-lifecycle")
RFC_PROTO_ID=$(echo "$PROTO_RESPONSE" | jq -r '.data[0].id // empty')

if [ -n "$RFC_PROTO_ID" ]; then
  echo "Found rfc-lifecycle protocol: $RFC_PROTO_ID"
  # Add skill as member of the protocol (protocol is also a skill — link via members)
  MEMBER_RESPONSE=$(po_curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/skills/$SKILL_ID/members" \
    -H "Content-Type: application/json" \
    -d "{
      \"member_id\": \"$RFC_PROTO_ID\",
      \"role\": \"protocol\"
    }")
  MEMBER_BODY=$(check_response "link protocol" "$MEMBER_RESPONSE" 2>/dev/null) && \
    echo "Linked skill to protocol $RFC_PROTO_ID" || \
    echo "Note: Could not link to protocol (may need manual linking via MCP)"
else
  echo "No rfc-lifecycle protocol found — run seed-rfc-lifecycle-protocol.sh first"
  echo "Then link manually: POST /api/skills/$SKILL_ID/members with protocol ID"
fi
echo ""

# ── 3. Verify ──────────────────────────────────────────────────────
echo "=== Verifying skill ==="
SKILL_DATA=$(po_curl -s "$BASE_URL/api/skills/$SKILL_ID")
echo "Skill name:     $(echo "$SKILL_DATA" | jq -r '.name')"
echo "Status:         $(echo "$SKILL_DATA" | jq -r '.status')"
echo "Triggers:       $(echo "$SKILL_DATA" | jq -r '.trigger_patterns | length')"
echo "Tags:           $(echo "$SKILL_DATA" | jq -c '.tags')"
echo ""

echo "=== Done ==="
echo "Skill ID: $SKILL_ID"
echo "Inspect: curl $BASE_URL/api/skills/$SKILL_ID | jq"
