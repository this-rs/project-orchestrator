#!/usr/bin/env bash
# M2 Bridge the-source — Integration test
# Tests skill import with episodes on a fresh PO instance (no auth)
#
# Since instance 1 has auth enabled, we generate fake episodes locally
# and test the full import flow on instance 2 (fresh graph, no auth).
#
# Prerequisites:
# - Instance 2 running on localhost:6601 (M2 test, fresh graph, no auth)
#
# Usage: bash tests/m2_bridge_test.sh

set -euo pipefail

INSTANCE2="http://localhost:6601"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== M2 Bridge the-source — Integration Test ==="
echo "Instance 2: $INSTANCE2"
echo "Temp dir: $TMPDIR"
echo ""

# ─── Step 0: Health check ────────────────────────────────────────────────
echo "── Step 0: Health check"
curl -sf "$INSTANCE2/api/projects" > /dev/null && echo "  Instance 2: OK" || { echo "  Instance 2: FAILED (is it running on port 6601?)"; exit 1; }
echo ""

# ─── Step 1: Create a project on instance 2 ──────────────────────────────
echo "── Step 1: Create project on instance 2"
PROJECT2=$(curl -sf "$INSTANCE2/api/projects" \
  -H "Content-Type: application/json" \
  -d '{"name": "m2-test-receiver", "description": "M2 test: receiving instance for episodic transfer", "root_path": "/tmp/m2-test"}')
PROJECT2_ID=$(echo "$PROJECT2" | jq -r '.id')
echo "  Project created: $PROJECT2_ID"
echo ""

# ─── Step 2: Generate fake SkillPackage with episodes ─────────────────────
echo "── Step 2: Generate SkillPackage with 8 fake episodes (5 with lessons)"
cat > "$TMPDIR/skill_package.json" << 'PKGEOF'
{
  "schema_version": 2,
  "metadata": {
    "format": "po-skill/v2",
    "exported_at": "2026-03-12T03:00:00Z",
    "note_count": 2,
    "decision_count": 1
  },
  "skill": {
    "name": "m2-episodic-transfer-test",
    "description": "Test skill carrying episodic memories from a source PO instance. Contains patterns about graph operations, protocol FSM design, and knowledge transfer.",
    "tags": ["m2", "test", "episodic", "knowledge-transfer"],
    "trigger_patterns": [],
    "context_template": null,
    "cohesion": 0.75
  },
  "notes": [
    {
      "note_type": "pattern",
      "content": "Episodic memory enables cross-instance knowledge transfer by capturing WHY decisions were made, not just WHAT was changed. The Stimulus/Process/Outcome/Lesson structure provides full cognitive context.",
      "importance": "high",
      "tags": ["episodic", "knowledge-transfer", "architecture"]
    },
    {
      "note_type": "guideline",
      "content": "When importing episodes from another instance, always verify gap-closing precision: what fraction of imported episodes wire to local entities. Below 50% suggests domain mismatch.",
      "importance": "medium",
      "tags": ["episodic", "import", "quality"]
    }
  ],
  "decisions": [
    {
      "description": "Use weighted composite score (comprehensibility 40%, replay fidelity 30%, gap-closing 30%) for episodic evaluation",
      "rationale": "Comprehensibility is weighted highest because a lesson that cannot be understood out of context has zero transfer value regardless of other metrics.",
      "alternatives": ["Equal weights (33/33/33)", "Gap-closing dominant (20/20/60)"],
      "chosen_option": "Comprehensibility-dominant (40/30/30)"
    }
  ],
  "protocols": [],
  "episodes": [
    {
      "schema_version": 1,
      "stimulus": {
        "request": "Implement PRODUCED_DURING relation for ProtocolRun → Note/Decision tracking",
        "trigger": "protocol_transition"
      },
      "process": {
        "had_reasoning_tree": true,
        "states_visited": ["analyze", "implement", "validate", "done"],
        "duration_ms": 45000
      },
      "outcome": {
        "notes_produced": 3,
        "decisions_made": 1,
        "commits_made": 1,
        "files_modified": 4,
        "note_summaries": [],
        "decision_summaries": []
      },
      "validation": {
        "feedback_type": "explicit_positive",
        "score": 0.92,
        "evidence_count": 4
      },
      "lesson": {
        "abstract_pattern": "When adding a new graph relation, always create a Neo4j index on the relation properties and a backfill migration for existing data. Without the index, queries degrade from O(1) to O(n) on the relation scan.",
        "domain_tags": ["neo4j", "schema-migration", "performance"],
        "portability_layer": 2,
        "confidence": 0.88
      }
    },
    {
      "schema_version": 1,
      "stimulus": {
        "request": "Fix protocol FSM re-entry causing duplicate side effects",
        "trigger": "user_request"
      },
      "process": {
        "had_reasoning_tree": false,
        "states_visited": ["detect", "diagnose", "fix", "done"],
        "duration_ms": 22000
      },
      "outcome": {
        "notes_produced": 1,
        "decisions_made": 0,
        "commits_made": 1,
        "files_modified": 2,
        "note_summaries": [],
        "decision_summaries": []
      },
      "validation": {
        "feedback_type": "implicit_positive",
        "score": 0.85,
        "evidence_count": 1
      },
      "lesson": {
        "abstract_pattern": "Protocol FSM states should be idempotent — re-entering a state must not duplicate side effects. Guard transitions with a check on whether the action was already performed.",
        "domain_tags": ["fsm", "protocol", "idempotency"],
        "portability_layer": 3,
        "confidence": 0.91
      }
    },
    {
      "schema_version": 1,
      "stimulus": {
        "request": "Optimize Neo4j batch operations for skill import",
        "trigger": "protocol_transition"
      },
      "process": {
        "had_reasoning_tree": true,
        "states_visited": ["plan", "execute", "review", "done"],
        "duration_ms": 38000
      },
      "outcome": {
        "notes_produced": 2,
        "decisions_made": 1,
        "commits_made": 1,
        "files_modified": 3,
        "note_summaries": [],
        "decision_summaries": []
      },
      "validation": {
        "feedback_type": "explicit_positive",
        "score": 0.95,
        "evidence_count": 3
      },
      "lesson": {
        "abstract_pattern": "Batch UNWIND is critical for Neo4j performance — never N+1 loop over node creation. A single UNWIND query creating 50 nodes is 20x faster than 50 individual CREATE queries.",
        "domain_tags": ["neo4j", "performance", "batch"],
        "portability_layer": 2,
        "confidence": 0.94
      }
    },
    {
      "schema_version": 1,
      "stimulus": {
        "request": "Add episode mega-tool to MCP server",
        "trigger": "manual"
      },
      "process": {
        "had_reasoning_tree": false,
        "states_visited": ["start", "process", "done"],
        "duration_ms": 15000
      },
      "outcome": {
        "notes_produced": 1,
        "decisions_made": 0,
        "commits_made": 1,
        "files_modified": 4,
        "note_summaries": [],
        "decision_summaries": []
      },
      "validation": {
        "feedback_type": "implicit_positive",
        "score": 0.78,
        "evidence_count": 1
      },
      "lesson": {
        "abstract_pattern": "MCP mega-tool dispatch requires updating 4 locations: tools.rs schema, handlers.rs mega_tools array, mega_tool_to_legacy mapping, and try_handle_http HTTP dispatch. Missing any one causes silent routing failures.",
        "domain_tags": ["mcp", "architecture", "dispatch"],
        "portability_layer": 1,
        "confidence": 0.82
      }
    },
    {
      "schema_version": 1,
      "stimulus": {
        "request": "Improve knowledge retrieval for cross-project search",
        "trigger": "protocol_transition"
      },
      "process": {
        "had_reasoning_tree": true,
        "states_visited": ["analyze", "implement", "validate", "done"],
        "duration_ms": 52000
      },
      "outcome": {
        "notes_produced": 4,
        "decisions_made": 2,
        "commits_made": 2,
        "files_modified": 6,
        "note_summaries": [],
        "decision_summaries": []
      },
      "validation": {
        "feedback_type": "explicit_positive",
        "score": 0.89,
        "evidence_count": 6
      },
      "lesson": {
        "abstract_pattern": "Always prefer batch operations over N+1 loops when writing to a database — use UNWIND or bulk insert instead of individual queries. This applies universally to any graph or relational database.",
        "domain_tags": ["performance", "database", "universal"],
        "portability_layer": 3,
        "confidence": 0.96
      }
    },
    {
      "schema_version": 1,
      "stimulus": {
        "request": "Debug failing test after struct field addition",
        "trigger": "manual"
      },
      "process": {
        "had_reasoning_tree": false,
        "states_visited": ["detect", "diagnose", "fix", "done"],
        "duration_ms": 8000
      },
      "outcome": {
        "notes_produced": 0,
        "decisions_made": 0,
        "commits_made": 1,
        "files_modified": 2,
        "note_summaries": [],
        "decision_summaries": []
      },
      "validation": {
        "feedback_type": "none",
        "score": null,
        "evidence_count": 0
      },
      "lesson": null
    },
    {
      "schema_version": 1,
      "stimulus": {
        "request": "Run clippy and fix warnings",
        "trigger": "manual"
      },
      "process": {
        "had_reasoning_tree": false,
        "states_visited": ["start", "process", "done"],
        "duration_ms": 5000
      },
      "outcome": {
        "notes_produced": 0,
        "decisions_made": 0,
        "commits_made": 1,
        "files_modified": 3,
        "note_summaries": [],
        "decision_summaries": []
      },
      "validation": {
        "feedback_type": "implicit_positive",
        "score": 0.7,
        "evidence_count": 1
      },
      "lesson": null
    },
    {
      "schema_version": 1,
      "stimulus": {
        "request": "Implement artifact export endpoint for M2 bridge evaluation",
        "trigger": "protocol_transition"
      },
      "process": {
        "had_reasoning_tree": true,
        "states_visited": ["plan", "execute", "review", "done"],
        "duration_ms": 35000
      },
      "outcome": {
        "notes_produced": 2,
        "decisions_made": 1,
        "commits_made": 1,
        "files_modified": 5,
        "note_summaries": [],
        "decision_summaries": []
      },
      "validation": {
        "feedback_type": "explicit_positive",
        "score": 0.87,
        "evidence_count": 3
      },
      "lesson": null
    }
  ]
}
PKGEOF
PKG_SIZE=$(wc -c < "$TMPDIR/skill_package.json" | tr -d ' ')
EPISODE_COUNT=$(jq '.episodes | length' "$TMPDIR/skill_package.json")
EPISODES_WITH_LESSONS=$(jq '[.episodes[] | select(.lesson != null)] | length' "$TMPDIR/skill_package.json")
echo "  SkillPackage: ${PKG_SIZE} bytes"
echo "  Episodes: $EPISODE_COUNT (with lessons: $EPISODES_WITH_LESSONS)"
echo ""

# ─── Step 3: Import SkillPackage into instance 2 ─────────────────────────
echo "── Step 3: Import SkillPackage into instance 2"
IMPORT_BODY=$(jq -n \
  --arg pid "$PROJECT2_ID" \
  --slurpfile pkg "$TMPDIR/skill_package.json" \
  '{project_id: $pid, package: $pkg[0], conflict_strategy: "merge"}')

IMPORT_RESULT=$(curl -sf "$INSTANCE2/api/skills/import" \
  -H "Content-Type: application/json" \
  -d "$IMPORT_BODY" 2>&1) || { echo "  Import failed! Response:"; echo "$IMPORT_RESULT"; exit 1; }
echo "  Import result:"
echo "$IMPORT_RESULT" | jq .
NOTES_IMPORTED=$(echo "$IMPORT_RESULT" | jq '.notes_imported // 0')
EPISODES_IMPORTED=$(echo "$IMPORT_RESULT" | jq '.episodes_imported // 0')
DECISIONS_IMPORTED=$(echo "$IMPORT_RESULT" | jq '.decisions_imported // 0')
echo ""

# ─── Step 4: Verify imported notes on instance 2 ─────────────────────────
echo "── Step 4: Verify imported notes on instance 2"
NOTES2=$(curl -sf "$INSTANCE2/api/notes?slug=m2-test-receiver&limit=100")
TOTAL_NOTES=$(echo "$NOTES2" | jq '.total // (.items | length)')
EPISODE_NOTES=$(echo "$NOTES2" | jq '[.items[] | select(.tags[]? == "imported_episode")] | length')
IMPORTED_TAGS=$(echo "$NOTES2" | jq '[.items[] | select(.tags[]? == "imported")] | length')
echo "  Total notes on instance 2: $TOTAL_NOTES"
echo "  Notes tagged 'imported': $IMPORTED_TAGS"
echo "  Notes tagged 'imported_episode': $EPISODE_NOTES"

# Show episode-derived note details
if [ "$EPISODE_NOTES" -gt 0 ]; then
  echo ""
  echo "  Episode-derived notes:"
  echo "$NOTES2" | jq -r '.items[] | select(.tags[]? == "imported_episode") | "    - [\(.importance)] \(.content[0:80])..."'
fi
echo ""

# ─── Step 5: Verify skills on instance 2 ──────────────────────────────────
echo "── Step 5: Verify skills on instance 2"
SKILLS2=$(curl -sf "$INSTANCE2/api/skills?project_id=$PROJECT2_ID")
SKILL_COUNT=$(echo "$SKILLS2" | jq '.items | length')
echo "  Skills on instance 2: $SKILL_COUNT"
if [ "$SKILL_COUNT" -gt 0 ]; then
  echo "$SKILLS2" | jq -r '.items[] | "    - \(.name) (status: \(.status), energy: \(.energy))"'
fi
echo ""

# ─── Step 6: Test artifact export from instance 2 ────────────────────────
echo "── Step 6: Export artifact from instance 2 (round-trip test)"
ARTIFACT2=$(curl -sf "$INSTANCE2/api/episodes/export-artifact" \
  -H "Content-Type: application/json" \
  -d "{\"project_id\": \"$PROJECT2_ID\", \"include_structure\": true, \"max_episodes\": 50}" 2>&1) || echo "  (export returned empty — expected if no protocol runs yet)"
if echo "$ARTIFACT2" | jq . > /dev/null 2>&1; then
  EDGE_COUNT2=$(echo "$ARTIFACT2" | jq '.stats.edge_count // 0')
  EPISODE_COUNT2=$(echo "$ARTIFACT2" | jq '.stats.episode_count // 0')
  echo "  Artifact from instance 2: edges=$EDGE_COUNT2, episodes=$EPISODE_COUNT2"
else
  EDGE_COUNT2=0
  EPISODE_COUNT2=0
  echo "  No artifact available (no protocol runs on fresh instance — expected)"
fi
echo ""

# ─── Summary ──────────────────────────────────────────────────────────────
echo "============================================"
echo "=== M2 INTEGRATION TEST SUMMARY ==="
echo "============================================"
echo ""
echo "Input:"
echo "  SkillPackage: ${PKG_SIZE} bytes"
echo "  Episodes: $EPISODE_COUNT (with lessons: $EPISODES_WITH_LESSONS)"
echo "  Notes in package: 2"
echo "  Decisions in package: 1"
echo ""
echo "Output (Instance 2: m2-test-receiver):"
echo "  Notes imported: $NOTES_IMPORTED"
echo "  Episodes imported: $EPISODES_IMPORTED"
echo "  Decisions imported: $DECISIONS_IMPORTED"
echo "  Episode-derived notes: $EPISODE_NOTES"
echo "  Skills created: $SKILL_COUNT"
echo ""

# ─── Assertions ───────────────────────────────────────────────────────────
PASS=0
FAIL=0

check() {
  local desc="$1" cond="$2"
  if eval "$cond"; then
    echo "  PASS: $desc"
    PASS=$((PASS+1))
  else
    echo "  FAIL: $desc"
    FAIL=$((FAIL+1))
  fi
}

echo "Assertions:"
check "Import returned success" '[ -n "$IMPORT_RESULT" ]'
check "At least 2 notes imported (package notes)" '[ "$NOTES_IMPORTED" -ge 2 ]'
check "At least 1 episode imported" '[ "$EPISODES_IMPORTED" -ge 1 ]'
check "Skill created on instance 2" '[ "$SKILL_COUNT" -ge 1 ]'
check "Episode-derived notes created (lessons → pattern notes)" '[ "$EPISODE_NOTES" -ge 1 ]'
check "Total notes > package notes (episodes added extra)" '[ "$TOTAL_NOTES" -gt 2 ]'
echo ""
echo "Results: $PASS passed, $FAIL failed"

# ─── Cleanup ──────────────────────────────────────────────────────────────
echo ""
echo "── Cleanup: removing test project from instance 2"
curl -sf -X DELETE "$INSTANCE2/api/projects/$PROJECT2_ID" > /dev/null 2>&1 && echo "  Project deleted" || echo "  (delete failed, manual cleanup needed)"

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
echo ""
echo "M2 integration test PASSED."
