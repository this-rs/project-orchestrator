#!/usr/bin/env bash
# Collapse the append-only Alert backlog onto one node per condition.
#
# NOTE: the server now does this itself at startup — data migration
# `2026-09-fold-legacy-alerts` (src/neo4j/data_migrations.rs) — so upgraded
# installs need no manual step. This script remains for manual repairs and
# for DRY_RUN inspection.
#
# Context: create_alert_node used CREATE, so every heartbeat tick appended a
# node. Measured 2026-08-30: 1,445,290 Alert nodes = 40.5% of the whole graph;
# git_drift alone held 1,355,752 nodes for 1,591 distinct messages (852x), one
# message repeated 116,083 times. Measured 2026-09-23, after the dedup fix
# shipped: 1,585,595 legacy nodes without a dedup_key.
#
# Constraint-safe. The server now creates a UNIQUE constraint on
# Alert.dedup_key at startup and MERGEs new alerts on it, so by the time this
# runs there is usually already ONE keyed node per live condition. The first
# version of this script tagged every legacy node with its key and collapsed
# afterwards; with the constraint in place that first SET is rejected outright.
# Instead, each batch folds legacy nodes straight into their target:
#   - the existing node carrying that dedup_key, if the server created one;
#   - otherwise ONE legacy node, which alone receives the key.
# No two nodes ever hold the same key, so the constraint is never violated.
#
# Keys are built exactly like AlertNode::make_dedup_key / the heartbeat checks:
#   "<alert_type>:<project_id|global>:<subject>", subject = behind-origin,
#   project-stagnating, guidelines-missing, else the message.
# The surviving node keeps: the OLDEST first_seen, the NEWEST last_seen and its
# message, the highest severity seen, acknowledged=true if ANY folded node was
# (never silently un-ack), and occurrence_count summed.
#
# IDEMPOTENT and INCREMENTAL: bounded batches, safe to interrupt and rerun.
#
#   DRY_RUN=1 ./scripts/collapse-duplicate-alerts.sh
#       runs ONE real batch inside a transaction, prints what it did, then
#       ROLLS BACK: nothing is written.
#   ./scripts/collapse-duplicate-alerts.sh
#       runs until no legacy node is left.
set -euo pipefail

CFG="${CFG:-config.yaml}"
NEO_BASE="${NEO_BASE:-http://localhost:7474/db/neo4j}"
USER="${NEO_USER:-neo4j}"
PASS="${NEO_PASS:-$(grep -A3 '^neo4j:' "$CFG" | grep 'password:' | sed 's/.*password: *//' | tr -d '"'"'"'')}"
BATCH="${BATCH:-5000}"
DRY_RUN="${DRY_RUN:-0}"
MAX_RETRIES="${MAX_RETRIES:-5}"

body() { python3 -c 'import json,sys;print(json.dumps({"statements":[{"statement":sys.stdin.read()}]}))' <<<"$1"; }
post() { curl -s -u "$USER:$PASS" -H 'Content-Type: application/json' -d "$(body "$2")" "$NEO_BASE/$1"; }

# Print the first Cypher error of a response, or nothing.
err_of() { python3 -c "import sys,json;e=json.load(sys.stdin).get('errors') or [];print(e[0]['message'][:300] if e else '')" <<<"$1"; }
# Print row 0 of result 0 as space-separated values.
row_of() { python3 -c "import sys,json;d=json.load(sys.stdin)['results'][0]['data'];print(' '.join(str(v) for v in d[0]['row']) if d else '')" <<<"$1"; }

legacy_count() { row_of "$(post tx/commit 'MATCH (a:Alert) WHERE a.dedup_key IS NULL RETURN count(a)')"; }

FOLD="
MATCH (a:Alert) WHERE a.dedup_key IS NULL
WITH a LIMIT $BATCH
WITH a,
     a.alert_type + ':' +
     CASE WHEN a.project_id IS NULL OR a.project_id = '' THEN 'global' ELSE a.project_id END + ':' +
     CASE a.alert_type
       WHEN 'git_drift'      THEN 'behind-origin'
       WHEN 'stagnation'     THEN 'project-stagnating'
       WHEN 'convention_gap' THEN 'guidelines-missing'
       ELSE a.message END AS k
WITH k, collect(a) AS olds
OPTIONAL MATCH (e:Alert {dedup_key: k})
WITH k, olds, e, CASE WHEN e IS NULL THEN head(olds) ELSE e END AS keep
WITH k, olds, e, keep,
     [x IN olds WHERE x <> keep] AS dupes,
     reduce(n = null, x IN olds |
       CASE WHEN n IS NULL OR x.created_at > n.t THEN {t: x.created_at, m: x.message} ELSE n END) AS newest,
     reduce(m = coalesce(e.first_seen, e.created_at), x IN olds |
       CASE WHEN m IS NULL OR x.created_at < m THEN x.created_at ELSE m END) AS first,
     coalesce(e.occurrence_count, 0) + size(olds) AS occ,
     [s IN [e.severity] + [x IN olds | x.severity] WHERE s IS NOT NULL] AS sevs,
     coalesce(e.acknowledged, false) OR any(x IN olds WHERE coalesce(x.acknowledged, false)) AS ack
SET keep.dedup_key        = k,
    keep.occurrence_count = occ,
    keep.first_seen       = first,
    keep.last_seen        = CASE WHEN e IS NULL THEN newest.t ELSE coalesce(e.last_seen, newest.t) END,
    keep.message          = CASE WHEN e IS NULL THEN newest.m ELSE keep.message END,
    keep.severity         = CASE WHEN 'critical' IN sevs THEN 'critical'
                                 WHEN 'warning'  IN sevs THEN 'warning'
                                 ELSE 'info' END,
    keep.acknowledged     = ack
FOREACH (d IN dupes | DETACH DELETE d)
RETURN count(k) AS conditions,
       sum(size(olds)) AS folded,
       sum(size(dupes)) AS deleted,
       sum(CASE WHEN e IS NULL THEN 1 ELSE 0 END) AS newly_keyed"

BEFORE=$(legacy_count)
echo "==> legacy Alert nodes without dedup_key: $BEFORE (batch=$BATCH)"

if [ "$DRY_RUN" = "1" ]; then
  echo "==> DRY RUN: one batch inside a transaction, then rollback"
  R=$(post tx "$FOLD")
  COMMIT=$(python3 -c "import sys,json;print(json.load(sys.stdin).get('commit',''))" <<<"$R")
  [ -n "$COMMIT" ] && curl -s -X DELETE -u "$USER:$PASS" "${COMMIT%/commit}" >/dev/null
  E=$(err_of "$R")
  if [ -n "$E" ]; then echo "    ERROR: $E"; exit 1; fi
  read -r C F D K <<<"$(row_of "$R")"
  echo "    would fold $F legacy nodes into $C condition(s): delete $D, key $K new node(s)"
  echo "==> rolled back; legacy nodes still: $(legacy_count)"
  exit 0
fi

echo "==> folding legacy alerts into one node per condition"
FAILS=0
while :; do
  R=$(post tx/commit "$FOLD")
  E=$(err_of "$R")
  if [ -n "$E" ]; then
    # Typically a transient conflict with a heartbeat MERGE on the same key;
    # the next attempt folds into the node the server just created.
    FAILS=$((FAILS + 1))
    echo "    batch failed ($FAILS/$MAX_RETRIES): $E"
    [ "$FAILS" -ge "$MAX_RETRIES" ] && { echo "==> giving up"; exit 1; }
    continue
  fi
  FAILS=0
  read -r C F D K <<<"$(row_of "$R")"
  [ "${F:-0}" -eq 0 ] && break
  echo "    folded $F into $C condition(s): deleted $D, keyed $K"
done

echo "==> recomputing priority"
post tx/commit "MATCH (a:Alert) SET a.priority = CASE WHEN coalesce(a.acknowledged,false) THEN 0.0 ELSE
      (CASE a.severity WHEN 'critical' THEN 1.0 WHEN 'warning' THEN 0.5 ELSE 0.2 END)
      * (1.0 + CASE WHEN log(toFloat(coalesce(a.occurrence_count,1)))/10.0 > 0.5 THEN 0.5
                    ELSE log(toFloat(coalesce(a.occurrence_count,1)))/10.0 END) END
    RETURN count(a)" >/dev/null
echo "==> done"
echo "    legacy nodes left: $(legacy_count)"
echo "    Alert nodes now:   $(row_of "$(post tx/commit 'MATCH (a:Alert) RETURN count(a)')")"
