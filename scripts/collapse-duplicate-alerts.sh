#!/usr/bin/env bash
# Collapse the append-only Alert backlog onto one node per condition.
#
# Context: create_alert_node used CREATE, so every heartbeat tick appended a
# node. Measured 2026-08-30: 1,445,290 Alert nodes = 40.5% of the whole graph;
# git_drift alone held 1,355,752 nodes for 1,591 distinct messages (852x), one
# message repeated 116,083 times.
#
# This script is IDEMPOTENT and INCREMENTAL: it collapses in bounded batches so
# it never opens a multi-million-node transaction. Run it repeatedly until
# `remaining` reaches 0. Safe to interrupt at any point.
#
# The surviving node keeps: the OLDEST created_at (as first_seen), the NEWEST
# created_at (as last_seen) and its message, the highest severity seen, and
# acknowledged=true if ANY duplicate was acknowledged (never silently un-ack).
set -euo pipefail

CFG="${CFG:-config.yaml}"
NEO_URL="${NEO_URL:-http://localhost:7474/db/neo4j/tx/commit}"
USER="${NEO_USER:-neo4j}"
PASS="${NEO_PASS:-$(grep -A3 '^neo4j:' "$CFG" | grep 'password:' | sed 's/.*password: *//' | tr -d '"'"'"'')}"
BATCH="${BATCH:-200}"

cy() { curl -s -u "$USER:$PASS" -H 'Content-Type: application/json' \
        -d "{\"statements\":[{\"statement\":$(python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))' <<<"$1")}]}" \
        "$NEO_URL"; }

# Backfill dedup_key on legacy nodes that have none. Subject falls back to the
# message, EXCEPT for git_drift where the volatile commit count is stripped so
# the whole backlog for a project collapses to one node.
echo "==> backfilling dedup_key (batch=$BATCH)"
while :; do
  R=$(cy "MATCH (a:Alert) WHERE a.dedup_key IS NULL WITH a LIMIT $BATCH
      WITH a, coalesce(a.project_id,'global') AS scope,
           CASE WHEN a.alert_type = 'git_drift'  THEN 'behind-origin'
                WHEN a.alert_type = 'stagnation' THEN 'project-stagnating'
                WHEN a.alert_type = 'convention_gap' THEN 'guidelines-missing'
                ELSE a.message END AS subject
      SET a.dedup_key = a.alert_type + ':' + scope + ':' + subject
      RETURN count(a) AS n")
  N=$(python3 -c "import sys,json;print(json.load(sys.stdin)['results'][0]['data'][0]['row'][0])" <<<"$R")
  echo "    tagged $N"
  [ "$N" -eq 0 ] && break
done

echo "==> collapsing duplicates (batch=$BATCH)"
while :; do
  R=$(cy "MATCH (a:Alert) WITH a.dedup_key AS k, count(*) AS c WHERE c > 1
      WITH k ORDER BY c DESC LIMIT $BATCH
      CALL { WITH k
        MATCH (a:Alert {dedup_key: k})
        WITH a ORDER BY a.created_at ASC
        WITH collect(a) AS all, count(*) AS total
        WITH all, total, head(all) AS keep, last(all) AS newest
        SET keep.occurrence_count = coalesce(keep.occurrence_count,0) + total,
            keep.first_seen       = head(all).created_at,
            keep.last_seen        = newest.created_at,
            keep.message          = newest.message,
            keep.severity         = CASE
                WHEN any(x IN all WHERE x.severity = 'critical') THEN 'critical'
                WHEN any(x IN all WHERE x.severity = 'warning')  THEN 'warning'
                ELSE 'info' END,
            keep.acknowledged     = any(x IN all WHERE coalesce(x.acknowledged,false))
        WITH keep, [x IN all WHERE x <> keep] AS dupes
        FOREACH (d IN dupes | DETACH DELETE d)
        RETURN size(dupes) AS removed
      }
      RETURN sum(removed) AS removed")
  N=$(python3 -c "import sys,json;d=json.load(sys.stdin);r=d['results'][0]['data'];print(r[0]['row'][0] or 0 if r else 0)" <<<"$R")
  echo "    removed $N"
  [ "$N" -eq 0 ] && break
done

echo "==> recomputing priority"
cy "MATCH (a:Alert) SET a.priority = CASE WHEN coalesce(a.acknowledged,false) THEN 0.0 ELSE
      (CASE a.severity WHEN 'critical' THEN 1.0 WHEN 'warning' THEN 0.5 ELSE 0.2 END)
      * (1.0 + CASE WHEN log(toFloat(coalesce(a.occurrence_count,1)))/10.0 > 0.5 THEN 0.5
                    ELSE log(toFloat(coalesce(a.occurrence_count,1)))/10.0 END) END
    RETURN count(a) AS scored" >/dev/null
echo "==> done"
cy "MATCH (a:Alert) RETURN count(a) AS remaining" | python3 -c "import sys,json;print('    Alert nodes now:', json.load(sys.stdin)['results'][0]['data'][0]['row'][0])"
