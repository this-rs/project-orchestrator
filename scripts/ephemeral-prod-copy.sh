#!/usr/bin/env bash
#
# An ephemeral COPY of production, to prove a knowledge-repair migration before
# it touches the real graph.
#
# Why a copy and not three fixtures: `2026-11-retype-auto-observed` rewrites the
# type, the importance, the horizon and the energy of every note a heuristic
# wrote. Whether that is right is a question about four thousand notes written
# over months, not about a fixture. Measured on the real population (project
# 6fc5c5f7): 417 detector notes, 417 claiming gotcha/guideline/pattern/rfc, 353
# of them CONSOLIDATED — the copy is what makes that visible.
#
# Usage:
#   scripts/ephemeral-prod-copy.sh up      # create the copy (export + load)
#   scripts/ephemeral-prod-copy.sh test    # run the repair proof on the copy
#   scripts/ephemeral-prod-copy.sh down    # destroy the copy
#
# Nothing here writes to the source: the export is a read query, and the load
# targets the throwaway container only.
#
# NOTE ON SIZE — a whole-instance copy (30 900 notes, 124 MB of Cypher because
# every note carries its change history) exhausts the JVM heap of a 3 G instance
# mid-transaction. Export the population you are repairing: that is what
# `EXPORT_MATCH` is for, and it is what the measurement is about.

set -euo pipefail

SRC_CONTAINER="${SRC_CONTAINER:-orchestrator-neo4j}"
SRC_USER="${SRC_USER:-neo4j}"
SRC_PASSWORD="${SRC_PASSWORD:-orchestrator123}"
SRC_HTTP="${SRC_HTTP:-http://localhost:7474}"

DEST_CONTAINER="${DEST_CONTAINER:-po-ephemeral-neo4j}"
DEST_PORT="${DEST_PORT:-27699}"
DEST_PASSWORD="testpassword"
WORK="${WORK:-/tmp/po-ephemeral-copy}"

# What the copy contains. Default: the notes a heuristic wrote, which is the
# population every repair discussed here is about.
EXPORT_MATCH="${EXPORT_MATCH:-MATCH (n:Note) WHERE n.created_by IN ['system/observation-detector','system/rfc-detector'] OR 'auto-observed' IN coalesce(n.tags,[]) RETURN n}"

if [ "${DEST_PORT}" = "7687" ]; then
  echo "refusing to run: DEST_PORT 7687 is the live instance" >&2
  exit 1
fi

up() {
  mkdir -p "${WORK}/chunks"
  echo "== export from ${SRC_CONTAINER} (read-only)"
  jq -n --arg q "CALL apoc.export.cypher.query(\$q, null, {format:'cypher-shell', batchSize:300, separateFiles:false, stream:true, storeNodeIds:false, cleanupImportFile:false}) YIELD cypherStatements AS c RETURN c" \
        --arg match "${EXPORT_MATCH}" \
        '{statements: [{statement: "$q", parameters: {q: $match}}]}' > "${WORK}/payload.json"
  curl -sf -u "${SRC_USER}:${SRC_PASSWORD}" -H 'Content-Type: application/json' \
    -X POST "${SRC_HTTP}/db/neo4j/tx/commit" -d @"${WORK}/payload.json" -o "${WORK}/export.json"
  jq -r '.results[0].data[].row[0]' "${WORK}/export.json" > "${WORK}/notes.cypher"

  echo "== one file per transaction"
  rm -f "${WORK}/chunks/"*.cypher
  awk -v dir="${WORK}/chunks" '/^:begin/{n++; f=sprintf("%s/chunk-%04d.cypher", dir, n); inb=1}
       inb && !/^CALL db.awaitIndexes/{print > f}
       /^:commit/{close(f); inb=0}' "${WORK}/notes.cypher"

  echo "== throwaway instance on port ${DEST_PORT}"
  docker rm -f "${DEST_CONTAINER}" >/dev/null 2>&1 || true
  docker run -d --name "${DEST_CONTAINER}" \
    -e NEO4J_AUTH="neo4j/${DEST_PASSWORD}" \
    -e NEO4J_PLUGINS='["apoc"]' \
    -e NEO4J_dbms_security_procedures_unrestricted='apoc.*' \
    -e NEO4J_server_memory_heap_max__size=3G \
    -p "${DEST_PORT}:7687" -p 7475:7474 neo4j:5.26.20-community >/dev/null

  for _ in $(seq 1 60); do
    docker exec "${DEST_CONTAINER}" cypher-shell -u neo4j -p "${DEST_PASSWORD}" \
      "RETURN 1" --format plain >/dev/null 2>&1 && break
    sleep 2
  done

  echo "== load (first chunk alone: it creates the constraint; then sequentially)"
  docker cp "${WORK}/chunks" "${DEST_CONTAINER}:/tmp/chunks" >/dev/null
  docker exec "${DEST_CONTAINER}" sh -c '
    set -e
    cypher-shell -u neo4j -p testpassword --format plain -f /tmp/chunks/chunk-0001.cypher >/dev/null
    for f in $(ls /tmp/chunks/chunk-*.cypher | tail -n +2); do
      cypher-shell -u neo4j -p testpassword --format plain -f "$f" >/dev/null
    done'

  echo "== before-state:"
  docker exec "${DEST_CONTAINER}" cypher-shell -u neo4j -p "${DEST_PASSWORD}" \
    "MATCH (n:Note) RETURN count(n) AS notes,
     count(CASE WHEN n.note_type IN ['gotcha','guideline','pattern','rfc'] THEN 1 END) AS claimed_knowledge,
     count(CASE WHEN n.memory_horizon='consolidated' THEN 1 END) AS consolidated,
     count(CASE WHEN n.status='active' THEN 1 END) AS active" --format plain
}

test_copy() {
  echo "== the real code path, against the copy"
  ( cd "$(dirname "$0")/.." && \
    NEO4J_URI="bolt://localhost:${DEST_PORT}" NEO4J_USER=neo4j NEO4J_PASSWORD="${DEST_PASSWORD}" \
    cargo test --test retype_auto_observed -- --nocapture )
}

down() {
  docker rm -f "${DEST_CONTAINER}" >/dev/null 2>&1 || true
  echo "destroyed ${DEST_CONTAINER}"
}

case "${1:-}" in
  up) up ;;
  test) test_copy ;;
  down) down ;;
  *) echo "usage: $0 {up|test|down}" >&2; exit 2 ;;
esac
