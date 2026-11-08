#!/usr/bin/env bash
#
# An ephemeral COPY of production, to prove a knowledge-repair migration before
# it touches the real graph.
#
# Why: the migration `2026-11-retype-auto-observed` rewrites the type, the
# importance, the horizon and the energy of every note a heuristic wrote. Its
# effect cannot be judged on three hand-made fixtures — the population that
# matters is 4 000 notes written over months, and the thing that was actually
# wrong (a janitor that never reaches the low-energy queue) only shows up at
# that scale. So: copy the notes, load them into a throwaway instance, run the
# REAL migration through the REAL code path, and measure before/after.
#
# Usage:
#   scripts/ephemeral-prod-copy.sh up      # create the copy (export + load)
#   scripts/ephemeral-prod-copy.sh test    # run the migration tests on the copy
#   scripts/ephemeral-prod-copy.sh down    # destroy the copy
#
# Nothing here ever writes to the source. The export is a read query; the
# load targets the throwaway container only, and the script refuses to run if
# the target URI is the live one.

set -euo pipefail

SRC_CONTAINER="${SRC_CONTAINER:-orchestrator-neo4j}"
SRC_USER="${SRC_USER:-neo4j}"
SRC_PASSWORD="${SRC_PASSWORD:-orchestrator123}"
SRC_HTTP="${SRC_HTTP:-http://localhost:7474}"
SRC_BOLT="${SRC_BOLT:-bolt://localhost:7687}"

DEST_CONTAINER="${DEST_CONTAINER:-po-ephemeral-neo4j}"
DEST_PORT="${DEST_PORT:-27699}"
DEST_PASSWORD="testpassword"
WORK="${WORK:-/tmp/po-ephemeral-copy}"

# The one guard that matters: a copy is never the original.
if [ "${DEST_PORT}" = "7687" ]; then
  echo "refusing to run: DEST_PORT 7687 is the live instance" >&2
  exit 1
fi

up() {
  mkdir -p "${WORK}/chunks"
  echo "== export the Note population from ${SRC_CONTAINER} (read-only)"
  curl -sf -u "${SRC_USER}:${SRC_PASSWORD}" -H 'Content-Type: application/json' \
    -X POST "${SRC_HTTP}/db/neo4j/tx/commit" \
    -d '{"statements":[{"statement":"CALL apoc.export.cypher.query('\''MATCH (n:Note) RETURN n'\'', null, {format:'\''cypher-shell'\'', batchSize:500, separateFiles:false, stream:true, storeNodeIds:false, cleanupImportFile:false}) YIELD cypherStatements AS c RETURN c"}]}' \
    -o "${WORK}/export.json"
  jq -r '.results[0].data[].row[0]' "${WORK}/export.json" > "${WORK}/notes.cypher"

  echo "== split into one transaction per batch"
  # APOC emits one :begin/:commit block per batchSize nodes. Loading the file in
  # one cypher-shell run holds a transaction long enough to exhaust the heap on a
  # real population; one file per block, loaded in parallel, does not.
  rm -f "${WORK}/chunks/"*.cypher
  awk '/^:begin/{n++; f=sprintf("'"${WORK}"'/chunks/chunk-%04d.cypher", n); inb=1}
       inb && !/^CALL db.awaitIndexes/{print > f}
       /^:commit/{close(f); inb=0}' "${WORK}/notes.cypher"

  echo "== create the throwaway instance on port ${DEST_PORT}"
  docker rm -f "${DEST_CONTAINER}" >/dev/null 2>&1 || true
  docker run -d --name "${DEST_CONTAINER}" \
    -e NEO4J_AUTH="neo4j/${DEST_PASSWORD}" \
    -e NEO4J_PLUGINS='["apoc"]' \
    -e NEO4J_dbms_security_procedures_unrestricted='apoc.*' \
    -e NEO4J_server_memory_heap_max__size=4G \
    -p "${DEST_PORT}:7687" -p 7475:7474 neo4j:5.26.20-community >/dev/null

  echo "== wait for it"
  for _ in $(seq 1 60); do
    docker exec "${DEST_CONTAINER}" cypher-shell -u neo4j -p "${DEST_PASSWORD}" \
      "RETURN 1" --format plain >/dev/null 2>&1 && break
    sleep 2
  done

  echo "== load (one transaction per chunk, six at a time)"
  docker cp "${WORK}/chunks" "${DEST_CONTAINER}:/tmp/chunks" >/dev/null
  # The first chunk carries the index/constraint creation: it must run alone.
  docker exec "${DEST_CONTAINER}" cypher-shell -u neo4j -p "${DEST_PASSWORD}" \
    --format plain -f /tmp/chunks/chunk-0001.cypher >/dev/null
  docker exec "${DEST_CONTAINER}" sh -c \
    'ls /tmp/chunks/chunk-*.cypher | tail -n +2 | xargs -P 6 -I{} cypher-shell -u neo4j -p testpassword --format plain -f {} >/dev/null'

  echo "== done. Before-state:"
  docker exec "${DEST_CONTAINER}" cypher-shell -u neo4j -p "${DEST_PASSWORD}" \
    "MATCH (n:Note) RETURN count(n) AS notes,
     sum(CASE WHEN n.status='active' THEN 1 ELSE 0 END) AS active,
     sum(CASE WHEN n.memory_horizon='consolidated' THEN 1 ELSE 0 END) AS consolidated,
     count(CASE WHEN n.created_by IN ['system/observation-detector','system/rfc-detector'] THEN 1 END) AS detector" \
    --format plain
}

test_copy() {
  echo "== the real code path, against the copy"
  ( cd "$(dirname "$0")/.." && \
    NEO4J_URI="bolt://localhost:${DEST_PORT}" NEO4J_USER=neo4j NEO4J_PASSWORD="${DEST_PASSWORD}" \
    cargo test --test data_migrations -- --nocapture )
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
