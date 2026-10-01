// =============================================================================
// Migration 002 — Tombstone: mark legacy placeholder tombstones
// =============================================================================
//
// Context:
//   Before real Ed25519 signing, retract persisted (:Tombstone) nodes with a
//   fake signature ('0' * 128) and issuer 'did:local:unknown' (or empty). They
//   prove nothing and must never be treated as authoritative revocations.
//   The application already classifies them at read time (`legacy: true`,
//   `verified: false` in GET /sharing/tombstones, and is_tombstoned() ignores
//   them); this migration persists an explicit marker so operators can query
//   them. It deletes NOTHING.
//
// STATUS: NOT EXECUTED. Review, then run manually:
//   cypher-shell -u neo4j -p <password> < migrations/002_tombstone_legacy_marker.cypher
//
// Idempotent. Reversible: see the ROLLBACK section at the bottom.
// =============================================================================

// Step 1: Preview (read-only)
MATCH (t:Tombstone)
WHERE t.signature_hex IS NULL OR t.signature_hex =~ '^0*$'
   OR t.issuer_did IS NULL OR t.issuer_did = '' OR t.issuer_did = 'did:local:unknown'
RETURN count(t) AS legacy_candidates;

// Step 2: Mark them (records which migration did it, so rollback is exact)
MATCH (t:Tombstone)
WHERE t.legacy IS NULL
  AND (t.signature_hex IS NULL OR t.signature_hex =~ '^0*$'
       OR t.issuer_did IS NULL OR t.issuer_did = '' OR t.issuer_did = 'did:local:unknown')
SET t.legacy = true,
    t.legacy_marked_by = '002'
RETURN count(t) AS marked;

// =============================================================================
// ROLLBACK (run only to undo Step 2; removes only what migration 002 added)
// =============================================================================
// MATCH (t:Tombstone)
// WHERE t.legacy_marked_by = '002'
// REMOVE t.legacy, t.legacy_marked_by
// RETURN count(t) AS unmarked;
