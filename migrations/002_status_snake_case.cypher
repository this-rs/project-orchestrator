// =============================================================================
// Migration 002 — Release / Milestone / WorkspaceMilestone status -> snake_case
// =============================================================================
//
// Context:
//   Release used to be written with Debug formatting (PascalCase: 'InProgress'),
//   Milestone/WorkspaceMilestone with serde (snake_case: 'in_progress').
//   Canonical encoding is now snake_case for these three labels. Plan, Task and
//   Step are NOT touched (still PascalCase). Reads and filters accept both, so
//   this migration is optional but recommended. Idempotent.
//
// Run:   cypher-shell -u neo4j -p <password> < migrations/002_status_snake_case.cypher
//
// ---- FORWARD ----------------------------------------------------------------
MATCH (r:Release) WHERE r.status IN ['Planned','InProgress','Released','Cancelled']
SET r.status_legacy = r.status,
    r.status = CASE r.status
      WHEN 'Planned' THEN 'planned' WHEN 'InProgress' THEN 'in_progress'
      WHEN 'Released' THEN 'released' WHEN 'Cancelled' THEN 'cancelled' END;

MATCH (m:Milestone) WHERE m.status IN ['Planned','Open','InProgress','Completed','Closed']
SET m.status_legacy = m.status,
    m.status = CASE m.status
      WHEN 'Planned' THEN 'planned' WHEN 'Open' THEN 'open'
      WHEN 'InProgress' THEN 'in_progress' WHEN 'Completed' THEN 'completed'
      WHEN 'Closed' THEN 'closed' END;

MATCH (m:WorkspaceMilestone) WHERE m.status IN ['Planned','Open','InProgress','Completed','Closed']
SET m.status_legacy = m.status,
    m.status = CASE m.status
      WHEN 'Planned' THEN 'planned' WHEN 'Open' THEN 'open'
      WHEN 'InProgress' THEN 'in_progress' WHEN 'Completed' THEN 'completed'
      WHEN 'Closed' THEN 'closed' END;

// ---- ROLLBACK (run separately; restores the exact previous value) -----------
// Only rows that carry `status_legacy` (i.e. changed by the forward step) are
// restored; rows that were already snake_case (Milestone) have no legacy value
// and keep it.
//
// MATCH (n) WHERE (n:Release OR n:Milestone OR n:WorkspaceMilestone)
//   AND n.status_legacy IS NOT NULL
// SET n.status = n.status_legacy
// REMOVE n.status_legacy;
//
// Caveat: rolling back Release to PascalCase is only needed with code older than
// this change; current code reads both. Milestone rows that were snake_case
// before the forward step were never altered (the WHERE only matches PascalCase).
