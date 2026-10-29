//! Data migrations — one-time, batched, idempotent cleanups run by the
//! server in the background at startup.
//!
//! These exist so that an install upgrading to a fixed version gets its data
//! repaired without any manual step. Each migration:
//! - runs in bounded batches (a batch query returns how many items it
//!   `processed`), until a batch processes nothing;
//! - is idempotent — a batch only ever touches items still needing it, so an
//!   interrupted run simply resumes on the next start;
//! - records a `(:DataMigration {id})` marker once complete, so later starts
//!   skip it.
//!
//! Every batch query takes a `$scope` parameter: `null` for the whole
//! database, or a project id to restrict the migration to one project (used
//! by the integration tests, which share one Neo4j instance).
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::time::Instant;

use anyhow::{Context, Result};
use neo4rs::query;
use serde::Serialize;
use tracing::{info, warn};
use uuid::Uuid;

use super::client::Neo4jClient;

/// Batch size for data migrations — small enough to keep each transaction
/// short while the server is serving requests.
const BATCH: i64 = 2000;

/// Consecutive failures of one batch before the migration gives up for this
/// start (a failure is typically a transient lock conflict with live writes).
const MAX_RETRIES: u32 = 5;

/// Hard cap on batches per migration and start, against a batch query that
/// would keep reporting progress without converging.
const MAX_BATCHES: u32 = 10_000;

/// Relationships that make a skill referenced by something other than its
/// member notes/decisions. A referenced skill is never archived as a
/// duplicate nor deleted as empty.
const SKILL_REFERENCES: &str = "MASTERS|USED_SKILL|BELONGS_TO_SKILL|LINKED_TO|HAS_SKILL";

pub struct DataMigration {
    pub id: &'static str,
    pub description: &'static str,
    /// Batch query. Params: `$scope` (string or null), `$batch`. Must return
    /// one row with a `processed` integer column.
    pub batch: String,
    /// Optional query run once after the last batch. Param: `$scope`.
    pub finalize: Option<&'static str>,
}

/// Fold append-only legacy alerts (no `dedup_key`) into one node per
/// condition. Same logic as `scripts/collapse-duplicate-alerts.sh`: the key
/// mirrors `AlertNode::make_dedup_key`; the surviving node keeps the oldest
/// first_seen, the newest last_seen/message, the highest severity, stays
/// acknowledged if any folded node was, and sums occurrence counts. Folding
/// into an already-keyed node never creates a second holder of a key, so the
/// UNIQUE constraint on `Alert.dedup_key` is never violated.
const FOLD_LEGACY_ALERTS: &str = r#"
MATCH (a:Alert) WHERE a.dedup_key IS NULL AND ($scope IS NULL OR a.project_id = $scope)
WITH a LIMIT $batch
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
RETURN coalesce(sum(size(olds)), 0) AS processed
"#;

const RECOMPUTE_ALERT_PRIORITY: &str = r#"
MATCH (a:Alert) WHERE $scope IS NULL OR a.project_id = $scope
SET a.priority = CASE WHEN coalesce(a.acknowledged, false) THEN 0.0 ELSE
      (CASE a.severity WHEN 'critical' THEN 1.0 WHEN 'warning' THEN 0.5 ELSE 0.2 END)
      * (1.0 + CASE WHEN log(toFloat(coalesce(a.occurrence_count, 1))) / 10.0 > 0.5 THEN 0.5
                    ELSE log(toFloat(coalesce(a.occurrence_count, 1))) / 10.0 END) END
"#;

/// Archive duplicate live skills: within each (project, name) group of
/// non-archived skills, keep the richest one (most members, then
/// referenced, then oldest) and archive the unreferenced others, detaching
/// their members (exactly what orphan archiving does). Referenced
/// duplicates are left alone too: archiving them could strand a persona or
/// a protocol on an archived skill.
///
/// The duplicates were produced by skill evolution comparing clusters with a
/// truncated snapshot (see `get_live_skills_for_project`).
fn archive_duplicate_skills_query() -> String {
    format!(
        r#"
MATCH (s:Skill)
WHERE s.status <> 'archived' AND ($scope IS NULL OR s.project_id = $scope)
WITH s.project_id AS p, s.name AS n, collect(s) AS grp
WHERE size(grp) > 1
UNWIND grp AS s
WITH p, n, s,
     COUNT {{ (s)<-[:MEMBER_OF|MEMBER_OF_SKILL]-() }} AS members,
     COUNT {{ (s)<-[:{SKILL_REFERENCES}]-() }} + COUNT {{ (pr:Protocol) WHERE pr.skill_id = s.id }} AS refs
ORDER BY p, n, members DESC, refs DESC, s.created_at ASC
WITH p, n, collect({{skill: s, refs: refs}}) AS ranked
UNWIND tail(ranked) AS r
WITH r WHERE r.refs = 0
WITH r.skill AS d LIMIT $batch
SET d.status = 'archived', d.updated_at = datetime(), d.archived_reason = 'duplicate'
WITH d
OPTIONAL MATCH (d)<-[m:MEMBER_OF|MEMBER_OF_SKILL]-()
DELETE m
RETURN count(DISTINCT d) AS processed
"#
    )
}

/// Delete archived skills that nothing points to anymore: no member, no
/// reference (see [`SKILL_REFERENCES`]) and no protocol `skill_id`. Params:
/// `$scope`, `$batch`, and `$before` (ISO datetime or null): only skills
/// archived before it are deleted.
fn purge_archived_empty_skills_query() -> String {
    format!(
        r#"
MATCH (s:Skill {{status: 'archived'}})
WHERE ($scope IS NULL OR s.project_id = $scope)
  AND ($before IS NULL OR s.updated_at < datetime($before))
  AND NOT (s)<-[:MEMBER_OF|MEMBER_OF_SKILL|{SKILL_REFERENCES}]-()
  AND NOT EXISTS {{ MATCH (pr:Protocol) WHERE pr.skill_id = s.id }}
WITH s LIMIT $batch
WITH collect(s) AS doomed
FOREACH (x IN doomed | DETACH DELETE x)
RETURN size(doomed) AS processed
"#
    )
}

/// Restore notes archived by the compounding energy-decay bug.
///
/// `update_energy_scores` re-applied the whole decay since `last_activated`
/// to the already-decayed energy on every call, and deep maintenance called it
/// once per project: energy of idle notes collapsed, and consolidation
/// archived them as `low_energy_60d` — ~4,000 notes on m4, 3,051 of them
/// consolidated gotchas/patterns/guidelines. Their energy is reset to the
/// value the intended decay gives (90-day time constant from the last
/// activation, as the maintenance and `boost_energy` used), floored at 0.1 so
/// they are not re-archived immediately. `restored_from_energy_bug` makes
/// each note eligible once: a later, legitimate archival sticks. A note that
/// was superseded is never restored — newer knowledge wins — and invalidated
/// notes (status `obsolete`) are not candidates.
const RESTORE_ENERGY_DECAY_VICTIMS: &str = r#"
MATCH (n:Note {status: 'archived'})
WHERE ($scope IS NULL OR n.project_id = $scope)
  AND n.changes_json CONTAINS 'low_energy_60d'
  AND n.restored_from_energy_bug IS NULL
  // Newer knowledge wins: never bring back a note that was replaced.
  AND n.superseded_by IS NULL
  AND NOT ()-[:SUPERSEDES]->(n)
WITH n LIMIT $batch
WITH n, toFloat(duration.inSeconds(datetime(coalesce(n.last_activated, n.created_at)), datetime()).seconds) / 86400.0 AS idle_days
WITH n, exp(-idle_days / 90.0) AS intended
SET n.status = 'active',
    n.energy = CASE WHEN intended > 0.1 THEN intended ELSE 0.1 END,
    n.energy_updated_at = datetime(),
    n.restored_from_energy_bug = datetime()
RETURN count(n) AS processed
"#;

/// Raise the energy of active notes the same bug over-decayed, to the value
/// the intended decay gives — never lowering it. Stamps `energy_updated_at`
/// so the fixed incremental decay starts from now.
const REBASE_ACTIVE_NOTE_ENERGY: &str = r#"
MATCH (n:Note {status: 'active'})
WHERE ($scope IS NULL OR n.project_id = $scope)
  AND n.energy_rebased_at IS NULL
WITH n LIMIT $batch
WITH n, toFloat(duration.inSeconds(datetime(coalesce(n.last_activated, n.created_at)), datetime()).seconds) / 86400.0 AS idle_days
WITH n, exp(-idle_days / 90.0) AS intended
SET n.energy = CASE WHEN intended > coalesce(n.energy, 0.0) THEN intended ELSE n.energy END,
    n.energy_updated_at = datetime(),
    n.energy_rebased_at = datetime()
RETURN count(n) AS processed
"#;

/// Re-home the KNOWLEDGE that points INTO a worktree copy, onto the canonical
/// file, BEFORE the purge deletes the copies.
///
/// A worktree path `…/_wt/<branch>/<relpath>` (or `.wt`, `.rlwt`, `worktrees`)
/// is a copy of the canonical `…/<relpath>`. Notes, decisions, feature graphs,
/// sessions and process steps that linked to a copy — 257 123 edges measured on
/// 29/10/2026, 96% of them onto `File` — must follow the file, not vanish with
/// the copy.
///
/// The scaffolding depth VARIES, and assuming one depth loses knowledge: a
/// fixed two-level strip would have detached 14 282 edges under `.rlwt/src/`
/// (the worktree root IS `.rlwt`) and 23 703 under `worktrees/tasks/<name>/`.
/// The candidates are therefore generated at every depth and the first that
/// names a real file wins; `elun`'s agent worktrees, nested under
/// `src/.auto-claude/worktrees/tasks/<name>/`, need the `.auto-claude` rule —
/// 651 of its 738 files have a canonical twin.
///
/// Each edge is re-pointed to the canonical twin (`File` by `path`, the symbols
/// by `id`, all indexed) with `apoc.merge.relationship`: a note already linked
/// to the twin keeps one edge (dedup), one linked only to the copy moves. The
/// copy edge is deleted in every case, so a branch-only file (no canonical
/// twin) loses its link but the NOTE survives — the explicit, measured choice,
/// not a silent `DETACH`.
const REHOME_WORKTREE_KNOWLEDGE: &str = r#"
MATCH (w)
WHERE (w:File OR w:Function OR w:Struct OR w:Trait)
  AND any(seg IN ['/_wt/', '/.wt/', '/.rlwt/', '/worktrees/'] WHERE
        coalesce(w.path, '') CONTAINS seg OR coalesce(w.file_path, '') CONTAINS seg)
  AND ($scope IS NULL OR w.project_id = $scope)
MATCH (k)-[r]->(w)
WHERE type(r) IN ['LINKED_TO', 'AFFECTS', 'INCLUDES_ENTITY', 'DISCUSSED', 'STEP_IN_PROCESS']
WITH k, r, w, split(coalesce(w.path, w.file_path), '/') AS parts
WITH k, r, w, parts,
     [i IN range(0, size(parts)-1) WHERE parts[i] IN ['_wt', 'worktrees', '.wt', '.rlwt'] | i] AS hits,
     apoc.coll.indexOf(parts, '.auto-claude') AS ac,
     apoc.coll.indexOf(parts, 'tasks') AS tk
WITH k, r, w, parts, hits[0] AS d, ac, tk
WHERE d IS NOT NULL
WITH k, r, w, parts, d, ac, tk
LIMIT $batch
// The number of levels of scaffolding after the worktree directory is NOT
// fixed: `_wt/<branch>/` keeps one, `worktrees/tasks/<name>/` keeps two,
// `.rlwt/` keeps none (the worktree root IS `.rlwt`), and elun nests its
// agent worktrees under `src/.auto-claude/worktrees/tasks/<name>/`. Measured
// 29/10/2026: 14 282 knowledge edges under `.rlwt/src/` and 23 703 under
// `worktrees/tasks/` would have been DETACHED by a fixed two-level strip.
// So: try the shallower strips first, take the first that names a real file.
WITH k, r, w,
     apoc.text.join(parts[0..d] + parts[d+1..], '/') AS c0,
     apoc.text.join(parts[0..d] + parts[d+2..], '/') AS c1,
     apoc.text.join(parts[0..d] + parts[d+3..], '/') AS c2,
     apoc.text.join(parts[0..d] + parts[d+4..], '/') AS c3,
     CASE WHEN ac >= 0 AND ac < d AND tk > ac
          THEN apoc.text.join(parts[0..ac] + parts[tk+2..], '/') ELSE null END AS ca,
     CASE WHEN w:File THEN '' ELSE substring(coalesce(w.id, ''), size(coalesce(w.file_path, ''))) END AS rest
WITH k, r, w, rest,
     CASE
       WHEN EXISTS { MATCH (:File {path: c0}) } THEN c0
       WHEN EXISTS { MATCH (:File {path: c1}) } THEN c1
       WHEN EXISTS { MATCH (:File {path: c2}) } THEN c2
       WHEN EXISTS { MATCH (:File {path: c3}) } THEN c3
       WHEN ca IS NOT NULL AND EXISTS { MATCH (:File {path: ca}) } THEN ca
       ELSE null END AS canon
OPTIONAL MATCH (cf:File {path: canon})
OPTIONAL MATCH (cFn:Function {id: canon + rest})
OPTIONAL MATCH (cSt:Struct {id: canon + rest})
OPTIONAL MATCH (cTr:Trait {id: canon + rest})
WITH k, r, coalesce(cf, cFn, cSt, cTr) AS c
CALL apoc.do.when(c IS NOT NULL,
  'CALL apoc.merge.relationship($k, $t, {}, $p, $c) YIELD rel RETURN 1',
  'RETURN 0',
  {k: k, t: type(r), p: properties(r), c: c}) YIELD value
DELETE r
RETURN count(*) AS processed
"#;

/// Delete the code graph of git WORKTREES.
///
/// A worktree is a checkout of the SAME repository on another branch, so every
/// file and symbol under `_wt/`, `.wt/`, `.rlwt/` or `worktrees/` is a copy of
/// code the project's own tree already carries. Indexing them multiplied the
/// graph by the number of branches: measured on 29/10/2026, **1 424 108** of
/// 4 665 569 nodes stood under those segments — 511 678 `Function` and 356 834
/// `Impl`, five times more `Impl` in the copies than in the real source.
///
/// `DETACH DELETE` takes their edges with them. It runs AFTER
/// [`REHOME_WORKTREE_KNOWLEDGE`], which already moved the knowledge onto the
/// canonical files; what the purge drops is therefore the structural edges
/// between copies plus the branch-only links (no canonical twin). No project
/// has its `root_path` inside a worktree (verified), so nothing legitimate is
/// lost.
///
/// This only cleans what was indexed BEFORE the sync filter learned the rule
/// (`IGNORED_PATH_SEGMENTS`, `crate::orchestrator`), which keeps them from
/// coming back.
const PURGE_WORKTREE_NODES: &str = r#"
MATCH (n)
WHERE ($scope IS NULL OR n.project_id = $scope)
  AND (n:File OR n:Function OR n:Struct OR n:Impl OR n:Enum OR n:Trait OR n:Import)
  AND any(seg IN ['/_wt/', '/.wt/', '/.rlwt/', '/worktrees/'] WHERE
        coalesce(n.path, '') CONTAINS seg OR coalesce(n.file_path, '') CONTAINS seg)
WITH n LIMIT $batch
WITH collect(n) AS doomed
FOREACH (x IN doomed | DETACH DELETE x)
RETURN size(doomed) AS processed
"#;

/// Outcome of one migration on this start.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MigrationOutcome {
    pub id: String,
    /// Already completed on a previous start.
    pub skipped: bool,
    pub processed: i64,
    pub batches: u32,
    pub completed: bool,
    pub error: Option<String>,
}

impl Neo4jClient {
    /// The ordered list of data migrations.
    pub fn data_migrations() -> Vec<DataMigration> {
        vec![
            DataMigration {
                id: "2026-09-fold-legacy-alerts",
                description: "Fold append-only legacy alerts into one node per condition",
                batch: FOLD_LEGACY_ALERTS.to_string(),
                finalize: Some(RECOMPUTE_ALERT_PRIORITY),
            },
            DataMigration {
                id: "2026-09-archive-duplicate-skills",
                description: "Archive duplicate skills created by the truncated evolution snapshot",
                batch: archive_duplicate_skills_query(),
                finalize: None,
            },
            DataMigration {
                id: "2026-09-restore-energy-decay-victims",
                description: "Restore notes archived by the compounding energy-decay bug",
                batch: RESTORE_ENERGY_DECAY_VICTIMS.to_string(),
                finalize: None,
            },
            DataMigration {
                id: "2026-09-rebase-active-note-energy",
                description: "Raise over-decayed energy of active notes to the intended value",
                batch: REBASE_ACTIVE_NOTE_ENERGY.to_string(),
                finalize: None,
            },
            DataMigration {
                id: "2026-09-purge-empty-archived-skills",
                description: "Delete archived skills with no member and no reference",
                batch: purge_archived_empty_skills_query(),
                finalize: None,
            },
            DataMigration {
                id: "2026-10-rehome-worktree-knowledge",
                description:
                    "Re-point knowledge that links a worktree copy onto the canonical file",
                batch: REHOME_WORKTREE_KNOWLEDGE.to_string(),
                finalize: None,
            },
            DataMigration {
                id: "2026-10-purge-worktree-nodes",
                description:
                    "Delete the code graph of git worktrees (copies of the same repository)",
                batch: PURGE_WORKTREE_NODES.to_string(),
                finalize: None,
            },
        ]
    }

    /// Run every pending data migration, in order. Never fails: errors are
    /// reported per migration and the next start retries.
    pub async fn run_data_migrations(&self) -> Vec<MigrationOutcome> {
        let mut outcomes = Vec::new();
        for migration in Self::data_migrations() {
            let outcome = match self.migration_completed(migration.id).await {
                Ok(true) => MigrationOutcome {
                    id: migration.id.to_string(),
                    skipped: true,
                    completed: true,
                    ..Default::default()
                },
                Ok(false) => {
                    info!(
                        id = migration.id,
                        "Data migration: {}", migration.description
                    );
                    let outcome = self.run_migration_batches(&migration, None).await;
                    if outcome.completed {
                        if let Err(e) = self.mark_migration_completed(&outcome).await {
                            warn!(id = migration.id, error = %e, "Data migration: failed to record completion");
                        }
                    }
                    outcome
                }
                Err(e) => MigrationOutcome {
                    id: migration.id.to_string(),
                    error: Some(format!("{e:#}")),
                    ..Default::default()
                },
            };
            if let Some(ref e) = outcome.error {
                // Later migrations may depend on earlier ones (skills are
                // purged after dedup): stop here, retry on next start.
                warn!(id = %outcome.id, error = %e, "Data migration failed, will retry on next start");
                outcomes.push(outcome);
                break;
            }
            outcomes.push(outcome);
        }
        outcomes
    }

    /// Run one migration by id, restricted to a project (`scope`), without
    /// touching completion markers. For tests and targeted repairs.
    pub async fn run_data_migration_scoped(
        &self,
        id: &str,
        scope: Uuid,
    ) -> Result<MigrationOutcome> {
        let migration = Self::data_migrations()
            .into_iter()
            .find(|m| m.id == id)
            .with_context(|| format!("unknown data migration {id}"))?;
        Ok(self
            .run_migration_batches(&migration, Some(scope.to_string()))
            .await)
    }

    /// Delete a project's archived skills that nothing references anymore and
    /// that were archived before `archived_before`. Returns how many were
    /// deleted. Skill evolution archives orphaned skills on every pass; this
    /// keeps them from piling up forever.
    pub async fn purge_archived_empty_skills(
        &self,
        project_id: Uuid,
        archived_before: chrono::DateTime<chrono::Utc>,
    ) -> Result<i64> {
        let q = purge_archived_empty_skills_query();
        let mut total = 0;
        for _ in 0..MAX_BATCHES {
            let processed = self
                .batch_processed(
                    query(&q)
                        .param("scope", project_id.to_string())
                        .param("batch", BATCH)
                        .param("before", archived_before.to_rfc3339()),
                )
                .await?;
            total += processed;
            if processed == 0 {
                break;
            }
        }
        Ok(total)
    }

    async fn run_migration_batches(
        &self,
        migration: &DataMigration,
        scope: Option<String>,
    ) -> MigrationOutcome {
        let started = Instant::now();
        let mut outcome = MigrationOutcome {
            id: migration.id.to_string(),
            ..Default::default()
        };
        let mut failures = 0;
        while outcome.batches < MAX_BATCHES {
            let q = query(&migration.batch)
                .param("scope", scope.clone())
                .param("batch", BATCH)
                .param("before", Option::<String>::None);
            match self.batch_processed(q).await {
                Ok(0) => {
                    outcome.completed = true;
                    break;
                }
                Ok(n) => {
                    failures = 0;
                    outcome.batches += 1;
                    outcome.processed += n;
                    info!(
                        id = migration.id,
                        batch = outcome.batches,
                        processed = outcome.processed,
                        "Data migration progress"
                    );
                }
                Err(e) => {
                    failures += 1;
                    if failures >= MAX_RETRIES {
                        outcome.error = Some(format!("{e:#}"));
                        return outcome;
                    }
                    warn!(id = migration.id, attempt = failures, error = %e, "Data migration batch failed, retrying");
                }
            }
        }
        if !outcome.completed {
            outcome.error = Some(format!("did not converge within {MAX_BATCHES} batches"));
            return outcome;
        }
        if let Some(finalize) = migration.finalize {
            if let Err(e) = self
                .graph
                .run(query(finalize).param("scope", scope.clone()))
                .await
            {
                outcome.completed = false;
                outcome.error = Some(format!("finalize: {e:#}"));
                return outcome;
            }
        }
        info!(
            id = migration.id,
            processed = outcome.processed,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "Data migration complete"
        );
        outcome
    }

    async fn batch_processed(&self, q: neo4rs::Query) -> Result<i64> {
        let mut result = self.graph.execute(q).await?;
        let row = result
            .next()
            .await?
            .context("migration batch returned no row")?;
        Ok(row.get::<i64>("processed").unwrap_or(0))
    }

    async fn migration_completed(&self, id: &str) -> Result<bool> {
        let mut result = self
            .graph
            .execute(
                query(
                    "MATCH (m:DataMigration {id: $id}) RETURN m.completed_at IS NOT NULL AS done",
                )
                .param("id", id),
            )
            .await?;
        Ok(match result.next().await? {
            Some(row) => row.get::<bool>("done").unwrap_or(false),
            None => false,
        })
    }

    async fn mark_migration_completed(&self, outcome: &MigrationOutcome) -> Result<()> {
        self.graph
            .run(
                query(
                    "MERGE (m:DataMigration {id: $id})
                     SET m.completed_at = datetime(), m.processed = $processed",
                )
                .param("id", outcome.id.clone())
                .param("processed", outcome.processed),
            )
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    #[test]
    fn test_migrations_are_ordered_and_unique() {
        let ids: Vec<&str> = Neo4jClient::data_migrations()
            .iter()
            .map(|m| m.id)
            .collect();
        // Dedup must run before the purge: it is what empties the duplicates.
        assert_eq!(
            ids,
            vec![
                "2026-09-fold-legacy-alerts",
                "2026-09-archive-duplicate-skills",
                "2026-09-restore-energy-decay-victims",
                "2026-09-rebase-active-note-energy",
                "2026-09-purge-empty-archived-skills",
                "2026-10-rehome-worktree-knowledge",
                "2026-10-purge-worktree-nodes",
            ]
        );
    }

    /// The worktree purge deletes COPIES of the project's code, never the code
    /// itself: the predicate is anchored on the directory segments the sync
    /// filter excludes, and it must name all four of them.
    #[test]
    fn test_worktree_purge_only_matches_worktree_segments() {
        let m = Neo4jClient::data_migrations()
            .into_iter()
            .find(|m| m.id == "2026-10-purge-worktree-nodes")
            .expect("the worktree purge is registered");
        for seg in ["/_wt/", "/.wt/", "/.rlwt/", "/worktrees/"] {
            assert!(m.batch.contains(seg), "the purge must name {seg}");
        }
        // It reads the two properties that carry a path, and nothing else.
        assert!(m.batch.contains("n.path"));
        assert!(m.batch.contains("n.file_path"));
        // A copy is only deleted outright: DETACH takes its edges with it.
        assert!(m.batch.contains("DETACH DELETE"));
    }

    #[test]
    fn test_every_batch_query_is_scoped_bounded_and_reports_progress() {
        for m in Neo4jClient::data_migrations() {
            assert!(m.batch.contains("$scope"), "{} must honour $scope", m.id);
            assert!(m.batch.contains("$batch"), "{} must be bounded", m.id);
            assert!(
                m.batch.contains("AS processed"),
                "{} must report progress",
                m.id
            );
        }
    }

    #[test]
    fn test_skill_queries_never_touch_referenced_skills() {
        let dedup = archive_duplicate_skills_query();
        let purge = purge_archived_empty_skills_query();
        for q in [&dedup, &purge] {
            assert!(q.contains(SKILL_REFERENCES));
            assert!(
                q.contains("pr.skill_id = s.id"),
                "protocols reference skills by property"
            );
        }
        assert!(dedup.contains("WITH r WHERE r.refs = 0"));
    }
}
