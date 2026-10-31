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

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use neo4rs::{query, BoltType, Row};
use serde::Serialize;
use tracing::{info, warn};
use uuid::Uuid;

use super::client::Neo4jClient;
use super::models::SymbolOwner;
use super::symbol_key::{
    enclosing_owner, stable_external_trait_key, stable_impl_key, stable_import_key,
    stable_symbol_key_from_graph, stable_type_key,
};

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
    /// Batch query. Params: `$scope` (string or null), `$batch`.
    ///
    /// Without `compute`, it does the work itself and returns one row with a
    /// `processed` integer column. With `compute`, it only SELECTS up to
    /// `$batch` rows still needing the work; `compute.items` turns each row
    /// into the properties to write and `compute.write` writes them.
    pub batch: String,
    /// The Rust half of a computed migration — see [`ComputedBatch`].
    pub compute: Option<ComputedBatch>,
    /// Optional query run once after the last batch. Param: `$scope`.
    pub finalize: Option<&'static str>,
}

/// A migration whose new values are computed in Rust, by the very functions
/// the write paths use, instead of re-implemented in Cypher.
///
/// The stable-key backfills are the reason this exists: four Cypher constants
/// re-implementing four Rust functions of `symbol_key.rs` had already drifted
/// (`Some("  ")` was no trait for Rust and a trait called `  ` for Cypher).
/// With one grammar there is nothing to keep in step.
pub struct ComputedBatch {
    /// One selected row → the properties to write (`eid` + the new values).
    /// An error here is a bug in the SELECT (it returned a row the grammar
    /// cannot key) and fails the migration loudly instead of re-selecting the
    /// same page until `MAX_BATCHES`.
    pub items: fn(&Row) -> Result<HashMap<String, BoltType>>,
    /// `UNWIND $items AS item MATCH (n) WHERE elementId(n) = item.eid SET ...
    /// RETURN count(n) AS processed[, ... AS deferred]`. Must write the property
    /// the SELECT tests for `IS NULL`, on every item, or the migration cannot
    /// converge.
    pub write: &'static str,
}

/// What one batch reports back.
#[derive(Debug, Default, Clone, Copy)]
struct Progress {
    processed: i64,
    deferred: i64,
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
    /// Nodes the migration deliberately left without a value because the graph
    /// cannot settle it (`key_deferred`, see the stable-key backfill).
    pub deferred: i64,
    pub batches: u32,
    pub completed: bool,
    pub error: Option<String>,
}

/// Backfill the stable identity of functions (see `symbol_key.rs`).
///
/// The legacy `id` embeds the line number, so the same function accumulated
/// one node per edit. The stable key (`file::[Owner::]name[(types)]`) is
/// written as a shadow property first — reversible, and the prerequisite of
/// the deduplication migration.
///
/// The owner of a legacy node is not an edge (`DEFINED_IN`: 0 relations on
/// 31/10/2026) — it is deduced by line inclusion in the `Impl` and `Struct`
/// nodes of the same file, the rule `parser::assign_owners` applies at parse
/// time (`symbol_key::enclosing_owner`). Measured without this qualifier:
/// 7 168 (file, signature) groups mixed functions of different impls —
/// 19 885 keys needed where one would have been set.
///
/// **A guess is refused when it could be wrong.** `Impl`/`Struct` ids carry
/// no line, so their ranges are those of the LAST sync; a `Function` node of
/// an older version (never deleted) may now sit inside another block. When a
/// file holds several `Function` nodes of one name at different lines, the
/// graph cannot tell the current one from the stale ones, so none of them
/// gets a key: `key_deferred = true`, and the next sync writes the current
/// version with its true owner (the write path clears the flag); T2 deletes
/// the still-flagged ones without folding them into anything. Measured on
/// 31/10/2026: 40 584 (file, name) groups hold several line versions —
/// 223 492 of 504 571 Function nodes (44 %) are deferred, 281 079 keyed.
/// The same deferral applies when a block of the file has no lines (it
/// could own the function and nobody can tell; measured: 0 of 59 994 Impl,
/// 0 of 50 407 Struct).
///
/// Scope: `Impl` and `Import` nodes carry no `project_id` (0 of 59 994 and
/// 0 of 289 491 on 31/10/2026 — only the Function/Struct/Trait/Enum batch
/// paths denormalised it), so `n.project_id = $scope` alone was a no-op for
/// them: a scoped run also reaches a node through its `File`.
///
/// Paging: `stable_key IS NULL` is not indexable, so a batch that restarted
/// from `MATCH (n:Function)` scanned every node again (245 scans of 490 k
/// nodes). Each batch instead seeks the `file_path` index from the cursor
/// `$after` (the last file_path of the previous batch, `>=` because a file
/// can hold more than one batch) and stops at `$batch` rows.
///
/// Convergence: the SELECT only returns nodes whose `file_path` and `name`
/// exist and that carry neither a key nor the flag; every selected node
/// leaves the selection once written (key or flag). A row the grammar cannot
/// decode is an error, not a silent unqualified key.
const SELECT_FUNCTIONS_WITHOUT_STABLE_KEY: &str = r#"
MATCH (n:Function)
WHERE n.file_path >= $after
  AND n.stable_key IS NULL AND n.key_deferred IS NULL
  AND n.file_path IS NOT NULL AND n.name IS NOT NULL
  AND ($scope IS NULL OR n.project_id = $scope
       OR EXISTS { (n)<-[:CONTAINS|HAS_IMPORT]-(:File {project_id: $scope}) })
WITH n ORDER BY n.file_path LIMIT $batch
OPTIONAL MATCH (i:Impl)
WHERE i.file_path = n.file_path
  AND i.line_start <= n.line_start AND n.line_end <= i.line_end
WITH n, collect(i) AS impls
OPTIONAL MATCH (s:Struct)
WHERE s.file_path = n.file_path
  AND s.line_start <= n.line_start AND n.line_end <= s.line_end
WITH n, impls, collect(s) AS structs
WITH n, impls + structs AS blocks
RETURN elementId(n) AS eid, n.file_path AS cursor,
       n.file_path AS file_path, n.name AS name, n.params AS params,
       n.line_start AS line_start, n.line_end AS line_end,
       [b IN blocks | {start: b.line_start, end: b.line_end,
                       type_name: coalesce(b.for_type, b.name, ''),
                       trait_name: CASE WHEN b:Impl THEN coalesce(b.trait_name, '') ELSE '' END}] AS blocks,
       EXISTS { MATCH (m:Function)
                WHERE m.file_path = n.file_path AND m.name = n.name
                  AND m.line_start <> n.line_start } AS has_sibling,
       EXISTS { MATCH (b:Impl|Struct)
                WHERE b.file_path = n.file_path
                  AND (b.line_start IS NULL OR b.line_end IS NULL) } AS broken_blocks
"#;

const WRITE_FUNCTION_STABLE_KEY: &str = r#"
UNWIND $items AS item
MATCH (n:Function) WHERE elementId(n) = item.eid
SET n.stable_key = item.stable_key,
    n.owner_type = item.owner_type,
    n.owner_trait = item.owner_trait,
    n.key_deferred = item.key_deferred
RETURN count(n) AS processed,
       sum(CASE WHEN item.key_deferred = true THEN 1 ELSE 0 END) AS deferred
"#;

/// A block of the function's file that encloses it, as the SELECT returns it.
/// Strictly typed on purpose: a block with no lines cannot be here (the
/// SELECT's inclusion test excludes it and `broken_blocks` reports it), so a
/// decoding failure is a real defect and fails the batch loudly.
#[derive(Debug, serde::Deserialize)]
struct OwnerBlock {
    start: i64,
    end: i64,
    type_name: String,
    trait_name: String,
}

fn function_items(row: &Row) -> Result<HashMap<String, BoltType>> {
    let eid: String = row.get("eid")?;
    let mut m = HashMap::new();
    m.insert("eid".into(), eid.into());

    let has_sibling: bool = row.get("has_sibling")?;
    let broken_blocks: bool = row.get("broken_blocks")?;
    if has_sibling || broken_blocks {
        m.insert("stable_key".into(), Option::<String>::None.into());
        m.insert("owner_type".into(), Option::<String>::None.into());
        m.insert("owner_trait".into(), Option::<String>::None.into());
        m.insert("key_deferred".into(), true.into());
        return Ok(m);
    }

    let file_path: String = row.get("file_path")?;
    let name: String = row.get("name")?;
    let params: Option<String> = row.get("params").ok();
    let line_start: i64 = row.get("line_start")?;
    let line_end: i64 = row.get("line_end")?;
    let blocks: Vec<OwnerBlock> = row.get("blocks").context("decoding the enclosing blocks")?;
    let candidates = blocks
        .into_iter()
        .filter(|b| !b.type_name.is_empty())
        .map(|b| {
            (
                b.start as u32,
                b.end as u32,
                SymbolOwner::new(b.type_name, Some(b.trait_name)),
            )
        });
    let owner = enclosing_owner(line_start as u32, line_end as u32, candidates);
    let key = stable_symbol_key_from_graph(&file_path, owner.as_ref(), &name, params.as_deref());
    m.insert("stable_key".into(), key.into());
    m.insert(
        "owner_type".into(),
        owner.as_ref().map(|o| o.type_name.clone()).into(),
    );
    m.insert(
        "owner_trait".into(),
        owner.and_then(|o| o.trait_name).into(),
    );
    m.insert("key_deferred".into(), Option::<bool>::None.into());
    Ok(m)
}

/// Same shadow key for the type declarations (struct, trait, enum).
///
/// Their legacy id was already `<file>:<name>` — no line, so they never
/// accumulated versions. The key is still written: the identity must be one
/// grammar for every declaration, otherwise a reader cannot tell "no stable key"
/// from "this label was forgotten". The placeholder of an external trait
/// (`is_external`, no file) keeps its id as key, see `stable_external_trait_key`.
///
/// `(n:Struct|Trait|Enum)` is a label expression (Neo4j 5): three label scans
/// unioned, where `MATCH (n) WHERE n:Struct OR ...` was a scan of EVERY node
/// of the database per batch.
const SELECT_TYPES_WITHOUT_STABLE_KEY: &str = r#"
MATCH (n:Struct|Trait|Enum)
WHERE n.file_path >= $after
  AND n.stable_key IS NULL
  AND n.file_path IS NOT NULL AND n.name IS NOT NULL
  AND ($scope IS NULL OR n.project_id = $scope
       OR EXISTS { (n)<-[:CONTAINS|HAS_IMPORT]-(:File {project_id: $scope}) })
WITH n ORDER BY n.file_path LIMIT $batch
RETURN elementId(n) AS eid, n.file_path AS cursor, n.file_path AS file_path, n.name AS name,
       coalesce(n.is_external, false) AS is_external, n.full_path AS full_path
"#;

const WRITE_STABLE_KEY: &str = r#"
UNWIND $items AS item
MATCH (n) WHERE elementId(n) = item.eid
SET n.stable_key = item.stable_key
RETURN count(n) AS processed
"#;

fn type_items(row: &Row) -> Result<HashMap<String, BoltType>> {
    let eid: String = row.get("eid")?;
    let file_path: String = row.get("file_path")?;
    let name: String = row.get("name")?;
    let is_external: bool = row.get("is_external").unwrap_or(false);
    let full_path: Option<String> = row.get("full_path").ok();
    let key = if is_external {
        stable_external_trait_key(full_path.as_deref().unwrap_or(&name))
    } else {
        stable_type_key(&file_path, &name)
    };
    let mut m = HashMap::new();
    m.insert("eid".into(), eid.into());
    m.insert("stable_key".into(), key.into());
    Ok(m)
}

/// Same shadow key for `impl` blocks.
///
/// The line leaves the key: moving an impl must not create a second impl of the
/// same type. `trait_name` is stored as an empty string when there is no trait
/// (7 284 of 59 994 on 31/10/2026, none NULL) — `stable_impl_key` treats a
/// blank name as no trait, the same way for this backfill and for the write path.
const SELECT_IMPLS_WITHOUT_STABLE_KEY: &str = r#"
MATCH (n:Impl)
WHERE n.file_path >= $after
  AND n.stable_key IS NULL
  AND n.file_path IS NOT NULL AND n.for_type IS NOT NULL
  AND ($scope IS NULL OR n.project_id = $scope
       OR EXISTS { (n)<-[:CONTAINS|HAS_IMPORT]-(:File {project_id: $scope}) })
WITH n ORDER BY n.file_path LIMIT $batch
RETURN elementId(n) AS eid, n.file_path AS cursor, n.file_path AS file_path,
       n.for_type AS for_type, n.trait_name AS trait_name
"#;

fn impl_items(row: &Row) -> Result<HashMap<String, BoltType>> {
    let eid: String = row.get("eid")?;
    let file_path: String = row.get("file_path")?;
    let for_type: String = row.get("for_type")?;
    let trait_name: Option<String> = row.get("trait_name").ok();
    let mut m = HashMap::new();
    m.insert("eid".into(), eid.into());
    m.insert(
        "stable_key".into(),
        stable_impl_key(&file_path, &for_type, trait_name.as_deref()).into(),
    );
    Ok(m)
}

/// Same shadow key for imports — the line leaves it too (`stable_import_key`
/// has the measure: 38 555 of 289 491 nodes exist only because a line moved).
///
/// A legacy node gets occurrence 0: the graph cannot tell the versions of one
/// import apart from a genuine duplicate in the current file, so they all get
/// the bare key and T2 folds them into one. From this version on, the parser
/// ranks true duplicates (`parser::assign_import_occurrences`).
const SELECT_IMPORTS_WITHOUT_STABLE_KEY: &str = r#"
MATCH (n:Import)
WHERE n.file_path >= $after
  AND n.stable_key IS NULL
  AND n.file_path IS NOT NULL AND n.path IS NOT NULL
  AND ($scope IS NULL OR n.project_id = $scope
       OR EXISTS { (n)<-[:CONTAINS|HAS_IMPORT]-(:File {project_id: $scope}) })
WITH n ORDER BY n.file_path LIMIT $batch
RETURN elementId(n) AS eid, n.file_path AS cursor, n.file_path AS file_path, n.path AS path
"#;

const WRITE_IMPORT_STABLE_KEY: &str = r#"
UNWIND $items AS item
MATCH (n:Import) WHERE elementId(n) = item.eid
SET n.stable_key = item.stable_key, n.occurrence = 0
RETURN count(n) AS processed
"#;

fn import_items(row: &Row) -> Result<HashMap<String, BoltType>> {
    let eid: String = row.get("eid")?;
    let file_path: String = row.get("file_path")?;
    let path: String = row.get("path")?;
    let mut m = HashMap::new();
    m.insert("eid".into(), eid.into());
    m.insert(
        "stable_key".into(),
        stable_import_key(&file_path, &path, 0).into(),
    );
    Ok(m)
}

/// Backfill `File.uid` — the identity of a file, independent of its mount point.
///
/// The stored path is canonical (the sync resolves links) while
/// `Project.root_path` is what the user declared, often a symlink: the relative
/// path therefore comes from `canonical_path`, filled in Rust just before this
/// run (`ensure_project_canonical_paths`). Without it, `strip_prefix` has
/// nothing to strip and every file of such a project would keep its absolute
/// path in the identity — the very defect this series removes.
const BACKFILL_FILE_UID: &str = r#"
MATCH (p:Project)
WHERE p.canonical_path IS NOT NULL AND p.canonical_path <> ''
  AND ($scope IS NULL OR p.id = $scope)
MATCH (f:File {project_id: p.id})
WHERE f.uid IS NULL
WITH f, p LIMIT $batch
WITH f, p, CASE WHEN f.path STARTS WITH p.canonical_path + '/'
                THEN substring(f.path, size(p.canonical_path) + 1)
                ELSE f.path END AS rel
SET f.uid = p.id + ':' + rel
RETURN count(f) AS processed
"#;

/// Backfill `File.uid` for the files that belong to no project.
///
/// The owner is `-`, exactly like `neo4j::file_uid(None, path)`: one grammar,
/// two call sites, and a file that later joins a project gets a new identity
/// rather than a silent reuse of the project-less one.
const BACKFILL_FILE_UID_ORPHANS: &str = r#"
MATCH (f:File)
WHERE f.uid IS NULL
  AND (f.project_id IS NULL OR f.project_id = '')
  AND ($scope IS NULL OR f.project_id = $scope)
WITH f LIMIT $batch
SET f.uid = '-:' + f.path
RETURN count(f) AS processed
"#;

impl Neo4jClient {
    /// The ordered list of data migrations.
    pub fn data_migrations() -> Vec<DataMigration> {
        vec![
            DataMigration {
                id: "2026-09-fold-legacy-alerts",
                description: "Fold append-only legacy alerts into one node per condition",
                batch: FOLD_LEGACY_ALERTS.to_string(),
                compute: None,
                finalize: Some(RECOMPUTE_ALERT_PRIORITY),
            },
            DataMigration {
                id: "2026-09-archive-duplicate-skills",
                description: "Archive duplicate skills created by the truncated evolution snapshot",
                batch: archive_duplicate_skills_query(),
                compute: None,
                finalize: None,
            },
            DataMigration {
                id: "2026-09-restore-energy-decay-victims",
                description: "Restore notes archived by the compounding energy-decay bug",
                batch: RESTORE_ENERGY_DECAY_VICTIMS.to_string(),
                compute: None,
                finalize: None,
            },
            DataMigration {
                id: "2026-09-rebase-active-note-energy",
                description: "Raise over-decayed energy of active notes to the intended value",
                batch: REBASE_ACTIVE_NOTE_ENERGY.to_string(),
                compute: None,
                finalize: None,
            },
            DataMigration {
                id: "2026-09-purge-empty-archived-skills",
                description: "Delete archived skills with no member and no reference",
                batch: purge_archived_empty_skills_query(),
                compute: None,
                finalize: None,
            },
            DataMigration {
                id: "2026-10-rehome-worktree-knowledge",
                description:
                    "Re-point knowledge that links a worktree copy onto the canonical file",
                batch: REHOME_WORKTREE_KNOWLEDGE.to_string(),
                compute: None,
                finalize: None,
            },
            DataMigration {
                id: "2026-10-purge-worktree-nodes",
                description:
                    "Delete the code graph of git worktrees (copies of the same repository)",
                batch: PURGE_WORKTREE_NODES.to_string(),
                compute: None,
                finalize: None,
            },
            DataMigration {
                id: "2026-10-backfill-stable-key",
                description:
                    "Write the stable identity of functions as a shadow property (owner-qualified, line removed)",
                batch: SELECT_FUNCTIONS_WITHOUT_STABLE_KEY.to_string(),
                compute: Some(ComputedBatch {
                    items: function_items,
                    write: WRITE_FUNCTION_STABLE_KEY,
                }),
                finalize: None,
            },
            DataMigration {
                id: "2026-10-backfill-stable-key-types",
                description:
                    "Write the stable identity of structs, traits and enums as a shadow property",
                batch: SELECT_TYPES_WITHOUT_STABLE_KEY.to_string(),
                compute: Some(ComputedBatch {
                    items: type_items,
                    write: WRITE_STABLE_KEY,
                }),
                finalize: None,
            },
            DataMigration {
                id: "2026-10-backfill-stable-key-impls",
                description:
                    "Write the stable identity of impl blocks as a shadow property (line removed)",
                batch: SELECT_IMPLS_WITHOUT_STABLE_KEY.to_string(),
                compute: Some(ComputedBatch {
                    items: impl_items,
                    write: WRITE_STABLE_KEY,
                }),
                finalize: None,
            },
            DataMigration {
                id: "2026-10-backfill-stable-key-imports",
                description:
                    "Write the stable identity of imports as a shadow property (line removed)",
                batch: SELECT_IMPORTS_WITHOUT_STABLE_KEY.to_string(),
                compute: Some(ComputedBatch {
                    items: import_items,
                    write: WRITE_IMPORT_STABLE_KEY,
                }),
                finalize: None,
            },
            DataMigration {
                id: "2026-10-backfill-file-uid",
                description:
                    "Give every file a mount-independent identity (project + relative path)",
                batch: BACKFILL_FILE_UID.to_string(),
                finalize: None,
            },
            DataMigration {
                id: "2026-10-backfill-file-uid-orphans",
                description: "Same identity for the files that belong to no project",
                batch: BACKFILL_FILE_UID_ORPHANS.to_string(),
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
                .await?
                .processed;
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
        // Cursor of a computed migration: the last `file_path` of the previous
        // batch (`""` sorts before every path). A plain migration ignores it.
        let mut after = String::new();
        while outcome.batches < MAX_BATCHES {
            let q = query(&migration.batch)
                .param("scope", scope.clone())
                .param("batch", BATCH)
                .param("after", after.clone())
                .param("before", Option::<String>::None);
            let step = match &migration.compute {
                None => self.batch_processed(q).await,
                Some(compute) => match self.computed_batch(q, compute).await {
                    Ok((progress, cursor)) => {
                        if let Some(c) = cursor {
                            after = c;
                        }
                        Ok(progress)
                    }
                    Err(e) => Err(e),
                },
            };
            match step {
                Ok(Progress { processed: 0, .. }) => {
                    outcome.completed = true;
                    break;
                }
                Ok(Progress {
                    processed,
                    deferred,
                }) => {
                    failures = 0;
                    outcome.batches += 1;
                    outcome.processed += processed;
                    outcome.deferred += deferred;
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

    /// One batch of a computed migration: select the rows, compute their new
    /// values in Rust, write them. Returns how many nodes were written.
    ///
    /// The write must reach every selected node: a shortfall means the SELECT
    /// and the WRITE disagree on what a node is, and re-selecting the same
    /// page forever is exactly the non-convergence this guards against — so
    /// it is an error, retried `MAX_RETRIES` times, then reported.
    async fn computed_batch(
        &self,
        select: neo4rs::Query,
        compute: &ComputedBatch,
    ) -> Result<(Progress, Option<String>)> {
        let mut result = self.graph.execute(select).await?;
        let mut items = Vec::new();
        let mut cursor: Option<String> = None;
        while let Some(row) = result.next().await? {
            items.push((compute.items)(&row).context("computing the new value of a selected row")?);
            if let Ok(c) = row.get::<String>("cursor") {
                if cursor.as_deref().is_none_or(|prev| prev < c.as_str()) {
                    cursor = Some(c);
                }
            }
        }
        if items.is_empty() {
            return Ok((Progress::default(), None));
        }
        let selected = items.len() as i64;
        let written = self
            .batch_processed(query(compute.write).param("items", items))
            .await?;
        if written.processed != selected {
            bail!(
                "the write reached {} of the {selected} selected nodes",
                written.processed
            );
        }
        Ok((written, cursor))
    }

    async fn batch_processed(&self, q: neo4rs::Query) -> Result<Progress> {
        let mut result = self.graph.execute(q).await?;
        let row = result
            .next()
            .await?
            .context("migration batch returned no row")?;
        Ok(Progress {
            processed: row.get::<i64>("processed").unwrap_or(0),
            deferred: row.get::<i64>("deferred").unwrap_or(0),
        })
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
                "2026-10-backfill-stable-key",
                "2026-10-backfill-stable-key-types",
                "2026-10-backfill-stable-key-impls",
                "2026-10-backfill-stable-key-imports",
                "2026-10-backfill-file-uid",
                "2026-10-backfill-file-uid-orphans",
            ]
        );
    }

    /// The identity is one grammar for every declaration: a label left without a
    /// backfill would be indistinguishable from a label with nothing to write.
    /// The grammar itself lives in `symbol_key.rs` (one implementation for the
    /// write paths and the backfill); what this pins is the SHAPE of the four
    /// backfills: computed in Rust, one per label family, convergent.
    #[test]
    fn test_stable_key_backfills_cover_every_label() {
        let migrations = Neo4jClient::data_migrations();
        let backfills: Vec<&DataMigration> = migrations
            .iter()
            .filter(|m| m.id.starts_with("2026-10-backfill-stable-key"))
            .collect();
        assert_eq!(
            backfills.len(),
            4,
            "one backfill per identity shape — functions, types, impls, imports"
        );
        for m in &backfills {
            let compute = m
                .compute
                .as_ref()
                .unwrap_or_else(|| panic!("{} computes its key in Rust, not in Cypher", m.id));
            assert!(
                compute.write.contains("SET n.stable_key = item.stable_key"),
                "{} must write the shadow key",
                m.id
            );
            assert!(
                m.batch.contains("n.stable_key IS NULL"),
                "{} must be idempotent",
                m.id
            );
            // Review 2, point 6: a batch seeks the file_path index from a cursor
            // instead of scanning the label again.
            assert!(
                m.batch.contains("n.file_path >= $after")
                    && m.batch.contains("ORDER BY n.file_path LIMIT $batch")
                    && m.batch.contains("n.file_path AS cursor"),
                "{} must page by file_path",
                m.id
            );
            // Finding 3 of the review (31/10/2026): a selected node the grammar
            // cannot key would be re-selected until MAX_BATCHES. The SELECT must
            // exclude every node lacking a property the grammar reads.
            assert!(
                m.batch.contains("n.file_path IS NOT NULL"),
                "{} must not select a node it cannot key",
                m.id
            );
            assert!(
                !m.batch.contains("line_start") || m.id == "2026-10-backfill-stable-key",
                "{}: the line is not part of any identity",
                m.id
            );
        }

        let by_id = |id: &str| -> &DataMigration {
            backfills
                .iter()
                .find(|m| m.id == id)
                .unwrap_or_else(|| panic!("{id} is registered"))
        };

        let functions = by_id("2026-10-backfill-stable-key");
        assert!(functions.batch.contains("(n:Function)"));
        assert!(functions.batch.contains("n.name IS NOT NULL"));
        // The owner of a legacy function is deduced by line inclusion in the
        // Impl and Struct nodes of ITS file — the rule of `enclosing_owner`.
        assert!(functions.batch.contains("(i:Impl)"));
        assert!(functions.batch.contains("(s:Struct)"));
        assert!(functions.batch.contains("i.file_path = n.file_path"));
        assert!(functions.batch.contains("s.file_path = n.file_path"));
        let write = functions.compute.as_ref().unwrap().write;
        assert!(write.contains("n.owner_type = item.owner_type"));
        assert!(write.contains("n.owner_trait = item.owner_trait"));
        // Review 2, point 2: a node whose owner would be a guess (another
        // Function of the same name at another line of the file, or a block of
        // the file without lines) is flagged, never keyed — and the flag keeps
        // it out of the next selection, so the migration still converges.
        assert!(functions.batch.contains("n.key_deferred IS NULL"));
        assert!(functions.batch.contains("AS has_sibling"));
        assert!(functions.batch.contains("AS broken_blocks"));
        assert!(write.contains("n.key_deferred = item.key_deferred"));
        assert!(write.contains("AS deferred"));
        // Point 7: the blocks travel as one list of maps, not four parallel lists.
        assert!(functions.batch.contains("}] AS blocks"));
        assert!(!functions.batch.contains("block_starts"));

        let types = by_id("2026-10-backfill-stable-key-types");
        // A label expression, not `MATCH (n) WHERE n:Struct OR ...` (an
        // AllNodesScan per batch — finding 6 of the review).
        assert!(types.batch.contains("(n:Struct|Trait|Enum)"));
        assert!(!types.batch.contains("MATCH (n)\n"));
        assert!(types.batch.contains("n.name IS NOT NULL"));
        assert!(types.batch.contains("is_external"));

        let impls = by_id("2026-10-backfill-stable-key-impls");
        assert!(impls.batch.contains("(n:Impl)"));
        assert!(impls.batch.contains("n.for_type IS NOT NULL"));
        assert!(impls.batch.contains("trait_name"));

        let imports = by_id("2026-10-backfill-stable-key-imports");
        assert!(imports.batch.contains("(n:Import)"));
        assert!(imports.batch.contains("n.path IS NOT NULL"));
        // The line LEFT the import identity (finding 9): 38 555 of 289 491
        // import nodes existed only because a line moved.
        assert!(!imports.batch.contains("n.line"));
        assert!(imports
            .compute
            .as_ref()
            .unwrap()
            .write
            .contains("n.occurrence = 0"));
    }

    /// `File.uid` n'a de valeur que si la migration qui le remplit lit la même
    /// racine que l'écriture : sinon les fichiers d'un projet monté par un lien
    /// garderaient leur chemin absolu dans l'identité.
    #[test]
    fn test_file_uid_backfills_read_the_canonical_root() {
        let migrations = Neo4jClient::data_migrations();
        let by_id = |id: &str| -> String {
            migrations
                .iter()
                .find(|m| m.id == id)
                .unwrap_or_else(|| panic!("{id} is registered"))
                .batch
                .clone()
        };

        let projects = by_id("2026-10-backfill-file-uid");
        assert!(projects.contains("n.canonical_path") || projects.contains("p.canonical_path"));
        assert!(
            projects.contains("substring("),
            "the relative path is a suffix"
        );
        assert!(projects.contains("f.uid IS NULL"), "idempotent");
        assert!(projects.contains("SET f.uid"));

        let orphans = by_id("2026-10-backfill-file-uid-orphans");
        assert!(
            orphans.contains("'-:'"),
            "an ownerless file is owned by `-`, like file_uid(None, path)"
        );
        assert!(orphans.contains("f.uid IS NULL"));
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
            // A plain migration reports progress itself; a computed one through
            // its write, which must address the selected nodes and nothing else.
            match &m.compute {
                None => assert!(
                    m.batch.contains("AS processed"),
                    "{} must report progress",
                    m.id
                ),
                Some(c) => {
                    assert!(m.batch.contains("elementId(n) AS eid"), "{}", m.id);
                    assert!(m.batch.contains("$after"), "{} must take the cursor", m.id);
                    assert!(c.write.contains("UNWIND $items AS item"), "{}", m.id);
                    assert!(c.write.contains("elementId(n) = item.eid"), "{}", m.id);
                    assert!(c.write.contains("AS processed"), "{}", m.id);
                }
            }
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
