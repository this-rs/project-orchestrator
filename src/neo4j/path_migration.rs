//! Rename a project's code graph from the canonical (alias) path to the
//! **declared** root — T7, lot 3/3. Explicit command, never run at boot.
//!
//! The sync used to resolve links before writing, so a project declared as
//! `~/projects/po/backend` (a symlink) was indexed under
//! `~/.openclaw/workspace/skills/project-orchestrator/…`: 815 `File`, 132 111
//! `Function` and 1 878 notes anchored on paths no user ever typed (measured
//! 31/10/2026). The write boundary now produces the declared form
//! (`file_uid::under_declared_root`); this module brings the existing nodes to
//! the same form, in place.
//!
//! # Why a command and not a `DataMigration`
//!
//! A boot migration starts on any restart, cannot be stopped, and cannot be
//! rehearsed. Renaming 500 000 nodes and deleting 631 worktree copies is a
//! destructive write on the production graph: it is run **by someone**, after a
//! dry-run that prints the measures, with `--apply` — and it can be run again
//! (every pass is idempotent) or stopped between passes.
//!
//! # Rename, never recreate
//!
//! 28 833 `(:Note)-[:LINKED_TO]->(:File)` edges hang on the NODES: a `SET
//! f.path` keeps every one of them, a `CREATE` + `DELETE` would lose them all.
//! No pass here creates a `File`; the static tests enforce it.
//!
//! # The symbols carry the path three times
//!
//! `file_path`, `id` (`<file_path>:<name>[:<line>]`) and `stable_key`
//! (`<file_path>::<rest>`, `symbol_key.rs`) all start with the file path. The
//! three are rewritten by the same prefix replacement — one transform, not a
//! second grammar. Recomputing `stable_key` from its parts, or nulling it and
//! replaying the four `2026-10-backfill-stable-key*` boot migrations, would
//! have introduced a window where the key is wrong or absent while the sync
//! writes; the prefix rewrite has no window and no second implementation.
//!
//! # `File.uid` is never touched
//!
//! It is computed on the path **relative to the root**, which the rename does
//! not change. That it survives untouched is the proof lot 2/3 was designed
//! right.
//!
//! # Worktree copies (`.wl`) are purged, not renamed
//!
//! `…/.wl/<branch>/<rel>` is a copy of `…/<rel>`. `--purge-wl` first re-points
//! the knowledge edges (`LINKED_TO`, `AFFECTS`, …) onto the twin under the
//! declared root — **only when the twin exists**: an edge is deleted only once
//! its replacement is created (the lesson of `2026-10-rehome-worktree-knowledge`)
//! — then `DETACH DELETE`s the copies and their symbols. The purge refuses to
//! start when the measures after the rename are wrong (a note that would lose
//! its last anchor, an anchor pointing at nothing).
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use anyhow::{Context, Result};
use neo4rs::query;
use serde::Serialize;
use uuid::Uuid;

use super::client::Neo4jClient;
use super::data_migrations::{DataMigration, MigrationOutcome};
use super::file_uid::canonical_root;

/// Every label that carries a `file_path` (and an `id` prefixed by it).
pub const SYMBOL_LABELS: &[&str] = &[
    "Function", "Struct", "Trait", "Enum", "Impl", "Import", "Module",
];

/// Segment of the `.wl` worktrees — the copies `--purge-wl` deletes. The other
/// worktree families (`_wt`, `.wt`, `.rlwt`, `worktrees`) were purged at boot
/// by `2026-10-purge-worktree-nodes`; `.wl` was not in its list.
pub const WL_SEGMENT: &str = "/.wl/";

/// Cypher list of every worktree segment. A copy is never renamed: renaming it
/// would move it under the declared root and hide it from the purge.
const WORKTREE_SEGMENTS: &str = "['/.wl/', '/_wt/', '/.wt/', '/.rlwt/', '/worktrees/']";

/// Knowledge edge types re-pointed before a copy is deleted — the same list as
/// `2026-10-rehome-worktree-knowledge`. Structural edges (`CONTAINS`,
/// `IMPORTS`) go with the copy on purpose.
const KNOWLEDGE_EDGES: &str =
    "['LINKED_TO', 'AFFECTS', 'INCLUDES_ENTITY', 'DISCUSSED', 'STEP_IN_PROCESS']";

/// The project clause shared by every rename pass: the project is read by
/// `$scope`, and nothing happens when the two roots are the same (no alias).
const RENAMABLE_PROJECT: &str = r#"
MATCH (p:Project {id: $scope})
WHERE p.canonical_path IS NOT NULL AND p.canonical_path <> ''
  AND p.root_path IS NOT NULL AND p.root_path <> ''
  AND p.root_path <> p.canonical_path
"#;

/// (a) `File.path`: alias → declared. In place (`SET`), never recreated.
///
/// Idempotent: a renamed path no longer starts with the canonical root. A file
/// whose declared twin already exists is left alone (the `file_path` constraint
/// would refuse the rename) and reported by the measures, never silently merged.
fn rename_files_query() -> String {
    format!(
        r#"{RENAMABLE_PROJECT}
MATCH (f:File {{project_id: p.id}})
WHERE f.path STARTS WITH p.canonical_path + '/'
  AND none(seg IN {WORKTREE_SEGMENTS} WHERE f.path CONTAINS seg)
  AND NOT EXISTS {{ MATCH (:File {{path: p.root_path + substring(f.path, size(p.canonical_path))}}) }}
WITH f, p LIMIT $batch
SET f.path = p.root_path + substring(f.path, size(p.canonical_path))
RETURN count(f) AS processed
"#
    )
}

/// (b) + (d) one label: `file_path`, `id` and `stable_key` rewritten by the
/// same prefix replacement. A symbol whose declared twin (`id`) already exists
/// is skipped — the `<label>_id` constraint would refuse it.
///
/// Ownership is the path prefix, with the project id **when present**: until
/// #1045 the batch upserts stamped `project_id` on Function/Struct/Trait/Enum
/// only — measured 31/10/2026, 9 640 `Impl` and 15 863 `Import` under the alias
/// carry none, and they keep none until they are rewritten. A pass keyed on
/// `{project_id}` alone would have left those two labels in the old vocabulary,
/// silently.
fn rename_symbols_query(label: &str) -> String {
    format!(
        r#"{RENAMABLE_PROJECT}
MATCH (n:{label})
WHERE n.file_path STARTS WITH p.canonical_path + '/'
  AND (n.project_id = p.id OR n.project_id IS NULL OR n.project_id = '')
  AND none(seg IN {WORKTREE_SEGMENTS} WHERE n.file_path CONTAINS seg)
  AND NOT (n.id STARTS WITH p.canonical_path + '/'
           AND EXISTS {{ MATCH (:{label} {{id: p.root_path + substring(n.id, size(p.canonical_path))}}) }})
WITH n, p LIMIT $batch
SET n.file_path = p.root_path + substring(n.file_path, size(p.canonical_path)),
    n.id = CASE WHEN n.id STARTS WITH p.canonical_path + '/'
                THEN p.root_path + substring(n.id, size(p.canonical_path))
                ELSE n.id END,
    n.stable_key = CASE WHEN n.stable_key STARTS WITH p.canonical_path + '/'
                        THEN p.root_path + substring(n.stable_key, size(p.canonical_path))
                        ELSE n.stable_key END
RETURN count(n) AS processed
"#
    )
}

/// (c) `Anchor.target_id` — for a `file` anchor it IS the path; for a
/// `function` anchor it is the symbol id, prefixed by the path. Without this
/// pass they point at nothing, silently. `snapshot_path` follows.
fn rename_anchors_query() -> String {
    format!(
        r#"{RENAMABLE_PROJECT}
MATCH (a:Anchor)
WHERE a.target_type IN ['file', 'function']
  AND a.target_id STARTS WITH p.canonical_path + '/'
  AND none(seg IN {WORKTREE_SEGMENTS} WHERE a.target_id CONTAINS seg)
  AND NOT EXISTS {{ MATCH (:Anchor {{session_id: a.session_id, target_type: a.target_type,
                                   target_id: p.root_path + substring(a.target_id, size(p.canonical_path))}}) }}
WITH a, p LIMIT $batch
SET a.target_id = p.root_path + substring(a.target_id, size(p.canonical_path)),
    a.snapshot_path = CASE WHEN a.snapshot_path STARTS WITH p.canonical_path + '/'
                           THEN p.root_path + substring(a.snapshot_path, size(p.canonical_path))
                           ELSE a.snapshot_path END
RETURN count(a) AS processed
"#
    )
}

/// Cypher: the declared twin of a `.wl` copy. `…/.wl/<branch>/<rel>` →
/// `<root_path>/<rel>` — one level of scaffolding, measured on 31/10/2026 (all
/// 631 copies sit under one `.wl/rl5-admission/`).
fn wl_twin_expr(path: &str, root: &str) -> String {
    format!(
        "{root} + '/' + substring(split({path}, '{WL_SEGMENT}')[1], \
         size(split(split({path}, '{WL_SEGMENT}')[1], '/')[0]) + 1)"
    )
}

/// Purge step 1: re-point the knowledge edges of a `.wl` copy onto its twin.
///
/// `MATCH (c:File {path: twin})`, not `OPTIONAL MATCH`: an edge is deleted only
/// once its replacement exists. A copy without a twin keeps its edges until the
/// purge gate decides what they are worth — never a silent `DETACH`.
fn rehome_wl_knowledge_query() -> String {
    let twin = wl_twin_expr("w.path", "p.root_path");
    format!(
        r#"
MATCH (p:Project {{id: $scope}})
WHERE p.root_path IS NOT NULL AND p.root_path <> ''
MATCH (k)-[r]->(w:File {{project_id: p.id}})
WHERE w.path CONTAINS '{WL_SEGMENT}'
  AND type(r) IN {KNOWLEDGE_EDGES}
WITH k, r, {twin} AS twin
MATCH (c:File {{path: twin}})
WITH k, r, c LIMIT $batch
CALL apoc.merge.relationship(k, type(r), {{}}, properties(r), c) YIELD rel
DELETE r
RETURN count(*) AS processed
"#
    )
}

/// Purge step 2: the `.wl` files and the symbols they contain.
fn purge_wl_files_query() -> String {
    let labels = symbol_label_predicate("s");
    format!(
        r#"
MATCH (p:Project {{id: $scope}})
MATCH (f:File {{project_id: p.id}})
WHERE f.path CONTAINS '{WL_SEGMENT}'
WITH f LIMIT $batch
OPTIONAL MATCH (f)-[:CONTAINS]->(s)
WHERE {labels}
WITH collect(DISTINCT f) AS files, collect(s) AS symbols
FOREACH (x IN symbols | DETACH DELETE x)
FOREACH (x IN files | DETACH DELETE x)
RETURN size(files) AS processed
"#
    )
}

/// Purge step 3: the `.wl` symbols no `CONTAINS` edge reached (measured: the
/// copies' symbols have no `project_id`, so they are found by their path).
fn purge_wl_symbols_query() -> String {
    let labels = symbol_label_predicate("n");
    format!(
        r#"
MATCH (p:Project {{id: $scope}})
WHERE p.canonical_path IS NOT NULL AND p.canonical_path <> ''
MATCH (n)
WHERE {labels}
  AND n.file_path CONTAINS '{WL_SEGMENT}'
  AND (n.file_path STARTS WITH p.canonical_path + '/' OR n.file_path STARTS WITH p.root_path + '/')
WITH n LIMIT $batch
WITH collect(n) AS doomed
FOREACH (x IN doomed | DETACH DELETE x)
RETURN size(doomed) AS processed
"#
    )
}

fn symbol_label_predicate(var: &str) -> String {
    SYMBOL_LABELS
        .iter()
        .map(|l| format!("{var}:{l}"))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// The control queries — the six of note `088c0d1a`, plus what the purge gate
/// needs. Every one takes `$pid`, `$root` (declared) and `$canonical`, computed
/// in Rust so that a dry-run works before `canonical_path` is stored. Reads only.
pub mod measure {
    use super::{
        symbol_label_predicate, wl_twin_expr, KNOWLEDGE_EDGES, WL_SEGMENT, WORKTREE_SEGMENTS,
    };

    /// 1. Files still under the alias (worktree copies excluded).
    pub fn alias_files() -> String {
        format!(
            "MATCH (f:File {{project_id: $pid}}) WHERE f.path STARTS WITH $canonical + '/' \
             AND none(seg IN {WORKTREE_SEGMENTS} WHERE f.path CONTAINS seg) RETURN count(f) AS n"
        )
    }

    /// 1'. … of which a twin already exists under the declared root: not renamable.
    pub fn alias_files_blocked_by_twin() -> String {
        format!(
            "MATCH (f:File {{project_id: $pid}}) WHERE f.path STARTS WITH $canonical + '/' \
             AND none(seg IN {WORKTREE_SEGMENTS} WHERE f.path CONTAINS seg) \
             AND EXISTS {{ MATCH (:File {{path: $root + substring(f.path, size($canonical))}}) }} \
             RETURN count(f) AS n"
        )
    }

    /// 1b. Symbols of one label still under the alias — the project id when
    /// present (Impl and Import carry none), the path prefix always.
    pub fn alias_symbols(label: &str) -> String {
        format!(
            "MATCH (n:{label}) WHERE n.file_path STARTS WITH $canonical + '/' \
             AND (n.project_id = $pid OR n.project_id IS NULL OR n.project_id = '') \
             AND none(seg IN {WORKTREE_SEGMENTS} WHERE n.file_path CONTAINS seg) RETURN count(n) AS n"
        )
    }

    /// 2. Files under the declared root.
    pub fn declared_files() -> String {
        "MATCH (f:File {project_id: $pid}) WHERE f.path STARTS WITH $root + '/' RETURN count(f) AS n"
            .to_string()
    }

    /// 3. Notes of the project with no file anchor at all. Must never rise.
    pub fn notes_without_file_anchor() -> String {
        "MATCH (n:Note {project_id: $pid}) WHERE NOT (n)-[:LINKED_TO]->(:File) RETURN count(n) AS n"
            .to_string()
    }

    /// 4. `file` anchors naming one of the project's roots and pointing at no `File`.
    pub fn dangling_file_anchors() -> String {
        "MATCH (a:Anchor {target_type: 'file'}) \
         WHERE (a.target_id STARTS WITH $root + '/' OR a.target_id STARTS WITH $canonical + '/') \
         AND NOT EXISTS { MATCH (:File {path: a.target_id}) } RETURN count(a) AS n"
            .to_string()
    }

    /// 4b. Anchors still naming the alias.
    pub fn alias_anchors() -> String {
        "MATCH (a:Anchor) WHERE a.target_type IN ['file', 'function'] \
         AND a.target_id STARTS WITH $canonical + '/' RETURN count(a) AS n"
            .to_string()
    }

    /// 5. `uid` coverage: total…
    pub fn files_total() -> String {
        "MATCH (f:File {project_id: $pid}) RETURN count(f) AS n".to_string()
    }

    /// 5. … and with a uid.
    pub fn files_with_uid() -> String {
        "MATCH (f:File {project_id: $pid}) WHERE f.uid IS NOT NULL RETURN count(f) AS n".to_string()
    }

    /// 6. `stable_key` that no longer starts with its own `file_path + '::'`.
    pub fn incoherent_stable_keys() -> String {
        format!(
            "MATCH (n) WHERE ({}) \
             AND (n.file_path STARTS WITH $root + '/' OR n.file_path STARTS WITH $canonical + '/') \
             AND n.stable_key IS NOT NULL \
             AND NOT n.stable_key STARTS WITH n.file_path + '::' RETURN count(n) AS n",
            symbol_label_predicate("n")
        )
    }

    pub fn wl_files() -> String {
        format!(
            "MATCH (f:File {{project_id: $pid}}) WHERE f.path CONTAINS '{WL_SEGMENT}' RETURN count(f) AS n"
        )
    }

    pub fn wl_symbols() -> String {
        format!(
            "MATCH (n) WHERE ({}) AND n.file_path CONTAINS '{WL_SEGMENT}' \
             AND (n.file_path STARTS WITH $canonical + '/' OR n.file_path STARTS WITH $root + '/') \
             RETURN count(n) AS n",
            symbol_label_predicate("n")
        )
    }

    pub fn wl_knowledge_edges() -> String {
        format!(
            "MATCH (k)-[r]->(w:File {{project_id: $pid}}) WHERE w.path CONTAINS '{WL_SEGMENT}' \
             AND type(r) IN {KNOWLEDGE_EDGES} RETURN count(r) AS n"
        )
    }

    pub fn wl_knowledge_edges_with_twin() -> String {
        let twin = wl_twin_expr("w.path", "$root");
        format!(
            "MATCH (k)-[r]->(w:File {{project_id: $pid}}) WHERE w.path CONTAINS '{WL_SEGMENT}' \
             AND type(r) IN {KNOWLEDGE_EDGES} \
             AND EXISTS {{ MATCH (:File {{path: {twin}}}) }} RETURN count(r) AS n"
        )
    }

    /// Notes whose ONLY file anchors are `.wl` copies without a declared twin:
    /// the purge would take their last anchor. Gate of `--purge-wl`.
    pub fn notes_losing_last_anchor() -> String {
        let twin = wl_twin_expr("path", "$root");
        format!(
            "MATCH (n:Note)-[:LINKED_TO]->(w:File {{project_id: $pid}}) WHERE w.path CONTAINS '{WL_SEGMENT}' \
             WITH n, collect(w.path) AS wl \
             WHERE NOT EXISTS {{ MATCH (n)-[:LINKED_TO]->(o:File) WHERE NOT o.path CONTAINS '{WL_SEGMENT}' }} \
             AND none(path IN wl WHERE EXISTS {{ MATCH (:File {{path: {twin}}}) }}) \
             RETURN count(n) AS n"
        )
    }

    /// Every control query, for the static tests.
    pub fn all() -> Vec<(&'static str, String)> {
        vec![
            ("alias_files", alias_files()),
            ("alias_files_blocked_by_twin", alias_files_blocked_by_twin()),
            ("alias_symbols", alias_symbols("Function")),
            ("declared_files", declared_files()),
            ("notes_without_file_anchor", notes_without_file_anchor()),
            ("dangling_file_anchors", dangling_file_anchors()),
            ("alias_anchors", alias_anchors()),
            ("files_total", files_total()),
            ("files_with_uid", files_with_uid()),
            ("incoherent_stable_keys", incoherent_stable_keys()),
            ("wl_files", wl_files()),
            ("wl_symbols", wl_symbols()),
            ("wl_knowledge_edges", wl_knowledge_edges()),
            (
                "wl_knowledge_edges_with_twin",
                wl_knowledge_edges_with_twin(),
            ),
            ("notes_losing_last_anchor", notes_losing_last_anchor()),
        ]
    }
}

/// Static ids: `DataMigration.id` is `&'static str`, and these never reach the
/// `(:DataMigration)` markers — the command is replayable by design.
fn symbol_pass_id(label: &str) -> &'static str {
    match label {
        "Function" => "file-paths/rename-functions",
        "Struct" => "file-paths/rename-structs",
        "Trait" => "file-paths/rename-traits",
        "Enum" => "file-paths/rename-enums",
        "Impl" => "file-paths/rename-impls",
        "Import" => "file-paths/rename-imports",
        "Module" => "file-paths/rename-modules",
        other => unreachable!("unknown symbol label {other}"),
    }
}

/// The rename passes, in the order they must run: files, then every symbol
/// label, then the anchors. `stable_key` is rewritten inside the symbol pass.
pub fn rename_passes() -> Vec<DataMigration> {
    let mut passes = vec![DataMigration {
        id: "file-paths/rename-files",
        description: "File.path: alias → declared root (in place)",
        batch: rename_files_query(),
        finalize: None,
        compute: None,
    }];
    for label in SYMBOL_LABELS {
        passes.push(DataMigration {
            id: symbol_pass_id(label),
            description: "file_path, id and stable_key: alias → declared root",
            batch: rename_symbols_query(label),
            finalize: None,
            compute: None,
        });
    }
    passes.push(DataMigration {
        id: "file-paths/rename-anchors",
        description: "Anchor.target_id (file, function): alias → declared root",
        batch: rename_anchors_query(),
        finalize: None,
        compute: None,
    });
    passes
}

/// The purge passes, in order: re-home the knowledge, then delete the copies.
pub fn purge_passes() -> Vec<DataMigration> {
    vec![
        DataMigration {
            id: "file-paths/rehome-wl-knowledge",
            description: "Knowledge edges of .wl copies → declared twin (only when it exists)",
            batch: rehome_wl_knowledge_query(),
            finalize: None,
            compute: None,
        },
        DataMigration {
            id: "file-paths/purge-wl-files",
            description: "DETACH DELETE the .wl files and their symbols",
            batch: purge_wl_files_query(),
            finalize: None,
            compute: None,
        },
        DataMigration {
            id: "file-paths/purge-wl-symbols",
            description: "DETACH DELETE the .wl symbols no file reached",
            batch: purge_wl_symbols_query(),
            finalize: None,
            compute: None,
        },
    ]
}

/// A `File` of the alias whose declared twin already exists: the rename pass
/// skipped it (the `file_path` constraint). The twin is the fresher node (a
/// sync wrote it), the alias carries the older edges: fold the alias INTO the
/// twin — every edge moves, deduplicated, the twin's properties win — and the
/// alias disappears. Never skipped in silence: a skipped alias would keep two
/// vocabularies in the graph and the purge gate would never open.
fn merge_file_twins_query() -> String {
    format!(
        r#"{RENAMABLE_PROJECT}
MATCH (n:File {{project_id: p.id}})
WHERE n.path STARTS WITH p.canonical_path + '/'
  AND none(seg IN {WORKTREE_SEGMENTS} WHERE n.path CONTAINS seg)
MATCH (t:File {{path: p.root_path + substring(n.path, size(p.canonical_path))}})
WITH n, t LIMIT $batch
CALL apoc.refactor.mergeNodes([t, n], {{properties: 'discard', mergeRels: true}}) YIELD node
RETURN count(node) AS processed
"#
    )
}

/// Same fold for the symbols of one label whose declared twin (`id`) exists.
fn merge_symbol_twins_query(label: &str) -> String {
    format!(
        r#"{RENAMABLE_PROJECT}
MATCH (n:{label})
WHERE n.file_path STARTS WITH p.canonical_path + '/'
  AND (n.project_id = p.id OR n.project_id IS NULL OR n.project_id = '')
  AND none(seg IN {WORKTREE_SEGMENTS} WHERE n.file_path CONTAINS seg)
  AND n.id STARTS WITH p.canonical_path + '/'
MATCH (t:{label} {{id: p.root_path + substring(n.id, size(p.canonical_path))}})
WITH n, t LIMIT $batch
CALL apoc.refactor.mergeNodes([t, n], {{properties: 'discard', mergeRels: true}}) YIELD node
RETURN count(node) AS processed
"#
    )
}

fn merge_pass_id(label: &str) -> &'static str {
    match label {
        "Function" => "file-paths/merge-function-twins",
        "Struct" => "file-paths/merge-struct-twins",
        "Trait" => "file-paths/merge-trait-twins",
        "Enum" => "file-paths/merge-enum-twins",
        "Impl" => "file-paths/merge-impl-twins",
        "Import" => "file-paths/merge-import-twins",
        "Module" => "file-paths/merge-module-twins",
        other => unreachable!("unknown symbol label {other}"),
    }
}

/// The twin-merge passes, run AFTER [`rename_passes`]: what the rename had to
/// skip because a declared twin existed is folded into that twin.
pub fn merge_twin_passes() -> Vec<DataMigration> {
    let mut passes = vec![DataMigration {
        id: "file-paths/merge-file-twins",
        description: "Fold an alias File into its declared twin (edges move, alias deleted)",
        batch: merge_file_twins_query(),
        finalize: None,
        compute: None,
    }];
    for label in SYMBOL_LABELS {
        passes.push(DataMigration {
            id: merge_pass_id(label),
            description: "Fold an alias symbol into its declared twin (edges move, alias deleted)",
            batch: merge_symbol_twins_query(label),
            finalize: None,
            compute: None,
        });
    }
    passes
}

/// What the command knows about the project before touching anything.
#[derive(Debug, Clone, Serialize)]
pub struct RenamePlan {
    pub project_id: Uuid,
    pub slug: String,
    /// `Project.root_path`, as the user declared it.
    pub declared_root: String,
    /// `canonical_root(declared_root)`, resolved here in Rust.
    pub canonical_root: String,
    /// `Project.canonical_path` as stored — `None` when never filled.
    pub stored_canonical: Option<String>,
    /// `Project.paths_declared`: the sync already writes the declared form.
    pub paths_declared: bool,
}

impl RenamePlan {
    /// Nothing to rename when the declared root is not a link.
    pub fn has_alias(&self) -> bool {
        !self.declared_root.is_empty()
            && !self.canonical_root.is_empty()
            && self.declared_root.trim_end_matches('/') != self.canonical_root.trim_end_matches('/')
    }
}

/// The control measures — the six of note `088c0d1a`, plus what the purge needs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct PathMeasures {
    /// 1. Files still under the alias (worktree copies excluded).
    pub alias_files: i64,
    /// … of which blocked: a twin already exists under the declared root.
    pub alias_files_blocked_by_twin: i64,
    /// 1b. Symbols still under the alias, per label.
    pub alias_symbols: Vec<(String, i64)>,
    /// 2. Files under the declared root.
    pub declared_files: i64,
    /// 3. Notes of the project with no `LINKED_TO` file at all.
    pub notes_without_file_anchor: i64,
    /// 4. `file` anchors of this project pointing at no `File`.
    pub dangling_file_anchors: i64,
    /// 4b. Anchors still naming the alias.
    pub alias_anchors: i64,
    /// 5. `uid` coverage.
    pub files_total: i64,
    pub files_with_uid: i64,
    /// 6. `stable_key` that does not start with its own `file_path + '::'`.
    pub incoherent_stable_keys: i64,
    /// Worktree copies.
    pub wl_files: i64,
    pub wl_symbols: i64,
    /// Knowledge edges onto `.wl` files, and how many have a declared twin.
    pub wl_knowledge_edges: i64,
    pub wl_knowledge_edges_with_twin: i64,
    /// Notes whose ONLY file anchors are `.wl` copies without a twin.
    pub notes_losing_last_anchor: i64,
}

/// Why a purge is refused. Pure, so the gate is testable without a graph.
pub fn purge_gate(before: &PathMeasures, after: &PathMeasures) -> Result<(), Vec<String>> {
    let mut reasons = Vec::new();
    if after.dangling_file_anchors > 0 {
        reasons.push(format!(
            "{} ancre(s) `file` pointent dans le vide après le renommage",
            after.dangling_file_anchors
        ));
    }
    if after.notes_losing_last_anchor > 0 {
        reasons.push(format!(
            "{} note(s) perdraient leur dernière ancre fichier (copie .wl sans jumeau)",
            after.notes_losing_last_anchor
        ));
    }
    if after.notes_without_file_anchor > before.notes_without_file_anchor {
        reasons.push(format!(
            "les notes sans ancre fichier ont augmenté ({} -> {}) : le renommage a perdu des arêtes",
            before.notes_without_file_anchor, after.notes_without_file_anchor
        ));
    }
    if after.alias_files > 0 || after.alias_symbols.iter().any(|(_, n)| *n > 0) {
        reasons.push(
            "des chemins de l'alias subsistent : purger maintenant laisserait deux vocabulaires"
                .to_string(),
        );
    }
    if reasons.is_empty() {
        Ok(())
    } else {
        Err(reasons)
    }
}

/// Human-readable measures, one line each, French like the rest of the CLI.
pub fn render_measures(title: &str, m: &PathMeasures) -> String {
    let mut out = String::new();
    out.push_str(&format!("== {title}\n"));
    out.push_str(&format!(
        "  1. File sous l'alias (hors .wl)            : {}  (dont {} bloqué(s) par un jumeau déjà présent)\n",
        m.alias_files, m.alias_files_blocked_by_twin
    ));
    for (label, n) in &m.alias_symbols {
        out.push_str(&format!(
            "     {label:<8} file_path sous l'alias          : {n}\n"
        ));
    }
    out.push_str(&format!(
        "  2. File sous la racine déclarée             : {}\n",
        m.declared_files
    ));
    out.push_str(&format!(
        "  3. Notes du projet sans ancre fichier       : {}\n",
        m.notes_without_file_anchor
    ));
    out.push_str(&format!(
        "  4. Ancres `file` pointant dans le vide      : {}  (ancres nommant encore l'alias : {})\n",
        m.dangling_file_anchors, m.alias_anchors
    ));
    out.push_str(&format!(
        "  5. Couverture File.uid                      : {}/{}\n",
        m.files_with_uid, m.files_total
    ));
    out.push_str(&format!(
        "  6. stable_key incohérents avec file_path    : {}\n",
        m.incoherent_stable_keys
    ));
    out.push_str(&format!(
        "  .wl : {} File, {} symboles ; {} arête(s) de connaissance, {} avec jumeau ; {} note(s) perdraient leur dernière ancre\n",
        m.wl_files,
        m.wl_symbols,
        m.wl_knowledge_edges,
        m.wl_knowledge_edges_with_twin,
        m.notes_losing_last_anchor
    ));
    out
}

/// What `--apply` would do, from the measures.
pub fn render_plan(plan: &RenamePlan, m: &PathMeasures, purge_wl: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "== Projet `{}` ({})\n  racine déclarée  : {}\n  racine canonique : {}\n  canonical_path stocké : {}\n  paths_declared        : {}\n",
        plan.slug,
        plan.project_id,
        plan.declared_root,
        plan.canonical_root,
        plan.stored_canonical.as_deref().unwrap_or("(absent — sera posé par --apply)"),
        if plan.paths_declared {
            "true (le sync écrit déjà la forme déclarée)"
        } else {
            "absent (le sync écrit la forme canonique jusqu'à --apply)"
        }
    ));
    if !plan.has_alias() {
        out.push_str("  La racine déclarée n'est pas un lien : rien à renommer.\n");
        return out;
    }
    let symbols: i64 = m.alias_symbols.iter().map(|(_, n)| n).sum();
    out.push_str("== Ce que --apply ferait (par lots de 2 000, idempotent)\n");
    out.push_str(&format!(
        "  (a) SET File.path sur {} nœud(s) : `{}/…` -> `{}/…` (aucun nœud créé)\n",
        m.alias_files - m.alias_files_blocked_by_twin,
        plan.canonical_root,
        plan.declared_root
    ));
    out.push_str(&format!(
        "  (b) SET file_path, id sur {symbols} symbole(s) ({})\n",
        SYMBOL_LABELS.join(", ")
    ));
    out.push_str(&format!(
        "  (c) SET Anchor.target_id sur {} ancre(s) `file`/`function`\n",
        m.alias_anchors
    ));
    out.push_str(
        "  (d) stable_key réécrit par le même remplacement de préfixe, dans la passe (b)\n",
    );
    out.push_str("  (e) File.uid : intact (calculé sur le chemin relatif)\n");
    out.push_str(&format!(
        "  (f) fusion des jumeaux : {} File de l'alias ont déjà un jumeau déclaré — arêtes déplacées, alias supprimé (jamais sauté)\n",
        m.alias_files_blocked_by_twin
    ));
    out.push_str("  marqueur Project.paths_declared posé AVANT (a) : un sync pendant la migration écrit la forme d'arrivée\n");
    if purge_wl {
        out.push_str(&format!(
            "  --purge-wl : réancrer {} arête(s) sur le jumeau déclaré, puis DETACH DELETE {} File .wl et {} symboles\n",
            m.wl_knowledge_edges_with_twin, m.wl_files, m.wl_symbols
        ));
    } else if m.wl_files > 0 {
        out.push_str(&format!(
            "  ({} File .wl et {} symboles restent : ajouter --purge-wl avec --apply pour les purger)\n",
            m.wl_files, m.wl_symbols
        ));
    }
    out
}

impl Neo4jClient {
    /// Read the project by slug and resolve its roots. Reads only.
    pub async fn file_path_plan(&self, slug: &str) -> Result<RenamePlan> {
        let project = self
            .get_project_by_slug(slug)
            .await?
            .with_context(|| format!("aucun projet avec le slug `{slug}`"))?;
        // `~` expanded: a tilde root is what the user typed, not a prefix any
        // stored path starts with.
        let declared_root = crate::expand_tilde(project.root_path.trim())
            .trim_end_matches('/')
            .to_string();
        anyhow::ensure!(
            !declared_root.is_empty(),
            "le projet `{slug}` n'a pas de root_path : rien à renommer"
        );
        let (stored_canonical, paths_declared) = self.project_markers(project.id).await?;
        Ok(RenamePlan {
            project_id: project.id,
            slug: project.slug,
            canonical_root: canonical_root(Some(&declared_root)),
            declared_root,
            stored_canonical,
            paths_declared,
        })
    }

    /// `(canonical_path as stored, paths_declared)`.
    async fn project_markers(&self, project_id: Uuid) -> Result<(Option<String>, bool)> {
        let mut result = self
            .graph
            .execute(
                query(
                    "MATCH (p:Project {id: $id}) \
                     RETURN p.canonical_path AS c, coalesce(p.paths_declared, false) AS d",
                )
                .param("id", project_id.to_string()),
            )
            .await?;
        Ok(match result.next().await? {
            Some(row) => (
                row.get::<String>("c").ok().filter(|c| !c.is_empty()),
                row.get::<bool>("d").unwrap_or(false),
            ),
            None => (None, false),
        })
    }

    /// The control measures. Reads only — this is what the dry-run prints.
    pub async fn file_path_measures(&self, plan: &RenamePlan) -> Result<PathMeasures> {
        let p = |q: String| {
            query(&q)
                .param("pid", plan.project_id.to_string())
                .param("root", plan.declared_root.clone())
                .param("canonical", plan.canonical_root.clone())
        };
        let mut alias_symbols = Vec::with_capacity(SYMBOL_LABELS.len());
        for label in SYMBOL_LABELS {
            let n = self.count(p(measure::alias_symbols(label))).await?;
            alias_symbols.push((label.to_string(), n));
        }
        Ok(PathMeasures {
            alias_files: self.count(p(measure::alias_files())).await?,
            alias_files_blocked_by_twin: self
                .count(p(measure::alias_files_blocked_by_twin()))
                .await?,
            alias_symbols,
            declared_files: self.count(p(measure::declared_files())).await?,
            notes_without_file_anchor: self.count(p(measure::notes_without_file_anchor())).await?,
            dangling_file_anchors: self.count(p(measure::dangling_file_anchors())).await?,
            alias_anchors: self.count(p(measure::alias_anchors())).await?,
            files_total: self.count(p(measure::files_total())).await?,
            files_with_uid: self.count(p(measure::files_with_uid())).await?,
            incoherent_stable_keys: self.count(p(measure::incoherent_stable_keys())).await?,
            wl_files: self.count(p(measure::wl_files())).await?,
            wl_symbols: self.count(p(measure::wl_symbols())).await?,
            wl_knowledge_edges: self.count(p(measure::wl_knowledge_edges())).await?,
            wl_knowledge_edges_with_twin: self
                .count(p(measure::wl_knowledge_edges_with_twin()))
                .await?,
            notes_losing_last_anchor: self.count(p(measure::notes_losing_last_anchor())).await?,
        })
    }

    async fn count(&self, q: neo4rs::Query) -> Result<i64> {
        let mut result = self.graph.execute(q).await?;
        let row = result
            .next()
            .await?
            .context("count query returned no row")?;
        Ok(row.get::<i64>("n").unwrap_or(0))
    }

    /// Run the rename passes (a)–(d). Writes `canonical_path` first, because
    /// the passes read it from the node. Stops at the first failed pass.
    pub async fn file_path_rename(&self, plan: &RenamePlan) -> Result<Vec<MigrationOutcome>> {
        self.ensure_project_canonical_paths()
            .await
            .context("canonical_path repair")?;
        // The marker FIRST: from here every sync writes the declared form, the
        // form the passes below converge to. A sync that runs during the
        // migration can at worst create a declared twin of an alias node — the
        // merge passes fold it. Marking last would let such a sync create new
        // alias nodes after the rename had passed them.
        self.mark_paths_declared(plan.project_id)
            .await
            .context("paths_declared marker")?;
        let mut outcomes = self.run_passes(plan, rename_passes()).await?;
        outcomes.extend(self.run_passes(plan, merge_twin_passes()).await?);
        Ok(outcomes)
    }

    /// Run the purge passes — the caller has checked [`purge_gate`].
    pub async fn file_path_purge_wl(&self, plan: &RenamePlan) -> Result<Vec<MigrationOutcome>> {
        self.run_passes(plan, purge_passes()).await
    }

    async fn run_passes(
        &self,
        plan: &RenamePlan,
        passes: Vec<DataMigration>,
    ) -> Result<Vec<MigrationOutcome>> {
        let scope = Some(plan.project_id.to_string());
        let mut outcomes = Vec::new();
        for pass in passes {
            let outcome = self.run_migration_batches(&pass, scope.clone()).await;
            let failed = outcome.error.clone();
            outcomes.push(outcome);
            if let Some(e) = failed {
                anyhow::bail!("passe `{}` en échec : {e}", pass.id);
            }
        }
        Ok(outcomes)
    }
}

/// The command. Prints as it goes; returns the process exit code.
pub async fn run(client: &Neo4jClient, slug: &str, apply: bool, purge_wl: bool) -> Result<i32> {
    let plan = client.file_path_plan(slug).await?;
    let before = client.file_path_measures(&plan).await?;
    print!("{}", render_plan(&plan, &before, purge_wl));
    print!("{}", render_measures("Mesures avant", &before));
    if !plan.has_alias() {
        return Ok(0);
    }
    if !apply {
        println!("Dry-run : rien n'a été écrit. Relancer avec --apply pour exécuter.");
        return Ok(0);
    }

    println!("== Renommage (--apply)");
    for o in client.file_path_rename(&plan).await? {
        println!(
            "  {:<36} {:>8} nœud(s), {} lot(s)",
            o.id, o.processed, o.batches
        );
    }
    let after = client.file_path_measures(&plan).await?;
    print!("{}", render_measures("Mesures après renommage", &after));

    if !purge_wl {
        if after.wl_files > 0 {
            println!(
                "{} File .wl restent : relancer avec --apply --purge-wl une fois ces mesures relues.",
                after.wl_files
            );
        }
        return Ok(0);
    }

    if let Err(reasons) = purge_gate(&before, &after) {
        println!("== Purge .wl REFUSÉE");
        for r in reasons {
            println!("  - {r}");
        }
        return Ok(2);
    }
    println!("== Purge .wl (--purge-wl)");
    for o in client.file_path_purge_wl(&plan).await? {
        println!(
            "  {:<36} {:>8} élément(s), {} lot(s)",
            o.id, o.processed, o.batches
        );
    }
    let purged = client.file_path_measures(&plan).await?;
    print!("{}", render_measures("Mesures après purge", &purged));
    if purged.notes_without_file_anchor > after.notes_without_file_anchor {
        println!(
            "ATTENTION : les notes sans ancre fichier ont augmenté ({} -> {}).",
            after.notes_without_file_anchor, purged.notes_without_file_anchor
        );
        return Ok(2);
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_passes() -> Vec<DataMigration> {
        let mut v = rename_passes();
        v.extend(merge_twin_passes());
        v.extend(purge_passes());
        v
    }

    /// Un jumeau déclaré existant n'est jamais sauté en silence : l'alias est
    /// FONDU dans le jumeau (arêtes déplacées et dédupliquées, propriétés du
    /// jumeau conservées), puis disparaît. Sinon deux vocabulaires subsistent et
    /// le portillon de purge ne s'ouvre jamais.
    #[test]
    fn an_alias_with_a_declared_twin_is_folded_into_it() {
        let passes = merge_twin_passes();
        assert_eq!(passes.len(), 1 + SYMBOL_LABELS.len());
        for pass in &passes {
            assert!(pass.id.starts_with("file-paths/merge-"), "{}", pass.id);
            assert!(pass.batch.contains("p.canonical_path"));
            assert!(pass.batch.contains("p.root_path <> p.canonical_path"));
            // Le jumeau d'abord dans la liste : c'est lui qui survit.
            assert!(
                pass.batch.contains(
                    "apoc.refactor.mergeNodes([t, n], {properties: 'discard', mergeRels: true})"
                ),
                "{}",
                pass.id
            );
            assert!(
                pass.batch.contains("MATCH (t:"),
                "the twin MUST exist (MATCH)"
            );
            assert!(!pass.batch.contains("OPTIONAL MATCH"));
            assert!(!pass.batch.contains("CREATE"));
            assert!(
                pass.batch.contains("none(seg IN"),
                "copies are never folded"
            );
        }
        assert!(merge_file_twins_query()
            .contains("MATCH (t:File {path: p.root_path + substring(n.path"));
        for label in SYMBOL_LABELS {
            assert!(merge_symbol_twins_query(label).contains(&format!(
                "MATCH (t:{label} {{id: p.root_path + substring(n.id"
            )));
        }
    }

    #[test]
    fn the_merge_runs_after_the_rename_and_the_marker_is_set_first() {
        let src = include_str!("path_migration.rs");
        let body = &src[src.find("pub async fn file_path_rename").unwrap()..];
        let mark = body.find("mark_paths_declared(plan.project_id)").unwrap();
        let rename = body.find("run_passes(plan, rename_passes())").unwrap();
        let merge = body.find("run_passes(plan, merge_twin_passes())").unwrap();
        assert!(mark < rename && rename < merge);
    }

    /// Hors du chemin de boot : aucune passe n'est une `DataMigration` enregistrée.
    #[test]
    fn the_rename_is_not_a_boot_migration() {
        let boot: Vec<&str> = Neo4jClient::data_migrations()
            .iter()
            .map(|m| m.id)
            .collect();
        for pass in all_passes() {
            assert!(
                !boot.contains(&pass.id),
                "{} must not run at boot: the rename is triggered, never automatic",
                pass.id
            );
            assert!(pass.id.starts_with("file-paths/"));
        }
    }

    #[test]
    fn every_pass_is_scoped_bounded_and_reports_progress() {
        for pass in all_passes() {
            assert!(
                pass.batch.contains("$scope"),
                "{} must honour $scope",
                pass.id
            );
            assert!(
                pass.batch.contains("LIMIT $batch"),
                "{} must be bounded",
                pass.id
            );
            assert!(
                pass.batch.contains("AS processed"),
                "{} must report progress",
                pass.id
            );
        }
    }

    /// Renommer, jamais recréer : les 28 833 arêtes des notes tiennent aux nœuds.
    #[test]
    fn the_rename_passes_never_create_a_node() {
        let merge_file = ["MERGE (", "f:File"].concat();
        for pass in rename_passes() {
            assert!(
                !pass.batch.contains("CREATE"),
                "{} must not CREATE",
                pass.id
            );
            assert!(!pass.batch.contains("MERGE"), "{} must not MERGE", pass.id);
            assert!(!pass.batch.contains(&merge_file));
            assert!(
                !pass.batch.contains("DELETE"),
                "{} renames, it deletes nothing",
                pass.id
            );
        }
    }

    /// Les passes lisent les DEUX racines sur le nœud Project et ne font rien
    /// quand elles sont égales (pas d'alias).
    #[test]
    fn the_rename_passes_read_both_roots_from_the_project() {
        for pass in rename_passes() {
            assert!(pass.batch.contains("p.canonical_path"), "{}", pass.id);
            assert!(pass.batch.contains("p.root_path"), "{}", pass.id);
            assert!(
                pass.batch.contains("p.root_path <> p.canonical_path"),
                "{}",
                pass.id
            );
            // Idempotence : le prédicat devient faux dès que le SET est passé.
            assert!(
                pass.batch.contains("STARTS WITH p.canonical_path + '/'"),
                "{}",
                pass.id
            );
            assert!(
                pass.batch.contains("p.root_path + substring("),
                "{}",
                pass.id
            );
        }
    }

    #[test]
    fn the_file_pass_renames_the_path_and_leaves_the_uid_alone() {
        let q = rename_files_query();
        assert!(q.contains("SET f.path = p.root_path + substring(f.path, size(p.canonical_path))"));
        assert!(
            !q.contains("uid"),
            "File.uid is computed on the relative path: untouched"
        );
        // Un jumeau déjà présent bloque le renommage au lieu de violer la contrainte.
        assert!(q.contains("NOT EXISTS { MATCH (:File {path: p.root_path + substring("));
    }

    /// Une copie de worktree n'est jamais renommée : elle est purgée.
    #[test]
    fn worktree_copies_are_excluded_from_every_rename() {
        for pass in rename_passes() {
            for seg in ["/.wl/", "/_wt/", "/.wt/", "/.rlwt/", "/worktrees/"] {
                assert!(pass.batch.contains(seg), "{} must exclude {seg}", pass.id);
            }
            assert!(pass.batch.contains("none(seg IN"));
        }
    }

    /// Les sept labels qui portent `file_path`, et pour chacun les trois
    /// propriétés qui contiennent le chemin.
    #[test]
    fn every_symbol_label_has_a_pass_that_rewrites_the_three_path_properties() {
        let passes = rename_passes();
        for label in SYMBOL_LABELS {
            let q = &passes
                .iter()
                .find(|p| p.batch.contains(&format!("MATCH (n:{label})\n")))
                .unwrap_or_else(|| panic!("no pass for {label}"))
                .batch;
            // Impl et Import n'ont pas de project_id (mesuré : 9 640 et 15 863
            // sous l'alias) : l'appartenance est le préfixe, l'id quand il existe.
            assert!(q.contains("n.project_id = p.id OR n.project_id IS NULL"));
            assert!(!q.contains(&format!("(n:{label} {{project_id")));
            assert!(q.contains("SET n.file_path = p.root_path + substring(n.file_path"));
            assert!(q.contains("n.id = CASE WHEN n.id STARTS WITH p.canonical_path + '/'"));
            assert!(q.contains(
                "n.stable_key = CASE WHEN n.stable_key STARTS WITH p.canonical_path + '/'"
            ));
            assert!(q.contains(&format!(
                "MATCH (:{label} {{id: p.root_path + substring(n.id"
            )));
        }
        assert_eq!(SYMBOL_LABELS.len(), 7);
    }

    /// (d) La réécriture de `stable_key` par préfixe n'est juste que si chaque
    /// grammaire commence par `file_path::` — c'est le cas des quatre.
    #[test]
    fn the_stable_key_prefix_rewrite_matches_the_grammar() {
        use crate::neo4j::models::SymbolOwner;
        use crate::neo4j::symbol_key::*;
        let fp = "/Users/x/.openclaw/repo/src/a.rs";
        let owner = SymbolOwner {
            type_name: "Parser".into(),
            trait_name: Some("Debug".into()),
        };
        for key in [
            stable_symbol_key(fp, None, "f", &[]),
            // La méthode porte son propriétaire APRÈS le chemin : le préfixe tient.
            stable_symbol_key(fp, Some(&owner), "fmt", &[]),
            stable_type_key(fp, "T"),
            stable_impl_key(fp, "T", Some("Debug")),
            stable_import_key(fp, "std::fmt", 0),
        ] {
            assert!(key.starts_with(&format!("{fp}::")), "{key}");
            // Le même remplacement que sur file_path donne la clé recalculée.
            let renamed = format!(
                "/Users/x/projects/repo{}",
                &key[fp.rfind("/src/").unwrap()..]
            );
            assert!(renamed.starts_with("/Users/x/projects/repo/src/a.rs::"));
        }
        // Un trait externe n'a pas de chemin : sa clé (et son id) ne commencent
        // pas par la racine canonique, donc aucune passe ne les touche.
        let external = stable_external_trait_key("std::fmt::Debug");
        assert!(external.starts_with("external:trait:"));
        assert!(!external.starts_with('/'));
        let q = rename_symbols_query("Trait");
        assert!(q.contains("n.stable_key STARTS WITH p.canonical_path + '/'"));
        assert!(q.contains("n.id STARTS WITH p.canonical_path + '/'"));
    }

    #[test]
    fn the_anchor_pass_covers_file_and_function_anchors() {
        let q = rename_anchors_query();
        assert!(q.contains("a.target_type IN ['file', 'function']"));
        assert!(q.contains("SET a.target_id = p.root_path + substring(a.target_id"));
        assert!(q.contains("a.snapshot_path"));
        // La contrainte (session_id, target_type, target_id) est respectée.
        assert!(q.contains("session_id: a.session_id, target_type: a.target_type"));
    }

    /// La leçon de `2026-10-rehome-worktree-knowledge` : une arête n'est
    /// supprimée que si son remplacement a été créé — MATCH, pas OPTIONAL MATCH.
    #[test]
    fn the_rehome_deletes_an_edge_only_once_its_twin_exists() {
        let q = rehome_wl_knowledge_query();
        assert!(q.contains("MATCH (c:File {path: twin})"));
        assert!(!q.contains("OPTIONAL MATCH"));
        assert!(q.contains("apoc.merge.relationship(k, type(r), {}, properties(r), c)"));
        assert!(q.contains("DELETE r"));
        assert!(
            !q.contains("DETACH"),
            "the rehome moves edges, it deletes no node"
        );
        for t in [
            "LINKED_TO",
            "AFFECTS",
            "INCLUDES_ENTITY",
            "DISCUSSED",
            "STEP_IN_PROCESS",
        ] {
            assert!(q.contains(t));
        }
    }

    #[test]
    fn the_purge_only_touches_wl_copies_of_the_project() {
        for q in [purge_wl_files_query(), purge_wl_symbols_query()] {
            assert!(q.contains("DETACH DELETE"));
            assert!(q.contains(WL_SEGMENT));
            assert!(q.contains("$scope"));
        }
        assert!(purge_wl_files_query().contains("(f)-[:CONTAINS]->(s)"));
        for label in SYMBOL_LABELS {
            assert!(purge_wl_files_query().contains(&format!("s:{label}")));
        }
    }

    #[test]
    fn the_wl_twin_strips_one_level_of_scaffolding() {
        let e = wl_twin_expr("w.path", "p.root_path");
        assert!(e.starts_with("p.root_path + '/' + substring(split(w.path, '/.wl/')[1]"));
    }

    #[test]
    fn purge_passes_run_rehome_first() {
        let ids: Vec<&str> = purge_passes().iter().map(|p| p.id).collect();
        assert_eq!(
            ids,
            vec![
                "file-paths/rehome-wl-knowledge",
                "file-paths/purge-wl-files",
                "file-paths/purge-wl-symbols"
            ]
        );
    }

    #[test]
    fn rename_passes_run_files_then_symbols_then_anchors() {
        let ids: Vec<&str> = rename_passes().iter().map(|p| p.id).collect();
        assert_eq!(ids.first(), Some(&"file-paths/rename-files"));
        assert_eq!(ids.last(), Some(&"file-paths/rename-anchors"));
        assert_eq!(ids.len(), 2 + SYMBOL_LABELS.len());
    }

    fn measures(dangling: i64, losing: i64, without: i64, alias: i64) -> PathMeasures {
        PathMeasures {
            dangling_file_anchors: dangling,
            notes_losing_last_anchor: losing,
            notes_without_file_anchor: without,
            alias_files: alias,
            ..Default::default()
        }
    }

    #[test]
    fn the_purge_gate_refuses_on_bad_measures() {
        let before = measures(0, 0, 5320, 815);
        assert!(purge_gate(&before, &measures(0, 0, 5320, 0)).is_ok());
        assert!(
            purge_gate(&before, &measures(1, 0, 5320, 0)).is_err(),
            "dangling anchor"
        );
        assert!(
            purge_gate(&before, &measures(0, 1, 5320, 0)).is_err(),
            "note losing anchor"
        );
        assert!(
            purge_gate(&before, &measures(0, 0, 5321, 0)).is_err(),
            "notes without anchor rose"
        );
        assert!(
            purge_gate(&before, &measures(0, 0, 5320, 3)).is_err(),
            "alias paths remain"
        );
        let reasons = purge_gate(&before, &measures(2, 1, 5320, 0)).unwrap_err();
        assert_eq!(reasons.len(), 2);
    }

    /// Le dry-run n'écrit rien : chaque requête de contrôle est en lecture
    /// seule et nomme les racines par paramètre (calculées en Rust, pour tourner
    /// avant que `canonical_path` soit posé).
    #[test]
    fn every_control_query_is_read_only_and_parameterised() {
        let all = measure::all();
        assert_eq!(all.len(), 15);
        for (name, q) in &all {
            for verb in ["SET ", "CREATE", "MERGE", "DELETE", "REMOVE"] {
                assert!(!q.contains(verb), "{name} must not {verb}");
            }
            assert!(q.contains("AS n"), "{name} returns one count");
            assert!(
                q.contains("$pid") || q.contains("$canonical"),
                "{name} is scoped to the project"
            );
        }
        // Les six de la note 088c0d1a, dans l'ordre.
        for key in [
            "alias_files",
            "declared_files",
            "notes_without_file_anchor",
            "dangling_file_anchors",
            "files_with_uid",
            "incoherent_stable_keys",
        ] {
            assert!(all.iter().any(|(n, _)| *n == key), "{key} is measured");
        }
    }

    fn plan() -> RenamePlan {
        RenamePlan {
            project_id: Uuid::parse_str("00333b5f-2d0a-4467-9c98-155e55d2b7e5").unwrap(),
            slug: "project-orchestrator-backend".into(),
            declared_root: "/Users/triviere/projects/project-orchestrator/backend".into(),
            canonical_root: "/Users/triviere/.openclaw/workspace/skills/project-orchestrator"
                .into(),
            stored_canonical: None,
            paths_declared: false,
        }
    }

    #[test]
    fn a_plan_without_alias_has_nothing_to_rename() {
        let mut p = plan();
        assert!(p.has_alias());
        p.canonical_root = p.declared_root.clone();
        assert!(!p.has_alias());
        assert!(render_plan(&p, &PathMeasures::default(), false).contains("rien à renommer"));
    }

    #[test]
    fn the_dry_run_names_the_counts_and_the_order() {
        let m = PathMeasures {
            alias_files: 815,
            alias_symbols: vec![("Function".into(), 132_111)],
            wl_files: 631,
            wl_symbols: 22_276,
            ..Default::default()
        };
        let out = render_plan(&plan(), &m, false);
        assert!(out.contains("(a) SET File.path sur 815"));
        assert!(out.contains("(b) SET file_path, id sur 132111"));
        assert!(out.contains("(e) File.uid : intact"));
        assert!(out.contains("--purge-wl"));
        assert!(out.contains("sera posé par --apply"));
        let out = render_measures("Mesures avant", &m);
        assert!(out.contains("Function"));
        assert!(out.contains("631 File"));
    }
}
