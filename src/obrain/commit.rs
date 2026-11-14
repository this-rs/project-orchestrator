//! `CommitStore` on obrain — the commit domain: the `Commit` nodes, their
//! `TOUCHES` links to the files they changed, and the `CO_CHANGED` /
//! `CO_CHANGED_TRANSITIVE` edges between files.
//!
//! # What serves (6 of the 22 methods)
//!
//! The commits themselves (`create_commit`, `get_commit`, `delete_commit`), the
//! plan link (`link_commit_to_plan`, `get_plan_commits` — a plan is ported in
//! `plan.rs`), and `update_project_co_change_timestamp` (a property of the
//! project, ported in `project.rs`).
//!
//! # What is refused by name ([`REFUSEES`], 16 methods)
//!
//! - `find_dependent_files`, `find_impacted_files`, `find_callers`: they read the
//!   code graph (`IMPORTS`, `CALLS`, `Function`), which the code domain owns and
//!   has not ported. A port that answered `vec![]` would be a false success.
//! - `link_task_to_files`, `link_commit_to_task`, `link_commit_to_task_from`,
//!   `get_task_commits`: they link or read a `Task`, a domain not ported.
//! - `ping_freshness_for_files`: it writes `freshness_pinged_at` on the `Note`
//!   nodes linked to files; the note domain (`note.rs`) resolves its links only
//!   to `Project` and `Note`, so a `Note → File` link is "unresolved" there, and
//!   pinging such a note would be a write the note domain cannot read back.
//! - The `TOUCHES` / `CO_CHANGED` machinery — `create_commit_touches`,
//!   `get_commit_files`, `get_file_history`, `compute_co_changed`,
//!   `get_co_change_graph`, `get_file_co_changers`,
//!   `compute_co_changed_transitive`, `get_file_transitive_co_changers`: every
//!   one of them stands on `File` nodes, and NO production code writes the
//!   `File` label on obrain (the code domain is not ported; its `upsert_file`
//!   refuses). Served, they would be false successes, and two of them would lose
//!   data for good: `create_commit_touches` would answer `Ok` while dropping
//!   every file (a `TOUCHES` without its `File` is skipped, as Neo4j's `MATCH`),
//!   and `compute_co_changed` would answer `Ok(0)` and stamp
//!   `last_co_change_computed_at`, so the incremental run (`since` = that stamp)
//!   would never replay the commits it skipped. The reads would answer `vec![]`.
//!
//! The code of that machinery is KEPT, whole and tested, behind the refusal
//! (the `*_porte` methods below, exercised by this module's tests through the
//! test hook [`ObrainGraphStore::ecrire_fichier_pour_test`]). Lifting the refusal
//! is: the eight trait methods call their `*_porte`, the eight names leave
//! [`REFUSEES`] — once the coupling below is in place.
//!
//! ## The future coupling with the code domain
//!
//! This domain resolves a path to its `File` node through `fichier_par_chemin`
//! / `fichier_par_noeud`, built AT OPEN by scanning the `File` label. Nothing
//! keeps them current afterwards but the test hook. When `code.rs` writes
//! `File` (create, delete, a path or `project_id` that changes), it must update
//! these two maps in the same write — or this domain must read the code
//! domain's own index instead of keeping its copy. Without that, a file created
//! after open is invisible here until the next restart.
//!
//! # The `CO_CHANGED` edge — the schema the other domains consume
//!
//! This is the decision the following lots (chat, code) read. A `CO_CHANGED`
//! edge links two `File` nodes and carries the co-change count.
//!
//! - **Endpoints.** Two `File` nodes. The `File` node is OWNED and WRITTEN by the
//!   code domain (`File {path, language, hash, last_parsed, project_id}`); this
//!   domain only READS `File` nodes by `path`, exactly as Neo4j does
//!   (`MATCH (file:File {path})`). The two labels must stay the one string
//!   `"File"` (`ETIQUETTE_FICHIER`).
//! - **Direction.** ONE directed edge per UNORDERED pair, with
//!   `src = File(path_a)`, `dst = File(path_b)` where `path_a < path_b`
//!   (byte-wise lexicographic). This collapses Neo4j's bidirectional relation to
//!   a single canonical edge, so `count_co_change_pairs` is a `.len()` and no
//!   pair is double-counted.
//! - **Properties.**
//!   - `project_id` (string, the `Uuid`): which project the co-change belongs to.
//!   - `count` (int64): the number of commits in which both files appeared. It is
//!     INCREMENTAL: `compute_co_changed` ADDS the co-occurrences of its window to
//!     the existing count, exactly as Neo4j's `coalesce(r.count, 0) + co_count`.
//!   - `last_at` (string, RFC 3339 in UTC with a `Z`, as Neo4j's
//!     `toString(datetime())`): when the edge was last updated ("now", not a
//!     commit's date).
//!
//! # The `CO_CHANGED_TRANSITIVE` edge
//!
//! Same endpoints and the same canonical direction. Properties: `project_id`
//! (string), `score` (float64, the product of the normalized `count` along the
//! path), `depth` (int64, 2 or 3), `via` (list of strings, the intermediate file
//! paths, from `path_a` to `path_b`), `computed_at` (string, RFC 3339 with `Z`).
//! Re-computed in full on each call: the previous edges of the project are
//! deleted first, ALWAYS — also when the project has no `CO_CHANGED` edge left,
//! as Neo4j's cleanup does.
//!
//! # Divergences from Neo4j, named
//!
//! - A commit whose `hash`/`message`/`author`/`timestamp` is missing, or whose
//!   `timestamp` does not parse, is quarantined at open; Neo4j reads a bad
//!   timestamp back as "now". Its hash stays RESERVED (as in `note.rs`): a
//!   second node of a hash quarantines the hash too, and `get_commit`,
//!   `create_commit`, `delete_commit` and `link_commit_to_plan` of a reserved
//!   hash refuse with its name — never "absent", never a duplicate node, never
//!   a deletion that leaves the quarantined node behind; `get_plan_commits`
//!   omits it. `create_commit` is a MERGE on `hash` as Neo4j's is,
//!   so a commit registered twice is rewritten, never refused and never
//!   duplicated (`tests/store_conformance.rs::check_commit_upsert_on_hash`).
//! - **Pair counting — obrain corrects a Neo4j defect.** Neo4j's query pairs
//!   `files[i]`, `files[j]` (`i < j`, in `collect()` order) and then keeps
//!   `WHERE f1.path < f2.path`: a pair collected in the other order is DROPPED,
//!   so Neo4j under-counts by an amount that depends on the engine's collect
//!   order. obrain canonicalizes every pair and counts each co-occurrence once.
//! - **The key is `(project_id, path_a, path_b)`**, where Neo4j MERGEs on the
//!   pair alone and lets `project_id` be the last writer's. Two projects that
//!   co-change the same pair get two edges, never one overwritten. The
//!   project-scoped reads (`get_co_change_graph`) see one; the path-scoped reads
//!   (`get_file_co_changers`, `get_file_transitive_co_changers`, not scoped by
//!   project, as in Neo4j) would see the other path twice: they are
//!   deduplicated by path, keeping the strongest (`count`, then the latest
//!   `last_at`; the highest `score` for the transitive one).
//! - `compute_co_changed` only counts `File` nodes whose `project_id` property
//!   equals the target project — Neo4j's `MATCH (f:File {project_id})` — and it
//!   silently skips the `TOUCHES` target that has no `File` node (Neo4j's `MATCH`,
//!   not `MERGE`, on the file). It never creates a `File` node. The project
//!   timestamp is written in the SAME write as the edges (Neo4j: a separate
//!   query after them).
//! - `max_relations` caps by `(count DESC, path_a, path_b)`; the "cleanup" that
//!   Neo4j wraps in a non-fatal warning is, here, part of the same durable write.
//!   A NEGATIVE `max_relations` caps nothing: Neo4j's `SKIP -1` fails, the
//!   warning swallows it, and no edge is deleted — the same outcome here.
//! - **The transitive walk is restricted to the project**: it follows only the
//!   project's `CO_CHANGED` edges and skips a pair with a direct edge in the
//!   project. Neo4j starts from a `File` of the project but walks any
//!   `CO_CHANGED` edge and checks a direct edge of any project.
//! - **`via` is deterministic.** Among the paths of the best score, obrain keeps
//!   the smallest `via` (lexicographic); Neo4j's `head(collect(via_nodes))` takes
//!   whichever path the engine met first, and not even one of the best score.
//! - **A negative `limit`** on a read answers an empty list; Neo4j refuses a
//!   negative `LIMIT` (a Cypher error).
//! - `create_commit_touches` sets `additions` and `deletions` each on its own: a
//!   `deletions >= 0` is written even when `additions` is absent. Neo4j's
//!   `WITH r, f WHERE f.additions >= 0 … SET r.deletions` chain drops the
//!   `deletions` of a file whose `additions` is absent.
//!
//! # Storage, reads, atomicity
//!
//! A commit is one `Commit` node (`hash`, `message`, `author`, `timestamp`), with
//! the draft marker `_brouillon` written first and removed last, as in
//! `project.rs`; a creation killed half-way is deleted at open. A REWRITE of an
//! existing commit has no marker: each property is its own engine write, so a
//! crash in the middle may leave some of the new values next to some of the
//! old ones (each value whole). The commit still reads, and the next
//! registration of the same history rewrites it.
//!
//! `RESULTED_IN`, `TOUCHES`, `CO_CHANGED` and `CO_CHANGED_TRANSITIVE` are engine
//! edges; they have no marker, so an interruption can leave a subset of them (as
//! in `note.rs`): the next call re-runs. Every read is served from the
//! in-memory index — never from the engine directly, which shows a write before
//! it is durable — decoded from the base at open and published after each
//! durable write. The index holds the commits, the `plan → commits` and
//! `commit → TOUCHES` maps, a `path → File` map (see the coupling above), and
//! the two co-change maps keyed by `(project_id, path_a, path_b)`.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, SecondsFormat, Utc};
use obrain_common::types::{EdgeId, NodeId, Value};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::project::poser_horodatage;
use super::{Ecriture, NonPorte, ObrainGraphStore, RapportOuverture, SEUIL_COLONNE_BLOB};
use crate::neo4j::models::*;
use crate::neo4j::traits::CommitStore;

const LABEL: &str = "Commit";
/// The `File` label, owned and written by the code domain; this domain reads it.
const ETIQUETTE_FICHIER: &str = "File";
const TOUCHE: &str = "TOUCHES";
const CO_CHANGE: &str = "CO_CHANGED";
const CO_CHANGE_TRANSITIF: &str = "CO_CHANGED_TRANSITIVE";
const RESULTAT: &str = "RESULTED_IN";
const BROUILLON: &str = "_brouillon";

/// The methods of `CommitStore` refused on every call, as `CommitStore::<method>`.
pub const REFUSEES: &[&str] = &[
    "CommitStore::find_dependent_files",
    "CommitStore::find_impacted_files",
    "CommitStore::find_callers",
    "CommitStore::link_task_to_files",
    "CommitStore::link_commit_to_task",
    "CommitStore::link_commit_to_task_from",
    "CommitStore::get_task_commits",
    "CommitStore::ping_freshness_for_files",
    // The TOUCHES / CO_CHANGED machinery: no production code writes `File`.
    "CommitStore::create_commit_touches",
    "CommitStore::get_commit_files",
    "CommitStore::get_file_history",
    "CommitStore::compute_co_changed",
    "CommitStore::get_co_change_graph",
    "CommitStore::get_file_co_changers",
    "CommitStore::compute_co_changed_transitive",
    "CommitStore::get_file_transitive_co_changers",
];

/// Why the TOUCHES / CO_CHANGED machinery is refused (see the module doc).
const SANS_FICHIER: &str = "s'appuie sur les nœuds File, qu'aucun code de production n'écrit sur \
     obrain (domaine code non porté) : servie, elle répondrait Ok en jetant les fichiers, ou une \
     liste vide — une fausse réussite. Son code est prêt (commit.rs), derrière ce refus";

#[cfg_attr(not(test), allow(dead_code))]
/// "Now", as Neo4j's `toString(datetime())` writes it: RFC 3339, UTC, `Z`.
fn maintenant() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::AutoSi, true)
}

/// Does this value go to a blob column (a string above the threshold, or a
/// string inside a list — as in `note.rs`)?
fn va_en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
        || v.as_list().is_some_and(|l| {
            l.iter()
                .any(|e| e.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB))
        })
}

/// A list of strings with one above the blob threshold.
#[cfg_attr(not(test), allow(dead_code))]
fn liste_en_colonne_blob(l: &[String]) -> bool {
    l.iter().any(|s| s.len() > SEUIL_COLONNE_BLOB)
}

/// The refusal of a hash reserved by a quarantined node (see `IndexCommits::reserves`).
fn refus_reserve<T>(methode: &'static str, hash: &str) -> Result<T> {
    // Not `NonPorte`: the method IS ported; this hash is what it refuses.
    Err(anyhow!(
        "{methode} : le hash {hash} est réservé : un nœud Commit de ce hash est sur disque, en quarantaine \
         depuis l'ouverture (illisible, ou second nœud du même hash). Il n'est ni lu, ni repris, \
         ni écrasé, ni supprimé : réparer la base à la main"
    ))
}

/// Refuses a method of the commit domain that obrain does not port, by name.
fn refus<T>(methode: &'static str, raison: &'static str) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).context(raison))
}

/// The domain value of a `Commit` node from a property lookup.
fn lire_commit(get: &dyn Fn(&str) -> Option<Value>) -> Result<CommitNode> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let hash = chaine("hash").ok_or_else(|| anyhow!("nœud Commit sans propriété `hash`"))?;
    let requise =
        |cle: &str| chaine(cle).ok_or_else(|| anyhow!("commit {hash} sans propriété `{cle}`"));
    let message = requise("message")?;
    let author = requise("author")?;
    let timestamp = requise("timestamp")?
        .parse::<DateTime<Utc>>()
        .map_err(|e| anyhow!("commit {hash} : `timestamp` illisible : {e}"))?;
    Ok(CommitNode {
        hash,
        message,
        author,
        timestamp,
    })
}

/// The canonical pair: `path_a < path_b`.
fn canon(a: &str, b: &str) -> (String, String) {
    if a < b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// One commit: where it lives in the base, and its decoded value.
pub(super) struct EntreeCommit {
    noeud: NodeId,
    commit: CommitNode,
}

impl EntreeCommit {
    fn decoder(db: &ObrainDB, noeud: NodeId) -> Result<Self> {
        let n = db
            .get_node(noeud)
            .ok_or_else(|| anyhow!("nœud Commit {noeud:?} indexé mais absent de la base"))?;
        let get = |cle: &str| {
            n.get_property(cle)
                .filter(|v| !matches!(v, Value::Null))
                .cloned()
        };
        Ok(Self {
            noeud,
            commit: lire_commit(&get)?,
        })
    }
}

/// A `File` node, as this domain reads it (written by the code domain).
struct EntreeFichier {
    noeud: NodeId,
    project_id: Option<Uuid>,
}

/// A `TOUCHES` edge of a commit, to the file of its key.
#[derive(Clone)]
struct Touche {
    arete: EdgeId,
    additions: Option<i64>,
    deletions: Option<i64>,
}

/// A `CO_CHANGED` edge, canonical (`path_a < path_b`).
struct EntreeCoChangee {
    arete: EdgeId,
    path_a: String,
    path_b: String,
    count: i64,
    last_at: Option<String>,
}

/// A `CO_CHANGED_TRANSITIVE` edge, canonical (`path_a < path_b`).
struct EntreeCoChangeeTransitive {
    arete: EdgeId,
    path_a: String,
    path_b: String,
    score: f64,
    depth: i64,
    via: Vec<String>,
}

/// In-memory indexes of the commit domain, rebuilt at open.
#[derive(Default)]
pub(super) struct IndexCommits {
    par_hash: HashMap<String, EntreeCommit>,
    /// The reverse map: engine node → hash.
    par_noeud: HashMap<NodeId, String>,
    /// `RESULTED_IN`: plan node → the hashes of its commits.
    resultats: HashMap<NodeId, HashSet<String>>,
    /// `TOUCHES`: commit hash → file path → the edge.
    touches: HashMap<String, HashMap<String, Touche>>,
    /// The hashes of the `Commit` nodes quarantined at open (unreadable, or a
    /// second node of a hash): RESERVED. `get_commit`, `create_commit` and
    /// `delete_commit` of such a hash refuse by name — never "absent", never a
    /// second node, never a deletion that leaves the quarantined node behind —
    /// until the base is repaired by hand. As `note.rs` and `chat.rs`.
    reserves: HashSet<String>,
    fichier_par_chemin: HashMap<String, EntreeFichier>,
    /// The reverse map: engine node → path (for the edge reads).
    fichier_par_noeud: HashMap<NodeId, String>,
    co_changees: HashMap<(Uuid, String, String), EntreeCoChangee>,
    co_changees_transitives: HashMap<(Uuid, String, String), EntreeCoChangeeTransitive>,
}

impl IndexCommits {
    /// Rebuild the index from the persisted nodes and edges at open.
    pub(super) fn reconstruire(db: &ObrainDB, rapport: &mut RapportOuverture) -> Result<Self> {
        let store = db.store();
        let mut index = Self::default();

        // Commits.
        for noeud in store.nodes_by_label(LABEL) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud Commit {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            if get(BROUILLON).is_some() || (n.properties.is_empty() && store.out_degree(noeud) == 0)
            {
                for (_, arete) in store.edges_from(noeud, Direction::Outgoing) {
                    db.delete_edge(arete);
                }
                for (_, arete) in store.edges_from(noeud, Direction::Incoming) {
                    db.delete_edge(arete);
                }
                db.delete_node(noeud);
                rapport.brouillons_supprimes += 1;
                continue;
            }
            let commit = match lire_commit(&get) {
                Ok(c) => c,
                Err(e) => {
                    rapport
                        .en_quarantaine
                        .push(format!("nœud Commit {noeud:?} : {e}"));
                    if let Some(hash) = get("hash").and_then(|v| v.as_str().map(str::to_string)) {
                        index.reserves.insert(hash);
                    }
                    continue;
                }
            };
            if index.par_hash.contains_key(&commit.hash) {
                rapport.en_quarantaine.push(format!(
                    "nœud Commit {noeud:?} : hash {} déjà pris",
                    commit.hash
                ));
                index.reserves.insert(commit.hash.clone());
                continue;
            }
            for (source, arete) in store.edges_from(noeud, Direction::Incoming) {
                if db
                    .get_edge(arete)
                    .is_some_and(|e| e.edge_type.as_str() == RESULTAT)
                {
                    index
                        .resultats
                        .entry(source)
                        .or_default()
                        .insert(commit.hash.clone());
                }
            }
            index.par_noeud.insert(noeud, commit.hash.clone());
            index
                .par_hash
                .insert(commit.hash.clone(), EntreeCommit { noeud, commit });
        }

        // A reserved hash is not served, even when one of its nodes decoded: two
        // nodes of one hash are ambiguous, and a quarantined one must not be
        // shadowed by a readable twin.
        for hash in &index.reserves {
            if let Some(e) = index.par_hash.remove(hash) {
                index.par_noeud.remove(&e.noeud);
            }
            for hashes in index.resultats.values_mut() {
                hashes.remove(hash);
            }
        }

        // Files (owned by the code domain; read here).
        for noeud in store.nodes_by_label(ETIQUETTE_FICHIER) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud File {noeud:?} : illisible"));
                continue;
            };
            let chaine = |cle: &str| {
                n.get_property(cle)
                    .and_then(|v| v.as_str().map(str::to_string))
                    .filter(|s| !s.is_empty())
            };
            let Some(path) = chaine("path") else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud File {noeud:?} : sans `path`"));
                continue;
            };
            if index.fichier_par_chemin.contains_key(&path) {
                rapport
                    .en_quarantaine
                    .push(format!("nœud File {noeud:?} : chemin {} déjà pris", path));
                continue;
            }
            let project_id = chaine("project_id").and_then(|s| s.parse::<Uuid>().ok());
            index.fichier_par_noeud.insert(noeud, path.clone());
            index
                .fichier_par_chemin
                .insert(path, EntreeFichier { noeud, project_id });
        }

        // TOUCHES edges: the out-edges of the commits, to a known file.
        for (hash, entree) in &index.par_hash {
            for (cible, arete) in store.edges_from(entree.noeud, Direction::Outgoing) {
                let Some(e) = db.get_edge(arete) else {
                    continue;
                };
                if e.edge_type.as_str() != TOUCHE {
                    continue;
                }
                let Some(path) = index.fichier_par_noeud.get(&cible) else {
                    continue;
                };
                let entier = |cle: &str| e.get_property(cle).and_then(|v| v.as_int64());
                index.touches.entry(hash.clone()).or_default().insert(
                    path.clone(),
                    Touche {
                        arete,
                        additions: entier("additions"),
                        deletions: entier("deletions"),
                    },
                );
            }
        }

        // CO_CHANGED / CO_CHANGED_TRANSITIVE edges: the out-edges of the files.
        for (noeud, path) in &index.fichier_par_noeud.clone() {
            for (cible, arete) in store.edges_from(*noeud, Direction::Outgoing) {
                let Some(e) = db.get_edge(arete) else {
                    continue;
                };
                let ty = e.edge_type.as_str();
                if ty != CO_CHANGE && ty != CO_CHANGE_TRANSITIF {
                    continue;
                }
                let Some(autre) = index.fichier_par_noeud.get(&cible) else {
                    continue;
                };
                let (path_a, path_b) = canon(path, autre);
                let texte = |cle: &str| {
                    e.get_property(cle)
                        .and_then(|v| v.as_str().map(str::to_string))
                };
                let entier = |cle: &str| e.get_property(cle).and_then(|v| v.as_int64());
                let Some(project_id) = texte("project_id").and_then(|s| s.parse::<Uuid>().ok())
                else {
                    rapport
                        .en_quarantaine
                        .push(format!("arête {ty} {arete:?} : `project_id` illisible"));
                    continue;
                };
                if ty == CO_CHANGE {
                    let Some(count) = entier("count") else {
                        rapport
                            .en_quarantaine
                            .push(format!("arête CO_CHANGED {arete:?} : sans `count`"));
                        continue;
                    };
                    let cle = (project_id, path_a.clone(), path_b.clone());
                    if index.co_changees.contains_key(&cle) {
                        rapport
                            .en_quarantaine
                            .push(format!("arête CO_CHANGED {arete:?} : paire déjà prise"));
                        continue;
                    }
                    index.co_changees.insert(
                        cle,
                        EntreeCoChangee {
                            arete,
                            path_a,
                            path_b,
                            count,
                            last_at: texte("last_at"),
                        },
                    );
                } else {
                    let score = e
                        .get_property("score")
                        .and_then(|v| v.as_float64())
                        .unwrap_or(0.0);
                    let depth = entier("depth").unwrap_or(2);
                    let via = e
                        .get_property("via")
                        .and_then(|v| v.as_list())
                        .map(|l| {
                            l.iter()
                                .filter_map(|x| x.as_str().map(str::to_string))
                                .collect()
                        })
                        .unwrap_or_default();
                    let cle = (project_id, path_a.clone(), path_b.clone());
                    if index.co_changees_transitives.contains_key(&cle) {
                        rapport.en_quarantaine.push(format!(
                            "arête CO_CHANGED_TRANSITIVE {arete:?} : paire déjà prise"
                        ));
                        continue;
                    }
                    index.co_changees_transitives.insert(
                        cle,
                        EntreeCoChangeeTransitive {
                            arete,
                            path_a,
                            path_b,
                            score,
                            depth,
                            via,
                        },
                    );
                }
            }
        }

        Ok(index)
    }
}

#[cfg(any(test, feature = "mock-store"))]
impl ObrainGraphStore {
    /// Test hook: write a `File` node the way the code domain will (`File`,
    /// `path`, `project_id`), publishing it to the commit index. The commit
    /// domain only READS `File` nodes; this lets its tests and the end-to-end
    /// proof exercise TOUCHES and CO_CHANGED before the code domain is linked.
    /// Absent from production builds.
    #[doc(hidden)]
    pub async fn ecrire_fichier_pour_test(&self, chemin: &str, projet: Option<Uuid>) -> Result<()> {
        let chemin = chemin.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud = w.db.create_node(&[ETIQUETTE_FICHIER]);
                w.db.set_node_property(noeud, "path", Value::from(chemin.clone()));
                w.db.set_node_property(
                    noeud,
                    "project_id",
                    Value::from(projet.map(|p| p.to_string()).unwrap_or_default()),
                );
                w.rendre_durable()?;
                w.index_mut()?
                    .commits
                    .fichier_par_noeud
                    .insert(noeud, chemin.clone());
                w.index_mut()?.commits.fichier_par_chemin.insert(
                    chemin,
                    EntreeFichier {
                        noeud,
                        project_id: projet,
                    },
                );
                Ok(())
            },
        )
        .await
    }
}

// The refusals keep their parameters' names for the reader; they are unused by design.
#[allow(unused_variables)]
#[async_trait]
impl CommitStore for ObrainGraphStore {
    async fn find_dependent_files(
        &self,
        file_path: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> Result<Vec<String>> {
        refus(
            "CommitStore::find_dependent_files",
            "lit les arêtes IMPORTS/CONTAINS du graphe de code, domaine non porté ; une liste vide serait une fausse réussite",
        )
    }

    async fn find_impacted_files(
        &self,
        file_path: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> Result<Vec<String>> {
        refus(
            "CommitStore::find_impacted_files",
            "lit les arêtes IMPORTS et CALLS du graphe de code, domaine non porté ; une liste vide serait une fausse réussite",
        )
    }

    async fn find_callers(
        &self,
        function_id: &str,
        project_id: Option<Uuid>,
    ) -> Result<Vec<FunctionNode>> {
        refus(
            "CommitStore::find_callers",
            "lit les nœuds Function et les arêtes CALLS du graphe de code, domaine non porté",
        )
    }

    async fn link_task_to_files(&self, task_id: Uuid, file_paths: &[String]) -> Result<()> {
        refus(
            "CommitStore::link_task_to_files",
            "lie une tâche à des fichiers : le domaine Task n'est pas porté",
        )
    }

    async fn create_commit(&self, commit: &CommitNode) -> Result<()> {
        let c = commit.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let proprietes = vec![
                    ("hash", Value::from(c.hash.clone())),
                    ("message", Value::from(c.message.clone())),
                    ("author", Value::from(c.author.clone())),
                    ("timestamp", Value::from(c.timestamp.to_rfc3339())),
                ];
                // `create_commit` is a MERGE on `hash`, as Neo4j's (`MERGE
                // (c:Commit {hash}) SET ...`) and the mock's `insert`: re-registering
                // a commit that is already there rewrites its three values. It must
                // not fail — a re-sync of the same history stops on the first commit
                // it meets twice — and it must not make a second node, or
                // `get_commit` answers whichever one it meets first.
                let (existant, ancien_en_blob) = {
                    let etat = w.index()?;
                    if etat.commits.reserves.contains(&c.hash) {
                        return refus_reserve("CommitStore::create_commit", &c.hash);
                    }
                    match etat.commits.par_hash.get(&c.hash) {
                        Some(e) => (
                            Some(e.noeud),
                            // Replacing a value that WAS in a blob column clears
                            // that slot, which the WAL does not recover either: the
                            // old value counts too.
                            e.commit.message.len() > SEUIL_COLONNE_BLOB
                                || e.commit.author.len() > SEUIL_COLONNE_BLOB,
                        ),
                        None => (None, false),
                    }
                };
                w.flush_complet =
                    ancien_en_blob || proprietes.iter().any(|(_, v)| va_en_colonne_blob(v));
                if let Some(noeud) = existant {
                    // A rewrite has no draft marker to lean on, and the properties
                    // are separate engine writes: a crash between them can leave
                    // some new values next to some old ones (each whole). The
                    // commit still reads; the next registration rewrites it.
                    for (cle, valeur) in proprietes {
                        w.db.set_node_property(noeud, cle, valeur);
                    }
                    w.rendre_durable()?;
                    let entree = w.apres_ecriture(EntreeCommit::decoder(w.db, noeud))?;
                    w.index_mut()?
                        .commits
                        .par_hash
                        .insert(c.hash.clone(), entree);
                    return Ok(());
                }
                let noeud = w.db.create_node(&[LABEL]);
                w.db.set_node_property(noeud, BROUILLON, Value::from(true));
                for (cle, valeur) in proprietes {
                    w.db.set_node_property(noeud, cle, valeur);
                }
                w.rendre_durable_avant_le_marqueur()?;
                w.db.remove_node_property(noeud, BROUILLON);
                let entree = w.apres_ecriture(EntreeCommit::decoder(w.db, noeud))?;
                w.rendre_durable()?;
                let hash = c.hash.clone();
                w.index_mut()?.commits.par_noeud.insert(noeud, hash.clone());
                w.index_mut()?.commits.par_hash.insert(hash, entree);
                Ok(())
            },
        )
        .await
    }

    async fn get_commit(&self, hash: &str) -> Result<Option<CommitNode>> {
        self.lire(|_, etat| {
            if etat.commits.reserves.contains(hash) {
                return refus_reserve("CommitStore::get_commit", hash);
            }
            Ok(etat.commits.par_hash.get(hash).map(|e| e.commit.clone()))
        })
    }

    async fn link_commit_to_task(&self, commit_hash: &str, task_id: Uuid) -> Result<()> {
        refus(
            "CommitStore::link_commit_to_task",
            "lie un commit à une tâche (RESOLVED_BY) : le domaine Task n'est pas porté",
        )
    }

    async fn link_commit_to_task_from(
        &self,
        commit_hash: &str,
        task_id: Uuid,
        source: &str,
    ) -> Result<()> {
        refus(
            "CommitStore::link_commit_to_task_from",
            "lie un commit à une tâche avec provenance (RESOLVED_BY) : le domaine Task n'est pas porté",
        )
    }

    async fn link_commit_to_plan(&self, commit_hash: &str, plan_id: Uuid) -> Result<()> {
        let hash = commit_hash.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (plan_noeud, commit_noeud) = {
                    let etat = w.index()?;
                    let Some(pn) = etat.plans.noeud_de(plan_id) else {
                        return Ok(());
                    };
                    if etat.commits.reserves.contains(&hash) {
                        return refus_reserve("CommitStore::link_commit_to_plan", &hash);
                    }
                    let Some(cn) = etat.commits.par_hash.get(&hash).map(|e| e.noeud) else {
                        return Ok(());
                    };
                    // MERGE is idempotent: an existing RESULTED_IN edge is kept.
                    if etat
                        .commits
                        .resultats
                        .get(&pn)
                        .is_some_and(|h| h.contains(&hash))
                    {
                        return Ok(());
                    }
                    (pn, cn)
                };
                w.db.create_edge(plan_noeud, commit_noeud, RESULTAT);
                w.rendre_durable()?;
                w.index_mut()?
                    .commits
                    .resultats
                    .entry(plan_noeud)
                    .or_default()
                    .insert(hash);
                Ok(())
            },
        )
        .await
    }

    async fn get_task_commits(&self, task_id: Uuid) -> Result<Vec<CommitNode>> {
        refus(
            "CommitStore::get_task_commits",
            "lit les commits d'une tâche (RESOLVED_BY) : le domaine Task n'est pas porté",
        )
    }

    async fn get_plan_commits(&self, plan_id: Uuid) -> Result<Vec<CommitNode>> {
        self.lire(|_, etat| {
            let Some(plan_noeud) = etat.plans.noeud_de(plan_id) else {
                return Ok(Vec::new());
            };
            let mut commits: Vec<CommitNode> = etat
                .commits
                .resultats
                .get(&plan_noeud)
                .into_iter()
                .flatten()
                .filter_map(|hash| etat.commits.par_hash.get(hash))
                .map(|e| e.commit.clone())
                .collect();
            commits.sort_by(|a, b| {
                b.timestamp
                    .cmp(&a.timestamp)
                    .then_with(|| b.hash.cmp(&a.hash))
            });
            Ok(commits)
        })
    }

    async fn delete_commit(&self, hash: &str) -> Result<()> {
        let hash = hash.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud = {
                    let etat = w.index()?;
                    if etat.commits.reserves.contains(&hash) {
                        return refus_reserve("CommitStore::delete_commit", &hash);
                    }
                    let Some(n) = etat.commits.par_hash.get(&hash).map(|e| e.noeud) else {
                        return Ok(());
                    };
                    n
                };
                // DETACH DELETE: every edge of the commit goes with it.
                let aretes: Vec<EdgeId> =
                    w.db.store()
                        .edges_from(noeud, Direction::Outgoing)
                        .into_iter()
                        .chain(w.db.store().edges_from(noeud, Direction::Incoming))
                        .map(|(_, a)| a)
                        .collect();
                for a in aretes {
                    w.db.delete_edge(a);
                }
                if !w.db.delete_node(noeud) {
                    // The edges are already gone: what is on disk is no longer the
                    // state before this call. Same contract as `note.rs`: everything
                    // is refused until a restart rebuilds the index from the base.
                    w.interne
                        .indetermine
                        .store(true, std::sync::atomic::Ordering::Release);
                    bail!(
                        "obrain a refusé la suppression du nœud {noeud:?} (commit {hash}) APRÈS \
                         la suppression de ses arêtes : base indéterminée, redémarrer"
                    );
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                let idx = &mut etat.commits;
                idx.par_hash.remove(&hash);
                idx.par_noeud.remove(&noeud);
                idx.touches.remove(&hash);
                for hashes in idx.resultats.values_mut() {
                    hashes.remove(&hash);
                }
                Ok(())
            },
        )
        .await
    }

    async fn create_commit_touches(
        &self,
        commit_hash: &str,
        files: &[FileChangedInfo],
    ) -> Result<()> {
        refus("CommitStore::create_commit_touches", SANS_FICHIER)
    }

    async fn get_commit_files(&self, commit_hash: &str) -> Result<Vec<CommitFileInfo>> {
        refus("CommitStore::get_commit_files", SANS_FICHIER)
    }

    async fn get_file_history(
        &self,
        file_path: &str,
        limit: Option<i64>,
    ) -> Result<Vec<FileHistoryEntry>> {
        refus("CommitStore::get_file_history", SANS_FICHIER)
    }

    async fn ping_freshness_for_files(&self, file_paths: &[String]) -> Result<usize> {
        refus(
            "CommitStore::ping_freshness_for_files",
            "pingerait des notes liées à des fichiers ; le domaine Note ne résout pas (encore) ses liens vers File, et pinger une note qu'il ne sait pas relire serait une écriture aveugle",
        )
    }

    async fn compute_co_changed(
        &self,
        project_id: Uuid,
        since: Option<DateTime<Utc>>,
        min_count: i64,
        max_relations: i64,
    ) -> Result<i64> {
        refus(
            "CommitStore::compute_co_changed",
            "s'appuie sur les nœuds File, qu'aucun code de production n'écrit sur obrain : \
             servie, elle répondrait Ok(0) et horodaterait last_co_change_computed_at, et le \
             calcul incrémental ne rejouerait jamais les commits sautés. Son code est prêt \
             (commit.rs), derrière ce refus",
        )
    }

    async fn update_project_co_change_timestamp(&self, id: Uuid) -> Result<()> {
        self.horodater(id, "last_co_change_computed_at").await
    }

    async fn get_co_change_graph(
        &self,
        project_id: Uuid,
        min_count: i64,
        limit: i64,
    ) -> Result<Vec<CoChangePair>> {
        refus("CommitStore::get_co_change_graph", SANS_FICHIER)
    }

    async fn get_file_co_changers(
        &self,
        file_path: &str,
        min_count: i64,
        limit: i64,
    ) -> Result<Vec<CoChanger>> {
        refus("CommitStore::get_file_co_changers", SANS_FICHIER)
    }

    async fn compute_co_changed_transitive(
        &self,
        project_id: Uuid,
        max_depth: i64,
        min_score: f64,
    ) -> Result<i64> {
        refus("CommitStore::compute_co_changed_transitive", SANS_FICHIER)
    }

    async fn get_file_transitive_co_changers(
        &self,
        file_path: &str,
        min_score: f64,
        limit: i64,
    ) -> Result<Vec<TransitiveCoChanger>> {
        refus("CommitStore::get_file_transitive_co_changers", SANS_FICHIER)
    }
}

#[cfg_attr(not(test), allow(dead_code))]
/// The new pair `(score, via)` beats the kept one: a higher score, or the same
/// score and a smaller `via` — so the edge written does not depend on the order
/// the walk meets the paths in.
fn meilleur(score: f64, via: &[String], garde: &(f64, i64, Vec<String>)) -> bool {
    score > garde.0 || (score == garde.0 && via < garde.2.as_slice())
}

/// The TOUCHES / CO_CHANGED machinery, kept whole BEHIND the refusal (module
/// doc): the trait methods refuse by name until the code domain writes `File`.
/// This module's tests run it directly. Lifting the refusal = the trait methods
/// call these, and their names leave [`REFUSEES`].
#[cfg_attr(not(test), allow(dead_code))]
impl ObrainGraphStore {
    async fn create_commit_touches_porte(
        &self,
        commit_hash: &str,
        files: &[FileChangedInfo],
    ) -> Result<()> {
        if files.is_empty() {
            return Ok(());
        }
        let hash = commit_hash.to_string();
        let fichiers: Vec<FileChangedInfo> = files.to_vec();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // A path above the blob threshold (`file_path` of the edge) is
                // not recovered by the WAL: the full flush. The value it replaces
                // on an existing edge is the same path.
                w.flush_complet = fichiers.iter().any(|f| f.path.len() > SEUIL_COLONNE_BLOB);
                // Everything read from the index, before any write.
                let (commit_noeud, noeuds, mut touches) = {
                    let etat = w.index()?;
                    let Some(cn) = etat.commits.par_hash.get(&hash).map(|e| e.noeud) else {
                        return Ok(());
                    };
                    let noeuds: HashMap<String, NodeId> = fichiers
                        .iter()
                        .filter_map(|f| {
                            etat.commits
                                .fichier_par_chemin
                                .get(&f.path)
                                .map(|e| (f.path.clone(), e.noeud))
                        })
                        .collect();
                    let touches = etat.commits.touches.get(&hash).cloned().unwrap_or_default();
                    (cn, noeuds, touches)
                };
                for f in &fichiers {
                    let Some(fichier) = noeuds.get(&f.path).copied() else {
                        continue; // a file without a node is skipped, as in Neo4j
                    };
                    let touche = match touches.entry(f.path.clone()) {
                        Entry::Occupied(o) => o.into_mut(),
                        Entry::Vacant(v) => v.insert(Touche {
                            arete: w.db.create_edge(commit_noeud, fichier, TOUCHE),
                            additions: None,
                            deletions: None,
                        }),
                    };
                    w.db.set_edge_property(touche.arete, "file_path", Value::from(f.path.clone()));
                    // Each on its own (module doc: Neo4j drops `deletions` when
                    // `additions` is absent).
                    if let Some(a) = f.additions.filter(|a| *a >= 0) {
                        w.db.set_edge_property(touche.arete, "additions", Value::from(a));
                        touche.additions = Some(a);
                    }
                    if let Some(d) = f.deletions.filter(|d| *d >= 0) {
                        w.db.set_edge_property(touche.arete, "deletions", Value::from(d));
                        touche.deletions = Some(d);
                    }
                }
                w.rendre_durable()?;
                if !touches.is_empty() {
                    w.index_mut()?.commits.touches.insert(hash, touches);
                }
                Ok(())
            },
        )
        .await
    }

    fn get_commit_files_porte(&self, commit_hash: &str) -> Result<Vec<CommitFileInfo>> {
        self.lire(|_, etat| {
            let mut fichiers: Vec<CommitFileInfo> = etat
                .commits
                .touches
                .get(commit_hash)
                .into_iter()
                .flatten()
                .map(|(path, t)| CommitFileInfo {
                    path: path.clone(),
                    additions: t.additions,
                    deletions: t.deletions,
                })
                .collect();
            fichiers.sort_by(|a, b| a.path.cmp(&b.path));
            Ok(fichiers)
        })
    }

    fn get_file_history_porte(
        &self,
        file_path: &str,
        limit: Option<i64>,
    ) -> Result<Vec<FileHistoryEntry>> {
        let limite = limit.unwrap_or(50);
        self.lire(|_, etat| {
            let mut historique: Vec<FileHistoryEntry> = etat
                .commits
                .touches
                .iter()
                .filter_map(|(hash, fichiers)| {
                    let t = fichiers.get(file_path)?;
                    let c = &etat.commits.par_hash.get(hash)?.commit;
                    Some(FileHistoryEntry {
                        hash: c.hash.clone(),
                        message: c.message.clone(),
                        author: c.author.clone(),
                        timestamp: c.timestamp,
                        additions: t.additions,
                        deletions: t.deletions,
                    })
                })
                .collect();
            historique.sort_by(|a, b| {
                b.timestamp
                    .cmp(&a.timestamp)
                    .then_with(|| b.hash.cmp(&a.hash))
            });
            historique.truncate(limite.max(0) as usize);
            Ok(historique)
        })
    }

    async fn compute_co_changed_porte(
        &self,
        project_id: Uuid,
        since: Option<DateTime<Utc>>,
        min_count: i64,
        max_relations: i64,
    ) -> Result<i64> {
        let since = since.unwrap_or(DateTime::UNIX_EPOCH);
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // 1. Co-occurrences, from the index (read lock, in memory). Each
                //    pair is canonicalized (module doc: Neo4j drops the pairs
                //    collected in the other order).
                let mut paires: HashMap<(String, String), i64> = HashMap::new();
                let mut a_ecrire: Vec<((String, String), i64, NodeId, NodeId)> = Vec::new();
                let mut etat_projet: HashMap<(String, String), EntreeCoChangee>;
                {
                    let etat = w.index()?;
                    let idx = &etat.commits;
                    for (hash, fichiers) in &idx.touches {
                        let Some(e) = idx.par_hash.get(hash) else {
                            continue;
                        };
                        if e.commit.timestamp <= since {
                            continue;
                        }
                        let mut chemins: Vec<&String> = fichiers
                            .keys()
                            .filter(|p| {
                                idx.fichier_par_chemin
                                    .get(*p)
                                    .is_some_and(|f| f.project_id == Some(project_id))
                            })
                            .collect();
                        chemins.sort();
                        let n = chemins.len();
                        if n > 1 && n <= 50 {
                            for i in 0..n {
                                for j in (i + 1)..n {
                                    *paires
                                        .entry((chemins[i].clone(), chemins[j].clone()))
                                        .or_insert(0) += 1;
                                }
                            }
                        }
                    }

                    // 2. Which pairs are written, with their file nodes.
                    let mut cles: Vec<((String, String), i64)> = paires
                        .into_iter()
                        .filter(|(_, c)| *c >= min_count)
                        .collect();
                    cles.sort_by(|x, y| x.0.cmp(&y.0));
                    for ((path_a, path_b), co_count) in cles {
                        let (Some(fa), Some(fb)) = (
                            idx.fichier_par_chemin.get(&path_a),
                            idx.fichier_par_chemin.get(&path_b),
                        ) else {
                            continue;
                        };
                        a_ecrire.push(((path_a, path_b), co_count, fa.noeud, fb.noeud));
                    }

                    // 3. The current state of the project's CO_CHANGED edges.
                    etat_projet = idx
                        .co_changees
                        .iter()
                        .filter(|((pid, _, _), _)| *pid == project_id)
                        .map(|((_, a, b), e)| {
                            (
                                (a.clone(), b.clone()),
                                EntreeCoChangee {
                                    arete: e.arete,
                                    path_a: a.clone(),
                                    path_b: b.clone(),
                                    count: e.count,
                                    last_at: e.last_at.clone(),
                                },
                            )
                        })
                        .collect();
                }
                let total = a_ecrire.len() as i64;

                let maintenant = maintenant();
                for ((path_a, path_b), co_count, na, nb) in a_ecrire {
                    let (arete, ancien) = match etat_projet.get(&(path_a.clone(), path_b.clone())) {
                        Some(e) => (e.arete, e.count),
                        None => (w.db.create_edge(na, nb, CO_CHANGE), 0),
                    };
                    let nouveau = ancien + co_count;
                    w.db.set_edge_property(arete, "count", Value::from(nouveau));
                    w.db.set_edge_property(arete, "last_at", Value::from(maintenant.clone()));
                    w.db.set_edge_property(
                        arete,
                        "project_id",
                        Value::from(project_id.to_string()),
                    );
                    etat_projet.insert(
                        (path_a.clone(), path_b.clone()),
                        EntreeCoChangee {
                            arete,
                            path_a,
                            path_b,
                            count: nouveau,
                            last_at: Some(maintenant.clone()),
                        },
                    );
                }

                // 4. Cap to max_relations: keep the strongest, delete the rest. A
                //    negative cap caps nothing (module doc).
                if max_relations >= 0 {
                    let mut ordonnees: Vec<(String, String, i64)> = etat_projet
                        .values()
                        .map(|e| (e.path_a.clone(), e.path_b.clone(), e.count))
                        .collect();
                    ordonnees.sort_by(|a, b| {
                        b.2.cmp(&a.2)
                            .then_with(|| a.0.cmp(&b.0))
                            .then_with(|| a.1.cmp(&b.1))
                    });
                    for (path_a, path_b, _) in ordonnees.into_iter().skip(max_relations as usize) {
                        if let Some(e) = etat_projet.remove(&(path_a, path_b)) {
                            w.db.delete_edge(e.arete);
                        }
                    }
                }

                // 5. Neo4j's phase 3, in the same write: the project carries when its
                //    co-change was computed — what an incremental run reads as `since`.
                let horodatage = poser_horodatage(w, project_id, "last_co_change_computed_at")?;

                w.rendre_durable()?;
                {
                    let mut etat = w.index_mut()?;
                    etat.commits
                        .co_changees
                        .retain(|(pid, _, _), _| *pid != project_id);
                    for ((path_a, path_b), e) in etat_projet {
                        etat.commits
                            .co_changees
                            .insert((project_id, path_a, path_b), e);
                    }
                    if let Some(h) = horodatage {
                        h.publier(&mut etat);
                    }
                }
                Ok(total)
            },
        )
        .await
    }

    fn get_co_change_graph_porte(
        &self,
        project_id: Uuid,
        min_count: i64,
        limit: i64,
    ) -> Result<Vec<CoChangePair>> {
        self.lire(|_, etat| {
            let mut paires: Vec<CoChangePair> = etat
                .commits
                .co_changees
                .iter()
                .filter(|((pid, _, _), e)| *pid == project_id && e.count >= min_count)
                .map(|(_, e)| CoChangePair {
                    file_a: e.path_a.clone(),
                    file_b: e.path_b.clone(),
                    count: e.count,
                    last_at: e.last_at.clone(),
                })
                .collect();
            paires.sort_by(|a, b| {
                b.count
                    .cmp(&a.count)
                    .then_with(|| a.file_a.cmp(&b.file_a))
                    .then_with(|| a.file_b.cmp(&b.file_b))
            });
            paires.truncate(limit.max(0) as usize);
            Ok(paires)
        })
    }

    fn get_file_co_changers_porte(
        &self,
        file_path: &str,
        min_count: i64,
        limit: i64,
    ) -> Result<Vec<CoChanger>> {
        self.lire(|_, etat| {
            // One entry per path: the same pair may have an edge in two projects
            // (module doc); the strongest is kept.
            let mut par_chemin: HashMap<String, CoChanger> = HashMap::new();
            for e in etat.commits.co_changees.values() {
                if !(e.path_a == file_path || e.path_b == file_path) || e.count < min_count {
                    continue;
                }
                let path = if e.path_a == file_path {
                    e.path_b.clone()
                } else {
                    e.path_a.clone()
                };
                let candidat = CoChanger {
                    path: path.clone(),
                    count: e.count,
                    last_at: e.last_at.clone(),
                };
                match par_chemin.entry(path) {
                    Entry::Vacant(v) => {
                        v.insert(candidat);
                    }
                    Entry::Occupied(mut o) => {
                        let garde = o.get();
                        if (candidat.count, &candidat.last_at) > (garde.count, &garde.last_at) {
                            o.insert(candidat);
                        }
                    }
                }
            }
            let mut changers: Vec<CoChanger> = par_chemin.into_values().collect();
            changers.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.path.cmp(&b.path)));
            changers.truncate(limit.max(0) as usize);
            Ok(changers)
        })
    }

    async fn compute_co_changed_transitive_porte(
        &self,
        project_id: Uuid,
        max_depth: i64,
        min_score: f64,
    ) -> Result<i64> {
        let max_depth = max_depth.clamp(1, 3);
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // 1. Delete the existing transitive edges of the project — ALWAYS,
                //    before anything can return: Neo4j's cleanup runs even when the
                //    project has no CO_CHANGED edge left, so must this one.
                {
                    let etat = w.index()?;
                    let anciennes: Vec<&EntreeCoChangeeTransitive> = etat
                        .commits
                        .co_changees_transitives
                        .iter()
                        .filter(|((pid, _, _), _)| *pid == project_id)
                        .map(|(_, e)| e)
                        .collect();
                    // An old `via` in a blob column: its slot is cleared with the
                    // edge, which the WAL does not recover either.
                    let ancien_en_blob = anciennes.iter().any(|e| liste_en_colonne_blob(&e.via));
                    let a_supprimer: Vec<EdgeId> = anciennes.iter().map(|e| e.arete).collect();
                    drop(anciennes);
                    drop(etat);
                    w.flush_complet = ancien_en_blob;
                    for arete in a_supprimer {
                        w.db.delete_edge(arete);
                    }
                }

                // 2. Normalization: the max CO_CHANGED count of the project; the
                //    CO_CHANGED adjacency of the project, and the direct pairs.
                let (max_count, adj, directes) = {
                    let etat = w.index()?;
                    let mut maxi: i64 = 0;
                    let mut adj: HashMap<String, Vec<(String, i64)>> = HashMap::new();
                    let mut directes: HashSet<(String, String)> = HashSet::new();
                    for ((pid, a, b), e) in &etat.commits.co_changees {
                        if *pid != project_id {
                            continue;
                        }
                        maxi = maxi.max(e.count);
                        directes.insert((a.clone(), b.clone()));
                        adj.entry(a.clone()).or_default().push((b.clone(), e.count));
                        adj.entry(b.clone()).or_default().push((a.clone(), e.count));
                    }
                    (maxi, adj, directes)
                };

                let mut transitifs: HashMap<(String, String), (f64, i64, Vec<String>)> =
                    HashMap::new();
                if max_count > 0 {
                    let max_count = max_count as f64;
                    // 3. Depth-2 walk.
                    for (f1, voisins) in &adj {
                        for (x, w_fx) in voisins {
                            for (f2, w_xf2) in &adj[x] {
                                if f2 == f1 {
                                    continue;
                                }
                                let (a, b) = canon(f1, f2);
                                if directes.contains(&(a.clone(), b.clone())) {
                                    continue;
                                }
                                let score =
                                    ((*w_fx as f64) / max_count) * ((*w_xf2 as f64) / max_count);
                                if score < min_score {
                                    continue;
                                }
                                let via = vec![x.clone()];
                                match transitifs.entry((a, b)) {
                                    Entry::Occupied(mut o) => {
                                        if meilleur(score, &via, o.get()) {
                                            *o.get_mut() = (score, 2, via);
                                        }
                                    }
                                    Entry::Vacant(v) => {
                                        v.insert((score, 2, via));
                                    }
                                }
                            }
                        }
                    }

                    // 4. Depth-3 walk (only the pairs depth 2 did not cover).
                    if max_depth >= 3 {
                        let couvertes: HashSet<(String, String)> =
                            transitifs.keys().cloned().collect();
                        for (f1, voisins) in &adj {
                            for (x, w_fx) in voisins {
                                for (y, w_xy) in &adj[x] {
                                    if y == f1 {
                                        continue;
                                    }
                                    for (f2, w_yf2) in &adj[y] {
                                        if f2 == f1 || f2 == x {
                                            continue;
                                        }
                                        let (a, b) = canon(f1, f2);
                                        let cle = (a, b);
                                        if directes.contains(&cle) || couvertes.contains(&cle) {
                                            continue;
                                        }
                                        let score = ((*w_fx as f64) / max_count)
                                            * ((*w_xy as f64) / max_count)
                                            * ((*w_yf2 as f64) / max_count);
                                        if score < min_score {
                                            continue;
                                        }
                                        // `via` from path_a to path_b.
                                        let via = if *f1 == cle.0 {
                                            vec![x.clone(), y.clone()]
                                        } else {
                                            vec![y.clone(), x.clone()]
                                        };
                                        match transitifs.entry(cle) {
                                            Entry::Occupied(mut o) => {
                                                if meilleur(score, &via, o.get()) {
                                                    *o.get_mut() = (score, 3, via);
                                                }
                                            }
                                            Entry::Vacant(v) => {
                                                v.insert((score, 3, via));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // 5. Write the edges.
                let mut a_ecrire: Vec<_> = transitifs.into_iter().collect();
                a_ecrire.sort_by(|x, y| x.0.cmp(&y.0));
                let maintenant = maintenant();
                let mut nouvelles: Vec<EntreeCoChangeeTransitive> = Vec::new();
                for ((path_a, path_b), (score, depth, via)) in a_ecrire {
                    let (na, nb) = {
                        let etat = w.index()?;
                        let (Some(fa), Some(fb)) = (
                            etat.commits.fichier_par_chemin.get(&path_a),
                            etat.commits.fichier_par_chemin.get(&path_b),
                        ) else {
                            continue;
                        };
                        (fa.noeud, fb.noeud)
                    };
                    let arete = w.db.create_edge(na, nb, CO_CHANGE_TRANSITIF);
                    w.db.set_edge_property(
                        arete,
                        "project_id",
                        Value::from(project_id.to_string()),
                    );
                    w.db.set_edge_property(arete, "score", Value::from(score));
                    w.db.set_edge_property(arete, "depth", Value::from(depth));
                    w.db.set_edge_property(arete, "via", Value::from(via.clone()));
                    w.db.set_edge_property(arete, "computed_at", Value::from(maintenant.clone()));
                    nouvelles.push(EntreeCoChangeeTransitive {
                        arete,
                        path_a,
                        path_b,
                        score,
                        depth,
                        via,
                    });
                }
                let total = nouvelles.len() as i64;
                // A new `via` above the blob threshold: the full flush.
                w.flush_complet |= nouvelles.iter().any(|e| liste_en_colonne_blob(&e.via));

                w.rendre_durable()?;
                {
                    let mut etat = w.index_mut()?;
                    etat.commits
                        .co_changees_transitives
                        .retain(|(pid, _, _), _| *pid != project_id);
                    for e in nouvelles {
                        etat.commits
                            .co_changees_transitives
                            .insert((project_id, e.path_a.clone(), e.path_b.clone()), e);
                    }
                }
                Ok(total)
            },
        )
        .await
    }

    fn get_file_transitive_co_changers_porte(
        &self,
        file_path: &str,
        min_score: f64,
        limit: i64,
    ) -> Result<Vec<TransitiveCoChanger>> {
        self.lire(|_, etat| {
            // One entry per path, the highest score kept (module doc).
            let mut par_chemin: HashMap<String, TransitiveCoChanger> = HashMap::new();
            for e in etat.commits.co_changees_transitives.values() {
                if !(e.path_a == file_path || e.path_b == file_path) || e.score < min_score {
                    continue;
                }
                let path = if e.path_a == file_path {
                    e.path_b.clone()
                } else {
                    e.path_a.clone()
                };
                let candidat = TransitiveCoChanger {
                    path: path.clone(),
                    score: e.score,
                    depth: e.depth,
                    via: e.via.clone(),
                };
                match par_chemin.entry(path) {
                    Entry::Vacant(v) => {
                        v.insert(candidat);
                    }
                    Entry::Occupied(mut o) => {
                        let garde = o.get();
                        if candidat.score > garde.score
                            || (candidat.score == garde.score && candidat.via < garde.via)
                        {
                            o.insert(candidat);
                        }
                    }
                }
            }
            let mut changers: Vec<TransitiveCoChanger> = par_chemin.into_values().collect();
            changers.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.path.cmp(&b.path))
            });
            changers.truncate(limit.max(0) as usize);
            Ok(changers)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::models::{PlanNode, PlanStatus};
    use crate::neo4j::traits::PlanStore;

    fn un_commit(hash: &str, decalage: i64) -> CommitNode {
        CommitNode {
            hash: hash.to_string(),
            message: format!("message {hash}"),
            author: "test".to_string(),
            timestamp: chrono::TimeZone::timestamp_opt(&Utc, 1_700_000_000 + decalage, 0).unwrap(),
        }
    }

    fn un_plan(project_id: Option<Uuid>) -> PlanNode {
        PlanNode {
            id: Uuid::new_v4(),
            title: "plan".to_string(),
            description: "description".to_string(),
            status: PlanStatus::Draft,
            created_at: Utc::now(),
            created_by: "test".to_string(),
            priority: 1,
            project_id,
            execution_context: None,
            persona: None,
        }
    }

    #[tokio::test]
    async fn cycle_de_vie_d_un_commit() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let c = un_commit("abc", 0);
        store.create_commit(&c).await.unwrap();
        assert_eq!(
            store.get_commit("abc").await.unwrap().unwrap().message,
            "message abc"
        );
        assert!(store.get_commit("absent").await.unwrap().is_none());
        store.delete_commit("abc").await.unwrap();
        assert!(store.get_commit("abc").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn commit_lie_a_un_plan() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let plan = un_plan(None);
        store.create_plan(&plan).await.unwrap();
        let c = un_commit("def", 0);
        store.create_commit(&c).await.unwrap();
        store.link_commit_to_plan("def", plan.id).await.unwrap();
        // idempotent: a MERGE. Counted in the ENGINE — the index is a set, it
        // would hide a second edge.
        store.link_commit_to_plan("def", plan.id).await.unwrap();
        let commits = store.get_plan_commits(plan.id).await.unwrap();
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].hash, "def");
        assert_eq!(
            aretes_resultat(&store, "def"),
            1,
            "une seule arête RESULTED_IN"
        );
    }

    /// The RESULTED_IN edges of the commit `hash`, counted in the engine.
    fn aretes_resultat(store: &ObrainGraphStore, hash: &str) -> usize {
        let noeud = store
            .interne
            .etat
            .read()
            .unwrap()
            .commits
            .par_hash
            .get(hash)
            .unwrap()
            .noeud;
        let db = &store.interne.db;
        db.store()
            .edges_from(noeud, Direction::Incoming)
            .into_iter()
            .filter(|(_, a)| {
                db.get_edge(*a)
                    .is_some_and(|e| e.edge_type.as_str() == RESULTAT)
            })
            .count()
    }

    /// A `Commit` node quarantined at open (here: an unreadable `timestamp`)
    /// keeps its hash RESERVED: `get_commit` / `create_commit` / `delete_commit`
    /// of that hash refuse by name, and none of them makes a second node of the
    /// hash or answers as if it were absent.
    #[tokio::test]
    async fn le_hash_d_un_commit_en_quarantaine_reste_reserve() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        {
            let store = ouvrir(&chemin, true);
            store
                .ecrire(
                    || (),
                    |(), w: &mut Ecriture<'_>| {
                        let n = w.db.create_node(&[LABEL]);
                        for (cle, valeur) in [
                            ("hash", "q"),
                            ("message", "m"),
                            ("author", "a"),
                            ("timestamp", "pas une date"),
                        ] {
                            w.db.set_node_property(n, cle, Value::from(valeur));
                        }
                        w.rendre_durable()
                    },
                )
                .await
                .unwrap();
            store.close().await.unwrap();
        }
        let store = ouvrir(&chemin, false);
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
        let erreur = store
            .get_commit("q")
            .await
            .expect_err("get d'un hash réservé");
        assert!(format!("{erreur:#}").contains("réservé"), "{erreur:#}");
        let erreur = store
            .create_commit(&un_commit("q", 0))
            .await
            .expect_err("create d'un hash réservé");
        assert!(format!("{erreur:#}").contains("réservé"), "{erreur:#}");
        let erreur = store
            .delete_commit("q")
            .await
            .expect_err("delete d'un hash réservé");
        assert!(format!("{erreur:#}").contains("réservé"), "{erreur:#}");
        let plan = un_plan(None);
        store.create_plan(&plan).await.unwrap();
        let erreur = store
            .link_commit_to_plan("q", plan.id)
            .await
            .expect_err("lien d'un hash réservé");
        assert!(format!("{erreur:#}").contains("réservé"), "{erreur:#}");
        assert_eq!(
            store.interne.db.store().nodes_by_label(LABEL).len(),
            1,
            "aucun second nœud Commit pour le hash réservé"
        );
        // Another hash is served as usual.
        store.create_commit(&un_commit("r", 0)).await.unwrap();
        assert!(store.get_commit("r").await.unwrap().is_some());
    }

    /// `create_commit` is a MERGE on `hash`, as Neo4j's and the mock's: a commit
    /// registered twice is rewritten, not refused and not duplicated — a second
    /// node would be caught at the next open as "hash déjà pris".
    #[tokio::test]
    async fn deux_creations_du_meme_commit_le_reecrivent_sans_le_dupliquer() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().to_path_buf();
        let premier = un_commit("merge", 0);
        let mut second = premier.clone();
        second.message = "message réécrit".to_string();
        second.author = "quelqu'un d'autre".to_string();
        second.timestamp = premier.timestamp + chrono::Duration::seconds(60);
        {
            let store = ObrainGraphStore::ouvrir(
                chemin.clone(),
                crate::obrain::Durabilite::Wal,
                true,
                None,
            )
            .unwrap();
            store.create_commit(&premier).await.unwrap();
            store.create_commit(&second).await.unwrap();
            assert_eq!(
                store.get_commit("merge").await.unwrap().unwrap().message,
                "message réécrit",
                "la seconde création réécrit le nœud existant"
            );
            store.close().await.unwrap();
        }
        let store =
            ObrainGraphStore::ouvrir(chemin, crate::obrain::Durabilite::Wal, false, None).unwrap();
        let rapport = store.rapport_d_ouverture().clone();
        assert!(
            rapport.en_quarantaine.is_empty(),
            "un second nœud Commit pour le même hash serait mis en quarantaine : {rapport:?}"
        );
        let relu = store.get_commit("merge").await.unwrap().unwrap();
        assert_eq!(
            (relu.message.as_str(), relu.author.as_str(), relu.timestamp),
            ("message réécrit", "quelqu'un d'autre", second.timestamp),
            "et les trois valeurs ont survécu à la fermeture"
        );
    }

    // ------------------------------------------------------------------
    // The TOUCHES / CO_CHANGED machinery, behind the refusal: run directly
    // through the `*_porte` methods, on `File` nodes written by the test hook.
    // ------------------------------------------------------------------

    use crate::neo4j::models::{ProjectNode, ProjectProfile};
    use crate::neo4j::traits::ProjectStore;
    use crate::obrain::Durabilite;

    fn un_projet() -> ProjectNode {
        let id = Uuid::new_v4();
        ProjectNode {
            id,
            name: "co-change".to_string(),
            slug: format!("co-change-{id}"),
            root_path: String::new(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: false,
            profile: ProjectProfile::Software,
        }
    }

    fn touche(path: &str, additions: Option<i64>, deletions: Option<i64>) -> FileChangedInfo {
        FileChangedInfo {
            path: path.to_string(),
            additions,
            deletions,
        }
    }

    fn ouvrir(chemin: &std::path::Path, creer: bool) -> ObrainGraphStore {
        ObrainGraphStore::ouvrir(chemin.to_path_buf(), Durabilite::Wal, creer, None).unwrap()
    }

    /// The project, its files, and one commit per list of paths (`h1`, `h2`, …,
    /// one second apart), each touching its files.
    async fn historique(store: &ObrainGraphStore, fichiers: &[&str], commits: &[&[&str]]) -> Uuid {
        let projet = un_projet();
        store.create_project(&projet).await.unwrap();
        for f in fichiers {
            store
                .ecrire_fichier_pour_test(f, Some(projet.id))
                .await
                .unwrap();
        }
        for (i, chemins) in commits.iter().enumerate() {
            let hash = format!("h{}", i + 1);
            store
                .create_commit(&un_commit(&hash, i as i64 + 1))
                .await
                .unwrap();
            let touches: Vec<FileChangedInfo> =
                chemins.iter().map(|p| touche(p, None, None)).collect();
            store
                .create_commit_touches_porte(&hash, &touches)
                .await
                .unwrap();
        }
        projet.id
    }

    fn paires(graphe: &[CoChangePair]) -> Vec<(&str, &str, i64)> {
        graphe
            .iter()
            .map(|p| (p.file_a.as_str(), p.file_b.as_str(), p.count))
            .collect()
    }

    /// End to end, AFTER a close and a reopen: TOUCHES, CO_CHANGED, the
    /// transitive edge (path, depth, via, score) and the project timestamp are
    /// read back from the rebuilt index — and `last_at` is written with a `Z`.
    #[tokio::test]
    async fn co_change_transitif_et_horodatage_survivent_a_la_reouverture() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let store = ouvrir(&chemin, true);
        let projet = un_projet();
        store.create_project(&projet).await.unwrap();
        for f in ["src/a.rs", "src/b.rs", "src/c.rs"] {
            store
                .ecrire_fichier_pour_test(f, Some(projet.id))
                .await
                .unwrap();
        }
        store.create_commit(&un_commit("h1", 1)).await.unwrap();
        store.create_commit(&un_commit("h2", 2)).await.unwrap();
        store
            .create_commit_touches_porte(
                "h1",
                &[
                    touche("src/a.rs", Some(5), Some(1)),
                    touche("src/b.rs", None, None),
                ],
            )
            .await
            .unwrap();
        store
            .create_commit_touches_porte(
                "h2",
                // `deletions` without `additions`: written (module doc).
                &[
                    touche("src/b.rs", None, None),
                    touche("src/c.rs", None, Some(2)),
                ],
            )
            .await
            .unwrap();
        assert_eq!(
            store
                .compute_co_changed_porte(projet.id, None, 1, 100)
                .await
                .unwrap(),
            2
        );
        let horodatage = store
            .get_project(projet.id)
            .await
            .unwrap()
            .unwrap()
            .last_co_change_computed_at
            .expect("compute_co_changed horodate le projet");
        let graphe = store.get_co_change_graph_porte(projet.id, 1, 100).unwrap();
        assert_eq!(
            paires(&graphe),
            [("src/a.rs", "src/b.rs", 1), ("src/b.rs", "src/c.rs", 1)]
        );
        let last_at = graphe[0].last_at.clone().unwrap();
        assert!(
            last_at.ends_with('Z'),
            "`last_at` au format de Neo4j (`…Z`), pas `+00:00` : {last_at}"
        );
        assert_eq!(
            store
                .compute_co_changed_transitive_porte(projet.id, 2, 0.1)
                .await
                .unwrap(),
            1
        );
        store.close().await.unwrap();

        let store = ouvrir(&chemin, false);
        assert!(
            store.rapport_d_ouverture().en_quarantaine.is_empty(),
            "{:?}",
            store.rapport_d_ouverture()
        );
        let fichiers = store.get_commit_files_porte("h1").unwrap();
        assert_eq!(
            fichiers
                .iter()
                .map(|f| (f.path.as_str(), f.additions, f.deletions))
                .collect::<Vec<_>>(),
            [("src/a.rs", Some(5), Some(1)), ("src/b.rs", None, None)]
        );
        let fichiers = store.get_commit_files_porte("h2").unwrap();
        assert_eq!(
            (
                fichiers[1].path.as_str(),
                fichiers[1].additions,
                fichiers[1].deletions
            ),
            ("src/c.rs", None, Some(2))
        );
        let hist = store.get_file_history_porte("src/b.rs", None).unwrap();
        assert_eq!(
            hist.iter().map(|h| h.hash.as_str()).collect::<Vec<_>>(),
            ["h2", "h1"]
        );
        let relu = store.get_co_change_graph_porte(projet.id, 1, 100).unwrap();
        assert_eq!(paires(&relu), paires(&graphe));
        assert_eq!(relu[0].last_at.as_deref(), Some(last_at.as_str()));
        let co = store
            .get_file_co_changers_porte("src/b.rs", 1, 100)
            .unwrap();
        assert_eq!(
            co.iter().map(|c| c.path.as_str()).collect::<Vec<_>>(),
            ["src/a.rs", "src/c.rs"]
        );
        let trans = store
            .get_file_transitive_co_changers_porte("src/a.rs", 0.1, 100)
            .unwrap();
        assert_eq!(trans.len(), 1, "le transitif est relu après réouverture");
        assert_eq!(
            (
                trans[0].path.as_str(),
                trans[0].depth,
                trans[0].via.clone(),
                trans[0].score
            ),
            ("src/c.rs", 2, vec!["src/b.rs".to_string()], 1.0)
        );
        assert_eq!(
            store
                .get_project(projet.id)
                .await
                .unwrap()
                .unwrap()
                .last_co_change_computed_at,
            Some(horodatage),
            "l'horodatage du projet est relu après réouverture"
        );
    }

    /// `count` is incremental, as Neo4j's `coalesce(r.count, 0) + co_count`: two
    /// runs over the same window add up; a run with `since` adds only the commits
    /// after it.
    #[tokio::test]
    async fn le_count_se_cumule_d_une_execution_a_l_autre() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let projet = historique(&store, &["a", "b"], &[&["a", "b"], &["a", "b"]]).await;
        store
            .compute_co_changed_porte(projet, None, 1, 100)
            .await
            .unwrap();
        assert_eq!(
            paires(&store.get_co_change_graph_porte(projet, 1, 100).unwrap()),
            [("a", "b", 2)]
        );
        store
            .compute_co_changed_porte(projet, None, 1, 100)
            .await
            .unwrap();
        assert_eq!(
            paires(&store.get_co_change_graph_porte(projet, 1, 100).unwrap()),
            [("a", "b", 4)]
        );
        // Only h2 is after h1's instant.
        let depuis = un_commit("h1", 1).timestamp;
        store
            .compute_co_changed_porte(projet, Some(depuis), 1, 100)
            .await
            .unwrap();
        assert_eq!(
            paires(&store.get_co_change_graph_porte(projet, 1, 100).unwrap()),
            [("a", "b", 5)]
        );
    }

    /// `max_relations` below the number of pairs keeps the strongest, by
    /// `(count DESC, path_a, path_b)`, deletes the others for good, and the
    /// answer is still the number of pairs merged (Neo4j's `count(r)`).
    #[tokio::test]
    async fn max_relations_sous_le_nombre_de_paires_garde_les_plus_fortes() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let store = ouvrir(&chemin, true);
        let projet = historique(
            &store,
            &["a", "b", "c"],
            &[&["a", "b"], &["a", "b"], &["b", "c"], &["a", "c"]],
        )
        .await;
        assert_eq!(
            store
                .compute_co_changed_porte(projet, None, 1, 2)
                .await
                .unwrap(),
            3
        );
        assert_eq!(
            paires(&store.get_co_change_graph_porte(projet, 1, 100).unwrap()),
            [("a", "b", 2), ("a", "c", 1)]
        );
        store.close().await.unwrap();
        let store = ouvrir(&chemin, false);
        assert_eq!(
            paires(&store.get_co_change_graph_porte(projet, 1, 100).unwrap()),
            [("a", "b", 2), ("a", "c", 1)],
            "l'arête écartée est supprimée de la base, pas seulement de l'index"
        );
    }

    /// A negative `max_relations` caps nothing: Neo4j's `SKIP -1` fails inside a
    /// non-fatal cleanup and deletes no edge.
    #[tokio::test]
    async fn max_relations_negatif_ne_supprime_rien() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let projet = historique(&store, &["a", "b", "c"], &[&["a", "b"], &["b", "c"]]).await;
        store
            .compute_co_changed_porte(projet, None, 1, -1)
            .await
            .unwrap();
        assert_eq!(
            paires(&store.get_co_change_graph_porte(projet, 1, 100).unwrap()),
            [("a", "b", 1), ("b", "c", 1)]
        );
    }

    /// The transitive edges of the project are purged on EVERY recompute — also
    /// when the project has no CO_CHANGED edge left (Neo4j's cleanup runs
    /// before it looks at the counts). Otherwise a transitive edge outlives the
    /// direct edges it was derived from.
    #[tokio::test]
    async fn le_transitif_est_purge_meme_sans_co_changement() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let projet = historique(&store, &["a", "b", "c"], &[&["a", "b"], &["b", "c"]]).await;
        store
            .compute_co_changed_porte(projet, None, 1, 100)
            .await
            .unwrap();
        assert_eq!(
            store
                .compute_co_changed_transitive_porte(projet, 2, 0.1)
                .await
                .unwrap(),
            1
        );
        // max_relations = 0: every CO_CHANGED edge of the project goes.
        store
            .compute_co_changed_porte(projet, None, 1, 0)
            .await
            .unwrap();
        assert!(store
            .get_co_change_graph_porte(projet, 1, 100)
            .unwrap()
            .is_empty());
        assert_eq!(
            store
                .compute_co_changed_transitive_porte(projet, 2, 0.1)
                .await
                .unwrap(),
            0
        );
        assert!(
            store
                .get_file_transitive_co_changers_porte("a", 0.0, 100)
                .unwrap()
                .is_empty(),
            "l'ancienne arête transitive est purgée"
        );
    }

    /// Two paths of the same score (`a–x–c`, `a–y–c`): the `via` written is the
    /// smallest, whatever order the walk meets them in. Several fresh stores, so
    /// that the per-map hash seeds differ.
    #[tokio::test]
    async fn via_est_deterministe_a_score_egal() {
        for _ in 0..8 {
            let store = ObrainGraphStore::open_temporary().unwrap();
            let projet = historique(
                &store,
                &["a", "c", "x", "y"],
                &[&["a", "x"], &["x", "c"], &["a", "y"], &["y", "c"]],
            )
            .await;
            store
                .compute_co_changed_porte(projet, None, 1, 100)
                .await
                .unwrap();
            store
                .compute_co_changed_transitive_porte(projet, 2, 0.1)
                .await
                .unwrap();
            let trans = store
                .get_file_transitive_co_changers_porte("a", 0.1, 100)
                .unwrap();
            assert_eq!(trans.len(), 1);
            assert_eq!(trans[0].via, vec!["x".to_string()]);
            let trans = store
                .get_file_transitive_co_changers_porte("x", 0.1, 100)
                .unwrap();
            assert_eq!(trans.len(), 1);
            assert_eq!(trans[0].via, vec!["a".to_string()]);
        }
    }

    /// The same pair with an edge in two projects (the key carries the project):
    /// `get_file_co_changers`, not scoped by project, answers the path ONCE, with
    /// the strongest count.
    #[tokio::test]
    async fn les_co_changeurs_sont_dedoublonnes_par_chemin() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let projet = historique(&store, &["a", "b"], &[&["a", "b"]]).await;
        store
            .compute_co_changed_porte(projet, None, 1, 100)
            .await
            .unwrap();
        {
            let mut etat = store.interne.etat.write().unwrap();
            let arete = etat.commits.co_changees.values().next().unwrap().arete;
            etat.commits.co_changees.insert(
                (Uuid::new_v4(), "a".to_string(), "b".to_string()),
                EntreeCoChangee {
                    arete,
                    path_a: "a".to_string(),
                    path_b: "b".to_string(),
                    count: 3,
                    last_at: None,
                },
            );
        }
        let co = store.get_file_co_changers_porte("a", 1, 100).unwrap();
        assert_eq!(
            co.iter()
                .map(|c| (c.path.as_str(), c.count))
                .collect::<Vec<_>>(),
            [("b", 3)]
        );
    }

    /// The reads are served from the index published AFTER the durability step,
    /// never from the engine, which shows a write as soon as it is made: while a
    /// write is between its engine writes and its fsync, a reader sees the state
    /// before it.
    #[tokio::test]
    async fn les_lectures_ne_voient_pas_une_ecriture_pas_encore_durable() {
        let store = std::sync::Arc::new(
            ObrainGraphStore::open_temporary_avec(Durabilite::LenteAPourTest(600)).unwrap(),
        );
        let plan = un_plan(None);
        store.create_plan(&plan).await.unwrap();
        store.create_commit(&un_commit("lent", 0)).await.unwrap();
        store.ecrire_fichier_pour_test("f", None).await.unwrap();

        let ecrivain = {
            let s = store.clone();
            let plan_id = plan.id;
            tokio::spawn(async move { s.link_commit_to_plan("lent", plan_id).await })
        };
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(
            store.get_plan_commits(plan.id).await.unwrap().is_empty(),
            "RESULTED_IN lu avant d'être durable"
        );
        ecrivain.await.unwrap().unwrap();
        assert_eq!(store.get_plan_commits(plan.id).await.unwrap().len(), 1);

        let ecrivain = {
            let s = store.clone();
            tokio::spawn(async move {
                s.create_commit_touches_porte("lent", &[touche("f", Some(1), Some(1))])
                    .await
            })
        };
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(
            store.get_commit_files_porte("lent").unwrap().is_empty(),
            "TOUCHES lu (commit) avant d'être durable"
        );
        assert!(
            store.get_file_history_porte("f", None).unwrap().is_empty(),
            "TOUCHES lu (fichier) avant d'être durable"
        );
        ecrivain.await.unwrap().unwrap();
        assert_eq!(store.get_commit_files_porte("lent").unwrap().len(), 1);
        assert_eq!(store.get_file_history_porte("f", None).unwrap().len(), 1);
    }

    // ------------------------------------------------------------------
    // Values above the blob threshold: the write that carries them (or
    // replaces one) pays the FULL flush. Observed without `close()`, through
    // the engine's own fsync counter: a full flush runs more of them than the
    // WAL fsync alone (as `tests/obrain_backend.rs`).
    // ------------------------------------------------------------------

    fn fsyncs(store: &ObrainGraphStore) -> u64 {
        store.fsyncs_du_wal().expect("une base persistante")
    }

    /// The commit message: 256 bytes is the WAL fsync, 257 the full flush, on
    /// creation; and a rewrite that REPLACES a long message by a short one pays
    /// it too (the old slot is cleared).
    #[tokio::test]
    async fn un_message_long_paie_le_flush_complet() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let mut court = un_commit("court", 0);
        court.message = "x".repeat(256);
        let t = fsyncs(&store);
        store.create_commit(&court).await.unwrap();
        let cout_court = fsyncs(&store) - t;
        let mut long = un_commit("long", 0);
        long.message = "x".repeat(257);
        let t = fsyncs(&store);
        store.create_commit(&long).await.unwrap();
        let cout_long = fsyncs(&store) - t;
        assert!(cout_long > cout_court, "{cout_long} contre {cout_court}");
        long.message = "court désormais".to_string();
        let t = fsyncs(&store);
        store.create_commit(&long).await.unwrap();
        let cout_remplacement = fsyncs(&store) - t;
        let t = fsyncs(&store);
        store.create_commit(&court).await.unwrap(); // short over short
        let cout_court_sur_court = fsyncs(&store) - t;
        assert!(
            cout_remplacement > cout_court_sur_court,
            "l'ancienne valeur longue compte : {cout_remplacement} contre {cout_court_sur_court}"
        );
    }

    /// A TOUCHES edge carries the path (`file_path`): a path above the threshold
    /// pays the full flush.
    #[tokio::test]
    async fn un_chemin_long_dans_touches_paie_le_flush_complet() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let court = "c".repeat(256);
        let long = "l".repeat(257);
        store.ecrire_fichier_pour_test(&court, None).await.unwrap();
        store.ecrire_fichier_pour_test(&long, None).await.unwrap();
        store.create_commit(&un_commit("h", 0)).await.unwrap();
        let t = fsyncs(&store);
        store
            .create_commit_touches_porte("h", &[touche(&court, None, None)])
            .await
            .unwrap();
        let cout_court = fsyncs(&store) - t;
        let t = fsyncs(&store);
        store
            .create_commit_touches_porte("h", &[touche(&long, None, None)])
            .await
            .unwrap();
        let cout_long = fsyncs(&store) - t;
        assert!(cout_long > cout_court, "{cout_long} contre {cout_court}");
    }

    /// A transitive edge carries `via`, a list of paths: one path above the
    /// threshold pays the full flush — when it is written, and when the next
    /// recompute deletes the edge that holds it.
    #[tokio::test]
    async fn un_via_long_paie_le_flush_complet() {
        async fn cout(intermediaire: &str) -> (u64, u64) {
            let store = ObrainGraphStore::open_temporary().unwrap();
            let projet = historique(
                &store,
                &["a", intermediaire, "c"],
                &[&["a", intermediaire], &[intermediaire, "c"]],
            )
            .await;
            store
                .compute_co_changed_porte(projet, None, 1, 100)
                .await
                .unwrap();
            let t = fsyncs(&store);
            assert_eq!(
                store
                    .compute_co_changed_transitive_porte(projet, 2, 0.1)
                    .await
                    .unwrap(),
                1
            );
            let ecriture = fsyncs(&store) - t;
            // Every direct edge goes: the recompute only deletes the old edge.
            store
                .compute_co_changed_porte(projet, None, 1, 0)
                .await
                .unwrap();
            let t = fsyncs(&store);
            store
                .compute_co_changed_transitive_porte(projet, 2, 0.1)
                .await
                .unwrap();
            (ecriture, fsyncs(&store) - t)
        }
        let (ecriture_courte, purge_courte) = cout(&"b".repeat(256)).await;
        let (ecriture_longue, purge_longue) = cout(&"b".repeat(257)).await;
        assert!(
            ecriture_longue > ecriture_courte,
            "écriture : {ecriture_longue} contre {ecriture_courte}"
        );
        assert!(
            purge_longue > purge_courte,
            "purge de l'ancienne valeur : {purge_longue} contre {purge_courte}"
        );
    }
}
