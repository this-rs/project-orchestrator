//! `CodeStore` on obrain — the core of the code domain: the `File` nodes, their
//! membership of a project, and the three project switches the contract keeps here.
//!
//! What is ported (14 of the 85 methods): the files read and maintained (`get_file`,
//! `get_file_language`, `find_files_by_path_suffix`, `list_project_files`,
//! `count_project_files`, `get_project_file_paths`, `link_file_to_project`,
//! `delete_file`, `delete_stale_files`), the two language statistics, and
//! `set_scaffolding_override` / `set_watch_enabled` / `set_default_note_energy` (they
//! write a property of the `Project` node, owned by `project.rs`). What is refused by
//! name ([`REFUSEES`], 71 methods): everything that writes or reads a symbol
//! (`Function`, `Struct`, `Trait`, `Enum`, `Impl`, `Import`), an `IMPORTS` / `CALLS` /
//! `EXTENDS` / … edge, a process, a graph-analytics metric, the `CO_CHANGED` edges —
//! and the two file WRITES, `upsert_file` and `batch_upsert_files`. Each refusal carries
//! its reason, and `mod.rs` keeps the list apart from the methods that serve
//! (`methodes_qui_servent`). A port that answered those with an empty list would be a
//! false success.
//!
//! # Why the file writes are refused while the symbols are
//!
//! The sync (`src/orchestrator/runner.rs`, `store_parsed_file_for_project_with_ctx` and
//! the batched store) writes the `File` WITH ITS HASH first, then its symbols. If the
//! file write served and the symbol writes refused, the sync would fail after the hash
//! is stored, and every later sync would skip the file as unchanged (`get_file` → same
//! hash): it would never be parsed again, even once the symbols are ported. Refusing the
//! file write too keeps the hash unwritten, so the first sync after the symbols are
//! ported parses everything. The writer is complete and tested (`ecrire_fichiers`, test
//! builds only: [`ObrainGraphStore::semer_fichiers_pour_essai`]); it is plugged back into
//! `upsert_file` / `batch_upsert_files` together with the symbol writes.
//!
//! # Storage
//!
//! A file is one `File` node (`path`, `language`, `hash`, `last_parsed`, `project_id`).
//! Its membership of a project is a real `Project -[:CONTAINS]-> File` edge, as in
//! Neo4j: `list_project_files`, `count_project_files` and `delete_stale_files` read it,
//! while the `project_id` PROPERTY (what `get_file`, `find_files_by_path_suffix` and the
//! per-project language statistics read) can differ from it, exactly as in Neo4j
//! (`link_file_to_project` adds an edge and leaves the property alone). Reads are
//! served from an in-memory index (path → file, project → paths) rebuilt at open by one
//! scan of the `File` label and one read of each project's `CONTAINS` edges.
//!
//! Writes follow the pattern of the other domains: one writer, the draft marker
//! `_brouillon` removed last (a creation killed half-way is deleted at open, with its
//! edge), one WAL fsync per call, a full flush BEFORE the marker is removed when a value
//! goes to a blob column (see `SEUIL_COLONNE_BLOB`), publication to the index only
//! after durability. A batch is ONE writer section and ONE fsync.
//!
//! # Differences with the Neo4j adapter (`src/neo4j/code.rs`), all named
//!
//! - **Identity is the path.** Neo4j keys a file on `uid` = `PathPolicy::uid(project,
//!   relative path)` and adopts a node already at that path. Here there is no `uid`: the
//!   path is the key, which is what Neo4j's own `file_path` uniqueness constraint
//!   amounts to. Two projects upserting the same path therefore share ONE node whose
//!   `project_id` is the last writer's, and both projects `CONTAIN` it (as in Neo4j
//!   after adoption). The uid-based re-mounting of a repository (same relative path
//!   under another root) is NOT reproduced: it is a decision about the identity of files,
//!   not a port.
//! - **A stored node that does not decode is quarantined** (`last_parsed` unreadable, a
//!   missing property) instead of being read with "now" as its date, and so is a second
//!   node at a path already taken. It then counts in no read. Its PATH (when it has a
//!   readable one) stays known to the index: every write and every deletion that touches
//!   a quarantined path refuses by name (`CodeStore::<method> (quarantaine)`), before
//!   writing anything. Otherwise a write would create a second `File` beside the
//!   quarantined one, and deleting the indexed node of a duplicated path would let the
//!   duplicate come back as the file at the next open. `delete_stale_files` refuses as
//!   well when the project CONTAINS a quarantined path it would have to delete. The
//!   quarantine is lifted by repairing the base by hand, then reopening.
//! - **A deletion the engine refuses leaves the store indeterminate.** obrain's
//!   `delete_node` detaches the node's edges BEFORE it deletes the node, so a `false`
//!   may come after a write; and in `delete_stale_files` the earlier files of the call
//!   are already gone. As everywhere in this backend (`note.rs`, `delete_note`), the
//!   store then refuses everything until a restart rebuilds the index from the base.
//! - **A batch is not atomic against a crash.** Neo4j runs a chunk in one transaction;
//!   here the nodes are written one by one then made durable once. A creation killed
//!   half-way is deleted at open (draft marker), but an UPDATE killed half-way may leave a
//!   file with its new `hash` and its old `last_parsed`. The next sync rewrites it.
//!   `delete_stale_files` likewise may stop half-way; the next call finishes it.
//! - **A file that has symbols is not deleted.** Neo4j deletes the file AND its
//!   `CONTAINS` children. Symbols are not ported, so a base that holds some (written
//!   elsewhere) would be left with orphans: `delete_file` and `delete_stale_files` then
//!   refuse by name (`CodeStore::delete_file (symboles)`), before writing anything.
//! - **Order.** `find_files_by_path_suffix`, `get_project_file_paths` and the stale paths
//!   are sorted by path (Neo4j leaves them unordered); `list_project_files` is sorted by
//!   path as the query says; the statistics are sorted by count descending then language.
//! - **Deleting a project is refused** by `ProjectStore::delete_project` as soon as a file
//!   exists (the cascade over files is not ported): that refusal is already there, it
//!   counts every node of the base.
//!
//! Costs: a get is a hash lookup; `list_project_files` is O(k) in the project's files;
//! `find_files_by_path_suffix` and the global statistics are O(F log F) over all files
//! (they scan the index, as the Neo4j query scans the label); a batch of n files is O(n)
//! engine writes and one fsync. Open adds one scan of the `File` label.

use std::collections::{BTreeSet, HashMap, HashSet};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::project::IndexProjets;
use super::{Ecriture, NonPorte, ObrainGraphStore, RapportOuverture};
use crate::neo4j::models::*;
use crate::neo4j::traits::CodeStore;
use crate::parser::FunctionCall;

const ETIQUETTE_FICHIER: &str = "File";
/// The edge from a project to a file it contains.
const CONTIENT: &str = "CONTAINS";
/// The draft marker: present from the first record of a creation to the last.
const BROUILLON: &str = "_brouillon";

/// The methods this domain refuses by name, as `CodeStore::<method>`. They are
/// written (so the build sees them as implemented) but they answer [`NonPorte`].
pub const REFUSEES: &[&str] = &[
    "CodeStore::upsert_file",
    "CodeStore::batch_upsert_files",
    "CodeStore::count_orphan_files",
    "CodeStore::count_co_change_pairs",
    "CodeStore::count_protocol_states_transitions",
    "CodeStore::invalidate_computed_properties",
    "CodeStore::upsert_function",
    "CodeStore::upsert_struct",
    "CodeStore::upsert_trait",
    "CodeStore::find_trait_by_name",
    "CodeStore::upsert_enum",
    "CodeStore::upsert_impl",
    "CodeStore::create_import_relationship",
    "CodeStore::upsert_import",
    "CodeStore::create_imports_symbol_relationship",
    "CodeStore::create_call_relationship",
    "CodeStore::batch_upsert_functions",
    "CodeStore::batch_upsert_structs",
    "CodeStore::batch_upsert_traits",
    "CodeStore::batch_upsert_enums",
    "CodeStore::batch_upsert_impls",
    "CodeStore::batch_upsert_imports",
    "CodeStore::batch_create_import_relationships",
    "CodeStore::batch_create_imports_symbol_relationships",
    "CodeStore::batch_create_call_relationships",
    "CodeStore::batch_create_extends_relationships",
    "CodeStore::batch_create_implements_relationships",
    "CodeStore::cleanup_cross_project_calls",
    "CodeStore::cleanup_builtin_calls",
    "CodeStore::migrate_calls_confidence",
    "CodeStore::cleanup_sync_data",
    "CodeStore::get_callees",
    "CodeStore::create_uses_type_relationship",
    "CodeStore::find_trait_implementors",
    "CodeStore::get_type_traits",
    "CodeStore::get_impl_blocks",
    "CodeStore::get_class_hierarchy",
    "CodeStore::find_subclasses",
    "CodeStore::find_interface_implementors",
    "CodeStore::list_processes",
    "CodeStore::get_process_detail",
    "CodeStore::get_entry_points",
    "CodeStore::get_file_functions_summary",
    "CodeStore::get_file_structs_summary",
    "CodeStore::get_file_import_paths_list",
    "CodeStore::find_symbol_references",
    "CodeStore::get_file_direct_imports",
    "CodeStore::get_function_callers_by_name",
    "CodeStore::get_function_callees_by_name",
    "CodeStore::get_callers_with_confidence",
    "CodeStore::get_callees_with_confidence",
    "CodeStore::get_most_connected_files",
    "CodeStore::get_most_connected_files_detailed",
    "CodeStore::get_most_connected_files_for_project",
    "CodeStore::get_project_communities",
    "CodeStore::get_node_analytics",
    "CodeStore::get_affected_communities",
    "CodeStore::get_code_health_report",
    "CodeStore::compute_maintenance_snapshot",
    "CodeStore::compute_scaffolding_level",
    "CodeStore::detect_global_stagnation",
    "CodeStore::get_circular_dependencies",
    "CodeStore::get_node_gds_metrics",
    "CodeStore::get_project_percentiles",
    "CodeStore::get_top_bridges_by_betweenness",
    "CodeStore::get_file_symbol_names",
    "CodeStore::get_function_caller_count",
    "CodeStore::get_trait_info",
    "CodeStore::get_trait_implementors_detailed",
    "CodeStore::get_type_trait_implementations",
    "CodeStore::get_type_impl_blocks_detailed",
];

/// Why `upsert_file` / `batch_upsert_files` refuse (see the module header).
const RAISON_ECRITURE_FICHIER: &str = "la synchro écrit le File avec son hash PUIS ses symboles, \
     qui ne sont pas portés : le hash écrit, l'échec qui suit ferait sauter ce fichier par \
     hash à toutes les synchros suivantes, et il ne serait jamais reparsé. Refusé tant que \
     les écritures de symboles le sont";

/// A property lookup over one node: `None` for an absent or null property.
type Lecture<'a> = &'a dyn Fn(&str) -> Option<Value>;

/// A refusal, by name, with the reason it is not ported.
fn refus<T>(methode: &'static str, raison: &'static str) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).context(raison))
}

// ============================================================================
// Index
// ============================================================================

/// One file held in the index: where it lives, and what it decodes to.
struct Fichier {
    noeud: NodeId,
    valeur: FileNode,
}

/// The files of the base, rebuilt at open.
#[derive(Default)]
pub(super) struct IndexCode {
    /// `path → file`.
    fichiers: HashMap<String, Fichier>,
    /// `project → paths it CONTAINS` (sorted: the order of `list_project_files`).
    contenus: HashMap<Uuid, BTreeSet<String>>,
    /// The paths of the quarantined nodes (a duplicate, or a node that does not
    /// decode but has a readable path). Every write and deletion on them refuses.
    quarantaine: HashSet<String>,
    /// `project → quarantined paths it CONTAINS` (what `delete_stale_files` cannot
    /// see in `contenus` and must not silently leave behind).
    quarantaine_contenue: HashMap<Uuid, BTreeSet<String>>,
}

impl IndexCode {
    /// Refuse by name when one of `paths` is in quarantine: nothing is written.
    fn refus_si_quarantaine<'a>(
        &self,
        methode: &'static str,
        mut paths: impl Iterator<Item = &'a String>,
    ) -> Result<()> {
        if let Some(p) = paths.find(|p| self.quarantaine.contains(*p)) {
            return Err(anyhow::Error::new(NonPorte(methode)).context(format!(
                "le chemin {p} a un nœud File en quarantaine (doublon ou nœud illisible) : \
                 y écrire créerait un second File, y supprimer laisserait le doublon revenir \
                 à la prochaine ouverture. Réparer la base, puis rouvrir"
            )));
        }
        Ok(())
    }
}

impl IndexCode {
    /// Rebuild the index: a scan of the `File` label (drafts deleted, what does not
    /// decode quarantined), then each project's `CONTAINS` edges.
    pub(super) fn reconstruire(
        db: &ObrainDB,
        projets: &IndexProjets,
        rapport: &mut RapportOuverture,
    ) -> Self {
        let mut idx = Self::default();
        let mut chemin_de: HashMap<NodeId, String> = HashMap::new();
        let mut chemin_en_quarantaine: HashMap<NodeId, String> = HashMap::new();
        let store = db.store();
        for noeud in store.nodes_by_label(ETIQUETTE_FICHIER) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud File {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            if get(BROUILLON).is_some() || (n.properties.is_empty() && store.out_degree(noeud) == 0)
            {
                db.delete_node(noeud);
                rapport.brouillons_supprimes += 1;
                continue;
            }
            match lire_fichier(&get) {
                Ok(valeur) if idx.fichiers.contains_key(&valeur.path) => {
                    rapport.en_quarantaine.push(format!(
                        "nœud File {noeud:?} : chemin {} déjà pris",
                        valeur.path
                    ));
                    idx.quarantaine.insert(valeur.path.clone());
                    chemin_en_quarantaine.insert(noeud, valeur.path);
                }
                Ok(valeur) => {
                    chemin_de.insert(noeud, valeur.path.clone());
                    idx.fichiers
                        .insert(valeur.path.clone(), Fichier { noeud, valeur });
                }
                Err(e) => {
                    rapport
                        .en_quarantaine
                        .push(format!("nœud File {noeud:?} : {e}"));
                    if let Some(path) = chaine(&get, "path") {
                        idx.quarantaine.insert(path.clone());
                        chemin_en_quarantaine.insert(noeud, path);
                    }
                }
            }
        }
        for (projet, noeud_projet) in projets.entrees() {
            for (cible, _) in voisins(db, noeud_projet, CONTIENT) {
                if let Some(chemin) = chemin_de.get(&cible) {
                    idx.contenus
                        .entry(projet)
                        .or_default()
                        .insert(chemin.clone());
                } else if let Some(chemin) = chemin_en_quarantaine.get(&cible) {
                    idx.quarantaine_contenue
                        .entry(projet)
                        .or_default()
                        .insert(chemin.clone());
                }
            }
        }
        idx
    }
}

/// Outgoing neighbours of `noeud` through edges of `type_`, read by the engine.
fn voisins(
    db: &ObrainDB,
    noeud: NodeId,
    type_: &str,
) -> Vec<(NodeId, obrain_common::types::EdgeId)> {
    db.session().get_neighbors_outgoing_by_type(noeud, type_)
}

// ============================================================================
// Decoding and encoding
// ============================================================================

fn chaine(get: Lecture<'_>, cle: &str) -> Option<String> {
    get(cle).and_then(|v| v.as_str().map(str::to_string))
}

fn texte_requis(get: Lecture<'_>, cle: &str) -> Result<String> {
    chaine(get, cle).ok_or_else(|| anyhow!("File sans propriété `{cle}`"))
}

fn lire_fichier(get: Lecture<'_>) -> Result<FileNode> {
    let path = texte_requis(get, "path")?;
    let brut = texte_requis(get, "last_parsed")?;
    let last_parsed = brut
        .parse::<DateTime<Utc>>()
        .with_context(|| format!("File {path} : `last_parsed` illisible (« {brut} »)"))?;
    Ok(FileNode {
        language: texte_requis(get, "language")?,
        hash: texte_requis(get, "hash")?,
        last_parsed,
        // `""` is a file with no project, as the Neo4j writer stores it.
        project_id: chaine(get, "project_id").and_then(|s| s.parse().ok()),
        path,
    })
}

#[cfg(any(test, feature = "mock-store"))]
fn props_fichier(f: &FileNode) -> Vec<(&'static str, Value)> {
    vec![
        ("path", Value::from(f.path.clone())),
        ("language", Value::from(f.language.clone())),
        ("hash", Value::from(f.hash.clone())),
        ("last_parsed", Value::from(f.last_parsed.to_rfc3339())),
        (
            "project_id",
            Value::from(f.project_id.map(|p| p.to_string()).unwrap_or_default()),
        ),
    ]
}

/// Decode a node once it has been written (the index is then published from this).
#[cfg(any(test, feature = "mock-store"))]
fn relire(db: &ObrainDB, noeud: NodeId) -> Result<FileNode> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud {noeud:?} absent juste après son écriture"))?;
    let get = move |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    };
    lire_fichier(&get)
}

/// Does this value go to a blob column (a string above the threshold)? Such a value
/// is not recovered by the WAL: see `mod.rs` (`SEUIL_COLONNE_BLOB`).
#[cfg(any(test, feature = "mock-store"))]
fn valeur_en_blob(v: &Value) -> bool {
    v.as_str()
        .is_some_and(|s| s.len() > super::SEUIL_COLONNE_BLOB)
}

#[cfg(any(test, feature = "mock-store"))]
fn fichier_en_blob(f: &FileNode) -> bool {
    props_fichier(f).iter().any(|(_, v)| valeur_en_blob(v))
}

// ============================================================================
// The store
// ============================================================================

#[cfg(any(test, feature = "mock-store"))]
impl ObrainGraphStore {
    /// Create or update `files` (last one wins on a repeated path), link each to its
    /// project when that project exists, and make it all durable ONCE.
    ///
    /// Not reachable from the contract while the symbol writes refuse (see the
    /// module header): compiled for test builds only, through
    /// [`ObrainGraphStore::semer_fichiers_pour_essai`]. Each engine write is preceded
    /// by a crash point (`Interruption::ApresEtape`).
    async fn ecrire_fichiers(&self, files: Vec<FileNode>) -> Result<()> {
        if files.is_empty() {
            return Ok(());
        }
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // What the index says now: the existing nodes, the project nodes, the
                // memberships already there.
                let (existants, projets, membres) = {
                    let etat = w.index()?;
                    // A quarantined path would get a SECOND File beside the one held
                    // back: refused before anything is written.
                    etat.code.refus_si_quarantaine(
                        "CodeStore::upsert_file (quarantaine)",
                        files.iter().map(|f| &f.path),
                    )?;
                    let mut existants: HashMap<String, (NodeId, bool)> = HashMap::new();
                    let mut projets: HashMap<Uuid, NodeId> = HashMap::new();
                    let mut membres: HashSet<(Uuid, String)> = HashSet::new();
                    for f in &files {
                        if let Some(e) = etat.code.fichiers.get(&f.path) {
                            existants.insert(f.path.clone(), (e.noeud, fichier_en_blob(&e.valeur)));
                        }
                        if let Some(p) = f.project_id {
                            if let Some(n) = etat.projets.noeud_de(p) {
                                projets.insert(p, n);
                            }
                            if etat
                                .code
                                .contenus
                                .get(&p)
                                .is_some_and(|s| s.contains(&f.path))
                            {
                                membres.insert((p, f.path.clone()));
                            }
                        }
                    }
                    (existants, projets, membres)
                };

                let mut noeuds: HashMap<String, NodeId> = HashMap::new();
                let mut crees: Vec<NodeId> = Vec::new();
                let mut ajoutes: Vec<(Uuid, String)> = Vec::new();
                let mut dernier: HashMap<String, FileNode> = HashMap::new();
                let mut flush = false;
                for f in &files {
                    flush |= fichier_en_blob(f);
                    let noeud = match noeuds.get(&f.path).copied() {
                        Some(n) => n,
                        None => match existants.get(&f.path) {
                            Some((n, ancien_en_blob)) => {
                                // Replacing a value that was in a blob column clears that
                                // slot, which the WAL does not recover either.
                                flush |= *ancien_en_blob;
                                *n
                            }
                            None => {
                                w.etape()?;
                                let n = w.db.create_node(&[ETIQUETTE_FICHIER]);
                                w.etape()?;
                                w.db.set_node_property(n, BROUILLON, Value::from(true));
                                crees.push(n);
                                n
                            }
                        },
                    };
                    for (cle, valeur) in props_fichier(f) {
                        w.etape()?;
                        w.db.set_node_property(noeud, cle, valeur);
                    }
                    noeuds.insert(f.path.clone(), noeud);
                    if let Some(p) = f.project_id {
                        let cle = (p, f.path.clone());
                        if let Some(pn) = projets.get(&p) {
                            if !membres.contains(&cle) && !ajoutes.contains(&cle) {
                                // MERGE: a membership is not written twice.
                                w.etape()?;
                                let _ = w.db.create_edge(*pn, noeud, CONTIENT);
                                ajoutes.push(cle);
                            }
                        }
                    }
                    dernier.insert(f.path.clone(), f.clone());
                }
                // A value in a blob column reaches the disk at the full flush: do it
                // BEFORE the markers go, so a crash leaves drafts, never a half file.
                if flush {
                    w.flush_complet = true;
                    w.rendre_durable_avant_le_marqueur()?;
                }
                for n in &crees {
                    w.etape()?;
                    w.db.remove_node_property(*n, BROUILLON);
                }
                let mut publies: Vec<(String, Fichier)> = Vec::with_capacity(dernier.len());
                for path in dernier.keys() {
                    let noeud = noeuds[path];
                    let valeur = w.apres_ecriture(relire(w.db, noeud))?;
                    publies.push((path.clone(), Fichier { noeud, valeur }));
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for (path, f) in publies {
                    etat.code.fichiers.insert(path, f);
                }
                for (p, path) in ajoutes {
                    etat.code.contenus.entry(p).or_default().insert(path);
                }
                Ok(())
            },
        )
        .await
    }

    /// Write files through the writer the contract does not expose yet (see the
    /// module header): what the tests of the reads, the deletions and the crash
    /// points start from. Test builds only.
    #[doc(hidden)]
    pub async fn semer_fichiers_pour_essai(&self, files: &[FileNode]) -> Result<()> {
        self.ecrire_fichiers(files.to_vec()).await
    }

    /// Outgoing edges of a project's node as the ENGINE counts them (a draft deleted
    /// at open must take its `CONTAINS` edge with it). Test builds only.
    #[doc(hidden)]
    pub fn aretes_sortantes_du_projet_pour_essai(&self, id: Uuid) -> Option<usize> {
        let noeud = self.interne.etat.read().ok()?.projets.noeud_de(id)?;
        Some(self.interne.db.store().out_degree(noeud))
    }

    /// Delete a file's node in the ENGINE behind the index, so that the next
    /// `delete_node` of it answers `false` (an index that disagrees with the base).
    /// Test builds only.
    #[doc(hidden)]
    pub fn retirer_du_moteur_pour_essai(&self, path: &str) -> bool {
        let Some(noeud) = self
            .interne
            .etat
            .read()
            .ok()
            .and_then(|e| e.code.fichiers.get(path).map(|f| f.noeud))
        else {
            return false;
        };
        self.interne.db.delete_node(noeud)
    }
}

/// Delete a file's node. obrain detaches the edges BEFORE it deletes the node
/// (`ObrainDB::delete_node`), so a `false` can come after a write — and in a
/// multi-delete, the earlier nodes are gone: the store is then indeterminate (as
/// `note.rs` does for `delete_note`), and refuses everything until a restart rebuilds
/// the index from the base.
fn supprimer_fichier(w: &mut Ecriture<'_>, noeud: NodeId, path: &str) -> Result<()> {
    w.etape()?;
    if !w.db.delete_node(noeud) {
        w.interne
            .indetermine
            .store(true, std::sync::atomic::Ordering::Release);
        bail!(
            "obrain a refusé la suppression du nœud {noeud:?} (fichier {path}) : ses arêtes, \
             et les fichiers supprimés avant lui dans cet appel, peuvent être déjà partis. \
             Base INDÉTERMINÉE, tout est refusé jusqu'au redémarrage"
        );
    }
    Ok(())
}

/// The stale files of a project, and the refusal when one has symbols.
fn refus_si_symboles(db: &ObrainDB, noeuds: &[NodeId]) -> Result<()> {
    if noeuds.iter().any(|n| !voisins(db, *n, CONTIENT).is_empty()) {
        return Err(anyhow::Error::new(NonPorte("CodeStore::delete_file (symboles)")).context(
            "un fichier à supprimer porte des symboles (arêtes CONTAINS sortantes) que ce backend \
             n'adresse pas : les supprimer avec lui, ou les laisser orphelins, est une décision \
             qui attend le portage des symboles",
        ));
    }
    Ok(())
}

#[async_trait]
impl CodeStore for ObrainGraphStore {
    // ------------------------------------------------------------------------
    // Files
    // ------------------------------------------------------------------------

    async fn get_project_file_paths(&self, project_id: Uuid) -> Result<Vec<String>> {
        self.lire(|_, etat| {
            if !etat.projets.contient(project_id) {
                return Ok(Vec::new());
            }
            Ok(etat
                .code
                .contenus
                .get(&project_id)
                .map(|s| s.iter().cloned().collect())
                .unwrap_or_default())
        })
    }

    async fn delete_file(&self, path: &str) -> Result<()> {
        let path = path.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud = {
                    let etat = w.index()?;
                    etat.code.refus_si_quarantaine(
                        "CodeStore::delete_file (quarantaine)",
                        [&path].into_iter(),
                    )?;
                    let Some(noeud) = etat.code.fichiers.get(&path).map(|f| f.noeud) else {
                        return Ok(());
                    };
                    noeud
                };
                refus_si_symboles(w.db, &[noeud])?;
                supprimer_fichier(w, noeud, &path)?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.code.fichiers.remove(&path);
                for membres in etat.code.contenus.values_mut() {
                    membres.remove(&path);
                }
                Ok(())
            },
        )
        .await
    }

    async fn delete_stale_files(
        &self,
        project_id: Uuid,
        valid_paths: &[String],
    ) -> Result<(usize, usize, Vec<String>)> {
        let valides: HashSet<String> = valid_paths.iter().cloned().collect();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let perimees: Vec<(String, NodeId)> = {
                    let etat = w.index()?;
                    if !etat.projets.contient(project_id) {
                        return Ok((0, 0, Vec::new()));
                    }
                    // A quarantined path the project contains and that is stale, or a
                    // stale path that has a quarantined duplicate: refused, nothing
                    // written.
                    let perimes_en_quarantaine = etat
                        .code
                        .quarantaine_contenue
                        .get(&project_id)
                        .into_iter()
                        .flatten()
                        .filter(|p| !valides.contains(*p));
                    etat.code.refus_si_quarantaine(
                        "CodeStore::delete_stale_files (quarantaine)",
                        perimes_en_quarantaine.chain(
                            etat.code
                                .contenus
                                .get(&project_id)
                                .into_iter()
                                .flatten()
                                .filter(|p| !valides.contains(*p)),
                        ),
                    )?;
                    etat.code
                        .contenus
                        .get(&project_id)
                        .into_iter()
                        .flatten()
                        .filter(|p| !valides.contains(*p))
                        .filter_map(|p| etat.code.fichiers.get(p).map(|f| (p.clone(), f.noeud)))
                        .collect()
                };
                if perimees.is_empty() {
                    return Ok((0, 0, Vec::new()));
                }
                let noeuds: Vec<NodeId> = perimees.iter().map(|(_, n)| *n).collect();
                // Everything is checked BEFORE the first deletion.
                refus_si_symboles(w.db, &noeuds)?;
                for (path, noeud) in &perimees {
                    supprimer_fichier(w, *noeud, path)?;
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                let mut chemins = Vec::with_capacity(perimees.len());
                for (path, _) in perimees {
                    etat.code.fichiers.remove(&path);
                    for membres in etat.code.contenus.values_mut() {
                        membres.remove(&path);
                    }
                    chemins.push(path);
                }
                // Symbols are refused above when present: none was deleted with them.
                Ok((chemins.len(), 0, chemins))
            },
        )
        .await
    }

    async fn link_file_to_project(&self, file_path: &str, project_id: Uuid) -> Result<()> {
        let path = file_path.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (fichier, projet) = {
                    let etat = w.index()?;
                    etat.code.refus_si_quarantaine(
                        "CodeStore::link_file_to_project (quarantaine)",
                        [&path].into_iter(),
                    )?;
                    let (Some(f), Some(p)) = (
                        etat.code.fichiers.get(&path).map(|f| f.noeud),
                        etat.projets.noeud_de(project_id),
                    ) else {
                        return Ok(());
                    };
                    if etat
                        .code
                        .contenus
                        .get(&project_id)
                        .is_some_and(|s| s.contains(&path))
                    {
                        return Ok(());
                    }
                    (f, p)
                };
                let _ = w.db.create_edge(projet, fichier, CONTIENT);
                w.rendre_durable()?;
                w.index_mut()?
                    .code
                    .contenus
                    .entry(project_id)
                    .or_default()
                    .insert(path);
                Ok(())
            },
        )
        .await
    }

    async fn upsert_file(&self, _file: &FileNode) -> Result<()> {
        refus("CodeStore::upsert_file", RAISON_ECRITURE_FICHIER)
    }

    async fn batch_upsert_files(&self, _files: &[FileNode]) -> Result<()> {
        refus("CodeStore::batch_upsert_files", RAISON_ECRITURE_FICHIER)
    }

    async fn get_file(&self, path: &str) -> Result<Option<FileNode>> {
        self.lire(|_, etat| Ok(etat.code.fichiers.get(path).map(|f| f.valeur.clone())))
    }

    async fn find_files_by_path_suffix(
        &self,
        suffix: &str,
        limit: usize,
    ) -> Result<Vec<(String, Option<Uuid>)>> {
        self.lire(|_, etat| {
            let mut v: Vec<(String, Option<Uuid>)> = etat
                .code
                .fichiers
                .values()
                .filter(|f| f.valeur.path.ends_with(suffix))
                .map(|f| (f.valeur.path.clone(), f.valeur.project_id))
                .collect();
            v.sort();
            v.truncate(limit);
            Ok(v)
        })
    }

    async fn list_project_files(&self, project_id: Uuid) -> Result<Vec<FileNode>> {
        self.lire(|_, etat| {
            if !etat.projets.contient(project_id) {
                return Ok(Vec::new());
            }
            Ok(etat
                .code
                .contenus
                .get(&project_id)
                .into_iter()
                .flatten()
                .filter_map(|p| etat.code.fichiers.get(p))
                .map(|f| FileNode {
                    // The query returns the project it was asked for, not the property.
                    project_id: Some(project_id),
                    ..f.valeur.clone()
                })
                .collect())
        })
    }

    async fn count_project_files(&self, project_id: Uuid) -> Result<i64> {
        self.lire(|_, etat| {
            if !etat.projets.contient(project_id) {
                return Ok(0);
            }
            Ok(etat
                .code
                .contenus
                .get(&project_id)
                .map_or(0, |s| s.len() as i64))
        })
    }

    async fn get_file_language(&self, path: &str) -> Result<Option<String>> {
        self.lire(|_, etat| {
            Ok(etat
                .code
                .fichiers
                .get(path)
                .map(|f| f.valeur.language.clone()))
        })
    }

    async fn get_language_stats(&self) -> Result<Vec<LanguageStatsNode>> {
        self.lire(|_, etat| Ok(stats_de(etat.code.fichiers.values().map(|f| &f.valeur))))
    }

    async fn get_language_stats_for_project(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<LanguageStatsNode>> {
        self.lire(|_, etat| {
            // By the `project_id` PROPERTY, as the query does (not by the edge).
            Ok(stats_de(
                etat.code
                    .fichiers
                    .values()
                    .map(|f| &f.valeur)
                    .filter(|f| f.project_id == Some(project_id)),
            ))
        })
    }

    // ------------------------------------------------------------------------
    // Project switches (a property of the `Project` node)
    // ------------------------------------------------------------------------

    async fn set_scaffolding_override(&self, project_id: Uuid, level: Option<u8>) -> Result<()> {
        self.modifier_propriete_projet(
            project_id,
            "scaffolding_override",
            level.map(|l| Value::from(i64::from(l.min(4)))),
        )
        .await
    }

    async fn set_watch_enabled(&self, project_id: Uuid, enabled: bool) -> Result<()> {
        self.modifier_propriete_projet(project_id, "watch_enabled", Some(Value::from(enabled)))
            .await
    }

    async fn set_default_note_energy(&self, project_id: Uuid, energy: Option<f64>) -> Result<()> {
        self.modifier_propriete_projet(
            project_id,
            "default_note_energy",
            energy.map(|e| Value::from(e.clamp(0.0, 1.0))),
        )
        .await
    }

    // ------------------------------------------------------------------------
    // Refused by name (see `REFUSEES`)
    // ------------------------------------------------------------------------

    async fn count_orphan_files(&self, _project_id: Uuid) -> Result<i64> {
        refus(
            "CodeStore::count_orphan_files",
            "compte les fichiers sans arête IMPORTS ni enfant Function : ces arêtes et ces nœuds ne sont pas portés, tous les fichiers sembleraient orphelins",
        )
    }

    async fn count_co_change_pairs(&self, _project_id: Uuid, _min_count: i64) -> Result<i64> {
        refus(
            "CodeStore::count_co_change_pairs",
            "lit les arêtes CO_CHANGED, écrites par CommitStore::compute_co_changed (commits non portés) avec des propriétés (project_id, count) dont le schéma obrain est à décider ; un 0 serait indiscernable d'un calcul jamais fait",
        )
    }

    async fn count_protocol_states_transitions(&self, _project_id: Uuid) -> Result<(i64, i64)> {
        refus(
            "CodeStore::count_protocol_states_transitions",
            "lit les Protocol, ProtocolState et ProtocolTransition : domaine protocole non porté",
        )
    }

    async fn invalidate_computed_properties(
        &self,
        _project_id: Uuid,
        _paths: &[String],
    ) -> Result<u64> {
        refus(
            "CodeStore::invalidate_computed_properties",
            "marque `*_version = -1` sur les fichiers et leurs voisins 1-2 sauts par IMPORTS : arêtes et propriétés calculées non portées",
        )
    }

    async fn upsert_function(&self, _func: &FunctionNode) -> Result<()> {
        refus(
            "CodeStore::upsert_function",
            "les nœuds Function/Struct/Trait/Enum/Impl/Import ne sont pas portés : leur clé d'identité (id de fonction, chemin+nom), leur arête CONTAINS depuis le fichier et le sort des symboles d'un fichier supprimé sont à décider",
        )
    }

    async fn upsert_struct(&self, _s: &StructNode) -> Result<()> {
        refus(
            "CodeStore::upsert_struct",
            "les nœuds Function/Struct/Trait/Enum/Impl/Import ne sont pas portés : leur clé d'identité (id de fonction, chemin+nom), leur arête CONTAINS depuis le fichier et le sort des symboles d'un fichier supprimé sont à décider",
        )
    }

    async fn upsert_trait(&self, _t: &TraitNode) -> Result<()> {
        refus(
            "CodeStore::upsert_trait",
            "les nœuds Function/Struct/Trait/Enum/Impl/Import ne sont pas portés : leur clé d'identité (id de fonction, chemin+nom), leur arête CONTAINS depuis le fichier et le sort des symboles d'un fichier supprimé sont à décider",
        )
    }

    async fn find_trait_by_name(&self, _name: &str) -> Result<Option<String>> {
        refus(
            "CodeStore::find_trait_by_name",
            "les nœuds Function/Struct/Trait/Enum/Impl/Import ne sont pas portés : leur clé d'identité (id de fonction, chemin+nom), leur arête CONTAINS depuis le fichier et le sort des symboles d'un fichier supprimé sont à décider",
        )
    }

    async fn upsert_enum(&self, _e: &EnumNode) -> Result<()> {
        refus(
            "CodeStore::upsert_enum",
            "les nœuds Function/Struct/Trait/Enum/Impl/Import ne sont pas portés : leur clé d'identité (id de fonction, chemin+nom), leur arête CONTAINS depuis le fichier et le sort des symboles d'un fichier supprimé sont à décider",
        )
    }

    async fn upsert_impl(&self, _impl_node: &ImplNode) -> Result<()> {
        refus(
            "CodeStore::upsert_impl",
            "les nœuds Function/Struct/Trait/Enum/Impl/Import ne sont pas portés : leur clé d'identité (id de fonction, chemin+nom), leur arête CONTAINS depuis le fichier et le sort des symboles d'un fichier supprimé sont à décider",
        )
    }

    async fn create_import_relationship(
        &self,
        _from_file: &str,
        _to_file: &str,
        _import_path: &str,
    ) -> Result<()> {
        refus(
            "CodeStore::create_import_relationship",
            "les arêtes IMPORTS / IMPORTS_SYMBOL / CALLS / USES_TYPE / EXTENDS / IMPLEMENTS joignent des symboles qui ne sont pas portés ; sans leur cible, une écriture serait un succès fantôme",
        )
    }

    async fn upsert_import(&self, _import: &ImportNode) -> Result<()> {
        refus(
            "CodeStore::upsert_import",
            "les nœuds Function/Struct/Trait/Enum/Impl/Import ne sont pas portés : leur clé d'identité (id de fonction, chemin+nom), leur arête CONTAINS depuis le fichier et le sort des symboles d'un fichier supprimé sont à décider",
        )
    }

    async fn create_imports_symbol_relationship(
        &self,
        _import_id: &str,
        _symbol_name: &str,
        _project_id: Option<Uuid>,
    ) -> Result<()> {
        refus(
            "CodeStore::create_imports_symbol_relationship",
            "les arêtes IMPORTS / IMPORTS_SYMBOL / CALLS / USES_TYPE / EXTENDS / IMPLEMENTS joignent des symboles qui ne sont pas portés ; sans leur cible, une écriture serait un succès fantôme",
        )
    }

    async fn create_call_relationship(
        &self,
        _caller_id: &str,
        _callee_name: &str,
        _project_id: Option<Uuid>,
        _confidence: f64,
        _reason: &str,
    ) -> Result<()> {
        refus(
            "CodeStore::create_call_relationship",
            "les arêtes IMPORTS / IMPORTS_SYMBOL / CALLS / USES_TYPE / EXTENDS / IMPLEMENTS joignent des symboles qui ne sont pas portés ; sans leur cible, une écriture serait un succès fantôme",
        )
    }

    async fn batch_upsert_functions(&self, _functions: &[FunctionNode]) -> Result<()> {
        refus(
            "CodeStore::batch_upsert_functions",
            "les nœuds Function/Struct/Trait/Enum/Impl/Import ne sont pas portés : leur clé d'identité (id de fonction, chemin+nom), leur arête CONTAINS depuis le fichier et le sort des symboles d'un fichier supprimé sont à décider",
        )
    }

    async fn batch_upsert_structs(&self, _structs: &[StructNode]) -> Result<()> {
        refus(
            "CodeStore::batch_upsert_structs",
            "les nœuds Function/Struct/Trait/Enum/Impl/Import ne sont pas portés : leur clé d'identité (id de fonction, chemin+nom), leur arête CONTAINS depuis le fichier et le sort des symboles d'un fichier supprimé sont à décider",
        )
    }

    async fn batch_upsert_traits(&self, _traits: &[TraitNode]) -> Result<()> {
        refus(
            "CodeStore::batch_upsert_traits",
            "les nœuds Function/Struct/Trait/Enum/Impl/Import ne sont pas portés : leur clé d'identité (id de fonction, chemin+nom), leur arête CONTAINS depuis le fichier et le sort des symboles d'un fichier supprimé sont à décider",
        )
    }

    async fn batch_upsert_enums(&self, _enums: &[EnumNode]) -> Result<()> {
        refus(
            "CodeStore::batch_upsert_enums",
            "les nœuds Function/Struct/Trait/Enum/Impl/Import ne sont pas portés : leur clé d'identité (id de fonction, chemin+nom), leur arête CONTAINS depuis le fichier et le sort des symboles d'un fichier supprimé sont à décider",
        )
    }

    async fn batch_upsert_impls(&self, _impls: &[ImplNode]) -> Result<()> {
        refus(
            "CodeStore::batch_upsert_impls",
            "les nœuds Function/Struct/Trait/Enum/Impl/Import ne sont pas portés : leur clé d'identité (id de fonction, chemin+nom), leur arête CONTAINS depuis le fichier et le sort des symboles d'un fichier supprimé sont à décider",
        )
    }

    async fn batch_upsert_imports(&self, _imports: &[ImportNode]) -> Result<()> {
        refus(
            "CodeStore::batch_upsert_imports",
            "les nœuds Function/Struct/Trait/Enum/Impl/Import ne sont pas portés : leur clé d'identité (id de fonction, chemin+nom), leur arête CONTAINS depuis le fichier et le sort des symboles d'un fichier supprimé sont à décider",
        )
    }

    async fn batch_create_import_relationships(
        &self,
        _relationships: &[(String, String, String)],
    ) -> Result<()> {
        refus(
            "CodeStore::batch_create_import_relationships",
            "les arêtes IMPORTS / IMPORTS_SYMBOL / CALLS / USES_TYPE / EXTENDS / IMPLEMENTS joignent des symboles qui ne sont pas portés ; sans leur cible, une écriture serait un succès fantôme",
        )
    }

    async fn batch_create_imports_symbol_relationships(
        &self,
        _relationships: &[(String, String, Option<Uuid>)],
    ) -> Result<()> {
        refus(
            "CodeStore::batch_create_imports_symbol_relationships",
            "les arêtes IMPORTS / IMPORTS_SYMBOL / CALLS / USES_TYPE / EXTENDS / IMPLEMENTS joignent des symboles qui ne sont pas portés ; sans leur cible, une écriture serait un succès fantôme",
        )
    }

    async fn batch_create_call_relationships(
        &self,
        _calls: &[FunctionCall],
        _project_id: Option<Uuid>,
    ) -> Result<()> {
        refus(
            "CodeStore::batch_create_call_relationships",
            "les arêtes IMPORTS / IMPORTS_SYMBOL / CALLS / USES_TYPE / EXTENDS / IMPLEMENTS joignent des symboles qui ne sont pas portés ; sans leur cible, une écriture serait un succès fantôme",
        )
    }

    async fn batch_create_extends_relationships(
        &self,
        _rels: &[(String, String, String, String)],
    ) -> Result<()> {
        refus(
            "CodeStore::batch_create_extends_relationships",
            "les arêtes IMPORTS / IMPORTS_SYMBOL / CALLS / USES_TYPE / EXTENDS / IMPLEMENTS joignent des symboles qui ne sont pas portés ; sans leur cible, une écriture serait un succès fantôme",
        )
    }

    async fn batch_create_implements_relationships(
        &self,
        _rels: &[(String, String, String, String)],
    ) -> Result<()> {
        refus(
            "CodeStore::batch_create_implements_relationships",
            "les arêtes IMPORTS / IMPORTS_SYMBOL / CALLS / USES_TYPE / EXTENDS / IMPLEMENTS joignent des symboles qui ne sont pas portés ; sans leur cible, une écriture serait un succès fantôme",
        )
    }

    async fn cleanup_cross_project_calls(&self) -> Result<i64> {
        refus(
            "CodeStore::cleanup_cross_project_calls",
            "nettoyage ou migration d'arêtes CALLS / de données de synchronisation : balaie des symboles et des arêtes non portés",
        )
    }

    async fn cleanup_builtin_calls(&self) -> Result<i64> {
        refus(
            "CodeStore::cleanup_builtin_calls",
            "nettoyage ou migration d'arêtes CALLS / de données de synchronisation : balaie des symboles et des arêtes non portés",
        )
    }

    async fn migrate_calls_confidence(&self) -> Result<i64> {
        refus(
            "CodeStore::migrate_calls_confidence",
            "nettoyage ou migration d'arêtes CALLS / de données de synchronisation : balaie des symboles et des arêtes non portés",
        )
    }

    async fn cleanup_sync_data(&self) -> Result<i64> {
        refus(
            "CodeStore::cleanup_sync_data",
            "nettoyage ou migration d'arêtes CALLS / de données de synchronisation : balaie des symboles et des arêtes non portés",
        )
    }

    async fn get_callees(&self, _function_id: &str, _depth: u32) -> Result<Vec<FunctionNode>> {
        refus(
            "CodeStore::get_callees",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn create_uses_type_relationship(
        &self,
        _function_id: &str,
        _type_name: &str,
    ) -> Result<()> {
        refus(
            "CodeStore::create_uses_type_relationship",
            "les arêtes IMPORTS / IMPORTS_SYMBOL / CALLS / USES_TYPE / EXTENDS / IMPLEMENTS joignent des symboles qui ne sont pas portés ; sans leur cible, une écriture serait un succès fantôme",
        )
    }

    async fn find_trait_implementors(&self, _trait_name: &str) -> Result<Vec<String>> {
        refus(
            "CodeStore::find_trait_implementors",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_type_traits(&self, _type_name: &str) -> Result<Vec<String>> {
        refus(
            "CodeStore::get_type_traits",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_impl_blocks(&self, _type_name: &str) -> Result<Vec<serde_json::Value>> {
        refus(
            "CodeStore::get_impl_blocks",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_class_hierarchy(
        &self,
        _type_name: &str,
        _max_depth: u32,
    ) -> Result<serde_json::Value> {
        refus(
            "CodeStore::get_class_hierarchy",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn find_subclasses(&self, _class_name: &str) -> Result<Vec<serde_json::Value>> {
        refus(
            "CodeStore::find_subclasses",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn find_interface_implementors(
        &self,
        _interface_name: &str,
    ) -> Result<Vec<serde_json::Value>> {
        refus(
            "CodeStore::find_interface_implementors",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn list_processes(&self, _project_id: uuid::Uuid) -> Result<Vec<serde_json::Value>> {
        refus(
            "CodeStore::list_processes",
            "les processus (Process, STEP_IN_PROCESS) sont écrits par l'analyse, domaine non porté",
        )
    }

    async fn get_process_detail(&self, _process_id: &str) -> Result<Option<serde_json::Value>> {
        refus(
            "CodeStore::get_process_detail",
            "les processus (Process, STEP_IN_PROCESS) sont écrits par l'analyse, domaine non porté",
        )
    }

    async fn get_entry_points(
        &self,
        _project_id: uuid::Uuid,
        _limit: usize,
    ) -> Result<Vec<serde_json::Value>> {
        refus(
            "CodeStore::get_entry_points",
            "les processus (Process, STEP_IN_PROCESS) sont écrits par l'analyse, domaine non porté",
        )
    }

    async fn get_file_functions_summary(&self, _path: &str) -> Result<Vec<FunctionSummaryNode>> {
        refus(
            "CodeStore::get_file_functions_summary",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_file_structs_summary(&self, _path: &str) -> Result<Vec<StructSummaryNode>> {
        refus(
            "CodeStore::get_file_structs_summary",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_file_import_paths_list(&self, _path: &str) -> Result<Vec<String>> {
        refus(
            "CodeStore::get_file_import_paths_list",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn find_symbol_references(
        &self,
        _symbol: &str,
        _limit: usize,
        _project_id: Option<Uuid>,
    ) -> Result<Vec<SymbolReferenceNode>> {
        refus(
            "CodeStore::find_symbol_references",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_file_direct_imports(&self, _path: &str) -> Result<Vec<FileImportNode>> {
        refus(
            "CodeStore::get_file_direct_imports",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_function_callers_by_name(
        &self,
        _function_name: &str,
        _depth: u32,
        _project_id: Option<Uuid>,
    ) -> Result<Vec<String>> {
        refus(
            "CodeStore::get_function_callers_by_name",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_function_callees_by_name(
        &self,
        _function_name: &str,
        _depth: u32,
        _project_id: Option<Uuid>,
    ) -> Result<Vec<String>> {
        refus(
            "CodeStore::get_function_callees_by_name",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_callers_with_confidence(
        &self,
        _function_name: &str,
        _project_id: Option<Uuid>,
    ) -> Result<Vec<(String, String, f64, String)>> {
        refus(
            "CodeStore::get_callers_with_confidence",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_callees_with_confidence(
        &self,
        _function_name: &str,
        _project_id: Option<Uuid>,
    ) -> Result<Vec<(String, String, f64, String)>> {
        refus(
            "CodeStore::get_callees_with_confidence",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_most_connected_files(&self, _limit: usize) -> Result<Vec<String>> {
        refus(
            "CodeStore::get_most_connected_files",
            "métrique d'analyse de graphe (GDS : pagerank, betweenness, communautés, santé) calculée sur IMPORTS/CALLS et écrite sur les nœuds par l'analyse : ni les arêtes ni l'écrivain ne sont portés",
        )
    }

    async fn get_most_connected_files_detailed(
        &self,
        _limit: usize,
    ) -> Result<Vec<ConnectedFileNode>> {
        refus(
            "CodeStore::get_most_connected_files_detailed",
            "métrique d'analyse de graphe (GDS : pagerank, betweenness, communautés, santé) calculée sur IMPORTS/CALLS et écrite sur les nœuds par l'analyse : ni les arêtes ni l'écrivain ne sont portés",
        )
    }

    async fn get_most_connected_files_for_project(
        &self,
        _project_id: Uuid,
        _limit: usize,
    ) -> Result<Vec<ConnectedFileNode>> {
        refus(
            "CodeStore::get_most_connected_files_for_project",
            "métrique d'analyse de graphe (GDS : pagerank, betweenness, communautés, santé) calculée sur IMPORTS/CALLS et écrite sur les nœuds par l'analyse : ni les arêtes ni l'écrivain ne sont portés",
        )
    }

    async fn get_project_communities(&self, _project_id: Uuid) -> Result<Vec<CommunityRow>> {
        refus(
            "CodeStore::get_project_communities",
            "métrique d'analyse de graphe (GDS : pagerank, betweenness, communautés, santé) calculée sur IMPORTS/CALLS et écrite sur les nœuds par l'analyse : ni les arêtes ni l'écrivain ne sont portés",
        )
    }

    async fn get_node_analytics(
        &self,
        _identifier: &str,
        _node_type: &str,
    ) -> Result<Option<NodeAnalyticsRow>> {
        refus(
            "CodeStore::get_node_analytics",
            "métrique d'analyse de graphe (GDS : pagerank, betweenness, communautés, santé) calculée sur IMPORTS/CALLS et écrite sur les nœuds par l'analyse : ni les arêtes ni l'écrivain ne sont portés",
        )
    }

    async fn get_affected_communities(&self, _file_paths: &[String]) -> Result<Vec<String>> {
        refus(
            "CodeStore::get_affected_communities",
            "métrique d'analyse de graphe (GDS : pagerank, betweenness, communautés, santé) calculée sur IMPORTS/CALLS et écrite sur les nœuds par l'analyse : ni les arêtes ni l'écrivain ne sont portés",
        )
    }

    async fn get_code_health_report(
        &self,
        _project_id: Uuid,
        _god_function_threshold: usize,
    ) -> Result<crate::neo4j::models::CodeHealthReport> {
        refus(
            "CodeStore::get_code_health_report",
            "métrique d'analyse de graphe (GDS : pagerank, betweenness, communautés, santé) calculée sur IMPORTS/CALLS et écrite sur les nœuds par l'analyse : ni les arêtes ni l'écrivain ne sont portés",
        )
    }

    async fn compute_maintenance_snapshot(
        &self,
        _project_id: Uuid,
    ) -> Result<crate::neo4j::models::MaintenanceSnapshot> {
        refus(
            "CodeStore::compute_maintenance_snapshot",
            "métrique d'analyse de graphe (GDS : pagerank, betweenness, communautés, santé) calculée sur IMPORTS/CALLS et écrite sur les nœuds par l'analyse : ni les arêtes ni l'écrivain ne sont portés",
        )
    }

    async fn compute_scaffolding_level(
        &self,
        _project_id: Uuid,
        _scaffolding_override: Option<u8>,
    ) -> Result<crate::neo4j::models::ScaffoldingLevel> {
        refus(
            "CodeStore::compute_scaffolding_level",
            "agrège des notes, décisions, fichiers et symboles de plusieurs domaines (niveau d'échafaudage calculé) : domaines non portés",
        )
    }

    async fn detect_global_stagnation(
        &self,
        _project_id: Uuid,
    ) -> Result<crate::neo4j::models::StagnationReport> {
        refus(
            "CodeStore::detect_global_stagnation",
            "métrique d'analyse de graphe (GDS : pagerank, betweenness, communautés, santé) calculée sur IMPORTS/CALLS et écrite sur les nœuds par l'analyse : ni les arêtes ni l'écrivain ne sont portés",
        )
    }

    async fn get_circular_dependencies(&self, _project_id: Uuid) -> Result<Vec<Vec<String>>> {
        refus(
            "CodeStore::get_circular_dependencies",
            "métrique d'analyse de graphe (GDS : pagerank, betweenness, communautés, santé) calculée sur IMPORTS/CALLS et écrite sur les nœuds par l'analyse : ni les arêtes ni l'écrivain ne sont portés",
        )
    }

    async fn get_node_gds_metrics(
        &self,
        _node_path: &str,
        _node_type: &str,
        _project_id: Uuid,
    ) -> Result<Option<NodeGdsMetrics>> {
        refus(
            "CodeStore::get_node_gds_metrics",
            "métrique d'analyse de graphe (GDS : pagerank, betweenness, communautés, santé) calculée sur IMPORTS/CALLS et écrite sur les nœuds par l'analyse : ni les arêtes ni l'écrivain ne sont portés",
        )
    }

    async fn get_project_percentiles(&self, _project_id: Uuid) -> Result<ProjectPercentiles> {
        refus(
            "CodeStore::get_project_percentiles",
            "métrique d'analyse de graphe (GDS : pagerank, betweenness, communautés, santé) calculée sur IMPORTS/CALLS et écrite sur les nœuds par l'analyse : ni les arêtes ni l'écrivain ne sont portés",
        )
    }

    async fn get_top_bridges_by_betweenness(
        &self,
        _project_id: Uuid,
        _limit: usize,
    ) -> Result<Vec<BridgeFile>> {
        refus(
            "CodeStore::get_top_bridges_by_betweenness",
            "métrique d'analyse de graphe (GDS : pagerank, betweenness, communautés, santé) calculée sur IMPORTS/CALLS et écrite sur les nœuds par l'analyse : ni les arêtes ni l'écrivain ne sont portés",
        )
    }

    async fn get_file_symbol_names(&self, _path: &str) -> Result<FileSymbolNamesNode> {
        refus(
            "CodeStore::get_file_symbol_names",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_function_caller_count(
        &self,
        _function_name: &str,
        _project_id: Option<Uuid>,
    ) -> Result<i64> {
        refus(
            "CodeStore::get_function_caller_count",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_trait_info(&self, _trait_name: &str) -> Result<Option<TraitInfoNode>> {
        refus(
            "CodeStore::get_trait_info",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_trait_implementors_detailed(
        &self,
        _trait_name: &str,
    ) -> Result<Vec<TraitImplementorNode>> {
        refus(
            "CodeStore::get_trait_implementors_detailed",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_type_trait_implementations(
        &self,
        _type_name: &str,
    ) -> Result<Vec<TypeTraitInfoNode>> {
        refus(
            "CodeStore::get_type_trait_implementations",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }

    async fn get_type_impl_blocks_detailed(
        &self,
        _type_name: &str,
    ) -> Result<Vec<ImplBlockDetailNode>> {
        refus(
            "CodeStore::get_type_impl_blocks_detailed",
            "lit des symboles (Function/Struct/Trait/Enum/Impl) ou leurs arêtes, qui ne sont pas portés : une liste vide serait un faux résultat",
        )
    }
}

/// Files per language, most numerous first then by language (the query's order).
fn stats_de<'a>(files: impl Iterator<Item = &'a FileNode>) -> Vec<LanguageStatsNode> {
    let mut par_langue: HashMap<&str, usize> = HashMap::new();
    for f in files {
        *par_langue.entry(f.language.as_str()).or_default() += 1;
    }
    let mut v: Vec<LanguageStatsNode> = par_langue
        .into_iter()
        .map(|(language, file_count)| LanguageStatsNode {
            language: language.to_string(),
            file_count,
        })
        .collect();
    v.sort_by(|a, b| {
        b.file_count
            .cmp(&a.file_count)
            .then_with(|| a.language.cmp(&b.language))
    });
    v
}
