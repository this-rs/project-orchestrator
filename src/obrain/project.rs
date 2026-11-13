//! `ProjectStore` on obrain — the first ported domain.
//!
//! A project is one node labelled `Project` whose properties carry the SAME
//! names and the same lossy conventions as the Neo4j adapter
//! (`src/neo4j/project.rs`), which is the reference of the conformance suite:
//!
//! - creation stores `id`, `name`, `slug`, `root_path`, `canonical_path`,
//!   `declared_path`, `paths_declared = true`, `description`, `created_at`,
//!   `watch_enabled`, `profile` — and nothing else: the other fields of
//!   `ProjectNode` are only ever written by their own setters;
//! - an absent description is stored as `""` and reads back as `Some("")`;
//! - an update or a timestamp on an unknown project is a silent no-op.
//!
//! Two deliberate differences, both on the strict side:
//!
//! - timestamps are stored as RFC 3339 **strings** (obrain has temporal types,
//!   but the decoding stays a plain string, as the Cypher/GQL mapping
//!   recommends). A `created_at` that does not parse makes the node
//!   unaddressable (see "Opening"); the Neo4j adapter replaces it with "now";
//! - `delete_project` deletes only while the base holds nothing but `Project`
//!   nodes. The Neo4j cascade walks plans, tasks, notes, files, feature graphs,
//!   commits (linked by edge OR by a `project_id` property) — domains this
//!   backend has not ported and therefore cannot have written. Any other node
//!   in the base means data from elsewhere (a migrated base) that the cascade
//!   would have to reach: the delete is refused by name rather than leaving
//!   orphans. The check is one counter (O(1)); it neither scans edges nor
//!   properties.
//!
//! Order: `list_projects` sorts by `(name, id)` ascending — the total order
//! `ORDER BY p.name, p.id ASC` states. Names compare by Unicode code point;
//! the id compares as a `Uuid`, which is the order of its canonical string
//! (fixed-width lowercase hex, same positions for the hyphens).
//!
//! # Reads are served from memory, and a write is visible once durable
//!
//! The decoded `ProjectNode` and its `PathPolicy` are held in the index, decoded
//! once (at open, and again after each write to the node): a read is a hash
//! lookup and a clone, with no engine call and no disk, whatever the size of the
//! base. That is sound because this backend is the single writer of the base —
//! the one thing that could change a node behind the index — and the index is
//! rebuilt from the base at every open.
//!
//! Every write (create, update, timestamp, delete) is published to the index
//! AFTER its durability step: a reader never sees a value, nor an absence, that
//! a restart could take back. Until then it is served the previous state.
//!
//! # Atomicity, built by ordering
//!
//! obrain's direct API writes one WAL record per property and has no
//! multi-property transaction. A creation killed half-way would leave a
//! `Project` node missing its `id`, or its `created_at`. So:
//!
//! - **create**: the node is created, the draft marker `_brouillon` is set
//!   FIRST, the eleven properties follow, the marker is removed LAST. The WAL
//!   is replayed in order, so a surviving record implies the ones before it. A
//!   value in a blob column is not in the WAL: when the creation carries one, the
//!   full flush runs BEFORE the marker is removed (measured: a killed creation
//!   otherwise reopened as a project that looked whole and had lost its description).
//!   **The removal of the marker is the commit point**, and the fsync that
//!   follows is the acknowledgement. A crash between the two leaves a complete,
//!   unmarked project that opening PROMOTES although the caller never got its
//!   `Ok`: the "committed, acknowledgement lost" window every journaled store
//!   has. A crash before the removal leaves a draft that opening DELETES, whatever
//!   it holds — even all eleven properties: without the commit point, it is not a
//!   project (all-or-nothing, like Neo4j's transaction).
//! - **update of the root**: `declared_path` is written FIRST, `canonical_path`
//!   second, `root_path` LAST. Both derived paths are pure functions of
//!   `root_path`; `declared_root_of` has no side effect and `canonical_root`
//!   resolves links. Any interruption therefore leaves `declared_path` disagreeing
//!   with `declared_root_of(root_path)` — the one pure check opening makes — and
//!   opening recomputes both derived paths from `root_path`. No reachable
//!   interruption leaves `canonical_path` alone out of step (it is never written
//!   before `declared_path`); a base from another writer that does is caught by the
//!   second pure check: an empty `canonical_path` on a project with a root.
//! - **opening** repairs, and never fails on what it repairs. What it cannot
//!   address (foreign data without `id`/`name`/`slug`/`created_at`, a duplicated id
//!   or slug) is quarantined: left untouched, ignored, and named in
//!   [`RapportOuverture`](super::RapportOuverture).
//!
//! What this does NOT give: an UPDATE of a blob-sized value (a string above 256
//! bytes) killed before its full flush may lose that property altogether — neither
//! the old value nor the new — because obrain's blob columns are not WAL-covered
//! (an obrain lot: log blob writes, or persist the column registry with each
//! write). And the properties of ONE update land one by one, so after a crash the
//! base can hold `name` new and `description` old. Each state is a valid project,
//! and the reopened base is one of them. A real transactional multi-property write
//! is an obrain lot (the `Session` has `create_node_with_props` but no
//! transactional `set_node_property`).

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

#[cfg(any(test, feature = "mock-store"))]
use super::Interruption;
use super::{Ecriture, NonPorte, ObrainGraphStore, RapportOuverture, SEUIL_COLONNE_BLOB};
use crate::neo4j::file_uid::{canonical_root, declared_root_of, PathPolicy};
use crate::neo4j::models::{ProjectNode, ProjectProfile};
use crate::neo4j::traits::ProjectStore;

const LABEL: &str = "Project";

/// Does this value go to a blob column (a string above the threshold)?
fn va_en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
}
/// The draft marker: present from the first record of a creation to the last.
const BROUILLON: &str = "_brouillon";
/// The properties a creation writes, in the order it writes them.
const CLES_DE_CREATION: [&str; 11] = [
    "id",
    "name",
    "slug",
    "root_path",
    "canonical_path",
    "declared_path",
    "paths_declared",
    "description",
    "created_at",
    "watch_enabled",
    "profile",
];

/// One project: where it lives in the base, and what the base says about it,
/// decoded once. Reads are served from here.
struct Entree {
    noeud: NodeId,
    projet: ProjectNode,
    politique: PathPolicy,
}

impl Entree {
    /// Read the node (one hydration) and decode it.
    fn decoder(db: &ObrainDB, noeud: NodeId) -> Result<Self> {
        let get = noeud_lisible(db, noeud)?;
        let projet = lire_projet(&get)?;
        let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
        let politique = PathPolicy::new(
            &chaine("declared_path")
                .or_else(|| chaine("root_path"))
                .unwrap_or_default(),
            &chaine("canonical_path").unwrap_or_default(),
            get("paths_declared")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        );
        Ok(Self {
            noeud,
            projet,
            politique,
        })
    }
}

/// `Uuid → project` and `slug → Uuid` for the `Project` nodes.
#[derive(Default)]
pub(super) struct IndexProjets {
    par_id: HashMap<Uuid, Entree>,
    par_slug: HashMap<String, Uuid>,
}

impl IndexProjets {
    /// The engine node of a project this backend addresses (used by the other
    /// domains that attach to a project).
    pub(super) fn noeud_de(&self, id: Uuid) -> Option<NodeId> {
        self.par_id.get(&id).map(|e| e.noeud)
    }

    /// Is there a project of this id? Read by the other domains' creations (a
    /// release, an environment, a milestone belongs to a project).
    pub(super) fn contient(&self, id: Uuid) -> bool {
        self.par_id.contains_key(&id)
    }

    /// Every project, as the readers are served them (unsorted).
    pub(super) fn tous(&self) -> Vec<ProjectNode> {
        self.par_id.values().map(|e| e.projet.clone()).collect()
    }

    /// The project `id`, as the index holds it (decoded, never a read of the engine).
    pub(super) fn projet(&self, id: Uuid) -> Option<&ProjectNode> {
        self.par_id.get(&id).map(|e| &e.projet)
    }

    /// Every project this backend addresses, with its engine node.
    pub(super) fn entrees(&self) -> impl Iterator<Item = (Uuid, NodeId)> + '_ {
        self.par_id.iter().map(|(id, e)| (*id, e.noeud))
    }
    /// Rebuild the index from the persisted `Project` nodes, repairing what an
    /// interrupted write left (see the module doc). A scan of the node zone.
    pub(super) fn reconstruire(db: &ObrainDB) -> Result<(Self, RapportOuverture)> {
        let store = db.store();
        let mut index = Self::default();
        let mut rapport = RapportOuverture::default();
        for noeud in store.nodes_by_label(LABEL) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud Project {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            // A draft is a creation that never reached its commit point: deleted,
            // whatever it holds (module doc). An empty node is the same creation,
            // killed even earlier.
            if get(BROUILLON).is_some() || (n.properties.is_empty() && store.out_degree(noeud) == 0)
            {
                db.delete_node(noeud);
                rapport.brouillons_supprimes += 1;
                continue;
            }
            let Ok(projet) = lire_projet(&get) else {
                // Not ours: no marker, and not addressable.
                rapport.en_quarantaine.push(format!(
                    "nœud Project {noeud:?} : id/name/slug/created_at illisibles"
                ));
                continue;
            };

            if index.par_id.contains_key(&projet.id) {
                rapport.en_quarantaine.push(format!(
                    "nœud Project {noeud:?} : id {} déjà pris",
                    projet.id
                ));
                continue;
            }
            if let Some(autre) = index.par_slug.get(&projet.slug) {
                rapport.en_quarantaine.push(format!(
                    "nœud Project {noeud:?} : slug « {} » déjà celui du projet {autre}",
                    projet.slug
                ));
                continue;
            }

            // A torn path trio. `root_path` is the commit point of an update, and
            // `declared_path` the first thing an update writes: any interruption
            // leaves them disagreeing. Both checks are pure (no filesystem).
            let declare = get("declared_path").and_then(|v| v.as_str().map(str::to_string));
            let canonique = get("canonical_path").and_then(|v| v.as_str().map(str::to_string));
            let attendu = declared_root_of(&projet.root_path);
            let dechire = declare.as_deref() != Some(attendu.as_str())
                || (!attendu.is_empty() && canonique.as_deref().unwrap_or("").is_empty());
            if dechire {
                // Recomputed from `root_path`: the one value every state agrees on.
                db.set_node_property(
                    noeud,
                    "canonical_path",
                    Value::from(canonical_root(Some(&projet.root_path))),
                );
                db.set_node_property(noeud, "declared_path", Value::from(attendu));
                rapport.chemins_repares += 1;
            }
            // Decoded AFTER the repairs: what is served is what the base now says.
            index.par_id.insert(projet.id, Entree::decoder(db, noeud)?);
            index.par_slug.insert(projet.slug, projet.id);
        }
        Ok((index, rapport))
    }
}

/// The domain value of a `Project` node from a property lookup — the
/// counterpart of `Neo4jClient::node_to_project`, field for field. A lookup
/// answers `None` for an absent or null property.
fn lire_projet(get: &dyn Fn(&str) -> Option<Value>) -> Result<ProjectNode> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let instant = |cle: &str| chaine(cle).and_then(|s| s.parse::<DateTime<Utc>>().ok());

    let id: Uuid = chaine("id")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("nœud Project sans propriété `id` lisible"))?;
    let created_at =
        chaine("created_at").ok_or_else(|| anyhow!("projet {id} sans propriété `created_at`"))?;
    let created_at: DateTime<Utc> = created_at
        .parse()
        .map_err(|e| anyhow!("projet {id} : `created_at` illisible (« {created_at} ») : {e}"))?;

    Ok(ProjectNode {
        id,
        name: chaine("name").ok_or_else(|| anyhow!("projet {id} sans propriété `name`"))?,
        slug: chaine("slug").ok_or_else(|| anyhow!("projet {id} sans propriété `slug`"))?,
        // Absent or empty for a project with no codebase.
        root_path: chaine("root_path").unwrap_or_default(),
        description: chaine("description"),
        created_at,
        last_synced: instant("last_synced"),
        analytics_computed_at: instant("analytics_computed_at"),
        last_co_change_computed_at: instant("last_co_change_computed_at"),
        // None means default (1.0).
        default_note_energy: get("default_note_energy")
            .and_then(|v| v.as_float64())
            .filter(|&v| v < 1.0),
        scaffolding_override: get("scaffolding_override")
            .and_then(|v| v.as_int64())
            .map(|v| v as u8),
        sharing_policy: chaine("sharing_policy").and_then(|s| serde_json::from_str(&s).ok()),
        watch_enabled: get("watch_enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        // A node without the property predates profiles: it is software.
        profile: chaine("profile")
            .and_then(|s| s.parse::<ProjectProfile>().ok())
            .unwrap_or_default(),
    })
}

/// A property lookup over ONE hydration of the node (`get_node` reads every
/// property once; the lookups are then hash probes).
pub(super) fn noeud_lisible(
    db: &ObrainDB,
    noeud: NodeId,
) -> Result<impl Fn(&str) -> Option<Value>> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud Project {noeud:?} indexé mais absent de la base"))?;
    Ok(move |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    })
}

/// The three path values of a root, in the order an update writes them:
/// `declared_path`, `canonical_path`, `root_path` (module doc). Computed on the
/// blocking pool, before any lock: `canonical_root` resolves links.
fn trio_de_chemins(racine: String) -> [(&'static str, Value); 3] {
    [
        ("declared_path", Value::from(declared_root_of(&racine))),
        ("canonical_path", Value::from(canonical_root(Some(&racine)))),
        ("root_path", Value::from(racine)),
    ]
}

#[async_trait]
impl ProjectStore for ObrainGraphStore {
    async fn create_project(&self, project: &ProjectNode) -> Result<()> {
        let p = project.clone();
        self.ecrire(
            // Prepared on the blocking pool, before the writer lock: the filesystem
            // is consulted here (`canonical_root` resolves links).
            move || {
                let proprietes: Vec<(&'static str, Value)> = vec![
                    ("id", Value::from(p.id.to_string())),
                    ("name", Value::from(p.name.clone())),
                    ("slug", Value::from(p.slug.clone())),
                    ("root_path", Value::from(p.root_path.clone())),
                    (
                        "canonical_path",
                        Value::from(canonical_root(Some(&p.root_path))),
                    ),
                    ("declared_path", Value::from(declared_root_of(&p.root_path))),
                    ("paths_declared", Value::from(true)),
                    (
                        "description",
                        Value::from(p.description.clone().unwrap_or_default()),
                    ),
                    ("created_at", Value::from(p.created_at.to_rfc3339())),
                    ("watch_enabled", Value::from(p.watch_enabled)),
                    ("profile", Value::from(p.profile.as_str())),
                ];
                debug_assert!(proprietes
                    .iter()
                    .map(|(k, _)| *k)
                    .eq(CLES_DE_CREATION.iter().copied()));
                (p, proprietes)
            },
            move |(p, proprietes), w: &mut Ecriture<'_>| {
                // Validate first: nothing is written when one of these refuses.
                // The writer lock is ours, so the answer cannot change under us.
                {
                    let etat = w.index()?;
                    if etat.projets.par_id.contains_key(&p.id) {
                        bail!(
                            "un projet d'id {} existe déjà (unicité de Project.id)",
                            p.id
                        );
                    }
                    if let Some(autre) = etat.projets.par_slug.get(&p.slug) {
                        bail!(
                            "le slug « {} » est déjà celui du projet {autre} (unicité de \
                             Project.slug)",
                            p.slug
                        );
                    }
                }
                w.flush_complet = proprietes.iter().any(|(_, v)| va_en_colonne_blob(v));
                let noeud = w.db.create_node(&[LABEL]);
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeNoeud)?;
                w.db.set_node_property(noeud, BROUILLON, Value::from(true));
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeMarqueur)?;
                for (n, (cle, valeur)) in proprietes.into_iter().enumerate() {
                    w.db.set_node_property(noeud, cle, valeur);
                    #[cfg(any(test, feature = "mock-store"))]
                    w.point(Interruption::ApresLaPropriete(n + 1))?;
                }
                // A value in a blob column is only recovered by the full flush: it
                // must be on disk BEFORE the marker says the project is complete.
                w.rendre_durable_avant_le_marqueur()?;
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::AvantLeRetraitDuMarqueur)?;
                // The commit point.
                w.db.remove_node_property(noeud, BROUILLON);
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeRetraitDuMarqueur)?;
                // What the readers will be served is what the base now says.
                let entree = w.apres_ecriture(Entree::decoder(w.db, noeud))?;
                // Durable BEFORE it is visible: a reader never sees a project that
                // a restart could lose.
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.projets.par_slug.insert(p.slug, p.id);
                etat.projets.par_id.insert(p.id, entree);
                Ok(())
            },
        )
        .await
    }

    // The reads below touch no disk and no engine: the decoded project is held in
    // the index, published by every write once durable (this backend is the single
    // writer of the base, and the index is rebuilt from it at open).

    async fn get_project(&self, id: Uuid) -> Result<Option<ProjectNode>> {
        self.lire(|_, etat| Ok(etat.projets.par_id.get(&id).map(|e| e.projet.clone())))
    }

    async fn path_policy(&self, id: Uuid) -> Result<PathPolicy> {
        self.lire(|_, etat| {
            // A project that does not exist has no vocabulary.
            Ok(etat
                .projets
                .par_id
                .get(&id)
                .map_or_else(PathPolicy::none, |e| e.politique.clone()))
        })
    }

    async fn get_project_by_slug(&self, slug: &str) -> Result<Option<ProjectNode>> {
        self.lire(|_, etat| {
            Ok(etat
                .projets
                .par_slug
                .get(slug)
                .and_then(|id| etat.projets.par_id.get(id))
                .map(|e| e.projet.clone()))
        })
    }

    async fn list_projects(&self) -> Result<Vec<ProjectNode>> {
        self.lire(|_, etat| {
            let mut projets: Vec<ProjectNode> = etat
                .projets
                .par_id
                .values()
                .map(|e| e.projet.clone())
                .collect();
            // ORDER BY p.name, p.id ASC — total: the id breaks every tie. `Uuid`
            // order is the order of its canonical string (module doc).
            projets.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.id.cmp(&b.id)));
            Ok(projets)
        })
    }

    async fn update_project(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<Option<String>>,
        root_path: Option<String>,
        profile: Option<ProjectProfile>,
    ) -> Result<()> {
        self.ecrire(
            // The filesystem is consulted on the blocking pool, before any lock.
            move || root_path.map(trio_de_chemins),
            move |chemins, w: &mut Ecriture<'_>| {
                let (noeud, ancien_en_blob) = {
                    let etat = w.index()?;
                    let Some(e) = etat.projets.par_id.get(&id) else {
                        return Ok(());
                    };
                    // Replacing a value that WAS in a blob column clears that slot,
                    // which the WAL does not recover either: the old value counts too.
                    let ancien = (name.is_some() && e.projet.name.len() > SEUIL_COLONNE_BLOB)
                        || (description.is_some()
                            && e.projet
                                .description
                                .as_ref()
                                .is_some_and(|d| d.len() > SEUIL_COLONNE_BLOB))
                        || (chemins.is_some()
                            && [
                                e.projet.root_path.as_str(),
                                e.politique.declared_root.as_str(),
                                e.politique.canonical_root.as_str(),
                            ]
                            .iter()
                            .any(|c| c.len() > SEUIL_COLONNE_BLOB));
                    (e.noeud, ancien)
                };
                w.flush_complet = ancien_en_blob
                    || name.iter().any(|n| n.len() > SEUIL_COLONNE_BLOB)
                    || description
                        .iter()
                        .flatten()
                        .any(|d| d.len() > SEUIL_COLONNE_BLOB)
                    || chemins.iter().flatten().any(|(_, v)| va_en_colonne_blob(v));
                // The engine writes: readers keep being served the previous value
                // from the index until this write is durable.
                if let Some(name) = name {
                    w.db.set_node_property(noeud, "name", Value::from(name));
                }
                if let Some(description) = description {
                    w.db.set_node_property(
                        noeud,
                        "description",
                        Value::from(description.unwrap_or_default()),
                    );
                }
                if let Some(profile) = profile {
                    w.db.set_node_property(noeud, "profile", Value::from(profile.as_str()));
                }
                if let Some(trio) = chemins {
                    // `declared_path`, `canonical_path`, then `root_path` LAST: the
                    // commit point an interrupted update is recomputed from at open.
                    for (n, (cle, valeur)) in trio.into_iter().enumerate() {
                        w.db.set_node_property(noeud, cle, valeur);
                        #[cfg(any(test, feature = "mock-store"))]
                        w.point(Interruption::ApresLeChemin(n + 1))?;
                    }
                }
                let entree = w.apres_ecriture(Entree::decoder(w.db, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.projets.par_id.insert(id, entree);
                Ok(())
            },
        )
        .await
    }

    async fn update_project_synced(&self, id: Uuid) -> Result<()> {
        self.horodater(id, "last_synced").await
    }

    async fn update_project_analytics_timestamp(&self, id: Uuid) -> Result<()> {
        self.horodater(id, "analytics_computed_at").await
    }

    async fn delete_project(&self, id: Uuid, _project_name: &str) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, slug, projets) = {
                    let etat = w.index()?;
                    let Some(e) = etat.projets.par_id.get(&id) else {
                        return Ok(());
                    };
                    (e.noeud, e.projet.slug.clone(), etat.projets.par_id.len())
                };
                // One counter, no scan: every live node of the base against the
                // projects this backend addresses. `project_name` only tags the notes
                // and decisions the cascade archives: part of what is refused.
                let noeuds = w.db.store().node_count();
                if noeuds != projets {
                    return Err(anyhow::Error::new(NonPorte(
                        "ProjectStore::delete_project (cascade)",
                    ))
                    .context(format!(
                        "la base contient {} nœud(s) hors des {projets} projet(s) que ce \
                         backend adresse : des données écrites hors de lui (plans, tâches, \
                         notes, fichiers, graphes de fonctionnalités, commits…) que la \
                         suppression en cascade devrait atteindre, par relation ou par \
                         `project_id`. Elle attend le portage des domaines concernés",
                        noeuds.saturating_sub(projets)
                    )));
                }
                if !w.db.delete_node(noeud) {
                    bail!("obrain a refusé la suppression du nœud {noeud:?} (projet {id})");
                }
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLaSuppression)?;
                // Gone for the readers once the deletion is durable, not before.
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.projets.par_id.remove(&id);
                etat.projets.par_slug.remove(&slug);
                Ok(())
            },
        )
        .await
    }
}

impl ObrainGraphStore {
    /// Set `cle` to now on the project, if it exists.
    async fn horodater(&self, id: Uuid, cle: &'static str) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = w.index()?.projets.par_id.get(&id).map(|e| e.noeud) else {
                    return Ok(());
                };
                w.db.set_node_property(noeud, cle, Value::from(Utc::now().to_rfc3339()));
                let entree = w.apres_ecriture(Entree::decoder(w.db, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.projets.par_id.insert(id, entree);
                Ok(())
            },
        )
        .await
    }
}
