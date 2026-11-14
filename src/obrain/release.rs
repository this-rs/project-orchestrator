//! `ReleaseStore` on obrain — the second ported domain (vague V2).
//!
//! What is ported: the releases, the environments, their deployments and the
//! milestones, with the lists that read them. What is refused by name
//! ([`REFUSEES`]): every method that reads or links a `Task`, a `Plan`, a
//! `Commit` or a `Step`, which belong to domains not ported yet. A port that
//! answered those with an empty list or a silent `Ok` would be a false success,
//! so they answer `NonPorte` instead.
//!
//! The values, the order and the errors are those of the Neo4j adapter
//! (`src/neo4j/release.rs`, `environment.rs`, `milestone.rs`, `project.rs`), which
//! is the reference of the conformance suite. Stored shapes match too: the same
//! property names, the same `""` for an absent optional string, the same
//! snake_case statuses. Differences, all on the strict side, are named here:
//!
//! - a `created_at` / `started_at` that does not parse makes the node
//!   unaddressable (quarantined at open) instead of being replaced by "now";
//! - a second release, environment or milestone with an existing id is refused
//!   (Neo4j would keep two nodes with the same id);
//! - the Commit link of a deployment (`DEPLOYS`) is NOT written: commits are not
//!   ported, so there is no node to link to. This is a limit, not a refusal: the
//!   deployment itself is complete, and only a query starting from a commit would
//!   see the missing edge.
//! - a release or a milestone without a `status` is quarantined at open (Neo4j's
//!   read fails on it too). An UNKNOWN stored status decodes as `planned` /
//!   `open`, as on Neo4j, but the status filters compare the DECODED status where
//!   Neo4j compares the stored string: such a node passes the filter `planned` /
//!   `open` here and no filter there (a migrated base only: this backend writes
//!   the snake_case spelling);
//! - an update writes its fields one by one (one WAL record each, obrain has no
//!   multi-property write): `update_milestone(status=closed, closed_at=now)` killed
//!   between the two can reopen `closed` with no `closed_at`, a state Neo4j's
//!   single `SET` never produces. Each field is valid on its own; the limit is the
//!   one `project.rs` describes;
//! - `delete_environment` deletes the environment and its deployments as ONE
//!   operation, but in several engine records: the environment is marked
//!   `_suppression` FIRST, its deployments are deleted, the environment LAST. A
//!   restart that finds the marker finishes the deletion in the same order —
//!   deployments, then the environment, so a cut during that repair leaves the
//!   marker for the next open (counted in `RapportOuverture::suppressions_achevees`):
//!   a cut never leaves an environment with part of its deployments, nor
//!   deployments of a deleted environment, which Neo4j's `DETACH DELETE d, e`
//!   never does either. The deployments taken are every `Deployment` naming the
//!   environment, quarantined ones included.
//!
//! Same reference defect, mirrored on purpose and flagged: the Neo4j search
//! filter of `list_projects_filtered` reads `p.title`, which projects do not
//! have, so the search matches the DESCRIPTION only. The conformance suite asserts
//! the Neo4j values, so this port keeps them; fixing it is a decision for both.
//!
//! # Storage and reads
//!
//! Each entity is one node (`Release`, `Environment`, `Deployment`, `Milestone`)
//! carrying its properties. Its parent link (project or environment) is a property
//! (`project_id`, `environment_id`), not an edge: the index holds the children of
//! each parent. Reads are served from the in-memory index, decoded from the base
//! at open and after each write, exactly like the projects (see `project.rs`).
//!
//! Writes go through the same `ecrire` as the projects: one WAL fsync per write, the
//! full flush for a value in a blob column, the draft marker `_brouillon` removed
//! last (a creation killed half-way is deleted at open), publication to the index
//! only after the durability step.
//!
//! Costs: a get is a hash lookup; a list of a project's releases is O(k log k) in
//! its releases k; `list_projects_filtered` is O(P log P) in the number of projects
//! (it is sorted per call, as Neo4j sorts it per call). Open rebuilds the index by a
//! scan of the four labels: O(releases + environments + deployments + milestones),
//! not O(all nodes). That scan is the one open cost that grows with this domain.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, SecondsFormat, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use serde::Serialize;
use uuid::Uuid;

#[cfg(any(test, feature = "mock-store"))]
use super::Interruption;
use super::{Ecriture, NonPorte, ObrainGraphStore, RapportOuverture, SEUIL_COLONNE_BLOB};
use crate::neo4j::client::{pascal_to_snake_case, status_variants};
use crate::neo4j::models::*;
use crate::neo4j::traits::ReleaseStore;

const ETIQUETTE_RELEASE: &str = "Release";
const ETIQUETTE_ENVIRONNEMENT: &str = "Environment";
const ETIQUETTE_DEPLOIEMENT: &str = "Deployment";
const ETIQUETTE_JALON: &str = "Milestone";
/// The draft marker: present from the first record of a creation to the last.
const BROUILLON: &str = "_brouillon";
/// The deletion marker of an environment: set before its deployments are
/// deleted, gone with the environment. Opening finishes what carries it.
const SUPPRESSION: &str = "_suppression";

/// The methods this domain refuses by name. They are written (so the build sees
/// them as implemented) but they answer [`NonPorte`]; see `mod.rs`
/// (`methodes_qui_servent`).
pub const REFUSEES: &[&str] = &[
    "ReleaseStore::add_task_to_release",
    "ReleaseStore::add_commit_to_release",
    "ReleaseStore::remove_commit_from_release",
    "ReleaseStore::get_release_details",
    "ReleaseStore::get_release_tasks",
    "ReleaseStore::add_task_to_milestone",
    "ReleaseStore::link_plan_to_milestone",
    "ReleaseStore::unlink_plan_from_milestone",
    "ReleaseStore::get_milestone_details",
    "ReleaseStore::get_milestone_progress",
    "ReleaseStore::get_milestone_tasks_with_plans",
    "ReleaseStore::get_milestone_steps_batch",
    "ReleaseStore::get_milestone_tasks",
    "ReleaseStore::get_project_progress",
    "ReleaseStore::get_project_task_dependencies",
    "ReleaseStore::get_project_tasks",
    "ReleaseStore::list_plans_filtered",
    "ReleaseStore::list_all_tasks_filtered",
];

/// A property lookup over one node: `None` for an absent or null property.
type Lecture<'a> = &'a dyn Fn(&str) -> Option<Value>;

// ============================================================================
// Index
// ============================================================================

/// One entity held in the index: where it lives, and what it decodes to.
pub(super) struct Entree<T> {
    noeud: NodeId,
    valeur: T,
}

/// The index of the four entities of this domain, rebuilt from the base at open.
#[derive(Default)]
pub(super) struct IndexReleases {
    releases: HashMap<Uuid, Entree<ReleaseNode>>,
    releases_par_projet: HashMap<Uuid, Vec<Uuid>>,
    environnements: HashMap<Uuid, Entree<EnvironmentNode>>,
    environnements_par_projet: HashMap<Uuid, Vec<Uuid>>,
    deploiements: HashMap<Uuid, Entree<DeploymentNode>>,
    deploiements_par_env: HashMap<Uuid, Vec<Uuid>>,
    /// The quarantined `Deployment` nodes of each environment (not served, but
    /// deleted with their environment).
    deploiements_hors_index: HashMap<Uuid, Vec<NodeId>>,
    jalons: HashMap<Uuid, Entree<MilestoneNode>>,
    jalons_par_projet: HashMap<Uuid, Vec<Uuid>>,
}

/// Insert an entity under its parent.
fn publier<T>(
    carte: &mut HashMap<Uuid, Entree<T>>,
    par: &mut HashMap<Uuid, Vec<Uuid>>,
    parent: Uuid,
    id: Uuid,
    entree: Entree<T>,
) {
    par.entry(parent).or_default().push(id);
    carte.insert(id, entree);
}

/// Remove an entity from the index. Its parent never changes after creation, so the
/// parent list is found by the caller's knowledge of it.
fn oublier<T>(
    carte: &mut HashMap<Uuid, Entree<T>>,
    par: &mut HashMap<Uuid, Vec<Uuid>>,
    parent: Uuid,
    id: Uuid,
) {
    if let Some(ids) = par.get_mut(&parent) {
        ids.retain(|x| *x != id);
    }
    carte.remove(&id);
}

/// The children of a parent, in index order (callers sort them).
fn enfants<'a, T>(
    carte: &'a HashMap<Uuid, Entree<T>>,
    par: &HashMap<Uuid, Vec<Uuid>>,
    parent: Uuid,
) -> Vec<&'a T> {
    par.get(&parent)
        .map(|ids| {
            ids.iter()
                .filter_map(|id| carte.get(id))
                .map(|e| &e.valeur)
                .collect()
        })
        .unwrap_or_default()
}

/// Read every node of one label, repairing what an interrupted creation left: a
/// draft (marker still there, or no property at all) is deleted, whatever it holds.
/// What cannot be decoded is quarantined: left untouched, named in the report.
fn charger<T>(
    db: &ObrainDB,
    etiquette: &str,
    decodeur: fn(Lecture<'_>) -> Result<T>,
    rapport: &mut RapportOuverture,
) -> Vec<(NodeId, T)> {
    charger_avec_quarantaine(db, etiquette, decodeur, rapport).0
}

/// [`charger`], and the nodes it quarantined for not decoding.
fn charger_avec_quarantaine<T>(
    db: &ObrainDB,
    etiquette: &str,
    decodeur: fn(Lecture<'_>) -> Result<T>,
    rapport: &mut RapportOuverture,
) -> (Vec<(NodeId, T)>, Vec<NodeId>) {
    let mut quarantaine = Vec::new();
    let mut lus = Vec::new();
    let store = db.store();
    for noeud in store.nodes_by_label(etiquette) {
        let Some(n) = db.get_node(noeud) else {
            rapport
                .en_quarantaine
                .push(format!("nœud {etiquette} {noeud:?} : illisible"));
            continue;
        };
        let get = |cle: &str| {
            n.get_property(cle)
                .filter(|v| !matches!(v, Value::Null))
                .cloned()
        };
        if get(BROUILLON).is_some() || (n.properties.is_empty() && store.out_degree(noeud) == 0) {
            db.delete_node(noeud);
            rapport.brouillons_supprimes += 1;
            continue;
        }
        match decodeur(&get) {
            Ok(valeur) => lus.push((noeud, valeur)),
            Err(e) => {
                rapport
                    .en_quarantaine
                    .push(format!("nœud {etiquette} {noeud:?} : {e}"));
                quarantaine.push(noeud);
            }
        }
    }
    (lus, quarantaine)
}

/// The `environment_id` a `Deployment` node stores, when it reads as a UUID
/// (also for a node that does not decode as a whole).
fn environnement_de(db: &ObrainDB, noeud: NodeId) -> Option<Uuid> {
    db.get_node(noeud)?
        .get_property("environment_id")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse().ok())
}

/// Finish the environment deletions an interruption left (the `_suppression`
/// marker still on the environment): every `Deployment` whose `environment_id`
/// names a marked environment — decodable or not — FIRST, the environment LAST,
/// as the online deletion does. A cut during this repair leaves the marker, and
/// the next open finishes again. `budget` bounds the engine deletions: `usize::MAX`
/// in production; a test passes less to cut the repair half way. Returns the
/// number of deletions finished.
fn achever_suppressions(db: &ObrainDB, budget: usize) -> usize {
    let mut marques: Vec<(NodeId, Option<Uuid>)> = Vec::new();
    for noeud in db.store().nodes_by_label(ETIQUETTE_ENVIRONNEMENT) {
        let Some(n) = db.get_node(noeud) else {
            continue;
        };
        if n.get_property(SUPPRESSION)
            .is_some_and(|v| !matches!(v, Value::Null))
        {
            let id = n
                .get_property("id")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<Uuid>().ok());
            marques.push((noeud, id));
        }
    }
    if marques.is_empty() {
        return 0;
    }
    let ids: HashSet<Uuid> = marques.iter().filter_map(|(_, id)| *id).collect();
    let mut reste = budget;
    for noeud in db.store().nodes_by_label(ETIQUETTE_DEPLOIEMENT) {
        if environnement_de(db, noeud).is_some_and(|e| ids.contains(&e)) {
            if reste == 0 {
                return 0;
            }
            db.delete_node(noeud);
            reste -= 1;
        }
    }
    let mut achevees = 0;
    for (noeud, _) in marques {
        if reste == 0 {
            break;
        }
        db.delete_node(noeud);
        reste -= 1;
        achevees += 1;
    }
    achevees
}

impl IndexReleases {
    /// Rebuild the four indexes from the base. A scan of the four labels.
    pub(super) fn reconstruire(db: &ObrainDB) -> Result<(Self, RapportOuverture)> {
        let mut idx = Self::default();
        let mut rapport = RapportOuverture::default();

        // An environment whose deletion had started (module doc): finished now,
        // its deployments first, before anything is read.
        rapport.suppressions_achevees += achever_suppressions(db, usize::MAX);

        for (noeud, r) in charger(db, ETIQUETTE_RELEASE, lire_release, &mut rapport) {
            if idx.releases.contains_key(&r.id) {
                rapport
                    .en_quarantaine
                    .push(format!("nœud Release {noeud:?} : id {} déjà pris", r.id));
                continue;
            }
            let (id, parent) = (r.id, r.project_id);
            publier(
                &mut idx.releases,
                &mut idx.releases_par_projet,
                parent,
                id,
                Entree { noeud, valeur: r },
            );
        }
        for (noeud, e) in charger(
            db,
            ETIQUETTE_ENVIRONNEMENT,
            lire_environnement,
            &mut rapport,
        ) {
            if idx.environnements.contains_key(&e.id) {
                rapport.en_quarantaine.push(format!(
                    "nœud Environment {noeud:?} : id {} déjà pris",
                    e.id
                ));
                continue;
            }
            let (id, parent) = (e.id, e.project_id);
            publier(
                &mut idx.environnements,
                &mut idx.environnements_par_projet,
                parent,
                id,
                Entree { noeud, valeur: e },
            );
        }
        let (deploiements, illisibles) =
            charger_avec_quarantaine(db, ETIQUETTE_DEPLOIEMENT, lire_deploiement, &mut rapport);
        // A quarantined deployment still belongs to its environment: its deletion
        // must take it, as Neo4j's `DETACH DELETE d, e` does.
        for noeud in illisibles {
            if let Some(env) = environnement_de(db, noeud) {
                idx.deploiements_hors_index
                    .entry(env)
                    .or_default()
                    .push(noeud);
            }
        }
        for (noeud, d) in deploiements {
            if idx.deploiements.contains_key(&d.id) {
                rapport
                    .en_quarantaine
                    .push(format!("nœud Deployment {noeud:?} : id {} déjà pris", d.id));
                idx.deploiements_hors_index
                    .entry(d.environment_id)
                    .or_default()
                    .push(noeud);
                continue;
            }
            let (id, parent) = (d.id, d.environment_id);
            publier(
                &mut idx.deploiements,
                &mut idx.deploiements_par_env,
                parent,
                id,
                Entree { noeud, valeur: d },
            );
        }
        for (noeud, j) in charger(db, ETIQUETTE_JALON, lire_jalon, &mut rapport) {
            if idx.jalons.contains_key(&j.id) {
                rapport
                    .en_quarantaine
                    .push(format!("nœud Milestone {noeud:?} : id {} déjà pris", j.id));
                continue;
            }
            let (id, parent) = (j.id, j.project_id);
            publier(
                &mut idx.jalons,
                &mut idx.jalons_par_projet,
                parent,
                id,
                Entree { noeud, valeur: j },
            );
        }
        Ok((idx, rapport))
    }
}

// ============================================================================
// Decoding and encoding (the same names and conventions as the Neo4j adapter)
// ============================================================================

fn chaine(get: Lecture<'_>, cle: &str) -> Option<String> {
    get(cle).and_then(|v| v.as_str().map(str::to_string))
}

/// A string that is absent when empty (the repo convention: `""` means absent).
fn chaine_pleine(get: Lecture<'_>, cle: &str) -> Option<String> {
    chaine(get, cle).filter(|s| !s.is_empty())
}

fn instant(get: Lecture<'_>, cle: &str) -> Option<DateTime<Utc>> {
    chaine_pleine(get, cle).and_then(|s| s.parse().ok())
}

fn uuid_requis(get: Lecture<'_>, cle: &str, de: &str) -> Result<Uuid> {
    let brut = chaine(get, cle).ok_or_else(|| anyhow!("{de} sans propriété `{cle}`"))?;
    brut.parse::<Uuid>()
        .with_context(|| format!("{de} : `{cle}` n'est pas un UUID (« {brut} »)"))
}

fn texte_requis(get: Lecture<'_>, cle: &str, de: &str) -> Result<String> {
    chaine(get, cle).ok_or_else(|| anyhow!("{de} sans propriété `{cle}`"))
}

fn instant_requis(get: Lecture<'_>, cle: &str, de: &str) -> Result<DateTime<Utc>> {
    let brut = chaine_pleine(get, cle).ok_or_else(|| anyhow!("{de} sans `{cle}`"))?;
    brut.parse::<DateTime<Utc>>()
        .with_context(|| format!("{de} : `{cle}` illisible (« {brut} »)"))
}

/// A `status` stored as snake_case (or PascalCase, from an older writer). Absent
/// is an error, as Neo4j's `node.get::<String>("status")?`; an unknown value is
/// `None`, which the caller reads as the default, as Neo4j does.
fn statut_requis<T: serde::de::DeserializeOwned>(get: Lecture<'_>, de: &str) -> Result<Option<T>> {
    let brut = texte_requis(get, "status", de)?;
    Ok(serde_json::from_str(&format!("\"{}\"", pascal_to_snake_case(&brut))).ok())
}

/// The stored spelling of a serde-encoded enum (snake_case).
fn texte_serde<T: Serialize>(valeur: &T) -> String {
    serde_json::to_value(valeur)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

/// The stored spelling of an optional timestamp: `to_rfc3339`, or `""` when absent.
fn horodatage_optionnel(d: Option<DateTime<Utc>>) -> String {
    d.map(|d| d.to_rfc3339()).unwrap_or_default()
}

/// The stored spelling of a timestamp of the deployments and environments:
/// fixed-width microseconds with `Z`, so the string order is chronological.
fn ts(d: DateTime<Utc>) -> String {
    d.to_rfc3339_opts(SecondsFormat::Micros, true)
}

fn lire_release(get: Lecture<'_>) -> Result<ReleaseNode> {
    const DE: &str = "Release";
    let id = uuid_requis(get, "id", DE)?;
    Ok(ReleaseNode {
        id,
        version: texte_requis(get, "version", DE)?,
        title: chaine_pleine(get, "title"),
        description: chaine_pleine(get, "description"),
        status: statut_requis(get, DE)?.unwrap_or(ReleaseStatus::Planned),
        target_date: instant(get, "target_date"),
        released_at: instant(get, "released_at"),
        created_at: instant_requis(get, "created_at", DE)?,
        project_id: uuid_requis(get, "project_id", DE)?,
    })
}

fn lire_environnement(get: Lecture<'_>) -> Result<EnvironmentNode> {
    const DE: &str = "Environment";
    Ok(EnvironmentNode {
        id: uuid_requis(get, "id", DE)?,
        project_id: uuid_requis(get, "project_id", DE)?,
        name: texte_requis(get, "name", DE)?,
        kind: chaine(get, "kind")
            .and_then(|s| s.parse::<EnvironmentKind>().ok())
            .unwrap_or(EnvironmentKind::Other),
        url: chaine_pleine(get, "url"),
        description: chaine_pleine(get, "description"),
        config: chaine_pleine(get, "config"),
        created_at: instant_requis(get, "created_at", DE)?,
    })
}

fn lire_deploiement(get: Lecture<'_>) -> Result<DeploymentNode> {
    const DE: &str = "Deployment";
    Ok(DeploymentNode {
        id: uuid_requis(get, "id", DE)?,
        environment_id: uuid_requis(get, "environment_id", DE)?,
        version: chaine_pleine(get, "version"),
        commit_sha: chaine_pleine(get, "commit_sha"),
        status: chaine(get, "status")
            .and_then(|s| s.parse::<DeploymentStatus>().ok())
            .unwrap_or(DeploymentStatus::Pending),
        notes: chaine_pleine(get, "notes"),
        created_by: chaine(get, "created_by").unwrap_or_default(),
        started_at: instant_requis(get, "started_at", DE)?,
        finished_at: instant(get, "finished_at"),
    })
}

fn lire_jalon(get: Lecture<'_>) -> Result<MilestoneNode> {
    const DE: &str = "Milestone";
    Ok(MilestoneNode {
        id: uuid_requis(get, "id", DE)?,
        title: texte_requis(get, "title", DE)?,
        description: chaine_pleine(get, "description"),
        status: statut_requis(get, DE)?.unwrap_or(MilestoneStatus::Open),
        target_date: instant(get, "target_date"),
        closed_at: instant(get, "closed_at"),
        created_at: instant_requis(get, "created_at", DE)?,
        project_id: uuid_requis(get, "project_id", DE)?,
    })
}

/// Decode a node once it has been written (the index is then published from this).
fn relire<T>(db: &ObrainDB, noeud: NodeId, decodeur: fn(Lecture<'_>) -> Result<T>) -> Result<T> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud {noeud:?} absent juste après son écriture"))?;
    let get = move |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    };
    decodeur(&get)
}

fn props_release(r: &ReleaseNode) -> Vec<(&'static str, Value)> {
    vec![
        ("id", Value::from(r.id.to_string())),
        ("version", Value::from(r.version.clone())),
        ("title", Value::from(r.title.clone().unwrap_or_default())),
        (
            "description",
            Value::from(r.description.clone().unwrap_or_default()),
        ),
        ("status", Value::from(texte_serde(&r.status))),
        (
            "target_date",
            Value::from(horodatage_optionnel(r.target_date)),
        ),
        (
            "released_at",
            Value::from(horodatage_optionnel(r.released_at)),
        ),
        ("created_at", Value::from(r.created_at.to_rfc3339())),
        ("project_id", Value::from(r.project_id.to_string())),
    ]
}

fn props_environnement(e: &EnvironmentNode) -> Vec<(&'static str, Value)> {
    vec![
        ("id", Value::from(e.id.to_string())),
        ("project_id", Value::from(e.project_id.to_string())),
        ("name", Value::from(e.name.clone())),
        ("kind", Value::from(e.kind.as_str())),
        ("url", Value::from(e.url.clone().unwrap_or_default())),
        (
            "description",
            Value::from(e.description.clone().unwrap_or_default()),
        ),
        ("config", Value::from(e.config.clone().unwrap_or_default())),
        ("created_at", Value::from(ts(e.created_at))),
    ]
}

fn props_deploiement(d: &DeploymentNode) -> Vec<(&'static str, Value)> {
    vec![
        ("id", Value::from(d.id.to_string())),
        ("environment_id", Value::from(d.environment_id.to_string())),
        (
            "version",
            Value::from(d.version.clone().unwrap_or_default()),
        ),
        (
            "commit_sha",
            Value::from(d.commit_sha.clone().unwrap_or_default()),
        ),
        ("status", Value::from(d.status.as_str())),
        ("notes", Value::from(d.notes.clone().unwrap_or_default())),
        ("created_by", Value::from(d.created_by.clone())),
        ("started_at", Value::from(ts(d.started_at))),
        (
            "finished_at",
            Value::from(d.finished_at.map(ts).unwrap_or_default()),
        ),
    ]
}

fn props_jalon(j: &MilestoneNode) -> Vec<(&'static str, Value)> {
    vec![
        ("id", Value::from(j.id.to_string())),
        ("title", Value::from(j.title.clone())),
        (
            "description",
            Value::from(j.description.clone().unwrap_or_default()),
        ),
        ("status", Value::from(texte_serde(&j.status))),
        (
            "target_date",
            Value::from(horodatage_optionnel(j.target_date)),
        ),
        ("closed_at", Value::from(horodatage_optionnel(j.closed_at))),
        ("created_at", Value::from(j.created_at.to_rfc3339())),
        ("project_id", Value::from(j.project_id.to_string())),
    ]
}

// ============================================================================
// Writes: the draft marker, the blob rule, the publication
// ============================================================================

/// Does this value go to a blob column (a string above the threshold)? Such a
/// value is not recovered by the WAL: see `mod.rs` (`SEUIL_COLONNE_BLOB`).
fn valeur_en_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
}

fn texte_en_blob(s: &str) -> bool {
    s.len() > SEUIL_COLONNE_BLOB
}

/// Create a node with its properties, under the draft marker. The marker is removed
/// last (the commit point). A value in a blob column is flushed fully before the
/// marker goes (`Ecriture::rendre_durable_avant_le_marqueur`).
fn creer_noeud(
    w: &mut Ecriture<'_>,
    etiquette: &str,
    proprietes: Vec<(&'static str, Value)>,
) -> Result<NodeId> {
    w.flush_complet = proprietes.iter().any(|(_, v)| valeur_en_blob(v));
    let noeud = w.db.create_node(&[etiquette]);
    #[cfg(any(test, feature = "mock-store"))]
    w.point(Interruption::ApresLeNoeud)?;
    w.db.set_node_property(noeud, BROUILLON, Value::from(true));
    #[cfg(any(test, feature = "mock-store"))]
    w.point(Interruption::ApresLeMarqueur)?;
    for (n, (cle, valeur)) in proprietes.into_iter().enumerate() {
        w.db.set_node_property(noeud, cle, valeur);
        #[cfg(any(test, feature = "mock-store"))]
        w.point(Interruption::ApresLaPropriete(n + 1))?;
        // Only the test hook reads the rank.
        #[cfg(not(any(test, feature = "mock-store")))]
        let _ = n;
    }
    w.rendre_durable_avant_le_marqueur()?;
    #[cfg(any(test, feature = "mock-store"))]
    w.point(Interruption::AvantLeRetraitDuMarqueur)?;
    w.db.remove_node_property(noeud, BROUILLON);
    #[cfg(any(test, feature = "mock-store"))]
    w.point(Interruption::ApresLeRetraitDuMarqueur)?;
    Ok(noeud)
}

/// Overwrite some properties of an existing node. `ancien_en_blob` says whether the
/// value being replaced was itself in a blob column (its clearing is not in the WAL
/// either).
fn ecrire_champs(
    w: &mut Ecriture<'_>,
    noeud: NodeId,
    ancien_en_blob: bool,
    champs: Vec<(&'static str, Value)>,
) {
    w.flush_complet = ancien_en_blob || champs.iter().any(|(_, v)| valeur_en_blob(v));
    for (cle, valeur) in champs {
        w.db.set_node_property(noeud, cle, valeur);
    }
}

fn blob_release(r: &ReleaseNode) -> bool {
    [
        Some(r.version.as_str()),
        r.title.as_deref(),
        r.description.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(texte_en_blob)
}

fn blob_environnement(e: &EnvironmentNode) -> bool {
    [
        Some(e.name.as_str()),
        e.url.as_deref(),
        e.description.as_deref(),
        e.config.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(texte_en_blob)
}

fn blob_deploiement(d: &DeploymentNode) -> bool {
    [
        d.version.as_deref(),
        d.commit_sha.as_deref(),
        d.notes.as_deref(),
        Some(d.created_by.as_str()),
    ]
    .into_iter()
    .flatten()
    .any(texte_en_blob)
}

fn blob_jalon(j: &MilestoneNode) -> bool {
    [Some(j.title.as_str()), j.description.as_deref()]
        .into_iter()
        .flatten()
        .any(texte_en_blob)
}

// ============================================================================
// Reads: the ordering, as Neo4j states it
// ============================================================================

/// Sort by a primary key (ascending or descending), then by the id (ascending or
/// descending). Neo4j's `ORDER BY a DESC, b ASC` is this, with both keys named.
fn trier<T>(
    v: &mut [T],
    primaire: impl Fn(&T, &T) -> Ordering,
    desc: bool,
    id: impl Fn(&T) -> Uuid,
    id_desc: bool,
) {
    v.sort_by(|a, b| {
        let o = primaire(a, b);
        let o = if desc { o.reverse() } else { o };
        let i = id(a).cmp(&id(b));
        let i = if id_desc { i.reverse() } else { i };
        o.then(i)
    });
}

/// `None` sorts after every value (Neo4j's null order: last ascending, first
/// descending, which the reversal of a descending sort gives).
fn cmp_nul_en_dernier<T: Ord>(a: &Option<T>, b: &Option<T>) -> Ordering {
    match (a, b) {
        (Some(x), Some(y)) => x.cmp(y),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// The stored string of an optional timestamp, for an ordering on it: Neo4j orders
/// those strings, not the instants (`""` for an absent one sorts first).
fn cle_horodatage(d: Option<DateTime<Utc>>) -> String {
    horodatage_optionnel(d)
}

/// `sort_order` is descending unless it says exactly `asc`, as the Neo4j code.
fn est_descendant(sort_order: &str) -> bool {
    sort_order != "asc"
}

/// The filter of a status list: the stored spellings it accepts (any case).
fn statuts_acceptes(statuses: &Option<Vec<String>>) -> Option<Vec<String>> {
    statuses
        .as_ref()
        .filter(|s| !s.is_empty())
        .map(|s| s.iter().flat_map(|x| status_variants(x)).collect())
}

fn statut_accepte(accepte: &Option<Vec<String>>, stocke: &str) -> bool {
    accepte
        .as_ref()
        .is_none_or(|a| a.iter().any(|x| x == stocke))
}

/// One page: `SKIP offset LIMIT limit` over the sorted matches, and their total.
fn page<T>(v: Vec<T>, offset: usize, limit: usize) -> (Vec<T>, usize) {
    let total = v.len();
    (v.into_iter().skip(offset).take(limit).collect(), total)
}

// ============================================================================
// The store
// ============================================================================

impl ObrainGraphStore {
    /// Update the properties of one entity, found in the index. A missing id is a
    /// silent no-op, as the Neo4j `MATCH … SET`. The index is republished from the
    /// decoded node, once durable.
    ///
    /// `lecture` finds the entity under the READ lock of the index; `acces`, its
    /// mutable twin, is taken only to publish (the write lock is for publishing,
    /// `mod.rs`).
    async fn modifier_entite<T: Send + 'static>(
        &self,
        id: Uuid,
        champs: Vec<(&'static str, Value)>,
        lecture: fn(&IndexReleases) -> &HashMap<Uuid, Entree<T>>,
        acces: fn(&mut IndexReleases) -> &mut HashMap<Uuid, Entree<T>>,
        blob: fn(&T) -> bool,
        decodeur: fn(Lecture<'_>) -> Result<T>,
    ) -> Result<()> {
        if champs.is_empty() {
            return Ok(());
        }
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, ancien_blob) = {
                    let etat = w.index()?;
                    let Some(e) = lecture(&etat.releases).get(&id) else {
                        return Ok(());
                    };
                    (e.noeud, blob(&e.valeur))
                };
                ecrire_champs(w, noeud, ancien_blob, champs);
                let valeur = w.apres_ecriture(relire(w.db, noeud, decodeur))?;
                w.rendre_durable()?;
                acces(&mut w.index_mut()?.releases).insert(id, Entree { noeud, valeur });
                Ok(())
            },
        )
        .await
    }
}

#[async_trait]
impl ReleaseStore for ObrainGraphStore {
    // ------------------------------------------------------------------------
    // Release
    // ------------------------------------------------------------------------

    async fn create_release(&self, release: &ReleaseNode) -> Result<()> {
        let r = release.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                {
                    let etat = w.index()?;
                    if !etat.projets.contient(r.project_id) {
                        bail!(
                            "Failed to create release: Project {} not found in Neo4j",
                            r.project_id
                        );
                    }
                    if etat.releases.releases.contains_key(&r.id) {
                        bail!("une release d'id {} existe déjà", r.id);
                    }
                }
                let noeud = creer_noeud(w, ETIQUETTE_RELEASE, props_release(&r))?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, lire_release))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                let idx = &mut etat.releases;
                publier(
                    &mut idx.releases,
                    &mut idx.releases_par_projet,
                    r.project_id,
                    r.id,
                    Entree { noeud, valeur },
                );
                Ok(())
            },
        )
        .await
    }

    async fn get_release(&self, id: Uuid) -> Result<Option<ReleaseNode>> {
        self.lire(|_, etat| Ok(etat.releases.releases.get(&id).map(|e| e.valeur.clone())))
    }

    async fn list_project_releases(&self, project_id: Uuid) -> Result<Vec<ReleaseNode>> {
        self.lire(|_, etat| {
            let idx = &etat.releases;
            let mut v: Vec<ReleaseNode> =
                enfants(&idx.releases, &idx.releases_par_projet, project_id)
                    .into_iter()
                    .cloned()
                    .collect();
            // ORDER BY r.created_at DESC, r.id DESC
            trier(
                &mut v,
                |a, b| a.created_at.cmp(&b.created_at),
                true,
                |r| r.id,
                true,
            );
            Ok(v)
        })
    }

    async fn update_release(
        &self,
        id: Uuid,
        status: Option<ReleaseStatus>,
        target_date: Option<DateTime<Utc>>,
        released_at: Option<DateTime<Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        let mut champs: Vec<(&'static str, Value)> = Vec::new();
        if let Some(s) = status {
            champs.push(("status", Value::from(texte_serde(&s))));
        }
        if let Some(d) = target_date {
            champs.push(("target_date", Value::from(d.to_rfc3339())));
        }
        if let Some(d) = released_at {
            champs.push(("released_at", Value::from(d.to_rfc3339())));
        }
        if let Some(t) = title {
            champs.push(("title", Value::from(t)));
        }
        if let Some(d) = description {
            champs.push(("description", Value::from(d)));
        }
        self.modifier_entite(
            id,
            champs,
            |i| &i.releases,
            |i| &mut i.releases,
            blob_release,
            lire_release,
        )
        .await
    }

    async fn delete_release(&self, release_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, parent) = {
                    let etat = w.index()?;
                    let Some(e) = etat.releases.releases.get(&release_id) else {
                        return Ok(());
                    };
                    (e.noeud, e.valeur.project_id)
                };
                w.supprimer_noeuds(&[noeud], &format!("suppression de la release {release_id}"))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                let idx = &mut etat.releases;
                oublier(
                    &mut idx.releases,
                    &mut idx.releases_par_projet,
                    parent,
                    release_id,
                );
                Ok(())
            },
        )
        .await
    }

    async fn list_releases_filtered(
        &self,
        project_id: Uuid,
        statuses: Option<Vec<String>>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<ReleaseNode>, usize)> {
        let accepte = statuts_acceptes(&statuses);
        let sort_by = sort_by.map(str::to_string);
        let desc = est_descendant(sort_order);
        self.lire(|_, etat| {
            let idx = &etat.releases;
            let mut v: Vec<ReleaseNode> =
                enfants(&idx.releases, &idx.releases_par_projet, project_id)
                    .into_iter()
                    .filter(|r| statut_accepte(&accepte, &texte_serde(&r.status)))
                    .cloned()
                    .collect();
            let primaire = |a: &ReleaseNode, b: &ReleaseNode| match sort_by.as_deref() {
                Some("version") => a.version.cmp(&b.version),
                Some("target_date") => {
                    cle_horodatage(a.target_date).cmp(&cle_horodatage(b.target_date))
                }
                Some("released_at") => {
                    cle_horodatage(a.released_at).cmp(&cle_horodatage(b.released_at))
                }
                Some("title") => a
                    .title
                    .as_deref()
                    .unwrap_or("")
                    .cmp(b.title.as_deref().unwrap_or("")),
                _ => a.created_at.cmp(&b.created_at),
            };
            // ORDER BY <field> <dir>, r.id ASC
            trier(&mut v, primaire, desc, |r| r.id, false);
            Ok(page(v, offset, limit))
        })
    }

    // ------------------------------------------------------------------------
    // Release / milestone links to tasks, plans and commits: not ported
    // ------------------------------------------------------------------------

    async fn add_task_to_release(&self, _release_id: Uuid, _task_id: Uuid) -> Result<()> {
        Err(NonPorte("ReleaseStore::add_task_to_release").into())
    }

    async fn add_commit_to_release(&self, _release_id: Uuid, _commit_hash: &str) -> Result<()> {
        Err(NonPorte("ReleaseStore::add_commit_to_release").into())
    }

    async fn remove_commit_from_release(
        &self,
        _release_id: Uuid,
        _commit_hash: &str,
    ) -> Result<()> {
        Err(NonPorte("ReleaseStore::remove_commit_from_release").into())
    }

    async fn get_release_details(
        &self,
        _release_id: Uuid,
    ) -> Result<Option<(ReleaseNode, Vec<TaskNode>, Vec<CommitNode>)>> {
        Err(NonPorte("ReleaseStore::get_release_details").into())
    }

    async fn get_release_tasks(&self, _release_id: Uuid) -> Result<Vec<TaskNode>> {
        Err(NonPorte("ReleaseStore::get_release_tasks").into())
    }

    // ------------------------------------------------------------------------
    // Environment
    // ------------------------------------------------------------------------

    async fn create_environment(&self, env: &EnvironmentNode) -> Result<()> {
        let e = env.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                {
                    let etat = w.index()?;
                    let existe_projet = etat.projets.contient(e.project_id);
                    let nom_pris = enfants(
                        &etat.releases.environnements,
                        &etat.releases.environnements_par_projet,
                        e.project_id,
                    )
                    .iter()
                    .any(|x| x.name == e.name);
                    if !existe_projet || nom_pris {
                        bail!(
                            "Cannot create environment '{}': project not found or name already used",
                            e.name
                        );
                    }
                    if etat.releases.environnements.contains_key(&e.id) {
                        bail!("un environnement d'id {} existe déjà", e.id);
                    }
                }
                let noeud = creer_noeud(w, ETIQUETTE_ENVIRONNEMENT, props_environnement(&e))?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, lire_environnement))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                let idx = &mut etat.releases;
                publier(
                    &mut idx.environnements,
                    &mut idx.environnements_par_projet,
                    e.project_id,
                    e.id,
                    Entree { noeud, valeur },
                );
                Ok(())
            },
        )
        .await
    }

    async fn get_environment(&self, id: Uuid) -> Result<Option<EnvironmentNode>> {
        self.lire(|_, etat| {
            Ok(etat
                .releases
                .environnements
                .get(&id)
                .map(|e| e.valeur.clone()))
        })
    }

    async fn list_project_environments(&self, project_id: Uuid) -> Result<Vec<EnvironmentNode>> {
        self.lire(|_, etat| {
            let idx = &etat.releases;
            let mut v: Vec<EnvironmentNode> = enfants(
                &idx.environnements,
                &idx.environnements_par_projet,
                project_id,
            )
            .into_iter()
            .cloned()
            .collect();
            // ORDER BY e.created_at ASC, e.name ASC, e.id ASC
            trier(
                &mut v,
                |a, b| {
                    a.created_at
                        .cmp(&b.created_at)
                        .then_with(|| a.name.cmp(&b.name))
                },
                false,
                |e| e.id,
                false,
            );
            Ok(v)
        })
    }

    async fn update_environment(
        &self,
        id: Uuid,
        name: Option<String>,
        kind: Option<EnvironmentKind>,
        url: Option<String>,
        description: Option<String>,
        config: Option<String>,
    ) -> Result<()> {
        let mut champs: Vec<(&'static str, Value)> = Vec::new();
        if let Some(n) = name {
            champs.push(("name", Value::from(n)));
        }
        if let Some(k) = kind {
            champs.push(("kind", Value::from(k.as_str())));
        }
        if let Some(u) = url {
            champs.push(("url", Value::from(u)));
        }
        if let Some(d) = description {
            champs.push(("description", Value::from(d)));
        }
        if let Some(c) = config {
            champs.push(("config", Value::from(c)));
        }
        self.modifier_entite(
            id,
            champs,
            |i| &i.environnements,
            |i| &mut i.environnements,
            blob_environnement,
            lire_environnement,
        )
        .await
    }

    async fn delete_environment(&self, id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // The environment and its deployments, in one durable step, behind
                // a deletion marker (module doc).
                let (noeud, parent, deploiements, hors_index) = {
                    let etat = w.index()?;
                    let Some(e) = etat.releases.environnements.get(&id) else {
                        return Ok(());
                    };
                    let ids = etat
                        .releases
                        .deploiements_par_env
                        .get(&id)
                        .cloned()
                        .unwrap_or_default();
                    let deps: Vec<(NodeId, Uuid)> = ids
                        .into_iter()
                        .filter_map(|i| etat.releases.deploiements.get(&i).map(|d| (d.noeud, i)))
                        .collect();
                    let hors_index = etat
                        .releases
                        .deploiements_hors_index
                        .get(&id)
                        .cloned()
                        .unwrap_or_default();
                    (e.noeud, e.valeur.project_id, deps, hors_index)
                };
                // The deployments, then the environment: all checked live first
                // (nothing written otherwise), then the marker, the commit point of
                // the deletion: from there a restart finishes it (module doc).
                let mut noeuds: Vec<NodeId> = deploiements.iter().map(|(n, _)| *n).collect();
                noeuds.extend(hors_index);
                noeuds.push(noeud);
                let quoi = format!("suppression de l'environnement {id}");
                w.verifier_vivants(&noeuds, &quoi)?;
                w.db.set_node_property(noeud, SUPPRESSION, Value::from(true));
                w.etape()?;
                // Past the marker, a failure leaves a deletion that the next open
                // finishes: what the base holds is no longer what the index serves.
                let r = w.supprimer_noeuds(&noeuds, &quoi);
                w.apres_ecriture(r)?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                let idx = &mut etat.releases;
                idx.deploiements_hors_index.remove(&id);
                for (_, dep_id) in deploiements {
                    oublier(
                        &mut idx.deploiements,
                        &mut idx.deploiements_par_env,
                        id,
                        dep_id,
                    );
                }
                oublier(
                    &mut idx.environnements,
                    &mut idx.environnements_par_projet,
                    parent,
                    id,
                );
                Ok(())
            },
        )
        .await
    }

    // ------------------------------------------------------------------------
    // Deployment
    // ------------------------------------------------------------------------

    async fn create_deployment(&self, deployment: &DeploymentNode) -> Result<()> {
        let d = deployment.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                {
                    let etat = w.index()?;
                    if !etat.releases.environnements.contains_key(&d.environment_id) {
                        bail!(
                            "Cannot create deployment: environment {} not found",
                            d.environment_id
                        );
                    }
                    if etat.releases.deploiements.contains_key(&d.id) {
                        bail!("un déploiement d'id {} existe déjà", d.id);
                    }
                }
                let noeud = creer_noeud(w, ETIQUETTE_DEPLOIEMENT, props_deploiement(&d))?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, lire_deploiement))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                let idx = &mut etat.releases;
                publier(
                    &mut idx.deploiements,
                    &mut idx.deploiements_par_env,
                    d.environment_id,
                    d.id,
                    Entree { noeud, valeur },
                );
                Ok(())
            },
        )
        .await
    }

    async fn get_deployment(&self, id: Uuid) -> Result<Option<DeploymentNode>> {
        self.lire(|_, etat| {
            Ok(etat
                .releases
                .deploiements
                .get(&id)
                .map(|d| d.valeur.clone()))
        })
    }

    async fn list_environment_deployments(
        &self,
        environment_id: Uuid,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<DeploymentNode>, usize)> {
        self.lire(|_, etat| {
            let idx = &etat.releases;
            let mut v: Vec<DeploymentNode> =
                enfants(&idx.deploiements, &idx.deploiements_par_env, environment_id)
                    .into_iter()
                    .cloned()
                    .collect();
            // ORDER BY d.started_at DESC, d.id DESC
            trier(
                &mut v,
                |a, b| a.started_at.cmp(&b.started_at),
                true,
                |d| d.id,
                true,
            );
            Ok(page(v, offset, limit))
        })
    }

    async fn update_deployment(
        &self,
        id: Uuid,
        status: Option<DeploymentStatus>,
        finished_at: Option<DateTime<Utc>>,
        notes: Option<String>,
    ) -> Result<()> {
        let mut champs: Vec<(&'static str, Value)> = Vec::new();
        if let Some(s) = status {
            champs.push(("status", Value::from(s.as_str())));
        }
        if let Some(f) = finished_at {
            champs.push(("finished_at", Value::from(ts(f))));
        }
        if let Some(n) = notes {
            champs.push(("notes", Value::from(n)));
        }
        self.modifier_entite(
            id,
            champs,
            |i| &i.deploiements,
            |i| &mut i.deploiements,
            blob_deploiement,
            lire_deploiement,
        )
        .await
    }

    async fn get_deployment_matrix(&self, project_id: Uuid) -> Result<Vec<DeploymentMatrixEntry>> {
        self.lire(|_, etat| {
            let idx = &etat.releases;
            let mut envs: Vec<EnvironmentNode> = enfants(
                &idx.environnements,
                &idx.environnements_par_projet,
                project_id,
            )
            .into_iter()
            .cloned()
            .collect();
            trier(
                &mut envs,
                |a, b| {
                    a.created_at
                        .cmp(&b.created_at)
                        .then_with(|| a.name.cmp(&b.name))
                },
                false,
                |e| e.id,
                false,
            );
            let mut matrice = Vec::with_capacity(envs.len());
            for environment in envs {
                let mut deps: Vec<DeploymentNode> =
                    enfants(&idx.deploiements, &idx.deploiements_par_env, environment.id)
                        .into_iter()
                        .cloned()
                        .collect();
                trier(
                    &mut deps,
                    |a, b| a.started_at.cmp(&b.started_at),
                    true,
                    |d| d.id,
                    true,
                );
                let recents: Vec<DeploymentNode> =
                    deps.into_iter().take(DEPLOYMENT_MATRIX_RECENT).collect();
                matrice.push(DeploymentMatrixEntry {
                    environment,
                    recent_statuses: recents.iter().map(|d| d.status).collect(),
                    latest_deployment: recents.into_iter().next(),
                });
            }
            Ok(matrice)
        })
    }

    // ------------------------------------------------------------------------
    // Milestone
    // ------------------------------------------------------------------------

    async fn create_milestone(&self, milestone: &MilestoneNode) -> Result<()> {
        let j = milestone.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                {
                    let etat = w.index()?;
                    if !etat.projets.contient(j.project_id) {
                        bail!(
                            "Failed to create milestone: Project {} not found in Neo4j",
                            j.project_id
                        );
                    }
                    if etat.releases.jalons.contains_key(&j.id) {
                        bail!("un jalon d'id {} existe déjà", j.id);
                    }
                }
                let noeud = creer_noeud(w, ETIQUETTE_JALON, props_jalon(&j))?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, lire_jalon))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                let idx = &mut etat.releases;
                publier(
                    &mut idx.jalons,
                    &mut idx.jalons_par_projet,
                    j.project_id,
                    j.id,
                    Entree { noeud, valeur },
                );
                Ok(())
            },
        )
        .await
    }

    async fn get_milestone(&self, id: Uuid) -> Result<Option<MilestoneNode>> {
        self.lire(|_, etat| Ok(etat.releases.jalons.get(&id).map(|j| j.valeur.clone())))
    }

    async fn list_project_milestones(&self, project_id: Uuid) -> Result<Vec<MilestoneNode>> {
        self.lire(|_, etat| {
            let idx = &etat.releases;
            let mut v: Vec<MilestoneNode> =
                enfants(&idx.jalons, &idx.jalons_par_projet, project_id)
                    .into_iter()
                    .cloned()
                    .collect();
            // ORDER BY m.target_date ASC, m.created_at ASC, m.id ASC
            trier(
                &mut v,
                |a, b| {
                    cle_horodatage(a.target_date)
                        .cmp(&cle_horodatage(b.target_date))
                        .then_with(|| a.created_at.cmp(&b.created_at))
                },
                false,
                |j| j.id,
                false,
            );
            Ok(v)
        })
    }

    async fn update_milestone(
        &self,
        id: Uuid,
        status: Option<MilestoneStatus>,
        target_date: Option<DateTime<Utc>>,
        closed_at: Option<DateTime<Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        let mut champs: Vec<(&'static str, Value)> = Vec::new();
        if let Some(s) = status {
            champs.push(("status", Value::from(texte_serde(&s))));
        }
        if let Some(d) = target_date {
            champs.push(("target_date", Value::from(d.to_rfc3339())));
        }
        if let Some(d) = closed_at {
            champs.push(("closed_at", Value::from(d.to_rfc3339())));
        }
        if let Some(t) = title {
            champs.push(("title", Value::from(t)));
        }
        if let Some(d) = description {
            champs.push(("description", Value::from(d)));
        }
        self.modifier_entite(
            id,
            champs,
            |i| &i.jalons,
            |i| &mut i.jalons,
            blob_jalon,
            lire_jalon,
        )
        .await
    }

    async fn delete_milestone(&self, milestone_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, parent) = {
                    let etat = w.index()?;
                    let Some(j) = etat.releases.jalons.get(&milestone_id) else {
                        return Ok(());
                    };
                    (j.noeud, j.valeur.project_id)
                };
                w.supprimer_noeuds(&[noeud], &format!("suppression du jalon {milestone_id}"))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                let idx = &mut etat.releases;
                oublier(
                    &mut idx.jalons,
                    &mut idx.jalons_par_projet,
                    parent,
                    milestone_id,
                );
                Ok(())
            },
        )
        .await
    }

    async fn list_milestones_filtered(
        &self,
        project_id: Uuid,
        statuses: Option<Vec<String>>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<MilestoneNode>, usize)> {
        let accepte = statuts_acceptes(&statuses);
        let sort_by = sort_by.map(str::to_string);
        let desc = est_descendant(sort_order);
        self.lire(|_, etat| {
            let idx = &etat.releases;
            let mut v: Vec<MilestoneNode> =
                enfants(&idx.jalons, &idx.jalons_par_projet, project_id)
                    .into_iter()
                    .filter(|j| statut_accepte(&accepte, &texte_serde(&j.status)))
                    .cloned()
                    .collect();
            let primaire = |a: &MilestoneNode, b: &MilestoneNode| match sort_by.as_deref() {
                Some("title") => a.title.cmp(&b.title),
                Some("created_at") => a.created_at.cmp(&b.created_at),
                _ => cle_horodatage(a.target_date).cmp(&cle_horodatage(b.target_date)),
            };
            // ORDER BY <field> <dir>, m.id ASC
            trier(&mut v, primaire, desc, |j| j.id, false);
            Ok(page(v, offset, limit))
        })
    }

    // ------------------------------------------------------------------------
    // Milestone links and progress: not ported (they read tasks and plans)
    // ------------------------------------------------------------------------

    async fn add_task_to_milestone(&self, _milestone_id: Uuid, _task_id: Uuid) -> Result<()> {
        Err(NonPorte("ReleaseStore::add_task_to_milestone").into())
    }

    async fn link_plan_to_milestone(&self, _plan_id: Uuid, _milestone_id: Uuid) -> Result<()> {
        Err(NonPorte("ReleaseStore::link_plan_to_milestone").into())
    }

    async fn unlink_plan_from_milestone(&self, _plan_id: Uuid, _milestone_id: Uuid) -> Result<()> {
        Err(NonPorte("ReleaseStore::unlink_plan_from_milestone").into())
    }

    async fn get_milestone_details(
        &self,
        _milestone_id: Uuid,
    ) -> Result<Option<(MilestoneNode, Vec<TaskNode>)>> {
        Err(NonPorte("ReleaseStore::get_milestone_details").into())
    }

    async fn get_milestone_progress(&self, _milestone_id: Uuid) -> Result<(u32, u32, u32, u32)> {
        Err(NonPorte("ReleaseStore::get_milestone_progress").into())
    }

    async fn get_milestone_tasks_with_plans(
        &self,
        _milestone_id: Uuid,
    ) -> Result<Vec<TaskWithPlan>> {
        Err(NonPorte("ReleaseStore::get_milestone_tasks_with_plans").into())
    }

    async fn get_milestone_steps_batch(
        &self,
        _milestone_id: Uuid,
    ) -> Result<HashMap<Uuid, Vec<StepNode>>> {
        Err(NonPorte("ReleaseStore::get_milestone_steps_batch").into())
    }

    async fn get_milestone_tasks(&self, _milestone_id: Uuid) -> Result<Vec<TaskNode>> {
        Err(NonPorte("ReleaseStore::get_milestone_tasks").into())
    }

    // ------------------------------------------------------------------------
    // Project-wide reads over tasks and plans: not ported
    // ------------------------------------------------------------------------

    async fn get_project_progress(&self, _project_id: Uuid) -> Result<(u32, u32, u32, u32)> {
        Err(NonPorte("ReleaseStore::get_project_progress").into())
    }

    async fn get_project_task_dependencies(&self, _project_id: Uuid) -> Result<Vec<(Uuid, Uuid)>> {
        Err(NonPorte("ReleaseStore::get_project_task_dependencies").into())
    }

    async fn get_project_tasks(&self, _project_id: Uuid) -> Result<Vec<TaskNode>> {
        Err(NonPorte("ReleaseStore::get_project_tasks").into())
    }

    async fn list_plans_filtered(
        &self,
        _project_id: Option<Uuid>,
        _workspace_slug: Option<&str>,
        _statuses: Option<Vec<String>>,
        _priority_min: Option<i32>,
        _priority_max: Option<i32>,
        _search: Option<&str>,
        _limit: usize,
        _offset: usize,
        _sort_by: Option<&str>,
        _sort_order: &str,
    ) -> Result<(Vec<PlanNode>, usize)> {
        Err(NonPorte("ReleaseStore::list_plans_filtered").into())
    }

    async fn list_all_tasks_filtered(
        &self,
        _plan_id: Option<Uuid>,
        _project_id: Option<Uuid>,
        _workspace_slug: Option<&str>,
        _statuses: Option<Vec<String>>,
        _priority_min: Option<i32>,
        _priority_max: Option<i32>,
        _tags: Option<Vec<String>>,
        _assigned_to: Option<&str>,
        _limit: usize,
        _offset: usize,
        _sort_by: Option<&str>,
        _sort_order: &str,
    ) -> Result<(Vec<TaskWithPlan>, usize)> {
        Err(NonPorte("ReleaseStore::list_all_tasks_filtered").into())
    }

    // ------------------------------------------------------------------------
    // Projects
    // ------------------------------------------------------------------------

    async fn list_projects_filtered(
        &self,
        search: Option<&str>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<ProjectNode>, usize)> {
        // `add_search_filter("p", …)` of the Neo4j adapter: a case-insensitive
        // CONTAINS on `title` OR `description`. Projects have no `title`, so only
        // the description can match (see the module doc).
        let recherche = search
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.to_lowercase());
        let sort_by = sort_by.map(str::to_string);
        let desc = est_descendant(sort_order);
        self.lire(|_, etat| {
            let mut v: Vec<ProjectNode> = etat
                .projets
                .tous()
                .into_iter()
                .filter(|p| match &recherche {
                    None => true,
                    Some(r) => p
                        .description
                        .as_deref()
                        .is_some_and(|d| d.to_lowercase().contains(r.as_str())),
                })
                .collect();
            let primaire = |a: &ProjectNode, b: &ProjectNode| match sort_by.as_deref() {
                Some("created_at") => a.created_at.cmp(&b.created_at),
                Some("last_synced") => cmp_nul_en_dernier(&a.last_synced, &b.last_synced),
                _ => a.name.cmp(&b.name),
            };
            // ORDER BY <field> <dir>, p.id ASC
            trier(&mut v, primaire, desc, |p| p.id, false);
            Ok(page(v, offset, limit))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::ProjectStore;
    use crate::obrain::{Creation, ValeurBrute};

    fn projet() -> ProjectNode {
        let id = Uuid::new_v4();
        ProjectNode {
            id,
            name: format!("projet {id}"),
            slug: format!("projet-{id}"),
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
            profile: Default::default(),
        }
    }

    /// The repair at open is itself cut after ONE engine deletion. It deletes the
    /// deployments first, so the cut leaves the marked environment, which the next
    /// open finishes: no deployment of a deleted environment is ever served.
    #[tokio::test]
    async fn a_cut_during_the_repair_leaves_the_marker_for_the_next_open() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = projet();
        let e = EnvironmentNode {
            id: Uuid::new_v4(),
            project_id: p.id,
            name: "préprod".into(),
            kind: EnvironmentKind::Staging,
            url: None,
            description: None,
            config: None,
            created_at: Utc::now(),
        };
        let ds: Vec<DeploymentNode> = (0..2)
            .map(|_| DeploymentNode {
                id: Uuid::new_v4(),
                environment_id: e.id,
                version: None,
                commit_sha: None,
                status: DeploymentStatus::Succeeded,
                notes: None,
                created_by: "ci".into(),
                started_at: Utc::now(),
                finished_at: None,
            })
            .collect();
        {
            let s = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            s.create_project(&p).await.unwrap();
            s.create_environment(&e).await.unwrap();
            for d in &ds {
                s.create_deployment(d).await.unwrap();
            }
            // The state a cut after the marker of `delete_environment` leaves.
            assert!(s.ecrire_propriete_brute_pour_test(
                "Environment",
                e.id,
                SUPPRESSION,
                Some(ValeurBrute::Booleen(true))
            ));
            s.close().await.unwrap();
        }
        {
            let db = ObrainDB::open(&chemin).unwrap();
            assert_eq!(achever_suppressions(&db, 1), 0, "cut before the end");
            db.close().unwrap();
        }
        let s = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        for d in &ds {
            assert!(
                s.get_deployment(d.id).await.unwrap().is_none(),
                "a deployment of a deleted environment is served"
            );
        }
        assert!(s.get_environment(e.id).await.unwrap().is_none());
        assert_eq!(s.rapport_d_ouverture().suppressions_achevees, 1);
        assert_eq!(s.noeuds_de_label_pour_test(ETIQUETTE_DEPLOIEMENT), 0);
    }
}
