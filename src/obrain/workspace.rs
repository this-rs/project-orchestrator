//! `WorkspaceStore` on obrain — the workspace domain (V2 of `CONSOLIDATION.md`).
//!
//! The domain is four kinds of node, labelled and propertied EXACTLY as the
//! Neo4j adapter (`src/neo4j/workspace.rs`), which is the specification:
//! `Workspace`, `WorkspaceMilestone`, `Resource`, `Component`. The relations
//! are edges of the same names (`BELONGS_TO_WORKSPACE` from a project,
//! `HAS_WORKSPACE_MILESTONE` / `HAS_RESOURCE` / `HAS_COMPONENT` from a
//! workspace, `IMPLEMENTS_RESOURCE` / `USES_RESOURCE` from a project,
//! `DEPENDS_ON_COMPONENT` between components, `MAPS_TO_PROJECT` from a
//! component).
//!
//! Conventions copied from the Neo4j adapter (the conformance suite compares):
//! - an absent optional string is stored as `""` and reads back as `Some("")`
//!   (Neo4j's `node.get(..).ok()`); an absent `tags` list reads as empty;
//! - `resource_type` / `component_type` are stored in their `Debug` spelling
//!   and decoded with Neo4j's lowercase mapping; a milestone `status` is stored
//!   snake_case; timestamps are RFC 3339 strings;
//! - a create that names a parent which does not exist writes NOTHING for a
//!   milestone or a component (Neo4j's `MATCH` finds no parent, so its `CREATE`
//!   never runs). A resource is written, and simply not linked to a workspace
//!   that does not exist (Neo4j's `MERGE` after a `MATCH` that finds nothing);
//! - the link and update of an unknown id are silent no-ops, except the
//!   partial-update answers (`update_*` returning `bool`).
//!
//! Deliberate differences, on the strict side and listed in the PR:
//! - `create_*` refuses a duplicate `id`; a workspace also refuses a duplicate
//!   `slug` (Neo4j's schema has no such constraint, so it accepts both);
//! - a component name is unique inside its workspace: `create_component` and
//!   `update_component` refuse a second one (Neo4j accepts it, and
//!   `upsert_derived_component`, whose identity is (workspace, name), could not
//!   then tell the two apart).
//!
//! # Order
//!
//! Each list is sorted by a TOTAL key: `workspaces` by `(name, id)`, `resources`
//! and `components` by `(name, id)`, `milestones` by `(target_date, title, id)`
//! with `target_date` read as the stored string (`""` when absent, which sorts
//! first, as in Neo4j). Projects by `(name, id)`, the project's order. The
//! Neo4j queries of `get_resource_implementers`, `get_resource_consumers` and
//! `get_workspace_topology` state NO order: here they are sorted by the keys
//! above, which is a stricter contract.
//!
//! # Reads are served from memory, and a write is visible once durable
//!
//! Same discipline as `project.rs`: every write is published to the index AFTER
//! its durability step, and the index is rebuilt at open from the base. The
//! index holds the decoded domain values and the relations as sets of ids, so a
//! read costs O(size of its answer), never O(base).
//!
//! The rebuild at open reads the four labels (a scan of those nodes, not of the
//! base) and the edges of each node it indexes. The edges are the truth of the
//! relations: a child whose edge to its parent is missing is stored, but it is
//! listed under no parent — the same answer Neo4j gives.
//!
//! # Atomicity, built by ordering
//!
//! As for `project.rs`: a creation sets the draft marker `_brouillon` first and
//! removes it LAST, after its edges; opening deletes every draft of these four
//! labels. Multi-property updates
//! are not atomic (obrain has no multi-property write): an interruption can leave
//! some of the properties of one update and not the others; each state is a
//! valid value of the node.
//!
//! `delete_workspace` deletes its children first and the workspace LAST, so an
//! interruption leaves a workspace that still exists and can be deleted again.
//!
//! # Refused by name (not ported)
//!
//! - `list_all_workspace_milestones_filtered`, `count_all_workspace_milestones`:
//!   a cross-workspace listing ordered over every milestone would need a global
//!   ordered index that does not exist yet (no obrain issue: a PO index).
//! - the nine milestone↔task / milestone↔plan methods and the milestone
//!   progress, tasks and steps: they read or write `Task`, `Plan`, `Step`, which
//!   the plan domain (not ported) owns.
//! - `compute_coupling_matrix`: it aggregates over the file, skill, note and
//!   project domains, none of them ported.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{EdgeId, NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::project::{self, IndexProjets};
use super::{Ecriture, NonPorte, ObrainGraphStore, RapportOuverture};
use crate::neo4j::client::status_variants;
use crate::neo4j::models::{
    ComponentDependency, ComponentNode, ComponentType, ComponentUpdate, CouplingMatrix,
    DerivedComponentWrite, MilestoneStatus, ProjectNode, ResourceNode, ResourceType, StepNode,
    TaskWithPlan, WorkspaceMilestoneNode, WorkspaceNode,
};
use crate::neo4j::traits::WorkspaceStore;

const ESPACE: &str = "Workspace";
const JALON: &str = "WorkspaceMilestone";
const RESSOURCE: &str = "Resource";
const COMPOSANT: &str = "Component";
const BROUILLON: &str = "_brouillon";

const APPARTIENT: &str = "BELONGS_TO_WORKSPACE";
const A_JALON: &str = "HAS_WORKSPACE_MILESTONE";
const A_RESSOURCE: &str = "HAS_RESOURCE";
const A_COMPOSANT: &str = "HAS_COMPONENT";
const IMPLEMENTE: &str = "IMPLEMENTS_RESOURCE";
const UTILISE: &str = "USES_RESOURCE";
const DEPEND: &str = "DEPENDS_ON_COMPONENT";
const MAPPE: &str = "MAPS_TO_PROJECT";

// ============================================================================
// The index
// ============================================================================

/// One node of the domain: where it lives, and what the base says about it.
struct Fiche<T> {
    noeud: NodeId,
    valeur: T,
}

/// The workspace domain's index. Relations are sets of ids; the engine is read
/// only for what the index does not hold (the edges' properties, at open).
#[derive(Default)]
pub(super) struct IndexEspaces {
    espaces: HashMap<Uuid, Fiche<WorkspaceNode>>,
    par_slug: HashMap<String, Uuid>,
    jalons: HashMap<Uuid, Fiche<WorkspaceMilestoneNode>>,
    ressources: HashMap<Uuid, Fiche<ResourceNode>>,
    composants: HashMap<Uuid, Fiche<ComponentNode>>,
    par_nom_composant: HashMap<(Uuid, String), Uuid>,
    /// workspace → its milestones / resources / components.
    jalons_de: HashMap<Uuid, BTreeSet<Uuid>>,
    ressources_de: HashMap<Uuid, BTreeSet<Uuid>>,
    composants_de: HashMap<Uuid, BTreeSet<Uuid>>,
    /// workspace → projects (`BELONGS_TO_WORKSPACE`), and the converse.
    projets_de: HashMap<Uuid, BTreeSet<Uuid>>,
    espaces_de: HashMap<Uuid, BTreeSet<Uuid>>,
    /// resource → projects implementing / using it.
    implementeurs: HashMap<Uuid, BTreeSet<Uuid>>,
    consommateurs: HashMap<Uuid, BTreeSet<Uuid>>,
    /// component → projects it maps to.
    projets_du_composant: HashMap<Uuid, BTreeSet<Uuid>>,
    /// component → (dependency → (protocol, required)), and the converse.
    dependances: HashMap<Uuid, BTreeMap<Uuid, (String, bool)>>,
    dependants: HashMap<Uuid, BTreeSet<Uuid>>,
}

impl IndexEspaces {
    /// Rebuild the index from the base at open: the four labels' nodes (drafts
    /// deleted, unreadable ones quarantined), then the relations, read from the
    /// edges of those nodes and of the projects.
    pub(super) fn reconstruire(
        db: &ObrainDB,
        projets: &IndexProjets,
        rapport: &mut RapportOuverture,
    ) -> Result<Self> {
        let mut idx = Self::default();
        // Engine node → domain id, for every node the index accepts (projects too:
        // the edges of a project are read through this map).
        let mut uuid_de: HashMap<NodeId, Uuid> = projets.entrees().map(|(id, n)| (n, id)).collect();

        for (noeud, id, valeur) in charger(db, ESPACE, rapport, decoder_espace)? {
            if idx.espaces.contains_key(&id) || idx.par_slug.contains_key(&valeur.slug) {
                quarantaine(rapport, ESPACE, noeud, "id ou slug déjà pris");
                continue;
            }
            idx.par_slug.insert(valeur.slug.clone(), id);
            uuid_de.insert(noeud, id);
            idx.espaces.insert(id, Fiche { noeud, valeur });
        }
        for (noeud, id, valeur) in charger(db, JALON, rapport, decoder_jalon)? {
            if idx.jalons.contains_key(&id) {
                quarantaine(rapport, JALON, noeud, "id déjà pris");
                continue;
            }
            uuid_de.insert(noeud, id);
            idx.jalons.insert(id, Fiche { noeud, valeur });
        }
        for (noeud, id, valeur) in charger(db, RESSOURCE, rapport, decoder_ressource)? {
            if idx.ressources.contains_key(&id) {
                quarantaine(rapport, RESSOURCE, noeud, "id déjà pris");
                continue;
            }
            uuid_de.insert(noeud, id);
            idx.ressources.insert(id, Fiche { noeud, valeur });
        }
        for (noeud, id, valeur) in charger(db, COMPOSANT, rapport, decoder_composant)? {
            let cle = (valeur.workspace_id, valeur.name.clone());
            if idx.composants.contains_key(&id) || idx.par_nom_composant.contains_key(&cle) {
                quarantaine(
                    rapport,
                    COMPOSANT,
                    noeud,
                    "id ou nom déjà pris dans le workspace",
                );
                continue;
            }
            idx.par_nom_composant.insert(cle, id);
            uuid_de.insert(noeud, id);
            idx.composants.insert(id, Fiche { noeud, valeur });
        }

        // Workspace → its children (`HAS_*`). A child is listed only where its
        // edge says so.
        let espaces: Vec<(Uuid, NodeId)> = idx.espaces.iter().map(|(i, f)| (*i, f.noeud)).collect();
        for (ws, wn) in espaces {
            for (c, _) in cibles(db, wn, A_JALON, &uuid_de) {
                if idx.jalons.contains_key(&c) {
                    idx.jalons_de.entry(ws).or_default().insert(c);
                }
            }
            for (c, _) in cibles(db, wn, A_RESSOURCE, &uuid_de) {
                if idx.ressources.contains_key(&c) {
                    idx.ressources_de.entry(ws).or_default().insert(c);
                }
            }
            for (c, _) in cibles(db, wn, A_COMPOSANT, &uuid_de) {
                if idx.composants.contains_key(&c) {
                    idx.composants_de.entry(ws).or_default().insert(c);
                }
            }
        }

        // Project → workspace, and project → resource (implements / uses).
        for (pid, pn) in projets.entrees() {
            for (ws, _) in cibles(db, pn, APPARTIENT, &uuid_de) {
                if idx.espaces.contains_key(&ws) {
                    idx.projets_de.entry(ws).or_default().insert(pid);
                    idx.espaces_de.entry(pid).or_default().insert(ws);
                }
            }
            for (r, _) in cibles(db, pn, IMPLEMENTE, &uuid_de) {
                if idx.ressources.contains_key(&r) {
                    idx.implementeurs.entry(r).or_default().insert(pid);
                }
            }
            for (r, _) in cibles(db, pn, UTILISE, &uuid_de) {
                if idx.ressources.contains_key(&r) {
                    idx.consommateurs.entry(r).or_default().insert(pid);
                }
            }
        }

        // Component → project (`MAPS_TO_PROJECT`), and component → component.
        let composants: Vec<(Uuid, NodeId)> =
            idx.composants.iter().map(|(i, f)| (*i, f.noeud)).collect();
        for (cid, cn) in composants {
            for (p, _) in cibles(db, cn, MAPPE, &uuid_de) {
                if projets.projet(p).is_some() {
                    idx.projets_du_composant.entry(cid).or_default().insert(p);
                }
            }
            for (to, eid) in cibles(db, cn, DEPEND, &uuid_de) {
                if idx.composants.contains_key(&to) {
                    let (protocole, requis) = lire_dependance(db, eid);
                    idx.dependances
                        .entry(cid)
                        .or_default()
                        .insert(to, (protocole, requis));
                    idx.dependants.entry(to).or_default().insert(cid);
                }
            }
        }
        Ok(idx)
    }

    /// Forget a component everywhere the index holds it (its own node, its name,
    /// its workspace, its projects, its dependencies in both directions).
    fn oublier_composant(&mut self, cid: Uuid) {
        if let Some(f) = self.composants.remove(&cid) {
            self.par_nom_composant
                .remove(&(f.valeur.workspace_id, f.valeur.name.clone()));
            if let Some(s) = self.composants_de.get_mut(&f.valeur.workspace_id) {
                s.remove(&cid);
            }
        }
        self.projets_du_composant.remove(&cid);
        if let Some(sortants) = self.dependances.remove(&cid) {
            for to in sortants.keys() {
                if let Some(s) = self.dependants.get_mut(to) {
                    s.remove(&cid);
                }
            }
        }
        if let Some(entrants) = self.dependants.remove(&cid) {
            for from in entrants {
                if let Some(m) = self.dependances.get_mut(&from) {
                    m.remove(&cid);
                }
            }
        }
    }

    /// Forget a milestone (its node and its workspace's list).
    fn oublier_jalon(&mut self, jid: Uuid) {
        if let Some(f) = self.jalons.remove(&jid) {
            if let Some(s) = self.jalons_de.get_mut(&f.valeur.workspace_id) {
                s.remove(&jid);
            }
        }
    }

    /// Forget a resource (its node, its workspace's list, its implementers and
    /// consumers).
    fn oublier_ressource(&mut self, rid: Uuid) {
        if let Some(f) = self.ressources.remove(&rid) {
            if let Some(ws) = f.valeur.workspace_id {
                if let Some(s) = self.ressources_de.get_mut(&ws) {
                    s.remove(&rid);
                }
            }
        }
        self.implementeurs.remove(&rid);
        self.consommateurs.remove(&rid);
    }
}

/// Every node of `etiquette` the index may hold, as `(node, id, value)`. A draft
/// (the marker still there, or an empty node with no edge: a creation killed
/// before its commit point) is deleted, whatever it holds, and counted in the
/// report; an unreadable node is quarantined there.
fn charger<T>(
    db: &ObrainDB,
    etiquette: &str,
    rapport: &mut RapportOuverture,
    decoder: Decodeur<T>,
) -> Result<Vec<(NodeId, Uuid, T)>> {
    let store = db.store();
    let mut out = Vec::new();
    for noeud in store.nodes_by_label(etiquette) {
        let Some(n) = db.get_node(noeud) else {
            quarantaine(rapport, etiquette, noeud, "illisible");
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
        match decoder(&get) {
            Ok((id, valeur)) => out.push((noeud, id, valeur)),
            Err(e) => quarantaine(rapport, etiquette, noeud, &e.to_string()),
        }
    }
    Ok(out)
}

fn quarantaine(rapport: &mut RapportOuverture, etiquette: &str, noeud: NodeId, raison: &str) {
    rapport
        .en_quarantaine
        .push(format!("nœud {etiquette} {noeud:?} : {raison}"));
}

/// The domain ids of the nodes `noeud` points to through edges of `type_`, with
/// the edge. A target the map does not know (not indexed) is left out.
fn cibles(
    db: &ObrainDB,
    noeud: NodeId,
    type_: &str,
    uuid_de: &HashMap<NodeId, Uuid>,
) -> Vec<(Uuid, EdgeId)> {
    voisins(db, noeud, type_)
        .into_iter()
        .filter_map(|(cible, eid)| uuid_de.get(&cible).map(|id| (*id, eid)))
        .collect()
}

/// Outgoing neighbours of `noeud` through edges of `type_`, read by the engine.
fn voisins(db: &ObrainDB, noeud: NodeId, type_: &str) -> Vec<(NodeId, EdgeId)> {
    db.session().get_neighbors_outgoing_by_type(noeud, type_)
}

fn lire_dependance(db: &ObrainDB, eid: EdgeId) -> (String, bool) {
    let session = db.session();
    let protocole = session
        .get_edge_property(eid, "protocol")
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default();
    let requis = session
        .get_edge_property(eid, "required")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    (protocole, requis)
}

// ============================================================================
// Decoding (the counterpart of the Neo4j `node_to_*` helpers)
// ============================================================================

type Get<'a> = &'a dyn Fn(&str) -> Option<Value>;

/// A decoder of one node: its domain id and value, from a property lookup.
type Decodeur<T> = fn(Get<'_>) -> Result<(Uuid, T)>;

fn chaine(get: Get<'_>, cle: &str) -> Option<String> {
    get(cle).and_then(|v| v.as_str().map(str::to_string))
}

fn liste(get: Get<'_>, cle: &str) -> Vec<String> {
    get(cle)
        .and_then(|v| {
            v.as_list().map(|l| {
                l.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
        })
        .unwrap_or_default()
}

fn id_requis(get: Get<'_>, cle: &str, quoi: &str) -> Result<Uuid> {
    chaine(get, cle)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("{quoi} sans propriété `{cle}` lisible"))
}

fn date_requise(get: Get<'_>, cle: &str, quoi: &str) -> Result<DateTime<Utc>> {
    let s = chaine(get, cle).ok_or_else(|| anyhow!("{quoi} sans propriété `{cle}`"))?;
    s.parse()
        .map_err(|e| anyhow!("{quoi} : `{cle}` illisible (« {s} ») : {e}"))
}

fn optionnelle_id(get: Get<'_>, cle: &str) -> Option<Uuid> {
    chaine(get, cle).and_then(|s| s.parse().ok())
}

fn valeur_liste(items: &[String]) -> Value {
    Value::List(
        items
            .iter()
            .map(|s| Value::from(s.as_str()))
            .collect::<Vec<Value>>()
            .into(),
    )
}

fn decoder_espace(get: Get<'_>) -> Result<(Uuid, WorkspaceNode)> {
    let id = id_requis(get, "id", "nœud Workspace")?;
    let quoi = format!("workspace {id}");
    let valeur = WorkspaceNode {
        id,
        name: chaine(get, "name").ok_or_else(|| anyhow!("{quoi} sans `name`"))?,
        slug: chaine(get, "slug").ok_or_else(|| anyhow!("{quoi} sans `slug`"))?,
        description: chaine(get, "description"),
        created_at: date_requise(get, "created_at", &quoi)?,
        updated_at: chaine(get, "updated_at").and_then(|s| s.parse().ok()),
        metadata: chaine(get, "metadata")
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| serde_json::json!({})),
    };
    Ok((id, valeur))
}

fn decoder_jalon(get: Get<'_>) -> Result<(Uuid, WorkspaceMilestoneNode)> {
    let id = id_requis(get, "id", "nœud WorkspaceMilestone")?;
    let quoi = format!("jalon {id}");
    let status = chaine(get, "status")
        .and_then(|s| serde_json::from_value::<MilestoneStatus>(serde_json::Value::String(s)).ok())
        .unwrap_or(MilestoneStatus::Open);
    let valeur = WorkspaceMilestoneNode {
        id,
        workspace_id: id_requis(get, "workspace_id", &quoi)?,
        title: chaine(get, "title").ok_or_else(|| anyhow!("{quoi} sans `title`"))?,
        description: chaine(get, "description"),
        status,
        target_date: chaine(get, "target_date").and_then(|s| s.parse().ok()),
        closed_at: chaine(get, "closed_at").and_then(|s| s.parse().ok()),
        created_at: date_requise(get, "created_at", &quoi)?,
        tags: liste(get, "tags"),
    };
    Ok((id, valeur))
}

/// Neo4j's mapping of a stored resource type (lowercase, underscores ignored).
fn type_de_ressource(s: &str) -> ResourceType {
    match s.to_lowercase().as_str() {
        "apicontract" | "api_contract" => ResourceType::ApiContract,
        "protobuf" => ResourceType::Protobuf,
        "graphqlschema" | "graphql_schema" => ResourceType::GraphqlSchema,
        "jsonschema" | "json_schema" => ResourceType::JsonSchema,
        "databaseschema" | "database_schema" => ResourceType::DatabaseSchema,
        "sharedtypes" | "shared_types" => ResourceType::SharedTypes,
        "config" => ResourceType::Config,
        "documentation" => ResourceType::Documentation,
        _ => ResourceType::Other,
    }
}

fn decoder_ressource(get: Get<'_>) -> Result<(Uuid, ResourceNode)> {
    let id = id_requis(get, "id", "nœud Resource")?;
    let quoi = format!("ressource {id}");
    let valeur = ResourceNode {
        id,
        workspace_id: optionnelle_id(get, "workspace_id"),
        project_id: optionnelle_id(get, "project_id"),
        name: chaine(get, "name").ok_or_else(|| anyhow!("{quoi} sans `name`"))?,
        resource_type: type_de_ressource(&chaine(get, "resource_type").unwrap_or_default()),
        file_path: chaine(get, "file_path").ok_or_else(|| anyhow!("{quoi} sans `file_path`"))?,
        url: chaine(get, "url"),
        format: chaine(get, "format"),
        version: chaine(get, "version"),
        description: chaine(get, "description"),
        created_at: date_requise(get, "created_at", &quoi)?,
        updated_at: chaine(get, "updated_at").and_then(|s| s.parse().ok()),
        metadata: chaine(get, "metadata")
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| serde_json::json!({})),
    };
    Ok((id, valeur))
}

/// Neo4j's mapping of a stored component type; an unknown spelling is `Other`.
fn type_de_composant(s: &str) -> ComponentType {
    match s.to_lowercase().as_str() {
        "service" => ComponentType::Service,
        "frontend" => ComponentType::Frontend,
        "worker" => ComponentType::Worker,
        "database" => ComponentType::Database,
        "messagequeue" | "message_queue" => ComponentType::MessageQueue,
        "cache" => ComponentType::Cache,
        "gateway" => ComponentType::Gateway,
        "external" => ComponentType::External,
        "library" => ComponentType::Library,
        "cli" => ComponentType::Cli,
        _ => ComponentType::Other,
    }
}

fn decoder_composant(get: Get<'_>) -> Result<(Uuid, ComponentNode)> {
    let id = id_requis(get, "id", "nœud Component")?;
    let quoi = format!("composant {id}");
    let valeur = ComponentNode {
        id,
        workspace_id: id_requis(get, "workspace_id", &quoi)?,
        name: chaine(get, "name").ok_or_else(|| anyhow!("{quoi} sans `name`"))?,
        component_type: type_de_composant(&chaine(get, "component_type").unwrap_or_default()),
        description: chaine(get, "description"),
        runtime: chaine(get, "runtime"),
        config: chaine(get, "config")
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| serde_json::json!({})),
        created_at: date_requise(get, "created_at", &quoi)?,
        tags: liste(get, "tags"),
    };
    Ok((id, valeur))
}

/// Decode the node `noeud` now (after its writes), as the index will hold it.
fn relire<T>(db: &ObrainDB, noeud: NodeId, decoder: Decodeur<T>) -> Result<(Uuid, T)> {
    let get = project::noeud_lisible(db, noeud)?;
    decoder(&get)
}

/// Create a node of `etiquette` as a draft: the marker first, the properties,
/// the edges (`liens`), then the marker removed last — the commit point. Returns
/// the node; the caller decodes and publishes it once durable.
fn creer_noeud(
    w: &mut Ecriture<'_>,
    etiquette: &'static str,
    proprietes: Vec<(&'static str, Value)>,
    liens: impl FnOnce(&ObrainDB, NodeId),
) -> Result<NodeId> {
    let db = w.db;
    let noeud = db.create_node(&[etiquette]);
    db.set_node_property(noeud, BROUILLON, Value::from(true));
    for (cle, valeur) in proprietes {
        db.set_node_property(noeud, cle, valeur);
    }
    liens(db, noeud);
    db.remove_node_property(noeud, BROUILLON);
    Ok(noeud)
}

/// Sort key of a milestone: its target date as stored (`""` when absent), then
/// title, then id.
fn cle_jalon(m: &WorkspaceMilestoneNode) -> (String, String, Uuid) {
    (
        m.target_date.map(|d| d.to_rfc3339()).unwrap_or_default(),
        m.title.clone(),
        m.id,
    )
}

/// Neo4j's slug rule (`src/neo4j/workspace.rs::validate_workspace_slug`), which
/// that module keeps private to the Neo4j adapter: same bounds, same message.
fn valider_slug(slug: &str) -> Result<()> {
    let longueur_ok = (2..=64).contains(&slug.len());
    let forme_ok = {
        let b = slug.as_bytes();
        let alnum = |c: u8| c.is_ascii_lowercase() || c.is_ascii_digit();
        longueur_ok
            && b.first().is_some_and(|&c| alnum(c))
            && b.last().is_some_and(|&c| alnum(c))
            && b.iter().all(|&c| alnum(c) || c == b'-')
    };
    if !longueur_ok {
        bail!(
            "Slug must be between 2 and 64 characters, got {}. \
             Expected format: ^[a-z0-9][a-z0-9-]*[a-z0-9]$",
            slug.len()
        );
    }
    if !forme_ok {
        bail!(
            "Invalid slug '{}'. Slugs must match ^[a-z0-9][a-z0-9-]*[a-z0-9]$ \
             (lowercase alphanumeric + hyphens, no leading/trailing hyphens)",
            slug
        );
    }
    Ok(())
}

/// A refusal of a method whose domain is not ported. The message is the one every
/// refusal of the backend has (`NonPorte`); `_raison` is the reason, kept at the
/// call site where it is read.
fn non_porte<T>(methode: &'static str, _raison: &'static str) -> Result<T> {
    Err(NonPorte(methode).into())
}

// ============================================================================
// The contract
// ============================================================================

#[async_trait]
impl WorkspaceStore for ObrainGraphStore {
    // ----- workspaces -------------------------------------------------------

    async fn create_workspace(&self, workspace: &WorkspaceNode) -> Result<()> {
        let ws = workspace.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                {
                    let etat = w.index()?;
                    if etat.espaces.espaces.contains_key(&ws.id) {
                        bail!(
                            "un workspace d'id {} existe déjà (unicité de Workspace.id)",
                            ws.id
                        );
                    }
                    if let Some(autre) = etat.espaces.par_slug.get(&ws.slug) {
                        bail!(
                            "le slug « {} » est déjà celui du workspace {autre} (unicité de \
                             Workspace.slug)",
                            ws.slug
                        );
                    }
                }
                let proprietes = vec![
                    ("id", Value::from(ws.id.to_string())),
                    ("name", Value::from(ws.name.clone())),
                    ("slug", Value::from(ws.slug.clone())),
                    (
                        "description",
                        Value::from(ws.description.clone().unwrap_or_default()),
                    ),
                    ("created_at", Value::from(ws.created_at.to_rfc3339())),
                    ("metadata", Value::from(ws.metadata.to_string())),
                ];
                let noeud = creer_noeud(w, ESPACE, proprietes, |_, _| {})?;
                let (id, valeur) = w.apres_ecriture(relire(w.db, noeud, decoder_espace))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.espaces.par_slug.insert(valeur.slug.clone(), id);
                etat.espaces.espaces.insert(id, Fiche { noeud, valeur });
                Ok(())
            },
        )
        .await
    }

    async fn get_workspace(&self, id: Uuid) -> Result<Option<WorkspaceNode>> {
        self.lire(|_, etat| Ok(etat.espaces.espaces.get(&id).map(|f| f.valeur.clone())))
    }

    async fn get_workspace_by_slug(&self, slug: &str) -> Result<Option<WorkspaceNode>> {
        self.lire(|_, etat| {
            Ok(etat
                .espaces
                .par_slug
                .get(slug)
                .and_then(|id| etat.espaces.espaces.get(id))
                .map(|f| f.valeur.clone()))
        })
    }

    async fn list_workspaces(&self) -> Result<Vec<WorkspaceNode>> {
        self.lire(|_, etat| {
            let mut out: Vec<WorkspaceNode> = etat
                .espaces
                .espaces
                .values()
                .map(|f| f.valeur.clone())
                .collect();
            // ORDER BY w.name, w.id ASC — total.
            out.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.id.cmp(&b.id)));
            Ok(out)
        })
    }

    async fn update_workspace(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        metadata: Option<serde_json::Value>,
        slug: Option<String>,
    ) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // The slug is validated and its uniqueness checked whether or not
                // the workspace exists (Neo4j runs the check first, then `SET`).
                if let Some(nouveau) = &slug {
                    valider_slug(nouveau)?;
                    let etat = w.index()?;
                    if let Some(autre) = etat.espaces.par_slug.get(nouveau) {
                        if *autre != id {
                            bail!(
                                "CONFLICT: Slug '{}' is already taken by another workspace",
                                nouveau
                            );
                        }
                    }
                }
                let trouve = {
                    let etat = w.index()?;
                    etat.espaces
                        .espaces
                        .get(&id)
                        .map(|f| (f.noeud, f.valeur.slug.clone()))
                };
                let Some((noeud, ancien_slug)) = trouve else {
                    return Ok(());
                };
                let db = w.db;
                if let Some(n) = &name {
                    db.set_node_property(noeud, "name", Value::from(n.clone()));
                }
                if let Some(d) = &description {
                    db.set_node_property(noeud, "description", Value::from(d.clone()));
                }
                if let Some(m) = &metadata {
                    db.set_node_property(noeud, "metadata", Value::from(m.to_string()));
                }
                if let Some(s) = &slug {
                    db.set_node_property(noeud, "slug", Value::from(s.clone()));
                }
                // Neo4j always sets `updated_at`, even for an empty update.
                db.set_node_property(noeud, "updated_at", Value::from(Utc::now().to_rfc3339()));
                let (nouvel_id, valeur) = w.apres_ecriture(relire(db, noeud, decoder_espace))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if valeur.slug != ancien_slug {
                    etat.espaces.par_slug.remove(&ancien_slug);
                    etat.espaces.par_slug.insert(valeur.slug.clone(), nouvel_id);
                }
                etat.espaces
                    .espaces
                    .insert(nouvel_id, Fiche { noeud, valeur });
                Ok(())
            },
        )
        .await
    }

    async fn delete_workspace(&self, id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, jalons, ressources, composants) = {
                    let etat = w.index()?;
                    let Some(f) = etat.espaces.espaces.get(&id) else {
                        return Ok(());
                    };
                    let de = |m: Option<&BTreeSet<Uuid>>| -> Vec<Uuid> {
                        m.into_iter().flatten().copied().collect::<Vec<_>>()
                    };
                    let jalons: Vec<Uuid> = de(etat.espaces.jalons_de.get(&id));
                    let ressources: Vec<Uuid> = de(etat.espaces.ressources_de.get(&id));
                    let composants: Vec<Uuid> = de(etat.espaces.composants_de.get(&id));
                    (f.noeud, jalons, ressources, composants)
                };
                // Children first, the workspace last: an interruption leaves a
                // workspace that still exists and can be deleted again.
                let db = w.db;
                let mut supprimes: Vec<(Uuid, NodeId, &'static str)> = Vec::new();
                {
                    let etat = w.index()?;
                    for j in &jalons {
                        if let Some(f) = etat.espaces.jalons.get(j) {
                            supprimes.push((*j, f.noeud, "jalon"));
                        }
                    }
                    for r in &ressources {
                        if let Some(f) = etat.espaces.ressources.get(r) {
                            supprimes.push((*r, f.noeud, "ressource"));
                        }
                    }
                    for c in &composants {
                        if let Some(f) = etat.espaces.composants.get(c) {
                            supprimes.push((*c, f.noeud, "composant"));
                        }
                    }
                }
                for (_, noeud_enfant, _) in &supprimes {
                    db.delete_node(*noeud_enfant);
                }
                if !db.delete_node(noeud) {
                    bail!("obrain a refusé la suppression du workspace {id}");
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for (cid, _, genre) in &supprimes {
                    match *genre {
                        "jalon" => etat.espaces.oublier_jalon(*cid),
                        "ressource" => etat.espaces.oublier_ressource(*cid),
                        _ => etat.espaces.oublier_composant(*cid),
                    }
                }
                if let Some(f) = etat.espaces.espaces.remove(&id) {
                    etat.espaces.par_slug.remove(&f.valeur.slug);
                }
                etat.espaces.jalons_de.remove(&id);
                etat.espaces.ressources_de.remove(&id);
                etat.espaces.composants_de.remove(&id);
                if let Some(projets) = etat.espaces.projets_de.remove(&id) {
                    for p in projets {
                        if let Some(s) = etat.espaces.espaces_de.get_mut(&p) {
                            s.remove(&id);
                        }
                    }
                }
                Ok(())
            },
        )
        .await
    }

    async fn add_project_to_workspace(&self, workspace_id: Uuid, project_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (wn, pn) = {
                    let etat = w.index()?;
                    let Some(wn) = etat.espaces.espaces.get(&workspace_id).map(|f| f.noeud) else {
                        return Ok(());
                    };
                    let Some(pn) = etat.projets.noeud_de(project_id) else {
                        return Ok(());
                    };
                    // MERGE: an existing membership is not written twice.
                    if etat
                        .espaces
                        .projets_de
                        .get(&workspace_id)
                        .is_some_and(|s| s.contains(&project_id))
                    {
                        return Ok(());
                    }
                    (wn, pn)
                };
                let _ = w.db.create_edge(pn, wn, APPARTIENT);
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.espaces
                    .projets_de
                    .entry(workspace_id)
                    .or_default()
                    .insert(project_id);
                etat.espaces
                    .espaces_de
                    .entry(project_id)
                    .or_default()
                    .insert(workspace_id);
                Ok(())
            },
        )
        .await
    }

    async fn remove_project_from_workspace(
        &self,
        workspace_id: Uuid,
        project_id: Uuid,
    ) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (wn, pn) = {
                    let etat = w.index()?;
                    let membre = etat
                        .espaces
                        .projets_de
                        .get(&workspace_id)
                        .is_some_and(|s| s.contains(&project_id));
                    if !membre {
                        return Ok(());
                    }
                    let (Some(wn), Some(pn)) = (
                        etat.espaces.espaces.get(&workspace_id).map(|f| f.noeud),
                        etat.projets.noeud_de(project_id),
                    ) else {
                        return Ok(());
                    };
                    (wn, pn)
                };
                for (cible, eid) in voisins(w.db, pn, APPARTIENT) {
                    if cible == wn {
                        w.db.delete_edge(eid);
                    }
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(s) = etat.espaces.projets_de.get_mut(&workspace_id) {
                    s.remove(&project_id);
                }
                if let Some(s) = etat.espaces.espaces_de.get_mut(&project_id) {
                    s.remove(&workspace_id);
                }
                Ok(())
            },
        )
        .await
    }

    async fn list_workspace_projects(&self, workspace_id: Uuid) -> Result<Vec<ProjectNode>> {
        self.lire(|_, etat| {
            let mut out: Vec<ProjectNode> = etat
                .espaces
                .projets_de
                .get(&workspace_id)
                .into_iter()
                .flatten()
                .filter_map(|p| etat.projets.projet(*p).cloned())
                .collect();
            // ORDER BY p.name, p.id ASC — total.
            out.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.id.cmp(&b.id)));
            Ok(out)
        })
    }

    async fn get_project_workspace(&self, project_id: Uuid) -> Result<Option<WorkspaceNode>> {
        // Neo4j answers any one of the workspaces a project belongs to; this
        // answers the smallest id, which is deterministic.
        self.lire(|_, etat| {
            Ok(etat
                .espaces
                .espaces_de
                .get(&project_id)
                .and_then(|s| s.iter().next())
                .and_then(|id| etat.espaces.espaces.get(id))
                .map(|f| f.valeur.clone()))
        })
    }

    async fn compute_coupling_matrix(&self, _workspace_id: Uuid) -> Result<CouplingMatrix> {
        non_porte(
            "WorkspaceStore::compute_coupling_matrix",
            "la matrice de couplage agrège fichiers, skills, notes et projets : ces domaines ne \
             sont pas portés",
        )
    }

    // ----- workspace milestones ---------------------------------------------

    async fn create_workspace_milestone(&self, milestone: &WorkspaceMilestoneNode) -> Result<()> {
        let m = milestone.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let wn = {
                    let etat = w.index()?;
                    if etat.espaces.jalons.contains_key(&m.id) {
                        bail!(
                            "un jalon d'id {} existe déjà (unicité de WorkspaceMilestone.id)",
                            m.id
                        );
                    }
                    // Neo4j's `MATCH` on the workspace: no workspace, nothing is created.
                    match etat.espaces.espaces.get(&m.workspace_id) {
                        Some(f) => f.noeud,
                        None => return Ok(()),
                    }
                };
                let proprietes = vec![
                    ("id", Value::from(m.id.to_string())),
                    ("workspace_id", Value::from(m.workspace_id.to_string())),
                    ("title", Value::from(m.title.clone())),
                    (
                        "description",
                        Value::from(m.description.clone().unwrap_or_default()),
                    ),
                    ("status", Value::from(m.status.to_string())),
                    (
                        "target_date",
                        Value::from(m.target_date.map(|d| d.to_rfc3339()).unwrap_or_default()),
                    ),
                    ("created_at", Value::from(m.created_at.to_rfc3339())),
                    ("tags", valeur_liste(&m.tags)),
                ];
                let noeud = creer_noeud(w, JALON, proprietes, |db, noeud| {
                    let _ = db.create_edge(wn, noeud, A_JALON);
                })?;
                let (id, valeur) = w.apres_ecriture(relire(w.db, noeud, decoder_jalon))?;
                let ws = valeur.workspace_id;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.espaces.jalons_de.entry(ws).or_default().insert(id);
                etat.espaces.jalons.insert(id, Fiche { noeud, valeur });
                Ok(())
            },
        )
        .await
    }

    async fn get_workspace_milestone(&self, id: Uuid) -> Result<Option<WorkspaceMilestoneNode>> {
        self.lire(|_, etat| Ok(etat.espaces.jalons.get(&id).map(|f| f.valeur.clone())))
    }

    async fn list_workspace_milestones(
        &self,
        workspace_id: Uuid,
    ) -> Result<Vec<WorkspaceMilestoneNode>> {
        self.lire(|_, etat| {
            // Sort the references (the keys are computed once per milestone), clone
            // what is returned only.
            let mut refs: Vec<&WorkspaceMilestoneNode> = etat
                .espaces
                .jalons_de
                .get(&workspace_id)
                .into_iter()
                .flatten()
                .filter_map(|j| etat.espaces.jalons.get(j).map(|f| &f.valeur))
                .collect();
            refs.sort_by_cached_key(|m| cle_jalon(m));
            Ok(refs.into_iter().cloned().collect())
        })
    }

    async fn list_workspace_milestones_filtered(
        &self,
        workspace_id: Uuid,
        status: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<WorkspaceMilestoneNode>, usize)> {
        let variantes = status.map(status_variants);
        self.lire(|_, etat| {
            let mut refs: Vec<&WorkspaceMilestoneNode> = etat
                .espaces
                .jalons_de
                .get(&workspace_id)
                .into_iter()
                .flatten()
                .filter_map(|j| etat.espaces.jalons.get(j).map(|f| &f.valeur))
                .filter(|m| {
                    variantes
                        .as_ref()
                        .is_none_or(|v| v.contains(&m.status.to_string()))
                })
                .collect();
            refs.sort_by_cached_key(|m| cle_jalon(m));
            let total = refs.len();
            let page = refs.into_iter().skip(offset).take(limit).cloned().collect();
            Ok((page, total))
        })
    }

    async fn list_all_workspace_milestones_filtered(
        &self,
        _workspace_id: Option<Uuid>,
        _status: Option<&str>,
        _limit: usize,
        _offset: usize,
    ) -> Result<Vec<(WorkspaceMilestoneNode, String, String, String)>> {
        non_porte(
            "WorkspaceStore::list_all_workspace_milestones_filtered",
            "la liste de tous les jalons, triée sur l'ensemble, demande un index ordonné global \
             qui n'existe pas encore",
        )
    }

    async fn count_all_workspace_milestones(
        &self,
        _workspace_id: Option<Uuid>,
        _status: Option<&str>,
    ) -> Result<usize> {
        non_porte(
            "WorkspaceStore::count_all_workspace_milestones",
            "le décompte de tous les jalons filtrés demande un compteur global par statut qui \
             n'existe pas encore",
        )
    }

    async fn update_workspace_milestone(
        &self,
        id: Uuid,
        title: Option<String>,
        description: Option<String>,
        status: Option<MilestoneStatus>,
        target_date: Option<DateTime<Utc>>,
    ) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let trouve = {
                    let etat = w.index()?;
                    etat.espaces.jalons.get(&id).map(|f| f.noeud)
                };
                let Some(noeud) = trouve else {
                    return Ok(false);
                };
                let db = w.db;
                if let Some(t) = &title {
                    db.set_node_property(noeud, "title", Value::from(t.clone()));
                }
                if let Some(d) = &description {
                    db.set_node_property(noeud, "description", Value::from(d.clone()));
                }
                if let Some(s) = status {
                    db.set_node_property(noeud, "status", Value::from(s.to_string()));
                }
                if let Some(td) = target_date {
                    db.set_node_property(noeud, "target_date", Value::from(td.to_rfc3339()));
                }
                let (nid, valeur) = w.apres_ecriture(relire(db, noeud, decoder_jalon))?;
                w.rendre_durable()?;
                w.index_mut()?
                    .espaces
                    .jalons
                    .insert(nid, Fiche { noeud, valeur });
                Ok(true)
            },
        )
        .await
    }

    async fn delete_workspace_milestone(&self, id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let trouve = {
                    let etat = w.index()?;
                    etat.espaces.jalons.get(&id).map(|f| f.noeud)
                };
                let Some(noeud) = trouve else {
                    return Ok(());
                };
                w.db.delete_node(noeud);
                w.rendre_durable()?;
                w.index_mut()?.espaces.oublier_jalon(id);
                Ok(())
            },
        )
        .await
    }

    async fn add_task_to_workspace_milestone(
        &self,
        _milestone_id: Uuid,
        _task_id: Uuid,
    ) -> Result<()> {
        non_porte(
            "WorkspaceStore::add_task_to_workspace_milestone",
            "lie un Task, que le domaine plan (non porté) possède",
        )
    }

    async fn remove_task_from_workspace_milestone(
        &self,
        _milestone_id: Uuid,
        _task_id: Uuid,
    ) -> Result<()> {
        non_porte(
            "WorkspaceStore::remove_task_from_workspace_milestone",
            "délie un Task, que le domaine plan (non porté) possède",
        )
    }

    async fn link_plan_to_workspace_milestone(
        &self,
        _plan_id: Uuid,
        _milestone_id: Uuid,
    ) -> Result<()> {
        non_porte(
            "WorkspaceStore::link_plan_to_workspace_milestone",
            "lie un Plan, que le domaine plan (non porté) possède",
        )
    }

    async fn unlink_plan_from_workspace_milestone(
        &self,
        _plan_id: Uuid,
        _milestone_id: Uuid,
    ) -> Result<()> {
        non_porte(
            "WorkspaceStore::unlink_plan_from_workspace_milestone",
            "délie un Plan, que le domaine plan (non porté) possède",
        )
    }

    async fn get_workspace_milestone_progress(
        &self,
        _milestone_id: Uuid,
    ) -> Result<(u32, u32, u32, u32)> {
        non_porte(
            "WorkspaceStore::get_workspace_milestone_progress",
            "le progrès lit les Task des plans liés, que le domaine plan (non porté) possède",
        )
    }

    async fn get_workspace_milestone_tasks(
        &self,
        _milestone_id: Uuid,
    ) -> Result<Vec<TaskWithPlan>> {
        non_porte(
            "WorkspaceStore::get_workspace_milestone_tasks",
            "lit les Task et Plan liés, que le domaine plan (non porté) possède",
        )
    }

    async fn get_workspace_milestone_steps(
        &self,
        _milestone_id: Uuid,
    ) -> Result<std::collections::HashMap<Uuid, Vec<StepNode>>> {
        non_porte(
            "WorkspaceStore::get_workspace_milestone_steps",
            "lit les Step des Task liés, que le domaine plan (non porté) possède",
        )
    }

    // ----- resources --------------------------------------------------------

    async fn create_resource(&self, resource: &ResourceNode) -> Result<()> {
        let r = resource.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (wn, pn) = {
                    let etat = w.index()?;
                    if etat.espaces.ressources.contains_key(&r.id) {
                        bail!(
                            "une ressource d'id {} existe déjà (unicité de Resource.id)",
                            r.id
                        );
                    }
                    // Neo4j creates the resource, then links it to each parent that exists.
                    let wn = r
                        .workspace_id
                        .and_then(|id| etat.espaces.espaces.get(&id).map(|f| f.noeud));
                    let pn = r.project_id.and_then(|id| etat.projets.noeud_de(id));
                    (wn, pn)
                };
                let proprietes = vec![
                    ("id", Value::from(r.id.to_string())),
                    (
                        "workspace_id",
                        Value::from(r.workspace_id.map(|i| i.to_string()).unwrap_or_default()),
                    ),
                    (
                        "project_id",
                        Value::from(r.project_id.map(|i| i.to_string()).unwrap_or_default()),
                    ),
                    ("name", Value::from(r.name.clone())),
                    (
                        "resource_type",
                        Value::from(format!("{:?}", r.resource_type)),
                    ),
                    ("file_path", Value::from(r.file_path.clone())),
                    ("url", Value::from(r.url.clone().unwrap_or_default())),
                    ("format", Value::from(r.format.clone().unwrap_or_default())),
                    (
                        "version",
                        Value::from(r.version.clone().unwrap_or_default()),
                    ),
                    (
                        "description",
                        Value::from(r.description.clone().unwrap_or_default()),
                    ),
                    ("created_at", Value::from(r.created_at.to_rfc3339())),
                    ("metadata", Value::from(r.metadata.to_string())),
                ];
                let noeud = creer_noeud(w, RESSOURCE, proprietes, |db, noeud| {
                    if let Some(wn) = wn {
                        let _ = db.create_edge(wn, noeud, A_RESSOURCE);
                    }
                    if let Some(pn) = pn {
                        let _ = db.create_edge(pn, noeud, A_RESSOURCE);
                    }
                })?;
                let (id, valeur) = w.apres_ecriture(relire(w.db, noeud, decoder_ressource))?;
                let ws = valeur.workspace_id;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let (Some(ws), true) = (ws, wn.is_some()) {
                    etat.espaces.ressources_de.entry(ws).or_default().insert(id);
                }
                etat.espaces.ressources.insert(id, Fiche { noeud, valeur });
                Ok(())
            },
        )
        .await
    }

    async fn get_resource(&self, id: Uuid) -> Result<Option<ResourceNode>> {
        self.lire(|_, etat| Ok(etat.espaces.ressources.get(&id).map(|f| f.valeur.clone())))
    }

    async fn list_workspace_resources(&self, workspace_id: Uuid) -> Result<Vec<ResourceNode>> {
        self.lire(|_, etat| {
            let mut out: Vec<ResourceNode> = etat
                .espaces
                .ressources_de
                .get(&workspace_id)
                .into_iter()
                .flatten()
                .filter_map(|r| etat.espaces.ressources.get(r).map(|f| f.valeur.clone()))
                .collect();
            out.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.id.cmp(&b.id)));
            Ok(out)
        })
    }

    async fn update_resource(
        &self,
        id: Uuid,
        name: Option<String>,
        file_path: Option<String>,
        url: Option<String>,
        version: Option<String>,
        description: Option<String>,
    ) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let trouve = {
                    let etat = w.index()?;
                    etat.espaces.ressources.get(&id).map(|f| f.noeud)
                };
                let Some(noeud) = trouve else {
                    return Ok(false);
                };
                let db = w.db;
                for (cle, v) in [
                    ("name", name),
                    ("file_path", file_path),
                    ("url", url),
                    ("version", version),
                    ("description", description),
                ] {
                    if let Some(v) = v {
                        db.set_node_property(noeud, cle, Value::from(v));
                    }
                }
                let (rid, valeur) = w.apres_ecriture(relire(db, noeud, decoder_ressource))?;
                w.rendre_durable()?;
                w.index_mut()?
                    .espaces
                    .ressources
                    .insert(rid, Fiche { noeud, valeur });
                Ok(true)
            },
        )
        .await
    }

    async fn delete_resource(&self, id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let trouve = {
                    let etat = w.index()?;
                    etat.espaces.ressources.get(&id).map(|f| f.noeud)
                };
                let Some(noeud) = trouve else {
                    return Ok(());
                };
                w.db.delete_node(noeud);
                w.rendre_durable()?;
                w.index_mut()?.espaces.oublier_ressource(id);
                Ok(())
            },
        )
        .await
    }

    async fn link_project_implements_resource(
        &self,
        project_id: Uuid,
        resource_id: Uuid,
    ) -> Result<()> {
        lier_projet_ressource(self, project_id, resource_id, IMPLEMENTE).await
    }

    async fn link_project_uses_resource(&self, project_id: Uuid, resource_id: Uuid) -> Result<()> {
        lier_projet_ressource(self, project_id, resource_id, UTILISE).await
    }

    async fn get_resource_implementers(&self, resource_id: Uuid) -> Result<Vec<ProjectNode>> {
        self.lire(|_, etat| {
            Ok(projets_de_ressource(
                etat,
                resource_id,
                &etat.espaces.implementeurs,
            ))
        })
    }

    async fn get_resource_consumers(&self, resource_id: Uuid) -> Result<Vec<ProjectNode>> {
        self.lire(|_, etat| {
            Ok(projets_de_ressource(
                etat,
                resource_id,
                &etat.espaces.consommateurs,
            ))
        })
    }

    // ----- components -------------------------------------------------------

    async fn create_component(&self, component: &ComponentNode) -> Result<()> {
        let c = component.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let wn = {
                    let etat = w.index()?;
                    if etat.espaces.composants.contains_key(&c.id) {
                        bail!(
                            "un composant d'id {} existe déjà (unicité de Component.id)",
                            c.id
                        );
                    }
                    if etat
                        .espaces
                        .par_nom_composant
                        .contains_key(&(c.workspace_id, c.name.clone()))
                    {
                        bail!(
                            "le workspace {} a déjà un composant nommé « {} » (unicité de son nom)",
                            c.workspace_id,
                            c.name
                        );
                    }
                    // Neo4j's `MATCH` on the workspace: no workspace, nothing is created.
                    match etat.espaces.espaces.get(&c.workspace_id) {
                        Some(f) => f.noeud,
                        None => return Ok(()),
                    }
                };
                let proprietes = vec![
                    ("id", Value::from(c.id.to_string())),
                    ("workspace_id", Value::from(c.workspace_id.to_string())),
                    ("name", Value::from(c.name.clone())),
                    (
                        "component_type",
                        Value::from(format!("{:?}", c.component_type)),
                    ),
                    (
                        "description",
                        Value::from(c.description.clone().unwrap_or_default()),
                    ),
                    (
                        "runtime",
                        Value::from(c.runtime.clone().unwrap_or_default()),
                    ),
                    ("config", Value::from(c.config.to_string())),
                    ("created_at", Value::from(c.created_at.to_rfc3339())),
                    ("tags", valeur_liste(&c.tags)),
                ];
                let noeud = creer_noeud(w, COMPOSANT, proprietes, |db, noeud| {
                    let _ = db.create_edge(wn, noeud, A_COMPOSANT);
                })?;
                let (id, valeur) = w.apres_ecriture(relire(w.db, noeud, decoder_composant))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.espaces
                    .composants_de
                    .entry(valeur.workspace_id)
                    .or_default()
                    .insert(id);
                etat.espaces
                    .par_nom_composant
                    .insert((valeur.workspace_id, valeur.name.clone()), id);
                etat.espaces.composants.insert(id, Fiche { noeud, valeur });
                Ok(())
            },
        )
        .await
    }

    async fn upsert_derived_component(&self, write: DerivedComponentWrite) -> Result<Uuid> {
        let DerivedComponentWrite {
            workspace_id,
            name,
            component_type,
            description,
            runtime,
            tags,
            config,
        } = write;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (wn, existant) = {
                    let etat = w.index()?;
                    let Some(wn) = etat.espaces.espaces.get(&workspace_id).map(|f| f.noeud) else {
                        bail!(
                            "workspace {workspace_id} not found while upserting component '{name}'"
                        );
                    };
                    let existant = etat
                        .espaces
                        .par_nom_composant
                        .get(&(workspace_id, name.clone()))
                        .and_then(|cid| etat.espaces.composants.get(cid))
                        .map(|f| (f.noeud, f.valeur.clone()));
                    (wn, existant)
                };
                let db = w.db;
                match existant {
                    // MERGE matched: the SET rules of the Neo4j query, field by field.
                    Some((noeud, ancien)) => {
                        let cid = ancien.id;
                        db.set_node_property(
                            noeud,
                            "component_type",
                            Value::from(format!("{:?}", component_type)),
                        );
                        let runtime = runtime.unwrap_or_default();
                        if !runtime.is_empty() {
                            db.set_node_property(noeud, "runtime", Value::from(runtime));
                        }
                        db.set_node_property(noeud, "config", Value::from(config.to_string()));
                        if ancien.description.as_deref().unwrap_or("").is_empty() {
                            db.set_node_property(
                                noeud,
                                "description",
                                Value::from(description.unwrap_or_default()),
                            );
                        }
                        if ancien.tags.is_empty() {
                            db.set_node_property(noeud, "tags", valeur_liste(&tags));
                        }
                        let (_, valeur) = w.apres_ecriture(relire(db, noeud, decoder_composant))?;
                        w.rendre_durable()?;
                        w.index_mut()?
                            .espaces
                            .composants
                            .insert(cid, Fiche { noeud, valeur });
                        Ok(cid)
                    }
                    // MERGE created: the ON CREATE values, then the same SET rules.
                    None => {
                        let cid = Uuid::new_v4();
                        let proprietes = vec![
                            ("id", Value::from(cid.to_string())),
                            ("workspace_id", Value::from(workspace_id.to_string())),
                            ("name", Value::from(name.clone())),
                            (
                                "component_type",
                                Value::from(format!("{:?}", component_type)),
                            ),
                            ("description", Value::from(description.unwrap_or_default())),
                            ("runtime", Value::from(runtime.unwrap_or_default())),
                            ("config", Value::from(config.to_string())),
                            ("created_at", Value::from(Utc::now().to_rfc3339())),
                            ("tags", valeur_liste(&tags)),
                        ];
                        let noeud = creer_noeud(w, COMPOSANT, proprietes, |db, noeud| {
                            let _ = db.create_edge(wn, noeud, A_COMPOSANT);
                        })?;
                        let (id, valeur) =
                            w.apres_ecriture(relire(w.db, noeud, decoder_composant))?;
                        w.rendre_durable()?;
                        let mut etat = w.index_mut()?;
                        etat.espaces
                            .composants_de
                            .entry(workspace_id)
                            .or_default()
                            .insert(id);
                        etat.espaces
                            .par_nom_composant
                            .insert((workspace_id, name), id);
                        etat.espaces.composants.insert(id, Fiche { noeud, valeur });
                        Ok(id)
                    }
                }
            },
        )
        .await
    }

    async fn get_component(&self, id: Uuid) -> Result<Option<ComponentNode>> {
        self.lire(|_, etat| Ok(etat.espaces.composants.get(&id).map(|f| f.valeur.clone())))
    }

    async fn list_components(&self, workspace_id: Uuid) -> Result<Vec<ComponentNode>> {
        self.lire(|_, etat| {
            let mut out: Vec<ComponentNode> = etat
                .espaces
                .composants_de
                .get(&workspace_id)
                .into_iter()
                .flatten()
                .filter_map(|c| etat.espaces.composants.get(c).map(|f| f.valeur.clone()))
                .collect();
            out.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.id.cmp(&b.id)));
            Ok(out)
        })
    }

    async fn update_component(&self, id: Uuid, patch: ComponentUpdate) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let trouve = {
                    let etat = w.index()?;
                    etat.espaces
                        .composants
                        .get(&id)
                        .map(|f| (f.noeud, f.valeur.workspace_id, f.valeur.name.clone()))
                };
                let Some((noeud, ws, ancien_nom)) = trouve else {
                    return Ok(false);
                };
                if let Some(nouveau) = &patch.name {
                    let etat = w.index()?;
                    if let Some(autre) = etat.espaces.par_nom_composant.get(&(ws, nouveau.clone()))
                    {
                        if *autre != id {
                            bail!(
                                "le workspace {ws} a déjà un composant nommé « {nouveau} » \
                                 (unicité de son nom)"
                            );
                        }
                    }
                }
                let db = w.db;
                if let Some(n) = &patch.name {
                    db.set_node_property(noeud, "name", Value::from(n.clone()));
                }
                if let Some(t) = &patch.component_type {
                    db.set_node_property(noeud, "component_type", Value::from(format!("{:?}", t)));
                }
                if let Some(d) = &patch.description {
                    db.set_node_property(noeud, "description", Value::from(d.clone()));
                }
                if let Some(r) = &patch.runtime {
                    db.set_node_property(noeud, "runtime", Value::from(r.clone()));
                }
                if let Some(c) = &patch.config {
                    db.set_node_property(noeud, "config", Value::from(c.to_string()));
                }
                if let Some(t) = &patch.tags {
                    db.set_node_property(noeud, "tags", valeur_liste(t));
                }
                let (cid, valeur) = w.apres_ecriture(relire(db, noeud, decoder_composant))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if valeur.name != ancien_nom {
                    etat.espaces.par_nom_composant.remove(&(ws, ancien_nom));
                    etat.espaces
                        .par_nom_composant
                        .insert((ws, valeur.name.clone()), cid);
                }
                etat.espaces.composants.insert(cid, Fiche { noeud, valeur });
                Ok(true)
            },
        )
        .await
    }

    async fn delete_component(&self, id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let trouve = {
                    let etat = w.index()?;
                    etat.espaces.composants.get(&id).map(|f| f.noeud)
                };
                let Some(noeud) = trouve else {
                    return Ok(());
                };
                // DETACH DELETE: the component's dependencies and mappings go with it.
                w.db.delete_node(noeud);
                w.rendre_durable()?;
                w.index_mut()?.espaces.oublier_composant(id);
                Ok(())
            },
        )
        .await
    }

    async fn add_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
        protocol: Option<String>,
        required: bool,
    ) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (c1, c2) = {
                    let etat = w.index()?;
                    let (Some(a), Some(b)) = (
                        etat.espaces.composants.get(&component_id).map(|f| f.noeud),
                        etat.espaces.composants.get(&depends_on_id).map(|f| f.noeud),
                    ) else {
                        return Ok(());
                    };
                    (a, b)
                };
                let protocole = protocol.unwrap_or_default();
                // MERGE the relation, then SET its two properties (on the existing
                // one or on the new one).
                let existante = voisins(w.db, c1, DEPEND)
                    .into_iter()
                    .find(|(cible, _)| *cible == c2)
                    .map(|(_, eid)| eid);
                let eid = match existante {
                    Some(eid) => eid,
                    None => w.db.create_edge(c1, c2, DEPEND),
                };
                w.db.set_edge_property(eid, "protocol", Value::from(protocole.clone()));
                w.db.set_edge_property(eid, "required", Value::from(required));
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.espaces
                    .dependances
                    .entry(component_id)
                    .or_default()
                    .insert(depends_on_id, (protocole, required));
                etat.espaces
                    .dependants
                    .entry(depends_on_id)
                    .or_default()
                    .insert(component_id);
                Ok(())
            },
        )
        .await
    }

    async fn remove_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
    ) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (c1, c2) = {
                    let etat = w.index()?;
                    let (Some(a), Some(b)) = (
                        etat.espaces.composants.get(&component_id).map(|f| f.noeud),
                        etat.espaces.composants.get(&depends_on_id).map(|f| f.noeud),
                    ) else {
                        return Ok(());
                    };
                    (a, b)
                };
                for (cible, eid) in voisins(w.db, c1, DEPEND) {
                    if cible == c2 {
                        w.db.delete_edge(eid);
                    }
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(m) = etat.espaces.dependances.get_mut(&component_id) {
                    m.remove(&depends_on_id);
                }
                if let Some(s) = etat.espaces.dependants.get_mut(&depends_on_id) {
                    s.remove(&component_id);
                }
                Ok(())
            },
        )
        .await
    }

    async fn map_component_to_project(&self, component_id: Uuid, project_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (cn, pn) = {
                    let etat = w.index()?;
                    let (Some(c), Some(p)) = (
                        etat.espaces.composants.get(&component_id).map(|f| f.noeud),
                        etat.projets.noeud_de(project_id),
                    ) else {
                        return Ok(());
                    };
                    if etat
                        .espaces
                        .projets_du_composant
                        .get(&component_id)
                        .is_some_and(|s| s.contains(&project_id))
                    {
                        return Ok(());
                    }
                    (c, p)
                };
                let _ = w.db.create_edge(cn, pn, MAPPE);
                w.rendre_durable()?;
                w.index_mut()?
                    .espaces
                    .projets_du_composant
                    .entry(component_id)
                    .or_default()
                    .insert(project_id);
                Ok(())
            },
        )
        .await
    }

    async fn get_workspace_topology(
        &self,
        workspace_id: Uuid,
    ) -> Result<Vec<(ComponentNode, Option<String>, Vec<ComponentDependency>)>> {
        self.lire(|_, etat| {
            let mut composants: Vec<&ComponentNode> = etat
                .espaces
                .composants_de
                .get(&workspace_id)
                .into_iter()
                .flatten()
                .filter_map(|c| etat.espaces.composants.get(c).map(|f| &f.valeur))
                .collect();
            composants.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.id.cmp(&b.id)));
            let mut out = Vec::new();
            for c in composants {
                let dependances: Vec<ComponentDependency> = etat
                    .espaces
                    .dependances
                    .get(&c.id)
                    .into_iter()
                    .flat_map(|m| m.iter())
                    .map(|(to, (protocole, requis))| ComponentDependency {
                        from_id: c.id,
                        to_id: *to,
                        protocol: Some(protocole.clone()),
                        required: *requis,
                    })
                    .collect();
                // One row per mapped project, as the Neo4j `OPTIONAL MATCH` gives; a
                // component mapped to none is one row with no project name.
                let mut noms: Vec<(String, Uuid)> = etat
                    .espaces
                    .projets_du_composant
                    .get(&c.id)
                    .into_iter()
                    .flatten()
                    .filter_map(|p| etat.projets.projet(*p).map(|pn| (pn.name.clone(), pn.id)))
                    .collect();
                noms.sort();
                if noms.is_empty() {
                    out.push((c.clone(), None, dependances));
                } else {
                    for (nom, _) in noms {
                        out.push((c.clone(), Some(nom), dependances.clone()));
                    }
                }
            }
            Ok(out)
        })
    }
}

/// `MERGE (p)-[:IMPLEMENTS_RESOURCE|USES_RESOURCE]->(r)` when both exist.
async fn lier_projet_ressource(
    store: &ObrainGraphStore,
    project_id: Uuid,
    resource_id: Uuid,
    type_: &'static str,
) -> Result<()> {
    store
        .ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (pn, rn) = {
                    let etat = w.index()?;
                    let (Some(p), Some(r)) = (
                        etat.projets.noeud_de(project_id),
                        etat.espaces.ressources.get(&resource_id).map(|f| f.noeud),
                    ) else {
                        return Ok(());
                    };
                    let deja = if type_ == IMPLEMENTE {
                        &etat.espaces.implementeurs
                    } else {
                        &etat.espaces.consommateurs
                    };
                    if deja
                        .get(&resource_id)
                        .is_some_and(|s| s.contains(&project_id))
                    {
                        return Ok(());
                    }
                    (p, r)
                };
                let _ = w.db.create_edge(pn, rn, type_);
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                let carte = if type_ == IMPLEMENTE {
                    &mut etat.espaces.implementeurs
                } else {
                    &mut etat.espaces.consommateurs
                };
                carte.entry(resource_id).or_default().insert(project_id);
                Ok(())
            },
        )
        .await
}

/// The projects of a resource through one of its two relations, sorted by
/// `(name, id)`.
fn projets_de_ressource(
    etat: &super::Etat,
    resource_id: Uuid,
    carte: &HashMap<Uuid, BTreeSet<Uuid>>,
) -> Vec<ProjectNode> {
    let mut out: Vec<ProjectNode> = carte
        .get(&resource_id)
        .into_iter()
        .flatten()
        .filter_map(|p| etat.projets.projet(*p).cloned())
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.id.cmp(&b.id)));
    out
}
