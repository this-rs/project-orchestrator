//! `FeatureGraphStore` on obrain — a PARTIAL port of the domain.
//!
//! Ported: the feature graph NODE itself — `create_feature_graph`,
//! `get_feature_graph`, `list_feature_graphs`, `delete_feature_graph`.
//!
//! Refused by name (`NonPorte`), because each one reads or writes the nodes of
//! the Code domain (`File`, `Function`, `Struct`, `Trait`, `Enum`, `CALLS`,
//! `IMPORTS`, `CONTAINS`…), which is not ported (obrain #120):
//! `get_feature_graph_detail`, `add_entity_to_feature_graph`,
//! `remove_entity_from_feature_graph`, `auto_build_feature_graph`,
//! `refresh_feature_graph`, `get_top_entry_functions`,
//! `get_feature_graph_statistics`, `compare_feature_graphs`,
//! `find_overlapping_feature_graphs`.
//!
//! The node has the SAME property names and conventions as the Neo4j adapter
//! (`src/neo4j/feature_graph.rs`, the conformance reference):
//! `id`, `name`, `description` (`""` when absent, read back as `None`),
//! `project_id` (string), `created_at` / `updated_at` (RFC 3339 strings),
//! `entry_function` (`""` when absent, read back as `Some("")`, as Neo4j does),
//! `build_depth` (integer, `0` when absent), `include_relations` (JSON string,
//! `""` when absent, read back as `None`).
//!
//! Deviations from Neo4j, all declared in the PR:
//!
//! - no `BELONGS_TO` edge to the `Project` node: the project index is private to
//!   `project.rs`, and the link is carried by `project_id`. The Neo4j create
//!   also accepts a graph whose project does not exist; this one does too.
//! - `created_at` / `updated_at` that do not parse make the node unaddressable
//!   (quarantined at open), where Neo4j substitutes "now".
//! - `entity_count` is `None` on `get_feature_graph` (as Neo4j) and `Some(n)`
//!   on `list_feature_graphs`, where `n` is the number of outgoing
//!   `INCLUDES_ENTITY` edges, counted at open and at creation. This backend
//!   writes no such edge (adding an entity is refused), so `n` is nonzero only
//!   for data written elsewhere (a migrated base).
//! - `delete_feature_graph` deletes the node and its edges (obrain's
//!   `delete_node` removes them, as `DETACH DELETE` does): the entities at the
//!   other end are not touched.
//!
//! Atomicity, durability and the reads are those of `project.rs`: a creation
//! writes a draft marker first and removes it last (the commit point), a draft is
//! deleted at open, a value above [`SEUIL_COLONNE_BLOB`] pays the full flush
//! before the marker is removed, and a write is published to the index only
//! after its durability step.
//!
//! Order: `list_feature_graphs` sorts by `(updated_at, id)` DESCENDING, as
//! `ORDER BY fg.updated_at DESC, fg.id DESC`. `updated_at` compares as the
//! stored STRING, which is what Neo4j compares.
//!
//! Effect on the cascade: `delete_project` counts every node of the base
//! against the projects it addresses, so it refuses while a feature graph
//! exists. That is the refusal the cascade already gives for an unported domain.

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
use crate::neo4j::models::{
    FeatureGraphComparison, FeatureGraphDetail, FeatureGraphNode, FeatureGraphOverlap,
    FeatureGraphStatistics,
};
use crate::neo4j::traits::FeatureGraphStore;

const LABEL: &str = "FeatureGraph";
/// The draft marker: present from the first record of a creation to the last.
const BROUILLON: &str = "_brouillon";
/// The relation a graph holds to each entity it includes.
const INCLUDES_ENTITY: &str = "INCLUDES_ENTITY";

/// Does this value go to a blob column (a string above the threshold)?
fn va_en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
}

/// The properties a creation writes, in the order it writes them. Absent
/// optional values are stored with the Neo4j adapter's placeholders.
fn proprietes_de_creation(g: &FeatureGraphNode) -> Vec<(&'static str, Value)> {
    vec![
        ("id", Value::from(g.id.to_string())),
        ("name", Value::from(g.name.clone())),
        (
            "description",
            Value::from(g.description.clone().unwrap_or_default()),
        ),
        ("project_id", Value::from(g.project_id.to_string())),
        ("created_at", Value::from(g.created_at.to_rfc3339())),
        ("updated_at", Value::from(g.updated_at.to_rfc3339())),
        (
            "entry_function",
            Value::from(g.entry_function.clone().unwrap_or_default()),
        ),
        (
            "build_depth",
            Value::from(g.build_depth.map(i64::from).unwrap_or(0)),
        ),
        (
            "include_relations",
            Value::from(
                g.include_relations
                    .as_ref()
                    .map(|r| serde_json::to_string(r).unwrap_or_default())
                    .unwrap_or_default(),
            ),
        ),
    ]
}

/// The domain value of a `FeatureGraph` node from a property lookup — the
/// counterpart of `node_to_feature_graph` (Neo4j), field for field except the
/// strict timestamps (module doc). `entity_count` is left `None`.
fn lire_graphe(get: &dyn Fn(&str) -> Option<Value>) -> Result<FeatureGraphNode> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let id: Uuid = chaine("id")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("nœud FeatureGraph sans propriété `id` lisible"))?;
    let instant = |cle: &str| -> Result<DateTime<Utc>> {
        let s = chaine(cle).ok_or_else(|| anyhow!("graphe {id} sans propriété `{cle}`"))?;
        s.parse()
            .map_err(|e| anyhow!("graphe {id} : `{cle}` illisible (« {s} ») : {e}"))
    };
    Ok(FeatureGraphNode {
        id,
        name: chaine("name").ok_or_else(|| anyhow!("graphe {id} sans propriété `name`"))?,
        description: chaine("description").filter(|s| !s.is_empty()),
        project_id: chaine("project_id")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| anyhow!("graphe {id} : `project_id` illisible"))?,
        created_at: instant("created_at")?,
        updated_at: instant("updated_at")?,
        entity_count: None,
        // Read back as the stored string, even when empty (as Neo4j does).
        entry_function: chaine("entry_function"),
        build_depth: get("build_depth")
            .and_then(|v| v.as_int64())
            .and_then(|v| u32::try_from(v).ok()),
        include_relations: chaine("include_relations").and_then(|s| serde_json::from_str(&s).ok()),
    })
}

/// One feature graph: where it lives in the base, what the base says about it,
/// decoded once. Reads are served from here.
struct Entree {
    noeud: NodeId,
    graphe: FeatureGraphNode,
    /// Outgoing `INCLUDES_ENTITY` edges, as the base holds them.
    entites: i64,
    /// `updated_at` as stored: the sort key of `list_feature_graphs` (Neo4j
    /// sorts the string).
    tri_maj: String,
}

impl Entree {
    /// Read the node (one hydration, one degree count) and decode it.
    fn decoder(db: &ObrainDB, noeud: NodeId) -> Result<Self> {
        let n = db
            .get_node(noeud)
            .ok_or_else(|| anyhow!("nœud FeatureGraph {noeud:?} indexé mais absent de la base"))?;
        let get = |cle: &str| {
            n.get_property(cle)
                .filter(|v| !matches!(v, Value::Null))
                .cloned()
        };
        let graphe = lire_graphe(&get)?;
        let tri_maj = get("updated_at")
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default();
        let entites = db
            .store()
            .out_degree_by_type(noeud, Some(INCLUDES_ENTITY)) as i64;
        Ok(Self {
            noeud,
            graphe,
            entites,
            tri_maj,
        })
    }
}

/// `Uuid → feature graph`, rebuilt from the `FeatureGraph` nodes at open.
#[derive(Default)]
pub(super) struct IndexGraphes {
    par_id: HashMap<Uuid, Entree>,
}

impl IndexGraphes {
    /// Rebuild the index, deleting the drafts an interrupted creation left and
    /// quarantining what cannot be addressed (see `project.rs`, "Opening").
    pub(super) fn reconstruire(db: &ObrainDB, rapport: &mut RapportOuverture) -> Result<Self> {
        let store = db.store();
        let mut index = Self::default();
        for noeud in store.nodes_by_label(LABEL) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud FeatureGraph {noeud:?} : illisible"));
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
                rapport.graphes_brouillons_supprimes += 1;
                continue;
            }
            let Ok(graphe) = lire_graphe(&get) else {
                rapport.en_quarantaine.push(format!(
                    "nœud FeatureGraph {noeud:?} : id/name/project_id/dates illisibles"
                ));
                continue;
            };
            if index.par_id.contains_key(&graphe.id) {
                rapport.en_quarantaine.push(format!(
                    "nœud FeatureGraph {noeud:?} : id {} déjà pris",
                    graphe.id
                ));
                continue;
            }
            index.par_id.insert(graphe.id, Entree::decoder(db, noeud)?);
        }
        Ok(index)
    }
}

/// A read or write of the Code domain, refused by name.
fn non_porte<T>(methode: &'static str) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)))
}

#[async_trait]
impl FeatureGraphStore for ObrainGraphStore {
    async fn create_feature_graph(&self, graph: &FeatureGraphNode) -> Result<()> {
        let g = graph.clone();
        self.ecrire(
            move || {
                let proprietes = proprietes_de_creation(&g);
                (g, proprietes)
            },
            move |(g, proprietes), w: &mut Ecriture<'_>| {
                // Validate first: nothing is written when this refuses.
                {
                    let etat = w.index()?;
                    if etat.graphes.par_id.contains_key(&g.id) {
                        bail!(
                            "une feature graph d'id {} existe déjà (unicité de FeatureGraph.id)",
                            g.id
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
                // A blob-sized value is recovered only by the full flush: it must be
                // on disk BEFORE the marker says the graph is complete.
                w.rendre_durable_avant_le_marqueur()?;
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::AvantLeRetraitDuMarqueur)?;
                // The commit point.
                w.db.remove_node_property(noeud, BROUILLON);
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeRetraitDuMarqueur)?;
                let entree = w.apres_ecriture(Entree::decoder(w.db, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.graphes.par_id.insert(g.id, entree);
                Ok(())
            },
        )
        .await
    }

    async fn get_feature_graph(&self, id: Uuid) -> Result<Option<FeatureGraphNode>> {
        self.lire(|_, etat| Ok(etat.graphes.par_id.get(&id).map(|e| e.graphe.clone())))
    }

    async fn get_feature_graph_detail(&self, _id: Uuid) -> Result<Option<FeatureGraphDetail>> {
        non_porte("FeatureGraphStore::get_feature_graph_detail (entités : domaine Code non porté)")
    }

    async fn list_feature_graphs(
        &self,
        project_id: Option<Uuid>,
    ) -> Result<Vec<FeatureGraphNode>> {
        self.lire(|_, etat| {
            let mut entrees: Vec<&Entree> = etat
                .graphes
                .par_id
                .values()
                .filter(|e| project_id.is_none_or(|pid| e.graphe.project_id == pid))
                .collect();
            // ORDER BY fg.updated_at DESC, fg.id DESC — total: the id breaks ties.
            entrees.sort_by(|a, b| {
                b.tri_maj
                    .cmp(&a.tri_maj)
                    .then_with(|| b.graphe.id.cmp(&a.graphe.id))
            });
            Ok(entrees
                .into_iter()
                .map(|e| {
                    let mut g = e.graphe.clone();
                    g.entity_count = Some(e.entites);
                    g
                })
                .collect())
        })
    }

    async fn delete_feature_graph(&self, id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud = {
                    let etat = w.index()?;
                    match etat.graphes.par_id.get(&id) {
                        Some(e) => e.noeud,
                        None => return Ok(false),
                    }
                };
                // `delete_node` removes the node's edges too (DETACH DELETE).
                if !w.db.delete_node(noeud) {
                    bail!("obrain a refusé la suppression du nœud {noeud:?} (graphe {id})");
                }
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLaSuppression)?;
                // Gone for the readers once the deletion is durable, not before.
                w.rendre_durable()?;
                w.index_mut()?.graphes.par_id.remove(&id);
                Ok(true)
            },
        )
        .await
    }

    async fn add_entity_to_feature_graph(
        &self,
        _feature_graph_id: Uuid,
        _entity_type: &str,
        _entity_id: &str,
        _role: Option<&str>,
        _project_id: Option<Uuid>,
    ) -> Result<()> {
        non_porte("FeatureGraphStore::add_entity_to_feature_graph (domaine Code non porté)")
    }

    async fn remove_entity_from_feature_graph(
        &self,
        _feature_graph_id: Uuid,
        _entity_type: &str,
        _entity_id: &str,
    ) -> Result<bool> {
        non_porte("FeatureGraphStore::remove_entity_from_feature_graph (domaine Code non porté)")
    }

    async fn auto_build_feature_graph(
        &self,
        _name: &str,
        _description: Option<&str>,
        _project_id: Uuid,
        _entry_function: &str,
        _depth: u32,
        _include_relations: Option<&[String]>,
        _filter_community: Option<bool>,
    ) -> Result<FeatureGraphDetail> {
        non_porte("FeatureGraphStore::auto_build_feature_graph (domaine Code non porté)")
    }

    async fn refresh_feature_graph(&self, _id: Uuid) -> Result<Option<FeatureGraphDetail>> {
        non_porte("FeatureGraphStore::refresh_feature_graph (domaine Code non porté)")
    }

    async fn get_top_entry_functions(&self, _project_id: Uuid, _limit: usize) -> Result<Vec<String>> {
        non_porte("FeatureGraphStore::get_top_entry_functions (domaine Code non porté)")
    }

    async fn get_feature_graph_statistics(
        &self,
        _id: Uuid,
    ) -> Result<Option<FeatureGraphStatistics>> {
        non_porte("FeatureGraphStore::get_feature_graph_statistics (domaine Code non porté)")
    }

    async fn compare_feature_graphs(
        &self,
        _id_a: Uuid,
        _id_b: Uuid,
    ) -> Result<Option<FeatureGraphComparison>> {
        non_porte("FeatureGraphStore::compare_feature_graphs (domaine Code non porté)")
    }

    async fn find_overlapping_feature_graphs(
        &self,
        _id: Uuid,
        _min_overlap: f64,
    ) -> Result<Vec<FeatureGraphOverlap>> {
        non_porte("FeatureGraphStore::find_overlapping_feature_graphs (domaine Code non porté)")
    }
}
