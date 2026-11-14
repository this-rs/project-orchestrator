//! `FeatureGraphStore` on obrain: the feature-graph NODE (lot O-FEATUREGRAPH-1).
//!
//! A feature graph is mostly a view on the code graph: its entities are `File`,
//! `Function`, `Struct`, `Trait`… nodes reached by `INCLUDES_ENTITY`, and most of
//! its methods walk `CALLS`/`IMPORTS` edges. The code domain is not ported, so
//! only what reads or writes the node itself is served here.
//!
//! PORTED, with the semantics of `src/neo4j/feature_graph.rs`:
//! - `create_feature_graph`, `get_feature_graph`, `list_feature_graphs`
//!   (`entity_count` is the number of `INCLUDES_ENTITY` edges on disk, whatever
//!   their target), `delete_feature_graph` (DETACH: every edge goes, the persona
//!   `SCOPED_TO` edges included, see `persona.rs`);
//! - `get_feature_graph_detail` for a graph that INCLUDES NOTHING: its entities
//!   and relations are then honestly empty. A graph with `INCLUDES_ENTITY` edges
//!   (a migrated base) is refused by name: listing them needs the code domain.
//!
//! REFUSED by name (`NonPorte`, listed in [`REFUSEES`]): adding or removing an
//! entity, the auto-build and its refresh (call-graph BFS), the top entry
//! functions, the statistics, the comparison and the overlap search. All read or
//! write the code graph.
//!
//! # Layout
//!
//! One node labelled `FeatureGraph`, with the properties Neo4j writes and its
//! LOSSY conventions, kept so that the two backends answer the same: an absent
//! `description` is stored `""` (read back `None`), an absent `entry_function`
//! is stored `""` (read back `Some("")`, as Neo4j does), an absent `build_depth`
//! is stored `0` (read back `Some(0)`), `include_relations` is a JSON string
//! (`""` when absent, read back `None`). Plus the edge
//! `(:FeatureGraph)-[:BELONGS_TO]->(:Project)`.
//!
//! # Divergences, all on the strict side
//!
//! - `create_feature_graph` for a project this backend does not hold is REFUSED;
//!   Neo4j creates the node without its `BELONGS_TO` edge.
//! - A second creation with an id already taken (or reserved by a quarantined
//!   node) is refused; Neo4j would create a second node.
//! - A node whose `created_at`/`updated_at` does not parse is quarantined at open;
//!   Neo4j reads it as "now".
//! - A `build_depth` above `i64::MAX` cannot be written (u32: cannot happen).
//!
//! - A deletion carries the marker `_suppression`, set first: a deletion
//!   interrupted after it is FINISHED at open (`RapportOuverture::suppressions_achevees`),
//!   never served as a graph with part of its edges gone.
//! - An out-edge the engine cannot read back makes the detail refuse (it may be an
//!   `INCLUDES_ENTITY`), as a persona's unreadable edge does.
//!
//! # Order and atomicity
//!
//! `list_feature_graphs` sorts by `updated_at DESC, id DESC` comparing the STORED
//! string, as Neo4j does (two RFC 3339 spellings with different fraction lengths
//! compare as strings, not as instants: Neo4j's order, kept). The
//! creation follows the draft-marker protocol of `project.rs`, and pays the FULL
//! flush before the marker is removed when it carries a string above
//! [`SEUIL_COLONNE_BLOB`](super::SEUIL_COLONNE_BLOB) (description, the
//! `include_relations` JSON), as `plan.rs` does until #1190 retires it; reads come from
//! the in-memory index, which a write publishes only once durable.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{EdgeId, NodeId, Value};
use obrain_core::graph::Direction;
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

pub(super) const LABEL: &str = "FeatureGraph";
const BROUILLON: &str = "_brouillon";
/// The deletion marker: set first by `delete_feature_graph`; a node that still
/// carries it at open has its deletion finished there.
const SUPPRESSION: &str = "_suppression";
const APPARTIENT: &str = "BELONGS_TO";
const INCLUT: &str = "INCLUDES_ENTITY";

/// The methods of `FeatureGraphStore` refused on every call, as
/// `FeatureGraphStore::<method>`: they read or write the code graph (files,
/// functions, structs, traits and their CALLS/IMPORTS edges), a domain not ported.
pub const REFUSEES: &[&str] = &[
    "FeatureGraphStore::add_entity_to_feature_graph",
    "FeatureGraphStore::remove_entity_from_feature_graph",
    "FeatureGraphStore::auto_build_feature_graph",
    "FeatureGraphStore::refresh_feature_graph",
    "FeatureGraphStore::get_top_entry_functions",
    "FeatureGraphStore::get_feature_graph_statistics",
    "FeatureGraphStore::compare_feature_graphs",
    "FeatureGraphStore::find_overlapping_feature_graphs",
];

/// Refuses a method of the feature-graph domain, by name, with its cause.
fn refus<T>(methode: &'static str) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).context(
        "lit ou écrit le graphe de code (fichiers, fonctions, structures, traits, arêtes \
         CALLS/IMPORTS) : domaine non porté sur obrain",
    ))
}

/// The properties a creation writes, in the order it writes them: those of
/// Neo4j's `CREATE`, with its conventions for the absent values.
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

/// Does this value go to a blob column (a string above the threshold)? Such a write
/// pays the full flush (`flush_complet`), as in `plan.rs` and `note.rs`.
fn va_en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
        || v.as_list().is_some_and(|l| {
            l.iter()
                .any(|e| e.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB))
        })
}

/// The domain value of a `FeatureGraph` node: the counterpart of
/// `Neo4jClient::node_to_feature_graph`, except that an unreadable timestamp is
/// an error (quarantine), not "now".
fn lire_graphe(get: &dyn Fn(&str) -> Option<Value>) -> Result<FeatureGraphNode> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let id: Uuid = chaine("id")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("nœud FeatureGraph sans propriété `id` lisible"))?;
    let requise = |cle: &str| {
        chaine(cle).ok_or_else(|| anyhow!("graphe de fonctionnalité {id} sans propriété `{cle}`"))
    };
    let instant = |cle: &str| -> Result<DateTime<Utc>> {
        let brut = requise(cle)?;
        brut.parse().map_err(|e| {
            anyhow!("graphe de fonctionnalité {id} : `{cle}` illisible (« {brut} ») : {e}")
        })
    };
    let project_id: Uuid = requise("project_id")?
        .parse()
        .map_err(|e| anyhow!("graphe de fonctionnalité {id} : `project_id` illisible : {e}"))?;
    Ok(FeatureGraphNode {
        id,
        name: requise("name")?,
        description: chaine("description").filter(|s| !s.is_empty()),
        project_id,
        created_at: instant("created_at")?,
        updated_at: instant("updated_at")?,
        entity_count: None,
        entry_function: chaine("entry_function"),
        build_depth: get("build_depth")
            .and_then(|v| v.as_int64())
            .filter(|v| *v >= 0)
            .and_then(|v| u32::try_from(v).ok()),
        include_relations: chaine("include_relations").and_then(|s| serde_json::from_str(&s).ok()),
    })
}

/// One feature graph: its node, its decoded value, and how many
/// `INCLUDES_ENTITY` edges leave it on disk.
#[derive(Clone)]
pub(super) struct EntreeGraphe {
    noeud: NodeId,
    graphe: FeatureGraphNode,
    entites: usize,
    /// Out-edges the engine could not read back (type unknown): they may be
    /// `INCLUDES_ENTITY`, so the detail refuses rather than answer "no entity".
    inconnues: usize,
    /// `updated_at` AS STORED: the sort key of `list_feature_graphs`, compared as
    /// the string Neo4j compares (taken from the earlier, uncommitted port).
    tri_maj: String,
}

impl EntreeGraphe {
    fn decoder(db: &ObrainDB, noeud: NodeId, entites: usize) -> Result<Self> {
        let n = db
            .get_node(noeud)
            .ok_or_else(|| anyhow!("nœud FeatureGraph {noeud:?} indexé mais absent de la base"))?;
        let get = |cle: &str| {
            n.get_property(cle)
                .filter(|v| !matches!(v, Value::Null))
                .cloned()
        };
        Ok(Self {
            noeud,
            graphe: lire_graphe(&get)?,
            entites,
            inconnues: 0,
            tri_maj: get("updated_at")
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default(),
        })
    }

    /// The value a listing answers: with its `entity_count`.
    fn compte(&self) -> FeatureGraphNode {
        let mut g = self.graphe.clone();
        g.entity_count = Some(self.entites as i64);
        g
    }
}

/// `Uuid → feature graph`, rebuilt from the `FeatureGraph` nodes at every open.
#[derive(Default)]
pub(super) struct IndexGraphes {
    par_id: HashMap<Uuid, EntreeGraphe>,
    /// Ids of nodes on disk that could not be read at open: never reused.
    reserves: HashSet<Uuid>,
}

impl IndexGraphes {
    /// The node of a feature graph this backend holds (for `SCOPED_TO`).
    pub(super) fn noeud_de(&self, id: Uuid) -> Option<NodeId> {
        self.par_id.get(&id).map(|e| e.noeud)
    }

    /// Rebuilds the index: drafts are deleted with their edges, what cannot be
    /// read is quarantined (named in the report, left untouched, its id reserved).
    pub(super) fn reconstruire(db: &ObrainDB, rapport: &mut RapportOuverture) -> Self {
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
            let suppression = get(SUPPRESSION).is_some();
            if suppression
                || get(BROUILLON).is_some()
                || (n.properties.is_empty() && store.out_degree(noeud) == 0)
            {
                for (_, arete) in store
                    .edges_from(noeud, Direction::Outgoing)
                    .into_iter()
                    .chain(store.edges_from(noeud, Direction::Incoming))
                {
                    db.delete_edge(arete);
                }
                db.delete_node(noeud);
                if suppression {
                    rapport.suppressions_achevees += 1;
                } else {
                    rapport.brouillons_supprimes += 1;
                }
                continue;
            }
            let graphe = match lire_graphe(&get) {
                Ok(g) => g,
                Err(e) => {
                    if let Some(id) =
                        get("id").and_then(|v| v.as_str().and_then(|s| s.parse().ok()))
                    {
                        index.reserves.insert(id);
                    }
                    rapport
                        .en_quarantaine
                        .push(format!("nœud FeatureGraph {noeud:?} : {e}"));
                    continue;
                }
            };
            if index.par_id.contains_key(&graphe.id) {
                rapport.en_quarantaine.push(format!(
                    "nœud FeatureGraph {noeud:?} : id {} déjà pris",
                    graphe.id
                ));
                continue;
            }
            // Walked, not `out_degree_by_type`: the engine's trait default answers 0
            // for a store without the per-type index, a silent wrong count.
            let (mut entites, mut inconnues) = (0, 0);
            for (_, a) in store.edges_from(noeud, Direction::Outgoing) {
                match db.get_edge(a) {
                    Some(e) if &*e.edge_type == INCLUT => entites += 1,
                    Some(_) => {}
                    None => inconnues += 1,
                }
            }
            let tri_maj = get("updated_at")
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            index.par_id.insert(
                graphe.id,
                EntreeGraphe {
                    noeud,
                    graphe,
                    entites,
                    inconnues,
                    tri_maj,
                },
            );
        }
        index
    }
}

#[async_trait]
impl FeatureGraphStore for ObrainGraphStore {
    async fn create_feature_graph(&self, graph: &FeatureGraphNode) -> Result<()> {
        let g = graph.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // Validate first: nothing is written when one of these refuses.
                let projet = {
                    let etat = w.index()?;
                    if etat.graphes.par_id.contains_key(&g.id) {
                        bail!(
                            "un graphe de fonctionnalité d'id {} existe déjà (unicité de \
                             FeatureGraph.id)",
                            g.id
                        );
                    }
                    if etat.graphes.reserves.contains(&g.id) {
                        bail!(
                            "l'id {} est réservé : un nœud FeatureGraph de cet id est sur disque, \
                             illisible à l'ouverture (quarantaine)",
                            g.id
                        );
                    }
                    etat.projets.noeud_de(g.project_id).ok_or_else(|| {
                        anyhow!(
                            "le projet {} n'existe pas : le graphe de fonctionnalité {} ne peut pas \
                             y être rattaché (le backend obrain ne crée pas de graphe orphelin)",
                            g.project_id,
                            g.id
                        )
                    })?
                };
                let proprietes = proprietes_de_creation(&g);
                w.flush_complet = proprietes.iter().any(|(_, v)| va_en_colonne_blob(v));
                let noeud = w.db.create_node(&[LABEL]);
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeNoeud)?;
                w.db.set_node_property(noeud, BROUILLON, Value::from(true));
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeMarqueur)?;
                for (_k, (cle, valeur)) in proprietes.into_iter().enumerate() {
                    w.db.set_node_property(noeud, cle, valeur);
                    #[cfg(any(test, feature = "mock-store"))]
                    w.point(Interruption::ApresLaPropriete(_k + 1))?;
                }
                w.db.create_edge(noeud, projet, APPARTIENT);
                // A blob-sized value is recovered only by the full flush: on disk
                // BEFORE the marker says the graph is complete.
                w.rendre_durable_avant_le_marqueur()?;
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::AvantLeRetraitDuMarqueur)?;
                // The commit point.
                w.db.remove_node_property(noeud, BROUILLON);
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeRetraitDuMarqueur)?;
                let entree = w.apres_ecriture(EntreeGraphe::decoder(w.db, noeud, 0))?;
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

    /// Served for a graph that includes nothing; refused by name otherwise.
    async fn get_feature_graph_detail(&self, id: Uuid) -> Result<Option<FeatureGraphDetail>> {
        self.lire(|_, etat| {
            let Some(e) = etat.graphes.par_id.get(&id) else {
                return Ok(None);
            };
            if e.entites + e.inconnues > 0 {
                return Err(anyhow::Error::new(NonPorte(
                    "FeatureGraphStore::get_feature_graph_detail",
                ))
                .context(format!(
                    "le graphe {id} inclut {} entité(s) (arêtes INCLUDES_ENTITY) et {} arête(s) \
                     illisible(s) : les décrire demande le domaine du code, non porté",
                    e.entites, e.inconnues
                )));
            }
            Ok(Some(FeatureGraphDetail {
                graph: e.graphe.clone(),
                entities: vec![],
                relations: vec![],
            }))
        })
    }

    async fn list_feature_graphs(&self, project_id: Option<Uuid>) -> Result<Vec<FeatureGraphNode>> {
        self.lire(|_, etat| {
            let mut entrees: Vec<&EntreeGraphe> = etat
                .graphes
                .par_id
                .values()
                .filter(|e| project_id.is_none_or(|p| e.graphe.project_id == p))
                .collect();
            // ORDER BY fg.updated_at DESC, fg.id DESC: Neo4j compares the stored
            // string, and so does this (a total order: the id breaks the ties).
            entrees.sort_by(|a, b| {
                b.tri_maj
                    .cmp(&a.tri_maj)
                    .then_with(|| b.graphe.id.cmp(&a.graphe.id))
            });
            Ok(entrees.into_iter().map(EntreeGraphe::compte).collect())
        })
    }

    async fn delete_feature_graph(&self, id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = w.index()?.graphes.noeud_de(id) else {
                    return Ok(false);
                };
                // The deletion marker first: from here on, a crash leaves a node that
                // opening finishes deleting, never a graph served with part of its
                // entities gone.
                w.db.set_node_property(noeud, SUPPRESSION, Value::from(true));
                // DETACH DELETE: every edge, in both directions (INCLUDES_ENTITY,
                // BELONGS_TO, the personas' SCOPED_TO).
                let aretes: Vec<EdgeId> =
                    w.db.store()
                        .edges_from(noeud, Direction::Outgoing)
                        .into_iter()
                        .chain(w.db.store().edges_from(noeud, Direction::Incoming))
                        .map(|(_, a)| a)
                        .collect();
                for a in aretes {
                    w.etape()?;
                    w.db.delete_edge(a);
                }
                w.etape()?;
                if !w.db.delete_node(noeud) {
                    // The edges are gone already: same contract as `apres_ecriture`.
                    w.interne
                        .indetermine
                        .store(true, std::sync::atomic::Ordering::Release);
                    bail!(
                        "obrain a refusé la suppression du nœud {noeud:?} (graphe de \
                         fonctionnalité {id}) APRÈS la suppression de ses arêtes : base \
                         indéterminée, redémarrer"
                    );
                }
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLaSuppression)?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.graphes.par_id.remove(&id);
                etat.personas.oublier_graphe(id);
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
        refus("FeatureGraphStore::add_entity_to_feature_graph")
    }

    async fn remove_entity_from_feature_graph(
        &self,
        _feature_graph_id: Uuid,
        _entity_type: &str,
        _entity_id: &str,
    ) -> Result<bool> {
        refus("FeatureGraphStore::remove_entity_from_feature_graph")
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
        refus("FeatureGraphStore::auto_build_feature_graph")
    }

    async fn refresh_feature_graph(&self, _id: Uuid) -> Result<Option<FeatureGraphDetail>> {
        refus("FeatureGraphStore::refresh_feature_graph")
    }

    async fn get_top_entry_functions(
        &self,
        _project_id: Uuid,
        _limit: usize,
    ) -> Result<Vec<String>> {
        refus("FeatureGraphStore::get_top_entry_functions")
    }

    async fn get_feature_graph_statistics(
        &self,
        _id: Uuid,
    ) -> Result<Option<FeatureGraphStatistics>> {
        refus("FeatureGraphStore::get_feature_graph_statistics")
    }

    async fn compare_feature_graphs(
        &self,
        _id_a: Uuid,
        _id_b: Uuid,
    ) -> Result<Option<FeatureGraphComparison>> {
        refus("FeatureGraphStore::compare_feature_graphs")
    }

    async fn find_overlapping_feature_graphs(
        &self,
        _id: Uuid,
        _min_overlap: f64,
    ) -> Result<Vec<FeatureGraphOverlap>> {
        refus("FeatureGraphStore::find_overlapping_feature_graphs")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::models::{ProjectNode, ProjectProfile};
    use crate::neo4j::traits::ProjectStore;
    use crate::obrain::Creation;
    use chrono::TimeZone;

    fn projet() -> ProjectNode {
        let id = Uuid::new_v4();
        ProjectNode {
            id,
            name: "fg".into(),
            slug: format!("fg-{id}"),
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

    fn graphe(projet: Uuid, nom: &str, decalage: i64) -> FeatureGraphNode {
        let t = Utc.timestamp_opt(1_700_000_000 + decalage, 0).unwrap();
        FeatureGraphNode {
            id: Uuid::new_v4(),
            name: nom.into(),
            description: None,
            project_id: projet,
            created_at: t,
            updated_at: t,
            entity_count: None,
            entry_function: None,
            build_depth: None,
            include_relations: None,
        }
    }

    #[tokio::test]
    async fn cycle_de_vie_et_conventions_de_neo4j() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let p = projet();
        store.create_project(&p).await.unwrap();
        let a = graphe(p.id, "a", 0);
        let mut b = graphe(p.id, "b", 10);
        b.entry_function = Some("main".into());
        b.build_depth = Some(3);
        b.include_relations = Some(vec!["CALLS".into()]);
        b.description = Some("d".into());
        store.create_feature_graph(&a).await.unwrap();
        store.create_feature_graph(&b).await.unwrap();
        assert!(store.create_feature_graph(&a).await.is_err(), "id unique");

        // Les conventions de Neo4j pour un graphe manuel.
        let relu = store.get_feature_graph(a.id).await.unwrap().unwrap();
        assert_eq!(relu.entry_function.as_deref(), Some(""));
        assert_eq!(relu.build_depth, Some(0));
        assert_eq!(relu.include_relations, None);
        assert_eq!(relu.description, None);
        let relu = store.get_feature_graph(b.id).await.unwrap().unwrap();
        assert_eq!(relu.entry_function.as_deref(), Some("main"));
        assert_eq!(relu.build_depth, Some(3));
        assert_eq!(relu.include_relations, Some(vec!["CALLS".to_string()]));

        let liste = store.list_feature_graphs(Some(p.id)).await.unwrap();
        assert_eq!(
            liste.iter().map(|g| g.id).collect::<Vec<_>>(),
            vec![b.id, a.id]
        );
        assert!(liste.iter().all(|g| g.entity_count == Some(0)));
        let detail = store.get_feature_graph_detail(a.id).await.unwrap().unwrap();
        assert!(detail.entities.is_empty() && detail.relations.is_empty());
        assert!(store.delete_feature_graph(a.id).await.unwrap());
        assert!(!store.delete_feature_graph(a.id).await.unwrap());
        assert!(store.get_feature_graph(a.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn un_projet_absent_est_refuse() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let g = graphe(Uuid::new_v4(), "orphelin", 0);
        assert!(store.create_feature_graph(&g).await.is_err());
        assert!(store.get_feature_graph(g.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn survit_a_la_reouverture() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = projet();
        let g = graphe(p.id, "durable", 0);
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_project(&p).await.unwrap();
            store.create_feature_graph(&g).await.unwrap();
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        let relu = store.get_feature_graph(g.id).await.unwrap().unwrap();
        assert_eq!(relu.name, "durable");
        assert_eq!(relu.created_at, g.created_at);
    }
}
