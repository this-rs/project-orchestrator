//! `PersonaStore` on obrain: the persona node and its edges to the domains this
//! backend holds (lot O-PERSONA-1).
//!
//! PORTED, with the semantics of `src/neo4j/persona.rs`:
//! - the node: `create_persona`, `get_persona`, `update_persona`,
//!   `delete_persona` (DETACH), `list_personas`, `list_global_personas`,
//!   `increment_persona_activation`;
//! - the edges to ported domains: `SCOPED_TO` a feature graph
//!   (`set_persona_feature_graph`, `remove_persona_feature_graph`, see
//!   `featuregraph.rs`), `USES` a note (`add_persona_note`, `remove_persona_note`,
//!   `auto_link_note_to_persona`), `EXTENDS` a persona (`add_persona_extends`,
//!   `remove_persona_extends`);
//! - `get_persona_subgraph`, for a persona whose every out-edge this backend
//!   resolves. A persona that also KNOWS a file or a function, MASTERS a skill,
//!   FOLLOWS a protocol or USES a decision (a migrated base), or whose edge is
//!   unreadable, is refused by name: its subgraph would be silently short.
//!
//! REFUSED by name ([`REFUSEES`]), with their cause:
//! - `MASTERS` skill, `FOLLOWS` protocol, `USES` decision, `KNOWS` file/function
//!   (add, remove, auto-link): their domains are not ported;
//! - every matching that reads `KNOWS` files, the import graph or the Louvain
//!   communities (`find_personas_for_file`, `get_all_persona_knows`,
//!   `find_adjacent_personas`, `find_relevant_personas_for_note`,
//!   `find_relevant_personas_for_decision`, `detect_personas`), the maintenance
//!   (decay and pruning of `KNOWS`), the affinity (Jaccard on `KNOWS` files and
//!   note `SYNAPSE` density), the merge (transfers `KNOWS`), and
//!   `find_synapse_linked_personas` (note synapses, not ported).
//!
//! # Layout
//!
//! One node labelled `Persona`, with the properties Neo4j's `CREATE` writes (an
//! absent option is an absent property; `energy_history` is a JSON string;
//! `status` and `origin` are their `Display` spellings), plus the edges
//! `(:Persona)-[:BELONGS_TO]->(:Project)`, `-[:USES {weight}]->(:Note)`,
//! `-[:EXTENDS]->(:Persona)`, `-[:SCOPED_TO]->(:FeatureGraph)`. As in Neo4j, the
//! creation writes neither `last_activated` nor `updated_at`, and the update
//! writes neither `origin`, `project_id`, `energy_boost_accumulated` nor
//! `energy_history`.
//!
//! # Divergences, all on the strict side or named
//!
//! - `create_persona` for a project this backend does not hold is REFUSED (Neo4j
//!   creates a global persona instead); an id already taken, or reserved by a
//!   quarantined node, is refused; a non-finite `energy` (the sort key) and a
//!   `timeout_secs` above `i64::MAX` are refused (Neo4j stores the first and
//!   wraps the second).
//! - `update_persona` of an absent persona is an error (Neo4j: silent no-op; the
//!   mock: an error). It refuses a non-finite `energy` and a `timeout_secs` above
//!   `i64::MAX` too, as the creation does.
//! - `increment_persona_activation` returns a storage failure (Neo4j swallows it);
//!   an absent persona stays a no-op. It writes `last_activated`/`updated_at` as
//!   RFC 3339 strings, so they read back; Neo4j writes a temporal its own decoder
//!   drops.
//! - The edge writes (`set_persona_feature_graph`, `add_persona_note`,
//!   `auto_link_note_to_persona`, `add_persona_extends`) refuse when either end is
//!   absent, or for a non-finite weight; Neo4j's `MATCH` makes them silent no-ops
//!   (and `set_persona_feature_graph` to an absent graph deletes the old scope).
//! - A node whose `id`, `name` or `created_at` cannot be read, whose `status` is
//!   absent or unknown, whose `energy` is absent, or whose `project_id` is present
//!   but not a UUID, is quarantined at open; its id is reserved. Neo4j decodes such
//!   a node with defaults (`emerging`, 0.5, global) but filters, sorts (`null`
//!   first in DESC) and lists it on the stored value: a default here would answer
//!   differently.
//! - `get_persona_subgraph` also refuses when an ANCESTOR it walks has an
//!   `EXTENDS` this backend does not resolve, or when a node it does not index (a
//!   quarantined persona) EXTENDS the persona: the parents or children would be
//!   short.
//! - `remove_persona_note` / `remove_persona_extends` find the edge by the target's
//!   `id`, readable or quarantined, as Neo4j's `MATCH` does: the edge to a
//!   quarantined target is removed, not silently kept.
//! - `set_persona_feature_graph` is two writes (old scope deleted, then new edge):
//!   a crash between them leaves the persona unscoped, never scoped twice.
//! - `children` of the subgraph are sorted by id (Neo4j: no order).
//!
//! # Atomicity
//!
//! The creation follows the draft-marker protocol of `project.rs`. A `USES` edge
//! is created with its `weight` (two WAL records): an edge read back without a
//! weight is a torn write, or a foreign edge; it is named in the open report,
//! makes the subgraph refuse, and the next `add_persona_note` of that pair
//! repairs it (the writers look the edge up on disk, not in the index). A deletion
//! sets the marker `_suppression` first, then removes the edges and the node: a
//! deletion interrupted after the marker is FINISHED at open
//! (`RapportOuverture::suppressions_achevees`), never served as a persona with
//! part of its edges gone. A write that carries a string above
//! [`SEUIL_COLONNE_BLOB`](super::SEUIL_COLONNE_BLOB) (a description, a system
//! prompt, the `energy_history` JSON), or that replaces or removes one, pays the
//! FULL flush, before the marker is removed for a creation (the pattern of
//! `plan.rs` and `note.rs`, until #1190 retires it). Reads come from the
//! in-memory index, published only once a write is durable.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{EdgeId, NodeId, Value};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use uuid::Uuid;

#[cfg(any(test, feature = "mock-store"))]
use super::Interruption;
use super::{
    featuregraph, note, Ecriture, NonPorte, ObrainGraphStore, RapportOuverture, SEUIL_COLONNE_BLOB,
};
use crate::neo4j::models::{
    PersonaNode, PersonaStatus, PersonaSubgraph, PersonaSubgraphStats, PersonaWeightedRelation,
};
use crate::neo4j::persona::{PersonaAffinityScore, PersonaProposal};
use crate::neo4j::traits::PersonaStore;

const LABEL: &str = "Persona";
const BROUILLON: &str = "_brouillon";
/// The deletion marker: set first by `delete_persona`; a node that still carries it
/// at open has its deletion finished there.
const SUPPRESSION: &str = "_suppression";
const APPARTIENT: &str = "BELONGS_TO";
const UTILISE: &str = "USES";
const ETEND: &str = "EXTENDS";
const PORTEE: &str = "SCOPED_TO";
/// Depth of the `EXTENDS*1..10` walk of Neo4j's subgraph.
const PROFONDEUR_HERITAGE: usize = 10;

/// The methods of `PersonaStore` refused on every call, as `PersonaStore::<method>`.
pub const REFUSEES: &[&str] = &[
    "PersonaStore::add_persona_skill",
    "PersonaStore::remove_persona_skill",
    "PersonaStore::add_persona_protocol",
    "PersonaStore::remove_persona_protocol",
    "PersonaStore::add_persona_file",
    "PersonaStore::remove_persona_file",
    "PersonaStore::add_persona_function",
    "PersonaStore::remove_persona_function",
    "PersonaStore::add_persona_decision",
    "PersonaStore::remove_persona_decision",
    "PersonaStore::find_personas_for_file",
    "PersonaStore::get_all_persona_knows",
    "PersonaStore::maintain_personas",
    "PersonaStore::detect_personas",
    "PersonaStore::find_adjacent_personas",
    "PersonaStore::find_relevant_personas_for_note",
    "PersonaStore::find_relevant_personas_for_decision",
    "PersonaStore::auto_link_decision_to_persona",
    "PersonaStore::auto_link_file_to_persona",
    "PersonaStore::compute_persona_affinity",
    "PersonaStore::merge_personas",
    "PersonaStore::find_synapse_linked_personas",
];

const CAUSE_SKILL: &str = "relation MASTERS vers une compétence : domaine Skill non porté";
const CAUSE_PROTOCOLE: &str = "relation FOLLOWS vers un protocole : domaine Protocol non porté";
const CAUSE_DECISION: &str = "relation USES vers une décision : domaine Decision non porté";
const CAUSE_CODE: &str =
    "relation KNOWS vers un fichier ou une fonction : domaine du code non porté";
const CAUSE_SYNAPSE: &str = "lit les SYNAPSE entre notes, non portées sur obrain";

/// A refusal, by name, with its cause.
fn refus<T>(methode: &'static str, cause: &'static str) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).context(cause))
}

/// The properties a creation writes, in the order it writes them: those of
/// Neo4j's `CREATE` (an absent option is not written).
fn proprietes_de_creation(p: &PersonaNode) -> Result<Vec<(&'static str, Value)>> {
    if !p.energy.is_finite() {
        bail!(
            "`energy` non fini ({}) refusé : c'est la clé de tri des listes. Neo4j l'accepterait ; \
             divergence assumée",
            p.energy
        );
    }
    let mut v: Vec<(&'static str, Value)> = vec![
        ("id", Value::from(p.id.to_string())),
        ("name", Value::from(p.name.clone())),
        ("description", Value::from(p.description.clone())),
        ("status", Value::from(p.status.to_string())),
    ];
    v.extend(
        proprietes_optionnelles(p)?
            .into_iter()
            .flat_map(|(k, x)| x.map(|x| (k, x))),
    );
    v.extend([
        ("energy", Value::from(p.energy)),
        ("cohesion", Value::from(p.cohesion)),
        ("activation_count", Value::from(p.activation_count)),
        ("success_rate", Value::from(p.success_rate)),
        ("avg_duration_secs", Value::from(p.avg_duration_secs)),
        ("origin", Value::from(p.origin.to_string())),
        (
            "energy_boost_accumulated",
            Value::from(p.energy_boost_accumulated),
        ),
        (
            "energy_history",
            Value::from(serde_json::to_string(&p.energy_history)?),
        ),
        ("created_at", Value::from(p.created_at.to_rfc3339())),
    ]);
    if let Some(pid) = p.project_id {
        v.push(("project_id", Value::from(pid.to_string())));
    }
    Ok(v)
}

/// The optional properties both the creation and the update write, `None` when
/// the field is absent (the update then removes the property, as Neo4j's
/// `SET p.x = null` does).
fn proprietes_optionnelles(p: &PersonaNode) -> Result<Vec<(&'static str, Option<Value>)>> {
    let timeout = match p.timeout_secs {
        Some(t) => Some(Value::from(i64::try_from(t).map_err(|_| {
            anyhow!(
                "`timeout_secs` = {t} dépasse i64::MAX : refusé (Neo4j l'enregistrerait négatif)"
            )
        })?)),
        None => None,
    };
    Ok(vec![
        (
            "complexity_default",
            p.complexity_default.clone().map(Value::from),
        ),
        ("timeout_secs", timeout),
        ("max_cost_usd", p.max_cost_usd.map(Value::from)),
        (
            "model_preference",
            p.model_preference.clone().map(Value::from),
        ),
        (
            "system_prompt_override",
            p.system_prompt_override.clone().map(Value::from),
        ),
    ])
}

/// The domain value of a `Persona` node: the counterpart of
/// `Neo4jClient::node_to_persona` (same defaults for the optional fields). `id`,
/// `name`, `created_at`, a known `status` and a numeric `energy` are REQUIRED, and
/// a present `project_id` must be a UUID: Neo4j decodes them with defaults but
/// filters, sorts and lists on the STORED value, so a default here would answer
/// differently. Such a node is quarantined.
fn lire_persona(get: &dyn Fn(&str) -> Option<Value>) -> Result<PersonaNode> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let non_vide = |cle: &str| chaine(cle).filter(|s| !s.is_empty());
    let reel = |cle: &str| get(cle).and_then(|v| v.as_float64());
    let id: Uuid = chaine("id")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("nœud Persona sans propriété `id` lisible"))?;
    let name = chaine("name").ok_or_else(|| anyhow!("persona {id} sans propriété `name`"))?;
    let brut = chaine("created_at").ok_or_else(|| anyhow!("persona {id} sans `created_at`"))?;
    let created_at: DateTime<Utc> = brut
        .parse()
        .map_err(|e| anyhow!("persona {id} : `created_at` illisible (« {brut} ») : {e}"))?;
    let status_brut = chaine("status").ok_or_else(|| anyhow!("persona {id} sans `status`"))?;
    let status: PersonaStatus = status_brut
        .parse()
        .map_err(|_| anyhow!("persona {id} : `status` inconnu (« {status_brut} »)"))?;
    let energy = reel("energy").ok_or_else(|| anyhow!("persona {id} sans `energy` numérique"))?;
    let project_id = match chaine("project_id") {
        None => None,
        Some(s) => Some(
            s.parse::<Uuid>()
                .map_err(|_| anyhow!("persona {id} : `project_id` illisible (« {s} »)"))?,
        ),
    };
    Ok(PersonaNode {
        id,
        project_id,
        name,
        description: chaine("description").unwrap_or_default(),
        status,
        complexity_default: non_vide("complexity_default"),
        timeout_secs: get("timeout_secs")
            .and_then(|v| v.as_int64())
            .filter(|v| *v >= 0)
            .map(|v| v as u64),
        max_cost_usd: reel("max_cost_usd"),
        model_preference: non_vide("model_preference"),
        system_prompt_override: non_vide("system_prompt_override"),
        energy,
        cohesion: reel("cohesion").unwrap_or(0.0),
        activation_count: get("activation_count")
            .and_then(|v| v.as_int64())
            .unwrap_or(0),
        success_rate: reel("success_rate").unwrap_or(0.0),
        avg_duration_secs: reel("avg_duration_secs").unwrap_or(0.0),
        last_activated: chaine("last_activated").and_then(|s| s.parse().ok()),
        energy_boost_accumulated: reel("energy_boost_accumulated").unwrap_or(0.0),
        energy_history: chaine("energy_history")
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default(),
        origin: chaine("origin")
            .and_then(|s| s.parse().ok())
            .unwrap_or_default(),
        created_at,
        updated_at: chaine("updated_at").and_then(|s| s.parse().ok()),
    })
}

/// The `id` property of a node, as a `Uuid`.
fn id_du_noeud(db: &ObrainDB, n: NodeId) -> Option<Uuid> {
    db.get_node(n)?.get_property("id")?.as_str()?.parse().ok()
}

/// The out-edges of `de` of type `type_` that reach `vers`, read from the ENGINE
/// (the writers' truth: a torn edge the index ignores is found here).
fn aretes_vers(db: &ObrainDB, de: NodeId, vers: Option<NodeId>, type_: &str) -> Vec<EdgeId> {
    db.store()
        .edges_from(de, Direction::Outgoing)
        .into_iter()
        .filter(|(cible, a)| {
            vers.is_none_or(|v| v == *cible)
                && db.get_edge(*a).is_some_and(|e| &*e.edge_type == type_)
        })
        .map(|(_, a)| a)
        .collect()
}

/// The out-edges of `de` of type `type_` whose target's `id` is `vers`, whatever
/// that target is (indexed or quarantined): Neo4j's `MATCH (p)-[r]->({id})` does not
/// depend on the target being readable.
fn aretes_vers_id(db: &ObrainDB, de: NodeId, vers: Uuid, type_: &str) -> Vec<EdgeId> {
    db.store()
        .edges_from(de, Direction::Outgoing)
        .into_iter()
        .filter(|(cible, a)| {
            id_du_noeud(db, *cible) == Some(vers)
                && db.get_edge(*a).is_some_and(|e| &*e.edge_type == type_)
        })
        .map(|(_, a)| a)
        .collect()
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

/// A finite weight, or a refusal before any write.
fn poids_fini(poids: f64) -> Result<()> {
    if !poids.is_finite() {
        bail!(
            "poids non fini ({poids}) refusé : c'est une clé de tri du sous-graphe. Neo4j \
             l'accepterait ; divergence assumée"
        );
    }
    Ok(())
}

/// One persona: its node, its decoded value, and its out-edges.
#[derive(Clone)]
pub(super) struct EntreePersona {
    noeud: NodeId,
    persona: PersonaNode,
    /// `USES` → note: the edge and its weight.
    notes: BTreeMap<Uuid, (EdgeId, f64)>,
    /// `EXTENDS` → parent persona.
    parents: BTreeMap<Uuid, EdgeId>,
    /// `SCOPED_TO` → feature graph (at most one once this backend has written it).
    graphes: BTreeMap<Uuid, EdgeId>,
    /// Out-edges this backend does not resolve: a domain not ported (KNOWS,
    /// MASTERS, FOLLOWS, USES → decision…), a missing or quarantined target, or an
    /// unreadable weight. Their presence makes the subgraph refuse by name. The
    /// flag says the edge is (or may be, when unreadable) an `EXTENDS`: on an
    /// ANCESTOR, it makes the subgraph of its descendants refuse too.
    etrangeres: BTreeMap<EdgeId, bool>,
    /// Incoming `EXTENDS` from a node this backend does not index (a quarantined
    /// persona): the children list would omit it, so the subgraph refuses.
    enfants_inconnus: BTreeSet<EdgeId>,
}

impl EntreePersona {
    fn decoder(db: &ObrainDB, noeud: NodeId, modele: Option<&EntreePersona>) -> Result<Self> {
        let n = db
            .get_node(noeud)
            .ok_or_else(|| anyhow!("nœud Persona {noeud:?} indexé mais absent de la base"))?;
        let get = |cle: &str| {
            n.get_property(cle)
                .filter(|v| !matches!(v, Value::Null))
                .cloned()
        };
        let persona = lire_persona(&get)?;
        Ok(match modele {
            Some(m) => Self {
                noeud,
                persona,
                ..m.clone()
            },
            None => Self {
                noeud,
                persona,
                notes: BTreeMap::new(),
                parents: BTreeMap::new(),
                graphes: BTreeMap::new(),
                etrangeres: BTreeMap::new(),
                enfants_inconnus: BTreeSet::new(),
            },
        })
    }
}

/// `Uuid → persona`, rebuilt from the `Persona` nodes at every open.
#[derive(Default)]
pub(super) struct IndexPersonas {
    par_id: HashMap<Uuid, EntreePersona>,
    /// Ids of nodes on disk that could not be read at open: never reused.
    reserves: HashSet<Uuid>,
}

impl IndexPersonas {
    /// A note was deleted (DETACH): the personas lost their `USES` edge to it.
    pub(super) fn oublier_note(&mut self, note: Uuid) {
        for e in self.par_id.values_mut() {
            e.notes.remove(&note);
        }
    }

    /// A feature graph was deleted (DETACH): the personas lost their scope.
    pub(super) fn oublier_graphe(&mut self, graphe: Uuid) {
        for e in self.par_id.values_mut() {
            e.graphes.remove(&graphe);
        }
    }

    /// Rebuilds the index, after the notes and the feature graphs: drafts are
    /// deleted with their edges, unreadable nodes quarantined (their id reserved),
    /// and the out-edges resolved against the indexes of the ported domains.
    pub(super) fn reconstruire(
        db: &ObrainDB,
        notes: &note::IndexNotes,
        graphes: &featuregraph::IndexGraphes,
        rapport: &mut RapportOuverture,
    ) -> Self {
        let store = db.store();
        let mut index = Self::default();
        let mut noeuds: HashMap<NodeId, Uuid> = HashMap::new();
        for noeud in store.nodes_by_label(LABEL) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud Persona {noeud:?} : illisible"));
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
            let persona = match lire_persona(&get) {
                Ok(p) => p,
                Err(e) => {
                    if let Some(id) =
                        get("id").and_then(|v| v.as_str().and_then(|s| s.parse().ok()))
                    {
                        index.reserves.insert(id);
                    }
                    rapport
                        .en_quarantaine
                        .push(format!("nœud Persona {noeud:?} : {e}"));
                    continue;
                }
            };
            if index.par_id.contains_key(&persona.id) {
                rapport.en_quarantaine.push(format!(
                    "nœud Persona {noeud:?} : id {} déjà pris",
                    persona.id
                ));
                continue;
            }
            noeuds.insert(noeud, persona.id);
            index.par_id.insert(
                persona.id,
                EntreePersona {
                    noeud,
                    persona,
                    notes: BTreeMap::new(),
                    parents: BTreeMap::new(),
                    graphes: BTreeMap::new(),
                    etrangeres: BTreeMap::new(),
                    enfants_inconnus: BTreeSet::new(),
                },
            );
        }

        // The out-edges, once every persona is known (EXTENDS targets a persona).
        for (noeud, id) in &noeuds {
            let Some(e) = index.par_id.get_mut(id) else {
                continue;
            };
            for (cible, arete) in store.edges_from(*noeud, Direction::Outgoing) {
                let Some(a) = db.get_edge(arete) else {
                    e.etrangeres.insert(arete, true);
                    continue;
                };
                let type_ = &*a.edge_type;
                let cible_id = id_du_noeud(db, cible);
                let resolu = match type_ {
                    APPARTIENT => true,
                    UTILISE => match cible_id.filter(|u| notes.noeud_de(*u) == Some(cible)) {
                        Some(n) => match a.get_property("weight").and_then(|v| v.as_float64()) {
                            Some(w) if !e.notes.contains_key(&n) => {
                                e.notes.insert(n, (arete, w));
                                true
                            }
                            Some(_) => false,
                            None => {
                                rapport.en_quarantaine.push(format!(
                                    "arête USES de la persona {id} vers la note {n} sans poids \
                                     lisible : ignorée (sous-graphe refusé jusqu'à sa réécriture)"
                                ));
                                false
                            }
                        },
                        None => false,
                    },
                    ETEND => match cible_id.filter(|u| noeuds.get(&cible) == Some(u)) {
                        Some(p) if !e.parents.contains_key(&p) => {
                            e.parents.insert(p, arete);
                            true
                        }
                        _ => false,
                    },
                    PORTEE => match cible_id.filter(|u| graphes.noeud_de(*u) == Some(cible)) {
                        Some(g) if !e.graphes.contains_key(&g) => {
                            e.graphes.insert(g, arete);
                            true
                        }
                        _ => false,
                    },
                    _ => false,
                };
                if !resolu {
                    e.etrangeres.insert(arete, type_ == ETEND);
                }
            }
            // Incoming EXTENDS from a node this backend does not index.
            for (source, arete) in store.edges_from(*noeud, Direction::Incoming) {
                let etend = db.get_edge(arete).is_none_or(|a| &*a.edge_type == ETEND);
                if etend && !noeuds.contains_key(&source) {
                    e.enfants_inconnus.insert(arete);
                }
            }
        }
        index
    }
}

impl ObrainGraphStore {
    /// The persona entry of `id`, cloned (a write publishes a new state).
    fn entree_persona(w: &Ecriture<'_>, id: Uuid) -> Result<Option<EntreePersona>> {
        Ok(w.index()?.personas.par_id.get(&id).cloned())
    }

    /// A `USES` edge persona → note with `poids`: created, or its weight
    /// rewritten (`max` keeps the larger of the two, as the auto-link does).
    async fn ecrire_utilise(
        &self,
        persona_id: Uuid,
        note_id: Uuid,
        poids: f64,
        max: bool,
    ) -> Result<()> {
        poids_fini(poids)?;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (mut e, cible) = {
                    let etat = w.index()?;
                    let e = etat
                        .personas
                        .par_id
                        .get(&persona_id)
                        .cloned()
                        .ok_or_else(|| anyhow!("persona {persona_id} introuvable"))?;
                    let cible = etat.notes.noeud_de(note_id).ok_or_else(|| {
                        anyhow!("note {note_id} introuvable : l'arête USES n'est pas écrite")
                    })?;
                    (e, cible)
                };
                let existantes = aretes_vers(w.db, e.noeud, Some(cible), UTILISE);
                let arete = match existantes.first() {
                    Some(a) => {
                        let actuel = w
                            .db
                            .get_edge(*a)
                            .and_then(|x| x.get_property("weight").and_then(|v| v.as_float64()));
                        let nouveau = match actuel {
                            Some(v) if max && v >= poids => v,
                            _ => poids,
                        };
                        w.db.set_edge_property(*a, "weight", Value::from(nouveau));
                        e.notes.insert(note_id, (*a, nouveau));
                        *a
                    }
                    None => {
                        let a = w.db.create_edge_with_props(
                            e.noeud,
                            cible,
                            UTILISE,
                            [("weight", Value::from(poids))],
                        );
                        e.notes.insert(note_id, (a, poids));
                        a
                    }
                };
                e.etrangeres.remove(&arete);
                w.rendre_durable()?;
                w.index_mut()?.personas.par_id.insert(persona_id, e);
                Ok(())
            },
        )
        .await
    }

    /// Delete every out-edge of `type_` from a persona to `vers` (all of them
    /// when `None`), and publish. An absent persona is a no-op, as Neo4j.
    async fn retirer_aretes(
        &self,
        persona_id: Uuid,
        type_: &'static str,
        vers: Option<Uuid>,
    ) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(mut e) = Self::entree_persona(w, persona_id)? else {
                    return Ok(());
                };
                // By the target's id, readable or quarantined (see `aretes_vers_id`).
                let aretes = match vers {
                    None => aretes_vers(w.db, e.noeud, None, type_),
                    Some(id) => aretes_vers_id(w.db, e.noeud, id, type_),
                };
                if aretes.is_empty() {
                    return Ok(());
                }
                for a in &aretes {
                    w.etape()?;
                    w.db.delete_edge(*a);
                    e.etrangeres.remove(a);
                }
                match (type_, vers) {
                    (UTILISE, Some(n)) => {
                        e.notes.remove(&n);
                    }
                    (ETEND, Some(p)) => {
                        e.parents.remove(&p);
                    }
                    (PORTEE, None) => e.graphes.clear(),
                    _ => {}
                }
                w.rendre_durable()?;
                w.index_mut()?.personas.par_id.insert(persona_id, e);
                Ok(())
            },
        )
        .await
    }
}

/// `ORDER BY p.energy DESC, p.id DESC`.
fn ordre_de_liste(a: &PersonaNode, b: &PersonaNode) -> std::cmp::Ordering {
    b.energy.total_cmp(&a.energy).then_with(|| b.id.cmp(&a.id))
}

// The refusals keep their parameters' names for the reader; they are unused by design.
#[allow(unused_variables)]
#[async_trait]
impl PersonaStore for ObrainGraphStore {
    async fn create_persona(&self, persona: &PersonaNode) -> Result<()> {
        let p = persona.clone();
        let proprietes = proprietes_de_creation(&p)?;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // Validate first: nothing is written when one of these refuses.
                let projet = {
                    let etat = w.index()?;
                    if etat.personas.par_id.contains_key(&p.id) {
                        bail!(
                            "une persona d'id {} existe déjà (unicité de Persona.id)",
                            p.id
                        );
                    }
                    if etat.personas.reserves.contains(&p.id) {
                        bail!(
                            "l'id {} est réservé : un nœud Persona de cet id est sur disque, \
                             illisible à l'ouverture (quarantaine)",
                            p.id
                        );
                    }
                    match p.project_id {
                        Some(pid) => Some(etat.projets.noeud_de(pid).ok_or_else(|| {
                            anyhow!(
                                "le projet {pid} n'existe pas : la persona {} ne peut pas y être \
                                 rattachée (Neo4j en ferait une persona globale en silence)",
                                p.id
                            )
                        })?),
                        None => None,
                    }
                };
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
                if let Some(projet) = projet {
                    w.db.create_edge(noeud, projet, APPARTIENT);
                }
                // A blob-sized value is recovered only by the full flush: on disk
                // BEFORE the marker says the persona is complete.
                w.rendre_durable_avant_le_marqueur()?;
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::AvantLeRetraitDuMarqueur)?;
                // The commit point.
                w.db.remove_node_property(noeud, BROUILLON);
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeRetraitDuMarqueur)?;
                let entree = w.apres_ecriture(EntreePersona::decoder(w.db, noeud, None))?;
                w.rendre_durable()?;
                w.index_mut()?.personas.par_id.insert(p.id, entree);
                Ok(())
            },
        )
        .await
    }

    async fn get_persona(&self, id: Uuid) -> Result<Option<PersonaNode>> {
        self.lire(|_, etat| Ok(etat.personas.par_id.get(&id).map(|e| e.persona.clone())))
    }

    async fn update_persona(&self, persona: &PersonaNode) -> Result<()> {
        if !persona.energy.is_finite() {
            bail!(
                "`energy` non fini ({}) refusé : c'est la clé de tri des listes",
                persona.energy
            );
        }
        let p = persona.clone();
        let optionnelles = proprietes_optionnelles(&p)?;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let e = Self::entree_persona(w, p.id)?.ok_or_else(|| {
                    anyhow!(
                        "persona {} introuvable : rien n'est mis à jour (Neo4j l'ignorerait en \
                         silence)",
                        p.id
                    )
                })?;
                let noeud = e.noeud;
                let mut fixes: Vec<(&str, Value)> = vec![
                    ("name", Value::from(p.name.clone())),
                    ("description", Value::from(p.description.clone())),
                    ("status", Value::from(p.status.to_string())),
                    ("energy", Value::from(p.energy)),
                    ("cohesion", Value::from(p.cohesion)),
                    ("activation_count", Value::from(p.activation_count)),
                    ("success_rate", Value::from(p.success_rate)),
                    ("avg_duration_secs", Value::from(p.avg_duration_secs)),
                ];
                let mut a_retirer: Vec<&str> = Vec::new();
                for (cle, valeur) in optionnelles {
                    match valeur {
                        Some(v) => fixes.push((cle, v)),
                        None => a_retirer.push(cle),
                    }
                }
                match p.last_activated {
                    Some(d) => fixes.push(("last_activated", Value::from(d.to_rfc3339()))),
                    None => a_retirer.push("last_activated"),
                }
                fixes.push(("updated_at", Value::from(Utc::now().to_rfc3339())));
                // The full flush when a NEW value goes to a blob column, or when an OLD
                // one (replaced or removed) is there.
                let ancienne = &e.persona;
                let ancien_en_blob = [
                    Some(ancienne.name.as_str()),
                    Some(ancienne.description.as_str()),
                    ancienne.complexity_default.as_deref(),
                    ancienne.model_preference.as_deref(),
                    ancienne.system_prompt_override.as_deref(),
                ]
                .into_iter()
                .flatten()
                .any(|s| s.len() > SEUIL_COLONNE_BLOB);
                w.flush_complet =
                    ancien_en_blob || fixes.iter().any(|(_, v)| va_en_colonne_blob(v));
                for (cle, valeur) in fixes {
                    w.etape()?;
                    w.db.set_node_property(noeud, cle, valeur);
                }
                for cle in a_retirer {
                    w.etape()?;
                    w.db.remove_node_property(noeud, cle);
                }
                let entree = w.apres_ecriture(EntreePersona::decoder(w.db, noeud, Some(&e)))?;
                w.rendre_durable()?;
                w.index_mut()?.personas.par_id.insert(p.id, entree);
                Ok(())
            },
        )
        .await
    }

    async fn delete_persona(&self, id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(e) = Self::entree_persona(w, id)? else {
                    return Ok(false);
                };
                let noeud = e.noeud;
                // The deletion marker first: from here on, a crash leaves a node that
                // opening finishes deleting, never a persona served with part of its
                // edges gone.
                w.db.set_node_property(noeud, SUPPRESSION, Value::from(true));
                // DETACH DELETE: every edge, in both directions.
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
                    w.interne
                        .indetermine
                        .store(true, std::sync::atomic::Ordering::Release);
                    bail!(
                        "obrain a refusé la suppression du nœud {noeud:?} (persona {id}) APRÈS la \
                         suppression de ses arêtes : base indéterminée, redémarrer"
                    );
                }
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLaSuppression)?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.personas.par_id.remove(&id);
                // Its children lost their EXTENDS edge to it.
                for autre in etat.personas.par_id.values_mut() {
                    autre.parents.remove(&id);
                }
                Ok(true)
            },
        )
        .await
    }

    async fn list_personas(
        &self,
        project_id: Uuid,
        status: Option<PersonaStatus>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<PersonaNode>, usize)> {
        self.lire(|_, etat| {
            let mut personas: Vec<PersonaNode> = etat
                .personas
                .par_id
                .values()
                .filter(|e| {
                    e.persona.project_id == Some(project_id)
                        && status.is_none_or(|s| e.persona.status == s)
                })
                .map(|e| e.persona.clone())
                .collect();
            personas.sort_by(ordre_de_liste);
            let total = personas.len();
            Ok((
                personas.into_iter().skip(offset).take(limit).collect(),
                total,
            ))
        })
    }

    async fn list_global_personas(&self) -> Result<Vec<PersonaNode>> {
        self.lire(|_, etat| {
            let mut personas: Vec<PersonaNode> = etat
                .personas
                .par_id
                .values()
                .filter(|e| e.persona.project_id.is_none())
                .map(|e| e.persona.clone())
                .collect();
            personas.sort_by(ordre_de_liste);
            Ok(personas)
        })
    }

    async fn add_persona_skill(&self, persona_id: Uuid, skill_id: Uuid) -> Result<()> {
        refus("PersonaStore::add_persona_skill", CAUSE_SKILL)
    }

    async fn remove_persona_skill(&self, persona_id: Uuid, skill_id: Uuid) -> Result<()> {
        refus("PersonaStore::remove_persona_skill", CAUSE_SKILL)
    }

    async fn add_persona_protocol(&self, persona_id: Uuid, protocol_id: Uuid) -> Result<()> {
        refus("PersonaStore::add_persona_protocol", CAUSE_PROTOCOLE)
    }

    async fn remove_persona_protocol(&self, persona_id: Uuid, protocol_id: Uuid) -> Result<()> {
        refus("PersonaStore::remove_persona_protocol", CAUSE_PROTOCOLE)
    }

    async fn increment_persona_activation(&self, persona_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // An absent persona is a no-op, as Neo4j's MATCH.
                let Some(e) = Self::entree_persona(w, persona_id)? else {
                    return Ok(());
                };
                let maintenant = Value::from(Utc::now().to_rfc3339());
                w.etape()?;
                w.db.set_node_property(
                    e.noeud,
                    "activation_count",
                    Value::from(e.persona.activation_count + 1),
                );
                w.etape()?;
                w.db.set_node_property(e.noeud, "last_activated", maintenant.clone());
                w.etape()?;
                w.db.set_node_property(e.noeud, "updated_at", maintenant);
                let entree = w.apres_ecriture(EntreePersona::decoder(w.db, e.noeud, Some(&e)))?;
                w.rendre_durable()?;
                w.index_mut()?.personas.par_id.insert(persona_id, entree);
                Ok(())
            },
        )
        .await
    }

    async fn set_persona_feature_graph(
        &self,
        persona_id: Uuid,
        feature_graph_id: Uuid,
    ) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (mut e, cible) = {
                    let etat = w.index()?;
                    let e = etat
                        .personas
                        .par_id
                        .get(&persona_id)
                        .cloned()
                        .ok_or_else(|| anyhow!("persona {persona_id} introuvable"))?;
                    let cible = etat.graphes.noeud_de(feature_graph_id).ok_or_else(|| {
                        anyhow!(
                            "graphe de fonctionnalité {feature_graph_id} introuvable : la portée \
                             n'est pas changée (Neo4j effacerait l'ancienne sans en poser de \
                             nouvelle)"
                        )
                    })?;
                    (e, cible)
                };
                // Replaces the scope: every SCOPED_TO edge goes, then the new one.
                for a in aretes_vers(w.db, e.noeud, None, PORTEE) {
                    w.etape()?;
                    w.db.delete_edge(a);
                    e.etrangeres.remove(&a);
                }
                w.etape()?;
                let a = w.db.create_edge(e.noeud, cible, PORTEE);
                e.graphes.clear();
                e.graphes.insert(feature_graph_id, a);
                w.rendre_durable()?;
                w.index_mut()?.personas.par_id.insert(persona_id, e);
                Ok(())
            },
        )
        .await
    }

    async fn remove_persona_feature_graph(&self, persona_id: Uuid) -> Result<()> {
        self.retirer_aretes(persona_id, PORTEE, None).await
    }

    async fn add_persona_file(&self, persona_id: Uuid, file_path: &str, weight: f64) -> Result<()> {
        refus("PersonaStore::add_persona_file", CAUSE_CODE)
    }

    async fn remove_persona_file(&self, persona_id: Uuid, file_path: &str) -> Result<()> {
        refus("PersonaStore::remove_persona_file", CAUSE_CODE)
    }

    async fn add_persona_function(
        &self,
        persona_id: Uuid,
        function_id: &str,
        weight: f64,
    ) -> Result<()> {
        refus("PersonaStore::add_persona_function", CAUSE_CODE)
    }

    async fn remove_persona_function(&self, persona_id: Uuid, function_id: &str) -> Result<()> {
        refus("PersonaStore::remove_persona_function", CAUSE_CODE)
    }

    async fn add_persona_note(&self, persona_id: Uuid, note_id: Uuid, weight: f64) -> Result<()> {
        self.ecrire_utilise(persona_id, note_id, weight, false)
            .await
    }

    async fn remove_persona_note(&self, persona_id: Uuid, note_id: Uuid) -> Result<()> {
        self.retirer_aretes(persona_id, UTILISE, Some(note_id))
            .await
    }

    async fn add_persona_decision(
        &self,
        persona_id: Uuid,
        decision_id: Uuid,
        weight: f64,
    ) -> Result<()> {
        refus("PersonaStore::add_persona_decision", CAUSE_DECISION)
    }

    async fn remove_persona_decision(&self, persona_id: Uuid, decision_id: Uuid) -> Result<()> {
        refus("PersonaStore::remove_persona_decision", CAUSE_DECISION)
    }

    async fn add_persona_extends(&self, child_id: Uuid, parent_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (mut e, cible) = {
                    let etat = w.index()?;
                    let e = etat
                        .personas
                        .par_id
                        .get(&child_id)
                        .cloned()
                        .ok_or_else(|| anyhow!("persona {child_id} introuvable"))?;
                    let cible = etat
                        .personas
                        .par_id
                        .get(&parent_id)
                        .map(|p| p.noeud)
                        .ok_or_else(|| anyhow!("persona parente {parent_id} introuvable"))?;
                    (e, cible)
                };
                // MERGE: an existing edge is kept.
                if let Some(a) = aretes_vers(w.db, e.noeud, Some(cible), ETEND).first() {
                    if e.parents.get(&parent_id) == Some(a) {
                        return Ok(());
                    }
                    e.etrangeres.remove(a);
                    e.parents.insert(parent_id, *a);
                } else {
                    let a = w.db.create_edge(e.noeud, cible, ETEND);
                    e.parents.insert(parent_id, a);
                }
                w.rendre_durable()?;
                w.index_mut()?.personas.par_id.insert(child_id, e);
                Ok(())
            },
        )
        .await
    }

    async fn remove_persona_extends(&self, child_id: Uuid, parent_id: Uuid) -> Result<()> {
        self.retirer_aretes(child_id, ETEND, Some(parent_id)).await
    }

    /// Served when every out-edge of the persona is resolved; refused by name otherwise.
    async fn get_persona_subgraph(&self, persona_id: Uuid) -> Result<PersonaSubgraph> {
        self.lire(|_, etat| {
            let idx = &etat.personas.par_id;
            let e = idx
                .get(&persona_id)
                .ok_or_else(|| anyhow!("Persona {persona_id} not found"))?;
            let refus_incomplet = |raison: String| -> Result<PersonaSubgraph> {
                Err(anyhow::Error::new(NonPorte("PersonaStore::get_persona_subgraph"))
                    .context(raison))
            };
            if !e.enfants_inconnus.is_empty() {
                return refus_incomplet(format!(
                    "{} arête(s) EXTENDS entrante(s) vers la persona {persona_id} viennent d'un                      nœud que ce backend n'indexe pas (en quarantaine) : la liste des enfants                      serait incomplète en silence",
                    e.enfants_inconnus.len()
                ));
            }
            if !e.etrangeres.is_empty() {
                return Err(
                    anyhow::Error::new(NonPorte("PersonaStore::get_persona_subgraph")).context(
                        format!(
                    "la persona {persona_id} porte {} arête(s) que ce backend ne résout pas \
                     (KNOWS, MASTERS, FOLLOWS, USES vers une décision, cible absente ou poids \
                     illisible) : son sous-graphe serait incomplet en silence",
                    e.etrangeres.len()
                ),
                    ),
                );
            }
            let relation = |t: &str, id: String, poids: f64| PersonaWeightedRelation {
                entity_type: t.to_string(),
                entity_id: id,
                weight: poids,
            };
            // USES → current notes, `ORDER BY weight DESC, entity_id DESC`.
            let mut notes: Vec<PersonaWeightedRelation> = e
                .notes
                .iter()
                .filter(|(n, _)| etat.notes.est_courante(**n))
                .map(|(n, (_, w))| relation("note", n.to_string(), *w))
                .collect();
            notes.sort_by(|a, b| {
                b.weight
                    .total_cmp(&a.weight)
                    .then_with(|| b.entity_id.cmp(&a.entity_id))
            });
            // EXTENDS*1..10, `ORDER BY depth ASC, id ASC`, each ancestor at its least depth.
            let mut profondeurs: HashMap<Uuid, usize> = HashMap::new();
            let mut file: VecDeque<(Uuid, usize)> = VecDeque::from([(persona_id, 0)]);
            while let Some((courant, d)) = file.pop_front() {
                if d == PROFONDEUR_HERITAGE {
                    continue;
                }
                let Some(c) = idx.get(&courant) else { continue };
                // An ancestor whose own EXTENDS this backend does not resolve: the
                // walk would stop short of what Neo4j reaches.
                if c.etrangeres.values().any(|etend| *etend) {
                    return refus_incomplet(format!(
                        "l'ancêtre {courant} de la persona {persona_id} porte une arête EXTENDS                          que ce backend ne résout pas (cible en quarantaine ou arête illisible) :                          la liste des parents serait incomplète en silence"
                    ));
                }
                for parent in c.parents.keys() {
                    if !profondeurs.contains_key(parent) {
                        profondeurs.insert(*parent, d + 1);
                        file.push_back((*parent, d + 1));
                    }
                }
            }
            let mut ancetres: Vec<(usize, String)> = profondeurs
                .into_iter()
                .map(|(id, d)| (d, id.to_string()))
                .collect();
            ancetres.sort();
            let parents = ancetres
                .into_iter()
                .map(|(_, id)| relation("persona", id, 1.0))
                .collect();
            let mut enfants: Vec<String> = idx
                .values()
                .filter(|x| x.parents.contains_key(&persona_id))
                .map(|x| x.persona.id.to_string())
                .collect();
            enfants.sort();
            let children = enfants
                .into_iter()
                .map(|id| relation("persona", id, 1.0))
                .collect();
            let feature_graph_id = e.graphes.keys().next().copied();
            let total_entities = notes.len() + usize::from(feature_graph_id.is_some());
            Ok(PersonaSubgraph {
                persona_id,
                persona_name: e.persona.name.clone(),
                files: vec![],
                functions: vec![],
                notes,
                decisions: vec![],
                skills: vec![],
                protocols: vec![],
                feature_graph_id,
                parents,
                children,
                stats: PersonaSubgraphStats {
                    total_entities,
                    coverage_score: 0.0,
                    freshness: 0.0,
                },
            })
        })
    }

    async fn find_personas_for_file(
        &self,
        file_path: &str,
        project_id: Uuid,
    ) -> Result<Vec<(PersonaNode, f64)>> {
        refus("PersonaStore::find_personas_for_file", CAUSE_CODE)
    }

    async fn get_all_persona_knows(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(PersonaNode, String, f64)>> {
        refus("PersonaStore::get_all_persona_knows", CAUSE_CODE)
    }

    async fn maintain_personas(&self, project_id: Uuid) -> Result<(usize, usize, usize)> {
        refus(
            "PersonaStore::maintain_personas",
            "décroît et élague les KNOWS vers fichiers et fonctions, recalcule la cohésion sur \
             le graphe de code : domaine du code non porté",
        )
    }

    async fn detect_personas(&self, project_id: Uuid) -> Result<Vec<PersonaProposal>> {
        refus(
            "PersonaStore::detect_personas",
            "lit les communautés Louvain des fichiers : domaine du code non porté",
        )
    }

    async fn find_adjacent_personas(
        &self,
        file_path: &str,
        project_id: Uuid,
    ) -> Result<Vec<(Uuid, String)>> {
        refus(
            "PersonaStore::find_adjacent_personas",
            "lit les KNOWS et le graphe IMPORTS des fichiers : domaine du code non porté",
        )
    }

    async fn find_relevant_personas_for_note(
        &self,
        file_paths: &[String],
        project_id: Uuid,
    ) -> Result<Vec<(Uuid, f64)>> {
        refus("PersonaStore::find_relevant_personas_for_note", CAUSE_CODE)
    }

    async fn find_relevant_personas_for_decision(
        &self,
        decision_id: Uuid,
        project_id: Uuid,
    ) -> Result<Vec<(Uuid, f64)>> {
        refus(
            "PersonaStore::find_relevant_personas_for_decision",
            "lit les AFFECTS d'une décision et les KNOWS : domaines Decision et code non portés",
        )
    }

    async fn auto_link_note_to_persona(
        &self,
        persona_id: Uuid,
        note_id: Uuid,
        weight: f64,
    ) -> Result<()> {
        self.ecrire_utilise(persona_id, note_id, weight, true).await
    }

    async fn auto_link_decision_to_persona(
        &self,
        persona_id: Uuid,
        decision_id: Uuid,
        weight: f64,
    ) -> Result<()> {
        refus(
            "PersonaStore::auto_link_decision_to_persona",
            CAUSE_DECISION,
        )
    }

    async fn auto_link_file_to_persona(
        &self,
        persona_id: Uuid,
        file_path: &str,
        weight: f64,
    ) -> Result<()> {
        refus("PersonaStore::auto_link_file_to_persona", CAUSE_CODE)
    }

    async fn compute_persona_affinity(
        &self,
        persona_a: Uuid,
        persona_b: Uuid,
    ) -> Result<PersonaAffinityScore> {
        refus(
            "PersonaStore::compute_persona_affinity",
            "Jaccard des KNOWS fichiers et densité de SYNAPSE entre notes : domaine du code et \
             synapses non portés",
        )
    }

    async fn merge_personas(&self, keep_id: Uuid, merge_id: Uuid) -> Result<()> {
        refus(
            "PersonaStore::merge_personas",
            "transfère les KNOWS vers fichiers et fonctions : domaine du code non porté",
        )
    }

    async fn find_synapse_linked_personas(
        &self,
        persona_id: Uuid,
    ) -> Result<Vec<(Uuid, String, f64)>> {
        refus("PersonaStore::find_synapse_linked_personas", CAUSE_SYNAPSE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::models::{PersonaOrigin, ProjectNode, ProjectProfile};
    use crate::neo4j::traits::{NoteStore, ProjectStore};
    use crate::notes::{Note, NoteType};
    use crate::obrain::Creation;
    use chrono::TimeZone;

    fn projet() -> ProjectNode {
        let id = Uuid::new_v4();
        ProjectNode {
            id,
            name: "persona".into(),
            slug: format!("persona-{id}"),
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

    pub(crate) fn persona(projet: Option<Uuid>, nom: &str, energie: f64) -> PersonaNode {
        PersonaNode {
            id: Uuid::new_v4(),
            project_id: projet,
            name: nom.into(),
            description: format!("{nom} desc"),
            status: PersonaStatus::Active,
            complexity_default: Some("complex".into()),
            timeout_secs: Some(600),
            max_cost_usd: Some(1.5),
            model_preference: None,
            system_prompt_override: None,
            energy: energie,
            cohesion: 0.2,
            activation_count: 3,
            success_rate: 0.9,
            avg_duration_secs: 12.0,
            last_activated: None,
            energy_boost_accumulated: 0.1,
            energy_history: vec![0.4, 0.5],
            origin: PersonaOrigin::AutoBuild,
            created_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
            updated_at: None,
        }
    }

    #[tokio::test]
    async fn cycle_de_vie_ordre_et_aretes() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let p = projet();
        store.create_project(&p).await.unwrap();
        let a = persona(Some(p.id), "a", 0.3);
        let b = persona(Some(p.id), "b", 0.9);
        let g = persona(None, "globale", 0.5);
        for x in [&a, &b, &g] {
            store.create_persona(x).await.unwrap();
        }
        assert!(store.create_persona(&a).await.is_err(), "id unique");
        let mut orpheline = persona(Some(Uuid::new_v4()), "orpheline", 0.1);
        assert!(store.create_persona(&orpheline).await.is_err());
        orpheline.project_id = None;
        orpheline.energy = f64::NAN;
        assert!(store.create_persona(&orpheline).await.is_err());

        let relue = store.get_persona(a.id).await.unwrap().unwrap();
        assert_eq!(relue.energy_history, vec![0.4, 0.5]);
        assert_eq!(relue.origin, PersonaOrigin::AutoBuild);
        assert_eq!(relue.timeout_secs, Some(600));
        let (liste, total) = store.list_personas(p.id, None, 10, 0).await.unwrap();
        assert_eq!(total, 2);
        assert_eq!(
            liste.iter().map(|x| x.id).collect::<Vec<_>>(),
            vec![b.id, a.id]
        );
        assert_eq!(store.list_global_personas().await.unwrap()[0].id, g.id);

        // Arêtes : note, héritage.
        let n = Note::new(Some(p.id), NoteType::Tip, "astuce".into(), "alice".into());
        store.create_note(&n).await.unwrap();
        store.add_persona_note(a.id, n.id, 0.4).await.unwrap();
        store
            .auto_link_note_to_persona(a.id, n.id, 0.2)
            .await
            .unwrap();
        store.add_persona_extends(a.id, b.id).await.unwrap();
        store.add_persona_extends(b.id, g.id).await.unwrap();
        let sg = store.get_persona_subgraph(a.id).await.unwrap();
        assert_eq!(sg.notes.len(), 1);
        assert_eq!(sg.notes[0].weight, 0.4, "auto-link garde le maximum");
        assert_eq!(
            sg.parents
                .iter()
                .map(|r| r.entity_id.clone())
                .collect::<Vec<_>>(),
            vec![b.id.to_string(), g.id.to_string()]
        );
        assert_eq!(
            store.get_persona_subgraph(b.id).await.unwrap().children[0].entity_id,
            a.id.to_string()
        );
        // La note supprimée emporte l'arête.
        store.delete_note(n.id).await.unwrap();
        assert!(store
            .get_persona_subgraph(a.id)
            .await
            .unwrap()
            .notes
            .is_empty());
        // La parente supprimée emporte l'héritage.
        assert!(store.delete_persona(b.id).await.unwrap());
        assert!(!store.delete_persona(b.id).await.unwrap());
        assert!(store
            .get_persona_subgraph(a.id)
            .await
            .unwrap()
            .parents
            .is_empty());

        store.increment_persona_activation(a.id).await.unwrap();
        let relue = store.get_persona(a.id).await.unwrap().unwrap();
        assert_eq!(relue.activation_count, 4);
        assert!(relue.last_activated.is_some());
    }

    #[tokio::test]
    async fn persona_et_aretes_survivent_a_la_reouverture() {
        use crate::neo4j::models::FeatureGraphNode;
        use crate::neo4j::traits::FeatureGraphStore;
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = projet();
        let a = persona(Some(p.id), "a", 0.3);
        let b = persona(None, "b", 0.3);
        let n = Note::new(Some(p.id), NoteType::Tip, "astuce".into(), "alice".into());
        let fg = FeatureGraphNode {
            id: Uuid::new_v4(),
            name: "fg".into(),
            description: None,
            project_id: p.id,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            entity_count: None,
            entry_function: None,
            build_depth: None,
            include_relations: None,
        };
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_project(&p).await.unwrap();
            store.create_note(&n).await.unwrap();
            store.create_feature_graph(&fg).await.unwrap();
            store.create_persona(&a).await.unwrap();
            store.create_persona(&b).await.unwrap();
            store.add_persona_note(a.id, n.id, 0.7).await.unwrap();
            store.add_persona_extends(a.id, b.id).await.unwrap();
            store.set_persona_feature_graph(a.id, fg.id).await.unwrap();
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
        let sg = store.get_persona_subgraph(a.id).await.unwrap();
        assert_eq!(sg.notes[0].weight, 0.7);
        assert_eq!(sg.parents[0].entity_id, b.id.to_string());
        assert_eq!(sg.feature_graph_id, Some(fg.id));
        assert_eq!(sg.stats.total_entities, 2);
        // Le graphe supprimé emporte la portée.
        store.delete_feature_graph(fg.id).await.unwrap();
        assert_eq!(
            store
                .get_persona_subgraph(a.id)
                .await
                .unwrap()
                .feature_graph_id,
            None
        );
    }
}
