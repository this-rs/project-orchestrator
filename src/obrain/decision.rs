//! `DecisionStore` on obrain — the decision NODE, its direct project attachment,
//! and its supersession. A PARTIAL port: each method is either written here or
//! refused by name (`NonPorte`) with the reason in its context. Nothing is
//! answered with an empty or default value in place of a refusal.
//!
//! PORTED here, with the Neo4j semantics of `src/neo4j/decision.rs`:
//! - `create_decision_global`, `get_decision`, `update_decision`,
//!   `delete_decision`;
//! - `get_decision_project_id` and `get_decision_project_ids`, through the
//!   direct `(:Project)-[:HAS_DECISION]->(:Decision)` edge only;
//! - `supersede_decision`, the `SUPERSEDES` edge between two decisions.
//!
//! REFUSED by name: everything that reads or writes a `Task` (the task-bound
//! `create_decision`, the `INFORMED_BY` traversals of the project id, the
//! timeline filter, the status list's task column, `get_decisions_for_entity`),
//! the `AFFECTS` edges whose targets are code entities (`File`, `Function`,
//! `Struct`, … — the code domain is not ported), and the three embedding
//! methods (see below). They are refused, never answered empty.
//!
//! # Layout
//!
//! A decision is one node labelled `Decision`, with the properties Neo4j writes
//! in `create_decision_global`: `id`, `description`, `rationale`,
//! `alternatives` (a list), `chosen_option` (the empty string for `None`),
//! `decided_by`, `decided_at` (RFC 3339 string), `status` (the lowercase
//! variant, as `DecisionStatus::to_string`), `sharing_consent`. The project of
//! a decision is NOT a property (Neo4j stores none): it is the incoming
//! `HAS_DECISION` edge from the `Project`, read at open into the index.
//!
//! # Divergences, all on the strict side or named
//!
//! - a creation whose `id` is already taken is refused (Neo4j would create a
//!   second node with the same id);
//! - `create_decision_global` with a `project_id` whose project this backend
//!   does not hold is refused (Neo4j errors too, with its own message);
//! - `get_decision_project_id` / `get_decision_project_ids` resolve only the
//!   direct `HAS_DECISION` edge: the task chain (`Decision←Task←Plan←Project`)
//!   cannot exist here because the task-bound `create_decision` is refused, so
//!   a decision's only project is the one it was attached to at creation;
//! - the three embedding methods are refused by name: obrain does not journal
//!   vector values in its WAL (trame `VecColumnWrite` without value; measured
//!   13/11), so an embedding acknowledged could not be guaranteed durable —
//!   the same reason as `ChatStore::search_episodes_by_embedding`;
//! - `decided_at` is stored as an RFC 3339 **string**; Neo4j stores a `datetime`
//!   (`datetime($decided_at)`). A DIVERGENCE in favour of this backend: here the
//!   instant written is the instant read back; `Neo4jClient::node_to_decision`
//!   reads it with `get::<String>`, which a `datetime` does not convert to
//!   (`vendor/neo4rs/src/convert.rs`, `TryFrom<BoltType> for String` accepts
//!   only `BoltType::String`), so Neo4j falls back to `Utc::now()` — the time of
//!   the READ. (A migration reading Neo4j through `node_to_decision` would lose
//!   every date.) The string sorts chronologically only for the format written
//!   here (`DateTime<Utc>::to_rfc3339`: always `+00:00`, a fraction of 0/3/6/9
//!   digits, a 4-digit year); a mixed format (a `Z` suffix, another offset)
//!   would be read without error and sorted wrong. No ported method sorts or
//!   filters on `decided_at`; the future port of the timeline must sort on the
//!   PARSED value (or normalise the writes);
//! - `update_decision`, `delete_decision` and `supersede_decision` on an id that
//!   is RESERVED (a node of that id is on disk but quarantined at open) refuse,
//!   named, rather than answer `Ok` without touching it: Neo4j would act on
//!   that node. An id that is neither held nor reserved stays the silent no-op
//!   of Neo4j's `MATCH`;
//! - a decision attached to a project that opening QUARANTINED (unreadable, so
//!   absent from the project index) reads as having no project: Neo4j, which
//!   has no quarantine, would return that project's id. The decision itself is
//!   still served; only its project is not, because it cannot be verified;
//! - `supersede_decision` draws the `SUPERSEDES` edge ONCE, and only then sets
//!   `created_at` on it (RFC 3339 string). Neo4j writes
//!   `MERGE (new)-[:SUPERSEDES {created_at: datetime()}]->(old)`: the `MERGE`
//!   pattern carries a fresh `datetime()`, so it never matches an earlier edge
//!   and a repeated call draws a second one. Here a repeated call converges to
//!   the single edge the `MERGE` intends (a retry after a crash must, too);
//! - `ProjectStore::delete_project` refuses (`NonPorte`, cascade) as soon as the
//!   base holds a decision, attached to that project or not: its check counts
//!   every live node against the projects and plans it addresses, and the Neo4j
//!   cascade would have to reach the decisions (it archives them). Writing a
//!   decision therefore makes every project of the base undeletable here, until
//!   the cascade is ported;
//! - a decision node whose `status` does not parse decodes as `Accepted`, as
//!   `Neo4jClient::node_to_decision` does (the model's own default is
//!   `Proposed`); `embedding` is never loaded (`None`), and `scar_intensity`
//!   reads 0.0 for nodes created here (it is not written by
//!   `create_decision_global`, in Neo4j either).
//!
//! # Atomicity
//!
//! A creation follows the draft-marker protocol of `project.rs`: the node, its
//! properties and its `HAS_DECISION` edge are written under the marker
//! `_brouillon`, the marker is removed last (the commit point), and opening
//! deletes any draft with its edges. A value above [`SEUIL_COLONNE_BLOB`] — a
//! long `description` or `rationale` — makes the creation pay the FULL flush
//! before the marker goes.
//!
//! A multi-property update is not atomic. `update_decision` writes each given
//! property as its own engine write, and `supersede_decision` writes the
//! status, then the edge, then the edge's `created_at`: a crash in between
//! leaves some of them written and the others not, which opening does not
//! repair (each state reads as a valid decision). A retry of the same call
//! converges: the writes are idempotent, the edge is drawn once, and an edge
//! found without its `created_at` gets it.
//!
//! Durability of long values: a value above [`SEUIL_COLONNE_BLOB`] is not
//! recovered by the WAL alone, so every write that puts one — or REPLACES one
//! that sat there — pays the full flush: the creation (any property, list
//! elements included), and `update_decision` for `description`, `rationale`
//! and `chosen_option`, old and new value (as `note.rs` and `plan.rs`). A test
//! counts the fsyncs without any `close()`. The interruption tests stop a write
//! with an error and then close cleanly: they check the order of the writes
//! (the marker protocol), not what a `kill -9` leaves in the WAL.
//!
//! `delete_decision` sets the draft marker on the node BEFORE it removes the
//! node's edges: a crash in the middle leaves a draft, which opening deletes
//! with its remaining edges. The deletion is thus finished at open, never left
//! as a decision stripped of some of its edges.

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
use super::{project, Ecriture, NonPorte, ObrainGraphStore, RapportOuverture, SEUIL_COLONNE_BLOB};
use crate::episodes::distill_models::SharingConsent;
use crate::neo4j::models::{
    AffectsRelation, DecisionListItem, DecisionNode, DecisionStatus, DecisionTimelineEntry,
};
use crate::neo4j::traits::DecisionStore;

const LABEL: &str = "Decision";
const BROUILLON: &str = "_brouillon";
const APPARTIENT: &str = "HAS_DECISION";
const SUCCEDE: &str = "SUPERSEDES";

/// The methods of `DecisionStore` refused on every call, by name. Listed here so
/// the accounting of `tests/obrain_backend.rs` and `methodes_qui_servent` can
/// tell a refusal from a port.
pub const REFUSEES: &[&str] = &[
    // Reads or writes a `Task` (the `INFORMED_BY` edge): the task domain is not ported.
    "DecisionStore::create_decision",
    "DecisionStore::get_all_decisions_with_task_id",
    "DecisionStore::get_decision_timeline",
    "DecisionStore::list_decisions_by_status",
    // Reads `Task.affected_files` (indirect path) and AFFECTS edges to code entities.
    "DecisionStore::get_decisions_for_entity",
    // AFFECTS edges: their targets (File/Function/Struct/… and Task) are not ported,
    // so an edge could not be written without a phantom target, nor read back.
    "DecisionStore::get_decisions_affecting",
    "DecisionStore::add_decision_affects",
    "DecisionStore::remove_decision_affects",
    "DecisionStore::list_decision_affects",
    // Embeddings: obrain does not journal vector values (see module doc).
    "DecisionStore::set_decision_embedding",
    "DecisionStore::get_decision_embedding",
    "DecisionStore::search_decisions_by_vector",
    "DecisionStore::get_decisions_without_embedding",
];

/// A refusal, by name, with the reason it is not ported.
fn refus<T>(methode: &'static str, raison: &'static str) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).context(raison))
}

/// Does this value go to a blob column (a string above the threshold, or a
/// string inside a list)?
fn va_en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
        || v.as_list().is_some_and(|l| {
            l.iter()
                .any(|e| e.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB))
        })
}

/// The properties `create_decision_global` writes, as Neo4j names them; the
/// values are Neo4j's, except `decided_at`, a string here and a `datetime` there.
fn proprietes_de_creation(d: &DecisionNode) -> Vec<(&'static str, Value)> {
    vec![
        ("id", Value::from(d.id.to_string())),
        ("description", Value::from(d.description.clone())),
        ("rationale", Value::from(d.rationale.clone())),
        (
            "alternatives",
            Value::List(
                d.alternatives
                    .iter()
                    .map(|a| Value::from(a.clone()))
                    .collect::<Vec<_>>()
                    .into(),
            ),
        ),
        (
            "chosen_option",
            Value::from(d.chosen_option.clone().unwrap_or_default()),
        ),
        ("decided_by", Value::from(d.decided_by.clone())),
        ("decided_at", Value::from(d.decided_at.to_rfc3339())),
        ("status", Value::from(d.status.to_string())),
        (
            "sharing_consent",
            Value::from(d.sharing_consent.as_db_str()),
        ),
    ]
}

/// The domain value of a `Decision` node from a property lookup: the counterpart
/// of `Neo4jClient::node_to_decision`.
fn lire_decision(get: &dyn Fn(&str) -> Option<Value>) -> Result<DecisionNode> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let id: Uuid = chaine("id")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("nœud Decision sans propriété `id` lisible"))?;
    let description = chaine("description")
        .ok_or_else(|| anyhow!("décision {id} sans propriété `description`"))?;
    let rationale =
        chaine("rationale").ok_or_else(|| anyhow!("décision {id} sans propriété `rationale`"))?;
    let alternatives = get("alternatives")
        .and_then(|v| {
            v.as_list().map(|l| {
                l.iter()
                    .filter_map(|e| e.as_str().map(str::to_string))
                    .collect()
            })
        })
        .unwrap_or_default();
    let chosen_option = chaine("chosen_option").filter(|s| !s.is_empty());
    let decided_by = chaine("decided_by").unwrap_or_default();
    let decided_at = chaine("decided_at")
        .and_then(|s| s.parse::<DateTime<Utc>>().ok())
        .unwrap_or_else(Utc::now);
    // Neo4j's `node_to_decision` defaults a missing status to `Accepted`.
    let status = chaine("status")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DecisionStatus::Accepted);
    let embedding_model = chaine("embedding_model");
    let scar_intensity = get("scar_intensity")
        .and_then(|v| v.as_float64())
        .unwrap_or(0.0);
    let sharing_consent = chaine("sharing_consent")
        .map(|s| SharingConsent::from_db_str(&s))
        .unwrap_or_default();
    Ok(DecisionNode {
        id,
        description,
        rationale,
        alternatives,
        chosen_option,
        decided_by,
        decided_at,
        status,
        embedding: None,
        embedding_model,
        scar_intensity,
        sharing_consent,
    })
}

/// One decision: where it lives in the base, what the base says about it
/// (decoded once), and the project it is attached to directly, if any.
#[derive(Clone)]
struct EntreeDecision {
    noeud: NodeId,
    decision: DecisionNode,
    projet: Option<Uuid>,
}

impl EntreeDecision {
    /// Decode the node (one hydration) into an entry with the given project.
    fn decoder(db: &ObrainDB, noeud: NodeId, projet: Option<Uuid>) -> Result<Self> {
        let n = db
            .get_node(noeud)
            .ok_or_else(|| anyhow!("nœud Decision {noeud:?} indexé mais absent de la base"))?;
        let get = |cle: &str| {
            n.get_property(cle)
                .filter(|v| !matches!(v, Value::Null))
                .cloned()
        };
        let decision = lire_decision(&get)?;
        Ok(Self {
            noeud,
            decision,
            projet,
        })
    }
}

/// The index of the decisions.
#[derive(Default)]
pub(super) struct IndexDecisions {
    par_id: HashMap<Uuid, EntreeDecision>,
    /// Ids of nodes that are on disk but could not be read at open (quarantined):
    /// reserved, so a later creation refuses rather than overwrites what is there.
    reserves: HashSet<Uuid>,
}

impl IndexDecisions {
    fn inscrire(&mut self, e: EntreeDecision) {
        self.par_id.insert(e.decision.id, e);
    }

    fn desinscrire(&mut self, id: Uuid) -> Option<EntreeDecision> {
        self.par_id.remove(&id)
    }

    fn publier(&mut self, e: EntreeDecision) {
        self.inscrire(e);
    }

    pub(super) fn reconstruire(
        db: &ObrainDB,
        projets: &project::IndexProjets,
        rapport: &mut RapportOuverture,
    ) -> Result<Self> {
        let store = db.store();
        let mut index = Self::default();
        let mut vus: Vec<(NodeId, Uuid)> = Vec::new();
        for noeud in store.nodes_by_label(LABEL) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud Decision {noeud:?} : illisible"));
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
            let decision = match lire_decision(&get) {
                Ok(v) => v,
                Err(e) => {
                    if let Some(id) =
                        get("id").and_then(|v| v.as_str().and_then(|t| t.parse::<Uuid>().ok()))
                    {
                        index.reserves.insert(id);
                    }
                    rapport
                        .en_quarantaine
                        .push(format!("nœud Decision {noeud:?} : {e}"));
                    continue;
                }
            };
            if index.par_id.contains_key(&decision.id) {
                // The id is on disk twice: the second node is quarantined, and the
                // id stays reserved even once the first is deleted, so a later
                // creation never lands next to the node left in quarantine.
                index.reserves.insert(decision.id);
                rapport.en_quarantaine.push(format!(
                    "nœud Decision {noeud:?} : id {} déjà pris",
                    decision.id
                ));
                continue;
            }
            let id = decision.id;
            index.inscrire(EntreeDecision {
                noeud,
                decision,
                projet: None,
            });
            vus.push((noeud, id));
        }

        // The project: the incoming `HAS_DECISION` edge of every indexed decision.
        // A decision attached to a project absent from the project index (a node
        // with a non-canonical id, or one quarantined at open) reads as having no
        // project: it cannot be verified. A DIVERGENCE, named in the module doc:
        // Neo4j has no quarantine and would return that project's id.
        for (noeud, id) in &vus {
            let mut projet: Option<Uuid> = None;
            for (source, arete) in store.edges_from(*noeud, Direction::Incoming) {
                let Some(e) = db.get_edge(arete) else {
                    continue;
                };
                if &*e.edge_type != APPARTIENT {
                    continue;
                }
                let pid = db.get_node(source).and_then(|n| {
                    n.get_property("id")
                        .and_then(|v| v.as_str().and_then(|t| t.parse::<Uuid>().ok()))
                });
                // The project of that id must be THIS node: a duplicate quarantined at
                // open carries the id of the held project, and is not it.
                if let Some(pid) = pid.filter(|p| projets.noeud_de(*p) == Some(source)) {
                    projet = Some(projet.map_or(pid, |existant| existant.min(pid)));
                }
            }
            if let Some(p) = projet {
                if let Some(mut e) = index.desinscrire(*id) {
                    e.projet = Some(p);
                    index.inscrire(e);
                }
            }
        }
        Ok(index)
    }
}

impl ObrainGraphStore {
    /// The decision entry of `id`, from the indexes, as a clone (the write
    /// closures work on a copy and publish a new state).
    fn entree_decision(w: &Ecriture<'_>, id: Uuid) -> Result<Option<EntreeDecision>> {
        Ok(w.index()?.decisions.par_id.get(&id).cloned())
    }

    /// A refusal for a write on a RESERVED id (on disk, quarantined at open): it
    /// is neither touched nor answered `Ok` as if absent (module doc).
    fn refuser_si_reserve(w: &Ecriture<'_>, id: Uuid, operation: &str) -> Result<()> {
        if w.index()?.decisions.reserves.contains(&id) {
            bail!(
                "{operation} de la décision {id} refusé : un nœud Decision de cet id est sur \
                 disque, illisible ou en double à l'ouverture (quarantaine, voir le rapport \
                 d'ouverture). Il n'est ni modifié ni ignoré en silence"
            );
        }
        Ok(())
    }

    /// The decoded state of a decision, read back from the engine after a write.
    fn relire_decision(w: &Ecriture<'_>, e: &EntreeDecision) -> Result<EntreeDecision> {
        w.apres_ecriture(EntreeDecision::decoder(w.db, e.noeud, e.projet))
    }
}

/// The ported half of `DecisionStore`. The name of the trait method is the name
/// of its refusal, so a refusal is findable by grepping the contract.
#[async_trait]
impl DecisionStore for ObrainGraphStore {
    async fn create_decision(&self, _task_id: Uuid, _decision: &DecisionNode) -> Result<()> {
        refus(
            "DecisionStore::create_decision",
            "écrit l'arête `(Task)-[:INFORMED_BY]->(Decision)` : domaine Task non porté",
        )
    }

    async fn create_decision_global(
        &self,
        decision: &DecisionNode,
        project_id: Option<Uuid>,
    ) -> Result<()> {
        let d = decision.clone();
        self.ecrire(
            move || d,
            move |d, w: &mut Ecriture<'_>| {
                let (projet_noeud, proprietes) = {
                    let etat = w.index()?;
                    if etat.decisions.par_id.contains_key(&d.id) {
                        bail!("une décision d'id {} existe déjà", d.id);
                    }
                    if etat.decisions.reserves.contains(&d.id) {
                        bail!(
                            "l'id {} est réservé : un nœud Decision de cet id est sur disque, \
                             illisible à l'ouverture (quarantaine). Il n'est ni repris ni écrasé",
                            d.id
                        );
                    }
                    let projet = match project_id {
                        Some(p) => {
                            let n = etat.projets.noeud_de(p);
                            if n.is_none() {
                                bail!(
                                    "création d'une décision rattachée au projet {p} : projet \
                                     inconnu de ce backend ; rien n'est créé"
                                );
                            }
                            n
                        }
                        None => None,
                    };
                    (projet, proprietes_de_creation(&d))
                };
                w.flush_complet = proprietes.iter().any(|(_, v)| va_en_colonne_blob(v));
                let noeud = w.db.create_node(&[LABEL]);
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeNoeud)?;
                w.etape()?;
                w.db.set_node_property(noeud, BROUILLON, Value::from(true));
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeMarqueur)?;
                for (k, (cle, valeur)) in proprietes.into_iter().enumerate() {
                    w.etape()?;
                    w.db.set_node_property(noeud, cle, valeur);
                    #[cfg(any(test, feature = "mock-store"))]
                    w.point(Interruption::ApresLaPropriete(k + 1))?;
                }
                if let Some(projet) = projet_noeud {
                    let _ = w.db.create_edge(projet, noeud, APPARTIENT);
                }
                w.rendre_durable_avant_le_marqueur()?;
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::AvantLeRetraitDuMarqueur)?;
                w.etape()?;
                w.db.remove_node_property(noeud, BROUILLON);
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeRetraitDuMarqueur)?;
                let entree = EntreeDecision::decoder(w.db, noeud, project_id).and_then(|e| {
                    if e.decision.id == d.id {
                        Ok(e)
                    } else {
                        bail!("la décision relue n'a pas l'id écrit")
                    }
                });
                let entree = w.apres_ecriture(entree)?;
                w.rendre_durable()?;
                w.index_mut()?.decisions.publier(entree);
                Ok(())
            },
        )
        .await
    }

    async fn get_decision(&self, decision_id: Uuid) -> Result<Option<DecisionNode>> {
        self.lire(|_, etat| {
            Ok(etat
                .decisions
                .par_id
                .get(&decision_id)
                .map(|e| e.decision.clone()))
        })
    }

    async fn get_decision_project_id(&self, decision_id: Uuid) -> Result<Option<String>> {
        self.lire(|_, etat| {
            Ok(etat
                .decisions
                .par_id
                .get(&decision_id)
                .and_then(|e| e.projet)
                .map(|p| p.to_string()))
        })
    }

    async fn get_decision_project_ids(&self, decision_ids: &[Uuid]) -> Result<HashMap<Uuid, Uuid>> {
        self.lire(|_, etat| {
            let mut out = HashMap::new();
            for id in decision_ids {
                if let Some(e) = etat.decisions.par_id.get(id) {
                    if let Some(p) = e.projet {
                        out.insert(*id, p);
                    }
                }
            }
            Ok(out)
        })
    }

    async fn update_decision(
        &self,
        decision_id: Uuid,
        description: Option<String>,
        rationale: Option<String>,
        chosen_option: Option<String>,
        status: Option<DecisionStatus>,
    ) -> Result<()> {
        if description.is_none()
            && rationale.is_none()
            && chosen_option.is_none()
            && status.is_none()
        {
            return Ok(());
        }
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                Self::refuser_si_reserve(w, decision_id, "update_decision")?;
                let Some(ancienne) = Self::entree_decision(w, decision_id)? else {
                    // Neo4j's MATCH on an unknown id is a silent no-op.
                    return Ok(());
                };
                // The OLD value too: replacing a value that sat in a blob column clears
                // that slot, which the WAL does not recover (as in note.rs).
                w.flush_complet = ancienne.decision.description.len() > SEUIL_COLONNE_BLOB
                    || ancienne.decision.rationale.len() > SEUIL_COLONNE_BLOB
                    || description
                        .as_ref()
                        .is_some_and(|c| c.len() > SEUIL_COLONNE_BLOB)
                    || rationale
                        .as_ref()
                        .is_some_and(|c| c.len() > SEUIL_COLONNE_BLOB)
                    // `chosen_option` is a free string too: its old and its new value.
                    || ancienne
                        .decision
                        .chosen_option
                        .as_ref()
                        .is_some_and(|c| c.len() > SEUIL_COLONNE_BLOB)
                    || chosen_option
                        .as_ref()
                        .is_some_and(|c| c.len() > SEUIL_COLONNE_BLOB);
                let noeud = ancienne.noeud;
                if let Some(c) = &description {
                    w.etape()?;
                    w.db.set_node_property(noeud, "description", Value::from(c.clone()));
                }
                if let Some(c) = &rationale {
                    w.etape()?;
                    w.db.set_node_property(noeud, "rationale", Value::from(c.clone()));
                }
                if let Some(c) = &chosen_option {
                    w.etape()?;
                    w.db.set_node_property(noeud, "chosen_option", Value::from(c.clone()));
                }
                if let Some(s) = status {
                    w.etape()?;
                    w.db.set_node_property(noeud, "status", Value::from(s.to_string()));
                }
                let entree = Self::relire_decision(w, &ancienne)?;
                w.rendre_durable()?;
                w.index_mut()?.decisions.publier(entree);
                Ok(())
            },
        )
        .await
    }

    async fn delete_decision(&self, decision_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                Self::refuser_si_reserve(w, decision_id, "delete_decision")?;
                let Some(e) = Self::entree_decision(w, decision_id)? else {
                    // Neo4j's DETACH DELETE on an unknown id is a silent no-op.
                    return Ok(());
                };
                let noeud = e.noeud;
                // The draft marker first: from here a crash leaves a draft, which
                // opening deletes with whatever edges remain (see the module doc).
                w.etape()?;
                w.db.set_node_property(noeud, BROUILLON, Value::from(true));
                // DETACH DELETE: every edge of the decision goes with it, both ways.
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
                        "obrain a refusé la suppression du nœud {noeud:?} (décision \
                         {decision_id}) APRÈS la suppression de ses arêtes : base indéterminée, \
                         redémarrer"
                    );
                }
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLaSuppression)?;
                w.rendre_durable()?;
                w.index_mut()?.decisions.desinscrire(decision_id);
                Ok(())
            },
        )
        .await
    }

    async fn get_decisions_for_entity(
        &self,
        _entity_type: &str,
        _entity_id: &str,
        _limit: u32,
    ) -> Result<Vec<DecisionNode>> {
        refus(
            "DecisionStore::get_decisions_for_entity",
            "parcourt `Task.affected_files` (INFORMED_BY) et les arêtes AFFECTS vers des \
             entités de code : domaines non portés",
        )
    }

    async fn set_decision_embedding(
        &self,
        _decision_id: Uuid,
        _embedding: &[f32],
        _model: &str,
    ) -> Result<()> {
        refus(
            "DecisionStore::set_decision_embedding",
            "obrain ne journalise pas la valeur des vecteurs dans son WAL (trame \
             VecColumnWrite sans valeur) : un embedding acquitté ne serait pas garanti durable",
        )
    }

    async fn get_decision_embedding(&self, _decision_id: Uuid) -> Result<Option<Vec<f32>>> {
        refus(
            "DecisionStore::get_decision_embedding",
            "obrain ne journalise pas la valeur des vecteurs dans son WAL (trame \
             VecColumnWrite sans valeur) : un embedding ne peut être ni écrit ni relu",
        )
    }

    async fn get_all_decisions_with_task_id(&self) -> Result<Vec<(DecisionNode, Uuid)>> {
        refus(
            "DecisionStore::get_all_decisions_with_task_id",
            "lit l'arête `(Task)-[:INFORMED_BY]->(Decision)` : domaine Task non porté",
        )
    }

    async fn get_decisions_without_embedding(&self) -> Result<Vec<(Uuid, String, String)>> {
        refus(
            "DecisionStore::get_decisions_without_embedding",
            "backfill des embeddings : les embeddings ne sont pas portés (voir \
             `set_decision_embedding`)",
        )
    }

    async fn search_decisions_by_vector(
        &self,
        _query_embedding: &[f32],
        _limit: usize,
        _project_id: Option<&str>,
    ) -> Result<Vec<(DecisionNode, f64)>> {
        refus(
            "DecisionStore::search_decisions_by_vector",
            "recherche vectorielle : les embeddings ne sont pas portés (voir \
             `set_decision_embedding`)",
        )
    }

    async fn get_decisions_affecting(
        &self,
        _entity_type: &str,
        _entity_id: &str,
        _status_filter: Option<&str>,
    ) -> Result<Vec<DecisionNode>> {
        refus(
            "DecisionStore::get_decisions_affecting",
            "lit les arêtes AFFECTS vers des entités de code (File/Function/… et Task) : \
             domaines non portés",
        )
    }

    async fn add_decision_affects(
        &self,
        _decision_id: Uuid,
        _entity_type: &str,
        _entity_id: &str,
        _impact_description: Option<&str>,
    ) -> Result<()> {
        refus(
            "DecisionStore::add_decision_affects",
            "écrit une arête AFFECTS vers une entité de code (File/Function/… et Task) : \
             domaines non portés, l'arête ne pourrait pas être écrite sans cible fantôme",
        )
    }

    async fn remove_decision_affects(
        &self,
        _decision_id: Uuid,
        _entity_type: &str,
        _entity_id: &str,
    ) -> Result<()> {
        refus(
            "DecisionStore::remove_decision_affects",
            "retire une arête AFFECTS vers une entité de code (File/Function/… et Task) : \
             domaines non portés",
        )
    }

    async fn list_decision_affects(&self, _decision_id: Uuid) -> Result<Vec<AffectsRelation>> {
        refus(
            "DecisionStore::list_decision_affects",
            "lit les arêtes AFFECTS vers des entités de code (File/Function/… et Task) : \
             domaines non portés",
        )
    }

    async fn supersede_decision(&self, new_decision_id: Uuid, old_decision_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                Self::refuser_si_reserve(w, new_decision_id, "supersede_decision")?;
                Self::refuser_si_reserve(w, old_decision_id, "supersede_decision")?;
                let Some(nouvelle) = Self::entree_decision(w, new_decision_id)? else {
                    // Neo4j's MATCH on an unknown id is a silent no-op.
                    return Ok(());
                };
                let Some(ancienne) = Self::entree_decision(w, old_decision_id)? else {
                    return Ok(());
                };
                w.etape()?;
                w.db.set_node_property(
                    ancienne.noeud,
                    "status",
                    Value::from(DecisionStatus::Superseded.to_string()),
                );
                // MERGE: the edge is drawn once (see the module doc for Neo4j's).
                let deja: Option<(EdgeId, bool)> =
                    w.db.store()
                        .edges_from(nouvelle.noeud, Direction::Outgoing)
                        .into_iter()
                        .filter(|(cible, _)| *cible == ancienne.noeud)
                        .find_map(|(_, a)| {
                            let e = w.db.get_edge(a)?;
                            (&*e.edge_type == SUCCEDE).then(|| {
                                let date = e
                                    .get_property("created_at")
                                    .is_some_and(|v| !matches!(v, Value::Null));
                                (a, date)
                            })
                        });
                let arete = match deja {
                    Some((a, _)) => a,
                    None => {
                        w.etape()?;
                        w.db.create_edge(nouvelle.noeud, ancienne.noeud, SUCCEDE)
                    }
                };
                // Also when the edge is there WITHOUT its date: a call cut between the
                // edge and its `created_at` is finished by the retry.
                if !deja.is_some_and(|(_, date)| date) {
                    w.etape()?;
                    w.db.set_edge_property(
                        arete,
                        "created_at",
                        Value::from(Utc::now().to_rfc3339()),
                    );
                }
                let entree = Self::relire_decision(w, &ancienne)?;
                w.rendre_durable()?;
                w.index_mut()?.decisions.publier(entree);
                Ok(())
            },
        )
        .await
    }

    async fn get_decision_timeline(
        &self,
        _task_id: Option<Uuid>,
        _from: Option<&str>,
        _to: Option<&str>,
    ) -> Result<Vec<DecisionTimelineEntry>> {
        refus(
            "DecisionStore::get_decision_timeline",
            "filtre par tâche (`(Task)-[:INFORMED_BY]->(Decision)`) : domaine Task non porté",
        )
    }

    async fn list_decisions_by_status(
        &self,
        _status: DecisionStatus,
        _project_id: Option<Uuid>,
        _workspace_slug: Option<&str>,
        _limit: usize,
        _offset: usize,
    ) -> Result<(Vec<DecisionListItem>, usize)> {
        refus(
            "DecisionStore::list_decisions_by_status",
            "lit la tâche (INFORMED_BY) et le périmètre workspace (BELONGS_TO_WORKSPACE) : \
             domaines non portés",
        )
    }
}
