//! The constraints of a plan, and the `HAS_PLAN` links of a plan to its
//! projects, on obrain — two parts of [`PlanStore`](crate::neo4j::traits::PlanStore)
//! (`plan.rs` holds the trait `impl`; its methods for these two parts call the
//! functions written here).
//!
//! PORTED, with the Neo4j semantics of `src/neo4j/constraint.rs` and
//! `src/neo4j/plan.rs`: `create_constraint`, `get_constraint`,
//! `get_plan_constraints`, `update_constraint`, `delete_constraint`,
//! `link_plan_to_project`, `unlink_plan_from_project`, `list_plan_project_slugs`.
//!
//! # Layout
//!
//! - A constraint is one node labelled `Constraint` with the properties Neo4j
//!   writes: `id`, `constraint_type` (the `Display` spelling, `performance`…),
//!   `description`, `enforced_by` (the empty string for none, read back as
//!   `None`, as Neo4j does). Its plan holds it by the edge
//!   `(:Plan)-[:CONSTRAINED_BY]->(:Constraint)`.
//! - A plan belongs to a project by the edge `(:Project)-[:HAS_PLAN]->(:Plan)`
//!   AND by its `project_id` property, as in Neo4j. `create_plan` on this backend
//!   writes only the property (`plan.rs`): its implicit link is the property.
//!   The projects of a plan are therefore the sources of its `HAS_PLAN` edges
//!   PLUS its `project_id` — exactly the set Neo4j has for a plan created by
//!   `create_plan` (whose `MERGE` draws the edge). Every read scoped to a project
//!   (`list_project_plans`, `count_project_plans`, `list_plans_for_project`) and
//!   the guard of `delete_project` read that set (`plan::Entree::appartient_a`).
//!
//! # Semantics kept from Neo4j
//!
//! - `link_plan_to_project` on a missing plan or project is a silent no-op
//!   (`MATCH`); otherwise it OVERWRITES `project_id` and `MERGE`s the edge: the
//!   previous links stay (it is not a move), and linking twice draws ONE edge.
//! - `unlink_plan_from_project` removes EVERY `HAS_PLAN` of the plan and clears
//!   `project_id`; a plan with no project is left as it is.
//! - `list_plan_project_slugs`: the slugs of the plan's projects, sorted and
//!   without duplicates; empty for a missing plan.
//! - `create_constraint` on a missing plan is refused (as Neo4j does); an id
//!   already taken is refused (Neo4j has a uniqueness constraint on
//!   `Constraint.id`). `update_constraint` / `delete_constraint` on a missing
//!   constraint are silent no-ops; `update_constraint` with nothing to set
//!   writes nothing. `delete_constraint` is a `DETACH DELETE`.
//! - `get_plan_constraints`: Neo4j states no order. Served in creation order of
//!   the nodes (`NodeId`), which is deterministic; nothing should rely on it.
//!
//! # Divergences, on the strict side
//!
//! - A constraint whose `id`, `constraint_type` or `description` is unreadable
//!   is quarantined at open (named in the report, left untouched, its id
//!   reserved so that no creation takes it); Neo4j would fail the read.
//! - A constraint whose plan this backend does not address (quarantined plan)
//!   is still served by `get_constraint`, and by no `get_plan_constraints`. A
//!   constraint, or a `HAS_PLAN` edge, counts only from the very node the index
//!   addresses, never from a quarantined node that shares its id.
//! - `link_plan_to_project` towards a plan or a project that the base holds but
//!   this backend quarantined is REFUSED (an error naming the quarantine), where
//!   Neo4j's `MATCH` would find the node and link it. An id the base does not
//!   hold at all stays a silent no-op, as in Neo4j.
//! - `unlink_plan_from_project` clears a plan whose only link is its
//!   `project_id` (no edge). Neo4j's `MATCH` on `HAS_PLAN` would leave such a
//!   plan alone, but on Neo4j it cannot arise from the API (its `create_plan`
//!   draws the edge), while on obrain it is the normal state of a plan created
//!   in a project, whose property stands for that edge: the outcome is the same.
//!
//! # Atomicity, built by ordering
//!
//! - **create_constraint**: the draft marker `_brouillon` is set first, the
//!   properties and the `CONSTRAINED_BY` edge follow, the marker is removed last
//!   (the commit point). Opening deletes a draft with its edges.
//! - **link_plan_to_project**: no marker is needed, the order makes every
//!   prefix equivalent to "before" or "after". (1) the previous `project_id`, if
//!   it had no edge, gets one (the membership does not change); (2) `project_id`
//!   is overwritten (the membership is now the final one, through the property);
//!   (3) the edge of the new project is drawn if it is missing (redundant with
//!   the property, so nothing changes for a reader).
//! - **unlink_plan_from_project**: the marker `_deliaison` is set first; the
//!   edges go, `project_id` is cleared, the marker goes last. Opening FINISHES an
//!   unlink that still carries the marker (`achever_les_deliaisons`): once
//!   requested, it is carried out, never left half-way.
//! - **update_constraint**: the properties are written one by one with no marker
//!   (as `update_plan` and the notes' updates): a crash in between leaves some of
//!   them written.
//! - **delete_constraint**: its edges go, then the node. A crash in between
//!   leaves a constraint with no plan; the retry deletes it.
//!
//! Every write is published to the index only after its durability step, and a
//! failure after a first engine write makes the store indeterminate.
//!
//! A value longer than `SEUIL_COLONNE_BLOB` (a long `description` or
//! `enforced_by`) goes to a blob column that the WAL does not recover (see
//! `mod.rs`): `create_constraint` then pays the full flush BEFORE the marker is
//! removed, and `update_constraint` pays it when the new value or the value it
//! replaces is that long, as `plan.rs` and `note.rs` do.

use std::collections::{BTreeSet, HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use obrain_common::types::{EdgeId, NodeId, Value};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::{plan, project, Ecriture, ObrainGraphStore, RapportOuverture, SEUIL_COLONNE_BLOB};
use crate::neo4j::models::{ConstraintNode, ConstraintType};

const LABEL: &str = "Constraint";
const LABEL_PLAN: &str = "Plan";
const LABEL_PROJET: &str = "Project";
/// The draft marker of a creation.
const BROUILLON: &str = "_brouillon";
/// The marker of an unlink in progress, on the `Plan` node.
const DELIAISON: &str = "_deliaison";
const CONTRAINT_PAR: &str = "CONSTRAINED_BY";
const A_LE_PLAN: &str = "HAS_PLAN";

/// Does this value go to a blob column (a string above the threshold)?
fn va_en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
}

/// The Neo4j reading of a stored type: lowercased, an unknown value is `Other`.
fn type_depuis(s: &str) -> ConstraintType {
    match s.to_lowercase().as_str() {
        "performance" => ConstraintType::Performance,
        "compatibility" => ConstraintType::Compatibility,
        "security" => ConstraintType::Security,
        "style" => ConstraintType::Style,
        "testing" => ConstraintType::Testing,
        _ => ConstraintType::Other,
    }
}

/// The domain value of a `Constraint` node. `id`, `constraint_type` and
/// `description` are required, as in Neo4j's `node.get(..)?`.
fn lire_contrainte(get: &dyn Fn(&str) -> Option<Value>) -> Result<ConstraintNode> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let id: Uuid = chaine("id")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("nœud Constraint sans propriété `id` lisible"))?;
    let requise =
        |cle: &str| chaine(cle).ok_or_else(|| anyhow!("contrainte {id} sans propriété `{cle}`"));
    Ok(ConstraintNode {
        id,
        constraint_type: type_depuis(&requise("constraint_type")?),
        description: requise("description")?,
        enforced_by: chaine("enforced_by").filter(|s| !s.is_empty()),
    })
}

/// The edges of type `type_arete` arriving at `noeud`, with their source.
fn aretes_entrantes(db: &ObrainDB, noeud: NodeId, type_arete: &str) -> Vec<(EdgeId, NodeId)> {
    db.store()
        .edges_from(noeud, Direction::Incoming)
        .into_iter()
        .filter_map(|(_, a)| {
            let e = db.get_edge(a)?;
            (&*e.edge_type == type_arete && e.dst == noeud).then_some((a, e.src))
        })
        .collect()
}

/// Every edge of `noeud`, both directions (what a `DETACH DELETE` removes).
fn toutes_les_aretes(db: &ObrainDB, noeud: NodeId) -> Vec<EdgeId> {
    let store = db.store();
    let mut aretes: Vec<EdgeId> = store
        .edges_from(noeud, Direction::Outgoing)
        .into_iter()
        .chain(store.edges_from(noeud, Direction::Incoming))
        .map(|(_, a)| a)
        .collect();
    aretes.sort_unstable();
    aretes.dedup();
    aretes
}

/// The `id` of a node carrying `label`, if readable.
fn id_du_noeud(db: &ObrainDB, noeud: NodeId, label: &str) -> Option<Uuid> {
    let labels = db.get_node_labels(noeud)?;
    if !labels.iter().any(|l| l.as_str() == label) {
        return None;
    }
    db.get_node(noeud)?
        .get_property("id")
        .and_then(|v| v.as_str().and_then(|s| s.parse().ok()))
}

/// The projects a plan node is linked to by a `HAS_PLAN` edge (its
/// `project_id` aside). Read by `plan::Entree` at each decoding. Only an edge
/// from the very NODE `projets` indexes counts: a quarantined `Project` node
/// that shares the id of an indexed one links nothing.
pub(super) fn projets_lies(
    db: &ObrainDB,
    noeud_plan: NodeId,
    projets: &project::IndexProjets,
) -> BTreeSet<Uuid> {
    aretes_entrantes(db, noeud_plan, A_LE_PLAN)
        .into_iter()
        .filter_map(|(_, src)| {
            let id = id_du_noeud(db, src, LABEL_PROJET)?;
            (projets.noeud_de(id) == Some(src)).then_some(id)
        })
        .collect()
}

/// Is there, in the base, a node `label` of this id that the indexes do not
/// address (a quarantined one)? A scan of that label: paid only on the miss
/// path of a link.
fn en_quarantaine(db: &ObrainDB, label: &str, id: Uuid) -> bool {
    db.store()
        .nodes_by_label(label)
        .into_iter()
        .any(|n| id_du_noeud(db, n, label) == Some(id))
}

/// Opening: finishes every unlink interrupted after its marker (module doc).
/// Runs BEFORE the plan index is rebuilt, so the plans decode finished.
pub(super) fn achever_les_deliaisons(db: &ObrainDB, rapport: &mut RapportOuverture) {
    for noeud in db.store().nodes_by_label(LABEL_PLAN) {
        let marque = db
            .get_node(noeud)
            .and_then(|n| n.get_property(DELIAISON).cloned())
            .is_some_and(|v| !matches!(v, Value::Null));
        if !marque {
            continue;
        }
        for (a, _) in aretes_entrantes(db, noeud, A_LE_PLAN) {
            db.delete_edge(a);
        }
        db.set_node_property(noeud, "project_id", Value::from(""));
        db.remove_node_property(noeud, DELIAISON);
        rapport.deliaisons_achevees += 1;
    }
}

/// One constraint: its node, its value, its plan (if this backend addresses it).
pub(super) struct EntreeContrainte {
    noeud: NodeId,
    contrainte: ConstraintNode,
    plan: Option<Uuid>,
}

impl EntreeContrainte {
    fn decoder(db: &ObrainDB, noeud: NodeId, plan: Option<Uuid>) -> Result<Self> {
        let n = db
            .get_node(noeud)
            .ok_or_else(|| anyhow!("nœud Constraint {noeud:?} indexé mais absent de la base"))?;
        let get = |cle: &str| {
            n.get_property(cle)
                .filter(|v| !matches!(v, Value::Null))
                .cloned()
        };
        Ok(Self {
            noeud,
            contrainte: lire_contrainte(&get)?,
            plan,
        })
    }
}

/// `Uuid → constraint`, rebuilt from the `Constraint` nodes at every open.
#[derive(Default)]
pub(super) struct IndexContraintes {
    par_id: HashMap<Uuid, EntreeContrainte>,
    /// Ids of quarantined nodes: no creation may take them.
    reserves: HashSet<Uuid>,
}

impl IndexContraintes {
    /// Rebuilds the index, after the plans: drafts are deleted with their edges,
    /// unreadable nodes quarantined (their id reserved), duplicates quarantined.
    pub(super) fn reconstruire(
        db: &ObrainDB,
        plans: &plan::IndexPlans,
        rapport: &mut RapportOuverture,
    ) -> Self {
        let store = db.store();
        let mut index = Self::default();
        for noeud in store.nodes_by_label(LABEL) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud Constraint {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            if get(BROUILLON).is_some() || n.properties.is_empty() {
                for a in toutes_les_aretes(db, noeud) {
                    db.delete_edge(a);
                }
                db.delete_node(noeud);
                rapport.brouillons_supprimes += 1;
                continue;
            }
            let contrainte = match lire_contrainte(&get) {
                Ok(c) => c,
                Err(e) => {
                    if let Some(id) =
                        get("id").and_then(|v| v.as_str().and_then(|s| s.parse().ok()))
                    {
                        index.reserves.insert(id);
                    }
                    rapport
                        .en_quarantaine
                        .push(format!("nœud Constraint {noeud:?} : {e}"));
                    continue;
                }
            };
            if index.par_id.contains_key(&contrainte.id) {
                // Reserved too: once the indexed one is deleted, a new creation of
                // this id would lose to this older node at the next open.
                index.reserves.insert(contrainte.id);
                rapport.en_quarantaine.push(format!(
                    "nœud Constraint {noeud:?} : id {} déjà pris",
                    contrainte.id
                ));
                continue;
            }
            // Held by the indexed plan NODE, not by a quarantined one of the same id.
            let plan = aretes_entrantes(db, noeud, CONTRAINT_PAR)
                .into_iter()
                .find_map(|(_, src)| {
                    let id = id_du_noeud(db, src, LABEL_PLAN)?;
                    (plans.par_id.get(&id).map(|e| e.noeud) == Some(src)).then_some(id)
                });
            index.par_id.insert(
                contrainte.id,
                EntreeContrainte {
                    noeud,
                    contrainte,
                    plan,
                },
            );
        }
        index
    }
}

impl ObrainGraphStore {
    pub(super) async fn creer_contrainte(
        &self,
        plan_id: Uuid,
        contrainte: &ConstraintNode,
    ) -> Result<()> {
        let c = contrainte.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // Validate first: nothing is written when one of these refuses.
                let noeud_plan = {
                    let etat = w.index()?;
                    let Some(p) = etat.plans.par_id.get(&plan_id) else {
                        bail!("Failed to create constraint: Plan {plan_id} not found");
                    };
                    if etat.contraintes.par_id.contains_key(&c.id)
                        || etat.contraintes.reserves.contains(&c.id)
                    {
                        bail!(
                            "une contrainte d'id {} existe déjà (unicité de Constraint.id)",
                            c.id
                        );
                    }
                    p.noeud
                };
                let proprietes = [
                    ("id", Value::from(c.id.to_string())),
                    (
                        "constraint_type",
                        Value::from(c.constraint_type.to_string()),
                    ),
                    ("description", Value::from(c.description.clone())),
                    (
                        "enforced_by",
                        Value::from(c.enforced_by.clone().unwrap_or_default()),
                    ),
                ];
                // A value past the blob threshold is not recovered by the WAL: the
                // full flush runs before the marker goes (as in plan.rs / note.rs).
                w.flush_complet = proprietes.iter().any(|(_, v)| va_en_colonne_blob(v));
                let noeud = w.db.create_node(&[LABEL]);
                w.etape()?;
                w.db.set_node_property(noeud, BROUILLON, Value::from(true));
                w.etape()?;
                for (cle, valeur) in proprietes {
                    w.db.set_node_property(noeud, cle, valeur);
                    w.etape()?;
                }
                w.db.create_edge(noeud_plan, noeud, CONTRAINT_PAR);
                w.etape()?;
                w.rendre_durable_avant_le_marqueur()?;
                // The commit point.
                w.db.remove_node_property(noeud, BROUILLON);
                let entree = w.apres_ecriture(
                    EntreeContrainte::decoder(w.db, noeud, Some(plan_id)).and_then(|e| {
                        if e.contrainte.id == c.id {
                            Ok(e)
                        } else {
                            Err(anyhow!("la contrainte relue n'a pas l'id écrit"))
                        }
                    }),
                )?;
                w.rendre_durable()?;
                w.index_mut()?.contraintes.par_id.insert(c.id, entree);
                Ok(())
            },
        )
        .await
    }

    pub(super) fn contrainte_par_id(&self, id: Uuid) -> Result<Option<ConstraintNode>> {
        self.lire(|_, etat| {
            Ok(etat
                .contraintes
                .par_id
                .get(&id)
                .map(|e| e.contrainte.clone()))
        })
    }

    pub(super) fn contraintes_du_plan(&self, plan_id: Uuid) -> Result<Vec<ConstraintNode>> {
        self.lire(|_, etat| {
            let mut entrees: Vec<&EntreeContrainte> = etat
                .contraintes
                .par_id
                .values()
                .filter(|e| e.plan == Some(plan_id))
                .collect();
            entrees.sort_by_key(|e| e.noeud);
            Ok(entrees.into_iter().map(|e| e.contrainte.clone()).collect())
        })
    }

    pub(super) async fn modifier_contrainte(
        &self,
        id: Uuid,
        description: Option<String>,
        constraint_type: Option<ConstraintType>,
        enforced_by: Option<String>,
    ) -> Result<()> {
        // Nothing to set: nothing is written (Neo4j answers the same).
        if description.is_none() && constraint_type.is_none() && enforced_by.is_none() {
            return Ok(());
        }
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // A missing constraint is a silent no-op, as in Neo4j.
                let Some((noeud, plan, ancien_en_blob)) =
                    w.index()?.contraintes.par_id.get(&id).map(|e| {
                        let c = &e.contrainte;
                        let ancien = c.description.len() > SEUIL_COLONNE_BLOB
                            || c.enforced_by
                                .as_ref()
                                .is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB);
                        (e.noeud, e.plan, ancien)
                    })
                else {
                    return Ok(());
                };
                // The OLD values too: replacing a value that sat in a blob column
                // clears that slot, which the WAL does not recover (as in plan.rs).
                w.flush_complet = ancien_en_blob
                    || description
                        .iter()
                        .chain(enforced_by.iter())
                        .any(|s| s.len() > SEUIL_COLONNE_BLOB);
                if let Some(d) = description {
                    w.db.set_node_property(noeud, "description", Value::from(d));
                    w.etape()?;
                }
                if let Some(t) = constraint_type {
                    w.db.set_node_property(noeud, "constraint_type", Value::from(t.to_string()));
                    w.etape()?;
                }
                if let Some(e) = enforced_by {
                    w.db.set_node_property(noeud, "enforced_by", Value::from(e));
                    w.etape()?;
                }
                let entree = w.apres_ecriture(EntreeContrainte::decoder(w.db, noeud, plan))?;
                w.rendre_durable()?;
                w.index_mut()?.contraintes.par_id.insert(id, entree);
                Ok(())
            },
        )
        .await
    }

    pub(super) async fn supprimer_contrainte(&self, id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = w.index()?.contraintes.par_id.get(&id).map(|e| e.noeud) else {
                    return Ok(());
                };
                // DETACH DELETE: the edges first, then the node.
                for a in toutes_les_aretes(w.db, noeud) {
                    w.db.delete_edge(a);
                    w.etape()?;
                }
                if !w.db.delete_node(noeud) {
                    w.interne
                        .indetermine
                        .store(true, std::sync::atomic::Ordering::Release);
                    bail!(
                        "obrain a refusé la suppression du nœud {noeud:?} (contrainte {id}) APRÈS \
                         la suppression de ses arêtes : base indéterminée, redémarrer"
                    );
                }
                w.etape()?;
                w.rendre_durable()?;
                w.index_mut()?.contraintes.par_id.remove(&id);
                Ok(())
            },
        )
        .await
    }

    pub(super) async fn lier_plan_au_projet(&self, plan_id: Uuid, projet: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud_plan, noeud_projet, ancien) = {
                    let etat = w.index()?;
                    // `MATCH` on both: one absent from the BASE is a silent no-op, as
                    // in Neo4j. One the base holds but this backend quarantined is
                    // refused: Neo4j would find it and link it.
                    let (Some(p), Some(n)) = (
                        etat.plans.par_id.get(&plan_id),
                        etat.projets.noeud_de(projet),
                    ) else {
                        let plan_absent = !etat.plans.par_id.contains_key(&plan_id);
                        let projet_absent = !etat.projets.contient(projet);
                        if plan_absent && en_quarantaine(w.db, LABEL_PLAN, plan_id) {
                            bail!(
                                "link_plan_to_project : le plan {plan_id} est dans la base mais \
                                 en quarantaine sur ce backend (nœud illisible ou id en double, \
                                 voir le rapport d'ouverture) : refusé plutôt qu'ignoré"
                            );
                        }
                        if projet_absent && en_quarantaine(w.db, LABEL_PROJET, projet) {
                            bail!(
                                "link_plan_to_project : le projet {projet} est dans la base mais \
                                 en quarantaine sur ce backend (nœud illisible ou id en double, \
                                 voir le rapport d'ouverture) : refusé plutôt qu'ignoré"
                            );
                        }
                        return Ok(());
                    };
                    let ancien = p
                        .plan
                        .project_id
                        .filter(|q| *q != projet && !p.lies.contains(q))
                        .and_then(|q| etat.projets.noeud_de(q));
                    (p.noeud, n, ancien)
                };
                let sources: HashSet<NodeId> = aretes_entrantes(w.db, noeud_plan, A_LE_PLAN)
                    .into_iter()
                    .map(|(_, s)| s)
                    .collect();
                let mut ecrit = false;
                // (1) The previous project, held only by the property, gets its edge.
                if let Some(q) = ancien.filter(|q| !sources.contains(q)) {
                    w.db.create_edge(q, noeud_plan, A_LE_PLAN);
                    w.etape()?;
                    ecrit = true;
                }
                // (2) `SET plan.project_id`.
                let actuel =
                    w.db.get_node(noeud_plan)
                        .and_then(|n| n.get_property("project_id").cloned())
                        .and_then(|v| v.as_str().map(str::to_string));
                if actuel.as_deref() != Some(projet.to_string().as_str()) {
                    w.db.set_node_property(
                        noeud_plan,
                        "project_id",
                        Value::from(projet.to_string()),
                    );
                    w.etape()?;
                    ecrit = true;
                }
                // (3) `MERGE (project)-[:HAS_PLAN]->(plan)`: never a second edge.
                if !sources.contains(&noeud_projet) {
                    w.db.create_edge(noeud_projet, noeud_plan, A_LE_PLAN);
                    w.etape()?;
                    ecrit = true;
                }
                if !ecrit {
                    return Ok(());
                }
                let entree = w.apres_ecriture(plan::Entree::relire(w, noeud_plan))?;
                w.rendre_durable()?;
                w.index_mut()?.plans.par_id.insert(plan_id, entree);
                Ok(())
            },
        )
        .await
    }

    pub(super) async fn delier_plan(&self, plan_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud_plan = {
                    let etat = w.index()?;
                    let Some(p) = etat.plans.par_id.get(&plan_id) else {
                        return Ok(());
                    };
                    // Neo4j reaches the `SET` only when a `HAS_PLAN` matched.
                    if p.lies.is_empty() && p.plan.project_id.is_none() {
                        return Ok(());
                    }
                    p.noeud
                };
                w.db.set_node_property(noeud_plan, DELIAISON, Value::from(true));
                w.etape()?;
                for (a, _) in aretes_entrantes(w.db, noeud_plan, A_LE_PLAN) {
                    w.db.delete_edge(a);
                    w.etape()?;
                }
                w.db.set_node_property(noeud_plan, "project_id", Value::from(""));
                w.etape()?;
                w.db.remove_node_property(noeud_plan, DELIAISON);
                w.etape()?;
                let entree = w.apres_ecriture(plan::Entree::relire(w, noeud_plan))?;
                w.rendre_durable()?;
                w.index_mut()?.plans.par_id.insert(plan_id, entree);
                Ok(())
            },
        )
        .await
    }

    pub(super) fn slugs_des_projets_du_plan(&self, plan_id: Uuid) -> Result<Vec<String>> {
        self.lire(|_, etat| {
            let Some(p) = etat.plans.par_id.get(&plan_id) else {
                return Ok(Vec::new());
            };
            let mut slugs: Vec<String> = p
                .projets()
                .into_iter()
                .filter_map(|id| etat.projets.projet(id).map(|x| x.slug.clone()))
                .collect();
            slugs.sort();
            slugs.dedup();
            Ok(slugs)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::models::{PlanNode, PlanStatus, ProjectNode};
    use crate::neo4j::traits::{PlanStore, ProjectStore};
    use crate::obrain::{Creation, Interruption, NonPorte};
    use chrono::Utc;

    fn projet(nom: &str) -> ProjectNode {
        let id = Uuid::new_v4();
        ProjectNode {
            id,
            name: format!("{nom} {id}"),
            slug: format!("{nom}-{id}"),
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

    fn un_plan(projet: Option<Uuid>) -> PlanNode {
        PlanNode {
            id: Uuid::new_v4(),
            title: "plan".into(),
            description: "d".into(),
            status: PlanStatus::Draft,
            created_at: Utc::now(),
            created_by: "test".into(),
            priority: 1,
            project_id: projet,
            execution_context: None,
            persona: None,
        }
    }

    fn contrainte(desc: &str) -> ConstraintNode {
        ConstraintNode {
            id: Uuid::new_v4(),
            constraint_type: ConstraintType::Security,
            description: desc.into(),
            enforced_by: Some("clippy".into()),
        }
    }

    /// The `HAS_PLAN` edges of a plan in the engine, whatever the index says.
    fn aretes_has_plan(store: &ObrainGraphStore, plan_id: Uuid) -> usize {
        let noeud = store.interne.etat.read().unwrap().plans.par_id[&plan_id].noeud;
        aretes_entrantes(&store.interne.db, noeud, A_LE_PLAN).len()
    }

    #[tokio::test]
    async fn contraintes_cycle_de_vie_et_reouverture_apres_close() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = un_plan(None);
        let (a, b) = (contrainte("a"), contrainte("b"));
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_plan(&p).await.unwrap();
            store.create_constraint(p.id, &a).await.unwrap();
            store.create_constraint(p.id, &b).await.unwrap();
            // Id taken: refused; missing plan: refused.
            assert!(store.create_constraint(p.id, &a).await.is_err());
            assert!(store
                .create_constraint(Uuid::new_v4(), &contrainte("x"))
                .await
                .is_err());
            store
                .update_constraint(
                    a.id,
                    Some("a2".into()),
                    Some(ConstraintType::Performance),
                    Some(String::new()),
                )
                .await
                .unwrap();
            // Missing: silent no-ops.
            store
                .update_constraint(Uuid::new_v4(), Some("z".into()), None, None)
                .await
                .unwrap();
            store.delete_constraint(Uuid::new_v4()).await.unwrap();
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(store.rapport_d_ouverture(), &RapportOuverture::default());
        let lues = store.get_plan_constraints(p.id).await.unwrap();
        assert_eq!(
            lues.iter().map(|c| c.id).collect::<Vec<_>>(),
            vec![a.id, b.id]
        );
        let a2 = store.get_constraint(a.id).await.unwrap().unwrap();
        assert_eq!(a2.description, "a2");
        assert_eq!(a2.constraint_type, ConstraintType::Performance);
        assert_eq!(a2.enforced_by, None, "an empty enforced_by reads None");
        assert_eq!(
            store
                .get_constraint(b.id)
                .await
                .unwrap()
                .unwrap()
                .enforced_by,
            Some("clippy".into())
        );
        store.delete_constraint(a.id).await.unwrap();
        assert!(store.get_constraint(a.id).await.unwrap().is_none());
        assert_eq!(store.get_plan_constraints(p.id).await.unwrap().len(), 1);
        store.close().await.unwrap();
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert!(store.get_constraint(a.id).await.unwrap().is_none());
        assert_eq!(store.get_plan_constraints(p.id).await.unwrap().len(), 1);
    }

    /// A creation interrupted at each of its engine writes, the store dropped
    /// WITHOUT close: the reopen deletes the draft (nothing half-made is served),
    /// and the creation can be made again.
    #[tokio::test]
    async fn creation_interrompue_sans_close_est_un_brouillon_supprime() {
        let dir = tempfile::tempdir().unwrap();
        let mut atteints = 0;
        for k in 1..=10 {
            let chemin = dir.path().join(format!("base-{k}"));
            let p = un_plan(None);
            let c = contrainte("c");
            {
                let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
                store.create_plan(&p).await.unwrap();
                store.interrompre_au_prochain(Interruption::AvantEtape(k));
                if store.create_constraint(p.id, &c).await.is_ok() {
                    break;
                }
                atteints += 1;
                assert!(store.get_constraint(c.id).await.unwrap().is_none());
                // No close.
            }
            let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
            assert_eq!(store.rapport_d_ouverture().brouillons_supprimes, 1, "k={k}");
            assert!(store.get_constraint(c.id).await.unwrap().is_none(), "k={k}");
            store.create_constraint(p.id, &c).await.unwrap();
            assert_eq!(store.get_plan_constraints(p.id).await.unwrap().len(), 1);
            store.close().await.unwrap();
        }
        assert_eq!(atteints, 7, "node, marker, 4 properties, edge");
    }

    /// A constraint node with no readable `description` is quarantined, its id
    /// reserved.
    #[tokio::test]
    async fn contrainte_illisible_en_quarantaine_et_id_reserve() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = un_plan(None);
        let c = contrainte("c");
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_plan(&p).await.unwrap();
            store.create_constraint(p.id, &c).await.unwrap();
            let noeud = store.interne.etat.read().unwrap().contraintes.par_id[&c.id].noeud;
            store.interne.db.remove_node_property(noeud, "description");
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
        assert!(store.get_constraint(c.id).await.unwrap().is_none());
        assert!(store.get_plan_constraints(p.id).await.unwrap().is_empty());
        let err = store.create_constraint(p.id, &c).await.unwrap_err();
        assert!(format!("{err:#}").contains("existe déjà"), "{err:#}");
    }

    #[tokio::test]
    async fn double_liaison_une_seule_arete_et_portee_des_lectures() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let (a, b) = (projet("a"), projet("b"));
        store.create_project(&a).await.unwrap();
        store.create_project(&b).await.unwrap();
        let p = un_plan(Some(a.id));
        store.create_plan(&p).await.unwrap();
        assert_eq!(
            aretes_has_plan(&store, p.id),
            0,
            "create_plan: the property"
        );

        store.link_plan_to_project(p.id, b.id).await.unwrap();
        store.link_plan_to_project(p.id, b.id).await.unwrap();
        // a (materialized from the property) + b, once.
        assert_eq!(aretes_has_plan(&store, p.id), 2);
        store.link_plan_to_project(p.id, a.id).await.unwrap();
        assert_eq!(aretes_has_plan(&store, p.id), 2, "MERGE: no second edge");
        assert_eq!(
            store.get_plan(p.id).await.unwrap().unwrap().project_id,
            Some(a.id)
        );
        // Missing plan or project: no-op.
        store
            .link_plan_to_project(Uuid::new_v4(), a.id)
            .await
            .unwrap();
        store
            .link_plan_to_project(p.id, Uuid::new_v4())
            .await
            .unwrap();
        assert_eq!(aretes_has_plan(&store, p.id), 2);

        // Both projects see the plan in their scoped reads.
        for proj in [a.id, b.id] {
            assert_eq!(store.count_project_plans(proj).await.unwrap(), 1);
            assert_eq!(store.list_project_plans(proj).await.unwrap().len(), 1);
            assert_eq!(
                store
                    .list_plans_for_project(proj, None, 10, 0)
                    .await
                    .unwrap()
                    .1,
                1
            );
        }
        let mut attendus = vec![a.slug.clone(), b.slug.clone()];
        attendus.sort();
        assert_eq!(store.list_plan_project_slugs(p.id).await.unwrap(), attendus);
        assert!(store
            .list_plan_project_slugs(Uuid::new_v4())
            .await
            .unwrap()
            .is_empty());

        // delete_project refuses for EACH project the plan is linked to.
        for proj in [&a, &b] {
            let err = store.delete_project(proj.id, &proj.name).await.unwrap_err();
            assert_eq!(
                err.downcast_ref::<NonPorte>(),
                Some(&NonPorte("ProjectStore::delete_project (cascade)"))
            );
        }

        store.unlink_plan_from_project(p.id).await.unwrap();
        assert_eq!(aretes_has_plan(&store, p.id), 0);
        assert_eq!(
            store.get_plan(p.id).await.unwrap().unwrap().project_id,
            None
        );
        assert!(store
            .list_plan_project_slugs(p.id)
            .await
            .unwrap()
            .is_empty());
        for proj in [a.id, b.id] {
            assert_eq!(store.count_project_plans(proj).await.unwrap(), 0);
        }
        // Unlinking an unlinked plan: nothing.
        store.unlink_plan_from_project(p.id).await.unwrap();
        // Now the projects go.
        store.delete_project(a.id, &a.name).await.unwrap();
        store.delete_project(b.id, &b.name).await.unwrap();
    }

    #[tokio::test]
    async fn liens_survivent_a_la_reouverture() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let (a, b) = (projet("a"), projet("b"));
        let p = un_plan(Some(a.id));
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_project(&a).await.unwrap();
            store.create_project(&b).await.unwrap();
            store.create_plan(&p).await.unwrap();
            store.link_plan_to_project(p.id, b.id).await.unwrap();
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(aretes_has_plan(&store, p.id), 2);
        assert_eq!(store.list_plan_project_slugs(p.id).await.unwrap().len(), 2);
        assert_eq!(store.count_project_plans(a.id).await.unwrap(), 1);
        assert_eq!(
            store.get_plan(p.id).await.unwrap().unwrap().project_id,
            Some(b.id)
        );
    }

    /// A link interrupted after each of its three writes, dropped WITHOUT close.
    /// The plan was created in `a` (property only, no edge); linking it to `b`
    /// writes (1) the edge of `a`, (2) `project_id = b`, (3) the edge of `b`.
    /// Each surviving prefix is checked exactly: it reads as "before" (k=1) or
    /// as "after" (k=2, 3), never as a third state. Then the link is replayed and
    /// MERGE leaves exactly two edges.
    #[tokio::test]
    async fn liaison_interrompue_avant_ou_apres() {
        let dir = tempfile::tempdir().unwrap();
        for k in 1..=3 {
            let chemin = dir.path().join(format!("base-{k}"));
            let (a, b) = (projet("a"), projet("b"));
            let p = un_plan(Some(a.id));
            {
                let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
                store.create_project(&a).await.unwrap();
                store.create_project(&b).await.unwrap();
                store.create_plan(&p).await.unwrap();
                store.interrompre_au_prochain(Interruption::AvantEtape(k));
                let issue = store.link_plan_to_project(p.id, b.id).await;
                assert!(issue.is_err(), "k={k}: the hook must have fired");
                assert!(format!("{:#}", issue.unwrap_err()).contains("interruption injectée"));
            }
            let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
            let lu = store.get_plan(p.id).await.unwrap().unwrap();
            let slugs = store.list_plan_project_slugs(p.id).await.unwrap();
            let mut les_deux = vec![a.slug.clone(), b.slug.clone()];
            les_deux.sort();
            let (aretes, project_id, attendus) = match k {
                // (1) only: the edge of `a`, same membership as before.
                1 => (1, Some(a.id), vec![a.slug.clone()]),
                // (2): `project_id = b`, the membership is the final one.
                2 => (1, Some(b.id), les_deux.clone()),
                // (3): everything written, only the acknowledgement missing.
                _ => (2, Some(b.id), les_deux.clone()),
            };
            assert_eq!(aretes_has_plan(&store, p.id), aretes, "k={k}");
            assert_eq!(lu.project_id, project_id, "k={k}");
            assert_eq!(slugs, attendus, "k={k}");
            assert_eq!(
                store.count_project_plans(b.id).await.unwrap(),
                i64::from(k >= 2),
                "k={k}"
            );
            store.link_plan_to_project(p.id, b.id).await.unwrap();
            assert_eq!(aretes_has_plan(&store, p.id), 2, "k={k}");
            assert_eq!(store.list_plan_project_slugs(p.id).await.unwrap(), les_deux);
            store.close().await.unwrap();
        }
    }

    /// An unlink interrupted after its marker, dropped WITHOUT close: opening
    /// finishes it.
    #[tokio::test]
    async fn deliaison_interrompue_est_achevee_a_l_ouverture() {
        let dir = tempfile::tempdir().unwrap();
        for k in 1..=4 {
            let chemin = dir.path().join(format!("base-{k}"));
            let (a, b) = (projet("a"), projet("b"));
            let p = un_plan(Some(a.id));
            {
                let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
                store.create_project(&a).await.unwrap();
                store.create_project(&b).await.unwrap();
                store.create_plan(&p).await.unwrap();
                store.link_plan_to_project(p.id, b.id).await.unwrap();
                store.interrompre_au_prochain(Interruption::AvantEtape(k));
                assert!(store.unlink_plan_from_project(p.id).await.is_err(), "k={k}");
            }
            let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
            assert_eq!(store.rapport_d_ouverture().deliaisons_achevees, 1, "k={k}");
            assert!(store
                .list_plan_project_slugs(p.id)
                .await
                .unwrap()
                .is_empty());
            assert_eq!(aretes_has_plan(&store, p.id), 0, "k={k}");
            assert_eq!(
                store.get_plan(p.id).await.unwrap().unwrap().project_id,
                None
            );
            store.close().await.unwrap();
            // The repair was durable: a second open finds nothing to do.
            let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
            assert_eq!(store.rapport_d_ouverture().deliaisons_achevees, 0, "k={k}");
            store.close().await.unwrap();
        }
    }

    /// Copies a node, its labels and every property, as a base written by
    /// someone else could hold it (two nodes with one id). Raw engine writes.
    fn dupliquer(store: &ObrainGraphStore, noeud: NodeId, label: &str) -> NodeId {
        let db = &store.interne.db;
        let n = db.get_node(noeud).unwrap();
        let copie = db.create_node(&[label]);
        for (k, v) in n.properties.iter() {
            db.set_node_property(copie, k.as_ref(), v.clone());
        }
        copie
    }

    fn noeud_projet(store: &ObrainGraphStore, id: Uuid) -> NodeId {
        store
            .interne
            .etat
            .read()
            .unwrap()
            .projets
            .noeud_de(id)
            .unwrap()
    }

    /// Review #1210, point 1: a constraint hangs from a plan NODE. Two `Plan`
    /// nodes share an id, the second is quarantined: a constraint drawn from the
    /// quarantined one is not served as the indexed plan's.
    #[tokio::test]
    async fn contrainte_d_un_plan_en_quarantaine_n_est_pas_celle_du_plan_indexe() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = un_plan(None);
        let c = contrainte("du double");
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_plan(&p).await.unwrap();
            let original = store.interne.etat.read().unwrap().plans.par_id[&p.id].noeud;
            let double = dupliquer(&store, original, LABEL_PLAN);
            let db = &store.interne.db;
            let n = db.create_node(&[LABEL]);
            db.set_node_property(n, "id", Value::from(c.id.to_string()));
            db.set_node_property(n, "constraint_type", Value::from("security"));
            db.set_node_property(n, "description", Value::from("du double"));
            db.create_edge(double, n, CONTRAINT_PAR);
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
        assert!(
            store.get_plan_constraints(p.id).await.unwrap().is_empty(),
            "the constraint of the quarantined node is not the indexed plan's"
        );
        assert!(store.get_constraint(c.id).await.unwrap().is_some());
    }

    /// Review #1210, point 2: a `HAS_PLAN` edge from a QUARANTINED `Project`
    /// node that shares the id of an indexed project does not put the plan in
    /// that project (reads and the `delete_project` guard).
    #[tokio::test]
    async fn arete_d_un_projet_en_quarantaine_ne_rattache_pas_le_plan() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let a = projet("a");
        let p = un_plan(None);
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_project(&a).await.unwrap();
            store.create_plan(&p).await.unwrap();
            let double = dupliquer(&store, noeud_projet(&store, a.id), LABEL_PROJET);
            let plan = store.interne.etat.read().unwrap().plans.par_id[&p.id].noeud;
            store.interne.db.create_edge(double, plan, A_LE_PLAN);
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
        assert_eq!(store.count_project_plans(a.id).await.unwrap(), 0);
        assert!(store.list_project_plans(a.id).await.unwrap().is_empty());
        assert!(store
            .list_plan_project_slugs(p.id)
            .await
            .unwrap()
            .is_empty());
        // The guard is not blocked by a plan that is not the project's (the
        // quarantined node still counts as a foreign node: refused for THAT reason).
        let err = store.delete_project(a.id, &a.name).await.unwrap_err();
        assert!(!format!("{err:#}").contains("rattaché"), "{err:#}");
    }

    /// Review #1210, point 3: a duplicate constraint id quarantined at open is
    /// reserved: once the indexed one is deleted, the id cannot be reused (the
    /// old quarantined node would win the next open and hide the new one).
    #[tokio::test]
    async fn doublon_de_contrainte_reserve_son_id() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = un_plan(None);
        let c = contrainte("c");
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_plan(&p).await.unwrap();
            store.create_constraint(p.id, &c).await.unwrap();
            let n = store.interne.etat.read().unwrap().contraintes.par_id[&c.id].noeud;
            dupliquer(&store, n, LABEL);
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
        store.delete_constraint(c.id).await.unwrap();
        let err = store.create_constraint(p.id, &c).await.unwrap_err();
        assert!(format!("{err:#}").contains("existe déjà"), "{err:#}");
    }

    /// Review #1210, point 5: linking to a project, or a plan, that the base
    /// holds but this backend quarantined is REFUSED by name (Neo4j's `MATCH`
    /// would find the node and link it); an id the base does not hold at all is
    /// still a silent no-op.
    #[tokio::test]
    async fn liaison_a_un_noeud_en_quarantaine_est_refusee() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let (a, b) = (projet("a"), projet("b"));
        let (p, q) = (un_plan(None), un_plan(None));
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_project(&a).await.unwrap();
            store.create_project(&b).await.unwrap();
            store.create_plan(&p).await.unwrap();
            store.create_plan(&q).await.unwrap();
            let db = &store.interne.db;
            db.remove_node_property(noeud_projet(&store, b.id), "name");
            let nq = store.interne.etat.read().unwrap().plans.par_id[&q.id].noeud;
            db.remove_node_property(nq, "title");
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 2);
        let err = store.link_plan_to_project(p.id, b.id).await.unwrap_err();
        assert!(format!("{err:#}").contains("quarantaine"), "{err:#}");
        let err = store.link_plan_to_project(q.id, a.id).await.unwrap_err();
        assert!(format!("{err:#}").contains("quarantaine"), "{err:#}");
        assert!(store
            .list_plan_project_slugs(p.id)
            .await
            .unwrap()
            .is_empty());
        // Absent from the base: Neo4j's MATCH finds nothing either.
        store
            .link_plan_to_project(p.id, Uuid::new_v4())
            .await
            .unwrap();
        store
            .link_plan_to_project(Uuid::new_v4(), a.id)
            .await
            .unwrap();
    }

    /// Review #1210, point 6: a plan still linked to two projects is deleted
    /// (DETACH DELETE), then both projects go, and nothing comes back at open.
    #[tokio::test]
    async fn suppression_d_un_plan_encore_lie_a_deux_projets() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let (a, b) = (projet("a"), projet("b"));
        let p = un_plan(Some(a.id));
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_project(&a).await.unwrap();
            store.create_project(&b).await.unwrap();
            store.create_plan(&p).await.unwrap();
            store.link_plan_to_project(p.id, b.id).await.unwrap();
            store.delete_plan(p.id).await.unwrap();
            assert!(store.get_plan(p.id).await.unwrap().is_none());
            for proj in [&a, &b] {
                assert_eq!(store.count_project_plans(proj.id).await.unwrap(), 0);
                store.delete_project(proj.id, &proj.name).await.unwrap();
            }
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(store.rapport_d_ouverture(), &RapportOuverture::default());
        assert!(store.get_plan(p.id).await.unwrap().is_none());
        for proj in [&a, &b] {
            assert!(store.get_project(proj.id).await.unwrap().is_none());
        }
        assert_eq!(store.interne.db.store().node_count(), 0);
    }

    /// Decision e5713801: a constraint carrying a value past the blob threshold
    /// pays the FULL flush, at creation and at update (new value, or the old one
    /// it replaces). Observed without any close, through the engine's counter of
    /// fsyncs: a full flush runs more of them than the WAL fsync alone.
    #[tokio::test]
    async fn une_valeur_en_colonne_blob_paie_le_flush_complet() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let fsyncs = || store.fsyncs_du_wal().unwrap();
        let p = un_plan(None);
        store.create_plan(&p).await.unwrap();
        let court = contrainte(&"x".repeat(SEUIL_COLONNE_BLOB));
        let long = contrainte(&"x".repeat(SEUIL_COLONNE_BLOB + 1));

        let t = fsyncs();
        store.create_constraint(p.id, &court).await.unwrap();
        let cout_court = fsyncs() - t;
        let t = fsyncs();
        store.create_constraint(p.id, &long).await.unwrap();
        let cout_long = fsyncs() - t;
        assert!(cout_court >= 1, "the WAL fsync ran");
        assert!(
            cout_long > cout_court,
            "create: {cout_long} fsyncs for 257 bytes against {cout_court} for 256"
        );

        // Update: a short value over a short one is the baseline.
        let maj = |id: Uuid, d: String| store.update_constraint(id, Some(d), None, None);
        let t = fsyncs();
        maj(court.id, "y".into()).await.unwrap();
        let base = fsyncs() - t;
        // A short value REPLACING a long one: the blob slot is cleared.
        let t = fsyncs();
        maj(long.id, "y".into()).await.unwrap();
        let remplace = fsyncs() - t;
        assert!(
            remplace > base,
            "update over a blob: {remplace} against {base}"
        );
        // A long value written: the blob is new.
        let t = fsyncs();
        maj(court.id, "z".repeat(SEUIL_COLONNE_BLOB + 1))
            .await
            .unwrap();
        let ecrit = fsyncs() - t;
        assert!(ecrit > base, "update to a blob: {ecrit} against {base}");
        // A long enforced_by counts too.
        let t = fsyncs();
        store
            .update_constraint(
                long.id,
                None,
                None,
                Some("e".repeat(SEUIL_COLONNE_BLOB + 1)),
            )
            .await
            .unwrap();
        let par = fsyncs() - t;
        assert!(par > base, "long enforced_by: {par} against {base}");
        assert_eq!(
            store
                .get_constraint(court.id)
                .await
                .unwrap()
                .unwrap()
                .description
                .len(),
            SEUIL_COLONNE_BLOB + 1
        );
    }

    /// The cascade of `delete_plan` to its constraints is not ported: a plan that
    /// holds one is refused by name (its outgoing `CONSTRAINED_BY`), left whole;
    /// once its constraints are deleted, it goes.
    #[tokio::test]
    async fn supprimer_un_plan_qui_a_une_contrainte_est_refuse_par_son_nom() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let (p, autre) = (un_plan(None), un_plan(None));
        store.create_plan(&p).await.unwrap();
        store.create_plan(&autre).await.unwrap();
        let c = contrainte("c");
        store.create_constraint(p.id, &c).await.unwrap();
        let err = store.delete_plan(p.id).await.unwrap_err();
        assert_eq!(
            err.downcast_ref::<NonPorte>(),
            Some(&NonPorte("PlanStore::delete_plan (cascade)"))
        );
        assert!(store.get_plan(p.id).await.unwrap().is_some());
        assert_eq!(store.get_plan_constraints(p.id).await.unwrap().len(), 1);
        // A plan with no constraint goes, whatever the others hold.
        store.delete_plan(autre.id).await.unwrap();
        store.delete_constraint(c.id).await.unwrap();
        store.delete_plan(p.id).await.unwrap();
        assert!(store.get_plan(p.id).await.unwrap().is_none());
    }
}
