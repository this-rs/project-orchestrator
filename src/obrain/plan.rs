//! `PlanStore` on obrain: the core of the plan domain (lot O-PLAN-1).
//!
//! PORTED here, with the Neo4j semantics of `src/neo4j/plan.rs`:
//! - `create_plan`, `get_plan`, `list_active_plans`, `list_project_plans`,
//!   `count_project_plans`, `list_plans_for_project`;
//! - `update_plan`, `update_plan_status`;
//! - `delete_plan`, for a plan that holds nothing else (see below).
//!
//! PORTED in `constraint.rs` (this `impl` calls it): the constraints of a plan,
//! and the HAS_PLAN links of a plan to its projects (`link_plan_to_project`,
//! `unlink_plan_from_project`, `list_plan_project_slugs`).
//!
//! REFUSED by name (`NonPorte`): everything that touches tasks, steps, the day
//! plan, dependencies, the graphs and waves, and the enrichment. They are
//! refused, not answered empty.
//!
//! # Layout
//!
//! A plan is one node labelled `Plan`, with the properties Neo4j writes:
//! `id`, `title`, `description`, `status` (the Rust variant name, as Neo4j
//! does: `Draft`, `InProgress`…), `created_at` (RFC 3339 string), `created_by`,
//! `priority`, `project_id` (the empty string for none).
//!
//! The status is read as Neo4j reads it, in two ways: DECODED through
//! `pascal_to_snake_case` (`InProgress` and `in_progress` are both
//! `InProgress`, an unknown value is `Draft`), and FILTERED on the stored
//! string (`list_active_plans` and the project lists keep `Draft`, `Approved`,
//! `InProgress` as stored; `list_plans_for_project` compares the stored string
//! with `plan_status_values`). A migrated `in_progress` therefore decodes as
//! `InProgress` and is not active, exactly as on Neo4j.
//!
//! Membership in a project is the `project_id` property PLUS the incoming
//! `HAS_PLAN` edges from the project NODES the index addresses
//! ([`Entree::appartient_a`], `constraint.rs`); the project lists also require
//! the project to exist (Neo4j's `MATCH (project:Project {id})`). Neo4j stores
//! the membership as `HAS_PLAN` edges and keeps the property in step with the
//! LAST project linked (`link_plan_to_project` overwrites it, keeps the previous
//! edges; `unlink_plan_from_project` empties both). `create_plan` here writes the
//! property alone, which stands for the edge Neo4j's `create_plan` draws;
//! `link_plan_to_project` draws the edges. A migrated plan linked to A then to B
//! (`project_id = B`, two edges) is therefore served under A and B, as on Neo4j.
//!
//! # Divergences, all on the strict side
//!
//! - `create_plan` with a `project_id` whose project does not exist is REFUSED.
//!   Neo4j's `MATCH` on the project silently creates a plan with no link.
//! - `delete_plan` deletes only a plan with no OUTGOING edge: the Neo4j cascade
//!   follows `HAS_TASK` and `CONSTRAINED_BY` from the plan (tasks, their steps and
//!   decisions, constraints), and that cascade is not ported. A plan that has such
//!   an edge (a constraint written by `create_constraint`, or a migrated base) is
//!   refused by name; once its constraints are deleted, it goes. The check
//!   is the plan's degree (O(1)); the nodes of other domains do not count.
//! - A plan whose `created_at` does not parse, or that lacks `title`,
//!   `description`, `created_by`, `status` or `priority`, is quarantined at open
//!   (as projects are). Neo4j fails the read of such a node, except for an
//!   unparsable `created_at`, which it replaces with "now".
//! - The ids of the plans quarantined at open are RESERVED: `create_plan` with
//!   one of them is refused, as Neo4j's uniqueness constraint would refuse it.
//! - A migrated plan with a `project_id` and NO `HAS_PLAN` edge (Neo4j writes the
//!   property before the `MATCH (project) MERGE (project)-[:HAS_PLAN]->(p)`, so a
//!   plan created while its project did not exist has the property alone) is
//!   listed and counted under that project here, and not on Neo4j. It cannot be
//!   told from a plan this backend wrote, which has no edge either; it also keeps
//!   `delete_project` of that project refused.
//! - Deleting a plan that a migrated note links (`LINKED_TO`) does not update the
//!   note's unresolved anchors (`note.rs`): `get_note_anchors` of that note keeps
//!   refusing until a restart, then answers. Strict side, transient.
//! - An update writes its properties one by one (one WAL record each; obrain has
//!   no multi-property write): a crash in the middle of `update_plan` can reopen
//!   with the new title and the old description. Each state is a valid plan;
//!   Neo4j's single `SET` is atomic. Same limit as `project.rs`.
//!
//! # Order and atomicity
//!
//! Lists sort by `ORDER BY p.priority DESC, p.created_at DESC, p.id DESC`, a
//! total order (the id breaks every tie). A creation follows the draft-marker
//! protocol of `project.rs`: the marker is set first and removed last, and
//! opening deletes what still carries it (the interruption points are those of
//! `create_project`, and `tests/store_conformance.rs` stops a creation at each
//! of them). Reads come from the in-memory index, which a write publishes only
//! once durable.

use std::collections::{BTreeSet, HashMap};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::project::IndexProjets;
#[cfg(any(test, feature = "mock-store"))]
use super::Interruption;
use super::{constraint, Ecriture, NonPorte, ObrainGraphStore, SEUIL_COLONNE_BLOB};
use crate::neo4j::client::pascal_to_snake_case;
use crate::neo4j::models::*;
use crate::neo4j::plan::plan_status_values;
use crate::neo4j::traits::PlanStore;
use crate::plan::models::{TaskDetails, UpdatePlanRequest, UpdateStepRequest, UpdateTaskRequest};

const LABEL: &str = "Plan";
/// The draft marker: present from the first record of a creation to the last.
const BROUILLON: &str = "_brouillon";

/// The methods of `PlanStore` refused on every call, as `PlanStore::<method>`:
/// tasks, steps, the day plan, dependencies, the graphs and waves, and the
/// enrichment. They are refused by name, never answered empty.
pub const REFUSEES: &[&str] = &[
    "PlanStore::create_task",
    "PlanStore::get_plan_tasks",
    "PlanStore::get_task_with_full_details",
    "PlanStore::analyze_task_impact",
    "PlanStore::find_blocked_tasks",
    "PlanStore::update_task_status",
    "PlanStore::interrupt_task",
    "PlanStore::assign_task",
    "PlanStore::add_task_dependency",
    "PlanStore::remove_task_dependency",
    "PlanStore::get_task_blockers",
    "PlanStore::get_tasks_blocked_by",
    "PlanStore::get_task_dependencies",
    "PlanStore::get_plan_dependency_graph",
    "PlanStore::get_task_enrichment_counts",
    "PlanStore::get_task_enrichment_data",
    "PlanStore::get_plan_critical_path",
    "PlanStore::compute_waves",
    "PlanStore::get_next_available_task",
    "PlanStore::get_task",
    "PlanStore::update_task",
    "PlanStore::update_task_enrichment",
    "PlanStore::delete_task",
    "PlanStore::get_project_for_task",
    "PlanStore::get_plan_id_for_task",
    "PlanStore::get_day_plan",
    "PlanStore::set_day_plan",
    "PlanStore::add_to_day",
    "PlanStore::remove_from_day",
    "PlanStore::create_step",
    "PlanStore::get_task_steps",
    "PlanStore::update_step",
    "PlanStore::update_step_status",
    "PlanStore::get_task_step_progress",
    "PlanStore::get_step",
    "PlanStore::delete_step",
    "PlanStore::complete_pending_steps_for_task",
];

/// Does this value go to a blob column (a string above the threshold)?
fn va_en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
}

/// The Neo4j spelling of a status: the Rust variant name.
fn nom_statut(s: &PlanStatus) -> &'static str {
    match s {
        PlanStatus::Draft => "Draft",
        PlanStatus::Approved => "Approved",
        PlanStatus::InProgress => "InProgress",
        PlanStatus::Completed => "Completed",
        PlanStatus::Cancelled => "Cancelled",
    }
}

/// Decodes a stored status as `Neo4jClient::node_to_plan` does: through
/// `pascal_to_snake_case`, so `InProgress` and `in_progress` are both
/// `InProgress`; an unknown value reads as `Draft`.
fn statut_depuis(s: &str) -> PlanStatus {
    serde_json::from_str(&format!("\"{}\"", pascal_to_snake_case(s))).unwrap_or(PlanStatus::Draft)
}

/// Active statuses, on the STORED string: Neo4j's
/// `WHERE p.status IN ['Draft', 'Approved', 'InProgress']`.
fn est_actif(stocke: &str) -> bool {
    matches!(stocke, "Draft" | "Approved" | "InProgress")
}

/// `ORDER BY priority DESC, created_at DESC, id DESC`.
fn ordre_de_liste(a: &PlanNode, b: &PlanNode) -> std::cmp::Ordering {
    b.priority
        .cmp(&a.priority)
        .then_with(|| b.created_at.cmp(&a.created_at))
        .then_with(|| b.id.cmp(&a.id))
}

/// The properties a creation writes, in the order it writes them.
fn proprietes_de_creation(p: &PlanNode) -> Vec<(&'static str, Value)> {
    vec![
        ("id", Value::from(p.id.to_string())),
        ("title", Value::from(p.title.clone())),
        ("description", Value::from(p.description.clone())),
        ("status", Value::from(nom_statut(&p.status))),
        ("created_at", Value::from(p.created_at.to_rfc3339())),
        ("created_by", Value::from(p.created_by.clone())),
        ("priority", Value::from(p.priority as i64)),
        (
            "project_id",
            Value::from(p.project_id.map(|id| id.to_string()).unwrap_or_default()),
        ),
    ]
}

/// The domain value of a `Plan` node from a property lookup, with its stored
/// status. A missing `title`, `description`, `status`, `created_by`, `priority`
/// or `created_at` is an error, as in `Neo4jClient::node_to_plan`.
fn lire_plan(get: &dyn Fn(&str) -> Option<Value>) -> Result<(PlanNode, String)> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let id: Uuid = chaine("id")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("nœud Plan sans propriété `id` lisible"))?;
    let requise =
        |cle: &str| chaine(cle).ok_or_else(|| anyhow!("plan {id} sans propriété `{cle}`"));
    let created_at_brut = requise("created_at")?;
    let created_at: DateTime<Utc> = created_at_brut
        .parse()
        .map_err(|e| anyhow!("plan {id} : `created_at` illisible (« {created_at_brut} ») : {e}"))?;
    let statut = requise("status")?;
    let plan = PlanNode {
        id,
        title: requise("title")?,
        description: requise("description")?,
        status: statut_depuis(&statut),
        created_at,
        created_by: requise("created_by")?,
        priority: get("priority")
            .and_then(|v| v.as_int64())
            .ok_or_else(|| anyhow!("plan {id} sans propriété `priority`"))?
            as i32,
        project_id: chaine("project_id")
            .filter(|s| !s.is_empty())
            .and_then(|s| s.parse().ok()),
        execution_context: chaine("execution_context").filter(|s| !s.is_empty()),
        persona: chaine("persona").filter(|s| !s.is_empty()),
    };
    Ok((plan, statut))
}

/// One plan: where it lives in the base, its decoded value, its status as
/// stored (what the filters compare, as Neo4j's `WHERE p.status IN …`), and the
/// projects its `HAS_PLAN` edges come from.
pub(super) struct Entree {
    pub(super) noeud: NodeId,
    pub(super) plan: PlanNode,
    statut: String,
    pub(super) lies: BTreeSet<Uuid>,
}

impl Entree {
    /// Is the plan in this project: its `project_id`, or a `HAS_PLAN` edge.
    pub(super) fn appartient_a(&self, projet: Uuid) -> bool {
        self.plan.project_id == Some(projet) || self.lies.contains(&projet)
    }

    /// Every project the plan belongs to.
    pub(super) fn projets(&self) -> BTreeSet<Uuid> {
        self.lies
            .iter()
            .copied()
            .chain(self.plan.project_id)
            .collect()
    }

    /// Re-reads the node after a write, under the read lock of the indexes
    /// (released before returning).
    pub(super) fn relire(w: &Ecriture<'_>, noeud: NodeId) -> Result<Self> {
        let etat = w.index()?;
        Self::decoder(w.db, noeud, &etat.projets)
    }

    /// Reads the node (one hydration) and its `HAS_PLAN` edges from the projects
    /// of `projets`, and decodes it.
    pub(super) fn decoder(db: &ObrainDB, noeud: NodeId, projets: &IndexProjets) -> Result<Self> {
        let n = db
            .get_node(noeud)
            .ok_or_else(|| anyhow!("nœud Plan {noeud:?} indexé mais absent de la base"))?;
        let get = |cle: &str| {
            n.get_property(cle)
                .filter(|v| !matches!(v, Value::Null))
                .cloned()
        };
        let (plan, statut) = lire_plan(&get)?;
        Ok(Self {
            noeud,
            plan,
            statut,
            lies: constraint::projets_lies(db, noeud, projets),
        })
    }
}

/// `Uuid → plan`, rebuilt from the `Plan` nodes at every open.
#[derive(Default)]
pub(super) struct IndexPlans {
    pub(super) par_id: HashMap<Uuid, Entree>,
    /// The ids of the plans quarantined at open: not served, but taken. A
    /// creation with one of them is refused (Neo4j's uniqueness constraint on
    /// `Plan.id`), as `note.rs` and `chat.rs` do.
    reserves: std::collections::HashSet<Uuid>,
}

impl IndexPlans {
    /// The engine node of a plan this backend addresses (read by `commit.rs`
    /// to link a commit to a plan, or to list a plan's commits).
    pub(super) fn noeud_de(&self, id: Uuid) -> Option<NodeId> {
        self.par_id.get(&id).map(|e| e.noeud)
    }

    /// Rebuilds the index. Returns it with the number of drafts deleted and the
    /// nodes quarantined (named, left untouched). `projets` is read for the
    /// `HAS_PLAN` edges (module doc).
    pub(super) fn reconstruire(
        db: &ObrainDB,
        projets: &IndexProjets,
    ) -> Result<(Self, usize, Vec<String>)> {
        let store = db.store();
        let mut index = Self::default();
        let mut brouillons = 0;
        let mut quarantaine = Vec::new();
        for noeud in store.nodes_by_label(LABEL) {
            let Some(n) = db.get_node(noeud) else {
                quarantaine.push(format!("nœud Plan {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            // A creation that never reached its commit point (module doc).
            if get(BROUILLON).is_some() || (n.properties.is_empty() && store.out_degree(noeud) == 0)
            {
                db.delete_node(noeud);
                brouillons += 1;
                continue;
            }
            let (plan, statut) = match lire_plan(&get) {
                Ok(lu) => lu,
                Err(e) => {
                    quarantaine.push(format!("nœud Plan {noeud:?} : {e}"));
                    if let Some(id) = get("id")
                        .and_then(|v| v.as_str().map(str::to_string))
                        .and_then(|s| s.parse::<Uuid>().ok())
                    {
                        index.reserves.insert(id);
                    }
                    continue;
                }
            };
            if index.par_id.contains_key(&plan.id) {
                quarantaine.push(format!("nœud Plan {noeud:?} : id {} déjà pris", plan.id));
                continue;
            }
            let lies = constraint::projets_lies(db, noeud, projets);
            index.par_id.insert(
                plan.id,
                Entree {
                    noeud,
                    plan,
                    statut,
                    lies,
                },
            );
        }
        Ok((index, brouillons, quarantaine))
    }
}

/// Refuses a method of the plan domain that obrain does not port, by name.
fn refus<T>(methode: &'static str) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)))
}

// The refusals keep their parameters' names for the reader; they are unused by design.
#[allow(unused_variables)]
#[async_trait]
impl PlanStore for ObrainGraphStore {
    async fn create_plan(&self, plan: &PlanNode) -> Result<()> {
        let p = plan.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let proprietes = proprietes_de_creation(&p);
                // Validate first: nothing is written when one of these refuses.
                {
                    let etat = w.index()?;
                    if etat.plans.par_id.contains_key(&p.id) {
                        bail!("un plan d'id {} existe déjà (unicité de Plan.id)", p.id);
                    }
                    if etat.plans.reserves.contains(&p.id) {
                        bail!(
                            "l'id {} est celui d'un plan en quarantaine (nœud illisible ou id \
                             en double) : réservé, la création est refusée (unicité de Plan.id)",
                            p.id
                        );
                    }
                    if let Some(projet) = p.project_id {
                        if !etat.projets.contient(projet) {
                            bail!(
                                "le projet {projet} n'existe pas : le plan {} ne peut pas y être \
                                 rattaché (le backend obrain ne crée pas de plan orphelin)",
                                p.id
                            );
                        }
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
                w.rendre_durable_avant_le_marqueur()?;
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::AvantLeRetraitDuMarqueur)?;
                // The commit point.
                w.db.remove_node_property(noeud, BROUILLON);
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeRetraitDuMarqueur)?;
                let entree = w.apres_ecriture(Entree::relire(w, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.plans.par_id.insert(p.id, entree);
                Ok(())
            },
        )
        .await
    }

    async fn get_plan(&self, id: Uuid) -> Result<Option<PlanNode>> {
        self.lire(|_, etat| Ok(etat.plans.par_id.get(&id).map(|e| e.plan.clone())))
    }

    async fn list_active_plans(&self) -> Result<Vec<PlanNode>> {
        self.lire(|_, etat| {
            let mut plans: Vec<PlanNode> = etat
                .plans
                .par_id
                .values()
                .filter(|e| est_actif(&e.statut))
                .map(|e| e.plan.clone())
                .collect();
            plans.sort_by(ordre_de_liste);
            Ok(plans)
        })
    }

    async fn list_project_plans(&self, project_id: Uuid) -> Result<Vec<PlanNode>> {
        self.lire(|_, etat| {
            // `MATCH (project:Project {id})-[:HAS_PLAN]->(p)`: no project, no plan.
            if !etat.projets.contient(project_id) {
                return Ok(Vec::new());
            }
            let mut plans: Vec<PlanNode> = etat
                .plans
                .par_id
                .values()
                .filter(|e| e.appartient_a(project_id) && est_actif(&e.statut))
                .map(|e| e.plan.clone())
                .collect();
            plans.sort_by(ordre_de_liste);
            Ok(plans)
        })
    }

    async fn count_project_plans(&self, project_id: Uuid) -> Result<i64> {
        self.lire(|_, etat| {
            if !etat.projets.contient(project_id) {
                return Ok(0);
            }
            let n = etat
                .plans
                .par_id
                .values()
                .filter(|e| e.appartient_a(project_id) && est_actif(&e.statut))
                .count();
            Ok(n as i64)
        })
    }

    async fn list_plans_for_project(
        &self,
        project_id: Uuid,
        status_filter: Option<Vec<String>>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<PlanNode>, usize)> {
        self.lire(|_, etat| {
            if !etat.projets.contient(project_id) {
                return Ok((Vec::new(), 0));
            }
            // Neo4j's normalisation of the filter, then `p.status IN $statuses` on
            // the stored string.
            let filtre = plan_status_values(status_filter.as_deref());
            let mut plans: Vec<PlanNode> = etat
                .plans
                .par_id
                .values()
                .filter(|e| {
                    e.appartient_a(project_id)
                        && filtre.as_ref().is_none_or(|f| f.contains(&e.statut))
                })
                .map(|e| e.plan.clone())
                .collect();
            plans.sort_by(ordre_de_liste);
            let total = plans.len();
            Ok((plans.into_iter().skip(offset).take(limit).collect(), total))
        })
    }

    async fn update_plan(&self, id: Uuid, updates: &UpdatePlanRequest) -> Result<()> {
        // A closed or indeterminate base refuses, even an empty update.
        self.garde()?;
        // Nothing to set: nothing is written (Neo4j answers the same).
        if updates.title.is_none() && updates.description.is_none() && updates.priority.is_none() {
            return Ok(());
        }
        let u = updates.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, ancien_en_blob) = {
                    let etat = w.index()?;
                    // A missing plan is a silent no-op, as in Neo4j.
                    let Some(e) = etat.plans.par_id.get(&id) else {
                        return Ok(());
                    };
                    let ancien = e.plan.title.len() > SEUIL_COLONNE_BLOB
                        || e.plan.description.len() > SEUIL_COLONNE_BLOB;
                    (e.noeud, ancien)
                };
                w.flush_complet = ancien_en_blob
                    || u.title
                        .iter()
                        .chain(u.description.iter())
                        .any(|s| s.len() > SEUIL_COLONNE_BLOB);
                if let Some(titre) = u.title {
                    w.db.set_node_property(noeud, "title", Value::from(titre));
                }
                if let Some(description) = u.description {
                    w.db.set_node_property(noeud, "description", Value::from(description));
                }
                if let Some(priorite) = u.priority {
                    w.db.set_node_property(noeud, "priority", Value::from(priorite as i64));
                }
                let entree = w.apres_ecriture(Entree::relire(w, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.plans.par_id.insert(id, entree);
                Ok(())
            },
        )
        .await
    }

    async fn update_plan_status(&self, id: Uuid, status: PlanStatus) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // A missing plan is a silent no-op, as in Neo4j.
                let Some(noeud) = w.index()?.plans.par_id.get(&id).map(|e| e.noeud) else {
                    return Ok(());
                };
                w.db.set_node_property(noeud, "status", Value::from(nom_statut(&status)));
                let entree = w.apres_ecriture(Entree::relire(w, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.plans.par_id.insert(id, entree);
                Ok(())
            },
        )
        .await
    }

    async fn delete_plan(&self, plan_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = w.index()?.plans.par_id.get(&plan_id).map(|e| e.noeud) else {
                    return Ok(());
                };
                // The Neo4j cascade starts from the plan's OUTGOING edges
                // (`HAS_TASK`, `CONSTRAINED_BY`): what it would reach hangs there,
                // and that cascade is not ported. The incoming `HAS_PLAN` are
                // detached by the deletion, as `DETACH DELETE`.
                let sortantes = w.db.store().out_degree(noeud);
                if sortantes > 0 {
                    return Err(
                        anyhow::Error::new(NonPorte("PlanStore::delete_plan (cascade)")).context(
                            format!(
                                "le plan {plan_id} a {sortantes} arête(s) sortante(s) \
                                 (CONSTRAINED_BY : ses contraintes ; HAS_TASK : les tâches d'une \
                                 base migrée) que la suppression en cascade devrait atteindre. \
                                 Elle n'est pas portée : supprimer d'abord ses contraintes"
                            ),
                        ),
                    );
                }
                if !w.db.delete_node(noeud) {
                    bail!("obrain a refusé la suppression du nœud {noeud:?} (plan {plan_id})");
                }
                w.rendre_durable()?;
                w.index_mut()?.plans.par_id.remove(&plan_id);
                Ok(())
            },
        )
        .await
    }

    async fn link_plan_to_project(&self, plan_id: Uuid, project_id: Uuid) -> Result<()> {
        self.lier_plan_au_projet(plan_id, project_id).await
    }

    async fn list_plan_project_slugs(&self, plan_id: Uuid) -> Result<Vec<String>> {
        self.slugs_des_projets_du_plan(plan_id)
    }

    async fn unlink_plan_from_project(&self, plan_id: Uuid) -> Result<()> {
        self.delier_plan(plan_id).await
    }

    async fn create_task(&self, plan_id: Uuid, task: &TaskNode) -> Result<()> {
        refus("PlanStore::create_task")
    }

    async fn get_plan_tasks(&self, plan_id: Uuid) -> Result<Vec<TaskNode>> {
        refus("PlanStore::get_plan_tasks")
    }

    async fn get_task_with_full_details(&self, task_id: Uuid) -> Result<Option<TaskDetails>> {
        refus("PlanStore::get_task_with_full_details")
    }

    async fn analyze_task_impact(&self, task_id: Uuid) -> Result<Vec<String>> {
        refus("PlanStore::analyze_task_impact")
    }

    async fn find_blocked_tasks(&self, plan_id: Uuid) -> Result<Vec<(TaskNode, Vec<TaskNode>)>> {
        refus("PlanStore::find_blocked_tasks")
    }

    async fn update_task_status(&self, task_id: Uuid, status: TaskStatus) -> Result<()> {
        refus("PlanStore::update_task_status")
    }

    async fn interrupt_task(&self, task_id: Uuid, reason: &str) -> Result<bool> {
        refus("PlanStore::interrupt_task")
    }

    async fn assign_task(&self, task_id: Uuid, agent_id: &str) -> Result<()> {
        refus("PlanStore::assign_task")
    }

    async fn add_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()> {
        refus("PlanStore::add_task_dependency")
    }

    async fn remove_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()> {
        refus("PlanStore::remove_task_dependency")
    }

    async fn get_task_blockers(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        refus("PlanStore::get_task_blockers")
    }

    async fn get_tasks_blocked_by(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        refus("PlanStore::get_tasks_blocked_by")
    }

    async fn get_task_dependencies(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        refus("PlanStore::get_task_dependencies")
    }

    async fn get_plan_dependency_graph(
        &self,
        plan_id: Uuid,
    ) -> Result<(Vec<TaskNode>, Vec<(Uuid, Uuid)>)> {
        refus("PlanStore::get_plan_dependency_graph")
    }

    async fn get_task_enrichment_counts(
        &self,
        task_ids: &[String],
    ) -> Result<std::collections::HashMap<String, crate::neo4j::plan::TaskEnrichmentCounts>> {
        refus("PlanStore::get_task_enrichment_counts")
    }

    async fn get_task_enrichment_data(
        &self,
        task_ids: &[String],
    ) -> Result<std::collections::HashMap<String, crate::neo4j::plan::TaskEnrichmentData>> {
        refus("PlanStore::get_task_enrichment_data")
    }

    async fn get_plan_critical_path(&self, plan_id: Uuid) -> Result<Vec<TaskNode>> {
        refus("PlanStore::get_plan_critical_path")
    }

    async fn compute_waves(
        &self,
        plan_id: Uuid,
    ) -> Result<crate::neo4j::plan::WaveComputationResult> {
        refus("PlanStore::compute_waves")
    }

    async fn get_next_available_task(&self, plan_id: Uuid) -> Result<Option<TaskNode>> {
        refus("PlanStore::get_next_available_task")
    }

    async fn get_task(&self, task_id: Uuid) -> Result<Option<TaskNode>> {
        refus("PlanStore::get_task")
    }

    async fn update_task(&self, task_id: Uuid, updates: &UpdateTaskRequest) -> Result<()> {
        refus("PlanStore::update_task")
    }

    async fn update_task_enrichment(
        &self,
        task_id: Uuid,
        execution_context: Option<&str>,
        persona: Option<&str>,
        prompt_cache: Option<&str>,
    ) -> Result<()> {
        refus("PlanStore::update_task_enrichment")
    }

    async fn delete_task(&self, task_id: Uuid) -> Result<()> {
        refus("PlanStore::delete_task")
    }

    async fn get_project_for_task(&self, task_id: Uuid) -> Result<Option<ProjectNode>> {
        refus("PlanStore::get_project_for_task")
    }

    async fn get_plan_id_for_task(&self, task_id: Uuid) -> Result<Option<Uuid>> {
        refus("PlanStore::get_plan_id_for_task")
    }

    async fn get_day_plan(&self, user_id: Uuid, date: &str) -> Result<Vec<(Uuid, u32)>> {
        refus("PlanStore::get_day_plan")
    }

    async fn set_day_plan(&self, user_id: Uuid, date: &str, ids: &[Uuid]) -> Result<()> {
        refus("PlanStore::set_day_plan")
    }

    async fn add_to_day(&self, user_id: Uuid, date: &str, task_id: Uuid) -> Result<()> {
        refus("PlanStore::add_to_day")
    }

    async fn remove_from_day(&self, user_id: Uuid, date: &str, task_id: Uuid) -> Result<()> {
        refus("PlanStore::remove_from_day")
    }

    async fn create_step(&self, task_id: Uuid, step: &StepNode) -> Result<()> {
        refus("PlanStore::create_step")
    }

    async fn get_task_steps(&self, task_id: Uuid) -> Result<Vec<StepNode>> {
        refus("PlanStore::get_task_steps")
    }

    async fn update_step(&self, step_id: Uuid, updates: &UpdateStepRequest) -> Result<()> {
        refus("PlanStore::update_step")
    }

    async fn update_step_status(&self, step_id: Uuid, status: StepStatus) -> Result<()> {
        refus("PlanStore::update_step_status")
    }

    async fn get_task_step_progress(&self, task_id: Uuid) -> Result<(u32, u32)> {
        refus("PlanStore::get_task_step_progress")
    }

    async fn get_step(&self, step_id: Uuid) -> Result<Option<StepNode>> {
        refus("PlanStore::get_step")
    }

    async fn delete_step(&self, step_id: Uuid) -> Result<()> {
        refus("PlanStore::delete_step")
    }

    async fn complete_pending_steps_for_task(&self, task_id: Uuid) -> Result<u32> {
        refus("PlanStore::complete_pending_steps_for_task")
    }

    async fn create_constraint(&self, plan_id: Uuid, constraint: &ConstraintNode) -> Result<()> {
        self.creer_contrainte(plan_id, constraint).await
    }

    async fn get_plan_constraints(&self, plan_id: Uuid) -> Result<Vec<ConstraintNode>> {
        self.contraintes_du_plan(plan_id)
    }

    async fn get_constraint(&self, constraint_id: Uuid) -> Result<Option<ConstraintNode>> {
        self.contrainte_par_id(constraint_id)
    }

    async fn update_constraint(
        &self,
        constraint_id: Uuid,
        description: Option<String>,
        constraint_type: Option<ConstraintType>,
        enforced_by: Option<String>,
    ) -> Result<()> {
        self.modifier_contrainte(constraint_id, description, constraint_type, enforced_by)
            .await
    }

    async fn delete_constraint(&self, constraint_id: Uuid) -> Result<()> {
        self.supprimer_contrainte(constraint_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obrain::Creation;
    use chrono::TimeZone;

    fn plan(titre: &str, priorite: i32, statut: PlanStatus, decalage: i64) -> PlanNode {
        PlanNode {
            id: Uuid::new_v4(),
            title: titre.to_string(),
            description: "description".to_string(),
            status: statut,
            created_at: Utc.timestamp_opt(1_700_000_000 + decalage, 0).unwrap(),
            created_by: "test".to_string(),
            priority: priorite,
            project_id: None,
            execution_context: None,
            persona: None,
        }
    }

    #[tokio::test]
    async fn cycle_de_vie_ordre_et_filtre() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let a = plan("a", 1, PlanStatus::Draft, 0);
        let b = plan("b", 5, PlanStatus::InProgress, 0);
        let c = plan("c", 5, PlanStatus::Completed, 10);
        for p in [&a, &b, &c] {
            store.create_plan(p).await.unwrap();
        }
        assert_eq!(store.get_plan(a.id).await.unwrap().unwrap().title, "a");
        // priorité décroissante, puis created_at décroissant ; terminé exclu
        let actifs: Vec<String> = store
            .list_active_plans()
            .await
            .unwrap()
            .into_iter()
            .map(|p| p.title)
            .collect();
        assert_eq!(actifs, vec!["b", "a"]);
        store
            .update_plan_status(a.id, PlanStatus::Completed)
            .await
            .unwrap();
        assert_eq!(store.list_active_plans().await.unwrap().len(), 1);
        store
            .update_plan(
                b.id,
                &UpdatePlanRequest {
                    title: Some("b2".into()),
                    description: None,
                    priority: Some(0),
                },
            )
            .await
            .unwrap();
        let got = store.get_plan(b.id).await.unwrap().unwrap();
        assert_eq!((got.title.as_str(), got.priority), ("b2", 0));
        store.delete_plan(c.id).await.unwrap();
        assert!(store.get_plan(c.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn rattachement_a_un_projet_absent_est_refuse() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let mut p = plan("orphelin", 0, PlanStatus::Draft, 0);
        p.project_id = Some(Uuid::new_v4());
        let e = store.create_plan(&p).await.unwrap_err();
        assert!(
            format!("{e:#}").contains("n'existe pas"),
            "the refusal names the missing project, not another failure: {e:#}"
        );
        assert!(store.get_plan(p.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn plan_survit_a_la_reouverture() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = plan("durable", 2, PlanStatus::Approved, 0);
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_plan(&p).await.unwrap();
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        let relu = store.get_plan(p.id).await.unwrap().unwrap();
        assert_eq!(relu.title, "durable");
        assert_eq!(relu.status, PlanStatus::Approved);
    }
}
