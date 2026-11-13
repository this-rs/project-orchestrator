//! `PlanStore` on obrain: the core of the plan domain (lot O-PLAN-1).
//!
//! PORTED here, with the Neo4j semantics of `src/neo4j/plan.rs`:
//! - `create_plan`, `get_plan`, `list_active_plans`, `list_project_plans`,
//!   `count_project_plans`, `list_plans_for_project`;
//! - `update_plan`, `update_plan_status`;
//! - `delete_plan`, for a plan that holds nothing else (see below).
//!
//! REFUSED by name (`NonPorte`): everything that touches tasks, steps,
//! constraints, the day plan, dependencies, the graphs and waves, the
//! enrichment, and the HAS_PLAN links that a plan can have with SEVERAL
//! projects (`link_plan_to_project`, `unlink_plan_from_project`,
//! `list_plan_project_slugs`). They are refused, not answered empty.
//!
//! # Layout
//!
//! A plan is one node labelled `Plan`, with the properties Neo4j writes:
//! `id`, `title`, `description`, `status` (the Rust variant name, as Neo4j
//! does: `Draft`, `InProgress`…), `created_at` (RFC 3339 string), `created_by`,
//! `priority`, `project_id` (the empty string for none).
//!
//! Membership in a project is the `project_id` property. Neo4j stores it as a
//! `HAS_PLAN` edge and keeps the property in step. The two agree for every plan
//! that this backend creates or reads, because the only way to change
//! `project_id` is `link_plan_to_project`, which is refused.
//!
//! # Divergences, all on the strict side
//!
//! - `create_plan` with a `project_id` whose project does not exist is REFUSED.
//!   Neo4j's `MATCH` on the project silently creates a plan with no link.
//! - `delete_plan` deletes only a plan with nothing else in the base: the Neo4j
//!   cascade reaches tasks, steps, decisions and constraints, which this backend
//!   does not port. The check is one counter, as in `delete_project`.
//! - A plan whose `created_at` does not parse, or that lacks `title`,
//!   `description` or `created_by`, is quarantined at open (as projects are),
//!   where Neo4j would replace the timestamp with "now".
//!
//! # Order and atomicity
//!
//! Lists sort by `ORDER BY p.priority DESC, p.created_at DESC, p.id DESC`, a
//! total order (the id breaks every tie). A creation follows the draft-marker
//! protocol of `project.rs`: the marker is set first and removed last, and
//! opening deletes what still carries it. Reads come from the in-memory index,
//! which a write publishes only once durable.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::{Ecriture, NonPorte, ObrainGraphStore, SEUIL_COLONNE_BLOB};
use crate::neo4j::models::*;
use crate::neo4j::traits::PlanStore;
use crate::plan::models::{TaskDetails, UpdatePlanRequest, UpdateStepRequest, UpdateTaskRequest};

const LABEL: &str = "Plan";
/// The draft marker: present from the first record of a creation to the last.
const BROUILLON: &str = "_brouillon";

/// The methods of `PlanStore` refused on every call, as `PlanStore::<method>`:
/// tasks, steps, constraints, the day plan, dependencies, the graphs and
/// waves, the enrichment, and the HAS_PLAN links that a plan can have with
/// several projects. They are refused by name, never answered empty.
pub const REFUSEES: &[&str] = &[
    "PlanStore::link_plan_to_project",
    "PlanStore::list_plan_project_slugs",
    "PlanStore::unlink_plan_from_project",
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
    "PlanStore::create_constraint",
    "PlanStore::get_plan_constraints",
    "PlanStore::get_constraint",
    "PlanStore::update_constraint",
    "PlanStore::delete_constraint",
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

/// Decodes a stored status. An unknown value reads as `Draft`, as Neo4j does.
fn statut_depuis(s: &str) -> PlanStatus {
    match s {
        "Approved" => PlanStatus::Approved,
        "InProgress" => PlanStatus::InProgress,
        "Completed" => PlanStatus::Completed,
        "Cancelled" => PlanStatus::Cancelled,
        _ => PlanStatus::Draft,
    }
}

/// Active statuses: what `list_active_plans` and the counts read.
fn est_actif(s: &PlanStatus) -> bool {
    matches!(
        s,
        PlanStatus::Draft | PlanStatus::Approved | PlanStatus::InProgress
    )
}

/// `ORDER BY priority DESC, created_at DESC, id DESC`.
fn ordre_de_liste(a: &PlanNode, b: &PlanNode) -> std::cmp::Ordering {
    b.priority
        .cmp(&a.priority)
        .then_with(|| b.created_at.cmp(&a.created_at))
        .then_with(|| b.id.cmp(&a.id))
}

/// The status filter of `list_plans_for_project`, as Neo4j reads it: an empty
/// filter means none; a known spelling (`in_progress`, `InProgress`…) maps to its
/// variant; an unknown value matches no plan, so it is dropped.
fn filtre_statuts(statuts: Option<&[String]>) -> Option<Vec<&'static str>> {
    let statuts = statuts?;
    if statuts.is_empty() {
        return None;
    }
    Some(
        statuts
            .iter()
            .filter_map(|s| {
                let normalise = match s.to_lowercase().as_str() {
                    "draft" => Some("Draft"),
                    "approved" => Some("Approved"),
                    "in_progress" => Some("InProgress"),
                    "completed" => Some("Completed"),
                    "cancelled" => Some("Cancelled"),
                    _ => None,
                };
                normalise.or_else(|| {
                    ["Draft", "Approved", "InProgress", "Completed", "Cancelled"]
                        .into_iter()
                        .find(|nom| *nom == s.as_str())
                })
            })
            .collect(),
    )
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

/// The domain value of a `Plan` node from a property lookup. A missing
/// `title`, `description`, `created_by` or `created_at` is an error, as in
/// `Neo4jClient::node_to_plan`.
fn lire_plan(get: &dyn Fn(&str) -> Option<Value>) -> Result<PlanNode> {
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
    Ok(PlanNode {
        id,
        title: requise("title")?,
        description: requise("description")?,
        status: statut_depuis(&chaine("status").unwrap_or_default()),
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
    })
}

/// One plan: where it lives in the base, and its decoded value.
pub(super) struct Entree {
    noeud: NodeId,
    pub(super) plan: PlanNode,
}

impl Entree {
    /// Reads the node (one hydration) and decodes it.
    fn decoder(db: &ObrainDB, noeud: NodeId) -> Result<Self> {
        let n = db
            .get_node(noeud)
            .ok_or_else(|| anyhow!("nœud Plan {noeud:?} indexé mais absent de la base"))?;
        let get = |cle: &str| {
            n.get_property(cle)
                .filter(|v| !matches!(v, Value::Null))
                .cloned()
        };
        Ok(Self {
            noeud,
            plan: lire_plan(&get)?,
        })
    }
}

/// `Uuid → plan`, rebuilt from the `Plan` nodes at every open.
#[derive(Default)]
pub(super) struct IndexPlans {
    pub(super) par_id: HashMap<Uuid, Entree>,
}

impl IndexPlans {
    /// Rebuilds the index. Returns it with the number of drafts deleted and the
    /// nodes quarantined (named, left untouched).
    pub(super) fn reconstruire(db: &ObrainDB) -> Result<(Self, usize, Vec<String>)> {
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
            let plan = match lire_plan(&get) {
                Ok(plan) => plan,
                Err(e) => {
                    quarantaine.push(format!("nœud Plan {noeud:?} : {e}"));
                    continue;
                }
            };
            if index.par_id.contains_key(&plan.id) {
                quarantaine.push(format!("nœud Plan {noeud:?} : id {} déjà pris", plan.id));
                continue;
            }
            index.par_id.insert(plan.id, Entree { noeud, plan });
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
                w.db.set_node_property(noeud, BROUILLON, Value::from(true));
                for (cle, valeur) in proprietes {
                    w.db.set_node_property(noeud, cle, valeur);
                }
                w.rendre_durable_avant_le_marqueur()?;
                // The commit point.
                w.db.remove_node_property(noeud, BROUILLON);
                let entree = w.apres_ecriture(Entree::decoder(w.db, noeud))?;
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
                .filter(|e| est_actif(&e.plan.status))
                .map(|e| e.plan.clone())
                .collect();
            plans.sort_by(ordre_de_liste);
            Ok(plans)
        })
    }

    async fn list_project_plans(&self, project_id: Uuid) -> Result<Vec<PlanNode>> {
        self.lire(|_, etat| {
            let mut plans: Vec<PlanNode> = etat
                .plans
                .par_id
                .values()
                .filter(|e| e.plan.project_id == Some(project_id) && est_actif(&e.plan.status))
                .map(|e| e.plan.clone())
                .collect();
            plans.sort_by(ordre_de_liste);
            Ok(plans)
        })
    }

    async fn count_project_plans(&self, project_id: Uuid) -> Result<i64> {
        self.lire(|_, etat| {
            let n = etat
                .plans
                .par_id
                .values()
                .filter(|e| e.plan.project_id == Some(project_id) && est_actif(&e.plan.status))
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
            let filtre = filtre_statuts(status_filter.as_deref());
            let mut plans: Vec<PlanNode> = etat
                .plans
                .par_id
                .values()
                .filter(|e| {
                    e.plan.project_id == Some(project_id)
                        && filtre
                            .as_ref()
                            .is_none_or(|f| f.contains(&nom_statut(&e.plan.status)))
                })
                .map(|e| e.plan.clone())
                .collect();
            plans.sort_by(ordre_de_liste);
            let total = plans.len();
            Ok((plans.into_iter().skip(offset).take(limit).collect(), total))
        })
    }

    async fn update_plan(&self, id: Uuid, updates: &UpdatePlanRequest) -> Result<()> {
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
                let entree = w.apres_ecriture(Entree::decoder(w.db, noeud))?;
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
                let entree = w.apres_ecriture(Entree::decoder(w.db, noeud))?;
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
                let (noeud, projets, plans) = {
                    let etat = w.index()?;
                    let Some(e) = etat.plans.par_id.get(&plan_id) else {
                        return Ok(());
                    };
                    (e.noeud, etat.projets.nombre(), etat.plans.par_id.len())
                };
                // One counter, no scan: every live node of the base against the
                // projects and plans this backend addresses. Anything else is a
                // task, step, decision or constraint the cascade must reach.
                let noeuds = w.db.store().node_count();
                if noeuds != projets + plans {
                    return Err(
                        anyhow::Error::new(NonPorte("PlanStore::delete_plan (cascade)")).context(
                            format!(
                            "la base contient {} nœud(s) hors des projets et plans que ce backend \
                             adresse : tâches, étapes, décisions ou contraintes que la suppression \
                             en cascade devrait atteindre. Elle attend le portage de ces domaines",
                            noeuds.saturating_sub(projets + plans)
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
        refus("PlanStore::link_plan_to_project")
    }

    async fn list_plan_project_slugs(&self, plan_id: Uuid) -> Result<Vec<String>> {
        refus("PlanStore::list_plan_project_slugs")
    }

    async fn unlink_plan_from_project(&self, plan_id: Uuid) -> Result<()> {
        refus("PlanStore::unlink_plan_from_project")
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
        refus("PlanStore::create_constraint")
    }

    async fn get_plan_constraints(&self, plan_id: Uuid) -> Result<Vec<ConstraintNode>> {
        refus("PlanStore::get_plan_constraints")
    }

    async fn get_constraint(&self, constraint_id: Uuid) -> Result<Option<ConstraintNode>> {
        refus("PlanStore::get_constraint")
    }

    async fn update_constraint(
        &self,
        constraint_id: Uuid,
        description: Option<String>,
        constraint_type: Option<ConstraintType>,
        enforced_by: Option<String>,
    ) -> Result<()> {
        refus("PlanStore::update_constraint")
    }

    async fn delete_constraint(&self, constraint_id: Uuid) -> Result<()> {
        refus("PlanStore::delete_constraint")
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
        assert!(store.create_plan(&p).await.is_err());
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
