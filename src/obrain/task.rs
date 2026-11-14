//! `PlanStore` on obrain: the TASKS of a plan, their dependencies, the graphs
//! computed from them (dependency graph, critical path, waves) and the day plan.
//! The steps of a task are in `step.rs`; `plan.rs` delegates to both.
//!
//! PORTED here, with the Neo4j semantics of `src/neo4j/task.rs`,
//! `src/neo4j/plan.rs` and `src/neo4j/day_plan.rs`:
//! `create_task`, `get_task`, `get_plan_tasks`, `update_task`,
//! `update_task_status`, `interrupt_task`, `update_task_enrichment`,
//! `delete_task`, `get_plan_id_for_task`, `get_project_for_task`,
//! `get_task_with_full_details`, `add_task_dependency`,
//! `remove_task_dependency`, `get_task_dependencies`, `get_task_blockers`,
//! `get_tasks_blocked_by`, `find_blocked_tasks`, `get_next_available_task`,
//! `get_plan_dependency_graph`, `get_plan_critical_path`, `compute_waves`,
//! `get_day_plan`, `set_day_plan`, `add_to_day`, `remove_from_day`.
//!
//! REFUSED by name ([`REFUS`], each with its cause): `assign_task` (the `Agent`
//! node and its `WORKING_ON` edge), `analyze_task_impact` (`MODIFIES` and
//! `IMPORTS` between `File` nodes: the code domain), `get_task_enrichment_counts`
//! and `get_task_enrichment_data` (notes linked to a task, decisions, chat
//! sessions spawned for it, files discussed: domains whose task side is not
//! ported).
//!
//! # Layout (the Neo4j one)
//!
//! - `(:Plan)-[:HAS_TASK]->(:Task)`, `(:Task)-[:HAS_STEP]->(:Step)`,
//!   `(:Task)-[:DEPENDS_ON]->(:Task)` (any two tasks, of any plans),
//!   `(:DayPlan {user_id, date, updated_at})-[:INCLUDES {order}]->(:Task)`.
//! - A task's properties are those `Neo4jClient::create_task` writes, with its
//!   lossy conventions (`title` `""` for none, `priority` `0` for none — it then
//!   reads back `Some(0)` —, `estimated_complexity`/`estimated_minutes` `0` for
//!   none, `due_date` `""` for none, `frustration_score` `0.0`, `updated_at` =
//!   `created_at`); statuses are the Rust variant names (`InProgress`).
//!   Timestamps are RFC 3339 strings (Neo4j: `datetime()`); the order they give
//!   is the instant's.
//!
//! # Orders (Neo4j's, with a total tie-break where Neo4j leaves one free)
//!
//! - plan tasks, next available task: `priority DESC (none = 0), created_at, id`;
//! - dependencies, blockers, blocked-by: `priority DESC (none = 0), id DESC`;
//! - day plan: `order` (then the task id, for a tie only a foreign base can hold);
//! - `find_blocked_tasks` and the edges of `get_plan_dependency_graph` have no
//!   `ORDER BY` in Neo4j: they follow the plan-task order here (blockers in the
//!   blockers' order), one of the orders Neo4j may return;
//! - the critical path is the Neo4j query's: the longest chain from a task that
//!   depends on no task to a task no task depends on, every node in the plan, the
//!   tie broken by the list of ids, greatest first;
//! - the waves are computed by `neo4j::plan::compute_waves_from_graph`, the ONE
//!   algorithm the Neo4j adapter runs too (extracted from it, not rewritten).
//!
//! # Divergences, all on the strict side or not observable through the contract
//!
//! - `create_task` on a plan this backend does not hold is REFUSED (Neo4j's
//!   `MATCH` creates nothing and answers `Ok`); an id already taken, or reserved
//!   by a quarantined node, is refused (Neo4j would create a second node).
//! - A task or step whose status is not one of the contract's, or whose
//!   `created_at` does not parse, is QUARANTINED at open (Neo4j reads the first as
//!   `Pending` and the second as "now").
//! - `get_plan_critical_path` and `compute_waves` refuse a plan whose
//!   dependencies hold a cycle (`compute_waves` does on Neo4j too; Neo4j's
//!   critical-path query walks the cycle's edges once each).
//! - `get_task_with_full_details` reads a step's status as stored. Neo4j's
//!   `node_to_step` lowercases `InProgress` into `inprogress`, which does not
//!   parse, and so reads an in-progress step as `Pending` there (a Neo4j bug, not
//!   reproduced).
//! - `update_task_status(Blocked)` adds 0.2 to `frustration_score` (capped at 1).
//!   Written LAST, so a crash before it leaves the score untouched; but a crash
//!   after it and before the acknowledgement leaves a caller that does not know
//!   the call landed, and a replay adds 0.2 once more. Neo4j's transaction has the
//!   same effect only for a lost acknowledgement. The score is an indication,
//!   capped: accepted, named here.
//! - `add_to_day` with a task that does not exist creates no `DayPlan` node
//!   (Neo4j's `MERGE` creates an empty one first; nothing in the contract reads it).
//! - Anything this index cannot account for refuses instead of answering partly:
//!   an id reserved by a quarantined node, a plan or task whose base holds a node
//!   or an edge the index could not read (`incomplets`), and a task or step
//!   carrying an edge of an unported domain (`etrangeres`) for the reads and the
//!   cascade that would reach it (`delete_task`, `delete_step`,
//!   `get_task_with_full_details` for `INFORMED_BY`/`MODIFIES`). No ported domain
//!   writes such an edge; only a migrated or foreign base can hold one.
//!
//! # Atomicity
//!
//! obrain's direct writes have no transaction: every property and edge is its own
//! WAL record, and a crash keeps a PREFIX of them. So:
//! - a value that goes to obrain's blob column (a string above
//!   `SEUIL_COLONNE_BLOB`, or a list holding one), written or REPLACED, makes the
//!   write pay the full flush before it is acknowledged, and a creation pays it
//!   before its commit point (the `plan.rs` / `note.rs` rule): the WAL does not
//!   recover such a value at this pin;
//! - a creation (task, step, day plan) carries the draft marker `_brouillon` from
//!   its first record to its last (after the `HAS_TASK`/`HAS_STEP` edge); opening
//!   deletes a draft with its edges;
//! - a deletion (task with its steps, step) first sets the marker `_suppression`;
//!   opening finishes any deletion that carries it. Without it, a crash between
//!   the parent edge and the node would leave an orphan that opening could only
//!   quarantine;
//! - an `INCLUDES` edge is created with its `order`; one without (a crash between
//!   the two records) is deleted at open (Neo4j's `get_day_plan` skips it anyway);
//! - the other multi-writes (updates, status changes, `complete_pending_steps`,
//!   `set_day_plan`) leave, after a crash, a prefix that a retry of the same call
//!   makes converge. An error AFTER the first engine write of any of them marks
//!   the store INDETERMINATE (every operation refused until a restart rebuilds the
//!   index from the base), the `note.rs` rule.
//!
//! Reads are served from the in-memory index, which a write publishes only once
//! durable (the WAL fsync of `mod.rs`).

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use chrono::{DateTime, NaiveDate, Utc};
use obrain_common::types::{EdgeId, NodeId, Value};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::{
    plan, step, Ecriture, NonPorte, ObrainGraphStore, RapportOuverture, SEUIL_COLONNE_BLOB,
};
use crate::neo4j::models::*;
use crate::neo4j::plan::{compute_waves_from_graph, WaveComputationResult};
use crate::plan::models::{TaskDetails, UpdateTaskRequest};

pub(super) const LABEL_TACHE: &str = "Task";
pub(super) const LABEL_ETAPE: &str = "Step";
const LABEL_JOUR: &str = "DayPlan";
pub(super) const A_TACHE: &str = "HAS_TASK";
pub(super) const A_ETAPE: &str = "HAS_STEP";
const A_DEPEND: &str = "DEPENDS_ON";
const A_INCLUT: &str = "INCLUDES";
/// The draft marker of a creation: present from its first record to its last.
pub(super) const BROUILLON: &str = "_brouillon";
/// The marker of a deletion: set first, the node goes last.
pub(super) const SUPPRESSION: &str = "_suppression";

/// The task-and-step methods of `PlanStore` refused on every call, with their
/// cause. Listed in `plan::REFUSEES` too (the accounting).
pub const REFUS: &[(&str, &str)] = &[
    (
        "PlanStore::assign_task",
        "l'assignation écrit l'arête WORKING_ON depuis le nœud Agent (`MATCH (a:Agent)`) : \
         domaine Agent non porté ; répondre Ok sans rien écrire serait le no-op silencieux de \
         Neo4j quand l'agent manque, pas une assignation",
    ),
    (
        "PlanStore::analyze_task_impact",
        "lit les arêtes MODIFIES (Task→File) et IMPORTS (File→File) : domaine code non porté",
    ),
    (
        "PlanStore::get_task_enrichment_counts",
        "compte les notes liées à la tâche (LINKED_TO) et ses décisions (HAS_DECISION) : côté \
         tâche des domaines Note et Decision non porté",
    ),
    (
        "PlanStore::get_task_enrichment_data",
        "lit les sessions de chat lancées pour la tâche (SPAWNED_BY {task_id}), les fichiers \
         discutés (DISCUSSED→File), les notes et les décisions : côté tâche de ces domaines non porté",
    ),
];

/// The refusal of `methode`, with its cause from [`REFUS`].
pub(super) fn refus<T>(methode: &'static str) -> Result<T> {
    let cause = REFUS
        .iter()
        .find(|(m, _)| *m == methode)
        .map(|(_, c)| *c)
        .unwrap_or("non porté");
    Err(anyhow::Error::new(NonPorte(methode)).context(cause))
}

// ============================================================================
// Names and values
// ============================================================================

/// The Neo4j spelling of a task status: `format!("{:?}")`.
fn nom_statut(s: &TaskStatus) -> &'static str {
    match s {
        TaskStatus::Pending => "Pending",
        TaskStatus::InProgress => "InProgress",
        TaskStatus::Blocked => "Blocked",
        TaskStatus::Completed => "Completed",
        TaskStatus::Failed => "Failed",
        TaskStatus::Interrupted => "Interrupted",
    }
}

fn statut_depuis(s: &str) -> Option<TaskStatus> {
    Some(match s {
        "Pending" => TaskStatus::Pending,
        "InProgress" => TaskStatus::InProgress,
        "Blocked" => TaskStatus::Blocked,
        "Completed" => TaskStatus::Completed,
        "Failed" => TaskStatus::Failed,
        "Interrupted" => TaskStatus::Interrupted,
        _ => return None,
    })
}

/// A list of strings, as Neo4j stores `tags`, `acceptance_criteria`, `affected_files`.
fn liste(v: &[String]) -> Value {
    Value::List(
        v.iter()
            .map(|x| Value::from(x.clone()))
            .collect::<Vec<_>>()
            .into(),
    )
}

/// Does this value go to obrain's blob column (a string above
/// [`SEUIL_COLONNE_BLOB`], or a list holding one)? Such a value is not recovered by
/// the WAL: the write that carries it, or that replaces it, pays the full flush
/// (`mod.rs`), as in `plan.rs` and `note.rs`.
pub(super) fn va_en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
        || v.as_list().is_some_and(|l| {
            l.iter()
                .any(|e| e.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB))
        })
}

/// A list property: absent reads empty (Neo4j's `unwrap_or_default`); anything
/// but a list of strings is an error (quarantine).
fn lire_liste(get: &dyn Fn(&str) -> Option<Value>, cle: &str, id: Uuid) -> Result<Vec<String>> {
    let Some(v) = get(cle) else {
        return Ok(vec![]);
    };
    let Some(l) = v.as_list() else {
        bail!("tâche {id} : `{cle}` n'est pas une liste");
    };
    l.iter()
        .map(|e| {
            e.as_str()
                .map(str::to_string)
                .ok_or_else(|| anyhow!("tâche {id} : `{cle}` contient un non-texte"))
        })
        .collect()
}

/// A property lookup over ONE hydration of the node.
pub(super) fn lecteur(
    db: &ObrainDB,
    noeud: NodeId,
    etiquette: &str,
) -> Result<impl Fn(&str) -> Option<Value>> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud {etiquette} {noeud:?} indexé mais absent de la base"))?;
    Ok(move |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    })
}

/// The properties `Neo4jClient::create_task` writes, in the order written here.
fn proprietes_de_creation(t: &TaskNode) -> Vec<(&'static str, Value)> {
    let cree = t.created_at.to_rfc3339();
    vec![
        ("id", Value::from(t.id.to_string())),
        ("title", Value::from(t.title.clone().unwrap_or_default())),
        ("description", Value::from(t.description.clone())),
        ("status", Value::from(nom_statut(&t.status))),
        ("priority", Value::from(t.priority.unwrap_or(0) as i64)),
        ("tags", liste(&t.tags)),
        ("acceptance_criteria", liste(&t.acceptance_criteria)),
        ("affected_files", liste(&t.affected_files)),
        (
            "estimated_complexity",
            Value::from(t.estimated_complexity.map(|c| c as i64).unwrap_or(0)),
        ),
        ("frustration_score", Value::from(0.0f64)),
        (
            "due_date",
            Value::from(
                t.due_date
                    .map(|d| d.format("%Y-%m-%d").to_string())
                    .unwrap_or_default(),
            ),
        ),
        (
            "estimated_minutes",
            Value::from(t.estimated_minutes.map(|m| m as i64).unwrap_or(0)),
        ),
        ("created_at", Value::from(cree.clone())),
        ("updated_at", Value::from(cree)),
    ]
}

/// The domain value of a `Task` node: the counterpart of
/// `Neo4jClient::node_to_task`, strict where it would invent a value (an unknown
/// status, an unreadable `created_at`).
fn lire_tache(get: &dyn Fn(&str) -> Option<Value>) -> Result<TaskNode> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let instant = |cle: &str| chaine(cle).and_then(|s| s.parse::<DateTime<Utc>>().ok());
    let entier = |cle: &str| get(cle).and_then(|v| v.as_int64());
    let positif = |cle: &str| entier(cle).filter(|&v| v > 0).map(|v| v as u32);
    let non_vide = |cle: &str| chaine(cle).filter(|s| !s.is_empty());
    let id: Uuid = chaine("id")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("nœud Task sans propriété `id` lisible"))?;
    let requise =
        |cle: &str| chaine(cle).ok_or_else(|| anyhow!("tâche {id} sans propriété `{cle}`"));
    let statut_brut = requise("status")?;
    let status = statut_depuis(&statut_brut)
        .ok_or_else(|| anyhow!("tâche {id} : statut inconnu « {statut_brut} »"))?;
    let cree_brut = requise("created_at")?;
    let created_at: DateTime<Utc> = cree_brut
        .parse()
        .map_err(|e| anyhow!("tâche {id} : `created_at` illisible (« {cree_brut} ») : {e}"))?;
    Ok(TaskNode {
        id,
        title: non_vide("title"),
        description: requise("description")?,
        status,
        assigned_to: chaine("assigned_to"),
        interrupted_at: chaine("interrupted_at")
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|d| d.with_timezone(&Utc)),
        interrupted_reason: non_vide("interrupted_reason"),
        priority: entier("priority").map(|v| v as i32),
        tags: lire_liste(get, "tags", id)?,
        acceptance_criteria: lire_liste(get, "acceptance_criteria", id)?,
        affected_files: lire_liste(get, "affected_files", id)?,
        estimated_complexity: positif("estimated_complexity"),
        actual_complexity: positif("actual_complexity"),
        due_date: non_vide("due_date").and_then(|s| NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok()),
        estimated_minutes: positif("estimated_minutes"),
        created_at,
        updated_at: instant("updated_at"),
        started_at: instant("started_at"),
        completed_at: instant("completed_at"),
        frustration_score: get("frustration_score")
            .and_then(|v| v.as_float64())
            .unwrap_or(0.0),
        execution_context: non_vide("execution_context"),
        persona: non_vide("persona"),
        prompt_cache: non_vide("prompt_cache"),
    })
}

fn decoder_tache(db: &ObrainDB, noeud: NodeId) -> Result<TaskNode> {
    let get = lecteur(db, noeud, LABEL_TACHE)?;
    lire_tache(&get)
}

/// `ORDER BY COALESCE(t.priority, 0) DESC, t.created_at, t.id ASC`.
fn ordre_du_plan(a: &TaskNode, b: &TaskNode) -> Ordering {
    b.priority
        .unwrap_or(0)
        .cmp(&a.priority.unwrap_or(0))
        .then_with(|| a.created_at.cmp(&b.created_at))
        .then_with(|| a.id.cmp(&b.id))
}

/// `ORDER BY COALESCE(x.priority, 0) DESC, x.id DESC`.
fn ordre_des_dependances(a: &TaskNode, b: &TaskNode) -> Ordering {
    b.priority
        .unwrap_or(0)
        .cmp(&a.priority.unwrap_or(0))
        .then_with(|| b.id.cmp(&a.id))
}

// ============================================================================
// The index
// ============================================================================

/// One task: its node, its decoded value, its plan.
#[derive(Clone)]
pub(super) struct EntreeTache {
    pub(super) noeud: NodeId,
    pub(super) tache: TaskNode,
    pub(super) plan: Uuid,
    /// The edges of unported domains on the node, as `TYPE (sens étiquette)`.
    pub(super) etrangeres: Vec<String>,
}

/// One step: its node, its decoded value, its task.
#[derive(Clone)]
pub(super) struct EntreeEtape {
    pub(super) noeud: NodeId,
    pub(super) etape: StepNode,
    pub(super) tache: Uuid,
    pub(super) etrangeres: Vec<String>,
}

/// One `INCLUDES` edge of a day plan.
#[derive(Clone)]
struct Inclusion {
    tache: Uuid,
    ordre: i64,
    arete: EdgeId,
}

/// One `DayPlan` node.
#[derive(Clone)]
struct EntreeJour {
    noeud: NodeId,
    inclus: Vec<Inclusion>,
}

/// Tasks, steps, dependencies and day plans, rebuilt from the base at open.
#[derive(Default)]
pub(super) struct IndexTaches {
    pub(super) taches: HashMap<Uuid, EntreeTache>,
    par_plan: HashMap<Uuid, HashSet<Uuid>>,
    pub(super) etapes: HashMap<Uuid, EntreeEtape>,
    pub(super) par_tache: HashMap<Uuid, HashSet<Uuid>>,
    /// task → (the task it depends on → the `DEPENDS_ON` edge).
    deps: HashMap<Uuid, HashMap<Uuid, EdgeId>>,
    /// task → the tasks that depend on it.
    dependants: HashMap<Uuid, HashSet<Uuid>>,
    jours: HashMap<(Uuid, String), EntreeJour>,
    /// Ids of `Task` / `Step` nodes quarantined at open: neither taken again nor
    /// answered for.
    pub(super) reserves_taches: HashSet<Uuid>,
    pub(super) reserves_etapes: HashSet<Uuid>,
    /// Plans and tasks of which the base holds a part this index could not read:
    /// the reads that would miss it refuse.
    pub(super) incomplets: HashMap<Uuid, Vec<String>>,
    jours_incomplets: HashMap<(Uuid, String), Vec<String>>,
}

impl IndexTaches {
    fn inscrire_tache(&mut self, e: EntreeTache) {
        self.par_plan.entry(e.plan).or_default().insert(e.tache.id);
        self.taches.insert(e.tache.id, e);
    }

    /// A new decoded state of a task already indexed.
    fn publier_tache(&mut self, id: Uuid, tache: TaskNode) {
        if let Some(e) = self.taches.get_mut(&id) {
            e.tache = tache;
        }
    }

    pub(super) fn inscrire_etape(&mut self, e: EntreeEtape) {
        self.par_tache
            .entry(e.tache)
            .or_default()
            .insert(e.etape.id);
        self.etapes.insert(e.etape.id, e);
    }

    pub(super) fn retirer_etape(&mut self, id: Uuid) {
        if let Some(e) = self.etapes.remove(&id) {
            if let Some(s) = self.par_tache.get_mut(&e.tache) {
                s.remove(&id);
            }
        }
    }

    /// Drops a deleted task and everything the index held of it.
    fn retirer_tache(&mut self, id: Uuid) {
        for etape in self.par_tache.remove(&id).unwrap_or_default() {
            self.etapes.remove(&etape);
        }
        if let Some(e) = self.taches.remove(&id) {
            if let Some(s) = self.par_plan.get_mut(&e.plan) {
                s.remove(&id);
            }
        }
        for dep in self.deps.remove(&id).unwrap_or_default().into_keys() {
            if let Some(s) = self.dependants.get_mut(&dep) {
                s.remove(&id);
            }
        }
        for dependant in self.dependants.remove(&id).unwrap_or_default() {
            if let Some(m) = self.deps.get_mut(&dependant) {
                m.remove(&id);
            }
        }
        for jour in self.jours.values_mut() {
            jour.inclus.retain(|i| i.tache != id);
        }
    }

    /// The task `id` for a read: `Err` when its id is reserved by a quarantined
    /// node, `Ok(None)` when the base holds no such task.
    pub(super) fn tache(&self, id: Uuid) -> Result<Option<&EntreeTache>> {
        if self.reserves_taches.contains(&id) {
            bail!(
                "la tâche {id} est en quarantaine (nœud illisible à l'ouverture) : le backend obrain \
                 ne répond pas pour elle"
            );
        }
        Ok(self.taches.get(&id))
    }

    /// `Err` when the base holds a part of `id` (plan or task) this index could not read.
    pub(super) fn complet(&self, id: Uuid) -> Result<()> {
        if let Some(raisons) = self.incomplets.get(&id) {
            bail!(
                "{id} : la base en contient une partie que l'index n'a pas pu lire ({}) ; la \
                 réponse serait partielle, elle est refusée",
                raisons.join(" ; ")
            );
        }
        Ok(())
    }

    fn noter_incomplet(&mut self, id: Uuid, raison: String) {
        self.incomplets.entry(id).or_default().push(raison);
    }

    /// The tasks of a plan, in the plan-task order.
    fn taches_du_plan(&self, plan: Uuid) -> Vec<&TaskNode> {
        let mut v: Vec<&TaskNode> = self
            .par_plan
            .get(&plan)
            .into_iter()
            .flatten()
            .filter_map(|id| self.taches.get(id).map(|e| &e.tache))
            .collect();
        v.sort_by(|a, b| ordre_du_plan(a, b));
        v
    }

    /// The tasks `id` depends on (unsorted).
    fn dependances(&self, id: Uuid) -> Vec<&TaskNode> {
        self.deps
            .get(&id)
            .into_iter()
            .flat_map(|m| m.keys())
            .filter_map(|d| self.taches.get(d).map(|e| &e.tache))
            .collect()
    }

    /// The tasks depending on `id` (unsorted).
    fn dependantes(&self, id: Uuid) -> Vec<&TaskNode> {
        self.dependants
            .get(&id)
            .into_iter()
            .flatten()
            .filter_map(|d| self.taches.get(d).map(|e| &e.tache))
            .collect()
    }

    /// The tasks of the plan and the `DEPENDS_ON` edges between two of them.
    fn graphe(&self, plan: Uuid) -> (Vec<TaskNode>, Vec<(Uuid, Uuid)>) {
        let taches = self.taches_du_plan(plan);
        let rang: HashMap<Uuid, usize> =
            taches.iter().enumerate().map(|(i, t)| (t.id, i)).collect();
        let mut aretes = Vec::new();
        for t in &taches {
            let mut cibles: Vec<Uuid> = self
                .deps
                .get(&t.id)
                .into_iter()
                .flat_map(|m| m.keys())
                .filter(|d| rang.contains_key(d))
                .copied()
                .collect();
            cibles.sort_by_key(|d| rang[d]);
            aretes.extend(cibles.into_iter().map(|d| (t.id, d)));
        }
        (taches.into_iter().cloned().collect(), aretes)
    }

    /// A plan's reads refuse when the base holds a part of it, or of one of its
    /// tasks, that this index could not read.
    fn plan_complet(&self, plan: Uuid) -> Result<()> {
        self.complet(plan)?;
        for id in self.par_plan.get(&plan).into_iter().flatten() {
            self.complet(*id)?;
        }
        Ok(())
    }

    /// Rebuilds the index at open: drafts are deleted, marked deletions finished,
    /// unreadable nodes quarantined (their ids reserved), and what refers to them
    /// marked incomplete. A scan of the `Task`, `Step` and `DayPlan` nodes and of
    /// their edges, paid at open only.
    pub(super) fn reconstruire(
        db: &ObrainDB,
        plans: &plan::IndexPlans,
        rapport: &mut RapportOuverture,
    ) -> Result<Self> {
        let store = db.store();
        let mut index = Self::default();
        let plan_de_noeud: HashMap<NodeId, Uuid> =
            plans.par_id.iter().map(|(id, e)| (e.noeud, *id)).collect();
        let mut tache_de_noeud: HashMap<NodeId, Uuid> = HashMap::new();
        let mut etape_de_noeud: HashMap<NodeId, Uuid> = HashMap::new();

        let type_de = |a: EdgeId| db.get_edge(a).map(|e| e.edge_type.to_string());
        let etiquette = |n: NodeId| {
            db.get_node_labels(n)
                .and_then(|l| l.into_iter().next())
                .unwrap_or_default()
        };

        // ---- Tasks ----
        for noeud in store.nodes_by_label(LABEL_TACHE) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud Task {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            let isole = store.out_degree(noeud) == 0 && store.in_degree(noeud) == 0;
            if get(BROUILLON).is_some() || (n.properties.is_empty() && isole) {
                supprimer_a_l_ouverture(db, noeud);
                rapport.brouillons_supprimes += 1;
                continue;
            }
            if get(SUPPRESSION).is_some() {
                for (cible, a) in store.edges_from(noeud, Direction::Outgoing) {
                    if type_de(a).as_deref() == Some(A_ETAPE) && etiquette(cible) == LABEL_ETAPE {
                        supprimer_a_l_ouverture(db, cible);
                    }
                }
                supprimer_a_l_ouverture(db, noeud);
                rapport.brouillons_supprimes += 1;
                continue;
            }
            // The parent: exactly one HAS_TASK from a plan this backend holds.
            let parents: Vec<NodeId> = store
                .edges_from(noeud, Direction::Incoming)
                .into_iter()
                .filter(|(_, a)| type_de(*a).as_deref() == Some(A_TACHE))
                .map(|(src, _)| src)
                .collect();
            let plans_parents: Vec<Uuid> = parents
                .iter()
                .filter_map(|p| plan_de_noeud.get(p).copied())
                .collect();
            let id_brut = get("id").and_then(|v| v.as_str().and_then(|s| s.parse::<Uuid>().ok()));
            let decode = lire_tache(&get).and_then(|t| match parents.len() {
                1 if plans_parents.len() == 1 => Ok(t),
                0 => Err(anyhow!("tâche {} sans plan (aucune arête HAS_TASK)", t.id)),
                1 => Err(anyhow!(
                    "tâche {} rattachée à un plan que l'index ne tient pas (en quarantaine)",
                    t.id
                )),
                k => Err(anyhow!("tâche {} rattachée à {k} plans", t.id)),
            });
            let tache = match decode {
                Ok(t) if !index.taches.contains_key(&t.id) => t,
                Ok(t) => {
                    rapport
                        .en_quarantaine
                        .push(format!("nœud Task {noeud:?} : id {} déjà pris", t.id));
                    for p in &plans_parents {
                        index.noter_incomplet(*p, format!("tâche {} en double", t.id));
                    }
                    continue;
                }
                Err(e) => {
                    if let Some(id) = id_brut {
                        index.reserves_taches.insert(id);
                    }
                    for p in &plans_parents {
                        index.noter_incomplet(*p, format!("nœud Task {noeud:?} en quarantaine"));
                    }
                    rapport
                        .en_quarantaine
                        .push(format!("nœud Task {noeud:?} : {e}"));
                    continue;
                }
            };
            tache_de_noeud.insert(noeud, tache.id);
            index.inscrire_tache(EntreeTache {
                noeud,
                plan: plans_parents[0],
                tache,
                etrangeres: vec![],
            });
        }

        // A plan quarantined by `plan.rs` (unreadable, or a second node of a taken
        // id): its tasks cannot be read, so its plan-level reads refuse rather than
        // answer "no task".
        for noeud in store.nodes_by_label("Plan") {
            if plan_de_noeud.contains_key(&noeud) {
                continue;
            }
            let id = db.get_node(noeud).and_then(|n| {
                n.get_property("id")
                    .and_then(|v| v.as_str().and_then(|s| s.parse::<Uuid>().ok()))
            });
            if let Some(id) = id {
                index.noter_incomplet(id, format!("nœud Plan {noeud:?} en quarantaine"));
            }
        }

        // ---- Steps ----
        for noeud in store.nodes_by_label(LABEL_ETAPE) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud Step {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            let isole = store.out_degree(noeud) == 0 && store.in_degree(noeud) == 0;
            if get(BROUILLON).is_some()
                || get(SUPPRESSION).is_some()
                || (n.properties.is_empty() && isole)
            {
                supprimer_a_l_ouverture(db, noeud);
                rapport.brouillons_supprimes += 1;
                continue;
            }
            let parents: Vec<NodeId> = store
                .edges_from(noeud, Direction::Incoming)
                .into_iter()
                .filter(|(_, a)| type_de(*a).as_deref() == Some(A_ETAPE))
                .map(|(src, _)| src)
                .collect();
            let taches_parentes: Vec<Uuid> = parents
                .iter()
                .filter_map(|p| tache_de_noeud.get(p).copied())
                .collect();
            let id_brut = get("id").and_then(|v| v.as_str().and_then(|s| s.parse::<Uuid>().ok()));
            let decode = step::lire_etape(&get).and_then(|s| match parents.len() {
                1 if taches_parentes.len() == 1 => Ok(s),
                0 => Err(anyhow!("étape {} sans tâche (aucune arête HAS_STEP)", s.id)),
                1 => Err(anyhow!(
                    "étape {} rattachée à une tâche que l'index ne tient pas (en quarantaine)",
                    s.id
                )),
                k => Err(anyhow!("étape {} rattachée à {k} tâches", s.id)),
            });
            match decode {
                Ok(s) if !index.etapes.contains_key(&s.id) => {
                    etape_de_noeud.insert(noeud, s.id);
                    index.inscrire_etape(EntreeEtape {
                        noeud,
                        tache: taches_parentes[0],
                        etape: s,
                        etrangeres: vec![],
                    });
                }
                Ok(s) => {
                    rapport
                        .en_quarantaine
                        .push(format!("nœud Step {noeud:?} : id {} déjà pris", s.id));
                    for t in &taches_parentes {
                        index.noter_incomplet(*t, format!("étape {} en double", s.id));
                    }
                }
                Err(e) => {
                    if let Some(id) = id_brut {
                        index.reserves_etapes.insert(id);
                    }
                    for t in &taches_parentes {
                        index.noter_incomplet(*t, format!("nœud Step {noeud:?} en quarantaine"));
                    }
                    rapport
                        .en_quarantaine
                        .push(format!("nœud Step {noeud:?} : {e}"));
                }
            }
        }

        // ---- Day plans ----
        for noeud in store.nodes_by_label(LABEL_JOUR) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud DayPlan {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            let isole = store.out_degree(noeud) == 0 && store.in_degree(noeud) == 0;
            if get(BROUILLON).is_some() || (n.properties.is_empty() && isole) {
                supprimer_a_l_ouverture(db, noeud);
                rapport.brouillons_supprimes += 1;
                continue;
            }
            let utilisateur = get("user_id").and_then(|v| v.as_str().and_then(|s| s.parse().ok()));
            let date = get("date").and_then(|v| v.as_str().map(str::to_string));
            let (Some(utilisateur), Some(date)) = (utilisateur, date) else {
                rapport.en_quarantaine.push(format!(
                    "nœud DayPlan {noeud:?} : `user_id` ou `date` illisible"
                ));
                continue;
            };
            let cle: (Uuid, String) = (utilisateur, date);
            if index.jours.contains_key(&cle) {
                rapport.en_quarantaine.push(format!(
                    "nœud DayPlan {noeud:?} : ({}, {}) déjà pris",
                    cle.0, cle.1
                ));
                index
                    .jours_incomplets
                    .entry(cle)
                    .or_default()
                    .push(format!("nœud DayPlan {noeud:?} en double"));
                continue;
            }
            let mut inclus = Vec::new();
            for (cible, a) in store.edges_from(noeud, Direction::Outgoing) {
                let Some(e) = db.get_edge(a) else { continue };
                if &*e.edge_type != A_INCLUT {
                    index
                        .jours_incomplets
                        .entry(cle.clone())
                        .or_default()
                        .push(format!("arête {} hors du domaine", e.edge_type));
                    continue;
                }
                let Some(ordre) = e.get_property("order").and_then(|v| v.as_int64()) else {
                    // A creation of the edge cut before its `order`: repaired.
                    db.delete_edge(a);
                    rapport.brouillons_supprimes += 1;
                    continue;
                };
                match tache_de_noeud.get(&cible) {
                    Some(t) => inclus.push(Inclusion {
                        tache: *t,
                        ordre,
                        arete: a,
                    }),
                    None => index
                        .jours_incomplets
                        .entry(cle.clone())
                        .or_default()
                        .push(format!(
                            "INCLUDES vers {cible:?}, qui n'est pas une tâche indexée"
                        )),
                }
            }
            index.jours.insert(cle, EntreeJour { noeud, inclus });
        }

        // ---- Dependencies and foreign edges ----
        let mut etrangeres_taches: Vec<(Uuid, Vec<String>)> = Vec::new();
        let mut incomplets: Vec<(Uuid, String)> = Vec::new();
        for (noeud, id) in &tache_de_noeud {
            let mut etrangeres = Vec::new();
            for (cible, a) in store.edges_from(*noeud, Direction::Outgoing) {
                let t = type_de(a).unwrap_or_default();
                match t.as_str() {
                    A_DEPEND => match tache_de_noeud.get(&cible) {
                        Some(dep) => {
                            let m = index.deps.entry(*id).or_default();
                            if m.insert(*dep, a).is_some() {
                                incomplets.push((*id, format!("DEPENDS_ON en double vers {dep}")));
                            }
                            index.dependants.entry(*dep).or_default().insert(*id);
                        }
                        None => incomplets.push((
                            *id,
                            format!("DEPENDS_ON vers {cible:?}, qui n'est pas une tâche indexée"),
                        )),
                    },
                    A_ETAPE if etiquette(cible) == LABEL_ETAPE => {
                        // Indexed, or quarantined (and then already noted).
                    }
                    _ => etrangeres.push(format!("{t} (sortante vers {})", etiquette(cible))),
                }
            }
            for (source, a) in store.edges_from(*noeud, Direction::Incoming) {
                let t = type_de(a).unwrap_or_default();
                match t.as_str() {
                    A_TACHE if plan_de_noeud.contains_key(&source) => {}
                    A_DEPEND => {
                        if !tache_de_noeud.contains_key(&source) {
                            incomplets.push((
                                *id,
                                format!(
                                    "DEPENDS_ON depuis {source:?}, qui n'est pas une tâche indexée"
                                ),
                            ));
                        }
                    }
                    A_INCLUT if etiquette(source) == LABEL_JOUR => {}
                    _ => etrangeres.push(format!("{t} (entrante depuis {})", etiquette(source))),
                }
            }
            if !etrangeres.is_empty() {
                etrangeres_taches.push((*id, etrangeres));
            }
        }
        for (id, e) in etrangeres_taches {
            if let Some(entree) = index.taches.get_mut(&id) {
                entree.etrangeres = e;
            }
        }
        for (id, raison) in incomplets {
            index.noter_incomplet(id, raison);
        }
        for (noeud, id) in &etape_de_noeud {
            let mut etrangeres = Vec::new();
            for (cible, a) in store.edges_from(*noeud, Direction::Outgoing) {
                let t = type_de(a).unwrap_or_default();
                etrangeres.push(format!("{t} (sortante vers {})", etiquette(cible)));
            }
            for (source, a) in store.edges_from(*noeud, Direction::Incoming) {
                let t = type_de(a).unwrap_or_default();
                if !(t == A_ETAPE && tache_de_noeud.contains_key(&source)) {
                    etrangeres.push(format!("{t} (entrante depuis {})", etiquette(source)));
                }
            }
            if let Some(e) = index.etapes.get_mut(id) {
                e.etrangeres = etrangeres;
            }
        }
        // A task with an unreadable part makes its plan's reads refuse too.
        let a_propager: Vec<(Uuid, String)> = index
            .incomplets
            .keys()
            .filter_map(|id| {
                index
                    .taches
                    .get(id)
                    .map(|e| (e.plan, format!("tâche {id} incomplète")))
            })
            .collect();
        for (plan, raison) in a_propager {
            index.noter_incomplet(plan, raison);
        }
        Ok(index)
    }
}

/// At open: a node and its edges go (a draft, or a marked deletion to finish).
fn supprimer_a_l_ouverture(db: &ObrainDB, noeud: NodeId) {
    let store = db.store();
    for (_, a) in store.edges_from(noeud, Direction::Outgoing) {
        db.delete_edge(a);
    }
    for (_, a) in store.edges_from(noeud, Direction::Incoming) {
        db.delete_edge(a);
    }
    db.delete_node(noeud);
}

// ============================================================================
// Write helpers
// ============================================================================

/// Runs the engine writes of one call. An error once they have started (an
/// injected interruption in the tests; an engine refusal in production) makes the
/// store INDETERMINATE: what is on disk is a prefix of the writes, the index does
/// not know which, and only a restart (which rebuilds the index) knows.
pub(super) fn ecritures<T>(
    w: &mut Ecriture<'_>,
    f: impl FnOnce(&mut Ecriture<'_>) -> Result<T>,
) -> Result<T> {
    f(w).map_err(|e| {
        w.interne
            .indetermine
            .store(true, std::sync::atomic::Ordering::Release);
        e.context(format!(
            "base obrain {} : une écriture s'est arrêtée après sa première écriture moteur ; \
             l'état sur disque est un préfixe de ces écritures. Tout est refusé jusqu'au \
             redémarrage, qui reconstruit l'index (et répare brouillons et suppressions marquées)",
            w.interne.chemin.display()
        ))
    })
}

/// Sets (`Some`) or removes (`None`) each property, one engine write each.
pub(super) fn poser(
    w: &mut Ecriture<'_>,
    noeud: NodeId,
    champs: Vec<(&'static str, Option<Value>)>,
) -> Result<()> {
    // The NEW value going to the blob column, or the OLD one it replaces or
    // removes (clearing that slot is not recovered by the WAL either): the full
    // flush runs before the write is acknowledged.
    let ancien = lecteur(w.db, noeud, "indexé")?;
    if champs.iter().any(|(cle, v)| {
        v.as_ref().is_some_and(va_en_colonne_blob)
            || ancien(cle).is_some_and(|o| va_en_colonne_blob(&o))
    }) {
        w.flush_complet = true;
    }
    for (cle, valeur) in champs {
        w.etape()?;
        match valeur {
            Some(v) => w.db.set_node_property(noeud, cle, v),
            None => {
                w.db.remove_node_property(noeud, cle);
            }
        }
    }
    Ok(())
}

/// A creation under the draft marker: node, marker, properties, the edge from
/// its parent, then the marker removed (the commit point). Call inside [`ecritures`].
pub(super) fn creer_sous_brouillon(
    w: &mut Ecriture<'_>,
    etiquette: &str,
    proprietes: Vec<(&'static str, Value)>,
    parent: Option<(NodeId, &str)>,
) -> Result<NodeId> {
    w.flush_complet = proprietes.iter().any(|(_, v)| va_en_colonne_blob(v));
    w.etape()?;
    let noeud = w.db.create_node(&[etiquette]);
    w.etape()?;
    w.db.set_node_property(noeud, BROUILLON, Value::from(true));
    for (cle, valeur) in proprietes {
        w.etape()?;
        w.db.set_node_property(noeud, cle, valeur);
    }
    if let Some((source, type_arete)) = parent {
        w.etape()?;
        w.db.create_edge(source, noeud, type_arete);
    }
    // A blob-sized value reaches the disk BEFORE the commit point: a crash before
    // the flush leaves a draft (deleted at open), never a node missing a value.
    w.rendre_durable_avant_le_marqueur()?;
    w.etape()?;
    w.db.remove_node_property(noeud, BROUILLON);
    Ok(noeud)
}

/// DETACH DELETE of a node: its edges, then the node. Call inside [`ecritures`].
pub(super) fn detacher(w: &mut Ecriture<'_>, noeud: NodeId) -> Result<()> {
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
        bail!("obrain a refusé la suppression du nœud {noeud:?}, indexé");
    }
    Ok(())
}

fn maintenant() -> Value {
    Value::from(Utc::now().to_rfc3339())
}

// ============================================================================
// The methods (`plan.rs` delegates to them)
// ============================================================================

impl ObrainGraphStore {
    // ---------------------------------------------------------------- tasks

    pub(super) async fn tache_creer(&self, plan_id: Uuid, task: &TaskNode) -> Result<()> {
        let t = task.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud_plan = {
                    let etat = w.index()?;
                    if etat.taches.taches.contains_key(&t.id) {
                        bail!("une tâche d'id {} existe déjà (unicité de Task.id)", t.id);
                    }
                    if etat.taches.reserves_taches.contains(&t.id) {
                        bail!(
                            "l'id {} est réservé : un nœud Task de cet id est sur disque, \
                             illisible à l'ouverture (quarantaine). Il n'est ni repris ni écrasé",
                            t.id
                        );
                    }
                    let Some(p) = etat.plans.par_id.get(&plan_id) else {
                        bail!(
                            "le plan {plan_id} n'existe pas : la tâche {} n'est pas créée (Neo4j \
                             ne créerait rien et répondrait Ok ; le backend obrain le dit)",
                            t.id
                        );
                    };
                    p.noeud
                };
                let proprietes = proprietes_de_creation(&t);
                let noeud = ecritures(w, |w| {
                    creer_sous_brouillon(w, LABEL_TACHE, proprietes, Some((noeud_plan, A_TACHE)))
                })?;
                let tache = w.apres_ecriture(decoder_tache(w.db, noeud).and_then(|r| {
                    if r.id == t.id {
                        Ok(r)
                    } else {
                        bail!("la tâche relue n'a pas l'id écrit")
                    }
                }))?;
                w.rendre_durable()?;
                w.index_mut()?.taches.inscrire_tache(EntreeTache {
                    noeud,
                    tache,
                    plan: plan_id,
                    etrangeres: vec![],
                });
                Ok(())
            },
        )
        .await
    }

    pub(super) async fn tache_lire(&self, task_id: Uuid) -> Result<Option<TaskNode>> {
        self.lire(|_, etat| Ok(etat.taches.tache(task_id)?.map(|e| e.tache.clone())))
    }

    pub(super) async fn taches_du_plan(&self, plan_id: Uuid) -> Result<Vec<TaskNode>> {
        self.lire(|_, etat| {
            etat.taches.plan_complet(plan_id)?;
            Ok(etat
                .taches
                .taches_du_plan(plan_id)
                .into_iter()
                .cloned()
                .collect())
        })
    }

    pub(super) async fn tache_plan(&self, task_id: Uuid) -> Result<Option<Uuid>> {
        self.lire(|_, etat| Ok(etat.taches.tache(task_id)?.map(|e| e.plan)))
    }

    pub(super) async fn tache_projet(&self, task_id: Uuid) -> Result<Option<ProjectNode>> {
        self.lire(|_, etat| {
            let Some(e) = etat.taches.tache(task_id)? else {
                return Ok(None);
            };
            Ok(etat
                .plans
                .par_id
                .get(&e.plan)
                .and_then(|p| p.plan.project_id)
                .and_then(|p| etat.projets.projet(p).cloned()))
        })
    }

    pub(super) async fn tache_details(&self, task_id: Uuid) -> Result<Option<TaskDetails>> {
        self.lire(|_, etat| {
            let idx = &etat.taches;
            let Some(e) = idx.tache(task_id)? else {
                return Ok(None);
            };
            idx.complet(task_id)?;
            let hors: Vec<&String> = e
                .etrangeres
                .iter()
                .filter(|a| a.starts_with("INFORMED_BY ") || a.starts_with("MODIFIES "))
                .collect();
            if !hors.is_empty() {
                return Err(
                    anyhow::Error::new(NonPorte("PlanStore::get_task_with_full_details")).context(
                        format!(
                            "la tâche {task_id} porte des arêtes de domaines non portés ({}) : \
                             ses décisions et ses fichiers ne peuvent pas être lus ici",
                            hors.iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    ),
                );
            }
            let mut depends_on: Vec<Uuid> = idx
                .deps
                .get(&task_id)
                .into_iter()
                .flat_map(|m| m.keys())
                .copied()
                .collect();
            depends_on.sort();
            Ok(Some(TaskDetails {
                task: e.tache.clone(),
                steps: step::etapes_triees(idx, task_id),
                // No ported domain can link a decision or a file to a task (see the
                // module doc), and a foreign one refuses above: these are exact.
                decisions: vec![],
                depends_on,
                modifies_files: vec![],
            }))
        })
    }

    pub(super) async fn tache_maj(&self, task_id: Uuid, updates: &UpdateTaskRequest) -> Result<()> {
        let u = updates.clone();
        let mut champs: Vec<(&'static str, Option<Value>)> = Vec::new();
        if let Some(t) = &u.title {
            champs.push(("title", Some(Value::from(t.clone()))));
        }
        if let Some(d) = &u.description {
            champs.push(("description", Some(Value::from(d.clone()))));
        }
        if let Some(p) = u.priority {
            champs.push(("priority", Some(Value::from(p as i64))));
        }
        if let Some(t) = &u.tags {
            champs.push(("tags", Some(liste(t))));
        }
        if let Some(c) = &u.acceptance_criteria {
            champs.push(("acceptance_criteria", Some(liste(c))));
        }
        if let Some(f) = &u.affected_files {
            champs.push(("affected_files", Some(liste(f))));
        }
        if let Some(c) = u.actual_complexity {
            champs.push(("actual_complexity", Some(Value::from(c as i64))));
        }
        if let Some(c) = u.estimated_complexity {
            champs.push(("estimated_complexity", Some(Value::from(c as i64))));
        }
        if let Some(d) = u.due_date {
            champs.push((
                "due_date",
                Some(Value::from(d.format("%Y-%m-%d").to_string())),
            ));
        }
        if let Some(m) = u.estimated_minutes {
            champs.push(("estimated_minutes", Some(Value::from(m as i64))));
        }
        if let Some(a) = &u.assigned_to {
            champs.push(("assigned_to", Some(Value::from(a.clone()))));
        }
        // Nothing to set (a `status` alone included): nothing is written, as Neo4j.
        if champs.is_empty() {
            return Ok(());
        }
        champs.push(("updated_at", Some(maintenant())));
        self.tache_ecrire(task_id, move |_| Ok(Some(champs)))
            .await
            .map(|_| ())
    }

    /// Writes the properties `champs` computes from the current task (`None`:
    /// nothing to write) and publishes the new state. Answers whether it wrote.
    /// A task the base does not hold is a silent no-op, as Neo4j's `MATCH`.
    async fn tache_ecrire<F>(&self, task_id: Uuid, champs: F) -> Result<bool>
    where
        F: FnOnce(&TaskNode) -> Result<Option<Vec<(&'static str, Option<Value>)>>> + Send + 'static,
    {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, champs) = {
                    let etat = w.index()?;
                    let Some(e) = etat.taches.tache(task_id)? else {
                        return Ok(false);
                    };
                    match champs(&e.tache)? {
                        Some(c) => (e.noeud, c),
                        None => return Ok(false),
                    }
                };
                ecritures(w, |w| poser(w, noeud, champs))?;
                let tache = w.apres_ecriture(decoder_tache(w.db, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.taches.publier_tache(task_id, tache);
                Ok(true)
            },
        )
        .await
    }

    pub(super) async fn tache_statut(&self, task_id: Uuid, status: TaskStatus) -> Result<()> {
        self.tache_ecrire(task_id, move |t| {
            let now = maintenant();
            let mut champs = vec![("status", Some(Value::from(nom_statut(&status))))];
            match status {
                TaskStatus::InProgress => champs.push(("started_at", Some(now.clone()))),
                TaskStatus::Completed | TaskStatus::Failed => {
                    champs.push(("completed_at", Some(now.clone())))
                }
                _ => {}
            }
            champs.push(("updated_at", Some(now)));
            champs.push(("interrupted_at", None));
            champs.push(("interrupted_reason", None));
            // Last: of the writes of this call, the only one a retry does not repeat
            // identically. Once it is on disk, every other one is (a WAL prefix).
            if status == TaskStatus::Blocked {
                let f = (t.frustration_score + 0.2).min(1.0);
                champs.push(("frustration_score", Some(Value::from(f))));
            }
            Ok(Some(champs))
        })
        .await
        .map(|_| ())
    }

    pub(super) async fn tache_interrompre(&self, task_id: Uuid, reason: &str) -> Result<bool> {
        let raison = reason.to_owned();
        self.tache_ecrire(task_id, move |t| {
            if !matches!(
                t.status,
                TaskStatus::Pending
                    | TaskStatus::InProgress
                    | TaskStatus::Blocked
                    | TaskStatus::Interrupted
            ) {
                return Ok(None);
            }
            let now = maintenant();
            let depuis = match t.interrupted_at {
                Some(d) => Value::from(d.to_rfc3339()),
                None => now.clone(),
            };
            Ok(Some(vec![
                ("status", Some(Value::from("Interrupted"))),
                ("interrupted_at", Some(depuis)),
                ("interrupted_reason", Some(Value::from(raison))),
                ("updated_at", Some(now)),
            ]))
        })
        .await
    }

    pub(super) async fn tache_enrichissement(
        &self,
        task_id: Uuid,
        execution_context: Option<&str>,
        persona: Option<&str>,
        prompt_cache: Option<&str>,
    ) -> Result<()> {
        let mut champs: Vec<(&'static str, Option<Value>)> = Vec::new();
        if let Some(c) = execution_context {
            champs.push(("execution_context", Some(Value::from(c.to_string()))));
        }
        if let Some(p) = persona {
            champs.push(("persona", Some(Value::from(p.to_string()))));
        }
        if let Some(p) = prompt_cache {
            champs.push(("prompt_cache", Some(Value::from(p.to_string()))));
        }
        if champs.is_empty() {
            return Ok(());
        }
        champs.push(("updated_at", Some(maintenant())));
        self.tache_ecrire(task_id, move |_| Ok(Some(champs)))
            .await
            .map(|_| ())
    }

    pub(super) async fn tache_supprimer(&self, task_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, etapes) = {
                    let etat = w.index()?;
                    let idx = &etat.taches;
                    let Some(e) = idx.tache(task_id)? else {
                        return Ok(());
                    };
                    idx.complet(task_id)?;
                    let etapes: Vec<&EntreeEtape> = idx
                        .par_tache
                        .get(&task_id)
                        .into_iter()
                        .flatten()
                        .filter_map(|s| idx.etapes.get(s))
                        .collect();
                    let hors: Vec<String> = e
                        .etrangeres
                        .iter()
                        .cloned()
                        .chain(etapes.iter().flat_map(|s| s.etrangeres.iter().cloned()))
                        .collect();
                    if !hors.is_empty() {
                        return Err(anyhow::Error::new(NonPorte(
                            "PlanStore::delete_task (cascade)",
                        ))
                        .context(format!(
                            "la tâche {task_id} ou ses étapes portent des arêtes de domaines non \
                             portés ({}) : la suppression en cascade (décisions comprises) ne peut \
                             pas les atteindre ici",
                            hors.join(", ")
                        )));
                    }
                    (e.noeud, etapes.iter().map(|s| s.noeud).collect::<Vec<_>>())
                };
                ecritures(w, |w| {
                    w.etape()?;
                    w.db.set_node_property(noeud, SUPPRESSION, Value::from(true));
                    // Each step is marked too BEFORE any edge goes: a step whose
                    // HAS_STEP is already gone is then still recognised at open as a
                    // deletion to finish, never an orphan to quarantine.
                    for s in &etapes {
                        w.etape()?;
                        w.db.set_node_property(*s, SUPPRESSION, Value::from(true));
                    }
                    for s in etapes {
                        detacher(w, s)?;
                    }
                    detacher(w, noeud)
                })?;
                w.rendre_durable()?;
                w.index_mut()?.taches.retirer_tache(task_id);
                Ok(())
            },
        )
        .await
    }

    // ---------------------------------------------------------- dependencies

    pub(super) async fn dependance_ajouter(&self, task_id: Uuid, depends_on: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (de, vers) = {
                    let etat = w.index()?;
                    let idx = &etat.taches;
                    let (Some(a), Some(b)) = (idx.tache(task_id)?, idx.tache(depends_on)?) else {
                        // Neo4j's MATCH: an absent task makes it a no-op.
                        return Ok(());
                    };
                    // MERGE: an existing edge is kept, not doubled.
                    if idx
                        .deps
                        .get(&task_id)
                        .is_some_and(|m| m.contains_key(&depends_on))
                    {
                        return Ok(());
                    }
                    (a.noeud, b.noeud)
                };
                let arete = ecritures(w, |w| {
                    w.etape()?;
                    Ok(w.db.create_edge(de, vers, A_DEPEND))
                })?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.taches
                    .deps
                    .entry(task_id)
                    .or_default()
                    .insert(depends_on, arete);
                etat.taches
                    .dependants
                    .entry(depends_on)
                    .or_default()
                    .insert(task_id);
                Ok(())
            },
        )
        .await
    }

    pub(super) async fn dependance_retirer(&self, task_id: Uuid, depends_on: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let arete = {
                    let etat = w.index()?;
                    let idx = &etat.taches;
                    idx.tache(task_id)?;
                    idx.tache(depends_on)?;
                    match idx.deps.get(&task_id).and_then(|m| m.get(&depends_on)) {
                        Some(a) => *a,
                        None => return Ok(()),
                    }
                };
                ecritures(w, |w| {
                    w.etape()?;
                    if !w.db.delete_edge(arete) {
                        bail!("obrain a refusé la suppression de l'arête DEPENDS_ON {arete:?}");
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(m) = etat.taches.deps.get_mut(&task_id) {
                    m.remove(&depends_on);
                }
                if let Some(s) = etat.taches.dependants.get_mut(&depends_on) {
                    s.remove(&task_id);
                }
                Ok(())
            },
        )
        .await
    }

    /// `quoi`: the tasks `task_id` depends on (`true`) or that depend on it.
    /// `seulement_non_terminees`: only those not `Completed` (the blockers).
    fn voisins(
        &self,
        task_id: Uuid,
        dependances: bool,
        seulement_non_terminees: bool,
    ) -> Result<Vec<TaskNode>> {
        self.lire(|_, etat| {
            let idx = &etat.taches;
            if idx.tache(task_id)?.is_none() {
                return Ok(vec![]);
            }
            idx.complet(task_id)?;
            let mut v: Vec<TaskNode> = if dependances {
                idx.dependances(task_id)
            } else {
                idx.dependantes(task_id)
            }
            .into_iter()
            .filter(|t| !seulement_non_terminees || t.status != TaskStatus::Completed)
            .cloned()
            .collect();
            v.sort_by(ordre_des_dependances);
            Ok(v)
        })
    }

    pub(super) async fn dependances_de(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        self.voisins(task_id, true, false)
    }

    pub(super) async fn bloqueurs_de(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        self.voisins(task_id, true, true)
    }

    pub(super) async fn bloquees_par(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        self.voisins(task_id, false, false)
    }

    pub(super) async fn taches_bloquees(
        &self,
        plan_id: Uuid,
    ) -> Result<Vec<(TaskNode, Vec<TaskNode>)>> {
        self.lire(|_, etat| {
            let idx = &etat.taches;
            idx.plan_complet(plan_id)?;
            let mut r = Vec::new();
            for t in idx.taches_du_plan(plan_id) {
                if t.status != TaskStatus::Pending {
                    continue;
                }
                let mut bloqueurs: Vec<TaskNode> = idx
                    .dependances(t.id)
                    .into_iter()
                    .filter(|d| d.status != TaskStatus::Completed)
                    .cloned()
                    .collect();
                if bloqueurs.is_empty() {
                    continue;
                }
                bloqueurs.sort_by(ordre_des_dependances);
                r.push((t.clone(), bloqueurs));
            }
            Ok(r)
        })
    }

    pub(super) async fn tache_suivante(&self, plan_id: Uuid) -> Result<Option<TaskNode>> {
        self.lire(|_, etat| {
            let idx = &etat.taches;
            idx.plan_complet(plan_id)?;
            Ok(idx
                .taches_du_plan(plan_id)
                .into_iter()
                .find(|t| {
                    t.status == TaskStatus::Pending
                        && idx
                            .dependances(t.id)
                            .iter()
                            .all(|d| d.status == TaskStatus::Completed)
                })
                .cloned())
        })
    }

    pub(super) async fn graphe_du_plan(
        &self,
        plan_id: Uuid,
    ) -> Result<(Vec<TaskNode>, Vec<(Uuid, Uuid)>)> {
        self.lire(|_, etat| {
            etat.taches.plan_complet(plan_id)?;
            Ok(etat.taches.graphe(plan_id))
        })
    }

    pub(super) async fn chemin_critique(&self, plan_id: Uuid) -> Result<Vec<TaskNode>> {
        self.lire(|_, etat| {
            let idx = &etat.taches;
            idx.plan_complet(plan_id)?;
            chemin_critique(idx, plan_id)
        })
    }

    pub(super) async fn vagues(&self, plan_id: Uuid) -> Result<WaveComputationResult> {
        let (taches, aretes) = self.graphe_du_plan(plan_id).await?;
        compute_waves_from_graph(&taches, aretes)
    }

    // -------------------------------------------------------------- day plan

    pub(super) async fn jour_lire(&self, user_id: Uuid, date: &str) -> Result<Vec<(Uuid, u32)>> {
        let cle = (user_id, date.to_string());
        self.lire(move |_, etat| {
            let idx = &etat.taches;
            if let Some(r) = idx.jours_incomplets.get(&cle) {
                bail!(
                    "plan du jour ({}, {}) : la base en contient une partie illisible ({}) ; \
                     refusé plutôt que partiel",
                    cle.0,
                    cle.1,
                    r.join(" ; ")
                );
            }
            let Some(j) = idx.jours.get(&cle) else {
                return Ok(vec![]);
            };
            let mut v: Vec<(i64, Uuid)> = j.inclus.iter().map(|i| (i.ordre, i.tache)).collect();
            v.sort();
            Ok(v.into_iter().map(|(o, t)| (t, o as u32)).collect())
        })
    }

    /// The common part of the three day-plan writes: refuses an incomplete day
    /// plan and a quarantined task id.
    fn jour_verifier(idx: &IndexTaches, cle: &(Uuid, String), ids: &[Uuid]) -> Result<()> {
        if let Some(r) = idx.jours_incomplets.get(cle) {
            bail!(
                "plan du jour ({}, {}) : la base en contient une partie illisible ({}) ; \
                 l'écriture est refusée",
                cle.0,
                cle.1,
                r.join(" ; ")
            );
        }
        for id in ids {
            idx.tache(*id)?;
        }
        Ok(())
    }

    /// The `DayPlan` node of `cle`: the existing one (its `updated_at` set), or a
    /// new one under the draft marker. Call inside [`ecritures`].
    fn jour_noeud(
        w: &mut Ecriture<'_>,
        existant: Option<NodeId>,
        cle: &(Uuid, String),
    ) -> Result<NodeId> {
        match existant {
            Some(n) => {
                poser(w, n, vec![("updated_at", Some(maintenant()))])?;
                Ok(n)
            }
            None => creer_sous_brouillon(
                w,
                LABEL_JOUR,
                vec![
                    ("user_id", Value::from(cle.0.to_string())),
                    ("date", Value::from(cle.1.clone())),
                    ("updated_at", maintenant()),
                ],
                None,
            ),
        }
    }

    pub(super) async fn jour_fixer(&self, user_id: Uuid, date: &str, ids: &[Uuid]) -> Result<()> {
        let cle = (user_id, date.to_string());
        let ids = ids.to_vec();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (existant, anciennes, cibles) = {
                    let etat = w.index()?;
                    let idx = &etat.taches;
                    Self::jour_verifier(idx, &cle, &ids)?;
                    let j = idx.jours.get(&cle);
                    // Position in `ids` is the order, unknown ids skipped (their
                    // position kept, as Neo4j's UNWIND before its MATCH).
                    let cibles: Vec<(Uuid, NodeId, i64)> = ids
                        .iter()
                        .enumerate()
                        .filter_map(|(i, id)| idx.taches.get(id).map(|e| (*id, e.noeud, i as i64)))
                        .collect();
                    (
                        j.map(|j| j.noeud),
                        j.map(|j| j.inclus.iter().map(|i| i.arete).collect::<Vec<_>>())
                            .unwrap_or_default(),
                        cibles,
                    )
                };
                let (noeud, inclus) = ecritures(w, |w| {
                    let noeud = Self::jour_noeud(w, existant, &cle)?;
                    for a in anciennes {
                        w.etape()?;
                        w.db.delete_edge(a);
                    }
                    let mut inclus = Vec::new();
                    for (tache, cible, ordre) in cibles {
                        w.etape()?;
                        let arete = w.db.create_edge_with_props(
                            noeud,
                            cible,
                            A_INCLUT,
                            [("order", Value::from(ordre))],
                        );
                        inclus.push(Inclusion {
                            tache,
                            ordre,
                            arete,
                        });
                    }
                    Ok((noeud, inclus))
                })?;
                w.rendre_durable()?;
                w.index_mut()?
                    .taches
                    .jours
                    .insert(cle, EntreeJour { noeud, inclus });
                Ok(())
            },
        )
        .await
    }

    pub(super) async fn jour_ajouter(
        &self,
        user_id: Uuid,
        date: &str,
        task_id: Uuid,
    ) -> Result<()> {
        let cle = (user_id, date.to_string());
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (existant, cible, ordre) = {
                    let etat = w.index()?;
                    let idx = &etat.taches;
                    Self::jour_verifier(idx, &cle, &[task_id])?;
                    let Some(t) = idx.taches.get(&task_id) else {
                        // No such task: Neo4j's MATCH stops there (module doc).
                        return Ok(());
                    };
                    let j = idx.jours.get(&cle);
                    let deja = j.is_some_and(|j| j.inclus.iter().any(|i| i.tache == task_id));
                    let ordre = j
                        .and_then(|j| j.inclus.iter().map(|i| i.ordre).max())
                        .map_or(0, |m| m + 1);
                    (j.map(|j| j.noeud), (!deja).then_some(t.noeud), ordre)
                };
                let (noeud, nouvelle) = ecritures(w, |w| {
                    let noeud = Self::jour_noeud(w, existant, &cle)?;
                    let nouvelle = match cible {
                        Some(c) => {
                            w.etape()?;
                            Some(w.db.create_edge_with_props(
                                noeud,
                                c,
                                A_INCLUT,
                                [("order", Value::from(ordre))],
                            ))
                        }
                        None => None,
                    };
                    Ok((noeud, nouvelle))
                })?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                let j = etat.taches.jours.entry(cle).or_insert(EntreeJour {
                    noeud,
                    inclus: vec![],
                });
                if let Some(arete) = nouvelle {
                    j.inclus.push(Inclusion {
                        tache: task_id,
                        ordre,
                        arete,
                    });
                }
                Ok(())
            },
        )
        .await
    }

    pub(super) async fn jour_retirer(
        &self,
        user_id: Uuid,
        date: &str,
        task_id: Uuid,
    ) -> Result<()> {
        let cle = (user_id, date.to_string());
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, aretes) = {
                    let etat = w.index()?;
                    let idx = &etat.taches;
                    Self::jour_verifier(idx, &cle, &[task_id])?;
                    let Some(j) = idx.jours.get(&cle) else {
                        return Ok(());
                    };
                    let aretes: Vec<EdgeId> = j
                        .inclus
                        .iter()
                        .filter(|i| i.tache == task_id)
                        .map(|i| i.arete)
                        .collect();
                    (j.noeud, aretes)
                };
                ecritures(w, |w| {
                    poser(w, noeud, vec![("updated_at", Some(maintenant()))])?;
                    for a in &aretes {
                        w.etape()?;
                        w.db.delete_edge(*a);
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                if let Some(j) = w.index_mut()?.taches.jours.get_mut(&cle) {
                    j.inclus.retain(|i| i.tache != task_id);
                }
                Ok(())
            },
        )
        .await
    }
}

// ============================================================================
// The critical path
// ============================================================================

/// The Neo4j query of `get_plan_critical_path`, computed on the index:
///
/// ```text
/// start: a plan task with no DEPENDS_ON to any task;
/// end:   a plan task no task DEPENDS_ON;
/// path = (start)<-[:DEPENDS_ON*0..]-(end), every node in the plan;
/// ORDER BY length(path) DESC, [ids of the path] DESC LIMIT 1
/// ```
///
/// On an acyclic graph a path visits each node once, so the longest one is a
/// longest chain, found by dynamic programming over the dependants, and the
/// greatest id list among the longest is built greedily: lists of equal length
/// compare element by element, so the greatest first id wins, then the greatest
/// next one among those that still reach the length. A cycle among the plan's
/// tasks is refused (module doc).
fn chemin_critique(idx: &IndexTaches, plan: Uuid) -> Result<Vec<TaskNode>> {
    let membres: HashSet<Uuid> = idx.par_plan.get(&plan).cloned().unwrap_or_default();
    if membres.is_empty() {
        return Ok(vec![]);
    }
    // In the direction of the path: from a task to the plan tasks depending on it.
    let suivants = |v: Uuid| -> Vec<Uuid> {
        idx.dependants
            .get(&v)
            .into_iter()
            .flatten()
            .filter(|u| membres.contains(u))
            .copied()
            .collect()
    };
    // Kahn on the plan's subgraph: a topological order, or a cycle.
    let mut degre: HashMap<Uuid, usize> = membres.iter().map(|v| (*v, 0)).collect();
    for v in &membres {
        for u in suivants(*v) {
            *degre.get_mut(&u).expect("member") += 1;
        }
    }
    let mut file: Vec<Uuid> = degre
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(v, _)| *v)
        .collect();
    let mut ordre = Vec::with_capacity(membres.len());
    while let Some(v) = file.pop() {
        ordre.push(v);
        for u in suivants(v) {
            let d = degre.get_mut(&u).expect("member");
            *d -= 1;
            if *d == 0 {
                file.push(u);
            }
        }
    }
    if ordre.len() < membres.len() {
        bail!(
            "le plan {plan} a un cycle de dépendances : le chemin critique n'est pas défini (Neo4j \
             parcourrait les arêtes du cycle une fois chacune ; le backend obrain refuse)"
        );
    }
    // best[v]: length of the longest path from v to an end, if any.
    let mut meilleur: HashMap<Uuid, Option<usize>> = HashMap::new();
    for v in ordre.iter().rev() {
        let est_fin = idx.dependants.get(v).is_none_or(|s| s.is_empty());
        let l = if est_fin {
            Some(0)
        } else {
            suivants(*v)
                .into_iter()
                .filter_map(|u| meilleur[&u].map(|l| l + 1))
                .max()
        };
        meilleur.insert(*v, l);
    }
    let est_debut = |v: &Uuid| idx.deps.get(v).is_none_or(|m| m.is_empty());
    let Some(longueur) = membres
        .iter()
        .filter(|v| est_debut(v))
        .filter_map(|v| meilleur[v])
        .max()
    else {
        return Ok(vec![]);
    };
    let mut courant = *membres
        .iter()
        .filter(|v| est_debut(v) && meilleur[*v] == Some(longueur))
        .max()
        .expect("a start of that length");
    let mut chemin = vec![courant];
    for reste in (0..longueur).rev() {
        courant = suivants(courant)
            .into_iter()
            .filter(|u| meilleur[u] == Some(reste))
            .max()
            .expect("a successor one step shorter");
        chemin.push(courant);
    }
    Ok(chemin
        .into_iter()
        .filter_map(|id| idx.taches.get(&id).map(|e| e.tache.clone()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::PlanStore;
    use crate::obrain::{Creation, Interruption};
    use chrono::TimeZone;

    fn un_plan() -> PlanNode {
        PlanNode {
            id: Uuid::new_v4(),
            title: "plan".into(),
            description: "d".into(),
            status: PlanStatus::Draft,
            created_at: Utc::now(),
            created_by: "test".into(),
            priority: 0,
            project_id: None,
            execution_context: None,
            persona: None,
        }
    }

    fn une_tache(titre: &str, priorite: Option<i32>, decalage: i64) -> TaskNode {
        TaskNode {
            id: Uuid::new_v4(),
            title: Some(titre.into()),
            description: format!("tâche {titre}"),
            status: TaskStatus::Pending,
            assigned_to: None,
            interrupted_at: None,
            interrupted_reason: None,
            priority: priorite,
            tags: vec!["a".into(), "b".into()],
            acceptance_criteria: vec!["ok".into()],
            affected_files: vec![],
            estimated_complexity: Some(3),
            actual_complexity: None,
            due_date: NaiveDate::from_ymd_opt(2026, 12, 1),
            estimated_minutes: Some(30),
            created_at: Utc.timestamp_opt(1_700_000_000 + decalage, 0).unwrap(),
            updated_at: None,
            started_at: None,
            completed_at: None,
            frustration_score: 0.0,
            execution_context: None,
            persona: None,
            prompt_cache: None,
        }
    }

    #[tokio::test]
    async fn chemin_critique_et_bris_d_egalite_par_ids() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let p = un_plan();
        store.create_plan(&p).await.unwrap();
        // a <- b <- c (c depends on b, b on a) ; d isolated ; e <- f.
        let ts: Vec<TaskNode> = ["a", "b", "c", "d", "e", "f"]
            .iter()
            .enumerate()
            .map(|(i, t)| une_tache(t, Some(1), i as i64))
            .collect();
        for t in &ts {
            store.create_task(p.id, t).await.unwrap();
        }
        store.add_task_dependency(ts[1].id, ts[0].id).await.unwrap();
        store.add_task_dependency(ts[2].id, ts[1].id).await.unwrap();
        store.add_task_dependency(ts[5].id, ts[4].id).await.unwrap();
        let chemin: Vec<Uuid> = store
            .get_plan_critical_path(p.id)
            .await
            .unwrap()
            .into_iter()
            .map(|t| t.id)
            .collect();
        assert_eq!(chemin, vec![ts[0].id, ts[1].id, ts[2].id]);
        // Two chains of equal length: the greatest id list wins.
        store.add_task_dependency(ts[3].id, ts[5].id).await.unwrap();
        let chemin: Vec<Uuid> = store
            .get_plan_critical_path(p.id)
            .await
            .unwrap()
            .into_iter()
            .map(|t| t.id)
            .collect();
        let attendu = if ts[0].id > ts[4].id {
            vec![ts[0].id, ts[1].id, ts[2].id]
        } else {
            vec![ts[4].id, ts[5].id, ts[3].id]
        };
        assert_eq!(chemin, attendu);
        // A cycle is refused.
        store.add_task_dependency(ts[0].id, ts[2].id).await.unwrap();
        assert!(store.get_plan_critical_path(p.id).await.is_err());
        assert!(store.compute_waves(p.id).await.is_err());
    }

    #[tokio::test]
    async fn suppression_marquee_interrompue_est_achevee_a_l_ouverture() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = un_plan();
        let t = une_tache("t", None, 0);
        let s = StepNode {
            id: Uuid::new_v4(),
            order: 0,
            description: "s".into(),
            status: StepStatus::Pending,
            verification: None,
            created_at: Utc::now(),
            updated_at: None,
            completed_at: None,
            execution_context: None,
            persona: None,
        };
        // Stop delete_task after its k-th engine write, for every k, then reopen.
        for k in 1..=12 {
            let base = chemin.join(format!("k{k}"));
            let store = ObrainGraphStore::open(&base, Creation::Permise).unwrap();
            store.create_plan(&p).await.unwrap();
            store.create_task(p.id, &t).await.unwrap();
            store.create_step(t.id, &s).await.unwrap();
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let r = store.delete_task(t.id).await;
            let atteint = r.is_err();
            if atteint {
                // Indeterminate: every operation refuses until a restart.
                assert!(store.get_task(t.id).await.is_err());
            }
            store.close().await.unwrap();
            drop(store);
            let store = ObrainGraphStore::open(&base, Creation::Interdite).unwrap();
            assert!(
                store.rapport_d_ouverture().en_quarantaine.is_empty(),
                "k={k}"
            );
            let relue = store.get_task(t.id).await.unwrap();
            let etape = store.get_step(s.id).await.unwrap();
            // Either nothing was deleted (stopped before the marker) or all of it.
            assert_eq!(relue.is_some(), etape.is_some(), "k={k}");
            if relue.is_some() {
                store.delete_task(t.id).await.unwrap();
            }
            assert!(store.get_task(t.id).await.unwrap().is_none(), "k={k}");
            assert!(store.get_step(s.id).await.unwrap().is_none(), "k={k}");
            store.close().await.unwrap();
            if !atteint {
                break;
            }
        }
    }
}
