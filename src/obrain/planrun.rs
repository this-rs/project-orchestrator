//! `PlanRunStore` on obrain: the runs of a plan, the plan triggers with their
//! firings and signal reservations, the journal of external events and the
//! delivery claims (lot O-PLANRUN). The alerts, event triggers and lifecycle
//! hooks are `TriggerStore`'s (`trigger.rs`, whose node helpers this file uses).
//!
//! PORTED here, with the Neo4j semantics of `src/neo4j/plan_run.rs`,
//! `src/neo4j/trigger.rs` and `src/neo4j/ingress.rs` (23 of the 32 methods):
//! - plan runs: `create_plan_run`, `update_plan_run`, `get_plan_run`,
//!   `list_active_plan_runs`, `list_all_plan_runs`, `list_plan_runs`;
//! - plan triggers: `create_trigger`, `get_trigger`, `list_triggers`,
//!   `list_all_triggers`, `disable_trigger`, `enable_trigger_as`,
//!   `delete_trigger`, `record_trigger_firing`, `reserve_trigger_signal`,
//!   `list_trigger_firings`, and the plan's third-party mark
//!   (`mark_third_party_write` for a `PlanContent::Plan`,
//!   `plan_third_party_written_at`);
//! - ingress: `record_external_event`, `list_external_events`,
//!   `claim_external_delivery`, `release_external_delivery`,
//!   `purge_external_events_before`.
//!
//! REFUSED by name (`NonPorte`, see [`REFUSEES`]), never answered empty:
//! - the six `AgentExecution` methods: an execution is created only with its
//!   `Task` (`MATCH (t:Task …)`, `EXECUTES`), a domain not ported; with no way to
//!   create one, every read would be an empty answer;
//! - `get_all_pagerank_values`, `get_community_risk_vectors`,
//!   `get_all_risk_score_values`: they read the code graph (`File`, `Function`),
//!   not ported.
//!
//! And ONE method refuses only in some cases: `mark_third_party_write` of a
//! task, step, constraint or decision refuses by name (its own name, with the
//! cause in the context): finding the plan of that content goes through those
//! domains.
//!
//! # Layout
//!
//! One node per entity, labelled `PlanRun`, `Trigger`, `TriggerFiring`,
//! `TriggerSignal`, `ExternalEvent`, `ExternalDelivery`, with the properties
//! Neo4j writes (a run keeps its whole state in `state_json`, and the flat
//! fields beside it; a trigger's `config` and `author` are JSON strings). The
//! links are PROPERTIES, as in `plan.rs`: `plan_id` (Neo4j: `RUNS`, `TRIGGERS`),
//! `trigger_id` (`FIRED_BY`), `plan_run_id` (`STARTED`). The third-party mark is
//! the `third_party_written_at` property of the `Plan` node, as in Neo4j.
//!
//! # Divergences
//!
//! Strict side:
//! - `create_plan_run` and `create_trigger` of a plan that does not exist are
//!   REFUSED. Neo4j's `MATCH (p:Plan …)` silently writes nothing (and
//!   `create_trigger` then returns the trigger as if it were stored).
//! - `record_trigger_firing` of a trigger that does not exist is REFUSED (Neo4j
//!   writes nothing, silently).
//! - `update_plan_run` of a run that does not exist is REFUSED (« PlanRun not
//!   found »); Neo4j's `MATCH … SET` is a silent no-op.
//! - An id already taken (run, trigger, firing, external event) is refused;
//!   Neo4j `CREATE`s a second node with the same id.
//! - The keys of a quarantined node stay RESERVED and are refused by name: an
//!   unreadable delivery is never claimed again (Neo4j's `MERGE` would match it
//!   and answer `false`), an unreadable signal never `Reserved` again in its
//!   window (a second run), an unreadable run, trigger, firing or event id never
//!   created again.
//! - A negative `limit` / `offset` is refused (as Neo4j refuses a negative
//!   `SKIP` / `LIMIT`, with a driver error).
//! - A stored value that does not decode — a `state_json`, a trigger type, a
//!   `config` or `author` that is not JSON, a status, a date, a list of task ids —
//!   puts the node in QUARANTINE at open (named, untouched, never served). Neo4j
//!   reads it as a default (`Event`, `Null`, no author, `Running`) or falls back
//!   to the flat fields. An unreadable third-party mark makes the methods that
//!   read it (`plan_third_party_written_at`, `create_trigger`,
//!   `enable_trigger_as`) refuse for that plan, where Neo4j reads `None`.
//!
//! Read side:
//! - A run is read from `state_json` (as Neo4j, which prefers it); the lists
//!   order and filter on the decoded state, which the flat fields mirror.
//! - The workspace of `list_all_plan_runs` is the plan's `project_id` (the only
//!   project link `plan.rs` writes).
//!
//! # Atomicity
//!
//! A creation follows the draft-marker protocol of `project.rs`. Two writes touch
//! two nodes, and each is ordered so that what a crash leaves is repaired at open:
//! - `record_trigger_firing` commits the firing, THEN raises the trigger's
//!   `fire_count`. At open, a trigger whose `fire_count` is below the number of
//!   its firings is raised to it (the firings are never deleted without their
//!   trigger, so in Neo4j the two are equal).
//! - `reserve_trigger_signal` commits the signal, THEN moves `last_fired` to the
//!   same instant. At open, a trigger whose `last_fired` is before its latest
//!   signal is moved to it.
//!
//! The guards — one signal per `(trigger, key)` and the cooldown, one claim per
//! `(trigger, delivery key)`, one node per id — are checked under the writer lock:
//! of N concurrent reservations of one key exactly one is `Reserved`.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::trigger::{
    analyser, booleen, borne, chaine, charger, cle_reservee, compte, creer, creer_noeud,
    date_facultative, date_requise, entier, facultative, flottant, indexer, instant,
    json_facultatif, poser, reecrire, refus_reserve, relire, requise, supprimer, texte,
    uuid_facultatif, uuid_ou_vide, uuid_requis, Fiche, Lecteur,
};
use super::{Ecriture, NonPorte, ObrainGraphStore, RapportOuverture};
use crate::ingress::{ExternalEvent, IngressOutcome};
use crate::neo4j::agent_execution::{AgentExecutionNode, CostBreakdownRow, CostGroupBy};
use crate::neo4j::traits::PlanRunStore;
use crate::runner::{
    PlanContent, PlanRunStatus, RunnerState, SignalReservation, Trigger, TriggerAuthor,
    TriggerFiring, TriggerSource, TriggerType,
};

const RUN: &str = "PlanRun";
const DECLENCHEUR: &str = "Trigger";
const TIR: &str = "TriggerFiring";
const SIGNAL: &str = "TriggerSignal";
const EVENEMENT: &str = "ExternalEvent";
const LIVRAISON: &str = "ExternalDelivery";
/// The third-party mark, on the `Plan` node.
const MARQUE: &str = "third_party_written_at";
/// How long a reserved signal is remembered (`TRIGGER_SIGNAL_RETENTION_HOURS`
/// of the Neo4j adapter, private to it).
const RETENTION_SIGNAL_HEURES: i64 = 24;
/// How many expired signals one reservation purges (Neo4j: `LIMIT 1000`).
const PURGE_SIGNAUX: usize = 1000;

/// The methods of `PlanRunStore` refused on every call, as
/// `PlanRunStore::<method>`: the agent executions (they need a `Task`) and the
/// statistics of the code graph. `mark_third_party_write` refuses only for a
/// content other than a plan, and is not listed.
pub const REFUSEES: &[&str] = &[
    "PlanRunStore::create_agent_execution",
    "PlanRunStore::update_agent_execution",
    "PlanRunStore::get_agent_executions_for_run",
    "PlanRunStore::record_agent_execution_verification",
    "PlanRunStore::list_running_agent_executions",
    "PlanRunStore::agent_execution_cost_breakdown",
    "PlanRunStore::get_all_pagerank_values",
    "PlanRunStore::get_community_risk_vectors",
    "PlanRunStore::get_all_risk_score_values",
];

/// Refuses a method that obrain does not port, by name, with its cause.
fn refus<T>(methode: &'static str, cause: &str) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).context(cause.to_string()))
}

const CAUSE_TACHE: &str = "une exécution d'agent se crée avec sa tâche (EXECUTES) : le domaine \
                           Task n'est pas porté, aucune exécution ne peut exister ici";
const CAUSE_CODE: &str = "lit le graphe de code (File, Function), domaine non porté";

// ============================================================================
// Plan runs
// ============================================================================

fn liste_uuid(ids: &[Uuid]) -> Value {
    texte(
        ids.iter()
            .map(|u| u.to_string())
            .collect::<Vec<_>>()
            .join(","),
    )
}

fn lire_liste_uuid(brut: &str, cle: &str, quoi: &str) -> Result<Vec<Uuid>> {
    if brut.is_empty() {
        return Ok(Vec::new());
    }
    brut.split(',')
        .map(|s| analyser(s.trim(), cle, quoi))
        .collect()
}

/// The flat fields Neo4j writes on a run, at creation and at every update.
fn proprietes_communes_run(s: &RunnerState) -> Result<Vec<(&'static str, Value)>> {
    Ok(vec![
        ("current_wave", Value::from(i64::try_from(s.current_wave)?)),
        ("status", texte(s.status.to_string())),
        ("cost_usd", Value::from(s.cost_usd)),
        ("notional_usd", Value::from(s.notional_usd)),
        ("tokens_used", Value::from(i64::try_from(s.tokens_used)?)),
        ("total_tasks", Value::from(i64::try_from(s.total_tasks)?)),
        ("completed_tasks", liste_uuid(&s.completed_tasks)),
        ("failed_tasks", liste_uuid(&s.failed_tasks)),
        ("state_json", texte(serde_json::to_string(s)?)),
    ])
}

fn proprietes_creation_run(s: &RunnerState) -> Result<Vec<(&'static str, Value)>> {
    let mut p = vec![
        ("run_id", texte(s.run_id.to_string())),
        ("plan_id", texte(s.plan_id.to_string())),
        ("git_branch", texte(s.git_branch.clone())),
        ("started_at", instant(s.started_at)),
        (
            "triggered_by",
            texte(serde_json::to_string(&s.triggered_by)?),
        ),
    ];
    p.extend(proprietes_communes_run(s)?);
    Ok(p)
}

fn statut_run(brut: &str, quoi: &str) -> Result<PlanRunStatus> {
    Ok(match brut {
        "running" => PlanRunStatus::Running,
        "completed" => PlanRunStatus::Completed,
        "completed_with_errors" => PlanRunStatus::CompletedWithErrors,
        "failed" => PlanRunStatus::Failed,
        "cancelled" => PlanRunStatus::Cancelled,
        "budget_exceeded" => PlanRunStatus::BudgetExceeded,
        "interrupted" => PlanRunStatus::Interrupted,
        // Neo4j reads an unknown status as `running`: a finished run would be resumed.
        _ => bail!("{quoi} : `status` inconnu (« {brut} »)"),
    })
}

fn lire_run(get: Lecteur) -> Result<RunnerState> {
    let run_id = uuid_requis(get, "run_id", "nœud PlanRun")?;
    let quoi = format!("exécution de plan {run_id}");
    let quoi = quoi.as_str();
    if let Some(brut) = facultative(get, "state_json", quoi)? {
        let etat: RunnerState = serde_json::from_str(&brut)
            .map_err(|e| anyhow!("{quoi} : `state_json` illisible : {e}"))?;
        if etat.run_id != run_id {
            bail!(
                "{quoi} : `state_json` porte le run_id {} (le nœud : {run_id})",
                etat.run_id
            );
        }
        return Ok(etat);
    }
    // A node written before `state_json`: its flat fields, as Neo4j reads them.
    let entier_non_signe = |cle: &str| -> Result<usize> {
        match entier(get, cle, quoi)? {
            Some(n) => borne(n, cle, quoi),
            None => Ok(0),
        }
    };
    Ok(RunnerState {
        run_id,
        plan_id: uuid_requis(get, "plan_id", quoi)?,
        total_tasks: entier_non_signe("total_tasks")?,
        current_wave: entier_non_signe("current_wave")?,
        current_task_id: uuid_facultatif(get, "current_task_id", quoi)?,
        current_task_title: None,
        active_agents: Vec::new(),
        completed_tasks: lire_liste_uuid(
            &chaine(get, "completed_tasks", quoi)?.unwrap_or_default(),
            "completed_tasks",
            quoi,
        )?,
        failed_tasks: lire_liste_uuid(
            &chaine(get, "failed_tasks", quoi)?.unwrap_or_default(),
            "failed_tasks",
            quoi,
        )?,
        retry_counts: HashMap::new(),
        git_branch: chaine(get, "git_branch", quoi)?.unwrap_or_default(),
        started_at: date_requise(get, "started_at", quoi)?,
        completed_at: date_facultative(get, "completed_at", quoi)?,
        status: statut_run(&requise(get, "status", quoi)?, quoi)?,
        cost_usd: flottant(get, "cost_usd", quoi)?.unwrap_or(0.0),
        notional_usd: flottant(get, "notional_usd", quoi)?.unwrap_or(0.0),
        tokens_used: match entier(get, "tokens_used", quoi)? {
            Some(n) => borne(n, "tokens_used", quoi)?,
            None => 0,
        },
        unknown_cost_attempts: 0,
        triggered_by: json_facultatif::<TriggerSource>(get, "triggered_by", quoi)?
            .unwrap_or(TriggerSource::Manual),
        project_id: uuid_facultatif(get, "project_id", quoi)?,
        lifecycle_run_id: uuid_facultatif(get, "lifecycle_run_id", quoi)?,
        cwd: None,
    })
}

/// `ORDER BY started_at DESC, run_id DESC`.
fn ordre_runs(a: &RunnerState, b: &RunnerState) -> std::cmp::Ordering {
    b.started_at
        .cmp(&a.started_at)
        .then_with(|| b.run_id.cmp(&a.run_id))
}

// ============================================================================
// Plan triggers, firings, signals
// ============================================================================

fn nom_type(t: &TriggerType) -> String {
    t.to_string()
}

fn type_depuis(brut: &str, quoi: &str) -> Result<TriggerType> {
    Ok(match brut {
        "schedule" => TriggerType::Schedule,
        "webhook" => TriggerType::Webhook,
        "event" => TriggerType::Event,
        "chat" => TriggerType::Chat,
        // Neo4j reads an unknown type as `event`.
        _ => bail!("{quoi} : `trigger_type` inconnu (« {brut} »)"),
    })
}

fn auteur(a: Option<&TriggerAuthor>) -> Result<Value> {
    Ok(texte(match a {
        Some(a) => serde_json::to_string(a)?,
        None => String::new(),
    }))
}

fn proprietes_declencheur(
    t: &Trigger,
    marque: Option<DateTime<Utc>>,
) -> Result<Vec<(&'static str, Value)>> {
    let mut p = vec![
        ("id", texte(t.id.to_string())),
        ("plan_id", texte(t.plan_id.to_string())),
        ("trigger_type", texte(nom_type(&t.trigger_type))),
        ("config", texte(serde_json::to_string(&t.config)?)),
        ("enabled", Value::from(t.enabled)),
        (
            "cooldown_secs",
            Value::from(i64::try_from(t.cooldown_secs)?),
        ),
        ("fire_count", Value::from(0_i64)),
        ("created_at", instant(t.created_at)),
        ("author", auteur(t.author.as_ref())?),
    ];
    if let Some(m) = marque {
        p.push(("approved_mark", instant(m)));
    }
    Ok(p)
}

fn lire_declencheur(get: Lecteur) -> Result<Trigger> {
    let id = uuid_requis(get, "id", "nœud Trigger")?;
    let quoi = format!("déclencheur de plan {id}");
    let quoi = quoi.as_str();
    let enabled = booleen(get, "enabled", quoi)?.unwrap_or(true);
    Ok(Trigger {
        id,
        plan_id: uuid_requis(get, "plan_id", quoi)?,
        trigger_type: type_depuis(&requise(get, "trigger_type", quoi)?, quoi)?,
        config: json_facultatif(get, "config", quoi)?.unwrap_or(serde_json::Value::Null),
        enabled,
        cooldown_secs: match entier(get, "cooldown_secs", quoi)? {
            Some(n) => borne(n, "cooldown_secs", quoi)?,
            None => 0,
        },
        last_fired: date_facultative(get, "last_fired", quoi)?,
        fire_count: match entier(get, "fire_count", quoi)? {
            Some(n) => borne(n, "fire_count", quoi)?,
            None => 0,
        },
        created_at: date_requise(get, "created_at", quoi)?,
        // An author that does not decode is an error: read as `None`, it would
        // silently change who the trigger's runs start as.
        author: json_facultatif(get, "author", quoi)?,
        // A disabled_reason only means something on a disabled trigger.
        disabled_reason: facultative(get, "disabled_reason", quoi)?.filter(|_| !enabled),
        approved_mark: date_facultative(get, "approved_mark", quoi)?,
    })
}

fn proprietes_tir(f: &TriggerFiring) -> Result<Vec<(&'static str, Value)>> {
    Ok(vec![
        ("id", texte(f.id.to_string())),
        ("trigger_id", texte(f.trigger_id.to_string())),
        ("fired_at", instant(f.fired_at)),
        (
            "source_payload",
            texte(match &f.source_payload {
                Some(p) => serde_json::to_string(p)?,
                None => String::new(),
            }),
        ),
        ("plan_run_id", uuid_ou_vide(f.plan_run_id)),
        (
            "start_error",
            texte(f.start_error.clone().unwrap_or_default()),
        ),
    ])
}

fn lire_tir(get: Lecteur) -> Result<TriggerFiring> {
    let id = uuid_requis(get, "id", "nœud TriggerFiring")?;
    let quoi = format!("tir {id}");
    let quoi = quoi.as_str();
    Ok(TriggerFiring {
        id,
        trigger_id: uuid_requis(get, "trigger_id", quoi)?,
        plan_run_id: uuid_facultatif(get, "plan_run_id", quoi)?,
        fired_at: date_requise(get, "fired_at", quoi)?,
        source_payload: json_facultatif(get, "source_payload", quoi)?,
        start_error: facultative(get, "start_error", quoi)?,
    })
}

/// One reserved signal: `(trigger, key)` and when.
#[derive(Clone)]
struct Signal {
    trigger_id: Uuid,
    cle: String,
    reserve_a: DateTime<Utc>,
}

fn proprietes_signal(s: &Signal) -> Vec<(&'static str, Value)> {
    vec![
        ("trigger_id", texte(s.trigger_id.to_string())),
        ("key", texte(s.cle.clone())),
        ("reserved_at", instant(s.reserve_a)),
    ]
}

fn lire_signal(get: Lecteur) -> Result<Signal> {
    let trigger_id = uuid_requis(get, "trigger_id", "nœud TriggerSignal")?;
    let quoi = format!("signal du déclencheur {trigger_id}");
    let quoi = quoi.as_str();
    Ok(Signal {
        trigger_id,
        cle: requise(get, "key", quoi)?,
        reserve_a: date_requise(get, "reserved_at", quoi)?,
    })
}

// ============================================================================
// Ingress
// ============================================================================

fn nom_issue(o: IngressOutcome) -> &'static str {
    match o {
        IngressOutcome::Accepted => "accepted",
        IngressOutcome::Rejected => "rejected",
    }
}

fn proprietes_evenement(e: &ExternalEvent) -> Result<Vec<(&'static str, Value)>> {
    let opt = |s: &Option<String>| texte(s.clone().unwrap_or_default());
    Ok(vec![
        ("id", texte(e.id.to_string())),
        ("trigger_id", texte(e.trigger_id.to_string())),
        ("source", texte(e.source.clone())),
        ("outcome", texte(nom_issue(e.outcome))),
        ("reason", opt(&e.reason)),
        ("delivery_id", opt(&e.delivery_id)),
        ("event_type", opt(&e.event_type)),
        ("action", opt(&e.action)),
        ("body_bytes", Value::from(i64::try_from(e.body_bytes)?)),
        ("received_at", instant(e.received_at)),
    ])
}

fn lire_evenement(get: Lecteur) -> Result<ExternalEvent> {
    let id = uuid_requis(get, "id", "nœud ExternalEvent")?;
    let quoi = format!("événement externe {id}");
    let quoi = quoi.as_str();
    let issue = requise(get, "outcome", quoi)?;
    Ok(ExternalEvent {
        id,
        trigger_id: uuid_requis(get, "trigger_id", quoi)?,
        source: requise(get, "source", quoi)?,
        outcome: match issue.as_str() {
            "accepted" => IngressOutcome::Accepted,
            "rejected" => IngressOutcome::Rejected,
            _ => bail!("{quoi} : `outcome` inconnu (« {issue} »)"),
        },
        reason: facultative(get, "reason", quoi)?,
        delivery_id: facultative(get, "delivery_id", quoi)?,
        event_type: facultative(get, "event_type", quoi)?,
        action: facultative(get, "action", quoi)?,
        body_bytes: borne(
            entier(get, "body_bytes", quoi)?
                .ok_or_else(|| anyhow!("{quoi} sans propriété `body_bytes`"))?,
            "body_bytes",
            quoi,
        )?,
        received_at: date_requise(get, "received_at", quoi)?,
    })
}

/// One claimed delivery.
struct Livraison {
    trigger_id: Uuid,
    cle: String,
}

fn lire_livraison(get: Lecteur) -> Result<Livraison> {
    let trigger_id = uuid_requis(get, "trigger_id", "nœud ExternalDelivery")?;
    let quoi = format!("livraison du déclencheur {trigger_id}");
    let quoi = quoi.as_str();
    // `claimed_at` is read to be checked, not kept.
    date_requise(get, "claimed_at", quoi)?;
    Ok(Livraison {
        trigger_id,
        cle: requise(get, "delivery_key", quoi)?,
    })
}

// ============================================================================
// The index
// ============================================================================

/// The third-party mark of a plan, read from its node.
enum Marque {
    /// The mark, and the plan node it was read on (a plan deleted and created
    /// again under the same id is another node, without this mark).
    Lue(NodeId, DateTime<Utc>),
    /// Present but not a date: every method that reads it refuses for this plan.
    Illisible(NodeId, String),
}

/// Plan runs, plan triggers and what hangs on them, by id, rebuilt at open.
#[derive(Default)]
pub(super) struct IndexExecutions {
    runs: HashMap<Uuid, Fiche<RunnerState>>,
    declencheurs: HashMap<Uuid, Fiche<Trigger>>,
    tirs: HashMap<Uuid, Fiche<TriggerFiring>>,
    signaux: HashMap<(Uuid, String), Fiche<Signal>>,
    evenements: HashMap<Uuid, Fiche<ExternalEvent>>,
    livraisons: HashMap<(Uuid, String), NodeId>,
    marques: HashMap<Uuid, Marque>,
    /// The keys of the quarantined nodes ([`cle_reservee`]): an id, a signal, a
    /// delivery that is on disk but unreadable is never taken, reserved or
    /// claimed a second time.
    reserves: HashSet<String>,
}

impl IndexExecutions {
    /// Refuses, by name, a key a quarantined node of `label` holds.
    fn garde_reserve(&self, label: &str, parties: &[&str], quoi: &str) -> Result<()> {
        if self.reserves.contains(&cle_reservee(label, parties)) {
            return Err(refus_reserve(quoi));
        }
        Ok(())
    }

    /// Rebuilds the index: drafts deleted, unreadable nodes quarantined, and the
    /// two repairs of the module doc (a `fire_count` below the firings, a
    /// `last_fired` before the latest signal), made durable before serving.
    pub(super) fn reconstruire(
        db: &ObrainDB,
        plans: &super::plan::IndexPlans,
        rapport: &mut RapportOuverture,
    ) -> Result<Self> {
        let mut index = Self::default();
        for (id, entree) in &plans.par_id {
            let Some(n) = db.get_node(entree.noeud) else {
                continue;
            };
            let Some(v) = n.get_property(MARQUE).filter(|v| !matches!(v, Value::Null)) else {
                continue;
            };
            let marque = match v.as_str().map(|s| s.parse::<DateTime<Utc>>()) {
                Some(Ok(d)) => Marque::Lue(entree.noeud, d),
                _ => {
                    let raison = format!("plan {id} : `{MARQUE}` illisible ({v:?})");
                    rapport
                        .en_quarantaine
                        .push(format!("{raison} — le plan est servi, sa marque refusée"));
                    Marque::Illisible(entree.noeud, raison)
                }
            };
            index.marques.insert(*id, marque);
        }
        index.runs = indexer(
            charger(db, RUN, rapport, &mut index.reserves, &lire_run),
            RUN,
            rapport,
            |r| r.run_id,
        );
        index.declencheurs = indexer(
            charger(
                db,
                DECLENCHEUR,
                rapport,
                &mut index.reserves,
                &lire_declencheur,
            ),
            DECLENCHEUR,
            rapport,
            |t| t.id,
        );
        index.tirs = indexer(
            charger(db, TIR, rapport, &mut index.reserves, &lire_tir),
            TIR,
            rapport,
            |f| f.id,
        );
        for f in charger(db, SIGNAL, rapport, &mut index.reserves, &lire_signal) {
            let cle = (f.valeur.trigger_id, f.valeur.cle.clone());
            if index.signaux.contains_key(&cle) {
                rapport.en_quarantaine.push(format!(
                    "nœud {SIGNAL} {:?} : signal ({}, {}) déjà pris",
                    f.noeud, cle.0, cle.1
                ));
                continue;
            }
            index.signaux.insert(cle, f);
        }
        index.evenements = indexer(
            charger(db, EVENEMENT, rapport, &mut index.reserves, &lire_evenement),
            EVENEMENT,
            rapport,
            |e| e.id,
        );
        for f in charger(db, LIVRAISON, rapport, &mut index.reserves, &lire_livraison) {
            let cle = (f.valeur.trigger_id, f.valeur.cle);
            if index.livraisons.contains_key(&cle) {
                rapport.en_quarantaine.push(format!(
                    "nœud {LIVRAISON} {:?} : livraison ({}, {}) déjà réservée",
                    f.noeud, cle.0, cle.1
                ));
                continue;
            }
            index.livraisons.insert(cle, f.noeud);
        }

        // The repairs of a crash between the two writes of a firing or a reservation.
        let mut tirs_par_declencheur: HashMap<Uuid, u64> = HashMap::new();
        for f in index.tirs.values() {
            *tirs_par_declencheur.entry(f.valeur.trigger_id).or_default() += 1;
        }
        let mut dernier_signal: HashMap<Uuid, DateTime<Utc>> = HashMap::new();
        for s in index.signaux.values() {
            let d = dernier_signal
                .entry(s.valeur.trigger_id)
                .or_insert(s.valeur.reserve_a);
            *d = (*d).max(s.valeur.reserve_a);
        }
        let mut repares = 0;
        for (id, f) in index.declencheurs.iter_mut() {
            let tirs = tirs_par_declencheur.get(id).copied().unwrap_or(0);
            if f.valeur.fire_count < tirs {
                db.set_node_property(f.noeud, "fire_count", Value::from(i64::try_from(tirs)?));
                f.valeur.fire_count = tirs;
                repares += 1;
            }
            if let Some(&d) = dernier_signal.get(id) {
                if f.valeur.last_fired.is_none_or(|l| l < d) {
                    db.set_node_property(f.noeud, "last_fired", instant(d));
                    f.valeur.last_fired = Some(d);
                    repares += 1;
                }
            }
        }
        if repares > 0 {
            tracing::warn!(
                repares,
                "base obrain : compteurs de déclencheurs réparés à l'ouverture (écriture \
                 interrompue entre un tir ou une réservation et son déclencheur)"
            );
            db.flush_wal()
                .map_err(|e| anyhow!("réparations des déclencheurs non rendues durables : {e}"))?;
        }
        Ok(index)
    }
}

/// The mark of a plan as the methods read it: `None` when it has none (or is
/// not the node the mark was read on), an error when it is unreadable.
fn marque_du_plan(etat: &super::Etat, plan: Uuid) -> Result<Option<DateTime<Utc>>> {
    let Some(entree) = etat.plans.par_id.get(&plan) else {
        return Ok(None);
    };
    match etat.executions.marques.get(&plan) {
        Some(Marque::Lue(n, d)) if *n == entree.noeud => Ok(Some(*d)),
        Some(Marque::Illisible(n, raison)) if *n == entree.noeud => Err(anyhow!(
            "{raison} : la marque tierce de ce plan ne peut pas être lue ; refusé plutôt que \
             lu comme absente"
        )),
        _ => Ok(None),
    }
}

#[allow(unused_variables)]
#[async_trait]
impl PlanRunStore for ObrainGraphStore {
    async fn create_plan_run(&self, state: &RunnerState) -> Result<()> {
        let s = state.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let proprietes = proprietes_creation_run(&s)?;
                {
                    let etat = w.index()?;
                    if !etat.plans.par_id.contains_key(&s.plan_id) {
                        bail!(
                            "le plan {} n'existe pas : l'exécution {} ne peut pas s'y rattacher \
                             (Neo4j n'écrirait rien, sans le dire)",
                            s.plan_id,
                            s.run_id
                        );
                    }
                    etat.executions.garde_reserve(
                        RUN,
                        &[&s.run_id.to_string()],
                        &format!("l'id d'exécution {}", s.run_id),
                    )?;
                    if etat.executions.runs.contains_key(&s.run_id) {
                        bail!("une exécution de plan d'id {} existe déjà", s.run_id);
                    }
                }
                let fiche = creer(w, RUN, proprietes, &lire_run)?;
                w.index_mut()?.executions.runs.insert(s.run_id, fiche);
                Ok(())
            },
        )
        .await
    }

    async fn update_plan_run(&self, state: &RunnerState) -> Result<()> {
        let s = state.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = w.index()?.executions.runs.get(&s.run_id).map(|f| f.noeud) else {
                    bail!("PlanRun not found: {}", s.run_id);
                };
                let mut maj: Vec<(&'static str, Option<Value>)> = proprietes_communes_run(&s)?
                    .into_iter()
                    .map(|(k, v)| (k, Some(v)))
                    .collect();
                maj.push((
                    "current_task_id",
                    s.current_task_id.map(|t| texte(t.to_string())),
                ));
                if let Some(c) = s.completed_at {
                    maj.push(("completed_at", Some(instant(c))));
                }
                let fiche = reecrire(w, noeud, maj, &lire_run)?;
                w.index_mut()?.executions.runs.insert(s.run_id, fiche);
                Ok(())
            },
        )
        .await
    }

    async fn get_plan_run(&self, run_id: Uuid) -> Result<Option<RunnerState>> {
        self.lire(|_, etat| Ok(etat.executions.runs.get(&run_id).map(|f| f.valeur.clone())))
    }

    async fn list_active_plan_runs(&self) -> Result<Vec<RunnerState>> {
        self.lire(|_, etat| {
            let mut runs: Vec<&RunnerState> = etat
                .executions
                .runs
                .values()
                .map(|f| &f.valeur)
                .filter(|r| r.status == PlanRunStatus::Running)
                .collect();
            runs.sort_by(|a, b| ordre_runs(a, b));
            Ok(runs.into_iter().cloned().collect())
        })
    }

    async fn list_all_plan_runs(
        &self,
        limit: i64,
        offset: i64,
        status: Option<&str>,
        workspace_slug: Option<&str>,
    ) -> Result<Vec<RunnerState>> {
        let (limit, offset) = (compte(limit, "limit")?, compte(offset, "offset")?);
        self.lire(|_, etat| {
            let projets: Option<HashSet<Uuid>> =
                workspace_slug.map(|slug| etat.espaces.projets_du_slug(slug));
            let mut runs: Vec<&RunnerState> = etat
                .executions
                .runs
                .values()
                .map(|f| &f.valeur)
                .filter(|r| status.is_none_or(|s| r.status.to_string() == s))
                .filter(|r| {
                    projets.as_ref().is_none_or(|ps| {
                        etat.plans
                            .par_id
                            .get(&r.plan_id)
                            .and_then(|p| p.plan.project_id)
                            .is_some_and(|p| ps.contains(&p))
                    })
                })
                .collect();
            runs.sort_by(|a, b| ordre_runs(a, b));
            Ok(runs.into_iter().skip(offset).take(limit).cloned().collect())
        })
    }

    async fn list_plan_runs(&self, plan_id: Uuid, limit: i64) -> Result<Vec<RunnerState>> {
        let limit = compte(limit, "limit")?;
        self.lire(|_, etat| {
            let mut runs: Vec<&RunnerState> = etat
                .executions
                .runs
                .values()
                .map(|f| &f.valeur)
                .filter(|r| r.plan_id == plan_id)
                .collect();
            runs.sort_by(|a, b| ordre_runs(a, b));
            Ok(runs.into_iter().take(limit).cloned().collect())
        })
    }

    async fn create_trigger(&self, trigger: &Trigger) -> Result<Trigger> {
        let t = trigger.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let marque = {
                    let etat = w.index()?;
                    if !etat.plans.par_id.contains_key(&t.plan_id) {
                        bail!(
                            "le plan {} n'existe pas : le déclencheur {} ne peut pas s'y \
                             rattacher (Neo4j n'écrirait rien et le rendrait comme écrit)",
                            t.plan_id,
                            t.id
                        );
                    }
                    etat.executions.garde_reserve(
                        DECLENCHEUR,
                        &[&t.id.to_string()],
                        &format!("l'id de déclencheur {}", t.id),
                    )?;
                    if etat.executions.declencheurs.contains_key(&t.id) {
                        bail!("un déclencheur de plan d'id {} existe déjà", t.id);
                    }
                    // The plan as approved: its mark, read in this write.
                    marque_du_plan(&etat, t.plan_id)?
                };
                let fiche = creer(
                    w,
                    DECLENCHEUR,
                    proprietes_declencheur(&t, marque)?,
                    &lire_declencheur,
                )?;
                let stocke = fiche.valeur.clone();
                w.index_mut()?.executions.declencheurs.insert(t.id, fiche);
                Ok(stocke)
            },
        )
        .await
    }

    async fn get_trigger(&self, trigger_id: Uuid) -> Result<Option<Trigger>> {
        self.lire(|_, etat| {
            Ok(etat
                .executions
                .declencheurs
                .get(&trigger_id)
                .map(|f| f.valeur.clone()))
        })
    }

    async fn list_triggers(&self, plan_id: Uuid) -> Result<Vec<Trigger>> {
        self.lire(|_, etat| {
            let mut ts: Vec<&Trigger> = etat
                .executions
                .declencheurs
                .values()
                .map(|f| &f.valeur)
                .filter(|t| t.plan_id == plan_id)
                .collect();
            // ORDER BY created_at DESC, id DESC.
            ts.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(b.id.cmp(&a.id)));
            Ok(ts.into_iter().cloned().collect())
        })
    }

    async fn disable_trigger(
        &self,
        trigger_id: Uuid,
        reason: Option<&str>,
    ) -> Result<Option<Trigger>> {
        let raison = reason.unwrap_or_default().to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = w
                    .index()?
                    .executions
                    .declencheurs
                    .get(&trigger_id)
                    .map(|f| f.noeud)
                else {
                    return Ok(None);
                };
                let fiche = reecrire(
                    w,
                    noeud,
                    vec![
                        ("enabled", Some(Value::from(false))),
                        ("disabled_reason", Some(texte(raison))),
                    ],
                    &lire_declencheur,
                )?;
                let lu = fiche.valeur.clone();
                w.index_mut()?
                    .executions
                    .declencheurs
                    .insert(trigger_id, fiche);
                Ok(Some(lu))
            },
        )
        .await
    }

    async fn mark_third_party_write(&self, content: PlanContent) -> Result<Option<Uuid>> {
        let plan_id = match content {
            PlanContent::Plan(id) => id,
            autre => {
                return refus(
                    "PlanRunStore::mark_third_party_write",
                    &format!(
                        "{autre:?} : trouver le plan de ce contenu passe par les tâches, étapes, \
                         contraintes ou décisions, domaines non portés (seul PlanContent::Plan \
                         est servi)"
                    ),
                );
            }
        };
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = w.index()?.plans.par_id.get(&plan_id).map(|e| e.noeud) else {
                    return Ok(None);
                };
                let maintenant = Utc::now();
                poser(w, noeud, vec![(MARQUE, Some(instant(maintenant)))])?;
                w.rendre_durable()?;
                w.index_mut()?
                    .executions
                    .marques
                    .insert(plan_id, Marque::Lue(noeud, maintenant));
                Ok(Some(plan_id))
            },
        )
        .await
    }

    async fn plan_third_party_written_at(&self, plan_id: Uuid) -> Result<Option<DateTime<Utc>>> {
        self.lire(|_, etat| marque_du_plan(etat, plan_id))
    }

    async fn delete_trigger(&self, trigger_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, tirs, signaux) = {
                    let etat = w.index()?;
                    let ex = &etat.executions;
                    let tirs: Vec<(Uuid, NodeId)> = ex
                        .tirs
                        .iter()
                        .filter(|(_, f)| f.valeur.trigger_id == trigger_id)
                        .map(|(id, f)| (*id, f.noeud))
                        .collect();
                    let signaux: Vec<((Uuid, String), NodeId)> = ex
                        .signaux
                        .iter()
                        .filter(|((t, _), _)| *t == trigger_id)
                        .map(|(k, f)| (k.clone(), f.noeud))
                        .collect();
                    (
                        ex.declencheurs.get(&trigger_id).map(|f| f.noeud),
                        tirs,
                        signaux,
                    )
                };
                if noeud.is_none() && tirs.is_empty() && signaux.is_empty() {
                    return Ok(());
                }
                // The firings and signals first, the trigger last: a crash half-way
                // leaves a trigger with less history, never history without its trigger.
                for (_, n) in &tirs {
                    supprimer(w, *n)?;
                }
                for (_, n) in &signaux {
                    supprimer(w, *n)?;
                }
                if let Some(n) = noeud {
                    supprimer(w, n)?;
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for (id, _) in &tirs {
                    etat.executions.tirs.remove(id);
                }
                for (k, _) in &signaux {
                    etat.executions.signaux.remove(k);
                }
                etat.executions.declencheurs.remove(&trigger_id);
                Ok(())
            },
        )
        .await
    }

    async fn record_trigger_firing(&self, firing: &TriggerFiring) -> Result<()> {
        let f = firing.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let proprietes = proprietes_tir(&f)?;
                let (noeud_declencheur, compte_avant) = {
                    let etat = w.index()?;
                    let Some(d) = etat.executions.declencheurs.get(&f.trigger_id) else {
                        bail!(
                            "le déclencheur {} n'existe pas : le tir {} ne peut pas être \
                             enregistré (Neo4j n'écrirait rien, sans le dire)",
                            f.trigger_id,
                            f.id
                        );
                    };
                    etat.executions.garde_reserve(
                        TIR,
                        &[&f.id.to_string()],
                        &format!("l'id de tir {}", f.id),
                    )?;
                    if etat.executions.tirs.contains_key(&f.id) {
                        bail!("un tir d'id {} existe déjà", f.id);
                    }
                    (d.noeud, d.valeur.fire_count)
                };
                // The firing committed first, the count after (module doc).
                let noeud_tir = creer_noeud(w, TIR, proprietes)?;
                let apres = compte_avant.saturating_add(1);
                poser(
                    w,
                    noeud_declencheur,
                    vec![("fire_count", Some(Value::from(i64::try_from(apres)?)))],
                )?;
                let tir = w.apres_ecriture(relire(w.db, noeud_tir, &lire_tir))?;
                let declencheur =
                    w.apres_ecriture(relire(w.db, noeud_declencheur, &lire_declencheur))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.executions.tirs.insert(f.id, tir);
                etat.executions
                    .declencheurs
                    .insert(f.trigger_id, declencheur);
                Ok(())
            },
        )
        .await
    }

    async fn enable_trigger_as(
        &self,
        trigger_id: Uuid,
        author: &TriggerAuthor,
    ) -> Result<Option<Trigger>> {
        let a = author.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, marque) = {
                    let etat = w.index()?;
                    let Some(f) = etat.executions.declencheurs.get(&trigger_id) else {
                        return Ok(None);
                    };
                    (f.noeud, marque_du_plan(&etat, f.valeur.plan_id)?)
                };
                let fiche = reecrire(
                    w,
                    noeud,
                    vec![
                        ("enabled", Some(Value::from(true))),
                        ("author", Some(auteur(Some(&a))?)),
                        // The plan's mark now (none, or no plan: the property goes).
                        ("approved_mark", marque.map(instant)),
                        ("disabled_reason", None),
                    ],
                    &lire_declencheur,
                )?;
                let lu = fiche.valeur.clone();
                w.index_mut()?
                    .executions
                    .declencheurs
                    .insert(trigger_id, fiche);
                Ok(Some(lu))
            },
        )
        .await
    }

    async fn reserve_trigger_signal(
        &self,
        trigger_id: Uuid,
        key: &str,
        cooldown_secs: u64,
    ) -> Result<SignalReservation> {
        let cle = key.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let maintenant = Utc::now();
                let limite = maintenant - chrono::Duration::hours(RETENTION_SIGNAL_HEURES);
                let (decision, declencheur, ancien, perimes) = {
                    let etat = w.index()?;
                    let ex = &etat.executions;
                    // A signal on disk but unreadable may have started a run: never
                    // `Reserved` again.
                    ex.garde_reserve(
                        SIGNAL,
                        &[&trigger_id.to_string(), &cle],
                        &format!("le signal « {cle} » du déclencheur {trigger_id}"),
                    )?;
                    let perimes: Vec<((Uuid, String), NodeId)> = ex
                        .signaux
                        .iter()
                        .filter(|(_, s)| s.valeur.reserve_a < limite)
                        .take(PURGE_SIGNAUX)
                        .map(|(k, s)| (k.clone(), s.noeud))
                        .collect();
                    match ex.declencheurs.get(&trigger_id) {
                        // No such trigger: `Duplicate`, as Neo4j's `MATCH` returns no row.
                        None => (SignalReservation::Duplicate, None, None, perimes),
                        Some(d) => {
                            let vu = ex.signaux.get(&(trigger_id, cle.clone()));
                            let frais = vu.is_none_or(|s| s.valeur.reserve_a < limite);
                            let refroidi = cooldown_secs == 0
                                || d.valeur.last_fired.is_none_or(|l| {
                                    let ecoule = (maintenant - l).num_seconds();
                                    i64::try_from(cooldown_secs).is_ok_and(|c| ecoule >= c)
                                });
                            let decision = match (frais, refroidi) {
                                (false, _) => SignalReservation::Duplicate,
                                (true, false) => SignalReservation::Cooldown,
                                (true, true) => SignalReservation::Reserved,
                            };
                            (decision, Some(d.noeud), vu.map(|s| s.noeud), perimes)
                        }
                    }
                };
                let mut nouveau = None;
                let mut declencheur_relu = None;
                if decision == SignalReservation::Reserved {
                    let declencheur = declencheur
                        .ok_or_else(|| anyhow!("réservation sans déclencheur ({trigger_id})"))?;
                    // An expired signal of the same key gives way to the new one.
                    if let Some(n) = ancien {
                        supprimer(w, n)?;
                    }
                    let signal = Signal {
                        trigger_id,
                        cle: cle.clone(),
                        reserve_a: maintenant,
                    };
                    // The signal committed first, `last_fired` after (module doc).
                    let n = creer_noeud(w, SIGNAL, proprietes_signal(&signal))?;
                    poser(
                        w,
                        declencheur,
                        vec![("last_fired", Some(instant(maintenant)))],
                    )?;
                    nouveau = Some(Fiche {
                        noeud: n,
                        valeur: signal,
                    });
                    declencheur_relu =
                        Some(w.apres_ecriture(relire(w.db, declencheur, &lire_declencheur))?);
                }
                // The purge of expired signals, after every reservation (as Neo4j).
                let perimes: Vec<_> = perimes
                    .into_iter()
                    .filter(|(k, _)| {
                        !(decision == SignalReservation::Reserved
                            && *k == (trigger_id, cle.clone()))
                    })
                    .collect();
                for (_, n) in &perimes {
                    supprimer(w, *n)?;
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for (k, _) in &perimes {
                    etat.executions.signaux.remove(k);
                }
                if let Some(s) = nouveau {
                    etat.executions.signaux.insert((trigger_id, cle), s);
                }
                if let Some(d) = declencheur_relu {
                    etat.executions.declencheurs.insert(trigger_id, d);
                }
                Ok(decision)
            },
        )
        .await
    }

    async fn list_all_triggers(&self, trigger_type: Option<&str>) -> Result<Vec<Trigger>> {
        self.lire(|_, etat| {
            let mut ts: Vec<&Trigger> = etat
                .executions
                .declencheurs
                .values()
                .map(|f| &f.valeur)
                .filter(|t| trigger_type.is_none_or(|ty| nom_type(&t.trigger_type) == ty))
                .collect();
            ts.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(b.id.cmp(&a.id)));
            Ok(ts.into_iter().cloned().collect())
        })
    }

    async fn list_trigger_firings(
        &self,
        trigger_id: Uuid,
        limit: i64,
    ) -> Result<Vec<TriggerFiring>> {
        let limit = compte(limit, "limit")?;
        self.lire(|_, etat| {
            let mut tirs: Vec<&TriggerFiring> = etat
                .executions
                .tirs
                .values()
                .map(|f| &f.valeur)
                .filter(|f| f.trigger_id == trigger_id)
                .collect();
            // ORDER BY fired_at DESC, id DESC.
            tirs.sort_by(|a, b| b.fired_at.cmp(&a.fired_at).then(b.id.cmp(&a.id)));
            Ok(tirs.into_iter().take(limit).cloned().collect())
        })
    }

    async fn record_external_event(&self, event: &ExternalEvent) -> Result<()> {
        let e = event.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                w.index()?.executions.garde_reserve(
                    EVENEMENT,
                    &[&e.id.to_string()],
                    &format!("l'id d'événement externe {}", e.id),
                )?;
                if w.index()?.executions.evenements.contains_key(&e.id) {
                    bail!("un événement externe d'id {} existe déjà", e.id);
                }
                let fiche = creer(w, EVENEMENT, proprietes_evenement(&e)?, &lire_evenement)?;
                w.index_mut()?.executions.evenements.insert(e.id, fiche);
                Ok(())
            },
        )
        .await
    }

    async fn list_external_events(
        &self,
        trigger_id: Uuid,
        limit: i64,
    ) -> Result<Vec<ExternalEvent>> {
        let limit = compte(limit, "limit")?;
        self.lire(|_, etat| {
            let mut evs: Vec<&ExternalEvent> = etat
                .executions
                .evenements
                .values()
                .map(|f| &f.valeur)
                .filter(|e| e.trigger_id == trigger_id)
                .collect();
            // ORDER BY received_at DESC, id DESC.
            evs.sort_by(|a, b| b.received_at.cmp(&a.received_at).then(b.id.cmp(&a.id)));
            Ok(evs.into_iter().take(limit).cloned().collect())
        })
    }

    async fn claim_external_delivery(&self, trigger_id: Uuid, delivery_key: &str) -> Result<bool> {
        let cle = delivery_key.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // A delivery claimed on disk but unreadable: never claimed again.
                w.index()?.executions.garde_reserve(
                    LIVRAISON,
                    &[&trigger_id.to_string(), &cle],
                    &format!("la livraison « {cle} » du déclencheur {trigger_id}"),
                )?;
                if w.index()?
                    .executions
                    .livraisons
                    .contains_key(&(trigger_id, cle.clone()))
                {
                    return Ok(false);
                }
                let fiche = creer(
                    w,
                    LIVRAISON,
                    vec![
                        ("trigger_id", texte(trigger_id.to_string())),
                        ("delivery_key", texte(cle.clone())),
                        ("claimed_at", instant(Utc::now())),
                    ],
                    &lire_livraison,
                )?;
                w.index_mut()?
                    .executions
                    .livraisons
                    .insert((trigger_id, cle), fiche.noeud);
                Ok(true)
            },
        )
        .await
    }

    async fn release_external_delivery(
        &self,
        trigger_id: Uuid,
        delivery_key: &str,
    ) -> Result<bool> {
        let cle = (trigger_id, delivery_key.to_string());
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                w.index()?.executions.garde_reserve(
                    LIVRAISON,
                    &[&cle.0.to_string(), &cle.1],
                    &format!("la livraison « {} » du déclencheur {}", cle.1, cle.0),
                )?;
                let Some(noeud) = w.index()?.executions.livraisons.get(&cle).copied() else {
                    return Ok(false);
                };
                supprimer(w, noeud)?;
                w.rendre_durable()?;
                w.index_mut()?.executions.livraisons.remove(&cle);
                Ok(true)
            },
        )
        .await
    }

    async fn purge_external_events_before(&self, before: DateTime<Utc>) -> Result<u64> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let vieux: Vec<(Uuid, NodeId)> = w
                    .index()?
                    .executions
                    .evenements
                    .iter()
                    .filter(|(_, f)| f.valeur.received_at < before)
                    .map(|(id, f)| (*id, f.noeud))
                    .collect();
                if vieux.is_empty() {
                    return Ok(0);
                }
                for (_, n) in &vieux {
                    supprimer(w, *n)?;
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for (id, _) in &vieux {
                    etat.executions.evenements.remove(id);
                }
                Ok(vieux.len() as u64)
            },
        )
        .await
    }

    async fn create_agent_execution(&self, ae: &AgentExecutionNode) -> Result<()> {
        refus("PlanRunStore::create_agent_execution", CAUSE_TACHE)
    }

    async fn update_agent_execution(&self, ae: &AgentExecutionNode) -> Result<()> {
        refus("PlanRunStore::update_agent_execution", CAUSE_TACHE)
    }

    async fn get_agent_executions_for_run(&self, run_id: Uuid) -> Result<Vec<AgentExecutionNode>> {
        refus("PlanRunStore::get_agent_executions_for_run", CAUSE_TACHE)
    }

    async fn record_agent_execution_verification(
        &self,
        agent_execution_id: Uuid,
        verification_json: &str,
    ) -> Result<()> {
        refus(
            "PlanRunStore::record_agent_execution_verification",
            CAUSE_TACHE,
        )
    }

    async fn list_running_agent_executions(&self) -> Result<Vec<AgentExecutionNode>> {
        refus("PlanRunStore::list_running_agent_executions", CAUSE_TACHE)
    }

    async fn agent_execution_cost_breakdown(
        &self,
        group_by: CostGroupBy,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<CostBreakdownRow>> {
        refus("PlanRunStore::agent_execution_cost_breakdown", CAUSE_TACHE)
    }

    async fn get_all_pagerank_values(&self, project_id: Uuid) -> Result<Vec<f64>> {
        refus("PlanRunStore::get_all_pagerank_values", CAUSE_CODE)
    }

    async fn get_community_risk_vectors(&self, project_id: Uuid) -> Result<Vec<Vec<f64>>> {
        refus("PlanRunStore::get_community_risk_vectors", CAUSE_CODE)
    }

    async fn get_all_risk_score_values(&self, project_id: Uuid) -> Result<Vec<f64>> {
        refus("PlanRunStore::get_all_risk_score_values", CAUSE_CODE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::models::{PlanNode, PlanStatus};
    use crate::neo4j::traits::PlanStore;
    use crate::obrain::{Creation, Interruption};
    use chrono::TimeZone;
    use futures::future::join_all;

    fn a(s: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1_700_000_000 + s, 0).unwrap()
    }

    async fn plan(store: &ObrainGraphStore) -> PlanNode {
        let p = PlanNode {
            id: Uuid::new_v4(),
            title: "plan".into(),
            description: "d".into(),
            status: PlanStatus::Draft,
            created_at: a(0),
            created_by: "t".into(),
            priority: 0,
            project_id: None,
            execution_context: None,
            persona: None,
        };
        store.create_plan(&p).await.unwrap();
        p
    }

    fn run(plan: Uuid, s: i64) -> RunnerState {
        let mut r = RunnerState::new(Uuid::new_v4(), plan, 3, TriggerSource::Manual);
        r.started_at = a(s);
        r
    }

    fn declencheur(plan: Uuid, ty: TriggerType, s: i64) -> Trigger {
        Trigger {
            id: Uuid::new_v4(),
            plan_id: plan,
            trigger_type: ty,
            config: serde_json::json!({"cron": "0 2 * * *"}),
            enabled: true,
            cooldown_secs: 0,
            last_fired: None,
            fire_count: 41,
            created_at: a(s),
            author: None,
            disabled_reason: None,
            approved_mark: Some(a(999)),
        }
    }

    fn tir(trigger: Uuid, s: i64) -> TriggerFiring {
        TriggerFiring {
            id: Uuid::new_v4(),
            trigger_id: trigger,
            plan_run_id: None,
            fired_at: a(s),
            source_payload: Some(serde_json::json!({"n": s})),
            start_error: None,
        }
    }

    fn auteur_test() -> TriggerAuthor {
        TriggerAuthor {
            sub: "u".into(),
            email: "u@x".into(),
            name: "U".into(),
            token_type: None,
            scope: None,
            session_id: None,
            third_party: false,
            ceiling: None,
            tool_profile: None,
            recorded_at: a(5),
        }
    }

    #[tokio::test]
    async fn un_run_avance_puis_survit_a_la_reouverture() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let tache = Uuid::new_v4();
        let mut r;
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            let p = plan(&store).await;
            r = run(p.id, 0);
            r.cwd = Some("/tmp/x".into());
            store.create_plan_run(&r).await.unwrap();
            r.current_task_id = Some(tache);
            r.completed_tasks = vec![tache];
            r.retry_counts.insert(tache, 2);
            r.status = PlanRunStatus::Completed;
            r.completed_at = Some(a(9));
            store.update_plan_run(&r).await.unwrap();
            assert!(store.list_active_plan_runs().await.unwrap().is_empty());
            // A run of a missing plan, a missing run, a taken id: refused.
            assert!(store
                .create_plan_run(&run(Uuid::new_v4(), 1))
                .await
                .is_err());
            assert!(store.update_plan_run(&run(p.id, 1)).await.is_err());
            assert!(store.create_plan_run(&r).await.is_err());
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(store.rapport_d_ouverture(), &RapportOuverture::default());
        let lu = store.get_plan_run(r.run_id).await.unwrap().unwrap();
        assert_eq!(lu.status, PlanRunStatus::Completed);
        assert_eq!(lu.completed_tasks, vec![tache]);
        assert_eq!(lu.retry_counts.get(&tache), Some(&2));
        assert_eq!(lu.cwd.as_deref(), Some("/tmp/x"));
        assert_eq!(lu.completed_at, Some(a(9)));
        assert_eq!(store.list_plan_runs(r.plan_id, 10).await.unwrap().len(), 1);
        assert_eq!(
            store
                .list_all_plan_runs(10, 0, Some("completed"), None)
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(store
            .list_all_plan_runs(10, 0, Some("running"), None)
            .await
            .unwrap()
            .is_empty());
        assert!(store.list_all_plan_runs(-1, 0, None, None).await.is_err());
    }

    #[tokio::test]
    async fn huit_creations_concurrentes_d_un_meme_run_une_seule_gagne() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let p = plan(&store).await;
        let r = run(p.id, 0);
        let issues = join_all((0..8).map(|_| store.create_plan_run(&r))).await;
        assert_eq!(issues.iter().filter(|i| i.is_ok()).count(), 1, "{issues:?}");
        assert_eq!(store.list_plan_runs(p.id, 10).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn declencheur_marque_tirs_et_cascade() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let p = plan(&store).await;
        let t = declencheur(p.id, TriggerType::Schedule, 0);
        let stocke = store.create_trigger(&t).await.unwrap();
        assert_eq!(
            stocke.fire_count, 0,
            "the store writes 0, whatever is given"
        );
        assert_eq!(
            stocke.approved_mark, None,
            "no mark yet: the given one is ignored"
        );
        // A trigger of a missing plan is refused.
        assert!(store
            .create_trigger(&declencheur(Uuid::new_v4(), TriggerType::Event, 0))
            .await
            .is_err());
        assert_eq!(
            store
                .mark_third_party_write(PlanContent::Plan(p.id))
                .await
                .unwrap(),
            Some(p.id)
        );
        assert_eq!(
            store
                .mark_third_party_write(PlanContent::Plan(Uuid::new_v4()))
                .await
                .unwrap(),
            None
        );
        let marque = store
            .plan_third_party_written_at(p.id)
            .await
            .unwrap()
            .unwrap();
        let e = store
            .mark_third_party_write(PlanContent::Task(Uuid::new_v4()))
            .await
            .unwrap_err();
        assert_eq!(
            e.downcast_ref::<NonPorte>(),
            Some(&NonPorte("PlanRunStore::mark_third_party_write"))
        );
        let off = store
            .disable_trigger(t.id, Some("no_author"))
            .await
            .unwrap()
            .unwrap();
        assert!(!off.enabled);
        assert_eq!(off.disabled_reason.as_deref(), Some("no_author"));
        let on = store
            .enable_trigger_as(t.id, &auteur_test())
            .await
            .unwrap()
            .unwrap();
        assert!(on.enabled);
        assert_eq!(on.disabled_reason, None);
        assert_eq!(on.author, Some(auteur_test()));
        assert_eq!(on.approved_mark, Some(marque), "the plan as approved");
        assert!(store
            .enable_trigger_as(Uuid::new_v4(), &auteur_test())
            .await
            .unwrap()
            .is_none());
        for s in [3, 1, 2] {
            store.record_trigger_firing(&tir(t.id, s)).await.unwrap();
        }
        assert!(store
            .record_trigger_firing(&tir(Uuid::new_v4(), 0))
            .await
            .is_err());
        let tirs = store.list_trigger_firings(t.id, 2).await.unwrap();
        assert_eq!(
            tirs.iter().map(|f| f.fired_at).collect::<Vec<_>>(),
            vec![a(3), a(2)]
        );
        assert_eq!(
            store.get_trigger(t.id).await.unwrap().unwrap().fire_count,
            3
        );
        assert_eq!(
            store
                .list_all_triggers(Some("schedule"))
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(store
            .list_all_triggers(Some("webhook"))
            .await
            .unwrap()
            .is_empty());
        assert_eq!(
            store.reserve_trigger_signal(t.id, "k", 0).await.unwrap(),
            SignalReservation::Reserved
        );
        store.delete_trigger(t.id).await.unwrap();
        assert!(store.get_trigger(t.id).await.unwrap().is_none());
        assert!(store
            .list_trigger_firings(t.id, 10)
            .await
            .unwrap()
            .is_empty());
        // Only the plan is left.
        assert_eq!(store.interne.db.store().node_count(), 1);
    }

    #[tokio::test]
    async fn reservations_concurrentes_une_seule_gagne_et_le_delai_tient() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let p = plan(&store).await;
        let t = declencheur(p.id, TriggerType::Webhook, 0);
        store.create_trigger(&t).await.unwrap();
        let issues = join_all((0..8).map(|_| store.reserve_trigger_signal(t.id, "même", 0))).await;
        let gagnes = issues
            .iter()
            .filter(|i| matches!(i, Ok(SignalReservation::Reserved)))
            .count();
        assert_eq!(gagnes, 1, "{issues:?}");
        assert!(issues.iter().all(|i| matches!(
            i,
            Ok(SignalReservation::Reserved | SignalReservation::Duplicate)
        )));
        // The salvo above moved `last_fired`: under a one-hour cooldown every new key
        // waits (deterministic, no race here; the race is the test below).
        let cles: Vec<String> = (0..8).map(|i| format!("k{i}")).collect();
        let issues = join_all(
            cles.iter()
                .map(|k| store.reserve_trigger_signal(t.id, k, 3600)),
        )
        .await;
        assert!(
            issues
                .iter()
                .all(|i| matches!(i, Ok(SignalReservation::Cooldown))),
            "the first reservation started the cooldown: {issues:?}"
        );
        // On a FRESH trigger, eight distinct keys at once under a one-hour cooldown:
        // exactly one fires, seven wait.
        let neuf = declencheur(p.id, TriggerType::Webhook, 1);
        store.create_trigger(&neuf).await.unwrap();
        let issues = join_all(
            cles.iter()
                .map(|k| store.reserve_trigger_signal(neuf.id, k, 3600)),
        )
        .await;
        let compte = |r: SignalReservation| {
            issues
                .iter()
                .filter(|i| matches!(i, Ok(x) if *x == r))
                .count()
        };
        assert_eq!(
            (
                compte(SignalReservation::Reserved),
                compte(SignalReservation::Cooldown)
            ),
            (1, 7),
            "{issues:?}"
        );
        assert_eq!(
            store
                .reserve_trigger_signal(Uuid::new_v4(), "x", 0)
                .await
                .unwrap(),
            SignalReservation::Duplicate,
            "no such trigger"
        );
        assert!(store
            .get_trigger(t.id)
            .await
            .unwrap()
            .unwrap()
            .last_fired
            .is_some());
    }

    #[tokio::test]
    async fn une_creation_interrompue_laisse_un_brouillon_supprime_a_l_ouverture() {
        let dir = tempfile::tempdir().unwrap();
        // creer_noeud: before the node, before the marker, before each of the nine
        // properties, before the marker goes: 12 points.
        for k in 1..=12 {
            let chemin = dir.path().join(format!("creation-{k}"));
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            let p = plan(&store).await;
            let t = declencheur(p.id, TriggerType::Event, 0);
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let e = store.create_trigger(&t).await.unwrap_err();
            assert!(
                format!("{e:#}").contains("interruption injectée"),
                "{k}: {e:#}"
            );
            assert!(store.get_trigger(t.id).await.unwrap().is_none(), "{k}");
            store.close().await.unwrap();
            drop(store);
            let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
            let rapport = store.rapport_d_ouverture();
            assert!(rapport.en_quarantaine.is_empty(), "{k}: {rapport:?}");
            let attendu = usize::from(k > 1);
            assert_eq!(rapport.brouillons_supprimes, attendu, "{k}: {rapport:?}");
            assert!(store.get_trigger(t.id).await.unwrap().is_none(), "{k}");
            // The same creation goes through once the draft is gone.
            store.create_trigger(&t).await.unwrap();
            store.close().await.unwrap();
        }
    }

    #[tokio::test]
    async fn un_tir_interrompu_avant_son_compte_est_repare_a_l_ouverture() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let t;
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            let p = plan(&store).await;
            t = declencheur(p.id, TriggerType::Event, 0);
            store.create_trigger(&t).await.unwrap();
            store.record_trigger_firing(&tir(t.id, 1)).await.unwrap();
            // The firing node has six properties: step 10 is the count, after the
            // firing's commit point.
            store.interrompre_au_prochain(Interruption::AvantEtape(10));
            assert!(store.record_trigger_firing(&tir(t.id, 2)).await.is_err());
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(store.rapport_d_ouverture(), &RapportOuverture::default());
        assert_eq!(store.list_trigger_firings(t.id, 10).await.unwrap().len(), 2);
        assert_eq!(
            store.get_trigger(t.id).await.unwrap().unwrap().fire_count,
            2,
            "the count is raised to the committed firings"
        );
    }

    #[tokio::test]
    async fn un_declencheur_illisible_est_en_quarantaine() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let (bon, mauvais);
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            let p = plan(&store).await;
            bon = declencheur(p.id, TriggerType::Chat, 0);
            store.create_trigger(&bon).await.unwrap();
            mauvais = declencheur(p.id, TriggerType::Chat, 1);
            let db = &store.interne.db;
            let n = db.create_node(&[DECLENCHEUR]);
            for (k, v) in proprietes_declencheur(&mauvais, None).unwrap() {
                db.set_node_property(n, k, v);
            }
            // An author Neo4j would read as `None`: the run would start as nobody.
            db.set_node_property(n, "author", Value::from("{pas du json"));
            db.commit_wal().unwrap();
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        let rapport = store.rapport_d_ouverture();
        assert_eq!(rapport.en_quarantaine.len(), 1, "{rapport:?}");
        assert!(rapport.en_quarantaine[0].contains("author"), "{rapport:?}");
        assert!(store.get_trigger(mauvais.id).await.unwrap().is_none());
        assert!(store.get_trigger(bon.id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn les_refus_sont_nommes() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let id = Uuid::new_v4();
        let cas: Vec<(anyhow::Error, &str)> = vec![
            (
                store
                    .create_agent_execution(&AgentExecutionNode::new(id, id))
                    .await
                    .unwrap_err(),
                "PlanRunStore::create_agent_execution",
            ),
            (
                store
                    .update_agent_execution(&AgentExecutionNode::new(id, id))
                    .await
                    .unwrap_err(),
                "PlanRunStore::update_agent_execution",
            ),
            (
                store.get_agent_executions_for_run(id).await.unwrap_err(),
                "PlanRunStore::get_agent_executions_for_run",
            ),
            (
                store
                    .record_agent_execution_verification(id, "{}")
                    .await
                    .unwrap_err(),
                "PlanRunStore::record_agent_execution_verification",
            ),
            (
                store.list_running_agent_executions().await.unwrap_err(),
                "PlanRunStore::list_running_agent_executions",
            ),
            (
                store
                    .agent_execution_cost_breakdown(CostGroupBy::Model, a(0), a(1))
                    .await
                    .unwrap_err(),
                "PlanRunStore::agent_execution_cost_breakdown",
            ),
            (
                store.get_all_pagerank_values(id).await.unwrap_err(),
                "PlanRunStore::get_all_pagerank_values",
            ),
            (
                store.get_community_risk_vectors(id).await.unwrap_err(),
                "PlanRunStore::get_community_risk_vectors",
            ),
            (
                store.get_all_risk_score_values(id).await.unwrap_err(),
                "PlanRunStore::get_all_risk_score_values",
            ),
        ];
        assert_eq!(cas.len(), REFUSEES.len());
        for (e, nom) in cas {
            assert!(REFUSEES.contains(&nom), "{nom}");
            assert_eq!(e.downcast_ref::<NonPorte>(), Some(&NonPorte(nom)), "{nom}");
        }
    }
}
