//! `ProtocolRunStore` on obrain: runs of a protocol and their runtime states
//! (lot O-PROTOCOL). The protocols themselves are in `protocol.rs`.
//!
//! PORTED here, with the Neo4j semantics of `src/neo4j/protocol.rs` (14 of the 18
//! methods): `create_protocol_run` (with the one-running-root-run guard),
//! `get_protocol_run`, `update_protocol_run` (compare-and-swap on `version`),
//! `list_protocol_runs`, `list_all_protocol_runs`, `list_child_runs`,
//! `count_child_runs`, `get_run_tree`, `delete_protocol_run`,
//! `find_active_run_for_project`, `list_completed_runs_for_project`, and the
//! three runtime-state methods.
//!
//! REFUSED by name (`NonPorte`, see [`REFUSEES`]), never answered empty:
//! - `create_produced_during`, `get_run_outcomes`: they link and read Notes AND
//!   Decisions to a run (`PRODUCED_DURING`); the decision domain is not ported, so
//!   half of what they name does not exist here;
//! - `persist_reasoning_tree`, `get_run_reasoning_tree_id`: the persisted
//!   reasoning trees (`REASONING_FOR`) belong to a domain not ported.
//!
//! # Layout
//!
//! One node per run (`ProtocolRun`) and per runtime state (`RuntimeState`), with
//! the properties Neo4j writes. The links are PROPERTIES, as in `plan.rs`:
//! `protocol_id` (Neo4j: `INSTANCE_OF`), `parent_run_id` (`CHILD_OF`), `plan_id`
//! and `task_id` (`LINKED_TO_PLAN`, `LINKED_TO_TASK`), `run_id` and `generated_by`
//! on a runtime state (`HAS_RUNTIME_STATE`, `GENERATED_BY`).
//!
//! # Divergences
//!
//! Strict side:
//! - `create_protocol_run` of a run whose protocol does not exist is REFUSED with
//!   its own message. Neo4j's `MATCH` returns no row, and the method then reports
//!   « concurrent run already running », which is false.
//! - `create_runtime_state` of a run or a generating state that does not exist is
//!   REFUSED (Neo4j's `MATCH`es silently write nothing).
//! - A run id already taken is refused (Neo4j `CREATE`s a second node).
//! - `update_protocol_run` of a run that does not exist says « not found » (as the
//!   in-memory store does); Neo4j answers `OptimisticLockError` for both a stale
//!   version and a missing run. A stale version keeps the `OptimisticLockError`
//!   text that `protocol_handlers.rs` maps to a conflict.
//!
//! Read side:
//! - A child is attached to its parent while that parent EXISTS. Neo4j attaches it
//!   by an edge written at creation when the parent was there (best effort), which
//!   is lost when the parent is deleted: the two agree, except for a child created
//!   before its parent, which Neo4j never attaches.
//! - Dates are compared parsed, not as RFC 3339 strings.
//! - The runs are scanned in memory (no index by protocol or by parent): a run is
//!   a few hundred bytes and `list_*` pages in the process, not in the engine.
//!
//! # Atomicity
//!
//! Creation follows the draft-marker protocol of `project.rs`. The root-run
//! guard is checked under the writer lock, so of eight concurrent root creations
//! exactly one wins (Neo4j's `WHERE NOT EXISTS` reads before it locks). A deletion
//! removes the runtime states first and the run last.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::protocol::{
    a_un_blob, charger, creer, date_facultative, date_requise, enumeration, facultative, non_signe,
    reecrire, requise, supprimer, uuid_facultatif, uuid_requis, Fiche, Lecteur,
};
use super::{Ecriture, NonPorte, ObrainGraphStore, RapportOuverture};
use crate::neo4j::protocol::ProducedArtefact;
use crate::neo4j::traits::ProtocolRunStore;
use crate::protocol::{ProtocolRun, RunStatus, RuntimeState, StateVisit};

const RUN: &str = "ProtocolRun";
const ETAT_RUNTIME: &str = "RuntimeState";
/// The depth of `get_run_tree` below its root (`CHILD_OF*1..5`).
const PROFONDEUR_ARBRE: usize = 5;

/// The methods of `ProtocolRunStore` refused on every call, as
/// `ProtocolRunStore::<method>`: they read or link what other domains own.
pub const REFUSEES: &[&str] = &[
    "ProtocolRunStore::create_produced_during",
    "ProtocolRunStore::get_run_outcomes",
    "ProtocolRunStore::persist_reasoning_tree",
    "ProtocolRunStore::get_run_reasoning_tree_id",
];

/// Refuses a method that obrain does not port, by name.
fn refus<T>(methode: &'static str) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)))
}

fn texte(s: impl Into<String>) -> Value {
    Value::from(s.into())
}

fn proprietes_run(r: &ProtocolRun) -> Result<Vec<(&'static str, Value)>> {
    let opt = |u: Option<Uuid>| texte(u.map(|u| u.to_string()).unwrap_or_default());
    Ok(vec![
        ("id", texte(r.id.to_string())),
        ("protocol_id", texte(r.protocol_id.to_string())),
        ("project_id", opt(r.project_id)),
        ("plan_id", opt(r.plan_id)),
        ("task_id", opt(r.task_id)),
        ("parent_run_id", opt(r.parent_run_id)),
        ("current_state", texte(r.current_state.to_string())),
        (
            "states_visited_json",
            texte(serde_json::to_string(&r.states_visited)?),
        ),
        ("status", texte(r.status.to_string())),
        ("started_at", texte(r.started_at.to_rfc3339())),
        (
            "completed_at",
            texte(r.completed_at.map(|d| d.to_rfc3339()).unwrap_or_default()),
        ),
        ("error", texte(r.error.clone().unwrap_or_default())),
        ("triggered_by", texte(r.triggered_by.clone())),
        ("depth", Value::from(i64::from(r.depth))),
        (
            "version",
            Value::from(i64::try_from(r.version).unwrap_or(i64::MAX)),
        ),
        ("runner_managed", Value::from(r.runner_managed)),
    ])
}

fn lire_run(get: Lecteur) -> Result<ProtocolRun> {
    let id = uuid_requis(get, "id", "nœud ProtocolRun")?;
    let quoi = format!("run {id}");
    let quoi = quoi.as_str();
    let visites: Vec<StateVisit> = match facultative(get, "states_visited_json") {
        None => Vec::new(),
        Some(brut) => serde_json::from_str(&brut)
            .map_err(|e| anyhow!("{quoi} : `states_visited_json` illisible : {e}"))?,
    };
    Ok(ProtocolRun {
        id,
        protocol_id: uuid_requis(get, "protocol_id", quoi)?,
        project_id: uuid_facultatif(get, "project_id", quoi)?,
        plan_id: uuid_facultatif(get, "plan_id", quoi)?,
        task_id: uuid_facultatif(get, "task_id", quoi)?,
        parent_run_id: uuid_facultatif(get, "parent_run_id", quoi)?,
        current_state: uuid_requis(get, "current_state", quoi)?,
        states_visited: visites,
        // An unknown status is an error: a failed run must never read as running.
        status: enumeration(get, "status", quoi)?,
        started_at: date_requise(get, "started_at", quoi)?,
        completed_at: date_facultative(get, "completed_at", quoi)?,
        error: facultative(get, "error"),
        triggered_by: super::protocol::facultative(get, "triggered_by")
            .unwrap_or_else(|| "manual".to_string()),
        depth: non_signe(
            get("depth").and_then(|v| v.as_int64()).unwrap_or(0),
            u32::MAX,
        ),
        version: non_signe(
            get("version").and_then(|v| v.as_int64()).unwrap_or(0),
            u64::MAX,
        ),
        runner_managed: get("runner_managed")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    })
}

fn proprietes_runtime(s: &RuntimeState) -> Vec<(&'static str, Value)> {
    let opt = |u: Option<Uuid>| texte(u.map(|u| u.to_string()).unwrap_or_default());
    vec![
        ("id", texte(s.id.to_string())),
        ("run_id", texte(s.run_id.to_string())),
        ("generated_by", texte(s.generated_by.to_string())),
        ("name", texte(s.name.clone())),
        ("index", Value::from(i64::from(s.index))),
        ("sub_protocol_id", opt(s.sub_protocol_id)),
        ("action", texte(s.action.clone().unwrap_or_default())),
        ("status", texte(s.status.clone())),
        ("next_runtime_state_id", opt(s.next_runtime_state_id)),
        ("linking_strategy", texte(s.linking_strategy.to_string())),
    ]
}

fn lire_runtime(get: Lecteur) -> Result<RuntimeState> {
    let id = uuid_requis(get, "id", "nœud RuntimeState")?;
    let quoi = format!("état d'exécution {id}");
    let quoi = quoi.as_str();
    Ok(RuntimeState {
        id,
        run_id: uuid_requis(get, "run_id", quoi)?,
        generated_by: uuid_requis(get, "generated_by", quoi)?,
        name: requise(get, "name", quoi)?,
        index: non_signe(
            get("index").and_then(|v| v.as_int64()).unwrap_or(0),
            u32::MAX,
        ),
        sub_protocol_id: uuid_facultatif(get, "sub_protocol_id", quoi)?,
        action: facultative(get, "action"),
        status: facultative(get, "status").unwrap_or_else(|| "pending".to_string()),
        next_runtime_state_id: uuid_facultatif(get, "next_runtime_state_id", quoi)?,
        linking_strategy: enumeration(get, "linking_strategy", quoi)?,
    })
}

/// Ids with their engine nodes.
pub(super) type Noeuds = Vec<(Uuid, NodeId)>;

/// The runs and runtime states, decoded by id, rebuilt at every open.
#[derive(Default)]
pub(super) struct IndexRuns {
    runs: HashMap<Uuid, Fiche<ProtocolRun>>,
    etats: HashMap<Uuid, Fiche<RuntimeState>>,
}

impl IndexRuns {
    /// Rebuilds the index. A run of a protocol that is not there, a runtime state
    /// of a run that is not there, a repeated id: named in the report, untouched.
    pub(super) fn reconstruire(
        db: &ObrainDB,
        protocoles: &super::protocol::IndexProtocoles,
        rapport: &mut RapportOuverture,
    ) -> Self {
        let mut index = Self::default();
        for f in charger(db, RUN, rapport, &lire_run) {
            if !protocoles.protocoles.contains_key(&f.valeur.protocol_id) {
                rapport.en_quarantaine.push(format!(
                    "nœud {RUN} {:?} : le protocole {} n'existe pas",
                    f.noeud, f.valeur.protocol_id
                ));
            } else {
                match index.runs.entry(f.valeur.id) {
                    Entry::Occupied(_) => rapport.en_quarantaine.push(format!(
                        "nœud {RUN} {:?} : id {} déjà pris",
                        f.noeud, f.valeur.id
                    )),
                    Entry::Vacant(v) => {
                        v.insert(f);
                    }
                }
            }
        }
        for f in charger(db, ETAT_RUNTIME, rapport, &lire_runtime) {
            if !index.runs.contains_key(&f.valeur.run_id) {
                rapport.en_quarantaine.push(format!(
                    "nœud {ETAT_RUNTIME} {:?} : le run {} n'existe pas",
                    f.noeud, f.valeur.run_id
                ));
            } else {
                match index.etats.entry(f.valeur.id) {
                    Entry::Occupied(_) => rapport.en_quarantaine.push(format!(
                        "nœud {ETAT_RUNTIME} {:?} : id {} déjà pris",
                        f.noeud, f.valeur.id
                    )),
                    Entry::Vacant(v) => {
                        v.insert(f);
                    }
                }
            }
        }
        index
    }

    /// The runs of a protocol and the runtime states of those runs, with their
    /// engine nodes: what the cascade of `delete_protocol` reaches.
    pub(super) fn du_protocole(&self, protocole: Uuid) -> (Noeuds, Noeuds) {
        let runs: Vec<(Uuid, NodeId)> = self
            .runs
            .iter()
            .filter(|(_, f)| f.valeur.protocol_id == protocole)
            .map(|(id, f)| (*id, f.noeud))
            .collect();
        let ids: HashSet<Uuid> = runs.iter().map(|(id, _)| *id).collect();
        let etats = self
            .etats
            .iter()
            .filter(|(_, f)| ids.contains(&f.valeur.run_id))
            .map(|(id, f)| (*id, f.noeud))
            .collect();
        (runs, etats)
    }

    /// Forgets runs and runtime states (after their nodes are gone and durable).
    pub(super) fn retirer(&mut self, runs: &[Uuid], etats: &[Uuid]) {
        for id in runs {
            self.runs.remove(id);
        }
        for id in etats {
            self.etats.remove(id);
        }
    }

    fn enfants<'a>(&'a self, parent: Uuid) -> impl Iterator<Item = &'a ProtocolRun> + 'a {
        self.runs
            .values()
            .map(|f| &f.valeur)
            .filter(move |r| r.parent_run_id == Some(parent))
    }
}

/// `ORDER BY started_at DESC, id DESC`.
fn plus_recent_d_abord(a: &ProtocolRun, b: &ProtocolRun) -> std::cmp::Ordering {
    b.started_at
        .cmp(&a.started_at)
        .then_with(|| b.id.cmp(&a.id))
}

/// A run belongs to a project through its own `project_id` or, when it has none,
/// through its protocol's.
fn est_du_projet(
    run: &ProtocolRun,
    projet: Uuid,
    protocoles: &super::protocol::IndexProtocoles,
) -> bool {
    match run.project_id {
        Some(p) => p == projet,
        None => protocoles
            .protocoles
            .get(&run.protocol_id)
            .is_some_and(|f| f.valeur.project_id == Some(projet)),
    }
}

// The refusals keep their parameters' names for the reader; they are unused by design.
#[allow(unused_variables)]
#[async_trait]
impl ProtocolRunStore for ObrainGraphStore {
    async fn create_protocol_run(&self, run: &ProtocolRun) -> Result<()> {
        let r = run.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let proprietes = proprietes_run(&r)?;
                {
                    let etat = w.index()?;
                    if !etat.protocoles.protocoles.contains_key(&r.protocol_id) {
                        bail!(
                            "le protocole {} n'existe pas : le run {} ne peut pas s'y rattacher",
                            r.protocol_id,
                            r.id
                        );
                    }
                    if etat.runs.runs.contains_key(&r.id) {
                        bail!("un run d'id {} existe déjà", r.id);
                    }
                    // One running ROOT run per protocol; a child is exempt (its
                    // parent manages the exclusion).
                    if r.parent_run_id.is_none()
                        && etat.runs.runs.values().any(|e| {
                            e.valeur.protocol_id == r.protocol_id
                                && e.valeur.status == RunStatus::Running
                                && e.valeur.parent_run_id.is_none()
                        })
                    {
                        bail!(
                            "Skipped: concurrent run already running for protocol {}",
                            r.protocol_id
                        );
                    }
                }
                let fiche = creer(w, RUN, proprietes, &lire_run)?;
                w.index_mut()?.runs.runs.insert(r.id, fiche);
                Ok(())
            },
        )
        .await
    }

    async fn get_protocol_run(&self, run_id: Uuid) -> Result<Option<ProtocolRun>> {
        self.lire(|_, etat| Ok(etat.runs.runs.get(&run_id).map(|f| f.valeur.clone())))
    }

    async fn update_protocol_run(&self, run: &mut ProtocolRun) -> Result<()> {
        let r = run.clone();
        let nouvelle = self
            .ecrire(
                || (),
                move |(), w: &mut Ecriture<'_>| {
                    let (noeud, ancien) = {
                        let etat = w.index()?;
                        let Some(f) = etat.runs.runs.get(&r.id) else {
                            bail!("ProtocolRun not found: {}", r.id);
                        };
                        if f.valeur.version != r.version {
                            bail!(
                                "OptimisticLockError: protocol run {} was modified concurrently \
                                 (expected version {}, found {})",
                                r.id,
                                r.version,
                                f.valeur.version
                            );
                        }
                        (f.noeud, f.valeur.clone())
                    };
                    let ancien_en_blob = a_un_blob(&proprietes_run(&ancien)?);
                    let nouvelle = r
                        .version
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("version du run {} au maximum", r.id))?;
                    let mut ecrit = r.clone();
                    ecrit.version = nouvelle;
                    // Only what Neo4j's update sets: the links, `started_at`,
                    // `depth` and `triggered_by` are the run's identity.
                    let maj = proprietes_run(&ecrit)?
                        .into_iter()
                        .filter(|(k, _)| {
                            matches!(
                                *k,
                                "current_state"
                                    | "states_visited_json"
                                    | "status"
                                    | "completed_at"
                                    | "error"
                                    | "version"
                                    | "runner_managed"
                            )
                        })
                        .collect();
                    let fiche = reecrire(w, noeud, maj, ancien_en_blob, &lire_run)?;
                    w.index_mut()?.runs.runs.insert(r.id, fiche);
                    Ok(nouvelle)
                },
            )
            .await?;
        // The caller's struct stays in step with the store.
        run.version = nouvelle;
        Ok(())
    }

    async fn list_protocol_runs(
        &self,
        protocol_id: Uuid,
        status: Option<RunStatus>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<ProtocolRun>, usize)> {
        self.lire(|_, etat| {
            let mut runs: Vec<&ProtocolRun> = etat
                .runs
                .runs
                .values()
                .map(|f| &f.valeur)
                .filter(|r| r.protocol_id == protocol_id && status.is_none_or(|s| r.status == s))
                .collect();
            runs.sort_by(|a, b| plus_recent_d_abord(a, b));
            let total = runs.len();
            Ok((
                runs.into_iter().skip(offset).take(limit).cloned().collect(),
                total,
            ))
        })
    }

    async fn list_all_protocol_runs(
        &self,
        status: Option<RunStatus>,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<ProtocolRun>, usize)> {
        self.lire(|_, etat| {
            // The scope is resolved through the OWNING PROTOCOL's `project_id`;
            // `project_id` wins over `workspace_slug`. An unknown slug is no project.
            let portee: Option<HashSet<Uuid>> = if let Some(p) = project_id {
                Some(HashSet::from([p]))
            } else {
                workspace_slug.map(|slug| etat.espaces.projets_du_slug(slug))
            };
            let mut runs: Vec<&ProtocolRun> = etat
                .runs
                .runs
                .values()
                .map(|f| &f.valeur)
                .filter(|r| status.is_none_or(|s| r.status == s))
                .filter(|r| {
                    portee.as_ref().is_none_or(|projets| {
                        etat.protocoles
                            .protocoles
                            .get(&r.protocol_id)
                            .and_then(|p| p.valeur.project_id)
                            .is_some_and(|p| projets.contains(&p))
                    })
                })
                .collect();
            runs.sort_by(|a, b| plus_recent_d_abord(a, b));
            let total = runs.len();
            Ok((
                runs.into_iter().skip(offset).take(limit).cloned().collect(),
                total,
            ))
        })
    }

    async fn list_child_runs(&self, parent_run_id: Uuid) -> Result<Vec<ProtocolRun>> {
        self.lire(|_, etat| {
            if !etat.runs.runs.contains_key(&parent_run_id) {
                return Ok(Vec::new());
            }
            let mut enfants: Vec<ProtocolRun> = etat.runs.enfants(parent_run_id).cloned().collect();
            // ORDER BY started_at ASC, id ASC.
            enfants.sort_by(|a, b| {
                a.started_at
                    .cmp(&b.started_at)
                    .then_with(|| a.id.cmp(&b.id))
            });
            Ok(enfants)
        })
    }

    async fn count_child_runs(&self, parent_run_id: Uuid) -> Result<usize> {
        self.lire(|_, etat| {
            if !etat.runs.runs.contains_key(&parent_run_id) {
                return Ok(0);
            }
            Ok(etat.runs.enfants(parent_run_id).count())
        })
    }

    async fn get_run_tree(&self, root_run_id: Uuid) -> Result<Vec<ProtocolRun>> {
        self.lire(|_, etat| {
            let Some(racine) = etat.runs.runs.get(&root_run_id) else {
                return Ok(Vec::new());
            };
            let mut vus: HashSet<Uuid> = HashSet::from([root_run_id]);
            let mut arbre: Vec<ProtocolRun> = vec![racine.valeur.clone()];
            let mut frontiere = vec![root_run_id];
            // `CHILD_OF*1..5`: at most five levels below the root; a cycle is walked once.
            for _ in 0..PROFONDEUR_ARBRE {
                let mut suivante = Vec::new();
                for parent in &frontiere {
                    for enfant in etat.runs.enfants(*parent) {
                        if vus.insert(enfant.id) {
                            arbre.push(enfant.clone());
                            suivante.push(enfant.id);
                        }
                    }
                }
                if suivante.is_empty() {
                    break;
                }
                frontiere = suivante;
            }
            // ORDER BY depth ASC, started_at ASC, id ASC (the `depth` property).
            arbre.sort_by(|a, b| {
                a.depth
                    .cmp(&b.depth)
                    .then_with(|| a.started_at.cmp(&b.started_at))
                    .then_with(|| a.id.cmp(&b.id))
            });
            Ok(arbre)
        })
    }

    async fn delete_protocol_run(&self, run_id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, etats) = {
                    let etat = w.index()?;
                    let Some(f) = etat.runs.runs.get(&run_id) else {
                        return Ok(false);
                    };
                    let etats: Vec<(Uuid, NodeId)> = etat
                        .runs
                        .etats
                        .iter()
                        .filter(|(_, e)| e.valeur.run_id == run_id)
                        .map(|(id, e)| (*id, e.noeud))
                        .collect();
                    (f.noeud, etats)
                };
                // The runtime states first, the run last (module doc).
                for (_, n) in &etats {
                    supprimer(w, *n)?;
                }
                supprimer(w, noeud)?;
                w.rendre_durable()?;
                w.index_mut()?.runs.retirer(
                    &[run_id],
                    &etats.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
                );
                Ok(true)
            },
        )
        .await
    }

    async fn create_produced_during(
        &self,
        entity_type: &str,
        entity_id: Uuid,
        run_id: Uuid,
    ) -> Result<bool> {
        refus("ProtocolRunStore::create_produced_during")
    }

    async fn get_run_outcomes(&self, run_id: Uuid) -> Result<Vec<ProducedArtefact>> {
        refus("ProtocolRunStore::get_run_outcomes")
    }

    async fn find_active_run_for_project(&self, project_id: Uuid) -> Result<Option<Uuid>> {
        self.lire(|_, etat| {
            // The most recently started running run whose protocol exists.
            Ok(etat
                .runs
                .runs
                .values()
                .map(|f| &f.valeur)
                .filter(|r| r.status == RunStatus::Running)
                .filter(|r| est_du_projet(r, project_id, &etat.protocoles))
                .filter(|r| etat.protocoles.protocoles.contains_key(&r.protocol_id))
                .max_by(|a, b| plus_recent_d_abord(b, a))
                .map(|r| r.id))
        })
    }

    async fn persist_reasoning_tree(
        &self,
        tree: &crate::reasoning::ReasoningTree,
        linked_entity_type: Option<&str>,
        linked_entity_id: Option<Uuid>,
    ) -> Result<Uuid> {
        refus("ProtocolRunStore::persist_reasoning_tree")
    }

    async fn get_run_reasoning_tree_id(&self, run_id: Uuid) -> Result<Option<Uuid>> {
        refus("ProtocolRunStore::get_run_reasoning_tree_id")
    }

    async fn list_completed_runs_for_project(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<ProtocolRun>> {
        self.lire(|_, etat| {
            let mut runs: Vec<&ProtocolRun> = etat
                .runs
                .runs
                .values()
                .map(|f| &f.valeur)
                .filter(|r| r.status == RunStatus::Completed)
                .filter(|r| est_du_projet(r, project_id, &etat.protocoles))
                .filter(|r| etat.protocoles.protocoles.contains_key(&r.protocol_id))
                .collect();
            // ORDER BY completed_at DESC, id DESC (a run with no date sorts last).
            runs.sort_by(|a, b| {
                let date = |r: &ProtocolRun| -> Option<DateTime<Utc>> { r.completed_at };
                date(b).cmp(&date(a)).then_with(|| b.id.cmp(&a.id))
            });
            Ok(runs.into_iter().take(limit).cloned().collect())
        })
    }

    async fn create_runtime_state(&self, state: &RuntimeState) -> Result<()> {
        let s = state.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                {
                    let etat = w.index()?;
                    if !etat.runs.runs.contains_key(&s.run_id) {
                        bail!(
                            "le run {} n'existe pas : l'état d'exécution {} ne peut pas s'y \
                             rattacher (Neo4j n'écrit alors rien, sans le dire)",
                            s.run_id,
                            s.id
                        );
                    }
                    if !etat.protocoles.etats.contains_key(&s.generated_by) {
                        bail!(
                            "l'état de protocole {} n'existe pas : il ne peut pas générer \
                             l'état d'exécution {} (Neo4j n'écrit alors rien, sans le dire)",
                            s.generated_by,
                            s.id
                        );
                    }
                    if etat.runs.etats.contains_key(&s.id) {
                        bail!("un état d'exécution d'id {} existe déjà", s.id);
                    }
                }
                let fiche = creer(w, ETAT_RUNTIME, proprietes_runtime(&s), &lire_runtime)?;
                w.index_mut()?.runs.etats.insert(s.id, fiche);
                Ok(())
            },
        )
        .await
    }

    async fn get_runtime_states(&self, run_id: Uuid) -> Result<Vec<RuntimeState>> {
        self.lire(|_, etat| {
            let mut etats: Vec<RuntimeState> = etat
                .runs
                .etats
                .values()
                .filter(|f| f.valeur.run_id == run_id)
                .map(|f| f.valeur.clone())
                .collect();
            // ORDER BY index ASC (the id breaks the ties).
            etats.sort_by(|a, b| a.index.cmp(&b.index).then_with(|| a.id.cmp(&b.id)));
            Ok(etats)
        })
    }

    async fn delete_runtime_states(&self, run_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let etats: Vec<(Uuid, NodeId)> = w
                    .index()?
                    .runs
                    .etats
                    .iter()
                    .filter(|(_, f)| f.valeur.run_id == run_id)
                    .map(|(id, f)| (*id, f.noeud))
                    .collect();
                if etats.is_empty() {
                    return Ok(());
                }
                for (_, n) in &etats {
                    supprimer(w, *n)?;
                }
                w.rendre_durable()?;
                let ids: Vec<Uuid> = etats.iter().map(|(i, _)| *i).collect();
                w.index_mut()?.runs.retirer(&[], &ids);
                Ok(())
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::ProtocolStore;
    use crate::obrain::Creation;
    use crate::protocol::{Protocol, ProtocolState};
    use chrono::TimeZone;

    fn a(s: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1_700_000_000 + s, 0).unwrap()
    }

    async fn protocole(store: &ObrainGraphStore) -> (Protocol, ProtocolState) {
        let etat_initial = ProtocolState::start(Uuid::new_v4(), "start");
        let mut p = Protocol::new(None, "p", etat_initial.id);
        p.id = etat_initial.protocol_id;
        store.upsert_protocol(&p).await.unwrap();
        store.upsert_protocol_state(&etat_initial).await.unwrap();
        (p, etat_initial)
    }

    fn run(p: &Protocol, parent: Option<&ProtocolRun>, s: i64) -> ProtocolRun {
        let mut r = ProtocolRun::new(p.id, p.entry_state, "start");
        r.started_at = a(s);
        r.parent_run_id = parent.map(|x| x.id);
        r.depth = parent.map(|x| x.depth + 1).unwrap_or(0);
        r
    }

    #[tokio::test]
    async fn un_run_avance_puis_survit_a_la_reouverture() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let id;
        let nouvel_etat = Uuid::new_v4();
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            let (p, _) = protocole(&store).await;
            let mut r = run(&p, None, 0);
            id = r.id;
            store.create_protocol_run(&r).await.unwrap();
            r.current_state = nouvel_etat;
            r.status = RunStatus::Completed;
            r.completed_at = Some(a(5));
            store.update_protocol_run(&mut r).await.unwrap();
            assert_eq!(r.version, 1);
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        let relu = store.get_protocol_run(id).await.unwrap().unwrap();
        assert_eq!(relu.current_state, nouvel_etat);
        assert_eq!(relu.status, RunStatus::Completed);
        assert_eq!(relu.version, 1);
        assert_eq!(relu.completed_at, Some(a(5)));
        assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
    }

    #[tokio::test]
    async fn la_garde_du_run_racine_et_le_cas() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let (p, _) = protocole(&store).await;
        let racine = run(&p, None, 0);
        store.create_protocol_run(&racine).await.unwrap();
        let second = run(&p, None, 1);
        let refus = store.create_protocol_run(&second).await.unwrap_err();
        assert!(refus.to_string().contains("Skipped: concurrent run"));
        assert!(store.get_protocol_run(second.id).await.unwrap().is_none());
        // A child is exempt.
        store
            .create_protocol_run(&run(&p, Some(&racine), 2))
            .await
            .unwrap();
        // A stale version loses.
        let mut frais = store.get_protocol_run(racine.id).await.unwrap().unwrap();
        let mut perime = frais.clone();
        frais.status = RunStatus::Failed;
        store.update_protocol_run(&mut frais).await.unwrap();
        perime.status = RunStatus::Cancelled;
        let e = store.update_protocol_run(&mut perime).await.unwrap_err();
        assert!(e.to_string().contains("OptimisticLockError"));
        assert_eq!(perime.version, 0, "a lost CAS leaves the caller's version");
        // A missing run is not a conflict.
        let mut absent = run(&p, None, 9);
        let e = store.update_protocol_run(&mut absent).await.unwrap_err();
        assert!(e.to_string().contains("not found"));
        // A run of a missing protocol is refused with its own message.
        let mut orphelin = run(&p, None, 10);
        orphelin.protocol_id = Uuid::new_v4();
        let e = store.create_protocol_run(&orphelin).await.unwrap_err();
        assert!(e.to_string().contains("n'existe pas"));
    }

    #[tokio::test]
    async fn arbre_enfants_et_listes() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let (p, _) = protocole(&store).await;
        let racine = run(&p, None, 0);
        let mut chaine = vec![racine.clone()];
        for i in 1..=6 {
            let r = run(&p, chaine.last(), i);
            chaine.push(r);
        }
        for r in &chaine {
            store.create_protocol_run(r).await.unwrap();
        }
        let arbre = store.get_run_tree(racine.id).await.unwrap();
        assert_eq!(arbre.len(), 6, "root + five levels, the sixth is cut");
        assert_eq!(arbre[0].id, racine.id);
        assert_eq!(store.count_child_runs(racine.id).await.unwrap(), 1);
        assert_eq!(store.list_child_runs(racine.id).await.unwrap().len(), 1);
        assert!(store.get_run_tree(Uuid::new_v4()).await.unwrap().is_empty());
        let (page, total) = store.list_protocol_runs(p.id, None, 2, 0).await.unwrap();
        assert_eq!((page.len(), total), (2, 7));
        assert_eq!(page[0].id, chaine[6].id, "newest first");
        // Once the parent is gone its children are no longer its children.
        assert!(store.delete_protocol_run(racine.id).await.unwrap());
        assert_eq!(store.count_child_runs(racine.id).await.unwrap(), 0);
        assert!(!store.delete_protocol_run(racine.id).await.unwrap());
    }

    #[tokio::test]
    async fn etats_d_execution_et_cascade_du_protocole() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let (p, generateur) = protocole(&store).await;
        let r = run(&p, None, 0);
        store.create_protocol_run(&r).await.unwrap();
        for i in [2, 0, 1] {
            let s = RuntimeState::new(r.id, generateur.id, format!("s{i}"), i);
            store.create_runtime_state(&s).await.unwrap();
        }
        let noms: Vec<String> = store
            .get_runtime_states(r.id)
            .await
            .unwrap()
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(noms, ["s0", "s1", "s2"]);
        // A runtime state needs its run and its generating state.
        assert!(store
            .create_runtime_state(&RuntimeState::new(Uuid::new_v4(), generateur.id, "x", 0))
            .await
            .is_err());
        assert!(store
            .create_runtime_state(&RuntimeState::new(r.id, Uuid::new_v4(), "x", 0))
            .await
            .is_err());
        store.delete_runtime_states(r.id).await.unwrap();
        assert!(store.get_runtime_states(r.id).await.unwrap().is_empty());
        // The cascade of the protocol reaches runs and what they hold.
        let s = RuntimeState::new(r.id, generateur.id, "y", 0);
        store.create_runtime_state(&s).await.unwrap();
        assert!(store.delete_protocol(p.id).await.unwrap());
        assert!(store.get_protocol_run(r.id).await.unwrap().is_none());
        assert!(store.get_runtime_states(r.id).await.unwrap().is_empty());
        assert_eq!(store.interne.db.store().node_count(), 0);
    }

    #[tokio::test]
    async fn actifs_termines_et_refus_nommes() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let (p, _) = protocole(&store).await;
        let projet = Uuid::new_v4();
        let mut r = run(&p, None, 0);
        r.project_id = Some(projet);
        store.create_protocol_run(&r).await.unwrap();
        assert_eq!(
            store.find_active_run_for_project(projet).await.unwrap(),
            Some(r.id)
        );
        assert_eq!(
            store
                .find_active_run_for_project(Uuid::new_v4())
                .await
                .unwrap(),
            None
        );
        let mut r = store.get_protocol_run(r.id).await.unwrap().unwrap();
        r.status = RunStatus::Completed;
        r.completed_at = Some(a(3));
        store.update_protocol_run(&mut r).await.unwrap();
        assert_eq!(
            store
                .list_completed_runs_for_project(projet, 5)
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(store
            .find_active_run_for_project(projet)
            .await
            .unwrap()
            .is_none());
        let e = store.get_run_outcomes(r.id).await.unwrap_err();
        assert_eq!(
            e.downcast_ref::<NonPorte>(),
            Some(&NonPorte("ProtocolRunStore::get_run_outcomes"))
        );
        let e = store
            .create_produced_during("note", Uuid::new_v4(), r.id)
            .await
            .unwrap_err();
        assert!(e.downcast_ref::<NonPorte>().is_some());
        assert!(store.get_run_reasoning_tree_id(r.id).await.is_err());
    }
}
