//! `PlanStore` on obrain: the STEPS of a task (`(:Task)-[:HAS_STEP]->(:Step)`),
//! with the Neo4j semantics of `src/neo4j/step.rs`. The index, its rebuilding at
//! open, the atomicity rules and the divergences are those of `task.rs`.
//!
//! PORTED: `create_step`, `get_step`, `get_task_steps`, `update_step`,
//! `update_step_status`, `delete_step`, `get_task_step_progress`,
//! `complete_pending_steps_for_task`. Nothing of the step half is refused.
//!
//! A step's properties are those `Neo4jClient::create_step` writes: `id`,
//! `order`, `description`, `status` (variant name), `verification` (`""` for
//! none), `created_at`, `updated_at` (= `created_at`). `execution_context` and
//! `persona` always read `None`, as Neo4j's step readers do.
//!
//! Durability: a creation or an update carrying a value above
//! `SEUIL_COLONNE_BLOB` (or replacing one) pays the full flush, through
//! `task::creer_sous_brouillon` and `task::poser` (see `task.rs`).
//!
//! Order: `ORDER BY s.order`, then the id for a tie (Neo4j leaves it free).
//! `create_step` on a task this backend does not hold is REFUSED (Neo4j creates
//! nothing and answers `Ok`).

use anyhow::{anyhow, bail, Result};
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::task::{
    creer_sous_brouillon, detacher, ecritures, lecteur, poser, EntreeEtape, IndexTaches, A_ETAPE,
    LABEL_ETAPE, SUPPRESSION,
};
use super::{Ecriture, NonPorte, ObrainGraphStore};
use crate::neo4j::models::{StepNode, StepStatus};
use crate::plan::models::UpdateStepRequest;

fn nom_statut(s: &StepStatus) -> &'static str {
    match s {
        StepStatus::Pending => "Pending",
        StepStatus::InProgress => "InProgress",
        StepStatus::Completed => "Completed",
        StepStatus::Skipped => "Skipped",
    }
}

fn statut_depuis(s: &str) -> Option<StepStatus> {
    Some(match s {
        "Pending" => StepStatus::Pending,
        "InProgress" => StepStatus::InProgress,
        "Completed" => StepStatus::Completed,
        "Skipped" => StepStatus::Skipped,
        _ => return None,
    })
}

/// The domain value of a `Step` node (Neo4j's `get_task_steps` decoding, strict
/// on an unknown status and an unreadable `created_at`).
pub(super) fn lire_etape(get: &dyn Fn(&str) -> Option<Value>) -> Result<StepNode> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let instant = |cle: &str| chaine(cle).and_then(|s| s.parse::<DateTime<Utc>>().ok());
    let id: Uuid = chaine("id")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("nœud Step sans propriété `id` lisible"))?;
    let requise =
        |cle: &str| chaine(cle).ok_or_else(|| anyhow!("étape {id} sans propriété `{cle}`"));
    let ordre = get("order")
        .and_then(|v| v.as_int64())
        .ok_or_else(|| anyhow!("étape {id} sans propriété `order`"))?;
    let statut_brut = requise("status")?;
    let status = statut_depuis(&statut_brut)
        .ok_or_else(|| anyhow!("étape {id} : statut inconnu « {statut_brut} »"))?;
    let cree_brut = requise("created_at")?;
    let created_at: DateTime<Utc> = cree_brut
        .parse()
        .map_err(|e| anyhow!("étape {id} : `created_at` illisible (« {cree_brut} ») : {e}"))?;
    Ok(StepNode {
        id,
        // Neo4j's `unsigned`: a negative reads 0, too large saturates.
        order: u32::try_from(ordre.max(0)).unwrap_or(u32::MAX),
        description: requise("description")?,
        status,
        verification: chaine("verification").filter(|s| !s.is_empty()),
        created_at,
        updated_at: instant("updated_at"),
        completed_at: instant("completed_at"),
        execution_context: None,
        persona: None,
    })
}

fn decoder(db: &ObrainDB, noeud: NodeId) -> Result<StepNode> {
    let get = lecteur(db, noeud, LABEL_ETAPE)?;
    lire_etape(&get)
}

/// The steps of a task, `ORDER BY order` (then id).
pub(super) fn etapes_triees(idx: &IndexTaches, tache: Uuid) -> Vec<StepNode> {
    let mut v: Vec<StepNode> = idx
        .par_tache
        .get(&tache)
        .into_iter()
        .flatten()
        .filter_map(|s| idx.etapes.get(s).map(|e| e.etape.clone()))
        .collect();
    v.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.id.cmp(&b.id)));
    v
}

/// Properties to set (`Some`) or remove (`None`).
type Champs = Vec<(&'static str, Option<Value>)>;

fn maintenant() -> Value {
    Value::from(Utc::now().to_rfc3339())
}

impl IndexTaches {
    /// The step `id` for a read: `Err` when its id is reserved by a quarantined node.
    fn etape(&self, id: Uuid) -> Result<Option<&EntreeEtape>> {
        if self.reserves_etapes.contains(&id) {
            bail!(
                "l'étape {id} est en quarantaine (nœud illisible à l'ouverture) : le backend obrain \
                 ne répond pas pour elle"
            );
        }
        Ok(self.etapes.get(&id))
    }

    /// The steps of a task for a read: `Err` for a quarantined task, or one with a
    /// step this index could not read.
    fn etapes_lisibles(&self, tache: Uuid) -> Result<Vec<StepNode>> {
        self.tache(tache)?;
        self.complet(tache)?;
        Ok(etapes_triees(self, tache))
    }
}

impl ObrainGraphStore {
    pub(super) async fn etape_creer(&self, task_id: Uuid, step: &StepNode) -> Result<()> {
        let s = step.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud_tache = {
                    let etat = w.index()?;
                    let idx = &etat.taches;
                    if idx.etapes.contains_key(&s.id) {
                        bail!("une étape d'id {} existe déjà (unicité de Step.id)", s.id);
                    }
                    if idx.reserves_etapes.contains(&s.id) {
                        bail!(
                            "l'id {} est réservé : un nœud Step de cet id est sur disque, \
                             illisible à l'ouverture (quarantaine). Il n'est ni repris ni écrasé",
                            s.id
                        );
                    }
                    let Some(t) = idx.tache(task_id)? else {
                        bail!(
                            "la tâche {task_id} n'existe pas : l'étape {} n'est pas créée (Neo4j \
                             ne créerait rien et répondrait Ok ; le backend obrain le dit)",
                            s.id
                        );
                    };
                    t.noeud
                };
                let cree = s.created_at.to_rfc3339();
                let proprietes = vec![
                    ("id", Value::from(s.id.to_string())),
                    ("order", Value::from(s.order as i64)),
                    ("description", Value::from(s.description.clone())),
                    ("status", Value::from(nom_statut(&s.status))),
                    (
                        "verification",
                        Value::from(s.verification.clone().unwrap_or_default()),
                    ),
                    ("created_at", Value::from(cree.clone())),
                    ("updated_at", Value::from(cree)),
                ];
                let noeud = ecritures(w, |w| {
                    creer_sous_brouillon(w, LABEL_ETAPE, proprietes, Some((noeud_tache, A_ETAPE)))
                })?;
                let etape = w.apres_ecriture(decoder(w.db, noeud).and_then(|r| {
                    if r.id == s.id {
                        Ok(r)
                    } else {
                        bail!("l'étape relue n'a pas l'id écrit")
                    }
                }))?;
                w.rendre_durable()?;
                w.index_mut()?.taches.inscrire_etape(EntreeEtape {
                    noeud,
                    etape,
                    tache: task_id,
                    etrangeres: vec![],
                });
                Ok(())
            },
        )
        .await
    }

    pub(super) async fn etape_lire(&self, step_id: Uuid) -> Result<Option<StepNode>> {
        self.lire(|_, etat| Ok(etat.taches.etape(step_id)?.map(|e| e.etape.clone())))
    }

    pub(super) async fn etapes_de(&self, task_id: Uuid) -> Result<Vec<StepNode>> {
        self.lire(|_, etat| etat.taches.etapes_lisibles(task_id))
    }

    pub(super) async fn etapes_progression(&self, task_id: Uuid) -> Result<(u32, u32)> {
        self.lire(|_, etat| {
            let etapes = etat.taches.etapes_lisibles(task_id)?;
            let faites = etapes
                .iter()
                .filter(|s| s.status == StepStatus::Completed)
                .count();
            Ok((faites as u32, etapes.len() as u32))
        })
    }

    /// Writes, on each step `cibles` picks among the current state, the properties
    /// it gives, then publishes. Answers how many steps were written.
    async fn etapes_ecrire<F>(&self, cibles: F) -> Result<u32>
    where
        F: FnOnce(&IndexTaches) -> Result<Vec<(Uuid, Champs)>> + Send + 'static,
    {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let travail: Vec<(Uuid, NodeId, Champs)> = {
                    let etat = w.index()?;
                    let idx = &etat.taches;
                    cibles(idx)?
                        .into_iter()
                        .filter_map(|(id, c)| idx.etapes.get(&id).map(|e| (id, e.noeud, c)))
                        .collect()
                };
                if travail.is_empty() {
                    return Ok(0);
                }
                let noeuds: Vec<(Uuid, NodeId)> =
                    travail.iter().map(|(id, n, _)| (*id, *n)).collect();
                ecritures(w, |w| {
                    for (_, noeud, champs) in travail {
                        poser(w, noeud, champs)?;
                    }
                    Ok(())
                })?;
                let relues: Vec<(Uuid, StepNode)> = noeuds
                    .iter()
                    .map(|(id, n)| w.apres_ecriture(decoder(w.db, *n)).map(|s| (*id, s)))
                    .collect::<Result<_>>()?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for (id, s) in relues {
                    if let Some(e) = etat.taches.etapes.get_mut(&id) {
                        e.etape = s;
                    }
                }
                Ok(noeuds.len() as u32)
            },
        )
        .await
    }

    pub(super) async fn etape_maj(&self, step_id: Uuid, updates: &UpdateStepRequest) -> Result<()> {
        // Only a status (or nothing): skipped, as Neo4j (`update_step_status` does it).
        if updates.description.is_none() && updates.verification.is_none() {
            return Ok(());
        }
        let u = updates.clone();
        self.etapes_ecrire(move |idx| {
            if idx.etape(step_id)?.is_none() {
                return Ok(vec![]);
            }
            let mut champs = vec![("updated_at", Some(maintenant()))];
            if let Some(d) = u.description {
                champs.push(("description", Some(Value::from(d))));
            }
            if let Some(v) = u.verification {
                champs.push(("verification", Some(Value::from(v))));
            }
            Ok(vec![(step_id, champs)])
        })
        .await
        .map(|_| ())
    }

    pub(super) async fn etape_statut(&self, step_id: Uuid, status: StepStatus) -> Result<()> {
        self.etapes_ecrire(move |idx| {
            if idx.etape(step_id)?.is_none() {
                return Ok(vec![]);
            }
            let now = maintenant();
            let mut champs = vec![("status", Some(Value::from(nom_statut(&status))))];
            if matches!(status, StepStatus::Completed | StepStatus::Skipped) {
                champs.push(("completed_at", Some(now.clone())));
            }
            champs.push(("updated_at", Some(now)));
            Ok(vec![(step_id, champs)])
        })
        .await
        .map(|_| ())
    }

    pub(super) async fn etapes_completer(&self, task_id: Uuid) -> Result<u32> {
        self.etapes_ecrire(move |idx| {
            let now = maintenant();
            Ok(idx
                .etapes_lisibles(task_id)?
                .into_iter()
                .filter(|s| matches!(s.status, StepStatus::Pending | StepStatus::InProgress))
                .map(|s| {
                    (
                        s.id,
                        vec![
                            ("status", Some(Value::from("Completed"))),
                            ("completed_at", Some(now.clone())),
                            ("updated_at", Some(now.clone())),
                        ],
                    )
                })
                .collect())
        })
        .await
    }

    pub(super) async fn etape_supprimer(&self, step_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud = {
                    let etat = w.index()?;
                    let Some(e) = etat.taches.etape(step_id)? else {
                        return Ok(());
                    };
                    if !e.etrangeres.is_empty() {
                        return Err(anyhow::Error::new(NonPorte(
                            "PlanStore::delete_step (cascade)",
                        ))
                        .context(format!(
                            "l'étape {step_id} porte des arêtes de domaines non portés ({}) : le \
                             DETACH DELETE ne peut pas les atteindre ici",
                            e.etrangeres.join(", ")
                        )));
                    }
                    e.noeud
                };
                ecritures(w, |w| {
                    w.etape()?;
                    w.db.set_node_property(noeud, SUPPRESSION, Value::from(true));
                    detacher(w, noeud)
                })?;
                w.rendre_durable()?;
                w.index_mut()?.taches.retirer_etape(step_id);
                Ok(())
            },
        )
        .await
    }
}
