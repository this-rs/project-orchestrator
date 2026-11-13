//! `TriggerStore` on obrain: the alerts (Heartbeat Engine), the event triggers
//! and the lifecycle hooks. The reference is the Neo4j adapter
//! (`src/neo4j/alert.rs`, `src/neo4j/event_trigger.rs`,
//! `src/neo4j/lifecycle_hook.rs`): the same labels, the same property names and
//! the same lossy conventions (an absent optional text is stored as `""` and
//! reads back as `None`).
//!
//! What is ported, and what is refused by name:
//!
//! - `create_alert`: the MERGE on `dedup_key` is an in-memory lookup under the
//!   writer lock (one writer: the MERGE cannot race). The priority is the one the
//!   Cypher computes (`weight * (1 + min(ln(count) / 10, 0.5))`, 0 when
//!   acknowledged), NOT `AlertNode::compute_priority`, which adds a recency
//!   factor the Neo4j write never applied.
//! - `list_alerts` with a `workspace_slug` and no `project_id` is REFUSED: the
//!   filter walks `(:Workspace)<-[:BELONGS_TO_WORKSPACE]-(:Project)`, and the
//!   workspace domain is not ported here. Answering it would be a silent empty
//!   or unfiltered page.
//! - `create_event_trigger` and `create_lifecycle_hook` REFUSE an id already
//!   taken. Neo4j's `CREATE` would write a second node under the same id, and
//!   `get` would then answer the first one: a divergence, not a value to copy.
//!
//! Order (all total, the id breaks every tie; `Uuid` order is the order of its
//! canonical string):
//!
//! - pending alerts and `list_alerts`: `priority DESC, last_seen DESC, id DESC`
//!   (`last_seen` falls back to `created_at`, as the `coalesce` does);
//! - event triggers: `created_at DESC, id DESC`;
//! - lifecycle hooks: `priority ASC, id ASC`.
//!
//! Timestamps are compared as instants, not as strings. The Neo4j query compares
//! the RFC 3339 strings, which agree with the instants for the strings this
//! adapter writes (one writer, one format).
//!
//! Durability follows `project.rs`: a creation writes a draft marker first and
//! removes it last (the commit point), a blob-sized value forces the full flush
//! before the marker goes, opening deletes a draft, and every write is published
//! to the in-memory index only after its durability step.
//!
//! Reads are served from the index, which holds the decoded values and is
//! rebuilt from the base at open. Known cost, not yet measured here: the list
//! queries sort the whole domain (O(N log N) per call), and the open rebuild
//! scans the three labels. An indexed order is an obrain lot.
//!
//! Not written here: the `EventTrigger` relation to a protocol
//! (`link_trigger_to_protocol`, not on the contract) and the `HAS_ALERT`
//! relation from a project to its alerts (Neo4j writes it, nothing reads it).

use std::cmp::Ordering;
use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::{Ecriture, NonPorte, ObrainGraphStore, RapportOuverture, SEUIL_COLONNE_BLOB};
use crate::events::trigger::EventTrigger;
use crate::lifecycle::{
    LifecycleActionType, LifecycleHook, LifecycleScope, UpdateLifecycleHookRequest,
};
use crate::neo4j::models::{AlertNode, AlertSeverity};
use crate::neo4j::traits::TriggerStore;

const LABEL_ALERTE: &str = "Alert";
const LABEL_DECLENCHEUR: &str = "EventTrigger";
const LABEL_HOOK: &str = "LifecycleHook";
/// The draft marker: present from the first record of a creation to the last.
const BROUILLON: &str = "_brouillon";

/// Does this value go to a blob column (a string above the threshold)?
fn en_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
}

/// One record: where it lives in the base, and its decoded value.
struct Entree<T> {
    noeud: NodeId,
    valeur: T,
}

/// The in-memory indexes of the three labels, rebuilt from the base at open.
#[derive(Default)]
pub(super) struct IndexDeclencheurs {
    alertes: HashMap<Uuid, Entree<AlertNode>>,
    alertes_par_cle: HashMap<String, Uuid>,
    declencheurs: HashMap<Uuid, Entree<EventTrigger>>,
    hooks: HashMap<Uuid, Entree<LifecycleHook>>,
}

type Lecteur<'a> = &'a dyn Fn(&str) -> Option<Value>;

impl IndexDeclencheurs {
    /// Rebuild the indexes from the persisted nodes of the three labels. A draft
    /// (the marker still there) is deleted; a record this backend cannot decode,
    /// or whose id or dedup key is taken, is quarantined: left untouched and named.
    pub(super) fn reconstruire(db: &ObrainDB) -> Result<(Self, RapportOuverture)> {
        let store = db.store();
        let mut index = Self::default();
        let mut rapport = RapportOuverture::default();

        for noeud in store.nodes_by_label(LABEL_ALERTE) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud Alert {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            if get(BROUILLON).is_some() {
                db.delete_node(noeud);
                rapport.brouillons_supprimes += 1;
                continue;
            }
            match lire_alerte(&get) {
                Ok(a) if index.alertes.contains_key(&a.id) => rapport
                    .en_quarantaine
                    .push(format!("nœud Alert {noeud:?} : id {} déjà pris", a.id)),
                Ok(a) if index.alertes_par_cle.contains_key(&a.dedup_key) => {
                    rapport.en_quarantaine.push(format!(
                        "nœud Alert {noeud:?} : dedup_key « {} » déjà pris",
                        a.dedup_key
                    ))
                }
                Ok(a) => {
                    index.alertes_par_cle.insert(a.dedup_key.clone(), a.id);
                    index.alertes.insert(a.id, Entree { noeud, valeur: a });
                }
                Err(e) => rapport
                    .en_quarantaine
                    .push(format!("nœud Alert {noeud:?} : {e}")),
            }
        }

        for noeud in store.nodes_by_label(LABEL_DECLENCHEUR) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud EventTrigger {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            if get(BROUILLON).is_some() {
                db.delete_node(noeud);
                rapport.brouillons_supprimes += 1;
                continue;
            }
            match lire_declencheur(&get) {
                Ok(t) if index.declencheurs.contains_key(&t.id) => rapport.en_quarantaine.push(
                    format!("nœud EventTrigger {noeud:?} : id {} déjà pris", t.id),
                ),
                Ok(t) => {
                    index.declencheurs.insert(t.id, Entree { noeud, valeur: t });
                }
                Err(e) => rapport
                    .en_quarantaine
                    .push(format!("nœud EventTrigger {noeud:?} : {e}")),
            }
        }

        for noeud in store.nodes_by_label(LABEL_HOOK) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud LifecycleHook {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            if get(BROUILLON).is_some() {
                db.delete_node(noeud);
                rapport.brouillons_supprimes += 1;
                continue;
            }
            match lire_hook(&get) {
                Ok(h) if index.hooks.contains_key(&h.id) => rapport.en_quarantaine.push(format!(
                    "nœud LifecycleHook {noeud:?} : id {} déjà pris",
                    h.id
                )),
                Ok(h) => {
                    index.hooks.insert(h.id, Entree { noeud, valeur: h });
                }
                Err(e) => rapport
                    .en_quarantaine
                    .push(format!("nœud LifecycleHook {noeud:?} : {e}")),
            }
        }

        Ok((index, rapport))
    }
}

// ============================================================================
// Decoding: the counterparts of `parse_alert_row`, `node_to_event_trigger` and
// `parse_lifecycle_hook_node`, field for field.
// ============================================================================

fn texte(get: Lecteur<'_>, cle: &str) -> Option<String> {
    get(cle).and_then(|v| v.as_str().map(str::to_string))
}

/// A text property, where `""` is the graph's "none".
fn texte_non_vide(get: Lecteur<'_>, cle: &str) -> Option<String> {
    texte(get, cle).filter(|s| !s.is_empty())
}

fn instant(get: Lecteur<'_>, cle: &str) -> Option<DateTime<Utc>> {
    texte_non_vide(get, cle)
        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
        .map(|d| d.with_timezone(&Utc))
}

fn requis(get: Lecteur<'_>, cle: &str) -> Result<String> {
    texte(get, cle).ok_or_else(|| anyhow!("propriété `{cle}` absente ou non textuelle"))
}

fn lire_alerte(get: Lecteur<'_>) -> Result<AlertNode> {
    let id: Uuid = requis(get, "id")?.parse()?;
    let severity: AlertSeverity = requis(get, "severity")?
        .parse()
        .map_err(|e: String| anyhow!(e))?;
    // A node without a readable creation date reads as "now", as the Neo4j
    // adapter does.
    let created_at = instant(get, "created_at").unwrap_or_else(Utc::now);
    let acknowledged = get("acknowledged")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let occurrence_count = u64::try_from(
        get("occurrence_count")
            .and_then(|v| v.as_int64())
            .unwrap_or(1)
            .max(0),
    )
    .unwrap_or(u64::MAX)
    .max(1);
    let last_seen = instant(get, "last_seen").or(Some(created_at));
    let priority = get("priority")
        .and_then(|v| v.as_float64())
        .unwrap_or_else(|| {
            AlertNode::compute_priority(
                severity,
                occurrence_count,
                last_seen.unwrap_or(created_at),
                acknowledged,
                Utc::now(),
            )
        });
    Ok(AlertNode {
        id,
        alert_type: requis(get, "alert_type")?,
        severity,
        message: requis(get, "message")?,
        project_id: texte_non_vide(get, "project_id").and_then(|s| s.parse().ok()),
        acknowledged,
        acknowledged_by: texte_non_vide(get, "acknowledged_by"),
        acknowledged_at: instant(get, "acknowledged_at"),
        created_at,
        dedup_key: texte(get, "dedup_key").unwrap_or_default(),
        occurrence_count,
        first_seen: instant(get, "first_seen").or(Some(created_at)),
        last_seen,
        priority,
    })
}

fn lire_declencheur(get: Lecteur<'_>) -> Result<EventTrigger> {
    Ok(EventTrigger {
        id: requis(get, "id")?.parse::<Uuid>()?,
        name: requis(get, "name")?,
        protocol_id: requis(get, "protocol_id")?.parse::<Uuid>()?,
        entity_type_pattern: texte_non_vide(get, "entity_type_pattern"),
        action_pattern: texte_non_vide(get, "action_pattern"),
        payload_conditions: texte_non_vide(get, "payload_conditions")
            .and_then(|s| serde_json::from_str(&s).ok()),
        cooldown_secs: get("cooldown_secs").and_then(|v| v.as_int64()).unwrap_or(0) as u32,
        enabled: get("enabled").and_then(|v| v.as_bool()).unwrap_or(true),
        project_scope: texte_non_vide(get, "project_scope").and_then(|s| s.parse().ok()),
        created_at: requis(get, "created_at")?.parse::<DateTime<Utc>>()?,
        updated_at: requis(get, "updated_at")?.parse::<DateTime<Utc>>()?,
    })
}

/// The `Debug` name of an enum variant, in snake case, for serde (as the Neo4j
/// adapter reads `scope` and `action_type`).
fn snake_depuis_debug(s: &str) -> String {
    let mut resultat = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            resultat.push('_');
        }
        resultat.push(ch.to_ascii_lowercase());
    }
    resultat
}

fn lire_hook(get: Lecteur<'_>) -> Result<LifecycleHook> {
    let scope: LifecycleScope = serde_json::from_str(&format!(
        "\"{}\"",
        snake_depuis_debug(&requis(get, "scope")?)
    ))
    .unwrap_or(LifecycleScope::Task);
    let action_type: LifecycleActionType = serde_json::from_str(&format!(
        "\"{}\"",
        snake_depuis_debug(&requis(get, "action_type")?)
    ))
    .unwrap_or(LifecycleActionType::EmitAlert);
    let action_config = texte(get, "action_config")
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(serde_json::Value::Object(Default::default()));
    Ok(LifecycleHook {
        id: requis(get, "id")?.parse::<Uuid>()?,
        name: requis(get, "name")?,
        description: texte_non_vide(get, "description"),
        scope,
        on_status: requis(get, "on_status")?,
        action_type,
        action_config,
        priority: get("priority")
            .and_then(|v| v.as_int64())
            .ok_or_else(|| anyhow!("propriété `priority` absente"))? as i32,
        enabled: get("enabled").and_then(|v| v.as_bool()).unwrap_or(true),
        builtin: get("builtin").and_then(|v| v.as_bool()).unwrap_or(false),
        project_id: texte_non_vide(get, "project_id").and_then(|s| s.parse().ok()),
        created_at: texte(get, "created_at")
            .and_then(|s| s.parse::<DateTime<Utc>>().ok())
            .unwrap_or_else(Utc::now),
        // Stored as a string, so it reads back. The Neo4j adapter writes a
        // datetime and reads it as a string, which never matches: its read-back
        // is `None`. Deliberate divergence, see the module doc.
        updated_at: texte(get, "updated_at").and_then(|s| s.parse::<DateTime<Utc>>().ok()),
    })
}

/// Read a node once and decode it with `decode` (the write path re-reads what it
/// just wrote, before publishing it).
fn relire<T>(db: &ObrainDB, noeud: NodeId, decode: fn(Lecteur<'_>) -> Result<T>) -> Result<T> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud {noeud:?} indexé mais absent de la base"))?;
    let get = move |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    };
    decode(&get)
}

// ============================================================================
// Writes. Each one validates before it writes (obrain has no rollback), writes
// under the writer lock, makes the write durable, and only then publishes it.
// ============================================================================

/// Create a node: the draft marker first, the properties, the marker removed
/// LAST (the commit point). A blob-sized value forces the full flush before the
/// marker goes (`Ecriture::rendre_durable_avant_le_marqueur`).
fn creer_noeud(
    w: &mut Ecriture<'_>,
    etiquette: &str,
    proprietes: &[(&'static str, Value)],
) -> Result<NodeId> {
    w.flush_complet = proprietes.iter().any(|(_, v)| en_blob(v));
    let noeud = w.db.create_node(&[etiquette]);
    w.db.set_node_property(noeud, BROUILLON, Value::from(true));
    for (cle, valeur) in proprietes {
        w.db.set_node_property(noeud, *cle, valeur.clone());
    }
    w.rendre_durable_avant_le_marqueur()?;
    w.db.remove_node_property(noeud, BROUILLON);
    Ok(noeud)
}

/// Overwrite properties of an existing node. Replacing a value that WAS in a
/// blob column clears that slot, which the WAL does not recover either: the old
/// value counts too (as in `project.rs`).
fn ecrire_proprietes(
    w: &mut Ecriture<'_>,
    noeud: NodeId,
    proprietes: &[(&'static str, Value)],
) -> Result<()> {
    let ancien_en_blob = w.db.get_node(noeud).is_some_and(|n| {
        proprietes
            .iter()
            .any(|(cle, _)| n.get_property(*cle).is_some_and(en_blob))
    });
    w.flush_complet = ancien_en_blob || proprietes.iter().any(|(_, v)| en_blob(v));
    for (cle, valeur) in proprietes {
        w.db.set_node_property(noeud, *cle, valeur.clone());
    }
    Ok(())
}

/// The priority the Cypher of `create_alert` computes.
fn priorite_alerte(severite: &AlertSeverity, compte: u64, acquittee: bool) -> f64 {
    if acquittee {
        return 0.0;
    }
    let chronicite = ((compte as f64).ln() / 10.0).min(0.5);
    severite.weight() * (1.0 + chronicite)
}

/// `ORDER BY coalesce(priority, 0) DESC, coalesce(last_seen, created_at) DESC, id DESC`.
fn ordre_alertes(a: &AlertNode, b: &AlertNode) -> Ordering {
    b.priority
        .total_cmp(&a.priority)
        .then_with(|| {
            b.last_seen
                .unwrap_or(b.created_at)
                .cmp(&a.last_seen.unwrap_or(a.created_at))
        })
        .then_with(|| b.id.cmp(&a.id))
}

fn garde_id_libre(existe: bool, label: &str, id: Uuid) -> Result<()> {
    if existe {
        bail!("un {label} d'id {id} existe déjà (unicité de l'id de {label})");
    }
    Ok(())
}

#[async_trait]
impl TriggerStore for ObrainGraphStore {
    // ------------------------------------------------------------------------
    // Alerts
    // ------------------------------------------------------------------------

    async fn create_alert(&self, alert: &AlertNode) -> Result<()> {
        let a = alert.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let maintenant = Utc::now().to_rfc3339();
                let cle = if a.dedup_key.is_empty() {
                    AlertNode::make_dedup_key(&a.alert_type, a.project_id, &a.message)
                } else {
                    a.dedup_key.clone()
                };
                // The MERGE on `dedup_key`: an existing condition is collapsed onto
                // its node, a new one is created. One writer, so no race.
                let existante: Option<(NodeId, AlertNode)> = {
                    let etat = w.index()?;
                    let idx = &etat.declencheurs;
                    idx.alertes_par_cle
                        .get(&cle)
                        .and_then(|id| idx.alertes.get(id))
                        .map(|e| (e.noeud, e.valeur.clone()))
                };
                match existante {
                    Some((noeud, avant)) => {
                        let compte = avant.occurrence_count + 1;
                        let priorite = priorite_alerte(&a.severity, compte, avant.acknowledged);
                        ecrire_proprietes(
                            w,
                            noeud,
                            &[
                                ("severity", Value::from(a.severity.to_string())),
                                ("message", Value::from(a.message.clone())),
                                ("last_seen", Value::from(maintenant)),
                                ("occurrence_count", Value::from(compte as i64)),
                                ("priority", Value::from(priorite)),
                            ],
                        )?;
                        let valeur = w.apres_ecriture(relire(w.db, noeud, lire_alerte))?;
                        w.rendre_durable()?;
                        let id = valeur.id;
                        w.index_mut()?
                            .declencheurs
                            .alertes
                            .insert(id, Entree { noeud, valeur });
                    }
                    None => {
                        let priorite = priorite_alerte(&a.severity, 1, false);
                        let noeud = creer_noeud(
                            w,
                            LABEL_ALERTE,
                            &[
                                ("id", Value::from(a.id.to_string())),
                                ("alert_type", Value::from(a.alert_type.clone())),
                                ("severity", Value::from(a.severity.to_string())),
                                ("message", Value::from(a.message.clone())),
                                (
                                    "project_id",
                                    Value::from(
                                        a.project_id.map(|p| p.to_string()).unwrap_or_default(),
                                    ),
                                ),
                                ("acknowledged", Value::from(false)),
                                ("created_at", Value::from(maintenant.clone())),
                                ("first_seen", Value::from(maintenant.clone())),
                                ("last_seen", Value::from(maintenant)),
                                ("occurrence_count", Value::from(1_i64)),
                                ("dedup_key", Value::from(cle.clone())),
                                ("priority", Value::from(priorite)),
                            ],
                        )?;
                        let valeur = w.apres_ecriture(relire(w.db, noeud, lire_alerte))?;
                        w.rendre_durable()?;
                        let id = valeur.id;
                        let mut etat = w.index_mut()?;
                        etat.declencheurs.alertes_par_cle.insert(cle, id);
                        etat.declencheurs
                            .alertes
                            .insert(id, Entree { noeud, valeur });
                    }
                }
                Ok(())
            },
        )
        .await
    }

    async fn list_pending_alerts(
        &self,
        project_id: Option<Uuid>,
        limit: usize,
    ) -> Result<Vec<AlertNode>> {
        self.lire(|_, etat| {
            let mut alertes: Vec<AlertNode> = etat
                .declencheurs
                .alertes
                .values()
                .map(|e| &e.valeur)
                .filter(|a| !a.acknowledged)
                .filter(|a| project_id.is_none_or(|p| a.project_id == Some(p)))
                .cloned()
                .collect();
            alertes.sort_by(ordre_alertes);
            alertes.truncate(limit);
            Ok(alertes)
        })
    }

    async fn acknowledge_alert(&self, alert_id: Uuid, acknowledged_by: &str) -> Result<()> {
        let par = acknowledged_by.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // An unknown alert is a silent no-op, as the `MATCH ... SET` is.
                let noeud = w
                    .index()?
                    .declencheurs
                    .alertes
                    .get(&alert_id)
                    .map(|e| e.noeud);
                let Some(noeud) = noeud else {
                    return Ok(());
                };
                ecrire_proprietes(
                    w,
                    noeud,
                    &[
                        ("acknowledged", Value::from(true)),
                        ("acknowledged_by", Value::from(par)),
                        ("acknowledged_at", Value::from(Utc::now().to_rfc3339())),
                    ],
                )?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, lire_alerte))?;
                w.rendre_durable()?;
                w.index_mut()?
                    .declencheurs
                    .alertes
                    .insert(alert_id, Entree { noeud, valeur });
                Ok(())
            },
        )
        .await
    }

    async fn get_alert(&self, alert_id: Uuid) -> Result<Option<AlertNode>> {
        self.lire(|_, etat| {
            Ok(etat
                .declencheurs
                .alertes
                .get(&alert_id)
                .map(|e| e.valeur.clone()))
        })
    }

    async fn list_alerts(
        &self,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<AlertNode>, usize)> {
        // `project_id` wins over `workspace_slug`, as in the Neo4j adapter. The
        // workspace filter walks the workspace domain, which is not ported here.
        if project_id.is_none() && workspace_slug.is_some() {
            return Err(NonPorte(
                "TriggerStore::list_alerts (filtre par workspace_slug : le domaine Workspace n'est pas porté)",
            )
            .into());
        }
        self.lire(|_, etat| {
            let mut alertes: Vec<AlertNode> = etat
                .declencheurs
                .alertes
                .values()
                .map(|e| &e.valeur)
                .filter(|a| project_id.is_none_or(|p| a.project_id == Some(p)))
                .cloned()
                .collect();
            alertes.sort_by(ordre_alertes);
            let total = alertes.len();
            let page: Vec<AlertNode> = alertes.into_iter().skip(offset).take(limit).collect();
            Ok((page, total))
        })
    }

    // ------------------------------------------------------------------------
    // EventTriggers
    // ------------------------------------------------------------------------

    async fn list_event_triggers(
        &self,
        project_scope: Option<Uuid>,
        enabled_only: bool,
    ) -> Result<Vec<EventTrigger>> {
        self.lire(|_, etat| {
            let mut triggers: Vec<EventTrigger> = etat
                .declencheurs
                .declencheurs
                .values()
                .map(|e| &e.valeur)
                .filter(|t| {
                    // A scope lists its own triggers AND the global ones (no scope).
                    project_scope
                        .is_none_or(|s| t.project_scope.is_none() || t.project_scope == Some(s))
                })
                .filter(|t| !enabled_only || t.enabled)
                .cloned()
                .collect();
            // ORDER BY created_at DESC, id DESC
            triggers.sort_by(|a, b| {
                b.created_at
                    .cmp(&a.created_at)
                    .then_with(|| b.id.cmp(&a.id))
            });
            Ok(triggers)
        })
    }

    async fn create_event_trigger(&self, trigger: &EventTrigger) -> Result<Uuid> {
        let t = trigger.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                garde_id_libre(
                    w.index()?.declencheurs.declencheurs.contains_key(&t.id),
                    "déclencheur",
                    t.id,
                )?;
                // The timestamps are the write's, as in the Neo4j `CREATE`.
                let maintenant = Utc::now().to_rfc3339();
                let noeud = creer_noeud(
                    w,
                    LABEL_DECLENCHEUR,
                    &[
                        ("id", Value::from(t.id.to_string())),
                        ("name", Value::from(t.name.clone())),
                        ("protocol_id", Value::from(t.protocol_id.to_string())),
                        (
                            "entity_type_pattern",
                            Value::from(t.entity_type_pattern.clone().unwrap_or_default()),
                        ),
                        (
                            "action_pattern",
                            Value::from(t.action_pattern.clone().unwrap_or_default()),
                        ),
                        (
                            "payload_conditions",
                            Value::from(
                                t.payload_conditions
                                    .as_ref()
                                    .map(|v| serde_json::to_string(v).unwrap_or_default())
                                    .unwrap_or_default(),
                            ),
                        ),
                        ("cooldown_secs", Value::from(i64::from(t.cooldown_secs))),
                        ("enabled", Value::from(t.enabled)),
                        (
                            "project_scope",
                            Value::from(t.project_scope.map(|u| u.to_string()).unwrap_or_default()),
                        ),
                        ("created_at", Value::from(maintenant.clone())),
                        ("updated_at", Value::from(maintenant)),
                    ],
                )?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, lire_declencheur))?;
                w.rendre_durable()?;
                w.index_mut()?
                    .declencheurs
                    .declencheurs
                    .insert(t.id, Entree { noeud, valeur });
                Ok(t.id)
            },
        )
        .await
    }

    async fn get_event_trigger(&self, id: Uuid) -> Result<Option<EventTrigger>> {
        self.lire(|_, etat| {
            Ok(etat
                .declencheurs
                .declencheurs
                .get(&id)
                .map(|e| e.valeur.clone()))
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn update_event_trigger(
        &self,
        id: Uuid,
        enabled: Option<bool>,
        name: Option<String>,
        entity_type_pattern: Option<Option<String>>,
        action_pattern: Option<Option<String>>,
        payload_conditions: Option<Option<serde_json::Value>>,
        cooldown_secs: Option<u32>,
        project_scope: Option<Option<Uuid>>,
    ) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud = w
                    .index()?
                    .declencheurs
                    .declencheurs
                    .get(&id)
                    .map(|e| e.noeud);
                let Some(noeud) = noeud else {
                    return Ok(false);
                };
                let mut proprietes: Vec<(&'static str, Value)> =
                    vec![("updated_at", Value::from(Utc::now().to_rfc3339()))];
                if let Some(e) = enabled {
                    proprietes.push(("enabled", Value::from(e)));
                }
                if let Some(n) = name {
                    proprietes.push(("name", Value::from(n)));
                }
                if let Some(etp) = entity_type_pattern {
                    proprietes.push(("entity_type_pattern", Value::from(etp.unwrap_or_default())));
                }
                if let Some(ap) = action_pattern {
                    proprietes.push(("action_pattern", Value::from(ap.unwrap_or_default())));
                }
                if let Some(pc) = payload_conditions {
                    proprietes.push((
                        "payload_conditions",
                        Value::from(
                            pc.map(|v| serde_json::to_string(&v).unwrap_or_default())
                                .unwrap_or_default(),
                        ),
                    ));
                }
                if let Some(cd) = cooldown_secs {
                    proprietes.push(("cooldown_secs", Value::from(i64::from(cd))));
                }
                if let Some(ps) = project_scope {
                    proprietes.push((
                        "project_scope",
                        Value::from(ps.map(|u| u.to_string()).unwrap_or_default()),
                    ));
                }
                ecrire_proprietes(w, noeud, &proprietes)?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, lire_declencheur))?;
                w.rendre_durable()?;
                w.index_mut()?
                    .declencheurs
                    .declencheurs
                    .insert(id, Entree { noeud, valeur });
                Ok(true)
            },
        )
        .await
    }

    async fn delete_event_trigger(&self, id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud = w
                    .index()?
                    .declencheurs
                    .declencheurs
                    .get(&id)
                    .map(|e| e.noeud);
                let Some(noeud) = noeud else {
                    return Ok(false);
                };
                if !w.db.delete_node(noeud) {
                    bail!("obrain a refusé la suppression du nœud {noeud:?} (déclencheur {id})");
                }
                w.rendre_durable()?;
                w.index_mut()?.declencheurs.declencheurs.remove(&id);
                Ok(true)
            },
        )
        .await
    }

    // ------------------------------------------------------------------------
    // LifecycleHooks
    // ------------------------------------------------------------------------

    async fn create_lifecycle_hook(&self, hook: &LifecycleHook) -> Result<()> {
        let h = hook.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                garde_id_libre(
                    w.index()?.declencheurs.hooks.contains_key(&h.id),
                    "hook de cycle de vie",
                    h.id,
                )?;
                // As the Neo4j `CREATE`: `updated_at` starts at `created_at`, and the
                // `updated_at` of the value passed in is not written.
                let cree = h.created_at.to_rfc3339();
                let noeud = creer_noeud(
                    w,
                    LABEL_HOOK,
                    &[
                        ("id", Value::from(h.id.to_string())),
                        ("name", Value::from(h.name.clone())),
                        (
                            "description",
                            Value::from(h.description.clone().unwrap_or_default()),
                        ),
                        ("scope", Value::from(format!("{:?}", h.scope))),
                        ("on_status", Value::from(h.on_status.clone())),
                        ("action_type", Value::from(format!("{:?}", h.action_type))),
                        (
                            "action_config",
                            Value::from(serde_json::to_string(&h.action_config)?),
                        ),
                        ("priority", Value::from(i64::from(h.priority))),
                        ("enabled", Value::from(h.enabled)),
                        ("builtin", Value::from(h.builtin)),
                        (
                            "project_id",
                            Value::from(h.project_id.map(|p| p.to_string()).unwrap_or_default()),
                        ),
                        ("created_at", Value::from(cree.clone())),
                        ("updated_at", Value::from(cree)),
                    ],
                )?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, lire_hook))?;
                w.rendre_durable()?;
                w.index_mut()?
                    .declencheurs
                    .hooks
                    .insert(h.id, Entree { noeud, valeur });
                Ok(())
            },
        )
        .await
    }

    async fn get_lifecycle_hook(&self, id: Uuid) -> Result<Option<LifecycleHook>> {
        self.lire(|_, etat| Ok(etat.declencheurs.hooks.get(&id).map(|e| e.valeur.clone())))
    }

    async fn list_lifecycle_hooks(&self, project_id: Option<Uuid>) -> Result<Vec<LifecycleHook>> {
        self.lire(|_, etat| {
            let mut hooks: Vec<LifecycleHook> = etat
                .declencheurs
                .hooks
                .values()
                .map(|e| &e.valeur)
                .filter(|h| {
                    project_id.is_none_or(|p| h.project_id.is_none() || h.project_id == Some(p))
                })
                .cloned()
                .collect();
            // ORDER BY priority ASC, id ASC
            hooks.sort_by(|a, b| a.priority.cmp(&b.priority).then_with(|| a.id.cmp(&b.id)));
            Ok(hooks)
        })
    }

    async fn update_lifecycle_hook(
        &self,
        id: Uuid,
        updates: &UpdateLifecycleHookRequest,
    ) -> Result<()> {
        let u = (
            updates.name.clone(),
            updates.description.clone(),
            updates.on_status.clone(),
            updates.action_config.clone(),
            updates.priority,
            updates.enabled,
        );
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // An unknown hook is a silent no-op, as the `MATCH ... SET` is.
                let noeud = w.index()?.declencheurs.hooks.get(&id).map(|e| e.noeud);
                let Some(noeud) = noeud else {
                    return Ok(());
                };
                let (name, description, on_status, action_config, priority, enabled) = u;
                let mut proprietes: Vec<(&'static str, Value)> =
                    vec![("updated_at", Value::from(Utc::now().to_rfc3339()))];
                if let Some(n) = name {
                    proprietes.push(("name", Value::from(n)));
                }
                if let Some(d) = description {
                    proprietes.push(("description", Value::from(d.unwrap_or_default())));
                }
                if let Some(s) = on_status {
                    proprietes.push(("on_status", Value::from(s)));
                }
                if let Some(c) = action_config {
                    proprietes.push(("action_config", Value::from(serde_json::to_string(&c)?)));
                }
                if let Some(p) = priority {
                    proprietes.push(("priority", Value::from(i64::from(p))));
                }
                if let Some(e) = enabled {
                    proprietes.push(("enabled", Value::from(e)));
                }
                ecrire_proprietes(w, noeud, &proprietes)?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, lire_hook))?;
                w.rendre_durable()?;
                w.index_mut()?
                    .declencheurs
                    .hooks
                    .insert(id, Entree { noeud, valeur });
                Ok(())
            },
        )
        .await
    }

    async fn delete_lifecycle_hook(&self, id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let cible = w
                    .index()?
                    .declencheurs
                    .hooks
                    .get(&id)
                    .map(|e| (e.noeud, e.valeur.builtin));
                let Some((noeud, builtin)) = cible else {
                    return Ok(());
                };
                if builtin {
                    bail!("Cannot delete builtin lifecycle hook");
                }
                if !w.db.delete_node(noeud) {
                    bail!("obrain a refusé la suppression du nœud {noeud:?} (hook {id})");
                }
                w.rendre_durable()?;
                w.index_mut()?.declencheurs.hooks.remove(&id);
                Ok(())
            },
        )
        .await
    }

    async fn list_hooks_for_scope(
        &self,
        scope: &LifecycleScope,
        on_status: &str,
        project_id: Option<Uuid>,
    ) -> Result<Vec<LifecycleHook>> {
        let scope = scope.clone();
        let on_status = on_status.to_string();
        self.lire(|_, etat| {
            let mut hooks: Vec<LifecycleHook> = etat
                .declencheurs
                .hooks
                .values()
                .map(|e| &e.valeur)
                .filter(|h| h.scope == scope && h.on_status == on_status && h.enabled)
                .filter(|h| match project_id {
                    // Project hooks AND global ones (no project).
                    Some(p) => h.project_id.is_none() || h.project_id == Some(p),
                    None => h.project_id.is_none(),
                })
                .cloned()
                .collect();
            hooks.sort_by(|a, b| a.priority.cmp(&b.priority).then_with(|| a.id.cmp(&b.id)));
            Ok(hooks)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alerte(cle: &str, severite: AlertSeverity, projet: Option<Uuid>) -> AlertNode {
        AlertNode::new_for_subject(
            "git_drift".to_string(),
            severite,
            format!("message {cle}"),
            projet,
            cle,
        )
    }

    #[tokio::test]
    async fn une_meme_condition_se_fusionne_et_son_priorite_suit_le_cypher() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let a = alerte("sujet", AlertSeverity::Critical, None);
        store.create_alert(&a).await.unwrap();
        store.create_alert(&a).await.unwrap();
        let (liste, total) = store.list_alerts(None, None, 10, 0).await.unwrap();
        assert_eq!(total, 1, "une seule condition, un seul nœud");
        assert_eq!(liste[0].occurrence_count, 2);
        let attendu = 1.0 * (1.0 + (2f64.ln() / 10.0));
        assert!((liste[0].priority - attendu).abs() < 1e-12);
    }

    #[tokio::test]
    async fn un_acquittement_sort_de_la_liste_en_attente() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let a = alerte("x", AlertSeverity::Warning, None);
        store.create_alert(&a).await.unwrap();
        assert_eq!(store.list_pending_alerts(None, 10).await.unwrap().len(), 1);
        store.acknowledge_alert(a.id, "moi").await.unwrap();
        assert!(store
            .list_pending_alerts(None, 10)
            .await
            .unwrap()
            .is_empty());
        let relue = store.get_alert(a.id).await.unwrap().unwrap();
        assert!(relue.acknowledged);
        assert_eq!(relue.acknowledged_by.as_deref(), Some("moi"));
    }

    #[tokio::test]
    async fn filtre_par_workspace_est_refuse_par_son_nom() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let err = store
            .list_alerts(None, Some("ws"), 10, 0)
            .await
            .expect_err("refus attendu");
        assert!(
            err.downcast_ref::<NonPorte>().is_some(),
            "un refus nommé, pas un résultat vide : {err:#}"
        );
    }

    #[tokio::test]
    async fn declencheurs_portee_et_ordre() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let projet = Uuid::new_v4();
        let mut t = EventTrigger {
            id: Uuid::new_v4(),
            name: "t1".into(),
            protocol_id: Uuid::new_v4(),
            entity_type_pattern: Some("project".into()),
            action_pattern: None,
            payload_conditions: Some(serde_json::json!({"k": "v"})),
            cooldown_secs: 5,
            enabled: true,
            project_scope: Some(projet),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        store.create_event_trigger(&t).await.unwrap();
        t.id = Uuid::new_v4();
        t.name = "global".into();
        t.project_scope = None;
        store.create_event_trigger(&t).await.unwrap();
        // Scoped to `projet`: its own trigger and the global one.
        assert_eq!(
            store
                .list_event_triggers(Some(projet), false)
                .await
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            store
                .list_event_triggers(Some(Uuid::new_v4()), false)
                .await
                .unwrap()
                .len(),
            1,
            "un autre projet ne voit que le global"
        );
        let relu = store.get_event_trigger(t.id).await.unwrap().unwrap();
        assert_eq!(relu.cooldown_secs, 5);
        assert!(store
            .update_event_trigger(t.id, Some(false), None, None, None, None, None, None)
            .await
            .unwrap());
        assert_eq!(
            store.list_event_triggers(None, true).await.unwrap().len(),
            1,
            "le déclencheur désactivé sort, le premier reste actif"
        );
        assert!(store.delete_event_trigger(t.id).await.unwrap());
        assert!(!store.delete_event_trigger(t.id).await.unwrap());
    }

    #[tokio::test]
    async fn hook_builtin_non_supprimable_et_ordre_par_priorite() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let mut builtin = LifecycleHook::new(
            "b".into(),
            LifecycleScope::Task,
            "done".into(),
            LifecycleActionType::EmitAlert,
            serde_json::json!({}),
        );
        builtin.builtin = true;
        builtin.priority = 50;
        store.create_lifecycle_hook(&builtin).await.unwrap();
        let mut autre = LifecycleHook::new(
            "a".into(),
            LifecycleScope::Task,
            "done".into(),
            LifecycleActionType::EmitAlert,
            serde_json::json!({}),
        );
        autre.priority = 10;
        store.create_lifecycle_hook(&autre).await.unwrap();

        let ordre = store
            .list_hooks_for_scope(&LifecycleScope::Task, "done", None)
            .await
            .unwrap();
        assert_eq!(
            ordre.iter().map(|h| h.priority).collect::<Vec<_>>(),
            [10, 50]
        );
        assert!(store.delete_lifecycle_hook(builtin.id).await.is_err());
        store.delete_lifecycle_hook(autre.id).await.unwrap();
        assert!(store.get_lifecycle_hook(autre.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn les_valeurs_survivent_a_la_reouverture() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let a = alerte("durable", AlertSeverity::Info, None);
        {
            let store = tokio::task::spawn_blocking({
                let chemin = chemin.clone();
                move || ObrainGraphStore::open(&chemin, super::super::Creation::Permise)
            })
            .await
            .unwrap()
            .unwrap();
            store.create_alert(&a).await.unwrap();
            store.close().await.unwrap();
        }
        let store = tokio::task::spawn_blocking(move || {
            ObrainGraphStore::open(&chemin, super::super::Creation::Interdite)
        })
        .await
        .unwrap()
        .unwrap();
        let relue = store.get_alert(a.id).await.unwrap().expect("alerte relue");
        assert_eq!(relue.message, a.message);
        assert_eq!(relue.dedup_key, a.dedup_key);
    }
}
