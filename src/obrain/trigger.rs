//! `TriggerStore` on obrain: the alerts of the heartbeat, the event triggers and
//! the lifecycle hooks (lot O-TRIGGER). The plan triggers, their firings and the
//! plan runs are `PlanRunStore`'s (`planrun.rs`).
//!
//! PORTED here, with the Neo4j semantics of `src/neo4j/alert.rs`,
//! `src/neo4j/event_trigger.rs` and `src/neo4j/lifecycle_hook.rs`: ALL SIXTEEN
//! methods of the trait. Nothing of it is refused — none of them reads or links
//! a domain that is not ported (an alert's `project_id`, an event trigger's
//! `protocol_id` and `project_scope`, a hook's `project_id` are properties that
//! Neo4j does not check either; only `list_alerts` resolves a workspace, which is
//! ported).
//!
//! # Layout
//!
//! One node per entity, labelled `Alert`, `EventTrigger`, `LifecycleHook`, with
//! the properties Neo4j writes. A missing optional string is the empty string,
//! as in Neo4j; `payload_conditions` and `action_config` are JSON strings; a
//! hook's `scope` and `action_type` are the Rust variant names (`Task`,
//! `EmitAlert`), as Neo4j writes them (`format!("{:?}")`). Dates are RFC 3339
//! strings (Neo4j stores a hook's dates as `datetime`: same instant).
//! The `HAS_ALERT` edge Neo4j adds from an existing project is not written: no
//! reader of either backend traverses it (the alert's `project_id` answers).
//!
//! # Divergences, all on the strict side
//!
//! - A creation whose `id` is already taken (by an alert of ANOTHER `dedup_key`,
//!   an event trigger, a hook) is REFUSED. Neo4j `CREATE`s a second node with
//!   the same id, which `get_*` then reads at random.
//! - A stored value that does not decode — a severity, a hook scope or action
//!   type, a `payload_conditions` or `action_config` that is not JSON, a date —
//!   puts the node in QUARANTINE at open (named in the report, untouched, never
//!   served). Neo4j reads it as a default (`Task`, `EmitAlert`, `{}`, `None`,
//!   "now") or, in a list, skips it in silence.
//! - The keys of a quarantined node (its `id`, an alert's `dedup_key`) stay
//!   RESERVED: a creation that would take one is refused by name. Otherwise the
//!   condition of an unreadable alert would start again at one occurrence, its
//!   acknowledgement lost, and an id would end up on two nodes.
//! - A missing `created_at` (alert, hook) is a quarantine, where Neo4j reads
//!   "now". A missing `enabled` (event trigger, hook) reads as NOT enabled:
//!   Neo4j reads it as `true` but its filters (`enabled = true`) exclude the
//!   node, so the hooks and triggers it would run are the ones obrain runs. The
//!   other missing values read as Neo4j reads them (an alert's
//!   `first_seen`/`last_seen` as its `created_at`, `occurrence_count` as 1, a
//!   missing `priority` recomputed for the reader but SORTED as 0, as
//!   `coalesce(priority, 0)` does; `builtin` false).
//! - A hook's `created_at` / `updated_at` are stored as strings and read back as
//!   written (`updated_at` = the creation instant after a creation, as Neo4j
//!   sets it). Neo4j writes them as `datetime` and reads them as strings, which
//!   never matches: its read-back is "now" and `None` (seen first by an earlier,
//!   uncommitted port of this domain, `sauvegarde/trigger-session-af008d00`).
//! - `list_alerts` / `list_pending_alerts` order the parsed dates; Neo4j compares
//!   the raw RFC 3339 strings (same order for the fixed-width strings both write).
//!
//! # Atomicity
//!
//! A creation follows the draft-marker protocol of `project.rs`: the marker is
//! set first and removed last, opening deletes what still carries it. An update
//! rewrites properties of a live node (one record each, as everywhere in this
//! backend: an interrupted update can leave some of them new). Two creations of
//! one `dedup_key` are serialized by the writer lock: of N concurrent ones, one
//! creates the node and the others count on it. Reads come from the in-memory
//! index, which a write publishes only once durable.
//!
//! The helpers of the first half of this file (decoding a node, the draft
//! protocol) are shared with `planrun.rs`.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, SecondsFormat, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::{Ecriture, ObrainGraphStore, RapportOuverture, SEUIL_COLONNE_BLOB};
use crate::events::trigger::EventTrigger;
use crate::lifecycle::{
    LifecycleActionType, LifecycleHook, LifecycleScope, UpdateLifecycleHookRequest,
};
use crate::neo4j::models::{AlertNode, AlertSeverity};
use crate::neo4j::traits::TriggerStore;

const ALERTE: &str = "Alert";
const DECLENCHEUR: &str = "EventTrigger";
const CROCHET: &str = "LifecycleHook";
/// The draft marker: present from the first record of a creation to the last.
pub(super) const BROUILLON: &str = "_brouillon";

// ============================================================================
// Shared with `planrun.rs`: decoding and writing one node
// ============================================================================

/// A property lookup on one node (nulls read as absent).
pub(super) type Lecteur<'a> = &'a dyn Fn(&str) -> Option<Value>;

/// Where an entity lives in the base, and its decoded value.
pub(super) struct Fiche<T> {
    pub(super) noeud: NodeId,
    pub(super) valeur: T,
}

pub(super) fn texte(s: impl Into<String>) -> Value {
    Value::from(s.into())
}

/// An instant as this backend writes it: RFC 3339, nanoseconds, `Z` (fixed
/// width, so the string order is the time order).
pub(super) fn instant(d: DateTime<Utc>) -> Value {
    texte(d.to_rfc3339_opts(SecondsFormat::Nanos, true))
}

pub(super) fn uuid_ou_vide(u: Option<Uuid>) -> Value {
    texte(u.map(|u| u.to_string()).unwrap_or_default())
}

/// A string property. Present with another type: an error (never read as absent).
pub(super) fn chaine(get: Lecteur, cle: &str, quoi: &str) -> Result<Option<String>> {
    match get(cle) {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_string()))
            .ok_or_else(|| anyhow!("{quoi} : `{cle}` n'est pas une chaîne ({v:?})")),
    }
}

/// A required string property (the empty string is a value).
pub(super) fn requise(get: Lecteur, cle: &str, quoi: &str) -> Result<String> {
    chaine(get, cle, quoi)?.ok_or_else(|| anyhow!("{quoi} sans propriété `{cle}`"))
}

/// A string property that may be absent or empty (both: `None`).
pub(super) fn facultative(get: Lecteur, cle: &str, quoi: &str) -> Result<Option<String>> {
    Ok(chaine(get, cle, quoi)?.filter(|s| !s.is_empty()))
}

pub(super) fn analyser<T: std::str::FromStr>(brut: &str, cle: &str, quoi: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    brut.parse()
        .map_err(|e| anyhow!("{quoi} : `{cle}` illisible (« {brut} ») : {e}"))
}

pub(super) fn uuid_requis(get: Lecteur, cle: &str, quoi: &str) -> Result<Uuid> {
    analyser(&requise(get, cle, quoi)?, cle, quoi)
}

pub(super) fn uuid_facultatif(get: Lecteur, cle: &str, quoi: &str) -> Result<Option<Uuid>> {
    facultative(get, cle, quoi)?
        .map(|b| analyser(&b, cle, quoi))
        .transpose()
}

pub(super) fn date_requise(get: Lecteur, cle: &str, quoi: &str) -> Result<DateTime<Utc>> {
    analyser(&requise(get, cle, quoi)?, cle, quoi)
}

pub(super) fn date_facultative(
    get: Lecteur,
    cle: &str,
    quoi: &str,
) -> Result<Option<DateTime<Utc>>> {
    facultative(get, cle, quoi)?
        .map(|b| analyser(&b, cle, quoi))
        .transpose()
}

/// An integer property; present with another type: an error.
pub(super) fn entier(get: Lecteur, cle: &str, quoi: &str) -> Result<Option<i64>> {
    match get(cle) {
        None => Ok(None),
        Some(v) => v
            .as_int64()
            .map(Some)
            .ok_or_else(|| anyhow!("{quoi} : `{cle}` n'est pas un entier ({v:?})")),
    }
}

/// An integer property that must fit in `T` (no silent truncation).
pub(super) fn borne<T: TryFrom<i64>>(v: i64, cle: &str, quoi: &str) -> Result<T> {
    T::try_from(v).map_err(|_| anyhow!("{quoi} : `{cle}` hors bornes ({v})"))
}

pub(super) fn booleen(get: Lecteur, cle: &str, quoi: &str) -> Result<Option<bool>> {
    match get(cle) {
        None => Ok(None),
        Some(v) => v
            .as_bool()
            .map(Some)
            .ok_or_else(|| anyhow!("{quoi} : `{cle}` n'est pas un booléen ({v:?})")),
    }
}

pub(super) fn flottant(get: Lecteur, cle: &str, quoi: &str) -> Result<Option<f64>> {
    match get(cle) {
        None => Ok(None),
        Some(v) => v
            .as_float64()
            .or_else(|| v.as_int64().map(|i| i as f64))
            .map(Some)
            .ok_or_else(|| anyhow!("{quoi} : `{cle}` n'est pas un nombre ({v:?})")),
    }
}

/// A JSON property: absent or empty is `None`, a value that does not parse is an
/// error (never a silent `None`).
pub(super) fn json_facultatif<T: serde::de::DeserializeOwned>(
    get: Lecteur,
    cle: &str,
    quoi: &str,
) -> Result<Option<T>> {
    facultative(get, cle, quoi)?
        .map(|brut| {
            serde_json::from_str(&brut)
                .map_err(|e| anyhow!("{quoi} : `{cle}` n'est pas du JSON valide : {e}"))
        })
        .transpose()
}

/// Reads one node and decodes it.
pub(super) fn relire<T>(
    db: &ObrainDB,
    noeud: NodeId,
    decoder: &dyn Fn(Lecteur) -> Result<T>,
) -> Result<Fiche<T>> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud {noeud:?} indexé mais absent de la base"))?;
    let get = |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    };
    Ok(Fiche {
        noeud,
        valeur: decoder(&get)?,
    })
}

/// The key a quarantined node RESERVES, as `<label>:<parts>`: its id, and for the
/// entities keyed otherwise, that key too. Built the same way by the writers that
/// check it ([`cle_reservee`]).
pub(super) fn cle_reservee(label: &str, parties: &[&str]) -> String {
    format!("{label}:{}", parties.join("|"))
}

/// What a node that does not decode still says about itself, if it is readable:
/// the keys it holds, which stay RESERVED (as `reserves` in `note.rs`). Without
/// this, a delivery or a signal whose node is unreadable would be claimed or
/// reserved a second time, and an id or a `dedup_key` would be created again.
fn cles_d_un_noeud(label: &str, get: Lecteur) -> Vec<String> {
    let brut = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let uuid = |cle: &str| brut(cle).and_then(|s| s.parse::<Uuid>().ok());
    let mut cles = Vec::new();
    for champ in ["id", "run_id"] {
        if let Some(u) = uuid(champ) {
            cles.push(cle_reservee(label, &[&u.to_string()]));
        }
    }
    if let Some(k) = brut("dedup_key").filter(|k| !k.is_empty()) {
        cles.push(cle_reservee(label, &["cle", &k]));
    }
    if let Some(t) = uuid("trigger_id") {
        for champ in ["key", "delivery_key"] {
            if let Some(k) = brut(champ) {
                cles.push(cle_reservee(label, &[&t.to_string(), &k]));
            }
        }
    }
    cles
}

/// The refusal of a write that would take a reserved key.
pub(super) fn refus_reserve(quoi: &str) -> anyhow::Error {
    anyhow!(
        "{quoi} est réservé : un nœud de cette clé est sur disque, illisible à l'ouverture \
         (quarantaine). Il n'est ni repris, ni écrasé, ni servi une seconde fois"
    )
}

/// Every node of a label, decoded. A draft (a creation that never reached its
/// commit point) is deleted and counted; a node that does not decode is named in
/// the report, left untouched, and its keys go to `reserves`.
pub(super) fn charger<T>(
    db: &ObrainDB,
    label: &str,
    rapport: &mut RapportOuverture,
    reserves: &mut HashSet<String>,
    decoder: &dyn Fn(Lecteur) -> Result<T>,
) -> Vec<Fiche<T>> {
    let store = db.store();
    let mut fiches = Vec::new();
    for noeud in store.nodes_by_label(label) {
        let Some(n) = db.get_node(noeud) else {
            rapport
                .en_quarantaine
                .push(format!("nœud {label} {noeud:?} : illisible"));
            continue;
        };
        let get = |cle: &str| {
            n.get_property(cle)
                .filter(|v| !matches!(v, Value::Null))
                .cloned()
        };
        if get(BROUILLON).is_some() || (n.properties.is_empty() && store.out_degree(noeud) == 0) {
            db.delete_node(noeud);
            rapport.brouillons_supprimes += 1;
            continue;
        }
        match decoder(&get) {
            Ok(valeur) => fiches.push(Fiche { noeud, valeur }),
            Err(e) => {
                reserves.extend(cles_d_un_noeud(label, &get));
                rapport
                    .en_quarantaine
                    .push(format!("nœud {label} {noeud:?} : {e}"))
            }
        }
    }
    fiches
}

/// Indexes decoded nodes by a key; a key already taken is a quarantine.
pub(super) fn indexer<K: std::hash::Hash + Eq + std::fmt::Display, T>(
    fiches: Vec<Fiche<T>>,
    label: &str,
    rapport: &mut RapportOuverture,
    cle: impl Fn(&T) -> K,
) -> HashMap<K, Fiche<T>> {
    let mut index = HashMap::new();
    for f in fiches {
        match index.entry(cle(&f.valeur)) {
            Entry::Occupied(o) => rapport.en_quarantaine.push(format!(
                "nœud {label} {:?} : clé {} déjà prise",
                f.noeud,
                o.key()
            )),
            Entry::Vacant(v) => {
                v.insert(f);
            }
        }
    }
    index
}

/// Does this value go to a blob column (a string above the threshold, or a list
/// holding one)? Those values are not recovered by the WAL of the pinned obrain:
/// a write that carries one pays the full flush (`mod.rs`, [`SEUIL_COLONNE_BLOB`]).
pub(super) fn va_en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
        || v.as_list()
            .is_some_and(|l| l.iter().any(va_en_colonne_blob))
}

/// Writes a creation up to its commit point: the node, the draft marker, the
/// properties, then the marker removed. A value that goes to a blob column makes
/// the FULL flush run before the marker goes (as `plan.rs`): a crash before it
/// leaves a draft, never a node that reads whole and lost a value. NOT durable
/// yet: the caller runs `rendre_durable` once its whole write is done.
pub(super) fn creer_noeud(
    w: &mut Ecriture<'_>,
    label: &str,
    proprietes: Vec<(&'static str, Value)>,
) -> Result<NodeId> {
    w.flush_complet |= proprietes.iter().any(|(_, v)| va_en_colonne_blob(v));
    w.etape()?;
    let noeud = w.db.create_node(&[label]);
    w.etape()?;
    w.db.set_node_property(noeud, BROUILLON, Value::from(true));
    for (cle, valeur) in proprietes {
        w.etape()?;
        w.db.set_node_property(noeud, cle, valeur);
    }
    w.rendre_durable_avant_le_marqueur()?;
    w.etape()?;
    // The commit point.
    w.db.remove_node_property(noeud, BROUILLON);
    Ok(noeud)
}

/// A whole creation: [`creer_noeud`], read back, made durable.
pub(super) fn creer<T>(
    w: &mut Ecriture<'_>,
    label: &str,
    proprietes: Vec<(&'static str, Value)>,
    decoder: &dyn Fn(Lecteur) -> Result<T>,
) -> Result<Fiche<T>> {
    let noeud = creer_noeud(w, label, proprietes)?;
    let fiche = w.apres_ecriture(relire(w.db, noeud, decoder))?;
    w.rendre_durable()?;
    Ok(fiche)
}

/// Overwrites (or, for `None`, removes) properties of a live node. A new value
/// that goes to a blob column, or an OLD one that sat there and is replaced or
/// removed, makes the write pay the full flush (as `project.rs`). Not durable:
/// the caller makes the whole write durable.
pub(super) fn poser(
    w: &mut Ecriture<'_>,
    noeud: NodeId,
    proprietes: Vec<(&'static str, Option<Value>)>,
) -> Result<()> {
    let ancien_en_blob = w.db.get_node(noeud).is_some_and(|n| {
        proprietes
            .iter()
            .any(|(cle, _)| n.get_property(cle).is_some_and(va_en_colonne_blob))
    });
    w.flush_complet |= ancien_en_blob
        || proprietes
            .iter()
            .any(|(_, v)| v.as_ref().is_some_and(va_en_colonne_blob));
    for (cle, valeur) in proprietes {
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

/// Overwrites properties of a live node, reads it back, makes it durable.
pub(super) fn reecrire<T>(
    w: &mut Ecriture<'_>,
    noeud: NodeId,
    proprietes: Vec<(&'static str, Option<Value>)>,
    decoder: &dyn Fn(Lecteur) -> Result<T>,
) -> Result<Fiche<T>> {
    poser(w, noeud, proprietes)?;
    let fiche = w.apres_ecriture(relire(w.db, noeud, decoder))?;
    w.rendre_durable()?;
    Ok(fiche)
}

/// Deletes one node. A refusal of the engine leaves the index and the base
/// apart, so it makes the base indeterminate (restart rebuilds the index).
pub(super) fn supprimer(w: &mut Ecriture<'_>, noeud: NodeId) -> Result<()> {
    w.etape()?;
    let fait = w.db.delete_node(noeud);
    w.apres_ecriture(if fait {
        Ok(())
    } else {
        Err(anyhow!("obrain a refusé la suppression du nœud {noeud:?}"))
    })
}

/// A count passed by the caller (`limit`, `offset`): negative is refused, as
/// Neo4j refuses a negative `SKIP` / `LIMIT`.
pub(super) fn compte(v: i64, nom: &str) -> Result<usize> {
    usize::try_from(v).map_err(|_| anyhow!("`{nom}` négatif ({v}) : refusé"))
}

// ============================================================================
// Alerts
// ============================================================================

fn proprietes_alerte(a: &AlertNode) -> Vec<(&'static str, Value)> {
    let opt_date = |d: Option<DateTime<Utc>>| d.map(instant).unwrap_or_else(|| texte(""));
    vec![
        ("id", texte(a.id.to_string())),
        ("alert_type", texte(a.alert_type.clone())),
        ("severity", texte(a.severity.to_string())),
        ("message", texte(a.message.clone())),
        ("project_id", uuid_ou_vide(a.project_id)),
        ("acknowledged", Value::from(a.acknowledged)),
        (
            "acknowledged_by",
            texte(a.acknowledged_by.clone().unwrap_or_default()),
        ),
        ("acknowledged_at", opt_date(a.acknowledged_at)),
        ("created_at", instant(a.created_at)),
        ("dedup_key", texte(a.dedup_key.clone())),
        (
            "occurrence_count",
            Value::from(i64::try_from(a.occurrence_count).unwrap_or(i64::MAX)),
        ),
        ("first_seen", opt_date(a.first_seen)),
        ("last_seen", opt_date(a.last_seen)),
        ("priority", Value::from(a.priority)),
    ]
}

fn lire_alerte(get: Lecteur) -> Result<AlertNode> {
    let id = uuid_requis(get, "id", "nœud Alert")?;
    let quoi = format!("alerte {id}");
    let quoi = quoi.as_str();
    let severity: AlertSeverity = analyser(&requise(get, "severity", quoi)?, "severity", quoi)?;
    let created_at = date_requise(get, "created_at", quoi)?;
    let acknowledged = booleen(get, "acknowledged", quoi)?.unwrap_or(false);
    // As Neo4j reads a node written before these fields: the creation instant, 1.
    let first_seen = date_facultative(get, "first_seen", quoi)?.or(Some(created_at));
    let last_seen = date_facultative(get, "last_seen", quoi)?.or(Some(created_at));
    let occurrence_count = match entier(get, "occurrence_count", quoi)? {
        Some(n) => borne::<u64>(n.max(1), "occurrence_count", quoi)?,
        None => 1,
    };
    let priority = match flottant(get, "priority", quoi)? {
        Some(p) => p,
        None => AlertNode::compute_priority(
            severity,
            occurrence_count,
            last_seen.unwrap_or(created_at),
            acknowledged,
            Utc::now(),
        ),
    };
    Ok(AlertNode {
        id,
        alert_type: requise(get, "alert_type", quoi)?,
        severity,
        message: requise(get, "message", quoi)?,
        project_id: uuid_facultatif(get, "project_id", quoi)?,
        acknowledged,
        acknowledged_by: facultative(get, "acknowledged_by", quoi)?,
        acknowledged_at: date_facultative(get, "acknowledged_at", quoi)?,
        created_at,
        dedup_key: chaine(get, "dedup_key", quoi)?.unwrap_or_default(),
        occurrence_count,
        first_seen,
        last_seen,
        priority,
    })
}

/// The priority Neo4j's `create_alert` stores: the severity weight times the
/// chronicity (no recency term), 0 once acknowledged.
fn priorite_stockee(severite: AlertSeverity, occurrences: u64, acquittee: bool) -> f64 {
    if acquittee {
        return 0.0;
    }
    let chronicite = ((occurrences.max(1) as f64).ln() / 10.0).min(0.5);
    severite.weight() * (1.0 + chronicite)
}

/// `ORDER BY coalesce(priority, 0) DESC, coalesce(last_seen, created_at) DESC, id DESC`.
/// An alert with no STORED priority (`sans_priorite`, a node written before the
/// field) sorts as 0, as the `coalesce` does, whatever priority it reads with.
fn ordre_alertes(
    sans_priorite: &HashSet<Uuid>,
) -> impl Fn(&&AlertNode, &&AlertNode) -> std::cmp::Ordering + '_ {
    move |a, b| {
        let cle = |x: &AlertNode| {
            if sans_priorite.contains(&x.id) {
                0.0
            } else {
                x.priority
            }
        };
        ordre_alertes_par(a, b, cle(a), cle(b))
    }
}

fn ordre_alertes_par(a: &AlertNode, b: &AlertNode, pa: f64, pb: f64) -> std::cmp::Ordering {
    pb.total_cmp(&pa)
        .then_with(|| {
            b.last_seen
                .unwrap_or(b.created_at)
                .cmp(&a.last_seen.unwrap_or(a.created_at))
        })
        .then_with(|| b.id.cmp(&a.id))
}

// ============================================================================
// Event triggers
// ============================================================================

fn proprietes_declencheur(t: &EventTrigger) -> Result<Vec<(&'static str, Value)>> {
    let conditions = match &t.payload_conditions {
        Some(v) => serde_json::to_string(v)?,
        None => String::new(),
    };
    Ok(vec![
        ("id", texte(t.id.to_string())),
        ("name", texte(t.name.clone())),
        ("protocol_id", texte(t.protocol_id.to_string())),
        (
            "entity_type_pattern",
            texte(t.entity_type_pattern.clone().unwrap_or_default()),
        ),
        (
            "action_pattern",
            texte(t.action_pattern.clone().unwrap_or_default()),
        ),
        ("payload_conditions", texte(conditions)),
        ("cooldown_secs", Value::from(i64::from(t.cooldown_secs))),
        ("enabled", Value::from(t.enabled)),
        ("project_scope", uuid_ou_vide(t.project_scope)),
        ("created_at", instant(t.created_at)),
        ("updated_at", instant(t.updated_at)),
    ])
}

fn lire_declencheur(get: Lecteur) -> Result<EventTrigger> {
    let id = uuid_requis(get, "id", "nœud EventTrigger")?;
    let quoi = format!("déclencheur d'événement {id}");
    let quoi = quoi.as_str();
    Ok(EventTrigger {
        id,
        name: requise(get, "name", quoi)?,
        protocol_id: uuid_requis(get, "protocol_id", quoi)?,
        entity_type_pattern: facultative(get, "entity_type_pattern", quoi)?,
        action_pattern: facultative(get, "action_pattern", quoi)?,
        payload_conditions: json_facultatif(get, "payload_conditions", quoi)?,
        cooldown_secs: match entier(get, "cooldown_secs", quoi)? {
            Some(n) => borne(n, "cooldown_secs", quoi)?,
            None => 0,
        },
        // Absent: NOT enabled. Neo4j's filters (`enabled = true`) exclude such a
        // node, and a hook or trigger is never run on a guess.
        enabled: booleen(get, "enabled", quoi)?.unwrap_or(false),
        project_scope: uuid_facultatif(get, "project_scope", quoi)?,
        created_at: date_requise(get, "created_at", quoi)?,
        updated_at: date_requise(get, "updated_at", quoi)?,
    })
}

// ============================================================================
// Lifecycle hooks
// ============================================================================

/// The Neo4j spelling of a scope: the variant name (`format!("{:?}")`).
fn nom_portee(s: &LifecycleScope) -> String {
    format!("{s:?}")
}

fn portee_depuis(brut: &str, quoi: &str) -> Result<LifecycleScope> {
    Ok(match brut {
        "Task" => LifecycleScope::Task,
        "Plan" => LifecycleScope::Plan,
        "Step" => LifecycleScope::Step,
        "Milestone" => LifecycleScope::Milestone,
        _ => bail!("{quoi} : `scope` inconnue (« {brut} »)"),
    })
}

fn action_depuis(brut: &str, quoi: &str) -> Result<LifecycleActionType> {
    Ok(match brut {
        "CascadeChildren" => LifecycleActionType::CascadeChildren,
        "McpCall" => LifecycleActionType::McpCall,
        "CreateNote" => LifecycleActionType::CreateNote,
        "EmitAlert" => LifecycleActionType::EmitAlert,
        "StartProtocol" => LifecycleActionType::StartProtocol,
        _ => bail!("{quoi} : `action_type` inconnue (« {brut} »)"),
    })
}

fn proprietes_crochet(h: &LifecycleHook) -> Result<Vec<(&'static str, Value)>> {
    Ok(vec![
        ("id", texte(h.id.to_string())),
        ("name", texte(h.name.clone())),
        (
            "description",
            texte(h.description.clone().unwrap_or_default()),
        ),
        ("scope", texte(nom_portee(&h.scope))),
        ("on_status", texte(h.on_status.clone())),
        ("action_type", texte(format!("{:?}", h.action_type))),
        (
            "action_config",
            texte(serde_json::to_string(&h.action_config)?),
        ),
        ("priority", Value::from(i64::from(h.priority))),
        ("enabled", Value::from(h.enabled)),
        ("builtin", Value::from(h.builtin)),
        ("project_id", uuid_ou_vide(h.project_id)),
        ("created_at", instant(h.created_at)),
        // Neo4j's creation sets `updated_at` to the creation instant.
        ("updated_at", instant(h.created_at)),
    ])
}

fn lire_crochet(get: Lecteur) -> Result<LifecycleHook> {
    let id = uuid_requis(get, "id", "nœud LifecycleHook")?;
    let quoi = format!("crochet de cycle de vie {id}");
    let quoi = quoi.as_str();
    Ok(LifecycleHook {
        id,
        name: requise(get, "name", quoi)?,
        description: facultative(get, "description", quoi)?,
        scope: portee_depuis(&requise(get, "scope", quoi)?, quoi)?,
        on_status: requise(get, "on_status", quoi)?,
        action_type: action_depuis(&requise(get, "action_type", quoi)?, quoi)?,
        // Absent or empty: `{}`, as Neo4j reads it; present and not JSON: an error.
        action_config: json_facultatif(get, "action_config", quoi)?
            .unwrap_or_else(|| serde_json::Value::Object(Default::default())),
        priority: borne(
            entier(get, "priority", quoi)?
                .ok_or_else(|| anyhow!("{quoi} sans propriété `priority`"))?,
            "priority",
            quoi,
        )?,
        // Absent: NOT enabled. Neo4j's filters (`enabled = true`) exclude such a
        // node, and a hook or trigger is never run on a guess.
        enabled: booleen(get, "enabled", quoi)?.unwrap_or(false),
        builtin: booleen(get, "builtin", quoi)?.unwrap_or(false),
        project_id: uuid_facultatif(get, "project_id", quoi)?,
        created_at: date_requise(get, "created_at", quoi)?,
        updated_at: date_facultative(get, "updated_at", quoi)?,
    })
}

/// `ORDER BY priority ASC, id ASC`.
fn ordre_crochets(a: &LifecycleHook, b: &LifecycleHook) -> std::cmp::Ordering {
    a.priority.cmp(&b.priority).then_with(|| a.id.cmp(&b.id))
}

// ============================================================================
// The index
// ============================================================================

/// Alerts, event triggers and lifecycle hooks, decoded by id, rebuilt at open.
#[derive(Default)]
pub(super) struct IndexSurveillance {
    alertes: HashMap<Uuid, Fiche<AlertNode>>,
    /// `dedup_key → alert id` (the key Neo4j `MERGE`s on).
    par_cle: HashMap<String, Uuid>,
    declencheurs: HashMap<Uuid, Fiche<EventTrigger>>,
    crochets: HashMap<Uuid, Fiche<LifecycleHook>>,
    /// The keys of the quarantined nodes ([`cle_reservee`]): never taken again.
    reserves: HashSet<String>,
    /// Alerts read without a stored `priority` (sorted as 0, as Neo4j does).
    sans_priorite: HashSet<Uuid>,
}

impl IndexSurveillance {
    /// Rebuilds the index. A draft is deleted; a node that does not decode, an id
    /// or a `dedup_key` already taken, is named in the report and left untouched.
    pub(super) fn reconstruire(db: &ObrainDB, rapport: &mut RapportOuverture) -> Self {
        let mut index = Self::default();
        for f in charger(db, ALERTE, rapport, &mut index.reserves, &lire_alerte) {
            let a = &f.valeur;
            let stockee = db
                .get_node(f.noeud)
                .and_then(|n| n.get_property("priority").cloned())
                .is_some_and(|v| !matches!(v, Value::Null));
            if !stockee {
                index.sans_priorite.insert(a.id);
            }
            if index.alertes.contains_key(&a.id)
                || (!a.dedup_key.is_empty() && index.par_cle.contains_key(&a.dedup_key))
            {
                rapport.en_quarantaine.push(format!(
                    "nœud {ALERTE} {:?} : id {} ou dedup_key « {} » déjà pris",
                    f.noeud, a.id, a.dedup_key
                ));
                continue;
            }
            if !a.dedup_key.is_empty() {
                index.par_cle.insert(a.dedup_key.clone(), a.id);
            }
            index.alertes.insert(a.id, f);
        }
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
        index.crochets = indexer(
            charger(db, CROCHET, rapport, &mut index.reserves, &lire_crochet),
            CROCHET,
            rapport,
            |h| h.id,
        );
        index
    }

    /// Is this id taken by any entity of this domain?
    fn id_pris(&self, id: Uuid) -> bool {
        self.alertes.contains_key(&id)
            || self.declencheurs.contains_key(&id)
            || self.crochets.contains_key(&id)
    }

    /// Refuses, by name, an id (or a `dedup_key`) a quarantined node holds.
    fn garde_reserve(&self, id: Uuid, dedup_key: Option<&str>) -> Result<()> {
        let id = id.to_string();
        for label in [ALERTE, DECLENCHEUR, CROCHET] {
            if self.reserves.contains(&cle_reservee(label, &[&id])) {
                return Err(refus_reserve(&format!("l'id {id}")));
            }
        }
        if let Some(k) = dedup_key {
            if self.reserves.contains(&cle_reservee(ALERTE, &["cle", k])) {
                return Err(refus_reserve(&format!("la dedup_key « {k} »")));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl TriggerStore for ObrainGraphStore {
    async fn create_alert(&self, alert: &AlertNode) -> Result<()> {
        let a = alert.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let cle = if a.dedup_key.is_empty() {
                    AlertNode::make_dedup_key(&a.alert_type, a.project_id, &a.message)
                } else {
                    a.dedup_key.clone()
                };
                let maintenant = Utc::now();
                let existante = {
                    let etat = w.index()?;
                    match etat.surveillance.par_cle.get(&cle) {
                        Some(id) => {
                            let f =
                                etat.surveillance.alertes.get(id).ok_or_else(|| {
                                    anyhow!("index des alertes incohérent ({id})")
                                })?;
                            Some((f.noeud, f.valeur.clone()))
                        }
                        None => {
                            etat.surveillance.garde_reserve(a.id, Some(&cle))?;
                            if etat.surveillance.id_pris(a.id) {
                                bail!(
                                    "l'id {} est déjà pris par une autre entité (alerte d'une \
                                     autre dedup_key, déclencheur ou crochet) : refusé",
                                    a.id
                                );
                            }
                            None
                        }
                    }
                };
                let fiche = match existante {
                    // Seen again: one more occurrence on the same node (Neo4j's `SET`).
                    Some((noeud, ancienne)) => {
                        let occurrences = ancienne.occurrence_count.saturating_add(1);
                        let priorite =
                            priorite_stockee(a.severity, occurrences, ancienne.acknowledged);
                        reecrire(
                            w,
                            noeud,
                            vec![
                                ("severity", Some(texte(a.severity.to_string()))),
                                ("message", Some(texte(a.message.clone()))),
                                ("last_seen", Some(instant(maintenant))),
                                (
                                    "occurrence_count",
                                    Some(Value::from(
                                        i64::try_from(occurrences).unwrap_or(i64::MAX),
                                    )),
                                ),
                                ("priority", Some(Value::from(priorite))),
                            ],
                            &lire_alerte,
                        )?
                    }
                    // First sight: Neo4j's `ON CREATE SET`, then one occurrence.
                    None => {
                        let nouvelle = AlertNode {
                            id: a.id,
                            alert_type: a.alert_type.clone(),
                            severity: a.severity,
                            message: a.message.clone(),
                            project_id: a.project_id,
                            acknowledged: false,
                            acknowledged_by: None,
                            acknowledged_at: None,
                            created_at: maintenant,
                            dedup_key: cle.clone(),
                            occurrence_count: 1,
                            first_seen: Some(maintenant),
                            last_seen: Some(maintenant),
                            priority: priorite_stockee(a.severity, 1, false),
                        };
                        creer(w, ALERTE, proprietes_alerte(&nouvelle), &lire_alerte)?
                    }
                };
                let mut etat = w.index_mut()?;
                // Its priority is stored now (both paths write it).
                etat.surveillance.sans_priorite.remove(&fiche.valeur.id);
                etat.surveillance.par_cle.insert(cle, fiche.valeur.id);
                etat.surveillance.alertes.insert(fiche.valeur.id, fiche);
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
            let mut alertes: Vec<&AlertNode> = etat
                .surveillance
                .alertes
                .values()
                .map(|f| &f.valeur)
                .filter(|a| !a.acknowledged)
                .filter(|a| project_id.is_none_or(|p| a.project_id == Some(p)))
                .collect();
            alertes.sort_by(ordre_alertes(&etat.surveillance.sans_priorite));
            Ok(alertes.into_iter().take(limit).cloned().collect())
        })
    }

    async fn acknowledge_alert(&self, alert_id: Uuid, acknowledged_by: &str) -> Result<()> {
        let par = acknowledged_by.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // A missing alert is a silent no-op, as Neo4j's `MATCH … SET`.
                let Some(noeud) = w
                    .index()?
                    .surveillance
                    .alertes
                    .get(&alert_id)
                    .map(|f| f.noeud)
                else {
                    return Ok(());
                };
                let fiche = reecrire(
                    w,
                    noeud,
                    vec![
                        ("acknowledged", Some(Value::from(true))),
                        ("acknowledged_by", Some(texte(par))),
                        ("acknowledged_at", Some(instant(Utc::now()))),
                    ],
                    &lire_alerte,
                )?;
                w.index_mut()?.surveillance.alertes.insert(alert_id, fiche);
                Ok(())
            },
        )
        .await
    }

    async fn get_alert(&self, alert_id: Uuid) -> Result<Option<AlertNode>> {
        self.lire(|_, etat| {
            Ok(etat
                .surveillance
                .alertes
                .get(&alert_id)
                .map(|f| f.valeur.clone()))
        })
    }

    async fn list_alerts(
        &self,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<AlertNode>, usize)> {
        self.lire(|_, etat| {
            // `project_id` wins over `workspace_slug`; an unknown slug is no project.
            let portee: Option<HashSet<Uuid>> = match (project_id, workspace_slug) {
                (Some(p), _) => Some(HashSet::from([p])),
                (None, Some(slug)) => Some(etat.espaces.projets_du_slug(slug)),
                (None, None) => None,
            };
            let mut alertes: Vec<&AlertNode> = etat
                .surveillance
                .alertes
                .values()
                .map(|f| &f.valeur)
                .filter(|a| {
                    portee
                        .as_ref()
                        .is_none_or(|ps| a.project_id.is_some_and(|p| ps.contains(&p)))
                })
                .collect();
            alertes.sort_by(ordre_alertes(&etat.surveillance.sans_priorite));
            let total = alertes.len();
            Ok((
                alertes
                    .into_iter()
                    .skip(offset)
                    .take(limit)
                    .cloned()
                    .collect(),
                total,
            ))
        })
    }

    async fn list_event_triggers(
        &self,
        project_scope: Option<Uuid>,
        enabled_only: bool,
    ) -> Result<Vec<EventTrigger>> {
        self.lire(|_, etat| {
            let mut declencheurs: Vec<&EventTrigger> = etat
                .surveillance
                .declencheurs
                .values()
                .map(|f| &f.valeur)
                // A project's own triggers and the global ones.
                .filter(|t| {
                    project_scope
                        .is_none_or(|p| t.project_scope.is_none() || t.project_scope == Some(p))
                })
                .filter(|t| !enabled_only || t.enabled)
                .collect();
            // ORDER BY created_at DESC, id DESC.
            declencheurs.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(b.id.cmp(&a.id)));
            Ok(declencheurs.into_iter().cloned().collect())
        })
    }

    async fn create_event_trigger(&self, trigger: &EventTrigger) -> Result<Uuid> {
        let mut t = trigger.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                w.index()?.surveillance.garde_reserve(t.id, None)?;
                if w.index()?.surveillance.id_pris(t.id) {
                    bail!(
                        "l'id {} est déjà pris par une autre entité de ce domaine : refusé",
                        t.id
                    );
                }
                // Neo4j stamps both dates with the instant of the write.
                let maintenant = Utc::now();
                t.created_at = maintenant;
                t.updated_at = maintenant;
                let fiche = creer(
                    w,
                    DECLENCHEUR,
                    proprietes_declencheur(&t)?,
                    &lire_declencheur,
                )?;
                w.index_mut()?.surveillance.declencheurs.insert(t.id, fiche);
                Ok(t.id)
            },
        )
        .await
    }

    async fn get_event_trigger(&self, id: Uuid) -> Result<Option<EventTrigger>> {
        self.lire(|_, etat| {
            Ok(etat
                .surveillance
                .declencheurs
                .get(&id)
                .map(|f| f.valeur.clone()))
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
                let Some(noeud) = w
                    .index()?
                    .surveillance
                    .declencheurs
                    .get(&id)
                    .map(|f| f.noeud)
                else {
                    return Ok(false);
                };
                let mut maj: Vec<(&'static str, Option<Value>)> =
                    vec![("updated_at", Some(instant(Utc::now())))];
                if let Some(e) = enabled {
                    maj.push(("enabled", Some(Value::from(e))));
                }
                if let Some(n) = name {
                    maj.push(("name", Some(texte(n))));
                }
                if let Some(p) = entity_type_pattern {
                    maj.push(("entity_type_pattern", Some(texte(p.unwrap_or_default()))));
                }
                if let Some(p) = action_pattern {
                    maj.push(("action_pattern", Some(texte(p.unwrap_or_default()))));
                }
                if let Some(c) = payload_conditions {
                    let brut = match c {
                        Some(v) => serde_json::to_string(&v)?,
                        None => String::new(),
                    };
                    maj.push(("payload_conditions", Some(texte(brut))));
                }
                if let Some(c) = cooldown_secs {
                    maj.push(("cooldown_secs", Some(Value::from(i64::from(c)))));
                }
                if let Some(s) = project_scope {
                    maj.push(("project_scope", Some(uuid_ou_vide(s))));
                }
                let fiche = reecrire(w, noeud, maj, &lire_declencheur)?;
                w.index_mut()?.surveillance.declencheurs.insert(id, fiche);
                Ok(true)
            },
        )
        .await
    }

    async fn delete_event_trigger(&self, id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = w
                    .index()?
                    .surveillance
                    .declencheurs
                    .get(&id)
                    .map(|f| f.noeud)
                else {
                    return Ok(false);
                };
                supprimer(w, noeud)?;
                w.rendre_durable()?;
                w.index_mut()?.surveillance.declencheurs.remove(&id);
                Ok(true)
            },
        )
        .await
    }

    async fn create_lifecycle_hook(&self, hook: &LifecycleHook) -> Result<()> {
        let h = hook.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                w.index()?.surveillance.garde_reserve(h.id, None)?;
                if w.index()?.surveillance.id_pris(h.id) {
                    bail!(
                        "l'id {} est déjà pris par une autre entité de ce domaine : refusé",
                        h.id
                    );
                }
                let fiche = creer(w, CROCHET, proprietes_crochet(&h)?, &lire_crochet)?;
                w.index_mut()?.surveillance.crochets.insert(h.id, fiche);
                Ok(())
            },
        )
        .await
    }

    async fn get_lifecycle_hook(&self, id: Uuid) -> Result<Option<LifecycleHook>> {
        self.lire(|_, etat| {
            Ok(etat
                .surveillance
                .crochets
                .get(&id)
                .map(|f| f.valeur.clone()))
        })
    }

    async fn list_lifecycle_hooks(&self, project_id: Option<Uuid>) -> Result<Vec<LifecycleHook>> {
        self.lire(|_, etat| {
            let mut crochets: Vec<&LifecycleHook> = etat
                .surveillance
                .crochets
                .values()
                .map(|f| &f.valeur)
                .filter(|h| {
                    project_id.is_none_or(|p| h.project_id.is_none() || h.project_id == Some(p))
                })
                .collect();
            crochets.sort_by(|a, b| ordre_crochets(a, b));
            Ok(crochets.into_iter().cloned().collect())
        })
    }

    async fn update_lifecycle_hook(
        &self,
        id: Uuid,
        updates: &UpdateLifecycleHookRequest,
    ) -> Result<()> {
        // `UpdateLifecycleHookRequest` is not `Clone`: its fields, owned.
        let (nom, description, statut, config, priorite, actif) = (
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
                // A missing hook is a silent no-op, as Neo4j's `MATCH … SET`.
                let Some(noeud) = w.index()?.surveillance.crochets.get(&id).map(|f| f.noeud) else {
                    return Ok(());
                };
                let mut maj: Vec<(&'static str, Option<Value>)> =
                    vec![("updated_at", Some(instant(Utc::now())))];
                if let Some(n) = nom {
                    maj.push(("name", Some(texte(n))));
                }
                if let Some(d) = description {
                    maj.push(("description", Some(texte(d.unwrap_or_default()))));
                }
                if let Some(s) = statut {
                    maj.push(("on_status", Some(texte(s))));
                }
                if let Some(c) = config {
                    maj.push(("action_config", Some(texte(serde_json::to_string(&c)?))));
                }
                if let Some(p) = priorite {
                    maj.push(("priority", Some(Value::from(i64::from(p)))));
                }
                if let Some(e) = actif {
                    maj.push(("enabled", Some(Value::from(e))));
                }
                let fiche = reecrire(w, noeud, maj, &lire_crochet)?;
                w.index_mut()?.surveillance.crochets.insert(id, fiche);
                Ok(())
            },
        )
        .await
    }

    async fn delete_lifecycle_hook(&self, id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some((noeud, builtin)) = w
                    .index()?
                    .surveillance
                    .crochets
                    .get(&id)
                    .map(|f| (f.noeud, f.valeur.builtin))
                else {
                    return Ok(());
                };
                if builtin {
                    bail!("Cannot delete builtin lifecycle hook");
                }
                supprimer(w, noeud)?;
                w.rendre_durable()?;
                w.index_mut()?.surveillance.crochets.remove(&id);
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
        self.lire(|_, etat| {
            let mut crochets: Vec<&LifecycleHook> = etat
                .surveillance
                .crochets
                .values()
                .map(|f| &f.valeur)
                .filter(|h| h.scope == *scope && h.on_status == on_status && h.enabled)
                // With a project: its hooks and the global ones; without: the global ones.
                .filter(|h| {
                    h.project_id.is_none() || (project_id.is_some() && h.project_id == project_id)
                })
                .collect();
            crochets.sort_by(|a, b| ordre_crochets(a, b));
            Ok(crochets.into_iter().cloned().collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obrain::Creation;
    use futures::future::join_all;

    fn alerte(sujet: &str, message: &str) -> AlertNode {
        AlertNode::new_for_subject(
            "test".to_string(),
            AlertSeverity::Warning,
            message.to_string(),
            None,
            sujet,
        )
    }

    fn crochet(nom: &str, portee: LifecycleScope, priorite: i32) -> LifecycleHook {
        let mut h = LifecycleHook::new(
            nom.to_string(),
            portee,
            "completed".to_string(),
            LifecycleActionType::CascadeChildren,
            serde_json::json!({"k": nom}),
        );
        h.priority = priorite;
        h
    }

    fn declencheur(nom: &str, portee: Option<Uuid>) -> EventTrigger {
        EventTrigger {
            id: Uuid::new_v4(),
            name: nom.to_string(),
            protocol_id: Uuid::new_v4(),
            entity_type_pattern: Some("task".into()),
            action_pattern: None,
            payload_conditions: Some(serde_json::json!({"status": ["done", "failed"]})),
            cooldown_secs: 30,
            enabled: true,
            project_scope: portee,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn une_alerte_revue_compte_sur_le_meme_noeud_et_s_acquitte() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let a = alerte("cond", "premier");
        store.create_alert(&a).await.unwrap();
        let mut b = alerte("cond", "second");
        b.severity = AlertSeverity::Critical;
        store.create_alert(&b).await.unwrap();
        assert!(store.get_alert(b.id).await.unwrap().is_none());
        let lue = store.get_alert(a.id).await.unwrap().unwrap();
        assert_eq!(lue.occurrence_count, 2);
        assert_eq!(lue.message, "second");
        assert_eq!(lue.severity, AlertSeverity::Critical);
        assert!((lue.priority - priorite_stockee(AlertSeverity::Critical, 2, false)).abs() < 1e-12);
        assert_eq!(store.list_pending_alerts(None, 10).await.unwrap().len(), 1);
        store.acknowledge_alert(a.id, "alice").await.unwrap();
        let lue = store.get_alert(a.id).await.unwrap().unwrap();
        assert!(lue.acknowledged);
        assert_eq!(lue.acknowledged_by.as_deref(), Some("alice"));
        assert!(store
            .list_pending_alerts(None, 10)
            .await
            .unwrap()
            .is_empty());
        // Seen again once acknowledged: still acknowledged, priority 0.
        store
            .create_alert(&alerte("cond", "troisième"))
            .await
            .unwrap();
        let lue = store.get_alert(a.id).await.unwrap().unwrap();
        assert!(lue.acknowledged);
        assert_eq!(lue.priority, 0.0);
        assert_eq!(lue.occurrence_count, 3);
        // Acknowledging what does not exist is a no-op.
        store.acknowledge_alert(Uuid::new_v4(), "x").await.unwrap();
    }

    #[tokio::test]
    async fn huit_creations_concurrentes_d_une_cle_font_un_noeud_et_huit_occurrences() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let alertes: Vec<AlertNode> = (0..8).map(|i| alerte("course", &format!("m{i}"))).collect();
        let issues = join_all(alertes.iter().map(|a| store.create_alert(a))).await;
        assert!(issues.iter().all(|r| r.is_ok()), "{issues:?}");
        let (toutes, total) = store.list_alerts(None, None, 100, 0).await.unwrap();
        assert_eq!(total, 1, "{toutes:?}");
        assert_eq!(toutes[0].occurrence_count, 8);
        assert_eq!(store.interne.db.store().node_count(), 1);
    }

    #[tokio::test]
    async fn un_id_d_alerte_deja_pris_par_une_autre_cle_est_refuse() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let a = alerte("une", "m");
        store.create_alert(&a).await.unwrap();
        let mut b = alerte("autre", "m");
        b.id = a.id;
        assert!(store.create_alert(&b).await.is_err());
        assert_eq!(store.list_alerts(None, None, 10, 0).await.unwrap().1, 1);
    }

    #[tokio::test]
    async fn declencheurs_et_crochets_survivent_a_la_reouverture() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let projet = Uuid::new_v4();
        let global = declencheur("global", None);
        let local = declencheur("local", Some(projet));
        let mut h1 = crochet("h1", LifecycleScope::Task, 5);
        h1.description = Some("décrit".into());
        let mut h2 = crochet("h2", LifecycleScope::Task, 1);
        h2.project_id = Some(projet);
        h2.builtin = true;
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_event_trigger(&global).await.unwrap();
            store.create_event_trigger(&local).await.unwrap();
            assert!(store
                .update_event_trigger(
                    local.id,
                    Some(false),
                    None,
                    Some(None),
                    None,
                    Some(None),
                    Some(7),
                    None
                )
                .await
                .unwrap());
            assert!(!store
                .update_event_trigger(
                    Uuid::new_v4(),
                    Some(true),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None
                )
                .await
                .unwrap());
            store.create_lifecycle_hook(&h1).await.unwrap();
            store.create_lifecycle_hook(&h2).await.unwrap();
            store
                .update_lifecycle_hook(
                    h1.id,
                    &UpdateLifecycleHookRequest {
                        name: None,
                        description: Some(None),
                        on_status: None,
                        action_config: None,
                        priority: None,
                        enabled: None,
                    },
                )
                .await
                .unwrap();
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(store.rapport_d_ouverture(), &RapportOuverture::default());
        let l = store.get_event_trigger(local.id).await.unwrap().unwrap();
        assert!(!l.enabled);
        assert_eq!(l.cooldown_secs, 7);
        assert_eq!(l.entity_type_pattern, None);
        assert_eq!(l.payload_conditions, None);
        let g = store.get_event_trigger(global.id).await.unwrap().unwrap();
        assert_eq!(g.payload_conditions, global.payload_conditions);
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
                .list_event_triggers(Some(projet), true)
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            store
                .list_event_triggers(Some(Uuid::new_v4()), false)
                .await
                .unwrap()
                .len(),
            1,
            "another project sees only the global trigger"
        );
        let relu = store.get_lifecycle_hook(h1.id).await.unwrap().unwrap();
        assert_eq!(relu.description, None);
        assert_eq!(relu.action_config, h1.action_config);
        assert_eq!(relu.action_type, LifecycleActionType::CascadeChildren);
        // Priority ASC; the global hook only without a project.
        let pour: Vec<Uuid> = store
            .list_hooks_for_scope(&LifecycleScope::Task, "completed", Some(projet))
            .await
            .unwrap()
            .iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(pour, vec![h2.id, h1.id]);
        let globaux = store
            .list_hooks_for_scope(&LifecycleScope::Task, "completed", None)
            .await
            .unwrap();
        assert_eq!(globaux.len(), 1);
        assert!(store.delete_lifecycle_hook(h2.id).await.is_err(), "builtin");
        store.delete_lifecycle_hook(h1.id).await.unwrap();
        assert!(store.get_lifecycle_hook(h1.id).await.unwrap().is_none());
        assert!(store.delete_event_trigger(global.id).await.unwrap());
        assert!(!store.delete_event_trigger(global.id).await.unwrap());
    }

    #[tokio::test]
    async fn un_noeud_illisible_est_mis_en_quarantaine_jamais_lu_par_defaut() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let bon = crochet("bon", LifecycleScope::Plan, 1);
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_lifecycle_hook(&bon).await.unwrap();
            let db = &store.interne.db;
            // A hook whose scope Neo4j would read as `Task` in silence.
            let n = db.create_node(&[CROCHET]);
            for (k, v) in proprietes_crochet(&crochet("mauvais", LifecycleScope::Task, 2)).unwrap()
            {
                db.set_node_property(n, k, v);
            }
            db.set_node_property(n, "scope", Value::from("Galaxy"));
            // An alert with an unknown severity.
            let n = db.create_node(&[ALERTE]);
            for (k, v) in proprietes_alerte(&alerte("x", "y")) {
                db.set_node_property(n, k, v);
            }
            db.set_node_property(n, "severity", Value::from("apocalyptic"));
            // A draft event trigger: deleted at open.
            let n = db.create_node(&[DECLENCHEUR]);
            db.set_node_property(n, BROUILLON, Value::from(true));
            db.set_node_property(n, "id", Value::from(Uuid::new_v4().to_string()));
            db.commit_wal().unwrap();
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        let rapport = store.rapport_d_ouverture();
        assert_eq!(rapport.brouillons_supprimes, 1, "{rapport:?}");
        assert_eq!(rapport.en_quarantaine.len(), 2, "{rapport:?}");
        assert!(rapport.en_quarantaine.iter().any(|q| q.contains("Galaxy")));
        assert!(rapport
            .en_quarantaine
            .iter()
            .any(|q| q.contains("apocalyptic")));
        let crochets = store.list_lifecycle_hooks(None).await.unwrap();
        assert_eq!(crochets.len(), 1, "the quarantined hook is never served");
        assert_eq!(crochets[0].id, bon.id);
        assert!(store
            .list_alerts(None, None, 10, 0)
            .await
            .unwrap()
            .0
            .is_empty());
    }
}
