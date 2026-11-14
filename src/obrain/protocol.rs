//! `ProtocolStore` on obrain: protocols, their states and their transitions
//! (lot O-PROTOCOL). The runs that execute them are in `protocolrun.rs`.
//!
//! PORTED here, with the Neo4j semantics of `src/neo4j/protocol.rs`: all eleven
//! methods of the trait — `upsert_protocol`, `get_protocol`, `list_protocols`,
//! `get_protocol_by_name_and_project`, `delete_protocol`, and the upsert, read
//! and delete of states and transitions. Nothing of this trait is refused: the
//! cascade of `delete_protocol` reaches the protocol's states and transitions,
//! its runs and their runtime states — all ported, here and in the sister
//! domain (`protocolrun.rs`).
//!
//! # Layout
//!
//! One node per entity, labelled `Protocol`, `ProtocolState`,
//! `ProtocolTransition`, with the properties Neo4j writes (`terminal_states_json`,
//! `trigger_config_json`, `guard_expr`… are JSON strings; a missing optional value
//! is the empty string; a state timeout is seconds, `-1` for none).
//!
//! Membership is a PROPERTY (`protocol_id`, `project_id`), as in `plan.rs`:
//! Neo4j also writes `HAS_STATE`, `HAS_TRANSITION`, `BELONGS_TO` and
//! `BELONGS_TO_SKILL` edges, and keeps the property in step. No reader of
//! either domain traverses those edges that the property does not answer.
//!
//! # Divergences, all on the strict side
//!
//! - `upsert_protocol` with a `project_id` whose project does not exist is
//!   REFUSED. Neo4j's `MATCH` on the project silently writes nothing.
//! - Upserting a state or a transition of a protocol that does not exist is
//!   REFUSED (Neo4j's `MATCH (proto…)` silently writes nothing).
//! - A stored JSON value or enum that does not parse is an error at open
//!   (the node is quarantined, named, untouched), where Neo4j drops it to its
//!   default: a guard, a status, a terminal-state list never turns silently into
//!   "none".
//! - Fail closed on a MISSING value too, for the keys that exist since their
//!   label does and that Neo4j writes on every write (the empty string for
//!   "none"): a key ABSENT there is a value lost, an error at open
//!   (quarantine), where Neo4j reads it as "none": `terminal_states_json`,
//!   `trigger_config_json`, `relevance_vector_json` (protocol), `guard`
//!   (transition) and, in `protocolrun.rs`, `states_visited_json`, `error` and
//!   `status` of a run.
//! - The keys that came LATER, with no back-fill of the nodes written before
//!   (nothing in `data_migrations.rs`; `ensure_default_protocols` does not
//!   rewrite existing transitions): `guard_expr` (2026-11-07),
//!   `generator_config_json` (2026-03-10), `prompt_fragment`,
//!   `available_tools_json`, `forbidden_actions_json` (2026-03-19). ABSENT reads
//!   as "none", as in Neo4j — a migrated base keeps its old transitions and
//!   states; a value PRESENT that does not read is still an error (quarantine).
//! - `delete_protocol` reaches the protocol's states, transitions and runs by
//!   their `protocol_id` PROPERTY, and ALSO by the edges Neo4j follows
//!   (`HAS_STATE`, `HAS_TRANSITION` from the protocol, `INSTANCE_OF` to it), and
//!   reaches the nodes of these labels the index does not serve (quarantined at
//!   open) by their raw `protocol_id`: what Neo4j's `DETACH DELETE` removes,
//!   readable or not, is removed here too, edges included. A migrated node whose
//!   edge and property name two different protocols goes with both.
//! - `delete_protocol` also deletes the runtime states of the runs it deletes:
//!   Neo4j leaves them orphaned (`DETACH DELETE s, t, r, proto` does not reach
//!   `RuntimeState`).
//! - `list_protocols` orders by the parsed `updated_at`; Neo4j compares the raw
//!   RFC 3339 strings, which misorder timestamps of different fraction width.
//! - The `state_timeout_explicit` marker, which only serves a Neo4j driver
//!   repair at boot (`data_migrations.rs`), is not written.
//! - `upsert_protocol` keeps the stored `created_at` of an existing protocol, as
//!   Neo4j's `ON CREATE SET` does.
//!
//! # Atomicity
//!
//! A creation follows the draft-marker protocol of `project.rs`. A cascade
//! deletes the leaves first and the protocol node LAST, one durable commit at the
//! end: a crash half-way leaves a protocol with fewer children, never children
//! without their protocol, and a second `delete_protocol` finishes the job.
//! Reads come from the in-memory index, which a write publishes only once durable.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::str::FromStr;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use serde::de::DeserializeOwned;
use uuid::Uuid;

use super::{Ecriture, ObrainGraphStore, RapportOuverture, SEUIL_COLONNE_BLOB};
use crate::neo4j::traits::ProtocolStore;
use crate::protocol::{Protocol, ProtocolCategory, ProtocolState, ProtocolTransition};

const PROTOCOLE: &str = "Protocol";
const ETAT: &str = "ProtocolState";
const TRANSITION: &str = "ProtocolTransition";
/// The draft marker: present from the first record of a creation to the last.
pub(super) const BROUILLON: &str = "_brouillon";

// ============================================================================
// Shared with `protocolrun.rs`: decoding and writing one node
// ============================================================================

/// A property lookup on one node (nulls read as absent).
pub(super) type Lecteur<'a> = &'a dyn Fn(&str) -> Option<Value>;

/// Where an entity lives in the base, and its decoded value.
pub(super) struct Fiche<T> {
    pub(super) noeud: NodeId,
    pub(super) valeur: T,
}

/// Does this value go to a blob column (a string above the threshold)?
pub(super) fn va_en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
}

pub(super) fn a_un_blob(proprietes: &[(&'static str, Value)]) -> bool {
    proprietes.iter().any(|(_, v)| va_en_colonne_blob(v))
}

/// A string property as stored, the empty string included (absent: `None`).
pub(super) fn chaine(get: Lecteur, cle: &str) -> Option<String> {
    get(cle).and_then(|v| v.as_str().map(str::to_string))
}

/// A string property written on EVERY write (the empty string for "none"):
/// absent, or not a string, is a value lost — an error, never `None` (fail
/// closed, see the module doc). The empty string is `None`.
pub(super) fn texte_attendu(get: Lecteur, cle: &str, quoi: &str) -> Result<Option<String>> {
    match get(cle) {
        None => bail!(
            "{quoi} : propriété `{cle}` absente, alors qu'elle est écrite à chaque écriture \
             (la chaîne vide pour « aucune ») : valeur perdue, le nœud n'est pas lu avec un \
             défaut"
        ),
        Some(v) => match v.as_str() {
            None => bail!("{quoi} : `{cle}` n'est pas une chaîne ({v:?})"),
            Some("") => Ok(None),
            Some(s) => Ok(Some(s.to_string())),
        },
    }
}

/// A JSON property that came LATER than its label (module doc): absent is
/// `None`, as in Neo4j; present, it reads as [`json_attendu`] (empty is `None`,
/// not a string or not JSON is an error).
pub(super) fn json_tardif<T: DeserializeOwned>(
    get: Lecteur,
    cle: &str,
    quoi: &str,
) -> Result<Option<T>> {
    if get(cle).is_none() {
        return Ok(None);
    }
    json_attendu(get, cle, quoi)
}

/// A string property that came later than its label: absent is `None`.
fn texte_tardif(get: Lecteur, cle: &str, quoi: &str) -> Result<Option<String>> {
    if get(cle).is_none() {
        return Ok(None);
    }
    texte_attendu(get, cle, quoi)
}

/// A JSON property written on every write: absent is an error (see
/// [`texte_attendu`]), empty is `None`, a value that does not parse is an error.
pub(super) fn json_attendu<T: DeserializeOwned>(
    get: Lecteur,
    cle: &str,
    quoi: &str,
) -> Result<Option<T>> {
    texte_attendu(get, cle, quoi)?
        .map(|brut| {
            serde_json::from_str(&brut)
                .map_err(|e| anyhow!("{quoi} : `{cle}` n'est pas du JSON valide : {e}"))
        })
        .transpose()
}

/// A required string property.
pub(super) fn requise(get: Lecteur, cle: &str, quoi: &str) -> Result<String> {
    chaine(get, cle).ok_or_else(|| anyhow!("{quoi} sans propriété `{cle}`"))
}

/// A string property that may be absent or empty (both: `None`).
pub(super) fn facultative(get: Lecteur, cle: &str) -> Option<String> {
    chaine(get, cle).filter(|s| !s.is_empty())
}

pub(super) fn uuid_requis(get: Lecteur, cle: &str, quoi: &str) -> Result<Uuid> {
    let brut = requise(get, cle, quoi)?;
    brut.parse()
        .map_err(|e| anyhow!("{quoi} : `{cle}` illisible (« {brut} ») : {e}"))
}

pub(super) fn uuid_facultatif(get: Lecteur, cle: &str, quoi: &str) -> Result<Option<Uuid>> {
    facultative(get, cle)
        .map(|brut| {
            brut.parse()
                .map_err(|e| anyhow!("{quoi} : `{cle}` illisible (« {brut} ») : {e}"))
        })
        .transpose()
}

pub(super) fn date_requise(get: Lecteur, cle: &str, quoi: &str) -> Result<DateTime<Utc>> {
    let brut = requise(get, cle, quoi)?;
    brut.parse()
        .map_err(|e| anyhow!("{quoi} : `{cle}` illisible (« {brut} ») : {e}"))
}

pub(super) fn date_facultative(
    get: Lecteur,
    cle: &str,
    quoi: &str,
) -> Result<Option<DateTime<Utc>>> {
    facultative(get, cle)
        .map(|brut| {
            brut.parse()
                .map_err(|e| anyhow!("{quoi} : `{cle}` illisible (« {brut} ») : {e}"))
        })
        .transpose()
}

/// An enum property: absent or empty reads as the default; an unknown spelling is
/// an error.
pub(super) fn enumeration<T: FromStr + Default>(get: Lecteur, cle: &str, quoi: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match facultative(get, cle) {
        None => Ok(T::default()),
        Some(brut) => brut
            .parse()
            .map_err(|e| anyhow!("{quoi} : `{cle}` illisible (« {brut} ») : {e}")),
    }
}

/// An enum property written on every write, never empty: absent or empty is an
/// error (fail closed), as is an unknown spelling.
pub(super) fn enumeration_requise<T: FromStr>(get: Lecteur, cle: &str, quoi: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let brut = texte_attendu(get, cle, quoi)?
        .ok_or_else(|| anyhow!("{quoi} : `{cle}` vide, alors qu'elle n'est jamais écrite vide"))?;
    brut.parse()
        .map_err(|e| anyhow!("{quoi} : `{cle}` illisible (« {brut} ») : {e}"))
}

/// An optional enum property: absent or empty is `None`.
fn enumeration_facultative<T: FromStr>(get: Lecteur, cle: &str, quoi: &str) -> Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    facultative(get, cle)
        .map(|brut| {
            brut.parse()
                .map_err(|e| anyhow!("{quoi} : `{cle}` illisible (« {brut} ») : {e}"))
        })
        .transpose()
}

/// A count / size / duration: negative is 0, too large saturates.
pub(super) fn non_signe<T: TryFrom<i64>>(v: i64, max: T) -> T {
    T::try_from(v.max(0)).unwrap_or(max)
}

/// The timeout of a state as the graph stores it: seconds, or `-1` for none.
/// `0` is "no timeout" too (a zero timeout would expire the state on the spot).
fn timeout_lu(stocke: i64) -> Option<u64> {
    u64::try_from(stocke).ok().filter(|&s| s > 0)
}

fn timeout_ecrit(timeout: Option<u64>) -> i64 {
    match timeout {
        None | Some(0) => -1,
        Some(v) => i64::try_from(v).unwrap_or(i64::MAX),
    }
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

/// Every node of a label, decoded. A draft (a creation that never reached its
/// commit point) is deleted and counted; a node that does not decode is named in
/// the report and left untouched.
pub(super) fn charger<T>(
    db: &ObrainDB,
    label: &str,
    rapport: &mut RapportOuverture,
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
            Err(e) => rapport
                .en_quarantaine
                .push(format!("nœud {label} {noeud:?} : {e}")),
        }
    }
    fiches
}

/// Writes a creation: the node, the draft marker, the properties, the commit
/// point (the marker removed), then one durable commit. Returns the node read back.
pub(super) fn creer<T>(
    w: &mut Ecriture<'_>,
    label: &str,
    proprietes: Vec<(&'static str, Value)>,
    decoder: &dyn Fn(Lecteur) -> Result<T>,
) -> Result<Fiche<T>> {
    w.flush_complet = a_un_blob(&proprietes);
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
    let fiche = w.apres_ecriture(relire(w.db, noeud, decoder))?;
    w.rendre_durable()?;
    Ok(fiche)
}

/// Overwrites properties of an existing node, then one durable commit. A value
/// that sat in a blob column and is replaced forces the full flush (as in
/// `project.rs`), hence `ancien_en_blob`.
pub(super) fn reecrire<T>(
    w: &mut Ecriture<'_>,
    noeud: NodeId,
    proprietes: Vec<(&'static str, Value)>,
    ancien_en_blob: bool,
    decoder: &dyn Fn(Lecteur) -> Result<T>,
) -> Result<Fiche<T>> {
    w.flush_complet = ancien_en_blob || a_un_blob(&proprietes);
    for (cle, valeur) in proprietes {
        w.etape()?;
        w.db.set_node_property(noeud, cle, valeur);
    }
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

/// Deletes one node with its edges (`DETACH DELETE`): for nodes this backend
/// did not write (a migrated base links by edge), or does not serve.
pub(super) fn supprimer_detache(w: &mut Ecriture<'_>, noeud: NodeId) -> Result<()> {
    let store = w.db.store();
    let aretes: Vec<_> = store
        .edges_from(noeud, Direction::Outgoing)
        .into_iter()
        .chain(store.edges_from(noeud, Direction::Incoming))
        .map(|(_, a)| a)
        .collect();
    for a in aretes {
        w.etape()?;
        w.db.delete_edge(a);
    }
    supprimer(w, noeud)
}

// ============================================================================
// Encoding and decoding of the three entities
// ============================================================================

fn texte(s: impl Into<String>) -> Value {
    Value::from(s.into())
}

fn json<T: serde::Serialize>(v: &Option<T>, quoi: &str) -> Result<Value> {
    Ok(texte(match v {
        Some(v) => serde_json::to_string(v).map_err(|e| anyhow!("{quoi} : sérialisation : {e}"))?,
        None => String::new(),
    }))
}

fn proprietes_protocole(p: &Protocol) -> Result<Vec<(&'static str, Value)>> {
    Ok(vec![
        ("id", texte(p.id.to_string())),
        ("name", texte(p.name.clone())),
        ("description", texte(p.description.clone())),
        (
            "project_id",
            texte(p.project_id.map(|u| u.to_string()).unwrap_or_default()),
        ),
        (
            "skill_id",
            texte(p.skill_id.map(|u| u.to_string()).unwrap_or_default()),
        ),
        ("entry_state", texte(p.entry_state.to_string())),
        (
            "terminal_states_json",
            texte(serde_json::to_string(&p.terminal_states)?),
        ),
        ("protocol_category", texte(p.protocol_category.to_string())),
        ("trigger_mode", texte(p.trigger_mode.to_string())),
        (
            "trigger_config_json",
            json(&p.trigger_config, "trigger_config")?,
        ),
        (
            "relevance_vector_json",
            json(&p.relevance_vector, "relevance_vector")?,
        ),
        (
            "last_triggered_at",
            texte(
                p.last_triggered_at
                    .map(|d| d.to_rfc3339())
                    .unwrap_or_default(),
            ),
        ),
        ("created_at", texte(p.created_at.to_rfc3339())),
        ("updated_at", texte(p.updated_at.to_rfc3339())),
    ])
}

fn lire_protocole(get: Lecteur) -> Result<Protocol> {
    let id = uuid_requis(get, "id", "nœud Protocol")?;
    let quoi = format!("protocole {id}");
    let quoi = quoi.as_str();
    Ok(Protocol {
        id,
        name: requise(get, "name", quoi)?,
        description: chaine(get, "description").unwrap_or_default(),
        project_id: uuid_facultatif(get, "project_id", quoi)?,
        skill_id: uuid_facultatif(get, "skill_id", quoi)?,
        entry_state: uuid_requis(get, "entry_state", quoi)?,
        terminal_states: json_attendu(get, "terminal_states_json", quoi)?.unwrap_or_default(),
        protocol_category: enumeration(get, "protocol_category", quoi)?,
        trigger_mode: enumeration(get, "trigger_mode", quoi)?,
        trigger_config: json_attendu(get, "trigger_config_json", quoi)?,
        relevance_vector: json_attendu(get, "relevance_vector_json", quoi)?,
        last_triggered_at: date_facultative(get, "last_triggered_at", quoi)?,
        created_at: date_requise(get, "created_at", quoi)?,
        updated_at: date_requise(get, "updated_at", quoi)?,
    })
}

fn proprietes_etat(s: &ProtocolState) -> Result<Vec<(&'static str, Value)>> {
    Ok(vec![
        ("id", texte(s.id.to_string())),
        ("protocol_id", texte(s.protocol_id.to_string())),
        ("name", texte(s.name.clone())),
        ("description", texte(s.description.clone())),
        ("action", texte(s.action.clone().unwrap_or_default())),
        ("state_type", texte(s.state_type.to_string())),
        (
            "sub_protocol_id",
            texte(s.sub_protocol_id.map(|u| u.to_string()).unwrap_or_default()),
        ),
        (
            "completion_strategy",
            texte(
                s.completion_strategy
                    .map(|c| c.to_string())
                    .unwrap_or_default(),
            ),
        ),
        (
            "on_failure_strategy",
            texte(
                s.on_failure_strategy
                    .map(|c| c.to_string())
                    .unwrap_or_default(),
            ),
        ),
        (
            "generator_config_json",
            json(&s.generator_config, "generator_config")?,
        ),
        (
            "prompt_fragment",
            texte(s.prompt_fragment.clone().unwrap_or_default()),
        ),
        (
            "available_tools_json",
            json(&s.available_tools, "available_tools")?,
        ),
        (
            "forbidden_actions_json",
            json(&s.forbidden_actions, "forbidden_actions")?,
        ),
        (
            "state_timeout_secs",
            Value::from(timeout_ecrit(s.state_timeout_secs)),
        ),
    ])
}

fn lire_etat(get: Lecteur) -> Result<ProtocolState> {
    let id = uuid_requis(get, "id", "nœud ProtocolState")?;
    let quoi = format!("état {id}");
    let quoi = quoi.as_str();
    Ok(ProtocolState {
        id,
        protocol_id: uuid_requis(get, "protocol_id", quoi)?,
        name: requise(get, "name", quoi)?,
        description: chaine(get, "description").unwrap_or_default(),
        action: facultative(get, "action"),
        state_type: enumeration(get, "state_type", quoi)?,
        sub_protocol_id: uuid_facultatif(get, "sub_protocol_id", quoi)?,
        completion_strategy: enumeration_facultative(get, "completion_strategy", quoi)?,
        on_failure_strategy: enumeration_facultative(get, "on_failure_strategy", quoi)?,
        // Keys that came later than the label: absent is "none" (module doc).
        generator_config: json_tardif(get, "generator_config_json", quoi)?,
        prompt_fragment: texte_tardif(get, "prompt_fragment", quoi)?,
        available_tools: json_tardif(get, "available_tools_json", quoi)?,
        forbidden_actions: json_tardif(get, "forbidden_actions_json", quoi)?,
        state_timeout_secs: get("state_timeout_secs")
            .and_then(|v| v.as_int64())
            .and_then(timeout_lu),
    })
}

fn proprietes_transition(t: &ProtocolTransition) -> Result<Vec<(&'static str, Value)>> {
    Ok(vec![
        ("id", texte(t.id.to_string())),
        ("protocol_id", texte(t.protocol_id.to_string())),
        ("from_state", texte(t.from_state.to_string())),
        ("to_state", texte(t.to_state.to_string())),
        ("trigger", texte(t.trigger.clone())),
        ("guard", texte(t.guard.clone().unwrap_or_default())),
        ("guard_expr", json(&t.guard_expr, "guard_expr")?),
    ])
}

fn lire_transition(get: Lecteur) -> Result<ProtocolTransition> {
    let id = uuid_requis(get, "id", "nœud ProtocolTransition")?;
    let quoi = format!("transition {id}");
    let quoi = quoi.as_str();
    Ok(ProtocolTransition {
        id,
        protocol_id: uuid_requis(get, "protocol_id", quoi)?,
        from_state: uuid_requis(get, "from_state", quoi)?,
        to_state: uuid_requis(get, "to_state", quoi)?,
        trigger: requise(get, "trigger", quoi)?,
        // Fail closed: a guard lost (absent) or that no longer parses is an
        // error, never a dropped guard (that would open a guarded transition).
        guard: texte_attendu(get, "guard", quoi)?,
        // `guard_expr` came later (2026-11-07): absent is "no expression", as in
        // Neo4j; present and unreadable is an error.
        guard_expr: json_tardif(get, "guard_expr", quoi)?,
    })
}

// ============================================================================
// The index
// ============================================================================

/// The protocol domain's index: decoded entities by id, rebuilt at every open.
#[derive(Default)]
pub(super) struct IndexProtocoles {
    pub(super) protocoles: HashMap<Uuid, Fiche<Protocol>>,
    pub(super) etats: HashMap<Uuid, Fiche<ProtocolState>>,
    pub(super) transitions: HashMap<Uuid, Fiche<ProtocolTransition>>,
    /// State and transition nodes on disk the index does not serve, with their
    /// raw `protocol_id`: `delete_protocol` still reaches them.
    hors_index: Vec<(NodeId, Option<Uuid>)>,
}

impl IndexProtocoles {
    /// Rebuilds the index from the nodes. Drafts are deleted; what does not
    /// decode, repeats an id, or names a protocol that is not there is named in
    /// the report and left untouched.
    pub(super) fn reconstruire(db: &ObrainDB, rapport: &mut RapportOuverture) -> Self {
        let mut index = Self::default();
        for f in charger(db, PROTOCOLE, rapport, &lire_protocole) {
            if index.protocoles.contains_key(&f.valeur.id) {
                rapport.en_quarantaine.push(format!(
                    "nœud {PROTOCOLE} {:?} : id {} déjà pris",
                    f.noeud, f.valeur.id
                ));
                continue;
            }
            index.protocoles.insert(f.valeur.id, f);
        }
        for f in charger(db, ETAT, rapport, &lire_etat) {
            if !index.protocoles.contains_key(&f.valeur.protocol_id) {
                rapport.en_quarantaine.push(format!(
                    "nœud {ETAT} {:?} : le protocole {} n'existe pas",
                    f.noeud, f.valeur.protocol_id
                ));
            } else {
                match index.etats.entry(f.valeur.id) {
                    Entry::Occupied(_) => rapport.en_quarantaine.push(format!(
                        "nœud {ETAT} {:?} : id {} déjà pris",
                        f.noeud, f.valeur.id
                    )),
                    Entry::Vacant(v) => {
                        v.insert(f);
                    }
                }
            }
        }
        for f in charger(db, TRANSITION, rapport, &lire_transition) {
            if !index.protocoles.contains_key(&f.valeur.protocol_id) {
                rapport.en_quarantaine.push(format!(
                    "nœud {TRANSITION} {:?} : le protocole {} n'existe pas",
                    f.noeud, f.valeur.protocol_id
                ));
            } else {
                match index.transitions.entry(f.valeur.id) {
                    Entry::Occupied(_) => rapport.en_quarantaine.push(format!(
                        "nœud {TRANSITION} {:?} : id {} déjà pris",
                        f.noeud, f.valeur.id
                    )),
                    Entry::Vacant(v) => {
                        v.insert(f);
                    }
                }
            }
        }
        let servis: std::collections::HashSet<NodeId> = index
            .etats
            .values()
            .map(|f| f.noeud)
            .chain(index.transitions.values().map(|f| f.noeud))
            .collect();
        for label in [ETAT, TRANSITION] {
            for noeud in db.store().nodes_by_label(label) {
                if servis.contains(&noeud) {
                    continue;
                }
                let protocole = db.get_node(noeud).and_then(|n| {
                    n.get_property("protocol_id")
                        .and_then(|v| v.as_str().and_then(|t| t.parse::<Uuid>().ok()))
                });
                index.hors_index.push((noeud, protocole));
            }
        }
        index
    }
}

// ============================================================================
// The trait
// ============================================================================

#[async_trait]
impl ProtocolStore for ObrainGraphStore {
    async fn upsert_protocol(&self, protocol: &Protocol) -> Result<()> {
        let p = protocol.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let proprietes = proprietes_protocole(&p)?;
                let existant = {
                    let etat = w.index()?;
                    if let Some(projet) = p.project_id {
                        if !etat.projets.contient(projet) {
                            bail!(
                                "le projet {projet} n'existe pas : le protocole {} ne peut pas y \
                                 être rattaché (le backend obrain n'écrit pas un protocole \
                                 orphelin ; Neo4j n'écrit alors rien, sans le dire)",
                                p.id
                            );
                        }
                    }
                    etat.protocoles
                        .protocoles
                        .get(&p.id)
                        .map(|f| (f.noeud, f.valeur.clone()))
                };
                let fiche = match existant {
                    None => creer(w, PROTOCOLE, proprietes, &lire_protocole)?,
                    Some((noeud, ancien)) => {
                        let ancien_en_blob = a_un_blob(&proprietes_protocole(&ancien)?);
                        // `created_at` is set on creation only (`ON CREATE SET`).
                        let maj = proprietes
                            .into_iter()
                            .filter(|(k, _)| !matches!(*k, "id" | "created_at"))
                            .collect();
                        reecrire(w, noeud, maj, ancien_en_blob, &lire_protocole)?
                    }
                };
                w.index_mut()?.protocoles.protocoles.insert(p.id, fiche);
                Ok(())
            },
        )
        .await
    }

    async fn get_protocol(&self, id: Uuid) -> Result<Option<Protocol>> {
        self.lire(|_, etat| {
            Ok(etat
                .protocoles
                .protocoles
                .get(&id)
                .map(|f| f.valeur.clone()))
        })
    }

    async fn list_protocols(
        &self,
        project_id: Option<Uuid>,
        category: Option<ProtocolCategory>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Protocol>, usize)> {
        self.lire(|_, etat| {
            // A project sees its own protocols AND the transverse ones (no project);
            // no project sees only the transverse ones.
            let mut liste: Vec<&Protocol> = etat
                .protocoles
                .protocoles
                .values()
                .map(|f| &f.valeur)
                .filter(|p| {
                    p.project_id.is_none() || (project_id.is_some() && p.project_id == project_id)
                })
                .filter(|p| category.is_none_or(|c| p.protocol_category == c))
                .collect();
            // ORDER BY updated_at DESC, id DESC: total.
            liste.sort_by(|a, b| {
                b.updated_at
                    .cmp(&a.updated_at)
                    .then_with(|| b.id.cmp(&a.id))
            });
            let total = liste.len();
            Ok((
                liste
                    .into_iter()
                    .skip(offset)
                    .take(limit)
                    .cloned()
                    .collect(),
                total,
            ))
        })
    }

    async fn delete_protocol(&self, id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, etats, transitions, runs, etats_runtime, detaches) = {
                    let etat = w.index()?;
                    let Some(f) = etat.protocoles.protocoles.get(&id) else {
                        return Ok(false);
                    };
                    // What the index does not serve, or reaches only by edge (module
                    // doc): deleted WITH its edges, as Neo4j's DETACH DELETE.
                    let mut detaches: Vec<NodeId> = etat
                        .protocoles
                        .hors_index
                        .iter()
                        .filter(|(_, p)| *p == Some(id))
                        .map(|(n, _)| *n)
                        .collect();
                    let runs_hors_index = etat.runs.hors_index_du_protocole(id);
                    detaches.extend(runs_hors_index.iter().map(|(_, n)| *n));
                    let store = w.db.store();
                    for (voisin, arete) in store
                        .edges_from(f.noeud, Direction::Outgoing)
                        .into_iter()
                        .chain(store.edges_from(f.noeud, Direction::Incoming))
                    {
                        let Some(e) = w.db.get_edge(arete) else {
                            continue;
                        };
                        let sortante = e.src == f.noeud;
                        let suivie = match &*e.edge_type {
                            "HAS_STATE" | "HAS_TRANSITION" => sortante,
                            "INSTANCE_OF" => !sortante,
                            _ => false,
                        };
                        if suivie && voisin != f.noeud {
                            detaches.push(voisin);
                        }
                    }
                    let etats: Vec<(Uuid, NodeId)> = etat
                        .protocoles
                        .etats
                        .iter()
                        .filter(|(_, e)| e.valeur.protocol_id == id)
                        .map(|(i, e)| (*i, e.noeud))
                        .collect();
                    let transitions: Vec<(Uuid, NodeId)> = etat
                        .protocoles
                        .transitions
                        .iter()
                        .filter(|(_, t)| t.valeur.protocol_id == id)
                        .map(|(i, t)| (*i, t.noeud))
                        .collect();
                    let (runs, etats_runtime) = etat.runs.du_protocole(id);
                    // The runtime states of every run reached, served or not.
                    let ids_runs: std::collections::HashSet<Uuid> = runs
                        .iter()
                        .map(|(i, _)| *i)
                        .chain(runs_hors_index.iter().filter_map(|(i, _)| *i))
                        .collect();
                    detaches.extend(etat.runs.etats_hors_index_des(&ids_runs));
                    // A node the index serves is deleted in its own place below.
                    let servis: std::collections::HashSet<NodeId> = etats
                        .iter()
                        .chain(&transitions)
                        .chain(&runs)
                        .chain(&etats_runtime)
                        .map(|(_, n)| *n)
                        .collect();
                    let mut vus = std::collections::HashSet::new();
                    detaches.retain(|n| !servis.contains(n) && vus.insert(*n));
                    (f.noeud, etats, transitions, runs, etats_runtime, detaches)
                };
                // Leaves first, the protocol node last (module doc).
                for (_, n) in &etats_runtime {
                    supprimer_detache(w, *n)?;
                }
                for n in &detaches {
                    supprimer_detache(w, *n)?;
                }
                for (_, n) in &runs {
                    supprimer_detache(w, *n)?;
                }
                for (_, n) in &etats {
                    supprimer_detache(w, *n)?;
                }
                for (_, n) in &transitions {
                    supprimer_detache(w, *n)?;
                }
                supprimer_detache(w, noeud)?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.runs.retirer(
                    &runs.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
                    &etats_runtime.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
                );
                for (i, _) in &etats {
                    etat.protocoles.etats.remove(i);
                }
                for (i, _) in &transitions {
                    etat.protocoles.transitions.remove(i);
                }
                let detaches: std::collections::HashSet<NodeId> = detaches.into_iter().collect();
                etat.protocoles
                    .hors_index
                    .retain(|(n, _)| !detaches.contains(n));
                etat.runs.oublier_hors_index(&detaches);
                etat.protocoles.protocoles.remove(&id);
                Ok(true)
            },
        )
        .await
    }

    async fn upsert_protocol_state(&self, state: &ProtocolState) -> Result<()> {
        let s = state.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let proprietes = proprietes_etat(&s)?;
                let existant = {
                    let etat = w.index()?;
                    if !etat.protocoles.protocoles.contains_key(&s.protocol_id) {
                        bail!(
                            "le protocole {} n'existe pas : l'état {} ne peut pas lui être \
                             rattaché (Neo4j n'écrit alors rien, sans le dire)",
                            s.protocol_id,
                            s.id
                        );
                    }
                    etat.protocoles
                        .etats
                        .get(&s.id)
                        .map(|f| (f.noeud, f.valeur.clone()))
                };
                let fiche = match existant {
                    None => creer(w, ETAT, proprietes, &lire_etat)?,
                    Some((noeud, ancien)) => {
                        let ancien_en_blob = a_un_blob(&proprietes_etat(&ancien)?);
                        let maj = proprietes.into_iter().filter(|(k, _)| *k != "id").collect();
                        reecrire(w, noeud, maj, ancien_en_blob, &lire_etat)?
                    }
                };
                w.index_mut()?.protocoles.etats.insert(s.id, fiche);
                Ok(())
            },
        )
        .await
    }

    async fn get_protocol_states(&self, protocol_id: Uuid) -> Result<Vec<ProtocolState>> {
        self.lire(|_, etat| {
            let mut etats: Vec<ProtocolState> = etat
                .protocoles
                .etats
                .values()
                .filter(|f| f.valeur.protocol_id == protocol_id)
                .map(|f| f.valeur.clone())
                .collect();
            // ORDER BY state_type, name, id ASC (the type as its stored spelling).
            etats.sort_by(|a, b| {
                a.state_type
                    .to_string()
                    .cmp(&b.state_type.to_string())
                    .then_with(|| a.name.cmp(&b.name))
                    .then_with(|| a.id.cmp(&b.id))
            });
            Ok(etats)
        })
    }

    async fn get_protocol_by_name_and_project(
        &self,
        name: &str,
        project_id: Uuid,
    ) -> Result<Option<Uuid>> {
        self.lire(|_, etat| {
            // Neo4j takes any one match (`LIMIT 1`); this takes the smallest id,
            // which is deterministic.
            Ok(etat
                .protocoles
                .protocoles
                .values()
                .map(|f| &f.valeur)
                .filter(|p| p.name == name && p.project_id == Some(project_id))
                .map(|p| p.id)
                .min())
        })
    }

    async fn delete_protocol_state(&self, state_id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = w.index()?.protocoles.etats.get(&state_id).map(|f| f.noeud)
                else {
                    return Ok(false);
                };
                supprimer(w, noeud)?;
                w.rendre_durable()?;
                w.index_mut()?.protocoles.etats.remove(&state_id);
                Ok(true)
            },
        )
        .await
    }

    async fn upsert_protocol_transition(&self, transition: &ProtocolTransition) -> Result<()> {
        let t = transition.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let proprietes = proprietes_transition(&t)?;
                let existant = {
                    let etat = w.index()?;
                    if !etat.protocoles.protocoles.contains_key(&t.protocol_id) {
                        bail!(
                            "le protocole {} n'existe pas : la transition {} ne peut pas lui \
                             être rattachée (Neo4j n'écrit alors rien, sans le dire)",
                            t.protocol_id,
                            t.id
                        );
                    }
                    etat.protocoles
                        .transitions
                        .get(&t.id)
                        .map(|f| (f.noeud, f.valeur.clone()))
                };
                let fiche = match existant {
                    None => creer(w, TRANSITION, proprietes, &lire_transition)?,
                    Some((noeud, ancien)) => {
                        let ancien_en_blob = a_un_blob(&proprietes_transition(&ancien)?);
                        let maj = proprietes.into_iter().filter(|(k, _)| *k != "id").collect();
                        reecrire(w, noeud, maj, ancien_en_blob, &lire_transition)?
                    }
                };
                w.index_mut()?.protocoles.transitions.insert(t.id, fiche);
                Ok(())
            },
        )
        .await
    }

    async fn get_protocol_transitions(&self, protocol_id: Uuid) -> Result<Vec<ProtocolTransition>> {
        self.lire(|_, etat| {
            let mut transitions: Vec<ProtocolTransition> = etat
                .protocoles
                .transitions
                .values()
                .filter(|f| f.valeur.protocol_id == protocol_id)
                .map(|f| f.valeur.clone())
                .collect();
            // ORDER BY trigger, id.
            transitions.sort_by(|a, b| a.trigger.cmp(&b.trigger).then_with(|| a.id.cmp(&b.id)));
            Ok(transitions)
        })
    }

    async fn delete_protocol_transition(&self, transition_id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = w
                    .index()?
                    .protocoles
                    .transitions
                    .get(&transition_id)
                    .map(|f| f.noeud)
                else {
                    return Ok(false);
                };
                supprimer(w, noeud)?;
                w.rendre_durable()?;
                w.index_mut()?.protocoles.transitions.remove(&transition_id);
                Ok(true)
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obrain::Creation;
    use crate::protocol::StateType;

    fn protocole(nom: &str, projet: Option<Uuid>) -> Protocol {
        Protocol::new(projet, nom, Uuid::new_v4())
    }

    fn etat(protocole: &Protocol, nom: &str, t: StateType) -> ProtocolState {
        let mut s = ProtocolState::new(protocole.id, nom);
        s.state_type = t;
        s
    }

    #[test]
    fn timeouts_never_wrap() {
        assert_eq!(timeout_lu(-1), None);
        assert_eq!(timeout_lu(0), None);
        assert_eq!(timeout_lu(255), Some(255));
        assert_eq!(timeout_ecrit(None), -1);
        assert_eq!(timeout_ecrit(Some(0)), -1);
        assert_eq!(timeout_ecrit(Some(u64::MAX)), i64::MAX);
    }

    #[tokio::test]
    async fn cycle_de_vie_etats_transitions_et_cascade() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let p = protocole("p", None);
        store.upsert_protocol(&p).await.unwrap();
        let a = etat(&p, "a", StateType::Start);
        let b = etat(&p, "b", StateType::Terminal);
        for s in [&a, &b] {
            store.upsert_protocol_state(s).await.unwrap();
        }
        let t = ProtocolTransition::new(p.id, a.id, b.id, "go");
        store.upsert_protocol_transition(&t).await.unwrap();
        assert_eq!(store.get_protocol_states(p.id).await.unwrap().len(), 2);
        assert_eq!(
            store.get_protocol_transitions(p.id).await.unwrap()[0].trigger,
            "go"
        );
        assert!(store.delete_protocol_state(b.id).await.unwrap());
        assert!(!store.delete_protocol_state(b.id).await.unwrap());
        assert!(store.delete_protocol_transition(t.id).await.unwrap());
        assert!(!store.delete_protocol_transition(t.id).await.unwrap());
        // The cascade takes the remaining state; a second delete finds nothing.
        assert!(store.delete_protocol(p.id).await.unwrap());
        assert!(!store.delete_protocol(p.id).await.unwrap());
        assert!(store.get_protocol_states(p.id).await.unwrap().is_empty());
        assert_eq!(store.interne.db.store().node_count(), 0);
    }

    #[tokio::test]
    async fn un_etat_ou_une_transition_sans_protocole_est_refuse() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let p = protocole("fantome", None);
        let s = etat(&p, "s", StateType::Start);
        assert!(store.upsert_protocol_state(&s).await.is_err());
        let rattache = protocole("orphelin", Some(Uuid::new_v4()));
        assert!(store.upsert_protocol(&rattache).await.is_err());
        assert!(store.get_protocol(rattache.id).await.unwrap().is_none());
        assert_eq!(store.interne.db.store().node_count(), 0);
    }

    #[tokio::test]
    async fn l_upsert_garde_created_at_et_remplace_le_reste() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let mut p = protocole("avant", None);
        store.upsert_protocol(&p).await.unwrap();
        let cree = p.created_at;
        p.name = "après".into();
        p.created_at = cree + chrono::Duration::days(1);
        p.updated_at = cree + chrono::Duration::days(1);
        store.upsert_protocol(&p).await.unwrap();
        let relu = store.get_protocol(p.id).await.unwrap().unwrap();
        assert_eq!(relu.name, "après");
        assert_eq!(relu.created_at, cree, "ON CREATE SET: created_at is kept");
        assert_eq!(store.interne.db.store().node_count(), 1);
    }

    #[tokio::test]
    async fn la_liste_filtre_par_categorie_et_pagine() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let mut systeme = protocole("systeme", None);
        systeme.protocol_category = ProtocolCategory::System;
        let metier = protocole("metier", None);
        for p in [&systeme, &metier] {
            store.upsert_protocol(p).await.unwrap();
        }
        let (tous, total) = store.list_protocols(None, None, 10, 0).await.unwrap();
        assert_eq!((tous.len(), total), (2, 2));
        let (sys, total) = store
            .list_protocols(None, Some(ProtocolCategory::System), 10, 0)
            .await
            .unwrap();
        assert_eq!((sys.len(), total), (1, 1));
        let (page, total) = store.list_protocols(None, None, 1, 1).await.unwrap();
        assert_eq!((page.len(), total), (1, 2));
        // The name lookup is scoped to a project: a transverse protocol has none.
        assert_eq!(
            store
                .get_protocol_by_name_and_project("metier", Uuid::new_v4())
                .await
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn un_protocole_survit_a_la_reouverture_avec_ses_enfants() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = protocole("durable", None);
        let a = etat(&p, "a", StateType::Start);
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.upsert_protocol(&p).await.unwrap();
            store.upsert_protocol_state(&a).await.unwrap();
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(
            store.get_protocol(p.id).await.unwrap().unwrap().name,
            "durable"
        );
        assert_eq!(store.get_protocol_states(p.id).await.unwrap()[0].id, a.id);
        assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
    }
}
