//! `ProtocolStore` et `ProtocolRunStore` sur obrain — le domaine « protocole ».
//!
//! La référence est `src/neo4j/protocol.rs`. Les valeurs, les noms de propriétés et les
//! conventions de codage sont ceux de la référence : une absence s'écrit `""` (ou
//! `[]`/`{}` pour du JSON) et se relit `None`; un JSON illisible se relit comme absent,
//! sauf le `guard_expr` d'une transition, qui échoue fermé (voir plus bas).
//!
//! # Stockage
//!
//! Un nœud par entité : `Protocol`, `ProtocolState`, `ProtocolTransition`,
//! `ProtocolRun`, `RuntimeState`. Les liens de la référence (HAS_STATE,
//! HAS_TRANSITION, INSTANCE_OF, CHILD_OF, HAS_RUNTIME_STATE, GENERATED_BY) n'existent
//! pas sur obrain : ils sont portés par les propriétés `protocol_id`, `run_id`,
//! `parent_run_id`, et par les index mémoire reconstruits à l'ouverture.
//!
//! # Écriture
//!
//! Comme `project.rs`, et pour la même raison (obrain n'a pas de transaction
//! multi-propriétés) : une CRÉATION pose le marqueur `_brouillon` en premier, les
//! propriétés ensuite, et retire le marqueur en dernier — ce retrait est le point de
//! validation. L'ouverture supprime les brouillons. Une valeur en colonne blob (chaîne
//! au-delà de [`SEUIL_COLONNE_BLOB`]) impose le flush complet avant le retrait du
//! marqueur, comme pour un projet.
//!
//! Une MISE À JOUR n'est pas atomique entre propriétés : un crash peut laisser certaines
//! propriétés anciennes et d'autres nouvelles. Chaque état reste lisible, mais ce n'est
//! pas une transaction. C'est la même limite que `ProjectStore::update_project`.
//!
//! La suppression d'un protocole retire d'abord le protocole (point de validation) puis
//! ses états, transitions, runs et runtime states. Un crash entre les deux laisse des
//! orphelins, que l'ouverture supprime (ils ne sont adressés par aucun protocole).
//!
//! Les lectures sont servies depuis l'index mémoire ; une écriture n'y est publiée qu'une
//! fois durable.
//!
//! # Ordre
//!
//! Chaque liste est triée par une clé TOTALE, celle de la référence, avec l'id comme
//! départage. Les horodatages sont comparés comme chaînes (ce que fait Neo4j sur les
//! valeurs RFC 3339 écrites par ce backend).
//!
//! # Écarts assumés avec la référence
//!
//! - Un UUID de run ou de runtime state déjà présent est REFUSÉ (la référence fait un
//!   `CREATE` et duplique le nœud, ce qui casse ensuite toute lecture par id).
//! - Les arêtes BELONGS_TO_SKILL, LINKED_TO_PLAN, LINKED_TO_TASK ne sont pas écrites :
//!   les domaines Skill, Plan et Task ne sont pas portés. Ce sont des liens « au mieux »
//!   dans la référence, et aucune lecture de ce domaine ne les suit.
//! - Le marqueur `state_timeout_explicit` (migration propre au backend Neo4j) n'est pas
//!   écrit : obrain n'a pas de valeur fantôme héritée à réparer.
//! - Les horodatages absents ou illisibles font refuser le nœud à l'ouverture (la
//!   référence les remplace par « maintenant »).
//! - Les nœuds illisibles sont exclus des listes et du total (la référence les compte
//!   dans le total).
//!
//! # Silences conservés de la référence
//!
//! Ces appels ne créent rien et répondent `Ok`, comme la référence : `upsert_protocol`
//! dont le projet n'existe pas, `upsert_protocol_state` / `upsert_protocol_transition`
//! dont le protocole n'existe pas, `create_runtime_state` dont le run ou l'état n'existe
//! pas. Ce sont des choix écrits dans la référence, pas des faux succès introduits ici.
//!
//! # Refusés par nom
//!
//! - `create_produced_during`, `get_run_outcomes` : Note et Decision ne sont pas portés.
//! - `persist_reasoning_tree`, `get_run_reasoning_tree_id` : le domaine Reasoning n'est
//!   pas porté.
//! - `list_all_protocol_runs` avec `workspace_slug` sans `project_id` : la relation
//!   projet-workspace relève du domaine Workspace.
//!
//! # Coûts
//!
//! Lectures par id, par nom, par run et par parent : une recherche dans un index, sans
//! balayage. Les listes paginées (`list_protocols`, `list_all_protocol_runs`,
//! `list_completed_runs_for_project`) trient en mémoire ; leur coût croît avec le nombre
//! d'entités du périmètre. Un index trié par clé de tri les rendrait indépendantes de N :
//! c'est le reliquat de ce lot, pas une propriété acquise.

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::str::FromStr;

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use serde::de::DeserializeOwned;
use serde::Serialize;
use uuid::Uuid;

use super::{Ecriture, NonPorte, ObrainGraphStore, SEUIL_COLONNE_BLOB};
use crate::neo4j::traits::{ProtocolRunStore, ProtocolStore};
use crate::protocol::{
    Protocol, ProtocolCategory, ProtocolRun, ProtocolState, ProtocolTransition, RunStatus,
    RuntimeState, StateVisit,
};

const PROTOCOLE: &str = "Protocol";
const ETAT: &str = "ProtocolState";
const TRANSITION: &str = "ProtocolTransition";
const RUN: &str = "ProtocolRun";
const RUNTIME: &str = "RuntimeState";

/// Le marqueur de brouillon : présent du premier au dernier enregistrement d'une création.
const BROUILLON: &str = "_brouillon";

/// Le message de la référence quand un run ne peut pas être créé.
const MESSAGE_NON_CREABLE: &str = "Skipped: concurrent run already running for protocol";

/// Profondeur maximale de l'arbre de runs, comme `CHILD_OF*1..5` de la référence.
const PROFONDEUR_ARBRE: usize = 5;

type Lecture<'a> = &'a dyn Fn(&str) -> Option<Value>;
type Props = Vec<(&'static str, Value)>;

// ============================================================================
// Codage
// ============================================================================

fn en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
}

/// `Option<Uuid>` → chaîne, `""` pour l'absence.
fn v_uuid(u: Option<Uuid>) -> Value {
    Value::from(u.map(|u| u.to_string()).unwrap_or_default())
}

/// `Option<&T>` → JSON, `""` pour l'absence (une erreur de sérialisation donne `""`, comme la référence).
fn v_json<T: Serialize>(v: Option<&T>) -> Value {
    Value::from(
        v.map(|x| serde_json::to_string(x).unwrap_or_default())
            .unwrap_or_default(),
    )
}

fn v_instant(d: Option<DateTime<Utc>>) -> Value {
    Value::from(d.map(|d| d.to_rfc3339()).unwrap_or_default())
}

fn v_texte_opt(s: &Option<String>) -> Value {
    Value::from(s.clone().unwrap_or_default())
}

/// Un timeout de state tel qu'il est stocké : secondes, ou `-1` pour « pas de timeout ».
/// `None` et `0` sont `-1`; un nombre au-delà de `i64::MAX` sature (ne se replie pas en négatif).
fn timeout_ecrit(t: Option<u64>) -> i64 {
    match t {
        None | Some(0) => -1,
        Some(v) => i64::try_from(v).unwrap_or(i64::MAX),
    }
}

/// L'inverse : un entier négatif ou nul est « pas de timeout » (`None`).
fn timeout_lu(stocke: i64) -> Option<u64> {
    u64::try_from(stocke).ok().filter(|&s| s > 0)
}

/// Un compteur lu d'un entier signé : négatif → 0, trop grand → `max` (même règle que `neo4j::unsigned`).
fn non_signe<T: TryFrom<i64>>(v: i64, max: T) -> T {
    T::try_from(v.max(0)).unwrap_or(max)
}

fn touche_blob(props: &Props) -> bool {
    props.iter().any(|(_, v)| en_colonne_blob(v))
}

// ----------------------------------------------------------------------------
// Décodage (les mêmes règles que `Neo4jClient::node_to_*`)
// ----------------------------------------------------------------------------

fn chaine(get: Lecture<'_>, cle: &str) -> Option<String> {
    get(cle).and_then(|v| v.as_str().map(str::to_string))
}

fn chaine_non_vide(get: Lecture<'_>, cle: &str) -> Option<String> {
    chaine(get, cle).filter(|s| !s.is_empty())
}

fn texte_requis(get: Lecture<'_>, cle: &str) -> Result<String> {
    chaine(get, cle).ok_or_else(|| anyhow!("propriété `{cle}` absente"))
}

fn uuid_requis(get: Lecture<'_>, cle: &str) -> Result<Uuid> {
    let s = texte_requis(get, cle)?;
    s.parse::<Uuid>()
        .with_context(|| format!("propriété `{cle}` : UUID illisible « {s} »"))
}

fn uuid_non_vide(get: Lecture<'_>, cle: &str) -> Option<Uuid> {
    chaine_non_vide(get, cle).and_then(|s| s.parse().ok())
}

fn json_non_vide<T: DeserializeOwned>(get: Lecture<'_>, cle: &str) -> Option<T> {
    chaine_non_vide(get, cle).and_then(|s| serde_json::from_str(&s).ok())
}

fn enum_ou_defaut<T: FromStr + Default>(get: Lecture<'_>, cle: &str) -> T {
    chaine(get, cle)
        .and_then(|s| s.parse().ok())
        .unwrap_or_default()
}

fn enum_non_vide<T: FromStr>(get: Lecture<'_>, cle: &str) -> Option<T> {
    chaine_non_vide(get, cle).and_then(|s| s.parse().ok())
}

fn instant_requis(get: Lecture<'_>, cle: &str) -> Result<DateTime<Utc>> {
    let s = texte_requis(get, cle)?;
    s.parse::<DateTime<Utc>>()
        .with_context(|| format!("propriété `{cle}` : instant illisible « {s} »"))
}

fn instant_non_vide(get: Lecture<'_>, cle: &str) -> Option<DateTime<Utc>> {
    chaine_non_vide(get, cle).and_then(|s| s.parse().ok())
}

fn entier(get: Lecture<'_>, cle: &str) -> i64 {
    get(cle).and_then(|v| v.as_int64()).unwrap_or(0)
}

fn lire_protocole(get: Lecture<'_>) -> Result<Protocol> {
    let terminal_states = chaine(get, "terminal_states_json")
        .and_then(|s| serde_json::from_str::<Vec<Uuid>>(&s).ok())
        .unwrap_or_default();
    Ok(Protocol {
        id: uuid_requis(get, "id")?,
        name: texte_requis(get, "name")?,
        description: chaine(get, "description").unwrap_or_default(),
        project_id: uuid_non_vide(get, "project_id"),
        skill_id: uuid_non_vide(get, "skill_id"),
        entry_state: uuid_requis(get, "entry_state")?,
        terminal_states,
        protocol_category: enum_ou_defaut(get, "protocol_category"),
        trigger_mode: enum_ou_defaut(get, "trigger_mode"),
        trigger_config: json_non_vide(get, "trigger_config_json"),
        relevance_vector: json_non_vide(get, "relevance_vector_json"),
        last_triggered_at: instant_non_vide(get, "last_triggered_at"),
        created_at: instant_requis(get, "created_at")?,
        updated_at: instant_requis(get, "updated_at")?,
    })
}

fn lire_etat(get: Lecture<'_>) -> Result<ProtocolState> {
    Ok(ProtocolState {
        id: uuid_requis(get, "id")?,
        protocol_id: uuid_requis(get, "protocol_id")?,
        name: texte_requis(get, "name")?,
        description: chaine(get, "description").unwrap_or_default(),
        action: chaine_non_vide(get, "action"),
        state_type: enum_ou_defaut(get, "state_type"),
        sub_protocol_id: uuid_non_vide(get, "sub_protocol_id"),
        completion_strategy: enum_non_vide(get, "completion_strategy"),
        on_failure_strategy: enum_non_vide(get, "on_failure_strategy"),
        generator_config: json_non_vide(get, "generator_config_json"),
        prompt_fragment: chaine_non_vide(get, "prompt_fragment"),
        available_tools: json_non_vide(get, "available_tools_json"),
        forbidden_actions: json_non_vide(get, "forbidden_actions_json"),
        state_timeout_secs: get("state_timeout_secs")
            .and_then(|v| v.as_int64())
            .and_then(timeout_lu),
    })
}

fn lire_transition(get: Lecture<'_>) -> Result<ProtocolTransition> {
    // Échec fermé : un guard stocké qui ne se relit plus est une erreur, jamais une
    // transition ouverte (la référence fait de même).
    let guard_expr: Option<crate::protocol::guard::Guard> = match chaine_non_vide(get, "guard_expr")
    {
        None => None,
        Some(s) => Some(
            serde_json::from_str(&s).context("Stored guard_expr is not a valid guard")?,
        ),
    };
    Ok(ProtocolTransition {
        id: uuid_requis(get, "id")?,
        protocol_id: uuid_requis(get, "protocol_id")?,
        from_state: uuid_requis(get, "from_state")?,
        to_state: uuid_requis(get, "to_state")?,
        trigger: texte_requis(get, "trigger")?,
        guard: chaine_non_vide(get, "guard"),
        guard_expr,
    })
}

fn lire_run(get: Lecture<'_>) -> Result<ProtocolRun> {
    Ok(ProtocolRun {
        id: uuid_requis(get, "id")?,
        protocol_id: uuid_requis(get, "protocol_id")?,
        project_id: uuid_non_vide(get, "project_id"),
        plan_id: uuid_non_vide(get, "plan_id"),
        task_id: uuid_non_vide(get, "task_id"),
        parent_run_id: uuid_non_vide(get, "parent_run_id"),
        current_state: uuid_requis(get, "current_state")?,
        states_visited: chaine(get, "states_visited_json")
            .and_then(|s| serde_json::from_str::<Vec<StateVisit>>(&s).ok())
            .unwrap_or_default(),
        status: enum_ou_defaut(get, "status"),
        started_at: instant_requis(get, "started_at")?,
        completed_at: instant_non_vide(get, "completed_at"),
        error: chaine_non_vide(get, "error"),
        triggered_by: chaine(get, "triggered_by").unwrap_or_else(|| "manual".to_string()),
        depth: non_signe(entier(get, "depth"), u32::MAX),
        version: non_signe(entier(get, "version"), u64::MAX),
        runner_managed: get("runner_managed")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    })
}

fn lire_runtime(get: Lecture<'_>) -> Result<RuntimeState> {
    Ok(RuntimeState {
        id: uuid_requis(get, "id")?,
        run_id: uuid_requis(get, "run_id")?,
        generated_by: uuid_requis(get, "generated_by")?,
        name: texte_requis(get, "name")?,
        index: non_signe(entier(get, "index"), u32::MAX),
        sub_protocol_id: uuid_non_vide(get, "sub_protocol_id"),
        action: chaine_non_vide(get, "action"),
        status: chaine(get, "status").unwrap_or_else(|| "pending".to_string()),
        next_runtime_state_id: uuid_non_vide(get, "next_runtime_state_id"),
        linking_strategy: enum_ou_defaut(get, "linking_strategy"),
    })
}

// ----------------------------------------------------------------------------
// Écriture : les propriétés de chaque entité, dans l'ordre d'écriture
// ----------------------------------------------------------------------------

fn props_protocole(p: &Protocol, creation: bool) -> Result<Props> {
    let mut v: Props = vec![
        ("id", Value::from(p.id.to_string())),
        ("name", Value::from(p.name.clone())),
        ("description", Value::from(p.description.clone())),
        ("project_id", v_uuid(p.project_id)),
        ("skill_id", v_uuid(p.skill_id)),
        ("entry_state", Value::from(p.entry_state.to_string())),
        (
            "terminal_states_json",
            Value::from(serde_json::to_string(&p.terminal_states)?),
        ),
        (
            "protocol_category",
            Value::from(p.protocol_category.to_string()),
        ),
        ("trigger_mode", Value::from(p.trigger_mode.to_string())),
        ("trigger_config_json", v_json(p.trigger_config.as_ref())),
        (
            "relevance_vector_json",
            v_json(p.relevance_vector.as_ref()),
        ),
        ("last_triggered_at", v_instant(p.last_triggered_at)),
        ("updated_at", Value::from(p.updated_at.to_rfc3339())),
    ];
    // `ON CREATE SET created_at` : une mise à jour garde la date de création.
    if creation {
        v.push(("created_at", Value::from(p.created_at.to_rfc3339())));
    }
    Ok(v)
}

fn props_etat(s: &ProtocolState) -> Props {
    vec![
        ("id", Value::from(s.id.to_string())),
        ("protocol_id", Value::from(s.protocol_id.to_string())),
        ("name", Value::from(s.name.clone())),
        ("description", Value::from(s.description.clone())),
        ("action", v_texte_opt(&s.action)),
        ("state_type", Value::from(s.state_type.to_string())),
        ("sub_protocol_id", v_uuid(s.sub_protocol_id)),
        (
            "completion_strategy",
            Value::from(
                s.completion_strategy
                    .map(|c| c.to_string())
                    .unwrap_or_default(),
            ),
        ),
        (
            "on_failure_strategy",
            Value::from(
                s.on_failure_strategy
                    .map(|c| c.to_string())
                    .unwrap_or_default(),
            ),
        ),
        ("generator_config_json", v_json(s.generator_config.as_ref())),
        (
            "prompt_fragment",
            Value::from(s.prompt_fragment.clone().unwrap_or_default()),
        ),
        ("available_tools_json", v_json(s.available_tools.as_ref())),
        (
            "forbidden_actions_json",
            v_json(s.forbidden_actions.as_ref()),
        ),
        (
            "state_timeout_secs",
            Value::from(timeout_ecrit(s.state_timeout_secs)),
        ),
    ]
}

fn props_transition(t: &ProtocolTransition) -> Result<Props> {
    let guard_expr = match &t.guard_expr {
        Some(g) => serde_json::to_string(g).context("Failed to serialize the transition guard_expr")?,
        None => String::new(),
    };
    Ok(vec![
        ("id", Value::from(t.id.to_string())),
        ("protocol_id", Value::from(t.protocol_id.to_string())),
        ("from_state", Value::from(t.from_state.to_string())),
        ("to_state", Value::from(t.to_state.to_string())),
        ("trigger", Value::from(t.trigger.clone())),
        ("guard", v_texte_opt(&t.guard)),
        ("guard_expr", Value::from(guard_expr)),
    ])
}

fn props_run_creation(r: &ProtocolRun) -> Result<Props> {
    Ok(vec![
        ("id", Value::from(r.id.to_string())),
        ("protocol_id", Value::from(r.protocol_id.to_string())),
        ("project_id", v_uuid(r.project_id)),
        ("plan_id", v_uuid(r.plan_id)),
        ("task_id", v_uuid(r.task_id)),
        ("parent_run_id", v_uuid(r.parent_run_id)),
        ("current_state", Value::from(r.current_state.to_string())),
        (
            "states_visited_json",
            Value::from(serde_json::to_string(&r.states_visited)?),
        ),
        ("status", Value::from(r.status.to_string())),
        ("started_at", Value::from(r.started_at.to_rfc3339())),
        ("completed_at", v_instant(r.completed_at)),
        ("error", v_texte_opt(&r.error)),
        ("triggered_by", Value::from(r.triggered_by.clone())),
        ("depth", Value::from(i64::from(r.depth))),
        ("version", Value::from(r.version as i64)),
        ("runner_managed", Value::from(r.runner_managed)),
    ])
}

/// Ce qu'une mise à jour de run écrit : les seules propriétés que `update_protocol_run` touche.
fn props_run_maj(r: &ProtocolRun) -> Result<Props> {
    Ok(vec![
        ("current_state", Value::from(r.current_state.to_string())),
        (
            "states_visited_json",
            Value::from(serde_json::to_string(&r.states_visited)?),
        ),
        ("status", Value::from(r.status.to_string())),
        ("completed_at", v_instant(r.completed_at)),
        ("error", v_texte_opt(&r.error)),
        ("version", Value::from(r.version as i64)),
        ("runner_managed", Value::from(r.runner_managed)),
    ])
}

fn props_runtime(s: &RuntimeState) -> Props {
    vec![
        ("id", Value::from(s.id.to_string())),
        ("run_id", Value::from(s.run_id.to_string())),
        ("generated_by", Value::from(s.generated_by.to_string())),
        ("name", Value::from(s.name.clone())),
        ("index", Value::from(i64::from(s.index))),
        ("sub_protocol_id", v_uuid(s.sub_protocol_id)),
        ("action", v_texte_opt(&s.action)),
        ("status", Value::from(s.status.clone())),
        ("next_runtime_state_id", v_uuid(s.next_runtime_state_id)),
        (
            "linking_strategy",
            Value::from(s.linking_strategy.to_string()),
        ),
    ]
}

// ============================================================================
// Index mémoire
// ============================================================================

struct Entree<T> {
    noeud: NodeId,
    valeur: T,
}

struct EntreeProtocole {
    noeud: NodeId,
    valeur: Protocol,
    /// `updated_at` tel qu'il est stocké : la clé de tri de la référence.
    maj: String,
}

struct EntreeRun {
    noeud: NodeId,
    valeur: ProtocolRun,
    /// `started_at` tel qu'il est stocké.
    debut: String,
    /// `completed_at` tel qu'il est stocké, s'il est là.
    fin: Option<String>,
}

/// Une entité que l'ouverture n'a pas pu décoder. Ses lectures échouent ; ses écritures
/// réécrivent le nœud par-dessus (une réparation par l'appelant).
struct Illisible {
    noeud: NodeId,
    protocole: Option<Uuid>,
    erreur: String,
}

/// Ce que l'ouverture a supprimé ou mis en quarantaine dans le domaine protocole.
#[derive(Default)]
pub(super) struct RapportProtocoles {
    /// Brouillons et orphelins supprimés.
    pub(super) supprimes: usize,
    pub(super) en_quarantaine: Vec<String>,
}

/// Les index du domaine protocole. Un fichier de cette forme, pas de
/// `HashMap<Uuid, Vec<…>>` partout : chaque lecture a une seule clé d'accès.
#[derive(Default)]
pub(super) struct IndexProtocoles {
    par_id: HashMap<Uuid, EntreeProtocole>,
    /// `(projet, nom) → ids` : `get_protocol_by_name_and_project`. Un ensemble, parce
    /// que la référence n'interdit pas deux protocoles de même nom dans un projet.
    par_nom: HashMap<(Uuid, String), BTreeSet<Uuid>>,
    etats: HashMap<Uuid, Entree<ProtocolState>>,
    etats_de: HashMap<Uuid, Vec<Uuid>>,
    transitions: HashMap<Uuid, Entree<ProtocolTransition>>,
    transitions_de: HashMap<Uuid, Vec<Uuid>>,
    runs: HashMap<Uuid, EntreeRun>,
    runs_de: HashMap<Uuid, Vec<Uuid>>,
    /// `parent → enfants` : les runs dont le parent existait à la création.
    enfants: HashMap<Uuid, Vec<Uuid>>,
    /// Les runs en cours (tous).
    actifs: HashSet<Uuid>,
    /// `protocole → runs racines en cours` : le garde de `create_protocol_run`.
    racines_actives: HashMap<Uuid, HashSet<Uuid>>,
    runtime: HashMap<Uuid, Entree<RuntimeState>>,
    runtime_de: HashMap<Uuid, Vec<Uuid>>,
    /// Entités illisibles, par `(genre, id)`. Le genre est le libellé du nœud.
    illisibles: HashMap<(&'static str, Uuid), Illisible>,
}

fn liste_ajouter(map: &mut HashMap<Uuid, Vec<Uuid>>, cle: Uuid, id: Uuid) {
    let liste = map.entry(cle).or_default();
    if !liste.contains(&id) {
        liste.push(id);
    }
}

fn liste_retirer(map: &mut HashMap<Uuid, Vec<Uuid>>, cle: Uuid, id: Uuid) {
    let vide = match map.get_mut(&cle) {
        Some(liste) => {
            liste.retain(|x| *x != id);
            liste.is_empty()
        }
        None => false,
    };
    if vide {
        map.remove(&cle);
    }
}

/// Ordre décroissant des fins de run, nulls en premier (comme `ORDER BY … DESC` de Cypher).
fn cmp_fin_croissant(x: &Option<String>, y: &Option<String>) -> Ordering {
    match (x, y) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(a), Some(b)) => a.cmp(b),
    }
}

impl IndexProtocoles {
    /// Reconstruit les index depuis la base, et répare ce que l'ouverture doit réparer :
    /// les brouillons, les orphelins (une suppression interrompue). Une entité qu'on ne
    /// sait pas décoder est mise en quarantaine, jamais devinée.
    pub(super) fn reconstruire(db: &ObrainDB) -> Result<(Self, RapportProtocoles)> {
        let store = db.store();
        let mut idx = Self::default();
        let mut rap = RapportProtocoles::default();
        let mut presents_proto: HashSet<Uuid> = HashSet::new();
        let mut presents_run: HashSet<Uuid> = HashSet::new();

        // 1. Protocoles : aucun n'est orphelin.
        for noeud in store.nodes_by_label(PROTOCOLE) {
            let Some(n) = db.get_node(noeud) else {
                rap.en_quarantaine
                    .push(format!("nœud {PROTOCOLE} {noeud:?} : illisible"));
                continue;
            };
            if n.get_property(BROUILLON).is_some() || n.properties.is_empty() {
                db.delete_node(noeud);
                rap.supprimes += 1;
                continue;
            }
            let lecteur = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            let get: Lecture<'_> = &lecteur;
            let id_brut = chaine(get, "id").and_then(|s| s.parse::<Uuid>().ok());
            if let Some(id) = id_brut {
                presents_proto.insert(id);
            }
            match lire_protocole(get) {
                Err(e) => {
                    rap.en_quarantaine
                        .push(format!("nœud {PROTOCOLE} {noeud:?} : {e:#}"));
                    if let Some(id) = id_brut {
                        idx.illisibles.insert(
                            (PROTOCOLE, id),
                            Illisible {
                                noeud,
                                protocole: Some(id),
                                erreur: format!("{e:#}"),
                            },
                        );
                    }
                }
                Ok(p) if idx.par_id.contains_key(&p.id) => rap
                    .en_quarantaine
                    .push(format!("nœud {PROTOCOLE} {noeud:?} : id {} déjà pris", p.id)),
                Ok(p) => {
                    let maj = chaine(get, "updated_at").unwrap_or_default();
                    idx.publier_protocole(EntreeProtocole {
                        noeud,
                        valeur: p,
                        maj,
                    });
                }
            }
        }

        // 2. États : orphelins si leur protocole n'existe pas.
        for noeud in store.nodes_by_label(ETAT) {
            let Some(n) = db.get_node(noeud) else {
                rap.en_quarantaine
                    .push(format!("nœud {ETAT} {noeud:?} : illisible"));
                continue;
            };
            if n.get_property(BROUILLON).is_some() || n.properties.is_empty() {
                db.delete_node(noeud);
                rap.supprimes += 1;
                continue;
            }
            let lecteur = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            let get: Lecture<'_> = &lecteur;
            let id_brut = chaine(get, "id").and_then(|s| s.parse::<Uuid>().ok());
            let proto_brut = chaine(get, "protocol_id").and_then(|s| s.parse::<Uuid>().ok());
            match lire_etat(get) {
                Err(e) => {
                    rap.en_quarantaine
                        .push(format!("nœud {ETAT} {noeud:?} : {e:#}"));
                    if let Some(id) = id_brut {
                        idx.illisibles.insert(
                            (ETAT, id),
                            Illisible {
                                noeud,
                                protocole: proto_brut,
                                erreur: format!("{e:#}"),
                            },
                        );
                    }
                }
                Ok(s) if !presents_proto.contains(&s.protocol_id) => {
                    db.delete_node(noeud);
                    rap.supprimes += 1;
                }
                Ok(s) if idx.etats.contains_key(&s.id) => rap
                    .en_quarantaine
                    .push(format!("nœud {ETAT} {noeud:?} : id {} déjà pris", s.id)),
                Ok(s) => idx.publier_etat(noeud, s),
            }
        }

        // 3. Transitions : orphelines si leur protocole n'existe pas.
        for noeud in store.nodes_by_label(TRANSITION) {
            let Some(n) = db.get_node(noeud) else {
                rap.en_quarantaine
                    .push(format!("nœud {TRANSITION} {noeud:?} : illisible"));
                continue;
            };
            if n.get_property(BROUILLON).is_some() || n.properties.is_empty() {
                db.delete_node(noeud);
                rap.supprimes += 1;
                continue;
            }
            let lecteur = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            let get: Lecture<'_> = &lecteur;
            let id_brut = chaine(get, "id").and_then(|s| s.parse::<Uuid>().ok());
            let proto_brut = chaine(get, "protocol_id").and_then(|s| s.parse::<Uuid>().ok());
            match lire_transition(get) {
                Err(e) => {
                    rap.en_quarantaine
                        .push(format!("nœud {TRANSITION} {noeud:?} : {e:#}"));
                    if let Some(id) = id_brut {
                        idx.illisibles.insert(
                            (TRANSITION, id),
                            Illisible {
                                noeud,
                                protocole: proto_brut,
                                erreur: format!("{e:#}"),
                            },
                        );
                    }
                }
                Ok(t) if !presents_proto.contains(&t.protocol_id) => {
                    db.delete_node(noeud);
                    rap.supprimes += 1;
                }
                Ok(t) if idx.transitions.contains_key(&t.id) => rap
                    .en_quarantaine
                    .push(format!("nœud {TRANSITION} {noeud:?} : id {} déjà pris", t.id)),
                Ok(t) => idx.publier_transition(noeud, t),
            }
        }

        // 4. Runs : orphelins si leur protocole n'existe pas.
        for noeud in store.nodes_by_label(RUN) {
            let Some(n) = db.get_node(noeud) else {
                rap.en_quarantaine
                    .push(format!("nœud {RUN} {noeud:?} : illisible"));
                continue;
            };
            if n.get_property(BROUILLON).is_some() || n.properties.is_empty() {
                db.delete_node(noeud);
                rap.supprimes += 1;
                continue;
            }
            let lecteur = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            let get: Lecture<'_> = &lecteur;
            let id_brut = chaine(get, "id").and_then(|s| s.parse::<Uuid>().ok());
            let proto_brut = chaine(get, "protocol_id").and_then(|s| s.parse::<Uuid>().ok());
            match lire_run(get) {
                Err(e) => {
                    rap.en_quarantaine
                        .push(format!("nœud {RUN} {noeud:?} : {e:#}"));
                    if let Some(id) = id_brut {
                        presents_run.insert(id);
                        idx.illisibles.insert(
                            (RUN, id),
                            Illisible {
                                noeud,
                                protocole: proto_brut,
                                erreur: format!("{e:#}"),
                            },
                        );
                    }
                }
                Ok(r) if !presents_proto.contains(&r.protocol_id) => {
                    db.delete_node(noeud);
                    rap.supprimes += 1;
                }
                Ok(r) if idx.runs.contains_key(&r.id) => rap
                    .en_quarantaine
                    .push(format!("nœud {RUN} {noeud:?} : id {} déjà pris", r.id)),
                Ok(r) => {
                    presents_run.insert(r.id);
                    let debut = chaine(get, "started_at").unwrap_or_default();
                    let fin = chaine_non_vide(get, "completed_at");
                    idx.publier_run(EntreeRun {
                        noeud,
                        valeur: r,
                        debut,
                        fin,
                    });
                }
            }
        }
        // Les liens de parenté : un enfant est lié si son parent existe dans la base.
        // (La référence ne pose le lien qu'à la création, et seulement si le parent existait :
        // une fois le parent présent, c'est la même condition.)
        let liens: Vec<(Uuid, Uuid)> = idx
            .runs
            .values()
            .filter_map(|e| e.valeur.parent_run_id.map(|p| (p, e.valeur.id)))
            .filter(|(p, _)| presents_run.contains(p))
            .collect();
        for (parent, enfant) in liens {
            liste_ajouter(&mut idx.enfants, parent, enfant);
        }

        // 5. Runtime states : orphelins si leur run n'existe pas.
        for noeud in store.nodes_by_label(RUNTIME) {
            let Some(n) = db.get_node(noeud) else {
                rap.en_quarantaine
                    .push(format!("nœud {RUNTIME} {noeud:?} : illisible"));
                continue;
            };
            if n.get_property(BROUILLON).is_some() || n.properties.is_empty() {
                db.delete_node(noeud);
                rap.supprimes += 1;
                continue;
            }
            let lecteur = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            let get: Lecture<'_> = &lecteur;
            let run_brut = chaine(get, "run_id").and_then(|s| s.parse::<Uuid>().ok());
            match lire_runtime(get) {
                Err(e) => {
                    rap.en_quarantaine
                        .push(format!("nœud {RUNTIME} {noeud:?} : {e:#}"));
                    if let Some(id) = chaine(get, "id").and_then(|s| s.parse::<Uuid>().ok()) {
                        idx.illisibles.insert(
                            (RUNTIME, id),
                            Illisible {
                                noeud,
                                protocole: None,
                                erreur: format!("{e:#}"),
                            },
                        );
                    }
                }
                Ok(_) if !run_brut.is_some_and(|r| presents_run.contains(&r)) => {
                    db.delete_node(noeud);
                    rap.supprimes += 1;
                }
                Ok(s) if idx.runtime.contains_key(&s.id) => rap
                    .en_quarantaine
                    .push(format!("nœud {RUNTIME} {noeud:?} : id {} déjà pris", s.id)),
                Ok(s) => idx.publier_runtime(noeud, s),
            }
        }

        Ok((idx, rap))
    }

    // ------------------------------------------------------------------
    // Requêtes sur l'index
    // ------------------------------------------------------------------

    /// Le nœud d'une entité de ce genre, lisible ou non, s'il existe.
    fn noeud_connu(&self, genre: &'static str, id: Uuid) -> Option<NodeId> {
        let trouve = match genre {
            PROTOCOLE => self.par_id.get(&id).map(|e| e.noeud),
            ETAT => self.etats.get(&id).map(|e| e.noeud),
            TRANSITION => self.transitions.get(&id).map(|e| e.noeud),
            RUN => self.runs.get(&id).map(|e| e.noeud),
            _ => self.runtime.get(&id).map(|e| e.noeud),
        };
        trouve.or_else(|| self.illisibles.get(&(genre, id)).map(|i| i.noeud))
    }

    /// Refuse une lecture d'une entité illisible, comme la référence (son décodage échoue).
    fn lisible(&self, genre: &'static str, id: Uuid) -> Result<()> {
        match self.illisibles.get(&(genre, id)) {
            Some(i) => Err(anyhow!("{genre} {id} illisible dans la base obrain : {}", i.erreur)),
            None => Ok(()),
        }
    }

    fn protocole_existe(&self, id: Uuid) -> bool {
        self.noeud_connu(PROTOCOLE, id).is_some()
    }

    fn racine_active(&self, protocole: Uuid) -> bool {
        self.racines_actives
            .get(&protocole)
            .is_some_and(|s| !s.is_empty())
    }

    /// Le projet effectif d'un run : le sien, sinon celui de son protocole (la règle de
    /// `find_active_run_for_project` et de `list_completed_runs_for_project`).
    fn projet_effectif(&self, e: &EntreeRun) -> Option<Uuid> {
        e.valeur.project_id.or_else(|| {
            self.par_id
                .get(&e.valeur.protocol_id)
                .and_then(|p| p.valeur.project_id)
        })
    }

    fn runs_du_protocole(&self, protocole: Uuid) -> impl Iterator<Item = &EntreeRun> + '_ {
        self.runs_de
            .get(&protocole)
            .into_iter()
            .flatten()
            .filter_map(|id| self.runs.get(id))
    }

    fn enfants_de(&self, parent: Uuid) -> impl Iterator<Item = &EntreeRun> + '_ {
        self.enfants
            .get(&parent)
            .into_iter()
            .flatten()
            .filter_map(|id| self.runs.get(id))
    }

    fn runtime_du_run(&self, run: Uuid) -> Vec<NodeId> {
        self.runtime_de
            .get(&run)
            .into_iter()
            .flatten()
            .filter_map(|id| self.runtime.get(id))
            .map(|e| e.noeud)
            .collect()
    }

    /// Les nœuds d'un protocole à retirer avec lui : états, transitions, runs et leurs runtime states.
    fn noeuds_du_protocole(&self, protocole: Uuid) -> Vec<NodeId> {
        let mut v: Vec<NodeId> = Vec::new();
        v.extend(
            self.etats_de
                .get(&protocole)
                .into_iter()
                .flatten()
                .filter_map(|id| self.etats.get(id))
                .map(|e| e.noeud),
        );
        v.extend(
            self.transitions_de
                .get(&protocole)
                .into_iter()
                .flatten()
                .filter_map(|id| self.transitions.get(id))
                .map(|e| e.noeud),
        );
        for run in self.runs_de.get(&protocole).into_iter().flatten() {
            if let Some(e) = self.runs.get(run) {
                v.push(e.noeud);
            }
            v.extend(self.runtime_du_run(*run));
        }
        v
    }

    /// Une transition illisible de ce protocole fait échouer sa lecture (échec fermé).
    fn transition_lisible_de(&self, protocole: Uuid) -> Result<()> {
        let mauvaise = self
            .illisibles
            .iter()
            .find(|((genre, _), i)| *genre == TRANSITION && i.protocole == Some(protocole));
        match mauvaise {
            Some((_, i)) => Err(anyhow!("{}", i.erreur)),
            None => Ok(()),
        }
    }

    // ------------------------------------------------------------------
    // Publication (après durabilité)
    // ------------------------------------------------------------------

    fn retirer_nom(&mut self, p: &Protocol, id: Uuid) {
        if let Some(pid) = p.project_id {
            let cle = (pid, p.name.clone());
            let vide = match self.par_nom.get_mut(&cle) {
                Some(ids) => {
                    ids.remove(&id);
                    ids.is_empty()
                }
                None => false,
            };
            if vide {
                self.par_nom.remove(&cle);
            }
        }
    }

    fn publier_protocole(&mut self, e: EntreeProtocole) {
        let id = e.valeur.id;
        if let Some(ancien) = self.par_id.remove(&id) {
            self.retirer_nom(&ancien.valeur, id);
        }
        if let Some(pid) = e.valeur.project_id {
            self.par_nom
                .entry((pid, e.valeur.name.clone()))
                .or_default()
                .insert(id);
        }
        self.illisibles.remove(&(PROTOCOLE, id));
        self.par_id.insert(id, e);
    }

    fn publier_etat(&mut self, noeud: NodeId, valeur: ProtocolState) {
        let id = valeur.id;
        let pid = valeur.protocol_id;
        if let Some(ancien) = self.etats.insert(id, Entree { noeud, valeur }) {
            liste_retirer(&mut self.etats_de, ancien.valeur.protocol_id, id);
        }
        liste_ajouter(&mut self.etats_de, pid, id);
        self.illisibles.remove(&(ETAT, id));
    }

    fn publier_transition(&mut self, noeud: NodeId, valeur: ProtocolTransition) {
        let id = valeur.id;
        let pid = valeur.protocol_id;
        if let Some(ancien) = self.transitions.insert(id, Entree { noeud, valeur }) {
            liste_retirer(&mut self.transitions_de, ancien.valeur.protocol_id, id);
        }
        liste_ajouter(&mut self.transitions_de, pid, id);
        self.illisibles.remove(&(TRANSITION, id));
    }

    /// Un run en cours est dans `actifs`, et, s'il est racine, dans `racines_actives`.
    fn desinscrire_actif(&mut self, id: Uuid) {
        self.actifs.remove(&id);
        let protocole = self.runs.get(&id).map(|e| e.valeur.protocol_id);
        if let Some(pid) = protocole {
            let vide = match self.racines_actives.get_mut(&pid) {
                Some(s) => {
                    s.remove(&id);
                    s.is_empty()
                }
                None => false,
            };
            if vide {
                self.racines_actives.remove(&pid);
            }
        }
    }

    fn inscrire_actif(&mut self, e: &EntreeRun) {
        if e.valeur.status == RunStatus::Running {
            self.actifs.insert(e.valeur.id);
            if e.valeur.parent_run_id.is_none() {
                self.racines_actives
                    .entry(e.valeur.protocol_id)
                    .or_default()
                    .insert(e.valeur.id);
            }
        }
    }

    fn publier_run(&mut self, e: EntreeRun) {
        let id = e.valeur.id;
        let protocole = e.valeur.protocol_id;
        self.desinscrire_actif(id);
        self.inscrire_actif(&e);
        if self.runs.insert(id, e).is_none() {
            liste_ajouter(&mut self.runs_de, protocole, id);
        }
        self.illisibles.remove(&(RUN, id));
    }

    fn publier_runtime(&mut self, noeud: NodeId, valeur: RuntimeState) {
        let id = valeur.id;
        let run = valeur.run_id;
        self.runtime.insert(id, Entree { noeud, valeur });
        liste_ajouter(&mut self.runtime_de, run, id);
        self.illisibles.remove(&(RUNTIME, id));
    }

    fn oter_etat(&mut self, id: Uuid) {
        if let Some(e) = self.etats.remove(&id) {
            liste_retirer(&mut self.etats_de, e.valeur.protocol_id, id);
        }
        self.illisibles.remove(&(ETAT, id));
    }

    fn oter_transition(&mut self, id: Uuid) {
        if let Some(e) = self.transitions.remove(&id) {
            liste_retirer(&mut self.transitions_de, e.valeur.protocol_id, id);
        }
        self.illisibles.remove(&(TRANSITION, id));
    }

    fn oter_runtime_du_run(&mut self, run: Uuid) {
        if let Some(ids) = self.runtime_de.remove(&run) {
            for id in ids {
                self.runtime.remove(&id);
            }
        }
    }

    fn oter_run(&mut self, id: Uuid) {
        self.desinscrire_actif(id);
        if let Some(e) = self.runs.remove(&id) {
            liste_retirer(&mut self.runs_de, e.valeur.protocol_id, id);
            if let Some(parent) = e.valeur.parent_run_id {
                liste_retirer(&mut self.enfants, parent, id);
            }
        }
        self.enfants.remove(&id);
        self.oter_runtime_du_run(id);
        self.illisibles.remove(&(RUN, id));
    }

    fn oter_protocole(&mut self, id: Uuid) {
        if let Some(e) = self.par_id.remove(&id) {
            self.retirer_nom(&e.valeur, id);
        }
        self.illisibles.retain(|(_, _), i| i.protocole != Some(id));
        self.illisibles.remove(&(PROTOCOLE, id));
        for etat in self.etats_de.remove(&id).unwrap_or_default() {
            self.etats.remove(&etat);
        }
        for t in self.transitions_de.remove(&id).unwrap_or_default() {
            self.transitions.remove(&t);
        }
        for run in self.runs_de.remove(&id).unwrap_or_default() {
            self.oter_run(run);
        }
    }
}

// ============================================================================
// Écriture d'un nœud
// ============================================================================

/// Écrit `props` sur `existant`, ou crée un nœud `etiquette` par la séquence de création
/// (marqueur d'abord, retrait en dernier). Le contrôle de validité est fait par l'appelant,
/// AVANT cet appel : rien n'est écrit quand l'appelant refuse.
fn ecrire_noeud(
    w: &mut Ecriture<'_>,
    existant: Option<NodeId>,
    etiquette: &str,
    props: &Props,
) -> Result<NodeId> {
    // Une valeur blob, neuve ou remplacée, n'est pas dans le WAL : le flush complet suit.
    let ancien_blob = existant
        .and_then(|n| w.db.get_node(n))
        .is_some_and(|n| {
            props
                .iter()
                .any(|(cle, _)| n.get_property(cle).is_some_and(en_colonne_blob))
        });
    w.flush_complet = ancien_blob || touche_blob(props);
    match existant {
        Some(n) => {
            for (cle, valeur) in props {
                w.db.set_node_property(n, cle, valeur.clone());
            }
            Ok(n)
        }
        None => {
            let n = w.db.create_node(&[etiquette]);
            w.db.set_node_property(n, BROUILLON, Value::from(true));
            for (cle, valeur) in props {
                w.db.set_node_property(n, cle, valeur.clone());
            }
            w.rendre_durable_avant_le_marqueur()?;
            w.db.remove_node_property(n, BROUILLON);
            Ok(n)
        }
    }
}

/// Relit un nœud qu'on vient d'écrire, avec le décodage de l'ouverture : ce que les
/// lecteurs verront est ce que la base dit.
fn relire<T>(
    db: &ObrainDB,
    noeud: NodeId,
    lire: impl FnOnce(Lecture<'_>) -> Result<T>,
) -> Result<T> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud {noeud:?} absent juste après son écriture"))?;
    let lecteur = |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    };
    lire(&lecteur)
}

fn entree_run(noeud: NodeId, get: Lecture<'_>) -> Result<EntreeRun> {
    Ok(EntreeRun {
        noeud,
        valeur: lire_run(get)?,
        debut: chaine(get, "started_at").unwrap_or_default(),
        fin: chaine_non_vide(get, "completed_at"),
    })
}

fn erreur_non_portee(methode: &'static str) -> anyhow::Error {
    anyhow::Error::new(NonPorte(methode))
}

// ============================================================================
// ProtocolStore
// ============================================================================

#[async_trait]
impl ProtocolStore for ObrainGraphStore {
    async fn upsert_protocol(&self, protocol: &Protocol) -> Result<()> {
        let p = protocol.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let existant = {
                    let etat = w.index()?;
                    // Comme la référence : un projet inconnu, rien n'est écrit.
                    if p.project_id.is_some_and(|pid| !etat.projets.existe(pid)) {
                        return Ok(());
                    }
                    etat.protocoles.noeud_connu(PROTOCOLE, p.id)
                };
                let props = props_protocole(&p, existant.is_none())?;
                let noeud = ecrire_noeud(w, existant, PROTOCOLE, &props)?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, lire_protocole))?;
                w.rendre_durable()?;
                w.index_mut()?.protocoles.publier_protocole(EntreeProtocole {
                    noeud,
                    valeur,
                    maj: p.updated_at.to_rfc3339(),
                });
                Ok(())
            },
        )
        .await
    }

    async fn get_protocol(&self, id: Uuid) -> Result<Option<Protocol>> {
        self.lire(|_, etat| {
            etat.protocoles.lisible(PROTOCOLE, id)?;
            Ok(etat.protocoles.par_id.get(&id).map(|e| e.valeur.clone()))
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
            // Portée : le projet demandé plus les transverses ; sans projet, les transverses seuls.
            let mut liste: Vec<&EntreeProtocole> = etat
                .protocoles
                .par_id
                .values()
                .filter(|e| {
                    let p = &e.valeur;
                    let dans_portee = match project_id {
                        Some(pid) => p.project_id.is_none() || p.project_id == Some(pid),
                        None => p.project_id.is_none(),
                    };
                    dans_portee && category.is_none_or(|c| p.protocol_category == c)
                })
                .collect();
            let total = liste.len();
            liste.sort_by(|a, b| b.maj.cmp(&a.maj).then_with(|| b.valeur.id.cmp(&a.valeur.id)));
            let page = liste
                .into_iter()
                .skip(offset)
                .take(limit)
                .map(|e| e.valeur.clone())
                .collect();
            Ok((page, total))
        })
    }

    async fn delete_protocol(&self, id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, enfants) = {
                    let etat = w.index()?;
                    let Some(noeud) = etat.protocoles.noeud_connu(PROTOCOLE, id) else {
                        return Ok(false);
                    };
                    (noeud, etat.protocoles.noeuds_du_protocole(id))
                };
                // Le protocole d'abord : son retrait est le point de validation. Ses fils
                // deviennent orphelins si on s'arrête ici, et l'ouverture les retire.
                if !w.db.delete_node(noeud) {
                    bail!("obrain a refusé la suppression du protocole {id}");
                }
                for n in enfants {
                    if !w.db.delete_node(n) {
                        bail!("obrain a refusé la suppression d'un fils du protocole {id}");
                    }
                }
                w.rendre_durable()?;
                w.index_mut()?.protocoles.oter_protocole(id);
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
                let existant = {
                    let etat = w.index()?;
                    if !etat.protocoles.protocole_existe(s.protocol_id) {
                        return Ok(());
                    }
                    etat.protocoles.noeud_connu(ETAT, s.id)
                };
                let props = props_etat(&s);
                let noeud = ecrire_noeud(w, existant, ETAT, &props)?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, lire_etat))?;
                w.rendre_durable()?;
                w.index_mut()?.protocoles.publier_etat(noeud, valeur);
                Ok(())
            },
        )
        .await
    }

    async fn get_protocol_states(&self, protocol_id: Uuid) -> Result<Vec<ProtocolState>> {
        self.lire(|_, etat| {
            let mut liste: Vec<ProtocolState> = etat
                .protocoles
                .etats_de
                .get(&protocol_id)
                .into_iter()
                .flatten()
                .filter_map(|id| etat.protocoles.etats.get(id))
                .map(|e| e.valeur.clone())
                .collect();
            // ORDER BY s.state_type, s.name, s.id
            liste.sort_by_cached_key(|s| (s.state_type.to_string(), s.name.clone(), s.id));
            Ok(liste)
        })
    }

    async fn get_protocol_by_name_and_project(
        &self,
        name: &str,
        project_id: Uuid,
    ) -> Result<Option<Uuid>> {
        self.lire(|_, etat| {
            Ok(etat
                .protocoles
                .par_nom
                .get(&(project_id, name.to_string()))
                .and_then(|ids| ids.iter().next().copied()))
        })
    }

    async fn delete_protocol_state(&self, state_id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud = {
                    let etat = w.index()?;
                    match etat.protocoles.noeud_connu(ETAT, state_id) {
                        Some(n) => n,
                        None => return Ok(false),
                    }
                };
                if !w.db.delete_node(noeud) {
                    bail!("obrain a refusé la suppression de l'état {state_id}");
                }
                w.rendre_durable()?;
                w.index_mut()?.protocoles.oter_etat(state_id);
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
                let existant = {
                    let etat = w.index()?;
                    if !etat.protocoles.protocole_existe(t.protocol_id) {
                        return Ok(());
                    }
                    etat.protocoles.noeud_connu(TRANSITION, t.id)
                };
                let props = props_transition(&t)?;
                let noeud = ecrire_noeud(w, existant, TRANSITION, &props)?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, lire_transition))?;
                w.rendre_durable()?;
                w.index_mut()?
                    .protocoles
                    .publier_transition(noeud, valeur);
                Ok(())
            },
        )
        .await
    }

    async fn get_protocol_transitions(
        &self,
        protocol_id: Uuid,
    ) -> Result<Vec<ProtocolTransition>> {
        self.lire(|_, etat| {
            etat.protocoles.transition_lisible_de(protocol_id)?;
            let mut liste: Vec<ProtocolTransition> = etat
                .protocoles
                .transitions_de
                .get(&protocol_id)
                .into_iter()
                .flatten()
                .filter_map(|id| etat.protocoles.transitions.get(id))
                .map(|e| e.valeur.clone())
                .collect();
            // ORDER BY t.trigger, t.id
            liste.sort_by_cached_key(|t| (t.trigger.clone(), t.id));
            Ok(liste)
        })
    }

    async fn delete_protocol_transition(&self, transition_id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud = {
                    let etat = w.index()?;
                    match etat.protocoles.noeud_connu(TRANSITION, transition_id) {
                        Some(n) => n,
                        None => return Ok(false),
                    }
                };
                if !w.db.delete_node(noeud) {
                    bail!("obrain a refusé la suppression de la transition {transition_id}");
                }
                w.rendre_durable()?;
                w.index_mut()?.protocoles.oter_transition(transition_id);
                Ok(true)
            },
        )
        .await
    }
}

// ============================================================================
// ProtocolRunStore
// ============================================================================

#[async_trait]
impl ProtocolRunStore for ObrainGraphStore {
    async fn create_protocol_run(&self, run: &ProtocolRun) -> Result<()> {
        let r = run.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let parent_lie = {
                    let etat = w.index()?;
                    if etat.protocoles.noeud_connu(RUN, r.id).is_some() {
                        bail!("un run d'id {} existe déjà", r.id);
                    }
                    // Comme la référence : pas de protocole, ou un run racine en cours sur
                    // ce protocole, et la création est refusée avec le même message.
                    if !etat.protocoles.protocole_existe(r.protocol_id) {
                        bail!("{MESSAGE_NON_CREABLE} {}", r.protocol_id);
                    }
                    if r.parent_run_id.is_none() && etat.protocoles.racine_active(r.protocol_id) {
                        bail!("{MESSAGE_NON_CREABLE} {}", r.protocol_id);
                    }
                    r.parent_run_id
                        .filter(|p| etat.protocoles.noeud_connu(RUN, *p).is_some())
                };
                let props = props_run_creation(&r)?;
                let noeud = ecrire_noeud(w, None, RUN, &props)?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, |get| entree_run(noeud, get)))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.protocoles.publier_run(valeur);
                if let Some(parent) = parent_lie {
                    liste_ajouter(&mut etat.protocoles.enfants, parent, r.id);
                }
                Ok(())
            },
        )
        .await
    }

    async fn get_protocol_run(&self, run_id: Uuid) -> Result<Option<ProtocolRun>> {
        self.lire(|_, etat| {
            etat.protocoles.lisible(RUN, run_id)?;
            Ok(etat.protocoles.runs.get(&run_id).map(|e| e.valeur.clone()))
        })
    }

    async fn update_protocol_run(&self, run: &mut ProtocolRun) -> Result<()> {
        let courant = run.clone();
        let nouvelle_version = self
            .ecrire(
                || (),
                move |(), w: &mut Ecriture<'_>| {
                    let noeud = {
                        let etat = w.index()?;
                        match etat.protocoles.runs.get(&courant.id) {
                            Some(e) if e.valeur.version == courant.version => e.noeud,
                            _ => bail!(
                                "OptimisticLockError: protocol run {} was modified concurrently (expected version {}, stale)",
                                courant.id,
                                courant.version
                            ),
                        }
                    };
                    let mut suivant = courant.clone();
                    suivant.version = courant.version + 1;
                    let props = props_run_maj(&suivant)?;
                    let n = ecrire_noeud(w, Some(noeud), RUN, &props)?;
                    let valeur = w.apres_ecriture(relire(w.db, n, |get| entree_run(n, get)))?;
                    w.rendre_durable()?;
                    w.index_mut()?.protocoles.publier_run(valeur);
                    Ok(suivant.version)
                },
            )
            .await?;
        // Mise à jour en place, comme la référence : l'appelant reste synchrone avec la base.
        run.version = nouvelle_version;
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
            let mut liste: Vec<&EntreeRun> = etat
                .protocoles
                .runs_du_protocole(protocol_id)
                .filter(|e| status.is_none_or(|s| e.valeur.status == s))
                .collect();
            let total = liste.len();
            liste.sort_by(|a, b| b.debut.cmp(&a.debut).then_with(|| b.valeur.id.cmp(&a.valeur.id)));
            let page = liste
                .into_iter()
                .skip(offset)
                .take(limit)
                .map(|e| e.valeur.clone())
                .collect();
            Ok((page, total))
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
        if project_id.is_none() && workspace_slug.is_some() {
            return Err(erreur_non_portee(
                "ProtocolRunStore::list_all_protocol_runs (workspace_slug)",
            ));
        }
        self.lire(|_, etat| {
            let mut liste: Vec<&EntreeRun> = etat
                .protocoles
                .runs
                .values()
                .filter(|e| {
                    if !status.is_none_or(|s| e.valeur.status == s) {
                        return false;
                    }
                    match project_id {
                        // La référence joint le PROTOCOLE du run : c'est son projet qui compte.
                        Some(pid) => etat
                            .protocoles
                            .par_id
                            .get(&e.valeur.protocol_id)
                            .is_some_and(|p| p.valeur.project_id == Some(pid)),
                        None => true,
                    }
                })
                .collect();
            let total = liste.len();
            liste.sort_by(|a, b| b.debut.cmp(&a.debut).then_with(|| b.valeur.id.cmp(&a.valeur.id)));
            let page = liste
                .into_iter()
                .skip(offset)
                .take(limit)
                .map(|e| e.valeur.clone())
                .collect();
            Ok((page, total))
        })
    }

    async fn list_child_runs(&self, parent_run_id: Uuid) -> Result<Vec<ProtocolRun>> {
        self.lire(|_, etat| {
            let mut liste: Vec<&EntreeRun> = etat.protocoles.enfants_de(parent_run_id).collect();
            liste.sort_by(|a, b| {
                a.debut
                    .cmp(&b.debut)
                    .then_with(|| a.valeur.id.cmp(&b.valeur.id))
            });
            Ok(liste.into_iter().map(|e| e.valeur.clone()).collect())
        })
    }

    async fn count_child_runs(&self, parent_run_id: Uuid) -> Result<usize> {
        self.lire(|_, etat| {
            Ok(etat
                .protocoles
                .enfants
                .get(&parent_run_id)
                .map_or(0, |v| v.len()))
        })
    }

    async fn get_run_tree(&self, root_run_id: Uuid) -> Result<Vec<ProtocolRun>> {
        self.lire(|_, etat| {
            let idx = &etat.protocoles;
            let Some(racine) = idx.runs.get(&root_run_id) else {
                return Ok(Vec::new());
            };
            let mut tous: Vec<&EntreeRun> = vec![racine];
            let mut vus: HashSet<Uuid> = HashSet::from([root_run_id]);
            let mut frontiere = vec![root_run_id];
            for _ in 0..PROFONDEUR_ARBRE {
                let mut suivante = Vec::new();
                for parent in &frontiere {
                    for enfant in idx.enfants.get(parent).into_iter().flatten() {
                        if vus.insert(*enfant) {
                            if let Some(e) = idx.runs.get(enfant) {
                                tous.push(e);
                            }
                            suivante.push(*enfant);
                        }
                    }
                }
                if suivante.is_empty() {
                    break;
                }
                frontiere = suivante;
            }
            // ORDER BY r.depth, r.started_at, r.id
            tous.sort_by(|a, b| {
                a.valeur
                    .depth
                    .cmp(&b.valeur.depth)
                    .then_with(|| a.debut.cmp(&b.debut))
                    .then_with(|| a.valeur.id.cmp(&b.valeur.id))
            });
            Ok(tous.into_iter().map(|e| e.valeur.clone()).collect())
        })
    }

    async fn delete_protocol_run(&self, run_id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, runtime) = {
                    let etat = w.index()?;
                    let Some(noeud) = etat.protocoles.noeud_connu(RUN, run_id) else {
                        return Ok(false);
                    };
                    (noeud, etat.protocoles.runtime_du_run(run_id))
                };
                for n in runtime {
                    w.db.delete_node(n);
                }
                if !w.db.delete_node(noeud) {
                    bail!("obrain a refusé la suppression du run {run_id}");
                }
                w.rendre_durable()?;
                w.index_mut()?.protocoles.oter_run(run_id);
                Ok(true)
            },
        )
        .await
    }

    async fn create_produced_during(
        &self,
        _entity_type: &str,
        _entity_id: Uuid,
        _run_id: Uuid,
    ) -> Result<bool> {
        Err(erreur_non_portee("ProtocolRunStore::create_produced_during"))
    }

    async fn get_run_outcomes(
        &self,
        _run_id: Uuid,
    ) -> Result<Vec<crate::neo4j::protocol::ProducedArtefact>> {
        Err(erreur_non_portee("ProtocolRunStore::get_run_outcomes"))
    }

    async fn find_active_run_for_project(&self, project_id: Uuid) -> Result<Option<Uuid>> {
        self.lire(|_, etat| {
            let idx = &etat.protocoles;
            // ORDER BY r.started_at DESC, r.id DESC LIMIT 1
            let meilleur = idx
                .actifs
                .iter()
                .filter_map(|id| idx.runs.get(id))
                .filter(|e| idx.protocole_existe(e.valeur.protocol_id))
                .filter(|e| idx.projet_effectif(e) == Some(project_id))
                .max_by(|a, b| {
                    a.debut
                        .cmp(&b.debut)
                        .then_with(|| a.valeur.id.cmp(&b.valeur.id))
                });
            Ok(meilleur.map(|e| e.valeur.id))
        })
    }

    async fn persist_reasoning_tree(
        &self,
        _tree: &crate::reasoning::ReasoningTree,
        _linked_entity_type: Option<&str>,
        _linked_entity_id: Option<Uuid>,
    ) -> Result<Uuid> {
        Err(erreur_non_portee("ProtocolRunStore::persist_reasoning_tree"))
    }

    async fn get_run_reasoning_tree_id(&self, _run_id: Uuid) -> Result<Option<Uuid>> {
        Err(erreur_non_portee("ProtocolRunStore::get_run_reasoning_tree_id"))
    }

    async fn list_completed_runs_for_project(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<ProtocolRun>> {
        self.lire(|_, etat| {
            let idx = &etat.protocoles;
            let mut liste: Vec<&EntreeRun> = idx
                .runs
                .values()
                .filter(|e| e.valeur.status == RunStatus::Completed)
                .filter(|e| idx.protocole_existe(e.valeur.protocol_id))
                .filter(|e| idx.projet_effectif(e) == Some(project_id))
                .collect();
            // ORDER BY r.completed_at DESC, r.id DESC
            liste.sort_by(|a, b| {
                cmp_fin_croissant(&b.fin, &a.fin).then_with(|| b.valeur.id.cmp(&a.valeur.id))
            });
            Ok(liste
                .into_iter()
                .take(limit)
                .map(|e| e.valeur.clone())
                .collect())
        })
    }

    async fn create_runtime_state(&self, state: &RuntimeState) -> Result<()> {
        let s = state.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                {
                    let etat = w.index()?;
                    if etat.protocoles.noeud_connu(RUNTIME, s.id).is_some() {
                        bail!("un runtime state d'id {} existe déjà", s.id);
                    }
                    // Comme la référence (`MATCH` sur le run et sur l'état) : l'un des deux
                    // absent, rien n'est créé, et rien n'est signalé.
                    let existe = etat.protocoles.noeud_connu(RUN, s.run_id).is_some()
                        && etat.protocoles.noeud_connu(ETAT, s.generated_by).is_some();
                    if !existe {
                        return Ok(());
                    }
                }
                let props = props_runtime(&s);
                let noeud = ecrire_noeud(w, None, RUNTIME, &props)?;
                let valeur = w.apres_ecriture(relire(w.db, noeud, lire_runtime))?;
                w.rendre_durable()?;
                w.index_mut()?.protocoles.publier_runtime(noeud, valeur);
                Ok(())
            },
        )
        .await
    }

    async fn get_runtime_states(&self, run_id: Uuid) -> Result<Vec<RuntimeState>> {
        self.lire(|_, etat| {
            let idx = &etat.protocoles;
            let mut liste: Vec<&RuntimeState> = idx
                .runtime_de
                .get(&run_id)
                .into_iter()
                .flatten()
                .filter_map(|id| idx.runtime.get(id))
                .map(|e| &e.valeur)
                .collect();
            // ORDER BY rs.index, rs.id
            liste.sort_by(|a, b| a.index.cmp(&b.index).then_with(|| a.id.cmp(&b.id)));
            Ok(liste.into_iter().cloned().collect())
        })
    }

    async fn delete_runtime_states(&self, run_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeuds = w.index()?.protocoles.runtime_du_run(run_id);
                for n in noeuds {
                    if !w.db.delete_node(n) {
                        bail!("obrain a refusé la suppression d'un runtime state du run {run_id}");
                    }
                }
                w.rendre_durable()?;
                w.index_mut()?.protocoles.oter_runtime_du_run(run_id);
                Ok(())
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obrain::Creation;
    use crate::protocol::{TriggerMode, ProtocolCategory};

    fn protocole(nom: &str) -> Protocol {
        Protocol {
            id: Uuid::new_v4(),
            name: nom.to_string(),
            description: "une description".to_string(),
            project_id: None,
            skill_id: None,
            entry_state: Uuid::new_v4(),
            terminal_states: vec![Uuid::new_v4()],
            protocol_category: ProtocolCategory::default(),
            trigger_mode: TriggerMode::default(),
            trigger_config: None,
            relevance_vector: None,
            last_triggered_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn run(protocole: Uuid, parent: Option<Uuid>) -> ProtocolRun {
        ProtocolRun {
            id: Uuid::new_v4(),
            protocol_id: protocole,
            project_id: None,
            plan_id: None,
            task_id: None,
            parent_run_id: parent,
            current_state: Uuid::new_v4(),
            states_visited: vec![],
            status: RunStatus::Running,
            started_at: Utc::now(),
            completed_at: None,
            error: None,
            triggered_by: "manual".to_string(),
            depth: 0,
            version: 0,
            runner_managed: false,
        }
    }

    /// Ce qui est acquitté survit à la fermeture et à la réouverture ; un seul run racine
    /// en cours par protocole ; le CAS refuse une version périmée.
    #[tokio::test]
    async fn protocole_et_runs_survivent_a_la_reouverture() {
        let dossier = tempfile::tempdir().unwrap();
        let p = protocole("cycle-de-vie");
        let etat = ProtocolState {
            id: Uuid::new_v4(),
            protocol_id: p.id,
            name: "début".to_string(),
            description: String::new(),
            action: None,
            state_type: Default::default(),
            sub_protocol_id: None,
            completion_strategy: None,
            on_failure_strategy: None,
            generator_config: None,
            prompt_fragment: Some("x".repeat(400)),
            available_tools: None,
            forbidden_actions: None,
            state_timeout_secs: Some(30),
        };
        let r = run(p.id, None);

        {
            let store = ObrainGraphStore::open(dossier.path(), Creation::Permise).unwrap();
            store.upsert_protocol(&p).await.unwrap();
            store.upsert_protocol_state(&etat).await.unwrap();
            store.create_protocol_run(&r).await.unwrap();
            // Un second run racine en cours sur le même protocole est refusé.
            assert!(store.create_protocol_run(&run(p.id, None)).await.is_err());
            // Un enfant est toujours accepté.
            store.create_protocol_run(&run(p.id, Some(r.id))).await.unwrap();
            store.close().await.unwrap();
        }

        let store = ObrainGraphStore::open(dossier.path(), Creation::Permise).unwrap();
        let relu = store.get_protocol(p.id).await.unwrap().unwrap();
        assert_eq!(relu.name, "cycle-de-vie");
        let etats = store.get_protocol_states(p.id).await.unwrap();
        assert_eq!(etats.len(), 1);
        assert_eq!(etats[0].prompt_fragment.as_deref().map(str::len), Some(400));
        assert_eq!(etats[0].state_timeout_secs, Some(30));

        let mut relu_run = store.get_protocol_run(r.id).await.unwrap().unwrap();
        assert_eq!(store.count_child_runs(r.id).await.unwrap(), 1);
        let arbre = store.get_run_tree(r.id).await.unwrap();
        assert_eq!(arbre.len(), 2);
        assert_eq!(arbre[0].id, r.id);

        // CAS : la version 0 est la bonne une fois ; la seconde écriture avec 0 est périmée.
        let mut copie = relu_run.clone();
        store.update_protocol_run(&mut relu_run).await.unwrap();
        assert_eq!(relu_run.version, 1);
        assert!(store.update_protocol_run(&mut copie).await.is_err());
        // Plus de run racine en cours : une nouvelle racine peut être créée seulement si
        // l'ancienne n'est plus en cours.
        relu_run.status = RunStatus::Completed;
        store.update_protocol_run(&mut relu_run).await.unwrap();
        store.create_protocol_run(&run(p.id, None)).await.unwrap();
        store.close().await.unwrap();
    }
}
