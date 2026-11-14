//! `ChatStore` on obrain — the chat sessions, their events, the anchors, the
//! episodes, the provider settings and the conversation segments. A PARTIAL
//! port: every method is either written here or refused by name (`NonPorte`),
//! never answered with an empty or default value in place of a refusal.
//!
//! A chat session is one node labelled `ChatSession`, whose properties carry
//! the names and the lossy conventions of the Neo4j adapter (`src/neo4j/chat.rs`,
//! the reference): `""` for an absent optional string, `0.0` for an absent
//! `total_cost_usd`, `shell_moved` written `false` at creation, and so on. The
//! extra scalar properties the session node carries beside `ChatSessionNode`
//! (`shell_moved`, `children_cost_usd`, `resume_fingerprint`, `anchor_seq`) are
//! held in the index entry and served by their own methods, exactly like the
//! node does.
//!
//! # Edges
//!
//! - `(:ChatSession)-[:SPAWNED_BY {type, run_id, task_id, created_at}]->(:ChatSession)`
//!   — the lineage edge. `create_chat_session` writes a bare edge (only
//!   `created_at`) when the `spawned_by` JSON names a parent;
//!   `create_spawned_by_relation` fills the run metadata on it (or creates it).
//!   The tree / root / children reads use the `spawned_by` JSON (the canonical
//!   source, as the mock and the Neo4j fallback agree); the edge is kept for
//!   `get_run_sessions`.
//!
//! Edges NOT written, each a named divergence:
//!
//! - `(:ChatSession)-[:HAS_EVENT]->(:ChatEvent)`: an event is a bare node that
//!   carries its `session_id`, which is what every event read goes through;
//! - `(:SessionEpisode)-[:EPISODE_OF]->(:ChatSession)`: the episode carries its
//!   `session_id`, and is deleted with its session;
//! - `(:ConversationSegment)-[:PRECEDES]->(:ConversationSegment)`: no read of
//!   the contract follows it;
//! - `(:Agent)-[:LIVED]->(:SessionEpisode)`: Neo4j makes the agent node and the
//!   edge best effort at `save_session_episode`; the agent domain is not ported,
//!   so nothing is written and nothing is refused (an episode is worth more than
//!   the edge that names it, as on Neo4j);
//! - `HAS_CHAT_SESSION` (project → session): read only by refused methods
//!   (`get_discussed_co_changers`).
//!
//! # Creations
//!
//! Every node a write CREATES carries the draft marker `_brouillon` first and
//! loses it last (the commit point); opening deletes a draft, and a node with
//! neither property nor edge. A creation interrupted half-way is therefore never
//! read back with default values in place of what was not written.
//!
//! # Differences, all on the strict side or named
//!
//! - a session whose `id` is already taken is refused (Neo4j would create a
//!   second node with the same id);
//! - a write that UPDATES several properties of an existing node is not atomic
//!   (obrain has no multi-write transaction; Neo4j does it in one statement).
//!   The intermediate states an interruption can leave, each named:
//!   - an anchor operation writes `anchor_seq` FIRST, then the anchor, then its
//!     journal entry: an interruption leaves a gap in the numbering (the version
//!     moved, nothing changed), or an anchor changed without its journal entry —
//!     never two journal entries with the same `seq`. An existing anchor is
//!     rewritten property by property: an interruption can leave some properties
//!     new and others old (each one valid);
//!   - `update_chat_session_provider` clears `resume_token`, `capabilities` and
//!     `cli_session_id` FIRST: an interruption can leave the old provider with
//!     no resume handle (the next turn starts fresh), or the new `provider_id`
//!     beside the old `model` / `routed_by` / `engine` — never a resume handle of
//!     one provider family under another provider. `restore_chat_session_provider`
//!     writes the resume handles LAST, for the same reason;
//!   - the other session updates write their properties one by one; each is valid
//!     alone;
//! - `delete_chat_session` also deletes the session's `ChatEvent`s (Neo4j leaves
//!   them orphaned: its `DETACH DELETE` takes the session and its episode only);
//! - `create_spawned_by_relation` for a child whose `SPAWNED_BY` edge points to
//!   ANOTHER parent is refused (Neo4j's `MERGE` would add a second lineage edge;
//!   the index holds one parent per session);
//! - a segment or boundary node that cannot be read at open is quarantined, and
//!   `get_segments` / `get_boundaries` of its conversation refuse until a `put`
//!   replaces the set (Neo4j fails the read the same way);
//! - `get_chat_events` with a negative `limit` returns nothing (Neo4j errs);
//! - `search_episodes_by_embedding` is refused by name: obrain does not
//!   journal vector values in its WAL (trame `VecColumnWrite` without value),
//!   so an embedding acknowledged could not be guaranteed durable. The embedding
//!   of an episode is therefore neither written nor served
//!   (`list_session_episodes` reads the payload, which has no vector, as on Neo4j).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{EdgeId, NodeId, Value};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::{Ecriture, NonPorte, ObrainGraphStore, RapportOuverture, SEUIL_COLONNE_BLOB};
use crate::chat::anchor::{
    self, Anchor, AnchorActor, AnchorError, AnchorEvent, AnchorEventKind, AnchorOp, AnchorPage,
    AnchorRole, AnchorSet, AnchorSetChange, AnchorState, AnchorTargetType,
};
use crate::chat::provider::cognitive::segment::{Boundary, Segment};
use crate::chat::provider::policy::SessionAccess;
use crate::episodes::models::Episode;
use crate::neo4j::models::{
    ChatEventRecord, ChatSessionNode, DiscussedEntity, ExecutionPlace, ExecutionPlaceEvent,
    ExecutionPlaceEventKind, LinkedSessionInfo, PlansTaskGraph, ProjectWorkspaceRow, SessionInfo,
    SessionLinkRow, SessionTreeNode, SessionWithLinks,
};
use crate::neo4j::traits::ChatStore;

const SESSION: &str = "ChatSession";
const EVENEMENT: &str = "ChatEvent";
const REGLAGE: &str = "LlmSetting";
const EPISODE: &str = "SessionEpisode";
const ANCRE: &str = "Anchor";
const EVENEMENT_ANCRE: &str = "AnchorEvent";
const LIEU: &str = "ExecutionPlaceEvent";
const SEGMENT: &str = "ConversationSegment";
const FRONTIERE: &str = "ConversationBoundary";
const RACINE_AGENT: &str = "AgentIdentityRoot";
const BROUILLON: &str = "_brouillon";
const ARETE_SPAWNED: &str = "SPAWNED_BY";

/// The methods of `ChatStore` refused on every call, by name. Listed here so the
/// accounting of `tests/obrain_backend.rs` and `methodes_qui_servent` can tell a
/// refusal from a port.
pub const REFUSEES: &[&str] = &[
    // Read or write a `Plan`, `Task` or `PlanRun`: the plan domain is not ported.
    "ChatStore::link_session_to_run",
    "ChatStore::get_session_link_rows",
    "ChatStore::get_plans_task_graph",
    // Migrations (never run at startup); they sweep or re-derive state.
    "ChatStore::backfill_project_anchors",
    "ChatStore::backfill_spawned_by_edges",
    "ChatStore::revert_inferred_anchors",
    "ChatStore::backfill_discussed",
    // DISCUSSED: the target domains (File/Function/Struct/Trait/Enum/Plan/Task/
    // Decision) are not ported, so a relation could not be written without a
    // phantom target.
    "ChatStore::add_discussed",
    "ChatStore::get_session_entities",
    "ChatStore::get_discussed_co_changers",
    // Linking: reads/writes AgentExecution, Task and Plan, not ported.
    "ChatStore::create_associated_with",
    "ChatStore::get_session_links",
    "ChatStore::get_sessions_for_plan",
    "ChatStore::get_sessions_for_task",
    "ChatStore::get_session_links_batch",
    "ChatStore::get_task_plan_id",
    // Embedding search: obrain does not journal vector values (see module doc).
    "ChatStore::search_episodes_by_embedding",
];

/// A refusal, by name, with the reason it is not ported.
fn refus<T>(methode: &'static str, raison: &'static str) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).context(raison))
}

fn non_signe(v: i64) -> u64 {
    u64::try_from(v).unwrap_or(u64::MAX)
}

fn opt_non_vide(s: Option<String>) -> Option<String> {
    s.filter(|v| !v.is_empty())
}

/// Does this value go to a blob column (a string above the threshold, or a
/// string inside a list)?
fn va_en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
        || v.as_list().is_some_and(|l| {
            l.iter()
                .any(|e| e.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB))
        })
}

/// A property lookup over one hydration of a node.
fn lecteur(db: &ObrainDB, noeud: NodeId) -> Result<impl Fn(&str) -> Option<Value> + '_> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud indexé {noeud:?} mais absent de la base"))?;
    Ok(move |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    })
}

// ============================================================================
// ChatSessionNode <-> properties
// ============================================================================

/// The properties a creation writes, in the order it writes them. Mirrors the
/// `CREATE` of `Neo4jClient::create_chat_session` (plus `shell_moved: false`).
fn proprietes_session(session: &ChatSessionNode) -> Vec<(&'static str, Value)> {
    vec![
        ("id", Value::from(session.id.to_string())),
        (
            "cli_session_id",
            Value::from(session.cli_session_id.clone().unwrap_or_default()),
        ),
        (
            "project_slug",
            Value::from(session.project_slug.clone().unwrap_or_default()),
        ),
        (
            "workspace_slug",
            Value::from(session.workspace_slug.clone().unwrap_or_default()),
        ),
        ("cwd", Value::from(session.cwd.clone())),
        (
            "title",
            Value::from(session.title.clone().unwrap_or_default()),
        ),
        ("model", Value::from(session.model.clone())),
        ("created_at", Value::from(session.created_at.to_rfc3339())),
        ("updated_at", Value::from(session.updated_at.to_rfc3339())),
        ("message_count", Value::from(session.message_count)),
        (
            "total_cost_usd",
            Value::from(session.total_cost_usd.unwrap_or(0.0)),
        ),
        (
            "conversation_id",
            Value::from(session.conversation_id.clone().unwrap_or_default()),
        ),
        (
            "preview",
            Value::from(session.preview.clone().unwrap_or_default()),
        ),
        (
            "permission_mode",
            Value::from(session.permission_mode.clone().unwrap_or_default()),
        ),
        (
            "add_dirs",
            Value::from(
                serde_json::to_string(&session.add_dirs.clone().unwrap_or_default())
                    .unwrap_or_else(|_| "[]".to_string()),
            ),
        ),
        (
            "spawned_by",
            Value::from(session.spawned_by.clone().unwrap_or_default()),
        ),
        (
            "provider_id",
            Value::from(session.provider_id.clone().unwrap_or_default()),
        ),
        (
            "routed_by",
            Value::from(session.routed_by.clone().unwrap_or_default()),
        ),
        (
            "routing_mode",
            Value::from(session.routing_mode.clone().unwrap_or_default()),
        ),
        (
            "routing_pool",
            Value::from(session.routing_pool.clone().unwrap_or_default()),
        ),
        (
            "capabilities",
            Value::from(session.capabilities.clone().unwrap_or_default()),
        ),
        (
            "engine",
            Value::from(session.engine.clone().unwrap_or_default()),
        ),
        (
            "resume_token",
            Value::from(session.resume_token.clone().unwrap_or_default()),
        ),
        (
            "execution_place",
            Value::from(session.execution_place.as_str()),
        ),
        ("project_inferred", Value::from(session.project_inferred)),
        ("access", Value::from(session.access.as_str())),
        (
            "owner",
            Value::from(session.owner.clone().unwrap_or_default()),
        ),
        ("file_scope_lifted", Value::from(session.file_scope_lifted)),
        (
            "mirror_of",
            Value::from(session.mirror_of.clone().unwrap_or_default()),
        ),
        ("shell_moved", Value::from(false)),
    ]
}

/// The counterpart of `Neo4jClient::parse_chat_session_node`.
fn lire_session(get: &dyn Fn(&str) -> Option<Value>) -> Result<ChatSessionNode> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let non_vide = |cle: &str| chaine(cle).filter(|s| !s.is_empty());
    let id: Uuid = chaine("id")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("nœud ChatSession sans propriété `id` lisible"))?;
    let cwd = chaine("cwd").ok_or_else(|| anyhow!("session {id} sans `cwd`"))?;
    let model = chaine("model").ok_or_else(|| anyhow!("session {id} sans `model`"))?;
    let add_dirs_json = chaine("add_dirs").unwrap_or_default();
    let add_dirs: Option<Vec<String>> = if add_dirs_json.is_empty() {
        None
    } else {
        serde_json::from_str(&add_dirs_json)
            .ok()
            .and_then(|v: Vec<String>| if v.is_empty() { None } else { Some(v) })
    };
    let total_cost_usd = get("total_cost_usd")
        .and_then(|v| v.as_float64())
        .unwrap_or(0.0);
    Ok(ChatSessionNode {
        id,
        cli_session_id: non_vide("cli_session_id"),
        project_slug: non_vide("project_slug"),
        workspace_slug: non_vide("workspace_slug"),
        cwd,
        title: non_vide("title"),
        model,
        created_at: chaine("created_at")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(Utc::now),
        updated_at: chaine("updated_at")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(Utc::now),
        message_count: get("message_count").and_then(|v| v.as_int64()).unwrap_or(0),
        total_cost_usd: if total_cost_usd == 0.0 {
            None
        } else {
            Some(total_cost_usd)
        },
        conversation_id: non_vide("conversation_id"),
        preview: non_vide("preview"),
        permission_mode: non_vide("permission_mode"),
        add_dirs,
        spawned_by: non_vide("spawned_by"),
        provider_id: non_vide("provider_id"),
        routed_by: non_vide("routed_by"),
        routing_mode: non_vide("routing_mode"),
        routing_pool: non_vide("routing_pool"),
        capabilities: non_vide("capabilities"),
        engine: non_vide("engine"),
        resume_token: non_vide("resume_token"),
        execution_place: ExecutionPlace::parse_stored(
            &chaine("execution_place").unwrap_or_default(),
        ),
        project_inferred: get("project_inferred")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        access: SessionAccess::parse_stored(&chaine("access").unwrap_or_default()),
        owner: non_vide("owner"),
        file_scope_lifted: get("file_scope_lifted")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        mirror_of: non_vide("mirror_of"),
    })
}

// ============================================================================
// Events <-> properties
// ============================================================================

fn proprietes_evenement(session_id: Uuid, ev: &ChatEventRecord) -> Vec<(&'static str, Value)> {
    vec![
        ("id", Value::from(ev.id.to_string())),
        ("session_id", Value::from(session_id.to_string())),
        ("seq", Value::from(ev.seq)),
        ("event_type", Value::from(ev.event_type.clone())),
        ("data", Value::from(ev.data.clone())),
        ("created_at", Value::from(ev.created_at.to_rfc3339())),
    ]
}

fn lire_evenement(get: &dyn Fn(&str) -> Option<Value>) -> Result<ChatEventRecord> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    Ok(ChatEventRecord {
        id: chaine("id")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| anyhow!("événement sans `id` lisible"))?,
        session_id: chaine("session_id")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| anyhow!("événement sans `session_id` lisible"))?,
        // Required, as `parse_chat_event_node` requires them: a default `seq` or
        // an empty `event_type` would replay an event nobody wrote.
        seq: get("seq")
            .and_then(|v| v.as_int64())
            .ok_or_else(|| anyhow!("événement sans `seq`"))?,
        event_type: chaine("event_type").ok_or_else(|| anyhow!("événement sans `event_type`"))?,
        data: chaine("data").ok_or_else(|| anyhow!("événement sans `data`"))?,
        created_at: chaine("created_at")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(Utc::now),
    })
}

// ============================================================================
// Anchors <-> properties
// ============================================================================

fn proprietes_ancre(a: &Anchor) -> Vec<(&'static str, Value)> {
    vec![
        ("id", Value::from(a.id.to_string())),
        ("session_id", Value::from(a.session_id.to_string())),
        ("target_type", Value::from(a.target_type.as_str())),
        ("target_id", Value::from(a.target_id.clone())),
        (
            "roles",
            Value::List(
                a.roles
                    .iter()
                    .map(|r| Value::from(r.as_str()))
                    .collect::<Vec<_>>()
                    .into(),
            ),
        ),
        ("state", Value::from(a.state.as_str())),
        ("by", Value::from(a.by.as_str())),
        ("inferred", Value::from(a.inferred)),
        ("confidence", Value::from(a.confidence)),
        (
            "snapshot_name",
            Value::from(a.snapshot_name.clone().unwrap_or_default()),
        ),
        (
            "snapshot_path",
            Value::from(a.snapshot_path.clone().unwrap_or_default()),
        ),
        (
            "snapshot_type",
            Value::from(a.snapshot_type.clone().unwrap_or_default()),
        ),
        ("rev", Value::from(a.rev.clone().unwrap_or_default())),
        ("version", Value::from(a.version as i64)),
        ("created_at", Value::from(anchor::format_ts(a.created_at))),
        ("updated_at", Value::from(anchor::format_ts(a.updated_at))),
    ]
}

fn lire_ancre(get: &dyn Fn(&str) -> Option<Value>) -> Result<Anchor> {
    use crate::chat::anchor::*;
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let requis = |cle: &str| chaine(cle).ok_or_else(|| anyhow!("ancre sans `{cle}`"));
    let bad = |q: &str| anyhow!("ancre corrompue : {q}");
    let roles: Vec<String> = get("roles")
        .and_then(|v| {
            v.as_list().map(|l| {
                l.iter()
                    .filter_map(|e| e.as_str().map(str::to_string))
                    .collect()
            })
        })
        .unwrap_or_default();
    Ok(Anchor {
        id: requis("id")?.parse()?,
        session_id: requis("session_id")?.parse()?,
        target_type: AnchorTargetType::parse(&requis("target_type")?)
            .ok_or_else(|| bad("target_type"))?,
        target_id: requis("target_id")?,
        roles: roles
            .iter()
            .map(|r| AnchorRole::parse(r).ok_or_else(|| bad("role")))
            .collect::<Result<_>>()?,
        state: AnchorState::parse(&requis("state")?).ok_or_else(|| bad("state"))?,
        by: AnchorActor::parse(&requis("by")?).ok_or_else(|| bad("by"))?,
        inferred: get("inferred").and_then(|v| v.as_bool()).unwrap_or(false),
        confidence: get("confidence")
            .and_then(|v| v.as_float64())
            .unwrap_or(1.0),
        snapshot_name: opt_non_vide(chaine("snapshot_name")),
        snapshot_path: opt_non_vide(chaine("snapshot_path")),
        snapshot_type: opt_non_vide(chaine("snapshot_type")),
        rev: opt_non_vide(chaine("rev")),
        version: non_signe(get("version").and_then(|v| v.as_int64()).unwrap_or(1)).max(1),
        created_at: parse_ts(&requis("created_at")?)?,
        updated_at: parse_ts(&requis("updated_at")?)?,
    })
}

fn parse_ts(s: &str) -> Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(s)?.with_timezone(&Utc))
}

fn proprietes_evenement_ancre(ev: &AnchorEvent) -> Vec<(&'static str, Value)> {
    let json =
        |v: &Option<serde_json::Value>| v.as_ref().map(|v| v.to_string()).unwrap_or_default();
    vec![
        ("id", Value::from(ev.id.to_string())),
        ("session_id", Value::from(ev.session_id.to_string())),
        ("anchor_id", Value::from(ev.anchor_id.to_string())),
        ("seq", Value::from(ev.seq as i64)),
        ("kind", Value::from(ev.kind.as_str())),
        ("by", Value::from(ev.by.as_str())),
        ("actor", Value::from(ev.actor.clone())),
        ("at", Value::from(anchor::format_ts(ev.at))),
        ("before", Value::from(json(&ev.before))),
        ("after", Value::from(json(&ev.after))),
    ]
}

fn lire_evenement_ancre(get: &dyn Fn(&str) -> Option<Value>) -> Result<AnchorEvent> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let requis = |cle: &str| chaine(cle).ok_or_else(|| anyhow!("journal d'ancre sans `{cle}`"));
    let bad = |q: &str| anyhow!("journal d'ancre corrompu : {q}");
    let json = |cle: &str| opt_non_vide(chaine(cle)).and_then(|j| serde_json::from_str(&j).ok());
    Ok(AnchorEvent {
        id: requis("id")?.parse()?,
        seq: non_signe(get("seq").and_then(|v| v.as_int64()).unwrap_or(0)),
        session_id: requis("session_id")?.parse()?,
        anchor_id: requis("anchor_id")?.parse()?,
        kind: AnchorEventKind::parse(&requis("kind")?).ok_or_else(|| bad("kind"))?,
        by: AnchorActor::parse(&requis("by")?).ok_or_else(|| bad("by"))?,
        actor: requis("actor")?,
        at: parse_ts(&requis("at")?)?,
        before: json("before"),
        after: json("after"),
    })
}

// ============================================================================
// Execution-place events <-> properties
// ============================================================================

fn proprietes_lieu(e: &ExecutionPlaceEvent) -> Vec<(&'static str, Value)> {
    vec![
        ("id", Value::from(e.id.to_string())),
        ("placement_id", Value::from(e.placement_id.to_string())),
        ("kind", Value::from(e.kind.as_str())),
        ("task_id", Value::from(e.task_id.to_string())),
        ("plan_id", Value::from(e.plan_id.to_string())),
        ("requested_by", Value::from(e.requested_by.clone())),
        ("owner", Value::from(e.owner.clone())),
        ("actor", Value::from(e.actor.clone().unwrap_or_default())),
        (
            "session_id",
            Value::from(e.session_id.map(|s| s.to_string()).unwrap_or_default()),
        ),
        ("path", Value::from(e.path.clone())),
        ("repository", Value::from(e.repository.clone())),
        ("base_ref", Value::from(e.base_ref.clone())),
        ("base_sha", Value::from(e.base_sha.clone())),
        ("branch", Value::from(e.branch.clone())),
        ("detail", Value::from(e.detail.clone().unwrap_or_default())),
        ("at", Value::from(format_lieu_at(e.at))),
    ]
}

/// `at` as Neo4j writes it (`create_execution_place_event`): microseconds, `Z`.
fn format_lieu_at(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(chrono::SecondsFormat::Micros, true)
}

fn lire_lieu(get: &dyn Fn(&str) -> Option<Value>) -> Result<ExecutionPlaceEvent> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let requis = |cle: &str| chaine(cle).ok_or_else(|| anyhow!("événement de lieu sans `{cle}`"));
    let uuid = |cle: &str| requis(cle).and_then(|s| s.parse().map_err(|e| anyhow!("{cle} : {e}")));
    Ok(ExecutionPlaceEvent {
        id: uuid("id")?,
        placement_id: uuid("placement_id")?,
        kind: ExecutionPlaceEventKind::parse(&requis("kind")?)
            .ok_or_else(|| anyhow!("genre d'événement de lieu inconnu"))?,
        task_id: uuid("task_id")?,
        plan_id: uuid("plan_id")?,
        requested_by: requis("requested_by")?,
        owner: requis("owner")?,
        actor: opt_non_vide(chaine("actor")),
        session_id: opt_non_vide(chaine("session_id")).and_then(|s| s.parse().ok()),
        path: requis("path")?,
        repository: requis("repository")?,
        base_ref: requis("base_ref")?,
        base_sha: requis("base_sha")?,
        branch: requis("branch")?,
        detail: opt_non_vide(chaine("detail")),
        at: parse_ts(&requis("at")?)?,
    })
}

// ============================================================================
// Entries and index
// ============================================================================

/// The `SPAWNED_BY` edge of a session to its parent, with the metadata the edge
/// carries (absent for a bare edge written by `create_chat_session`).
#[derive(Clone)]
struct Parentage {
    arete: EdgeId,
    parent_id: Uuid,
    spawn_type: Option<String>,
    run_id: Option<Uuid>,
    task_id: Option<Uuid>,
}

/// One chat session: where it lives in the base, its decoded state, and the
/// extra scalar properties the node carries beside `ChatSessionNode`.
#[derive(Clone)]
pub(super) struct EntreeSession {
    noeud: NodeId,
    session: ChatSessionNode,
    parent: Option<Parentage>,
    shell_moved: Option<bool>,
    anchor_seq: u64,
    children_cost_usd: f64,
    resume_fingerprint: Option<String>,
}

#[derive(Clone)]
struct EntreeEvenement {
    noeud: NodeId,
    evenement: ChatEventRecord,
}

#[derive(Clone)]
struct EntreeReglage {
    noeud: NodeId,
    valeur: String,
}

#[derive(Clone)]
struct EntreeEpisode {
    noeud: NodeId,
    projet: Uuid,
    episode: Episode,
}

#[derive(Clone)]
struct EntreeAncre {
    noeud: NodeId,
    ancre: Anchor,
}

#[derive(Clone)]
struct EntreeEvenementAncre {
    noeud: NodeId,
    evenement: AnchorEvent,
}

#[derive(Clone)]
struct EntreeLieu {
    evenement: ExecutionPlaceEvent,
}

/// The index of the chat domain.
#[derive(Default)]
pub(super) struct IndexChat {
    /// Ids of session nodes on disk that could not be read at open (quarantined).
    reserves: HashSet<Uuid>,
    sessions: HashMap<Uuid, EntreeSession>,
    /// parent session id → child session ids (SPAWNED_BY edge).
    enfants_de: HashMap<Uuid, BTreeSet<Uuid>>,
    /// session id → its events, ordered by seq.
    evenements: HashMap<Uuid, Vec<EntreeEvenement>>,
    /// (scope, key) → LlmSetting node.
    reglages: HashMap<(String, String), EntreeReglage>,
    /// session id → its episode (one per session).
    episodes: HashMap<Uuid, EntreeEpisode>,
    ancres: HashMap<Uuid, EntreeAncre>,
    ancres_par_session: HashMap<Uuid, BTreeSet<Uuid>>,
    ancres_par_cible: HashMap<(String, String), BTreeSet<Uuid>>,
    evenements_ancre: HashMap<Uuid, Vec<EntreeEvenementAncre>>,
    evenements_lieu: HashMap<Uuid, Vec<EntreeLieu>>,
    /// conversation → segment id → entry.
    segments: HashMap<String, BTreeMap<String, EntreeSegment>>,
    /// conversation → its segment nodes that could not be read at open
    /// (quarantined). The conversation's segments are refused until a `put`
    /// replaces the set (and deletes these nodes).
    segments_illisibles: HashMap<String, Vec<NodeId>>,
    /// conversation → key (`dimension#at_turn`) → boundary.
    frontieres: HashMap<String, BTreeMap<String, EntreeFrontiere>>,
    /// Same as `segments_illisibles`, for the boundaries.
    frontieres_illisibles: HashMap<String, Vec<NodeId>>,
    racine_agent: Option<Uuid>,
    noeud_racine_agent: Option<NodeId>,
}

#[derive(Clone)]
struct EntreeFrontiere {
    noeud: NodeId,
    frontiere: Boundary,
}

#[derive(Clone)]
struct EntreeSegment {
    noeud: NodeId,
    segment: Segment,
}

/// The parent carried by the `spawned_by` JSON of a session, as the mock reads it.
fn parent_de(
    session: &ChatSessionNode,
) -> (Option<String>, Option<String>, Option<Uuid>, Option<Uuid>) {
    let value: Option<serde_json::Value> = session
        .spawned_by
        .as_deref()
        .and_then(|raw| serde_json::from_str(raw).ok());
    let text = |key: &str| {
        value
            .as_ref()
            .and_then(|v| v.get(key))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    };
    (
        text("parent_session_id"),
        text("type"),
        text("run_id").and_then(|s| s.parse().ok()),
        text("task_id").and_then(|s| s.parse().ok()),
    )
}

impl EntreeSession {
    fn decoder(db: &ObrainDB, noeud: NodeId, parent: Option<Parentage>) -> Result<Self> {
        let get = lecteur(db, noeud)?;
        let session = lire_session(&get)?;
        Ok(Self {
            noeud,
            session,
            parent,
            shell_moved: get("shell_moved").and_then(|v| v.as_bool()),
            anchor_seq: non_signe(get("anchor_seq").and_then(|v| v.as_int64()).unwrap_or(0)),
            children_cost_usd: get("children_cost_usd")
                .and_then(|v| v.as_float64())
                .unwrap_or(0.0),
            resume_fingerprint: opt_non_vide(
                get("resume_fingerprint").and_then(|v| v.as_str().map(str::to_string)),
            ),
        })
    }
}

/// At open: a node left by an interrupted creation — its draft marker still set,
/// or neither property nor edge — is deleted (edges first) and counted. Returns
/// whether it was.
fn purger_brouillon(db: &ObrainDB, noeud: NodeId, rapport: &mut RapportOuverture) -> bool {
    let store = db.store();
    let Some(n) = db.get_node(noeud) else {
        return false;
    };
    let brouillon = n
        .get_property(BROUILLON)
        .is_some_and(|v| !matches!(v, Value::Null));
    if !brouillon && (!n.properties.is_empty() || store.out_degree(noeud) != 0) {
        return false;
    }
    for (_, arete) in store.edges_from(noeud, Direction::Outgoing) {
        db.delete_edge(arete);
    }
    for (_, arete) in store.edges_from(noeud, Direction::Incoming) {
        db.delete_edge(arete);
    }
    db.delete_node(noeud);
    rapport.brouillons_supprimes += 1;
    true
}

impl IndexChat {
    fn inscrire_session(&mut self, e: EntreeSession) {
        let id = e.session.id;
        if let Some(p) = &e.parent {
            self.enfants_de.entry(p.parent_id).or_default().insert(id);
        }
        self.sessions.insert(id, e);
    }

    fn desinscrire_session(&mut self, id: Uuid) -> Option<EntreeSession> {
        let e = self.sessions.remove(&id)?;
        if let Some(p) = &e.parent {
            if let Some(s) = self.enfants_de.get_mut(&p.parent_id) {
                s.remove(&id);
                if s.is_empty() {
                    self.enfants_de.remove(&p.parent_id);
                }
            }
        }
        Some(e)
    }

    /// Publish a new state of a session (after its durability step).
    fn publier_session(&mut self, e: EntreeSession) {
        self.desinscrire_session(e.session.id);
        self.inscrire_session(e);
    }

    /// Rebuild the index from the persisted nodes at open. A scan of each label.
    pub(super) fn reconstruire(db: &ObrainDB, rapport: &mut RapportOuverture) -> Result<Self> {
        let store = db.store();
        let mut index = Self::default();
        let mut vus: Vec<(NodeId, Uuid)> = Vec::new();

        // Sessions, then their SPAWNED_BY edges.
        for noeud in store.nodes_by_label(SESSION) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud ChatSession {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            if purger_brouillon(db, noeud, rapport) {
                continue;
            }
            match EntreeSession::decoder(db, noeud, None) {
                Ok(e) => {
                    let id = e.session.id;
                    if index.sessions.contains_key(&id) {
                        rapport
                            .en_quarantaine
                            .push(format!("nœud ChatSession {noeud:?} : id déjà pris"));
                        continue;
                    }
                    index.inscrire_session(e);
                    vus.push((noeud, id));
                }
                Err(e) => {
                    if let Some(id) =
                        get("id").and_then(|v| v.as_str().and_then(|t| t.parse::<Uuid>().ok()))
                    {
                        index.reserves.insert(id);
                    }
                    rapport
                        .en_quarantaine
                        .push(format!("nœud ChatSession {noeud:?} : {e}"));
                    continue;
                }
            }
        }

        // The SPAWNED_BY edges, re-linked after every session is indexed.
        let mut liens: Vec<(Uuid, Parentage)> = Vec::new();
        for (noeud, id) in &vus {
            for (cible, arete) in store.edges_from(*noeud, Direction::Outgoing) {
                let Some(e) = db.get_edge(arete) else {
                    continue;
                };
                if &*e.edge_type != ARETE_SPAWNED {
                    continue;
                }
                let Some(cible_id) = db.get_node(cible).and_then(|n| {
                    n.get_property("id")
                        .and_then(|v| v.as_str().map(str::to_string))
                }) else {
                    continue;
                };
                let Ok(parent_id) = cible_id.parse::<Uuid>() else {
                    continue;
                };
                if !index.sessions.contains_key(&parent_id) {
                    continue;
                }
                let texte = |cle: &str| {
                    e.get_property(cle)
                        .and_then(|v| v.as_str().map(str::to_string))
                        .filter(|s| !s.is_empty())
                };
                liens.push((
                    *id,
                    Parentage {
                        arete,
                        parent_id,
                        spawn_type: texte("type"),
                        run_id: texte("run_id").and_then(|s| s.parse().ok()),
                        task_id: texte("task_id").and_then(|s| s.parse().ok()),
                    },
                ));
            }
        }
        for (id, parentage) in liens {
            if let Some(mut e) = index.desinscrire_session(id) {
                e.parent = Some(parentage);
                index.inscrire_session(e);
            }
        }

        // Events.
        let mut evenements: Vec<(Uuid, EntreeEvenement)> = Vec::new();
        for noeud in store.nodes_by_label(EVENEMENT) {
            if purger_brouillon(db, noeud, rapport) {
                continue;
            }
            let Some(n) = db.get_node(noeud) else {
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            match lire_evenement(&get) {
                Ok(ev) => evenements.push((
                    ev.session_id,
                    EntreeEvenement {
                        noeud,
                        evenement: ev,
                    },
                )),
                Err(e) => {
                    rapport
                        .en_quarantaine
                        .push(format!("nœud ChatEvent {noeud:?} : {e}"));
                }
            }
        }
        evenements.sort_by(|a, b| a.1.evenement.seq.cmp(&b.1.evenement.seq));
        for (sid, e) in evenements {
            index.evenements.entry(sid).or_default().push(e);
        }

        // LlmSetting.
        for noeud in store.nodes_by_label(REGLAGE) {
            if purger_brouillon(db, noeud, rapport) {
                continue;
            }
            let Some(n) = db.get_node(noeud) else {
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            let texte = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
            // `value` is required: a setting read back `""` would be a value nobody
            // wrote (Neo4j writes the three in one `MERGE … SET`).
            let (Some(scope), Some(key), Some(valeur)) =
                (texte("scope"), texte("key"), texte("value"))
            else {
                rapport.en_quarantaine.push(format!(
                    "nœud LlmSetting {noeud:?} : sans scope, key ou value"
                ));
                continue;
            };
            index
                .reglages
                .insert((scope, key), EntreeReglage { noeud, valeur });
        }

        // Episodes.
        for noeud in store.nodes_by_label(EPISODE) {
            if purger_brouillon(db, noeud, rapport) {
                continue;
            }
            let Some(n) = db.get_node(noeud) else {
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            let texte = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
            let (Some(sid), Some(payload)) = (texte("session_id"), texte("payload")) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud SessionEpisode {noeud:?} : incomplet"));
                continue;
            };
            let Ok(sid) = sid.parse::<Uuid>() else {
                rapport.en_quarantaine.push(format!(
                    "nœud SessionEpisode {noeud:?} : session_id illisible"
                ));
                continue;
            };
            match serde_json::from_str::<Episode>(&payload) {
                Ok(episode) => {
                    // The project is the payload's, not the `project_id` copy: a
                    // rewrite interrupted between the two cannot list the episode
                    // under a project its payload does not name.
                    index.episodes.insert(
                        sid,
                        EntreeEpisode {
                            noeud,
                            projet: episode.project_id,
                            episode,
                        },
                    );
                }
                Err(e) => {
                    rapport
                        .en_quarantaine
                        .push(format!("nœud SessionEpisode {noeud:?} : {e}"));
                }
            }
        }

        // Anchors.
        let mut ancres: Vec<EntreeAncre> = Vec::new();
        for noeud in store.nodes_by_label(ANCRE) {
            if purger_brouillon(db, noeud, rapport) {
                continue;
            }
            let Some(n) = db.get_node(noeud) else {
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            match lire_ancre(&get) {
                Ok(a) => ancres.push(EntreeAncre { noeud, ancre: a }),
                Err(e) => {
                    rapport
                        .en_quarantaine
                        .push(format!("nœud Anchor {noeud:?} : {e}"));
                }
            }
        }
        for e in ancres {
            index
                .ancres_par_session
                .entry(e.ancre.session_id)
                .or_default()
                .insert(e.ancre.id);
            index
                .ancres_par_cible
                .entry((
                    e.ancre.target_type.as_str().to_string(),
                    e.ancre.target_id.clone(),
                ))
                .or_default()
                .insert(e.ancre.id);
            index.ancres.insert(e.ancre.id, e);
        }

        // Anchor journal.
        let mut evt_ancres: Vec<EntreeEvenementAncre> = Vec::new();
        for noeud in store.nodes_by_label(EVENEMENT_ANCRE) {
            if purger_brouillon(db, noeud, rapport) {
                continue;
            }
            let Some(n) = db.get_node(noeud) else {
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            match lire_evenement_ancre(&get) {
                Ok(ev) => evt_ancres.push(EntreeEvenementAncre {
                    noeud,
                    evenement: ev,
                }),
                Err(e) => {
                    rapport
                        .en_quarantaine
                        .push(format!("nœud AnchorEvent {noeud:?} : {e}"));
                }
            }
        }
        evt_ancres.sort_by(|a, b| {
            (a.evenement.seq, a.evenement.at, a.evenement.id).cmp(&(
                b.evenement.seq,
                b.evenement.at,
                b.evenement.id,
            ))
        });
        for e in evt_ancres {
            index
                .evenements_ancre
                .entry(e.evenement.session_id)
                .or_default()
                .push(e);
        }

        // Execution-place events.
        let mut lieux: Vec<EntreeLieu> = Vec::new();
        for noeud in store.nodes_by_label(LIEU) {
            if purger_brouillon(db, noeud, rapport) {
                continue;
            }
            let Some(n) = db.get_node(noeud) else {
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            match lire_lieu(&get) {
                Ok(ev) => lieux.push(EntreeLieu { evenement: ev }),
                Err(e) => {
                    rapport
                        .en_quarantaine
                        .push(format!("nœud ExecutionPlaceEvent {noeud:?} : {e}"));
                }
            }
        }
        for e in lieux {
            index
                .evenements_lieu
                .entry(e.evenement.task_id)
                .or_default()
                .push(e);
        }

        // Segments.
        for noeud in store.nodes_by_label(SEGMENT) {
            if purger_brouillon(db, noeud, rapport) {
                continue;
            }
            let Some(n) = db.get_node(noeud) else {
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            let texte = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
            let Some(conversation) = texte("conversation") else {
                rapport.en_quarantaine.push(format!(
                    "nœud ConversationSegment {noeud:?} : sans conversation"
                ));
                continue;
            };
            // An unreadable segment is an ERROR for its conversation, never a
            // silently shorter history (as `get_segments_impl` on Neo4j).
            let lu = texte("id")
                .zip(texte("json"))
                .ok_or_else(|| anyhow!("sans id ou json"))
                .and_then(|(id, json)| Ok((id, serde_json::from_str::<Segment>(&json)?)));
            match lu {
                Ok((id, segment)) => {
                    index
                        .segments
                        .entry(conversation)
                        .or_default()
                        .insert(id, EntreeSegment { noeud, segment });
                }
                Err(e) => {
                    rapport.en_quarantaine.push(format!(
                        "nœud ConversationSegment {noeud:?} ({conversation}) : {e}"
                    ));
                    index
                        .segments_illisibles
                        .entry(conversation)
                        .or_default()
                        .push(noeud);
                }
            }
        }

        // Boundaries.
        for noeud in store.nodes_by_label(FRONTIERE) {
            if purger_brouillon(db, noeud, rapport) {
                continue;
            }
            let Some(n) = db.get_node(noeud) else {
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            let texte = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
            let Some(conversation) = texte("conversation") else {
                rapport.en_quarantaine.push(format!(
                    "nœud ConversationBoundary {noeud:?} : sans conversation"
                ));
                continue;
            };
            let lu = texte("json")
                .ok_or_else(|| anyhow!("sans json"))
                .and_then(|json| Ok(serde_json::from_str::<Boundary>(&json)?));
            match lu {
                Ok(frontiere) => {
                    let cle = format!("{}#{}", frontiere.dimension.as_str(), frontiere.at_turn);
                    index
                        .frontieres
                        .entry(conversation)
                        .or_default()
                        .insert(cle, EntreeFrontiere { noeud, frontiere });
                }
                Err(e) => {
                    rapport.en_quarantaine.push(format!(
                        "nœud ConversationBoundary {noeud:?} ({conversation}) : {e}"
                    ));
                    index
                        .frontieres_illisibles
                        .entry(conversation)
                        .or_default()
                        .push(noeud);
                }
            }
        }

        // Agent identity root.
        for noeud in store.nodes_by_label(RACINE_AGENT) {
            if purger_brouillon(db, noeud, rapport) {
                continue;
            }
            let Some(n) = db.get_node(noeud) else {
                continue;
            };
            let Some(ns) = n
                .get_property("namespace")
                .and_then(|v| v.as_str().and_then(|t| t.parse::<Uuid>().ok()))
            else {
                continue;
            };
            index.racine_agent = Some(ns);
            index.noeud_racine_agent = Some(noeud);
            break;
        }

        Ok(index)
    }
}

// ============================================================================
// Helpers on the store
// ============================================================================

impl ObrainGraphStore {
    fn entree_session_de(w: &Ecriture<'_>, id: Uuid) -> Result<Option<EntreeSession>> {
        Ok(w.index()?.chat.sessions.get(&id).cloned())
    }

    fn relire_session(w: &Ecriture<'_>, e: &EntreeSession) -> Result<EntreeSession> {
        w.apres_ecriture(EntreeSession::decoder(w.db, e.noeud, e.parent.clone()))
    }

    fn noeud_session(w: &Ecriture<'_>, id: Uuid) -> Result<Option<NodeId>> {
        Ok(w.index()?.chat.sessions.get(&id).map(|e| e.noeud))
    }
}

/// Create a node the way a creation must (see the module doc): the draft marker
/// first, the properties, the marker removed last (the commit point). A step
/// before each engine write, for the interruption tests.
fn creer_noeud(
    w: &mut Ecriture<'_>,
    etiquette: &'static str,
    proprietes: Vec<(&'static str, Value)>,
) -> Result<NodeId> {
    let noeud = w.db.create_node(&[etiquette]);
    w.etape()?;
    w.db.set_node_property(noeud, BROUILLON, Value::from(true));
    for (cle, valeur) in proprietes {
        w.etape()?;
        w.db.set_node_property(noeud, cle, valeur);
    }
    w.rendre_durable_avant_le_marqueur()?;
    w.etape()?;
    w.db.remove_node_property(noeud, BROUILLON);
    Ok(noeud)
}

/// Write properties of an existing node, a step before each.
fn poser(
    w: &mut Ecriture<'_>,
    noeud: NodeId,
    proprietes: Vec<(&'static str, Value)>,
) -> Result<()> {
    for (cle, valeur) in proprietes {
        w.etape()?;
        w.db.set_node_property(noeud, cle, valeur);
    }
    Ok(())
}

/// DETACH DELETE of a node: its edges go first, then the node itself.
fn detacher(w: &mut Ecriture<'_>, noeud: NodeId) -> Result<()> {
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
        w.interne
            .indetermine
            .store(true, std::sync::atomic::Ordering::Release);
        bail!(
            "obrain a refusé la suppression du nœud {noeud:?} APRÈS la suppression de ses \
             arêtes : base indéterminée, redémarrer"
        );
    }
    Ok(())
}

// ============================================================================
// ChatStore
// ============================================================================

/// The ported core of `ChatStore`. The name of the trait method is the name of
/// its refusal, so a refusal is findable by grepping the contract.
#[async_trait]
impl ChatStore for ObrainGraphStore {
    // ------------------------------------------------------------------
    // Sessions
    // ------------------------------------------------------------------

    async fn create_chat_session(&self, session: &ChatSessionNode) -> Result<()> {
        let n = session.clone();
        self.ecrire(
            move || n,
            move |n, w: &mut Ecriture<'_>| {
                {
                    let etat = w.index()?;
                    if etat.chat.sessions.contains_key(&n.id) {
                        bail!("une session d'id {} existe déjà", n.id);
                    }
                    if etat.chat.reserves.contains(&n.id) {
                        bail!(
                            "l'id {} est réservé : un nœud ChatSession de cet id est sur disque, \
                             illisible (quarantaine)",
                            n.id
                        );
                    }
                }
                let proprietes = proprietes_session(&n);
                w.flush_complet = proprietes.iter().any(|(_, v)| va_en_colonne_blob(v));
                let noeud = w.db.create_node(&[SESSION]);
                w.etape()?;
                w.db.set_node_property(noeud, BROUILLON, Value::from(true));
                for (cle, valeur) in proprietes {
                    w.etape()?;
                    w.db.set_node_property(noeud, cle, valeur);
                }
                // The lineage edge, from the `spawned_by` JSON (bare: created_at only).
                let parent = n
                    .spawned_by
                    .as_deref()
                    .and_then(crate::chat::types::SpawnedBy::from_json_str)
                    .and_then(|sb| sb.parent_session_id());
                let mut parentage = None;
                if let Some(pid) = parent {
                    let pnoeud = w.index()?.chat.sessions.get(&pid).map(|e| e.noeud);
                    if let Some(pnoeud) = pnoeud {
                        let arete = w.db.create_edge(noeud, pnoeud, ARETE_SPAWNED);
                        w.etape()?;
                        w.db.set_edge_property(
                            arete,
                            "created_at",
                            Value::from(Utc::now().to_rfc3339()),
                        );
                        parentage = Some(Parentage {
                            arete,
                            parent_id: pid,
                            spawn_type: None,
                            run_id: None,
                            task_id: None,
                        });
                    }
                }
                w.rendre_durable_avant_le_marqueur()?;
                w.etape()?;
                w.db.remove_node_property(noeud, BROUILLON);
                let entree = EntreeSession::decoder(w.db, noeud, parentage).and_then(|e| {
                    if e.session.id == n.id {
                        Ok(e)
                    } else {
                        bail!("la session relue n'a pas l'id écrit")
                    }
                });
                let entree = w.apres_ecriture(entree)?;
                w.rendre_durable()?;
                w.index_mut()?.chat.publier_session(entree);
                Ok(())
            },
        )
        .await
    }

    async fn get_chat_session(&self, id: Uuid) -> Result<Option<ChatSessionNode>> {
        self.lire(|_, etat| Ok(etat.chat.sessions.get(&id).map(|e| e.session.clone())))
    }

    async fn list_chat_sessions(
        &self,
        project_slug: Option<&str>,
        workspace_slug: Option<&str>,
        limit: usize,
        offset: usize,
        include_detached: bool,
    ) -> Result<(Vec<ChatSessionNode>, usize)> {
        let ps = project_slug.map(str::to_string);
        let ws = workspace_slug.map(str::to_string);
        self.lire(move |_, etat| {
            let ws_project_slugs: HashSet<String> = if let Some(ws) = &ws {
                etat.espaces
                    .lignes_projets_espaces(&etat.projets)
                    .into_iter()
                    .filter(|r| &r.workspace_slug == ws)
                    .map(|r| r.project_slug)
                    .collect()
            } else {
                HashSet::new()
            };
            let mut filtree: Vec<ChatSessionNode> = etat
                .chat
                .sessions
                .values()
                .filter(|e| {
                    if !include_detached && (e.session.spawned_by.is_some() || e.parent.is_some()) {
                        return false;
                    }
                    if let Some(slug) = &ps {
                        e.session.project_slug.as_deref() == Some(slug.as_str())
                    } else if let Some(ws) = &ws {
                        e.session.workspace_slug.as_deref() == Some(ws.as_str())
                            || e.session
                                .project_slug
                                .as_deref()
                                .is_some_and(|p| ws_project_slugs.contains(p))
                    } else {
                        true
                    }
                })
                .map(|e| e.session.clone())
                .collect();
            filtree.sort_by(|a, b| b.updated_at.cmp(&a.updated_at).then(b.id.cmp(&a.id)));
            let total = filtree.len();
            let page = filtree.into_iter().skip(offset).take(limit).collect();
            Ok((page, total))
        })
    }

    async fn get_session_children(&self, parent_id: Uuid) -> Result<Vec<ChatSessionNode>> {
        self.lire(move |_, etat| {
            let parent_str = parent_id.to_string();
            // A parent that no longer exists has no children.
            if !etat.chat.sessions.contains_key(&parent_id) {
                return Ok(Vec::new());
            }
            let mut enfants: Vec<ChatSessionNode> = etat
                .chat
                .sessions
                .values()
                .filter(|e| parent_de(&e.session).0.as_deref() == Some(parent_str.as_str()))
                .map(|e| e.session.clone())
                .collect();
            enfants.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
            Ok(enfants)
        })
    }

    async fn create_spawned_by_relation(
        &self,
        child_session_id: &str,
        parent_session_id: &str,
        spawn_type: &str,
        run_id: Option<Uuid>,
        task_id: Option<Uuid>,
    ) -> Result<()> {
        let (cid, pid, st) = (
            child_session_id.to_string(),
            parent_session_id.to_string(),
            spawn_type.to_string(),
        );
        self.ecrire(
            move || (),
            move |(), w: &mut Ecriture<'_>| {
                let (Ok(cid), Ok(pid)) = (cid.parse::<Uuid>(), pid.parse::<Uuid>()) else {
                    return Ok(());
                };
                let (cnoeud, pnoeud, existant) = {
                    let etat = w.index()?;
                    let (Some(c), Some(p)) =
                        (etat.chat.sessions.get(&cid), etat.chat.sessions.get(&pid))
                    else {
                        return Ok(());
                    };
                    (
                        c.noeud,
                        p.noeud,
                        c.parent.as_ref().map(|x| (x.arete, x.parent_id)),
                    )
                };
                // The index holds ONE parent per session: an edge to another parent
                // is refused, never repointed in the index alone (see module doc).
                if let Some((_, autre)) = existant.filter(|(_, p)| *p != pid) {
                    bail!(
                        "la session {cid} a déjà un SPAWNED_BY vers {autre} : un second parent                          ({pid}) est refusé"
                    );
                }
                let arete = match existant {
                    Some((a, _)) => a,
                    None => {
                        let a = w.db.create_edge(cnoeud, pnoeud, ARETE_SPAWNED);
                        w.etape()?;
                        w.db.set_edge_property(
                            a,
                            "created_at",
                            Value::from(Utc::now().to_rfc3339()),
                        );
                        a
                    }
                };
                let ancien = {
                    let etat = w.index()?;
                    etat.chat.sessions.get(&cid).and_then(|e| e.parent.clone())
                };
                // A call that carries a run is authoritative; one that does not only
                // fills what is missing.
                let type_final = if run_id.is_some() {
                    Some(st.clone())
                } else {
                    ancien
                        .as_ref()
                        .and_then(|p| p.spawn_type.clone())
                        .or_else(|| Some(st.clone()))
                };
                let run_final = run_id.or_else(|| ancien.as_ref().and_then(|p| p.run_id));
                let task_final = task_id.or_else(|| ancien.as_ref().and_then(|p| p.task_id));
                if let Some(t) = &type_final {
                    w.etape()?;
                    w.db.set_edge_property(arete, "type", Value::from(t.clone()));
                }
                if let Some(r) = run_final {
                    w.etape()?;
                    w.db.set_edge_property(arete, "run_id", Value::from(r.to_string()));
                }
                if let Some(k) = task_final {
                    w.etape()?;
                    w.db.set_edge_property(arete, "task_id", Value::from(k.to_string()));
                }
                w.rendre_durable()?;
                let parentage = Parentage {
                    arete,
                    parent_id: pid,
                    spawn_type: type_final,
                    run_id: run_final,
                    task_id: task_final,
                };
                let mut etat = w.index_mut()?;
                if let Some(mut e) = etat.chat.desinscrire_session(cid) {
                    e.parent = Some(parentage);
                    etat.chat.inscrire_session(e);
                }
                Ok(())
            },
        )
        .await
    }

    async fn get_session_tree(&self, session_id: &str) -> Result<Vec<SessionTreeNode>> {
        let sid = session_id.to_string();
        self.lire(move |_, etat| {
            let sessions = &etat.chat.sessions;
            let Some(root) = sessions.values().find(|e| e.session.id.to_string() == sid) else {
                return Ok(Vec::new());
            };
            let node_of = |e: &EntreeSession, depth: u32| {
                let (parent, kind, run_id, task_id) = parent_de(&e.session);
                SessionTreeNode {
                    session_id: e.session.id.to_string(),
                    parent_session_id: if depth == 0 { None } else { parent },
                    spawn_type: if depth == 0 { None } else { kind },
                    run_id: if depth == 0 { None } else { run_id },
                    task_id: if depth == 0 { None } else { task_id },
                    depth,
                    created_at: Some(e.session.created_at),
                    provider_id: None,
                    model: None,
                    cost_usd: None,
                    subtree_cost_usd: None,
                    max_depth: None,
                    max_children: None,
                    status: "closed".to_string(),
                    routed_by: None,
                }
            };
            let mut nodes = vec![node_of(root, 0)];
            let mut frontier = vec![root.session.id.to_string()];
            let mut seen: HashSet<String> = frontier.iter().cloned().collect();
            let mut depth = 0u32;
            while !frontier.is_empty() && depth < 10 {
                depth += 1;
                let mut next = Vec::new();
                // The whole depth at once, ordered as Neo4j orders it:
                // `depth, created_at, id` (never by parent, never by map order).
                let parents: HashSet<&str> = frontier.iter().map(String::as_str).collect();
                let mut children: Vec<&EntreeSession> = sessions
                    .values()
                    .filter(|e| {
                        parent_de(&e.session)
                            .0
                            .as_deref()
                            .is_some_and(|p| parents.contains(p))
                    })
                    .filter(|e| !seen.contains(&e.session.id.to_string()))
                    .collect();
                children.sort_by_key(|e| (e.session.created_at, e.session.id));
                for child in children {
                    seen.insert(child.session.id.to_string());
                    nodes.push(node_of(child, depth));
                    next.push(child.session.id.to_string());
                }
                frontier = next;
            }
            Ok(nodes)
        })
    }

    async fn get_session_root(&self, session_id: &str) -> Result<Option<String>> {
        let sid = session_id.to_string();
        self.lire(move |_, etat| {
            let sessions = &etat.chat.sessions;
            let by_id = |id: &str| sessions.values().find(|e| e.session.id.to_string() == id);
            let Some(mut current) = by_id(&sid) else {
                return Ok(None);
            };
            let mut seen = HashSet::from([current.session.id.to_string()]);
            for _ in 0..10 {
                let Some(parent_id) = parent_de(&current.session).0 else {
                    break;
                };
                if !seen.insert(parent_id.clone()) {
                    break;
                }
                match by_id(&parent_id) {
                    Some(parent) => current = parent,
                    None => break,
                }
            }
            Ok(Some(current.session.id.to_string()))
        })
    }

    async fn get_run_sessions(&self, run_id: Uuid) -> Result<Vec<SessionInfo>> {
        self.lire(move |_, etat| {
            let mut out = Vec::new();
            for e in etat.chat.sessions.values() {
                if e.parent.as_ref().and_then(|p| p.run_id) == Some(run_id) {
                    out.push(SessionInfo {
                        session_id: e.session.id.to_string(),
                        title: e.session.title.clone(),
                        model: e.session.model.clone(),
                        spawn_type: e.parent.as_ref().and_then(|p| p.spawn_type.clone()),
                        task_id: e.parent.as_ref().and_then(|p| p.task_id),
                        created_at: e.session.created_at,
                    });
                }
            }
            out.sort_by(|a, b| {
                a.created_at
                    .cmp(&b.created_at)
                    .then(a.session_id.cmp(&b.session_id))
            });
            if !out.is_empty() {
                return Ok(out);
            }
            let rid = run_id.to_string();
            for e in etat.chat.sessions.values() {
                let sb = e.session.spawned_by.as_deref().unwrap_or_default();
                if !sb.is_empty() && sb.contains(&rid) {
                    out.push(SessionInfo {
                        session_id: e.session.id.to_string(),
                        title: e.session.title.clone(),
                        model: e.session.model.clone(),
                        spawn_type: Some("runner".to_string()),
                        task_id: parent_de(&e.session).3,
                        created_at: e.session.created_at,
                    });
                }
            }
            out.sort_by(|a, b| {
                a.created_at
                    .cmp(&b.created_at)
                    .then(a.session_id.cmp(&b.session_id))
            });
            Ok(out)
        })
    }

    async fn list_project_workspace_rows(&self) -> Result<Vec<ProjectWorkspaceRow>> {
        self.lire(|_, etat| Ok(etat.espaces.lignes_projets_espaces(&etat.projets)))
    }

    #[allow(clippy::too_many_arguments)]
    async fn update_chat_session(
        &self,
        id: Uuid,
        cli_session_id: Option<String>,
        title: Option<String>,
        message_count: Option<i64>,
        total_cost_usd: Option<f64>,
        conversation_id: Option<String>,
        preview: Option<String>,
    ) -> Result<Option<ChatSessionNode>> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(ancienne) = Self::entree_session_de(w, id)? else {
                    return Ok(None);
                };
                let noeud = ancienne.noeud;
                w.flush_complet = preview
                    .as_ref()
                    .is_some_and(|p| p.len() > SEUIL_COLONNE_BLOB)
                    || ancienne
                        .session
                        .preview
                        .as_ref()
                        .is_some_and(|p| p.len() > SEUIL_COLONNE_BLOB);
                w.etape()?;
                w.db.set_node_property(noeud, "updated_at", Value::from(Utc::now().to_rfc3339()));
                if let Some(v) = cli_session_id {
                    w.etape()?;
                    w.db.set_node_property(noeud, "cli_session_id", Value::from(v));
                }
                if let Some(v) = title {
                    w.etape()?;
                    w.db.set_node_property(noeud, "title", Value::from(v));
                }
                if let Some(v) = message_count {
                    w.etape()?;
                    w.db.set_node_property(noeud, "message_count", Value::from(v));
                }
                if let Some(v) = total_cost_usd.filter(|v| v.is_finite()) {
                    w.etape()?;
                    w.db.set_node_property(noeud, "total_cost_usd", Value::from(v));
                }
                if let Some(v) = conversation_id {
                    w.etape()?;
                    w.db.set_node_property(noeud, "conversation_id", Value::from(v));
                }
                if let Some(v) = preview {
                    w.etape()?;
                    w.db.set_node_property(noeud, "preview", Value::from(v));
                }
                let entree = Self::relire_session(w, &ancienne)?;
                w.rendre_durable()?;
                let session = entree.session.clone();
                w.index_mut()?.chat.publier_session(entree);
                Ok(Some(session))
            },
        )
        .await
    }

    async fn update_chat_session_permission_mode(&self, id: Uuid, mode: &str) -> Result<()> {
        let m = mode.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = Self::noeud_session(w, id)? else {
                    return Ok(());
                };
                let maintenant = Utc::now();
                poser(
                    w,
                    noeud,
                    vec![
                        ("permission_mode", Value::from(m.clone())),
                        ("updated_at", Value::from(maintenant.to_rfc3339())),
                    ],
                )?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(e) = etat.chat.sessions.get_mut(&id) {
                    e.session.permission_mode = if m.is_empty() { None } else { Some(m) };
                    e.session.updated_at = maintenant;
                }
                Ok(())
            },
        )
        .await
    }

    async fn bind_chat_session_project(&self, id: Uuid, project_slug: &str) -> Result<bool> {
        let slug = project_slug.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(ancienne) = Self::entree_session_de(w, id)? else {
                    return Ok(false);
                };
                if ancienne
                    .session
                    .project_slug
                    .as_deref()
                    .is_some_and(|s| !s.trim().is_empty())
                {
                    return Ok(false);
                }
                w.etape()?;
                w.db.set_node_property(ancienne.noeud, "project_slug", Value::from(slug.clone()));
                w.db.set_node_property(
                    ancienne.noeud,
                    "updated_at",
                    Value::from(Utc::now().to_rfc3339()),
                );
                let entree = Self::relire_session(w, &ancienne)?;
                w.rendre_durable()?;
                w.index_mut()?.chat.publier_session(entree);
                Ok(true)
            },
        )
        .await
    }

    async fn add_chat_session_children_cost(&self, id: Uuid, delta_usd: f64) -> Result<f64> {
        let delta = if delta_usd.is_finite() && delta_usd > 0.0 {
            delta_usd
        } else {
            0.0
        };
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = Self::noeud_session(w, id)? else {
                    return Ok(0.0);
                };
                let ancien = w
                    .index()?
                    .chat
                    .sessions
                    .get(&id)
                    .map(|e| e.children_cost_usd)
                    .unwrap_or(0.0);
                let total = ancien + delta;
                w.etape()?;
                w.db.set_node_property(noeud, "children_cost_usd", Value::from(total));
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(e) = etat.chat.sessions.get_mut(&id) {
                    e.children_cost_usd = total;
                }
                Ok(total)
            },
        )
        .await
    }

    async fn get_chat_session_children_cost(&self, id: Uuid) -> Result<f64> {
        self.lire(|_, etat| {
            Ok(etat
                .chat
                .sessions
                .get(&id)
                .map(|e| e.children_cost_usd)
                .unwrap_or(0.0))
        })
    }

    async fn update_chat_session_file_scope_lifted(&self, id: Uuid, lifted: bool) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = Self::noeud_session(w, id)? else {
                    return Ok(());
                };
                let maintenant = Utc::now();
                poser(
                    w,
                    noeud,
                    vec![
                        ("file_scope_lifted", Value::from(lifted)),
                        ("updated_at", Value::from(maintenant.to_rfc3339())),
                    ],
                )?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(e) = etat.chat.sessions.get_mut(&id) {
                    e.session.file_scope_lifted = lifted;
                    e.session.updated_at = maintenant;
                }
                Ok(())
            },
        )
        .await
    }

    async fn update_chat_session_model(&self, id: Uuid, model: &str) -> Result<()> {
        let m = model.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = Self::noeud_session(w, id)? else {
                    return Ok(());
                };
                let maintenant = Utc::now();
                poser(
                    w,
                    noeud,
                    vec![
                        ("model", Value::from(m.clone())),
                        ("updated_at", Value::from(maintenant.to_rfc3339())),
                    ],
                )?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(e) = etat.chat.sessions.get_mut(&id) {
                    e.session.model = m;
                    e.session.updated_at = maintenant;
                }
                Ok(())
            },
        )
        .await
    }

    async fn update_chat_session_routing(
        &self,
        id: Uuid,
        routing_mode: Option<&str>,
        routing_pool: Option<&str>,
        routed_by: Option<&str>,
    ) -> Result<bool> {
        let (rm, rp, rb) = (
            routing_mode.unwrap_or_default().to_string(),
            routing_pool.unwrap_or_default().to_string(),
            routed_by.unwrap_or_default().to_string(),
        );
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = Self::noeud_session(w, id)? else {
                    return Ok(false);
                };
                let maintenant = Utc::now();
                poser(
                    w,
                    noeud,
                    vec![
                        ("routing_mode", Value::from(rm.clone())),
                        ("routing_pool", Value::from(rp.clone())),
                        ("routed_by", Value::from(rb.clone())),
                        ("updated_at", Value::from(maintenant.to_rfc3339())),
                    ],
                )?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(e) = etat.chat.sessions.get_mut(&id) {
                    e.session.routing_mode = if rm.is_empty() { None } else { Some(rm) };
                    e.session.routing_pool = if rp.is_empty() { None } else { Some(rp) };
                    e.session.routed_by = if rb.is_empty() { None } else { Some(rb) };
                    e.session.updated_at = maintenant;
                }
                Ok(true)
            },
        )
        .await
    }

    async fn update_chat_session_provider(
        &self,
        id: Uuid,
        provider_id: &str,
        model: &str,
        routed_by: &str,
        engine: &str,
    ) -> Result<bool> {
        let (pid, m, rb, eng) = (
            provider_id.to_string(),
            model.to_string(),
            routed_by.to_string(),
            engine.to_string(),
        );
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = Self::noeud_session(w, id)? else {
                    return Ok(false);
                };
                let maintenant = Utc::now();
                // The resume handles of the old provider go FIRST: an interruption
                // never leaves them under the new provider (see module doc).
                poser(
                    w,
                    noeud,
                    vec![
                        ("resume_token", Value::from("")),
                        ("capabilities", Value::from("")),
                        ("cli_session_id", Value::from("")),
                        ("provider_id", Value::from(pid.clone())),
                        ("model", Value::from(m.clone())),
                        ("routed_by", Value::from(rb.clone())),
                        ("engine", Value::from(eng.clone())),
                        ("updated_at", Value::from(maintenant.to_rfc3339())),
                    ],
                )?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(e) = etat.chat.sessions.get_mut(&id) {
                    // `""` is written, `None` is what a re-read serves: publish that.
                    e.session.provider_id = opt_non_vide(Some(pid));
                    e.session.model = m;
                    e.session.routed_by = opt_non_vide(Some(rb));
                    e.session.resume_token = None;
                    e.session.capabilities = None;
                    e.session.cli_session_id = None;
                    e.session.engine = opt_non_vide(Some(eng));
                    e.session.updated_at = maintenant;
                }
                Ok(true)
            },
        )
        .await
    }

    async fn restore_chat_session_provider(
        &self,
        node: &ChatSessionNode,
        provider_id: &str,
    ) -> Result<bool> {
        let (pid, n) = (provider_id.to_string(), node.clone());
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = Self::noeud_session(w, n.id)? else {
                    return Ok(false);
                };
                let maintenant = Utc::now();
                // The provider first, its resume handles LAST: an interruption never
                // leaves a handle under a provider it does not belong to.
                poser(
                    w,
                    noeud,
                    vec![
                        ("provider_id", Value::from(pid.clone())),
                        ("model", Value::from(n.model.clone())),
                        (
                            "routed_by",
                            Value::from(n.routed_by.clone().unwrap_or_default()),
                        ),
                        ("engine", Value::from(n.engine.clone().unwrap_or_default())),
                        ("updated_at", Value::from(maintenant.to_rfc3339())),
                        (
                            "capabilities",
                            Value::from(n.capabilities.clone().unwrap_or_default()),
                        ),
                        (
                            "cli_session_id",
                            Value::from(n.cli_session_id.clone().unwrap_or_default()),
                        ),
                        (
                            "resume_token",
                            Value::from(n.resume_token.clone().unwrap_or_default()),
                        ),
                    ],
                )?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(e) = etat.chat.sessions.get_mut(&n.id) {
                    e.session.provider_id = opt_non_vide(Some(pid));
                    e.session.model = n.model.clone();
                    e.session.routed_by = opt_non_vide(n.routed_by.clone());
                    e.session.resume_token = opt_non_vide(n.resume_token.clone());
                    e.session.capabilities = opt_non_vide(n.capabilities.clone());
                    e.session.cli_session_id = opt_non_vide(n.cli_session_id.clone());
                    e.session.engine = opt_non_vide(n.engine.clone());
                    e.session.updated_at = maintenant;
                }
                Ok(true)
            },
        )
        .await
    }

    async fn get_llm_setting(&self, scope: &str, key: &str) -> Result<Option<String>> {
        let (s, k) = (scope.to_string(), key.to_string());
        self.lire(move |_, etat| Ok(etat.chat.reglages.get(&(s, k)).map(|e| e.valeur.clone())))
    }

    async fn put_llm_setting(&self, scope: &str, key: &str, value: &str) -> Result<()> {
        let (s, k, v) = (scope.to_string(), key.to_string(), value.to_string());
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let existant = w
                    .index()?
                    .chat
                    .reglages
                    .get(&(s.clone(), k.clone()))
                    .map(|e| e.noeud);
                w.flush_complet = v.len() > SEUIL_COLONNE_BLOB;
                let valeur = vec![
                    ("value", Value::from(v.clone())),
                    ("updated_at", Value::from(Utc::now().to_rfc3339())),
                ];
                // A creation is a draft until its value is written: an interruption
                // never leaves a setting that reads back `""`.
                let noeud = match existant {
                    Some(n) => {
                        poser(w, n, valeur)?;
                        n
                    }
                    None => {
                        let mut proprietes = vec![
                            ("scope", Value::from(s.clone())),
                            ("key", Value::from(k.clone())),
                        ];
                        proprietes.extend(valeur);
                        creer_noeud(w, REGLAGE, proprietes)?
                    }
                };
                w.rendre_durable()?;
                w.index_mut()?
                    .chat
                    .reglages
                    .insert((s, k), EntreeReglage { noeud, valeur: v });
                Ok(())
            },
        )
        .await
    }

    async fn delete_llm_setting(&self, scope: &str, key: &str) -> Result<bool> {
        let (s, k) = (scope.to_string(), key.to_string());
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = w
                    .index()?
                    .chat
                    .reglages
                    .get(&(s.clone(), k.clone()))
                    .map(|e| e.noeud)
                else {
                    return Ok(false);
                };
                detacher(w, noeud)?;
                w.rendre_durable()?;
                w.index_mut()?.chat.reglages.remove(&(s, k));
                Ok(true)
            },
        )
        .await
    }

    async fn list_llm_settings(&self, scope: &str, prefix: &str) -> Result<Vec<(String, String)>> {
        let (s, p) = (scope.to_string(), prefix.to_string());
        self.lire(move |_, etat| {
            let mut out: Vec<(String, String)> = etat
                .chat
                .reglages
                .iter()
                .filter(|((sc, k), _)| *sc == s && k.starts_with(&p))
                .map(|((_, k), e)| (k.clone(), e.valeur.clone()))
                .collect();
            out.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(out)
        })
    }

    async fn update_chat_session_harness(
        &self,
        id: Uuid,
        capabilities: Option<&str>,
        resume_token: Option<&str>,
    ) -> Result<()> {
        let (cap, tok) = (
            capabilities.map(str::to_string),
            resume_token.map(str::to_string),
        );
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = Self::noeud_session(w, id)? else {
                    return Ok(());
                };
                let maintenant = Utc::now();
                let mut proprietes = Vec::new();
                if let Some(c) = &cap {
                    proprietes.push(("capabilities", Value::from(c.clone())));
                }
                if let Some(t) = &tok {
                    proprietes.push(("resume_token", Value::from(t.clone())));
                }
                proprietes.push(("updated_at", Value::from(maintenant.to_rfc3339())));
                poser(w, noeud, proprietes)?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(e) = etat.chat.sessions.get_mut(&id) {
                    if let Some(c) = cap {
                        e.session.capabilities = if c.is_empty() { None } else { Some(c) };
                    }
                    if let Some(t) = tok {
                        e.session.resume_token = if t.is_empty() { None } else { Some(t) };
                    }
                    e.session.updated_at = maintenant;
                }
                Ok(())
            },
        )
        .await
    }

    async fn set_chat_session_fingerprint(&self, id: Uuid, fingerprint: &str) -> Result<()> {
        let fp = fingerprint.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = Self::noeud_session(w, id)? else {
                    return Ok(());
                };
                w.flush_complet = fp.len() > SEUIL_COLONNE_BLOB;
                let maintenant = Utc::now();
                poser(
                    w,
                    noeud,
                    vec![
                        ("resume_fingerprint", Value::from(fp.clone())),
                        ("updated_at", Value::from(maintenant.to_rfc3339())),
                    ],
                )?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(e) = etat.chat.sessions.get_mut(&id) {
                    e.resume_fingerprint = if fp.is_empty() { None } else { Some(fp) };
                    // Written above, so published: `list_chat_sessions` sorts on it.
                    e.session.updated_at = maintenant;
                }
                Ok(())
            },
        )
        .await
    }

    async fn get_chat_session_fingerprint(&self, id: Uuid) -> Result<Option<String>> {
        self.lire(|_, etat| {
            Ok(etat
                .chat
                .sessions
                .get(&id)
                .and_then(|e| e.resume_fingerprint.clone()))
        })
    }

    async fn set_chat_session_shell_moved(&self, id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = Self::noeud_session(w, id)? else {
                    return Ok(());
                };
                w.etape()?;
                w.db.set_node_property(noeud, "shell_moved", Value::from(true));
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(e) = etat.chat.sessions.get_mut(&id) {
                    e.shell_moved = Some(true);
                }
                Ok(())
            },
        )
        .await
    }

    async fn get_chat_session_shell_moved(&self, id: Uuid) -> Result<Option<bool>> {
        self.lire(|_, etat| Ok(etat.chat.sessions.get(&id).and_then(|e| e.shell_moved)))
    }

    async fn backfill_chat_session_previews(&self) -> Result<usize> {
        use crate::chat::session_record::title_and_preview;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let mut a_remplir: Vec<(Uuid, String, String)> = Vec::new();
                {
                    let etat = w.index()?;
                    for (sid, e) in &etat.chat.sessions {
                        if e.session.preview.as_deref().is_some_and(|p| !p.is_empty()) {
                            continue;
                        }
                        let Some(evts) = etat.chat.evenements.get(sid) else {
                            continue;
                        };
                        let Some(premier) = evts
                            .iter()
                            .find(|ev| ev.evenement.event_type == "user_message")
                        else {
                            continue;
                        };
                        if let Ok(data) =
                            serde_json::from_str::<serde_json::Value>(&premier.evenement.data)
                        {
                            if let Some((title, preview)) = data
                                .get("content")
                                .and_then(|v| v.as_str())
                                .and_then(title_and_preview)
                            {
                                a_remplir.push((*sid, title, preview));
                            }
                        }
                    }
                }
                let count = a_remplir.len();
                let mut publiees = Vec::new();
                for (sid, title, preview) in &a_remplir {
                    let Some(ancienne) = Self::entree_session_de(w, *sid)? else {
                        continue;
                    };
                    w.flush_complet = preview.len() > SEUIL_COLONNE_BLOB;
                    w.etape()?;
                    w.db.set_node_property(ancienne.noeud, "title", Value::from(title.clone()));
                    w.db.set_node_property(ancienne.noeud, "preview", Value::from(preview.clone()));
                    w.db.set_node_property(
                        ancienne.noeud,
                        "updated_at",
                        Value::from(Utc::now().to_rfc3339()),
                    );
                    let entree = Self::relire_session(w, &ancienne)?;
                    publiees.push(entree);
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for e in publiees {
                    etat.chat.publier_session(e);
                }
                Ok(count)
            },
        )
        .await
    }

    async fn delete_chat_session(&self, id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(entree) = Self::entree_session_de(w, id)? else {
                    return Ok(false);
                };
                let noeud = entree.noeud;
                // Anchors + their journal, the episode, the events, then the session.
                let ancres: Vec<(Uuid, NodeId, (String, String))> = {
                    let etat = w.index()?;
                    etat.chat
                        .ancres_par_session
                        .get(&id)
                        .into_iter()
                        .flatten()
                        .filter_map(|aid| {
                            etat.chat.ancres.get(aid).map(|e| {
                                (
                                    *aid,
                                    e.noeud,
                                    (
                                        e.ancre.target_type.as_str().to_string(),
                                        e.ancre.target_id.clone(),
                                    ),
                                )
                            })
                        })
                        .collect()
                };
                for (_, a, _) in &ancres {
                    detacher(w, *a)?;
                }
                let evt_ancres: Vec<NodeId> = {
                    let etat = w.index()?;
                    etat.chat
                        .evenements_ancre
                        .get(&id)
                        .into_iter()
                        .flatten()
                        .map(|e| e.noeud)
                        .collect()
                };
                for a in evt_ancres {
                    detacher(w, a)?;
                }
                let ep = w.index()?.chat.episodes.get(&id).map(|e| e.noeud);
                if let Some(ep) = ep {
                    detacher(w, ep)?;
                }
                let evenements: Vec<NodeId> = {
                    let etat = w.index()?;
                    etat.chat
                        .evenements
                        .get(&id)
                        .into_iter()
                        .flatten()
                        .map(|e| e.noeud)
                        .collect()
                };
                for ev in evenements {
                    detacher(w, ev)?;
                }
                // The children lose their SPAWNED_BY edge with it (DETACH DELETE).
                let enfants: Vec<Uuid> = w
                    .index()?
                    .chat
                    .enfants_de
                    .get(&id)
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect();
                detacher(w, noeud)?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                let idx = &mut etat.chat;
                for eid in enfants {
                    if let Some(mut e) = idx.desinscrire_session(eid) {
                        e.parent = None;
                        idx.inscrire_session(e);
                    }
                }
                idx.desinscrire_session(id);
                idx.ancres_par_session.remove(&id);
                // Only THIS session's anchors leave the target pages, by their key.
                for (aid, _, cible) in ancres {
                    idx.ancres.remove(&aid);
                    if let Some(s) = idx.ancres_par_cible.get_mut(&cible) {
                        s.remove(&aid);
                        if s.is_empty() {
                            idx.ancres_par_cible.remove(&cible);
                        }
                    }
                }
                idx.evenements_ancre.remove(&id);
                idx.evenements.remove(&id);
                idx.episodes.remove(&id);
                Ok(true)
            },
        )
        .await
    }

    // ------------------------------------------------------------------
    // Anchors
    // ------------------------------------------------------------------

    async fn apply_anchor_op(
        &self,
        session_id: Uuid,
        op: crate::chat::anchor::AnchorOp,
    ) -> Result<crate::chat::anchor::AnchorChange> {
        Ok(self.apply_anchor_op_if(session_id, op, None).await?.change)
    }

    async fn apply_anchor_op_if(
        &self,
        session_id: Uuid,
        op: crate::chat::anchor::AnchorOp,
        if_match: Option<u64>,
    ) -> Result<AnchorSetChange> {
        use crate::chat::anchor::apply_op;
        self.ecrire(
            move || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud_session, current) = {
                    let etat = w.index()?;
                    let Some(e) = etat.chat.sessions.get(&session_id) else {
                        return Err(anyhow::Error::new(AnchorError::SessionNotFound(session_id)));
                    };
                    (e.noeud, e.anchor_seq)
                };
                if let Some(expected) = if_match.filter(|v| *v != current) {
                    return Err(anyhow::Error::new(AnchorError::SetVersionConflict {
                        expected,
                        actual: current,
                    }));
                }
                let seq = current + 1;
                let existing: Vec<Anchor> = {
                    let etat = w.index()?;
                    let mut v: Vec<Anchor> = etat
                        .chat
                        .ancres_par_session
                        .get(&session_id)
                        .into_iter()
                        .flatten()
                        .filter_map(|aid| etat.chat.ancres.get(aid).map(|e| e.ancre.clone()))
                        .collect();
                    v.sort_by(|a, b| (a.created_at, a.id).cmp(&(b.created_at, b.id)));
                    v
                };
                let mut change = apply_op(session_id, &existing, op, Utc::now())?;
                if let Some(ev) = change.event.as_mut() {
                    ev.seq = seq;
                }
                // `anchor_seq` FIRST: an interruption leaves a gap in the numbering,
                // never two journal entries with the same `seq` (see module doc).
                if change.event.is_some() {
                    w.etape()?;
                    w.db.set_node_property(noeud_session, "anchor_seq", Value::from(seq as i64));
                }
                let mut noeud_ancre: Option<NodeId> = None;
                if change.deleted {
                    let cible = w
                        .index()?
                        .chat
                        .ancres
                        .get(&change.anchor.id)
                        .map(|e| e.noeud);
                    if let Some(n) = cible {
                        detacher(w, n)?;
                    }
                } else if change.event.is_some() {
                    let a = &change.anchor;
                    let existant = w.index()?.chat.ancres.get(&a.id).map(|e| e.noeud);
                    let noeud = match existant {
                        Some(n) => {
                            poser(w, n, proprietes_ancre(a))?;
                            n
                        }
                        None => creer_noeud(w, ANCRE, proprietes_ancre(a))?,
                    };
                    noeud_ancre = Some(noeud);
                }
                let mut entree_journal: Option<EntreeEvenementAncre> = None;
                if let Some(ev) = &change.event {
                    let noeud = creer_noeud(w, EVENEMENT_ANCRE, proprietes_evenement_ancre(ev))?;
                    entree_journal = Some(EntreeEvenementAncre {
                        noeud,
                        evenement: ev.clone(),
                    });
                }
                w.rendre_durable()?;
                {
                    let mut etat = w.index_mut()?;
                    let idx = &mut etat.chat;
                    if change.event.is_some() {
                        if let Some(e) = idx.sessions.get_mut(&session_id) {
                            e.anchor_seq = seq;
                        }
                    }
                    // The journal entry is published with the change it records:
                    // `seq` grows under the writer lock, so appending keeps the order.
                    if let Some(entree) = entree_journal {
                        idx.evenements_ancre
                            .entry(session_id)
                            .or_default()
                            .push(entree);
                    }
                    if change.deleted {
                        let a = &change.anchor;
                        idx.ancres.remove(&a.id);
                        if let Some(s) = idx.ancres_par_session.get_mut(&session_id) {
                            s.remove(&a.id);
                            if s.is_empty() {
                                idx.ancres_par_session.remove(&session_id);
                            }
                        }
                        let cle = (a.target_type.as_str().to_string(), a.target_id.clone());
                        if let Some(s) = idx.ancres_par_cible.get_mut(&cle) {
                            s.remove(&a.id);
                            if s.is_empty() {
                                idx.ancres_par_cible.remove(&cle);
                            }
                        }
                    } else if let Some(noeud) = noeud_ancre {
                        let ancre = change.anchor.clone();
                        idx.ancres.insert(
                            ancre.id,
                            EntreeAncre {
                                noeud,
                                ancre: ancre.clone(),
                            },
                        );
                        idx.ancres_par_session
                            .entry(session_id)
                            .or_default()
                            .insert(ancre.id);
                        idx.ancres_par_cible
                            .entry((
                                ancre.target_type.as_str().to_string(),
                                ancre.target_id.clone(),
                            ))
                            .or_default()
                            .insert(ancre.id);
                    }
                }
                let version = if change.event.is_some() { seq } else { current };
                Ok(AnchorSetChange { change, version })
            },
        )
        .await
    }

    async fn get_anchor_set(&self, session_id: Uuid) -> Result<Option<AnchorSet>> {
        self.lire(move |_, etat| {
            let Some(e) = etat.chat.sessions.get(&session_id) else {
                return Ok(None);
            };
            let mut ancres: Vec<Anchor> = etat
                .chat
                .ancres_par_session
                .get(&session_id)
                .into_iter()
                .flatten()
                .filter_map(|aid| etat.chat.ancres.get(aid).map(|e| e.ancre.clone()))
                .collect();
            ancres.sort_by(|a, b| (a.created_at, a.id).cmp(&(b.created_at, b.id)));
            Ok(Some(AnchorSet {
                version: e.anchor_seq,
                anchors: ancres,
            }))
        })
    }

    async fn add_anchor(
        &self,
        session_id: Uuid,
        new: crate::chat::anchor::NewAnchor,
    ) -> Result<crate::chat::anchor::Anchor> {
        Ok(self
            .apply_anchor_op(session_id, crate::chat::anchor::AnchorOp::Add(new))
            .await?
            .anchor)
    }

    async fn promote_anchor_role(
        &self,
        session_id: Uuid,
        anchor_id: Uuid,
        role: AnchorRole,
        expected_version: u64,
        by: AnchorActor,
        actor: &str,
    ) -> Result<Anchor> {
        Ok(self
            .apply_anchor_op(
                session_id,
                AnchorOp::Promote {
                    anchor_id,
                    role,
                    expected_version,
                    by,
                    actor: actor.to_string(),
                },
            )
            .await?
            .anchor)
    }

    async fn demote_anchor_role(
        &self,
        session_id: Uuid,
        anchor_id: Uuid,
        role: AnchorRole,
        expected_version: u64,
        by: AnchorActor,
        actor: &str,
    ) -> Result<Anchor> {
        Ok(self
            .apply_anchor_op(
                session_id,
                AnchorOp::Demote {
                    anchor_id,
                    role,
                    expected_version,
                    by,
                    actor: actor.to_string(),
                },
            )
            .await?
            .anchor)
    }

    async fn remove_anchor(
        &self,
        session_id: Uuid,
        anchor_id: Uuid,
        expected_version: u64,
        by: AnchorActor,
        actor: &str,
    ) -> Result<()> {
        self.apply_anchor_op(
            session_id,
            AnchorOp::Remove {
                anchor_id,
                expected_version,
                by,
                actor: actor.to_string(),
            },
        )
        .await?;
        Ok(())
    }

    async fn set_anchor_state(
        &self,
        session_id: Uuid,
        anchor_id: Uuid,
        state: AnchorState,
        expected_version: Option<u64>,
        by: AnchorActor,
        actor: &str,
    ) -> Result<Anchor> {
        Ok(self
            .apply_anchor_op(
                session_id,
                AnchorOp::SetState {
                    anchor_id,
                    state,
                    expected_version,
                    by,
                    actor: actor.to_string(),
                },
            )
            .await?
            .anchor)
    }

    async fn list_session_anchors(&self, session_id: Uuid) -> Result<Vec<Anchor>> {
        self.lire(move |_, etat| {
            let mut ancres: Vec<Anchor> = etat
                .chat
                .ancres_par_session
                .get(&session_id)
                .into_iter()
                .flatten()
                .filter_map(|aid| etat.chat.ancres.get(aid).map(|e| e.ancre.clone()))
                .collect();
            ancres.sort_by(|a, b| (a.created_at, a.id).cmp(&(b.created_at, b.id)));
            Ok(ancres)
        })
    }

    async fn list_sessions_for_target(
        &self,
        target_type: AnchorTargetType,
        target_id: &str,
        state: Option<AnchorState>,
        limit: usize,
        cursor: Option<&str>,
    ) -> Result<AnchorPage> {
        let limit = limit.clamp(1, 200);
        let tid = target_id.to_string();
        let cur = cursor.map(str::to_string);
        self.lire(move |_, etat| {
            let cle = (target_type.as_str().to_string(), tid);
            let mut items: Vec<Anchor> = etat
                .chat
                .ancres_par_cible
                .get(&cle)
                .into_iter()
                .flatten()
                .filter_map(|aid| etat.chat.ancres.get(aid).map(|e| e.ancre.clone()))
                .filter(|a| state.is_none_or(|s| a.state == s))
                .filter(|a| cur.as_deref().is_none_or(|c| a.id.to_string().as_str() > c))
                .collect();
            items.sort_by(|a, b| a.id.cmp(&b.id));
            let next_cursor = if items.len() > limit {
                items.truncate(limit);
                items.last().map(|a| a.id.to_string())
            } else {
                None
            };
            Ok(AnchorPage { items, next_cursor })
        })
    }

    async fn list_anchor_events(&self, session_id: Uuid) -> Result<Vec<AnchorEvent>> {
        self.lire(move |_, etat| {
            Ok(etat
                .chat
                .evenements_ancre
                .get(&session_id)
                .into_iter()
                .flatten()
                .map(|e| e.evenement.clone())
                .collect())
        })
    }

    async fn create_execution_place_event(&self, event: &ExecutionPlaceEvent) -> Result<()> {
        let e = event.clone();
        self.ecrire(
            move || e,
            move |mut e, w: &mut Ecriture<'_>| {
                creer_noeud(w, LIEU, proprietes_lieu(&e))?;
                w.rendre_durable()?;
                // What is published is what a re-read serves: `at` to the microsecond.
                e.at = parse_ts(&format_lieu_at(e.at))?;
                w.index_mut()?
                    .chat
                    .evenements_lieu
                    .entry(e.task_id)
                    .or_default()
                    .push(EntreeLieu { evenement: e });
                Ok(())
            },
        )
        .await
    }

    async fn list_execution_place_events(&self, task_id: Uuid) -> Result<Vec<ExecutionPlaceEvent>> {
        self.lire(move |_, etat| {
            let mut v: Vec<ExecutionPlaceEvent> = etat
                .chat
                .evenements_lieu
                .get(&task_id)
                .into_iter()
                .flatten()
                .map(|e| e.evenement.clone())
                .collect();
            v.sort_by(|a, b| a.at.cmp(&b.at).then(a.id.cmp(&b.id)));
            Ok(v)
        })
    }

    async fn mark_anchors_dangling(
        &self,
        target_type: AnchorTargetType,
        target_id: &str,
    ) -> Result<usize> {
        use crate::chat::anchor::DANGLING_ACTOR;
        let tt = target_type;
        let tid = target_id.to_string();
        let todo: Vec<(Uuid, Uuid)> = self.lire(move |_, etat| {
            let cle = (tt.as_str().to_string(), tid);
            Ok(etat
                .chat
                .ancres_par_cible
                .get(&cle)
                .into_iter()
                .flatten()
                .filter_map(|aid| {
                    etat.chat
                        .ancres
                        .get(aid)
                        .map(|e| (e.ancre.session_id, e.ancre.id))
                })
                .filter(|(_, aid)| {
                    etat.chat
                        .ancres
                        .get(aid)
                        .is_some_and(|e| e.ancre.state != AnchorState::Dangling)
                })
                .collect())
        })?;
        let mut changed = 0usize;
        for (session_id, anchor_id) in todo {
            let op = AnchorOp::SetState {
                anchor_id,
                state: AnchorState::Dangling,
                expected_version: None,
                by: AnchorActor::System,
                actor: DANGLING_ACTOR.to_string(),
            };
            match self.apply_anchor_op(session_id, op).await {
                Ok(c) if c.event.is_some() => changed += 1,
                Ok(_) => {}
                Err(e) => match e.downcast_ref::<AnchorError>() {
                    Some(AnchorError::NotFound(_) | AnchorError::SessionNotFound(_)) => {}
                    _ => return Err(e),
                },
            }
        }
        Ok(changed)
    }

    // ------------------------------------------------------------------
    // Events
    // ------------------------------------------------------------------

    async fn store_chat_events(
        &self,
        session_id: Uuid,
        events: Vec<ChatEventRecord>,
    ) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        self.ecrire(
            move || events,
            move |events, w: &mut Ecriture<'_>| {
                let existe = w.index()?.chat.sessions.contains_key(&session_id);
                if !existe {
                    return Ok(());
                }
                let mut nouvelles = Vec::with_capacity(events.len());
                for ev in &events {
                    let noeud = creer_noeud(w, EVENEMENT, proprietes_evenement(session_id, ev))?;
                    nouvelles.push(EntreeEvenement {
                        noeud,
                        evenement: ev.clone(),
                    });
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                let vec = etat.chat.evenements.entry(session_id).or_default();
                vec.extend(nouvelles);
                vec.sort_by(|a, b| a.evenement.seq.cmp(&b.evenement.seq));
                Ok(())
            },
        )
        .await
    }

    async fn get_chat_events(
        &self,
        session_id: Uuid,
        after_seq: i64,
        limit: i64,
    ) -> Result<Vec<ChatEventRecord>> {
        self.lire(move |_, etat| {
            let mut v: Vec<ChatEventRecord> = etat
                .chat
                .evenements
                .get(&session_id)
                .into_iter()
                .flatten()
                .filter(|e| e.evenement.seq > after_seq)
                .map(|e| e.evenement.clone())
                .collect();
            v.sort_by(|a, b| a.seq.cmp(&b.seq));
            // `LIMIT 0` is no row, as on Neo4j (a negative limit too: see module doc).
            v.truncate(usize::try_from(limit).unwrap_or(0));
            Ok(v)
        })
    }

    async fn get_chat_events_paginated(
        &self,
        session_id: Uuid,
        offset: i64,
        limit: i64,
    ) -> Result<Vec<ChatEventRecord>> {
        self.lire(move |_, etat| {
            let mut v: Vec<ChatEventRecord> = etat
                .chat
                .evenements
                .get(&session_id)
                .into_iter()
                .flatten()
                .map(|e| e.evenement.clone())
                .collect();
            v.sort_by(|a, b| a.seq.cmp(&b.seq));
            let o = offset.max(0) as usize;
            let l = limit.max(0) as usize;
            Ok(v.into_iter().skip(o).take(l).collect())
        })
    }

    async fn count_chat_events(&self, session_id: Uuid) -> Result<i64> {
        self.lire(|_, etat| {
            Ok(etat
                .chat
                .evenements
                .get(&session_id)
                .map(|v| v.len() as i64)
                .unwrap_or(0))
        })
    }

    async fn get_attention_events(&self, session_ids: &[Uuid]) -> Result<Vec<ChatEventRecord>> {
        let ids: HashSet<Uuid> = session_ids.iter().copied().collect();
        const TYPES: [&str; 5] = [
            "permission_request",
            "permission_decision",
            "ask_user_question",
            "user_message",
            "session_error",
        ];
        self.lire(move |_, etat| {
            let mut v: Vec<ChatEventRecord> = Vec::new();
            for sid in &ids {
                if let Some(evts) = etat.chat.evenements.get(sid) {
                    for e in evts {
                        if TYPES.contains(&e.evenement.event_type.as_str()) {
                            let mut ev = e.evenement.clone();
                            if ev.event_type == "user_message" {
                                ev.data = String::new();
                            }
                            v.push(ev);
                        }
                    }
                }
            }
            v.sort_by(|a, b| (a.session_id, a.seq).cmp(&(b.session_id, b.seq)));
            Ok(v)
        })
    }

    async fn get_latest_chat_event_seq(&self, session_id: Uuid) -> Result<i64> {
        self.lire(|_, etat| {
            Ok(etat
                .chat
                .evenements
                .get(&session_id)
                .into_iter()
                .flatten()
                .map(|e| e.evenement.seq)
                .max()
                .unwrap_or(0))
        })
    }

    async fn delete_chat_events(&self, session_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeuds: Vec<NodeId> = w
                    .index()?
                    .chat
                    .evenements
                    .get(&session_id)
                    .into_iter()
                    .flatten()
                    .map(|e| e.noeud)
                    .collect();
                for n in noeuds {
                    detacher(w, n)?;
                }
                w.rendre_durable()?;
                w.index_mut()?.chat.evenements.remove(&session_id);
                Ok(())
            },
        )
        .await
    }

    // ------------------------------------------------------------------
    // Episodes
    // ------------------------------------------------------------------

    async fn save_session_episode(&self, episode: &Episode) -> Result<bool> {
        let Some(session) = episode.session.as_ref() else {
            return Ok(false);
        };
        let mut sans_vecteur = episode.clone();
        let _ = sans_vecteur.embedding.take();
        let payload = serde_json::to_string(&sans_vecteur)?;
        let sid = session.session_id;
        let pid = episode.project_id;
        let at = episode.collected_at;
        let eid = episode.id;
        self.ecrire(
            move || (),
            move |(), w: &mut Ecriture<'_>| {
                let existe = w.index()?.chat.sessions.contains_key(&sid);
                if !existe {
                    return Ok(false);
                }
                let existant = w.index()?.chat.episodes.get(&sid).map(|e| e.noeud);
                w.flush_complet = payload.len() > SEUIL_COLONNE_BLOB;
                let proprietes = vec![
                    ("id", Value::from(eid.to_string())),
                    ("project_id", Value::from(pid.to_string())),
                    ("collected_at", Value::from(at.to_rfc3339())),
                    ("session_id", Value::from(sid.to_string())),
                    ("payload", Value::from(payload)),
                ];
                let noeud = match existant {
                    Some(n) => {
                        poser(w, n, proprietes)?;
                        n
                    }
                    None => creer_noeud(w, EPISODE, proprietes)?,
                };
                w.rendre_durable()?;
                // What was written, so what a re-read serves: no vector.
                w.index_mut()?.chat.episodes.insert(
                    sid,
                    EntreeEpisode {
                        noeud,
                        projet: pid,
                        episode: sans_vecteur,
                    },
                );
                Ok(true)
            },
        )
        .await
    }

    async fn list_session_episodes(&self, project_id: Uuid, limit: usize) -> Result<Vec<Episode>> {
        self.lire(move |_, etat| {
            let mut v: Vec<Episode> = etat
                .chat
                .episodes
                .values()
                .filter(|e| e.projet == project_id)
                .map(|e| e.episode.clone())
                .collect();
            v.sort_by(|a, b| b.collected_at.cmp(&a.collected_at).then(b.id.cmp(&a.id)));
            v.truncate(limit);
            Ok(v)
        })
    }

    async fn agent_identity_namespace(&self) -> Result<Uuid> {
        if let Some(ns) = self.lire(|_, etat| Ok(etat.chat.racine_agent))? {
            return Ok(ns);
        }
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                if let Some(ns) = w.index()?.chat.racine_agent {
                    return Ok(ns);
                }
                let ns = Uuid::new_v4();
                // A draft until its namespace is written: an interruption never
                // leaves a root that reads as absent and makes the next call create
                // a second one.
                let noeud = creer_noeud(
                    w,
                    RACINE_AGENT,
                    vec![
                        ("key", Value::from("root")),
                        ("namespace", Value::from(ns.to_string())),
                    ],
                )?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.chat.racine_agent = Some(ns);
                etat.chat.noeud_racine_agent = Some(noeud);
                Ok(ns)
            },
        )
        .await
    }

    // ------------------------------------------------------------------
    // Segments
    // ------------------------------------------------------------------

    async fn put_segments(&self, conversation: &str, segments: &[Segment]) -> Result<()> {
        let conversation = conversation.to_string();
        let segments = segments.to_vec();
        self.ecrire(
            move || (),
            move |(), w: &mut Ecriture<'_>| {
                let mut ecrites: Vec<(String, NodeId, Segment)> = Vec::new();
                for segment in &segments {
                    let json = serde_json::to_string(segment)?;
                    let id = segment.id.as_str().to_string();
                    let existant = w
                        .index()?
                        .chat
                        .segments
                        .get(&conversation)
                        .and_then(|m| m.get(&id).map(|e| e.noeud));
                    w.flush_complet |= json.len() > SEUIL_COLONNE_BLOB;
                    let proprietes = vec![
                        ("conversation", Value::from(conversation.clone())),
                        ("id", Value::from(id.clone())),
                        ("dimension", Value::from(segment.dimension.as_str())),
                        ("start_turn", Value::from(i64::from(segment.start_turn))),
                        ("json", Value::from(json)),
                    ];
                    let noeud = match existant {
                        Some(n) => {
                            poser(w, n, proprietes)?;
                            n
                        }
                        None => creer_noeud(w, SEGMENT, proprietes)?,
                    };
                    ecrites.push((id, noeud, segment.clone()));
                }
                // prune what the conversation no longer has
                let ids: HashSet<String> =
                    segments.iter().map(|s| s.id.as_str().to_string()).collect();
                let a_supprimer: Vec<NodeId> = w
                    .index()?
                    .chat
                    .segments
                    .get(&conversation)
                    .into_iter()
                    .flat_map(|m| m.iter())
                    .filter(|(id, _)| !ids.contains(*id))
                    .map(|(_, e)| e.noeud)
                    .collect();
                // The unreadable nodes of the conversation go too: `put` replaces the
                // set, and these were in it (healed, as Neo4j's MERGE would).
                let illisibles: Vec<NodeId> = w
                    .index()?
                    .chat
                    .segments_illisibles
                    .get(&conversation)
                    .cloned()
                    .unwrap_or_default();
                let a_supprimer: Vec<NodeId> = a_supprimer.into_iter().chain(illisibles).collect();
                for n in a_supprimer {
                    detacher(w, n)?;
                }
                w.rendre_durable()?;
                let carte: BTreeMap<String, EntreeSegment> = ecrites
                    .into_iter()
                    .map(|(id, noeud, segment)| (id, EntreeSegment { noeud, segment }))
                    .collect();
                let mut etat = w.index_mut()?;
                etat.chat.segments_illisibles.remove(&conversation);
                etat.chat.segments.insert(conversation, carte);
                Ok(())
            },
        )
        .await
    }

    async fn get_segments(&self, conversation: &str) -> Result<Vec<Segment>> {
        let c = conversation.to_string();
        self.lire(move |_, etat| {
            if let Some(n) = etat.chat.segments_illisibles.get(&c) {
                bail!(
                    "conversation {c} : {} segment(s) illisible(s) en quarantaine ; un historique \
                     tronqué n'est jamais servi",
                    n.len()
                );
            }
            let mut v: Vec<Segment> = etat
                .chat
                .segments
                .get(&c)
                .into_iter()
                .flat_map(|m| m.values())
                .map(|e| e.segment.clone())
                .collect();
            v.sort_by(|a, b| (a.dimension, a.start_turn).cmp(&(b.dimension, b.start_turn)));
            Ok(v)
        })
    }

    async fn put_boundaries(&self, conversation: &str, boundaries: &[Boundary]) -> Result<()> {
        let conversation = conversation.to_string();
        let boundaries = boundaries.to_vec();
        self.ecrire(
            move || (),
            move |(), w: &mut Ecriture<'_>| {
                let mut ecrites: Vec<(String, NodeId, Boundary)> = Vec::new();
                for frontiere in &boundaries {
                    let json = serde_json::to_string(frontiere)?;
                    let cle = format!("{}#{}", frontiere.dimension.as_str(), frontiere.at_turn);
                    let existant = w
                        .index()?
                        .chat
                        .frontieres
                        .get(&conversation)
                        .and_then(|m| m.get(&cle).map(|e| e.noeud));
                    w.flush_complet |= json.len() > SEUIL_COLONNE_BLOB;
                    let proprietes = vec![
                        ("conversation", Value::from(conversation.clone())),
                        ("dimension", Value::from(frontiere.dimension.as_str())),
                        ("at_turn", Value::from(i64::from(frontiere.at_turn))),
                        ("from", Value::from(frontiere.from.as_str().to_string())),
                        ("to", Value::from(frontiere.to.as_str().to_string())),
                        ("json", Value::from(json)),
                    ];
                    let noeud = match existant {
                        Some(n) => {
                            poser(w, n, proprietes)?;
                            n
                        }
                        None => creer_noeud(w, FRONTIERE, proprietes)?,
                    };
                    ecrites.push((cle, noeud, frontiere.clone()));
                }
                let keys: HashSet<String> = boundaries
                    .iter()
                    .map(|b| format!("{}#{}", b.dimension.as_str(), b.at_turn))
                    .collect();
                let a_supprimer: Vec<NodeId> = w
                    .index()?
                    .chat
                    .frontieres
                    .get(&conversation)
                    .into_iter()
                    .flat_map(|m| m.iter())
                    .filter(|(k, _)| !keys.contains(*k))
                    .map(|(_, e)| e.noeud)
                    .collect();
                let illisibles: Vec<NodeId> = w
                    .index()?
                    .chat
                    .frontieres_illisibles
                    .get(&conversation)
                    .cloned()
                    .unwrap_or_default();
                let a_supprimer: Vec<NodeId> = a_supprimer.into_iter().chain(illisibles).collect();
                for n in a_supprimer {
                    detacher(w, n)?;
                }
                w.rendre_durable()?;
                let carte: BTreeMap<String, EntreeFrontiere> = ecrites
                    .into_iter()
                    .map(|(cle, noeud, frontiere)| (cle, EntreeFrontiere { noeud, frontiere }))
                    .collect();
                let mut etat = w.index_mut()?;
                etat.chat.frontieres_illisibles.remove(&conversation);
                etat.chat.frontieres.insert(conversation, carte);
                Ok(())
            },
        )
        .await
    }

    async fn get_boundaries(&self, conversation: &str) -> Result<Vec<Boundary>> {
        let c = conversation.to_string();
        self.lire(move |_, etat| {
            if let Some(n) = etat.chat.frontieres_illisibles.get(&c) {
                bail!(
                    "conversation {c} : {} frontière(s) illisible(s) en quarantaine ; une liste \
                     tronquée n'est jamais servie",
                    n.len()
                );
            }
            let mut v: Vec<Boundary> = etat
                .chat
                .frontieres
                .get(&c)
                .into_iter()
                .flat_map(|m| m.values())
                .map(|e| e.frontiere.clone())
                .collect();
            v.sort_by(|a, b| (a.at_turn, a.dimension).cmp(&(b.at_turn, b.dimension)));
            Ok(v)
        })
    }

    // ------------------------------------------------------------------
    // Refused by name
    // ------------------------------------------------------------------

    async fn link_session_to_run(
        &self,
        _session_id: &str,
        _run_id: Uuid,
        _plan_id: Option<Uuid>,
        _task_id: Option<Uuid>,
    ) -> Result<bool> {
        refus(
            "ChatStore::link_session_to_run",
            "le domaine PlanRun n'est pas porté",
        )
    }

    async fn get_session_link_rows(&self, _session_ids: &[Uuid]) -> Result<Vec<SessionLinkRow>> {
        refus(
            "ChatStore::get_session_link_rows",
            "lit PlanRun, Task et Plan : domaines non portés",
        )
    }

    async fn get_plans_task_graph(&self, _plan_ids: &[Uuid]) -> Result<PlansTaskGraph> {
        refus(
            "ChatStore::get_plans_task_graph",
            "lit Task et Plan : domaines non portés",
        )
    }

    async fn backfill_project_anchors(&self) -> Result<crate::chat::anchor::BackfillReport> {
        refus(
            "ChatStore::backfill_project_anchors",
            "migration non portée (jamais exécutée au démarrage)",
        )
    }

    async fn backfill_spawned_by_edges(&self) -> Result<usize> {
        refus(
            "ChatStore::backfill_spawned_by_edges",
            "migration non portée (jamais exécutée au démarrage)",
        )
    }

    async fn revert_inferred_anchors(&self) -> Result<usize> {
        refus(
            "ChatStore::revert_inferred_anchors",
            "migration non portée (jamais exécutée au démarrage)",
        )
    }

    async fn add_discussed(
        &self,
        _session_id: Uuid,
        _entities: &[(String, String)],
    ) -> Result<usize> {
        refus(
            "ChatStore::add_discussed",
            "les cibles (File/Function/Struct/Trait/Enum/Plan/Task/Decision) ne sont pas portées",
        )
    }

    async fn get_session_entities(
        &self,
        _session_id: Uuid,
        _project_id: Option<Uuid>,
    ) -> Result<Vec<DiscussedEntity>> {
        refus(
            "ChatStore::get_session_entities",
            "les cibles discutées ne sont pas portées",
        )
    }

    async fn get_discussed_co_changers(
        &self,
        _project_id: Uuid,
        _max_sessions: i64,
        _max_results: i64,
    ) -> Result<Vec<crate::neo4j::models::CoChanger>> {
        refus(
            "ChatStore::get_discussed_co_changers",
            "trois de ses entrées manquent : l'arête HAS_CHAT_SESSION (projet → session, non \
             écrite ici), les liens DISCUSSED vers File (`add_discussed` est refusé : ses cibles \
             — Function/Struct/Trait/Enum/Plan/Task/Decision — ne sont pas portées), et les nœuds \
             File eux-mêmes, que le domaine code écrit et qui n'existe pas encore. L'arête \
             CO_CHANGED a son code (`commit.rs`), mais elle est refusée elle aussi tant que \
             File n'est pas écrit",
        )
    }

    async fn backfill_discussed(&self) -> Result<(usize, usize, usize)> {
        refus(
            "ChatStore::backfill_discussed",
            "migration non portée (jamais exécutée au démarrage)",
        )
    }

    async fn create_associated_with(
        &self,
        _session_id: Uuid,
        _entity_type: &str,
        _entity_id: Uuid,
        _source: &str,
    ) -> Result<bool> {
        refus(
            "ChatStore::create_associated_with",
            "lie Plan ou Task : domaines non portés",
        )
    }

    async fn get_session_links(&self, _session_id: Uuid) -> Result<LinkedSessionInfo> {
        refus(
            "ChatStore::get_session_links",
            "lit AgentExecution, Task et Plan : domaines non portés",
        )
    }

    async fn get_sessions_for_plan(&self, _plan_id: Uuid) -> Result<Vec<SessionWithLinks>> {
        refus(
            "ChatStore::get_sessions_for_plan",
            "lit AgentExecution, Task et Plan : domaines non portés",
        )
    }

    async fn get_sessions_for_task(&self, _task_id: Uuid) -> Result<Vec<SessionWithLinks>> {
        refus(
            "ChatStore::get_sessions_for_task",
            "lit AgentExecution et Task : domaines non portés",
        )
    }

    async fn get_session_links_batch(
        &self,
        _session_ids: &[Uuid],
    ) -> Result<HashMap<Uuid, LinkedSessionInfo>> {
        refus(
            "ChatStore::get_session_links_batch",
            "lit AgentExecution, Task et Plan : domaines non portés",
        )
    }

    async fn get_task_plan_id(&self, _task_id: Uuid) -> Result<Option<Uuid>> {
        refus(
            "ChatStore::get_task_plan_id",
            "lit Task et Plan : domaines non portés",
        )
    }

    async fn search_episodes_by_embedding(
        &self,
        _project_id: Uuid,
        _embedding: &[f32],
        _limit: usize,
    ) -> Result<Vec<(Episode, f64)>> {
        refus(
            "ChatStore::search_episodes_by_embedding",
            "obrain ne journalise pas la valeur des vecteurs dans son WAL",
        )
    }
}

#[cfg(test)]
mod tests {
    //! Each test names the finding of the post-merge review of #1193 it proves
    //! (B = blocking, C = to fix, M = minor).
    use super::*;
    use crate::chat::anchor::NewAnchor;
    use crate::chat::provider::cognitive::segment::{
        Dimension, Fingerprint, Observation, SegmentPolicy, Segmenter,
    };
    use crate::chat::types::SpawnedBy;
    use crate::episodes::models::{
        FeedbackType, Outcome, Process, SessionTrace, Stimulus, StimulusTrigger, Validation,
    };
    use crate::obrain::{Creation, Interruption};
    use std::path::Path;

    fn session(id: Uuid) -> ChatSessionNode {
        let now = Utc::now();
        ChatSessionNode {
            id,
            cli_session_id: None,
            project_slug: Some("p".to_string()),
            workspace_slug: None,
            cwd: "/tmp".to_string(),
            title: None,
            model: "sonnet".to_string(),
            created_at: now,
            updated_at: now,
            message_count: 0,
            total_cost_usd: None,
            conversation_id: None,
            preview: None,
            permission_mode: None,
            add_dirs: None,
            spawned_by: None,
            provider_id: None,
            routed_by: None,
            routing_mode: None,
            routing_pool: None,
            capabilities: None,
            engine: None,
            resume_token: None,
            execution_place: ExecutionPlace::Project,
            project_inferred: false,
            access: SessionAccess::Normal,
            owner: None,
            file_scope_lifted: false,
            mirror_of: None,
        }
    }

    fn enfant_de(id: Uuid, parent: Uuid, created_at: DateTime<Utc>) -> ChatSessionNode {
        let mut s = session(id);
        s.created_at = created_at;
        s.spawned_by = Some(
            SpawnedBy::Conversation {
                parent_session_id: parent,
                tool_use_id: None,
            }
            .to_json_string(),
        );
        s
    }

    fn ouvrir(chemin: &Path) -> ObrainGraphStore {
        ObrainGraphStore::open(chemin, Creation::Permise).unwrap()
    }

    fn ancre(cible: &str) -> NewAnchor {
        NewAnchor::new(
            AnchorTargetType::File,
            cible,
            [AnchorRole::Focus],
            AnchorActor::User,
            "moi",
        )
    }

    fn evenement(session_id: Uuid, seq: i64) -> ChatEventRecord {
        ChatEventRecord {
            id: Uuid::new_v4(),
            session_id,
            seq,
            event_type: "user_message".to_string(),
            data: r#"{"content":"bonjour"}"#.to_string(),
            created_at: Utc::now(),
        }
    }

    fn noeuds(store: &ObrainGraphStore, etiquette: &str) -> usize {
        store.interne.db.store().nodes_by_label(etiquette).len()
    }

    fn segmentation() -> Segmenter {
        let obs = |turn: u32, id: &str| Observation {
            turn,
            prints: [(Dimension::Harness, Fingerprint::new([id]))]
                .into_iter()
                .collect(),
        };
        Segmenter::replay(
            SegmentPolicy::default(),
            7,
            &[obs(0, "deepseek"), obs(1, "claude-code")],
        )
        .unwrap()
    }

    /// B1: deleting a session takes ITS anchors out of the target pages, and
    /// only them.
    #[tokio::test]
    async fn b1_supprimer_une_session_garde_les_ancres_des_autres_sur_la_cible() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        for id in [a, b] {
            store.create_chat_session(&session(id)).await.unwrap();
            store.add_anchor(id, ancre("src/main.rs")).await.unwrap();
        }
        assert!(store.delete_chat_session(b).await.unwrap());
        let page = store
            .list_sessions_for_target(AnchorTargetType::File, "src/main.rs", None, 50, None)
            .await
            .unwrap();
        let sessions: Vec<Uuid> = page.items.iter().map(|x| x.session_id).collect();
        assert_eq!(sessions, vec![a]);
        // The deleted session's anchors left the index, not only its pages.
        assert_eq!(store.interne.etat.read().unwrap().chat.ancres.len(), 1);
        assert_eq!(
            store
                .mark_anchors_dangling(AnchorTargetType::File, "src/main.rs")
                .await
                .unwrap(),
            1
        );
    }

    /// B2: the anchor journal is served as soon as the change it records.
    #[tokio::test]
    async fn b2_journal_d_ancre_lu_sans_reouverture() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let s = Uuid::new_v4();
        store.create_chat_session(&session(s)).await.unwrap();
        let a = store.add_anchor(s, ancre("src/lib.rs")).await.unwrap();
        let journal = store.list_anchor_events(s).await.unwrap();
        assert_eq!(journal.len(), 1);
        assert_eq!(
            (journal[0].seq, journal[0].anchor_id, journal[0].kind),
            (1, a.id, AnchorEventKind::Added)
        );
        store
            .set_anchor_state(
                s,
                a.id,
                AnchorState::Dangling,
                None,
                AnchorActor::User,
                "moi",
            )
            .await
            .unwrap();
        let seqs: Vec<u64> = store
            .list_anchor_events(s)
            .await
            .unwrap()
            .iter()
            .map(|e| e.seq)
            .collect();
        assert_eq!(seqs, vec![1, 2]);
    }

    /// B2: the journal of a deleted session is deleted with it (on disk too).
    #[tokio::test]
    async fn b2_journal_d_ancre_supprime_avec_sa_session_apres_reouverture() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = ouvrir(&base);
        let s = Uuid::new_v4();
        store.create_chat_session(&session(s)).await.unwrap();
        store.add_anchor(s, ancre("src/lib.rs")).await.unwrap();
        assert!(store.delete_chat_session(s).await.unwrap());
        assert!(store.list_anchor_events(s).await.unwrap().is_empty());
        store.close().await.unwrap();
        drop(store);
        let store = ouvrir(&base);
        assert!(store.list_anchor_events(s).await.unwrap().is_empty());
        assert_eq!(noeuds(&store, EVENEMENT_ANCRE), 0);
    }

    /// C1: an event whose creation was interrupted is never replayed with
    /// default values; it is absent, or whole.
    #[tokio::test]
    async fn c1_evenement_interrompu_absent_ou_entier() {
        let dir = tempfile::tempdir().unwrap();
        let mut atteints = 0;
        for k in 1..=20 {
            let base = dir.path().join(format!("b{k}"));
            let store = ouvrir(&base);
            let s = Uuid::new_v4();
            store.create_chat_session(&session(s)).await.unwrap();
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store.store_chat_events(s, vec![evenement(s, 1)]).await;
            store.close().await.unwrap();
            drop(store);
            if issue.is_ok() {
                break;
            }
            atteints += 1;
            let store = ouvrir(&base);
            assert!(
                store.rapport_d_ouverture().en_quarantaine.is_empty(),
                "k={k} : {:?}",
                store.rapport_d_ouverture().en_quarantaine
            );
            let evts = store.get_chat_events(s, 0, 10).await.unwrap();
            assert!(
                evts.is_empty() || evts[0].event_type == "user_message",
                "k={k} : {evts:?}"
            );
            assert_eq!(store.count_chat_events(s).await.unwrap(), evts.len() as i64);
            store.close().await.unwrap();
        }
        assert!(atteints >= 7, "{atteints} points atteints");
    }

    /// C2: a setting whose creation was interrupted never reads back `""`.
    #[tokio::test]
    async fn c2_reglage_interrompu_jamais_lu_vide() {
        let dir = tempfile::tempdir().unwrap();
        let mut atteints = 0;
        for k in 1..=20 {
            let base = dir.path().join(format!("b{k}"));
            let store = ouvrir(&base);
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store.put_llm_setting("global", "cle", "valeur").await;
            store.close().await.unwrap();
            drop(store);
            if issue.is_ok() {
                break;
            }
            atteints += 1;
            let store = ouvrir(&base);
            assert!(
                store.rapport_d_ouverture().en_quarantaine.is_empty(),
                "k={k}"
            );
            let lu = store.get_llm_setting("global", "cle").await.unwrap();
            assert!(
                lu.is_none() || lu.as_deref() == Some("valeur"),
                "k={k} : {lu:?}"
            );
            store.close().await.unwrap();
        }
        assert!(atteints >= 2, "{atteints} points atteints");
    }

    /// C3 (a): a provider change interrupted at any write never leaves the old
    /// resume handles under the new provider; a restore interrupted never leaves
    /// the restored handles under the provider it replaces.
    #[tokio::test]
    async fn c3_changement_de_fournisseur_interrompu_sans_jeton_d_une_autre_famille() {
        let dir = tempfile::tempdir().unwrap();
        let mut ancienne = session(Uuid::new_v4());
        ancienne.provider_id = Some("ancien".to_string());
        ancienne.resume_token = Some("jeton-ancien".to_string());
        ancienne.cli_session_id = Some("cli-ancien".to_string());
        let mut atteints = 0;
        for k in 1..=20 {
            let base = dir.path().join(format!("u{k}"));
            let store = ouvrir(&base);
            store.create_chat_session(&ancienne).await.unwrap();
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store
                .update_chat_session_provider(ancienne.id, "nouveau", "m2", "r", "e")
                .await;
            store.close().await.unwrap();
            drop(store);
            if issue.is_ok() {
                break;
            }
            atteints += 1;
            let store = ouvrir(&base);
            let s = store.get_chat_session(ancienne.id).await.unwrap().unwrap();
            if s.provider_id.as_deref() == Some("nouveau") {
                assert_eq!((s.resume_token, s.cli_session_id), (None, None), "k={k}");
            }
            store.close().await.unwrap();
        }
        assert!(atteints >= 8, "{atteints} points atteints (changement)");
        let mut atteints = 0;
        for k in 1..=20 {
            let base = dir.path().join(format!("r{k}"));
            let store = ouvrir(&base);
            store.create_chat_session(&ancienne).await.unwrap();
            store
                .update_chat_session_provider(ancienne.id, "nouveau", "m2", "r", "e")
                .await
                .unwrap();
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store
                .restore_chat_session_provider(&ancienne, "ancien")
                .await;
            store.close().await.unwrap();
            drop(store);
            if issue.is_ok() {
                break;
            }
            atteints += 1;
            let store = ouvrir(&base);
            let s = store.get_chat_session(ancienne.id).await.unwrap().unwrap();
            if s.resume_token.is_some() || s.cli_session_id.is_some() {
                assert_eq!(s.provider_id.as_deref(), Some("ancien"), "k={k}");
            }
            store.close().await.unwrap();
        }
        assert!(atteints >= 8, "{atteints} points atteints (restauration)");
    }

    /// C3 (b): an anchor operation interrupted at any write never leaves two
    /// journal entries with the same `seq`.
    #[tokio::test]
    async fn c3_operation_d_ancre_interrompue_sans_seq_en_double() {
        let dir = tempfile::tempdir().unwrap();
        let mut atteints = 0;
        for k in 1..=60 {
            let base = dir.path().join(format!("b{k}"));
            let store = ouvrir(&base);
            let s = Uuid::new_v4();
            store.create_chat_session(&session(s)).await.unwrap();
            store.add_anchor(s, ancre("a.rs")).await.unwrap();
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store.add_anchor(s, ancre("b.rs")).await;
            store.close().await.unwrap();
            drop(store);
            if issue.is_ok() {
                break;
            }
            atteints += 1;
            let store = ouvrir(&base);
            assert!(
                store.rapport_d_ouverture().en_quarantaine.is_empty(),
                "k={k} : {:?}",
                store.rapport_d_ouverture().en_quarantaine
            );
            store.add_anchor(s, ancre("c.rs")).await.unwrap();
            let seqs: Vec<u64> = store
                .list_anchor_events(s)
                .await
                .unwrap()
                .iter()
                .map(|e| e.seq)
                .collect();
            let uniques: BTreeSet<u64> = seqs.iter().copied().collect();
            assert_eq!(uniques.len(), seqs.len(), "k={k} : {seqs:?}");
            store.close().await.unwrap();
        }
        assert!(atteints >= 10, "{atteints} points atteints");
    }

    /// C4: an unreadable segment or boundary fails the read of its conversation
    /// (never a shorter history), is quarantined, and a `put` heals it.
    #[tokio::test]
    async fn c4_segment_et_frontiere_illisibles_refusent_la_lecture() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let seg = segmentation();
        let (segments, frontieres) = (seg.segments(), seg.boundaries().to_vec());
        assert!(!frontieres.is_empty());
        let store = ouvrir(&base);
        store.put_segments("conv", &segments).await.unwrap();
        store.put_boundaries("conv", &frontieres).await.unwrap();
        // What a corruption leaves: a node of the conversation whose JSON is unreadable.
        {
            let db = &store.interne.db;
            let n = db.create_node(&[SEGMENT]);
            db.set_node_property(n, "conversation", Value::from("conv"));
            db.set_node_property(n, "id", Value::from("abime"));
            db.set_node_property(n, "json", Value::from("{pas du json"));
            let f = db.create_node(&[FRONTIERE]);
            db.set_node_property(f, "conversation", Value::from("conv"));
            db.set_node_property(f, "dimension", Value::from("topic"));
        }
        store.close().await.unwrap();
        drop(store);
        let store = ouvrir(&base);
        let quarantaine = &store.rapport_d_ouverture().en_quarantaine;
        assert!(
            quarantaine
                .iter()
                .any(|q| q.contains("ConversationSegment"))
                && quarantaine
                    .iter()
                    .any(|q| q.contains("ConversationBoundary")),
            "{quarantaine:?}"
        );
        assert!(store.get_segments("conv").await.is_err());
        assert!(store.get_boundaries("conv").await.is_err());
        assert!(store.get_segments("autre").await.unwrap().is_empty());
        // `put` replaces the set: the unreadable nodes go with it.
        store.put_segments("conv", &segments).await.unwrap();
        store.put_boundaries("conv", &frontieres).await.unwrap();
        assert_eq!(store.get_segments("conv").await.unwrap(), segments);
        assert_eq!(store.get_boundaries("conv").await.unwrap(), frontieres);
        store.close().await.unwrap();
        drop(store);
        let store = ouvrir(&base);
        assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
        assert_eq!(store.get_segments("conv").await.unwrap(), segments);
        assert_eq!(store.get_boundaries("conv").await.unwrap(), frontieres);
    }

    /// C5: the `updated_at` a fingerprint write puts on disk is the one served,
    /// before and after a restart.
    #[tokio::test]
    async fn c5_empreinte_publie_le_updated_at_ecrit() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = ouvrir(&base);
        let s = session(Uuid::new_v4());
        store.create_chat_session(&s).await.unwrap();
        let avant = store
            .get_chat_session(s.id)
            .await
            .unwrap()
            .unwrap()
            .updated_at;
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        store
            .set_chat_session_fingerprint(s.id, "fp")
            .await
            .unwrap();
        let publie = store
            .get_chat_session(s.id)
            .await
            .unwrap()
            .unwrap()
            .updated_at;
        assert!(publie > avant);
        store.update_chat_session_model(s.id, "opus").await.unwrap();
        let publie = store
            .get_chat_session(s.id)
            .await
            .unwrap()
            .unwrap()
            .updated_at;
        store.close().await.unwrap();
        drop(store);
        let store = ouvrir(&base);
        assert_eq!(
            store
                .get_chat_session(s.id)
                .await
                .unwrap()
                .unwrap()
                .updated_at,
            publie
        );
    }

    /// C6: an empty `routed_by` / `engine` is served `None`, as a re-read does.
    #[tokio::test]
    async fn c6_fournisseur_chaines_vides_servies_none() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = ouvrir(&base);
        let s = session(Uuid::new_v4());
        store.create_chat_session(&s).await.unwrap();
        store
            .update_chat_session_provider(s.id, "p", "m", "", "")
            .await
            .unwrap();
        let publie = store.get_chat_session(s.id).await.unwrap().unwrap();
        assert_eq!(
            (publie.routed_by.clone(), publie.engine.clone()),
            (None, None)
        );
        store.close().await.unwrap();
        drop(store);
        let store = ouvrir(&base);
        assert_eq!(store.get_chat_session(s.id).await.unwrap().unwrap(), publie);
    }

    /// C7: the episode served is the one written: without its vector.
    #[tokio::test]
    async fn c7_episode_servi_sans_le_vecteur_non_ecrit() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let s = Uuid::new_v4();
        store.create_chat_session(&session(s)).await.unwrap();
        let projet = Uuid::new_v4();
        let episode = Episode {
            id: Uuid::new_v4(),
            project_id: projet,
            stimulus: Stimulus {
                request: "r".to_string(),
                trigger: StimulusTrigger::UserRequest,
                timestamp: Utc::now(),
                context_hash: None,
            },
            process: Process {
                reasoning_tree_id: None,
                states_visited: vec![],
                state_visits: vec![],
                duration_ms: None,
            },
            outcome: Outcome {
                note_ids: vec![],
                decision_ids: vec![],
                commit_shas: vec![],
                files_modified: 0,
            },
            validation: Validation {
                feedback_type: FeedbackType::None,
                score: None,
                evidence_count: 0,
            },
            lesson: None,
            collected_at: Utc::now(),
            source_run_id: None,
            session: Some(SessionTrace {
                session_id: s,
                message_count: 1,
                close_reason: "closed".to_string(),
                total_cost_usd: None,
                agent_id: None,
            }),
            embedding: Some(vec![0.5; 4]),
        };
        assert!(store.save_session_episode(&episode).await.unwrap());
        let liste = store.list_session_episodes(projet, 10).await.unwrap();
        assert_eq!(liste.len(), 1);
        assert_eq!(liste[0].embedding, None);
    }

    /// M2: a lineage edge to another parent is refused, never repointed in the
    /// index alone.
    #[tokio::test]
    async fn m2_spawned_by_vers_un_autre_parent_refuse() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let (a, b, c) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        store.create_chat_session(&session(a)).await.unwrap();
        store.create_chat_session(&session(b)).await.unwrap();
        store
            .create_chat_session(&enfant_de(c, a, Utc::now()))
            .await
            .unwrap();
        let run = Uuid::new_v4();
        assert!(store
            .create_spawned_by_relation(&c.to_string(), &b.to_string(), "runner", Some(run), None)
            .await
            .is_err());
        assert!(store.get_run_sessions(run).await.unwrap().is_empty());
        // The same parent still fills the edge.
        store
            .create_spawned_by_relation(&c.to_string(), &a.to_string(), "runner", Some(run), None)
            .await
            .unwrap();
        assert_eq!(store.get_run_sessions(run).await.unwrap().len(), 1);
    }

    /// M3: the tree is ordered `depth, created_at, id` across the whole depth.
    #[tokio::test]
    async fn m3_arbre_ordonne_par_profondeur_date_puis_id() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let t = Utc::now();
        let s = |n: i64| t + chrono::Duration::seconds(n);
        let (r, c1, c2, g1, g2) = (
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        );
        store.create_chat_session(&session(r)).await.unwrap();
        store
            .create_chat_session(&enfant_de(c1, r, s(2)))
            .await
            .unwrap();
        store
            .create_chat_session(&enfant_de(c2, r, s(1)))
            .await
            .unwrap();
        // g1 is younger than g2 but hangs under the later parent.
        store
            .create_chat_session(&enfant_de(g1, c1, s(3)))
            .await
            .unwrap();
        store
            .create_chat_session(&enfant_de(g2, c2, s(4)))
            .await
            .unwrap();
        let ordre: Vec<String> = store
            .get_session_tree(&r.to_string())
            .await
            .unwrap()
            .into_iter()
            .map(|n| n.session_id)
            .collect();
        let attendu: Vec<String> = [r, c2, c1, g1, g2].iter().map(Uuid::to_string).collect();
        assert_eq!(ordre, attendu);
    }

    /// M4: `limit = 0` is no row, as Neo4j's `LIMIT 0`.
    #[tokio::test]
    async fn m4_get_chat_events_limit_zero_ne_rend_rien() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let s = Uuid::new_v4();
        store.create_chat_session(&session(s)).await.unwrap();
        store
            .store_chat_events(s, vec![evenement(s, 1), evenement(s, 2)])
            .await
            .unwrap();
        assert!(store.get_chat_events(s, 0, 0).await.unwrap().is_empty());
        assert_eq!(store.get_chat_events(s, 0, 1).await.unwrap().len(), 1);
    }

    /// M5: an execution-place event's `at` is written as Neo4j writes it
    /// (microseconds, `Z`), and served as a re-read serves it.
    #[tokio::test]
    async fn m5_lieu_at_a_la_microseconde() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = ouvrir(&base);
        let at = DateTime::parse_from_rfc3339("2026-01-02T03:04:05.123456789+00:00")
            .unwrap()
            .with_timezone(&Utc);
        let e = ExecutionPlaceEvent {
            id: Uuid::new_v4(),
            placement_id: Uuid::new_v4(),
            kind: ExecutionPlaceEventKind::Requested,
            task_id: Uuid::new_v4(),
            plan_id: Uuid::new_v4(),
            requested_by: "person:moi".to_string(),
            owner: "moi".to_string(),
            actor: None,
            session_id: None,
            path: "/x".to_string(),
            repository: "r".to_string(),
            base_ref: "main".to_string(),
            base_sha: "abc".to_string(),
            branch: "b".to_string(),
            detail: None,
            at,
        };
        store.create_execution_place_event(&e).await.unwrap();
        let attendu = DateTime::parse_from_rfc3339("2026-01-02T03:04:05.123456Z")
            .unwrap()
            .with_timezone(&Utc);
        let publie = store.list_execution_place_events(e.task_id).await.unwrap();
        assert_eq!(publie[0].at, attendu);
        let brut = store
            .interne
            .db
            .store()
            .nodes_by_label(LIEU)
            .into_iter()
            .next()
            .and_then(|n| store.interne.db.get_node(n))
            .and_then(|n| {
                n.get_property("at")
                    .and_then(|v| v.as_str().map(str::to_string))
            })
            .unwrap();
        assert_eq!(brut, "2026-01-02T03:04:05.123456Z");
        store.close().await.unwrap();
        drop(store);
        let store = ouvrir(&base);
        assert_eq!(
            store.list_execution_place_events(e.task_id).await.unwrap(),
            publie
        );
    }

    /// M7: an agent root whose creation was interrupted never makes the next
    /// call create a second one.
    #[tokio::test]
    async fn m7_racine_d_agent_interrompue_reste_unique() {
        let dir = tempfile::tempdir().unwrap();
        let mut atteints = 0;
        for k in 1..=10 {
            let base = dir.path().join(format!("b{k}"));
            let store = ouvrir(&base);
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store.agent_identity_namespace().await;
            store.close().await.unwrap();
            drop(store);
            if issue.is_ok() {
                break;
            }
            atteints += 1;
            let store = ouvrir(&base);
            assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
            let ns = store.agent_identity_namespace().await.unwrap();
            store.close().await.unwrap();
            drop(store);
            let store = ouvrir(&base);
            assert_eq!(noeuds(&store, RACINE_AGENT), 1, "k={k}");
            assert_eq!(store.agent_identity_namespace().await.unwrap(), ns, "k={k}");
            store.close().await.unwrap();
        }
        assert!(atteints >= 1);
    }
}
