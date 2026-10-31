//! `/ws/activity` — the live feed of everything the orchestrator does.
//!
//! ## Why this endpoint exists next to `/ws/events`
//!
//! `/ws/events` already streams raw CRUD events and can filter them by
//! `project_id`. That filter is exactly what the Activity Hub needs, and it is
//! also exactly what makes `/ws/events` unusable for it: the events that matter
//! most arrive **unscoped**. The runner emits its `RunnerEvent` bridge with
//! `project_id: None`, and so does the chat manager when a session starts or
//! stops streaming. Filtering that stream by project would silently drop every
//! plan-run and conversation event.
//!
//! This endpoint fixes the attribution instead of dropping the events: it keeps
//! a small index (plan run → project, protocol run → project, session → project
//! slug) refreshed from the graph, fills in the missing `project_id`, and only
//! then applies the filter. The payload is the normalized [`ActivityEvent`]
//! (sequence number, category, title, status, raw detail) rather than the raw
//! CRUD envelope, so the client filters on one axis it can reason about.
//!
//! ## Replay
//!
//! `?last_seq=` replays what the client missed since the snapshot it took, out
//! of the in-memory ring of the [`ActivityHub`] (last 1000 events). The
//! subscription is opened **before** the replay is read, and duplicates are
//! dropped by keeping the highest `seq` sent: an event published during the
//! hand-off is delivered exactly once, never zero times.
//!
//! ## Backpressure
//!
//! A slow client makes the broadcast channel lag. Rather than disconnecting it,
//! the handler sends `{"kind":"lag_dropped","skipped":N}` and carries on: the
//! client knows its log has a hole and can re-read the snapshot, which is the
//! only honest recovery.
use super::handlers::OrchestratorState;
use super::ws_auth::CookieAuthResult;
use crate::auth::jwt::Claims;
use crate::chat::manager::ChatManager;
use crate::events::{ActivityCategory, ActivityEvent};
use crate::neo4j::GraphStore;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Query, State, WebSocketUpgrade,
    },
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::time::{interval, Duration};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// How often the attribution index is refreshed from the graph.
const INDEX_REFRESH_INTERVAL: Duration = Duration::from_secs(10);
/// Minimum delay between two refreshes triggered by an unknown run id.
const UNKNOWN_ID_REFRESH_FLOOR: Duration = Duration::from_secs(2);

/// Query parameters of `/ws/activity`.
#[derive(Debug, Deserialize, Default)]
pub struct WsActivityQuery {
    /// Restrict to one project. Events that cannot be attributed are dropped
    /// while this is set — an unattributed event is not known to be ours.
    pub project_id: Option<Uuid>,
    /// Restrict to the projects of one workspace (best effort: conversations
    /// carry a workspace slug, plan runs do not).
    pub workspace_slug: Option<String>,
    /// Comma-separated categories: `plan_run,protocol_run,chat_session,entity,system`.
    pub entity_types: Option<String>,
    /// Comma-separated statuses to keep (`running`, `completed`, `in_progress`…).
    /// Events with no status in their payload are always kept.
    pub statuses: Option<String>,
    /// Replay every event strictly newer than this sequence number.
    pub last_seq: Option<u64>,
    /// One-time ticket for auth (fallback when cookies aren't sent on WS upgrade).
    pub ticket: Option<String>,
}

/// Frames sent by `/ws/activity`.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ActivityFrame<'a> {
    /// Sent first: the sequence number the live stream starts from, and how many
    /// events were replayed before it.
    Ready { seq: u64, replayed: usize },
    /// One normalized event.
    Activity { event: &'a ActivityEvent },
    /// The client was too slow: `skipped` events were dropped by the bus.
    LagDropped { skipped: u64 },
}

/// Attribution index: the graph knowledge the CRUD events do not carry.
///
/// Rebuilt from the graph every [`INDEX_REFRESH_INTERVAL`], and on demand when
/// an event names a run id the index has never seen (a run started after the
/// last refresh).
#[derive(Debug, Default)]
struct AttributionIndex {
    /// plan run id → owning project
    plan_runs: HashMap<String, Option<Uuid>>,
    /// protocol run id → owning project
    protocol_runs: HashMap<String, Option<Uuid>>,
    /// chat session id → project slug
    sessions: HashMap<String, Option<String>>,
}

impl AttributionIndex {
    async fn refresh(&mut self, graph: &Arc<dyn GraphStore>, chat: Option<&Arc<ChatManager>>) {
        if let Ok(runs) = graph.list_active_plan_runs().await {
            self.plan_runs = runs
                .into_iter()
                .map(|run| (run.run_id.to_string(), run.project_id))
                .collect();
        }
        if let Ok((runs, _)) = graph
            .list_all_protocol_runs(
                Some(crate::protocol::RunStatus::Running),
                None,
                None,
                200,
                0,
            )
            .await
        {
            self.protocol_runs = runs
                .into_iter()
                .map(|run| (run.id.to_string(), run.project_id))
                .collect();
        }
        if let Some(manager) = chat {
            let live: HashMap<Uuid, _> = manager.session_activity_map().await;
            let mut sessions = HashMap::with_capacity(live.len());
            for session_id in live.keys() {
                let slug = graph
                    .get_chat_session(*session_id)
                    .await
                    .ok()
                    .flatten()
                    .and_then(|node| node.project_slug);
                sessions.insert(session_id.to_string(), slug);
            }
            self.sessions = sessions;
        }
    }

    /// The project id an event belongs to, when the index can tell.
    ///
    /// A conversation is deliberately absent: its node carries the project's
    /// *slug*, not its id, so conversations are matched on the slug (see
    /// [`Filters::accepts`]).
    fn project_of(&self, event: &ActivityEvent) -> Option<Uuid> {
        if let Some(pid) = event.project_id.as_deref().and_then(|p| p.parse().ok()) {
            return Some(pid);
        }
        match event.category {
            ActivityCategory::PlanRun => self.plan_runs.get(&event.entity_id).copied().flatten(),
            ActivityCategory::ProtocolRun => {
                self.protocol_runs.get(&event.entity_id).copied().flatten()
            }
            _ => None,
        }
    }

    /// The project slug a conversation belongs to.
    fn slug_of_session(&self, session_id: &str) -> Option<&str> {
        self.sessions.get(session_id).and_then(|s| s.as_deref())
    }

    /// Whether the index knows this event's subject at all.
    fn knows(&self, event: &ActivityEvent) -> bool {
        match event.category {
            ActivityCategory::PlanRun => self.plan_runs.contains_key(&event.entity_id),
            ActivityCategory::ProtocolRun => self.protocol_runs.contains_key(&event.entity_id),
            ActivityCategory::ChatSession => self.sessions.contains_key(&event.entity_id),
            _ => true,
        }
    }
}

/// What the connection keeps between events.
#[derive(Debug)]
struct Filters {
    project_id: Option<Uuid>,
    /// Slug of the requested project (conversations are matched on it).
    project_slug: Option<String>,
    workspace_slug: Option<String>,
    /// Projects of the requested workspace: a workspace is a set of projects, not
    /// a field on a run, so events are matched on membership.
    workspace_project_ids: Option<HashSet<Uuid>>,
    workspace_project_slugs: Option<HashSet<String>>,
    categories: Option<HashSet<ActivityCategory>>,
    statuses: Option<HashSet<String>>,
}

impl Filters {
    fn parse(query: &WsActivityQuery) -> Result<Self, String> {
        let categories = match query.entity_types.as_deref() {
            None => None,
            Some(raw) => {
                let mut set = HashSet::new();
                for name in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    match ActivityCategory::parse(name) {
                        Some(category) => {
                            set.insert(category);
                        }
                        None => {
                            return Err(format!(
                                "unknown entity type `{name}` (expected one of: plan_run, protocol_run, chat_session, entity, system)"
                            ));
                        }
                    }
                }
                Some(set)
            }
        };
        let statuses = query.statuses.as_deref().map(|raw| {
            raw.split(',')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect::<HashSet<_>>()
        });
        Ok(Self {
            project_id: query.project_id,
            project_slug: None,
            workspace_slug: query.workspace_slug.clone(),
            workspace_project_ids: None,
            workspace_project_slugs: None,
            categories,
            statuses,
        })
    }

    fn is_scoped(&self) -> bool {
        self.project_id.is_some() || self.workspace_slug.is_some()
    }

    fn accepts(
        &self,
        event: &ActivityEvent,
        project_id: Option<Uuid>,
        session_slug: Option<&str>,
    ) -> bool {
        if let Some(categories) = &self.categories {
            if !categories.contains(&event.category) {
                return false;
            }
        }
        if let Some(statuses) = &self.statuses {
            if let Some(status) = &event.status {
                if !statuses.contains(&status.to_ascii_lowercase()) {
                    return false;
                }
            }
        }
        if let Some(wanted) = self.project_id {
            // A conversation is scoped by slug (that is what its node carries);
            // every other event is scoped by project id.
            let matched = project_id == Some(wanted)
                || (self.project_slug.is_some()
                    && session_slug.is_some()
                    && session_slug == self.project_slug.as_deref());
            if !matched {
                return false;
            }
        } else if self.workspace_slug.is_some() {
            // A workspace is resolved to its projects once, at connect. An event
            // we cannot attribute to one of them is dropped: a workspace page must
            // not show another workspace's work.
            let by_id = project_id
                .zip(self.workspace_project_ids.as_ref())
                .map(|(id, ids)| ids.contains(&id))
                .unwrap_or(false);
            let by_slug = session_slug
                .zip(self.workspace_project_slugs.as_ref())
                .map(|(slug, slugs)| slugs.contains(slug))
                .unwrap_or(false);
            if !by_id && !by_slug {
                return false;
            }
        }
        true
    }
}

/// WebSocket upgrade handler for `/ws/activity`.
///
/// Authentication happens **before** the upgrade, like every other WS route:
/// cookie first, ticket as fallback (WKWebView does not send cookies on upgrade).
pub async fn ws_activity(
    ws: WebSocketUpgrade,
    State(state): State<OrchestratorState>,
    Query(query): Query<WsActivityQuery>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let filters = match Filters::parse(&query) {
        Ok(filters) => filters,
        Err(reason) => {
            warn!(reason = %reason, "WS /ws/activity: refusing malformed filter");
            return (StatusCode::BAD_REQUEST, reason).into_response();
        }
    };

    let neo4j = state.orchestrator.neo4j_arc();
    let auth_result = super::ws_auth::ws_authenticate(
        &headers,
        &state.auth_config,
        &neo4j,
        query.ticket.as_deref(),
        &state.ws_ticket_store,
    )
    .await;

    match auth_result {
        CookieAuthResult::Authenticated(claims) => {
            // A workspace is a set of projects: resolve it once, before the
            // upgrade, so an unknown slug is a 404 instead of a connection that
            // silently filters everything out.
            let mut filters = filters;
            if filters.project_id.is_none() {
                if let Some(slug) = filters.workspace_slug.clone() {
                    match resolve_workspace_scope(&neo4j, &slug).await {
                        Ok((ids, slugs)) => {
                            filters.workspace_project_ids = Some(ids);
                            filters.workspace_project_slugs = Some(slugs);
                        }
                        Err(status) => return status.into_response(),
                    }
                }
            }
            info!(
                email = %claims.email,
                project = ?query.project_id,
                workspace = ?query.workspace_slug,
                "WS /ws/activity: authenticated, upgrading"
            );
            ws.on_upgrade(move |socket| {
                handle_activity_ws(socket, state, filters, query.last_seq, claims)
            })
            .into_response()
        }
        CookieAuthResult::Invalid(reason) => {
            warn!(reason = %reason, "WS /ws/activity: auth REJECTED (401)");
            (StatusCode::UNAUTHORIZED, reason).into_response()
        }
    }
}

/// The project ids and slugs of a workspace. `NOT_FOUND` when it does not exist.
async fn resolve_workspace_scope(
    graph: &Arc<dyn GraphStore>,
    slug: &str,
) -> Result<(HashSet<Uuid>, HashSet<String>), StatusCode> {
    let workspace = match graph.get_workspace_by_slug(slug).await {
        Ok(Some(workspace)) => workspace,
        Ok(None) => {
            warn!(workspace = %slug, "WS /ws/activity: unknown workspace");
            return Err(StatusCode::NOT_FOUND);
        }
        Err(error) => {
            warn!(%error, "WS /ws/activity: workspace lookup failed");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };
    match graph.list_workspace_projects(workspace.id).await {
        Ok(projects) => Ok((
            projects.iter().map(|project| project.id).collect(),
            projects
                .iter()
                .map(|project| project.slug.clone())
                .collect(),
        )),
        Err(error) => {
            warn!(%error, "WS /ws/activity: workspace projects lookup failed");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// The connection loop: replay, then live until the client leaves.
async fn handle_activity_ws(
    mut socket: WebSocket,
    state: OrchestratorState,
    mut filters: Filters,
    last_seq: Option<u64>,
    claims: Claims,
) {
    super::ws_auth::wait_ready_then_auth_ok(&mut socket, &claims, None).await;
    let (mut sender, mut receiver) = socket.split();

    let hub = state.event_bus.activity().clone();
    let graph = state.orchestrator.neo4j_arc();
    let chat = state.chat_manager.clone();

    // Subscribe BEFORE replaying: an event published during the hand-off is then
    // in the receiver, and the `seq > sent` rule below drops the duplicate. Doing
    // it the other way round would lose it.
    let mut rx = hub.subscribe();

    // The project slug is what conversations are compared against; resolve it
    // once from the requested project id.
    if let Some(project_id) = filters.project_id {
        filters.project_slug = graph
            .get_project(project_id)
            .await
            .ok()
            .flatten()
            .map(|project| project.slug);
    }

    let mut index = AttributionIndex::default();
    index.refresh(&graph, chat.as_ref()).await;

    let replay_from = last_seq.unwrap_or(0);
    let lost = hub.lost_before(replay_from);
    let replayed = hub.replay_since(replay_from);
    let mut sent_seq = replay_from;

    if !send_json(
        &mut sender,
        &ActivityFrame::Ready {
            seq: hub.latest_seq(),
            replayed: replayed.len(),
        },
    )
    .await
    {
        return;
    }
    if lost > 0 {
        // The client asked for events the buffer no longer holds: say so instead
        // of pretending the stream is contiguous.
        if !send_json(&mut sender, &ActivityFrame::LagDropped { skipped: lost }).await {
            return;
        }
    }
    for mut event in replayed {
        if event.seq <= sent_seq {
            continue;
        }
        sent_seq = event.seq;
        attribute(&mut event, &index);
        if filters.accepts(
            &event,
            index.project_of(&event),
            index.slug_of_session(&event.entity_id),
        ) && !send_json(&mut sender, &ActivityFrame::Activity { event: &event }).await
        {
            return;
        }
    }

    let mut ping_interval = interval(Duration::from_secs(30));
    ping_interval.tick().await; // skip the immediate first tick
    let mut refresh_interval = interval(INDEX_REFRESH_INTERVAL);
    refresh_interval.tick().await;
    let mut last_forced_refresh = tokio::time::Instant::now();

    loop {
        tokio::select! {
            received = rx.recv() => {
                match received {
                    Ok(mut event) => {
                        if event.seq <= sent_seq {
                            continue;
                        }
                        sent_seq = event.seq;
                        attribute(&mut event, &index);
                        // A run or session we have never seen (started after the
                        // last refresh): look again, at most once every couple of
                        // seconds, so a burst of unknown ids cannot hammer Neo4j.
                        if filters.is_scoped()
                            && !index.knows(&event)
                            && event.category != ActivityCategory::Entity
                            && last_forced_refresh.elapsed() >= UNKNOWN_ID_REFRESH_FLOOR
                        {
                            index.refresh(&graph, chat.as_ref()).await;
                            last_forced_refresh = tokio::time::Instant::now();
                            attribute(&mut event, &index);
                        }
                        let project_id = index.project_of(&event);
                        let session_slug = index.slug_of_session(&event.entity_id).map(str::to_string);
                        if filters.accepts(&event, project_id, session_slug.as_deref())
                            && !send_json(&mut sender, &ActivityFrame::Activity { event: &event }).await
                        {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        warn!(skipped, "WS /ws/activity: client lagged, dropping events");
                        if !send_json(&mut sender, &ActivityFrame::LagDropped { skipped }).await {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        debug!("WS /ws/activity: activity channel closed");
                        break;
                    }
                }
            }

            _ = refresh_interval.tick() => {
                index.refresh(&graph, chat.as_ref()).await;
                last_forced_refresh = tokio::time::Instant::now();
            }

            incoming = receiver.next() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Pong(_))) => {}
                    Some(Ok(_)) => {}
                    Some(Err(error)) => {
                        debug!(%error, "WS /ws/activity: socket error");
                        break;
                    }
                }
            }

            _ = ping_interval.tick() => {
                if sender.send(Message::Ping(Vec::new().into())).await.is_err() {
                    break;
                }
            }
        }
    }

    debug!(email = %claims.email, "WS /ws/activity disconnected");
}

/// Fill in what the emitter did not stamp.
///
/// The runner and the chat manager publish with `project_id: None` (see the
/// module docs); the index knows who owns the run or the session, so the event
/// can be attributed before any filter looks at it.
fn attribute(event: &mut ActivityEvent, index: &AttributionIndex) {
    if event.project_id.is_none() {
        if let Some(project_id) = index.project_of(event) {
            event.project_id = Some(project_id.to_string());
        }
    }
}

/// Send one JSON frame. Returns `false` when the socket is gone.
async fn send_json<S>(
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    value: &S,
) -> bool
where
    S: Serialize,
{
    let payload = match serde_json::to_string(value) {
        Ok(payload) => payload,
        Err(error) => {
            warn!(%error, "WS /ws/activity: failed to serialize a frame");
            return true; // not a transport failure: keep the connection
        }
    };
    sender.send(Message::Text(payload.into())).await.is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::ActivityEvent;

    fn event(
        category: ActivityCategory,
        entity_id: &str,
        project_id: Option<&str>,
    ) -> ActivityEvent {
        ActivityEvent {
            seq: 1,
            ts: "2026-01-01T00:00:00Z".to_string(),
            category,
            entity_type: "runner".to_string(),
            entity_id: entity_id.to_string(),
            action: "updated".to_string(),
            project_id: project_id.map(str::to_string),
            title: None,
            status: None,
            detail: serde_json::Value::Null,
        }
    }

    fn filters(query: WsActivityQuery) -> Filters {
        Filters::parse(&query).expect("valid filters")
    }

    #[test]
    fn unknown_entity_type_is_refused() {
        let error = Filters::parse(&WsActivityQuery {
            entity_types: Some("plan_run,nope".to_string()),
            ..Default::default()
        })
        .unwrap_err();
        assert!(
            error.contains("nope"),
            "error should name the culprit: {error}"
        );
    }

    #[test]
    fn categories_filter_is_applied() {
        let f = filters(WsActivityQuery {
            entity_types: Some("plan_run".to_string()),
            ..Default::default()
        });
        assert!(f.accepts(&event(ActivityCategory::PlanRun, "r1", None), None, None));
        assert!(!f.accepts(
            &event(ActivityCategory::ChatSession, "s1", None),
            None,
            None
        ));
    }

    #[test]
    fn project_filter_drops_unattributed_events() {
        let project = Uuid::nil();
        let f = filters(WsActivityQuery {
            project_id: Some(project),
            ..Default::default()
        });
        // Attributed to the right project: kept.
        assert!(f.accepts(
            &event(ActivityCategory::PlanRun, "r1", Some(&project.to_string())),
            Some(project),
            None
        ));
        // Attributed to another project: dropped.
        assert!(!f.accepts(
            &event(
                ActivityCategory::PlanRun,
                "r2",
                Some(&Uuid::new_v4().to_string())
            ),
            Some(Uuid::new_v4()),
            None
        ));
        // Not attributable at all: dropped, never leaked.
        assert!(!f.accepts(&event(ActivityCategory::PlanRun, "r3", None), None, None));
    }

    #[test]
    fn conversation_is_matched_on_its_project_slug() {
        let project = Uuid::nil();
        let f = Filters {
            project_id: Some(project),
            project_slug: Some("nexus-sdk".to_string()),
            workspace_slug: None,
            workspace_project_ids: None,
            workspace_project_slugs: None,
            categories: None,
            statuses: None,
        };
        assert!(f.accepts(
            &event(ActivityCategory::ChatSession, "s1", None),
            None,
            Some("nexus-sdk")
        ));
        assert!(!f.accepts(
            &event(ActivityCategory::ChatSession, "s2", None),
            None,
            Some("other")
        ));
    }

    #[test]
    fn workspace_filter_keeps_only_its_own_projects() {
        let mine = Uuid::new_v4();
        let theirs = Uuid::new_v4();
        let mut f = filters(WsActivityQuery {
            workspace_slug: Some("ws".to_string()),
            ..Default::default()
        });
        f.workspace_project_ids = Some(HashSet::from([mine]));
        f.workspace_project_slugs = Some(HashSet::from(["nexus-sdk".to_string()]));

        // A run of one of the workspace's projects: kept.
        assert!(f.accepts(
            &event(ActivityCategory::PlanRun, "r1", None),
            Some(mine),
            None
        ));
        // A run of another workspace: dropped — a workspace page must not show it.
        assert!(!f.accepts(
            &event(ActivityCategory::PlanRun, "r2", None),
            Some(theirs),
            None
        ));
        // A conversation is matched on its project slug.
        assert!(f.accepts(
            &event(ActivityCategory::ChatSession, "s1", None),
            None,
            Some("nexus-sdk")
        ));
        assert!(!f.accepts(
            &event(ActivityCategory::ChatSession, "s2", None),
            None,
            Some("elsewhere")
        ));
        // Unattributable: dropped, never leaked.
        assert!(!f.accepts(&event(ActivityCategory::PlanRun, "r3", None), None, None));
    }

    #[test]
    fn status_filter_keeps_events_without_a_status() {
        let f = filters(WsActivityQuery {
            statuses: Some("running".to_string()),
            ..Default::default()
        });
        let mut running = event(ActivityCategory::PlanRun, "r1", None);
        running.status = Some("running".to_string());
        let mut completed = event(ActivityCategory::PlanRun, "r2", None);
        completed.status = Some("completed".to_string());
        let unknown = event(ActivityCategory::PlanRun, "r3", None);
        assert!(f.accepts(&running, None, None));
        assert!(!f.accepts(&completed, None, None));
        assert!(
            f.accepts(&unknown, None, None),
            "no status in the payload is not a refusal"
        );
    }

    #[test]
    fn attribution_fills_the_missing_project() {
        let project = Uuid::nil();
        let mut index = AttributionIndex::default();
        index.plan_runs.insert("run-1".to_string(), Some(project));
        let mut ev = event(ActivityCategory::PlanRun, "run-1", None);
        attribute(&mut ev, &index);
        assert_eq!(ev.project_id.as_deref(), Some(project.to_string().as_str()));
    }

    #[test]
    fn index_knows_only_what_it_indexed() {
        let index = AttributionIndex::default();
        assert!(!index.knows(&event(ActivityCategory::PlanRun, "run-1", None)));
        assert!(index.knows(&event(ActivityCategory::Entity, "task-1", None)));
    }
}
