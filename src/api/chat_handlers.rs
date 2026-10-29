//! Chat API handlers — session management (streaming via WebSocket)
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use crate::api::handlers::{AppError, OrchestratorState};
use crate::api::query::{PaginatedResponse, PaginationParams};
use crate::chat::types::{
    ChatLinkedPlan, ChatLinkedRfc, ChatLinkedTask, ChatRequest, ChatSession, CreateSessionResponse,
    MessageSearchResult, SessionActivity,
};
use crate::events::{CrudAction, CrudEvent, EntityType, EventEmitter};

#[allow(unused_imports)]
use crate::neo4j::traits::*;
use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Create session + send first message
// ============================================================================

/// POST /api/chat/sessions — Create a new chat session and send the first
/// message, OR resume an existing session when `session_id` is provided.
///
/// The resume path mirrors the WS handler's 3-branch routing:
/// 1. session active locally → `send_message`
/// 2. session owned by another instance → `try_remote_send` (NATS RPC)
/// 3. no instance owns it → `resume_session` (respawns the CLI)
///
/// This is the path used by REST/MCP callers (`chat` tool, `send_message`
/// action) which previously could only START conversations: `ChatRequest`
/// documented `session_id` as "Session ID to resume" but `create_session`
/// ignored it and always generated a fresh UUID.
/// Whether this new-session request lifts the session's file scope (decision fce98b8a): a
/// PERSON ([`crate::auth::jwt::Claims::may_lift_file_scope`]: not an agent session, not a
/// long-lived MCP token, not the server's account, not the anonymous user) opens a NEW session
/// and names trust (`trust` or `bypassPermissions`) explicitly. A resume, a mode left to the
/// default and every agent caller do not.
pub(crate) fn person_lifts_file_scope(
    caller: &crate::chat::envelope::SpawnCaller,
    request: &crate::chat::types::ChatRequest,
) -> bool {
    use crate::chat::envelope::{mode_rank, SpawnCaller, TRUST_RANK};
    matches!(caller, SpawnCaller::Human)
        && request.session_id.is_none()
        && request
            .user_claims
            .as_ref()
            .is_some_and(|claims| claims.may_lift_file_scope())
        && request
            .permission_mode
            .as_deref()
            .is_some_and(|mode| mode_rank(mode) >= TRUST_RANK)
}

pub async fn create_session(
    State(state): State<OrchestratorState>,
    claims: Option<axum::Extension<crate::auth::jwt::Claims>>,
    headers: axum::http::HeaderMap,
    Json(mut request): Json<ChatRequest>,
) -> Result<Json<CreateSessionResponse>, AppError> {
    use crate::chat::envelope;

    // Who is asking? A chat session calling through its MCP server is held to
    // its spawn envelope (decision A17); a person is not.
    let caller = envelope::identify_caller(
        claims.as_ref().map(|c| &c.0),
        envelope::session_header(&headers),
        state.auth_config.is_some(),
    )?;

    // Inject authenticated user claims into the request so ChatManager
    // can generate a session token for the MCP subprocess.
    if let Some(axum::Extension(c)) = claims {
        request.user_claims = Some(c);
    }
    // A person opening a session in trust lifts its file scope (decision fce98b8a). Never
    // from the body (`serde(skip)`), never for an agent, an MCP token or the server's account.
    request.file_scope_lifted = person_lifts_file_scope(&caller, &request);

    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    // The opening of an agent's child, held until `create_session` below has returned (the
    // child is then live): the envelope counts it for the parent's next openings (P18-suite,
    // point 6 — an `Agent` call of the same parent in flight could otherwise pass the cap).
    let opening = request
        .session_id
        .is_none()
        .then(|| caller.reserve_opening(&chat_manager.child_openings))
        .flatten();
    if let envelope::SpawnCaller::Agent { session_id, .. } = &caller {
        let graph = state.orchestrator.neo4j_arc();
        match request.session_id.as_deref() {
            // Sending to an existing session: only one this session spawned.
            Some(target) => envelope::ensure_child_of(graph.as_ref(), session_id, target).await?,
            // Opening a session: inside the caller's envelope, parent recorded
            // from the token.
            None => {
                let env = envelope::envelope_for_caller(
                    graph.as_ref(),
                    chat_manager.as_ref(),
                    &caller,
                    chat_manager.max_live_children(),
                    opening.as_ref(),
                )
                .await?
                .expect("an agent caller always has an envelope");
                let default_mode = chat_manager.default_permission_mode().await;
                env.apply(&mut request, &default_mode)?;
                request.spawned_by = Some(envelope::conversation_spawned_by(env.parent_session_id));
            }
        }
    }

    // Fold the references and the attached documents into the message once,
    // before either path: both persist and broadcast `request.message`, so the
    // chips survive replay. Also makes inert any block typed into the text.
    request.message = crate::refs::compose::compose_user_message(
        &state.orchestrator.neo4j_arc(),
        &request.message,
        &request.refs,
        &request.attachments,
        chat_manager.refs_v1_enabled(),
    )
    .await?;
    request.attachments.clear();
    request.refs.clear();

    // ── Resume path ────────────────────────────────────────────────────────
    if let Some(sid) = request.session_id.clone() {
        Uuid::parse_str(&sid)
            .map_err(|_| AppError::BadRequest("Invalid session_id UUID".to_string()))?;
        // A read-only mirror of a Claude Code sub-agent takes no message (P18c): 409
        // `mirror_read_only`, before anything is sent, proxied or resumed.
        chat_manager
            .refuse_if_mirror(&sid)
            .await
            .map_err(|e| AppError::from_open_error(e, None))?;

        chat_manager
            .check_provider_binding(&sid, request.provider.as_deref())
            .await
            .map_err(|e| AppError::from_open_error(e, request.provider.as_deref()))?;

        // Read before the liveness check and the NATS RPC (up to its timeout): a resume
        // that completes meanwhile then receives this message instead of being resumed
        // over (follow-up of the review of #696).
        let from = chat_manager.resume_point(&sid).await;
        if chat_manager.is_session_active(&sid).await {
            // 1. Session is local — send directly into the running CLI
            chat_manager
                .send_message(&sid, &request.message)
                .await
                .map_err(AppError::Internal)?;
        } else if chat_manager
            .try_remote_send(&sid, &request.message, "user_message")
            .await
            .unwrap_or(false)
        {
            // 2. Message proxied to the owning instance via NATS RPC
        } else {
            // 3. No instance owns the session — resume locally (spawns CLI).
            //    Surfaces a 404 when the session doesn't exist in Neo4j.
            chat_manager
                .resume_session_from(&sid, &request.message, request.user_claims.as_ref(), from)
                .await
                .map_err(|e| {
                    let typed =
                        crate::chat::provider::errors::classify_open_error(&e, None).is_some();
                    if !typed && e.to_string().contains("not found") {
                        AppError::NotFound(format!("Session {} not found", sid))
                    } else {
                        AppError::from_open_error(e, request.provider.as_deref())
                    }
                })?;
        }

        // Same side-effects as the WS path: entity extraction + live refresh
        super::ws_chat_handler::spawn_entity_extraction(&state, &sid, &request.message);
        state.event_bus.emit(
            CrudEvent::new(EntityType::ChatSession, CrudAction::Updated, &sid).with_payload(
                serde_json::json!({
                    "project_slug": request.project_slug,
                    "resumed": true,
                }),
            ),
        );

        // A resume never changes the session's access or place: say what is stored
        // (a client that resumes must learn that the session is read-only without a
        // second call). An unreadable record answers the defaults, as before.
        let (execution_place, access) =
            stored_place_and_access(state.orchestrator.neo4j(), &sid).await;
        return Ok(Json(CreateSessionResponse {
            session_id: sid.clone(),
            stream_url: format!("/ws/chat/{}", sid),
            execution_place,
            access,
            notices: Vec::new(),
        }));
    }

    // ── Create path (no session_id) ────────────────────────────────────────
    let response = chat_manager
        .create_session(&request)
        .await
        .map_err(|e| AppError::from_open_error(e, request.provider.as_deref()))?;

    // T4.3: Extract code entities from the first message and create DISCUSSED relations (non-blocking)
    super::ws_chat_handler::spawn_entity_extraction(&state, &response.session_id, &request.message);

    // The session lists learn it from the `created` event `ChatManager` emits itself for
    // every session it opens (with `spawned_by`): emitting one here sent it twice.
    Ok(Json(response))
}

// ============================================================================
// Message history
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct MessagesQuery {
    #[serde(default = "default_messages_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_messages_limit() -> usize {
    50
}

/// GET /api/chat/sessions/{id}/messages — Get message history
///
/// Returns persisted chat events as `messages`. Each event includes its full
/// payload (type, content, tool info, etc.) plus injected `seq` and `created_at`
/// metadata (`created_at`: seconds since the epoch, milliseconds as the fraction).
/// The frontend reconstructs the ChatMessage UI model from these events.
pub async fn list_messages(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
    Query(query): Query<MessagesQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    let loaded = chat_manager
        .get_session_messages(
            &session_id.to_string(),
            Some(query.limit),
            Some(query.offset),
        )
        .await
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("not found") || msg.contains("no conversation_id") {
                AppError::NotFound(msg)
            } else {
                AppError::Internal(e)
            }
        })?;

    // Events are sorted by seq ASC from Neo4j. Each event's `data` field is
    // the JSON-serialized ChatEvent (includes type tag, tool_use, tool_result, etc.)
    let messages: Vec<serde_json::Value> = loaded
        .events
        .iter()
        .map(|e| {
            // Parse the data field back to a JSON object to return structured events
            let mut obj = serde_json::from_str::<serde_json::Value>(&e.data)
                .unwrap_or_else(|_| serde_json::json!({ "type": e.event_type, "raw": e.data }));
            // Inject metadata + ensure "type" tag is always present
            // (legacy user_message events were stored as {"content":"..."} without a type tag)
            if let Some(map) = obj.as_object_mut() {
                map.entry("type".to_string())
                    .or_insert_with(|| serde_json::json!(e.event_type));
                // Only inject the Neo4j UUID as "id" if the event doesn't already have one.
                // tool_use and tool_result events carry a Claude tool_call_id in their "id" field
                // which must be preserved for proper tool_use ↔ tool_result matching.
                map.entry("id".to_string())
                    .or_insert_with(|| serde_json::json!(e.id.to_string()));
                map.insert("seq".to_string(), serde_json::json!(e.seq));
                map.insert(
                    "created_at".to_string(),
                    serde_json::json!(super::ws_chat_handler::wire_seconds(e.created_at)),
                );
            }
            obj
        })
        .collect();

    Ok(Json(serde_json::json!({
        "messages": messages,
        "total_count": loaded.total_count,
        "has_more": loaded.has_more,
        "offset": loaded.offset,
        "limit": loaded.limit,
    })))
}

// ============================================================================
// Session CRUD
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct SessionsListQuery {
    #[serde(default)]
    pub project_slug: Option<String>,
    #[serde(default)]
    pub workspace_slug: Option<String>,
    /// Filter by plan ID — returns sessions linked to this plan (both runner + manual).
    #[serde(default)]
    pub plan_id: Option<Uuid>,
    /// Filter by task ID — returns sessions linked to this task.
    #[serde(default)]
    pub task_id: Option<Uuid>,
    /// Include detached sessions (spawned by runner/sub-agent). Defaults to false.
    #[serde(default)]
    pub include_detached: bool,
    #[serde(flatten)]
    pub pagination: PaginationParams,
}

/// Where and with what access a stored session runs, for the answer of a resume.
/// A record that cannot be read answers the defaults (project, normal).
async fn stored_place_and_access(
    graph: &dyn crate::neo4j::traits::GraphStore,
    session_id: &str,
) -> (
    crate::neo4j::models::ExecutionPlace,
    crate::chat::provider::policy::SessionAccess,
) {
    let Ok(id) = Uuid::parse_str(session_id) else {
        return Default::default();
    };
    match graph.get_chat_session(id).await {
        Ok(Some(s)) => (s.execution_place, s.access),
        _ => Default::default(),
    }
}

/// What a stored capabilities snapshot says of `images`; `None` when there is no snapshot
/// or it does not say.
fn snapshot_images(capabilities: Option<&serde_json::Value>) -> Option<bool> {
    capabilities?.get("images")?.as_bool()
}

/// Convert a ChatSessionNode to a ChatSession API response (without links).
fn session_node_to_response(s: crate::neo4j::models::ChatSessionNode) -> ChatSession {
    ChatSession {
        id: s.id.to_string(),
        cli_session_id: s.cli_session_id,
        project_slug: s.project_slug,
        workspace_slug: s.workspace_slug,
        cwd: s.cwd,
        title: s.title,
        model: s.model,
        created_at: s.created_at.to_rfc3339(),
        updated_at: s.updated_at.to_rfc3339(),
        message_count: s.message_count,
        total_cost_usd: s.total_cost_usd,
        conversation_id: s.conversation_id,
        preview: s.preview,
        permission_mode: s.permission_mode,
        add_dirs: s.add_dirs,
        spawned_by: s.spawned_by.and_then(|sb| serde_json::from_str(&sb).ok()),
        linked_plans: Vec::new(),
        linked_tasks: Vec::new(),
        linked_rfcs: Vec::new(),
        activity: None,
        provider_id: s.provider_id,
        routing_mode: s.routing_mode,
        routing_pool: s
            .routing_pool
            .as_deref()
            .and_then(|json| serde_json::from_str(json).ok()),
        capabilities: None,
        effective_capabilities: None,
        routed_by: s.routed_by,
        execution_place: s.execution_place,
        access: s.access,
        mirror_of: s.mirror_of,
    }
}

/// Stamp every session in `items` with its live activity, read in ONE pass
/// over the chat manager's in-memory map.
///
/// Why this exists: `is_streaming` is an `AtomicBool` inside `ActiveSession`
/// and nothing writes it to Neo4j, so a listing built from the graph alone
/// cannot say which conversation is working. Clients used to learn it only
/// from `chat_session` CRUD events that happened to arrive while their list
/// was mounted — meaning a reload showed every running conversation as idle.
///
/// Quiet sessions keep `activity: None` so a page of cold sessions is
/// exactly as small on the wire as it was before this field existed.
async fn stamp_activity(state: &OrchestratorState, items: &mut [ChatSession]) {
    let Some(cm) = state.chat_manager.as_ref() else {
        return;
    };
    let snap = cm.live_session_snapshot().await;
    if snap.live.is_empty() {
        return;
    }
    for item in items.iter_mut() {
        if let Ok(id) = item.id.parse::<Uuid>() {
            let activity = snap.activity_for(id);
            if !activity.is_quiet() {
                item.activity = Some(activity);
            }
        }
    }
}

/// Enrich a ChatSession with linked plans/tasks/RFCs from a LinkedSessionInfo.
fn enrich_session_with_links(
    session: &mut ChatSession,
    links: &crate::neo4j::models::LinkedSessionInfo,
) {
    session.linked_plans = links
        .linked_plans
        .iter()
        .map(|p| ChatLinkedPlan {
            id: p.id.to_string(),
            title: p.title.clone(),
            source: p.source.clone(),
        })
        .collect();
    session.linked_tasks = links
        .linked_tasks
        .iter()
        .map(|t| ChatLinkedTask {
            id: t.id.to_string(),
            title: t.title.clone(),
            source: t.source.clone(),
        })
        .collect();
    session.linked_rfcs = links
        .linked_rfcs
        .iter()
        .map(|r| ChatLinkedRfc {
            id: r.id.to_string(),
            title: r.title.clone(),
        })
        .collect();
}

/// GET /api/chat/sessions — List chat sessions
///
/// Supports filtering by `plan_id` or `task_id` (returns linked sessions via dual-path query).
/// When neither is provided, falls back to the standard project_slug/workspace_slug filter.
pub async fn list_sessions(
    State(state): State<OrchestratorState>,
    Query(query): Query<SessionsListQuery>,
) -> Result<Json<PaginatedResponse<ChatSession>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let neo4j = state.orchestrator.neo4j();

    // If plan_id or task_id is provided, use the specialized query
    if let Some(plan_id) = query.plan_id {
        let sessions_with_links = neo4j
            .get_sessions_for_plan(plan_id)
            .await
            .map_err(AppError::Internal)?;

        let total = sessions_with_links.len();
        let mut items: Vec<ChatSession> = sessions_with_links
            .into_iter()
            .skip(query.pagination.offset)
            .take(query.pagination.validated_limit())
            .map(|sw| {
                let mut session = session_node_to_response(sw.session);
                enrich_session_with_links(&mut session, &sw.links);
                session
            })
            .collect();
        stamp_activity(&state, &mut items).await;

        return Ok(Json(PaginatedResponse::new(
            items,
            total,
            query.pagination.validated_limit(),
            query.pagination.offset,
        )));
    }

    if let Some(task_id) = query.task_id {
        let sessions_with_links = neo4j
            .get_sessions_for_task(task_id)
            .await
            .map_err(AppError::Internal)?;

        let total = sessions_with_links.len();
        let mut items: Vec<ChatSession> = sessions_with_links
            .into_iter()
            .skip(query.pagination.offset)
            .take(query.pagination.validated_limit())
            .map(|sw| {
                let mut session = session_node_to_response(sw.session);
                enrich_session_with_links(&mut session, &sw.links);
                session
            })
            .collect();
        stamp_activity(&state, &mut items).await;

        return Ok(Json(PaginatedResponse::new(
            items,
            total,
            query.pagination.validated_limit(),
            query.pagination.offset,
        )));
    }

    // Standard list with optional project_slug/workspace_slug filter
    let (sessions, total) = neo4j
        .list_chat_sessions(
            query.project_slug.as_deref(),
            query.workspace_slug.as_deref(),
            query.pagination.validated_limit(),
            query.pagination.offset,
            query.include_detached,
        )
        .await
        .map_err(AppError::Internal)?;

    // Batch-enrich with links
    let session_ids: Vec<Uuid> = sessions.iter().map(|s| s.id).collect();
    let links_map = neo4j
        .get_session_links_batch(&session_ids)
        .await
        .unwrap_or_default();

    let mut items: Vec<ChatSession> = sessions
        .into_iter()
        .map(|s| {
            let sid = s.id;
            let mut session = session_node_to_response(s);
            if let Some(links) = links_map.get(&sid) {
                enrich_session_with_links(&mut session, links);
            }
            session
        })
        .collect();
    stamp_activity(&state, &mut items).await;

    Ok(Json(PaginatedResponse::new(
        items,
        total,
        query.pagination.validated_limit(),
        query.pagination.offset,
    )))
}

/// GET /api/chat/sessions/{id} — Get session details (enriched with linked plans/tasks/RFCs)
pub async fn get_session(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
) -> Result<Json<ChatSession>, AppError> {
    let neo4j = state.orchestrator.neo4j();

    let node = neo4j
        .get_chat_session(session_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Session {} not found", session_id)))?;

    // The frozen capability snapshot rides on the single-session read only.
    let capabilities = node
        .capabilities
        .as_deref()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok());
    let snapshot_images = snapshot_images(capabilities.as_ref());
    let mut session = session_node_to_response(node);
    session.capabilities = capabilities;
    // F-R4: what the next turn can really carry, given the routing candidates.
    if let Some(chat_manager) = state.chat_manager.as_ref() {
        session.effective_capabilities = chat_manager
            .effective_capabilities(&session_id.to_string(), snapshot_images)
            .await;
    }

    // Enrich with linked entities (best-effort — don't fail if enrichment fails)
    if let Ok(links) = neo4j.get_session_links(session_id).await {
        enrich_session_with_links(&mut session, &links);
    }

    stamp_activity(&state, std::slice::from_mut(&mut session)).await;

    Ok(Json(session))
}

/// GET /api/chat/sessions/{id}/children — Get child sessions spawned by this session
pub async fn get_session_children(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
) -> Result<Json<Vec<ChatSession>>, AppError> {
    let children = state
        .orchestrator
        .neo4j()
        .get_session_children(session_id)
        .await
        .map_err(AppError::Internal)?;

    let mut items: Vec<ChatSession> = children.into_iter().map(session_node_to_response).collect();
    stamp_activity(&state, &mut items).await;

    Ok(Json(items))
}

/// GET /api/chat/sessions/{id}/tree — Get the full session tree rooted at this session
pub async fn get_session_tree(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
) -> Result<Json<Vec<crate::neo4j::models::SessionTreeNode>>, AppError> {
    let mut tree = state
        .orchestrator
        .neo4j()
        .get_session_tree(&session_id.to_string())
        .await
        .map_err(AppError::Internal)?;
    // Which provider and model ran each node and what it cost, read from the
    // sessions themselves; a session that cannot be read stays unannotated.
    let chat_manager = state.chat_manager.as_ref();
    let mut info = std::collections::HashMap::new();
    for node in &tree {
        if let Ok(id) = node.session_id.parse::<Uuid>() {
            if let Ok(Some(s)) = state.orchestrator.neo4j().get_chat_session(id).await {
                let status = match chat_manager {
                    Some(cm) if cm.is_session_active(&node.session_id).await => {
                        Some("active".to_string())
                    }
                    _ => None,
                };
                info.insert(
                    node.session_id.clone(),
                    crate::chat::tree::NodeInfo {
                        provider_id: s.provider_id,
                        model: Some(s.model),
                        cost_usd: s.total_cost_usd,
                        status,
                        routed_by: s.routed_by,
                    },
                );
            }
        }
    }
    let cap = state
        .chat_manager
        .as_ref()
        .map_or(crate::chat::envelope::MAX_LIVE_CHILDREN, |manager| {
            manager.max_live_children()
        });
    crate::chat::tree::annotate(&mut tree, &info, cap);
    Ok(Json(tree))
}

/// GET /api/chat/runs/{run_id}/costs — the run's two counters (marginal: real
/// spend; notional: subscription / free) and the split by model, provider and
/// task class (A21). An unknown cost is counted apart, never as zero.
pub async fn get_run_costs(
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<crate::chat::cost::CostReport>, AppError> {
    let executions = state
        .orchestrator
        .neo4j()
        .get_agent_executions_for_run(run_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(crate::chat::cost::report(&executions)))
}

/// GET /api/chat/sessions/{id}/usage — the token usage of a conversation, turn by turn:
/// input (not cached), output, cache reads and writes, reasoning, and the cache-hit ratio.
/// Read from the `result` events stored with the turns (every engine, every provider).
pub async fn get_session_usage(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
) -> Result<Json<crate::chat::cost::SessionUsage>, AppError> {
    let neo4j = state.orchestrator.neo4j();
    neo4j
        .get_chat_session(session_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Session {} not found", session_id)))?;
    const PAGE: i64 = 1000;
    let mut results = Vec::new();
    let mut after = -1;
    loop {
        let page = neo4j
            .get_chat_events(session_id, after, PAGE)
            .await
            .map_err(AppError::Internal)?;
        let Some(last) = page.last().map(|e| e.seq) else {
            break;
        };
        results.extend(
            page.iter()
                .filter(|e| e.event_type == "result")
                .filter_map(|e| {
                    Some((
                        e.seq,
                        serde_json::from_str::<serde_json::Value>(&e.data).ok()?,
                    ))
                }),
        );
        if (page.len() as i64) < PAGE || last <= after {
            break;
        }
        after = last;
    }
    Ok(Json(crate::chat::cost::session_usage(
        results.iter().map(|(seq, data)| (*seq, data)),
    )))
}

/// GET /api/chat/runs/{run_id}/sessions — Get all sessions for a PlanRun
pub async fn get_run_sessions(
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<Vec<crate::neo4j::models::SessionInfo>>, AppError> {
    let sessions = state
        .orchestrator
        .neo4j()
        .get_run_sessions(run_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(sessions))
}

/// DELETE /api/chat/sessions/{id} — Delete a session
pub async fn delete_session(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    // The chat manager closes the session if it runs and removes what it keeps on
    // disk (a native session's transcript) before the node.
    let deleted = match &state.chat_manager {
        Some(chat_manager) => chat_manager.delete_session(session_id).await,
        None => {
            state
                .orchestrator
                .neo4j()
                .delete_chat_session(session_id)
                .await
        }
    }
    .map_err(AppError::Internal)?;

    if deleted {
        // Emit CRUD event for live refresh
        state.event_bus.emit(CrudEvent::new(
            EntityType::ChatSession,
            CrudAction::Deleted,
            session_id.to_string(),
        ));

        Ok(Json(serde_json::json!({ "deleted": true })))
    } else {
        Err(AppError::NotFound(format!(
            "Session {} not found",
            session_id
        )))
    }
}

/// POST /api/chat/sessions/{id}/close — Close a live session WITHOUT deleting it.
///
/// Interrupts the running turn, disconnects the CLI and removes the session from
/// the live map (freeing its envelope slot), but keeps the node and its
/// transcript: a later message resumes it. Distinct from DELETE, which removes
/// the node. This is the explicit closure a delegated session needs when its
/// provider process died and no turn is left to end it (ZOMBIE-CLI).
pub async fn close_session(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let Some(chat_manager) = state.chat_manager.as_ref() else {
        // Without a live manager nothing is running; there is nothing to close.
        return Ok(Json(serde_json::json!({
            "closed": false,
            "reason": "no live chat manager",
        })));
    };
    chat_manager
        .close_session(&session_id.to_string())
        .await
        .map_err(AppError::Internal)?;
    // The node survives but its live state changed: tell the UI to refresh.
    state.event_bus.emit(CrudEvent::new(
        EntityType::ChatSession,
        CrudAction::Updated,
        session_id.to_string(),
    ));
    Ok(Json(serde_json::json!({ "closed": true })))
}

// ============================================================================
// Cancel running tools (T3 of plan 28e9afe3)
// ============================================================================

/// POST /api/chat/sessions/{id}/cancel-tools — Kill the currently-running
/// tool subprocess(es) of a session WITHOUT ending the LLM turn.
///
/// Sends `SIGINT` to every descendant of the CLI process. The running
/// shell child (find, npm install, …) exits with code 130, BashTool
/// returns a normal `tool_result` with `isError: true`, and the agent's
/// turn continues — it can decide to call another tool, abandon, retry…
///
/// Distinct from the existing interrupt mechanism, which ends the turn
/// (sets `interrupt_flag` + sends the SDK control_request `interrupt`
/// that triggers `QueryEngine.abortController.abort()` in the CLI).
/// See decision `d2bf0e7b` of plan 28e9afe3 for the empirical analysis
/// behind this distinction.
///
/// ## Response codes
///
/// **200** with `CancelToolsResult { cli_pid, killed_pids, capped }` when the
/// instance holding the session acted (here or over NATS). The frontend distinguishes:
/// - `capped: false, !killed_pids.is_empty()` → tool(s) cancelled OK
/// - `capped: false, killed_pids.is_empty()` → no tool was running
///   (agent thinking, between turns) — silently no-op
/// - `capped: true` → rate cap hit (10/60s/session), display a
///   "slow down" toast and disable the button briefly
///
/// Otherwise a typed body `{error, code, retryable}` (`cancel_error`): 422
/// `unsupported` (the provider cannot: Claude Code over SSH), 409
/// `owner_unreachable` (no instance holds the session live — with or without
/// NATS), 504 `owner_timeout` (`retryable: false` here: a late cancel may have
/// happened, a retry would stop tools started since), 410 `session_gone`
/// (retryable: the session moved), 502 `owner_protocol` / `owner_failed` /
/// `relay_failed`.
///
/// **404** — `chat_manager` not configured (server not started with
/// chat support).
///
/// (Rationale for 200-not-429 on cap: matches existing PO patterns
/// like `cancel_run` which returns 200 with structured body. Avoids
/// adding an `AppError::TooManyRequests` variant just for one
/// endpoint.)
///
/// ## Cross-instance routing
///
/// If the session lives on another instance, the request goes over NATS
/// (`events.chat.{id}.cancel_tools`, request/reply, `chat::cancel_relay`) and
/// the answer is the owner's: what it stopped (its `cli_pid`, `killed_pids`),
/// or its typed refusal / the typed reason there is none (above).
pub async fn cancel_tools(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let chat_manager = state.chat_manager.as_ref().ok_or_else(|| {
        AppError::NotFound("chat_manager not configured on this server".to_string())
    })?;

    let result = chat_manager
        .cancel_running_tools(&session_id.to_string())
        .await
        // A provider that cannot stop tools (Claude Code over SSH: no `tool_cancel`)
        // is a typed refusal (422 `unsupported`), never a 500 nor a success; so is
        // a session another instance holds and does not answer for (`cancel_error`).
        .map_err(cancel_error)?;

    Ok(Json(serde_json::to_value(&result).unwrap_or_default()))
}

// ============================================================================
// Interrupt (REST fallback for the WebSocket `interrupt` frame)
// ============================================================================

/// Body of `POST /api/chat/sessions/{id}/interrupt`. Entirely optional —
/// an absent body, or `{}`, means "stop everything".
#[derive(Debug, Deserialize, Default)]
pub struct InterruptRequest {
    /// `"turn_and_tools"` (default) ends the turn and SIGINTs every
    /// descendant of the CLI. `"turn"` ends the turn only, leaving
    /// background subprocesses (`Bash`/`Monitor`) alive.
    #[serde(default)]
    pub scope: Option<String>,
    /// Also stop every descendant session (delegations). The answer then
    /// carries `cascade: { stopped, total }`.
    #[serde(default)]
    pub cascade: Option<bool>,
}

/// POST /api/chat/sessions/{id}/interrupt — End the current turn of a
/// session.
///
/// ## Why this exists
///
/// Interrupting used to be reachable **only** through the WebSocket frame
/// `{"type":"interrupt"}` (`ws_chat_handler.rs`). That made Stop depend on
/// a healthy socket, which is exactly the condition that fails: when the
/// chat WS drops on a route change, `ChatWebSocket.send()` returns `false`,
/// the frontend dropped the return value, and the interrupt vanished with
/// no error anywhere. Meanwhile the frontend had *already* been calling
/// this very path (`chatApi.interruptSession`) for detached runs and child
/// sessions — into a 404, swallowed by four empty `catch` blocks.
///
/// So this endpoint is both the missing route those callers expected and
/// the transport-independent fallback the composer's Stop button needs.
///
/// ## Response
///
/// Always **200** with an `InterruptOutcome`. Read `delivered`:
/// - `true` — a live local turn was interrupted.
/// - `false` with `routed: "nats"` — not local; published for whichever
///   instance owns the session.
/// - `false` with `routed: "none"` — nothing was stopped anywhere. The UI
///   should clear its "Stopping…" state rather than spin forever.
///
/// **400** — unknown `scope`. **404** — `chat_manager` not configured.
pub async fn interrupt_session(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
    body: Option<Json<InterruptRequest>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let chat_manager = state.chat_manager.as_ref().ok_or_else(|| {
        AppError::NotFound("chat_manager not configured on this server".to_string())
    })?;

    let (scope, cascade) = match body {
        Some(Json(b)) => (
            b.scope.unwrap_or_else(|| "turn_and_tools".to_string()),
            b.cascade.unwrap_or(false),
        ),
        None => ("turn_and_tools".to_string(), false),
    };

    let kill_tools = match scope.as_str() {
        "turn_and_tools" => true,
        "turn" => false,
        other => {
            return Err(AppError::BadRequest(format!(
                "unknown interrupt scope '{other}' (expected 'turn' or 'turn_and_tools')"
            )))
        }
    };

    // Descendants first (leaves before parents), so a parent never gets the
    // time to start a new child while its subtree is being stopped.
    let cascade_report = if cascade {
        let mut descendants = Vec::new();
        let mut frontier = vec![session_id];
        let mut seen = std::collections::HashSet::from([session_id]);
        while let Some(parent) = frontier.pop() {
            let children = state
                .orchestrator
                .neo4j()
                .get_session_children(parent)
                .await
                .map_err(AppError::Internal)?;
            for child in children {
                if seen.insert(child.id) {
                    descendants.push(child.id);
                    frontier.push(child.id);
                }
            }
        }
        let total = descendants.len();
        let mut stopped = 0usize;
        for id in descendants.into_iter().rev() {
            let sid = id.to_string();
            if chat_manager.is_session_active(&sid).await
                && chat_manager
                    .interrupt_scoped(&sid, kill_tools)
                    .await
                    .is_ok()
            {
                stopped += 1;
            }
        }
        Some(serde_json::json!({ "stopped": stopped, "total": total }))
    } else {
        None
    };

    let outcome = chat_manager
        .interrupt_scoped(&session_id.to_string(), kill_tools)
        .await
        .map_err(AppError::Internal)?;

    let mut body = serde_json::to_value(&outcome).unwrap_or_default();
    if let (Some(report), Some(obj)) = (cascade_report, body.as_object_mut()) {
        obj.insert("cascade".to_string(), report);
    }
    Ok(Json(body))
}

// ============================================================================
// Action routes: answer a permission, send a message (REST twins of the WS frames)
// ============================================================================

/// Body of `POST /api/chat/sessions/{id}/permissions/{request_id}`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PermissionAnswerRequest {
    pub allow: bool,
    /// How long an approval lasts: `once` (absent), `session`, `always`.
    #[serde(default)]
    pub scope: crate::chat::types::PermissionAnswerScope,
}

/// Body of `POST /api/chat/sessions/{id}/messages`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SendMessageRequest {
    pub content: String,
    /// Ids of documents already uploaded through `POST /api/documents`.
    #[serde(default)]
    pub attachments: Vec<Uuid>,
    /// References the message points at (`{kind, id}` objects). Checked and
    /// folded into the message by `refs::compose`; ignored when `refs_v1` is off.
    #[serde(default)]
    pub refs: Vec<serde_json::Value>,
}

/// The person a lasting-rule route serves: a signed-in PERSON with their own session
/// token (P11b, "human routes only"). An agent session's token, an MCP token, the
/// server's account and the anonymous user are refused: an agent never lists nor
/// revokes the approvals of its person (it cannot add any either: a lasting rule only
/// comes from that person answering a permission request of their own session).
fn lasting_rules_person(
    claims: &Option<axum::Extension<crate::auth::jwt::Claims>>,
) -> Result<String, AppError> {
    match claims {
        Some(axum::Extension(c)) if c.may_keep_lasting_approvals() => Ok(c.sub.clone()),
        Some(_) => Err(AppError::Forbidden(
            "lasting permission rules are managed by their person only, signed in".to_string(),
        )),
        None => Err(AppError::Unauthorized(
            "sign in to manage your lasting permission rules".to_string(),
        )),
    }
}

/// Query of `GET /api/chat/permission-rules`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PermissionRulesQuery {
    /// Only the rules of this project (slug).
    #[serde(default)]
    pub project: Option<String>,
}

/// One lasting rule as `GET /api/chat/permission-rules` lists it.
#[derive(Debug, Serialize)]
pub struct PermissionRuleView {
    pub id: Uuid,
    pub project: String,
    /// The project's root directory (canonical) it is bound to: it only covers a call of a
    /// session running in that very directory.
    pub root: String,
    /// The call asked to run outside the sandbox. Never kept by this server (refused for
    /// good), shown should a rule file say so.
    pub sandbox_disabled: bool,
    /// The engine it holds on: `claude_code` (the Claude Code CLI) or `native`.
    pub engine: String,
    pub tool: String,
    /// What it covers, as the decisions say it (`Bash: cat README.md`): the identical call.
    pub rule: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub session_id: Option<String>,
    /// Whether the backend's comparator still grants it, decided as a session of its project
    /// would (a rule edited by hand, one whose read-only declaration was withdrawn or is
    /// redefined by the project's `.mcp.json` is listed but holds nothing). It still needs
    /// a session running in its root to apply.
    pub holds: bool,
}

/// GET /api/chat/permission-rules[?project=<slug>]
///
/// The caller's lasting approvals (`always`, P11b), newest first. Human route only
/// (a person signed in with their own session token).
///
/// - **200** `{ "enabled": bool, "rules": [PermissionRuleView] }` — `enabled: false` when
///   this server keeps no lasting approval (`always` is then never offered).
/// - **401** — not signed in. **403** — an agent's, an MCP or the server's token.
pub async fn list_permission_rules(
    State(state): State<OrchestratorState>,
    claims: Option<axum::Extension<crate::auth::jwt::Claims>>,
    Query(query): Query<PermissionRulesQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let person = lasting_rules_person(&claims)?;
    let chat_manager = state.chat_manager.as_ref().ok_or_else(|| {
        AppError::NotFound("chat_manager not configured on this server".to_string())
    })?;
    let Some(store) = chat_manager.lasting_rule_store() else {
        return Ok(Json(serde_json::json!({ "enabled": false, "rules": [] })));
    };
    let declared = chat_manager.config.read_only_mcp_tools.clone();
    let mut rules: Vec<PermissionRuleView> =
        tokio::task::spawn_blocking(move || store.rules(&person))
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
            .into_iter()
            .filter(|r| query.project.as_ref().is_none_or(|p| *p == r.project))
            .map(|r| PermissionRuleView {
                // Decided as a session of that project would decide it: the declared
                // read-only MCP tools its `.mcp.json` does not redefine (review of #714,
                // N2); a rule with no root (older format) holds nothing.
                holds: !r.root.is_empty()
                    && r.grant(&crate::chat::session_grants::declarations_for_project(
                        &declared,
                        Some(std::path::Path::new(&r.root)),
                    ))
                    .is_some(),
                sandbox_disabled: r
                    .input
                    .get("dangerouslyDisableSandbox")
                    .is_some_and(|v| !v.is_null() && *v != serde_json::Value::Bool(false)),
                id: r.id,
                project: r.project,
                root: r.root,
                engine: r.asker,
                tool: r.tool,
                rule: r.rule,
                created_at: r.created_at,
                session_id: r.session_id,
            })
            .collect();
    rules.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(Json(serde_json::json!({ "enabled": true, "rules": rules })))
}

/// DELETE /api/chat/permission-rules/{id}
///
/// Revokes one of the caller's lasting approvals: from the next request on, the call it
/// covered is asked again, in every session (the rules are read at each request).
///
/// - **204** — revoked. **404** — no such rule of the caller (or none kept here).
/// - **401** / **403** — as for the list.
pub async fn revoke_permission_rule(
    State(state): State<OrchestratorState>,
    claims: Option<axum::Extension<crate::auth::jwt::Claims>>,
    Path(id): Path<Uuid>,
) -> Result<axum::http::StatusCode, AppError> {
    let person = lasting_rules_person(&claims)?;
    let chat_manager = state.chat_manager.as_ref().ok_or_else(|| {
        AppError::NotFound("chat_manager not configured on this server".to_string())
    })?;
    let store = chat_manager
        .lasting_rule_store()
        .ok_or_else(|| AppError::NotFound(format!("no lasting permission rule {id}")))?;
    let revoked = tokio::task::spawn_blocking(move || store.revoke(&person, id))
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;
    if revoked {
        Ok(axum::http::StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!(
            "no lasting permission rule {id}"
        )))
    }
}

const PERMISSION_GONE_REASON: &str = "le CLI qui demandait s'est arrêté ; continue par un message \
     (POST .../messages)";

/// POST /api/chat/sessions/{id}/permissions/{request_id} `{ "allow": bool, "scope"?: "once" | "session" | "always" }`
///
/// REST twin of the WS `permission_response` frame: both go through
/// `ChatManager::route_permission_response`, so there is one routing.
///
/// - **200** `{ "routed": "local" | "remote" }` — delivered to the CLI.
/// - **400** `permission_scope_unsupported` — the session does not offer that scope
///   (`capabilities.permission_scopes`); nothing was answered, the request still waits.
/// - **403** — the session belongs to another person (`ChatSessionNode::owner`).
/// - **404** — unknown session, or no such permission request on it.
/// - **409** — the request was already decided (double click, two tabs).
/// - **410** — the CLI that asked is gone; the answer is REFUSED, never
///   silently dropped. Continue with `POST .../messages`.
pub async fn respond_permission(
    State(state): State<OrchestratorState>,
    Path((session_id, request_id)): Path<(Uuid, String)>,
    claims: Option<axum::Extension<crate::auth::jwt::Claims>>,
    Json(body): Json<PermissionAnswerRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::chat::attention::{permission_status, PermissionStatus};
    use crate::chat::manager::PermissionDeliveryError;

    let chat_manager = state.chat_manager.as_ref().ok_or_else(|| {
        AppError::NotFound("chat_manager not configured on this server".to_string())
    })?;
    if request_id.trim().is_empty() {
        return Err(AppError::BadRequest("request_id must not be empty".into()));
    }
    let neo4j = state.orchestrator.neo4j();
    let session = neo4j
        .get_chat_session(session_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Session {} not found", session_id)))?;
    // A read-only mirror of a Claude Code sub-agent (P18c) is answered in its parent.
    if session.mirror_of.is_some() {
        return Err(AppError::Provider(Box::new(
            crate::chat::mirror::MirrorReadOnly::failure(),
        )));
    }
    // Only the person the session belongs to answers it (review #679 finding 6): another
    // signed-in user could otherwise approve, or grant for the session, a call in someone
    // else's conversation. A caller without claims (authentication off) is nobody: a
    // session that has an owner is refused, as on the WebSocket (which carries the
    // anonymous user then); a session without one is answered as before.
    let caller = claims
        .as_ref()
        .map_or("", |axum::Extension(c)| c.sub.as_str());
    if !session.answerable_by(caller) {
        return Err(AppError::Forbidden(
            "only the person this conversation belongs to can answer its permission requests"
                .to_string(),
        ));
    }
    // A lasting approval only from the person, signed in with their own session token (review
    // of #714, S1): an MCP token or the server's account answers once or for the session.
    if body.allow
        && body.scope == crate::chat::types::PermissionAnswerScope::Always
        && !claims
            .as_ref()
            .is_some_and(|axum::Extension(c)| c.may_keep_lasting_approvals())
    {
        return Err(AppError::Forbidden(
            "only the person, signed in with their own session, can allow a call for good"
                .to_string(),
        ));
    }

    let status = |events: Vec<crate::neo4j::models::ChatEventRecord>| {
        permission_status(&events, &request_id)
    };
    let stored = || async {
        neo4j
            .get_attention_events(&[session_id])
            .await
            .map(status)
            .map_err(AppError::Internal)
    };
    let already = || {
        AppError::Conflict(format!(
            "permission request {request_id} was already decided"
        ))
    };

    // A decision already stored wins over everything: a second click on a
    // session that has since died is still "already decided", not "gone".
    if stored().await? == PermissionStatus::Decided {
        return Err(already());
    }

    match chat_manager
        .route_permission_response(
            &session_id.to_string(),
            &request_id,
            body.allow,
            body.scope,
            true,
        )
        .await
    {
        Ok(route) => Ok(Json(serde_json::json!({ "routed": route }))),
        Err(PermissionDeliveryError::SessionDead(_)) => {
            Err(AppError::Gone(PERMISSION_GONE_REASON.to_string()))
        }
        Err(PermissionDeliveryError::NotPending) => {
            // Alive CLI, request not waiting: decided a moment ago (race with
            // the persisted decision) or never asked.
            match stored().await? {
                PermissionStatus::Unknown => Err(AppError::NotFound(format!(
                    "no permission request {request_id} on session {session_id}"
                ))),
                _ => Err(already()),
            }
        }
        Err(PermissionDeliveryError::ScopeUnsupported(scope)) => {
            Err(AppError::BadRequest(format!(
                "{}: this session cannot keep a permission for the scope '{}'",
                crate::chat::manager::PERMISSION_SCOPE_UNSUPPORTED_RPC,
                scope.as_str()
            )))
        }
        Err(PermissionDeliveryError::Failed(e)) => Err(AppError::Internal(e)),
    }
}

/// POST /api/chat/sessions/{id}/messages `{ "content": "…" }`
///
/// REST twin of the WS `user_message` frame, same routing
/// (`ChatManager::route_user_message`): local CLI → owning instance →
/// `resume_session`. A message to a dead session therefore respawns the CLI
/// and KEEPS the session's identity and links. This is also THE way to answer
/// a question (`input_response` is just a `send_message`), including an
/// orphan one.
///
/// - **200** `{ "routed": "local" | "remote" | "resumed" | "resumed_after_send_failure" }`
/// - **400** — empty `content`. **404** — unknown session.
pub async fn send_session_message(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
    claims: Option<axum::Extension<crate::auth::jwt::Claims>>,
    Json(body): Json<SendMessageRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::chat::manager::MessageDeliveryError;

    let chat_manager = state.chat_manager.as_ref().ok_or_else(|| {
        AppError::NotFound("chat_manager not configured on this server".to_string())
    })?;
    if body.content.trim().is_empty() {
        return Err(AppError::BadRequest("content must not be empty".into()));
    }
    let node = state
        .orchestrator
        .neo4j()
        .get_chat_session(session_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Session {} not found", session_id)))?;

    let sid = session_id.to_string();
    // A read-only mirror of a Claude Code sub-agent takes no message (P18c): 409
    // `mirror_read_only`, before anything is composed, extracted or routed (read from the
    // node just read: no second lookup).
    if node.mirror_of.is_some() {
        return Err(AppError::Provider(Box::new(
            crate::chat::mirror::MirrorReadOnly::failure(),
        )));
    }
    let claims = claims.map(|axum::Extension(c)| c);
    let content = crate::refs::compose::compose_user_message(
        &state.orchestrator.neo4j_arc(),
        &body.content,
        &body.refs,
        &body.attachments,
        chat_manager.refs_v1_enabled(),
    )
    .await?;

    // Same side effect as the WS path — only for a message that was accepted: a
    // refused one must not feed the graph (DISCUSSED, neural reinforcement).
    super::ws_chat_handler::spawn_entity_extraction(&state, &sid, &body.content);

    match chat_manager
        .route_user_message(&sid, &content, claims.as_ref())
        .await
    {
        Ok(route) => Ok(Json(serde_json::json!({ "routed": route }))),
        Err(MessageDeliveryError::Resume(e)) => Err(mirror_or_internal(e)),
        Err(MessageDeliveryError::SendAndResume { resume, .. }) => Err(mirror_or_internal(resume)),
        Err(MessageDeliveryError::MirrorReadOnly) => Err(AppError::Provider(Box::new(
            crate::chat::mirror::MirrorReadOnly::failure(),
        ))),
    }
}

/// A failed delivery: the typed 409 `mirror_read_only` when a mirror refused it (P18c),
/// an internal error otherwise (as before).
fn mirror_or_internal(e: anyhow::Error) -> AppError {
    if e.chain()
        .any(|c| c.is::<crate::chat::mirror::MirrorReadOnly>())
    {
        AppError::Provider(Box::new(crate::chat::mirror::MirrorReadOnly::failure()))
    } else {
        AppError::Internal(e)
    }
}

// ============================================================================
// Background tasks (T6 + T8 of plan 754a1379)
// ============================================================================

/// GET /api/chat/sessions/{id}/background-tasks — Snapshot of the
/// background subprocesses currently tracked for the session.
///
/// Returns the same `Vec<BackgroundTaskInfo>` carried by the live
/// `ChatEvent::ActiveTasksUpdate` broadcasts. The frontend hits this
/// endpoint on WebSocket connect / reconnect to re-hydrate its
/// toolbar indicator and in-progress MonitorCards before the next
/// live event arrives, avoiding a "blank toolbar" flicker.
///
/// ## Response codes
///
/// **200** — Always. An empty array is a legitimate response (no
/// background subprocesses currently tracked, or the session is
/// remote / unknown locally — the frontend treats both the same).
///
/// **404** — `chat_manager` not configured (server not started with
/// chat support).
pub async fn get_background_tasks(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let chat_manager = state.chat_manager.as_ref().ok_or_else(|| {
        AppError::NotFound("chat_manager not configured on this server".to_string())
    })?;

    let tasks = chat_manager
        .get_active_background_tasks(&session_id.to_string())
        .await;

    Ok(Json(serde_json::json!({ "tasks": tasks })))
}

/// Body of `GET /api/chat/live-activity`.
#[derive(Debug, Serialize, Deserialize)]
pub struct LiveActivityResponse {
    /// When the snapshot was taken, so a client can tell a fresh "nothing
    /// is running" from a response it failed to refresh.
    pub generated_at: String,
    /// Activity per session id. A session absent from this map is quiet:
    /// absence is an answer, not missing data.
    pub sessions: std::collections::HashMap<String, SessionActivity>,
}

/// GET /api/chat/live-activity — what every known session is doing, now.
///
/// Reads the `ChatManager`'s in-memory map only: no Neo4j, no pagination,
/// no per-session locks beyond the ones `live_session_snapshot` already
/// takes. It exists so a conversation list can *reconcile* its working
/// indicators instead of trusting that it received every CRUD event.
///
/// That distinction is the whole point. An indicator driven by events alone
/// has two failure modes a user actually hits: a list mounted after a turn
/// started shows it as idle, and a dropped "streaming stopped" event leaves
/// a "Working…" that never clears. Both are unfixable from the event stream
/// itself — only re-reading the truth fixes them.
///
/// Answers `{}` (not 404) when no chat manager is configured, so a client
/// may poll it unconditionally.
pub async fn get_live_activity(
    State(state): State<OrchestratorState>,
) -> Json<LiveActivityResponse> {
    let generated_at = chrono::Utc::now().to_rfc3339();
    let sessions = match state.chat_manager.as_ref() {
        Some(cm) => cm
            .session_activity_map()
            .await
            .into_iter()
            .map(|(id, a)| (id.to_string(), a))
            .collect(),
        None => std::collections::HashMap::new(),
    };
    Json(LiveActivityResponse {
        generated_at,
        sessions,
    })
}

/// POST /api/chat/sessions/{id}/cancel-task/{task_id} — Cancel a
/// single tracked background task by id.
///
/// Granular companion to `cancel-tools`: instead of nuking every
/// descendant of the CLI, this targets one Monitor / Bash bg by its
/// `tool_use_id` (= map key, ≡ `correlation_id` on the related
/// BackgroundOutput events).
///
/// V2 semantics (plan fc35b25e): marks the entry for removal in
/// the tracking map, broadcasts a fresh `ActiveTasksUpdate`, **and**
/// SIGINTs the subprocess subtree (root + descendants) when the
/// async PID claim has populated `task.pid`. `killed_pids` lists the
/// PIDs that received the signal. Falls back to V1 map-side-only
/// cancel when the pid is still `None` (claim race or subprocess
/// crashed before discovery) — see the doc-comment on
/// `ChatManager::cancel_task` for the full state machine.
///
/// ## Response codes
///
/// **200** with `CancelTaskResult { task_id, killed_pids, capped }` when the
/// instance holding the session acted (here or over NATS):
/// - `capped: false` → cancel applied (idempotent on an unknown or ended
///   task_id — see `ChatManager::cancel_task`).
/// - `capped: true` → rate cap hit (30/5min/session); display a
///   "slow down" toast and disable the button briefly.
///
/// Otherwise the typed bodies of `cancel_tools` (`cancel_error`); here a 504
/// `owner_timeout` IS retryable: a stopped task stays stopped.
///
/// **404** — `chat_manager` not configured.
pub async fn cancel_task(
    State(state): State<OrchestratorState>,
    Path((session_id, task_id)): Path<(Uuid, String)>,
) -> Result<Json<serde_json::Value>, AppError> {
    let chat_manager = state.chat_manager.as_ref().ok_or_else(|| {
        AppError::NotFound("chat_manager not configured on this server".to_string())
    })?;

    let result = chat_manager
        .cancel_task(&session_id.to_string(), &task_id)
        .await
        // Same refusal as cancel-tools, also when the session lives on another
        // instance (its answer comes back over NATS).
        .map_err(cancel_error)?;

    Ok(Json(serde_json::to_value(&result).unwrap_or_default()))
}

// ============================================================================
// Update session (rename)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct UpdateSessionRequest {
    pub title: Option<String>,
}

/// PATCH /api/chat/sessions/{id} — Update a session (currently: rename)
pub async fn update_session(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
    Json(request): Json<UpdateSessionRequest>,
) -> Result<Json<ChatSession>, AppError> {
    let updated = state
        .orchestrator
        .neo4j()
        .update_chat_session(session_id, None, request.title, None, None, None, None)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Session {} not found", session_id)))?;

    // Emit CRUD event for live refresh
    state.event_bus.emit(
        CrudEvent::new(
            EntityType::ChatSession,
            CrudAction::Updated,
            session_id.to_string(),
        )
        .with_payload(serde_json::json!({
            "title": updated.title,
        })),
    );

    // Field-for-field copy of `session_node_to_response`, so a rename cannot
    // start answering a different shape from the rest of the endpoints.
    let mut session = session_node_to_response(updated);
    stamp_activity(&state, std::slice::from_mut(&mut session)).await;
    Ok(Json(session))
}

// ============================================================================
// Routing of one conversation (the chat menu)
// ============================================================================

/// PUT /api/chat/sessions/{id}/routing — change how THIS conversation is routed,
/// from its next turn on.
///
/// Body `{ "auto": true }` hands the conversation back to PO (`routing_mode: full`,
/// `routed_by: auto`, a model imposed before is released). Body
/// `{ "auto": false, "routing_pool": [{ "provider", "model" }, ...] }`: one model ticked =
/// strict (`primary`, that model imposed now and never substituted, `routed_by: request`),
/// two or more = `mixed` (PO routes among them). Answers the session (`ChatSession`, with
/// `routing_mode`, `routing_pool`, `routed_by`, `model`).
///
/// Stored on the session only: the global and project routing settings are never
/// written. A conversation runs on ONE provider: per turn, only the ticked models of the
/// session's provider are candidates; a strict model on another provider, or a pool
/// with none on it, is refused (400 `routing_pool_other_provider`): moving the
/// conversation is `POST .../switch-provider`. 400 `invalid_routing_pool`: nothing
/// ticked, or an entry without provider or model. 404 unknown session. A person's
/// call: 403 for an agent session token, like a model change (WebSocket only).
pub async fn set_session_routing(
    State(state): State<OrchestratorState>,
    claims: Option<axum::Extension<crate::auth::jwt::Claims>>,
    headers: axum::http::HeaderMap,
    Path(session_id): Path<Uuid>,
    Json(body): Json<crate::chat::types::SessionRoutingRequest>,
) -> Result<Json<ChatSession>, AppError> {
    use crate::chat::envelope;
    use crate::chat::types::SessionRoutingError;
    let caller = envelope::identify_caller(
        claims.as_ref().map(|c| &c.0),
        envelope::session_header(&headers),
        state.auth_config.is_some(),
    )?;
    if matches!(caller, envelope::SpawnCaller::Agent { .. }) {
        return Err(AppError::Forbidden(
            "only a signed-in user can change how a conversation is routed".to_string(),
        ));
    }
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;
    let updated = chat_manager
        .set_session_routing(&session_id.to_string(), &body)
        .await
        .map_err(|error| match error.downcast_ref::<SessionRoutingError>() {
            Some(SessionRoutingError::NotFound) => {
                AppError::NotFound(format!("Session {session_id} not found"))
            }
            Some(refusal) => {
                AppError::Provider(Box::new(crate::chat::provider::errors::OpenFailure {
                    status: axum::http::StatusCode::BAD_REQUEST.as_u16(),
                    code: refusal.code(),
                    message: refusal.to_string(),
                    provider_id: None,
                    action: None,
                    retryable: false,
                    retry_after_ms: None,
                    resets_at_ms: None,
                    fallbacks: None,
                    consent_scope: None,
                    candidates: None,
                }))
            }
            None => AppError::Internal(error),
        })?;
    state.event_bus.emit(
        CrudEvent::new(
            EntityType::ChatSession,
            CrudAction::Updated,
            session_id.to_string(),
        )
        .with_payload(serde_json::json!({
            "routing_mode": updated.routing_mode,
            "routed_by": updated.routed_by,
        })),
    );
    let snapshot = updated
        .capabilities
        .as_deref()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok());
    let mut session = session_node_to_response(updated);
    // F-R4: the routing changed, so may what the next turn can carry.
    session.effective_capabilities = chat_manager
        .effective_capabilities(&session_id.to_string(), snapshot_images(snapshot.as_ref()))
        .await;
    stamp_activity(&state, std::slice::from_mut(&mut session)).await;
    Ok(Json(session))
}

// ============================================================================
// Search messages across sessions
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct SearchMessagesQuery {
    /// Full-text search query
    pub q: String,
    /// Optional project slug filter
    #[serde(default)]
    pub project_slug: Option<String>,
    /// Maximum number of session groups to return (default 10)
    #[serde(default = "default_search_limit")]
    pub limit: usize,
}

fn default_search_limit() -> usize {
    10
}

/// GET /api/chat/search — Search messages across all sessions
pub async fn search_messages(
    State(state): State<OrchestratorState>,
    Query(query): Query<SearchMessagesQuery>,
) -> Result<Json<Vec<MessageSearchResult>>, AppError> {
    if query.q.trim().is_empty() {
        return Err(AppError::BadRequest(
            "Search query 'q' cannot be empty".to_string(),
        ));
    }

    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    let results = chat_manager
        .search_messages(&query.q, query.limit, query.project_slug.as_deref())
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(results))
}

// ============================================================================
// Backfill
// ============================================================================

/// POST /api/chat/sessions/backfill-previews — Backfill title/preview for existing
/// sessions, and the record of the sessions of the agent engine (`agent_records`).
pub async fn backfill_previews(
    State(state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    // Phase 1: backfill from Neo4j events (fast, for sessions with stored events)
    let neo4j_count = state
        .orchestrator
        .neo4j()
        .backfill_chat_session_previews()
        .await
        .map_err(AppError::Internal)?;

    // Phase 2: backfill from Meilisearch (for older sessions without Neo4j events)
    let meili_count = if let Some(chat_manager) = &state.chat_manager {
        chat_manager
            .backfill_previews_from_meilisearch()
            .await
            .unwrap_or(0)
    } else {
        0
    };

    // Phase 3: the record (message count, cost, title) of the sessions the agent
    // engine served before it kept one, from their persisted events.
    let agent_count = if let Some(chat_manager) = &state.chat_manager {
        chat_manager
            .backfill_agent_session_records()
            .await
            .map_err(AppError::Internal)?
    } else {
        0
    };

    let total = neo4j_count + meili_count;
    Ok(Json(serde_json::json!({
        "updated": total,
        "from_neo4j": neo4j_count,
        "from_meilisearch": meili_count,
        "agent_records": agent_count,
        "message": format!("Backfilled title/preview for {} sessions", total)
    })))
}

// ============================================================================
// Permission config (runtime GET/PUT)
// ============================================================================

/// Response for GET /api/chat/config/permissions.
/// Extends PermissionConfig with the default model from config.yaml
/// so the frontend knows which model to pre-select for new conversations.
#[derive(Debug, Serialize)]
pub struct PermissionConfigResponse {
    /// Permission mode: "default", "acceptEdits", "plan", "bypassPermissions"
    pub mode: String,
    /// Tool patterns to explicitly allow
    pub allowed_tools: Vec<String>,
    /// Tool patterns to explicitly disallow
    pub disallowed_tools: Vec<String>,
    /// Default model from config.yaml (used by frontend for new conversation pre-selection)
    pub default_model: String,
}

/// GET /api/chat/config/permissions — Return current runtime permission config
pub async fn get_chat_permissions(
    State(state): State<OrchestratorState>,
) -> Result<Json<PermissionConfigResponse>, AppError> {
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    let config = chat_manager.get_permission_config().await;
    let default_model = chat_manager.config.default_model.clone();

    Ok(Json(PermissionConfigResponse {
        mode: config.mode,
        allowed_tools: config.allowed_tools,
        disallowed_tools: config.disallowed_tools,
        default_model,
    }))
}

/// PUT /api/chat/config/permissions — Update runtime permission config
///
/// Accepts a JSON body with `mode`, `allowed_tools`, and `disallowed_tools`.
/// Validates the mode string before applying. Returns the updated config.
pub async fn update_chat_permissions(
    State(state): State<OrchestratorState>,
    Json(new_config): Json<crate::chat::PermissionConfig>,
) -> Result<Json<crate::chat::PermissionConfig>, AppError> {
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    let updated = chat_manager
        .update_permission_config(new_config)
        .await
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    Ok(Json(updated))
}

// ============================================================================
// Chat environment config (PATH, CLI path, auto-update)
// ============================================================================

/// Response for GET /api/chat/config
#[derive(Debug, Serialize)]
pub struct ChatConfigResponse {
    pub mode: String,
    pub allowed_tools: Vec<String>,
    pub disallowed_tools: Vec<String>,
    pub default_model: String,
    pub process_path: Option<String>,
    pub claude_cli_path: Option<String>,
    pub auto_update_cli: bool,
    pub auto_update_app: bool,
}

/// GET /api/chat/config — Return full chat configuration
pub async fn get_chat_config(
    State(state): State<OrchestratorState>,
) -> Result<Json<ChatConfigResponse>, AppError> {
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    let perm = chat_manager.get_permission_config().await;
    let env = chat_manager.get_env_config().await;

    Ok(Json(ChatConfigResponse {
        mode: perm.mode,
        allowed_tools: perm.allowed_tools,
        disallowed_tools: perm.disallowed_tools,
        default_model: chat_manager.config.default_model.clone(),
        process_path: env.process_path,
        claude_cli_path: env.claude_cli_path,
        auto_update_cli: env.auto_update_cli,
        auto_update_app: env.auto_update_app,
    }))
}

/// GET /api/chat/models — Live Claude model catalog for the model selector.
///
/// Backed by a stale-while-revalidate cache (see `chat::model_catalog`):
/// returns instantly from cache, refreshing in the background at most every
/// 12h when an Anthropic API key is configured. Falls back to a small
/// static list when no key is configured or the live fetch fails — this
/// endpoint never errors on the caller.
pub async fn get_model_catalog(
    State(state): State<OrchestratorState>,
) -> Json<Vec<crate::chat::model_catalog::ModelDefinition>> {
    Json(state.model_catalog.get_models().await)
}

/// Request body for PATCH /api/chat/config
#[derive(Debug, Deserialize)]
pub struct UpdateChatConfigRequest {
    /// Permission mode override
    pub mode: Option<String>,
    /// Allowed tool patterns override
    pub allowed_tools: Option<Vec<String>>,
    /// Disallowed tool patterns override
    pub disallowed_tools: Option<Vec<String>>,
    /// Process PATH (empty string = clear, non-empty = set)
    pub process_path: Option<String>,
    /// Claude CLI path (empty string = clear, non-empty = set)
    pub claude_cli_path: Option<String>,
    /// Auto-update CLI toggle
    pub auto_update_cli: Option<bool>,
    /// Auto-update application toggle
    pub auto_update_app: Option<bool>,
}

/// PATCH /api/chat/config — Update chat configuration (partial merge).
/// Persists changes to config.yaml.
pub async fn update_chat_config(
    State(state): State<OrchestratorState>,
    Json(body): Json<UpdateChatConfigRequest>,
) -> Result<Json<ChatConfigResponse>, AppError> {
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    // Update permission fields if provided
    if body.mode.is_some() || body.allowed_tools.is_some() || body.disallowed_tools.is_some() {
        let current = chat_manager.get_permission_config().await;
        let new_perm = crate::chat::PermissionConfig {
            mode: body.mode.unwrap_or(current.mode),
            allowed_tools: body.allowed_tools.unwrap_or(current.allowed_tools),
            disallowed_tools: body.disallowed_tools.unwrap_or(current.disallowed_tools),
        };
        // Validate mode
        if !crate::chat::config::PermissionConfig::is_valid_mode(&new_perm.mode) {
            return Err(AppError::BadRequest(format!(
                "Invalid permission mode '{}'. Valid modes: {:?}",
                new_perm.mode,
                crate::chat::config::PermissionConfig::valid_modes()
            )));
        }
        chat_manager
            .update_permission_config(new_perm)
            .await
            .map_err(|e| AppError::BadRequest(e.to_string()))?;
    }

    // Update environment config fields
    // Empty string means "clear" (set to None), non-empty means "set"
    if let Some(ref path_val) = body.process_path {
        let new_val = if path_val.is_empty() {
            None
        } else {
            Some(path_val.clone())
        };
        chat_manager.update_process_path(new_val).await;
    }
    if let Some(ref cli_val) = body.claude_cli_path {
        let new_val = if cli_val.is_empty() {
            None
        } else {
            Some(cli_val.clone())
        };
        chat_manager.update_claude_cli_path(new_val).await;
    }
    if let Some(auto_update) = body.auto_update_cli {
        chat_manager.update_auto_update_cli(auto_update).await;
    }
    if let Some(auto_update_app) = body.auto_update_app {
        chat_manager.update_auto_update_app(auto_update_app).await;
        // The update service reads its own flag: without this the toggle was
        // persisted and echoed back but only took effect after a restart.
        crate::update::service::apply_auto_update(auto_update_app);
    }

    // Persist to config.yaml if path is known
    chat_manager
        .persist_chat_config_to_yaml()
        .await
        .map_err(|e| {
            tracing::warn!("Failed to persist chat config to YAML: {}", e);
            AppError::Internal(anyhow::anyhow!("Failed to persist config: {}", e))
        })?;

    // Return updated config
    let perm = chat_manager.get_permission_config().await;
    let env = chat_manager.get_env_config().await;
    Ok(Json(ChatConfigResponse {
        mode: perm.mode,
        allowed_tools: perm.allowed_tools,
        disallowed_tools: perm.disallowed_tools,
        default_model: chat_manager.config.default_model.clone(),
        process_path: env.process_path,
        claude_cli_path: env.claude_cli_path,
        auto_update_cli: env.auto_update_cli,
        auto_update_app: env.auto_update_app,
    }))
}

/// GET /api/chat/detect-path — Detect user's PATH from login shell
pub async fn detect_path() -> Json<serde_json::Value> {
    match crate::chat::path_detect::detect_user_path().await {
        Some(path) => Json(serde_json::json!({ "path": path })),
        None => Json(serde_json::json!({
            "path": null,
            "error": "Could not detect PATH from login shell"
        })),
    }
}

// ============================================================================
// CLI version management (check + install/upgrade)
// ============================================================================

/// GET /api/chat/cli/auth-status — Check the Claude CLI authentication status.
///
/// Returns whether the user is logged in, the auth method, and email.
pub async fn get_cli_auth_status() -> Json<crate::chat::cli_auth::CliAuthStatus> {
    Json(crate::chat::cli_auth::check_cli_auth_status().await)
}

/// GET /api/chat/cli/status — Check installed CLI version and update availability.
///
/// Returns the full CLI version status including installed version, latest
/// npm version, whether an update is available, and whether it's a local build.
/// The npm version check has a 10s timeout built-in — will return `latest_version: null`
/// if npm is unreachable.
pub async fn get_cli_status() -> Json<crate::chat::cli_version::CliVersionStatus> {
    Json(crate::chat::cli_version::check_cli_status().await)
}

/// Query of `GET /api/chat/providers`.
#[derive(Debug, Deserialize)]
pub struct ListProvidersQuery {
    /// When given, `allowed_for_project` is answered for this project.
    pub project_slug: Option<String>,
}

/// GET /api/chat/providers — the provider instances, their health and the
/// capabilities of each model, before any session exists (A42).
///
/// Never answers a secret: an instance carries a credential reference and the
/// origin of its endpoint, nothing more.
pub async fn list_providers(
    State(state): State<OrchestratorState>,
    Query(query): Query<ListProvidersQuery>,
) -> Result<Json<crate::chat::provider::listing::ProviderListing>, AppError> {
    use crate::chat::provider::listing::{self, HealthEntry, ModelEntry};
    use nexus_claude::agent::AgentProvider;
    use nexus_claude::providers::claude_code::{ClaudeCodeConfig, ClaudeCodeProvider};

    let default_model = state
        .chat_manager
        .as_ref()
        .map(|m| m.resolve_model(None))
        .unwrap_or_else(|| crate::chat::ChatConfig::from_env().default_model);

    // The CLI the server is set to use (`CLAUDE_CLI_PATH`), as a session would.
    let mut claude_config = ClaudeCodeConfig::default();
    if let Some(manager) = state.chat_manager.as_ref() {
        claude_config.cli_path = manager
            .get_env_config()
            .await
            .claude_cli_path
            .filter(|p| !p.trim().is_empty())
            .map(std::path::PathBuf::from);
    }
    let provider = ClaudeCodeProvider::new(claude_config);
    let health = HealthEntry::from_nexus(&provider.health().await);
    let mut models: Vec<ModelEntry> = provider
        .catalog()
        .await
        .unwrap_or_default()
        .iter()
        .map(|m| {
            ModelEntry::new(
                m.id.clone(),
                m.is_default.then(|| "default".to_string()),
                &provider.capabilities(Some(&m.id)),
            )
        })
        .collect();
    if models.is_empty() {
        models.push(ModelEntry::new(
            default_model.clone(),
            Some("default".to_string()),
            &provider.capabilities(Some(&default_model)),
        ));
    }

    // `images` is what nexus declares for Claude Code (true since the façade
    // writes image blocks, A12 revised), the same on both engines: the legacy
    // engine no longer needs a forced value, the agent engine sends them inline.
    let mut entries = vec![listing::builtin_claude_code(
        health, models, // Claude Code needs no consent, in a project or outside one.
        true,
    )];
    // The stored instances, with the consent of the asked project.
    entries.extend(
        super::provider_handlers::stored_entries(
            state.orchestrator.neo4j(),
            query.project_slug.as_deref(),
        )
        .await?,
    );
    // The pilot's configured target is the default when it names an instance.
    let configured = state
        .orchestrator
        .neo4j()
        .get_llm_setting(
            crate::chat::provider::settings::GLOBAL,
            crate::chat::provider::settings::ROLES_KEY,
        )
        .await
        .ok()
        .flatten()
        .and_then(|raw| {
            serde_json::from_str::<crate::chat::provider::settings::RoleAssignments>(&raw).ok()
        })
        .and_then(|roles| roles.pilot.map(|p| p.provider));
    // Without the Claude CLI, what a chat naming nothing would open (B2).
    let usable_here = super::provider_handlers::allowed_here(
        state.orchestrator.neo4j(),
        query.project_slug.as_deref(),
    )
    .await?;
    let claude_cli = match state.chat_manager.as_ref() {
        Some(manager) => manager.claude_code_usable().await,
        None => entries
            .first()
            .is_some_and(|p| p.health.state != "cli_not_found"),
    };
    let mut body = listing::assemble(entries, configured.as_deref(), claude_cli, &usable_here);
    // The routing mode in force for the asked project (R2): who chooses the
    // provider, and how far the learnt choices are trusted. Shown, not applied.
    let (routing, scope) = crate::chat::provider::cognitive::load_routing(
        state.orchestrator.neo4j(),
        query.project_slug.as_deref(),
    )
    .await
    .map_err(AppError::Internal)?;
    body.routing = Some(listing::RoutingSummary::new(&routing, scope));
    Ok(Json(body))
}

/// Request body for POST /api/chat/cli/install
#[derive(Debug, Deserialize)]
pub struct InstallCliRequest {
    /// Target version to install (e.g., "2.6.0"). None means "latest".
    pub version: Option<String>,
}

/// POST /api/chat/cli/install — Download/upgrade the Claude Code CLI.
///
/// Always returns 200 with `success: true/false` — the frontend displays the message.
/// No 500 for installation failures (they are expected user-facing scenarios).
pub async fn install_cli(
    State(state): State<OrchestratorState>,
    Json(body): Json<InstallCliRequest>,
) -> Json<crate::chat::cli_version::CliInstallResult> {
    let result = crate::chat::cli_version::install_or_upgrade_cli(body.version.as_deref()).await;
    // A CLI that just appeared becomes the default again at the next opening (B2).
    if let Some(manager) = state.chat_manager.as_ref() {
        manager.forget_claude_cli_detection();
    }
    // ...and gets the MCP integration the boot skipped without it (review of #722).
    if result.success {
        let setup_config = crate::setup_claude::SetupConfig {
            mcp_server_path: crate::chat::ChatConfig::detect_mcp_server_path_public(),
            server_port: state.server_port,
            jwt_secret: state.auth_config.as_ref().map(|a| a.jwt_secret.clone()),
        };
        match tokio::task::spawn_blocking(move || {
            crate::setup_claude::setup_claude_code(&setup_config)
        })
        .await
        {
            Ok(Ok(outcome)) => tracing::info!(?outcome, "Claude Code MCP configured after install"),
            Ok(Err(error)) => {
                tracing::warn!(%error, "Claude Code MCP setup after install failed (non-fatal)")
            }
            Err(error) => {
                tracing::warn!(%error, "Claude Code MCP setup after install did not finish")
            }
        }
    }
    Json(result)
}

// ============================================================================
// DISCUSSED relations (ChatSession → Entity)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AddDiscussedRequest {
    /// List of entities: each is `{ entity_type, entity_id }`
    pub entities: Vec<DiscussedEntityInput>,
}

#[derive(Debug, Deserialize)]
pub struct DiscussedEntityInput {
    /// Entity type: "File", "Function", "Struct", "Trait", "Enum"
    pub entity_type: String,
    /// Entity identifier: file path (for File) or symbol name
    pub entity_id: String,
}

/// POST /api/chat/sessions/{id}/discussed — Add DISCUSSED relations
pub async fn add_discussed(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
    Json(body): Json<AddDiscussedRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    if body.entities.is_empty() {
        return Ok(Json(serde_json::json!({ "created": 0 })));
    }

    let entities: Vec<(String, String)> = body
        .entities
        .into_iter()
        .map(|e| (e.entity_type, e.entity_id))
        .collect();

    let created = state
        .orchestrator
        .neo4j()
        .add_discussed(session_id, &entities)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({ "created": created })))
}

/// GET /api/chat/sessions/{id}/discussed — Get entities discussed in a session
pub async fn get_session_entities(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
    Query(params): Query<SessionEntitiesQuery>,
) -> Result<Json<Vec<crate::neo4j::models::DiscussedEntity>>, AppError> {
    let project_id = params
        .project_id
        .as_ref()
        .and_then(|s| s.parse::<Uuid>().ok());

    let entities = state
        .orchestrator
        .neo4j()
        .get_session_entities(session_id, project_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(entities))
}

#[derive(Debug, Deserialize)]
pub struct SessionEntitiesQuery {
    /// Optional project_id to scope results (security: no cross-project leaks)
    #[serde(default)]
    pub project_id: Option<String>,
}

// ============================================================================
// Plan ↔ Session linking
// ============================================================================

/// GET /api/plans/{id}/sessions — Get all chat sessions linked to a plan
pub async fn get_plan_sessions(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<Vec<crate::neo4j::models::SessionWithLinks>>, AppError> {
    let sessions = state
        .orchestrator
        .neo4j()
        .get_sessions_for_plan(plan_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(sessions))
}

/// GET /api/tasks/{id}/sessions — Get all chat sessions linked to a task
pub async fn get_task_sessions(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
) -> Result<Json<Vec<crate::neo4j::models::SessionWithLinks>>, AppError> {
    let sessions = state
        .orchestrator
        .neo4j()
        .get_sessions_for_task(task_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(sessions))
}

#[derive(Debug, Deserialize)]
pub struct AssociateSessionRequest {
    pub entity_type: String,
    pub entity_id: Uuid,
    #[serde(default = "default_associate_source")]
    pub source: String,
}

fn default_associate_source() -> String {
    "manual".to_string()
}

/// POST /api/chat/sessions/{id}/associate — Create ASSOCIATED_WITH relation
pub async fn associate_session(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
    Json(body): Json<AssociateSessionRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    // Validate entity_type
    match body.entity_type.as_str() {
        "Plan" | "Task" => {}
        other => {
            return Err(AppError::BadRequest(format!(
                "Invalid entity_type '{}', must be 'Plan' or 'Task'",
                other
            )));
        }
    }

    let created = state
        .orchestrator
        .neo4j()
        .create_associated_with(session_id, &body.entity_type, body.entity_id, &body.source)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "created": created,
        "session_id": session_id,
        "entity_type": body.entity_type,
        "entity_id": body.entity_id,
        "source": body.source
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_bearer_token, test_chat_session};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;

    /// Create an authenticated GET request for a given URI
    fn auth_get(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    /// Create an authenticated DELETE request for a given URI
    fn auth_delete(uri: &str) -> Request<Body> {
        Request::builder()
            .method("DELETE")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    /// Create an authenticated POST request with JSON body
    fn auth_post(uri: &str, body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .header("authorization", test_bearer_token())
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    /// Build an OrchestratorState with mock backends (no ChatManager)
    async fn mock_server_state() -> OrchestratorState {
        mock_server_state_with(None).await
    }

    /// [`mock_server_state`] with a chat manager whose answer to "is the Claude Code CLI on
    /// this server?" is fixed (`Some(present)`): the listing's default does not depend on
    /// the machine the tests run on.
    async fn mock_server_state_with(claude_cli: Option<bool>) -> OrchestratorState {
        let app_state = mock_app_state();
        let chat_manager = claude_cli.map(|present| {
            Arc::new(
                crate::chat::ChatManager::new_without_memory(
                    app_state.neo4j.clone(),
                    app_state.meili.clone(),
                    crate::chat::ChatConfig::default(),
                )
                .with_claude_cli_present(present),
            )
        });
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            // A root account is configured; the usual test token is NOT root.
            auth_config: Some(crate::AuthConfig {
                root_account: Some(crate::RootAccountConfig {
                    email: ROOT_EMAIL.to_string(),
                    name: "Root".to_string(),
                    password_hash: "unused".to_string(),
                }),
                ..crate::test_helpers::test_auth_config()
            }),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            remote_mcp: crate::RemoteMcpConfig::default(),
            ingress: Default::default(),
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
            model_catalog: crate::chat::model_catalog::ModelCatalogCache::new(None),
            vault: crate::vault::VaultService::ephemeral(),
        })
    }

    /// Build a test router with mock state
    async fn test_app() -> axum::Router {
        let state = mock_server_state().await;
        create_router(state)
    }

    /// Build a test router with pre-seeded sessions
    async fn test_app_with_sessions(
        sessions: &[crate::neo4j::models::ChatSessionNode],
    ) -> axum::Router {
        let app_state = mock_app_state();
        for s in sessions {
            app_state.neo4j.create_chat_session(s).await.unwrap();
        }
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            // A root account is configured; the usual test token is NOT root.
            auth_config: Some(crate::AuthConfig {
                root_account: Some(crate::RootAccountConfig {
                    email: ROOT_EMAIL.to_string(),
                    name: "Root".to_string(),
                    password_hash: "unused".to_string(),
                }),
                ..crate::test_helpers::test_auth_config()
            }),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            remote_mcp: crate::RemoteMcpConfig::default(),
            ingress: Default::default(),
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
            model_catalog: crate::chat::model_catalog::ModelCatalogCache::new(None),
            vault: crate::vault::VaultService::ephemeral(),
        });
        create_router(state)
    }

    // ====================================================================
    // SessionsListQuery serde
    // ====================================================================

    #[test]
    fn test_sessions_list_query_defaults() {
        let json = r#"{}"#;
        let query: SessionsListQuery = serde_json::from_str(json).unwrap();
        assert!(query.project_slug.is_none());
    }

    #[test]
    fn test_sessions_list_query_with_project() {
        let json = r#"{"project_slug": "my-project"}"#;
        let query: SessionsListQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.project_slug.as_deref(), Some("my-project"));
    }

    // ====================================================================
    // GET /api/chat/sessions — list
    // ====================================================================

    #[tokio::test]
    async fn test_list_sessions_empty() {
        let app = test_app().await;
        let resp = app.oneshot(auth_get("/api/chat/sessions")).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["total"], 0);
        assert_eq!(json["items"].as_array().unwrap().len(), 0);
    }

    // ====================================================================
    // GET /api/chat/providers
    // ====================================================================

    #[tokio::test]
    async fn providers_lists_the_builtin_instance_with_the_documented_fields() {
        // The CLI is declared present: the default does not depend on the machine.
        let app = create_router(mock_server_state_with(Some(true)).await);
        let resp = app
            .oneshot(auth_get("/api/chat/providers?project_slug=p"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["default_provider"], "claude-code");
        let p = &json["providers"][0];
        assert_eq!(p["id"], "claude-code");
        assert_eq!(p["kind"], "claude_code");
        assert_eq!(p["builtin"], true);
        assert_eq!(p["allowed_for_project"], true);
        assert_eq!(p["credential"], "none");
        assert!(p["health"]["state"].is_string());
        assert!(p["models"][0]["capabilities"].is_object());
    }

    #[tokio::test]
    async fn without_the_claude_cli_the_listing_has_no_default_until_an_instance_is_allowed() {
        // B2 on the listing: CLI absent → claude-code is listed (unhealthy) but is not the
        // default; nothing else usable → null; the only allowed instance → that instance.
        let app = create_router(mock_server_state_with(Some(false)).await);
        let (status, json) = call_json(&app, auth_get("/api/chat/providers?project_slug=p")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(json["default_provider"].is_null(), "{json}");
        assert_eq!(json["providers"][0]["id"], "claude-code");
        assert_eq!(json["providers"][0]["is_default"], false);

        call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers",
                deepseek("https://8.8.8.8/v1"),
            ),
        )
        .await;
        let (_, json) = call_json(&app, auth_get("/api/chat/providers?project_slug=p")).await;
        assert!(
            json["default_provider"].is_null(),
            "not allowed yet: {json}"
        );
        let (status, _) = call_json(
            &app,
            auth_json(
                "PUT",
                "/api/projects/p/llm-consents/deepseek",
                serde_json::json!({"origin": "https://8.8.8.8"}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (_, json) = call_json(&app, auth_get("/api/chat/providers?project_slug=p")).await;
        assert_eq!(json["default_provider"], "deepseek", "{json}");
        // Another project has not allowed it: still no default there.
        let (_, json) = call_json(&app, auth_get("/api/chat/providers?project_slug=q")).await;
        assert!(json["default_provider"].is_null(), "{json}");
    }

    /// The listing no longer forces `images` for Claude Code: nexus declares it,
    /// for every model of the catalogue, whatever engine serves the session.
    #[tokio::test]
    async fn nexus_declares_images_for_claude_code_so_the_listing_forces_nothing() {
        use nexus_claude::agent::AgentProvider;
        use nexus_claude::providers::claude_code::{ClaudeCodeConfig, ClaudeCodeProvider};
        let provider = ClaudeCodeProvider::new(ClaudeCodeConfig::default());
        assert!(provider.capabilities(None).images);
        for m in provider.catalog().await.unwrap_or_default() {
            assert!(provider.capabilities(Some(&m.id)).images, "{}", m.id);
        }
    }

    #[tokio::test]
    async fn providers_reports_images_for_claude_code_on_the_legacy_engine() {
        let app = test_app().await;
        let resp = app.oneshot(auth_get("/api/chat/providers")).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["providers"][0]["models"][0]["capabilities"]["images"],
            true
        );
    }

    // ====================================================================
    // Provider settings: instances, consent, roles, aliases, policy
    // ====================================================================

    const ROOT_EMAIL: &str = "root@ffs.holdings";

    /// A JSON request signed as the root account of `mock_server_state`.
    fn root_json(method: &str, uri: &str, body: serde_json::Value) -> Request<Body> {
        let token = crate::auth::jwt::encode_jwt(
            Uuid::new_v5(&Uuid::NAMESPACE_URL, ROOT_EMAIL.as_bytes()),
            ROOT_EMAIL,
            "Root",
            "test-secret-key-minimum-32-chars!!",
            3600,
        )
        .unwrap();
        Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn auth_json(method: &str, uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .header("authorization", test_bearer_token())
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn call_json(app: &axum::Router, req: Request<Body>) -> (StatusCode, serde_json::Value) {
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    fn deepseek(url: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "deepseek", "kind": "openai_compatible", "label": "DeepSeek",
            "base_url": url, "default_model": "deepseek-chat",
            "cost_source": "priced", "credential_ref": "vault:deepseek"
        })
    }

    #[tokio::test]
    async fn an_instance_is_created_listed_and_never_carries_a_secret() {
        let app = test_app().await;
        let (status, body) = call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers",
                deepseek("https://8.8.8.8/v1"),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{body}");
        assert_eq!(body["origin"], "https://8.8.8.8");
        assert_eq!(body["credential_ref"], "vault:deepseek");

        let (status, _) = call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers",
                deepseek("https://8.8.8.8/v1"),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);

        let (_, listing) = call_json(&app, auth_get("/api/chat/providers?project_slug=p")).await;
        let providers = listing["providers"].as_array().unwrap();
        assert_eq!(providers.len(), 2);
        let entry = providers.iter().find(|p| p["id"] == "deepseek").unwrap();
        assert_eq!(entry["endpoint_origin"], "https://8.8.8.8");
        assert_eq!(entry["credential"], "vault:deepseek");
        assert_eq!(entry["allowed_for_project"], false);
        assert_eq!(entry["builtin"], false);
    }

    #[tokio::test]
    async fn a_secret_in_a_body_or_a_forbidden_endpoint_is_refused() {
        let app = test_app().await;
        let mut with_key = deepseek("https://8.8.8.8/v1");
        with_key["api_key"] = serde_json::json!("sk-live-123");
        let (status, _) = call_json(&app, auth_json("POST", "/api/chat/providers", with_key)).await;
        assert!(status.is_client_error(), "{status}");

        let mut bare = deepseek("https://8.8.8.8/v1");
        bare["credential_ref"] = serde_json::json!("sk-live-123");
        let (status, body) = call_json(&app, auth_json("POST", "/api/chat/providers", bare)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(!body.to_string().contains("sk-live"), "{body}");

        for url in [
            "http://8.8.8.8/v1",
            "https://169.254.169.254/latest",
            "https://10.0.0.5/v1",
            "https://u:p@8.8.8.8/v1",
        ] {
            let (status, _) = call_json(
                &app,
                auth_json("POST", "/api/chat/providers", deepseek(url)),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{url}");
        }
        let mut builtin = deepseek("https://8.8.8.8/v1");
        builtin["id"] = serde_json::json!("claude-code");
        let (status, _) = call_json(&app, auth_json("POST", "/api/chat/providers", builtin)).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        let (status, _) = call_json(
            &app,
            auth_json(
                "DELETE",
                "/api/chat/providers/claude-code",
                serde_json::json!({}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn get_provider_returns_the_full_instance_to_a_human() {
        let app = test_app().await;
        let (status, _) = call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers",
                deepseek("https://8.8.8.8/v1"),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);

        // A human gets the detail the edit form pre-fills, path included.
        let (status, body) = call_json(&app, auth_get("/api/chat/providers/deepseek")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["id"], "deepseek");
        assert_eq!(body["base_url"], "https://8.8.8.8/v1");
        assert_eq!(body["origin"], "https://8.8.8.8");
        assert_eq!(body["default_model"], "deepseek-chat");
        assert_eq!(body["cost_source"], "priced");
        assert_eq!(body["credential_ref"], "vault:deepseek");
        assert_eq!(body["builtin"], false);
        assert!(body.get("preset").is_some(), "{body}");

        // Unknown instance: 404. The built-in one is refused like update/delete.
        let (status, _) = call_json(&app, auth_get("/api/chat/providers/nope")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = call_json(&app, auth_get("/api/chat/providers/claude-code")).await;
        assert_eq!(status, StatusCode::FORBIDDEN);

        // The listing still never carries a URL or a path.
        let (status, listing) = call_json(&app, auth_get("/api/chat/providers")).await;
        assert_eq!(status, StatusCode::OK);
        let text = listing.to_string();
        assert!(!text.contains("base_url"), "{text}");
        assert!(!text.contains("/v1"), "{text}");
        let entry = listing["providers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["id"] == "deepseek")
            .unwrap();
        assert_eq!(entry["endpoint_origin"], "https://8.8.8.8");
    }

    // ── GET /api/chat/providers/{id}/models ─────────────────────────────────

    /// An instance stored straight into the mock graph. The API refuses a private
    /// address on creation, which is not what these tests are about: they point the
    /// instance at a local fake endpoint.
    async fn models_app(kind: &str, base_url: &str, default_model: Option<&str>) -> axum::Router {
        use crate::chat::provider::settings::{InstanceRecord, GLOBAL, INSTANCE_PREFIX};
        let state = mock_server_state().await;
        let record = InstanceRecord {
            id: "local".into(),
            kind: kind.into(),
            preset: (kind == "openai_compatible").then(|| "llama_server".to_string()),
            label: "Local".into(),
            base_url: base_url.into(),
            origin: "http://127.0.0.1".into(),
            default_model: default_model.map(str::to_string),
            cost_source: "free".into(),
            credential_ref: "none".into(),
            ..Default::default()
        };
        state
            .orchestrator
            .neo4j_arc()
            .put_llm_setting(
                GLOBAL,
                &format!("{INSTANCE_PREFIX}local"),
                &serde_json::to_string(&record).unwrap(),
            )
            .await
            .unwrap();
        create_router(state)
    }

    /// A fake endpoint answering `GET /v1/models` with `status` and `body`.
    async fn fake_models_endpoint(body: serde_json::Value, status: u16) -> wiremock::MockServer {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .mount(&server)
            .await;
        server
    }

    fn model_ids(body: &serde_json::Value) -> Vec<String> {
        body.as_array()
            .expect("an array of models")
            .iter()
            .map(|m| m["id"].as_str().expect("an id").to_string())
            .collect()
    }

    #[tokio::test]
    async fn the_models_of_an_openai_compatible_instance_are_the_ones_its_endpoint_lists() {
        let server = fake_models_endpoint(
            serde_json::json!({"object": "list", "data": [{"id": "b"}, {"id": "a"}, {"id": "c"}, {"id": "b"}]}),
            200,
        )
        .await;
        let app = models_app(
            "openai_compatible",
            &format!("{}/v1", server.uri()),
            Some("a"),
        )
        .await;
        let (status, body) = call_json(&app, auth_get("/api/chat/providers/local/models")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        // The stored default first, the endpoint's order after it, duplicates collapsed.
        assert_eq!(model_ids(&body), ["a", "b", "c"]);
    }

    #[tokio::test]
    async fn an_instance_saved_without_a_default_still_gets_a_picker() {
        let server = fake_models_endpoint(
            serde_json::json!({"object": "list", "data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]}),
            200,
        )
        .await;
        let app = models_app("openai_compatible", &format!("{}/v1", server.uri()), None).await;
        let (status, body) = call_json(&app, auth_get("/api/chat/providers/local/models")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(model_ids(&body), ["deepseek-chat", "deepseek-reasoner"]);
    }

    #[tokio::test]
    async fn a_default_the_endpoint_does_not_list_is_kept_first() {
        let server = fake_models_endpoint(
            serde_json::json!({"object": "list", "data": [{"id": "a"}]}),
            200,
        )
        .await;
        let app = models_app(
            "openai_compatible",
            &format!("{}/v1", server.uri()),
            Some("z"),
        )
        .await;
        let (_, body) = call_json(&app, auth_get("/api/chat/providers/local/models")).await;
        assert_eq!(model_ids(&body), ["z", "a"]);
    }

    #[tokio::test]
    async fn a_refusing_endpoint_falls_back_to_the_stored_default() {
        let server = fake_models_endpoint(serde_json::json!({"error": "nope"}), 500).await;
        let app = models_app(
            "openai_compatible",
            &format!("{}/v1", server.uri()),
            Some("kept"),
        )
        .await;
        let (status, body) = call_json(&app, auth_get("/api/chat/providers/local/models")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(model_ids(&body), ["kept"]);
    }

    #[tokio::test]
    async fn an_unreachable_endpoint_or_an_empty_list_falls_back_to_the_stored_default() {
        // Nothing listens on this port: the listing fails, the picker is not left empty.
        let app = models_app("openai_compatible", "http://127.0.0.1:9/v1", Some("kept")).await;
        let (status, body) = call_json(&app, auth_get("/api/chat/providers/local/models")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(model_ids(&body), ["kept"]);

        let server =
            fake_models_endpoint(serde_json::json!({"object": "list", "data": []}), 200).await;
        let app = models_app(
            "openai_compatible",
            &format!("{}/v1", server.uri()),
            Some("kept"),
        )
        .await;
        let (_, body) = call_json(&app, auth_get("/api/chat/providers/local/models")).await;
        assert_eq!(model_ids(&body), ["kept"]);
    }

    #[tokio::test]
    async fn a_process_instance_answers_with_its_stored_default_without_any_network() {
        // codex chooses its own models: no endpoint is asked (this one would refuse).
        let app = models_app("codex", "", Some("gpt-x")).await;
        let (status, body) = call_json(&app, auth_get("/api/chat/providers/local/models")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(model_ids(&body), ["gpt-x"]);
        // And an unknown instance is a 404, not an empty list.
        let (status, _) = call_json(&app, auth_get("/api/chat/providers/nope/models")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_provider_is_refused_to_an_agent_token() {
        use crate::api::provider_handlers::get_provider;
        use axum::extract::{Path, State};
        use axum::Extension;

        let state = mock_server_state().await;
        let record = serde_json::json!({
            "id": "deepseek", "kind": "openai_compatible", "preset": "deepseek",
            "label": "DeepSeek", "base_url": "https://api.deepseek.com/v1",
            "origin": "https://api.deepseek.com", "default_model": "deepseek-chat",
            "cost_source": "priced", "credential_ref": "vault:deepseek"
        });
        state
            .orchestrator
            .neo4j_arc()
            .put_llm_setting("global", "instance:deepseek", &record.to_string())
            .await
            .unwrap();

        // An agent_session token (unbound: the claims are what the handler reads).
        let human = crate::auth::jwt::Claims::service_account("agent");
        let (token, _) = crate::auth::jwt::generate_session_token(
            &human,
            None,
            &crate::test_helpers::test_auth_config().jwt_secret,
            600,
        )
        .unwrap();
        let agent = crate::auth::jwt::decode_jwt(
            &token,
            &crate::test_helpers::test_auth_config().jwt_secret,
        )
        .unwrap();
        assert!(!agent.is_human());
        let err = get_provider(
            State(state.clone()),
            Extension(agent),
            Path("deepseek".into()),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AppError::Forbidden(_)), "{err:?}");

        // The same record, read by a person, carries the full URL and the preset.
        let person = crate::auth::jwt::Claims {
            token_type: None,
            ..crate::auth::jwt::Claims::service_account("person")
        };
        let axum::Json(view) =
            get_provider(State(state), Extension(person), Path("deepseek".into()))
                .await
                .unwrap();
        assert_eq!(view["base_url"], "https://api.deepseek.com/v1");
        assert_eq!(view["preset"], "deepseek");
        assert_eq!(view["credential_ref"], "vault:deepseek");
    }

    #[tokio::test]
    async fn a_codex_instance_is_created_behind_the_gate_with_a_process_identity() {
        let app = test_app().await;
        let body = serde_json::json!({"id": "codex", "kind": "codex", "label": "Codex"});
        let (status, resp) =
            call_json(&app, auth_json("POST", "/api/chat/providers", body.clone())).await;
        assert_eq!(status, StatusCode::CREATED, "{resp}");
        assert_eq!(resp["origin"], "process:codex");
        let (_, listing) = call_json(&app, auth_get("/api/chat/providers?project_slug=p")).await;
        let entry = listing["providers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["id"] == "codex")
            .unwrap()
            .clone();
        assert_eq!(entry["kind"], "codex");
        assert!(
            entry["endpoint_origin"].is_null(),
            "a process has no endpoint"
        );
        // Consent is tied to the process identity.
        let (status, _) = call_json(
            &app,
            auth_json(
                "PUT",
                "/api/projects/p/llm-consent",
                serde_json::json!({"provider_id": "codex", "origin": "process:codex"}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        // A URL or a command line has no place in it.
        let mut with_url = body.clone();
        with_url["id"] = serde_json::json!("codex2");
        with_url["base_url"] = serde_json::json!("https://8.8.8.8/v1");
        let (status, _) = call_json(&app, auth_json("POST", "/api/chat/providers", with_url)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let cmd = serde_json::json!({"id": "x", "kind": "acp", "command": ["sh", "-c", "evil"]});
        let (status, _) = call_json(&app, auth_json("POST", "/api/chat/providers", cmd)).await;
        assert!(status.is_client_error(), "{status}");
        // ACP names a declared command; none is declared here.
        let acp = serde_json::json!({"id": "oc", "kind": "acp", "preset": "opencode"});
        let (status, resp) = call_json(&app, auth_json("POST", "/api/chat/providers", acp)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{resp}");
        assert!(
            resp.to_string().contains("CHAT_PROVIDER_ACP_COMMANDS"),
            "{resp}"
        );
        // The test route is a health check of the process: still a 200 verdict.
        let (status, resp) =
            call_json(&app, auth_json("POST", "/api/chat/providers/test", body)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(resp["ok"].is_boolean(), "{resp}");
    }

    const REMOTE_KEY: &str =
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEXTCY3J636nEyMNqNrVj6HXnIXUnXL8sk4c5J6tek3N";

    fn remote_body() -> serde_json::Value {
        serde_json::json!({
            "kind": "claude_code_remote", "label": "Build box 1",
            "host": "127.0.0.1", "ssh_user": "deploy", "ssh_port": 1,
            "host_key": REMOTE_KEY, "remote_cwd": "/srv/work",
            "credential_ref": "vault:ssh-box", "default_model": "sonnet",
        })
    }

    #[tokio::test]
    async fn a_remote_instance_goes_from_the_route_to_a_provider_and_never_to_the_local_claude() {
        let state = mock_server_state().await;
        let app = create_router(state.clone());

        // Create: the id is forced, the origin is the machine, the fingerprint is computed.
        let (status, created) = call_json(
            &app,
            auth_json("POST", "/api/chat/providers", remote_body()),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{created}");
        assert_eq!(created["id"], "claude-code@build-box-1");
        assert_eq!(created["origin"], "ssh:deploy@127.0.0.1:1");
        assert_eq!(
            created["host_key_fingerprint"],
            "SHA256:lP63ZdLutNnRU0/59cDaFw2mPoJzdasi0I3zFrtS3Ak"
        );
        // The reserved id and a pasted key stay refused.
        let mut reserved = remote_body();
        reserved["id"] = serde_json::json!("claude-code");
        let (status, _) = call_json(&app, auth_json("POST", "/api/chat/providers", reserved)).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        let mut pasted = remote_body();
        pasted["credential_ref"] = serde_json::json!("-----BEGIN OPENSSH PRIVATE KEY-----");
        let (status, resp) =
            call_json(&app, auth_json("POST", "/api/chat/providers", pasted)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{resp}");
        assert!(!resp.to_string().contains("BEGIN OPENSSH"), "{resp}");
        let mut private_field = remote_body();
        private_field["private_key"] = serde_json::json!("x");
        let (status, _) = call_json(
            &app,
            auth_json("POST", "/api/chat/providers", private_field),
        )
        .await;
        assert!(status.is_client_error(), "{status}");

        // The listing exposes kind, host and fingerprint, never a key.
        let (_, listing) = call_json(&app, auth_get("/api/chat/providers?project_slug=p")).await;
        let entry = listing["providers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["id"] == "claude-code@build-box-1")
            .expect("listed")
            .clone();
        assert_eq!(entry["kind"], "claude_code_remote");
        assert_eq!(entry["remote"]["host"], "127.0.0.1");
        assert_eq!(entry["remote"]["ssh_port"], 1);
        assert_eq!(
            entry["remote"]["host_key_fingerprint"],
            "SHA256:lP63ZdLutNnRU0/59cDaFw2mPoJzdasi0I3zFrtS3Ak"
        );
        assert!(entry["endpoint_origin"].is_null());
        assert_eq!(entry["allowed_for_project"], false);
        assert_eq!(listing["providers"][0]["id"], "claude-code");
        assert!(listing["providers"][0].get("remote").is_none());
        assert!(
            !listing.to_string().contains("AAAAC3NzaC1lZDI1NTE5"),
            "no key blob"
        );

        // Consent is tied to the machine, and a change of host revokes it.
        let (status, _) = call_json(
            &app,
            auth_json(
                "PUT",
                "/api/projects/p/llm-consent",
                serde_json::json!({"provider_id": "claude-code@build-box-1", "origin": "ssh:deploy@127.0.0.1:1"}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // The real route -> record -> provider -> open path, on a manager over the
        // SAME store: the vault holds the key and grants it to the instance; the
        // machine does not answer (loopback port 1).
        let vault = crate::vault::VaultService::ephemeral();
        vault
            .init(
                "correct horse battery staple".into(),
                chrono::Duration::hours(1),
            )
            .await
            .unwrap();
        let now = chrono::Utc::now();
        vault
            .put(
                "ssh-box",
                "-----BEGIN OPENSSH PRIVATE KEY-----\nx\n-----END OPENSSH PRIVATE KEY-----",
                None,
                now,
            )
            .unwrap();
        vault
            .grant(
                crate::vault::grants::SecretSelector::Names(["ssh-box".to_string()].into()),
                crate::vault::grants::GrantScope::Provider("claude-code@build-box-1".into()),
                chrono::Duration::hours(1),
                None,
                now,
            )
            .unwrap();
        let config = crate::chat::config::ChatConfig {
            jwt_secret: Some("test-secret-test-secret-test-secret".to_string()),
            max_sessions: 10,
            ..Default::default()
        };
        let manager = crate::chat::manager::ChatManager::new_without_memory(
            state.orchestrator.neo4j_arc(),
            mock_app_state().meili,
            config,
        )
        .with_vault(vault);
        let provider = manager
            .provider_for("claude-code@build-box-1")
            .await
            .unwrap();
        assert_eq!(provider.id(), "claude-code@build-box-1");
        assert_eq!(
            provider.kind(),
            nexus_claude::agent::ProviderKind::ClaudeCode
        );
        let request = crate::chat::types::ChatRequest {
            file_scope_lifted: false,
            access: None,
            routing_pool: None,
            routing_mode: None,
            attachments: Vec::new(),
            refs: Vec::new(),
            message: "hi".into(),
            session_id: None,
            cwd: std::env::temp_dir().display().to_string(),
            project_slug: Some("p".into()),
            model: None,
            provider: Some("claude-code@build-box-1".into()),
            task_alias: None,
            persona_alias: None,
            run_provider: None,
            run_model: None,
            max_tokens: None,
            max_cost_usd: None,
            task_class: None,
            permission_mode: Some("default".into()),
            add_dirs: None,
            workspace_slug: None,
            user_claims: Some(crate::auth::jwt::Claims::service_account("t")),
            spawned_by: None,
            task_context: None,
            subagent_type: None,
            route_auto: false,
            scaffolding_override: None,
            runner_context: None,
            routing_decision_id: None,
            denied_tools: vec![],
        };
        let err = manager.create_session(&request).await.unwrap_err();
        let failure = crate::chat::provider::errors::classify_open_error(
            &err,
            Some("claude-code@build-box-1"),
        )
        .unwrap_or_else(|| panic!("not a typed failure: {err:#}"));
        // Preflight refused it: the machine is unreachable. Nothing ran locally.
        assert_eq!(failure.code, "endpoint_unreachable", "{err:#}");
        assert!(!format!("{err:#} {failure:?}").contains("PRIVATE KEY"));
        assert!(manager.agent_runtime.is_empty().await, "no live session");

        // Trust is refused for a remote machine by default...
        let mut trust = request.clone();
        trust.permission_mode = Some("bypassPermissions".into());
        let err = manager.create_session(&trust).await.unwrap_err();
        let failure = crate::chat::provider::errors::classify_open_error(&err, None).unwrap();
        assert_eq!(failure.code, "unsupported", "{err:#}");
        // ... and passes that gate once the machine allows it (then the preflight
        // refuses the unreachable machine, which is how we know it got that far).
        let (status, _) = call_json(
            &app,
            auth_json(
                "PATCH",
                "/api/chat/providers/claude-code@build-box-1",
                serde_json::json!({"allow_trust": true}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let err = manager.create_session(&trust).await.unwrap_err();
        let failure = crate::chat::provider::errors::classify_open_error(&err, None).unwrap();
        assert_eq!(failure.code, "endpoint_unreachable", "{err:#}");

        // A change of host revokes the consent: the same request is now refused.
        let (status, patched) = call_json(
            &app,
            auth_json(
                "PATCH",
                "/api/chat/providers/claude-code@build-box-1",
                serde_json::json!({"host": "127.0.0.2"}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(patched["consents_invalidated"], true);
        let err = manager.create_session(&request).await.unwrap_err();
        let failure = crate::chat::provider::errors::classify_open_error(&err, None).unwrap();
        assert_eq!(failure.code, "endpoint_not_allowed", "{err:#}");
    }

    #[tokio::test]
    async fn the_host_key_route_refuses_hostile_hosts_before_any_process_starts() {
        let app = test_app().await;
        for host in [
            "-oProxyCommand=touch /tmp/pwned",
            "-oProxyCommand=x",
            "host name",
            "a;b",
            "a|b",
            "$(id)",
            "`id`",
            "",
            "a\nb",
        ] {
            let (status, resp) = call_json(
                &app,
                auth_json(
                    "POST",
                    "/api/chat/providers/ssh-host-key",
                    serde_json::json!({"host": host}),
                ),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{host:?}: {resp}");
        }
        let (status, _) = call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers/ssh-host-key",
                serde_json::json!({"host": "h", "ssh_port": 0}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        // Unknown fields are refused like everywhere else in this family.
        let (status, _) = call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers/ssh-host-key",
                serde_json::json!({"host": "h", "extra": 1}),
            ),
        )
        .await;
        assert!(status.is_client_error());
        // Without a token: refused (same auth as the other provider writes).
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat/providers/ssh-host-key")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"host":"h"}"#))
            .unwrap();
        let (status, _) = call_json(&app, req).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn the_run_costs_route_answers_the_two_counters_by_model_provider_and_class() {
        let h = action_harness(None).await;
        let run = Uuid::new_v4();
        let mut rows = Vec::new();
        for (provider, model, class, usd, basis) in [
            ("claude-code", "opus", "complex", 2.0, "reported"),
            ("claude-code", "opus", "simple", 3.0, "subscription"),
            ("local", "llama", "simple", 0.0, "unknown"),
        ] {
            let mut ae =
                crate::neo4j::agent_execution::AgentExecutionNode::new(run, Uuid::new_v4());
            ae.provider_id = provider.into();
            ae.model = Some(model.into());
            ae.task_class = Some(class.into());
            ae.cost_usd = usd;
            ae.cost_basis = Some(basis.into());
            h.graph.create_agent_execution(&ae).await.unwrap();
            rows.push(ae);
        }
        let (status, body) = call(&h.app, auth_get(&format!("/api/chat/runs/{run}/costs"))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"]["marginal_usd"], 2.0);
        assert_eq!(body["total"]["notional_usd"], 3.0);
        assert_eq!(body["total"]["unknown_cost_executions"], 1);
        assert_eq!(body["by_provider"]["claude-code"]["executions"], 2);
        assert_eq!(body["by_task_class"]["simple"]["executions"], 2);
        assert_eq!(body["by_model"]["llama"]["unknown_cost_executions"], 1);
    }

    /// CTX0: a conversation's usage, turn by turn, read from its stored `result` events:
    /// a native turn's cache reads are there, an unreported cache write stays absent, and
    /// only `result` events count. An unknown session is a 404.
    #[tokio::test]
    async fn the_session_usage_route_sums_the_stored_turns_with_their_cache_reads() {
        let h = action_harness(None).await;
        let session = crate::test_helpers::test_chat_session(None);
        h.graph.create_chat_session(&session).await.unwrap();
        let sid = session.id;
        let record =
            |seq: i64, kind: &str, data: serde_json::Value| crate::neo4j::models::ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: sid,
                seq,
                event_type: kind.to_string(),
                data: data.to_string(),
                created_at: chrono::Utc::now(),
            };
        h.graph
            .store_chat_events(
                sid,
                vec![
                    record(0, "user_message", serde_json::json!({"type": "user_message", "content": "a"})),
                    record(1, "result", serde_json::json!({
                        "type": "result", "model": "deepseek-v4-pro",
                        "usage": {"input_tokens": 4000, "output_tokens": 100, "cache_read_tokens": 0}
                    })),
                    record(2, "user_message", serde_json::json!({"type": "user_message", "content": "b"})),
                    record(3, "result", serde_json::json!({
                        "type": "result", "model": "deepseek-v4-pro",
                        "usage": {"input_tokens": 200, "output_tokens": 50, "cache_read_tokens": 3800}
                    })),
                ],
            )
            .await
            .unwrap();
        let (status, body) =
            call(&h.app, auth_get(&format!("/api/chat/sessions/{sid}/usage"))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["turns"], 2, "{body}");
        assert_eq!(body["input_tokens"], 4200);
        assert_eq!(body["output_tokens"], 150);
        assert_eq!(body["cache_read_tokens"], 3800);
        assert!(
            body["cache_creation_tokens"].is_null(),
            "never reported: {body}"
        );
        assert_eq!(body["by_turn"][1]["seq"], 3);
        assert_eq!(body["by_turn"][1]["cache_hit_ratio"], 0.95);
        let (status, _) = call(
            &h.app,
            auth_get(&format!("/api/chat/sessions/{}/usage", Uuid::new_v4())),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn the_send_journal_is_readable_by_a_person_only_filtered_and_newest_first() {
        let h = action_harness(None).await;
        for (ms, project) in [(1_000, "a"), (2_000, "b"), (3_000, "a")] {
            h.graph
                .put_llm_setting(
                    "journal",
                    &format!("send:{ms}:sess-{ms}"),
                    &serde_json::json!({"session_id": format!("sess-{ms}"), "project": project, "provider": "local", "origin": "https://x"}).to_string(),
                )
                .await
                .unwrap();
        }
        let (status, body) = call(&h.app, auth_get("/api/chat/send-journal")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let ids: Vec<_> = body["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["session_id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids, ["sess-3000", "sess-2000", "sess-1000"]);
        let (_, body) = call(
            &h.app,
            auth_get("/api/chat/send-journal?project_slug=a&limit=1"),
        )
        .await;
        assert_eq!(body["entries"].as_array().unwrap().len(), 1);
        assert_eq!(body["entries"][0]["session_id"], "sess-3000");
        // An agent token has no business reading it.
        let token = agent_bearer(Uuid::new_v4());
        let (status, _) = call(
            &h.app,
            agent_req(&token, "GET", "/api/chat/send-journal", ""),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn a_saved_instances_status_is_its_real_health_not_unknown() {
        let app = test_app().await;
        let mut body = deepseek("http://127.0.0.1:9/v1");
        body["credential_ref"] = serde_json::json!("none");
        let (status, _) = call_json(&app, auth_json("POST", "/api/chat/providers", body)).await;
        assert_eq!(status, StatusCode::CREATED);
        let (status, health) =
            call_json(&app, auth_get("/api/chat/providers/deepseek/status")).await;
        assert_eq!(status, StatusCode::OK);
        assert_ne!(health["state"], "ok", "{health}");
        assert_ne!(
            health["state"], "unknown",
            "a closed port is measured: {health}"
        );
    }

    /// VERIFIER: creating a third-party instance while authentication is off is
    /// refused with 409 `security_gate_closed` (A32, documented in provider-errors.md).
    #[tokio::test]
    async fn verifier_creating_a_third_party_instance_without_authentication_is_409() {
        let mut state = Arc::try_unwrap(mock_server_state().await)
            .ok()
            .expect("sole owner of the state");
        state.auth_config = None;
        let app = create_router(Arc::new(state));
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat/providers")
            .header("content-type", "application/json")
            .body(Body::from(deepseek("https://8.8.8.8/v1").to_string()))
            .unwrap();
        let (status, body) = call_json(&app, req).await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}");
    }

    #[tokio::test]
    async fn an_env_credential_is_refused_unless_the_variable_is_declared() {
        let app = test_app().await;
        let mut body = deepseek("https://8.8.8.8/v1");
        body["credential_ref"] = serde_json::json!("env:HOME");
        let (status, resp) = call_json(&app, auth_json("POST", "/api/chat/providers", body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{resp}");
        assert!(
            resp.to_string().contains("CHAT_PROVIDER_ENV_CREDENTIALS"),
            "{resp}"
        );
    }

    #[tokio::test]
    async fn the_test_route_always_answers_200_with_a_verdict() {
        let app = test_app().await;
        let (status, body) = call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers/test",
                deepseek("https://10.1.2.3/v1"),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["ok"], false);
        assert_eq!(body["health"]["code"], "endpoint_private_address");
        let (_, body) = call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers/test",
                deepseek("https://8.8.8.8/v1"),
            ),
        )
        .await;
        // A draft with a key is not tried: the key would go to an endpoint nobody saved.
        assert_eq!(body["ok"], false);
        assert_eq!(
            body["health"]["code"],
            "credential_test_requires_saved_instance"
        );
        // Saved, then tried: the key lives in a vault nobody unlocked: a verdict, no connection.
        let (status, _) = call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers",
                deepseek("https://8.8.8.8/v1"),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        let (_, body) = call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers/test",
                deepseek("https://8.8.8.8/v1"),
            ),
        )
        .await;
        assert_eq!(body["ok"], false);
        assert_eq!(body["health"]["code"], "credentials_locked");
        assert_eq!(body["health"]["state"], "auth_required");
        // A closed local port with no credential: unreachable, still a 200.
        let mut local = deepseek("http://127.0.0.1:9/v1");
        local["credential_ref"] = serde_json::json!("none");
        let (status, body) =
            call_json(&app, auth_json("POST", "/api/chat/providers/test", local)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["ok"], false, "{body}");
    }

    #[tokio::test]
    async fn consent_is_bound_to_the_origin_and_stops_holding_when_it_changes() {
        let app = test_app().await;
        call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers",
                deepseek("https://8.8.8.8/v1"),
            ),
        )
        .await;

        let (status, body) = call_json(
            &app,
            auth_json(
                "PUT",
                "/api/projects/p/llm-consent",
                serde_json::json!({"provider_id": "deepseek", "origin": "https://1.1.1.1"}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}");

        let (status, body) = call_json(
            &app,
            auth_json(
                "PUT",
                "/api/projects/p/llm-consent",
                serde_json::json!({"provider_id": "deepseek", "origin": "https://8.8.8.8"}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["valid"], true);
        assert!(body["consented_by"].is_string() && body["consented_at"].is_string());

        let (_, listing) = call_json(&app, auth_get("/api/chat/providers?project_slug=p")).await;
        let entry = listing["providers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["id"] == "deepseek")
            .unwrap()
            .clone();
        assert_eq!(entry["allowed_for_project"], true);

        // The instance moves to another origin: the consent no longer holds.
        let (status, patched) = call_json(
            &app,
            auth_json(
                "PATCH",
                "/api/chat/providers/deepseek",
                serde_json::json!({"base_url": "https://1.1.1.1/v1"}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{patched}");
        assert_eq!(patched["consents_invalidated"], true);
        let (_, rows) = call_json(&app, auth_get("/api/projects/p/llm-consents")).await;
        assert_eq!(rows[0]["valid"], false);
        let (_, listing) = call_json(&app, auth_get("/api/chat/providers?project_slug=p")).await;
        let entry = listing["providers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["id"] == "deepseek")
            .unwrap()
            .clone();
        assert_eq!(entry["allowed_for_project"], false);

        let (status, _) = call_json(
            &app,
            auth_json(
                "DELETE",
                "/api/projects/p/llm-consent/deepseek",
                serde_json::json!({}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        let (status, _) = call_json(
            &app,
            auth_json(
                "DELETE",
                "/api/projects/p/llm-consent/deepseek",
                serde_json::json!({}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn the_consent_for_chats_outside_a_project_follows_the_same_rules() {
        // B3: PUT/GET/DELETE /api/chat/llm-consents[/{id}], same body and rules as a
        // project's consent, stored apart from every project.
        let app = test_app().await;
        call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers",
                deepseek("https://8.8.8.8/v1"),
            ),
        )
        .await;
        // The origin the person was shown must still be the instance's.
        let (status, body) = call_json(
            &app,
            root_json(
                "PUT",
                "/api/chat/llm-consents/deepseek",
                serde_json::json!({"origin": "https://1.1.1.1"}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}");
        // claude-code needs none; an unknown instance is a 404.
        let (status, _) = call_json(
            &app,
            root_json(
                "PUT",
                "/api/chat/llm-consents/claude-code",
                serde_json::json!({"origin": "x"}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = call_json(
            &app,
            root_json(
                "PUT",
                "/api/chat/llm-consents",
                serde_json::json!({"provider_id": "ghost", "origin": "https://8.8.8.8"}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        let (status, body) = call_json(
            &app,
            root_json(
                "PUT",
                "/api/chat/llm-consents",
                serde_json::json!({"provider_id": "deepseek", "origin": "https://8.8.8.8"}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["valid"], true);
        let (_, rows) = call_json(&app, auth_get("/api/chat/llm-consents")).await;
        assert_eq!(rows.as_array().unwrap().len(), 1);
        assert_eq!(rows[0]["provider_id"], "deepseek");
        // The listing without a project reads that scope (nit 6); a project does not.
        let allowed = |listing: &serde_json::Value| {
            listing["providers"]
                .as_array()
                .unwrap()
                .iter()
                .find(|p| p["id"] == "deepseek")
                .unwrap()["allowed_for_project"]
                .clone()
        };
        let (_, listing) = call_json(&app, auth_get("/api/chat/providers")).await;
        assert_eq!(allowed(&listing), true);
        let (_, listing) = call_json(&app, auth_get("/api/chat/providers?project_slug=p")).await;
        assert_eq!(allowed(&listing), false);
        // Not a project's consent.
        let (_, rows) = call_json(&app, auth_get("/api/projects/p/llm-consents")).await;
        assert!(rows.as_array().unwrap().is_empty(), "{rows}");

        let (status, _) = call_json(
            &app,
            root_json(
                "DELETE",
                "/api/chat/llm-consents/deepseek",
                serde_json::json!({}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        let (_, rows) = call_json(&app, auth_get("/api/chat/llm-consents")).await;
        assert!(rows.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn only_root_allows_or_withdraws_a_provider_for_the_chats_outside_a_project() {
        // Review of #722, finding 1: the no-project consent decides where EVERY user's
        // no-project chats may go (and makes an instance their default): root only.
        let app = test_app().await;
        call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers",
                deepseek("https://8.8.8.8/v1"),
            ),
        )
        .await;
        let allow = serde_json::json!({"provider_id": "deepseek", "origin": "https://8.8.8.8"});
        let origin = serde_json::json!({"origin": "https://8.8.8.8"});
        for (method, uri, body) in [
            ("PUT", "/api/chat/llm-consents", allow.clone()),
            ("PUT", "/api/chat/llm-consents/deepseek", origin.clone()),
        ] {
            let (status, body) = call_json(&app, auth_json(method, uri, body)).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{method} {uri}: {body}");
        }
        let (_, rows) = call_json(&app, auth_get("/api/chat/llm-consents")).await;
        assert!(rows.as_array().unwrap().is_empty(), "nothing was stored");
        let (status, _) = call_json(&app, root_json("PUT", "/api/chat/llm-consents", allow)).await;
        assert_eq!(status, StatusCode::OK);
        let (status, _) = call_json(
            &app,
            auth_json(
                "DELETE",
                "/api/chat/llm-consents/deepseek",
                serde_json::json!({}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        let (_, rows) = call_json(&app, auth_get("/api/chat/llm-consents")).await;
        assert_eq!(rows.as_array().unwrap().len(), 1, "still there");
        // A project's consent stays open to any signed-in person.
        let (status, body) = call_json(
            &app,
            auth_json("PUT", "/api/projects/p/llm-consents/deepseek", origin),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }

    #[tokio::test]
    async fn roles_aliases_and_policy_round_trip_and_are_validated() {
        let app = test_app().await;
        call_json(
            &app,
            auth_json(
                "POST",
                "/api/chat/providers",
                deepseek("https://8.8.8.8/v1"),
            ),
        )
        .await;

        let (_, roles) = call_json(&app, auth_get("/api/chat/roles")).await;
        assert_eq!(roles, serde_json::json!({}), "absent = single provider");
        let want = serde_json::json!({"pilot": {"provider": "claude-code"}, "executor": {"provider": "deepseek", "alias": "fast"}});
        // The global pilot routes every user's chats: another signed-in person gets 403.
        let (status, body) =
            call_json(&app, auth_json("PUT", "/api/chat/roles", want.clone())).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
        let (_, unchanged) = call_json(&app, auth_get("/api/chat/roles")).await;
        assert_eq!(unchanged, serde_json::json!({}));
        let (status, _) = call_json(&app, root_json("PUT", "/api/chat/roles", want.clone())).await;
        assert_eq!(status, StatusCode::OK);
        let (_, got) = call_json(&app, auth_get("/api/chat/roles")).await;
        assert_eq!(got, want);
        let (status, _) = call_json(
            &app,
            root_json(
                "PUT",
                "/api/chat/roles",
                serde_json::json!({"pilot": {"provider": "ghost"}}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = call_json(
            &app,
            auth_json(
                "PUT",
                "/api/projects/p/llm-roles",
                serde_json::json!({"executor": {"provider": "claude-code"}}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (_, project_roles) = call_json(&app, auth_get("/api/projects/p/llm-roles")).await;
        assert_eq!(
            project_roles,
            serde_json::json!({"executor": {"provider": "claude-code"}})
        );

        let aliases = serde_json::json!([{"alias": "fast", "provider": "deepseek", "model": "deepseek-chat"}]);
        let (status, _) = call_json(
            &app,
            auth_json("PUT", "/api/chat/model-aliases", aliases.clone()),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (_, got) = call_json(&app, auth_get("/api/chat/model-aliases")).await;
        assert_eq!(got, aliases);

        let (_, policy) = call_json(&app, auth_get("/api/chat/model-policy")).await;
        assert_eq!(policy["mode"], "off", "ships off");
        let ok = serde_json::json!({"mode": "shadow", "rules": {"runner.simple": "fast"}, "fallback": ["fast"], "caps": {"per_run_usd": 5.0}});
        let (status, _) = call_json(&app, auth_json("PUT", "/api/chat/model-policy", ok)).await;
        assert_eq!(status, StatusCode::OK);
        let bad = serde_json::json!({"mode": "enforce", "rules": {"chat": "ghost"}, "fallback": [], "caps": {}});
        let (status, _) = call_json(&app, auth_json("PUT", "/api/chat/model-policy", bad)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // ====================================================================
    // Run routing, task alias, interrupt cascade, tree annotation
    // ====================================================================

    /// B36: the session tree carries `status`, `routed_by`, `provider_id`,
    /// `model` and `cost_usd` per node (a cold session is `closed`).
    #[tokio::test]
    async fn the_session_tree_reports_status_and_routed_by_per_node() {
        let h = action_harness(None).await;
        let mut session = crate::test_helpers::test_chat_session(None);
        session.provider_id = Some("fake".to_string());
        session.routed_by = Some("task".to_string());
        session.total_cost_usd = Some(1.5);
        h.graph.create_chat_session(&session).await.unwrap();

        let uri = format!("/api/chat/sessions/{}/tree", session.id);
        let (status, body) = call(&h.app, auth_get(&uri)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let tree = body.as_array().expect("an array of nodes");
        assert_eq!(tree.len(), 1, "{body}");
        assert_eq!(tree[0]["provider_id"], "fake");
        assert_eq!(tree[0]["model"], "claude-opus-4-6");
        assert_eq!(tree[0]["cost_usd"], 1.5);
        assert_eq!(tree[0]["routed_by"], "task");
        assert_eq!(tree[0]["status"], "closed");
    }

    #[tokio::test]
    async fn a_task_model_alias_must_be_defined_is_stored_and_shown_and_can_be_cleared() {
        let h = action_harness(None).await;
        let plan = crate::test_helpers::test_plan();
        h.graph.create_plan(&plan).await.unwrap();
        let task = crate::test_helpers::test_task();
        h.graph.create_task(plan.id, &task).await.unwrap();
        let uri = format!("/api/tasks/{}", task.id);
        let patch = |body: serde_json::Value| {
            Request::builder()
                .method("PATCH")
                .uri(uri.clone())
                .header("content-type", "application/json")
                .header("authorization", test_bearer_token())
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        // Not defined: refused.
        let (status, _) = call(&h.app, patch(serde_json::json!({"model_alias": "fast"}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        // Define it, then set it.
        h.graph
            .put_llm_setting(
                "global",
                "model_aliases",
                &serde_json::json!([{"alias": "fast", "provider": "claude-code", "model": "m"}])
                    .to_string(),
            )
            .await
            .unwrap();
        let (status, _) = call(&h.app, patch(serde_json::json!({"model_alias": "fast"}))).await;
        assert!(status.is_success(), "{status}");
        let (_, body) = call(&h.app, auth_get(&uri)).await;
        assert_eq!(body["task"]["model_alias"], "fast");
        // Empty clears it.
        let (status, _) = call(&h.app, patch(serde_json::json!({"model_alias": ""}))).await;
        assert!(status.is_success());
        let (_, body) = call(&h.app, auth_get(&uri)).await;
        assert!(body["task"].get("model_alias").is_none(), "{body}");
        // An unknown task is a 404, nothing stored.
        let ghost = Request::builder()
            .method("PATCH")
            .uri(format!("/api/tasks/{}", Uuid::new_v4()))
            .header("content-type", "application/json")
            .header("authorization", test_bearer_token())
            .body(Body::from(r#"{"model_alias":"fast"}"#))
            .unwrap();
        let (status, _) = call(&h.app, ghost).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn a_run_naming_an_unknown_provider_is_refused_before_it_starts() {
        let h = action_harness(None).await;
        let plan = crate::test_helpers::test_plan();
        h.graph.create_plan(&plan).await.unwrap();
        let req = Request::builder()
            .method("POST")
            .uri(format!("/api/plans/{}/run", plan.id))
            .header("content-type", "application/json")
            .header("authorization", test_bearer_token())
            .body(Body::from(
                r#"{"cwd": ".", "provider": "ghost", "model": "m", "max_tokens": 1000}"#,
            ))
            .unwrap();
        let (status, body) = call(&h.app, req).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    }

    #[tokio::test]
    async fn interrupt_with_cascade_stops_the_descendants_and_reports_the_count() {
        let h = action_harness(None).await;
        let parent = crate::test_helpers::test_chat_session(None);
        let mut child = crate::test_helpers::test_chat_session(None);
        child.spawned_by = Some(
            serde_json::json!({"type": "delegation", "parent_session_id": parent.id.to_string()})
                .to_string(),
        );
        let mut idle_child = crate::test_helpers::test_chat_session(None);
        idle_child.spawned_by = child.spawned_by.clone();
        for n in [&parent, &child, &idle_child] {
            h.graph.create_chat_session(n).await.unwrap();
        }
        for n in [&parent, &child] {
            test_support::insert_live_session_without_cli(&h.manager, &n.id.to_string()).await;
        }
        let req = Request::builder()
            .method("POST")
            .uri(format!("/api/chat/sessions/{}/interrupt", parent.id))
            .header("content-type", "application/json")
            .header("authorization", test_bearer_token())
            .body(Body::from(
                r#"{"scope": "turn_and_tools", "cascade": true}"#,
            ))
            .unwrap();
        let (status, body) = call(&h.app, req).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        // Two children persisted, one of them live: 1 of 2 stopped.
        assert_eq!(body["cascade"]["total"], 2, "{body}");
        assert_eq!(body["cascade"]["stopped"], 1, "{body}");
        // Without the flag the answer has no cascade part (unchanged shape).
        let plain = Request::builder()
            .method("POST")
            .uri(format!("/api/chat/sessions/{}/interrupt", parent.id))
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap();
        let (_, body) = call(&h.app, plain).await;
        assert!(body.get("cascade").is_none());
    }

    // ====================================================================
    // An agent session token reaches only the sessions it spawned
    // ====================================================================

    /// A bound, live agent token for `session` (restricted profile, like a third party's).
    fn agent_bearer(session: Uuid) -> String {
        let claims = crate::auth::jwt::Claims::service_account("agent");
        let binding = crate::auth::jwt::AgentSessionBinding {
            session_id: session.to_string(),
            ceiling: Some("default".into()),
            tool_profile: Some("restricted".into()),
            third_party: false,
        };
        let (token, jti) = crate::auth::jwt::generate_session_token(
            &claims,
            Some(&binding),
            "test-secret-key-minimum-32-chars!!",
            3600,
        )
        .unwrap();
        crate::auth::agent_tokens::register(&jti, Some(&session.to_string()));
        format!("Bearer {token}")
    }

    fn agent_req(token: &str, method: &str, uri: &str, body: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .header("authorization", token)
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    /// Moving a conversation sends it to another origin: a person's call. An agent is
    /// refused even on its own session; a signed-in user gets past the gate (and meets
    /// the switch's own checks: here, the session is already on that provider).
    #[tokio::test]
    async fn only_a_signed_in_user_can_move_a_conversation_to_another_provider() {
        let h = action_harness(None).await;
        let own = seed_session(&h).await;
        let uri = format!("/api/chat/sessions/{own}/switch-provider");
        let body = r#"{"provider":"claude-code","message":"go on"}"#;

        let (status, resp) = call(&h.app, agent_req(&agent_bearer(own), "POST", &uri, body)).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{resp}");

        let (status, resp) = call(&h.app, auth_post(&uri, body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{resp}");
        assert!(resp.to_string().contains("already on"), "{resp}");
    }

    #[tokio::test]
    async fn an_agent_token_cannot_write_to_a_session_it_did_not_spawn() {
        let h = action_harness(None).await;
        let parent = crate::test_helpers::test_chat_session(None);
        let mut child = crate::test_helpers::test_chat_session(None);
        child.spawned_by = Some(
            serde_json::json!({"type": "conversation", "parent_session_id": parent.id.to_string()})
                .to_string(),
        );
        let human = crate::test_helpers::test_chat_session(None);
        for n in [&parent, &child, &human] {
            h.graph.create_chat_session(n).await.unwrap();
        }
        let token = agent_bearer(parent.id);

        // Every mutating route under /api/chat/sessions/{id}: a stranger is refused.
        let routes: [(&str, String, &str); 5] = [
            (
                "POST",
                format!("/api/chat/sessions/{}/messages", human.id),
                r#"{"content":"x"}"#,
            ),
            (
                "POST",
                format!("/api/chat/sessions/{}/interrupt", human.id),
                "{}",
            ),
            (
                "POST",
                format!("/api/chat/sessions/{}/cancel-tools", human.id),
                "{}",
            ),
            (
                "PATCH",
                format!("/api/chat/sessions/{}", human.id),
                r#"{"title":"x"}"#,
            ),
            ("DELETE", format!("/api/chat/sessions/{}", human.id), ""),
        ];
        for (method, uri, body) in &routes {
            let (status, _) = call(&h.app, agent_req(&token, method, uri, body)).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{method} {uri}");
        }
        // The session itself is not a child of itself: an agent does not drive its own session through REST either.
        let own = format!("/api/chat/sessions/{}/interrupt", parent.id);
        let (status, _) = call(&h.app, agent_req(&token, "POST", &own, "{}")).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        // Annotating its own session stays possible (what the MCP tools do).
        let (status, _) = call(
            &h.app,
            agent_req(
                &token,
                "POST",
                &format!("/api/chat/sessions/{}/discussed", parent.id),
                r#"{"entities":[]}"#,
            ),
        )
        .await;
        assert_ne!(status, StatusCode::FORBIDDEN, "own annotation");
        // Its own child: not refused by the boundary.
        let (status, _) = call(
            &h.app,
            agent_req(
                &token,
                "POST",
                &format!("/api/chat/sessions/{}/interrupt", child.id),
                "{}",
            ),
        )
        .await;
        assert_ne!(
            status,
            StatusCode::FORBIDDEN,
            "a child of the session is reachable"
        );
        // Reads stay open (the tree, the session).
        let (status, _) = call(
            &h.app,
            agent_req(
                &token,
                "GET",
                &format!("/api/chat/sessions/{}", human.id),
                "",
            ),
        )
        .await;
        assert_ne!(status, StatusCode::FORBIDDEN);
        // A person is not held to it.
        let (status, _) = call(
            &h.app,
            auth_post(&format!("/api/chat/sessions/{}/interrupt", human.id), "{}"),
        )
        .await;
        assert_ne!(status, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn providers_answers_the_no_project_consent_without_a_project() {
        // Review of #722, nit 6: with no project asked, the scope is the chats outside any
        // project; claude-code needs no consent there either.
        let app = test_app().await;
        let resp = app.oneshot(auth_get("/api/chat/providers")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["providers"][0]["allowed_for_project"], true);
    }

    #[tokio::test]
    async fn test_list_sessions_with_data() {
        let s1 = test_chat_session(Some("proj-a"));
        let s2 = test_chat_session(Some("proj-b"));
        let app = test_app_with_sessions(&[s1, s2]).await;

        let resp = app.oneshot(auth_get("/api/chat/sessions")).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["total"], 2);
        assert_eq!(json["items"].as_array().unwrap().len(), 2);
    }

    // ====================================================================
    // Live activity — what a conversation is doing *now*
    // ====================================================================

    /// A client may poll this endpoint unconditionally: no chat manager is
    /// "nothing is running", not an error. Answering 404 here would make every
    /// conversation list log failures on a server built without chat.
    #[tokio::test]
    async fn live_activity_answers_an_empty_map_when_chat_is_not_configured() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/chat/live-activity"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(
            json["sessions"].as_object().unwrap().is_empty(),
            "no manager means no live session, not a missing field"
        );
        assert!(
            json["generated_at"].as_str().is_some(),
            "a client must be able to tell a fresh empty answer from a stale one"
        );
    }

    /// The field is omitted, not sent as zeroes: a page of cold sessions costs
    /// exactly what it cost before live activity existed.
    #[tokio::test]
    async fn a_quiet_session_carries_no_activity_field_at_all() {
        let app = test_app_with_sessions(&[test_chat_session(Some("proj-a"))]).await;
        let resp = app.oneshot(auth_get("/api/chat/sessions")).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let item = &json["items"][0];
        assert!(item["id"].is_string(), "the session itself is still there");
        assert!(
            item.get("activity").is_none(),
            "quiet sessions stay off the wire: got {}",
            item
        );
    }

    /// The shape the frontend decodes. Pinned here because the status line in
    /// the conversation list reads these five fields by name.
    #[test]
    fn session_activity_serialises_the_five_fields_the_ui_reads() {
        let json = serde_json::to_value(SessionActivity {
            live: true,
            streaming: true,
            pending_permissions: 2,
            monitors: 1,
            bash_tasks: 3,
        })
        .unwrap();
        assert_eq!(json["live"], true);
        assert_eq!(json["streaming"], true);
        assert_eq!(json["pending_permissions"], 2);
        assert_eq!(json["monitors"], 1);
        assert_eq!(json["bash_tasks"], 3);
    }

    #[tokio::test]
    async fn test_list_sessions_filter_by_project() {
        let s1 = test_chat_session(Some("proj-a"));
        let s2 = test_chat_session(Some("proj-a"));
        let s3 = test_chat_session(Some("proj-b"));
        let app = test_app_with_sessions(&[s1, s2, s3]).await;

        let resp = app
            .oneshot(auth_get("/api/chat/sessions?project_slug=proj-a"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["total"], 2);
    }

    // ====================================================================
    // GET /api/chat/sessions/{id} — get
    // ====================================================================

    #[tokio::test]
    async fn test_get_session_found() {
        let session = test_chat_session(Some("my-proj"));
        let session_id = session.id;
        let app = test_app_with_sessions(&[session]).await;

        let resp = app
            .oneshot(auth_get(&format!("/api/chat/sessions/{}", session_id)))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["id"], session_id.to_string());
        assert_eq!(json["project_slug"], "my-proj");
        assert_eq!(json["model"], "claude-opus-4-6");
    }

    #[tokio::test]
    async fn test_get_session_not_found() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();

        let resp = app
            .oneshot(auth_get(&format!("/api/chat/sessions/{}", fake_id)))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ====================================================================
    // DELETE /api/chat/sessions/{id} — delete
    // ====================================================================

    #[tokio::test]
    async fn test_delete_session_found() {
        let session = test_chat_session(None);
        let session_id = session.id;
        let app = test_app_with_sessions(&[session]).await;

        let resp = app
            .oneshot(auth_delete(&format!("/api/chat/sessions/{}", session_id)))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["deleted"], true);
    }

    #[tokio::test]
    async fn test_delete_session_not_found() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();

        let resp = app
            .oneshot(auth_delete(&format!("/api/chat/sessions/{}", fake_id)))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ====================================================================
    // POST /api/chat/sessions/{id}/close — close (non-destructive)
    // ====================================================================

    /// A server with no live chat manager has nothing running to close: the
    /// endpoint answers, it never fails.
    #[tokio::test]
    async fn close_on_a_server_without_a_chat_manager_says_nothing_is_running() {
        let app = test_app().await;
        let sid = Uuid::new_v4();

        let (status, body) = call(
            &app,
            auth_post(&format!("/api/chat/sessions/{sid}/close"), "{}"),
        )
        .await;

        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["closed"], false);
        assert_eq!(body["reason"], "no live chat manager");
    }

    /// Closing a live session frees its envelope slot but keeps its node: a later
    /// message resumes it (the explicit closure a delegated zombie needs).
    #[tokio::test]
    async fn close_frees_a_live_session_but_keeps_its_node() {
        let h = action_harness(None).await;
        let sid = seed_session(&h).await;
        test_support::insert_live_session_without_cli(&h.manager, &sid.to_string()).await;
        assert_eq!(
            h.manager.active_session_count().await,
            1,
            "the session is live"
        );

        let (status, body) = call(
            &h.app,
            auth_post(&format!("/api/chat/sessions/{sid}/close"), "{}"),
        )
        .await;

        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["closed"], true);
        assert_eq!(
            h.manager.active_session_count().await,
            0,
            "its slot is freed"
        );
        assert!(
            h.graph.get_chat_session(sid).await.unwrap().is_some(),
            "the node and its transcript survive: a later message resumes it"
        );
    }

    // ====================================================================
    // POST endpoints — no chat_manager returns error
    // ====================================================================

    #[tokio::test]
    async fn test_create_session_no_chat_manager() {
        let app = test_app().await;

        let resp = app
            .oneshot(auth_post(
                "/api/chat/sessions",
                r#"{"message":"Hello","cwd":"/tmp"}"#,
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    // ====================================================================
    // MessagesQuery serde
    // ====================================================================

    #[test]
    fn test_messages_query_defaults() {
        let json = r#"{}"#;
        let query: MessagesQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.limit, 50);
        assert_eq!(query.offset, 0);
    }

    #[test]
    fn test_messages_query_custom() {
        let json = r#"{"limit": 10, "offset": 5}"#;
        let query: MessagesQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.limit, 10);
        assert_eq!(query.offset, 5);
    }

    #[test]
    fn test_default_messages_limit_value() {
        assert_eq!(default_messages_limit(), 50);
    }

    // ====================================================================
    // GET /api/chat/sessions/{id}/messages — no chat_manager
    // ====================================================================

    #[tokio::test]
    async fn test_list_messages_no_chat_manager() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();

        let resp = app
            .oneshot(auth_get(&format!(
                "/api/chat/sessions/{}/messages",
                fake_id
            )))
            .await
            .unwrap();

        // No chat_manager → 500 Internal Server Error
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_list_messages_invalid_session_id() {
        let app = test_app().await;

        let resp = app
            .oneshot(auth_get("/api/chat/sessions/not-a-uuid/messages"))
            .await
            .unwrap();

        // Invalid UUID in path → 400 Bad Request
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ====================================================================
    // GET /api/chat/sessions/{id} — conversation_id field
    // ====================================================================

    #[tokio::test]
    async fn test_get_session_includes_conversation_id() {
        let mut session = test_chat_session(Some("my-proj"));
        session.conversation_id = Some("conv-test-123".into());
        let session_id = session.id;
        let app = test_app_with_sessions(&[session]).await;

        let resp = app
            .oneshot(auth_get(&format!("/api/chat/sessions/{}", session_id)))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["conversation_id"], "conv-test-123");
    }

    #[tokio::test]
    async fn test_get_session_conversation_id_null() {
        let session = test_chat_session(None);
        let session_id = session.id;
        let app = test_app_with_sessions(&[session]).await;

        let resp = app
            .oneshot(auth_get(&format!("/api/chat/sessions/{}", session_id)))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["conversation_id"].is_null());
    }

    // ====================================================================
    // GET /api/chat/sessions — list includes conversation_id
    // ====================================================================

    #[tokio::test]
    async fn test_list_sessions_includes_conversation_id() {
        let mut session = test_chat_session(Some("proj-a"));
        session.conversation_id = Some("conv-xyz".into());
        let app = test_app_with_sessions(&[session]).await;

        let resp = app.oneshot(auth_get("/api/chat/sessions")).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["items"][0]["conversation_id"], "conv-xyz");
    }

    // ====================================================================
    // workspace_slug & add_dirs in session responses
    // ====================================================================

    #[tokio::test]
    async fn test_get_session_includes_workspace_slug() {
        let mut session = test_chat_session(Some("my-proj"));
        session.workspace_slug = Some("my-workspace".into());
        let session_id = session.id;
        let app = test_app_with_sessions(&[session]).await;

        let resp = app
            .oneshot(auth_get(&format!("/api/chat/sessions/{}", session_id)))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["workspace_slug"], "my-workspace");
    }

    #[tokio::test]
    async fn test_get_session_includes_add_dirs() {
        let mut session = test_chat_session(Some("my-proj"));
        session.add_dirs = Some(vec!["/extra/a".into(), "/extra/b".into()]);
        let session_id = session.id;
        let app = test_app_with_sessions(&[session]).await;

        let resp = app
            .oneshot(auth_get(&format!("/api/chat/sessions/{}", session_id)))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let dirs = json["add_dirs"].as_array().unwrap();
        assert_eq!(dirs.len(), 2);
        assert_eq!(dirs[0], "/extra/a");
        assert_eq!(dirs[1], "/extra/b");
    }

    #[tokio::test]
    async fn test_get_session_workspace_null_by_default() {
        let session = test_chat_session(None);
        let session_id = session.id;
        let app = test_app_with_sessions(&[session]).await;

        let resp = app
            .oneshot(auth_get(&format!("/api/chat/sessions/{}", session_id)))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["workspace_slug"].is_null());
        assert!(json["add_dirs"].is_null());
    }

    #[tokio::test]
    async fn test_list_sessions_includes_workspace_and_add_dirs() {
        let mut s1 = test_chat_session(Some("proj-a"));
        s1.workspace_slug = Some("ws-1".into());
        s1.add_dirs = Some(vec!["/dir/x".into()]);
        let app = test_app_with_sessions(&[s1]).await;

        let resp = app.oneshot(auth_get("/api/chat/sessions")).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["items"][0]["workspace_slug"], "ws-1");
        assert_eq!(json["items"][0]["add_dirs"][0], "/dir/x");
    }

    // ====================================================================
    // Permission config helpers
    // ====================================================================

    /// Create an authenticated PUT request with JSON body
    fn auth_put(uri: &str, body: &str) -> Request<Body> {
        Request::builder()
            .method("PUT")
            .uri(uri)
            .header("content-type", "application/json")
            .header("authorization", test_bearer_token())
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    /// Build an OrchestratorState with a ChatManager (using mock backends)
    async fn mock_server_state_with_chat() -> OrchestratorState {
        let app_state = mock_app_state();
        let chat_config = crate::chat::config::ChatConfig::default();
        let chat_manager = Arc::new(crate::chat::ChatManager::new_without_memory(
            app_state.neo4j.clone(),
            app_state.meili.clone(),
            chat_config,
        ));
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: Some(chat_manager),
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            // A root account is configured; the usual test token is NOT root.
            auth_config: Some(crate::AuthConfig {
                root_account: Some(crate::RootAccountConfig {
                    email: ROOT_EMAIL.to_string(),
                    name: "Root".to_string(),
                    password_hash: "unused".to_string(),
                }),
                ..crate::test_helpers::test_auth_config()
            }),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            remote_mcp: crate::RemoteMcpConfig::default(),
            ingress: Default::default(),
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
            model_catalog: crate::chat::model_catalog::ModelCatalogCache::new(None),
            vault: crate::vault::VaultService::ephemeral(),
        })
    }

    /// Build a test router with ChatManager
    async fn test_app_with_chat() -> axum::Router {
        let state = mock_server_state_with_chat().await;
        create_router(state)
    }

    // ====================================================================
    // GET /api/chat/config/permissions — returns defaults
    // ====================================================================

    #[tokio::test]
    async fn test_get_chat_permissions_returns_defaults() {
        let app = test_app_with_chat().await;
        let resp = app
            .oneshot(auth_get("/api/chat/config/permissions"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["mode"], "default");
        // MCP tools are pre-approved by default
        let allowed = json["allowed_tools"].as_array().unwrap();
        assert_eq!(allowed.len(), 1);
        assert_eq!(allowed[0], "mcp__project-orchestrator__*");
        assert_eq!(json["disallowed_tools"].as_array().unwrap().len(), 0);
        // default_model is now included from ChatConfig
        assert!(
            json["default_model"].is_string(),
            "default_model should be present"
        );
    }

    // ====================================================================
    // GET /api/chat/config/permissions — no chat_manager returns 500
    // ====================================================================

    #[tokio::test]
    async fn test_get_chat_permissions_no_chat_manager() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/chat/config/permissions"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    // ====================================================================
    // PUT /api/chat/config/permissions — updates config
    // ====================================================================

    #[tokio::test]
    async fn test_update_chat_permissions_changes_mode() {
        let state = mock_server_state_with_chat().await;
        let app = create_router(state.clone());

        // PUT to update
        let resp = app
            .oneshot(auth_put(
                "/api/chat/config/permissions",
                r#"{"mode":"default","allowed_tools":["Read","Bash(git *)"],"disallowed_tools":[]}"#,
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["mode"], "default");
        assert_eq!(json["allowed_tools"][0], "Read");
        assert_eq!(json["allowed_tools"][1], "Bash(git *)");

        // Verify GET reflects the update
        let app2 = create_router(state);
        let resp = app2
            .oneshot(auth_get("/api/chat/config/permissions"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["mode"], "default");
        assert_eq!(json["allowed_tools"].as_array().unwrap().len(), 2);
    }

    // ====================================================================
    // PUT /api/chat/config/permissions — rejects invalid mode
    // ====================================================================

    #[tokio::test]
    async fn test_update_chat_permissions_rejects_invalid_mode() {
        let app = test_app_with_chat().await;

        let resp = app
            .oneshot(auth_put(
                "/api/chat/config/permissions",
                r#"{"mode":"yolo","allowed_tools":[],"disallowed_tools":[]}"#,
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("Invalid permission mode"));
    }

    // ====================================================================
    // PUT /api/chat/config/permissions — partial JSON uses serde defaults
    // ====================================================================

    #[tokio::test]
    async fn test_update_chat_permissions_partial_json() {
        let app = test_app_with_chat().await;

        // Only set mode — allowed_tools defaults to MCP tools, disallowed_tools defaults to empty
        let resp = app
            .oneshot(auth_put(
                "/api/chat/config/permissions",
                r#"{"mode":"acceptEdits"}"#,
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["mode"], "acceptEdits");
        // MCP tools pre-approved by default when allowed_tools is not specified
        let allowed = json["allowed_tools"].as_array().unwrap();
        assert_eq!(allowed.len(), 1);
        assert_eq!(allowed[0], "mcp__project-orchestrator__*");
        assert_eq!(json["disallowed_tools"].as_array().unwrap().len(), 0);
    }
    /// The model catalog must be reachable WITHOUT credentials.
    ///
    /// The setup wizard (`/setup`) runs before login, so while this route sat
    /// behind `require_auth` the wizard's fetch 401'd and the selector could
    /// only ever render a hardcoded fallback list. Now that the frontend keeps
    /// no such list, an authenticated route here would leave that screen with
    /// no models at all.
    #[tokio::test]
    async fn test_model_catalog_is_publicly_reachable() {
        let app = test_app().await;
        let anonymous = Request::builder()
            .uri("/api/chat/models")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(anonymous).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/api/chat/models must not require auth"
        );
    }

    /// The payload must stay free of presentation values.
    ///
    /// A Tailwind class shipped from the backend is purged from the production
    /// bundle (Tailwind v4 here runs with no config and no safelist, and never
    /// scans .rs files), so the dots would silently lose their color in prod
    /// only. Colors belong to the frontend; the API sends `family` instead.
    #[tokio::test]
    async fn test_model_catalog_payload_is_semantic_not_presentational() {
        let app = test_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/chat/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let raw = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(
            !raw.contains("bg-"),
            "no Tailwind class may cross the wire: {raw}"
        );

        let models: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let first = &models.as_array().expect("array")[0];
        for field in ["id", "family", "version", "tier", "shortLabel", "fullLabel"] {
            assert!(first.get(field).is_some(), "missing field {field}: {first}");
        }
        assert_eq!(first["id"], "claude-opus-5-5");
        assert_eq!(first["family"], "opus");
        assert_eq!(first["version"], "5.5");
        assert_eq!(first["tier"], "current");
        assert_eq!(first["shortLabel"], "Opus 5.5");
        assert_eq!(first["fullLabel"], "Claude Opus 5.5");
    }

    // ====================================================================
    // Action routes: POST .../permissions/{request_id} and POST .../messages
    // ====================================================================

    use crate::chat::manager::{test_support, ChatManager};
    use crate::chat::types::ChatEvent;
    use crate::neo4j::models::ChatEventRecord;

    struct ActionHarness {
        app: axum::Router,
        manager: Arc<ChatManager>,
        graph: Arc<dyn crate::neo4j::traits::GraphStore>,
        mock: Arc<crate::neo4j::mock::MockGraphStore>,
    }

    async fn action_harness(cli_path: Option<&str>) -> ActionHarness {
        action_harness_with(cli_path, true).await
    }

    async fn action_harness_with(cli_path: Option<&str>, refs_v1: bool) -> ActionHarness {
        action_harness_keeping(cli_path, refs_v1, None).await
    }

    /// [`action_harness_with`], the lasting approvals kept in `lasting_rules_dir`.
    async fn action_harness_keeping(
        cli_path: Option<&str>,
        refs_v1: bool,
        lasting_rules_dir: Option<std::path::PathBuf>,
    ) -> ActionHarness {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let app_state = crate::test_helpers::mock_app_state_with_graph(mock.clone());
        let graph = app_state.neo4j.clone();
        let mut config = test_support::chat_config();
        config.claude_cli_path = cli_path.map(str::to_string);
        config.lasting_rules_dir = lasting_rules_dir;
        let manager = Arc::new(
            ChatManager::new_without_memory(
                app_state.neo4j.clone(),
                app_state.meili.clone(),
                config,
            )
            .with_refs_v1(refs_v1),
        );
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let base = mock_server_state().await;
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: Some(manager.clone()),
            event_bus: base.event_bus.clone(),
            nats_emitter: None,
            // A root account is configured; the usual test token is NOT root.
            auth_config: Some(crate::AuthConfig {
                root_account: Some(crate::RootAccountConfig {
                    email: ROOT_EMAIL.to_string(),
                    name: "Root".to_string(),
                    password_hash: "unused".to_string(),
                }),
                ..crate::test_helpers::test_auth_config()
            }),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            remote_mcp: crate::RemoteMcpConfig::default(),
            ingress: Default::default(),
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
            model_catalog: crate::chat::model_catalog::ModelCatalogCache::new(None),
            vault: crate::vault::VaultService::ephemeral(),
        });
        ActionHarness {
            app: create_router(state),
            manager,
            graph,
            mock,
        }
    }

    async fn seed_session(h: &ActionHarness) -> Uuid {
        let s = test_chat_session(None);
        h.graph.create_chat_session(&s).await.unwrap();
        s.id
    }

    async fn seed_event(h: &ActionHarness, sid: Uuid, seq: i64, kind: &str, ev: ChatEvent) {
        h.graph
            .store_chat_events(
                sid,
                vec![ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: sid,
                    seq,
                    event_type: kind.to_string(),
                    data: serde_json::to_string(&ev).unwrap(),
                    created_at: chrono::Utc::now(),
                }],
            )
            .await
            .unwrap();
    }

    async fn seed_permission_request(h: &ActionHarness, sid: Uuid, seq: i64, id: &str) {
        seed_event(
            h,
            sid,
            seq,
            "permission_request",
            ChatEvent::PermissionRequest {
                id: id.to_string(),
                tool: "Bash".to_string(),
                input: serde_json::json!({"command": "ls"}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
                tool_use_id: None,
                always: None,
                consent: None,
            },
        )
        .await;
    }

    async fn call(app: &axum::Router, req: Request<Body>) -> (StatusCode, serde_json::Value) {
        let res = app.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), 64 * 1024)
            .await
            .unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        )
    }

    fn perm_uri(sid: Uuid, rid: &str) -> String {
        format!("/api/chat/sessions/{sid}/permissions/{rid}")
    }

    #[tokio::test]
    async fn permission_to_a_live_session_reaches_the_cli_then_a_second_answer_is_409() {
        let h = action_harness(None).await;
        let sid = seed_session(&h).await;
        seed_permission_request(&h, sid, 1, "req-1").await;
        let (mut stdin, _) =
            test_support::insert_live_session(&h.manager, &sid.to_string(), false, &["req-1"])
                .await;

        let (status, body) = call(
            &h.app,
            auth_post(&perm_uri(sid, "req-1"), r#"{"allow":true}"#),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["routed"], "local");
        let written: serde_json::Value =
            serde_json::from_str(&stdin.try_recv().expect("the CLI got the answer")).unwrap();
        assert_eq!(written["response"]["request_id"], "req-1");
        assert_eq!(written["response"]["response"]["behavior"], "allow");
        assert_eq!(
            written["response"]["response"]["updatedInput"]["command"], "ls",
            "the original tool input is handed back"
        );

        // Double click / second tab: refused, nothing more reaches the CLI.
        let (status, body) = call(
            &h.app,
            auth_post(&perm_uri(sid, "req-1"), r#"{"allow":false}"#),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}");
        assert!(
            stdin.try_recv().is_err(),
            "a second answer must not be sent"
        );
    }

    #[tokio::test]
    async fn permission_deny_is_forwarded_as_a_deny() {
        let h = action_harness(None).await;
        let sid = seed_session(&h).await;
        seed_permission_request(&h, sid, 1, "req-d").await;
        let (mut stdin, _) =
            test_support::insert_live_session(&h.manager, &sid.to_string(), false, &["req-d"])
                .await;
        let (status, _) = call(
            &h.app,
            auth_post(&perm_uri(sid, "req-d"), r#"{"allow":false}"#),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let written: serde_json::Value = serde_json::from_str(&stdin.try_recv().unwrap()).unwrap();
        assert_eq!(written["response"]["response"]["behavior"], "deny");
    }

    #[tokio::test]
    async fn permission_already_claimed_but_not_yet_persisted_is_409_not_a_second_send() {
        // The request is stored without a decision and the live CLI no longer
        // holds it pending: another answer claimed it a moment ago.
        let h = action_harness(None).await;
        let sid = seed_session(&h).await;
        seed_permission_request(&h, sid, 1, "req-race").await;
        let (mut stdin, _) =
            test_support::insert_live_session(&h.manager, &sid.to_string(), false, &[]).await;
        let (status, _) = call(
            &h.app,
            auth_post(&perm_uri(sid, "req-race"), r#"{"allow":true}"#),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert!(stdin.try_recv().is_err());
    }

    #[tokio::test]
    async fn permission_never_asked_on_a_live_session_is_404() {
        let h = action_harness(None).await;
        let sid = seed_session(&h).await;
        let _live =
            test_support::insert_live_session(&h.manager, &sid.to_string(), false, &[]).await;
        let (status, _) = call(
            &h.app,
            auth_post(&perm_uri(sid, "nope"), r#"{"allow":true}"#),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn permission_to_a_dead_session_is_410_with_a_readable_reason() {
        let h = action_harness(None).await;
        let sid = seed_session(&h).await;
        seed_permission_request(&h, sid, 1, "req-orphan").await;

        let (status, body) = call(
            &h.app,
            auth_post(&perm_uri(sid, "req-orphan"), r#"{"allow":true}"#),
        )
        .await;
        assert_eq!(status, StatusCode::GONE, "never a silent 204: {body}");
        let reason = body["error"].as_str().unwrap();
        assert!(reason.contains("s'est arrêté"), "{reason}");
        assert!(reason.contains("continue par un message"), "{reason}");
    }

    /// P11b: a person lists and revokes THEIR lasting approvals, signed in; another person
    /// sees none of them and cannot revoke them; an agent's token is refused.
    #[tokio::test]
    async fn a_person_lists_and_revokes_only_their_own_lasting_rules() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("rules");
        let h = action_harness_keeping(None, true, Some(root.clone())).await;
        let (alice, bob) = (Uuid::new_v4(), Uuid::new_v4());
        let store = crate::chat::lasting_rules::LastingRuleStore::new(root);
        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();
        let place = crate::chat::lasting_rules::LastingScope::of_session(
            Some(&store),
            Some(&alice.to_string()),
            Some("proj"),
            Some(&project_root),
            &project_root,
        )
        .unwrap();
        let asked = crate::chat::session_grants::AskedCall {
            asker: crate::chat::session_grants::Asker::ClaudeCode,
            tool: "Bash".into(),
            canonical: None,
            input: serde_json::json!({"command": "cat README.md"}),
        };
        let grant = crate::chat::session_grants::lasting_grant_for(&asked).unwrap();
        let rule = place.keep(&asked, &grant, "s1").unwrap();

        let as_person = |method: &str, uri: String, person: Uuid| {
            let token = crate::auth::jwt::encode_jwt(
                person,
                "someone@ffs.holdings",
                "Someone",
                "test-secret-key-minimum-32-chars!!",
                28800,
            )
            .unwrap();
            Request::builder()
                .method(method)
                .uri(uri)
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap()
        };
        let list = "/api/chat/permission-rules".to_string();
        let (status, body) = call(&h.app, as_person("GET", list.clone(), alice)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["enabled"], true);
        assert_eq!(body["rules"][0]["rule"], "Bash: cat README.md", "{body}");
        assert_eq!(body["rules"][0]["engine"], "claude_code", "{body}");
        assert_eq!(body["rules"][0]["holds"], true, "{body}");
        let (_, body) = call(
            &h.app,
            as_person("GET", format!("{list}?project=other"), alice),
        )
        .await;
        assert_eq!(body["rules"], serde_json::json!([]), "{body}");
        // Bob sees nothing of Alice's, and cannot revoke her rule.
        let (_, body) = call(&h.app, as_person("GET", list.clone(), bob)).await;
        assert_eq!(body["rules"], serde_json::json!([]), "{body}");
        let revoke = format!("/api/chat/permission-rules/{}", rule.id);
        let (status, _) = call(&h.app, as_person("DELETE", revoke.clone(), bob)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(store.rules(&alice.to_string()).len(), 1);
        // Alice revokes it: gone, and a second revoke is 404.
        let (status, _) = call(&h.app, as_person("DELETE", revoke.clone(), alice)).await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        assert!(store.rules(&alice.to_string()).is_empty());
        let (status, _) = call(&h.app, as_person("DELETE", revoke, alice)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        // Not signed in: 401.
        let anonymous = Request::builder()
            .method("GET")
            .uri(list)
            .body(Body::empty())
            .unwrap();
        let (status, _) = call(&h.app, anonymous).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    /// Review of #714, S1: an MCP token of the session's owner may answer once or for the
    /// session, never `always` (403, nothing answered); the person's own session token goes
    /// through (here 410: the CLI is gone).
    #[tokio::test]
    async fn an_mcp_token_cannot_allow_a_call_for_good() {
        let h = action_harness(None).await;
        let owner = Uuid::new_v4();
        let session = crate::neo4j::models::ChatSessionNode {
            file_scope_lifted: false,
            owner: Some(owner.to_string()),
            ..test_chat_session(None)
        };
        h.graph.create_chat_session(&session).await.unwrap();
        let sid = session.id;
        seed_permission_request(&h, sid, 1, "req-mcp").await;
        let secret = "test-secret-key-minimum-32-chars!!";
        let (mcp, jti) = crate::auth::jwt::encode_mcp_token(
            owner,
            "owner@ffs.holdings",
            "Owner",
            "mcp:read mcp:write",
            secret,
            3600,
        )
        .unwrap();
        h.graph
            .create_mcp_token(
                owner,
                &jti,
                "client",
                "mcp:read mcp:write",
                chrono::Utc::now() + chrono::Duration::hours(1),
            )
            .await
            .unwrap();
        let human =
            crate::auth::jwt::encode_jwt(owner, "owner@ffs.holdings", "Owner", secret, 3600)
                .unwrap();
        let post = |token: &str, scope: &str| {
            Request::builder()
                .method("POST")
                .uri(perm_uri(sid, "req-mcp"))
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(format!(r#"{{"allow":true,"scope":"{scope}"}}"#)))
                .unwrap()
        };
        let (status, body) = call(&h.app, post(&mcp, "always")).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
        let events = h.graph.get_attention_events(&[sid]).await.unwrap();
        assert_eq!(
            crate::chat::attention::permission_status(&events, "req-mcp"),
            crate::chat::attention::PermissionStatus::Pending,
        );
        // The same token answering for the session is not refused by this rule.
        let (status, body) = call(&h.app, post(&mcp, "session")).await;
        assert_eq!(status, StatusCode::GONE, "{body}");
        let (status, body) = call(&h.app, post(&human, "always")).await;
        assert_eq!(status, StatusCode::GONE, "the person goes through: {body}");
    }

    /// An agent's token, an MCP token, the server's account, the anonymous user: never
    /// a person managing lasting approvals.
    #[test]
    fn only_a_person_signed_in_manages_lasting_rules() {
        use crate::auth::jwt::Claims;
        let person = Claims {
            sub: Uuid::new_v4().to_string(),
            email: "a@ffs.holdings".into(),
            name: "A".into(),
            iat: 0,
            exp: 0,
            token_type: None,
            scope: None,
            jti: None,
        };
        let ext = |c: Claims| Some(axum::Extension(c));
        assert_eq!(
            lasting_rules_person(&ext(person.clone())).unwrap(),
            person.sub
        );
        for kind in ["agent_session", "mcp", "vault"] {
            let c = Claims {
                token_type: Some(kind.into()),
                ..person.clone()
            };
            assert!(
                matches!(lasting_rules_person(&ext(c)), Err(AppError::Forbidden(_))),
                "{kind}"
            );
        }
        assert!(matches!(
            lasting_rules_person(&ext(Claims::service_account("po"))),
            Err(AppError::Forbidden(_))
        ));
        assert!(matches!(
            lasting_rules_person(&ext(Claims::anonymous())),
            Err(AppError::Forbidden(_))
        ));
        assert!(matches!(
            lasting_rules_person(&None),
            Err(AppError::Unauthorized(_))
        ));
    }

    /// Review #679 finding 6: a person answers only the permission requests of THEIR
    /// conversation. Another signed-in user gets 403 and nothing is answered (the request
    /// still waits for its owner); the owner goes through (here 410: the CLI is gone).
    #[tokio::test]
    async fn only_the_owner_of_a_session_answers_its_permission_requests() {
        let h = action_harness(None).await;
        let owner = Uuid::new_v4();
        let session = crate::neo4j::models::ChatSessionNode {
            file_scope_lifted: false,
            owner: Some(owner.to_string()),
            ..test_chat_session(None)
        };
        h.graph.create_chat_session(&session).await.unwrap();
        let sid = session.id;
        seed_permission_request(&h, sid, 1, "req-owned").await;
        let post_as = |person: Uuid| {
            let token = crate::auth::jwt::encode_jwt(
                person,
                "someone@ffs.holdings",
                "Someone",
                "test-secret-key-minimum-32-chars!!",
                28800,
            )
            .unwrap();
            Request::builder()
                .method("POST")
                .uri(perm_uri(sid, "req-owned"))
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(r#"{"allow":true,"scope":"session"}"#))
                .unwrap()
        };

        let (status, body) = call(&h.app, post_as(Uuid::new_v4())).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
        // Nothing was decided for the owner.
        let events = h.graph.get_attention_events(&[sid]).await.unwrap();
        assert_eq!(
            crate::chat::attention::permission_status(&events, "req-owned"),
            crate::chat::attention::PermissionStatus::Pending,
        );

        let (status, body) = call(&h.app, post_as(owner)).await;
        assert_eq!(status, StatusCode::GONE, "the owner is let through: {body}");
    }

    /// A session nobody owns (opened by the server itself, or before owners existed) is
    /// answered as before: by any signed-in person.
    #[tokio::test]
    async fn a_session_without_an_owner_is_answered_by_any_person() {
        let h = action_harness(None).await;
        let sid = seed_session(&h).await;
        seed_permission_request(&h, sid, 1, "req-free").await;
        let (status, body) = call(
            &h.app,
            auth_post(&perm_uri(sid, "req-free"), r#"{"allow":true}"#),
        )
        .await;
        assert_eq!(status, StatusCode::GONE, "{body}");
    }

    #[tokio::test]
    async fn permission_already_decided_on_a_dead_session_is_409() {
        let h = action_harness(None).await;
        let sid = seed_session(&h).await;
        seed_permission_request(&h, sid, 1, "req-x").await;
        seed_event(
            &h,
            sid,
            2,
            "permission_decision",
            ChatEvent::PermissionDecision {
                id: "req-x".into(),
                allow: true,
                scope: None,
                rule: None,
            },
        )
        .await;
        let (status, _) = call(
            &h.app,
            auth_post(&perm_uri(sid, "req-x"), r#"{"allow":true}"#),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn permission_to_an_unknown_session_is_404() {
        let h = action_harness(None).await;
        let (status, _) = call(
            &h.app,
            auth_post(&perm_uri(Uuid::new_v4(), "r"), r#"{"allow":true}"#),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn permission_rejects_a_body_without_allow() {
        let h = action_harness(None).await;
        let sid = seed_session(&h).await;
        let (status, _) = call(
            &h.app,
            auth_post(&perm_uri(sid, "r"), r#"{"allowed":true}"#),
        )
        .await;
        assert!(status.is_client_error(), "{status}");
    }

    #[tokio::test]
    async fn action_routes_require_authentication() {
        let h = action_harness(None).await;
        let sid = Uuid::new_v4();
        for (uri, body) in [
            (perm_uri(sid, "r"), r#"{"allow":true}"#),
            (
                format!("/api/chat/sessions/{sid}/messages"),
                r#"{"content":"x"}"#,
            ),
        ] {
            let req = Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap();
            let (status, _) = call(&h.app, req).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED);
        }
    }

    fn msg_uri(sid: Uuid) -> String {
        format!("/api/chat/sessions/{sid}/messages")
    }

    /// A parent session and its read-only mirror of a Claude Code sub-agent (P18c).
    async fn seed_mirror(h: &ActionHarness) -> (Uuid, Uuid) {
        let parent = seed_session(h).await;
        let mirror = crate::neo4j::models::ChatSessionNode {
            mirror_of: Some(crate::chat::mirror::MIRROR_OF_CLAUDE_CODE.to_string()),
            spawned_by: Some(
                crate::chat::types::SpawnedBy::Conversation {
                    parent_session_id: parent,
                    tool_use_id: Some("toolu_task".into()),
                }
                .to_json_string(),
            ),
            ..test_chat_session(None)
        };
        h.graph.create_chat_session(&mirror).await.unwrap();
        (parent, mirror.id)
    }

    /// P18c: every REST write to a mirror is refused with 409 `mirror_read_only` — a
    /// message, a message through `POST /api/chat/sessions` (`chat(send_message)` with a
    /// `session_id`), a permission answer — and nothing reaches its thread.
    #[tokio::test]
    async fn a_mirror_refuses_every_rest_write_with_409_mirror_read_only() {
        let h = action_harness(None).await;
        let (_parent, mirror) = seed_mirror(&h).await;
        for req in [
            auth_post(&msg_uri(mirror), r#"{"content":"hello"}"#),
            auth_post(
                "/api/chat/sessions",
                &serde_json::json!({"message": "hello", "session_id": mirror.to_string()})
                    .to_string(),
            ),
            auth_post(&perm_uri(mirror, "r1"), r#"{"allow":true}"#),
            // M1: a move to another provider would open a writable session holding the call.
            auth_post(
                &format!("/api/chat/sessions/{mirror}/switch-provider"),
                r#"{"provider":"claude-code","message":"go on"}"#,
            ),
        ] {
            let (status, body) = call(&h.app, req).await;
            assert_eq!(status, StatusCode::CONFLICT, "{body}");
            assert_eq!(body["code"], "mirror_read_only", "{body}");
            assert_eq!(body["retryable"], false, "{body}");
        }
        assert!(h
            .graph
            .get_chat_events(mirror, 0, 100)
            .await
            .unwrap()
            .is_empty());
        assert!(!h.manager.is_session_active(&mirror.to_string()).await);
        assert_eq!(
            h.graph.get_session_children(_parent).await.unwrap().len(),
            1,
            "no second child holds the call"
        );
    }

    /// P18c (M7): deleting an OPEN mirror ends it for its tap — never written again, never
    /// written anew for its call — and it is no longer live.
    #[tokio::test]
    async fn deleting_an_open_mirror_stops_its_tap() {
        let h = action_harness(None).await;
        let parent = seed_session(&h).await;
        let tap = h.manager.mirror_tap(&parent.to_string()).expect("a tap");
        tap.observe(&ChatEvent::ToolUse {
            id: "toolu_task".into(),
            tool: "Task".into(),
            input: serde_json::json!({"prompt": "go"}),
            parent_tool_use_id: None,
            category: None,
            canonical: None,
        });
        let said = |t: &str| ChatEvent::AssistantText {
            content: t.into(),
            parent_tool_use_id: Some("toolu_task".into()),
        };
        tap.observe(&said("working"));
        tap.flushed().await;
        let mirror =
            crate::chat::mirror::test_support::mirrors_of(h.graph.as_ref(), parent).await[0].id;
        assert!(h.manager.is_session_streaming(&mirror.to_string()).await);

        let (status, body) = call(
            &h.app,
            Request::builder()
                .method("DELETE")
                .uri(format!("/api/chat/sessions/{mirror}"))
                .header("authorization", test_bearer_token())
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(!h.manager.is_session_streaming(&mirror.to_string()).await);
        tap.observe(&said("after the delete"));
        tap.flushed().await;
        assert!(h.graph.get_chat_session(mirror).await.unwrap().is_none());
        assert!(
            !crate::chat::mirror::test_support::thread(h.graph.as_ref(), mirror)
                .await
                .iter()
                .any(|e| matches!(e, ChatEvent::AssistantText { content, .. } if content == "after the delete")),
            "never written again"
        );
        assert!(
            crate::chat::mirror::test_support::mirrors_of(h.graph.as_ref(), parent)
                .await
                .is_empty(),
            "not written anew for its call"
        );
    }

    /// P18c: the session object (GET) and the parent's children carry `mirror_of`; a normal
    /// session leaves the field off the wire.
    #[tokio::test]
    async fn a_mirror_says_so_in_the_session_and_the_children() {
        let h = action_harness(None).await;
        let (parent, mirror) = seed_mirror(&h).await;
        let (status, body) = call(&h.app, auth_get(&format!("/api/chat/sessions/{mirror}"))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["mirror_of"], "claude_code", "{body}");
        assert_eq!(body["spawned_by"]["tool_use_id"], "toolu_task", "{body}");
        let (status, body) = call(
            &h.app,
            auth_get(&format!("/api/chat/sessions/{parent}/children")),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let children = body
            .as_array()
            .cloned()
            .unwrap_or_else(|| body["items"].as_array().cloned().unwrap_or_default());
        assert_eq!(children.len(), 1, "{body}");
        assert_eq!(children[0]["mirror_of"], "claude_code", "{body}");
        let (_, body) = call(&h.app, auth_get(&format!("/api/chat/sessions/{parent}"))).await;
        assert!(body.get("mirror_of").is_none(), "{body}");
    }

    /// Store a one-chunk document so a message can refer to it.
    async fn seed_document(h: &ActionHarness) -> Uuid {
        use crate::documents::DocumentFormat;
        use crate::neo4j::document::{Document, DocumentChunk};
        let id = Uuid::new_v4();
        let doc = Document {
            id,
            filename: "notes.txt".to_string(),
            format: DocumentFormat::PlainText,
            sha256: "0".repeat(64),
            size_bytes: 5,
            page_count: 0,
            chunk_count: 1,
            warnings: vec![],
            created_at: chrono::Utc::now(),
            project_id: None,
            session_id: None,
            extracted: true,
            mime_type: Some("text/plain".to_string()),
        };
        let chunk = DocumentChunk {
            id: Uuid::new_v4(),
            text: "hello".to_string(),
            start_byte: 0,
            end_byte: 5,
            page: None,
            ordinal: 0,
            embedding: None,
        };
        h.graph.create_document(&doc, &[chunk]).await.unwrap();
        id
    }

    #[tokio::test]
    async fn message_naming_an_unknown_attachment_is_refused_not_sent_without_it() {
        let h = action_harness(Some("/nonexistent/claude-cli")).await;
        let sid = seed_session(&h).await;
        let ghost = Uuid::new_v4();
        let body = format!(r#"{{"content":"lis ça","attachments":["{ghost}"]}}"#);
        let (status, resp) = call(&h.app, auth_post(&msg_uri(sid), &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{resp}");
        assert!(resp.to_string().contains("does not exist"), "{resp}");
    }

    #[tokio::test]
    async fn message_with_a_known_attachment_gets_past_validation_to_delivery() {
        // The id exists, so the request is not a 400: it proceeds to delivery,
        // which fails here only because there is no CLI to resume.
        let h = action_harness(Some("/nonexistent/claude-cli")).await;
        let sid = seed_session(&h).await;
        let doc = seed_document(&h).await;
        let body = format!(r#"{{"content":"lis ça","attachments":["{doc}"]}}"#);
        let (status, resp) = call(&h.app, auth_post(&msg_uri(sid), &body)).await;
        assert_ne!(status, StatusCode::BAD_REQUEST, "{resp}");
        assert_ne!(status, StatusCode::UNPROCESSABLE_ENTITY, "{resp}");
    }

    // ----- references (`refs_v1`) over REST -----

    fn errors_fixture() -> serde_json::Value {
        let path = format!(
            "{}/tests/fixtures/refs/errors.json",
            env!("CARGO_MANIFEST_DIR")
        );
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn fixture_case(name: &str) -> serde_json::Value {
        errors_fixture()["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap_or_else(|| panic!("no fixture case {name}"))["body"]
            .clone()
    }

    #[tokio::test]
    async fn a_message_with_invalid_refs_is_a_400_with_the_body_of_the_fixture() {
        let h = action_harness(Some("/nonexistent/claude-cli")).await;
        let sid = seed_session(&h).await;
        let plan = Uuid::new_v4();
        // unknown kind at index 2
        let body = format!(
            r#"{{"content":"x","refs":[{{"kind":"plan","id":"{plan}"}},{{"kind":"note","id":"{plan}"}},{{"kind":"step","id":"{plan}"}}]}}"#
        );
        let (status, resp) = call(&h.app, auth_post(&msg_uri(sid), &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{resp}");
        assert_eq!(resp, fixture_case("unknown kind at index 2"));

        // more than 20
        let many: Vec<_> = (1..=21u128)
            .map(|n| serde_json::json!({"kind": "task", "id": Uuid::from_u128(n)}))
            .collect();
        let body = serde_json::json!({"content": "x", "refs": many}).to_string();
        let (status, resp) = call(&h.app, auth_post(&msg_uri(sid), &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(resp, fixture_case("too many refs"));

        // a link that is not allowed, bad id
        let body = r#"{"content":"x","refs":[{"kind":"link","id":"javascript:alert(1)"}]}"#;
        let (_, resp) = call(&h.app, auth_post(&msg_uri(sid), body)).await;
        assert_eq!(resp, fixture_case("bad link"));
        let body =
            r#"{"content":"x","refs":[{"kind":"plan","id":"nope"},{"kind":"plan","id":"nope"}]}"#;
        let (_, resp) = call(&h.app, auth_post(&msg_uri(sid), body)).await;
        assert_eq!(resp["reason"], "bad_id");
        assert_eq!(resp["index"], 0);
    }

    #[tokio::test]
    async fn a_refused_message_leaves_no_trace_in_the_graph() {
        // A message refused with a 400 must not have fed the knowledge graph
        // (DISCUSSED relations, neural reinforcement) on its way out.
        let h = action_harness(Some("/nonexistent/claude-cli")).await;
        let sid = seed_session(&h).await;
        let text = "regarde src/main.rs et Cargo.toml";
        let body = serde_json::json!({
            "content": text,
            "refs": [{"kind": "step", "id": Uuid::new_v4()}]
        })
        .to_string();
        let (status, _) = call(&h.app, auth_post(&msg_uri(sid), &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert!(
            h.mock.discussed_calls.read().await.is_empty(),
            "a refused message must not write DISCUSSED relations"
        );

        // Control: the same words, accepted, DO reach the extraction (the check
        // above is not vacuous).
        let body = serde_json::json!({"content": text}).to_string();
        let _ = call(&h.app, auth_post(&msg_uri(sid), &body)).await;
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert!(!h.mock.discussed_calls.read().await.is_empty());
    }

    #[tokio::test]
    async fn the_first_message_with_invalid_refs_creates_no_session() {
        let h = action_harness(Some("/nonexistent/claude-cli")).await;
        let body = r#"{"message":"x","cwd":"/tmp","refs":[{"kind":"link","id":"http://127.0.0.1/admin"}]}"#;
        let (status, resp) = call(&h.app, auth_post("/api/chat/sessions", body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{resp}");
        assert_eq!(resp["code"], "refs_invalid");
        assert_eq!(resp["reason"], "bad_link");
        let sessions = h
            .graph
            .list_chat_sessions(None, None, 10, 0, true)
            .await
            .unwrap();
        assert!(sessions.0.is_empty());
    }

    #[tokio::test]
    async fn with_the_switch_off_refs_are_ignored_like_on_an_older_server() {
        let h = action_harness_with(Some("/nonexistent/claude-cli"), false).await;
        let sid = seed_session(&h).await;
        // The same body that is a 400 with the switch on is not refused…
        let body = r#"{"content":"x","refs":[{"kind":"step","id":"nope"}]}"#;
        let (status, resp) = call(&h.app, auth_post(&msg_uri(sid), body)).await;
        // Not refused: it went on to delivery, which fails here only because there
        // is no CLI to resume (a 500), exactly like a message without refs.
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{resp}");
        // …and a valid ref is not folded into the text: the CLI sees the text only.
        let mut cli = test_support::insert_mock_cli_session(&h.manager, &sid.to_string()).await;
        let plan = Uuid::new_v4();
        let body = format!(r#"{{"content":"just text","refs":[{{"kind":"plan","id":"{plan}"}}]}}"#);
        let (status, _) = call(&h.app, auth_post(&msg_uri(sid), &body)).await;
        assert_eq!(status, StatusCode::OK);
        let sent =
            tokio::time::timeout(std::time::Duration::from_secs(10), cli.sent_input_rx.recv())
                .await
                .unwrap()
                .unwrap();
        assert_eq!(sent.message["content"], "just text");
    }

    #[tokio::test]
    async fn a_message_with_refs_reaches_the_cli_expanded_through_the_rest_route() {
        let h = action_harness(Some("/nonexistent/claude-cli")).await;
        let sid = seed_session(&h).await;
        let task = {
            let plan = crate::neo4j::models::PlanNode::new_for_project(
                "p".into(),
                "d".into(),
                "t".into(),
                5,
                Uuid::new_v4(),
            );
            h.graph.create_plan(&plan).await.unwrap();
            let task = crate::refs::test_support::task_titled(Some("Tâche REST"), "corps");
            h.graph.create_task(plan.id, &task).await.unwrap();
            task
        };
        let mut cli = test_support::insert_mock_cli_session(&h.manager, &sid.to_string()).await;
        let body = serde_json::json!({
            "content": "regarde #task:x",
            "refs": [{"kind": "task", "id": task.id}]
        })
        .to_string();
        let (status, resp) = call(&h.app, auth_post(&msg_uri(sid), &body)).await;
        assert_eq!(status, StatusCode::OK, "{resp}");
        let sent =
            tokio::time::timeout(std::time::Duration::from_secs(10), cli.sent_input_rx.recv())
                .await
                .unwrap()
                .unwrap();
        let prompt = sent.message["content"].as_str().unwrap().to_string();
        assert!(prompt.starts_with("regarde #task:x"), "{prompt}");
        assert!(
            prompt.contains("<po-context") && prompt.contains("Tâche REST"),
            "{prompt}"
        );
        assert!(!prompt.contains("<po-refs>"), "{prompt}");
    }

    #[tokio::test]
    async fn the_version_endpoint_announces_refs_v1_only_while_it_is_on() {
        for on in [true, false] {
            let h = action_harness_with(None, on).await;
            let req = Request::builder()
                .uri("/api/version")
                .body(Body::empty())
                .unwrap();
            let (status, resp) = call(&h.app, req).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(resp["features"]["refs_v1"], on, "{resp}");
        }
    }

    /// AM-1: the endpoint says which commit runs (`build.git_sha`, baked by
    /// build.rs), so `cargo xtask in-service` can compare it with main. The
    /// key is always present: a full sha from a checkout, `null` without git.
    #[tokio::test]
    async fn the_version_endpoint_exposes_the_commit_it_was_built_from() {
        let h = action_harness_with(None, true).await;
        let req = Request::builder()
            .uri("/api/version")
            .body(Body::empty())
            .unwrap();
        let (status, resp) = call(&h.app, req).await;
        assert_eq!(status, StatusCode::OK);
        let build = resp["build"].as_object().expect("build object");
        assert!(build.contains_key("git_sha"), "{resp}");
        match crate::update::version::git_sha() {
            Some(sha) => {
                assert_eq!(resp["build"]["git_sha"], sha, "{resp}");
                assert_eq!(sha.len(), 40, "{sha}");
            }
            None => assert!(resp["build"]["git_sha"].is_null(), "{resp}"),
        }
    }

    #[tokio::test]
    async fn first_message_naming_an_unknown_attachment_creates_no_session() {
        let h = action_harness(Some("/nonexistent/claude-cli")).await;
        let ghost = Uuid::new_v4();
        let body = format!(r#"{{"message":"lis ça","cwd":"/tmp","attachments":["{ghost}"]}}"#);
        let (status, resp) = call(&h.app, auth_post("/api/chat/sessions", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{resp}");
        let sessions = h
            .graph
            .list_chat_sessions(None, None, 10, 0, true)
            .await
            .unwrap();
        assert!(
            sessions.0.is_empty(),
            "no session may be created for a refused message"
        );
    }

    #[tokio::test]
    async fn message_to_a_live_session_goes_through_the_same_send_path() {
        let h = action_harness(None).await;
        let sid = seed_session(&h).await;
        // Streaming: send_message queues the text for the running turn.
        let (_stdin, queue) =
            test_support::insert_live_session(&h.manager, &sid.to_string(), true, &[]).await;
        let (status, body) = call(&h.app, auth_post(&msg_uri(sid), r#"{"content":"vas-y"}"#)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["routed"], "local");
        assert_eq!(queue.lock().await.len(), 1);
    }

    #[tokio::test]
    async fn message_to_a_dead_session_attempts_resume_session_not_a_410() {
        // No CLI at this path: resume_session is attempted (it is what
        // spawns the CLI) and fails. A 410/404 would mean it was never tried.
        let h = action_harness(Some("/nonexistent/claude-cli")).await;
        let sid = seed_session(&h).await;
        assert!(!h.manager.is_session_active(&sid.to_string()).await);
        let (status, body) = call(
            &h.app,
            auth_post(&msg_uri(sid), r#"{"content":"reprends"}"#),
        )
        .await;
        assert_ne!(status, StatusCode::GONE);
        assert_ne!(status, StatusCode::NOT_FOUND, "{body}");
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
        // A 500 here can only come from `AppError::Internal`, which is what
        // `resume_session` returns when the spawn fails — the 410/404 checks
        // above are what prove it was *tried*. The body must therefore be the
        // generic message and must NOT name the internal failure: this PR stops
        // leaking DB errors, paths and queries to clients, and the old
        // assertion on "resumed InteractiveClient" depended on that leak.
        assert_eq!(
            body["error"].as_str().unwrap(),
            crate::api::handlers::INTERNAL_ERROR_MESSAGE,
            "an internal failure must reach the client generic: {body}"
        );
    }

    #[tokio::test]
    async fn a_resume_answers_with_the_access_stored_on_the_session() {
        use crate::chat::provider::policy::SessionAccess;
        let h = action_harness(None).await;
        let mut s = test_chat_session(None);
        s.access = SessionAccess::ReadOnly;
        h.graph.create_chat_session(&s).await.unwrap();
        let (_, access) = super::stored_place_and_access(h.graph.as_ref(), &s.id.to_string()).await;
        assert_eq!(access, SessionAccess::ReadOnly);
        // The answer carries it on the wire; a normal session leaves it off.
        let resp = crate::chat::types::CreateSessionResponse {
            access,
            ..Default::default()
        };
        assert_eq!(serde_json::to_value(&resp).unwrap()["access"], "read_only");
        let normal =
            serde_json::to_value(crate::chat::types::CreateSessionResponse::default()).unwrap();
        assert!(normal.get("access").is_none());
        // Unknown or malformed ids fall back to the defaults.
        let (_, a) = super::stored_place_and_access(h.graph.as_ref(), "nope").await;
        assert_eq!(a, SessionAccess::Normal);
        let (_, a) =
            super::stored_place_and_access(h.graph.as_ref(), &Uuid::new_v4().to_string()).await;
        assert_eq!(a, SessionAccess::Normal);
    }

    #[tokio::test]
    async fn message_validation_and_unknown_session() {
        let h = action_harness(None).await;
        let sid = seed_session(&h).await;
        let (status, _) = call(&h.app, auth_post(&msg_uri(sid), r#"{"content":"   "}"#)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = call(
            &h.app,
            auth_post(&msg_uri(Uuid::new_v4()), r#"{"content":"x"}"#),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}

// ============================================================================
// Provider switch (B-SW)
// ============================================================================

/// Body of `POST /api/chat/sessions/{id}/switch-provider`.
#[derive(Debug, Deserialize)]
pub struct SwitchProviderBody {
    /// The provider instance to move the conversation to.
    pub provider: String,
    /// Model on that provider (absent: the instance's default).
    #[serde(default)]
    pub model: Option<String>,
    /// The next message, sent on the new provider with the earlier conversation
    /// relayed in front of it.
    pub message: String,
}

/// POST /api/chat/sessions/{id}/switch-provider — continue a conversation on
/// another provider.
///
/// Opens a new session on the target (same project and directory), relays the
/// earlier conversation to it, then closes the old session. A refusal (consent,
/// endpoint guard, security gate, unknown provider) leaves the old session
/// untouched. A chat session calling through its MCP server is refused: moving a
/// conversation to another provider sends it to another origin, a person's call.
pub async fn switch_provider(
    State(state): State<OrchestratorState>,
    claims: Option<axum::Extension<crate::auth::jwt::Claims>>,
    headers: axum::http::HeaderMap,
    Path(session_id): Path<Uuid>,
    Json(body): Json<SwitchProviderBody>,
) -> Result<Json<crate::chat::types::SwitchProviderResponse>, AppError> {
    use crate::chat::envelope;
    let caller = envelope::identify_caller(
        claims.as_ref().map(|c| &c.0),
        envelope::session_header(&headers),
        state.auth_config.is_some(),
    )?;
    if matches!(caller, envelope::SpawnCaller::Agent { .. }) {
        return Err(AppError::Forbidden(
            "only a signed-in user can move a conversation to another provider".to_string(),
        ));
    }
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    let response = chat_manager
        .switch_session_provider(
            &session_id.to_string(),
            &body.provider,
            body.model.as_deref(),
            &body.message,
            claims.map(|c| c.0),
        )
        .await
        .map_err(|e| switch_error(e, &session_id.to_string()))?;

    // The new session's `created` event (with `switched_from`) is emitted by `ChatManager`.
    Ok(Json(response))
}

/// How a refused or failed provider switch is told to the client: the switch's own
/// refusals are 400 (404 for an unknown session); anything else came from opening
/// the new session and keeps its typed open-failure status (consent, endpoint guard,
/// security gate, unknown provider).
fn switch_error(error: anyhow::Error, session_id: &str) -> AppError {
    use crate::chat::types::SwitchProviderError;
    match error.downcast_ref::<SwitchProviderError>() {
        Some(SwitchProviderError::NotFound) => {
            AppError::NotFound(format!("Session {session_id} not found"))
        }
        Some(refusal) => match refusal.code() {
            Some(code) => {
                AppError::Provider(Box::new(crate::chat::provider::errors::OpenFailure {
                    status: 400,
                    code,
                    message: refusal.to_string(),
                    provider_id: match refusal {
                        SwitchProviderError::SameProvider(id) => Some(id.clone()),
                        _ => None,
                    },
                    action: None,
                    retryable: false,
                    retry_after_ms: None,
                    resets_at_ms: None,
                    fallbacks: None,
                    consent_scope: None,
                    candidates: None,
                }))
            }
            None => AppError::BadRequest(refusal.to_string()),
        },
        None => AppError::from_open_error(error, None),
    }
}

#[cfg(test)]
mod switch_provider_tests {
    use super::*;
    use crate::chat::types::SwitchProviderError;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    #[test]
    fn an_unknown_session_is_a_404_that_names_it() {
        let err = switch_error(anyhow::Error::new(SwitchProviderError::NotFound), "abc");
        assert!(
            matches!(&err, AppError::NotFound(m) if m.contains("abc")),
            "{err:?}"
        );
    }

    #[test]
    fn an_invalid_session_id_is_a_400_with_its_reason() {
        let refusal = SwitchProviderError::InvalidSession;
        let text = refusal.to_string();
        let err = switch_error(anyhow::Error::new(refusal), "abc");
        assert!(
            matches!(&err, AppError::BadRequest(m) if *m == text),
            "{err:?}"
        );
    }

    /// The body of a coded refusal, as the client reads it.
    fn coded(refusal: SwitchProviderError) -> serde_json::Value {
        match switch_error(anyhow::Error::new(refusal), "abc") {
            AppError::Provider(failure) => {
                assert_eq!(failure.status, 400);
                failure.to_json()
            }
            other => panic!("a typed refusal is expected: {other:?}"),
        }
    }

    #[test]
    fn an_empty_message_is_a_400_coded_empty_message_with_its_wording_kept() {
        let body = coded(SwitchProviderError::EmptyMessage);
        assert_eq!(body["code"], "empty_message");
        assert_eq!(
            body["error"],
            "a message is needed to continue the conversation on the new provider"
        );
        assert_eq!(body["retryable"], false);
    }

    #[test]
    fn the_same_provider_is_a_400_coded_same_provider_with_its_wording_kept() {
        let body = coded(SwitchProviderError::SameProvider("local".into()));
        assert_eq!(body["code"], "same_provider");
        assert_eq!(
            body["error"],
            "the session is already on 'local': change its model instead"
        );
        assert_eq!(body["provider_id"], "local");
        assert_eq!(body["retryable"], false);
    }

    #[test]
    fn a_failure_of_opening_the_new_session_keeps_its_typed_status() {
        let open_failure = || {
            anyhow::Error::new(nexus_claude::agent::ProviderError::invalid(
                "no default model",
            ))
        };
        let kept = AppError::from_open_error(open_failure(), None);
        let told = switch_error(open_failure(), "abc");
        assert_eq!(
            format!("{told:?}"),
            format!("{kept:?}"),
            "an open failure must reach the client exactly as create_session tells it"
        );
        // And it is not one of the switch's own answers.
        assert!(!matches!(&told, AppError::NotFound(_)), "{told:?}");
    }
}

/// The HTTP answer of a cancel (`cancel-tools`, `cancel-task`) that did not
/// happen: the provider's refusal (`unsupported` 422, …, `chat::provider::errors`)
/// or why the instance holding the session gave no answer
/// (`chat::cancel_relay::CancelRelayError`: `owner_unreachable` 409,
/// `owner_timeout` 504, `session_gone` 410, `owner_protocol` / `owner_failed` 502).
/// Anything else is a 500.
pub(crate) fn cancel_error(error: anyhow::Error) -> AppError {
    use crate::chat::cancel_relay::CancelRelayError;
    if let Some(relay) = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<CancelRelayError>())
    {
        let failure = relay.failure();
        tracing::warn!(code = failure.code, error = %error, "cancel: no answer from the instance holding the session");
        return AppError::Provider(Box::new(failure));
    }
    match crate::chat::provider::errors::classify_open_error(&error, None) {
        Some(failure) => {
            tracing::warn!(code = failure.code, error = %error, "cancel refused by the provider");
            AppError::Provider(Box::new(failure))
        }
        None => AppError::Internal(error),
    }
}

#[cfg(test)]
mod file_scope_lift_tests {
    use super::person_lifts_file_scope;
    use crate::auth::jwt::{Claims, TOKEN_TYPE_AGENT_SESSION, TOKEN_TYPE_MCP};
    use crate::chat::envelope::SpawnCaller;
    use crate::chat::types::ChatRequest;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    fn person() -> Claims {
        Claims {
            sub: uuid::Uuid::new_v4().to_string(),
            email: "me@example.com".into(),
            ..Claims::anonymous()
        }
    }

    fn opening(claims: Option<Claims>, mode: Option<&str>) -> ChatRequest {
        ChatRequest {
            user_claims: claims,
            permission_mode: mode.map(str::to_owned),
            ..Default::default()
        }
    }

    /// Decision fce98b8a: only a person opening a NEW session with trust named explicitly lifts
    /// its file scope. A long-lived MCP token (classified as a person by `identify_caller`), an
    /// agent session, the server's account, the anonymous user, an agent caller, a resume and a
    /// mode left to the default never do.
    #[test]
    fn only_a_person_opening_a_session_in_trust_lifts_its_file_scope() {
        let human = SpawnCaller::Human;
        for mode in ["trust", "bypassPermissions"] {
            assert!(
                person_lifts_file_scope(&human, &opening(Some(person()), Some(mode))),
                "{mode}"
            );
        }
        for mode in [
            None,
            Some("default"),
            Some("acceptEdits"),
            Some("auto"),
            Some("plan"),
        ] {
            assert!(
                !person_lifts_file_scope(&human, &opening(Some(person()), mode)),
                "{mode:?}"
            );
        }
        let mcp = Claims {
            token_type: Some(TOKEN_TYPE_MCP.into()),
            ..person()
        };
        let agent = Claims {
            token_type: Some(TOKEN_TYPE_AGENT_SESSION.into()),
            ..person()
        };
        for (who, claims) in [
            ("mcp token", Some(mcp)),
            ("agent session", Some(agent)),
            (
                "service account",
                Some(Claims::service_account("runner-agent:x")),
            ),
            ("anonymous", Some(Claims::anonymous())),
            ("no claims", None),
        ] {
            assert!(
                !person_lifts_file_scope(&human, &opening(claims, Some("trust"))),
                "{who}"
            );
        }
        let caller = SpawnCaller::Agent {
            session_id: uuid::Uuid::new_v4().to_string(),
            token_ceiling: None,
        };
        assert!(!person_lifts_file_scope(
            &caller,
            &opening(Some(person()), Some("trust"))
        ));
        let resume = ChatRequest {
            session_id: Some(uuid::Uuid::new_v4().to_string()),
            ..opening(Some(person()), Some("trust"))
        };
        assert!(!person_lifts_file_scope(&human, &resume));
    }

    /// The request body cannot set the flag: it is not deserialised.
    #[test]
    fn the_request_body_cannot_lift_the_file_scope() {
        let request: ChatRequest = serde_json::from_value(serde_json::json!({
            "message": "go",
            "cwd": "/tmp",
            "permission_mode": "trust",
            "file_scope_lifted": true,
        }))
        .unwrap();
        assert!(!request.file_scope_lifted);
    }
}

// ============================================================================
// Context anchors of a conversation (T4 of plan 09c88ffe)
// ============================================================================

/// The anchors API of a conversation (T4 of plan 09c88ffe): list, add, change
/// the role of and remove the context anchors of a chat session, and the
/// reverse read — the conversations anchored on an entity — paginated.
///
/// Rules this layer adds to those of [`crate::chat::anchor`]:
///
/// - **Who placed the anchor comes from the authenticated principal, never from
///   the body.** A person's own session token places as `user`; an agent
///   session's token, an MCP token, a vault token, the server's account and the
///   anonymous user place as `agent` ([`anchor_principal`]). The body has no
///   `by` / `placed_by` / `inferred` / `confidence` field: such keys are
///   ignored. That matters because only a `user` anchor on a project or a
///   workspace opens that project to the conversation (decision 75e160f4).
/// - **Idempotent add.** The same target twice is one anchor: the second call
///   answers `200` with the existing anchor and writes nothing (no journal entry,
///   no version bump, no event).
/// - **Optimistic concurrency on the whole set.** Every answer carries the
///   version of the anchor set in `ETag` (`"<n>"`); a mutation sent with
///   `If-Match: "<n>"` is refused with `409` and the current set when the set has
///   changed since. The check and the write are one transaction in the store.
/// - **Live updates.** Every committed change emits a CRUD event
///   (`chat_session` / `updated`, payload `anchors`) on `/ws/events`; the channel
///   is the shared CRUD stream, no entity id in any channel name.
/// - An agent session's token writes only on its own session or a session it
///   spawned (the `require_auth` middleware, rule 4d).
pub mod anchors {
    use super::{session_node_to_response, stamp_activity};
    use crate::api::handlers::{AppError, OrchestratorState};
    use crate::auth::jwt::Claims;
    use crate::chat::anchor::{
        validate_target_id, Anchor, AnchorActor, AnchorError, AnchorOp, AnchorRole, AnchorSet,
        AnchorSetChange, AnchorState, AnchorTargetType, NewAnchor,
    };
    use crate::chat::types::ChatSession;
    use crate::events::{CrudAction, CrudEvent, EntityType, EventEmitter};
    use axum::extract::{Path, Query, State};
    use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
    use axum::response::{IntoResponse, Response};
    use axum::Json;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeSet;
    use uuid::Uuid;

    /// Largest page of the reverse read.
    pub const MAX_PAGE: usize = 100;
    /// Page of the reverse read when the caller names none.
    pub const DEFAULT_PAGE: usize = 20;

    /// Who is placing or changing an anchor, as recorded on it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct AnchorPrincipal {
        pub by: AnchorActor,
        /// Free-form identity journaled with the change (never a secret).
        pub actor: String,
    }

    /// The actor of an anchor change, derived from the authenticated principal
    /// ONLY. `user` for a person signed in with their own session token (with
    /// authentication off: any caller that does not present itself as a chat
    /// session through `X-Session-Id`, as [`crate::chat::envelope::identify_caller`]
    /// does); `agent` for everything else.
    pub fn anchor_principal(
        claims: Option<&Claims>,
        session_header: Option<&str>,
        auth_enabled: bool,
    ) -> AnchorPrincipal {
        let agent = |actor: String| AnchorPrincipal {
            by: AnchorActor::Agent,
            actor,
        };
        if let Some(c) = claims.filter(|c| c.is_agent_session()) {
            let sid = crate::auth::jwt::agent_session_binding(c)
                .map(|b| b.session_id)
                .unwrap_or_default();
            return agent(format!("agent_session:{sid}"));
        }
        if !auth_enabled {
            if let Some(sid) = session_header.map(str::trim).filter(|s| !s.is_empty()) {
                return agent(format!("agent_session:{sid}"));
            }
            return AnchorPrincipal {
                by: AnchorActor::User,
                actor: format!("user:{}", claims.map(|c| c.sub.as_str()).unwrap_or("local")),
            };
        }
        match claims {
            Some(c) if c.may_keep_lasting_approvals() => AnchorPrincipal {
                by: AnchorActor::User,
                actor: format!("user:{}", c.sub),
            },
            Some(c) if c.is_mcp_token() => agent(format!("mcp_token:{}", c.sub)),
            Some(c) => agent(format!("token:{}", c.sub)),
            None => agent("unauthenticated".to_string()),
        }
    }

    fn principal_of(
        state: &OrchestratorState,
        claims: &Option<axum::Extension<Claims>>,
        headers: &HeaderMap,
    ) -> AnchorPrincipal {
        anchor_principal(
            claims.as_ref().map(|c| &c.0),
            crate::chat::envelope::session_header(headers),
            state.auth_config.is_some(),
        )
    }

    // ============================================================================
    // Wire types
    // ============================================================================

    /// The anchors of a session and the version of the set (also in `ETag`).
    #[derive(Debug, Serialize)]
    pub struct AnchorSetResponse {
        pub session_id: Uuid,
        pub version: u64,
        pub anchors: Vec<Anchor>,
    }

    /// Result of a mutation.
    #[derive(Debug, Serialize)]
    pub struct AnchorChangeResponse {
        /// The anchor after the change (the removed one for a removal).
        pub anchor: Anchor,
        /// The anchor did not exist before (add only).
        pub created: bool,
        /// Something was written (false: idempotent no-op).
        pub changed: bool,
        /// The version of the set after the change.
        pub version: u64,
    }

    /// `POST /api/chat/sessions/{id}/anchors`. Who places the anchor is NOT a field:
    /// it comes from the token.
    #[derive(Debug, Deserialize)]
    pub struct AddAnchorBody {
        pub target_type: String,
        pub target_id: String,
        /// Default: `["mention"]`.
        #[serde(default)]
        pub roles: Vec<String>,
        #[serde(default)]
        pub snapshot_name: Option<String>,
        #[serde(default)]
        pub snapshot_path: Option<String>,
        #[serde(default)]
        pub snapshot_type: Option<String>,
        #[serde(default)]
        pub rev: Option<String>,
    }

    /// `PATCH /api/chat/sessions/{id}/anchors/{anchor_id}`.
    #[derive(Debug, Deserialize)]
    pub struct UpdateAnchorBody {
        /// `promote` (give the role) or `demote` (take it away).
        pub action: String,
        pub role: String,
        /// Version of the anchor the caller read; default: the current one.
        #[serde(default)]
        pub version: Option<u64>,
    }

    /// Query of `DELETE /api/chat/sessions/{id}/anchors/{anchor_id}`.
    #[derive(Debug, Default, Deserialize)]
    pub struct RemoveAnchorQuery {
        #[serde(default)]
        pub version: Option<u64>,
    }

    /// Query of `GET /api/chat/anchored-sessions`.
    #[derive(Debug, Deserialize)]
    pub struct AnchoredSessionsQuery {
        pub target_type: String,
        pub target_id: String,
        /// Only the anchors in this state (`live`, `moved`, ...). Default: all.
        #[serde(default)]
        pub state: Option<String>,
        /// Page size, 1..=100 (default 20).
        #[serde(default)]
        pub limit: Option<usize>,
        /// `next_cursor` of the previous page.
        #[serde(default)]
        pub cursor: Option<String>,
    }

    /// One conversation anchored on the entity, with its anchor.
    #[derive(Debug, Serialize)]
    pub struct AnchoredSession {
        pub session: ChatSession,
        pub anchor: Anchor,
    }

    /// A page of the reverse read.
    #[derive(Debug, Serialize)]
    pub struct AnchoredSessionsPage {
        pub items: Vec<AnchoredSession>,
        /// Pass back as `cursor`; absent on the last page.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub next_cursor: Option<String>,
    }

    // ============================================================================
    // Helpers
    // ============================================================================

    fn etag(version: u64) -> HeaderValue {
        HeaderValue::from_str(&format!("\"{version}\"")).expect("a number is a valid header value")
    }

    /// `If-Match`: `"<n>"`, `W/"<n>"` or `<n>`; `*` (or absent) = no check.
    pub fn parse_if_match(headers: &HeaderMap) -> Result<Option<u64>, AppError> {
        let Some(raw) = headers.get(header::IF_MATCH) else {
            return Ok(None);
        };
        let raw = raw
            .to_str()
            .map_err(|_| AppError::BadRequest("If-Match: not a valid header value".into()))?
            .trim();
        if raw == "*" {
            return Ok(None);
        }
        let bare = raw.strip_prefix("W/").unwrap_or(raw).trim_matches('"');
        bare.parse::<u64>().map(Some).map_err(|_| {
            AppError::BadRequest(format!(
                "If-Match: expected the ETag of the anchor set (\"<n>\"), got {raw}"
            ))
        })
    }

    fn parse_target_type(raw: &str) -> Result<AnchorTargetType, AppError> {
        AnchorTargetType::parse(raw).ok_or_else(|| {
            let known: Vec<&str> = AnchorTargetType::ALL.iter().map(|t| t.as_str()).collect();
            AppError::BadRequest(format!(
                "unknown target_type `{raw}` (expected one of {})",
                known.join(", ")
            ))
        })
    }

    fn parse_role(raw: &str) -> Result<AnchorRole, AppError> {
        AnchorRole::parse(raw).ok_or_else(|| {
            AppError::BadRequest(format!(
                "unknown role `{raw}` (expected focus, mention or origin)"
            ))
        })
    }

    fn json_with_etag(status: StatusCode, version: u64, body: impl Serialize) -> Response {
        let mut resp = (status, Json(body)).into_response();
        resp.headers_mut().insert(header::ETAG, etag(version));
        resp
    }

    async fn load_set(state: &OrchestratorState, session_id: Uuid) -> Result<AnchorSet, AppError> {
        state
            .orchestrator
            .neo4j()
            .get_anchor_set(session_id)
            .await
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::NotFound(format!("Session {session_id} not found")))
    }

    /// The answer to a refused operation. A conflict (`409`) carries the current
    /// set and its `ETag`, so the client can rebase without a second read.
    async fn refusal(state: &OrchestratorState, session_id: Uuid, err: anyhow::Error) -> Response {
        let Some(e) = err.downcast_ref::<AnchorError>() else {
            return AppError::Internal(err).into_response();
        };
        let status = match e {
            AnchorError::InvalidTargetId { .. }
            | AnchorError::EmptyRoles
            | AnchorError::InvalidConfidence
            | AnchorError::FieldTooLong { .. }
            | AnchorError::LastRole => StatusCode::BAD_REQUEST,
            AnchorError::OriginImmutable
            | AnchorError::AgentCannotFocus
            | AnchorError::AgentRoleForbidden(_) => StatusCode::FORBIDDEN,
            AnchorError::NotFound(_) | AnchorError::SessionNotFound(_) => StatusCode::NOT_FOUND,
            AnchorError::FocusCapReached { .. }
            | AnchorError::SessionCapReached { .. }
            | AnchorError::OriginAlreadyExists
            | AnchorError::AlreadyAnchored { .. }
            | AnchorError::VersionConflict { .. }
            | AnchorError::SetVersionConflict { .. } => StatusCode::CONFLICT,
        };
        let mut body = serde_json::json!({ "error": e.to_string(), "code": e.code() });
        if let AnchorError::AlreadyAnchored { anchor_id } = e {
            body["anchor_id"] = serde_json::json!(anchor_id);
        }
        if status != StatusCode::CONFLICT {
            return (status, Json(body)).into_response();
        }
        match load_set(state, session_id).await {
            Ok(set) => {
                body["current"] =
                    serde_json::json!({ "version": set.version, "anchors": set.anchors });
                json_with_etag(status, set.version, body)
            }
            Err(_) => (status, Json(body)).into_response(),
        }
    }

    /// The CRUD event of a committed change, on the shared `/ws/events` stream.
    fn emit_change(state: &OrchestratorState, session_id: Uuid, applied: &AnchorSetChange) {
        let Some(ev) = &applied.change.event else {
            return;
        };
        let a = &applied.change.anchor;
        state.event_bus.emit(
            CrudEvent::new(
                EntityType::ChatSession,
                CrudAction::Updated,
                session_id.to_string(),
            )
            .with_payload(serde_json::json!({
                "anchors": {
                    "change": ev.kind.as_str(),
                    "anchor_id": a.id,
                    "target_type": a.target_type,
                    "target_id": a.target_id,
                    "roles": a.roles,
                    "state": a.state,
                    "by": ev.by,
                    "version": applied.version,
                }
            })),
        );
    }

    async fn apply(
        state: &OrchestratorState,
        session_id: Uuid,
        op: AnchorOp,
        if_match: Option<u64>,
    ) -> Result<AnchorSetChange, Response> {
        match state
            .orchestrator
            .neo4j()
            .apply_anchor_op_if(session_id, op, if_match)
            .await
        {
            Ok(applied) => {
                emit_change(state, session_id, &applied);
                Ok(applied)
            }
            Err(e) => Err(refusal(state, session_id, e).await),
        }
    }

    fn change_response(status: StatusCode, applied: AnchorSetChange) -> Response {
        let version = applied.version;
        json_with_etag(
            status,
            version,
            AnchorChangeResponse {
                changed: applied.change.event.is_some(),
                created: applied.change.created,
                anchor: applied.change.anchor,
                version,
            },
        )
    }

    /// The current version of `anchor_id` in `set` (the caller's, when given).
    fn anchor_version(
        set: &AnchorSet,
        anchor_id: Uuid,
        given: Option<u64>,
    ) -> Result<u64, AppError> {
        if let Some(v) = given {
            return Ok(v);
        }
        set.anchors
            .iter()
            .find(|a| a.id == anchor_id)
            .map(|a| a.version)
            .ok_or_else(|| {
                AppError::NotFound(format!("Anchor {anchor_id} not found in this session"))
            })
    }

    // ============================================================================
    // Handlers
    // ============================================================================

    /// GET /api/chat/sessions/{id}/anchors — the anchors and the version of the set.
    pub async fn list_anchors(
        State(state): State<OrchestratorState>,
        Path(session_id): Path<Uuid>,
    ) -> Result<Response, AppError> {
        let set = load_set(&state, session_id).await?;
        Ok(json_with_etag(
            StatusCode::OK,
            set.version,
            AnchorSetResponse {
                session_id,
                version: set.version,
                anchors: set.anchors,
            },
        ))
    }

    /// POST /api/chat/sessions/{id}/anchors — anchor the session on an entity.
    /// `201` when created, `200` when the same target already held those roles.
    pub async fn add_anchor(
        State(state): State<OrchestratorState>,
        claims: Option<axum::Extension<Claims>>,
        headers: HeaderMap,
        Path(session_id): Path<Uuid>,
        Json(body): Json<AddAnchorBody>,
    ) -> Result<Response, AppError> {
        let if_match = parse_if_match(&headers)?;
        let principal = principal_of(&state, &claims, &headers);
        let target_type = parse_target_type(&body.target_type)?;
        let roles: BTreeSet<AnchorRole> = if body.roles.is_empty() {
            [AnchorRole::Mention].into()
        } else {
            body.roles
                .iter()
                .map(|r| parse_role(r))
                .collect::<Result<_, _>>()?
        };
        let mut new = NewAnchor::new(
            target_type,
            body.target_id,
            roles,
            principal.by,
            principal.actor,
        );
        new.snapshot_name = body.snapshot_name;
        new.snapshot_path = body.snapshot_path;
        new.snapshot_type = body.snapshot_type;
        new.rev = body.rev;
        match apply(&state, session_id, AnchorOp::Add(new), if_match).await {
            Ok(applied) => {
                let status = if applied.change.created {
                    StatusCode::CREATED
                } else {
                    StatusCode::OK
                };
                Ok(change_response(status, applied))
            }
            Err(resp) => Ok(resp),
        }
    }

    /// PATCH /api/chat/sessions/{id}/anchors/{anchor_id} — give (`promote`) or take
    /// away (`demote`) one role.
    pub async fn update_anchor(
        State(state): State<OrchestratorState>,
        claims: Option<axum::Extension<Claims>>,
        headers: HeaderMap,
        Path((session_id, anchor_id)): Path<(Uuid, Uuid)>,
        Json(body): Json<UpdateAnchorBody>,
    ) -> Result<Response, AppError> {
        let if_match = parse_if_match(&headers)?;
        let principal = principal_of(&state, &claims, &headers);
        let role = parse_role(&body.role)?;
        let set = load_set(&state, session_id).await?;
        let expected_version = anchor_version(&set, anchor_id, body.version)?;
        let op = match body.action.as_str() {
            "promote" => AnchorOp::Promote {
                anchor_id,
                role,
                expected_version,
                by: principal.by,
                actor: principal.actor,
            },
            "demote" => AnchorOp::Demote {
                anchor_id,
                role,
                expected_version,
                by: principal.by,
                actor: principal.actor,
            },
            other => {
                return Err(AppError::BadRequest(format!(
                    "unknown action `{other}` (expected promote or demote)"
                )))
            }
        };
        match apply(&state, session_id, op, if_match).await {
            Ok(applied) => Ok(change_response(StatusCode::OK, applied)),
            Err(resp) => Ok(resp),
        }
    }

    /// DELETE /api/chat/sessions/{id}/anchors/{anchor_id} — remove an anchor (never
    /// the origin one). The target is left alone.
    pub async fn remove_anchor(
        State(state): State<OrchestratorState>,
        claims: Option<axum::Extension<Claims>>,
        headers: HeaderMap,
        Path((session_id, anchor_id)): Path<(Uuid, Uuid)>,
        Query(query): Query<RemoveAnchorQuery>,
    ) -> Result<Response, AppError> {
        let if_match = parse_if_match(&headers)?;
        let principal = principal_of(&state, &claims, &headers);
        let set = load_set(&state, session_id).await?;
        let expected_version = anchor_version(&set, anchor_id, query.version)?;
        let op = AnchorOp::Remove {
            anchor_id,
            expected_version,
            by: principal.by,
            actor: principal.actor,
        };
        match apply(&state, session_id, op, if_match).await {
            Ok(applied) => Ok(change_response(StatusCode::OK, applied)),
            Err(resp) => Ok(resp),
        }
    }

    /// GET /api/chat/anchored-sessions?target_type=&target_id= — the conversations
    /// anchored on one entity, keyset-paginated (page ≤ 100, `next_cursor`).
    pub async fn list_anchored_sessions(
        State(state): State<OrchestratorState>,
        Query(q): Query<AnchoredSessionsQuery>,
    ) -> Result<Json<AnchoredSessionsPage>, AppError> {
        let target_type = parse_target_type(&q.target_type)?;
        let target_id = validate_target_id(target_type, &q.target_id)
            .map_err(|e| AppError::BadRequest(e.to_string()))?;
        let anchor_state = q
            .state
            .as_deref()
            .map(|s| {
                AnchorState::parse(s)
                    .ok_or_else(|| AppError::BadRequest(format!("unknown anchor state `{s}`")))
            })
            .transpose()?;
        let limit = q.limit.unwrap_or(DEFAULT_PAGE);
        if !(1..=MAX_PAGE).contains(&limit) {
            return Err(AppError::BadRequest(format!(
                "limit must be between 1 and {MAX_PAGE}"
            )));
        }
        if let Some(c) = q.cursor.as_deref() {
            Uuid::parse_str(c).map_err(|_| AppError::BadRequest("invalid cursor".into()))?;
        }
        let graph = state.orchestrator.neo4j();
        let page = graph
            .list_sessions_for_target(
                target_type,
                &target_id,
                anchor_state,
                limit,
                q.cursor.as_deref(),
            )
            .await
            .map_err(AppError::Internal)?;
        let mut sessions = Vec::with_capacity(page.items.len());
        let mut anchors = Vec::with_capacity(page.items.len());
        for anchor in page.items {
            // A session deleted between the two reads is left out (its anchors go
            // with it).
            if let Some(node) = graph
                .get_chat_session(anchor.session_id)
                .await
                .map_err(AppError::Internal)?
            {
                sessions.push(session_node_to_response(node));
                anchors.push(anchor);
            }
        }
        stamp_activity(&state, &mut sessions).await;
        Ok(Json(AnchoredSessionsPage {
            items: sessions
                .into_iter()
                .zip(anchors)
                .map(|(session, anchor)| AnchoredSession { session, anchor })
                .collect(),
            next_cursor: page.next_cursor,
        }))
    }

    // ============================================================================
    // Tests — through the real router and the authentication middleware
    // ============================================================================

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::chat::anchor_resolver::resolve_session_scope;
        use crate::graph::neighborhood::RawNode;
        use crate::neo4j::mock::MockGraphStore;
        #[allow(unused_imports)]
        use crate::neo4j::traits::*;
        #[allow(unused_imports)]
        use crate::neo4j::traits::*;
        use crate::test_helpers::{
            router_with_auth_and_events, test_bearer_token, test_chat_session, test_project_named,
        };
        use axum::body::Body;
        use axum::http::Request;
        use serde_json::{json, Value};
        use std::sync::Arc;
        use tower::ServiceExt;

        const SECRET: &str = "test-secret-key-minimum-32-chars!!";

        struct Fx {
            app: axum::Router,
            bus: Arc<crate::events::HybridEmitter>,
            graph: Arc<MockGraphStore>,
            beta: Uuid,
        }

        /// Projects alpha and beta; the sessions created by [`Fx::session`] belong to alpha.
        async fn fx() -> Fx {
            let graph = Arc::new(MockGraphStore::new());
            let alpha = test_project_named("alpha");
            let beta = test_project_named("beta");
            graph.create_project(&alpha).await.unwrap();
            graph.create_project(&beta).await.unwrap();
            let (app, bus) = router_with_auth_and_events(graph.clone()).await;
            Fx {
                app,
                bus,
                graph,
                beta: beta.id,
            }
        }

        impl Fx {
            async fn session(&self) -> Uuid {
                let s = test_chat_session(Some("alpha"));
                self.graph.create_chat_session(&s).await.unwrap();
                s.id
            }
        }

        /// A person's own session token.
        fn human() -> String {
            test_bearer_token()
        }

        /// The token of chat session `id`'s MCP server (an agent).
        fn agent(id: Uuid) -> String {
            let bare = test_bearer_token();
            let person =
                crate::auth::jwt::decode_jwt(bare.trim_start_matches("Bearer "), SECRET).unwrap();
            let binding = crate::auth::jwt::AgentSessionBinding {
                session_id: id.to_string(),
                ceiling: None,
                tool_profile: None,
                third_party: false,
            };
            let (token, jti) =
                crate::auth::jwt::generate_session_token(&person, Some(&binding), SECRET, 600)
                    .unwrap();
            crate::auth::agent_tokens::register(&jti, Some(&id.to_string()));
            format!("Bearer {token}")
        }

        struct Answer {
            status: StatusCode,
            etag: Option<String>,
            body: Value,
        }

        async fn call(
            app: &axum::Router,
            method: &str,
            uri: &str,
            token: &str,
            if_match: Option<&str>,
            body: Option<Value>,
        ) -> Answer {
            let mut req = Request::builder()
                .method(method)
                .uri(uri)
                .header("authorization", token);
            if let Some(v) = if_match {
                req = req.header("if-match", v);
            }
            let body = match body {
                Some(b) => {
                    req = req.header("content-type", "application/json");
                    Body::from(b.to_string())
                }
                None => Body::empty(),
            };
            let resp = app.clone().oneshot(req.body(body).unwrap()).await.unwrap();
            let status = resp.status();
            let etag = resp
                .headers()
                .get(header::ETAG)
                .map(|v| v.to_str().unwrap().to_string());
            let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
                .await
                .unwrap();
            Answer {
                status,
                etag,
                body: serde_json::from_slice(&bytes).unwrap_or(Value::Null),
            }
        }

        fn anchors_uri(sid: Uuid) -> String {
            format!("/api/chat/sessions/{sid}/anchors")
        }

        /// A note of beta, with one neighbour, in the graph the resolver walks.
        async fn seed_beta_note(f: &Fx) -> String {
            let note = Uuid::new_v4().to_string();
            let near = Uuid::new_v4().to_string();
            for (id, label) in [(&note, "note de beta"), (&near, "voisin de beta")] {
                f.graph
                    .neighborhood_ownership
                    .write()
                    .await
                    .project_of
                    .insert(id.clone(), f.beta.to_string());
                f.graph.neighborhood_graph.write().await.add_node(RawNode {
                    id: id.clone(),
                    node_type: "note".into(),
                    label: label.into(),
                    subtitle: None,
                    weight: 1.0,
                });
            }
            f.graph
                .neighborhood_graph
                .write()
                .await
                .add_edge(&note, &near, "RELATES_TO", 1.0);
            note
        }

        /// Anchors `sid` on beta's note and on beta itself with `token`, the body
        /// claiming a human placed them; returns whether the note got admitted
        /// (beta opened to the conversation) and the stored `by` of the project anchor.
        async fn place_beta(f: &Fx, sid: Uuid, token: &str) -> (bool, String) {
            let note = seed_beta_note(f).await;
            let forged = |t: &str, id: &str| {
                json!({
                    "target_type": t, "target_id": id, "roles": ["mention"],
                    // never read: who placed the anchor is the token's business
                    "by": "user", "placed_by": "human", "inferred": false, "confidence": 1.0
                })
            };
            let a = call(
                &f.app,
                "POST",
                &anchors_uri(sid),
                token,
                None,
                Some(forged("note", &note)),
            )
            .await;
            assert_eq!(a.status, StatusCode::CREATED, "{}", a.body);
            let p = call(
                &f.app,
                "POST",
                &anchors_uri(sid),
                token,
                None,
                Some(forged("project", &f.beta.to_string())),
            )
            .await;
            assert_eq!(p.status, StatusCode::CREATED, "{}", p.body);
            let by = p.body["anchor"]["by"].as_str().unwrap().to_string();
            let scope = resolve_session_scope(f.graph.as_ref(), sid, b"secret")
                .await
                .unwrap();
            let admitted = scope
                .anchors
                .iter()
                .any(|x| x.target_type == AnchorTargetType::Note);
            (admitted, by)
        }

        /// The security property: an agent's token posting `placed_by: human` (and
        /// `by: user`) places as `agent`, so its project anchor opens nothing; the
        /// same request with a person's token opens beta.
        #[tokio::test]
        async fn an_agent_token_claiming_a_human_placement_places_as_agent_and_opens_no_project() {
            let f = fx().await;
            let by_agent = f.session().await;
            let (admitted, by) = place_beta(&f, by_agent, &agent(by_agent)).await;
            assert_eq!(by, "agent");
            assert!(!admitted, "an agent's project anchor must not open beta");
            let stored = f.graph.list_session_anchors(by_agent).await.unwrap();
            assert!(stored.iter().all(|a| a.by == AnchorActor::Agent));

            let by_human = f.session().await;
            let (admitted, by) = place_beta(&f, by_human, &human()).await;
            assert_eq!(by, "user");
            assert!(admitted, "a person's project anchor opens beta");
        }

        #[test]
        fn the_actor_comes_from_the_principal_only() {
            use crate::auth::jwt::{Claims, TOKEN_TYPE_AGENT_SESSION, TOKEN_TYPE_MCP};
            let person = Claims {
                sub: Uuid::new_v4().to_string(),
                email: "me@example.com".into(),
                ..Claims::anonymous()
            };
            let with_type = |t: &str| Claims {
                token_type: Some(t.into()),
                ..person.clone()
            };
            let by = |c: Option<&Claims>, header: Option<&str>, auth: bool| {
                anchor_principal(c, header, auth).by
            };
            assert_eq!(by(Some(&person), None, true), AnchorActor::User);
            // a session header changes nothing for a signed-in person
            assert_eq!(by(Some(&person), Some("s"), true), AnchorActor::User);
            for c in [
                with_type(TOKEN_TYPE_AGENT_SESSION),
                with_type(TOKEN_TYPE_MCP),
                with_type("vault"),
                Claims::anonymous(),
                Claims::service_account("runner-agent:x"),
            ] {
                assert_eq!(by(Some(&c), None, true), AnchorActor::Agent, "{c:?}");
            }
            assert_eq!(by(None, None, true), AnchorActor::Agent);
            // authentication off: a chat session presents itself by its header
            assert_eq!(
                by(Some(&Claims::anonymous()), None, false),
                AnchorActor::User
            );
            assert_eq!(
                by(Some(&Claims::anonymous()), Some("sid"), false),
                AnchorActor::Agent
            );
            assert_eq!(
                by(Some(&with_type(TOKEN_TYPE_AGENT_SESSION)), None, false),
                AnchorActor::Agent
            );
        }

        #[tokio::test]
        async fn the_same_anchor_twice_is_one_anchor_one_version_and_one_event() {
            let f = fx().await;
            let sid = f.session().await;
            let mut events = f.bus.subscribe();
            let body =
                json!({"target_type": "file", "target_id": "src/lib.rs", "roles": ["focus"]});
            let first = call(
                &f.app,
                "POST",
                &anchors_uri(sid),
                &human(),
                None,
                Some(body.clone()),
            )
            .await;
            assert_eq!(first.status, StatusCode::CREATED);
            assert_eq!(first.etag.as_deref(), Some("\"1\""));
            let again = call(
                &f.app,
                "POST",
                &anchors_uri(sid),
                &human(),
                None,
                Some(body),
            )
            .await;
            assert_eq!(again.status, StatusCode::OK);
            assert_eq!(again.body["changed"], json!(false));
            assert_eq!(again.body["anchor"]["id"], first.body["anchor"]["id"]);
            assert_eq!(again.etag.as_deref(), Some("\"1\""));
            let list = call(&f.app, "GET", &anchors_uri(sid), &human(), None, None).await;
            assert_eq!(list.body["anchors"].as_array().unwrap().len(), 1);
            assert_eq!(list.body["version"], json!(1));

            let ev = events.try_recv().expect("one event for the creation");
            assert_eq!(ev.entity_type, EntityType::ChatSession);
            assert_eq!(ev.action, CrudAction::Updated);
            assert_eq!(ev.entity_id, sid.to_string());
            assert_eq!(ev.payload["anchors"]["change"], json!("added"));
            assert_eq!(ev.payload["anchors"]["target_id"], json!("src/lib.rs"));
            assert_eq!(ev.payload["anchors"]["version"], json!(1));
            assert!(events.try_recv().is_err(), "the no-op emits nothing");
        }

        #[tokio::test]
        async fn a_stale_if_match_is_refused_with_409_and_the_current_set() {
            let f = fx().await;
            let sid = f.session().await;
            let v0 = call(&f.app, "GET", &anchors_uri(sid), &human(), None, None).await;
            assert_eq!(v0.etag.as_deref(), Some("\"0\""));
            let add = |id: &str| json!({"target_type": "file", "target_id": id});
            let ok = call(
                &f.app,
                "POST",
                &anchors_uri(sid),
                &human(),
                Some("\"0\""),
                Some(add("a.rs")),
            )
            .await;
            assert_eq!(ok.status, StatusCode::CREATED);
            let stale = call(
                &f.app,
                "POST",
                &anchors_uri(sid),
                &human(),
                Some("\"0\""),
                Some(add("b.rs")),
            )
            .await;
            assert_eq!(stale.status, StatusCode::CONFLICT);
            assert_eq!(stale.body["code"], json!("anchor_set_version_conflict"));
            assert_eq!(stale.body["current"]["version"], json!(1));
            assert_eq!(
                stale.body["current"]["anchors"].as_array().unwrap().len(),
                1
            );
            assert_eq!(stale.etag.as_deref(), Some("\"1\""));
            // a malformed If-Match is a 400, `*` means no check
            let bad = call(
                &f.app,
                "POST",
                &anchors_uri(sid),
                &human(),
                Some("abc"),
                Some(add("b.rs")),
            )
            .await;
            assert_eq!(bad.status, StatusCode::BAD_REQUEST);
            let any = call(
                &f.app,
                "POST",
                &anchors_uri(sid),
                &human(),
                Some("*"),
                Some(add("b.rs")),
            )
            .await;
            assert_eq!(any.status, StatusCode::CREATED);

            // promote, demote and remove go through the same check
            let id = ok.body["anchor"]["id"].as_str().unwrap().to_string();
            let one = format!("{}/{id}", anchors_uri(sid));
            let patch = json!({"action": "promote", "role": "focus"});
            let stale = call(
                &f.app,
                "PATCH",
                &one,
                &human(),
                Some("\"1\""),
                Some(patch.clone()),
            )
            .await;
            assert_eq!(stale.status, StatusCode::CONFLICT);
            let promoted = call(&f.app, "PATCH", &one, &human(), Some("\"2\""), Some(patch)).await;
            assert_eq!(promoted.status, StatusCode::OK, "{}", promoted.body);
            assert!(promoted.body["anchor"]["roles"]
                .as_array()
                .unwrap()
                .contains(&json!("focus")));
            let gone = call(&f.app, "DELETE", &one, &human(), Some("\"3\""), None).await;
            assert_eq!(gone.status, StatusCode::OK);
            assert_eq!(gone.etag.as_deref(), Some("\"4\""));
            let missing = call(&f.app, "DELETE", &one, &human(), None, None).await;
            assert_eq!(missing.status, StatusCode::NOT_FOUND);
        }

        /// Two concurrent changes on the same read: one wins, the other gets 409, and
        /// the cap of three focus anchors holds whatever the interleaving.
        #[tokio::test]
        async fn two_concurrent_changes_one_409_and_the_focus_cap_holds() {
            let f = fx().await;
            let sid = f.session().await;
            let focus =
                |id: &str| json!({"target_type": "file", "target_id": id, "roles": ["focus"]});
            for id in ["a.rs", "b.rs"] {
                let r = call(
                    &f.app,
                    "POST",
                    &anchors_uri(sid),
                    &human(),
                    None,
                    Some(focus(id)),
                )
                .await;
                assert_eq!(r.status, StatusCode::CREATED);
            }
            // same If-Match on both: exactly one passes
            let uri = anchors_uri(sid);
            let me = human();
            let (x, y) = tokio::join!(
                call(
                    &f.app,
                    "POST",
                    &uri,
                    &me,
                    Some("\"2\""),
                    Some(focus("c.rs"))
                ),
                call(
                    &f.app,
                    "POST",
                    &uri,
                    &me,
                    Some("\"2\""),
                    Some(focus("d.rs"))
                ),
            );
            let mut statuses = [x.status, y.status];
            statuses.sort();
            assert_eq!(statuses, [StatusCode::CREATED, StatusCode::CONFLICT]);
            // no If-Match, at the cap: the fourth focus is refused
            let (x, y) = tokio::join!(
                call(&f.app, "POST", &uri, &me, None, Some(focus("e.rs"))),
                call(&f.app, "POST", &uri, &me, None, Some(focus("f.rs"))),
            );
            for r in [&x, &y] {
                assert_eq!(r.status, StatusCode::CONFLICT);
                assert_eq!(r.body["code"], json!("focus_cap_reached"));
            }
            let set = f.graph.get_anchor_set(sid).await.unwrap().unwrap();
            let focus_count = set
                .anchors
                .iter()
                .filter(|a| a.has_role(AnchorRole::Focus))
                .count();
            assert_eq!(focus_count, 3);
        }

        #[tokio::test]
        async fn an_agent_writes_mentions_on_its_own_session_only() {
            let f = fx().await;
            let own = f.session().await;
            let other = f.session().await;
            let token = agent(own);
            let mention = json!({"target_type": "file", "target_id": "src/a.rs"});
            let ok = call(
                &f.app,
                "POST",
                &anchors_uri(own),
                &token,
                None,
                Some(mention.clone()),
            )
            .await;
            assert_eq!(ok.status, StatusCode::CREATED, "{}", ok.body);
            // not its session, nor one it spawned
            let refused = call(
                &f.app,
                "POST",
                &anchors_uri(other),
                &token,
                None,
                Some(mention),
            )
            .await;
            assert!(!refused.status.is_success(), "{}", refused.status);
            assert!(f
                .graph
                .list_session_anchors(other)
                .await
                .unwrap()
                .is_empty());
            // never focus, nor origin
            for role in ["focus", "origin"] {
                let r = call(
                    &f.app,
                    "POST",
                    &anchors_uri(own),
                    &token,
                    None,
                    Some(json!({"target_type": "file", "target_id": "src/b.rs", "roles": [role]})),
                )
                .await;
                assert_eq!(r.status, StatusCode::FORBIDDEN, "{role}: {}", r.body);
            }
            // and a focus a person gave stays out of its reach
            let h = call(
                &f.app,
                "POST",
                &anchors_uri(own),
                &human(),
                None,
                Some(json!({"target_type": "file", "target_id": "src/c.rs", "roles": ["focus"]})),
            )
            .await;
            let id = h.body["anchor"]["id"].as_str().unwrap();
            let r = call(
                &f.app,
                "DELETE",
                &format!("{}/{id}", anchors_uri(own)),
                &token,
                None,
                None,
            )
            .await;
            assert_eq!(r.status, StatusCode::FORBIDDEN);
            assert_eq!(r.body["code"], json!("agent_role_forbidden"));
        }

        #[tokio::test]
        async fn the_reverse_read_is_paginated_by_cursor() {
            let f = fx().await;
            let plan = Uuid::new_v4().to_string();
            let mut expected = BTreeSet::new();
            for _ in 0..5 {
                let sid = f.session().await;
                expected.insert(sid.to_string());
                let r = call(
                    &f.app,
                    "POST",
                    &anchors_uri(sid),
                    &human(),
                    None,
                    Some(json!({"target_type": "plan", "target_id": plan})),
                )
                .await;
                assert_eq!(r.status, StatusCode::CREATED);
            }
            let mut seen = BTreeSet::new();
            let mut cursor: Option<String> = None;
            let mut pages = Vec::new();
            loop {
                let mut uri = format!(
                    "/api/chat/anchored-sessions?target_type=plan&target_id={}&limit=2",
                    plan.to_uppercase()
                );
                if let Some(c) = &cursor {
                    uri.push_str(&format!("&cursor={c}"));
                }
                let page = call(&f.app, "GET", &uri, &human(), None, None).await;
                assert_eq!(page.status, StatusCode::OK, "{}", page.body);
                let items = page.body["items"].as_array().unwrap().clone();
                pages.push(items.len());
                for it in &items {
                    assert_eq!(it["anchor"]["target_id"], json!(plan));
                    assert!(seen.insert(it["session"]["id"].as_str().unwrap().to_string()));
                }
                match page.body["next_cursor"].as_str() {
                    Some(c) => cursor = Some(c.to_string()),
                    None => break,
                }
            }
            assert_eq!(pages, vec![2, 2, 1]);
            assert_eq!(seen, expected);
            for bad in [
                format!("/api/chat/anchored-sessions?target_type=plan&target_id={plan}&limit=101"),
                format!("/api/chat/anchored-sessions?target_type=plan&target_id={plan}&limit=0"),
                "/api/chat/anchored-sessions?target_type=plan&target_id=nope".to_string(),
                format!("/api/chat/anchored-sessions?target_type=reflection&target_id={plan}"),
                format!("/api/chat/anchored-sessions?target_type=plan&target_id={plan}&cursor=x"),
            ] {
                let r = call(&f.app, "GET", &bad, &human(), None, None).await;
                assert_eq!(r.status, StatusCode::BAD_REQUEST, "{bad}");
            }
        }

        /// The scenario of `harness_neo4j::anchors::the_set_version_and_if_match_on_real_neo4j`,
        /// same numbers, on the mock: both stores count the version the same way.
        #[tokio::test]
        async fn the_set_version_scenario_on_the_mock() {
            let store = MockGraphStore::new();
            assert!(store
                .get_anchor_set(Uuid::new_v4())
                .await
                .unwrap()
                .is_none());
            let s = test_chat_session(None);
            store.create_chat_session(&s).await.unwrap();
            let sid = s.id;
            let user = |id: &str| {
                NewAnchor::new(
                    AnchorTargetType::File,
                    id,
                    [AnchorRole::Mention],
                    AnchorActor::User,
                    "t",
                )
            };
            let typed = |e: anyhow::Error| e.downcast::<AnchorError>().unwrap();
            assert_eq!(store.get_anchor_set(sid).await.unwrap().unwrap().version, 0);
            let first = store
                .apply_anchor_op_if(sid, AnchorOp::Add(user("src/a.rs")), Some(0))
                .await
                .unwrap();
            assert_eq!((first.change.created, first.version), (true, 1));
            let again = store
                .apply_anchor_op_if(sid, AnchorOp::Add(user("src/a.rs")), Some(1))
                .await
                .unwrap();
            assert_eq!((again.change.event.is_none(), again.version), (true, 1));
            let stale = store
                .apply_anchor_op_if(sid, AnchorOp::Add(user("src/b.rs")), Some(0))
                .await
                .unwrap_err();
            assert_eq!(
                typed(stale),
                AnchorError::SetVersionConflict {
                    expected: 0,
                    actual: 1
                }
            );
            assert_eq!(store.list_session_anchors(sid).await.unwrap().len(), 1);
            let second = store
                .apply_anchor_op_if(sid, AnchorOp::Add(user("src/b.rs")), Some(1))
                .await
                .unwrap();
            assert_eq!(second.version, 2);
            let promoted = store
                .apply_anchor_op_if(
                    sid,
                    AnchorOp::Promote {
                        anchor_id: first.change.anchor.id,
                        role: AnchorRole::Focus,
                        expected_version: 1,
                        by: AnchorActor::User,
                        actor: "t".into(),
                    },
                    None,
                )
                .await
                .unwrap();
            assert_eq!(promoted.version, 3);
            let set = store.get_anchor_set(sid).await.unwrap().unwrap();
            assert_eq!(set.version, 3);
            let ids: Vec<Uuid> = set.anchors.iter().map(|x| x.id).collect();
            assert_eq!(ids, vec![first.change.anchor.id, second.change.anchor.id]);
            let seqs: Vec<u64> = store
                .list_anchor_events(sid)
                .await
                .unwrap()
                .iter()
                .map(|e| e.seq)
                .collect();
            assert_eq!(seqs, vec![1, 2, 3]);
            let removed = store
                .apply_anchor_op_if(
                    sid,
                    AnchorOp::Remove {
                        anchor_id: second.change.anchor.id,
                        expected_version: 1,
                        by: AnchorActor::User,
                        actor: "t".into(),
                    },
                    Some(3),
                )
                .await
                .unwrap();
            assert_eq!(removed.version, 4);
            assert!(store.delete_chat_session(sid).await.unwrap());
            assert!(store.get_anchor_set(sid).await.unwrap().is_none());
        }

        #[tokio::test]
        async fn an_unknown_session_is_a_404() {
            let f = fx().await;
            let sid = Uuid::new_v4();
            let r = call(&f.app, "GET", &anchors_uri(sid), &human(), None, None).await;
            assert_eq!(r.status, StatusCode::NOT_FOUND);
            let r = call(
                &f.app,
                "POST",
                &anchors_uri(sid),
                &human(),
                None,
                Some(json!({"target_type": "file", "target_id": "a.rs"})),
            )
            .await;
            assert_eq!(r.status, StatusCode::NOT_FOUND);
            assert_eq!(r.body["code"], json!("session_not_found"));
        }
    }
}
