//! `POST /api/episodes/collect` with a `session_id` — an episode taken from a
//! chat session, on demand, without a protocol run.
//!
//! Before this route accepted a session, a transition that happened outside any
//! protocol run could only be captured by the close of the session itself
//! (`collector::collect_session_episode`, saved by the chat manager). Asking for
//! it — the measure of task PR4 of plan 42d49a31 — was refused: `run_id` was
//! required. The relaxation is additive: the `run_id` path is unchanged.
//!
//! Fixture: one project, one chat session that received a message, so the
//! session has an episode to give. Nothing is saved by the route (symmetry with
//! the `run_id` path, which saves nothing either).

use crate::api::episode_handlers::{snapshot_close_reason, CollectEpisodeRequest};
use crate::episodes::collector::session_episode_id;
use crate::episodes::models::close_reason;
use crate::neo4j::mock::MockGraphStore;
#[allow(unused_imports)]
use crate::neo4j::traits::*;
use crate::test_helpers::{
    router_with_auth_and_events, test_bearer_token, test_chat_session, test_project_named,
};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;
use uuid::Uuid;

struct Fx {
    app: axum::Router,
    bus: Arc<crate::events::HybridEmitter>,
    graph: Arc<MockGraphStore>,
    project_id: Uuid,
    session_id: Uuid,
}

async fn fixture() -> Fx {
    let graph = Arc::new(MockGraphStore::new());
    let project = test_project_named("alpha");
    graph.create_project(&project).await.unwrap();
    let mut session = test_chat_session(Some("alpha"));
    session.preview = Some("make the episode reachable".into());
    session.message_count = 3;
    graph.create_chat_session(&session).await.unwrap();
    let (app, bus) = router_with_auth_and_events(graph.clone()).await;
    Fx {
        app,
        bus,
        graph,
        project_id: project.id,
        session_id: session.id,
    }
}

fn post(body: Value) -> Request<Body> {
    Request::post("/api/episodes/collect")
        .header("authorization", test_bearer_token())
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn call(fx: &Fx, body: Value) -> (StatusCode, Value) {
    let response = fx.app.clone().oneshot(post(body)).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1_000_000)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).to_string()));
    (status, json)
}

/// The witness of task PR4: `{session_id, project_id}` is accepted and returns
/// the session's episode. Before the fix the body was rejected (`run_id` missing).
#[tokio::test]
async fn an_episode_is_collected_from_a_session() {
    let fx = fixture().await;
    let mut events = fx.bus.subscribe();

    let (status, body) = call(
        &fx,
        json!({"session_id": fx.session_id, "project_id": fx.project_id}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let episode = &body["episode"];
    assert_eq!(episode["id"], session_episode_id(fx.session_id).to_string());
    assert_eq!(episode["project_id"], fx.project_id.to_string());
    assert_eq!(
        episode["session"]["session_id"],
        fx.session_id.to_string(),
        "{body}"
    );
    assert_eq!(episode["stimulus"]["request"], "make the episode reachable");
    // No chat manager serves this router: whether the session still runs is not
    // known here, and the episode says so instead of claiming a close.
    assert_eq!(episode["session"]["close_reason"], close_reason::UNKNOWN);

    // Nothing is saved: the route reads, the close writes.
    let stored = fx
        .graph
        .list_session_episodes(fx.project_id, 10)
        .await
        .unwrap();
    assert!(stored.is_empty(), "a collect must not save: {stored:?}");

    // The event of the run path, with the session's episode as its id.
    let event = events.try_recv().expect("an Episode event is emitted");
    assert_eq!(event.entity_type, crate::events::EntityType::Episode);
    assert_eq!(
        event.entity_id,
        session_episode_id(fx.session_id).to_string()
    );
    assert_eq!(event.payload["session_id"], fx.session_id.to_string());
    assert_eq!(event.project_id, Some(fx.project_id.to_string()));
}

/// `project_id` is not needed to find a session's project (the session names it).
#[tokio::test]
async fn a_session_collect_needs_no_project_id() {
    let fx = fixture().await;
    let (status, body) = call(&fx, json!({"session_id": fx.session_id})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["episode"]["project_id"], fx.project_id.to_string());
}

/// A `project_id` that is not the session's would misattribute the episode.
#[tokio::test]
async fn a_session_collect_with_a_foreign_project_is_refused() {
    let fx = fixture().await;
    let other = Uuid::new_v4();
    let (status, body) = call(
        &fx,
        json!({"session_id": fx.session_id, "project_id": other}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = body.to_string();
    assert!(message.contains(&other.to_string()), "{message}");
    assert!(message.contains(&fx.project_id.to_string()), "{message}");
}

/// A session that left nothing (unknown here) yields no episode and no event.
#[tokio::test]
async fn an_unknown_session_yields_no_episode_and_no_event() {
    let fx = fixture().await;
    let mut events = fx.bus.subscribe();
    let (status, body) = call(&fx, json!({"session_id": Uuid::new_v4()})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["episode"].is_null(), "{body}");
    assert!(events.try_recv().is_err(), "no episode, no event");
}

/// Neither a run nor a session: the refusal names both fields, so the caller
/// knows what the route accepts.
#[tokio::test]
async fn collecting_without_run_or_session_is_refused() {
    let fx = fixture().await;
    let (status, body) = call(&fx, json!({"project_id": fx.project_id})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = body.to_string();
    assert!(message.contains("run_id"), "{message}");
    assert!(message.contains("session_id"), "{message}");
}

/// Both at once: the route does not guess which episode is wanted.
#[tokio::test]
async fn collecting_with_both_run_and_session_is_refused() {
    let fx = fixture().await;
    let (status, body) = call(
        &fx,
        json!({
            "run_id": Uuid::new_v4(),
            "session_id": fx.session_id,
            "project_id": fx.project_id
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = body.to_string();
    assert!(message.contains("run_id"), "{message}");
    assert!(message.contains("session_id"), "{message}");
}

/// The run path keeps its contract: `project_id` is still required with `run_id`
/// (an optional field for the session path must not loosen the run path).
#[tokio::test]
async fn a_run_collect_still_requires_its_project() {
    let fx = fixture().await;
    let (status, body) = call(&fx, json!({"run_id": Uuid::new_v4()})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.to_string().contains("project_id"), "{body}");
}

/// The run path, unchanged: an unknown run yields no episode (as before).
#[tokio::test]
async fn a_run_collect_is_unchanged() {
    let fx = fixture().await;
    let (status, body) = call(
        &fx,
        json!({"run_id": Uuid::new_v4(), "project_id": fx.project_id}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["episode"].is_null(), "{body}");
}

/// The request as the MCP tool and older clients send it still parses.
#[test]
fn the_request_still_parses_with_run_id_and_project_id_only() {
    let req: CollectEpisodeRequest = serde_json::from_value(json!({
        "run_id": "550e8400-e29b-41d4-a716-446655440000",
        "project_id": "660e8400-e29b-41d4-a716-446655440000"
    }))
    .unwrap();
    assert!(req.run_id.is_some());
    assert!(req.session_id.is_none());
    assert!(req.project_id.is_some());
}

/// A live session is `open`, not closed; one this instance does not run is
/// `unknown` — never a close reason invented for it.
#[test]
fn the_snapshot_reason_tells_a_live_session_from_one_not_run_here() {
    assert_eq!(snapshot_close_reason(true), close_reason::OPEN);
    assert_eq!(snapshot_close_reason(false), close_reason::UNKNOWN);
    for r in [close_reason::OPEN, close_reason::UNKNOWN] {
        assert!(
            close_reason::ALL.contains(&r),
            "{r} must be a listed reason"
        );
    }
}

/// With a chat manager serving the router, liveness is asked of it: a session
/// it does not run is `unknown` here too (it may have closed, or run elsewhere).
#[tokio::test]
async fn a_chat_manager_is_asked_whether_the_session_runs() {
    use crate::api::episode_handlers::collect_episode;
    use axum::extract::{Json, State};

    let graph = Arc::new(MockGraphStore::new());
    let project = test_project_named("alpha");
    graph.create_project(&project).await.unwrap();
    let mut session = test_chat_session(Some("alpha"));
    session.preview = Some("hello".into());
    graph.create_chat_session(&session).await.unwrap();
    let manager = Arc::new(crate::chat::ChatManager::new_without_memory(
        graph.clone(),
        Arc::new(crate::meilisearch::mock::MockSearchStore::new()),
        crate::chat::config::ChatConfig::default(),
    ));
    let state = crate::test_helpers::server_state_with_chat(manager, graph).await;
    let Json(response) = collect_episode(
        State(state),
        Json(CollectEpisodeRequest {
            run_id: None,
            session_id: Some(session.id),
            project_id: None,
        }),
    )
    .await
    .unwrap();
    let episode = response.episode.expect("the session has an episode");
    assert_eq!(
        episode.session.unwrap().close_reason,
        close_reason::UNKNOWN,
        "the manager does not run this session"
    );
}
