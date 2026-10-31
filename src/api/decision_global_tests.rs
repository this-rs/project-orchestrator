//! `POST /api/decisions` — a decision attached to NO task.
//!
//! This route is the HTTP entry point the MCP tool `decision/add` takes when no
//! `task_id` is given. Before it existed, a choice made outside any plan had no
//! way to be recorded: it was either lost, or forced into an unrelated task and
//! then read back as if the task had decided it.
//!
//! Fixture: one project, one plan, one task — so a route that silently attached
//! the decision to the only task around would be caught.

use crate::neo4j::mock::MockGraphStore;
#[allow(unused_imports)]
use crate::neo4j::traits::*;
use crate::test_helpers::{
    router_with_auth, test_bearer_token, test_plan, test_project_named, test_task_titled,
};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;
use uuid::Uuid;

struct Fx {
    app: axum::Router,
    graph: Arc<MockGraphStore>,
    task_id: Uuid,
}

async fn fixture() -> Fx {
    let graph = Arc::new(MockGraphStore::new());
    let project = test_project_named("proj");
    graph.create_project(&project).await.unwrap();
    let plan = test_plan();
    graph.create_plan(&plan).await.unwrap();
    graph
        .link_plan_to_project(plan.id, project.id)
        .await
        .unwrap();
    let task = test_task_titled("the only task of the fixture");
    graph.create_task(plan.id, &task).await.unwrap();
    Fx {
        app: router_with_auth(graph.clone()).await,
        graph,
        task_id: task.id,
    }
}

fn post(uri: &str, body: Value) -> Request<Body> {
    Request::post(uri)
        .header("authorization", test_bearer_token())
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn get(uri: &str) -> Request<Body> {
    Request::get(uri)
        .header("authorization", test_bearer_token())
        .body(Body::empty())
        .unwrap()
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), 1_000_000)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap_or(Value::Null)
}

/// The contract of PR4: a decision with no task is accepted, and it is readable
/// afterwards. Without this route the call had no address at all.
#[tokio::test]
async fn a_decision_can_be_recorded_without_a_task() {
    let fx = fixture().await;

    let response = fx
        .app
        .clone()
        .oneshot(post(
            "/api/decisions",
            json!({
                "description": "zip 9.1.3 everywhere",
                "rationale": "two RustSec advisories reddened the CI",
                "alternatives": ["pin an older candle-core"],
                "chosen_option": "force zip 9 at the workspace level",
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let created = body_json(response).await;

    let decision_id: Uuid = created["id"]
        .as_str()
        .expect("the created decision carries its id")
        .parse()
        .expect("a decision id is a UUID");

    // Readable back through the ordinary getter.
    let response = fx
        .app
        .clone()
        .oneshot(get(&format!("/api/decisions/{decision_id}")))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let fetched = body_json(response).await;
    assert_eq!(fetched["description"], "zip 9.1.3 everywhere");

    // …and attached to NO task, even though the fixture holds exactly one.
    let links = fx.graph.task_decisions.read().await;
    let attached_to_the_only_task = links.get(&fx.task_id).cloned().unwrap_or_default();
    assert!(
        !attached_to_the_only_task.contains(&decision_id),
        "a decision recorded without a task was silently attached to task {}",
        fx.task_id
    );
}

/// The task-bound route is untouched: its integrity check still rejects an
/// unknown task (bug 3693617b — reopening it would make a decision point at a
/// task that does not exist).
#[tokio::test]
async fn the_task_bound_route_still_rejects_an_unknown_task() {
    let fx = fixture().await;
    let response = fx
        .app
        .oneshot(post(
            &format!("/api/tasks/{}/decisions", Uuid::new_v4()),
            json!({"description": "d", "rationale": "r"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

/// A `project_id` that resolves to nothing must not block the recording: the
/// scoping is best-effort, the decision is the point.
#[tokio::test]
async fn an_unknown_project_does_not_block_the_decision() {
    let fx = fixture().await;
    let response = fx
        .app
        .oneshot(post(
            "/api/decisions",
            json!({
                "description": "the graph is the source of truth",
                "rationale": "no such project is stored",
                "project_id": Uuid::new_v4(),
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}
