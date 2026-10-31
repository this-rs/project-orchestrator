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
    router_with_auth_and_events, test_bearer_token, test_plan, test_project_named, test_task_titled,
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
    let (app, bus) = router_with_auth_and_events(graph.clone()).await;
    Fx {
        app,
        bus,
        graph,
        project_id: project.id,
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

/// A `project_id` that names no project is not a scope: the decision is refused
/// (404) instead of being recorded without the scope the caller asked for, in
/// silence. (Review of 31/10/2026, point 3.)
#[tokio::test]
async fn an_unknown_project_is_refused() {
    let fx = fixture().await;
    let ghost = Uuid::new_v4();
    let response = fx
        .app
        .clone()
        .oneshot(post(
            "/api/decisions",
            json!({
                "description": "the graph is the source of truth",
                "rationale": "no such project is stored",
                "project_id": ghost,
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_json(response).await;
    assert!(body.to_string().contains(&ghost.to_string()), "{body}");
    assert!(
        fx.graph.decisions.read().await.is_empty(),
        "nothing is recorded under a scope that does not exist"
    );
}

/// A decision scoped to a project is visible where the project's decisions are
/// read: `GET /api/decisions?status=…&project_id=P`. Before the direct
/// `(Project)-[:HAS_DECISION]->(Decision)` edge, the scope lived only in the
/// search document and every project-scoped read missed it. (Point 1.)
#[tokio::test]
async fn a_project_scoped_decision_is_listed_under_its_project() {
    let fx = fixture().await;
    let response = fx
        .app
        .clone()
        .oneshot(post(
            "/api/decisions",
            json!({
                "description": "pin zip 9",
                "rationale": "advisories",
                "project_id": fx.project_id,
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let created = body_json(response).await;
    let decision_id = created["id"].as_str().unwrap().to_string();

    let response = fx
        .app
        .clone()
        .oneshot(get(&format!(
            "/api/decisions?status=proposed&project_id={}",
            fx.project_id
        )))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let listed = body_json(response).await;
    assert_eq!(listed["total"], 1, "{listed}");
    assert_eq!(listed["items"][0]["id"], decision_id, "{listed}");
    assert_eq!(
        listed["items"][0]["project_id"],
        fx.project_id.to_string(),
        "{listed}"
    );
    assert!(listed["items"][0].get("task_id").is_none(), "{listed}");

    // Another project's scope does not see it.
    let other = test_project_named("other");
    fx.graph.create_project(&other).await.unwrap();
    let response = fx
        .app
        .clone()
        .oneshot(get(&format!(
            "/api/decisions?status=proposed&project_id={}",
            other.id
        )))
        .await
        .unwrap();
    let listed = body_json(response).await;
    assert_eq!(listed["total"], 0, "{listed}");

    // And the project of the decision is the one the batched lookup gives.
    let projects = fx
        .graph
        .get_decision_project_ids(&[decision_id.parse().unwrap()])
        .await
        .unwrap();
    assert_eq!(
        projects.get(&decision_id.parse::<Uuid>().unwrap()),
        Some(&fx.project_id)
    );
}

/// The `Decision created` event of a scoped decision carries the project, so
/// the subscribers filtered by project receive it. (Point 6.)
#[tokio::test]
async fn the_created_event_of_a_scoped_decision_names_its_project() {
    let fx = fixture().await;
    let mut events = fx.bus.subscribe();
    let response = fx
        .app
        .clone()
        .oneshot(post(
            "/api/decisions",
            json!({
                "description": "d",
                "rationale": "r",
                "project_id": fx.project_id,
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let created = body_json(response).await;
    // The route's own event (the PlanManager emits on another bus).
    let mut seen = Vec::new();
    while let Ok(event) = events.try_recv() {
        if event.entity_type == crate::events::EntityType::Decision
            && event.entity_id == created["id"].as_str().unwrap()
        {
            seen.push(event);
        }
    }
    assert!(!seen.is_empty(), "a Decision event is emitted by the route");
    for event in &seen {
        assert_eq!(
            event.project_id,
            Some(fx.project_id.to_string()),
            "{event:?}"
        );
    }
}

/// The persona growth hook runs for a scoped decision, as it does on the
/// task-bound route: the project is known, the relevant personas can be found.
/// (Point 7.) The hook is spawned: the test waits for it, bounded.
#[tokio::test]
async fn a_scoped_decision_runs_the_persona_growth_hook() {
    let fx = fixture().await;
    let persona = Uuid::new_v4();
    fx.graph
        .relevant_personas
        .write()
        .await
        .insert(fx.project_id, vec![(persona, 0.9)]);
    let response = fx
        .app
        .clone()
        .oneshot(post(
            "/api/decisions",
            json!({"description": "d", "rationale": "r", "project_id": fx.project_id}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let created = body_json(response).await;
    let decision_id: Uuid = created["id"].as_str().unwrap().parse().unwrap();

    let mut linked = None;
    for _ in 0..200 {
        linked = fx
            .graph
            .persona_decisions
            .read()
            .await
            .get(&persona)
            .and_then(|m| m.get(&decision_id).copied());
        if linked.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    // weight = (0.9 * 0.8).max(0.3)
    let linked = linked.expect("the hook linked the relevant persona");
    assert!((linked - 0.72).abs() < 1e-9, "{linked}");
}

/// The growth hook shared by both routes, branch by branch: relevant personas are
/// linked with `(weight * 0.8).max(0.3)`, none links nothing, and a graph that
/// cannot answer is non-fatal. (Point 7.)
#[tokio::test]
async fn the_growth_hook_links_relevant_personas_and_survives_a_failing_graph() {
    use crate::api::handlers::link_decision_to_relevant_personas;
    let fx = fixture().await;
    let decision = Uuid::new_v4();
    let (strong, weak) = (Uuid::new_v4(), Uuid::new_v4());

    // No relevant persona: nothing linked.
    assert_eq!(
        link_decision_to_relevant_personas(fx.graph.as_ref(), decision, fx.project_id).await,
        0
    );

    fx.graph
        .relevant_personas
        .write()
        .await
        .insert(fx.project_id, vec![(strong, 1.0), (weak, 0.1)]);
    assert_eq!(
        link_decision_to_relevant_personas(fx.graph.as_ref(), decision, fx.project_id).await,
        2
    );
    let links = fx.graph.persona_decisions.read().await;
    assert_eq!(links[&strong][&decision], 0.8);
    assert_eq!(links[&weak][&decision], 0.3, "floored at 0.3");
    drop(links);

    fx.graph
        .fail_reads
        .lock()
        .unwrap()
        .insert("find_relevant_personas_for_decision");
    assert_eq!(
        link_decision_to_relevant_personas(fx.graph.as_ref(), decision, fx.project_id).await,
        0,
        "a failing graph is logged, not propagated"
    );
}
