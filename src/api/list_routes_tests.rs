//! Handler tests for the cross-cutting list routes the cockpit relies on:
//! decisions by status, protocol runs across protocols, and the
//! `workspace_slug` filter on rfcs / alerts / notes needs-review.
//!
//! Fixture: two workspaces (`ws-a`, `ws-b`), one project each, with one item
//! of every kind per project, so a workspace filter that is ignored (or a
//! default that is not "all workspaces") makes an assertion fail.

use crate::api::handlers::ServerState;
use crate::api::routes::create_router;
use crate::events::EventBus;
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::models::*;
use crate::neo4j::GraphStore;
use crate::notes::{Note, NoteImportance, NoteScope, NoteStatus, NoteType};
use crate::orchestrator::{FileWatcher, Orchestrator};
use crate::protocol::{Protocol, ProtocolRun, RunStatus};
use crate::test_helpers::{
    mock_app_state_with_graph, test_bearer_token, test_decision, test_plan, test_project_named,
    test_task_titled, test_workspace,
};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use tower::ServiceExt;
use uuid::Uuid;

struct Fx {
    app: axum::Router,
    project_a: Uuid,
    project_b: Uuid,
    dec_a: Uuid,
    dec_b: Uuid,
    run_a: Uuid,
    run_b: Uuid,
    rfc_a: Uuid,
    rfc_b: Uuid,
    note_a: Uuid,
    note_b: Uuid,
    alert_a: Uuid,
    alert_b: Uuid,
}

async fn seed_workspace(g: &MockGraphStore, slug: &str) -> (Uuid, Uuid, Uuid, Uuid, Uuid, Uuid) {
    // workspace + project
    let mut ws = test_workspace();
    ws.slug = slug.to_string();
    ws.name = slug.to_string();
    g.create_workspace(&ws).await.unwrap();
    let project = test_project_named(&format!("proj-{slug}"));
    g.create_project(&project).await.unwrap();
    g.add_project_to_workspace(ws.id, project.id).await.unwrap();

    // plan -> task -> proposed decision + accepted decision
    let plan = test_plan();
    g.create_plan(&plan).await.unwrap();
    g.link_plan_to_project(plan.id, project.id).await.unwrap();
    let task = test_task_titled("t");
    g.create_task(plan.id, &task).await.unwrap();
    let mut proposed = test_decision(&format!("proposed in {slug}"), "r");
    proposed.status = DecisionStatus::Proposed;
    g.create_decision(task.id, &proposed).await.unwrap();
    let accepted = test_decision(&format!("accepted in {slug}"), "r");
    g.create_decision(task.id, &accepted).await.unwrap();

    // protocol with one running and one completed run
    let proto = Protocol::new(project.id, format!("proto-{slug}"), Uuid::new_v4());
    g.upsert_protocol(&proto).await.unwrap();
    let running = ProtocolRun::new(proto.id, proto.entry_state, "start");
    g.create_protocol_run(&running).await.unwrap();
    // (the store allows one running run per protocol: use a second protocol)
    let proto2 = Protocol::new(project.id, format!("proto2-{slug}"), Uuid::new_v4());
    g.upsert_protocol(&proto2).await.unwrap();
    let mut done = ProtocolRun::new(proto2.id, proto2.entry_state, "start");
    done.status = RunStatus::Completed;
    g.create_protocol_run(&done).await.unwrap();

    // an rfc note, and a note that needs review
    let mut rfc = Note::new_full(
        Some(project.id),
        NoteType::Rfc,
        NoteImportance::Medium,
        NoteScope::Project,
        format!("# RFC {slug}"),
        vec!["rfc-status:draft".to_string()],
        "test".to_string(),
    );
    rfc.status = NoteStatus::Active;
    g.create_note(&rfc).await.unwrap();
    let mut stale = Note::new_full(
        Some(project.id),
        NoteType::Tip,
        NoteImportance::Medium,
        NoteScope::Project,
        format!("review me {slug}"),
        vec![],
        "test".to_string(),
    );
    stale.status = NoteStatus::NeedsReview;
    g.create_note(&stale).await.unwrap();

    // an alert
    let now = chrono::Utc::now();
    let alert = AlertNode {
        id: Uuid::new_v4(),
        alert_type: "git_drift".into(),
        severity: AlertSeverity::Warning,
        message: format!("drift {slug}"),
        project_id: Some(project.id),
        acknowledged: false,
        acknowledged_by: None,
        acknowledged_at: None,
        created_at: now,
        dedup_key: format!("git_drift:{}:x", project.id),
        occurrence_count: 1,
        first_seen: Some(now),
        last_seen: Some(now),
        priority: 0.5,
    };
    g.create_alert(&alert).await.unwrap();

    (
        project.id,
        proposed.id,
        running.id,
        rfc.id,
        stale.id,
        alert.id,
    )
}

async fn fixture() -> Fx {
    let graph = Arc::new(MockGraphStore::new());
    let a = seed_workspace(&graph, "ws-a").await;
    let b = seed_workspace(&graph, "ws-b").await;
    let app_state = mock_app_state_with_graph(graph);
    let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
    let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
        orchestrator.clone(),
    )));
    let state = Arc::new(ServerState {
        orchestrator,
        watcher,
        chat_manager: None,
        event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
            EventBus::default(),
        ))),
        nats_emitter: None,
        auth_config: Some(crate::test_helpers::test_auth_config()),
        serve_frontend: false,
        frontend_path: "./dist".to_string(),
        setup_completed: true,
        server_port: 6600,
        public_url: None,
        remote_mcp: crate::RemoteMcpConfig::default(),
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
    });
    Fx {
        app: create_router(state),
        project_a: a.0,
        project_b: b.0,
        dec_a: a.1,
        dec_b: b.1,
        run_a: a.2,
        run_b: b.2,
        rfc_a: a.3,
        rfc_b: b.3,
        note_a: a.4,
        note_b: b.4,
        alert_a: a.5,
        alert_b: b.5,
    }
}

async fn get(app: &axum::Router, uri: &str) -> (StatusCode, serde_json::Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(uri)
                .header("authorization", test_bearer_token())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

fn ids(json: &serde_json::Value, key: &str) -> Vec<String> {
    let arr = json.get("items").unwrap_or(json).as_array().unwrap();
    let mut v: Vec<String> = arr
        .iter()
        .map(|x| x[key].as_str().unwrap().to_string())
        .collect();
    v.sort();
    v
}

fn sorted(mut v: Vec<Uuid>) -> Vec<String> {
    v.sort();
    v.into_iter().map(|u| u.to_string()).collect()
}

// ---------------------------------------------------------------- decisions

#[tokio::test]
async fn decisions_by_status_spans_all_workspaces_by_default() {
    let fx = fixture().await;
    let (st, json) = get(&fx.app, "/api/decisions?status=proposed").await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(json["total"], 2);
    // accepted decisions are excluded, proposed ones of both workspaces kept
    assert_eq!(ids(&json, "id"), sorted(vec![fx.dec_a, fx.dec_b]));
    // each item says which project it belongs to (lane attribution)
    let by_id: Vec<(String, String)> = json["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| {
            (
                i["id"].as_str().unwrap().into(),
                i["project_id"].as_str().unwrap().into(),
            )
        })
        .collect();
    assert!(by_id.contains(&(fx.dec_a.to_string(), fx.project_a.to_string())));
    assert!(by_id.contains(&(fx.dec_b.to_string(), fx.project_b.to_string())));
}

#[tokio::test]
async fn decisions_by_status_filters_by_workspace_and_project() {
    let fx = fixture().await;
    let (_, json) = get(
        &fx.app,
        "/api/decisions?status=proposed&workspace_slug=ws-a",
    )
    .await;
    assert_eq!(ids(&json, "id"), sorted(vec![fx.dec_a]));
    let uri = format!("/api/decisions?status=proposed&project_id={}", fx.project_b);
    let (_, json) = get(&fx.app, &uri).await;
    assert_eq!(ids(&json, "id"), sorted(vec![fx.dec_b]));
    // unknown workspace matches nothing (never "everything")
    let (st, json) = get(
        &fx.app,
        "/api/decisions?status=proposed&workspace_slug=nope",
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(json["total"], 0);
    // a blank slug is the "all workspaces" lane
    let (_, json) = get(&fx.app, "/api/decisions?status=proposed&workspace_slug=").await;
    assert_eq!(json["total"], 2);
}

#[tokio::test]
async fn decisions_by_status_paginates_and_rejects_bad_status() {
    let fx = fixture().await;
    let (_, json) = get(&fx.app, "/api/decisions?status=proposed&limit=1").await;
    assert_eq!(json["items"].as_array().unwrap().len(), 1);
    assert_eq!(json["total"], 2);
    assert_eq!(json["has_more"], true);
    let (st, _) = get(&fx.app, "/api/decisions?status=bogus").await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let (st, _) = get(&fx.app, "/api/decisions").await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn decision_slug_is_never_interpolated_into_a_response_or_error() {
    let fx = fixture().await;
    let (st, json) = get(
        &fx.app,
        "/api/decisions?status=proposed&workspace_slug=x%27%7D)%20RETURN%201//",
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(json["total"], 0);
}

// ----------------------------------------------------------- protocol runs

#[tokio::test]
async fn protocol_runs_listable_without_a_protocol() {
    let fx = fixture().await;
    let (st, json) = get(&fx.app, "/api/protocols/runs?status=running").await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(json["total"], 2);
    assert_eq!(ids(&json, "id"), sorted(vec![fx.run_a, fx.run_b]));
    for r in json["items"].as_array().unwrap() {
        assert_eq!(r["status"], "running");
    }
    // no status = every status (4 runs)
    let (_, json) = get(&fx.app, "/api/protocols/runs").await;
    assert_eq!(json["total"], 4);
}

#[tokio::test]
async fn protocol_runs_filter_by_workspace_and_reject_bad_status() {
    let fx = fixture().await;
    let (_, json) = get(
        &fx.app,
        "/api/protocols/runs?status=running&workspace_slug=ws-b",
    )
    .await;
    assert_eq!(ids(&json, "id"), sorted(vec![fx.run_b]));
    let (st, _) = get(&fx.app, "/api/protocols/runs?status=bogus").await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
}

// ------------------------------------------------- workspace_slug filters

#[tokio::test]
async fn rfcs_accept_workspace_slug() {
    let fx = fixture().await;
    let (_, all) = get(&fx.app, "/api/rfcs").await;
    assert_eq!(ids(&all, "id"), sorted(vec![fx.rfc_a, fx.rfc_b]));
    let (_, a) = get(&fx.app, "/api/rfcs?workspace_slug=ws-a").await;
    assert_eq!(ids(&a, "id"), sorted(vec![fx.rfc_a]));
    let (_, none) = get(&fx.app, "/api/rfcs?workspace_slug=nope").await;
    assert_eq!(none["total"], 0);
}

#[tokio::test]
async fn alerts_accept_workspace_slug() {
    let fx = fixture().await;
    let (_, all) = get(&fx.app, "/api/alerts").await;
    assert_eq!(ids(&all, "id"), sorted(vec![fx.alert_a, fx.alert_b]));
    let (_, b) = get(&fx.app, "/api/alerts?workspace_slug=ws-b").await;
    assert_eq!(ids(&b, "id"), sorted(vec![fx.alert_b]));
    let (_, none) = get(&fx.app, "/api/alerts?workspace_slug=nope").await;
    assert_eq!(none["total"], 0);
}

#[tokio::test]
async fn notes_needs_review_accept_workspace_slug() {
    let fx = fixture().await;
    let (_, all) = get(&fx.app, "/api/notes/needs-review").await;
    assert_eq!(ids(&all, "id"), sorted(vec![fx.note_a, fx.note_b]));
    let (_, a) = get(&fx.app, "/api/notes/needs-review?workspace_slug=ws-a").await;
    assert_eq!(ids(&a, "id"), sorted(vec![fx.note_a]));
    let (_, none) = get(&fx.app, "/api/notes/needs-review?workspace_slug=nope").await;
    assert_eq!(none.as_array().unwrap().len(), 0);
}
