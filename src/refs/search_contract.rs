//! The search response against the golden fixture
//! `tests/fixtures/refs/search_response.json`: the fixture wins, the data is
//! arranged to match it. The fixture shows a task with a project but no
//! workspace next to a plan of the *same* project that has one, which one
//! store cannot produce (a project belongs to one workspace or to none): the
//! two items are therefore produced by two stores and compared one by one.

use std::sync::Arc;

use serde_json::Value;
use uuid::Uuid;

use super::access::AccessPolicy;
use super::search::{search, RefSearchParams};
use super::test_support::{note_with, task_titled, user};
use super::wire::RefSearchResponse;
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::models::{PlanNode, PlanStatus, TaskStatus};
use crate::neo4j::GraphStore;
use crate::notes::NoteType;
use crate::test_helpers::{test_project, test_workspace};

fn fixture() -> Value {
    let raw = include_str!("../../tests/fixtures/refs/search_response.json");
    serde_json::from_str(raw).unwrap()
}

fn id_of(v: &Value) -> Uuid {
    Uuid::parse_str(v["id"].as_str().unwrap()).unwrap()
}

/// The request of the fixture, as the query string would carry it.
fn fixture_params(f: &Value) -> RefSearchParams {
    let q = &f["request"]["query"];
    RefSearchParams {
        q: q["q"].as_str().map(str::to_string),
        kinds: q["kinds"].as_str().map(str::to_string),
        limit: q["limit"].as_u64().map(|n| n.to_string()),
        ..Default::default()
    }
}

async fn seed_project(g: &MockGraphStore, scope: &Value, workspace: Option<&Value>) {
    let mut p = test_project();
    p.id = id_of(&scope["project"]);
    p.slug = scope["project"]["slug"].as_str().unwrap().into();
    p.name = scope["project"]["name"].as_str().unwrap().into();
    g.create_project(&p).await.unwrap();
    if let Some(w) = workspace {
        let mut ws = test_workspace();
        ws.id = id_of(w);
        ws.slug = w["slug"].as_str().unwrap().into();
        ws.name = w["name"].as_str().unwrap().into();
        g.create_workspace(&ws).await.unwrap();
        g.add_project_to_workspace(ws.id, p.id).await.unwrap();
    }
}

fn plan_from(item: &Value) -> PlanNode {
    let mut plan = PlanNode::new_for_project(
        item["label"].as_str().unwrap().into(),
        "les refs".into(),
        "t".into(),
        5,
        id_of(&item["project"]),
    );
    plan.id = id_of(item);
    plan.status = PlanStatus::InProgress;
    plan
}

async fn run(g: MockGraphStore, params: RefSearchParams) -> Value {
    let query = params.into_query().unwrap();
    let out: RefSearchResponse =
        search(Arc::new(g), &AccessPolicy::open_instance(), &user(), &query)
            .await
            .unwrap();
    serde_json::to_value(&out).unwrap()
}

#[tokio::test]
async fn plan_and_rfc_items_are_the_fixture_items() {
    let f = fixture();
    let items = f["response"]["items"].as_array().unwrap();
    let (plan_item, rfc_item) = (&items[0], &items[2]);

    let g = MockGraphStore::new();
    seed_project(&g, plan_item, Some(&plan_item["workspace"])).await;
    let plan = plan_from(plan_item);
    g.create_plan(&plan).await.unwrap();
    for i in 0..12 {
        g.create_task(plan.id, &task_titled(Some(&format!("t{i}")), "d"))
            .await
            .unwrap();
    }
    let mut rfc = note_with(
        None,
        NoteType::Rfc,
        &serde_json::json!({
            "title": rfc_item["label"],
            "sections": [{"title": "Refs", "content": "..."}]
        })
        .to_string(),
        vec![],
    );
    rfc.id = id_of(rfc_item);
    g.create_note(&rfc).await.unwrap();

    let mut params = fixture_params(&f);
    params.kinds = Some("plan,rfc".into());
    let got = run(g, params).await;
    assert_eq!(got["items"][0], *plan_item);
    assert_eq!(got["items"][1], *rfc_item);
    assert_eq!(got["items"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn the_task_item_is_the_fixture_item() {
    let f = fixture();
    let task_item = &f["response"]["items"][1];

    let g = MockGraphStore::new();
    seed_project(&g, task_item, None).await;
    let mut plan = PlanNode::new_for_project(
        "Chat : références #/@".into(),
        "d".into(),
        "t".into(),
        5,
        id_of(&task_item["project"]),
    );
    plan.status = PlanStatus::InProgress;
    g.create_plan(&plan).await.unwrap();
    let mut task = task_titled(Some(task_item["label"].as_str().unwrap()), "d");
    task.id = id_of(task_item);
    task.status = TaskStatus::InProgress;
    g.create_task(plan.id, &task).await.unwrap();

    let mut params = fixture_params(&f);
    params.kinds = Some("task".into());
    let got = run(g, params).await;
    assert_eq!(got["items"][0], *task_item);
}

#[tokio::test]
async fn an_empty_result_is_the_fixture_empty_response() {
    let f = fixture();
    let got = run(MockGraphStore::new(), fixture_params(&f)).await;
    assert_eq!(got, f["empty_response"]);
}

#[test]
fn the_fixture_request_parses_into_the_documented_query() {
    let f = fixture();
    let q = fixture_params(&f).into_query().unwrap();
    assert_eq!(q.q, "refs");
    assert_eq!(q.limit, 20);
    assert_eq!(
        q.kinds.iter().map(|k| k.as_str()).collect::<Vec<_>>(),
        ["plan", "task", "rfc"]
    );
}
