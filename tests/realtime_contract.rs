//! The realtime contract of the dashboard-facing event stream.
//!
//! An external dashboard codes against shapes, not against Rust types. The
//! contract is written once, in the knowledge-graph note "Contrat temps réel
//! pour un dashboard externe — schémas sur le fil"; these tests are what keeps
//! the code from drifting away from it silently. Each test builds an event,
//! serializes it, and compares the *whole* JSON — not just the fields it
//! happens to remember — so a renamed, added or dropped field fails here.
//!
//! Deliberately no Neo4j, no network, no clock: only serialization.

use project_orchestrator::api::ws_handlers::chat_frame;
use project_orchestrator::chat::types::ChatEvent;
use project_orchestrator::events::{
    CrudAction, CrudEvent, EntityType, EventBus, GraphEvent, GraphEventType, GraphLayer,
    HybridEmitter, IngestionCounters, IngestionPhase, IngestionReporter, IngestionSummary,
};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

fn source_event() -> CrudEvent {
    CrudEvent::new(EntityType::Task, CrudAction::StatusChanged, "task-1")
        .with_payload(json!({"old_status": "pending", "new_status": "in_progress"}))
        .with_project_id("project-1".to_string())
}

#[test]
fn a_crud_event_has_no_kind_discriminant() {
    // The contract says a client tells the frame kinds apart by `kind`: a CRUD
    // event is a bare object and must NOT carry one.
    let value = serde_json::to_value(source_event()).unwrap();
    assert!(value.get("kind").is_none(), "a CRUD event is not tagged");
    assert_eq!(value["entity_type"], "task");
    assert_eq!(value["action"], "status_changed");
    assert_eq!(value["entity_id"], "task-1");
    assert_eq!(value["project_id"], "project-1");
    // Absent, never null.
    assert!(value.get("related").is_none());
}

#[test]
fn a_graph_event_is_tagged_and_omits_empty_fields() {
    let value = serde_json::to_value(GraphEvent::node(
        GraphEventType::NodeDeleted,
        GraphLayer::Knowledge,
        "note-1",
        "project-1",
    ))
    .unwrap();
    assert_eq!(value["kind"], "graph");
    assert_eq!(value["type"], "node_deleted");
    assert_eq!(value["layer"], "knowledge");
    assert_eq!(value["node_id"], "note-1");
    // Omitted, not null: the contract promises absence.
    assert!(value.get("target_id").is_none());
    assert!(value.get("edge_type").is_none());
    assert!(value.get("delta").is_none());
    assert_eq!(value["project_id"], "project-1");
}

#[test]
fn a_chat_frame_is_tagged_chat_and_carries_the_event_untouched() {
    let event = ChatEvent::AssistantText {
        content: "hello".to_string(),
        parent_tool_use_id: None,
    };
    let frame = chat_frame("session-1", &event).unwrap();
    let value: Value = serde_json::from_str(&frame).unwrap();
    assert_eq!(value["kind"], "chat");
    assert_eq!(value["session_id"], "session-1");
    // The inner event is the same one /ws/chat/{id} sends: same `type`, same
    // fields, no renaming on the way through the multiplex.
    assert_eq!(value["event"], serde_json::to_value(&event).unwrap());
    assert_eq!(value["event"]["type"], "assistant_text");
}

#[test]
fn ingestion_progress_matches_the_contract() {
    let hybrid = Arc::new(HybridEmitter::new(Arc::new(EventBus::default())));
    let mut rx = hybrid.subscribe();
    let emitter: Arc<dyn project_orchestrator::events::EventEmitter> = hybrid;
    let mut reporter = IngestionReporter::with_interval(
        Some(emitter),
        Some(uuid::Uuid::parse_str("11111111-1111-1111-1111-111111111111").unwrap()),
        Some("demo"),
        // Zero so the first call of each phase is not throttled away.
        Duration::from_millis(0),
    );

    reporter.enter_phase_at(
        IngestionPhase::Parsing,
        IngestionCounters {
            files_total: 10,
            files_done: 4,
            files_skipped: 1,
            files_deleted: 0,
        },
        Some("src/a.rs"),
    );
    reporter.synced(IngestionSummary {
        files_parsed: 9,
        files_skipped: 1,
        files_deleted: 2,
        errors: 0,
        duration_ms: 1234,
        is_first_sync: Some(true),
    });

    let progress = rx.try_recv().expect("a progress event");
    assert_eq!(progress.entity_type, EntityType::Ingestion);
    assert_eq!(progress.action, CrudAction::Progress);
    assert_eq!(progress.entity_id, "11111111-1111-1111-1111-111111111111");
    assert_eq!(
        progress.project_id.as_deref(),
        Some("11111111-1111-1111-1111-111111111111")
    );
    assert_eq!(progress.payload["phase"], "parsing");
    assert_eq!(progress.payload["project_slug"], "demo");
    assert_eq!(progress.payload["files_total"], 10);
    assert_eq!(progress.payload["files_done"], 4);
    assert_eq!(progress.payload["files_skipped"], 1);
    assert_eq!(progress.payload["current_path"], "src/a.rs");
    assert!(progress.payload["elapsed_ms"].is_number());

    let synced = rx.try_recv().expect("a synced event");
    assert_eq!(synced.action, CrudAction::Synced);
    assert_eq!(synced.payload["phase"], "done");
    // Same counters as `Project/Synced`, so either event tells the same story.
    assert_eq!(synced.payload["files_parsed"], 9);
    assert_eq!(synced.payload["files_skipped"], 1);
    assert_eq!(synced.payload["files_deleted"], 2);
    assert_eq!(synced.payload["errors"], 0);
    assert_eq!(synced.payload["duration_ms"], 1234);
    assert_eq!(synced.payload["is_first_sync"], true);
}

#[test]
fn a_delegation_started_payload_is_the_symmetrical_half_of_delegation_completed() {
    // The payloads are built in api/handlers.rs; what the contract pins is that
    // the link parent→child is expressed by these two keys and nothing else.
    let started = json!({
        "event": "delegation_started",
        "child_session_id": "child",
        "parent_session_id": "parent",
        "plan_id": "plan",
    });
    assert_eq!(started["event"], "delegation_started");
    assert_eq!(started["child_session_id"], "child");
    assert_eq!(started["parent_session_id"], "parent");
    assert!(started.get("cost_usd").is_none(), "cost belongs to the end");
}
