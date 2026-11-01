//! `/ws/activity` over a real socket: the real router on a loopback port, a real
//! WebSocket client, the real handler. Each test fails if the endpoint stops
//! refusing a malformed filter, stops attributing unscoped runner/chat events,
//! stops replaying what a reconnecting client missed, or stops saying that the
//! stream has a hole.
use crate::neo4j::traits::*;

use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message as WsMessage;
use uuid::Uuid;

use crate::api::handlers::ServerState;
use crate::chat::manager::{test_support, ChatManager};
use crate::events::{EntityType, EventEmitter};
use crate::neo4j::mock::MockGraphStore;
use crate::test_helpers::{
    mock_app_state_with_graph, router_with_auth, server_state_with_chat, test_chat_session,
    test_project_named, test_workspace,
};

const WAIT: Duration = Duration::from_secs(10);

type Client =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

struct Rig {
    state: Arc<ServerState>,
    graph: Arc<MockGraphStore>,
    manager: Arc<ChatManager>,
    addr: std::net::SocketAddr,
}

/// A server (no-auth mode) over an empty store, keeping the state so a test can
/// publish on the very bus the endpoint listens to.
async fn rig() -> Rig {
    let graph = Arc::new(MockGraphStore::new());
    let app_state = mock_app_state_with_graph(graph.clone());
    let manager = Arc::new(ChatManager::new_without_memory(
        graph.clone(),
        app_state.meili.clone(),
        test_support::chat_config(),
    ));
    let state = server_state_with_chat(manager.clone(), graph.clone()).await;
    let app = crate::api::routes::create_router(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Rig {
        state,
        graph,
        manager,
        addr,
    }
}

impl Rig {
    /// Open `/ws/activity?{query}` and complete the `ready` / `auth_ok` hand-off.
    async fn connect(&self, query: &str) -> Client {
        let url = format!("ws://{}/ws/activity?{}", self.addr, query);
        let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        ws.send(WsMessage::text("ready")).await.unwrap();
        frame_where(&mut ws, |v| v["type"] == "auth_ok").await;
        ws
    }

    /// The status the upgrade is refused with.
    async fn refused(&self, query: &str) -> u16 {
        let url = format!("ws://{}/ws/activity?{}", self.addr, query);
        match tokio_tungstenite::connect_async(url).await {
            Err(tokio_tungstenite::tungstenite::Error::Http(response)) => {
                response.status().as_u16()
            }
            other => panic!("expected a refused upgrade, got {:?}", other.map(|_| ())),
        }
    }

    fn emit(&self, entity_type: EntityType, id: &str, payload: Value, project: Option<Uuid>) {
        self.state
            .event_bus
            .emit_updated(entity_type, id, payload, project.map(|p| p.to_string()));
    }

    async fn project(&self, name: &str) -> crate::neo4j::models::ProjectNode {
        let project = test_project_named(name);
        self.graph.create_project(&project).await.unwrap();
        project
    }

    async fn plan_run(&self, project: Option<Uuid>) -> Uuid {
        let mut run = crate::runner::RunnerState::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            1,
            crate::runner::TriggerSource::Manual,
        );
        run.project_id = project;
        self.graph.create_plan_run(&run).await.unwrap();
        run.run_id
    }
}

/// The next JSON frame satisfying `pred`.
async fn frame_where(ws: &mut Client, pred: impl Fn(&Value) -> bool) -> Value {
    loop {
        let msg = tokio::time::timeout(WAIT, ws.next())
            .await
            .expect("a frame within 10 s")
            .expect("the socket is open")
            .expect("a frame");
        if let WsMessage::Text(t) = msg {
            if let Ok(v) = serde_json::from_str::<Value>(t.as_str()) {
                if pred(&v) {
                    return v;
                }
            }
        }
    }
}

async fn kind(ws: &mut Client, wanted: &'static str) -> Value {
    frame_where(ws, move |v| v["kind"] == wanted).await
}

/// No `activity` frame may arrive within `quiet`.
async fn nothing_for(ws: &mut Client, quiet: Duration) {
    let end = tokio::time::Instant::now() + quiet;
    loop {
        let left = end.saturating_duration_since(tokio::time::Instant::now());
        match tokio::time::timeout(left, ws.next()).await {
            Err(_) => return,
            Ok(Some(Ok(WsMessage::Text(t)))) => {
                let v: Value = serde_json::from_str(t.as_str()).unwrap_or(Value::Null);
                assert_ne!(v["kind"], "activity", "an event leaked: {v}");
            }
            Ok(Some(Ok(_))) => {}
            Ok(_) => return,
        }
    }
}

#[tokio::test]
async fn a_malformed_entity_type_is_refused_before_the_upgrade() {
    let rig = rig().await;
    assert_eq!(rig.refused("entity_types=plan_run,nope").await, 400);
}

#[tokio::test]
async fn an_unknown_workspace_is_a_404_not_a_silent_empty_feed() {
    let rig = rig().await;
    assert_eq!(rig.refused("workspace_slug=ghost").await, 404);
}

#[tokio::test]
async fn a_bad_credential_is_a_401_before_the_upgrade() {
    let graph = Arc::new(MockGraphStore::new());
    let app = router_with_auth(graph).await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    let url = format!("ws://{addr}/ws/activity?ticket=not-a-ticket");
    match tokio_tungstenite::connect_async(url).await {
        Err(tokio_tungstenite::tungstenite::Error::Http(response)) => {
            assert_eq!(response.status().as_u16(), 401)
        }
        other => panic!("expected 401, got {:?}", other.map(|_| ())),
    }
}

#[tokio::test]
async fn ready_frame_announces_the_sequence_the_live_stream_starts_from() {
    let rig = rig().await;
    rig.emit(EntityType::Task, "t1", json!({"title": "a"}), None);
    rig.emit(EntityType::Task, "t2", json!({"title": "b"}), None);
    let mut ws = rig.connect("").await;
    let ready = kind(&mut ws, "ready").await;
    assert_eq!(ready["seq"], 2);
    // No `last_seq`: replays from 0, so both buffered events come first.
    assert_eq!(ready["replayed"], 2);
    let first = kind(&mut ws, "activity").await;
    assert_eq!(first["event"]["entity_id"], "t1");
    assert_eq!(first["event"]["title"], "a");
}

#[tokio::test]
async fn last_seq_replays_only_what_the_client_missed() {
    let rig = rig().await;
    for id in ["t1", "t2", "t3"] {
        rig.emit(EntityType::Task, id, json!({}), None);
    }
    let mut ws = rig.connect("last_seq=1").await;
    let ready = kind(&mut ws, "ready").await;
    assert_eq!(ready["replayed"], 2);
    assert_eq!(kind(&mut ws, "activity").await["event"]["entity_id"], "t2");
    assert_eq!(kind(&mut ws, "activity").await["event"]["entity_id"], "t3");
}

#[tokio::test]
async fn a_replay_older_than_the_buffer_says_how_many_events_are_gone() {
    let rig = rig().await;
    // 1010 events into a 1000-event ring: seq 1..=10 are evicted.
    for _ in 0..1010 {
        rig.emit(EntityType::Task, "t", json!({}), None);
    }
    // Filtered down to nothing so the test reads two frames, not a thousand.
    let mut ws = rig.connect("entity_types=system").await;
    let ready = kind(&mut ws, "ready").await;
    assert_eq!(ready["seq"], 1010);
    let lag = kind(&mut ws, "lag_dropped").await;
    assert_eq!(lag["skipped"], 10);
}

#[tokio::test]
async fn an_unscoped_plan_run_event_is_attributed_to_its_project_and_filtered_on_it() {
    let rig = rig().await;
    let mine = rig.project("mine").await;
    let theirs = rig.project("theirs").await;
    let my_run = rig.plan_run(Some(mine.id)).await;
    let their_run = rig.plan_run(Some(theirs.id)).await;
    let mut ws = rig.connect(&format!("project_id={}", mine.id)).await;
    kind(&mut ws, "ready").await;

    // The runner publishes with `project_id: None`: the index fills it in.
    rig.emit(
        EntityType::Runner,
        &their_run.to_string(),
        json!({"status": "running"}),
        None,
    );
    rig.emit(
        EntityType::Runner,
        &my_run.to_string(),
        json!({"status": "running"}),
        None,
    );
    let got = kind(&mut ws, "activity").await;
    assert_eq!(got["event"]["entity_id"], my_run.to_string());
    assert_eq!(got["event"]["project_id"], mine.id.to_string());
}

#[tokio::test]
async fn a_protocol_run_is_attributed_through_the_index_too() {
    let rig = rig().await;
    let mine = rig.project("mine").await;
    let mut run = crate::protocol::ProtocolRun::new(Uuid::new_v4(), Uuid::new_v4(), "start");
    run.project_id = Some(mine.id);
    rig.graph.create_protocol_run(&run).await.unwrap();
    let mut ws = rig.connect(&format!("project_id={}", mine.id)).await;
    kind(&mut ws, "ready").await;

    rig.emit(
        EntityType::ProtocolRun,
        &run.id.to_string(),
        json!({"status": "running"}),
        None,
    );
    let got = kind(&mut ws, "activity").await;
    assert_eq!(got["event"]["category"], "protocol_run");
    assert_eq!(got["event"]["project_id"], mine.id.to_string());
}

#[tokio::test]
async fn a_conversation_is_matched_on_its_project_slug() {
    let rig = rig().await;
    let mine = rig.project("mine").await;
    let ours = test_chat_session(Some(&mine.slug));
    let other = test_chat_session(Some("elsewhere"));
    for s in [&ours, &other] {
        rig.graph.create_chat_session(s).await.unwrap();
        test_support::insert_mock_cli_session(&rig.manager, &s.id.to_string()).await;
    }
    let mut ws = rig.connect(&format!("project_id={}", mine.id)).await;
    kind(&mut ws, "ready").await;

    rig.emit(
        EntityType::ChatSession,
        &other.id.to_string(),
        json!({}),
        None,
    );
    rig.emit(
        EntityType::ChatSession,
        &ours.id.to_string(),
        json!({}),
        None,
    );
    let got = kind(&mut ws, "activity").await;
    assert_eq!(got["event"]["entity_id"], ours.id.to_string());
}

#[tokio::test]
async fn a_workspace_feed_keeps_the_projects_of_that_workspace_only() {
    let rig = rig().await;
    let inside = rig.project("inside").await;
    let outside = rig.project("outside").await;
    let ws_node = test_workspace();
    rig.graph.create_workspace(&ws_node).await.unwrap();
    rig.graph
        .add_project_to_workspace(ws_node.id, inside.id)
        .await
        .unwrap();
    let mut ws = rig
        .connect(&format!("workspace_slug={}", ws_node.slug))
        .await;
    kind(&mut ws, "ready").await;

    rig.emit(EntityType::Plan, "p-out", json!({}), Some(outside.id));
    rig.emit(EntityType::Plan, "p-in", json!({}), Some(inside.id));
    let got = kind(&mut ws, "activity").await;
    assert_eq!(got["event"]["entity_id"], "p-in");
}

#[tokio::test]
async fn a_run_started_after_connect_is_found_by_a_forced_refresh() {
    let rig = rig().await;
    let mine = rig.project("mine").await;
    let mut ws = rig.connect(&format!("project_id={}", mine.id)).await;
    kind(&mut ws, "ready").await;

    // The run exists in the graph but the index (built at connect) never saw it,
    // and the refresh floor is two seconds: wait it out, then publish.
    let run = rig.plan_run(Some(mine.id)).await;
    tokio::time::sleep(Duration::from_millis(2100)).await;
    rig.emit(EntityType::Runner, &run.to_string(), json!({}), None);
    let got = kind(&mut ws, "activity").await;
    assert_eq!(got["event"]["entity_id"], run.to_string());
    assert_eq!(got["event"]["project_id"], mine.id.to_string());
}

#[tokio::test]
async fn a_client_that_cannot_keep_up_is_told_how_many_events_it_missed() {
    let rig = rig().await;
    let mut ws = rig.connect("entity_types=system").await;
    kind(&mut ws, "ready").await;
    // Current-thread runtime: this loop never yields, so the endpoint's task does
    // not drain its receiver and the 1024-slot channel overflows.
    for _ in 0..1500 {
        rig.emit(EntityType::Task, "t", json!({}), None);
    }
    let lag = kind(&mut ws, "lag_dropped").await;
    assert!(lag["skipped"].as_u64().unwrap() > 0, "{lag}");
}

#[tokio::test]
async fn a_status_filter_drops_events_of_another_status_on_the_live_stream() {
    let rig = rig().await;
    let mut ws = rig.connect("statuses=running").await;
    kind(&mut ws, "ready").await;
    rig.emit(
        EntityType::Task,
        "done",
        json!({"status": "completed"}),
        None,
    );
    nothing_for(&mut ws, Duration::from_millis(300)).await;
    rig.emit(EntityType::Task, "go", json!({"status": "running"}), None);
    assert_eq!(kind(&mut ws, "activity").await["event"]["entity_id"], "go");
}

#[tokio::test]
async fn closing_the_socket_releases_the_subscription() {
    let rig = rig().await;
    let hub = rig.state.event_bus.activity().clone();
    let mut ws = rig.connect("").await;
    kind(&mut ws, "ready").await;
    assert_eq!(hub.subscriber_count(), 1);
    ws.close(None).await.unwrap();
    let end = tokio::time::Instant::now() + WAIT;
    while hub.subscriber_count() != 0 {
        assert!(
            tokio::time::Instant::now() < end,
            "the endpoint kept its subscription after the client left"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn a_client_message_other_than_close_does_not_end_the_feed() {
    let rig = rig().await;
    let mut ws = rig.connect("").await;
    kind(&mut ws, "ready").await;
    ws.send(WsMessage::text("noise")).await.unwrap();
    rig.emit(EntityType::Task, "after-noise", json!({}), None);
    assert_eq!(
        kind(&mut ws, "activity").await["event"]["entity_id"],
        "after-noise"
    );
}

#[tokio::test]
async fn a_pong_from_the_client_is_ignored_and_the_feed_goes_on() {
    let rig = rig().await;
    let mut ws = rig.connect("").await;
    kind(&mut ws, "ready").await;
    ws.send(WsMessage::Pong(Vec::new().into())).await.unwrap();
    rig.emit(EntityType::Task, "after-pong", json!({}), None);
    assert_eq!(
        kind(&mut ws, "activity").await["event"]["entity_id"],
        "after-pong"
    );
}

#[tokio::test]
async fn a_protocol_error_from_the_client_ends_the_connection_and_its_subscription() {
    use tokio::io::AsyncWriteExt;
    let rig = rig().await;
    let hub = rig.state.event_bus.activity().clone();
    let mut ws = rig.connect("").await;
    kind(&mut ws, "ready").await;
    assert_eq!(hub.subscriber_count(), 1);
    // An unmasked frame with a reserved opcode: not a WebSocket frame a client
    // may send, so the server side reads an error, not a message.
    match ws.get_mut() {
        tokio_tungstenite::MaybeTlsStream::Plain(tcp) => {
            tcp.write_all(&[0x83, 0x00]).await.unwrap();
            tcp.flush().await.unwrap();
        }
        _ => unreachable!("loopback is plain TCP"),
    }
    let end = tokio::time::Instant::now() + WAIT;
    while hub.subscriber_count() != 0 {
        assert!(
            tokio::time::Instant::now() < end,
            "a socket error must end the connection"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn a_failing_protocol_run_lookup_does_not_sink_the_feed() {
    let rig = rig().await;
    rig.graph
        .fail_reads
        .lock()
        .unwrap()
        .insert("list_all_protocol_runs");
    let mine = rig.project("mine").await;
    let run = rig.plan_run(Some(mine.id)).await;
    let mut ws = rig.connect(&format!("project_id={}", mine.id)).await;
    kind(&mut ws, "ready").await;
    // The plan-run half of the index still works: the run is attributed.
    rig.emit(EntityType::Runner, &run.to_string(), json!({}), None);
    let got = kind(&mut ws, "activity").await;
    assert_eq!(got["event"]["project_id"], mine.id.to_string());
}
