//! La liaison chat ne se tait pas pendant qu'un tour tourne, sur un vrai
//! serveur et un vrai client : chaque test échoue si le handler cesse de
//! redire « ça tourne toujours » entre deux événements.
//!
//! MESURÉ le 18/11/2026, conversation deepseek `b4f70927` (log backend) : la
//! session parente n'a rien dit pendant **659 s** puis **273 s** (un sous-agent
//! délégué travaillait), et la page a reconnecté la socket **huit fois en
//! onze minutes** — chaque reconnexion rejouant tout le snapshot mi-flux, le
//! fil restant sur « … ». La cause n'est pas le transport : c'est le chien de
//! garde de la page (60 s sans trame sur une session qui se dit en cours) qui
//! ne peut pas distinguer un tour silencieux d'une socket morte, parce qu'un
//! `Ping` de WebSocket ne remonte jamais jusqu'à JavaScript.
use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message as WsMessage;

use crate::chat::manager::{test_support, ChatManager};
use crate::neo4j::traits::ChatStore;
use crate::test_helpers::{mock_app_state_with_graph, test_chat_session};

const WAIT: Duration = Duration::from_secs(10);

type Client =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

struct Rig {
    sid: String,
    addr: std::net::SocketAddr,
}

/// A server (no-auth mode) over a store holding one chat session, registered
/// live as streaming — or idle — without any CLI: the handler reads the
/// session's own flag, it never asks the CLI whether it is alive.
async fn rig(streaming: bool) -> Rig {
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let session = test_chat_session(None);
    graph.create_chat_session(&session).await.unwrap();
    let app_state = mock_app_state_with_graph(graph.clone());
    let manager = Arc::new(ChatManager::new_without_memory(
        graph.clone(),
        app_state.meili.clone(),
        test_support::chat_config(),
    ));
    let sid = session.id.to_string();
    // An explicit (never spawned) binary path: building the client must not
    // search the machine, and nothing here ever talks to a CLI.
    let client = nexus_claude::InteractiveClient::new(nexus_claude::ClaudeCodeOptions {
        model: Some("test".into()),
        cli_path: Some("/nonexistent/claude".into()),
        ..Default::default()
    })
    .expect("an explicit CLI path is never searched for");
    test_support::insert_session_with_client(&manager, &sid, client, streaming, &[]).await;
    let addr = crate::test_helpers::serve_chat(manager, graph).await;
    Rig { sid, addr }
}

async fn connect(rig: &Rig) -> Client {
    let url = format!(
        "ws://{}/ws/chat/{}?last_event=999999999999999",
        rig.addr, rig.sid
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
    ws.send(WsMessage::text("ready")).await.unwrap();
    ws
}

/// The next JSON frame satisfying `pred`, or `None` once `within` has passed.
async fn frame_where(
    ws: &mut Client,
    within: Duration,
    pred: impl Fn(&Value) -> bool,
) -> Option<Value> {
    let deadline = tokio::time::Instant::now() + within;
    loop {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        if left.is_zero() {
            return None;
        }
        match tokio::time::timeout(left, ws.next()).await {
            Err(_) => return None,
            Ok(None) | Ok(Some(Err(_))) => return None,
            Ok(Some(Ok(msg))) => {
                if let WsMessage::Text(t) = msg {
                    if let Ok(v) = serde_json::from_str::<Value>(t.as_str()) {
                        if pred(&v) {
                            return Some(v);
                        }
                    }
                }
            }
        }
    }
}

/// Waits for one beat of the socket. A `Ping` frame proves the keepalive arm
/// actually ran, so an ABSENCE of `streaming_status` measured afterwards is a
/// fact about the guard — not about a timer that never fired. Without this, the
/// idle test would pass on a handler that emits nothing at all.
async fn wait_for_ping(ws: &mut Client, within: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + within;
    loop {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        if left.is_zero() {
            return false;
        }
        match tokio::time::timeout(left, ws.next()).await {
            Err(_) => return false,
            Ok(None) | Ok(Some(Err(_))) => return false,
            Ok(Some(Ok(WsMessage::Ping(_)))) => return true,
            Ok(Some(Ok(_))) => {}
        }
    }
}

fn streaming_status(v: &Value) -> bool {
    v["type"] == "streaming_status"
}

/// A turn that has nothing to say — a delegated sub-agent, a long command —
/// must still tell the client it is alive, otherwise the page's stall watchdog
/// reconnects every 80 s and the thread never settles.
#[tokio::test]
async fn a_turn_that_says_nothing_still_tells_the_client_it_is_alive() {
    let rig = rig(true).await;
    let mut ws = connect(&rig).await;

    // Joining mid-stream already says it once (Phase 1.5b snapshot) — that is
    // the frame the client would otherwise be left with, for minutes.
    let attach = frame_where(&mut ws, WAIT, streaming_status)
        .await
        .expect("the mid-stream join announces the running turn");
    assert_eq!(attach["is_streaming"], true, "{attach}");

    // Then, with no turn event at all, the beat keeps the client's clock fresh.
    for beat in 1..=2 {
        let kept = frame_where(&mut ws, WAIT, streaming_status)
            .await
            .unwrap_or_else(|| panic!("keepalive {beat} never came: a silent turn looks dead"));
        assert_eq!(kept["is_streaming"], true, "{kept}");
    }
}

/// The frame must mean something: an idle session says nothing, so a client
/// cannot mistake silence for a running turn (or the reverse).
#[tokio::test]
async fn an_idle_session_says_nothing() {
    let rig = rig(false).await;
    let mut ws = connect(&rig).await;

    // Prove the beat runs before concluding from its silence: one Ping is the
    // same arm of the same `select!` that would carry the frame.
    assert!(
        wait_for_ping(&mut ws, WAIT).await,
        "no keepalive beat ever reached this client: the absence below would prove nothing"
    );

    // Then four more beats of the same interval, with nothing to announce.
    let seen = frame_where(&mut ws, Duration::from_millis(1200), streaming_status).await;
    assert!(
        seen.is_none(),
        "an idle session announced a running turn: {seen:?}"
    );
}
