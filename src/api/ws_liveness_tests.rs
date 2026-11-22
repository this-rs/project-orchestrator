//! Un pair qui garde le TCP ouvert mais ne répond pas aux `Ping` est coupé, sur
//! CHACUNE des quatre sockets, avec un transport BRUT qui n'auto-répond pas.
//!
//! Les tests de `keepalive_ws_tests.rs` utilisent `tokio-tungstenite`, dont le
//! client renvoie les `Pong` tout seul dès qu'on lit le flux : ils ne peuvent
//! pas voir ce défaut, et un simple EOF ne distingue pas « coupé par le serveur
//! pour silence » de « coupé par le client ». Ici le client est un `TcpStream`
//! sur lequel on écrit la poignée de main et les trames à la main : il lit les
//! `Ping`, et ne répond que quand le test le décide. Les durées (250 ms de
//! période, 600 ms d'échéance) sont celles de `cfg(test)` de `ws_liveness`.

use std::net::SocketAddr;
use std::time::Duration;

use axum::extract::ws::Message;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::Instant;

use super::keepalive_ws_tests::rig;
use super::ws_liveness::{Liveness, WS_PING_EVERY, WS_PONG_TIMEOUT};

// ── Liveness, sans réseau ──────────────────────────────────────────────

#[tokio::test(start_paused = true)]
async fn the_unanswered_ping_expires_after_the_timeout_not_before() {
    let mut live = Liveness::new(Duration::from_secs(60));
    let t0 = Instant::now();
    live.next_ping(t0).unwrap();
    assert!(live.next_ping(t0 + Duration::from_secs(30)).is_ok());
    let err = live.next_ping(t0 + Duration::from_secs(60)).unwrap_err();
    assert_eq!(err.waited, Duration::from_secs(60));
    assert_eq!(err.unanswered, 2);
}

#[tokio::test(start_paused = true)]
async fn a_pong_acknowledges_its_ping_and_the_older_ones() {
    let mut live = Liveness::new(Duration::from_secs(60));
    let t0 = Instant::now();
    let p0 = live.next_ping(t0).unwrap();
    let p1 = live.next_ping(t0 + Duration::from_secs(30)).unwrap();
    assert_eq!(live.pending(), 2);
    assert!(
        live.on_pong(&p1),
        "a Pong for the newest acknowledges the older"
    );
    assert_eq!(live.pending(), 0);
    assert!(
        !live.on_pong(&p0),
        "a second Pong for a gone ping counts for nothing"
    );
    assert!(live.next_ping(t0 + Duration::from_secs(300)).is_ok());
}

#[tokio::test(start_paused = true)]
async fn malformed_unsolicited_or_future_pongs_acknowledge_nothing() {
    let mut live = Liveness::new(Duration::from_secs(60));
    let t0 = Instant::now();
    live.next_ping(t0).unwrap();
    assert!(!live.on_pong(&[]));
    assert!(!live.on_pong(&[0, 1, 2]));
    assert!(!live.on_pong(&99u64.to_be_bytes()));
    assert_eq!(live.pending(), 1);
    assert!(live.next_ping(t0 + Duration::from_secs(60)).is_err());
}

#[tokio::test(start_paused = true)]
async fn answering_only_the_old_ping_leaves_the_new_one_due() {
    let mut live = Liveness::new(Duration::from_secs(60));
    let t0 = Instant::now();
    let p0 = live.next_ping(t0).unwrap();
    live.next_ping(t0 + Duration::from_secs(30)).unwrap();
    assert!(live.on_pong(&p0));
    // The second ping (sent at +30 s) is the oldest left: due at +90 s.
    assert!(live.next_ping(t0 + Duration::from_secs(89)).is_ok());
    assert!(live.next_ping(t0 + Duration::from_secs(150)).is_err());
}

// ── Transport brut ─────────────────────────────────────────────────────

struct Raw {
    tcp: TcpStream,
    buf: Vec<u8>,
}

#[derive(Debug)]
enum Frame {
    Ping(Vec<u8>),
    Close,
    Other,
    /// Le serveur a fermé le TCP (EOF ou RST).
    Eof,
    /// Rien reçu dans le délai : le TCP est resté ouvert.
    Quiet,
}

impl Raw {
    async fn connect(addr: SocketAddr, path: &str) -> Raw {
        let tcp = TcpStream::connect(addr).await.unwrap();
        Raw::upgrade(tcp, addr, path).await
    }

    async fn upgrade(tcp: TcpStream, addr: SocketAddr, path: &str) -> Raw {
        Raw::upgrade_with(tcp, addr, path, 600).await
    }

    async fn upgrade_with(mut tcp: TcpStream, addr: SocketAddr, path: &str, pong_ms: u64) -> Raw {
        let req = format!(
            "GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\
             Sec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
             x-test-ws-ping-ms: 250\r\nx-test-ws-pong-ms: {pong_ms}\r\n\r\n"
        );
        tcp.write_all(req.as_bytes()).await.unwrap();
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            tcp.read_exact(&mut byte).await.unwrap();
            head.push(byte[0]);
        }
        let head = String::from_utf8_lossy(&head).to_string();
        assert!(head.starts_with("HTTP/1.1 101"), "upgrade refused: {head}");
        Raw {
            tcp,
            buf: Vec::new(),
        }
    }

    /// Une trame cliente (masquée, clé nulle : légal, la charge reste telle quelle).
    async fn send(&mut self, opcode: u8, payload: &[u8]) {
        let mut f = vec![0x80 | opcode];
        match payload.len() {
            n if n < 126 => f.push(0x80 | n as u8),
            n if n <= 0xFFFF => {
                f.push(0x80 | 126);
                f.extend_from_slice(&(n as u16).to_be_bytes());
            }
            n => {
                f.push(0x80 | 127);
                f.extend_from_slice(&(n as u64).to_be_bytes());
            }
        }
        f.extend_from_slice(&[0, 0, 0, 0]);
        f.extend_from_slice(payload);
        // Bounded: a server that stopped reading must not block the test client.
        let _ = tokio::time::timeout(Duration::from_millis(300), self.tcp.write_all(&f)).await;
    }

    async fn next(&mut self, within: Duration) -> Frame {
        let deadline = Instant::now() + within;
        loop {
            if let Some((frame, used)) = parse(&self.buf) {
                self.buf.drain(..used);
                return frame;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            let mut chunk = [0u8; 4096];
            match tokio::time::timeout(left, self.tcp.read(&mut chunk)).await {
                Err(_) => return Frame::Quiet,
                Ok(Ok(0)) | Ok(Err(_)) => return Frame::Eof,
                Ok(Ok(n)) => self.buf.extend_from_slice(&chunk[..n]),
            }
        }
    }
}

fn parse(buf: &[u8]) -> Option<(Frame, usize)> {
    if buf.len() < 2 {
        return None;
    }
    let opcode = buf[0] & 0x0f;
    let (len, head) = match buf[1] & 0x7f {
        126 if buf.len() >= 4 => (u16::from_be_bytes([buf[2], buf[3]]) as usize, 4),
        127 if buf.len() >= 10 => (
            u64::from_be_bytes(buf[2..10].try_into().unwrap()) as usize,
            10,
        ),
        126 | 127 => return None,
        n => (n as usize, 2),
    };
    if buf.len() < head + len {
        return None;
    }
    let payload = buf[head..head + len].to_vec();
    let frame = match opcode {
        0x9 => Frame::Ping(payload),
        0x8 => Frame::Close,
        _ => Frame::Other,
    };
    Some((frame, head + len))
}

#[derive(Clone, Copy, Debug)]
enum Socket {
    Events,
    Activity,
    Run,
    Chat,
}

async fn open(which: Socket) -> Raw {
    let rig = rig(false).await;
    let (path, ready) = match which {
        Socket::Events => ("/ws/events".to_string(), true),
        Socket::Activity => ("/ws/activity".to_string(), true),
        Socket::Run => (format!("/ws/run/{}", uuid::Uuid::new_v4()), true),
        Socket::Chat => (
            format!("/ws/chat/{}?last_event=999999999999999", rig.sid),
            true,
        ),
    };
    let mut raw = Raw::connect(rig.addr, &path).await;
    if ready {
        raw.send(0x1, b"ready").await;
    }
    raw
}

/// Lit jusqu'à la fermeture. Renvoie (pings vus, instant de la fermeture depuis
/// `since`). Échoue si le serveur ne ferme pas avant `limit`.
async fn until_closed(raw: &mut Raw, since: Instant, limit: Duration) -> (usize, Duration) {
    let mut pings = 0;
    loop {
        match raw.next(limit.saturating_sub(since.elapsed())).await {
            Frame::Ping(_) => pings += 1,
            Frame::Close | Frame::Eof => {
                // Après un Close, le serveur doit aussi lâcher le TCP.
                let mut tail = raw.next(Duration::from_secs(3)).await;
                while matches!(tail, Frame::Other | Frame::Ping(_) | Frame::Close) {
                    tail = raw.next(Duration::from_secs(3)).await;
                }
                assert!(
                    matches!(tail, Frame::Eof),
                    "TCP still open after Close: {tail:?}"
                );
                return (pings, since.elapsed());
            }
            Frame::Other => {}
            Frame::Quiet => panic!("the server never closed a silent peer (after {pings} pings)"),
        }
    }
}

/// Un pair qui lit les Ping et ne répond JAMAIS est coupé, ni avant l'échéance,
/// ni longtemps après.
async fn silent_peer_is_closed(which: Socket) {
    let mut raw = open(which).await;
    let start = Instant::now();
    let (pings, at) = until_closed(&mut raw, start, WS_PONG_TIMEOUT * 5).await;
    assert!(pings >= 2, "{which:?}: only {pings} pings before the close");
    assert!(
        at >= WS_PONG_TIMEOUT,
        "{which:?}: closed after {at:?}, before the timeout {WS_PONG_TIMEOUT:?}"
    );
    assert!(
        at <= WS_PONG_TIMEOUT + WS_PING_EVERY * 3,
        "{which:?}: closed after {at:?}, far after the timeout"
    );
}

/// Un pair qui répond correctement reste connecté sur plusieurs échéances.
async fn answering_peer_stays_open(which: Socket) {
    let mut raw = open(which).await;
    let start = Instant::now();
    let mut answered = 0;
    while start.elapsed() < WS_PONG_TIMEOUT * 5 {
        match raw.next(WS_PING_EVERY * 4).await {
            Frame::Ping(p) => {
                raw.send(0xA, &p).await;
                answered += 1;
            }
            Frame::Close | Frame::Eof => panic!("{which:?}: closed a peer that answers"),
            Frame::Other | Frame::Quiet => {}
        }
    }
    assert!(
        answered as u128 >= (WS_PONG_TIMEOUT.as_millis() * 5 / WS_PING_EVERY.as_millis()) / 2,
        "{which:?}: only {answered} pings seen, the heartbeat did not run"
    );
}

/// Ni du trafic ordinaire, ni des Pong vides, inventés ou périmés ne prolongent.
async fn noise_does_not_extend(which: Socket) {
    let mut raw = open(which).await;
    let start = Instant::now();
    let mut first: Option<Vec<u8>> = None;
    let mut noise = tokio::time::interval(Duration::from_millis(50));
    let limit = WS_PONG_TIMEOUT * 5;
    let pings = loop {
        tokio::select! {
            f = raw.next(Duration::from_millis(40)) => match f {
                // Le premier Ping est acquitté une fois, correctement.
                Frame::Ping(p) if first.is_none() => {
                    raw.send(0xA, &p).await;
                    first = Some(p);
                }
                Frame::Close | Frame::Eof => break start.elapsed(),
                _ => {}
            },
            _ = noise.tick() => {
                raw.send(0x1, b"{\"type\":\"noise\"}").await; // trafic applicatif
                raw.send(0xA, b"").await;                       // Pong sans charge
                raw.send(0xA, &777u64.to_be_bytes()).await;     // jamais émis
                if let Some(p) = &first {
                    raw.send(0xA, p).await;                     // déjà acquitté : périmé
                }
            }
        }
        assert!(
            start.elapsed() < limit,
            "{which:?}: noise kept the peer alive"
        );
    };
    // Le premier Ping a été acquitté (à ~250 ms) ; le suivant (500 ms) ne l'est
    // jamais : échéance à 500+600, contrôlée au tick suivant. Une prolongation
    // par le bruit repousserait la coupure bien au-delà.
    assert!(
        pings >= WS_PONG_TIMEOUT,
        "{which:?}: closed too early ({pings:?})"
    );
    assert!(
        pings <= WS_PING_EVERY * 2 + WS_PONG_TIMEOUT + WS_PING_EVERY * 3,
        "{which:?}: noise extended the deadline ({pings:?})"
    );
}

macro_rules! per_socket {
    ($($name:ident => $f:ident($s:expr);)*) => {
        $(#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn $name() { $f($s).await }) *
    };
}

per_socket! {
    events_silent_peer_is_closed => silent_peer_is_closed(Socket::Events);
    activity_silent_peer_is_closed => silent_peer_is_closed(Socket::Activity);
    run_silent_peer_is_closed => silent_peer_is_closed(Socket::Run);
    chat_silent_peer_is_closed => silent_peer_is_closed(Socket::Chat);
    events_answering_peer_stays_open => answering_peer_stays_open(Socket::Events);
    activity_answering_peer_stays_open => answering_peer_stays_open(Socket::Activity);
    run_answering_peer_stays_open => answering_peer_stays_open(Socket::Run);
    chat_answering_peer_stays_open => answering_peer_stays_open(Socket::Chat);
    events_noise_does_not_extend => noise_does_not_extend(Socket::Events);
    activity_noise_does_not_extend => noise_does_not_extend(Socket::Activity);
    run_noise_does_not_extend => noise_does_not_extend(Socket::Run);
    chat_noise_does_not_extend => noise_does_not_extend(Socket::Chat);
}

// ── Un client qui CESSE DE LIRE ────────────────────────────────────────
//
// Les tests ci-dessus lisent les `Ping`. Ici le client ne lit PLUS RIEN, avec
// un tout petit tampon de réception, pendant que le serveur produit assez de
// messages pour remplir la sortie : l'écriture applicative bloque, la boucle ne
// revient plus à son `select!`, et sans écriture bornée l'échéance du `Pong`
// ne s'exécute jamais. On observe côté SERVEUR : on cherche, parmi les
// descripteurs de ce processus, celui dont le pair est l'adresse locale du
// client (la socket acceptée) ; il doit disparaître dans un délai borné.

use std::sync::Arc;

use crate::chat::manager::{test_support, ChatManager};
use crate::events::{CrudAction, CrudEvent, EntityType, EventEmitter};
use crate::neo4j::traits::ChatStore;
use crate::test_helpers::{mock_app_state_with_graph, test_chat_session};

struct Full {
    addr: SocketAddr,
    sid: String,
    manager: Arc<ChatManager>,
    bus: Arc<crate::events::HybridEmitter>,
}

async fn full_rig() -> Full {
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
    let client = nexus_claude::InteractiveClient::new(nexus_claude::ClaudeCodeOptions {
        model: Some("test".into()),
        cli_path: Some("/nonexistent/claude".into()),
        ..Default::default()
    })
    .unwrap();
    test_support::insert_session_with_client(&manager, &sid, client, false, &[]).await;
    let state = crate::test_helpers::server_state_with_chat(manager.clone(), graph).await;
    let bus = state.event_bus.clone();
    let app = crate::api::routes::create_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Full {
        addr,
        sid,
        manager,
        bus,
    }
}

/// Y a-t-il, dans ce processus, une socket dont le pair est `client_local` ?
/// C'est la socket acceptée par le serveur pour CE client, et elle seule.
fn server_side_socket_exists(client_local: SocketAddr) -> bool {
    use std::os::fd::FromRawFd;
    let Ok(dir) = std::fs::read_dir("/dev/fd") else {
        panic!("/dev/fd unreadable: cannot observe descriptors");
    };
    for entry in dir.flatten() {
        let Some(fd) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<i32>().ok())
        else {
            continue;
        };
        // SAFETY: never dropped (ManuallyDrop): the descriptor stays owned by its real owner.
        let sock = std::mem::ManuallyDrop::new(unsafe { std::net::TcpStream::from_raw_fd(fd) });
        if sock.peer_addr().ok() == Some(client_local) {
            return true;
        }
    }
    false
}

fn big() -> String {
    "x".repeat(128 * 1024)
}

fn crud(entity_type: EntityType, payload: serde_json::Value) -> CrudEvent {
    CrudEvent {
        entity_type,
        action: CrudAction::Updated,
        entity_id: big(),
        related: None,
        payload,
        timestamp: "2026-01-01T00:00:00Z".to_string(),
        project_id: None,
    }
}

/// One big application message for `which`, produced server-side.
async fn emit_one(which: Socket, rig: &Full, run_id: uuid::Uuid) {
    match which {
        Socket::Events | Socket::Activity => rig
            .bus
            .emit(crud(EntityType::Project, serde_json::json!({ "p": big() }))),
        Socket::Run => {
            let ev = crate::runner::RunnerEvent::PlanStarted {
                run_id,
                plan_id: uuid::Uuid::new_v4(),
                plan_title: big(),
                total_tasks: 1,
                total_waves: 1,
                prediction: None,
            };
            rig.bus
                .emit(crud(EntityType::Runner, serde_json::to_value(&ev).unwrap()))
        }
        Socket::Chat => {
            rig.manager
                .emit_on_session(
                    &rig.sid,
                    crate::chat::ChatEvent::UserMessage { content: big() },
                )
                .await
        }
    }
}

fn path_for(which: Socket, rig: &Full, run_id: uuid::Uuid) -> String {
    match which {
        Socket::Events => "/ws/events".to_string(),
        Socket::Activity => "/ws/activity".to_string(),
        Socket::Run => format!("/ws/run/{run_id}"),
        Socket::Chat => format!("/ws/chat/{}?last_event=999999999999999", rig.sid),
    }
}

/// Tests that block a server write (the stall tests, the error-path test, the race
/// tests) run one at a time: the race tests detect "the server is blocked" through
/// the process-wide [`BLOCKED_WRITES`] counter, which only means something if no
/// other test blocks a write meanwhile.
static BLOCKING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// A Pong that is READY while the loop is blocked in a long write must not lose
/// the race against the (overdue) Ping tick once the loop is free again.
///
/// Six clients connect to one server and read up to their first Ping. The server is
/// then asked for a lot of data while they read NOTHING; the test waits until the
/// server REPORTS that all six writes are blocked (the `BLOCKED_WRITES` counter
/// of `BoundedSink`: a measured precondition, not a sleep), and only then do the
/// clients answer their Ping. The Pongs sit unread in the server's socket for
/// 1.5 Pong deadlines; when the clients start reading, the loop frees up with BOTH
/// the overdue tick and the Pong ready. Without `biased` + reading first, the
/// select! picks the tick and closes a live connection.
async fn pong_race(which: Socket) {
    use super::ws_liveness::BLOCKED_WRITES;
    use std::sync::atomic::Ordering::SeqCst;
    const N: usize = 6;
    let _g = BLOCKING.lock().await;
    // A looser Pong deadline than the other tests: the point is the ORDER in which
    // a ready Pong and an overdue tick are served, not machine speed.
    let race_pong = Duration::from_millis(2500);
    let rig = full_rig().await;
    let run_id = uuid::Uuid::new_v4();
    let mut raws = Vec::new();
    let mut firsts = Vec::new();
    for _ in 0..N {
        let sock = tokio::net::TcpSocket::new_v4().unwrap();
        sock.set_recv_buffer_size(4096).unwrap();
        let tcp = sock.connect(rig.addr).await.unwrap();
        let mut raw = Raw::upgrade_with(tcp, rig.addr, &path_for(which, &rig, run_id), 2500).await;
        raw.send(0x1, b"ready").await;
        let first = loop {
            match raw.next(Duration::from_secs(3)).await {
                Frame::Ping(p) => break p,
                Frame::Close | Frame::Eof | Frame::Quiet => panic!("{which:?}: no first Ping"),
                Frame::Other => {}
            }
        };
        raws.push(raw);
        firsts.push(first);
    }
    // Make every server write block; the clients read nothing.
    let before = BLOCKED_WRITES.load(SeqCst);
    for _ in 0..60 {
        emit_one(which, &rig, run_id).await;
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while BLOCKED_WRITES.load(SeqCst) < before + N as u64 {
        assert!(
            Instant::now() < deadline,
            "{which:?}: precondition not met, the server never reported {N} blocked writes \
             (saw {})",
            BLOCKED_WRITES.load(SeqCst) - before
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // The server IS blocked: answer now, still without reading.
    for (raw, first) in raws.iter_mut().zip(&firsts) {
        raw.send(0xA, first).await;
    }
    tokio::time::sleep(race_pong * 3 / 2).await;
    // Then behave: read everything, answer every Ping, for several periods.
    let results =
        futures::future::join_all(raws.iter_mut().map(|raw| async move {
            let until = Instant::now() + race_pong * 3;
            let mut pings = 0;
            while Instant::now() < until {
                match raw.next(WS_PING_EVERY * 4).await {
                    Frame::Ping(p) => {
                        pings += 1;
                        raw.send(0xA, &p).await;
                    }
                    Frame::Close | Frame::Eof => return Err(
                        "a live connection was closed (the Pong lost the race against the tick)"
                            .to_string(),
                    ),
                    Frame::Other | Frame::Quiet => {}
                }
            }
            if pings < 2 {
                return Err(format!("heartbeat did not resume ({pings} pings)"));
            }
            Ok(())
        }))
        .await;
    let failures: Vec<_> = results.into_iter().filter_map(|r| r.err()).collect();
    assert!(
        failures.is_empty(),
        "{which:?}: {} of {N} connections failed: {failures:?}",
        failures.len()
    );
}

macro_rules! race_tests {
    ($($name:ident => $s:expr;)*) => {
        $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn $name() { pong_race($s).await }) *
    };
}

race_tests! {
    events_pong_in_flight_is_not_lost_to_the_tick => Socket::Events;
    activity_pong_in_flight_is_not_lost_to_the_tick => Socket::Activity;
    run_pong_in_flight_is_not_lost_to_the_tick => Socket::Run;
    chat_pong_in_flight_is_not_lost_to_the_tick => Socket::Chat;
}

async fn stalled_reader_is_dropped(which: Socket) {
    let _g = BLOCKING.lock().await;
    let rig = full_rig().await;
    let run_id = uuid::Uuid::new_v4();
    let (path, ready) = match which {
        Socket::Events => ("/ws/events".to_string(), true),
        Socket::Activity => ("/ws/activity".to_string(), true),
        Socket::Run => (format!("/ws/run/{run_id}"), true),
        Socket::Chat => (
            format!("/ws/chat/{}?last_event=999999999999999", rig.sid),
            true,
        ),
    };
    // A tiny receive buffer, set BEFORE connecting so the window stays small.
    let sock = tokio::net::TcpSocket::new_v4().unwrap();
    sock.set_recv_buffer_size(4096).unwrap();
    let tcp = sock.connect(rig.addr).await.unwrap();
    let client_local = tcp.local_addr().unwrap();
    let mut raw = Raw::upgrade(tcp, rig.addr, &path).await;
    if ready {
        raw.send(0x1, b"ready").await;
    }
    // The handler subscribes after the upgrade (and `ready`): let it get there.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        server_side_socket_exists(client_local),
        "{which:?}: no server-side socket"
    );

    // The server produces far more than the socket buffers hold; the client reads NOTHING.
    for _ in 0..200 {
        emit_one(which, &rig, run_id).await;
        tokio::task::yield_now().await;
    }

    // Bounded: the stalled write fails after WS_SEND_TIMEOUT, the handler leaves
    // its loop, the socket is dropped. Nothing else (no Pong, no read) helps it.
    // A 128 KiB message is allowed WS_SEND_TIMEOUT plus 16 s (8 KiB/s minimum rate).
    let deadline = Instant::now() + WS_PONG_TIMEOUT * 6 + Duration::from_secs(30);
    while server_side_socket_exists(client_local) {
        assert!(
            Instant::now() < deadline,
            "{which:?}: the server still holds the socket of a client that stopped reading"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    drop(raw);
}

macro_rules! stall_tests {
    ($($name:ident => $s:expr;)*) => {
        $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn $name() { stalled_reader_is_dropped($s).await }) *
    };
}

stall_tests! {
    events_stalled_reader_is_dropped => Socket::Events;
    activity_stalled_reader_is_dropped => Socket::Activity;
    run_stalled_reader_is_dropped => Socket::Run;
    chat_stalled_reader_is_dropped => Socket::Chat;
}

/// Unité : une écriture bloquée échoue après le délai, sans réseau.
#[tokio::test(start_paused = true)]
async fn a_blocked_sink_fails_after_its_limit() {
    use futures::SinkExt;
    // A sink that never becomes ready: `Poll::Pending` forever.
    struct Never;
    impl futures::Sink<Message> for Never {
        type Error = axum::Error;
        fn poll_ready(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Pending
        }
        fn start_send(self: std::pin::Pin<&mut Self>, _: Message) -> Result<(), Self::Error> {
            Ok(())
        }
        fn poll_flush(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Pending
        }
        fn poll_close(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Pending
        }
    }
    let mut sink = super::ws_liveness::BoundedSink::with_limit(Never, Duration::from_secs(5));
    let t0 = Instant::now();
    let r = sink.send(Message::Text("x".into())).await;
    assert!(r.is_err());
    assert_eq!(t0.elapsed(), Duration::from_secs(5));
}

/// Le chemin des erreurs du handler de chat, dont l'envoi ignore son résultat
/// (`let _ = send(...)`) : le client envoie des messages invalides dont l'erreur
/// de désérialisation répète le long nom de variante (donc une grosse réponse
/// à chaque fois), et ne lit rien. La connexion doit être fermée, ses
/// descripteurs libérés, dans un délai borné.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chat_error_reply_path_with_a_client_that_stopped_reading_is_dropped() {
    let _g = BLOCKING.lock().await;
    let rig = full_rig().await;
    let sock = tokio::net::TcpSocket::new_v4().unwrap();
    sock.set_recv_buffer_size(4096).unwrap();
    let tcp = sock.connect(rig.addr).await.unwrap();
    let client_local = tcp.local_addr().unwrap();
    let mut raw = Raw::upgrade(
        tcp,
        rig.addr,
        &format!("/ws/chat/{}?last_event=999999999999999", rig.sid),
    )
    .await;
    raw.send(0x1, b"ready").await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(server_side_socket_exists(client_local));
    let bad = format!("{{\"type\":\"{}\"}}", "v".repeat(60_000));
    for _ in 0..120 {
        raw.send(0x1, bad.as_bytes()).await;
    }
    let deadline = Instant::now() + WS_PONG_TIMEOUT * 8 + Duration::from_secs(30);
    while server_side_socket_exists(client_local) {
        assert!(
            Instant::now() < deadline,
            "the chat handler kept the socket of a client that stopped reading (error path)"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    drop(raw);
}

/// A sink whose flush takes `delay` (it advances, slowly) after each message.
struct SlowFlush {
    delay: Duration,
    sleep: Option<std::pin::Pin<Box<tokio::time::Sleep>>>,
}

impl futures::Sink<Message> for SlowFlush {
    type Error = axum::Error;
    fn poll_ready(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }
    fn start_send(mut self: std::pin::Pin<&mut Self>, _: Message) -> Result<(), Self::Error> {
        let d = self.delay;
        self.sleep = Some(Box::pin(tokio::time::sleep(d)));
        Ok(())
    }
    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        use std::future::Future;
        let this = self.get_mut();
        if let Some(sleep) = this.sleep.as_mut() {
            if sleep.as_mut().poll(cx).is_pending() {
                return std::task::Poll::Pending;
            }
        }
        this.sleep = None;
        std::task::Poll::Ready(Ok(()))
    }
    fn poll_close(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }
}

/// A multi-megabyte message to a slow client that IS advancing (a mobile link)
/// is not cut: the allowed time grows with the message (8 KiB/s minimum rate).
/// A client slower than that, or stopped, is. Red with a flat 60 s total bound.
#[tokio::test(start_paused = true)]
async fn a_big_message_to_a_slow_but_advancing_client_is_not_cut() {
    use futures::SinkExt;
    let big = Message::Text("x".repeat(1_000_000).into()); // 122 s extra at 8 KiB/s
    let mut ok = super::ws_liveness::BoundedSink::with_limit(
        SlowFlush {
            delay: Duration::from_secs(100),
            sleep: None,
        },
        Duration::from_secs(5),
    );
    let t0 = Instant::now();
    ok.send(big.clone())
        .await
        .expect("a slow client that advances must not be cut");
    assert_eq!(t0.elapsed(), Duration::from_secs(100));

    let mut stuck = super::ws_liveness::BoundedSink::with_limit(
        SlowFlush {
            delay: Duration::from_secs(500),
            sleep: None,
        },
        Duration::from_secs(5),
    );
    let t0 = Instant::now();
    assert!(stuck.send(big).await.is_err());
    assert_eq!(t0.elapsed(), Duration::from_secs(5 + 122));
}

/// A client blocked on a huge message is cut at the ABSOLUTE ceiling, not after
/// the proportional time (20 MiB would be 2560 s extra): red without the ceiling.
#[tokio::test(start_paused = true)]
async fn a_client_blocked_on_a_huge_message_is_cut_at_the_absolute_ceiling() {
    use futures::SinkExt;
    let huge = Message::Text("x".repeat(20 * 1024 * 1024).into());
    let mut stuck = super::ws_liveness::BoundedSink::with_limit(
        SlowFlush {
            delay: Duration::from_secs(1_000_000),
            sleep: None,
        },
        Duration::from_secs(5),
    );
    let t0 = Instant::now();
    assert!(stuck.send(huge).await.is_err());
    assert_eq!(t0.elapsed(), super::ws_liveness::WS_MAX_SEND_TIMEOUT);
}
