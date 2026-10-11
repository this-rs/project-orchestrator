//! A NATS broker for the tests: the core protocol only (INFO/CONNECT, PING/PONG,
//! SUB/UNSUB, PUB/HPUB → MSG/HMSG, `*` and `>` wildcards), in process, on a
//! loopback port. Enough for `async_nats` publish, subscribe and request/reply,
//! so the cross-instance paths of the chat run against a real client without a
//! `nats-server` on the machine.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::mpsc;

struct Sub {
    subject: String,
    tx: mpsc::UnboundedSender<Vec<u8>>,
    /// Messages left before the subscription ends (`UNSUB sid max`).
    left: Option<u64>,
}

#[derive(Default)]
struct State {
    /// (connection, sid) → subscription.
    subs: HashMap<(u64, String), Sub>,
}

/// A running broker; it stops with the test runtime.
pub(crate) struct TestBroker {
    pub url: String,
}

impl TestBroker {
    pub(crate) async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let state = Arc::new(Mutex::new(State::default()));
        let next_conn = Arc::new(AtomicU64::new(1));
        tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                let conn = next_conn.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(serve(socket, conn, port, Arc::clone(&state)));
            }
        });
        Self {
            url: format!("nats://127.0.0.1:{port}"),
        }
    }

    /// A client of this broker.
    pub(crate) async fn client(&self) -> async_nats::Client {
        async_nats::connect(&self.url).await.unwrap()
    }
}

fn matches(pattern: &str, subject: &str) -> bool {
    let mut p = pattern.split('.');
    let mut s = subject.split('.');
    loop {
        match (p.next(), s.next()) {
            (Some(">"), Some(_)) => return true,
            (Some("*"), Some(_)) => {}
            (Some(a), Some(b)) if a == b => {}
            (None, None) => return true,
            _ => return false,
        }
    }
}

/// Delivers one message to every matching subscription.
fn route(
    state_ref: &Mutex<State>,
    subject: &str,
    reply: Option<&str>,
    hdr: Option<&[u8]>,
    body: &[u8],
) {
    let mut state = state_ref.lock().unwrap();
    let mut done = Vec::new();
    let mut delivered = false;
    for (key, sub) in state.subs.iter_mut() {
        if !matches(&sub.subject, subject) {
            continue;
        }
        delivered = true;
        let reply = reply.map(|r| format!(" {r}")).unwrap_or_default();
        let mut frame = match hdr {
            Some(h) => format!(
                "HMSG {subject} {}{reply} {} {}\r\n",
                key.1,
                h.len(),
                h.len() + body.len()
            )
            .into_bytes(),
            None => format!("MSG {subject} {}{reply} {}\r\n", key.1, body.len()).into_bytes(),
        };
        if let Some(h) = hdr {
            frame.extend_from_slice(h);
        }
        frame.extend_from_slice(body);
        frame.extend_from_slice(b"\r\n");
        let _ = sub.tx.send(frame);
        if let Some(left) = sub.left.as_mut() {
            *left -= 1;
            if *left == 0 {
                done.push(key.clone());
            }
        }
    }
    for key in done {
        state.subs.remove(&key);
    }
    // A request nobody subscribes to: like nats-server for a client that sent
    // `no_responders` (async_nats always does), an empty 503 status message on the
    // reply subject, so the requester learns it at once.
    if let (false, Some(inbox)) = (delivered, reply) {
        drop(state);
        route_status(state_ref, inbox);
    }
}

/// The "no responders" status message on `inbox`.
fn route_status(state: &Mutex<State>, inbox: &str) {
    route(state, inbox, None, Some(b"NATS/1.0 503\r\n\r\n"), b"");
}

async fn serve(socket: tokio::net::TcpStream, conn: u64, port: u16, state: Arc<Mutex<State>>) {
    let (read, mut write) = socket.into_split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
    let info = format!(
        "INFO {{\"server_id\":\"test\",\"version\":\"2.10.0\",\"go\":\"go\",\"host\":\"127.0.0.1\",\"port\":{port},\"headers\":true,\"max_payload\":8388608,\"proto\":1,\"client_id\":{conn}}}\r\n"
    );
    let _ = tx.send(info.into_bytes());
    tokio::spawn(async move {
        while let Some(frame) = rx.recv().await {
            if write.write_all(&frame).await.is_err() {
                break;
            }
        }
    });
    let mut reader = BufReader::new(read);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        let Some(op) = parts.first() else { continue };
        match op.to_ascii_uppercase().as_str() {
            "PING" => {
                let _ = tx.send(b"PONG\r\n".to_vec());
            }
            "SUB" => {
                // SUB <subject> [queue] <sid>
                let subject = parts[1].to_string();
                let sid = parts[parts.len() - 1].to_string();
                state.lock().unwrap().subs.insert(
                    (conn, sid),
                    Sub {
                        subject,
                        tx: tx.clone(),
                        left: None,
                    },
                );
            }
            "UNSUB" => {
                let sid = parts[1].to_string();
                let mut state = state.lock().unwrap();
                match parts.get(2).and_then(|m| m.parse::<u64>().ok()) {
                    Some(max) if max > 0 => {
                        if let Some(sub) = state.subs.get_mut(&(conn, sid)) {
                            sub.left = Some(max);
                        }
                    }
                    _ => {
                        state.subs.remove(&(conn, sid));
                    }
                }
            }
            "PUB" => {
                // PUB <subject> [reply] <size>
                let size: usize = parts[parts.len() - 1].parse().unwrap_or(0);
                let reply = (parts.len() == 4).then(|| parts[2]);
                let mut body = vec![0u8; size + 2];
                if reader.read_exact(&mut body).await.is_err() {
                    break;
                }
                body.truncate(size);
                route(&state, parts[1], reply, None, &body);
            }
            "HPUB" => {
                // HPUB <subject> [reply] <hdr size> <total size>
                let total: usize = parts[parts.len() - 1].parse().unwrap_or(0);
                let hdr: usize = parts[parts.len() - 2].parse().unwrap_or(0);
                let reply = (parts.len() == 5).then(|| parts[2]);
                let mut all = vec![0u8; total + 2];
                if reader.read_exact(&mut all).await.is_err() {
                    break;
                }
                all.truncate(total);
                let (h, body) = all.split_at(hdr);
                route(&state, parts[1], reply, Some(h), body);
            }
            // CONNECT, PONG, anything else: nothing to answer (verbose is off).
            _ => {}
        }
    }
    state.lock().unwrap().subs.retain(|(c, _), _| *c != conn);
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[test]
    fn wildcards_match_like_nats() {
        assert!(matches("a.b", "a.b"));
        assert!(matches("a.*", "a.b"));
        assert!(!matches("a.*", "a.b.c"));
        assert!(matches("a.>", "a.b.c"));
        assert!(!matches("a.>", "a"));
        assert!(!matches("a.b", "a.c"));
    }

    #[tokio::test]
    async fn publish_subscribe_and_request_reply_work_through_the_broker() {
        let broker = TestBroker::start().await;
        let a = broker.client().await;
        let b = broker.client().await;
        let mut sub = a.subscribe("x.y").await.unwrap();
        a.flush().await.unwrap();
        b.publish("x.y", "hi".into()).await.unwrap();
        b.flush().await.unwrap();
        assert_eq!(&sub.next().await.unwrap().payload[..], b"hi");

        let mut service = a.subscribe("svc").await.unwrap();
        a.flush().await.unwrap();
        let responder = a.clone();
        tokio::spawn(async move {
            let msg = service.next().await.unwrap();
            responder
                .publish(msg.reply.unwrap(), "pong".into())
                .await
                .unwrap();
        });
        let reply = b.request("svc", "ping".into()).await.unwrap();
        assert_eq!(&reply.payload[..], b"pong");
    }
}
