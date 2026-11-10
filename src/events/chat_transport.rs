//! Inter-instance chat transport, abstracted from the concrete bus.
//!
//! The CRUD [`EventEmitter`](super::types::EventEmitter) only models
//! fire-and-forget entity events. The chat domain needs more: per-session
//! publish/subscribe, interrupt and cancel-tools signalling, streaming
//! snapshots and a request/reply "cancel a task" relay. Today all of it is
//! consumed concretely as `Arc<NatsEmitter>`; this module introduces the
//! [`ChatTransport`] trait so a single-instance PO (no NATS) can run on an
//! in-memory transport and the NATS-backed path is one adapter among others.
//!
//! This is a **first increment**: only `publish_chat_event` /
//! `subscribe_chat_events` are migrated onto the trait. The other operations
//! are declared here and implemented by both transports, but their call sites
//! still go through `NatsEmitter` until the next slice.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use super::nats::{ChatRpcResponse, NatsEmitter};
use super::transport::{Message, Subject, Subscriber};
use super::types::TransportState;
use crate::chat::types::ChatEvent;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Why a cancel asked of another instance came back without an answer
/// ([`ChatTransport::request_cancel`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelayFailure {
    /// Nobody subscribes to the subject: no instance holds the session live.
    NoResponders,
    /// No answer within the timeout: the owner is there but did not answer.
    TimedOut,
    /// The request could not be sent or read.
    Transport(String),
}

/// Snapshot of a streaming session's current state, used for cross-instance mid-stream join.
///
/// When a client connects to a remote instance (one that doesn't own the streaming session),
/// this struct is returned via request/reply so the client can reconstruct the
/// in-progress assistant turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingSnapshot {
    pub is_streaming: bool,
    pub partial_text: String,
    pub events: Vec<ChatEvent>,
}

/// Inter-instance transport for chat: publish/subscribe of a session's events
/// plus the cross-instance control signals.
///
/// # Guarantees
///
/// * **Delivery** — `publish_*` methods are fire-and-forget: they return
///   immediately and a failure is logged, never raised to the caller. A
///   `subscribe_*` that returns `Ok` yields every message published *after*
///   the subscription was established (no replay of what came before).
/// * **Order** — within one session, events are delivered in publication
///   order. The NATS adapter preserves the per-connection order; the
///   in-memory transport is a FIFO broadcast per session.
/// * **Timeout** — `request_streaming_snapshot` waits 2 s, `request_cancel`
///   waits the transport's cancel timeout (10 s by default), then reports
///   `None` / [`RelayFailure::TimedOut`].
/// * **Mono-instance losses** — with no second instance, an interrupt, a
///   cancel-tools signal or a streaming-snapshot request has no remote
///   listener: the in-memory transport reports them as no-ops (an empty
///   subscription / `None` / [`RelayFailure::NoResponders`]). Nothing is
///   lost locally: the local broadcast (`events_tx`) still carries the event.
#[async_trait::async_trait]
pub trait ChatTransport: Send + Sync {
    /// Publish a [`ChatEvent`] to the session's channel. Fire-and-forget.
    fn publish_chat_event(&self, session_id: &str, event: ChatEvent);

    /// Subscribe to a session's chat events.
    async fn subscribe_chat_events(&self, session_id: &str) -> anyhow::Result<Subscriber>;

    /// Publish an interrupt signal for a session. Fire-and-forget.
    fn publish_chat_interrupt(&self, session_id: &str);

    /// Subscribe to a session's interrupt signals.
    async fn subscribe_chat_interrupt(&self, session_id: &str) -> anyhow::Result<Subscriber>;

    /// Publish a cancel-tools signal for a session. Fire-and-forget.
    fn publish_cancel_tools(&self, session_id: &str);

    /// Subscribe to a session's cancel-tools signals.
    async fn subscribe_cancel_tools(&self, session_id: &str) -> anyhow::Result<Subscriber>;

    /// Request a streaming snapshot from the instance that owns the session.
    async fn request_streaming_snapshot(&self, session_id: &str) -> Option<StreamingSnapshot>;

    /// Subscribe to streaming-snapshot requests (the owning instance answers each).
    async fn subscribe_snapshot_requests(&self, session_id: &str) -> anyhow::Result<Subscriber>;

    /// Publish an opaque payload on an opaque reply subject (the reply inbox of
    /// a request this instance received). Used by the request/reply relays
    /// (snapshots, RPC, `cancel_relay`) to answer an inbound request.
    async fn publish(&self, subject: &Subject, payload: Vec<u8>) -> anyhow::Result<()>;

    /// Flush the transport so publishes already queued reach the far side.
    async fn flush(&self) -> anyhow::Result<()>;

    /// Health of the transport connection, for the status endpoint:
    /// connected, disconnected, or unknown (a transport with no remote).
    fn transport_state(&self) -> TransportState;

    /// Build the cancel-tools subject for a session.
    fn cancel_tools_subject(&self, session_id: &str) -> String;

    /// Build the cancel-task subject for a session.
    fn cancel_task_subject(&self, session_id: &str) -> String;

    /// Subscribe to cancel-task requests for a session (the owning instance
    /// answers each on its reply subject).
    async fn subscribe_cancel_task(&self, session_id: &str) -> anyhow::Result<Subscriber>;

    /// Subscribe to RPC "send a message" requests for a session (the owning
    /// instance processes each and answers with a `ChatRpcResponse`).
    async fn subscribe_rpc_send(&self, session_id: &str) -> anyhow::Result<Subscriber>;

    /// Ask the instance that owns a session to send a message to it (the RPC
    /// "send a message" request), returning its `ChatRpcResponse`, or `None`
    /// when nobody answers within the timeout.
    async fn request_send_message(
        &self,
        session_id: &str,
        message: &str,
        message_type: &str,
    ) -> Option<ChatRpcResponse>;

    /// Ask the instance that owns a session to cancel it, returning its raw
    /// answer or why there is none ([`RelayFailure`]).
    /// Not an `async fn`: the provisional-answer predicate is a higher-ranked
    /// `Fn(&[u8])`, which `#[async_trait]` would otherwise tie to the future's
    /// lifetime and break the delegation to the NATS emitter.
    fn request_cancel<'a>(
        &'a self,
        subject: String,
        payload: serde_json::Value,
        is_provisional: &'a (dyn Fn(&[u8]) -> bool + Sync),
        grace: Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, RelayFailure>> + Send + 'a>,
    >;
}

/// Identifier of the chat transport built by default — the NATS-backed adapter
/// in use today. An unknown `events.transport` is a startup error, so an
/// operator typo fails loudly instead of silently building another transport.
pub const DEFAULT_TRANSPORT: &str = "nats";

/// Default journal file name for the `journal` transport when
/// `events.journal_path` is not set. Relative to the process working directory;
/// the startup log announces the resolved location.
pub const DEFAULT_JOURNAL_PATH: &str = "chat-journal.log";

/// Which chat transport to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportKind {
    /// The NATS-backed adapter — cross-instance chat sync (current behaviour).
    Nats,
    /// The persisted, replayable journal — single instance, chat events survive
    /// a restart.
    Journal,
    /// The in-memory transport — single instance, no persistence.
    Memory,
}

/// Resolve the transport to build from its configuration value. Pure selection:
/// the construction itself lives in [`build_chat_transport`].
pub fn select_transport(value: &str) -> anyhow::Result<TransportKind> {
    match value {
        "nats" => Ok(TransportKind::Nats),
        "journal" => Ok(TransportKind::Journal),
        "memory" => Ok(TransportKind::Memory),
        other => anyhow::bail!(
            "unknown events.transport '{other}'; accepted values: ['nats', 'journal', 'memory']"
        ),
    }
}

/// Build the [`ChatTransport`] selected by `kind`.
///
/// This is the single construction point shared by the boot path and the
/// integration tests: the boot wiring passes the already connected NATS emitter
/// for [`TransportKind::Nats`] (returned unchanged), and this function builds
/// the journal or in-memory transport otherwise. The journal file's parent
/// directory is created if absent, and the resolved location plus the retention
/// bound are announced on the startup log.
pub fn build_chat_transport(
    kind: TransportKind,
    nats: Option<Arc<NatsEmitter>>,
    journal_path: Option<&str>,
    max_events: usize,
) -> anyhow::Result<Option<Arc<dyn ChatTransport>>> {
    match kind {
        TransportKind::Nats => Ok(nats.map(|e| {
            let t: Arc<dyn ChatTransport> = e;
            t
        })),
        TransportKind::Memory => {
            tracing::info!("Chat transport: in-memory (mono-instance, no persistence)");
            Ok(Some(Arc::new(InMemoryChatTransport::new())))
        }
        TransportKind::Journal => {
            let path = std::path::PathBuf::from(journal_path.unwrap_or(DEFAULT_JOURNAL_PATH));
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        anyhow::anyhow!("cannot create journal directory {}: {e}", parent.display())
                    })?;
                }
            }
            let journal = JournalTransport::open(&path, max_events)?;
            tracing::info!(
                path = %path.display(),
                max_events = journal.max_events(),
                "Chat transport: journal — chat events are persisted and replayable"
            );
            Ok(Some(Arc::new(journal)))
        }
    }
}

/// In-memory [`ChatTransport`]: chat events stay in this process.
///
/// Lets a single-instance PO run without NATS. `publish_chat_event` broadcasts
/// to the process-local subscribers of the session; there are none in a single
/// instance, so the behaviour is a no-op exactly like the absent-NATS path.
/// Cross-instance control (`request_cancel`, snapshots) has no remote owner and
/// reports the documented mono-instance results.
#[derive(Clone, Default)]
pub struct InMemoryChatTransport {
    sessions: Arc<Mutex<HashMap<String, tokio::sync::broadcast::Sender<ChatEvent>>>>,
}

impl InMemoryChatTransport {
    /// A transport with no sessions yet.
    pub fn new() -> Self {
        Self::default()
    }

    fn sender(&self, session_id: &str) -> tokio::sync::broadcast::Sender<ChatEvent> {
        let mut sessions = self.sessions.lock().unwrap();
        sessions
            .entry(session_id.to_string())
            .or_insert_with(|| tokio::sync::broadcast::channel(1024).0)
            .clone()
    }
}

#[async_trait::async_trait]
impl ChatTransport for InMemoryChatTransport {
    fn publish_chat_event(&self, session_id: &str, event: ChatEvent) {
        let sessions = self.sessions.lock().unwrap();
        if let Some(tx) = sessions.get(session_id) {
            // No receivers -> `Err`, which is fine: nobody is listening.
            let _ = tx.send(event);
        }
    }

    async fn subscribe_chat_events(&self, session_id: &str) -> anyhow::Result<Subscriber> {
        let rx = self.sender(session_id).subscribe();
        let stream = tokio_stream::wrappers::BroadcastStream::new(rx)
            .filter_map(|r| async move { r.ok() })
            .map(|event: ChatEvent| Message {
                payload: serde_json::to_vec(&event).unwrap_or_default(),
                reply: None,
            });
        Ok(Subscriber::from_stream(stream))
    }

    fn publish_chat_interrupt(&self, _session_id: &str) {
        // Mono-instance: an interrupt is delivered in-process, never over the
        // transport, so there is no remote listener to signal.
    }

    async fn subscribe_chat_interrupt(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }

    fn publish_cancel_tools(&self, _session_id: &str) {
        // Mono-instance: no remote owner to signal.
    }

    async fn subscribe_cancel_tools(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }

    async fn request_streaming_snapshot(&self, _session_id: &str) -> Option<StreamingSnapshot> {
        None
    }

    async fn subscribe_snapshot_requests(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }
    async fn publish(&self, _subject: &Subject, _payload: Vec<u8>) -> anyhow::Result<()> {
        // Mono-instance: no remote reply inbox to deliver to.
        Ok(())
    }

    async fn flush(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn transport_state(&self) -> TransportState {
        TransportState::Unknown
    }

    fn cancel_tools_subject(&self, session_id: &str) -> String {
        format!("chat.{}.cancel_tools", session_id)
    }

    fn cancel_task_subject(&self, session_id: &str) -> String {
        format!("chat.{}.cancel_task", session_id)
    }

    async fn subscribe_cancel_task(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }

    async fn subscribe_rpc_send(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }
    async fn request_send_message(
        &self,
        _session_id: &str,
        _message: &str,
        _message_type: &str,
    ) -> Option<ChatRpcResponse> {
        None
    }

    fn request_cancel<'a>(
        &'a self,
        _subject: String,
        _payload: serde_json::Value,
        _is_provisional: &'a (dyn Fn(&[u8]) -> bool + Sync),
        _grace: Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, RelayFailure>> + Send + 'a>,
    > {
        Box::pin(async move { Err(RelayFailure::NoResponders) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn in_memory_publish_reaches_a_subscriber_in_order() {
        let transport = InMemoryChatTransport::new();
        let mut sub = transport.subscribe_chat_events("s1").await.unwrap();

        transport.publish_chat_event(
            "s1",
            ChatEvent::UserMessage {
                content: "one".into(),
            },
        );
        transport.publish_chat_event(
            "s1",
            ChatEvent::UserMessage {
                content: "two".into(),
            },
        );

        let first: ChatEvent = serde_json::from_slice(&sub.next().await.unwrap().payload).unwrap();
        let second: ChatEvent = serde_json::from_slice(&sub.next().await.unwrap().payload).unwrap();
        assert!(matches!(first, ChatEvent::UserMessage { content } if content == "one"));
        assert!(matches!(second, ChatEvent::UserMessage { content } if content == "two"));
    }

    #[tokio::test]
    async fn in_memory_cancel_reports_no_responders() {
        let transport = InMemoryChatTransport::new();
        let outcome = transport
            .request_cancel(
                "ignored".into(),
                serde_json::json!({}),
                &|_| false,
                Duration::from_secs(1),
            )
            .await;
        assert_eq!(outcome, Err(RelayFailure::NoResponders));
    }
}

// ============================================================================
// JournalTransport — a persisted, replayable ChatTransport
// ============================================================================

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Default bound on the number of events retained in the journal.
///
/// The journal is append-only and *bounded*: once more than this many events
/// have been written, the oldest are truncated from the bottom and the bound is
/// announced ([`JournalTransport::max_events`], [`JournalTransport::oldest_cursor`]).
/// A client whose last-seen cursor is below the retained window receives
/// [`ReplayError::CursorTooOld`] — never a silently amputated stream.
///
/// Sized for a busy single instance: 100 000 chat events comfortably covers
/// more than a day of transcripts while keeping the journal small enough to
/// replay in one read. Callers that need a different bound open the journal
/// with [`JournalTransport::open`] and pass their own `max_events`.
pub const DEFAULT_MAX_EVENTS: usize = 100_000;

/// One persisted chat event and its position in the journal.
///
/// `cursor` is monotonic and never reused: it starts at 1 and, after a reopen,
/// continues from `last_cursor + 1` (the journal derives it from what is on
/// disk, so a restart can never collide).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEvent {
    pub cursor: u64,
    pub session_id: String,
    pub event: ChatEvent,
}

/// Why a replay could not be served.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    /// The requested cursor is older than the oldest event still retained: the
    /// journal has already truncated those events. `oldest` is the first cursor
    /// still available, `requested` is the `since` the client asked for.
    CursorTooOld { oldest: u64, requested: u64 },
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayError::CursorTooOld { oldest, requested } => write!(
                f,
                "replay cursor {requested} is older than the journal's retained window \
                 (oldest retained cursor {oldest}); a replay would be silently truncated"
            ),
        }
    }
}

impl std::error::Error for ReplayError {}

/// A [`ChatTransport`] that persists every chat event to an append-only journal
/// and can replay them after a restart.
///
/// # Storage
///
/// The journal is one file, one JSON [`JournalEvent`] per line, written with an
/// append and a `sync_data` so a process crash does not lose what was already
/// published. Retention is bounded: when the retained window exceeds
/// `max_events`, the oldest entries are truncated from the bottom and the file
/// is rewritten to the surviving window.
///
/// # Replay
///
/// A live client subscribes as usual ([`ChatTransport::subscribe_chat_events`],
/// no replay). A returning client calls [`JournalTransport::replay`] with its
/// last-seen cursor and receives every retained event after it, in order — or a
/// typed [`ReplayError::CursorTooOld`] if its cursor is below the window (the
/// loss is visible, never silent). The caller then subscribes for what comes
/// next; the replay-or-live union is at-least-once.
#[derive(Clone)]
pub struct JournalTransport {
    inner: Arc<JournalInner>,
}

struct JournalInner {
    state: Mutex<JournalState>,
    sessions: Mutex<HashMap<String, tokio::sync::broadcast::Sender<ChatEvent>>>,
    path: PathBuf,
    max_events: usize,
}

struct JournalState {
    entries: VecDeque<JournalEvent>,
    next_cursor: u64,
}

impl JournalTransport {
    /// Open (or create) the journal at `path`, retaining at most `max_events`.
    ///
    /// On an existing file the cursors are read back and continued (no
    /// collision across restarts), and any excess beyond `max_events` left on
    /// disk is truncated from the bottom before the transport is handed out.
    pub fn open(path: impl Into<PathBuf>, max_events: usize) -> anyhow::Result<Self> {
        if max_events == 0 {
            anyhow::bail!("journal retention (max_events) must be at least one event");
        }
        let path = path.into();

        let mut entries = VecDeque::new();
        let mut next_cursor = 1u64;
        if path.exists() {
            let bytes = std::fs::read(&path)?;
            for line in bytes.split(|b| *b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                let entry: JournalEvent = serde_json::from_slice(line).map_err(|e| {
                    anyhow::anyhow!("corrupt journal entry in {}: {e}", path.display())
                })?;
                next_cursor = next_cursor.max(entry.cursor + 1);
                entries.push_back(entry);
            }
        }

        if truncate_to(&mut entries, max_events) > 0 {
            rewrite_file(&path, &entries)?;
        }

        Ok(JournalTransport {
            inner: Arc::new(JournalInner {
                state: Mutex::new(JournalState {
                    entries,
                    next_cursor,
                }),
                sessions: Mutex::new(HashMap::new()),
                path,
                max_events,
            }),
        })
    }

    /// The retention bound this journal was opened with (announced, never implicit).
    pub fn max_events(&self) -> usize {
        self.inner.max_events
    }

    /// The oldest cursor still retained, if any event has been written yet.
    pub fn oldest_cursor(&self) -> Option<u64> {
        self.inner
            .state
            .lock()
            .unwrap()
            .entries
            .front()
            .map(|e| e.cursor)
    }

    /// The cursor the next published event will receive (monotonic, no reuse).
    pub fn next_cursor(&self) -> u64 {
        self.inner.state.lock().unwrap().next_cursor
    }

    /// Replay every retained event published after `since` (exclusive), in
    /// order. `since` is the last cursor the caller already holds (0 = none).
    ///
    /// Returns [`ReplayError::CursorTooOld`] when `since` is below the retained
    /// window — i.e. events between `since` and the window were truncated — so
    /// the loss is explicit instead of a silently amputated stream.
    pub fn replay(&self, since: u64) -> Result<Vec<JournalEvent>, ReplayError> {
        let state = self.inner.state.lock().unwrap();
        match state.entries.front() {
            None => Ok(Vec::new()),
            Some(oldest) => {
                let oldest = oldest.cursor;
                if since + 1 < oldest {
                    return Err(ReplayError::CursorTooOld {
                        oldest,
                        requested: since,
                    });
                }
                Ok(state
                    .entries
                    .iter()
                    .filter(|e| e.cursor > since)
                    .cloned()
                    .collect())
            }
        }
    }

    fn sender(&self, session_id: &str) -> tokio::sync::broadcast::Sender<ChatEvent> {
        let mut sessions = self.inner.sessions.lock().unwrap();
        sessions
            .entry(session_id.to_string())
            .or_insert_with(|| tokio::sync::broadcast::channel(1024).0)
            .clone()
    }
}

/// Drop the oldest entries until `max_events` remain; returns how many dropped.
fn truncate_to(entries: &mut VecDeque<JournalEvent>, max_events: usize) -> usize {
    let mut dropped = 0;
    while entries.len() > max_events {
        entries.pop_front();
        dropped += 1;
    }
    dropped
}

/// Append one entry as a JSON line, durably (write + sync).
fn append_entry(path: &Path, entry: &JournalEvent) -> anyhow::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    serde_json::to_writer(&mut file, entry)?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    Ok(())
}

/// Rewrite the whole file from the retained window (used when truncating the bottom).
fn rewrite_file(path: &Path, entries: &VecDeque<JournalEvent>) -> anyhow::Result<()> {
    let mut file = std::fs::File::create(path)?;
    for entry in entries {
        serde_json::to_writer(&mut file, entry)?;
        file.write_all(b"\n")?;
    }
    file.sync_data()?;
    Ok(())
}

#[async_trait::async_trait]
impl ChatTransport for JournalTransport {
    fn publish_chat_event(&self, session_id: &str, event: ChatEvent) {
        {
            let mut state = self.inner.state.lock().unwrap();
            let cursor = state.next_cursor;
            state.next_cursor += 1;
            let entry = JournalEvent {
                cursor,
                session_id: session_id.to_string(),
                event: event.clone(),
            };
            state.entries.push_back(entry.clone());
            if let Err(e) = append_entry(&self.inner.path, &entry) {
                tracing::error!(error = %e, "journal append failed");
            }
            if truncate_to(&mut state.entries, self.inner.max_events) > 0 {
                if let Err(e) = rewrite_file(&self.inner.path, &state.entries) {
                    tracing::error!(error = %e, "journal compaction failed");
                }
            }
        }

        // Live broadcast (fire-and-forget), after the durable append.
        let sessions = self.inner.sessions.lock().unwrap();
        if let Some(tx) = sessions.get(session_id) {
            let _ = tx.send(event);
        }
    }

    async fn subscribe_chat_events(&self, session_id: &str) -> anyhow::Result<Subscriber> {
        let rx = self.sender(session_id).subscribe();
        let stream = tokio_stream::wrappers::BroadcastStream::new(rx)
            .filter_map(|r| async move { r.ok() })
            .map(|event: ChatEvent| Message {
                payload: serde_json::to_vec(&event).unwrap_or_default(),
                reply: None,
            });
        Ok(Subscriber::from_stream(stream))
    }

    fn publish_chat_interrupt(&self, _session_id: &str) {}

    async fn subscribe_chat_interrupt(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }

    fn publish_cancel_tools(&self, _session_id: &str) {}

    async fn subscribe_cancel_tools(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }

    async fn request_streaming_snapshot(&self, _session_id: &str) -> Option<StreamingSnapshot> {
        None
    }

    async fn subscribe_snapshot_requests(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }

    async fn publish(&self, _subject: &Subject, _payload: Vec<u8>) -> anyhow::Result<()> {
        Ok(())
    }

    async fn flush(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn transport_state(&self) -> TransportState {
        TransportState::Unknown
    }

    fn cancel_tools_subject(&self, session_id: &str) -> String {
        format!("chat.{}.cancel_tools", session_id)
    }

    fn cancel_task_subject(&self, session_id: &str) -> String {
        format!("chat.{}.cancel_task", session_id)
    }

    async fn subscribe_cancel_task(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }

    async fn subscribe_rpc_send(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }

    async fn request_send_message(
        &self,
        _session_id: &str,
        _message: &str,
        _message_type: &str,
    ) -> Option<ChatRpcResponse> {
        None
    }

    fn request_cancel<'a>(
        &'a self,
        _subject: String,
        _payload: serde_json::Value,
        _is_provisional: &'a (dyn Fn(&[u8]) -> bool + Sync),
        _grace: Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, RelayFailure>> + Send + 'a>,
    > {
        Box::pin(async move { Err(RelayFailure::NoResponders) })
    }
}

#[cfg(test)]
mod journal_tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn retention_reports_cursor_too_old_not_amputated() {
        let dir = tempfile::tempdir().unwrap();
        let transport = JournalTransport::open(dir.path().join("journal.log"), 50).unwrap();

        for i in 0..100u32 {
            transport.publish_chat_event(
                "s1",
                ChatEvent::UserMessage {
                    content: format!("event-{i}"),
                },
            );
        }

        // Oldest retained = cursor 51: asking for 0 is below the window.
        assert_eq!(transport.oldest_cursor(), Some(51));
        assert_eq!(
            transport.replay(0).unwrap_err(),
            ReplayError::CursorTooOld {
                oldest: 51,
                requested: 0
            }
        );

        // Replaying from the exact boundary is complete, never amputated.
        let events = transport.replay(50).unwrap();
        assert_eq!(events.len(), 50);
        assert_eq!(events.first().unwrap().cursor, 51);
        assert_eq!(events.last().unwrap().cursor, 100);
    }

    #[tokio::test]
    async fn reopen_continues_cursors_and_replays_correct_events() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.log");

        {
            let transport = JournalTransport::open(&path, 1_000).unwrap();
            for i in 0..5u32 {
                transport.publish_chat_event(
                    "s1",
                    ChatEvent::UserMessage {
                        content: format!("first-{i}"),
                    },
                );
            }
            assert_eq!(transport.next_cursor(), 6);
        } // dropped == closed

        {
            let transport = JournalTransport::open(&path, 1_000).unwrap();
            assert_eq!(transport.next_cursor(), 6, "cursors continue, no collision");
            transport.publish_chat_event(
                "s1",
                ChatEvent::UserMessage {
                    content: "sixth".into(),
                },
            );
            assert_eq!(transport.next_cursor(), 7);

            let events = transport.replay(0).unwrap();
            assert_eq!(events.len(), 6);
            assert_eq!(events[0].cursor, 1);
            assert_eq!(events[5].cursor, 6);
            assert_eq!(events[5].session_id, "s1");
            assert!(
                matches!(&events[5].event, ChatEvent::UserMessage { content } if content == "sixth")
            );
        }
    }

    #[tokio::test]
    async fn live_subscriber_receives_while_publisher_publishes() {
        let dir = tempfile::tempdir().unwrap();
        let transport = JournalTransport::open(dir.path().join("journal.log"), 100).unwrap();

        let mut sub = transport.subscribe_chat_events("s1").await.unwrap();
        transport.publish_chat_event(
            "s1",
            ChatEvent::UserMessage {
                content: "live-one".into(),
            },
        );
        transport.publish_chat_event(
            "s1",
            ChatEvent::UserMessage {
                content: "live-two".into(),
            },
        );

        let first: ChatEvent = serde_json::from_slice(&sub.next().await.unwrap().payload).unwrap();
        let second: ChatEvent = serde_json::from_slice(&sub.next().await.unwrap().payload).unwrap();
        assert!(matches!(first, ChatEvent::UserMessage { content } if content == "live-one"));
        assert!(matches!(second, ChatEvent::UserMessage { content } if content == "live-two"));
    }

    #[tokio::test]
    async fn usable_through_the_chat_transport_trait_object() {
        let dir = tempfile::tempdir().unwrap();
        let transport: Box<dyn ChatTransport> =
            Box::new(JournalTransport::open(dir.path().join("journal.log"), 100).unwrap());

        let mut sub = transport.subscribe_chat_events("s1").await.unwrap();
        transport.publish_chat_event(
            "s1",
            ChatEvent::UserMessage {
                content: "trait".into(),
            },
        );

        let evt: ChatEvent = serde_json::from_slice(&sub.next().await.unwrap().payload).unwrap();
        assert!(matches!(evt, ChatEvent::UserMessage { content } if content == "trait"));
        assert_eq!(transport.transport_state(), TransportState::Unknown);
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;

    #[test]
    fn transport_selector_maps_known_values_and_names_rejected_values() {
        assert_eq!(select_transport("nats").unwrap(), TransportKind::Nats);
        assert_eq!(select_transport("journal").unwrap(), TransportKind::Journal);
        assert_eq!(select_transport("memory").unwrap(), TransportKind::Memory);

        let err = select_transport("kafka").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("kafka"),
            "error must name the rejected value: {msg}"
        );
        assert!(
            msg.contains("nats") && msg.contains("journal") && msg.contains("memory"),
            "error must name the accepted values: {msg}"
        );
    }

    #[tokio::test]
    async fn journal_transport_from_the_selection_path_persists_and_replays() {
        let dir = tempfile::tempdir().unwrap();
        // Nested path exercises the "parent directory is created if absent" path.
        let path = dir.path().join("nested").join("journal.log");
        let path_str = path.to_str().unwrap();

        // Same construction path as boot: selector, then builder.
        let kind = select_transport("journal").unwrap();
        let transport = build_chat_transport(kind, None, Some(path_str), DEFAULT_MAX_EVENTS)
            .unwrap()
            .expect("journal transport is always built");

        transport.publish_chat_event(
            "s1",
            ChatEvent::UserMessage {
                content: "persisted".into(),
            },
        );

        // The event is on disk, one JSON line per event.
        let bytes = std::fs::read(&path).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("\"cursor\":1"), "journal line: {text}");
        assert!(text.contains("persisted"), "journal line: {text}");

        // A fresh open (simulating a restart / replay client) replays it.
        let replay_client = JournalTransport::open(&path, DEFAULT_MAX_EVENTS).unwrap();
        let events = replay_client.replay(0).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].cursor, 1);
        assert!(
            matches!(&events[0].event, ChatEvent::UserMessage { content } if content == "persisted")
        );
    }

    #[tokio::test]
    async fn journal_transport_reports_cursor_too_old_for_a_truncated_window() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.log");
        let path_str = path.to_str().unwrap();

        // Build with a tiny retention bound so truncation happens quickly.
        let kind = select_transport("journal").unwrap();
        let transport = build_chat_transport(kind, None, Some(path_str), 5)
            .unwrap()
            .expect("journal transport is always built");

        for i in 0..10u32 {
            transport.publish_chat_event(
                "s1",
                ChatEvent::UserMessage {
                    content: format!("event-{i}"),
                },
            );
        }

        // Cursor 0 is now below the retained window (oldest = 6).
        let replay_client = JournalTransport::open(&path, 5).unwrap();
        assert_eq!(replay_client.oldest_cursor(), Some(6));
        assert_eq!(
            replay_client.replay(0).unwrap_err(),
            ReplayError::CursorTooOld {
                oldest: 6,
                requested: 0
            }
        );
    }
}
