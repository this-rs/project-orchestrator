//! The provider-neutral session engine (task B13, decision A18).
//!
//! `CHAT_PROVIDER_PATH=agent` makes [`ChatManager`](super::manager::ChatManager)
//! drive a session through an `Arc<dyn AgentSession>` (nexus contract) instead
//! of the Claude `InteractiveClient`. This module owns what that needs and
//! nothing Claude specific: the live sessions, the turn driver that turns the
//! provider's event stream into `ChatEvent`s (through [`EventMapper`]), the
//! out-of-turn pump, persistence of the events for replay, and the answers to
//! permission requests, interrupts, model and mode changes.
//!
//! The legacy path stays as it was; the manager decides which one serves a
//! session and a session never changes engine.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use futures::StreamExt;
use nexus_claude::agent::{
    AgentEvent, AgentProvider, AgentSession, Capabilities, InterruptScope, PermissionDecision,
    PolicyMode, ProviderError, TurnInput,
};
use tokio::sync::{broadcast, Mutex, RwLock};
use uuid::Uuid;

use super::provider::event_map::{out_of_band_to_chat_events, EventMapper};
use super::types::ChatEvent;
use crate::neo4j::models::ChatEventRecord;
use crate::neo4j::GraphStore;

const BROADCAST_BUFFER: usize = 1024;
/// How long the out-of-turn pump waits for the rest of a provider message
/// before it flushes what it has.
const OOB_FLUSH: Duration = Duration::from_millis(50);

/// Replaces every secret value delivered by the vault inside an event of the
/// provider, before anything stores or broadcasts it (same rule as the Claude
/// path: `ChatManager::mask_cli_message`). Fails CLOSED: an event that holds a
/// secret and cannot be masked is replaced by a marker that surfaces as a
/// visible error, never passed on in clear.
pub(crate) fn mask_agent_event(event: AgentEvent) -> AgentEvent {
    let masker = crate::vault::mask::global().snapshot();
    mask_agent_event_with(&masker, event)
}

/// [`mask_agent_event`] against an explicit masker (testable).
pub(crate) fn mask_agent_event_with(
    masker: &crate::vault::mask::Masker,
    event: AgentEvent,
) -> AgentEvent {
    match crate::vault::mask::mask_serde(masker, event) {
        Ok(masked) => masked,
        Err(_unmasked) => {
            tracing::error!(
                "vault: a provider event holding a secret could not be masked; withholding it"
            );
            AgentEvent::ProviderNotice {
                kind: crate::chat::manager::MASKING_FAILED_SUBTYPE.to_string(),
                data: serde_json::Value::Null,
            }
        }
    }
}

/// What a session on the agent engine does NOT do, as the identifiers the
/// frontend knows (`hooks`, `message_queue`, `auto_continue`, `retry`,
/// `compaction`, `nats`, `enrichment`, `images`).
///
/// Two sources, kept apart on purpose:
/// - what THIS ENGINE (the backend) has not ported, whatever the provider can do:
///   hooks (no relay yet), message queue, auto-continue, retry, NATS fan-out, entity
///   enrichment;
/// - what THE SESSION's capabilities say it cannot do: `images`, and `compaction`
///   when the provider emits no compaction signal.
pub fn degraded_features(caps: &Capabilities) -> Vec<String> {
    let mut missing = vec![
        "hooks",
        "message_queue",
        "auto_continue",
        "retry",
        "nats",
        "enrichment",
    ];
    if !caps.compaction_signal {
        missing.push("compaction");
    }
    if !caps.images {
        missing.push("images");
    }
    missing.into_iter().map(str::to_string).collect()
}

/// Where the runtime finds a provider instance by identifier. The nexus
/// registry plugs in here; until then only the built-in instance exists.
pub trait ProviderSource: Send + Sync {
    /// The instance, when it exists.
    fn get(&self, provider_id: &str) -> Option<Arc<dyn AgentProvider>>;
}

/// One live session of the agent path.
pub struct AgentSessionHandle {
    /// Identifier of the session (the Neo4j `ChatSession` id).
    pub session_id: String,
    /// Provider instance serving it.
    pub provider_id: String,
    /// The provider session.
    pub session: Arc<dyn AgentSession>,
    /// Capabilities frozen at open.
    pub capabilities: Capabilities,
    /// Broadcast of the session's events.
    pub events_tx: broadcast::Sender<ChatEvent>,
    /// A turn is being streamed.
    pub is_streaming: AtomicBool,
    /// Text of the turn in progress (mid-stream joins).
    pub streaming_text: Mutex<String>,
    /// Structured events of the turn in progress (mid-stream joins).
    pub streaming_events: Mutex<Vec<ChatEvent>>,
    /// `{ id, kind }` stamped on `system_init`.
    provider: serde_json::Value,
    /// Tool policy stamped on `system_init`.
    tool_policy: serde_json::Value,
    /// What this session does not do (see [`degraded_features`]).
    degraded: Vec<String>,
    next_seq: AtomicI64,
    mapper: Mutex<EventMapper>,
    graph: Arc<dyn GraphStore>,
    uuid: Option<Uuid>,
}

impl AgentSessionHandle {
    /// Persists (except transient events) and broadcasts one event.
    pub async fn emit(&self, mut event: ChatEvent) {
        // What only the session owner knows rides on `system_init`.
        if let ChatEvent::SystemInit {
            provider,
            capabilities,
            tool_policy,
            engine,
            degraded_features,
            ..
        } = &mut event
        {
            // The client must be able to tell which engine drives the session,
            // and, when Claude Code was forced here, what it no longer does.
            engine.get_or_insert_with(|| "agent".to_string());
            degraded_features.get_or_insert_with(|| self.degraded.clone());
            provider.get_or_insert_with(|| self.provider.clone());
            capabilities.get_or_insert_with(|| {
                serde_json::to_value(&self.capabilities).unwrap_or_default()
            });
            tool_policy.get_or_insert_with(|| self.tool_policy.clone());
        }
        match &event {
            ChatEvent::StreamDelta { text, .. } => self.streaming_text.lock().await.push_str(text),
            ChatEvent::StreamingStatus { .. } | ChatEvent::PendingQueue { .. } => {}
            other => self.streaming_events.lock().await.push(other.clone()),
        }
        if !matches!(
            event,
            ChatEvent::StreamDelta { .. }
                | ChatEvent::StreamingStatus { .. }
                | ChatEvent::PendingQueue { .. }
        ) {
            if let Some(uuid) = self.uuid {
                let record = ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: uuid,
                    seq: self.next_seq.fetch_add(1, Ordering::SeqCst),
                    event_type: event.event_type().to_string(),
                    data: serde_json::to_string(&event).unwrap_or_default(),
                    created_at: chrono::Utc::now(),
                };
                let _ = self.graph.store_chat_events(uuid, vec![record]).await;
            }
        }
        let _ = self.events_tx.send(event);
    }

    /// Starts a turn and drives it to its terminal event in the background.
    /// A turn already running is `turn_in_progress`.
    pub async fn send_message(self: &Arc<Self>, text: &str) -> Result<()> {
        if self
            .is_streaming
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(anyhow::Error::new(ProviderError::TurnInProgress));
        }
        self.emit(ChatEvent::UserMessage {
            content: text.to_string(),
        })
        .await;
        let stream = match self.session.send_turn(TurnInput::text(text)).await {
            Ok(stream) => stream,
            Err(error) => {
                self.is_streaming.store(false, Ordering::SeqCst);
                return Err(anyhow::Error::new(error));
            }
        };
        self.streaming_text.lock().await.clear();
        self.streaming_events.lock().await.clear();
        self.emit(ChatEvent::StreamingStatus { is_streaming: true })
            .await;
        let me = Arc::clone(self);
        tokio::spawn(async move {
            let mut stream = stream;
            while let Some(event) = stream.next().await {
                // Terminal-ness is read on the original: a withheld terminal event
                // still ends the turn.
                let terminal = event.is_terminal();
                let event = mask_agent_event(event);
                let chat_events = me.mapper.lock().await.map(&event);
                for chat_event in chat_events {
                    me.emit(chat_event).await;
                }
                if terminal {
                    break;
                }
            }
            me.is_streaming.store(false, Ordering::SeqCst);
            me.streaming_text.lock().await.clear();
            me.streaming_events.lock().await.clear();
            me.emit(ChatEvent::StreamingStatus {
                is_streaming: false,
            })
            .await;
        });
        Ok(())
    }

    /// Answers a permission request.
    pub async fn answer_permission(&self, request_id: &str, allow: bool) -> Result<()> {
        let decision = if allow {
            PermissionDecision::allow_once()
        } else {
            PermissionDecision::deny()
        };
        self.session
            .answer_permission(request_id, decision)
            .await
            .map_err(anyhow::Error::new)?;
        self.emit(ChatEvent::PermissionDecision {
            id: request_id.to_string(),
            allow,
        })
        .await;
        Ok(())
    }

    /// Interrupts the turn and the tools it runs.
    pub async fn interrupt(&self) -> Result<()> {
        self.session
            .interrupt(InterruptScope::TurnAndTools)
            .await
            .map(|_| ())
            .map_err(anyhow::Error::new)
    }

    /// Changes the model live.
    pub async fn set_model(&self, model: &str) -> Result<()> {
        self.session
            .set_model(model)
            .await
            .map_err(anyhow::Error::new)?;
        self.emit(ChatEvent::ModelChanged {
            model: model.to_string(),
        })
        .await;
        Ok(())
    }

    /// Changes the policy mode live.
    pub async fn set_policy_mode(&self, mode: PolicyMode, native: Option<&str>) -> Result<()> {
        self.session
            .set_policy_mode(mode, native)
            .await
            .map_err(anyhow::Error::new)?;
        self.emit(ChatEvent::PermissionModeChanged {
            mode: native
                .map(str::to_string)
                .unwrap_or_else(|| super::provider::policy::legacy_name(mode).to_string()),
            policy_mode: Some(super::provider::policy::neutral_name(mode).to_string()),
        })
        .await;
        Ok(())
    }
}

/// The live sessions of the agent path.
pub struct AgentRuntime {
    sessions: RwLock<HashMap<String, Arc<AgentSessionHandle>>>,
    graph: Arc<dyn GraphStore>,
}

impl AgentRuntime {
    /// A runtime persisting its events in `graph`.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            graph,
        }
    }

    /// The live session, if this runtime owns it.
    pub async fn get(&self, session_id: &str) -> Option<Arc<AgentSessionHandle>> {
        self.sessions.read().await.get(session_id).cloned()
    }

    /// Whether this runtime owns a live session of that id.
    pub async fn owns(&self, session_id: &str) -> bool {
        self.sessions.read().await.contains_key(session_id)
    }

    /// Number of live sessions.
    pub async fn len(&self) -> usize {
        self.sessions.read().await.len()
    }

    /// Registers a session just opened by a provider and starts its
    /// out-of-turn pump. `first_seq` is the next event number to persist.
    pub async fn adopt(
        &self,
        session_id: &str,
        provider_id: &str,
        session: Arc<dyn AgentSession>,
        first_seq: i64,
        provider_kind: &str,
        tool_policy: serde_json::Value,
    ) -> Arc<AgentSessionHandle> {
        let (events_tx, _) = broadcast::channel(BROADCAST_BUFFER);
        let handle = Arc::new(AgentSessionHandle {
            session_id: session_id.to_string(),
            provider_id: provider_id.to_string(),
            capabilities: session.capabilities().clone(),
            session: Arc::clone(&session),
            events_tx,
            is_streaming: AtomicBool::new(false),
            streaming_text: Mutex::new(String::new()),
            streaming_events: Mutex::new(Vec::new()),
            provider: serde_json::json!({ "id": provider_id, "kind": provider_kind }),
            tool_policy,
            degraded: degraded_features(session.capabilities()),
            next_seq: AtomicI64::new(first_seq),
            mapper: Mutex::new(EventMapper::new()),
            graph: Arc::clone(&self.graph),
            uuid: Uuid::parse_str(session_id).ok(),
        });
        if let Some(oob) = session.out_of_band() {
            let pump = Arc::clone(&handle);
            tokio::spawn(async move { pump_out_of_band(pump, oob).await });
        }
        self.sessions
            .write()
            .await
            .insert(session_id.to_string(), Arc::clone(&handle));
        handle
    }

    /// Closes a session: tells the provider, announces `session_closed` (A45),
    /// forgets it.
    pub async fn close(&self, session_id: &str) -> Result<()> {
        let handle = self
            .sessions
            .write()
            .await
            .remove(session_id)
            .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
        let _ = tokio::time::timeout(Duration::from_secs(5), handle.session.close()).await;
        handle
            .emit(ChatEvent::SessionClosed {
                session_id: session_id.to_string(),
                reason: Some("closed".to_string()),
            })
            .await;
        Ok(())
    }
}

/// Reads the out-of-turn stream, regroups what belongs to one provider
/// message and emits it: the batch is flushed when the provider pauses, when
/// it grows large, and when the stream ends.
async fn pump_out_of_band(
    handle: Arc<AgentSessionHandle>,
    mut stream: nexus_claude::agent::EventStream,
) {
    let mut batch = Vec::new();
    loop {
        let item = if batch.is_empty() {
            Some(stream.next().await)
        } else {
            tokio::time::timeout(OOB_FLUSH, stream.next()).await.ok()
        };
        match item {
            Some(Some(event)) => {
                batch.push(event);
                if batch.len() >= 64 {
                    flush(&handle, &mut batch).await;
                }
            }
            Some(None) => {
                flush(&handle, &mut batch).await;
                break;
            }
            None => flush(&handle, &mut batch).await,
        }
    }
}

async fn flush(handle: &AgentSessionHandle, batch: &mut Vec<nexus_claude::agent::AgentEvent>) {
    if batch.is_empty() {
        return;
    }
    let events: Vec<AgentEvent> = std::mem::take(batch)
        .into_iter()
        .map(mask_agent_event)
        .collect();
    let chat_events = {
        let mut mapper = handle.mapper.lock().await;
        out_of_band_to_chat_events(&events, &mut mapper, chrono::Utc::now())
    };
    for event in chat_events {
        handle.emit(event).await;
    }
}

/// The built-in provider instances: Claude Code, with the CLI path of the
/// server configuration. The nexus registry replaces this.
pub struct BuiltinProviders {
    claude: Arc<dyn AgentProvider>,
}

impl BuiltinProviders {
    /// The built-in instances.
    pub fn new(claude_cli_path: Option<String>) -> Self {
        use nexus_claude::providers::claude_code::{ClaudeCodeConfig, ClaudeCodeProvider};
        let mut config = ClaudeCodeConfig::default();
        config.cli_path = claude_cli_path.map(std::path::PathBuf::from);
        Self {
            claude: Arc::new(ClaudeCodeProvider::new(config)),
        }
    }
}

impl ProviderSource for BuiltinProviders {
    fn get(&self, provider_id: &str) -> Option<Arc<dyn AgentProvider>> {
        (provider_id == super::provider::resolver::CLAUDE_CODE).then(|| Arc::clone(&self.claude))
    }
}

/// A scriptable provider for tests: no process, no network.
#[cfg(test)]
pub(crate) mod fake {
    use super::*;
    use async_trait::async_trait;
    use futures::channel::mpsc::{unbounded, UnboundedSender};
    use nexus_claude::agent::{
        AgentEvent, CancelOutcome, CancelScope, EventStream, InterruptOutcome, ModelInfo,
        ProviderHealth, ProviderKind, QuestionAnswer, ResumeToken, SessionSpec,
    };
    use std::sync::Mutex as StdMutex;

    /// What the fake saw and what the test can push.
    #[derive(Default)]
    pub struct FakeState {
        pub permission_answers: StdMutex<Vec<(String, PermissionDecision)>>,
        pub interrupts: StdMutex<Vec<InterruptScope>>,
        pub models: StdMutex<Vec<String>>,
        pub modes: StdMutex<Vec<PolicyMode>>,
        pub closed: AtomicBool,
        pub turns_started: StdMutex<Vec<String>>,
        pub turn_tx: StdMutex<Option<UnboundedSender<AgentEvent>>>,
        pub oob_tx: StdMutex<Option<UnboundedSender<AgentEvent>>>,
        pub opened_specs: StdMutex<Vec<SessionSpec>>,
        pub resumed_with: StdMutex<Vec<ResumeToken>>,
    }

    impl FakeState {
        pub fn push(&self, event: AgentEvent) {
            self.turn_tx
                .lock()
                .unwrap()
                .as_ref()
                .expect("a turn is open")
                .unbounded_send(event)
                .unwrap();
        }
        pub fn push_oob(&self, event: AgentEvent) {
            self.oob_tx
                .lock()
                .unwrap()
                .as_ref()
                .expect("oob stream taken")
                .unbounded_send(event)
                .unwrap();
        }
        pub fn end_turn(&self) {
            self.turn_tx.lock().unwrap().take();
        }
    }

    pub struct FakeSession {
        pub state: Arc<FakeState>,
        caps: Capabilities,
        oob: StdMutex<Option<EventStream>>,
    }

    #[async_trait]
    impl AgentSession for FakeSession {
        fn capabilities(&self) -> &Capabilities {
            &self.caps
        }
        fn resume_token(&self) -> Option<ResumeToken> {
            Some(ResumeToken::claude_code_session("fake-provider-session"))
        }
        async fn send_turn(&self, input: TurnInput) -> Result<EventStream, ProviderError> {
            let text = input
                .blocks
                .iter()
                .filter_map(|b| match b {
                    nexus_claude::agent::InputBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<String>();
            self.state.turns_started.lock().unwrap().push(text);
            let (tx, rx) = unbounded();
            *self.state.turn_tx.lock().unwrap() = Some(tx);
            Ok(Box::pin(rx))
        }
        async fn answer_permission(
            &self,
            request_id: &str,
            decision: PermissionDecision,
        ) -> Result<(), ProviderError> {
            self.state
                .permission_answers
                .lock()
                .unwrap()
                .push((request_id.to_string(), decision));
            Ok(())
        }
        async fn answer_question(
            &self,
            _question_id: &str,
            _answer: QuestionAnswer,
        ) -> Result<(), ProviderError> {
            Err(ProviderError::unsupported("answer_question"))
        }
        async fn interrupt(
            &self,
            scope: InterruptScope,
        ) -> Result<InterruptOutcome, ProviderError> {
            self.state.interrupts.lock().unwrap().push(scope);
            self.state.end_turn();
            Ok(InterruptOutcome::default())
        }
        async fn cancel_tools(&self, _scope: CancelScope) -> Result<CancelOutcome, ProviderError> {
            Ok(CancelOutcome::default())
        }
        async fn set_model(&self, model: &str) -> Result<(), ProviderError> {
            self.state.models.lock().unwrap().push(model.to_string());
            Ok(())
        }
        async fn set_policy_mode(
            &self,
            mode: PolicyMode,
            _native: Option<&str>,
        ) -> Result<(), ProviderError> {
            self.state.modes.lock().unwrap().push(mode);
            Ok(())
        }
        fn out_of_band(&self) -> Option<EventStream> {
            self.oob.lock().unwrap().take()
        }
        async fn close(&self) -> Result<(), ProviderError> {
            self.state.closed.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    #[derive(Clone)]
    pub struct FakeProvider {
        pub state: Arc<FakeState>,
        pub fail_open: Arc<StdMutex<Option<ProviderError>>>,
        /// Capabilities the sessions of this provider declare.
        pub caps: Arc<StdMutex<Capabilities>>,
    }

    impl FakeProvider {
        pub fn new() -> Self {
            Self {
                state: Arc::new(FakeState::default()),
                fail_open: Arc::new(StdMutex::new(None)),
                caps: Arc::new(StdMutex::new(Capabilities::none())),
            }
        }
        fn session(&self) -> Arc<FakeSession> {
            let (tx, rx) = unbounded();
            *self.state.oob_tx.lock().unwrap() = Some(tx);
            Arc::new(FakeSession {
                state: Arc::clone(&self.state),
                caps: self.caps.lock().unwrap().clone(),
                oob: StdMutex::new(Some(Box::pin(rx))),
            })
        }
    }

    #[async_trait]
    impl AgentProvider for FakeProvider {
        fn id(&self) -> &str {
            "claude-code"
        }
        fn kind(&self) -> ProviderKind {
            ProviderKind::ClaudeCode
        }
        async fn health(&self) -> ProviderHealth {
            ProviderHealth::ok(None)
        }
        async fn catalog(&self) -> Result<Vec<ModelInfo>, ProviderError> {
            Ok(Vec::new())
        }
        fn capabilities(&self, _model: Option<&str>) -> Capabilities {
            Capabilities::none()
        }
        async fn open(&self, spec: SessionSpec) -> Result<Arc<dyn AgentSession>, ProviderError> {
            if let Some(e) = self.fail_open.lock().unwrap().take() {
                return Err(e);
            }
            self.state.opened_specs.lock().unwrap().push(spec);
            Ok(self.session())
        }
        async fn resume(
            &self,
            spec: SessionSpec,
            token: ResumeToken,
        ) -> Result<Arc<dyn AgentSession>, ProviderError> {
            self.state.resumed_with.lock().unwrap().push(token);
            self.state.opened_specs.lock().unwrap().push(spec);
            Ok(self.session())
        }
    }

    impl ProviderSource for FakeProvider {
        fn get(&self, provider_id: &str) -> Option<Arc<dyn AgentProvider>> {
            (provider_id == "claude-code").then(|| Arc::new(self.clone()) as Arc<dyn AgentProvider>)
        }
    }
}

#[cfg(test)]
mod mask_tests {
    use super::*;
    use crate::vault::mask::Masker;
    use nexus_claude::agent::ToolOutput;

    fn tool_result(text: &str) -> AgentEvent {
        AgentEvent::ToolResult {
            id: "t".into(),
            output: Some(ToolOutput::Text(text.into())),
            is_error: false,
            seq: None,
            parent: None,
        }
    }

    #[test]
    fn a_vault_secret_in_a_provider_event_is_masked() {
        let masker = Masker::from_values([("KEY", "sk-agent-secret-9876")]);
        let out = mask_agent_event_with(&masker, tool_result("the key is sk-agent-secret-9876 ok"));
        let wire = serde_json::to_string(&out).unwrap();
        assert!(!wire.contains("sk-agent-secret-9876"), "{wire}");
        assert!(
            wire.contains("the key is"),
            "the rest of the text is kept: {wire}"
        );
    }

    #[test]
    fn an_event_that_cannot_be_masked_is_withheld_not_passed_in_clear() {
        // The secret collides with a key of the event's own JSON form.
        let secret = "input_complete";
        let masker = Masker::from_values([("KEY", secret)]);
        let event = AgentEvent::ToolCall {
            id: "t".into(),
            name: "x".into(),
            input: serde_json::json!({ "note": format!("value {secret}") }),
            category: Default::default(),
            canonical: None,
            input_complete: true,
            seq: None,
            parent: None,
        };
        let out = mask_agent_event_with(&masker, event);
        assert!(
            matches!(&out, AgentEvent::ProviderNotice { kind, .. } if kind == crate::chat::manager::MASKING_FAILED_SUBTYPE),
            "{out:?}"
        );
        let chat = super::super::provider::event_map::EventMapper::new().map(&out);
        assert!(
            matches!(&chat[0], crate::chat::types::ChatEvent::Error { message, .. }
            if message == crate::chat::manager::MASKING_FAILED_MESSAGE)
        );
    }
}
