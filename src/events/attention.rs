//! `attention_changed` — the light signal "something changed in what waits on
//! the user", relayed on `/ws/events`.
//!
//! The payload is deliberately minimal: identifiers, the workspace slug and
//! the REASONS (a closed enum). It NEVER carries the text of a command or of a
//! question: that text stays on the session's own WebSocket and in
//! `GET /api/attention`. The cockpit refetches, it does not rebuild state from
//! events.
//!
//! Emission sites call [`notify_attention`]. The raw event they emit is
//! coalesced by the [`AttentionRelay`] owned by the `HybridEmitter`: all
//! changes about the same subject arriving inside [`COALESCE_WINDOW`] produce
//! ONE event (reasons merged), enriched with the workspace slug.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use uuid::Uuid;

use super::types::{CrudAction, CrudEvent, EntityType, EventEmitter};
use crate::neo4j::traits::GraphStore;

/// Server-side coalescing window.
pub const COALESCE_WINDOW: Duration = Duration::from_millis(100);

/// Why a thread entered or left an attention band. Closed set: no free text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AttentionReason {
    PermissionRequest,
    AskUserQuestion,
    PermissionDecision,
    UserMessage,
    SessionActive,
    SessionInactive,
    PlanStarted,
    PlanCompleted,
    TaskFailed,
    BudgetExceeded,
}

impl AttentionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PermissionRequest => "permission_request",
            Self::AskUserQuestion => "ask_user_question",
            Self::PermissionDecision => "permission_decision",
            Self::UserMessage => "user_message",
            Self::SessionActive => "session_active",
            Self::SessionInactive => "session_inactive",
            Self::PlanStarted => "plan_started",
            Self::PlanCompleted => "plan_completed",
            Self::TaskFailed => "task_failed",
            Self::BudgetExceeded => "budget_exceeded",
        }
    }
}

/// What the change is about.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AttentionSubject {
    Session(String),
    Plan(String),
    /// A runner run (when the plan id is not at hand at the emission site).
    PlanRun(String),
}

impl AttentionSubject {
    fn id_key(&self) -> &'static str {
        match self {
            Self::Session(_) => "session_id",
            Self::Plan(_) => "plan_id",
            Self::PlanRun(_) => "run_id",
        }
    }

    fn id(&self) -> &str {
        match self {
            Self::Session(i) | Self::Plan(i) | Self::PlanRun(i) => i,
        }
    }

    fn from_payload(payload: &serde_json::Value) -> Option<Self> {
        let get = |k: &str| payload.get(k).and_then(|v| v.as_str()).map(str::to_string);
        get("session_id")
            .map(Self::Session)
            .or_else(|| get("plan_id").map(Self::Plan))
            .or_else(|| get("run_id").map(Self::PlanRun))
    }
}

impl CrudEvent {
    /// Raw `attention_changed` event (before coalescing / enrichment).
    pub fn attention_changed(subject: &AttentionSubject, reason: AttentionReason) -> Self {
        CrudEvent::new(
            EntityType::AttentionChanged,
            CrudAction::Updated,
            subject.id(),
        )
        .with_payload(serde_json::json!({
            subject.id_key(): subject.id(),
            "reasons": [reason.as_str()],
        }))
    }
}

/// Emit an `attention_changed` through `emitter` (fire-and-forget).
pub fn notify_attention(
    emitter: &Option<Arc<dyn EventEmitter>>,
    subject: AttentionSubject,
    reason: AttentionReason,
) {
    if let Some(e) = emitter {
        e.emit(CrudEvent::attention_changed(&subject, reason));
    }
}

/// Resolves the workspace slug a subject belongs to (best effort).
#[async_trait]
pub trait AttentionScope: Send + Sync {
    async fn workspace_slug(&self, subject: &AttentionSubject) -> Option<ResolvedScope>;
}

/// Result of a scope resolution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedScope {
    pub workspace_slug: Option<String>,
    /// For a run: the plan it executes.
    pub plan_id: Option<String>,
}

/// [`AttentionScope`] backed by the graph store.
pub struct GraphAttentionScope(pub Arc<dyn GraphStore>);

#[async_trait]
impl AttentionScope for GraphAttentionScope {
    async fn workspace_slug(&self, subject: &AttentionSubject) -> Option<ResolvedScope> {
        let g = &self.0;
        match subject {
            AttentionSubject::Session(id) => {
                let s = g.get_chat_session(Uuid::parse_str(id).ok()?).await.ok()??;
                if let Some(ws) = s.workspace_slug {
                    return Some(ResolvedScope {
                        workspace_slug: Some(ws),
                        plan_id: None,
                    });
                }
                let project = g
                    .get_project_by_slug(s.project_slug.as_deref()?)
                    .await
                    .ok()??;
                let ws = g.get_project_workspace(project.id).await.ok()??;
                Some(ResolvedScope {
                    workspace_slug: Some(ws.slug),
                    plan_id: None,
                })
            }
            AttentionSubject::Plan(id) => plan_scope(g, Uuid::parse_str(id).ok()?).await,
            AttentionSubject::PlanRun(id) => {
                let run = g.get_plan_run(Uuid::parse_str(id).ok()?).await.ok()??;
                let mut r = plan_scope(g, run.plan_id).await.unwrap_or_default();
                r.plan_id = Some(run.plan_id.to_string());
                Some(r)
            }
        }
    }
}

async fn plan_scope(g: &Arc<dyn GraphStore>, plan_id: Uuid) -> Option<ResolvedScope> {
    let plan = g.get_plan(plan_id).await.ok()??;
    let ws = g.get_project_workspace(plan.project_id?).await.ok()??;
    Some(ResolvedScope {
        workspace_slug: Some(ws.slug),
        plan_id: None,
    })
}

struct Pending {
    reasons: BTreeSet<String>,
}

/// Coalesces `attention_changed` events per subject.
pub struct AttentionRelay {
    window: Duration,
    pending: Mutex<HashMap<AttentionSubject, Pending>>,
    scope: OnceLock<Arc<dyn AttentionScope>>,
}

impl Default for AttentionRelay {
    fn default() -> Self {
        Self::new(COALESCE_WINDOW)
    }
}

impl AttentionRelay {
    pub fn new(window: Duration) -> Self {
        Self {
            window,
            pending: Mutex::new(HashMap::new()),
            scope: OnceLock::new(),
        }
    }

    /// Install the workspace resolver (once; later calls are ignored).
    pub fn set_scope(&self, scope: Arc<dyn AttentionScope>) {
        let _ = self.scope.set(scope);
    }

    /// Queue `event`; `sink` receives the coalesced result after the window.
    /// Returns the event back when it cannot be coalesced (no async runtime,
    /// malformed payload): the caller then dispatches it right away.
    pub fn push<F>(self: &Arc<Self>, event: CrudEvent, sink: F) -> Option<CrudEvent>
    where
        F: Fn(CrudEvent) + Send + Sync + 'static,
    {
        let Some(subject) = AttentionSubject::from_payload(&event.payload) else {
            return Some(event);
        };
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return Some(event);
        };
        let reasons: Vec<String> = event
            .payload
            .get("reasons")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|r| r.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        let first = {
            let mut map = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            let first = map.is_empty();
            map.entry(subject)
                .or_insert_with(|| Pending {
                    reasons: BTreeSet::new(),
                })
                .reasons
                .extend(reasons);
            first
        };
        if first {
            let me = self.clone();
            handle.spawn(async move {
                tokio::time::sleep(me.window).await;
                let drained: Vec<(AttentionSubject, Pending)> = {
                    let mut map = me.pending.lock().unwrap_or_else(|e| e.into_inner());
                    map.drain().collect()
                };
                for (subject, p) in drained {
                    let resolved = match me.scope.get() {
                        Some(s) => s.workspace_slug(&subject).await.unwrap_or_default(),
                        None => ResolvedScope::default(),
                    };
                    let mut payload = serde_json::json!({
                        subject.id_key(): subject.id(),
                        "reasons": p.reasons.into_iter().collect::<Vec<_>>(),
                    });
                    if let Some(ws) = resolved.workspace_slug {
                        payload["workspace_slug"] = serde_json::Value::String(ws);
                    }
                    if let Some(pid) = resolved.plan_id {
                        payload["plan_id"] = serde_json::Value::String(pid);
                    }
                    sink(
                        CrudEvent::new(
                            EntityType::AttentionChanged,
                            CrudAction::Updated,
                            subject.id(),
                        )
                        .with_payload(payload),
                    );
                }
            });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{EventBus, HybridEmitter};

    struct FixedScope;
    #[async_trait]
    impl AttentionScope for FixedScope {
        async fn workspace_slug(&self, _s: &AttentionSubject) -> Option<ResolvedScope> {
            Some(ResolvedScope {
                workspace_slug: Some("ws-a".into()),
                plan_id: None,
            })
        }
    }

    fn hybrid() -> (HybridEmitter, tokio::sync::broadcast::Receiver<CrudEvent>) {
        let h = HybridEmitter::new(Arc::new(EventBus::default()));
        h.attention_relay().set_scope(Arc::new(FixedScope));
        let rx = h.subscribe();
        (h, rx)
    }

    #[tokio::test]
    async fn burst_on_one_session_is_one_event_with_merged_reasons() {
        let (h, mut rx) = hybrid();
        let sid = AttentionSubject::Session(Uuid::new_v4().to_string());
        let emitter: Option<Arc<dyn EventEmitter>> = Some(Arc::new(h));
        for r in [
            AttentionReason::PermissionRequest,
            AttentionReason::UserMessage,
            AttentionReason::PermissionRequest,
        ] {
            notify_attention(&emitter, sid.clone(), r);
        }
        // nothing before the window closes
        assert!(rx.try_recv().is_err());
        tokio::time::sleep(COALESCE_WINDOW * 3).await;
        let ev = rx.try_recv().expect("one coalesced event");
        assert!(rx.try_recv().is_err(), "exactly one event for the burst");
        assert_eq!(ev.entity_type, EntityType::AttentionChanged);
        assert_eq!(ev.payload["workspace_slug"], "ws-a");
        assert_eq!(
            ev.payload["reasons"],
            serde_json::json!(["permission_request", "user_message"])
        );
    }

    #[tokio::test]
    async fn distinct_subjects_keep_distinct_events() {
        let (h, mut rx) = hybrid();
        let emitter: Option<Arc<dyn EventEmitter>> = Some(Arc::new(h));
        notify_attention(
            &emitter,
            AttentionSubject::Session(Uuid::new_v4().to_string()),
            AttentionReason::UserMessage,
        );
        notify_attention(
            &emitter,
            AttentionSubject::Plan(Uuid::new_v4().to_string()),
            AttentionReason::PlanStarted,
        );
        tokio::time::sleep(COALESCE_WINDOW * 3).await;
        assert!(rx.try_recv().is_ok());
        assert!(rx.try_recv().is_ok());
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn changes_after_the_window_start_a_new_event() {
        let (h, mut rx) = hybrid();
        let sid = AttentionSubject::Session(Uuid::new_v4().to_string());
        let emitter: Option<Arc<dyn EventEmitter>> = Some(Arc::new(h));
        notify_attention(&emitter, sid.clone(), AttentionReason::UserMessage);
        tokio::time::sleep(COALESCE_WINDOW * 3).await;
        notify_attention(&emitter, sid, AttentionReason::SessionInactive);
        tokio::time::sleep(COALESCE_WINDOW * 3).await;
        assert!(rx.try_recv().is_ok());
        assert!(rx.try_recv().is_ok());
    }

    #[test]
    fn without_runtime_the_event_goes_out_immediately() {
        let (h, mut rx) = hybrid();
        let emitter: Option<Arc<dyn EventEmitter>> = Some(Arc::new(h));
        notify_attention(
            &emitter,
            AttentionSubject::Session("s".into()),
            AttentionReason::UserMessage,
        );
        assert!(rx.try_recv().is_ok());
    }

    #[test]
    fn payload_never_contains_free_text() {
        let ev = CrudEvent::attention_changed(
            &AttentionSubject::Session("s1".into()),
            AttentionReason::PermissionRequest,
        );
        let keys: Vec<_> = ev.payload.as_object().unwrap().keys().cloned().collect();
        assert!(keys.iter().all(|k| k == "session_id" || k == "reasons"));
    }
}
