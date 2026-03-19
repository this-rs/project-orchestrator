//! # EventReactor — Reactive event processing engine
//!
//! The EventReactor subscribes to the [`EventBus`](super::EventBus) broadcast channel
//! and dispatches matching events to registered reaction handlers. It runs as a background
//! tokio task and is resilient to handler panics.
//!
//! ## Architecture
//!
//! ```text
//! EventBus (broadcast) ──► EventReactor (background loop)
//!                              │
//!                              ├─ Rule 1: Project::Synced  → bootstrap_knowledge_fabric
//!                              ├─ Rule 2: Task::StatusChanged(completed) → check plan progress
//!                              ├─ Rule 3: Plan::StatusChanged(completed) → collect episode
//!                              └─ Rule N: custom rules...
//! ```
//!
//! ## Built-in Reactions
//!
//! | EntityType | CrudAction    | Reaction                                    |
//! |------------|---------------|---------------------------------------------|
//! | Project    | Synced        | bootstrap_knowledge_fabric / update_fabric   |
//! | Task       | StatusChanged | Check if all plan tasks completed → auto-complete plan |
//! | Plan       | StatusChanged | Collect episode if protocol run linked       |

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use super::types::{CrudAction, CrudEvent, EntityType};

/// Type alias for async reaction handler functions.
///
/// Each handler receives the event and an opaque context (typically `Arc<ServerState>`).
/// Handlers are `Send + Sync` and return a pinned future.
pub type ReactionFn = Arc<
    dyn Fn(CrudEvent, Arc<dyn std::any::Any + Send + Sync>) -> Pin<Box<dyn Future<Output = ()> + Send>>
        + Send
        + Sync,
>;

/// A reaction rule that matches events and dispatches to a handler.
///
/// Both `entity_type` and `action` are optional filters:
/// - `None` means "match any"
/// - `Some(x)` means "match only x"
pub struct ReactionRule {
    /// Human-readable name for logging
    pub name: String,
    /// Filter by entity type (None = match all)
    pub entity_type: Option<EntityType>,
    /// Filter by action (None = match all)
    pub action: Option<CrudAction>,
    /// The async handler to invoke
    pub handler: ReactionFn,
}

impl ReactionRule {
    /// Check if this rule matches the given event.
    pub fn matches(&self, event: &CrudEvent) -> bool {
        let type_match = self
            .entity_type
            .as_ref()
            .map_or(true, |t| t == &event.entity_type);
        let action_match = self
            .action
            .as_ref()
            .map_or(true, |a| a == &event.action);
        type_match && action_match
    }
}

impl fmt::Debug for ReactionRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReactionRule")
            .field("name", &self.name)
            .field("entity_type", &self.entity_type)
            .field("action", &self.action)
            .finish()
    }
}

/// Runtime stats for the reactor, accessible from the status endpoint.
#[derive(Debug, Clone)]
pub struct ReactorStats {
    /// Whether the reactor background loop is running
    pub running: bool,
    /// Number of registered rules
    pub rules_count: usize,
    /// Total events received from the bus
    pub events_received: u64,
    /// Total events that matched at least one rule
    pub events_matched: u64,
    /// Total handler invocations (an event can trigger multiple handlers)
    pub handlers_invoked: u64,
    /// Total handler errors (panics caught)
    pub handler_errors: u64,
}

/// Shared counters for the reactor, thread-safe via atomics.
#[derive(Debug)]
pub struct ReactorCounters {
    pub running: AtomicBool,
    pub events_received: AtomicU64,
    pub events_matched: AtomicU64,
    pub handlers_invoked: AtomicU64,
    pub handler_errors: AtomicU64,
}

impl Default for ReactorCounters {
    fn default() -> Self {
        Self {
            running: AtomicBool::new(false),
            events_received: AtomicU64::new(0),
            events_matched: AtomicU64::new(0),
            handlers_invoked: AtomicU64::new(0),
            handler_errors: AtomicU64::new(0),
        }
    }
}

impl ReactorCounters {
    /// Snapshot the current counters into a `ReactorStats`.
    pub fn snapshot(&self, rules_count: usize) -> ReactorStats {
        ReactorStats {
            running: self.running.load(Ordering::Relaxed),
            rules_count,
            events_received: self.events_received.load(Ordering::Relaxed),
            events_matched: self.events_matched.load(Ordering::Relaxed),
            handlers_invoked: self.handlers_invoked.load(Ordering::Relaxed),
            handler_errors: self.handler_errors.load(Ordering::Relaxed),
        }
    }
}

/// The EventReactor — subscribes to the event bus and dispatches to handlers.
pub struct EventReactor {
    rules: Vec<ReactionRule>,
    receiver: broadcast::Receiver<CrudEvent>,
    context: Arc<dyn std::any::Any + Send + Sync>,
    counters: Arc<ReactorCounters>,
}

impl EventReactor {
    /// Get a reference to the shared counters (for the status endpoint).
    pub fn counters(&self) -> Arc<ReactorCounters> {
        Arc::clone(&self.counters)
    }

    /// Get the number of registered rules.
    pub fn rules_count(&self) -> usize {
        self.rules.len()
    }

    /// Run the reactor loop. This consumes self and runs until the broadcast
    /// channel is closed (all senders dropped) or the task is cancelled.
    pub async fn run(mut self) {
        let rules_count = self.rules.len();
        info!(rules = rules_count, "EventReactor started");
        self.counters.running.store(true, Ordering::Relaxed);

        loop {
            match self.receiver.recv().await {
                Ok(event) => {
                    self.counters
                        .events_received
                        .fetch_add(1, Ordering::Relaxed);

                    let mut matched = false;
                    for rule in &self.rules {
                        if rule.matches(&event) {
                            matched = true;
                            self.counters
                                .handlers_invoked
                                .fetch_add(1, Ordering::Relaxed);

                            let handler = Arc::clone(&rule.handler);
                            let ctx = Arc::clone(&self.context);
                            let event_clone = event.clone();
                            let rule_name = rule.name.clone();
                            let counters = Arc::clone(&self.counters);

                            // Spawn each handler as a separate task for isolation
                            tokio::spawn(async move {
                                let result = tokio::spawn(handler(event_clone, ctx)).await;
                                match result {
                                    Ok(()) => {
                                        debug!(rule = %rule_name, "Reaction handler completed");
                                    }
                                    Err(e) => {
                                        counters.handler_errors.fetch_add(1, Ordering::Relaxed);
                                        error!(
                                            rule = %rule_name,
                                            error = %e,
                                            "Reaction handler panicked"
                                        );
                                    }
                                }
                            });
                        }
                    }

                    if matched {
                        self.counters
                            .events_matched
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!(
                        skipped = n,
                        "EventReactor lagged, skipped events"
                    );
                }
                Err(broadcast::error::RecvError::Closed) => {
                    info!("EventReactor channel closed, shutting down");
                    break;
                }
            }
        }

        self.counters.running.store(false, Ordering::Relaxed);
        info!("EventReactor stopped");
    }
}

/// Builder for constructing an EventReactor with registered rules.
///
/// # Example
///
/// ```ignore
/// let reactor = ReactorBuilder::new(event_bus.subscribe(), state.clone())
///     .on(
///         "project-synced",
///         Some(EntityType::Project),
///         Some(CrudAction::Synced),
///         Arc::new(|event, ctx| Box::pin(async move {
///             // handle project sync...
///         })),
///     )
///     .build();
///
/// tokio::spawn(reactor.run());
/// ```
pub struct ReactorBuilder {
    rules: Vec<ReactionRule>,
    receiver: broadcast::Receiver<CrudEvent>,
    context: Arc<dyn std::any::Any + Send + Sync>,
}

impl ReactorBuilder {
    /// Create a new builder with a broadcast receiver and shared context.
    pub fn new(
        receiver: broadcast::Receiver<CrudEvent>,
        context: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Self {
        Self {
            rules: Vec::new(),
            receiver,
            context,
        }
    }

    /// Register a reaction rule.
    ///
    /// - `name`: human-readable name for logging
    /// - `entity_type`: `Some(X)` to filter, `None` for any
    /// - `action`: `Some(X)` to filter, `None` for any
    /// - `handler`: async handler function
    pub fn on(
        mut self,
        name: impl Into<String>,
        entity_type: Option<EntityType>,
        action: Option<CrudAction>,
        handler: ReactionFn,
    ) -> Self {
        self.rules.push(ReactionRule {
            name: name.into(),
            entity_type,
            action,
            handler,
        });
        self
    }

    /// Build the reactor. The counters are returned separately for the status endpoint.
    pub fn build(self) -> (EventReactor, Arc<ReactorCounters>) {
        let counters = Arc::new(ReactorCounters::default());
        let reactor = EventReactor {
            rules: self.rules,
            receiver: self.receiver,
            context: self.context,
            counters: Arc::clone(&counters),
        };
        (reactor, counters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventEmitter;
    use std::sync::atomic::AtomicUsize;
    use tokio::time::{sleep, Duration};

    fn make_event(entity_type: EntityType, action: CrudAction) -> CrudEvent {
        CrudEvent::new(entity_type, action, "test-id")
    }

    #[test]
    fn test_rule_matches_exact() {
        let rule = ReactionRule {
            name: "test".into(),
            entity_type: Some(EntityType::Project),
            action: Some(CrudAction::Synced),
            handler: Arc::new(|_, _| Box::pin(async {})),
        };

        assert!(rule.matches(&make_event(EntityType::Project, CrudAction::Synced)));
        assert!(!rule.matches(&make_event(EntityType::Project, CrudAction::Created)));
        assert!(!rule.matches(&make_event(EntityType::Task, CrudAction::Synced)));
    }

    #[test]
    fn test_rule_matches_wildcard_action() {
        let rule = ReactionRule {
            name: "test".into(),
            entity_type: Some(EntityType::Task),
            action: None, // match any action
            handler: Arc::new(|_, _| Box::pin(async {})),
        };

        assert!(rule.matches(&make_event(EntityType::Task, CrudAction::Created)));
        assert!(rule.matches(&make_event(EntityType::Task, CrudAction::Deleted)));
        assert!(!rule.matches(&make_event(EntityType::Plan, CrudAction::Created)));
    }

    #[test]
    fn test_rule_matches_wildcard_entity() {
        let rule = ReactionRule {
            name: "test".into(),
            entity_type: None, // match any entity
            action: Some(CrudAction::StatusChanged),
            handler: Arc::new(|_, _| Box::pin(async {})),
        };

        assert!(rule.matches(&make_event(EntityType::Task, CrudAction::StatusChanged)));
        assert!(rule.matches(&make_event(EntityType::Plan, CrudAction::StatusChanged)));
        assert!(!rule.matches(&make_event(EntityType::Plan, CrudAction::Updated)));
    }

    #[test]
    fn test_rule_matches_wildcard_both() {
        let rule = ReactionRule {
            name: "catch-all".into(),
            entity_type: None,
            action: None,
            handler: Arc::new(|_, _| Box::pin(async {})),
        };

        assert!(rule.matches(&make_event(EntityType::Task, CrudAction::Created)));
        assert!(rule.matches(&make_event(EntityType::Note, CrudAction::Deleted)));
    }

    #[tokio::test]
    async fn test_reactor_receives_and_dispatches() {
        let bus = crate::events::EventBus::default();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let (reactor, counters) = ReactorBuilder::new(
            bus.subscribe(),
            Arc::new(()) as Arc<dyn std::any::Any + Send + Sync>,
        )
        .on(
            "count-tasks",
            Some(EntityType::Task),
            Some(CrudAction::Created),
            Arc::new(move |_event, _ctx| {
                let c = Arc::clone(&counter_clone);
                Box::pin(async move {
                    c.fetch_add(1, Ordering::Relaxed);
                })
            }),
        )
        .build();

        let handle = tokio::spawn(reactor.run());

        // Emit events
        bus.emit_created(EntityType::Task, "t1", serde_json::Value::Null, None);
        bus.emit_created(EntityType::Task, "t2", serde_json::Value::Null, None);
        bus.emit_created(EntityType::Plan, "p1", serde_json::Value::Null, None); // should not match

        sleep(Duration::from_millis(50)).await;

        assert_eq!(counter.load(Ordering::Relaxed), 2);
        assert_eq!(counters.events_received.load(Ordering::Relaxed), 3);
        assert_eq!(counters.events_matched.load(Ordering::Relaxed), 2);
        assert_eq!(counters.handlers_invoked.load(Ordering::Relaxed), 2);

        handle.abort();
    }

    #[tokio::test]
    async fn test_reactor_survives_handler_panic() {
        let bus = crate::events::EventBus::default();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let (reactor, counters) = ReactorBuilder::new(
            bus.subscribe(),
            Arc::new(()) as Arc<dyn std::any::Any + Send + Sync>,
        )
        .on(
            "panic-rule",
            Some(EntityType::Plan),
            Some(CrudAction::Created),
            Arc::new(|_event, _ctx| {
                Box::pin(async move {
                    panic!("intentional test panic");
                })
            }),
        )
        .on(
            "safe-rule",
            Some(EntityType::Task),
            Some(CrudAction::Created),
            Arc::new(move |_event, _ctx| {
                let c = Arc::clone(&counter_clone);
                Box::pin(async move {
                    c.fetch_add(1, Ordering::Relaxed);
                })
            }),
        )
        .build();

        let handle = tokio::spawn(reactor.run());

        // First: trigger the panic handler
        bus.emit_created(EntityType::Plan, "p1", serde_json::Value::Null, None);
        sleep(Duration::from_millis(50)).await;

        // Then: trigger the safe handler — reactor should still be alive
        bus.emit_created(EntityType::Task, "t1", serde_json::Value::Null, None);
        sleep(Duration::from_millis(50)).await;

        assert_eq!(counter.load(Ordering::Relaxed), 1);
        assert!(counters.handler_errors.load(Ordering::Relaxed) >= 1);

        handle.abort();
    }
}
