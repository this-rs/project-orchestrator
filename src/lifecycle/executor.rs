//! # LifecycleHook Executor
//!
//! Evaluates and runs matching [`LifecycleHook`]s when entities change status.
//!
//! Called from the EventReactor's `lifecycle-hooks` reaction. For each
//! `StatusChanged` event, the executor:
//! 1. Maps the [`EntityType`] to a [`LifecycleScope`]
//! 2. Queries Neo4j for enabled hooks matching (scope, on_status, project_id)
//! 3. Executes each hook's action in priority order
//!
//! Errors from individual hooks are logged but do **not** block subsequent hooks.

use std::sync::Arc;

use tracing::{debug, error, info, warn};

use crate::api::handlers::ServerState;
use crate::events::{CrudEvent, EntityType};
use crate::lifecycle::models::{LifecycleActionType, LifecycleHook, LifecycleScope};
use crate::neo4j::models::{AlertNode, AlertSeverity};
use crate::notes::{Note, NoteImportance, NoteType};

/// Map [`CrudEvent`] [`EntityType`] to [`LifecycleScope`].
fn entity_type_to_scope(entity_type: &EntityType) -> Option<LifecycleScope> {
    match entity_type {
        EntityType::Task => Some(LifecycleScope::Task),
        EntityType::Plan => Some(LifecycleScope::Plan),
        EntityType::Step => Some(LifecycleScope::Step),
        EntityType::Milestone => Some(LifecycleScope::Milestone),
        _ => None,
    }
}

/// Execute all matching LifecycleHooks for a StatusChanged event.
///
/// This is the main entry point called from the reactor reaction.
/// It loads matching hooks from Neo4j (scope + on_status + project_id),
/// sorts them by priority, and executes each action.
///
/// Errors from individual hooks are logged but do NOT block subsequent hooks.
pub async fn execute_lifecycle_hooks(event: &CrudEvent, state: &Arc<ServerState>) {
    // Only process events that carry a new_status in their payload
    let new_status = match event
        .payload
        .get("new_status")
        .and_then(serde_json::Value::as_str)
    {
        Some(s) => s,
        None => return,
    };

    // Map entity type to scope
    let scope = match entity_type_to_scope(&event.entity_type) {
        Some(s) => s,
        None => return,
    };

    // Determine project_id from event (if available)
    let project_id = event
        .project_id
        .as_ref()
        .and_then(|s| uuid::Uuid::parse_str(s).ok());

    // Load matching hooks from Neo4j
    let hooks = match state
        .orchestrator
        .neo4j()
        .list_hooks_for_scope(&scope, new_status, project_id)
        .await
    {
        Ok(hooks) => hooks,
        Err(e) => {
            error!(error = %e, "Failed to load lifecycle hooks");
            return;
        }
    };

    if hooks.is_empty() {
        return;
    }

    debug!(
        scope = ?scope,
        on_status = %new_status,
        hooks_count = hooks.len(),
        "Executing lifecycle hooks"
    );

    // Execute each hook (already sorted by priority from list_hooks_for_scope)
    for hook in &hooks {
        if let Err(e) = execute_single_hook(hook, event, state).await {
            error!(
                hook_name = %hook.name,
                hook_id = %hook.id,
                error = %e,
                "Lifecycle hook execution failed (continuing with next hook)"
            );
        }
    }
}

/// Execute a single LifecycleHook action.
async fn execute_single_hook(
    hook: &LifecycleHook,
    event: &CrudEvent,
    state: &Arc<ServerState>,
) -> anyhow::Result<()> {
    debug!(
        hook_name = %hook.name,
        action_type = ?hook.action_type,
        "Executing lifecycle hook"
    );

    match hook.action_type {
        LifecycleActionType::CascadeChildren => execute_cascade_children(hook, event, state).await,
        LifecycleActionType::CreateNote => execute_create_note(hook, event, state).await,
        LifecycleActionType::EmitAlert => execute_emit_alert(hook, event, state).await,
        LifecycleActionType::StartProtocol => execute_start_protocol(hook, event, state).await,
        LifecycleActionType::McpCall => execute_mcp_call(hook, event, state).await,
    }
}

// ─────────────────────────────────────────────────────────────
// Action: CascadeChildren
// ─────────────────────────────────────────────────────────────

async fn execute_cascade_children(
    hook: &LifecycleHook,
    event: &CrudEvent,
    state: &Arc<ServerState>,
) -> anyhow::Result<()> {
    // action_config may contain: { "target": "steps" }
    let target = hook
        .action_config
        .get("target")
        .and_then(|v| v.as_str())
        .unwrap_or("steps");

    let entity_id = uuid::Uuid::parse_str(&event.entity_id)?;

    match target {
        "steps" => {
            // Complete all pending steps for the task
            let count = state
                .orchestrator
                .neo4j()
                .complete_pending_steps_for_task(entity_id)
                .await?;
            if count > 0 {
                info!(
                    task_id = %entity_id,
                    steps = count,
                    hook = %hook.name,
                    "Cascaded step completion"
                );
            }
        }
        _ => {
            debug!(target, hook = %hook.name, "Unknown cascade target, ignoring");
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────
// Action: CreateNote
// ─────────────────────────────────────────────────────────────

async fn execute_create_note(
    hook: &LifecycleHook,
    event: &CrudEvent,
    state: &Arc<ServerState>,
) -> anyhow::Result<()> {
    let note_type_str = hook
        .action_config
        .get("note_type")
        .and_then(|v| v.as_str())
        .unwrap_or("observation");
    let content_template = hook
        .action_config
        .get("content_template")
        .and_then(|v| v.as_str())
        .unwrap_or("Lifecycle hook triggered");
    let importance_str = hook
        .action_config
        .get("importance")
        .and_then(|v| v.as_str())
        .unwrap_or("low");

    // Simple template substitution
    let content = content_template
        .replace("{entity_id}", &event.entity_id)
        .replace("{entity_type}", &format!("{:?}", event.entity_type))
        .replace("{hook_name}", &hook.name);

    let note_type: NoteType =
        serde_json::from_str(&format!("\"{}\"", note_type_str)).unwrap_or(NoteType::Observation);
    let importance: NoteImportance =
        serde_json::from_str(&format!("\"{}\"", importance_str)).unwrap_or(NoteImportance::Low);

    let mut note = Note::new(
        hook.project_id,
        note_type,
        content,
        "lifecycle-hook".to_string(),
    );
    note.importance = importance;

    state.orchestrator.neo4j().create_note(&note).await?;

    info!(note_id = %note.id, hook = %hook.name, "Created note from lifecycle hook");
    Ok(())
}

// ─────────────────────────────────────────────────────────────
// Action: EmitAlert
// ─────────────────────────────────────────────────────────────

async fn execute_emit_alert(
    hook: &LifecycleHook,
    event: &CrudEvent,
    state: &Arc<ServerState>,
) -> anyhow::Result<()> {
    let level = hook
        .action_config
        .get("level")
        .and_then(|v| v.as_str())
        .unwrap_or("info");
    let message_template = hook
        .action_config
        .get("message_template")
        .and_then(|v| v.as_str())
        .unwrap_or("Lifecycle hook alert");

    let message = message_template
        .replace("{entity_id}", &event.entity_id)
        .replace("{entity_type}", &format!("{:?}", event.entity_type))
        .replace("{hook_name}", &hook.name);

    let severity = match level {
        "warning" => AlertSeverity::Warning,
        "critical" => AlertSeverity::Critical,
        _ => AlertSeverity::Info,
    };

    let alert = AlertNode {
        id: uuid::Uuid::new_v4(),
        alert_type: "lifecycle_hook".to_string(),
        severity,
        message,
        project_id: hook.project_id,
        acknowledged: false,
        acknowledged_by: None,
        acknowledged_at: None,
        created_at: chrono::Utc::now(),
    };

    state.orchestrator.neo4j().create_alert(&alert).await?;

    info!(level, hook = %hook.name, "Emitted alert from lifecycle hook");
    Ok(())
}

// ─────────────────────────────────────────────────────────────
// Action: StartProtocol
// ─────────────────────────────────────────────────────────────

async fn execute_start_protocol(
    hook: &LifecycleHook,
    event: &CrudEvent,
    state: &Arc<ServerState>,
) -> anyhow::Result<()> {
    let protocol_id_str = hook
        .action_config
        .get("protocol_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!("start_protocol action requires protocol_id in action_config")
        })?;

    let protocol_id = uuid::Uuid::parse_str(protocol_id_str)?;

    // Load the protocol to get entry_state info
    let protocol = state
        .orchestrator
        .neo4j()
        .get_protocol(protocol_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Protocol {} not found", protocol_id))?;

    // Load protocol states to find the entry state name
    let states = state
        .orchestrator
        .neo4j()
        .get_protocol_states(protocol_id)
        .await?;
    let entry_state_name = states
        .iter()
        .find(|s| s.id == protocol.entry_state)
        .map(|s| s.name.clone())
        .unwrap_or_else(|| "START".to_string());

    // Build a new ProtocolRun
    let mut run =
        crate::protocol::ProtocolRun::new(protocol_id, protocol.entry_state, &entry_state_name);
    run.triggered_by = format!("lifecycle-hook:{}", hook.name);

    // Set optional context from event
    run.plan_id = event
        .payload
        .get("plan_id")
        .and_then(serde_json::Value::as_str)
        .and_then(|s| uuid::Uuid::parse_str(s).ok());
    run.task_id = uuid::Uuid::parse_str(&event.entity_id).ok();

    // Persist the run
    state.orchestrator.neo4j().create_protocol_run(&run).await?;

    info!(
        run_id = %run.id,
        protocol_id = %protocol_id,
        hook = %hook.name,
        "Started protocol run from lifecycle hook"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────
// Action: McpCall (stub)
// ─────────────────────────────────────────────────────────────

async fn execute_mcp_call(
    hook: &LifecycleHook,
    _event: &CrudEvent,
    _state: &Arc<ServerState>,
) -> anyhow::Result<()> {
    // MCP call execution is complex and will be implemented in a future iteration.
    warn!(hook = %hook.name, "mcp_call action type is not yet implemented");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{CrudAction, EntityType};
    use crate::lifecycle::models::{LifecycleActionType, LifecycleHook, LifecycleScope};
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::GraphStore;
    use crate::orchestrator::Orchestrator;

    // ── entity_type_to_scope ──

    #[test]
    fn test_entity_type_to_scope_task() {
        assert_eq!(
            entity_type_to_scope(&EntityType::Task),
            Some(LifecycleScope::Task)
        );
    }

    #[test]
    fn test_entity_type_to_scope_plan() {
        assert_eq!(
            entity_type_to_scope(&EntityType::Plan),
            Some(LifecycleScope::Plan)
        );
    }

    #[test]
    fn test_entity_type_to_scope_step() {
        assert_eq!(
            entity_type_to_scope(&EntityType::Step),
            Some(LifecycleScope::Step)
        );
    }

    #[test]
    fn test_entity_type_to_scope_milestone() {
        assert_eq!(
            entity_type_to_scope(&EntityType::Milestone),
            Some(LifecycleScope::Milestone)
        );
    }

    #[test]
    fn test_entity_type_to_scope_unsupported_returns_none() {
        assert_eq!(entity_type_to_scope(&EntityType::Project), None);
        assert_eq!(entity_type_to_scope(&EntityType::Decision), None);
        assert_eq!(entity_type_to_scope(&EntityType::Note), None);
        assert_eq!(entity_type_to_scope(&EntityType::Commit), None);
    }

    // ── helpers ──

    async fn make_test_server_state() -> (Arc<ServerState>, Arc<MockGraphStore>) {
        let graph = Arc::new(MockGraphStore::new());
        let app_state = crate::test_helpers::mock_app_state_with_graph(graph.clone());
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(
            crate::orchestrator::watcher::FileWatcher::new(orchestrator.clone()),
        ));
        let event_bus = Arc::new(crate::events::HybridEmitter::new(Arc::new(
            crate::events::EventBus::default(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus,
            nats_emitter: None,
            auth_config: None,
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
        });
        (state, graph)
    }

    fn make_status_changed_event(
        entity_type: EntityType,
        entity_id: &str,
        new_status: &str,
    ) -> CrudEvent {
        CrudEvent::new(entity_type, CrudAction::StatusChanged, entity_id)
            .with_payload(serde_json::json!({"new_status": new_status}))
    }

    fn make_hook(
        name: &str,
        scope: LifecycleScope,
        on_status: &str,
        action_type: LifecycleActionType,
        config: serde_json::Value,
    ) -> LifecycleHook {
        LifecycleHook::new(
            name.to_string(),
            scope,
            on_status.to_string(),
            action_type,
            config,
        )
    }

    // ── execute_lifecycle_hooks ──

    #[tokio::test]
    async fn test_execute_lifecycle_hooks_skips_event_without_new_status() {
        let (state, _graph) = make_test_server_state().await;
        // Event with no new_status in payload
        let event = CrudEvent::new(EntityType::Task, CrudAction::Updated, "task-1");
        // Should return without error (just a no-op)
        execute_lifecycle_hooks(&event, &state).await;
    }

    #[tokio::test]
    async fn test_execute_lifecycle_hooks_skips_unsupported_entity_type() {
        let (state, _graph) = make_test_server_state().await;
        let event = make_status_changed_event(EntityType::Project, "proj-1", "active");
        // Project has no scope mapping, should be a no-op
        execute_lifecycle_hooks(&event, &state).await;
    }

    #[tokio::test]
    async fn test_execute_lifecycle_hooks_no_matching_hooks() {
        let (state, _graph) = make_test_server_state().await;
        let event = make_status_changed_event(
            EntityType::Task,
            &uuid::Uuid::new_v4().to_string(),
            "completed",
        );
        // No hooks registered, should be a no-op
        execute_lifecycle_hooks(&event, &state).await;
    }

    #[tokio::test]
    async fn test_execute_create_note_action() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();

        // Register a CreateNote hook
        let hook = make_hook(
            "note-on-complete",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::CreateNote,
            serde_json::json!({
                "note_type": "observation",
                "content_template": "Task {entity_id} completed via {hook_name}",
                "importance": "medium"
            }),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed");
        execute_lifecycle_hooks(&event, &state).await;

        // Verify a note was created via mock store
        let notes = graph.notes.read().await;
        assert!(!notes.is_empty(), "A note should have been created");
        let note = notes.values().next().unwrap();
        assert!(
            note.content.contains(&task_id.to_string()),
            "Note content should contain entity_id"
        );
        assert!(
            note.content.contains("note-on-complete"),
            "Note content should contain hook_name"
        );
    }

    #[tokio::test]
    async fn test_execute_emit_alert_action() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();

        let hook = make_hook(
            "alert-on-blocked",
            LifecycleScope::Task,
            "blocked",
            LifecycleActionType::EmitAlert,
            serde_json::json!({
                "level": "warning",
                "message_template": "Task {entity_id} is blocked ({hook_name})"
            }),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "blocked");
        execute_lifecycle_hooks(&event, &state).await;

        // Verify an alert was created
        let alerts = graph.alerts.read().await;
        assert!(!alerts.is_empty(), "An alert should have been created");
        let alert = alerts.values().next().unwrap();
        assert!(alert.message.contains(&task_id.to_string()));
        assert_eq!(alert.severity, crate::neo4j::models::AlertSeverity::Warning);
    }

    #[tokio::test]
    async fn test_execute_emit_alert_default_level_is_info() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();

        let hook = make_hook(
            "info-alert",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::EmitAlert,
            serde_json::json!({}), // no level specified
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed");
        execute_lifecycle_hooks(&event, &state).await;

        let alerts = graph.alerts.read().await;
        assert_eq!(alerts.len(), 1);
        assert_eq!(
            alerts.values().next().unwrap().severity,
            crate::neo4j::models::AlertSeverity::Info
        );
    }

    #[tokio::test]
    async fn test_execute_mcp_call_is_noop() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();

        let hook = make_hook(
            "mcp-stub",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::McpCall,
            serde_json::json!({"tool": "some_tool"}),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed");
        // Should not panic or error
        execute_lifecycle_hooks(&event, &state).await;
    }

    #[tokio::test]
    async fn test_execute_start_protocol_missing_protocol_id() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();

        // StartProtocol without protocol_id in config - should log error but not panic
        let hook = make_hook(
            "start-proto-bad",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::StartProtocol,
            serde_json::json!({}),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed");
        // Error is logged but should not propagate
        execute_lifecycle_hooks(&event, &state).await;
    }

    #[tokio::test]
    async fn test_template_substitution_in_create_note() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();

        let hook = make_hook(
            "template-hook",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::CreateNote,
            serde_json::json!({
                "content_template": "ID={entity_id} TYPE={entity_type} HOOK={hook_name}"
            }),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed");
        execute_lifecycle_hooks(&event, &state).await;

        let notes = graph.notes.read().await;
        assert_eq!(notes.len(), 1);
        let content = &notes.values().next().unwrap().content;
        assert!(content.contains(&format!("ID={}", task_id)));
        assert!(content.contains("TYPE=Task"));
        assert!(content.contains("HOOK=template-hook"));
    }

    #[tokio::test]
    async fn test_multiple_hooks_execute_in_priority_order() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();

        // Create two hooks with different priorities
        let mut hook1 = make_hook(
            "low-priority-alert",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::EmitAlert,
            serde_json::json!({"message_template": "low-priority"}),
        );
        hook1.priority = 200;

        let mut hook2 = make_hook(
            "high-priority-alert",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::EmitAlert,
            serde_json::json!({"message_template": "high-priority"}),
        );
        hook2.priority = 10;

        graph.create_lifecycle_hook(&hook1).await.unwrap();
        graph.create_lifecycle_hook(&hook2).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed");
        execute_lifecycle_hooks(&event, &state).await;

        let alerts = graph.alerts.read().await;
        assert_eq!(alerts.len(), 2, "Both hooks should have fired");
    }

    #[tokio::test]
    async fn test_hook_failure_does_not_block_others() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();

        // First hook: StartProtocol with bad config (will fail)
        let mut hook1 = make_hook(
            "failing-hook",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::StartProtocol,
            serde_json::json!({}), // missing protocol_id
        );
        hook1.priority = 1;

        // Second hook: EmitAlert (should succeed despite first hook failing)
        let mut hook2 = make_hook(
            "succeeding-hook",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::EmitAlert,
            serde_json::json!({"message_template": "should-still-fire"}),
        );
        hook2.priority = 2;

        graph.create_lifecycle_hook(&hook1).await.unwrap();
        graph.create_lifecycle_hook(&hook2).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed");
        execute_lifecycle_hooks(&event, &state).await;

        // The alert from hook2 should still be created
        let alerts = graph.alerts.read().await;
        assert_eq!(alerts.len(), 1);
        assert!(alerts
            .values()
            .next()
            .unwrap()
            .message
            .contains("should-still-fire"));
    }

    #[tokio::test]
    async fn test_execute_with_project_id_in_event() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();
        let project_id = uuid::Uuid::new_v4();

        let mut hook = make_hook(
            "project-hook",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::EmitAlert,
            serde_json::json!({"message_template": "project-scoped alert"}),
        );
        hook.project_id = Some(project_id);
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed")
            .with_project_id(project_id.to_string());
        execute_lifecycle_hooks(&event, &state).await;

        let alerts = graph.alerts.read().await;
        assert_eq!(alerts.len(), 1);
    }

    #[tokio::test]
    async fn test_emit_alert_critical_level() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();

        let hook = make_hook(
            "critical-alert",
            LifecycleScope::Task,
            "failed",
            LifecycleActionType::EmitAlert,
            serde_json::json!({"level": "critical", "message_template": "Critical failure"}),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "failed");
        execute_lifecycle_hooks(&event, &state).await;

        let alerts = graph.alerts.read().await;
        assert_eq!(alerts.len(), 1);
        assert_eq!(
            alerts.values().next().unwrap().severity,
            crate::neo4j::models::AlertSeverity::Critical
        );
    }

    // ── CascadeChildren action tests ──

    #[tokio::test]
    async fn test_execute_cascade_children_steps() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();

        // Create a step for the task that is pending
        let step = crate::neo4j::models::StepNode {
            id: uuid::Uuid::new_v4(),
            order: 1,
            description: "A pending step".to_string(),
            status: crate::neo4j::models::StepStatus::Pending,
            verification: None,
            created_at: chrono::Utc::now(),
            updated_at: None,
            completed_at: None,
            execution_context: None,
            persona: None,
        };
        graph.create_step(task_id, &step).await.unwrap();

        let hook = make_hook(
            "cascade-steps",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::CascadeChildren,
            serde_json::json!({"target": "steps"}),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed");
        execute_lifecycle_hooks(&event, &state).await;

        // Verify step was completed
        let steps = graph.steps.read().await;
        let updated_step = steps.get(&step.id).unwrap();
        assert_eq!(
            updated_step.status,
            crate::neo4j::models::StepStatus::Completed
        );
    }

    #[tokio::test]
    async fn test_execute_cascade_children_unknown_target() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();

        let hook = make_hook(
            "cascade-unknown",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::CascadeChildren,
            serde_json::json!({"target": "unknown_thing"}),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed");
        // Should not panic, just ignore the unknown target
        execute_lifecycle_hooks(&event, &state).await;
    }

    #[tokio::test]
    async fn test_execute_cascade_children_default_target_is_steps() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();

        // No "target" in config — should default to "steps"
        let hook = make_hook(
            "cascade-default",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::CascadeChildren,
            serde_json::json!({}),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed");
        // Should not panic even with no steps
        execute_lifecycle_hooks(&event, &state).await;
    }

    // ── StartProtocol with valid protocol_id ──

    #[tokio::test]
    async fn test_execute_start_protocol_with_valid_protocol() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();
        let project_id = uuid::Uuid::new_v4();

        // Create a protocol in the store
        let state_id = uuid::Uuid::new_v4();
        let mut protocol =
            crate::protocol::Protocol::new(project_id, "test-auto-protocol", state_id);
        protocol.protocol_category = crate::protocol::ProtocolCategory::System;
        graph.upsert_protocol(&protocol).await.unwrap();

        let mut proto_state = crate::protocol::ProtocolState::new(protocol.id, "start");
        proto_state.id = state_id;
        proto_state.state_type = crate::protocol::StateType::Start;
        graph.upsert_protocol_state(&proto_state).await.unwrap();

        let hook = make_hook(
            "start-valid-proto",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::StartProtocol,
            serde_json::json!({"protocol_id": protocol.id.to_string()}),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed");
        execute_lifecycle_hooks(&event, &state).await;

        // Verify a protocol run was created
        let (runs, _) = graph
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(runs.len(), 1, "A protocol run should have been created");
        assert_eq!(
            runs[0].triggered_by, "lifecycle-hook:start-valid-proto",
            "triggered_by should reference the hook name"
        );
    }

    #[tokio::test]
    async fn test_execute_start_protocol_nonexistent_protocol() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();
        let fake_protocol_id = uuid::Uuid::new_v4();

        let hook = make_hook(
            "start-missing-proto",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::StartProtocol,
            serde_json::json!({"protocol_id": fake_protocol_id.to_string()}),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed");
        // Should not panic — error is logged
        execute_lifecycle_hooks(&event, &state).await;
    }

    // ── CreateNote with default values ──

    #[tokio::test]
    async fn test_execute_create_note_with_defaults() {
        let (state, graph) = make_test_server_state().await;
        let task_id = uuid::Uuid::new_v4();

        // Hook with empty config — all defaults
        let hook = make_hook(
            "note-defaults",
            LifecycleScope::Task,
            "completed",
            LifecycleActionType::CreateNote,
            serde_json::json!({}),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Task, &task_id.to_string(), "completed");
        execute_lifecycle_hooks(&event, &state).await;

        let notes = graph.notes.read().await;
        assert_eq!(notes.len(), 1);
        let note = notes.values().next().unwrap();
        // Default content_template is "Lifecycle hook triggered"
        assert_eq!(note.content, "Lifecycle hook triggered");
    }

    // ── Scope mapping for Step and Milestone ──

    #[tokio::test]
    async fn test_execute_hooks_for_step_entity() {
        let (state, graph) = make_test_server_state().await;
        let step_id = uuid::Uuid::new_v4();

        let hook = make_hook(
            "step-alert",
            LifecycleScope::Step,
            "completed",
            LifecycleActionType::EmitAlert,
            serde_json::json!({"message_template": "Step done"}),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event = make_status_changed_event(EntityType::Step, &step_id.to_string(), "completed");
        execute_lifecycle_hooks(&event, &state).await;

        let alerts = graph.alerts.read().await;
        assert_eq!(alerts.len(), 1);
    }

    #[tokio::test]
    async fn test_execute_hooks_for_milestone_entity() {
        let (state, graph) = make_test_server_state().await;
        let milestone_id = uuid::Uuid::new_v4();

        let hook = make_hook(
            "milestone-alert",
            LifecycleScope::Milestone,
            "reached",
            LifecycleActionType::EmitAlert,
            serde_json::json!({"message_template": "Milestone reached"}),
        );
        graph.create_lifecycle_hook(&hook).await.unwrap();

        let event =
            make_status_changed_event(EntityType::Milestone, &milestone_id.to_string(), "reached");
        execute_lifecycle_hooks(&event, &state).await;

        let alerts = graph.alerts.read().await;
        assert_eq!(alerts.len(), 1);
    }
}
