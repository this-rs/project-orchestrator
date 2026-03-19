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
