//! API route definitions

use super::chat_handlers;
use super::code_handlers;
use super::handlers::{self, OrchestratorState};
use super::note_handlers;
use super::project_handlers;
use super::workspace_handlers;
use super::ws_handlers;
use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

/// Create the API router
pub fn create_router(state: OrchestratorState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // Health check
        .route("/health", get(handlers::health))
        // ====================================================================
        // Projects (multi-project support)
        // ====================================================================
        .route(
            "/api/projects",
            get(project_handlers::list_projects).post(project_handlers::create_project),
        )
        .route(
            "/api/projects/{slug}",
            get(project_handlers::get_project)
                .patch(project_handlers::update_project)
                .delete(project_handlers::delete_project),
        )
        .route(
            "/api/projects/{slug}/sync",
            post(project_handlers::sync_project),
        )
        .route(
            "/api/projects/{slug}/plans",
            get(project_handlers::list_project_plans),
        )
        .route(
            "/api/projects/{slug}/code/search",
            get(project_handlers::search_project_code),
        )
        // Releases (by project_id)
        .route(
            "/api/projects/{project_id}/releases",
            get(handlers::list_releases).post(handlers::create_release),
        )
        // Milestones (by project_id)
        .route(
            "/api/projects/{project_id}/milestones",
            get(handlers::list_milestones).post(handlers::create_milestone),
        )
        // Roadmap (aggregated view)
        .route(
            "/api/projects/{project_id}/roadmap",
            get(handlers::get_project_roadmap),
        )
        // ====================================================================
        // Plans (global or legacy)
        .route(
            "/api/plans",
            get(handlers::list_plans).post(handlers::create_plan),
        )
        .route(
            "/api/plans/{plan_id}",
            get(handlers::get_plan)
                .patch(handlers::update_plan_status)
                .delete(handlers::delete_plan),
        )
        .route(
            "/api/plans/{plan_id}/project",
            axum::routing::put(handlers::link_plan_to_project)
                .delete(handlers::unlink_plan_from_project),
        )
        .route(
            "/api/plans/{plan_id}/next-task",
            get(handlers::get_next_task),
        )
        .route(
            "/api/plans/{plan_id}/dependency-graph",
            get(handlers::get_plan_dependency_graph),
        )
        .route(
            "/api/plans/{plan_id}/critical-path",
            get(handlers::get_plan_critical_path),
        )
        // Constraints
        .route(
            "/api/plans/{plan_id}/constraints",
            get(handlers::get_plan_constraints).post(handlers::add_constraint),
        )
        .route(
            "/api/constraints/{constraint_id}",
            get(handlers::get_constraint)
                .patch(handlers::update_constraint)
                .delete(handlers::delete_constraint),
        )
        // Tasks (global listing)
        .route("/api/tasks", get(handlers::list_all_tasks))
        // Tasks (plan-scoped)
        .route("/api/plans/{plan_id}/tasks", post(handlers::add_task))
        .route(
            "/api/tasks/{task_id}",
            get(handlers::get_task)
                .patch(handlers::update_task)
                .delete(handlers::delete_task),
        )
        // Task dependencies
        .route(
            "/api/tasks/{task_id}/dependencies",
            post(handlers::add_task_dependencies),
        )
        .route(
            "/api/tasks/{task_id}/dependencies/{dep_id}",
            axum::routing::delete(handlers::remove_task_dependency),
        )
        .route(
            "/api/tasks/{task_id}/blockers",
            get(handlers::get_task_blockers),
        )
        .route(
            "/api/tasks/{task_id}/blocking",
            get(handlers::get_tasks_blocked_by),
        )
        // Steps
        .route(
            "/api/tasks/{task_id}/steps",
            get(handlers::get_task_steps).post(handlers::add_step),
        )
        .route(
            "/api/tasks/{task_id}/steps/progress",
            get(handlers::get_step_progress),
        )
        .route(
            "/api/steps/{step_id}",
            get(handlers::get_step)
                .patch(handlers::update_step)
                .delete(handlers::delete_step),
        )
        // Context
        .route(
            "/api/plans/{plan_id}/tasks/{task_id}/context",
            get(handlers::get_task_context),
        )
        .route(
            "/api/plans/{plan_id}/tasks/{task_id}/prompt",
            get(handlers::get_task_prompt),
        )
        // Decisions
        .route(
            "/api/tasks/{task_id}/decisions",
            post(handlers::add_decision),
        )
        .route(
            "/api/decisions/{decision_id}",
            get(handlers::get_decision)
                .patch(handlers::update_decision)
                .delete(handlers::delete_decision),
        )
        .route("/api/decisions/search", get(handlers::search_decisions))
        // Sync
        .route("/api/sync", post(handlers::sync_directory))
        // Releases
        .route(
            "/api/releases/{release_id}",
            get(handlers::get_release)
                .patch(handlers::update_release)
                .delete(handlers::delete_release),
        )
        .route(
            "/api/releases/{release_id}/tasks",
            post(handlers::add_task_to_release),
        )
        .route(
            "/api/releases/{release_id}/commits",
            post(handlers::add_commit_to_release),
        )
        // Milestones
        .route(
            "/api/milestones/{milestone_id}",
            get(handlers::get_milestone)
                .patch(handlers::update_milestone)
                .delete(handlers::delete_milestone),
        )
        .route(
            "/api/milestones/{milestone_id}/tasks",
            post(handlers::add_task_to_milestone),
        )
        .route(
            "/api/milestones/{milestone_id}/progress",
            get(handlers::get_milestone_progress),
        )
        // Commits
        .route("/api/commits", post(handlers::create_commit))
        .route(
            "/api/tasks/{task_id}/commits",
            get(handlers::get_task_commits).post(handlers::link_commit_to_task),
        )
        .route(
            "/api/plans/{plan_id}/commits",
            get(handlers::get_plan_commits).post(handlers::link_commit_to_plan),
        )
        // Webhooks
        .route("/api/wake", post(handlers::wake))
        .route("/hooks/wake", post(handlers::wake)) // Alias for compatibility
        // ====================================================================
        // File Watcher (auto-sync on file changes)
        // ====================================================================
        .route(
            "/api/watch",
            get(handlers::watch_status)
                .post(handlers::start_watch)
                .delete(handlers::stop_watch),
        )
        // ====================================================================
        // Code Exploration (Graph + Search powered)
        // ====================================================================
        // Search code semantically (Meilisearch)
        .route("/api/code/search", get(code_handlers::search_code))
        // Get symbols in a file (Neo4j)
        .route(
            "/api/code/symbols/{file_path}",
            get(code_handlers::get_file_symbols),
        )
        // Find all references to a symbol
        .route("/api/code/references", get(code_handlers::find_references))
        // Get file dependencies (imports + dependents)
        .route(
            "/api/code/dependencies/{file_path}",
            get(code_handlers::get_file_dependencies),
        )
        // Get call graph for a function
        .route("/api/code/callgraph", get(code_handlers::get_call_graph))
        // Analyze impact of changes
        .route("/api/code/impact", get(code_handlers::analyze_impact))
        // Get architecture overview
        .route(
            "/api/code/architecture",
            get(code_handlers::get_architecture),
        )
        // Find similar code (POST because of body)
        .route("/api/code/similar", post(code_handlers::find_similar_code))
        // Trait implementation exploration
        .route(
            "/api/code/trait-impls",
            get(code_handlers::find_trait_implementations),
        )
        .route(
            "/api/code/type-traits",
            get(code_handlers::find_type_traits),
        )
        .route("/api/code/impl-blocks", get(code_handlers::get_impl_blocks))
        // ====================================================================
        // Knowledge Notes
        // ====================================================================
        // Notes CRUD
        .route(
            "/api/notes",
            get(note_handlers::list_notes).post(note_handlers::create_note),
        )
        .route(
            "/api/notes/{note_id}",
            get(note_handlers::get_note)
                .patch(note_handlers::update_note)
                .delete(note_handlers::delete_note),
        )
        // Notes search
        .route("/api/notes/search", get(note_handlers::search_notes))
        // Notes needing review
        .route(
            "/api/notes/needs-review",
            get(note_handlers::get_notes_needing_review),
        )
        // Update staleness scores
        .route(
            "/api/notes/update-staleness",
            post(note_handlers::update_staleness_scores),
        )
        // Notes for a project
        .route(
            "/api/projects/{project_id}/notes",
            get(note_handlers::list_project_notes),
        )
        // Note lifecycle operations
        .route(
            "/api/notes/{note_id}/confirm",
            post(note_handlers::confirm_note),
        )
        .route(
            "/api/notes/{note_id}/invalidate",
            post(note_handlers::invalidate_note),
        )
        .route(
            "/api/notes/{note_id}/supersede",
            post(note_handlers::supersede_note),
        )
        // Note linking
        .route(
            "/api/notes/{note_id}/links",
            post(note_handlers::link_note_to_entity),
        )
        .route(
            "/api/notes/{note_id}/links/{entity_type}/{entity_id}",
            axum::routing::delete(note_handlers::unlink_note_from_entity),
        )
        // Context notes (direct + propagated)
        .route("/api/notes/context", get(note_handlers::get_context_notes))
        .route(
            "/api/notes/propagated",
            get(note_handlers::get_propagated_notes),
        )
        // Entity notes (direct only)
        .route(
            "/api/entities/{entity_type}/{entity_id}/notes",
            get(note_handlers::get_entity_notes),
        )
        // ====================================================================
        // Meilisearch Maintenance
        // ====================================================================
        .route(
            "/api/meilisearch/stats",
            get(handlers::get_meilisearch_stats),
        )
        .route(
            "/api/meilisearch/orphans",
            axum::routing::delete(handlers::delete_meilisearch_orphans),
        )
        // ====================================================================
        // Workspaces
        // ====================================================================
        .route(
            "/api/workspaces",
            get(workspace_handlers::list_workspaces).post(workspace_handlers::create_workspace),
        )
        .route(
            "/api/workspaces/{slug}",
            get(workspace_handlers::get_workspace)
                .patch(workspace_handlers::update_workspace)
                .delete(workspace_handlers::delete_workspace),
        )
        .route(
            "/api/workspaces/{slug}/overview",
            get(workspace_handlers::get_workspace_overview),
        )
        .route(
            "/api/workspaces/{slug}/projects",
            get(workspace_handlers::list_workspace_projects)
                .post(workspace_handlers::add_project_to_workspace),
        )
        .route(
            "/api/workspaces/{slug}/projects/{project_id}",
            axum::routing::delete(workspace_handlers::remove_project_from_workspace),
        )
        // Workspace Milestones
        .route(
            "/api/workspaces/{slug}/milestones",
            get(workspace_handlers::list_workspace_milestones)
                .post(workspace_handlers::create_workspace_milestone),
        )
        .route(
            "/api/workspace-milestones",
            get(workspace_handlers::list_all_workspace_milestones),
        )
        .route(
            "/api/workspace-milestones/{id}",
            get(workspace_handlers::get_workspace_milestone)
                .patch(workspace_handlers::update_workspace_milestone)
                .delete(workspace_handlers::delete_workspace_milestone),
        )
        .route(
            "/api/workspace-milestones/{id}/tasks",
            post(workspace_handlers::add_task_to_workspace_milestone),
        )
        .route(
            "/api/workspace-milestones/{id}/progress",
            get(workspace_handlers::get_workspace_milestone_progress),
        )
        // Resources
        .route(
            "/api/workspaces/{slug}/resources",
            get(workspace_handlers::list_resources).post(workspace_handlers::create_resource),
        )
        .route(
            "/api/resources/{id}",
            get(workspace_handlers::get_resource)
                .patch(workspace_handlers::update_resource)
                .delete(workspace_handlers::delete_resource),
        )
        .route(
            "/api/resources/{id}/projects",
            post(workspace_handlers::link_resource_to_project),
        )
        // Components
        .route(
            "/api/workspaces/{slug}/components",
            get(workspace_handlers::list_components).post(workspace_handlers::create_component),
        )
        .route(
            "/api/components/{id}",
            get(workspace_handlers::get_component)
                .patch(workspace_handlers::update_component)
                .delete(workspace_handlers::delete_component),
        )
        .route(
            "/api/components/{id}/dependencies",
            post(workspace_handlers::add_component_dependency),
        )
        .route(
            "/api/components/{id}/dependencies/{dep_id}",
            axum::routing::delete(workspace_handlers::remove_component_dependency),
        )
        .route(
            "/api/components/{id}/project",
            axum::routing::put(workspace_handlers::map_component_to_project),
        )
        .route(
            "/api/workspaces/{slug}/topology",
            get(workspace_handlers::get_workspace_topology),
        )
        // ====================================================================
        // WebSocket CRUD Event Notifications
        // ====================================================================
        .route("/ws/events", get(ws_handlers::ws_events))
        // ====================================================================
        // Chat (SSE streaming + session management)
        // ====================================================================
        .route(
            "/api/chat/sessions",
            get(chat_handlers::list_sessions).post(chat_handlers::create_session),
        )
        .route(
            "/api/chat/sessions/{id}",
            get(chat_handlers::get_session).delete(chat_handlers::delete_session),
        )
        .route(
            "/api/chat/sessions/{id}/stream",
            get(chat_handlers::stream_events),
        )
        .route(
            "/api/chat/sessions/{id}/messages",
            get(chat_handlers::list_messages).post(chat_handlers::send_message),
        )
        .route(
            "/api/chat/sessions/{id}/interrupt",
            post(chat_handlers::interrupt_session),
        )
        // Middleware
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state)
}
