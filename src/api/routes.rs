//! API route definitions

use super::code_handlers;
use super::handlers::{self, OrchestratorState};
use super::project_handlers;
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
            get(project_handlers::get_project).delete(project_handlers::delete_project),
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
            get(handlers::get_plan).patch(handlers::update_plan_status),
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
            axum::routing::delete(handlers::delete_constraint),
        )
        // Tasks (global listing)
        .route("/api/tasks", get(handlers::list_all_tasks))
        // Tasks (plan-scoped)
        .route("/api/plans/{plan_id}/tasks", post(handlers::add_task))
        .route(
            "/api/tasks/{task_id}",
            get(handlers::get_task).patch(handlers::update_task),
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
            axum::routing::patch(handlers::update_step),
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
        .route("/api/decisions/search", get(handlers::search_decisions))
        // Sync
        .route("/api/sync", post(handlers::sync_directory))
        // Releases
        .route(
            "/api/releases/{release_id}",
            get(handlers::get_release).patch(handlers::update_release),
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
            get(handlers::get_milestone).patch(handlers::update_milestone),
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
        // Middleware
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state)
}
