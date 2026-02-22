//! API route definitions
//!
//! Routes are split into public (no auth) and protected (require valid JWT).
//! The `require_auth` middleware is applied only to protected routes.

use super::auth_handlers;
use super::chat_handlers;
use super::code_handlers;
use super::handlers::{self, OrchestratorState};
use super::note_handlers;
use super::project_handlers;
use super::workspace_handlers;
use super::ws_chat_handler;
use super::ws_handlers;
use crate::auth::middleware::require_auth;
use axum::http::{header, Method};
use axum::{
    middleware::from_fn_with_state,
    routing::{delete, get, post},
    Router,
};
use tower_http::cors::CorsLayer;
#[cfg(not(feature = "embedded-frontend"))]
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

/// Create the API router with public + protected route groups.
///
/// Frontend serving behavior depends on the build:
/// - **`embedded-frontend` feature ON**: The SPA is baked into the binary via
///   rust-embed; `serve_frontend` and `frontend_path` are ignored.
/// - **`embedded-frontend` feature OFF** (default): When `serve_frontend` is true,
///   a `ServeDir` fallback serves files from `frontend_path` with SPA routing.
///
/// In both cases, Axum's explicit routes (/api/*, /auth/*, /ws/*, /health,
/// /hooks/*, /internal/*) always take priority — only unmatched paths hit the fallback.
pub fn create_router(state: OrchestratorState) -> Router {
    let cors = build_cors(&state);

    let public = public_routes();
    let protected = protected_routes().layer(from_fn_with_state(state.clone(), require_auth));

    let router = public
        .merge(protected)
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state.clone());

    attach_frontend(router, &state)
}

/// Attach frontend serving to the router.
///
/// When the `embedded-frontend` feature is active, the SPA is served from
/// memory (rust-embed). Otherwise, it's served from the filesystem via ServeDir.
#[cfg(feature = "embedded-frontend")]
fn attach_frontend(router: Router, _state: &OrchestratorState) -> Router {
    tracing::info!("Frontend serving: embedded (rust-embed, compiled into binary)");
    router.fallback(super::embedded_frontend::serve_embedded)
}

#[cfg(not(feature = "embedded-frontend"))]
fn attach_frontend(router: Router, state: &OrchestratorState) -> Router {
    if state.serve_frontend {
        // Uses `fallback()` (not `not_found_service()`) to preserve 200 status
        // on SPA routes — `not_found_service` would force 404 on every fallback.
        let index_path = format!("{}/index.html", state.frontend_path);
        let serve_dir = ServeDir::new(&state.frontend_path).fallback(ServeFile::new(index_path));

        // Prevent stale caching: WKWebView (Tauri desktop) aggressively caches
        // HTTP responses using heuristic expiration when no Cache-Control is set.
        // Without this, index.html can be served from a previous app version even
        // after reinstalling — the WebKit cache lives in ~/Library/WebKit/ and
        // ~/Library/Caches/<bundle-id>/, NOT inside the .app bundle.
        let service = tower::ServiceBuilder::new()
            .layer(tower_http::set_header::SetResponseHeaderLayer::overriding(
                header::CACHE_CONTROL,
                header::HeaderValue::from_static("no-cache, no-store, must-revalidate"),
            ))
            .service(serve_dir);

        router.fallback_service(service)
    } else {
        router
    }
}

/// Build CORS layer with credentials support for cookie-based auth.
///
/// `allow_credentials(true)` is required for browsers to send the `refresh_token`
/// HttpOnly cookie on cross-origin requests (including WebSocket upgrades).
///
/// CORS spec requires that when `credentials: true`:
/// - Origins MUST be explicit (not `*`)
/// - Methods MUST be explicit (not `*`)
/// - Headers MUST be explicit (not `*`)
fn build_cors(state: &OrchestratorState) -> CorsLayer {
    // Build allowed origins from ServerState (localhost:{port}, tauri://localhost, public_url)
    let origin_strings = state.allowed_origins();

    let origins: Vec<axum::http::HeaderValue> = origin_strings
        .iter()
        .filter_map(|s| s.parse::<axum::http::HeaderValue>().ok())
        .collect();

    CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers([
            header::AUTHORIZATION,
            header::CONTENT_TYPE,
            header::ACCEPT,
            header::COOKIE,
        ])
        .allow_credentials(true)
}

// ============================================================================
// Public routes (no authentication required)
// ============================================================================

/// Routes accessible without authentication.
///
/// Includes: health check, OAuth login/callback, webhook endpoints, internal events.
fn public_routes() -> Router<OrchestratorState> {
    Router::new()
        // Health check, version & setup status
        .route("/health", get(handlers::health))
        .route("/api/version", get(handlers::get_version))
        .route("/api/setup-status", get(handlers::setup_status))
        // ================================================================
        // Auth (public — login flow + discovery)
        // ================================================================
        .route("/auth/providers", get(auth_handlers::get_auth_providers))
        .route("/auth/login", post(auth_handlers::password_login))
        .route("/auth/register", post(auth_handlers::register))
        // OIDC generic routes
        .route("/auth/oidc", get(auth_handlers::oidc_login))
        .route("/auth/oidc/callback", post(auth_handlers::oidc_callback))
        // Legacy Google routes (alias → same underlying OIDC flow for google_login,
        // google_callback kept for direct Google OAuth backward compat)
        .route("/auth/google", get(auth_handlers::google_login))
        .route(
            "/auth/google/callback",
            post(auth_handlers::google_callback),
        )
        // Token refresh (public — reads cookie, not Bearer. Allows refresh with expired JWT)
        .route("/auth/refresh", post(auth_handlers::refresh_token))
        // Logout (public — revokes refresh token cookie, works with expired JWT)
        .route("/auth/logout", post(auth_handlers::logout))
        // WS ticket (public — reads cookie to issue a short-lived WS auth ticket.
        // Workaround for WKWebView not sending cookies on WebSocket upgrades.)
        .route("/auth/ws-ticket", post(auth_handlers::ws_ticket))
        // ================================================================
        // WebSocket (public — auth via cookie or ticket fallback)
        // ================================================================
        .route("/ws/events", get(ws_handlers::ws_events))
        .route("/ws/chat/{session_id}", get(ws_chat_handler::ws_chat))
        // ================================================================
        // Webhooks & Internal
        // ================================================================
        .route("/hooks/wake", post(handlers::wake))
        // DEPRECATED: Use NATS for inter-process events. Kept for backward compatibility.
        .route("/internal/events", post(handlers::receive_event))
}

// ============================================================================
// Protected routes (require valid JWT)
// ============================================================================

/// Routes that require a valid Bearer JWT token.
///
/// The `require_auth` middleware layer is applied by `create_router`.
fn protected_routes() -> Router<OrchestratorState> {
    Router::new()
        // ================================================================
        // Auth (protected — user info)
        // ================================================================
        .route("/auth/me", get(auth_handlers::get_me))
        // ================================================================
        // Projects (multi-project support)
        // ================================================================
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
        // ================================================================
        // Plans (global or legacy)
        // ================================================================
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
        .route(
            "/api/releases/{release_id}/commits/{commit_sha}",
            delete(handlers::remove_commit_from_release),
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
            "/api/milestones/{milestone_id}/plans",
            post(handlers::link_plan_to_milestone),
        )
        .route(
            "/api/milestones/{milestone_id}/plans/{plan_id}",
            delete(handlers::unlink_plan_from_milestone),
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
        // Webhooks (protected — /api prefix)
        .route("/api/wake", post(handlers::wake))
        // ================================================================
        // File Watcher (auto-sync on file changes)
        // ================================================================
        .route(
            "/api/watch",
            get(handlers::watch_status)
                .post(handlers::start_watch)
                .delete(handlers::stop_watch),
        )
        // ================================================================
        // Code Exploration (Graph + Search powered)
        // ================================================================
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
        // ================================================================
        // Feature Graphs
        // ================================================================
        .route(
            "/api/feature-graphs",
            get(code_handlers::list_feature_graphs).post(code_handlers::create_feature_graph),
        )
        .route(
            "/api/feature-graphs/auto-build",
            post(code_handlers::auto_build_feature_graph),
        )
        .route(
            "/api/feature-graphs/{id}",
            get(code_handlers::get_feature_graph).delete(code_handlers::delete_feature_graph),
        )
        .route(
            "/api/feature-graphs/{id}/entities",
            post(code_handlers::add_entity_to_feature_graph),
        )
        // ================================================================
        // Structural Analytics
        // ================================================================
        .route(
            "/api/code/communities",
            get(code_handlers::get_code_communities),
        )
        .route("/api/code/health", get(code_handlers::get_code_health))
        // ================================================================
        // Implementation Planner
        // ================================================================
        .route(
            "/api/code/plan-implementation",
            post(code_handlers::plan_implementation),
        )
        // ================================================================
        // Knowledge Notes
        // ================================================================
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
        .route(
            "/api/notes/neurons/search",
            get(note_handlers::search_neurons),
        )
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
        // ================================================================
        // Admin — Embedding Backfill
        // ================================================================
        .route(
            "/api/admin/backfill-embeddings",
            post(note_handlers::start_backfill_embeddings)
                .delete(note_handlers::cancel_backfill_embeddings),
        )
        .route(
            "/api/admin/backfill-embeddings/status",
            get(note_handlers::get_backfill_embeddings_status),
        )
        // ================================================================
        // Admin — Synapse Backfill
        // ================================================================
        .route(
            "/api/admin/backfill-synapses",
            post(note_handlers::start_backfill_synapses)
                .delete(note_handlers::cancel_backfill_synapses),
        )
        .route(
            "/api/admin/backfill-synapses/status",
            get(note_handlers::get_backfill_synapses_status),
        )
        // ================================================================
        // Meilisearch Maintenance
        // ================================================================
        .route(
            "/api/meilisearch/stats",
            get(handlers::get_meilisearch_stats),
        )
        .route(
            "/api/meilisearch/orphans",
            axum::routing::delete(handlers::delete_meilisearch_orphans),
        )
        // ================================================================
        // Workspaces
        // ================================================================
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
            get(workspace_handlers::list_workspace_milestone_tasks)
                .post(workspace_handlers::add_task_to_workspace_milestone),
        )
        .route(
            "/api/workspace-milestones/{id}/plans",
            post(workspace_handlers::link_plan_to_workspace_milestone),
        )
        .route(
            "/api/workspace-milestones/{id}/plans/{plan_id}",
            delete(workspace_handlers::unlink_plan_from_workspace_milestone),
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
        // ================================================================
        // Chat (session management — streaming via WebSocket above)
        // ================================================================
        .route(
            "/api/chat/sessions",
            get(chat_handlers::list_sessions).post(chat_handlers::create_session),
        )
        .route("/api/chat/search", get(chat_handlers::search_messages))
        .route(
            "/api/chat/sessions/backfill-previews",
            post(chat_handlers::backfill_previews),
        )
        .route(
            "/api/chat/sessions/{id}",
            get(chat_handlers::get_session).delete(chat_handlers::delete_session),
        )
        .route(
            "/api/chat/sessions/{id}/messages",
            get(chat_handlers::list_messages),
        )
        // Chat permission config (runtime GET/PUT)
        .route(
            "/api/chat/config/permissions",
            get(chat_handlers::get_chat_permissions).put(chat_handlers::update_chat_permissions),
        )
        // Chat full configuration (GET/PATCH — includes permissions + env config)
        .route(
            "/api/chat/config",
            get(chat_handlers::get_chat_config).patch(chat_handlers::update_chat_config),
        )
        // Detect user PATH from login shell
        .route("/api/chat/detect-path", get(chat_handlers::detect_path))
        // CLI version management (check + install/upgrade)
        .route("/api/chat/cli/status", get(chat_handlers::get_cli_status))
        .route("/api/chat/cli/install", post(chat_handlers::install_cli))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventBus;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_auth_config};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt; // for oneshot

    /// Build a test router with serve_frontend enabled, pointing at a temp dir
    async fn test_app_with_frontend(dir: &std::path::Path) -> Router {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(RwLock::new(FileWatcher::new(orchestrator.clone())));
        let state = Arc::new(handlers::ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: true,
            frontend_path: dir.to_str().unwrap().to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
        });
        create_router(state)
    }

    /// Build a test router with serve_frontend disabled
    async fn test_app_no_frontend() -> Router {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(RwLock::new(FileWatcher::new(orchestrator.clone())));
        let state = Arc::new(handlers::ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
        });
        create_router(state)
    }

    /// Create a fake dist/ directory with index.html and an asset file
    fn create_fake_dist() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("index.html"),
            "<!DOCTYPE html><html><body>SPA</body></html>",
        )
        .unwrap();
        // Create assets subdirectory with a JS file
        let assets = dir.path().join("assets");
        std::fs::create_dir(&assets).unwrap();
        std::fs::write(assets.join("app.js"), "console.log('hello');").unwrap();
        dir
    }

    // ====================================================================
    // SPA fallback: GET / returns index.html
    // ====================================================================

    #[tokio::test]
    async fn test_spa_root_returns_index_html() {
        let dist = create_fake_dist();
        let app = test_app_with_frontend(dist.path()).await;

        let resp = app
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 10_000)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            text.contains("SPA"),
            "Expected index.html content, got: {text}"
        );
    }

    // ====================================================================
    // SPA fallback: GET /workspaces (client-side route) returns index.html
    // ====================================================================

    #[tokio::test]
    async fn test_spa_client_route_returns_index_html() {
        let dist = create_fake_dist();
        let app = test_app_with_frontend(dist.path()).await;

        let resp = app
            .oneshot(Request::get("/workspaces").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 10_000)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("SPA"), "SPA route should return index.html");
    }

    // ====================================================================
    // Static assets: GET /assets/app.js returns the JS file
    // ====================================================================

    #[tokio::test]
    async fn test_static_asset_served() {
        let dist = create_fake_dist();
        let app = test_app_with_frontend(dist.path()).await;

        let resp = app
            .oneshot(Request::get("/assets/app.js").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 10_000)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("console.log"), "Expected JS content");
    }

    // ====================================================================
    // API routes NOT intercepted: /health still works
    // ====================================================================

    #[tokio::test]
    async fn test_health_not_intercepted_by_fallback() {
        let dist = create_fake_dist();
        let app = test_app_with_frontend(dist.path()).await;

        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 10_000)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        // /health returns JSON with status field, NOT index.html
        assert!(
            text.contains("\"status\""),
            "Expected JSON health response, got: {text}"
        );
        assert!(!text.contains("SPA"), "Health should NOT return index.html");
    }

    // ====================================================================
    // serve_frontend=false: SPA routes return 404
    // ====================================================================

    #[tokio::test]
    async fn test_no_frontend_no_spa_fallback() {
        let app = test_app_no_frontend().await;

        let resp = app
            .oneshot(Request::get("/some-spa-route").body(Body::empty()).unwrap())
            .await
            .unwrap();

        // Without frontend serving, unmatched routes have no fallback.
        // The response will be either 404 (no route) or 401 (auth middleware rejects).
        // Either way, it must NOT be 200 with index.html content.
        let status = resp.status();
        assert!(
            status == StatusCode::NOT_FOUND || status == StatusCode::UNAUTHORIZED,
            "Expected 404 or 401 without frontend, got: {status}"
        );
    }

    // ====================================================================
    // /auth/callback (React Router route) gets index.html, not 404
    // ====================================================================

    #[tokio::test]
    async fn test_auth_callback_spa_route_returns_index_html() {
        let dist = create_fake_dist();
        let app = test_app_with_frontend(dist.path()).await;

        let resp = app
            .oneshot(Request::get("/auth/callback").body(Body::empty()).unwrap())
            .await
            .unwrap();

        // /auth/callback is NOT a registered backend route (backend has /auth/oidc/callback POST).
        // As a GET, it should fall through to the SPA fallback and return index.html.
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 10_000)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            text.contains("SPA"),
            "/auth/callback GET should return SPA index.html"
        );
    }

    // ====================================================================
    // Cache-Control: index.html (and SPA fallback) must have no-cache
    // ====================================================================

    #[tokio::test]
    async fn test_index_html_has_no_cache_headers() {
        let dist = create_fake_dist();
        let app = test_app_with_frontend(dist.path()).await;

        let resp = app
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let cache_control = resp
            .headers()
            .get("cache-control")
            .map(|v| v.to_str().unwrap().to_string());
        assert_eq!(
            cache_control.as_deref(),
            Some("no-cache, no-store, must-revalidate"),
            "index.html must have no-cache to prevent stale WKWebView caching"
        );
    }

    // ====================================================================
    // Cache-Control: static assets also get no-cache (ServeDir wrapper)
    // ====================================================================

    #[tokio::test]
    async fn test_static_asset_has_cache_control() {
        let dist = create_fake_dist();
        let app = test_app_with_frontend(dist.path()).await;

        let resp = app
            .oneshot(Request::get("/assets/app.js").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let cache_control = resp
            .headers()
            .get("cache-control")
            .map(|v| v.to_str().unwrap().to_string());
        assert!(
            cache_control.is_some(),
            "Static assets must have Cache-Control header"
        );
    }
}
