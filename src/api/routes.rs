//! API route definitions
//!
//! Routes are split into public (no auth) and protected (require valid JWT).
//! The `require_auth` middleware is applied only to protected routes.

use super::auth_handlers;
use super::chat_handlers;
use super::code_handlers;
use super::episode_handlers;
use super::handlers::{self, OrchestratorState};
use super::hook_handlers;
use super::note_handlers;
use super::profile_handlers;
use super::project_handlers;
use super::protocol_handlers;
use super::reason_handlers;
use super::registry_handlers;
use super::rfc_handlers;
use super::skill_handlers;
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
        // ================================================================
        // Hook activation (public — called from Claude Code hooks, rate limited)
        // ================================================================
        .route("/api/hooks/activate", post(hook_handlers::activate_hook))
        .route("/api/hooks/health", get(hook_handlers::hooks_health))
        .route(
            "/api/hooks/session-context",
            get(hook_handlers::session_context),
        )
        .route(
            "/api/hooks/resolve-project",
            get(hook_handlers::resolve_project),
        )
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
        .route(
            "/api/projects/{slug}/graph",
            get(project_handlers::get_project_graph),
        )
        .route(
            "/api/projects/{slug}/intelligence/summary",
            get(project_handlers::get_intelligence_summary),
        )
        .route(
            "/api/projects/{slug}/embeddings/projection",
            get(project_handlers::get_embeddings_projection),
        )
        .route(
            "/api/projects/{slug}/health-dashboard",
            get(project_handlers::get_health_dashboard),
        )
        .route(
            "/api/projects/{slug}/auto-roadmap",
            get(project_handlers::get_auto_roadmap),
        )
        .route(
            "/api/projects/{slug}/scaffolding",
            get(project_handlers::get_scaffolding_level)
                .put(project_handlers::set_scaffolding_level),
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
        .route("/api/plans/{plan_id}/waves", get(handlers::get_plan_waves))
        // Runner
        .route("/api/plans/{plan_id}/run", post(handlers::run_plan))
        .route(
            "/api/plans/{plan_id}/run/status",
            get(handlers::get_run_status),
        )
        .route(
            "/api/plans/{plan_id}/run/cancel",
            post(handlers::cancel_run),
        )
        .route(
            "/api/plans/{plan_id}/run/auto-pr",
            post(handlers::create_auto_pr),
        )
        // Plan Runs
        .route("/api/plans/{plan_id}/runs", get(handlers::list_plan_runs))
        .route("/api/runs/{run_id}", get(handlers::get_plan_run))
        .route(
            "/api/plans/{plan_id}/runs/compare",
            post(handlers::compare_plan_runs),
        )
        .route(
            "/api/plans/{plan_id}/runs/predict",
            post(handlers::predict_plan_run),
        )
        // Triggers
        .route(
            "/api/plans/{plan_id}/triggers",
            get(handlers::list_triggers).post(handlers::create_trigger),
        )
        .route(
            "/api/triggers/{trigger_id}",
            delete(handlers::delete_trigger),
        )
        .route(
            "/api/triggers/{trigger_id}/enable",
            post(handlers::enable_trigger),
        )
        .route(
            "/api/triggers/{trigger_id}/disable",
            post(handlers::disable_trigger),
        )
        .route(
            "/api/triggers/{trigger_id}/firings",
            get(handlers::list_trigger_firings),
        )
        // Webhooks
        .route(
            "/api/webhooks/{trigger_id}",
            post(handlers::receive_webhook),
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
        .route(
            "/api/plans/{plan_id}/tasks/{task_id}/build_prompt",
            post(handlers::build_task_prompt),
        )
        // Task delegation — sub-agent orchestration
        .route(
            "/api/plans/{plan_id}/tasks/{task_id}/delegate",
            post(handlers::delegate_task),
        )
        // Pre-enrichment pipeline
        .route("/api/plans/{plan_id}/enrich", post(handlers::enrich_plan))
        .route(
            "/api/plans/{plan_id}/tasks/{task_id}/enrich",
            post(handlers::enrich_task),
        )
        // Decisions
        .route(
            "/api/tasks/{task_id}/decisions",
            post(handlers::add_decision),
        )
        .route(
            "/api/decisions/affecting",
            get(handlers::get_decisions_affecting),
        )
        .route(
            "/api/decisions/timeline",
            get(handlers::get_decision_timeline),
        )
        .route(
            "/api/decisions/{decision_id}",
            get(handlers::get_decision)
                .patch(handlers::update_decision)
                .delete(handlers::delete_decision),
        )
        .route("/api/decisions/search", get(handlers::search_decisions))
        .route(
            "/api/decisions/search-semantic",
            get(handlers::search_decisions_semantic),
        )
        // Decision Affects
        .route(
            "/api/decisions/{decision_id}/affects",
            post(handlers::add_decision_affects)
                .get(handlers::list_decision_affects)
                .delete(handlers::remove_decision_affects_query),
        )
        .route(
            "/api/decisions/{decision_id}/affects/{entity_type}/{entity_id}",
            delete(handlers::remove_decision_affects),
        )
        // Decision Supersedes
        .route(
            "/api/decisions/{new_id}/supersedes/{old_id}",
            post(handlers::supersede_decision),
        )
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
        // TOUCHES — Commit ↔ File queries
        .route(
            "/api/commits/{commit_sha}/files",
            get(handlers::get_commit_files),
        )
        .route("/api/files/history", get(handlers::get_file_history))
        // CO_CHANGED — File coupling queries
        .route(
            "/api/projects/{project_id}/co-changes",
            get(handlers::get_co_change_graph),
        )
        .route(
            "/api/files/co-changers",
            get(handlers::get_file_co_changers),
        )
        // Backfill TOUCHES from git history
        .route(
            "/api/projects/{project_slug}/backfill-touches",
            post(handlers::backfill_commit_touches),
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
            "/api/code/symbols/{*file_path}",
            get(code_handlers::get_file_symbols),
        )
        // Find all references to a symbol
        .route("/api/code/references", get(code_handlers::find_references))
        // Get file dependencies (imports + dependents)
        .route(
            "/api/code/dependencies/{*file_path}",
            get(code_handlers::get_file_dependencies),
        )
        // Get call graph for a function
        .route("/api/code/callgraph", get(code_handlers::get_call_graph))
        // Analyze impact of changes
        .route("/api/code/impact", get(code_handlers::analyze_impact))
        // Multi-signal impact fusion (Plan 4)
        .route(
            "/api/code/impact/multi",
            get(code_handlers::analyze_impact_v2),
        )
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
        .route(
            "/api/feature-graphs/{id}/statistics",
            get(code_handlers::get_feature_graph_statistics),
        )
        .route(
            "/api/feature-graphs/{id}/overlapping",
            get(code_handlers::find_overlapping_feature_graphs),
        )
        .route(
            "/api/feature-graphs/compare",
            get(code_handlers::compare_feature_graphs),
        )
        // ================================================================
        // Structural Analytics
        // ================================================================
        .route(
            "/api/code/communities",
            get(code_handlers::get_code_communities),
        )
        .route("/api/code/health", get(code_handlers::get_code_health))
        .route(
            "/api/code/node-importance",
            get(code_handlers::get_node_importance),
        )
        // Churn hotspots, knowledge gaps, risk assessment (T5.5, T5.6, T5.7)
        .route(
            "/api/code/hotspots",
            get(code_handlers::get_change_hotspots),
        )
        .route(
            "/api/code/knowledge-gaps",
            get(code_handlers::get_knowledge_gaps),
        )
        .route(
            "/api/code/risk-assessment",
            get(code_handlers::get_risk_assessment),
        )
        .route("/api/code/homeostasis", get(code_handlers::get_homeostasis))
        .route(
            "/api/code/structural-drift",
            get(code_handlers::get_structural_drift),
        )
        // ================================================================
        // Heritage Navigation
        // ================================================================
        .route(
            "/api/code/class-hierarchy",
            get(code_handlers::get_class_hierarchy),
        )
        .route("/api/code/subclasses", get(code_handlers::find_subclasses))
        .route(
            "/api/code/interface-implementors",
            get(code_handlers::find_interface_implementors),
        )
        // ================================================================
        // Process Detection & Navigation
        // ================================================================
        .route(
            "/api/code/processes/detect",
            post(code_handlers::detect_processes),
        )
        .route("/api/code/processes", get(code_handlers::list_processes))
        .route(
            "/api/code/processes/detail",
            get(code_handlers::get_process_detail),
        )
        .route(
            "/api/code/entry-points",
            get(code_handlers::get_entry_points),
        )
        // ================================================================
        // Community Enrichment
        // ================================================================
        .route(
            "/api/code/communities/enrich",
            post(code_handlers::enrich_communities),
        )
        // ================================================================
        // Bridge Subgraph (GraIL)
        // ================================================================
        .route("/api/code/bridge", get(code_handlers::get_bridge))
        // ================================================================
        // Topological Firewall (GraIL Plan 3)
        // ================================================================
        .route(
            "/api/code/topology/check",
            get(code_handlers::check_topology),
        )
        .route(
            "/api/code/topology/rules",
            get(code_handlers::list_topology_rules).post(code_handlers::create_topology_rule),
        )
        .route(
            "/api/code/topology/rules/{rule_id}",
            delete(code_handlers::delete_topology_rule),
        )
        .route(
            "/api/code/topology/check-file",
            post(code_handlers::check_file_topology),
        )
        // ================================================================
        // Structural DNA
        // ================================================================
        .route(
            "/api/code/structural-profile",
            post(code_handlers::get_structural_profile),
        )
        .route(
            "/api/code/structural-twins",
            post(code_handlers::find_structural_twins),
        )
        .route(
            "/api/code/structural-clusters",
            post(code_handlers::cluster_dna),
        )
        .route(
            "/api/code/structural-twins/cross-project",
            post(code_handlers::find_cross_project_twins),
        )
        // ================================================================
        // Link Prediction
        // ================================================================
        .route(
            "/api/code/predict-links",
            post(code_handlers::predict_missing_links),
        )
        .route(
            "/api/code/link-plausibility",
            post(code_handlers::check_link_plausibility),
        )
        // ================================================================
        // Stress Testing
        // ================================================================
        .route(
            "/api/code/stress-test-node",
            post(code_handlers::stress_test_node),
        )
        .route(
            "/api/code/stress-test-edge",
            post(code_handlers::stress_test_edge),
        )
        .route(
            "/api/code/stress-test-cascade",
            post(code_handlers::stress_test_cascade),
        )
        .route("/api/code/find-bridges", post(code_handlers::find_bridges))
        // ================================================================
        // Context Cards (GraIL Plan 8)
        // ================================================================
        .route(
            "/api/code/context-card",
            get(code_handlers::get_context_card),
        )
        .route(
            "/api/code/context-cards/refresh",
            post(code_handlers::refresh_context_cards),
        )
        // ================================================================
        // WL Fingerprint & Isomorphic Groups (GraIL Plan 7)
        // ================================================================
        .route("/api/code/fingerprint", get(code_handlers::get_fingerprint))
        .route("/api/code/isomorphic", get(code_handlers::find_isomorphic))
        .route(
            "/api/code/structural-templates",
            get(code_handlers::suggest_structural_templates),
        )
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
            "/api/notes/search-semantic",
            get(note_handlers::search_notes_semantic),
        )
        .route(
            "/api/notes/neurons/search",
            get(note_handlers::search_neurons),
        )
        .route(
            "/api/notes/neurons/reinforce",
            post(note_handlers::reinforce_neurons),
        )
        .route(
            "/api/notes/neurons/decay",
            post(note_handlers::decay_synapses),
        )
        .route(
            "/api/notes/neurons/heal-scars",
            post(note_handlers::heal_scars),
        )
        .route(
            "/api/notes/consolidate-memory",
            post(note_handlers::consolidate_memory),
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
        // Update energy scores (Hebbian decay)
        .route(
            "/api/notes/update-energy",
            post(note_handlers::update_energy_scores),
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
        // Unified context knowledge (notes + decisions + commits)
        .route(
            "/api/notes/context-knowledge",
            get(note_handlers::get_context_knowledge),
        )
        // Enriched propagated knowledge (notes + decisions + relation stats)
        .route(
            "/api/notes/propagated-knowledge",
            get(note_handlers::get_propagated_knowledge),
        )
        // Entity notes (direct only)
        .route(
            "/api/entities/{entity_type}/{entity_id}/notes",
            get(note_handlers::get_entity_notes),
        )
        // ================================================================
        // Analysis Profiles
        // ================================================================
        .route(
            "/api/analysis-profiles",
            get(profile_handlers::list_analysis_profiles)
                .post(profile_handlers::create_analysis_profile),
        )
        .route(
            "/api/analysis-profiles/{profile_id}",
            get(profile_handlers::get_analysis_profile)
                .delete(profile_handlers::delete_analysis_profile),
        )
        // ================================================================
        // Skills (Neural Skills)
        // ================================================================
        .route(
            "/api/skills",
            get(skill_handlers::list_skills).post(skill_handlers::create_skill),
        )
        .route(
            "/api/skills/{skill_id}",
            get(skill_handlers::get_skill)
                .put(skill_handlers::update_skill)
                .delete(skill_handlers::delete_skill),
        )
        .route(
            "/api/skills/{skill_id}/members",
            get(skill_handlers::get_skill_members).post(skill_handlers::add_skill_member),
        )
        .route(
            "/api/skills/{skill_id}/members/{entity_type}/{entity_id}",
            axum::routing::delete(skill_handlers::remove_skill_member),
        )
        .route(
            "/api/skills/{skill_id}/activate",
            post(skill_handlers::activate_skill),
        )
        .route(
            "/api/skills/{skill_id}/export",
            get(skill_handlers::export_skill),
        )
        .route("/api/skills/import", post(skill_handlers::import_skill))
        .route(
            "/api/skills/{skill_id}/health",
            get(skill_handlers::get_skill_health),
        )
        .route(
            "/api/skills/{skill_id}/split",
            post(skill_handlers::split_skill),
        )
        .route("/api/skills/merge", post(skill_handlers::merge_skills))
        // ================================================================
        // Skill Registry (Pattern Federation)
        // ================================================================
        .route(
            "/api/registry/publish",
            post(registry_handlers::publish_skill),
        )
        .route(
            "/api/registry/search",
            get(registry_handlers::search_registry),
        )
        .route(
            "/api/registry/{id}",
            get(registry_handlers::get_published_skill),
        )
        .route(
            "/api/registry/{id}/import",
            post(registry_handlers::import_from_registry),
        )
        // ================================================================
        // RFCs (Notes with note_type=rfc, frontend-friendly API)
        // ================================================================
        .route(
            "/api/rfcs",
            get(rfc_handlers::list_rfcs).post(rfc_handlers::create_rfc),
        )
        .route(
            "/api/rfcs/{rfc_id}",
            get(rfc_handlers::get_rfc)
                .patch(rfc_handlers::update_rfc)
                .delete(rfc_handlers::delete_rfc),
        )
        .route(
            "/api/rfcs/{rfc_id}/transition",
            post(rfc_handlers::transition_rfc),
        )
        // ================================================================
        // Protocols (Pattern Federation)
        // ================================================================
        .route(
            "/api/protocols",
            get(protocol_handlers::list_protocols).post(protocol_handlers::create_protocol),
        )
        .route(
            "/api/protocols/route",
            get(protocol_handlers::route_protocols),
        )
        .route(
            "/api/protocols/compose",
            post(protocol_handlers::compose_protocol),
        )
        .route(
            "/api/protocols/simulate",
            post(protocol_handlers::simulate_activation),
        )
        .route(
            "/api/protocols/{protocol_id}",
            get(protocol_handlers::get_protocol)
                .put(protocol_handlers::update_protocol)
                .delete(protocol_handlers::delete_protocol),
        )
        .route(
            "/api/protocols/{protocol_id}/states",
            get(protocol_handlers::list_states).post(protocol_handlers::add_state),
        )
        .route(
            "/api/protocols/{protocol_id}/states/{state_id}",
            axum::routing::delete(protocol_handlers::delete_state),
        )
        .route(
            "/api/protocols/{protocol_id}/transitions",
            get(protocol_handlers::list_transitions).post(protocol_handlers::add_transition),
        )
        .route(
            "/api/protocols/{protocol_id}/transitions/{transition_id}",
            axum::routing::delete(protocol_handlers::delete_transition),
        )
        .route(
            "/api/protocols/{protocol_id}/link-skill",
            post(protocol_handlers::link_to_skill),
        )
        // Protocol Runs (FSM Runtime)
        .route(
            "/api/protocols/{protocol_id}/runs",
            get(protocol_handlers::list_runs).post(protocol_handlers::start_run),
        )
        .route(
            "/api/protocols/runs/{run_id}",
            get(protocol_handlers::get_run).delete(protocol_handlers::delete_run),
        )
        .route(
            "/api/protocols/runs/{run_id}/transition",
            post(protocol_handlers::fire_transition),
        )
        .route(
            "/api/protocols/runs/{run_id}/cancel",
            post(protocol_handlers::cancel_run),
        )
        .route(
            "/api/protocols/runs/{run_id}/fail",
            post(protocol_handlers::fail_run),
        )
        .route(
            "/api/protocols/runs/{run_id}/progress",
            post(protocol_handlers::report_progress),
        )
        .route(
            "/api/protocols/runs/{run_id}/children",
            get(protocol_handlers::get_run_children),
        )
        .route(
            "/api/protocols/runs/{run_id}/tree",
            get(protocol_handlers::get_run_tree),
        )
        // ================================================================
        // Reasoning Tree
        // ================================================================
        .route("/api/reason", post(reason_handlers::reason))
        .route(
            "/api/reason/{tree_id}/feedback",
            post(reason_handlers::reason_feedback),
        )
        // ================================================================
        // Episodes (Episodic Memory)
        // ================================================================
        .route("/api/episodes", get(episode_handlers::list_episodes))
        .route(
            "/api/episodes/collect",
            post(episode_handlers::collect_episode),
        )
        .route(
            "/api/episodes/anonymize",
            post(episode_handlers::anonymize_episode),
        )
        .route(
            "/api/episodes/export-artifact",
            post(episode_handlers::export_artifact),
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
        // Admin — Decision Reindex & Backfill
        // ================================================================
        .route(
            "/api/admin/reindex-decisions",
            post(handlers::reindex_decisions),
        )
        .route(
            "/api/admin/backfill-decision-embeddings",
            post(handlers::backfill_decision_embeddings),
        )
        .route(
            "/api/admin/backfill-decision-project-slugs",
            post(handlers::backfill_decision_project_slugs),
        )
        .route(
            "/api/admin/backfill-discussed",
            post(handlers::backfill_discussed),
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
        // Admin Cleanup
        // ================================================================
        .route(
            "/api/admin/cleanup-cross-project-calls",
            post(handlers::cleanup_cross_project_calls),
        )
        .route(
            "/api/admin/cleanup-builtin-calls",
            post(handlers::cleanup_builtin_calls),
        )
        .route(
            "/api/admin/migrate-calls-confidence",
            post(handlers::migrate_calls_confidence),
        )
        .route(
            "/api/admin/cleanup-sync-data",
            post(handlers::cleanup_sync_data),
        )
        // ================================================================
        // Admin — Knowledge Fabric
        // ================================================================
        .route(
            "/api/admin/update-fabric-scores",
            post(handlers::update_fabric_scores),
        )
        .route(
            "/api/admin/bootstrap-knowledge-fabric",
            post(handlers::bootstrap_knowledge_fabric),
        )
        .route("/api/admin/audit-gaps", post(handlers::audit_gaps))
        .route(
            "/api/admin/persist-health-report",
            post(handlers::persist_health_report),
        )
        .route(
            "/api/admin/reinforce-isomorphic",
            post(handlers::reinforce_isomorphic_synapses),
        )
        .route("/api/admin/detect-skills", post(handlers::detect_skills))
        .route(
            "/api/admin/detect-skill-fission",
            post(handlers::detect_skill_fission),
        )
        .route(
            "/api/admin/detect-skill-fusion",
            post(handlers::detect_skill_fusion),
        )
        .route(
            "/api/admin/auto-anchor-notes",
            post(handlers::auto_anchor_notes),
        )
        .route(
            "/api/admin/reconstruct-knowledge",
            post(handlers::reconstruct_knowledge),
        )
        .route(
            "/api/admin/skill-maintenance",
            post(handlers::skill_maintenance),
        )
        .route(
            "/api/admin/detect-stagnation/{project_id}",
            get(handlers::detect_stagnation),
        )
        .route(
            "/api/admin/deep-maintenance/{project_id}",
            post(handlers::run_deep_maintenance),
        )
        // NOTE: /api/admin/install-hooks removed — hooks are now managed
        // in-process via SkillActivationHook (zero config required)
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
        .route(
            "/api/workspaces/{slug}/coupling-matrix",
            get(workspace_handlers::get_coupling_matrix),
        )
        // Workspace Intelligence (aggregated graph + summary)
        .route(
            "/api/workspaces/{slug}/graph",
            get(workspace_handlers::get_workspace_graph),
        )
        .route(
            "/api/workspaces/{slug}/intelligence/summary",
            get(workspace_handlers::get_workspace_intelligence_summary),
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
        .route(
            "/api/chat/sessions/{id}/children",
            get(chat_handlers::get_session_children),
        )
        .route(
            "/api/chat/sessions/{id}/tree",
            get(chat_handlers::get_session_tree),
        )
        .route(
            "/api/chat/runs/{run_id}/sessions",
            get(chat_handlers::get_run_sessions),
        )
        // DISCUSSED relations (ChatSession → Entity)
        .route(
            "/api/chat/sessions/{id}/discussed",
            post(chat_handlers::add_discussed).get(chat_handlers::get_session_entities),
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
        // CLI auth status
        .route(
            "/api/chat/cli/auth-status",
            get(chat_handlers::get_cli_auth_status),
        )
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
            registry_remote_url: None,
            oidc_client: None,
            identity: None,
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
            registry_remote_url: None,
            oidc_client: None,
            identity: None,
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
