//! Main orchestrator runner

use crate::embeddings::{EmbeddingProvider, FastEmbedProvider, HttpEmbeddingProvider};
use crate::events::{
    CrudAction, CrudEvent, EntityType as EventEntityType, EventEmitter, HybridEmitter,
};
use crate::graph::{AnalyticsConfig, AnalyticsDebouncer, AnalyticsEngine, GraphAnalyticsEngine};
use crate::neo4j::models::*;
use crate::neurons::{AutoReinforcementConfig, SpreadingActivationEngine};
use crate::notes::{EntityType, NoteLifecycleManager, NoteManager};
use crate::parser::{CodeParser, ParsedFile};
use crate::plan::models::*;
use crate::plan::PlanManager;
use crate::AppState;
use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use walkdir::WalkDir;

use super::context::ContextBuilder;

/// Normalize a file path to an absolute canonical form.
/// - Resolves `~` to home directory
/// - Resolves `.` and `..`
/// - Makes relative paths absolute using current_dir
/// - Returns the path as a String
fn normalize_path(path: &str) -> String {
    let expanded = if let Some(rest) = path.strip_prefix('~') {
        if let Some(home) = dirs::home_dir() {
            home.join(rest.trim_start_matches('/'))
                .to_string_lossy()
                .to_string()
        } else {
            path.to_string()
        }
    } else {
        path.to_string()
    };

    let p = std::path::Path::new(&expanded);
    if p.is_absolute() {
        // Try to canonicalize (resolves symlinks and ..)
        match p.canonicalize() {
            Ok(canonical) => canonical.to_string_lossy().to_string(),
            Err(_) => {
                // File might not exist yet — do manual cleanup of . and ..
                let mut components = Vec::new();
                for component in p.components() {
                    match component {
                        std::path::Component::ParentDir => {
                            components.pop();
                        }
                        std::path::Component::CurDir => {}
                        other => components.push(other),
                    }
                }
                let cleaned: std::path::PathBuf = components.iter().collect();
                cleaned.to_string_lossy().to_string()
            }
        }
    } else {
        // Relative path — prepend current_dir
        match std::env::current_dir() {
            Ok(cwd) => {
                let abs = cwd.join(p);
                match abs.canonicalize() {
                    Ok(canonical) => canonical.to_string_lossy().to_string(),
                    Err(_) => abs.to_string_lossy().to_string(),
                }
            }
            Err(_) => expanded,
        }
    }
}

// ============================================================================
// Analytics Staleness
// ============================================================================

/// Report on the staleness of a project's GDS analytics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StalenessReport {
    pub project_id: Uuid,
    /// True if analytics need recomputation.
    pub is_stale: bool,
    /// How long ago analytics were last computed (None if never computed).
    pub analytics_age: Option<std::time::Duration>,
    /// When analytics were last computed (None if never).
    pub analytics_computed_at: Option<chrono::DateTime<chrono::Utc>>,
    /// When the project code was last synced.
    pub last_synced: Option<chrono::DateTime<chrono::Utc>>,
}

// ============================================================================
// Feature Graph LLM Generation
// ============================================================================

/// A proposal from the LLM for a feature graph to create.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct FeatureGraphProposal {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub entry_function: String,
    #[serde(default = "FeatureGraphProposal::default_depth")]
    pub depth: u32,
    #[serde(default)]
    pub include_relations: Option<Vec<String>>,
}

impl FeatureGraphProposal {
    fn default_depth() -> u32 {
        2
    }
}

/// Main orchestrator for coordinating AI agents
pub struct Orchestrator {
    state: AppState,
    plan_manager: Arc<PlanManager>,
    context_builder: Arc<ContextBuilder>,
    parser: Arc<RwLock<CodeParser>>,
    note_manager: Arc<NoteManager>,
    note_lifecycle: Arc<NoteLifecycleManager>,
    planner: Arc<super::ImplementationPlanner>,
    analytics: Arc<dyn AnalyticsEngine>,
    analytics_debouncer: AnalyticsDebouncer,
    activation_engine: Option<Arc<SpreadingActivationEngine>>,
    auto_reinforcement: AutoReinforcementConfig,
    event_bus: Option<Arc<HybridEmitter>>,
    event_emitter: Option<Arc<dyn EventEmitter>>,
    /// Embedding provider for code embeddings (File/Function nodes).
    /// Shared with NoteManager and SpreadingActivationEngine.
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
}

/// Try to create an embedding provider from environment variables.
///
/// Provider selection via `EMBEDDING_PROVIDER` env var:
/// - `local` (default) → [`FastEmbedProvider`] using in-process ONNX Runtime (zero external dependency)
/// - `http` → [`HttpEmbeddingProvider`] using any OpenAI-compatible API (for premium/external models)
/// - `disabled` → No embedding provider (semantic search unavailable)
///
/// Returns `None` if disabled or initialization fails. Logs the result at info level.
fn init_embedding_provider() -> Option<Arc<dyn EmbeddingProvider>> {
    let provider_type = std::env::var("EMBEDDING_PROVIDER")
        .unwrap_or_else(|_| "local".to_string())
        .to_lowercase();

    match provider_type.as_str() {
        "http" => init_http_embedding_provider(),
        "disabled" | "none" | "off" => {
            tracing::info!("Embedding provider disabled (EMBEDDING_PROVIDER=disabled)");
            None
        }
        // Default: local fastembed
        _ => init_local_embedding_provider(),
    }
}

/// Initialize the local fastembed ONNX embedding provider (default).
fn init_local_embedding_provider() -> Option<Arc<dyn EmbeddingProvider>> {
    match FastEmbedProvider::from_env() {
        Ok(provider) => {
            tracing::info!(
                model = provider.model_name(),
                dimensions = provider.dimensions(),
                provider = "local",
                "Embedding provider initialized (local ONNX via fastembed)"
            );
            Some(Arc::new(provider))
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                "Failed to initialize local embedding provider"
            );
            None
        }
    }
}

/// Initialize the HTTP-based embedding provider (Ollama, OpenAI, etc.)
fn init_http_embedding_provider() -> Option<Arc<dyn EmbeddingProvider>> {
    match HttpEmbeddingProvider::from_env() {
        Some(provider) => {
            tracing::info!(
                model = provider.model_name(),
                dimensions = provider.dimensions(),
                provider = "http",
                "Embedding provider initialized (HTTP)"
            );
            Some(Arc::new(provider))
        }
        None => {
            tracing::info!("HTTP embedding provider not configured (set EMBEDDING_URL to enable)");
            None
        }
    }
}

impl Orchestrator {
    /// Create a new orchestrator
    pub async fn new(state: AppState) -> Result<Self> {
        let plan_manager = Arc::new(PlanManager::new(state.neo4j.clone(), state.meili.clone()));

        let embedding_provider = init_embedding_provider();

        let mut note_manager = NoteManager::new(state.neo4j.clone(), state.meili.clone());
        if let Some(ref provider) = embedding_provider {
            note_manager = note_manager.with_embedding_provider(provider.clone());
        }
        let note_manager = Arc::new(note_manager);

        let activation_engine = embedding_provider.clone().map(|provider| {
            Arc::new(SpreadingActivationEngine::new(
                state.neo4j.clone() as Arc<dyn crate::neo4j::GraphStore>,
                provider,
            ))
        });

        let context_builder = Arc::new(ContextBuilder::new(
            state.neo4j.clone(),
            state.meili.clone(),
            plan_manager.clone(),
            note_manager.clone(),
        ));

        let parser = Arc::new(RwLock::new(CodeParser::new()?));
        let note_lifecycle = Arc::new(NoteLifecycleManager::new());
        let planner = Arc::new(super::ImplementationPlanner::new(
            state.neo4j.clone(),
            state.meili.clone(),
            plan_manager.clone(),
            note_manager.clone(),
        ));
        let analytics: Arc<dyn AnalyticsEngine> = Arc::new(GraphAnalyticsEngine::new(
            state.neo4j.clone(),
            AnalyticsConfig::default(),
        ));
        let analytics_debouncer = AnalyticsDebouncer::with_graph_store(
            analytics.clone(),
            2000,
            Some(state.neo4j.clone()),
        );

        Ok(Self {
            state,
            plan_manager,
            context_builder,
            parser,
            note_manager,
            note_lifecycle,
            planner,
            analytics,
            analytics_debouncer,
            activation_engine,
            auto_reinforcement: AutoReinforcementConfig::default(),
            event_bus: None,
            event_emitter: None,
            embedding_provider: embedding_provider.clone(),
        })
    }

    /// Create a new orchestrator with a HybridEmitter for CRUD notifications
    ///
    /// Used by the HTTP server — the HybridEmitter provides both local broadcast
    /// (for WebSocket subscribe) and optional NATS (for inter-process sync).
    pub async fn with_event_bus(state: AppState, event_bus: Arc<HybridEmitter>) -> Result<Self> {
        let emitter: Arc<dyn EventEmitter> = event_bus.clone();
        let embedding_provider = init_embedding_provider();

        let plan_manager = Arc::new(PlanManager::with_event_emitter(
            state.neo4j.clone(),
            state.meili.clone(),
            emitter.clone(),
        ));

        let mut note_manager = NoteManager::with_event_emitter(
            state.neo4j.clone(),
            state.meili.clone(),
            emitter.clone(),
        );
        if let Some(ref provider) = embedding_provider {
            note_manager = note_manager.with_embedding_provider(provider.clone());
        }
        let note_manager = Arc::new(note_manager);

        let activation_engine = embedding_provider.clone().map(|provider| {
            Arc::new(SpreadingActivationEngine::new(
                state.neo4j.clone() as Arc<dyn crate::neo4j::GraphStore>,
                provider,
            ))
        });

        let context_builder = Arc::new(ContextBuilder::new(
            state.neo4j.clone(),
            state.meili.clone(),
            plan_manager.clone(),
            note_manager.clone(),
        ));

        let parser = Arc::new(RwLock::new(CodeParser::new()?));
        let note_lifecycle = Arc::new(NoteLifecycleManager::new());
        let planner = Arc::new(super::ImplementationPlanner::new(
            state.neo4j.clone(),
            state.meili.clone(),
            plan_manager.clone(),
            note_manager.clone(),
        ));
        let analytics: Arc<dyn AnalyticsEngine> = Arc::new(GraphAnalyticsEngine::new(
            state.neo4j.clone(),
            AnalyticsConfig::default(),
        ));
        let analytics_debouncer = AnalyticsDebouncer::with_graph_store(
            analytics.clone(),
            2000,
            Some(state.neo4j.clone()),
        );

        Ok(Self {
            state,
            plan_manager,
            context_builder,
            parser,
            note_manager,
            note_lifecycle,
            planner,
            analytics,
            analytics_debouncer,
            activation_engine,
            auto_reinforcement: AutoReinforcementConfig::default(),
            event_bus: Some(event_bus),
            event_emitter: Some(emitter),
            embedding_provider: embedding_provider.clone(),
        })
    }

    /// Create a new orchestrator with a generic EventEmitter
    ///
    /// Used by both the HTTP server (with `HybridEmitter`) and the MCP server
    /// (with `NatsEmitter` for cross-instance sync).
    pub async fn with_event_emitter(
        state: AppState,
        emitter: Arc<dyn EventEmitter>,
    ) -> Result<Self> {
        let embedding_provider = init_embedding_provider();

        let plan_manager = Arc::new(PlanManager::with_event_emitter(
            state.neo4j.clone(),
            state.meili.clone(),
            emitter.clone(),
        ));

        let mut note_manager = NoteManager::with_event_emitter(
            state.neo4j.clone(),
            state.meili.clone(),
            emitter.clone(),
        );
        if let Some(ref provider) = embedding_provider {
            note_manager = note_manager.with_embedding_provider(provider.clone());
        }
        let note_manager = Arc::new(note_manager);

        let activation_engine = embedding_provider.clone().map(|provider| {
            Arc::new(SpreadingActivationEngine::new(
                state.neo4j.clone() as Arc<dyn crate::neo4j::GraphStore>,
                provider,
            ))
        });

        let context_builder = Arc::new(ContextBuilder::new(
            state.neo4j.clone(),
            state.meili.clone(),
            plan_manager.clone(),
            note_manager.clone(),
        ));

        let parser = Arc::new(RwLock::new(CodeParser::new()?));
        let note_lifecycle = Arc::new(NoteLifecycleManager::new());
        let planner = Arc::new(super::ImplementationPlanner::new(
            state.neo4j.clone(),
            state.meili.clone(),
            plan_manager.clone(),
            note_manager.clone(),
        ));
        let analytics: Arc<dyn AnalyticsEngine> = Arc::new(GraphAnalyticsEngine::new(
            state.neo4j.clone(),
            AnalyticsConfig::default(),
        ));
        let analytics_debouncer = AnalyticsDebouncer::with_graph_store(
            analytics.clone(),
            2000,
            Some(state.neo4j.clone()),
        );

        Ok(Self {
            state,
            plan_manager,
            context_builder,
            parser,
            note_manager,
            note_lifecycle,
            planner,
            analytics,
            analytics_debouncer,
            activation_engine,
            auto_reinforcement: AutoReinforcementConfig::default(),
            event_bus: None,
            event_emitter: Some(emitter),
            embedding_provider: embedding_provider.clone(),
        })
    }

    /// Get the hybrid emitter (if configured — only available on HTTP server)
    pub fn event_bus(&self) -> Option<&Arc<HybridEmitter>> {
        self.event_bus.as_ref()
    }

    /// Emit a CRUD event (no-op if no event emitter is configured)
    fn emit(&self, event: CrudEvent) {
        if let Some(emitter) = &self.event_emitter {
            emitter.emit(event);
        }
    }

    /// Get the plan manager
    pub fn plan_manager(&self) -> &Arc<PlanManager> {
        &self.plan_manager
    }

    /// Get the context builder
    pub fn context_builder(&self) -> &Arc<ContextBuilder> {
        &self.context_builder
    }

    /// Get the graph store
    pub fn neo4j(&self) -> &dyn crate::neo4j::GraphStore {
        self.state.neo4j.as_ref()
    }

    /// Get the graph store as Arc (for sharing with ChatManager etc.)
    pub fn neo4j_arc(&self) -> Arc<dyn crate::neo4j::GraphStore> {
        self.state.neo4j.clone()
    }

    /// Spawn analytics computation in background (non-blocking).
    ///
    /// Same as `analyze_project_safe` but runs in a `tokio::spawn` so the caller
    /// returns immediately. Used by sync_project to avoid blocking the MCP/REST
    /// response while analytics are computed.
    pub fn spawn_analyze_project(&self, project_id: Uuid) {
        let analytics = self.analytics.clone();
        let neo4j = self.neo4j_arc();
        let emitter: Option<Arc<dyn EventEmitter>> = self.event_emitter.clone();
        tokio::spawn(async move {
            let start = std::time::Instant::now();
            match analytics.analyze_project(project_id).await {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    tracing::info!(
                        "Analytics computed for project {} in {:?} (files: {} nodes/{} edges, functions: {} nodes/{} edges)",
                        project_id,
                        elapsed,
                        result.file_analytics.node_count,
                        result.file_analytics.edge_count,
                        result.function_analytics.node_count,
                        result.function_analytics.edge_count,
                    );
                    if let Some(emitter) = &emitter {
                        emitter.emit(
                            CrudEvent::new(
                                EventEntityType::Project,
                                CrudAction::Updated,
                                project_id.to_string(),
                            )
                            .with_payload(serde_json::json!({
                                "type": "analytics_computed",
                                "file_nodes": result.file_analytics.node_count,
                                "file_edges": result.file_analytics.edge_count,
                                "file_communities": result.file_analytics.communities.len(),
                                "function_nodes": result.function_analytics.node_count,
                                "function_edges": result.function_analytics.edge_count,
                                "computation_ms": elapsed.as_millis() as u64,
                            })),
                        );
                    }
                    if let Err(e) = neo4j.update_project_analytics_timestamp(project_id).await {
                        tracing::warn!(
                            "Failed to update analytics_computed_at for project {}: {}",
                            project_id,
                            e
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Background analytics computation failed for project {}: {}",
                        project_id,
                        e
                    );
                }
            }
        });
    }

    /// Compute graph analytics for a project (synchronous, best-effort).
    ///
    /// Runs the full analytics pipeline (PageRank, Betweenness, Louvain, etc.)
    /// and persists scores back to Neo4j. Logs timing on success, warns on failure.
    /// Errors are caught and logged — analytics failures never break the sync pipeline.
    pub async fn analyze_project_safe(&self, project_id: Uuid) {
        let start = std::time::Instant::now();
        match self.analytics.analyze_project(project_id).await {
            Ok(result) => {
                let elapsed = start.elapsed();
                tracing::info!(
                    "Analytics computed for project {} in {:?} (files: {} nodes/{} edges, functions: {} nodes/{} edges)",
                    project_id,
                    elapsed,
                    result.file_analytics.node_count,
                    result.file_analytics.edge_count,
                    result.function_analytics.node_count,
                    result.function_analytics.edge_count,
                );
                // Emit event so frontend/other instances know analytics were refreshed
                self.emit(
                    CrudEvent::new(
                        EventEntityType::Project,
                        CrudAction::Updated,
                        project_id.to_string(),
                    )
                    .with_payload(serde_json::json!({
                        "type": "analytics_computed",
                        "file_nodes": result.file_analytics.node_count,
                        "file_edges": result.file_analytics.edge_count,
                        "file_communities": result.file_analytics.communities.len(),
                        "function_nodes": result.function_analytics.node_count,
                        "function_edges": result.function_analytics.edge_count,
                        "computation_ms": elapsed.as_millis() as u64,
                    })),
                );
                // Update the project's analytics_computed_at timestamp
                if let Err(e) = self
                    .neo4j()
                    .update_project_analytics_timestamp(project_id)
                    .await
                {
                    tracing::warn!(
                        "Failed to update analytics_computed_at for project {}: {}",
                        project_id,
                        e
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    "Analytics computation failed for project {}: {}",
                    project_id,
                    e
                );
            }
        }
    }

    /// Check whether analytics for a project are stale and need recomputation.
    ///
    /// A project's analytics are considered stale if:
    /// - `analytics_computed_at` is `None` (never computed)
    /// - `last_synced > analytics_computed_at` (code was synced since last analytics)
    ///
    /// Returns a `StalenessReport` with details about the staleness.
    pub async fn check_analytics_staleness(&self, project_id: Uuid) -> Result<StalenessReport> {
        let project = self
            .neo4j()
            .get_project(project_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Project {} not found", project_id))?;

        let analytics_computed_at = project.analytics_computed_at;
        let last_synced = project.last_synced;

        let is_stale = match (analytics_computed_at, last_synced) {
            // Never computed → stale
            (None, _) => true,
            // Computed but never synced → not stale (edge case, analytics exist but no code)
            (Some(_), None) => false,
            // Both exist → stale if synced after analytics
            (Some(analytics), Some(synced)) => synced > analytics,
        };

        let analytics_age = analytics_computed_at.map(|at| {
            let now = chrono::Utc::now();
            (now - at).to_std().unwrap_or_default()
        });

        Ok(StalenessReport {
            project_id,
            is_stale,
            analytics_age,
            analytics_computed_at,
            last_synced,
        })
    }

    /// Check staleness for all projects and return a list of stale project IDs.
    pub async fn get_stale_projects(&self) -> Result<Vec<StalenessReport>> {
        let projects = self.neo4j().list_projects().await?;
        let mut stale = Vec::new();
        for project in &projects {
            let report = self.check_analytics_staleness(project.id).await?;
            if report.is_stale {
                stale.push(report);
            }
        }
        Ok(stale)
    }

    /// Spawn a background task that refreshes all auto-built feature graphs
    /// for a project, or auto-generates them if none exist.
    /// Best-effort: logs info on success, warn on failure.
    /// Does not block the caller.
    pub fn spawn_refresh_feature_graphs(&self, project_id: uuid::Uuid) {
        let neo4j = self.neo4j_arc();
        tokio::spawn(async move {
            let graphs = match neo4j.list_feature_graphs(Some(project_id)).await {
                Ok(g) => g,
                Err(e) => {
                    tracing::warn!(
                        "Failed to list feature graphs for refresh (project {}): {}",
                        project_id,
                        e
                    );
                    return;
                }
            };

            let auto_built: Vec<_> = graphs
                .iter()
                .filter(|g| g.entry_function.is_some())
                .collect();

            // --- Auto-generate if no feature graphs exist at all ---
            if graphs.is_empty() {
                tracing::info!(
                    "No feature graphs found for project {} — attempting LLM-based proposal",
                    project_id
                );

                // Try LLM-based proposal first
                let proposals =
                    Orchestrator::propose_feature_graphs_with_llm(neo4j.as_ref(), project_id).await;

                if !proposals.is_empty() {
                    tracing::info!(
                        "LLM proposed {} feature graph(s) for project {} — creating them",
                        proposals.len(),
                        project_id
                    );

                    for proposal in &proposals {
                        let include_rels = proposal.include_relations.as_deref();
                        match neo4j
                            .auto_build_feature_graph(
                                &proposal.name,
                                proposal.description.as_deref(),
                                project_id,
                                &proposal.entry_function,
                                proposal.depth,
                                include_rels,
                                None,
                            )
                            .await
                        {
                            Ok(detail) => {
                                tracing::info!(
                                    "Created LLM-proposed feature graph '{}' ({}) with {} entities",
                                    detail.graph.name,
                                    detail.graph.id,
                                    detail.entities.len()
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to create LLM-proposed feature graph '{}' (entry: '{}'): {}",
                                    proposal.name,
                                    proposal.entry_function,
                                    e
                                );
                            }
                        }
                    }
                    return;
                }

                // Fallback: LLM returned empty — use naive top-functions approach
                tracing::info!(
                    "LLM proposal returned empty for project {} — falling back to top entry functions",
                    project_id
                );

                let top_functions = match neo4j.get_top_entry_functions(project_id, 10).await {
                    Ok(fns) => fns,
                    Err(e) => {
                        tracing::warn!(
                            "Failed to get top entry functions for project {}: {}",
                            project_id,
                            e
                        );
                        return;
                    }
                };

                if top_functions.is_empty() {
                    tracing::info!(
                        "No connected functions found for project {} — skipping auto-generation",
                        project_id
                    );
                    return;
                }

                tracing::info!(
                    "Fallback: auto-generating {} feature graph(s) for project {}",
                    top_functions.len(),
                    project_id
                );

                for func_name in &top_functions {
                    match neo4j
                        .auto_build_feature_graph(
                            func_name, None, project_id, func_name, 2, None, None,
                        )
                        .await
                    {
                        Ok(detail) => {
                            tracing::info!(
                                "Auto-generated feature graph '{}' ({}) with {} entities",
                                detail.graph.name,
                                detail.graph.id,
                                detail.entities.len()
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to auto-generate feature graph for '{}': {}",
                                func_name,
                                e
                            );
                        }
                    }
                }
                return;
            }

            // --- Refresh existing auto-built feature graphs ---
            if auto_built.is_empty() {
                return;
            }

            tracing::info!(
                "Refreshing {} auto-built feature graph(s) for project {}",
                auto_built.len(),
                project_id
            );

            for fg in &auto_built {
                match neo4j.refresh_feature_graph(fg.id).await {
                    Ok(Some(_)) => {
                        tracing::info!("Refreshed feature graph '{}' ({})", fg.name, fg.id);
                    }
                    Ok(None) => {
                        // Should not happen since we filtered, but handle gracefully
                        tracing::debug!(
                            "Feature graph '{}' ({}) skipped (no entry_function)",
                            fg.name,
                            fg.id
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to refresh feature graph '{}' ({}): {}",
                            fg.name,
                            fg.id,
                            e
                        );
                    }
                }
            }
        });
    }

    /// Gather codebase context for LLM-based feature graph proposal.
    ///
    /// Collects top functions, module structure, and existing feature graphs
    /// to build a JSON context that the LLM can analyze.
    async fn gather_codebase_context(
        neo4j: &dyn crate::neo4j::GraphStore,
        project_id: Uuid,
    ) -> Result<String> {
        // Top 30 functions by connectivity
        let top_functions = neo4j.get_top_entry_functions(project_id, 30).await?;

        // Module structure: most connected files
        let connected_files = neo4j
            .get_most_connected_files_for_project(project_id, 20)
            .await
            .unwrap_or_default();

        // Language stats
        let lang_stats = neo4j
            .get_language_stats_for_project(project_id)
            .await
            .unwrap_or_default();

        // Existing feature graphs (to avoid duplicates)
        let existing_fgs = neo4j
            .list_feature_graphs(Some(project_id))
            .await
            .unwrap_or_default();

        let existing_fg_json: Vec<serde_json::Value> = existing_fgs
            .iter()
            .map(|fg| {
                serde_json::json!({
                    "name": fg.name,
                    "entry_function": fg.entry_function,
                    "entity_count": fg.entity_count,
                })
            })
            .collect();

        let files_json: Vec<serde_json::Value> = connected_files
            .iter()
            .map(|f| {
                serde_json::json!({
                    "path": f.path,
                    "imports": f.imports,
                    "dependents": f.dependents,
                })
            })
            .collect();

        let langs_json: Vec<serde_json::Value> = lang_stats
            .iter()
            .map(|l| {
                serde_json::json!({
                    "language": l.language,
                    "file_count": l.file_count,
                })
            })
            .collect();

        let context = serde_json::json!({
            "top_functions": top_functions,
            "connected_files": files_json,
            "language_stats": langs_json,
            "existing_feature_graphs": existing_fg_json,
        });

        Ok(context.to_string())
    }

    /// Build the prompt that asks the LLM to propose feature graphs.
    fn build_feature_graph_prompt(codebase_context: &str) -> String {
        format!(
            r#"You are analyzing a codebase to identify meaningful feature graphs.

A **feature graph** maps a coherent feature or subsystem by starting from an entry-point function and traversing its call graph (callers + callees), related types (structs/enums via impl blocks), traits, and imports.

## Your task

Analyze the codebase context below and propose **5 to 15 feature graphs** that cover the main features/subsystems of this project. Each proposal needs:
- **name**: A short, human-readable name (e.g. "Code Sync Pipeline", "Authentication Flow", "MCP Tool Dispatch")
- **description**: One sentence explaining what this feature covers
- **entry_function**: The function name to start BFS traversal from (must be one of the top_functions listed)
- **depth**: BFS depth 1-5 (use 2-3 for focused features, 4-5 for broad subsystems)
- **include_relations**: Optional array of relation types to traverse. Options: "calls", "implements_trait", "implements_for", "imports". Omit or set to null for all relations.

## Guidelines
- Choose entry points that represent the **core logic** of each feature, not utility functions
- Prefer functions that are **high in the call hierarchy** (many callees) over leaf functions
- Give names that a developer would use to describe the feature (not just the function name)
- Avoid duplicating existing feature graphs (listed below)
- Group related functions into the same feature graph via the right entry point
- Cover as much of the codebase as possible without excessive overlap

## Codebase context

{codebase_context}

## Response format

Respond with ONLY a JSON array, no markdown fences, no explanation:
[
  {{
    "name": "Feature Name",
    "description": "What this feature does",
    "entry_function": "function_name",
    "depth": 2,
    "include_relations": null
  }}
]"#,
            codebase_context = codebase_context
        )
    }

    /// Use a oneshot LLM call to propose feature graphs for a project.
    ///
    /// Follows the same pattern as `ChatManager::refine_context_with_oneshot`:
    /// InteractiveClient + max_turns(1) + BypassPermissions.
    ///
    /// Returns an empty Vec on any failure (best-effort).
    async fn propose_feature_graphs_with_llm(
        neo4j: &dyn crate::neo4j::GraphStore,
        project_id: Uuid,
    ) -> Vec<FeatureGraphProposal> {
        // 1. Gather context
        let context = match Self::gather_codebase_context(neo4j, project_id).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    "Failed to gather codebase context for LLM feature graph proposal: {}",
                    e
                );
                return Vec::new();
            }
        };

        // 2. Build prompt
        let prompt = Self::build_feature_graph_prompt(&context);

        // 3. Oneshot LLM call
        let model = std::env::var("FEATURE_GRAPH_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-20250514".into());

        if model.is_empty() {
            tracing::info!("FEATURE_GRAPH_MODEL is empty, skipping LLM feature graph proposal");
            return Vec::new();
        }

        tracing::info!(
            "Proposing feature graphs via LLM (model: {}) for project {}",
            model,
            project_id
        );

        use nexus_claude::{ClaudeCodeOptions, InteractiveClient, PermissionMode};

        #[allow(deprecated)]
        let options = ClaudeCodeOptions::builder()
            .model(&model)
            .system_prompt("You are a code architecture analyst. Respond only with valid JSON.")
            .permission_mode(PermissionMode::BypassPermissions)
            .max_turns(1)
            .build();

        let mut client = match InteractiveClient::new(options) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    "Failed to create LLM client for feature graph proposal: {}",
                    e
                );
                return Vec::new();
            }
        };

        if let Err(e) = client.connect().await {
            tracing::warn!(
                "Failed to connect LLM client for feature graph proposal: {}",
                e
            );
            return Vec::new();
        }

        let messages = match client.send_and_receive(prompt).await {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("LLM feature graph proposal call failed: {}", e);
                let _ = client.disconnect().await;
                return Vec::new();
            }
        };

        let _ = client.disconnect().await;

        // 4. Extract text from response
        use nexus_claude::{ContentBlock, Message};
        let mut response_text = String::new();
        for msg in &messages {
            if let Message::Assistant { message, .. } = msg {
                for block in &message.content {
                    if let ContentBlock::Text(text) = block {
                        response_text.push_str(&text.text);
                    }
                }
            }
        }

        if response_text.is_empty() {
            tracing::warn!("LLM returned empty response for feature graph proposal");
            return Vec::new();
        }

        // 5. Parse JSON — try to extract array from response
        //    The LLM might wrap it in markdown fences, so strip those
        let json_str = response_text
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        match serde_json::from_str::<Vec<FeatureGraphProposal>>(json_str) {
            Ok(proposals) => {
                tracing::info!(
                    "LLM proposed {} feature graph(s) for project {}",
                    proposals.len(),
                    project_id
                );
                proposals
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to parse LLM feature graph proposals: {}. Response: {}",
                    e,
                    &response_text[..response_text.len().min(500)]
                );
                Vec::new()
            }
        }
    }

    /// Get the search store
    pub fn meili(&self) -> &dyn crate::meilisearch::SearchStore {
        self.state.meili.as_ref()
    }

    /// Get the search store as Arc (for sharing with ChatManager etc.)
    pub fn meili_arc(&self) -> Arc<dyn crate::meilisearch::SearchStore> {
        self.state.meili.clone()
    }

    /// Get the note manager
    pub fn note_manager(&self) -> &Arc<NoteManager> {
        &self.note_manager
    }

    /// Get the note lifecycle manager
    pub fn note_lifecycle(&self) -> &Arc<NoteLifecycleManager> {
        &self.note_lifecycle
    }

    /// Get the implementation planner
    pub fn planner(&self) -> &Arc<super::ImplementationPlanner> {
        &self.planner
    }

    /// Get the graph analytics engine
    pub fn analytics(&self) -> &Arc<dyn AnalyticsEngine> {
        &self.analytics
    }

    /// Get the analytics debouncer (for incremental sync triggers)
    pub fn analytics_debouncer(&self) -> &AnalyticsDebouncer {
        &self.analytics_debouncer
    }

    /// Get the spreading activation engine (if embedding provider is available).
    ///
    /// Returns `None` when `EMBEDDING_PROVIDER=disabled` or initialization failed.
    pub fn activation_engine(&self) -> Option<&Arc<SpreadingActivationEngine>> {
        self.activation_engine.as_ref()
    }

    /// Get the auto-reinforcement configuration.
    pub fn auto_reinforcement_config(&self) -> &AutoReinforcementConfig {
        &self.auto_reinforcement
    }

    // ========================================================================
    // Sync operations
    // ========================================================================

    /// Sync a directory to the knowledge base (legacy, no project)
    pub async fn sync_directory(&self, dir_path: &Path) -> Result<SyncResult> {
        self.sync_directory_for_project(dir_path, None, None).await
    }

    /// Sync a directory to the knowledge base for a specific project
    pub async fn sync_directory_for_project(
        &self,
        dir_path: &Path,
        project_id: Option<Uuid>,
        project_slug: Option<&str>,
    ) -> Result<SyncResult> {
        self.sync_directory_for_project_with_options(dir_path, project_id, project_slug, false)
            .await
    }

    /// Sync a directory to the knowledge base for a specific project, with options
    pub async fn sync_directory_for_project_with_options(
        &self,
        dir_path: &Path,
        project_id: Option<Uuid>,
        project_slug: Option<&str>,
        force: bool,
    ) -> Result<SyncResult> {
        let project_slug = project_slug.map(|s| s.to_string());
        let mut result = SyncResult::default();
        let mut synced_paths: HashSet<String> = HashSet::new();

        // All supported languages - must match SupportedLanguage::from_extension()
        let extensions = [
            "rs", // Rust
            "ts", "tsx", "js", "jsx",  // TypeScript/JavaScript
            "py",   // Python
            "go",   // Go
            "java", // Java
            "c", "h", // C
            "cpp", "cc", "cxx", "hpp", "hxx", // C++
            "rb",  // Ruby
            "php", // PHP
            "kt", "kts",   // Kotlin
            "swift", // Swift
            "sh", "bash", // Bash
        ];

        for entry in WalkDir::new(dir_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default();

            if !extensions.contains(&ext) {
                continue;
            }

            // Skip ignored directories (shared constant with watcher.rs)
            let path_str = path.to_string_lossy();
            if super::should_ignore_path(&path_str) {
                continue;
            }

            // Track the path for cleanup
            synced_paths.insert(path_str.to_string());

            match self
                .sync_file_for_project_with_options(
                    path,
                    project_id,
                    project_slug.as_deref(),
                    force,
                )
                .await
            {
                Ok(synced) => {
                    if synced {
                        result.files_synced += 1;
                    } else {
                        result.files_skipped += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to sync {}: {}", path.display(), e);
                    result.errors += 1;
                }
            }
        }

        // Clean up stale files if we have a project_id
        if let Some(pid) = project_id {
            let valid_paths: Vec<String> = synced_paths.into_iter().collect();
            match self.neo4j().delete_stale_files(pid, &valid_paths).await {
                Ok((files_deleted, symbols_deleted)) => {
                    result.files_deleted = files_deleted;
                    result.symbols_deleted = symbols_deleted;
                }
                Err(e) => {
                    tracing::warn!("Failed to clean up stale files: {}", e);
                }
            }
        }

        Ok(result)
    }

    /// Sync a single file to the knowledge base (legacy, no project)
    pub async fn sync_file(&self, path: &Path) -> Result<bool> {
        self.sync_file_for_project(path, None, None).await
    }

    /// Sync a single file to the knowledge base for a specific project
    pub async fn sync_file_for_project(
        &self,
        path: &Path,
        project_id: Option<Uuid>,
        project_slug: Option<&str>,
    ) -> Result<bool> {
        self.sync_file_for_project_with_options(path, project_id, project_slug, false)
            .await
    }

    /// Sync a single file with force option (skip hash check when force=true)
    pub async fn sync_file_for_project_with_options(
        &self,
        path: &Path,
        project_id: Option<Uuid>,
        project_slug: Option<&str>,
        force: bool,
    ) -> Result<bool> {
        let content = tokio::fs::read_to_string(path)
            .await
            .context("Failed to read file")?;

        // Normalize path to absolute form for consistent Neo4j storage
        let path_str = normalize_path(&path.to_string_lossy());
        if !force {
            if let Some(existing) = self.state.neo4j.get_file(&path_str).await? {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(content.as_bytes());
                let hash = hex::encode(hasher.finalize());

                if existing.hash == hash {
                    return Ok(false); // File unchanged
                }
            }
        }

        // Parse the file using normalized path
        let norm_path = std::path::Path::new(&path_str);
        let parsed = {
            let mut parser = self.parser.write().await;
            parser.parse_file(norm_path, &content)?
        };

        // Store in Neo4j with project association
        self.store_parsed_file_for_project(&parsed, project_id)
            .await?;

        // Index in Meilisearch only if project context is available
        if let (Some(pid), Some(slug)) = (project_id, project_slug) {
            let doc = CodeParser::to_code_document(&parsed, &pid.to_string(), slug);
            self.state.meili.index_code(&doc).await?;
        }

        // Verify notes attached to this file
        self.verify_notes_for_file(&path_str, &parsed, &content)
            .await?;

        Ok(true)
    }

    /// Verify notes attached to a file after it has been modified
    ///
    /// This checks if any notes anchored to entities in this file need
    /// status updates due to code changes.
    async fn verify_notes_for_file(
        &self,
        file_path: &str,
        parsed: &ParsedFile,
        source: &str,
    ) -> Result<()> {
        // Get all notes attached to this file
        let notes = self
            .state
            .neo4j
            .get_notes_for_entity(&EntityType::File, file_path)
            .await?;

        if notes.is_empty() {
            return Ok(());
        }

        tracing::debug!("Verifying {} notes for file: {}", notes.len(), file_path);

        // Create FileInfo from parsed data
        let file_info = NoteLifecycleManager::create_file_info(parsed, source);

        // Verify each note's anchors
        let results = self
            .note_lifecycle
            .verify_notes_for_file(&notes, &file_info);

        // Process verification results
        for result in results {
            if !result.all_valid {
                if let Some(update) = result.suggested_update {
                    // Update note status
                    self.state
                        .neo4j
                        .update_note(
                            result.note_id,
                            None,
                            None,
                            Some(update.new_status),
                            None,
                            None,
                        )
                        .await?;

                    // Update Meilisearch index
                    self.state
                        .meili
                        .update_note_status(
                            &result.note_id.to_string(),
                            &update.new_status.to_string(),
                        )
                        .await?;

                    tracing::info!(
                        "Note {} status changed to {:?}: {}",
                        result.note_id,
                        update.new_status,
                        update.reason
                    );
                }
            }
        }

        // Also verify notes attached to functions/structs in this file
        self.verify_notes_for_file_symbols(file_path, &file_info)
            .await?;

        // Verify assertion notes (pass already-fetched notes to avoid duplicate query)
        self.verify_assertions_for_file(file_path, &file_info, &notes)
            .await?;

        Ok(())
    }

    /// Verify assertion notes that apply to a file
    async fn verify_assertions_for_file(
        &self,
        file_path: &str,
        file_info: &crate::notes::FileInfo,
        file_notes: &[crate::notes::Note],
    ) -> Result<()> {
        use crate::notes::{NoteStatus, NoteType, ViolationAction};

        // Filter assertion notes from the already-fetched file notes
        let assertion_notes: Vec<_> = file_notes
            .iter()
            .filter(|n| n.note_type == NoteType::Assertion)
            .cloned()
            .collect();

        if assertion_notes.is_empty() {
            return Ok(());
        }

        tracing::debug!(
            "Verifying {} assertion notes for file: {}",
            assertion_notes.len(),
            file_path
        );

        // Verify each assertion
        let results = self
            .note_lifecycle
            .verify_assertions_for_file(&assertion_notes, file_info);

        for result in results {
            if !result.passed {
                // Find the note to get the violation action
                let note = assertion_notes.iter().find(|n| n.id == result.note_id);

                if let Some(note) = note {
                    if let Some(ref rule) = note.assertion_rule {
                        match rule.on_violation {
                            ViolationAction::Warn => {
                                tracing::warn!(
                                    "Assertion failed (warning): note {} - {}",
                                    result.note_id,
                                    result.message
                                );
                            }
                            ViolationAction::FlagNote | ViolationAction::Block => {
                                // Update note status to needs_review
                                self.state
                                    .neo4j
                                    .update_note(
                                        result.note_id,
                                        None,
                                        None,
                                        Some(NoteStatus::NeedsReview),
                                        None,
                                        None,
                                    )
                                    .await?;

                                self.state
                                    .meili
                                    .update_note_status(&result.note_id.to_string(), "needs_review")
                                    .await?;

                                tracing::warn!(
                                    "Assertion failed: note {} flagged for review - {}",
                                    result.note_id,
                                    result.message
                                );
                            }
                        }
                    }
                }
            } else {
                tracing::debug!(
                    "Assertion passed: note {} - {}",
                    result.note_id,
                    result.message
                );
            }
        }

        Ok(())
    }

    /// Verify notes attached to symbols (functions, structs) within a file
    async fn verify_notes_for_file_symbols(
        &self,
        file_path: &str,
        file_info: &crate::notes::FileInfo,
    ) -> Result<()> {
        // Verify notes attached to functions
        for func in &file_info.functions {
            let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
            let notes = self
                .state
                .neo4j
                .get_notes_for_entity(&EntityType::Function, &func_id)
                .await?;

            if notes.is_empty() {
                continue;
            }

            let results = self.note_lifecycle.verify_notes_for_file(&notes, file_info);

            for result in results {
                if !result.all_valid {
                    if let Some(update) = result.suggested_update {
                        self.state
                            .neo4j
                            .update_note(
                                result.note_id,
                                None,
                                None,
                                Some(update.new_status),
                                None,
                                None,
                            )
                            .await?;

                        self.state
                            .meili
                            .update_note_status(
                                &result.note_id.to_string(),
                                &update.new_status.to_string(),
                            )
                            .await?;

                        tracing::info!(
                            "Note {} (on {}) status changed to {:?}: {}",
                            result.note_id,
                            func.name,
                            update.new_status,
                            update.reason
                        );
                    }
                }
            }
        }

        // Verify notes attached to structs
        for s in &file_info.structs {
            let struct_id = format!("{}:{}", file_path, s.name);
            let notes = self
                .state
                .neo4j
                .get_notes_for_entity(&EntityType::Struct, &struct_id)
                .await?;

            if notes.is_empty() {
                continue;
            }

            let results = self.note_lifecycle.verify_notes_for_file(&notes, file_info);

            for result in results {
                if !result.all_valid {
                    if let Some(update) = result.suggested_update {
                        self.state
                            .neo4j
                            .update_note(
                                result.note_id,
                                None,
                                None,
                                Some(update.new_status),
                                None,
                                None,
                            )
                            .await?;

                        self.state
                            .meili
                            .update_note_status(
                                &result.note_id.to_string(),
                                &update.new_status.to_string(),
                            )
                            .await?;

                        tracing::info!(
                            "Note {} (on {}) status changed to {:?}: {}",
                            result.note_id,
                            s.name,
                            update.new_status,
                            update.reason
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Store a parsed file in Neo4j with project association
    async fn store_parsed_file_for_project(
        &self,
        parsed: &ParsedFile,
        project_id: Option<Uuid>,
    ) -> Result<()> {
        // Store file node (path already normalized in sync_file_for_project)
        let file_node = FileNode {
            path: normalize_path(&parsed.path),
            language: parsed.language.clone(),
            hash: parsed.hash.clone(),
            last_parsed: chrono::Utc::now(),
            project_id,
        };
        self.state.neo4j.upsert_file(&file_node).await?;

        // Store functions
        for func in &parsed.functions {
            self.state.neo4j.upsert_function(func).await?;
        }

        // Store structs
        for s in &parsed.structs {
            self.state.neo4j.upsert_struct(s).await?;
        }

        // Store traits
        for t in &parsed.traits {
            self.state.neo4j.upsert_trait(t).await?;
        }

        // Store enums
        for e in &parsed.enums {
            self.state.neo4j.upsert_enum(e).await?;
        }

        // Store impl blocks with relationships
        for impl_block in &parsed.impl_blocks {
            self.state.neo4j.upsert_impl(impl_block).await?;
        }

        // Store imports and create File→IMPORTS→File + IMPORTS_SYMBOL relationships
        for import in &parsed.imports {
            self.state.neo4j.upsert_import(import).await?;

            // Resolve imports to file paths (language-aware)
            let resolved_files =
                self.resolve_imports_for_language(import, &parsed.path, &parsed.language);
            for target_file in &resolved_files {
                self.state
                    .neo4j
                    .create_import_relationship(&parsed.path, target_file, &import.path)
                    .await
                    .ok(); // Ignore errors (target file might not exist yet)
            }

            // Create IMPORTS_SYMBOL relationships for imported symbols
            let import_id = format!("{}:{}:{}", import.file_path, import.line, import.path);
            let symbols = Self::extract_imported_symbols(import);
            for symbol_name in &symbols {
                self.state
                    .neo4j
                    .create_imports_symbol_relationship(&import_id, symbol_name, project_id)
                    .await
                    .ok(); // Ignore errors (symbol might not exist yet)
            }
        }

        // Store function call relationships (scoped to project to prevent cross-project pollution)
        for call in &parsed.function_calls {
            self.state
                .neo4j
                .create_call_relationship(&call.caller_id, &call.callee_name, project_id)
                .await?;
        }

        // Embed file and functions (best-effort, non-blocking)
        if self.embedding_provider.is_some() {
            let provider = self.embedding_provider.clone().unwrap();
            let neo4j = self.state.neo4j.clone();
            let parsed_clone = parsed.clone();
            let file_path = normalize_path(&parsed.path);
            tokio::spawn(async move {
                if let Err(e) = Self::embed_parsed_file(&provider, &neo4j, &parsed_clone, &file_path).await {
                    tracing::warn!(file = %file_path, error = %e, "Failed to embed file (best-effort)");
                }
            });
        }

        Ok(())
    }

    /// Build a summary text for embedding a file.
    ///
    /// Format: "path (language) — symbols: fn1, fn2, struct1, trait1..."
    /// This captures the semantic identity of the file without its full content.
    fn build_file_embedding_text(parsed: &ParsedFile) -> String {
        let mut symbols = Vec::new();

        for func in &parsed.functions {
            symbols.push(format!("fn {}", func.name));
        }
        for s in &parsed.structs {
            symbols.push(format!("struct {}", s.name));
        }
        for t in &parsed.traits {
            symbols.push(format!("trait {}", t.name));
        }
        for e in &parsed.enums {
            symbols.push(format!("enum {}", e.name));
        }

        let symbols_str = if symbols.is_empty() {
            String::new()
        } else {
            format!(" — symbols: {}", symbols.join(", "))
        };

        format!("{} ({}){}", parsed.path, parsed.language, symbols_str)
    }

    /// Build a summary text for embedding a function.
    ///
    /// Format: "name(param1: Type, param2: Type) -> ReturnType — docstring"
    fn build_function_embedding_text(func: &crate::neo4j::models::FunctionNode) -> String {
        let params = func
            .params
            .iter()
            .map(|p| {
                if let Some(ref ty) = p.type_name {
                    format!("{}: {}", p.name, ty)
                } else {
                    p.name.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        let ret = func
            .return_type
            .as_ref()
            .map(|r| format!(" -> {}", r))
            .unwrap_or_default();

        let doc = func
            .docstring
            .as_ref()
            .map(|d| format!(" — {}", d))
            .unwrap_or_default();

        format!("{}({}){}{}", func.name, params, ret, doc)
    }

    /// Embed a parsed file and its public functions into Neo4j.
    ///
    /// Best-effort: logs warnings on failure but never returns errors that block sync.
    /// Uses embed_batch for efficiency (one API call for file + all functions).
    async fn embed_parsed_file(
        provider: &Arc<dyn EmbeddingProvider>,
        neo4j: &Arc<dyn crate::neo4j::GraphStore>,
        parsed: &ParsedFile,
        file_path: &str,
    ) -> Result<()> {
        // Build texts to embed
        let file_text = Self::build_file_embedding_text(parsed);

        // Collect functions that are worth embedding (public or have docstrings)
        let embeddable_functions: Vec<&crate::neo4j::models::FunctionNode> = parsed
            .functions
            .iter()
            .filter(|f| {
                f.docstring.is_some()
                    || matches!(
                        f.visibility,
                        crate::neo4j::models::Visibility::Public
                    )
            })
            .collect();

        let func_texts: Vec<String> = embeddable_functions
            .iter()
            .map(|f| Self::build_function_embedding_text(f))
            .collect();

        // Build batch: file text first, then function texts
        let mut all_texts = vec![file_text];
        all_texts.extend(func_texts.clone());

        // Single batch embed call
        let embeddings = provider.embed_batch(&all_texts).await?;

        if embeddings.is_empty() {
            return Ok(());
        }

        let model = provider.model_name().to_string();

        // Store file embedding (first result)
        neo4j
            .set_file_embedding(file_path, &embeddings[0], &model)
            .await?;

        // Store function embeddings (remaining results)
        for (i, func) in embeddable_functions.iter().enumerate() {
            if let Some(emb) = embeddings.get(i + 1) {
                if let Err(e) = neo4j
                    .set_function_embedding(&func.name, file_path, emb, &model)
                    .await
                {
                    tracing::warn!(
                        function = %func.name,
                        file = %file_path,
                        error = %e,
                        "Failed to store function embedding"
                    );
                }
            }
        }

        tracing::debug!(
            file = %file_path,
            functions = embeddable_functions.len(),
            "Embedded file and functions"
        );

        Ok(())
    }

    /// Extract imported symbol names from an ImportNode.
    ///
    /// Sources:
    /// - `items` field (populated by TS/Python parsers)
    /// - Rust import path terminal segment (e.g., `crate::types::Message` → `Message`)
    /// - Rust grouped imports (e.g., `{Result, SdkError}` → `["Result", "SdkError"]`)
    fn extract_imported_symbols(import: &ImportNode) -> Vec<String> {
        let mut symbols = Vec::new();

        // Use items field if populated (TS/Python parsers)
        if !import.items.is_empty() {
            for item in &import.items {
                let name = item.trim();
                if !name.is_empty()
                    && name
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                {
                    symbols.push(name.to_string());
                }
            }
        }

        // For Rust, extract from the path
        let path = &import.path;
        if let Some(brace_start) = path.rfind('{') {
            // Grouped: extract names from {A, B, C}
            if let Some(brace_end) = path.rfind('}') {
                let inner = &path[brace_start + 1..brace_end];
                for item in inner.split(',') {
                    let name = item.trim();
                    // Only include type-like names (start with uppercase)
                    if !name.is_empty()
                        && name
                            .chars()
                            .next()
                            .map(|c| c.is_uppercase())
                            .unwrap_or(false)
                    {
                        // Handle nested like `errors::{Result}` — take the last segment
                        let final_name = name.rsplit("::").next().unwrap_or(name).trim();
                        if !final_name.is_empty() && !symbols.contains(&final_name.to_string()) {
                            symbols.push(final_name.to_string());
                        }
                    }
                }
            }
        } else {
            // Flat: take the last segment if it looks like a type
            let last = path.rsplit("::").next().unwrap_or(path);
            if !last.is_empty()
                && last
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
                && !symbols.contains(&last.to_string())
            {
                symbols.push(last.to_string());
            }
        }

        symbols
    }

    /// Flatten grouped Rust imports into individual module paths.
    ///
    /// Examples:
    /// - `crate::{errors::{Result}, transport::Transport}` → `["crate::errors", "crate::transport"]`
    /// - `crate::neo4j::client` → `["crate::neo4j::client"]` (unchanged)
    fn flatten_grouped_import(import_path: &str) -> Vec<String> {
        let trimmed = import_path.trim();

        // If no braces, return as-is
        if !trimmed.contains('{') {
            return vec![trimmed.to_string()];
        }

        // Find the prefix before the first `{`
        let brace_pos = match trimmed.find('{') {
            Some(p) => p,
            None => return vec![trimmed.to_string()],
        };

        let prefix = trimmed[..brace_pos].trim_end_matches("::");

        // Extract content inside the outermost braces
        let inner = &trimmed[brace_pos + 1..];
        let inner = inner.trim();
        // Remove trailing }
        let inner = if let Some(stripped) = inner.strip_suffix('}') {
            stripped
        } else {
            inner
        };

        let mut results = Vec::new();
        let mut depth = 0;
        let mut current = String::new();

        for ch in inner.chars() {
            match ch {
                '{' => {
                    depth += 1;
                    current.push(ch);
                }
                '}' => {
                    depth -= 1;
                    current.push(ch);
                }
                ',' if depth == 0 => {
                    let item = current.trim().to_string();
                    if !item.is_empty() {
                        let full = format!("{}::{}", prefix, item);
                        results.extend(Self::flatten_grouped_import(&full));
                    }
                    current.clear();
                }
                _ => {
                    current.push(ch);
                }
            }
        }

        // Handle last item
        let item = current.trim().to_string();
        if !item.is_empty() {
            let full = format!("{}::{}", prefix, item);
            results.extend(Self::flatten_grouped_import(&full));
        }

        results
    }

    /// Resolve a Rust import path to actual file paths.
    /// Returns multiple paths for grouped imports like `crate::{errors, transport}`.
    ///
    /// Examples:
    /// - `crate::neo4j::client` → `["src/neo4j/client.rs"]`
    /// - `crate::{errors::{Result}, transport::Transport}` → `["src/errors.rs", "src/transport.rs"]`
    /// - External crates (std::, serde::) → `[]`
    fn resolve_rust_imports(&self, import_path: &str, source_file: &str) -> Vec<String> {
        let paths = Self::flatten_grouped_import(import_path);
        let mut resolved = Vec::new();

        for path in paths {
            if let Some(file) = self.resolve_single_rust_import(&path, source_file) {
                if !resolved.contains(&file) {
                    resolved.push(file);
                }
            }
        }

        resolved
    }

    /// Resolve a single (non-grouped) Rust import path to a file path
    fn resolve_single_rust_import(&self, import_path: &str, source_file: &str) -> Option<String> {
        let path: Vec<&str> = import_path.split("::").collect();
        if path.is_empty() {
            return None;
        }

        let first = path[0];

        // External crates - skip
        if !matches!(first, "crate" | "super" | "self") {
            return None;
        }

        // Get the source file's directory
        let source_path = Path::new(source_file);
        let source_dir = source_path.parent()?;

        // Find the project root (where Cargo.toml is)
        let mut project_root = source_dir;
        while !project_root.join("Cargo.toml").exists() {
            project_root = project_root.parent()?;
            if project_root.as_os_str().is_empty() {
                return None;
            }
        }

        let src_dir = project_root.join("src");

        // Build the target path based on import type
        let target_path = match first {
            "crate" => {
                if path.len() < 2 {
                    return None;
                }
                let module_path = &path[1..path.len().saturating_sub(1)];
                if module_path.is_empty() {
                    return None;
                }
                let mut target = src_dir.clone();
                for part in module_path {
                    target = target.join(part);
                }
                target
            }
            "super" => {
                let mut target = source_dir.to_path_buf();
                for part in &path[1..path.len().saturating_sub(1)] {
                    if *part == "super" {
                        target = target.parent()?.to_path_buf();
                    } else {
                        target = target.join(part);
                    }
                }
                target
            }
            "self" => {
                let mut target = source_dir.to_path_buf();
                for part in &path[1..path.len().saturating_sub(1)] {
                    target = target.join(part);
                }
                target
            }
            _ => return None,
        };

        // Try .rs file first, then mod.rs
        let rs_file = target_path.with_extension("rs");
        if rs_file.exists() {
            return Some(normalize_path(&rs_file.to_string_lossy()));
        }

        let mod_file = target_path.join("mod.rs");
        if mod_file.exists() {
            return Some(normalize_path(&mod_file.to_string_lossy()));
        }

        // Also try without removing the last segment (in case it's a module not a type)
        if first == "crate" {
            let module_path = &path[1..];
            let mut target = src_dir;
            for part in module_path {
                target = target.join(part);
            }

            let rs_file = target.with_extension("rs");
            if rs_file.exists() {
                return Some(normalize_path(&rs_file.to_string_lossy()));
            }

            let mod_file = target.join("mod.rs");
            if mod_file.exists() {
                return Some(normalize_path(&mod_file.to_string_lossy()));
            }
        }

        None
    }

    /// Resolve a TypeScript/JavaScript import path to a file path
    ///
    /// Handles: `./relative`, `../parent`, `@/alias` paths.
    /// Tries extensions: .ts, .tsx, .js, .jsx, then /index.ts, /index.js
    /// Skips bare specifiers (npm packages like `react`, `lodash`).
    fn resolve_typescript_import(&self, import_path: &str, source_file: &str) -> Option<String> {
        // Only resolve relative imports
        if !import_path.starts_with('.') && !import_path.starts_with('@') {
            return None; // npm package — skip
        }

        let source_path = Path::new(source_file);
        let source_dir = source_path.parent()?;

        let target = if import_path.starts_with('.') {
            source_dir.join(import_path)
        } else {
            // @/ alias — try to find project root with tsconfig
            let mut root = source_dir;
            loop {
                if root.join("tsconfig.json").exists() || root.join("package.json").exists() {
                    break;
                }
                root = root.parent()?;
                if root.as_os_str().is_empty() {
                    return None;
                }
            }
            root.join("src").join(&import_path[2..]) // @/foo → src/foo
        };

        // Try extensions in order
        let extensions = [".ts", ".tsx", ".js", ".jsx"];
        for ext in &extensions {
            let with_ext = target.with_extension(&ext[1..]);
            if with_ext.exists() {
                return Some(normalize_path(&with_ext.to_string_lossy()));
            }
        }

        // Try index files
        let index_files = ["index.ts", "index.tsx", "index.js", "index.jsx"];
        for idx in &index_files {
            let index_path = target.join(idx);
            if index_path.exists() {
                return Some(normalize_path(&index_path.to_string_lossy()));
            }
        }

        None
    }

    /// Resolve a Python import path to a file path
    ///
    /// Handles relative imports (`from . import foo`, `from ..models import Bar`)
    /// and absolute imports starting with project package name.
    /// Tries `module.py` then `module/__init__.py`.
    /// Skips stdlib and pip packages.
    fn resolve_python_import(&self, import_path: &str, source_file: &str) -> Option<String> {
        // Python imports from tree-sitter look like "." or ".models" or "..utils" or "package.module"
        if !import_path.starts_with('.') {
            // Absolute import — might be project-local or external
            // Heuristic: check if the first segment matches a directory in project root
            let source_path = Path::new(source_file);
            let source_dir = source_path.parent()?;

            // Find project root
            let mut root = source_dir;
            loop {
                if root.join("setup.py").exists()
                    || root.join("pyproject.toml").exists()
                    || root.join("setup.cfg").exists()
                {
                    break;
                }
                root = root.parent()?;
                if root.as_os_str().is_empty() {
                    return None; // No project root found — likely external
                }
            }

            let parts: Vec<&str> = import_path.split('.').collect();
            if parts.is_empty() {
                return None;
            }

            // Check if the first part is a local package
            if !root.join(parts[0]).is_dir() {
                return None; // External package
            }

            let mut target = root.to_path_buf();
            for part in &parts {
                target = target.join(part);
            }

            let py_file = target.with_extension("py");
            if py_file.exists() {
                return Some(normalize_path(&py_file.to_string_lossy()));
            }

            let init_file = target.join("__init__.py");
            if init_file.exists() {
                return Some(normalize_path(&init_file.to_string_lossy()));
            }

            return None;
        }

        // Relative import
        let source_path = Path::new(source_file);
        let source_dir = source_path.parent()?;

        // Count leading dots
        let dots = import_path.chars().take_while(|c| *c == '.').count();
        let module_part = &import_path[dots..];

        let mut target = source_dir.to_path_buf();
        for _ in 1..dots {
            target = target.parent()?.to_path_buf();
        }

        if !module_part.is_empty() {
            for part in module_part.split('.') {
                target = target.join(part);
            }
        }

        let py_file = target.with_extension("py");
        if py_file.exists() {
            return Some(normalize_path(&py_file.to_string_lossy()));
        }

        let init_file = target.join("__init__.py");
        if init_file.exists() {
            return Some(normalize_path(&init_file.to_string_lossy()));
        }

        None
    }

    /// Resolve import paths to file paths based on language
    fn resolve_imports_for_language(
        &self,
        import: &ImportNode,
        parsed_path: &str,
        language: &str,
    ) -> Vec<String> {
        match language {
            "rust" => self.resolve_rust_imports(&import.path, parsed_path),
            "typescript" | "javascript" | "tsx" | "jsx" => self
                .resolve_typescript_import(&import.path, parsed_path)
                .into_iter()
                .collect(),
            "python" => self
                .resolve_python_import(&import.path, parsed_path)
                .into_iter()
                .collect(),
            _ => Vec::new(),
        }
    }

    // ========================================================================
    // Agent dispatch
    // ========================================================================

    /// Dispatch a task to an agent
    pub async fn dispatch_task(
        &self,
        task_id: Uuid,
        plan_id: Uuid,
        agent_id: &str,
    ) -> Result<String> {
        // Mark task as in progress
        self.plan_manager
            .update_task(
                task_id,
                UpdateTaskRequest {
                    status: Some(TaskStatus::InProgress),
                    assigned_to: Some(agent_id.to_string()),
                    ..Default::default()
                },
            )
            .await?;

        // Build context
        let context = self.context_builder.build_context(task_id, plan_id).await?;

        // Generate prompt
        let prompt = self.context_builder.generate_prompt(&context);

        Ok(prompt)
    }

    /// Handle task completion from an agent
    pub async fn handle_task_completion(
        &self,
        task_id: Uuid,
        success: bool,
        summary: &str,
        files_modified: &[String],
    ) -> Result<()> {
        let status = if success {
            TaskStatus::Completed
        } else {
            TaskStatus::Failed
        };

        // Update task status
        self.plan_manager
            .update_task(
                task_id,
                UpdateTaskRequest {
                    status: Some(status),
                    ..Default::default()
                },
            )
            .await?;

        // Link modified files
        if !files_modified.is_empty() {
            self.plan_manager
                .link_task_to_files(task_id, files_modified)
                .await?;
        }

        // Re-sync modified files
        for file_path in files_modified {
            let path = Path::new(file_path);
            if path.exists() {
                if let Err(e) = self.sync_file(path).await {
                    tracing::warn!("Failed to re-sync {}: {}", file_path, e);
                }
            }
        }

        tracing::info!("Task {} completed: {}", task_id, summary);
        Ok(())
    }

    // ========================================================================
    // Orchestration loop
    // ========================================================================

    /// Run the main orchestration loop
    pub async fn run_loop(&self, plan_id: Uuid) -> Result<()> {
        loop {
            // Check for next available task
            let next_task = self.plan_manager.get_next_available_task(plan_id).await?;

            match next_task {
                Some(task) => {
                    tracing::info!("Found available task: {}", task.description);
                    // In a real implementation, this would dispatch to an actual agent
                    // For now, we just log it
                }
                None => {
                    // Check if plan is complete
                    let details = self.plan_manager.get_plan_details(plan_id).await?;
                    if let Some(d) = details {
                        let all_complete = d
                            .tasks
                            .iter()
                            .all(|t| t.task.status == TaskStatus::Completed);

                        if all_complete {
                            tracing::info!("Plan {} completed!", plan_id);
                            self.plan_manager
                                .update_plan_status(plan_id, PlanStatus::Completed)
                                .await?;
                            break;
                        }
                    }

                    // Wait before checking again
                    tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
                }
            }
        }

        Ok(())
    }

    // ========================================================================
    // CRUD wrappers — mutation + event emission
    // ========================================================================

    // --- Projects ---

    /// Create a project and emit event
    pub async fn create_project(&self, project: &ProjectNode) -> Result<()> {
        self.neo4j().create_project(project).await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Project,
                CrudAction::Created,
                project.id.to_string(),
            )
            .with_payload(serde_json::json!({"name": &project.name, "slug": &project.slug, "root_path": &project.root_path})),
        );
        Ok(())
    }

    /// Update a project and emit event
    pub async fn update_project(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<Option<String>>,
        root_path: Option<String>,
    ) -> Result<()> {
        self.neo4j()
            .update_project(id, name.clone(), description, root_path.clone())
            .await?;
        let mut payload = serde_json::Map::new();
        if let Some(ref n) = name {
            payload.insert("name".into(), serde_json::json!(n));
        }
        if let Some(ref rp) = root_path {
            payload.insert("root_path".into(), serde_json::json!(rp));
        }
        self.emit(
            CrudEvent::new(
                EventEntityType::Project,
                CrudAction::Updated,
                id.to_string(),
            )
            .with_payload(serde_json::Value::Object(payload)),
        );
        Ok(())
    }

    /// Delete a project and emit event
    pub async fn delete_project(&self, id: Uuid) -> Result<()> {
        self.neo4j().delete_project(id).await?;
        self.emit(CrudEvent::new(
            EventEntityType::Project,
            CrudAction::Deleted,
            id.to_string(),
        ));
        Ok(())
    }

    // --- Plans (link/unlink only — CRUD is in PlanManager) ---

    /// Link a plan to a project and emit event
    pub async fn link_plan_to_project(&self, plan_id: Uuid, project_id: Uuid) -> Result<()> {
        self.neo4j()
            .link_plan_to_project(plan_id, project_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Plan,
                CrudAction::Linked,
                plan_id.to_string(),
            )
            .with_payload(serde_json::json!({"project_id": project_id.to_string()})),
        );
        Ok(())
    }

    /// Unlink a plan from its project and emit event
    pub async fn unlink_plan_from_project(&self, plan_id: Uuid) -> Result<()> {
        self.neo4j().unlink_plan_from_project(plan_id).await?;
        self.emit(CrudEvent::new(
            EventEntityType::Plan,
            CrudAction::Unlinked,
            plan_id.to_string(),
        ));
        Ok(())
    }

    // --- Task dependencies ---

    /// Add a task dependency and emit event
    pub async fn add_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()> {
        self.neo4j()
            .add_task_dependency(task_id, depends_on_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Task,
                CrudAction::Linked,
                task_id.to_string(),
            )
            .with_payload(serde_json::json!({"depends_on": depends_on_id.to_string()})),
        );
        Ok(())
    }

    /// Remove a task dependency and emit event
    pub async fn remove_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()> {
        self.neo4j()
            .remove_task_dependency(task_id, depends_on_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Task,
                CrudAction::Unlinked,
                task_id.to_string(),
            )
            .with_payload(serde_json::json!({"depends_on": depends_on_id.to_string()})),
        );
        Ok(())
    }

    // --- Steps/Decisions/Constraints (delete + update only — create is in PlanManager) ---

    /// Delete a step and emit event
    pub async fn delete_step(&self, step_id: Uuid) -> Result<()> {
        self.neo4j().delete_step(step_id).await?;
        self.emit(CrudEvent::new(
            EventEntityType::Step,
            CrudAction::Deleted,
            step_id.to_string(),
        ));
        Ok(())
    }

    /// Update a decision and emit event
    pub async fn update_decision(
        &self,
        decision_id: Uuid,
        description: Option<String>,
        rationale: Option<String>,
        chosen_option: Option<String>,
    ) -> Result<()> {
        self.neo4j()
            .update_decision(decision_id, description, rationale, chosen_option)
            .await?;
        self.emit(CrudEvent::new(
            EventEntityType::Decision,
            CrudAction::Updated,
            decision_id.to_string(),
        ));
        Ok(())
    }

    /// Delete a decision and emit event
    pub async fn delete_decision(&self, decision_id: Uuid) -> Result<()> {
        self.neo4j().delete_decision(decision_id).await?;
        self.emit(CrudEvent::new(
            EventEntityType::Decision,
            CrudAction::Deleted,
            decision_id.to_string(),
        ));
        Ok(())
    }

    /// Update a constraint and emit event
    pub async fn update_constraint(
        &self,
        constraint_id: Uuid,
        description: Option<String>,
        constraint_type: Option<ConstraintType>,
        enforced_by: Option<String>,
    ) -> Result<()> {
        self.neo4j()
            .update_constraint(constraint_id, description, constraint_type, enforced_by)
            .await?;
        self.emit(CrudEvent::new(
            EventEntityType::Constraint,
            CrudAction::Updated,
            constraint_id.to_string(),
        ));
        Ok(())
    }

    /// Delete a constraint and emit event
    pub async fn delete_constraint(&self, constraint_id: Uuid) -> Result<()> {
        self.neo4j().delete_constraint(constraint_id).await?;
        self.emit(CrudEvent::new(
            EventEntityType::Constraint,
            CrudAction::Deleted,
            constraint_id.to_string(),
        ));
        Ok(())
    }

    // --- Commits ---

    /// Create a commit and emit event
    pub async fn create_commit(&self, commit: &CommitNode) -> Result<()> {
        self.neo4j().create_commit(commit).await?;
        self.emit(
            CrudEvent::new(EventEntityType::Commit, CrudAction::Created, &commit.hash)
                .with_payload(serde_json::json!({"message": &commit.message})),
        );
        Ok(())
    }

    /// Link a commit to a task and emit event
    pub async fn link_commit_to_task(&self, commit_hash: &str, task_id: Uuid) -> Result<()> {
        self.neo4j()
            .link_commit_to_task(commit_hash, task_id)
            .await?;
        self.emit(
            CrudEvent::new(EventEntityType::Commit, CrudAction::Linked, commit_hash)
                .with_payload(serde_json::json!({"task_id": task_id.to_string()})),
        );
        Ok(())
    }

    /// Link a commit to a plan and emit event
    pub async fn link_commit_to_plan(&self, commit_hash: &str, plan_id: Uuid) -> Result<()> {
        self.neo4j()
            .link_commit_to_plan(commit_hash, plan_id)
            .await?;
        self.emit(
            CrudEvent::new(EventEntityType::Commit, CrudAction::Linked, commit_hash)
                .with_payload(serde_json::json!({"plan_id": plan_id.to_string()})),
        );
        Ok(())
    }

    // --- Releases ---

    /// Create a release and emit event
    pub async fn create_release(&self, release: &ReleaseNode) -> Result<()> {
        self.neo4j().create_release(release).await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Release,
                CrudAction::Created,
                release.id.to_string(),
            )
            .with_payload(serde_json::json!({"version": &release.version}))
            .with_project_id(release.project_id.to_string()),
        );
        Ok(())
    }

    /// Update a release and emit event
    pub async fn update_release(
        &self,
        id: Uuid,
        status: Option<ReleaseStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        released_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        self.neo4j()
            .update_release(id, status, target_date, released_at, title, description)
            .await?;
        self.emit(CrudEvent::new(
            EventEntityType::Release,
            CrudAction::Updated,
            id.to_string(),
        ));
        Ok(())
    }

    /// Delete a release and emit event
    pub async fn delete_release(&self, release_id: Uuid) -> Result<()> {
        self.neo4j().delete_release(release_id).await?;
        self.emit(CrudEvent::new(
            EventEntityType::Release,
            CrudAction::Deleted,
            release_id.to_string(),
        ));
        Ok(())
    }

    /// Add a task to a release and emit event
    pub async fn add_task_to_release(&self, release_id: Uuid, task_id: Uuid) -> Result<()> {
        self.neo4j()
            .add_task_to_release(release_id, task_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Release,
                CrudAction::Linked,
                release_id.to_string(),
            )
            .with_payload(serde_json::json!({"task_id": task_id.to_string()})),
        );
        Ok(())
    }

    /// Add a commit to a release and emit event
    pub async fn add_commit_to_release(&self, release_id: Uuid, commit_hash: &str) -> Result<()> {
        self.neo4j()
            .add_commit_to_release(release_id, commit_hash)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Release,
                CrudAction::Linked,
                release_id.to_string(),
            )
            .with_payload(serde_json::json!({"commit_hash": commit_hash})),
        );
        Ok(())
    }

    /// Remove a commit from a release and emit event
    pub async fn remove_commit_from_release(
        &self,
        release_id: Uuid,
        commit_hash: &str,
    ) -> Result<()> {
        self.neo4j()
            .remove_commit_from_release(release_id, commit_hash)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Release,
                CrudAction::Unlinked,
                release_id.to_string(),
            )
            .with_payload(serde_json::json!({"commit_hash": commit_hash})),
        );
        Ok(())
    }

    // --- Milestones ---

    /// Create a milestone and emit event
    pub async fn create_milestone(&self, milestone: &MilestoneNode) -> Result<()> {
        self.neo4j().create_milestone(milestone).await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Milestone,
                CrudAction::Created,
                milestone.id.to_string(),
            )
            .with_payload(serde_json::json!({"title": &milestone.title}))
            .with_project_id(milestone.project_id.to_string()),
        );
        Ok(())
    }

    /// Update a milestone and emit event
    pub async fn update_milestone(
        &self,
        id: Uuid,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        closed_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        self.neo4j()
            .update_milestone(id, status, target_date, closed_at, title, description)
            .await?;
        self.emit(CrudEvent::new(
            EventEntityType::Milestone,
            CrudAction::Updated,
            id.to_string(),
        ));
        Ok(())
    }

    /// Delete a milestone and emit event
    pub async fn delete_milestone(&self, milestone_id: Uuid) -> Result<()> {
        self.neo4j().delete_milestone(milestone_id).await?;
        self.emit(CrudEvent::new(
            EventEntityType::Milestone,
            CrudAction::Deleted,
            milestone_id.to_string(),
        ));
        Ok(())
    }

    /// Add a task to a milestone and emit event
    pub async fn add_task_to_milestone(&self, milestone_id: Uuid, task_id: Uuid) -> Result<()> {
        self.neo4j()
            .add_task_to_milestone(milestone_id, task_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Milestone,
                CrudAction::Linked,
                milestone_id.to_string(),
            )
            .with_payload(serde_json::json!({"task_id": task_id.to_string()})),
        );
        Ok(())
    }

    /// Link a plan to a project milestone and emit event
    pub async fn link_plan_to_milestone(&self, plan_id: Uuid, milestone_id: Uuid) -> Result<()> {
        self.neo4j()
            .link_plan_to_milestone(plan_id, milestone_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Milestone,
                CrudAction::Linked,
                milestone_id.to_string(),
            )
            .with_payload(serde_json::json!({"plan_id": plan_id.to_string()})),
        );
        Ok(())
    }

    /// Unlink a plan from a project milestone and emit event
    pub async fn unlink_plan_from_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> Result<()> {
        self.neo4j()
            .unlink_plan_from_milestone(plan_id, milestone_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Milestone,
                CrudAction::Unlinked,
                milestone_id.to_string(),
            )
            .with_payload(serde_json::json!({"plan_id": plan_id.to_string()})),
        );
        Ok(())
    }

    // --- Workspaces ---

    /// Create a workspace and emit event
    pub async fn create_workspace(&self, workspace: &WorkspaceNode) -> Result<()> {
        self.neo4j().create_workspace(workspace).await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Workspace,
                CrudAction::Created,
                workspace.id.to_string(),
            )
            .with_payload(serde_json::json!({"name": &workspace.name, "slug": &workspace.slug})),
        );
        Ok(())
    }

    /// Update a workspace and emit event
    pub async fn update_workspace(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        self.neo4j()
            .update_workspace(id, name, description, metadata)
            .await?;
        self.emit(CrudEvent::new(
            EventEntityType::Workspace,
            CrudAction::Updated,
            id.to_string(),
        ));
        Ok(())
    }

    /// Delete a workspace and emit event
    pub async fn delete_workspace(&self, id: Uuid) -> Result<()> {
        self.neo4j().delete_workspace(id).await?;
        self.emit(CrudEvent::new(
            EventEntityType::Workspace,
            CrudAction::Deleted,
            id.to_string(),
        ));
        Ok(())
    }

    /// Add a project to a workspace and emit event
    pub async fn add_project_to_workspace(
        &self,
        workspace_id: Uuid,
        project_id: Uuid,
    ) -> Result<()> {
        self.neo4j()
            .add_project_to_workspace(workspace_id, project_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Workspace,
                CrudAction::Linked,
                workspace_id.to_string(),
            )
            .with_payload(serde_json::json!({"project_id": project_id.to_string()})),
        );
        Ok(())
    }

    /// Remove a project from a workspace and emit event
    pub async fn remove_project_from_workspace(
        &self,
        workspace_id: Uuid,
        project_id: Uuid,
    ) -> Result<()> {
        self.neo4j()
            .remove_project_from_workspace(workspace_id, project_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Workspace,
                CrudAction::Unlinked,
                workspace_id.to_string(),
            )
            .with_payload(serde_json::json!({"project_id": project_id.to_string()})),
        );
        Ok(())
    }

    // --- Workspace Milestones ---

    /// Create a workspace milestone and emit event
    pub async fn create_workspace_milestone(
        &self,
        milestone: &WorkspaceMilestoneNode,
    ) -> Result<()> {
        self.neo4j().create_workspace_milestone(milestone).await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::WorkspaceMilestone,
                CrudAction::Created,
                milestone.id.to_string(),
            )
            .with_payload(serde_json::json!({"title": &milestone.title})),
        );
        Ok(())
    }

    /// Update a workspace milestone and emit event
    pub async fn update_workspace_milestone(
        &self,
        id: Uuid,
        title: Option<String>,
        description: Option<String>,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<()> {
        self.neo4j()
            .update_workspace_milestone(id, title, description, status, target_date)
            .await?;
        self.emit(CrudEvent::new(
            EventEntityType::WorkspaceMilestone,
            CrudAction::Updated,
            id.to_string(),
        ));
        Ok(())
    }

    /// Delete a workspace milestone and emit event
    pub async fn delete_workspace_milestone(&self, id: Uuid) -> Result<()> {
        self.neo4j().delete_workspace_milestone(id).await?;
        self.emit(CrudEvent::new(
            EventEntityType::WorkspaceMilestone,
            CrudAction::Deleted,
            id.to_string(),
        ));
        Ok(())
    }

    /// Add a task to a workspace milestone and emit event
    pub async fn add_task_to_workspace_milestone(
        &self,
        milestone_id: Uuid,
        task_id: Uuid,
    ) -> Result<()> {
        self.neo4j()
            .add_task_to_workspace_milestone(milestone_id, task_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::WorkspaceMilestone,
                CrudAction::Linked,
                milestone_id.to_string(),
            )
            .with_payload(serde_json::json!({"task_id": task_id.to_string()})),
        );
        Ok(())
    }

    /// Link a plan to a workspace milestone and emit event
    pub async fn link_plan_to_workspace_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> Result<()> {
        self.neo4j()
            .link_plan_to_workspace_milestone(plan_id, milestone_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::WorkspaceMilestone,
                CrudAction::Linked,
                milestone_id.to_string(),
            )
            .with_payload(serde_json::json!({"plan_id": plan_id.to_string()})),
        );
        Ok(())
    }

    /// Unlink a plan from a workspace milestone and emit event
    pub async fn unlink_plan_from_workspace_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> Result<()> {
        self.neo4j()
            .unlink_plan_from_workspace_milestone(plan_id, milestone_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::WorkspaceMilestone,
                CrudAction::Unlinked,
                milestone_id.to_string(),
            )
            .with_payload(serde_json::json!({"plan_id": plan_id.to_string()})),
        );
        Ok(())
    }

    // --- Resources ---

    /// Create a resource and emit event
    pub async fn create_resource(&self, resource: &ResourceNode) -> Result<()> {
        self.neo4j().create_resource(resource).await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Resource,
                CrudAction::Created,
                resource.id.to_string(),
            )
            .with_payload(serde_json::json!({"name": &resource.name})),
        );
        Ok(())
    }

    /// Update a resource and emit event
    pub async fn update_resource(
        &self,
        id: Uuid,
        name: Option<String>,
        file_path: Option<String>,
        url: Option<String>,
        version: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        self.neo4j()
            .update_resource(id, name, file_path, url, version, description)
            .await?;
        self.emit(CrudEvent::new(
            EventEntityType::Resource,
            CrudAction::Updated,
            id.to_string(),
        ));
        Ok(())
    }

    /// Delete a resource and emit event
    pub async fn delete_resource(&self, id: Uuid) -> Result<()> {
        self.neo4j().delete_resource(id).await?;
        self.emit(CrudEvent::new(
            EventEntityType::Resource,
            CrudAction::Deleted,
            id.to_string(),
        ));
        Ok(())
    }

    /// Link a project to a resource (implements) and emit event
    pub async fn link_project_implements_resource(
        &self,
        project_id: Uuid,
        resource_id: Uuid,
    ) -> Result<()> {
        self.neo4j()
            .link_project_implements_resource(project_id, resource_id)
            .await?;
        self.emit(
            CrudEvent::new(EventEntityType::Resource, CrudAction::Linked, resource_id.to_string())
                .with_payload(serde_json::json!({"project_id": project_id.to_string(), "link_type": "implements"})),
        );
        Ok(())
    }

    /// Link a project to a resource (uses) and emit event
    pub async fn link_project_uses_resource(
        &self,
        project_id: Uuid,
        resource_id: Uuid,
    ) -> Result<()> {
        self.neo4j()
            .link_project_uses_resource(project_id, resource_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Resource,
                CrudAction::Linked,
                resource_id.to_string(),
            )
            .with_payload(
                serde_json::json!({"project_id": project_id.to_string(), "link_type": "uses"}),
            ),
        );
        Ok(())
    }

    // --- Components ---

    /// Create a component and emit event
    pub async fn create_component(&self, component: &ComponentNode) -> Result<()> {
        self.neo4j().create_component(component).await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Component,
                CrudAction::Created,
                component.id.to_string(),
            )
            .with_payload(serde_json::json!({"name": &component.name})),
        );
        Ok(())
    }

    /// Update a component and emit event
    pub async fn update_component(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        runtime: Option<String>,
        config: Option<serde_json::Value>,
        tags: Option<Vec<String>>,
    ) -> Result<()> {
        self.neo4j()
            .update_component(id, name, description, runtime, config, tags)
            .await?;
        self.emit(CrudEvent::new(
            EventEntityType::Component,
            CrudAction::Updated,
            id.to_string(),
        ));
        Ok(())
    }

    /// Delete a component and emit event
    pub async fn delete_component(&self, id: Uuid) -> Result<()> {
        self.neo4j().delete_component(id).await?;
        self.emit(CrudEvent::new(
            EventEntityType::Component,
            CrudAction::Deleted,
            id.to_string(),
        ));
        Ok(())
    }

    /// Add a component dependency and emit event
    pub async fn add_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
        protocol: Option<String>,
        required: bool,
    ) -> Result<()> {
        self.neo4j()
            .add_component_dependency(component_id, depends_on_id, protocol, required)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Component,
                CrudAction::Linked,
                component_id.to_string(),
            )
            .with_payload(serde_json::json!({"depends_on": depends_on_id.to_string()})),
        );
        Ok(())
    }

    /// Remove a component dependency and emit event
    pub async fn remove_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
    ) -> Result<()> {
        self.neo4j()
            .remove_component_dependency(component_id, depends_on_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Component,
                CrudAction::Unlinked,
                component_id.to_string(),
            )
            .with_payload(serde_json::json!({"depends_on": depends_on_id.to_string()})),
        );
        Ok(())
    }

    /// Map a component to a project and emit event
    pub async fn map_component_to_project(
        &self,
        component_id: Uuid,
        project_id: Uuid,
    ) -> Result<()> {
        self.neo4j()
            .map_component_to_project(component_id, project_id)
            .await?;
        self.emit(
            CrudEvent::new(
                EventEntityType::Component,
                CrudAction::Linked,
                component_id.to_string(),
            )
            .with_payload(serde_json::json!({"project_id": project_id.to_string()})),
        );
        Ok(())
    }
}

/// Result of a sync operation
#[derive(Debug, Default)]
pub struct SyncResult {
    pub files_synced: usize,
    pub files_skipped: usize,
    pub files_deleted: usize,
    pub symbols_deleted: usize,
    pub errors: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{CrudAction, EntityType as EventEntityType, EventBus, HybridEmitter};
    use crate::test_helpers::*;

    /// Helper: create an Orchestrator with HybridEmitter, return (orchestrator, receiver)
    async fn orch_with_bus() -> (Orchestrator, tokio::sync::broadcast::Receiver<CrudEvent>) {
        let state = mock_app_state();
        let bus = Arc::new(EventBus::default());
        let hybrid = Arc::new(HybridEmitter::new(bus));
        let rx = hybrid.subscribe();
        let orch = Orchestrator::with_event_bus(state, hybrid).await.unwrap();
        (orch, rx)
    }

    // ── constructors ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_new_has_no_event_bus() {
        let orch = Orchestrator::new(mock_app_state()).await.unwrap();
        assert!(orch.event_bus().is_none());
    }

    #[tokio::test]
    async fn test_with_event_bus_has_bus() {
        let (orch, _rx) = orch_with_bus().await;
        assert!(orch.event_bus().is_some());
    }

    #[tokio::test]
    async fn test_accessors() {
        let (orch, _rx) = orch_with_bus().await;
        let _ = orch.plan_manager();
        let _ = orch.context_builder();
        let _ = orch.neo4j();
        let _ = orch.note_manager();
        let _ = orch.note_lifecycle();
    }

    // ── Projects ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_create_project_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let project = test_project();
        orch.create_project(&project).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.entity_type, EventEntityType::Project);
        assert_eq!(ev.action, CrudAction::Created);
        assert_eq!(ev.entity_id, project.id.to_string());
    }

    #[tokio::test]
    async fn test_update_project_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let project = test_project();
        orch.neo4j().create_project(&project).await.unwrap();
        orch.update_project(project.id, Some("new-name".into()), None, None)
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Updated);
    }

    #[tokio::test]
    async fn test_delete_project_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let project = test_project();
        orch.neo4j().create_project(&project).await.unwrap();
        orch.delete_project(project.id).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Deleted);
    }

    // ── Plan link/unlink ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_link_plan_to_project_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let plan_id = Uuid::new_v4();
        let project_id = Uuid::new_v4();
        orch.link_plan_to_project(plan_id, project_id)
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.entity_type, EventEntityType::Plan);
        assert_eq!(ev.action, CrudAction::Linked);
    }

    #[tokio::test]
    async fn test_unlink_plan_from_project_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.unlink_plan_from_project(Uuid::new_v4()).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Unlinked);
    }

    // ── Task dependencies ────────────────────────────────────────────

    #[tokio::test]
    async fn test_add_task_dependency_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let t1 = Uuid::new_v4();
        let t2 = Uuid::new_v4();
        orch.add_task_dependency(t1, t2).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.entity_type, EventEntityType::Task);
        assert_eq!(ev.action, CrudAction::Linked);
    }

    #[tokio::test]
    async fn test_remove_task_dependency_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.remove_task_dependency(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Unlinked);
    }

    // ── Steps / Decisions / Constraints ──────────────────────────────

    #[tokio::test]
    async fn test_delete_step_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let id = Uuid::new_v4();
        orch.delete_step(id).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.entity_type, EventEntityType::Step);
        assert_eq!(ev.action, CrudAction::Deleted);
    }

    #[tokio::test]
    async fn test_update_decision_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let id = Uuid::new_v4();
        orch.update_decision(id, Some("desc".into()), None, None)
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.entity_type, EventEntityType::Decision);
        assert_eq!(ev.action, CrudAction::Updated);
    }

    #[tokio::test]
    async fn test_delete_decision_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.delete_decision(Uuid::new_v4()).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Deleted);
    }

    #[tokio::test]
    async fn test_update_constraint_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.update_constraint(Uuid::new_v4(), Some("desc".into()), None, None)
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.entity_type, EventEntityType::Constraint);
        assert_eq!(ev.action, CrudAction::Updated);
    }

    #[tokio::test]
    async fn test_delete_constraint_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.delete_constraint(Uuid::new_v4()).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Deleted);
    }

    // ── Commits ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_create_commit_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let commit = test_commit("abc123", "feat: test");
        orch.create_commit(&commit).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.entity_type, EventEntityType::Commit);
        assert_eq!(ev.action, CrudAction::Created);
        assert_eq!(ev.entity_id, "abc123");
    }

    #[tokio::test]
    async fn test_link_commit_to_task_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.link_commit_to_task("abc", Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Linked);
    }

    #[tokio::test]
    async fn test_link_commit_to_plan_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.link_commit_to_plan("abc", Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Linked);
    }

    // ── Releases ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_create_release_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let release = test_release(Uuid::new_v4(), "1.0.0");
        orch.create_release(&release).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.entity_type, EventEntityType::Release);
        assert_eq!(ev.action, CrudAction::Created);
        assert!(ev.project_id.is_some());
    }

    #[tokio::test]
    async fn test_update_release_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.update_release(Uuid::new_v4(), None, None, None, None, None)
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Updated);
    }

    #[tokio::test]
    async fn test_delete_release_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.delete_release(Uuid::new_v4()).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Deleted);
    }

    #[tokio::test]
    async fn test_add_task_to_release_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.add_task_to_release(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Linked);
    }

    #[tokio::test]
    async fn test_add_commit_to_release_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.add_commit_to_release(Uuid::new_v4(), "abc123")
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Linked);
    }

    // ── Milestones ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_create_milestone_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let ms = test_milestone(Uuid::new_v4(), "v1 launch");
        orch.create_milestone(&ms).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.entity_type, EventEntityType::Milestone);
        assert_eq!(ev.action, CrudAction::Created);
    }

    #[tokio::test]
    async fn test_update_milestone_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.update_milestone(Uuid::new_v4(), None, None, None, None, None)
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Updated);
    }

    #[tokio::test]
    async fn test_delete_milestone_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.delete_milestone(Uuid::new_v4()).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Deleted);
    }

    #[tokio::test]
    async fn test_add_task_to_milestone_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.add_task_to_milestone(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Linked);
    }

    #[tokio::test]
    async fn test_link_plan_to_milestone_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.link_plan_to_milestone(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Linked);
        assert_eq!(ev.entity_type, EventEntityType::Milestone);
    }

    #[tokio::test]
    async fn test_unlink_plan_from_milestone_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.unlink_plan_from_milestone(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Unlinked);
        assert_eq!(ev.entity_type, EventEntityType::Milestone);
    }

    // ── Workspaces ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_create_workspace_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let ws = test_workspace();
        orch.create_workspace(&ws).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.entity_type, EventEntityType::Workspace);
        assert_eq!(ev.action, CrudAction::Created);
    }

    #[tokio::test]
    async fn test_update_workspace_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let ws = test_workspace();
        orch.neo4j().create_workspace(&ws).await.unwrap();
        orch.update_workspace(ws.id, Some("new".into()), None, None)
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Updated);
    }

    #[tokio::test]
    async fn test_delete_workspace_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.delete_workspace(Uuid::new_v4()).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Deleted);
    }

    #[tokio::test]
    async fn test_add_project_to_workspace_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.add_project_to_workspace(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Linked);
    }

    #[tokio::test]
    async fn test_remove_project_from_workspace_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.remove_project_from_workspace(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Unlinked);
    }

    // ── Workspace Milestones ─────────────────────────────────────────

    #[tokio::test]
    async fn test_create_workspace_milestone_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let ms = WorkspaceMilestoneNode {
            id: Uuid::new_v4(),
            workspace_id: Uuid::new_v4(),
            title: "Cross-project milestone".into(),
            description: None,
            status: MilestoneStatus::Open,
            target_date: None,
            closed_at: None,
            created_at: chrono::Utc::now(),
            tags: vec![],
        };
        orch.create_workspace_milestone(&ms).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.entity_type, EventEntityType::WorkspaceMilestone);
        assert_eq!(ev.action, CrudAction::Created);
    }

    #[tokio::test]
    async fn test_update_workspace_milestone_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.update_workspace_milestone(Uuid::new_v4(), Some("t".into()), None, None, None)
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Updated);
    }

    #[tokio::test]
    async fn test_delete_workspace_milestone_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.delete_workspace_milestone(Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Deleted);
    }

    #[tokio::test]
    async fn test_add_task_to_workspace_milestone_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.add_task_to_workspace_milestone(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Linked);
    }

    #[tokio::test]
    async fn test_link_plan_to_workspace_milestone_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.link_plan_to_workspace_milestone(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Linked);
        assert_eq!(ev.entity_type, EventEntityType::WorkspaceMilestone);
    }

    #[tokio::test]
    async fn test_unlink_plan_from_workspace_milestone_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.unlink_plan_from_workspace_milestone(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Unlinked);
        assert_eq!(ev.entity_type, EventEntityType::WorkspaceMilestone);
    }

    // ── Resources ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_create_resource_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let res = ResourceNode {
            id: Uuid::new_v4(),
            workspace_id: Some(Uuid::new_v4()),
            project_id: None,
            name: "API spec".into(),
            resource_type: ResourceType::ApiContract,
            file_path: "api.yaml".into(),
            url: None,
            format: Some("openapi".into()),
            version: Some("1.0".into()),
            description: None,
            created_at: chrono::Utc::now(),
            updated_at: None,
            metadata: serde_json::json!({}),
        };
        orch.create_resource(&res).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.entity_type, EventEntityType::Resource);
        assert_eq!(ev.action, CrudAction::Created);
    }

    #[tokio::test]
    async fn test_update_resource_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.update_resource(Uuid::new_v4(), Some("n".into()), None, None, None, None)
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Updated);
    }

    #[tokio::test]
    async fn test_delete_resource_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.delete_resource(Uuid::new_v4()).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Deleted);
    }

    #[tokio::test]
    async fn test_link_project_implements_resource_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.link_project_implements_resource(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Linked);
        assert!(!ev.payload.is_null());
    }

    #[tokio::test]
    async fn test_link_project_uses_resource_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.link_project_uses_resource(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Linked);
    }

    // ── Components ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_create_component_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        let comp = ComponentNode {
            id: Uuid::new_v4(),
            workspace_id: Uuid::new_v4(),
            name: "api-gateway".into(),
            component_type: ComponentType::Gateway,
            description: None,
            runtime: None,
            config: serde_json::json!({}),
            created_at: chrono::Utc::now(),
            tags: vec![],
        };
        orch.create_component(&comp).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.entity_type, EventEntityType::Component);
        assert_eq!(ev.action, CrudAction::Created);
    }

    #[tokio::test]
    async fn test_update_component_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.update_component(Uuid::new_v4(), Some("n".into()), None, None, None, None)
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Updated);
    }

    #[tokio::test]
    async fn test_delete_component_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.delete_component(Uuid::new_v4()).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Deleted);
    }

    #[tokio::test]
    async fn test_add_component_dependency_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.add_component_dependency(Uuid::new_v4(), Uuid::new_v4(), Some("grpc".into()), true)
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Linked);
    }

    #[tokio::test]
    async fn test_remove_component_dependency_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.remove_component_dependency(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Unlinked);
    }

    #[tokio::test]
    async fn test_map_component_to_project_emits_event() {
        let (orch, mut rx) = orch_with_bus().await;
        orch.map_component_to_project(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Linked);
    }

    // ── emit without bus (no-op, no panic) ───────────────────────────

    #[tokio::test]
    async fn test_wrapper_without_bus_does_not_panic() {
        let orch = Orchestrator::new(mock_app_state()).await.unwrap();
        let project = test_project();
        // Should succeed silently even without event bus
        orch.create_project(&project).await.unwrap();
        orch.delete_project(project.id).await.unwrap();
    }

    // ── with_event_emitter ──────────────────────────────────────────

    #[tokio::test]
    async fn test_with_event_emitter_has_no_event_bus() {
        // with_event_emitter should NOT set the event_bus field (only event_emitter)
        let state = mock_app_state();
        let bus = Arc::new(EventBus::default());
        let emitter: Arc<dyn EventEmitter> = bus;
        let orch = Orchestrator::with_event_emitter(state, emitter)
            .await
            .unwrap();
        // event_bus is None (only set by with_event_bus for WS subscribe)
        assert!(orch.event_bus().is_none());
    }

    #[tokio::test]
    async fn test_with_event_emitter_emits_events() {
        use std::sync::Mutex;

        struct RecordingEmitter(Mutex<Vec<CrudEvent>>);
        impl EventEmitter for RecordingEmitter {
            fn emit(&self, event: CrudEvent) {
                self.0.lock().unwrap().push(event);
            }
        }

        let emitter = Arc::new(RecordingEmitter(Mutex::new(Vec::new())));
        let state = mock_app_state();
        let orch = Orchestrator::with_event_emitter(state, emitter.clone())
            .await
            .unwrap();

        let project = test_project();
        orch.create_project(&project).await.unwrap();

        let events = emitter.0.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].entity_type, EventEntityType::Project);
        assert_eq!(events[0].action, CrudAction::Created);
    }

    #[tokio::test]
    async fn test_with_event_emitter_managers_emit_too() {
        use std::sync::Mutex;

        struct RecordingEmitter(Mutex<Vec<CrudEvent>>);
        impl EventEmitter for RecordingEmitter {
            fn emit(&self, event: CrudEvent) {
                self.0.lock().unwrap().push(event);
            }
        }

        let emitter = Arc::new(RecordingEmitter(Mutex::new(Vec::new())));
        let state = mock_app_state();
        let orch = Orchestrator::with_event_emitter(state, emitter.clone())
            .await
            .unwrap();

        // Create plan via PlanManager — should also use the emitter
        let plan = orch
            .plan_manager()
            .create_plan(
                crate::plan::models::CreatePlanRequest {
                    title: "Test".into(),
                    description: "Desc".into(),
                    project_id: None,
                    priority: Some(1),
                    constraints: None,
                },
                "agent",
            )
            .await
            .unwrap();

        let events = emitter.0.lock().unwrap();
        // PlanManager should have emitted a Created event
        assert!(
            events.iter().any(|e| e.entity_type == EventEntityType::Plan
                && e.action == CrudAction::Created
                && e.entity_id == plan.id.to_string()),
            "PlanManager should emit via the shared EventEmitter"
        );
    }

    // ========================================================================
    // Feature Graph Proposal Parsing Tests
    // ========================================================================

    #[test]
    fn test_parse_feature_graph_proposals_valid_json() {
        let json = r#"[
            {
                "name": "Code Sync Pipeline",
                "description": "Handles file sync and tree-sitter parsing",
                "entry_function": "sync_directory",
                "depth": 3,
                "include_relations": ["calls", "imports"]
            },
            {
                "name": "MCP Tool Dispatch",
                "description": "Routes MCP tool calls to handlers",
                "entry_function": "handle_tools_call",
                "depth": 2
            }
        ]"#;

        let proposals: Vec<FeatureGraphProposal> = serde_json::from_str(json).unwrap();
        assert_eq!(proposals.len(), 2);

        assert_eq!(proposals[0].name, "Code Sync Pipeline");
        assert_eq!(
            proposals[0].description.as_deref(),
            Some("Handles file sync and tree-sitter parsing")
        );
        assert_eq!(proposals[0].entry_function, "sync_directory");
        assert_eq!(proposals[0].depth, 3);
        assert_eq!(
            proposals[0].include_relations.as_ref().unwrap(),
            &vec!["calls".to_string(), "imports".to_string()]
        );

        assert_eq!(proposals[1].name, "MCP Tool Dispatch");
        assert_eq!(proposals[1].depth, 2);
        assert!(proposals[1].include_relations.is_none());
    }

    #[test]
    fn test_parse_feature_graph_proposals_defaults() {
        // Minimal JSON — depth defaults to 2, include_relations defaults to None
        let json = r#"[{
            "name": "Auth Flow",
            "entry_function": "authenticate"
        }]"#;

        let proposals: Vec<FeatureGraphProposal> = serde_json::from_str(json).unwrap();
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].depth, 2);
        assert!(proposals[0].description.is_none());
        assert!(proposals[0].include_relations.is_none());
    }

    #[test]
    fn test_parse_feature_graph_proposals_invalid_json() {
        let json = "this is not json at all";
        let result = serde_json::from_str::<Vec<FeatureGraphProposal>>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_feature_graph_proposals_missing_required_field() {
        // Missing entry_function — should fail
        let json = r#"[{"name": "No Entry"}]"#;
        let result = serde_json::from_str::<Vec<FeatureGraphProposal>>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_feature_graph_proposals_strips_markdown_fences() {
        let response = "```json\n[{\"name\": \"Test\", \"entry_function\": \"foo\"}]\n```";
        let json_str = response
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let proposals: Vec<FeatureGraphProposal> = serde_json::from_str(json_str).unwrap();
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].name, "Test");
        assert_eq!(proposals[0].entry_function, "foo");
    }

    // ========================================================================
    // gather_codebase_context Tests
    // ========================================================================

    use crate::neo4j::models::{FileNode, FunctionNode, Visibility};
    use chrono::Utc;

    fn test_file(path: &str, project_id: Uuid) -> FileNode {
        FileNode {
            path: path.to_string(),
            language: "rust".to_string(),
            hash: "abc123".to_string(),
            last_parsed: Utc::now(),
            project_id: Some(project_id),
        }
    }

    fn test_function(name: &str, file_path: &str) -> FunctionNode {
        FunctionNode {
            name: name.to_string(),
            visibility: Visibility::Public,
            params: vec![],
            return_type: None,
            generics: vec![],
            is_async: false,
            is_unsafe: false,
            complexity: 1,
            file_path: file_path.to_string(),
            line_start: 1,
            line_end: 10,
            docstring: None,
        }
    }

    #[tokio::test]
    async fn test_gather_codebase_context_returns_valid_json() {
        let state = mock_app_state();
        let neo4j = state.neo4j.as_ref();
        let project = test_project();
        neo4j.create_project(&project).await.unwrap();

        neo4j
            .upsert_file(&test_file("src/main.rs", project.id))
            .await
            .unwrap();
        neo4j
            .upsert_function(&test_function("main", "src/main.rs"))
            .await
            .unwrap();

        let context = Orchestrator::gather_codebase_context(neo4j, project.id)
            .await
            .unwrap();

        // Should be valid JSON with all expected keys
        let parsed: serde_json::Value = serde_json::from_str(&context).unwrap();
        assert!(parsed.get("top_functions").unwrap().is_array());
        assert!(parsed.get("connected_files").unwrap().is_array());
        assert!(parsed.get("language_stats").unwrap().is_array());
        assert!(parsed.get("existing_feature_graphs").unwrap().is_array());
    }

    #[tokio::test]
    async fn test_gather_codebase_context_empty_project() {
        let state = mock_app_state();
        let neo4j = state.neo4j.as_ref();
        let project = test_project();
        neo4j.create_project(&project).await.unwrap();

        let context = Orchestrator::gather_codebase_context(neo4j, project.id)
            .await
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&context).unwrap();
        assert_eq!(parsed["top_functions"].as_array().unwrap().len(), 0);
        assert_eq!(parsed["connected_files"].as_array().unwrap().len(), 0);
        assert_eq!(
            parsed["existing_feature_graphs"].as_array().unwrap().len(),
            0
        );
    }

    #[tokio::test]
    async fn test_gather_codebase_context_includes_existing_feature_graphs() {
        let state = mock_app_state();
        let neo4j = state.neo4j.as_ref();
        let project = test_project();
        neo4j.create_project(&project).await.unwrap();

        neo4j
            .upsert_file(&test_file("src/lib.rs", project.id))
            .await
            .unwrap();
        neo4j
            .upsert_function(&test_function("entry_fn", "src/lib.rs"))
            .await
            .unwrap();

        // Create a feature graph
        neo4j
            .auto_build_feature_graph(
                "Test FG",
                Some("desc"),
                project.id,
                "entry_fn",
                2,
                None,
                None,
            )
            .await
            .unwrap();

        let context = Orchestrator::gather_codebase_context(neo4j, project.id)
            .await
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&context).unwrap();
        let fgs = parsed["existing_feature_graphs"].as_array().unwrap();
        assert_eq!(fgs.len(), 1);
        assert_eq!(fgs[0]["name"], "Test FG");
    }

    #[tokio::test]
    async fn test_gather_codebase_context_includes_language_stats() {
        let state = mock_app_state();
        let neo4j = state.neo4j.as_ref();
        let project = test_project();
        neo4j.create_project(&project).await.unwrap();

        neo4j
            .upsert_file(&test_file("src/main.rs", project.id))
            .await
            .unwrap();
        neo4j
            .upsert_file(&test_file("src/lib.rs", project.id))
            .await
            .unwrap();

        let context = Orchestrator::gather_codebase_context(neo4j, project.id)
            .await
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&context).unwrap();
        let langs = parsed["language_stats"].as_array().unwrap();
        assert!(
            !langs.is_empty(),
            "Should have language stats for rust files"
        );
        assert_eq!(langs[0]["language"], "rust");
    }

    // ========================================================================
    // build_feature_graph_prompt Tests
    // ========================================================================

    #[test]
    fn test_build_feature_graph_prompt_contains_context() {
        let context = r#"{"top_functions":["foo","bar"]}"#;
        let prompt = Orchestrator::build_feature_graph_prompt(context);

        assert!(prompt.contains("feature graph"));
        assert!(prompt.contains(context));
        assert!(prompt.contains("entry_function"));
        assert!(prompt.contains("JSON array"));
        assert!(prompt.contains("5 to 15"));
    }

    #[test]
    fn test_build_feature_graph_prompt_has_response_format() {
        let prompt = Orchestrator::build_feature_graph_prompt("{}");

        // Should contain the JSON example structure
        assert!(prompt.contains("\"name\": \"Feature Name\""));
        assert!(prompt.contains("\"depth\": 2"));
        assert!(prompt.contains("include_relations"));
    }

    // ========================================================================
    // FeatureGraphProposal edge cases
    // ========================================================================

    #[test]
    fn test_parse_feature_graph_proposal_depth_zero() {
        let json = r#"[{"name": "Test", "entry_function": "foo", "depth": 0}]"#;
        let proposals: Vec<FeatureGraphProposal> = serde_json::from_str(json).unwrap();
        assert_eq!(proposals[0].depth, 0);
    }

    #[test]
    fn test_parse_feature_graph_proposal_depth_max() {
        let json = r#"[{"name": "Test", "entry_function": "foo", "depth": 5}]"#;
        let proposals: Vec<FeatureGraphProposal> = serde_json::from_str(json).unwrap();
        assert_eq!(proposals[0].depth, 5);
    }

    #[test]
    fn test_parse_feature_graph_proposal_empty_include_relations() {
        let json = r#"[{"name": "Test", "entry_function": "foo", "include_relations": []}]"#;
        let proposals: Vec<FeatureGraphProposal> = serde_json::from_str(json).unwrap();
        assert_eq!(proposals[0].include_relations.as_ref().unwrap().len(), 0);
    }

    #[test]
    fn test_parse_feature_graph_proposal_empty_array() {
        let json = "[]";
        let proposals: Vec<FeatureGraphProposal> = serde_json::from_str(json).unwrap();
        assert!(proposals.is_empty());
    }

    #[test]
    fn test_parse_feature_graph_proposal_unicode_names() {
        let json = r#"[{"name": "Gestion des données 📊", "entry_function": "process_données", "description": "Traitement des entrées utilisateur"}]"#;
        let proposals: Vec<FeatureGraphProposal> = serde_json::from_str(json).unwrap();
        assert_eq!(proposals[0].name, "Gestion des données 📊");
        assert!(proposals[0].description.is_some());
    }

    #[test]
    fn test_parse_feature_graph_proposals_strips_plain_markdown_fences() {
        // Just ``` without json tag
        let response = "```\n[{\"name\": \"Test\", \"entry_function\": \"bar\"}]\n```";
        let json_str = response
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let proposals: Vec<FeatureGraphProposal> = serde_json::from_str(json_str).unwrap();
        assert_eq!(proposals[0].entry_function, "bar");
    }

    // ── analytics integration ──────────────────────────────────────────

    #[tokio::test]
    async fn test_analyze_project_safe_computes_and_emits() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::GraphStore;

        // Build custom state with a shared MockGraphStore reference
        let mock_store = Arc::new(MockGraphStore::new());
        let state = crate::test_helpers::mock_app_state_with_graph(mock_store.clone());
        let bus = Arc::new(crate::events::EventBus::default());
        let hybrid = Arc::new(HybridEmitter::new(bus));
        let mut rx = hybrid.subscribe();
        let orch = Orchestrator::with_event_bus(state, hybrid).await.unwrap();

        let project = test_project();
        orch.create_project(&project).await.unwrap();
        let _create_ev = rx.recv().await.unwrap(); // drain create event

        // Seed files via the shared MockGraphStore
        let files = ["src/main.rs", "src/lib.rs", "src/api.rs"];
        for path in &files {
            let file = crate::neo4j::models::FileNode {
                path: path.to_string(),
                language: "rust".to_string(),
                hash: "abc".to_string(),
                last_parsed: chrono::Utc::now(),
                project_id: Some(project.id),
            };
            GraphStore::upsert_file(&*mock_store, &file).await.unwrap();
        }
        // Register files as project files and add imports
        {
            let mut pf = mock_store.project_files.write().await;
            pf.entry(project.id)
                .or_default()
                .extend(files.iter().map(|s| s.to_string()));
        }
        {
            let mut imports = mock_store.import_relationships.write().await;
            imports
                .entry("src/main.rs".to_string())
                .or_default()
                .push("src/lib.rs".to_string());
            imports
                .entry("src/main.rs".to_string())
                .or_default()
                .push("src/api.rs".to_string());
            imports
                .entry("src/api.rs".to_string())
                .or_default()
                .push("src/lib.rs".to_string());
        }

        // Run analytics (synchronous, best-effort)
        orch.analyze_project_safe(project.id).await;

        // Verify analytics scores were persisted to MockGraphStore
        {
            let fa = mock_store.file_analytics.read().await;
            assert_eq!(fa.len(), 3, "Should have analytics for 3 files");
            let main_rs = &fa["src/main.rs"];
            assert!(main_rs.pagerank > 0.0, "PageRank should be positive");
            assert!(
                !main_rs.community_label.is_empty(),
                "Community label should be set"
            );
        }

        // Verify CrudEvent was emitted with analytics_computed payload
        let ev = rx.recv().await.unwrap();
        assert_eq!(ev.entity_type, EventEntityType::Project);
        assert_eq!(ev.action, CrudAction::Updated);
        assert_eq!(ev.entity_id, project.id.to_string());
        assert!(!ev.payload.is_null(), "Payload should be set");
        assert_eq!(ev.payload["type"], "analytics_computed");
        assert_eq!(ev.payload["file_nodes"], 3);
    }

    #[tokio::test]
    async fn test_analyze_project_safe_empty_project_no_error() {
        let orch = Orchestrator::new(mock_app_state()).await.unwrap();
        let project = test_project();
        orch.create_project(&project).await.unwrap();

        // Should not panic or error on empty project
        orch.analyze_project_safe(project.id).await;
    }

    // ── Staleness detection ─────────────────────────────────────────

    #[tokio::test]
    async fn test_staleness_never_computed_is_stale() {
        let orch = Orchestrator::new(mock_app_state()).await.unwrap();
        let project = test_project();
        orch.create_project(&project).await.unwrap();

        let report = orch.check_analytics_staleness(project.id).await.unwrap();
        assert!(report.is_stale, "Project with no analytics should be stale");
        assert!(report.analytics_computed_at.is_none());
        assert!(report.analytics_age.is_none());
    }

    #[tokio::test]
    async fn test_staleness_computed_after_sync_is_fresh() {
        let orch = Orchestrator::new(mock_app_state()).await.unwrap();
        let mut project = test_project();
        project.last_synced = Some(chrono::Utc::now() - chrono::Duration::seconds(10));
        orch.create_project(&project).await.unwrap();

        // Simulate analytics computed AFTER the sync
        orch.neo4j()
            .update_project_analytics_timestamp(project.id)
            .await
            .unwrap();

        let report = orch.check_analytics_staleness(project.id).await.unwrap();
        assert!(
            !report.is_stale,
            "Analytics computed after sync should not be stale"
        );
        assert!(report.analytics_computed_at.is_some());
    }

    #[tokio::test]
    async fn test_staleness_synced_after_analytics_is_stale() {
        let orch = Orchestrator::new(mock_app_state()).await.unwrap();
        let mut project = test_project();
        // Set analytics_computed_at in the past
        project.analytics_computed_at = Some(chrono::Utc::now() - chrono::Duration::seconds(60));
        // Set last_synced AFTER analytics
        project.last_synced = Some(chrono::Utc::now());
        orch.create_project(&project).await.unwrap();

        let report = orch.check_analytics_staleness(project.id).await.unwrap();
        assert!(
            report.is_stale,
            "Project synced after analytics should be stale"
        );
        assert!(report.analytics_age.is_some());
    }

    #[tokio::test]
    async fn test_staleness_nonexistent_project_errors() {
        let orch = Orchestrator::new(mock_app_state()).await.unwrap();
        let result = orch.check_analytics_staleness(Uuid::new_v4()).await;
        assert!(result.is_err(), "Nonexistent project should error");
    }

    #[tokio::test]
    async fn test_get_stale_projects_returns_only_stale() {
        let orch = Orchestrator::new(mock_app_state()).await.unwrap();

        // Project 1: never computed → stale
        let p1 = test_project_named("stale-project");
        orch.create_project(&p1).await.unwrap();

        // Project 2: computed after sync → fresh
        let mut p2 = test_project_named("fresh-project");
        p2.last_synced = Some(chrono::Utc::now() - chrono::Duration::seconds(10));
        p2.analytics_computed_at = Some(chrono::Utc::now());
        orch.create_project(&p2).await.unwrap();

        let stale = orch.get_stale_projects().await.unwrap();
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].project_id, p1.id);
    }

    // ── code embedding tests ────────────────────────────────────────

    #[test]
    fn test_build_file_embedding_text() {
        use crate::neo4j::models::{FunctionNode, StructNode, Visibility};

        let parsed = ParsedFile {
            path: "src/api/handlers.rs".to_string(),
            language: "rust".to_string(),
            hash: "abc123".to_string(),
            functions: vec![
                FunctionNode {
                    name: "create_plan".to_string(),
                    visibility: Visibility::Public,
                    params: vec![],
                    return_type: Some("Result<Plan>".to_string()),
                    generics: vec![],
                    is_async: true,
                    is_unsafe: false,
                    complexity: 3,
                    file_path: "src/api/handlers.rs".to_string(),
                    line_start: 10,
                    line_end: 50,
                    docstring: None,
                },
            ],
            structs: vec![
                StructNode {
                    name: "PlanRequest".to_string(),
                    file_path: "src/api/handlers.rs".to_string(),
                    visibility: Visibility::Public,
                    generics: vec![],
                    line_start: 1,
                    line_end: 5,
                    docstring: None,
                },
            ],
            traits: vec![],
            enums: vec![],
            imports: vec![],
            impl_blocks: vec![],
            function_calls: vec![],
            symbols: vec![],
        };

        let text = Orchestrator::build_file_embedding_text(&parsed);
        assert!(text.contains("src/api/handlers.rs"));
        assert!(text.contains("(rust)"));
        assert!(text.contains("fn create_plan"));
        assert!(text.contains("struct PlanRequest"));
    }

    #[test]
    fn test_build_function_embedding_text() {
        use crate::neo4j::models::{FunctionNode, Parameter, Visibility};

        let func = FunctionNode {
            name: "plan_implementation".to_string(),
            visibility: Visibility::Public,
            params: vec![
                Parameter {
                    name: "request".to_string(),
                    type_name: Some("PlanRequest".to_string()),
                },
            ],
            return_type: Some("Result<ImplementationPlan>".to_string()),
            generics: vec![],
            is_async: true,
            is_unsafe: false,
            complexity: 10,
            file_path: "src/orchestrator/planner.rs".to_string(),
            line_start: 100,
            line_end: 200,
            docstring: Some("Generate implementation phases from code analysis".to_string()),
        };

        let text = Orchestrator::build_function_embedding_text(&func);
        assert!(text.contains("plan_implementation"));
        assert!(text.contains("request: PlanRequest"));
        assert!(text.contains("-> Result<ImplementationPlan>"));
        assert!(text.contains("Generate implementation phases"));
    }

    #[test]
    fn test_build_function_embedding_text_minimal() {
        use crate::neo4j::models::{FunctionNode, Visibility};

        let func = FunctionNode {
            name: "main".to_string(),
            visibility: Visibility::Private,
            params: vec![],
            return_type: None,
            generics: vec![],
            is_async: false,
            is_unsafe: false,
            complexity: 1,
            file_path: "src/main.rs".to_string(),
            line_start: 1,
            line_end: 3,
            docstring: None,
        };

        let text = Orchestrator::build_function_embedding_text(&func);
        assert_eq!(text, "main()");
    }
}
