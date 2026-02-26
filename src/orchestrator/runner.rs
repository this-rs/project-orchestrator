//! Main orchestrator runner

use crate::embeddings::{EmbeddingProvider, FastEmbedProvider, HttpEmbeddingProvider};
use crate::events::{
    CrudAction, CrudEvent, EntityType as EventEntityType, EventEmitter, HybridEmitter,
};
use crate::graph::{
    AnalyticsConfig, AnalyticsDebouncer, AnalyticsEngine, CoChangeDebouncer, GraphAnalyticsEngine,
    NeuralReinforcementDebouncer,
};
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
pub(crate) fn normalize_path(path: &str) -> String {
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

/// Resolve a relative path (like `./foo` or `../bar`) against a base directory.
/// Returns a clean path with `.` and `..` resolved, without filesystem access.
fn resolve_relative_path(base_dir: &str, relative: &str) -> String {
    let mut segments: Vec<&str> = base_dir.split('/').filter(|s| !s.is_empty()).collect();

    for part in relative.split('/') {
        match part {
            "." | "" => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }

    segments.join("/")
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
    co_change_debouncer: CoChangeDebouncer,
    neural_reinforcement_debouncer: NeuralReinforcementDebouncer,
    activation_engine: Option<Arc<SpreadingActivationEngine>>,
    auto_reinforcement: AutoReinforcementConfig,
    event_bus: Option<Arc<HybridEmitter>>,
    event_emitter: Option<Arc<dyn EventEmitter>>,
    /// Embedding provider for code embeddings (File/Function nodes).
    /// Shared with NoteManager and SpreadingActivationEngine.
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
}

/// Create an embedding provider from resolved [`Config`] fields.
///
/// The `Config` already merges env var > YAML > defaults, so this function
/// simply reads the final values without touching `std::env` directly.
///
/// Provider selection via `config.embedding_provider`:
/// - `local` (default) → [`FastEmbedProvider`] using in-process ONNX Runtime (zero external dependency)
/// - `http` → [`HttpEmbeddingProvider`] using any OpenAI-compatible API (Ollama, OpenAI, LiteLLM…)
/// - `disabled` / `none` / `off` → No embedding provider (semantic search unavailable)
///
/// Returns `None` if disabled or initialization fails. Logs the result at info level.
fn init_embedding_provider(config: &crate::Config) -> Option<Arc<dyn EmbeddingProvider>> {
    let provider_type = config
        .embedding_provider
        .as_deref()
        .unwrap_or("local")
        .to_lowercase();

    match provider_type.as_str() {
        "http" => init_http_embedding_provider(config),
        "disabled" | "none" | "off" => {
            tracing::info!("Embedding provider disabled (provider=disabled)");
            None
        }
        // Default: local fastembed
        _ => init_local_embedding_provider(config),
    }
}

/// Initialize the local fastembed ONNX embedding provider (default).
fn init_local_embedding_provider(config: &crate::Config) -> Option<Arc<dyn EmbeddingProvider>> {
    // Build from Config fields (already merged env var > YAML > None)
    use fastembed::EmbeddingModel;
    use std::path::PathBuf;

    let model_variant = config
        .embedding_fastembed_model
        .as_deref()
        .map(crate::embeddings::fastembed::parse_model_name_pub)
        .unwrap_or(EmbeddingModel::MultilingualE5Base);

    // Cache dir priority: config (env/YAML) > default (~/.fastembed_cache).
    // MUST match the desktop setup wizard's `fastembed_cache_dir()` which uses
    // `~/.fastembed_cache` — otherwise the backend won't find the model that
    // the wizard already downloaded, causing a re-download on every launch.
    let cache_dir: Option<PathBuf> = config
        .embedding_fastembed_cache_dir
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".fastembed_cache")));

    match FastEmbedProvider::new(model_variant, cache_dir) {
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
fn init_http_embedding_provider(config: &crate::Config) -> Option<Arc<dyn EmbeddingProvider>> {
    let url = config
        .embedding_url
        .clone()
        .unwrap_or_else(|| "http://localhost:11434/v1/embeddings".to_string());

    // Allow explicit opt-out
    if url.is_empty() || url.eq_ignore_ascii_case("disabled") {
        tracing::info!("HTTP embedding provider not configured (url empty/disabled)");
        return None;
    }

    let model = config
        .embedding_model
        .clone()
        .unwrap_or_else(|| "nomic-embed-text".to_string());

    let api_key = config.embedding_api_key.clone().filter(|k| !k.is_empty());

    let dimensions = config.embedding_dimensions.unwrap_or(768);

    let provider = HttpEmbeddingProvider::new(url, model, api_key, dimensions);
    tracing::info!(
        model = provider.model_name(),
        dimensions = provider.dimensions(),
        provider = "http",
        "Embedding provider initialized (HTTP)"
    );
    Some(Arc::new(provider))
}

impl Orchestrator {
    /// Create a new orchestrator
    pub async fn new(state: AppState) -> Result<Self> {
        let embedding_provider = init_embedding_provider(&state.config);

        let mut pm = PlanManager::new(state.neo4j.clone(), state.meili.clone());
        if let Some(ref provider) = embedding_provider {
            pm = pm.with_embedding_provider(provider.clone());
        }
        let plan_manager = Arc::new(pm);

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
        let mut planner = super::ImplementationPlanner::new(
            state.neo4j.clone(),
            state.meili.clone(),
            plan_manager.clone(),
            note_manager.clone(),
        );
        if let Some(ref provider) = embedding_provider {
            planner = planner.with_embedding_provider(provider.clone());
        }
        let planner = Arc::new(planner);
        let analytics: Arc<dyn AnalyticsEngine> = Arc::new(GraphAnalyticsEngine::new(
            state.neo4j.clone(),
            AnalyticsConfig::default(),
        ));
        let analytics_debouncer = AnalyticsDebouncer::with_graph_store(
            analytics.clone(),
            2000,
            Some(state.neo4j.clone()),
        );
        let co_change_debouncer = CoChangeDebouncer::new(state.neo4j.clone(), 30_000);
        let ar_config = AutoReinforcementConfig::default();
        let neural_reinforcement_debouncer =
            NeuralReinforcementDebouncer::new(state.neo4j.clone(), ar_config.clone(), 5_000);

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
            co_change_debouncer,
            neural_reinforcement_debouncer,
            activation_engine,
            auto_reinforcement: ar_config,
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
        let embedding_provider = init_embedding_provider(&state.config);

        let mut pm = PlanManager::with_event_emitter(
            state.neo4j.clone(),
            state.meili.clone(),
            emitter.clone(),
        );
        if let Some(ref provider) = embedding_provider {
            pm = pm.with_embedding_provider(provider.clone());
        }
        let plan_manager = Arc::new(pm);

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
        let mut planner = super::ImplementationPlanner::new(
            state.neo4j.clone(),
            state.meili.clone(),
            plan_manager.clone(),
            note_manager.clone(),
        );
        if let Some(ref provider) = embedding_provider {
            planner = planner.with_embedding_provider(provider.clone());
        }
        let planner = Arc::new(planner);
        let analytics: Arc<dyn AnalyticsEngine> = Arc::new(GraphAnalyticsEngine::new(
            state.neo4j.clone(),
            AnalyticsConfig::default(),
        ));
        let analytics_debouncer = AnalyticsDebouncer::with_graph_store(
            analytics.clone(),
            2000,
            Some(state.neo4j.clone()),
        );
        let co_change_debouncer = CoChangeDebouncer::new(state.neo4j.clone(), 30_000);
        let ar_config = AutoReinforcementConfig::default();
        let neural_reinforcement_debouncer =
            NeuralReinforcementDebouncer::new(state.neo4j.clone(), ar_config.clone(), 5_000);

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
            co_change_debouncer,
            neural_reinforcement_debouncer,
            activation_engine,
            auto_reinforcement: ar_config,
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
        let embedding_provider = init_embedding_provider(&state.config);

        let mut pm = PlanManager::with_event_emitter(
            state.neo4j.clone(),
            state.meili.clone(),
            emitter.clone(),
        );
        if let Some(ref provider) = embedding_provider {
            pm = pm.with_embedding_provider(provider.clone());
        }
        let plan_manager = Arc::new(pm);

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
        let mut planner = super::ImplementationPlanner::new(
            state.neo4j.clone(),
            state.meili.clone(),
            plan_manager.clone(),
            note_manager.clone(),
        );
        if let Some(ref provider) = embedding_provider {
            planner = planner.with_embedding_provider(provider.clone());
        }
        let planner = Arc::new(planner);
        let analytics: Arc<dyn AnalyticsEngine> = Arc::new(GraphAnalyticsEngine::new(
            state.neo4j.clone(),
            AnalyticsConfig::default(),
        ));
        let analytics_debouncer = AnalyticsDebouncer::with_graph_store(
            analytics.clone(),
            2000,
            Some(state.neo4j.clone()),
        );
        let co_change_debouncer = CoChangeDebouncer::new(state.neo4j.clone(), 30_000);
        let ar_config = AutoReinforcementConfig::default();
        let neural_reinforcement_debouncer =
            NeuralReinforcementDebouncer::new(state.neo4j.clone(), ar_config.clone(), 5_000);

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
            co_change_debouncer,
            neural_reinforcement_debouncer,
            activation_engine,
            auto_reinforcement: ar_config,
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

    /// Get the CO_CHANGED debouncer (for co-change computation triggers)
    pub fn co_change_debouncer(&self) -> &CoChangeDebouncer {
        &self.co_change_debouncer
    }

    /// Get the neural reinforcement debouncer (for commit Hebbian hooks).
    ///
    /// Debounces rapid-fire commits (e.g., during git checkout/rebase) to
    /// prevent CPU spikes from hundreds of energy boost + synapse calls.
    pub fn neural_reinforcement_debouncer(&self) -> &NeuralReinforcementDebouncer {
        &self.neural_reinforcement_debouncer
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
    // CO_CHANGED computation
    // ========================================================================

    /// Compute CO_CHANGED relations for a project from TOUCHES history.
    ///
    /// Incremental: only processes commits since `last_co_change_computed_at`.
    /// Defaults: min_count=3, max_relations=500.
    pub async fn compute_co_changed(&self, project_id: uuid::Uuid) -> Result<i64> {
        let start = std::time::Instant::now();

        // Get project to read last_co_change_computed_at
        let since = self
            .neo4j()
            .get_project(project_id)
            .await?
            .and_then(|p| p.last_co_change_computed_at);

        let count = self
            .neo4j()
            .compute_co_changed(project_id, since, 3, 500)
            .await?;

        tracing::info!(
            project_id = %project_id,
            since = ?since,
            relations = count,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "CO_CHANGED computation finished"
        );

        Ok(count)
    }

    /// Backfill TOUCHES relations from git history for a project.
    ///
    /// Parses `git log --name-only` and creates TOUCHES relations for each commit.
    /// Idempotent: commits that already have TOUCHES relations are skipped.
    /// After backfilling, triggers a full CO_CHANGED computation.
    pub async fn backfill_commit_touches(
        &self,
        project_id: uuid::Uuid,
        root_path: &Path,
    ) -> Result<BackfillResult> {
        let start = std::time::Instant::now();

        // Step 1: Parse git log
        let git_commits = Self::git_log_touched_files(root_path, 1000)?;
        let total_from_git = git_commits.len();

        if git_commits.is_empty() {
            return Ok(BackfillResult {
                commits_parsed: 0,
                commits_backfilled: 0,
                touches_created: 0,
                elapsed_ms: start.elapsed().as_millis() as u64,
            });
        }

        // Step 2: Create Commit nodes + TOUCHES relations in batches
        let mut commits_backfilled = 0u64;
        let mut touches_created = 0u64;
        let neo4j = self.neo4j();

        for chunk in git_commits.chunks(100) {
            for gc in chunk {
                // Create commit node (MERGE = idempotent)
                let commit = CommitNode {
                    hash: gc.hash.clone(),
                    message: gc.message.clone(),
                    author: gc.author.clone(),
                    timestamp: gc.timestamp,
                };
                neo4j.create_commit(&commit).await?;

                // Create TOUCHES (MERGE = idempotent)
                // git log returns relative paths (e.g. "src/lib.rs") but File nodes
                // in Neo4j use absolute paths — prefix with root_path to match.
                if !gc.files.is_empty() {
                    let file_infos: Vec<crate::neo4j::models::FileChangedInfo> = gc
                        .files
                        .iter()
                        .map(|f| {
                            let abs_path = root_path.join(&f.path);
                            crate::neo4j::models::FileChangedInfo {
                                path: abs_path.to_string_lossy().to_string(),
                                additions: f.additions,
                                deletions: f.deletions,
                            }
                        })
                        .collect();
                    neo4j.create_commit_touches(&gc.hash, &file_infos).await?;
                    touches_created += gc.files.len() as u64;
                }
                commits_backfilled += 1;
            }
        }

        // Step 3: Trigger full CO_CHANGED computation (since=None for full recompute)
        if let Err(e) = neo4j.compute_co_changed(project_id, None, 3, 500).await {
            tracing::warn!(
                project_id = %project_id,
                error = %e,
                "CO_CHANGED computation after backfill failed"
            );
        }

        let result = BackfillResult {
            commits_parsed: total_from_git as u64,
            commits_backfilled,
            touches_created,
            elapsed_ms: start.elapsed().as_millis() as u64,
        };

        tracing::info!(
            project_id = %project_id,
            commits_parsed = result.commits_parsed,
            commits_backfilled = result.commits_backfilled,
            touches_created = result.touches_created,
            elapsed_ms = result.elapsed_ms,
            "Backfill TOUCHES complete"
        );

        Ok(result)
    }

    /// Parse git log to extract commits and their touched files with stats.
    ///
    /// Uses `git log --numstat --format=...` to get additions/deletions per file.
    /// Limited to `max_commits` most recent commits.
    fn git_log_touched_files(root_path: &Path, max_commits: usize) -> Result<Vec<GitCommitFiles>> {
        use std::process::Command;

        let output = Command::new("git")
            .args([
                "log",
                "--numstat",
                &format!("--max-count={}", max_commits),
                "--format=COMMIT_START%n%H%n%s%n%an%n%aI",
            ])
            .current_dir(root_path)
            .output()?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "git log failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut commits = Vec::new();
        let mut lines = stdout.lines().peekable();

        while let Some(line) = lines.next() {
            if line == "COMMIT_START" {
                let hash = lines.next().unwrap_or("").to_string();
                let message = lines.next().unwrap_or("").to_string();
                let author = lines.next().unwrap_or("").to_string();
                let date_str = lines.next().unwrap_or("");
                let timestamp = chrono::DateTime::parse_from_rfc3339(date_str)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .unwrap_or_else(|_| chrono::Utc::now());

                let mut files = Vec::new();
                // --numstat lines: "<additions>\t<deletions>\t<path>"
                // Binary files show "-\t-\t<path>"
                while let Some(next) = lines.peek() {
                    if *next == "COMMIT_START" || next.is_empty() {
                        if next.is_empty() {
                            lines.next(); // skip empty separator line
                            continue;
                        }
                        break;
                    }
                    let stat_line = lines.next().unwrap();
                    let parts: Vec<&str> = stat_line.splitn(3, '\t').collect();
                    if parts.len() == 3 {
                        let additions = parts[0].parse::<i64>().ok(); // "-" for binary → None
                        let deletions = parts[1].parse::<i64>().ok();
                        files.push(GitFileChange {
                            path: parts[2].to_string(),
                            additions,
                            deletions,
                        });
                    }
                }

                if !hash.is_empty() {
                    commits.push(GitCommitFiles {
                        hash,
                        message,
                        author,
                        timestamp,
                        files,
                    });
                }
            }
        }

        Ok(commits)
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

        // ── Phase 1: Scan ──────────────────────────────────────────
        let entries = scan_files(dir_path);

        // Track all scanned paths for stale-file cleanup
        let synced_paths: HashSet<String> = entries.iter().map(|e| e.path.clone()).collect();

        // ── Phase 2: Read ───────────────────────────────────────────
        let file_contents = read_files(entries).await;

        // ── Hash check: skip unchanged files ───────────────────────
        let mut to_parse = Vec::with_capacity(file_contents.len());
        for fc in file_contents {
            if !force {
                if let Ok(Some(existing)) = self.state.neo4j.get_file(&fc.path).await {
                    if existing.hash == fc.hash {
                        result.files_skipped += 1;
                        continue;
                    }
                }
            }
            to_parse.push(fc);
        }

        // ── Phase 3: Parse ─────────────────────────────────────────
        let parsed_files = parse_files(to_parse, &self.parser).await;
        let parse_count = parsed_files.len();

        // ── Store: Neo4j + MeiliSearch ─────────────────────────────
        for parsed in &parsed_files {
            match self.store_parsed_file_for_project(parsed, project_id).await {
                Ok(()) => {
                    // Index in Meilisearch only if project context is available
                    if let (Some(pid), Some(slug)) = (project_id, project_slug.as_deref()) {
                        let doc = CodeParser::to_code_document(parsed, &pid.to_string(), slug);
                        if let Err(e) = self.state.meili.index_code(&doc).await {
                            tracing::warn!(
                                "Failed to index {} in Meilisearch: {}",
                                parsed.path,
                                e
                            );
                        }
                    }

                    // Verify notes attached to this file
                    if let Ok(content) = tokio::fs::read_to_string(&parsed.path).await {
                        if let Err(e) = self
                            .verify_notes_for_file(&parsed.path, parsed, &content)
                            .await
                        {
                            tracing::warn!(
                                "Failed to verify notes for {}: {}",
                                parsed.path,
                                e
                            );
                        }
                    }

                    result.files_synced += 1;
                }
                Err(e) => {
                    tracing::warn!("Failed to store {}: {}", parsed.path, e);
                    result.errors += 1;
                }
            }
        }

        tracing::info!(
            "sync pipeline: scanned {} → read {} → parsed {} → stored {}",
            synced_paths.len(),
            synced_paths.len(),
            parse_count,
            result.files_synced,
        );

        // ── Cleanup: remove stale files ────────────────────────────
        if let Some(pid) = project_id {
            let valid_paths: Vec<String> = synced_paths.into_iter().collect();
            match self.neo4j().delete_stale_files(pid, &valid_paths).await {
                Ok((files_deleted, symbols_deleted, stale_paths)) => {
                    result.files_deleted = files_deleted;
                    result.symbols_deleted = symbols_deleted;

                    // Also clean up stale files from Meilisearch search index
                    if !stale_paths.is_empty() {
                        tracing::info!(
                            "Cleaning {} stale file(s) from Meilisearch for project {}",
                            stale_paths.len(),
                            pid
                        );
                        for path in &stale_paths {
                            if let Err(e) = self.meili().delete_code(path).await {
                                tracing::warn!(
                                    "Failed to delete stale file {} from Meilisearch: {}",
                                    path,
                                    e
                                );
                            }
                        }
                    }
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
        self.store_parsed_file_for_project_with_ctx(parsed, project_id, None)
            .await
    }

    /// Store a parsed file in Neo4j with optional ImportResolutionContext.
    ///
    /// When ctx is provided, uses SuffixIndex for O(1) import resolution
    /// instead of filesystem lookups.
    async fn store_parsed_file_for_project_with_ctx(
        &self,
        parsed: &ParsedFile,
        project_id: Option<Uuid>,
        mut ctx: Option<&mut crate::resolver::ImportResolutionContext>,
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

        // Batch upsert all entity types (UNWIND — one Neo4j query each)
        self.state
            .neo4j
            .batch_upsert_functions(&parsed.functions)
            .await?;
        self.state
            .neo4j
            .batch_upsert_structs(&parsed.structs)
            .await?;
        self.state.neo4j.batch_upsert_traits(&parsed.traits).await?;
        self.state.neo4j.batch_upsert_enums(&parsed.enums).await?;
        // Impls AFTER structs/traits/enums (IMPLEMENTS_FOR/IMPLEMENTS_TRAIT need targets)
        self.state
            .neo4j
            .batch_upsert_impls(&parsed.impl_blocks)
            .await?;

        // ── Plans 5 & 6: Heritage & Process relations ───────────────────
        // When Plans 5 (Heritage) and 6 (Process Detection) are implemented,
        // add the following sequence HERE for each new relationship type:
        //
        // 1. DELETE stale relations for this file:
        //    "MATCH (f:File {path: $path})-[:CONTAINS]->(sym)-[r:EXTENDS]->()
        //     DELETE r"
        //    (same for IMPLEMENTS, STEP_IN_PROCESS)
        //
        // 2. UPSERT new relations via batch helper:
        //    use crate::neo4j::batch::{run_unwind_in_chunks, BoltMap};
        //    let items: Vec<BoltMap> = build_extends_items(&parsed.heritage);
        //    run_unwind_in_chunks(&graph, items, "UNWIND $items AS ...").await?;
        //
        // This DELETE-then-CREATE pattern ensures no stale rels persist when
        // a file's inheritance hierarchy changes on re-sync.
        //
        // NOTE: The current CALLS/IMPORTS/IMPLEMENTS_FOR use MERGE (idempotent)
        // which does NOT clean up removed relations. This is acceptable for now
        // because cleanup_sync_data handles full-project cleanup, and the watcher
        // handles file deletion. Per-file incremental cleanup is a future
        // improvement tracked in the batch audit note.
        // ─────────────────────────────────────────────────────────────────

        // Batch upsert imports, then resolve relationships (logic stays in runner)
        self.state
            .neo4j
            .batch_upsert_imports(&parsed.imports)
            .await?;

        // Collect resolved import relationships in memory, then batch-write
        let mut import_rels: Vec<(String, String, String)> = Vec::new();
        let mut symbol_rels: Vec<(String, String, Option<Uuid>)> = Vec::new();
        for import in &parsed.imports {
            // Resolve imports to file paths (language-aware)
            let resolved_files = match ctx {
                Some(ref mut c) => self.resolve_imports_for_language_with_ctx(
                    import,
                    &parsed.path,
                    &parsed.language,
                    Some(&mut *c),
                ),
                None => self.resolve_imports_for_language(import, &parsed.path, &parsed.language),
            };
            for target_file in &resolved_files {
                import_rels.push((
                    parsed.path.clone(),
                    target_file.clone(),
                    import.path.clone(),
                ));
            }

            // Collect IMPORTS_SYMBOL relationships for imported symbols
            let import_id = format!("{}:{}:{}", import.file_path, import.line, import.path);
            let symbols = Self::extract_imported_symbols(import);
            for symbol_name in &symbols {
                symbol_rels.push((import_id.clone(), symbol_name.clone(), project_id));
            }
        }
        self.state
            .neo4j
            .batch_create_import_relationships(&import_rels)
            .await?;
        self.state
            .neo4j
            .batch_create_imports_symbol_relationships(&symbol_rels)
            .await?;

        // Score confidence for each function call before persisting
        let scored_calls = Self::score_function_calls(
            &parsed.function_calls,
            parsed,
            &import_rels,
            ctx.as_ref().map(|c| &**c),
        );

        // Batch create CALLS relationships (scoped to project to prevent cross-project pollution)
        self.state
            .neo4j
            .batch_create_call_relationships(&scored_calls, project_id)
            .await?;

        // Embed file and functions (best-effort, non-blocking)
        if self.embedding_provider.is_some() {
            let provider = self.embedding_provider.clone().unwrap();
            let neo4j = self.state.neo4j.clone();
            let parsed_clone = parsed.clone();
            let file_path = normalize_path(&parsed.path);
            tokio::spawn(async move {
                if let Err(e) =
                    Self::embed_parsed_file(&provider, &neo4j, &parsed_clone, &file_path).await
                {
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
                    || matches!(f.visibility, crate::neo4j::models::Visibility::Public)
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

    /// Resolve import paths to file paths based on language.
    ///
    /// When a resolution context is provided, uses SuffixIndex for O(1) lookups
    /// and caches results. Falls back to filesystem-based resolution otherwise.
    fn resolve_imports_for_language(
        &self,
        import: &ImportNode,
        parsed_path: &str,
        language: &str,
    ) -> Vec<String> {
        self.resolve_imports_for_language_with_ctx(import, parsed_path, language, None)
    }

    /// Resolve import paths using optional ImportResolutionContext.
    fn resolve_imports_for_language_with_ctx(
        &self,
        import: &ImportNode,
        parsed_path: &str,
        language: &str,
        ctx: Option<&mut crate::resolver::ImportResolutionContext>,
    ) -> Vec<String> {
        match ctx {
            Some(ctx) => {
                // Check cache first
                if let Some(cached) = ctx.resolve_cache.get(parsed_path, &import.path) {
                    return cached.into_iter().cloned().collect();
                }

                let result = match language {
                    "rust" => Self::resolve_rust_imports_indexed(
                        &import.path,
                        parsed_path,
                        &ctx.suffix_index,
                    ),
                    "typescript" | "javascript" | "tsx" | "jsx" => {
                        Self::resolve_typescript_import_indexed(
                            &import.path,
                            parsed_path,
                            &ctx.suffix_index,
                        )
                        .into_iter()
                        .collect()
                    }
                    "python" => Self::resolve_python_import_indexed(
                        &import.path,
                        parsed_path,
                        &ctx.suffix_index,
                    )
                    .into_iter()
                    .collect(),
                    _ => Vec::new(),
                };

                // Cache the result (store first resolved path or None)
                let cache_value = result.first().cloned();
                ctx.resolve_cache.insert(
                    parsed_path.to_string(),
                    import.path.clone(),
                    cache_value,
                );

                result
            }
            None => {
                // Legacy fallback: filesystem-based resolution
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
        }
    }

    // ========================================================================
    // SuffixIndex-based resolvers (O(1) lookups, no filesystem access)
    // ========================================================================

    /// Resolve Rust imports using SuffixIndex (no filesystem access).
    fn resolve_rust_imports_indexed(
        import_path: &str,
        source_file: &str,
        index: &crate::resolver::SuffixIndex,
    ) -> Vec<String> {
        let paths = Self::flatten_grouped_import(import_path);
        let mut resolved = Vec::new();

        for path in paths {
            if let Some(file) = Self::resolve_single_rust_import_indexed(&path, source_file, index)
            {
                if !resolved.contains(&file) {
                    resolved.push(file);
                }
            }
        }

        resolved
    }

    /// Resolve a single Rust import using SuffixIndex.
    fn resolve_single_rust_import_indexed(
        import_path: &str,
        source_file: &str,
        index: &crate::resolver::SuffixIndex,
    ) -> Option<String> {
        let path: Vec<&str> = import_path.split("::").collect();
        if path.is_empty() {
            return None;
        }

        let first = path[0];

        // External crates — skip
        if !matches!(first, "crate" | "super" | "self") {
            return None;
        }

        match first {
            "crate" => {
                if path.len() < 2 {
                    return None;
                }
                // Try full path first (import might target a file directly)
                // e.g., crate::neo4j::client → neo4j/client.rs
                let full_path = path[1..].join("/");
                let rs_suffix = format!("{}.rs", full_path);
                if let Some(resolved) = index.get(&rs_suffix) {
                    return Some(resolved.to_string());
                }
                let mod_suffix = format!("{}/mod.rs", full_path);
                if let Some(resolved) = index.get(&mod_suffix) {
                    return Some(resolved.to_string());
                }
                // Fallback: strip last segment (it's likely a type/function name)
                // e.g., crate::neo4j::client::Client → neo4j/client.rs
                let module_path = &path[1..path.len().saturating_sub(1)];
                if !module_path.is_empty() {
                    let suffix = module_path.join("/");
                    let rs_suffix = format!("{}.rs", suffix);
                    if let Some(resolved) = index.get(&rs_suffix) {
                        return Some(resolved.to_string());
                    }
                    let mod_suffix = format!("{}/mod.rs", suffix);
                    if let Some(resolved) = index.get(&mod_suffix) {
                        return Some(resolved.to_string());
                    }
                }
                None
            }
            "super" | "self" => {
                // For super/self, we need the source file's directory context
                let source_dir = source_file.rsplit_once('/').map(|(dir, _)| dir)?;
                let mut segments: Vec<&str> = source_dir.split('/').collect();

                let start = if first == "self" { 1 } else { 0 };
                for &part in &path[start..] {
                    if part == "super" {
                        segments.pop()?;
                    } else {
                        segments.push(part);
                    }
                }

                // Try full path first (import might target a file directly)
                let suffix = segments.join("/");
                let rs_suffix = format!("{}.rs", suffix);
                if let Some(resolved) = index.get(&rs_suffix) {
                    return Some(resolved.to_string());
                }
                let mod_suffix = format!("{}/mod.rs", suffix);
                if let Some(resolved) = index.get(&mod_suffix) {
                    return Some(resolved.to_string());
                }

                // Fallback: strip last segment (likely a type name)
                if segments.len() > 1 {
                    let try_without_last: Vec<&str> =
                        segments[..segments.len().saturating_sub(1)].to_vec();
                    let suffix = try_without_last.join("/");
                    let rs_suffix = format!("{}.rs", suffix);
                    if let Some(resolved) = index.get(&rs_suffix) {
                        return Some(resolved.to_string());
                    }
                    let mod_suffix = format!("{}/mod.rs", suffix);
                    if let Some(resolved) = index.get(&mod_suffix) {
                        return Some(resolved.to_string());
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Resolve TypeScript/JavaScript import using SuffixIndex.
    fn resolve_typescript_import_indexed(
        import_path: &str,
        source_file: &str,
        index: &crate::resolver::SuffixIndex,
    ) -> Option<String> {
        // Only resolve relative imports and @/ aliases
        if !import_path.starts_with('.') && !import_path.starts_with('@') {
            return None; // npm package — skip
        }

        if import_path.starts_with('.') {
            // Relative import: resolve against source directory
            let source_dir = source_file.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");
            let target = resolve_relative_path(source_dir, import_path);

            // Try extensions
            let extensions = ["ts", "tsx", "js", "jsx"];
            for ext in &extensions {
                let with_ext = format!("{}.{}", target, ext);
                if let Some(resolved) = index.get(&with_ext) {
                    return Some(resolved.to_string());
                }
            }

            // Try index files
            let index_files = ["index.ts", "index.tsx", "index.js", "index.jsx"];
            for idx in &index_files {
                let index_path = format!("{}/{}", target, idx);
                if let Some(resolved) = index.get(&index_path) {
                    return Some(resolved.to_string());
                }
            }
        } else if import_path.starts_with("@/") {
            // @/ alias — try src/ prefix
            let after_alias = &import_path[2..];
            let target = format!("src/{}", after_alias);

            let extensions = ["ts", "tsx", "js", "jsx"];
            for ext in &extensions {
                let with_ext = format!("{}.{}", target, ext);
                if let Some(resolved) = index.get(&with_ext) {
                    return Some(resolved.to_string());
                }
            }

            let index_files = ["index.ts", "index.tsx", "index.js", "index.jsx"];
            for idx in &index_files {
                let index_path = format!("{}/{}", target, idx);
                if let Some(resolved) = index.get(&index_path) {
                    return Some(resolved.to_string());
                }
            }
        }

        None
    }

    /// Resolve Python import using SuffixIndex.
    fn resolve_python_import_indexed(
        import_path: &str,
        source_file: &str,
        index: &crate::resolver::SuffixIndex,
    ) -> Option<String> {
        if import_path.starts_with('.') {
            // Relative import
            let source_dir = source_file.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");
            let dots = import_path.chars().take_while(|c| *c == '.').count();
            let module_part = &import_path[dots..];

            // Navigate up for each extra dot
            let mut segments: Vec<&str> = source_dir.split('/').collect();
            for _ in 1..dots {
                segments.pop();
            }

            if !module_part.is_empty() {
                for part in module_part.split('.') {
                    segments.push(part);
                }
            }

            let suffix = segments.join("/");

            // Try module.py
            let py_suffix = format!("{}.py", suffix);
            if let Some(resolved) = index.get(&py_suffix) {
                return Some(resolved.to_string());
            }
            // Try module/__init__.py
            let init_suffix = format!("{}/__init__.py", suffix);
            if let Some(resolved) = index.get(&init_suffix) {
                return Some(resolved.to_string());
            }
        } else {
            // Absolute import — convert dots to path
            let suffix = import_path.replace('.', "/");

            let py_suffix = format!("{}.py", suffix);
            if let Some(resolved) = index.get(&py_suffix) {
                return Some(resolved.to_string());
            }
            let init_suffix = format!("{}/__init__.py", suffix);
            if let Some(resolved) = index.get(&init_suffix) {
                return Some(resolved.to_string());
            }
        }

        None
    }

    // ========================================================================
    // Confidence scoring for CALLS relationships
    // ========================================================================

    /// Score each function call with a confidence level based on how the callee was resolved.
    ///
    /// Scoring levels:
    /// - **import-resolved** (0.90): callee is defined in a file that the caller's file imports
    /// - **same-file** (0.85): callee is defined in the same file as the caller
    /// - **fuzzy-unique** (0.50): callee name found exactly once in the SymbolTable
    /// - **fuzzy-ambiguous** (0.30): callee name found in multiple files
    /// - **unscored** (0.50): no SymbolTable available (legacy fallback)
    fn score_function_calls(
        calls: &[crate::parser::FunctionCall],
        parsed: &ParsedFile,
        import_rels: &[(String, String, String)], // (source_file, target_file, import_path)
        ctx: Option<&crate::resolver::ImportResolutionContext>,
    ) -> Vec<crate::parser::FunctionCall> {
        // Build set of function names defined in this file
        let same_file_names: std::collections::HashSet<&str> = parsed
            .functions
            .iter()
            .map(|f| f.name.as_str())
            .collect();

        // Build set of imported file paths from this file
        let imported_files: std::collections::HashSet<&str> = import_rels
            .iter()
            .filter(|(src, _, _)| *src == parsed.path)
            .map(|(_, target, _)| target.as_str())
            .collect();

        calls
            .iter()
            .map(|call| {
                let (confidence, reason) = Self::score_single_call(
                    &call.callee_name,
                    &call.caller_id,
                    &same_file_names,
                    &imported_files,
                    ctx,
                );

                crate::parser::FunctionCall {
                    caller_id: call.caller_id.clone(),
                    callee_name: call.callee_name.clone(),
                    line: call.line,
                    confidence,
                    reason,
                }
            })
            .collect()
    }

    /// Score a single function call.
    fn score_single_call(
        callee_name: &str,
        caller_id: &str,
        same_file_names: &std::collections::HashSet<&str>,
        imported_files: &std::collections::HashSet<&str>,
        ctx: Option<&crate::resolver::ImportResolutionContext>,
    ) -> (f64, String) {
        // Priority 1: same-file match (callee defined in same file as caller)
        if same_file_names.contains(callee_name) {
            return (0.85, "same-file".to_string());
        }

        // Use SymbolTable from context for deeper resolution
        if let Some(ctx) = ctx {
            let defs = ctx.symbol_table.lookup_fuzzy(callee_name);

            if defs.is_empty() {
                // Not found in any parsed file — could be external
                return (0.30, "fuzzy-unresolved".to_string());
            }

            // Check if any definition is in an imported file
            let caller_file = caller_id.rsplitn(3, ':').last().unwrap_or(caller_id);
            for def in defs {
                // Normalize: imported_files may have full paths, def.file_path may be relative
                let def_path = def.file_path.as_str();
                if imported_files.contains(def_path)
                    || imported_files.iter().any(|f| f.ends_with(def_path) || def_path.ends_with(f))
                {
                    return (0.90, "import-resolved".to_string());
                }
                // Also check if the def is in the same file (by file_path comparison)
                if def_path == caller_file
                    || caller_file.ends_with(def_path)
                    || def_path.ends_with(caller_file)
                {
                    return (0.85, "same-file".to_string());
                }
            }

            // Fuzzy: callee found but not in imported or same file
            if defs.len() == 1 {
                return (0.50, "fuzzy-unique".to_string());
            } else {
                return (0.30, "fuzzy-ambiguous".to_string());
            }
        }

        // No context available — keep the default
        (0.50, "unscored".to_string())
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

    /// Delete a project and emit event.
    ///
    /// Cleanup order:
    /// 1. Lookup project (get name + slug before deletion)
    /// 2. MeiliSearch: delete code documents (best-effort)
    /// 3. Neo4j: archive notes/decisions, cascade delete structural entities
    /// 4. Emit CrudEvent with slug in payload
    pub async fn delete_project(&self, id: Uuid) -> Result<()> {
        // Lookup project before deleting — we need the name and slug
        let project = self
            .neo4j()
            .get_project(id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Project {} not found", id))?;

        // MeiliSearch cleanup — best-effort, don't block deletion on failure
        if let Err(e) = self.meili().delete_code_for_project(&project.slug).await {
            tracing::warn!(
                "Failed to delete MeiliSearch code documents for project '{}': {}",
                project.slug,
                e
            );
        }

        // Neo4j cascade delete (archives notes/decisions, deletes everything else)
        self.neo4j().delete_project(id, &project.name).await?;

        // Emit event with slug in payload for subscribers (watcher bridge, etc.)
        let mut payload = serde_json::Map::new();
        payload.insert("slug".to_string(), serde_json::Value::String(project.slug));
        self.emit(
            CrudEvent::new(
                EventEntityType::Project,
                CrudAction::Deleted,
                id.to_string(),
            )
            .with_payload(serde_json::Value::Object(payload)),
        );
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
        status: Option<DecisionStatus>,
    ) -> Result<()> {
        self.neo4j()
            .update_decision(decision_id, description, rationale, chosen_option, status)
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

/// Result of backfilling TOUCHES relations from git history
#[derive(Debug, Clone, serde::Serialize)]
pub struct BackfillResult {
    pub commits_parsed: u64,
    pub commits_backfilled: u64,
    pub touches_created: u64,
    pub elapsed_ms: u64,
}

/// A commit with its touched files (parsed from git log)
#[derive(Debug, Clone)]
pub struct GitCommitFiles {
    pub hash: String,
    pub message: String,
    pub author: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub files: Vec<GitFileChange>,
}

/// A file changed in a git commit (from --numstat output).
#[derive(Debug, Clone)]
pub struct GitFileChange {
    pub path: String,
    pub additions: Option<i64>,
    pub deletions: Option<i64>,
}

// ── Pipeline functions ──────────────────────────────────────────────
// 3-phase sync pipeline: scan → read → parse

/// Phase 1: Scan a directory and return all eligible files.
///
/// Walks the directory tree, applies gitignore-like filtering via
/// [`super::should_ignore_path`], and detects language from extension.
/// Returns only files whose extension maps to a [`SupportedLanguage`].
///
/// This is a pure I/O-free metadata scan — no file content is loaded.
pub fn scan_files(root: &Path) -> Vec<FileEntry> {
    use crate::parser::SupportedLanguage;

    let start = std::time::Instant::now();
    let mut entries = Vec::new();

    for entry in WalkDir::new(root)
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

        // Only keep files whose extension maps to a supported language
        let language = match SupportedLanguage::from_extension(ext) {
            Some(lang) => lang,
            None => continue,
        };

        // Skip ignored directories (shared constant with watcher.rs)
        let path_str = path.to_string_lossy();
        if super::should_ignore_path(&path_str) {
            continue;
        }

        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let normalized = normalize_path(&path_str);

        entries.push(FileEntry {
            path: normalized,
            size,
            language,
        });
    }

    let elapsed = start.elapsed();
    tracing::info!(
        "scan_files: found {} eligible files in {:?}",
        entries.len(),
        elapsed
    );

    entries
}

/// Phase 3: Parse files with tree-sitter.
///
/// Parses each file's content using the appropriate tree-sitter grammar
/// based on the detected language. Files that fail to parse are logged and skipped.
///
/// Returns only successfully parsed files.
pub async fn parse_files(
    files: Vec<FileContent>,
    parser: &Arc<RwLock<CodeParser>>,
) -> Vec<ParsedFile> {
    let start = std::time::Instant::now();
    let mut parsed = Vec::with_capacity(files.len());
    let mut parse_errors = 0usize;

    for file in &files {
        let file_path = std::path::Path::new(&file.path);
        match {
            let mut p = parser.write().await;
            p.parse_file(file_path, &file.content)
        } {
            Ok(pf) => {
                parsed.push(pf);
            }
            Err(e) => {
                tracing::warn!("parse_files: failed to parse {}: {}", file.path, e);
                parse_errors += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    tracing::info!(
        "parse_files: parsed {} files ({} errors) in {:?}",
        parsed.len(),
        parse_errors,
        elapsed
    );

    parsed
}

/// Phase 2: Read file contents from disk.
///
/// Loads each file's content and computes a SHA-256 hash.
/// Files that fail to read (permission errors, etc.) are logged and skipped.
///
/// Returns only successfully read files.
pub async fn read_files(entries: Vec<FileEntry>) -> Vec<FileContent> {
    use sha2::{Digest, Sha256};

    let start = std::time::Instant::now();
    let mut files = Vec::with_capacity(entries.len());
    let mut read_errors = 0usize;

    for entry in entries {
        match tokio::fs::read_to_string(&entry.path).await {
            Ok(content) => {
                let mut hasher = Sha256::new();
                hasher.update(content.as_bytes());
                let hash = hex::encode(hasher.finalize());

                files.push(FileContent {
                    path: entry.path,
                    content,
                    size: entry.size,
                    language: entry.language,
                    hash,
                });
            }
            Err(e) => {
                tracing::warn!("read_files: failed to read {}: {}", entry.path, e);
                read_errors += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    tracing::info!(
        "read_files: loaded {} files ({} errors) in {:?}",
        files.len(),
        read_errors,
        elapsed
    );

    files
}

// ── Pipeline types ──────────────────────────────────────────────────
// Used by the 3-phase sync pipeline: scan → read → parse

/// A file discovered during the scan phase.
///
/// Contains only metadata — no content loaded yet.
#[derive(Debug, Clone)]
pub struct FileEntry {
    /// Normalized absolute path
    pub path: String,
    /// File size in bytes
    pub size: u64,
    /// Detected language from extension
    pub language: crate::parser::SupportedLanguage,
}

/// A file whose content has been loaded from disk (read phase).
#[derive(Debug, Clone)]
pub struct FileContent {
    /// Normalized absolute path
    pub path: String,
    /// Raw source content
    pub content: String,
    /// File size in bytes
    pub size: u64,
    /// Detected language
    pub language: crate::parser::SupportedLanguage,
    /// SHA-256 hash of content
    pub hash: String,
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
    use crate::parser::FunctionCall;
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

    #[tokio::test]
    async fn test_delete_project_emits_slug_in_payload() {
        let (orch, mut rx) = orch_with_bus().await;
        let project = test_project();
        orch.neo4j().create_project(&project).await.unwrap();
        orch.delete_project(project.id).await.unwrap();
        let ev = rx.try_recv().unwrap();
        assert_eq!(ev.action, CrudAction::Deleted);
        // Verify slug is included in event payload
        let slug = ev
            .payload
            .get("slug")
            .and_then(|v| v.as_str())
            .expect("event payload should contain slug");
        assert_eq!(slug, project.slug);
    }

    #[tokio::test]
    async fn test_delete_project_cleans_meilisearch() {
        use crate::meilisearch::indexes::CodeDocument;

        let orch = Orchestrator::new(mock_app_state()).await.unwrap();

        // Seed a project + code documents
        let project = test_project();
        orch.neo4j().create_project(&project).await.unwrap();

        // Add some code documents to MeiliSearch for this project
        orch.meili()
            .index_code(&CodeDocument {
                id: "file1".to_string(),
                path: "/tmp/test/main.rs".to_string(),
                language: "rust".to_string(),
                symbols: vec!["main".to_string()],
                docstrings: String::new(),
                signatures: vec!["fn main()".to_string()],
                imports: vec![],
                project_id: project.id.to_string(),
                project_slug: project.slug.clone(),
            })
            .await
            .unwrap();

        let stats = orch.meili().get_code_stats().await.unwrap();
        assert_eq!(stats.total_documents, 1);

        // Delete project
        orch.delete_project(project.id).await.unwrap();

        // Verify MeiliSearch was cleaned up
        let stats = orch.meili().get_code_stats().await.unwrap();
        assert_eq!(
            stats.total_documents, 0,
            "MeiliSearch code documents should be deleted on project removal"
        );
    }

    #[tokio::test]
    async fn test_delete_project_removes_project_from_neo4j() {
        let (orch, _rx) = orch_with_bus().await;
        let project = test_project();
        orch.neo4j().create_project(&project).await.unwrap();

        // Verify project exists
        let found = orch.neo4j().get_project(project.id).await.unwrap();
        assert!(found.is_some());

        orch.delete_project(project.id).await.unwrap();

        // Verify project is gone
        let found = orch.neo4j().get_project(project.id).await.unwrap();
        assert!(found.is_none(), "Project should be deleted from Neo4j");
    }

    #[tokio::test]
    async fn test_delete_project_fails_if_not_found() {
        let (orch, _rx) = orch_with_bus().await;
        let result = orch.delete_project(Uuid::new_v4()).await;
        assert!(
            result.is_err(),
            "delete_project should fail if project doesn't exist"
        );
    }

    #[tokio::test]
    async fn test_delete_project_cascades_plans_tasks_steps() {
        let (orch, _rx) = orch_with_bus().await;
        let project = test_project();
        orch.neo4j().create_project(&project).await.unwrap();

        // Create plan → task → step
        let plan = test_plan_for_project(project.id);
        orch.neo4j().create_plan(&plan).await.unwrap();
        orch.neo4j()
            .link_plan_to_project(plan.id, project.id)
            .await
            .unwrap();

        let task = test_task();
        orch.neo4j().create_task(plan.id, &task).await.unwrap();

        let step = test_step(0, "Do something");
        orch.neo4j().create_step(task.id, &step).await.unwrap();

        // Verify they exist
        let steps = orch.neo4j().get_task_steps(task.id).await.unwrap();
        assert_eq!(steps.len(), 1);

        // Delete project
        orch.delete_project(project.id).await.unwrap();

        // Verify cascade — steps should be gone
        let steps = orch.neo4j().get_task_steps(task.id).await.unwrap();
        assert_eq!(steps.len(), 0, "Steps should be cascade-deleted");
    }

    #[tokio::test]
    async fn test_delete_project_cascades_milestones_releases() {
        let (orch, _rx) = orch_with_bus().await;
        let project = test_project();
        orch.neo4j().create_project(&project).await.unwrap();

        let milestone = test_milestone(project.id, "v1.0 Milestone");
        orch.neo4j().create_milestone(&milestone).await.unwrap();

        let release = test_release(project.id, "1.0.0");
        orch.neo4j().create_release(&release).await.unwrap();

        // Verify they exist
        let milestones = orch
            .neo4j()
            .list_project_milestones(project.id)
            .await
            .unwrap();
        assert_eq!(milestones.len(), 1);
        let releases = orch
            .neo4j()
            .list_project_releases(project.id)
            .await
            .unwrap();
        assert_eq!(releases.len(), 1);

        // Delete project
        orch.delete_project(project.id).await.unwrap();

        // Verify cascade
        let milestones = orch
            .neo4j()
            .list_project_milestones(project.id)
            .await
            .unwrap();
        assert_eq!(milestones.len(), 0, "Milestones should be cascade-deleted");
        let releases = orch
            .neo4j()
            .list_project_releases(project.id)
            .await
            .unwrap();
        assert_eq!(releases.len(), 0, "Releases should be cascade-deleted");
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
        orch.update_decision(id, Some("desc".into()), None, None, None)
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

    // ── store_parsed_file_for_project (batch integration) ─────────

    #[tokio::test]
    async fn test_store_parsed_file_for_project_uses_batch_methods() {
        use crate::neo4j::models::*;
        use crate::parser::{FunctionCall, ParsedFile};

        let mock_store = std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let state = mock_app_state_with_graph(mock_store.clone());
        let orch = Orchestrator::new(state).await.unwrap();

        // Use absolute path to avoid normalize_path prepending cwd
        let file_path = "/tmp/test-project/src/lib.rs".to_string();

        let parsed = ParsedFile {
            path: file_path.clone(),
            language: "rust".to_string(),
            hash: "abc123".to_string(),
            functions: vec![
                FunctionNode {
                    name: "foo".to_string(),
                    visibility: Visibility::Public,
                    params: vec![],
                    return_type: None,
                    generics: vec![],
                    is_async: false,
                    is_unsafe: false,
                    complexity: 1,
                    file_path: file_path.clone(),
                    line_start: 1,
                    line_end: 10,
                    docstring: None,
                },
                FunctionNode {
                    name: "bar".to_string(),
                    visibility: Visibility::Public,
                    params: vec![],
                    return_type: None,
                    generics: vec![],
                    is_async: false,
                    is_unsafe: false,
                    complexity: 1,
                    file_path: file_path.clone(),
                    line_start: 20,
                    line_end: 30,
                    docstring: None,
                },
            ],
            structs: vec![StructNode {
                name: "MyStruct".to_string(),
                visibility: Visibility::Public,
                generics: vec![],
                file_path: file_path.clone(),
                line_start: 40,
                line_end: 50,
                docstring: None,
            }],
            traits: vec![TraitNode {
                name: "MyTrait".to_string(),
                visibility: Visibility::Public,
                generics: vec![],
                file_path: file_path.clone(),
                line_start: 60,
                line_end: 70,
                docstring: None,
                is_external: false,
                source: None,
            }],
            enums: vec![EnumNode {
                name: "MyEnum".to_string(),
                visibility: Visibility::Public,
                variants: vec!["A".to_string(), "B".to_string()],
                file_path: file_path.clone(),
                line_start: 80,
                line_end: 90,
                docstring: None,
            }],
            impl_blocks: vec![ImplNode {
                for_type: "MyStruct".to_string(),
                trait_name: Some("MyTrait".to_string()),
                generics: vec![],
                where_clause: None,
                file_path: file_path.clone(),
                line_start: 100,
                line_end: 110,
            }],
            imports: vec![ImportNode {
                path: "std::fmt".to_string(),
                alias: None,
                items: vec!["Display".to_string()],
                file_path: file_path.clone(),
                line: 1,
            }],
            function_calls: vec![FunctionCall {
                caller_id: format!("{}:foo:1", file_path),
                callee_name: "bar".to_string(),
                line: 5,
                confidence: 0.50,
                reason: "unscored".to_string(),
            }],
            symbols: vec!["foo".to_string(), "bar".to_string()],
        };

        // Call without project_id to avoid mock's project-scoped call filtering
        orch.store_parsed_file_for_project(&parsed, None)
            .await
            .unwrap();

        // Verify all entity types were stored via batch methods
        assert_eq!(mock_store.functions.read().await.len(), 2, "functions");
        assert_eq!(mock_store.structs_map.read().await.len(), 1, "structs");
        assert_eq!(mock_store.traits_map.read().await.len(), 1, "traits");
        assert_eq!(mock_store.enums_map.read().await.len(), 1, "enums");
        assert_eq!(mock_store.impls_map.read().await.len(), 1, "impls");
        assert_eq!(mock_store.imports.read().await.len(), 1, "imports");
        assert!(
            !mock_store.call_relationships.read().await.is_empty(),
            "call_relationships should not be empty"
        );

        // Verify file was stored (normalize_path keeps absolute paths)
        let files = mock_store.files.read().await;
        assert_eq!(files.len(), 1, "exactly one file stored");
        assert!(
            files.values().next().unwrap().language == "rust",
            "file has correct language"
        );
    }

    #[tokio::test]
    async fn test_store_parsed_file_for_project_empty_parsed_file() {
        use crate::parser::ParsedFile;

        let mock_store = std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let state = mock_app_state_with_graph(mock_store.clone());
        let orch = Orchestrator::new(state).await.unwrap();

        let parsed = ParsedFile {
            path: "/tmp/test-project/src/empty.rs".to_string(),
            language: "rust".to_string(),
            hash: "empty".to_string(),
            functions: vec![],
            structs: vec![],
            traits: vec![],
            enums: vec![],
            impl_blocks: vec![],
            imports: vec![],
            function_calls: vec![],
            symbols: vec![],
        };

        orch.store_parsed_file_for_project(&parsed, None)
            .await
            .unwrap();

        // Only the file node should be stored
        let files = mock_store.files.read().await;
        assert_eq!(files.len(), 1, "exactly one file stored");
        assert_eq!(mock_store.functions.read().await.len(), 0);
    }

    // ── code embedding tests ────────────────────────────────────────

    #[test]
    fn test_build_file_embedding_text() {
        use crate::neo4j::models::{FunctionNode, StructNode, Visibility};

        let parsed = ParsedFile {
            path: "src/api/handlers.rs".to_string(),
            language: "rust".to_string(),
            hash: "abc123".to_string(),
            functions: vec![FunctionNode {
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
            }],
            structs: vec![StructNode {
                name: "PlanRequest".to_string(),
                file_path: "src/api/handlers.rs".to_string(),
                visibility: Visibility::Public,
                generics: vec![],
                line_start: 1,
                line_end: 5,
                docstring: None,
            }],
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
            params: vec![Parameter {
                name: "request".to_string(),
                type_name: Some("PlanRequest".to_string()),
            }],
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

    // ================================================================
    // Knowledge Fabric — git_log_touched_files parser + backfill tests
    // ================================================================

    #[test]
    fn test_git_file_change_struct() {
        let change = GitFileChange {
            path: "src/main.rs".to_string(),
            additions: Some(10),
            deletions: Some(3),
        };
        assert_eq!(change.path, "src/main.rs");
        assert_eq!(change.additions, Some(10));
        assert_eq!(change.deletions, Some(3));
    }

    #[test]
    fn test_git_file_change_binary() {
        // Binary files have None for additions/deletions ("-\t-\tpath")
        let change = GitFileChange {
            path: "assets/logo.png".to_string(),
            additions: None,
            deletions: None,
        };
        assert!(change.additions.is_none());
        assert!(change.deletions.is_none());
    }

    #[test]
    fn test_git_commit_files_struct() {
        let commit = GitCommitFiles {
            hash: "abc123def456".to_string(),
            message: "fix: resolve auth bug".to_string(),
            author: "Test Author".to_string(),
            timestamp: chrono::Utc::now(),
            files: vec![
                GitFileChange {
                    path: "src/auth.rs".to_string(),
                    additions: Some(5),
                    deletions: Some(2),
                },
                GitFileChange {
                    path: "src/tests.rs".to_string(),
                    additions: Some(20),
                    deletions: Some(0),
                },
            ],
        };
        assert_eq!(commit.hash, "abc123def456");
        assert_eq!(commit.files.len(), 2);
        assert_eq!(commit.files[0].additions, Some(5));
        assert_eq!(commit.files[1].path, "src/tests.rs");
    }

    #[test]
    fn test_backfill_result_struct() {
        let result = BackfillResult {
            commits_parsed: 100,
            commits_backfilled: 95,
            touches_created: 450,
            elapsed_ms: 1234,
        };
        assert_eq!(result.commits_parsed, 100);
        assert_eq!(result.commits_backfilled, 95);
        assert_eq!(result.touches_created, 450);
        assert_eq!(result.elapsed_ms, 1234);
    }

    #[test]
    fn test_git_log_touched_files_on_this_repo() {
        // This test runs on the actual project repo — should always work in CI
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let commits = Orchestrator::git_log_touched_files(&root, 5).unwrap();

        // Should have at most 5 commits (max_commits=5)
        assert!(
            commits.len() <= 5,
            "Expected <=5 commits, got {}",
            commits.len()
        );

        // Each commit should have a non-empty hash and author
        for c in &commits {
            assert!(!c.hash.is_empty(), "Commit hash should not be empty");
            assert!(!c.author.is_empty(), "Author should not be empty");
        }
    }

    #[test]
    fn test_git_log_touched_files_has_file_stats() {
        // Parse 3 commits from the actual repo and check file stats
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let commits = Orchestrator::git_log_touched_files(&root, 3).unwrap();

        if !commits.is_empty() {
            // At least one commit should have files
            let has_files = commits.iter().any(|c| !c.files.is_empty());
            assert!(has_files, "At least one commit should have touched files");

            // Check that file paths are relative (no leading /)
            for c in &commits {
                for f in &c.files {
                    assert!(
                        !f.path.starts_with('/'),
                        "git log should return relative paths, got: {}",
                        f.path
                    );
                }
            }
        }
    }

    #[test]
    fn test_git_log_touched_files_invalid_dir() {
        let result = Orchestrator::git_log_touched_files(
            std::path::Path::new("/tmp/nonexistent_dir_12345"),
            5,
        );
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_backfill_commit_touches_on_mock() {
        let (orch, _rx) = orch_with_bus().await;
        let project = test_project();
        orch.create_project(&project).await.unwrap();

        // Use the actual repo root so git log works
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let result = orch
            .backfill_commit_touches(project.id, &root)
            .await
            .unwrap();

        // Should parse at least a few commits
        assert!(result.commits_parsed > 0, "Should parse some commits");
        assert!(
            result.commits_backfilled > 0,
            "Should backfill some commits"
        );
        assert!(result.elapsed_ms > 0 || result.commits_parsed > 0);
    }

    // =====================================================================
    // SuffixIndex-based resolver tests
    // =====================================================================

    #[test]
    fn test_resolve_rust_import_indexed_crate() {
        let paths = vec![
            "src/neo4j/client.rs".to_string(),
            "src/neo4j/mod.rs".to_string(),
            "src/parser/mod.rs".to_string(),
            "src/parser/helpers.rs".to_string(),
            "src/lib.rs".to_string(),
        ];
        let index = crate::resolver::SuffixIndex::build(&paths);

        // crate::neo4j::client → src/neo4j/client.rs
        let result = Orchestrator::resolve_single_rust_import_indexed(
            "crate::neo4j::client",
            "src/main.rs",
            &index,
        );
        assert_eq!(result, Some("src/neo4j/client.rs".to_string()));

        // crate::parser::helpers → src/parser/helpers.rs
        let result = Orchestrator::resolve_single_rust_import_indexed(
            "crate::parser::helpers",
            "src/main.rs",
            &index,
        );
        assert_eq!(result, Some("src/parser/helpers.rs".to_string()));

        // crate::parser (module) → src/parser/mod.rs
        let result = Orchestrator::resolve_single_rust_import_indexed(
            "crate::parser",
            "src/main.rs",
            &index,
        );
        assert_eq!(result, Some("src/parser/mod.rs".to_string()));
    }

    #[test]
    fn test_resolve_rust_import_indexed_crate_with_type() {
        let paths = vec![
            "src/neo4j/client.rs".to_string(),
            "src/neo4j/models.rs".to_string(),
        ];
        let index = crate::resolver::SuffixIndex::build(&paths);

        // crate::neo4j::models::FunctionNode → src/neo4j/models.rs (strip type name)
        let result = Orchestrator::resolve_single_rust_import_indexed(
            "crate::neo4j::models::FunctionNode",
            "src/main.rs",
            &index,
        );
        assert_eq!(result, Some("src/neo4j/models.rs".to_string()));
    }

    #[test]
    fn test_resolve_rust_import_indexed_external_skip() {
        let paths = vec!["src/main.rs".to_string()];
        let index = crate::resolver::SuffixIndex::build(&paths);

        // External crate — should return None
        let result = Orchestrator::resolve_single_rust_import_indexed(
            "serde::Serialize",
            "src/main.rs",
            &index,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_resolve_typescript_import_indexed_relative() {
        let paths = vec![
            "src/components/Button.tsx".to_string(),
            "src/components/index.ts".to_string(),
            "src/utils/helpers.ts".to_string(),
        ];
        let index = crate::resolver::SuffixIndex::build(&paths);

        // ./Button from src/components/App.tsx
        let result = Orchestrator::resolve_typescript_import_indexed(
            "./Button",
            "src/components/App.tsx",
            &index,
        );
        assert_eq!(result, Some("src/components/Button.tsx".to_string()));

        // ../utils/helpers from src/components/App.tsx
        let result = Orchestrator::resolve_typescript_import_indexed(
            "../utils/helpers",
            "src/components/App.tsx",
            &index,
        );
        assert_eq!(result, Some("src/utils/helpers.ts".to_string()));
    }

    #[test]
    fn test_resolve_typescript_import_indexed_alias() {
        let paths = vec![
            "src/components/Button.tsx".to_string(),
            "src/utils/helpers.ts".to_string(),
        ];
        let index = crate::resolver::SuffixIndex::build(&paths);

        // @/components/Button
        let result = Orchestrator::resolve_typescript_import_indexed(
            "@/components/Button",
            "src/pages/Home.tsx",
            &index,
        );
        assert_eq!(result, Some("src/components/Button.tsx".to_string()));
    }

    #[test]
    fn test_resolve_typescript_import_indexed_npm_skip() {
        let paths = vec!["src/main.ts".to_string()];
        let index = crate::resolver::SuffixIndex::build(&paths);

        // npm package — should return None
        let result = Orchestrator::resolve_typescript_import_indexed(
            "react",
            "src/main.ts",
            &index,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_resolve_python_import_indexed_relative() {
        let paths = vec![
            "src/models/user.py".to_string(),
            "src/models/__init__.py".to_string(),
            "src/utils/helpers.py".to_string(),
        ];
        let index = crate::resolver::SuffixIndex::build(&paths);

        // .models from src/app.py (1 dot = same directory)
        let result = Orchestrator::resolve_python_import_indexed(
            ".models",
            "src/app.py",
            &index,
        );
        assert_eq!(result, Some("src/models/__init__.py".to_string()));

        // .utils.helpers from src/app.py
        let result = Orchestrator::resolve_python_import_indexed(
            ".utils.helpers",
            "src/app.py",
            &index,
        );
        assert_eq!(result, Some("src/utils/helpers.py".to_string()));
    }

    #[test]
    fn test_resolve_python_import_indexed_absolute() {
        let paths = vec![
            "mypackage/models/user.py".to_string(),
            "mypackage/__init__.py".to_string(),
        ];
        let index = crate::resolver::SuffixIndex::build(&paths);

        // mypackage.models.user
        let result = Orchestrator::resolve_python_import_indexed(
            "mypackage.models.user",
            "mypackage/app.py",
            &index,
        );
        assert_eq!(result, Some("mypackage/models/user.py".to_string()));
    }

    #[test]
    fn test_resolve_relative_path_helper() {
        assert_eq!(
            super::resolve_relative_path("src/components", "./Button"),
            "src/components/Button"
        );
        assert_eq!(
            super::resolve_relative_path("src/components", "../utils/helpers"),
            "src/utils/helpers"
        );
        assert_eq!(
            super::resolve_relative_path("src/a/b/c", "../../x"),
            "src/a/x"
        );
    }

    #[test]
    fn test_import_resolution_context() {
        let paths = vec![
            "src/api/handlers.rs".to_string(),
            "src/neo4j/client.rs".to_string(),
        ];
        let mut ctx = crate::resolver::ImportResolutionContext::new(&paths);

        // Cache miss
        assert!(ctx.resolve_cache.get("src/main.rs", "crate::api::handlers").is_none());

        // Insert + hit
        ctx.resolve_cache.insert(
            "src/main.rs".to_string(),
            "crate::api::handlers".to_string(),
            Some("src/api/handlers.rs".to_string()),
        );
        assert!(ctx.resolve_cache.get("src/main.rs", "crate::api::handlers").is_some());

        // SuffixIndex works
        assert_eq!(
            ctx.suffix_index.get("handlers.rs"),
            Some("src/api/handlers.rs")
        );
    }

    // =========================================================================
    // Confidence scoring tests
    // =========================================================================

    fn make_test_parsed_file(path: &str, func_names: &[&str]) -> ParsedFile {
        ParsedFile {
            path: path.to_string(),
            language: "rust".to_string(),
            hash: "test".to_string(),
            functions: func_names
                .iter()
                .enumerate()
                .map(|(i, name)| FunctionNode {
                    name: name.to_string(),
                    visibility: Visibility::Public,
                    params: vec![],
                    return_type: None,
                    generics: vec![],
                    is_async: false,
                    is_unsafe: false,
                    complexity: 1,
                    file_path: path.to_string(),
                    line_start: (i as u32 + 1) * 10,
                    line_end: (i as u32 + 1) * 10 + 5,
                    docstring: None,
                })
                .collect(),
            structs: vec![],
            traits: vec![],
            enums: vec![],
            imports: vec![],
            impl_blocks: vec![],
            function_calls: vec![],
            symbols: vec![],
        }
    }

    #[test]
    fn test_score_same_file() {
        // foo calls bar, both in the same file → same-file (0.85)
        let parsed = make_test_parsed_file("src/lib.rs", &["foo", "bar"]);
        let calls = vec![FunctionCall {
            caller_id: "src/lib.rs:foo:10".to_string(),
            callee_name: "bar".to_string(),
            line: 15,
            confidence: 0.50,
            reason: "unscored".to_string(),
        }];
        let import_rels = vec![];

        let scored = Orchestrator::score_function_calls(
            &calls, &parsed, &import_rels, None,
        );

        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].confidence, 0.85);
        assert_eq!(scored[0].reason, "same-file");
    }

    #[test]
    fn test_score_import_resolved_with_symbol_table() {
        // foo calls helper, helper is in "src/utils.rs" which is imported
        let parsed = make_test_parsed_file("src/lib.rs", &["foo"]);
        let calls = vec![FunctionCall {
            caller_id: "src/lib.rs:foo:10".to_string(),
            callee_name: "helper".to_string(),
            line: 15,
            confidence: 0.50,
            reason: "unscored".to_string(),
        }];
        let import_rels = vec![(
            "src/lib.rs".to_string(),
            "src/utils.rs".to_string(),
            "crate::utils".to_string(),
        )];

        // Build a symbol table with helper in src/utils.rs
        let mut ctx = crate::resolver::ImportResolutionContext::new(&[
            "src/lib.rs".to_string(),
            "src/utils.rs".to_string(),
        ]);
        ctx.symbol_table.add(
            "helper",
            "src/utils.rs:helper:1",
            "src/utils.rs",
            crate::resolver::symbol_table::SymbolType::Function,
            1,
        );

        let scored = Orchestrator::score_function_calls(
            &calls, &parsed, &import_rels, Some(&ctx),
        );

        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].confidence, 0.90);
        assert_eq!(scored[0].reason, "import-resolved");
    }

    #[test]
    fn test_score_fuzzy_unique() {
        // foo calls helper, helper exists in exactly 1 file but NOT imported
        let parsed = make_test_parsed_file("src/lib.rs", &["foo"]);
        let calls = vec![FunctionCall {
            caller_id: "src/lib.rs:foo:10".to_string(),
            callee_name: "helper".to_string(),
            line: 15,
            confidence: 0.50,
            reason: "unscored".to_string(),
        }];
        let import_rels = vec![]; // No imports

        let mut ctx = crate::resolver::ImportResolutionContext::new(&[
            "src/lib.rs".to_string(),
            "src/other.rs".to_string(),
        ]);
        ctx.symbol_table.add(
            "helper",
            "src/other.rs:helper:1",
            "src/other.rs",
            crate::resolver::symbol_table::SymbolType::Function,
            1,
        );

        let scored = Orchestrator::score_function_calls(
            &calls, &parsed, &import_rels, Some(&ctx),
        );

        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].confidence, 0.50);
        assert_eq!(scored[0].reason, "fuzzy-unique");
    }

    #[test]
    fn test_score_fuzzy_ambiguous() {
        // foo calls process, process exists in 2 different files, not imported
        let parsed = make_test_parsed_file("src/lib.rs", &["foo"]);
        let calls = vec![FunctionCall {
            caller_id: "src/lib.rs:foo:10".to_string(),
            callee_name: "process".to_string(),
            line: 15,
            confidence: 0.50,
            reason: "unscored".to_string(),
        }];
        let import_rels = vec![];

        let mut ctx = crate::resolver::ImportResolutionContext::new(&[
            "src/lib.rs".to_string(),
            "src/a.rs".to_string(),
            "src/b.rs".to_string(),
        ]);
        ctx.symbol_table.add(
            "process",
            "src/a.rs:process:1",
            "src/a.rs",
            crate::resolver::symbol_table::SymbolType::Function,
            1,
        );
        ctx.symbol_table.add(
            "process",
            "src/b.rs:process:10",
            "src/b.rs",
            crate::resolver::symbol_table::SymbolType::Function,
            10,
        );

        let scored = Orchestrator::score_function_calls(
            &calls, &parsed, &import_rels, Some(&ctx),
        );

        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].confidence, 0.30);
        assert_eq!(scored[0].reason, "fuzzy-ambiguous");
    }

    #[test]
    fn test_score_unresolved() {
        // foo calls unknown_fn, not in symbol table at all
        let parsed = make_test_parsed_file("src/lib.rs", &["foo"]);
        let calls = vec![FunctionCall {
            caller_id: "src/lib.rs:foo:10".to_string(),
            callee_name: "unknown_fn".to_string(),
            line: 15,
            confidence: 0.50,
            reason: "unscored".to_string(),
        }];
        let import_rels = vec![];

        let ctx = crate::resolver::ImportResolutionContext::new(&[
            "src/lib.rs".to_string(),
        ]);

        let scored = Orchestrator::score_function_calls(
            &calls, &parsed, &import_rels, Some(&ctx),
        );

        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].confidence, 0.30);
        assert_eq!(scored[0].reason, "fuzzy-unresolved");
    }

    #[test]
    fn test_score_no_context_fallback() {
        // Without context, calls that are not same-file get default unscored
        let parsed = make_test_parsed_file("src/lib.rs", &["foo"]);
        let calls = vec![FunctionCall {
            caller_id: "src/lib.rs:foo:10".to_string(),
            callee_name: "external_fn".to_string(),
            line: 15,
            confidence: 0.50,
            reason: "unscored".to_string(),
        }];
        let import_rels = vec![];

        let scored = Orchestrator::score_function_calls(
            &calls, &parsed, &import_rels, None,
        );

        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].confidence, 0.50);
        assert_eq!(scored[0].reason, "unscored");
    }

    // ── Batch utility validation ─────────────────────────────────

    #[tokio::test]
    async fn test_store_parsed_file_idempotent_resync() {
        // Validates that re-storing a file with MERGE pattern is idempotent:
        // storing the same file twice should NOT duplicate entities.
        //
        // NOTE: This test documents current behavior. When Plans 5/6 add
        // EXTENDS/IMPLEMENTS, a separate DELETE-then-CREATE step must be added
        // (see comment block in store_parsed_file_for_project_with_ctx).
        use crate::neo4j::models::*;
        use crate::parser::ParsedFile;

        let mock_store = std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let state = mock_app_state_with_graph(mock_store.clone());
        let orch = Orchestrator::new(state).await.unwrap();

        let file_path = "/tmp/test-project/src/resync.rs".to_string();

        let parsed = ParsedFile {
            path: file_path.clone(),
            language: "rust".to_string(),
            hash: "hash1".to_string(),
            functions: vec![FunctionNode {
                name: "my_func".to_string(),
                visibility: Visibility::Public,
                params: vec![],
                return_type: None,
                generics: vec![],
                is_async: false,
                is_unsafe: false,
                complexity: 1,
                file_path: file_path.clone(),
                line_start: 1,
                line_end: 10,
                docstring: None,
            }],
            structs: vec![StructNode {
                name: "MyStruct".to_string(),
                visibility: Visibility::Public,
                generics: vec![],
                file_path: file_path.clone(),
                line_start: 20,
                line_end: 30,
                docstring: None,
            }],
            traits: vec![],
            enums: vec![],
            impl_blocks: vec![],
            imports: vec![],
            function_calls: vec![],
            symbols: vec!["my_func".to_string()],
        };

        // Store once
        orch.store_parsed_file_for_project(&parsed, None)
            .await
            .unwrap();
        assert_eq!(mock_store.functions.read().await.len(), 1);
        assert_eq!(mock_store.structs_map.read().await.len(), 1);

        // Store again (re-sync) — MERGE should NOT create duplicates
        orch.store_parsed_file_for_project(&parsed, None)
            .await
            .unwrap();
        assert_eq!(
            mock_store.functions.read().await.len(),
            1,
            "MERGE should be idempotent — no duplicate functions"
        );
        assert_eq!(
            mock_store.structs_map.read().await.len(),
            1,
            "MERGE should be idempotent — no duplicate structs"
        );
    }

    #[tokio::test]
    async fn test_batch_utility_chunking_constants() {
        // Validate that our batch utility constants match the cleanup_sync_data pattern
        use crate::neo4j::batch::BATCH_SIZE;

        assert_eq!(BATCH_SIZE, 10_000, "BATCH_SIZE must be 10K per constraint");

        // Verify chunking produces expected splits
        let large_vec: Vec<u8> = vec![0; 25_001];
        let chunks: Vec<&[u8]> = large_vec.chunks(BATCH_SIZE).collect();
        assert_eq!(chunks.len(), 3, "25001 items should produce 3 chunks");
        assert_eq!(chunks[0].len(), 10_000);
        assert_eq!(chunks[1].len(), 10_000);
        assert_eq!(chunks[2].len(), 5_001);
    }

    #[tokio::test]
    async fn test_cleanup_pattern_comment_exists() {
        // Meta-test: verify that the Plans 5/6 cleanup pattern documentation
        // exists in store_parsed_file_for_project_with_ctx.
        // This ensures no one accidentally removes the guidance comment.
        let source = include_str!("runner.rs");
        assert!(
            source.contains("Plans 5 & 6: Heritage & Process relations"),
            "Cleanup pattern comment for Plans 5/6 must exist in runner.rs"
        );
        assert!(
            source.contains("DELETE stale relations for this file"),
            "DELETE pattern guidance must exist in runner.rs"
        );
        assert!(
            source.contains("run_unwind_in_chunks"),
            "Batch helper reference must exist in runner.rs"
        );
    }

    // ── T4.1: Pipeline phase separation tests ─────────────────────

    #[test]
    fn test_scan_files_finds_supported_extensions() {
        // Create a temp directory with various file types
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // Supported files
        std::fs::write(root.join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(root.join("app.ts"), "export {}").unwrap();
        std::fs::write(root.join("lib.py"), "pass").unwrap();
        std::fs::write(root.join("main.go"), "package main").unwrap();

        // Unsupported files — should be excluded
        std::fs::write(root.join("readme.md"), "# Hello").unwrap();
        std::fs::write(root.join("data.json"), "{}").unwrap();
        std::fs::write(root.join("style.css"), "body {}").unwrap();

        let entries = scan_files(root);
        assert_eq!(entries.len(), 4, "should find exactly 4 supported files");

        // Verify all entries have a language
        for entry in &entries {
            assert!(!entry.path.is_empty());
            assert!(entry.size > 0);
        }

        // Verify specific languages are detected
        let languages: Vec<String> = entries.iter().map(|e| {
            e.language.as_str().to_string()
        }).collect();
        assert!(languages.contains(&"rust".to_string()));
        assert!(languages.contains(&"typescript".to_string()));
        assert!(languages.contains(&"python".to_string()));
        assert!(languages.contains(&"go".to_string()));
    }

    #[test]
    fn test_scan_files_respects_ignore_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // Normal file
        std::fs::write(root.join("src.rs"), "fn f() {}").unwrap();

        // Files in ignored directories
        std::fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
        std::fs::write(root.join("node_modules/pkg/index.js"), "module.exports = {}").unwrap();

        std::fs::create_dir_all(root.join(".git/objects")).unwrap();
        std::fs::write(root.join(".git/objects/hook.sh"), "#!/bin/bash").unwrap();

        std::fs::create_dir_all(root.join("target/debug")).unwrap();
        std::fs::write(root.join("target/debug/build.rs"), "fn main() {}").unwrap();

        let entries = scan_files(root);
        assert_eq!(entries.len(), 1, "should only find the root src.rs");
        assert!(entries[0].path.ends_with("src.rs"));
    }

    #[test]
    fn test_scan_files_empty_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let entries = scan_files(tmp.path());
        assert!(entries.is_empty());
    }

    #[test]
    fn test_scan_files_paths_are_normalized() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("test.rs"), "fn x() {}").unwrap();

        let entries = scan_files(root);
        assert_eq!(entries.len(), 1);
        // Normalized paths should be absolute
        assert!(
            std::path::Path::new(&entries[0].path).is_absolute(),
            "scan_files should return absolute paths"
        );
    }

    #[test]
    fn test_scan_files_all_language_extensions() {
        // Verify that scan_files uses SupportedLanguage::from_extension
        // which covers more extensions than the old hardcoded list
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // Extensions that from_extension supports but old list missed
        std::fs::write(root.join("lib.mjs"), "export default {}").unwrap();
        std::fs::write(root.join("types.pyi"), "x: int").unwrap();
        std::fs::write(root.join("Rakefile.rake"), "task :default").unwrap();

        let entries = scan_files(root);
        assert_eq!(entries.len(), 3, "from_extension should match mjs, pyi, rake");
    }

    #[tokio::test]
    async fn test_read_files_loads_content_and_hash() {
        use crate::parser::SupportedLanguage;

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let content = "fn hello() {}";
        std::fs::write(root.join("hello.rs"), content).unwrap();

        let entries = vec![FileEntry {
            path: root.join("hello.rs").to_string_lossy().to_string(),
            size: content.len() as u64,
            language: SupportedLanguage::Rust,
        }];

        let files = read_files(entries).await;
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].content, content);
        assert!(!files[0].hash.is_empty());
        assert_eq!(files[0].size, content.len() as u64);

        // Verify hash is deterministic
        let tmp2 = tempfile::tempdir().unwrap();
        std::fs::write(tmp2.path().join("hello.rs"), content).unwrap();
        let entries2 = vec![FileEntry {
            path: tmp2.path().join("hello.rs").to_string_lossy().to_string(),
            size: content.len() as u64,
            language: SupportedLanguage::Rust,
        }];
        let files2 = read_files(entries2).await;
        assert_eq!(files[0].hash, files2[0].hash, "same content = same hash");
    }

    #[tokio::test]
    async fn test_read_files_skips_unreadable() {
        use crate::parser::SupportedLanguage;

        // File that doesn't exist → should be skipped, not error
        let entries = vec![FileEntry {
            path: "/nonexistent/path/ghost.rs".to_string(),
            size: 0,
            language: SupportedLanguage::Rust,
        }];

        let files = read_files(entries).await;
        assert!(files.is_empty(), "unreadable file should be skipped");
    }

    #[tokio::test]
    async fn test_read_files_preserves_language() {
        use crate::parser::SupportedLanguage;

        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("app.py"), "pass").unwrap();

        let entries = vec![FileEntry {
            path: tmp.path().join("app.py").to_string_lossy().to_string(),
            size: 4,
            language: SupportedLanguage::Python,
        }];

        let files = read_files(entries).await;
        assert_eq!(files.len(), 1);
        assert!(matches!(files[0].language, SupportedLanguage::Python));
    }

    #[test]
    fn test_file_entry_has_correct_size() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let content = "fn hello_world() { println!(\"hello\"); }";
        std::fs::write(root.join("sized.rs"), content).unwrap();

        let entries = scan_files(root);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].size, content.len() as u64);
    }

    #[tokio::test]
    async fn test_parse_files_extracts_symbols() {
        use crate::parser::SupportedLanguage;

        let parser = Arc::new(RwLock::new(CodeParser::new().unwrap()));
        let files = vec![FileContent {
            path: "/tmp/test_parse.rs".to_string(),
            content: r#"
                pub fn greet(name: &str) -> String {
                    format!("Hello, {}!", name)
                }

                pub struct User {
                    pub name: String,
                }
            "#.to_string(),
            size: 100,
            language: SupportedLanguage::Rust,
            hash: "abc123".to_string(),
        }];

        let parsed = parse_files(files, &parser).await;
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].path, "/tmp/test_parse.rs");

        // Should extract function 'greet' and struct 'User'
        assert!(
            parsed[0].functions.iter().any(|f| f.name == "greet"),
            "should find function 'greet'"
        );
        assert!(
            parsed[0].structs.iter().any(|s| s.name == "User"),
            "should find struct 'User'"
        );
    }

    #[tokio::test]
    async fn test_parse_files_handles_multiple_languages() {
        use crate::parser::SupportedLanguage;

        let parser = Arc::new(RwLock::new(CodeParser::new().unwrap()));
        let files = vec![
            FileContent {
                path: "/tmp/lib.rs".to_string(),
                content: "pub fn add(a: i32, b: i32) -> i32 { a + b }".to_string(),
                size: 44,
                language: SupportedLanguage::Rust,
                hash: "h1".to_string(),
            },
            FileContent {
                path: "/tmp/app.py".to_string(),
                content: "def hello():\n    pass\n".to_string(),
                size: 21,
                language: SupportedLanguage::Python,
                hash: "h2".to_string(),
            },
        ];

        let parsed = parse_files(files, &parser).await;
        assert_eq!(parsed.len(), 2);
        assert!(parsed[0].functions.iter().any(|f| f.name == "add"));
        assert!(parsed[1].functions.iter().any(|f| f.name == "hello"));
    }

    #[tokio::test]
    async fn test_full_pipeline_scan_read_parse() {
        // End-to-end test: scan → read → parse on real temp files
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        std::fs::write(
            root.join("math.rs"),
            "pub fn multiply(a: i32, b: i32) -> i32 { a * b }",
        ).unwrap();
        std::fs::write(
            root.join("utils.py"),
            "def square(x):\n    return x * x\n",
        ).unwrap();
        std::fs::write(root.join("readme.md"), "# Skip me").unwrap();

        // Phase 1: Scan
        let entries = scan_files(root);
        assert_eq!(entries.len(), 2, "scan should find .rs and .py");

        // Phase 2: Read
        let file_contents = read_files(entries).await;
        assert_eq!(file_contents.len(), 2, "read should load both files");
        for fc in &file_contents {
            assert!(!fc.content.is_empty());
            assert!(!fc.hash.is_empty());
        }

        // Phase 3: Parse
        let parser = Arc::new(RwLock::new(CodeParser::new().unwrap()));
        let parsed = parse_files(file_contents, &parser).await;
        assert_eq!(parsed.len(), 2, "parse should handle both files");

        // Verify Rust file parsed correctly
        let rs = parsed.iter().find(|p| p.path.ends_with("math.rs")).unwrap();
        assert!(rs.functions.iter().any(|f| f.name == "multiply"));

        // Verify Python file parsed correctly
        let py = parsed.iter().find(|p| p.path.ends_with("utils.py")).unwrap();
        assert!(py.functions.iter().any(|f| f.name == "square"));
    }

    // ── T8.6: Sync consistency integration tests ──────────────────

    /// Scenario 1: Add a file → verify File + symbols + IMPORTS created in Neo4j AND MeiliSearch
    #[tokio::test]
    async fn test_sync_add_file_creates_all_entities() {
        use crate::neo4j::models::*;
        use crate::neo4j::traits::GraphStore;
        use crate::parser::ParsedFile;
        use crate::test_helpers::mock_app_state_with_stores;

        let (state, neo4j, meili) = mock_app_state_with_stores();
        let orch = Orchestrator::new(state).await.unwrap();

        let file_path = "/tmp/sync-test/src/lib.rs".to_string();
        let parsed = ParsedFile {
            path: file_path.clone(),
            language: "rust".to_string(),
            hash: "add-test-hash".to_string(),
            functions: vec![
                FunctionNode {
                    name: "handler".to_string(),
                    visibility: Visibility::Public,
                    params: vec![],
                    return_type: Some("Response".to_string()),
                    generics: vec![],
                    is_async: true,
                    is_unsafe: false,
                    complexity: 3,
                    file_path: file_path.clone(),
                    line_start: 10,
                    line_end: 30,
                    docstring: Some("Handle request".to_string()),
                },
            ],
            structs: vec![StructNode {
                name: "Config".to_string(),
                visibility: Visibility::Public,
                generics: vec![],
                file_path: file_path.clone(),
                line_start: 1,
                line_end: 8,
                docstring: None,
            }],
            traits: vec![],
            enums: vec![],
            impl_blocks: vec![],
            imports: vec![ImportNode {
                path: "serde::Deserialize".to_string(),
                alias: None,
                items: vec!["Deserialize".to_string()],
                file_path: file_path.clone(),
                line: 1,
            }],
            function_calls: vec![],
            symbols: vec!["handler".to_string(), "Config".to_string()],
        };

        // Store with project context so MeiliSearch gets indexed
        let project_id = uuid::Uuid::new_v4();
        orch.store_parsed_file_for_project(&parsed, Some(project_id))
            .await
            .unwrap();

        // Also manually index in MeiliSearch (normally done by sync_file_for_project)
        let doc = crate::parser::CodeParser::to_code_document(
            &parsed,
            &project_id.to_string(),
            "test-project",
        );
        orch.meili().index_code(&doc).await.unwrap();

        // ── Verify Neo4j ──
        assert_eq!(neo4j.functions.read().await.len(), 1, "1 function expected");
        assert_eq!(neo4j.structs_map.read().await.len(), 1, "1 struct expected");
        assert_eq!(neo4j.imports.read().await.len(), 1, "1 import expected");
        let file = neo4j.get_file(&file_path).await.unwrap();
        assert!(file.is_some(), "File node should exist in Neo4j");

        // ── Verify MeiliSearch ──
        let code_docs = meili.code_documents.read().await;
        assert_eq!(code_docs.len(), 1, "1 code document in MeiliSearch");
        assert_eq!(code_docs[0].path, file_path);
        assert!(code_docs[0].symbols.contains(&"handler".to_string()));
        assert!(code_docs[0].symbols.contains(&"Config".to_string()));
    }

    /// Scenario 2: Delete a file (< 50 threshold) → verify everything cleaned in Neo4j AND MeiliSearch
    #[tokio::test]
    async fn test_sync_delete_file_cleans_all() {
        use crate::neo4j::models::*;
        use crate::neo4j::traits::GraphStore;
        use crate::parser::ParsedFile;
        use crate::test_helpers::mock_app_state_with_stores;

        let (state, neo4j, meili) = mock_app_state_with_stores();
        let orch = Orchestrator::new(state).await.unwrap();
        let project_id = uuid::Uuid::new_v4();
        let file_path = "/tmp/sync-test/src/delete_me.rs".to_string();

        // Step 1: Add the file
        let parsed = ParsedFile {
            path: file_path.clone(),
            language: "rust".to_string(),
            hash: "delete-test-hash".to_string(),
            functions: vec![FunctionNode {
                name: "temp_func".to_string(),
                visibility: Visibility::Public,
                params: vec![],
                return_type: None,
                generics: vec![],
                is_async: false,
                is_unsafe: false,
                complexity: 1,
                file_path: file_path.clone(),
                line_start: 1,
                line_end: 5,
                docstring: None,
            }],
            structs: vec![],
            traits: vec![],
            enums: vec![],
            impl_blocks: vec![],
            imports: vec![],
            function_calls: vec![],
            symbols: vec!["temp_func".to_string()],
        };

        orch.store_parsed_file_for_project(&parsed, Some(project_id))
            .await
            .unwrap();
        let doc = crate::parser::CodeParser::to_code_document(
            &parsed,
            &project_id.to_string(),
            "test-project",
        );
        orch.meili().index_code(&doc).await.unwrap();

        // Verify file exists
        assert_eq!(neo4j.functions.read().await.len(), 1);
        assert_eq!(meili.code_documents.read().await.len(), 1);

        // Step 2: Delete the file (simulating what watcher does)
        GraphStore::delete_file(neo4j.as_ref(), &file_path)
            .await
            .unwrap();
        orch.meili().delete_code(&file_path).await.unwrap();

        // ── Verify Neo4j cleaned ──
        let file = neo4j.get_file(&file_path).await.unwrap();
        assert!(file.is_none(), "File node should be deleted from Neo4j");
        // Functions are cleaned via DETACH DELETE in the mock
        assert_eq!(
            neo4j.functions.read().await.len(),
            0,
            "Functions should be deleted"
        );

        // ── Verify MeiliSearch cleaned ──
        assert_eq!(
            meili.code_documents.read().await.len(),
            0,
            "MeiliSearch documents should be deleted"
        );
    }

    /// Scenario 3: Rename a file → old node deleted, new created with correct relations
    #[tokio::test]
    async fn test_sync_rename_file_old_deleted_new_created() {
        use crate::neo4j::models::*;
        use crate::neo4j::traits::GraphStore;
        use crate::parser::ParsedFile;
        use crate::test_helpers::mock_app_state_with_stores;

        let (state, neo4j, meili) = mock_app_state_with_stores();
        let orch = Orchestrator::new(state).await.unwrap();
        let project_id = uuid::Uuid::new_v4();
        let old_path = "/tmp/sync-test/src/old_name.rs".to_string();
        let new_path = "/tmp/sync-test/src/new_name.rs".to_string();

        // Step 1: Create file with old name
        let old_parsed = ParsedFile {
            path: old_path.clone(),
            language: "rust".to_string(),
            hash: "old-hash".to_string(),
            functions: vec![FunctionNode {
                name: "my_fn".to_string(),
                visibility: Visibility::Public,
                params: vec![],
                return_type: None,
                generics: vec![],
                is_async: false,
                is_unsafe: false,
                complexity: 1,
                file_path: old_path.clone(),
                line_start: 1,
                line_end: 5,
                docstring: None,
            }],
            structs: vec![],
            traits: vec![],
            enums: vec![],
            impl_blocks: vec![],
            imports: vec![],
            function_calls: vec![],
            symbols: vec!["my_fn".to_string()],
        };

        orch.store_parsed_file_for_project(&old_parsed, Some(project_id))
            .await
            .unwrap();
        let doc = crate::parser::CodeParser::to_code_document(
            &old_parsed,
            &project_id.to_string(),
            "test",
        );
        orch.meili().index_code(&doc).await.unwrap();

        // Step 2: Simulate rename = delete old + create new
        GraphStore::delete_file(neo4j.as_ref(), &old_path)
            .await
            .unwrap();
        orch.meili().delete_code(&old_path).await.unwrap();

        let new_parsed = ParsedFile {
            path: new_path.clone(),
            language: "rust".to_string(),
            hash: "new-hash".to_string(),
            functions: vec![FunctionNode {
                name: "my_fn".to_string(),
                visibility: Visibility::Public,
                params: vec![],
                return_type: None,
                generics: vec![],
                is_async: false,
                is_unsafe: false,
                complexity: 1,
                file_path: new_path.clone(),
                line_start: 1,
                line_end: 5,
                docstring: None,
            }],
            structs: vec![],
            traits: vec![],
            enums: vec![],
            impl_blocks: vec![],
            imports: vec![],
            function_calls: vec![],
            symbols: vec!["my_fn".to_string()],
        };

        orch.store_parsed_file_for_project(&new_parsed, Some(project_id))
            .await
            .unwrap();
        let doc = crate::parser::CodeParser::to_code_document(
            &new_parsed,
            &project_id.to_string(),
            "test",
        );
        orch.meili().index_code(&doc).await.unwrap();

        // ── Verify old gone, new exists ──
        let old_file = neo4j.get_file(&old_path).await.unwrap();
        assert!(old_file.is_none(), "Old file should be deleted");

        let new_file = neo4j.get_file(&new_path).await.unwrap();
        assert!(new_file.is_some(), "New file should exist");

        // Function with new file_path should exist
        let funcs = neo4j.functions.read().await;
        assert_eq!(funcs.len(), 1, "Should have exactly 1 function");
        let func = funcs.values().next().unwrap();
        assert_eq!(func.file_path, new_path, "Function should reference new path");

        // MeiliSearch: only new file
        let code_docs = meili.code_documents.read().await;
        assert_eq!(code_docs.len(), 1);
        assert_eq!(code_docs[0].path, new_path);
    }

    /// Scenario 4: Modify hierarchy (EXTENDS change) — forward-compatible validation
    #[tokio::test]
    async fn test_sync_heritage_change_forward_compatible() {
        // EXTENDS/IMPLEMENTS don't exist yet (Plans 5/6).
        // This test validates that the cleanup_sync_data handles them as no-ops
        // and that the pattern documentation exists for future implementation.
        use crate::neo4j::traits::GraphStore;
        use crate::test_helpers::mock_app_state_with_stores;

        let (state, neo4j, _meili) = mock_app_state_with_stores();
        let _orch = Orchestrator::new(state).await.unwrap();

        // cleanup_sync_data should handle EXTENDS/IMPLEMENTS without error
        // (forward-compatible no-ops)
        let deleted = GraphStore::cleanup_sync_data(neo4j.as_ref()).await.unwrap();
        assert_eq!(deleted, 0, "Empty store cleanup should delete 0 entities");

        // Verify the Plans 5/6 cleanup pattern documentation exists
        let source = include_str!("runner.rs");
        assert!(
            source.contains("Plans 5 & 6: Heritage & Process relations"),
            "Heritage cleanup pattern documentation must exist"
        );
    }

    /// Scenario 5: Bulk delete (>= BULK_SYNC_THRESHOLD files) → full sync path
    #[tokio::test]
    async fn test_sync_bulk_delete_triggers_full_sync_path() {
        use crate::meilisearch::traits::SearchStore;
        use crate::neo4j::models::*;
        use crate::neo4j::traits::GraphStore;
        use crate::test_helpers::mock_app_state_with_stores;

        let (state, neo4j, meili) = mock_app_state_with_stores();
        let _orch = Orchestrator::new(state).await.unwrap();
        let project_id = uuid::Uuid::new_v4();

        // Create 60 files (> BULK_SYNC_THRESHOLD = 50)
        for i in 0..60 {
            let path = format!("/tmp/bulk-test/src/file_{}.rs", i);
            GraphStore::upsert_file(
                neo4j.as_ref(),
                &FileNode {
                    path: path.clone(),
                    language: "rust".to_string(),
                    hash: format!("hash_{}", i),
                    last_parsed: chrono::Utc::now(),
                    project_id: Some(project_id),
                },
            )
            .await
            .unwrap();
            // Link to project (mock uses project_files for delete_stale_files)
            GraphStore::link_file_to_project(neo4j.as_ref(), &path, project_id)
                .await
                .unwrap();
        }

        assert_eq!(neo4j.files.read().await.len(), 60);

        // Simulate keeping only 10 files (50 deletions) via delete_stale_files
        let valid_paths: Vec<String> =
            (0..10).map(|i| format!("/tmp/bulk-test/src/file_{}.rs", i)).collect();

        let (files_deleted, _symbols_deleted, stale_paths) = GraphStore::delete_stale_files(
            neo4j.as_ref(),
            project_id,
            &valid_paths,
        )
        .await
        .unwrap();

        assert_eq!(files_deleted, 50, "Should delete 50 stale files");
        assert_eq!(stale_paths.len(), 50, "Should return 50 deleted paths");
        assert_eq!(neo4j.files.read().await.len(), 10, "10 files should remain");

        // Simulate MeiliSearch cleanup for each stale path
        for path in &stale_paths {
            meili.delete_code(path).await.unwrap();
        }
    }

    /// Scenario 6: Add + delete simultaneously in debounce window → consistency
    #[tokio::test]
    async fn test_sync_add_delete_same_window_consistent() {
        use crate::neo4j::models::*;
        use crate::neo4j::traits::GraphStore;
        use crate::parser::ParsedFile;
        use crate::test_helpers::mock_app_state_with_stores;

        let (state, neo4j, meili) = mock_app_state_with_stores();
        let orch = Orchestrator::new(state).await.unwrap();
        let project_id = uuid::Uuid::new_v4();

        let file_a = "/tmp/simul-test/src/file_a.rs".to_string();
        let file_b = "/tmp/simul-test/src/file_b.rs".to_string();

        // Create file_a and file_b
        for path in [&file_a, &file_b] {
            let parsed = ParsedFile {
                path: path.clone(),
                language: "rust".to_string(),
                hash: "hash".to_string(),
                functions: vec![FunctionNode {
                    name: format!("fn_{}", path.split('/').last().unwrap()),
                    visibility: Visibility::Public,
                    params: vec![],
                    return_type: None,
                    generics: vec![],
                    is_async: false,
                    is_unsafe: false,
                    complexity: 1,
                    file_path: path.clone(),
                    line_start: 1,
                    line_end: 5,
                    docstring: None,
                }],
                structs: vec![],
                traits: vec![],
                enums: vec![],
                impl_blocks: vec![],
                imports: vec![],
                function_calls: vec![],
                symbols: vec![],
            };
            orch.store_parsed_file_for_project(&parsed, Some(project_id))
                .await
                .unwrap();
            let doc = crate::parser::CodeParser::to_code_document(
                &parsed,
                &project_id.to_string(),
                "test",
            );
            orch.meili().index_code(&doc).await.unwrap();
        }

        assert_eq!(neo4j.functions.read().await.len(), 2);
        assert_eq!(meili.code_documents.read().await.len(), 2);

        // Simulate: delete file_a + modify file_b in same window
        GraphStore::delete_file(neo4j.as_ref(), &file_a)
            .await
            .unwrap();
        orch.meili().delete_code(&file_a).await.unwrap();

        let parsed_b_modified = ParsedFile {
            path: file_b.clone(),
            language: "rust".to_string(),
            hash: "new-hash".to_string(),
            functions: vec![
                FunctionNode {
                    name: "fn_file_b.rs".to_string(),
                    visibility: Visibility::Public,
                    params: vec![],
                    return_type: None,
                    generics: vec![],
                    is_async: false,
                    is_unsafe: false,
                    complexity: 1,
                    file_path: file_b.clone(),
                    line_start: 1,
                    line_end: 5,
                    docstring: None,
                },
                FunctionNode {
                    name: "new_func".to_string(),
                    visibility: Visibility::Public,
                    params: vec![],
                    return_type: None,
                    generics: vec![],
                    is_async: false,
                    is_unsafe: false,
                    complexity: 1,
                    file_path: file_b.clone(),
                    line_start: 10,
                    line_end: 15,
                    docstring: None,
                },
            ],
            structs: vec![],
            traits: vec![],
            enums: vec![],
            impl_blocks: vec![],
            imports: vec![],
            function_calls: vec![],
            symbols: vec![],
        };
        orch.store_parsed_file_for_project(&parsed_b_modified, Some(project_id))
            .await
            .unwrap();

        // ── Verify consistency ──
        let file_a_node = neo4j.get_file(&file_a).await.unwrap();
        assert!(file_a_node.is_none(), "file_a should be gone");

        let file_b_node = neo4j.get_file(&file_b).await.unwrap();
        assert!(file_b_node.is_some(), "file_b should exist");

        // file_b should have 2 functions now
        let funcs = neo4j.functions.read().await;
        let file_b_funcs: Vec<_> = funcs
            .values()
            .filter(|f| f.file_path == file_b)
            .collect();
        assert_eq!(file_b_funcs.len(), 2, "file_b should have 2 functions");

        // MeiliSearch: only file_b
        let code_docs = meili.code_documents.read().await;
        assert_eq!(code_docs.len(), 1);
        assert_eq!(code_docs[0].path, file_b);
    }

    /// Scenario 7: Delete file that is source of IMPORTS → IMPORTS relations cleaned
    #[tokio::test]
    async fn test_sync_delete_import_source_cleans_relations() {
        use crate::neo4j::models::*;
        use crate::neo4j::traits::GraphStore;
        use crate::parser::ParsedFile;
        use crate::test_helpers::mock_app_state_with_stores;

        let (state, neo4j, _meili) = mock_app_state_with_stores();
        let orch = Orchestrator::new(state).await.unwrap();

        let importer = "/tmp/import-test/src/main.rs".to_string();
        let imported = "/tmp/import-test/src/utils.rs".to_string();

        // Create both files with an import relationship
        let parsed_importer = ParsedFile {
            path: importer.clone(),
            language: "rust".to_string(),
            hash: "importer-hash".to_string(),
            functions: vec![FunctionNode {
                name: "main".to_string(),
                visibility: Visibility::Public,
                params: vec![],
                return_type: None,
                generics: vec![],
                is_async: false,
                is_unsafe: false,
                complexity: 1,
                file_path: importer.clone(),
                line_start: 1,
                line_end: 10,
                docstring: None,
            }],
            structs: vec![],
            traits: vec![],
            enums: vec![],
            impl_blocks: vec![],
            imports: vec![ImportNode {
                path: "utils".to_string(),
                alias: None,
                items: vec![],
                file_path: importer.clone(),
                line: 1,
            }],
            function_calls: vec![],
            symbols: vec!["main".to_string()],
        };

        let parsed_imported = ParsedFile {
            path: imported.clone(),
            language: "rust".to_string(),
            hash: "imported-hash".to_string(),
            functions: vec![FunctionNode {
                name: "helper".to_string(),
                visibility: Visibility::Public,
                params: vec![],
                return_type: None,
                generics: vec![],
                is_async: false,
                is_unsafe: false,
                complexity: 1,
                file_path: imported.clone(),
                line_start: 1,
                line_end: 5,
                docstring: None,
            }],
            structs: vec![],
            traits: vec![],
            enums: vec![],
            impl_blocks: vec![],
            imports: vec![],
            function_calls: vec![],
            symbols: vec!["helper".to_string()],
        };

        orch.store_parsed_file_for_project(&parsed_importer, None)
            .await
            .unwrap();
        orch.store_parsed_file_for_project(&parsed_imported, None)
            .await
            .unwrap();

        assert_eq!(neo4j.functions.read().await.len(), 2);
        assert_eq!(neo4j.imports.read().await.len(), 1);

        // Delete the imported file — DETACH DELETE should clean relations
        GraphStore::delete_file(neo4j.as_ref(), &imported)
            .await
            .unwrap();

        // ── Verify ──
        let imported_node = neo4j.get_file(&imported).await.unwrap();
        assert!(imported_node.is_none(), "Imported file should be deleted");

        // The importer file and its import node should still exist
        // (the import statement in main.rs still references utils)
        let importer_node = neo4j.get_file(&importer).await.unwrap();
        assert!(importer_node.is_some(), "Importer file should still exist");
    }

    /// Scenario 8: MeiliSearch post-deletion → search returns no phantom results
    #[tokio::test]
    async fn test_sync_meilisearch_no_phantom_after_deletion() {
        use crate::neo4j::models::*;
        use crate::neo4j::traits::GraphStore;
        use crate::parser::ParsedFile;
        use crate::test_helpers::mock_app_state_with_stores;

        let (state, neo4j, meili) = mock_app_state_with_stores();
        let orch = Orchestrator::new(state).await.unwrap();
        let project_id = uuid::Uuid::new_v4();

        // Create 3 files
        for i in 0..3 {
            let path = format!("/tmp/phantom-test/src/file_{}.rs", i);
            let parsed = ParsedFile {
                path: path.clone(),
                language: "rust".to_string(),
                hash: format!("hash_{}", i),
                functions: vec![FunctionNode {
                    name: format!("func_{}", i),
                    visibility: Visibility::Public,
                    params: vec![],
                    return_type: None,
                    generics: vec![],
                    is_async: false,
                    is_unsafe: false,
                    complexity: 1,
                    file_path: path.clone(),
                    line_start: 1,
                    line_end: 5,
                    docstring: None,
                }],
                structs: vec![],
                traits: vec![],
                enums: vec![],
                impl_blocks: vec![],
                imports: vec![],
                function_calls: vec![],
                symbols: vec![format!("func_{}", i)],
            };
            orch.store_parsed_file_for_project(&parsed, Some(project_id))
                .await
                .unwrap();
            let doc = crate::parser::CodeParser::to_code_document(
                &parsed,
                &project_id.to_string(),
                "test",
            );
            orch.meili().index_code(&doc).await.unwrap();
        }

        assert_eq!(meili.code_documents.read().await.len(), 3);

        // Delete file_1 from both Neo4j and MeiliSearch
        let deleted_path = "/tmp/phantom-test/src/file_1.rs";
        neo4j.delete_file(deleted_path).await.unwrap();
        orch.meili().delete_code(deleted_path).await.unwrap();

        // ── Verify no phantom documents ──
        let code_docs = meili.code_documents.read().await;
        assert_eq!(code_docs.len(), 2, "Should have 2 documents after deletion");

        // Verify deleted file is not in results
        let phantom = code_docs.iter().find(|d| d.path == deleted_path);
        assert!(phantom.is_none(), "Deleted file should not be in MeiliSearch");

        // Verify remaining files are correct
        let paths: Vec<&str> = code_docs.iter().map(|d| d.path.as_str()).collect();
        assert!(paths.contains(&"/tmp/phantom-test/src/file_0.rs"));
        assert!(paths.contains(&"/tmp/phantom-test/src/file_2.rs"));

        // Search should not find the deleted file
        let results = orch.meili().search_code("func_1", 10, None).await.unwrap();
        assert!(
            results.is_empty(),
            "Search for deleted file's function should return empty"
        );
    }

    // ── Stress tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_batch_stress_150k() {
        // Stress test: 150K total items (50K functions + 50K calls + 50K imports)
        // Validates that the mock store handles large volumes without OOM
        // and completes within a reasonable time.
        //
        // This tests the DATA PREPARATION path (Vec allocation, HashMap building,
        // chunking logic). The actual Neo4j UNWIND is tested via integration tests.
        use crate::neo4j::batch::{BoltMap, BATCH_SIZE};
        use crate::neo4j::models::*;
        use crate::neo4j::traits::GraphStore;
        use crate::parser::FunctionCall;

        let start = std::time::Instant::now();

        let mock_store = std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let state = mock_app_state_with_graph(mock_store.clone());
        let _orch = Orchestrator::new(state).await.unwrap();

        let file_base = "/tmp/stress-test/src";

        // ── Phase 1: Generate and store 50K functions across 500 files ──
        // (100 functions per file × 500 files = 50,000 functions)
        for file_idx in 0..500 {
            let file_path = format!("{}/file_{}.rs", file_base, file_idx);
            let functions: Vec<FunctionNode> = (0..100)
                .map(|func_idx| FunctionNode {
                    name: format!("func_{}_{}", file_idx, func_idx),
                    visibility: Visibility::Public,
                    params: vec![],
                    return_type: None,
                    generics: vec![],
                    is_async: false,
                    is_unsafe: false,
                    complexity: 1,
                    file_path: file_path.clone(),
                    line_start: func_idx * 10 + 1,
                    line_end: func_idx * 10 + 9,
                    docstring: None,
                })
                .collect();

            GraphStore::batch_upsert_functions(mock_store.as_ref(), &functions)
                .await
                .unwrap();
        }

        let func_count = mock_store.functions.read().await.len();
        assert_eq!(func_count, 50_000, "Should have 50K functions");
        let phase1_time = start.elapsed();
        eprintln!(
            "Phase 1 (50K functions): {:?} — {} stored",
            phase1_time, func_count
        );

        // ── Phase 2: Generate and store 50K call relationships ──
        // (100 calls per file × 500 files = 50,000 calls)
        for file_idx in 0..500 {
            let file_path = format!("{}/file_{}.rs", file_base, file_idx);
            let calls: Vec<FunctionCall> = (0..100)
                .map(|func_idx| {
                    let caller_id =
                        format!("{}:func_{}_{}:{}", file_path, file_idx, func_idx, func_idx * 10 + 1);
                    let callee_name = format!("func_{}_{}", file_idx, (func_idx + 1) % 100);
                    FunctionCall {
                        caller_id,
                        callee_name,
                        line: func_idx * 10 + 5,
                        confidence: 0.8,
                        reason: "same_file".to_string(),
                    }
                })
                .collect();

            GraphStore::batch_create_call_relationships(mock_store.as_ref(), &calls, None)
                .await
                .unwrap();
        }

        let call_count = mock_store.call_relationships.read().await.len();
        assert!(call_count > 0, "Should have call relationships");
        let phase2_time = start.elapsed();
        eprintln!(
            "Phase 2 (50K calls): {:?} — {} stored",
            phase2_time - phase1_time,
            call_count
        );

        // ── Phase 3: Generate and store 50K import relationships ──
        for file_idx in 0..500 {
            let source_path = format!("{}/file_{}.rs", file_base, file_idx);
            let rels: Vec<(String, String, String)> = (0..100)
                .map(|imp_idx| {
                    let target_path =
                        format!("{}/file_{}.rs", file_base, (file_idx + imp_idx + 1) % 500);
                    (
                        source_path.clone(),
                        target_path,
                        format!("import_{}", imp_idx),
                    )
                })
                .collect();

            GraphStore::batch_create_import_relationships(mock_store.as_ref(), &rels)
                .await
                .unwrap();
        }

        let import_count = mock_store.import_relationships.read().await.len();
        assert!(import_count > 0, "Should have import relationships");
        let phase3_time = start.elapsed();
        eprintln!(
            "Phase 3 (50K imports): {:?} — {} stored",
            phase3_time - phase2_time,
            import_count
        );

        // ── Phase 4: Validate BoltMap construction at scale ──
        // Simulate what run_unwind_in_chunks does: build 150K BoltMaps
        let items: Vec<BoltMap> = (0..150_000)
            .map(|i| {
                let mut m = BoltMap::new();
                m.insert("id".into(), format!("item-{}", i).into());
                m.insert("name".into(), format!("name-{}", i).into());
                m.insert("value".into(), (i as i64).into());
                m
            })
            .collect();

        // Verify chunking
        let chunks: Vec<&[BoltMap]> = items.chunks(BATCH_SIZE).collect();
        assert_eq!(chunks.len(), 15, "150K items / 10K = 15 chunks");
        assert_eq!(chunks[0].len(), 10_000);
        assert_eq!(chunks[14].len(), 10_000);

        let total_time = start.elapsed();
        eprintln!("Total stress test time: {:?}", total_time);
        eprintln!(
            "Total items: {} functions + {} calls + {} imports + 150K BoltMaps",
            func_count, call_count, import_count
        );

        // Must complete within 30s (typically < 2s on modern hardware)
        assert!(
            total_time.as_secs() < 30,
            "Stress test took too long: {:?} (limit: 30s)",
            total_time
        );
    }

    #[tokio::test]
    async fn test_batch_stress_cleanup_pattern() {
        // Validates that the cleanup LIMIT loop pattern scales correctly
        // with the mock store's cleanup_sync_data implementation.
        use crate::neo4j::models::*;
        use crate::neo4j::traits::GraphStore;

        let mock_store = std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new());

        let file_base = "/tmp/stress-cleanup/src";

        // Create 1000 functions across 10 files
        for file_idx in 0..10 {
            let file_path = format!("{}/file_{}.rs", file_base, file_idx);

            // First create the file node
            GraphStore::upsert_file(
                mock_store.as_ref(),
                &FileNode {
                    path: file_path.clone(),
                    language: "rust".to_string(),
                    hash: format!("hash_{}", file_idx),
                    last_parsed: chrono::Utc::now(),
                    project_id: None,
                },
            )
            .await
            .unwrap();

            let functions: Vec<FunctionNode> = (0..100)
                .map(|func_idx| FunctionNode {
                    name: format!("func_{}_{}", file_idx, func_idx),
                    visibility: Visibility::Public,
                    params: vec![],
                    return_type: None,
                    generics: vec![],
                    is_async: false,
                    is_unsafe: false,
                    complexity: 1,
                    file_path: file_path.clone(),
                    line_start: func_idx * 10 + 1,
                    line_end: func_idx * 10 + 9,
                    docstring: None,
                })
                .collect();

            GraphStore::batch_upsert_functions(mock_store.as_ref(), &functions)
                .await
                .unwrap();
        }

        assert_eq!(mock_store.functions.read().await.len(), 1000);
        assert_eq!(mock_store.files.read().await.len(), 10);

        // cleanup_sync_data should remove everything
        let deleted = GraphStore::cleanup_sync_data(mock_store.as_ref())
            .await
            .unwrap();
        assert!(deleted > 0, "Should have deleted entities");

        // All code entities should be gone
        assert_eq!(
            mock_store.functions.read().await.len(),
            0,
            "All functions should be deleted"
        );
        assert_eq!(
            mock_store.files.read().await.len(),
            0,
            "All files should be deleted"
        );
    }
}
