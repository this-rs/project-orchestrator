//! Status & ReasoningTree Injection Stage for the Chat Enrichment Pipeline.
//!
//! Enriches the user message with:
//!
//! 1. **Work status**: In-progress plans and tasks for the current project,
//!    formatted as a concise summary so the LLM is aware of ongoing work.
//! 2. **Protocol status** (stub): Placeholder for Pattern Federation protocol
//!    runs — currently returns empty, ready for T3 integration.
//! 3. **ReasoningTree**: If a `ReasoningTreeEngine` is available, generates
//!    a decision tree from the user message and injects suggested actions
//!    when confidence > threshold.
//!
//! # Design choices
//!
//! - Status queries use `list_plans_for_project` + `get_plan_tasks` from GraphStore
//! - ReasoningTree is optional (`Option<Arc<ReasoningTreeEngine>>`) since it
//!   requires an `EmbeddingProvider` that ChatManager may not have
//! - Protocol status uses a trait stub (`ProtocolStatusProvider`) for future
//!   Pattern Federation integration

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::time::timeout;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::chat::enrichment::{
    EnrichmentConfig, EnrichmentInput, ParallelEnrichmentStage, StageOutput,
};
use crate::neo4j::models::TaskStatus;
use crate::neo4j::traits::GraphStore;
use crate::reasoning::{ReasoningTree, ReasoningTreeConfig, ReasoningTreeEngine};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the status injection stage.
#[derive(Debug, Clone)]
pub struct StatusInjectionConfig {
    /// Maximum number of in-progress plans to show (default: 3).
    pub max_plans: usize,
    /// Maximum number of in-progress tasks per plan to show (default: 5).
    pub max_tasks_per_plan: usize,
    /// Minimum confidence for ReasoningTree injection (default: 0.5).
    pub reasoning_confidence_threshold: f64,
    /// Timeout for status queries in milliseconds (default: 200).
    pub query_timeout_ms: u64,
    /// Timeout for ReasoningTree generation in milliseconds (default: 300).
    pub reasoning_timeout_ms: u64,
}

impl Default for StatusInjectionConfig {
    fn default() -> Self {
        Self {
            max_plans: 3,
            max_tasks_per_plan: 5,
            reasoning_confidence_threshold: 0.5,
            query_timeout_ms: 200,
            reasoning_timeout_ms: 300,
        }
    }
}

// ============================================================================
// Protocol Status Provider (stub for Pattern Federation)
// ============================================================================

/// Status of an active protocol run (Pattern Federation).
///
/// This is a stub prepared for T3 — Pattern Federation integration.
/// When T3 is implemented, the stub provider will be replaced with a real one
/// that queries active FSM runs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProtocolRunStatus {
    /// Protocol/pattern name (e.g., "code_review", "deploy_pipeline")
    pub protocol_name: String,
    /// Current FSM state (e.g., "awaiting_approval", "running_tests")
    pub current_state: String,
    /// Progress percentage (0-100)
    pub progress: u8,
    /// Human-readable status message
    pub status_message: String,
    /// Contextual prompt fragment from the current state (injected into prompt)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_fragment: Option<String>,
    /// Tools allowed in the current state (empty = all)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available_tools: Option<Vec<String>>,
    /// Actions forbidden in the current state
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forbidden_actions: Option<Vec<String>>,
}

/// Trait for providing active protocol run status.
///
/// Stub for Pattern Federation (T3). Currently implemented by `NoOpProtocolProvider`
/// which always returns an empty list. When Pattern Federation is implemented,
/// a real provider will query active FSM runs from the graph.
#[async_trait::async_trait]
pub trait ProtocolStatusProvider: Send + Sync {
    /// Get all active protocol runs for a project.
    async fn get_active_runs(&self, project_id: Uuid) -> Result<Vec<ProtocolRunStatus>>;
}

/// No-op implementation that always returns empty.
#[deprecated(note = "Use GraphProtocolProvider instead")]
pub struct NoOpProtocolProvider;

#[async_trait::async_trait]
#[allow(deprecated)]
impl ProtocolStatusProvider for NoOpProtocolProvider {
    async fn get_active_runs(&self, _project_id: Uuid) -> Result<Vec<ProtocolRunStatus>> {
        Ok(Vec::new())
    }
}

/// Real implementation backed by Neo4j graph.
///
/// Queries active (running) protocol runs for a project and enriches each
/// with the current state's `prompt_fragment`, `available_tools`, and
/// `forbidden_actions`.
pub struct GraphProtocolProvider {
    graph: Arc<dyn GraphStore>,
}

impl GraphProtocolProvider {
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }
}

#[async_trait::async_trait]
impl ProtocolStatusProvider for GraphProtocolProvider {
    async fn get_active_runs(&self, project_id: Uuid) -> Result<Vec<ProtocolRunStatus>> {
        // 1. Get all protocols for this project
        let (protocols, _) = self.graph.list_protocols(project_id, None, 100, 0).await?;

        if protocols.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();

        // 2. For each protocol, check for running runs
        for proto in &protocols {
            let (runs, _) = self
                .graph
                .list_protocol_runs(proto.id, Some(crate::protocol::RunStatus::Running), 10, 0)
                .await?;

            for run in &runs {
                // 3. Get the current state to extract prompt fields
                let states = self.graph.get_protocol_states(proto.id).await?;
                let current = states.iter().find(|s| s.id == run.current_state);

                let (state_name, prompt_fragment, available_tools, forbidden_actions) =
                    match current {
                        Some(s) => (
                            s.name.clone(),
                            s.prompt_fragment.clone(),
                            s.available_tools.clone(),
                            s.forbidden_actions.clone(),
                        ),
                        None => (format!("unknown({})", run.current_state), None, None, None),
                    };

                // Calculate progress from states_visited
                let total_states = states.len().max(1);
                let visited = run.states_visited.len();
                let progress = ((visited as f64 / total_states as f64) * 100.0).min(99.0) as u8;

                results.push(ProtocolRunStatus {
                    protocol_name: proto.name.clone(),
                    current_state: state_name,
                    progress,
                    status_message: format!(
                        "Running ({}/{} states visited)",
                        visited, total_states
                    ),
                    prompt_fragment,
                    available_tools,
                    forbidden_actions,
                });
            }
        }

        Ok(results)
    }
}

// ============================================================================
// Stage implementation
// ============================================================================

/// Enrichment stage that injects work status and reasoning tree into context.
pub struct StatusInjectionStage {
    graph: Arc<dyn GraphStore>,
    reasoning_engine: Option<Arc<ReasoningTreeEngine>>,
    protocol_provider: Arc<dyn ProtocolStatusProvider>,
    config: StatusInjectionConfig,
}

impl StatusInjectionStage {
    /// Create a new status injection stage (without ReasoningTree support).
    #[allow(deprecated)]
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self {
            graph,
            reasoning_engine: None,
            protocol_provider: Arc::new(NoOpProtocolProvider),
            config: StatusInjectionConfig::default(),
        }
    }

    /// Create with a ReasoningTreeEngine for full functionality.
    #[allow(deprecated)]
    pub fn with_reasoning(
        graph: Arc<dyn GraphStore>,
        reasoning_engine: Arc<ReasoningTreeEngine>,
    ) -> Self {
        Self {
            graph,
            reasoning_engine: Some(reasoning_engine),
            protocol_provider: Arc::new(NoOpProtocolProvider),
            config: StatusInjectionConfig::default(),
        }
    }

    /// Create with full configuration.
    pub fn with_config(
        graph: Arc<dyn GraphStore>,
        reasoning_engine: Option<Arc<ReasoningTreeEngine>>,
        protocol_provider: Arc<dyn ProtocolStatusProvider>,
        config: StatusInjectionConfig,
    ) -> Self {
        Self {
            graph,
            reasoning_engine,
            protocol_provider,
            config,
        }
    }

    /// Resolve project_id from project_slug via the graph.
    async fn resolve_project_id(&self, slug: &str) -> Option<Uuid> {
        match self.graph.get_project_by_slug(slug).await {
            Ok(Some(project)) => Some(project.id),
            Ok(None) => None,
            Err(e) => {
                warn!(
                    "[status_injection] Failed to resolve project slug '{}': {}",
                    slug, e
                );
                None
            }
        }
    }

    /// Query in-progress plans and their tasks for a project.
    async fn query_work_status(&self, project_id: Uuid) -> Option<String> {
        let (plans, _total) = match self
            .graph
            .list_plans_for_project(
                project_id,
                Some(vec!["in_progress".to_string()]),
                self.config.max_plans,
                0,
            )
            .await
        {
            Ok(result) => result,
            Err(e) => {
                debug!("[status_injection] Failed to list in-progress plans: {}", e);
                return None;
            }
        };

        if plans.is_empty() {
            return None;
        }

        let mut content = String::new();

        for plan in &plans {
            content.push_str(&format!(
                "- **Plan:** {} (priority: {})\n",
                plan.title, plan.priority
            ));

            // Get tasks for this plan
            match self.graph.get_plan_tasks(plan.id).await {
                Ok(tasks) => {
                    let active_tasks: Vec<_> = tasks
                        .iter()
                        .filter(|t| {
                            t.status == TaskStatus::InProgress || t.status == TaskStatus::Blocked
                        })
                        .take(self.config.max_tasks_per_plan)
                        .collect();

                    let completed = tasks
                        .iter()
                        .filter(|t| t.status == TaskStatus::Completed)
                        .count();
                    let total = tasks.len();
                    let progress = if total > 0 {
                        (completed as f64 / total as f64 * 100.0) as u32
                    } else {
                        0
                    };

                    content.push_str(&format!(
                        "  Progress: {}/{}  ({}%)\n",
                        completed, total, progress
                    ));

                    for task in active_tasks {
                        let title = task.title.as_deref().unwrap_or(&task.description);
                        let status_icon = match task.status {
                            TaskStatus::InProgress => "🔄",
                            TaskStatus::Blocked => "🚫",
                            _ => "·",
                        };
                        content.push_str(&format!(
                            "  {} {}: {}\n",
                            status_icon,
                            match task.status {
                                TaskStatus::InProgress => "in_progress",
                                TaskStatus::Blocked => "blocked",
                                _ => "pending",
                            },
                            truncate(title, 80),
                        ));
                    }
                }
                Err(e) => {
                    debug!(
                        "[status_injection] Failed to get tasks for plan {}: {}",
                        plan.id, e
                    );
                }
            }
        }

        if content.is_empty() {
            None
        } else {
            Some(content)
        }
    }

    /// Generate a ReasoningTree from the user message (if engine is available).
    async fn generate_reasoning_tree(
        &self,
        message: &str,
        project_id: Option<Uuid>,
    ) -> Option<ReasoningTree> {
        let engine = self.reasoning_engine.as_ref()?;

        let config = ReasoningTreeConfig::default();
        match engine.build(message, project_id, &config).await {
            Ok(tree) => {
                if tree.confidence >= self.config.reasoning_confidence_threshold {
                    debug!(
                        "[status_injection] ReasoningTree generated: confidence={:.2}, nodes={}",
                        tree.confidence, tree.node_count
                    );
                    Some(tree)
                } else {
                    debug!(
                        "[status_injection] ReasoningTree below threshold: confidence={:.2} < {:.2}",
                        tree.confidence, self.config.reasoning_confidence_threshold
                    );
                    None
                }
            }
            Err(e) => {
                debug!("[status_injection] ReasoningTree generation failed: {}", e);
                None
            }
        }
    }

    /// Render a ReasoningTree into a markdown section.
    fn render_reasoning_tree(tree: &ReasoningTree) -> String {
        let mut content = format!(
            "**Confidence:** {:.0}% | **Nodes:** {}\n\n",
            tree.confidence * 100.0,
            tree.node_count,
        );

        // Collect suggested actions
        let actions = tree.suggested_actions();
        if !actions.is_empty() {
            content.push_str("**Suggested actions:**\n");
            for (i, action) in actions.iter().take(5).enumerate() {
                content.push_str(&format!(
                    "{}. `{}({})` — confidence: {:.0}%\n",
                    i + 1,
                    action.tool,
                    action.action,
                    action.confidence * 100.0,
                ));
            }
        }

        content
    }
}

/// Truncate a string to a maximum length, adding "…" if truncated.
fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        // Find a char boundary
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

#[async_trait::async_trait]
impl ParallelEnrichmentStage for StatusInjectionStage {
    async fn execute(&self, input: &EnrichmentInput) -> Result<StageOutput> {
        let mut output = StageOutput::new(self.name());

        // Need a project scope
        let project_slug = match &input.project_slug {
            Some(slug) => slug.clone(),
            None => return Ok(output), // No project scope, skip
        };

        let project_id = if let Some(id) = input.project_id {
            Some(id)
        } else {
            self.resolve_project_id(&project_slug).await
        };

        let query_timeout = Duration::from_millis(self.config.query_timeout_ms);
        let reasoning_timeout = Duration::from_millis(self.config.reasoning_timeout_ms);

        // ── Query 1: Work status (plans + tasks) ────────────────────────
        let status_content = if let Some(pid) = project_id {
            match timeout(query_timeout, self.query_work_status(pid)).await {
                Ok(content) => content,
                Err(_) => {
                    debug!("[status_injection] Work status query timed out");
                    None
                }
            }
        } else {
            None
        };

        // ── Query 2: Protocol status ─────────────────────────────────────
        let (protocol_content, protocol_context) = if let Some(pid) = project_id {
            match timeout(query_timeout, self.protocol_provider.get_active_runs(pid)).await {
                Ok(Ok(runs)) if !runs.is_empty() => {
                    let mut status_content = String::new();
                    let mut context_content = String::new();

                    for run in &runs {
                        status_content.push_str(&format!(
                            "- **{}**: {} ({}%) — {}\n",
                            run.protocol_name, run.current_state, run.progress, run.status_message,
                        ));

                        // Render prompt_fragment as protocol context
                        if let Some(ref fragment) = run.prompt_fragment {
                            context_content.push_str(&format!(
                                "<protocol_context state=\"{}\" protocol=\"{}\">\n{}\n</protocol_context>\n",
                                run.current_state, run.protocol_name, fragment,
                            ));
                        }
                        // Render forbidden_actions as warnings
                        if let Some(ref actions) = run.forbidden_actions {
                            if !actions.is_empty() {
                                context_content.push_str(&format!(
                                    "\n⚠️ **Forbidden in state '{}'**: {}\n",
                                    run.current_state,
                                    actions.join(", "),
                                ));
                            }
                        }
                    }

                    (
                        Some(status_content),
                        if context_content.is_empty() {
                            None
                        } else {
                            Some(context_content)
                        },
                    )
                }
                Ok(Ok(_)) => (None, None),
                Ok(Err(e)) => {
                    debug!("[status_injection] Protocol status query failed: {}", e);
                    (None, None)
                }
                Err(_) => {
                    debug!("[status_injection] Protocol status query timed out");
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        // ── Query 3: ReasoningTree (optional) ───────────────────────────
        let reasoning_content = if self.reasoning_engine.is_some() {
            match timeout(
                reasoning_timeout,
                self.generate_reasoning_tree(&input.message, project_id),
            )
            .await
            {
                Ok(Some(tree)) => Some(Self::render_reasoning_tree(&tree)),
                Ok(None) => None,
                Err(_) => {
                    debug!("[status_injection] ReasoningTree generation timed out");
                    None
                }
            }
        } else {
            None
        };

        // ── Inject sections ─────────────────────────────────────────────
        if let Some(content) = status_content {
            output.add_section("Work In Progress", content, self.name());
        }

        if let Some(content) = protocol_content {
            output.add_section("Active Protocols", content, self.name());
        }

        if let Some(content) = protocol_context {
            output.add_section("Active Protocol Context", content, self.name());
        }

        if let Some(content) = reasoning_content {
            output.add_section("Suggested Reasoning Plan", content, self.name());
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        "status_injection"
    }

    fn is_enabled(&self, config: &EnrichmentConfig) -> bool {
        config.status_injection
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::enrichment::ParallelEnrichmentStage;

    // ── Config tests ────────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let config = StatusInjectionConfig::default();
        assert_eq!(config.max_plans, 3);
        assert_eq!(config.max_tasks_per_plan, 5);
        assert!((config.reasoning_confidence_threshold - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.query_timeout_ms, 200);
        assert_eq!(config.reasoning_timeout_ms, 300);
    }

    // ── NoOp protocol provider tests ────────────────────────────────────

    #[tokio::test]
    #[allow(deprecated)]
    async fn test_noop_protocol_provider() {
        let provider = NoOpProtocolProvider;
        let runs = provider.get_active_runs(Uuid::new_v4()).await.unwrap();
        assert!(runs.is_empty(), "NoOp provider should return empty runs");
    }

    // ── GraphProtocolProvider tests ─────────────────────────────────────

    #[tokio::test]
    async fn test_graph_provider_no_protocols() {
        let mock = crate::neo4j::mock::MockGraphStore::new();
        let provider = GraphProtocolProvider::new(Arc::new(mock));
        let runs = provider.get_active_runs(Uuid::new_v4()).await.unwrap();
        assert!(runs.is_empty());
    }

    #[tokio::test]
    async fn test_graph_provider_with_running_run() {
        let mock = crate::neo4j::mock::MockGraphStore::new();
        let project_id = Uuid::new_v4();

        // Create project
        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test-project".to_string(),
            slug: "test-project".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
        };
        mock.create_project(&project).await.unwrap();

        // Create protocol
        let start_id = Uuid::new_v4();
        let mut proto = crate::protocol::Protocol::new(project_id, "code-review", start_id);

        // Create states with prompt_fragment
        let state_analyze = crate::protocol::ProtocolState {
            id: start_id,
            protocol_id: proto.id,
            name: "analyze".to_string(),
            description: "Analyze code".to_string(),
            action: None,
            state_type: crate::protocol::StateType::Start,
            sub_protocol_id: None,
            completion_strategy: None,
            on_failure_strategy: None,
            generator_config: None,
            prompt_fragment: Some(
                "Focus on analyzing code changes. Use code(search) and code(analyze_impact)."
                    .to_string(),
            ),
            available_tools: Some(vec!["code".to_string(), "note".to_string()]),
            forbidden_actions: Some(vec![
                "Do not create commits".to_string(),
                "Do not modify files".to_string(),
            ]),
            state_timeout_secs: None,
        };

        let done_id = Uuid::new_v4();
        let state_done = crate::protocol::ProtocolState {
            id: done_id,
            protocol_id: proto.id,
            name: "done".to_string(),
            description: "Complete".to_string(),
            action: None,
            state_type: crate::protocol::StateType::Terminal,
            sub_protocol_id: None,
            completion_strategy: None,
            on_failure_strategy: None,
            generator_config: None,
            prompt_fragment: None,
            available_tools: None,
            forbidden_actions: None,
            state_timeout_secs: None,
        };

        proto.terminal_states = vec![done_id];
        mock.upsert_protocol(&proto).await.unwrap();
        mock.upsert_protocol_state(&state_analyze).await.unwrap();
        mock.upsert_protocol_state(&state_done).await.unwrap();

        // Create a running run in the "analyze" state
        let run = crate::protocol::ProtocolRun {
            id: Uuid::new_v4(),
            protocol_id: proto.id,
            plan_id: None,
            task_id: None,
            parent_run_id: None,
            current_state: start_id,
            states_visited: vec![crate::protocol::StateVisit {
                state_id: start_id,
                state_name: "analyze".to_string(),
                entered_at: chrono::Utc::now(),
                exited_at: None,
                duration_ms: None,
                trigger: None,
                progress_snapshot: None,
            }],
            status: crate::protocol::RunStatus::Running,
            started_at: chrono::Utc::now(),
            completed_at: None,
            error: None,
            triggered_by: "manual".to_string(),
            depth: 0,
            version: 1,
            runner_managed: false,
        };
        mock.create_protocol_run(&run).await.unwrap();

        // Query
        let provider = GraphProtocolProvider::new(Arc::new(mock));
        let runs = provider.get_active_runs(project_id).await.unwrap();

        assert_eq!(runs.len(), 1);
        let r = &runs[0];
        assert_eq!(r.protocol_name, "code-review");
        assert_eq!(r.current_state, "analyze");
        assert!(r.prompt_fragment.is_some());
        assert!(r
            .prompt_fragment
            .as_ref()
            .unwrap()
            .contains("analyzing code"));
        assert_eq!(
            r.available_tools,
            Some(vec!["code".to_string(), "note".to_string()])
        );
        assert_eq!(r.forbidden_actions.as_ref().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_graph_provider_no_running_runs() {
        let mock = crate::neo4j::mock::MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test-project".to_string(),
            slug: "test-project".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
        };
        mock.create_project(&project).await.unwrap();

        let start_id = Uuid::new_v4();
        let proto = crate::protocol::Protocol::new(project_id, "code-review", start_id);
        mock.upsert_protocol(&proto).await.unwrap();

        let state = crate::protocol::ProtocolState::new(proto.id, "analyze");
        mock.upsert_protocol_state(&state).await.unwrap();

        // No runs created → empty result
        let provider = GraphProtocolProvider::new(Arc::new(mock));
        let runs = provider.get_active_runs(project_id).await.unwrap();
        assert!(runs.is_empty());
    }

    // ── truncate tests ──────────────────────────────────────────────────

    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_exact() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_long() {
        let result = truncate("hello world", 5);
        assert_eq!(result, "hello");
    }

    // ── Stage basic tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_stage_name() {
        let stage = make_test_stage();
        assert_eq!(stage.name(), "status_injection");
    }

    #[tokio::test]
    async fn test_stage_is_enabled() {
        let stage = make_test_stage();
        let config = EnrichmentConfig {
            status_injection: true,
            ..Default::default()
        };
        assert!(stage.is_enabled(&config));

        let disabled = EnrichmentConfig {
            status_injection: false,
            ..Default::default()
        };
        assert!(!stage.is_enabled(&disabled));
    }

    #[tokio::test]
    async fn test_stage_no_project_skips() {
        let stage = make_test_stage();
        let input = EnrichmentInput {
            message: "Hello world".to_string(),
            session_id: Uuid::new_v4(),
            project_slug: None,
            project_id: None,
            cwd: None,
            protocol_run_id: None,
            protocol_state: None,
            excluded_note_ids: Default::default(),
            reasoning_path_tracker: None,
        };
        let output = stage.execute(&input).await.unwrap();
        assert!(
            output.sections.is_empty(),
            "Should skip when no project scope"
        );
    }

    #[tokio::test]
    async fn test_stage_with_project_runs_without_error() {
        let stage = make_test_stage();
        let input = EnrichmentInput {
            message: "What tasks are in progress?".to_string(),
            session_id: Uuid::new_v4(),
            project_slug: Some("test-project".to_string()),
            project_id: None,
            cwd: None,
            protocol_run_id: None,
            protocol_state: None,
            excluded_note_ids: Default::default(),
            reasoning_path_tracker: None,
        };

        // With mock graph (no data), should complete without errors
        let result = stage.execute(&input).await;
        assert!(result.is_ok(), "Stage should not error with mock graph");
    }

    // ── ReasoningTree rendering tests ───────────────────────────────────

    #[test]
    fn test_render_reasoning_tree_basic() {
        use crate::reasoning::models::{Action, EntitySource, ReasoningNode};

        let mut tree = ReasoningTree::new("test request", None);

        let mut root = ReasoningNode::new(
            EntitySource::Note,
            Uuid::new_v4().to_string(),
            0.9,
            "High-relevance tip note for test",
        );
        root.action = Some(Action {
            tool: "code".to_string(),
            action: "get_file_symbols".to_string(),
            params: serde_json::json!({"file_path": "src/main.rs"}),
            confidence: 0.85,
        });
        tree.add_root(root);

        let rendered = StatusInjectionStage::render_reasoning_tree(&tree);
        assert!(rendered.contains("Confidence:"));
        assert!(rendered.contains("Suggested actions:"));
        assert!(rendered.contains("code(get_file_symbols)"));
    }

    #[test]
    fn test_render_reasoning_tree_no_actions() {
        use crate::reasoning::models::{EntitySource, ReasoningNode};

        let mut tree = ReasoningTree::new("test request", None);

        // Add a root node without an action — tree will have confidence > 0
        let root = ReasoningNode::new(
            EntitySource::Note,
            "note-1",
            0.6,
            "A context note without suggested action",
        );
        tree.add_root(root);

        let rendered = StatusInjectionStage::render_reasoning_tree(&tree);
        assert!(rendered.contains("Confidence:"));
        assert!(!rendered.contains("Suggested actions:"));
    }

    // ── Protocol status provider contract ───────────────────────────────

    #[test]
    fn test_protocol_run_status_serializable() {
        let status = ProtocolRunStatus {
            protocol_name: "code_review".to_string(),
            current_state: "awaiting_approval".to_string(),
            progress: 75,
            status_message: "Waiting for reviewer".to_string(),
            prompt_fragment: None,
            available_tools: None,
            forbidden_actions: None,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("code_review"));
        assert!(json.contains("75"));
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    fn make_test_stage() -> StatusInjectionStage {
        use crate::neo4j::mock::MockGraphStore;
        let graph = Arc::new(MockGraphStore::new());
        StatusInjectionStage::new(graph)
    }
}
