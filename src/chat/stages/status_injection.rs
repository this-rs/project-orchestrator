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
    EnrichmentConfig, EnrichmentContext, EnrichmentInput, EnrichmentStage,
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
#[allow(dead_code)] // Fields will be used when Pattern Federation is implemented
pub struct ProtocolRunStatus {
    /// Protocol/pattern name (e.g., "code_review", "deploy_pipeline")
    pub protocol_name: String,
    /// Current FSM state (e.g., "awaiting_approval", "running_tests")
    pub current_state: String,
    /// Progress percentage (0-100)
    pub progress: u8,
    /// Human-readable status message
    pub status_message: String,
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

/// No-op implementation that always returns empty (stub for T3).
pub struct NoOpProtocolProvider;

#[async_trait::async_trait]
impl ProtocolStatusProvider for NoOpProtocolProvider {
    async fn get_active_runs(&self, _project_id: Uuid) -> Result<Vec<ProtocolRunStatus>> {
        Ok(Vec::new())
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
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self {
            graph,
            reasoning_engine: None,
            protocol_provider: Arc::new(NoOpProtocolProvider),
            config: StatusInjectionConfig::default(),
        }
    }

    /// Create with a ReasoningTreeEngine for full functionality.
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
impl EnrichmentStage for StatusInjectionStage {
    async fn execute(&self, input: &EnrichmentInput, ctx: &mut EnrichmentContext) -> Result<()> {
        // Need a project scope
        let project_slug = match &input.project_slug {
            Some(slug) => slug.clone(),
            None => return Ok(()), // No project scope, skip
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

        // ── Query 2: Protocol status (stub) ─────────────────────────────
        let protocol_content = if let Some(pid) = project_id {
            match timeout(query_timeout, self.protocol_provider.get_active_runs(pid)).await {
                Ok(Ok(runs)) if !runs.is_empty() => {
                    let mut content = String::new();
                    for run in &runs {
                        content.push_str(&format!(
                            "- **{}**: {} ({}%) — {}\n",
                            run.protocol_name, run.current_state, run.progress, run.status_message,
                        ));
                    }
                    Some(content)
                }
                Ok(Ok(_)) => None, // Empty runs
                Ok(Err(e)) => {
                    debug!("[status_injection] Protocol status query failed: {}", e);
                    None
                }
                Err(_) => {
                    debug!("[status_injection] Protocol status query timed out");
                    None
                }
            }
        } else {
            None
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
            ctx.add_section("Work In Progress", content, self.name());
        }

        if let Some(content) = protocol_content {
            ctx.add_section("Active Protocols", content, self.name());
        }

        if let Some(content) = reasoning_content {
            ctx.add_section("Suggested Reasoning Plan", content, self.name());
        }

        Ok(())
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
    use crate::chat::enrichment::EnrichmentContext;

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
    async fn test_noop_protocol_provider() {
        let provider = NoOpProtocolProvider;
        let runs = provider.get_active_runs(Uuid::new_v4()).await.unwrap();
        assert!(runs.is_empty(), "NoOp provider should return empty runs");
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
        };
        let mut ctx = EnrichmentContext::default();
        stage.execute(&input, &mut ctx).await.unwrap();
        assert!(!ctx.has_content(), "Should skip when no project scope");
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
        };
        let mut ctx = EnrichmentContext::default();

        // With mock graph (no data), should complete without errors
        let result = stage.execute(&input, &mut ctx).await;
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
