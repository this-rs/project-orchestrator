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

        // 2. Running runs of every protocol, concurrently (one round trip each, not one
        //    after the other: on a project with many protocols the sequential loop alone
        //    exceeded the stage's budget), and the states once per protocol that has runs.
        let per_protocol = futures::future::join_all(protocols.iter().map(|proto| async move {
            let (runs, _) = self
                .graph
                .list_protocol_runs(proto.id, Some(crate::protocol::RunStatus::Running), 10, 0)
                .await?;
            let states = if runs.is_empty() {
                Vec::new()
            } else {
                self.graph.get_protocol_states(proto.id).await?
            };
            anyhow::Ok((proto, runs, states))
        }))
        .await;

        let mut results = Vec::new();
        for outcome in per_protocol {
            let (proto, runs, states) = outcome?;
            for run in &runs {
                // 3. The current state's prompt fields
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

        // The tasks of every plan, concurrently.
        let tasks_per_plan =
            futures::future::join_all(plans.iter().map(|plan| self.graph.get_plan_tasks(plan.id)))
                .await;

        for (plan, plan_tasks) in plans.iter().zip(tasks_per_plan) {
            content.push_str(&format!(
                "- **Plan:** {} (priority: {})\n",
                plan.title, plan.priority
            ));

            match plan_tasks {
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

        // The three queries run CONCURRENTLY, each under its own limit. One after the other
        // their limits added up (200 + 200 + 300 ms) to more than the stage's own budget, so
        // a slow graph timed the whole stage out and every section was lost.
        let work_status = async {
            if let Some(pid) = project_id {
                match timeout(query_timeout, self.query_work_status(pid)).await {
                    Ok(content) => content,
                    Err(_) => {
                        debug!("[status_injection] Work status query timed out");
                        None
                    }
                }
            } else {
                None
            }
        };
        let protocol_status = async {
            if let Some(pid) = project_id {
                match timeout(query_timeout, self.protocol_provider.get_active_runs(pid)).await {
                    Ok(Ok(runs)) if !runs.is_empty() => {
                        let mut status_content = String::new();
                        let mut context_content = String::new();

                        for run in &runs {
                            status_content.push_str(&format!(
                                "- **{}**: {} ({}%) — {}\n",
                                run.protocol_name,
                                run.current_state,
                                run.progress,
                                run.status_message,
                            ));

                            // Render prompt_fragment as protocol context
                            if let Some(ref fragment) = run.prompt_fragment {
                                context_content.push_str(&format!(
                                "<protocol_context state=\"{}\" protocol=\"{}\">\n{}\n</protocol_context>\n",
                                crate::chat::untrusted::escape_attr(&run.current_state),
                                crate::chat::untrusted::escape_attr(&run.protocol_name),
                                fragment,
                            ));
                            }
                            // Render forbidden_actions as warnings
                            if let Some(ref actions) = run.forbidden_actions {
                                if !actions.is_empty() {
                                    context_content.push_str(&format!(
                                        "\n⚠️ **Forbidden in state '{}'**: {}\n",
                                        crate::chat::untrusted::escape_attr(&run.current_state),
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
            }
        };
        let reasoning = async {
            if self.reasoning_engine.is_some() {
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
            }
        };
        let (status_content, (protocol_content, protocol_context), reasoning_content) =
            tokio::join!(work_status, protocol_status, reasoning);

        // ── Inject sections ─────────────────────────────────────────────
        if let Some(content) = status_content {
            output.add_section(
                "Work In Progress",
                // Plan and task titles are graph content: data.
                crate::chat::untrusted::wrap_graph(
                    &content,
                    "plan_status",
                    input.project_slug.as_deref(),
                ),
                self.name(),
                crate::chat::enrichment::EnrichmentSource::StatusInjection,
            );
        }

        if let Some(content) = protocol_content {
            output.add_section(
                "Active Protocols",
                // Protocol names and status messages: data.
                crate::chat::untrusted::wrap_graph(
                    &content,
                    "protocol",
                    input.project_slug.as_deref(),
                ),
                self.name(),
                crate::chat::enrichment::EnrichmentSource::StatusInjection,
            );
        }

        if let Some(content) = protocol_context {
            output.add_section(
                "Active Protocol Context",
                content,
                self.name(),
                crate::chat::enrichment::EnrichmentSource::StatusInjection,
            );
        }

        if let Some(content) = reasoning_content {
            output.add_section(
                "Suggested Reasoning Plan",
                content,
                self.name(),
                crate::chat::enrichment::EnrichmentSource::StatusInjection,
            );
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
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::models::PlanStatus;

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
            watch_enabled: true,
            profile: Default::default(),
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
            watch_enabled: true,
            profile: Default::default(),
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

    /// The runs of every protocol are asked for concurrently, and the states only of the
    /// protocols that have one: a protocol with no running run costs nothing more than
    /// its (empty) run list, and a protocol with one reports its current state's name.
    #[tokio::test]
    async fn test_graph_provider_queries_every_protocol_concurrently() {
        let mock = MockGraphStore::new();
        let project = crate::test_helpers::test_project_named("concurrent");
        let project_id = project.id;
        mock.create_project(&project).await.unwrap();

        // A protocol with a running run...
        let mut busy = crate::protocol::Protocol::new(project_id, "busy", Uuid::new_v4());
        let busy_state = crate::protocol::ProtocolState::start(busy.id, "running");
        let busy_done = crate::protocol::ProtocolState::terminal(busy.id, "done");
        busy.entry_state = busy_state.id;
        busy.terminal_states = vec![busy_done.id];
        mock.upsert_protocol(&busy).await.unwrap();
        mock.upsert_protocol_state(&busy_state).await.unwrap();
        mock.upsert_protocol_state(&busy_done).await.unwrap();
        mock.create_protocol_run(&running_run(busy.id, busy_state.id))
            .await
            .unwrap();

        // ... and one with none (its states are never read).
        let mut quiet = crate::protocol::Protocol::new(project_id, "quiet", Uuid::new_v4());
        let quiet_state = crate::protocol::ProtocolState::start(quiet.id, "waiting");
        quiet.entry_state = quiet_state.id;
        mock.upsert_protocol(&quiet).await.unwrap();
        mock.upsert_protocol_state(&quiet_state).await.unwrap();

        let provider = GraphProtocolProvider::new(Arc::new(mock));
        let runs = provider.get_active_runs(project_id).await.unwrap();

        assert_eq!(runs.len(), 1, "only the busy protocol has a run");
        assert_eq!(runs[0].protocol_name, "busy");
        assert_eq!(runs[0].current_state, "running");
        assert_eq!(runs[0].progress, 50, "one of its two states is visited");
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
            restricted_content: false,
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
            restricted_content: false,
            reasoning_path_tracker: None,
        };

        // With mock graph (no data), should complete without errors
        let result = stage.execute(&input).await;
        assert!(result.is_ok(), "Stage should not error with mock graph");
    }

    // ── Stage with a graph that has content ─────────────────────────────

    /// Every in-progress plan's tasks are fetched concurrently (one query per plan) and
    /// the section lists each plan with its progress and its active tasks, completed
    /// ones being counted but not listed.
    #[tokio::test]
    async fn test_stage_injects_work_status_for_several_plans() {
        let mock = Arc::new(MockGraphStore::new());
        let project = crate::test_helpers::test_project_named("work-status");
        let project_id = project.id;
        mock.create_project(&project).await.unwrap();

        let mut first = crate::test_helpers::test_plan_for_project(project_id);
        first.title = "First plan".to_string();
        first.status = PlanStatus::InProgress;
        mock.create_plan(&first).await.unwrap();
        mock.create_task(
            first.id,
            &task_with_status("Wire the stage", TaskStatus::InProgress),
        )
        .await
        .unwrap();
        mock.create_task(
            first.id,
            &task_with_status("Settle the config", TaskStatus::Completed),
        )
        .await
        .unwrap();

        let mut second = crate::test_helpers::test_plan_for_project(project_id);
        second.title = "Second plan".to_string();
        second.status = PlanStatus::InProgress;
        mock.create_plan(&second).await.unwrap();
        mock.create_task(
            second.id,
            &task_with_status("Await a decision", TaskStatus::Blocked),
        )
        .await
        .unwrap();

        let stage = StatusInjectionStage::new(mock);
        let output = stage
            .execute(&input_for(Some(project_id), "What is in progress?"))
            .await
            .unwrap();

        let section = output
            .sections
            .iter()
            .find(|s| s.title == "Work In Progress")
            .expect("a work-status section");
        let content = &section.content;
        assert!(content.contains("**Plan:** First plan"), "{content}");
        assert!(content.contains("**Plan:** Second plan"), "{content}");
        assert!(content.contains("Progress: 1/2  (50%)"), "{content}");
        assert!(content.contains("Wire the stage"), "{content}");
        assert!(content.contains("Await a decision"), "{content}");
        assert!(
            !content.contains("Settle the config"),
            "a completed task is counted, not listed: {content}"
        );
    }

    /// A running protocol run is rendered twice: its status line, and its current
    /// state's context (prompt fragment, forbidden actions), through the real graph
    /// provider.
    #[tokio::test]
    async fn test_stage_injects_active_protocol_and_its_context() {
        let mock = Arc::new(MockGraphStore::new());
        let project = crate::test_helpers::test_project_named("protocols");
        let project_id = project.id;
        mock.create_project(&project).await.unwrap();

        let mut proto = crate::protocol::Protocol::new(project_id, "code-review", Uuid::new_v4());
        let mut state = crate::protocol::ProtocolState::start(proto.id, "analyze");
        state.prompt_fragment = Some("Read the diff before you review it.".to_string());
        state.forbidden_actions = Some(vec!["git push".to_string()]);
        proto.entry_state = state.id;
        mock.upsert_protocol(&proto).await.unwrap();
        mock.upsert_protocol_state(&state).await.unwrap();
        mock.create_protocol_run(&running_run(proto.id, state.id))
            .await
            .unwrap();

        let stage = StatusInjectionStage::with_config(
            Arc::clone(&mock) as Arc<dyn GraphStore>,
            None,
            Arc::new(GraphProtocolProvider::new(mock)),
            StatusInjectionConfig::default(),
        );
        let output = stage
            .execute(&input_for(Some(project_id), "Where are we?"))
            .await
            .unwrap();

        let status = output
            .sections
            .iter()
            .find(|s| s.title == "Active Protocols")
            .expect("a protocol-status section");
        assert!(status.content.contains("code-review"), "{}", status.content);
        assert!(status.content.contains("analyze"), "{}", status.content);

        let context = output
            .sections
            .iter()
            .find(|s| s.title == "Active Protocol Context")
            .expect("a protocol-context section");
        assert!(
            context
                .content
                .contains("Read the diff before you review it."),
            "{}",
            context.content
        );
        assert!(context.content.contains("git push"), "{}", context.content);
    }

    /// A run whose state carries neither a prompt fragment nor forbidden actions has no
    /// context to inject: the status line stands alone, no empty context section.
    #[tokio::test]
    async fn test_stage_skips_protocol_context_when_the_state_brings_none() {
        let mock = Arc::new(MockGraphStore::new());
        let project = crate::test_helpers::test_project_named("bare-protocol");
        let project_id = project.id;
        mock.create_project(&project).await.unwrap();

        let mut proto = crate::protocol::Protocol::new(project_id, "deploy", Uuid::new_v4());
        let state = crate::protocol::ProtocolState::start(proto.id, "uploading");
        proto.entry_state = state.id;
        mock.upsert_protocol(&proto).await.unwrap();
        mock.upsert_protocol_state(&state).await.unwrap();
        mock.create_protocol_run(&running_run(proto.id, state.id))
            .await
            .unwrap();

        let stage = StatusInjectionStage::with_config(
            Arc::clone(&mock) as Arc<dyn GraphStore>,
            None,
            Arc::new(GraphProtocolProvider::new(mock)),
            StatusInjectionConfig::default(),
        );
        let output = stage
            .execute(&input_for(Some(project_id), "Where are we?"))
            .await
            .unwrap();

        let status = output
            .sections
            .iter()
            .find(|s| s.title == "Active Protocols")
            .expect("a protocol-status section");
        assert!(status.content.contains("deploy"), "{}", status.content);
        assert!(
            output
                .sections
                .iter()
                .all(|s| s.title != "Active Protocol Context"),
            "nothing to inject as context"
        );
    }

    /// A protocol query that answers with no run adds no section at all: an empty
    /// answer is not an empty section.
    #[tokio::test]
    async fn test_stage_skips_protocol_sections_when_no_run_is_active() {
        let mock = Arc::new(MockGraphStore::new());
        let project = crate::test_helpers::test_project_named("quiet");
        mock.create_project(&project).await.unwrap();

        let stage = StatusInjectionStage::new(mock);
        let output = stage
            .execute(&input_for(Some(project.id), "Where are we?"))
            .await
            .unwrap();

        assert!(
            output.sections.is_empty(),
            "nothing to say: {:?}",
            output.sections.iter().map(|s| &s.title).collect::<Vec<_>>()
        );
    }

    /// A provider that fails costs its section, not the stage.
    #[tokio::test]
    async fn test_stage_survives_a_failing_protocol_provider() {
        let mock = Arc::new(MockGraphStore::new());
        let project = crate::test_helpers::test_project_named("failing-protocol");
        let project_id = project.id;
        mock.create_project(&project).await.unwrap();

        let mut plan = crate::test_helpers::test_plan_for_project(project_id);
        plan.title = "Still worth saying".to_string();
        plan.status = PlanStatus::InProgress;
        mock.create_plan(&plan).await.unwrap();

        let stage = StatusInjectionStage::with_config(
            mock,
            None,
            Arc::new(FailingProvider),
            StatusInjectionConfig::default(),
        );
        let output = stage
            .execute(&input_for(Some(project_id), "Where are we?"))
            .await
            .unwrap();

        assert!(
            output
                .sections
                .iter()
                .all(|s| s.title != "Active Protocols"),
            "the failed query adds nothing"
        );
        let work = output
            .sections
            .iter()
            .find(|s| s.title == "Work In Progress")
            .expect("the other queries still answered");
        assert!(
            work.content.contains("Still worth saying"),
            "{}",
            work.content
        );
    }

    /// A protocol query slower than its own limit is dropped at that limit: the stage
    /// returns without protocol sections instead of waiting for it.
    #[tokio::test]
    async fn test_stage_protocol_query_times_out() {
        let mock = Arc::new(MockGraphStore::new());
        let project = crate::test_helpers::test_project_named("slow-protocol");
        mock.create_project(&project).await.unwrap();

        let stage = StatusInjectionStage::with_config(
            mock,
            None,
            Arc::new(SlowProvider {
                delay: Duration::from_secs(5),
            }),
            StatusInjectionConfig {
                query_timeout_ms: 10,
                ..Default::default()
            },
        );
        let started = std::time::Instant::now();
        let output = stage
            .execute(&input_for(Some(project.id), "Where are we?"))
            .await
            .unwrap();

        assert!(output.sections.is_empty());
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the stage did not wait for the slow query"
        );
    }

    /// The work-status query has a limit of its own: blocked on the plans store, it is
    /// dropped there and the stage still answers.
    #[tokio::test]
    async fn test_stage_work_status_query_times_out() {
        let mock = Arc::new(MockGraphStore::new());
        let project = crate::test_helpers::test_project_named("slow-plans");
        let project_id = project.id;
        mock.create_project(&project).await.unwrap();
        let mut plan = crate::test_helpers::test_plan_for_project(project_id);
        plan.title = "Never read in time".to_string();
        plan.status = PlanStatus::InProgress;
        mock.create_plan(&plan).await.unwrap();

        let stage = StatusInjectionStage::with_config(
            Arc::clone(&mock) as Arc<dyn GraphStore>,
            None,
            Arc::new(GraphProtocolProvider::new(
                Arc::clone(&mock) as Arc<dyn GraphStore>
            )),
            StatusInjectionConfig {
                query_timeout_ms: 10,
                ..Default::default()
            },
        );

        // Holding the plans store keeps the work-status query pending past its limit.
        let held = mock.plans.write().await;
        let started = std::time::Instant::now();
        let output = stage
            .execute(&input_for(Some(project_id), "Where are we?"))
            .await
            .unwrap();
        drop(held);

        assert!(output.sections.is_empty());
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the stage did not wait for the blocked query"
        );
    }

    /// With an engine available, the stage renders its tree: high-confidence trees are
    /// injected, ones under the threshold are not.
    #[tokio::test]
    async fn test_stage_injects_a_reasoning_tree() {
        let mock = Arc::new(MockGraphStore::new());
        let project = crate::test_helpers::test_project_named("reasoning");
        mock.create_project(&project).await.unwrap();

        let stage = StatusInjectionStage::with_config(
            Arc::clone(&mock) as Arc<dyn GraphStore>,
            Some(reasoning_engine(
                Arc::clone(&mock) as Arc<dyn GraphStore>,
                Arc::new(crate::embeddings::MockEmbeddingProvider::new(768)),
            )),
            Arc::new(GraphProtocolProvider::new(mock)),
            StatusInjectionConfig {
                // An empty graph makes a confidence-0 tree: the threshold decides.
                reasoning_confidence_threshold: 0.0,
                ..Default::default()
            },
        );
        let output = stage
            .execute(&input_for(Some(project.id), "How does the chat work?"))
            .await
            .unwrap();

        let section = output
            .sections
            .iter()
            .find(|s| s.title == "Suggested Reasoning Plan")
            .expect("a reasoning section");
        assert!(
            section.content.contains("**Confidence:** 0%"),
            "{}",
            section.content
        );
        assert!(
            section.content.contains("**Nodes:** 0"),
            "{}",
            section.content
        );
    }

    /// The same tree, under the default threshold: no section.
    #[tokio::test]
    async fn test_stage_skips_a_reasoning_tree_under_the_threshold() {
        let mock = Arc::new(MockGraphStore::new());
        let project = crate::test_helpers::test_project_named("reasoning-quiet");
        mock.create_project(&project).await.unwrap();

        let stage = StatusInjectionStage::with_config(
            Arc::clone(&mock) as Arc<dyn GraphStore>,
            Some(reasoning_engine(
                Arc::clone(&mock) as Arc<dyn GraphStore>,
                Arc::new(crate::embeddings::MockEmbeddingProvider::new(768)),
            )),
            Arc::new(GraphProtocolProvider::new(mock)),
            StatusInjectionConfig::default(),
        );
        let output = stage
            .execute(&input_for(Some(project.id), "How does the chat work?"))
            .await
            .unwrap();

        assert!(
            output
                .sections
                .iter()
                .all(|s| s.title != "Suggested Reasoning Plan"),
            "a tree under the threshold is not injected"
        );
    }

    /// A slow embedding holds the tree past the reasoning limit: the tree is dropped,
    /// the stage answers.
    #[tokio::test]
    async fn test_stage_reasoning_query_times_out() {
        let mock = Arc::new(MockGraphStore::new());
        let project = crate::test_helpers::test_project_named("slow-reasoning");
        mock.create_project(&project).await.unwrap();

        let stage = StatusInjectionStage::with_config(
            Arc::clone(&mock) as Arc<dyn GraphStore>,
            Some(reasoning_engine(
                Arc::clone(&mock) as Arc<dyn GraphStore>,
                Arc::new(SlowEmbedder {
                    delay: Duration::from_secs(5),
                }),
            )),
            Arc::new(GraphProtocolProvider::new(mock)),
            StatusInjectionConfig {
                reasoning_timeout_ms: 10,
                ..Default::default()
            },
        );
        let started = std::time::Instant::now();
        let output = stage
            .execute(&input_for(Some(project.id), "How does the chat work?"))
            .await
            .unwrap();

        assert!(output.sections.is_empty());
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the stage did not wait for the slow tree"
        );
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

    /// A running run of `protocol_id`, its current state being `current_state`.
    fn running_run(protocol_id: Uuid, current_state: Uuid) -> crate::protocol::ProtocolRun {
        crate::protocol::ProtocolRun {
            id: Uuid::new_v4(),
            protocol_id,
            plan_id: None,
            task_id: None,
            parent_run_id: None,
            current_state,
            states_visited: vec![crate::protocol::StateVisit {
                state_id: current_state,
                state_name: "running".to_string(),
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
        }
    }

    /// The input of a stage call already scoped to `project_id` (no slug resolution).
    fn input_for(project_id: Option<Uuid>, message: &str) -> EnrichmentInput {
        EnrichmentInput {
            message: message.to_string(),
            session_id: Uuid::new_v4(),
            project_slug: Some("test-project".to_string()),
            project_id,
            cwd: None,
            protocol_run_id: None,
            protocol_state: None,
            excluded_note_ids: Default::default(),
            restricted_content: false,
            reasoning_path_tracker: None,
        }
    }

    fn task_with_status(title: &str, status: TaskStatus) -> crate::neo4j::models::TaskNode {
        let mut task = crate::test_helpers::test_task_titled(title);
        task.status = status;
        task
    }

    fn reasoning_engine(
        graph: Arc<dyn GraphStore>,
        embedder: Arc<dyn crate::embeddings::EmbeddingProvider>,
    ) -> Arc<ReasoningTreeEngine> {
        Arc::new(ReasoningTreeEngine::new(
            graph,
            embedder,
            crate::reasoning::ReasoningTreeCache::with_config(10, Duration::from_secs(60)),
        ))
    }

    /// A protocol provider that always fails.
    struct FailingProvider;

    #[async_trait::async_trait]
    impl ProtocolStatusProvider for FailingProvider {
        async fn get_active_runs(&self, _project_id: Uuid) -> Result<Vec<ProtocolRunStatus>> {
            anyhow::bail!("the graph is unreachable")
        }
    }

    /// A protocol provider that answers after `delay`.
    struct SlowProvider {
        delay: Duration,
    }

    #[async_trait::async_trait]
    impl ProtocolStatusProvider for SlowProvider {
        async fn get_active_runs(&self, _project_id: Uuid) -> Result<Vec<ProtocolRunStatus>> {
            tokio::time::sleep(self.delay).await;
            Ok(Vec::new())
        }
    }

    /// An embedding provider that answers after `delay` (to hold a tree past its limit).
    struct SlowEmbedder {
        delay: Duration,
    }

    #[async_trait::async_trait]
    impl crate::embeddings::EmbeddingProvider for SlowEmbedder {
        async fn embed_text(&self, _text: &str) -> Result<Vec<f32>> {
            tokio::time::sleep(self.delay).await;
            Ok(vec![0.0; 8])
        }

        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            tokio::time::sleep(self.delay).await;
            Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
        }

        fn dimensions(&self) -> usize {
            8
        }

        fn model_name(&self) -> &str {
            "slow-test-embedding"
        }
    }

    fn make_test_stage() -> StatusInjectionStage {
        use crate::neo4j::mock::MockGraphStore;
        let graph = Arc::new(MockGraphStore::new());
        StatusInjectionStage::new(graph)
    }
}
