//! ForkContextStage — Injects fork/sub-conversation context into the enrichment pipeline.
//!
//! When the current session is part of a fork tree, this stage adds:
//! 1. **For parent sessions**: summaries of completed child forks + status of active forks
//! 2. **For fork sessions**: the parent scope (why this fork exists, its task/persona context)
//!
//! This allows the parent to see results from child forks, and child forks to
//! understand their scope within the larger conversation.
//!
//! # Budget
//! - Max 500 tokens output
//! - Single Neo4j query for fork children + one for parent metadata
//! - Controlled by `ENRICHMENT_FORK_CONTEXT` env var (default: true)

use crate::chat::enrichment::{
    EnrichmentConfig, EnrichmentInput, EnrichmentSource, ParallelEnrichmentStage, StageOutput,
};
use crate::neo4j::traits::GraphStore;
use anyhow::Result;
use std::sync::Arc;
use tracing::debug;

/// ForkContextStage injects fork tree awareness into the prompt.
pub struct ForkContextStage {
    graph_store: Arc<dyn GraphStore>,
}

impl ForkContextStage {
    pub fn new(graph_store: Arc<dyn GraphStore>) -> Self {
        Self { graph_store }
    }

    /// Build context for a parent session: show completed fork summaries + active forks.
    async fn build_parent_context(&self, session_id: &str) -> Result<Option<String>> {
        let children = self
            .graph_store
            .get_fork_children(session_id, false)
            .await?;

        if children.is_empty() {
            return Ok(None);
        }

        let mut lines = Vec::new();
        lines.push("## Active Sub-conversations (Forks)".to_string());

        let mut has_content = false;

        for child in &children {
            let status = child.fork_status.as_deref().unwrap_or("unknown");
            let fork_type = child.fork_type.as_deref().unwrap_or("unknown");
            let title = child.title.as_deref().unwrap_or("Untitled");

            let intent = child.fork_intent.as_deref().unwrap_or("unknown");

            match status {
                "completed" => {
                    lines.push(format!(
                        "- **[Completed|{}]** {} (type: {}, messages: {})",
                        intent, title, fork_type, child.message_count
                    ));
                    has_content = true;
                }
                "active" => {
                    lines.push(format!(
                        "- **[Active|{}]** {} (type: {}, messages: {})",
                        intent, title, fork_type, child.message_count
                    ));
                    has_content = true;
                }
                "cancelled" => {
                    // Don't show cancelled forks
                }
                _ => {
                    lines.push(format!(
                        "- **[{}]** {} (type: {})",
                        status, title, fork_type
                    ));
                    has_content = true;
                }
            }
        }

        if !has_content {
            return Ok(None);
        }

        Ok(Some(lines.join("\n")))
    }

    /// Build context for a fork session: show why it was created and its scope.
    /// Intent-aware: injects different behavioral guidance based on fork_intent.
    async fn build_fork_context(
        &self,
        session_id: &str,
        fork_depth: u32,
        fork_type: &str,
        fork_intent: Option<&str>,
        context_snapshot: Option<&str>,
    ) -> Result<Option<String>> {
        let mut lines = Vec::new();
        let intent_label = fork_intent.unwrap_or("unknown");
        lines.push(format!(
            "## Fork Context (depth: {}, type: {}, intent: {})",
            fork_depth, fork_type, intent_label
        ));
        lines.push("You are operating in a **sub-conversation** (fork).".to_string());

        // Parse context snapshot for task/persona info
        let snapshot_value =
            context_snapshot.and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok());

        if let Some(ref v) = snapshot_value {
            if let Some(task_id) = v.get("task_id").and_then(|v| v.as_str()) {
                lines.push(format!("- **Scoped to task**: {}", task_id));
            }
            if let Some(persona) = v.get("persona").and_then(|v| v.as_str()) {
                lines.push(format!("- **Persona**: {}", persona));
            }
            if let Some(msg) = v.get("initial_message").and_then(|v| v.as_str()) {
                let truncated: String = msg.chars().take(200).collect();
                lines.push(format!("- **Initial objective**: {}", truncated));
            }
            if let Some(parent) = v.get("parent_session_id").and_then(|v| v.as_str()) {
                lines.push(format!("- **Parent session**: {}", parent));
            }
        }

        // Intent-specific behavioral guidance (RFC: Subchat Lifecycle Intelligence)
        match intent_label {
            "job" => {
                lines.push(String::new());
                lines.push("### Lifecycle: Job (one-off task)".to_string());
                lines.push(
                    "You are focused on completing a specific task. When you have finished:"
                        .to_string(),
                );
                lines.push("1. Summarize what you did".to_string());
                lines.push("2. State explicitly that the job is done".to_string());
                lines.push("3. The system will auto-close this conversation".to_string());
            }
            "role" => {
                lines.push(String::new());
                lines.push("### Lifecycle: Role (persistent persona)".to_string());
                lines.push("You embody a persistent role/persona. You are always available for questions in your domain.".to_string());
                lines.push("This conversation never closes automatically — the user controls its lifecycle.".to_string());
                lines.push(
                    "Stay in character. Provide expert guidance within your specialization."
                        .to_string(),
                );
            }
            "scope" => {
                lines.push(String::new());
                lines.push("### Lifecycle: Scope (sub-project)".to_string());
                lines.push(
                    "You manage a defined scope (plan, milestone, or feature area).".to_string(),
                );
                lines.push("Track progress across tasks. This conversation closes when the scope is fully completed.".to_string());
            }
            _ => {
                // "unknown" or legacy — no extra guidance, backward compatible
            }
        }

        // Check for sibling forks (other children of the same parent)
        let parent_id = snapshot_value.as_ref().and_then(|v| {
            v.get("parent_session_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });

        if let Some(ref parent) = parent_id {
            let siblings = self.graph_store.get_fork_children(parent, false).await?;
            let other_siblings: Vec<_> = siblings
                .iter()
                .filter(|s| s.id.to_string() != session_id)
                .collect();

            if !other_siblings.is_empty() {
                lines.push(format!(
                    "\n**Sibling forks**: {} other sub-conversation(s) running in parallel.",
                    other_siblings.len()
                ));
                for sib in other_siblings.iter().take(3) {
                    let status = sib.fork_status.as_deref().unwrap_or("unknown");
                    let title = sib.title.as_deref().unwrap_or("Untitled");
                    lines.push(format!("  - [{}] {}", status, title));
                }
            }
        }

        lines.push("\nCommunicate results via **notes and decisions** in the knowledge graph — the parent session will see them.".to_string());

        Ok(Some(lines.join("\n")))
    }
}

#[async_trait::async_trait]
impl ParallelEnrichmentStage for ForkContextStage {
    async fn execute(&self, input: &EnrichmentInput) -> Result<StageOutput> {
        let mut output = StageOutput::new("fork_context");
        let session_id = input.session_id.to_string();

        // Fetch session metadata to determine if we're a fork or a parent
        let session = self.graph_store.get_chat_session(input.session_id).await?;
        let Some(session) = session else {
            return Ok(output);
        };

        // Case 1: This session IS a fork (fork_depth > 0)
        if session.fork_depth > 0 {
            if let Some(content) = self
                .build_fork_context(
                    &session_id,
                    session.fork_depth,
                    session.fork_type.as_deref().unwrap_or("unknown"),
                    session.fork_intent.as_deref(),
                    session.fork_context_snapshot.as_deref(),
                )
                .await?
            {
                output.add_section(
                    "Fork Context",
                    content,
                    "fork_context",
                    EnrichmentSource::ForkContext,
                );
            }
        }

        // Case 2: This session has fork children (is a parent)
        if let Some(content) = self.build_parent_context(&session_id).await? {
            output.add_section(
                "Sub-conversations",
                content,
                "fork_context",
                EnrichmentSource::ForkContext,
            );
        }

        if !output.sections.is_empty() {
            debug!(
                session_id = %session_id,
                fork_depth = session.fork_depth,
                sections = output.sections.len(),
                "ForkContextStage injected fork awareness"
            );
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        "fork_context"
    }

    fn is_enabled(&self, _config: &EnrichmentConfig) -> bool {
        std::env::var("ENRICHMENT_FORK_CONTEXT")
            .map(|v| v != "false" && v != "0")
            .unwrap_or(true) // Enabled by default
    }
}
