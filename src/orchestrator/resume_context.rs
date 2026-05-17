//! `project.resume_context` orchestration.
//!
//! Single-call context recovery for an agent starting (or resuming) a chat
//! session. Combines three signals an agent would otherwise have to discover
//! by hand at warm-up:
//!
//! 1. **cwd → project_slug** (via [`crate::skills::project_resolver::infer_project_slug_from_cwd`])
//! 2. **branch → resolved task** (via [`crate::orchestrator::branch_parser::resolve_task_from_branch`])
//! 3. **Last session + active plans + recent notes + pending alerts**
//!    (via [`crate::chat::continuity::load_session_context`])
//!
//! The output is structured for direct consumption by an MCP/REST endpoint
//! (`Serialize`) AND carries a `to_markdown()` rendering for prompt injection.
//!
//! All sub-queries run inside `load_session_context` already use `tokio::join!`
//! to stay under the 300ms budget; the branch resolution and project lookup
//! are O(1) thanks to the 5-minute project cache.

use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use tracing::debug;
use uuid::Uuid;

use crate::chat::continuity::{load_session_context, SessionResume};
use crate::neo4j::models::TaskStatus;
use crate::neo4j::traits::GraphStore;
use crate::orchestrator::branch_parser::resolve_task_from_branch;
use crate::skills::project_resolver::infer_project_slug_from_cwd;

// ============================================================================
// Request / Response types
// ============================================================================

/// Input for [`compute_resume_context`].
#[derive(Debug, Clone, Deserialize)]
pub struct ResumeContextRequest {
    /// Working directory the agent is operating in. Mandatory — drives the
    /// project_slug inference. Tilde-expansion is applied downstream.
    pub cwd: String,
    /// Optional active git branch. When provided AND parseable, drives the
    /// branch → task resolution. When `None`, `resolved_task` will be `None`.
    pub branch: Option<String>,
}

/// Lightweight project descriptor surfaced in the response.
#[derive(Debug, Clone, Serialize)]
pub struct ResumeContextProject {
    pub id: Uuid,
    pub slug: String,
    pub name: String,
    pub root_path: String,
}

/// Light task descriptor for `resolved_task`.
#[derive(Debug, Clone, Serialize)]
pub struct ResumeContextTask {
    pub id: Uuid,
    pub title: String,
    pub status: String,
    pub plan_id: Option<Uuid>,
    pub priority: Option<i32>,
}

/// Outcome of branch → task resolution.
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedTaskInfo {
    /// Parsed external_id (e.g. "T210", "T246.7").
    pub external_id: String,
    /// Concrete task if one was found in the project; `None` when the branch
    /// id was parsed but no Task currently carries it.
    pub task: Option<ResumeContextTask>,
}

/// Full response payload.
#[derive(Debug, Clone, Serialize)]
pub struct ResumeContextResponse {
    /// `None` when the cwd doesn't match any registered project root_path.
    /// The remaining fields are still meaningful (resolved_task may be parsed
    /// from the branch even without a matching project).
    pub project: Option<ResumeContextProject>,
    /// Current branch as provided in the request (echo, helps the agent
    /// confirm what was used).
    pub current_branch: Option<String>,
    /// Branch → task resolution outcome.
    pub resolved_task: Option<ResolvedTaskInfo>,
    /// Last session + active plans + recent notes + pending alerts. Populated
    /// only when a project was resolved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_resume: Option<SessionResume>,
    /// Total elapsed time for the orchestration in milliseconds.
    pub elapsed_ms: u64,
}

impl ResumeContextResponse {
    /// True iff the response carries actionable context (project found OR a
    /// task was parsed from the branch).
    pub fn has_content(&self) -> bool {
        self.project.is_some() || self.resolved_task.is_some()
    }
}

// ============================================================================
// Orchestration
// ============================================================================

/// Compute the full resume-context payload.
///
/// This function is safe to call on a fresh agent boot — every sub-query is
/// non-fatal (failures degrade to `None`/empty rather than propagating).
///
/// Resolution order:
/// 1. cwd → project_slug (longest-prefix match against registered roots).
/// 2. branch → external_id + (best-effort) task lookup against the resolved project.
/// 3. Session continuity bundle for the resolved project.
pub async fn compute_resume_context(
    graph: &Arc<dyn GraphStore>,
    req: &ResumeContextRequest,
) -> Result<ResumeContextResponse> {
    let start = Instant::now();

    // --- Step 1: cwd → project ---
    let slug = infer_project_slug_from_cwd(graph.as_ref(), &req.cwd).await?;
    let project = match slug {
        Some(ref s) => graph.get_project_by_slug(s).await?,
        None => None,
    };

    let project_view = project.as_ref().map(|p| ResumeContextProject {
        id: p.id,
        slug: p.slug.clone(),
        name: p.name.clone(),
        root_path: p.root_path.clone(),
    });

    // --- Step 2: branch → resolved task (best-effort) ---
    let resolved_task = match (project.as_ref(), req.branch.as_deref()) {
        (Some(p), Some(branch)) if !branch.is_empty() => {
            resolve_task_from_branch(graph, p.id, branch)
                .await
                .map(|resolved| ResolvedTaskInfo {
                    external_id: resolved.external_id,
                    task: resolved.task.map(task_to_view),
                })
        }
        // Parse-only fallback when no project matched (still useful: surfaces
        // the id the user is conceptually working on).
        (None, Some(branch)) if !branch.is_empty() => {
            crate::orchestrator::branch_parser::parse_branch_to_external_id(branch).map(
                |external_id| ResolvedTaskInfo {
                    external_id,
                    task: None,
                },
            )
        }
        _ => None,
    };

    // --- Step 3: Session continuity bundle ---
    let session_resume = if let Some(ref s) = slug {
        // load_session_context already runs its sub-queries in parallel.
        match load_session_context(graph, s).await {
            Ok(r) => Some(r),
            Err(e) => {
                debug!("[resume_context] load_session_context failed: {}", e);
                None
            }
        }
    } else {
        None
    };

    let elapsed_ms = start.elapsed().as_millis() as u64;
    debug!(
        cwd = %req.cwd,
        slug = ?slug,
        resolved_task = ?resolved_task.as_ref().map(|r| &r.external_id),
        elapsed_ms,
        "[resume_context] computed"
    );

    Ok(ResumeContextResponse {
        project: project_view,
        current_branch: req.branch.clone(),
        resolved_task,
        session_resume,
        elapsed_ms,
    })
}

fn task_to_view(t: crate::neo4j::models::TaskNode) -> ResumeContextTask {
    ResumeContextTask {
        id: t.id,
        title: t
            .title
            .unwrap_or_else(|| t.description.chars().take(80).collect()),
        status: match t.status {
            TaskStatus::Pending => "pending",
            TaskStatus::InProgress => "in_progress",
            TaskStatus::Blocked => "blocked",
            TaskStatus::Completed => "completed",
            TaskStatus::Failed => "failed",
        }
        .to_string(),
        plan_id: None, // Plan reverse-lookup not surfaced here — keep payload minimal.
        priority: t.priority,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_content_is_false_for_empty_response() {
        let r = ResumeContextResponse {
            project: None,
            current_branch: None,
            resolved_task: None,
            session_resume: None,
            elapsed_ms: 0,
        };
        assert!(!r.has_content());
    }

    #[test]
    fn has_content_true_when_project_resolved() {
        let r = ResumeContextResponse {
            project: Some(ResumeContextProject {
                id: Uuid::new_v4(),
                slug: "rustorch".to_string(),
                name: "Rustorch".to_string(),
                root_path: "/Users/x/projects/rustorch".to_string(),
            }),
            current_branch: None,
            resolved_task: None,
            session_resume: None,
            elapsed_ms: 12,
        };
        assert!(r.has_content());
    }

    #[test]
    fn has_content_true_when_task_parsed_without_project() {
        let r = ResumeContextResponse {
            project: None,
            current_branch: Some("feat/t210-mamba2".to_string()),
            resolved_task: Some(ResolvedTaskInfo {
                external_id: "T210".to_string(),
                task: None,
            }),
            session_resume: None,
            elapsed_ms: 3,
        };
        assert!(r.has_content());
    }
}
