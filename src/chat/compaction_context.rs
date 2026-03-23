//! Post-compaction context re-injection module.
//!
//! After Claude Code compacts a conversation, critical project context is lost.
//! This module builds a structured summary (<2000 tokens / ~6000 chars) to
//! re-inject via `inject_hint()` or `send_message()`, restoring:
//! - Current task (title, description, acceptance criteria)
//! - Step progress
//! - Plan overview & progression
//! - Active constraints
//! - Recent decisions
//! - Critical notes
//!
//! Output uses `<system-reminder>` XML tags to distinguish from user content.

use std::fmt::Write as FmtWrite;
use std::sync::Arc;

use anyhow::Result;
use uuid::Uuid;

use crate::chat::types::SessionWorkLogSnapshot;
use crate::neo4j::models::*;
use crate::neo4j::traits::GraphStore;
use crate::notes::{NoteFilters, NoteImportance, NoteStatus, NoteType};
use crate::plan::models::TaskDetails;

/// Maximum output size in characters (~2000 tokens).
const MAX_MARKDOWN_CHARS: usize = 6000;

/// Maximum number of recent decisions to include.
const MAX_DECISIONS: usize = 3;

/// Maximum number of critical notes to include.
const MAX_NOTES: usize = 5;

/// Maximum length for a single note content snippet.
const MAX_NOTE_SNIPPET: usize = 200;

/// Maximum length for custom instructions output.
const MAX_CUSTOM_INSTRUCTIONS_CHARS: usize = 2000;

/// Maximum number of pending/in-progress tasks to include in session context.
const MAX_PENDING_TASKS: usize = 8;

// ============================================================================
// CompactionContext — the assembled context data
// ============================================================================

/// Assembled context data ready for formatting.
#[derive(Debug, Clone, Default)]
pub struct CompactionContext {
    // Task context (runner mode)
    pub task_title: Option<String>,
    pub task_description: Option<String>,
    pub acceptance_criteria: Vec<String>,
    pub affected_files: Vec<String>,
    pub steps: Vec<StepSummary>,

    // Plan context
    pub plan_title: Option<String>,
    pub plan_progress: Option<PlanProgress>,

    // Constraints
    pub constraints: Vec<ConstraintSummary>,

    // Decisions (most recent)
    pub decisions: Vec<DecisionSummary>,

    // Critical notes
    pub notes: Vec<NoteSummary>,

    // Session context (interactive mode)
    pub project_name: Option<String>,
    pub project_description: Option<String>,

    // Active plans for interactive session (plans with status in_progress)
    pub active_plans: Vec<PlanSummary>,

    // Pending/in-progress tasks across active plans
    pub pending_tasks: Vec<TaskSummary>,

    // Work already done this session (from SessionWorkLog snapshot)
    pub work_log: Option<SessionWorkLogSnapshot>,
}

/// Summary of an active plan for post-compaction context.
#[derive(Debug, Clone)]
pub struct PlanSummary {
    pub title: String,
    pub total_tasks: u32,
    pub completed_tasks: u32,
    pub in_progress_tasks: u32,
    pub pending_tasks: u32,
}

/// Summary of a pending/in-progress task for post-compaction context.
#[derive(Debug, Clone)]
pub struct TaskSummary {
    pub title: String,
    pub status: String,
    pub affected_files: Vec<String>,
    pub steps: Vec<StepSummary>,
}

/// Step with status for progress display.
#[derive(Debug, Clone)]
pub struct StepSummary {
    pub order: u32,
    pub description: String,
    pub status: String,
}

/// Plan-level progress.
#[derive(Debug, Clone)]
pub struct PlanProgress {
    pub total_tasks: u32,
    pub completed_tasks: u32,
    pub in_progress_tasks: u32,
}

/// Compact constraint representation.
#[derive(Debug, Clone)]
pub struct ConstraintSummary {
    pub constraint_type: String,
    pub description: String,
}

/// Compact decision representation.
#[derive(Debug, Clone)]
pub struct DecisionSummary {
    pub description: String,
    pub rationale: String,
}

/// Compact note representation.
#[derive(Debug, Clone)]
pub struct NoteSummary {
    pub note_type: String,
    pub importance: String,
    pub content: String,
}

// ============================================================================
// CompactionContextBuilder
// ============================================================================

/// Builds a `CompactionContext` by querying the graph store.
pub struct CompactionContextBuilder {
    graph: Arc<dyn GraphStore>,
}

impl CompactionContextBuilder {
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }

    /// Build context for a runner task (plan_id + task_id).
    ///
    /// Fetches: task details (title, description, acceptance_criteria, steps),
    /// plan (title, progression), constraints, recent decisions (max 3),
    /// critical notes attached to the project (max 5).
    pub async fn build_for_task(&self, plan_id: Uuid, task_id: Uuid) -> Result<CompactionContext> {
        let mut ctx = CompactionContext::default();

        // Fetch task details (includes steps, decisions, affected files)
        let task_details = self.graph.get_task_with_full_details(task_id).await?;

        if let Some(details) = task_details {
            let TaskDetails {
                task,
                steps,
                decisions,
                modifies_files,
                ..
            } = details;

            ctx.task_title = task.title.clone();
            ctx.task_description = Some(truncate(&task.description, 500));
            ctx.acceptance_criteria = task.acceptance_criteria.clone();
            ctx.affected_files = if modifies_files.is_empty() {
                task.affected_files.clone()
            } else {
                modifies_files
            };

            ctx.steps = steps
                .iter()
                .map(|s| StepSummary {
                    order: s.order,
                    description: truncate(&s.description, 120),
                    status: format!("{:?}", s.status).to_lowercase(),
                })
                .collect();

            // Take most recent decisions (up to MAX_DECISIONS)
            ctx.decisions = decisions
                .iter()
                .rev()
                .take(MAX_DECISIONS)
                .map(|d| DecisionSummary {
                    description: truncate(&d.description, 150),
                    rationale: truncate(&d.rationale, 150),
                })
                .collect();
        }

        // Fetch plan info
        if let Ok(Some(plan)) = self.graph.get_plan(plan_id).await {
            ctx.plan_title = Some(plan.title.clone());

            // Compute plan progress from tasks
            if let Ok(tasks) = self.graph.get_plan_tasks(plan_id).await {
                let total = tasks.len() as u32;
                let completed = tasks
                    .iter()
                    .filter(|t| t.status == TaskStatus::Completed)
                    .count() as u32;
                let in_progress = tasks
                    .iter()
                    .filter(|t| t.status == TaskStatus::InProgress)
                    .count() as u32;
                ctx.plan_progress = Some(PlanProgress {
                    total_tasks: total,
                    completed_tasks: completed,
                    in_progress_tasks: in_progress,
                });
            }
        }

        // Fetch constraints
        if let Ok(constraints) = self.graph.get_plan_constraints(plan_id).await {
            ctx.constraints = constraints
                .iter()
                .map(|c| ConstraintSummary {
                    constraint_type: format!("{:?}", c.constraint_type),
                    description: truncate(&c.description, 150),
                })
                .collect();
        }

        // Fetch critical notes for the project (if task is linked to a project)
        if let Ok(Some(project)) = self.graph.get_project_for_task(task_id).await {
            self.fetch_critical_notes(&mut ctx, Some(project.id)).await;
        }

        Ok(ctx)
    }

    /// Build context for an interactive session.
    ///
    /// Fetches: project info, active plans with pending/in-progress tasks,
    /// critical notes (guidelines, gotchas).
    /// Works even without a project (returns minimal context).
    pub async fn build_for_session(&self, project_slug: Option<&str>) -> Result<CompactionContext> {
        let mut ctx = CompactionContext::default();

        if let Some(slug) = project_slug {
            if let Ok(Some(project)) = self.graph.get_project_by_slug(slug).await {
                ctx.project_name = Some(project.name.clone());
                ctx.project_description = project.description.as_ref().map(|d| truncate(d, 300));

                // Fetch active plans (status = in_progress) for this project
                self.fetch_active_plans_and_tasks(&mut ctx, project.id)
                    .await;

                // Fetch critical notes for this project
                self.fetch_critical_notes(&mut ctx, Some(project.id)).await;
            }
        } else {
            // No project — fetch global critical notes only
            self.fetch_critical_notes(&mut ctx, None).await;
        }

        Ok(ctx)
    }

    /// Fetch active plans (in_progress) and their pending/in-progress tasks with steps.
    async fn fetch_active_plans_and_tasks(&self, ctx: &mut CompactionContext, project_id: Uuid) {
        // Get plans filtered by in_progress status
        let plans = match self
            .graph
            .list_plans_for_project(
                project_id,
                Some(vec!["in_progress".to_string()]),
                5, // max 5 active plans
                0,
            )
            .await
        {
            Ok((plans, _)) => plans,
            Err(_) => return,
        };

        for plan in &plans {
            // Get all tasks for this plan
            let tasks = match self.graph.get_plan_tasks(plan.id).await {
                Ok(t) => t,
                Err(_) => continue,
            };

            let total = tasks.len() as u32;
            let completed = tasks
                .iter()
                .filter(|t| t.status == TaskStatus::Completed)
                .count() as u32;
            let in_progress = tasks
                .iter()
                .filter(|t| t.status == TaskStatus::InProgress)
                .count() as u32;
            let pending = tasks
                .iter()
                .filter(|t| t.status == TaskStatus::Pending)
                .count() as u32;

            ctx.active_plans.push(PlanSummary {
                title: plan.title.clone(),
                total_tasks: total,
                completed_tasks: completed,
                in_progress_tasks: in_progress,
                pending_tasks: pending,
            });

            // Collect pending/in-progress tasks with their steps
            let active_tasks: Vec<&TaskNode> = tasks
                .iter()
                .filter(|t| t.status == TaskStatus::Pending || t.status == TaskStatus::InProgress)
                .collect();

            for task in active_tasks.iter().take(MAX_PENDING_TASKS) {
                let task_title = task
                    .title
                    .clone()
                    .unwrap_or_else(|| truncate(&task.description, 80));

                // Fetch steps for this task
                let steps = match self.graph.get_task_steps(task.id).await {
                    Ok(s) => s
                        .iter()
                        .map(|s| StepSummary {
                            order: s.order,
                            description: truncate(&s.description, 100),
                            status: format!("{:?}", s.status).to_lowercase(),
                        })
                        .collect(),
                    Err(_) => vec![],
                };

                ctx.pending_tasks.push(TaskSummary {
                    title: task_title,
                    status: format!("{:?}", task.status).to_lowercase(),
                    affected_files: task.affected_files.clone(),
                    steps,
                });
            }
        }
    }

    /// Fetch critical notes (guidelines + gotchas with importance >= High).
    async fn fetch_critical_notes(&self, ctx: &mut CompactionContext, project_id: Option<Uuid>) {
        let filters = NoteFilters {
            status: Some(vec![NoteStatus::Active]),
            note_type: Some(vec![NoteType::Guideline, NoteType::Gotcha]),
            importance: Some(vec![NoteImportance::High, NoteImportance::Critical]),
            tags: None,
            scope_type: None,
            search: None,
            min_staleness: None,
            max_staleness: None,
            global_only: if project_id.is_none() {
                Some(true)
            } else {
                None
            },
            limit: Some(MAX_NOTES as i64),
            offset: None,
            sort_by: None,
            sort_order: None,
        };

        if let Ok((notes, _)) = self.graph.list_notes(project_id, None, &filters).await {
            ctx.notes = notes
                .iter()
                .take(MAX_NOTES)
                .map(|n| NoteSummary {
                    note_type: format!("{:?}", n.note_type).to_lowercase(),
                    importance: format!("{:?}", n.importance).to_lowercase(),
                    content: truncate(&n.content, MAX_NOTE_SNIPPET),
                })
                .collect();
        }
    }
}

// ============================================================================
// Formatting — to_markdown() and to_custom_instructions()
// ============================================================================

impl CompactionContext {
    /// Format context as compact Markdown wrapped in `<system-reminder>` tags.
    ///
    /// Output is guaranteed to be under `MAX_MARKDOWN_CHARS` (~2000 tokens).
    pub fn to_markdown(&self) -> String {
        let mut out = String::with_capacity(MAX_MARKDOWN_CHARS);

        out.push_str("<system-reminder>\n");
        out.push_str("# Post-Compaction Context (auto-restored)\n\n");

        // § Task Context
        if self.task_title.is_some() || self.task_description.is_some() {
            out.push_str("## Task Context\n");
            if let Some(title) = &self.task_title {
                let _ = writeln!(out, "**Task:** {title}");
            }
            if let Some(desc) = &self.task_description {
                let _ = writeln!(out, "{desc}");
            }
            if !self.acceptance_criteria.is_empty() {
                out.push_str("\n**Acceptance Criteria:**\n");
                for ac in &self.acceptance_criteria {
                    let _ = writeln!(out, "- {ac}");
                }
            }
            out.push('\n');
        }

        // § Project Context (interactive sessions)
        if let Some(name) = &self.project_name {
            out.push_str("## Project\n");
            let _ = writeln!(out, "**Project:** {name}");
            if let Some(desc) = &self.project_description {
                let _ = writeln!(out, "{desc}");
            }
            out.push('\n');
        }

        // § Active Plans (interactive session — plans with in_progress status)
        if !self.active_plans.is_empty() && out.len() < MAX_MARKDOWN_CHARS - 1500 {
            out.push_str("## Active Plans\n");
            for plan in &self.active_plans {
                let _ = writeln!(
                    out,
                    "- **{}** — {}/{} done, {} in progress, {} pending",
                    plan.title,
                    plan.completed_tasks,
                    plan.total_tasks,
                    plan.in_progress_tasks,
                    plan.pending_tasks
                );
                if out.len() > MAX_MARKDOWN_CHARS - 1200 {
                    break;
                }
            }
            out.push('\n');
        }

        // § Pending Objectives (interactive session — tasks pending/in_progress)
        if !self.pending_tasks.is_empty() && out.len() < MAX_MARKDOWN_CHARS - 1000 {
            out.push_str("## Pending Objectives\n");
            for task in &self.pending_tasks {
                let icon = match task.status.as_str() {
                    "inprogress" | "in_progress" => "🔄",
                    _ => "⬜",
                };
                let _ = writeln!(
                    out,
                    "{icon} {} — {}",
                    task.status.to_uppercase(),
                    task.title
                );

                // Show steps if budget allows
                if !task.steps.is_empty() && out.len() < MAX_MARKDOWN_CHARS - 800 {
                    for step in &task.steps {
                        let step_icon = match step.status.as_str() {
                            "completed" => "✅",
                            "in_progress" | "inprogress" => "🔄",
                            "skipped" => "⏭️",
                            _ => "⬜",
                        };
                        let _ = writeln!(out, "  {step_icon} {}. {}", step.order, step.description);
                        if out.len() > MAX_MARKDOWN_CHARS - 600 {
                            out.push_str("  _(truncated)_\n");
                            break;
                        }
                    }
                }

                // Show affected files if budget allows
                if !task.affected_files.is_empty() && out.len() < MAX_MARKDOWN_CHARS - 600 {
                    let files: Vec<&str> = task
                        .affected_files
                        .iter()
                        .take(3)
                        .map(|f| f.as_str())
                        .collect();
                    let _ = writeln!(out, "  Files: {}", files.join(", "));
                }

                if out.len() > MAX_MARKDOWN_CHARS - 500 {
                    out.push_str("_(truncated)_\n");
                    break;
                }
            }
            out.push('\n');
        }

        // § Plan Progress (runner mode — single plan context)
        if let Some(plan_title) = &self.plan_title {
            out.push_str("## Plan\n");
            let _ = write!(out, "**{plan_title}**");
            if let Some(prog) = &self.plan_progress {
                let _ = writeln!(
                    out,
                    " — {}/{} tasks done, {} in progress",
                    prog.completed_tasks, prog.total_tasks, prog.in_progress_tasks
                );
            } else {
                out.push('\n');
            }
            out.push('\n');
        }

        // § Steps Progress
        if !self.steps.is_empty() {
            out.push_str("## Steps Progress\n");
            for step in &self.steps {
                let icon = match step.status.as_str() {
                    "completed" => "✅",
                    "in_progress" | "inprogress" => "🔄",
                    "skipped" => "⏭️",
                    _ => "⬜",
                };
                let _ = writeln!(out, "{icon} {}. {}", step.order, step.description);

                // Budget check — stop early if approaching limit
                if out.len() > MAX_MARKDOWN_CHARS - 1000 {
                    out.push_str("_(truncated)_\n");
                    break;
                }
            }
            out.push('\n');
        }

        // § Affected Files
        if !self.affected_files.is_empty() && out.len() < MAX_MARKDOWN_CHARS - 800 {
            out.push_str("## Affected Files\n");
            for f in &self.affected_files {
                let _ = writeln!(out, "- `{f}`");
                if out.len() > MAX_MARKDOWN_CHARS - 600 {
                    out.push_str("_(truncated)_\n");
                    break;
                }
            }
            out.push('\n');
        }

        // § Work Already Done (from SessionWorkLog — prevents re-doing completed work)
        if let Some(wl) = &self.work_log {
            let has_content = !wl.files_modified.is_empty()
                || !wl.steps_completed.is_empty()
                || wl.tool_use_count > 0;
            if has_content && out.len() < MAX_MARKDOWN_CHARS - 600 {
                out.push_str("## Work Already Done\n");
                if !wl.files_modified.is_empty() {
                    let _ = writeln!(
                        out,
                        "**Files modified ({}):** {}",
                        wl.files_modified.len(),
                        wl.files_modified
                            .iter()
                            .take(10)
                            .map(|f| format!("`{f}`"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
                if !wl.steps_completed.is_empty() {
                    let _ = writeln!(out, "**Steps completed:** {}", wl.steps_completed.len());
                }
                if wl.tool_use_count > 0 {
                    let _ = writeln!(out, "**Tool uses this session:** {}", wl.tool_use_count);
                }
                out.push('\n');
            }
        }

        // § Active Constraints
        if !self.constraints.is_empty() && out.len() < MAX_MARKDOWN_CHARS - 600 {
            out.push_str("## Active Constraints\n");
            for c in &self.constraints {
                let _ = writeln!(out, "- **[{}]** {}", c.constraint_type, c.description);
                if out.len() > MAX_MARKDOWN_CHARS - 400 {
                    break;
                }
            }
            out.push('\n');
        }

        // § Key Decisions
        if !self.decisions.is_empty() && out.len() < MAX_MARKDOWN_CHARS - 400 {
            out.push_str("## Key Decisions\n");
            for d in &self.decisions {
                let _ = writeln!(out, "- {}: {}", d.description, d.rationale);
                if out.len() > MAX_MARKDOWN_CHARS - 200 {
                    break;
                }
            }
            out.push('\n');
        }

        // § Critical Notes
        if !self.notes.is_empty() && out.len() < MAX_MARKDOWN_CHARS - 200 {
            out.push_str("## Critical Notes\n");
            for n in &self.notes {
                let _ = writeln!(
                    out,
                    "- **[{}|{}]** {}",
                    n.note_type, n.importance, n.content
                );
                if out.len() > MAX_MARKDOWN_CHARS - 50 {
                    break;
                }
            }
            out.push('\n');
        }

        out.push_str("</system-reminder>");

        // Final safety truncation
        if out.len() > MAX_MARKDOWN_CHARS {
            out.truncate(MAX_MARKDOWN_CHARS - 20);
            out.push_str("\n</system-reminder>");
        }

        out
    }

    /// Produce enriched custom instructions (~2000 chars) to guide compaction.
    ///
    /// Tells the model what context to preserve during summarization:
    /// step progress, affected files, decisions, and work log (if available).
    pub fn to_custom_instructions(&self, work_log: Option<&SessionWorkLogSnapshot>) -> String {
        let mut out = String::with_capacity(MAX_CUSTOM_INSTRUCTIONS_CHARS);

        // What the agent is working on
        if let Some(title) = &self.task_title {
            let _ = write!(out, "The agent is working on: \"{title}\". ");
        } else if let Some(name) = &self.project_name {
            let _ = write!(out, "The agent is working on project: \"{name}\". ");
        }

        // Detailed step progress (completed/in_progress/pending)
        if !self.steps.is_empty() {
            out.push_str("Steps: ");
            for step in &self.steps {
                let icon = match step.status.as_str() {
                    "completed" => "✅",
                    "in_progress" | "inprogress" => "🔄",
                    "skipped" => "⏭️",
                    _ => "⬜",
                };
                let _ = write!(
                    out,
                    "{icon} {}. {} [{}]; ",
                    step.order,
                    truncate(&step.description, 80),
                    step.status
                );
                if out.len() > MAX_CUSTOM_INSTRUCTIONS_CHARS - 600 {
                    break;
                }
            }
        }

        // Key file paths to preserve (from task + work_log)
        let mut all_files: Vec<&str> = self.affected_files.iter().map(|f| f.as_str()).collect();
        if let Some(wl) = work_log {
            for f in &wl.files_modified {
                if !all_files.contains(&f.as_str()) {
                    all_files.push(f.as_str());
                }
            }
        }
        if !all_files.is_empty() {
            out.push_str("Affected files: ");
            let files: Vec<&str> = all_files.into_iter().take(8).collect();
            out.push_str(&files.join(", "));
            out.push_str(". ");
        }

        // Decision summaries to preserve
        if !self.decisions.is_empty() {
            out.push_str("Key decisions: ");
            let descs: Vec<&str> = self
                .decisions
                .iter()
                .take(3)
                .map(|d| d.description.as_str())
                .collect();
            out.push_str(&descs.join("; "));
            out.push_str(". ");
        }

        // Work log decisions (from session tracking)
        if let Some(wl) = work_log {
            if !wl.decisions_summary.is_empty() {
                out.push_str("Session decisions: ");
                let descs: Vec<&str> = wl
                    .decisions_summary
                    .iter()
                    .take(3)
                    .map(|d| d.as_str())
                    .collect();
                out.push_str(&descs.join("; "));
                out.push_str(". ");
            }
            if !wl.files_modified.is_empty() {
                let _ = write!(
                    out,
                    "Files already modified this session: {}. ",
                    wl.files_modified.join(", ")
                );
            }
        }

        // Pending objectives summary (interactive session)
        if !self.pending_tasks.is_empty() {
            let titles: Vec<&str> = self
                .pending_tasks
                .iter()
                .take(3)
                .map(|t| t.title.as_str())
                .collect();
            let _ = write!(out, "Pending objectives: {}. ", titles.join("; "));
        }

        // Active plans summary (interactive session)
        if !self.active_plans.is_empty() {
            let plan_names: Vec<&str> = self
                .active_plans
                .iter()
                .take(2)
                .map(|p| p.title.as_str())
                .collect();
            let _ = write!(out, "Active plans: {}. ", plan_names.join("; "));
        }

        // CRITICAL preservation instruction
        if self.task_title.is_some() || !self.steps.is_empty() || !self.pending_tasks.is_empty() {
            out.push_str("CRITICAL: In your summary, preserve: (1) which steps are completed vs pending, (2) files already modified, (3) architectural decisions taken. ");
        }

        // Truncate to budget
        if out.len() > MAX_CUSTOM_INSTRUCTIONS_CHARS {
            out.truncate(MAX_CUSTOM_INSTRUCTIONS_CHARS - 3);
            out.push_str("...");
        }

        out
    }
}

impl CompactionContext {
    /// Build a short summary of remaining objectives for auto-continue messages.
    ///
    /// Returns something like: "Remaining: T5 — Ajuster CompletionLoopDetected; T7 — Enrichir le compaction"
    /// or empty string if no pending tasks.
    pub fn pending_objectives_oneliner(&self) -> String {
        if self.pending_tasks.is_empty() {
            return String::new();
        }
        let tasks: Vec<String> = self
            .pending_tasks
            .iter()
            .filter(|t| t.status == "inprogress" || t.status == "pending")
            .take(4)
            .map(|t| truncate(&t.title, 60))
            .collect();
        if tasks.is_empty() {
            return String::new();
        }
        format!(" Remaining objectives: {}", tasks.join("; "))
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Truncate a string to `max_len` characters, adding "…" if truncated.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut result = String::with_capacity(max_len + 3);
        // Find a safe char boundary
        let end = s
            .char_indices()
            .take_while(|(i, _)| *i < max_len)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(max_len);
        result.push_str(&s[..end]);
        result.push('…');
        result
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_context() -> CompactionContext {
        CompactionContext {
            task_title: Some("Implement compaction context builder".to_string()),
            task_description: Some(
                "Create a module that assembles structured context to re-inject after compaction"
                    .to_string(),
            ),
            acceptance_criteria: vec![
                "Context must be under 2000 tokens".to_string(),
                "Must include task, steps, constraints, decisions, notes".to_string(),
            ],
            affected_files: vec![
                "src/chat/compaction_context.rs".to_string(),
                "src/chat/mod.rs".to_string(),
                "src/neo4j/traits.rs".to_string(),
            ],
            steps: vec![
                StepSummary {
                    order: 1,
                    description: "Create CompactionContextBuilder struct".to_string(),
                    status: "completed".to_string(),
                },
                StepSummary {
                    order: 2,
                    description: "Add build_for_session method".to_string(),
                    status: "in_progress".to_string(),
                },
                StepSummary {
                    order: 3,
                    description: "Implement to_markdown formatting".to_string(),
                    status: "pending".to_string(),
                },
                StepSummary {
                    order: 4,
                    description: "Implement to_custom_instructions".to_string(),
                    status: "pending".to_string(),
                },
                StepSummary {
                    order: 5,
                    description: "Add module to mod.rs and write tests".to_string(),
                    status: "pending".to_string(),
                },
            ],
            plan_title: Some("Améliorer la réinjection de contexte post-compaction".to_string()),
            plan_progress: Some(PlanProgress {
                total_tasks: 5,
                completed_tasks: 1,
                in_progress_tasks: 1,
            }),
            constraints: vec![
                ConstraintSummary {
                    constraint_type: "Style".to_string(),
                    description: "Use system-reminder XML tags for re-injected context".to_string(),
                },
                ConstraintSummary {
                    constraint_type: "Performance".to_string(),
                    description: "Build context in < 500ms including Neo4j queries".to_string(),
                },
                ConstraintSummary {
                    constraint_type: "Compatibility".to_string(),
                    description: "Must work with inject_hint() and send_message()".to_string(),
                },
            ],
            decisions: vec![
                DecisionSummary {
                    description: "Use GraphStore trait for testability".to_string(),
                    rationale: "Allows mock-based unit tests without Neo4j".to_string(),
                },
                DecisionSummary {
                    description: "Truncate per-section to stay under budget".to_string(),
                    rationale: "Progressive truncation keeps most important info".to_string(),
                },
            ],
            notes: vec![
                NoteSummary {
                    note_type: "guideline".to_string(),
                    importance: "critical".to_string(),
                    content: "All consumers use Arc<dyn GraphStore> — no concrete Neo4j types"
                        .to_string(),
                },
                NoteSummary {
                    note_type: "gotcha".to_string(),
                    importance: "high".to_string(),
                    content: "NATS listener accumulation on session resume causes duplicates"
                        .to_string(),
                },
            ],
            project_name: None,
            project_description: None,
            active_plans: vec![],
            pending_tasks: vec![],
            work_log: None,
        }
    }

    fn sample_session_context() -> CompactionContext {
        CompactionContext {
            project_name: Some("my-project".to_string()),
            project_description: Some("A cool project".to_string()),
            active_plans: vec![PlanSummary {
                title: "Fix UX issues".to_string(),
                total_tasks: 8,
                completed_tasks: 3,
                in_progress_tasks: 2,
                pending_tasks: 3,
            }],
            pending_tasks: vec![
                TaskSummary {
                    title: "Enrich compaction context".to_string(),
                    status: "inprogress".to_string(),
                    affected_files: vec!["src/chat/compaction_context.rs".to_string()],
                    steps: vec![
                        StepSummary {
                            order: 1,
                            description: "Add fields".to_string(),
                            status: "completed".to_string(),
                        },
                        StepSummary {
                            order: 2,
                            description: "Fetch plans".to_string(),
                            status: "in_progress".to_string(),
                        },
                    ],
                },
                TaskSummary {
                    title: "Fix auto-continue message".to_string(),
                    status: "pending".to_string(),
                    affected_files: vec!["src/chat/manager.rs".to_string()],
                    steps: vec![],
                },
            ],
            notes: vec![NoteSummary {
                note_type: "guideline".to_string(),
                importance: "critical".to_string(),
                content: "Always use traits".to_string(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_to_markdown_under_budget() {
        let ctx = sample_context();
        let md = ctx.to_markdown();

        assert!(
            md.len() <= MAX_MARKDOWN_CHARS,
            "Markdown output {} chars exceeds budget {}",
            md.len(),
            MAX_MARKDOWN_CHARS
        );
        assert!(md.starts_with("<system-reminder>"));
        assert!(md.ends_with("</system-reminder>"));
    }

    #[test]
    fn test_to_markdown_contains_sections() {
        let ctx = sample_context();
        let md = ctx.to_markdown();

        assert!(
            md.contains("## Task Context"),
            "Missing Task Context section"
        );
        assert!(
            md.contains("## Steps Progress"),
            "Missing Steps Progress section"
        );
        assert!(
            md.contains("## Active Constraints"),
            "Missing Constraints section"
        );
        assert!(md.contains("## Key Decisions"), "Missing Decisions section");
        assert!(md.contains("## Critical Notes"), "Missing Notes section");
        assert!(md.contains("## Plan"), "Missing Plan section");
    }

    #[test]
    fn test_to_markdown_step_icons() {
        let ctx = sample_context();
        let md = ctx.to_markdown();

        assert!(md.contains("✅"), "Missing completed icon");
        assert!(md.contains("🔄"), "Missing in_progress icon");
        assert!(md.contains("⬜"), "Missing pending icon");
    }

    #[test]
    fn test_to_markdown_plan_progress() {
        let ctx = sample_context();
        let md = ctx.to_markdown();

        assert!(
            md.contains("1/5 tasks done"),
            "Missing plan progress: {}",
            md
        );
    }

    #[test]
    fn test_to_custom_instructions_contains_files() {
        let ctx = sample_context();
        let ci = ctx.to_custom_instructions(None);

        assert!(
            ci.contains("src/chat/compaction_context.rs"),
            "Missing file path in custom instructions"
        );
        assert!(
            ci.contains("src/chat/mod.rs"),
            "Missing file path in custom instructions"
        );
    }

    #[test]
    fn test_to_custom_instructions_contains_task() {
        let ctx = sample_context();
        let ci = ctx.to_custom_instructions(None);

        assert!(
            ci.contains("Implement compaction context builder"),
            "Missing task title in custom instructions"
        );
    }

    #[test]
    fn test_to_custom_instructions_under_budget() {
        let ctx = sample_context();
        let ci = ctx.to_custom_instructions(None);

        assert!(
            ci.len() <= MAX_CUSTOM_INSTRUCTIONS_CHARS,
            "Custom instructions {} chars exceeds budget {}",
            ci.len(),
            MAX_CUSTOM_INSTRUCTIONS_CHARS
        );
    }

    #[test]
    fn test_to_custom_instructions_contains_decisions() {
        let ctx = sample_context();
        let ci = ctx.to_custom_instructions(None);

        assert!(
            ci.contains("GraphStore"),
            "Missing decision reference in custom instructions"
        );
    }

    #[test]
    fn test_to_custom_instructions_step_progress() {
        let ctx = sample_context();
        let ci = ctx.to_custom_instructions(None);

        // Enriched: shows individual step statuses
        assert!(
            ci.contains("[completed]"),
            "Missing completed step status: {}",
            ci
        );
        assert!(
            ci.contains("[in_progress]"),
            "Missing in_progress step status: {}",
            ci
        );
        assert!(
            ci.contains("[pending]"),
            "Missing pending step status: {}",
            ci
        );
    }

    #[test]
    fn test_empty_context_produces_valid_markdown() {
        let ctx = CompactionContext::default();
        let md = ctx.to_markdown();

        assert!(md.starts_with("<system-reminder>"));
        assert!(md.ends_with("</system-reminder>"));
        assert!(md.len() < 200, "Empty context should be minimal");
    }

    #[test]
    fn test_empty_context_produces_empty_instructions() {
        let ctx = CompactionContext::default();
        let ci = ctx.to_custom_instructions(None);

        assert!(ci.is_empty() || ci.len() < 10);
    }

    #[test]
    fn test_session_context_with_project() {
        let ctx = CompactionContext {
            project_name: Some("my-project".to_string()),
            project_description: Some("A cool project".to_string()),
            notes: vec![NoteSummary {
                note_type: "guideline".to_string(),
                importance: "critical".to_string(),
                content: "Always use traits".to_string(),
            }],
            ..Default::default()
        };
        let md = ctx.to_markdown();

        assert!(md.contains("## Project"));
        assert!(md.contains("my-project"));
        assert!(md.contains("## Critical Notes"));
    }

    #[test]
    fn test_truncate_helper() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello…");
        assert_eq!(truncate("", 5), "");
    }

    #[test]
    fn test_truncate_unicode_safety() {
        // Ensure truncation doesn't split multi-byte chars
        let s = "héllo wörld";
        let t = truncate(s, 5);
        // Should not panic and should be valid UTF-8
        assert!(t.len() <= 10); // 5 chars + "…" (3 bytes)
    }

    #[test]
    fn test_large_context_stays_under_budget() {
        // Create a context with many steps and long descriptions
        let mut ctx = sample_context();
        ctx.steps = (1..=50)
            .map(|i| StepSummary {
                order: i,
                description: format!(
                    "Step {i}: This is a very long step description that goes on and on to test truncation behavior in the markdown formatter"
                ),
                status: if i <= 10 { "completed" } else { "pending" }.to_string(),
            })
            .collect();
        ctx.affected_files = (0..20)
            .map(|i| format!("src/very/deep/nested/path/to/file_{i}.rs"))
            .collect();
        ctx.notes = (0..10)
            .map(|i| NoteSummary {
                note_type: "guideline".to_string(),
                importance: "critical".to_string(),
                content: format!(
                    "Critical note {i}: This is a very important note with lots of detail that should be truncated"
                ),
            })
            .collect();

        let md = ctx.to_markdown();
        assert!(
            md.len() <= MAX_MARKDOWN_CHARS,
            "Large context {} chars exceeds budget {}",
            md.len(),
            MAX_MARKDOWN_CHARS
        );
        assert!(md.ends_with("</system-reminder>"));
    }

    #[test]
    fn test_session_context_with_active_plans() {
        let ctx = sample_session_context();
        let md = ctx.to_markdown();

        assert!(
            md.contains("## Active Plans"),
            "Missing Active Plans section"
        );
        assert!(md.contains("Fix UX issues"), "Missing plan title");
        assert!(md.contains("3/8 done"), "Missing plan progress");
    }

    #[test]
    fn test_session_context_with_pending_objectives() {
        let ctx = sample_session_context();
        let md = ctx.to_markdown();

        assert!(
            md.contains("## Pending Objectives"),
            "Missing Pending Objectives section: {}",
            md
        );
        assert!(
            md.contains("Enrich compaction context"),
            "Missing task title"
        );
        assert!(
            md.contains("Fix auto-continue message"),
            "Missing pending task"
        );
        // Check step icons within objectives
        assert!(md.contains("✅"), "Missing completed step icon");
        assert!(md.contains("🔄"), "Missing in_progress icon");
    }

    #[test]
    fn test_session_context_under_budget() {
        let ctx = sample_session_context();
        let md = ctx.to_markdown();

        assert!(
            md.len() <= MAX_MARKDOWN_CHARS,
            "Session context {} chars exceeds budget {}",
            md.len(),
            MAX_MARKDOWN_CHARS
        );
    }

    #[test]
    fn test_pending_objectives_oneliner() {
        let ctx = sample_session_context();
        let oneliner = ctx.pending_objectives_oneliner();

        assert!(
            oneliner.contains("Enrich compaction context"),
            "Missing task in oneliner: {}",
            oneliner
        );
        assert!(
            oneliner.contains("Fix auto-continue message"),
            "Missing pending task in oneliner: {}",
            oneliner
        );
        assert!(
            oneliner.starts_with(" Remaining objectives:"),
            "Wrong prefix: {}",
            oneliner
        );
    }

    #[test]
    fn test_pending_objectives_oneliner_empty() {
        let ctx = CompactionContext::default();
        assert!(ctx.pending_objectives_oneliner().is_empty());
    }

    #[test]
    fn test_custom_instructions_includes_pending_tasks() {
        let ctx = sample_session_context();
        let ci = ctx.to_custom_instructions(None);

        assert!(
            ci.contains("Pending objectives:"),
            "Missing pending objectives in custom instructions: {}",
            ci
        );
        assert!(
            ci.contains("Active plans:"),
            "Missing active plans in custom instructions: {}",
            ci
        );
    }

    #[test]
    fn test_large_session_context_stays_under_budget() {
        let mut ctx = sample_session_context();
        // Add many pending tasks
        ctx.pending_tasks = (0..20)
            .map(|i| TaskSummary {
                title: format!(
                    "Task {i}: Very long task title that tests truncation behavior in the session context"
                ),
                status: if i % 2 == 0 { "inprogress" } else { "pending" }.to_string(),
                affected_files: vec![format!("src/module_{i}/handler.rs")],
                steps: (1..=5)
                    .map(|j| StepSummary {
                        order: j,
                        description: format!("Step {j} of task {i}: do something important"),
                        status: if j <= 2 { "completed" } else { "pending" }.to_string(),
                    })
                    .collect(),
            })
            .collect();
        ctx.active_plans = (0..5)
            .map(|i| PlanSummary {
                title: format!("Plan {i}: A very important plan with a long title"),
                total_tasks: 10,
                completed_tasks: 3,
                in_progress_tasks: 2,
                pending_tasks: 5,
            })
            .collect();

        let md = ctx.to_markdown();
        assert!(
            md.len() <= MAX_MARKDOWN_CHARS,
            "Large session context {} chars exceeds budget {}",
            md.len(),
            MAX_MARKDOWN_CHARS
        );
        assert!(md.ends_with("</system-reminder>"));
    }

    #[test]
    fn test_custom_instructions_enriched_with_steps_and_work_log() {
        let ctx = sample_context();
        let work_log = SessionWorkLogSnapshot {
            files_modified: vec![
                "src/chat/compaction_context.rs".to_string(),
                "src/chat/manager.rs".to_string(),
            ],
            files_read: vec!["src/chat/types.rs".to_string()],
            steps_completed: vec![Uuid::new_v4()],
            step_in_progress: Some(Uuid::new_v4()),
            decisions_summary: vec!["Use enriched custom_instructions for compaction".to_string()],
            last_tool_name: Some("Edit".to_string()),
            tool_use_count: 12,
        };

        let ci = ctx.to_custom_instructions(Some(&work_log));

        // Must be >500 chars (enriched beyond old limit)
        assert!(
            ci.len() > 500,
            "Enriched custom instructions should be >500 chars, got {}: {}",
            ci.len(),
            ci
        );

        // Must contain "preserve" (CRITICAL instruction)
        assert!(
            ci.contains("preserve"),
            "Missing CRITICAL preserve instruction: {}",
            ci
        );

        // Must contain file names from work_log
        assert!(
            ci.contains("src/chat/manager.rs"),
            "Missing work_log modified file: {}",
            ci
        );

        // Must contain session decisions
        assert!(
            ci.contains("enriched custom_instructions"),
            "Missing session decision: {}",
            ci
        );

        // Must contain step statuses
        assert!(
            ci.contains("[completed]"),
            "Missing completed step status: {}",
            ci
        );
        assert!(
            ci.contains("[in_progress]"),
            "Missing in_progress step status: {}",
            ci
        );
    }

    #[test]
    fn test_custom_instructions_enriched_under_budget() {
        let ctx = sample_context();
        let work_log = SessionWorkLogSnapshot {
            files_modified: (0..20)
                .map(|i| format!("src/module_{i}/handler.rs"))
                .collect(),
            files_read: vec![],
            steps_completed: vec![],
            step_in_progress: None,
            decisions_summary: (0..10)
                .map(|i| format!("Decision {i}: Use pattern X for feature Y"))
                .collect(),
            last_tool_name: None,
            tool_use_count: 100,
        };

        let ci = ctx.to_custom_instructions(Some(&work_log));

        assert!(
            ci.len() <= MAX_CUSTOM_INSTRUCTIONS_CHARS,
            "Enriched custom instructions {} chars exceeds budget {}",
            ci.len(),
            MAX_CUSTOM_INSTRUCTIONS_CHARS
        );
    }

    #[test]
    fn test_post_compaction_work_log_in_markdown() {
        let mut ctx = sample_context();
        let step1 = Uuid::new_v4();
        ctx.work_log = Some(SessionWorkLogSnapshot {
            files_modified: vec!["src/main.rs".to_string(), "src/lib.rs".to_string()],
            files_read: vec!["src/utils.rs".to_string()],
            steps_completed: vec![step1],
            step_in_progress: None,
            decisions_summary: vec![],
            last_tool_name: None,
            tool_use_count: 15,
        });

        let md = ctx.to_markdown();
        assert!(
            md.contains("## Work Already Done"),
            "Missing Work Already Done section in: {}",
            md
        );
        assert!(
            md.contains("`src/main.rs`"),
            "Missing files_modified in: {}",
            md
        );
        assert!(
            md.contains("`src/lib.rs`"),
            "Missing second file in: {}",
            md
        );
        assert!(
            md.contains("Files modified (2)"),
            "Missing file count in: {}",
            md
        );
        assert!(
            md.contains("**Steps completed:** 1"),
            "Missing steps completed count in: {}",
            md
        );
        assert!(
            md.contains("**Tool uses this session:** 15"),
            "Missing tool_use_count in: {}",
            md
        );
    }

    #[test]
    fn test_post_compaction_work_log_none_no_section() {
        let ctx = sample_context();
        // Default has work_log = None
        let md = ctx.to_markdown();
        assert!(
            !md.contains("## Work Already Done"),
            "Should not have Work Already Done when work_log is None"
        );
    }

    #[test]
    fn test_custom_instructions_without_work_log_still_has_preserve() {
        let ctx = sample_context();
        let ci = ctx.to_custom_instructions(None);

        // Even without work_log, should have CRITICAL instruction when there are steps
        assert!(
            ci.contains("CRITICAL"),
            "Missing CRITICAL instruction even without work_log: {}",
            ci
        );
    }

    #[test]
    fn test_custom_instructions_work_log_files_already_modified() {
        let ctx = sample_context();
        let wl = SessionWorkLogSnapshot {
            files_modified: vec!["src/new_file.rs".to_string(), "src/other.rs".to_string()],
            files_read: vec![],
            steps_completed: vec![],
            step_in_progress: None,
            decisions_summary: vec![],
            last_tool_name: None,
            tool_use_count: 0,
        };
        let ci = ctx.to_custom_instructions(Some(&wl));

        assert!(
            ci.contains("Files already modified this session"),
            "Missing files already modified line: {}",
            ci
        );
        assert!(
            ci.contains("src/new_file.rs"),
            "Missing work_log file: {}",
            ci
        );
    }

    #[test]
    fn test_custom_instructions_work_log_session_decisions() {
        let ctx = sample_context();
        let wl = SessionWorkLogSnapshot {
            files_modified: vec![],
            files_read: vec![],
            steps_completed: vec![],
            step_in_progress: None,
            decisions_summary: vec!["bash: cargo test".to_string(), "bash: cargo check".to_string()],
            last_tool_name: None,
            tool_use_count: 0,
        };
        let ci = ctx.to_custom_instructions(Some(&wl));

        assert!(
            ci.contains("Session decisions"),
            "Missing session decisions: {}",
            ci
        );
        assert!(
            ci.contains("cargo test"),
            "Missing decision content: {}",
            ci
        );
    }

    #[test]
    fn test_custom_instructions_active_plans_included() {
        let ctx = CompactionContext {
            project_name: Some("test-project".to_string()),
            active_plans: vec![
                PlanSummary {
                    title: "Plan A".to_string(),
                    total_tasks: 5,
                    completed_tasks: 2,
                    in_progress_tasks: 1,
                    pending_tasks: 2,
                },
            ],
            pending_tasks: vec![
                TaskSummary {
                    title: "Do something".to_string(),
                    status: "inprogress".to_string(),
                    affected_files: vec![],
                    steps: vec![],
                },
            ],
            ..Default::default()
        };
        let ci = ctx.to_custom_instructions(None);

        assert!(
            ci.contains("Active plans: Plan A"),
            "Missing active plan: {}",
            ci
        );
        assert!(
            ci.contains("Pending objectives: Do something"),
            "Missing pending objective: {}",
            ci
        );
    }

    #[test]
    fn test_custom_instructions_truncation_at_budget() {
        // Build a context with many steps to exceed 2000 chars
        let ctx = CompactionContext {
            task_title: Some("Very long task name".to_string()),
            steps: (0..50)
                .map(|i| StepSummary {
                    order: i,
                    description: format!("Step {} with a fairly long description to fill up chars", i),
                    status: "pending".to_string(),
                })
                .collect(),
            affected_files: (0..20)
                .map(|i| format!("src/module_{}/handler_{}.rs", i, i))
                .collect(),
            ..Default::default()
        };
        let ci = ctx.to_custom_instructions(None);

        assert!(
            ci.len() <= MAX_CUSTOM_INSTRUCTIONS_CHARS,
            "Custom instructions {} chars exceeds budget {}",
            ci.len(),
            MAX_CUSTOM_INSTRUCTIONS_CHARS
        );
        // When content exceeds budget, it gets truncated
        // The budget check is the important assertion above
    }

    #[test]
    fn test_to_markdown_work_log_under_budget() {
        let mut ctx = sample_context();
        ctx.work_log = Some(SessionWorkLogSnapshot {
            files_modified: vec!["src/a.rs".to_string(), "src/b.rs".to_string()],
            files_read: vec!["src/c.rs".to_string()],
            steps_completed: vec![uuid::Uuid::new_v4()],
            step_in_progress: None,
            decisions_summary: vec!["bash: cargo check".to_string()],
            last_tool_name: Some("Edit".to_string()),
            tool_use_count: 10,
        });
        let md = ctx.to_markdown();

        assert!(
            md.len() <= MAX_MARKDOWN_CHARS,
            "Markdown {} exceeds budget {}",
            md.len(),
            MAX_MARKDOWN_CHARS
        );
        assert!(
            md.contains("## Work Already Done"),
            "Missing Work Already Done section: {}",
            md
        );
        assert!(
            md.contains("`src/a.rs`"),
            "Missing work_log file: {}",
            md
        );
    }

    #[test]
    fn test_custom_instructions_work_log_merges_affected_files() {
        let ctx = CompactionContext {
            task_title: Some("Test task".to_string()),
            affected_files: vec!["src/existing.rs".to_string()],
            ..Default::default()
        };
        let wl = SessionWorkLogSnapshot {
            files_modified: vec!["src/new.rs".to_string()],
            files_read: vec![],
            steps_completed: vec![],
            step_in_progress: None,
            decisions_summary: vec![],
            last_tool_name: None,
            tool_use_count: 0,
        };
        let ci = ctx.to_custom_instructions(Some(&wl));

        // Both files should appear in "Affected files"
        assert!(
            ci.contains("src/existing.rs"),
            "Missing task affected file: {}",
            ci
        );
        assert!(
            ci.contains("src/new.rs") || ci.contains("Files already modified"),
            "Missing work_log file in output: {}",
            ci
        );
    }
}
