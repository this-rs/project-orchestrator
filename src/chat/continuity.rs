//! Session Continuity Engine — automatic context recovery across chat sessions.
//!
//! At the start of a new chat session, loads relevant context from the previous session:
//! - Last session's DISCUSSED entities
//! - Active tasks/plans in progress
//! - Recent notes and decisions created
//!
//! Produces a `SessionResume` that is formatted as a concise markdown section
//! and injected into the system prompt or enrichment context.
//!
//! **Performance target:** < 300ms total for all queries (parallel execution).

use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::neo4j::models::{DiscussedEntity, TaskNode};
use crate::neo4j::traits::GraphStore;
use crate::notes::models::{Note, NoteFilters, NoteStatus};

// ============================================================================
// Types
// ============================================================================

/// Context recovered from the previous session on the same project.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LastSessionContext {
    /// Session ID of the previous session
    pub session_id: Option<Uuid>,
    /// When the previous session was last active
    pub session_timestamp: Option<DateTime<Utc>>,
    /// Entities discussed in the previous session
    pub discussed_entities: Vec<DiscussedEntity>,
    /// Notes created during/around the previous session
    pub recent_notes: Vec<NoteSummary>,
}

/// Summary of an active plan with its task breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivePlanSummary {
    pub plan_id: Uuid,
    pub title: String,
    pub status: String,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub in_progress_tasks: Vec<TaskSummary>,
    pub pending_tasks: Vec<TaskSummary>,
}

/// Lightweight task summary for the session resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSummary {
    pub task_id: Uuid,
    pub title: String,
    pub status: String,
    pub priority: Option<i32>,
}

/// Lightweight note summary for the session resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteSummary {
    pub note_id: Uuid,
    pub note_type: String,
    pub importance: String,
    pub content_preview: String,
}

/// Current active work context for the project.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActiveWorkContext {
    /// Plans currently in progress
    pub active_plans: Vec<ActivePlanSummary>,
    /// Notes recently modified (last 48h)
    pub recent_notes: Vec<NoteSummary>,
}

/// Combined session resume — everything an agent needs to pick up where it left off.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionResume {
    /// Context from the previous session
    pub last_session: LastSessionContext,
    /// Current active work
    pub active_work: ActiveWorkContext,
    /// How long the context loading took (ms)
    pub load_time_ms: u64,
}

// ============================================================================
// Session Continuity Engine
// ============================================================================

/// Loads session continuity context for a new chat session.
///
/// Runs all queries in parallel to stay under the 300ms budget.
pub async fn load_session_context(
    graph: &Arc<dyn GraphStore>,
    project_slug: &str,
) -> Result<SessionResume> {
    let start = Instant::now();

    // Resolve project_id from slug
    let project = graph.get_project_by_slug(project_slug).await?;
    let Some(project) = project else {
        debug!(
            "[continuity] Project '{}' not found — returning empty resume",
            project_slug
        );
        return Ok(SessionResume::default());
    };
    let project_id = project.id;

    // Run all queries in parallel
    let (last_session_result, active_plans_result, recent_notes_result) = tokio::join!(
        load_last_session_context(graph, project_slug, project_id),
        load_active_plans(graph, project_id),
        load_recent_notes(graph, project_id),
    );

    let last_session = last_session_result.unwrap_or_else(|e| {
        warn!("[continuity] Failed to load last session: {}", e);
        LastSessionContext::default()
    });

    let active_plans = active_plans_result.unwrap_or_else(|e| {
        warn!("[continuity] Failed to load active plans: {}", e);
        vec![]
    });

    let recent_notes = recent_notes_result.unwrap_or_else(|e| {
        warn!("[continuity] Failed to load recent notes: {}", e);
        vec![]
    });

    let load_time_ms = start.elapsed().as_millis() as u64;
    debug!(
        "[continuity] Context loaded in {}ms — {} discussed entities, {} active plans, {} recent notes",
        load_time_ms,
        last_session.discussed_entities.len(),
        active_plans.len(),
        recent_notes.len(),
    );

    Ok(SessionResume {
        last_session,
        active_work: ActiveWorkContext {
            active_plans,
            recent_notes,
        },
        load_time_ms,
    })
}

// ============================================================================
// Query: Last Session Context
// ============================================================================

async fn load_last_session_context(
    graph: &Arc<dyn GraphStore>,
    project_slug: &str,
    project_id: Uuid,
) -> Result<LastSessionContext> {
    // Get the most recent session for this project (limit=1, offset=0)
    let (sessions, _) = graph
        .list_chat_sessions(Some(project_slug), None, 1, 0, false)
        .await?;

    let Some(last_session) = sessions.into_iter().next() else {
        return Ok(LastSessionContext::default());
    };

    // Get discussed entities from that session
    let discussed = graph
        .get_session_entities(last_session.id, Some(project_id))
        .await
        .unwrap_or_default();

    Ok(LastSessionContext {
        session_id: Some(last_session.id),
        session_timestamp: Some(last_session.updated_at),
        discussed_entities: discussed,
        recent_notes: vec![], // Filled by load_recent_notes separately
    })
}

// ============================================================================
// Query: Active Plans & Tasks
// ============================================================================

async fn load_active_plans(
    graph: &Arc<dyn GraphStore>,
    project_id: Uuid,
) -> Result<Vec<ActivePlanSummary>> {
    // Get plans that are in_progress or approved
    let status_filter = Some(vec!["in_progress".to_string(), "approved".to_string()]);
    let (plans, _) = graph
        .list_plans_for_project(project_id, status_filter, 5, 0)
        .await?;

    let mut summaries = Vec::with_capacity(plans.len());

    for plan in plans {
        let tasks = graph.get_plan_tasks(plan.id).await.unwrap_or_default();

        let total = tasks.len();
        let completed = tasks
            .iter()
            .filter(|t| t.status == crate::neo4j::models::TaskStatus::Completed)
            .count();

        let in_progress_tasks: Vec<TaskSummary> = tasks
            .iter()
            .filter(|t| t.status == crate::neo4j::models::TaskStatus::InProgress)
            .map(task_to_summary)
            .collect();

        let pending_tasks: Vec<TaskSummary> = tasks
            .iter()
            .filter(|t| t.status == crate::neo4j::models::TaskStatus::Pending)
            .take(3) // Only top 3 pending to keep it concise
            .map(task_to_summary)
            .collect();

        summaries.push(ActivePlanSummary {
            plan_id: plan.id,
            title: if plan.title.is_empty() {
                "Untitled Plan".to_string()
            } else {
                plan.title.clone()
            },
            status: format!("{:?}", plan.status),
            total_tasks: total,
            completed_tasks: completed,
            in_progress_tasks,
            pending_tasks,
        });
    }

    Ok(summaries)
}

fn task_to_summary(task: &TaskNode) -> TaskSummary {
    TaskSummary {
        task_id: task.id,
        title: task
            .title
            .clone()
            .unwrap_or_else(|| task.description.chars().take(60).collect()),
        status: format!("{:?}", task.status),
        priority: task.priority,
    }
}

// ============================================================================
// Query: Recent Notes
// ============================================================================

async fn load_recent_notes(
    graph: &Arc<dyn GraphStore>,
    project_id: Uuid,
) -> Result<Vec<NoteSummary>> {
    let filters = NoteFilters {
        status: Some(vec![NoteStatus::Active]),
        limit: Some(10),
        ..Default::default()
    };

    let (notes, _) = graph.list_notes(Some(project_id), None, &filters).await?;

    // Take only notes modified in the last 48h
    let cutoff = Utc::now() - chrono::Duration::hours(48);
    let recent: Vec<NoteSummary> = notes
        .into_iter()
        .filter(|n| n.created_at > cutoff)
        .take(5)
        .map(|n| note_to_summary(&n))
        .collect();

    Ok(recent)
}

fn note_to_summary(note: &Note) -> NoteSummary {
    let preview = if note.content.len() > 120 {
        // Find a valid UTF-8 char boundary at or before 120 bytes
        let mut end = 120;
        while end > 0 && !note.content.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &note.content[..end])
    } else {
        note.content.clone()
    };

    NoteSummary {
        note_id: note.id,
        note_type: format!("{:?}", note.note_type),
        importance: format!("{:?}", note.importance),
        content_preview: preview,
    }
}

// ============================================================================
// Rendering — Markdown formatting for system prompt injection
// ============================================================================

impl SessionResume {
    /// Returns true if there is any meaningful context to inject.
    pub fn has_content(&self) -> bool {
        !self.last_session.discussed_entities.is_empty()
            || !self.active_work.active_plans.is_empty()
            || !self.active_work.recent_notes.is_empty()
    }

    /// Render the session resume as a concise markdown section (< 1000 tokens).
    pub fn to_markdown(&self) -> String {
        if !self.has_content() {
            return String::new();
        }

        let mut md = String::with_capacity(2048);
        md.push_str("## Session Context\n\n");

        // === Last Session ===
        if !self.last_session.discussed_entities.is_empty() {
            md.push_str("### Previous Session\n");
            if let Some(ts) = self.last_session.session_timestamp {
                let ago = format_time_ago(ts);
                md.push_str(&format!("Last active: {}\n\n", ago));
            }

            md.push_str("**Files discussed:**\n");
            let files: Vec<&DiscussedEntity> = self
                .last_session
                .discussed_entities
                .iter()
                .filter(|e| e.entity_type.to_lowercase() == "file")
                .take(8)
                .collect();
            for f in &files {
                let path = f.file_path.as_deref().unwrap_or(&f.entity_id);
                md.push_str(&format!("- `{}`", path));
                if f.mention_count > 1 {
                    md.push_str(&format!(" (×{})", f.mention_count));
                }
                md.push('\n');
            }

            let non_files: Vec<&DiscussedEntity> = self
                .last_session
                .discussed_entities
                .iter()
                .filter(|e| e.entity_type.to_lowercase() != "file")
                .take(5)
                .collect();
            if !non_files.is_empty() {
                md.push_str("\n**Symbols discussed:**\n");
                for e in &non_files {
                    md.push_str(&format!("- `{}` ({})\n", e.entity_id, e.entity_type));
                }
            }
            md.push('\n');
        }

        // === Active Work ===
        if !self.active_work.active_plans.is_empty() {
            md.push_str("### Active Work\n");
            for plan in &self.active_work.active_plans {
                let pct = if plan.total_tasks > 0 {
                    (plan.completed_tasks as f64 / plan.total_tasks as f64 * 100.0) as u32
                } else {
                    0
                };
                md.push_str(&format!(
                    "**{}** — {}/{} tasks ({}%)\n",
                    plan.title, plan.completed_tasks, plan.total_tasks, pct,
                ));

                for t in &plan.in_progress_tasks {
                    md.push_str(&format!("  - 🔄 {} (in_progress)\n", t.title));
                }
                for t in plan.pending_tasks.iter().take(2) {
                    md.push_str(&format!("  - ⏳ {} (pending)\n", t.title));
                }
            }
            md.push('\n');
        }

        // === Recent Notes ===
        if !self.active_work.recent_notes.is_empty() {
            md.push_str("### Recent Knowledge\n");
            for note in &self.active_work.recent_notes {
                md.push_str(&format!(
                    "- **[{}]** ({}): {}\n",
                    note.note_type, note.importance, note.content_preview,
                ));
            }
            md.push('\n');
        }

        md
    }
}

/// Format a datetime as a human-readable "X ago" string.
fn format_time_ago(dt: DateTime<Utc>) -> String {
    let now = Utc::now();
    let diff = now.signed_duration_since(dt);

    if diff.num_days() > 0 {
        format!("{}d ago", diff.num_days())
    } else if diff.num_hours() > 0 {
        format!("{}h ago", diff.num_hours())
    } else if diff.num_minutes() > 0 {
        format!("{}min ago", diff.num_minutes())
    } else {
        "just now".to_string()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_resume_has_no_content() {
        let resume = SessionResume::default();
        assert!(!resume.has_content());
        assert!(resume.to_markdown().is_empty());
    }

    #[test]
    fn test_resume_with_discussed_entities_has_content() {
        let resume = SessionResume {
            last_session: LastSessionContext {
                session_id: Some(Uuid::new_v4()),
                session_timestamp: Some(Utc::now() - chrono::Duration::hours(2)),
                discussed_entities: vec![DiscussedEntity {
                    entity_type: "File".to_string(),
                    entity_id: "src/chat/manager.rs".to_string(),
                    mention_count: 3,
                    last_mentioned_at: Some(Utc::now().to_rfc3339()),
                    file_path: Some("src/chat/manager.rs".to_string()),
                }],
                recent_notes: vec![],
            },
            ..Default::default()
        };
        assert!(resume.has_content());
        let md = resume.to_markdown();
        assert!(md.contains("src/chat/manager.rs"));
        assert!(md.contains("×3"));
        assert!(md.contains("2h ago"));
    }

    #[test]
    fn test_resume_with_active_plans_renders_progress() {
        let resume = SessionResume {
            active_work: ActiveWorkContext {
                active_plans: vec![ActivePlanSummary {
                    plan_id: Uuid::new_v4(),
                    title: "TP4 — Session Continuity".to_string(),
                    status: "InProgress".to_string(),
                    total_tasks: 5,
                    completed_tasks: 2,
                    in_progress_tasks: vec![TaskSummary {
                        task_id: Uuid::new_v4(),
                        title: "TP4.1 — Build continuity engine".to_string(),
                        status: "InProgress".to_string(),
                        priority: Some(80),
                    }],
                    pending_tasks: vec![],
                }],
                recent_notes: vec![],
            },
            ..Default::default()
        };
        let md = resume.to_markdown();
        assert!(md.contains("TP4 — Session Continuity"));
        assert!(md.contains("2/5 tasks (40%)"));
        assert!(md.contains("TP4.1"));
    }

    #[test]
    fn test_format_time_ago() {
        let now = Utc::now();
        assert_eq!(format_time_ago(now), "just now");
        assert_eq!(
            format_time_ago(now - chrono::Duration::minutes(15)),
            "15min ago"
        );
        assert_eq!(format_time_ago(now - chrono::Duration::hours(3)), "3h ago");
        assert_eq!(format_time_ago(now - chrono::Duration::days(2)), "2d ago");
    }

    #[test]
    fn test_markdown_stays_under_token_budget() {
        // Build a resume with lots of data to test truncation
        let discussed: Vec<DiscussedEntity> = (0..20)
            .map(|i| DiscussedEntity {
                entity_type: "File".to_string(),
                entity_id: format!("src/module{}/handler.rs", i),
                mention_count: i + 1,
                last_mentioned_at: None,
                file_path: Some(format!("src/module{}/handler.rs", i)),
            })
            .collect();

        let resume = SessionResume {
            last_session: LastSessionContext {
                session_id: Some(Uuid::new_v4()),
                session_timestamp: Some(Utc::now()),
                discussed_entities: discussed,
                recent_notes: vec![],
            },
            active_work: ActiveWorkContext {
                active_plans: vec![ActivePlanSummary {
                    plan_id: Uuid::new_v4(),
                    title: "Test Plan".to_string(),
                    status: "InProgress".to_string(),
                    total_tasks: 10,
                    completed_tasks: 5,
                    in_progress_tasks: (0..3)
                        .map(|i| TaskSummary {
                            task_id: Uuid::new_v4(),
                            title: format!("Task {}", i),
                            status: "InProgress".to_string(),
                            priority: Some(80),
                        })
                        .collect(),
                    pending_tasks: (0..5)
                        .map(|i| TaskSummary {
                            task_id: Uuid::new_v4(),
                            title: format!("Pending Task {}", i),
                            status: "Pending".to_string(),
                            priority: Some(50),
                        })
                        .collect(),
                }],
                recent_notes: (0..5)
                    .map(|i| NoteSummary {
                        note_id: Uuid::new_v4(),
                        note_type: "Gotcha".to_string(),
                        importance: "High".to_string(),
                        content_preview: format!("Important note content #{}", i),
                    })
                    .collect(),
            },
            load_time_ms: 42,
        };

        let md = resume.to_markdown();
        // Rough token estimate: ~4 chars per token
        let estimated_tokens = md.len() / 4;
        assert!(
            estimated_tokens < 1000,
            "Markdown should be under 1000 tokens, got ~{}",
            estimated_tokens
        );
        // Files should be truncated to 8
        let file_count = md.matches("handler.rs").count();
        assert!(
            file_count <= 8,
            "Should show max 8 files, got {}",
            file_count
        );
    }

    // ========================================================================
    // E2E Scenario Tests (TP4.5)
    // ========================================================================

    /// E2E #1: New session → previous session context loaded
    #[test]
    fn test_e2e_session_resume_loaded_and_injected() {
        // Simulate a resume that would be loaded at session start
        let resume = SessionResume {
            last_session: LastSessionContext {
                session_id: Some(Uuid::new_v4()),
                session_timestamp: Some(Utc::now() - chrono::Duration::hours(1)),
                discussed_entities: vec![
                    DiscussedEntity {
                        entity_type: "File".to_string(),
                        entity_id: "src/chat/feedback.rs".to_string(),
                        mention_count: 5,
                        last_mentioned_at: Some(Utc::now().to_rfc3339()),
                        file_path: Some("src/chat/feedback.rs".to_string()),
                    },
                    DiscussedEntity {
                        entity_type: "Function".to_string(),
                        entity_id: "spawn_feedback".to_string(),
                        mention_count: 3,
                        last_mentioned_at: None,
                        file_path: None,
                    },
                ],
                recent_notes: vec![],
            },
            active_work: ActiveWorkContext {
                active_plans: vec![ActivePlanSummary {
                    plan_id: Uuid::new_v4(),
                    title: "TP4 — Session Continuity & Feedback".to_string(),
                    status: "InProgress".to_string(),
                    total_tasks: 5,
                    completed_tasks: 3,
                    in_progress_tasks: vec![TaskSummary {
                        task_id: Uuid::new_v4(),
                        title: "TP4.4 — Smart System Prompt".to_string(),
                        status: "InProgress".to_string(),
                        priority: Some(72),
                    }],
                    pending_tasks: vec![TaskSummary {
                        task_id: Uuid::new_v4(),
                        title: "TP4.5 — Tests E2E".to_string(),
                        status: "Pending".to_string(),
                        priority: Some(68),
                    }],
                }],
                recent_notes: vec![NoteSummary {
                    note_id: Uuid::new_v4(),
                    note_type: "Gotcha".to_string(),
                    importance: "Critical".to_string(),
                    content_preview: "Never use git add -A in backend".to_string(),
                }],
            },
            load_time_ms: 42,
        };

        assert!(resume.has_content());
        let md = resume.to_markdown();

        // Verify previous session entities are shown
        assert!(
            md.contains("src/chat/feedback.rs"),
            "Should contain discussed file"
        );
        assert!(
            md.contains("spawn_feedback"),
            "Should contain discussed function"
        );

        // Verify active work is shown
        assert!(
            md.contains("TP4 — Session Continuity"),
            "Should contain active plan"
        );
        assert!(md.contains("3/5 tasks"), "Should show progress fraction");
        assert!(
            md.contains("TP4.4 — Smart System Prompt"),
            "Should show in-progress task"
        );

        // Verify recent notes from last session
        assert!(
            md.contains("Never use git add -A"),
            "Should contain recent note"
        );

        // Verify markdown is concise (under 1000 tokens)
        let token_estimate = md.len() / 4;
        assert!(
            token_estimate < 1000,
            "Resume should be under 1000 tokens, got ~{}",
            token_estimate
        );
    }
}
