//! Context builder for agent tasks

use crate::meilisearch::client::MeiliClient;
use crate::neo4j::client::Neo4jClient;
use crate::neo4j::models::*;
use crate::notes::{EntityType, Note, NoteManager};
use crate::plan::models::*;
use crate::plan::PlanManager;
use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;

/// Builder for creating rich agent context
pub struct ContextBuilder {
    neo4j: Arc<Neo4jClient>,
    meili: Arc<MeiliClient>,
    plan_manager: Arc<PlanManager>,
    note_manager: Arc<NoteManager>,
}

impl ContextBuilder {
    /// Create a new context builder
    pub fn new(
        neo4j: Arc<Neo4jClient>,
        meili: Arc<MeiliClient>,
        plan_manager: Arc<PlanManager>,
    ) -> Self {
        let note_manager = Arc::new(NoteManager::new(neo4j.clone(), meili.clone()));
        Self {
            neo4j,
            meili,
            plan_manager,
            note_manager,
        }
    }

    /// Build full context for executing a task
    pub async fn build_context(&self, task_id: Uuid, plan_id: Uuid) -> Result<AgentContext> {
        // Get task details
        let task_details = self
            .plan_manager
            .get_task_details(task_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Task not found"))?;

        // Get plan constraints
        let plan_details = self
            .plan_manager
            .get_plan_details(plan_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Plan not found"))?;

        // Get file contexts for modified files (including notes)
        let mut target_files = Vec::new();
        for file_path in &task_details.modifies_files {
            let file_context = self.get_file_context_with_notes(file_path).await?;
            target_files.push(file_context);
        }

        // Search for similar code
        let similar_code = self
            .search_similar_code(&task_details.task.description, 5)
            .await?;

        // Search for related decisions
        let related_decisions = self
            .plan_manager
            .search_decisions(&task_details.task.description, 5)
            .await?;

        // Get notes for the task
        let task_notes = self
            .get_notes_for_entity(&EntityType::Task, &task_id.to_string())
            .await?;

        // Get notes for the plan
        let plan_notes = self
            .get_notes_for_entity(&EntityType::Plan, &plan_id.to_string())
            .await?;

        // Combine all notes
        let mut all_notes = task_notes;
        all_notes.extend(plan_notes);

        // Deduplicate notes by ID
        all_notes.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_notes.dedup_by_key(|n| n.id);

        Ok(AgentContext {
            task: task_details.task,
            steps: task_details.steps,
            constraints: plan_details.constraints,
            decisions: task_details.decisions,
            target_files,
            similar_code,
            related_decisions,
            notes: all_notes,
        })
    }

    /// Get context for a specific file
    pub async fn get_file_context(&self, file_path: &str) -> Result<FileContext> {
        // Get file info from Neo4j
        let file = self.neo4j.get_file(file_path).await?;

        // Get symbols in this file
        let symbols = self.get_file_symbols(file_path).await?;

        // Get dependent files (files that import this file)
        let dependent_files = self.neo4j.find_dependent_files(file_path, 3).await?;

        // Get files this file imports
        let dependencies = self.get_file_imports(file_path).await?;

        Ok(FileContext {
            path: file_path.to_string(),
            language: file.map(|f| f.language).unwrap_or_default(),
            symbols,
            dependent_files,
            dependencies,
            notes: Vec::new(),
        })
    }

    /// Get file context with attached notes
    pub async fn get_file_context_with_notes(&self, file_path: &str) -> Result<FileContext> {
        let mut context = self.get_file_context(file_path).await?;

        // Get notes for this file
        context.notes = self
            .get_notes_for_entity(&EntityType::File, file_path)
            .await?;

        Ok(context)
    }

    /// Get notes for an entity (direct + propagated)
    async fn get_notes_for_entity(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> Result<Vec<ContextNote>> {
        // Get contextual notes (direct + propagated)
        let note_context = self
            .note_manager
            .get_context_notes(entity_type, entity_id, 2, 0.2)
            .await?;

        let mut context_notes = Vec::new();

        // Convert direct notes
        for note in note_context.direct_notes {
            context_notes.push(self.note_to_context_note(&note, false, 1.0));
        }

        // Convert propagated notes
        for prop_note in note_context.propagated_notes {
            context_notes.push(ContextNote {
                id: prop_note.note.id,
                note_type: prop_note.note.note_type.to_string(),
                content: prop_note.note.content.clone(),
                importance: prop_note.note.importance.to_string(),
                source_entity: prop_note.source_entity,
                propagated: true,
                relevance_score: prop_note.relevance_score,
            });
        }

        Ok(context_notes)
    }

    /// Convert a Note to ContextNote
    fn note_to_context_note(&self, note: &Note, propagated: bool, relevance: f64) -> ContextNote {
        ContextNote {
            id: note.id,
            note_type: note.note_type.to_string(),
            content: note.content.clone(),
            importance: note.importance.to_string(),
            source_entity: match &note.scope {
                crate::notes::NoteScope::Workspace => "workspace".to_string(),
                crate::notes::NoteScope::Project => "project".to_string(),
                crate::notes::NoteScope::Module(m) => format!("module:{}", m),
                crate::notes::NoteScope::File(f) => format!("file:{}", f),
                crate::notes::NoteScope::Function(f) => format!("function:{}", f),
                crate::notes::NoteScope::Struct(s) => format!("struct:{}", s),
                crate::notes::NoteScope::Trait(t) => format!("trait:{}", t),
            },
            propagated,
            relevance_score: relevance,
        }
    }

    /// Get symbols defined in a file
    async fn get_file_symbols(&self, file_path: &str) -> Result<Vec<String>> {
        let q = neo4rs::query(
            r#"
            MATCH (f:File {path: $path})-[:CONTAINS]->(entity)
            RETURN entity.name AS name
            "#,
        )
        .param("path", file_path);

        let rows = self.neo4j.execute_with_params(q).await?;
        let symbols: Vec<String> = rows
            .into_iter()
            .filter_map(|r| r.get("name").ok())
            .collect();

        Ok(symbols)
    }

    /// Get imports for a file
    async fn get_file_imports(&self, file_path: &str) -> Result<Vec<String>> {
        let q = neo4rs::query(
            r#"
            MATCH (f:File {path: $path})-[:IMPORTS]->(imported:File)
            RETURN imported.path AS path
            "#,
        )
        .param("path", file_path);

        let rows = self.neo4j.execute_with_params(q).await?;
        let imports: Vec<String> = rows
            .into_iter()
            .filter_map(|r| r.get("path").ok())
            .collect();

        Ok(imports)
    }

    /// Search for similar code using Meilisearch
    async fn search_similar_code(&self, query: &str, limit: usize) -> Result<Vec<CodeReference>> {
        let hits = self
            .meili
            .search_code_with_scores(query, limit, None, None)
            .await?;

        let references = hits
            .into_iter()
            .map(|hit| CodeReference {
                path: hit.document.path,
                snippet: hit.document.docstrings.chars().take(500).collect(),
                relevance: hit.score as f32,
            })
            .collect();

        Ok(references)
    }

    /// Generate a prompt for an agent
    pub fn generate_prompt(&self, context: &AgentContext) -> String {
        let mut prompt = String::new();

        // Task description
        prompt.push_str(&format!("# Task: {}\n\n", context.task.description));

        // Constraints
        if !context.constraints.is_empty() {
            prompt.push_str("## Constraints\n");
            for constraint in &context.constraints {
                prompt.push_str(&format!(
                    "- [{:?}] {}\n",
                    constraint.constraint_type, constraint.description
                ));
            }
            prompt.push('\n');
        }

        // Steps
        if !context.steps.is_empty() {
            prompt.push_str("## Steps\n");
            for step in &context.steps {
                let status = match step.status {
                    StepStatus::Completed => "[x]",
                    _ => "[ ]",
                };
                prompt.push_str(&format!(
                    "{} {}. {}\n",
                    status,
                    step.order + 1,
                    step.description
                ));
                if let Some(ref verification) = step.verification {
                    prompt.push_str(&format!("   Verification: {}\n", verification));
                }
            }
            prompt.push('\n');
        }

        // Previous decisions
        if !context.decisions.is_empty() {
            prompt.push_str("## Decisions Already Made\n");
            for decision in &context.decisions {
                prompt.push_str(&format!(
                    "- **{}**: {}\n",
                    decision.description, decision.rationale
                ));
            }
            prompt.push('\n');
        }

        // Target files
        if !context.target_files.is_empty() {
            prompt.push_str("## Files to Modify\n");
            for file in &context.target_files {
                prompt.push_str(&format!("### {}\n", file.path));
                prompt.push_str(&format!("- Language: {}\n", file.language));
                if !file.symbols.is_empty() {
                    prompt.push_str(&format!("- Symbols: {}\n", file.symbols.join(", ")));
                }
                if !file.dependent_files.is_empty() {
                    prompt.push_str(&format!(
                        "- Impacted files: {}\n",
                        file.dependent_files.join(", ")
                    ));
                }
                prompt.push('\n');
            }
        }

        // Similar code
        if !context.similar_code.is_empty() {
            prompt.push_str("## Similar Code (for reference)\n");
            for code_ref in &context.similar_code {
                prompt.push_str(&format!(
                    "### {}\n```\n{}\n```\n\n",
                    code_ref.path, code_ref.snippet
                ));
            }
        }

        // Related decisions
        if !context.related_decisions.is_empty() {
            prompt.push_str("## Related Past Decisions\n");
            for decision in &context.related_decisions {
                prompt.push_str(&format!(
                    "- **{}** (by {}): {}\n",
                    decision.description, decision.decided_by, decision.rationale
                ));
            }
            prompt.push('\n');
        }

        // Knowledge Notes (guidelines, gotchas, patterns)
        if !context.notes.is_empty() {
            prompt.push_str("## Knowledge Notes\n");
            prompt.push_str(
                "The following notes contain important context, guidelines, and gotchas:\n\n",
            );

            // Group by importance
            let critical: Vec<_> = context
                .notes
                .iter()
                .filter(|n| n.importance == "critical")
                .collect();
            let high: Vec<_> = context
                .notes
                .iter()
                .filter(|n| n.importance == "high")
                .collect();
            let other: Vec<_> = context
                .notes
                .iter()
                .filter(|n| n.importance != "critical" && n.importance != "high")
                .collect();

            if !critical.is_empty() {
                prompt.push_str("### Critical\n");
                for note in critical {
                    let source = if note.propagated {
                        format!(" (via {})", note.source_entity)
                    } else {
                        String::new()
                    };
                    prompt.push_str(&format!(
                        "- **[{}]{}** {}\n",
                        note.note_type, source, note.content
                    ));
                }
                prompt.push('\n');
            }

            if !high.is_empty() {
                prompt.push_str("### Important\n");
                for note in high {
                    let source = if note.propagated {
                        format!(" (via {})", note.source_entity)
                    } else {
                        String::new()
                    };
                    prompt.push_str(&format!(
                        "- **[{}]{}** {}\n",
                        note.note_type, source, note.content
                    ));
                }
                prompt.push('\n');
            }

            if !other.is_empty() {
                prompt.push_str("### Other Notes\n");
                for note in other {
                    let source = if note.propagated {
                        format!(" (via {})", note.source_entity)
                    } else {
                        String::new()
                    };
                    prompt.push_str(&format!(
                        "- [{}]{} {}\n",
                        note.note_type, source, note.content
                    ));
                }
                prompt.push('\n');
            }
        }

        // File-specific notes
        let files_with_notes: Vec<_> = context
            .target_files
            .iter()
            .filter(|f| !f.notes.is_empty())
            .collect();
        if !files_with_notes.is_empty() {
            prompt.push_str("## File-Specific Notes\n");
            for file in files_with_notes {
                prompt.push_str(&format!("### {}\n", file.path));
                for note in &file.notes {
                    prompt.push_str(&format!("- [{}] {}\n", note.note_type, note.content));
                }
                prompt.push('\n');
            }
        }

        // Instructions
        prompt.push_str("## When Done\n");
        prompt.push_str("1. Update step status for completed steps\n");
        prompt.push_str("2. Record any architectural decisions made\n");
        prompt.push_str("3. Link files that were actually modified\n");
        prompt.push_str("4. Send completion notification via webhook\n");

        prompt
    }
}
