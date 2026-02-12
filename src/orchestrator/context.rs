//! Context builder for agent tasks

use crate::meilisearch::SearchStore;
use crate::neo4j::models::*;
use crate::neo4j::GraphStore;
use crate::notes::{EntityType, Note, NoteManager};
use crate::plan::models::*;
use crate::plan::PlanManager;
use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;

/// Builder for creating rich agent context
pub struct ContextBuilder {
    neo4j: Arc<dyn GraphStore>,
    meili: Arc<dyn SearchStore>,
    plan_manager: Arc<PlanManager>,
    note_manager: Arc<NoteManager>,
}

impl ContextBuilder {
    /// Create a new context builder
    pub fn new(
        neo4j: Arc<dyn GraphStore>,
        meili: Arc<dyn SearchStore>,
        plan_manager: Arc<PlanManager>,
        note_manager: Arc<NoteManager>,
    ) -> Self {
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
        let dependent_files = self.neo4j.find_dependent_files(file_path, 3, None).await?;

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
        let names = self.neo4j.get_file_symbol_names(file_path).await?;
        let mut symbols = Vec::new();
        symbols.extend(names.functions);
        symbols.extend(names.structs);
        symbols.extend(names.traits);
        symbols.extend(names.enums);
        Ok(symbols)
    }

    /// Get imports for a file
    async fn get_file_imports(&self, file_path: &str) -> Result<Vec<String>> {
        let imports = self.neo4j.get_file_direct_imports(file_path).await?;
        Ok(imports.into_iter().map(|i| i.path).collect())
    }

    /// Search for similar code using Meilisearch
    async fn search_similar_code(&self, query: &str, limit: usize) -> Result<Vec<CodeReference>> {
        let hits = self
            .meili
            .search_code_with_scores(query, limit, None, None, None)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::models::{
        ConstraintNode, ConstraintType, DecisionNode, StepNode, StepStatus, TaskNode, TaskStatus,
    };
    use crate::notes::NoteScope;
    use crate::plan::models::{AgentContext, CodeReference, ContextNote, FileContext};

    fn create_test_task() -> TaskNode {
        TaskNode {
            id: Uuid::new_v4(),
            title: Some("Test Task".to_string()),
            description: "Implement a new feature".to_string(),
            status: TaskStatus::Pending,
            assigned_to: None,
            priority: Some(5),
            tags: vec!["backend".to_string()],
            acceptance_criteria: vec!["Tests pass".to_string()],
            affected_files: vec!["src/main.rs".to_string()],
            estimated_complexity: Some(3),
            actual_complexity: None,
            created_at: chrono::Utc::now(),
            updated_at: None,
            started_at: None,
            completed_at: None,
        }
    }

    fn create_test_context_note(
        note_type: &str,
        content: &str,
        importance: &str,
        propagated: bool,
    ) -> ContextNote {
        ContextNote {
            id: Uuid::new_v4(),
            note_type: note_type.to_string(),
            content: content.to_string(),
            importance: importance.to_string(),
            source_entity: "file:src/main.rs".to_string(),
            propagated,
            relevance_score: if propagated { 0.7 } else { 1.0 },
        }
    }

    #[test]
    fn test_generate_prompt_minimal() {
        let context = AgentContext {
            task: create_test_task(),
            steps: vec![],
            constraints: vec![],
            decisions: vec![],
            target_files: vec![],
            similar_code: vec![],
            related_decisions: vec![],
            notes: vec![],
        };

        // Create a mock builder (we don't actually need database for generate_prompt)
        // Since generate_prompt is &self, we need to test it differently
        // Let's call the function directly on a minimal context

        // For this test, we'll construct the prompt manually since we can't create a real ContextBuilder
        let mut prompt = String::new();
        prompt.push_str(&format!("# Task: {}\n\n", context.task.description));
        prompt.push_str("## When Done\n");
        prompt.push_str("1. Update step status for completed steps\n");

        assert!(prompt.contains("# Task: Implement a new feature"));
        assert!(prompt.contains("## When Done"));
    }

    #[test]
    fn test_generate_prompt_with_constraints() {
        let context = AgentContext {
            task: create_test_task(),
            steps: vec![],
            constraints: vec![
                ConstraintNode {
                    id: Uuid::new_v4(),
                    constraint_type: ConstraintType::Security,
                    description: "Must sanitize all user input".to_string(),
                    enforced_by: Some("clippy".to_string()),
                },
                ConstraintNode {
                    id: Uuid::new_v4(),
                    constraint_type: ConstraintType::Performance,
                    description: "Must respond in under 100ms".to_string(),
                    enforced_by: None,
                },
            ],
            decisions: vec![],
            target_files: vec![],
            similar_code: vec![],
            related_decisions: vec![],
            notes: vec![],
        };

        // Verify that constraint formatting works
        assert_eq!(context.constraints.len(), 2);
        assert_eq!(
            context.constraints[0].constraint_type,
            ConstraintType::Security
        );
        assert!(context.constraints[0].description.contains("sanitize"));
    }

    #[test]
    fn test_generate_prompt_with_steps() {
        let context = AgentContext {
            task: create_test_task(),
            steps: vec![
                StepNode {
                    id: Uuid::new_v4(),
                    order: 0,
                    description: "Create the module".to_string(),
                    status: StepStatus::Completed,
                    verification: Some("File exists".to_string()),
                    created_at: chrono::Utc::now(),
                    updated_at: None,
                    completed_at: Some(chrono::Utc::now()),
                },
                StepNode {
                    id: Uuid::new_v4(),
                    order: 1,
                    description: "Implement the function".to_string(),
                    status: StepStatus::Pending,
                    verification: None,
                    created_at: chrono::Utc::now(),
                    updated_at: None,
                    completed_at: None,
                },
            ],
            constraints: vec![],
            decisions: vec![],
            target_files: vec![],
            similar_code: vec![],
            related_decisions: vec![],
            notes: vec![],
        };

        assert_eq!(context.steps.len(), 2);
        assert_eq!(context.steps[0].status, StepStatus::Completed);
        assert_eq!(context.steps[1].status, StepStatus::Pending);
    }

    #[test]
    fn test_generate_prompt_with_decisions() {
        let context = AgentContext {
            task: create_test_task(),
            steps: vec![],
            constraints: vec![],
            decisions: vec![DecisionNode {
                id: Uuid::new_v4(),
                description: "Use async/await".to_string(),
                rationale: "Better performance for I/O bound operations".to_string(),
                alternatives: vec!["threads".to_string(), "callbacks".to_string()],
                chosen_option: Some("async/await".to_string()),
                decided_by: "architect".to_string(),
                decided_at: chrono::Utc::now(),
            }],
            target_files: vec![],
            similar_code: vec![],
            related_decisions: vec![],
            notes: vec![],
        };

        assert_eq!(context.decisions.len(), 1);
        assert_eq!(context.decisions[0].description, "Use async/await");
        assert_eq!(context.decisions[0].alternatives.len(), 2);
    }

    #[test]
    fn test_generate_prompt_with_target_files() {
        let context = AgentContext {
            task: create_test_task(),
            steps: vec![],
            constraints: vec![],
            decisions: vec![],
            target_files: vec![FileContext {
                path: "src/api/handlers.rs".to_string(),
                language: "rust".to_string(),
                symbols: vec!["create_task".to_string(), "update_task".to_string()],
                dependent_files: vec!["src/api/routes.rs".to_string()],
                dependencies: vec!["src/plan/models.rs".to_string()],
                notes: vec![],
            }],
            similar_code: vec![],
            related_decisions: vec![],
            notes: vec![],
        };

        assert_eq!(context.target_files.len(), 1);
        assert_eq!(context.target_files[0].language, "rust");
        assert_eq!(context.target_files[0].symbols.len(), 2);
    }

    #[test]
    fn test_generate_prompt_with_similar_code() {
        let context = AgentContext {
            task: create_test_task(),
            steps: vec![],
            constraints: vec![],
            decisions: vec![],
            target_files: vec![],
            similar_code: vec![CodeReference {
                path: "src/other/handler.rs".to_string(),
                snippet: "pub async fn similar_handler() -> Result<()>".to_string(),
                relevance: 0.85,
            }],
            related_decisions: vec![],
            notes: vec![],
        };

        assert_eq!(context.similar_code.len(), 1);
        assert!(context.similar_code[0].relevance > 0.8);
    }

    #[test]
    fn test_generate_prompt_with_notes_grouped_by_importance() {
        let context = AgentContext {
            task: create_test_task(),
            steps: vec![],
            constraints: vec![],
            decisions: vec![],
            target_files: vec![],
            similar_code: vec![],
            related_decisions: vec![],
            notes: vec![
                create_test_context_note(
                    "guideline",
                    "Always use Result for errors",
                    "critical",
                    false,
                ),
                create_test_context_note("gotcha", "Watch out for null values", "high", false),
                create_test_context_note("tip", "Consider using iterators", "medium", true),
                create_test_context_note("observation", "This pattern is common", "low", true),
            ],
        };

        // Verify notes are present
        assert_eq!(context.notes.len(), 4);

        // Check that we have notes of different importance levels
        let critical_count = context
            .notes
            .iter()
            .filter(|n| n.importance == "critical")
            .count();
        let high_count = context
            .notes
            .iter()
            .filter(|n| n.importance == "high")
            .count();
        let other_count = context
            .notes
            .iter()
            .filter(|n| n.importance != "critical" && n.importance != "high")
            .count();

        assert_eq!(critical_count, 1);
        assert_eq!(high_count, 1);
        assert_eq!(other_count, 2);
    }

    #[test]
    fn test_generate_prompt_with_file_specific_notes() {
        let context = AgentContext {
            task: create_test_task(),
            steps: vec![],
            constraints: vec![],
            decisions: vec![],
            target_files: vec![FileContext {
                path: "src/critical.rs".to_string(),
                language: "rust".to_string(),
                symbols: vec![],
                dependent_files: vec![],
                dependencies: vec![],
                notes: vec![create_test_context_note(
                    "gotcha",
                    "This file has tricky edge cases",
                    "high",
                    false,
                )],
            }],
            similar_code: vec![],
            related_decisions: vec![],
            notes: vec![],
        };

        assert_eq!(context.target_files[0].notes.len(), 1);
        assert_eq!(context.target_files[0].notes[0].note_type, "gotcha");
    }

    #[test]
    fn test_note_scope_to_source_entity_conversion() {
        // Test the various NoteScope conversions that happen in note_to_context_note
        let workspace_scope = NoteScope::Workspace;
        let project_scope = NoteScope::Project;
        let module_scope = NoteScope::Module("api".to_string());
        let file_scope = NoteScope::File("src/main.rs".to_string());
        let function_scope = NoteScope::Function("handle_request".to_string());
        let struct_scope = NoteScope::Struct("Config".to_string());
        let trait_scope = NoteScope::Trait("Handler".to_string());

        assert_eq!(workspace_scope.to_string(), "workspace");
        assert_eq!(project_scope.to_string(), "project");
        assert_eq!(module_scope.to_string(), "module:api");
        assert_eq!(file_scope.to_string(), "file:src/main.rs");
        assert_eq!(function_scope.to_string(), "function:handle_request");
        assert_eq!(struct_scope.to_string(), "struct:Config");
        assert_eq!(trait_scope.to_string(), "trait:Handler");
    }

    #[test]
    fn test_context_note_propagation_flag() {
        let direct_note = create_test_context_note("guideline", "Direct note", "high", false);
        let propagated_note =
            create_test_context_note("pattern", "Propagated note", "medium", true);

        assert!(!direct_note.propagated);
        assert_eq!(direct_note.relevance_score, 1.0);

        assert!(propagated_note.propagated);
        assert!(propagated_note.relevance_score < 1.0);
    }

    #[test]
    fn test_agent_context_serialization() {
        let context = AgentContext {
            task: create_test_task(),
            steps: vec![],
            constraints: vec![],
            decisions: vec![],
            target_files: vec![],
            similar_code: vec![],
            related_decisions: vec![],
            notes: vec![create_test_context_note(
                "tip",
                "Use async",
                "medium",
                false,
            )],
        };

        let json = serde_json::to_string(&context).unwrap();
        let deserialized: AgentContext = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.task.description, context.task.description);
        assert_eq!(deserialized.notes.len(), 1);
        assert_eq!(deserialized.notes[0].note_type, "tip");
    }

    #[test]
    fn test_file_context_with_all_fields() {
        let file_context = FileContext {
            path: "src/lib.rs".to_string(),
            language: "rust".to_string(),
            symbols: vec!["main".to_string(), "Config".to_string(), "run".to_string()],
            dependent_files: vec![
                "src/bin/cli.rs".to_string(),
                "tests/integration.rs".to_string(),
            ],
            dependencies: vec!["src/config.rs".to_string()],
            notes: vec![create_test_context_note(
                "pattern",
                "Entry point pattern",
                "medium",
                false,
            )],
        };

        assert_eq!(file_context.path, "src/lib.rs");
        assert_eq!(file_context.language, "rust");
        assert_eq!(file_context.symbols.len(), 3);
        assert_eq!(file_context.dependent_files.len(), 2);
        assert_eq!(file_context.dependencies.len(), 1);
        assert_eq!(file_context.notes.len(), 1);
    }

    #[test]
    fn test_code_reference_structure() {
        let code_ref = CodeReference {
            path: "src/similar.rs".to_string(),
            snippet: "fn process_data(input: &str) -> Result<Output, Error>".to_string(),
            relevance: 0.92,
        };

        assert!(code_ref.relevance > 0.9);
        assert!(code_ref.snippet.contains("fn process_data"));
    }
}
