//! Context builder for agent tasks
//!
//! Provides two modes of prompt generation:
//! - `generate_prompt()` — static markdown prompt (backward compatible)
//! - `build_enriched_context()` — runs the EnrichmentPipeline before prompt assembly

use crate::chat::enrichment::{EnrichmentInput, EnrichmentPipeline};
use crate::meilisearch::SearchStore;
use crate::neo4j::models::*;
use crate::neo4j::GraphStore;
use crate::notes::{EntityType, Note, NoteManager};
use crate::plan::models::*;
use crate::plan::PlanManager;
use crate::runner::prompt::{PromptBuilder, StructuredPrompt};
use anyhow::Result;
use std::sync::Arc;
use tracing::debug;
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

        // Retrieval is scoped to the plan's own projects. Both searches used to
        // run against the WHOLE index: a React card task in the PO Frontend got
        // an eigensolver decision from obrain and replay-buffer code from elun —
        // pure noise in the agent's context, on every plan. A plan with no
        // project gets no retrieval at all rather than a guess from everywhere.
        let project_slugs = self
            .neo4j
            .list_plan_project_slugs(plan_id)
            .await
            .unwrap_or_default();
        let search_query = task_search_query(&task_details.task);

        let similar_code = self
            .search_similar_code(&search_query, 5, &project_slugs)
            .await?;

        let related_decisions = self
            .search_related_decisions(&search_query, 5, &project_slugs)
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

        // Deduplicate notes by ID, keeping the most relevant copy.
        //
        // `Vec::dedup_by_key` only removes CONSECUTIVE duplicates. After sorting
        // by relevance — where linked notes all score 1.0 — two copies of the
        // same note are not necessarily adjacent, so both survived: a note linked
        // to the task came back once through the task and once through its
        // project. A seen-set removes every repeat whatever its position.
        dedupe_notes_by_id(&mut all_notes);

        // Biomimicry: Frustration-Catharsis — inject signals when frustration is elevated
        let frustration_signals =
            FrustrationSignals::from_score(task_details.task.frustration_score);

        // When frustrated, widen the search for additional context (thermal noise injection)
        if let Some(ref signals) = frustration_signals {
            if signals.widen_search {
                // Fetch broader semantic notes to increase search radius
                let extra_notes = self
                    .get_notes_for_entity(&EntityType::Plan, &plan_id.to_string())
                    .await
                    .unwrap_or_default();
                for note in extra_notes {
                    if !all_notes.iter().any(|n| n.id == note.id) {
                        all_notes.push(note);
                    }
                }

                if signals.level == FrustrationLevel::Critical {
                    tracing::warn!(
                        task_id = %task_id,
                        frustration_score = task_details.task.frustration_score,
                        "🔴 CATHARSIS THRESHOLD REACHED: task frustration ≥ 0.9 — agent should consider deep reasoning or task re-evaluation"
                    );
                } else {
                    tracing::info!(
                        task_id = %task_id,
                        frustration_score = task_details.task.frustration_score,
                        level = ?signals.level,
                        "Frustration-Catharsis: widened knowledge search for frustrated task"
                    );
                }
            }
        }

        Ok(AgentContext {
            task: task_details.task,
            steps: task_details.steps,
            constraints: plan_details.constraints,
            decisions: task_details.decisions,
            target_files,
            similar_code,
            related_decisions,
            notes: all_notes,
            frustration_signals,
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

    /// Search for similar code, restricted to the given projects.
    ///
    /// Each project is searched separately (the index takes one project filter),
    /// then hits are merged by score. No project → no results: an unscoped search
    /// returns whatever is closest anywhere on the machine, which is how unrelated
    /// repositories ended up in task prompts.
    async fn search_similar_code(
        &self,
        query: &str,
        limit: usize,
        project_slugs: &[String],
    ) -> Result<Vec<CodeReference>> {
        let mut references: Vec<CodeReference> = Vec::new();
        for slug in project_slugs {
            let hits = self
                .meili
                .search_code_with_scores(query, limit, None, Some(slug), None)
                .await?;
            references.extend(hits.into_iter().map(|hit| CodeReference {
                path: hit.document.path,
                snippet: hit.document.docstrings.chars().take(500).collect(),
                relevance: hit.score as f32,
            }));
        }

        Ok(merge_code_references(references, limit))
    }

    /// Search past decisions, restricted to the given projects.
    ///
    /// Decisions carry no score, so per-project results are interleaved rather
    /// than concatenated — otherwise the alphabetically first project would take
    /// every slot and a backend task on a frontend+backend plan would only ever
    /// see frontend decisions.
    async fn search_related_decisions(
        &self,
        query: &str,
        limit: usize,
        project_slugs: &[String],
    ) -> Result<Vec<DecisionNode>> {
        let mut per_project: Vec<Vec<DecisionNode>> = Vec::new();
        for slug in project_slugs {
            per_project.push(
                self.plan_manager
                    .search_decisions(query, limit, Some(slug))
                    .await?,
            );
        }
        Ok(interleave_decisions(per_project, limit))
    }

    /// Build enriched context by running the EnrichmentPipeline on top of the base context.
    ///
    /// This is the preferred entry point for runner agents: it combines
    /// `build_context()` + EnrichmentPipeline + propagated notes into a
    /// single [`StructuredPrompt`] via [`PromptBuilder`].
    ///
    /// The `project_slug` and `project_id` are used to scope enrichment stages.
    /// `custom_sections` are appended as `PromptSection::Custom`.
    pub async fn build_enriched_context(
        &self,
        task_id: Uuid,
        plan_id: Uuid,
        pipeline: Option<&EnrichmentPipeline>,
        project_slug: Option<&str>,
        project_id: Option<Uuid>,
        custom_sections: Vec<String>,
    ) -> Result<StructuredPrompt> {
        // 1. Build base context (reuse existing logic)
        let context = self.build_context(task_id, plan_id).await?;

        // 2. Build a PromptBuilder from the AgentContext
        let mut builder = self.build_prompt_builder(&context);

        // 3. Run EnrichmentPipeline if provided
        if let Some(pipeline) = pipeline {
            let input = EnrichmentInput {
                message: context.task.description.clone(),
                session_id: Uuid::new_v4(), // ephemeral session for enrichment
                project_slug: project_slug.map(|s| s.to_string()),
                project_id,
                cwd: None,
                protocol_run_id: None,
                protocol_state: None,
                excluded_note_ids: Default::default(),
                reasoning_path_tracker: None,
            };

            let enrichment_ctx = pipeline.execute(&input).await;

            if enrichment_ctx.has_content() {
                let rendered = enrichment_ctx.render();
                builder = builder.with_enrichment(rendered);
                debug!(
                    task_id = %task_id,
                    sections = enrichment_ctx.sections.len(),
                    total_ms = enrichment_ctx.total_time_ms,
                    "EnrichmentPipeline injected into task context"
                );
            }
        }

        // 4. Inject propagated notes for affected files (Step 4 — Knowledge Fabric)
        let propagated_notes_text = self
            .collect_propagated_notes_for_files(&context.task.affected_files)
            .await;
        if !propagated_notes_text.is_empty() {
            builder = builder.with_propagated_notes(propagated_notes_text);
        }

        // 5. Add custom sections
        for custom in custom_sections {
            if !custom.is_empty() {
                builder = builder.with_custom(custom);
            }
        }

        Ok(builder.build_structured())
    }

    /// Build a [`PromptBuilder`] from an [`AgentContext`].
    ///
    /// Converts each part of the context into the appropriate [`PromptSection`].
    /// The caller can add more sections before calling `.build()`.
    pub fn build_prompt_builder(&self, context: &AgentContext) -> PromptBuilder {
        let mut builder = PromptBuilder::new();

        // Task statement — same rendering as generate_prompt (render_task_statement).
        let (heading, body) = render_task_statement(&context.task);
        let task_section = if body.is_empty() {
            format!("{}\n", heading)
        } else {
            format!("{}\n\n{}", heading, body)
        };
        builder = builder.with_task(task_section);

        // Constraints
        if !context.constraints.is_empty() {
            let mut s = String::new();
            for constraint in &context.constraints {
                s.push_str(&format!(
                    "- [{:?}] {}\n",
                    constraint.constraint_type, constraint.description
                ));
            }
            builder = builder.with_constraints(s);
        }

        // Steps
        if !context.steps.is_empty() {
            let mut s = String::new();
            for step in &context.steps {
                let status = match step.status {
                    StepStatus::Completed => "[x]",
                    _ => "[ ]",
                };
                s.push_str(&format!(
                    "{} {}. {} `[step_id: {}]`\n",
                    status,
                    step.order + 1,
                    step.description,
                    step.id
                ));
                if let Some(ref verification) = step.verification {
                    s.push_str(&format!("   Verification: {}\n", verification));
                }
            }
            builder = builder.with_steps(s);
        }

        // Knowledge Notes
        if !context.notes.is_empty() {
            let mut s = String::new();
            s.push_str(
                "The following notes contain important context, guidelines, and gotchas:\n\n",
            );

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
                s.push_str("### Critical\n");
                for note in critical {
                    let source = if note.propagated {
                        format!(" (via {})", note.source_entity)
                    } else {
                        String::new()
                    };
                    s.push_str(&format!(
                        "- **[{}]{}** {}\n",
                        note.note_type, source, note.content
                    ));
                }
                s.push('\n');
            }
            if !high.is_empty() {
                s.push_str("### Important\n");
                for note in high {
                    let source = if note.propagated {
                        format!(" (via {})", note.source_entity)
                    } else {
                        String::new()
                    };
                    s.push_str(&format!(
                        "- **[{}]{}** {}\n",
                        note.note_type, source, note.content
                    ));
                }
                s.push('\n');
            }
            if !other.is_empty() {
                s.push_str("### Other Notes\n");
                for note in other {
                    let source = if note.propagated {
                        format!(" (via {})", note.source_entity)
                    } else {
                        String::new()
                    };
                    s.push_str(&format!(
                        "- [{}]{} {}\n",
                        note.note_type, source, note.content
                    ));
                }
                s.push('\n');
            }
            builder = builder.with_knowledge_notes(s);
        }

        // File context
        if !context.target_files.is_empty() {
            let mut s = String::new();
            for file in &context.target_files {
                s.push_str(&format!("### {}\n", file.path));
                s.push_str(&format!("- Language: {}\n", file.language));
                if !file.symbols.is_empty() {
                    s.push_str(&format!("- Symbols: {}\n", file.symbols.join(", ")));
                }
                if !file.dependent_files.is_empty() {
                    s.push_str(&format!(
                        "- Impacted files: {}\n",
                        file.dependent_files.join(", ")
                    ));
                }
                // File-specific notes
                for note in &file.notes {
                    s.push_str(&format!("- [{}] {}\n", note.note_type, note.content));
                }
                s.push('\n');
            }
            builder = builder.with_file_context(s);
        }

        builder
    }

    /// Collect propagated notes for a list of file paths.
    ///
    /// Returns up to 5 propagated notes of importance >= medium, formatted as markdown.
    /// This implements Step 4 of the enrichment task (Knowledge Fabric injection).
    async fn collect_propagated_notes_for_files(&self, file_paths: &[String]) -> String {
        let mut all_propagated = Vec::new();

        for file_path in file_paths {
            match self
                .note_manager
                .get_propagated_notes(
                    &EntityType::File,
                    file_path,
                    2,    // max_depth
                    0.3,  // min_score
                    None, // default relation types
                    None, // no project filter
                    false,
                )
                .await
            {
                Ok(notes) => {
                    for note in notes {
                        // Filter: importance >= medium (exclude low)
                        let dominated_importance = note.note.importance.to_string();
                        if dominated_importance == "low" {
                            continue;
                        }
                        // Only include gotchas, guidelines, and patterns
                        let note_type_str = note.note.note_type.to_string();
                        if matches!(note_type_str.as_str(), "gotcha" | "guideline" | "pattern") {
                            all_propagated.push((file_path.clone(), note));
                        }
                    }
                }
                Err(e) => {
                    debug!(
                        file_path = %file_path,
                        error = %e,
                        "Failed to get propagated notes for file"
                    );
                }
            }
        }

        if all_propagated.is_empty() {
            return String::new();
        }

        // Sort by relevance score descending, take top 5
        all_propagated.sort_by(|a, b| {
            b.1.relevance_score
                .partial_cmp(&a.1.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_propagated.truncate(5);

        let mut output = String::new();
        output.push_str(
            "Propagated knowledge from the Knowledge Fabric (gotchas/guidelines/patterns):\n\n",
        );
        for (file_path, prop_note) in &all_propagated {
            output.push_str(&format!(
                "- **[{}]** ({}, via `{}`, relevance {:.0}%) {}\n",
                prop_note.note.note_type,
                prop_note.note.importance,
                file_path,
                prop_note.relevance_score * 100.0,
                prop_note.note.content,
            ));
        }
        output
    }

    /// Generate a prompt for an agent
    pub fn generate_prompt(&self, context: &AgentContext) -> String {
        let mut prompt = String::new();

        // Task statement: title, description, acceptance criteria.
        // Shared with build_prompt_builder so the two prompt paths cannot drift —
        // they already had, both dropping the acceptance criteria.
        let (heading, body) = render_task_statement(&context.task);
        prompt.push_str(&format!("# Task: {}\n\n", heading));
        if !body.is_empty() {
            prompt.push_str(&body);
            prompt.push('\n');
        }

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
                    "{} {}. {} `[step_id: {}]`\n",
                    status,
                    step.order + 1,
                    step.description,
                    step.id
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

    // ========================================================================
    // Pre-enrichment pipeline (Steps 2 & 3 of T9)
    // ========================================================================

    /// Pre-enrich a single task: build context, profile persona, cache the prompt.
    ///
    /// Persists `execution_context` (serialized AgentContext summary),
    /// `persona` (TaskProfile JSON), and `prompt_cache` (rendered prompt) on the
    /// TaskNode in Neo4j so that `execute_task()` can skip the expensive
    /// `build_context()` + `generate_prompt()` calls at runtime.
    pub async fn enrich_task(&self, task_id: Uuid, plan_id: Uuid) -> Result<EnrichmentResult> {
        use crate::runner::persona::profile_task;

        // 1. Build full context (the expensive part we want to cache)
        let context = self.build_context(task_id, plan_id).await?;

        // 2. Profile the task (persona)
        let steps_count = context.steps.len();
        let profile = profile_task(&context.task, steps_count);
        let persona_json = serde_json::to_string(&profile)?;

        // 3. Collect propagated notes for affected files
        let propagated_notes = self
            .collect_propagated_notes_for_files(&context.task.affected_files)
            .await;

        // 4. Generate the prompt (what the agent will receive)
        let mut prompt = self.generate_prompt(&context);
        if !propagated_notes.is_empty() {
            prompt.push_str("\n## Propagated Knowledge\n");
            prompt.push_str(&propagated_notes);
        }

        // 5. Build a compact execution_context summary (not the full prompt)
        let exec_ctx = serde_json::json!({
            "constraints_count": context.constraints.len(),
            "steps_count": steps_count,
            "target_files": context.target_files.iter().map(|f| &f.path).collect::<Vec<_>>(),
            "similar_code_count": context.similar_code.len(),
            "related_decisions_count": context.related_decisions.len(),
            "notes_count": context.notes.len(),
            "propagated_notes_len": propagated_notes.len(),
        });
        let exec_ctx_json = serde_json::to_string(&exec_ctx)?;

        // 6. Persist to Neo4j
        self.neo4j
            .update_task_enrichment(
                task_id,
                Some(&exec_ctx_json),
                Some(&persona_json),
                Some(&prompt),
            )
            .await?;

        debug!(
            task_id = %task_id,
            persona = %profile.complexity,
            prompt_len = prompt.len(),
            "Task pre-enrichment completed"
        );

        Ok(EnrichmentResult {
            task_id,
            persona: persona_json,
            prompt_len: prompt.len(),
            execution_context: exec_ctx_json,
        })
    }

    /// Pre-enrich all tasks in a plan.
    ///
    /// Iterates over every task in the plan, calls `enrich_task` for each,
    /// and returns a summary of results.
    pub async fn enrich_plan(&self, plan_id: Uuid) -> Result<PlanEnrichmentResult> {
        let tasks = self.neo4j.get_plan_tasks(plan_id).await?;
        if tasks.is_empty() {
            return Err(anyhow::anyhow!("Plan has no tasks or plan not found"));
        }

        let mut results = Vec::new();
        let mut errors = Vec::new();

        for task in &tasks {
            match self.enrich_task(task.id, plan_id).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    tracing::warn!(
                        task_id = %task.id,
                        error = %e,
                        "Failed to enrich task"
                    );
                    errors.push(format!("{}: {}", task.id, e));
                }
            }
        }

        Ok(PlanEnrichmentResult {
            plan_id,
            total_tasks: tasks.len(),
            enriched: results.len(),
            failed: errors.len(),
            results,
            errors,
        })
    }
}

/// Result of enriching a single task.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnrichmentResult {
    pub task_id: Uuid,
    pub persona: String,
    pub prompt_len: usize,
    pub execution_context: String,
}

/// Result of enriching all tasks in a plan.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlanEnrichmentResult {
    pub plan_id: Uuid,
    pub total_tasks: usize,
    pub enriched: usize,
    pub failed: usize,
    pub results: Vec<EnrichmentResult>,
    pub errors: Vec<String>,
}

// ============================================================================
// Standalone utilities
// ============================================================================

/// Pre-read affected files and format them as markdown code blocks.
///
/// Gives runner-spawned agents immediate file context so they can start coding
/// without spending turns on file discovery. Limits:
/// - Max `max_files` files (default 5)
/// - Max `max_lines` lines per file (default 200)
/// - Skips files > 500 lines (too large for prompt injection)
/// - Resolves relative paths against `cwd`
pub async fn pre_read_affected_files(
    cwd: &str,
    files: &[String],
    max_files: usize,
    max_lines: usize,
) -> String {
    use std::path::PathBuf;
    use tokio::fs;

    if files.is_empty() {
        return String::new();
    }

    let mut output = String::from("## Current File Contents\n\n");
    let mut files_read = 0;

    for file_path in files {
        if files_read >= max_files {
            output.push_str(&format!(
                "_({} more files not shown)_\n",
                files.len() - files_read
            ));
            break;
        }

        // Resolve path: if relative, prepend cwd
        let full_path = if file_path.starts_with('/') {
            PathBuf::from(file_path)
        } else {
            PathBuf::from(cwd).join(file_path)
        };

        match fs::read_to_string(&full_path).await {
            Ok(content) => {
                let lines: Vec<&str> = content.lines().collect();

                // Skip files that are too large
                if lines.len() > 500 {
                    output.push_str(&format!(
                        "### `{}`\n_({} lines — too large, read it yourself)_\n\n",
                        file_path,
                        lines.len()
                    ));
                    files_read += 1;
                    continue;
                }

                // Detect language from extension
                let lang = full_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|ext| match ext {
                        "rs" => "rust",
                        "ts" | "tsx" => "typescript",
                        "js" | "jsx" => "javascript",
                        "py" => "python",
                        "go" => "go",
                        "java" => "java",
                        "toml" => "toml",
                        "yaml" | "yml" => "yaml",
                        "json" => "json",
                        "md" => "markdown",
                        other => other,
                    })
                    .unwrap_or("");

                let truncated = lines.len() > max_lines;
                let display_lines = if truncated {
                    &lines[..max_lines]
                } else {
                    &lines[..]
                };

                output.push_str(&format!("### `{}`\n```{}\n", file_path, lang));
                for line in display_lines {
                    output.push_str(line);
                    output.push('\n');
                }
                output.push_str("```\n");
                if truncated {
                    output.push_str(&format!(
                        "_(truncated at {} lines, {} total)_\n",
                        max_lines,
                        lines.len()
                    ));
                }
                output.push('\n');
                files_read += 1;
            }
            Err(_) => {
                // File doesn't exist or can't be read — skip silently
                output.push_str(&format!(
                    "### `{}`\n_(file not found or unreadable)_\n\n",
                    file_path
                ));
                files_read += 1;
            }
        }
    }

    output
}

/// The task statement both prompt builders emit: a heading and a body.
///
/// The heading is the title when there is one, the description otherwise (the
/// historical behaviour, kept for tasks created without a title). The body then
/// carries the description — when it is not already the heading — and the
/// acceptance criteria.
///
/// The acceptance criteria were missing from BOTH prompt paths: an agent was
/// told what to do but never what "done" meant. Everything written there —
/// edge cases, accessibility, UX rules — was invisible to the one reader it
/// was written for.
pub(crate) fn render_task_statement(task: &TaskNode) -> (String, String) {
    let title = task
        .title
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty());

    let (heading, mut body) = match title {
        Some(title) => (title.to_string(), task.description.trim().to_string()),
        None => (task.description.trim().to_string(), String::new()),
    };

    let criteria: Vec<&str> = task
        .acceptance_criteria
        .iter()
        .map(|c| c.trim())
        .filter(|c| !c.is_empty())
        .collect();

    if !criteria.is_empty() {
        if !body.is_empty() {
            body.push_str("\n\n");
        }
        body.push_str("## Acceptance Criteria\n");
        body.push_str(
            "The task is done only when EVERY criterion below holds. \
             Check each one before reporting completion.\n",
        );
        for criterion in criteria {
            body.push_str(&format!("- [ ] {}\n", criterion));
        }
    }

    (heading, body)
}

/// Search text for a task: title and description together.
///
/// The description alone is often a paragraph of rationale; the title names the
/// thing being built, which is what similar code and past decisions match on.
fn task_search_query(task: &TaskNode) -> String {
    match task
        .title
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        Some(title) => format!("{} {}", title, task.description),
        None => task.description.clone(),
    }
}

/// Remove every repeated note id, keeping the most relevant copy.
///
/// Not `dedup_by_key`: that only drops CONSECUTIVE repeats, and after a sort by
/// relevance two copies of one note are not guaranteed to be neighbours.
fn dedupe_notes_by_id(notes: &mut Vec<ContextNote>) {
    notes.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut seen = std::collections::HashSet::new();
    notes.retain(|n| seen.insert(n.id));
}

/// Merge per-project code hits: best score first, one entry per path.
fn merge_code_references(mut refs: Vec<CodeReference>, limit: usize) -> Vec<CodeReference> {
    refs.sort_by(|a, b| {
        b.relevance
            .partial_cmp(&a.relevance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut seen = std::collections::HashSet::new();
    refs.retain(|r| seen.insert(r.path.clone()));
    refs.truncate(limit);
    refs
}

/// Interleave per-project decision lists (they carry no score to merge on),
/// dropping repeats, so each project gets a fair share of the slots.
fn interleave_decisions(per_project: Vec<Vec<DecisionNode>>, limit: usize) -> Vec<DecisionNode> {
    let mut iters: Vec<_> = per_project.into_iter().map(|v| v.into_iter()).collect();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    loop {
        let mut progressed = false;
        for it in iters.iter_mut() {
            if out.len() >= limit {
                return out;
            }
            if let Some(d) = it.next() {
                progressed = true;
                if seen.insert(d.id) {
                    out.push(d);
                }
            }
        }
        if !progressed {
            return out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::models::{
        ConstraintNode, ConstraintType, DecisionNode, DecisionStatus, StepNode, StepStatus,
        TaskNode, TaskStatus,
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
            interrupted_at: None,
            interrupted_reason: None,
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
            frustration_score: 0.0,
            execution_context: None,
            persona: None,
            prompt_cache: None,
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
            frustration_signals: None,
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
            frustration_signals: None,
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
                    execution_context: None,
                    persona: None,
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
                    execution_context: None,
                    persona: None,
                },
            ],
            constraints: vec![],
            decisions: vec![],
            target_files: vec![],
            similar_code: vec![],
            related_decisions: vec![],
            notes: vec![],
            frustration_signals: None,
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
                status: DecisionStatus::Accepted,
                embedding: None,
                embedding_model: None,
                scar_intensity: 0.0,
            }],
            target_files: vec![],
            similar_code: vec![],
            related_decisions: vec![],
            notes: vec![],
            frustration_signals: None,
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
            frustration_signals: None,
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
            frustration_signals: None,
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
            frustration_signals: None,
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
            frustration_signals: None,
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
            frustration_signals: None,
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

    // -----------------------------------------------------------------
    // Acceptance criteria reach the agent (regression: both prompt paths
    // used to render the description only)
    // -----------------------------------------------------------------

    fn context_for(task: TaskNode) -> AgentContext {
        AgentContext {
            task,
            steps: vec![],
            constraints: vec![],
            decisions: vec![],
            target_files: vec![],
            similar_code: vec![],
            related_decisions: vec![],
            notes: vec![],
            frustration_signals: None,
        }
    }

    fn mock_builder() -> ContextBuilder {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let search: Arc<dyn SearchStore> =
            Arc::new(crate::meilisearch::mock::MockSearchStore::new());
        ContextBuilder::new(
            graph.clone(),
            search.clone(),
            Arc::new(PlanManager::new(graph.clone(), search.clone())),
            Arc::new(NoteManager::new(graph, search)),
        )
    }

    #[test]
    fn task_statement_carries_title_description_and_criteria() {
        let mut task = create_test_task();
        task.acceptance_criteria = vec![
            "The command is shown in full, never truncated".into(),
            "Targets are at least 36 px".into(),
        ];
        let (heading, body) = render_task_statement(&task);

        assert_eq!(heading, "Test Task");
        assert!(body.starts_with("Implement a new feature"));
        assert!(body.contains("## Acceptance Criteria"));
        assert!(body.contains("- [ ] The command is shown in full, never truncated"));
        assert!(body.contains("- [ ] Targets are at least 36 px"));
    }

    #[test]
    fn task_without_title_keeps_description_as_heading_without_repeating_it() {
        let mut task = create_test_task();
        task.title = None;
        let (heading, body) = render_task_statement(&task);

        assert_eq!(heading, "Implement a new feature");
        assert!(
            !body.contains("Implement a new feature"),
            "description must not appear twice"
        );
        assert!(body.contains("- [ ] Tests pass"));
    }

    #[test]
    fn blank_title_and_blank_criteria_are_ignored() {
        let mut task = create_test_task();
        task.title = Some("   ".into());
        task.acceptance_criteria = vec!["  ".into(), String::new()];
        let (heading, body) = render_task_statement(&task);

        assert_eq!(heading, "Implement a new feature");
        assert!(
            !body.contains("Acceptance Criteria"),
            "no empty criteria section"
        );
    }

    #[test]
    fn generate_prompt_renders_acceptance_criteria() {
        let mut task = create_test_task();
        task.acceptance_criteria = vec!["Double tap is impossible".into()];
        let prompt = mock_builder().generate_prompt(&context_for(task));

        assert!(prompt.starts_with("# Task: Test Task\n"));
        assert!(prompt.contains("Implement a new feature"));
        assert!(prompt.contains("## Acceptance Criteria"));
        assert!(prompt.contains("- [ ] Double tap is impossible"));
    }

    #[test]
    fn both_prompt_paths_render_the_same_criteria() {
        // generate_prompt (runner fallback, get_prompt) and build_prompt_builder
        // (enriched path, prompt_cache) had drifted: neither rendered criteria.
        // They now share render_task_statement — this pins it.
        let mut task = create_test_task();
        task.acceptance_criteria = vec!["Region is named for screen readers".into()];
        let builder = mock_builder();
        let ctx = context_for(task);

        let plain = builder.generate_prompt(&ctx);
        let enriched = builder.build_prompt_builder(&ctx).build();

        for prompt in [&plain, &enriched] {
            assert!(prompt.contains("Test Task"));
            assert!(prompt.contains("## Acceptance Criteria"));
            assert!(prompt.contains("- [ ] Region is named for screen readers"));
        }
    }

    // -----------------------------------------------------------------
    // Note de-duplication (regression: dedup_by_key only drops neighbours)
    // -----------------------------------------------------------------

    #[test]
    fn non_adjacent_duplicate_notes_are_removed() {
        let a = create_test_context_note("context", "design ref A", "high", false);
        let b = create_test_context_note("rfc", "the RFC", "high", false);
        let mut a_again = a.clone();
        a_again.propagated = true;

        // [A, B, A] — both A copies score 1.0, so a stable sort keeps them apart,
        // which is exactly the case dedup_by_key missed.
        let mut notes = vec![a.clone(), b.clone(), a_again];
        dedupe_notes_by_id(&mut notes);

        assert_eq!(notes.len(), 2);
        assert_eq!(notes.iter().filter(|n| n.id == a.id).count(), 1);
        assert!(notes.iter().any(|n| n.id == b.id));
    }

    #[test]
    fn dedupe_keeps_the_most_relevant_copy() {
        let mut low = create_test_context_note("gotcha", "x", "high", false);
        low.relevance_score = 0.2;
        let mut high = low.clone();
        high.relevance_score = 0.9;

        let mut notes = vec![low, high];
        dedupe_notes_by_id(&mut notes);

        assert_eq!(notes.len(), 1);
        assert!((notes[0].relevance_score - 0.9).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------
    // Retrieval scoped to the plan's projects (regression: unscoped search
    // pulled code and decisions from every repository on the machine)
    // -----------------------------------------------------------------

    fn code_doc(
        path: &str,
        slug: &str,
        docstrings: &str,
    ) -> crate::meilisearch::indexes::CodeDocument {
        crate::meilisearch::indexes::CodeDocument {
            id: path.to_string(),
            path: path.to_string(),
            language: "rust".into(),
            symbols: vec![],
            docstrings: docstrings.to_string(),
            signatures: vec![],
            imports: vec![],
            project_id: Uuid::new_v4().to_string(),
            project_slug: slug.to_string(),
        }
    }

    #[test]
    fn code_references_are_merged_by_score_and_path() {
        let refs = vec![
            CodeReference {
                path: "a.rs".into(),
                snippet: "".into(),
                relevance: 0.4,
            },
            CodeReference {
                path: "b.rs".into(),
                snippet: "".into(),
                relevance: 0.9,
            },
            CodeReference {
                path: "a.rs".into(),
                snippet: "".into(),
                relevance: 0.7,
            },
            CodeReference {
                path: "c.rs".into(),
                snippet: "".into(),
                relevance: 0.1,
            },
        ];
        let merged = merge_code_references(refs, 2);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].path, "b.rs");
        assert_eq!(merged[1].path, "a.rs");
        assert!(
            (merged[1].relevance - 0.7).abs() < f32::EPSILON,
            "best copy kept"
        );
    }

    #[test]
    fn decisions_are_interleaved_so_every_project_gets_slots() {
        let mk = |d: &str| DecisionNode {
            id: Uuid::new_v4(),
            description: d.into(),
            rationale: String::new(),
            alternatives: vec![],
            chosen_option: None,
            decided_by: "agent".into(),
            decided_at: chrono::Utc::now(),
            status: DecisionStatus::Accepted,
            embedding: None,
            embedding_model: None,
            scar_intensity: 0.0,
        };
        let front = vec![mk("f1"), mk("f2"), mk("f3")];
        let back = vec![mk("b1"), mk("b2")];

        let out = interleave_decisions(vec![front, back], 3);
        let names: Vec<_> = out.iter().map(|d| d.description.as_str()).collect();
        assert_eq!(names, vec!["f1", "b1", "f2"]);
    }

    #[tokio::test]
    async fn similar_code_only_comes_from_the_plans_projects() {
        let builder = mock_builder();
        builder
            .meili
            .index_code(&code_doc(
                "front/card.tsx",
                "frontend",
                "attention card render",
            ))
            .await
            .unwrap();
        builder
            .meili
            .index_code(&code_doc(
                "obrain/solver.rs",
                "obrain",
                "attention card render",
            ))
            .await
            .unwrap();

        let hits = builder
            .search_similar_code("attention card", 5, &["frontend".to_string()])
            .await
            .unwrap();

        assert!(!hits.is_empty());
        assert!(
            hits.iter().all(|h| h.path.starts_with("front/")),
            "code from another project leaked into the task context: {:?}",
            hits.iter().map(|h| &h.path).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn a_plan_without_project_gets_no_retrieval_rather_than_everything() {
        let builder = mock_builder();
        builder
            .meili
            .index_code(&code_doc(
                "obrain/solver.rs",
                "obrain",
                "attention card render",
            ))
            .await
            .unwrap();

        let hits = builder
            .search_similar_code("attention card", 5, &[])
            .await
            .unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn plan_projects_include_linked_projects_not_only_the_primary() {
        // link_plan_to_project overwrites plan.project_id with the LAST project
        // linked; the membership is the HAS_PLAN edges. A backend task on a
        // frontend+backend plan must still search the backend.
        let graph = crate::neo4j::mock::MockGraphStore::new();
        let mk_project = |slug: &str| ProjectNode {
            id: Uuid::new_v4(),
            name: slug.into(),
            slug: slug.into(),
            root_path: String::new(),
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
        let front = mk_project("frontend");
        graph.create_project(&front).await.unwrap();

        let plan = PlanNode {
            id: Uuid::new_v4(),
            title: "p".into(),
            description: String::new(),
            status: PlanStatus::Draft,
            created_at: chrono::Utc::now(),
            created_by: "test".into(),
            priority: 1,
            project_id: None,
            execution_context: None,
            persona: None,
        };
        graph.create_plan(&plan).await.unwrap();
        graph.link_plan_to_project(plan.id, front.id).await.unwrap();

        let slugs = graph.list_plan_project_slugs(plan.id).await.unwrap();
        assert_eq!(slugs, vec!["frontend".to_string()]);

        let unknown = graph.list_plan_project_slugs(Uuid::new_v4()).await.unwrap();
        assert!(unknown.is_empty());
    }
}
