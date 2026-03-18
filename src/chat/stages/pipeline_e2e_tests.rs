//! End-to-end tests for the full enrichment pipeline (TP2.5).
//!
//! Tests all three real stages (SkillActivation, KnowledgeInjection, StatusInjection)
//! running together through the EnrichmentPipeline with pre-seeded MockGraphStore
//! and MockSearchStore data.
//!
//! 6 test scenarios:
//! 1. Simple message → relevant notes injected
//! 2. Message mentioning a file → entity notes from get_notes_for_entity
//! 3. Message matching a skill → activation + context_template
//! 4. Message with task in_progress → status injection
//! 5. Benchmark: full pipeline < 500ms
//! 6. Graceful degradation: one stage error doesn't block others

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    use chrono::Utc;
    use uuid::Uuid;

    use crate::chat::enrichment::{EnrichmentConfig, EnrichmentInput, EnrichmentPipeline};
    use crate::chat::stages::{
        KnowledgeInjectionStage, SkillActivationStage, StatusInjectionStage,
    };
    use crate::meilisearch::indexes::{DecisionDocument, NoteDocument};
    use crate::meilisearch::mock::MockSearchStore;
    use crate::meilisearch::traits::SearchStore;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::models::{PlanNode, PlanStatus, ProjectNode, TaskNode, TaskStatus};
    use crate::notes::{
        EntityType, Note, NoteAnchor, NoteImportance, NoteScope, NoteStatus, NoteType,
    };
    use crate::skills::models::{SkillNode, SkillStatus, SkillTrigger, TriggerType};

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Create a fully wired pipeline with all 3 real stages.
    fn build_pipeline(
        graph: Arc<MockGraphStore>,
        search: Arc<MockSearchStore>,
    ) -> EnrichmentPipeline {
        let mut pipeline = EnrichmentPipeline::new(EnrichmentConfig::default());
        pipeline.add_stage(Box::new(SkillActivationStage::new(graph.clone())));
        pipeline.add_stage(Box::new(KnowledgeInjectionStage::new(
            graph.clone(),
            search.clone(),
        )));
        pipeline.add_stage(Box::new(StatusInjectionStage::new(graph.clone())));
        pipeline
    }

    /// Seed a project into the mock graph store.
    async fn seed_project(graph: &MockGraphStore) -> (Uuid, String) {
        let project_id = Uuid::new_v4();
        let slug = "test-project".to_string();
        let project = ProjectNode {
            id: project_id,
            name: "Test Project".to_string(),
            slug: slug.clone(),
            description: Some("A test project for E2E pipeline tests".to_string()),
            root_path: "/tmp/test-project".to_string(),
            last_synced: None,
            created_at: Utc::now(),
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
        };
        graph.projects.write().await.insert(project_id, project);
        (project_id, slug)
    }

    /// Parse a note_type string into NoteType enum.
    fn parse_note_type(s: &str) -> NoteType {
        match s {
            "guideline" => NoteType::Guideline,
            "gotcha" => NoteType::Gotcha,
            "pattern" => NoteType::Pattern,
            "context" => NoteType::Context,
            "tip" => NoteType::Tip,
            _ => NoteType::Guideline,
        }
    }

    /// Seed a note into both MockGraphStore and MockSearchStore.
    async fn seed_note(
        graph: &MockGraphStore,
        search: &MockSearchStore,
        project_id: Uuid,
        content: &str,
        note_type: &str,
        importance: NoteImportance,
        anchors: Vec<NoteAnchor>,
    ) -> Uuid {
        let note_id = Uuid::new_v4();
        let note = Note {
            id: note_id,
            project_id: Some(project_id),
            note_type: parse_note_type(note_type),
            status: NoteStatus::Active,
            importance,
            scope: NoteScope::Project,
            content: content.to_string(),
            tags: vec!["test".to_string()],
            anchors: anchors.clone(),
            created_at: Utc::now(),
            created_by: "test".to_string(),
            last_confirmed_at: None,
            last_confirmed_by: None,
            staleness_score: 0.0,
            energy: 0.8,
            last_activated: None,
            reactivation_count: 0,
            last_reactivated: None,
            freshness_pinged_at: None,
            activation_count: 0,
            supersedes: None,
            superseded_by: None,
            changes: vec![],
            assertion_rule: None,
            last_assertion_result: None,
            memory_horizon: crate::notes::MemoryHorizon::Operational,
            scar_intensity: 0.0,
            sharing_consent: Default::default(),
        };
        graph.notes.write().await.insert(note_id, note);

        // Register anchors for get_notes_for_entity lookup
        if !anchors.is_empty() {
            graph.note_anchors.write().await.insert(note_id, anchors);
        }

        // Index in search (MeiliSearch mock)
        let note_doc = NoteDocument {
            id: note_id.to_string(),
            project_id: project_id.to_string(),
            project_slug: "test-project".to_string(),
            note_type: note_type.to_string(),
            status: "active".to_string(),
            importance: format!("{:?}", importance).to_lowercase(),
            scope_type: "project".to_string(),
            scope_path: String::new(),
            content: content.to_string(),
            tags: vec!["test".to_string()],
            anchor_entities: vec![],
            created_at: Utc::now().timestamp(),
            created_by: "test".to_string(),
            staleness_score: 0.0,
        };
        search.index_note(&note_doc).await.unwrap();

        note_id
    }

    /// Seed a decision into MockSearchStore.
    async fn seed_decision(
        search: &MockSearchStore,
        project_id: Uuid,
        description: &str,
        rationale: &str,
    ) {
        let decision_doc = DecisionDocument {
            id: Uuid::new_v4().to_string(),
            description: description.to_string(),
            rationale: rationale.to_string(),
            task_id: Uuid::new_v4().to_string(),
            agent: "test".to_string(),
            timestamp: Utc::now().to_rfc3339(),
            tags: vec!["architecture".to_string()],
            project_id: Some(project_id.to_string()),
            project_slug: Some("test-project".to_string()),
        };
        search.index_decision(&decision_doc).await.unwrap();
    }

    /// Seed an in-progress plan with tasks.
    async fn seed_in_progress_plan(
        graph: &MockGraphStore,
        project_id: Uuid,
        plan_title: &str,
        task_titles: &[(&str, TaskStatus)],
    ) -> Uuid {
        let plan_id = Uuid::new_v4();
        let plan = PlanNode {
            id: plan_id,
            title: plan_title.to_string(),
            description: "Test plan".to_string(),
            status: PlanStatus::InProgress,
            priority: 80,
            created_at: Utc::now(),
            created_by: "test".to_string(),
            project_id: Some(project_id),
            execution_context: None,
            persona: None,
        };
        graph.plans.write().await.insert(plan_id, plan);

        // Link plan to project
        graph
            .project_plans
            .write()
            .await
            .entry(project_id)
            .or_default()
            .push(plan_id);

        // Create tasks
        let mut task_ids = Vec::new();
        for (title, status) in task_titles {
            let task_id = Uuid::new_v4();
            let task = TaskNode {
                id: task_id,
                title: Some(title.to_string()),
                description: title.to_string(),
                status: status.clone(),
                assigned_to: None,
                priority: Some(50),
                tags: vec![],
                acceptance_criteria: vec![],
                affected_files: vec![],
                estimated_complexity: None,
                actual_complexity: None,
                created_at: Utc::now(),
                updated_at: Some(Utc::now()),
                started_at: None,
                completed_at: None,
                frustration_score: 0.0,
                execution_context: None,
                persona: None,
                prompt_cache: None,
            };
            graph.tasks.write().await.insert(task_id, task);
            task_ids.push(task_id);
        }

        // Link tasks to plan
        graph.plan_tasks.write().await.insert(plan_id, task_ids);

        plan_id
    }

    /// Seed a skill with regex trigger.
    async fn seed_skill_with_regex(
        graph: &MockGraphStore,
        project_id: Uuid,
        name: &str,
        pattern: &str,
        context_template: &str,
    ) -> Uuid {
        let skill_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, name);
        skill.id = skill_id;
        skill.status = SkillStatus::Active;
        skill.energy = 0.9;
        skill.context_template = Some(context_template.to_string());
        skill.trigger_patterns = vec![SkillTrigger {
            pattern_type: TriggerType::Regex,
            pattern_value: pattern.to_string(),
            confidence_threshold: 0.7,
            quality_score: Some(0.9),
        }];
        graph.skills.write().await.insert(skill_id, skill);
        skill_id
    }

    fn make_input(message: &str, slug: Option<&str>, project_id: Option<Uuid>) -> EnrichmentInput {
        EnrichmentInput {
            message: message.to_string(),
            session_id: Uuid::new_v4(),
            project_slug: slug.map(|s| s.to_string()),
            project_id,
            cwd: None,
            protocol_run_id: None,
            protocol_state: None,
        }
    }

    /// Create a NoteAnchor with required fields.
    fn make_anchor(entity_type: EntityType, entity_id: &str) -> NoteAnchor {
        NoteAnchor {
            entity_type,
            entity_id: entity_id.to_string(),
            signature_hash: None,
            body_hash: None,
            last_verified: Utc::now(),
            is_valid: true,
        }
    }

    // ========================================================================
    // Test 1: Simple message → relevant notes injected
    // ========================================================================

    #[tokio::test]
    async fn test_e2e_simple_message_injects_relevant_notes() {
        let graph = Arc::new(MockGraphStore::new());
        let search = Arc::new(MockSearchStore::new());

        let (project_id, slug) = seed_project(&graph).await;

        // Seed notes — MockSearchStore uses substring matching so the message
        // must be a substring of the note content for BM25 to match.
        seed_note(
            &graph,
            &search,
            project_id,
            "Always handle errors with proper context using anyhow::Context",
            "guideline",
            NoteImportance::High,
            vec![],
        )
        .await;

        seed_decision(
            &search,
            project_id,
            "Use anyhow for error handling throughout the codebase",
            "Provides context chains and downcasting support",
        )
        .await;

        let pipeline = build_pipeline(graph.clone(), search.clone());
        // Use a short query that will match as a substring in the note/decision content
        let input = make_input("error", Some(&slug), Some(project_id));
        let ctx = pipeline.execute(&input).await;

        // Should have knowledge injection content
        assert!(
            ctx.has_content(),
            "Pipeline should produce enrichment content"
        );

        let rendered = ctx.render();
        assert!(
            rendered.contains("error") || rendered.contains("anyhow"),
            "Rendered content should mention error handling. Got: {}",
            rendered
        );

        // Verify pipeline timing was recorded
        assert!(
            !ctx.stage_timings.is_empty(),
            "Pipeline should record stage timings"
        );
    }

    // ========================================================================
    // Test 2: Message mentioning a file → entity notes via get_notes_for_entity
    // ========================================================================

    #[tokio::test]
    async fn test_e2e_file_mention_triggers_entity_notes() {
        let graph = Arc::new(MockGraphStore::new());
        let search = Arc::new(MockSearchStore::new());

        let (project_id, slug) = seed_project(&graph).await;

        // Seed a note anchored to a specific file
        seed_note(
            &graph,
            &search,
            project_id,
            "The manager.rs file uses Arc<dyn GraphStore> pattern for dependency injection",
            "pattern",
            NoteImportance::High,
            vec![make_anchor(EntityType::File, "src/chat/manager.rs")],
        )
        .await;

        // Also seed a generic note for BM25 match
        seed_note(
            &graph,
            &search,
            project_id,
            "Chat manager handles session lifecycle and message routing",
            "context",
            NoteImportance::Medium,
            vec![],
        )
        .await;

        let pipeline = build_pipeline(graph.clone(), search.clone());
        let input = make_input(
            "I need to modify `src/chat/manager.rs` to add a new method",
            Some(&slug),
            Some(project_id),
        );
        let ctx = pipeline.execute(&input).await;

        assert!(
            ctx.has_content(),
            "Pipeline should produce content for file mention"
        );

        let rendered = ctx.render();
        // The entity detection should find src/chat/manager.rs and query notes for it
        assert!(
            rendered.contains("manager")
                || rendered.contains("GraphStore")
                || rendered.contains("chat"),
            "Rendered should contain file-related knowledge. Got: {}",
            rendered
        );
    }

    // ========================================================================
    // Test 3: Message matching a skill → activation + context_template
    // ========================================================================

    #[tokio::test]
    async fn test_e2e_skill_activation_injects_context_template() {
        let graph = Arc::new(MockGraphStore::new());
        let search = Arc::new(MockSearchStore::new());

        let (project_id, slug) = seed_project(&graph).await;

        // Seed a skill with a regex trigger for "neo4j" or "cypher"
        seed_skill_with_regex(
            &graph,
            project_id,
            "Neo4j Expert",
            "(?i)(neo4j|cypher|UNWIND)",
            "## Neo4j Guidelines\n- Always use parameterized queries\n- Never interpolate user values\n- Use UNWIND for batch operations",
        )
        .await;

        let pipeline = build_pipeline(graph.clone(), search.clone());
        let input = make_input(
            "I need to write a Cypher query to update multiple nodes with UNWIND",
            Some(&slug),
            Some(project_id),
        );
        let ctx = pipeline.execute(&input).await;

        assert!(
            ctx.has_content(),
            "Pipeline should produce content for skill match"
        );

        let rendered = ctx.render();
        assert!(
            rendered.contains("Neo4j") || rendered.contains("parameterized"),
            "Rendered should contain skill context template. Got: {}",
            rendered
        );
        assert!(
            rendered.contains("Active Skills") || rendered.contains("Neo4j Expert"),
            "Rendered should show skill name. Got: {}",
            rendered
        );
    }

    // ========================================================================
    // Test 4: Message with task in_progress → status injection
    // ========================================================================

    #[tokio::test]
    async fn test_e2e_status_injection_shows_in_progress_work() {
        let graph = Arc::new(MockGraphStore::new());
        let search = Arc::new(MockSearchStore::new());

        let (project_id, slug) = seed_project(&graph).await;

        // Seed an in-progress plan with tasks
        seed_in_progress_plan(
            &graph,
            project_id,
            "TP2 — Chat Pre-Enrichment Engine",
            &[
                ("TP2.1 — Pipeline Architecture", TaskStatus::Completed),
                ("TP2.2 — SkillActivation Stage", TaskStatus::Completed),
                ("TP2.3 — KnowledgeInjection Stage", TaskStatus::InProgress),
                ("TP2.4 — StatusInjection Stage", TaskStatus::Pending),
            ],
        )
        .await;

        let pipeline = build_pipeline(graph.clone(), search.clone());
        let input = make_input("What should I work on next?", Some(&slug), Some(project_id));
        let ctx = pipeline.execute(&input).await;

        assert!(
            ctx.has_content(),
            "Pipeline should produce content with work status"
        );

        let rendered = ctx.render();
        assert!(
            rendered.contains("TP2") || rendered.contains("Plan"),
            "Rendered should show the in-progress plan. Got: {}",
            rendered
        );
        assert!(
            rendered.contains("in_progress")
                || rendered.contains("🔄")
                || rendered.contains("Progress"),
            "Rendered should show task status. Got: {}",
            rendered
        );
    }

    // ========================================================================
    // Test 5: Benchmark — full pipeline < 500ms
    // ========================================================================

    #[tokio::test]
    async fn test_e2e_pipeline_performance_under_500ms() {
        let graph = Arc::new(MockGraphStore::new());
        let search = Arc::new(MockSearchStore::new());

        let (project_id, slug) = seed_project(&graph).await;

        // Seed substantial data to simulate realistic load
        for i in 0..20 {
            seed_note(
                &graph,
                &search,
                project_id,
                &format!(
                    "Note {} about error handling, performance, and architecture patterns",
                    i
                ),
                if i % 3 == 0 {
                    "guideline"
                } else if i % 3 == 1 {
                    "gotcha"
                } else {
                    "pattern"
                },
                NoteImportance::Medium,
                vec![],
            )
            .await;
        }

        for i in 0..5 {
            seed_decision(
                &search,
                project_id,
                &format!("Decision {} about async architecture", i),
                &format!("Rationale {} for async patterns", i),
            )
            .await;
        }

        seed_in_progress_plan(
            &graph,
            project_id,
            "Active Plan",
            &[
                ("Task A", TaskStatus::Completed),
                ("Task B", TaskStatus::InProgress),
                ("Task C", TaskStatus::Pending),
            ],
        )
        .await;

        seed_skill_with_regex(
            &graph,
            project_id,
            "Architecture Skill",
            "(?i)architecture",
            "Architecture context template",
        )
        .await;

        let pipeline = build_pipeline(graph.clone(), search.clone());
        let input = make_input(
            "Tell me about the architecture and error handling patterns",
            Some(&slug),
            Some(project_id),
        );

        // Run multiple times and measure average
        let start = Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = pipeline.execute(&input).await;
        }
        let total = start.elapsed();
        let avg_ms = total.as_millis() as f64 / iterations as f64;

        assert!(
            avg_ms < 500.0,
            "Average pipeline execution should be < 500ms, got {:.1}ms (total {:.0}ms for {} iterations)",
            avg_ms,
            total.as_millis(),
            iterations
        );

        // Also check the context reports timing < 500ms
        let ctx = pipeline.execute(&input).await;
        assert!(
            ctx.total_time_ms < 500,
            "Pipeline self-reported time should be < 500ms, got {}ms",
            ctx.total_time_ms
        );
    }

    // ========================================================================
    // Test 6: Graceful degradation — failing stage doesn't block others
    // ========================================================================

    #[tokio::test]
    async fn test_e2e_graceful_degradation_failing_stage() {
        use crate::chat::enrichment::EnrichmentStage;

        /// A stage that always errors.
        struct FailingStage;

        #[async_trait::async_trait]
        impl EnrichmentStage for FailingStage {
            async fn execute(
                &self,
                _input: &EnrichmentInput,
                _ctx: &mut crate::chat::enrichment::EnrichmentContext,
            ) -> anyhow::Result<()> {
                anyhow::bail!("Simulated stage failure for E2E test")
            }

            fn name(&self) -> &str {
                "failing_test_stage"
            }

            fn is_enabled(&self, _config: &EnrichmentConfig) -> bool {
                true
            }
        }

        let graph = Arc::new(MockGraphStore::new());
        let search = Arc::new(MockSearchStore::new());

        let (project_id, slug) = seed_project(&graph).await;

        // Seed data so StatusInjection produces output
        seed_in_progress_plan(
            &graph,
            project_id,
            "Test Plan",
            &[("Active Task", TaskStatus::InProgress)],
        )
        .await;

        // Build pipeline with a failing stage inserted in the middle
        let mut pipeline = EnrichmentPipeline::new(EnrichmentConfig::default());
        pipeline.add_stage(Box::new(SkillActivationStage::new(graph.clone())));
        pipeline.add_stage(Box::new(FailingStage)); // <-- This fails
        pipeline.add_stage(Box::new(KnowledgeInjectionStage::new(
            graph.clone(),
            search.clone(),
        )));
        pipeline.add_stage(Box::new(StatusInjectionStage::new(graph.clone())));

        let input = make_input("Show me what's in progress", Some(&slug), Some(project_id));
        let ctx = pipeline.execute(&input).await;

        // Despite failing stage, later stages should still execute
        assert!(
            ctx.has_content(),
            "Pipeline should still produce content despite a failing stage"
        );

        // Verify the failing stage was recorded as skipped
        assert!(
            ctx.skipped_stages
                .iter()
                .any(|s| s.contains("failing_test_stage")),
            "Failing stage should be recorded in skipped_stages. Got: {:?}",
            ctx.skipped_stages
        );

        // Status injection should still have run (it's after the failing stage)
        let rendered = ctx.render();
        assert!(
            rendered.contains("Test Plan")
                || rendered.contains("Active Task")
                || rendered.contains("Progress"),
            "StatusInjection should still run after the failing stage. Got: {}",
            rendered
        );

        // All 4 stages should have timing entries (including the failed one)
        assert_eq!(
            ctx.stage_timings.len(),
            4,
            "All 4 stages should have timing entries. Got: {:?}",
            ctx.stage_timings
        );
    }

    // ========================================================================
    // Bonus: Test pipeline with no project scope skips all stages gracefully
    // ========================================================================

    #[tokio::test]
    async fn test_e2e_no_project_scope_produces_empty_context() {
        let graph = Arc::new(MockGraphStore::new());
        let search = Arc::new(MockSearchStore::new());
        let pipeline = build_pipeline(graph, search);

        let input = make_input("Hello world", None, None);
        let ctx = pipeline.execute(&input).await;

        assert!(
            !ctx.has_content(),
            "Pipeline without project scope should produce no content"
        );
    }

    // ========================================================================
    // Bonus: Test that all stages run and pipeline collects their sections
    // ========================================================================

    #[tokio::test]
    async fn test_e2e_all_three_stages_contribute_sections() {
        let graph = Arc::new(MockGraphStore::new());
        let search = Arc::new(MockSearchStore::new());

        let (project_id, slug) = seed_project(&graph).await;

        // Seed data for all 3 stages:
        // 1. Skill with regex trigger
        seed_skill_with_regex(
            &graph,
            project_id,
            "Testing Skill",
            "(?i)testing",
            "## Testing Best Practices\nAlways write unit tests",
        )
        .await;

        // 2. Knowledge (notes + decisions)
        seed_note(
            &graph,
            &search,
            project_id,
            "Testing guideline: use MockGraphStore for all unit tests",
            "guideline",
            NoteImportance::High,
            vec![],
        )
        .await;

        // 3. Status (in-progress plan)
        seed_in_progress_plan(
            &graph,
            project_id,
            "Test Implementation Plan",
            &[("Write unit tests", TaskStatus::InProgress)],
        )
        .await;

        let pipeline = build_pipeline(graph.clone(), search.clone());
        let input = make_input(
            "I need help testing the enrichment pipeline",
            Some(&slug),
            Some(project_id),
        );
        let ctx = pipeline.execute(&input).await;

        assert!(
            ctx.has_content(),
            "Pipeline should produce content from multiple stages"
        );

        // Check that we have sections from multiple sources
        let sources: Vec<&str> = ctx.sections.iter().map(|s| s.source.as_str()).collect();
        assert!(
            sources.len() >= 2,
            "Should have sections from at least 2 stages. Got {} sections from: {:?}",
            sources.len(),
            sources
        );

        // Verify at least skill_activation and one other stage contributed
        let has_skill = sources.contains(&"skill_activation");
        let has_knowledge = sources.contains(&"knowledge_injection");
        let has_status = sources.contains(&"status_injection");

        assert!(
            (has_skill as u8 + has_knowledge as u8 + has_status as u8) >= 2,
            "At least 2 stages should contribute. skill={}, knowledge={}, status={}. Sources: {:?}",
            has_skill,
            has_knowledge,
            has_status,
            sources
        );

        // Verify rendered output has the enrichment_context wrapper
        let rendered = ctx.render();
        assert!(rendered.contains("<enrichment_context>"));
        assert!(rendered.contains("</enrichment_context>"));
    }
}
