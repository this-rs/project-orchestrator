//! Plan management operations

use super::models::*;
use crate::events::{CrudAction, CrudEvent, EntityType, EventBus};
use crate::meilisearch::indexes::DecisionDocument;
use crate::meilisearch::SearchStore;
use crate::neo4j::models::*;
use crate::neo4j::GraphStore;
use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;

/// Manager for plan operations
pub struct PlanManager {
    neo4j: Arc<dyn GraphStore>,
    meili: Arc<dyn SearchStore>,
    event_bus: Option<Arc<EventBus>>,
}

impl PlanManager {
    /// Create a new plan manager
    pub fn new(neo4j: Arc<dyn GraphStore>, meili: Arc<dyn SearchStore>) -> Self {
        Self {
            neo4j,
            meili,
            event_bus: None,
        }
    }

    /// Create a new plan manager with an event bus
    pub fn with_event_bus(
        neo4j: Arc<dyn GraphStore>,
        meili: Arc<dyn SearchStore>,
        event_bus: Arc<EventBus>,
    ) -> Self {
        Self {
            neo4j,
            meili,
            event_bus: Some(event_bus),
        }
    }

    /// Emit a CRUD event (no-op if event_bus is None)
    fn emit(&self, event: CrudEvent) {
        if let Some(bus) = &self.event_bus {
            bus.emit(event);
        }
    }

    // ========================================================================
    // Plan operations
    // ========================================================================

    /// Create a new plan
    pub async fn create_plan(&self, req: CreatePlanRequest, created_by: &str) -> Result<PlanNode> {
        let plan = if let Some(project_id) = req.project_id {
            PlanNode::new_for_project(
                req.title,
                req.description,
                created_by.to_string(),
                req.priority.unwrap_or(0),
                project_id,
            )
        } else {
            PlanNode::new(
                req.title,
                req.description,
                created_by.to_string(),
                req.priority.unwrap_or(0),
            )
        };

        self.neo4j.create_plan(&plan).await?;

        // Create constraints if provided
        if let Some(constraints) = req.constraints {
            for constraint_req in constraints {
                let constraint = ConstraintNode::new(
                    constraint_req.constraint_type,
                    constraint_req.description,
                    constraint_req.enforced_by,
                );
                self.add_constraint(plan.id, &constraint).await?;
            }
        }

        {
            let mut event =
                CrudEvent::new(EntityType::Plan, CrudAction::Created, plan.id.to_string())
                    .with_payload(serde_json::json!({"title": &plan.title}));
            if let Some(pid) = plan.project_id {
                event = event.with_project_id(pid.to_string());
            }
            self.emit(event);
        }

        Ok(plan)
    }

    /// Get a plan by ID
    pub async fn get_plan(&self, plan_id: Uuid) -> Result<Option<PlanNode>> {
        self.neo4j.get_plan(plan_id).await
    }

    /// List all active plans
    pub async fn list_active_plans(&self) -> Result<Vec<PlanNode>> {
        self.neo4j.list_active_plans().await
    }

    /// Update plan status
    pub async fn update_plan_status(&self, plan_id: Uuid, status: PlanStatus) -> Result<()> {
        self.neo4j
            .update_plan_status(plan_id, status.clone())
            .await?;
        self.emit(
            CrudEvent::new(EntityType::Plan, CrudAction::Updated, plan_id.to_string())
                .with_payload(serde_json::json!({"status": status})),
        );
        Ok(())
    }

    /// Delete a plan and all its related data
    pub async fn delete_plan(&self, plan_id: Uuid) -> Result<()> {
        self.neo4j.delete_plan(plan_id).await?;
        self.emit(CrudEvent::new(
            EntityType::Plan,
            CrudAction::Deleted,
            plan_id.to_string(),
        ));
        Ok(())
    }

    /// Get full plan details including tasks
    pub async fn get_plan_details(&self, plan_id: Uuid) -> Result<Option<PlanDetails>> {
        let plan = match self.neo4j.get_plan(plan_id).await? {
            Some(p) => p,
            None => return Ok(None),
        };

        let tasks = self.neo4j.get_plan_tasks(plan_id).await?;
        let mut task_details = Vec::new();

        for task in tasks {
            let details = self.get_task_details(task.id).await?;
            if let Some(d) = details {
                task_details.push(d);
            }
        }

        // Get constraints from Neo4j
        let constraints = self.neo4j.get_plan_constraints(plan_id).await?;

        Ok(Some(PlanDetails {
            plan,
            tasks: task_details,
            constraints,
        }))
    }

    // ========================================================================
    // Task operations
    // ========================================================================

    /// Add a task to a plan
    pub async fn add_task(&self, plan_id: Uuid, req: CreateTaskRequest) -> Result<TaskNode> {
        let task = TaskNode::new_full(
            req.title,
            req.description,
            req.priority,
            req.tags.unwrap_or_default(),
            req.acceptance_criteria.unwrap_or_default(),
            req.affected_files.unwrap_or_default(),
            req.estimated_complexity,
        );

        self.neo4j.create_task(plan_id, &task).await?;

        // Add dependencies
        if let Some(deps) = req.depends_on {
            for dep_id in deps {
                self.neo4j.add_task_dependency(task.id, dep_id).await?;
            }
        }

        // Add steps
        if let Some(steps) = req.steps {
            for (i, step_req) in steps.into_iter().enumerate() {
                let step = StepNode::new(i as u32, step_req.description, step_req.verification);
                self.add_step(task.id, &step).await?;
            }
        }

        self.emit(
            CrudEvent::new(EntityType::Task, CrudAction::Created, task.id.to_string())
                .with_payload(
                    serde_json::json!({"title": &task.title, "plan_id": plan_id.to_string()}),
                ),
        );

        Ok(task)
    }

    /// Get task details
    pub async fn get_task_details(&self, task_id: Uuid) -> Result<Option<TaskDetails>> {
        self.neo4j.get_task_with_full_details(task_id).await
    }

    /// Update task fields
    pub async fn update_task(&self, task_id: Uuid, req: UpdateTaskRequest) -> Result<()> {
        // Handle status change separately (has side effects like timestamps)
        if let Some(status) = req.status.clone() {
            self.neo4j.update_task_status(task_id, status).await?;
        }

        // Update all other fields via the full update method
        self.neo4j.update_task(task_id, &req).await?;

        self.emit(
            CrudEvent::new(EntityType::Task, CrudAction::Updated, task_id.to_string())
                .with_payload(serde_json::to_value(&req).unwrap_or_default()),
        );

        Ok(())
    }

    /// Delete a task and all its related data (steps, decisions)
    pub async fn delete_task(&self, task_id: Uuid) -> Result<()> {
        self.neo4j.delete_task(task_id).await?;
        self.emit(CrudEvent::new(
            EntityType::Task,
            CrudAction::Deleted,
            task_id.to_string(),
        ));
        Ok(())
    }

    /// Get next available task from a plan
    pub async fn get_next_available_task(&self, plan_id: Uuid) -> Result<Option<TaskNode>> {
        self.neo4j.get_next_available_task(plan_id).await
    }

    /// Link task to files it modifies
    pub async fn link_task_to_files(&self, task_id: Uuid, files: &[String]) -> Result<()> {
        self.neo4j.link_task_to_files(task_id, files).await
    }

    // ========================================================================
    // Step operations
    // ========================================================================

    /// Add a step to a task
    pub async fn add_step(&self, task_id: Uuid, step: &StepNode) -> Result<()> {
        self.neo4j.create_step(task_id, step).await?;
        self.emit(
            CrudEvent::new(EntityType::Step, CrudAction::Created, step.id.to_string())
                .with_payload(
                    serde_json::json!({"task_id": task_id.to_string(), "order": step.order}),
                ),
        );
        Ok(())
    }

    /// Update step status
    pub async fn update_step_status(&self, step_id: Uuid, status: StepStatus) -> Result<()> {
        self.neo4j
            .update_step_status(step_id, status.clone())
            .await?;
        self.emit(
            CrudEvent::new(EntityType::Step, CrudAction::Updated, step_id.to_string())
                .with_payload(serde_json::json!({"status": status})),
        );
        Ok(())
    }

    // ========================================================================
    // Decision operations
    // ========================================================================

    /// Record a decision for a task
    pub async fn add_decision(
        &self,
        task_id: Uuid,
        req: CreateDecisionRequest,
        decided_by: &str,
    ) -> Result<DecisionNode> {
        let decision = DecisionNode {
            id: Uuid::new_v4(),
            description: req.description.clone(),
            rationale: req.rationale.clone(),
            alternatives: req.alternatives.unwrap_or_default(),
            chosen_option: req.chosen_option.clone(),
            decided_by: decided_by.to_string(),
            decided_at: chrono::Utc::now(),
        };

        self.neo4j.create_decision(task_id, &decision).await?;

        // Index in Meilisearch for search
        // TODO: Get project_id and project_slug from the task's plan
        let doc = DecisionDocument {
            id: decision.id.to_string(),
            description: decision.description.clone(),
            rationale: decision.rationale.clone(),
            task_id: task_id.to_string(),
            agent: decided_by.to_string(),
            timestamp: decision.decided_at.to_rfc3339(),
            tags: vec![],
            project_id: None,
            project_slug: None,
        };
        self.meili.index_decision(&doc).await?;

        self.emit(
            CrudEvent::new(EntityType::Decision, CrudAction::Created, decision.id.to_string())
                .with_payload(serde_json::json!({"task_id": task_id.to_string(), "description": &decision.description})),
        );

        Ok(decision)
    }

    /// Search for related decisions
    pub async fn search_decisions(&self, query: &str, limit: usize) -> Result<Vec<DecisionNode>> {
        let docs = self.meili.search_decisions(query, limit).await?;

        // Convert documents to nodes
        let decisions = docs
            .into_iter()
            .map(|doc| DecisionNode {
                id: doc.id.parse().unwrap_or_else(|_| Uuid::new_v4()),
                description: doc.description,
                rationale: doc.rationale,
                alternatives: vec![],
                chosen_option: None,
                decided_by: doc.agent,
                decided_at: doc.timestamp.parse().unwrap_or_else(|_| chrono::Utc::now()),
            })
            .collect();

        Ok(decisions)
    }

    // ========================================================================
    // Constraint operations
    // ========================================================================

    /// Add a constraint to a plan
    pub async fn add_constraint(&self, plan_id: Uuid, constraint: &ConstraintNode) -> Result<()> {
        self.neo4j.create_constraint(plan_id, constraint).await?;
        self.emit(
            CrudEvent::new(EntityType::Constraint, CrudAction::Created, constraint.id.to_string())
                .with_payload(serde_json::json!({"plan_id": plan_id.to_string(), "type": constraint.constraint_type})),
        );
        Ok(())
    }

    // ========================================================================
    // Impact analysis
    // ========================================================================

    /// Analyze the impact of a task on the codebase
    pub async fn analyze_task_impact(&self, task_id: Uuid) -> Result<Vec<String>> {
        self.neo4j.analyze_task_impact(task_id).await
    }

    /// Find blocked tasks in a plan
    pub async fn find_blocked_tasks(
        &self,
        plan_id: Uuid,
    ) -> Result<Vec<(TaskNode, Vec<TaskNode>)>> {
        self.neo4j.find_blocked_tasks(plan_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;

    fn create_plan_manager() -> PlanManager {
        let state = mock_app_state();
        PlanManager::new(state.neo4j.clone(), state.meili.clone())
    }

    // =========================================================================
    // Plan CRUD
    // =========================================================================

    #[tokio::test]
    async fn test_create_plan() {
        let pm = create_plan_manager();
        let req = CreatePlanRequest {
            title: "Test Plan".to_string(),
            description: "A plan for testing".to_string(),
            project_id: None,
            priority: Some(5),
            constraints: None,
        };

        let plan = pm.create_plan(req, "test-agent").await.unwrap();
        assert_eq!(plan.title, "Test Plan");
        assert_eq!(plan.description, "A plan for testing");
        assert_eq!(plan.created_by, "test-agent");
        assert_eq!(plan.priority, 5);
        assert_eq!(plan.status, PlanStatus::Draft);
        assert!(plan.project_id.is_none());

        // Verify it can be fetched back
        let fetched = pm.get_plan(plan.id).await.unwrap();
        assert!(fetched.is_some());
        let fetched = fetched.unwrap();
        assert_eq!(fetched.id, plan.id);
        assert_eq!(fetched.title, "Test Plan");
    }

    #[tokio::test]
    async fn test_create_plan_with_constraints() {
        let pm = create_plan_manager();
        let req = CreatePlanRequest {
            title: "Constrained Plan".to_string(),
            description: "Plan with constraints".to_string(),
            project_id: None,
            priority: Some(8),
            constraints: Some(vec![
                CreateConstraintRequest {
                    constraint_type: ConstraintType::Performance,
                    description: "Response time under 100ms".to_string(),
                    enforced_by: Some("benchmark".to_string()),
                },
                CreateConstraintRequest {
                    constraint_type: ConstraintType::Security,
                    description: "Sanitize all user input".to_string(),
                    enforced_by: None,
                },
            ]),
        };

        let plan = pm.create_plan(req, "architect").await.unwrap();
        assert_eq!(plan.title, "Constrained Plan");

        // Verify constraints were stored via get_plan_details
        let details = pm.get_plan_details(plan.id).await.unwrap().unwrap();
        assert_eq!(
            details.constraints.len(),
            2,
            "Expected 2 constraints to be stored"
        );

        let types: Vec<ConstraintType> = details
            .constraints
            .iter()
            .map(|c| c.constraint_type.clone())
            .collect();
        assert!(types.contains(&ConstraintType::Performance));
        assert!(types.contains(&ConstraintType::Security));
    }

    #[tokio::test]
    async fn test_create_plan_for_project() {
        let pm = create_plan_manager();
        let project_id = Uuid::new_v4();

        let req = CreatePlanRequest {
            title: "Project Plan".to_string(),
            description: "Linked to a project".to_string(),
            project_id: Some(project_id),
            priority: Some(3),
            constraints: None,
        };

        let plan = pm.create_plan(req, "agent").await.unwrap();
        assert_eq!(plan.project_id, Some(project_id));

        let fetched = pm.get_plan(plan.id).await.unwrap().unwrap();
        assert_eq!(fetched.project_id, Some(project_id));
    }

    #[tokio::test]
    async fn test_get_plan_not_found() {
        let pm = create_plan_manager();
        let nonexistent_id = Uuid::new_v4();

        let result = pm.get_plan(nonexistent_id).await.unwrap();
        assert!(result.is_none(), "Expected None for non-existent plan ID");
    }

    #[tokio::test]
    async fn test_list_active_plans() {
        let pm = create_plan_manager();

        // Create several plans
        for i in 0..3 {
            let req = CreatePlanRequest {
                title: format!("Plan {}", i),
                description: format!("Description {}", i),
                project_id: None,
                priority: Some(i),
                constraints: None,
            };
            pm.create_plan(req, "agent").await.unwrap();
        }

        let plans = pm.list_active_plans().await.unwrap();
        assert_eq!(
            plans.len(),
            3,
            "Expected all 3 plans to be listed as active (Draft status)"
        );

        // All should be Draft
        for plan in &plans {
            assert_eq!(plan.status, PlanStatus::Draft);
        }
    }

    #[tokio::test]
    async fn test_update_plan_status() {
        let pm = create_plan_manager();
        let req = CreatePlanRequest {
            title: "Status Test".to_string(),
            description: "Will update status".to_string(),
            project_id: None,
            priority: Some(1),
            constraints: None,
        };
        let plan = pm.create_plan(req, "agent").await.unwrap();

        // Update to InProgress
        pm.update_plan_status(plan.id, PlanStatus::InProgress)
            .await
            .unwrap();

        let fetched = pm.get_plan(plan.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, PlanStatus::InProgress);

        // Update to Completed
        pm.update_plan_status(plan.id, PlanStatus::Completed)
            .await
            .unwrap();

        let fetched = pm.get_plan(plan.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, PlanStatus::Completed);

        // Completed plans should not appear in list_active_plans
        let active = pm.list_active_plans().await.unwrap();
        assert!(
            active.iter().all(|p| p.id != plan.id),
            "Completed plan should not appear in active plans"
        );
    }

    #[tokio::test]
    async fn test_delete_plan() {
        let pm = create_plan_manager();
        let req = CreatePlanRequest {
            title: "Delete Me".to_string(),
            description: "Will be deleted".to_string(),
            project_id: None,
            priority: Some(1),
            constraints: None,
        };
        let plan = pm.create_plan(req, "agent").await.unwrap();

        // Verify it exists
        assert!(pm.get_plan(plan.id).await.unwrap().is_some());

        // Delete it
        pm.delete_plan(plan.id).await.unwrap();

        // Verify it's gone
        assert!(
            pm.get_plan(plan.id).await.unwrap().is_none(),
            "Plan should be deleted"
        );
    }

    // =========================================================================
    // Task CRUD
    // =========================================================================

    #[tokio::test]
    async fn test_add_task() {
        let pm = create_plan_manager();
        let plan = pm
            .create_plan(
                CreatePlanRequest {
                    title: "Plan".to_string(),
                    description: "Desc".to_string(),
                    project_id: None,
                    priority: Some(1),
                    constraints: None,
                },
                "agent",
            )
            .await
            .unwrap();

        let task_req = CreateTaskRequest {
            title: Some("Implement feature".to_string()),
            description: "Add the new endpoint".to_string(),
            priority: Some(7),
            tags: Some(vec!["backend".to_string(), "api".to_string()]),
            acceptance_criteria: Some(vec!["Tests pass".to_string()]),
            affected_files: Some(vec!["src/api/routes.rs".to_string()]),
            depends_on: None,
            steps: None,
            estimated_complexity: Some(4),
        };

        let task = pm.add_task(plan.id, task_req).await.unwrap();
        assert_eq!(task.title, Some("Implement feature".to_string()));
        assert_eq!(task.description, "Add the new endpoint");
        assert_eq!(task.priority, Some(7));
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.tags, vec!["backend", "api"]);
        assert_eq!(task.acceptance_criteria, vec!["Tests pass"]);
        assert_eq!(task.affected_files, vec!["src/api/routes.rs"]);
        assert_eq!(task.estimated_complexity, Some(4));

        // Verify task appears in plan details
        let details = pm.get_plan_details(plan.id).await.unwrap().unwrap();
        assert_eq!(details.tasks.len(), 1);
        assert_eq!(details.tasks[0].task.id, task.id);
    }

    #[tokio::test]
    async fn test_add_task_with_steps() {
        let pm = create_plan_manager();
        let plan = pm
            .create_plan(
                CreatePlanRequest {
                    title: "Plan".to_string(),
                    description: "Desc".to_string(),
                    project_id: None,
                    priority: Some(1),
                    constraints: None,
                },
                "agent",
            )
            .await
            .unwrap();

        let task_req = CreateTaskRequest {
            title: Some("Multi-step task".to_string()),
            description: "Task with steps".to_string(),
            priority: None,
            tags: None,
            acceptance_criteria: None,
            affected_files: None,
            depends_on: None,
            steps: Some(vec![
                CreateStepRequest {
                    description: "Write the code".to_string(),
                    verification: Some("cargo check passes".to_string()),
                },
                CreateStepRequest {
                    description: "Write tests".to_string(),
                    verification: Some("cargo test passes".to_string()),
                },
                CreateStepRequest {
                    description: "Update docs".to_string(),
                    verification: None,
                },
            ]),
            estimated_complexity: None,
        };

        let task = pm.add_task(plan.id, task_req).await.unwrap();

        // Verify steps via task details
        let details = pm.get_task_details(task.id).await.unwrap().unwrap();
        assert_eq!(details.steps.len(), 3, "Expected 3 steps");

        // Steps should be ordered
        assert_eq!(details.steps[0].order, 0);
        assert_eq!(details.steps[0].description, "Write the code");
        assert_eq!(
            details.steps[0].verification,
            Some("cargo check passes".to_string())
        );

        assert_eq!(details.steps[1].order, 1);
        assert_eq!(details.steps[1].description, "Write tests");

        assert_eq!(details.steps[2].order, 2);
        assert!(details.steps[2].verification.is_none());

        // All steps should be pending
        for step in &details.steps {
            assert_eq!(step.status, StepStatus::Pending);
        }
    }

    #[tokio::test]
    async fn test_add_task_with_dependencies() {
        let pm = create_plan_manager();
        let plan = pm
            .create_plan(
                CreatePlanRequest {
                    title: "Plan".to_string(),
                    description: "Desc".to_string(),
                    project_id: None,
                    priority: Some(1),
                    constraints: None,
                },
                "agent",
            )
            .await
            .unwrap();

        // Create task A (no deps)
        let task_a = pm
            .add_task(
                plan.id,
                CreateTaskRequest {
                    title: Some("Task A".to_string()),
                    description: "First task".to_string(),
                    priority: None,
                    tags: None,
                    acceptance_criteria: None,
                    affected_files: None,
                    depends_on: None,
                    steps: None,
                    estimated_complexity: None,
                },
            )
            .await
            .unwrap();

        // Create task B that depends on A
        let task_b = pm
            .add_task(
                plan.id,
                CreateTaskRequest {
                    title: Some("Task B".to_string()),
                    description: "Depends on A".to_string(),
                    priority: None,
                    tags: None,
                    acceptance_criteria: None,
                    affected_files: None,
                    depends_on: Some(vec![task_a.id]),
                    steps: None,
                    estimated_complexity: None,
                },
            )
            .await
            .unwrap();

        // Verify dependency via task details
        let details_b = pm.get_task_details(task_b.id).await.unwrap().unwrap();
        assert!(
            details_b.depends_on.contains(&task_a.id),
            "Task B should depend on Task A"
        );

        // Task A should have no dependencies
        let details_a = pm.get_task_details(task_a.id).await.unwrap().unwrap();
        assert!(
            details_a.depends_on.is_empty(),
            "Task A should have no dependencies"
        );
    }

    #[tokio::test]
    async fn test_get_task_details() {
        let pm = create_plan_manager();
        let plan = pm
            .create_plan(
                CreatePlanRequest {
                    title: "Plan".to_string(),
                    description: "Desc".to_string(),
                    project_id: None,
                    priority: Some(1),
                    constraints: None,
                },
                "agent",
            )
            .await
            .unwrap();

        let task = pm
            .add_task(
                plan.id,
                CreateTaskRequest {
                    title: Some("Detailed Task".to_string()),
                    description: "Full details".to_string(),
                    priority: Some(5),
                    tags: Some(vec!["feature".to_string()]),
                    acceptance_criteria: Some(vec!["Works correctly".to_string()]),
                    affected_files: None,
                    depends_on: None,
                    steps: Some(vec![CreateStepRequest {
                        description: "Step 1".to_string(),
                        verification: None,
                    }]),
                    estimated_complexity: Some(3),
                },
            )
            .await
            .unwrap();

        // Add a decision to the task
        pm.add_decision(
            task.id,
            CreateDecisionRequest {
                description: "Use REST over gRPC".to_string(),
                rationale: "Simpler for frontend integration".to_string(),
                alternatives: Some(vec!["gRPC".to_string(), "GraphQL".to_string()]),
                chosen_option: Some("REST".to_string()),
            },
            "architect",
        )
        .await
        .unwrap();

        let details = pm.get_task_details(task.id).await.unwrap().unwrap();
        assert_eq!(details.task.title, Some("Detailed Task".to_string()));
        assert_eq!(details.steps.len(), 1);
        assert_eq!(details.decisions.len(), 1);
        assert_eq!(details.decisions[0].description, "Use REST over gRPC");
        assert_eq!(
            details.decisions[0].rationale,
            "Simpler for frontend integration"
        );
        assert_eq!(details.decisions[0].chosen_option, Some("REST".to_string()));
        assert!(details.depends_on.is_empty());
    }

    #[tokio::test]
    async fn test_get_task_details_not_found() {
        let pm = create_plan_manager();
        let result = pm.get_task_details(Uuid::new_v4()).await.unwrap();
        assert!(result.is_none(), "Expected None for non-existent task");
    }

    #[tokio::test]
    async fn test_update_task() {
        let pm = create_plan_manager();
        let plan = pm
            .create_plan(
                CreatePlanRequest {
                    title: "Plan".to_string(),
                    description: "Desc".to_string(),
                    project_id: None,
                    priority: Some(1),
                    constraints: None,
                },
                "agent",
            )
            .await
            .unwrap();

        let task = pm
            .add_task(
                plan.id,
                CreateTaskRequest {
                    title: Some("Original Title".to_string()),
                    description: "Original description".to_string(),
                    priority: Some(3),
                    tags: None,
                    acceptance_criteria: None,
                    affected_files: None,
                    depends_on: None,
                    steps: None,
                    estimated_complexity: None,
                },
            )
            .await
            .unwrap();

        // Update multiple fields
        pm.update_task(
            task.id,
            UpdateTaskRequest {
                title: Some("Updated Title".to_string()),
                status: Some(TaskStatus::InProgress),
                assigned_to: Some("agent-1".to_string()),
                priority: Some(9),
                tags: Some(vec!["urgent".to_string()]),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let details = pm.get_task_details(task.id).await.unwrap().unwrap();
        assert_eq!(details.task.title, Some("Updated Title".to_string()));
        assert_eq!(details.task.status, TaskStatus::InProgress);
        assert_eq!(details.task.assigned_to, Some("agent-1".to_string()));
        assert_eq!(details.task.priority, Some(9));
        assert_eq!(details.task.tags, vec!["urgent"]);
        assert!(
            details.task.started_at.is_some(),
            "started_at should be set when transitioning to InProgress"
        );
    }

    #[tokio::test]
    async fn test_delete_task() {
        let pm = create_plan_manager();
        let plan = pm
            .create_plan(
                CreatePlanRequest {
                    title: "Plan".to_string(),
                    description: "Desc".to_string(),
                    project_id: None,
                    priority: Some(1),
                    constraints: None,
                },
                "agent",
            )
            .await
            .unwrap();

        let task = pm
            .add_task(
                plan.id,
                CreateTaskRequest {
                    title: Some("Doomed Task".to_string()),
                    description: "Will be deleted".to_string(),
                    priority: None,
                    tags: None,
                    acceptance_criteria: None,
                    affected_files: None,
                    depends_on: None,
                    steps: Some(vec![CreateStepRequest {
                        description: "Step".to_string(),
                        verification: None,
                    }]),
                    estimated_complexity: None,
                },
            )
            .await
            .unwrap();

        // Verify it exists
        assert!(pm.get_task_details(task.id).await.unwrap().is_some());

        // Delete it
        pm.delete_task(task.id).await.unwrap();

        // Verify it's gone
        assert!(
            pm.get_task_details(task.id).await.unwrap().is_none(),
            "Task should be deleted"
        );
    }

    // =========================================================================
    // Task Dependencies & Ordering
    // =========================================================================

    #[tokio::test]
    async fn test_get_next_available_task() {
        let pm = create_plan_manager();
        let plan = pm
            .create_plan(
                CreatePlanRequest {
                    title: "Plan".to_string(),
                    description: "Desc".to_string(),
                    project_id: None,
                    priority: Some(1),
                    constraints: None,
                },
                "agent",
            )
            .await
            .unwrap();

        // Task A: no dependencies
        let task_a = pm
            .add_task(
                plan.id,
                CreateTaskRequest {
                    title: Some("Task A".to_string()),
                    description: "No deps".to_string(),
                    priority: None,
                    tags: None,
                    acceptance_criteria: None,
                    affected_files: None,
                    depends_on: None,
                    steps: None,
                    estimated_complexity: None,
                },
            )
            .await
            .unwrap();

        // Task B: depends on A
        let _task_b = pm
            .add_task(
                plan.id,
                CreateTaskRequest {
                    title: Some("Task B".to_string()),
                    description: "Depends on A".to_string(),
                    priority: None,
                    tags: None,
                    acceptance_criteria: None,
                    affected_files: None,
                    depends_on: Some(vec![task_a.id]),
                    steps: None,
                    estimated_complexity: None,
                },
            )
            .await
            .unwrap();

        // Next available should be Task A (B is blocked)
        let next = pm.get_next_available_task(plan.id).await.unwrap();
        assert!(next.is_some());
        let next = next.unwrap();
        assert_eq!(
            next.id, task_a.id,
            "Task A should be the next available task since it has no deps"
        );
    }

    #[tokio::test]
    async fn test_get_next_available_task_after_completion() {
        let pm = create_plan_manager();
        let plan = pm
            .create_plan(
                CreatePlanRequest {
                    title: "Plan".to_string(),
                    description: "Desc".to_string(),
                    project_id: None,
                    priority: Some(1),
                    constraints: None,
                },
                "agent",
            )
            .await
            .unwrap();

        // Task A: no deps
        let task_a = pm
            .add_task(
                plan.id,
                CreateTaskRequest {
                    title: Some("Task A".to_string()),
                    description: "First".to_string(),
                    priority: None,
                    tags: None,
                    acceptance_criteria: None,
                    affected_files: None,
                    depends_on: None,
                    steps: None,
                    estimated_complexity: None,
                },
            )
            .await
            .unwrap();

        // Task B: depends on A
        let task_b = pm
            .add_task(
                plan.id,
                CreateTaskRequest {
                    title: Some("Task B".to_string()),
                    description: "Second".to_string(),
                    priority: None,
                    tags: None,
                    acceptance_criteria: None,
                    affected_files: None,
                    depends_on: Some(vec![task_a.id]),
                    steps: None,
                    estimated_complexity: None,
                },
            )
            .await
            .unwrap();

        // Complete Task A
        pm.update_task(
            task_a.id,
            UpdateTaskRequest {
                status: Some(TaskStatus::Completed),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        // Now Task B should be available
        let next = pm.get_next_available_task(plan.id).await.unwrap();
        assert!(next.is_some());
        assert_eq!(
            next.unwrap().id,
            task_b.id,
            "Task B should become available after Task A is completed"
        );
    }

    // =========================================================================
    // Decisions
    // =========================================================================

    #[tokio::test]
    async fn test_add_decision() {
        let pm = create_plan_manager();
        let plan = pm
            .create_plan(
                CreatePlanRequest {
                    title: "Plan".to_string(),
                    description: "Desc".to_string(),
                    project_id: None,
                    priority: Some(1),
                    constraints: None,
                },
                "agent",
            )
            .await
            .unwrap();

        let task = pm
            .add_task(
                plan.id,
                CreateTaskRequest {
                    title: Some("Task".to_string()),
                    description: "Task desc".to_string(),
                    priority: None,
                    tags: None,
                    acceptance_criteria: None,
                    affected_files: None,
                    depends_on: None,
                    steps: None,
                    estimated_complexity: None,
                },
            )
            .await
            .unwrap();

        let decision = pm
            .add_decision(
                task.id,
                CreateDecisionRequest {
                    description: "Use Tokio runtime".to_string(),
                    rationale: "Best async runtime for Rust".to_string(),
                    alternatives: Some(vec!["async-std".to_string(), "smol".to_string()]),
                    chosen_option: Some("tokio".to_string()),
                },
                "engineer",
            )
            .await
            .unwrap();

        assert_eq!(decision.description, "Use Tokio runtime");
        assert_eq!(decision.rationale, "Best async runtime for Rust");
        assert_eq!(decision.alternatives, vec!["async-std", "smol"]);
        assert_eq!(decision.chosen_option, Some("tokio".to_string()));
        assert_eq!(decision.decided_by, "engineer");

        // Verify it appears in task details
        let details = pm.get_task_details(task.id).await.unwrap().unwrap();
        assert_eq!(details.decisions.len(), 1);
        assert_eq!(details.decisions[0].id, decision.id);
    }

    #[tokio::test]
    async fn test_search_decisions() {
        let pm = create_plan_manager();
        let plan = pm
            .create_plan(
                CreatePlanRequest {
                    title: "Plan".to_string(),
                    description: "Desc".to_string(),
                    project_id: None,
                    priority: Some(1),
                    constraints: None,
                },
                "agent",
            )
            .await
            .unwrap();

        let task = pm
            .add_task(
                plan.id,
                CreateTaskRequest {
                    title: Some("Task".to_string()),
                    description: "Task desc".to_string(),
                    priority: None,
                    tags: None,
                    acceptance_criteria: None,
                    affected_files: None,
                    depends_on: None,
                    steps: None,
                    estimated_complexity: None,
                },
            )
            .await
            .unwrap();

        // Add decisions
        pm.add_decision(
            task.id,
            CreateDecisionRequest {
                description: "Adopt microservices architecture".to_string(),
                rationale: "Better scalability".to_string(),
                alternatives: None,
                chosen_option: None,
            },
            "architect",
        )
        .await
        .unwrap();

        pm.add_decision(
            task.id,
            CreateDecisionRequest {
                description: "Use PostgreSQL database".to_string(),
                rationale: "Strong ACID compliance".to_string(),
                alternatives: None,
                chosen_option: None,
            },
            "architect",
        )
        .await
        .unwrap();

        // Search for "microservices" should find first decision
        let results = pm.search_decisions("microservices", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].description, "Adopt microservices architecture");

        // Search for "database" should find second decision (via description)
        let results = pm.search_decisions("database", 10).await.unwrap();
        assert_eq!(results.len(), 1);

        // Search for something not present
        let results = pm.search_decisions("nonexistent", 10).await.unwrap();
        assert!(results.is_empty());
    }

    // =========================================================================
    // Plan Details
    // =========================================================================

    #[tokio::test]
    async fn test_get_plan_details() {
        let pm = create_plan_manager();

        // Create plan with constraints
        let plan = pm
            .create_plan(
                CreatePlanRequest {
                    title: "Full Plan".to_string(),
                    description: "Plan with everything".to_string(),
                    project_id: None,
                    priority: Some(8),
                    constraints: Some(vec![CreateConstraintRequest {
                        constraint_type: ConstraintType::Style,
                        description: "Follow Rust conventions".to_string(),
                        enforced_by: Some("rustfmt".to_string()),
                    }]),
                },
                "architect",
            )
            .await
            .unwrap();

        // Add tasks
        let task1 = pm
            .add_task(
                plan.id,
                CreateTaskRequest {
                    title: Some("Task 1".to_string()),
                    description: "First task".to_string(),
                    priority: Some(5),
                    tags: None,
                    acceptance_criteria: None,
                    affected_files: None,
                    depends_on: None,
                    steps: Some(vec![CreateStepRequest {
                        description: "Step 1".to_string(),
                        verification: None,
                    }]),
                    estimated_complexity: None,
                },
            )
            .await
            .unwrap();

        let _task2 = pm
            .add_task(
                plan.id,
                CreateTaskRequest {
                    title: Some("Task 2".to_string()),
                    description: "Second task".to_string(),
                    priority: Some(3),
                    tags: None,
                    acceptance_criteria: None,
                    affected_files: None,
                    depends_on: Some(vec![task1.id]),
                    steps: None,
                    estimated_complexity: None,
                },
            )
            .await
            .unwrap();

        // Get full details
        let details = pm.get_plan_details(plan.id).await.unwrap().unwrap();

        // Verify plan
        assert_eq!(details.plan.title, "Full Plan");
        assert_eq!(details.plan.priority, 8);

        // Verify constraints
        assert_eq!(details.constraints.len(), 1);
        assert_eq!(
            details.constraints[0].constraint_type,
            ConstraintType::Style
        );
        assert_eq!(
            details.constraints[0].enforced_by,
            Some("rustfmt".to_string())
        );

        // Verify tasks
        assert_eq!(details.tasks.len(), 2);

        // Find task1 in details
        let t1_details = details
            .tasks
            .iter()
            .find(|t| t.task.id == task1.id)
            .unwrap();
        assert_eq!(t1_details.steps.len(), 1);
        assert!(t1_details.depends_on.is_empty());

        // Find task2 in details
        let t2_details = details
            .tasks
            .iter()
            .find(|t| t.task.title == Some("Task 2".to_string()))
            .unwrap();
        assert!(t2_details.depends_on.contains(&task1.id));
    }

    #[tokio::test]
    async fn test_get_plan_details_not_found() {
        let pm = create_plan_manager();
        let result = pm.get_plan_details(Uuid::new_v4()).await.unwrap();
        assert!(result.is_none(), "Expected None for non-existent plan");
    }
}
