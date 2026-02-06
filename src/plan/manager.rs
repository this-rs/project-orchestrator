//! Plan management operations

use super::models::*;
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
}

impl PlanManager {
    /// Create a new plan manager
    pub fn new(neo4j: Arc<dyn GraphStore>, meili: Arc<dyn SearchStore>) -> Self {
        Self { neo4j, meili }
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
        self.neo4j.update_plan_status(plan_id, status).await
    }

    /// Delete a plan and all its related data
    pub async fn delete_plan(&self, plan_id: Uuid) -> Result<()> {
        self.neo4j.delete_plan(plan_id).await
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

        Ok(())
    }

    /// Delete a task and all its related data (steps, decisions)
    pub async fn delete_task(&self, task_id: Uuid) -> Result<()> {
        self.neo4j.delete_task(task_id).await
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
        self.neo4j.create_step(task_id, step).await
    }

    /// Update step status
    pub async fn update_step_status(&self, step_id: Uuid, status: StepStatus) -> Result<()> {
        self.neo4j.update_step_status(step_id, status).await
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
        self.neo4j.create_constraint(plan_id, constraint).await
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
