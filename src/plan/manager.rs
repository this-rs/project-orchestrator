//! Plan management operations

use super::models::*;
use crate::meilisearch::client::MeiliClient;
use crate::meilisearch::indexes::DecisionDocument;
use crate::neo4j::client::Neo4jClient;
use crate::neo4j::models::*;
use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;

/// Manager for plan operations
pub struct PlanManager {
    neo4j: Arc<Neo4jClient>,
    meili: Arc<MeiliClient>,
}

impl PlanManager {
    /// Create a new plan manager
    pub fn new(neo4j: Arc<Neo4jClient>, meili: Arc<MeiliClient>) -> Self {
        Self { neo4j, meili }
    }

    // ========================================================================
    // Plan operations
    // ========================================================================

    /// Create a new plan
    pub async fn create_plan(&self, req: CreatePlanRequest, created_by: &str) -> Result<PlanNode> {
        let plan = PlanNode::new(
            req.title,
            req.description,
            created_by.to_string(),
            req.priority.unwrap_or(0),
        );

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
        // Get task from a query
        let q = neo4rs::query(
            r#"
            MATCH (t:Task {id: $id})
            OPTIONAL MATCH (t)-[:HAS_STEP]->(s:Step)
            OPTIONAL MATCH (t)-[:INFORMED_BY]->(d:Decision)
            OPTIONAL MATCH (t)-[:DEPENDS_ON]->(dep:Task)
            OPTIONAL MATCH (t)-[:MODIFIES]->(f:File)
            RETURN t,
                   collect(DISTINCT s) AS steps,
                   collect(DISTINCT d) AS decisions,
                   collect(DISTINCT dep.id) AS depends_on,
                   collect(DISTINCT f.path) AS files
            "#,
        )
        .param("id", task_id.to_string());

        let rows = self.neo4j.execute_with_params(q).await?;

        if rows.is_empty() {
            return Ok(None);
        }

        let row = &rows[0];
        let task_node: neo4rs::Node = row.get("t")?;

        let task = TaskNode {
            id: task_node.get::<String>("id")?.parse()?,
            title: task_node.get::<String>("title").ok().filter(|s| !s.is_empty()),
            description: task_node.get("description")?,
            status: serde_json::from_str(&format!(
                "\"{}\"",
                task_node.get::<String>("status")?.to_lowercase()
            ))
            .unwrap_or(TaskStatus::Pending),
            assigned_to: task_node.get("assigned_to").ok(),
            priority: task_node.get::<i64>("priority").ok().map(|v| v as i32),
            tags: task_node.get("tags").unwrap_or_default(),
            acceptance_criteria: task_node.get("acceptance_criteria").unwrap_or_default(),
            affected_files: task_node.get("affected_files").unwrap_or_default(),
            estimated_complexity: task_node
                .get::<i64>("estimated_complexity")
                .ok()
                .filter(|&v| v > 0)
                .map(|v| v as u32),
            actual_complexity: task_node
                .get::<i64>("actual_complexity")
                .ok()
                .filter(|&v| v > 0)
                .map(|v| v as u32),
            started_at: task_node
                .get::<String>("started_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            completed_at: task_node
                .get::<String>("completed_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            created_at: task_node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
        };

        // Parse steps from Neo4j nodes
        let step_nodes: Vec<neo4rs::Node> = row.get("steps").unwrap_or_default();
        let mut steps: Vec<StepNode> = step_nodes
            .iter()
            .filter_map(|node| {
                Some(StepNode {
                    id: node.get::<String>("id").ok()?.parse().ok()?,
                    order: node.get::<i64>("order").ok()? as u32,
                    description: node.get::<String>("description").ok()?,
                    status: node
                        .get::<String>("status")
                        .ok()
                        .and_then(|s| serde_json::from_str(&format!("\"{}\"", s.to_lowercase())).ok())
                        .unwrap_or(StepStatus::Pending),
                    verification: node.get::<String>("verification").ok().filter(|s| !s.is_empty()),
                })
            })
            .collect();
        // Sort steps by order
        steps.sort_by_key(|s| s.order);

        // Parse decisions from Neo4j nodes
        let decision_nodes: Vec<neo4rs::Node> = row.get("decisions").unwrap_or_default();
        let decisions: Vec<DecisionNode> = decision_nodes
            .iter()
            .filter_map(|node| {
                Some(DecisionNode {
                    id: node.get::<String>("id").ok()?.parse().ok()?,
                    description: node.get::<String>("description").ok()?,
                    rationale: node.get::<String>("rationale").ok()?,
                    alternatives: node.get::<Vec<String>>("alternatives").unwrap_or_default(),
                    chosen_option: node.get::<String>("chosen_option").ok().filter(|s| !s.is_empty()),
                    decided_by: node.get::<String>("decided_by").ok().unwrap_or_default(),
                    decided_at: node
                        .get::<String>("decided_at")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(chrono::Utc::now),
                })
            })
            .collect();

        // Parse dependencies
        let depends_on: Vec<String> = row.get("depends_on").unwrap_or_default();
        let depends_on: Vec<Uuid> = depends_on
            .into_iter()
            .filter_map(|s| s.parse().ok())
            .collect();

        let modifies_files: Vec<String> = row.get("files").unwrap_or_default();

        Ok(Some(TaskDetails {
            task,
            steps,
            decisions,
            depends_on,
            modifies_files,
        }))
    }

    /// Update task status
    pub async fn update_task(&self, task_id: Uuid, req: UpdateTaskRequest) -> Result<()> {
        if let Some(status) = req.status {
            self.neo4j.update_task_status(task_id, status).await?;
        }

        if let Some(agent_id) = req.assigned_to {
            self.neo4j.assign_task(task_id, &agent_id).await?;
        }

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
        let q = neo4rs::query(
            r#"
            MATCH (t:Task {id: $task_id})
            CREATE (s:Step {
                id: $id,
                order: $order,
                description: $description,
                status: $status,
                verification: $verification
            })
            CREATE (t)-[:HAS_STEP]->(s)
            "#,
        )
        .param("task_id", task_id.to_string())
        .param("id", step.id.to_string())
        .param("order", step.order as i64)
        .param("description", step.description.clone())
        .param("status", format!("{:?}", step.status))
        .param(
            "verification",
            step.verification.clone().unwrap_or_default(),
        );

        self.neo4j.execute_with_params(q).await?;
        Ok(())
    }

    /// Update step status
    pub async fn update_step_status(&self, step_id: Uuid, status: StepStatus) -> Result<()> {
        let q = neo4rs::query(
            r#"
            MATCH (s:Step {id: $id})
            SET s.status = $status
            "#,
        )
        .param("id", step_id.to_string())
        .param("status", format!("{:?}", status));

        self.neo4j.execute_with_params(q).await?;
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
                decided_at: doc
                    .timestamp
                    .parse()
                    .unwrap_or_else(|_| chrono::Utc::now()),
            })
            .collect();

        Ok(decisions)
    }

    // ========================================================================
    // Constraint operations
    // ========================================================================

    /// Add a constraint to a plan
    pub async fn add_constraint(&self, plan_id: Uuid, constraint: &ConstraintNode) -> Result<()> {
        let q = neo4rs::query(
            r#"
            MATCH (p:Plan {id: $plan_id})
            CREATE (c:Constraint {
                id: $id,
                constraint_type: $type,
                description: $description,
                enforced_by: $enforced_by
            })
            CREATE (p)-[:CONSTRAINED_BY]->(c)
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("id", constraint.id.to_string())
        .param("type", format!("{:?}", constraint.constraint_type))
        .param("description", constraint.description.clone())
        .param(
            "enforced_by",
            constraint.enforced_by.clone().unwrap_or_default(),
        );

        self.neo4j.execute_with_params(q).await?;
        Ok(())
    }

    // ========================================================================
    // Impact analysis
    // ========================================================================

    /// Analyze the impact of a task on the codebase
    pub async fn analyze_task_impact(&self, task_id: Uuid) -> Result<Vec<String>> {
        let q = neo4rs::query(
            r#"
            MATCH (t:Task {id: $id})-[:MODIFIES]->(f:File)
            OPTIONAL MATCH (f)<-[:IMPORTS*1..3]-(dependent:File)
            RETURN f.path AS file, collect(DISTINCT dependent.path) AS dependents
            "#,
        )
        .param("id", task_id.to_string());

        let rows = self.neo4j.execute_with_params(q).await?;
        let mut impacted = Vec::new();

        for row in rows {
            let file: String = row.get("file")?;
            impacted.push(file);

            let dependents: Vec<String> = row.get("dependents").unwrap_or_default();
            impacted.extend(dependents);
        }

        // Deduplicate
        impacted.sort();
        impacted.dedup();

        Ok(impacted)
    }

    /// Find blocked tasks in a plan
    pub async fn find_blocked_tasks(&self, plan_id: Uuid) -> Result<Vec<(TaskNode, Vec<TaskNode>)>> {
        let q = neo4rs::query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:HAS_TASK]->(t:Task {status: 'Pending'})
            MATCH (t)-[:DEPENDS_ON]->(blocker:Task)
            WHERE blocker.status <> 'Completed'
            RETURN t, collect(blocker) AS blockers
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let rows = self.neo4j.execute_with_params(q).await?;
        let mut result = Vec::new();

        for row in rows {
            let task_node: neo4rs::Node = row.get("t")?;
            let task = TaskNode {
                id: task_node.get::<String>("id")?.parse()?,
                title: task_node.get::<String>("title").ok().filter(|s| !s.is_empty()),
                description: task_node.get("description")?,
                status: TaskStatus::Pending,
                assigned_to: None,
                priority: task_node.get::<i64>("priority").ok().map(|v| v as i32),
                tags: task_node.get("tags").unwrap_or_default(),
                acceptance_criteria: task_node.get("acceptance_criteria").unwrap_or_default(),
                affected_files: task_node.get("affected_files").unwrap_or_default(),
                estimated_complexity: None,
                actual_complexity: None,
                started_at: None,
                completed_at: None,
                created_at: chrono::Utc::now(),
            };

            // Parse blockers from Neo4j nodes
            let blocker_nodes: Vec<neo4rs::Node> = row.get("blockers").unwrap_or_default();
            let blockers: Vec<TaskNode> = blocker_nodes
                .iter()
                .filter_map(|node| {
                    Some(TaskNode {
                        id: node.get::<String>("id").ok()?.parse().ok()?,
                        title: node.get::<String>("title").ok().filter(|s| !s.is_empty()),
                        description: node.get::<String>("description").ok()?,
                        status: node
                            .get::<String>("status")
                            .ok()
                            .and_then(|s| serde_json::from_str(&format!("\"{}\"", s.to_lowercase())).ok())
                            .unwrap_or(TaskStatus::Pending),
                        assigned_to: node.get::<String>("assigned_to").ok(),
                        priority: node.get::<i64>("priority").ok().map(|v| v as i32),
                        tags: node.get::<Vec<String>>("tags").unwrap_or_default(),
                        acceptance_criteria: node.get::<Vec<String>>("acceptance_criteria").unwrap_or_default(),
                        affected_files: node.get::<Vec<String>>("affected_files").unwrap_or_default(),
                        estimated_complexity: node.get::<i64>("estimated_complexity").ok().filter(|&v| v > 0).map(|v| v as u32),
                        actual_complexity: node.get::<i64>("actual_complexity").ok().filter(|&v| v > 0).map(|v| v as u32),
                        started_at: node.get::<String>("started_at").ok().and_then(|s| s.parse().ok()),
                        completed_at: node.get::<String>("completed_at").ok().and_then(|s| s.parse().ok()),
                        created_at: node.get::<String>("created_at").ok().and_then(|s| s.parse().ok()).unwrap_or_else(chrono::Utc::now),
                    })
                })
                .collect();
            result.push((task, blockers));
        }

        Ok(result)
    }
}
