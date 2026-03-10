//! Neo4j Plan operations

use super::client::{pascal_to_snake_case, Neo4jClient, WhereBuilder};
use super::models::*;
use crate::plan::models::UpdatePlanRequest;
use anyhow::{bail, Result};
use neo4rs::query;
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

// ============================================================================
// Task enrichment counts (for dependency graph view)
// ============================================================================

/// Counts of related entities for a task node (steps, notes, decisions)
#[derive(Debug, Clone, Default)]
pub struct TaskEnrichmentCounts {
    pub step_count: usize,
    pub completed_step_count: usize,
    pub note_count: usize,
    pub decision_count: usize,
}

/// Individual step info for dependency graph visualization
#[derive(Debug, Clone, Serialize)]
pub struct StepSummary {
    pub id: String,
    pub order: u32,
    pub description: String,
    pub status: String,
    pub verification: Option<String>,
}

/// Full enrichment data for a task node (counts + step details)
#[derive(Debug, Clone, Default)]
pub struct TaskEnrichmentData {
    pub counts: TaskEnrichmentCounts,
    pub steps: Vec<StepSummary>,
}

/// Compute file conflicts from a list of (id, affected_files) pairs.
/// Checks all pairs for shared affected_files. Used by both DAG and Wave views.
pub fn compute_file_conflicts(items: &[(Uuid, &[String])]) -> Vec<FileConflict> {
    let mut conflicts = Vec::new();
    for i in 0..items.len() {
        let files_a: HashSet<&str> = items[i].1.iter().map(|s| s.as_str()).collect();
        if files_a.is_empty() {
            continue;
        }
        for j in (i + 1)..items.len() {
            let shared: Vec<String> = items[j]
                .1
                .iter()
                .filter(|f| files_a.contains(f.as_str()))
                .cloned()
                .collect();
            if !shared.is_empty() {
                conflicts.push(FileConflict {
                    task_a: items[i].0,
                    task_b: items[j].0,
                    shared_files: shared,
                });
            }
        }
    }
    conflicts
}

// ============================================================================
// Wave computation types
// ============================================================================

/// A single wave containing tasks that can be executed in parallel
#[derive(Debug, Serialize, Clone)]
pub struct Wave {
    /// Wave number (1-indexed)
    pub wave_number: usize,
    /// Tasks in this wave
    pub tasks: Vec<WaveTask>,
    /// Number of tasks in this wave
    pub task_count: usize,
    /// Whether this wave was split due to file conflicts
    pub split_from_conflicts: bool,
}

/// A task within a wave
#[derive(Debug, Serialize, Clone)]
pub struct WaveTask {
    pub id: Uuid,
    pub title: Option<String>,
    pub status: String,
    pub priority: Option<i32>,
    pub affected_files: Vec<String>,
    pub depends_on: Vec<Uuid>,
}

/// A conflict between two tasks sharing affected_files
#[derive(Debug, Serialize, Clone)]
pub struct FileConflict {
    pub task_a: Uuid,
    pub task_b: Uuid,
    pub shared_files: Vec<String>,
}

/// Summary statistics for wave computation
#[derive(Debug, Serialize)]
pub struct WaveSummary {
    pub total_tasks: usize,
    pub total_waves: usize,
    pub max_parallel: usize,
    pub critical_path_length: usize,
    pub dependency_edges: usize,
    pub conflicts_detected: usize,
}

/// Complete wave computation result
#[derive(Debug, Serialize)]
pub struct WaveComputationResult {
    pub waves: Vec<Wave>,
    pub summary: WaveSummary,
    pub conflicts: Vec<FileConflict>,
    pub edges: Vec<(Uuid, Uuid)>,
}

impl Neo4jClient {
    // ========================================================================
    // Plan operations
    // ========================================================================

    /// Create a new plan
    pub async fn create_plan(&self, plan: &PlanNode) -> Result<()> {
        let q = query(
            r#"
            CREATE (p:Plan {
                id: $id,
                title: $title,
                description: $description,
                status: $status,
                created_at: datetime($created_at),
                created_by: $created_by,
                priority: $priority,
                project_id: $project_id
            })
            "#,
        )
        .param("id", plan.id.to_string())
        .param("title", plan.title.clone())
        .param("description", plan.description.clone())
        .param("status", format!("{:?}", plan.status))
        .param("created_at", plan.created_at.to_rfc3339())
        .param("created_by", plan.created_by.clone())
        .param("priority", plan.priority as i64)
        .param(
            "project_id",
            plan.project_id.map(|id| id.to_string()).unwrap_or_default(),
        );

        self.graph.run(q).await?;

        // Link to project if specified
        if let Some(project_id) = plan.project_id {
            let q = query(
                r#"
                MATCH (project:Project {id: $project_id})
                MATCH (plan:Plan {id: $plan_id})
                MERGE (project)-[:HAS_PLAN]->(plan)
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("plan_id", plan.id.to_string());

            self.graph.run(q).await?;
        }

        Ok(())
    }

    /// Get a plan by ID
    pub async fn get_plan(&self, id: Uuid) -> Result<Option<PlanNode>> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})
            RETURN p
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            Ok(Some(self.node_to_plan(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Helper to convert Neo4j node to PlanNode
    pub(crate) fn node_to_plan(&self, node: &neo4rs::Node) -> Result<PlanNode> {
        Ok(PlanNode {
            id: node.get::<String>("id")?.parse()?,
            title: node.get("title")?,
            description: node.get("description")?,
            status: serde_json::from_str(&format!(
                "\"{}\"",
                pascal_to_snake_case(&node.get::<String>("status")?)
            ))
            .unwrap_or(PlanStatus::Draft),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            created_by: node.get("created_by")?,
            priority: node.get::<i64>("priority")? as i32,
            project_id: node.get::<String>("project_id").ok().and_then(|s| {
                if s.is_empty() {
                    None
                } else {
                    s.parse().ok()
                }
            }),
            execution_context: node
                .get::<String>("execution_context")
                .ok()
                .filter(|s| !s.is_empty()),
            persona: node.get::<String>("persona").ok().filter(|s| !s.is_empty()),
        })
    }

    /// List all active plans
    pub async fn list_active_plans(&self) -> Result<Vec<PlanNode>> {
        let q = query(
            r#"
            MATCH (p:Plan)
            WHERE p.status IN ['Draft', 'Approved', 'InProgress']
            RETURN p
            ORDER BY p.priority DESC, p.created_at DESC
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut plans = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            plans.push(self.node_to_plan(&node)?);
        }

        Ok(plans)
    }

    /// List active plans for a specific project
    pub async fn list_project_plans(&self, project_id: Uuid) -> Result<Vec<PlanNode>> {
        let q = query(
            r#"
            MATCH (project:Project {id: $project_id})-[:HAS_PLAN]->(p:Plan)
            WHERE p.status IN ['Draft', 'Approved', 'InProgress']
            RETURN p
            ORDER BY p.priority DESC, p.created_at DESC
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut plans = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            plans.push(self.node_to_plan(&node)?);
        }

        Ok(plans)
    }

    /// Count plans for a project (lightweight COUNT query, no data transfer).
    pub async fn count_project_plans(&self, project_id: Uuid) -> Result<i64> {
        let q = query(
            "MATCH (project:Project {id: $project_id})-[:HAS_PLAN]->(p:Plan) WHERE p.status IN ['Draft', 'Approved', 'InProgress'] RETURN count(p) AS cnt",
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            Ok(row.get::<i64>("cnt")?)
        } else {
            Ok(0)
        }
    }

    /// List plans for a project with filters
    pub async fn list_plans_for_project(
        &self,
        project_id: Uuid,
        status_filter: Option<Vec<String>>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<PlanNode>, usize)> {
        // Build status filter
        let status_clause = if let Some(statuses) = &status_filter {
            if !statuses.is_empty() {
                let status_list: Vec<String> = statuses
                    .iter()
                    .map(|s| {
                        // Convert to PascalCase for enum matching
                        let pascal = match s.to_lowercase().as_str() {
                            "draft" => "Draft",
                            "approved" => "Approved",
                            "in_progress" => "InProgress",
                            "completed" => "Completed",
                            "cancelled" => "Cancelled",
                            _ => s.as_str(),
                        };
                        format!("'{}'", pascal)
                    })
                    .collect();
                format!("AND p.status IN [{}]", status_list.join(", "))
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Count total
        let count_q = query(&format!(
            r#"
            MATCH (project:Project {{id: $project_id}})-[:HAS_PLAN]->(p:Plan)
            WHERE true {}
            RETURN count(p) AS total
            "#,
            status_clause
        ))
        .param("project_id", project_id.to_string());

        let count_rows = self.execute_with_params(count_q).await?;
        let total: i64 = count_rows
            .first()
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        // Get plans
        let q = query(&format!(
            r#"
            MATCH (project:Project {{id: $project_id}})-[:HAS_PLAN]->(p:Plan)
            WHERE true {}
            RETURN p
            ORDER BY p.priority DESC, p.created_at DESC
            SKIP $offset
            LIMIT $limit
            "#,
            status_clause
        ))
        .param("project_id", project_id.to_string())
        .param("offset", offset as i64)
        .param("limit", limit as i64);

        let mut result = self.graph.execute(q).await?;
        let mut plans = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            plans.push(self.node_to_plan(&node)?);
        }

        Ok((plans, total as usize))
    }

    /// Update plan fields (title, description, priority)
    pub async fn update_plan(&self, id: Uuid, updates: &UpdatePlanRequest) -> Result<()> {
        let mut set_clauses = Vec::new();

        if updates.title.is_some() {
            set_clauses.push("p.title = $title");
        }
        if updates.description.is_some() {
            set_clauses.push("p.description = $description");
        }
        if updates.priority.is_some() {
            set_clauses.push("p.priority = $priority");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!("MATCH (p:Plan {{id: $id}}) SET {}", set_clauses.join(", "));
        let mut q = query(&cypher).param("id", id.to_string());

        if let Some(title) = &updates.title {
            q = q.param("title", title.clone());
        }
        if let Some(description) = &updates.description {
            q = q.param("description", description.clone());
        }
        if let Some(priority) = updates.priority {
            q = q.param("priority", priority as i64);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Update plan status
    pub async fn update_plan_status(&self, id: Uuid, status: PlanStatus) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})
            SET p.status = $status
            "#,
        )
        .param("id", id.to_string())
        .param("status", format!("{:?}", status));

        self.graph.run(q).await?;
        Ok(())
    }

    /// Link a plan to a project (creates HAS_PLAN relationship)
    pub async fn link_plan_to_project(&self, plan_id: Uuid, project_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (project:Project {id: $project_id})
            MATCH (plan:Plan {id: $plan_id})
            SET plan.project_id = $project_id
            MERGE (project)-[:HAS_PLAN]->(plan)
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("plan_id", plan_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Unlink a plan from its project
    pub async fn unlink_plan_from_project(&self, plan_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (project:Project)-[r:HAS_PLAN]->(plan:Plan {id: $plan_id})
            DELETE r
            SET plan.project_id = null
            "#,
        )
        .param("plan_id", plan_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a plan and all its related data (tasks, steps, decisions, constraints)
    pub async fn delete_plan(&self, plan_id: Uuid) -> Result<()> {
        // Delete all steps belonging to tasks of this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})-[:HAS_TASK]->(t:Task)-[:HAS_STEP]->(s:Step)
            DETACH DELETE s
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        // Delete all decisions belonging to tasks of this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})-[:HAS_TASK]->(t:Task)-[:INFORMED_BY]->(d:Decision)
            DETACH DELETE d
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        // Delete all tasks belonging to this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})-[:HAS_TASK]->(t:Task)
            DETACH DELETE t
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        // Delete all constraints belonging to this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})-[:CONSTRAINED_BY]->(c:Constraint)
            DETACH DELETE c
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        // Delete the plan itself
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})
            DETACH DELETE p
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        Ok(())
    }

    /// Get dependency graph for a plan (all tasks and their dependencies)
    pub async fn get_plan_dependency_graph(
        &self,
        plan_id: Uuid,
    ) -> Result<(Vec<TaskNode>, Vec<(Uuid, Uuid)>)> {
        // Get all tasks in the plan
        let tasks = self.get_plan_tasks(plan_id).await?;

        // Get all DEPENDS_ON edges between tasks in this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:HAS_TASK]->(t:Task)-[:DEPENDS_ON]->(dep:Task)<-[:HAS_TASK]-(p)
            RETURN t.id AS from_id, dep.id AS to_id
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut edges = Vec::new();

        while let Some(row) = result.next().await? {
            let from_id: String = row.get("from_id")?;
            let to_id: String = row.get("to_id")?;
            if let (Ok(from), Ok(to)) = (from_id.parse::<Uuid>(), to_id.parse::<Uuid>()) {
                edges.push((from, to));
            }
        }

        Ok((tasks, edges))
    }

    /// Find critical path in a plan (longest chain of dependencies)
    pub async fn get_plan_critical_path(&self, plan_id: Uuid) -> Result<Vec<TaskNode>> {
        // Get all paths from tasks with no incoming deps to tasks with no outgoing deps
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:HAS_TASK]->(start:Task)
            WHERE NOT EXISTS { MATCH (start)-[:DEPENDS_ON]->(:Task) }
            MATCH (p)-[:HAS_TASK]->(end:Task)
            WHERE NOT EXISTS { MATCH (:Task)-[:DEPENDS_ON]->(end) }
            MATCH path = (start)<-[:DEPENDS_ON*0..]-(end)
            WHERE ALL(node IN nodes(path) WHERE (p)-[:HAS_TASK]->(node))
            WITH path, length(path) AS pathLength
            ORDER BY pathLength DESC
            LIMIT 1
            UNWIND nodes(path) AS task
            RETURN DISTINCT task
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("task")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// Batch-fetch step counts, note counts, and decision counts for a list of task IDs.
    /// Returns a HashMap<task_id_string, TaskEnrichmentCounts>.
    /// Uses a single Cypher query with UNWIND for efficiency.
    pub async fn get_task_enrichment_counts(
        &self,
        task_ids: &[String],
    ) -> Result<HashMap<String, TaskEnrichmentCounts>> {
        if task_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let q = query(
            r#"
            UNWIND $task_ids AS tid
            MATCH (t:Task {id: tid})
            OPTIONAL MATCH (t)-[:HAS_STEP]->(s:Step)
            WITH t, tid,
                 count(s) AS total_steps,
                 count(CASE WHEN s.status = 'Completed' THEN 1 END) AS done_steps
            OPTIONAL MATCH (t)-[:HAS_DECISION]->(d:Decision)
            WITH t, tid, total_steps, done_steps, count(d) AS decision_count
            OPTIONAL MATCH (n:Note)-[:LINKED_TO]->(t)
            RETURN tid,
                   total_steps,
                   done_steps,
                   decision_count,
                   count(n) AS note_count
            "#,
        )
        .param("task_ids", task_ids.to_vec());

        let mut result = self.graph.execute(q).await?;
        let mut map = HashMap::new();

        while let Some(row) = result.next().await? {
            let tid: String = row.get("tid")?;
            let step_count: i64 = row.get("total_steps").unwrap_or(0);
            let completed_step_count: i64 = row.get("done_steps").unwrap_or(0);
            let note_count: i64 = row.get("note_count").unwrap_or(0);
            let decision_count: i64 = row.get("decision_count").unwrap_or(0);

            map.insert(
                tid,
                TaskEnrichmentCounts {
                    step_count: step_count as usize,
                    completed_step_count: completed_step_count as usize,
                    note_count: note_count as usize,
                    decision_count: decision_count as usize,
                },
            );
        }

        Ok(map)
    }

    /// Batch-fetch full enrichment data (counts + individual step details) for dependency graph.
    /// Uses two efficient queries: one for counts, one for step details.
    pub async fn get_task_enrichment_data(
        &self,
        task_ids: &[String],
    ) -> Result<HashMap<String, TaskEnrichmentData>> {
        if task_ids.is_empty() {
            return Ok(HashMap::new());
        }

        // 1. Get counts (reuse existing logic)
        let counts = self.get_task_enrichment_counts(task_ids).await?;

        // 2. Get step details in a single batch query
        let q = query(
            r#"
            UNWIND $task_ids AS tid
            MATCH (t:Task {id: tid})-[:HAS_STEP]->(s:Step)
            RETURN tid,
                   s.id AS step_id,
                   COALESCE(s.order, 0) AS step_order,
                   s.description AS description,
                   s.status AS status,
                   s.verification AS verification
            ORDER BY tid, step_order
            "#,
        )
        .param("task_ids", task_ids.to_vec());

        let mut result = self.graph.execute(q).await?;
        let mut steps_map: HashMap<String, Vec<StepSummary>> = HashMap::new();

        while let Some(row) = result.next().await? {
            let tid: String = row.get("tid")?;
            let step_id: String = row.get("step_id").unwrap_or_default();
            let order: i64 = row.get("step_order").unwrap_or(0);
            let description: String = row.get("description").unwrap_or_default();
            let status: String = row.get("status").unwrap_or_else(|_| "Pending".to_string());
            let verification: Option<String> = row.get("verification").ok();

            steps_map.entry(tid).or_default().push(StepSummary {
                id: step_id,
                order: order as u32,
                description,
                status,
                verification,
            });
        }

        // 3. Merge counts + steps
        let mut data_map = HashMap::new();
        for tid in task_ids {
            data_map.insert(
                tid.clone(),
                TaskEnrichmentData {
                    counts: counts.get(tid).cloned().unwrap_or_default(),
                    steps: steps_map.remove(tid).unwrap_or_default(),
                },
            );
        }

        Ok(data_map)
    }

    /// Compute execution waves for a plan using topological sort (Kahn's algorithm)
    ///
    /// Waves represent groups of tasks that can be executed in parallel.
    /// Wave 1 contains tasks with no dependencies, wave 2 contains tasks
    /// that only depend on wave 1, etc.
    ///
    /// Returns an error if the dependency graph contains cycles.
    pub async fn compute_waves(&self, plan_id: Uuid) -> Result<WaveComputationResult> {
        // 1. Get the DAG: tasks + dependency edges
        let (tasks, edges) = self.get_plan_dependency_graph(plan_id).await?;

        if tasks.is_empty() {
            return Ok(WaveComputationResult {
                waves: vec![],
                summary: WaveSummary {
                    total_tasks: 0,
                    total_waves: 0,
                    max_parallel: 0,
                    critical_path_length: 0,
                    dependency_edges: 0,
                    conflicts_detected: 0,
                },
                conflicts: vec![],
                edges: vec![],
            });
        }

        // 2. Build adjacency list + in-degree map
        // edges are (from, to) where `from` DEPENDS_ON `to`
        // meaning `to` must complete before `from` can start
        let task_map: HashMap<Uuid, &TaskNode> = tasks.iter().map(|t| (t.id, t)).collect();

        // Dependencies: task -> set of tasks it depends on
        let mut deps_of: HashMap<Uuid, HashSet<Uuid>> = HashMap::new();
        // Reverse: task -> set of tasks that depend on it
        let mut dependents_of: HashMap<Uuid, HashSet<Uuid>> = HashMap::new();
        // In-degree: number of unresolved dependencies
        let mut in_degree: HashMap<Uuid, usize> = HashMap::new();

        // Initialize all tasks with zero in-degree
        for task in &tasks {
            in_degree.insert(task.id, 0);
            deps_of.insert(task.id, HashSet::new());
            dependents_of.insert(task.id, HashSet::new());
        }

        // Build edges
        for &(from, to) in &edges {
            // `from` depends on `to`
            if task_map.contains_key(&from) && task_map.contains_key(&to) {
                deps_of.entry(from).or_default().insert(to);
                dependents_of.entry(to).or_default().insert(from);
                *in_degree.entry(from).or_default() += 1;
            }
        }

        // 3. Kahn's algorithm with level grouping
        // Start with all tasks that have zero in-degree (no dependencies)
        let mut queue: VecDeque<Uuid> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        // Sort initial queue deterministically: by priority desc, then title asc
        let mut initial: Vec<Uuid> = queue.drain(..).collect();
        initial.sort_by(|a, b| {
            let pa = task_map[a].priority.unwrap_or(0);
            let pb = task_map[b].priority.unwrap_or(0);
            pb.cmp(&pa).then_with(|| {
                let ta = task_map[a].title.as_deref().unwrap_or("");
                let tb = task_map[b].title.as_deref().unwrap_or("");
                ta.cmp(tb)
            })
        });
        queue.extend(initial);

        let mut waves: Vec<Wave> = Vec::new();
        let mut processed_count = 0;
        let mut task_wave: HashMap<Uuid, usize> = HashMap::new(); // task_id -> wave_number

        while !queue.is_empty() {
            // All tasks in the current queue form one wave
            let current_level: Vec<Uuid> = queue.drain(..).collect();
            let wave_number = waves.len() + 1; // 1-indexed

            let mut wave_tasks: Vec<WaveTask> = Vec::new();
            let mut next_queue: Vec<Uuid> = Vec::new();

            for &task_id in &current_level {
                processed_count += 1;
                task_wave.insert(task_id, wave_number);

                if let Some(task) = task_map.get(&task_id) {
                    wave_tasks.push(WaveTask {
                        id: task.id,
                        title: task.title.clone(),
                        status: format!("{:?}", task.status),
                        priority: task.priority,
                        affected_files: task.affected_files.clone(),
                        depends_on: deps_of
                            .get(&task_id)
                            .map(|s| s.iter().copied().collect())
                            .unwrap_or_default(),
                    });
                }

                // Decrement in-degree for all dependents
                if let Some(dependents) = dependents_of.get(&task_id) {
                    for &dep_id in dependents {
                        if let Some(deg) = in_degree.get_mut(&dep_id) {
                            *deg -= 1;
                            if *deg == 0 {
                                next_queue.push(dep_id);
                            }
                        }
                    }
                }
            }

            // Sort wave tasks deterministically
            wave_tasks.sort_by(|a, b| {
                let pa = a.priority.unwrap_or(0);
                let pb = b.priority.unwrap_or(0);
                pb.cmp(&pa).then_with(|| {
                    let ta = a.title.as_deref().unwrap_or("");
                    let tb = b.title.as_deref().unwrap_or("");
                    ta.cmp(tb)
                })
            });

            let task_count = wave_tasks.len();
            waves.push(Wave {
                wave_number,
                tasks: wave_tasks,
                task_count,
                split_from_conflicts: false,
            });

            // Sort next queue deterministically before adding
            next_queue.sort_by(|a, b| {
                let pa = task_map[a].priority.unwrap_or(0);
                let pb = task_map[b].priority.unwrap_or(0);
                pb.cmp(&pa).then_with(|| {
                    let ta = task_map[a].title.as_deref().unwrap_or("");
                    let tb = task_map[b].title.as_deref().unwrap_or("");
                    ta.cmp(tb)
                })
            });
            queue.extend(next_queue);
        }

        // 4. Cycle detection: if we didn't process all tasks, there's a cycle
        if processed_count < tasks.len() {
            let cycle_tasks: Vec<String> = tasks
                .iter()
                .filter(|t| !task_wave.contains_key(&t.id))
                .map(|t| format!("{} ({})", t.title.as_deref().unwrap_or("untitled"), t.id))
                .collect();
            bail!(
                "Cycle detected in dependency graph! Tasks involved: {}",
                cycle_tasks.join(", ")
            );
        }

        // 5. Conflict splitting — detect affected_files intersections within each wave
        let mut all_conflicts: Vec<FileConflict> = Vec::new();
        let mut split_waves: Vec<Wave> = Vec::new();
        let mut wave_counter = 0usize;

        for wave in &waves {
            // Build conflict graph for tasks in this wave
            let mut conflicts_in_wave: Vec<(usize, usize, Vec<String>)> = Vec::new();

            for i in 0..wave.tasks.len() {
                let files_a: HashSet<&str> = wave.tasks[i]
                    .affected_files
                    .iter()
                    .map(|s| s.as_str())
                    .collect();
                if files_a.is_empty() {
                    continue;
                }
                for j in (i + 1)..wave.tasks.len() {
                    let shared: Vec<String> = wave.tasks[j]
                        .affected_files
                        .iter()
                        .filter(|f| files_a.contains(f.as_str()))
                        .cloned()
                        .collect();
                    if !shared.is_empty() {
                        conflicts_in_wave.push((i, j, shared.clone()));
                        all_conflicts.push(FileConflict {
                            task_a: wave.tasks[i].id,
                            task_b: wave.tasks[j].id,
                            shared_files: shared,
                        });
                    }
                }
            }

            if conflicts_in_wave.is_empty() {
                // No conflicts — keep wave as-is
                wave_counter += 1;
                split_waves.push(Wave {
                    wave_number: wave_counter,
                    tasks: wave.tasks.clone(),
                    task_count: wave.task_count,
                    split_from_conflicts: false,
                });
            } else {
                // Greedy graph coloring to partition into conflict-free groups
                let n = wave.tasks.len();
                let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
                for &(i, j, _) in &conflicts_in_wave {
                    adj[i].insert(j);
                    adj[j].insert(i);
                }

                let mut colors: Vec<Option<usize>> = vec![None; n];
                // Sort by degree descending for better coloring
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|a, b| adj[*b].len().cmp(&adj[*a].len()));

                for idx in order {
                    let used: HashSet<usize> = adj[idx]
                        .iter()
                        .filter_map(|&neighbor| colors[neighbor])
                        .collect();
                    let mut color = 0;
                    while used.contains(&color) {
                        color += 1;
                    }
                    colors[idx] = Some(color);
                }

                // Group tasks by color → each color = one sub-wave
                let max_color = colors.iter().filter_map(|c| *c).max().unwrap_or(0);
                for color in 0..=max_color {
                    let group_tasks: Vec<WaveTask> = wave
                        .tasks
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| colors[*i] == Some(color))
                        .map(|(_, t)| t.clone())
                        .collect();
                    if !group_tasks.is_empty() {
                        wave_counter += 1;
                        let task_count = group_tasks.len();
                        split_waves.push(Wave {
                            wave_number: wave_counter,
                            tasks: group_tasks,
                            task_count,
                            split_from_conflicts: true,
                        });
                    }
                }
            }
        }

        // 6. Compute critical path length (longest chain)
        let mut longest_path: HashMap<Uuid, usize> = HashMap::new();
        for wave in &split_waves {
            for task in &wave.tasks {
                let max_dep_path = task
                    .depends_on
                    .iter()
                    .filter_map(|dep_id| longest_path.get(dep_id))
                    .max()
                    .copied()
                    .unwrap_or(0);
                longest_path.insert(task.id, max_dep_path + 1);
            }
        }
        let critical_path_length = longest_path.values().max().copied().unwrap_or(0);

        // 7. Compute summary
        let max_parallel = split_waves.iter().map(|w| w.task_count).max().unwrap_or(0);

        let summary = WaveSummary {
            total_tasks: tasks.len(),
            total_waves: split_waves.len(),
            max_parallel,
            critical_path_length,
            dependency_edges: edges.len(),
            conflicts_detected: all_conflicts.len(),
        };

        Ok(WaveComputationResult {
            waves: split_waves,
            summary,
            conflicts: all_conflicts,
            edges,
        })
    }

    /// List plans with filters and pagination
    ///
    /// Returns (plans, total_count)
    #[allow(clippy::too_many_arguments)]
    pub async fn list_plans_filtered(
        &self,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
        statuses: Option<Vec<String>>,
        priority_min: Option<i32>,
        priority_max: Option<i32>,
        search: Option<&str>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<PlanNode>, usize)> {
        let mut where_builder = WhereBuilder::new();
        where_builder
            .add_status_filter("p", statuses)
            .add_priority_filter("p", priority_min, priority_max)
            .add_search_filter("p", search);

        let where_clause = where_builder.build();
        let order_field = match sort_by {
            Some("priority") => "COALESCE(p.priority, 0)",
            Some("title") => "p.title",
            Some("status") => "p.status",
            _ => "p.created_at",
        };
        let order_dir = if sort_order == "asc" { "ASC" } else { "DESC" };

        let match_clause = if let Some(pid) = project_id {
            format!(
                "MATCH (proj:Project {{id: '{}'}})-[:HAS_PLAN]->(p:Plan)",
                pid
            )
        } else if let Some(ws) = workspace_slug {
            format!(
                "MATCH (w:Workspace {{slug: '{}'}})<-[:BELONGS_TO_WORKSPACE]-(proj:Project)-[:HAS_PLAN]->(p:Plan)",
                ws
            )
        } else {
            "MATCH (p:Plan)".to_string()
        };

        // Count query
        let count_cypher = format!("{} {} RETURN count(p) AS total", match_clause, where_clause);
        let count_result = self.execute(&count_cypher).await?;
        let total: i64 = count_result
            .first()
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        // Data query
        let cypher = format!(
            r#"
            {}
            {}
            RETURN p
            ORDER BY {} {}
            SKIP {}
            LIMIT {}
            "#,
            match_clause, where_clause, order_field, order_dir, offset, limit
        );

        let mut result = self.graph.execute(query(&cypher)).await?;
        let mut plans = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            plans.push(self.node_to_plan(&node)?);
        }

        Ok((plans, total as usize))
    }

    // ========================================================================
    // Graph visualization — PM layer
    // ========================================================================

    /// Get all PM entities and edges for the graph visualization layer.
    /// Uses efficient bulk Cypher queries scoped by project_id.
    pub async fn get_pm_graph_data(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<(Vec<PmGraphNode>, Vec<PmGraphEdge>)> {
        let pid = project_id.to_string();
        let lim = limit as i64;

        // ── Query 1: All PM nodes via UNION ──
        let nodes_query = query(
            "MATCH (proj:Project {id: $pid})-[:HAS_PLAN]->(p:Plan)
             RETURN p.id AS id, 'plan' AS type, p.title AS label,
                    p.status AS status, p.priority AS priority,
                    null AS target_date, null AS version, null AS description
             ORDER BY p.priority DESC
             LIMIT $lim

             UNION ALL

             MATCH (proj:Project {id: $pid})-[:HAS_PLAN]->(pl:Plan)-[:HAS_TASK]->(t:Task)
             RETURN t.id AS id, 'task' AS type, t.title AS label,
                    t.status AS status, t.priority AS priority,
                    null AS target_date, null AS version, null AS description
             LIMIT $lim

             UNION ALL

             MATCH (proj:Project {id: $pid})-[:HAS_PLAN]->(:Plan)-[:HAS_TASK]->(t:Task)-[:HAS_STEP]->(s:Step)
             RETURN s.id AS id, 'step' AS type, s.description AS label,
                    s.status AS status, null AS priority,
                    null AS target_date, null AS version, null AS description
             LIMIT $lim

             UNION ALL

             MATCH (proj:Project {id: $pid})-[:HAS_MILESTONE]->(m:Milestone)
             RETURN m.id AS id, 'milestone' AS type, m.title AS label,
                    m.status AS status, null AS priority,
                    m.target_date AS target_date, null AS version, m.description AS description
             LIMIT $lim

             UNION ALL

             MATCH (proj:Project {id: $pid})-[:HAS_RELEASE]->(r:Release)
             RETURN r.id AS id, 'release' AS type, r.title AS label,
                    r.status AS status, null AS priority,
                    r.target_date AS target_date, r.version AS version, null AS description
             LIMIT $lim

             UNION ALL

             MATCH (proj:Project {id: $pid})-[:HAS_PLAN]->(pl:Plan)<-[:LINKED_TO_PLAN]-(c:Commit)
             RETURN c.sha AS id, 'commit' AS type,
                    COALESCE(substring(c.message, 0, 60), c.sha) AS label,
                    null AS status, null AS priority,
                    c.timestamp AS target_date, null AS version, null AS description
             LIMIT $lim

             UNION ALL

             MATCH (proj:Project {id: $pid})-[:HAS_PLAN]->(:Plan)-[:HAS_TASK]->(t:Task)<-[:LINKED_TO_TASK]-(c:Commit)
             RETURN c.sha AS id, 'commit' AS type,
                    COALESCE(substring(c.message, 0, 60), c.sha) AS label,
                    null AS status, null AS priority,
                    c.timestamp AS target_date, null AS version, null AS description
             LIMIT $lim",
        )
        .param("pid", pid.clone())
        .param("lim", lim);

        let mut result = self.graph.execute(nodes_query).await?;
        let mut nodes = Vec::new();
        while let Some(row) = result.next().await? {
            let id: String = row.get("id")?;
            let node_type: String = row.get("type")?;
            let label: String = row.get::<String>("label").unwrap_or_default();
            let status: Option<String> = row.get::<Option<String>>("status").ok().flatten();
            let priority: Option<i64> = row.get::<Option<i64>>("priority").ok().flatten();
            let target_date: Option<String> =
                row.get::<Option<String>>("target_date").ok().flatten();
            let version: Option<String> = row.get::<Option<String>>("version").ok().flatten();

            nodes.push(PmGraphNode {
                id,
                node_type,
                label: if label.chars().count() > 80 {
                    let truncated: String = label.chars().take(79).collect();
                    format!("{truncated}…")
                } else {
                    label
                },
                attributes: serde_json::json!({
                    "status": status,
                    "priority": priority,
                    "target_date": target_date,
                    "version": version,
                }),
            });
        }

        // ── Query 2: All PM edges via UNION ──
        let edges_query = query(
            "MATCH (proj:Project {id: $pid})-[:HAS_PLAN]->(pl:Plan)-[:HAS_TASK]->(t:Task)
             RETURN pl.id AS source, t.id AS target, 'HAS_TASK' AS rel_type

             UNION ALL

             MATCH (proj:Project {id: $pid})-[:HAS_PLAN]->(:Plan)-[:HAS_TASK]->(t:Task)-[:HAS_STEP]->(s:Step)
             RETURN t.id AS source, s.id AS target, 'HAS_STEP' AS rel_type

             UNION ALL

             MATCH (proj:Project {id: $pid})-[:HAS_PLAN]->(:Plan)-[:HAS_TASK]->(t1:Task)-[:DEPENDS_ON]->(t2:Task)
             RETURN t1.id AS source, t2.id AS target, 'DEPENDS_ON' AS rel_type

             UNION ALL

             MATCH (proj:Project {id: $pid})-[:HAS_PLAN]->(pl:Plan)-[:TARGETS_MILESTONE]->(m:Milestone)
             RETURN pl.id AS source, m.id AS target, 'TARGETS_MILESTONE' AS rel_type

             UNION ALL

             MATCH (proj:Project {id: $pid})-[:HAS_PLAN]->(pl:Plan)<-[:LINKED_TO_PLAN]-(c:Commit)
             RETURN c.sha AS source, pl.id AS target, 'LINKED_TO_PLAN' AS rel_type

             UNION ALL

             MATCH (proj:Project {id: $pid})-[:HAS_PLAN]->(:Plan)-[:HAS_TASK]->(t:Task)<-[:LINKED_TO_TASK]-(c:Commit)
             RETURN c.sha AS source, t.id AS target, 'LINKED_TO_TASK' AS rel_type

             UNION ALL

             MATCH (proj:Project {id: $pid})-[:HAS_PLAN]->(pl:Plan)<-[:LINKED_TO_PLAN]-(c:Commit)-[r:TOUCHES]->(f:File)
             RETURN c.sha AS source, f.path AS target, 'TOUCHES' AS rel_type

             UNION ALL

             MATCH (proj:Project {id: $pid})-[:HAS_PLAN]->(:Plan)-[:HAS_TASK]->(t:Task)<-[:LINKED_TO_TASK]-(c:Commit)-[r:TOUCHES]->(f:File)
             RETURN c.sha AS source, f.path AS target, 'TOUCHES' AS rel_type",
        )
        .param("pid", pid);

        let mut result = self.graph.execute(edges_query).await?;
        let mut edges = Vec::new();
        while let Some(row) = result.next().await? {
            edges.push(PmGraphEdge {
                source: row.get("source")?,
                target: row.get("target")?,
                rel_type: row.get("rel_type")?,
                attributes: None,
            });
        }

        Ok((nodes, edges))
    }
}
