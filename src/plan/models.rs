//! Plan-related models and DTOs

use crate::neo4j::models::{
    ConstraintNode, ConstraintType, DecisionNode, PlanNode, PlanStatus, StepNode, StepStatus,
    TaskNode, TaskStatus,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Request to create a new plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatePlanRequest {
    pub title: String,
    pub description: String,
    pub priority: Option<i32>,
    pub constraints: Option<Vec<CreateConstraintRequest>>,
}

/// Request to create a new task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTaskRequest {
    /// Short title for the task
    pub title: Option<String>,
    /// Detailed description of what needs to be done
    pub description: String,
    /// Priority (higher = more important)
    pub priority: Option<i32>,
    /// Labels/tags for categorization (e.g., "backend", "refactor", "bug")
    pub tags: Option<Vec<String>>,
    /// Acceptance criteria - conditions that must be met for completion
    pub acceptance_criteria: Option<Vec<String>>,
    /// Files expected to be modified
    pub affected_files: Option<Vec<String>>,
    /// Task IDs this task depends on
    pub depends_on: Option<Vec<Uuid>>,
    /// Steps/subtasks to complete this task
    pub steps: Option<Vec<CreateStepRequest>>,
    /// Estimated complexity (1-10)
    pub estimated_complexity: Option<u32>,
}

/// Request to create a new step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateStepRequest {
    pub description: String,
    pub verification: Option<String>,
}

/// Request to create a new constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateConstraintRequest {
    pub constraint_type: ConstraintType,
    pub description: String,
    pub enforced_by: Option<String>,
}

/// Request to update a task
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateTaskRequest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub status: Option<TaskStatus>,
    #[serde(default)]
    pub assigned_to: Option<String>,
    #[serde(default)]
    pub priority: Option<i32>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub acceptance_criteria: Option<Vec<String>>,
    #[serde(default)]
    pub affected_files: Option<Vec<String>>,
    #[serde(default)]
    pub actual_complexity: Option<u32>,
}

/// Request to add a step to a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddStepRequest {
    pub description: String,
    pub verification: Option<String>,
}

/// Request to update a step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStepRequest {
    pub description: Option<String>,
    pub status: Option<StepStatus>,
    pub verification: Option<String>,
}

/// Request to add a constraint to a plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddConstraintRequest {
    pub constraint_type: ConstraintType,
    pub description: String,
    pub enforced_by: Option<String>,
}

/// Request to record a decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateDecisionRequest {
    pub description: String,
    pub rationale: String,
    pub alternatives: Option<Vec<String>>,
    pub chosen_option: Option<String>,
}

/// Full plan details including tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanDetails {
    pub plan: PlanNode,
    pub tasks: Vec<TaskDetails>,
    pub constraints: Vec<ConstraintNode>,
}

/// Task details including steps and decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDetails {
    pub task: TaskNode,
    pub steps: Vec<StepNode>,
    pub decisions: Vec<DecisionNode>,
    pub depends_on: Vec<Uuid>,
    pub modifies_files: Vec<String>,
}

/// Agent context for executing a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContext {
    /// The task to execute
    pub task: TaskNode,

    /// Steps to complete
    pub steps: Vec<StepNode>,

    /// Plan constraints to respect
    pub constraints: Vec<ConstraintNode>,

    /// Related decisions already made
    pub decisions: Vec<DecisionNode>,

    /// Files this task will modify
    pub target_files: Vec<FileContext>,

    /// Similar code for reference
    pub similar_code: Vec<CodeReference>,

    /// Related past decisions for context
    pub related_decisions: Vec<DecisionNode>,
}

/// Context about a file to be modified
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileContext {
    pub path: String,
    pub language: String,
    /// Symbols (functions, structs) in this file
    pub symbols: Vec<String>,
    /// Files that import this file (will be impacted)
    pub dependent_files: Vec<String>,
    /// Files this file imports
    pub dependencies: Vec<String>,
}

/// Reference to similar code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeReference {
    pub path: String,
    pub snippet: String,
    pub relevance: f32,
}

impl PlanNode {
    /// Create a new plan node
    pub fn new(title: String, description: String, created_by: String, priority: i32) -> Self {
        Self {
            id: Uuid::new_v4(),
            title,
            description,
            status: PlanStatus::Draft,
            created_at: Utc::now(),
            created_by,
            priority,
            project_id: None,
        }
    }

    /// Create a new plan node for a specific project
    pub fn new_for_project(
        title: String,
        description: String,
        created_by: String,
        priority: i32,
        project_id: Uuid,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            title,
            description,
            status: PlanStatus::Draft,
            created_at: Utc::now(),
            created_by,
            priority,
            project_id: Some(project_id),
        }
    }
}

impl TaskNode {
    /// Create a new task node with minimal fields
    pub fn new(description: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            title: None,
            description,
            status: TaskStatus::Pending,
            assigned_to: None,
            priority: None,
            tags: vec![],
            acceptance_criteria: vec![],
            affected_files: vec![],
            estimated_complexity: None,
            actual_complexity: None,
            started_at: None,
            completed_at: None,
            created_at: Utc::now(),
        }
    }

    /// Create a new task node with all fields
    pub fn new_full(
        title: Option<String>,
        description: String,
        priority: Option<i32>,
        tags: Vec<String>,
        acceptance_criteria: Vec<String>,
        affected_files: Vec<String>,
        estimated_complexity: Option<u32>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            title,
            description,
            status: TaskStatus::Pending,
            assigned_to: None,
            priority,
            tags,
            acceptance_criteria,
            affected_files,
            estimated_complexity,
            actual_complexity: None,
            started_at: None,
            completed_at: None,
            created_at: Utc::now(),
        }
    }

    /// Check if task is available (pending and unassigned)
    pub fn is_available(&self) -> bool {
        self.status == TaskStatus::Pending && self.assigned_to.is_none()
    }
}

impl StepNode {
    /// Create a new step node
    pub fn new(order: u32, description: String, verification: Option<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            order,
            description,
            status: StepStatus::Pending,
            verification,
        }
    }
}

impl DecisionNode {
    /// Create a new decision node
    pub fn new(
        description: String,
        rationale: String,
        alternatives: Vec<String>,
        decided_by: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            description,
            rationale,
            alternatives,
            chosen_option: None,
            decided_by,
            decided_at: Utc::now(),
        }
    }
}

impl ConstraintNode {
    /// Create a new constraint node
    pub fn new(
        constraint_type: ConstraintType,
        description: String,
        enforced_by: Option<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            constraint_type,
            description,
            enforced_by,
        }
    }
}
