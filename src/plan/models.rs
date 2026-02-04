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
    /// Optional project ID to associate the plan with
    pub project_id: Option<Uuid>,
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

    /// Contextual notes (guidelines, gotchas, patterns) for the task
    #[serde(default)]
    pub notes: Vec<ContextNote>,
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
    /// Notes directly attached to this file or its entities
    #[serde(default)]
    pub notes: Vec<ContextNote>,
}

/// A note surfaced in context for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextNote {
    /// Note ID
    pub id: Uuid,
    /// Type of note (guideline, gotcha, pattern, etc.)
    pub note_type: String,
    /// Content of the note
    pub content: String,
    /// Importance level
    pub importance: String,
    /// Source entity the note is attached to
    pub source_entity: String,
    /// Whether this note was propagated (vs directly attached)
    pub propagated: bool,
    /// Relevance score (1.0 = direct, < 1.0 = propagated)
    pub relevance_score: f64,
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
        let now = Utc::now();
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
            created_at: now,
            updated_at: Some(now),
            started_at: None,
            completed_at: None,
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
        let now = Utc::now();
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
            created_at: now,
            updated_at: Some(now),
            started_at: None,
            completed_at: None,
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
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            order,
            description,
            status: StepStatus::Pending,
            verification,
            created_at: now,
            updated_at: Some(now),
            completed_at: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PlanNode Tests
    // =========================================================================

    #[test]
    fn test_plan_node_new() {
        let plan = PlanNode::new(
            "Test Plan".to_string(),
            "A test plan description".to_string(),
            "test-user".to_string(),
            5,
        );

        assert_eq!(plan.title, "Test Plan");
        assert_eq!(plan.description, "A test plan description");
        assert_eq!(plan.created_by, "test-user");
        assert_eq!(plan.priority, 5);
        assert_eq!(plan.status, PlanStatus::Draft);
        assert!(plan.project_id.is_none());
    }

    #[test]
    fn test_plan_node_new_for_project() {
        let project_id = Uuid::new_v4();
        let plan = PlanNode::new_for_project(
            "Project Plan".to_string(),
            "Plan linked to project".to_string(),
            "architect".to_string(),
            10,
            project_id,
        );

        assert_eq!(plan.title, "Project Plan");
        assert_eq!(plan.priority, 10);
        assert_eq!(plan.project_id, Some(project_id));
    }

    #[test]
    fn test_plan_node_serialization() {
        let plan = PlanNode::new(
            "Serialize Test".to_string(),
            "Description".to_string(),
            "user".to_string(),
            3,
        );

        let json = serde_json::to_string(&plan).unwrap();
        let deserialized: PlanNode = serde_json::from_str(&json).unwrap();

        assert_eq!(plan.id, deserialized.id);
        assert_eq!(plan.title, deserialized.title);
        assert_eq!(plan.status, deserialized.status);
    }

    // =========================================================================
    // TaskNode Tests
    // =========================================================================

    #[test]
    fn test_task_node_new_minimal() {
        let task = TaskNode::new("Implement feature".to_string());

        assert_eq!(task.description, "Implement feature");
        assert_eq!(task.status, TaskStatus::Pending);
        assert!(task.title.is_none());
        assert!(task.assigned_to.is_none());
        assert!(task.priority.is_none());
        assert!(task.tags.is_empty());
        assert!(task.acceptance_criteria.is_empty());
        assert!(task.affected_files.is_empty());
    }

    #[test]
    fn test_task_node_new_full() {
        let task = TaskNode::new_full(
            Some("Feature X".to_string()),
            "Implement feature X with tests".to_string(),
            Some(8),
            vec!["backend".to_string(), "api".to_string()],
            vec!["Tests pass".to_string(), "No regressions".to_string()],
            vec!["src/main.rs".to_string(), "src/lib.rs".to_string()],
            Some(5),
        );

        assert_eq!(task.title, Some("Feature X".to_string()));
        assert_eq!(task.priority, Some(8));
        assert_eq!(task.tags.len(), 2);
        assert!(task.tags.contains(&"backend".to_string()));
        assert_eq!(task.acceptance_criteria.len(), 2);
        assert_eq!(task.affected_files.len(), 2);
        assert_eq!(task.estimated_complexity, Some(5));
    }

    #[test]
    fn test_task_node_is_available_when_pending_and_unassigned() {
        let task = TaskNode::new("Available task".to_string());
        assert!(task.is_available());
    }

    #[test]
    fn test_task_node_not_available_when_assigned() {
        let mut task = TaskNode::new("Assigned task".to_string());
        task.assigned_to = Some("agent-1".to_string());
        assert!(!task.is_available());
    }

    #[test]
    fn test_task_node_not_available_when_in_progress() {
        let mut task = TaskNode::new("In progress task".to_string());
        task.status = TaskStatus::InProgress;
        assert!(!task.is_available());
    }

    #[test]
    fn test_task_node_not_available_when_completed() {
        let mut task = TaskNode::new("Completed task".to_string());
        task.status = TaskStatus::Completed;
        assert!(!task.is_available());
    }

    #[test]
    fn test_task_node_serialization() {
        let task = TaskNode::new_full(
            Some("Test".to_string()),
            "Description".to_string(),
            Some(5),
            vec!["tag".to_string()],
            vec!["criterion".to_string()],
            vec!["file.rs".to_string()],
            Some(3),
        );

        let json = serde_json::to_string(&task).unwrap();
        let deserialized: TaskNode = serde_json::from_str(&json).unwrap();

        assert_eq!(task.id, deserialized.id);
        assert_eq!(task.tags, deserialized.tags);
        assert_eq!(task.estimated_complexity, deserialized.estimated_complexity);
    }

    // =========================================================================
    // StepNode Tests
    // =========================================================================

    #[test]
    fn test_step_node_new() {
        let step = StepNode::new(
            0,
            "First step".to_string(),
            Some("Check file exists".to_string()),
        );

        assert_eq!(step.order, 0);
        assert_eq!(step.description, "First step");
        assert_eq!(step.status, StepStatus::Pending);
        assert_eq!(step.verification, Some("Check file exists".to_string()));
        assert!(step.completed_at.is_none());
    }

    #[test]
    fn test_step_node_without_verification() {
        let step = StepNode::new(1, "Second step".to_string(), None);

        assert_eq!(step.order, 1);
        assert!(step.verification.is_none());
    }

    #[test]
    fn test_step_node_serialization() {
        let step = StepNode::new(
            2,
            "Step description".to_string(),
            Some("Verify".to_string()),
        );

        let json = serde_json::to_string(&step).unwrap();
        let deserialized: StepNode = serde_json::from_str(&json).unwrap();

        assert_eq!(step.id, deserialized.id);
        assert_eq!(step.order, deserialized.order);
        assert_eq!(step.verification, deserialized.verification);
    }

    // =========================================================================
    // DecisionNode Tests
    // =========================================================================

    #[test]
    fn test_decision_node_new() {
        let decision = DecisionNode::new(
            "Use async/await".to_string(),
            "Better for I/O operations".to_string(),
            vec!["threads".to_string(), "callbacks".to_string()],
            "architect".to_string(),
        );

        assert_eq!(decision.description, "Use async/await");
        assert_eq!(decision.rationale, "Better for I/O operations");
        assert_eq!(decision.alternatives.len(), 2);
        assert!(decision.alternatives.contains(&"threads".to_string()));
        assert!(decision.chosen_option.is_none());
        assert_eq!(decision.decided_by, "architect");
    }

    #[test]
    fn test_decision_node_empty_alternatives() {
        let decision = DecisionNode::new(
            "Simple decision".to_string(),
            "No alternatives needed".to_string(),
            vec![],
            "user".to_string(),
        );

        assert!(decision.alternatives.is_empty());
    }

    #[test]
    fn test_decision_node_serialization() {
        let decision = DecisionNode::new(
            "Decision".to_string(),
            "Rationale".to_string(),
            vec!["alt1".to_string()],
            "decider".to_string(),
        );

        let json = serde_json::to_string(&decision).unwrap();
        let deserialized: DecisionNode = serde_json::from_str(&json).unwrap();

        assert_eq!(decision.id, deserialized.id);
        assert_eq!(decision.alternatives, deserialized.alternatives);
    }

    // =========================================================================
    // ConstraintNode Tests
    // =========================================================================

    #[test]
    fn test_constraint_node_new_with_enforced_by() {
        let constraint = ConstraintNode::new(
            ConstraintType::Security,
            "Must sanitize user input".to_string(),
            Some("clippy::security".to_string()),
        );

        assert_eq!(constraint.constraint_type, ConstraintType::Security);
        assert_eq!(constraint.description, "Must sanitize user input");
        assert_eq!(constraint.enforced_by, Some("clippy::security".to_string()));
    }

    #[test]
    fn test_constraint_node_new_without_enforced_by() {
        let constraint = ConstraintNode::new(
            ConstraintType::Performance,
            "Response time under 100ms".to_string(),
            None,
        );

        assert_eq!(constraint.constraint_type, ConstraintType::Performance);
        assert!(constraint.enforced_by.is_none());
    }

    #[test]
    fn test_constraint_node_all_types() {
        let types = vec![
            ConstraintType::Security,
            ConstraintType::Performance,
            ConstraintType::Style,
            ConstraintType::Compatibility,
            ConstraintType::Other,
        ];

        for constraint_type in types {
            let constraint = ConstraintNode::new(constraint_type.clone(), "test".to_string(), None);
            assert_eq!(constraint.constraint_type, constraint_type);
        }
    }

    #[test]
    fn test_constraint_node_serialization() {
        let constraint = ConstraintNode::new(
            ConstraintType::Style,
            "Follow Rust conventions".to_string(),
            Some("rustfmt".to_string()),
        );

        let json = serde_json::to_string(&constraint).unwrap();
        let deserialized: ConstraintNode = serde_json::from_str(&json).unwrap();

        assert_eq!(constraint.id, deserialized.id);
        assert_eq!(constraint.constraint_type, deserialized.constraint_type);
    }

    // =========================================================================
    // DTO Serialization Tests
    // =========================================================================

    #[test]
    fn test_create_plan_request_serialization() {
        let request = CreatePlanRequest {
            title: "New Plan".to_string(),
            description: "Plan description".to_string(),
            priority: Some(5),
            constraints: None,
            project_id: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: CreatePlanRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(request.title, deserialized.title);
        assert_eq!(request.priority, deserialized.priority);
    }

    #[test]
    fn test_create_task_request_serialization() {
        let request = CreateTaskRequest {
            title: Some("Task Title".to_string()),
            description: "Task description".to_string(),
            priority: Some(8),
            tags: Some(vec!["backend".to_string()]),
            acceptance_criteria: Some(vec!["Tests pass".to_string()]),
            affected_files: Some(vec!["src/main.rs".to_string()]),
            depends_on: None,
            steps: None,
            estimated_complexity: Some(4),
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: CreateTaskRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(request.title, deserialized.title);
        assert_eq!(request.tags, deserialized.tags);
    }

    #[test]
    fn test_update_task_request_partial() {
        // Test that UpdateTaskRequest works with partial fields
        let request = UpdateTaskRequest {
            status: Some(TaskStatus::InProgress),
            ..Default::default()
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: UpdateTaskRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.status, Some(TaskStatus::InProgress));
        assert!(deserialized.title.is_none());
        assert!(deserialized.assigned_to.is_none());
    }

    #[test]
    fn test_agent_context_serialization() {
        let context = AgentContext {
            task: TaskNode::new("Test task".to_string()),
            steps: vec![StepNode::new(0, "Step 1".to_string(), None)],
            constraints: vec![],
            decisions: vec![],
            target_files: vec![],
            similar_code: vec![],
            related_decisions: vec![],
            notes: vec![],
        };

        let json = serde_json::to_string(&context).unwrap();
        let deserialized: AgentContext = serde_json::from_str(&json).unwrap();

        assert_eq!(context.task.id, deserialized.task.id);
        assert_eq!(context.steps.len(), deserialized.steps.len());
    }

    #[test]
    fn test_file_context_serialization() {
        let context = FileContext {
            path: "src/test.rs".to_string(),
            language: "rust".to_string(),
            symbols: vec!["func1".to_string(), "Struct1".to_string()],
            dependent_files: vec!["src/other.rs".to_string()],
            dependencies: vec!["src/lib.rs".to_string()],
            notes: vec![],
        };

        let json = serde_json::to_string(&context).unwrap();
        let deserialized: FileContext = serde_json::from_str(&json).unwrap();

        assert_eq!(context.path, deserialized.path);
        assert_eq!(context.symbols, deserialized.symbols);
    }

    #[test]
    fn test_code_reference_serialization() {
        let reference = CodeReference {
            path: "src/example.rs".to_string(),
            snippet: "fn example() {}".to_string(),
            relevance: 0.85,
        };

        let json = serde_json::to_string(&reference).unwrap();
        let deserialized: CodeReference = serde_json::from_str(&json).unwrap();

        assert_eq!(reference.path, deserialized.path);
        assert!((reference.relevance - deserialized.relevance).abs() < 0.001);
    }
}
