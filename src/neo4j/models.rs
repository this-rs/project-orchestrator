//! Neo4j graph models representing code structure and plans

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Project Node (multi-project support)
// ============================================================================

/// A project/codebase being tracked
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectNode {
    pub id: Uuid,
    pub name: String,
    pub slug: String, // URL-safe identifier
    pub root_path: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_synced: Option<DateTime<Utc>>,
}

// ============================================================================
// Code Structure Nodes (from Tree-sitter)
// ============================================================================

/// A source file in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileNode {
    pub path: String,
    pub language: String,
    pub hash: String,
    pub last_parsed: DateTime<Utc>,
    #[serde(default)]
    pub project_id: Option<Uuid>,
}

/// A module/namespace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleNode {
    pub name: String,
    pub path: String,
    pub visibility: Visibility,
    pub file_path: String,
}

/// A struct/class definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructNode {
    pub name: String,
    pub visibility: Visibility,
    pub generics: Vec<String>,
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub docstring: Option<String>,
}

/// A trait/interface definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitNode {
    pub name: String,
    pub visibility: Visibility,
    pub generics: Vec<String>,
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub docstring: Option<String>,
    /// Whether this trait is from an external crate (std, serde, etc.)
    #[serde(default)]
    pub is_external: bool,
    /// Source crate for external traits (e.g., "std", "serde", "tokio")
    #[serde(default)]
    pub source: Option<String>,
}

/// An enum definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumNode {
    pub name: String,
    pub visibility: Visibility,
    pub variants: Vec<String>,
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub docstring: Option<String>,
}

/// A function/method definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionNode {
    pub name: String,
    pub visibility: Visibility,
    pub params: Vec<Parameter>,
    pub return_type: Option<String>,
    pub generics: Vec<String>,
    pub is_async: bool,
    pub is_unsafe: bool,
    pub complexity: u32,
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub docstring: Option<String>,
}

/// A function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub type_name: Option<String>,
}

/// An impl block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplNode {
    pub for_type: String,
    pub trait_name: Option<String>,
    pub generics: Vec<String>,
    pub where_clause: Option<String>,
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
}

/// A field in a struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldNode {
    pub name: String,
    pub type_name: String,
    pub visibility: Visibility,
    pub default_value: Option<String>,
}

/// An import/use statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportNode {
    pub path: String,
    pub alias: Option<String>,
    pub items: Vec<String>,
    pub file_path: String,
    pub line: u32,
}

/// Visibility level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    Public,
    Private,
    Crate,
    Super,
    InPath(String),
}

impl Default for Visibility {
    fn default() -> Self {
        Self::Private
    }
}

// ============================================================================
// Plan Nodes (for coordination)
// ============================================================================

/// A development plan containing tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanNode {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub status: PlanStatus,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
    pub priority: i32,
    #[serde(default)]
    pub project_id: Option<Uuid>,
}

/// Status of a plan
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Draft,
    Approved,
    InProgress,
    Completed,
    Cancelled,
}

/// A task within a plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskNode {
    pub id: Uuid,
    /// Short title for the task
    pub title: Option<String>,
    /// Detailed description of what needs to be done
    pub description: String,
    pub status: TaskStatus,
    pub assigned_to: Option<String>,
    /// Priority (higher = more important)
    pub priority: Option<i32>,
    /// Labels/tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
    /// Acceptance criteria - conditions that must be met for task completion
    #[serde(default)]
    pub acceptance_criteria: Vec<String>,
    /// Files expected to be modified by this task
    #[serde(default)]
    pub affected_files: Vec<String>,
    pub estimated_complexity: Option<u32>,
    pub actual_complexity: Option<u32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Status of a task
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Blocked,
    Completed,
    Failed,
}

/// A task with its parent plan information (for global task queries)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskWithPlan {
    /// The task itself
    #[serde(flatten)]
    pub task: TaskNode,
    /// ID of the parent plan
    pub plan_id: Uuid,
    /// Title of the parent plan
    pub plan_title: String,
}

/// A step within a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepNode {
    pub id: Uuid,
    pub order: u32,
    pub description: String,
    pub status: StepStatus,
    pub verification: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Status of a step
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Pending,
    InProgress,
    Completed,
    Skipped,
}

/// An architectural decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionNode {
    pub id: Uuid,
    pub description: String,
    pub rationale: String,
    pub alternatives: Vec<String>,
    pub chosen_option: Option<String>,
    pub decided_by: String,
    pub decided_at: DateTime<Utc>,
}

/// A constraint on a plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintNode {
    pub id: Uuid,
    pub constraint_type: ConstraintType,
    pub description: String,
    pub enforced_by: Option<String>,
}

/// Type of constraint
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintType {
    Performance,
    Compatibility,
    Security,
    Style,
    Testing,
    Other,
}

/// An agent that executes tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentNode {
    pub id: String,
    pub name: String,
    pub agent_type: String,
    pub capabilities: Vec<String>,
    pub current_task: Option<Uuid>,
    pub last_active: DateTime<Utc>,
}

/// A git commit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitNode {
    pub hash: String,
    pub message: String,
    pub author: String,
    pub timestamp: DateTime<Utc>,
}

// ============================================================================
// Release and Milestone Nodes
// ============================================================================

/// A release/version of the project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseNode {
    pub id: Uuid,
    /// Version string (e.g., "1.0.0", "2.0.0-beta")
    pub version: String,
    /// Human-readable title (e.g., "Initial Release")
    pub title: Option<String>,
    pub description: Option<String>,
    pub status: ReleaseStatus,
    pub target_date: Option<DateTime<Utc>>,
    pub released_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub project_id: Uuid,
}

/// Status of a release
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseStatus {
    Planned,
    InProgress,
    Released,
    Cancelled,
}

/// A milestone in the roadmap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilestoneNode {
    pub id: Uuid,
    pub title: String,
    pub description: Option<String>,
    pub status: MilestoneStatus,
    pub target_date: Option<DateTime<Utc>>,
    pub closed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub project_id: Uuid,
}

/// Status of a milestone
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MilestoneStatus {
    Open,
    Closed,
}

// ============================================================================
// Relationship types
// ============================================================================

/// Types of relationships in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RelationType {
    // Project
    BelongsTo,

    // Code structure
    Contains,
    Imports,
    HasField,
    Implements,
    ImplementsFor,
    ImplementsTrait,
    Calls,
    UsesType,
    Returns,

    // Plans
    HasTask,
    HasStep,
    HasPlan,
    DependsOn,
    Modifies,
    InformedBy,
    ConstrainedBy,
    VerifiedBy,
    Affects,

    // History
    Executed,
    Made,
    ResultedIn,
}
