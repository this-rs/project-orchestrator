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
    /// When GDS analytics (PageRank, Louvain, etc.) were last computed for this project.
    /// None if analytics have never been computed.
    pub analytics_computed_at: Option<DateTime<Utc>>,
    /// When CO_CHANGED relations were last computed from TOUCHES history.
    /// Used for incremental computation — only new commits since this date are processed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_co_change_computed_at: Option<DateTime<Utc>>,
    /// Manual scaffolding level override (0-4). When set, bypasses auto-computation.
    /// Biomimicry T8: allows forcing a specific cognitive level.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub scaffolding_override: Option<u8>,
    /// Sharing policy for P2P knowledge exchange (RFC Privacy §2.4).
    /// Serialized as JSON string in Neo4j. None = sharing disabled (opt-in default).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sharing_policy: Option<crate::episodes::distill_models::SharingPolicy>,
}

// ============================================================================
// Workspace Node (multi-project grouping)
// ============================================================================

/// A workspace that groups related projects together
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceNode {
    pub id: Uuid,
    pub name: String,
    /// URL-safe unique identifier
    pub slug: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
    /// Extensible metadata as JSON
    #[serde(default)]
    pub metadata: serde_json::Value,
}

// ============================================================================
// Chat Session Node
// ============================================================================

/// A chat session with Claude Code CLI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSessionNode {
    pub id: Uuid,
    /// Claude CLI session ID (for --resume)
    pub cli_session_id: Option<String>,
    /// Associated project slug
    pub project_slug: Option<String>,
    /// Associated workspace slug (if session spans a workspace)
    #[serde(default)]
    pub workspace_slug: Option<String>,
    /// Working directory
    pub cwd: String,
    /// Session title (auto-generated or user-provided)
    pub title: Option<String>,
    /// Model used
    pub model: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Number of messages exchanged
    #[serde(default)]
    pub message_count: i64,
    /// Total cost in USD
    #[serde(default)]
    pub total_cost_usd: Option<f64>,
    /// Nexus conversation ID (for message history retrieval)
    #[serde(default)]
    pub conversation_id: Option<String>,
    /// Preview text (first user message, truncated to ~200 chars)
    #[serde(default)]
    pub preview: Option<String>,
    /// Permission mode override for this session (None = use global config)
    #[serde(default)]
    pub permission_mode: Option<String>,
    /// Additional directories exposed to Claude CLI (serialized as JSON array string in Neo4j)
    #[serde(default)]
    pub add_dirs: Option<Vec<String>>,
    /// Origin of the session — JSON string stored in Neo4j.
    /// Pattern: `{"type":"runner","run_id":"...","plan_id":"..."}` for PlanRunner sessions.
    /// None/empty for normal user-initiated sessions.
    #[serde(default)]
    pub spawned_by: Option<String>,
}

// ============================================================================
// Session Tree Node (for discussion graph traversal)
// ============================================================================

/// A node in a session tree, representing a session and its position in
/// the SPAWNED_BY hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTreeNode {
    pub session_id: String,
    pub parent_session_id: Option<String>,
    pub spawn_type: Option<String>,
    pub run_id: Option<Uuid>,
    pub task_id: Option<Uuid>,
    pub depth: u32,
    pub created_at: Option<DateTime<Utc>>,
}

/// Lightweight session info for run-scoped queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub title: Option<String>,
    pub model: String,
    pub spawn_type: Option<String>,
    pub task_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

// ============================================================================
// Chat Event Record (for WebSocket replay & persistence)
// ============================================================================

/// A persisted chat event with sequence number for replay support.
///
/// Events are stored as `(:ChatSession)-[:HAS_EVENT]->(:ChatEvent)` in Neo4j.
/// `stream_delta` events are NOT persisted (accumulated in memory, stored as
/// a single `assistant_text` at the end).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatEventRecord {
    pub id: Uuid,
    /// Session this event belongs to
    pub session_id: Uuid,
    /// Monotonically increasing sequence number (per session)
    pub seq: i64,
    /// Event type: "user_message", "assistant_text", "thinking", "tool_use",
    /// "tool_result", "permission_request", "input_request", "result", "error"
    pub event_type: String,
    /// JSON-serialized event payload
    pub data: String,
    /// When this event was created
    pub created_at: DateTime<Utc>,
}

/// An entity discussed in a chat session (via DISCUSSED relation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscussedEntity {
    /// Entity type: "File", "Function", "Struct", "Trait", "Enum"
    pub entity_type: String,
    /// Entity identifier: file path (for File) or symbol name
    pub entity_id: String,
    /// Number of times the entity was mentioned in the session
    pub mention_count: i64,
    /// When the entity was last mentioned
    pub last_mentioned_at: Option<String>,
    /// Source file path (for symbols, not for File entities)
    pub file_path: Option<String>,
}

// ============================================================================
// Graph visualization — PM layer lightweight structs
// ============================================================================

/// A PM entity for the graph visualization (plan, task, step, milestone, release)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmGraphNode {
    pub id: String,
    pub node_type: String,
    pub label: String,
    pub attributes: serde_json::Value,
}

/// A PM edge for the graph visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmGraphEdge {
    pub source: String,
    pub target: String,
    pub rel_type: String,
    pub attributes: Option<serde_json::Value>,
}

/// A chat session for the graph visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatGraphSession {
    pub id: String,
    pub title: String,
    pub model: Option<String>,
    pub message_count: i64,
    pub total_cost_usd: f64,
    pub created_at: String,
}

/// A DISCUSSED edge for the graph visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatGraphDiscussed {
    pub session_id: String,
    pub entity_type: String,
    pub entity_id: String,
    pub mention_count: i64,
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
    /// Parent class name (single inheritance: Java, Python, PHP, Ruby, etc.)
    #[serde(default)]
    pub parent_class: Option<String>,
    /// Implemented interfaces/protocols (Java implements, TS implements, Swift protocol conformance, etc.)
    #[serde(default)]
    pub interfaces: Vec<String>,
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    Public,
    #[default]
    Private,
    Crate,
    Super,
    InPath(String),
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
    /// Pre-enriched execution context (JSON) — cached by plan enrich action.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_context: Option<String>,
    /// Pre-enriched persona profile (JSON) — cached by plan enrich action.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persona: Option<String>,
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
    /// Frustration score (0.0-1.0) — bio-inspired adaptive stress signal.
    /// Accumulates on blocked/failure events, decays on step completion.
    #[serde(default)]
    pub frustration_score: f64,
    /// Pre-enriched execution context (JSON) — cached by plan/task enrich action.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_context: Option<String>,
    /// Pre-enriched persona profile (JSON) — cached by plan/task enrich action.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persona: Option<String>,
    /// Pre-built prompt cache — full prompt ready for the runner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache: Option<String>,
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
    /// Status of the parent plan (e.g. "draft", "in_progress", "completed")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_status: Option<String>,
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
    /// Pre-enriched execution context (JSON) — cached by plan/task enrich action.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_context: Option<String>,
    /// Pre-enriched persona profile (JSON) — cached by plan/task enrich action.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persona: Option<String>,
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

/// Decision status lifecycle: proposed → accepted → deprecated/superseded
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionStatus {
    /// Newly created, not yet validated
    Proposed,
    /// Validated and active
    Accepted,
    /// No longer recommended but not replaced
    Deprecated,
    /// Replaced by another decision (via SUPERSEDES relation)
    Superseded,
}

impl std::fmt::Display for DecisionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Proposed => write!(f, "proposed"),
            Self::Accepted => write!(f, "accepted"),
            Self::Deprecated => write!(f, "deprecated"),
            Self::Superseded => write!(f, "superseded"),
        }
    }
}

impl std::str::FromStr for DecisionStatus {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "proposed" => Ok(Self::Proposed),
            "accepted" => Ok(Self::Accepted),
            "deprecated" => Ok(Self::Deprecated),
            "superseded" => Ok(Self::Superseded),
            _ => Err(anyhow::anyhow!("Invalid decision status: {}", s)),
        }
    }
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
    /// Lifecycle status: proposed, accepted, deprecated, superseded
    #[serde(default = "default_decision_status")]
    pub status: DecisionStatus,
    /// Vector embedding (768d, nomic-embed-text) for semantic search.
    /// Stored via `db.create.setNodeVectorProperty` for HNSW index compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f64>>,
    /// Name of the embedding model used (for traceability on re-embed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    /// Scar intensity from negative reasoning feedback (0.0 = no scar, 1.0 = max).
    /// Biomimicry: Elun HypersphereIdentity.Scar — penalizes decisions in search scoring.
    #[serde(default)]
    pub scar_intensity: f64,
}

fn default_decision_status() -> DecisionStatus {
    DecisionStatus::Proposed
}

/// An AFFECTS relation from a Decision to an entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectsRelation {
    /// Type of the affected entity (File, Function, Struct, Trait, etc.)
    pub entity_type: String,
    /// Identifier of the affected entity (path for File, id for others)
    pub entity_id: String,
    /// Human-readable name of the entity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_name: Option<String>,
    /// Optional description of how the decision impacts this entity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub impact_description: Option<String>,
}

/// A decision in a timeline view, with its supersession chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTimelineEntry {
    pub decision: DecisionNode,
    /// IDs of decisions this one supersedes (chain from newest to oldest)
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub supersedes_chain: Vec<Uuid>,
    /// ID of the decision that supersedes this one (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<Uuid>,
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

/// Info about a file changed in a commit (for TOUCHES relations).
///
/// Supports two JSON input formats (backward compatible):
/// - String: `"src/main.rs"` → `FileChangedInfo { path: "src/main.rs", additions: None, deletions: None }`
/// - Object: `{ "path": "src/main.rs", "additions": 10, "deletions": 3 }`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChangedInfo {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additions: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deletions: Option<i64>,
}

impl From<String> for FileChangedInfo {
    fn from(path: String) -> Self {
        Self {
            path,
            additions: None,
            deletions: None,
        }
    }
}

impl From<&str> for FileChangedInfo {
    fn from(path: &str) -> Self {
        Self {
            path: path.to_string(),
            additions: None,
            deletions: None,
        }
    }
}

/// Info about a file touched by a commit (returned by get_commit_files)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitFileInfo {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additions: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deletions: Option<i64>,
}

/// A commit in the history of a file (returned by get_file_history)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileHistoryEntry {
    pub hash: String,
    pub message: String,
    pub author: String,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additions: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deletions: Option<i64>,
}

/// A pair of files that co-change (returned by get_co_change_graph)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoChangePair {
    pub file_a: String,
    pub file_b: String,
    pub count: i64,
    pub last_at: Option<String>,
}

/// A file that co-changes with a given file (returned by get_file_co_changers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoChanger {
    pub path: String,
    pub count: i64,
    pub last_at: Option<String>,
}

/// Deserializes a list of file changes that can be either strings or objects.
/// This allows backward-compatible API: `["a.rs", "b.rs"]` or `[{"path": "a.rs", "additions": 10}]`
pub fn deserialize_files_changed<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<FileChangedInfo>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum FileEntry {
        Simple(String),
        Detailed(FileChangedInfo),
    }

    let opt: Option<Vec<FileEntry>> = Option::deserialize(deserializer)?;
    Ok(opt.map(|entries| {
        entries
            .into_iter()
            .map(|e| match e {
                FileEntry::Simple(path) => FileChangedInfo::from(path),
                FileEntry::Detailed(info) => info,
            })
            .collect()
    }))
}

// ============================================================================
// Release and Milestone Nodes
// ============================================================================

/// A detected business process (entry point → terminal through CALLS graph).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessNode {
    pub id: String,
    pub label: String,
    /// "intra_community" or "cross_community"
    pub process_type: String,
    pub step_count: u32,
    pub entry_point_id: String,
    pub terminal_id: String,
    pub communities: Vec<u32>,
    pub project_id: Option<Uuid>,
}

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
    Planned,
    Open,
    InProgress,
    Completed,
    Closed,
}

// ============================================================================
// Workspace Milestone Node (cross-project coordination)
// ============================================================================

/// A milestone that spans multiple projects in a workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceMilestoneNode {
    pub id: Uuid,
    pub workspace_id: Uuid,
    pub title: String,
    pub description: Option<String>,
    pub status: MilestoneStatus,
    pub target_date: Option<DateTime<Utc>>,
    pub closed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub tags: Vec<String>,
}

// ============================================================================
// Resource Node (shared contracts/specs)
// ============================================================================

/// Type of shared resource
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResourceType {
    /// OpenAPI, Swagger, RAML
    ApiContract,
    /// Protocol buffers
    Protobuf,
    /// GraphQL schema
    GraphqlSchema,
    /// JSON Schema
    JsonSchema,
    /// Database migrations, DDL
    DatabaseSchema,
    /// Shared type definitions
    SharedTypes,
    /// Configuration files
    Config,
    /// Technical documentation
    Documentation,
    /// Other resource type
    Other,
}

impl std::str::FromStr for ResourceType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().replace("_", "").as_str() {
            "apicontract" | "api_contract" => Ok(ResourceType::ApiContract),
            "protobuf" => Ok(ResourceType::Protobuf),
            "graphqlschema" | "graphql_schema" | "graphql" => Ok(ResourceType::GraphqlSchema),
            "jsonschema" | "json_schema" => Ok(ResourceType::JsonSchema),
            "databaseschema" | "database_schema" | "database" => Ok(ResourceType::DatabaseSchema),
            "sharedtypes" | "shared_types" => Ok(ResourceType::SharedTypes),
            "config" => Ok(ResourceType::Config),
            "documentation" | "docs" => Ok(ResourceType::Documentation),
            "other" => Ok(ResourceType::Other),
            _ => Err(format!("Unknown ResourceType: {}", s)),
        }
    }
}

/// A shared resource (contract, spec, schema) referenced by file path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceNode {
    pub id: Uuid,
    /// Workspace that owns this resource (if workspace-level)
    pub workspace_id: Option<Uuid>,
    /// Project that owns this resource (if project-level)
    pub project_id: Option<Uuid>,
    pub name: String,
    pub resource_type: ResourceType,
    /// Path to the spec file
    pub file_path: String,
    /// External URL (optional)
    pub url: Option<String>,
    /// Format hint (e.g., "openapi", "protobuf", "graphql")
    pub format: Option<String>,
    /// Version of the resource
    pub version: Option<String>,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
    /// Extensible metadata as JSON
    #[serde(default)]
    pub metadata: serde_json::Value,
}

// ============================================================================
// Component Node (deployment topology)
// ============================================================================

/// Type of deployment component
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComponentType {
    /// Backend service/API
    Service,
    /// Frontend application
    Frontend,
    /// Background worker
    Worker,
    /// Database
    Database,
    /// Message queue (RabbitMQ, Kafka, etc.)
    MessageQueue,
    /// Cache (Redis, Memcached, etc.)
    Cache,
    /// API Gateway / Load balancer
    Gateway,
    /// External service (third-party)
    External,
    /// Other component type
    Other,
}

impl std::str::FromStr for ComponentType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().replace("_", "").as_str() {
            "service" => Ok(ComponentType::Service),
            "frontend" => Ok(ComponentType::Frontend),
            "worker" => Ok(ComponentType::Worker),
            "database" | "db" => Ok(ComponentType::Database),
            "messagequeue" | "message_queue" | "queue" => Ok(ComponentType::MessageQueue),
            "cache" => Ok(ComponentType::Cache),
            "gateway" | "apigateway" | "api_gateway" => Ok(ComponentType::Gateway),
            "external" => Ok(ComponentType::External),
            "other" => Ok(ComponentType::Other),
            _ => Err(format!("Unknown ComponentType: {}", s)),
        }
    }
}

/// A component in the deployment topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentNode {
    pub id: Uuid,
    pub workspace_id: Uuid,
    pub name: String,
    pub component_type: ComponentType,
    pub description: Option<String>,
    /// Runtime environment (e.g., "docker", "kubernetes", "lambda")
    pub runtime: Option<String>,
    /// Configuration (env vars, ports, etc.) as JSON
    #[serde(default)]
    pub config: serde_json::Value,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// A dependency between components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentDependency {
    pub from_id: Uuid,
    pub to_id: Uuid,
    /// Communication protocol (e.g., "http", "grpc", "amqp", "tcp")
    pub protocol: Option<String>,
    /// Whether this dependency is required for the component to function
    #[serde(default = "default_true")]
    pub required: bool,
}

fn default_true() -> bool {
    true
}

// ============================================================================
// Code exploration result types
// ============================================================================

/// Summary of a function for code exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSummaryNode {
    pub name: String,
    pub signature: String,
    pub line: u32,
    pub is_async: bool,
    pub is_public: bool,
    pub complexity: u32,
    pub docstring: Option<String>,
}

/// Summary of a struct for code exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructSummaryNode {
    pub name: String,
    pub line: u32,
    pub is_public: bool,
    pub docstring: Option<String>,
}

/// A reference to a symbol found in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolReferenceNode {
    pub file_path: String,
    pub line: u32,
    pub context: String,
    pub reference_type: String,
}

/// Language statistics for architecture overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageStatsNode {
    pub language: String,
    pub file_count: usize,
}

/// Trait metadata from the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitInfoNode {
    pub is_external: bool,
    pub source: Option<String>,
}

/// A type that implements a trait
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitImplementorNode {
    pub type_name: String,
    pub file_path: String,
    pub line: u32,
}

/// Trait info for a type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeTraitInfoNode {
    pub name: String,
    pub full_path: Option<String>,
    pub file_path: String,
    pub is_external: bool,
    pub source: Option<String>,
}

/// Impl block detail with methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplBlockDetailNode {
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub trait_name: Option<String>,
    pub methods: Vec<String>,
}

/// File import info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileImportNode {
    pub path: String,
    pub language: String,
}

/// Aggregated symbol names for a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSymbolNamesNode {
    pub functions: Vec<String>,
    pub structs: Vec<String>,
    pub traits: Vec<String>,
    pub enums: Vec<String>,
}

/// A file with connection counts (imports + dependents) and optional GDS analytics scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectedFileNode {
    pub path: String,
    pub imports: i64,
    pub dependents: i64,
    /// PageRank score from graph analytics (higher = more structurally important)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pagerank: Option<f64>,
    /// Betweenness centrality score (higher = more bridge-like)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub betweenness: Option<f64>,
    /// Human-readable community label (e.g., "MCP Tool Pipeline")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub community_label: Option<String>,
    /// Numeric community identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub community_id: Option<i64>,
}

/// GDS analytics properties for a single node (File or Function).
/// Returned by `get_node_analytics()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAnalyticsRow {
    pub pagerank: Option<f64>,
    pub betweenness: Option<f64>,
    pub community_id: Option<i64>,
    pub community_label: Option<String>,
}

/// A community row returned by `get_project_communities()`.
/// Represents a Louvain community detected by graph analytics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityRow {
    /// Numeric community identifier
    pub community_id: i64,
    /// Human-readable community label
    pub community_label: String,
    /// Number of files in this community
    pub file_count: usize,
    /// Top files in this community (by pagerank, up to 3)
    pub key_files: Vec<String>,
    /// Number of unique WL hash fingerprints in this community (structural diversity)
    pub unique_fingerprints: usize,
}

/// A "god function" — a function with too many callers/callees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GodFunction {
    pub name: String,
    pub file: String,
    pub in_degree: usize,
    pub out_degree: usize,
}

/// Structural health report for a project's codebase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeHealthReport {
    pub god_functions: Vec<GodFunction>,
    pub orphan_files: Vec<String>,
    pub coupling_metrics: Option<CouplingMetrics>,
    /// WorldModel prediction accuracy (biomimicry T7)
    pub prediction_accuracy: Option<PredictionAccuracy>,
}

/// WorldModel prediction accuracy metrics (biomimicry T7).
/// Measures how well CO_CHANGED/DISCUSSED patterns predict file access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionAccuracy {
    /// Number of correctly predicted file accesses
    pub hits: i64,
    /// Total file accesses measured
    pub total: i64,
    /// Hit rate (hits / total), or 0.0 if total == 0
    pub accuracy: f64,
    /// Number of sessions analyzed
    pub sessions_analyzed: i64,
}

/// Snapshot of key health metrics before/after a maintenance cycle (biomimicry T11).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceSnapshot {
    /// Composite health score (god_functions penalty + orphan penalty + coupling bonus)
    pub health_score: f64,
    /// Number of synapses with strength > 0
    pub active_synapses: i64,
    /// Average energy across all notes
    pub mean_energy: f64,
    /// Total number of active skills
    pub skill_count: i64,
    /// Total number of active notes
    pub note_count: i64,
    /// Timestamp of the snapshot
    pub captured_at: String,
}

/// Report comparing pre/post maintenance snapshots (biomimicry T11).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceReport {
    pub before: MaintenanceSnapshot,
    pub after: MaintenanceSnapshot,
    /// Delta for each metric (after - before)
    pub delta_health_score: f64,
    pub delta_active_synapses: i64,
    pub delta_mean_energy: f64,
    pub delta_skill_count: i64,
    pub delta_note_count: i64,
    /// Success rate: fraction of metrics that improved or stayed stable
    pub success_rate: f64,
    /// Maintenance level that was run
    pub maintenance_level: String,
    /// Duration in milliseconds
    pub duration_ms: u64,
}

/// Scaffolding level for adaptive task complexity (biomimicry T8).
/// Inspired by Elun's DifficultyAdjustment and 5 cognitive levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaffoldingLevel {
    /// Level 0-4 (L0=reflexe, L1=associatif, L2=contextuel, L3=stratégique, L4=méta-cognitif)
    pub level: u8,
    /// Human-readable label
    pub label: String,
    /// Recommended steps per task at this level
    pub recommended_steps: String,
    /// Task success rate over last N tasks (completed / (completed + failed))
    pub task_success_rate: f64,
    /// Average frustration score across recent tasks (0.0-1.0)
    pub avg_frustration: f64,
    /// Scar density: average scar_intensity across project notes
    pub scar_density: f64,
    /// Homeostatic pain: fraction of homeostasis ratios out of equilibrium
    pub homeostasis_pain: f64,
    /// Composite competence score (0.0-1.0) combining all metrics
    pub competence_score: f64,
    /// Whether the level was manually overridden
    pub is_overridden: bool,
    /// Number of tasks analyzed
    pub tasks_analyzed: i64,
}

/// Deep maintenance report — aggressive cleanup when stagnation is detected (biomimicry T12).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepMaintenanceReport {
    /// Stagnation report that triggered deep maintenance
    pub stagnation: StagnationReport,
    /// Regular maintenance result from "full" level
    pub maintenance: Option<serde_json::Value>,
    /// Number of stale notes flagged for review
    pub stale_notes_flagged: usize,
    /// Number of stuck tasks identified (in_progress > 48h)
    pub stuck_tasks_found: usize,
    /// Recommendations generated
    pub recommendations: Vec<String>,
}

/// Global stagnation report — detects when an entire project is stuck (biomimicry T12).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagnationReport {
    /// Whether global stagnation is detected
    pub is_stagnating: bool,
    /// Number of tasks completed in the last 48h
    pub tasks_completed_48h: i64,
    /// Average frustration across in-progress tasks (0.0-1.0)
    pub avg_frustration: f64,
    /// Mean note energy trend: negative = declining
    pub energy_trend: f64,
    /// Number of new TOUCHES commits in the last 48h
    pub commits_48h: i64,
    /// Number of stagnation signals triggered (0-4)
    pub signals_triggered: u8,
    /// Human-readable recommendations
    pub recommendations: Vec<String>,
}

/// Coupling metrics from clustering coefficients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingMetrics {
    pub avg_clustering_coefficient: f64,
    pub max_clustering_coefficient: f64,
    pub most_coupled_file: Option<String>,
}

/// Knowledge graph audit report — gaps found in entity relations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditGapsReport {
    /// Total gaps found across all categories
    pub total_gaps: usize,
    /// Notes without any LINKED_TO relations to code entities
    pub orphan_notes: Vec<String>,
    /// Decisions without AFFECTS relations
    pub decisions_without_affects: Vec<String>,
    /// Commits without TOUCHES relations to files
    pub commits_without_touches: Vec<String>,
    /// Skills without HAS_MEMBER relations
    pub skills_without_members: Vec<String>,
    /// Relationship types in the graph with their counts
    pub relationship_type_counts: Vec<RelTypeCount>,
}

/// A relationship type and its count in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelTypeCount {
    pub rel_type: String,
    pub count: i64,
}

// ============================================================================
// Homeostasis Report (biomimicry — auto-regulation)
// ============================================================================

/// Severity level for a homeostatic ratio out of equilibrium.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HomeostasisSeverity {
    Ok,
    Warning,
    Critical,
}

/// A single homeostatic ratio with its equilibrium zone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeostasisRatio {
    /// Human-readable name of the ratio
    pub name: String,
    /// Current measured value
    pub value: f64,
    /// Target equilibrium range [min, max]
    pub target_range: (f64, f64),
    /// Absolute distance to nearest edge of the target range (0 = in zone)
    pub distance_to_equilibrium: f64,
    /// Severity based on distance
    pub severity: HomeostasisSeverity,
    /// Textual recommendation if out of zone
    pub recommendation: Option<String>,
}

/// Full homeostasis report for a project's knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeostasisReport {
    /// Individual ratios
    pub ratios: Vec<HomeostasisRatio>,
    /// Aggregated pain score (0.0 = perfect equilibrium, 1.0 = max pain)
    pub pain_score: f64,
    /// Overall recommendations
    pub recommendations: Vec<String>,
}

// ============================================================================
// Identity Manifold — Community structural identity & drift detection
// ============================================================================

/// Structural identity of a Louvain community (centroid of member fingerprints).
/// Biomimicry: maps to Elun's HypersphereIdentity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityIdentity {
    /// Louvain community identifier
    pub community_id: i64,
    /// Community label (human-readable)
    pub community_label: String,
    /// Centroid: mean of all member fingerprints (17-dims)
    pub centroid: Vec<f64>,
    /// Number of files in this community with fingerprints
    pub member_count: usize,
    /// Timestamp of last computation
    pub last_computed: chrono::DateTime<chrono::Utc>,
}

/// A file that has drifted from its community's structural identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralDrift {
    /// File path
    pub file_path: String,
    /// Community this file belongs to
    pub community_id: i64,
    /// Community label
    pub community_label: String,
    /// Euclidean distance from file fingerprint to community centroid
    pub drift_distance: f64,
    /// Severity based on threshold (ok, warning, critical)
    pub severity: HomeostasisSeverity,
    /// Suggestion if drift is critical (e.g., migrate to closer community)
    pub suggestion: Option<String>,
}

/// Report of structural drift across all communities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralDriftReport {
    /// Files sorted by drift distance (descending)
    pub drifting_files: Vec<StructuralDrift>,
    /// Community centroids computed
    pub centroids: Vec<CommunityIdentity>,
    /// Mean drift across all files
    pub mean_drift: f64,
    /// Number of files above warning threshold
    pub warning_count: usize,
    /// Number of files above critical threshold
    pub critical_count: usize,
}

// ============================================================================
// P2P Coupling (Biomimicry — inter-project influence field)
// ============================================================================

/// Coupling breakdown between two projects — 4 signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingSignals {
    /// Number of structural twins (files with similar fingerprints) across the two projects.
    pub structural_twins: usize,
    /// Number of skills imported from one project to the other.
    pub imported_skills: usize,
    /// Number of notes shared/propagated between projects.
    pub shared_notes: usize,
    /// Jaccard overlap of note tags between the two projects.
    pub tag_overlap: f64,
}

/// Coupling entry between two projects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectCoupling {
    /// First project ID.
    pub project_a_id: Uuid,
    /// First project name.
    pub project_a_name: String,
    /// Second project ID.
    pub project_b_id: Uuid,
    /// Second project name.
    pub project_b_name: String,
    /// Combined coupling strength ∈ [0, 1].
    pub coupling_strength: f64,
    /// Breakdown by signal.
    pub signals: CouplingSignals,
}

/// Full coupling matrix for a workspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingMatrix {
    /// Workspace ID.
    pub workspace_id: Uuid,
    /// All pairwise coupling entries.
    pub entries: Vec<ProjectCoupling>,
    /// Number of projects in the workspace.
    pub project_count: usize,
}

/// GDS metrics for a single node (file or function).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeGdsMetrics {
    pub node_path: String,
    pub node_type: String, // "file" or "function"
    pub pagerank: Option<f64>,
    pub betweenness: Option<f64>,
    pub clustering_coefficient: Option<f64>,
    pub community_id: Option<i64>,
    pub community_label: Option<String>,
    pub in_degree: i64,
    pub out_degree: i64,
    /// Fabric PageRank — computed on multi-layer graph (IMPORTS + CO_CHANGED + ...)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fabric_pagerank: Option<f64>,
    /// Fabric betweenness centrality — computed on multi-layer graph
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fabric_betweenness: Option<f64>,
    /// Fabric community ID — Louvain on multi-layer graph
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fabric_community_id: Option<i64>,
    /// Fabric community label — human-readable
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fabric_community_label: Option<String>,
}

/// Statistical percentiles for a project's GDS metrics distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectPercentiles {
    pub pagerank_p50: f64,
    pub pagerank_p80: f64,
    pub pagerank_p95: f64,
    pub betweenness_p50: f64,
    pub betweenness_p80: f64,
    pub betweenness_p95: f64,
    pub betweenness_mean: f64,
    pub betweenness_stddev: f64,
}

/// Interpretation of a node's structural importance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInterpretation {
    pub importance: String, // "critical", "high", "medium", "low"
    pub is_bridge: bool,
    pub risk_level: String, // "critical", "high", "medium", "low"
    pub summary: String,
}

/// A file with high betweenness centrality (bridge between communities).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeFile {
    pub path: String,
    pub betweenness: f64,
    pub community_label: Option<String>,
}

/// Neural network metrics for a project's SYNAPSE layer.
///
/// Summarizes the state of the note-level neural connections:
/// active synapses, average energy, weak synapse ratio, and dead notes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralMetrics {
    pub active_synapses: i64,
    pub avg_energy: f64,
    pub weak_synapses_ratio: f64,
    pub dead_notes_count: i64,
}

/// A note with its embedding vector, for UMAP 2D projection.
/// Lightweight struct returned by `get_note_embeddings_for_project`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteEmbeddingPoint {
    pub id: Uuid,
    pub embedding: Vec<f32>,
    pub note_type: String,
    pub importance: String,
    pub energy: f64,
    pub tags: Vec<String>,
    pub content_preview: String,
}

/// A feature graph — a named subgraph capturing all code entities related to a feature.
/// Reusable across sessions to avoid re-exploring the same feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGraphNode {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub project_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// Number of entities included in this feature graph (via INCLUDES_ENTITY).
    /// Populated by list_feature_graphs; None when not computed.
    pub entity_count: Option<i64>,
    /// Entry function used for auto_build (None for manually created graphs).
    pub entry_function: Option<String>,
    /// BFS depth used for auto_build.
    pub build_depth: Option<u32>,
    /// Relation types traversed during auto_build (None = all).
    pub include_relations: Option<Vec<String>>,
}

/// Role of an entity within a feature graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FeatureRole {
    /// The function used as entry point for the feature
    EntryPoint,
    /// Core functions reached via CALLS
    CoreLogic,
    /// Structs/enums used in the feature
    DataModel,
    /// Traits defining interfaces
    TraitContract,
    /// Handlers/routes exposed as API surface
    ApiSurface,
    /// Imported files and utility modules
    Support,
}

/// An entity included in a feature graph (file, function, struct, trait).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGraphEntity {
    pub entity_type: String,
    pub entity_id: String,
    /// Human-readable name (function name, struct name, file path)
    pub name: Option<String>,
    /// Role of this entity in the feature (entry_point, core_logic, data_model, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Normalized PageRank score (0.0–1.0) if GDS analytics are available.
    /// Allows visualizing the most important nodes in the feature graph.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub importance_score: Option<f64>,
}

/// A relationship between two entities inside a feature graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGraphRelation {
    pub source_type: String,
    pub source_id: String,
    pub target_type: String,
    pub target_id: String,
    pub relation_type: String,
}

/// Full feature graph with its included entities and intra-graph relations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGraphDetail {
    #[serde(flatten)]
    pub graph: FeatureGraphNode,
    pub entities: Vec<FeatureGraphEntity>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relations: Vec<FeatureGraphRelation>,
}

/// Statistics for a feature graph — coupling, cohesion and complexity metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGraphStatistics {
    /// Feature graph UUID
    pub id: Uuid,
    /// Feature graph name
    pub name: String,
    /// Total entity count
    pub entity_count: usize,
    /// Breakdown by entity type: {"file": 3, "function": 12, ...}
    pub entity_breakdown: std::collections::HashMap<String, usize>,
    /// Breakdown by role: {"core_logic": 8, "support": 4, ...}
    pub role_breakdown: std::collections::HashMap<String, usize>,
    /// Number of intra-graph relations (edges between entities in the graph)
    pub internal_edge_count: usize,
    /// Number of external dependencies (edges going out of the graph)
    pub external_edge_count: usize,
    /// Cohesion score: internal_edges / (entity_count * (entity_count - 1) / 2)
    /// Higher = more tightly coupled internally
    pub cohesion: f64,
    /// Coupling score: external_edges / (internal_edges + external_edges)
    /// Lower = more self-contained
    pub coupling: f64,
    /// Average importance score of entities (PageRank-based, if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_importance: Option<f64>,
    /// Number of entry points (entities with incoming edges from outside but no outgoing to outside)
    pub entry_points: Vec<String>,
    /// Number of exit points (entities with outgoing edges to outside but no incoming from outside)
    pub exit_points: Vec<String>,
}

/// Result of comparing two feature graphs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGraphComparison {
    /// First feature graph
    pub graph_a: FeatureGraphComparisonSide,
    /// Second feature graph
    pub graph_b: FeatureGraphComparisonSide,
    /// Entities shared by both graphs (entity_type, entity_id)
    pub shared_entities: Vec<FeatureGraphEntity>,
    /// Entities only in graph A
    pub unique_to_a: Vec<FeatureGraphEntity>,
    /// Entities only in graph B
    pub unique_to_b: Vec<FeatureGraphEntity>,
    /// Jaccard similarity: |shared| / |A ∪ B|
    pub similarity: f64,
}

/// Summary info for one side of a comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGraphComparisonSide {
    pub id: Uuid,
    pub name: String,
    pub entity_count: usize,
}

/// A feature graph that overlaps with a reference graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGraphOverlap {
    pub id: Uuid,
    pub name: String,
    pub entity_count: usize,
    /// Number of shared entities with the reference graph
    pub shared_count: usize,
    /// Shared entity identifiers (entity_type:entity_id)
    pub shared_entities: Vec<String>,
    /// Overlap ratio: shared_count / min(ref_count, this_count)
    pub overlap_ratio: f64,
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

    // Workspace relationships
    /// Project belongs to a workspace
    BelongsToWorkspace,
    /// Workspace has a workspace-level milestone
    HasWorkspaceMilestone,
    /// Workspace milestone includes a task (from any project)
    IncludesTask,
    /// Workspace milestone includes a project milestone
    IncludesProjectMilestone,
    /// Workspace or project has a resource
    HasResource,
    /// Project implements (provides) a resource/contract
    ImplementsResource,
    /// Project uses (consumes) a resource/contract
    UsesResource,
    /// Workspace has a deployment component
    HasComponent,
    /// Component depends on another component
    DependsOnComponent,
    /// Component maps to a project (source code)
    MapsToProject,

    // Feature graphs
    /// FeatureGraph includes an entity (File, Function, Struct, Trait)
    IncludesEntity,
}

// ============================================================================
// User / Auth
// ============================================================================

/// Authentication provider type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuthProvider {
    /// Password-based authentication (root account or registered users)
    Password,
    /// OIDC/OAuth provider (Google, Microsoft, Okta, etc.)
    Oidc,
}

impl std::fmt::Display for AuthProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthProvider::Password => write!(f, "password"),
            AuthProvider::Oidc => write!(f, "oidc"),
        }
    }
}

impl std::str::FromStr for AuthProvider {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "password" => Ok(AuthProvider::Password),
            "oidc" | "google" | "oauth" => Ok(AuthProvider::Oidc),
            _ => Err(format!("Unknown AuthProvider: {}", s)),
        }
    }
}

/// A user in the system (supports multiple auth providers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserNode {
    pub id: Uuid,
    pub email: String,
    pub name: String,
    pub picture_url: Option<String>,
    /// How this user authenticates
    pub auth_provider: AuthProvider,
    /// External identifier from the auth provider (e.g., Google "sub" claim).
    /// Present for OIDC users, absent for password users.
    #[serde(default)]
    pub external_id: Option<String>,
    /// Bcrypt hash of the password. Present for password users, absent for OIDC users.
    #[serde(default, skip_serializing)]
    pub password_hash: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_login_at: DateTime<Utc>,
}

/// A refresh token stored in the database (hashed).
///
/// The raw token is never stored — only its SHA-256 hash.
/// Used for the dual-token auth model (short-lived JWT + long-lived refresh cookie).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshTokenNode {
    /// SHA-256 hash of the raw token (primary lookup key)
    pub token_hash: String,
    /// User who owns this token
    pub user_id: Uuid,
    /// When this token expires
    pub expires_at: DateTime<Utc>,
    /// When this token was created
    pub created_at: DateTime<Utc>,
    /// Whether this token has been revoked (logout, rotation, etc.)
    pub revoked: bool,
}

// ============================================================================
// Analytics: Churn, Knowledge Density, Risk Score (T5.5, T5.6, T5.7)
// ============================================================================

/// Churn score for a file based on commit frequency and co-change patterns.
/// Computed from TOUCHES relations between Commit and File nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChurnScore {
    pub path: String,
    pub commit_count: i64,
    pub total_churn: i64, // additions + deletions
    pub co_change_count: i64,
    pub churn_score: f64, // normalized 0.0-1.0
}

/// Knowledge density for a file based on associated notes and decisions.
/// Files with low density are "knowledge gaps" that may need documentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileKnowledgeDensity {
    pub path: String,
    pub note_count: i64,
    pub decision_count: i64,
    pub knowledge_density: f64, // normalized 0.0-1.0
}

/// Composite risk score for a file combining structural importance, churn, and knowledge gaps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRiskScore {
    pub path: String,
    pub risk_score: f64,
    pub risk_level: String, // "low", "medium", "high", "critical"
    pub factors: RiskFactors,
}

/// Individual risk factor contributions for a file's composite risk score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactors {
    pub pagerank: f64,
    pub churn: f64,
    pub knowledge_gap: f64, // 1 - density
    pub betweenness: f64,
}

// ============================================================================
// Persona Node (Living Personas — Adaptive Knowledge Agents)
// ============================================================================

/// Origin of a persona: how it was created.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersonaOrigin {
    /// Created automatically from graph analysis (communities, call graph, etc.)
    AutoBuild,
    /// Created manually by the user or agent
    #[default]
    Manual,
    /// Imported from another project via PersonaPackage
    Imported,
}

impl std::fmt::Display for PersonaOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AutoBuild => write!(f, "auto_build"),
            Self::Manual => write!(f, "manual"),
            Self::Imported => write!(f, "imported"),
        }
    }
}

impl std::str::FromStr for PersonaOrigin {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "auto_build" => Ok(Self::AutoBuild),
            "manual" => Ok(Self::Manual),
            "imported" => Ok(Self::Imported),
            _ => Err(format!("Unknown PersonaOrigin: {s}")),
        }
    }
}

/// Status of a persona lifecycle.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersonaStatus {
    /// Recently used, ready for activation
    Active,
    /// Not used recently, still available
    Dormant,
    /// Newly created, accumulating knowledge
    #[default]
    Emerging,
    /// Archived, no longer available for activation
    Archived,
}

impl std::fmt::Display for PersonaStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Dormant => write!(f, "dormant"),
            Self::Emerging => write!(f, "emerging"),
            Self::Archived => write!(f, "archived"),
        }
    }
}

impl std::str::FromStr for PersonaStatus {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "active" => Ok(Self::Active),
            "dormant" => Ok(Self::Dormant),
            "emerging" => Ok(Self::Emerging),
            "archived" => Ok(Self::Archived),
            _ => Err(format!("Unknown PersonaStatus: {s}")),
        }
    }
}

/// A Living Persona — an adaptive knowledge agent connected to the graph.
///
/// Personas aggregate domain-specific knowledge (skills, protocols, files,
/// notes, decisions) into a coherent "expert profile" that evolves over time.
/// They form a runtime stack (PersonaStack) that switches dynamically based
/// on what files/functions the agent is touching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaNode {
    pub id: Uuid,
    /// Project this persona belongs to. None = global/portable persona.
    pub project_id: Option<Uuid>,
    /// Human-readable name (e.g., "neo4j-expert", "threejs-specialist")
    pub name: String,
    /// Description of the persona's domain of expertise
    #[serde(default)]
    pub description: String,
    /// Current lifecycle status
    #[serde(default)]
    pub status: PersonaStatus,

    // --- Execution parameters ---
    /// Default complexity classification: "simple", "complex", "creative"
    #[serde(default)]
    pub complexity_default: Option<String>,
    /// Guard timeout in seconds (overrides RunnerConfig.task_timeout_secs)
    pub timeout_secs: Option<u64>,
    /// Maximum cost in USD for tasks using this persona
    pub max_cost_usd: Option<f64>,
    /// Preferred model (e.g., "opus", "sonnet"). None = use default.
    pub model_preference: Option<String>,
    /// System prompt override/addition. None = use default.
    pub system_prompt_override: Option<String>,

    // --- Living metrics (evolve over time) ---
    /// Energy level [0,1] — reinforced on success, decayed on failure
    #[serde(default = "default_energy")]
    pub energy: f64,
    /// Knowledge cohesion [0,1] — how tightly connected its knowledge subgraph is
    #[serde(default)]
    pub cohesion: f64,
    /// Number of times this persona has been activated
    #[serde(default)]
    pub activation_count: i64,
    /// Success rate of tasks executed with this persona [0,1]
    #[serde(default)]
    pub success_rate: f64,
    /// Average task duration in seconds when this persona is active
    #[serde(default)]
    pub avg_duration_secs: f64,
    /// Last time this persona was activated
    pub last_activated: Option<DateTime<Utc>>,

    // --- Rate-limiting & history ---
    /// Accumulated energy boost this maintenance cycle (reset each cycle)
    #[serde(default)]
    pub energy_boost_accumulated: f64,
    /// Last 5 energy values for stagnation detection
    #[serde(default)]
    pub energy_history: Vec<f64>,

    // --- Bootstrap tracking ---
    /// How this persona was created
    #[serde(default)]
    pub origin: PersonaOrigin,

    // --- Timestamps ---
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}

fn default_energy() -> f64 {
    0.5
}

/// A weighted relation from a Persona to a knowledge entity.
///
/// Used for KNOWS (File/Function), USES (Note/Decision) relations
/// that carry a relevance weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaWeightedRelation {
    /// Target entity type (e.g., "file", "function", "note", "decision")
    pub entity_type: String,
    /// Target entity identifier (path for files/functions, UUID string for notes/decisions)
    pub entity_id: String,
    /// Relevance weight [0,1] — evolves with usage
    pub weight: f64,
}

/// Summary of a persona's knowledge subgraph — the full "expert context".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaSubgraph {
    /// Persona identity
    pub persona_id: Uuid,
    pub persona_name: String,
    /// Files this persona KNOWS (with weights)
    pub files: Vec<PersonaWeightedRelation>,
    /// Functions this persona KNOWS (with weights)
    pub functions: Vec<PersonaWeightedRelation>,
    /// Notes this persona USES (with weights)
    pub notes: Vec<PersonaWeightedRelation>,
    /// Decisions this persona USES (with weights)
    pub decisions: Vec<PersonaWeightedRelation>,
    /// Skills this persona MASTERS (with entity_type = "skill")
    pub skills: Vec<PersonaWeightedRelation>,
    /// Protocols this persona FOLLOWS (with entity_type = "protocol")
    pub protocols: Vec<PersonaWeightedRelation>,
    /// FeatureGraph this persona is SCOPED_TO (if any)
    pub feature_graph_id: Option<Uuid>,
    /// Parent personas via EXTENDS chain (with entity_type = "persona")
    pub parents: Vec<PersonaWeightedRelation>,
    /// Children personas that EXTEND this one (with entity_type = "persona")
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<PersonaWeightedRelation>,
    /// Statistics
    pub stats: PersonaSubgraphStats,
}

/// Statistics about a persona's knowledge subgraph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaSubgraphStats {
    /// Total number of entities in the subgraph
    pub total_entities: usize,
    /// Coverage score: how much of the project this persona covers [0,1]
    pub coverage_score: f64,
    /// Freshness: average recency of relations (based on last_activated and note timestamps)
    pub freshness: f64,
}

/// Portable persona package for export/import across projects.
///
/// Contains the persona definition and all its portable knowledge
/// (notes, decisions). Files/functions are project-specific and NOT included —
/// they are reconstructed on import via `auto_bind`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaPackage {
    /// Schema version for forward compatibility
    pub schema_version: u32,
    /// Persona identity (without project-specific IDs)
    pub persona: PortablePersona,
    /// Notes this persona USES (content, not IDs)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<PortablePersonaNote>,
    /// Decisions this persona USES (content, not IDs)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decisions: Vec<PortablePersonaDecision>,
    /// Skill names this persona MASTERS (for re-linking on import)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skill_names: Vec<String>,
    /// Source metadata for provenance tracking
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<PersonaPackageSource>,
}

/// Portable persona definition (no project-specific IDs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortablePersona {
    pub name: String,
    pub description: String,
    pub complexity_default: Option<String>,
    pub timeout_secs: Option<u64>,
    pub max_cost_usd: Option<f64>,
    pub model_preference: Option<String>,
    pub system_prompt_override: Option<String>,
    pub energy: f64,
    pub cohesion: f64,
    pub activation_count: i64,
    pub success_rate: f64,
}

/// A note carried inside a PersonaPackage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortablePersonaNote {
    pub note_type: String,
    pub content: String,
    pub importance: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Weight of the USES relation
    pub weight: f64,
}

/// A decision carried inside a PersonaPackage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortablePersonaDecision {
    pub description: String,
    pub rationale: String,
    pub chosen_option: String,
    /// Weight of the USES relation
    pub weight: f64,
}

/// Source metadata for provenance tracking in PersonaPackage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaPackageSource {
    pub project_name: Option<String>,
    pub exported_at: DateTime<Utc>,
}

/// Result of importing a PersonaPackage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaImportResult {
    pub persona_id: Uuid,
    pub persona_name: String,
    pub notes_imported: usize,
    pub decisions_imported: usize,
    pub skills_linked: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_node_serialization() {
        let workspace = WorkspaceNode {
            id: Uuid::new_v4(),
            name: "Test Workspace".to_string(),
            slug: "test-workspace".to_string(),
            description: Some("A test workspace".to_string()),
            created_at: Utc::now(),
            updated_at: None,
            metadata: serde_json::json!({"key": "value"}),
        };

        let json = serde_json::to_string(&workspace).unwrap();
        let deserialized: WorkspaceNode = serde_json::from_str(&json).unwrap();

        assert_eq!(workspace.id, deserialized.id);
        assert_eq!(workspace.name, deserialized.name);
        assert_eq!(workspace.slug, deserialized.slug);
        assert_eq!(workspace.description, deserialized.description);
    }

    #[test]
    fn test_workspace_milestone_node_serialization() {
        let milestone = WorkspaceMilestoneNode {
            id: Uuid::new_v4(),
            workspace_id: Uuid::new_v4(),
            title: "Q1 Launch".to_string(),
            description: Some("Cross-project milestone".to_string()),
            status: MilestoneStatus::Open,
            target_date: Some(Utc::now()),
            closed_at: None,
            created_at: Utc::now(),
            tags: vec!["launch".to_string(), "q1".to_string()],
        };

        let json = serde_json::to_string(&milestone).unwrap();
        let deserialized: WorkspaceMilestoneNode = serde_json::from_str(&json).unwrap();

        assert_eq!(milestone.id, deserialized.id);
        assert_eq!(milestone.title, deserialized.title);
        assert_eq!(milestone.tags, deserialized.tags);
    }

    #[test]
    fn test_resource_node_serialization() {
        let resource = ResourceNode {
            id: Uuid::new_v4(),
            workspace_id: Some(Uuid::new_v4()),
            project_id: None,
            name: "User API".to_string(),
            resource_type: ResourceType::ApiContract,
            file_path: "specs/openapi/users.yaml".to_string(),
            url: Some("https://api.example.com/spec".to_string()),
            format: Some("openapi".to_string()),
            version: Some("1.0.0".to_string()),
            description: Some("User service contract".to_string()),
            created_at: Utc::now(),
            updated_at: None,
            metadata: serde_json::json!({}),
        };

        let json = serde_json::to_string(&resource).unwrap();
        let deserialized: ResourceNode = serde_json::from_str(&json).unwrap();

        assert_eq!(resource.id, deserialized.id);
        assert_eq!(resource.name, deserialized.name);
        assert_eq!(resource.resource_type, deserialized.resource_type);
        assert_eq!(resource.file_path, deserialized.file_path);
    }

    #[test]
    fn test_component_node_serialization() {
        let component = ComponentNode {
            id: Uuid::new_v4(),
            workspace_id: Uuid::new_v4(),
            name: "API Gateway".to_string(),
            component_type: ComponentType::Gateway,
            description: Some("Main entry point".to_string()),
            runtime: Some("kubernetes".to_string()),
            config: serde_json::json!({"port": 8080, "replicas": 3}),
            created_at: Utc::now(),
            tags: vec!["infrastructure".to_string()],
        };

        let json = serde_json::to_string(&component).unwrap();
        let deserialized: ComponentNode = serde_json::from_str(&json).unwrap();

        assert_eq!(component.id, deserialized.id);
        assert_eq!(component.name, deserialized.name);
        assert_eq!(component.component_type, deserialized.component_type);
        assert_eq!(component.runtime, deserialized.runtime);
    }

    #[test]
    fn test_resource_type_from_str() {
        assert_eq!(
            "api_contract".parse::<ResourceType>().unwrap(),
            ResourceType::ApiContract
        );
        assert_eq!(
            "apicontract".parse::<ResourceType>().unwrap(),
            ResourceType::ApiContract
        );
        assert_eq!(
            "protobuf".parse::<ResourceType>().unwrap(),
            ResourceType::Protobuf
        );
        assert_eq!(
            "graphql_schema".parse::<ResourceType>().unwrap(),
            ResourceType::GraphqlSchema
        );
        assert_eq!(
            "graphql".parse::<ResourceType>().unwrap(),
            ResourceType::GraphqlSchema
        );
        assert_eq!(
            "json_schema".parse::<ResourceType>().unwrap(),
            ResourceType::JsonSchema
        );
        assert_eq!(
            "database_schema".parse::<ResourceType>().unwrap(),
            ResourceType::DatabaseSchema
        );
        assert_eq!(
            "database".parse::<ResourceType>().unwrap(),
            ResourceType::DatabaseSchema
        );
        assert_eq!(
            "shared_types".parse::<ResourceType>().unwrap(),
            ResourceType::SharedTypes
        );
        assert_eq!(
            "config".parse::<ResourceType>().unwrap(),
            ResourceType::Config
        );
        assert_eq!(
            "documentation".parse::<ResourceType>().unwrap(),
            ResourceType::Documentation
        );
        assert_eq!(
            "docs".parse::<ResourceType>().unwrap(),
            ResourceType::Documentation
        );
        assert_eq!(
            "other".parse::<ResourceType>().unwrap(),
            ResourceType::Other
        );

        // Test invalid
        assert!("invalid".parse::<ResourceType>().is_err());
    }

    #[test]
    fn test_component_type_from_str() {
        assert_eq!(
            "service".parse::<ComponentType>().unwrap(),
            ComponentType::Service
        );
        assert_eq!(
            "frontend".parse::<ComponentType>().unwrap(),
            ComponentType::Frontend
        );
        assert_eq!(
            "worker".parse::<ComponentType>().unwrap(),
            ComponentType::Worker
        );
        assert_eq!(
            "database".parse::<ComponentType>().unwrap(),
            ComponentType::Database
        );
        assert_eq!(
            "db".parse::<ComponentType>().unwrap(),
            ComponentType::Database
        );
        assert_eq!(
            "message_queue".parse::<ComponentType>().unwrap(),
            ComponentType::MessageQueue
        );
        assert_eq!(
            "queue".parse::<ComponentType>().unwrap(),
            ComponentType::MessageQueue
        );
        assert_eq!(
            "cache".parse::<ComponentType>().unwrap(),
            ComponentType::Cache
        );
        assert_eq!(
            "gateway".parse::<ComponentType>().unwrap(),
            ComponentType::Gateway
        );
        assert_eq!(
            "api_gateway".parse::<ComponentType>().unwrap(),
            ComponentType::Gateway
        );
        assert_eq!(
            "external".parse::<ComponentType>().unwrap(),
            ComponentType::External
        );
        assert_eq!(
            "other".parse::<ComponentType>().unwrap(),
            ComponentType::Other
        );

        // Test invalid
        assert!("invalid".parse::<ComponentType>().is_err());
    }

    #[test]
    fn test_component_dependency_serialization() {
        let dep = ComponentDependency {
            from_id: Uuid::new_v4(),
            to_id: Uuid::new_v4(),
            protocol: Some("http".to_string()),
            required: true,
        };

        let json = serde_json::to_string(&dep).unwrap();
        let deserialized: ComponentDependency = serde_json::from_str(&json).unwrap();

        assert_eq!(dep.from_id, deserialized.from_id);
        assert_eq!(dep.to_id, deserialized.to_id);
        assert_eq!(dep.protocol, deserialized.protocol);
        assert_eq!(dep.required, deserialized.required);
    }

    #[test]
    fn test_component_dependency_default_required() {
        // Test that required defaults to true when not specified
        let json = r#"{"from_id":"00000000-0000-0000-0000-000000000001","to_id":"00000000-0000-0000-0000-000000000002","protocol":"grpc"}"#;
        let dep: ComponentDependency = serde_json::from_str(json).unwrap();
        assert!(dep.required);
    }

    #[test]
    fn test_chat_session_node_with_workspace_and_add_dirs() {
        let session = ChatSessionNode {
            id: Uuid::new_v4(),
            cli_session_id: None,
            project_slug: Some("proj".to_string()),
            workspace_slug: Some("my-ws".to_string()),
            cwd: "/tmp".to_string(),
            title: None,
            model: "model".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            message_count: 0,
            total_cost_usd: Some(0.0),
            conversation_id: None,
            preview: None,
            permission_mode: None,
            add_dirs: Some(vec!["/dir/a".to_string(), "/dir/b".to_string()]),
            spawned_by: None,
        };

        let json = serde_json::to_string(&session).unwrap();
        let de: ChatSessionNode = serde_json::from_str(&json).unwrap();
        assert_eq!(de.workspace_slug.as_deref(), Some("my-ws"));
        assert_eq!(de.add_dirs.as_ref().unwrap().len(), 2);
        assert_eq!(de.add_dirs.as_ref().unwrap()[0], "/dir/a");
    }

    #[test]
    fn test_chat_session_node_workspace_fields_default() {
        let session = ChatSessionNode {
            id: Uuid::new_v4(),
            cli_session_id: None,
            project_slug: None,
            workspace_slug: None,
            cwd: "/tmp".to_string(),
            title: None,
            model: "model".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            message_count: 0,
            total_cost_usd: Some(0.0),
            conversation_id: None,
            preview: None,
            permission_mode: None,
            add_dirs: None,
            spawned_by: None,
        };

        let json = serde_json::to_string(&session).unwrap();
        let de: ChatSessionNode = serde_json::from_str(&json).unwrap();
        assert!(de.workspace_slug.is_none());
        assert!(de.add_dirs.is_none());
    }

    #[test]
    fn test_chat_session_node_deserialize_missing_workspace_fields() {
        // Backward compatibility: JSON without workspace_slug/add_dirs should deserialize fine
        let json = r#"{
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "cwd": "/tmp",
            "model": "claude-opus-4-6",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "message_count": 0,
            "total_cost_usd": 0.0
        }"#;
        let session: ChatSessionNode = serde_json::from_str(json).unwrap();
        assert!(session.workspace_slug.is_none());
        assert!(session.add_dirs.is_none());
    }

    // ========================================================================
    // Persona model tests
    // ========================================================================

    #[test]
    fn test_persona_origin_display_and_from_str() {
        for (variant, expected_str) in [
            (PersonaOrigin::AutoBuild, "auto_build"),
            (PersonaOrigin::Manual, "manual"),
            (PersonaOrigin::Imported, "imported"),
        ] {
            assert_eq!(variant.to_string(), expected_str);
            let parsed: PersonaOrigin = expected_str.parse().unwrap();
            assert_eq!(parsed, variant);
        }
        // Unknown value should error
        assert!("unknown".parse::<PersonaOrigin>().is_err());
    }

    #[test]
    fn test_persona_origin_default() {
        assert_eq!(PersonaOrigin::default(), PersonaOrigin::Manual);
    }

    #[test]
    fn test_persona_status_display_and_from_str() {
        for (variant, expected_str) in [
            (PersonaStatus::Active, "active"),
            (PersonaStatus::Dormant, "dormant"),
            (PersonaStatus::Emerging, "emerging"),
            (PersonaStatus::Archived, "archived"),
        ] {
            assert_eq!(variant.to_string(), expected_str);
            let parsed: PersonaStatus = expected_str.parse().unwrap();
            assert_eq!(parsed, variant);
        }
        assert!("invalid".parse::<PersonaStatus>().is_err());
    }

    #[test]
    fn test_persona_status_default() {
        assert_eq!(PersonaStatus::default(), PersonaStatus::Emerging);
    }

    #[test]
    fn test_persona_node_serialization_roundtrip() {
        let persona = PersonaNode {
            id: Uuid::new_v4(),
            project_id: Some(Uuid::new_v4()),
            name: "neo4j-expert".to_string(),
            description: "Expert in Neo4j graph queries".to_string(),
            status: PersonaStatus::Active,
            complexity_default: Some("complex".to_string()),
            timeout_secs: Some(300),
            max_cost_usd: Some(0.50),
            model_preference: Some("opus".to_string()),
            system_prompt_override: None,
            energy: 0.8,
            cohesion: 0.65,
            activation_count: 42,
            success_rate: 0.95,
            avg_duration_secs: 120.5,
            last_activated: Some(Utc::now()),
            energy_boost_accumulated: 0.0,
            energy_history: vec![],
            origin: PersonaOrigin::Manual,
            created_at: Utc::now(),
            updated_at: Some(Utc::now()),
        };

        let json = serde_json::to_string(&persona).unwrap();
        let de: PersonaNode = serde_json::from_str(&json).unwrap();

        assert_eq!(persona.id, de.id);
        assert_eq!(persona.name, de.name);
        assert_eq!(persona.status, de.status);
        assert_eq!(persona.energy, de.energy);
        assert_eq!(persona.activation_count, de.activation_count);
        assert_eq!(persona.origin, de.origin);
        assert_eq!(persona.complexity_default, de.complexity_default);
        assert_eq!(persona.model_preference, de.model_preference);
    }

    #[test]
    fn test_persona_node_defaults_on_minimal_json() {
        // Ensure serde defaults work for optional/defaulted fields
        let json = r#"{
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "minimal-persona",
            "description": "",
            "created_at": "2026-01-01T00:00:00Z"
        }"#;
        let persona: PersonaNode = serde_json::from_str(json).unwrap();
        assert_eq!(persona.status, PersonaStatus::Emerging);
        assert_eq!(persona.origin, PersonaOrigin::Manual);
        assert!((persona.energy - 0.5).abs() < f64::EPSILON); // default_energy()
        assert_eq!(persona.cohesion, 0.0);
        assert_eq!(persona.activation_count, 0);
        assert!(persona.project_id.is_none());
        assert!(persona.complexity_default.is_none());
    }

    #[test]
    fn test_persona_node_global_persona_no_project_id() {
        let persona = PersonaNode {
            id: Uuid::new_v4(),
            project_id: None,
            name: "global-expert".to_string(),
            description: "A global persona".to_string(),
            status: PersonaStatus::Active,
            complexity_default: None,
            timeout_secs: None,
            max_cost_usd: None,
            model_preference: None,
            system_prompt_override: None,
            energy: 0.5,
            cohesion: 0.0,
            activation_count: 0,
            success_rate: 0.0,
            avg_duration_secs: 0.0,
            last_activated: None,
            energy_boost_accumulated: 0.0,
            energy_history: vec![],
            origin: PersonaOrigin::Manual,
            created_at: Utc::now(),
            updated_at: None,
        };
        let json = serde_json::to_string(&persona).unwrap();
        let de: PersonaNode = serde_json::from_str(&json).unwrap();
        assert!(de.project_id.is_none());
    }

    #[test]
    fn test_persona_weighted_relation_serialization() {
        let rel = PersonaWeightedRelation {
            entity_type: "file".to_string(),
            entity_id: "/src/main.rs".to_string(),
            weight: 0.85,
        };
        let json = serde_json::to_string(&rel).unwrap();
        let de: PersonaWeightedRelation = serde_json::from_str(&json).unwrap();
        assert_eq!(de.entity_type, "file");
        assert_eq!(de.entity_id, "/src/main.rs");
        assert!((de.weight - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_persona_subgraph_stats_serialization() {
        let stats = PersonaSubgraphStats {
            total_entities: 42,
            coverage_score: 0.73,
            freshness: 0.91,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let de: PersonaSubgraphStats = serde_json::from_str(&json).unwrap();
        assert_eq!(de.total_entities, 42);
        assert!((de.coverage_score - 0.73).abs() < f64::EPSILON);
    }

    #[test]
    fn test_persona_subgraph_serialization() {
        let subgraph = PersonaSubgraph {
            persona_id: Uuid::new_v4(),
            persona_name: "test-persona".to_string(),
            files: vec![PersonaWeightedRelation {
                entity_type: "file".to_string(),
                entity_id: "/src/lib.rs".to_string(),
                weight: 1.0,
            }],
            functions: vec![],
            notes: vec![],
            decisions: vec![],
            skills: vec![],
            protocols: vec![],
            feature_graph_id: None,
            parents: vec![],
            children: vec![],
            stats: PersonaSubgraphStats {
                total_entities: 1,
                coverage_score: 0.1,
                freshness: 1.0,
            },
        };
        let json = serde_json::to_string(&subgraph).unwrap();
        let de: PersonaSubgraph = serde_json::from_str(&json).unwrap();
        assert_eq!(de.persona_name, "test-persona");
        assert_eq!(de.files.len(), 1);
        assert!(de.children.is_empty());
    }

    #[test]
    fn test_persona_package_serialization_roundtrip() {
        let package = PersonaPackage {
            schema_version: 1,
            persona: PortablePersona {
                name: "neo4j-expert".to_string(),
                description: "Expert in graph queries".to_string(),
                complexity_default: Some("complex".to_string()),
                timeout_secs: None,
                max_cost_usd: Some(1.0),
                model_preference: None,
                system_prompt_override: None,
                energy: 0.8,
                cohesion: 0.7,
                activation_count: 10,
                success_rate: 0.9,
            },
            notes: vec![PortablePersonaNote {
                note_type: "gotcha".to_string(),
                content: "Use run() not execute() for writes".to_string(),
                importance: "critical".to_string(),
                tags: vec!["neo4j".to_string()],
                weight: 1.0,
            }],
            decisions: vec![PortablePersonaDecision {
                description: "Use neo4rs".to_string(),
                rationale: "Best async driver".to_string(),
                chosen_option: "neo4rs".to_string(),
                weight: 0.9,
            }],
            skill_names: vec!["graph-queries".to_string()],
            source: Some(PersonaPackageSource {
                project_name: Some("project-orchestrator".to_string()),
                exported_at: Utc::now(),
            }),
        };

        let json = serde_json::to_string(&package).unwrap();
        let de: PersonaPackage = serde_json::from_str(&json).unwrap();
        assert_eq!(de.schema_version, 1);
        assert_eq!(de.persona.name, "neo4j-expert");
        assert_eq!(de.notes.len(), 1);
        assert_eq!(de.decisions.len(), 1);
        assert_eq!(de.skill_names, vec!["graph-queries"]);
        assert!(de.source.is_some());
    }

    #[test]
    fn test_persona_package_minimal() {
        // Empty notes/decisions/skills should be omitted from JSON
        let package = PersonaPackage {
            schema_version: 1,
            persona: PortablePersona {
                name: "minimal".to_string(),
                description: "".to_string(),
                complexity_default: None,
                timeout_secs: None,
                max_cost_usd: None,
                model_preference: None,
                system_prompt_override: None,
                energy: 0.5,
                cohesion: 0.0,
                activation_count: 0,
                success_rate: 0.0,
            },
            notes: vec![],
            decisions: vec![],
            skill_names: vec![],
            source: None,
        };
        let json = serde_json::to_string(&package).unwrap();
        // skip_serializing_if = "Vec::is_empty" should omit empty vecs
        assert!(!json.contains("\"notes\""));
        assert!(!json.contains("\"decisions\""));
        assert!(!json.contains("\"skill_names\""));
        assert!(!json.contains("\"source\""));
    }

    #[test]
    fn test_persona_import_result_serialization() {
        let result = PersonaImportResult {
            persona_id: Uuid::new_v4(),
            persona_name: "imported-expert".to_string(),
            notes_imported: 3,
            decisions_imported: 2,
            skills_linked: 1,
        };
        let json = serde_json::to_string(&result).unwrap();
        let de: PersonaImportResult = serde_json::from_str(&json).unwrap();
        assert_eq!(de.persona_name, "imported-expert");
        assert_eq!(de.notes_imported, 3);
        assert_eq!(de.skills_linked, 1);
    }

    #[test]
    fn test_persona_origin_serde_snake_case() {
        // Verify serde rename_all = "snake_case" works correctly
        let json = r#""auto_build""#;
        let origin: PersonaOrigin = serde_json::from_str(json).unwrap();
        assert_eq!(origin, PersonaOrigin::AutoBuild);

        let serialized = serde_json::to_string(&PersonaOrigin::AutoBuild).unwrap();
        assert_eq!(serialized, r#""auto_build""#);
    }

    #[test]
    fn test_persona_status_serde_snake_case() {
        for (json_str, expected) in [
            (r#""active""#, PersonaStatus::Active),
            (r#""dormant""#, PersonaStatus::Dormant),
            (r#""emerging""#, PersonaStatus::Emerging),
            (r#""archived""#, PersonaStatus::Archived),
        ] {
            let de: PersonaStatus = serde_json::from_str(json_str).unwrap();
            assert_eq!(de, expected);
        }
    }
}
