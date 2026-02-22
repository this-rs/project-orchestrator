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
}

/// Coupling metrics from clustering coefficients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingMetrics {
    pub avg_clustering_coefficient: f64,
    pub max_clustering_coefficient: f64,
    pub most_coupled_file: Option<String>,
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

/// Full feature graph with its included entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGraphDetail {
    #[serde(flatten)]
    pub graph: FeatureGraphNode,
    pub entities: Vec<FeatureGraphEntity>,
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
}
