//! Core data models for trajectory-based neural routing.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A complete reasoning trajectory — a sequence of decisions made during a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub id: Uuid,
    pub session_id: String,
    /// Embedding of the original query (256d, L2-normalized).
    pub query_embedding: Vec<f32>,
    /// Total RBCR reward for the entire trajectory.
    pub total_reward: f64,
    /// Number of decision steps.
    pub step_count: usize,
    /// Total duration in milliseconds.
    pub duration_ms: u64,
    /// Ordered list of decision nodes.
    pub nodes: Vec<TrajectoryNode>,
    pub created_at: DateTime<Utc>,
    /// Protocol run ID if this trajectory was produced during an active FSM run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_run_id: Option<Uuid>,
}

/// A single decision point in a reasoning trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryNode {
    pub id: Uuid,
    /// Embedding of the context at decision time (256d, L2-normalized).
    pub context_embedding: Vec<f32>,
    /// The action type chosen (e.g., "code_search", "note_get_context", "analyze_impact").
    pub action_type: String,
    /// Parameters passed to the action.
    pub action_params: serde_json::Value,
    /// Number of alternative actions considered.
    pub alternatives_count: usize,
    /// Index of the chosen action among alternatives.
    pub chosen_index: usize,
    /// Model confidence in this decision (0.0 - 1.0).
    pub confidence: f64,
    /// Local reward for this step (assigned by RewardDecomposer).
    pub local_reward: f64,
    /// Cumulative reward up to this step.
    pub cumulative_reward: f64,
    /// Time delta from previous node in milliseconds.
    pub delta_ms: u64,
    /// Order within the trajectory (0-based).
    pub order: usize,
}

/// A candidate action that was considered at a decision point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCandidate {
    pub action_type: String,
    pub action_params: serde_json::Value,
    /// Estimated score/probability for this candidate.
    pub score: f64,
}

/// The planned route returned by a router — a sequence of actions to execute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedAction {
    pub action_type: String,
    pub action_params: serde_json::Value,
    /// Confidence that this action is appropriate.
    pub confidence: f64,
}

/// Result from a nearest-neighbor route lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNRoute {
    /// The planned actions extracted from the best matching trajectory.
    pub actions: Vec<PlannedAction>,
    /// Cosine similarity to the source trajectory.
    pub similarity: f64,
    /// Combined score (similarity * recency * reward).
    pub score: f64,
    /// ID of the source trajectory this route was derived from.
    pub source_trajectory_id: Uuid,
    /// Total reward of the source trajectory.
    pub source_reward: f64,
}

/// Filter for querying trajectories.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrajectoryFilter {
    pub session_id: Option<String>,
    pub min_reward: Option<f64>,
    pub max_reward: Option<f64>,
    pub from_date: Option<DateTime<Utc>>,
    pub to_date: Option<DateTime<Utc>>,
    pub min_steps: Option<usize>,
    pub max_steps: Option<usize>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Statistics about stored trajectories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryStats {
    pub total_count: usize,
    pub avg_reward: f64,
    pub avg_step_count: f64,
    pub avg_duration_ms: f64,
    pub reward_distribution: RewardDistribution,
}

/// Distribution of rewards across trajectories.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RewardDistribution {
    pub min: f64,
    pub max: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
}

// ---------------------------------------------------------------------------
// Relations: USED_TOOL / TOUCHED_ENTITY
// ---------------------------------------------------------------------------

/// Record of a tool used at a decision point (USED_TOOL relation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUsage {
    /// MCP tool name (e.g., "code", "note", "plan").
    pub tool_name: String,
    /// MCP action within the tool (e.g., "search", "get_context").
    pub action: String,
    /// Serialized key parameters (stripped of PII).
    pub params_hash: String,
    /// Execution duration in milliseconds.
    pub duration_ms: Option<u64>,
    /// Whether the tool call succeeded.
    pub success: bool,
}

/// Record of an entity touched at a decision point (TOUCHED_ENTITY relation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TouchedEntity {
    /// Entity type: "file", "function", "note", "skill", "struct", "trait".
    pub entity_type: String,
    /// Entity identifier (file path, function name, UUID, etc.).
    pub entity_id: String,
    /// How the entity was touched: "read", "write", "search_hit", "context_load".
    pub access_mode: String,
    /// Relevance score (0.0-1.0) — how important was this entity to the decision.
    pub relevance: Option<f64>,
}
