//! API handlers for ReasoningTree operations.
//!
//! Provides two endpoints:
//! - `POST /api/reason` — Build a reasoning tree from a natural language query
//! - `POST /api/reason/{tree_id}/feedback` — Provide feedback on a reasoning tree

use crate::api::handlers::{AppError, OrchestratorState};
use crate::reasoning::models::ReasoningTreeConfig;
use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Request / Response types
// ============================================================================

/// Request body for `POST /api/reason`.
#[derive(Debug, Deserialize)]
pub struct ReasonRequest {
    /// Natural language query to reason about.
    pub request: String,
    /// Optional project scope (limits search to project's notes/decisions/skills).
    pub project_id: Option<Uuid>,
    /// Maximum tree depth (default: 4).
    pub depth: Option<u8>,
    /// Whether to generate MCP actions for leaf nodes (default: true).
    pub include_actions: Option<bool>,
    /// Maximum number of activated nodes (default: 50).
    pub max_nodes: Option<usize>,
}

/// Response for `POST /api/reason`.
#[derive(Debug, Serialize)]
pub struct ReasonResponse {
    /// The unique ID of the reasoning tree (for feedback).
    pub id: Uuid,
    /// The original request that generated this tree.
    pub request: String,
    /// Root nodes of the reasoning tree.
    pub roots: Vec<crate::reasoning::ReasoningNode>,
    /// Maximum depth reached.
    pub depth: usize,
    /// Overall confidence score (0.0 - 1.0).
    pub confidence: f64,
    /// Total number of nodes in the tree.
    pub node_count: usize,
    /// Time taken to build the tree in milliseconds.
    pub build_time_ms: u64,
    /// Project scope (if any).
    pub project_id: Option<Uuid>,
    /// Suggested MCP actions sorted by confidence.
    pub suggested_actions: Vec<crate::reasoning::Action>,
}

/// Request body for `POST /api/reason/{tree_id}/feedback`.
#[derive(Debug, Deserialize)]
pub struct ReasonFeedbackRequest {
    /// Node IDs that were actually followed/useful.
    pub followed_nodes: Vec<Uuid>,
    /// Outcome of following the reasoning path.
    pub outcome: FeedbackOutcome,
    /// Optional task ID to increment frustration on failure (biomimicry).
    #[serde(default)]
    pub task_id: Option<Uuid>,
}

/// Feedback outcome.
#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackOutcome {
    /// The reasoning path led to a successful action.
    Success,
    /// The reasoning path was partially useful.
    Partial,
    /// The reasoning path was not useful.
    Failure,
}

/// Response for `POST /api/reason/{tree_id}/feedback`.
#[derive(Debug, Serialize)]
pub struct ReasonFeedbackResponse {
    /// Whether the tree was found and feedback applied.
    pub applied: bool,
    /// Number of neurons whose energy was boosted.
    pub neurons_boosted: u64,
    /// Number of synapses reinforced.
    pub synapses_reinforced: usize,
    /// Energy boost applied.
    pub energy_boost: f64,
    /// Synapse weight boost applied.
    pub synapse_boost: f64,
    /// Number of nodes scarred (on failure).
    /// Biomimicry: Elun Scar — penalizes nodes that led to failed reasoning.
    #[serde(skip_serializing_if = "is_zero_usize")]
    pub scars_applied: usize,
    /// Whether the cached tree was invalidated.
    pub cache_invalidated: bool,
}

fn is_zero_usize(v: &usize) -> bool {
    *v == 0
}

// ============================================================================
// Handlers
// ============================================================================

/// Build a reasoning tree from a natural language query.
///
/// POST /api/reason
///
/// The tree emerges from the knowledge graph in 3 phases:
/// 1. **Activation**: Embed query → vector search notes, decisions, skills
/// 2. **Propagation**: BFS through SYNAPSE, LINKED_TO, AFFECTS with scoring
/// 3. **Cristallisation**: Transform activated subgraph into decision tree
///
/// Trees are cached in-memory (LRU with TTL). Subsequent identical queries
/// return the cached tree until it expires or is invalidated by feedback.
pub async fn reason(
    State(state): State<OrchestratorState>,
    Json(body): Json<ReasonRequest>,
) -> Result<Json<ReasonResponse>, AppError> {
    // Validate
    if body.request.trim().is_empty() {
        return Err(AppError::BadRequest("request cannot be empty".to_string()));
    }

    // Get the reasoning engine
    let engine = state.orchestrator.reasoning_engine().ok_or_else(|| {
        AppError::BadRequest(
            "Reasoning engine unavailable (embedding provider not configured)".to_string(),
        )
    })?;

    // Build config from request params
    let config = ReasoningTreeConfig {
        max_depth: body.depth.unwrap_or(4) as usize,
        max_nodes: body.max_nodes.unwrap_or(50),
        ..Default::default()
    };

    // Build the tree
    let tree = engine
        .build(&body.request, body.project_id, &config)
        .await
        .map_err(AppError::Internal)?;

    // Build response
    let response = ReasonResponse {
        id: tree.id,
        request: tree.request.clone(),
        suggested_actions: if body.include_actions.unwrap_or(true) {
            tree.suggested_actions().into_iter().cloned().collect()
        } else {
            vec![]
        },
        roots: tree.roots.clone(),
        depth: tree.depth,
        confidence: tree.confidence,
        node_count: tree.node_count,
        build_time_ms: tree.build_time_ms.unwrap_or(0),
        project_id: tree.project_id,
    };

    Ok(Json(response))
}

/// Provide feedback on a reasoning tree.
///
/// POST /api/reason/{tree_id}/feedback
///
/// When the agent follows a path in the reasoning tree and achieves a result,
/// it calls this endpoint to reinforce the underlying neural connections:
/// - **Success**: Boost energy of followed nodes + reinforce synapses between them
/// - **Partial**: Half boost
/// - **Failure**: Apply scars to followed nodes (biomimicry: Elun Knowledge Scars)
///
/// The cached tree is invalidated to ensure re-computation with updated scores.
pub async fn reason_feedback(
    State(state): State<OrchestratorState>,
    Path(tree_id): Path<Uuid>,
    Json(body): Json<ReasonFeedbackRequest>,
) -> Result<Json<ReasonFeedbackResponse>, AppError> {
    // Validate
    if body.followed_nodes.is_empty() {
        return Err(AppError::BadRequest(
            "followed_nodes cannot be empty".to_string(),
        ));
    }

    // Determine boost amounts based on outcome
    let (energy_boost, synapse_boost) = match body.outcome {
        FeedbackOutcome::Success => (0.15, 0.05),
        FeedbackOutcome::Partial => (0.075, 0.025),
        FeedbackOutcome::Failure => (0.0, 0.0),
    };

    let neo4j = state.orchestrator.neo4j();
    let mut neurons_boosted = 0u64;
    let mut synapses_reinforced = 0usize;
    let mut scars_applied = 0usize;

    if energy_boost > 0.0 {
        // Boost energy for each followed node
        // The followed_nodes are ReasoningNode IDs which map to note/decision UUIDs
        // We need to boost the underlying notes' energy
        for node_id in &body.followed_nodes {
            // Try to boost as a note (most common entity in reasoning trees)
            if neo4j.boost_energy(*node_id, energy_boost).await.is_ok() {
                neurons_boosted += 1;
            }
        }

        // Reinforce synapses between the followed nodes (Hebbian: co-activated = stronger)
        if body.followed_nodes.len() >= 2 {
            match neo4j
                .reinforce_synapses(&body.followed_nodes, synapse_boost)
                .await
            {
                Ok(count) => {
                    synapses_reinforced = count;
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to reinforce synapses during reasoning feedback"
                    );
                }
            }
        }
    }

    // Biomimicry: Elun Knowledge Scars — apply negative reinforcement on failure
    if matches!(body.outcome, FeedbackOutcome::Failure) {
        match neo4j.apply_scars(&body.followed_nodes, 0.2).await {
            Ok(count) => {
                scars_applied = count;
                tracing::info!(
                    scars_applied = count,
                    tree_id = %tree_id,
                    "Applied knowledge scars to {} nodes on reasoning failure",
                    count
                );
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Failed to apply knowledge scars during reasoning feedback"
                );
            }
        }
    }

    // Biomimicry: Frustration-Catharsis — increment frustration on failure
    if matches!(body.outcome, FeedbackOutcome::Failure) {
        if let Some(task_id) = body.task_id {
            match neo4j.increment_frustration(task_id, 0.3).await {
                Ok(new_score) => {
                    tracing::info!(
                        task_id = %task_id,
                        frustration_score = new_score,
                        "Frustration incremented (+0.3) on reasoning failure"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to increment frustration on reasoning failure"
                    );
                }
            }
        }
    }

    // Invalidate the cached tree so next reason call reflects updated scores
    let cache_invalidated = if let Some(engine) = state.orchestrator.reasoning_engine() {
        engine.invalidate_cache(tree_id).await
    } else {
        false
    };

    Ok(Json(ReasonFeedbackResponse {
        applied: true,
        neurons_boosted,
        synapses_reinforced,
        energy_boost,
        synapse_boost,
        scars_applied,
        cache_invalidated,
    }))
}
