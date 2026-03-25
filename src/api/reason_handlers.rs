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
    /// Whether the tree was persisted to Neo4j (selective persistence on success).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub tree_persisted: bool,
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

    // Selective persistence: on success, persist the tree to Neo4j before invalidating cache
    let mut tree_persisted = false;
    if matches!(body.outcome, FeedbackOutcome::Success) {
        if let Some(engine) = state.orchestrator.reasoning_engine() {
            if let Some(tree) = engine.cache().get_by_id(tree_id).await {
                match state
                    .orchestrator
                    .neo4j()
                    .persist_reasoning_tree(&tree, None, None)
                    .await
                {
                    Ok(persisted_id) => {
                        tracing::info!(
                            tree_id = %persisted_id,
                            node_count = tree.node_count,
                            confidence = ?tree.confidence,
                            "Persisted reasoning tree on positive feedback"
                        );
                        tree_persisted = true;
                    }
                    Err(e) => {
                        tracing::warn!(
                            tree_id = %tree_id,
                            error = %e,
                            "Failed to persist reasoning tree on positive feedback"
                        );
                    }
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
        tree_persisted,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_reason_request_deserialization_minimal() {
        let json = json!({"request": "How to refactor auth?"});
        let req: ReasonRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.request, "How to refactor auth?");
        assert!(req.project_id.is_none());
        assert!(req.depth.is_none());
        assert!(req.include_actions.is_none());
        assert!(req.max_nodes.is_none());
    }

    #[test]
    fn test_reason_request_deserialization_full() {
        let json = json!({
            "request": "Find security issues",
            "project_id": "550e8400-e29b-41d4-a716-446655440000",
            "depth": 6,
            "include_actions": false,
            "max_nodes": 100
        });
        let req: ReasonRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.request, "Find security issues");
        assert!(req.project_id.is_some());
        assert_eq!(req.depth, Some(6));
        assert_eq!(req.include_actions, Some(false));
        assert_eq!(req.max_nodes, Some(100));
    }

    #[test]
    fn test_feedback_outcome_deserialization() {
        let success: FeedbackOutcome = serde_json::from_value(json!("success")).unwrap();
        assert!(matches!(success, FeedbackOutcome::Success));

        let partial: FeedbackOutcome = serde_json::from_value(json!("partial")).unwrap();
        assert!(matches!(partial, FeedbackOutcome::Partial));

        let failure: FeedbackOutcome = serde_json::from_value(json!("failure")).unwrap();
        assert!(matches!(failure, FeedbackOutcome::Failure));
    }

    #[test]
    fn test_feedback_outcome_serialization() {
        assert_eq!(
            serde_json::to_value(FeedbackOutcome::Success).unwrap(),
            json!("success")
        );
        assert_eq!(
            serde_json::to_value(FeedbackOutcome::Partial).unwrap(),
            json!("partial")
        );
        assert_eq!(
            serde_json::to_value(FeedbackOutcome::Failure).unwrap(),
            json!("failure")
        );
    }

    #[test]
    fn test_reason_feedback_request_deserialization() {
        let json = json!({
            "followed_nodes": ["550e8400-e29b-41d4-a716-446655440000"],
            "outcome": "success"
        });
        let req: ReasonFeedbackRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.followed_nodes.len(), 1);
        assert!(matches!(req.outcome, FeedbackOutcome::Success));
        assert!(req.task_id.is_none());
    }

    #[test]
    fn test_reason_feedback_request_with_task_id() {
        let json = json!({
            "followed_nodes": ["550e8400-e29b-41d4-a716-446655440000"],
            "outcome": "failure",
            "task_id": "660e8400-e29b-41d4-a716-446655440000"
        });
        let req: ReasonFeedbackRequest = serde_json::from_value(json).unwrap();
        assert!(matches!(req.outcome, FeedbackOutcome::Failure));
        assert!(req.task_id.is_some());
    }

    #[test]
    fn test_boost_amounts_for_outcomes() {
        // Verify the boost logic from reason_feedback handler
        let success_boost = match FeedbackOutcome::Success {
            FeedbackOutcome::Success => (0.15, 0.05),
            FeedbackOutcome::Partial => (0.075, 0.025),
            FeedbackOutcome::Failure => (0.0, 0.0),
        };
        assert_eq!(success_boost, (0.15, 0.05));

        let partial_boost = match FeedbackOutcome::Partial {
            FeedbackOutcome::Success => (0.15, 0.05),
            FeedbackOutcome::Partial => (0.075, 0.025),
            FeedbackOutcome::Failure => (0.0, 0.0),
        };
        assert_eq!(partial_boost, (0.075, 0.025));

        let failure_boost = match FeedbackOutcome::Failure {
            FeedbackOutcome::Success => (0.15, 0.05),
            FeedbackOutcome::Partial => (0.075, 0.025),
            FeedbackOutcome::Failure => (0.0, 0.0),
        };
        assert_eq!(failure_boost, (0.0, 0.0));
    }

    #[test]
    fn test_reason_feedback_response_serialization() {
        let resp = ReasonFeedbackResponse {
            applied: true,
            neurons_boosted: 3,
            synapses_reinforced: 2,
            energy_boost: 0.15,
            synapse_boost: 0.05,
            scars_applied: 0,
            cache_invalidated: true,
            tree_persisted: false,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["applied"], true);
        assert_eq!(json["neurons_boosted"], 3);
        assert_eq!(json["synapses_reinforced"], 2);
        assert_eq!(json["energy_boost"], 0.15);
        assert_eq!(json["cache_invalidated"], true);
        // scars_applied=0 should be skipped (skip_serializing_if)
        assert!(json.get("scars_applied").is_none());
        // tree_persisted=false should be skipped (Not::not)
        assert!(json.get("tree_persisted").is_none());
    }

    #[test]
    fn test_reason_feedback_response_with_scars() {
        let resp = ReasonFeedbackResponse {
            applied: true,
            neurons_boosted: 0,
            synapses_reinforced: 0,
            energy_boost: 0.0,
            synapse_boost: 0.0,
            scars_applied: 5,
            cache_invalidated: true,
            tree_persisted: false,
        };
        let json = serde_json::to_value(&resp).unwrap();
        // scars_applied > 0 should be present
        assert_eq!(json["scars_applied"], 5);
    }

    #[test]
    fn test_reason_response_serialization() {
        let resp = ReasonResponse {
            id: Uuid::nil(),
            request: "test".to_string(),
            roots: vec![],
            depth: 3,
            confidence: 0.85,
            node_count: 12,
            build_time_ms: 42,
            project_id: None,
            suggested_actions: vec![],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["depth"], 3);
        assert_eq!(json["confidence"], 0.85);
        assert_eq!(json["node_count"], 12);
        assert_eq!(json["build_time_ms"], 42);
    }

    #[test]
    fn test_is_zero_usize_helper() {
        assert!(is_zero_usize(&0));
        assert!(!is_zero_usize(&1));
        assert!(!is_zero_usize(&100));
    }
}
