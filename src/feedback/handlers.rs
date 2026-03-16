//! REST API handlers for the feedback system.
//!
//! - POST /api/feedback — submit explicit feedback
//! - GET  /api/feedback — list feedback entries
//! - GET  /api/feedback/stats — aggregated statistics

use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use tracing::{debug, warn};

use crate::api::handlers::{AppError, OrchestratorState};

use super::models::*;
use super::tracker::OutcomeTracker;

// ============================================================================
// POST /api/feedback
// ============================================================================

/// Submit explicit feedback for a target entity.
///
/// Score must be in [-1.0, +1.0]. Returns 400 if out of range.
/// Automatically triggers score propagation in the background.
pub async fn create_feedback(
    State(state): State<OrchestratorState>,
    Json(req): Json<CreateFeedbackRequest>,
) -> Result<(StatusCode, Json<ExplicitFeedback>), AppError> {
    // Validate score range
    if !(-1.0..=1.0).contains(&req.score) {
        return Err(AppError::BadRequest(format!(
            "Score must be in [-1.0, +1.0], got {}",
            req.score
        )));
    }

    let feedback = ExplicitFeedback::new(
        req.target_type,
        req.target_id,
        req.score,
        "api".to_string(), // TODO: extract from auth context
    )
    .ok_or_else(|| AppError::BadRequest("Invalid feedback parameters".into()))?;

    let feedback = if let Some(comment) = req.comment {
        feedback.with_comment(comment)
    } else {
        feedback
    };

    let feedback = if let Some(pid) = req.project_id {
        feedback.with_project(pid)
    } else {
        feedback
    };

    let feedback = if let Some(sid) = req.session_id {
        feedback.with_session(sid)
    } else {
        feedback
    };

    // Store feedback via the OutcomeTracker
    let tracker = OutcomeTracker::global();
    tracker.record_explicit_feedback(feedback.clone()).await;

    // Trigger score propagation in the background
    let graph = state.orchestrator.neo4j_arc();
    let fb_clone = feedback.clone();
    tokio::spawn(async move {
        if let Err(e) = super::propagator::propagate_feedback(graph, &fb_clone).await {
            warn!("[feedback] Propagation failed for {}: {}", fb_clone.id, e);
        }
    });

    debug!(
        "[feedback] Recorded explicit feedback {} for {:?}/{}",
        feedback.id, feedback.target_type, feedback.target_id
    );

    Ok((StatusCode::CREATED, Json(feedback)))
}

// ============================================================================
// GET /api/feedback
// ============================================================================

/// List feedback entries with optional filters.
pub async fn list_feedback(
    State(_state): State<OrchestratorState>,
    Query(query): Query<ListFeedbackQuery>,
) -> Result<Json<Vec<ExplicitFeedback>>, AppError> {
    let tracker = OutcomeTracker::global();
    let entries = tracker
        .list_feedback(
            query.target_type.as_deref(),
            query.target_id,
            query.project_id,
            query.limit,
            query.offset,
        )
        .await;

    Ok(Json(entries))
}

// ============================================================================
// GET /api/feedback/stats
// ============================================================================

/// Get aggregated feedback statistics.
pub async fn get_feedback_stats(
    State(_state): State<OrchestratorState>,
    Query(query): Query<FeedbackStatsQuery>,
) -> Result<Json<FeedbackStatsResponse>, AppError> {
    let tracker = OutcomeTracker::global();
    let response = tracker
        .compute_stats(
            query.target_type.as_deref(),
            query.target_id,
            query.project_id,
        )
        .await;

    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_validation_in_range() {
        assert!((-1.0_f64..=1.0_f64).contains(&0.0));
        assert!((-1.0_f64..=1.0_f64).contains(&1.0));
        assert!((-1.0_f64..=1.0_f64).contains(&-1.0));
        assert!((-1.0_f64..=1.0_f64).contains(&0.5));
    }

    #[test]
    fn test_score_validation_out_of_range() {
        assert!(!(-1.0_f64..=1.0_f64).contains(&1.1));
        assert!(!(-1.0_f64..=1.0_f64).contains(&-1.1));
        assert!(!(-1.0_f64..=1.0_f64).contains(&f64::INFINITY));
    }

    #[test]
    fn test_create_feedback_request_deserialization() {
        let json = r#"{
            "target_type": "note",
            "target_id": "550e8400-e29b-41d4-a716-446655440000",
            "score": 0.8
        }"#;
        let req: CreateFeedbackRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.target_type, FeedbackTarget::Note);
        assert!((req.score - 0.8).abs() < f64::EPSILON);
        assert!(req.comment.is_none());
        assert!(req.project_id.is_none());
        assert!(req.session_id.is_none());
    }

    #[test]
    fn test_create_feedback_request_full() {
        let json = r#"{
            "target_type": "task",
            "target_id": "550e8400-e29b-41d4-a716-446655440000",
            "score": -0.5,
            "comment": "Could be better",
            "project_id": "660e8400-e29b-41d4-a716-446655440000",
            "session_id": "770e8400-e29b-41d4-a716-446655440000"
        }"#;
        let req: CreateFeedbackRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.target_type, FeedbackTarget::Task);
        assert!((req.score - (-0.5)).abs() < f64::EPSILON);
        assert_eq!(req.comment.unwrap(), "Could be better");
        assert!(req.project_id.is_some());
        assert!(req.session_id.is_some());
    }

    #[test]
    fn test_list_feedback_query_defaults() {
        let json = r#"{}"#;
        let query: ListFeedbackQuery = serde_json::from_str(json).unwrap();
        assert!(query.target_type.is_none());
        assert!(query.target_id.is_none());
        assert!(query.project_id.is_none());
    }

    #[test]
    fn test_feedback_stats_query_deserialization() {
        let json = r#"{"target_type": "note"}"#;
        let query: FeedbackStatsQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.target_type.unwrap(), "note");
    }

    #[tokio::test]
    async fn test_outcome_tracker_global_singleton() {
        let tracker1 = OutcomeTracker::global();
        let tracker2 = OutcomeTracker::global();
        let stats1 = tracker1.compute_stats(None, None, None).await;
        let stats2 = tracker2.compute_stats(None, None, None).await;
        assert_eq!(stats1.total_feedback_count, stats2.total_feedback_count);
    }
}
