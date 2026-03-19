//! API handlers for Analysis Profiles
//!
//! CRUD for analysis profiles that weight edge types for contextual analytics.

use super::handlers::{AppError, OrchestratorState};
use crate::events::EventEmitter;
use crate::graph::models::AnalysisProfile;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use std::collections::HashMap;

// ============================================================================
// Request / Query DTOs
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ListProfilesQuery {
    pub project_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateProfileBody {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// Project scope — None = global profile
    #[serde(default)]
    pub project_id: Option<String>,
    /// Edge type weights (e.g. {"IMPORTS": 0.8, "CALLS": 0.5})
    #[serde(default)]
    pub edge_weights: HashMap<String, f64>,
    /// Fusion weights for multi-signal analysis
    #[serde(default)]
    pub fusion_weights: Option<crate::graph::models::FusionWeights>,
}

// ============================================================================
// Handlers
// ============================================================================

/// List analysis profiles visible to a project (or all global profiles).
///
/// GET /api/analysis-profiles?project_id=...
pub async fn list_analysis_profiles(
    State(state): State<OrchestratorState>,
    Query(query): Query<ListProfilesQuery>,
) -> Result<Json<Vec<AnalysisProfile>>, AppError> {
    let profiles = state
        .orchestrator
        .neo4j()
        .list_analysis_profiles(query.project_id.as_deref())
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(profiles))
}

/// Create a new analysis profile.
///
/// POST /api/analysis-profiles
pub async fn create_analysis_profile(
    State(state): State<OrchestratorState>,
    Json(body): Json<CreateProfileBody>,
) -> Result<(StatusCode, Json<AnalysisProfile>), AppError> {
    if body.name.trim().is_empty() {
        return Err(AppError::BadRequest("name cannot be empty".to_string()));
    }

    if body.name.len() > 200 {
        return Err(AppError::BadRequest(
            "name must be 200 characters or less".to_string(),
        ));
    }

    // Validate edge weights are non-negative
    for (key, &weight) in &body.edge_weights {
        if weight < 0.0 {
            return Err(AppError::BadRequest(format!(
                "edge weight for '{}' must be non-negative, got {}",
                key, weight
            )));
        }
    }

    // Build the fusion weights, validating if provided
    let fusion_weights = if let Some(ref fw) = body.fusion_weights {
        fw.validate().map_err(AppError::BadRequest)?;
        fw.clone()
    } else {
        Default::default()
    };

    let profile = AnalysisProfile {
        id: uuid::Uuid::new_v4().to_string(),
        project_id: body.project_id,
        name: body.name,
        description: body.description,
        edge_weights: body.edge_weights,
        fusion_weights,
        is_builtin: false, // User-created profiles are never built-in
    };

    state
        .orchestrator
        .neo4j()
        .create_analysis_profile(&profile)
        .await
        .map_err(AppError::Internal)?;

    state.event_bus.emit_created(
        crate::events::EntityType::AnalysisProfile,
        &profile.id,
        serde_json::json!({
            "name": profile.name,
            "project_id": profile.project_id,
        }),
        profile.project_id.clone(),
    );

    Ok((StatusCode::CREATED, Json(profile)))
}

/// Get a single analysis profile by ID.
///
/// GET /api/analysis-profiles/:id
pub async fn get_analysis_profile(
    State(state): State<OrchestratorState>,
    Path(profile_id): Path<String>,
) -> Result<Json<AnalysisProfile>, AppError> {
    let profile = state
        .orchestrator
        .neo4j()
        .get_analysis_profile(&profile_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| {
            AppError::NotFound(format!("Analysis profile '{}' not found", profile_id))
        })?;

    Ok(Json(profile))
}

/// Delete an analysis profile by ID.
///
/// DELETE /api/analysis-profiles/:id
/// Returns 400 if the profile is built-in.
pub async fn delete_analysis_profile(
    State(state): State<OrchestratorState>,
    Path(profile_id): Path<String>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .delete_analysis_profile(&profile_id)
        .await
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("Cannot delete built-in") {
                AppError::BadRequest(msg)
            } else {
                AppError::Internal(e)
            }
        })?;

    state.event_bus.emit_deleted(
        crate::events::EntityType::AnalysisProfile,
        &profile_id,
        None,
    );

    Ok(StatusCode::NO_CONTENT)
}
