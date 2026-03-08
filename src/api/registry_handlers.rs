//! API handlers for the Skill Registry
//!
//! Endpoints:
//! - POST /api/registry/publish    — publish a skill to the registry
//! - GET  /api/registry/search     — search published skills
//! - GET  /api/registry/:id        — get a specific published skill (with full package)

use super::handlers::{AppError, OrchestratorState};
use super::hook_handlers::skill_cache;
use super::{PaginatedResponse, PaginationParams};
use crate::skills::registry::{build_published_skill, PublishedSkillSummary};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use uuid::Uuid;

// ============================================================================
// Query / Body structs
// ============================================================================

/// Body for POST /api/registry/publish
#[derive(Debug, Deserialize)]
pub struct PublishBody {
    /// ID of the skill to publish
    pub skill_id: Uuid,
    /// Project ID (ownership verification)
    pub project_id: Uuid,
    /// Optional override for source project name
    pub source_project_name: Option<String>,
}

/// Query parameters for GET /api/registry/search
#[derive(Debug, Deserialize)]
pub struct RegistrySearchQuery {
    /// Full-text search query
    pub query: Option<String>,
    /// Minimum trust score (0.0–1.0)
    pub min_trust: Option<f64>,
    /// Comma-separated tags (AND filter)
    pub tags: Option<String>,
    /// Max results (default 20, max 100)
    #[serde(flatten)]
    pub pagination: PaginationParams,
}

// ============================================================================
// Handlers
// ============================================================================

/// Publish a skill to the local registry.
///
/// POST /api/registry/publish
///
/// 1. Exports the skill as a SkillPackage
/// 2. Computes trust score
/// 3. Stores as PublishedSkill node in Neo4j
pub async fn publish_skill(
    State(state): State<OrchestratorState>,
    Json(body): Json<PublishBody>,
) -> Result<(StatusCode, Json<PublishedSkillSummary>), AppError> {
    let neo4j = state.orchestrator.neo4j();

    // Verify the skill exists and belongs to the project
    let skill = neo4j
        .get_skill(body.skill_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Skill {} not found", body.skill_id)))?;

    if skill.project_id != body.project_id {
        return Err(AppError::BadRequest(format!(
            "Skill {} does not belong to project {}",
            body.skill_id, body.project_id
        )));
    }

    // Determine source project name
    let project_name = if let Some(name) = body.source_project_name {
        name
    } else {
        neo4j
            .get_project(body.project_id)
            .await
            .map_err(AppError::Internal)?
            .map(|p| p.name)
            .unwrap_or_else(|| "unknown".to_string())
    };

    // Export the skill as a package
    let package = crate::skills::export_skill(body.skill_id, neo4j, Some(project_name.clone()))
        .await
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("not found") {
                AppError::NotFound(msg)
            } else {
                AppError::Internal(e)
            }
        })?;

    // Build the published skill with trust score
    let published = build_published_skill(&skill, package, project_name, None);

    // Store in Neo4j
    neo4j
        .upsert_published_skill(&published)
        .await
        .map_err(AppError::Internal)?;

    let summary = PublishedSkillSummary::from(&published);
    Ok((StatusCode::CREATED, Json(summary)))
}

/// Search published skills in the local (and optionally remote) registry.
///
/// GET /api/registry/search
///
/// When `registry_remote_url` is configured, results from the remote registry
/// are fetched concurrently and merged with local results. Remote failures are
/// logged but do not fail the request (graceful degradation).
pub async fn search_registry(
    State(state): State<OrchestratorState>,
    Query(query): Query<RegistrySearchQuery>,
) -> Result<Json<PaginatedResponse<PublishedSkillSummary>>, AppError> {
    let neo4j = state.orchestrator.neo4j();

    let limit = query.pagination.validated_limit().min(100);
    let offset = query.pagination.offset;

    // Parse tags from comma-separated string
    let tags: Option<Vec<String>> = query.tags.as_ref().map(|t| {
        t.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    });

    // Local search
    let (local_items, local_total) = neo4j
        .search_published_skills(
            query.query.as_deref(),
            query.min_trust,
            tags.as_deref(),
            limit,
            offset,
        )
        .await
        .map_err(AppError::Internal)?;

    let local_summaries: Vec<PublishedSkillSummary> =
        local_items.iter().map(|ps| ps.into()).collect();

    // Remote search (if configured)
    if let Some(remote_url) = &state.registry_remote_url {
        match crate::skills::registry::search_remote_registry(
            remote_url,
            query.query.as_deref(),
            query.min_trust,
            tags.as_deref(),
            limit,
            offset,
        )
        .await
        {
            Ok((remote_summaries, remote_total)) => {
                let merged = crate::skills::registry::merge_search_results(
                    local_summaries,
                    remote_summaries,
                );
                let combined_total = local_total + remote_total;
                // Re-paginate the merged results (take `limit` items)
                let page: Vec<PublishedSkillSummary> = merged.into_iter().take(limit).collect();
                return Ok(Json(PaginatedResponse::new(
                    page,
                    combined_total,
                    limit,
                    offset,
                )));
            }
            Err(e) => {
                tracing::warn!("Remote registry search failed (non-fatal): {}", e);
                // Fall through to local-only results
            }
        }
    }

    Ok(Json(PaginatedResponse::new(
        local_summaries,
        local_total,
        limit,
        offset,
    )))
}

/// Get a specific published skill with its full package.
///
/// GET /api/registry/:id
pub async fn get_published_skill(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
) -> Result<Json<crate::skills::registry::PublishedSkill>, AppError> {
    let neo4j = state.orchestrator.neo4j();

    let published = neo4j
        .get_published_skill(id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Published skill {} not found", id)))?;

    Ok(Json(published))
}

/// Import a skill from the registry into a target project.
///
/// POST /api/registry/:id/import
///
/// Fetches the published skill (local or remote), imports the package,
/// and increments the import count.
pub async fn import_from_registry(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
    Json(body): Json<ImportFromRegistryBody>,
) -> Result<(StatusCode, Json<crate::skills::ImportResult>), AppError> {
    let neo4j = state.orchestrator.neo4j();

    // Try local first, then remote if configured
    let published = match neo4j
        .get_published_skill(id)
        .await
        .map_err(AppError::Internal)?
    {
        Some(ps) => ps,
        None => {
            // Try remote registry if configured
            if let Some(remote_url) = &state.registry_remote_url {
                crate::skills::registry::get_remote_published_skill(remote_url, id)
                    .await
                    .map_err(|e| {
                        AppError::NotFound(format!(
                            "Published skill {} not found locally or remotely: {}",
                            id, e
                        ))
                    })?
            } else {
                return Err(AppError::NotFound(format!(
                    "Published skill {} not found",
                    id
                )));
            }
        }
    };

    // Parse conflict strategy
    let strategy = match body.conflict_strategy.as_deref() {
        Some(s) => s
            .parse::<crate::skills::ConflictStrategy>()
            .map_err(AppError::BadRequest)?,
        None => crate::skills::ConflictStrategy::Skip,
    };

    // Import the package into the target project
    let result = crate::skills::import_skill(&published.package, body.project_id, neo4j, strategy)
        .await
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("already exists") {
                AppError::Conflict(msg)
            } else if msg.contains("validation failed") {
                AppError::BadRequest(msg)
            } else {
                AppError::Internal(e)
            }
        })?;

    // Increment import count
    let _ = neo4j.increment_published_skill_imports(id).await;

    // Invalidate skill cache for the target project
    skill_cache().invalidate_project(&body.project_id).await;

    // Spawn event-triggered protocol runs
    crate::protocol::hooks::spawn_event_triggered_protocols(
        state.orchestrator.neo4j_arc(),
        body.project_id,
        "post_import",
    );

    Ok((StatusCode::CREATED, Json(result)))
}

/// Body for POST /api/registry/:id/import
#[derive(Debug, Deserialize)]
pub struct ImportFromRegistryBody {
    /// Target project to import into
    pub project_id: Uuid,
    /// Conflict resolution strategy: "skip", "merge", "replace"
    pub conflict_strategy: Option<String>,
}
