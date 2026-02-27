//! API handlers for Neural Skills

use super::handlers::{AppError, OrchestratorState};
use super::hook_handlers::skill_cache;
use super::{PaginatedResponse, PaginationParams};
use crate::skills::{
    ActivatedSkillContext, ConflictStrategy, ImportResult, SkillNode, SkillStatus, SkillTrigger,
    TriggerType, MAX_TRIGGER_PATTERN_LEN, REGEX_DFA_SIZE_LIMIT, REGEX_SIZE_LIMIT,
};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use chrono::Utc;
use serde::Deserialize;
use uuid::Uuid;

// ============================================================================
// Validation constants
// ============================================================================

const MAX_NAME_LEN: usize = 200;
const MAX_DESCRIPTION_LEN: usize = 5000;
const MAX_TAG_LEN: usize = 100;
const MAX_TAGS_COUNT: usize = 50;
const MAX_CONTEXT_TEMPLATE_LEN: usize = 50_000;

// ============================================================================
// Validation helpers
// ============================================================================

/// Validate trigger patterns: compile regexes with size limits, validate globs.
fn validate_trigger_patterns(triggers: &[SkillTrigger]) -> Result<(), String> {
    for (i, trigger) in triggers.iter().enumerate() {
        if trigger.pattern_value.len() > MAX_TRIGGER_PATTERN_LEN {
            return Err(format!(
                "trigger_patterns[{}]: pattern too long ({} > {} chars)",
                i,
                trigger.pattern_value.len(),
                MAX_TRIGGER_PATTERN_LEN
            ));
        }
        if !(0.0..=1.0).contains(&trigger.confidence_threshold) {
            return Err(format!(
                "trigger_patterns[{}]: confidence_threshold must be 0.0-1.0, got {}",
                i, trigger.confidence_threshold
            ));
        }
        match trigger.pattern_type {
            TriggerType::Regex => {
                if let Err(e) = regex::RegexBuilder::new(&trigger.pattern_value)
                    .case_insensitive(true)
                    .size_limit(REGEX_SIZE_LIMIT)
                    .dfa_size_limit(REGEX_DFA_SIZE_LIMIT)
                    .build()
                {
                    return Err(format!(
                        "trigger_patterns[{}]: invalid regex '{}': {}",
                        i, trigger.pattern_value, e
                    ));
                }
            }
            TriggerType::FileGlob => {
                if let Err(e) = glob::Pattern::new(&trigger.pattern_value) {
                    return Err(format!(
                        "trigger_patterns[{}]: invalid glob '{}': {}",
                        i, trigger.pattern_value, e
                    ));
                }
            }
            TriggerType::Semantic => {
                // Semantic triggers contain embedding vectors — no compile-time validation needed
            }
        }
    }
    Ok(())
}

/// Validate input length limits for skill fields.
fn validate_input_limits(
    name: Option<&str>,
    description: Option<&str>,
    tags: Option<&[String]>,
    context_template: Option<&str>,
) -> Result<(), String> {
    if let Some(name) = name {
        if name.len() > MAX_NAME_LEN {
            return Err(format!(
                "name too long ({} > {} chars)",
                name.len(),
                MAX_NAME_LEN
            ));
        }
    }
    if let Some(desc) = description {
        if desc.len() > MAX_DESCRIPTION_LEN {
            return Err(format!(
                "description too long ({} > {} chars)",
                desc.len(),
                MAX_DESCRIPTION_LEN
            ));
        }
    }
    if let Some(tags) = tags {
        if tags.len() > MAX_TAGS_COUNT {
            return Err(format!(
                "too many tags ({} > {})",
                tags.len(),
                MAX_TAGS_COUNT
            ));
        }
        for (i, tag) in tags.iter().enumerate() {
            if tag.len() > MAX_TAG_LEN {
                return Err(format!(
                    "tags[{}] too long ({} > {} chars)",
                    i,
                    tag.len(),
                    MAX_TAG_LEN
                ));
            }
        }
    }
    if let Some(tpl) = context_template {
        if tpl.len() > MAX_CONTEXT_TEMPLATE_LEN {
            return Err(format!(
                "context_template too long ({} > {} chars)",
                tpl.len(),
                MAX_CONTEXT_TEMPLATE_LEN
            ));
        }
    }
    Ok(())
}

// ============================================================================
// Query Parameters
// ============================================================================

/// Query parameters for listing skills
#[derive(Debug, Deserialize, Default)]
pub struct SkillsListQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
    /// Required: filter skills by project
    pub project_id: Uuid,
    /// Optional: filter by status (emerging, active, dormant, archived, imported)
    pub status: Option<String>,
}

// ============================================================================
// Request Bodies
// ============================================================================

/// Request body for creating a skill
#[derive(Debug, Deserialize)]
pub struct CreateSkillBody {
    pub project_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub tags: Option<Vec<String>>,
    pub trigger_patterns: Option<Vec<SkillTrigger>>,
    pub context_template: Option<String>,
}

/// Request body for updating a skill
#[derive(Debug, Deserialize)]
pub struct UpdateSkillBody {
    pub name: Option<String>,
    pub description: Option<String>,
    pub status: Option<String>,
    pub tags: Option<Vec<String>>,
    pub trigger_patterns: Option<Vec<SkillTrigger>>,
    pub context_template: Option<String>,
    pub energy: Option<f64>,
    pub cohesion: Option<f64>,
}

/// Request body for adding a member to a skill
#[derive(Debug, Deserialize)]
pub struct AddMemberBody {
    /// "note" or "decision"
    pub entity_type: String,
    pub entity_id: Uuid,
}

/// Response for skill members
#[derive(Debug, serde::Serialize)]
pub struct SkillMembersResponse {
    pub notes: Vec<crate::notes::Note>,
    pub decisions: Vec<crate::neo4j::models::DecisionNode>,
}

/// Request body for activating a skill
#[derive(Debug, Deserialize)]
pub struct ActivateSkillBody {
    pub query: String,
}

/// Query parameters for export
#[derive(Debug, Deserialize, Default)]
pub struct ExportQuery {
    /// Optional source project name for metadata
    pub source_project_name: Option<String>,
}

/// Request body for importing a skill
#[derive(Debug, Deserialize)]
pub struct ImportSkillBody {
    /// Target project ID
    pub project_id: Uuid,
    /// The SkillPackage to import
    pub package: crate::skills::SkillPackage,
    /// Conflict strategy: "skip" (default), "merge", "replace"
    #[serde(default)]
    pub conflict_strategy: Option<String>,
}

// ============================================================================
// Handlers — CRUD
// ============================================================================

/// List skills for a project
///
/// GET /api/skills?project_id=...&status=...&limit=...&offset=...
pub async fn list_skills(
    State(state): State<OrchestratorState>,
    Query(query): Query<SkillsListQuery>,
) -> Result<Json<PaginatedResponse<SkillNode>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let status_filter = match &query.status {
        Some(s) => Some(s.parse::<SkillStatus>().map_err(AppError::BadRequest)?),
        None => None,
    };

    let limit = query.pagination.validated_limit();
    let offset = query.pagination.offset;

    let (skills, total) = state
        .orchestrator
        .neo4j()
        .list_skills(query.project_id, status_filter, limit, offset)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(PaginatedResponse::new(skills, total, limit, offset)))
}

/// Create a new skill
///
/// POST /api/skills
pub async fn create_skill(
    State(state): State<OrchestratorState>,
    Json(body): Json<CreateSkillBody>,
) -> Result<(StatusCode, Json<SkillNode>), AppError> {
    if body.name.trim().is_empty() {
        return Err(AppError::BadRequest("name cannot be empty".to_string()));
    }

    // Validate input limits
    validate_input_limits(
        Some(&body.name),
        body.description.as_deref(),
        body.tags.as_deref(),
        body.context_template.as_deref(),
    )
    .map_err(AppError::BadRequest)?;

    // Validate trigger patterns
    if let Some(ref triggers) = body.trigger_patterns {
        validate_trigger_patterns(triggers).map_err(AppError::BadRequest)?;
    }

    let mut skill = SkillNode::new(body.project_id, body.name);
    if let Some(desc) = body.description {
        skill.description = desc;
    }
    if let Some(tags) = body.tags {
        skill.tags = tags;
    }
    if let Some(triggers) = body.trigger_patterns {
        skill.trigger_patterns = triggers;
    }
    skill.context_template = body.context_template;

    state
        .orchestrator
        .neo4j()
        .create_skill(&skill)
        .await
        .map_err(AppError::Internal)?;

    // Invalidate hook activation cache for this project
    skill_cache().invalidate_project(&skill.project_id).await;

    Ok((StatusCode::CREATED, Json(skill)))
}

/// Get a skill by ID
///
/// GET /api/skills/:id
pub async fn get_skill(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
) -> Result<Json<SkillNode>, AppError> {
    let skill = state
        .orchestrator
        .neo4j()
        .get_skill(skill_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Skill {} not found", skill_id)))?;

    Ok(Json(skill))
}

/// Update a skill
///
/// PUT /api/skills/:id
pub async fn update_skill(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
    Json(body): Json<UpdateSkillBody>,
) -> Result<Json<SkillNode>, AppError> {
    let mut skill = state
        .orchestrator
        .neo4j()
        .get_skill(skill_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Skill {} not found", skill_id)))?;

    // Validate input limits
    validate_input_limits(
        body.name.as_deref(),
        body.description.as_deref(),
        body.tags.as_deref(),
        body.context_template.as_deref(),
    )
    .map_err(AppError::BadRequest)?;

    // Validate trigger patterns
    if let Some(ref triggers) = body.trigger_patterns {
        validate_trigger_patterns(triggers).map_err(AppError::BadRequest)?;
    }

    // Apply updates
    if let Some(name) = body.name {
        if name.trim().is_empty() {
            return Err(AppError::BadRequest("name cannot be empty".to_string()));
        }
        skill.name = name;
    }
    if let Some(description) = body.description {
        skill.description = description;
    }
    if let Some(status_str) = body.status {
        let new_status: SkillStatus = status_str
            .parse()
            .map_err(|e: String| AppError::BadRequest(e))?;
        if !skill.status.can_transition_to(new_status) {
            return Err(AppError::BadRequest(format!(
                "Invalid status transition: {} → {}",
                skill.status, new_status
            )));
        }
        skill.status = new_status;
    }
    if let Some(tags) = body.tags {
        skill.tags = tags;
    }
    if let Some(triggers) = body.trigger_patterns {
        skill.trigger_patterns = triggers;
    }
    if let Some(template) = body.context_template {
        skill.context_template = Some(template);
    }
    if let Some(energy) = body.energy {
        skill.energy = energy.clamp(0.0, 1.0);
    }
    if let Some(cohesion) = body.cohesion {
        skill.cohesion = cohesion.clamp(0.0, 1.0);
    }
    skill.updated_at = Utc::now();

    state
        .orchestrator
        .neo4j()
        .update_skill(&skill)
        .await
        .map_err(AppError::Internal)?;

    // Invalidate hook activation cache for this project
    skill_cache().invalidate_project(&skill.project_id).await;

    Ok(Json(skill))
}

/// Delete a skill
///
/// DELETE /api/skills/:id
pub async fn delete_skill(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    // Get project_id before deletion for cache invalidation
    let skill = state
        .orchestrator
        .neo4j()
        .get_skill(skill_id)
        .await
        .map_err(AppError::Internal)?;

    let deleted = state
        .orchestrator
        .neo4j()
        .delete_skill(skill_id)
        .await
        .map_err(AppError::Internal)?;

    if deleted {
        // Invalidate hook activation cache for this project
        if let Some(skill) = skill {
            skill_cache().invalidate_project(&skill.project_id).await;
        }
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!("Skill {} not found", skill_id)))
    }
}

// ============================================================================
// Handlers — Membership
// ============================================================================

/// Get members (notes + decisions) of a skill
///
/// GET /api/skills/:id/members
pub async fn get_skill_members(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
) -> Result<Json<SkillMembersResponse>, AppError> {
    let (notes, decisions) = state
        .orchestrator
        .neo4j()
        .get_skill_members(skill_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(SkillMembersResponse { notes, decisions }))
}

/// Add a member (note or decision) to a skill
///
/// POST /api/skills/:id/members
pub async fn add_skill_member(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
    Json(body): Json<AddMemberBody>,
) -> Result<StatusCode, AppError> {
    if !["note", "decision"].contains(&body.entity_type.as_str()) {
        return Err(AppError::BadRequest(
            "entity_type must be 'note' or 'decision'".to_string(),
        ));
    }

    // Pre-check skill existence to return 404 instead of silent 201
    let skill = state
        .orchestrator
        .neo4j()
        .get_skill(skill_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Skill {} not found", skill_id)))?;

    state
        .orchestrator
        .neo4j()
        .add_skill_member(skill_id, &body.entity_type, body.entity_id)
        .await
        .map_err(AppError::Internal)?;

    // Invalidate cache — membership change may affect trigger evaluation
    skill_cache().invalidate_project(&skill.project_id).await;

    Ok(StatusCode::CREATED)
}

/// Remove a member from a skill
///
/// DELETE /api/skills/:id/members/:entity_type/:entity_id
pub async fn remove_skill_member(
    State(state): State<OrchestratorState>,
    Path((skill_id, entity_type, entity_id)): Path<(Uuid, String, Uuid)>,
) -> Result<StatusCode, AppError> {
    if !["note", "decision"].contains(&entity_type.as_str()) {
        return Err(AppError::BadRequest(
            "entity_type must be 'note' or 'decision'".to_string(),
        ));
    }

    // Get skill for project_id (needed for cache invalidation)
    let skill = state
        .orchestrator
        .neo4j()
        .get_skill(skill_id)
        .await
        .map_err(AppError::Internal)?;

    let removed = state
        .orchestrator
        .neo4j()
        .remove_skill_member(skill_id, &entity_type, entity_id)
        .await
        .map_err(AppError::Internal)?;

    if removed {
        // Invalidate cache — membership change may affect trigger evaluation
        if let Some(skill) = skill {
            skill_cache().invalidate_project(&skill.project_id).await;
        }
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!(
            "Member {} ({}) not found in skill {}",
            entity_id, entity_type, skill_id
        )))
    }
}

// ============================================================================
// Handlers — Activation
// ============================================================================

/// Activate a skill — retrieve enriched context from its members
///
/// POST /api/skills/:id/activate
pub async fn activate_skill(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
    Json(body): Json<ActivateSkillBody>,
) -> Result<Json<ActivatedSkillContext>, AppError> {
    if body.query.trim().is_empty() {
        return Err(AppError::BadRequest("query cannot be empty".to_string()));
    }

    // activate_skill already does get_skill internally — no need for pre-fetch.
    // The error message from activate_skill("Skill not found: <id>") is sufficient.
    let context = state
        .orchestrator
        .neo4j()
        .activate_skill(skill_id, &body.query)
        .await
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("not found") {
                AppError::NotFound(format!("Skill {} not found", skill_id))
            } else {
                AppError::Internal(e)
            }
        })?;

    Ok(Json(context))
}

// ============================================================================
// Handlers — Export / Import
// ============================================================================

/// Export a skill as a portable SkillPackage
///
/// GET /api/skills/:id/export?source_project_name=...
pub async fn export_skill(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
    Query(query): Query<ExportQuery>,
) -> Result<Json<crate::skills::SkillPackage>, AppError> {
    let package = crate::skills::export_skill(
        skill_id,
        state.orchestrator.neo4j(),
        query.source_project_name,
    )
    .await
    .map_err(|e| {
        let msg = e.to_string();
        if msg.contains("not found") {
            AppError::NotFound(format!("Skill {} not found", skill_id))
        } else {
            AppError::Internal(e)
        }
    })?;

    Ok(Json(package))
}

/// Import a skill from a SkillPackage into a target project
///
/// POST /api/skills/import
pub async fn import_skill(
    State(state): State<OrchestratorState>,
    Json(body): Json<ImportSkillBody>,
) -> Result<(StatusCode, Json<ImportResult>), AppError> {
    // Parse conflict strategy (default: Skip)
    let strategy = match body.conflict_strategy.as_deref() {
        Some(s) => s.parse::<ConflictStrategy>().map_err(AppError::BadRequest)?,
        None => ConflictStrategy::Skip,
    };

    let result = crate::skills::import_skill(
        &body.package,
        body.project_id,
        state.orchestrator.neo4j(),
        strategy,
    )
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

    // Invalidate skill cache for the target project
    skill_cache().invalidate_project(&body.project_id).await;

    Ok((StatusCode::CREATED, Json(result)))
}

// ============================================================================
// Handlers — Health
// ============================================================================

/// Get health metrics for a skill
///
/// GET /api/skills/:id/health
pub async fn get_skill_health(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
) -> Result<Json<crate::skills::SkillHealthMetrics>, AppError> {
    let skill = state
        .orchestrator
        .neo4j()
        .get_skill(skill_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Skill {} not found", skill_id)))?;

    let health = crate::skills::validation::compute_health(&skill, Utc::now());

    Ok(Json(health))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_trigger_patterns_valid_regex() {
        let triggers = vec![SkillTrigger::regex("neo4j|cypher", 0.6)];
        assert!(validate_trigger_patterns(&triggers).is_ok());
    }

    #[test]
    fn test_validate_trigger_patterns_invalid_regex() {
        let triggers = vec![SkillTrigger::regex("[invalid", 0.6)];
        let result = validate_trigger_patterns(&triggers);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid regex"));
    }

    #[test]
    fn test_validate_trigger_patterns_catastrophic_regex() {
        // Regex that would cause exponential backtracking / exceed size limits
        let evil = "(a+)+$".repeat(100);
        let triggers = vec![SkillTrigger::regex(evil, 0.6)];
        let result = validate_trigger_patterns(&triggers);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_trigger_patterns_valid_glob() {
        let triggers = vec![SkillTrigger::file_glob("src/neo4j/**", 0.8)];
        assert!(validate_trigger_patterns(&triggers).is_ok());
    }

    #[test]
    fn test_validate_trigger_patterns_invalid_glob() {
        let triggers = vec![SkillTrigger::file_glob("[invalid", 0.8)];
        let result = validate_trigger_patterns(&triggers);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid glob"));
    }

    #[test]
    fn test_validate_trigger_patterns_too_long() {
        let long_pattern = "a".repeat(MAX_TRIGGER_PATTERN_LEN + 1);
        let triggers = vec![SkillTrigger::regex(long_pattern, 0.6)];
        let result = validate_trigger_patterns(&triggers);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("pattern too long"));
    }

    #[test]
    fn test_validate_trigger_patterns_bad_confidence() {
        // Bypass constructor clamping to simulate deserialized input with bad confidence
        let triggers = vec![SkillTrigger {
            pattern_type: TriggerType::Regex,
            pattern_value: "test".to_string(),
            confidence_threshold: 1.5,
            quality_score: None,
        }];
        let result = validate_trigger_patterns(&triggers);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("confidence_threshold"));
    }

    #[test]
    fn test_validate_trigger_patterns_semantic_always_valid() {
        let triggers = vec![SkillTrigger::semantic("[0.1, 0.2]", 0.7)];
        assert!(validate_trigger_patterns(&triggers).is_ok());
    }

    #[test]
    fn test_validate_input_limits_name_too_long() {
        let name = "a".repeat(MAX_NAME_LEN + 1);
        let result = validate_input_limits(Some(&name), None, None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("name too long"));
    }

    #[test]
    fn test_validate_input_limits_description_too_long() {
        let desc = "a".repeat(MAX_DESCRIPTION_LEN + 1);
        let result = validate_input_limits(None, Some(&desc), None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("description too long"));
    }

    #[test]
    fn test_validate_input_limits_too_many_tags() {
        let tags: Vec<String> = (0..MAX_TAGS_COUNT + 1)
            .map(|i| format!("tag{}", i))
            .collect();
        let result = validate_input_limits(None, None, Some(&tags), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too many tags"));
    }

    #[test]
    fn test_validate_input_limits_tag_too_long() {
        let tags = vec!["a".repeat(MAX_TAG_LEN + 1)];
        let result = validate_input_limits(None, None, Some(&tags), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("tags[0] too long"));
    }

    #[test]
    fn test_validate_input_limits_context_template_too_long() {
        let tpl = "a".repeat(MAX_CONTEXT_TEMPLATE_LEN + 1);
        let result = validate_input_limits(None, None, None, Some(&tpl));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("context_template too long"));
    }

    #[test]
    fn test_validate_input_limits_all_valid() {
        let tags = vec!["tag1".to_string(), "tag2".to_string()];
        let result = validate_input_limits(
            Some("Valid Name"),
            Some("Valid description"),
            Some(&tags),
            Some("Valid template"),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_input_limits_none_values() {
        assert!(validate_input_limits(None, None, None, None).is_ok());
    }
}
