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
            TriggerType::McpAction => {
                // McpAction patterns are simple strings — validate non-empty
                if trigger.pattern_value.trim().is_empty() {
                    return Err(format!(
                        "trigger_patterns[{}]: mcp_action pattern must not be empty",
                        i
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
        Some(s) => s
            .parse::<ConflictStrategy>()
            .map_err(AppError::BadRequest)?,
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

    // Spawn event-triggered protocol runs (post_import)
    crate::protocol::hooks::spawn_event_triggered_protocols(
        state.orchestrator.neo4j_arc(),
        body.project_id,
        "post_import",
    );

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
// Skill Split & Merge (Biomimicry — Evolution Guards)
// ============================================================================

/// Request body for POST /api/skills/:id/split
#[derive(Debug, Deserialize)]
pub struct SplitSkillRequest {
    /// Optional: specific sub-cluster note IDs for each new skill.
    /// If omitted, re-runs detection to find sub-clusters automatically.
    pub sub_clusters: Option<Vec<Vec<String>>>,
}

/// Request body for POST /api/skills/merge
#[derive(Debug, Deserialize)]
pub struct MergeSkillsRequest {
    /// The skill IDs to merge together (minimum 2).
    pub skill_ids: Vec<Uuid>,
}

/// POST /api/skills/:id/split
///
/// Split a skill into multiple sub-skills based on cluster detection
/// or explicitly provided sub-cluster note IDs.
/// Archives the original skill and creates N new skills.
pub async fn split_skill(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
    Json(body): Json<SplitSkillRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::skills::evolution::{execute_evolution, SkillEvolution, MIN_MEMBERS_FOR_SPLIT};

    let graph = state.orchestrator.neo4j();

    // Verify skill exists and is not archived
    let skill = graph
        .get_skill(skill_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Skill {} not found", skill_id)))?;

    if skill.status == SkillStatus::Archived {
        return Err(AppError::BadRequest(
            "Cannot split an archived skill".into(),
        ));
    }

    // Get current members
    let (member_notes, _) = graph
        .get_skill_members(skill_id)
        .await
        .map_err(AppError::Internal)?;
    let member_ids: Vec<String> = member_notes.iter().map(|n| n.id.to_string()).collect();

    // Guardrail check
    if member_ids.len() < MIN_MEMBERS_FOR_SPLIT {
        return Err(AppError::BadRequest(format!(
            "Skill has {} members, minimum {} required for split",
            member_ids.len(),
            MIN_MEMBERS_FOR_SPLIT
        )));
    }

    // Build candidates from explicit sub-clusters or auto-detect
    let candidates: Vec<crate::skills::detection::SkillCandidate> = if let Some(sub_clusters) =
        body.sub_clusters
    {
        if sub_clusters.len() < 2 {
            return Err(AppError::BadRequest(
                "Need at least 2 sub-clusters for split".into(),
            ));
        }
        sub_clusters
            .into_iter()
            .enumerate()
            .map(|(i, notes)| crate::skills::detection::SkillCandidate {
                community_id: i as u32,
                size: notes.len(),
                cohesion: 0.5, // default for manual splits
                member_note_ids: notes,
                label: format!("{}-part{}", skill.name, i + 1),
            })
            .collect()
    } else {
        // Auto-detect sub-clusters via SYNAPSE graph
        let config = crate::skills::detection::SkillDetectionConfig::default();
        let edges = graph
            .get_synapse_graph(skill.project_id, config.min_synapse_weight)
            .await
            .map_err(AppError::Internal)?;
        let detection = crate::skills::detection::detect_skill_candidates(
            &edges,
            &skill.project_id.to_string(),
            &config,
        );
        // Find candidates overlapping with this skill
        let overlap_threshold = 0.3; // lower threshold for sub-cluster detection
        let matching: Vec<_> = detection
            .candidates
            .into_iter()
            .filter(|c| {
                let jaccard =
                    crate::skills::detection::jaccard_similarity(&c.member_note_ids, &member_ids);
                jaccard >= overlap_threshold
            })
            .collect();

        if matching.len() < 2 {
            return Err(AppError::BadRequest(
                "No sub-clusters detected for this skill. Provide explicit sub_clusters or wait for cluster divergence.".into()
            ));
        }
        matching
    };

    // Build evolution action and execute
    let evolution = vec![SkillEvolution::Split {
        skill_id,
        candidates: candidates.clone(),
    }];

    // Build notes_map for execution
    let mut notes_map = std::collections::HashMap::new();
    for note in &member_notes {
        notes_map.insert(note.id.to_string(), note.clone());
    }

    let result = execute_evolution(graph, &evolution, &notes_map, skill.project_id)
        .await
        .map_err(AppError::Internal)?;

    // Invalidate hook cache
    skill_cache().invalidate_project(&skill.project_id).await;

    let new_ids: Vec<Uuid> = result
        .split
        .iter()
        .flat_map(|e| e.new_skill_ids.clone())
        .collect();

    Ok(Json(serde_json::json!({
        "original_skill_id": skill_id,
        "archived": true,
        "new_skill_ids": new_ids,
        "sub_cluster_count": candidates.len(),
    })))
}

/// POST /api/skills/merge
///
/// Merge multiple skills into one. The skill with the highest energy
/// survives; others are archived and their members transferred.
pub async fn merge_skills(
    State(state): State<OrchestratorState>,
    Json(body): Json<MergeSkillsRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::skills::evolution::{execute_evolution, SkillEvolution, MAX_MEMBERS_AFTER_MERGE};

    if body.skill_ids.len() < 2 {
        return Err(AppError::BadRequest(
            "Need at least 2 skill IDs to merge".into(),
        ));
    }

    let graph = state.orchestrator.neo4j();

    // Verify all skills exist and are not archived
    let mut project_id: Option<Uuid> = None;
    let mut total_members = 0usize;
    let mut all_member_ids: Vec<String> = Vec::new();

    for &sid in &body.skill_ids {
        let skill = graph
            .get_skill(sid)
            .await
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::NotFound(format!("Skill {} not found", sid)))?;

        if skill.status == SkillStatus::Archived {
            return Err(AppError::BadRequest(format!("Skill {} is archived", sid)));
        }

        // Ensure all skills belong to the same project
        match project_id {
            None => project_id = Some(skill.project_id),
            Some(pid) if pid != skill.project_id => {
                return Err(AppError::BadRequest(
                    "Cannot merge skills from different projects".into(),
                ));
            }
            _ => {}
        }

        let (notes, _) = graph
            .get_skill_members(sid)
            .await
            .map_err(AppError::Internal)?;
        total_members += notes.len();
        all_member_ids.extend(notes.iter().map(|n| n.id.to_string()));
    }

    let project_id = project_id.unwrap();

    // Guardrail check
    if total_members > MAX_MEMBERS_AFTER_MERGE {
        return Err(AppError::BadRequest(format!(
            "Combined {} members exceeds maximum {} for merge",
            total_members, MAX_MEMBERS_AFTER_MERGE
        )));
    }

    // Build a synthetic candidate representing the merged skill
    let candidate = crate::skills::detection::SkillCandidate {
        community_id: 0,
        size: all_member_ids.len(),
        cohesion: 0.5,
        member_note_ids: all_member_ids,
        label: "merged".to_string(),
    };

    let evolution = vec![SkillEvolution::Merge {
        skill_ids: body.skill_ids.clone(),
        candidate,
    }];

    let notes_map = std::collections::HashMap::new();

    let result = execute_evolution(graph, &evolution, &notes_map, project_id)
        .await
        .map_err(AppError::Internal)?;

    // Invalidate hook cache
    skill_cache().invalidate_project(&project_id).await;

    let survivor_id = result.merged.first().map(|e| e.survivor_id);
    let absorbed_ids: Vec<Uuid> = result.merged.iter().map(|e| e.absorbed_id).collect();

    Ok(Json(serde_json::json!({
        "survivor_id": survivor_id,
        "absorbed_ids": absorbed_ids,
        "total_merged": result.merged.len(),
    })))
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

    // ================================================================
    // Async integration tests (mock backends)
    // ================================================================

    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_bearer_token, test_project};
    use axum::body::Body;
    use axum::http::{Request, StatusCode as HttpStatus};
    use std::sync::Arc;
    use tower::ServiceExt;

    fn auth_get(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    fn auth_post_json(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap()
    }

    fn auth_put_json(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("PUT")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap()
    }

    fn auth_delete(uri: &str) -> Request<Body> {
        Request::builder()
            .method("DELETE")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    async fn body_json(resp: axum::http::Response<Body>) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn test_app() -> axum::Router {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
        });
        create_router(state)
    }

    async fn test_app_with_project() -> (axum::Router, uuid::Uuid) {
        let app_state = mock_app_state();
        let project = test_project();
        let project_id = project.id;
        app_state.neo4j.create_project(&project).await.unwrap();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
        });
        (create_router(state), project_id)
    }

    #[tokio::test]
    async fn test_list_skills_empty() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_get(&format!("/api/skills?project_id={}", project_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["total"], 0);
        assert_eq!(json["items"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_create_skill() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_post_json(
                "/api/skills",
                serde_json::json!({
                    "project_id": project_id,
                    "name": "Test Skill",
                    "description": "A test skill",
                    "tags": ["test", "coverage"]
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::CREATED);
        let json = body_json(resp).await;
        assert_eq!(json["name"], "Test Skill");
        assert_eq!(json["description"], "A test skill");
        assert_eq!(json["status"], "emerging");
    }

    #[tokio::test]
    async fn test_create_skill_empty_name_fails() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_post_json(
                "/api/skills",
                serde_json::json!({
                    "project_id": project_id,
                    "name": "",
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_get_skill() {
        let app_state = mock_app_state();
        let project = test_project();
        app_state.neo4j.create_project(&project).await.unwrap();
        let skill = SkillNode::new(project.id, "My Skill");
        let skill_id = skill.id;
        app_state.neo4j.create_skill(&skill).await.unwrap();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
        });
        let app = create_router(state);

        let resp = app
            .oneshot(auth_get(&format!("/api/skills/{}", skill_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["name"], "My Skill");
    }

    #[tokio::test]
    async fn test_get_skill_not_found() {
        let app = test_app().await;
        let fake_id = uuid::Uuid::new_v4();
        let resp = app
            .oneshot(auth_get(&format!("/api/skills/{}", fake_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_delete_skill() {
        let app_state = mock_app_state();
        let project = test_project();
        app_state.neo4j.create_project(&project).await.unwrap();
        let skill = SkillNode::new(project.id, "To Delete");
        let skill_id = skill.id;
        app_state.neo4j.create_skill(&skill).await.unwrap();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
        });
        let app = create_router(state);

        let resp = app
            .oneshot(auth_delete(&format!("/api/skills/{}", skill_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_get_skill_members_empty() {
        let app_state = mock_app_state();
        let project = test_project();
        app_state.neo4j.create_project(&project).await.unwrap();
        let skill = SkillNode::new(project.id, "Skill With No Members");
        let skill_id = skill.id;
        app_state.neo4j.create_skill(&skill).await.unwrap();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
        });
        let app = create_router(state);

        let resp = app
            .oneshot(auth_get(&format!("/api/skills/{}/members", skill_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["notes"].as_array().unwrap().len(), 0);
        assert_eq!(json["decisions"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_update_skill() {
        let app_state = mock_app_state();
        let project = test_project();
        app_state.neo4j.create_project(&project).await.unwrap();
        let skill = SkillNode::new(project.id, "Original Name");
        let skill_id = skill.id;
        app_state.neo4j.create_skill(&skill).await.unwrap();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
        });
        let app = create_router(state);

        let resp = app
            .oneshot(auth_put_json(
                &format!("/api/skills/{}", skill_id),
                serde_json::json!({
                    "name": "Updated Name",
                    "description": "Updated desc"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["name"], "Updated Name");
    }

    #[tokio::test]
    async fn test_get_skill_health() {
        let app_state = mock_app_state();
        let project = test_project();
        app_state.neo4j.create_project(&project).await.unwrap();
        let skill = SkillNode::new(project.id, "Health Check");
        let skill_id = skill.id;
        app_state.neo4j.create_skill(&skill).await.unwrap();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
        });
        let app = create_router(state);

        let resp = app
            .oneshot(auth_get(&format!("/api/skills/{}/health", skill_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
    }
}
