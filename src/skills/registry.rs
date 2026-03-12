//! Skill Registry — Local storage and discovery of published SkillPackages
//!
//! The registry allows skills to be published (made discoverable) and searched.
//! Published skills are stored as `PublishedSkill` nodes in Neo4j with their
//! full SkillPackage JSON, trust score, and searchable metadata.
//!
//! # Architecture
//!
//! - **Local registry**: stored in Neo4j, same instance
//! - **Remote registry**: another PO instance accessed via HTTP (see `registry_handlers`)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::skills::package::SkillPackage;
use crate::skills::trust::compute_trust_score;

// ============================================================================
// Models
// ============================================================================

/// A published skill in the registry.
///
/// Wraps a [`SkillPackage`] with registry metadata (trust score, publish info).
/// Stored as a Neo4j node with label `PublishedSkill`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishedSkill {
    /// Unique registry entry ID
    pub id: Uuid,
    /// Name of the skill (copied from package for indexing)
    pub name: String,
    /// Description (copied from package for search)
    pub description: String,
    /// Tags (copied from package for filtering)
    pub tags: Vec<String>,
    /// The full portable package
    pub package: SkillPackage,
    /// Trust score computed at publish time
    pub trust_score: f64,
    /// Detailed trust breakdown
    pub trust_components: crate::skills::trust::TrustComponents,
    /// Trust level category
    pub trust_level: crate::skills::trust::TrustLevel,
    /// ID of the source project that published this skill
    pub source_project_id: Uuid,
    /// Name of the source project
    pub source_project_name: String,
    /// Who published this skill
    pub published_by: String,
    /// When this entry was published
    pub published_at: DateTime<Utc>,
    /// Number of times this skill has been imported from the registry
    #[serde(default)]
    pub import_count: i64,
}

/// Search parameters for the registry.
#[derive(Debug, Clone, Deserialize)]
pub struct RegistrySearchParams {
    /// Full-text search query
    pub query: Option<String>,
    /// Minimum trust score filter (0.0–1.0)
    pub min_trust: Option<f64>,
    /// Filter by tags (AND logic)
    pub tags: Option<Vec<String>>,
    /// Maximum number of results
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Skip N results
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize {
    20
}

/// Result of a registry search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrySearchResult {
    /// Matching published skills
    pub items: Vec<PublishedSkillSummary>,
    /// Total count (for pagination)
    pub total: usize,
}

/// Summary of a published skill (without the full package, for search results).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishedSkillSummary {
    /// Registry entry ID
    pub id: Uuid,
    /// Skill name
    pub name: String,
    /// Description
    pub description: String,
    /// Tags
    pub tags: Vec<String>,
    /// Trust score
    pub trust_score: f64,
    /// Trust level
    pub trust_level: crate::skills::trust::TrustLevel,
    /// Source project name
    pub source_project_name: String,
    /// When published
    pub published_at: DateTime<Utc>,
    /// Number of notes in the package
    pub note_count: usize,
    /// Number of protocols in the package
    pub protocol_count: usize,
    /// Number of times imported
    pub import_count: i64,
    /// Whether this entry comes from a remote registry
    #[serde(default)]
    pub is_remote: bool,
    /// Source instance URL (for remote entries)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_url: Option<String>,
}

impl From<&PublishedSkill> for PublishedSkillSummary {
    fn from(ps: &PublishedSkill) -> Self {
        Self {
            id: ps.id,
            name: ps.name.clone(),
            description: ps.description.clone(),
            tags: ps.tags.clone(),
            trust_score: ps.trust_score,
            trust_level: ps.trust_level,
            source_project_name: ps.source_project_name.clone(),
            published_at: ps.published_at,
            note_count: ps.package.notes.len(),
            protocol_count: ps.package.protocols.len(),
            import_count: ps.import_count,
            is_remote: false,
            remote_url: None,
        }
    }
}

// ============================================================================
// Publish logic
// ============================================================================

/// Request to publish a skill to the registry.
#[derive(Debug, Clone, Deserialize)]
pub struct PublishRequest {
    /// ID of the skill to publish
    pub skill_id: Uuid,
    /// Project ID (to verify ownership)
    pub project_id: Uuid,
    /// Optional override for source project name
    pub source_project_name: Option<String>,
}

/// Create a `PublishedSkill` from a skill node and its exported package.
pub fn build_published_skill(
    skill: &crate::skills::models::SkillNode,
    package: SkillPackage,
    project_name: String,
    execution_history: Option<&crate::skills::package::ExecutionHistory>,
) -> PublishedSkill {
    let trust = compute_trust_score(skill, execution_history);

    PublishedSkill {
        id: Uuid::new_v4(),
        name: skill.name.clone(),
        description: skill.description.clone(),
        tags: skill.tags.clone(),
        trust_score: trust.score,
        trust_components: trust.components,
        trust_level: trust.level,
        source_project_id: skill.project_id,
        source_project_name: project_name,
        published_by: "agent".to_string(),
        published_at: Utc::now(),
        import_count: 0,
        package,
    }
}

// ============================================================================
// Remote Registry Client
// ============================================================================

/// Response from a remote registry search endpoint.
///
/// Mirrors the `PaginatedResponse<PublishedSkillSummary>` format returned by
/// `GET /api/registry/search` on the remote instance.
#[derive(Debug, Clone, Deserialize)]
pub struct RemoteSearchResponse {
    pub items: Vec<PublishedSkillSummary>,
    pub total: usize,
}

/// Fetch published skills from a remote PO instance.
///
/// Calls `GET {remote_url}/api/registry/search` with the same query parameters
/// as the local search. Remote entries are marked with `is_remote=true` and
/// the `remote_url` field is populated.
///
/// Returns `Ok((items, total))` on success, `Err` on network/parse failure.
/// Callers should treat errors as non-fatal (remote unavailable → local only).
pub async fn search_remote_registry(
    remote_url: &str,
    query: Option<&str>,
    min_trust: Option<f64>,
    tags: Option<&[String]>,
    limit: usize,
    offset: usize,
) -> anyhow::Result<(Vec<PublishedSkillSummary>, usize)> {
    let base = remote_url.trim_end_matches('/');
    let mut url = format!(
        "{}/api/registry/search?limit={}&offset={}",
        base, limit, offset
    );

    if let Some(q) = query {
        url.push_str(&format!("&query={}", urlencoding::encode(q)));
    }
    if let Some(mt) = min_trust {
        url.push_str(&format!("&min_trust={}", mt));
    }
    if let Some(t) = tags {
        let tags_str = t.join(",");
        url.push_str(&format!("&tags={}", urlencoding::encode(&tags_str)));
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let resp = client.get(&url).send().await?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "Remote registry returned status {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        );
    }

    let mut remote_resp: RemoteSearchResponse = resp.json().await?;

    // Mark all remote entries
    for item in &mut remote_resp.items {
        item.is_remote = true;
        item.remote_url = Some(base.to_string());
    }

    Ok((remote_resp.items, remote_resp.total))
}

/// Fetch a specific published skill from a remote PO instance.
///
/// Calls `GET {remote_url}/api/registry/{id}` to retrieve the full
/// PublishedSkill (including the SkillPackage for import).
pub async fn get_remote_published_skill(
    remote_url: &str,
    id: Uuid,
) -> anyhow::Result<PublishedSkill> {
    let base = remote_url.trim_end_matches('/');
    let url = format!("{}/api/registry/{}", base, id);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let resp = client.get(&url).send().await?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "Remote registry returned status {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        );
    }

    let published: PublishedSkill = resp.json().await?;
    Ok(published)
}

/// Merge local and remote search results.
///
/// Returns combined results sorted by trust_score DESC, with remote entries
/// interleaved among local ones. Deduplicates by name (local takes precedence).
pub fn merge_search_results(
    local: Vec<PublishedSkillSummary>,
    remote: Vec<PublishedSkillSummary>,
) -> Vec<PublishedSkillSummary> {
    // Collect local names for dedup
    let local_names: std::collections::HashSet<String> =
        local.iter().map(|s| s.name.clone()).collect();

    // Filter out remote duplicates (same name as local)
    let remote_unique: Vec<PublishedSkillSummary> = remote
        .into_iter()
        .filter(|s| !local_names.contains(&s.name))
        .collect();

    // Merge and sort by trust_score DESC
    let mut merged = local;
    merged.extend(remote_unique);
    merged.sort_by(|a, b| {
        b.trust_score
            .partial_cmp(&a.trust_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.published_at.cmp(&a.published_at))
    });

    merged
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::models::SkillNode;
    use crate::skills::package::*;

    fn make_test_skill() -> SkillNode {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Test Skill");
        skill.energy = 0.8;
        skill.cohesion = 0.75;
        skill.activation_count = 50;
        skill.hit_rate = 0.9;
        skill.description = "A test skill".to_string();
        skill.tags = vec!["test".to_string()];
        skill
    }

    fn make_test_package() -> SkillPackage {
        SkillPackage {
            schema_version: CURRENT_SCHEMA_VERSION,
            metadata: PackageMetadata {
                format: FORMAT_ID.to_string(),
                exported_at: Utc::now(),
                source_project: Some("test-project".to_string()),
                stats: PackageStats {
                    note_count: 1,
                    decision_count: 0,
                    trigger_count: 1,
                    activation_count: 50,
                },
            },
            skill: PortableSkill {
                name: "Test Skill".to_string(),
                description: "A test skill".to_string(),
                trigger_patterns: vec![],
                context_template: None,
                tags: vec!["test".to_string()],
                cohesion: 0.75,
            },
            notes: vec![PortableNote {
                note_type: "guideline".to_string(),
                importance: "high".to_string(),
                content: "Test note".to_string(),
                tags: vec![],
            }],
            decisions: vec![],
            protocols: vec![],
            execution_history: None,
            source: None,
            episodes: Vec::new(),
        }
    }

    #[test]
    fn test_build_published_skill() {
        let skill = make_test_skill();
        let package = make_test_package();
        let published = build_published_skill(&skill, package, "My Project".to_string(), None);

        assert_eq!(published.name, "Test Skill");
        assert_eq!(published.source_project_name, "My Project");
        assert!(published.trust_score > 0.5); // good metrics → decent trust
        assert_eq!(published.import_count, 0);
    }

    #[test]
    fn test_published_skill_summary() {
        let skill = make_test_skill();
        let package = make_test_package();
        let published = build_published_skill(&skill, package, "My Project".to_string(), None);
        let summary = PublishedSkillSummary::from(&published);

        assert_eq!(summary.name, "Test Skill");
        assert_eq!(summary.note_count, 1);
        assert_eq!(summary.protocol_count, 0);
        assert!(!summary.is_remote);
        assert!(summary.remote_url.is_none());
    }

    #[test]
    fn test_trust_score_stored_on_publish() {
        let mut skill = make_test_skill();
        skill.energy = 0.0;
        skill.cohesion = 0.0;
        skill.activation_count = 0;
        skill.hit_rate = 0.0;

        let package = make_test_package();
        let published = build_published_skill(&skill, package, "Test".to_string(), None);

        // Low metrics → low trust
        assert!(published.trust_score < 0.3);
    }

    // =====================================================================
    // merge_search_results tests
    // =====================================================================

    fn make_summary(name: &str, trust: f64, is_remote: bool) -> PublishedSkillSummary {
        PublishedSkillSummary {
            id: Uuid::new_v4(),
            name: name.to_string(),
            description: format!("Desc for {}", name),
            tags: vec![],
            trust_score: trust,
            trust_level: if trust >= 0.8 {
                crate::skills::trust::TrustLevel::High
            } else {
                crate::skills::trust::TrustLevel::Medium
            },
            source_project_name: "proj".to_string(),
            published_at: Utc::now(),
            note_count: 1,
            protocol_count: 0,
            import_count: 0,
            is_remote,
            remote_url: if is_remote {
                Some("https://remote.example.com".to_string())
            } else {
                None
            },
        }
    }

    #[test]
    fn test_merge_deduplicates_by_name_local_wins() {
        let local = vec![make_summary("Skill A", 0.7, false)];
        let remote = vec![
            make_summary("Skill A", 0.9, true), // same name → should be filtered
            make_summary("Skill B", 0.8, true), // unique → should be kept
        ];

        let merged = merge_search_results(local, remote);
        assert_eq!(merged.len(), 2);

        // Skill B (0.8) should be first, Skill A (0.7) second
        assert_eq!(merged[0].name, "Skill B");
        assert!(merged[0].is_remote);
        assert_eq!(merged[1].name, "Skill A");
        assert!(!merged[1].is_remote); // local version kept
    }

    #[test]
    fn test_merge_sorted_by_trust_desc() {
        let local = vec![make_summary("Low Trust", 0.3, false)];
        let remote = vec![
            make_summary("High Trust", 0.95, true),
            make_summary("Mid Trust", 0.6, true),
        ];

        let merged = merge_search_results(local, remote);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].name, "High Trust");
        assert_eq!(merged[1].name, "Mid Trust");
        assert_eq!(merged[2].name, "Low Trust");
    }

    #[test]
    fn test_merge_empty_remote() {
        let local = vec![make_summary("A", 0.8, false), make_summary("B", 0.6, false)];
        let remote = vec![];

        let merged = merge_search_results(local, remote);
        assert_eq!(merged.len(), 2);
        assert!(!merged[0].is_remote);
    }

    #[test]
    fn test_merge_empty_local() {
        let local = vec![];
        let remote = vec![make_summary("Remote Only", 0.7, true)];

        let merged = merge_search_results(local, remote);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].name, "Remote Only");
        assert!(merged[0].is_remote);
    }
}
