//! Portable Skill Package format
//!
//! Defines the `SkillPackage` format for exporting and importing skills
//! across projects. The format is self-contained, versionned, and strips
//! all internal IDs (they are regenerated on import).
//!
//! # Format: `po-skill/v1`
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "metadata": { ... },
//!   "skill": { ... },
//!   "notes": [ ... ],
//!   "decisions": [ ... ]
//! }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::skills::models::{SkillTrigger, TriggerType};
use crate::skills::{REGEX_DFA_SIZE_LIMIT, REGEX_SIZE_LIMIT};

/// Current schema version for the SkillPackage format.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Format identifier for SkillPackage files.
pub const FORMAT_ID: &str = "po-skill/v1";

// ============================================================================
// SkillPackage — top-level portable format
// ============================================================================

/// A portable skill package for export/import across projects.
///
/// All internal IDs (project_id, note_id, etc.) are stripped — they are
/// regenerated on import. This makes packages project-independent and
/// safe to share.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillPackage {
    /// Format version for backward compatibility.
    pub schema_version: u32,
    /// Package metadata (export context, stats).
    pub metadata: PackageMetadata,
    /// The skill definition (name, triggers, template).
    pub skill: PortableSkill,
    /// Member notes (knowledge content).
    pub notes: Vec<PortableNote>,
    /// Member decisions (architectural choices).
    pub decisions: Vec<PortableDecision>,
}

// ============================================================================
// PackageMetadata
// ============================================================================

/// Metadata about the export context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageMetadata {
    /// Format identifier (always "po-skill/v1").
    pub format: String,
    /// When the package was exported.
    pub exported_at: DateTime<Utc>,
    /// Source project name (for provenance, optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_project: Option<String>,
    /// Export statistics.
    pub stats: PackageStats,
}

/// Statistics about the exported skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageStats {
    /// Number of notes in the package.
    pub note_count: usize,
    /// Number of decisions in the package.
    pub decision_count: usize,
    /// Number of trigger patterns.
    pub trigger_count: usize,
    /// Original activation count at time of export.
    pub activation_count: i64,
}

// ============================================================================
// PortableSkill — skill definition without internal IDs
// ============================================================================

/// Portable skill definition, stripped of internal IDs.
///
/// On import, a new SkillNode is created with fresh UUID, project_id,
/// and all counters reset to zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableSkill {
    /// Human-readable name.
    pub name: String,
    /// Description of the skill's domain.
    #[serde(default)]
    pub description: String,
    /// Trigger patterns for hook activation.
    #[serde(default)]
    pub trigger_patterns: Vec<SkillTrigger>,
    /// Pre-generated context template (Markdown).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_template: Option<String>,
    /// Tags for categorization.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Cohesion score at time of export (informational).
    #[serde(default)]
    pub cohesion: f64,
}

// ============================================================================
// PortableNote — note without internal IDs or runtime state
// ============================================================================

/// Portable note, stripped of internal IDs and runtime state.
///
/// Excludes: id, project_id, energy, staleness_score, last_activated,
/// supersedes, superseded_by, assertion_rule, last_assertion_result.
/// These are all regenerated or initialized on import.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableNote {
    /// Note type (guideline, gotcha, pattern, tip, etc.).
    pub note_type: String,
    /// Importance level (critical, high, medium, low).
    pub importance: String,
    /// The actual knowledge content.
    pub content: String,
    /// Tags for categorization.
    #[serde(default)]
    pub tags: Vec<String>,
}

// ============================================================================
// PortableDecision — decision without internal IDs
// ============================================================================

/// Portable decision, stripped of internal IDs.
///
/// Excludes: id, task_id, decided_by, decided_at, status, embedding.
/// The decision content (description, rationale, alternatives, chosen)
/// is the valuable knowledge to transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableDecision {
    /// What was decided.
    pub description: String,
    /// Why this choice was made.
    pub rationale: String,
    /// Alternatives that were considered.
    #[serde(default)]
    pub alternatives: Vec<String>,
    /// The chosen option.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_option: Option<String>,
}

// ============================================================================
// Validation
// ============================================================================

/// A validation error found in a SkillPackage.
#[derive(Debug, Clone, Serialize)]
pub struct PackageValidationError {
    /// Which field or section has the error.
    pub field: String,
    /// Human-readable error message.
    pub message: String,
}

impl std::fmt::Display for PackageValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

/// Validate a SkillPackage for import readiness.
///
/// Checks:
/// - `schema_version` is supported (currently: 1)
/// - Skill name is non-empty
/// - At least 1 note is present
/// - Regex triggers compile successfully
/// - FileGlob triggers are valid glob patterns
///
/// Returns `Ok(())` if valid, or a list of validation errors.
pub fn validate_package(package: &SkillPackage) -> Result<(), Vec<PackageValidationError>> {
    let mut errors = Vec::new();

    // Schema version
    if package.schema_version != CURRENT_SCHEMA_VERSION {
        errors.push(PackageValidationError {
            field: "schema_version".to_string(),
            message: format!(
                "Unsupported schema version {}. Expected {}.",
                package.schema_version, CURRENT_SCHEMA_VERSION
            ),
        });
    }

    // Skill name
    if package.skill.name.trim().is_empty() {
        errors.push(PackageValidationError {
            field: "skill.name".to_string(),
            message: "Skill name cannot be empty.".to_string(),
        });
    }

    // At least 1 note
    if package.notes.is_empty() {
        errors.push(PackageValidationError {
            field: "notes".to_string(),
            message: "Package must contain at least 1 note.".to_string(),
        });
    }

    // Validate trigger patterns
    for (i, trigger) in package.skill.trigger_patterns.iter().enumerate() {
        match trigger.pattern_type {
            TriggerType::Regex => {
                if let Err(e) = regex::RegexBuilder::new(&trigger.pattern_value)
                    .case_insensitive(true)
                    .size_limit(REGEX_SIZE_LIMIT)
                    .dfa_size_limit(REGEX_DFA_SIZE_LIMIT)
                    .build()
                {
                    errors.push(PackageValidationError {
                        field: format!("skill.trigger_patterns[{}]", i),
                        message: format!("Invalid regex '{}': {}", trigger.pattern_value, e),
                    });
                }
            }
            TriggerType::FileGlob => {
                if let Err(e) = glob::Pattern::new(&trigger.pattern_value) {
                    errors.push(PackageValidationError {
                        field: format!("skill.trigger_patterns[{}]", i),
                        message: format!("Invalid glob '{}': {}", trigger.pattern_value, e),
                    });
                }
            }
            TriggerType::Semantic => {
                // Semantic triggers are opaque embedding vectors — no validation needed
            }
        }
    }

    // Validate note content is non-empty
    for (i, note) in package.notes.iter().enumerate() {
        if note.content.trim().is_empty() {
            errors.push(PackageValidationError {
                field: format!("notes[{}].content", i),
                message: "Note content cannot be empty.".to_string(),
            });
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::models::SkillTrigger;

    fn make_valid_package() -> SkillPackage {
        SkillPackage {
            schema_version: CURRENT_SCHEMA_VERSION,
            metadata: PackageMetadata {
                format: FORMAT_ID.to_string(),
                exported_at: Utc::now(),
                source_project: Some("test-project".to_string()),
                stats: PackageStats {
                    note_count: 2,
                    decision_count: 1,
                    trigger_count: 1,
                    activation_count: 5,
                },
            },
            skill: PortableSkill {
                name: "Neo4j Performance".to_string(),
                description: "Query optimization knowledge".to_string(),
                trigger_patterns: vec![SkillTrigger::regex("neo4j|cypher", 0.7)],
                context_template: None,
                tags: vec!["neo4j".to_string(), "performance".to_string()],
                cohesion: 0.75,
            },
            notes: vec![
                PortableNote {
                    note_type: "guideline".to_string(),
                    importance: "high".to_string(),
                    content: "Always use UNWIND for batch operations".to_string(),
                    tags: vec!["neo4j".to_string()],
                },
                PortableNote {
                    note_type: "gotcha".to_string(),
                    importance: "critical".to_string(),
                    content: "Connection pool leak if not closed".to_string(),
                    tags: vec![],
                },
            ],
            decisions: vec![PortableDecision {
                description: "Use Neo4j 5.x driver".to_string(),
                rationale: "Better async support".to_string(),
                alternatives: vec!["Neo4j 4.x".to_string(), "Custom driver".to_string()],
                chosen_option: Some("neo4j-rust-driver 0.8".to_string()),
            }],
        }
    }

    #[test]
    fn test_valid_package_passes_validation() {
        let package = make_valid_package();
        assert!(validate_package(&package).is_ok());
    }

    #[test]
    fn test_invalid_schema_version() {
        let mut package = make_valid_package();
        package.schema_version = 99;
        let errors = validate_package(&package).unwrap_err();
        assert!(errors.iter().any(|e| e.field == "schema_version"));
    }

    #[test]
    fn test_empty_skill_name() {
        let mut package = make_valid_package();
        package.skill.name = "  ".to_string();
        let errors = validate_package(&package).unwrap_err();
        assert!(errors.iter().any(|e| e.field == "skill.name"));
    }

    #[test]
    fn test_no_notes() {
        let mut package = make_valid_package();
        package.notes.clear();
        let errors = validate_package(&package).unwrap_err();
        assert!(errors.iter().any(|e| e.field == "notes"));
    }

    #[test]
    fn test_invalid_regex_trigger() {
        let mut package = make_valid_package();
        package.skill.trigger_patterns = vec![SkillTrigger::regex("[invalid(", 0.7)];
        let errors = validate_package(&package).unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.field.contains("trigger_patterns")));
    }

    #[test]
    fn test_invalid_glob_trigger() {
        let mut package = make_valid_package();
        package.skill.trigger_patterns = vec![SkillTrigger::file_glob("[invalid", 0.8)];
        let errors = validate_package(&package).unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.field.contains("trigger_patterns")));
    }

    #[test]
    fn test_empty_note_content() {
        let mut package = make_valid_package();
        package.notes[0].content = "  ".to_string();
        let errors = validate_package(&package).unwrap_err();
        assert!(errors.iter().any(|e| e.field.contains("notes[0].content")));
    }

    #[test]
    fn test_serde_roundtrip() {
        let package = make_valid_package();
        let json = serde_json::to_string_pretty(&package).unwrap();
        let deserialized: SkillPackage = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.schema_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(deserialized.skill.name, "Neo4j Performance");
        assert_eq!(deserialized.notes.len(), 2);
        assert_eq!(deserialized.decisions.len(), 1);
        assert_eq!(deserialized.metadata.format, FORMAT_ID);
    }

    #[test]
    fn test_multiple_validation_errors() {
        let mut package = make_valid_package();
        package.schema_version = 99;
        package.skill.name = "".to_string();
        package.notes.clear();
        let errors = validate_package(&package).unwrap_err();
        assert!(errors.len() >= 3, "Expected 3+ errors, got {}", errors.len());
    }

    #[test]
    fn test_semantic_trigger_always_valid() {
        let mut package = make_valid_package();
        package.skill.trigger_patterns =
            vec![SkillTrigger::semantic("[0.1, 0.2, 0.3]", 0.7)];
        assert!(validate_package(&package).is_ok());
    }
}
