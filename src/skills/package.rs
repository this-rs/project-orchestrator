//! Portable Skill Package format
//!
//! Defines the `SkillPackage` format for exporting and importing skills
//! across projects. The format is self-contained, versioned, and strips
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
/// v1: notes + decisions
/// v2: + protocols + execution_history + source metadata
pub const CURRENT_SCHEMA_VERSION: u32 = 3;

/// Format identifier for SkillPackage files.
pub const FORMAT_ID: &str = "po-skill/v3";

/// Minimum supported schema version for backward compatibility.
pub const MIN_SUPPORTED_SCHEMA_VERSION: u32 = 1;

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

    // --- v2 fields (all optional for backward compatibility with v1) ---
    /// Protocols linked to this skill (FSM definitions).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protocols: Vec<PortableProtocol>,
    /// Execution history aggregated from protocol runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_history: Option<ExecutionHistory>,
    /// Source metadata for provenance tracking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceMetadata>,
    /// Episodic memories captured during skill usage (v3).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub episodes: Vec<crate::episodes::PortableEpisode>,

    // --- v3 fields (all optional for backward compatibility with v1/v2) ---
    /// Distilled episodes with trust proofs and anonymization metadata.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub distilled_episodes: Vec<crate::episodes::distill_models::DistillationEnvelope>,
    /// Aggregate trust score for the package (0.0 - 1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub package_trust: Option<f64>,
    /// Privacy report summarizing all anonymization applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy_report: Option<crate::episodes::distill_models::AnonymizationReport>,
    /// Privacy mode used during distillation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy_mode: Option<crate::episodes::distill_models::PrivacyMode>,
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
// v2 — Portable Protocol (FSM definition)
// ============================================================================

/// A portable protocol definition (states + transitions), stripped of internal IDs.
///
/// States and transitions reference each other by name (not UUID) to ensure
/// portability. On import, fresh UUIDs are generated and name-based references
/// are resolved to the new IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableProtocol {
    /// Protocol name (e.g., "wave-execution").
    pub name: String,
    /// Description of the protocol's purpose.
    #[serde(default)]
    pub description: String,
    /// Classification: "system" or "business".
    #[serde(default = "default_protocol_category")]
    pub category: String,
    /// Multi-dimensional relevance vector for context-aware routing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relevance_vector: Option<PortableRelevanceVector>,
    /// States in this protocol.
    pub states: Vec<PortableState>,
    /// Transitions between states (referencing states by name).
    pub transitions: Vec<PortableTransition>,
}

fn default_protocol_category() -> String {
    "business".to_string()
}

/// A portable FSM state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableState {
    /// Human-readable name (used as key for transition references).
    pub name: String,
    /// Description of this state's purpose.
    #[serde(default)]
    pub description: String,
    /// Optional action description (what should happen in this state).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// Role: "start", "intermediate", or "terminal".
    #[serde(default = "default_state_type")]
    pub state_type: String,
}

fn default_state_type() -> String {
    "intermediate".to_string()
}

/// A portable FSM transition (references states by name, not UUID).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableTransition {
    /// Source state name.
    pub from_state: String,
    /// Target state name.
    pub to_state: String,
    /// Trigger event name.
    pub trigger: String,
    /// Optional guard condition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard: Option<String>,
}

/// Portable relevance vector (5 dimensions, 0.0–1.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableRelevanceVector {
    pub phase: f64,
    pub structure: f64,
    pub domain: f64,
    pub resource: f64,
    pub lifecycle: f64,
}

// ============================================================================
// v2 — Execution History
// ============================================================================

/// Aggregated execution history from protocol runs at time of export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionHistory {
    /// Total number of activations.
    pub activation_count: i64,
    /// Success rate (0.0–1.0). Computed as completed_runs / total_runs.
    pub success_rate: f64,
    /// Average relevance score across activations.
    pub avg_score: f64,
    /// Number of distinct projects that used this skill.
    #[serde(default)]
    pub source_projects_count: usize,
}

// ============================================================================
// v2 — Source Metadata
// ============================================================================

/// Provenance information about where this package was exported from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMetadata {
    /// Source project name.
    pub project_name: String,
    /// Git remote URL (for project identity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_remote: Option<String>,
    /// PO instance identifier (for federation routing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
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

    // Schema version (accept v1 and v2)
    if package.schema_version < MIN_SUPPORTED_SCHEMA_VERSION
        || package.schema_version > CURRENT_SCHEMA_VERSION
    {
        errors.push(PackageValidationError {
            field: "schema_version".to_string(),
            message: format!(
                "Unsupported schema version {}. Supported: {}-{}.",
                package.schema_version, MIN_SUPPORTED_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION
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
            TriggerType::McpAction => {
                // McpAction patterns: "mega_tool" or "mega_tool:action"
                let trimmed = trigger.pattern_value.trim();
                if trimmed.is_empty() {
                    errors.push(PackageValidationError {
                        field: format!("skill.trigger_patterns[{}]", i),
                        message: "McpAction pattern must not be empty".to_string(),
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

    // Validate v2 protocols (if present)
    for (i, proto) in package.protocols.iter().enumerate() {
        if proto.name.trim().is_empty() {
            errors.push(PackageValidationError {
                field: format!("protocols[{}].name", i),
                message: "Protocol name cannot be empty.".to_string(),
            });
        }
        // Must have at least one start state
        let start_count = proto
            .states
            .iter()
            .filter(|s| s.state_type == "start")
            .count();
        if start_count == 0 && !proto.states.is_empty() {
            errors.push(PackageValidationError {
                field: format!("protocols[{}].states", i),
                message: "Protocol must have at least one 'start' state.".to_string(),
            });
        }
        // Validate transitions reference existing state names
        let state_names: std::collections::HashSet<&str> =
            proto.states.iter().map(|s| s.name.as_str()).collect();
        for (j, trans) in proto.transitions.iter().enumerate() {
            if !state_names.contains(trans.from_state.as_str()) {
                errors.push(PackageValidationError {
                    field: format!("protocols[{}].transitions[{}].from_state", i, j),
                    message: format!(
                        "Transition references unknown state '{}'.",
                        trans.from_state
                    ),
                });
            }
            if !state_names.contains(trans.to_state.as_str()) {
                errors.push(PackageValidationError {
                    field: format!("protocols[{}].transitions[{}].to_state", i, j),
                    message: format!("Transition references unknown state '{}'.", trans.to_state),
                });
            }
        }
    }

    // Validate metadata stats consistency
    if package.metadata.stats.note_count != package.notes.len() {
        errors.push(PackageValidationError {
            field: "metadata.stats.note_count".to_string(),
            message: format!(
                "Stats note_count ({}) does not match actual notes count ({}).",
                package.metadata.stats.note_count,
                package.notes.len()
            ),
        });
    }
    if package.metadata.stats.decision_count != package.decisions.len() {
        errors.push(PackageValidationError {
            field: "metadata.stats.decision_count".to_string(),
            message: format!(
                "Stats decision_count ({}) does not match actual decisions count ({}).",
                package.metadata.stats.decision_count,
                package.decisions.len()
            ),
        });
    }
    if package.metadata.stats.trigger_count != package.skill.trigger_patterns.len() {
        errors.push(PackageValidationError {
            field: "metadata.stats.trigger_count".to_string(),
            message: format!(
                "Stats trigger_count ({}) does not match actual trigger_patterns count ({}).",
                package.metadata.stats.trigger_count,
                package.skill.trigger_patterns.len()
            ),
        });
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
            protocols: vec![],
            execution_history: None,
            source: None,
            episodes: Vec::new(),
            distilled_episodes: Vec::new(),
            package_trust: None,
            privacy_report: None,
            privacy_mode: None,
        }
    }

    /// Create a valid v1 package (backward compatibility test)
    fn make_v1_package() -> SkillPackage {
        let mut pkg = make_valid_package();
        pkg.schema_version = 1;
        pkg
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
        assert!(errors.iter().any(|e| e.field.contains("trigger_patterns")));
    }

    #[test]
    fn test_invalid_glob_trigger() {
        let mut package = make_valid_package();
        package.skill.trigger_patterns = vec![SkillTrigger::file_glob("[invalid", 0.8)];
        let errors = validate_package(&package).unwrap_err();
        assert!(errors.iter().any(|e| e.field.contains("trigger_patterns")));
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
        assert!(
            errors.len() >= 3,
            "Expected 3+ errors, got {}",
            errors.len()
        );
    }

    #[test]
    fn test_semantic_trigger_always_valid() {
        let mut package = make_valid_package();
        package.skill.trigger_patterns = vec![SkillTrigger::semantic("[0.1, 0.2, 0.3]", 0.7)];
        assert!(validate_package(&package).is_ok());
    }

    #[test]
    fn test_v1_package_validates() {
        let package = make_v1_package();
        assert!(validate_package(&package).is_ok());
    }

    #[test]
    fn test_v2_with_protocols_validates() {
        let mut package = make_valid_package();
        package.protocols = vec![PortableProtocol {
            name: "test-protocol".to_string(),
            description: "A test protocol".to_string(),
            category: "business".to_string(),
            relevance_vector: Some(PortableRelevanceVector {
                phase: 0.5,
                structure: 0.8,
                domain: 0.5,
                resource: 0.7,
                lifecycle: 0.3,
            }),
            states: vec![
                PortableState {
                    name: "START".to_string(),
                    description: "Entry".to_string(),
                    action: None,
                    state_type: "start".to_string(),
                },
                PortableState {
                    name: "WORK".to_string(),
                    description: "Do work".to_string(),
                    action: Some("execute_task".to_string()),
                    state_type: "intermediate".to_string(),
                },
                PortableState {
                    name: "DONE".to_string(),
                    description: "Complete".to_string(),
                    action: None,
                    state_type: "terminal".to_string(),
                },
            ],
            transitions: vec![
                PortableTransition {
                    from_state: "START".to_string(),
                    to_state: "WORK".to_string(),
                    trigger: "begin".to_string(),
                    guard: None,
                },
                PortableTransition {
                    from_state: "WORK".to_string(),
                    to_state: "DONE".to_string(),
                    trigger: "complete".to_string(),
                    guard: Some("all_steps_done".to_string()),
                },
            ],
        }];
        package.execution_history = Some(ExecutionHistory {
            activation_count: 42,
            success_rate: 0.85,
            avg_score: 0.72,
            source_projects_count: 3,
        });
        package.source = Some(SourceMetadata {
            project_name: "test-project".to_string(),
            git_remote: Some("git@github.com:org/repo.git".to_string()),
            instance_id: None,
        });
        assert!(validate_package(&package).is_ok());
    }

    #[test]
    fn test_v2_protocol_bad_transition_ref() {
        let mut package = make_valid_package();
        package.protocols = vec![PortableProtocol {
            name: "bad-proto".to_string(),
            description: String::new(),
            category: "business".to_string(),
            relevance_vector: None,
            states: vec![PortableState {
                name: "START".to_string(),
                description: String::new(),
                action: None,
                state_type: "start".to_string(),
            }],
            transitions: vec![PortableTransition {
                from_state: "START".to_string(),
                to_state: "NONEXISTENT".to_string(),
                trigger: "go".to_string(),
                guard: None,
            }],
        }];
        let errors = validate_package(&package).unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("NONEXISTENT")));
    }

    #[test]
    fn test_v2_serde_roundtrip() {
        let mut package = make_valid_package();
        package.protocols = vec![PortableProtocol {
            name: "roundtrip-proto".to_string(),
            description: "Test".to_string(),
            category: "system".to_string(),
            relevance_vector: Some(PortableRelevanceVector {
                phase: 0.1,
                structure: 0.2,
                domain: 0.3,
                resource: 0.4,
                lifecycle: 0.5,
            }),
            states: vec![
                PortableState {
                    name: "S".to_string(),
                    description: String::new(),
                    action: None,
                    state_type: "start".to_string(),
                },
                PortableState {
                    name: "E".to_string(),
                    description: String::new(),
                    action: None,
                    state_type: "terminal".to_string(),
                },
            ],
            transitions: vec![PortableTransition {
                from_state: "S".to_string(),
                to_state: "E".to_string(),
                trigger: "done".to_string(),
                guard: None,
            }],
        }];
        package.execution_history = Some(ExecutionHistory {
            activation_count: 10,
            success_rate: 0.9,
            avg_score: 0.8,
            source_projects_count: 2,
        });

        let json = serde_json::to_string_pretty(&package).unwrap();
        let deserialized: SkillPackage = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.protocols.len(), 1);
        assert_eq!(deserialized.protocols[0].name, "roundtrip-proto");
        assert_eq!(deserialized.protocols[0].states.len(), 2);
        assert_eq!(deserialized.protocols[0].transitions.len(), 1);
        assert_eq!(
            deserialized
                .execution_history
                .as_ref()
                .unwrap()
                .activation_count,
            10
        );
    }

    #[test]
    fn test_v1_json_deserializes_to_v2_struct() {
        // Simulate a v1 JSON (no protocols/execution_history/source fields)
        let v1_json = serde_json::json!({
            "schema_version": 1,
            "metadata": {
                "format": "po-skill/v1",
                "exported_at": "2026-01-01T00:00:00Z",
                "stats": {
                    "note_count": 1,
                    "decision_count": 0,
                    "trigger_count": 1,
                    "activation_count": 5
                }
            },
            "skill": {
                "name": "V1 Skill",
                "description": "Legacy",
                "trigger_patterns": [{"pattern_type": "regex", "pattern_value": "test", "confidence_threshold": 0.7}],
                "tags": ["v1"],
                "cohesion": 0.5
            },
            "notes": [{
                "note_type": "guideline",
                "importance": "medium",
                "content": "V1 note",
                "tags": []
            }],
            "decisions": []
        });

        let package: SkillPackage = serde_json::from_value(v1_json).unwrap();
        assert_eq!(package.schema_version, 1);
        assert!(package.protocols.is_empty());
        assert!(package.execution_history.is_none());
        assert!(package.source.is_none());
        assert!(validate_package(&package).is_ok());
    }

    #[test]
    fn test_v2_json_deserializes_to_v3_struct() {
        let v2_json = serde_json::json!({
            "schema_version": 2,
            "metadata": {
                "format": "po-skill/v2",
                "exported_at": "2026-02-01T00:00:00Z",
                "stats": {
                    "note_count": 1,
                    "decision_count": 0,
                    "trigger_count": 1,
                    "activation_count": 10
                }
            },
            "skill": {
                "name": "V2 Skill",
                "description": "V2 format",
                "trigger_patterns": [{"pattern_type": "regex", "pattern_value": "test", "confidence_threshold": 0.7}],
                "tags": ["v2"],
                "cohesion": 0.6
            },
            "notes": [{
                "note_type": "guideline",
                "importance": "high",
                "content": "V2 note content",
                "tags": ["neo4j"]
            }],
            "decisions": [],
            "protocols": [],
            "execution_history": {
                "activation_count": 10,
                "success_rate": 0.9,
                "avg_score": 0.75,
                "source_projects_count": 2
            },
            "source": {
                "project_name": "test-project",
                "git_remote": "git@github.com:org/repo.git"
            }
        });

        let package: SkillPackage = serde_json::from_value(v2_json).unwrap();
        assert_eq!(package.schema_version, 2);
        assert_eq!(package.skill.name, "V2 Skill");
        assert!(package.distilled_episodes.is_empty());
        assert!(package.package_trust.is_none());
        assert!(package.privacy_report.is_none());
        assert!(package.privacy_mode.is_none());
        assert!(package.execution_history.is_some());
        assert!(package.source.is_some());
        assert!(validate_package(&package).is_ok());
    }

    #[test]
    fn test_v3_serde_roundtrip() {
        use crate::episodes::distill_models::*;
        use std::collections::HashMap;

        let mut package = make_valid_package();
        package.distilled_episodes = vec![DistillationEnvelope {
            lesson: DistilledLesson {
                abstract_pattern: "Always index relations".to_string(),
                domain_tags: vec!["neo4j".to_string()],
                portability_layer: PortabilityLayer::Domain,
                confidence: 0.85,
            },
            anonymized_content: "Use UNWIND for batch operations".to_string(),
            meta: DistillationMeta {
                pipeline_version: "1.0".to_string(),
                sensitivity_level: SensitivityLevel::Public,
                quality_score: 0.85,
                content_hash: "abc123def456".to_string(),
            },
            trust_proof: TrustProof {
                source_did: "did:key:z6MkTest".to_string(),
                signature_hex: "deadbeef".to_string(),
                trust_scores: HashMap::from([("confidence".to_string(), 0.85)]),
            },
            anonymization_report: None,
        }];
        package.package_trust = Some(0.85);
        package.privacy_mode = Some(PrivacyMode::Standard);

        let json = serde_json::to_string_pretty(&package).unwrap();
        let restored: SkillPackage = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.schema_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(restored.distilled_episodes.len(), 1);
        assert_eq!(
            restored.distilled_episodes[0].lesson.abstract_pattern,
            "Always index relations"
        );
        assert_eq!(restored.package_trust, Some(0.85));
        assert_eq!(restored.privacy_mode, Some(PrivacyMode::Standard));
    }
}
