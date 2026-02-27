//! Skill data model and DTOs
//!
//! Defines the core types for the Neural Skills system:
//! - [`SkillNode`]: A knowledge cluster detected from the SYNAPSE graph
//! - [`SkillStatus`]: Lifecycle status (Emerging → Active → Dormant → Archived)
//! - [`SkillTrigger`]: Pattern matching rules for hook activation
//! - [`TriggerType`]: Type of trigger (Regex, FileGlob, Semantic)
//! - [`ActivatedSkillContext`]: Response payload for hook activation

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

use crate::neo4j::models::DecisionNode;
use crate::neurons::activation::ActivatedNote;

// ============================================================================
// Enums
// ============================================================================

/// Lifecycle status of a Skill.
///
/// Skills follow an autonomous lifecycle driven by usage patterns:
/// - Emerging → Active: after 2+ successful hook activations
/// - Active → Dormant: after 30+ days of inactivity or energy < 0.1
/// - Dormant → Archived: after 90+ days without reactivation
/// - Imported: created from an external SkillPackage, in probation
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SkillStatus {
    /// Newly detected by Louvain, needs validation through usage
    #[default]
    Emerging,
    /// Proven useful, actively matched by hooks
    Active,
    /// Inactive for 30+ days or low energy, excluded from hook matching
    Dormant,
    /// Dead skill (90+ days inactive), data preserved but not matched
    Archived,
    /// Created from an imported SkillPackage, in probation period
    Imported,
}

impl fmt::Display for SkillStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Emerging => write!(f, "emerging"),
            Self::Active => write!(f, "active"),
            Self::Dormant => write!(f, "dormant"),
            Self::Archived => write!(f, "archived"),
            Self::Imported => write!(f, "imported"),
        }
    }
}

impl FromStr for SkillStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "emerging" => Ok(Self::Emerging),
            "active" => Ok(Self::Active),
            "dormant" => Ok(Self::Dormant),
            "archived" => Ok(Self::Archived),
            "imported" => Ok(Self::Imported),
            _ => Err(format!("Unknown skill status: {}", s)),
        }
    }
}

impl SkillStatus {
    /// Returns true if transitioning from `self` to `to` is a valid lifecycle transition.
    ///
    /// Valid transitions:
    /// - Emerging → Active, Dormant, Archived
    /// - Active → Dormant, Archived
    /// - Dormant → Active, Archived
    /// - Archived → (terminal state, no outgoing transitions)
    /// - Imported → Active, Archived
    /// - Same status → always valid (no-op)
    pub fn can_transition_to(self, to: Self) -> bool {
        if self == to {
            return true;
        }
        matches!(
            (self, to),
            (Self::Emerging, Self::Active)
                | (Self::Emerging, Self::Dormant)
                | (Self::Emerging, Self::Archived)
                | (Self::Active, Self::Dormant)
                | (Self::Active, Self::Archived)
                | (Self::Dormant, Self::Active)
                | (Self::Dormant, Self::Archived)
                | (Self::Imported, Self::Active)
                | (Self::Imported, Self::Archived)
        )
    }
}

/// Type of trigger pattern for matching tool inputs to skills.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TriggerType {
    /// Regular expression pattern (e.g., `neo4j|cypher|UNWIND`)
    Regex,
    /// File glob pattern (e.g., `src/neo4j/**`)
    FileGlob,
    /// Semantic vector centroid (embedding cosine similarity)
    Semantic,
    /// MCP mega-tool action pattern (e.g., `note`, `note:create`, `task:create`)
    /// Matches against the extracted MCP pattern via prefix check.
    McpAction,
}

impl fmt::Display for TriggerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Regex => write!(f, "regex"),
            Self::FileGlob => write!(f, "file_glob"),
            Self::Semantic => write!(f, "semantic"),
            Self::McpAction => write!(f, "mcp_action"),
        }
    }
}

impl FromStr for TriggerType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().replace("_", "").as_str() {
            "regex" => Ok(Self::Regex),
            "fileglob" | "glob" => Ok(Self::FileGlob),
            "semantic" => Ok(Self::Semantic),
            "mcpaction" | "mcp" => Ok(Self::McpAction),
            _ => Err(format!("Unknown trigger type: {}", s)),
        }
    }
}

// ============================================================================
// Trigger
// ============================================================================

/// A trigger pattern that determines when a skill should be activated.
///
/// Each skill can have multiple triggers of different types. During hook
/// activation, the tool input is evaluated against all triggers — if any
/// matches above the confidence threshold, the skill is activated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillTrigger {
    /// Type of pattern matching
    pub pattern_type: TriggerType,
    /// The pattern value:
    /// - Regex: a regular expression string
    /// - FileGlob: a glob pattern (e.g., `src/api/**`)
    /// - Semantic: JSON-encoded embedding vector centroid
    /// - McpAction: `mega_tool` or `mega_tool:action` (e.g., `note`, `note:create`)
    pub pattern_value: String,
    /// Minimum confidence score for this trigger to fire (0.0-1.0)
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,
    /// Quality score (F1) of this trigger. Triggers with quality < 0.3
    /// are considered unreliable and skipped during matching.
    #[serde(default)]
    pub quality_score: Option<f64>,
}

fn default_confidence_threshold() -> f64 {
    0.7
}

impl SkillTrigger {
    /// Create a new Regex trigger
    pub fn regex(pattern: impl Into<String>, confidence: f64) -> Self {
        Self {
            pattern_type: TriggerType::Regex,
            pattern_value: pattern.into(),
            confidence_threshold: confidence.clamp(0.0, 1.0),
            quality_score: None,
        }
    }

    /// Create a new FileGlob trigger
    pub fn file_glob(pattern: impl Into<String>, confidence: f64) -> Self {
        Self {
            pattern_type: TriggerType::FileGlob,
            pattern_value: pattern.into(),
            confidence_threshold: confidence.clamp(0.0, 1.0),
            quality_score: None,
        }
    }

    /// Create a new Semantic trigger with an embedding centroid
    pub fn semantic(embedding_json: impl Into<String>, confidence: f64) -> Self {
        Self {
            pattern_type: TriggerType::Semantic,
            pattern_value: embedding_json.into(),
            confidence_threshold: confidence.clamp(0.0, 1.0),
            quality_score: None,
        }
    }

    /// Create a new McpAction trigger.
    ///
    /// `pattern` is either:
    /// - `"mega_tool"` to match any action of that tool (e.g., `"note"`)
    /// - `"mega_tool:action"` to match a specific action (e.g., `"note:create"`)
    pub fn mcp_action(pattern: impl Into<String>, confidence: f64) -> Self {
        Self {
            pattern_type: TriggerType::McpAction,
            pattern_value: pattern.into(),
            confidence_threshold: confidence.clamp(0.0, 1.0),
            quality_score: None,
        }
    }

    /// Check if this trigger is considered reliable based on quality score.
    /// Triggers with quality < 0.3 are unreliable and should be skipped.
    pub fn is_reliable(&self) -> bool {
        self.quality_score.is_none_or(|q| q >= 0.3)
    }
}

// ============================================================================
// Skill Node
// ============================================================================

/// A Neural Skill — an emergent knowledge cluster from the SYNAPSE graph.
///
/// Skills are detected by Louvain community detection on note-to-note
/// synaptic connections. They represent coherent domains of expertise
/// (e.g., "Neo4j Performance", "API Authentication") and can be automatically
/// activated via Claude Code hooks to inject relevant context.
///
/// # Neo4j Relations
/// ```text
/// (Note)-[:MEMBER_OF]->(Skill)           — member notes
/// (Decision)-[:MEMBER_OF_SKILL]->(Skill) — member decisions
/// (Skill)-[:BELONGS_TO]->(Project)       — project ownership
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillNode {
    /// Unique identifier
    pub id: Uuid,
    /// Project this skill belongs to
    pub project_id: Uuid,
    /// Human-readable name (auto-generated from tags or LLM-enriched)
    pub name: String,
    /// Description of the skill's domain of expertise
    #[serde(default)]
    pub description: String,
    /// Current lifecycle status
    #[serde(default)]
    pub status: SkillStatus,

    // --- Trigger patterns ---
    /// Patterns used to match tool inputs during hook activation.
    /// Multiple triggers per skill, evaluated with OR logic.
    #[serde(default)]
    pub trigger_patterns: Vec<SkillTrigger>,
    /// Pre-generated context template (Markdown) with placeholders.
    /// Used as fallback or structure for the MCP tool, not for hooks
    /// (hooks use dynamic assembly with 800-token budget).
    pub context_template: Option<String>,

    // --- Metrics ---
    /// Average energy of member notes (0.0-1.0). Recalculated on maintenance.
    #[serde(default)]
    pub energy: f64,
    /// Louvain intra-cluster cohesion score (0.0-1.0). Higher = tighter cluster.
    #[serde(default)]
    pub cohesion: f64,
    /// Size of the detected Louvain community (cluster node count).
    #[serde(default)]
    pub coverage: i64,
    /// Number of member notes
    #[serde(default)]
    pub note_count: i64,
    /// Number of member decisions
    #[serde(default)]
    pub decision_count: i64,

    // --- Activation tracking ---
    /// Total number of times this skill was activated by hooks
    #[serde(default)]
    pub activation_count: i64,
    /// Hit rate: successful activations / total activations (0.0-1.0)
    #[serde(default)]
    pub hit_rate: f64,
    /// When this skill was last activated by a hook
    pub last_activated: Option<DateTime<Utc>>,

    // --- Versioning ---
    /// Schema version, incremented on re-detection (merge, split, grow)
    #[serde(default = "default_version")]
    pub version: i64,
    /// SHA256 fingerprint of the skill content (for export dedup)
    pub fingerprint: Option<String>,

    // --- Import tracking ---
    /// When this skill was imported (None for natively detected skills)
    pub imported_at: Option<DateTime<Utc>>,
    /// Whether this imported skill has been validated through usage
    #[serde(default)]
    pub is_validated: bool,

    // --- Tags ---
    /// Tags for categorization (derived from member note tags)
    #[serde(default)]
    pub tags: Vec<String>,

    // --- Timestamps ---
    /// When this skill was first detected
    pub created_at: DateTime<Utc>,
    /// When this skill was last updated (metrics, members, triggers)
    pub updated_at: DateTime<Utc>,
}

fn default_version() -> i64 {
    1
}

impl SkillNode {
    /// Create a new Emerging skill with minimal fields.
    pub fn new(project_id: Uuid, name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            project_id,
            name: name.into(),
            description: String::new(),
            status: SkillStatus::Emerging,
            trigger_patterns: vec![],
            context_template: None,
            energy: 0.0,
            cohesion: 0.0,
            coverage: 0,
            note_count: 0,
            decision_count: 0,
            activation_count: 0,
            hit_rate: 0.0,
            last_activated: None,
            version: 1,
            fingerprint: None,
            imported_at: None,
            is_validated: false,
            tags: vec![],
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a new skill with full configuration.
    pub fn new_full(
        project_id: Uuid,
        name: impl Into<String>,
        description: impl Into<String>,
        energy: f64,
        cohesion: f64,
        tags: Vec<String>,
    ) -> Self {
        let mut skill = Self::new(project_id, name);
        skill.description = description.into();
        skill.energy = energy.clamp(0.0, 1.0);
        skill.cohesion = cohesion.clamp(0.0, 1.0);
        skill.tags = tags;
        skill
    }

    /// Check if this skill is matchable by hooks (Active or Emerging).
    pub fn is_matchable(&self) -> bool {
        matches!(self.status, SkillStatus::Active | SkillStatus::Emerging)
    }

    /// Check if this skill is actively used.
    pub fn is_active(&self) -> bool {
        self.status == SkillStatus::Active
    }

    /// Check if this skill was imported from an external package.
    pub fn is_imported(&self) -> bool {
        self.imported_at.is_some()
    }

    /// Get reliable triggers (quality >= 0.3 or unscored).
    pub fn reliable_triggers(&self) -> Vec<&SkillTrigger> {
        self.trigger_patterns
            .iter()
            .filter(|t| t.is_reliable())
            .collect()
    }
}

// ============================================================================
// Activated Skill Context (hook response payload)
// ============================================================================

/// Response payload when a skill is activated by a hook.
///
/// Contains the matched skill, its activated notes/decisions,
/// and the assembled context text ready for injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivatedSkillContext {
    /// The skill that was activated
    pub skill: SkillNode,
    /// Notes activated by spreading activation within this skill
    pub activated_notes: Vec<ActivatedNote>,
    /// Architectural decisions relevant to the activation query
    pub relevant_decisions: Vec<DecisionNode>,
    /// Assembled context text (Markdown), ready for injection.
    /// Budget: 800 tokens max for hooks, unlimited for MCP tool.
    pub context_text: String,
    /// Confidence score of the trigger match (0.0-1.0)
    pub confidence: f64,
}

// ============================================================================
// DTOs
// ============================================================================

/// Request to create a new skill manually
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSkillRequest {
    /// Project this skill belongs to
    pub project_id: Uuid,
    /// Human-readable name
    pub name: String,
    /// Description of the skill's domain
    pub description: Option<String>,
    /// Initial tags
    pub tags: Option<Vec<String>>,
    /// Initial trigger patterns
    pub trigger_patterns: Option<Vec<SkillTrigger>>,
}

/// Request to update a skill
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateSkillRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub status: Option<SkillStatus>,
    pub tags: Option<Vec<String>>,
    pub trigger_patterns: Option<Vec<SkillTrigger>>,
    pub context_template: Option<String>,
}

/// Hook activation request payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookActivateRequest {
    /// Project ID (resolved from .po-config or cwd matching)
    pub project_id: Uuid,
    /// Name of the Claude Code tool being invoked
    pub tool_name: String,
    /// Raw tool input (varies by tool type)
    pub tool_input: serde_json::Value,
    /// Optional session ID for throttle tracking
    pub session_id: Option<String>,
}

/// Hook activation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookActivateResponse {
    /// Assembled context text for additionalContext injection
    pub context: String,
    /// Name of the matched skill
    pub skill_name: String,
    /// ID of the matched skill
    pub skill_id: Uuid,
    /// Confidence of the trigger match (0.0-1.0)
    pub confidence: f64,
    /// Number of notes included in the context
    pub notes_count: usize,
    /// Number of decisions included in the context
    pub decisions_count: usize,
}

/// Session context response for SessionStart hook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContextResponse {
    /// Active skills for this project
    pub active_skills: Vec<SkillSummary>,
    /// Currently active plan (if any)
    pub current_plan: Option<PlanSummary>,
    /// Currently active task (if any)
    pub current_task: Option<TaskSummary>,
    /// Critical notes for this project
    pub critical_notes: Vec<NoteSummary>,
}

/// Compact skill summary for session context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillSummary {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub note_count: i64,
    pub energy: f64,
    pub activation_count: i64,
    pub last_activated: Option<DateTime<Utc>>,
}

impl From<&SkillNode> for SkillSummary {
    fn from(skill: &SkillNode) -> Self {
        Self {
            id: skill.id,
            name: skill.name.clone(),
            description: skill.description.clone(),
            note_count: skill.note_count,
            energy: skill.energy,
            activation_count: skill.activation_count,
            last_activated: skill.last_activated,
        }
    }
}

/// Compact plan summary for session context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanSummary {
    pub id: Uuid,
    pub title: String,
    pub status: String,
    pub completed_tasks: i64,
    pub total_tasks: i64,
}

/// Compact task summary for session context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSummary {
    pub id: Uuid,
    pub title: String,
    pub status: String,
}

/// Compact note summary for session context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteSummary {
    pub id: Uuid,
    pub content: String,
    pub note_type: String,
    pub importance: String,
}

// ============================================================================
// Detection result types
// ============================================================================

/// Result of the skill detection pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Status of the detection
    pub status: DetectionStatus,
    /// Human-readable message
    pub message: String,
    /// Number of skills detected (clusters found)
    pub skills_detected: usize,
    /// Number of new skills created
    pub skills_created: usize,
    /// Number of existing skills updated
    pub skills_updated: usize,
    /// Number of skills unchanged
    pub skills_unchanged: usize,
    /// Suggestion for the user (e.g., "run backfill_synapses")
    pub suggestion: Option<String>,
}

/// Status of the detection process
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DetectionStatus {
    /// Detection completed successfully
    Success,
    /// Not enough data for meaningful detection
    InsufficientData,
    /// Detection failed
    Failed,
}

impl DetectionResult {
    /// Create a successful detection result
    pub fn success(detected: usize, created: usize, updated: usize, unchanged: usize) -> Self {
        Self {
            status: DetectionStatus::Success,
            message: format!(
                "Detected {} skill clusters: {} created, {} updated, {} unchanged",
                detected, created, updated, unchanged
            ),
            skills_detected: detected,
            skills_created: created,
            skills_updated: updated,
            skills_unchanged: unchanged,
            suggestion: None,
        }
    }

    /// Create an insufficient data result
    pub fn insufficient_data(note_count: usize, threshold: usize) -> Self {
        let suggestion = if note_count >= 10 {
            Some("Run admin(action: \"backfill_synapses\") to accelerate synapse creation.".into())
        } else {
            Some("Continue creating notes — synapses will form naturally through usage.".into())
        };

        Self {
            status: DetectionStatus::InsufficientData,
            message: format!(
                "Project has {} notes with synapses. Minimum required: {}.",
                note_count, threshold
            ),
            skills_detected: 0,
            skills_created: 0,
            skills_updated: 0,
            skills_unchanged: 0,
            suggestion,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- SkillStatus tests ---

    #[test]
    fn test_skill_status_default() {
        assert_eq!(SkillStatus::default(), SkillStatus::Emerging);
    }

    #[test]
    fn test_skill_status_display_and_parse() {
        let statuses = vec![
            (SkillStatus::Emerging, "emerging"),
            (SkillStatus::Active, "active"),
            (SkillStatus::Dormant, "dormant"),
            (SkillStatus::Archived, "archived"),
            (SkillStatus::Imported, "imported"),
        ];

        for (status, expected) in statuses {
            assert_eq!(status.to_string(), expected);
            assert_eq!(SkillStatus::from_str(expected).unwrap(), status);
        }
    }

    #[test]
    fn test_skill_status_parse_error() {
        assert!(SkillStatus::from_str("invalid").is_err());
    }

    #[test]
    fn test_skill_status_valid_transitions() {
        // Same-state is always valid
        for status in [
            SkillStatus::Emerging,
            SkillStatus::Active,
            SkillStatus::Dormant,
            SkillStatus::Archived,
            SkillStatus::Imported,
        ] {
            assert!(
                status.can_transition_to(status),
                "{status} → {status} should be valid"
            );
        }

        // Valid forward transitions
        assert!(SkillStatus::Emerging.can_transition_to(SkillStatus::Active));
        assert!(SkillStatus::Emerging.can_transition_to(SkillStatus::Dormant));
        assert!(SkillStatus::Emerging.can_transition_to(SkillStatus::Archived));
        assert!(SkillStatus::Active.can_transition_to(SkillStatus::Dormant));
        assert!(SkillStatus::Active.can_transition_to(SkillStatus::Archived));
        assert!(SkillStatus::Dormant.can_transition_to(SkillStatus::Active));
        assert!(SkillStatus::Dormant.can_transition_to(SkillStatus::Archived));
        assert!(SkillStatus::Imported.can_transition_to(SkillStatus::Active));
        assert!(SkillStatus::Imported.can_transition_to(SkillStatus::Archived));
    }

    #[test]
    fn test_skill_status_invalid_transitions() {
        // Archived is terminal
        assert!(!SkillStatus::Archived.can_transition_to(SkillStatus::Active));
        assert!(!SkillStatus::Archived.can_transition_to(SkillStatus::Emerging));
        assert!(!SkillStatus::Archived.can_transition_to(SkillStatus::Dormant));
        assert!(!SkillStatus::Archived.can_transition_to(SkillStatus::Imported));

        // Can't go back to Emerging
        assert!(!SkillStatus::Active.can_transition_to(SkillStatus::Emerging));
        assert!(!SkillStatus::Dormant.can_transition_to(SkillStatus::Emerging));

        // Can't become Imported after creation
        assert!(!SkillStatus::Active.can_transition_to(SkillStatus::Imported));
        assert!(!SkillStatus::Emerging.can_transition_to(SkillStatus::Imported));

        // Imported can't go to Dormant/Emerging
        assert!(!SkillStatus::Imported.can_transition_to(SkillStatus::Dormant));
        assert!(!SkillStatus::Imported.can_transition_to(SkillStatus::Emerging));
    }

    #[test]
    fn test_skill_status_serde_roundtrip() {
        for status in [
            SkillStatus::Emerging,
            SkillStatus::Active,
            SkillStatus::Dormant,
            SkillStatus::Archived,
            SkillStatus::Imported,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let deserialized: SkillStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, deserialized);
        }
    }

    // --- TriggerType tests ---

    #[test]
    fn test_trigger_type_display_and_parse() {
        let types = vec![
            (TriggerType::Regex, "regex"),
            (TriggerType::FileGlob, "file_glob"),
            (TriggerType::Semantic, "semantic"),
        ];

        for (tt, expected) in types {
            assert_eq!(tt.to_string(), expected);
        }

        // Parse with various formats
        assert_eq!(TriggerType::from_str("regex").unwrap(), TriggerType::Regex);
        assert_eq!(
            TriggerType::from_str("fileglob").unwrap(),
            TriggerType::FileGlob
        );
        assert_eq!(
            TriggerType::from_str("file_glob").unwrap(),
            TriggerType::FileGlob
        );
        assert_eq!(
            TriggerType::from_str("glob").unwrap(),
            TriggerType::FileGlob
        );
        assert_eq!(
            TriggerType::from_str("semantic").unwrap(),
            TriggerType::Semantic
        );
    }

    #[test]
    fn test_trigger_type_serde_roundtrip() {
        for tt in [
            TriggerType::Regex,
            TriggerType::FileGlob,
            TriggerType::Semantic,
        ] {
            let json = serde_json::to_string(&tt).unwrap();
            let deserialized: TriggerType = serde_json::from_str(&json).unwrap();
            assert_eq!(tt, deserialized);
        }
    }

    // --- SkillTrigger tests ---

    #[test]
    fn test_skill_trigger_constructors() {
        let regex = SkillTrigger::regex("neo4j|cypher", 0.6);
        assert_eq!(regex.pattern_type, TriggerType::Regex);
        assert_eq!(regex.pattern_value, "neo4j|cypher");
        assert_eq!(regex.confidence_threshold, 0.6);
        assert!(regex.quality_score.is_none());

        let glob = SkillTrigger::file_glob("src/neo4j/**", 0.8);
        assert_eq!(glob.pattern_type, TriggerType::FileGlob);
        assert_eq!(glob.pattern_value, "src/neo4j/**");

        let semantic = SkillTrigger::semantic("[0.1, 0.2, 0.3]", 0.75);
        assert_eq!(semantic.pattern_type, TriggerType::Semantic);
    }

    #[test]
    fn test_skill_trigger_reliability() {
        let mut trigger = SkillTrigger::regex("test", 0.5);

        // No quality score → reliable
        assert!(trigger.is_reliable());

        // High quality → reliable
        trigger.quality_score = Some(0.8);
        assert!(trigger.is_reliable());

        // Borderline → reliable
        trigger.quality_score = Some(0.3);
        assert!(trigger.is_reliable());

        // Low quality → unreliable
        trigger.quality_score = Some(0.2);
        assert!(!trigger.is_reliable());

        // Zero quality → unreliable
        trigger.quality_score = Some(0.0);
        assert!(!trigger.is_reliable());
    }

    #[test]
    fn test_skill_trigger_serde_roundtrip() {
        let trigger = SkillTrigger::regex("neo4j|cypher", 0.6);
        let json = serde_json::to_string(&trigger).unwrap();
        let deserialized: SkillTrigger = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.pattern_type, trigger.pattern_type);
        assert_eq!(deserialized.pattern_value, trigger.pattern_value);
        assert!(
            (deserialized.confidence_threshold - trigger.confidence_threshold).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_skill_trigger_default_confidence() {
        // Test that default confidence threshold is applied during deserialization
        let json = r#"{"pattern_type":"regex","pattern_value":"test"}"#;
        let trigger: SkillTrigger = serde_json::from_str(json).unwrap();
        assert!((trigger.confidence_threshold - 0.7).abs() < f64::EPSILON);
    }

    // --- SkillNode tests ---

    #[test]
    fn test_skill_node_new() {
        let project_id = Uuid::new_v4();
        let skill = SkillNode::new(project_id, "Neo4j Performance");

        assert_eq!(skill.project_id, project_id);
        assert_eq!(skill.name, "Neo4j Performance");
        assert_eq!(skill.status, SkillStatus::Emerging);
        assert!(skill.description.is_empty());
        assert!(skill.trigger_patterns.is_empty());
        assert_eq!(skill.energy, 0.0);
        assert_eq!(skill.cohesion, 0.0);
        assert_eq!(skill.activation_count, 0);
        assert_eq!(skill.version, 1);
        assert!(!skill.is_imported());
        assert!(skill.is_matchable());
    }

    #[test]
    fn test_skill_node_new_full() {
        let project_id = Uuid::new_v4();
        let skill = SkillNode::new_full(
            project_id,
            "API Auth",
            "Authentication and authorization patterns",
            0.75,
            0.82,
            vec!["auth".into(), "jwt".into()],
        );

        assert_eq!(skill.name, "API Auth");
        assert_eq!(
            skill.description,
            "Authentication and authorization patterns"
        );
        assert!((skill.energy - 0.75).abs() < f64::EPSILON);
        assert!((skill.cohesion - 0.82).abs() < f64::EPSILON);
        assert_eq!(skill.tags, vec!["auth", "jwt"]);
    }

    #[test]
    fn test_skill_node_energy_clamping() {
        let skill = SkillNode::new_full(Uuid::new_v4(), "test", "desc", 1.5, -0.3, vec![]);
        assert!((skill.energy - 1.0).abs() < f64::EPSILON);
        assert!((skill.cohesion - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_skill_node_matchable() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, "test");

        // Emerging → matchable
        skill.status = SkillStatus::Emerging;
        assert!(skill.is_matchable());

        // Active → matchable
        skill.status = SkillStatus::Active;
        assert!(skill.is_matchable());
        assert!(skill.is_active());

        // Dormant → not matchable
        skill.status = SkillStatus::Dormant;
        assert!(!skill.is_matchable());
        assert!(!skill.is_active());

        // Archived → not matchable
        skill.status = SkillStatus::Archived;
        assert!(!skill.is_matchable());

        // Imported → not matchable (needs to be promoted first)
        skill.status = SkillStatus::Imported;
        assert!(!skill.is_matchable());
    }

    #[test]
    fn test_skill_node_imported() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "test");
        assert!(!skill.is_imported());

        skill.imported_at = Some(Utc::now());
        assert!(skill.is_imported());
    }

    #[test]
    fn test_skill_node_reliable_triggers() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "test");

        let good = SkillTrigger {
            quality_score: Some(0.8),
            ..SkillTrigger::regex("good", 0.5)
        };
        let bad = SkillTrigger {
            quality_score: Some(0.1),
            ..SkillTrigger::regex("bad", 0.5)
        };
        let unscored = SkillTrigger::file_glob("src/**", 0.5);

        skill.trigger_patterns = vec![good, bad, unscored];

        let reliable = skill.reliable_triggers();
        assert_eq!(reliable.len(), 2); // good + unscored
        assert_eq!(reliable[0].pattern_value, "good");
        assert_eq!(reliable[1].pattern_value, "src/**");
    }

    #[test]
    fn test_skill_node_serde_roundtrip() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new_full(
            project_id,
            "Neo4j Performance",
            "Optimization patterns for Neo4j queries",
            0.72,
            0.68,
            vec!["neo4j".into(), "performance".into()],
        );
        skill.trigger_patterns = vec![
            SkillTrigger::regex("neo4j|cypher|UNWIND", 0.6),
            SkillTrigger::file_glob("src/neo4j/**", 0.8),
        ];
        skill.activation_count = 42;
        skill.hit_rate = 0.85;
        skill.note_count = 6;
        skill.decision_count = 2;
        skill.coverage = 4;
        skill.last_activated = Some(Utc::now());

        let json = serde_json::to_string_pretty(&skill).unwrap();
        let deserialized: SkillNode = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, skill.id);
        assert_eq!(deserialized.project_id, project_id);
        assert_eq!(deserialized.name, "Neo4j Performance");
        assert_eq!(deserialized.status, SkillStatus::Emerging);
        assert!((deserialized.energy - 0.72).abs() < f64::EPSILON);
        assert_eq!(deserialized.trigger_patterns.len(), 2);
        assert_eq!(deserialized.activation_count, 42);
        assert_eq!(deserialized.note_count, 6);
    }

    // --- SkillSummary tests ---

    #[test]
    fn test_skill_summary_from_node() {
        let skill = SkillNode::new_full(
            Uuid::new_v4(),
            "Test Skill",
            "Description",
            0.7,
            0.5,
            vec![],
        );
        let summary = SkillSummary::from(&skill);

        assert_eq!(summary.id, skill.id);
        assert_eq!(summary.name, "Test Skill");
        assert!((summary.energy - 0.7).abs() < f64::EPSILON);
    }

    // --- DetectionResult tests ---

    #[test]
    fn test_detection_result_success() {
        let result = DetectionResult::success(5, 3, 1, 1);
        assert_eq!(result.status, DetectionStatus::Success);
        assert_eq!(result.skills_detected, 5);
        assert_eq!(result.skills_created, 3);
        assert_eq!(result.skills_updated, 1);
        assert_eq!(result.skills_unchanged, 1);
        assert!(result.suggestion.is_none());
    }

    #[test]
    fn test_detection_result_insufficient_data_low() {
        let result = DetectionResult::insufficient_data(5, 15);
        assert_eq!(result.status, DetectionStatus::InsufficientData);
        assert!(result.message.contains("5 notes"));
        assert!(result.message.contains("15"));
        assert!(result
            .suggestion
            .unwrap()
            .contains("Continue creating notes"));
    }

    #[test]
    fn test_detection_result_insufficient_data_close() {
        let result = DetectionResult::insufficient_data(12, 15);
        assert_eq!(result.status, DetectionStatus::InsufficientData);
        assert!(result.suggestion.unwrap().contains("backfill_synapses"));
    }

    // --- HookActivateRequest tests ---

    #[test]
    fn test_hook_activate_request_serde() {
        let req = HookActivateRequest {
            project_id: Uuid::new_v4(),
            tool_name: "Grep".to_string(),
            tool_input: serde_json::json!({"pattern": "reinforce_synapses"}),
            session_id: Some("test-session".to_string()),
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: HookActivateRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.tool_name, "Grep");
        assert_eq!(deserialized.session_id, Some("test-session".to_string()));
    }

    // --- HookActivateResponse tests ---

    #[test]
    fn test_hook_activate_response_serde() {
        let resp = HookActivateResponse {
            context: "## Neo4j Performance\nSome context".to_string(),
            skill_name: "Neo4j Performance".to_string(),
            skill_id: Uuid::new_v4(),
            confidence: 0.85,
            notes_count: 4,
            decisions_count: 1,
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: HookActivateResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.skill_name, "Neo4j Performance");
        assert_eq!(deserialized.notes_count, 4);
    }
}
