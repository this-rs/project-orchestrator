//! Knowledge Notes models and DTOs
//!
//! Knowledge Notes capture contextual information from conversations:
//! guidelines, gotchas, patterns, tips, and assertions that can be
//! linked to code entities and automatically surfaced to agents.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

// ============================================================================
// Core Enums
// ============================================================================

/// Type of knowledge note
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum NoteType {
    /// A rule or best practice to follow
    Guideline,
    /// A trap or pitfall to avoid
    Gotcha,
    /// An established pattern in the codebase
    Pattern,
    /// Temporary context (expires quickly)
    Context,
    /// A useful tip or suggestion
    Tip,
    /// A general observation
    Observation,
    /// A verifiable rule/assertion about the code
    Assertion,
}

impl fmt::Display for NoteType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Guideline => write!(f, "guideline"),
            Self::Gotcha => write!(f, "gotcha"),
            Self::Pattern => write!(f, "pattern"),
            Self::Context => write!(f, "context"),
            Self::Tip => write!(f, "tip"),
            Self::Observation => write!(f, "observation"),
            Self::Assertion => write!(f, "assertion"),
        }
    }
}

impl FromStr for NoteType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "guideline" => Ok(Self::Guideline),
            "gotcha" => Ok(Self::Gotcha),
            "pattern" => Ok(Self::Pattern),
            "context" => Ok(Self::Context),
            "tip" => Ok(Self::Tip),
            "observation" => Ok(Self::Observation),
            "assertion" => Ok(Self::Assertion),
            _ => Err(format!("Unknown note type: {}", s)),
        }
    }
}

/// Lifecycle status of a note
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum NoteStatus {
    /// Note is valid and active
    #[default]
    Active,
    /// Code changed, note needs human review
    NeedsReview,
    /// Note hasn't been confirmed for a long time
    Stale,
    /// Note has been invalidated (code deleted, etc.)
    Obsolete,
    /// Note archived with history preserved
    Archived,
}

impl fmt::Display for NoteStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::NeedsReview => write!(f, "needs_review"),
            Self::Stale => write!(f, "stale"),
            Self::Obsolete => write!(f, "obsolete"),
            Self::Archived => write!(f, "archived"),
        }
    }
}

impl FromStr for NoteStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "active" => Ok(Self::Active),
            "needs_review" => Ok(Self::NeedsReview),
            "stale" => Ok(Self::Stale),
            "obsolete" => Ok(Self::Obsolete),
            "archived" => Ok(Self::Archived),
            _ => Err(format!("Unknown note status: {}", s)),
        }
    }
}

/// Importance level of a note
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum NoteImportance {
    Low,
    #[default]
    Medium,
    High,
    Critical,
}

impl NoteImportance {
    /// Get the numeric weight for scoring (0.0 - 1.0)
    pub fn weight(&self) -> f64 {
        match self {
            Self::Low => 0.3,
            Self::Medium => 0.5,
            Self::High => 0.8,
            Self::Critical => 1.0,
        }
    }

    /// Get the staleness decay multiplier (lower = decays slower)
    pub fn decay_factor(&self) -> f64 {
        match self {
            Self::Critical => 0.5,
            Self::High => 0.7,
            Self::Medium => 1.0,
            Self::Low => 1.3,
        }
    }
}

impl fmt::Display for NoteImportance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

impl FromStr for NoteImportance {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "critical" => Ok(Self::Critical),
            _ => Err(format!("Unknown importance level: {}", s)),
        }
    }
}

/// Type of entity a note can be anchored to
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Project,
    File,
    Module,
    Function,
    Struct,
    Trait,
    Enum,
    Impl,
    Task,
    Plan,
    Commit,
    Decision,
    // Workspace-related entities
    Workspace,
    WorkspaceMilestone,
    Resource,
    Component,
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Project => write!(f, "project"),
            Self::File => write!(f, "file"),
            Self::Module => write!(f, "module"),
            Self::Function => write!(f, "function"),
            Self::Struct => write!(f, "struct"),
            Self::Trait => write!(f, "trait"),
            Self::Enum => write!(f, "enum"),
            Self::Impl => write!(f, "impl"),
            Self::Task => write!(f, "task"),
            Self::Plan => write!(f, "plan"),
            Self::Commit => write!(f, "commit"),
            Self::Decision => write!(f, "decision"),
            Self::Workspace => write!(f, "workspace"),
            Self::WorkspaceMilestone => write!(f, "workspace_milestone"),
            Self::Resource => write!(f, "resource"),
            Self::Component => write!(f, "component"),
        }
    }
}

impl FromStr for EntityType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().replace("_", "").as_str() {
            "project" => Ok(Self::Project),
            "file" => Ok(Self::File),
            "module" => Ok(Self::Module),
            "function" => Ok(Self::Function),
            "struct" => Ok(Self::Struct),
            "trait" => Ok(Self::Trait),
            "enum" => Ok(Self::Enum),
            "impl" => Ok(Self::Impl),
            "task" => Ok(Self::Task),
            "plan" => Ok(Self::Plan),
            "commit" => Ok(Self::Commit),
            "decision" => Ok(Self::Decision),
            "workspace" => Ok(Self::Workspace),
            "workspacemilestone" => Ok(Self::WorkspaceMilestone),
            "resource" => Ok(Self::Resource),
            "component" => Ok(Self::Component),
            _ => Err(format!("Unknown entity type: {}", s)),
        }
    }
}

// ============================================================================
// Scope
// ============================================================================

/// Hierarchical scope for a note
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "path", rename_all = "snake_case")]
pub enum NoteScope {
    /// Applies to an entire workspace (propagates to all projects)
    Workspace,
    /// Applies to the entire project
    Project,
    /// Applies to a specific module
    Module(String),
    /// Applies to a specific file
    File(String),
    /// Applies to a specific function
    Function(String),
    /// Applies to a specific struct
    Struct(String),
    /// Applies to a specific trait
    Trait(String),
}

impl NoteScope {
    /// Get the scope level (higher = more specific)
    /// Lower numbers are broader scope (workspace = -1, project = 0)
    pub fn specificity(&self) -> i8 {
        match self {
            Self::Workspace => -1, // Broadest scope
            Self::Project => 0,
            Self::Module(_) => 1,
            Self::File(_) => 2,
            Self::Function(_) | Self::Struct(_) | Self::Trait(_) => 3,
        }
    }
}

impl fmt::Display for NoteScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workspace => write!(f, "workspace"),
            Self::Project => write!(f, "project"),
            Self::Module(path) => write!(f, "module:{}", path),
            Self::File(path) => write!(f, "file:{}", path),
            Self::Function(name) => write!(f, "function:{}", name),
            Self::Struct(name) => write!(f, "struct:{}", name),
            Self::Trait(name) => write!(f, "trait:{}", name),
        }
    }
}

// ============================================================================
// Anchoring
// ============================================================================

/// Semantic anchor linking a note to a code entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteAnchor {
    /// Type of entity this anchor points to
    pub entity_type: EntityType,
    /// Identifier of the entity in the graph (UUID or path)
    pub entity_id: String,
    /// SHA256 hash of the entity's signature (for functions, structs)
    pub signature_hash: Option<String>,
    /// SHA256 hash of the entity's body (for functions)
    pub body_hash: Option<String>,
    /// When this anchor was last verified as valid
    pub last_verified: DateTime<Utc>,
    /// Whether the anchor is currently valid
    #[serde(default = "default_true")]
    pub is_valid: bool,
}

fn default_true() -> bool {
    true
}

impl NoteAnchor {
    /// Create a new anchor for a code entity
    pub fn new(entity_type: EntityType, entity_id: String) -> Self {
        Self {
            entity_type,
            entity_id,
            signature_hash: None,
            body_hash: None,
            last_verified: Utc::now(),
            is_valid: true,
        }
    }

    /// Create an anchor with semantic hashes
    pub fn with_hashes(
        entity_type: EntityType,
        entity_id: String,
        signature_hash: Option<String>,
        body_hash: Option<String>,
    ) -> Self {
        Self {
            entity_type,
            entity_id,
            signature_hash,
            body_hash,
            last_verified: Utc::now(),
            is_valid: true,
        }
    }
}

// ============================================================================
// Change History
// ============================================================================

/// Type of change in note history
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChangeType {
    /// Note was created
    Created,
    /// Note content was updated
    Updated,
    /// Note was confirmed as still valid
    Confirmed,
    /// Note status changed
    StatusChanged,
    /// Note was migrated to a new entity (e.g., after rename)
    Migrated,
    /// Anchor was added
    AnchorAdded,
    /// Anchor was removed
    AnchorRemoved,
    /// Note was superseded by another
    Superseded,
}

/// A change in the note's history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteChange {
    /// When the change occurred
    pub timestamp: DateTime<Utc>,
    /// Type of change
    pub change_type: ChangeType,
    /// Additional details about the change
    pub details: Option<serde_json::Value>,
    /// Who made the change (user ID or agent ID)
    pub actor: String,
}

impl NoteChange {
    /// Create a new change record
    pub fn new(change_type: ChangeType, actor: String) -> Self {
        Self {
            timestamp: Utc::now(),
            change_type,
            details: None,
            actor,
        }
    }

    /// Create a change with details
    pub fn with_details(
        change_type: ChangeType,
        actor: String,
        details: serde_json::Value,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            change_type,
            details: Some(details),
            actor,
        }
    }
}

// ============================================================================
// Assertions
// ============================================================================

/// Type of assertion check
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AssertionCheckType {
    /// Check that an entity exists
    Exists,
    /// Check that an entity does NOT exist
    NotExists,
    /// Check that a signature contains specific elements
    SignatureContains,
    /// Check that a dependency relationship exists
    DependsOn,
    /// Check that a call relationship exists
    Calls,
    /// Check that a call relationship does NOT exist
    NoCalls,
    /// Check that a type implements a trait
    Implements,
}

/// Action to take when an assertion is violated
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ViolationAction {
    /// Just warn, don't change note status
    Warn,
    /// Flag the note as needing review
    #[default]
    FlagNote,
    /// Block (mark note as needs_review and log warning)
    Block,
}

/// A verifiable assertion rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertionRule {
    /// Type of check to perform
    pub check_type: AssertionCheckType,
    /// Target entity (e.g., "function:validate_user", "file:src/auth.rs")
    pub target: String,
    /// Optional file pattern to scope the check
    pub file_pattern: Option<String>,
    /// Additional parameters for the check
    pub parameters: Option<serde_json::Value>,
    /// What to do when the assertion fails
    #[serde(default)]
    pub on_violation: ViolationAction,
}

/// Result of verifying an assertion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertionResult {
    /// Whether the assertion passed
    pub passed: bool,
    /// Human-readable message
    pub message: String,
    /// Details about what was checked
    pub details: Option<serde_json::Value>,
    /// When the check was performed
    pub checked_at: DateTime<Utc>,
}

impl AssertionResult {
    pub fn passed(message: impl Into<String>) -> Self {
        Self {
            passed: true,
            message: message.into(),
            details: None,
            checked_at: Utc::now(),
        }
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self {
            passed: false,
            message: message.into(),
            details: None,
            checked_at: Utc::now(),
        }
    }
}

// ============================================================================
// Main Note Struct
// ============================================================================

/// A Knowledge Note capturing contextual information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    /// Unique identifier
    pub id: Uuid,
    /// Project this note belongs to (None = global note, cross-project)
    pub project_id: Option<Uuid>,
    /// Type of note
    pub note_type: NoteType,
    /// Current lifecycle status
    pub status: NoteStatus,
    /// Importance level
    pub importance: NoteImportance,
    /// Hierarchical scope
    pub scope: NoteScope,
    /// The actual content/text of the note
    pub content: String,
    /// Tags for categorization and search
    #[serde(default)]
    pub tags: Vec<String>,

    // Anchoring
    /// Semantic anchors to code entities
    #[serde(default)]
    pub anchors: Vec<NoteAnchor>,

    // Lifecycle
    /// When the note was created
    pub created_at: DateTime<Utc>,
    /// Who created the note
    pub created_by: String,
    /// When the note was last confirmed as valid
    pub last_confirmed_at: Option<DateTime<Utc>>,
    /// Who last confirmed the note
    pub last_confirmed_by: Option<String>,
    /// Current staleness score (0.0 = fresh, 1.0 = very stale)
    #[serde(default)]
    pub staleness_score: f64,

    // Succession
    /// ID of the note this one supersedes (if any)
    pub supersedes: Option<Uuid>,
    /// ID of the note that superseded this one (if any)
    pub superseded_by: Option<Uuid>,

    // History
    /// Change history for audit trail
    #[serde(default)]
    pub changes: Vec<NoteChange>,

    // Assertions
    /// Assertion rule if this is an assertion note
    pub assertion_rule: Option<AssertionRule>,
    /// Last assertion result (for assertion notes)
    pub last_assertion_result: Option<AssertionResult>,
}

impl Note {
    /// Create a new note with minimal fields
    pub fn new(
        project_id: Option<Uuid>,
        note_type: NoteType,
        content: String,
        created_by: String,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            project_id,
            note_type,
            status: NoteStatus::Active,
            importance: NoteImportance::Medium,
            scope: NoteScope::Project,
            content,
            tags: vec![],
            anchors: vec![],
            created_at: now,
            created_by: created_by.clone(),
            last_confirmed_at: Some(now),
            last_confirmed_by: Some(created_by.clone()),
            staleness_score: 0.0,
            supersedes: None,
            superseded_by: None,
            changes: vec![NoteChange::new(ChangeType::Created, created_by)],
            assertion_rule: None,
            last_assertion_result: None,
        }
    }

    /// Create a new note with full configuration
    pub fn new_full(
        project_id: Option<Uuid>,
        note_type: NoteType,
        importance: NoteImportance,
        scope: NoteScope,
        content: String,
        tags: Vec<String>,
        created_by: String,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            project_id,
            note_type,
            status: NoteStatus::Active,
            importance,
            scope,
            content,
            tags,
            anchors: vec![],
            created_at: now,
            created_by: created_by.clone(),
            last_confirmed_at: Some(now),
            last_confirmed_by: Some(created_by.clone()),
            staleness_score: 0.0,
            supersedes: None,
            superseded_by: None,
            changes: vec![NoteChange::new(ChangeType::Created, created_by)],
            assertion_rule: None,
            last_assertion_result: None,
        }
    }

    /// Add an anchor to this note
    pub fn add_anchor(&mut self, anchor: NoteAnchor, actor: &str) {
        self.anchors.push(anchor);
        self.changes
            .push(NoteChange::new(ChangeType::AnchorAdded, actor.to_string()));
    }

    /// Confirm this note is still valid
    pub fn confirm(&mut self, confirmed_by: &str) {
        let now = Utc::now();
        self.last_confirmed_at = Some(now);
        self.last_confirmed_by = Some(confirmed_by.to_string());
        self.staleness_score = 0.0;
        self.changes.push(NoteChange::new(
            ChangeType::Confirmed,
            confirmed_by.to_string(),
        ));
    }

    /// Check if the note is active and not stale
    pub fn is_active(&self) -> bool {
        self.status == NoteStatus::Active
    }

    /// Check if the note needs attention
    pub fn needs_attention(&self) -> bool {
        matches!(self.status, NoteStatus::NeedsReview | NoteStatus::Stale)
    }

    /// Get the base decay days for staleness calculation
    pub fn base_decay_days(&self) -> f64 {
        match self.note_type {
            NoteType::Context => 30.0, // Expires quickly
            NoteType::Tip => 90.0,     // Medium expiry
            NoteType::Observation => 90.0,
            NoteType::Gotcha => 180.0,
            NoteType::Guideline => 365.0, // Stable for a long time
            NoteType::Pattern => 365.0,
            NoteType::Assertion => f64::MAX, // Never stale (verified by code)
        }
    }
}

// ============================================================================
// Propagated Note (for context retrieval)
// ============================================================================

/// A note with propagation information for context retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagatedNote {
    /// The note itself
    pub note: Note,
    /// Relevance score (0.0 - 1.0)
    pub relevance_score: f64,
    /// Entity from which this note was propagated
    pub source_entity: String,
    /// Path through the graph to reach the target
    pub propagation_path: Vec<String>,
    /// Distance in the graph (0 = direct, 1+ = propagated)
    pub distance: u32,
    /// Average PageRank of nodes on the propagation path (for debug/introspection).
    /// Higher values mean the note was propagated through structurally important hubs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path_pagerank: Option<f64>,
}

// ============================================================================
// DTOs for API
// ============================================================================

/// Request to create a new note
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateNoteRequest {
    /// Project ID (optional — None for global/cross-project notes)
    pub project_id: Option<Uuid>,
    /// Type of note
    pub note_type: NoteType,
    /// Content of the note
    pub content: String,
    /// Importance level (optional, defaults to Medium)
    pub importance: Option<NoteImportance>,
    /// Scope (optional, defaults to Project)
    pub scope: Option<NoteScope>,
    /// Tags for categorization
    pub tags: Option<Vec<String>>,
    /// Initial anchors to attach
    pub anchors: Option<Vec<CreateAnchorRequest>>,
    /// Assertion rule (for assertion notes)
    pub assertion_rule: Option<AssertionRule>,
}

/// Request to create an anchor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateAnchorRequest {
    pub entity_type: EntityType,
    pub entity_id: String,
    pub signature_hash: Option<String>,
    pub body_hash: Option<String>,
}

/// Request to update a note
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateNoteRequest {
    pub content: Option<String>,
    pub importance: Option<NoteImportance>,
    pub status: Option<NoteStatus>,
    pub tags: Option<Vec<String>>,
}

/// Request to link a note to an entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkNoteRequest {
    pub entity_type: EntityType,
    pub entity_id: String,
}

/// Filters for listing notes
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoteFilters {
    /// Filter by status
    pub status: Option<Vec<NoteStatus>>,
    /// Filter by note type
    pub note_type: Option<Vec<NoteType>>,
    /// Filter by importance
    pub importance: Option<Vec<NoteImportance>>,
    /// Filter by tags (any match)
    pub tags: Option<Vec<String>>,
    /// Filter by scope type
    pub scope_type: Option<String>,
    /// Search in content
    pub search: Option<String>,
    /// Minimum staleness score
    pub min_staleness: Option<f64>,
    /// Maximum staleness score
    pub max_staleness: Option<f64>,
    /// Filter global notes only (no project_id)
    pub global_only: Option<bool>,
    /// Pagination: limit
    pub limit: Option<i64>,
    /// Pagination: offset
    pub offset: Option<i64>,
    /// Sort field
    pub sort_by: Option<String>,
    /// Sort order (asc/desc)
    pub sort_order: Option<String>,
}

/// Response with contextual notes for an entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteContextResponse {
    /// Direct notes attached to the entity
    pub direct_notes: Vec<Note>,
    /// Propagated notes from related entities
    pub propagated_notes: Vec<PropagatedNote>,
    /// Total count of relevant notes
    pub total_count: usize,
}

/// Note search result with relevance score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteSearchHit {
    pub note: Note,
    pub score: f64,
    /// Highlighted snippets from content
    pub highlights: Option<Vec<String>>,
}

// ============================================================================
// Status Update
// ============================================================================

/// Request to update note status with optional migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteStatusUpdate {
    pub note_id: Uuid,
    pub new_status: NoteStatus,
    pub reason: String,
    /// New hashes if the entity changed
    pub new_hashes: Option<(String, String)>,
    /// Migration target if entity was renamed
    pub migration_target: Option<MigrationTarget>,
}

/// Target for note migration (when entity is renamed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationTarget {
    pub entity_type: EntityType,
    pub new_entity_id: String,
    pub similarity_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_type_display_and_parse() {
        let types = vec![
            (NoteType::Guideline, "guideline"),
            (NoteType::Gotcha, "gotcha"),
            (NoteType::Pattern, "pattern"),
            (NoteType::Context, "context"),
            (NoteType::Tip, "tip"),
            (NoteType::Observation, "observation"),
            (NoteType::Assertion, "assertion"),
        ];

        for (note_type, expected) in types {
            assert_eq!(note_type.to_string(), expected);
            assert_eq!(NoteType::from_str(expected).unwrap(), note_type);
        }
    }

    #[test]
    fn test_note_status_display_and_parse() {
        let statuses = vec![
            (NoteStatus::Active, "active"),
            (NoteStatus::NeedsReview, "needs_review"),
            (NoteStatus::Stale, "stale"),
            (NoteStatus::Obsolete, "obsolete"),
            (NoteStatus::Archived, "archived"),
        ];

        for (status, expected) in statuses {
            assert_eq!(status.to_string(), expected);
            assert_eq!(NoteStatus::from_str(expected).unwrap(), status);
        }
    }

    #[test]
    fn test_note_importance_weights() {
        assert_eq!(NoteImportance::Low.weight(), 0.3);
        assert_eq!(NoteImportance::Medium.weight(), 0.5);
        assert_eq!(NoteImportance::High.weight(), 0.8);
        assert_eq!(NoteImportance::Critical.weight(), 1.0);
    }

    #[test]
    fn test_note_creation() {
        let project_id = Uuid::new_v4();
        let note = Note::new(
            Some(project_id),
            NoteType::Guideline,
            "Always use Result for error handling".to_string(),
            "claude".to_string(),
        );

        assert_eq!(note.project_id, Some(project_id));
        assert_eq!(note.note_type, NoteType::Guideline);
        assert_eq!(note.status, NoteStatus::Active);
        assert_eq!(note.importance, NoteImportance::Medium);
        assert!(note.is_active());
        assert!(!note.needs_attention());
        assert_eq!(note.changes.len(), 1);
        assert_eq!(note.changes[0].change_type, ChangeType::Created);
    }

    #[test]
    fn test_note_confirm() {
        let mut note = Note::new(
            Some(Uuid::new_v4()),
            NoteType::Tip,
            "Use async/await".to_string(),
            "user".to_string(),
        );
        note.staleness_score = 0.5;

        note.confirm("reviewer");

        assert_eq!(note.staleness_score, 0.0);
        assert!(note.last_confirmed_by.is_some());
        assert_eq!(note.changes.len(), 2);
        assert_eq!(note.changes[1].change_type, ChangeType::Confirmed);
    }

    #[test]
    fn test_note_scope_specificity() {
        assert!(NoteScope::Project.specificity() < NoteScope::Module("foo".into()).specificity());
        assert!(
            NoteScope::Module("foo".into()).specificity()
                < NoteScope::File("bar".into()).specificity()
        );
        assert!(
            NoteScope::File("bar".into()).specificity()
                < NoteScope::Function("baz".into()).specificity()
        );
    }

    #[test]
    fn test_note_serialization() {
        let note = Note::new(
            Some(Uuid::new_v4()),
            NoteType::Gotcha,
            "Watch out for null".to_string(),
            "agent".to_string(),
        );

        let json = serde_json::to_string(&note).unwrap();
        let deserialized: Note = serde_json::from_str(&json).unwrap();

        assert_eq!(note.id, deserialized.id);
        assert_eq!(note.note_type, deserialized.note_type);
        assert_eq!(note.content, deserialized.content);
    }

    #[test]
    fn test_assertion_result() {
        let passed = AssertionResult::passed("Check passed");
        assert!(passed.passed);

        let failed = AssertionResult::failed("Check failed");
        assert!(!failed.passed);
    }

    #[test]
    fn test_base_decay_days() {
        let context_note = Note::new(
            Some(Uuid::new_v4()),
            NoteType::Context,
            "temp".into(),
            "user".into(),
        );
        let guideline_note = Note::new(
            Some(Uuid::new_v4()),
            NoteType::Guideline,
            "stable".into(),
            "user".into(),
        );
        let assertion_note = Note::new(
            Some(Uuid::new_v4()),
            NoteType::Assertion,
            "rule".into(),
            "user".into(),
        );

        assert!(context_note.base_decay_days() < guideline_note.base_decay_days());
        assert!(guideline_note.base_decay_days() < assertion_note.base_decay_days());
    }

    #[test]
    fn test_entity_type_workspace_variants_display() {
        assert_eq!(EntityType::Workspace.to_string(), "workspace");
        assert_eq!(
            EntityType::WorkspaceMilestone.to_string(),
            "workspace_milestone"
        );
        assert_eq!(EntityType::Resource.to_string(), "resource");
        assert_eq!(EntityType::Component.to_string(), "component");
    }

    #[test]
    fn test_entity_type_workspace_variants_from_str() {
        assert_eq!(
            EntityType::from_str("workspace").unwrap(),
            EntityType::Workspace
        );
        assert_eq!(
            EntityType::from_str("workspacemilestone").unwrap(),
            EntityType::WorkspaceMilestone
        );
        assert_eq!(
            EntityType::from_str("resource").unwrap(),
            EntityType::Resource
        );
        assert_eq!(
            EntityType::from_str("component").unwrap(),
            EntityType::Component
        );
    }

    #[test]
    fn test_note_scope_workspace_specificity() {
        // Workspace is the broadest scope (specificity -1)
        assert_eq!(NoteScope::Workspace.specificity(), -1);
        // Workspace should be less specific than Project
        assert!(NoteScope::Workspace.specificity() < NoteScope::Project.specificity());
        // Workspace should be less specific than all other scopes
        assert!(NoteScope::Workspace.specificity() < NoteScope::Module("m".into()).specificity());
        assert!(NoteScope::Workspace.specificity() < NoteScope::File("f".into()).specificity());
    }

    #[test]
    fn test_note_scope_workspace_display() {
        assert_eq!(NoteScope::Workspace.to_string(), "workspace");
    }

    /// Conceptual test documenting PageRank-weighted propagation scoring.
    ///
    /// In Neo4j, the Cypher query computes:
    ///   avg_path_pagerank = mean(COALESCE(node.pagerank, 0.05)) over path nodes
    ///   score = (1/(distance+1)) * importance * (1 + avg_path_pagerank * 5)
    ///
    /// Expected behavior with two notes at equal distance (2 hops) and importance (high=0.8):
    ///   - Note A propagated via hub (pagerank 0.15): multiplier = 1 + 0.15 * 5 = 1.75
    ///     → score = (1/3) * 0.8 * 1.75 = 0.467
    ///   - Note B propagated via leaf (pagerank 0.01): multiplier = 1 + 0.01 * 5 = 1.05
    ///     → score = (1/3) * 0.8 * 1.05 = 0.280
    ///   - Note A should rank higher than Note B.
    ///
    /// Fallback: if no pagerank on any node, avg = 0.05, multiplier = 1.25
    ///   → score = (1/3) * 0.8 * 1.25 = 0.333 (slightly above old formula's 0.267)
    #[test]
    fn test_propagated_note_pagerank_scoring_documentation() {
        // Simulate the scoring formula in Rust (mirrors the Cypher computation)
        let distance: f64 = 2.0;
        let importance_weight: f64 = 0.8; // "high"

        // Via hub (pagerank 0.15)
        let hub_pagerank = 0.15;
        let hub_multiplier = 1.0 + hub_pagerank * 5.0;
        let hub_score = (1.0 / (distance + 1.0)) * importance_weight * hub_multiplier;

        // Via leaf (pagerank 0.01)
        let leaf_pagerank = 0.01;
        let leaf_multiplier = 1.0 + leaf_pagerank * 5.0;
        let leaf_score = (1.0 / (distance + 1.0)) * importance_weight * leaf_multiplier;

        // Via unknown node (no pagerank → default 0.05)
        let default_pagerank = 0.05;
        let default_multiplier = 1.0 + default_pagerank * 5.0;
        let default_score = (1.0 / (distance + 1.0)) * importance_weight * default_multiplier;

        // Old formula (without pagerank weighting)
        let old_score = (1.0 / (distance + 1.0)) * importance_weight;

        // Hub path scores higher than leaf path
        assert!(
            hub_score > leaf_score,
            "Hub path ({:.3}) should score higher than leaf path ({:.3})",
            hub_score,
            leaf_score
        );

        // Both are higher than old formula (multiplier always >= 1.0)
        assert!(hub_score > old_score);
        assert!(leaf_score > old_score);

        // Default (no pagerank data) is close to old formula (multiplier = 1.25)
        assert!((default_score - old_score).abs() < 0.1_f64);

        // Verify the PropagatedNote struct carries the path_pagerank field
        let pn = PropagatedNote {
            note: Note::new(None, NoteType::Tip, "test".to_string(), "test".to_string()),
            relevance_score: hub_score,
            source_entity: "hub_file".to_string(),
            propagation_path: vec!["hub_file".to_string(), "target".to_string()],
            distance: 2,
            path_pagerank: Some(hub_pagerank),
        };
        assert_eq!(pn.path_pagerank, Some(0.15));
        assert!(pn.relevance_score > 0.4);
    }
}
