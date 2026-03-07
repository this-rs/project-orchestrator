//! Protocol data model and DTOs
//!
//! Defines the core types for the Pattern Federation Protocol system:
//! - [`Protocol`]: A finite state machine (FSM) defining a workflow
//! - [`ProtocolState`]: A state within a protocol FSM
//! - [`ProtocolTransition`]: A transition between two states
//! - [`ProtocolCategory`]: System vs Business protocol classification
//! - [`StateType`]: Start, Intermediate, or Terminal state classification

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

use super::routing::RelevanceVector;

// ============================================================================
// Enums
// ============================================================================

/// Category of a Protocol.
///
/// Protocols are classified by their origin and trigger behavior:
/// - **System**: Auto-triggered by PO for self-maintenance (inference, health checks)
/// - **Business**: Created and piloted by the agent for user-facing workflows
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolCategory {
    /// Auto-triggered by PO for self-maintenance (hidden from UI by default)
    System,
    /// Created/piloted by the agent for user-facing workflows
    #[default]
    Business,
}

impl fmt::Display for ProtocolCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::Business => write!(f, "business"),
        }
    }
}

impl FromStr for ProtocolCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "system" => Ok(Self::System),
            "business" => Ok(Self::Business),
            _ => Err(format!("Unknown protocol category: {}", s)),
        }
    }
}

/// Type of a state within a protocol FSM.
///
/// Determines the behavior and role of the state:
/// - **Start**: The initial entry point (exactly one per protocol)
/// - **Intermediate**: A processing step between start and terminal
/// - **Terminal**: A final state where the protocol run ends
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum StateType {
    /// Initial entry point of the protocol
    Start,
    /// Processing step between start and terminal
    #[default]
    Intermediate,
    /// Final state where the protocol run ends
    Terminal,
}

impl fmt::Display for StateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Start => write!(f, "start"),
            Self::Intermediate => write!(f, "intermediate"),
            Self::Terminal => write!(f, "terminal"),
        }
    }
}

impl FromStr for StateType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "start" => Ok(Self::Start),
            "intermediate" => Ok(Self::Intermediate),
            "terminal" => Ok(Self::Terminal),
            _ => Err(format!("Unknown state type: {}", s)),
        }
    }
}

/// Trigger mode for automatic protocol execution.
///
/// Determines how a protocol can be started:
/// - **Manual**: Only via explicit `protocol(start_run)` — the default
/// - **Event**: Triggered automatically by system events (post_sync, post_import, etc.)
/// - **Scheduled**: Triggered on a periodic schedule (hourly, daily, weekly)
/// - **Auto**: Combines Event + Scheduled — triggered by both events and schedule
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TriggerMode {
    /// Only via explicit start_run
    #[default]
    Manual,
    /// Triggered by system events
    Event,
    /// Triggered on a periodic schedule
    Scheduled,
    /// Event + Scheduled combined
    Auto,
}

impl fmt::Display for TriggerMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manual => write!(f, "manual"),
            Self::Event => write!(f, "event"),
            Self::Scheduled => write!(f, "scheduled"),
            Self::Auto => write!(f, "auto"),
        }
    }
}

impl FromStr for TriggerMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "manual" => Ok(Self::Manual),
            "event" => Ok(Self::Event),
            "scheduled" => Ok(Self::Scheduled),
            "auto" => Ok(Self::Auto),
            _ => Err(format!("Unknown trigger mode: {}", s)),
        }
    }
}

impl TriggerMode {
    /// Returns true if this mode responds to events.
    pub fn listens_to_events(&self) -> bool {
        matches!(self, Self::Event | Self::Auto)
    }

    /// Returns true if this mode responds to scheduling.
    pub fn is_scheduled(&self) -> bool {
        matches!(self, Self::Scheduled | Self::Auto)
    }
}

/// Configuration for automatic protocol triggers.
///
/// Specifies which events and/or schedule should trigger the protocol.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TriggerConfig {
    /// System events that trigger this protocol (e.g., "post_sync", "post_import", "post_plan_complete")
    #[serde(default)]
    pub events: Vec<String>,
    /// Periodic schedule: "hourly", "daily", "weekly"
    pub schedule: Option<String>,
    /// Optional conditions that must be met (reserved for future guard expressions)
    #[serde(default)]
    pub conditions: Vec<String>,
}

// ============================================================================
// Protocol
// ============================================================================

/// A Protocol — a finite state machine (FSM) defining a workflow.
///
/// Protocols are the core building block of Pattern Federation. Each protocol
/// defines a set of states and transitions that model a repeatable process
/// (e.g., code review, deployment, bug triage).
///
/// # Neo4j Relations
/// ```text
/// (Protocol)-[:HAS_STATE]->(ProtocolState)           — states in this FSM
/// (Protocol)-[:HAS_TRANSITION]->(ProtocolTransition) — transitions in this FSM
/// (Protocol)-[:BELONGS_TO]->(Project)                — project ownership
/// (Protocol)-[:BELONGS_TO_SKILL]->(Skill)            — optional skill link
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Protocol {
    /// Unique identifier
    pub id: Uuid,
    /// Human-readable name (e.g., "Code Review Protocol")
    pub name: String,
    /// Description of the protocol's purpose and behavior
    #[serde(default)]
    pub description: String,
    /// Project this protocol belongs to
    pub project_id: Uuid,
    /// Optional link to a Neural Skill (for context injection)
    pub skill_id: Option<Uuid>,
    /// ID of the entry state (must be a StateType::Start state)
    pub entry_state: Uuid,
    /// IDs of terminal states (must be StateType::Terminal states)
    #[serde(default)]
    pub terminal_states: Vec<Uuid>,
    /// Classification: system (auto-triggered) or business (agent-driven)
    #[serde(default)]
    pub protocol_category: ProtocolCategory,
    /// How this protocol is triggered (manual, event, scheduled, auto)
    #[serde(default)]
    pub trigger_mode: TriggerMode,
    /// Configuration for event/scheduled triggers
    pub trigger_config: Option<TriggerConfig>,
    /// Multi-dimensional relevance profile for context-aware routing (T4)
    pub relevance_vector: Option<RelevanceVector>,
    /// When this protocol was last auto-triggered (for scheduling dedup)
    pub last_triggered_at: Option<DateTime<Utc>>,
    /// When this protocol was created
    pub created_at: DateTime<Utc>,
    /// When this protocol was last modified
    pub updated_at: DateTime<Utc>,
}

impl Protocol {
    /// Create a new Business protocol with minimal fields.
    pub fn new(project_id: Uuid, name: impl Into<String>, entry_state: Uuid) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: String::new(),
            project_id,
            skill_id: None,
            entry_state,
            terminal_states: vec![],
            protocol_category: ProtocolCategory::Business,
            trigger_mode: TriggerMode::Manual,
            trigger_config: None,
            relevance_vector: None,
            last_triggered_at: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a new protocol with full configuration.
    pub fn new_full(
        project_id: Uuid,
        name: impl Into<String>,
        description: impl Into<String>,
        entry_state: Uuid,
        terminal_states: Vec<Uuid>,
        category: ProtocolCategory,
    ) -> Self {
        let mut protocol = Self::new(project_id, name, entry_state);
        protocol.description = description.into();
        protocol.terminal_states = terminal_states;
        protocol.protocol_category = category;
        protocol
    }

    /// Returns true if this is a system-level protocol (auto-triggered).
    pub fn is_system(&self) -> bool {
        self.protocol_category == ProtocolCategory::System
    }

    /// Returns true if this protocol is linked to a skill.
    pub fn has_skill(&self) -> bool {
        self.skill_id.is_some()
    }
}

// ============================================================================
// ProtocolState
// ============================================================================

/// A state within a Protocol FSM.
///
/// Each state represents a phase in the protocol workflow. States can have
/// an optional `action` field describing what should happen when entering
/// this state (e.g., "run tests", "await review").
///
/// # Neo4j Relations
/// ```text
/// (ProtocolState)<-[:HAS_STATE]-(Protocol)    — belongs to protocol
/// (ProtocolState)-[:REFERENCES]->(Note)       — linked knowledge notes
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolState {
    /// Unique identifier
    pub id: Uuid,
    /// Protocol this state belongs to
    pub protocol_id: Uuid,
    /// Human-readable name (e.g., "Awaiting Review", "Tests Passed")
    pub name: String,
    /// Description of this state's purpose
    #[serde(default)]
    pub description: String,
    /// Optional action to execute when entering this state
    /// (e.g., MCP tool call, notification, etc.)
    pub action: Option<String>,
    /// Role of this state in the FSM lifecycle
    #[serde(default)]
    pub state_type: StateType,
}

impl ProtocolState {
    /// Create a new intermediate state.
    pub fn new(protocol_id: Uuid, name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            protocol_id,
            name: name.into(),
            description: String::new(),
            action: None,
            state_type: StateType::Intermediate,
        }
    }

    /// Create a new start state.
    pub fn start(protocol_id: Uuid, name: impl Into<String>) -> Self {
        let mut state = Self::new(protocol_id, name);
        state.state_type = StateType::Start;
        state
    }

    /// Create a new terminal state.
    pub fn terminal(protocol_id: Uuid, name: impl Into<String>) -> Self {
        let mut state = Self::new(protocol_id, name);
        state.state_type = StateType::Terminal;
        state
    }

    /// Returns true if this is the entry state.
    pub fn is_start(&self) -> bool {
        self.state_type == StateType::Start
    }

    /// Returns true if this is a terminal state.
    pub fn is_terminal(&self) -> bool {
        self.state_type == StateType::Terminal
    }
}

// ============================================================================
// ProtocolTransition
// ============================================================================

/// A transition between two states in a Protocol FSM.
///
/// Transitions define the edges of the state machine. Each transition has:
/// - A source state (`from_state`) and target state (`to_state`)
/// - A `trigger` that causes the transition (e.g., "tests_pass", "review_approved")
/// - An optional `guard` condition that must be true for the transition to fire
///
/// # Neo4j Relations
/// ```text
/// (ProtocolTransition)<-[:HAS_TRANSITION]-(Protocol)  — belongs to protocol
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolTransition {
    /// Unique identifier
    pub id: Uuid,
    /// Protocol this transition belongs to
    pub protocol_id: Uuid,
    /// Source state ID
    pub from_state: Uuid,
    /// Target state ID
    pub to_state: Uuid,
    /// Trigger event name (e.g., "tests_pass", "review_approved", "timeout")
    pub trigger: String,
    /// Optional guard condition (e.g., "coverage > 80%", "approvals >= 2")
    pub guard: Option<String>,
}

impl ProtocolTransition {
    /// Create a new transition.
    pub fn new(
        protocol_id: Uuid,
        from_state: Uuid,
        to_state: Uuid,
        trigger: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            protocol_id,
            from_state,
            to_state,
            trigger: trigger.into(),
            guard: None,
        }
    }

    /// Create a new transition with a guard condition.
    pub fn with_guard(
        protocol_id: Uuid,
        from_state: Uuid,
        to_state: Uuid,
        trigger: impl Into<String>,
        guard: impl Into<String>,
    ) -> Self {
        let mut transition = Self::new(protocol_id, from_state, to_state, trigger);
        transition.guard = Some(guard.into());
        transition
    }
}

// ============================================================================
// ProtocolRun (FSM Runtime)
// ============================================================================

/// Status of a protocol execution run.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// The run is actively progressing through states
    #[default]
    Running,
    /// The run reached a terminal state successfully
    Completed,
    /// The run was aborted due to an error
    Failed,
    /// The run was manually cancelled
    Cancelled,
}

impl fmt::Display for RunStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl FromStr for RunStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(format!("Unknown run status: {}", s)),
        }
    }
}

/// A state visit record in a protocol run's history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVisit {
    /// The state that was visited
    pub state_id: Uuid,
    /// The state name (denormalized for convenience)
    pub state_name: String,
    /// When this state was entered
    pub entered_at: DateTime<Utc>,
    /// The trigger that caused entry (None for the initial state)
    pub trigger: Option<String>,
}

/// A protocol run — an instance of a protocol FSM being executed.
///
/// Tracks the current state, history of visited states, and execution metadata.
/// A run progresses through states via transitions triggered by events.
///
/// # Neo4j Relations
/// ```text
/// (ProtocolRun)-[:INSTANCE_OF]->(Protocol)     — the protocol being executed
/// (ProtocolRun)-[:CURRENT_STATE]->(ProtocolState) — current position in FSM
/// (ProtocolRun)-[:LINKED_TO_PLAN]->(Plan)      — optional plan context
/// (ProtocolRun)-[:LINKED_TO_TASK]->(Task)      — optional task context
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolRun {
    /// Unique identifier
    pub id: Uuid,
    /// The protocol being executed
    pub protocol_id: Uuid,
    /// Optional plan context (e.g., wave-execution runs against a plan)
    pub plan_id: Option<Uuid>,
    /// Optional task context
    pub task_id: Option<Uuid>,
    /// Current state in the FSM
    pub current_state: Uuid,
    /// Ordered history of visited states
    #[serde(default)]
    pub states_visited: Vec<StateVisit>,
    /// Execution status
    #[serde(default)]
    pub status: RunStatus,
    /// When the run was started
    pub started_at: DateTime<Utc>,
    /// When the run completed (reached terminal state, failed, or cancelled)
    pub completed_at: Option<DateTime<Utc>>,
    /// Optional error message if status is Failed
    pub error: Option<String>,
    /// How this run was triggered: "manual", "event:post_sync", "schedule:daily", etc.
    #[serde(default = "default_triggered_by")]
    pub triggered_by: String,
}

fn default_triggered_by() -> String {
    "manual".to_string()
}

impl ProtocolRun {
    /// Create a new running protocol execution.
    pub fn new(
        protocol_id: Uuid,
        entry_state_id: Uuid,
        entry_state_name: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            protocol_id,
            plan_id: None,
            task_id: None,
            current_state: entry_state_id,
            states_visited: vec![StateVisit {
                state_id: entry_state_id,
                state_name: entry_state_name.into(),
                entered_at: now,
                trigger: None,
            }],
            status: RunStatus::Running,
            started_at: now,
            completed_at: None,
            error: None,
            triggered_by: "manual".to_string(),
        }
    }

    /// Returns true if the run is still active.
    pub fn is_active(&self) -> bool {
        self.status == RunStatus::Running
    }

    /// Returns true if the run has finished (completed, failed, or cancelled).
    pub fn is_finished(&self) -> bool {
        !self.is_active()
    }

    /// Record a state transition.
    pub fn visit_state(
        &mut self,
        state_id: Uuid,
        state_name: impl Into<String>,
        trigger: impl Into<String>,
    ) {
        self.current_state = state_id;
        self.states_visited.push(StateVisit {
            state_id,
            state_name: state_name.into(),
            entered_at: Utc::now(),
            trigger: Some(trigger.into()),
        });
    }

    /// Mark the run as completed (reached terminal state).
    pub fn complete(&mut self) {
        self.status = RunStatus::Completed;
        self.completed_at = Some(Utc::now());
    }

    /// Mark the run as failed with an error message.
    pub fn fail(&mut self, error: impl Into<String>) {
        self.status = RunStatus::Failed;
        self.completed_at = Some(Utc::now());
        self.error = Some(error.into());
    }

    /// Mark the run as cancelled.
    pub fn cancel(&mut self) {
        self.status = RunStatus::Cancelled;
        self.completed_at = Some(Utc::now());
    }
}

/// Progress report for a long-running protocol state.
///
/// Emitted as a WebSocket event during states that perform multiple
/// sub-actions (e.g., BACKFILL running backfill_synapses, update_energy, etc.).
/// The frontend FSM Viewer can display a progress bar per state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolRunProgress {
    /// The run being reported on
    pub run_id: Uuid,
    /// Current state name (e.g., "BACKFILL")
    pub state_name: String,
    /// Current sub-action being executed (e.g., "backfill_synapses")
    pub sub_action: String,
    /// Number of sub-actions processed so far
    pub processed: usize,
    /// Total number of sub-actions
    pub total: usize,
    /// Human-readable progress display (e.g., "3/7")
    pub display: String,
    /// Milliseconds elapsed since the state was entered
    pub elapsed_ms: u64,
}

impl ProtocolRunProgress {
    /// Create a new progress report.
    pub fn new(
        run_id: Uuid,
        state_name: impl Into<String>,
        sub_action: impl Into<String>,
        processed: usize,
        total: usize,
        elapsed_ms: u64,
    ) -> Self {
        Self {
            run_id,
            state_name: state_name.into(),
            sub_action: sub_action.into(),
            processed,
            total,
            display: format!("{processed}/{total}"),
            elapsed_ms,
        }
    }

    /// Returns a percentage (0-100).
    pub fn percentage(&self) -> u8 {
        if self.total == 0 {
            0
        } else {
            ((self.processed as f64 / self.total as f64) * 100.0).min(100.0) as u8
        }
    }
}

/// Result of a transition attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionResult {
    /// Whether the transition was successful
    pub success: bool,
    /// The new current state (after transition)
    pub current_state: Uuid,
    /// Name of the new current state
    pub current_state_name: String,
    /// Whether the run is now completed (reached terminal state)
    pub run_completed: bool,
    /// The updated run status
    pub status: RunStatus,
    /// Error message if transition failed
    pub error: Option<String>,
}

// ============================================================================
// DTOs
// ============================================================================

/// Request to start a protocol run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartRunRequest {
    /// The protocol to execute
    pub protocol_id: Uuid,
    /// Optional plan context
    pub plan_id: Option<Uuid>,
    /// Optional task context
    pub task_id: Option<Uuid>,
}

/// Request to fire a transition on a running protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireTransitionRequest {
    /// The trigger event to fire
    pub trigger: String,
}

/// Request to create a new protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateProtocolRequest {
    /// Project this protocol belongs to
    pub project_id: Uuid,
    /// Human-readable name
    pub name: String,
    /// Description of the protocol's purpose
    pub description: Option<String>,
    /// Optional link to a Neural Skill
    pub skill_id: Option<Uuid>,
    /// Classification
    pub protocol_category: Option<ProtocolCategory>,
    /// Trigger mode (manual, event, scheduled, auto)
    pub trigger_mode: Option<TriggerMode>,
    /// Trigger configuration (events, schedule, conditions)
    pub trigger_config: Option<TriggerConfig>,
    /// States to create (must include exactly one Start state)
    pub states: Vec<CreateProtocolStateRequest>,
    /// Transitions to create
    pub transitions: Vec<CreateProtocolTransitionRequest>,
}

/// Request to create a protocol state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateProtocolStateRequest {
    /// Human-readable name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Optional action
    pub action: Option<String>,
    /// State type (start, intermediate, terminal)
    pub state_type: StateType,
}

/// Request to create a protocol transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateProtocolTransitionRequest {
    /// Index into the states array for the source state
    pub from_state_index: usize,
    /// Index into the states array for the target state
    pub to_state_index: usize,
    /// Trigger event name
    pub trigger: String,
    /// Optional guard condition
    pub guard: Option<String>,
}

/// Request to update a protocol
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateProtocolRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub skill_id: Option<Option<Uuid>>,
    pub protocol_category: Option<ProtocolCategory>,
    pub trigger_mode: Option<TriggerMode>,
    pub trigger_config: Option<TriggerConfig>,
    pub relevance_vector: Option<RelevanceVector>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- ProtocolCategory tests ---

    #[test]
    fn test_protocol_category_default() {
        assert_eq!(ProtocolCategory::default(), ProtocolCategory::Business);
    }

    #[test]
    fn test_protocol_category_display_and_parse() {
        let categories = vec![
            (ProtocolCategory::System, "system"),
            (ProtocolCategory::Business, "business"),
        ];

        for (cat, expected) in categories {
            assert_eq!(cat.to_string(), expected);
            assert_eq!(ProtocolCategory::from_str(expected).unwrap(), cat);
        }
    }

    #[test]
    fn test_protocol_category_parse_error() {
        assert!(ProtocolCategory::from_str("invalid").is_err());
    }

    #[test]
    fn test_protocol_category_serde_roundtrip() {
        for cat in [ProtocolCategory::System, ProtocolCategory::Business] {
            let json = serde_json::to_string(&cat).unwrap();
            let deserialized: ProtocolCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(cat, deserialized);
        }
    }

    // --- StateType tests ---

    #[test]
    fn test_state_type_default() {
        assert_eq!(StateType::default(), StateType::Intermediate);
    }

    #[test]
    fn test_state_type_display_and_parse() {
        let types = vec![
            (StateType::Start, "start"),
            (StateType::Intermediate, "intermediate"),
            (StateType::Terminal, "terminal"),
        ];

        for (st, expected) in types {
            assert_eq!(st.to_string(), expected);
            assert_eq!(StateType::from_str(expected).unwrap(), st);
        }
    }

    #[test]
    fn test_state_type_parse_error() {
        assert!(StateType::from_str("invalid").is_err());
    }

    #[test]
    fn test_state_type_serde_roundtrip() {
        for st in [
            StateType::Start,
            StateType::Intermediate,
            StateType::Terminal,
        ] {
            let json = serde_json::to_string(&st).unwrap();
            let deserialized: StateType = serde_json::from_str(&json).unwrap();
            assert_eq!(st, deserialized);
        }
    }

    // --- Protocol tests ---

    #[test]
    fn test_protocol_new() {
        let project_id = Uuid::new_v4();
        let entry_state = Uuid::new_v4();
        let protocol = Protocol::new(project_id, "Code Review", entry_state);

        assert_eq!(protocol.project_id, project_id);
        assert_eq!(protocol.name, "Code Review");
        assert_eq!(protocol.entry_state, entry_state);
        assert_eq!(protocol.protocol_category, ProtocolCategory::Business);
        assert!(protocol.description.is_empty());
        assert!(protocol.skill_id.is_none());
        assert!(protocol.terminal_states.is_empty());
        assert!(!protocol.is_system());
        assert!(!protocol.has_skill());
    }

    #[test]
    fn test_protocol_new_full() {
        let project_id = Uuid::new_v4();
        let entry_state = Uuid::new_v4();
        let term1 = Uuid::new_v4();
        let term2 = Uuid::new_v4();

        let protocol = Protocol::new_full(
            project_id,
            "Deployment",
            "Automated deployment protocol",
            entry_state,
            vec![term1, term2],
            ProtocolCategory::System,
        );

        assert_eq!(protocol.name, "Deployment");
        assert_eq!(protocol.description, "Automated deployment protocol");
        assert_eq!(protocol.terminal_states, vec![term1, term2]);
        assert!(protocol.is_system());
    }

    #[test]
    fn test_protocol_has_skill() {
        let mut protocol = Protocol::new(Uuid::new_v4(), "test", Uuid::new_v4());
        assert!(!protocol.has_skill());

        protocol.skill_id = Some(Uuid::new_v4());
        assert!(protocol.has_skill());
    }

    #[test]
    fn test_protocol_serde_roundtrip() {
        let project_id = Uuid::new_v4();
        let entry_state = Uuid::new_v4();
        let mut protocol = Protocol::new_full(
            project_id,
            "Bug Triage",
            "Automated bug triage workflow",
            entry_state,
            vec![Uuid::new_v4()],
            ProtocolCategory::Business,
        );
        protocol.skill_id = Some(Uuid::new_v4());

        let json = serde_json::to_string_pretty(&protocol).unwrap();
        let deserialized: Protocol = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, protocol.id);
        assert_eq!(deserialized.project_id, project_id);
        assert_eq!(deserialized.name, "Bug Triage");
        assert_eq!(deserialized.protocol_category, ProtocolCategory::Business);
        assert_eq!(deserialized.skill_id, protocol.skill_id);
        assert_eq!(deserialized.terminal_states.len(), 1);
    }

    // --- ProtocolState tests ---

    #[test]
    fn test_protocol_state_new() {
        let protocol_id = Uuid::new_v4();
        let state = ProtocolState::new(protocol_id, "Processing");

        assert_eq!(state.protocol_id, protocol_id);
        assert_eq!(state.name, "Processing");
        assert_eq!(state.state_type, StateType::Intermediate);
        assert!(state.description.is_empty());
        assert!(state.action.is_none());
        assert!(!state.is_start());
        assert!(!state.is_terminal());
    }

    #[test]
    fn test_protocol_state_start() {
        let state = ProtocolState::start(Uuid::new_v4(), "Init");
        assert!(state.is_start());
        assert!(!state.is_terminal());
        assert_eq!(state.state_type, StateType::Start);
    }

    #[test]
    fn test_protocol_state_terminal() {
        let state = ProtocolState::terminal(Uuid::new_v4(), "Done");
        assert!(!state.is_start());
        assert!(state.is_terminal());
        assert_eq!(state.state_type, StateType::Terminal);
    }

    #[test]
    fn test_protocol_state_serde_roundtrip() {
        let mut state = ProtocolState::start(Uuid::new_v4(), "Init");
        state.description = "Initial state".into();
        state.action = Some("notify_start".into());

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: ProtocolState = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, state.id);
        assert_eq!(deserialized.name, "Init");
        assert_eq!(deserialized.state_type, StateType::Start);
        assert_eq!(deserialized.action, Some("notify_start".into()));
    }

    // --- ProtocolTransition tests ---

    #[test]
    fn test_protocol_transition_new() {
        let protocol_id = Uuid::new_v4();
        let from = Uuid::new_v4();
        let to = Uuid::new_v4();
        let transition = ProtocolTransition::new(protocol_id, from, to, "tests_pass");

        assert_eq!(transition.protocol_id, protocol_id);
        assert_eq!(transition.from_state, from);
        assert_eq!(transition.to_state, to);
        assert_eq!(transition.trigger, "tests_pass");
        assert!(transition.guard.is_none());
    }

    #[test]
    fn test_protocol_transition_with_guard() {
        let protocol_id = Uuid::new_v4();
        let from = Uuid::new_v4();
        let to = Uuid::new_v4();
        let transition = ProtocolTransition::with_guard(
            protocol_id,
            from,
            to,
            "review_approved",
            "approvals >= 2",
        );

        assert_eq!(transition.trigger, "review_approved");
        assert_eq!(transition.guard, Some("approvals >= 2".into()));
    }

    #[test]
    fn test_protocol_transition_serde_roundtrip() {
        let transition = ProtocolTransition::with_guard(
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            "deploy",
            "env == 'production'",
        );

        let json = serde_json::to_string(&transition).unwrap();
        let deserialized: ProtocolTransition = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, transition.id);
        assert_eq!(deserialized.trigger, "deploy");
        assert_eq!(deserialized.guard, Some("env == 'production'".into()));
    }

    // --- DTO tests ---

    #[test]
    fn test_update_protocol_request_default() {
        let req = UpdateProtocolRequest::default();
        assert!(req.name.is_none());
        assert!(req.description.is_none());
        assert!(req.skill_id.is_none());
        assert!(req.protocol_category.is_none());
    }
}
