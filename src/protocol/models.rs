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
    /// Generator state that dynamically creates RuntimeStates
    #[serde(rename = "generator")]
    Generator,
}

impl fmt::Display for StateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Start => write!(f, "start"),
            Self::Intermediate => write!(f, "intermediate"),
            Self::Terminal => write!(f, "terminal"),
            Self::Generator => write!(f, "generator"),
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
            "generator" => Ok(Self::Generator),
            _ => Err(format!("Unknown state type: {}", s)),
        }
    }
}

/// Strategy for linking generated RuntimeStates.
///
/// Determines how the dynamically generated states are connected:
/// - **Sequential**: States are chained one after another (A→B→C)
/// - **Parallel**: All states are independent and can run concurrently
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum LinkingStrategy {
    /// States are chained sequentially (A→B→C)
    #[default]
    Sequential,
    /// States run in parallel (no dependencies between them)
    Parallel,
}

impl fmt::Display for LinkingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sequential => write!(f, "sequential"),
            Self::Parallel => write!(f, "parallel"),
        }
    }
}

impl FromStr for LinkingStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sequential" => Ok(Self::Sequential),
            "parallel" => Ok(Self::Parallel),
            _ => Err(format!("Unknown linking strategy: {}", s)),
        }
    }
}

/// Configuration for a Generator state.
///
/// Specifies how RuntimeStates are dynamically generated when entering
/// a Generator state. The `data_source` determines where to fetch items,
/// `state_template` defines the name template, and `linking` controls
/// how the generated states are connected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Data source identifier (e.g., "test", "plan_tasks", "wave_items")
    pub data_source: String,
    /// Name template for generated states (e.g., "Task {index}")
    pub state_template: String,
    /// Optional sub-protocol to spawn for each generated state
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sub_protocol_id: Option<Uuid>,
    /// How generated states are linked together
    #[serde(default)]
    pub linking: LinkingStrategy,
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

/// Strategy for how a parent run handles child run completion in hierarchical FSMs.
///
/// When a protocol state has a `sub_protocol_id`, entering that state spawns
/// a child run. This strategy determines when the parent transitions out:
/// - **AllComplete**: Wait for ALL child runs to complete
/// - **AnyComplete**: Transition as soon as ANY child completes
/// - **Manual**: No auto-transition — requires explicit trigger
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum CompletionStrategy {
    /// Parent transitions when ALL children complete
    #[default]
    AllComplete,
    /// Parent transitions when ANY child completes
    AnyComplete,
    /// No automatic transition — requires explicit trigger
    Manual,
}

impl fmt::Display for CompletionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllComplete => write!(f, "all_complete"),
            Self::AnyComplete => write!(f, "any_complete"),
            Self::Manual => write!(f, "manual"),
        }
    }
}

impl FromStr for CompletionStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "all_complete" | "allcomplete" => Ok(Self::AllComplete),
            "any_complete" | "anycomplete" => Ok(Self::AnyComplete),
            "manual" => Ok(Self::Manual),
            _ => Err(format!("Unknown completion strategy: {}", s)),
        }
    }
}

/// Strategy for handling child run failures in hierarchical FSMs.
///
/// When a child run fails, the parent's macro-state uses this strategy
/// to decide what to do:
/// - **Abort**: Fail the parent run immediately (default)
/// - **Skip**: Fire 'child_skipped' on parent to advance past the macro-state
/// - **Retry**: Re-start the child run up to `max` times, then fall back to Abort
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum OnFailureStrategy {
    /// Fail the parent run immediately
    #[default]
    Abort,
    /// Skip the failed child and fire 'child_skipped' on parent
    Skip,
    /// Retry the child run up to `max` times, then Abort
    Retry {
        /// Maximum number of retry attempts
        max: u8,
    },
}

impl fmt::Display for OnFailureStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Abort => write!(f, "abort"),
            Self::Skip => write!(f, "skip"),
            Self::Retry { max } => write!(f, "retry({})", max),
        }
    }
}

impl FromStr for OnFailureStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "abort" => Ok(Self::Abort),
            "skip" => Ok(Self::Skip),
            s if s.starts_with("retry") => {
                // Parse "retry(N)" or "retry:N" or just "retry" (default max=3)
                let max = s
                    .trim_start_matches("retry")
                    .trim_start_matches(&['(', ':', ' '][..])
                    .trim_end_matches(')')
                    .parse::<u8>()
                    .unwrap_or(3);
                Ok(Self::Retry { max })
            }
            _ => Err(format!("Unknown on_failure strategy: {}", s)),
        }
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
    /// Optional sub-protocol to spawn when entering this state (hierarchical FSM)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sub_protocol_id: Option<Uuid>,
    /// How child run completion triggers parent transition (defaults to AllComplete)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_strategy: Option<CompletionStrategy>,
    /// How child run failure is handled (defaults to Abort)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_failure_strategy: Option<OnFailureStrategy>,
    /// Configuration for Generator states (only used when state_type == Generator)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator_config: Option<GeneratorConfig>,
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
            sub_protocol_id: None,
            completion_strategy: None,
            on_failure_strategy: None,
            generator_config: None,
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
    /// Optional parent run (for hierarchical execution)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<Uuid>,
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
    /// Nesting depth (0 = root, 1 = child of root, etc.)
    #[serde(default)]
    pub depth: u32,
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
            parent_run_id: None,
            depth: 0,
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

/// Info about a child run that completed during this transition.
///
/// Populated when a run reaches a terminal state and has a `parent_run_id`.
/// Used by the handler layer to emit `child_completed` / `child_failed` WebSocket events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChildCompletionInfo {
    /// The parent run that owns this child
    pub parent_run_id: Uuid,
    /// The child run that just completed
    pub child_run_id: Uuid,
    /// The protocol being executed by the child
    pub protocol_id: Uuid,
    /// Terminal status of the child run
    pub status: RunStatus,
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
    /// ID of the child run spawned by hierarchical FSM (if target state had sub_protocol_id)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub child_run_id: Option<Uuid>,
    /// Info about a child run that completed during this transition (for event emission)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub child_completion: Option<ChildCompletionInfo>,
}

// ============================================================================
// RuntimeState (Generator-produced dynamic states)
// ============================================================================

/// A dynamically generated state produced by a Generator state.
///
/// RuntimeStates are created at runtime when a protocol run enters a Generator
/// state. They represent individual work items derived from a data source.
///
/// # Neo4j Relations
/// ```text
/// (ProtocolRun)-[:HAS_RUNTIME_STATE]->(RuntimeState)
/// (RuntimeState)-[:GENERATED_BY]->(ProtocolState)   — the Generator state
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeState {
    /// Unique identifier
    pub id: Uuid,
    /// The protocol run this runtime state belongs to
    pub run_id: Uuid,
    /// The Generator ProtocolState that produced this state
    pub generated_by: Uuid,
    /// Human-readable name (derived from state_template)
    pub name: String,
    /// Position index within the generated sequence (0-based)
    pub index: u32,
    /// Optional sub-protocol to spawn for this runtime state
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sub_protocol_id: Option<Uuid>,
    /// Optional action to execute
    pub action: Option<String>,
    /// Current status of this runtime state
    #[serde(default = "default_runtime_status")]
    pub status: String,
}

fn default_runtime_status() -> String {
    "pending".to_string()
}

impl RuntimeState {
    /// Create a new pending RuntimeState.
    pub fn new(run_id: Uuid, generated_by: Uuid, name: impl Into<String>, index: u32) -> Self {
        Self {
            id: Uuid::new_v4(),
            run_id,
            generated_by,
            name: name.into(),
            index,
            sub_protocol_id: None,
            action: None,
            status: "pending".to_string(),
        }
    }
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
            StateType::Generator,
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

    // --- Generator / LinkingStrategy / RuntimeState tests ---

    #[test]
    fn test_state_type_generator_display_parse() {
        assert_eq!(StateType::Generator.to_string(), "generator");
        assert_eq!(
            StateType::from_str("generator").unwrap(),
            StateType::Generator
        );
        assert_eq!(
            StateType::from_str("Generator").unwrap(),
            StateType::Generator
        );
    }

    #[test]
    fn test_state_type_generator_serde() {
        let json = serde_json::to_string(&StateType::Generator).unwrap();
        assert_eq!(json, "\"generator\"");
        let deserialized: StateType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, StateType::Generator);
    }

    #[test]
    fn test_linking_strategy_default() {
        assert_eq!(LinkingStrategy::default(), LinkingStrategy::Sequential);
    }

    #[test]
    fn test_linking_strategy_display_parse() {
        assert_eq!(LinkingStrategy::Sequential.to_string(), "sequential");
        assert_eq!(LinkingStrategy::Parallel.to_string(), "parallel");
        assert_eq!(
            LinkingStrategy::from_str("sequential").unwrap(),
            LinkingStrategy::Sequential
        );
        assert_eq!(
            LinkingStrategy::from_str("parallel").unwrap(),
            LinkingStrategy::Parallel
        );
        assert!(LinkingStrategy::from_str("invalid").is_err());
    }

    #[test]
    fn test_linking_strategy_serde_roundtrip() {
        for ls in [LinkingStrategy::Sequential, LinkingStrategy::Parallel] {
            let json = serde_json::to_string(&ls).unwrap();
            let deserialized: LinkingStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(ls, deserialized);
        }
    }

    #[test]
    fn test_generator_config_serde_roundtrip() {
        let config = GeneratorConfig {
            data_source: "test".to_string(),
            state_template: "Task {index}".to_string(),
            sub_protocol_id: Some(Uuid::new_v4()),
            linking: LinkingStrategy::Parallel,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: GeneratorConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.data_source, "test");
        assert_eq!(deserialized.state_template, "Task {index}");
        assert_eq!(deserialized.sub_protocol_id, config.sub_protocol_id);
        assert_eq!(deserialized.linking, LinkingStrategy::Parallel);
    }

    #[test]
    fn test_generator_config_defaults() {
        let json = r#"{"data_source":"test","state_template":"Item {index}"}"#;
        let config: GeneratorConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.linking, LinkingStrategy::Sequential);
        assert!(config.sub_protocol_id.is_none());
    }

    #[test]
    fn test_protocol_state_generator_config() {
        let mut state = ProtocolState::new(Uuid::new_v4(), "Generator");
        state.state_type = StateType::Generator;
        state.generator_config = Some(GeneratorConfig {
            data_source: "plan_tasks".to_string(),
            state_template: "Task {index}".to_string(),
            sub_protocol_id: None,
            linking: LinkingStrategy::Sequential,
        });

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: ProtocolState = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.state_type, StateType::Generator);
        assert!(deserialized.generator_config.is_some());
        let gc = deserialized.generator_config.unwrap();
        assert_eq!(gc.data_source, "plan_tasks");
    }

    #[test]
    fn test_protocol_state_without_generator_config_compat() {
        // Existing states without generator_config should deserialize fine
        let state = ProtocolState::start(Uuid::new_v4(), "Start");
        let json = serde_json::to_string(&state).unwrap();
        let deserialized: ProtocolState = serde_json::from_str(&json).unwrap();
        assert!(deserialized.generator_config.is_none());
    }

    #[test]
    fn test_runtime_state_new() {
        let run_id = Uuid::new_v4();
        let generated_by = Uuid::new_v4();
        let rs = RuntimeState::new(run_id, generated_by, "Task 0", 0);

        assert_eq!(rs.run_id, run_id);
        assert_eq!(rs.generated_by, generated_by);
        assert_eq!(rs.name, "Task 0");
        assert_eq!(rs.index, 0);
        assert_eq!(rs.status, "pending");
        assert!(rs.sub_protocol_id.is_none());
        assert!(rs.action.is_none());
    }

    #[test]
    fn test_runtime_state_serde_roundtrip() {
        let mut rs = RuntimeState::new(Uuid::new_v4(), Uuid::new_v4(), "Item 3", 3);
        rs.action = Some("process_item".to_string());
        rs.status = "running".to_string();

        let json = serde_json::to_string(&rs).unwrap();
        let deserialized: RuntimeState = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, rs.id);
        assert_eq!(deserialized.name, "Item 3");
        assert_eq!(deserialized.index, 3);
        assert_eq!(deserialized.status, "running");
        assert_eq!(deserialized.action, Some("process_item".to_string()));
    }
}
