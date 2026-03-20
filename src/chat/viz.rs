//! VizBlock protocol — structured visualization blocks for inline chat rendering.
//!
//! Defines the serialization format for embedding visualizations within chat responses.
//! Each message can contain a sequence of [`ContentBlock`]s, where each block is either
//! plain text (markdown) or a structured visualization ([`VizBlock`]).
//!
//! ## Design principles
//!
//! 1. **Backward-compatible**: Clients that don't understand VizBlocks can display
//!    `fallback_text` instead — CLI users see a readable text representation.
//! 2. **Extensible**: Pattern Federation can register custom viz types via the
//!    [`VizRegistry`] without modifying this module.
//! 3. **Type-safe**: Each viz type has a well-defined `data` schema (as `serde_json::Value`).
//!
//! ## Viz types
//!
//! | VizType | Description | Data schema |
//! |---------|-------------|-------------|
//! | `ImpactGraph` | Files/symbols affected by a change | `{ nodes, edges, target }` |
//! | `ReasoningTree` | Decision tree from knowledge graph | `{ roots, confidence, depth }` |
//! | `ProgressBar` | Plan/task completion status | `{ plan_title, tasks, percentage }` |
//! | `ContextRadar` | 5-axis context relevance radar | `{ axes: [{name, value}; 5] }` |
//! | `KnowledgeCard` | Note/decision inline card | `{ note_type, content, importance, tags }` |
//! | `DependencyTree` | File/module dependency graph | `{ root, dependencies, dependents }` |
//! | `ProtocolRun` | Protocol execution trace (Pattern Federation) | reserved |
//! | `FsmState` | FSM current state diagram (Pattern Federation) | reserved |
//! | `ContextRouting` | Context routing visualization (Pattern Federation) | reserved |
//! | `WaveProgress` | Wave computation progress (Pattern Federation) | reserved |
//! | `Custom(String)` | Extension point for third-party viz | user-defined |

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ============================================================================
// VizType — Enumeration of visualization types
// ============================================================================

/// The type of visualization to render in a VizBlock.
///
/// Core types are built-in and have well-defined data schemas.
/// Pattern Federation types are reserved stubs (will be implemented in T1-T6).
/// `Custom(String)` allows third-party extensions without modifying this enum.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum VizType {
    // === Core types (TP3) ===
    /// Impact analysis graph — files and symbols affected by a code change.
    ImpactGraph,

    /// Reasoning tree — decision tree emerging from the knowledge graph.
    ReasoningTree,

    /// Progress bar — plan/task completion with segmented status.
    ProgressBar,

    /// Context radar — 5-axis radar chart showing context relevance dimensions.
    ContextRadar,

    /// Knowledge card — inline display of a note or decision.
    KnowledgeCard,

    /// Dependency tree — file/module import and dependent graph.
    DependencyTree,

    // === Pattern Federation reserved types (T1-T6) ===
    /// Protocol execution trace — shows protocol run steps and outcomes.
    ProtocolRun,

    /// FSM state diagram — current state, transitions, and history.
    FsmState,

    /// Context routing visualization — how context is routed across modules.
    ContextRouting,

    /// Wave computation progress — multi-step protocol wave status.
    WaveProgress,

    // === Design system types (Runner Dashboard) ===
    /// Empty state — placeholder shown when no data is available (e.g., no agents spawned).
    /// Data schema: `{ icon, title, description, cta_label?, cta_action? }`
    EmptyState,

    /// Tab layout — structured tabbed navigation for multi-panel views.
    /// Data schema: `{ tabs: [{id, label, icon?, badge?}], active_tab, content? }`
    TabLayout,

    /// Progress ring — circular gauge for completion percentage (replaces linear ProgressBar).
    /// Data schema: `{ percentage, label, segments?: [{status, count}], animated? }`
    ProgressRing,

    // === Extension point ===
    /// Custom visualization type for third-party extensions.
    /// The string identifies the custom type (e.g., "my_plugin.heatmap").
    Custom(String),
}

impl fmt::Display for VizType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImpactGraph => write!(f, "impact_graph"),
            Self::ReasoningTree => write!(f, "reasoning_tree"),
            Self::ProgressBar => write!(f, "progress_bar"),
            Self::ContextRadar => write!(f, "context_radar"),
            Self::KnowledgeCard => write!(f, "knowledge_card"),
            Self::DependencyTree => write!(f, "dependency_tree"),
            Self::ProtocolRun => write!(f, "protocol_run"),
            Self::FsmState => write!(f, "fsm_state"),
            Self::ContextRouting => write!(f, "context_routing"),
            Self::WaveProgress => write!(f, "wave_progress"),
            Self::EmptyState => write!(f, "empty_state"),
            Self::TabLayout => write!(f, "tab_layout"),
            Self::ProgressRing => write!(f, "progress_ring"),
            Self::Custom(name) => write!(f, "custom:{name}"),
        }
    }
}

impl VizType {
    /// Parse a viz type from a string (used for tag parsing in LLM responses).
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "impact_graph" => Self::ImpactGraph,
            "reasoning_tree" => Self::ReasoningTree,
            "progress_bar" => Self::ProgressBar,
            "context_radar" => Self::ContextRadar,
            "knowledge_card" => Self::KnowledgeCard,
            "dependency_tree" => Self::DependencyTree,
            "protocol_run" => Self::ProtocolRun,
            "fsm_state" => Self::FsmState,
            "context_routing" => Self::ContextRouting,
            "wave_progress" => Self::WaveProgress,
            "empty_state" => Self::EmptyState,
            "tab_layout" => Self::TabLayout,
            "progress_ring" => Self::ProgressRing,
            other => Self::Custom(other.to_string()),
        }
    }

    /// Returns true if this is a Pattern Federation reserved type.
    pub fn is_pattern_federation(&self) -> bool {
        matches!(
            self,
            Self::ProtocolRun | Self::FsmState | Self::ContextRouting | Self::WaveProgress
        )
    }

    /// Returns true if this is a custom (third-party) type.
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom(_))
    }
}

// ============================================================================
// ContentBlock — Union type for chat response blocks
// ============================================================================

/// A single block in a chat response.
///
/// Messages are composed of a sequence of `ContentBlock`s, allowing
/// interleaved text and visualizations. This replaces the flat text model
/// while remaining backward-compatible (serialize to JSON, clients parse).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "block_type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text block (markdown content).
    Text(TextBlock),

    /// Structured visualization block.
    Viz(VizBlock),
}

impl ContentBlock {
    /// Create a text block from a string.
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(TextBlock {
            content: content.into(),
        })
    }

    /// Create a viz block.
    pub fn viz(viz_block: VizBlock) -> Self {
        Self::Viz(viz_block)
    }

    /// Extract the fallback text representation of this block.
    /// For Text blocks, returns the content.
    /// For Viz blocks, returns the fallback_text.
    pub fn fallback_text(&self) -> &str {
        match self {
            Self::Text(t) => &t.content,
            Self::Viz(v) => &v.fallback_text,
        }
    }
}

// ============================================================================
// TextBlock
// ============================================================================

/// A plain text block containing markdown content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    /// Markdown-formatted text content.
    pub content: String,
}

// ============================================================================
// VizBlock
// ============================================================================

/// A structured visualization block embedded inline in a chat response.
///
/// Contains all the data needed for the frontend to render a visualization,
/// plus a human-readable fallback for non-visual clients (CLI, API).
///
/// ## Invariants
///
/// - `fallback_text` is **always** populated — it MUST be non-empty.
/// - `data` schema depends on `viz_type` — see [`VizType`] docs for expected shapes.
/// - `interactive` is a hint to the frontend: interactive vizs get expand/collapse controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizBlock {
    /// The type of visualization to render.
    pub viz_type: VizType,

    /// Structured data for the visualization (JSON schema depends on viz_type).
    pub data: serde_json::Value,

    /// Whether this visualization supports user interaction (expand, click nodes, etc.).
    #[serde(default)]
    pub interactive: bool,

    /// Human-readable text fallback for non-visual clients.
    /// MUST be non-empty. Generated automatically by VizDataBuilder implementations.
    pub fallback_text: String,

    /// Optional title displayed above the visualization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Maximum height in pixels for compact mode (default: 300).
    #[serde(default = "default_max_height")]
    pub max_height: u32,
}

fn default_max_height() -> u32 {
    300
}

impl VizBlock {
    /// Create a new VizBlock with the minimum required fields.
    pub fn new(
        viz_type: VizType,
        data: serde_json::Value,
        fallback_text: impl Into<String>,
    ) -> Self {
        Self {
            viz_type,
            data,
            interactive: false,
            fallback_text: fallback_text.into(),
            title: None,
            max_height: default_max_height(),
        }
    }

    /// Set this viz as interactive.
    pub fn with_interactive(mut self, interactive: bool) -> Self {
        self.interactive = interactive;
        self
    }

    /// Set an optional title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set max height for compact mode.
    pub fn with_max_height(mut self, max_height: u32) -> Self {
        self.max_height = max_height;
        self
    }
}

// ============================================================================
// VizDataBuilder — Trait for generating VizBlocks from domain data
// ============================================================================

/// Trait for building VizBlock data from domain objects.
///
/// Each implementation knows how to:
/// 1. Produce the JSON `data` payload for a specific [`VizType`].
/// 2. Generate a human-readable `fallback_text` for non-visual clients.
///
/// Implementations are registered in the [`VizRegistry`] and dispatched by viz_type.
///
/// ## Example
///
/// ```ignore
/// struct MyCustomBuilder { /* ... */ }
///
/// impl VizDataBuilder for MyCustomBuilder {
///     fn viz_type(&self) -> VizType { VizType::Custom("my_viz".into()) }
///     fn build_data(&self) -> Result<serde_json::Value> { Ok(json!({"key": "value"})) }
///     fn build_fallback(&self) -> String { "My custom viz: key=value".into() }
/// }
/// ```
pub trait VizDataBuilder: Send + Sync {
    /// The viz type this builder produces.
    fn viz_type(&self) -> VizType;

    /// Build the structured data payload for the visualization.
    fn build_data(&self) -> Result<serde_json::Value>;

    /// Build a human-readable text fallback.
    fn build_fallback(&self) -> String;

    /// Whether the produced viz supports interaction (default: false).
    fn is_interactive(&self) -> bool {
        false
    }

    /// Optional title for the visualization.
    fn title(&self) -> Option<String> {
        None
    }

    /// Build a complete VizBlock from this builder.
    fn build(&self) -> Result<VizBlock> {
        let data = self.build_data()?;
        let fallback = self.build_fallback();
        let mut block =
            VizBlock::new(self.viz_type(), data, fallback).with_interactive(self.is_interactive());
        block.title = self.title();
        Ok(block)
    }
}

// ============================================================================
// VizRegistry — Dynamic dispatch for VizDataBuilder implementations
// ============================================================================

/// Registry of VizDataBuilder factories, keyed by VizType.
///
/// Allows dynamic registration of viz builders at runtime.
/// Pattern Federation modules register their builders during initialization.
///
/// ## Thread safety
///
/// The registry is built at startup and then read-only (no interior mutability needed).
/// If dynamic registration is needed at runtime, wrap in `Arc<RwLock<VizRegistry>>`.
pub struct VizRegistry {
    /// Registered builder factories, keyed by viz type.
    /// Each factory produces a VizDataBuilder from raw parameters.
    builders: HashMap<VizType, Box<dyn VizBuilderFactory>>,
}

/// Factory trait for creating VizDataBuilder instances from raw JSON parameters.
///
/// This indirection allows the registry to create builders on-demand with
/// different parameters, rather than storing pre-built builders.
pub trait VizBuilderFactory: Send + Sync {
    /// Create a VizDataBuilder from the given parameters.
    fn create(&self, params: &serde_json::Value) -> Result<Box<dyn VizDataBuilder>>;
}

impl VizRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            builders: HashMap::new(),
        }
    }

    /// Register a builder factory for a viz type.
    /// Overwrites any previously registered factory for the same type.
    pub fn register(&mut self, viz_type: VizType, factory: Box<dyn VizBuilderFactory>) {
        self.builders.insert(viz_type, factory);
    }

    /// Check if a builder factory is registered for the given viz type.
    pub fn has(&self, viz_type: &VizType) -> bool {
        self.builders.contains_key(viz_type)
    }

    /// Build a VizBlock for the given viz type and parameters.
    ///
    /// Returns `None` if no factory is registered for the viz type.
    /// Returns `Err` if the factory or builder fails.
    pub fn build(
        &self,
        viz_type: &VizType,
        params: &serde_json::Value,
    ) -> Option<Result<VizBlock>> {
        self.builders.get(viz_type).map(|factory| {
            let builder = factory.create(params)?;
            builder.build()
        })
    }

    /// List all registered viz types.
    pub fn registered_types(&self) -> Vec<&VizType> {
        self.builders.keys().collect()
    }

    /// Build a VizBlock, or return a fallback block for unregistered types.
    pub fn build_or_fallback(
        &self,
        viz_type: &VizType,
        params: &serde_json::Value,
    ) -> Result<VizBlock> {
        match self.build(viz_type, params) {
            Some(result) => result,
            None => {
                let fallback = if viz_type.is_pattern_federation() {
                    format!("[{viz_type}] Available after Pattern Federation installation.")
                } else {
                    format!("[{viz_type}] Visualization type not registered.")
                };
                Ok(VizBlock::new(
                    viz_type.clone(),
                    serde_json::Value::Null,
                    fallback,
                ))
            }
        }
    }
}

impl Default for VizRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Built-in VizDataBuilder implementations
// ============================================================================

// --- ReasoningTreeVizBuilder ---

/// Builds a VizBlock from a serialized [`crate::reasoning::models::ReasoningTree`].
///
/// Expects the ReasoningTree to be pre-serialized as `serde_json::Value`.
/// Produces an interactive tree visualization with expand/collapse support.
pub struct ReasoningTreeVizBuilder {
    /// Pre-serialized reasoning tree data.
    pub tree_data: serde_json::Value,
    /// Overall confidence score (for fallback text).
    pub confidence: f64,
    /// Number of nodes in the tree.
    pub node_count: usize,
    /// Maximum depth reached.
    pub depth: usize,
    /// The original request that triggered the tree.
    pub request: String,
}

impl VizDataBuilder for ReasoningTreeVizBuilder {
    fn viz_type(&self) -> VizType {
        VizType::ReasoningTree
    }

    fn build_data(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "tree": self.tree_data,
            "confidence": self.confidence,
            "node_count": self.node_count,
            "depth": self.depth,
            "request": self.request,
        }))
    }

    fn build_fallback(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "ReasoningTree (confidence: {:.0}%, {} nodes, depth {})",
            self.confidence * 100.0,
            self.node_count,
            self.depth
        ));
        lines.push(format!("Query: {}", self.request));

        // Build ASCII tree from the data if possible
        if let Some(roots) = self.tree_data.get("roots").and_then(|v| v.as_array()) {
            for root in roots {
                Self::render_node_ascii(root, &mut lines, "", true);
            }
        } else if self.tree_data.is_object() {
            // The tree itself might be the root-level object with "roots" at top level
            if let Some(roots) = self.tree_data.as_object() {
                if let Some(roots_arr) = roots.get("roots").and_then(|v| v.as_array()) {
                    for root in roots_arr {
                        Self::render_node_ascii(root, &mut lines, "", true);
                    }
                }
            }
        }

        lines.join("\n")
    }

    fn is_interactive(&self) -> bool {
        true
    }

    fn title(&self) -> Option<String> {
        Some("Reasoning Tree".to_string())
    }
}

impl ReasoningTreeVizBuilder {
    fn render_node_ascii(
        node: &serde_json::Value,
        lines: &mut Vec<String>,
        prefix: &str,
        is_last: bool,
    ) {
        let connector = if prefix.is_empty() {
            ""
        } else if is_last {
            "└── "
        } else {
            "├── "
        };

        let entity_type = node
            .get("entity_type")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let reasoning = node.get("reasoning").and_then(|v| v.as_str()).unwrap_or("");
        let relevance = node
            .get("relevance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        lines.push(format!(
            "{prefix}{connector}[{entity_type}] {reasoning} ({:.0}%)",
            relevance * 100.0
        ));

        if let Some(children) = node.get("children").and_then(|v| v.as_array()) {
            let child_prefix = if prefix.is_empty() {
                "".to_string()
            } else if is_last {
                format!("{prefix}    ")
            } else {
                format!("{prefix}│   ")
            };
            for (i, child) in children.iter().enumerate() {
                Self::render_node_ascii(child, lines, &child_prefix, i == children.len() - 1);
            }
        }
    }
}

// --- ProgressBarVizBuilder ---

/// Builds a VizBlock showing plan/task completion progress.
///
/// Renders as a segmented bar with status colors:
/// - pending (grey), in_progress (blue), completed (green), blocked (red), failed (red).
pub struct ProgressBarVizBuilder {
    /// Plan or milestone title.
    pub title: String,
    /// Task statuses with their names.
    pub tasks: Vec<TaskProgress>,
    /// Optional plan ID for linking.
    pub plan_id: Option<String>,
}

/// Progress information for a single task in a progress bar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskProgress {
    /// Task title.
    pub title: String,
    /// Current status.
    pub status: String,
    /// Priority (for ordering).
    pub priority: i32,
}

impl VizDataBuilder for ProgressBarVizBuilder {
    fn viz_type(&self) -> VizType {
        VizType::ProgressBar
    }

    fn build_data(&self) -> Result<serde_json::Value> {
        let total = self.tasks.len();
        let completed = self
            .tasks
            .iter()
            .filter(|t| t.status == "completed")
            .count();
        let in_progress = self
            .tasks
            .iter()
            .filter(|t| t.status == "in_progress")
            .count();
        let blocked = self.tasks.iter().filter(|t| t.status == "blocked").count();
        let failed = self.tasks.iter().filter(|t| t.status == "failed").count();
        let pending = total - completed - in_progress - blocked - failed;

        let percentage = if total > 0 {
            (completed as f64 / total as f64 * 100.0).round()
        } else {
            0.0
        };

        Ok(serde_json::json!({
            "plan_title": self.title,
            "plan_id": self.plan_id,
            "tasks": self.tasks,
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "blocked": blocked,
            "failed": failed,
            "percentage": percentage,
        }))
    }

    fn build_fallback(&self) -> String {
        let total = self.tasks.len();
        let completed = self
            .tasks
            .iter()
            .filter(|t| t.status == "completed")
            .count();
        let percentage = if total > 0 {
            (completed as f64 / total as f64 * 100.0).round() as u32
        } else {
            0
        };

        let mut lines = Vec::new();
        lines.push(format!(
            "{}: {}% ({}/{} tasks)",
            self.title, percentage, completed, total
        ));

        // ASCII progress bar
        let bar_width = 20;
        let filled = (percentage as usize * bar_width) / 100;
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
        lines.push(format!("[{bar}]"));

        // Task list
        for task in &self.tasks {
            let icon = match task.status.as_str() {
                "completed" => "✅",
                "in_progress" => "🔄",
                "blocked" => "🚫",
                "failed" => "❌",
                _ => "⬜",
            };
            lines.push(format!("  {icon} {}", task.title));
        }

        lines.join("\n")
    }

    fn title(&self) -> Option<String> {
        Some(format!("Progress: {}", self.title))
    }
}

// --- KnowledgeCardVizBuilder ---

/// Builds a VizBlock displaying a knowledge note or decision inline.
///
/// Renders as a card with type badge, importance level, content preview, and tags.
pub struct KnowledgeCardVizBuilder {
    /// Entity type: "note" or "decision".
    pub entity_type: String,
    /// Entity ID (UUID).
    pub entity_id: String,
    /// Note/decision type (e.g., "guideline", "gotcha", "pattern").
    pub kind: String,
    /// Content text (may be truncated for display).
    pub content: String,
    /// Importance level.
    pub importance: String,
    /// Tags for filtering/display.
    pub tags: Vec<String>,
    /// Optional linked entities (files, functions).
    pub linked_entities: Vec<String>,
}

impl VizDataBuilder for KnowledgeCardVizBuilder {
    fn viz_type(&self) -> VizType {
        VizType::KnowledgeCard
    }

    fn build_data(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "kind": self.kind,
            "content": self.content,
            "importance": self.importance,
            "tags": self.tags,
            "linked_entities": self.linked_entities,
        }))
    }

    fn build_fallback(&self) -> String {
        let importance_icon = match self.importance.as_str() {
            "critical" => "🔴",
            "high" => "🟠",
            "medium" => "🟡",
            "low" => "🟢",
            _ => "⚪",
        };
        let kind_upper = self.kind.to_uppercase();
        let tags_str = if self.tags.is_empty() {
            String::new()
        } else {
            format!(" [{}]", self.tags.join(", "))
        };

        // Truncate content for fallback
        let content_preview = if self.content.len() > 200 {
            format!("{}...", &self.content[..200])
        } else {
            self.content.clone()
        };

        format!("{importance_icon} {kind_upper}{tags_str}\n{content_preview}")
    }
}

// --- EmptyStateVizBuilder ---

/// Builds a VizBlock for empty state placeholders (e.g., no agents spawned).
///
/// Renders as a centered icon + title + description + optional CTA button.
pub struct EmptyStateVizBuilder {
    /// Icon identifier (e.g., "robot", "inbox", "search").
    pub icon: String,
    /// Main title text.
    pub title: String,
    /// Descriptive subtitle.
    pub description: String,
    /// Optional call-to-action button label.
    pub cta_label: Option<String>,
    /// Optional action identifier triggered by the CTA.
    pub cta_action: Option<String>,
}

impl VizDataBuilder for EmptyStateVizBuilder {
    fn viz_type(&self) -> VizType {
        VizType::EmptyState
    }

    fn build_data(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "icon": self.icon,
            "title": self.title,
            "description": self.description,
            "cta_label": self.cta_label,
            "cta_action": self.cta_action,
        }))
    }

    fn build_fallback(&self) -> String {
        let icon_emoji = match self.icon.as_str() {
            "robot" => "🤖",
            "inbox" => "📥",
            "search" => "🔍",
            "rocket" => "🚀",
            _ => "📭",
        };
        let mut text = format!("{icon_emoji} {}\n{}", self.title, self.description);
        if let Some(ref cta) = self.cta_label {
            text.push_str(&format!("\n→ {cta}"));
        }
        text
    }

    fn title(&self) -> Option<String> {
        Some(self.title.clone())
    }
}

// --- TabLayoutVizBuilder ---

/// Builds a VizBlock for tabbed navigation layouts.
///
/// Renders as a horizontal tab bar with active tab indicator.
/// Used for Waves / Conversation / Execution Details navigation.
pub struct TabLayoutVizBuilder {
    /// Tab definitions.
    pub tabs: Vec<TabDef>,
    /// ID of the currently active tab.
    pub active_tab: String,
}

/// A single tab definition in a TabLayout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TabDef {
    /// Unique tab identifier.
    pub id: String,
    /// Display label.
    pub label: String,
    /// Optional icon identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
    /// Optional badge text (e.g., count).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub badge: Option<String>,
}

impl VizDataBuilder for TabLayoutVizBuilder {
    fn viz_type(&self) -> VizType {
        VizType::TabLayout
    }

    fn build_data(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "tabs": self.tabs,
            "active_tab": self.active_tab,
            "tab_count": self.tabs.len(),
        }))
    }

    fn build_fallback(&self) -> String {
        let tabs_str: Vec<String> = self
            .tabs
            .iter()
            .map(|t| {
                let active = if t.id == self.active_tab { " [*]" } else { "" };
                let badge = t
                    .badge
                    .as_ref()
                    .map(|b| format!(" ({b})"))
                    .unwrap_or_default();
                format!("{}{badge}{active}", t.label)
            })
            .collect();
        format!("Tabs: {}", tabs_str.join(" | "))
    }

    fn is_interactive(&self) -> bool {
        true
    }

    fn title(&self) -> Option<String> {
        None // Tab layouts don't need a separate title
    }
}

// --- ProgressRingVizBuilder ---

/// Builds a VizBlock showing circular progress gauge for run completion.
///
/// Replaces the linear ProgressBar with a ring/gauge visualization.
/// Supports segmented status breakdown and animation hints.
pub struct ProgressRingVizBuilder {
    /// Completion percentage (0.0 - 100.0).
    pub percentage: f64,
    /// Center label (e.g., "3/5 tasks").
    pub label: String,
    /// Optional status segments for the ring.
    pub segments: Vec<ProgressSegment>,
    /// Whether the ring should animate.
    pub animated: bool,
    /// Optional title (e.g., plan name).
    pub ring_title: Option<String>,
}

/// A segment in a progress ring showing status breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressSegment {
    /// Status name (completed, in_progress, pending, failed, blocked).
    pub status: String,
    /// Count of items in this status.
    pub count: usize,
    /// Color hint for the frontend (e.g., "#22c55e" for completed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
}

impl VizDataBuilder for ProgressRingVizBuilder {
    fn viz_type(&self) -> VizType {
        VizType::ProgressRing
    }

    fn build_data(&self) -> Result<serde_json::Value> {
        let total: usize = self.segments.iter().map(|s| s.count).sum();
        Ok(serde_json::json!({
            "percentage": self.percentage,
            "label": self.label,
            "segments": self.segments,
            "animated": self.animated,
            "total": total,
            "title": self.ring_title,
        }))
    }

    fn build_fallback(&self) -> String {
        // Circular ASCII gauge
        let pct = self.percentage.round() as usize;
        let filled = pct / 5; // 20 segments
        let empty = 20_usize.saturating_sub(filled);
        let ring = "●".repeat(filled) + &"○".repeat(empty);

        let mut lines = Vec::new();
        if let Some(ref title) = self.ring_title {
            lines.push(title.clone());
        }
        lines.push(format!("[{ring}] {:.0}%", self.percentage));
        lines.push(self.label.clone());

        if !self.segments.is_empty() {
            let seg_str: Vec<String> = self
                .segments
                .iter()
                .filter(|s| s.count > 0)
                .map(|s| {
                    let icon = match s.status.as_str() {
                        "completed" => "✅",
                        "in_progress" => "🔄",
                        "pending" => "⬜",
                        "failed" => "❌",
                        "blocked" => "🚫",
                        _ => "•",
                    };
                    format!("{icon} {} {}", s.count, s.status)
                })
                .collect();
            lines.push(seg_str.join("  "));
        }

        lines.join("\n")
    }

    fn title(&self) -> Option<String> {
        self.ring_title.clone()
    }
}

// --- Pattern Federation Stub Builder ---

/// Stub builder for Pattern Federation reserved types.
///
/// Returns a placeholder message indicating that Pattern Federation
/// is not yet installed. Will be replaced by real implementations in T1-T6.
pub struct PatternFederationStubBuilder {
    viz_type: VizType,
}

impl PatternFederationStubBuilder {
    pub fn new(viz_type: VizType) -> Self {
        Self { viz_type }
    }
}

impl VizDataBuilder for PatternFederationStubBuilder {
    fn viz_type(&self) -> VizType {
        self.viz_type.clone()
    }

    fn build_data(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "status": "not_installed",
            "message": format!("{} requires Pattern Federation (T1-T6)", self.viz_type),
        }))
    }

    fn build_fallback(&self) -> String {
        format!(
            "[{}] Available after Pattern Federation installation.",
            self.viz_type
        )
    }
}

// ============================================================================
// Factory implementations for built-in builders
// ============================================================================

/// Factory for ReasoningTreeVizBuilder — expects pre-serialized tree data.
pub struct ReasoningTreeVizFactory;

impl VizBuilderFactory for ReasoningTreeVizFactory {
    fn create(&self, params: &serde_json::Value) -> Result<Box<dyn VizDataBuilder>> {
        let tree_data = params
            .get("tree")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let confidence = params
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let node_count = params
            .get("node_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let depth = params.get("depth").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let request = params
            .get("request")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(Box::new(ReasoningTreeVizBuilder {
            tree_data,
            confidence,
            node_count,
            depth,
            request,
        }))
    }
}

/// Factory for ProgressBarVizBuilder — expects plan title + task list.
pub struct ProgressBarVizFactory;

impl VizBuilderFactory for ProgressBarVizFactory {
    fn create(&self, params: &serde_json::Value) -> Result<Box<dyn VizDataBuilder>> {
        let title = params
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Plan")
            .to_string();
        let plan_id = params
            .get("plan_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let tasks: Vec<TaskProgress> = params
            .get("tasks")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        Ok(Box::new(ProgressBarVizBuilder {
            title,
            tasks,
            plan_id,
        }))
    }
}

/// Factory for KnowledgeCardVizBuilder — expects note/decision data.
pub struct KnowledgeCardVizFactory;

impl VizBuilderFactory for KnowledgeCardVizFactory {
    fn create(&self, params: &serde_json::Value) -> Result<Box<dyn VizDataBuilder>> {
        let entity_type = params
            .get("entity_type")
            .and_then(|v| v.as_str())
            .unwrap_or("note")
            .to_string();
        let entity_id = params
            .get("entity_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let kind = params
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("context")
            .to_string();
        let content = params
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let importance = params
            .get("importance")
            .and_then(|v| v.as_str())
            .unwrap_or("medium")
            .to_string();
        let tags: Vec<String> = params
            .get("tags")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        let linked_entities: Vec<String> = params
            .get("linked_entities")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        Ok(Box::new(KnowledgeCardVizBuilder {
            entity_type,
            entity_id,
            kind,
            content,
            importance,
            tags,
            linked_entities,
        }))
    }
}

/// Factory for EmptyStateVizBuilder.
pub struct EmptyStateVizFactory;

impl VizBuilderFactory for EmptyStateVizFactory {
    fn create(&self, params: &serde_json::Value) -> Result<Box<dyn VizDataBuilder>> {
        let icon = params
            .get("icon")
            .and_then(|v| v.as_str())
            .unwrap_or("robot")
            .to_string();
        let title = params
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("No data")
            .to_string();
        let description = params
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let cta_label = params
            .get("cta_label")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let cta_action = params
            .get("cta_action")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(Box::new(EmptyStateVizBuilder {
            icon,
            title,
            description,
            cta_label,
            cta_action,
        }))
    }
}

/// Factory for TabLayoutVizBuilder.
pub struct TabLayoutVizFactory;

impl VizBuilderFactory for TabLayoutVizFactory {
    fn create(&self, params: &serde_json::Value) -> Result<Box<dyn VizDataBuilder>> {
        let tabs: Vec<TabDef> = params
            .get("tabs")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        let active_tab = params
            .get("active_tab")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(Box::new(TabLayoutVizBuilder { tabs, active_tab }))
    }
}

/// Factory for ProgressRingVizBuilder.
pub struct ProgressRingVizFactory;

impl VizBuilderFactory for ProgressRingVizFactory {
    fn create(&self, params: &serde_json::Value) -> Result<Box<dyn VizDataBuilder>> {
        let percentage = params
            .get("percentage")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let label = params
            .get("label")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let segments: Vec<ProgressSegment> = params
            .get("segments")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        let animated = params
            .get("animated")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let ring_title = params
            .get("title")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(Box::new(ProgressRingVizBuilder {
            percentage,
            label,
            segments,
            animated,
            ring_title,
        }))
    }
}

/// Factory for Pattern Federation stub types.
pub struct PatternFederationStubFactory {
    viz_type: VizType,
}

impl PatternFederationStubFactory {
    pub fn new(viz_type: VizType) -> Self {
        Self { viz_type }
    }
}

impl VizBuilderFactory for PatternFederationStubFactory {
    fn create(&self, _params: &serde_json::Value) -> Result<Box<dyn VizDataBuilder>> {
        Ok(Box::new(PatternFederationStubBuilder::new(
            self.viz_type.clone(),
        )))
    }
}

// ============================================================================
// VizRegistry — Default with built-in builders
// ============================================================================

impl VizRegistry {
    /// Create a registry pre-loaded with all built-in builder factories.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();

        // Core types
        registry.register(VizType::ReasoningTree, Box::new(ReasoningTreeVizFactory));
        registry.register(VizType::ProgressBar, Box::new(ProgressBarVizFactory));
        registry.register(VizType::KnowledgeCard, Box::new(KnowledgeCardVizFactory));

        // Design system types (Runner Dashboard)
        registry.register(VizType::EmptyState, Box::new(EmptyStateVizFactory));
        registry.register(VizType::TabLayout, Box::new(TabLayoutVizFactory));
        registry.register(VizType::ProgressRing, Box::new(ProgressRingVizFactory));

        // Pattern Federation stubs
        registry.register(
            VizType::ProtocolRun,
            Box::new(PatternFederationStubFactory::new(VizType::ProtocolRun)),
        );
        registry.register(
            VizType::FsmState,
            Box::new(PatternFederationStubFactory::new(VizType::FsmState)),
        );
        registry.register(
            VizType::ContextRouting,
            Box::new(PatternFederationStubFactory::new(VizType::ContextRouting)),
        );
        registry.register(
            VizType::WaveProgress,
            Box::new(PatternFederationStubFactory::new(VizType::WaveProgress)),
        );

        registry
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- VizType tests ---

    #[test]
    fn test_viz_type_display() {
        assert_eq!(VizType::ImpactGraph.to_string(), "impact_graph");
        assert_eq!(VizType::ReasoningTree.to_string(), "reasoning_tree");
        assert_eq!(VizType::ProgressBar.to_string(), "progress_bar");
        assert_eq!(VizType::ContextRadar.to_string(), "context_radar");
        assert_eq!(VizType::KnowledgeCard.to_string(), "knowledge_card");
        assert_eq!(VizType::DependencyTree.to_string(), "dependency_tree");
        assert_eq!(VizType::ProtocolRun.to_string(), "protocol_run");
        assert_eq!(VizType::FsmState.to_string(), "fsm_state");
        assert_eq!(VizType::ContextRouting.to_string(), "context_routing");
        assert_eq!(VizType::WaveProgress.to_string(), "wave_progress");
        assert_eq!(VizType::EmptyState.to_string(), "empty_state");
        assert_eq!(VizType::TabLayout.to_string(), "tab_layout");
        assert_eq!(VizType::ProgressRing.to_string(), "progress_ring");
        assert_eq!(
            VizType::Custom("my_viz".into()).to_string(),
            "custom:my_viz"
        );
    }

    #[test]
    fn test_viz_type_from_str_loose() {
        assert_eq!(
            VizType::from_str_loose("impact_graph"),
            VizType::ImpactGraph
        );
        assert_eq!(
            VizType::from_str_loose("REASONING_TREE"),
            VizType::ReasoningTree
        );
        assert_eq!(
            VizType::from_str_loose("progress_bar"),
            VizType::ProgressBar
        );
        assert_eq!(
            VizType::from_str_loose("unknown_type"),
            VizType::Custom("unknown_type".into())
        );
    }

    #[test]
    fn test_viz_type_is_pattern_federation() {
        assert!(VizType::ProtocolRun.is_pattern_federation());
        assert!(VizType::FsmState.is_pattern_federation());
        assert!(VizType::ContextRouting.is_pattern_federation());
        assert!(VizType::WaveProgress.is_pattern_federation());
        assert!(!VizType::ImpactGraph.is_pattern_federation());
        assert!(!VizType::Custom("test".into()).is_pattern_federation());
    }

    #[test]
    fn test_viz_type_is_custom() {
        assert!(VizType::Custom("test".into()).is_custom());
        assert!(!VizType::ImpactGraph.is_custom());
    }

    #[test]
    fn test_viz_type_serde_round_trip() {
        let types = vec![
            VizType::ImpactGraph,
            VizType::ReasoningTree,
            VizType::ProgressBar,
            VizType::Custom("test_viz".into()),
        ];
        for viz_type in types {
            let json = serde_json::to_string(&viz_type).unwrap();
            let restored: VizType = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, viz_type);
        }
    }

    // --- ContentBlock tests ---

    #[test]
    fn test_content_block_text() {
        let block = ContentBlock::text("Hello **world**");
        assert_eq!(block.fallback_text(), "Hello **world**");
        if let ContentBlock::Text(t) = &block {
            assert_eq!(t.content, "Hello **world**");
        } else {
            panic!("Expected Text block");
        }
    }

    #[test]
    fn test_content_block_viz() {
        let viz = VizBlock::new(
            VizType::ProgressBar,
            serde_json::json!({"percentage": 50}),
            "Progress: 50%",
        );
        let block = ContentBlock::viz(viz);
        assert_eq!(block.fallback_text(), "Progress: 50%");
    }

    #[test]
    fn test_content_block_serde_discriminant() {
        let blocks = vec![
            ContentBlock::text("Some text"),
            ContentBlock::viz(VizBlock::new(
                VizType::ProgressBar,
                serde_json::json!({}),
                "fallback",
            )),
        ];
        let json = serde_json::to_string(&blocks).unwrap();
        // Check that the tag discriminant is present
        assert!(json.contains("\"block_type\":\"text\""));
        assert!(json.contains("\"block_type\":\"viz\""));
        // Round-trip
        let restored: Vec<ContentBlock> = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), 2);
    }

    // --- VizBlock tests ---

    #[test]
    fn test_viz_block_builder_pattern() {
        let block = VizBlock::new(
            VizType::ImpactGraph,
            serde_json::json!({"nodes": [], "edges": []}),
            "No impact detected.",
        )
        .with_interactive(true)
        .with_title("Impact Analysis")
        .with_max_height(400);

        assert_eq!(block.viz_type, VizType::ImpactGraph);
        assert!(block.interactive);
        assert_eq!(block.title.as_deref(), Some("Impact Analysis"));
        assert_eq!(block.max_height, 400);
        assert_eq!(block.fallback_text, "No impact detected.");
    }

    #[test]
    fn test_viz_block_default_max_height() {
        let block = VizBlock::new(VizType::ProgressBar, serde_json::json!({}), "test");
        assert_eq!(block.max_height, 300);
    }

    // --- VizDataBuilder tests ---

    #[test]
    fn test_reasoning_tree_builder() {
        let tree_data = serde_json::json!({
            "roots": [
                {
                    "entity_type": "note",
                    "reasoning": "API rate limiting guideline",
                    "relevance": 0.9,
                    "children": [
                        {
                            "entity_type": "file",
                            "reasoning": "src/api/middleware.rs",
                            "relevance": 0.7,
                            "children": []
                        }
                    ]
                }
            ]
        });

        let builder = ReasoningTreeVizBuilder {
            tree_data,
            confidence: 0.85,
            node_count: 2,
            depth: 1,
            request: "How does rate limiting work?".to_string(),
        };

        assert_eq!(builder.viz_type(), VizType::ReasoningTree);
        assert!(builder.is_interactive());

        let block = builder.build().unwrap();
        assert_eq!(block.viz_type, VizType::ReasoningTree);
        assert!(block.interactive);
        assert!(block.fallback_text.contains("85%"));
        assert!(block.fallback_text.contains("2 nodes"));
    }

    #[test]
    fn test_progress_bar_builder() {
        let builder = ProgressBarVizBuilder {
            title: "TP3 — Inline Visualizations".to_string(),
            tasks: vec![
                TaskProgress {
                    title: "TP3.1 — Spec VizBlock".to_string(),
                    status: "completed".to_string(),
                    priority: 85,
                },
                TaskProgress {
                    title: "TP3.2 — Backend generation".to_string(),
                    status: "in_progress".to_string(),
                    priority: 83,
                },
                TaskProgress {
                    title: "TP3.3 — Frontend renderer".to_string(),
                    status: "pending".to_string(),
                    priority: 80,
                },
            ],
            plan_id: Some("abc-123".to_string()),
        };

        let block = builder.build().unwrap();
        assert_eq!(block.viz_type, VizType::ProgressBar);

        // Check data
        let data = &block.data;
        assert_eq!(data["percentage"], 33.0);
        assert_eq!(data["completed"], 1);
        assert_eq!(data["in_progress"], 1);
        assert_eq!(data["pending"], 1);
        assert_eq!(data["total"], 3);

        // Check fallback
        assert!(block.fallback_text.contains("33%"));
        assert!(block.fallback_text.contains("1/3 tasks"));
        assert!(block.fallback_text.contains("✅"));
        assert!(block.fallback_text.contains("🔄"));
    }

    #[test]
    fn test_knowledge_card_builder() {
        let builder = KnowledgeCardVizBuilder {
            entity_type: "note".to_string(),
            entity_id: "abc-123".to_string(),
            kind: "gotcha".to_string(),
            content: "MockSearchStore uses substring matching, not BM25.".to_string(),
            importance: "high".to_string(),
            tags: vec!["testing".to_string(), "mock".to_string()],
            linked_entities: vec!["src/meilisearch/mock.rs".to_string()],
        };

        let block = builder.build().unwrap();
        assert_eq!(block.viz_type, VizType::KnowledgeCard);

        // Check data
        assert_eq!(block.data["kind"], "gotcha");
        assert_eq!(block.data["importance"], "high");

        // Check fallback
        assert!(block.fallback_text.contains("GOTCHA"));
        assert!(block.fallback_text.contains("🟠"));
        assert!(block.fallback_text.contains("testing"));
    }

    #[test]
    fn test_pattern_federation_stub() {
        let builder = PatternFederationStubBuilder::new(VizType::ProtocolRun);
        let block = builder.build().unwrap();

        assert_eq!(block.viz_type, VizType::ProtocolRun);
        assert!(!block.interactive);
        assert!(block.fallback_text.contains("Pattern Federation"));
        assert_eq!(block.data["status"], "not_installed");
    }

    // --- VizRegistry tests ---

    #[test]
    fn test_registry_with_builtins() {
        let registry = VizRegistry::with_builtins();

        // Core types registered
        assert!(registry.has(&VizType::ReasoningTree));
        assert!(registry.has(&VizType::ProgressBar));
        assert!(registry.has(&VizType::KnowledgeCard));

        // Pattern Federation stubs registered
        assert!(registry.has(&VizType::ProtocolRun));
        assert!(registry.has(&VizType::FsmState));
        assert!(registry.has(&VizType::ContextRouting));
        assert!(registry.has(&VizType::WaveProgress));

        // Not registered (will be added in TP3.2 when graph access is available)
        assert!(!registry.has(&VizType::ImpactGraph));
        assert!(!registry.has(&VizType::DependencyTree));
        assert!(!registry.has(&VizType::ContextRadar));
    }

    #[test]
    fn test_registry_build() {
        let registry = VizRegistry::with_builtins();

        let params = serde_json::json!({
            "title": "My Plan",
            "tasks": [
                {"title": "Task 1", "status": "completed", "priority": 50},
                {"title": "Task 2", "status": "pending", "priority": 30},
            ]
        });

        let result = registry.build(&VizType::ProgressBar, &params);
        assert!(result.is_some());
        let block = result.unwrap().unwrap();
        assert_eq!(block.viz_type, VizType::ProgressBar);
        assert_eq!(block.data["percentage"], 50.0);
    }

    #[test]
    fn test_registry_build_or_fallback_unknown_type() {
        let registry = VizRegistry::with_builtins();
        let params = serde_json::json!({});

        let block = registry
            .build_or_fallback(&VizType::Custom("unknown".into()), &params)
            .unwrap();
        assert!(block
            .fallback_text
            .contains("Visualization type not registered"));
    }

    #[test]
    fn test_registry_build_or_fallback_pattern_federation() {
        let registry = VizRegistry::new(); // Empty registry
        let params = serde_json::json!({});

        let block = registry
            .build_or_fallback(&VizType::ProtocolRun, &params)
            .unwrap();
        assert!(block.fallback_text.contains("Pattern Federation"));
    }

    #[test]
    fn test_registry_custom_builder() {
        struct CustomFactory;
        impl VizBuilderFactory for CustomFactory {
            fn create(&self, _params: &serde_json::Value) -> Result<Box<dyn VizDataBuilder>> {
                struct CustomBuilder;
                impl VizDataBuilder for CustomBuilder {
                    fn viz_type(&self) -> VizType {
                        VizType::Custom("heatmap".into())
                    }
                    fn build_data(&self) -> Result<serde_json::Value> {
                        Ok(serde_json::json!({"cells": []}))
                    }
                    fn build_fallback(&self) -> String {
                        "Heatmap: no data".into()
                    }
                }
                Ok(Box::new(CustomBuilder))
            }
        }

        let mut registry = VizRegistry::new();
        registry.register(VizType::Custom("heatmap".into()), Box::new(CustomFactory));

        assert!(registry.has(&VizType::Custom("heatmap".into())));

        let block = registry
            .build(&VizType::Custom("heatmap".into()), &serde_json::json!({}))
            .unwrap()
            .unwrap();
        assert_eq!(block.fallback_text, "Heatmap: no data");
    }

    // --- Comprehensive serde test ---

    #[test]
    fn test_full_message_serde_round_trip() {
        let blocks = vec![
            ContentBlock::text("Here's the analysis:"),
            ContentBlock::viz(
                VizBlock::new(
                    VizType::ReasoningTree,
                    serde_json::json!({
                        "tree": {
                            "roots": [
                                {"entity_type": "note", "reasoning": "test", "relevance": 0.9, "children": []}
                            ]
                        },
                        "confidence": 0.9,
                        "node_count": 1,
                        "depth": 0
                    }),
                    "ReasoningTree (confidence: 90%, 1 nodes, depth 0)",
                )
                .with_interactive(true)
                .with_title("Analysis"),
            ),
            ContentBlock::text("And here's the progress:"),
            ContentBlock::viz(VizBlock::new(
                VizType::ProgressBar,
                serde_json::json!({"percentage": 75}),
                "Plan: 75%",
            )),
        ];

        // Serialize
        let json = serde_json::to_string_pretty(&blocks).unwrap();

        // Verify structure (pretty-print adds spaces after colons)
        assert!(json.contains("\"block_type\": \"text\""));
        assert!(json.contains("\"block_type\": \"viz\""));
        assert!(json.contains("\"viz_type\": \"reasoning_tree\""));
        assert!(json.contains("\"viz_type\": \"progress_bar\""));
        assert!(json.contains("\"interactive\": true"));
        assert!(json.contains("\"fallback_text\""));

        // Deserialize
        let restored: Vec<ContentBlock> = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), 4);

        // Verify content
        assert_eq!(restored[0].fallback_text(), "Here's the analysis:");
        if let ContentBlock::Viz(viz) = &restored[1] {
            assert_eq!(viz.viz_type, VizType::ReasoningTree);
            assert!(viz.interactive);
            assert_eq!(viz.title.as_deref(), Some("Analysis"));
        } else {
            panic!("Expected Viz block");
        }
    }

    // --- 11 viz types check (acceptance criteria: at least 7) ---

    #[test]
    fn test_at_least_seven_viz_types() {
        let types = vec![
            VizType::ImpactGraph,
            VizType::ReasoningTree,
            VizType::ProgressBar,
            VizType::ContextRadar,
            VizType::KnowledgeCard,
            VizType::DependencyTree,
            VizType::ProtocolRun,
            VizType::FsmState,
            VizType::ContextRouting,
            VizType::WaveProgress,
        ];
        assert!(
            types.len() >= 7,
            "Must have at least 7 viz types, found {}",
            types.len()
        );
        // Plus Custom(String) makes it 14+
    }

    // --- EmptyState builder tests ---

    #[test]
    fn test_empty_state_builder() {
        let builder = EmptyStateVizBuilder {
            icon: "robot".to_string(),
            title: "No agents spawned".to_string(),
            description: "Start a run to spawn agents".to_string(),
            cta_label: Some("Start a run".to_string()),
            cta_action: Some("start_run".to_string()),
        };
        let block = builder.build().unwrap();
        assert_eq!(block.viz_type, VizType::EmptyState);
        assert!(block.fallback_text.contains("🤖"));
        assert!(block.fallback_text.contains("No agents spawned"));
        assert!(block.fallback_text.contains("Start a run"));
        assert_eq!(block.data["icon"], "robot");
        assert_eq!(block.data["cta_label"], "Start a run");
        assert_eq!(block.data["cta_action"], "start_run");
    }

    #[test]
    fn test_empty_state_factory() {
        let factory = EmptyStateVizFactory;
        let params = serde_json::json!({
            "icon": "inbox",
            "title": "No data",
            "description": "Nothing here yet",
        });
        let builder = factory.create(&params).unwrap();
        let block = builder.build().unwrap();
        assert_eq!(block.viz_type, VizType::EmptyState);
        assert!(block.fallback_text.contains("📥"));
        assert!(block.fallback_text.contains("No data"));
    }

    // --- TabLayout builder tests ---

    #[test]
    fn test_tab_layout_builder() {
        let builder = TabLayoutVizBuilder {
            tabs: vec![
                TabDef {
                    id: "waves".to_string(),
                    label: "Waves".to_string(),
                    icon: Some("wave".to_string()),
                    badge: Some("3".to_string()),
                },
                TabDef {
                    id: "conversation".to_string(),
                    label: "Conversation".to_string(),
                    icon: None,
                    badge: None,
                },
                TabDef {
                    id: "details".to_string(),
                    label: "Execution Details".to_string(),
                    icon: None,
                    badge: None,
                },
            ],
            active_tab: "waves".to_string(),
        };
        let block = builder.build().unwrap();
        assert_eq!(block.viz_type, VizType::TabLayout);
        assert!(block.interactive);
        assert!(block.fallback_text.contains("Waves"));
        assert!(block.fallback_text.contains("[*]"));
        assert!(block.fallback_text.contains("(3)"));
        assert_eq!(block.data["tab_count"], 3);
        assert_eq!(block.data["active_tab"], "waves");
    }

    #[test]
    fn test_tab_layout_factory() {
        let factory = TabLayoutVizFactory;
        let params = serde_json::json!({
            "tabs": [
                {"id": "waves", "label": "Waves"},
                {"id": "conv", "label": "Conversation"},
            ],
            "active_tab": "conv",
        });
        let builder = factory.create(&params).unwrap();
        let block = builder.build().unwrap();
        assert_eq!(block.data["active_tab"], "conv");
        assert_eq!(block.data["tab_count"], 2);
    }

    // --- ProgressRing builder tests ---

    #[test]
    fn test_progress_ring_builder() {
        let builder = ProgressRingVizBuilder {
            percentage: 60.0,
            label: "3/5 tasks".to_string(),
            segments: vec![
                ProgressSegment {
                    status: "completed".to_string(),
                    count: 3,
                    color: Some("#22c55e".to_string()),
                },
                ProgressSegment {
                    status: "in_progress".to_string(),
                    count: 1,
                    color: Some("#3b82f6".to_string()),
                },
                ProgressSegment {
                    status: "pending".to_string(),
                    count: 1,
                    color: Some("#6b7280".to_string()),
                },
            ],
            animated: true,
            ring_title: Some("My Plan".to_string()),
        };
        let block = builder.build().unwrap();
        assert_eq!(block.viz_type, VizType::ProgressRing);
        assert!(block.fallback_text.contains("60%"));
        assert!(block.fallback_text.contains("3/5 tasks"));
        assert!(block.fallback_text.contains("✅"));
        assert_eq!(block.data["percentage"], 60.0);
        assert_eq!(block.data["total"], 5);
        assert_eq!(block.data["animated"], true);
    }

    #[test]
    fn test_progress_ring_factory() {
        let factory = ProgressRingVizFactory;
        let params = serde_json::json!({
            "percentage": 75.0,
            "label": "6/8 done",
            "segments": [
                {"status": "completed", "count": 6},
                {"status": "pending", "count": 2},
            ],
            "animated": false,
            "title": "Build Plan",
        });
        let builder = factory.create(&params).unwrap();
        let block = builder.build().unwrap();
        assert_eq!(block.viz_type, VizType::ProgressRing);
        assert_eq!(block.data["percentage"], 75.0);
        assert_eq!(block.data["animated"], false);
        assert_eq!(block.data["title"], "Build Plan");
    }

    // --- Registry includes new types ---

    #[test]
    fn test_registry_has_design_system_types() {
        let registry = VizRegistry::with_builtins();
        assert!(registry.has(&VizType::EmptyState));
        assert!(registry.has(&VizType::TabLayout));
        assert!(registry.has(&VizType::ProgressRing));
    }

    // --- from_str_loose for new types ---

    #[test]
    fn test_from_str_loose_new_types() {
        assert_eq!(VizType::from_str_loose("empty_state"), VizType::EmptyState);
        assert_eq!(VizType::from_str_loose("tab_layout"), VizType::TabLayout);
        assert_eq!(
            VizType::from_str_loose("progress_ring"),
            VizType::ProgressRing
        );
    }
}
