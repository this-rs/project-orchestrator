//! Chat Pre-Enrichment Pipeline
//!
//! Middleware that runs BEFORE each LLM call to enrich the user message
//! with relevant context from the knowledge graph.
//!
//! Architecture:
//! - [`ParallelEnrichmentStage`] trait — pluggable stages returning [`StageOutput`]
//! - [`EnrichmentPipeline`] — runs stages in parallel via `futures::join_all`, merges in order
//! - [`EnrichmentContext`] — accumulator for enrichment data (populated by pipeline merge)
//! - [`EnrichmentInput`] — immutable input data for all stages
//!
//! Stages added by later tasks (TP2.2–TP2.4):
//! - **SkillActivationStage** — matches message against skill trigger patterns
//! - **KnowledgeInjectionStage** — semantic search for notes/decisions
//! - **StatusInjectionStage** — in-progress tasks/plans + reasoning tree

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tracing::{debug, warn};
use uuid::Uuid;

// ============================================================================
// Configuration
// ============================================================================

/// Per-stage enable/disable configuration.
///
/// Defaults to all stages enabled. Can be overridden per-project or globally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentConfig {
    /// Enable the skill auto-activation stage.
    pub skill_activation: bool,
    /// Enable the knowledge injection stage (notes, decisions, propagated).
    pub knowledge_injection: bool,
    /// Enable the status injection stage (tasks, plans in progress).
    pub status_injection: bool,
    /// Enable the biomimicry stage (stagnation detection, homeostasis check).
    #[serde(default)]
    pub biomimicry: bool,
    /// Enable the reflex stage (co-change reminders, episode recall, scar warnings).
    #[serde(default)]
    pub reflex: bool,
    /// Enable the user profile stage (adaptive behavioral profile injection).
    #[serde(default)]
    pub user_profile: bool,
    /// Enable the persona stage (auto-detect relevant persona for mentioned files).
    #[serde(default)]
    pub persona: bool,
    /// Enable the reasoning tree injection stage.
    pub reasoning_tree: bool,
    /// Enable debug mode (logs timing and content of each stage).
    pub debug: bool,
    /// Maximum total time budget for the entire pipeline (ms).
    pub max_pipeline_ms: u64,
}

impl Default for EnrichmentConfig {
    fn default() -> Self {
        Self {
            skill_activation: true,
            knowledge_injection: true,
            status_injection: true,
            biomimicry: false, // Disabled by default — opt-in via ENRICHMENT_BIOMIMICRY=true
            reflex: true,      // Enabled by default — auto-skips when no reflexes match
            user_profile: false, // Disabled by default — opt-in via ENRICHMENT_USER_PROFILE=true
            persona: true,     // Enabled by default — auto-skips when no personas match
            reasoning_tree: true,
            debug: false,
            max_pipeline_ms: 500,
        }
    }
}

impl EnrichmentConfig {
    /// Read enrichment config from environment variables with fallback to defaults.
    ///
    /// Supported variables:
    /// - `ENRICHMENT_SKILL_ACTIVATION` — "false" or "0" to disable
    /// - `ENRICHMENT_KNOWLEDGE_INJECTION` — "false" or "0" to disable
    /// - `ENRICHMENT_STATUS_INJECTION` — "false" or "0" to disable
    /// - `ENRICHMENT_REASONING_TREE` — "false" or "0" to disable
    /// - `ENRICHMENT_DEBUG` — "true" or "1" to enable
    /// - `ENRICHMENT_MAX_PIPELINE_MS` — max pipeline time budget in ms
    pub fn from_env() -> Self {
        Self {
            skill_activation: std::env::var("ENRICHMENT_SKILL_ACTIVATION")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            knowledge_injection: std::env::var("ENRICHMENT_KNOWLEDGE_INJECTION")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            status_injection: std::env::var("ENRICHMENT_STATUS_INJECTION")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            biomimicry: std::env::var("ENRICHMENT_BIOMIMICRY")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            reflex: std::env::var("ENRICHMENT_REFLEX")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            user_profile: std::env::var("ENRICHMENT_USER_PROFILE")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            persona: std::env::var("ENRICHMENT_PERSONA")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            reasoning_tree: std::env::var("ENRICHMENT_REASONING_TREE")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            debug: std::env::var("ENRICHMENT_DEBUG")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            max_pipeline_ms: std::env::var("ENRICHMENT_MAX_PIPELINE_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(500),
        }
    }
}

// ============================================================================
// Input / Context
// ============================================================================

/// Immutable input provided to every enrichment stage.
#[derive(Debug, Clone)]
pub struct EnrichmentInput {
    /// The user's raw message.
    pub message: String,
    /// Chat session ID.
    pub session_id: Uuid,
    /// Project slug (if scoped to a project).
    pub project_slug: Option<String>,
    /// Project UUID (if available).
    pub project_id: Option<Uuid>,
    /// Working directory.
    pub cwd: Option<String>,
    /// Active protocol run ID (if session runs within a protocol FSM context).
    pub protocol_run_id: Option<Uuid>,
    /// Current protocol state name (e.g. "implement", "review").
    pub protocol_state: Option<String>,
    /// Note IDs already included in the system prompt (from ProjectContext guidelines/gotchas).
    /// Used by KnowledgeInjectionStage to avoid duplicating notes that are already present.
    pub excluded_note_ids: HashSet<String>,
    /// Optional reasoning path tracker for Hebbian reinforcement.
    /// When Some, StatusInjectionStage records traversed reasoning tree paths
    /// so they can be reinforced on session close.
    pub reasoning_path_tracker: Option<crate::chat::feedback::ReasoningPathTracker>,
}

/// Mutable context that accumulates enrichment data across stages.
///
/// Each stage can read what previous stages added and add its own data.
/// The context is [`Serialize`] for debug logging.
///
/// **Hints**: stages communicate inter-stage signals via `set_hint`/`get_hint`.
/// SkillActivationStage detects intent via `IntentDetector::detect()`, maps it to a
/// hint string (`"debug"`, `"explore"`, `"review"`, `"planning"`, `"general"`), and
/// sets `hint("intent")`. KnowledgeInjectionStage reads `hint("intent")`, builds an
/// `IntentWeightMap`, multiplies each note's BM25 score by the per-note-type weight,
/// and re-sorts notes by intent-adjusted score before rendering.
#[derive(Debug, Clone, Default, Serialize)]
pub struct EnrichmentContext {
    /// Sections of enriched content to prepend to the prompt.
    pub sections: Vec<EnrichmentSection>,
    /// Inter-stage hints: key-value signals that subsequent stages can read.
    ///
    /// Common hints:
    /// - `intent`: planning | code | debug | review | general
    /// - `scaffolding_level`: 0-4 (from project maturity)
    /// - `detected_files`: comma-separated file paths mentioned in the message
    /// - `detected_functions`: comma-separated function names mentioned
    pub hints: HashMap<String, String>,
    /// Timing information per stage (name, duration in ms).
    pub stage_timings: Vec<(String, u64)>,
    /// Total pipeline execution time in ms.
    pub total_time_ms: u64,
    /// Stages that were skipped (disabled or errored).
    pub skipped_stages: Vec<String>,
}

/// Typed source of an enrichment section, used for deterministic mapping
/// to [`PromptSection`] variants without relying on title string matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EnrichmentSource {
    /// Skill activation stage.
    SkillActivation,
    /// Knowledge injection stage (notes, decisions, propagated).
    KnowledgeInjection,
    /// Status injection stage (tasks, plans in progress).
    StatusInjection,
    /// Persona auto-detection stage.
    Persona,
    /// Reflex stage (co-change, episode recall, scar warnings).
    Reflex,
    /// File context stage (context cards, symbols).
    FileContext,
    /// Biomimicry stage (stagnation detection, homeostasis).
    Biomimicry,
    /// User profile stage (adaptive behavioral profile).
    UserProfile,
    /// Fallback for stages without a specific source type.
    #[default]
    Other,
}

/// A single section of enrichment content.
#[derive(Debug, Clone, Serialize)]
pub struct EnrichmentSection {
    /// Section title (e.g., "Active Skills", "Relevant Notes").
    pub title: String,
    /// Section content (markdown formatted).
    pub content: String,
    /// Source stage name (string identifier for logging/debugging).
    pub source: String,
    /// Typed source for deterministic PromptSection mapping.
    pub enrichment_source: EnrichmentSource,
}

impl EnrichmentContext {
    /// Add a new section of enriched content with a typed source.
    pub fn add_section(
        &mut self,
        title: impl Into<String>,
        content: impl Into<String>,
        source: impl Into<String>,
        enrichment_source: EnrichmentSource,
    ) {
        self.sections.push(EnrichmentSection {
            title: title.into(),
            content: content.into(),
            source: source.into(),
            enrichment_source,
        });
    }

    /// Set an inter-stage hint that subsequent stages can read.
    ///
    /// Common hints:
    /// - `intent`: planning | code | debug | review | general
    /// - `scaffolding_level`: 0-4
    /// - `detected_files`: comma-separated file paths
    pub fn set_hint(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.hints.insert(key.into(), value.into());
    }

    /// Get an inter-stage hint set by a previous stage.
    pub fn get_hint(&self, key: &str) -> Option<&str> {
        self.hints.get(key).map(|s| s.as_str())
    }

    /// Check if any enrichment content was produced.
    pub fn has_content(&self) -> bool {
        !self.sections.is_empty()
    }

    /// Convert enrichment sections to PromptSection variants.
    ///
    /// This bridges the EnrichmentPipeline → PromptBuilder unification:
    /// stages continue producing EnrichmentSections internally, but the
    /// output can be consumed as PromptSections by the FsmPromptComposer.
    ///
    /// Mapping uses the typed [`EnrichmentSource`] enum for deterministic dispatch:
    /// - `SkillActivation` → `PromptSection::SkillContext`
    /// - `KnowledgeInjection` → `PromptSection::KnowledgeNotes` (or `PropagatedNotes` for propagated)
    /// - `FileContext` → `PromptSection::FileContext`
    /// - `Persona` → `PromptSection::PersonaContext`
    /// - Others → `PromptSection::Enrichment`
    pub fn to_prompt_sections(&self) -> Vec<crate::runner::prompt::PromptSection> {
        use crate::runner::prompt::PromptSection;

        self.sections
            .iter()
            .map(|section| match section.enrichment_source {
                EnrichmentSource::SkillActivation => {
                    PromptSection::SkillContext(section.content.clone())
                }
                EnrichmentSource::KnowledgeInjection => {
                    // Distinguish propagated notes by title (sub-category within same source)
                    let title_lower = section.title.to_lowercase();
                    if title_lower.contains("propagated") {
                        PromptSection::PropagatedNotes(section.content.clone())
                    } else {
                        PromptSection::KnowledgeNotes(section.content.clone())
                    }
                }
                EnrichmentSource::FileContext => {
                    PromptSection::FileContext(section.content.clone())
                }
                EnrichmentSource::Persona => PromptSection::PersonaContext(section.content.clone()),
                EnrichmentSource::StatusInjection
                | EnrichmentSource::Reflex
                | EnrichmentSource::Biomimicry
                | EnrichmentSource::UserProfile
                | EnrichmentSource::Other => PromptSection::Enrichment(section.content.clone()),
            })
            .collect()
    }

    /// Render enrichment as a single markdown string for system prompt injection.
    ///
    /// Unlike `render()` which wraps in XML tags for user message prepending,
    /// this produces clean markdown sections for direct system prompt inclusion.
    pub fn to_system_prompt_markdown(&self) -> String {
        if self.sections.is_empty() {
            return String::new();
        }

        let mut output = String::new();
        for section in &self.sections {
            output.push_str(&format!("## {}\n{}\n\n", section.title, section.content));
        }
        output.trim_end().to_string()
    }

    /// Render all sections into a single string for prompt injection.
    ///
    /// Output format:
    /// ```text
    /// <enrichment_context>
    /// ## Section Title
    /// section content...
    ///
    /// ## Another Section
    /// more content...
    ///
    /// </enrichment_context>
    /// ```
    pub fn render(&self) -> String {
        if self.sections.is_empty() {
            return String::new();
        }

        let mut output = String::from("<enrichment_context>\n");
        for section in &self.sections {
            output.push_str(&format!("## {}\n{}\n\n", section.title, section.content));
        }
        output.push_str("</enrichment_context>\n\n");
        output
    }
}

// ============================================================================
// Stage output (parallel pipeline)
// ============================================================================

/// Output produced by a single enrichment stage in the parallel pipeline.
///
/// Instead of mutating an `EnrichmentContext` directly, each stage returns
/// a `StageOutput` which the pipeline merges in registration order.
#[derive(Debug, Clone, Default)]
pub struct StageOutput {
    /// Sections produced by this stage.
    pub sections: Vec<EnrichmentSection>,
    /// Hints to merge into the context (last writer wins across stages).
    pub hints: HashMap<String, String>,
    /// Stage name (for logging/timing).
    pub stage_name: String,
    /// Execution duration in milliseconds.
    pub duration_ms: u64,
}

impl StageOutput {
    /// Create a new empty StageOutput for the given stage name.
    pub fn new(stage_name: impl Into<String>) -> Self {
        Self {
            stage_name: stage_name.into(),
            ..Default::default()
        }
    }

    /// Add a section to this output with a typed source.
    pub fn add_section(
        &mut self,
        title: impl Into<String>,
        content: impl Into<String>,
        source: impl Into<String>,
        enrichment_source: EnrichmentSource,
    ) {
        self.sections.push(EnrichmentSection {
            title: title.into(),
            content: content.into(),
            source: source.into(),
            enrichment_source,
        });
    }

    /// Set a hint in this output.
    pub fn set_hint(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.hints.insert(key.into(), value.into());
    }
}

// ============================================================================
// Stage traits
// ============================================================================

/// A parallel enrichment stage that returns a [`StageOutput`] instead of
/// mutating an `EnrichmentContext`.
///
/// All stages implementing this trait can run concurrently via `futures::join_all`.
/// The pipeline merges outputs in registration order after all stages complete.
#[async_trait::async_trait]
pub trait ParallelEnrichmentStage: Send + Sync {
    /// Execute the stage and return its output.
    async fn execute(&self, input: &EnrichmentInput) -> Result<StageOutput>;

    /// Human-readable name for logging and debugging.
    fn name(&self) -> &str;

    /// Check whether this stage is enabled given the current config.
    fn is_enabled(&self, config: &EnrichmentConfig) -> bool;
}

// ============================================================================
// Pipeline
// ============================================================================

/// Ordered pipeline of enrichment stages.
///
/// Executes all enabled stages **in parallel** via `futures::join_all`,
/// then merges their outputs in registration order. Each stage has an
/// individual timeout; errors and timeouts produce an empty `StageOutput`
/// and are recorded in `skipped_stages`. A global pipeline timeout
/// (`max_pipeline_ms`) wraps the entire parallel execution.
pub struct EnrichmentPipeline {
    stages: Vec<Box<dyn ParallelEnrichmentStage>>,
    config: EnrichmentConfig,
}

impl EnrichmentPipeline {
    /// Create a new pipeline with the given config.
    pub fn new(config: EnrichmentConfig) -> Self {
        Self {
            stages: Vec::new(),
            config,
        }
    }

    /// Add a stage to the pipeline.
    pub fn add_parallel_stage(&mut self, stage: Box<dyn ParallelEnrichmentStage>) {
        self.stages.push(stage);
    }

    /// Execute all enabled stages in parallel.
    ///
    /// Graceful degradation: if a stage fails or times out, its error is logged
    /// and the pipeline continues with the remaining stages' outputs.
    /// Sections are merged in registration order to preserve deterministic output.
    pub async fn execute(&self, input: &EnrichmentInput) -> EnrichmentContext {
        let pipeline_start = Instant::now();
        let mut ctx = EnrichmentContext::default();
        let global_deadline = Duration::from_millis(self.config.max_pipeline_ms);
        // Per-stage timeout: use the global budget (each stage can use the full budget
        // since they run in parallel; the global timeout wraps everything).
        let per_stage_timeout = global_deadline;

        // Collect indices of enabled stages; record disabled ones as skipped.
        let mut enabled_indices = Vec::new();
        for (i, stage) in self.stages.iter().enumerate() {
            if stage.is_enabled(&self.config) {
                enabled_indices.push(i);
            } else {
                ctx.skipped_stages.push(stage.name().to_string());
                if self.config.debug {
                    debug!(
                        "[enrichment] Stage '{}' is disabled, skipping",
                        stage.name()
                    );
                }
            }
        }

        // Build futures for all enabled stages.
        let futures: Vec<_> = enabled_indices
            .iter()
            .map(|&i| {
                let stage = &self.stages[i];
                let stage_name = stage.name().to_string();
                let timeout_dur = per_stage_timeout;
                async move {
                    let stage_start = Instant::now();
                    let result = tokio::time::timeout(timeout_dur, stage.execute(input)).await;
                    let elapsed = stage_start.elapsed().as_millis() as u64;
                    (i, stage_name, result, elapsed)
                }
            })
            .collect();

        // Run all stages in parallel, wrapped in a global timeout.
        let results =
            match tokio::time::timeout(global_deadline, futures::future::join_all(futures)).await {
                Ok(results) => results,
                Err(_) => {
                    warn!(
                        "[enrichment] Pipeline hit global time budget ({}ms) after {}ms",
                        self.config.max_pipeline_ms,
                        pipeline_start.elapsed().as_millis()
                    );
                    // All stages timed out — return empty results
                    Vec::new()
                }
            };

        // Collect results keyed by registration index for ordered merge.
        let mut outputs: std::collections::BTreeMap<usize, StageOutput> =
            std::collections::BTreeMap::new();

        for (idx, stage_name, result, elapsed) in results {
            match result {
                Ok(Ok(mut output)) => {
                    output.duration_ms = elapsed;
                    if self.config.debug {
                        debug!(
                            "[enrichment] Stage '{}' completed in {}ms ({} sections)",
                            stage_name,
                            elapsed,
                            output.sections.len()
                        );
                    }
                    ctx.stage_timings.push((stage_name, elapsed));
                    outputs.insert(idx, output);
                }
                Ok(Err(e)) => {
                    ctx.stage_timings.push((stage_name.clone(), elapsed));
                    ctx.skipped_stages.push(format!("{}(error)", stage_name));
                    warn!(
                        "[enrichment] Stage '{}' failed after {}ms: {} — continuing pipeline",
                        stage_name, elapsed, e
                    );
                }
                Err(_) => {
                    ctx.stage_timings.push((stage_name.clone(), elapsed));
                    ctx.skipped_stages.push(format!("{}(timeout)", stage_name));
                    warn!(
                        "[enrichment] Stage '{}' timed out after {}ms",
                        stage_name, elapsed
                    );
                }
            }
        }

        // Merge outputs in registration order (preserves deterministic section ordering).
        for (_idx, output) in outputs {
            ctx.sections.extend(output.sections);
            // Hints: last writer wins (ordered by registration index).
            ctx.hints.extend(output.hints);
        }

        ctx.total_time_ms = pipeline_start.elapsed().as_millis() as u64;
        if self.config.debug {
            debug!(
                "[enrichment] Pipeline complete: {} sections, {}ms total, timings: {:?}",
                ctx.sections.len(),
                ctx.total_time_ms,
                ctx.stage_timings
            );
        }

        ctx
    }

    /// Get the current config.
    pub fn config(&self) -> &EnrichmentConfig {
        &self.config
    }

    /// Update the config.
    pub fn set_config(&mut self, config: EnrichmentConfig) {
        self.config = config;
    }
}

// ============================================================================
// Helper: enrich a prompt with context
// ============================================================================

/// Enrich a user prompt by prepending the enrichment context.
///
/// If the context has no content, returns the original prompt unchanged.
pub fn enrich_prompt(prompt: &str, ctx: &EnrichmentContext) -> String {
    if !ctx.has_content() {
        return prompt.to_string();
    }
    format!("{}{}", ctx.render(), prompt)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock stage for testing.
    struct MockStage {
        name: String,
        enabled: bool,
        should_fail: bool,
        content: Option<(String, String)>,
    }

    #[async_trait::async_trait]
    impl ParallelEnrichmentStage for MockStage {
        async fn execute(&self, _input: &EnrichmentInput) -> Result<StageOutput> {
            if self.should_fail {
                anyhow::bail!("Mock stage failure");
            }
            let mut output = StageOutput::new(self.name.clone());
            if let Some((title, content)) = &self.content {
                output.add_section(
                    title.clone(),
                    content.clone(),
                    self.name.clone(),
                    EnrichmentSource::Other,
                );
            }
            Ok(output)
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn is_enabled(&self, _config: &EnrichmentConfig) -> bool {
            self.enabled
        }
    }

    fn test_input() -> EnrichmentInput {
        EnrichmentInput {
            message: "test message".to_string(),
            session_id: Uuid::new_v4(),
            project_slug: Some("test-project".to_string()),
            project_id: None,
            cwd: None,
            protocol_run_id: None,
            protocol_state: None,
            excluded_note_ids: Default::default(),
            reasoning_path_tracker: None,
        }
    }

    #[tokio::test]
    async fn test_empty_pipeline() {
        let pipeline = EnrichmentPipeline::new(EnrichmentConfig::default());
        let ctx = pipeline.execute(&test_input()).await;
        assert!(!ctx.has_content());
        assert!(ctx.sections.is_empty());
        assert!(ctx.stage_timings.is_empty());
    }

    #[tokio::test]
    async fn test_pipeline_with_mock_stage() {
        let mut pipeline = EnrichmentPipeline::new(EnrichmentConfig::default());
        pipeline.add_parallel_stage(Box::new(MockStage {
            name: "test_stage".to_string(),
            enabled: true,
            should_fail: false,
            content: Some(("Test Section".to_string(), "test content".to_string())),
        }));

        let ctx = pipeline.execute(&test_input()).await;
        assert!(ctx.has_content());
        assert_eq!(ctx.sections.len(), 1);
        assert_eq!(ctx.sections[0].title, "Test Section");
        assert_eq!(ctx.sections[0].content, "test content");
        assert_eq!(ctx.sections[0].source, "test_stage");
        assert_eq!(ctx.stage_timings.len(), 1);
    }

    #[tokio::test]
    async fn test_pipeline_multiple_stages() {
        let mut pipeline = EnrichmentPipeline::new(EnrichmentConfig::default());
        pipeline.add_parallel_stage(Box::new(MockStage {
            name: "stage_a".to_string(),
            enabled: true,
            should_fail: false,
            content: Some(("Section A".to_string(), "content A".to_string())),
        }));
        pipeline.add_parallel_stage(Box::new(MockStage {
            name: "stage_b".to_string(),
            enabled: true,
            should_fail: false,
            content: Some(("Section B".to_string(), "content B".to_string())),
        }));

        let ctx = pipeline.execute(&test_input()).await;
        assert_eq!(ctx.sections.len(), 2);
        assert_eq!(ctx.sections[0].title, "Section A");
        assert_eq!(ctx.sections[1].title, "Section B");
        assert_eq!(ctx.stage_timings.len(), 2);
    }

    #[tokio::test]
    async fn test_pipeline_graceful_degradation() {
        let mut pipeline = EnrichmentPipeline::new(EnrichmentConfig::default());

        // Stage 1: fails
        pipeline.add_parallel_stage(Box::new(MockStage {
            name: "failing_stage".to_string(),
            enabled: true,
            should_fail: true,
            content: None,
        }));

        // Stage 2: succeeds (should still execute despite Stage 1 failure)
        pipeline.add_parallel_stage(Box::new(MockStage {
            name: "success_stage".to_string(),
            enabled: true,
            should_fail: false,
            content: Some(("After Failure".to_string(), "still works".to_string())),
        }));

        let ctx = pipeline.execute(&test_input()).await;
        assert!(ctx.has_content());
        assert_eq!(ctx.sections.len(), 1);
        assert_eq!(ctx.sections[0].title, "After Failure");
        assert!(ctx
            .skipped_stages
            .contains(&"failing_stage(error)".to_string()));
        // Both stages should have timing entries
        assert_eq!(ctx.stage_timings.len(), 2);
    }

    #[tokio::test]
    async fn test_pipeline_disabled_stage() {
        let mut pipeline = EnrichmentPipeline::new(EnrichmentConfig::default());
        pipeline.add_parallel_stage(Box::new(MockStage {
            name: "disabled_stage".to_string(),
            enabled: false,
            should_fail: false,
            content: Some(("Should Not Appear".to_string(), "nope".to_string())),
        }));

        let ctx = pipeline.execute(&test_input()).await;
        assert!(!ctx.has_content());
        assert!(ctx.skipped_stages.contains(&"disabled_stage".to_string()));
    }

    #[tokio::test]
    async fn test_enrichment_context_render() {
        let mut ctx = EnrichmentContext::default();
        ctx.add_section(
            "Notes",
            "- Note 1\n- Note 2",
            "knowledge",
            EnrichmentSource::KnowledgeInjection,
        );
        ctx.add_section(
            "Skills",
            "- Skill A",
            "skills",
            EnrichmentSource::SkillActivation,
        );

        let rendered = ctx.render();
        assert!(rendered.contains("<enrichment_context>"));
        assert!(rendered.contains("## Notes"));
        assert!(rendered.contains("- Note 1"));
        assert!(rendered.contains("## Skills"));
        assert!(rendered.contains("</enrichment_context>"));
    }

    #[tokio::test]
    async fn test_enrichment_context_render_empty() {
        let ctx = EnrichmentContext::default();
        assert_eq!(ctx.render(), "");
    }

    #[tokio::test]
    async fn test_enrich_prompt_with_content() {
        let mut ctx = EnrichmentContext::default();
        ctx.add_section("Context", "some context", "test", EnrichmentSource::Other);

        let enriched = enrich_prompt("original message", &ctx);
        assert!(enriched.starts_with("<enrichment_context>"));
        assert!(enriched.ends_with("original message"));
    }

    #[tokio::test]
    async fn test_enrich_prompt_no_content() {
        let ctx = EnrichmentContext::default();
        let enriched = enrich_prompt("original message", &ctx);
        assert_eq!(enriched, "original message");
    }

    #[tokio::test]
    async fn test_enrichment_context_serializable() {
        let mut ctx = EnrichmentContext::default();
        ctx.add_section("Test", "content", "source", EnrichmentSource::Other);
        ctx.stage_timings.push(("test".to_string(), 42));
        ctx.total_time_ms = 42;

        let json = serde_json::to_string(&ctx).unwrap();
        assert!(json.contains("\"total_time_ms\":42"));
        assert!(json.contains("\"title\":\"Test\""));
    }

    #[tokio::test]
    async fn test_pipeline_time_budget() {
        use tokio::time::sleep;

        /// A slow stage that sleeps for a while.
        struct SlowStage;

        #[async_trait::async_trait]
        impl ParallelEnrichmentStage for SlowStage {
            async fn execute(&self, _input: &EnrichmentInput) -> Result<StageOutput> {
                sleep(Duration::from_millis(600)).await;
                let mut output = StageOutput::new("slow_stage");
                output.add_section("Slow", "should not appear", "slow", EnrichmentSource::Other);
                Ok(output)
            }

            fn name(&self) -> &str {
                "slow_stage"
            }

            fn is_enabled(&self, _config: &EnrichmentConfig) -> bool {
                true
            }
        }

        let config = EnrichmentConfig {
            max_pipeline_ms: 50, // very tight budget
            ..Default::default()
        };
        let mut pipeline = EnrichmentPipeline::new(config);

        // Fast stage first
        pipeline.add_parallel_stage(Box::new(MockStage {
            name: "fast_stage".to_string(),
            enabled: true,
            should_fail: false,
            content: Some(("Fast".to_string(), "quick result".to_string())),
        }));

        // Slow stage — should either execute (within budget) or be skipped
        pipeline.add_parallel_stage(Box::new(SlowStage));

        let ctx = pipeline.execute(&test_input()).await;
        // At minimum, the fast stage should have completed
        assert!(ctx.sections.iter().any(|s| s.title == "Fast"));
    }

    #[test]
    fn test_enrichment_config_default() {
        let config = EnrichmentConfig::default();
        assert!(config.skill_activation);
        assert!(config.knowledge_injection);
        assert!(config.status_injection);
        assert!(config.reasoning_tree);
        assert!(!config.debug);
        assert_eq!(config.max_pipeline_ms, 500);
    }

    #[test]
    fn test_enrichment_config_from_env() {
        // Clear any existing vars first
        std::env::remove_var("ENRICHMENT_SKILL_ACTIVATION");
        std::env::remove_var("ENRICHMENT_KNOWLEDGE_INJECTION");
        std::env::remove_var("ENRICHMENT_STATUS_INJECTION");
        std::env::remove_var("ENRICHMENT_REASONING_TREE");
        std::env::remove_var("ENRICHMENT_DEBUG");
        std::env::remove_var("ENRICHMENT_MAX_PIPELINE_MS");

        // Defaults
        let config = EnrichmentConfig::from_env();
        assert!(config.skill_activation);
        assert!(config.knowledge_injection);
        assert!(config.status_injection);
        assert!(config.reasoning_tree);
        assert!(!config.debug);
        assert_eq!(config.max_pipeline_ms, 500);

        // Custom values
        std::env::set_var("ENRICHMENT_SKILL_ACTIVATION", "false");
        std::env::set_var("ENRICHMENT_KNOWLEDGE_INJECTION", "0");
        std::env::set_var("ENRICHMENT_DEBUG", "true");
        std::env::set_var("ENRICHMENT_MAX_PIPELINE_MS", "250");
        let config = EnrichmentConfig::from_env();
        assert!(!config.skill_activation);
        assert!(!config.knowledge_injection);
        assert!(config.status_injection); // not overridden
        assert!(config.debug);
        assert_eq!(config.max_pipeline_ms, 250);

        // Cleanup
        std::env::remove_var("ENRICHMENT_SKILL_ACTIVATION");
        std::env::remove_var("ENRICHMENT_KNOWLEDGE_INJECTION");
        std::env::remove_var("ENRICHMENT_STATUS_INJECTION");
        std::env::remove_var("ENRICHMENT_REASONING_TREE");
        std::env::remove_var("ENRICHMENT_DEBUG");
        std::env::remove_var("ENRICHMENT_MAX_PIPELINE_MS");
    }

    #[test]
    fn test_enrichment_config_deserialize() {
        let json = r#"{
            "skill_activation": true,
            "knowledge_injection": false,
            "status_injection": true,
            "reasoning_tree": false,
            "debug": true,
            "max_pipeline_ms": 300
        }"#;
        let config: EnrichmentConfig = serde_json::from_str(json).unwrap();
        assert!(config.skill_activation);
        assert!(!config.knowledge_injection);
        assert!(config.status_injection);
        assert!(!config.reasoning_tree);
        assert!(config.debug);
        assert_eq!(config.max_pipeline_ms, 300);
    }

    #[test]
    fn test_enrichment_config_roundtrip() {
        let config = EnrichmentConfig {
            skill_activation: false,
            knowledge_injection: true,
            status_injection: false,
            biomimicry: false,
            reflex: false,
            user_profile: false,
            persona: false,
            reasoning_tree: true,
            debug: true,
            max_pipeline_ms: 1000,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: EnrichmentConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.skill_activation, config.skill_activation);
        assert_eq!(deserialized.knowledge_injection, config.knowledge_injection);
        assert_eq!(deserialized.debug, config.debug);
        assert_eq!(deserialized.max_pipeline_ms, config.max_pipeline_ms);
    }

    // ── PromptSection bridge tests ──────────────────────────────────

    #[tokio::test]
    async fn test_to_prompt_sections_mapping() {
        use crate::runner::prompt::PromptSection;

        let mut ctx = EnrichmentContext::default();
        ctx.add_section(
            "Activated Skills",
            "skill content",
            "skill_stage",
            EnrichmentSource::SkillActivation,
        );
        ctx.add_section(
            "Relevant Notes",
            "note content",
            "knowledge_stage",
            EnrichmentSource::KnowledgeInjection,
        );
        ctx.add_section(
            "Propagated Notes",
            "propagated content",
            "knowledge_stage",
            EnrichmentSource::KnowledgeInjection,
        );
        ctx.add_section(
            "File Context",
            "file content",
            "file_stage",
            EnrichmentSource::FileContext,
        );
        ctx.add_section(
            "Persona Context",
            "persona content",
            "persona_stage",
            EnrichmentSource::Persona,
        );
        ctx.add_section(
            "Active Work",
            "task content",
            "status_stage",
            EnrichmentSource::StatusInjection,
        );

        let sections = ctx.to_prompt_sections();
        assert_eq!(sections.len(), 6, "Should produce 6 PromptSections");

        // Check correct mapping
        assert!(matches!(&sections[0], PromptSection::SkillContext(_)));
        assert!(matches!(&sections[1], PromptSection::KnowledgeNotes(_)));
        assert!(matches!(&sections[2], PromptSection::PropagatedNotes(_)));
        assert!(matches!(&sections[3], PromptSection::FileContext(_)));
        assert!(matches!(&sections[4], PromptSection::PersonaContext(_)));
        assert!(matches!(&sections[5], PromptSection::Enrichment(_)));
    }

    #[tokio::test]
    async fn test_to_prompt_sections_all_enrichment_variants() {
        use crate::runner::prompt::PromptSection;

        // Verify Biomimicry, Reflex, UserProfile, and Other all map to Enrichment
        let mut ctx = EnrichmentContext::default();
        ctx.add_section("Bio", "bio", "bio", EnrichmentSource::Biomimicry);
        ctx.add_section("Reflex", "reflex", "reflex", EnrichmentSource::Reflex);
        ctx.add_section(
            "Profile",
            "profile",
            "profile",
            EnrichmentSource::UserProfile,
        );
        ctx.add_section("Other", "other", "other", EnrichmentSource::Other);

        let sections = ctx.to_prompt_sections();
        assert_eq!(sections.len(), 4);
        for (i, section) in sections.iter().enumerate() {
            assert!(
                matches!(section, PromptSection::Enrichment(_)),
                "Section {} ({:?}) should map to Enrichment",
                i,
                section
            );
        }
    }

    #[tokio::test]
    async fn test_to_system_prompt_markdown() {
        let mut ctx = EnrichmentContext::default();
        ctx.add_section(
            "Skills",
            "- Skill A",
            "test",
            EnrichmentSource::SkillActivation,
        );
        ctx.add_section(
            "Notes",
            "- Note 1",
            "test",
            EnrichmentSource::KnowledgeInjection,
        );

        let md = ctx.to_system_prompt_markdown();
        assert!(md.contains("## Skills"));
        assert!(md.contains("- Skill A"));
        assert!(md.contains("## Notes"));
        assert!(md.contains("- Note 1"));
        assert!(!md.contains("<enrichment_context>"), "No XML wrapper");
    }

    #[tokio::test]
    async fn test_to_system_prompt_markdown_empty() {
        let ctx = EnrichmentContext::default();
        assert_eq!(ctx.to_system_prompt_markdown(), "");
    }

    // ── Parallel pipeline benchmark ───────────────────────────────────

    /// A mock stage that sleeps to simulate I/O latency.
    struct LatentMockStage {
        stage_name: String,
        latency_ms: u64,
    }

    #[async_trait::async_trait]
    impl ParallelEnrichmentStage for LatentMockStage {
        async fn execute(&self, _input: &EnrichmentInput) -> Result<StageOutput> {
            tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;
            let mut output = StageOutput::new(self.stage_name.clone());
            output.add_section(
                format!("Section from {}", self.stage_name),
                "content",
                self.stage_name.clone(),
                EnrichmentSource::Other,
            );
            Ok(output)
        }

        fn name(&self) -> &str {
            &self.stage_name
        }

        fn is_enabled(&self, _config: &EnrichmentConfig) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn test_parallel_pipeline_faster_than_sequential() {
        // 8 stages × 50ms = 400ms sequential, but parallel should be < 150ms
        let mut pipeline = EnrichmentPipeline::new(EnrichmentConfig {
            max_pipeline_ms: 1000,
            ..Default::default()
        });
        for i in 0..8 {
            pipeline.add_parallel_stage(Box::new(LatentMockStage {
                stage_name: format!("latent_{}", i),
                latency_ms: 50,
            }));
        }

        let start = Instant::now();
        let ctx = pipeline.execute(&test_input()).await;
        let elapsed = start.elapsed().as_millis();

        // All 8 stages should have produced sections
        assert_eq!(
            ctx.sections.len(),
            8,
            "All 8 stages should produce sections"
        );
        // Sections should be in registration order
        assert_eq!(ctx.sections[0].source, "latent_0");
        assert_eq!(ctx.sections[7].source, "latent_7");

        // Parallel: should complete in ~50-80ms, definitely < 150ms
        assert!(
            elapsed < 150,
            "Parallel pipeline with 8×50ms stages should complete in < 150ms, got {}ms",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_stage_output_helpers() {
        let mut output = StageOutput::new("test");
        assert_eq!(output.stage_name, "test");
        assert!(output.sections.is_empty());
        assert!(output.hints.is_empty());

        output.add_section("Title", "Content", "source", EnrichmentSource::Other);
        assert_eq!(output.sections.len(), 1);
        assert_eq!(output.sections[0].title, "Title");

        output.set_hint("key", "value");
        assert_eq!(output.hints.get("key").unwrap(), "value");
    }
}
