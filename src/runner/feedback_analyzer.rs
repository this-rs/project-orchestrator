//! Runner Feedback Analyzer — transforms collected feedback patterns into protocol proposals.
//!
//! Implements the full feedback loop:
//! 1. **Aggregation**: `analyze_runner_feedback` groups observation notes by pattern type
//! 2. **Detection**: Identifies recurring patterns (>3 threshold) and flags them as actionable
//! 3. **Proposal**: Generates RFC notes with proposed protocol adjustments
//! 4. **Application**: On RFC acceptance, composes new protocols and adds them to routing pool
//! 5. **Trust**: Emerging protocols get a trust score based on evidence strength

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;
use crate::notes::models::{Note, NoteFilters, NoteImportance, NoteStatus, NoteType};

// ============================================================================
// Constants
// ============================================================================

/// Minimum number of runs with the same pattern to flag as actionable.
const PATTERN_THRESHOLD: usize = 3;

/// Trust score below which a protocol is marked "emerging" and requires explicit validation.
const EMERGING_TRUST_THRESHOLD: f64 = 0.5;

/// Maximum trust score for a newly composed protocol from feedback.
const INITIAL_TRUST_SCORE: f64 = 0.3;

/// Trust increment per positive run on an emerging protocol.
const TRUST_INCREMENT_PER_RUN: f64 = 0.07;

// ============================================================================
// Pattern types
// ============================================================================

/// Categories of feedback patterns detected from runner observations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackPatternType {
    /// Commits made manually after a light run → suggest upgrade to full
    ManualCommitPostRun,
    /// PR step skipped in full runs → suggest downgrade to light
    SkipPrInFullRun,
    /// Same state overridden repeatedly → suggest variant without that state
    RepeatedStateOverride { state_name: String },
    /// Chat dissatisfaction about missing steps
    MissingStepComplaint,
    /// Custom transition fired repeatedly → formalize it
    RepeatedCustomTransition { trigger_name: String },
}

impl std::fmt::Display for FeedbackPatternType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ManualCommitPostRun => write!(f, "manual_commit_post_run"),
            Self::SkipPrInFullRun => write!(f, "skip_pr_in_full_run"),
            Self::RepeatedStateOverride { state_name } => {
                write!(f, "repeated_state_override:{}", state_name)
            }
            Self::MissingStepComplaint => write!(f, "missing_step_complaint"),
            Self::RepeatedCustomTransition { trigger_name } => {
                write!(f, "repeated_custom_transition:{}", trigger_name)
            }
        }
    }
}

// ============================================================================
// Pattern detection result
// ============================================================================

/// A detected feedback pattern with supporting evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Type of pattern detected
    pub pattern_type: FeedbackPatternType,
    /// Number of occurrences found
    pub count: usize,
    /// Note IDs that constitute the evidence
    pub evidence_note_ids: Vec<Uuid>,
    /// Run IDs where this pattern was observed
    pub run_ids: Vec<Uuid>,
    /// Whether this pattern crosses the actionable threshold
    pub is_actionable: bool,
    /// Suggested action based on the pattern
    pub suggestion: PatternSuggestion,
}

/// What the system suggests doing about a detected pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternSuggestion {
    /// Suggest upgrading from light to full protocol
    UpgradeToFull,
    /// Suggest downgrading from full to light protocol
    DowngradeToLight,
    /// Suggest creating a variant protocol without a specific state
    CreateVariantWithout { state_name: String },
    /// Suggest formalizing a custom transition into the protocol
    FormalizeTransition { trigger_name: String },
    /// Suggest adding a missing step to the protocol
    AddMissingStep,
    /// Not enough data yet — keep collecting
    KeepCollecting,
}

impl std::fmt::Display for PatternSuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UpgradeToFull => write!(f, "suggest_upgrade_to_full"),
            Self::DowngradeToLight => write!(f, "suggest_downgrade_to_light"),
            Self::CreateVariantWithout { state_name } => {
                write!(f, "create_variant_without:{}", state_name)
            }
            Self::FormalizeTransition { trigger_name } => {
                write!(f, "formalize_transition:{}", trigger_name)
            }
            Self::AddMissingStep => write!(f, "add_missing_step"),
            Self::KeepCollecting => write!(f, "keep_collecting"),
        }
    }
}

// ============================================================================
// Feedback report
// ============================================================================

/// Structured report from `analyze_runner_feedback`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackReport {
    /// Total number of feedback notes analyzed
    pub total_notes_analyzed: usize,
    /// Patterns grouped by type with counts and examples
    pub patterns: Vec<DetectedPattern>,
    /// Number of actionable patterns (count > threshold)
    pub actionable_count: usize,
    /// Timestamp of the analysis
    pub analyzed_at: String,
}

// ============================================================================
// Trust score for emerging protocols
// ============================================================================

/// Trust metadata for a protocol composed from feedback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolTrust {
    /// Trust score in [0, 1]
    pub score: f64,
    /// Number of source episodes that generated the feedback
    pub source_episode_count: usize,
    /// Number of positive runs after composition
    pub positive_runs: usize,
    /// Number of negative runs after composition
    pub negative_runs: usize,
    /// Status derived from trust score
    pub status: TrustStatus,
}

/// Trust-based status for emerging protocols.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustStatus {
    /// Trust < 0.5 — requires explicit validation before routing
    Emerging,
    /// Trust >= 0.5 — can be routed automatically
    Active,
}

impl ProtocolTrust {
    /// Create initial trust for a newly composed protocol.
    pub fn new(source_episode_count: usize) -> Self {
        // More source episodes = higher initial trust (capped at 0.45)
        let base = INITIAL_TRUST_SCORE + (source_episode_count as f64 * 0.02).min(0.15);
        let status = if base < EMERGING_TRUST_THRESHOLD {
            TrustStatus::Emerging
        } else {
            TrustStatus::Active
        };
        Self {
            score: base,
            source_episode_count,
            positive_runs: 0,
            negative_runs: 0,
            status,
        }
    }

    /// Record a positive run outcome and update trust.
    pub fn record_positive_run(&mut self) {
        self.positive_runs += 1;
        self.score = (self.score + TRUST_INCREMENT_PER_RUN).min(1.0);
        self.update_status();
    }

    /// Record a negative run outcome and update trust.
    pub fn record_negative_run(&mut self) {
        self.negative_runs += 1;
        self.score = (self.score - TRUST_INCREMENT_PER_RUN * 1.5).max(0.0);
        self.update_status();
    }

    fn update_status(&mut self) {
        self.status = if self.score < EMERGING_TRUST_THRESHOLD {
            TrustStatus::Emerging
        } else {
            TrustStatus::Active
        };
    }
}

// ============================================================================
// FeedbackAnalyzer
// ============================================================================

/// Analyzes runner feedback notes and detects actionable patterns.
///
/// This is the core engine of the feedback→protocol loop:
/// 1. Reads all runner-feedback observation notes
/// 2. Classifies them by pattern type
/// 3. Detects recurring patterns above threshold
/// 4. Generates RFC proposals for actionable patterns
/// 5. On acceptance, composes new protocols
pub struct FeedbackAnalyzer {
    graph: Arc<dyn GraphStore>,
}

impl FeedbackAnalyzer {
    /// Create a new feedback analyzer.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }

    // ========================================================================
    // Step 1: Aggregate feedback notes
    // ========================================================================

    /// Analyze all runner-feedback observation notes and produce a structured report.
    ///
    /// This is the entry point called by `admin(action: "analyze_runner_feedback")`.
    /// It fetches all notes tagged `runner-feedback`, classifies them by pattern type,
    /// and returns frequency counts with examples.
    pub async fn analyze_runner_feedback(
        &self,
        project_id: Option<Uuid>,
    ) -> Result<FeedbackReport> {
        // Fetch all runner-feedback tagged notes
        let filters = NoteFilters {
            note_type: Some(vec![NoteType::Observation]),
            tags: Some(vec!["runner-feedback".to_string()]),
            limit: Some(500),
            ..Default::default()
        };

        let (notes, _total) = self.graph.list_notes(project_id, None, &filters).await?;

        info!(
            project_id = ?project_id,
            notes_found = notes.len(),
            "Analyzing runner feedback notes"
        );

        // Classify notes into pattern buckets
        let mut pattern_buckets: HashMap<String, Vec<(Uuid, Option<Uuid>)>> = HashMap::new();

        for note in &notes {
            let (pattern_key, _run_id) = self.classify_note(note);
            let run_id = self.extract_run_id_from_content(&note.content);
            pattern_buckets
                .entry(pattern_key)
                .or_default()
                .push((note.id, run_id));
        }

        // Build detected patterns
        let patterns = self.build_patterns(pattern_buckets);
        let actionable_count = patterns.iter().filter(|p| p.is_actionable).count();

        let report = FeedbackReport {
            total_notes_analyzed: notes.len(),
            patterns,
            actionable_count,
            analyzed_at: Utc::now().to_rfc3339(),
        };

        info!(
            total_notes = report.total_notes_analyzed,
            patterns_found = report.patterns.len(),
            actionable = report.actionable_count,
            "Runner feedback analysis complete"
        );

        Ok(report)
    }

    // ========================================================================
    // Step 2: Pattern detection rules
    // ========================================================================

    /// Classify a feedback note into a pattern category based on its tags and content.
    fn classify_note(&self, note: &Note) -> (String, Option<String>) {
        // Check specific tags first
        if note.tags.contains(&"runner-manual-action".to_string()) {
            return ("manual_commit_post_run".to_string(), None);
        }

        if note.tags.contains(&"runner-override-skip".to_string()) {
            // Extract the skipped state name from content
            let state = self.extract_field(&note.content, "Skipped state");
            let key = format!("state_override:{}", state.as_deref().unwrap_or("unknown"));
            return (key, state);
        }

        if note
            .tags
            .contains(&"runner-override-transition".to_string())
        {
            let trigger = self.extract_field(&note.content, "Custom trigger");
            let key = format!(
                "custom_transition:{}",
                trigger.as_deref().unwrap_or("unknown")
            );
            return (key, trigger);
        }

        if note.tags.contains(&"runner-override-step".to_string()) {
            return ("missing_step".to_string(), None);
        }

        if note.tags.contains(&"runner-chat-feedback".to_string()) {
            return ("chat_dissatisfaction".to_string(), None);
        }

        if note.tags.contains(&"runner-episode".to_string()) {
            return ("episode".to_string(), None);
        }

        // Fallback: generic feedback
        ("other".to_string(), None)
    }

    /// Build DetectedPattern entries from classified buckets.
    fn build_patterns(
        &self,
        buckets: HashMap<String, Vec<(Uuid, Option<Uuid>)>>,
    ) -> Vec<DetectedPattern> {
        let mut patterns = Vec::new();

        for (key, entries) in &buckets {
            let count = entries.len();
            let evidence_note_ids: Vec<Uuid> = entries.iter().map(|(nid, _)| *nid).collect();
            let run_ids: Vec<Uuid> = entries.iter().filter_map(|(_, rid)| *rid).collect();
            let is_actionable = count > PATTERN_THRESHOLD;

            let (pattern_type, suggestion) = self.map_key_to_pattern(key, count);

            patterns.push(DetectedPattern {
                pattern_type,
                count,
                evidence_note_ids,
                run_ids,
                is_actionable,
                suggestion,
            });
        }

        // Sort by count descending
        patterns.sort_by(|a, b| b.count.cmp(&a.count));
        patterns
    }

    /// Map a classification key to a typed pattern and suggestion.
    fn map_key_to_pattern(
        &self,
        key: &str,
        count: usize,
    ) -> (FeedbackPatternType, PatternSuggestion) {
        if count <= PATTERN_THRESHOLD {
            // Below threshold — keep collecting regardless of type
            let pt = self.key_to_pattern_type(key);
            return (pt, PatternSuggestion::KeepCollecting);
        }

        match key {
            "manual_commit_post_run" => (
                FeedbackPatternType::ManualCommitPostRun,
                PatternSuggestion::UpgradeToFull,
            ),
            "chat_dissatisfaction" | "missing_step" => (
                FeedbackPatternType::MissingStepComplaint,
                PatternSuggestion::AddMissingStep,
            ),
            k if k.starts_with("state_override:") => {
                let state_name = k.trim_start_matches("state_override:").to_string();
                (
                    FeedbackPatternType::RepeatedStateOverride {
                        state_name: state_name.clone(),
                    },
                    PatternSuggestion::CreateVariantWithout { state_name },
                )
            }
            k if k.starts_with("custom_transition:") => {
                let trigger_name = k.trim_start_matches("custom_transition:").to_string();
                (
                    FeedbackPatternType::RepeatedCustomTransition {
                        trigger_name: trigger_name.clone(),
                    },
                    PatternSuggestion::FormalizeTransition { trigger_name },
                )
            }
            _ => (
                self.key_to_pattern_type(key),
                PatternSuggestion::KeepCollecting,
            ),
        }
    }

    /// Convert a raw key string to a FeedbackPatternType.
    fn key_to_pattern_type(&self, key: &str) -> FeedbackPatternType {
        match key {
            "manual_commit_post_run" => FeedbackPatternType::ManualCommitPostRun,
            "chat_dissatisfaction" | "missing_step" => FeedbackPatternType::MissingStepComplaint,
            k if k.starts_with("state_override:") => {
                let state_name = k.trim_start_matches("state_override:").to_string();
                FeedbackPatternType::RepeatedStateOverride { state_name }
            }
            k if k.starts_with("custom_transition:") => {
                let trigger_name = k.trim_start_matches("custom_transition:").to_string();
                FeedbackPatternType::RepeatedCustomTransition { trigger_name }
            }
            _ => FeedbackPatternType::MissingStepComplaint,
        }
    }

    // ========================================================================
    // Step 3: Generate RFC proposals
    // ========================================================================

    /// For each actionable pattern, generate an RFC note with the proposed protocol.
    ///
    /// The RFC contains:
    /// - Problem description
    /// - Proposed protocol (states + transitions as JSON)
    /// - Source evidence (episode IDs)
    /// - Status: "proposed" (awaiting user validation)
    ///
    /// Returns the IDs of created RFC notes.
    pub async fn generate_rfc_proposals(
        &self,
        project_id: Uuid,
        report: &FeedbackReport,
    ) -> Result<Vec<Uuid>> {
        let mut rfc_ids = Vec::new();

        for pattern in &report.patterns {
            if !pattern.is_actionable {
                continue;
            }

            let rfc_content = self.build_rfc_content(pattern);
            let mut rfc = Note::new(
                Some(project_id),
                NoteType::Rfc,
                rfc_content,
                format!("RFC: {}", pattern.suggestion),
            );
            rfc.importance = NoteImportance::High;
            rfc.status = NoteStatus::Active;
            rfc.tags = vec![
                "runner-feedback-rfc".to_string(),
                "proposed".to_string(),
                "auto-generated".to_string(),
                format!("pattern:{}", pattern.pattern_type),
            ];

            let rfc_id = rfc.id;
            self.graph.create_note(&rfc).await?;

            info!(
                rfc_id = %rfc_id,
                pattern = %pattern.pattern_type,
                suggestion = %pattern.suggestion,
                evidence_count = pattern.count,
                "Generated RFC proposal from feedback pattern"
            );

            rfc_ids.push(rfc_id);
        }

        Ok(rfc_ids)
    }

    /// Build the markdown content for an RFC note.
    fn build_rfc_content(&self, pattern: &DetectedPattern) -> String {
        let protocol_json = self.generate_protocol_spec(pattern);

        format!(
            "## RFC: Protocol Adjustment from Runner Feedback\n\n\
             ### Problem Detected\n\
             **Pattern**: {} (observed {} times, threshold: {})\n\
             **Suggestion**: {}\n\n\
             ### Evidence\n\
             - **Source notes**: {}\n\
             - **Run IDs**: {}\n\n\
             ### Proposed Protocol\n\
             ```json\n{}\n```\n\n\
             ### Rationale\n\
             {}\n\n\
             ### Status\n\
             **Proposed** — awaiting user validation. \
             Accept this RFC to automatically compose and activate the protocol.\n",
            pattern.pattern_type,
            pattern.count,
            PATTERN_THRESHOLD,
            pattern.suggestion,
            pattern
                .evidence_note_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            pattern
                .run_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            protocol_json,
            self.generate_rationale(pattern),
        )
    }

    /// Generate a protocol specification JSON for the proposed change.
    fn generate_protocol_spec(&self, pattern: &DetectedPattern) -> String {
        match &pattern.suggestion {
            PatternSuggestion::UpgradeToFull => serde_json::to_string_pretty(
                &serde_json::json!({
                    "name": "plan-runner-full-upgraded",
                    "description": "Full lifecycle with commit+PR, upgraded from light based on feedback",
                    "states": [
                        {"name": "init", "type": "start"},
                        {"name": "routing", "type": "intermediate"},
                        {"name": "executing", "type": "intermediate"},
                        {"name": "verifying", "type": "intermediate"},
                        {"name": "committing", "type": "intermediate"},
                        {"name": "creating_pr", "type": "intermediate"},
                        {"name": "collecting_feedback", "type": "intermediate"},
                        {"name": "completed", "type": "terminal"},
                        {"name": "failed", "type": "terminal"}
                    ],
                    "transitions": [
                        {"from": "init", "to": "routing", "trigger": "start"},
                        {"from": "routing", "to": "executing", "trigger": "protocol_selected"},
                        {"from": "executing", "to": "verifying", "trigger": "all_tasks_done"},
                        {"from": "verifying", "to": "committing", "trigger": "verification_passed"},
                        {"from": "committing", "to": "creating_pr", "trigger": "commit_created"},
                        {"from": "creating_pr", "to": "collecting_feedback", "trigger": "pr_created"},
                        {"from": "collecting_feedback", "to": "completed", "trigger": "feedback_collected"},
                        {"from": "verifying", "to": "failed", "trigger": "verification_failed"},
                        {"from": "executing", "to": "failed", "trigger": "execution_failed"}
                    ],
                    "source": "feedback_analysis",
                    "evidence_count": pattern.count
                }),
            )
            .unwrap_or_default(),

            PatternSuggestion::DowngradeToLight => serde_json::to_string_pretty(
                &serde_json::json!({
                    "name": "plan-runner-light-downgraded",
                    "description": "Light lifecycle without PR step, downgraded from full based on feedback",
                    "states": [
                        {"name": "init", "type": "start"},
                        {"name": "routing", "type": "intermediate"},
                        {"name": "executing", "type": "intermediate"},
                        {"name": "verifying", "type": "intermediate"},
                        {"name": "collecting_feedback", "type": "intermediate"},
                        {"name": "completed", "type": "terminal"},
                        {"name": "failed", "type": "terminal"}
                    ],
                    "transitions": [
                        {"from": "init", "to": "routing", "trigger": "start"},
                        {"from": "routing", "to": "executing", "trigger": "protocol_selected"},
                        {"from": "executing", "to": "verifying", "trigger": "all_tasks_done"},
                        {"from": "verifying", "to": "collecting_feedback", "trigger": "verification_passed"},
                        {"from": "collecting_feedback", "to": "completed", "trigger": "feedback_collected"},
                        {"from": "verifying", "to": "failed", "trigger": "verification_failed"}
                    ],
                    "source": "feedback_analysis",
                    "evidence_count": pattern.count
                }),
            )
            .unwrap_or_default(),

            PatternSuggestion::CreateVariantWithout { state_name } => {
                serde_json::to_string_pretty(&serde_json::json!({
                    "name": format!("plan-runner-no-{}", state_name),
                    "description": format!("Variant protocol without '{}' state, based on repeated overrides", state_name),
                    "removed_state": state_name,
                    "states": [
                        {"name": "init", "type": "start"},
                        {"name": "routing", "type": "intermediate"},
                        {"name": "executing", "type": "intermediate"},
                        {"name": "verifying", "type": "intermediate"},
                        {"name": "completed", "type": "terminal"},
                        {"name": "failed", "type": "terminal"}
                    ],
                    "transitions": [
                        {"from": "init", "to": "routing", "trigger": "start"},
                        {"from": "routing", "to": "executing", "trigger": "protocol_selected"},
                        {"from": "executing", "to": "verifying", "trigger": "all_tasks_done"},
                        {"from": "verifying", "to": "completed", "trigger": "verification_passed"},
                        {"from": "verifying", "to": "failed", "trigger": "verification_failed"}
                    ],
                    "source": "feedback_analysis",
                    "evidence_count": pattern.count
                }))
                .unwrap_or_default()
            }

            PatternSuggestion::FormalizeTransition { trigger_name } => {
                serde_json::to_string_pretty(&serde_json::json!({
                    "name": format!("plan-runner-with-{}", trigger_name),
                    "description": format!("Variant with formalized '{}' transition", trigger_name),
                    "added_transition": trigger_name,
                    "source": "feedback_analysis",
                    "evidence_count": pattern.count
                }))
                .unwrap_or_default()
            }

            PatternSuggestion::AddMissingStep => serde_json::to_string_pretty(
                &serde_json::json!({
                    "name": "plan-runner-extended",
                    "description": "Extended protocol with additional steps based on user complaints",
                    "source": "feedback_analysis",
                    "evidence_count": pattern.count,
                    "note": "Requires manual review to determine which steps to add"
                }),
            )
            .unwrap_or_default(),

            PatternSuggestion::KeepCollecting => String::from("{}"),
        }
    }

    /// Generate a human-readable rationale for the proposed change.
    fn generate_rationale(&self, pattern: &DetectedPattern) -> String {
        match &pattern.suggestion {
            PatternSuggestion::UpgradeToFull => format!(
                "Over {} runs, users manually created commits after light protocol runs completed. \
                 This indicates the light protocol is insufficient — upgrading to full will \
                 automate the commit+PR steps that users are doing manually.",
                pattern.count
            ),
            PatternSuggestion::DowngradeToLight => format!(
                "Over {} runs, users skipped the PR creation step in full protocol runs. \
                 This suggests the PR step adds unnecessary overhead for this workflow — \
                 downgrading to light removes it.",
                pattern.count
            ),
            PatternSuggestion::CreateVariantWithout { state_name } => format!(
                "Over {} runs, users overrode the '{}' state. \
                 Creating a variant without this state will streamline the workflow.",
                pattern.count, state_name
            ),
            PatternSuggestion::FormalizeTransition { trigger_name } => format!(
                "Over {} runs, users fired the custom '{}' transition. \
                 Formalizing it makes the workflow explicit and predictable.",
                pattern.count, trigger_name
            ),
            PatternSuggestion::AddMissingStep => format!(
                "Over {} instances of user complaints about missing steps. \
                 The protocol should be extended to cover the gaps users are filling manually.",
                pattern.count
            ),
            PatternSuggestion::KeepCollecting => {
                "Not enough data yet. Continue collecting feedback.".to_string()
            }
        }
    }

    // ========================================================================
    // Step 4: Apply accepted RFC → compose protocol
    // ========================================================================

    /// Handle an accepted RFC: extract the protocol spec and compose it.
    ///
    /// Called when a user accepts an RFC via `advance_rfc(trigger: accept)`.
    /// Returns the ID of the newly composed protocol.
    pub async fn apply_accepted_rfc(
        &self,
        rfc_note_id: Uuid,
        project_id: Uuid,
    ) -> Result<Option<AppliedProtocol>> {
        // Fetch the RFC note
        let rfc = self.graph.get_note(rfc_note_id).await?;
        let rfc = match rfc {
            Some(n) => n,
            None => {
                warn!(rfc_id = %rfc_note_id, "RFC note not found");
                return Ok(None);
            }
        };

        // Verify it's an RFC with the right tags
        if rfc.note_type != NoteType::Rfc || !rfc.tags.contains(&"runner-feedback-rfc".to_string())
        {
            warn!(rfc_id = %rfc_note_id, "Note is not a runner-feedback RFC");
            return Ok(None);
        }

        // Extract the protocol JSON from the RFC content
        let protocol_spec = self.extract_protocol_json(&rfc.content)?;

        // Extract evidence count for trust scoring
        let evidence_count = protocol_spec
            .get("evidence_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let proto_name = protocol_spec
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("feedback-protocol")
            .to_string();

        let proto_description = protocol_spec
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Compute initial trust
        let trust = ProtocolTrust::new(evidence_count);

        info!(
            rfc_id = %rfc_note_id,
            protocol_name = %proto_name,
            trust_score = trust.score,
            trust_status = ?trust.status,
            "Applying accepted RFC — composing new protocol"
        );

        // Mark RFC as accepted
        self.graph
            .update_note(
                rfc_note_id,
                None,
                None,
                Some(NoteStatus::Archived),
                Some(vec![
                    "runner-feedback-rfc".to_string(),
                    "accepted".to_string(),
                    "auto-generated".to_string(),
                ]),
                None,
            )
            .await?;

        Ok(Some(AppliedProtocol {
            name: proto_name,
            description: proto_description,
            project_id,
            protocol_spec,
            trust,
            rfc_note_id,
        }))
    }

    // ========================================================================
    // Step 5: Trust scoring
    // ========================================================================

    /// Compute trust score for an emerging protocol based on run outcomes.
    ///
    /// Trust starts low and increases with successful runs:
    /// - Initial trust = 0.3 + (source_episodes × 0.02, max 0.15)
    /// - Each positive run: +0.07
    /// - Each negative run: -0.105
    /// - Status transitions at threshold 0.5
    pub fn compute_trust(
        source_episode_count: usize,
        positive_runs: usize,
        negative_runs: usize,
    ) -> ProtocolTrust {
        let mut trust = ProtocolTrust::new(source_episode_count);
        for _ in 0..positive_runs {
            trust.record_positive_run();
        }
        for _ in 0..negative_runs {
            trust.record_negative_run();
        }
        trust
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Extract a field value from markdown content (e.g., "**Skipped state**: review")
    fn extract_field(&self, content: &str, field_name: &str) -> Option<String> {
        let pattern = format!("**{}**: ", field_name);
        content.find(&pattern).and_then(|pos| {
            let start = pos + pattern.len();
            let rest = &content[start..];
            let end = rest.find('\n').unwrap_or(rest.len());
            let value = rest[..end].trim().to_string();
            if value.is_empty() {
                None
            } else {
                Some(value)
            }
        })
    }

    /// Extract a Run ID UUID from note content.
    fn extract_run_id_from_content(&self, content: &str) -> Option<Uuid> {
        let pattern = "**Run ID**: ";
        content.find(pattern).and_then(|pos| {
            let start = pos + pattern.len();
            let rest = &content[start..];
            let end = rest.find('\n').unwrap_or(rest.len());
            let value = rest[..end].trim();
            Uuid::parse_str(value).ok()
        })
    }

    /// Extract protocol JSON from RFC content (between ```json and ```).
    fn extract_protocol_json(&self, content: &str) -> Result<serde_json::Value> {
        let start_marker = "```json\n";
        let end_marker = "\n```";

        let start = content
            .find(start_marker)
            .map(|p| p + start_marker.len())
            .ok_or_else(|| anyhow::anyhow!("No JSON block found in RFC content"))?;

        let rest = &content[start..];
        let end = rest
            .find(end_marker)
            .ok_or_else(|| anyhow::anyhow!("Unclosed JSON block in RFC content"))?;

        let json_str = &rest[..end];
        let value: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Invalid JSON in RFC: {}", e))?;

        Ok(value)
    }
}

/// Result of applying an accepted RFC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedProtocol {
    /// Protocol name from the RFC spec
    pub name: String,
    /// Protocol description
    pub description: String,
    /// Project this protocol belongs to
    pub project_id: Uuid,
    /// Full protocol specification JSON
    pub protocol_spec: serde_json::Value,
    /// Trust metadata for the new protocol
    pub trust: ProtocolTrust,
    /// RFC note ID that was accepted
    pub rfc_note_id: Uuid,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use tracing::debug;

    // ========================================================================
    // Helper: create feedback notes in mock
    // ========================================================================

    async fn create_feedback_notes(
        mock: &MockGraphStore,
        project_id: Uuid,
        tag: &str,
        count: usize,
        content_template: &str,
    ) -> Vec<Uuid> {
        let mut ids = Vec::new();
        for i in 0..count {
            let run_id = Uuid::new_v4();
            let content = content_template.replace("{run_id}", &run_id.to_string());
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                content,
                "runner-feedback".to_string(),
            );
            note.tags = vec![
                "runner-feedback".to_string(),
                tag.to_string(),
                "auto-generated".to_string(),
            ];
            note.importance = NoteImportance::High;
            let note_id = note.id;
            mock.notes.write().await.insert(note_id, note);
            ids.push(note_id);
            debug!("Created test note {} ({}/{})", note_id, i + 1, count);
        }
        ids
    }

    // ========================================================================
    // Step 1 tests: analyze_runner_feedback
    // ========================================================================

    #[tokio::test]
    async fn test_analyze_empty_feedback() {
        let mock = Arc::new(MockGraphStore::new());
        let analyzer = FeedbackAnalyzer::new(mock as Arc<dyn GraphStore>);

        let report = analyzer.analyze_runner_feedback(None).await.unwrap();
        assert_eq!(report.total_notes_analyzed, 0);
        assert_eq!(report.patterns.len(), 0);
        assert_eq!(report.actionable_count, 0);
    }

    #[tokio::test]
    async fn test_analyze_groups_by_pattern() {
        let mock = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // Create 2 manual-action notes and 3 chat-feedback notes
        create_feedback_notes(
            &mock,
            project_id,
            "runner-manual-action",
            2,
            "## Manual Action Detected Post-Run\n**Run ID**: {run_id}\n",
        )
        .await;

        create_feedback_notes(
            &mock,
            project_id,
            "runner-chat-feedback",
            3,
            "## Chat Feedback Post-Run\n**Run ID**: {run_id}\n",
        )
        .await;

        let analyzer = FeedbackAnalyzer::new(mock as Arc<dyn GraphStore>);
        let report = analyzer
            .analyze_runner_feedback(Some(project_id))
            .await
            .unwrap();

        assert_eq!(report.total_notes_analyzed, 5);
        assert!(
            report.patterns.len() >= 2,
            "Should have at least 2 pattern groups"
        );
    }

    // ========================================================================
    // Step 2 tests: pattern detection rules
    // ========================================================================

    #[tokio::test]
    async fn test_detect_manual_commit_pattern_actionable() {
        let mock = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // 4 manual commits → above threshold (>3)
        create_feedback_notes(
            &mock,
            project_id,
            "runner-manual-action",
            4,
            "## Manual Action Detected Post-Run\n**Run ID**: {run_id}\nCommit after light run",
        )
        .await;

        let analyzer = FeedbackAnalyzer::new(mock as Arc<dyn GraphStore>);
        let report = analyzer
            .analyze_runner_feedback(Some(project_id))
            .await
            .unwrap();

        assert_eq!(report.actionable_count, 1);
        let pattern = report.patterns.first().unwrap();
        assert_eq!(
            pattern.pattern_type,
            FeedbackPatternType::ManualCommitPostRun
        );
        assert!(pattern.is_actionable);
        assert!(
            matches!(pattern.suggestion, PatternSuggestion::UpgradeToFull),
            "Should suggest upgrade to full, got: {:?}",
            pattern.suggestion
        );
    }

    #[tokio::test]
    async fn test_detect_state_override_pattern() {
        let mock = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // 4 override-skip notes for "review" state
        for _ in 0..4 {
            let run_id = Uuid::new_v4();
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                format!(
                    "## User Override: State Skip\n**Run ID**: {}\n**Skipped state**: review\n",
                    run_id
                ),
                "runner-feedback".to_string(),
            );
            note.tags = vec![
                "runner-feedback".to_string(),
                "runner-override-skip".to_string(),
                "auto-generated".to_string(),
            ];
            mock.notes.write().await.insert(note.id, note);
        }

        let analyzer = FeedbackAnalyzer::new(mock as Arc<dyn GraphStore>);
        let report = analyzer
            .analyze_runner_feedback(Some(project_id))
            .await
            .unwrap();

        let override_pattern = report
            .patterns
            .iter()
            .find(|p| matches!(&p.pattern_type, FeedbackPatternType::RepeatedStateOverride { state_name } if state_name == "review"))
            .expect("Should detect review state override pattern");

        assert!(override_pattern.is_actionable);
        assert!(matches!(
            &override_pattern.suggestion,
            PatternSuggestion::CreateVariantWithout { state_name } if state_name == "review"
        ));
    }

    #[tokio::test]
    async fn test_below_threshold_not_actionable() {
        let mock = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // Only 2 notes — below threshold
        create_feedback_notes(
            &mock,
            project_id,
            "runner-manual-action",
            2,
            "## Manual Action\n**Run ID**: {run_id}\n",
        )
        .await;

        let analyzer = FeedbackAnalyzer::new(mock as Arc<dyn GraphStore>);
        let report = analyzer
            .analyze_runner_feedback(Some(project_id))
            .await
            .unwrap();

        assert_eq!(report.actionable_count, 0);
        assert!(!report.patterns[0].is_actionable);
        assert!(matches!(
            report.patterns[0].suggestion,
            PatternSuggestion::KeepCollecting
        ));
    }

    // ========================================================================
    // Step 3 tests: RFC generation
    // ========================================================================

    #[tokio::test]
    async fn test_generate_rfc_for_actionable_pattern() {
        let mock = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        create_feedback_notes(
            &mock,
            project_id,
            "runner-manual-action",
            5,
            "## Manual Action\n**Run ID**: {run_id}\n",
        )
        .await;

        let analyzer = FeedbackAnalyzer::new(mock.clone() as Arc<dyn GraphStore>);
        let report = analyzer
            .analyze_runner_feedback(Some(project_id))
            .await
            .unwrap();
        let rfc_ids = analyzer
            .generate_rfc_proposals(project_id, &report)
            .await
            .unwrap();

        assert_eq!(
            rfc_ids.len(),
            1,
            "Should generate one RFC for the actionable pattern"
        );

        // Verify the RFC note
        let notes = mock.notes.read().await;
        let rfc = notes.get(&rfc_ids[0]).unwrap();
        assert_eq!(rfc.note_type, NoteType::Rfc);
        assert!(rfc.tags.contains(&"runner-feedback-rfc".to_string()));
        assert!(rfc.tags.contains(&"proposed".to_string()));
        assert!(rfc.content.contains("```json"));
        assert!(rfc.content.contains("plan-runner-full-upgraded"));
    }

    // ========================================================================
    // Step 4 tests: Apply accepted RFC
    // ========================================================================

    #[tokio::test]
    async fn test_apply_accepted_rfc() {
        let mock = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // Create an RFC note with protocol spec
        let mut rfc = Note::new(
            Some(project_id),
            NoteType::Rfc,
            "## RFC\n```json\n{\"name\": \"test-protocol\", \"description\": \"test\", \"evidence_count\": 5}\n```\n".to_string(),
            "RFC: upgrade".to_string(),
        );
        rfc.tags = vec!["runner-feedback-rfc".to_string(), "proposed".to_string()];
        let rfc_id = rfc.id;
        mock.notes.write().await.insert(rfc_id, rfc);

        let analyzer = FeedbackAnalyzer::new(mock.clone() as Arc<dyn GraphStore>);
        let result = analyzer
            .apply_accepted_rfc(rfc_id, project_id)
            .await
            .unwrap();

        assert!(result.is_some());
        let applied = result.unwrap();
        assert_eq!(applied.name, "test-protocol");
        assert_eq!(applied.trust.source_episode_count, 5);
        assert_eq!(applied.trust.status, TrustStatus::Emerging);
        assert!(applied.trust.score < EMERGING_TRUST_THRESHOLD);
    }

    #[tokio::test]
    async fn test_apply_rfc_not_found() {
        let mock = Arc::new(MockGraphStore::new());
        let analyzer = FeedbackAnalyzer::new(mock as Arc<dyn GraphStore>);
        let result = analyzer
            .apply_accepted_rfc(Uuid::new_v4(), Uuid::new_v4())
            .await
            .unwrap();
        assert!(result.is_none());
    }

    // ========================================================================
    // Step 5 tests: Trust scoring
    // ========================================================================

    #[test]
    fn test_trust_initial_low() {
        let trust = ProtocolTrust::new(3);
        assert!(trust.score < EMERGING_TRUST_THRESHOLD);
        assert_eq!(trust.status, TrustStatus::Emerging);
    }

    #[test]
    fn test_trust_increases_with_positive_runs() {
        let mut trust = ProtocolTrust::new(5);
        // Need enough positive runs to cross 0.5
        for _ in 0..5 {
            trust.record_positive_run();
        }
        assert!(trust.score >= EMERGING_TRUST_THRESHOLD);
        assert_eq!(trust.status, TrustStatus::Active);
    }

    #[test]
    fn test_trust_decreases_with_negative_runs() {
        let mut trust = ProtocolTrust::new(5);
        // Get above threshold first
        for _ in 0..5 {
            trust.record_positive_run();
        }
        assert_eq!(trust.status, TrustStatus::Active);

        // Negative runs bring it back down
        for _ in 0..5 {
            trust.record_negative_run();
        }
        assert_eq!(trust.status, TrustStatus::Emerging);
    }

    #[test]
    fn test_trust_bounded() {
        let mut trust = ProtocolTrust::new(10);
        for _ in 0..100 {
            trust.record_positive_run();
        }
        assert!(trust.score <= 1.0);

        for _ in 0..200 {
            trust.record_negative_run();
        }
        assert!(trust.score >= 0.0);
    }

    #[test]
    fn test_compute_trust_utility() {
        let trust = FeedbackAnalyzer::compute_trust(5, 10, 2);
        assert!(trust.positive_runs == 10);
        assert!(trust.negative_runs == 2);
        // 5 episodes → base ≈ 0.4, +10×0.07=0.7, -2×0.105=0.21 → ~0.89
        assert!(trust.score > 0.5);
        assert_eq!(trust.status, TrustStatus::Active);
    }

    // ========================================================================
    // Internal helper tests
    // ========================================================================

    #[test]
    fn test_extract_field() {
        let analyzer = FeedbackAnalyzer::new(Arc::new(MockGraphStore::new()));
        let content = "## Override\n**Skipped state**: review\n**From state**: executing\n";
        assert_eq!(
            analyzer.extract_field(content, "Skipped state"),
            Some("review".to_string())
        );
        assert_eq!(
            analyzer.extract_field(content, "From state"),
            Some("executing".to_string())
        );
        assert_eq!(analyzer.extract_field(content, "Missing field"), None);
    }

    #[test]
    fn test_extract_run_id() {
        let analyzer = FeedbackAnalyzer::new(Arc::new(MockGraphStore::new()));
        let run_id = Uuid::new_v4();
        let content = format!("## Run\n**Run ID**: {}\n**Other**: stuff\n", run_id);
        assert_eq!(analyzer.extract_run_id_from_content(&content), Some(run_id));
    }

    #[test]
    fn test_extract_protocol_json() {
        let analyzer = FeedbackAnalyzer::new(Arc::new(MockGraphStore::new()));
        let content = "## RFC\n```json\n{\"name\": \"test\", \"states\": []}\n```\nDone.";
        let result = analyzer.extract_protocol_json(content).unwrap();
        assert_eq!(result["name"], "test");
    }

    #[test]
    fn test_extract_protocol_json_missing() {
        let analyzer = FeedbackAnalyzer::new(Arc::new(MockGraphStore::new()));
        let content = "No JSON here.";
        assert!(analyzer.extract_protocol_json(content).is_err());
    }
}
