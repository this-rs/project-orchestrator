//! # Episode→Skill Feedback Loop
//!
//! Analyzes episodes from completed protocol runs to extract patterns
//! of success/failure and create/enrich skills automatically.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ─── Pattern types ──────────────────────────────────────────────────────────

/// A pattern detected from analyzing episodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Pattern identifier.
    pub id: String,
    /// Pattern type.
    pub pattern_type: PatternType,
    /// Human-readable description.
    pub description: String,
    /// How many times this pattern was observed.
    pub frequency: usize,
    /// Confidence score `[0, 1]`.
    pub confidence: f64,
    /// Associated tech stacks.
    pub tech_stacks: Vec<String>,
    /// Which quality gates are involved.
    pub related_gates: Vec<String>,
    /// Suggested action/recommendation.
    pub recommendation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// A gate that frequently fails.
    FrequentGateFailure,
    /// Tasks of a certain type that cause regressions.
    RegressionProne,
    /// A retry strategy that often works.
    EffectiveRetry,
    /// A common root cause for failures.
    CommonRootCause,
    /// A successful pattern to replicate.
    SuccessPattern,
}

// ─── Episode data ───────────────────────────────────────────────────────────

/// Simplified episode data for analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeData {
    pub episode_id: Uuid,
    pub protocol_name: String,
    pub tech_stack: Vec<String>,
    pub outcome: EpisodeOutcome,
    pub gate_results: Vec<GateOutcome>,
    pub duration_ms: u64,
    pub retry_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpisodeOutcome {
    Success,
    Failure,
    Partial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateOutcome {
    pub gate_name: String,
    pub passed: bool,
    pub failure_message: Option<String>,
}

// ─── Skill recommendation ───────────────────────────────────────────────────

/// Skill creation recommendation produced from detected patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRecommendation {
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub trigger_patterns: Vec<String>,
    pub notes: Vec<NoteContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteContent {
    /// Note type: "gotcha", "pattern", "tip".
    pub note_type: String,
    /// Note content.
    pub content: String,
    /// Importance level: "critical", "high", "medium", "low".
    pub importance: String,
}

// ─── Episode Analyzer ───────────────────────────────────────────────────────

/// The Episode Analyzer.
///
/// Analyzes collections of episodes to detect recurring patterns (gate failures,
/// regressions, effective retries, common root causes, success patterns) and
/// generates skill recommendations from those patterns.
pub struct EpisodeAnalyzer {
    /// Minimum frequency to consider a pattern significant.
    min_frequency: usize,
    /// Minimum confidence threshold.
    min_confidence: f64,
}

impl EpisodeAnalyzer {
    /// Create a new analyzer.
    ///
    /// Defaults: `min_frequency = 3`, `min_confidence = 0.6`.
    pub fn new(min_frequency: usize, min_confidence: f64) -> Self {
        Self {
            min_frequency,
            min_confidence,
        }
    }

    /// Analyze a collection of episodes and extract significant patterns.
    ///
    /// Groups episodes by `protocol_name` + `tech_stack`, then detects:
    /// - Frequent gate failures
    /// - Regression-prone protocols
    /// - Effective retry strategies
    /// - Common root causes
    /// - Success patterns
    pub fn analyze(&self, episodes: &[EpisodeData]) -> Vec<DetectedPattern> {
        if episodes.is_empty() {
            return Vec::new();
        }

        let mut patterns = Vec::new();

        // Collect from each analysis pass
        patterns.extend(
            Self::analyze_gate_failures(episodes)
                .into_iter()
                .filter(|p| p.frequency >= self.min_frequency && p.confidence >= self.min_confidence),
        );

        patterns.extend(
            self.analyze_regression_prone(episodes)
                .into_iter()
                .filter(|p| p.frequency >= self.min_frequency && p.confidence >= self.min_confidence),
        );

        patterns.extend(
            Self::analyze_retry_effectiveness(episodes)
                .into_iter()
                .filter(|p| p.frequency >= self.min_frequency && p.confidence >= self.min_confidence),
        );

        patterns.extend(
            self.analyze_common_root_causes(episodes)
                .into_iter()
                .filter(|p| p.frequency >= self.min_frequency && p.confidence >= self.min_confidence),
        );

        patterns.extend(
            self.analyze_success_patterns(episodes)
                .into_iter()
                .filter(|p| p.frequency >= self.min_frequency && p.confidence >= self.min_confidence),
        );

        patterns
    }

    /// Generate skill recommendations from detected patterns.
    ///
    /// Each significant pattern produces a [`SkillRecommendation`] with a
    /// descriptive name, notes containing insights, and trigger patterns.
    pub fn recommend_skills(&self, patterns: &[DetectedPattern]) -> Vec<SkillRecommendation> {
        patterns
            .iter()
            .map(|pattern| {
                let (name, trigger) = match &pattern.pattern_type {
                    PatternType::FrequentGateFailure => (
                        format!("handle-{}-failure", pattern.related_gates.first().map_or("gate", |s| s.as_str())),
                        pattern.related_gates.iter().map(|g| format!("gate_failure:{g}")).collect(),
                    ),
                    PatternType::RegressionProne => (
                        format!("regression-guard-{}", slug(&pattern.description)),
                        vec![format!("protocol:{}", slug(&pattern.description))],
                    ),
                    PatternType::EffectiveRetry => (
                        format!("retry-strategy-{}", slug(&pattern.description)),
                        vec!["retry:needed".to_string()],
                    ),
                    PatternType::CommonRootCause => (
                        format!("root-cause-{}", slug(&pattern.description)),
                        vec![format!("error:{}", slug(&pattern.description))],
                    ),
                    PatternType::SuccessPattern => (
                        format!("apply-{}", slug(&pattern.description)),
                        vec![format!("pattern:{}", slug(&pattern.description))],
                    ),
                };

                let importance = if pattern.confidence >= 0.9 {
                    "critical"
                } else if pattern.confidence >= 0.75 {
                    "high"
                } else if pattern.confidence >= 0.5 {
                    "medium"
                } else {
                    "low"
                };

                let note_type = match &pattern.pattern_type {
                    PatternType::FrequentGateFailure | PatternType::CommonRootCause => "gotcha",
                    PatternType::SuccessPattern | PatternType::EffectiveRetry => "pattern",
                    PatternType::RegressionProne => "tip",
                };

                SkillRecommendation {
                    name,
                    description: pattern.description.clone(),
                    tags: pattern.tech_stacks.clone(),
                    trigger_patterns: trigger,
                    notes: vec![
                        NoteContent {
                            note_type: note_type.to_string(),
                            content: pattern.recommendation.clone(),
                            importance: importance.to_string(),
                        },
                        NoteContent {
                            note_type: "pattern".to_string(),
                            content: format!(
                                "Observed {} times with {:.0}% confidence",
                                pattern.frequency,
                                pattern.confidence * 100.0
                            ),
                            importance: "medium".to_string(),
                        },
                    ],
                }
            })
            .collect()
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Analyze gate failures across episodes.
    ///
    /// Counts how often each gate fails and produces `FrequentGateFailure`
    /// patterns for gates that fail frequently.
    pub fn analyze_gate_failures(episodes: &[EpisodeData]) -> Vec<DetectedPattern> {
        // gate_name -> (fail_count, total_count, tech_stacks)
        let mut gate_stats: HashMap<String, (usize, usize, Vec<String>)> = HashMap::new();

        for ep in episodes {
            for gate in &ep.gate_results {
                let entry = gate_stats
                    .entry(gate.gate_name.clone())
                    .or_insert_with(|| (0, 0, Vec::new()));
                entry.1 += 1;
                if !gate.passed {
                    entry.0 += 1;
                }
                for ts in &ep.tech_stack {
                    if !entry.2.contains(ts) {
                        entry.2.push(ts.clone());
                    }
                }
            }
        }

        gate_stats
            .into_iter()
            .filter(|(_, (fail_count, _, _))| *fail_count > 0)
            .map(|(gate_name, (fail_count, total_count, tech_stacks))| {
                let confidence = fail_count as f64 / total_count as f64;
                DetectedPattern {
                    id: format!("gate-failure-{gate_name}"),
                    pattern_type: PatternType::FrequentGateFailure,
                    description: format!("Gate '{gate_name}' fails frequently ({fail_count}/{total_count})"),
                    frequency: fail_count,
                    confidence,
                    tech_stacks,
                    related_gates: vec![gate_name.clone()],
                    recommendation: format!(
                        "Investigate why '{gate_name}' fails {:.0}% of the time. Consider adding pre-checks or adjusting thresholds.",
                        confidence * 100.0
                    ),
                }
            })
            .collect()
    }

    /// Analyze retry effectiveness across episodes.
    ///
    /// Identifies patterns where retries eventually lead to success, producing
    /// `EffectiveRetry` patterns.
    pub fn analyze_retry_effectiveness(episodes: &[EpisodeData]) -> Vec<DetectedPattern> {
        // protocol_name -> (retries_that_succeeded, total_retried, tech_stacks)
        let mut retry_stats: HashMap<String, (usize, usize, Vec<String>)> = HashMap::new();

        for ep in episodes {
            if ep.retry_count == 0 {
                continue;
            }
            let entry = retry_stats
                .entry(ep.protocol_name.clone())
                .or_insert_with(|| (0, 0, Vec::new()));
            entry.1 += 1;
            if ep.outcome == EpisodeOutcome::Success {
                entry.0 += 1;
            }
            for ts in &ep.tech_stack {
                if !entry.2.contains(ts) {
                    entry.2.push(ts.clone());
                }
            }
        }

        retry_stats
            .into_iter()
            .filter(|(_, (successes, _, _))| *successes > 0)
            .map(|(protocol, (successes, total, tech_stacks))| {
                let confidence = successes as f64 / total as f64;
                DetectedPattern {
                    id: format!("effective-retry-{protocol}"),
                    pattern_type: PatternType::EffectiveRetry,
                    description: format!("Retries in '{protocol}' often succeed ({successes}/{total})"),
                    frequency: successes,
                    confidence,
                    tech_stacks,
                    related_gates: Vec::new(),
                    recommendation: format!(
                        "Retrying '{protocol}' works {:.0}% of the time. Consider automating retry logic.",
                        confidence * 100.0
                    ),
                }
            })
            .collect()
    }

    /// Find protocols that frequently fail (regression-prone).
    fn analyze_regression_prone(&self, episodes: &[EpisodeData]) -> Vec<DetectedPattern> {
        // protocol_name -> (fail_count, total_count, tech_stacks)
        let mut proto_stats: HashMap<String, (usize, usize, Vec<String>)> = HashMap::new();

        for ep in episodes {
            let entry = proto_stats
                .entry(ep.protocol_name.clone())
                .or_insert_with(|| (0, 0, Vec::new()));
            entry.1 += 1;
            if ep.outcome == EpisodeOutcome::Failure {
                entry.0 += 1;
            }
            for ts in &ep.tech_stack {
                if !entry.2.contains(ts) {
                    entry.2.push(ts.clone());
                }
            }
        }

        proto_stats
            .into_iter()
            .filter(|(_, (fail_count, _, _))| *fail_count > 0)
            .map(|(protocol, (fail_count, total_count, tech_stacks))| {
                let confidence = fail_count as f64 / total_count as f64;
                DetectedPattern {
                    id: format!("regression-prone-{protocol}"),
                    pattern_type: PatternType::RegressionProne,
                    description: format!("Protocol '{protocol}' is regression-prone ({fail_count}/{total_count} failures)"),
                    frequency: fail_count,
                    confidence,
                    tech_stacks,
                    related_gates: Vec::new(),
                    recommendation: format!(
                        "Protocol '{protocol}' fails {:.0}% of the time. Add stricter pre-conditions or break into smaller steps.",
                        confidence * 100.0
                    ),
                }
            })
            .collect()
    }

    /// Extract common failure messages (root causes).
    fn analyze_common_root_causes(&self, episodes: &[EpisodeData]) -> Vec<DetectedPattern> {
        // failure_message -> (count, gate_names, tech_stacks)
        let mut cause_stats: HashMap<String, (usize, Vec<String>, Vec<String>)> = HashMap::new();

        for ep in episodes {
            for gate in &ep.gate_results {
                if let Some(ref msg) = gate.failure_message {
                    let normalized = msg.trim().to_lowercase();
                    let entry = cause_stats
                        .entry(normalized)
                        .or_insert_with(|| (0, Vec::new(), Vec::new()));
                    entry.0 += 1;
                    if !entry.1.contains(&gate.gate_name) {
                        entry.1.push(gate.gate_name.clone());
                    }
                    for ts in &ep.tech_stack {
                        if !entry.2.contains(ts) {
                            entry.2.push(ts.clone());
                        }
                    }
                }
            }
        }

        let total_failures: usize = cause_stats.values().map(|(c, _, _)| c).sum();

        cause_stats
            .into_iter()
            .map(|(message, (count, gates, tech_stacks))| {
                let confidence = if total_failures > 0 {
                    count as f64 / total_failures as f64
                } else {
                    0.0
                };
                DetectedPattern {
                    id: format!("root-cause-{}", slug(&message)),
                    pattern_type: PatternType::CommonRootCause,
                    description: message.clone(),
                    frequency: count,
                    confidence,
                    tech_stacks,
                    related_gates: gates,
                    recommendation: format!(
                        "Common failure: \"{message}\". Seen {count} times. Create a targeted fix or workaround."
                    ),
                }
            })
            .collect()
    }

    /// Detect success patterns (protocols + tech stacks that consistently succeed).
    fn analyze_success_patterns(&self, episodes: &[EpisodeData]) -> Vec<DetectedPattern> {
        // (protocol, tech_stack_key) -> (success_count, total_count, tech_stacks)
        let mut success_stats: HashMap<String, (usize, usize, Vec<String>)> = HashMap::new();

        for ep in episodes {
            let key = format!("{}:{}", ep.protocol_name, ep.tech_stack.join(","));
            let entry = success_stats
                .entry(key)
                .or_insert_with(|| (0, 0, ep.tech_stack.clone()));
            entry.1 += 1;
            if ep.outcome == EpisodeOutcome::Success {
                entry.0 += 1;
            }
        }

        success_stats
            .into_iter()
            .filter(|(_, (successes, _, _))| *successes > 0)
            .map(|(key, (successes, total, tech_stacks))| {
                let confidence = successes as f64 / total as f64;
                let protocol = key.split(':').next().unwrap_or(&key);
                DetectedPattern {
                    id: format!("success-{}", slug(&key)),
                    pattern_type: PatternType::SuccessPattern,
                    description: format!("Protocol '{protocol}' succeeds consistently ({successes}/{total})"),
                    frequency: successes,
                    confidence,
                    tech_stacks,
                    related_gates: Vec::new(),
                    recommendation: format!(
                        "Protocol '{protocol}' has a {:.0}% success rate. Replicate this pattern for similar tasks.",
                        confidence * 100.0
                    ),
                }
            })
            .collect()
    }
}

/// Create a URL-safe slug from a string.
fn slug(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_episode(
        protocol: &str,
        tech: &[&str],
        outcome: EpisodeOutcome,
        gates: Vec<GateOutcome>,
        retry_count: usize,
    ) -> EpisodeData {
        EpisodeData {
            episode_id: Uuid::new_v4(),
            protocol_name: protocol.to_string(),
            tech_stack: tech.iter().map(|s| s.to_string()).collect(),
            outcome,
            gate_results: gates,
            duration_ms: 1000,
            retry_count,
        }
    }

    fn gate(name: &str, passed: bool, msg: Option<&str>) -> GateOutcome {
        GateOutcome {
            gate_name: name.to_string(),
            passed,
            failure_message: msg.map(|s| s.to_string()),
        }
    }

    // ── Pattern detection ───────────────────────────────────────────────

    #[test]
    fn detects_frequent_gate_failures() {
        let episodes: Vec<EpisodeData> = (0..5)
            .map(|_| {
                make_episode(
                    "build",
                    &["rust"],
                    EpisodeOutcome::Failure,
                    vec![gate("cargo-check", false, Some("compilation error"))],
                    0,
                )
            })
            .collect();

        let analyzer = EpisodeAnalyzer::new(3, 0.5);
        let patterns = analyzer.analyze(&episodes);

        assert!(
            patterns.iter().any(|p| p.pattern_type == PatternType::FrequentGateFailure),
            "expected FrequentGateFailure pattern, got: {patterns:?}"
        );

        let gate_pattern = patterns
            .iter()
            .find(|p| p.pattern_type == PatternType::FrequentGateFailure)
            .unwrap();
        assert_eq!(gate_pattern.frequency, 5);
        assert!(gate_pattern.related_gates.contains(&"cargo-check".to_string()));
    }

    #[test]
    fn detects_effective_retries() {
        let episodes: Vec<EpisodeData> = (0..4)
            .map(|_| {
                make_episode(
                    "deploy",
                    &["docker"],
                    EpisodeOutcome::Success,
                    vec![gate("health-check", true, None)],
                    2, // retried twice before succeeding
                )
            })
            .collect();

        let analyzer = EpisodeAnalyzer::new(3, 0.5);
        let patterns = analyzer.analyze(&episodes);

        assert!(
            patterns.iter().any(|p| p.pattern_type == PatternType::EffectiveRetry),
            "expected EffectiveRetry pattern, got: {patterns:?}"
        );
    }

    #[test]
    fn detects_regression_prone_protocols() {
        let episodes: Vec<EpisodeData> = (0..4)
            .map(|_| {
                make_episode(
                    "migrate-db",
                    &["postgres"],
                    EpisodeOutcome::Failure,
                    vec![gate("schema-validate", false, Some("migration failed"))],
                    0,
                )
            })
            .collect();

        let analyzer = EpisodeAnalyzer::new(3, 0.5);
        let patterns = analyzer.analyze(&episodes);

        assert!(
            patterns.iter().any(|p| p.pattern_type == PatternType::RegressionProne),
            "expected RegressionProne pattern, got: {patterns:?}"
        );
    }

    #[test]
    fn detects_common_root_causes() {
        let episodes: Vec<EpisodeData> = (0..5)
            .map(|_| {
                make_episode(
                    "test",
                    &["rust"],
                    EpisodeOutcome::Failure,
                    vec![gate("cargo-test", false, Some("timeout exceeded"))],
                    0,
                )
            })
            .collect();

        let analyzer = EpisodeAnalyzer::new(3, 0.5);
        let patterns = analyzer.analyze(&episodes);

        assert!(
            patterns.iter().any(|p| p.pattern_type == PatternType::CommonRootCause),
            "expected CommonRootCause pattern, got: {patterns:?}"
        );
    }

    #[test]
    fn detects_success_patterns() {
        let episodes: Vec<EpisodeData> = (0..5)
            .map(|_| {
                make_episode(
                    "lint",
                    &["typescript"],
                    EpisodeOutcome::Success,
                    vec![gate("eslint", true, None)],
                    0,
                )
            })
            .collect();

        let analyzer = EpisodeAnalyzer::new(3, 0.5);
        let patterns = analyzer.analyze(&episodes);

        assert!(
            patterns.iter().any(|p| p.pattern_type == PatternType::SuccessPattern),
            "expected SuccessPattern pattern, got: {patterns:?}"
        );
    }

    // ── Skill recommendations ───────────────────────────────────────────

    #[test]
    fn generates_skill_recommendations() {
        let patterns = vec![
            DetectedPattern {
                id: "gate-failure-cargo-test".to_string(),
                pattern_type: PatternType::FrequentGateFailure,
                description: "Gate 'cargo-test' fails frequently (5/6)".to_string(),
                frequency: 5,
                confidence: 0.83,
                tech_stacks: vec!["rust".to_string()],
                related_gates: vec!["cargo-test".to_string()],
                recommendation: "Investigate test failures.".to_string(),
            },
            DetectedPattern {
                id: "effective-retry-deploy".to_string(),
                pattern_type: PatternType::EffectiveRetry,
                description: "Retries in 'deploy' often succeed".to_string(),
                frequency: 4,
                confidence: 0.75,
                tech_stacks: vec!["docker".to_string()],
                related_gates: Vec::new(),
                recommendation: "Automate retry logic.".to_string(),
            },
        ];

        let analyzer = EpisodeAnalyzer::new(3, 0.6);
        let skills = analyzer.recommend_skills(&patterns);

        assert_eq!(skills.len(), 2);

        // First skill — gate failure
        assert_eq!(skills[0].name, "handle-cargo-test-failure");
        assert!(skills[0].tags.contains(&"rust".to_string()));
        assert!(!skills[0].notes.is_empty());
        assert!(skills[0].trigger_patterns.iter().any(|t| t.contains("gate_failure")));

        // Second skill — retry
        assert!(skills[1].name.starts_with("retry-strategy-"));
        assert!(skills[1].tags.contains(&"docker".to_string()));
        assert_eq!(skills[1].notes.len(), 2);
    }

    #[test]
    fn skill_importance_based_on_confidence() {
        let high_confidence = DetectedPattern {
            id: "test".to_string(),
            pattern_type: PatternType::SuccessPattern,
            description: "good pattern".to_string(),
            frequency: 10,
            confidence: 0.95,
            tech_stacks: vec![],
            related_gates: vec![],
            recommendation: "Keep doing this.".to_string(),
        };

        let low_confidence = DetectedPattern {
            id: "test2".to_string(),
            pattern_type: PatternType::SuccessPattern,
            description: "weak pattern".to_string(),
            frequency: 3,
            confidence: 0.45,
            tech_stacks: vec![],
            related_gates: vec![],
            recommendation: "Maybe try this.".to_string(),
        };

        let analyzer = EpisodeAnalyzer::new(1, 0.0);
        let skills = analyzer.recommend_skills(&[high_confidence, low_confidence]);

        assert_eq!(skills[0].notes[0].importance, "critical");
        assert_eq!(skills[1].notes[0].importance, "low");
    }

    // ── Empty input ─────────────────────────────────────────────────────

    #[test]
    fn empty_episodes_returns_no_patterns() {
        let analyzer = EpisodeAnalyzer::new(3, 0.6);
        let patterns = analyzer.analyze(&[]);
        assert!(patterns.is_empty());
    }

    #[test]
    fn empty_patterns_returns_no_skills() {
        let analyzer = EpisodeAnalyzer::new(3, 0.6);
        let skills = analyzer.recommend_skills(&[]);
        assert!(skills.is_empty());
    }

    // ── Threshold filtering ─────────────────────────────────────────────

    #[test]
    fn patterns_below_min_frequency_are_excluded() {
        // Only 2 episodes — below min_frequency of 3
        let episodes = vec![
            make_episode(
                "build",
                &["rust"],
                EpisodeOutcome::Failure,
                vec![gate("cargo-check", false, Some("error"))],
                0,
            ),
            make_episode(
                "build",
                &["rust"],
                EpisodeOutcome::Failure,
                vec![gate("cargo-check", false, Some("error"))],
                0,
            ),
        ];

        let analyzer = EpisodeAnalyzer::new(3, 0.5);
        let patterns = analyzer.analyze(&episodes);

        // Gate failure count is 2, below threshold of 3
        assert!(
            !patterns.iter().any(|p| p.pattern_type == PatternType::FrequentGateFailure),
            "should not detect gate failure below min_frequency"
        );
    }

    #[test]
    fn patterns_below_min_confidence_are_excluded() {
        // Mix of pass and fail so confidence is low
        let mut episodes = Vec::new();
        for _ in 0..3 {
            episodes.push(make_episode(
                "build",
                &["rust"],
                EpisodeOutcome::Failure,
                vec![gate("cargo-check", false, Some("error"))],
                0,
            ));
        }
        for _ in 0..10 {
            episodes.push(make_episode(
                "build",
                &["rust"],
                EpisodeOutcome::Success,
                vec![gate("cargo-check", true, None)],
                0,
            ));
        }

        // Confidence for gate failure = 3/13 ≈ 0.23, below 0.5 threshold
        let analyzer = EpisodeAnalyzer::new(3, 0.5);
        let patterns = analyzer.analyze(&episodes);

        assert!(
            !patterns.iter().any(|p| p.pattern_type == PatternType::FrequentGateFailure),
            "should not detect gate failure below min_confidence"
        );
    }

    // ── Helper functions directly ───────────────────────────────────────

    #[test]
    fn analyze_gate_failures_groups_by_gate_name() {
        let episodes = vec![
            make_episode("a", &["rust"], EpisodeOutcome::Failure, vec![
                gate("cargo-check", false, Some("err")),
                gate("cargo-test", true, None),
            ], 0),
            make_episode("a", &["rust"], EpisodeOutcome::Failure, vec![
                gate("cargo-check", false, Some("err")),
                gate("cargo-test", false, Some("test failed")),
            ], 0),
        ];

        let patterns = EpisodeAnalyzer::analyze_gate_failures(&episodes);

        let check_pattern = patterns.iter().find(|p| p.related_gates.contains(&"cargo-check".to_string()));
        assert!(check_pattern.is_some());
        assert_eq!(check_pattern.unwrap().frequency, 2);

        let test_pattern = patterns.iter().find(|p| p.related_gates.contains(&"cargo-test".to_string()));
        assert!(test_pattern.is_some());
        assert_eq!(test_pattern.unwrap().frequency, 1);
    }

    #[test]
    fn analyze_retry_effectiveness_ignores_zero_retries() {
        let episodes = vec![
            make_episode("build", &["rust"], EpisodeOutcome::Success, vec![], 0),
            make_episode("build", &["rust"], EpisodeOutcome::Success, vec![], 0),
        ];

        let patterns = EpisodeAnalyzer::analyze_retry_effectiveness(&episodes);
        assert!(patterns.is_empty());
    }

    #[test]
    fn analyze_retry_effectiveness_tracks_success_rate() {
        let episodes = vec![
            make_episode("deploy", &["k8s"], EpisodeOutcome::Success, vec![], 2),
            make_episode("deploy", &["k8s"], EpisodeOutcome::Failure, vec![], 1),
            make_episode("deploy", &["k8s"], EpisodeOutcome::Success, vec![], 3),
        ];

        let patterns = EpisodeAnalyzer::analyze_retry_effectiveness(&episodes);
        assert_eq!(patterns.len(), 1);

        let p = &patterns[0];
        assert_eq!(p.frequency, 2); // 2 successes out of 3 retried
        assert!((p.confidence - 2.0 / 3.0).abs() < f64::EPSILON);
    }
}
