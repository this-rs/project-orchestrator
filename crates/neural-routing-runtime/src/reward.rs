//! Session reward heuristic — computes a composite reward from observable signals.
//!
//! The reward is used by `TrajectoryCollector.end_session()` to assign a quality
//! score to each session's trajectory before flushing to Neo4j.
//!
//! Signals (all in [0.0, 1.0]):
//! - `tool_success_rate`: successful tool calls / total tool calls
//! - `avg_confidence`: mean model confidence across decisions
//! - `duration_score`: sigmoid(duration, center) — penalizes very short/long sessions
//! - `decision_count_score`: min(1.0, count / target) — more decisions = richer signal
//! - `task_completion_rate`: tasks completed / tasks total (0 if no tasks)
//!
//! Composite: reward = Σ(weight_i × signal_i), clamped to [0.0, 1.0].

use std::time::Duration;

use crate::collector::DecisionRecord;
use crate::config::RewardHeuristicConfig;

/// Observable signals from a completed session.
#[derive(Debug, Clone)]
pub struct SessionSignals {
    /// Fraction of tool calls that succeeded (0.0 - 1.0).
    pub tool_success_rate: f64,
    /// Average model confidence across all decisions (0.0 - 1.0).
    pub avg_confidence: f64,
    /// Session duration in seconds.
    pub duration_secs: f64,
    /// Number of decisions recorded.
    pub decision_count: usize,
    /// Number of tasks completed during the session.
    pub tasks_completed: usize,
    /// Total number of tasks active during the session.
    pub tasks_total: usize,
}

impl SessionSignals {
    /// Extract signals from collected decision records.
    ///
    /// `tasks_completed` and `tasks_total` must be provided externally
    /// (from MCP task status changes observed during the session).
    pub fn from_decisions(
        decisions: &[DecisionRecord],
        duration: Duration,
        tasks_completed: usize,
        tasks_total: usize,
    ) -> Self {
        if decisions.is_empty() {
            return Self {
                tool_success_rate: 0.0,
                avg_confidence: 0.0,
                duration_secs: duration.as_secs_f64(),
                decision_count: 0,
                tasks_completed,
                tasks_total,
            };
        }

        // Tool success rate: count successful tool usages across all decisions
        let mut total_tools = 0usize;
        let mut successful_tools = 0usize;
        for d in decisions {
            for tu in &d.tool_usages {
                total_tools += 1;
                if tu.success {
                    successful_tools += 1;
                }
            }
        }
        let tool_success_rate = if total_tools > 0 {
            successful_tools as f64 / total_tools as f64
        } else {
            // No tool usages recorded — neutral (0.5)
            0.5
        };

        // Average confidence
        let confidence_sum: f64 = decisions.iter().map(|d| d.confidence).sum();
        let avg_confidence = confidence_sum / decisions.len() as f64;

        Self {
            tool_success_rate,
            avg_confidence,
            duration_secs: duration.as_secs_f64(),
            decision_count: decisions.len(),
            tasks_completed,
            tasks_total,
        }
    }
}

/// Computes a composite session reward from observable signals.
#[derive(Debug, Clone)]
pub struct SessionRewardComputer {
    config: RewardHeuristicConfig,
}

impl SessionRewardComputer {
    pub fn new(config: RewardHeuristicConfig) -> Self {
        Self { config }
    }

    /// Compute the composite reward for a session, clamped to [0.0, 1.0].
    pub fn compute(&self, signals: &SessionSignals) -> f64 {
        // Empty session → zero reward (no signal to learn from)
        if signals.decision_count == 0 {
            return 0.0;
        }

        let c = &self.config;

        // 1. Tool success rate (already in [0.0, 1.0])
        let tool_score = signals.tool_success_rate;

        // 2. Task completion rate
        let task_score = if signals.tasks_total > 0 {
            signals.tasks_completed as f64 / signals.tasks_total as f64
        } else {
            // No tasks in session — neutral (0.5)
            0.5
        };

        // 3. Average confidence (already in [0.0, 1.0])
        let confidence_score = signals.avg_confidence;

        // 4. Duration score — sigmoid centered at duration_center_secs
        //    sigmoid(x) = 1 / (1 + exp(-k*(x - center)))
        //    k = 0.02 gives a smooth curve: 60s → ~0.01, 300s → 0.5, 600s → ~0.998
        let k = 0.02;
        let duration_score =
            1.0 / (1.0 + (-k * (signals.duration_secs - c.duration_center_secs)).exp());

        // 5. Decision count score — saturates at target
        let decision_score = if c.decision_count_target > 0.0 {
            (signals.decision_count as f64 / c.decision_count_target).min(1.0)
        } else {
            1.0
        };

        // Weighted sum
        let raw = c.weight_tool_success * tool_score
            + c.weight_task_completion * task_score
            + c.weight_confidence * confidence_score
            + c.weight_duration * duration_score
            + c.weight_decision_count * decision_score;

        // Clamp to [0.0, 1.0]
        raw.clamp(0.0, 1.0)
    }
}

impl Default for SessionRewardComputer {
    fn default() -> Self {
        Self::new(RewardHeuristicConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use neural_routing_core::ToolUsage;

    fn make_decision(action: &str, confidence: f64, tool_success: bool) -> DecisionRecord {
        DecisionRecord {
            session_id: "test".to_string(),
            context_embedding: vec![],
            action_type: action.to_string(),
            action_params: serde_json::json!({}),
            alternatives_count: 3,
            chosen_index: 0,
            confidence,
            tool_usages: vec![ToolUsage {
                tool_name: "code".to_string(),
                action: "search".to_string(),
                params_hash: "h".to_string(),
                duration_ms: Some(10),
                success: tool_success,
            }],
            touched_entities: vec![],
            timestamp_ms: 0,
            query_embedding: vec![],
            node_features: vec![],
            protocol_run_id: None,
            protocol_state: None,
        }
    }

    #[test]
    fn test_empty_session_returns_zero() {
        let computer = SessionRewardComputer::default();
        let signals = SessionSignals {
            tool_success_rate: 0.0,
            avg_confidence: 0.0,
            duration_secs: 0.0,
            decision_count: 0,
            tasks_completed: 0,
            tasks_total: 0,
        };
        assert_eq!(computer.compute(&signals), 0.0);
    }

    #[test]
    fn test_all_success_high_reward() {
        let computer = SessionRewardComputer::default();
        let signals = SessionSignals {
            tool_success_rate: 1.0,
            avg_confidence: 0.9,
            duration_secs: 300.0, // center → 0.5
            decision_count: 10,   // target → 1.0
            tasks_completed: 3,
            tasks_total: 3, // 100% completion
        };
        let reward = computer.compute(&signals);
        assert!(
            reward > 0.7,
            "All-success session should have reward > 0.7, got {}",
            reward
        );
    }

    #[test]
    fn test_all_failure_low_reward() {
        let computer = SessionRewardComputer::default();
        let signals = SessionSignals {
            tool_success_rate: 0.0,
            avg_confidence: 0.1,
            duration_secs: 30.0, // very short → low duration score
            decision_count: 1,   // minimal
            tasks_completed: 0,
            tasks_total: 3, // 0% completion
        };
        let reward = computer.compute(&signals);
        assert!(
            reward < 0.3,
            "All-failure session should have reward < 0.3, got {}",
            reward
        );
    }

    #[test]
    fn test_single_decision_valid_reward() {
        let computer = SessionRewardComputer::default();
        let signals = SessionSignals {
            tool_success_rate: 1.0,
            avg_confidence: 0.8,
            duration_secs: 60.0,
            decision_count: 1,
            tasks_completed: 0,
            tasks_total: 0,
        };
        let reward = computer.compute(&signals);
        assert!(reward > 0.0 && reward <= 1.0, "Got {}", reward);
    }

    #[test]
    fn test_reward_clamped_to_unit_interval() {
        // Even with extreme weights, reward should be in [0, 1]
        let config = RewardHeuristicConfig {
            weight_tool_success: 5.0,
            weight_task_completion: 5.0,
            weight_confidence: 5.0,
            weight_duration: 5.0,
            weight_decision_count: 5.0,
            ..Default::default()
        };
        let computer = SessionRewardComputer::new(config);
        let signals = SessionSignals {
            tool_success_rate: 1.0,
            avg_confidence: 1.0,
            duration_secs: 300.0,
            decision_count: 100,
            tasks_completed: 10,
            tasks_total: 10,
        };
        let reward = computer.compute(&signals);
        assert_eq!(reward, 1.0, "Should be clamped to 1.0, got {}", reward);
    }

    #[test]
    fn test_duration_sigmoid_center() {
        let computer = SessionRewardComputer::default();
        // At center (300s), duration_score ≈ 0.5
        let signals = SessionSignals {
            tool_success_rate: 0.5,
            avg_confidence: 0.5,
            duration_secs: 300.0,
            decision_count: 5,
            tasks_completed: 0,
            tasks_total: 0,
        };
        let reward = computer.compute(&signals);
        // With default weights: 0.3*0.5 + 0.3*0.5 + 0.2*0.5 + 0.1*0.5 + 0.1*0.5 = 0.5
        assert!(
            (reward - 0.5).abs() < 0.05,
            "Expected ~0.5 at center, got {}",
            reward
        );
    }

    #[test]
    fn test_duration_sigmoid_short_session() {
        let computer = SessionRewardComputer::default();
        // Very short session (10s) → duration score near 0
        let signals = SessionSignals {
            tool_success_rate: 1.0,
            avg_confidence: 1.0,
            duration_secs: 10.0,
            decision_count: 10,
            tasks_completed: 5,
            tasks_total: 5,
        };
        let reward_short = computer.compute(&signals);

        // Long session (600s) → duration score near 1.0
        let signals_long = SessionSignals {
            duration_secs: 600.0,
            ..signals.clone()
        };
        let reward_long = computer.compute(&signals_long);

        assert!(
            reward_long > reward_short,
            "Longer session should score higher on duration: short={}, long={}",
            reward_short,
            reward_long
        );
    }

    #[test]
    fn test_from_decisions_all_success() {
        let decisions = vec![
            make_decision("code.search", 0.9, true),
            make_decision("note.get_context", 0.8, true),
            make_decision("code.analyze_impact", 0.85, true),
        ];
        let signals = SessionSignals::from_decisions(&decisions, Duration::from_secs(120), 1, 2);

        assert!((signals.tool_success_rate - 1.0).abs() < 1e-10);
        assert!((signals.avg_confidence - 0.85).abs() < 1e-10);
        assert_eq!(signals.decision_count, 3);
        assert_eq!(signals.tasks_completed, 1);
        assert_eq!(signals.tasks_total, 2);
    }

    #[test]
    fn test_from_decisions_mixed_success() {
        let decisions = vec![
            make_decision("code.search", 0.9, true),
            make_decision("code.search", 0.3, false),
        ];
        let signals = SessionSignals::from_decisions(&decisions, Duration::from_secs(60), 0, 0);

        assert!((signals.tool_success_rate - 0.5).abs() < 1e-10);
        assert!((signals.avg_confidence - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_from_decisions_empty() {
        let signals = SessionSignals::from_decisions(&[], Duration::from_secs(10), 0, 0);

        assert_eq!(signals.tool_success_rate, 0.0);
        assert_eq!(signals.avg_confidence, 0.0);
        assert_eq!(signals.decision_count, 0);
    }

    #[test]
    fn test_no_tasks_gives_neutral_task_score() {
        let computer = SessionRewardComputer::default();
        // No tasks → task_score = 0.5 (neutral, doesn't penalize)
        let with_tasks = SessionSignals {
            tool_success_rate: 0.8,
            avg_confidence: 0.8,
            duration_secs: 300.0,
            decision_count: 10,
            tasks_completed: 0,
            tasks_total: 0, // no tasks
        };
        let without_tasks = SessionSignals {
            tasks_completed: 1,
            tasks_total: 2, // 50% = same as neutral
            ..with_tasks.clone()
        };
        let r1 = computer.compute(&with_tasks);
        let r2 = computer.compute(&without_tasks);
        assert!(
            (r1 - r2).abs() < 0.01,
            "No tasks (neutral=0.5) should ≈ 50% completion: {} vs {}",
            r1,
            r2
        );
    }

    #[test]
    fn test_config_weights_are_respected() {
        // Set only tool_success weight to 1.0, all others to 0.0
        let config = RewardHeuristicConfig {
            weight_tool_success: 1.0,
            weight_task_completion: 0.0,
            weight_confidence: 0.0,
            weight_duration: 0.0,
            weight_decision_count: 0.0,
            ..Default::default()
        };
        let computer = SessionRewardComputer::new(config);

        let signals = SessionSignals {
            tool_success_rate: 0.75,
            avg_confidence: 0.0,
            duration_secs: 0.0,
            decision_count: 1,
            tasks_completed: 0,
            tasks_total: 0,
        };
        let reward = computer.compute(&signals);
        assert!(
            (reward - 0.75).abs() < 1e-10,
            "With only tool_success weight=1.0, reward should be 0.75, got {}",
            reward
        );
    }
}
