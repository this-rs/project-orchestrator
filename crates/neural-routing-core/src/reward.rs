//! Reward decomposition — credit assignment strategies for trajectory steps.
//!
//! Given a trajectory with a total RBCR reward, decompose it into per-step rewards
//! to enable step-level learning.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::models::TrajectoryNode;

/// Trait for reward decomposition strategies.
#[async_trait]
pub trait RewardStrategy: Send + Sync {
    /// Decompose a total reward into per-step rewards.
    /// Returns a Vec of (node_index, local_reward) tuples.
    async fn decompose(&self, nodes: &[TrajectoryNode], total_reward: f64) -> Result<Vec<f64>>;

    /// Name of this strategy (for logging/config).
    fn name(&self) -> &str;
}

/// Configuration for the reward decomposer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardConfig {
    /// Which strategy to use: "td", "hindsight", "attention"
    pub strategy: String,
    /// Discount factor for TD strategy (default: 0.99)
    pub gamma: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            strategy: "td".to_string(),
            gamma: 0.99,
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy 1: Temporal Difference (TD)
// ---------------------------------------------------------------------------

/// TD reward decomposition: r_t = R_final * (confidence_t / sum(confidence))
///
/// Steps with higher confidence get more credit. Simple but effective baseline.
pub struct TDRewardStrategy {
    pub gamma: f64,
}

impl TDRewardStrategy {
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }
}

impl Default for TDRewardStrategy {
    fn default() -> Self {
        Self::new(0.99)
    }
}

#[async_trait]
impl RewardStrategy for TDRewardStrategy {
    async fn decompose(&self, nodes: &[TrajectoryNode], total_reward: f64) -> Result<Vec<f64>> {
        if nodes.is_empty() {
            return Ok(vec![]);
        }

        let confidence_sum: f64 = nodes.iter().map(|n| n.confidence).sum();
        if confidence_sum < 1e-10 {
            // Equal distribution if all confidences are zero
            let equal = total_reward / nodes.len() as f64;
            return Ok(vec![equal; nodes.len()]);
        }

        // Apply discount factor: later steps are discounted
        let n = nodes.len();
        let rewards: Vec<f64> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let discount = self.gamma.powi((n - 1 - i) as i32);
                let weight = node.confidence / confidence_sum;
                total_reward * weight * discount
            })
            .collect();

        // Renormalize so sum matches total_reward
        let sum: f64 = rewards.iter().sum();
        if sum.abs() < 1e-10 {
            return Ok(vec![total_reward / n as f64; n]);
        }
        let scale = total_reward / sum;
        Ok(rewards.iter().map(|r| r * scale).collect())
    }

    fn name(&self) -> &str {
        "td"
    }
}

// ---------------------------------------------------------------------------
// Strategy 2: Hindsight
// ---------------------------------------------------------------------------

/// Hindsight reward: measures the improvement in context quality after each decision.
///
/// Approximated by: r_t = confidence_t * (1 / alternatives_count) — decisions with
/// few alternatives but high confidence get more credit (clear, decisive choices).
pub struct HindsightRewardStrategy;

#[async_trait]
impl RewardStrategy for HindsightRewardStrategy {
    async fn decompose(&self, nodes: &[TrajectoryNode], total_reward: f64) -> Result<Vec<f64>> {
        if nodes.is_empty() {
            return Ok(vec![]);
        }

        // Score: confidence * decisiveness (1/alternatives)
        let raw_scores: Vec<f64> = nodes
            .iter()
            .map(|n| {
                let decisiveness = 1.0 / (n.alternatives_count.max(1) as f64);
                n.confidence * decisiveness
            })
            .collect();

        let sum: f64 = raw_scores.iter().sum();
        if sum < 1e-10 {
            let equal = total_reward / nodes.len() as f64;
            return Ok(vec![equal; nodes.len()]);
        }

        Ok(raw_scores.iter().map(|s| total_reward * s / sum).collect())
    }

    fn name(&self) -> &str {
        "hindsight"
    }
}

// ---------------------------------------------------------------------------
// Strategy 3: Attention-based
// ---------------------------------------------------------------------------

/// Attention-based reward: uses time-weighted attention to credit assignment.
///
/// Steps that took longer (more deliberation) and had higher confidence get more credit.
/// Approximation of a learned attention network, without requiring training.
pub struct AttentionRewardStrategy;

#[async_trait]
impl RewardStrategy for AttentionRewardStrategy {
    async fn decompose(&self, nodes: &[TrajectoryNode], total_reward: f64) -> Result<Vec<f64>> {
        if nodes.is_empty() {
            return Ok(vec![]);
        }

        // Attention weight: softmax(confidence * log(1 + delta_ms))
        let max_delta = nodes.iter().map(|n| n.delta_ms).max().unwrap_or(1).max(1) as f64;

        let raw_scores: Vec<f64> = nodes
            .iter()
            .map(|n| {
                let time_weight = (1.0 + n.delta_ms as f64).ln() / (1.0 + max_delta).ln();
                n.confidence * (0.5 + 0.5 * time_weight)
            })
            .collect();

        // Softmax normalization
        let max_score = raw_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = raw_scores.iter().map(|s| (s - max_score).exp()).collect();
        let exp_sum: f64 = exp_scores.iter().sum();

        if exp_sum < 1e-10 {
            let equal = total_reward / nodes.len() as f64;
            return Ok(vec![equal; nodes.len()]);
        }

        Ok(exp_scores
            .iter()
            .map(|e| total_reward * e / exp_sum)
            .collect())
    }

    fn name(&self) -> &str {
        "attention"
    }
}

/// Create a reward strategy from its name.
pub fn create_reward_strategy(config: &RewardConfig) -> Box<dyn RewardStrategy> {
    match config.strategy.as_str() {
        "td" => Box::new(TDRewardStrategy::new(config.gamma)),
        "hindsight" => Box::new(HindsightRewardStrategy),
        "attention" => Box::new(AttentionRewardStrategy),
        _ => {
            tracing::warn!(
                strategy = %config.strategy,
                "Unknown reward strategy, falling back to TD"
            );
            Box::new(TDRewardStrategy::new(config.gamma))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_nodes(count: usize) -> Vec<TrajectoryNode> {
        (0..count)
            .map(|i| TrajectoryNode {
                id: Uuid::new_v4(),
                context_embedding: vec![],
                action_type: format!("action_{}", i),
                action_params: serde_json::Value::Null,
                alternatives_count: 3,
                chosen_index: 0,
                confidence: 0.5 + (i as f64) * 0.1,
                local_reward: 0.0,
                cumulative_reward: 0.0,
                delta_ms: 100 * (i as u64 + 1),
                order: i,
            })
            .collect()
    }

    #[tokio::test]
    async fn test_td_sum_matches_total() {
        let strategy = TDRewardStrategy::default();
        let nodes = make_nodes(5);
        let total = 1.0;

        let rewards = strategy.decompose(&nodes, total).await.unwrap();
        let sum: f64 = rewards.iter().sum();
        assert!(
            (sum - total).abs() < 0.05,
            "TD sum {} should be ~{} (within 5%)",
            sum,
            total
        );
    }

    #[tokio::test]
    async fn test_hindsight_sum_matches_total() {
        let strategy = HindsightRewardStrategy;
        let nodes = make_nodes(5);
        let total = 1.0;

        let rewards = strategy.decompose(&nodes, total).await.unwrap();
        let sum: f64 = rewards.iter().sum();
        assert!(
            (sum - total).abs() < 0.01,
            "Hindsight sum {} should be ~{}",
            sum,
            total
        );
    }

    #[tokio::test]
    async fn test_attention_sum_matches_total() {
        let strategy = AttentionRewardStrategy;
        let nodes = make_nodes(5);
        let total = 1.0;

        let rewards = strategy.decompose(&nodes, total).await.unwrap();
        let sum: f64 = rewards.iter().sum();
        assert!(
            (sum - total).abs() < 0.01,
            "Attention sum {} should be ~{}",
            sum,
            total
        );
    }

    #[tokio::test]
    async fn test_hindsight_non_uniform() {
        let strategy = HindsightRewardStrategy;
        let mut nodes = make_nodes(3);
        // First node: high confidence, few alternatives → should get more reward
        nodes[0].confidence = 0.95;
        nodes[0].alternatives_count = 1;
        // Last node: low confidence, many alternatives → should get less
        nodes[2].confidence = 0.3;
        nodes[2].alternatives_count = 10;

        let rewards = strategy.decompose(&nodes, 1.0).await.unwrap();
        assert!(
            rewards[0] > rewards[2],
            "High-confidence decisive node ({}) should get more reward than low-confidence indecisive ({})",
            rewards[0], rewards[2]
        );
    }

    #[tokio::test]
    async fn test_empty_nodes() {
        let strategy = TDRewardStrategy::default();
        let rewards = strategy.decompose(&[], 1.0).await.unwrap();
        assert!(rewards.is_empty());
    }

    #[tokio::test]
    async fn test_create_reward_strategy_td() {
        let config = RewardConfig {
            strategy: "td".to_string(),
            gamma: 0.99,
        };
        let strategy = create_reward_strategy(&config);
        assert_eq!(strategy.name(), "td");

        let nodes = make_nodes(3);
        let rewards = strategy.decompose(&nodes, 1.0).await.unwrap();
        assert_eq!(rewards.len(), 3);
        let sum: f64 = rewards.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.05,
            "TD rewards should sum to ~1.0, got {}",
            sum
        );
    }

    #[tokio::test]
    async fn test_create_reward_strategy_hindsight() {
        let config = RewardConfig {
            strategy: "hindsight".to_string(),
            gamma: 0.99,
        };
        let strategy = create_reward_strategy(&config);
        assert_eq!(strategy.name(), "hindsight");

        let nodes = make_nodes(3);
        let rewards = strategy.decompose(&nodes, 1.0).await.unwrap();
        assert_eq!(rewards.len(), 3);
        let sum: f64 = rewards.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Hindsight rewards should sum to ~1.0, got {}",
            sum
        );
    }

    #[tokio::test]
    async fn test_create_reward_strategy_attention() {
        let config = RewardConfig {
            strategy: "attention".to_string(),
            gamma: 0.99,
        };
        let strategy = create_reward_strategy(&config);
        assert_eq!(strategy.name(), "attention");

        let nodes = make_nodes(3);
        let rewards = strategy.decompose(&nodes, 1.0).await.unwrap();
        assert_eq!(rewards.len(), 3);
        let sum: f64 = rewards.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Attention rewards should sum to ~1.0, got {}",
            sum
        );
    }

    #[tokio::test]
    async fn test_create_reward_strategy_unknown_fallback() {
        let config = RewardConfig {
            strategy: "nonexistent_strategy".to_string(),
            gamma: 0.95,
        };
        let strategy = create_reward_strategy(&config);
        // Should fall back to TD
        assert_eq!(strategy.name(), "td");

        let nodes = make_nodes(3);
        let rewards = strategy.decompose(&nodes, 1.0).await.unwrap();
        assert_eq!(rewards.len(), 3);
    }

    #[tokio::test]
    async fn test_td_zero_confidences() {
        let strategy = TDRewardStrategy::default();
        let nodes: Vec<TrajectoryNode> = (0..4)
            .map(|i| TrajectoryNode {
                id: Uuid::new_v4(),
                context_embedding: vec![],
                action_type: format!("action_{}", i),
                action_params: serde_json::Value::Null,
                alternatives_count: 3,
                chosen_index: 0,
                confidence: 0.0, // all zero
                local_reward: 0.0,
                cumulative_reward: 0.0,
                delta_ms: 100,
                order: i,
            })
            .collect();

        let total = 2.0;
        let rewards = strategy.decompose(&nodes, total).await.unwrap();
        assert_eq!(rewards.len(), 4);
        // With all-zero confidences, should distribute equally
        for r in &rewards {
            assert!(
                (r - 0.5).abs() < 0.01,
                "Each node should get equal reward (~0.5), got {}",
                r
            );
        }
        let sum: f64 = rewards.iter().sum();
        assert!((sum - total).abs() < 0.01, "Sum should equal total reward");
    }

    #[tokio::test]
    async fn test_single_node_trajectory() {
        let td = TDRewardStrategy::default();
        let hindsight = HindsightRewardStrategy;
        let attention = AttentionRewardStrategy;

        let nodes = vec![TrajectoryNode {
            id: Uuid::new_v4(),
            context_embedding: vec![],
            action_type: "only_action".to_string(),
            action_params: serde_json::Value::Null,
            alternatives_count: 2,
            chosen_index: 0,
            confidence: 0.9,
            local_reward: 0.0,
            cumulative_reward: 0.0,
            delta_ms: 200,
            order: 0,
        }];

        let total = 1.0;

        // All strategies should assign 100% of reward to the single node
        let td_rewards = td.decompose(&nodes, total).await.unwrap();
        assert_eq!(td_rewards.len(), 1);
        assert!(
            (td_rewards[0] - total).abs() < 0.01,
            "TD: single node should get 100% reward, got {}",
            td_rewards[0]
        );

        let hs_rewards = hindsight.decompose(&nodes, total).await.unwrap();
        assert_eq!(hs_rewards.len(), 1);
        assert!(
            (hs_rewards[0] - total).abs() < 0.01,
            "Hindsight: single node should get 100% reward, got {}",
            hs_rewards[0]
        );

        let att_rewards = attention.decompose(&nodes, total).await.unwrap();
        assert_eq!(att_rewards.len(), 1);
        assert!(
            (att_rewards[0] - total).abs() < 0.01,
            "Attention: single node should get 100% reward, got {}",
            att_rewards[0]
        );
    }
}
