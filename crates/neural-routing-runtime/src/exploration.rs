//! ExplorationScheduler — adaptive exploration/exploitation trade-off.
//!
//! Mechanisms:
//! 1. **ε-greedy adaptatif**: ε = max(ε_min, ε_base × (1 - model_confidence))
//! 2. **Thompson Sampling**: Beta(α, β) posteriors per action
//! 3. **Info-gain bonus**: entropy-based bonus for unexplored graph regions
//!
//! The scheduler decides whether a query should explore (try new routes)
//! or exploit (use the best known route).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// ExplorationScheduler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationConfig {
    /// Base ε for ε-greedy (initial value).
    pub epsilon_base: f64,
    /// Minimum ε (floor — always explore this fraction).
    pub epsilon_min: f64,
    /// Decay rate for ε over trajectories.
    pub epsilon_decay: f64,
    /// Info-gain bonus weight (β).
    pub info_gain_beta: f64,
    /// Weight reduction for exploratory trajectories in training (0.0 - 1.0).
    pub exploratory_weight: f64,
    /// Thompson Sampling prior α (successes).
    pub ts_prior_alpha: f64,
    /// Thompson Sampling prior β (failures).
    pub ts_prior_beta: f64,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self {
            epsilon_base: 0.3,
            epsilon_min: 0.05,
            epsilon_decay: 0.001,
            info_gain_beta: 0.1,
            exploratory_weight: 0.7,
            ts_prior_alpha: 1.0,
            ts_prior_beta: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Thompson Sampling state
// ---------------------------------------------------------------------------

/// Beta distribution parameters for one action (Thompson Sampling).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPosterior {
    /// Number of successes (reward > threshold).
    pub alpha: f64,
    /// Number of failures (reward ≤ threshold).
    pub beta: f64,
    /// Number of times this action was selected.
    pub selection_count: usize,
}

impl ActionPosterior {
    fn new(prior_alpha: f64, prior_beta: f64) -> Self {
        Self {
            alpha: prior_alpha,
            beta: prior_beta,
            selection_count: 0,
        }
    }

    /// Update posterior with observed reward.
    fn update(&mut self, success: bool) {
        self.selection_count += 1;
        if success {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
    }

    /// Mean of the Beta distribution: α / (α + β).
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Variance of the Beta distribution.
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        (self.alpha * self.beta) / (ab * ab * (ab + 1.0))
    }

    /// Sample from the Beta distribution using the Jöhnk algorithm.
    /// Uses a simple deterministic seed for reproducibility.
    fn sample(&self, seed: u64) -> f64 {
        // Use a simple pseudo-random approach for Beta sampling.
        // For production, consider a proper RNG.
        beta_sample(self.alpha, self.beta, seed)
    }
}

// ---------------------------------------------------------------------------
// ExplorationScheduler
// ---------------------------------------------------------------------------

/// Exploration decision for a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationDecision {
    /// Whether this query should explore.
    pub should_explore: bool,
    /// Current ε value.
    pub epsilon: f64,
    /// Info-gain bonus for the top-ranked route.
    pub info_gain_bonus: f64,
    /// Thompson Sampling recommended action (if exploring).
    pub ts_recommendation: Option<String>,
    /// Reason for the decision.
    pub reason: ExplorationReason,
}

/// Why exploration was recommended (or not).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ExplorationReason {
    /// ε-greedy random exploration.
    EpsilonGreedy,
    /// Thompson Sampling selected an under-explored action.
    ThompsonSampling,
    /// Info-gain bonus for unexplored region.
    InfoGain,
    /// Exploiting the best known route.
    Exploit,
}

/// ExplorationScheduler — manages the exploration/exploitation trade-off.
pub struct ExplorationScheduler {
    config: ExplorationConfig,
    /// Thompson Sampling posteriors per action key.
    posteriors: HashMap<String, ActionPosterior>,
    /// Total trajectory count (for ε decay).
    trajectory_count: usize,
    /// Actions seen per region (for info-gain).
    region_visits: HashMap<String, usize>,
    /// Total region visits.
    total_region_visits: usize,
}

impl ExplorationScheduler {
    pub fn new(config: ExplorationConfig) -> Self {
        Self {
            posteriors: HashMap::new(),
            trajectory_count: 0,
            region_visits: HashMap::new(),
            total_region_visits: 0,
            config,
        }
    }

    /// Get the current ε value (decayed over time).
    pub fn current_epsilon(&self) -> f64 {
        let decayed = self.config.epsilon_base
            * (-self.config.epsilon_decay * self.trajectory_count as f64).exp();
        decayed.max(self.config.epsilon_min)
    }

    /// Get the current ε adapted to model confidence.
    pub fn adapted_epsilon(&self, model_confidence: f64) -> f64 {
        let base_eps = self.current_epsilon();
        // Higher confidence → less exploration
        (base_eps * (1.0 - model_confidence)).max(self.config.epsilon_min)
    }

    /// Decide whether to explore for a given query.
    ///
    /// - `model_confidence`: overall model confidence for this query (0-1).
    /// - `session_hash`: deterministic hash for consistent routing.
    /// - `candidate_actions`: actions available for this query.
    pub fn decide(
        &self,
        model_confidence: f64,
        session_hash: u64,
        candidate_actions: &[String],
    ) -> ExplorationDecision {
        let epsilon = self.adapted_epsilon(model_confidence);

        // Deterministic ε-greedy check
        let eps_bucket = (session_hash % 10000) as f64 / 10000.0;
        if eps_bucket < epsilon {
            // Explore! Use Thompson Sampling to pick the action
            let ts_action = self.thompson_sample(candidate_actions, session_hash);
            return ExplorationDecision {
                should_explore: true,
                epsilon,
                info_gain_bonus: 0.0,
                ts_recommendation: ts_action,
                reason: ExplorationReason::EpsilonGreedy,
            };
        }

        // Check info-gain: are there under-explored actions?
        let max_info_gain = self.compute_info_gain(candidate_actions);
        if max_info_gain > 0.5 && self.config.info_gain_beta > 0.0 {
            let ts_action = self.thompson_sample(candidate_actions, session_hash);
            return ExplorationDecision {
                should_explore: true,
                epsilon,
                info_gain_bonus: max_info_gain * self.config.info_gain_beta,
                ts_recommendation: ts_action,
                reason: ExplorationReason::InfoGain,
            };
        }

        // Exploit
        ExplorationDecision {
            should_explore: false,
            epsilon,
            info_gain_bonus: 0.0,
            ts_recommendation: None,
            reason: ExplorationReason::Exploit,
        }
    }

    /// Thompson Sampling: sample from each action's posterior and pick the best.
    fn thompson_sample(&self, actions: &[String], seed: u64) -> Option<String> {
        if actions.is_empty() {
            return None;
        }

        let mut best_action = None;
        let mut best_sample = f64::NEG_INFINITY;

        for (i, action) in actions.iter().enumerate() {
            let posterior = self.posteriors.get(action).cloned().unwrap_or_else(|| {
                ActionPosterior::new(self.config.ts_prior_alpha, self.config.ts_prior_beta)
            });

            let sample = posterior.sample(seed.wrapping_add(i as u64));
            if sample > best_sample {
                best_sample = sample;
                best_action = Some(action.clone());
            }
        }

        best_action
    }

    /// Compute info-gain bonus: entropy-based measure of how much we'd learn
    /// from exploring each action.
    fn compute_info_gain(&self, actions: &[String]) -> f64 {
        if actions.is_empty() || self.total_region_visits == 0 {
            return 1.0; // Maximum info-gain when no data
        }

        let mut max_gain = 0.0f64;
        for action in actions {
            let visits = *self.region_visits.get(action).unwrap_or(&0);
            if visits == 0 {
                return 1.0; // Completely unexplored action
            }
            let p = visits as f64 / self.total_region_visits as f64;
            let entropy = -p * p.ln();
            if entropy > max_gain {
                max_gain = entropy;
            }
        }

        max_gain
    }

    /// Update the scheduler after observing a trajectory result.
    pub fn observe_result(&mut self, action_key: &str, reward: f64, success_threshold: f64) {
        self.trajectory_count += 1;

        // Update Thompson Sampling posterior
        let posterior = self
            .posteriors
            .entry(action_key.to_string())
            .or_insert_with(|| {
                ActionPosterior::new(self.config.ts_prior_alpha, self.config.ts_prior_beta)
            });
        posterior.update(reward > success_threshold);

        // Update region visits
        *self
            .region_visits
            .entry(action_key.to_string())
            .or_insert(0) += 1;
        self.total_region_visits += 1;
    }

    /// Get Thompson Sampling posteriors for debugging/visualization.
    pub fn posteriors(&self) -> &HashMap<String, ActionPosterior> {
        &self.posteriors
    }

    /// Get the total trajectory count.
    pub fn trajectory_count(&self) -> usize {
        self.trajectory_count
    }

    /// Get the exploratory weight for training (reduced weight for exploratory trajectories).
    pub fn exploratory_weight(&self) -> f64 {
        self.config.exploratory_weight
    }
}

// ---------------------------------------------------------------------------
// Beta distribution sampling (Jöhnk's algorithm + Cheng's method)
// ---------------------------------------------------------------------------

/// Simple pseudo-random Beta sample using a deterministic seed.
/// Uses the inverse CDF method with a SplitMix64-based uniform generator.
fn beta_sample(alpha: f64, beta: f64, seed: u64) -> f64 {
    // For alpha = beta = 1: uniform
    if (alpha - 1.0).abs() < 1e-10 && (beta - 1.0).abs() < 1e-10 {
        return splitmix_uniform(seed);
    }

    // For integer alpha, beta ≤ 20: use order statistics of uniforms
    if alpha <= 20.0 && beta <= 20.0 && alpha == alpha.floor() && beta == beta.floor() {
        return beta_sample_order_stats(alpha as usize, beta as usize, seed);
    }

    // General case: use the mean as a deterministic approximation.
    // This is a simplification — in production, use a proper PRNG.
    // Adding variance based on seed for some stochasticity.
    let mean = alpha / (alpha + beta);
    let variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
    let std = variance.sqrt();

    // Simple perturbation
    let u = splitmix_uniform(seed);
    let z = inv_normal_approx(u);
    (mean + std * z).clamp(0.001, 0.999)
}

/// Beta sample via order statistics: Beta(a, b) = a-th order statistic of (a+b-1) uniforms.
fn beta_sample_order_stats(alpha: usize, beta: usize, seed: u64) -> f64 {
    let n = alpha + beta - 1;
    let mut values: Vec<f64> = (0..n)
        .map(|i| splitmix_uniform(seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15))))
        .collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values.get(alpha - 1).copied().unwrap_or(0.5)
}

/// SplitMix64-based uniform random in [0, 1).
fn splitmix_uniform(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z = z ^ (z >> 31);
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Simple inverse normal CDF approximation (Abramowitz & Stegun).
fn inv_normal_approx(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    // Rational approximation
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 {
        -result
    } else {
        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_epsilon() {
        let scheduler = ExplorationScheduler::new(ExplorationConfig::default());
        let eps = scheduler.current_epsilon();
        assert!((eps - 0.3).abs() < 1e-6, "Initial ε should be 0.3");
    }

    #[test]
    fn test_epsilon_decay() {
        let mut scheduler = ExplorationScheduler::new(ExplorationConfig {
            epsilon_base: 0.3,
            epsilon_min: 0.05,
            epsilon_decay: 0.01,
            ..Default::default()
        });

        // Simulate many trajectories
        for _ in 0..500 {
            scheduler.observe_result("code.search", 0.8, 0.5);
        }

        let eps = scheduler.current_epsilon();
        assert!(eps < 0.3, "ε should have decayed from 0.3, got {}", eps);
        assert!(
            eps >= 0.05,
            "ε should not go below minimum 0.05, got {}",
            eps
        );
    }

    #[test]
    fn test_adapted_epsilon() {
        let scheduler = ExplorationScheduler::new(ExplorationConfig::default());

        let eps_low_conf = scheduler.adapted_epsilon(0.1);
        let eps_high_conf = scheduler.adapted_epsilon(0.9);

        assert!(
            eps_low_conf > eps_high_conf,
            "Low confidence should give higher ε: {} vs {}",
            eps_low_conf,
            eps_high_conf
        );
    }

    #[test]
    fn test_decide_exploit_with_high_confidence() {
        let mut scheduler = ExplorationScheduler::new(ExplorationConfig {
            epsilon_base: 0.0, // Zero ε — never explore via epsilon-greedy
            epsilon_min: 0.0,
            ..Default::default()
        });

        // Observe results so info-gain doesn't override with "unexplored" bonus
        for _ in 0..20 {
            scheduler.observe_result("code.search", 0.9, 0.5);
            scheduler.observe_result("note.create", 0.7, 0.5);
        }

        let actions = vec!["code.search".to_string(), "note.create".to_string()];
        let decision = scheduler.decide(0.95, 42, &actions);

        // With ε=0, high confidence, and well-explored actions, should exploit
        assert!(
            !decision.should_explore,
            "Expected exploit, got {:?}",
            decision.reason
        );
    }

    #[test]
    fn test_decide_explore_with_high_epsilon() {
        let scheduler = ExplorationScheduler::new(ExplorationConfig {
            epsilon_base: 1.0, // Always explore
            epsilon_min: 1.0,
            ..Default::default()
        });

        let actions = vec!["code.search".to_string()];
        let decision = scheduler.decide(0.0, 42, &actions);

        assert!(decision.should_explore);
        assert_eq!(decision.reason, ExplorationReason::EpsilonGreedy);
    }

    #[test]
    fn test_thompson_sampling_updates() {
        let mut scheduler = ExplorationScheduler::new(ExplorationConfig::default());

        // Observe successes for action A
        for _ in 0..50 {
            scheduler.observe_result("action_A", 0.9, 0.5);
        }

        // Observe failures for action B
        for _ in 0..50 {
            scheduler.observe_result("action_B", 0.2, 0.5);
        }

        let posteriors = scheduler.posteriors();
        let a = posteriors.get("action_A").unwrap();
        let b = posteriors.get("action_B").unwrap();

        assert!(
            a.mean() > b.mean(),
            "Action A (successes) should have higher posterior mean than B (failures): {} vs {}",
            a.mean(),
            b.mean()
        );
    }

    #[test]
    fn test_info_gain_for_unexplored() {
        let scheduler = ExplorationScheduler::new(ExplorationConfig {
            info_gain_beta: 1.0,
            epsilon_base: 0.0, // Disable ε-greedy
            epsilon_min: 0.0,
            ..Default::default()
        });

        let actions = vec!["totally_new_action".to_string()];
        let decision = scheduler.decide(0.5, 42, &actions);

        // New actions should trigger info-gain exploration
        assert!(
            decision.should_explore,
            "Unexplored action should trigger exploration"
        );
        assert_eq!(decision.reason, ExplorationReason::InfoGain);
    }

    #[test]
    fn test_splitmix_uniform_range() {
        for seed in 0..1000u64 {
            let u = splitmix_uniform(seed);
            assert!(u >= 0.0 && u < 1.0, "Uniform sample out of range: {}", u);
        }
    }

    #[test]
    fn test_beta_sample_range() {
        for seed in 0..100u64 {
            let s = beta_sample(2.0, 5.0, seed);
            assert!(s > 0.0 && s < 1.0, "Beta sample out of range: {}", s);
        }
    }

    #[test]
    fn test_exploration_rate_converges() {
        // With many trajectories, exploration rate should decrease
        let config = ExplorationConfig {
            epsilon_base: 0.3,
            epsilon_min: 0.05,
            epsilon_decay: 0.01,
            ..Default::default()
        };
        let mut scheduler = ExplorationScheduler::new(config);

        let actions = vec!["code.search".to_string()];
        let mut explore_count_early = 0;
        let mut explore_count_late = 0;

        // Early phase
        for i in 0..100 {
            let decision = scheduler.decide(0.5, i, &actions);
            if decision.should_explore {
                explore_count_early += 1;
            }
            scheduler.observe_result("code.search", 0.7, 0.5);
        }

        // Late phase (after many more trajectories)
        for _ in 0..900 {
            scheduler.observe_result("code.search", 0.7, 0.5);
        }

        for i in 0..100 {
            let decision = scheduler.decide(0.5, i + 1000, &actions);
            if decision.should_explore {
                explore_count_late += 1;
            }
        }

        assert!(
            explore_count_late <= explore_count_early,
            "Late phase should explore less: early={}, late={}",
            explore_count_early,
            explore_count_late
        );
    }
}
