//! CQL (Conservative Q-Learning) — offline RL policy.
//!
//! Fallback when the Decision Transformer is OOD (out-of-distribution).
//! Conservative: penalizes Q-values for unseen state-action pairs via
//! logsumexp regularization (Kumar et al., 2020).
//!
//! Architecture:
//! - Double Q-networks (MLP) with soft target updates (Polyak averaging)
//! - Policy network (MLP) that outputs continuous action vectors
//! - CQL loss: L_bellman + α · E[logsumexp(Q(s, a_rand)) - E[Q(s, a_data)]]
//!
//! This is the conservative alternative to the Decision Transformer — more robust
//! on small datasets (<5K trajectories) at the cost of less expressiveness.

use candle_core::{DType, Device, Module, Result as CandleResult, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::dataset::{ACTION_DIM, STATE_DIM};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the CQL agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CQLConfig {
    /// State dimension (512d).
    pub state_dim: usize,
    /// Action dimension (256d).
    pub action_dim: usize,
    /// Hidden dimension for MLPs.
    pub hidden_dim: usize,
    /// CQL alpha — conservatism strength. Higher = more conservative.
    pub alpha: f64,
    /// Discount factor gamma.
    pub gamma: f64,
    /// Soft target update rate (Polyak averaging).
    pub tau: f64,
    /// Number of random actions for CQL logsumexp estimation.
    pub num_random_actions: usize,
}

impl Default for CQLConfig {
    fn default() -> Self {
        Self {
            state_dim: STATE_DIM,
            action_dim: ACTION_DIM,
            hidden_dim: 256,
            alpha: 1.0,
            gamma: 0.99,
            tau: 0.005,
            num_random_actions: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// MLP (shared building block)
// ---------------------------------------------------------------------------

/// Simple MLP with ReLU activations.
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    layers: Vec<Linear>,
}

impl MLP {
    /// Create an MLP: input_dim → hidden → hidden → output_dim.
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        vb: VarBuilder<'_>,
    ) -> CandleResult<Self> {
        let l1 = linear(input_dim, hidden_dim, vb.pp("l1"))?;
        let l2 = linear(hidden_dim, hidden_dim, vb.pp("l2"))?;
        let l3 = linear(hidden_dim, output_dim, vb.pp("l3"))?;
        Ok(Self {
            layers: vec![l1, l2, l3],
        })
    }

    /// Forward pass with ReLU between hidden layers.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut h = self.layers[0].forward(x)?;
        h = h.relu()?;
        h = self.layers[1].forward(&h)?;
        h = h.relu()?;
        h = self.layers[2].forward(&h)?;
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// CQL Policy
// ---------------------------------------------------------------------------

/// CQL agent with double Q-networks and a policy network.
pub struct CQLPolicy {
    pub config: CQLConfig,
    /// Q-network 1: (state, action) → Q-value (scalar).
    q1: MLP,
    /// Q-network 2: (state, action) → Q-value (scalar).
    q2: MLP,
    /// Target Q-network 1 (soft-updated copy of q1).
    q1_target: MLP,
    /// Target Q-network 2 (soft-updated copy of q2).
    q2_target: MLP,
    /// Policy network: state → action (tanh-bounded).
    policy: MLP,
    /// VarMap for Q-networks.
    q_varmap: VarMap,
    /// VarMap for target Q-networks.
    q_target_varmap: VarMap,
    /// VarMap for policy network.
    policy_varmap: VarMap,
}

impl CQLPolicy {
    /// Create a new CQL agent with random weights.
    pub fn new(config: CQLConfig) -> CandleResult<Self> {
        let device = Device::Cpu;
        let sa_dim = config.state_dim + config.action_dim; // Q input = concat(state, action)

        // Q-networks
        let q_varmap = VarMap::new();
        let q_vb = VarBuilder::from_varmap(&q_varmap, DType::F32, &device);
        let q1 = MLP::new(sa_dim, config.hidden_dim, 1, q_vb.pp("q1"))?;
        let q2 = MLP::new(sa_dim, config.hidden_dim, 1, q_vb.pp("q2"))?;

        // Target Q-networks (separate VarMap for frozen params)
        let q_target_varmap = VarMap::new();
        let qt_vb = VarBuilder::from_varmap(&q_target_varmap, DType::F32, &device);
        let q1_target = MLP::new(sa_dim, config.hidden_dim, 1, qt_vb.pp("q1"))?;
        let q2_target = MLP::new(sa_dim, config.hidden_dim, 1, qt_vb.pp("q2"))?;

        // Initialize targets as copies of Q-networks
        copy_params(&q_varmap, &q_target_varmap)?;

        // Policy network
        let policy_varmap = VarMap::new();
        let p_vb = VarBuilder::from_varmap(&policy_varmap, DType::F32, &device);
        let policy = MLP::new(
            config.state_dim,
            config.hidden_dim,
            config.action_dim,
            p_vb.pp("policy"),
        )?;

        info!(
            "CQL agent initialized: state_dim={}, action_dim={}, hidden={}, alpha={:.2}",
            config.state_dim, config.action_dim, config.hidden_dim, config.alpha
        );

        Ok(Self {
            config,
            q1,
            q2,
            q1_target,
            q2_target,
            policy,
            q_varmap,
            q_target_varmap,
            policy_varmap,
        })
    }

    /// Compute Q-value for a (state, action) pair using both Q-networks.
    ///
    /// Returns (q1_value, q2_value) as [B, 1] tensors.
    pub fn q_values(&self, states: &Tensor, actions: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let sa = Tensor::cat(&[states, actions], D::Minus1)?;
        let q1 = self.q1.forward(&sa)?;
        let q2 = self.q2.forward(&sa)?;
        Ok((q1, q2))
    }

    /// Compute target Q-value (minimum of both targets — conservative).
    pub fn target_q_value(&self, states: &Tensor, actions: &Tensor) -> CandleResult<Tensor> {
        let sa = Tensor::cat(&[states, actions], D::Minus1)?;
        let q1 = self.q1_target.forward(&sa)?;
        let q2 = self.q2_target.forward(&sa)?;
        // Conservative: take the minimum
        q1.minimum(&q2)
    }

    /// Predict action from current policy.
    ///
    /// Returns action [B, action_dim] bounded in [-1, 1] via tanh.
    pub fn predict_action(&self, states: &Tensor) -> CandleResult<Tensor> {
        let raw = self.policy.forward(states)?;
        raw.tanh()
    }

    /// Compute CQL loss components.
    ///
    /// - `states`: [B, state_dim]
    /// - `actions`: [B, action_dim]
    /// - `rewards`: [B, 1]
    /// - `next_states`: [B, state_dim]
    /// - `dones`: [B, 1] (1.0 if terminal, 0.0 otherwise)
    ///
    /// Returns (total_loss, bellman_loss, cql_penalty) as scalars.
    pub fn compute_loss(
        &self,
        states: &Tensor,
        actions: &Tensor,
        rewards: &Tensor,
        next_states: &Tensor,
        dones: &Tensor,
    ) -> CandleResult<(Tensor, f32, f32)> {
        let device = states.device();
        let batch_size = states.dim(0)?;

        // --- Bellman loss ---
        // Target: r + γ * (1 - done) * min(Q1_target, Q2_target)(s', π(s'))
        let next_actions = self.predict_action(next_states)?;
        let target_q = self.target_q_value(next_states, &next_actions)?;
        let not_done = dones.affine(-1.0, 1.0)?; // 1 - done
        let target = (rewards + (not_done * target_q)?.affine(self.config.gamma, 0.0)?)?;
        let target = target.detach(); // stop gradient

        let (q1, q2) = self.q_values(states, actions)?;
        let bellman_1 = ((&q1 - &target)?.sqr())?.mean_all()?;
        let bellman_2 = ((&q2 - &target)?.sqr())?.mean_all()?;
        let bellman_loss = (bellman_1 + bellman_2)?;

        // --- CQL regularization ---
        // logsumexp(Q(s, a_random)) - E[Q(s, a_data)]
        let n_rand = self.config.num_random_actions;
        let mut random_q_values = Vec::with_capacity(n_rand);

        for _i in 0..n_rand {
            // Random actions in [-1, 1]
            let rand_actions =
                Tensor::randn(0.0f32, 1.0, (batch_size, self.config.action_dim), device)?;
            let rand_actions = rand_actions.tanh()?;
            let (rq1, rq2) = self.q_values(states, &rand_actions)?;
            // Take min of both Q-networks
            let rq = rq1.minimum(&rq2)?;
            random_q_values.push(rq);
        }

        // Stack: [B, n_rand]
        let random_qs = Tensor::cat(&random_q_values, D::Minus1)?;

        // logsumexp over random actions
        let max_q = random_qs.max(D::Minus1)?.unsqueeze(D::Minus1)?;
        let shifted = (random_qs.clone() - max_q.broadcast_as(random_qs.shape())?)?;
        let logsumexp = (shifted.exp()?.sum(D::Minus1)?.log()? + max_q.squeeze(D::Minus1)?)?;
        let logsumexp_mean = logsumexp.mean_all()?;

        // E[Q(s, a_data)] — Q-value on dataset actions
        let data_q = q1.minimum(&q2)?;
        let data_q_mean = data_q.mean_all()?;

        // CQL penalty
        let cql_penalty = (logsumexp_mean - data_q_mean)?;

        // Total loss
        let total = (bellman_loss.clone() + cql_penalty.affine(self.config.alpha, 0.0)?)?;

        let bellman_val = bellman_loss.to_scalar::<f32>()?;
        let cql_val = cql_penalty.to_scalar::<f32>()?;

        Ok((total, bellman_val, cql_val))
    }

    /// Soft update target networks: θ_target = τ·θ + (1-τ)·θ_target.
    pub fn soft_update_targets(&self) -> CandleResult<()> {
        let tau = self.config.tau;
        let q_data = self.q_varmap.data().lock().unwrap();
        let target_data = self.q_target_varmap.data().lock().unwrap();

        for (name, t_var) in target_data.iter() {
            if let Some(q_var) = q_data.get(name) {
                let q_tensor = q_var.as_tensor();
                let t_tensor = t_var.as_tensor();
                let updated = (q_tensor.affine(tau, 0.0)? + t_tensor.affine(1.0 - tau, 0.0)?)?;
                t_var.set(&updated)?;
            }
        }

        Ok(())
    }

    /// Get references to VarMaps for external optimizer use.
    pub fn varmaps(&self) -> (&VarMap, &VarMap) {
        (&self.q_varmap, &self.policy_varmap)
    }

    /// Save all weights.
    pub fn save(&self, dir: &std::path::Path) -> CandleResult<()> {
        self.q_varmap.save(dir.join("cql_q.safetensors"))?;
        self.q_target_varmap
            .save(dir.join("cql_q_target.safetensors"))?;
        self.policy_varmap
            .save(dir.join("cql_policy.safetensors"))?;
        Ok(())
    }

    /// Load weights.
    pub fn load(&mut self, dir: &std::path::Path) -> CandleResult<()> {
        self.q_varmap.load(dir.join("cql_q.safetensors"))?;
        self.q_target_varmap
            .load(dir.join("cql_q_target.safetensors"))?;
        self.policy_varmap
            .load(dir.join("cql_policy.safetensors"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Copy all parameters from source to target VarMap (matched by name).
fn copy_params(src: &VarMap, dst: &VarMap) -> CandleResult<()> {
    let src_data = src.data().lock().unwrap();
    let dst_data = dst.data().lock().unwrap();
    for (name, dst_var) in dst_data.iter() {
        if let Some(src_var) = src_data.get(name) {
            dst_var.set(&src_var.as_tensor().clone())?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cql() -> CQLPolicy {
        let config = CQLConfig {
            state_dim: STATE_DIM,
            action_dim: ACTION_DIM,
            hidden_dim: 64,
            alpha: 1.0,
            gamma: 0.99,
            tau: 0.005,
            num_random_actions: 5,
        };
        CQLPolicy::new(config).unwrap()
    }

    #[test]
    fn test_cql_q_values() -> CandleResult<()> {
        let cql = make_cql();
        let device = Device::Cpu;
        let b = 4;

        let states = Tensor::randn(0.0f32, 1.0, (b, STATE_DIM), &device)?;
        let actions = Tensor::randn(0.0f32, 0.5, (b, ACTION_DIM), &device)?.tanh()?;

        let (q1, q2) = cql.q_values(&states, &actions)?;
        assert_eq!(q1.dims(), &[b, 1]);
        assert_eq!(q2.dims(), &[b, 1]);

        // Q-values should be finite
        let q1_val = q1.flatten_all()?.to_vec1::<f32>()?;
        for v in &q1_val {
            assert!(v.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_cql_predict_action() -> CandleResult<()> {
        let cql = make_cql();
        let device = Device::Cpu;

        let states = Tensor::randn(0.0f32, 1.0, (2, STATE_DIM), &device)?;
        let actions = cql.predict_action(&states)?;

        assert_eq!(actions.dims(), &[2, ACTION_DIM]);

        // Tanh-bounded
        let vals = actions.flatten_all()?.to_vec1::<f32>()?;
        for v in &vals {
            assert!(
                *v >= -1.0 && *v <= 1.0,
                "Action should be in [-1,1], got {}",
                v
            );
        }

        Ok(())
    }

    #[test]
    fn test_cql_loss_computation() -> CandleResult<()> {
        let cql = make_cql();
        let device = Device::Cpu;
        let b = 8;

        let states = Tensor::randn(0.0f32, 1.0, (b, STATE_DIM), &device)?;
        let actions = Tensor::randn(0.0f32, 0.5, (b, ACTION_DIM), &device)?.tanh()?;
        let rewards = Tensor::randn(0.0f32, 1.0, (b, 1), &device)?;
        let next_states = Tensor::randn(0.0f32, 1.0, (b, STATE_DIM), &device)?;
        let dones = Tensor::zeros((b, 1), DType::F32, &device)?;

        let (total, bellman, cql_pen) =
            cql.compute_loss(&states, &actions, &rewards, &next_states, &dones)?;

        let total_val = total.to_scalar::<f32>()?;
        assert!(total_val.is_finite(), "Total loss should be finite");
        assert!(bellman.is_finite(), "Bellman loss should be finite");
        assert!(cql_pen.is_finite(), "CQL penalty should be finite");
        assert!(bellman >= 0.0, "Bellman loss should be non-negative");

        Ok(())
    }

    #[test]
    fn test_soft_update() -> CandleResult<()> {
        let cql = make_cql();

        // Get initial target params
        let target_before: Vec<f32> = cql
            .q_target_varmap
            .all_vars()
            .first()
            .unwrap()
            .as_tensor()
            .flatten_all()?
            .to_vec1()?;

        // Modify Q params (simulate training step)
        for var in cql.q_varmap.all_vars() {
            let new_val = var.as_tensor().affine(2.0, 0.1)?;
            var.set(&new_val)?;
        }

        // Soft update
        cql.soft_update_targets()?;

        // Target should have moved slightly toward Q
        let target_after: Vec<f32> = cql
            .q_target_varmap
            .all_vars()
            .first()
            .unwrap()
            .as_tensor()
            .flatten_all()?
            .to_vec1()?;

        let diff: f32 = target_before
            .iter()
            .zip(target_after.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "Target params should change after soft update");

        Ok(())
    }

    #[test]
    fn test_target_conservative() -> CandleResult<()> {
        let cql = make_cql();
        let device = Device::Cpu;

        let states = Tensor::randn(0.0f32, 1.0, (3, STATE_DIM), &device)?;
        let actions = Tensor::randn(0.0f32, 0.5, (3, ACTION_DIM), &device)?.tanh()?;

        let target_q = cql.target_q_value(&states, &actions)?;
        let (q1, q2) = cql.q_values(&states, &actions)?;

        // Target Q = min(Q1_target, Q2_target)
        // After initialization, Q and Q_target have same weights,
        // so target should equal min(Q1, Q2)
        let expected_min = q1.minimum(&q2)?;
        let diff = (&target_q - &expected_min)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-4,
            "Initial target should match min(Q1, Q2), diff={}",
            diff
        );

        Ok(())
    }
}
