//! Decision Transformer — GPT-2 style sequence model for trajectory prediction.
//!
//! Architecture (Janner et al., 2021 adapted for MCP routing):
//! - 3 embedding heads: RTG → hidden, state → hidden, action → hidden
//! - Positional encoding: learned embeddings for timestep positions
//! - N causal transformer blocks (LN → CausalAttn → LN → MLP with GELU)
//! - Action prediction head with tanh bounding
//!
//! The sequence is interleaved: [RTG_0, state_0, action_0, RTG_1, state_1, action_1, ...]
//! Each timestep contributes 3 tokens. Causal masking ensures autoregressive prediction.
//!
//! ~3M parameters with default config, optimized for CPU inference (<15ms for 8 steps).

use candle_core::{DType, Device, Module, Result as CandleResult, Tensor, D};
use candle_nn::{layer_norm, linear, Activation, LayerNorm, Linear, VarBuilder};
#[cfg(test)]
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use crate::dataset::{ACTION_DIM, STATE_DIM};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Decision Transformer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTransformerConfig {
    /// State dimension (512d: query_embedding + context_embedding).
    pub state_dim: usize,
    /// Action dimension (256d: context_embedding).
    pub action_dim: usize,
    /// Hidden dimension for transformer layers.
    pub hidden_dim: usize,
    /// Number of transformer blocks.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Maximum trajectory length (in timesteps, not tokens).
    /// Each timestep = 3 tokens (RTG, state, action).
    pub max_timesteps: usize,
    /// Dropout rate (applied during training only).
    pub dropout: f64,
}

impl Default for DecisionTransformerConfig {
    fn default() -> Self {
        Self {
            state_dim: STATE_DIM,   // 512
            action_dim: ACTION_DIM, // 256
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 4,
            max_timesteps: 32,
            dropout: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// CausalSelfAttention
// ---------------------------------------------------------------------------

/// Multi-head causal self-attention (GPT-2 style).
///
/// Implements Q·K^T / √d_k with causal (lower-triangular) masking.
struct CausalSelfAttention {
    /// Combined QKV projection [hidden_dim → 3 * hidden_dim].
    qkv: Linear,
    /// Output projection [hidden_dim → hidden_dim].
    out_proj: Linear,
    /// Number of attention heads.
    num_heads: usize,
    /// Per-head dimension.
    head_dim: usize,
}

impl CausalSelfAttention {
    fn new(hidden_dim: usize, num_heads: usize, vb: VarBuilder<'_>) -> CandleResult<Self> {
        assert!(
            hidden_dim.is_multiple_of(num_heads),
            "hidden_dim must be divisible by num_heads"
        );
        let head_dim = hidden_dim / num_heads;
        let qkv = linear(hidden_dim, 3 * hidden_dim, vb.pp("qkv"))?;
        let out_proj = linear(hidden_dim, hidden_dim, vb.pp("out"))?;

        Ok(Self {
            qkv,
            out_proj,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass.
    ///
    /// x: [batch_size, seq_len, hidden_dim]
    /// Returns: [batch_size, seq_len, hidden_dim]
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, seq_len, _hidden) = x.dims3()?;

        // QKV projection: [B, T, 3*H]
        let qkv = self.qkv.forward(x)?;

        // Split into Q, K, V: each [B, T, H]
        let q = qkv.narrow(2, 0, self.num_heads * self.head_dim)?;
        let k = qkv.narrow(2, self.num_heads * self.head_dim, self.num_heads * self.head_dim)?;
        let v = qkv.narrow(
            2,
            2 * self.num_heads * self.head_dim,
            self.num_heads * self.head_dim,
        )?;

        // Reshape to [B, num_heads, T, head_dim] — contiguous for matmul
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Attention scores: Q·K^T / √d_k → [B, num_heads, T, T]
        let scale = (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn_weights = q.matmul(&k_t)?.affine(1.0 / scale, 0.0)?;

        // Causal mask: lower-triangular (future tokens masked with -inf)
        let mask = create_causal_mask(seq_len, attn_weights.device())?;
        let attn_weights = attn_weights.broadcast_add(&mask)?;

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        // Weighted sum: [B, num_heads, T, head_dim]
        let out = attn_weights.matmul(&v)?;

        // Reshape back to [B, T, hidden_dim]
        let out = out
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        self.out_proj.forward(&out)
    }
}

/// Create a causal (lower-triangular) attention mask.
///
/// Returns a [1, 1, T, T] tensor where future positions are -1e9 and past/present are 0.
fn create_causal_mask(seq_len: usize, device: &Device) -> CandleResult<Tensor> {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = -1e9;
        }
    }
    Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)
}

// ---------------------------------------------------------------------------
// TransformerBlock
// ---------------------------------------------------------------------------

/// A single transformer block: LN → Attention → LN → MLP (GELU).
struct TransformerBlock {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    mlp_fc: Linear,
    mlp_proj: Linear,
}

impl TransformerBlock {
    fn new(hidden_dim: usize, num_heads: usize, vb: VarBuilder<'_>) -> CandleResult<Self> {
        let ln1 = layer_norm(hidden_dim, candle_nn::LayerNormConfig::default(), vb.pp("ln1"))?;
        let attn = CausalSelfAttention::new(hidden_dim, num_heads, vb.pp("attn"))?;
        let ln2 = layer_norm(hidden_dim, candle_nn::LayerNormConfig::default(), vb.pp("ln2"))?;
        let ffn_dim = hidden_dim * 4; // Standard GPT-2: 4x expansion
        let mlp_fc = linear(hidden_dim, ffn_dim, vb.pp("mlp_fc"))?;
        let mlp_proj = linear(ffn_dim, hidden_dim, vb.pp("mlp_proj"))?;

        Ok(Self {
            ln1,
            attn,
            ln2,
            mlp_fc,
            mlp_proj,
        })
    }

    /// Forward pass with residual connections.
    ///
    /// x: [B, T, H] → [B, T, H]
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Pre-LN attention with residual
        let normed = self.ln1.forward(x)?;
        let attn_out = self.attn.forward(&normed)?;
        let x = (x + attn_out)?;

        // Pre-LN MLP with residual
        let normed = self.ln2.forward(&x)?;
        let h = self.mlp_fc.forward(&normed)?;
        let h = h.apply(&Activation::Gelu)?;
        let mlp_out = self.mlp_proj.forward(&h)?;
        let x = (x + mlp_out)?;

        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Decision Transformer
// ---------------------------------------------------------------------------

/// Decision Transformer — sequence model for trajectory prediction.
///
/// Following Janner et al. (2021), the input sequence is interleaved:
/// `[RTG_0, state_0, action_0, RTG_1, state_1, action_1, ...]`
///
/// Each timestep contributes 3 tokens. The model predicts the action at each timestep
/// conditioned on the return-to-go (desired future reward).
pub struct DecisionTransformer {
    /// Configuration.
    pub config: DecisionTransformerConfig,
    /// RTG embedding: 1 → hidden_dim.
    rtg_embed: Linear,
    /// State embedding: state_dim → hidden_dim.
    state_embed: Linear,
    /// Action embedding: action_dim → hidden_dim.
    action_embed: Linear,
    /// Learned positional embeddings: [3 * max_timesteps, hidden_dim].
    /// 3 tokens per timestep (RTG, state, action).
    pos_embed: Tensor,
    /// LayerNorm applied after embedding.
    embed_ln: LayerNorm,
    /// Transformer blocks.
    blocks: Vec<TransformerBlock>,
    /// Final LayerNorm.
    final_ln: LayerNorm,
    /// Action prediction head: hidden_dim → action_dim.
    action_head: Linear,
}

impl DecisionTransformer {
    /// Create a new Decision Transformer with random weights.
    pub fn new(config: DecisionTransformerConfig, vb: VarBuilder<'_>) -> CandleResult<Self> {
        let h = config.hidden_dim;

        // Embedding heads
        let rtg_embed = linear(1, h, vb.pp("rtg_embed"))?;
        let state_embed = linear(config.state_dim, h, vb.pp("state_embed"))?;
        let action_embed = linear(config.action_dim, h, vb.pp("action_embed"))?;

        // Positional embeddings: 3 tokens per timestep
        let max_tokens = 3 * config.max_timesteps;
        let pos_embed = vb.get_with_hints(
            (1, max_tokens, h),
            "pos_embed",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;

        let embed_ln = layer_norm(h, candle_nn::LayerNormConfig::default(), vb.pp("embed_ln"))?;

        // Transformer blocks
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let block =
                TransformerBlock::new(h, config.num_heads, vb.pp(format!("block_{}", i)))?;
            blocks.push(block);
        }

        let final_ln = layer_norm(h, candle_nn::LayerNormConfig::default(), vb.pp("final_ln"))?;

        // Action head: hidden → action_dim
        let action_head = linear(h, config.action_dim, vb.pp("action_head"))?;

        Ok(Self {
            config,
            rtg_embed,
            state_embed,
            action_embed,
            pos_embed,
            embed_ln,
            blocks,
            final_ln,
            action_head,
        })
    }

    /// Count total parameters.
    pub fn param_count(&self) -> usize {
        let h = self.config.hidden_dim;
        let s = self.config.state_dim;
        let a = self.config.action_dim;
        let n = self.config.num_layers;
        let max_tok = 3 * self.config.max_timesteps;

        // Embedding heads (weight + bias)
        let rtg = h + h;
        let state = s * h + h;
        let action = a * h + h;
        let pos = max_tok * h;
        let embed_ln_params = 2 * h; // gamma + beta

        // Per transformer block
        let qkv = h * 3 * h + 3 * h;
        let out = h * h + h;
        let ln_params = 2 * h; // per LN
        let ffn = h * 4 * h + 4 * h + 4 * h * h + h;
        let block_params = qkv + out + 2 * ln_params + ffn;

        // Final LN + action head
        let final_ln_params = 2 * h;
        let head = h * a + a;

        rtg + state + action + pos + embed_ln_params + n * block_params + final_ln_params + head
    }

    /// Forward pass for training.
    ///
    /// - `returns_to_go`: [B, T] — desired future return at each timestep
    /// - `states`: [B, T, state_dim] — state at each timestep
    /// - `actions`: [B, T, action_dim] — action taken at each timestep
    /// - `timesteps`: [B, T] — timestep indices (u32)
    /// - `attention_mask`: [B, T] — 1.0 for real tokens, 0.0 for padding
    ///
    /// Returns: predicted actions [B, T, action_dim] bounded in [-1, 1] via tanh.
    pub fn forward(
        &self,
        returns_to_go: &Tensor,
        states: &Tensor,
        actions: &Tensor,
        _timesteps: &Tensor,
        attention_mask: &Tensor,
    ) -> CandleResult<Tensor> {
        let (batch_size, seq_len, _) = states.dims3()?;

        // Embed each modality: [B, T, H]
        let rtg_tokens = self
            .rtg_embed
            .forward(&returns_to_go.unsqueeze(D::Minus1)?)?;
        let state_tokens = self.state_embed.forward(states)?;
        let action_tokens = self.action_embed.forward(actions)?;

        // Interleave: [RTG_0, state_0, action_0, RTG_1, state_1, action_1, ...]
        // Shape: [B, 3*T, H]
        let token_seq = interleave_tokens(&rtg_tokens, &state_tokens, &action_tokens)?;
        let total_len = 3 * seq_len;

        // Add positional embeddings (truncate to actual length)
        let pos = self.pos_embed.narrow(1, 0, total_len)?;
        let token_seq = token_seq.broadcast_add(&pos)?;

        // Embed LayerNorm
        let mut h = self.embed_ln.forward(&token_seq)?;

        // Build attention mask for the interleaved sequence [B, 3*T]
        // Each timestep mask value applies to all 3 of its tokens
        let interleaved_mask = expand_mask(attention_mask, seq_len)?;

        // Zero out padding tokens
        let mask_3d = interleaved_mask.unsqueeze(D::Minus1)?; // [B, 3*T, 1]
        h = h.broadcast_mul(&mask_3d)?;

        // Transformer blocks
        for block in &self.blocks {
            h = block.forward(&h)?;
            // Re-apply mask after each block to prevent padding leakage
            h = h.broadcast_mul(&mask_3d)?;
        }

        // Final LN
        h = self.final_ln.forward(&h)?;

        // Extract state token positions (index 1, 4, 7, ...) for action prediction
        // In the interleaved sequence, state tokens are at positions 3*t + 1
        let state_positions: Vec<i64> = (0..seq_len as i64).map(|t| 3 * t + 1).collect();
        let pos_tensor =
            Tensor::from_vec(state_positions, seq_len, h.device())?.unsqueeze(0)?; // [1, T]
        let pos_tensor = pos_tensor
            .broadcast_as((batch_size, seq_len))?
            .contiguous()?; // [B, T]

        // Gather state hidden representations
        let h_flat = h.reshape((batch_size, total_len, self.config.hidden_dim))?;
        let state_hiddens = batch_index_select(&h_flat, &pos_tensor)?; // [B, T, H]

        // Action prediction with tanh bounding
        let action_preds = self.action_head.forward(&state_hiddens)?;
        let action_preds = action_preds.tanh()?;

        Ok(action_preds)
    }

    /// Autoregressive inference: generate a route given RTG and initial state.
    ///
    /// - `rtg_target`: desired total return (scalar)
    /// - `initial_state`: [state_dim] — initial state embedding
    /// - `max_steps`: maximum number of steps to generate
    ///
    /// Returns: Vec of predicted action vectors [action_dim].
    pub fn generate(
        &self,
        rtg_target: f32,
        initial_state: &Tensor,
        max_steps: usize,
    ) -> CandleResult<Vec<Tensor>> {
        let device = initial_state.device();
        let max_steps = max_steps.min(self.config.max_timesteps);

        let mut actions = Vec::with_capacity(max_steps);
        let mut rtg_remaining = rtg_target;

        // Initialize with zeros for first action (will be predicted)
        let zero_action =
            Tensor::zeros((1, 1, self.config.action_dim), DType::F32, device)?;
        let state = initial_state.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, state_dim]

        for step in 0..max_steps {
            let t = step + 1;

            // Build RTG tensor: [1, t]
            let mut rtg_vec = Vec::with_capacity(t);
            // Re-compute RTG: target minus accumulated predicted rewards
            // (simplified: just decay linearly)
            for s in 0..t {
                let rtg_at_s = rtg_remaining * ((t - s) as f32 / t as f32);
                rtg_vec.push(rtg_at_s);
            }
            let rtg = Tensor::from_vec(rtg_vec, (1, t), device)?;

            // Build states: repeat initial_state for now (real inference would update)
            let states = state.broadcast_as((1, t, self.config.state_dim))?.contiguous()?;

            // Build actions: previous predicted actions + zero for current
            let action_list: Vec<Tensor> = actions
                .iter()
                .map(|a: &Tensor| a.unsqueeze(0).unwrap().unsqueeze(0).unwrap())
                .collect();
            let mut all_actions = if action_list.is_empty() {
                zero_action.broadcast_as((1, 1, self.config.action_dim))?.contiguous()?
            } else {
                let prev = Tensor::cat(&action_list, 1)?;
                Tensor::cat(&[prev, zero_action.clone()], 1)?
            };
            // Truncate/pad to t tokens
            if all_actions.dim(1)? > t {
                all_actions = all_actions.narrow(1, 0, t)?;
            }

            // Timesteps and mask
            let timesteps_vec: Vec<u32> = (0..t as u32).collect();
            let timesteps = Tensor::from_vec(timesteps_vec, (1, t), device)?;
            let mask = Tensor::ones((1, t), DType::F32, device)?;

            // Forward pass
            let pred_actions = self.forward(&rtg, &states, &all_actions, &timesteps, &mask)?;

            // Take the last predicted action
            let pred_action = pred_actions.narrow(1, t - 1, 1)?.squeeze(1)?; // [1, action_dim]
            let pred_action = pred_action.squeeze(0)?; // [action_dim]

            // Simple reward estimate: small decrement per step
            rtg_remaining -= rtg_target / max_steps as f32;

            actions.push(pred_action);
        }

        Ok(actions)
    }
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

/// Interleave 3 tensors: [B, T, H] each → [B, 3*T, H].
///
/// Result: [a_0, b_0, c_0, a_1, b_1, c_1, ...]
fn interleave_tokens(a: &Tensor, b: &Tensor, c: &Tensor) -> CandleResult<Tensor> {
    let (batch_size, seq_len, hidden) = a.dims3()?;

    // Stack along a new dim: [B, T, 3, H]
    let a = a.unsqueeze(2)?; // [B, T, 1, H]
    let b = b.unsqueeze(2)?;
    let c = c.unsqueeze(2)?;
    let stacked = Tensor::cat(&[a, b, c], 2)?; // [B, T, 3, H]

    // Reshape to [B, 3*T, H]
    stacked.reshape((batch_size, 3 * seq_len, hidden))
}

/// Expand a [B, T] mask to [B, 3*T] by repeating each value 3 times.
fn expand_mask(mask: &Tensor, seq_len: usize) -> CandleResult<Tensor> {
    let batch_size = mask.dim(0)?;
    // [B, T] → [B, T, 1] → broadcast [B, T, 3] → reshape [B, 3*T]
    let expanded = mask.unsqueeze(D::Minus1)?; // [B, T, 1]
    let expanded = expanded.broadcast_as((batch_size, seq_len, 3))?.contiguous()?;
    expanded.reshape((batch_size, 3 * seq_len))
}

/// Batch index select: gather specific positions from a sequence.
///
/// src: [B, L, D], indices: [B, K] → output: [B, K, D]
fn batch_index_select(src: &Tensor, indices: &Tensor) -> CandleResult<Tensor> {
    let (batch_size, _seq_len, _dim) = src.dims3()?;
    let _k = indices.dim(1)?;

    let mut rows = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let src_b = src.get(b)?; // [L, D]
        let idx_b = indices.get(b)?; // [K]
        let selected = src_b.index_select(&idx_b, 0)?; // [K, D]
        rows.push(selected);
    }

    Tensor::stack(&rows, 0) // [B, K, D]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model() -> (DecisionTransformer, VarMap) {
        let config = DecisionTransformerConfig {
            state_dim: STATE_DIM, // 512
            action_dim: ACTION_DIM, // 256
            hidden_dim: 128,
            num_layers: 2,
            num_heads: 4,
            max_timesteps: 16,
            dropout: 0.0,
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model = DecisionTransformer::new(config, vb).unwrap();
        (model, varmap)
    }

    #[test]
    fn test_causal_mask() -> CandleResult<()> {
        let mask = create_causal_mask(4, &Device::Cpu)?;
        let vals = mask.squeeze(0)?.squeeze(0)?.to_vec2::<f32>()?;

        // Diagonal and below should be 0
        assert!((vals[0][0]).abs() < 1e-6);
        assert!((vals[1][0]).abs() < 1e-6);
        assert!((vals[1][1]).abs() < 1e-6);
        assert!((vals[2][2]).abs() < 1e-6);

        // Above diagonal should be -1e9
        assert!(vals[0][1] < -1e8);
        assert!(vals[0][3] < -1e8);
        assert!(vals[2][3] < -1e8);

        Ok(())
    }

    #[test]
    fn test_interleave_tokens() -> CandleResult<()> {
        let device = Device::Cpu;
        let a = Tensor::ones((2, 3, 4), DType::F32, &device)?;
        let b = Tensor::full(2.0f32, (2, 3, 4), &device)?;
        let c = Tensor::full(3.0f32, (2, 3, 4), &device)?;

        let result = interleave_tokens(&a, &b, &c)?;
        assert_eq!(result.dims(), &[2, 9, 4]);

        // Check interleaving: positions 0,3,6 should be 1.0 (from a)
        let vals = result.get(0)?.to_vec2::<f32>()?;
        assert!((vals[0][0] - 1.0).abs() < 1e-6); // a[0]
        assert!((vals[1][0] - 2.0).abs() < 1e-6); // b[0]
        assert!((vals[2][0] - 3.0).abs() < 1e-6); // c[0]
        assert!((vals[3][0] - 1.0).abs() < 1e-6); // a[1]

        Ok(())
    }

    #[test]
    fn test_decision_transformer_forward() -> CandleResult<()> {
        let (model, _varmap) = make_model();
        let device = Device::Cpu;
        let batch_size = 2;
        let seq_len = 5;

        let rtg = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len), &device)?;
        let states = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, seq_len, STATE_DIM),
            &device,
        )?;
        let actions = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, seq_len, ACTION_DIM),
            &device,
        )?;
        let timesteps = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
        let mask = Tensor::ones((batch_size, seq_len), DType::F32, &device)?;

        let pred_actions = model.forward(&rtg, &states, &actions, &timesteps, &mask)?;

        // Output shape: [B, T, action_dim]
        assert_eq!(pred_actions.dims(), &[batch_size, seq_len, ACTION_DIM]);

        // Values should be bounded in [-1, 1] (tanh)
        let flat = pred_actions.flatten_all()?.to_vec1::<f32>()?;
        for v in &flat {
            assert!(
                *v >= -1.0 && *v <= 1.0,
                "tanh output should be in [-1,1], got {}",
                v
            );
        }

        Ok(())
    }

    #[test]
    fn test_param_count() {
        let (model, _) = make_model();
        let count = model.param_count();

        // With hidden=128, 2 layers, state=512, action=256:
        // Should be roughly 500K-2M params
        assert!(
            count > 100_000 && count < 5_000_000,
            "Expected param count 100K-5M, got {}",
            count
        );
    }

    #[test]
    fn test_decision_transformer_with_padding() -> CandleResult<()> {
        let (model, _varmap) = make_model();
        let device = Device::Cpu;

        let rtg = Tensor::randn(0.0f32, 1.0, (1, 8), &device)?;
        let states = Tensor::randn(0.0f32, 1.0, (1, 8, STATE_DIM), &device)?;
        let actions = Tensor::randn(0.0f32, 1.0, (1, 8, ACTION_DIM), &device)?;
        let timesteps = Tensor::zeros((1, 8), DType::U32, &device)?;

        // Mask: first 5 tokens real, last 3 padding
        let mask_data = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let mask = Tensor::from_vec(mask_data, (1, 8), &device)?;

        let pred_actions = model.forward(&rtg, &states, &actions, &timesteps, &mask)?;
        assert_eq!(pred_actions.dims(), &[1, 8, ACTION_DIM]);

        Ok(())
    }

    #[test]
    fn test_autoregressive_generation() -> CandleResult<()> {
        let (model, _varmap) = make_model();
        let device = Device::Cpu;

        let initial_state = Tensor::randn(0.0f32, 1.0, STATE_DIM, &device)?;
        let actions = model.generate(10.0, &initial_state, 4)?;

        assert_eq!(actions.len(), 4);
        for (i, a) in actions.iter().enumerate() {
            assert_eq!(
                a.dims(),
                &[ACTION_DIM],
                "Action {} should have dim {}",
                i,
                ACTION_DIM
            );
            // Check bounded
            let vals = a.to_vec1::<f32>()?;
            for v in &vals {
                assert!(*v >= -1.0 && *v <= 1.0);
            }
        }

        Ok(())
    }

    #[test]
    fn test_expand_mask() -> CandleResult<()> {
        let device = Device::Cpu;
        let mask = Tensor::from_vec(vec![1.0f32, 1.0, 0.0], (1, 3), &device)?;
        let expanded = expand_mask(&mask, 3)?;

        assert_eq!(expanded.dims(), &[1, 9]);
        let vals = expanded.to_vec2::<f32>()?;
        // Each value repeated 3 times
        assert!((vals[0][0] - 1.0).abs() < 1e-6);
        assert!((vals[0][1] - 1.0).abs() < 1e-6);
        assert!((vals[0][2] - 1.0).abs() < 1e-6);
        assert!((vals[0][3] - 1.0).abs() < 1e-6);
        assert!((vals[0][6]).abs() < 1e-6); // padding
        assert!((vals[0][7]).abs() < 1e-6);
        assert!((vals[0][8]).abs() < 1e-6);

        Ok(())
    }
}
