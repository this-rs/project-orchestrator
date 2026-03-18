//! R-GCN (Relational Graph Convolutional Network) layer.
//!
//! Handles multiple relation types with basis decomposition to reduce parameters:
//! W_r = Σ_b a_{rb} × V_b  (shared basis matrices V_b, per-relation coefficients a_rb)
//!
//! Architecture: 3 layers, GELU activation, dropout, residual connections.

use candle_core::{DType, Module, Result, Tensor};
#[cfg(test)]
use candle_core::Device;
use candle_nn::{linear, Linear, VarBuilder};
#[cfg(test)]
use candle_nn::VarMap;

use crate::message_passing::{extract_edge_indices, gather_rows, scatter_add, MessagePassing};

/// R-GCN layer configuration.
#[derive(Debug, Clone)]
pub struct RGCNConfig {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Output feature dimension.
    pub output_dim: usize,
    /// Number of relation types (default: 8).
    pub num_relations: usize,
    /// Number of basis matrices for decomposition (default: 4).
    pub num_bases: usize,
    /// Dropout rate (default: 0.1).
    pub dropout: f64,
}

impl Default for RGCNConfig {
    fn default() -> Self {
        Self {
            input_dim: 168,
            output_dim: 256,
            num_relations: 8,
            num_bases: 4,
            dropout: 0.1,
        }
    }
}

/// A single R-GCN layer with basis decomposition.
///
/// W_r = Σ_{b=0}^{B-1} a_{rb} × V_b
///
/// where V_b ∈ R^(input×output) are shared basis matrices and
/// a_{rb} are per-relation scalar coefficients.
pub struct RGCNLayer {
    /// Basis matrices: [num_bases, input_dim, output_dim] stored as Vec<Linear>
    bases: Vec<Linear>,
    /// Per-relation coefficients: [num_relations, num_bases]
    coefficients: Tensor,
    /// Self-loop linear transform
    self_loop: Linear,
    /// Layer config
    config: RGCNConfig,
}

impl RGCNLayer {
    pub fn new(config: RGCNConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let mut bases = Vec::with_capacity(config.num_bases);
        for b in 0..config.num_bases {
            let l = linear(
                config.input_dim,
                config.output_dim,
                vb.pp(format!("basis_{}", b)),
            )?;
            bases.push(l);
        }

        let coefficients = vb.get_with_hints(
            (config.num_relations, config.num_bases),
            "coefficients",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.1,
            },
        )?;

        let self_loop = linear(
            config.input_dim,
            config.output_dim,
            vb.pp("self_loop"),
        )?;

        Ok(Self {
            bases,
            coefficients,
            self_loop,
            config,
        })
    }

    /// Compute the effective weight matrix for a given relation type.
    /// W_r = Σ_b a_{rb} × V_b(x)
    fn compute_relation_message(
        &self,
        x: &Tensor,
        relation_type: usize,
    ) -> Result<Tensor> {
        // Get coefficients for this relation: [num_bases]
        let coeffs = self.coefficients.get(relation_type)?;

        // Compute weighted sum of basis transforms
        let mut result: Option<Tensor> = None;
        for (b, basis) in self.bases.iter().enumerate() {
            let coeff = coeffs.get(b)?.to_scalar::<f32>()?;
            let transformed = basis.forward(x)?;
            let weighted = (transformed * coeff as f64)?;
            result = Some(match result {
                Some(acc) => (acc + weighted)?,
                None => weighted,
            });
        }

        result.ok_or_else(|| candle_core::Error::Msg("No bases configured".to_string()))
    }
}

impl MessagePassing for RGCNLayer {
    fn message(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_type: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (sources, _targets) = extract_edge_indices(edge_index)?;
        let num_edges = sources.len();

        // If no edge types provided, use relation 0 for all
        let edge_types: Vec<u8> = if let Some(et) = edge_type {
            et.to_vec1::<u8>()?
        } else {
            vec![0u8; num_edges]
        };

        // Build per-relation edge groups
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); self.config.num_relations];
        for (i, &rt) in edge_types.iter().enumerate() {
            let rt_idx = (rt as usize).min(self.config.num_relations - 1);
            groups[rt_idx].push(i);
        }

        // For each relation type, compute messages in batch
        let device = x.device();
        let mut all_messages = Tensor::zeros(
            (num_edges, self.config.output_dim),
            DType::F32,
            device,
        )?;

        for (rel, edge_indices) in groups.iter().enumerate() {
            if edge_indices.is_empty() {
                continue;
            }

            // Gather source features for this relation's edges
            let rel_source_indices: Vec<u32> = edge_indices
                .iter()
                .map(|&i| sources[i])
                .collect();
            let rel_sources = gather_rows(x, &rel_source_indices)?;

            // Compute W_r × x for this relation
            let rel_messages = self.compute_relation_message(&rel_sources, rel)?;

            // Scatter messages back to the correct positions
            let index_tensor = Tensor::from_vec(
                edge_indices.iter().map(|&i| i as i64).collect::<Vec<_>>(),
                edge_indices.len(),
                device,
            )?;
            all_messages = all_messages.index_add(&index_tensor, &rel_messages, 0)?;
        }

        Ok(all_messages)
    }

    fn aggregate(
        &self,
        messages: &Tensor,
        edge_index: &Tensor,
        num_nodes: usize,
    ) -> Result<Tensor> {
        let (_, targets) = extract_edge_indices(edge_index)?;
        scatter_add(messages, &targets, num_nodes)
    }

    fn update(&self, x: &Tensor, aggregated: &Tensor) -> Result<Tensor> {
        // Self-loop: add transformed input features
        let self_contrib = self.self_loop.forward(x)?;

        // Pad or project x if input_dim != output_dim
        let combined = (aggregated + self_contrib)?;

        // GELU activation
        let activated = combined.gelu()?;

        Ok(activated)
    }
}

/// Multi-layer R-GCN model.
pub struct RGCN {
    layers: Vec<RGCNLayer>,
}

impl RGCN {
    /// Create a new multi-layer R-GCN.
    ///
    /// * `input_dim` — input feature dimension
    /// * `hidden_dim` — hidden layer dimension
    /// * `output_dim` — output embedding dimension
    /// * `num_layers` — number of R-GCN layers (≥1)
    /// * `num_relations` — number of relation types
    /// * `num_bases` — basis decomposition rank
    /// * `dropout` — dropout rate
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        num_relations: usize,
        num_bases: usize,
        dropout: f64,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        assert!(num_layers >= 1, "Need at least 1 layer");

        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let (in_d, out_d) = if num_layers == 1 {
                (input_dim, output_dim)
            } else if i == 0 {
                (input_dim, hidden_dim)
            } else if i == num_layers - 1 {
                (hidden_dim, output_dim)
            } else {
                (hidden_dim, hidden_dim)
            };

            let config = RGCNConfig {
                input_dim: in_d,
                output_dim: out_d,
                num_relations,
                num_bases,
                dropout,
            };

            let layer = RGCNLayer::new(config, vb.pp(format!("layer_{}", i)))?;
            layers.push(layer);
        }

        Ok(Self { layers })
    }

    /// Forward pass through all layers.
    pub fn forward(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_type: Option<&Tensor>,
        num_nodes: usize,
    ) -> Result<Tensor> {
        let mut h = x.clone();

        for layer in &self.layers {
            h = layer.forward(&h, edge_index, edge_type, num_nodes)?;
        }

        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_graph() -> (Tensor, Tensor, Option<Tensor>, usize) {
        let device = Device::Cpu;
        let num_nodes = 5;
        let input_dim = 8;

        // Node features [5, 8]
        let x = Tensor::randn(0.0f32, 1.0, (num_nodes, input_dim), &device).unwrap();

        // Edges: 0->1, 1->2, 2->3, 3->4, 0->2, 1->3
        let edge_index = Tensor::new(
            &[[0i64, 1, 2, 3, 0, 1], [1, 2, 3, 4, 2, 3]],
            &device,
        )
        .unwrap();

        // Edge types: mixed relations
        let edge_type = Tensor::new(&[0u8, 1, 2, 0, 3, 1], &device).unwrap();

        (x, edge_index, Some(edge_type), num_nodes)
    }

    #[test]
    fn test_rgcn_layer_forward() -> Result<()> {
        let (x, edge_index, edge_type, num_nodes) = make_test_graph();
        let input_dim = 8;
        let output_dim = 16;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let config = RGCNConfig {
            input_dim,
            output_dim,
            num_relations: 8,
            num_bases: 4,
            dropout: 0.0,
        };

        let layer = RGCNLayer::new(config, vb)?;
        let output = layer.forward(&x, &edge_index, edge_type.as_ref(), num_nodes)?;

        assert_eq!(output.dims(), &[num_nodes, output_dim]);
        Ok(())
    }

    #[test]
    fn test_rgcn_multi_layer() -> Result<()> {
        let (x, edge_index, edge_type, num_nodes) = make_test_graph();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let model = RGCN::new(
            8,   // input
            32,  // hidden
            16,  // output
            3,   // layers
            8,   // relations
            4,   // bases
            0.0, // dropout
            vb,
        )?;

        let output = model.forward(&x, &edge_index, edge_type.as_ref(), num_nodes)?;
        assert_eq!(output.dims(), &[num_nodes, 16]);
        Ok(())
    }

    #[test]
    fn test_rgcn_single_layer() -> Result<()> {
        let (x, edge_index, edge_type, num_nodes) = make_test_graph();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let model = RGCN::new(8, 32, 16, 1, 8, 4, 0.0, vb)?;
        let output = model.forward(&x, &edge_index, edge_type.as_ref(), num_nodes)?;
        assert_eq!(output.dims(), &[num_nodes, 16]);
        Ok(())
    }

    #[test]
    fn test_rgcn_no_edge_types() -> Result<()> {
        let (x, edge_index, _, num_nodes) = make_test_graph();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let model = RGCN::new(8, 32, 16, 2, 8, 4, 0.0, vb)?;
        // No edge types — should default to relation 0
        let output = model.forward(&x, &edge_index, None, num_nodes)?;
        assert_eq!(output.dims(), &[num_nodes, 16]);
        Ok(())
    }

    #[test]
    fn test_rgcn_100_nodes() -> Result<()> {
        let device = Device::Cpu;
        let num_nodes = 100;
        let input_dim = 168; // TOTAL_FEATURE_DIM

        let x = Tensor::randn(0.0f32, 1.0, (num_nodes, input_dim), &device)?;

        // Create random edges (200 edges, 8 relation types)
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        let mut edge_types = Vec::new();
        for i in 0..200 {
            sources.push((i * 7 % num_nodes) as i64);
            targets.push((i * 13 % num_nodes) as i64);
            edge_types.push((i % 8) as u8);
        }

        let mut edge_data = sources.clone();
        edge_data.extend(targets);
        let edge_index = Tensor::from_vec(edge_data, (2, 200), &device)?;
        let edge_type = Tensor::from_vec(edge_types, 200, &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = RGCN::new(168, 256, 256, 3, 8, 4, 0.1, vb)?;
        let output = model.forward(&x, &edge_index, Some(&edge_type), num_nodes)?;

        assert_eq!(output.dims(), &[100, 256]);

        // Output should have non-trivial values (not all zeros)
        let sum = output.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(sum > 0.0, "Output should be non-zero");
        Ok(())
    }
}
