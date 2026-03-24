//! GraphSAGE (SAmple and aggreGatE) layer.
//!
//! Inductive learning — can embed unseen nodes by sampling and aggregating
//! neighbor features. Critical for handling new files/functions added to the graph.
//!
//! Architecture:
//! - Per-relation attention weights (softmax over neighbor types)
//! - Mean aggregation with attention weighting
//! - Residual connections when dimensions match

#[cfg(test)]
use candle_core::{DType, Device};
use candle_core::{Module, Result, Tensor, D};
#[cfg(test)]
use candle_nn::VarMap;
use candle_nn::{linear, Linear, VarBuilder};

use crate::message_passing::{extract_edge_indices, gather_rows, scatter_mean, MessagePassing};

/// GraphSAGE layer configuration.
#[derive(Debug, Clone)]
pub struct GraphSAGEConfig {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Output feature dimension.
    pub output_dim: usize,
    /// Number of relation types (for per-type attention).
    pub num_relations: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Whether to use attention-weighted aggregation.
    pub use_attention: bool,
    /// Whether to L2-normalize output embeddings.
    pub normalize: bool,
}

impl Default for GraphSAGEConfig {
    fn default() -> Self {
        Self {
            input_dim: 169,
            output_dim: 256,
            num_relations: 8,
            dropout: 0.1,
            use_attention: true,
            normalize: true,
        }
    }
}

/// A single GraphSAGE layer with optional relation-type attention.
pub struct GraphSAGELayer {
    /// Linear transform for neighbor features.
    neigh_linear: Linear,
    /// Linear transform for self features.
    self_linear: Linear,
    /// Per-relation attention vectors: [num_relations, input_dim]
    attention_weights: Option<Tensor>,
    config: GraphSAGEConfig,
}

impl GraphSAGELayer {
    pub fn new(config: GraphSAGEConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let neigh_linear = linear(config.input_dim, config.output_dim, vb.pp("neigh"))?;

        let self_linear = linear(config.input_dim, config.output_dim, vb.pp("self"))?;

        let attention_weights = if config.use_attention {
            let attn = vb.get_with_hints(
                (config.num_relations, config.input_dim),
                "attention",
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )?;
            Some(attn)
        } else {
            None
        };

        Ok(Self {
            neigh_linear,
            self_linear,
            attention_weights,
            config,
        })
    }

    /// Compute attention scores for edges based on relation type.
    fn compute_attention(&self, source_features: &Tensor, edge_types: &[u8]) -> Result<Tensor> {
        let attn = self
            .attention_weights
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("No attention weights".to_string()))?;

        let num_edges = edge_types.len();
        let device = source_features.device();

        // Gather attention vectors per edge type
        let type_indices = Tensor::from_vec(
            edge_types.iter().map(|&t| t as i64).collect::<Vec<_>>(),
            num_edges,
            device,
        )?;
        let edge_attn = attn.index_select(&type_indices, 0)?; // [num_edges, input_dim]

        // Dot product between source features and attention vectors
        let scores = (source_features * &edge_attn)?.sum(D::Minus1)?; // [num_edges]

        Ok(scores)
    }
}

impl MessagePassing for GraphSAGELayer {
    fn message(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_type: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (sources, _targets) = extract_edge_indices(edge_index)?;

        // Gather source features
        let source_features = gather_rows(x, &sources)?;

        if let (true, Some(et)) = (self.config.use_attention, edge_type) {
            let edge_types = et.to_vec1::<u8>()?;
            let scores = self.compute_attention(&source_features, &edge_types)?;

            // Softmax per target node would require grouping — for simplicity,
            // we use a sigmoid gate instead (independent per edge)
            let gates = candle_nn::ops::sigmoid(&scores)?; // [num_edges]
            let gates = gates.unsqueeze(1)?; // [num_edges, 1]

            // Weighted messages
            let weighted = source_features.broadcast_mul(&gates)?;
            Ok(weighted)
        } else {
            Ok(source_features)
        }
    }

    fn aggregate(
        &self,
        messages: &Tensor,
        edge_index: &Tensor,
        num_nodes: usize,
    ) -> Result<Tensor> {
        let (_, targets) = extract_edge_indices(edge_index)?;
        // Mean aggregation (GraphSAGE default)
        scatter_mean(messages, &targets, num_nodes)
    }

    fn update(&self, x: &Tensor, aggregated: &Tensor) -> Result<Tensor> {
        // GraphSAGE update: concat(self_transform, neighbor_transform)
        // Simplified: sum of self and neighbor transforms
        let self_out = self.self_linear.forward(x)?;
        let neigh_out = self.neigh_linear.forward(aggregated)?;

        let combined = (self_out + neigh_out)?;
        let activated = combined.gelu()?;

        // Optional L2 normalization
        if self.config.normalize {
            let norm = activated
                .sqr()?
                .sum_keepdim(D::Minus1)?
                .sqrt()?
                .clamp(1e-12, f64::INFINITY)?;
            let normalized = activated.broadcast_div(&norm)?;
            Ok(normalized)
        } else {
            Ok(activated)
        }
    }
}

/// Multi-layer GraphSAGE model (inductively learns node embeddings).
pub struct GraphSAGE {
    layers: Vec<GraphSAGELayer>,
}

impl GraphSAGE {
    /// Create a new multi-layer GraphSAGE.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        num_relations: usize,
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

            let config = GraphSAGEConfig {
                input_dim: in_d,
                output_dim: out_d,
                num_relations,
                dropout,
                use_attention: true,
                normalize: i == num_layers - 1, // only normalize last layer
            };

            let layer = GraphSAGELayer::new(config, vb.pp(format!("layer_{}", i)))?;
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

    /// Inductively encode a NEW node (not seen during training).
    ///
    /// Given the new node's features and its neighbors' pre-computed embeddings,
    /// runs a single forward pass to produce an embedding.
    pub fn encode_new_node(
        &self,
        node_features: &Tensor,
        neighbor_embeddings: &Tensor,
    ) -> Result<Tensor> {
        // For a single new node with known neighbors, we can directly
        // apply the last layer's update rule
        if self.layers.is_empty() {
            return Ok(node_features.clone());
        }

        let last = &self.layers[self.layers.len() - 1];

        // Mean of neighbor embeddings as aggregated message
        let aggregated = neighbor_embeddings.mean(0)?.unsqueeze(0)?;
        let self_features = node_features.unsqueeze(0)?;

        let embedding = last.update(&self_features, &aggregated)?;
        embedding.squeeze(0)
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

        let x = Tensor::randn(0.0f32, 1.0, (num_nodes, input_dim), &device).unwrap();
        let edge_index =
            Tensor::new(&[[0i64, 1, 2, 3, 0, 1], [1, 2, 3, 4, 2, 3]], &device).unwrap();
        let edge_type = Tensor::new(&[0u8, 1, 2, 0, 3, 1], &device).unwrap();

        (x, edge_index, Some(edge_type), num_nodes)
    }

    #[test]
    fn test_graphsage_layer_forward() -> Result<()> {
        let (x, edge_index, edge_type, num_nodes) = make_test_graph();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let config = GraphSAGEConfig {
            input_dim: 8,
            output_dim: 16,
            num_relations: 8,
            dropout: 0.0,
            use_attention: true,
            normalize: true,
        };

        let layer = GraphSAGELayer::new(config, vb)?;
        let output = layer.forward(&x, &edge_index, edge_type.as_ref(), num_nodes)?;

        assert_eq!(output.dims(), &[num_nodes, 16]);

        // Check L2 normalization: each row should have unit norm
        let norms = output.sqr()?.sum(D::Minus1)?.sqrt()?;
        let norms_vec = norms.to_vec1::<f32>()?;
        for (i, norm) in norms_vec.iter().enumerate() {
            assert!(
                (norm - 1.0).abs() < 0.01,
                "Node {} norm = {}, expected ~1.0",
                i,
                norm
            );
        }
        Ok(())
    }

    #[test]
    fn test_graphsage_multi_layer() -> Result<()> {
        let (x, edge_index, edge_type, num_nodes) = make_test_graph();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let model = GraphSAGE::new(8, 32, 16, 3, 8, 0.0, vb)?;
        let output = model.forward(&x, &edge_index, edge_type.as_ref(), num_nodes)?;

        assert_eq!(output.dims(), &[num_nodes, 16]);
        Ok(())
    }

    #[test]
    fn test_graphsage_without_attention() -> Result<()> {
        let (x, edge_index, _, num_nodes) = make_test_graph();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let config = GraphSAGEConfig {
            input_dim: 8,
            output_dim: 16,
            num_relations: 8,
            dropout: 0.0,
            use_attention: false,
            normalize: false,
        };

        let layer = GraphSAGELayer::new(config, vb)?;
        let output = layer.forward(&x, &edge_index, None, num_nodes)?;

        assert_eq!(output.dims(), &[num_nodes, 16]);
        Ok(())
    }

    #[test]
    fn test_graphsage_inductive() -> Result<()> {
        // Test that a new unseen node can be encoded
        // The last layer expects hidden_dim input, so the new node must have hidden_dim features
        let device = Device::Cpu;
        let hidden_dim = 16;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = GraphSAGE::new(8, hidden_dim, 16, 2, 8, 0.0, vb)?;

        // New node features must match last layer's input_dim (= hidden_dim for multi-layer)
        let new_node = Tensor::randn(0.0f32, 1.0, hidden_dim, &device)?;
        // Its neighbors' pre-computed embeddings (3 neighbors, hidden_dim)
        let neighbors = Tensor::randn(0.0f32, 1.0, (3, hidden_dim), &device)?;

        let embedding = model.encode_new_node(&new_node, &neighbors)?;
        assert_eq!(embedding.dims(), &[16]);
        Ok(())
    }

    #[test]
    fn test_graphsage_100_nodes() -> Result<()> {
        let device = Device::Cpu;
        let num_nodes = 100;
        let input_dim = 169;

        let x = Tensor::randn(0.0f32, 1.0, (num_nodes, input_dim), &device)?;

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

        let model = GraphSAGE::new(169, 256, 256, 3, 8, 0.0, vb)?;
        let output = model.forward(&x, &edge_index, Some(&edge_type), num_nodes)?;

        assert_eq!(output.dims(), &[100, 256]);
        Ok(())
    }
}
