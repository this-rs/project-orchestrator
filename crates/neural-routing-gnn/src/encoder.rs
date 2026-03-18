//! Graph Encoder — orchestrates GNN layers to produce node embeddings.
//!
//! Supports both R-GCN and GraphSAGE architectures. The encoder is the main
//! entry point for producing graph-aware node embeddings from the knowledge graph.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

use crate::graph_sage::GraphSAGE;
use crate::rgcn::RGCN;

/// Configuration for the Graph Encoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEncoderConfig {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Hidden layer dimension.
    pub hidden_dim: usize,
    /// Output embedding dimension.
    pub output_dim: usize,
    /// Number of GNN layers.
    pub num_layers: usize,
    /// Number of relation types in the knowledge graph.
    pub num_relations: usize,
    /// Number of basis matrices (R-GCN only).
    pub num_bases: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Which GNN architecture to use.
    pub architecture: GNNArchitecture,
}

/// Supported GNN architectures.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum GNNArchitecture {
    /// R-GCN with basis decomposition (transductive).
    RGCN,
    /// GraphSAGE with attention (inductive — supports new nodes).
    GraphSAGE,
}

impl Default for GraphEncoderConfig {
    fn default() -> Self {
        Self {
            input_dim: 168, // TOTAL_FEATURE_DIM from features.rs
            hidden_dim: 256,
            output_dim: 256,
            num_layers: 3,
            num_relations: 8,
            num_bases: 4,
            dropout: 0.1,
            architecture: GNNArchitecture::GraphSAGE, // default: inductive
        }
    }
}

/// Graph Encoder — produces node embeddings from the knowledge graph.
pub enum GraphEncoder {
    RGCN(RGCN),
    GraphSAGE(GraphSAGE),
}

impl GraphEncoder {
    pub fn new(config: GraphEncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        match config.architecture {
            GNNArchitecture::RGCN => {
                let model = RGCN::new(
                    config.input_dim,
                    config.hidden_dim,
                    config.output_dim,
                    config.num_layers,
                    config.num_relations,
                    config.num_bases,
                    config.dropout,
                    vb,
                )?;
                Ok(Self::RGCN(model))
            }
            GNNArchitecture::GraphSAGE => {
                let model = GraphSAGE::new(
                    config.input_dim,
                    config.hidden_dim,
                    config.output_dim,
                    config.num_layers,
                    config.num_relations,
                    config.dropout,
                    vb,
                )?;
                Ok(Self::GraphSAGE(model))
            }
        }
    }

    /// Create with a new VarMap (convenience for inference).
    pub fn with_default_vars(config: GraphEncoderConfig) -> Result<(Self, VarMap)> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let encoder = Self::new(config, vb)?;
        Ok((encoder, varmap))
    }

    /// Forward pass: produce embeddings for all nodes.
    ///
    /// * `x` — [num_nodes, input_dim] node features
    /// * `edge_index` — [2, num_edges] source/target pairs
    /// * `edge_type` — [num_edges] relation type IDs (0-7)
    /// * `num_nodes` — total number of nodes
    pub fn forward(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_type: Option<&Tensor>,
        num_nodes: usize,
    ) -> Result<Tensor> {
        match self {
            Self::RGCN(model) => model.forward(x, edge_index, edge_type, num_nodes),
            Self::GraphSAGE(model) => model.forward(x, edge_index, edge_type, num_nodes),
        }
    }

    /// Inductively encode a new node (GraphSAGE only).
    pub fn encode_new_node(
        &self,
        node_features: &Tensor,
        neighbor_embeddings: &Tensor,
    ) -> Result<Tensor> {
        match self {
            Self::GraphSAGE(model) => model.encode_new_node(node_features, neighbor_embeddings),
            Self::RGCN(_) => Err(candle_core::Error::Msg(
                "R-GCN does not support inductive encoding. Use GraphSAGE.".to_string(),
            )),
        }
    }

    /// Whether this encoder supports inductive encoding of new nodes.
    pub fn is_inductive(&self) -> bool {
        matches!(self, Self::GraphSAGE(_))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_rgcn() -> Result<()> {
        let config = GraphEncoderConfig {
            input_dim: 8,
            hidden_dim: 16,
            output_dim: 16,
            num_layers: 2,
            architecture: GNNArchitecture::RGCN,
            ..Default::default()
        };

        let (encoder, _varmap) = GraphEncoder::with_default_vars(config)?;
        assert!(!encoder.is_inductive());

        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (5, 8), &device)?;
        let edge_index = Tensor::new(&[[0i64, 1, 2], [1, 2, 3]], &device)?;
        let edge_type = Tensor::new(&[0u8, 1, 2], &device)?;

        let output = encoder.forward(&x, &edge_index, Some(&edge_type), 5)?;
        assert_eq!(output.dims(), &[5, 16]);
        Ok(())
    }

    #[test]
    fn test_encoder_graphsage() -> Result<()> {
        let config = GraphEncoderConfig {
            input_dim: 8,
            hidden_dim: 16,
            output_dim: 16,
            num_layers: 2,
            architecture: GNNArchitecture::GraphSAGE,
            ..Default::default()
        };

        let (encoder, _varmap) = GraphEncoder::with_default_vars(config)?;
        assert!(encoder.is_inductive());

        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (5, 8), &device)?;
        let edge_index = Tensor::new(&[[0i64, 1, 2], [1, 2, 3]], &device)?;

        let output = encoder.forward(&x, &edge_index, None, 5)?;
        assert_eq!(output.dims(), &[5, 16]);
        Ok(())
    }

    #[test]
    fn test_encoder_default_config() {
        let config = GraphEncoderConfig::default();
        assert_eq!(config.input_dim, 168);
        assert_eq!(config.output_dim, 256);
        assert_eq!(config.architecture, GNNArchitecture::GraphSAGE);
    }

    #[test]
    fn test_encoder_inductive_encoding() -> Result<()> {
        let hidden_dim = 16;
        let config = GraphEncoderConfig {
            input_dim: 8,
            hidden_dim,
            output_dim: 16,
            num_layers: 2,
            architecture: GNNArchitecture::GraphSAGE,
            ..Default::default()
        };

        let (encoder, _) = GraphEncoder::with_default_vars(config)?;
        let device = Device::Cpu;

        // Last layer expects hidden_dim input
        let node = Tensor::randn(0.0f32, 1.0, hidden_dim, &device)?;
        let neighbors = Tensor::randn(0.0f32, 1.0, (3, hidden_dim), &device)?;

        let emb = encoder.encode_new_node(&node, &neighbors)?;
        assert_eq!(emb.dims(), &[16]);
        Ok(())
    }

    #[test]
    fn test_encoder_rgcn_no_inductive() -> Result<()> {
        let config = GraphEncoderConfig {
            input_dim: 8,
            hidden_dim: 16,
            output_dim: 16,
            num_layers: 1,
            architecture: GNNArchitecture::RGCN,
            ..Default::default()
        };

        let (encoder, _) = GraphEncoder::with_default_vars(config)?;
        let device = Device::Cpu;

        let node = Tensor::randn(0.0f32, 1.0, 8, &device)?;
        let neighbors = Tensor::randn(0.0f32, 1.0, (3, 16), &device)?;

        let result = encoder.encode_new_node(&node, &neighbors);
        assert!(result.is_err(), "R-GCN should not support inductive encoding");
        Ok(())
    }
}
