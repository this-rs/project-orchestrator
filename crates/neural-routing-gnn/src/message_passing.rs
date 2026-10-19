//! Message passing framework — the foundation for all GNN layers.
//!
//! Provides `scatter_add` for aggregating messages and the `MessagePassing` trait
//! that R-GCN and GraphSAGE build upon.

#[cfg(test)]
use candle_core::Device;
use candle_core::{DType, Result, Tensor};

/// Trait for message passing layers in graph neural networks.
///
/// Implements the message-passing paradigm:
/// 1. `message()` — compute messages from source to target along edges
/// 2. `aggregate()` — aggregate messages at each target node (scatter_add)
/// 3. `update()` — update node representation with aggregated messages
pub trait MessagePassing {
    /// Compute messages along edges.
    ///
    /// * `x` — node features [num_nodes, feature_dim]
    /// * `edge_index` — [2, num_edges] (source, target)
    /// * `edge_type` — [num_edges] relation type IDs (optional, for R-GCN)
    fn message(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_type: Option<&Tensor>,
    ) -> Result<Tensor>;

    /// Aggregate messages at each target node via scatter_add.
    fn aggregate(&self, messages: &Tensor, edge_index: &Tensor, num_nodes: usize)
        -> Result<Tensor>;

    /// Update node representations with aggregated messages.
    fn update(&self, x: &Tensor, aggregated: &Tensor) -> Result<Tensor>;

    /// Full forward pass: message -> (optional edge-weight scaling) -> aggregate -> update.
    ///
    /// * `edge_weight` — optional `[num_edges]` per-edge weights. When `Some`, every
    ///   message row is multiplied by its edge weight *before* aggregation, so an edge
    ///   of weight `0.0` contributes nothing (its message becomes the zero vector).
    ///   When `None`, no scaling happens at all and every edge weighs `1.0` — this is
    ///   the historical behaviour, kept bit-for-bit identical.
    ///
    /// Caveat on the denominator: `scatter_add` (R-GCN) simply drops a zero-weight
    /// message, but `scatter_mean` (GraphSAGE) still counts the edge in its divisor,
    /// so a `0.0` weight attenuates an edge there instead of removing it.
    fn forward(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_type: Option<&Tensor>,
        edge_weight: Option<&Tensor>,
        num_nodes: usize,
    ) -> Result<Tensor> {
        let messages = self.message(x, edge_index, edge_type)?;
        let messages = match edge_weight {
            Some(w) => {
                let weights = extract_edge_weights(w)?;
                scale_messages_by_weight(&messages, &weights)?
            }
            None => messages,
        };
        let aggregated = self.aggregate(&messages, edge_index, num_nodes)?;
        self.update(x, &aggregated)
    }
}

/// Scatter-add: aggregate `src` values at positions given by `index` into a tensor of shape [dim_size, feature_dim].
///
/// This is the core aggregation primitive for GNNs. For each edge (i, j),
/// the message from node i is added to the aggregation buffer at position j.
///
/// * `src` — [num_edges, feature_dim] messages to aggregate
/// * `index` — [num_edges] target node indices
/// * `dim_size` — total number of nodes (output rows)
///
/// Returns [dim_size, feature_dim] aggregated messages.
pub fn scatter_add(src: &Tensor, index: &[u32], dim_size: usize) -> Result<Tensor> {
    let feature_dim = src.dim(1)?;
    let device = src.device();

    // Initialize output as zeros
    let mut output = Tensor::zeros((dim_size, feature_dim), src.dtype(), device)?;

    // Accumulate: for each edge, add the message to the target position
    // Using index_add (candle supports this)
    let index_tensor = Tensor::from_vec(
        index.iter().map(|&i| i as i64).collect::<Vec<_>>(),
        index.len(),
        device,
    )?;

    output = output.index_add(&index_tensor, src, 0)?;

    Ok(output)
}

/// Scatter-mean: like scatter_add but divides by count.
///
/// * `src` — [num_edges, feature_dim]
/// * `index` — [num_edges] target indices
/// * `dim_size` — number of output nodes
pub fn scatter_mean(src: &Tensor, index: &[u32], dim_size: usize) -> Result<Tensor> {
    let _feature_dim = src.dim(1)?;
    let device = src.device();

    let summed = scatter_add(src, index, dim_size)?;

    // Count occurrences per target node
    let mut counts = vec![0.0f32; dim_size];
    for &idx in index {
        counts[idx as usize] += 1.0;
    }
    // Avoid division by zero
    for c in counts.iter_mut() {
        if *c == 0.0 {
            *c = 1.0;
        }
    }

    let counts_tensor = Tensor::from_vec(counts, (dim_size, 1), device)?;
    let result = summed.broadcast_div(&counts_tensor)?;

    Ok(result)
}

/// Extract source and target indices from edge_index tensor.
///
/// * `edge_index` — [2, num_edges]
///
/// Returns (source_indices, target_indices) as Vec<u32>.
pub fn extract_edge_indices(edge_index: &Tensor) -> Result<(Vec<u32>, Vec<u32>)> {
    let sources = edge_index.get(0)?.to_vec1::<i64>()?;
    let targets = edge_index.get(1)?.to_vec1::<i64>()?;

    Ok((
        sources.iter().map(|&v| v as u32).collect(),
        targets.iter().map(|&v| v as u32).collect(),
    ))
}

/// Gather rows from a tensor by index.
///
/// * `x` — [num_nodes, feature_dim]
/// * `indices` — [num_edges] node indices to gather
///
/// Returns [num_edges, feature_dim] gathered features.
pub fn gather_rows(x: &Tensor, indices: &[u32]) -> Result<Tensor> {
    let index_tensor = Tensor::from_vec(
        indices.iter().map(|&i| i as i64).collect::<Vec<_>>(),
        indices.len(),
        x.device(),
    )?;
    x.index_select(&index_tensor, 0)
}

/// Extract per-edge weights from a tensor.
///
/// * `edge_weight` — `[num_edges]` weights (any float dtype; cast to f32)
///
/// Returns one weight per edge, in edge order. A 2-D `[num_edges, 1]` tensor is also
/// accepted (it is flattened first), so callers do not have to reshape.
pub fn extract_edge_weights(edge_weight: &Tensor) -> Result<Vec<f32>> {
    let flattened = edge_weight.to_dtype(DType::F32)?.flatten_all()?;
    if flattened.rank() != 1 {
        return Err(candle_core::Error::Msg(format!(
            "extract_edge_weights expects a 1-D tensor, got shape {:?}",
            edge_weight.dims()
        )));
    }
    flattened.to_vec1::<f32>()
}

/// Scale each row of a message matrix by its per-edge weight.
///
/// * `messages` — `[num_edges, feature_dim]` messages (e.g. the output of `gather_rows`)
/// * `weights` — one weight per edge; `weights.len()` must equal `messages.dim(0)`
///
/// Returns a tensor of the same shape where row `e` is `messages[e] * weights[e]`.
/// A weight of `0.0` therefore yields an all-zero message row (the edge contributes
/// nothing to a `scatter_add` aggregation).
pub fn scale_messages_by_weight(messages: &Tensor, weights: &[f32]) -> Result<Tensor> {
    if messages.rank() != 2 {
        return Err(candle_core::Error::Msg(format!(
            "scale_messages_by_weight expects [num_edges, feature_dim] messages, got shape {:?}",
            messages.dims()
        )));
    }

    let num_edges = messages.dim(0)?;
    if weights.len() != num_edges {
        return Err(candle_core::Error::Msg(format!(
            "edge weight count ({}) does not match message count ({} edges)",
            weights.len(),
            num_edges
        )));
    }

    // [num_edges, 1] so the weight applies to the whole feature row (broadcast_mul).
    let weight_tensor = Tensor::from_vec(weights.to_vec(), (num_edges, 1), messages.device())?
        .to_dtype(messages.dtype())?;

    messages.broadcast_mul(&weight_tensor)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scatter_add_basic() -> Result<()> {
        let device = Device::Cpu;

        // 4 edges with 3-dim features
        let src = Tensor::new(
            &[
                [1.0f32, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            &device,
        )?;
        // Edges point to nodes 0, 1, 0, 2
        let index = vec![0u32, 1, 0, 2];

        let result = scatter_add(&src, &index, 3)?;

        // Node 0: [1+7, 2+8, 3+9] = [8, 10, 12]
        // Node 1: [4, 5, 6]
        // Node 2: [10, 11, 12]
        let expected = Tensor::new(
            &[[8.0f32, 10.0, 12.0], [4.0, 5.0, 6.0], [10.0, 11.0, 12.0]],
            &device,
        )?;

        let diff = (result - expected)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "scatter_add mismatch: diff = {}", diff);
        Ok(())
    }

    #[test]
    fn test_scatter_add_empty_nodes() -> Result<()> {
        let device = Device::Cpu;
        let src = Tensor::new(&[[1.0f32, 2.0]], &device)?;
        let index = vec![2u32]; // only node 2 gets a message

        let result = scatter_add(&src, &index, 4)?;
        // Nodes 0, 1, 3 should be zero
        let r = result.to_vec2::<f32>()?;
        assert_eq!(r[0], vec![0.0, 0.0]);
        assert_eq!(r[1], vec![0.0, 0.0]);
        assert_eq!(r[2], vec![1.0, 2.0]);
        assert_eq!(r[3], vec![0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn test_scatter_mean() -> Result<()> {
        let device = Device::Cpu;
        let src = Tensor::new(&[[2.0f32, 4.0], [6.0, 8.0], [10.0, 12.0]], &device)?;
        let index = vec![0u32, 0, 1]; // two messages to node 0, one to node 1

        let result = scatter_mean(&src, &index, 2)?;
        let r = result.to_vec2::<f32>()?;
        // Node 0: mean([2,4], [6,8]) = [4, 6]
        assert!((r[0][0] - 4.0).abs() < 1e-5);
        assert!((r[0][1] - 6.0).abs() < 1e-5);
        // Node 1: [10, 12]
        assert!((r[1][0] - 10.0).abs() < 1e-5);
        assert!((r[1][1] - 12.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_gather_rows() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &device)?;
        let indices = vec![2u32, 0, 1, 2];

        let result = gather_rows(&x, &indices)?;
        let r = result.to_vec2::<f32>()?;
        assert_eq!(r[0], vec![5.0, 6.0]); // node 2
        assert_eq!(r[1], vec![1.0, 2.0]); // node 0
        assert_eq!(r[2], vec![3.0, 4.0]); // node 1
        assert_eq!(r[3], vec![5.0, 6.0]); // node 2
        Ok(())
    }

    #[test]
    fn test_extract_edge_indices() -> Result<()> {
        let device = Device::Cpu;
        let edge_index = Tensor::new(&[[0i64, 1, 2], [1, 2, 0]], &device)?;

        let (src, tgt) = extract_edge_indices(&edge_index)?;
        assert_eq!(src, vec![0, 1, 2]);
        assert_eq!(tgt, vec![1, 2, 0]);
        Ok(())
    }

    #[test]
    fn test_extract_edge_weights() -> Result<()> {
        let device = Device::Cpu;

        let w = Tensor::new(&[1.0f32, 0.75, 0.5], &device)?;
        assert_eq!(extract_edge_weights(&w)?, vec![1.0, 0.75, 0.5]);

        // f64 weights are accepted (cast to f32) — the sampler stores f64.
        let w64 = Tensor::new(&[1.0f64, 0.25], &device)?;
        assert_eq!(extract_edge_weights(&w64)?, vec![1.0, 0.25]);

        // [num_edges, 1] is flattened.
        let w2d = Tensor::new(&[[1.0f32], [0.0]], &device)?;
        assert_eq!(extract_edge_weights(&w2d)?, vec![1.0, 0.0]);

        Ok(())
    }

    #[test]
    fn test_scale_messages_by_weight_basic() -> Result<()> {
        let device = Device::Cpu;
        let messages = Tensor::new(&[[1.0f32, 2.0], [4.0, 6.0], [0.5, 0.5]], &device)?;

        let scaled = scale_messages_by_weight(&messages, &[1.0, 0.5, 0.0])?;
        let r = scaled.to_vec2::<f32>()?;

        assert_eq!(r[0], vec![1.0, 2.0]);
        assert_eq!(r[1], vec![2.0, 3.0]);
        // A zero weight produces a zero message row.
        assert_eq!(r[2], vec![0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn test_scale_messages_by_weight_unit_is_identity() -> Result<()> {
        let device = Device::Cpu;
        let messages = Tensor::new(&[[1.5f32, -2.0], [4.0, 6.0]], &device)?;

        let scaled = scale_messages_by_weight(&messages, &[1.0, 1.0])?;
        let diff = (scaled - &messages)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6, "unit weights should be the identity: {}", diff);
        Ok(())
    }

    #[test]
    fn test_scale_messages_by_weight_length_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let messages = Tensor::new(&[[1.0f32, 2.0], [4.0, 6.0]], &device)?;

        // 3 weights for 2 edges → explicit error, never silent truncation.
        assert!(scale_messages_by_weight(&messages, &[1.0, 1.0, 1.0]).is_err());
        assert!(scale_messages_by_weight(&messages, &[1.0]).is_err());
        Ok(())
    }

    #[test]
    fn test_scale_messages_by_weight_preserves_shape() -> Result<()> {
        let device = Device::Cpu;
        let messages = Tensor::randn(0.0f32, 1.0, (7, 4), &device)?;
        let weights = vec![0.1f32; 7];

        let scaled = scale_messages_by_weight(&messages, &weights)?;
        assert_eq!(scaled.dims(), &[7, 4]);
        Ok(())
    }

    #[test]
    fn test_scatter_add_matches_naive() -> Result<()> {
        // Verify scatter_add matches a naive O(n²) implementation
        let device = Device::Cpu;
        let num_edges = 10;
        let dim = 4;
        let num_nodes = 5;

        let src_data: Vec<f32> = (0..(num_edges * dim)).map(|i| i as f32 * 0.1).collect();
        let src = Tensor::from_vec(src_data.clone(), (num_edges, dim), &device)?;
        let index: Vec<u32> = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4];

        let fast_result = scatter_add(&src, &index, num_nodes)?;

        // Naive implementation
        let mut naive = vec![vec![0.0f32; dim]; num_nodes];
        for (edge_idx, &target) in index.iter().enumerate() {
            for d in 0..dim {
                naive[target as usize][d] += src_data[edge_idx * dim + d];
            }
        }

        let fast_data = fast_result.to_vec2::<f32>()?;
        for node in 0..num_nodes {
            for d in 0..dim {
                assert!(
                    (fast_data[node][d] - naive[node][d]).abs() < 1e-5,
                    "Mismatch at node={}, dim={}: {} vs {}",
                    node,
                    d,
                    fast_data[node][d],
                    naive[node][d]
                );
            }
        }
        Ok(())
    }
}
