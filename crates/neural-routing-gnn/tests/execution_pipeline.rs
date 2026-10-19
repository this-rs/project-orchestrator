//! End-to-end execution proof for the GNN pipeline.
//!
//! The test walks the whole path once, in the order production would:
//!
//! 1. build a small but realistic `SubGraph` (10 nodes, several `RelationType`s
//!    including `Synapse`, per-edge weights, and the property spellings the graph
//!    actually stores — 768d `embedding`, `staleness_score`,
//!    `clustering_coefficient`, RFC 3339 `created_at`);
//! 2. `DataLoader::new` + `prepare` → train / validation / test splits;
//! 3. `training::train` for a few epochs on a light architecture, writing
//!    `<output_dir>/best_model.safetensors`;
//! 4. `CandleGnnEncoder` reloads that checkpoint;
//! 5. `encode_subgraph` / `encode_node` turn the subgraph into embeddings that
//!    are finite, non-zero, deterministic, and different from an untrained model.
//!
//! # Branching decision — why this is a test and not a production path
//!
//! Nothing in the repository calls `train()`, and `set_gnn_encoder` was removed
//! from the backend: the GNN has **no consumer**. Wiring the encoder back into
//! the backend as-is would be machinery without a purpose — embeddings produced
//! and then ignored. The pipeline is therefore exercised only here, where the
//! proof is cheap and cannot rot silently (CI compiles and runs it).
//!
//! Branching it back into production is a *consumer* decision, not a plumbing
//! one: it needs a routing decision that actually *uses* the embedding (a policy
//! scoring tool/provider candidates, or the `neural_routing` NN path reading GNN
//! node embeddings instead of raw features). Until that consumer exists, do not
//! reintroduce `set_gnn_encoder`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use neural_routing_gnn::training::{train, DataLoader, DataLoaderConfig, TrainingConfig};
use neural_routing_gnn::{
    CandleGnnEncoder, CandleGnnEncoderConfig, GNNArchitecture, GnnEncoder, GraphEncoderConfig,
    RelationType, SubGraph, SubGraphEdge, SubGraphNode, TOTAL_FEATURE_DIM,
};

/// Node count of the synthetic subgraph.
const NUM_NODES: usize = 10;
/// Voyage embedding width the feature builder accepts (anything else → zero block).
const VOYAGE_EMB_DIM: usize = 768;
/// Label of the node used for the single-node encoding assertion.
const TARGET_APP_ID: &str = "app_3";

/// `(source, target, relation index into RelationType::ALL, weight)`.
///
/// 16 edges over 10 nodes, with `Synapse` (index 4) as the dominant relation and
/// three other relation types present, so the relational path of the encoder is
/// exercised rather than a single averaged relation.
const EDGES: [(usize, usize, usize, f64); 16] = [
    (0, 1, 4, 1.00),
    (1, 2, 4, 0.85),
    (2, 3, 1, 0.70),
    (3, 4, 0, 0.55),
    (4, 5, 4, 1.00),
    (5, 6, 2, 0.40),
    (6, 7, 3, 0.65),
    (7, 8, 4, 0.90),
    (8, 9, 5, 0.30),
    (0, 2, 4, 0.95),
    (1, 3, 6, 0.50),
    (2, 4, 7, 0.45),
    (3, 5, 4, 0.80),
    (5, 7, 1, 0.60),
    (6, 8, 4, 0.75),
    (0, 9, 2, 0.35),
];

/// A temporary directory removed when the test ends (pass or fail).
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "neural-routing-gnn-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("create temp output dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Build one node carrying the property spellings the knowledge graph really has.
fn build_node(index: usize) -> SubGraphNode {
    let label = match index % 5 {
        0 => "Note",
        1 => "File",
        2 => "Decision",
        3 => "Function",
        _ => "Note",
    };

    // 768d deterministic pseudo-embedding: no random crate needed, and no NaN.
    let embedding: Vec<f64> = (0..VOYAGE_EMB_DIM)
        .map(|j| {
            let h = (index * 37 + j * 13) % 17;
            (h as f64) / 17.0 - 0.5
        })
        .collect();

    let mut properties: HashMap<String, serde_json::Value> = HashMap::new();
    // Voyage block — must be exactly 768d or the builder imputes zeros.
    properties.insert("embedding".to_string(), serde_json::json!(embedding));
    // Note-level spellings (read through the #891 fallbacks).
    properties.insert(
        "staleness_score".to_string(),
        serde_json::json!(0.05 * index as f64),
    );
    properties.insert(
        "clustering_coefficient".to_string(),
        serde_json::json!(0.10 + 0.07 * index as f64),
    );
    properties.insert(
        "created_at".to_string(),
        serde_json::json!(format!("2026-09-{:02}T12:00:00Z", index + 1)),
    );
    // Local metrics block.
    properties.insert(
        "energy".to_string(),
        serde_json::json!(0.90 - 0.05 * index as f64),
    );
    properties.insert(
        "scar_intensity".to_string(),
        serde_json::json!(0.02 * index as f64),
    );
    properties.insert(
        "synapse_count".to_string(),
        serde_json::json!(3 + index as i64),
    );
    properties.insert(
        "knowledge_density".to_string(),
        serde_json::json!(0.2 + 0.04 * index as f64),
    );
    // GDS block (written on :File in the real graph; harmless on the others).
    properties.insert(
        "pagerank".to_string(),
        serde_json::json!(0.01 + 0.02 * index as f64),
    );
    properties.insert(
        "betweenness".to_string(),
        serde_json::json!(0.005 * index as f64),
    );
    properties.insert("degree_in".to_string(), serde_json::json!(index as i64 % 4));
    properties.insert(
        "degree_out".to_string(),
        serde_json::json!((index as i64 + 2) % 4),
    );
    properties.insert("churn_score".to_string(), serde_json::json!(0.3));
    properties.insert("bridge_score".to_string(), serde_json::json!(0.1));

    SubGraphNode {
        element_id: format!("elem_{index}"),
        app_id: format!("app_{index}"),
        labels: vec![label.to_string()],
        properties,
    }
}

/// Build the synthetic subgraph: 10 labelled nodes, 16 weighted typed edges.
fn build_subgraph() -> SubGraph {
    let nodes: Vec<SubGraphNode> = (0..NUM_NODES).map(build_node).collect();

    let edges: Vec<SubGraphEdge> = EDGES
        .iter()
        .map(|&(s, t, rel, weight)| SubGraphEdge {
            source_element_id: format!("elem_{s}"),
            target_element_id: format!("elem_{t}"),
            relation_type: RelationType::ALL[rel],
            weight,
        })
        .collect();

    SubGraph {
        center_id: "elem_0".to_string(),
        k_hops: 2,
        nodes,
        edges,
    }
}

/// A deliberately small encoder: the proof is the plumbing, not the capacity.
fn encoder_config() -> GraphEncoderConfig {
    GraphEncoderConfig {
        input_dim: TOTAL_FEATURE_DIM,
        hidden_dim: 32,
        output_dim: 16,
        num_layers: 2,
        num_relations: RelationType::count(),
        num_bases: 4,
        dropout: 0.0,
        architecture: GNNArchitecture::GraphSAGE,
    }
}

#[test]
fn train_save_reload_and_encode_end_to_end() {
    let started = Instant::now();

    // --- 1. Data ---------------------------------------------------------
    let subgraph = build_subgraph();
    assert_eq!(subgraph.node_count(), NUM_NODES);
    assert_eq!(subgraph.edge_count(), EDGES.len());

    let mut loader = DataLoader::new(DataLoaderConfig {
        neg_ratio: 3,
        val_fraction: 0.2,
        test_fraction: 0.2,
        seed: 7,
    });
    let (train_split, val_split, test_split) = loader
        .prepare(&subgraph)
        .expect("DataLoader::prepare must build the three splits");

    for split in [&train_split, &val_split, &test_split] {
        assert_eq!(split.x.dims(), &[NUM_NODES, TOTAL_FEATURE_DIM]);
        assert!(
            split.pos_edges.dim(1).expect("pos edges") > 0,
            "{:?} split must contain supervision edges",
            split.split
        );
    }
    // The per-edge weights read by #891 must survive `prepare`.
    assert!(
        train_split.edge_weight.is_some(),
        "train split lost the per-edge weights"
    );

    // --- 2. Train --------------------------------------------------------
    let dir = TempDir::new("exec");
    let config = TrainingConfig {
        encoder_config: encoder_config(),
        epochs: 3,
        lr: 1e-2,
        weight_decay: 0.0,
        warmup_epochs: 1,
        max_grad_norm: 1.0,
        patience: 5,
        output_dir: dir.path().to_path_buf(),
        // No intermediate checkpoints: this test only wants `best_model`.
        checkpoint_every: 0,
    };

    let result = train(&config, &train_split, &val_split, &test_split)
        .expect("training must run to completion");

    // --- 3. Training result ---------------------------------------------
    assert!(!result.epoch_history.is_empty(), "no epoch was recorded");
    assert_eq!(result.epoch_history.len(), config.epochs);
    assert_eq!(result.architecture, "GraphSAGE");
    assert!(result.total_params > 0, "the model has no parameters");
    for metrics in &result.epoch_history {
        assert!(
            metrics.train_loss.is_finite(),
            "epoch {}: train loss is not finite ({})",
            metrics.epoch,
            metrics.train_loss
        );
        assert!(
            metrics.val_loss.is_finite(),
            "epoch {}: validation loss is not finite ({})",
            metrics.epoch,
            metrics.val_loss
        );
        assert!(metrics.train_loss >= 0.0, "BCE loss cannot be negative");
    }
    assert!(
        result.final_test_loss.is_finite(),
        "test loss is not finite: {}",
        result.final_test_loss
    );

    // --- 4. Checkpoint ---------------------------------------------------
    let checkpoint = dir.path().join("best_model.safetensors");
    assert!(
        checkpoint.exists(),
        "no checkpoint at {}",
        checkpoint.display()
    );
    let checkpoint_len = std::fs::metadata(&checkpoint)
        .expect("checkpoint metadata")
        .len();
    assert!(
        checkpoint_len > 1024,
        "checkpoint is suspiciously small: {checkpoint_len} bytes"
    );

    // --- 5. Reload and encode -------------------------------------------
    let encoder = CandleGnnEncoder::new(CandleGnnEncoderConfig {
        encoder_config: encoder_config(),
        checkpoint_path: Some(checkpoint.to_string_lossy().to_string()),
        norm_stats: loader.norm_stats().cloned(),
    })
    .expect("the checkpoint written by train() must load back");
    assert_eq!(encoder.output_dim(), 16);
    assert!(encoder.is_inductive());

    let embeddings = encoder
        .encode_subgraph(&subgraph)
        .expect("encoding the trained subgraph must work");
    assert_eq!(embeddings.len(), NUM_NODES);
    for (node, embedding) in embeddings.iter().enumerate() {
        assert_eq!(
            embedding.len(),
            16,
            "node {node}: wrong embedding dimension"
        );
        assert!(
            embedding.iter().all(|v| v.is_finite()),
            "node {node}: non-finite embedding value"
        );
    }

    // Non-zero: the embedding carries signal, not an all-zero / dead-ReLU block.
    let squared_norm: f32 = embeddings
        .iter()
        .flat_map(|e| e.iter())
        .map(|v| v * v)
        .sum();
    assert!(
        squared_norm > 0.0 && squared_norm.is_finite(),
        "embeddings are all zeros (squared norm {squared_norm})"
    );

    // Single-node convenience path agrees with the whole-subgraph pass.
    let targets: Vec<usize> = subgraph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.app_id == TARGET_APP_ID)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(targets.len(), 1, "target node must be in the subgraph");
    let node_embedding = encoder
        .encode_node(&subgraph, TARGET_APP_ID)
        .expect("encode_node must work on a loaded checkpoint");
    assert_eq!(node_embedding.len(), 16);
    assert!(
        node_embedding
            .iter()
            .zip(embeddings[targets[0]].iter())
            .all(|(a, b)| (a - b).abs() < 1e-6),
        "encode_node disagrees with encode_subgraph"
    );

    // Deterministic at inference (dropout is not applied on the forward path).
    let again = encoder
        .encode_subgraph(&subgraph)
        .expect("second encoding pass must work");
    assert_eq!(again, embeddings, "inference must be deterministic");

    // The checkpoint really carried weights: an untrained model of the same
    // architecture produces different embeddings for the same input.
    let fresh = CandleGnnEncoder::new(CandleGnnEncoderConfig {
        encoder_config: encoder_config(),
        checkpoint_path: None,
        norm_stats: loader.norm_stats().cloned(),
    })
    .expect("an untrained encoder must still build");
    let fresh_embeddings = fresh
        .encode_subgraph(&subgraph)
        .expect("untrained encoding must work");
    let max_delta = embeddings
        .iter()
        .zip(fresh_embeddings.iter())
        .flat_map(|(a, b)| a.iter().zip(b.iter()))
        .map(|(a, b)| (a - b).abs() as f64)
        .fold(0.0f64, f64::max);
    assert!(
        max_delta > 1e-3,
        "loaded checkpoint is indistinguishable from a fresh model (max delta {max_delta})"
    );

    println!(
        "gnn end-to-end: {} nodes, {} edges, {} epochs, best_val_loss={:.4}, test_auc={:.4}, \
         checkpoint={} bytes, max delta vs untrained={:.4}, total {:?}",
        NUM_NODES,
        EDGES.len(),
        result.epoch_history.len(),
        result.best_val_loss,
        result.final_test_auc,
        checkpoint_len,
        max_delta,
        started.elapsed()
    );
}
