//! Analytics engine — orchestrates the full pipeline.
//!
//! The `AnalyticsEngine` trait is the single entry point for all analytics
//! consumers (sync pipeline, MCP handlers). It encapsulates:
//!
//! 1. **Extraction**: Neo4j → petgraph via `GraphExtractor`
//! 2. **Computation**: PageRank, Betweenness, Louvain, Clustering, WCC
//! 3. **Persistence**: Write scores back to Neo4j via `AnalyticsWriter`
//!
//! The trait also enables mocking in downstream consumer tests.

use crate::neo4j::GraphStore;
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use super::algorithms::{compute_all, compute_context_cards, structural_dna, wl_subgraph_hash_all};
use super::enrichment::{CommunityEnricher, NoopCommunityEnricher};
use super::extraction::GraphExtractor;
use super::models::{AnalyticsConfig, FabricWeights, GraphAnalytics};
use super::writer::AnalyticsWriter;

// ============================================================================
// Output type
// ============================================================================

/// Combined analytics results for a full project analysis.
///
/// Contains separate analytics for the file-level and function-level graphs,
/// plus a timestamp of when the computation was performed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectAnalytics {
    /// Analytics computed on the file import graph
    pub file_analytics: GraphAnalytics,
    /// Analytics computed on the function call graph
    pub function_analytics: GraphAnalytics,
    /// When the analytics were computed
    pub computed_at: DateTime<Utc>,
}

// ============================================================================
// Trait
// ============================================================================

/// Analytics engine trait — single entry point for graph analytics.
///
/// Consumers use `Arc<dyn AnalyticsEngine>` for dependency injection.
/// The real implementation (`GraphAnalyticsEngine`) performs extract → compute → write.
/// A mock implementation (`MockAnalyticsEngine`) returns pre-configured results.
#[async_trait]
pub trait AnalyticsEngine: Send + Sync {
    /// Compute and persist analytics for a project's file import graph.
    ///
    /// Pipeline: extract file graph → compute_all → write scores to Neo4j → return analytics.
    async fn analyze_file_graph(&self, project_id: Uuid) -> Result<GraphAnalytics>;

    /// Compute and persist analytics for a project's function call graph.
    ///
    /// Pipeline: extract function graph → compute_all → write scores to Neo4j → return analytics.
    async fn analyze_function_graph(&self, project_id: Uuid) -> Result<GraphAnalytics>;

    /// Full analysis: compute both file and function graphs, persist scores.
    ///
    /// Returns a `ProjectAnalytics` with both result sets and a timestamp.
    async fn analyze_project(&self, project_id: Uuid) -> Result<ProjectAnalytics>;

    /// Compute analytics on the multi-layer fabric graph.
    ///
    /// The fabric graph combines IMPORTS + CO_CHANGED (and future layers)
    /// into a single weighted graph. Produces `fabric_pagerank`, `fabric_betweenness`,
    /// and `fabric_community_id` scores that reflect both structural AND temporal coupling.
    ///
    /// Pipeline: extract fabric graph → compute_all → write scores to Neo4j → return analytics.
    async fn analyze_fabric_graph(
        &self,
        project_id: Uuid,
        weights: &FabricWeights,
    ) -> Result<GraphAnalytics>;

    /// Detect business processes by scoring entry points, BFS traversal,
    /// deduplication, and classification.
    ///
    /// Pipeline: extract function graph → compute_all → score entry points →
    /// BFS trace → deduplicate → classify → persist Process nodes + STEP_IN_PROCESS edges.
    async fn detect_processes(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<crate::graph::process::Process>>;
}

// ============================================================================
// Concrete implementation
// ============================================================================

/// Real analytics engine backed by a `GraphStore`.
///
/// Composes `GraphExtractor` (extraction), `compute_all` (algorithms),
/// `CommunityEnricher` (optional LLM label enrichment),
/// and `AnalyticsWriter` (persistence) into a single pipeline.
pub struct GraphAnalyticsEngine {
    store: Arc<dyn GraphStore>,
    extractor: GraphExtractor,
    writer: AnalyticsWriter,
    config: AnalyticsConfig,
    enricher: Arc<dyn CommunityEnricher>,
}

impl GraphAnalyticsEngine {
    /// Create a new engine backed by the given GraphStore.
    ///
    /// Uses `NoopCommunityEnricher` by default (heuristic labels only).
    pub fn new(store: Arc<dyn GraphStore>, config: AnalyticsConfig) -> Self {
        Self {
            store: store.clone(),
            extractor: GraphExtractor::new(store.clone()),
            writer: AnalyticsWriter::new(store),
            config,
            enricher: Arc::new(NoopCommunityEnricher),
        }
    }

    /// Create a new engine with a custom community enricher.
    pub fn with_enricher(
        store: Arc<dyn GraphStore>,
        config: AnalyticsConfig,
        enricher: Arc<dyn CommunityEnricher>,
    ) -> Self {
        Self {
            store: store.clone(),
            extractor: GraphExtractor::new(store.clone()),
            writer: AnalyticsWriter::new(store),
            config,
            enricher,
        }
    }
}

#[async_trait]
impl AnalyticsEngine for GraphAnalyticsEngine {
    async fn analyze_file_graph(&self, project_id: Uuid) -> Result<GraphAnalytics> {
        // 1. Extract
        let graph = self.extractor.extract_file_graph(project_id).await?;

        // 2. Compute
        let mut analytics = compute_all(&graph, &self.config);

        // 2b. Enrich community labels (LLM or noop)
        if let Err(e) = self
            .enricher
            .enrich_labels(&mut analytics.communities, &graph, &analytics.metrics)
            .await
        {
            tracing::warn!(
                "Community enrichment failed, keeping heuristic labels: {}",
                e
            );
        }

        // 3. Persist base analytics
        self.writer.write_analytics(&analytics, &graph).await?;

        // 4. Compute and persist Structural DNA (depends on PageRank scores)
        let pr_scores: std::collections::HashMap<String, f64> = analytics
            .metrics
            .iter()
            .map(|(id, m)| (id.clone(), m.pagerank))
            .collect();

        let mut dna_map_for_cards = std::collections::HashMap::new();

        match structural_dna(&graph, &pr_scores, 10) {
            Ok(dna_map) if !dna_map.is_empty() => {
                if let Err(e) = self.writer.write_structural_dna(&dna_map, &graph).await {
                    tracing::warn!(
                        "Failed to persist structural DNA for project {}: {}",
                        project_id,
                        e
                    );
                } else {
                    tracing::debug!(
                        project_id = %project_id,
                        dna_count = dna_map.len(),
                        "Structural DNA computed and persisted"
                    );
                }
                dna_map_for_cards = dna_map;
            }
            Ok(_) => {
                tracing::debug!("Structural DNA: empty graph, skipping");
            }
            Err(e) => {
                tracing::warn!("Structural DNA computation failed: {}", e);
            }
        }

        // 5. Compute WL subgraph hashes (Plan 7)
        let wl_hashes = match wl_subgraph_hash_all(&graph, 2, 3) {
            Ok(hashes) => {
                tracing::debug!(
                    project_id = %project_id,
                    wl_count = hashes.len(),
                    "WL subgraph hashes computed (R=2, iterations=3)"
                );
                hashes
            }
            Err(e) => {
                tracing::warn!("WL subgraph hash computation failed: {}", e);
                std::collections::HashMap::new()
            }
        };

        // 6. Compute and persist Context Cards (aggregates analytics + DNA + WL hash)
        let context_cards = compute_context_cards(&graph, &analytics, &dna_map_for_cards, &wl_hashes);
        if !context_cards.is_empty() {
            if let Err(e) = self.store.batch_save_context_cards(&context_cards).await {
                tracing::warn!(
                    "Failed to persist context cards for project {}: {}",
                    project_id,
                    e
                );
            } else {
                tracing::debug!(
                    project_id = %project_id,
                    cards_count = context_cards.len(),
                    "Context cards computed and persisted"
                );
            }
        }

        Ok(analytics)
    }

    async fn analyze_function_graph(&self, project_id: Uuid) -> Result<GraphAnalytics> {
        // 1. Extract
        let graph = self.extractor.extract_function_graph(project_id).await?;

        // 2. Compute
        let mut analytics = compute_all(&graph, &self.config);

        // 2b. Enrich community labels (LLM or noop)
        if let Err(e) = self
            .enricher
            .enrich_labels(&mut analytics.communities, &graph, &analytics.metrics)
            .await
        {
            tracing::warn!(
                "Community enrichment failed, keeping heuristic labels: {}",
                e
            );
        }

        // 3. Persist
        self.writer.write_analytics(&analytics, &graph).await?;

        Ok(analytics)
    }

    async fn analyze_project(&self, project_id: Uuid) -> Result<ProjectAnalytics> {
        let file_analytics = self.analyze_file_graph(project_id).await?;
        let function_analytics = self.analyze_function_graph(project_id).await?;

        Ok(ProjectAnalytics {
            file_analytics,
            function_analytics,
            computed_at: Utc::now(),
        })
    }

    async fn analyze_fabric_graph(
        &self,
        project_id: Uuid,
        weights: &FabricWeights,
    ) -> Result<GraphAnalytics> {
        // 1. Extract multi-layer graph
        let graph = self
            .extractor
            .extract_fabric_graph(project_id, weights)
            .await?;

        // 2. Compute analytics (PageRank, Louvain, Betweenness all use edge weights)
        let mut analytics = compute_all(&graph, &self.config);

        // 2b. Enrich community labels (LLM or noop)
        if let Err(e) = self
            .enricher
            .enrich_labels(&mut analytics.communities, &graph, &analytics.metrics)
            .await
        {
            tracing::warn!(
                "Community enrichment failed, keeping heuristic labels: {}",
                e
            );
        }

        // 3. Persist to fabric_* properties (NOT the code-only properties)
        self.writer
            .write_fabric_analytics(&analytics, &graph)
            .await?;

        Ok(analytics)
    }

    async fn detect_processes(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<crate::graph::process::Process>> {
        use crate::graph::process::{self, ProcessConfig};
        use crate::neo4j::models::ProcessNode;

        // 1. Extract function CALLS graph
        let graph = self.extractor.extract_function_graph(project_id).await?;
        if graph.node_count() == 0 {
            return Ok(Vec::new());
        }

        // 2. Compute metrics (PageRank, communities, in/out degree)
        let analytics = compute_all(&graph, &self.config);

        // 3. Run process detection pipeline
        let config = ProcessConfig::default();
        let processes = process::detect_processes(&graph, &analytics.metrics, &config);

        if processes.is_empty() {
            return Ok(processes);
        }

        // 4. Persist: delete old processes, then upsert new ones
        if let Err(e) = self.store.delete_project_processes(project_id).await {
            tracing::warn!(
                "Failed to delete old processes for project {}: {}",
                project_id,
                e
            );
        }

        let process_nodes: Vec<ProcessNode> = processes
            .iter()
            .map(|p| ProcessNode {
                id: p.id.clone(),
                label: p.label.clone(),
                process_type: p.process_type.to_string(),
                step_count: p.steps.len() as u32,
                entry_point_id: p.entry_point_id.clone(),
                terminal_id: p.terminal_id.clone(),
                communities: p.communities.iter().copied().collect(),
                project_id: Some(project_id),
            })
            .collect();

        self.store.batch_upsert_processes(&process_nodes).await?;

        // 5. Create STEP_IN_PROCESS relationships
        let step_rels: Vec<(String, String, u32)> = processes
            .iter()
            .flat_map(|p| {
                p.steps
                    .iter()
                    .enumerate()
                    .map(move |(i, step_id)| (p.id.clone(), step_id.clone(), (i + 1) as u32))
            })
            .collect();

        self.store
            .batch_create_step_relationships(&step_rels)
            .await?;

        tracing::info!(
            project_id = %project_id,
            count = processes.len(),
            "Process detection complete"
        );

        Ok(processes)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::models::{FileNode, FunctionNode, Visibility};
    use crate::test_helpers::test_project;

    /// Seed a project with files forming two clusters connected by one edge.
    async fn seed_two_cluster_file_graph(store: &MockGraphStore, project_id: Uuid) {
        // Cluster A: 5 files, fully connected
        let cluster_a: Vec<String> = (0..5).map(|i| format!("src/api/file_a{}.rs", i)).collect();
        // Cluster B: 5 files, fully connected
        let cluster_b: Vec<String> = (0..5).map(|i| format!("src/db/file_b{}.rs", i)).collect();

        let all_files: Vec<String> = cluster_a.iter().chain(cluster_b.iter()).cloned().collect();

        // Seed files
        for path in &all_files {
            let file = FileNode {
                path: path.clone(),
                language: "rust".to_string(),
                hash: "abc123".to_string(),
                last_parsed: chrono::Utc::now(),
                project_id: Some(project_id),
            };
            store.upsert_file(&file).await.unwrap();
            store
                .project_files
                .write()
                .await
                .entry(project_id)
                .or_default()
                .push(path.clone());
        }

        // Seed intra-cluster imports (each file imports the next in cluster)
        let mut imports = store.import_relationships.write().await;
        for cluster in [&cluster_a, &cluster_b] {
            for i in 0..cluster.len() {
                for j in 0..cluster.len() {
                    if i != j {
                        imports
                            .entry(cluster[i].clone())
                            .or_default()
                            .push(cluster[j].clone());
                    }
                }
            }
        }
        // One bridge edge between clusters
        imports
            .entry(cluster_a[0].clone())
            .or_default()
            .push(cluster_b[0].clone());
    }

    /// Seed 8 functions with call relationships.
    async fn seed_functions(store: &MockGraphStore, project_id: Uuid) {
        let file_path = "src/main.rs";

        // Register file as project file
        store
            .project_files
            .write()
            .await
            .entry(project_id)
            .or_default()
            .push(file_path.to_string());

        let func_names = [
            "main",
            "setup",
            "handle_request",
            "validate",
            "parse",
            "transform",
            "serialize",
            "respond",
        ];

        for (i, name) in func_names.iter().enumerate() {
            let func = FunctionNode {
                name: name.to_string(),
                visibility: Visibility::Public,
                params: vec![],
                return_type: None,
                generics: vec![],
                is_async: false,
                is_unsafe: false,
                complexity: 1,
                file_path: file_path.to_string(),
                line_start: (i * 10) as u32,
                line_end: (i * 10 + 9) as u32,
                docstring: None,
            };
            store.upsert_function(&func).await.unwrap();
        }

        // Seed call relationships (qualified keys: "file_path::name")
        let calls = [
            ("main", "setup"),
            ("main", "handle_request"),
            ("handle_request", "validate"),
            ("handle_request", "parse"),
            ("validate", "parse"),
            ("parse", "transform"),
            ("transform", "serialize"),
            ("serialize", "respond"),
            ("setup", "validate"),
            ("respond", "serialize"),
        ];

        let mut cr = store.call_relationships.write().await;
        for (caller, callee) in &calls {
            let caller_id = format!("{}::{}", file_path, caller);
            cr.entry(caller_id).or_default().push(callee.to_string());
        }
    }

    #[tokio::test]
    async fn test_analyze_file_graph_two_clusters() {
        let store = Arc::new(MockGraphStore::new());
        let project = test_project();
        store.create_project(&project).await.unwrap();

        seed_two_cluster_file_graph(&store, project.id).await;

        let engine = GraphAnalyticsEngine::new(store.clone(), AnalyticsConfig::default());
        let analytics = engine.analyze_file_graph(project.id).await.unwrap();

        // 10 files total
        assert_eq!(analytics.node_count, 10);
        assert_eq!(analytics.metrics.len(), 10);

        // Should detect 2 communities (the two clusters)
        assert!(
            analytics.communities.len() >= 2,
            "Expected at least 2 communities, got {}",
            analytics.communities.len()
        );

        // Modularity should be positive (clear community structure)
        assert!(
            analytics.modularity > 0.0,
            "Expected positive modularity, got {}",
            analytics.modularity
        );

        // Verify scores were persisted to mock
        let fa = store.file_analytics.read().await;
        assert_eq!(fa.len(), 10, "All 10 file analytics should be persisted");

        // Check that a specific file has valid scores
        let file_a0 = &fa["src/api/file_a0.rs"];
        assert!(file_a0.pagerank > 0.0);
        assert!(!file_a0.community_label.is_empty());
    }

    #[tokio::test]
    async fn test_analyze_function_graph() {
        let store = Arc::new(MockGraphStore::new());
        let project = test_project();
        store.create_project(&project).await.unwrap();

        seed_functions(&store, project.id).await;

        let engine = GraphAnalyticsEngine::new(store.clone(), AnalyticsConfig::default());
        let analytics = engine.analyze_function_graph(project.id).await.unwrap();

        // 8 functions
        assert_eq!(analytics.node_count, 8);
        assert_eq!(analytics.metrics.len(), 8);
        assert_eq!(analytics.edge_count, 10);

        // Verify function scores were persisted
        let func_a = store.function_analytics.read().await;
        assert_eq!(func_a.len(), 8);

        let main_fn = &func_a["src/main.rs:main:0"];
        assert!(main_fn.pagerank > 0.0);
    }

    #[tokio::test]
    async fn test_analyze_project_full() {
        let store = Arc::new(MockGraphStore::new());
        let project = test_project();
        store.create_project(&project).await.unwrap();

        seed_two_cluster_file_graph(&store, project.id).await;
        seed_functions(&store, project.id).await;

        let engine = GraphAnalyticsEngine::new(store.clone(), AnalyticsConfig::default());
        let result = engine.analyze_project(project.id).await.unwrap();

        // File analytics
        assert_eq!(result.file_analytics.node_count, 10);
        assert!(result.file_analytics.communities.len() >= 2);

        // Function analytics
        assert_eq!(result.function_analytics.node_count, 8);
        assert_eq!(result.function_analytics.edge_count, 10);

        // Timestamp should be recent
        let now = Utc::now();
        let diff = now - result.computed_at;
        assert!(
            diff.num_seconds() < 5,
            "Timestamp should be recent, diff = {}s",
            diff.num_seconds()
        );

        // Both should be persisted
        let fa = store.file_analytics.read().await;
        let func_a = store.function_analytics.read().await;
        assert_eq!(fa.len(), 10);
        assert_eq!(func_a.len(), 8);
    }

    #[tokio::test]
    async fn test_analyze_empty_project() {
        let store = Arc::new(MockGraphStore::new());
        let project = test_project();
        store.create_project(&project).await.unwrap();

        let engine = GraphAnalyticsEngine::new(store.clone(), AnalyticsConfig::default());
        let result = engine.analyze_project(project.id).await.unwrap();

        // Both graphs should be empty but no errors
        assert_eq!(result.file_analytics.node_count, 0);
        assert_eq!(result.function_analytics.node_count, 0);
        assert!(result.file_analytics.metrics.is_empty());
        assert!(result.function_analytics.metrics.is_empty());

        // No analytics should be persisted
        let fa = store.file_analytics.read().await;
        let func_a = store.function_analytics.read().await;
        assert!(fa.is_empty());
        assert!(func_a.is_empty());
    }

    #[tokio::test]
    async fn test_analyze_single_file_no_imports() {
        let store = Arc::new(MockGraphStore::new());
        let project = test_project();
        store.create_project(&project).await.unwrap();

        // Seed a single file with no imports
        let file = FileNode {
            path: "src/lonely.rs".to_string(),
            language: "rust".to_string(),
            hash: "abc".to_string(),
            last_parsed: chrono::Utc::now(),
            project_id: Some(project.id),
        };
        store.upsert_file(&file).await.unwrap();
        store
            .project_files
            .write()
            .await
            .entry(project.id)
            .or_default()
            .push("src/lonely.rs".to_string());

        let engine = GraphAnalyticsEngine::new(store.clone(), AnalyticsConfig::default());
        let analytics = engine.analyze_file_graph(project.id).await.unwrap();

        assert_eq!(analytics.node_count, 1);
        assert_eq!(analytics.edge_count, 0);
        assert_eq!(analytics.communities.len(), 1);
        assert_eq!(analytics.communities[0].size, 1);
        assert_eq!(analytics.components.len(), 1);
        assert!(analytics.components[0].is_main);

        // Should detect orphan
        assert!(
            analytics
                .health
                .orphan_files
                .contains(&"src/lonely.rs".to_string()),
            "Single isolated file should be detected as orphan"
        );
    }
}
