//! Neo4j → petgraph extraction.
//!
//! Converts the knowledge graph stored in Neo4j into an in-memory `petgraph::DiGraph`
//! suitable for analytics computation. Uses the `GraphStore` trait (not raw Cypher)
//! to fetch nodes and relationships.
//!
//! The extraction is project-scoped: given a project ID, it fetches all code entities
//! and their relationships to build the directed graph.
//!
//! ## Two graph levels
//!
//! - **File graph**: nodes = files, edges = IMPORTS relationships
//! - **Function graph**: nodes = functions, edges = CALLS relationships

use crate::neo4j::GraphStore;
use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;

use super::models::{CodeEdge, CodeEdgeType, CodeGraph, CodeNode, CodeNodeType, FabricWeights};

/// Extracts code graphs from the knowledge graph (Neo4j) via the `GraphStore` trait.
///
/// Each extraction method performs at most 2 bulk queries:
/// one for nodes, one for edges.
pub struct GraphExtractor {
    store: Arc<dyn GraphStore>,
}

impl GraphExtractor {
    /// Create a new extractor backed by the given GraphStore.
    pub fn new(store: Arc<dyn GraphStore>) -> Self {
        Self { store }
    }

    /// Extract a file-level graph for a project.
    ///
    /// - Nodes: all `File` entities belonging to the project
    /// - Edges: `IMPORTS` relationships between files (within the same project)
    ///
    /// Returns a `CodeGraph` ready for analytics computation.
    pub async fn extract_file_graph(&self, project_id: Uuid) -> Result<CodeGraph> {
        // 1. Fetch all file nodes
        let files = self.store.list_project_files(project_id).await?;

        let mut graph = CodeGraph::with_capacity(files.len(), files.len() * 2);

        for file in &files {
            graph.add_node(CodeNode {
                id: file.path.clone(),
                node_type: CodeNodeType::File,
                path: Some(file.path.clone()),
                name: file
                    .path
                    .rsplit('/')
                    .next()
                    .unwrap_or(&file.path)
                    .to_string(),
                project_id: file.project_id.map(|id| id.to_string()),
            });
        }

        // 2. Fetch all import edges in bulk
        let edges = self.store.get_project_import_edges(project_id).await?;

        for (source, target) in &edges {
            // Only add edges where both nodes exist (robustness)
            if graph.get_index(source).is_some() && graph.get_index(target).is_some() {
                graph.add_edge(
                    source,
                    target,
                    CodeEdge {
                        edge_type: CodeEdgeType::Imports,
                        weight: 1.0,
                    },
                );
            }
        }

        // 3. Fetch EXTENDS edges (class inheritance)
        let extends_edges = self.store.get_project_extends_edges(project_id).await?;
        for (source, target) in &extends_edges {
            if graph.get_index(source).is_some() && graph.get_index(target).is_some() {
                graph.add_edge(
                    source,
                    target,
                    CodeEdge {
                        edge_type: CodeEdgeType::Extends,
                        weight: 0.95,
                    },
                );
            }
        }

        // 4. Fetch IMPLEMENTS edges (interface/protocol implementation)
        let implements_edges = self.store.get_project_implements_edges(project_id).await?;
        for (source, target) in &implements_edges {
            if graph.get_index(source).is_some() && graph.get_index(target).is_some() {
                graph.add_edge(
                    source,
                    target,
                    CodeEdge {
                        edge_type: CodeEdgeType::Implements,
                        weight: 0.85,
                    },
                );
            }
        }

        Ok(graph)
    }

    /// Extract a multi-layer fabric graph for a project.
    ///
    /// Combines multiple relationship types into a single petgraph,
    /// each with a configurable weight reflecting its coupling strength:
    /// - **IMPORTS** (structural code dependencies, weight: 0.8)
    /// - **CO_CHANGED** (temporal coupling from commits, weight: 0.4)
    ///
    /// Additional layers (AFFECTS, DISCUSSED, SYNAPSE) can be added by
    /// later tasks (T5.5, T5.6, T5.9) — the graph gracefully degrades
    /// when those edges don't exist yet.
    ///
    /// The resulting graph feeds into the same PageRank/Louvain/Betweenness
    /// algorithms, but produces "fabric" scores that reflect both structural
    /// AND temporal coupling patterns.
    pub async fn extract_fabric_graph(
        &self,
        project_id: Uuid,
        weights: &FabricWeights,
    ) -> Result<CodeGraph> {
        // 1. Fetch all file nodes (same as extract_file_graph)
        let files = self.store.list_project_files(project_id).await?;

        let mut graph = CodeGraph::with_capacity(files.len(), files.len() * 3);

        for file in &files {
            graph.add_node(CodeNode {
                id: file.path.clone(),
                node_type: CodeNodeType::File,
                path: Some(file.path.clone()),
                name: file
                    .path
                    .rsplit('/')
                    .next()
                    .unwrap_or(&file.path)
                    .to_string(),
                project_id: file.project_id.map(|id| id.to_string()),
            });
        }

        // 2. Layer 1: IMPORTS edges (structural)
        let import_edges = self.store.get_project_import_edges(project_id).await?;
        for (source, target) in &import_edges {
            if graph.get_index(source).is_some() && graph.get_index(target).is_some() {
                graph.add_edge(
                    source,
                    target,
                    CodeEdge {
                        edge_type: CodeEdgeType::Imports,
                        weight: weights.imports,
                    },
                );
            }
        }

        // 3. Layer 2: CO_CHANGED edges (temporal coupling)
        // Graceful degradation: if no CO_CHANGED data exists, this returns empty
        match self
            .store
            .get_co_change_graph(project_id, weights.co_changed_min_count, 10_000)
            .await
        {
            Ok(co_change_pairs) => {
                for pair in &co_change_pairs {
                    // Both nodes must exist in the file graph
                    if graph.get_index(&pair.file_a).is_some()
                        && graph.get_index(&pair.file_b).is_some()
                    {
                        // Scale CO_CHANGED weight by count (more co-changes = stronger coupling)
                        // Normalize: weight * min(count/10, 1.0) — cap at 10 co-changes
                        let count_factor = (pair.count as f64 / 10.0).min(1.0);
                        let edge_weight = weights.co_changed * count_factor;

                        // CO_CHANGED is bidirectional — add both directions
                        graph.add_edge(
                            &pair.file_a,
                            &pair.file_b,
                            CodeEdge {
                                edge_type: CodeEdgeType::CoChanged,
                                weight: edge_weight,
                            },
                        );
                        graph.add_edge(
                            &pair.file_b,
                            &pair.file_a,
                            CodeEdge {
                                edge_type: CodeEdgeType::CoChanged,
                                weight: edge_weight,
                            },
                        );
                    }
                }
            }
            Err(e) => {
                tracing::debug!(
                    project_id = %project_id,
                    error = %e,
                    "Fabric extraction: CO_CHANGED layer unavailable (graceful degradation)"
                );
            }
        }

        // 4. Layer 3: EXTENDS (class inheritance, very strong coupling)
        match self.store.get_project_extends_edges(project_id).await {
            Ok(extends_edges) => {
                for (source, target) in &extends_edges {
                    if graph.get_index(source).is_some() && graph.get_index(target).is_some() {
                        graph.add_edge(
                            source,
                            target,
                            CodeEdge {
                                edge_type: CodeEdgeType::Extends,
                                weight: weights.extends,
                            },
                        );
                    }
                }
                tracing::debug!(project_id = %project_id, extends_edges = extends_edges.len(), "Fabric extraction: EXTENDS layer added");
            }
            Err(e) => {
                tracing::debug!(project_id = %project_id, error = %e, "Fabric extraction: EXTENDS layer unavailable (graceful degradation)");
            }
        }

        // 5. Layer 4: IMPLEMENTS (interface implementation, strong coupling)
        match self.store.get_project_implements_edges(project_id).await {
            Ok(implements_edges) => {
                for (source, target) in &implements_edges {
                    if graph.get_index(source).is_some() && graph.get_index(target).is_some() {
                        graph.add_edge(
                            source,
                            target,
                            CodeEdge {
                                edge_type: CodeEdgeType::Implements,
                                weight: weights.implements,
                            },
                        );
                    }
                }
                tracing::debug!(project_id = %project_id, implements_edges = implements_edges.len(), "Fabric extraction: IMPLEMENTS layer added");
            }
            Err(e) => {
                tracing::debug!(project_id = %project_id, error = %e, "Fabric extraction: IMPLEMENTS layer unavailable (graceful degradation)");
            }
        }

        // Future layers (T5.5, T5.6):
        // - AFFECTS: Decision → File edges (requires get_project_affects_edges)
        // - DISCUSSED: Session → File edges (requires get_project_discussed_edges)

        // 6. Layer 5: SYNAPSE (neural connections bridged through Note→File)
        match self.store.get_project_synapse_edges(project_id).await {
            Ok(synapse_pairs) => {
                let mut added = 0;
                for (source, target, weight) in &synapse_pairs {
                    if let (Some(&src_idx), Some(&tgt_idx)) =
                        (graph.id_to_index.get(source), graph.id_to_index.get(target))
                    {
                        graph.graph.add_edge(
                            src_idx,
                            tgt_idx,
                            CodeEdge {
                                edge_type: CodeEdgeType::Synapse,
                                weight: weights.synapse * weight,
                            },
                        );
                        added += 1;
                    }
                }
                tracing::debug!(project_id = %project_id, synapse_edges = added, "Fabric extraction: SYNAPSE layer added");
            }
            Err(e) => {
                tracing::debug!(project_id = %project_id, error = %e, "Fabric extraction: SYNAPSE layer unavailable (graceful degradation)");
            }
        }

        Ok(graph)
    }

    /// Extract a function-level graph for a project.
    ///
    /// - Nodes: all functions that appear in CALLS relationships
    /// - Edges: `CALLS` relationships between functions (within the same project)
    ///
    /// Functions that neither call nor are called by anything are excluded
    /// (they have no edges and would be isolated nodes with no analytics value).
    pub async fn extract_function_graph(&self, project_id: Uuid) -> Result<CodeGraph> {
        // Fetch all call edges in bulk
        let edges = self.store.get_project_call_edges(project_id).await?;

        // Build the graph: nodes are discovered from edges
        let mut graph = CodeGraph::with_capacity(edges.len(), edges.len());

        for (caller, callee) in &edges {
            // Ensure both nodes exist (auto-created from edge endpoints)
            graph.add_node(CodeNode {
                id: caller.clone(),
                node_type: CodeNodeType::Function,
                path: None,
                name: caller.clone(),
                project_id: Some(project_id.to_string()),
            });
            graph.add_node(CodeNode {
                id: callee.clone(),
                node_type: CodeNodeType::Function,
                path: None,
                name: callee.clone(),
                project_id: Some(project_id.to_string()),
            });

            graph.add_edge(
                caller,
                callee,
                CodeEdge {
                    edge_type: CodeEdgeType::Calls,
                    weight: 1.0,
                },
            );
        }

        Ok(graph)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::models::{FunctionNode, Visibility};
    use crate::test_helpers::test_project;

    /// Seed a project with files and import relationships into the mock store.
    async fn seed_files_and_imports(
        store: &MockGraphStore,
        project_id: Uuid,
        files: &[&str],
        imports: &[(&str, &str)],
    ) {
        // Seed files using upsert_file + project_files registration
        for path in files {
            let file = crate::neo4j::models::FileNode {
                path: path.to_string(),
                language: "rust".to_string(),
                hash: "abc123".to_string(),
                last_parsed: chrono::Utc::now(),
                project_id: Some(project_id),
            };
            store.upsert_file(&file).await.unwrap();
            // Register in project_files (upsert_file doesn't always do this)
            store
                .project_files
                .write()
                .await
                .entry(project_id)
                .or_default()
                .push(path.to_string());
        }

        // Seed import relationships
        for (from, to) in imports {
            store
                .import_relationships
                .write()
                .await
                .entry(from.to_string())
                .or_default()
                .push(to.to_string());
        }
    }

    /// Seed a project with functions and call relationships into the mock store.
    /// `calls` uses simple function names; they are auto-qualified as "file_path::name".
    async fn seed_functions_and_calls(
        store: &MockGraphStore,
        project_id: Uuid,
        file_path: &str,
        func_names: &[&str],
        calls: &[(&str, &str)],
    ) {
        // Register the file as a project file
        store
            .project_files
            .write()
            .await
            .entry(project_id)
            .or_default()
            .push(file_path.to_string());

        // Seed functions (upsert_function stores them keyed as "file_path::name")
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

        // Seed call relationships using qualified IDs (file_path::name)
        // This matches the format used by create_call_relationship and the mock
        for (caller, callee) in calls {
            let caller_id = format!("{}::{}", file_path, caller);
            store
                .call_relationships
                .write()
                .await
                .entry(caller_id)
                .or_default()
                .push(callee.to_string());
        }
    }

    #[tokio::test]
    async fn test_extract_file_graph_5_files_6_imports() {
        let store = MockGraphStore::new();
        let project = test_project();
        store.create_project(&project).await.unwrap();

        let files = [
            "src/main.rs",
            "src/lib.rs",
            "src/api/mod.rs",
            "src/api/handlers.rs",
            "src/api/routes.rs",
        ];
        let imports = [
            ("src/main.rs", "src/lib.rs"),
            ("src/main.rs", "src/api/mod.rs"),
            ("src/api/mod.rs", "src/api/handlers.rs"),
            ("src/api/mod.rs", "src/api/routes.rs"),
            ("src/api/handlers.rs", "src/lib.rs"),
            ("src/api/routes.rs", "src/api/handlers.rs"),
        ];

        seed_files_and_imports(&store, project.id, &files, &imports).await;

        let extractor = GraphExtractor::new(Arc::new(store));
        let graph = extractor.extract_file_graph(project.id).await.unwrap();

        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 6);

        // Verify specific nodes exist
        assert!(graph.get_node("src/main.rs").is_some());
        assert!(graph.get_node("src/api/handlers.rs").is_some());

        // Verify node types
        let node = graph.get_node("src/main.rs").unwrap();
        assert_eq!(node.node_type, CodeNodeType::File);
        assert_eq!(node.name, "main.rs");
    }

    #[tokio::test]
    async fn test_extract_function_graph_8_funcs_10_calls() {
        let store = MockGraphStore::new();
        let project = test_project();
        store.create_project(&project).await.unwrap();

        let funcs = [
            "main",
            "setup",
            "handle_request",
            "validate",
            "parse",
            "transform",
            "serialize",
            "respond",
        ];
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

        seed_functions_and_calls(&store, project.id, "src/main.rs", &funcs, &calls).await;

        let extractor = GraphExtractor::new(Arc::new(store));
        let graph = extractor.extract_function_graph(project.id).await.unwrap();

        assert_eq!(graph.node_count(), 8);
        assert_eq!(graph.edge_count(), 10);

        // Verify specific nodes exist (function IDs are now "file_path:name:line_start")
        assert!(graph.get_node("src/main.rs:main:0").is_some());
        assert!(graph.get_node("src/main.rs:respond:70").is_some());
    }

    #[tokio::test]
    async fn test_extract_empty_project() {
        let store = MockGraphStore::new();
        let project = test_project();
        store.create_project(&project).await.unwrap();

        let extractor = GraphExtractor::new(Arc::new(store));

        let file_graph = extractor.extract_file_graph(project.id).await.unwrap();
        assert_eq!(file_graph.node_count(), 0);
        assert_eq!(file_graph.edge_count(), 0);

        let func_graph = extractor.extract_function_graph(project.id).await.unwrap();
        assert_eq!(func_graph.node_count(), 0);
        assert_eq!(func_graph.edge_count(), 0);
    }

    /// Seed structs with heritage relationships into the mock store.
    /// `extends` = vec of (child_name, child_file, parent_name, parent_file).
    /// `implements` = vec of (struct_name, struct_file, trait_name, trait_file).
    async fn seed_heritage(
        store: &MockGraphStore,
        _project_id: Uuid,
        extends: &[(&str, &str, &str, &str)],
        implements: &[(&str, &str, &str, &str)],
    ) {
        use crate::neo4j::models::{StructNode, TraitNode, Visibility};

        // Collect all structs (children + parents)
        let mut seen_structs = std::collections::HashSet::new();
        for (child_name, child_file, parent_name, parent_file) in extends {
            for (name, file) in [(child_name, child_file), (parent_name, parent_file)] {
                if seen_structs.insert((*name, *file)) {
                    store
                        .upsert_struct(&StructNode {
                            name: name.to_string(),
                            visibility: Visibility::Public,
                            generics: vec![],
                            file_path: file.to_string(),
                            line_start: 1,
                            line_end: 10,
                            docstring: None,
                            parent_class: None,
                            interfaces: vec![],
                        })
                        .await
                        .unwrap();
                }
            }
            // Seed the extends relationship
            store
                .call_relationships
                .write()
                .await
                .entry(format!("extends:{}", child_name))
                .or_default()
                .push(parent_name.to_string());
        }

        // Collect all structs and traits from implements
        for (struct_name, struct_file, trait_name, trait_file) in implements {
            if seen_structs.insert((*struct_name, *struct_file)) {
                store
                    .upsert_struct(&StructNode {
                        name: struct_name.to_string(),
                        visibility: Visibility::Public,
                        generics: vec![],
                        file_path: struct_file.to_string(),
                        line_start: 1,
                        line_end: 10,
                        docstring: None,
                        parent_class: None,
                        interfaces: vec![],
                    })
                    .await
                    .unwrap();
            }
            store
                .upsert_trait(&TraitNode {
                    name: trait_name.to_string(),
                    visibility: Visibility::Public,
                    generics: vec![],
                    file_path: trait_file.to_string(),
                    line_start: 1,
                    line_end: 10,
                    docstring: None,
                    is_external: false,
                    source: None,
                })
                .await
                .unwrap();
            // Seed the implements relationship
            store
                .call_relationships
                .write()
                .await
                .entry(format!("implements:{}", struct_name))
                .or_default()
                .push(trait_name.to_string());
        }
    }

    #[tokio::test]
    async fn test_extract_file_graph_with_heritage_edges() {
        let store = MockGraphStore::new();
        let project = test_project();
        store.create_project(&project).await.unwrap();

        // 4 files: models/animal.rs, models/dog.rs, traits/serializable.rs, traits/comparable.rs
        let files = [
            "src/models/animal.rs",
            "src/models/dog.rs",
            "src/traits/serializable.rs",
            "src/traits/comparable.rs",
        ];
        // 1 import: dog.rs imports animal.rs
        let imports = [("src/models/dog.rs", "src/models/animal.rs")];

        seed_files_and_imports(&store, project.id, &files, &imports).await;

        // Dog extends Animal (cross-file)
        let extends = [("Dog", "src/models/dog.rs", "Animal", "src/models/animal.rs")];
        // Dog implements Serializable and Comparable (cross-file)
        let implements_rels = [
            (
                "Dog",
                "src/models/dog.rs",
                "Serializable",
                "src/traits/serializable.rs",
            ),
            (
                "Dog",
                "src/models/dog.rs",
                "Comparable",
                "src/traits/comparable.rs",
            ),
        ];

        seed_heritage(&store, project.id, &extends, &implements_rels).await;

        let extractor = GraphExtractor::new(Arc::new(store));
        let graph = extractor.extract_file_graph(project.id).await.unwrap();

        assert_eq!(graph.node_count(), 4, "4 file nodes");
        // 1 import + 1 extends + 2 implements = 4 edges
        assert_eq!(graph.edge_count(), 4, "1 import + 1 extends + 2 implements");

        // Verify edge types
        let edge_types: Vec<CodeEdgeType> = graph
            .graph
            .edge_weights()
            .map(|e| e.edge_type.clone())
            .collect();
        assert!(
            edge_types.contains(&CodeEdgeType::Imports),
            "has import edge"
        );
        assert!(
            edge_types.contains(&CodeEdgeType::Extends),
            "has extends edge"
        );
        assert!(
            edge_types.contains(&CodeEdgeType::Implements),
            "has implements edge"
        );

        // Verify weights
        let extends_edge = graph
            .graph
            .edge_weights()
            .find(|e| e.edge_type == CodeEdgeType::Extends)
            .unwrap();
        assert!(
            (extends_edge.weight - 0.95).abs() < 0.001,
            "extends weight = 0.95"
        );

        let implements_edge = graph
            .graph
            .edge_weights()
            .find(|e| e.edge_type == CodeEdgeType::Implements)
            .unwrap();
        assert!(
            (implements_edge.weight - 0.85).abs() < 0.001,
            "implements weight = 0.85"
        );
    }

    #[tokio::test]
    async fn test_extract_fabric_graph_with_heritage_layers() {
        let store = MockGraphStore::new();
        let project = test_project();
        store.create_project(&project).await.unwrap();

        let files = ["src/base.rs", "src/child.rs", "src/iface.rs"];
        let imports = [("src/child.rs", "src/base.rs")];

        seed_files_and_imports(&store, project.id, &files, &imports).await;

        // child extends base (cross-file)
        let extends = [("Child", "src/child.rs", "Base", "src/base.rs")];
        // child implements iface (cross-file)
        let implements_rels = [("Child", "src/child.rs", "MyTrait", "src/iface.rs")];

        seed_heritage(&store, project.id, &extends, &implements_rels).await;

        let extractor = GraphExtractor::new(Arc::new(store));
        let weights = FabricWeights::default();
        let graph = extractor
            .extract_fabric_graph(project.id, &weights)
            .await
            .unwrap();

        assert_eq!(graph.node_count(), 3, "3 file nodes");
        // 1 import (weight 0.8) + 1 extends (weight 0.95) + 1 implements (weight 0.85) = 3 edges
        assert_eq!(graph.edge_count(), 3, "1 import + 1 extends + 1 implements");

        // Verify fabric weights are applied (not hardcoded)
        let extends_edge = graph
            .graph
            .edge_weights()
            .find(|e| e.edge_type == CodeEdgeType::Extends)
            .unwrap();
        assert!(
            (extends_edge.weight - weights.extends).abs() < 0.001,
            "extends uses FabricWeights.extends"
        );

        let implements_edge = graph
            .graph
            .edge_weights()
            .find(|e| e.edge_type == CodeEdgeType::Implements)
            .unwrap();
        assert!(
            (implements_edge.weight - weights.implements).abs() < 0.001,
            "implements uses FabricWeights.implements"
        );
    }

    #[tokio::test]
    async fn test_extract_file_graph_edge_with_missing_node_ignored() {
        let store = MockGraphStore::new();
        let project = test_project();
        store.create_project(&project).await.unwrap();

        // Seed only 2 files but add an import to a path not in the project
        let files = ["src/a.rs", "src/b.rs"];
        let imports = [
            ("src/a.rs", "src/b.rs"),       // valid: both nodes exist
            ("src/a.rs", "src/missing.rs"), // target not in project files
        ];

        seed_files_and_imports(&store, project.id, &files, &imports).await;

        let extractor = GraphExtractor::new(Arc::new(store));
        let graph = extractor.extract_file_graph(project.id).await.unwrap();

        assert_eq!(graph.node_count(), 2);
        // Only the valid edge should be present — missing.rs was filtered by
        // get_project_import_edges (scoped to project) OR by the robustness check
        assert_eq!(graph.edge_count(), 1);
    }
}
