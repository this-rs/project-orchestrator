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

use super::models::{CodeEdge, CodeEdgeType, CodeGraph, CodeNode, CodeNodeType};

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

        // Verify specific nodes exist
        assert!(graph.get_node("main").is_some());
        assert!(graph.get_node("respond").is_some());
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

    #[tokio::test]
    async fn test_extract_file_graph_edge_with_missing_node_ignored() {
        let store = MockGraphStore::new();
        let project = test_project();
        store.create_project(&project).await.unwrap();

        // Seed only 2 files but add an import to a path not in the project
        let files = ["src/a.rs", "src/b.rs"];
        let imports = [
            ("src/a.rs", "src/b.rs"),       // valid: both nodes exist
            ("src/a.rs", "src/missing.rs"),  // target not in project files
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
