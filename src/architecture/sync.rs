//! Writing a derived topology into the graph.
//!
//! This runs after every sync, so it has to be safe to replay: components are
//! upserted on (workspace, name) and dependency edges are already a `MERGE`.
//! `create_component` is a bare `CREATE` with no uniqueness constraint, which is
//! why it is not used here — replaying it would stack duplicates forever.
//!
//! The boundary this respects: a derivation may refine what it previously
//! derived, and may add what nobody recorded, but it never overwrites what a
//! person wrote. Descriptions and tags already present are left alone, and
//! components nobody derived are left in place.

use std::path::PathBuf;
use std::sync::Arc;

use uuid::Uuid;

use crate::neo4j::models::DerivedComponentWrite;
use crate::neo4j::traits::GraphStore;

use super::derive::{derive_topology, ProjectInput};
use super::DERIVED_CONFIG_KEY;

/// What a derivation run changed, for logs and for the API response.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct DerivationOutcome {
    pub components_written: usize,
    pub edges_written: usize,
    /// Projects skipped because their root path is not readable — a project may
    /// be registered long after the directory it pointed at has moved.
    pub projects_skipped: usize,
}

/// Read the repository's origin remote, used to recognise a git dependency as
/// another workspace project.
fn git_remote(root: &std::path::Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["remote", "get-url", "origin"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!url.is_empty()).then_some(url)
}

/// Derive the topology of one workspace from its projects' source trees and
/// write it to the graph.
pub async fn derive_and_store_workspace(
    graph: Arc<dyn GraphStore>,
    workspace_id: Uuid,
) -> anyhow::Result<DerivationOutcome> {
    let projects = graph.list_workspace_projects(workspace_id).await?;
    let mut outcome = DerivationOutcome::default();

    let mut inputs: Vec<ProjectInput> = Vec::new();
    for project in &projects {
        let root = PathBuf::from(&project.root_path);
        // A project with no readable root contributes nothing; skipping it is
        // better than failing the run for the whole workspace.
        if project.root_path.is_empty() || !root.is_dir() {
            outcome.projects_skipped += 1;
            continue;
        }
        inputs.push(ProjectInput {
            name: project.name.clone(),
            git_remote: git_remote(&root),
            root,
        });
    }

    if inputs.is_empty() {
        return Ok(outcome);
    }

    let topology = derive_topology(&inputs);

    // Components first: an edge needs both endpoints to exist, and
    // `add_component_dependency` matches on them rather than creating them.
    let mut id_by_name = std::collections::HashMap::new();
    for component in &topology.components {
        let config = serde_json::json!({
            DERIVED_CONFIG_KEY: component.provenance,
        });
        let id = graph
            .upsert_derived_component(DerivedComponentWrite {
                workspace_id,
                name: component.name.clone(),
                component_type: component.component_type.clone(),
                description: component.description.clone(),
                runtime: component.runtime.clone(),
                tags: component.tags.clone(),
                config,
            })
            .await?;
        id_by_name.insert(component.name.clone(), id);
        outcome.components_written += 1;

        // Tie the component back to the project it is, so the view can link to it.
        if let Some(project_name) = &component.project_name {
            if let Some(project) = projects.iter().find(|p| &p.name == project_name) {
                graph.map_component_to_project(id, project.id).await?;
            }
        }
    }

    for edge in &topology.edges {
        let (Some(&from), Some(&to)) = (id_by_name.get(&edge.from), id_by_name.get(&edge.to))
        else {
            // Both ends were just upserted, so this only happens if a component
            // failed to write. Skipping keeps the rest of the topology intact.
            continue;
        };
        graph
            .add_component_dependency(from, to, edge.protocol.clone(), edge.required)
            .await?;
        outcome.edges_written += 1;
    }

    Ok(outcome)
}

/// Derive every workspace the given project belongs to.
///
/// Architecture is a workspace-level property — an edge from one project to
/// another only exists relative to the workspace holding both — so a project
/// sync has to re-derive the whole workspace, not just that project.
pub async fn derive_for_project(
    graph: Arc<dyn GraphStore>,
    project_id: Uuid,
) -> anyhow::Result<DerivationOutcome> {
    let mut total = DerivationOutcome::default();

    for workspace in graph.list_workspaces().await? {
        let projects = graph.list_workspace_projects(workspace.id).await?;
        if !projects.iter().any(|p| p.id == project_id) {
            continue;
        }
        let outcome = derive_and_store_workspace(graph.clone(), workspace.id).await?;
        total.components_written += outcome.components_written;
        total.edges_written += outcome.edges_written;
        total.projects_skipped += outcome.projects_skipped;
    }

    Ok(total)
}

/// Fire-and-forget derivation, for the sync paths.
///
/// Best-effort on purpose: a failure here must never fail a sync. The
/// architecture view going stale is a far smaller problem than a sync that
/// refuses to complete, and the heartbeat check picks up whatever this misses.
pub fn spawn_derive_architecture(graph: Arc<dyn GraphStore>, project_id: Uuid) {
    tokio::spawn(async move {
        match derive_for_project(graph, project_id).await {
            Ok(outcome) if outcome.components_written > 0 => {
                tracing::info!(
                    components = outcome.components_written,
                    edges = outcome.edges_written,
                    skipped = outcome.projects_skipped,
                    "architecture derived"
                );
            }
            Ok(_) => {}
            Err(e) => tracing::warn!(error = %e, "architecture derivation failed"),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::models::{ComponentType, ProjectNode, WorkspaceNode};

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "arch-sync-{}-{}-{:?}",
            tag,
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    async fn workspace_with_project(
        store: &MockGraphStore,
        root: &std::path::Path,
        project_name: &str,
    ) -> (Uuid, Uuid) {
        let ws = WorkspaceNode {
            id: Uuid::new_v4(),
            name: "ws".into(),
            slug: "ws".into(),
            description: None,
            created_at: chrono::Utc::now(),
            updated_at: None,
            metadata: serde_json::json!({}),
        };
        store.create_workspace(&ws).await.unwrap();

        let project = ProjectNode {
            id: Uuid::new_v4(),
            name: project_name.into(),
            slug: project_name.into(),
            root_path: root.to_string_lossy().to_string(),
            description: None,
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
            profile: Default::default(),
        };
        store.create_project(&project).await.unwrap();
        store
            .add_project_to_workspace(ws.id, project.id)
            .await
            .unwrap();
        (ws.id, project.id)
    }

    #[tokio::test]
    async fn writes_components_and_edges_for_a_workspace() {
        let root = temp_root("write");
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"api\"\n\n[dependencies]\nneo4rs = \"0.8\"\naxum = \"0.8\"\n",
        )
        .unwrap();

        let store = MockGraphStore::new();
        let (ws_id, _) = workspace_with_project(&store, &root, "api").await;
        let graph: Arc<dyn GraphStore> = Arc::new(store);

        let outcome = derive_and_store_workspace(graph.clone(), ws_id)
            .await
            .unwrap();
        assert_eq!(outcome.components_written, 2);
        assert_eq!(outcome.edges_written, 1);

        let components = graph.list_components(ws_id).await.unwrap();
        let neo4j = components.iter().find(|c| c.name == "Neo4j").unwrap();
        assert_eq!(neo4j.component_type, ComponentType::Database);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn running_twice_does_not_duplicate_anything() {
        // The reason this writes through an upsert rather than create_component:
        // it runs on every sync, and a bare CREATE would stack a new Neo4j node
        // each time.
        let root = temp_root("idem");
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"api\"\n\n[dependencies]\nneo4rs = \"0.8\"\n",
        )
        .unwrap();

        let store = MockGraphStore::new();
        let (ws_id, _) = workspace_with_project(&store, &root, "api").await;
        let graph: Arc<dyn GraphStore> = Arc::new(store);

        derive_and_store_workspace(graph.clone(), ws_id)
            .await
            .unwrap();
        let after_first = graph.list_components(ws_id).await.unwrap().len();
        derive_and_store_workspace(graph.clone(), ws_id)
            .await
            .unwrap();
        derive_and_store_workspace(graph.clone(), ws_id)
            .await
            .unwrap();
        let after_third = graph.list_components(ws_id).await.unwrap().len();

        assert_eq!(after_first, after_third);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn records_where_each_component_came_from() {
        // A graph nobody typed is only worth trusting if every node can be
        // traced back to a file.
        let root = temp_root("prov");
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"api\"\n\n[dependencies]\nneo4rs = \"0.8\"\n",
        )
        .unwrap();

        let store = MockGraphStore::new();
        let (ws_id, _) = workspace_with_project(&store, &root, "api").await;
        let graph: Arc<dyn GraphStore> = Arc::new(store);

        derive_and_store_workspace(graph.clone(), ws_id)
            .await
            .unwrap();
        let components = graph.list_components(ws_id).await.unwrap();
        let neo4j = components.iter().find(|c| c.name == "Neo4j").unwrap();

        let provenance = &neo4j.config[DERIVED_CONFIG_KEY];
        assert_eq!(provenance["method"], "manifest");
        assert_eq!(provenance["file"], "Cargo.toml");
        assert_eq!(provenance["package"], "neo4rs");
        assert!(provenance["line"].as_u64().unwrap() > 0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn a_project_whose_directory_is_gone_is_skipped_not_fatal() {
        let store = MockGraphStore::new();
        let missing = PathBuf::from("/nonexistent/path/to/nowhere");
        let (ws_id, _) = workspace_with_project(&store, &missing, "ghost").await;
        let graph: Arc<dyn GraphStore> = Arc::new(store);

        let outcome = derive_and_store_workspace(graph, ws_id).await.unwrap();
        assert_eq!(outcome.projects_skipped, 1);
        assert_eq!(outcome.components_written, 0);
    }

    #[tokio::test]
    async fn a_workspace_with_no_projects_writes_nothing() {
        let store = MockGraphStore::new();
        let ws = WorkspaceNode {
            id: Uuid::new_v4(),
            name: "empty".into(),
            slug: "empty".into(),
            description: None,
            created_at: chrono::Utc::now(),
            updated_at: None,
            metadata: serde_json::json!({}),
        };
        store.create_workspace(&ws).await.unwrap();
        let graph: Arc<dyn GraphStore> = Arc::new(store);

        let outcome = derive_and_store_workspace(graph, ws.id).await.unwrap();
        assert_eq!(outcome, DerivationOutcome::default());
    }

    #[tokio::test]
    async fn syncing_one_project_rederives_its_whole_workspace() {
        // An edge between two projects only exists relative to the workspace
        // holding both, so a per-project derivation could never find one.
        let root = temp_root("wsscope");
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"api\"\n\n[dependencies]\nneo4rs = \"0.8\"\n",
        )
        .unwrap();

        let store = MockGraphStore::new();
        let (ws_id, project_id) = workspace_with_project(&store, &root, "api").await;
        let graph: Arc<dyn GraphStore> = Arc::new(store);

        let outcome = derive_for_project(graph.clone(), project_id).await.unwrap();
        assert!(outcome.components_written > 0);
        assert!(!graph.list_components(ws_id).await.unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }
}
