//! Integration tests for workspace functionality
//!
//! These tests require Neo4j and Meilisearch to be running.
//! Run with: cargo test --test workspace_tests

use project_orchestrator::neo4j::models::*;
use project_orchestrator::{AppState, Config};
use uuid::Uuid;

/// Get test configuration from environment or use defaults
fn test_config() -> Config {
    Config {
        neo4j_uri: std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into()),
        neo4j_user: std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into()),
        neo4j_password: std::env::var("NEO4J_PASSWORD")
            .unwrap_or_else(|_| "orchestrator123".into()),
        meilisearch_url: std::env::var("MEILISEARCH_URL")
            .unwrap_or_else(|_| "http://localhost:7700".into()),
        meilisearch_key: std::env::var("MEILISEARCH_KEY")
            .unwrap_or_else(|_| "orchestrator-meili-key-change-me".into()),
        workspace_path: ".".into(),
        server_port: 8080,
    }
}

/// Check if backends are available
async fn backends_available() -> bool {
    let config = test_config();

    // Check Meilisearch
    let meili_ok = reqwest::get(format!("{}/health", config.meilisearch_url))
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    if !meili_ok {
        eprintln!("Meilisearch not available at {}", config.meilisearch_url);
        return false;
    }

    // Check Neo4j
    let neo4j_ok = neo4rs::Graph::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    .is_ok();

    if !neo4j_ok {
        eprintln!("Neo4j not available at {}", config.neo4j_uri);
        return false;
    }

    true
}

// ============================================================================
// Workspace CRUD Tests
// ============================================================================

#[tokio::test]
async fn test_workspace_crud() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Test Workspace".to_string(),
        slug: format!("test-workspace-{}", Uuid::new_v4()),
        description: Some("A test workspace".to_string()),
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({"test": true}),
    };

    let result = state.neo4j.create_workspace(&workspace).await;
    assert!(
        result.is_ok(),
        "Should create workspace: {:?}",
        result.err()
    );

    // Get workspace by ID
    let retrieved = state.neo4j.get_workspace(workspace.id).await.unwrap();
    assert!(retrieved.is_some(), "Should retrieve workspace by ID");
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.name, workspace.name);

    // Get workspace by slug
    let by_slug = state
        .neo4j
        .get_workspace_by_slug(&workspace.slug)
        .await
        .unwrap();
    assert!(by_slug.is_some(), "Should retrieve workspace by slug");

    // Update workspace
    let result = state
        .neo4j
        .update_workspace(
            workspace.id,
            Some("Updated Workspace".to_string()),
            Some("Updated description".to_string()),
            None,
        )
        .await;
    assert!(result.is_ok(), "Should update workspace");

    let updated = state
        .neo4j
        .get_workspace(workspace.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(updated.name, "Updated Workspace");

    // List workspaces
    let workspaces = state.neo4j.list_workspaces().await.unwrap();
    assert!(workspaces.iter().any(|w| w.id == workspace.id));

    // Delete workspace
    let result = state.neo4j.delete_workspace(workspace.id).await;
    assert!(result.is_ok(), "Should delete workspace");

    let deleted = state.neo4j.get_workspace(workspace.id).await.unwrap();
    assert!(deleted.is_none(), "Workspace should be deleted");
}

#[tokio::test]
async fn test_workspace_project_association() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Project Association Test".to_string(),
        slug: format!("proj-assoc-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Create project
    let project = ProjectNode {
        id: Uuid::new_v4(),
        name: "Test Project".to_string(),
        slug: format!("test-project-{}", Uuid::new_v4()),
        root_path: "/tmp/test".to_string(),
        description: None,
        created_at: chrono::Utc::now(),
        last_synced: None,
    };
    state.neo4j.create_project(&project).await.unwrap();

    // Add project to workspace
    let result = state
        .neo4j
        .add_project_to_workspace(workspace.id, project.id)
        .await;
    assert!(result.is_ok(), "Should add project to workspace");

    // List workspace projects
    let projects = state
        .neo4j
        .list_workspace_projects(workspace.id)
        .await
        .unwrap();
    assert_eq!(projects.len(), 1);
    assert_eq!(projects[0].id, project.id);

    // Get project workspace
    let ws = state.neo4j.get_project_workspace(project.id).await.unwrap();
    assert!(ws.is_some());
    assert_eq!(ws.unwrap().id, workspace.id);

    // Remove project from workspace
    let result = state
        .neo4j
        .remove_project_from_workspace(workspace.id, project.id)
        .await;
    assert!(result.is_ok(), "Should remove project from workspace");

    let projects = state
        .neo4j
        .list_workspace_projects(workspace.id)
        .await
        .unwrap();
    assert!(projects.is_empty());

    // Cleanup
    state.neo4j.delete_project(project.id).await.unwrap();
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}

// ============================================================================
// Workspace Milestone Tests
// ============================================================================

#[tokio::test]
async fn test_workspace_milestone_crud() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Milestone Test Workspace".to_string(),
        slug: format!("milestone-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Create workspace milestone
    let milestone = WorkspaceMilestoneNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        title: "Test Milestone".to_string(),
        description: Some("A test milestone".to_string()),
        status: MilestoneStatus::Open,
        target_date: Some(chrono::Utc::now() + chrono::Duration::days(30)),
        closed_at: None,
        created_at: chrono::Utc::now(),
        tags: vec!["test".to_string()],
    };

    let result = state.neo4j.create_workspace_milestone(&milestone).await;
    assert!(
        result.is_ok(),
        "Should create milestone: {:?}",
        result.err()
    );

    // Get milestone
    let retrieved = state
        .neo4j
        .get_workspace_milestone(milestone.id)
        .await
        .unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().title, milestone.title);

    // List milestones
    let milestones = state
        .neo4j
        .list_workspace_milestones(workspace.id)
        .await
        .unwrap();
    assert_eq!(milestones.len(), 1);

    // Update milestone
    state
        .neo4j
        .update_workspace_milestone(
            milestone.id,
            Some("Updated Milestone".to_string()),
            None,
            Some(MilestoneStatus::Closed),
            None,
        )
        .await
        .unwrap();

    let updated = state
        .neo4j
        .get_workspace_milestone(milestone.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(updated.title, "Updated Milestone");

    // Cleanup
    state
        .neo4j
        .delete_workspace_milestone(milestone.id)
        .await
        .unwrap();
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}

// ============================================================================
// Resource Tests
// ============================================================================

#[tokio::test]
async fn test_resource_crud() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Resource Test Workspace".to_string(),
        slug: format!("resource-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Create resource
    let resource = ResourceNode {
        id: Uuid::new_v4(),
        workspace_id: Some(workspace.id),
        project_id: None,
        name: "API Contract".to_string(),
        resource_type: ResourceType::ApiContract,
        file_path: "/specs/openapi.yaml".to_string(),
        url: Some("https://api.example.com/docs".to_string()),
        format: Some("openapi".to_string()),
        version: Some("1.0.0".to_string()),
        description: Some("Main API contract".to_string()),
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };

    let result = state.neo4j.create_resource(&resource).await;
    assert!(result.is_ok(), "Should create resource: {:?}", result.err());

    // Get resource
    let retrieved = state.neo4j.get_resource(resource.id).await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().name, resource.name);

    // List workspace resources
    let resources = state
        .neo4j
        .list_workspace_resources(workspace.id)
        .await
        .unwrap();
    assert_eq!(resources.len(), 1);

    // Cleanup
    state.neo4j.delete_resource(resource.id).await.unwrap();
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}

#[tokio::test]
async fn test_resource_project_links() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace and projects
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Resource Links Test".to_string(),
        slug: format!("res-links-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    let api_project = ProjectNode {
        id: Uuid::new_v4(),
        name: "API Project".to_string(),
        slug: format!("api-project-{}", Uuid::new_v4()),
        root_path: "/api".to_string(),
        description: None,
        created_at: chrono::Utc::now(),
        last_synced: None,
    };
    state.neo4j.create_project(&api_project).await.unwrap();

    let frontend_project = ProjectNode {
        id: Uuid::new_v4(),
        name: "Frontend Project".to_string(),
        slug: format!("frontend-project-{}", Uuid::new_v4()),
        root_path: "/frontend".to_string(),
        description: None,
        created_at: chrono::Utc::now(),
        last_synced: None,
    };
    state.neo4j.create_project(&frontend_project).await.unwrap();

    // Create resource
    let resource = ResourceNode {
        id: Uuid::new_v4(),
        workspace_id: Some(workspace.id),
        project_id: None,
        name: "Shared API".to_string(),
        resource_type: ResourceType::ApiContract,
        file_path: "/specs/api.yaml".to_string(),
        url: None,
        format: Some("openapi".to_string()),
        version: None,
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_resource(&resource).await.unwrap();

    // Link projects
    state
        .neo4j
        .link_project_implements_resource(api_project.id, resource.id)
        .await
        .unwrap();
    state
        .neo4j
        .link_project_uses_resource(frontend_project.id, resource.id)
        .await
        .unwrap();

    // Get implementers
    let implementers = state
        .neo4j
        .get_resource_implementers(resource.id)
        .await
        .unwrap();
    assert_eq!(implementers.len(), 1);
    assert_eq!(implementers[0].id, api_project.id);

    // Get consumers
    let consumers = state
        .neo4j
        .get_resource_consumers(resource.id)
        .await
        .unwrap();
    assert_eq!(consumers.len(), 1);
    assert_eq!(consumers[0].id, frontend_project.id);

    // Cleanup
    state.neo4j.delete_resource(resource.id).await.unwrap();
    state.neo4j.delete_project(api_project.id).await.unwrap();
    state
        .neo4j
        .delete_project(frontend_project.id)
        .await
        .unwrap();
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}

// ============================================================================
// Component & Topology Tests
// ============================================================================

#[tokio::test]
async fn test_component_crud() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Component Test Workspace".to_string(),
        slug: format!("component-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Create component
    let component = ComponentNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        name: "API Service".to_string(),
        component_type: ComponentType::Service,
        description: Some("Main API service".to_string()),
        runtime: Some("docker".to_string()),
        config: serde_json::json!({"port": 8080}),
        created_at: chrono::Utc::now(),
        tags: vec!["api".to_string(), "backend".to_string()],
    };

    let result = state.neo4j.create_component(&component).await;
    assert!(
        result.is_ok(),
        "Should create component: {:?}",
        result.err()
    );

    // Get component
    let retrieved = state.neo4j.get_component(component.id).await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().name, component.name);

    // List components
    let components = state.neo4j.list_components(workspace.id).await.unwrap();
    assert_eq!(components.len(), 1);

    // Cleanup
    state.neo4j.delete_component(component.id).await.unwrap();
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}

#[tokio::test]
async fn test_component_dependencies() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Component Deps Test".to_string(),
        slug: format!("comp-deps-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Create components
    let api = ComponentNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        name: "API".to_string(),
        component_type: ComponentType::Service,
        description: None,
        runtime: None,
        config: serde_json::json!({}),
        created_at: chrono::Utc::now(),
        tags: vec![],
    };
    state.neo4j.create_component(&api).await.unwrap();

    let db = ComponentNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        name: "Database".to_string(),
        component_type: ComponentType::Database,
        description: None,
        runtime: None,
        config: serde_json::json!({}),
        created_at: chrono::Utc::now(),
        tags: vec![],
    };
    state.neo4j.create_component(&db).await.unwrap();

    // Add dependency
    let result = state
        .neo4j
        .add_component_dependency(api.id, db.id, Some("tcp".to_string()), true)
        .await;
    assert!(result.is_ok(), "Should add dependency");

    // Get topology
    let topology = state
        .neo4j
        .get_workspace_topology(workspace.id)
        .await
        .unwrap();
    assert_eq!(topology.len(), 2);

    // Find API component and check its dependencies
    let api_entry = topology.iter().find(|(c, _, _)| c.id == api.id).unwrap();
    assert_eq!(api_entry.2.len(), 1);
    assert_eq!(api_entry.2[0].to_id, db.id);

    // Remove dependency
    state
        .neo4j
        .remove_component_dependency(api.id, db.id)
        .await
        .unwrap();

    // Cleanup
    state.neo4j.delete_component(api.id).await.unwrap();
    state.neo4j.delete_component(db.id).await.unwrap();
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}

#[tokio::test]
async fn test_component_project_mapping() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Component Mapping Test".to_string(),
        slug: format!("comp-map-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Create project
    let project = ProjectNode {
        id: Uuid::new_v4(),
        name: "API Codebase".to_string(),
        slug: format!("api-codebase-{}", Uuid::new_v4()),
        root_path: "/code/api".to_string(),
        description: None,
        created_at: chrono::Utc::now(),
        last_synced: None,
    };
    state.neo4j.create_project(&project).await.unwrap();

    // Create component
    let component = ComponentNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        name: "API Service".to_string(),
        component_type: ComponentType::Service,
        description: None,
        runtime: None,
        config: serde_json::json!({}),
        created_at: chrono::Utc::now(),
        tags: vec![],
    };
    state.neo4j.create_component(&component).await.unwrap();

    // Map component to project
    let result = state
        .neo4j
        .map_component_to_project(component.id, project.id)
        .await;
    assert!(result.is_ok(), "Should map component to project");

    // Get topology and check mapping
    let topology = state
        .neo4j
        .get_workspace_topology(workspace.id)
        .await
        .unwrap();
    assert_eq!(topology.len(), 1);
    assert_eq!(topology[0].1, Some("API Codebase".to_string()));

    // Cleanup
    state.neo4j.delete_component(component.id).await.unwrap();
    state.neo4j.delete_project(project.id).await.unwrap();
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}
