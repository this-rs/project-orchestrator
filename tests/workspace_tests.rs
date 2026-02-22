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
        setup_completed: true,
        neo4j_uri: std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into()),
        neo4j_user: std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into()),
        neo4j_password: std::env::var("NEO4J_PASSWORD")
            .unwrap_or_else(|_| "orchestrator123".into()),
        meilisearch_url: std::env::var("MEILISEARCH_URL")
            .unwrap_or_else(|_| "http://localhost:7700".into()),
        meilisearch_key: std::env::var("MEILISEARCH_KEY")
            .unwrap_or_else(|_| "orchestrator-meili-key-change-me".into()),
        nats_url: None,
        workspace_path: ".".into(),
        server_port: 8080,
        auth_config: None,
        serve_frontend: false,
        frontend_path: "./dist".to_string(),
        public_url: None,
        chat_permissions: None,
        chat_default_model: None,
        chat_max_sessions: None,
        chat_max_turns: None,
        chat_session_timeout_secs: None,
        chat_process_path: None,
        chat_claude_cli_path: None,
        chat_auto_update_cli: None,
        chat_auto_update_app: None,
        config_yaml_path: None,
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
        analytics_computed_at: None,
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
        analytics_computed_at: None,
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
        analytics_computed_at: None,
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
        analytics_computed_at: None,
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

// ============================================================================
// Workspace Milestone Task Association Tests
// ============================================================================

#[tokio::test]
async fn test_workspace_milestone_task_association() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Milestone Task Test".to_string(),
        slug: format!("ms-task-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Create milestone
    let milestone = WorkspaceMilestoneNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        title: "Task Association Test Milestone".to_string(),
        description: None,
        status: MilestoneStatus::Open,
        target_date: None,
        closed_at: None,
        created_at: chrono::Utc::now(),
        tags: vec![],
    };
    state
        .neo4j
        .create_workspace_milestone(&milestone)
        .await
        .unwrap();

    // Create a plan and task (using neo4j::models types from wildcard import)
    let plan = PlanNode {
        id: Uuid::new_v4(),
        title: "Test Plan".to_string(),
        description: "A test plan".to_string(),
        status: PlanStatus::Draft,
        created_at: chrono::Utc::now(),
        priority: 5,
        created_by: "test".to_string(),
        project_id: None,
    };
    state.neo4j.create_plan(&plan).await.unwrap();

    let task = TaskNode {
        id: Uuid::new_v4(),
        title: Some("Test Task".to_string()),
        description: "A test task".to_string(),
        status: TaskStatus::Pending,
        created_at: chrono::Utc::now(),
        updated_at: None,
        started_at: None,
        completed_at: None,
        priority: Some(5),
        assigned_to: None,
        tags: vec![],
        estimated_complexity: None,
        actual_complexity: None,
        acceptance_criteria: vec![],
        affected_files: vec![],
    };
    state.neo4j.create_task(plan.id, &task).await.unwrap();

    // Add task to milestone
    let result = state
        .neo4j
        .add_task_to_workspace_milestone(milestone.id, task.id)
        .await;
    assert!(result.is_ok(), "Should add task to milestone");

    // Get milestone tasks
    let tasks = state
        .neo4j
        .get_workspace_milestone_tasks(milestone.id)
        .await
        .unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].task.id, task.id);
    assert!(
        tasks[0].plan_id != uuid::Uuid::nil(),
        "plan_id should be set"
    );

    // Get milestone progress
    let (total, completed, in_progress, pending) = state
        .neo4j
        .get_workspace_milestone_progress(milestone.id)
        .await
        .unwrap();
    assert_eq!(total, 1);
    assert_eq!(pending, 1);
    assert_eq!(completed, 0);
    assert_eq!(in_progress, 0);

    // Cleanup
    state.neo4j.delete_plan(plan.id).await.unwrap();
    state
        .neo4j
        .delete_workspace_milestone(milestone.id)
        .await
        .unwrap();
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}

// ============================================================================
// Resource Full Flow Tests
// ============================================================================

#[tokio::test]
async fn test_resource_full_flow() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Resource Full Flow Test".to_string(),
        slug: format!("res-flow-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Create resource with full metadata
    let resource = ResourceNode {
        id: Uuid::new_v4(),
        workspace_id: Some(workspace.id),
        project_id: None,
        name: "Full Resource".to_string(),
        resource_type: ResourceType::ApiContract,
        file_path: "/specs/api.yaml".to_string(),
        url: Some("https://api.example.com/docs".to_string()),
        format: Some("openapi".to_string()),
        version: Some("1.0.0".to_string()),
        description: Some("Full description".to_string()),
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({"key": "value", "nested": {"a": 1}}),
    };
    state.neo4j.create_resource(&resource).await.unwrap();

    // Verify all fields
    let retrieved = state
        .neo4j
        .get_resource(resource.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(retrieved.name, "Full Resource");
    assert_eq!(retrieved.resource_type, ResourceType::ApiContract);
    assert_eq!(
        retrieved.url,
        Some("https://api.example.com/docs".to_string())
    );
    assert_eq!(retrieved.format, Some("openapi".to_string()));
    assert_eq!(retrieved.version, Some("1.0.0".to_string()));
    assert_eq!(retrieved.metadata["key"], "value");
    assert_eq!(retrieved.metadata["nested"]["a"], 1);

    // List and verify
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

// ============================================================================
// Component Full Flow Tests
// ============================================================================

#[tokio::test]
async fn test_component_full_flow() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Component Full Flow Test".to_string(),
        slug: format!("comp-flow-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Create component with full config
    let component = ComponentNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        name: "Full Component".to_string(),
        component_type: ComponentType::Service,
        description: Some("Full description".to_string()),
        runtime: Some("kubernetes".to_string()),
        config: serde_json::json!({"port": 8080, "replicas": 3, "env": {"DEBUG": "true"}}),
        created_at: chrono::Utc::now(),
        tags: vec!["production".to_string(), "critical".to_string()],
    };
    state.neo4j.create_component(&component).await.unwrap();

    // Verify all fields
    let retrieved = state
        .neo4j
        .get_component(component.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(retrieved.name, "Full Component");
    assert_eq!(retrieved.component_type, ComponentType::Service);
    assert_eq!(retrieved.runtime, Some("kubernetes".to_string()));
    assert_eq!(retrieved.config["port"], 8080);
    assert_eq!(retrieved.config["replicas"], 3);
    assert!(retrieved.tags.contains(&"production".to_string()));
    assert!(retrieved.tags.contains(&"critical".to_string()));

    // List and verify
    let components = state.neo4j.list_components(workspace.id).await.unwrap();
    assert_eq!(components.len(), 1);

    // Cleanup
    state.neo4j.delete_component(component.id).await.unwrap();
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}

// ============================================================================
// Workspace Metadata Update Tests
// ============================================================================

#[tokio::test]
async fn test_workspace_metadata_update() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace with metadata
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Metadata Test Workspace".to_string(),
        slug: format!("metadata-test-{}", Uuid::new_v4()),
        description: Some("Original".to_string()),
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({"key1": "value1"}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Update with new metadata
    let result = state
        .neo4j
        .update_workspace(
            workspace.id,
            None,
            None,
            Some(serde_json::json!({"key2": "value2", "nested": {"a": 1}})),
        )
        .await;
    assert!(result.is_ok());

    // Verify
    let updated = state
        .neo4j
        .get_workspace(workspace.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(updated.metadata["key2"], "value2");
    assert_eq!(updated.metadata["nested"]["a"], 1);

    // Cleanup
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}

// ============================================================================
// Multi-Project Workspace Tests
// ============================================================================

#[tokio::test]
async fn test_multiple_projects_in_workspace() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Multi-Project Test".to_string(),
        slug: format!("multi-proj-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Create multiple projects
    let mut project_ids = Vec::new();
    for i in 1..=3 {
        let project = ProjectNode {
            id: Uuid::new_v4(),
            name: format!("Project {}", i),
            slug: format!("multi-proj-{}-{}", i, Uuid::new_v4()),
            root_path: format!("/projects/{}", i),
            description: None,
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
        };
        state.neo4j.create_project(&project).await.unwrap();
        state
            .neo4j
            .add_project_to_workspace(workspace.id, project.id)
            .await
            .unwrap();
        project_ids.push(project.id);
    }

    // Verify all projects are associated
    let projects = state
        .neo4j
        .list_workspace_projects(workspace.id)
        .await
        .unwrap();
    assert_eq!(projects.len(), 3);

    // Remove one project
    state
        .neo4j
        .remove_project_from_workspace(workspace.id, project_ids[0])
        .await
        .unwrap();

    let remaining = state
        .neo4j
        .list_workspace_projects(workspace.id)
        .await
        .unwrap();
    assert_eq!(remaining.len(), 2);

    // Cleanup
    for id in project_ids {
        state.neo4j.delete_project(id).await.unwrap();
    }
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}

// ============================================================================
// Resource Type Tests
// ============================================================================

#[tokio::test]
async fn test_resource_all_types() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Resource Types Test".to_string(),
        slug: format!("res-types-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Test each resource type
    let resource_types = vec![
        (ResourceType::ApiContract, "api_contract.yaml"),
        (ResourceType::Protobuf, "events.proto"),
        (ResourceType::GraphqlSchema, "schema.graphql"),
        (ResourceType::JsonSchema, "schema.json"),
        (ResourceType::DatabaseSchema, "migrations.sql"),
        (ResourceType::SharedTypes, "types.d.ts"),
        (ResourceType::Config, "config.yaml"),
        (ResourceType::Documentation, "docs.md"),
        (ResourceType::Other, "other.txt"),
    ];

    let mut resource_ids = Vec::new();
    for (rt, file_name) in resource_types {
        let resource = ResourceNode {
            id: Uuid::new_v4(),
            workspace_id: Some(workspace.id),
            project_id: None,
            name: format!("{:?} Resource", rt),
            resource_type: rt,
            file_path: format!("/specs/{}", file_name),
            url: None,
            format: None,
            version: None,
            description: None,
            created_at: chrono::Utc::now(),
            updated_at: None,
            metadata: serde_json::json!({}),
        };
        state.neo4j.create_resource(&resource).await.unwrap();
        resource_ids.push(resource.id);
    }

    // Verify all created
    let resources = state
        .neo4j
        .list_workspace_resources(workspace.id)
        .await
        .unwrap();
    assert_eq!(resources.len(), 9);

    // Cleanup
    for id in resource_ids {
        state.neo4j.delete_resource(id).await.unwrap();
    }
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}

// ============================================================================
// Component Type Tests
// ============================================================================

#[tokio::test]
async fn test_component_all_types() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Component Types Test".to_string(),
        slug: format!("comp-types-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Test each component type
    let component_types = vec![
        ComponentType::Service,
        ComponentType::Frontend,
        ComponentType::Worker,
        ComponentType::Database,
        ComponentType::MessageQueue,
        ComponentType::Cache,
        ComponentType::Gateway,
        ComponentType::External,
        ComponentType::Other,
    ];

    let mut component_ids = Vec::new();
    for ct in component_types {
        let component = ComponentNode {
            id: Uuid::new_v4(),
            workspace_id: workspace.id,
            name: format!("{:?} Component", ct),
            component_type: ct,
            description: None,
            runtime: None,
            config: serde_json::json!({}),
            created_at: chrono::Utc::now(),
            tags: vec![],
        };
        state.neo4j.create_component(&component).await.unwrap();
        component_ids.push(component.id);
    }

    // Verify all created
    let components = state.neo4j.list_components(workspace.id).await.unwrap();
    assert_eq!(components.len(), 9);

    // Cleanup
    for id in component_ids {
        state.neo4j.delete_component(id).await.unwrap();
    }
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}

// ============================================================================
// Complex Topology Tests
// ============================================================================

#[tokio::test]
async fn test_complex_component_topology() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create workspace
    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Complex Topology Test".to_string(),
        slug: format!("complex-topo-test-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    state.neo4j.create_workspace(&workspace).await.unwrap();

    // Create components: Gateway -> API -> DB, Cache
    let gateway = ComponentNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        name: "API Gateway".to_string(),
        component_type: ComponentType::Gateway,
        description: None,
        runtime: Some("nginx".to_string()),
        config: serde_json::json!({}),
        created_at: chrono::Utc::now(),
        tags: vec![],
    };
    state.neo4j.create_component(&gateway).await.unwrap();

    let api = ComponentNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        name: "User API".to_string(),
        component_type: ComponentType::Service,
        description: None,
        runtime: Some("docker".to_string()),
        config: serde_json::json!({}),
        created_at: chrono::Utc::now(),
        tags: vec![],
    };
    state.neo4j.create_component(&api).await.unwrap();

    let db = ComponentNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        name: "PostgreSQL".to_string(),
        component_type: ComponentType::Database,
        description: None,
        runtime: None,
        config: serde_json::json!({}),
        created_at: chrono::Utc::now(),
        tags: vec![],
    };
    state.neo4j.create_component(&db).await.unwrap();

    let cache = ComponentNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        name: "Redis".to_string(),
        component_type: ComponentType::Cache,
        description: None,
        runtime: None,
        config: serde_json::json!({}),
        created_at: chrono::Utc::now(),
        tags: vec![],
    };
    state.neo4j.create_component(&cache).await.unwrap();

    // Create dependencies
    state
        .neo4j
        .add_component_dependency(gateway.id, api.id, Some("http".to_string()), true)
        .await
        .unwrap();
    state
        .neo4j
        .add_component_dependency(api.id, db.id, Some("postgres".to_string()), true)
        .await
        .unwrap();
    state
        .neo4j
        .add_component_dependency(api.id, cache.id, Some("redis".to_string()), false)
        .await
        .unwrap();

    // Get full topology
    let topology = state
        .neo4j
        .get_workspace_topology(workspace.id)
        .await
        .unwrap();
    assert_eq!(topology.len(), 4);

    // Check gateway has 1 dependency
    let gateway_entry = topology
        .iter()
        .find(|(c, _, _)| c.id == gateway.id)
        .unwrap();
    assert_eq!(gateway_entry.2.len(), 1);

    // Check API has 2 dependencies
    let api_entry = topology.iter().find(|(c, _, _)| c.id == api.id).unwrap();
    assert_eq!(api_entry.2.len(), 2);

    // Check DB has no dependencies
    let db_entry = topology.iter().find(|(c, _, _)| c.id == db.id).unwrap();
    assert_eq!(db_entry.2.len(), 0);

    // Cleanup
    state.neo4j.delete_component(gateway.id).await.unwrap();
    state.neo4j.delete_component(api.id).await.unwrap();
    state.neo4j.delete_component(db.id).await.unwrap();
    state.neo4j.delete_component(cache.id).await.unwrap();
    state.neo4j.delete_workspace(workspace.id).await.unwrap();
}
