//! Integration tests for project-orchestrator
//!
//! These tests require Neo4j and Meilisearch to be running.
//! Run with: cargo test --test integration_tests

use project_orchestrator::neo4j::models::*;
use project_orchestrator::{AppState, Config};
use std::time::Duration;
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

    // Check Neo4j (try to connect)
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

#[tokio::test]
async fn test_app_state_initialization() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await;

    assert!(state.is_ok(), "AppState should initialize successfully");
}

#[tokio::test]
async fn test_neo4j_file_operations() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create a test file node
    let file = FileNode {
        path: format!("/test/file_{}.rs", Uuid::new_v4()),
        language: "rust".to_string(),
        hash: "abc123".to_string(),
        last_parsed: chrono::Utc::now(),
        project_id: None,
    };

    // Upsert file
    let result = state.neo4j.upsert_file(&file).await;
    assert!(result.is_ok(), "Should upsert file: {:?}", result.err());

    // Get file
    let retrieved = state.neo4j.get_file(&file.path).await.unwrap();
    assert!(retrieved.is_some(), "Should retrieve file");

    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.path, file.path);
    assert_eq!(retrieved.language, file.language);
    assert_eq!(retrieved.hash, file.hash);

    // Cleanup: delete the test file
    state.neo4j.delete_file(&file.path).await.unwrap();
}

#[tokio::test]
async fn test_neo4j_plan_operations() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create a test plan
    let plan = PlanNode::new(
        format!("Test Plan {}", Uuid::new_v4()),
        "Test description".to_string(),
        "test-agent".to_string(),
        5,
    );

    // Create plan
    let result = state.neo4j.create_plan(&plan).await;
    assert!(result.is_ok(), "Should create plan: {:?}", result.err());

    // Get plan
    let retrieved = state.neo4j.get_plan(plan.id).await.unwrap();
    assert!(retrieved.is_some(), "Should retrieve plan");

    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, plan.id);
    assert_eq!(retrieved.title, plan.title);

    // Update status
    let result = state
        .neo4j
        .update_plan_status(plan.id, PlanStatus::Approved)
        .await;
    assert!(result.is_ok(), "Should update plan status");

    // Verify status update
    let updated = state.neo4j.get_plan(plan.id).await.unwrap().unwrap();
    assert_eq!(updated.status, PlanStatus::Approved);

    // Cleanup: delete the test plan
    state.neo4j.delete_plan(plan.id).await.unwrap();
}

#[tokio::test]
async fn test_neo4j_task_operations() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create a plan first
    let plan = PlanNode::new(
        format!("Task Test Plan {}", Uuid::new_v4()),
        "Plan for task testing".to_string(),
        "test-agent".to_string(),
        1,
    );
    state.neo4j.create_plan(&plan).await.unwrap();

    // Create a task
    let task = TaskNode::new("Test task description".to_string());
    let result = state.neo4j.create_task(plan.id, &task).await;
    assert!(result.is_ok(), "Should create task: {:?}", result.err());

    // Get tasks for plan
    let tasks = state.neo4j.get_plan_tasks(plan.id).await.unwrap();
    assert_eq!(tasks.len(), 1, "Should have one task");
    assert_eq!(tasks[0].id, task.id);

    // Update task status
    let result = state
        .neo4j
        .update_task_status(task.id, TaskStatus::InProgress)
        .await;
    assert!(result.is_ok(), "Should update task status");

    // Get next available task (should be none since our task is in progress)
    let next = state.neo4j.get_next_available_task(plan.id).await.unwrap();
    assert!(next.is_none(), "No pending tasks should be available");

    // Cleanup: delete the test plan (which will also delete tasks)
    state.neo4j.delete_plan(plan.id).await.unwrap();
}

#[tokio::test]
async fn test_neo4j_task_dependencies() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create a plan
    let plan = PlanNode::new(
        format!("Dependency Test Plan {}", Uuid::new_v4()),
        "Plan for dependency testing".to_string(),
        "test-agent".to_string(),
        1,
    );
    state.neo4j.create_plan(&plan).await.unwrap();

    // Create task 1 (no dependencies)
    let task1 = TaskNode::new("Task 1 - Foundation".to_string());
    state.neo4j.create_task(plan.id, &task1).await.unwrap();

    // Create task 2 (depends on task 1)
    let task2 = TaskNode::new("Task 2 - Depends on Task 1".to_string());
    state.neo4j.create_task(plan.id, &task2).await.unwrap();
    state
        .neo4j
        .add_task_dependency(task2.id, task1.id)
        .await
        .unwrap();

    // Get next available task - should be task1 (task2 is blocked)
    let next = state.neo4j.get_next_available_task(plan.id).await.unwrap();
    assert!(next.is_some(), "Should have an available task");
    assert_eq!(next.unwrap().id, task1.id, "Task 1 should be available");

    // Complete task 1
    state
        .neo4j
        .update_task_status(task1.id, TaskStatus::Completed)
        .await
        .unwrap();

    // Now task 2 should be available
    let next = state.neo4j.get_next_available_task(plan.id).await.unwrap();
    assert!(next.is_some(), "Task 2 should now be available");
    assert_eq!(next.unwrap().id, task2.id);

    // Cleanup: delete the test plan (which will also delete tasks)
    state.neo4j.delete_plan(plan.id).await.unwrap();
}

#[tokio::test]
async fn test_meilisearch_code_indexing() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    use project_orchestrator::meilisearch::indexes::CodeDocument;

    // Create a test document
    let doc = CodeDocument {
        id: format!("test-{}", Uuid::new_v4()),
        path: "/test/example.rs".to_string(),
        language: "rust".to_string(),
        symbols: vec!["hello_world".to_string()],
        docstrings: "Says hello to the world".to_string(),
        signatures: vec!["fn hello_world()".to_string()],
        imports: vec![],
        project_id: "test-project-id".to_string(),
        project_slug: "test-project".to_string(),
    };

    // Index the document
    let result = state.meili.index_code(&doc).await;
    assert!(result.is_ok(), "Should index code: {:?}", result.err());

    // Wait a bit for indexing
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Search for it
    let results = state.meili.search_code("hello_world", 10, None).await;
    assert!(results.is_ok(), "Should search code: {:?}", results.err());

    // Note: Search results may not include our doc immediately due to async indexing
    // In production tests, we'd wait for the task to complete

    // Cleanup: delete the test document
    state.meili.delete_code(&doc.id).await.unwrap();
}

#[tokio::test]
async fn test_meilisearch_decision_indexing() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    use project_orchestrator::meilisearch::indexes::DecisionDocument;

    // Create a test decision document
    let doc = DecisionDocument {
        id: format!("decision-{}", Uuid::new_v4()),
        description: "Use async/await for all I/O operations".to_string(),
        rationale: "Better performance and resource utilization".to_string(),
        task_id: Uuid::new_v4().to_string(),
        agent: "test-agent".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        tags: vec!["architecture".to_string(), "async".to_string()],
        project_id: None,
        project_slug: None,
    };

    // Index the document
    let result = state.meili.index_decision(&doc).await;
    assert!(result.is_ok(), "Should index decision: {:?}", result.err());

    // Wait a bit for indexing
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Search for it
    let results = state.meili.search_decisions("async await", 10).await;
    assert!(
        results.is_ok(),
        "Should search decisions: {:?}",
        results.err()
    );

    // Cleanup: delete the test document
    state.meili.delete_decision(&doc.id).await.unwrap();
}

#[tokio::test]
async fn test_neo4j_stale_file_cleanup() {
    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    // Create a test project
    let project_id = Uuid::new_v4();
    let project = project_orchestrator::neo4j::models::ProjectNode {
        id: project_id,
        name: format!("Cleanup Test Project {}", project_id),
        slug: format!("cleanup-test-{}", project_id),
        root_path: "/tmp/test-cleanup".to_string(),
        description: Some("Project for testing stale file cleanup".to_string()),
        created_at: chrono::Utc::now(),
        last_synced: None,
    };
    state.neo4j.create_project(&project).await.unwrap();

    // Create some file nodes belonging to this project
    let file1_path = format!("/tmp/test-cleanup/file1_{}.rs", Uuid::new_v4());
    let file2_path = format!("/tmp/test-cleanup/file2_{}.rs", Uuid::new_v4());
    let file3_path = format!("/tmp/test-cleanup/file3_{}.rs", Uuid::new_v4());

    for path in [&file1_path, &file2_path, &file3_path] {
        let file = FileNode {
            path: path.clone(),
            language: "rust".to_string(),
            hash: "test-hash".to_string(),
            last_parsed: chrono::Utc::now(),
            project_id: Some(project_id),
        };
        state.neo4j.upsert_file(&file).await.unwrap();
        state
            .neo4j
            .link_file_to_project(path, project_id)
            .await
            .unwrap();
    }

    // Verify all 3 files exist
    let paths_before = state
        .neo4j
        .get_project_file_paths(project_id)
        .await
        .unwrap();
    assert_eq!(paths_before.len(), 3, "Should have 3 files before cleanup");

    // Now simulate a sync where only file1 and file2 exist (file3 was deleted)
    let valid_paths = vec![file1_path.clone(), file2_path.clone()];
    let (files_deleted, _symbols_deleted) = state
        .neo4j
        .delete_stale_files(project_id, &valid_paths)
        .await
        .unwrap();

    assert_eq!(files_deleted, 1, "Should delete 1 stale file");

    // Verify only 2 files remain
    let paths_after = state
        .neo4j
        .get_project_file_paths(project_id)
        .await
        .unwrap();
    assert_eq!(paths_after.len(), 2, "Should have 2 files after cleanup");
    assert!(
        paths_after.contains(&file1_path),
        "file1 should still exist"
    );
    assert!(
        paths_after.contains(&file2_path),
        "file2 should still exist"
    );
    assert!(
        !paths_after.contains(&file3_path),
        "file3 should be deleted"
    );

    // Cleanup: delete the test project
    state.neo4j.delete_project(project_id).await.unwrap();
}
