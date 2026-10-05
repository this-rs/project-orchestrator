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
        remote_mcp: project_orchestrator::RemoteMcpConfig::default(),
        chat_permissions: None,
        chat_default_model: None,
        chat_max_sessions: None,
        chat_max_turns: None,
        chat_session_timeout_secs: None,
        chat_process_path: None,
        chat_claude_cli_path: None,
        chat_auto_update_cli: None,
        chat_auto_update_app: None,
        embedding_provider: None,
        embedding_fastembed_model: None,
        embedding_fastembed_cache_dir: None,
        embedding_url: None,
        embedding_model: None,
        embedding_api_key: None,
        embedding_dimensions: None,
        anthropic_api_key: None,
        registry_remote_url: None,
        documents_storage_dir: None,
        neural_routing: Default::default(),
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
        analytics_computed_at: None,
        last_co_change_computed_at: None,
        default_note_energy: None,
        scaffolding_override: None,
        sharing_policy: None,
        watch_enabled: true,
        profile: Default::default(),
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
    let (files_deleted, _symbols_deleted, deleted_paths) = state
        .neo4j
        .delete_stale_files(project_id, &valid_paths)
        .await
        .unwrap();

    assert_eq!(files_deleted, 1, "Should delete 1 stale file");
    assert_eq!(deleted_paths.len(), 1, "Should return 1 deleted path");

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
    state
        .neo4j
        .delete_project(project_id, "test-project")
        .await
        .unwrap();
}

// ============================================================================
// GraphStore task operations (coverage batch — exercises src/neo4j/task.rs)
// ============================================================================

fn make_task(title: &str, status: TaskStatus) -> TaskNode {
    TaskNode {
        id: Uuid::new_v4(),
        title: Some(title.to_string()),
        description: format!("description for {title}"),
        status,
        assigned_to: None,
        priority: Some(5),
        tags: vec!["test".to_string()],
        acceptance_criteria: vec![],
        affected_files: vec![],
        estimated_complexity: Some(3),
        actual_complexity: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        started_at: None,
        completed_at: None,
        frustration_score: 0.0,
        execution_context: None,
        persona: None,
        prompt_cache: None,
    }
}

fn make_step(order: u32, desc: &str) -> StepNode {
    StepNode {
        id: Uuid::new_v4(),
        order,
        description: desc.to_string(),
        status: StepStatus::Pending,
        verification: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        completed_at: None,
        execution_context: None,
        persona: None,
    }
}

async fn setup_plan(state: &AppState) -> Uuid {
    let plan = PlanNode::new(
        format!("Plan {}", Uuid::new_v4()),
        "task-test plan".to_string(),
        "test-agent".to_string(),
        5,
    );
    state.neo4j.create_plan(&plan).await.unwrap();
    plan.id
}

#[tokio::test]
async fn test_neo4j_task_create_get_list() {
    if !backends_available().await {
        return;
    }
    let state = AppState::new(test_config()).await.unwrap();
    let plan_id = setup_plan(&state).await;
    let task = make_task("alpha", TaskStatus::Pending);
    state.neo4j.create_task(plan_id, &task).await.unwrap();

    let got = state.neo4j.get_task(task.id).await.unwrap();
    assert!(got.is_some());
    assert_eq!(got.unwrap().title.as_deref(), Some("alpha"));

    let tasks = state.neo4j.get_plan_tasks(plan_id).await.unwrap();
    assert!(tasks.iter().any(|t| t.id == task.id));

    state.neo4j.delete_task(task.id).await.ok();
    state.neo4j.delete_plan(plan_id).await.ok();
}

#[tokio::test]
async fn test_neo4j_task_get_not_found() {
    if !backends_available().await {
        return;
    }
    let state = AppState::new(test_config()).await.unwrap();
    let got = state.neo4j.get_task(Uuid::new_v4()).await.unwrap();
    assert!(got.is_none());
}

#[tokio::test]
async fn test_neo4j_task_update_status_and_assign() {
    if !backends_available().await {
        return;
    }
    let state = AppState::new(test_config()).await.unwrap();
    let plan_id = setup_plan(&state).await;
    let task = make_task("beta", TaskStatus::Pending);
    state.neo4j.create_task(plan_id, &task).await.unwrap();

    state
        .neo4j
        .update_task_status(task.id, TaskStatus::Completed)
        .await
        .unwrap();
    // assign_task exercises the assignment write path (relationship-based, not
    // necessarily reflected in the TaskNode.assigned_to scalar) — assert it
    // succeeds rather than asserting the scalar round-trips.
    state.neo4j.assign_task(task.id, "agent-7").await.unwrap();

    let got = state.neo4j.get_task(task.id).await.unwrap().unwrap();
    assert_eq!(got.status, TaskStatus::Completed);

    state.neo4j.delete_task(task.id).await.ok();
    state.neo4j.delete_plan(plan_id).await.ok();
}

#[tokio::test]
async fn test_neo4j_task_dependency_blockers() {
    if !backends_available().await {
        return;
    }
    let state = AppState::new(test_config()).await.unwrap();
    let plan_id = setup_plan(&state).await;
    let t1 = make_task("dep-base", TaskStatus::Pending);
    let t2 = make_task("dep-dependent", TaskStatus::Pending);
    state.neo4j.create_task(plan_id, &t1).await.unwrap();
    state.neo4j.create_task(plan_id, &t2).await.unwrap();

    state.neo4j.add_task_dependency(t2.id, t1.id).await.unwrap();
    let deps = state.neo4j.get_task_dependencies(t2.id).await.unwrap();
    assert!(deps.iter().any(|t| t.id == t1.id));
    let blocked_by = state.neo4j.get_tasks_blocked_by(t1.id).await.unwrap();
    assert!(blocked_by.iter().any(|t| t.id == t2.id));

    state
        .neo4j
        .remove_task_dependency(t2.id, t1.id)
        .await
        .unwrap();
    let deps_after = state.neo4j.get_task_dependencies(t2.id).await.unwrap();
    assert!(deps_after.iter().all(|t| t.id != t1.id));

    state.neo4j.delete_task(t1.id).await.ok();
    state.neo4j.delete_task(t2.id).await.ok();
    state.neo4j.delete_plan(plan_id).await.ok();
}

#[tokio::test]
async fn test_neo4j_task_update_fields() {
    if !backends_available().await {
        return;
    }
    let state = AppState::new(test_config()).await.unwrap();
    let plan_id = setup_plan(&state).await;
    let task = make_task("gamma", TaskStatus::Pending);
    state.neo4j.create_task(plan_id, &task).await.unwrap();

    let updates = project_orchestrator::plan::models::UpdateTaskRequest {
        title: Some("gamma-renamed".to_string()),
        priority: Some(9),
        ..Default::default()
    };
    state.neo4j.update_task(task.id, &updates).await.unwrap();

    let got = state.neo4j.get_task(task.id).await.unwrap().unwrap();
    assert_eq!(got.title.as_deref(), Some("gamma-renamed"));
    assert_eq!(got.priority, Some(9));

    state.neo4j.delete_task(task.id).await.ok();
    state.neo4j.delete_plan(plan_id).await.ok();
}

#[tokio::test]
async fn test_neo4j_task_delete() {
    if !backends_available().await {
        return;
    }
    let state = AppState::new(test_config()).await.unwrap();
    let plan_id = setup_plan(&state).await;
    let task = make_task("to-delete", TaskStatus::Pending);
    state.neo4j.create_task(plan_id, &task).await.unwrap();
    state.neo4j.delete_task(task.id).await.unwrap();
    assert!(state.neo4j.get_task(task.id).await.unwrap().is_none());
    state.neo4j.delete_plan(plan_id).await.ok();
}

#[tokio::test]
async fn test_neo4j_task_steps_and_progress() {
    if !backends_available().await {
        return;
    }
    let state = AppState::new(test_config()).await.unwrap();
    let plan_id = setup_plan(&state).await;
    let task = make_task("with-steps", TaskStatus::Pending);
    state.neo4j.create_task(plan_id, &task).await.unwrap();

    state
        .neo4j
        .create_step(task.id, &make_step(1, "first"))
        .await
        .unwrap();
    state
        .neo4j
        .create_step(task.id, &make_step(2, "second"))
        .await
        .unwrap();

    let steps = state.neo4j.get_task_steps(task.id).await.unwrap();
    assert_eq!(steps.len(), 2);
    let (done, total) = state.neo4j.get_task_step_progress(task.id).await.unwrap();
    assert_eq!((done, total), (0, 2));

    let completed = state
        .neo4j
        .complete_pending_steps_for_task(task.id)
        .await
        .unwrap();
    assert_eq!(completed, 2);
    let (done2, total2) = state.neo4j.get_task_step_progress(task.id).await.unwrap();
    assert_eq!((done2, total2), (2, 2));

    state.neo4j.delete_task(task.id).await.ok();
    state.neo4j.delete_plan(plan_id).await.ok();
}

#[tokio::test]
async fn test_neo4j_task_plan_and_next_available() {
    if !backends_available().await {
        return;
    }
    let state = AppState::new(test_config()).await.unwrap();
    let plan_id = setup_plan(&state).await;
    let task = make_task("ready", TaskStatus::Pending);
    state.neo4j.create_task(plan_id, &task).await.unwrap();

    let resolved_plan = state.neo4j.get_plan_id_for_task(task.id).await.unwrap();
    assert_eq!(resolved_plan, Some(plan_id));

    let next = state.neo4j.get_next_available_task(plan_id).await.unwrap();
    assert!(next.is_some());

    state.neo4j.delete_task(task.id).await.ok();
    state.neo4j.delete_plan(plan_id).await.ok();
}

/// Real-Neo4j check of the entity neighbourhood fetch: a note linked to a
/// file (LINKED_TO), a second note tied to the first by a SYNAPSE, and a
/// project container that must not be expanded past the first hop.
#[tokio::test]
async fn test_neo4j_entity_neighborhood() {
    use project_orchestrator::graph::neighborhood::{
        select_neighborhood, Layer, NeighborhoodParams,
    };
    use project_orchestrator::neo4j::client::Neo4jClient;

    let config = test_config();
    let client = match Neo4jClient::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping test: Neo4j not available: {e}");
            return;
        }
    };
    let raw_graph = neo4rs::Graph::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    .unwrap();

    let tag = Uuid::new_v4().to_string();
    let (n1, n2, n3) = (
        format!("{tag}-n1"),
        format!("{tag}-n2"),
        format!("{tag}-n3"),
    );
    let file = format!("/test/neighborhood_{tag}.rs");
    let project = format!("{tag}-p");
    let other_file = format!("/test/neighborhood_other_{tag}.rs");

    raw_graph
        .run(
            neo4rs::query(
                "CREATE (a:Note {id: $n1, content: 'first note'}), \
                        (b:Note {id: $n2, content: 'second note'}), \
                        (c:Note {id: $n3, content: 'third note'}), \
                        (f:File {path: $file}), \
                        (o:File {path: $other}), \
                        (p:Project {id: $p, name: 'neighborhood-test'}), \
                        (a)-[:LINKED_TO]->(f), \
                        (a)-[:SYNAPSE {weight: 0.8}]->(b), \
                        (p)-[:CONTAINS]->(f), \
                        (p)-[:CONTAINS]->(o), \
                        (c)-[:LINKED_TO]->(o)",
            )
            .param("n1", n1.clone())
            .param("n2", n2.clone())
            .param("n3", n3.clone())
            .param("file", file.clone())
            .param("other", other_file.clone())
            .param("p", project.clone()),
        )
        .await
        .unwrap();

    let all = NeighborhoodParams::clamped(Some(2), None, None, Layer::ALL.to_vec());

    // Unknown centre
    let missing = client
        .get_entity_neighborhood("note", &format!("{tag}-missing"), &all)
        .await
        .unwrap();
    assert!(missing.is_none());

    // Depth 1 from n1: the file and the second note, not the project
    let d1 = NeighborhoodParams::clamped(Some(1), None, None, Layer::ALL.to_vec());
    let raw = client
        .get_entity_neighborhood("note", &n1, &d1)
        .await
        .unwrap()
        .expect("centre exists");
    let res = select_neighborhood(&raw, &d1).unwrap();
    let ids: Vec<&str> = res.nodes.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&n1.as_str()), "centre included: {ids:?}");
    assert!(ids.contains(&n2.as_str()), "synapse neighbour: {ids:?}");
    assert!(ids.contains(&file.as_str()), "linked file: {ids:?}");
    assert!(!ids.contains(&project.as_str()), "project is 2 hops away");

    // Depth 2 reaches the project through the file, but the project is a
    // container: it must not open onto its other files or their notes.
    let raw = client
        .get_entity_neighborhood("note", &n1, &all)
        .await
        .unwrap()
        .expect("centre exists");
    let res = select_neighborhood(&raw, &all).unwrap();
    let ids: Vec<&str> = res.nodes.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&project.as_str()), "project at hop 2: {ids:?}");
    assert!(
        !ids.contains(&other_file.as_str()),
        "container not expanded"
    );
    assert!(!ids.contains(&n3.as_str()), "container not expanded");

    // Layer filter: neural only keeps the synapse
    let neural = NeighborhoodParams::clamped(Some(2), None, None, vec![Layer::Neural]);
    let raw = client
        .get_entity_neighborhood("note", &n1, &neural)
        .await
        .unwrap()
        .expect("centre exists");
    let res = select_neighborhood(&raw, &neural).unwrap();
    let ids: Vec<&str> = res.nodes.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&n2.as_str()));
    assert!(!ids.contains(&file.as_str()));

    // Centre may be a container: the project expands to its files
    let raw = client
        .get_entity_neighborhood("project", &project, &d1)
        .await
        .unwrap()
        .expect("project exists");
    let res = select_neighborhood(&raw, &d1).unwrap();
    let ids: Vec<&str> = res.nodes.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&file.as_str()) && ids.contains(&other_file.as_str()));

    // Cleanup
    raw_graph
        .run(
            neo4rs::query(
                "MATCH (x) WHERE x.id IN [$n1, $n2, $n3, $p] OR x.path IN [$file, $other] \
                 DETACH DELETE x",
            )
            .param("n1", n1)
            .param("n2", n2)
            .param("n3", n3)
            .param("p", project)
            .param("file", file)
            .param("other", other_file),
        )
        .await
        .unwrap();
}

// ============================================================================
// Cypher / MeiliSearch injection hardening — live-backend coverage
// ============================================================================
//
// The hardening of this PR replaced string-interpolated Cypher and MeiliSearch
// filters with bound parameters. A bound parameter can be *inert* in two ways
// that a pure unit test on the builder cannot see:
//
//   1. the placeholder is referenced by the query text but never bound (or
//      bound under another name) — Neo4j then raises ParameterMissing, or
//      silently matches nothing;
//   2. the encoding written by the writer no longer matches the encoding the
//      reader filters on — the filter becomes a guaranteed empty result, which
//      looks like "no rows" rather than like a bug.
//
// Both only show up against a real server, so these tests hit the Neo4j and
// MeiliSearch instances the coverage job provisions. Each one asserts the
// *positive* path (a legitimate filter still returns its row) and the
// *negative* path (a payload travels as data and matches nothing) — a guard
// that rejects everything would pass the negative half alone.

/// A Cypher/filter breakout attempt used as a plain value everywhere below.
const EVIL: &str = "a' OR 1=1 //";

#[tokio::test]
async fn test_trigger_type_filter_is_bound_not_spliced() {
    use project_orchestrator::neo4j::client::Neo4jClient;
    use project_orchestrator::runner::{Trigger, TriggerType};

    let config = test_config();
    let client = match Neo4jClient::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping test: Neo4j not available: {e}");
            return;
        }
    };
    let raw = neo4rs::Graph::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    .unwrap();

    let plan_id = Uuid::new_v4();
    raw.run(
        neo4rs::query("CREATE (p:Plan {id: $id, name: 'trigger-filter-test'})")
            .param("id", plan_id.to_string()),
    )
    .await
    .unwrap();

    let webhook = Trigger {
        id: Uuid::new_v4(),
        plan_id,
        trigger_type: TriggerType::Webhook,
        config: serde_json::json!({"secret": "s"}),
        enabled: true,
        cooldown_secs: 0,
        last_fired: None,
        fire_count: 0,
        created_at: chrono::Utc::now(),
    };
    let schedule = Trigger {
        id: Uuid::new_v4(),
        trigger_type: TriggerType::Schedule,
        config: serde_json::json!({"cron": "0 2 * * *"}),
        ..webhook.clone()
    };
    client.create_trigger_impl(&webhook).await.unwrap();
    client.create_trigger_impl(&schedule).await.unwrap();

    // Positive: the filter still selects, so the $trigger_type parameter is
    // really bound (an unbound placeholder would error or match nothing).
    let only_webhook = client
        .list_all_triggers_impl(Some("webhook"))
        .await
        .unwrap();
    assert!(
        only_webhook.iter().any(|t| t.id == webhook.id),
        "webhook trigger must be returned by its own filter"
    );
    assert!(
        !only_webhook.iter().any(|t| t.id == schedule.id),
        "schedule trigger must not leak into the webhook filter"
    );

    // No filter: both are listed.
    let all = client.list_all_triggers_impl(None).await.unwrap();
    assert!(all.iter().any(|t| t.id == webhook.id));
    assert!(all.iter().any(|t| t.id == schedule.id));

    // Negative: a breakout payload is compared as a literal value. Spliced,
    // `WHERE t.trigger_type = 'a' OR 1=1 //` would have returned every row.
    let injected = client.list_all_triggers_impl(Some(EVIL)).await.unwrap();
    assert!(
        injected.is_empty(),
        "payload must match nothing, got {} rows",
        injected.len()
    );
    // And the tautology must not have reached the server as Cypher.
    let injected2 = client
        .list_all_triggers_impl(Some("webhook' OR 1=1 //"))
        .await
        .unwrap();
    assert!(
        injected2.is_empty(),
        "`webhook' OR 1=1 //` must match nothing, got {}",
        injected2.len()
    );

    raw.run(
        neo4rs::query(
            "MATCH (p:Plan {id: $id}) OPTIONAL MATCH (t:Trigger)-[:TRIGGERS]->(p) \
             DETACH DELETE p, t",
        )
        .param("id", plan_id.to_string()),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn test_trigger_firing_binds_plan_run_id() {
    use project_orchestrator::neo4j::client::Neo4jClient;
    use project_orchestrator::runner::{Trigger, TriggerFiring, TriggerType};

    let config = test_config();
    let client = match Neo4jClient::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping test: Neo4j not available: {e}");
            return;
        }
    };
    let raw = neo4rs::Graph::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    .unwrap();

    let plan_id = Uuid::new_v4();
    let run_id = Uuid::new_v4();
    raw.run(
        neo4rs::query(
            "CREATE (p:Plan {id: $pid, name: 'firing-test'}), \
                    (r:PlanRun {run_id: $rid, status: 'running'})",
        )
        .param("pid", plan_id.to_string())
        .param("rid", run_id.to_string()),
    )
    .await
    .unwrap();

    let trigger = Trigger {
        id: Uuid::new_v4(),
        plan_id,
        trigger_type: TriggerType::Event,
        config: serde_json::Value::Null,
        enabled: true,
        cooldown_secs: 0,
        last_fired: None,
        fire_count: 0,
        created_at: chrono::Utc::now(),
    };
    client.create_trigger_impl(&trigger).await.unwrap();

    // With a run: the STARTED edge is created, so `MATCH (r:PlanRun {run_id:
    // $plan_run_id})` really resolved its bound parameter.
    let linked = TriggerFiring {
        id: Uuid::new_v4(),
        trigger_id: trigger.id,
        plan_run_id: Some(run_id),
        fired_at: chrono::Utc::now(),
        source_payload: Some(serde_json::json!({"body": EVIL})),
    };
    client.record_trigger_firing_impl(&linked).await.unwrap();

    let started: i64 = raw
        .execute(
            neo4rs::query(
                "MATCH (f:TriggerFiring {id: $fid})-[:STARTED]->(r:PlanRun {run_id: $rid}) \
                 RETURN count(*) AS c",
            )
            .param("fid", linked.id.to_string())
            .param("rid", run_id.to_string()),
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get("c")
        .unwrap();
    assert_eq!(started, 1, "the firing must be linked to its PlanRun");

    // Without a run: the firing is still recorded, with no STARTED edge.
    let unlinked = TriggerFiring {
        id: Uuid::new_v4(),
        trigger_id: trigger.id,
        plan_run_id: None,
        fired_at: chrono::Utc::now(),
        source_payload: None,
    };
    client.record_trigger_firing_impl(&unlinked).await.unwrap();
    let (exists, edges): (i64, i64) = {
        let mut r = raw
            .execute(
                neo4rs::query(
                    "MATCH (f:TriggerFiring {id: $fid}) \
                     OPTIONAL MATCH (f)-[s:STARTED]->() \
                     RETURN count(DISTINCT f) AS f, count(s) AS s",
                )
                .param("fid", unlinked.id.to_string()),
            )
            .await
            .unwrap();
        let row = r.next().await.unwrap().unwrap();
        (row.get("f").unwrap(), row.get("s").unwrap())
    };
    assert_eq!(exists, 1, "the firing must be recorded without a run");
    assert_eq!(edges, 0, "no STARTED edge without a plan_run_id");

    raw.run(
        neo4rs::query(
            "MATCH (p:Plan {id: $pid}) OPTIONAL MATCH (t:Trigger)-[:TRIGGERS]->(p) \
             OPTIONAL MATCH (f:TriggerFiring)-[:FIRED_BY]->(t) \
             DETACH DELETE p, t, f",
        )
        .param("pid", plan_id.to_string()),
    )
    .await
    .unwrap();
    raw.run(
        neo4rs::query("MATCH (r:PlanRun {run_id: $rid}) DETACH DELETE r")
            .param("rid", run_id.to_string()),
    )
    .await
    .unwrap();
}

/// Create a bare Project node and return its id.
async fn make_project(client: &project_orchestrator::neo4j::client::Neo4jClient) -> Uuid {
    let id = Uuid::new_v4();
    let project = ProjectNode {
        id,
        name: format!("Injection Test {id}"),
        slug: format!("injection-test-{id}"),
        root_path: String::new(),
        description: None,
        created_at: chrono::Utc::now(),
        last_synced: None,
        analytics_computed_at: None,
        last_co_change_computed_at: None,
        default_note_energy: None,
        scaffolding_override: None,
        sharing_policy: None,
        watch_enabled: false,
        profile: Default::default(),
    };
    client.create_project(&project).await.unwrap();
    id
}

/// Delete a test project and everything hanging off it, scoped by its id.
async fn drop_project(raw: &neo4rs::Graph, project_id: Uuid) {
    raw.run(
        neo4rs::query(
            "MATCH (p:Project {id: $id})              OPTIONAL MATCH (p)-[:HAS_MILESTONE|HAS_RELEASE]->(x)              DETACH DELETE p, x",
        )
        .param("id", project_id.to_string()),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn test_milestone_status_filter_is_bound_and_matches_stored_encoding() {
    use project_orchestrator::neo4j::client::Neo4jClient;

    let config = test_config();
    let client = match Neo4jClient::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping test: Neo4j not available: {e}");
            return;
        }
    };

    let raw = neo4rs::Graph::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    .unwrap();

    let project_id = make_project(&client).await;

    let in_progress = MilestoneNode {
        id: Uuid::new_v4(),
        title: "Hardening".to_string(),
        description: None,
        status: MilestoneStatus::InProgress,
        target_date: None,
        closed_at: None,
        created_at: chrono::Utc::now(),
        project_id,
    };
    let completed = MilestoneNode {
        id: Uuid::new_v4(),
        title: "Shipped".to_string(),
        status: MilestoneStatus::Completed,
        ..in_progress.clone()
    };
    client.create_milestone(&in_progress).await.unwrap();
    client.create_milestone(&completed).await.unwrap();

    // Positive, snake_case — this is the encoding `create_milestone` writes.
    // `add_status_filter` (PascalCase only) returned 0 rows here; the switch to
    // `add_status_filter_any_case` is what makes the filter work at all.
    let (rows, total) = client
        .list_milestones_filtered(
            project_id,
            Some(vec!["in_progress".into()]),
            10,
            0,
            None,
            "asc",
        )
        .await
        .unwrap();
    assert_eq!(
        total, 1,
        "snake_case status filter must match the stored row"
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, in_progress.id);

    // Positive, PascalCase — legacy callers must keep working.
    let (rows, total) = client
        .list_milestones_filtered(
            project_id,
            Some(vec!["InProgress".into()]),
            10,
            0,
            None,
            "asc",
        )
        .await
        .unwrap();
    assert_eq!(total, 1, "PascalCase status filter must match too");
    assert_eq!(rows[0].id, in_progress.id);

    // Both statuses.
    let (_, total) = client
        .list_milestones_filtered(
            project_id,
            Some(vec!["in_progress".into(), "completed".into()]),
            10,
            0,
            Some("title"),
            "desc",
        )
        .await
        .unwrap();
    assert_eq!(total, 2);

    // No filter.
    let (_, total) = client
        .list_milestones_filtered(project_id, None, 10, 0, Some("created_at"), "asc")
        .await
        .unwrap();
    assert_eq!(total, 2);

    // Negative: the payload is a value in an IN list, not Cypher. Spliced, the
    // `' OR 1=1 //` would have returned both rows.
    let (rows, total) = client
        .list_milestones_filtered(project_id, Some(vec![EVIL.into()]), 10, 0, None, "asc")
        .await
        .unwrap();
    assert_eq!(total, 0, "payload must match no milestone");
    assert!(rows.is_empty());

    drop_project(&raw, project_id).await;
}

#[tokio::test]
async fn test_release_status_round_trips_snake_case_and_filter_is_bound() {
    use project_orchestrator::neo4j::client::Neo4jClient;

    let config = test_config();
    let client = match Neo4jClient::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping test: Neo4j not available: {e}");
            return;
        }
    };
    let raw = neo4rs::Graph::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    .unwrap();

    let project_id = make_project(&client).await;

    let rel = ReleaseNode {
        id: Uuid::new_v4(),
        version: "1.0.0".to_string(),
        title: Some("First".to_string()),
        description: None,
        status: ReleaseStatus::InProgress,
        target_date: None,
        released_at: None,
        created_at: chrono::Utc::now(),
        project_id,
    };
    let planned = ReleaseNode {
        id: Uuid::new_v4(),
        version: "2.0.0".to_string(),
        status: ReleaseStatus::Planned,
        ..rel.clone()
    };
    client.create_release(&rel).await.unwrap();
    client.create_release(&planned).await.unwrap();

    // `create_release` must store the canonical snake_case encoding. With the
    // old `format!("{:?}")` this property held "InProgress", which no
    // snake_case reader could match.
    let stored: String = raw
        .execute(
            neo4rs::query("MATCH (r:Release {id: $id}) RETURN r.status AS s")
                .param("id", rel.id.to_string()),
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get("s")
        .unwrap();
    assert_eq!(stored, "in_progress", "release status must be snake_case");

    // Positive filter, both encodings.
    for spelling in ["in_progress", "InProgress"] {
        let (rows, total) = client
            .list_releases_filtered(project_id, Some(vec![spelling.into()]), 10, 0, None, "asc")
            .await
            .unwrap();
        assert_eq!(total, 1, "{spelling} must match the stored release");
        assert_eq!(rows[0].id, rel.id);
    }

    let (_, total) = client
        .list_releases_filtered(project_id, None, 10, 0, Some("version"), "desc")
        .await
        .unwrap();
    assert_eq!(total, 2);

    // Negative.
    let (rows, total) = client
        .list_releases_filtered(project_id, Some(vec![EVIL.into()]), 10, 0, None, "asc")
        .await
        .unwrap();
    assert_eq!(total, 0, "payload must match no release");
    assert!(rows.is_empty());

    // `update_release` must keep the snake_case encoding too.
    client
        .update_release(
            rel.id,
            Some(ReleaseStatus::Released),
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    let stored: String = raw
        .execute(
            neo4rs::query("MATCH (r:Release {id: $id}) RETURN r.status AS s")
                .param("id", rel.id.to_string()),
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get("s")
        .unwrap();
    assert_eq!(stored, "released", "update_release must stay snake_case");

    drop_project(&raw, project_id).await;
}

#[tokio::test]
async fn test_workspace_milestone_status_filters_are_bound() {
    use project_orchestrator::neo4j::client::Neo4jClient;

    let config = test_config();
    let client = match Neo4jClient::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping test: Neo4j not available: {e}");
            return;
        }
    };

    let ws = WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Injection WS".to_string(),
        slug: format!("injection-ws-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    };
    client.create_workspace(&ws).await.unwrap();

    let wm_in_progress = WorkspaceMilestoneNode {
        id: Uuid::new_v4(),
        workspace_id: ws.id,
        title: "Cross-project hardening".to_string(),
        description: None,
        status: MilestoneStatus::InProgress,
        target_date: None,
        closed_at: None,
        created_at: chrono::Utc::now(),
        tags: vec!["security".to_string()],
    };
    let wm_open = WorkspaceMilestoneNode {
        id: Uuid::new_v4(),
        title: "Later".to_string(),
        status: MilestoneStatus::Open,
        ..wm_in_progress.clone()
    };
    client
        .create_workspace_milestone(&wm_in_progress)
        .await
        .unwrap();
    client.create_workspace_milestone(&wm_open).await.unwrap();

    // Per-workspace listing: positive on the stored snake_case encoding, and
    // the status must survive the node -> model conversion.
    let (rows, total) = client
        .list_workspace_milestones_filtered(ws.id, Some("in_progress"), 10, 0)
        .await
        .unwrap();
    assert_eq!(total, 1, "in_progress must match the stored milestone");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, wm_in_progress.id);
    assert_eq!(
        rows[0].status,
        MilestoneStatus::InProgress,
        "node_to_workspace_milestone must decode snake_case status"
    );
    assert_eq!(rows[0].tags, vec!["security".to_string()]);

    // PascalCase input must work as well.
    let (_, total) = client
        .list_workspace_milestones_filtered(ws.id, Some("InProgress"), 10, 0)
        .await
        .unwrap();
    assert_eq!(total, 1, "PascalCase input must match too");

    let (_, total) = client
        .list_workspace_milestones_filtered(ws.id, None, 10, 0)
        .await
        .unwrap();
    assert_eq!(total, 2);

    // Negative.
    let (rows, total) = client
        .list_workspace_milestones_filtered(ws.id, Some(EVIL), 10, 0)
        .await
        .unwrap();
    assert_eq!(total, 0, "payload must match no workspace milestone");
    assert!(rows.is_empty());

    // Cross-workspace listing. The old code compared `wm.status` against the
    // PascalCase spelling while the writer stored snake_case, so this filter
    // could never match — the parameterization also fixed that mismatch.
    let all = client
        .list_all_workspace_milestones_filtered(Some(ws.id), Some("in_progress"), 10, 0)
        .await
        .unwrap();
    assert_eq!(all.len(), 1, "cross-workspace status filter must match");
    assert_eq!(all[0].0.id, wm_in_progress.id);

    let all = client
        .list_all_workspace_milestones_filtered(Some(ws.id), None, 10, 0)
        .await
        .unwrap();
    assert_eq!(all.len(), 2);

    let injected = client
        .list_all_workspace_milestones_filtered(Some(ws.id), Some(EVIL), 10, 0)
        .await
        .unwrap();
    assert!(injected.is_empty(), "payload must match nothing");

    // Counts must agree with the listings, workspace-scoped and global.
    assert_eq!(
        client
            .count_all_workspace_milestones(Some(ws.id), Some("in_progress"))
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        client
            .count_all_workspace_milestones(Some(ws.id), None)
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        client
            .count_all_workspace_milestones(Some(ws.id), Some(EVIL))
            .await
            .unwrap(),
        0
    );
    // Global (no workspace filter) must at least see our two rows.
    assert!(
        client
            .count_all_workspace_milestones(None, None)
            .await
            .unwrap()
            >= 2
    );

    client
        .delete_workspace_milestone(wm_in_progress.id)
        .await
        .ok();
    client.delete_workspace_milestone(wm_open.id).await.ok();
    client.delete_workspace(ws.id).await.ok();
}

#[tokio::test]
async fn test_agent_execution_completed_at_is_bound() {
    use project_orchestrator::neo4j::client::Neo4jClient;
    use project_orchestrator::neo4j::{AgentExecutionNode, AgentExecutionStatus};

    let config = test_config();
    let client = match Neo4jClient::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping test: Neo4j not available: {e}");
            return;
        }
    };
    let raw = neo4rs::Graph::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    .unwrap();

    let run_id = Uuid::new_v4();
    let task_id = Uuid::new_v4();
    raw.run(
        neo4rs::query(
            "CREATE (r:PlanRun {run_id: $rid, status: 'running'}), \
                    (t:Task {id: $tid, title: 'ae-test'})",
        )
        .param("rid", run_id.to_string())
        .param("tid", task_id.to_string()),
    )
    .await
    .unwrap();

    let mut ae = AgentExecutionNode {
        id: Uuid::new_v4(),
        run_id,
        task_id,
        session_id: None,
        started_at: chrono::Utc::now(),
        completed_at: None,
        cost_usd: 0.0,
        duration_secs: 0.0,
        status: AgentExecutionStatus::Running,
        tools_used: "[]".to_string(),
        files_modified: vec![],
        commits: vec![],
        persona_profile: String::new(),
        vector_json: None,
        report_json: None,
        execution_type: Default::default(),
        ..Default::default()
    };
    client.create_agent_execution_impl(&ae).await.unwrap();

    // Update with a completion timestamp: the value travels as $completed_at
    // and must still be understood by Cypher's datetime().
    let done_at = chrono::Utc::now();
    ae.completed_at = Some(done_at);
    ae.status = AgentExecutionStatus::Completed;
    ae.cost_usd = 1.25;
    ae.duration_secs = 42.0;
    ae.files_modified = vec!["src/lib.rs".to_string()];
    ae.commits = vec!["deadbeef".to_string()];
    client.update_agent_execution_impl(&ae).await.unwrap();

    let (completed, status): (bool, String) = {
        let mut r = raw
            .execute(
                neo4rs::query(
                    "MATCH (ae:AgentExecution {id: $id}) \
                     RETURN ae.completed_at IS NOT NULL AS done, ae.status AS st",
                )
                .param("id", ae.id.to_string()),
            )
            .await
            .unwrap();
        let row = r.next().await.unwrap().unwrap();
        (row.get("done").unwrap(), row.get("st").unwrap())
    };
    assert!(
        completed,
        "completed_at must be set — a $completed_at left unbound would have \
         raised ParameterMissing or stored nothing"
    );
    assert_eq!(status, "completed");

    // And an update that leaves completed_at unset must not reference the
    // parameter at all (an unbound $completed_at would make the query fail).
    let mut still_running = ae.clone();
    still_running.completed_at = None;
    still_running.status = AgentExecutionStatus::Running;
    client
        .update_agent_execution_impl(&still_running)
        .await
        .expect("update without completed_at must not reference $completed_at");

    raw.run(
        neo4rs::query(
            "MATCH (r:PlanRun {run_id: $rid}) OPTIONAL MATCH (ae:AgentExecution)-[:PART_OF]->(r) \
             DETACH DELETE r, ae",
        )
        .param("rid", run_id.to_string()),
    )
    .await
    .unwrap();
    raw.run(
        neo4rs::query("MATCH (t:Task {id: $tid}) DETACH DELETE t")
            .param("tid", task_id.to_string()),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn test_meili_filters_escape_payloads_and_stay_valid() {
    use project_orchestrator::meilisearch::indexes::{
        CodeDocument, DecisionDocument, NoteDocument,
    };

    if !backends_available().await {
        eprintln!("Skipping test: backends not available");
        return;
    }

    let config = test_config();
    let state = AppState::new(config).await.unwrap();

    let tag = Uuid::new_v4().to_string();
    let slug = format!("inj-{tag}");

    // --- notes ------------------------------------------------------------
    let note = NoteDocument {
        id: format!("note-{tag}"),
        project_id: Uuid::new_v4().to_string(),
        project_slug: slug.clone(),
        note_type: "guideline".to_string(),
        status: "active".to_string(),
        importance: "high".to_string(),
        scope_type: "project".to_string(),
        scope_path: "src".to_string(),
        content: "parameterize every filter".to_string(),
        tags: vec!["security".to_string()],
        anchor_entities: vec![],
        created_at: chrono::Utc::now().timestamp(),
        created_by: "test".to_string(),
        staleness_score: 0.0,
    };
    state.meili.index_note(&note).await.unwrap();
    tokio::time::sleep(Duration::from_millis(800)).await;

    // Positive: every filter field at once still finds the document, so the
    // quoted literals are valid MeiliSearch filter syntax.
    let hits = state
        .meili
        .search_notes_with_scores(
            "parameterize",
            10,
            Some(&slug),
            Some("guideline"),
            Some("active"),
            Some("high"),
        )
        .await
        .expect("a fully filtered note search must be accepted by MeiliSearch");
    assert!(
        hits.iter().any(|h| h.document.id == note.id),
        "the note must be found through its own filters"
    );

    // Negative: a payload that tries to close the literal and OR a tautology.
    // MeiliSearch must accept the escaped filter (no Err) and match nothing.
    for payload in [
        "x\" OR project_slug != \"zzz",
        "x\" OR status = \"active",
        "x\\",
        EVIL,
    ] {
        let injected = state
            .meili
            .search_notes_with_scores("parameterize", 10, Some(payload), None, None, None)
            .await
            .unwrap_or_else(|e| panic!("escaped filter must stay valid for {payload:?}: {e}"));
        assert!(
            injected.is_empty(),
            "payload {payload:?} must match no note, got {}",
            injected.len()
        );
    }

    // --- code -------------------------------------------------------------
    let code = CodeDocument {
        id: format!("code-{tag}"),
        path: format!("/inj/{tag}/example.rs"),
        language: "rust".to_string(),
        symbols: vec![format!("inj_{tag}")],
        docstrings: "injection fixture".to_string(),
        signatures: vec![],
        imports: vec![],
        project_id: Uuid::new_v4().to_string(),
        project_slug: slug.clone(),
    };
    state.meili.index_code(&code).await.unwrap();
    tokio::time::sleep(Duration::from_millis(800)).await;

    let hits = state
        .meili
        .search_code_with_scores(
            "injection",
            10,
            Some("rust"),
            Some(&slug),
            Some(&format!("/inj/{tag}/")),
        )
        .await
        .expect("language + slug + path_prefix filter must be valid");
    assert!(
        hits.iter().any(|h| h.document.id == code.id),
        "the code document must be found through its own filters"
    );

    for payload in ["x\" OR language != \"zzz", "x\\"] {
        let injected = state
            .meili
            .search_code_with_scores("injection", 10, Some(payload), None, None)
            .await
            .unwrap_or_else(|e| panic!("escaped code filter must stay valid for {payload:?}: {e}"));
        assert!(
            injected.is_empty(),
            "payload {payload:?} must match no code"
        );
        let injected = state
            .meili
            .search_code_with_scores("injection", 10, None, None, Some(payload))
            .await
            .unwrap_or_else(|e| panic!("escaped path prefix must stay valid for {payload:?}: {e}"));
        assert!(
            injected.is_empty(),
            "payload {payload:?} must match no path"
        );
    }

    // --- decisions --------------------------------------------------------
    let task_id = Uuid::new_v4().to_string();
    let decision = DecisionDocument {
        id: format!("dec-{tag}"),
        description: "bind the filters".to_string(),
        rationale: "injection fixture".to_string(),
        task_id: task_id.clone(),
        agent: "test".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        tags: vec![],
        project_id: None,
        project_slug: Some(slug.clone()),
    };
    state.meili.index_decision(&decision).await.unwrap();
    tokio::time::sleep(Duration::from_millis(800)).await;

    let hits = state
        .meili
        .search_decisions_in_project("bind", 10, Some(&slug))
        .await
        .expect("single-slug decision filter must be valid");
    assert!(hits.iter().any(|d| d.id == decision.id));

    let hits = state
        .meili
        .search_decisions_in_projects("bind", 10, &[slug.clone(), EVIL.to_string()])
        .await
        .expect("an IN [...] list with a payload must stay valid");
    assert!(
        hits.iter().any(|d| d.id == decision.id),
        "the legitimate slug must still match inside the IN list"
    );

    for payload in ["x\" OR project_slug != \"zzz", "x\\"] {
        let injected = state
            .meili
            .search_decisions_in_project("bind", 10, Some(payload))
            .await
            .unwrap_or_else(|e| panic!("escaped decision filter must stay valid: {e}"));
        assert!(
            injected.is_empty(),
            "payload {payload:?} must match nothing"
        );
    }

    // Deletions build the same escaped `field = "value"` filter. A payload must
    // be accepted as a literal and delete nothing — in particular it must not
    // widen into "delete everything".
    for payload in ["x\" OR project_slug != \"zzz", "x\\", EVIL] {
        state
            .meili
            .delete_code_for_project(payload)
            .await
            .unwrap_or_else(|e| {
                panic!("delete_code_for_project({payload:?}) must stay valid: {e}")
            });
        state
            .meili
            .delete_decisions_for_project(payload)
            .await
            .unwrap_or_else(|e| panic!("delete_decisions_for_project({payload:?}): {e}"));
        state
            .meili
            .delete_decisions_for_task(payload)
            .await
            .unwrap_or_else(|e| panic!("delete_decisions_for_task({payload:?}): {e}"));
        state
            .meili
            .delete_notes_for_project(payload)
            .await
            .unwrap_or_else(|e| panic!("delete_notes_for_project({payload:?}): {e}"));
    }

    // Our fixtures must have survived every injected deletion.
    let hits = state
        .meili
        .search_notes_with_scores("parameterize", 10, Some(&slug), None, None, None)
        .await
        .unwrap();
    assert!(
        hits.iter().any(|h| h.document.id == note.id),
        "an injected delete filter must not have removed unrelated notes"
    );
    let hits = state
        .meili
        .search_decisions_in_project("bind", 10, Some(&slug))
        .await
        .unwrap();
    assert!(
        hits.iter().any(|d| d.id == decision.id),
        "an injected delete filter must not have removed unrelated decisions"
    );

    // Cleanup: the real slug does delete.
    state.meili.delete_notes_for_project(&slug).await.unwrap();
    state
        .meili
        .delete_decisions_for_project(&slug)
        .await
        .unwrap();
    state.meili.delete_code_for_project(&slug).await.unwrap();
}

/// `entity_type` reaches a Cypher **label** position, which cannot be a bound
/// parameter — the whitelist in `safe_entity_label` is the only thing between
/// the caller and arbitrary Cypher. Both reverse-lookup entry points must
/// consult it *before* building their query.
#[tokio::test]
async fn test_decision_entity_label_whitelist_blocks_label_injection() {
    use project_orchestrator::neo4j::client::Neo4jClient;

    let config = test_config();
    let client = match Neo4jClient::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping test: Neo4j not available: {e}");
            return;
        }
    };

    // Positive: a known entity type is accepted and the query is valid Cypher
    // (an empty result is fine — what matters is Ok, not a syntax error).
    for ok_type in ["file", "function", "File", "struct"] {
        client
            .get_decisions_for_entity(ok_type, "src/lib.rs", 5)
            .await
            .unwrap_or_else(|e| panic!("get_decisions_for_entity({ok_type:?}) must work: {e}"));
        client
            .get_decisions_affecting(ok_type, "src/lib.rs", None)
            .await
            .unwrap_or_else(|e| panic!("get_decisions_affecting({ok_type:?}) must work: {e}"));
    }

    // A status filter is still honoured on the accepted path.
    client
        .get_decisions_affecting("file", "src/lib.rs", Some("accepted"))
        .await
        .unwrap();

    // Negative: anything that is not a known label is rejected outright. If it
    // were interpolated instead, `File) DETACH DELETE n //` would splice a
    // delete into the MATCH.
    for payload in [
        "File) DETACH DELETE n //",
        "a' OR 1=1 //",
        "Function {id: 'x'}) RETURN d //",
        "x\\",
        "",
        "Milestone|File",
    ] {
        let err = client
            .get_decisions_for_entity(payload, "src/lib.rs", 5)
            .await
            .expect_err(&format!("get_decisions_for_entity accepted {payload:?}"));
        assert!(
            err.to_string().contains("invalid entity_type"),
            "unexpected error for {payload:?}: {err}"
        );
        let err = client
            .get_decisions_affecting(payload, "src/lib.rs", None)
            .await
            .expect_err(&format!("get_decisions_affecting accepted {payload:?}"));
        assert!(
            err.to_string().contains("invalid entity_type"),
            "unexpected error for {payload:?}: {err}"
        );
    }
}
