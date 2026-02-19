//! Test helper factories and mock state builders
//!
//! Provides convenience functions for creating test objects with sensible defaults,
//! and helpers for building mock AppState / Orchestrator / ToolHandler instances.
#![allow(dead_code)]

use crate::meilisearch::mock::MockSearchStore;
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::models::*;
use crate::notes::{Note, NoteImportance, NoteScope, NoteType};
use crate::plan::models::*;
use crate::{AppState, AuthConfig};
use std::sync::Arc;
use uuid::Uuid;

// ============================================================================
// Mock state builders
// ============================================================================

/// Create a mock AppState with empty in-memory backends
pub fn mock_app_state() -> AppState {
    AppState {
        neo4j: Arc::new(MockGraphStore::new()),
        meili: Arc::new(MockSearchStore::new()),
        parser: Arc::new(crate::parser::CodeParser::new().expect("parser init")),
        config: Arc::new(crate::Config {
            setup_completed: true,
            neo4j_uri: "bolt://mock:7687".to_string(),
            neo4j_user: "neo4j".to_string(),
            neo4j_password: "mock".to_string(),
            meilisearch_url: "http://mock:7700".to_string(),
            meilisearch_key: "mock-key".to_string(),
            nats_url: None,
            workspace_path: ".".to_string(),
            server_port: 0,
            auth_config: None,
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            public_url: None,
            chat_permissions: None,
            chat_default_model: None,
            chat_max_sessions: None,
            chat_max_turns: None,
            chat_session_timeout_secs: None,
            config_yaml_path: None,
        }),
    }
}

/// Create a mock AppState with pre-seeded backends
pub fn mock_app_state_with(graph: MockGraphStore, search: MockSearchStore) -> AppState {
    AppState {
        neo4j: Arc::new(graph),
        meili: Arc::new(search),
        parser: Arc::new(crate::parser::CodeParser::new().expect("parser init")),
        config: Arc::new(crate::Config {
            setup_completed: true,
            neo4j_uri: "bolt://mock:7687".to_string(),
            neo4j_user: "neo4j".to_string(),
            neo4j_password: "mock".to_string(),
            meilisearch_url: "http://mock:7700".to_string(),
            meilisearch_key: "mock-key".to_string(),
            nats_url: None,
            workspace_path: ".".to_string(),
            server_port: 0,
            auth_config: None,
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            public_url: None,
            chat_permissions: None,
            chat_default_model: None,
            chat_max_sessions: None,
            chat_max_turns: None,
            chat_session_timeout_secs: None,
            config_yaml_path: None,
        }),
    }
}

/// Create a test AuthConfig suitable for integration tests.
///
/// Uses a fixed JWT secret and disables domain restriction.
/// Tests that use `create_router()` should pass `Some(test_auth_config())`
/// as `auth_config` to avoid deny-by-default 403 rejections.
pub fn test_auth_config() -> AuthConfig {
    AuthConfig {
        jwt_secret: "test-secret-key-minimum-32-chars!!".to_string(),
        access_token_expiry_secs: 900,
        refresh_token_expiry_secs: 604800,
        allowed_email_domain: None,
        allowed_emails: None,
        frontend_url: None,
        allow_registration: false,
        root_account: None,
        oidc: None,
        google_client_id: Some("test-client-id".to_string()),
        google_client_secret: Some("test-client-secret".to_string()),
        google_redirect_uri: Some("http://localhost:3000/auth/callback".to_string()),
    }
}

/// Generate a valid Bearer token string for test requests.
///
/// Returns the full header value: `"Bearer eyJ..."`.
/// Uses the same secret as `test_auth_config()`.
pub fn test_bearer_token() -> String {
    let token = crate::auth::jwt::encode_jwt(
        Uuid::new_v4(),
        "test@ffs.holdings",
        "Test User",
        "test-secret-key-minimum-32-chars!!",
        28800,
    )
    .expect("test token encoding should succeed");
    format!("Bearer {}", token)
}

// ============================================================================
// Test data factories
// ============================================================================

/// Create a test project with sensible defaults
pub fn test_project() -> ProjectNode {
    ProjectNode {
        id: Uuid::new_v4(),
        name: "test-project".to_string(),
        slug: "test-project".to_string(),
        root_path: "/tmp/test-project".to_string(),
        description: Some("A test project".to_string()),
        created_at: chrono::Utc::now(),
        last_synced: None,
    }
}

/// Create a test project with a specific name/slug
pub fn test_project_named(name: &str) -> ProjectNode {
    ProjectNode {
        id: Uuid::new_v4(),
        name: name.to_string(),
        slug: name.to_lowercase().replace(' ', "-"),
        root_path: format!("/tmp/{}", name.to_lowercase().replace(' ', "-")),
        description: Some(format!("Test project: {}", name)),
        created_at: chrono::Utc::now(),
        last_synced: None,
    }
}

/// Create a test plan with sensible defaults
pub fn test_plan() -> PlanNode {
    PlanNode::new(
        "Test Plan".to_string(),
        "A test plan for unit testing".to_string(),
        "test-agent".to_string(),
        5,
    )
}

/// Create a test plan linked to a project
pub fn test_plan_for_project(project_id: Uuid) -> PlanNode {
    PlanNode::new_for_project(
        "Test Plan".to_string(),
        "A test plan for unit testing".to_string(),
        "test-agent".to_string(),
        5,
        project_id,
    )
}

/// Create a test task with sensible defaults
pub fn test_task() -> TaskNode {
    TaskNode {
        id: Uuid::new_v4(),
        title: Some("Test Task".to_string()),
        description: "Implement test functionality".to_string(),
        status: TaskStatus::Pending,
        assigned_to: None,
        priority: Some(5),
        tags: vec!["test".to_string()],
        acceptance_criteria: vec!["Tests pass".to_string()],
        affected_files: vec![],
        estimated_complexity: Some(3),
        actual_complexity: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        started_at: None,
        completed_at: None,
    }
}

/// Create a test task with a specific title
pub fn test_task_titled(title: &str) -> TaskNode {
    let mut task = test_task();
    task.title = Some(title.to_string());
    task.description = format!("Task: {}", title);
    task
}

/// Create a test step
pub fn test_step(order: u32, description: &str) -> StepNode {
    StepNode::new(order, description.to_string(), None)
}

/// Create a test decision
pub fn test_decision(description: &str, rationale: &str) -> DecisionNode {
    DecisionNode {
        id: Uuid::new_v4(),
        description: description.to_string(),
        rationale: rationale.to_string(),
        alternatives: vec![],
        chosen_option: None,
        decided_by: "test-agent".to_string(),
        decided_at: chrono::Utc::now(),
    }
}

/// Create a test constraint
pub fn test_constraint(constraint_type: ConstraintType, description: &str) -> ConstraintNode {
    ConstraintNode {
        id: Uuid::new_v4(),
        constraint_type,
        description: description.to_string(),
        enforced_by: None,
    }
}

/// Create a test workspace
pub fn test_workspace() -> WorkspaceNode {
    WorkspaceNode {
        id: Uuid::new_v4(),
        name: "Test Workspace".to_string(),
        slug: "test-workspace".to_string(),
        description: Some("A test workspace".to_string()),
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    }
}

/// Create a test note
pub fn test_note(project_id: Uuid, note_type: NoteType, content: &str) -> Note {
    Note::new_full(
        Some(project_id),
        note_type,
        NoteImportance::Medium,
        NoteScope::Project,
        content.to_string(),
        vec![],
        "test-agent".to_string(),
    )
}

/// Create a test commit
pub fn test_commit(hash: &str, message: &str) -> CommitNode {
    CommitNode {
        hash: hash.to_string(),
        message: message.to_string(),
        author: "test-author".to_string(),
        timestamp: chrono::Utc::now(),
    }
}

/// Create a test release
pub fn test_release(project_id: Uuid, version: &str) -> ReleaseNode {
    ReleaseNode {
        id: Uuid::new_v4(),
        project_id,
        version: version.to_string(),
        title: Some(format!("Release {}", version)),
        description: None,
        status: ReleaseStatus::Planned,
        target_date: None,
        released_at: None,
        created_at: chrono::Utc::now(),
    }
}

/// Create a test chat session
pub fn test_chat_session(project_slug: Option<&str>) -> ChatSessionNode {
    ChatSessionNode {
        id: Uuid::new_v4(),
        cli_session_id: None,
        project_slug: project_slug.map(|s| s.to_string()),
        workspace_slug: None,
        cwd: "/tmp/test".to_string(),
        title: None,
        model: "claude-opus-4-6".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        message_count: 0,
        total_cost_usd: None,
        conversation_id: None,
        preview: None,
        permission_mode: None,
        add_dirs: None,
    }
}

/// Create a test user (OIDC provider)
pub fn test_user() -> UserNode {
    UserNode {
        id: Uuid::new_v4(),
        email: "test@ffs.holdings".to_string(),
        name: "Test User".to_string(),
        picture_url: Some("https://lh3.googleusercontent.com/test".to_string()),
        auth_provider: crate::neo4j::models::AuthProvider::Oidc,
        external_id: Some("google-id-123456".to_string()),
        password_hash: None,
        created_at: chrono::Utc::now(),
        last_login_at: chrono::Utc::now(),
    }
}

/// Create a test milestone
pub fn test_milestone(project_id: Uuid, title: &str) -> MilestoneNode {
    MilestoneNode {
        id: Uuid::new_v4(),
        project_id,
        title: title.to_string(),
        description: None,
        status: MilestoneStatus::Open,
        target_date: None,
        closed_at: None,
        created_at: chrono::Utc::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_app_state_creation() {
        let state = mock_app_state();
        assert_eq!(state.config.neo4j_uri, "bolt://mock:7687");
    }

    #[test]
    fn test_factory_functions_produce_valid_objects() {
        let project = test_project();
        assert!(!project.name.is_empty());

        let plan = test_plan();
        assert_eq!(plan.status, PlanStatus::Draft);

        let task = test_task();
        assert_eq!(task.status, TaskStatus::Pending);

        let step = test_step(0, "First step");
        assert_eq!(step.order, 0);

        let workspace = test_workspace();
        assert!(!workspace.slug.is_empty());
    }

    #[tokio::test]
    async fn test_mock_state_can_create_plan_manager() {
        let state = mock_app_state();
        let pm = crate::plan::PlanManager::new(state.neo4j.clone(), state.meili.clone());

        let plan = test_plan();
        let req = CreatePlanRequest {
            title: plan.title.clone(),
            description: plan.description.clone(),
            project_id: None,
            priority: Some(5),
            constraints: None,
        };

        let result = pm.create_plan(req, "test-agent").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_mock_state_can_create_note_manager() {
        let state = mock_app_state();
        let nm = crate::notes::NoteManager::new(state.neo4j.clone(), state.meili.clone());

        let project = test_project();
        // Store project first so note_to_document can find it
        state.neo4j.create_project(&project).await.unwrap();

        let req = crate::notes::CreateNoteRequest {
            project_id: Some(project.id),
            note_type: NoteType::Tip,
            content: "Test note content".to_string(),
            importance: Some(NoteImportance::Medium),
            scope: None,
            tags: None,
            anchors: None,
            assertion_rule: None,
        };

        let result = nm.create_note(req, "test-agent").await;
        assert!(result.is_ok());
    }
}
