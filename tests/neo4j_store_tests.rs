//! `GraphStore` trait methods exercised against a REAL Neo4j.
//!
//! Why this file exists: `codecov.yml` used to ignore every `src/neo4j/*.rs`
//! with the justification "all methods require a running Neo4j instance". The
//! CI *has* a Neo4j service (ci.yml: integration-tests, api-tests, coverage),
//! so the files were not untestable — they were untested. Measured on
//! 2026-10-01 (docs/COVERAGE.md): `commit`, `constraint`, `decision`,
//! `milestone`, `release`, `user`, `workspace`, `feature_graph`, `plan_run`,
//! `event_trigger` and `chat` sat at **0 %** with Neo4j up.
//!
//! Shape of each test: one `GraphStore` trait call per assertion (the trait,
//! not the inherent method, so `impl_graph_store.rs` is exercised too), the
//! happy path, then the error cases that production actually hits:
//!
//! - **unknown id** — does the call report "nothing matched", or claim success?
//! - **duplicate** — is a second create idempotent, or does it fan out?
//! - **empty input** — an all-`None` partial update, an empty list, "".
//!
//! Run with:
//! ```text
//! NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=testpassword \
//!   cargo test --test neo4j_store_tests
//! ```
//!
//! Every test allocates fresh UUIDs and deletes what it created, so the suite
//! is safe to run repeatedly against the same database and in parallel.
//!
//! WARNING: point `NEO4J_URI` at a disposable instance. The default
//! `bolt://localhost:7687` is a real Neo4j on many dev machines.

use project_orchestrator::chat::provider::cognitive::segment::{
    Boundary, Dimension, Fingerprint, Observation, Segment, SegmentPolicy, Segmenter,
};
use project_orchestrator::neo4j::models::*;
use project_orchestrator::neo4j::{GraphStore, Neo4jClient};
use uuid::Uuid;

// ===========================================================================
// Harness
// ===========================================================================

/// Connect to the Neo4j named by the environment.
///
/// Returns `None` when no Neo4j answers, so a developer without the service
/// still gets a green `cargo test`. In CI that silence would be a lie — a
/// skipped suite exits 0 and uploads a coverage report full of zeroes — so
/// when `CI` is set, an unreachable Neo4j is a hard failure.
async fn store() -> Option<Neo4jClient> {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "orchestrator123".into());

    match Neo4jClient::new(&uri, &user, &password).await {
        Ok(client) => Some(client),
        Err(err) => {
            if std::env::var("CI").is_ok() {
                panic!(
                    "Neo4j is required in CI but {uri} did not answer: {err}. \
                     A skipped suite would upload a coverage report in which \
                     src/neo4j/* is 0 % and look like the code is untestable."
                );
            }
            eprintln!("skipping: no Neo4j at {uri} ({err})");
            None
        }
    }
}

/// Bind a `&dyn GraphStore` or return early. Deliberately a trait object: an
/// inherent `Neo4jClient` method shadows the trait one, so calling through
/// `&dyn GraphStore` is what makes `impl_graph_store.rs` — the delegation layer
/// the API and MCP handlers actually go through — run too.
macro_rules! store_or_skip {
    () => {
        match store().await {
            Some(client) => client,
            None => return,
        }
    };
}

fn a_project() -> ProjectNode {
    ProjectNode {
        id: Uuid::new_v4(),
        name: "neo4j store test project".to_string(),
        slug: format!("n4j-store-{}", Uuid::new_v4()),
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
    }
}

fn a_workspace() -> WorkspaceNode {
    WorkspaceNode {
        id: Uuid::new_v4(),
        name: "neo4j store test workspace".to_string(),
        slug: format!("n4j-ws-{}", Uuid::new_v4()),
        description: None,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: serde_json::Value::Null,
    }
}

fn a_milestone(project_id: Uuid) -> MilestoneNode {
    MilestoneNode {
        id: Uuid::new_v4(),
        title: "M1".to_string(),
        description: Some("first".to_string()),
        status: MilestoneStatus::Planned,
        target_date: None,
        closed_at: None,
        created_at: chrono::Utc::now(),
        project_id,
    }
}

fn a_release(project_id: Uuid) -> ReleaseNode {
    ReleaseNode {
        id: Uuid::new_v4(),
        version: "0.0.0-test".to_string(),
        title: Some("Test release".to_string()),
        description: None,
        status: ReleaseStatus::Planned,
        target_date: None,
        released_at: None,
        created_at: chrono::Utc::now(),
        project_id,
    }
}

fn a_commit(hash: &str) -> CommitNode {
    CommitNode {
        hash: hash.to_string(),
        message: "test: a commit".to_string(),
        author: "tester".to_string(),
        timestamp: chrono::Utc::now(),
    }
}

fn a_decision() -> DecisionNode {
    DecisionNode {
        id: Uuid::new_v4(),
        description: "Use Neo4j for the graph".to_string(),
        rationale: "Cypher beats hand-rolled joins here".to_string(),
        alternatives: vec!["SQLite".to_string(), "in-memory".to_string()],
        chosen_option: Some("Neo4j".to_string()),
        decided_by: "tester".to_string(),
        decided_at: chrono::Utc::now(),
        status: DecisionStatus::Proposed,
        embedding: None,
        embedding_model: None,
        scar_intensity: 0.0,
        sharing_consent: Default::default(),
    }
}

fn a_password_user() -> UserNode {
    UserNode {
        id: Uuid::new_v4(),
        email: format!("n4j-store-{}@example.test", Uuid::new_v4()),
        name: "Test User".to_string(),
        picture_url: None,
        auth_provider: AuthProvider::Password,
        external_id: None,
        password_hash: Some("$2b$12$notarealhash".to_string()),
        created_at: chrono::Utc::now(),
        last_login_at: chrono::Utc::now(),
    }
}

// ===========================================================================
// src/neo4j/constraint.rs
// ===========================================================================

#[tokio::test]
async fn test_constraint_lifecycle_and_error_cases() {
    let client = store_or_skip!();
    let store: &dyn GraphStore = &client;

    let plan = PlanNode::new(
        format!("Constraint plan {}", Uuid::new_v4()),
        "plan carrying constraints".to_string(),
        "tester".to_string(),
        1,
    );
    store.create_plan(&plan).await.unwrap();

    // --- happy path ---
    let constraint = ConstraintNode {
        id: Uuid::new_v4(),
        constraint_type: ConstraintType::Performance,
        description: "p99 under 100ms".to_string(),
        enforced_by: Some("bench suite".to_string()),
    };
    store.create_constraint(plan.id, &constraint).await.unwrap();

    let fetched = store
        .get_constraint(constraint.id)
        .await
        .unwrap()
        .expect("the constraint just created must be readable");
    assert_eq!(fetched.description, "p99 under 100ms");
    assert_eq!(fetched.constraint_type, ConstraintType::Performance);
    assert_eq!(fetched.enforced_by.as_deref(), Some("bench suite"));

    let listed = store.get_plan_constraints(plan.id).await.unwrap();
    assert_eq!(listed.len(), 1, "the plan has exactly the one constraint");
    assert_eq!(listed[0].id, constraint.id);

    store
        .update_constraint(
            constraint.id,
            Some("p99 under 50ms".to_string()),
            Some(ConstraintType::Testing),
            None,
        )
        .await
        .unwrap();
    let updated = store.get_constraint(constraint.id).await.unwrap().unwrap();
    assert_eq!(updated.description, "p99 under 50ms");
    assert_eq!(updated.constraint_type, ConstraintType::Testing);
    assert_eq!(
        updated.enforced_by.as_deref(),
        Some("bench suite"),
        "a field left None must keep its stored value, not be blanked"
    );

    // --- error case: empty update (every field None) ---
    store
        .update_constraint(constraint.id, None, None, None)
        .await
        .expect("an update with nothing to set is a no-op, not an error");
    let untouched = store.get_constraint(constraint.id).await.unwrap().unwrap();
    assert_eq!(untouched.description, "p99 under 50ms");

    // --- error case: unknown id ---
    assert!(
        store
            .get_constraint(Uuid::new_v4())
            .await
            .unwrap()
            .is_none(),
        "an unknown constraint id reads as None, never as an error or a stub"
    );
    let unknown = Uuid::new_v4();
    store
        .update_constraint(unknown, Some("ghost".to_string()), None, None)
        .await
        .expect("updating an unknown constraint must not raise");
    assert!(
        store.get_constraint(unknown).await.unwrap().is_none(),
        "a partial update must MATCH, never MERGE an id into existence"
    );
    store
        .delete_constraint(unknown)
        .await
        .expect("deleting an unknown constraint is idempotent");

    // --- error case: creating a constraint under an unknown plan must return Err.
    //     Fix for bug 3693617b: Neo4j MATCH+CREATE was silently succeeding when
    //     the parent plan did not exist. ---
    let orphan_c = ConstraintNode::new(ConstraintType::Other, "orphan".into(), None);
    let err = store
        .create_constraint(Uuid::new_v4(), &orphan_c)
        .await
        .expect_err("create_constraint must fail when the plan does not exist");
    assert!(
        err.to_string().contains("not found in Neo4j"),
        "error message must name the missing parent: {err}"
    );
    assert!(
        store.get_constraint(orphan_c.id).await.unwrap().is_none(),
        "no constraint node must have been written on Err"
    );

    // --- error case: unknown plan has no constraints ---
    assert!(
        store
            .get_plan_constraints(Uuid::new_v4())
            .await
            .unwrap()
            .is_empty(),
        "an unknown plan id yields an empty list, not an error"
    );

    // --- cleanup ---
    store.delete_constraint(constraint.id).await.unwrap();
    assert!(store.get_constraint(constraint.id).await.unwrap().is_none());
    store.delete_plan(plan.id).await.unwrap();
}

// ===========================================================================
// src/neo4j/milestone.rs
// ===========================================================================

#[tokio::test]
async fn test_milestone_lifecycle_progress_and_error_cases() {
    let client = store_or_skip!();
    let store: &dyn GraphStore = &client;

    let project = a_project();
    store.create_project(&project).await.unwrap();

    // --- happy path ---
    let milestone = a_milestone(project.id);
    store.create_milestone(&milestone).await.unwrap();

    let fetched = store
        .get_milestone(milestone.id)
        .await
        .unwrap()
        .expect("the milestone just created must be readable");
    assert_eq!(fetched.title, "M1");
    assert_eq!(fetched.status, MilestoneStatus::Planned);
    assert_eq!(fetched.project_id, project.id);

    let listed = store.list_project_milestones(project.id).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, milestone.id);

    store
        .update_milestone(
            milestone.id,
            Some(MilestoneStatus::InProgress),
            None,
            None,
            Some("M1 renamed".to_string()),
            None,
        )
        .await
        .unwrap();
    let updated = store.get_milestone(milestone.id).await.unwrap().unwrap();
    assert_eq!(updated.status, MilestoneStatus::InProgress);
    assert_eq!(updated.title, "M1 renamed");

    // --- progress counts tasks reached through the milestone ---
    let plan = PlanNode::new(
        format!("Milestone plan {}", Uuid::new_v4()),
        "plan whose tasks belong to a milestone".to_string(),
        "tester".to_string(),
        1,
    );
    store.create_plan(&plan).await.unwrap();
    let task = TaskNode::new("milestone task".to_string());
    store.create_task(plan.id, &task).await.unwrap();
    store
        .add_task_to_milestone(milestone.id, task.id)
        .await
        .unwrap();

    let (total, _completed, _in_progress, _pending) =
        store.get_milestone_progress(milestone.id).await.unwrap();
    assert_eq!(total, 1, "the milestone counts its one task");

    // --- error case: unknown milestone has an all-zero progress, not an error ---
    assert_eq!(
        store.get_milestone_progress(Uuid::new_v4()).await.unwrap(),
        (0, 0, 0, 0),
        "progress of an unknown milestone is zeroes, not an error"
    );

    // --- error case: unknown project ---
    let orphan = a_milestone(Uuid::new_v4());
    let created = store.create_milestone(&orphan).await;
    assert!(
        created.is_err(),
        "creating a milestone under an unknown project must fail loudly, \
         not leave a milestone dangling outside any project"
    );
    assert!(store.get_milestone(orphan.id).await.unwrap().is_none());

    // --- error case: unknown milestone id ---
    assert!(store.get_milestone(Uuid::new_v4()).await.unwrap().is_none());
    let unknown = Uuid::new_v4();
    store
        .update_milestone(
            unknown,
            Some(MilestoneStatus::Closed),
            None,
            None,
            None,
            None,
        )
        .await
        .expect("updating an unknown milestone must not raise");
    assert!(
        store.get_milestone(unknown).await.unwrap().is_none(),
        "a partial update must MATCH, never MERGE an id into existence"
    );

    // --- error case: empty update ---
    store
        .update_milestone(milestone.id, None, None, None, None, None)
        .await
        .expect("an update with nothing to set is a no-op");
    let still = store.get_milestone(milestone.id).await.unwrap().unwrap();
    assert_eq!(still.title, "M1 renamed");

    // --- error case: unknown project lists nothing ---
    assert!(store
        .list_project_milestones(Uuid::new_v4())
        .await
        .unwrap()
        .is_empty());

    // --- cleanup ---
    store.delete_milestone(milestone.id).await.unwrap();
    assert!(store.get_milestone(milestone.id).await.unwrap().is_none());
    store.delete_plan(plan.id).await.unwrap();
    store
        .delete_project(project.id, &project.name)
        .await
        .unwrap();
}

// ===========================================================================
// src/neo4j/release.rs + src/neo4j/commit.rs
// ===========================================================================

#[tokio::test]
async fn test_release_lifecycle_commit_links_and_error_cases() {
    let client = store_or_skip!();
    let store: &dyn GraphStore = &client;

    let project = a_project();
    store.create_project(&project).await.unwrap();

    // --- happy path ---
    let release = a_release(project.id);
    store.create_release(&release).await.unwrap();

    let fetched = store
        .get_release(release.id)
        .await
        .unwrap()
        .expect("the release just created must be readable");
    assert_eq!(fetched.version, "0.0.0-test");
    assert_eq!(fetched.status, ReleaseStatus::Planned);

    let listed = store.list_project_releases(project.id).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, release.id);

    let released_at = chrono::Utc::now();
    store
        .update_release(
            release.id,
            Some(ReleaseStatus::Released),
            None,
            Some(released_at),
            None,
            Some("shipped".to_string()),
        )
        .await
        .unwrap();
    let updated = store.get_release(release.id).await.unwrap().unwrap();
    assert_eq!(updated.status, ReleaseStatus::Released);
    assert_eq!(updated.description.as_deref(), Some("shipped"));
    assert!(updated.released_at.is_some());
    assert_eq!(
        updated.title.as_deref(),
        Some("Test release"),
        "a field left None keeps its stored value"
    );

    // --- a commit joins the release ---
    let hash = format!("{:x}", Uuid::new_v4().as_u128());
    let commit = a_commit(&hash);
    store.create_commit(&commit).await.unwrap();
    store
        .add_commit_to_release(release.id, &hash)
        .await
        .unwrap();

    // --- duplicate: adding the same commit twice must not double the link ---
    store
        .add_commit_to_release(release.id, &hash)
        .await
        .expect("adding the same commit twice is idempotent (MERGE)");
    let details = store
        .get_release_details(release.id)
        .await
        .unwrap()
        .expect("release details must resolve");
    assert_eq!(
        details.2.len(),
        1,
        "the commit appears once, not once per add_commit_to_release call"
    );

    store
        .remove_commit_from_release(release.id, &hash)
        .await
        .unwrap();
    let after_removal = store
        .get_release_details(release.id)
        .await
        .unwrap()
        .unwrap();
    assert!(after_removal.2.is_empty(), "the link is gone");

    // --- error case: removing a link that is not there ---
    store
        .remove_commit_from_release(release.id, &hash)
        .await
        .expect("removing an absent link is idempotent");

    // --- error case: empty commit hash ---
    store
        .add_commit_to_release(release.id, "")
        .await
        .expect("an empty hash matches no commit and must not raise");
    let unchanged = store
        .get_release_details(release.id)
        .await
        .unwrap()
        .unwrap();
    assert!(
        unchanged.2.is_empty(),
        "an empty hash must not attach a phantom commit"
    );

    // --- error case: unknown ids ---
    assert!(store.get_release(Uuid::new_v4()).await.unwrap().is_none());
    assert!(store
        .get_release_details(Uuid::new_v4())
        .await
        .unwrap()
        .is_none());
    assert!(store
        .list_project_releases(Uuid::new_v4())
        .await
        .unwrap()
        .is_empty());
    let unknown = Uuid::new_v4();
    store
        .update_release(
            unknown,
            Some(ReleaseStatus::Cancelled),
            None,
            None,
            None,
            None,
        )
        .await
        .expect("updating an unknown release must not raise");
    assert!(
        store.get_release(unknown).await.unwrap().is_none(),
        "a partial update must MATCH, never MERGE an id into existence"
    );

    // --- error case: empty update ---
    store
        .update_release(release.id, None, None, None, None, None)
        .await
        .expect("an update with nothing to set is a no-op");
    assert_eq!(
        store.get_release(release.id).await.unwrap().unwrap().status,
        ReleaseStatus::Released
    );

    // --- error case: a release created under an unknown project must return Err.
    //     Fix for bug 3693617b: Neo4j MATCH+CREATE was silently succeeding when
    //     the parent project did not exist. ---
    let orphan = a_release(Uuid::new_v4());
    let err = store
        .create_release(&orphan)
        .await
        .expect_err("create_release must fail when the project does not exist");
    assert!(
        err.to_string().contains("not found in Neo4j"),
        "error message must name the missing parent: {err}"
    );
    assert!(
        store.get_release(orphan.id).await.unwrap().is_none(),
        "no release node must have been written on Err"
    );

    // --- cleanup ---
    store.delete_release(release.id).await.unwrap();
    assert!(store.get_release(release.id).await.unwrap().is_none());
    store.delete_commit(&hash).await.unwrap();
    store
        .delete_project(project.id, &project.name)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_commit_lifecycle_duplicate_and_error_cases() {
    let client = store_or_skip!();
    let store: &dyn GraphStore = &client;

    let plan = PlanNode::new(
        format!("Commit plan {}", Uuid::new_v4()),
        "plan whose task owns commits".to_string(),
        "tester".to_string(),
        1,
    );
    store.create_plan(&plan).await.unwrap();
    let task = TaskNode::new("commit task".to_string());
    store.create_task(plan.id, &task).await.unwrap();

    // --- happy path ---
    let hash = format!("{:x}", Uuid::new_v4().as_u128());
    let commit = a_commit(&hash);
    store.create_commit(&commit).await.unwrap();

    let fetched = store
        .get_commit(&hash)
        .await
        .unwrap()
        .expect("the commit just created must be readable");
    assert_eq!(fetched.hash, hash);
    assert_eq!(fetched.author, "tester");

    store.link_commit_to_task(&hash, task.id).await.unwrap();
    let task_commits = store.get_task_commits(task.id).await.unwrap();
    assert_eq!(task_commits.len(), 1);
    assert_eq!(task_commits[0].hash, hash);

    // --- duplicate: create_commit MERGEs on hash, so a re-create updates in
    //     place instead of producing a second node. ---
    let amended = CommitNode {
        hash: hash.clone(),
        message: "test: amended message".to_string(),
        author: "tester".to_string(),
        timestamp: commit.timestamp,
    };
    store.create_commit(&amended).await.unwrap();
    assert_eq!(
        store.get_commit(&hash).await.unwrap().unwrap().message,
        "test: amended message",
        "a second create on the same hash updates the node"
    );
    assert_eq!(
        store.get_task_commits(task.id).await.unwrap().len(),
        1,
        "re-creating must not fan out into a second Commit node"
    );

    // --- duplicate: linking twice must not double the relation ---
    store.link_commit_to_task(&hash, task.id).await.unwrap();
    assert_eq!(
        store.get_task_commits(task.id).await.unwrap().len(),
        1,
        "the commit appears once, not once per link call"
    );

    // --- error case: unknown / empty hash ---
    assert!(store
        .get_commit(&format!("{:x}", Uuid::new_v4().as_u128()))
        .await
        .unwrap()
        .is_none());
    assert!(
        store.get_commit("").await.unwrap().is_none(),
        "an empty hash matches nothing"
    );

    // --- error case: unknown task owns no commits ---
    assert!(store
        .get_task_commits(Uuid::new_v4())
        .await
        .unwrap()
        .is_empty());

    // --- error case: linking to an unknown task leaves no relation ---
    store
        .link_commit_to_task(&hash, Uuid::new_v4())
        .await
        .expect("linking to an unknown task must not raise");

    // --- error case: deleting an unknown commit is idempotent ---
    store
        .delete_commit(&format!("{:x}", Uuid::new_v4().as_u128()))
        .await
        .unwrap();

    // --- cleanup ---
    store.delete_commit(&hash).await.unwrap();
    assert!(store.get_commit(&hash).await.unwrap().is_none());
    store.delete_plan(plan.id).await.unwrap();
}

// ===========================================================================
// src/neo4j/decision.rs
// ===========================================================================

#[tokio::test]
async fn test_decision_lifecycle_and_error_cases() {
    let client = store_or_skip!();
    let store: &dyn GraphStore = &client;

    let plan = PlanNode::new(
        format!("Decision plan {}", Uuid::new_v4()),
        "plan whose task records decisions".to_string(),
        "tester".to_string(),
        1,
    );
    store.create_plan(&plan).await.unwrap();
    let task = TaskNode::new("decision task".to_string());
    store.create_task(plan.id, &task).await.unwrap();

    // --- happy path ---
    let decision = a_decision();
    store.create_decision(task.id, &decision).await.unwrap();

    let fetched = store
        .get_decision(decision.id)
        .await
        .unwrap()
        .expect("the decision just created must be readable");
    assert_eq!(fetched.description, "Use Neo4j for the graph");
    assert_eq!(fetched.status, DecisionStatus::Proposed);
    assert_eq!(
        fetched.alternatives.len(),
        2,
        "the alternatives list round-trips"
    );

    store
        .update_decision(
            decision.id,
            None,
            Some("measured, not assumed".to_string()),
            None,
            Some(DecisionStatus::Accepted),
        )
        .await
        .unwrap();
    let updated = store.get_decision(decision.id).await.unwrap().unwrap();
    assert_eq!(updated.rationale, "measured, not assumed");
    assert_eq!(updated.status, DecisionStatus::Accepted);
    assert_eq!(
        updated.description, "Use Neo4j for the graph",
        "a field left None keeps its stored value"
    );

    // --- error case: empty update ---
    store
        .update_decision(decision.id, None, None, None, None)
        .await
        .expect("an update with nothing to set is a no-op");
    assert_eq!(
        store
            .get_decision(decision.id)
            .await
            .unwrap()
            .unwrap()
            .status,
        DecisionStatus::Accepted
    );

    // --- error case: unknown id ---
    assert!(store.get_decision(Uuid::new_v4()).await.unwrap().is_none());
    let unknown = Uuid::new_v4();
    store
        .update_decision(unknown, Some("ghost".to_string()), None, None, None)
        .await
        .expect("updating an unknown decision must not raise");
    assert!(
        store.get_decision(unknown).await.unwrap().is_none(),
        "a partial update must MATCH, never MERGE an id into existence"
    );
    store
        .delete_decision(unknown)
        .await
        .expect("deleting an unknown decision is idempotent");

    // --- error case: creating a decision under an unknown task must return Err.
    //     Fix for bug 3693617b: Neo4j MATCH+CREATE was silently succeeding when
    //     the parent task did not exist. ---
    let orphan = a_decision();
    let err = store
        .create_decision(Uuid::new_v4(), &orphan)
        .await
        .expect_err("create_decision must fail when the task does not exist");
    assert!(
        err.to_string().contains("not found in Neo4j"),
        "error message must name the missing parent: {err}"
    );
    assert!(
        store.get_decision(orphan.id).await.unwrap().is_none(),
        "no decision node must have been written on Err"
    );

    // --- cleanup ---
    store.delete_decision(decision.id).await.unwrap();
    assert!(store.get_decision(decision.id).await.unwrap().is_none());
    store.delete_plan(plan.id).await.unwrap();
}

// ===========================================================================
// src/neo4j/user.rs
// ===========================================================================

#[tokio::test]
async fn test_user_upsert_is_idempotent_and_rejects_oidc_without_external_id() {
    let client = store_or_skip!();
    let store: &dyn GraphStore = &client;

    // --- happy path ---
    let user = a_password_user();
    let created = store.upsert_user(&user).await.unwrap();
    assert_eq!(created.id, user.id);
    assert_eq!(created.email, user.email);
    assert_eq!(created.auth_provider, AuthProvider::Password);

    let by_id = store
        .get_user_by_id(user.id)
        .await
        .unwrap()
        .expect("the user just created must be readable by id");
    assert_eq!(by_id.email, user.email);
    let by_email = store
        .get_user_by_email(&user.email)
        .await
        .unwrap()
        .expect("…and by email");
    assert_eq!(by_email.id, user.id);

    assert!(
        store
            .list_users()
            .await
            .unwrap()
            .iter()
            .any(|u| u.id == user.id),
        "the user shows up in list_users"
    );

    // --- duplicate: a second upsert MERGEs on (provider, email). It must
    //     update the profile and KEEP the original id — a new id would orphan
    //     every node already linked to the user. ---
    let second = UserNode {
        id: Uuid::new_v4(),
        name: "Renamed User".to_string(),
        ..user.clone()
    };
    let merged = store.upsert_user(&second).await.unwrap();
    assert_eq!(
        merged.id, user.id,
        "re-upserting keeps the stored id, it does not take the new one"
    );
    assert_eq!(merged.name, "Renamed User", "the profile is refreshed");
    assert!(
        store.get_user_by_id(second.id).await.unwrap().is_none(),
        "the id offered by the second upsert never becomes a user"
    );

    // --- error case: an OIDC user with no external_id has no stable key ---
    let broken = UserNode {
        id: Uuid::new_v4(),
        email: format!("oidc-{}@example.test", Uuid::new_v4()),
        auth_provider: AuthProvider::Oidc,
        external_id: None,
        password_hash: None,
        ..a_password_user()
    };
    let err = store
        .upsert_user(&broken)
        .await
        .expect_err("an OIDC user without external_id must be refused");
    assert!(
        err.to_string().contains("external_id"),
        "the error names the missing field, got: {err}"
    );
    assert!(
        store.get_user_by_id(broken.id).await.unwrap().is_none(),
        "the refused upsert wrote nothing"
    );

    // --- error case: unknown id / unknown and empty email ---
    assert!(store
        .get_user_by_id(Uuid::new_v4())
        .await
        .unwrap()
        .is_none());
    assert!(store
        .get_user_by_email(&format!("absent-{}@example.test", Uuid::new_v4()))
        .await
        .unwrap()
        .is_none());
    assert!(
        store.get_user_by_email("").await.unwrap().is_none(),
        "an empty email matches no user"
    );
}

// ===========================================================================
// src/neo4j/workspace.rs — the regression that used to pass silently
// ===========================================================================

/// `update_workspace_milestone` answers whether the id matched.
///
/// Regression guard for the fix in #465 (commit dde8bfbf, "silent failure
/// paths found by the design-ref audit"). Before it, the method ran
/// `MATCH … SET …` and returned `Ok(())` whichever way: an unknown id produced
/// a cheerful 204 at the API layer, and a partial update of nothing at all
/// looked like it had worked. It returns `Result<bool>` via `run_matched` now,
/// and every assertion below on `false` fails against the old signature.
///
/// Note for whoever reads the bug inventory of 2026-09-30: it claims this
/// method "panics on an unknown id". It never did — `git show
/// dde8bfbf^:src/neo4j/workspace.rs` shows a plain `Ok(())`. Silent success,
/// not a panic; a different bug needing a different assertion.
#[tokio::test]
async fn test_update_workspace_milestone_reports_whether_the_id_matched() {
    let client = store_or_skip!();
    let store: &dyn GraphStore = &client;

    let workspace = a_workspace();
    store.create_workspace(&workspace).await.unwrap();

    let milestone = WorkspaceMilestoneNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        title: "WM1".to_string(),
        description: None,
        status: MilestoneStatus::Planned,
        target_date: None,
        closed_at: None,
        created_at: chrono::Utc::now(),
        tags: vec!["test".to_string()],
    };
    store.create_workspace_milestone(&milestone).await.unwrap();

    // --- happy path: a real id reports true and the change lands ---
    let matched = store
        .update_workspace_milestone(
            milestone.id,
            Some("WM1 renamed".to_string()),
            None,
            Some(MilestoneStatus::InProgress),
            None,
        )
        .await
        .unwrap();
    assert!(matched, "an existing milestone reports matched = true");
    let updated = store
        .get_workspace_milestone(milestone.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(updated.title, "WM1 renamed");
    assert_eq!(updated.status, MilestoneStatus::InProgress);

    // --- the case that used to lie: unknown id ---
    let unknown = Uuid::new_v4();
    let matched = store
        .update_workspace_milestone(unknown, Some("ghost".to_string()), None, None, None)
        .await
        .expect("an unknown id is a normal answer, not an error");
    assert!(
        !matched,
        "an unknown id must report matched = false so the API can answer 404"
    );
    assert!(
        store
            .get_workspace_milestone(unknown)
            .await
            .unwrap()
            .is_none(),
        "and it must not have created the milestone on the way"
    );

    // --- empty update: nothing to SET, but existence is still reported ---
    let matched = store
        .update_workspace_milestone(milestone.id, None, None, None, None)
        .await
        .unwrap();
    assert!(
        matched,
        "an empty update on an existing milestone still reports true"
    );
    let matched = store
        .update_workspace_milestone(Uuid::new_v4(), None, None, None, None)
        .await
        .unwrap();
    assert!(
        !matched,
        "an empty update on an unknown id reports false — this is the path \
         that returned Ok(()) for everything before the fix"
    );
    assert_eq!(
        store
            .get_workspace_milestone(milestone.id)
            .await
            .unwrap()
            .unwrap()
            .title,
        "WM1 renamed",
        "the empty update changed nothing"
    );

    // --- cleanup ---
    store
        .delete_workspace_milestone(milestone.id)
        .await
        .unwrap();
    assert!(store
        .get_workspace_milestone(milestone.id)
        .await
        .unwrap()
        .is_none());
    store.delete_workspace(workspace.id).await.unwrap();
}

/// MCP tokens end to end against a real graph: issue, check, revoke, inventory,
/// purge.
///
/// Why this one earns its place beyond the coverage number: `purge_mcp_tokens`
/// is the only call in this family that **deletes rows**, and the sole reason
/// deleting is safe is that `is_mcp_token_active` answers `false` for a `jti`
/// it cannot find. The unit tests can only pin that in `MockGraphStore`; the
/// assertion after the purge below is what keeps it true in the Cypher. If a
/// future rewrite made an unknown `jti` return `true`, a purge would silently
/// re-enable every revoked token — and this test is what fails first.
#[tokio::test]
async fn test_mcp_token_lifecycle_revocation_and_purge() {
    let client = store_or_skip!();
    let store: &dyn GraphStore = &client;

    let user = Uuid::new_v4();
    let other = Uuid::new_v4();
    let future = chrono::Utc::now() + chrono::Duration::days(30);
    let past = chrono::Utc::now() - chrono::Duration::days(1);

    let live = format!("jti-live-{}", Uuid::new_v4());
    let revoked = format!("jti-revoked-{}", Uuid::new_v4());
    let expired = format!("jti-expired-{}", Uuid::new_v4());
    let foreign = format!("jti-foreign-{}", Uuid::new_v4());
    let never_issued = format!("jti-never-issued-{}", Uuid::new_v4());

    // --- happy path: issue, then ask whether each one still grants access ---
    store
        .create_mcp_token(user, &live, "working", "mcp:read", future)
        .await
        .unwrap();
    store
        .create_mcp_token(user, &revoked, "to revoke", "mcp:read mcp:write", future)
        .await
        .unwrap();
    store
        .create_mcp_token(user, &expired, "lapsed", "mcp:read", past)
        .await
        .unwrap();
    store
        .create_mcp_token(other, &foreign, "another user's", "mcp:read", future)
        .await
        .unwrap();

    assert!(store.is_mcp_token_active(&live).await.unwrap());
    assert!(
        !store.is_mcp_token_active(&expired).await.unwrap(),
        "expiry alone must reject, without any revocation"
    );
    assert!(
        !store.is_mcp_token_active(&never_issued).await.unwrap(),
        "an unknown jti must be rejected — this is the fail-closed arm the \
         purge below depends on"
    );

    // --- revocation, and what it reports ---
    assert!(store.revoke_mcp_token(user, &revoked).await.unwrap());
    assert!(!store.is_mcp_token_active(&revoked).await.unwrap());
    assert!(
        store.revoke_mcp_token(user, &revoked).await.unwrap(),
        "revoking twice is idempotent: the row is still there, so it matched"
    );
    assert!(
        !store.revoke_mcp_token(user, &never_issued).await.unwrap(),
        "an unknown jti reports NO match, so the API can answer 404 instead of \
         telling the operator a token was revoked when none was"
    );
    assert!(
        !store.revoke_mcp_token(user, &foreign).await.unwrap(),
        "another user's token is not ours to revoke"
    );
    assert!(
        store.is_mcp_token_active(&foreign).await.unwrap(),
        "and it is left working"
    );

    // --- the inventory deliberately keeps the dead ones: that is the audit
    //     trail, and the reason a separate purge has to exist at all ---
    assert_eq!(store.list_mcp_tokens(user).await.unwrap().len(), 3);

    // --- purge ---
    assert_eq!(
        store.purge_mcp_tokens(user).await.unwrap(),
        2,
        "the revoked one and the expired one, not the working one"
    );
    let left = store.list_mcp_tokens(user).await.unwrap();
    assert_eq!(left.len(), 1);
    assert_eq!(left[0].jti, live);
    assert!(
        store.is_mcp_token_active(&live).await.unwrap(),
        "a purge must never disable a token that still works"
    );
    assert!(
        !store.is_mcp_token_active(&revoked).await.unwrap(),
        "a purged row stays rejected — deleting it must not resurrect the token"
    );
    assert_eq!(
        store.purge_mcp_tokens(user).await.unwrap(),
        0,
        "a second purge has nothing left to do"
    );
    assert_eq!(
        store.list_mcp_tokens(other).await.unwrap().len(),
        1,
        "the purge never reached the other user's records"
    );

    // --- cleanup: leave the database as we found it ---
    store.revoke_mcp_token(user, &live).await.unwrap();
    store.purge_mcp_tokens(user).await.unwrap();
    store.revoke_mcp_token(other, &foreign).await.unwrap();
    store.purge_mcp_tokens(other).await.unwrap();
    assert!(store.list_mcp_tokens(user).await.unwrap().is_empty());
    assert!(store.list_mcp_tokens(other).await.unwrap().is_empty());
}

// ===========================================================================
// src/neo4j/segment.rs — the conversation segment query layer
// ===========================================================================

/// A two-turn harness switch: the first turn opens a segment on the harness
/// axis, the second (a different harness) confirms a boundary at once, so the
/// segmenter yields two segments and one boundary.
fn a_harness_switch() -> (Vec<Segment>, Vec<Boundary>) {
    let history = [
        Observation {
            turn: 0,
            prints: [(Dimension::Harness, Fingerprint::new(["deepseek"]))]
                .into_iter()
                .collect(),
        },
        Observation {
            turn: 1,
            prints: [(Dimension::Harness, Fingerprint::new(["claude-code"]))]
                .into_iter()
                .collect(),
        },
    ];
    let segmenter = Segmenter::replay(SegmentPolicy::default(), 7, &history).unwrap();
    (segmenter.segments(), segmenter.boundaries().to_vec())
}

#[tokio::test]
async fn test_segment_store_round_trips_and_replaces() {
    let client = store_or_skip!();
    let store: &dyn GraphStore = &client;

    let conversation = Uuid::new_v4().to_string();
    let (segments, boundaries) = a_harness_switch();
    assert!(!segments.is_empty(), "the fixture must open a segment");
    assert!(!boundaries.is_empty(), "the fixture must cut a boundary");

    // --- an unknown conversation reads as empty, never as an error ---
    assert!(store.get_segments(&conversation).await.unwrap().is_empty());
    assert!(store
        .get_boundaries(&conversation)
        .await
        .unwrap()
        .is_empty());

    // --- happy path: segments and boundaries round-trip ---
    store.put_segments(&conversation, &segments).await.unwrap();
    assert_eq!(store.get_segments(&conversation).await.unwrap(), segments);
    store
        .put_boundaries(&conversation, &boundaries)
        .await
        .unwrap();
    assert_eq!(
        store.get_boundaries(&conversation).await.unwrap(),
        boundaries
    );

    // --- another conversation stays empty ---
    let other = Uuid::new_v4().to_string();
    assert!(store.get_segments(&other).await.unwrap().is_empty());
    assert!(store.get_boundaries(&other).await.unwrap().is_empty());

    // --- a second put replaces, never accumulates ---
    store
        .put_segments(&conversation, &segments[..1])
        .await
        .unwrap();
    assert_eq!(
        store.get_segments(&conversation).await.unwrap(),
        segments[..1].to_vec(),
        "putting a subset must replace the stored set, not append to it"
    );
    store.put_boundaries(&conversation, &[]).await.unwrap();
    assert!(
        store
            .get_boundaries(&conversation)
            .await
            .unwrap()
            .is_empty(),
        "putting no boundaries clears the stored ones"
    );

    // --- cleanup: clear what this test wrote ---
    store.put_segments(&conversation, &[]).await.unwrap();
    store.put_boundaries(&conversation, &[]).await.unwrap();
    assert!(store.get_segments(&conversation).await.unwrap().is_empty());
    assert!(store
        .get_boundaries(&conversation)
        .await
        .unwrap()
        .is_empty());
}

// ===========================================================================
// src/neo4j/chat.rs
// ===========================================================================
//
// Excluded from coverage until B19 with "needs the fake CLI of task 3.2": a
// session is a Neo4j node, and these calls need no CLI, fake or real, to write
// and read one. The anchors, the provider settings and the spawned-by edge are
// exercised by `tests/harness_neo4j.rs` (also in the coverage pass); what
// follows covers the rest: the session fields, the event log, the lineage,
// the plan/task links and the DISCUSSED relations.

fn a_chat_session(project_slug: Option<&str>) -> ChatSessionNode {
    ChatSessionNode {
        id: Uuid::new_v4(),
        cli_session_id: None,
        project_slug: project_slug.map(str::to_string),
        workspace_slug: None,
        cwd: "/tmp/neo4j-store-chat".to_string(),
        title: None,
        model: "test-model".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        message_count: 0,
        total_cost_usd: None,
        conversation_id: None,
        preview: None,
        permission_mode: None,
        add_dirs: None,
        spawned_by: None,
        provider_id: None,
        routed_by: None,
        capabilities: None,
        engine: None,
        resume_token: None,
        execution_place: Default::default(),
        project_inferred: false,
        access: Default::default(),
        owner: None,
        file_scope_lifted: false,
        mirror_of: None,
        routing_pool: None,
        routing_mode: None,
    }
}

fn an_event(
    session_id: Uuid,
    seq: i64,
    event_type: &str,
    data: serde_json::Value,
) -> ChatEventRecord {
    ChatEventRecord {
        id: Uuid::new_v4(),
        session_id,
        seq,
        event_type: event_type.to_string(),
        data: data.to_string(),
        created_at: chrono::Utc::now(),
    }
}

#[tokio::test]
async fn test_chat_session_fields_round_trip_and_unknown_ids_match_nothing() {
    let client = store_or_skip!();
    let store: &dyn GraphStore = &client;

    let session = a_chat_session(None);
    store.create_chat_session(&session).await.unwrap();

    // --- the partial update sets what it is given, and only that ---
    let updated = store
        .update_chat_session(
            session.id,
            Some("cli-1".to_string()),
            Some("A title".to_string()),
            Some(3),
            Some(0.5),
            Some("conv-1".to_string()),
            Some("a preview".to_string()),
        )
        .await
        .unwrap()
        .expect("the session exists");
    assert_eq!(updated.cli_session_id.as_deref(), Some("cli-1"));
    assert_eq!(updated.title.as_deref(), Some("A title"));
    assert_eq!(updated.message_count, 3);
    assert_eq!(updated.total_cost_usd, Some(0.5));
    assert_eq!(updated.conversation_id.as_deref(), Some("conv-1"));
    assert_eq!(updated.preview.as_deref(), Some("a preview"));
    let unchanged = store
        .update_chat_session(session.id, None, None, None, None, None, None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        unchanged.title.as_deref(),
        Some("A title"),
        "an empty update keeps the fields"
    );
    assert!(
        store
            .update_chat_session(
                Uuid::new_v4(),
                None,
                Some("x".into()),
                None,
                None,
                None,
                None
            )
            .await
            .unwrap()
            .is_none(),
        "an unknown id is reported, never created"
    );

    // --- the per-field setters ---
    store
        .update_chat_session_permission_mode(session.id, "plan")
        .await
        .unwrap();
    store
        .update_chat_session_file_scope_lifted(session.id, true)
        .await
        .unwrap();
    store
        .update_chat_session_model(session.id, "other-model")
        .await
        .unwrap();
    let read = store.get_chat_session(session.id).await.unwrap().unwrap();
    assert_eq!(read.permission_mode.as_deref(), Some("plan"));
    assert!(read.file_scope_lifted);
    assert_eq!(read.model, "other-model");

    // --- routing: replaced, an absent value reads back as None ---
    assert!(store
        .update_chat_session_routing(session.id, Some("full"), Some("[\"a\"]"), Some("user"))
        .await
        .unwrap());
    let read = store.get_chat_session(session.id).await.unwrap().unwrap();
    assert_eq!(read.routing_mode.as_deref(), Some("full"));
    assert_eq!(read.routing_pool.as_deref(), Some("[\"a\"]"));
    assert_eq!(read.routed_by.as_deref(), Some("user"));
    assert!(store
        .update_chat_session_routing(session.id, None, None, None)
        .await
        .unwrap());
    let read = store.get_chat_session(session.id).await.unwrap().unwrap();
    assert_eq!(read.routing_mode, None);
    assert_eq!(read.routing_pool, None);
    assert!(!store
        .update_chat_session_routing(Uuid::new_v4(), Some("full"), None, None)
        .await
        .unwrap());

    // --- provider re-target clears the provider-specific fields; restore puts them back ---
    store
        .update_chat_session_harness(session.id, Some("{\"tools\":true}"), Some("token"))
        .await
        .unwrap();
    let before = store.get_chat_session(session.id).await.unwrap().unwrap();
    assert!(store
        .update_chat_session_provider(session.id, "openai", "gpt", "fallback", "agent")
        .await
        .unwrap());
    let moved = store.get_chat_session(session.id).await.unwrap().unwrap();
    assert_eq!(moved.provider_id.as_deref(), Some("openai"));
    assert_eq!(moved.model, "gpt");
    assert_eq!(moved.engine.as_deref(), Some("agent"));
    assert_eq!(moved.resume_token, None, "a token never crosses providers");
    assert_eq!(moved.capabilities, None);
    assert_eq!(moved.cli_session_id, None);
    assert!(store
        .restore_chat_session_provider(&before, "local")
        .await
        .unwrap());
    let restored = store.get_chat_session(session.id).await.unwrap().unwrap();
    assert_eq!(restored.provider_id.as_deref(), Some("local"));
    assert_eq!(restored.model, before.model);
    assert_eq!(restored.resume_token, before.resume_token);
    assert_eq!(restored.cli_session_id, before.cli_session_id);
    assert!(!store
        .update_chat_session_provider(Uuid::new_v4(), "p", "m", "r", "e")
        .await
        .unwrap());
    let mut ghost = before.clone();
    ghost.id = Uuid::new_v4();
    assert!(!store
        .restore_chat_session_provider(&ghost, "p")
        .await
        .unwrap());

    // --- shell moved: a new session has not moved, set is Some(true), unknown is None ---
    assert_eq!(
        store
            .get_chat_session_shell_moved(session.id)
            .await
            .unwrap(),
        Some(false)
    );
    store
        .set_chat_session_shell_moved(session.id)
        .await
        .unwrap();
    assert_eq!(
        store
            .get_chat_session_shell_moved(session.id)
            .await
            .unwrap(),
        Some(true)
    );
    assert_eq!(
        store
            .get_chat_session_shell_moved(Uuid::new_v4())
            .await
            .unwrap(),
        None
    );

    // --- delete reports whether something was there ---
    assert!(store.delete_chat_session(session.id).await.unwrap());
    assert!(!store.delete_chat_session(session.id).await.unwrap());
    assert!(store.get_chat_session(session.id).await.unwrap().is_none());
}

#[tokio::test]
async fn test_chat_events_replay_paginate_count_and_delete() {
    let client = store_or_skip!();
    let store: &dyn GraphStore = &client;

    let session = a_chat_session(None);
    store.create_chat_session(&session).await.unwrap();
    let sid = session.id;

    // --- empty log ---
    assert_eq!(store.count_chat_events(sid).await.unwrap(), 0);
    assert_eq!(store.get_latest_chat_event_seq(sid).await.unwrap(), 0);
    store.store_chat_events(sid, vec![]).await.unwrap();
    assert!(store.get_attention_events(&[]).await.unwrap().is_empty());

    store
        .store_chat_events(
            sid,
            vec![
                an_event(
                    sid,
                    1,
                    "user_message",
                    serde_json::json!({"content": "Bonjour, regarde src/main.rs"}),
                ),
                an_event(
                    sid,
                    2,
                    "assistant_text",
                    serde_json::json!({"content": "ok"}),
                ),
                an_event(
                    sid,
                    3,
                    "permission_request",
                    serde_json::json!({"id": "req-1"}),
                ),
                an_event(
                    sid,
                    4,
                    "permission_decision",
                    serde_json::json!({"id": "req-1", "allow": true}),
                ),
            ],
        )
        .await
        .unwrap();

    // --- replay after a seq, in order, bounded ---
    let after: Vec<i64> = store
        .get_chat_events(sid, 1, 2)
        .await
        .unwrap()
        .iter()
        .map(|e| e.seq)
        .collect();
    assert_eq!(after, vec![2, 3]);
    let page: Vec<i64> = store
        .get_chat_events_paginated(sid, 1, 10)
        .await
        .unwrap()
        .iter()
        .map(|e| e.seq)
        .collect();
    assert_eq!(page, vec![2, 3, 4]);
    assert_eq!(store.count_chat_events(sid).await.unwrap(), 4);
    assert_eq!(store.get_latest_chat_event_seq(sid).await.unwrap(), 4);

    // --- attention: only the kinds that matter, the user text blanked ---
    let attention = store.get_attention_events(&[sid]).await.unwrap();
    let kinds: Vec<&str> = attention.iter().map(|e| e.event_type.as_str()).collect();
    assert_eq!(
        kinds,
        vec!["user_message", "permission_request", "permission_decision"]
    );
    assert_eq!(attention[0].data, "", "the user's words are not read back");
    assert!(attention[1].data.contains("req-1"));

    // --- previews are backfilled from the first user message ---
    assert!(store.backfill_chat_session_previews().await.unwrap() >= 1);
    let read = store.get_chat_session(sid).await.unwrap().unwrap();
    assert!(
        read.preview
            .as_deref()
            .is_some_and(|p| p.contains("Bonjour")),
        "{:?}",
        read.preview
    );
    assert!(read.title.is_some());

    // --- delete the log, the session stays ---
    store.delete_chat_events(sid).await.unwrap();
    assert_eq!(store.count_chat_events(sid).await.unwrap(), 0);
    assert!(store.get_chat_events(sid, 0, 10).await.unwrap().is_empty());
    assert!(store.get_chat_session(sid).await.unwrap().is_some());

    // --- unknown session: nothing, no error ---
    let unknown = Uuid::new_v4();
    assert_eq!(store.count_chat_events(unknown).await.unwrap(), 0);
    store
        .store_chat_events(
            unknown,
            vec![an_event(unknown, 1, "user_message", serde_json::json!({}))],
        )
        .await
        .unwrap();
    assert_eq!(
        store.count_chat_events(unknown).await.unwrap(),
        0,
        "an event is never written outside a session"
    );

    store.delete_chat_session(sid).await.unwrap();
}

#[tokio::test]
async fn test_chat_lineage_runs_and_plan_task_links() {
    let client = store_or_skip!();
    let store: &dyn GraphStore = &client;

    let plan = PlanNode::new(
        format!("Chat plan {}", Uuid::new_v4()),
        "plan a conversation is linked to".to_string(),
        "tester".to_string(),
        1,
    );
    store.create_plan(&plan).await.unwrap();
    let task = TaskNode::new("chat task".to_string());
    store.create_task(plan.id, &task).await.unwrap();
    let run_id = Uuid::new_v4();
    store
        .create_plan_run(&project_orchestrator::runner::RunnerState::new(
            run_id,
            plan.id,
            1,
            Default::default(),
        ))
        .await
        .unwrap();

    let root = a_chat_session(None);
    let child = a_chat_session(None);
    let grandchild = a_chat_session(None);
    for s in [&root, &child, &grandchild] {
        store.create_chat_session(s).await.unwrap();
    }
    let (r, c, g) = (
        root.id.to_string(),
        child.id.to_string(),
        grandchild.id.to_string(),
    );
    store
        .create_spawned_by_relation(&c, &r, "delegate", Some(run_id), Some(task.id))
        .await
        .unwrap();
    store
        .create_spawned_by_relation(&g, &c, "delegate", None, None)
        .await
        .unwrap();

    // --- the tree, the root, the children ---
    let tree = store.get_session_tree(&r).await.unwrap();
    let depths: Vec<(String, u32)> = tree
        .iter()
        .map(|n| (n.session_id.clone(), n.depth))
        .collect();
    assert_eq!(depths, vec![(r.clone(), 0), (c.clone(), 1), (g.clone(), 2)]);
    assert_eq!(tree[1].parent_session_id.as_deref(), Some(r.as_str()));
    assert_eq!(tree[1].run_id, Some(run_id));
    assert_eq!(tree[1].task_id, Some(task.id));
    assert_eq!(tree[1].spawn_type.as_deref(), Some("delegate"));
    assert_eq!(
        store.get_session_root(&g).await.unwrap().as_deref(),
        Some(r.as_str())
    );
    assert_eq!(
        store.get_session_root(&r).await.unwrap().as_deref(),
        Some(r.as_str())
    );
    assert_eq!(
        store
            .get_session_root(&Uuid::new_v4().to_string())
            .await
            .unwrap(),
        None
    );
    let children: Vec<Uuid> = store
        .get_session_children(root.id)
        .await
        .unwrap()
        .iter()
        .map(|s| s.id)
        .collect();
    assert_eq!(children, vec![child.id]);

    // --- the sessions of a run ---
    let run_sessions = store.get_run_sessions(run_id).await.unwrap();
    assert_eq!(run_sessions.len(), 1);
    assert_eq!(run_sessions[0].session_id, c);
    assert_eq!(run_sessions[0].task_id, Some(task.id));

    // --- a session linked to its run, and the link rows ---
    assert!(store
        .link_session_to_run(&r, run_id, Some(plan.id), Some(task.id))
        .await
        .unwrap());
    assert!(
        !store
            .link_session_to_run(&r, Uuid::new_v4(), None, None)
            .await
            .unwrap(),
        "an unknown run links nothing"
    );
    assert!(store
        .create_associated_with(grandchild.id, "Plan", plan.id, "manual")
        .await
        .unwrap());
    assert!(store
        .create_associated_with(grandchild.id, "Task", task.id, "chat")
        .await
        .unwrap());
    assert!(store
        .create_associated_with(grandchild.id, "Milestone", plan.id, "manual")
        .await
        .is_err());
    assert!(
        !store
            .create_associated_with(grandchild.id, "Plan", Uuid::new_v4(), "manual")
            .await
            .unwrap(),
        "an unknown plan links nothing"
    );
    let rows = store
        .get_session_link_rows(&[root.id, child.id, grandchild.id])
        .await
        .unwrap();
    let kinds_of = |id: Uuid| -> Vec<SessionLinkKind> {
        let mut k: Vec<_> = rows
            .iter()
            .filter(|r| r.session_id == id)
            .map(|r| r.kind)
            .collect();
        k.sort_by_key(|k| format!("{k:?}"));
        k
    };
    assert_eq!(kinds_of(root.id), vec![SessionLinkKind::RunRelation]);
    assert_eq!(kinds_of(child.id), vec![SessionLinkKind::SpawnedByRelation]);
    assert_eq!(
        kinds_of(grandchild.id),
        vec![
            SessionLinkKind::PlanAssociation,
            SessionLinkKind::SpawnedByRelation,
            SessionLinkKind::TaskAssociation,
        ]
    );
    let run_row = rows.iter().find(|r| r.session_id == root.id).unwrap();
    assert_eq!(run_row.run_id, Some(run_id));
    assert_eq!(run_row.plan_id, Some(plan.id));
    assert!(store.get_session_link_rows(&[]).await.unwrap().is_empty());

    // --- the links of a session, one and many ---
    let links = store.get_session_links(grandchild.id).await.unwrap();
    assert_eq!(links.linked_tasks.len(), 1);
    assert_eq!(links.linked_tasks[0].id, task.id);
    assert_eq!(links.linked_tasks[0].source, "chat");
    assert!(links
        .linked_plans
        .iter()
        .any(|p| p.id == plan.id && p.source == "manual"));
    let batch = store
        .get_session_links_batch(&[grandchild.id, root.id])
        .await
        .unwrap();
    assert_eq!(batch[&grandchild.id].linked_tasks.len(), 1);
    assert!(batch
        .get(&root.id)
        .is_none_or(|l| l.linked_tasks.is_empty() && l.linked_plans.is_empty()));
    assert!(store.get_session_links_batch(&[]).await.unwrap().is_empty());

    // --- and the reverse: the sessions of a plan, of a task ---
    let for_plan = store.get_sessions_for_plan(plan.id).await.unwrap();
    assert!(for_plan.iter().any(|s| s.session.id == grandchild.id));
    let for_task = store.get_sessions_for_task(task.id).await.unwrap();
    assert_eq!(for_task.len(), 1);
    assert_eq!(for_task[0].session.id, grandchild.id);
    assert_eq!(for_task[0].source, "chat");
    assert_eq!(
        store.get_task_plan_id(task.id).await.unwrap(),
        Some(plan.id)
    );
    assert_eq!(store.get_task_plan_id(Uuid::new_v4()).await.unwrap(), None);

    // --- cleanup ---
    for s in [&root, &child, &grandchild] {
        store.delete_chat_session(s.id).await.unwrap();
    }
    store.delete_plan(plan.id).await.unwrap();
}

#[tokio::test]
async fn test_chat_discussed_entities_and_graph_data() {
    let client = store_or_skip!();
    let store: &dyn GraphStore = &client;

    let project = a_project();
    store.create_project(&project).await.unwrap();
    let path = format!("/tmp/n4j-chat-{}/src/lib.rs", Uuid::new_v4());
    store
        .upsert_file(&FileNode {
            path: path.clone(),
            language: "rust".to_string(),
            hash: "h".to_string(),
            last_parsed: chrono::Utc::now(),
            project_id: Some(project.id),
        })
        .await
        .unwrap();
    let session = a_chat_session(Some(&project.slug));
    store.create_chat_session(&session).await.unwrap();

    // --- nothing to link, an unknown kind, an unknown file ---
    assert_eq!(store.add_discussed(session.id, &[]).await.unwrap(), 0);
    assert_eq!(
        store
            .add_discussed(
                session.id,
                &[
                    ("Galaxy".to_string(), "x".to_string()),
                    ("File".to_string(), "/no/such/file.rs".to_string()),
                ],
            )
            .await
            .unwrap(),
        0
    );

    // --- a mention, then a second one: one relation, counted twice ---
    let mention = [("File".to_string(), path.clone())];
    assert_eq!(store.add_discussed(session.id, &mention).await.unwrap(), 1);
    assert_eq!(store.add_discussed(session.id, &mention).await.unwrap(), 1);
    let entities = store
        .get_session_entities(session.id, Some(project.id))
        .await
        .unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].entity_type, "File");
    assert_eq!(entities[0].entity_id, path);
    assert_eq!(entities[0].mention_count, 2);
    assert!(entities[0].last_mentioned_at.is_some());
    assert_eq!(
        store
            .get_session_entities(session.id, None)
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(
        store
            .get_session_entities(session.id, Some(Uuid::new_v4()))
            .await
            .unwrap()
            .is_empty(),
        "another project's view does not see this file"
    );

    // --- the chat layer of the project graph ---
    let (sessions, discussed) = store.get_chat_graph_data(project.id, 10).await.unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, session.id.to_string());
    assert_eq!(discussed.len(), 1);
    let (none, _) = store.get_chat_graph_data(Uuid::new_v4(), 10).await.unwrap();
    assert!(none.is_empty());

    // --- no co-change history: no prediction, no error ---
    assert!(store
        .get_discussed_co_changers(project.id, 5, 10)
        .await
        .unwrap()
        .is_empty());

    // --- cleanup ---
    store.delete_chat_session(session.id).await.unwrap();
    store
        .delete_project(project.id, &project.name)
        .await
        .unwrap();
}
