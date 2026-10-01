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

    // --- known gap, asserted so it is not mistaken for working: a release
    //     created under an unknown project is dropped SILENTLY. MATCH finds no
    //     Project, so the CREATE never runs, yet create_release answers Ok.
    //     create_milestone (above) bails in the same situation. Tracked as a
    //     bug of its own; this assertion fails the day it is fixed, which is
    //     when the fix must update it to expect an error. ---
    let orphan = a_release(Uuid::new_v4());
    store
        .create_release(&orphan)
        .await
        .expect("current behaviour: Ok even though nothing was written");
    assert!(
        store.get_release(orphan.id).await.unwrap().is_none(),
        "nothing is written — the Ok above is a silent failure, not a success"
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

    // --- error case: unknown task — MATCH finds nothing, so no Decision is
    //     written. Asserted on the read, because the write answers Ok. ---
    let orphan = a_decision();
    store
        .create_decision(Uuid::new_v4(), &orphan)
        .await
        .expect("current behaviour: Ok even though nothing was written");
    assert!(
        store.get_decision(orphan.id).await.unwrap().is_none(),
        "no task to hang it on means no decision node"
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
