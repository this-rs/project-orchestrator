//! The UNSCOPED anchor sweeps — `backfill_project_anchors`, `revert_inferred_anchors`,
//! `backfill_spawned_by_edges` — through the `GraphStore` trait, against a real Neo4j.
//!
//! One test, alone in its own binary, on purpose. An unscoped sweep writes to every session
//! of the database, so it cannot share a binary with tests that run in parallel on the same
//! database: `harness_neo4j` measured that interference (CI, 11/11/2026) and its tests use
//! the scoped `*_for(Some(..))` variants since. Cargo runs test binaries one after another,
//! so this one sees a quiet database. Only what this test created is asserted on; whatever
//! another suite left behind is swept too — that IS the unscoped behaviour — and is counted
//! but not checked.
//!
//! Like `harness_neo4j`, the test prints a notice and returns when no Neo4j is reachable;
//! `HARNESS_NEO4J_REQUIRED=1` turns that into a failure. Local run:
//!   NEO4J_URI=bolt://localhost:17687 NEO4J_PASSWORD=testpassword \
//!     cargo test --test anchor_sweeps_unscoped
mod common;

#[allow(unused_imports)]
use project_orchestrator::neo4j::traits::*;

use chrono::Utc;
use neo4rs::{query, Graph};
use project_orchestrator::chat::anchor::AnchorActor;
use project_orchestrator::chat::types::SpawnedBy;
use project_orchestrator::neo4j::client::Neo4jClient;
use project_orchestrator::neo4j::models::{ChatSessionNode, ProjectNode};
use project_orchestrator::neo4j::GraphStore;
use uuid::Uuid;

async fn env() -> Option<(Neo4jClient, Graph)> {
    let common::harness::Neo4jTarget {
        uri,
        user,
        password,
    } = common::harness::neo4j_target();
    let connected = async {
        let client = Neo4jClient::new(&uri, &user, &password).await.ok()?;
        let raw = Graph::new(&uri, &user, &password).await.ok()?;
        Some((client, raw))
    }
    .await;
    if connected.is_none() {
        assert!(
            std::env::var("HARNESS_NEO4J_REQUIRED").is_err(),
            "HARNESS_NEO4J_REQUIRED is set but no Neo4j answers at {uri}"
        );
        eprintln!("Skipping anchor_sweeps_unscoped: no Neo4j at {uri}");
    }
    connected
}

fn session(id: Uuid, project_slug: Option<String>, spawned_by: Option<String>) -> ChatSessionNode {
    ChatSessionNode {
        file_scope_lifted: false,
        mirror_of: None,
        routing_pool: None,
        routing_mode: None,
        id,
        cli_session_id: None,
        project_slug,
        workspace_slug: None,
        cwd: "/tmp/anchor-sweeps-unscoped".into(),
        title: None,
        model: "m".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        message_count: 0,
        total_cost_usd: None,
        conversation_id: None,
        preview: None,
        permission_mode: None,
        add_dirs: None,
        spawned_by,
        provider_id: Some("local".into()),
        routed_by: Some("request".into()),
        capabilities: None,
        engine: None,
        resume_token: None,
        execution_place: Default::default(),
        project_inferred: false,
        access: Default::default(),
        owner: None,
    }
}

async fn edge_count(raw: &Graph, child: Uuid, parent: Uuid) -> i64 {
    let mut r = raw
        .execute(
            query(
                "MATCH (:ChatSession {id: $c})-[r:SPAWNED_BY]->(:ChatSession {id: $p}) \
                 RETURN count(r) AS n",
            )
            .param("c", child.to_string())
            .param("p", parent.to_string()),
        )
        .await
        .unwrap();
    r.next().await.unwrap().unwrap().get("n").unwrap()
}

/// The three sweeps as the operator runs them: no scope, through the trait. Each one
/// finds what this test created, twice in a row gives the same state, and the revert
/// takes back exactly what the backfill inferred.
#[tokio::test]
async fn the_unscoped_sweeps_find_this_tests_data_through_the_trait() {
    let Some((client, raw)) = env().await else {
        return;
    };
    let store: &dyn GraphStore = &client;

    // a project, a session carrying its slug (the backfill's input), a parent and a
    // legacy child (JSON lineage, edge dropped: written before the edge existed)
    let slug = format!("sweeps-unscoped-{}", Uuid::new_v4().simple());
    let project = ProjectNode {
        id: Uuid::new_v4(),
        name: slug.clone(),
        slug: slug.clone(),
        root_path: "/tmp/anchor-sweeps-unscoped".to_string(),
        description: None,
        created_at: Utc::now(),
        last_synced: None,
        analytics_computed_at: None,
        last_co_change_computed_at: None,
        default_note_energy: None,
        scaffolding_override: None,
        sharing_policy: None,
        watch_enabled: false,
        profile: Default::default(),
    };
    store.create_project(&project).await.unwrap();
    let sid = Uuid::new_v4();
    store
        .create_chat_session(&session(sid, Some(slug.clone()), None))
        .await
        .unwrap();
    let (parent, child) = (Uuid::new_v4(), Uuid::new_v4());
    store
        .create_chat_session(&session(parent, None, None))
        .await
        .unwrap();
    let json = SpawnedBy::Conversation {
        parent_session_id: parent,
        tool_use_id: None,
    }
    .to_json_string();
    store
        .create_chat_session(&session(child, None, Some(json)))
        .await
        .unwrap();
    raw.run(
        query("MATCH (:ChatSession {id: $c})-[r:SPAWNED_BY]->() DELETE r")
            .param("c", child.to_string()),
    )
    .await
    .unwrap();
    assert_eq!(edge_count(&raw, child, parent).await, 0);

    // (1) project anchors: inferred, by the system, on the project; idempotent
    let first = store.backfill_project_anchors().await.unwrap();
    assert!(first.created >= 1, "{first:?}");
    let anchors = store.list_session_anchors(sid).await.unwrap();
    assert_eq!(anchors.len(), 1);
    assert!(anchors[0].inferred && anchors[0].by == AnchorActor::System);
    assert_eq!(anchors[0].target_id, project.id.to_string());
    let second = store.backfill_project_anchors().await.unwrap();
    assert!(second.already_present >= 1, "{second:?}");
    assert_eq!(store.list_session_anchors(sid).await.unwrap(), anchors);
    assert_eq!(store.list_anchor_events(sid).await.unwrap().len(), 1);

    // (2) lineage edges: the legacy child gets its edge, once
    assert!(store.backfill_spawned_by_edges().await.unwrap() >= 1);
    assert_eq!(edge_count(&raw, child, parent).await, 1);
    store.backfill_spawned_by_edges().await.unwrap();
    assert_eq!(edge_count(&raw, child, parent).await, 1);
    let kids = store.get_session_children(parent).await.unwrap();
    assert_eq!(kids.iter().map(|s| s.id).collect::<Vec<_>>(), vec![child]);

    // (3) revert: the inferred anchor and its journal go, nothing else of the session
    assert!(store.revert_inferred_anchors().await.unwrap() >= 1);
    assert!(store.list_session_anchors(sid).await.unwrap().is_empty());
    assert!(store.list_anchor_events(sid).await.unwrap().is_empty());
    assert!(store.get_chat_session(sid).await.unwrap().is_some());

    for id in [child, parent, sid] {
        store.delete_chat_session(id).await.unwrap();
    }
    store.delete_project(project.id, &slug).await.unwrap();
}
