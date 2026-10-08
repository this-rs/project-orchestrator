//! The Cypher of the provider harness against a real Neo4j: the `LlmSetting`
//! documents (instances, consents, roles, aliases, policy, journal) and the
//! provider fields of a `ChatSession`.
//!
//! Every scenario is scoped to a fresh scope / session id and cleans up after
//! itself, so it can share a throwaway instance with other suites. NEVER point
//! it at a live database. Run locally against a throwaway instance:
//!   docker run -d --rm --name po-mig -p 17687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5
//!   NEO4J_URI=bolt://localhost:17687 NEO4J_PASSWORD=testpassword cargo test --test harness_neo4j
//!
//! Like `data_migrations`, the tests print a notice and return when no Neo4j is
//! reachable; `HARNESS_NEO4J_REQUIRED=1` turns that into a failure.

use chrono::Utc;
use neo4rs::{query, Graph};
use project_orchestrator::neo4j::client::Neo4jClient;
use project_orchestrator::neo4j::models::ChatSessionNode;
use project_orchestrator::neo4j::GraphStore;
use uuid::Uuid;

struct Env {
    client: Neo4jClient,
    raw: Graph,
}

async fn env() -> Option<Env> {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:17687".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into());
    let connected = async {
        let client = Neo4jClient::new(&uri, &user, &password).await.ok()?;
        let raw = Graph::new(&uri, &user, &password).await.ok()?;
        Some(Env { client, raw })
    }
    .await;
    if connected.is_none() {
        assert!(
            std::env::var("HARNESS_NEO4J_REQUIRED").is_err(),
            "HARNESS_NEO4J_REQUIRED is set but no Neo4j answers at {uri}"
        );
        eprintln!("Skipping harness Neo4j tests: no Neo4j at {uri}");
    }
    connected
}

fn node(id: Uuid) -> ChatSessionNode {
    ChatSessionNode {
        routing_mode: None,
        id,
        cli_session_id: None,
        project_slug: None,
        workspace_slug: None,
        cwd: "/tmp/harness-neo4j-test".into(),
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
        spawned_by: None,
        provider_id: Some("local".into()),
        routed_by: Some("request".into()),
        capabilities: None,
        resume_token: None,
    }
}

#[tokio::test]
async fn llm_settings_put_get_replace_list_and_delete() {
    let Some(e) = env().await else { return };
    let scope = format!("test-scope-{}", Uuid::new_v4());
    let store: &dyn GraphStore = &e.client;

    assert_eq!(
        store.get_llm_setting(&scope, "instance:a").await.unwrap(),
        None
    );
    store
        .put_llm_setting(&scope, "instance:a", r#"{"v":1}"#)
        .await
        .unwrap();
    store
        .put_llm_setting(&scope, "instance:b", r#"{"v":2}"#)
        .await
        .unwrap();
    store
        .put_llm_setting(&scope, "roles", r#"{"pilot":null}"#)
        .await
        .unwrap();
    // Replace, not duplicate.
    store
        .put_llm_setting(&scope, "instance:a", r#"{"v":3}"#)
        .await
        .unwrap();
    assert_eq!(
        store
            .get_llm_setting(&scope, "instance:a")
            .await
            .unwrap()
            .as_deref(),
        Some(r#"{"v":3}"#)
    );
    let mut r = e
        .raw
        .execute(
            query("MATCH (s:LlmSetting {scope: $s, key: 'instance:a'}) RETURN count(s) AS c")
                .param("s", scope.clone()),
        )
        .await
        .unwrap();
    assert_eq!(
        r.next().await.unwrap().unwrap().get::<i64>("c").unwrap(),
        1,
        "MERGE keeps one node"
    );

    // Listing is scoped, prefix-filtered and ordered by key.
    let listed = store.list_llm_settings(&scope, "instance:").await.unwrap();
    assert_eq!(
        listed,
        vec![
            ("instance:a".to_string(), r#"{"v":3}"#.to_string()),
            ("instance:b".to_string(), r#"{"v":2}"#.to_string())
        ]
    );
    let other = format!("test-scope-{}", Uuid::new_v4());
    assert!(store
        .list_llm_settings(&other, "")
        .await
        .unwrap()
        .is_empty());

    // Delete answers whether it removed something.
    assert!(store
        .delete_llm_setting(&scope, "instance:a")
        .await
        .unwrap());
    assert!(!store
        .delete_llm_setting(&scope, "instance:a")
        .await
        .unwrap());
    assert_eq!(
        store.get_llm_setting(&scope, "instance:a").await.unwrap(),
        None
    );
    assert_eq!(store.list_llm_settings(&scope, "").await.unwrap().len(), 2);

    e.raw
        .run(query("MATCH (s:LlmSetting {scope: $s}) DELETE s").param("s", scope))
        .await
        .unwrap();
}

#[tokio::test]
async fn a_value_with_quotes_unicode_and_newlines_round_trips() {
    let Some(e) = env().await else { return };
    let scope = format!("test-scope-{}", Uuid::new_v4());
    let value = "{\"label\":\"D\u{e9}j\u{e0} \\\"vu\\\"\\n\",\"x\":[1,2]}";
    e.client.put_llm_setting(&scope, "k", value).await.unwrap();
    assert_eq!(
        e.client
            .get_llm_setting(&scope, "k")
            .await
            .unwrap()
            .as_deref(),
        Some(value)
    );
    e.client.delete_llm_setting(&scope, "k").await.unwrap();
}

#[tokio::test]
async fn the_provider_fields_of_a_session_are_stored_and_read_back() {
    let Some(e) = env().await else { return };
    let id = Uuid::new_v4();
    let store: &dyn GraphStore = &e.client;
    store.create_chat_session(&node(id)).await.unwrap();

    let read = store.get_chat_session(id).await.unwrap().expect("stored");
    assert_eq!(read.provider_id.as_deref(), Some("local"));
    assert_eq!(read.routed_by.as_deref(), Some("request"));
    assert_eq!(read.capabilities, None, "empty means absent");
    assert_eq!(read.resume_token, None);

    // Setting the capability snapshot leaves the token alone, and vice versa.
    store
        .update_chat_session_harness(id, Some(r#"{"tools":true}"#), None)
        .await
        .unwrap();
    store
        .update_chat_session_harness(id, None, Some(r#"{"k":"native","v":1,"d":{}}"#))
        .await
        .unwrap();
    let read = store.get_chat_session(id).await.unwrap().unwrap();
    assert_eq!(read.capabilities.as_deref(), Some(r#"{"tools":true}"#));
    assert_eq!(
        read.resume_token.as_deref(),
        Some(r#"{"k":"native","v":1,"d":{}}"#)
    );
    // A `None` / `None` call changes neither.
    store
        .update_chat_session_harness(id, None, None)
        .await
        .unwrap();
    let again = store.get_chat_session(id).await.unwrap().unwrap();
    assert_eq!(again.capabilities, read.capabilities);
    assert_eq!(again.resume_token, read.resume_token);

    e.raw
        .run(query("MATCH (s:ChatSession {id: $id}) DETACH DELETE s").param("id", id.to_string()))
        .await
        .unwrap();
}

#[tokio::test]
async fn a_session_written_before_the_harness_reads_back_without_provider_fields() {
    let Some(e) = env().await else { return };
    let id = Uuid::new_v4();
    // The node as the previous version wrote it: no provider property at all.
    e.raw
        .run(
            query(
                "CREATE (s:ChatSession {id: $id, cli_session_id: '', project_slug: '', workspace_slug: '', \
                 cwd: '/tmp/x', title: '', model: 'm', created_at: datetime(), updated_at: datetime(), \
                 message_count: 0, total_cost_usd: 0.0, conversation_id: '', preview: '', permission_mode: '', \
                 add_dirs: '', spawned_by: ''})",
            )
            .param("id", id.to_string()),
        )
        .await
        .unwrap();
    let read = e
        .client
        .get_chat_session(id)
        .await
        .unwrap()
        .expect("stored");
    assert_eq!(read.provider_id, None, "absent = claude-code");
    assert_eq!(read.routed_by, None);
    assert_eq!(read.capabilities, None);
    assert_eq!(read.resume_token, None);
    e.raw
        .run(query("MATCH (s:ChatSession {id: $id}) DETACH DELETE s").param("id", id.to_string()))
        .await
        .unwrap();
}

/// The routing document (decision R2) through the real Cypher: absent = default,
/// the project's document wins over the global one, deleting it returns the
/// project to the global one. The global scope is shared by nature: the test
/// restores it at the end.
#[tokio::test]
async fn routing_settings_project_override_wins_and_delete_returns_to_global() {
    use project_orchestrator::chat::provider::cognitive::{
        load_routing, stored_routing, LearningStage, ProviderRoutingMode, RoutingScope,
        RoutingSettings, ROUTING_KEY,
    };
    use project_orchestrator::chat::provider::settings::{project_scope, GLOBAL};

    let Some(e) = env().await else { return };
    let store: &dyn GraphStore = &e.client;
    let slug = format!("harness-routing-{}", Uuid::new_v4());
    let scope = project_scope(&slug);
    // Whatever another run left in the global scope is kept aside and put back.
    let previous_global = store.get_llm_setting(GLOBAL, ROUTING_KEY).await.unwrap();
    store.delete_llm_setting(GLOBAL, ROUTING_KEY).await.unwrap();

    // Nothing stored anywhere: the default, from nowhere.
    assert_eq!(stored_routing(store, &scope).await.unwrap(), None);
    assert_eq!(
        load_routing(store, Some(&slug)).await.unwrap(),
        (RoutingSettings::default(), RoutingScope::Default)
    );

    // A global document: every project inherits it, missing fields default.
    store
        .put_llm_setting(
            GLOBAL,
            ROUTING_KEY,
            r#"{"mode":"mixed","stage":"advisory"}"#,
        )
        .await
        .unwrap();
    let (s, from) = load_routing(store, Some(&slug)).await.unwrap();
    assert_eq!(s.mode, ProviderRoutingMode::Mixed);
    assert_eq!(s.stage, LearningStage::Advisory);
    assert_eq!(s.exploration_epsilon, 0.05);
    assert_eq!(from, RoutingScope::Global);

    // The project's own document wins, for that project only.
    let project = RoutingSettings {
        mode: ProviderRoutingMode::Full,
        stage: LearningStage::Shadow,
        cost_weight: 0.6,
        ..Default::default()
    };
    store
        .put_llm_setting(
            &scope,
            ROUTING_KEY,
            &serde_json::to_string(&project).unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        load_routing(store, Some(&slug)).await.unwrap(),
        (project.clone(), RoutingScope::Project)
    );
    assert_eq!(
        load_routing(store, Some("some-other-project"))
            .await
            .unwrap()
            .1,
        RoutingScope::Global
    );
    assert_eq!(
        load_routing(store, None).await.unwrap().1,
        RoutingScope::Global
    );

    // Deleting the override returns the project to the global document.
    assert!(store.delete_llm_setting(&scope, ROUTING_KEY).await.unwrap());
    let (s, from) = load_routing(store, Some(&slug)).await.unwrap();
    assert_eq!(
        (s.mode, from),
        (ProviderRoutingMode::Mixed, RoutingScope::Global)
    );
    assert!(!store.delete_llm_setting(&scope, ROUTING_KEY).await.unwrap());

    // Cleanup: the global scope as it was.
    store.delete_llm_setting(GLOBAL, ROUTING_KEY).await.unwrap();
    if let Some(raw) = previous_global {
        store
            .put_llm_setting(GLOBAL, ROUTING_KEY, &raw)
            .await
            .unwrap();
    }
}
