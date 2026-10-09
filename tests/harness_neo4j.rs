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
        execution_place: Default::default(),
        access: Default::default(),
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

// ---------------------------------------------------------------------------
// Chat anchors (typed context anchors) — the Cypher, on a real Neo4j.
// ---------------------------------------------------------------------------

mod anchors {
    use super::*;
    use project_orchestrator::chat::anchor::*;
    use project_orchestrator::chat::types::SpawnedBy;

    fn typed(e: anyhow::Error) -> AnchorError {
        e.downcast::<AnchorError>().expect("a typed AnchorError")
    }

    fn user(t: AnchorTargetType, id: &str, roles: &[AnchorRole]) -> NewAnchor {
        NewAnchor::new(t, id, roles.iter().copied(), AnchorActor::User, "harness")
    }

    #[tokio::test]
    async fn anchors_follow_the_rules_and_the_journal_on_real_neo4j() {
        let Some(e) = env().await else { return };
        let store: &dyn GraphStore = &e.client;
        let sid = Uuid::new_v4();
        store.create_chat_session(&node(sid)).await.unwrap();

        let plan = Uuid::new_v4().to_string();
        let a = store
            .add_anchor(
                sid,
                user(AnchorTargetType::Plan, &plan, &[AnchorRole::Mention]),
            )
            .await
            .unwrap();
        // adding the same thing again is idempotent
        let again = store
            .add_anchor(
                sid,
                user(AnchorTargetType::Plan, &plan, &[AnchorRole::Mention]),
            )
            .await
            .unwrap();
        assert_eq!(again.id, a.id);
        assert_eq!(store.list_session_anchors(sid).await.unwrap().len(), 1);

        let promoted = store
            .promote_anchor_role(
                sid,
                a.id,
                AnchorRole::Focus,
                1,
                AnchorActor::User,
                "harness",
            )
            .await
            .unwrap();
        assert_eq!(promoted.version, 2);
        let stale = store
            .promote_anchor_role(sid, a.id, AnchorRole::Focus, 1, AnchorActor::User, "x")
            .await
            .unwrap_err();
        assert!(matches!(typed(stale), AnchorError::VersionConflict { .. }));

        // origin is immutable
        let origin = store
            .add_anchor(
                sid,
                NewAnchor::new(
                    AnchorTargetType::Project,
                    Uuid::new_v4().to_string(),
                    [AnchorRole::Origin],
                    AnchorActor::System,
                    "sys",
                ),
            )
            .await
            .unwrap();
        let refused = store
            .remove_anchor(sid, origin.id, 1, AnchorActor::User, "harness")
            .await
            .unwrap_err();
        assert_eq!(typed(refused), AnchorError::OriginImmutable);

        // dangling keeps the anchors; the journal has one entry per change
        assert_eq!(
            store
                .mark_anchors_dangling(AnchorTargetType::Plan, &plan)
                .await
                .unwrap(),
            1
        );
        let anchors = store.list_session_anchors(sid).await.unwrap();
        assert_eq!(anchors.len(), 2);
        assert!(anchors.iter().any(|x| x.state == AnchorState::Dangling));
        let kinds: Vec<_> = store
            .list_anchor_events(sid)
            .await
            .unwrap()
            .iter()
            .map(|ev| ev.kind)
            .collect();
        assert_eq!(kinds.len(), 4);
        assert!(kinds.contains(&AnchorEventKind::StateChanged));

        // the composite unique key backs the "one anchor per target" rule
        let dup = e
            .raw
            .run(
                query("CREATE (:Anchor {id: $id, session_id: $sid, target_type: 'plan', target_id: $tid})")
                    .param("id", Uuid::new_v4().to_string())
                    .param("sid", sid.to_string())
                    .param("tid", plan.clone()),
            )
            .await;
        assert!(
            dup.is_err(),
            "the unique constraint must refuse a duplicate"
        );

        // deleting the session deletes its anchors and journal
        assert!(store.delete_chat_session(sid).await.unwrap());
        assert!(store.list_session_anchors(sid).await.unwrap().is_empty());
        assert!(store.list_anchor_events(sid).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn concurrent_focus_adds_never_exceed_the_cap() {
        let Some(e) = env().await else { return };
        let store = std::sync::Arc::new(e.client);
        let sid = Uuid::new_v4();
        store.create_chat_session(&node(sid)).await.unwrap();
        let mut set = tokio::task::JoinSet::new();
        for i in 0..8 {
            let s = store.clone();
            set.spawn(async move {
                s.add_anchor(
                    sid,
                    user(
                        AnchorTargetType::File,
                        &format!("src/f{i}.rs"),
                        &[AnchorRole::Focus],
                    ),
                )
                .await
                .is_ok()
            });
        }
        let mut ok = 0;
        while let Some(r) = set.join_next().await {
            ok += usize::from(r.unwrap());
        }
        assert_eq!(ok, MAX_FOCUS_ANCHORS);
        assert_eq!(
            store.list_session_anchors(sid).await.unwrap().len(),
            MAX_FOCUS_ANCHORS
        );
        store.delete_chat_session(sid).await.unwrap();
    }

    #[tokio::test]
    async fn reverse_query_pages_on_real_neo4j() {
        let Some(e) = env().await else { return };
        let store: &dyn GraphStore = &e.client;
        let target = Uuid::new_v4().to_string();
        let mut sids = vec![];
        for _ in 0..5 {
            let sid = Uuid::new_v4();
            store.create_chat_session(&node(sid)).await.unwrap();
            store
                .add_anchor(
                    sid,
                    user(AnchorTargetType::Note, &target, &[AnchorRole::Mention]),
                )
                .await
                .unwrap();
            sids.push(sid);
        }
        let mut seen = vec![];
        let mut cursor: Option<String> = None;
        loop {
            let page = store
                .list_sessions_for_target(
                    AnchorTargetType::Note,
                    &target,
                    None,
                    2,
                    cursor.as_deref(),
                )
                .await
                .unwrap();
            seen.extend(page.items.iter().map(|a| a.session_id));
            match page.next_cursor {
                Some(c) => cursor = Some(c),
                None => break,
            }
        }
        seen.sort();
        sids.sort();
        assert_eq!(seen, sids);
        for s in sids {
            store.delete_chat_session(s).await.unwrap();
        }
    }

    #[tokio::test]
    async fn backfill_is_idempotent_and_revert_removes_only_its_anchors() {
        let Some(e) = env().await else { return };
        let store: &dyn GraphStore = &e.client;
        let slug = format!("anchor-proj-{}", Uuid::new_v4().simple());
        let project = project_orchestrator::neo4j::models::ProjectNode {
            id: Uuid::new_v4(),
            name: slug.clone(),
            slug: slug.clone(),
            root_path: "/tmp/anchor-harness".to_string(),
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
        let mut n = node(sid);
        n.project_slug = Some(slug.clone());
        store.create_chat_session(&n).await.unwrap();

        store.backfill_project_anchors().await.unwrap();
        let first = store.list_session_anchors(sid).await.unwrap();
        store.backfill_project_anchors().await.unwrap();
        let second = store.list_session_anchors(sid).await.unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first, second, "two runs = same state");
        assert!(first[0].inferred && first[0].by == AnchorActor::System);
        assert_eq!(first[0].target_id, project.id.to_string());
        assert_eq!(store.list_anchor_events(sid).await.unwrap().len(), 1);

        assert!(store.revert_inferred_anchors().await.unwrap() >= 1);
        assert!(store.list_session_anchors(sid).await.unwrap().is_empty());
        assert!(store.list_anchor_events(sid).await.unwrap().is_empty());
        store.delete_chat_session(sid).await.unwrap();
        store.delete_project(project.id, &slug).await.unwrap();
    }

    #[tokio::test]
    async fn a_spawned_session_gets_the_edge_in_addition_to_the_json() {
        let Some(e) = env().await else { return };
        let store: &dyn GraphStore = &e.client;
        let parent = Uuid::new_v4();
        store.create_chat_session(&node(parent)).await.unwrap();
        let child = Uuid::new_v4();
        let mut c = node(child);
        let sb = SpawnedBy::Conversation {
            parent_session_id: parent,
            tool_use_id: None,
        };
        c.spawned_by = Some(sb.to_json_string());
        store.create_chat_session(&c).await.unwrap();
        // the pipeline's own call afterwards must not add a second edge
        store
            .create_spawned_by_relation(
                &child.to_string(),
                &parent.to_string(),
                "conversation",
                None,
                None,
            )
            .await
            .unwrap();

        let mut r = e
            .raw
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
        let n: i64 = r.next().await.unwrap().unwrap().get("n").unwrap();
        assert_eq!(n, 1);
        // the JSON is unchanged and reads still work
        let read = store.get_chat_session(child).await.unwrap().unwrap();
        assert_eq!(
            read.spawned_by.as_deref(),
            Some(sb.to_json_string().as_str())
        );
        assert_eq!(store.get_session_children(parent).await.unwrap().len(), 1);
        store.delete_chat_session(child).await.unwrap();
        store.delete_chat_session(parent).await.unwrap();
    }
}
