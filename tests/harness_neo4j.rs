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

// ---------------------------------------------------------------------------
// Synaptic writes and note propagation: the Cypher of `reinforce_synapses`,
// `weaken_node_synapses` and `get_propagated_notes`, scoped per project.
// ---------------------------------------------------------------------------

mod synapse_scope {
    use super::*;
    use project_orchestrator::episodes::distill_models::SharingConsent;
    use project_orchestrator::notes::{EntityType, Note, NoteImportance, NoteType};

    fn note(project: Uuid, importance: NoteImportance, consent: SharingConsent) -> Note {
        let mut n = Note::new(
            Some(project),
            NoteType::Guideline,
            format!("t0ab {}", Uuid::new_v4()),
            "t0ab".into(),
        );
        n.importance = importance;
        n.energy = 0.5;
        n.sharing_consent = consent;
        n
    }

    async fn make(
        e: &Env,
        project: Uuid,
        importance: NoteImportance,
        consent: SharingConsent,
    ) -> Note {
        let n = note(project, importance, consent);
        e.client.create_note(&n).await.unwrap();
        // create_note does not persist `sharing_consent`: set it explicitly.
        e.client
            .update_sharing_consent(n.id, &consent)
            .await
            .unwrap();
        n
    }

    async fn count(e: &Env, cypher: &str, ids: &[Uuid]) -> i64 {
        let ids: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        let mut r = e
            .raw
            .execute(query(cypher).param("ids", ids))
            .await
            .unwrap();
        r.next().await.unwrap().unwrap().get("n").unwrap()
    }

    async fn cleanup(e: &Env, ids: &[Uuid], file: Option<&str>) {
        let ids: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        e.raw
            .run(query("MATCH (n:Note) WHERE n.id IN $ids DETACH DELETE n").param("ids", ids))
            .await
            .unwrap();
        if let Some(p) = file {
            e.raw
                .run(query("MATCH (f:File {path: $p}) DETACH DELETE f").param("p", p.to_string()))
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn reinforce_synapses_counts_only_pairs_really_reinforced() {
        let Some(e) = env().await else { return };
        let (p, q) = (Uuid::new_v4(), Uuid::new_v4());
        let a1 = make(&e, p, NoteImportance::Medium, SharingConsent::NotSet).await;
        let a2 = make(&e, p, NoteImportance::Medium, SharingConsent::NotSet).await;
        let b1 = make(&e, q, NoteImportance::Medium, SharingConsent::NotSet).await;
        let ids = [a1.id, a2.id, b1.id];

        // 3 pairs, only (a1, a2) shares a project: 1 pair = 2 directed synapses.
        let n = e
            .client
            .reinforce_synapses(&[a1.id, a2.id, b1.id, Uuid::new_v4()], 0.1)
            .await
            .unwrap();
        let total = count(
            &e,
            "MATCH (a:Note)-[s:SYNAPSE]->(b:Note) WHERE a.id IN $ids AND b.id IN $ids \
             RETURN count(s) AS n",
            &ids,
        )
        .await;
        assert_eq!(total, 2, "only the same-project pair is wired");
        assert_eq!(
            n as i64, total,
            "the count is what was written, not an upper bound"
        );

        // Nothing reinforceable at all: 0, not pairs * 2.
        let none = e
            .client
            .reinforce_synapses(&[a1.id, b1.id], 0.1)
            .await
            .unwrap();
        assert_eq!(none, 0);
        cleanup(&e, &ids, None).await;
    }

    #[tokio::test]
    async fn weaken_node_synapses_leaves_cross_project_synapses_alone() {
        let Some(e) = env().await else { return };
        let (p, q) = (Uuid::new_v4(), Uuid::new_v4());
        let a1 = make(&e, p, NoteImportance::Medium, SharingConsent::NotSet).await;
        let a2 = make(&e, p, NoteImportance::Medium, SharingConsent::NotSet).await;
        let b1 = make(&e, q, NoteImportance::Medium, SharingConsent::NotSet).await;
        let ids = [a1.id, a2.id, b1.id];
        e.client
            .create_synapses(a1.id, &[(a2.id, 0.5)])
            .await
            .unwrap();
        e.client
            .create_synapses(a1.id, &[(b1.id, 0.5)])
            .await
            .unwrap();

        let weakened = e
            .client
            .weaken_node_synapses(a1.id, 0.1, 0.05)
            .await
            .unwrap();
        let mut r = e
            .raw
            .execute(
                query(
                    "MATCH (:Note {id: $a})-[s:SYNAPSE]-(:Note {id: $b}) RETURN collect(s.weight) AS w",
                )
                .param("a", a1.id.to_string())
                .param("b", b1.id.to_string()),
            )
            .await
            .unwrap();
        let w: Vec<f64> = r.next().await.unwrap().unwrap().get("w").unwrap();
        assert!(
            !w.is_empty() && w.iter().all(|x| (x - 0.5).abs() < 1e-9),
            "cross-project untouched: {w:?}"
        );
        assert_eq!(weakened, 2, "the same-project synapse, both directions");
        cleanup(&e, &ids, None).await;
    }

    /// 25 foreign notes outrank 5 local ones; a LIMIT before the project filter
    /// would return no local note at all.
    #[tokio::test]
    async fn get_propagated_notes_filters_project_before_the_limit() {
        let Some(e) = env().await else { return };
        let (me, other) = (Uuid::new_v4(), Uuid::new_v4());
        let file = format!("/t0ab/{}/f.rs", Uuid::new_v4());
        e.raw
            .run(query("CREATE (:File {path: $p})").param("p", file.clone()))
            .await
            .unwrap();
        let mut ids = Vec::new();
        let mut local = Vec::new();
        for _ in 0..25 {
            let n = make(
                &e,
                other,
                NoteImportance::Critical,
                SharingConsent::ExplicitAllow,
            )
            .await;
            e.client
                .link_note_to_entity(n.id, &EntityType::File, &file, None, None)
                .await
                .unwrap();
            ids.push(n.id);
        }
        for _ in 0..5 {
            let n = make(&e, me, NoteImportance::Low, SharingConsent::NotSet).await;
            e.client
                .link_note_to_entity(n.id, &EntityType::File, &file, None, None)
                .await
                .unwrap();
            ids.push(n.id);
            local.push(n.id);
        }

        let got = e
            .client
            .get_propagated_notes(&EntityType::File, &file, 2, 0.0, None, Some(me), false)
            .await
            .unwrap();
        let mut got_ids: Vec<Uuid> = got.iter().map(|p| p.note.id).collect();
        got_ids.sort();
        local.sort();
        assert_eq!(got_ids, local, "Project scope: exactly the 5 local notes");
        cleanup(&e, &ids, Some(&file)).await;
    }

    /// ExplicitDeny foreign notes (more relevant than the local ones) must not
    /// eat the LIMIT in CrossProject.
    #[tokio::test]
    async fn get_propagated_notes_explicit_deny_is_filtered_before_the_limit() {
        let Some(e) = env().await else { return };
        let (me, other) = (Uuid::new_v4(), Uuid::new_v4());
        let file = format!("/t0ab/{}/g.rs", Uuid::new_v4());
        e.raw
            .run(query("CREATE (:File {path: $p})").param("p", file.clone()))
            .await
            .unwrap();
        let mut ids = Vec::new();
        let mut local = Vec::new();
        for _ in 0..25 {
            let n = make(
                &e,
                other,
                NoteImportance::Critical,
                SharingConsent::ExplicitDeny,
            )
            .await;
            e.client
                .link_note_to_entity(n.id, &EntityType::File, &file, None, None)
                .await
                .unwrap();
            ids.push(n.id);
        }
        for _ in 0..5 {
            let n = make(&e, me, NoteImportance::Low, SharingConsent::NotSet).await;
            e.client
                .link_note_to_entity(n.id, &EntityType::File, &file, None, None)
                .await
                .unwrap();
            ids.push(n.id);
            local.push(n.id);
        }
        let got = e
            .client
            .get_propagated_notes(&EntityType::File, &file, 2, 0.0, None, Some(me), true)
            .await
            .unwrap();
        let mut got_ids: Vec<Uuid> = got.iter().map(|p| p.note.id).collect();
        got_ids.sort();
        local.sort();
        assert_eq!(
            got_ids, local,
            "CrossProject: denied notes never reach the LIMIT"
        );
        cleanup(&e, &ids, Some(&file)).await;
    }

    /// CrossProject: coupling weights the score BEFORE the cut. Uncoupled foreign
    /// notes (weight 0) that outrank the local ones raw must not push them out.
    #[tokio::test]
    async fn get_propagated_notes_cross_project_weights_before_the_limit() {
        let Some(e) = env().await else { return };
        let (me, other) = (Uuid::new_v4(), Uuid::new_v4());
        let file = format!("/t0ab/{}/h.rs", Uuid::new_v4());
        e.raw
            .run(query("CREATE (:File {path: $p})").param("p", file.clone()))
            .await
            .unwrap();
        let mut ids = Vec::new();
        let mut local = Vec::new();
        for _ in 0..25 {
            let n = make(
                &e,
                other,
                NoteImportance::Critical,
                SharingConsent::ExplicitAllow,
            )
            .await;
            e.client
                .link_note_to_entity(n.id, &EntityType::File, &file, None, None)
                .await
                .unwrap();
            ids.push(n.id);
        }
        for _ in 0..5 {
            let n = make(&e, me, NoteImportance::Low, SharingConsent::NotSet).await;
            e.client
                .link_note_to_entity(n.id, &EntityType::File, &file, None, None)
                .await
                .unwrap();
            ids.push(n.id);
            local.push(n.id);
        }
        let got = e
            .client
            .get_propagated_notes(&EntityType::File, &file, 2, 0.0, None, Some(me), true)
            .await
            .unwrap();
        assert!(got.len() <= 20);
        for l in &local {
            assert!(
                got.iter().any(|p| p.note.id == *l),
                "local note {l} was cut before the coupling weighting"
            );
        }
        // Weighted order: local notes (coupling 1) come before the uncoupled foreign ones.
        let first_five: Vec<Uuid> = got.iter().take(5).map(|p| p.note.id).collect();
        assert!(
            first_five.iter().all(|i| local.contains(i)),
            "{first_five:?}"
        );
        cleanup(&e, &ids, Some(&file)).await;
    }
}

// ---------------------------------------------------------------------------
// Follow-ups of the anchor model: lineage read from the edge, edge metadata,
// edge backfill, journal order.
// ---------------------------------------------------------------------------

mod anchor_followups {
    use super::*;
    use project_orchestrator::chat::types::SpawnedBy;

    fn spawned(parent: Uuid) -> String {
        SpawnedBy::Conversation {
            parent_session_id: parent,
            tool_use_id: None,
        }
        .to_json_string()
    }

    async fn edge_count(e: &Env, child: Uuid, parent: Uuid) -> i64 {
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
        r.next().await.unwrap().unwrap().get("n").unwrap()
    }

    async fn drop_edges(e: &Env, child: Uuid) {
        e.raw
            .run(
                query("MATCH (:ChatSession {id: $c})-[r:SPAWNED_BY]->() DELETE r")
                    .param("c", child.to_string()),
            )
            .await
            .unwrap();
    }

    /// A mixed set: `with_both` (JSON + edge), `legacy` (JSON only, no edge),
    /// `edge_only` (edge written by the pipeline, no JSON).
    #[tokio::test]
    async fn lineage_reads_the_edge_and_falls_back_to_the_json_without_duplicates() {
        let Some(e) = env().await else { return };
        let store: &dyn GraphStore = &e.client;
        let slug = format!("lineage-{}", Uuid::new_v4().simple());
        let (parent, with_both, legacy, edge_only) = (
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        );
        let mut p = node(parent);
        p.project_slug = Some(slug.clone());
        store.create_chat_session(&p).await.unwrap();
        for (id, json) in [(with_both, true), (legacy, true), (edge_only, false)] {
            let mut c = node(id);
            c.project_slug = Some(slug.clone());
            if json {
                c.spawned_by = Some(spawned(parent));
            }
            store.create_chat_session(&c).await.unwrap();
        }
        drop_edges(&e, legacy).await; // a session written before the edge existed
        store
            .create_spawned_by_relation(
                &edge_only.to_string(),
                &parent.to_string(),
                "conversation",
                None,
                None,
            )
            .await
            .unwrap();
        // pipeline call on a session that already has its edge: still one edge
        store
            .create_spawned_by_relation(
                &with_both.to_string(),
                &parent.to_string(),
                "conversation",
                None,
                None,
            )
            .await
            .unwrap();

        let mut kids: Vec<Uuid> = store
            .get_session_children(parent)
            .await
            .unwrap()
            .iter()
            .map(|s| s.id)
            .collect();
        kids.sort();
        let mut want = vec![with_both, legacy, edge_only];
        want.sort();
        assert_eq!(kids, want, "edge, JSON-only and edge-only, each once");

        // detached sessions stay out of the default list, all three of them
        let (listed, total) = store
            .list_chat_sessions(Some(&slug), None, 50, 0, false)
            .await
            .unwrap();
        assert_eq!(
            listed.iter().map(|s| s.id).collect::<Vec<_>>(),
            vec![parent]
        );
        assert_eq!(total, 1);
        let (all, total) = store
            .list_chat_sessions(Some(&slug), None, 50, 0, true)
            .await
            .unwrap();
        assert_eq!((all.len(), total), (4, 4));

        for id in [with_both, legacy, edge_only, parent] {
            store.delete_chat_session(id).await.unwrap();
        }
    }

    #[tokio::test]
    async fn run_metadata_of_the_edge_survives_merge_and_later_calls() {
        let Some(e) = env().await else { return };
        let store: &dyn GraphStore = &e.client;
        let (parent, child) = (Uuid::new_v4(), Uuid::new_v4());
        let (run, task) = (Uuid::new_v4(), Uuid::new_v4());
        store.create_chat_session(&node(parent)).await.unwrap();
        let mut c = node(child);
        c.spawned_by = Some(spawned(parent));
        store.create_chat_session(&c).await.unwrap();
        store
            .create_spawned_by_relation(
                &child.to_string(),
                &parent.to_string(),
                "plan_runner",
                Some(run),
                Some(task),
            )
            .await
            .unwrap();
        async fn props(e: &Env, child: Uuid) -> (String, String, String, bool) {
            let mut r = e
                .raw
                .execute(
                    query(
                        "MATCH (:ChatSession {id: $c})-[r:SPAWNED_BY]->() \
                         RETURN r.type AS t, r.run_id AS run, r.task_id AS task, \
                                r.created_at IS NOT NULL AS has_created",
                    )
                    .param("c", child.to_string()),
                )
                .await
                .unwrap();
            let row = r.next().await.unwrap().unwrap();
            (
                row.get("t").unwrap_or_default(),
                row.get("run").unwrap_or_default(),
                row.get("task").unwrap_or_default(),
                row.get("has_created").unwrap(),
            )
        }
        let expected = (
            "plan_runner".to_string(),
            run.to_string(),
            task.to_string(),
            true,
        );
        assert_eq!(props(&e, child).await, expected);
        // a later call that carries no run (the conversation path) must not erase it
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
        assert_eq!(props(&e, child).await, expected, "metadata kept");
        assert_eq!(edge_count(&e, child, parent).await, 1);
        store.delete_chat_session(child).await.unwrap();
        store.delete_chat_session(parent).await.unwrap();
    }
    #[tokio::test]
    async fn backfill_of_the_edges_is_idempotent() {
        let Some(e) = env().await else { return };
        let store: &dyn GraphStore = &e.client;
        let (parent, child, orphan) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        store.create_chat_session(&node(parent)).await.unwrap();
        let mut c = node(child);
        c.spawned_by = Some(spawned(parent));
        store.create_chat_session(&c).await.unwrap();
        // the parent of this one is not in the graph: nothing to link
        let mut o = node(orphan);
        o.spawned_by = Some(spawned(Uuid::new_v4()));
        store.create_chat_session(&o).await.unwrap();
        drop_edges(&e, child).await; // written before the edge existed

        async fn state(e: &Env, ids: [Uuid; 2]) -> (i64, i64, String) {
            let mut r = e
                .raw
                .execute(
                    query(
                        "MATCH (s:ChatSession) WHERE s.id IN $ids \
                         OPTIONAL MATCH (s)-[r:SPAWNED_BY]->(:ChatSession) \
                         RETURN count(r) AS edges, count(DISTINCT s) AS sessions, \
                                coalesce(toString(max(r.created_at)), '') AS created",
                    )
                    .param("ids", ids.map(|i| i.to_string()).to_vec()),
                )
                .await
                .unwrap();
            let row = r.next().await.unwrap().unwrap();
            (
                row.get("edges").unwrap(),
                row.get("sessions").unwrap(),
                row.get("created").unwrap(),
            )
        }
        assert_eq!(state(&e, [child, orphan]).await.0, 0);
        let first = store.backfill_spawned_by_edges().await.unwrap();
        assert!(first >= 1, "the legacy child gets its edge");
        let after_first = state(&e, [child, orphan]).await;
        assert_eq!((after_first.0, after_first.1), (1, 2));
        assert_eq!(edge_count(&e, child, parent).await, 1);
        store.backfill_spawned_by_edges().await.unwrap();
        assert_eq!(
            state(&e, [child, orphan]).await,
            after_first,
            "two runs = same state (edge count and created_at)"
        );
        // lineage reads see it through the edge
        let kids = store.get_session_children(parent).await.unwrap();
        assert_eq!(kids.iter().map(|s| s.id).collect::<Vec<_>>(), vec![child]);
        for id in [child, orphan, parent] {
            store.delete_chat_session(id).await.unwrap();
        }
    }

    #[tokio::test]
    async fn concurrent_events_of_one_session_get_gapless_increasing_seq() {
        use project_orchestrator::chat::anchor::*;
        let Some(e) = env().await else { return };
        let store = std::sync::Arc::new(e.client);
        let sid = Uuid::new_v4();
        store.create_chat_session(&node(sid)).await.unwrap();
        const N: usize = 30;
        let mut set = tokio::task::JoinSet::new();
        for i in 0..N {
            let s = store.clone();
            set.spawn(async move {
                s.apply_anchor_op(
                    sid,
                    AnchorOp::Add(NewAnchor::new(
                        AnchorTargetType::File,
                        format!("src/seq{i}.rs"),
                        [AnchorRole::Mention],
                        AnchorActor::User,
                        "harness",
                    )),
                )
                .await
                .unwrap()
                .event
                .unwrap()
                .seq
            });
        }
        let mut returned = Vec::new();
        while let Some(r) = set.join_next().await {
            returned.push(r.unwrap());
        }
        returned.sort();
        let want: Vec<u64> = (1..=N as u64).collect();
        assert_eq!(returned, want, "no gap, no duplicate in what was returned");
        let journal = store.list_anchor_events(sid).await.unwrap();
        assert_eq!(
            journal.iter().map(|e| e.seq).collect::<Vec<_>>(),
            want,
            "the journal comes back ordered by seq"
        );
        store.delete_chat_session(sid).await.unwrap();
    }

    #[tokio::test]
    async fn events_without_seq_sort_first_by_time_then_the_numbered_ones() {
        use project_orchestrator::chat::anchor::*;
        let Some(e) = env().await else { return };
        let store: &dyn GraphStore = &e.client;
        let sid = Uuid::new_v4();
        store.create_chat_session(&node(sid)).await.unwrap();
        let add = |t: &'static str| {
            NewAnchor::new(
                AnchorTargetType::File,
                t,
                [AnchorRole::Mention],
                AnchorActor::User,
                "harness",
            )
        };
        let a = store
            .apply_anchor_op(sid, AnchorOp::Add(add("a.rs")))
            .await
            .unwrap()
            .anchor
            .id;
        let b = store
            .apply_anchor_op(sid, AnchorOp::Add(add("b.rs")))
            .await
            .unwrap()
            .anchor
            .id;
        // turn them into events written before numbering: no seq, b older than a
        e.raw
            .run(
                query(
                    "MATCH (e:AnchorEvent {session_id: $sid}) REMOVE e.seq \
                     SET e.at = CASE e.anchor_id WHEN $a THEN '2020-01-02T00:00:00.000000Z' \
                                                 ELSE '2020-01-01T00:00:00.000000Z' END",
                )
                .param("sid", sid.to_string())
                .param("a", a.to_string()),
            )
            .await
            .unwrap();
        let c = store
            .apply_anchor_op(sid, AnchorOp::Add(add("c.rs")))
            .await
            .unwrap()
            .anchor
            .id;
        let order: Vec<Uuid> = store
            .list_anchor_events(sid)
            .await
            .unwrap()
            .iter()
            .map(|e| e.anchor_id)
            .collect();
        assert_eq!(order, vec![b, a, c]);
        store.delete_chat_session(sid).await.unwrap();
    }
}
