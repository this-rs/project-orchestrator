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
        // create_note persists the consent carried by the note.
        e.client.create_note(&n).await.unwrap();
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

// ---------------------------------------------------------------------------
// Anchor resolver (T3): the scoped neighbourhood Cypher, on a real Neo4j.
// ---------------------------------------------------------------------------
//
// Every scenario builds ONE graph twice: in Neo4j (raw Cypher or the store) and
// in an `InMemoryGraph` with `InMemoryOwnership`. `MockGraphStore` is
// `cfg(test)`-only, so these tests compare against `expand_in_memory_scoped`,
// the function the mock's `get_scoped_entity_neighborhood` delegates to.

mod anchor_scope {
    use super::*;
    use project_orchestrator::chat::anchor::*;
    use project_orchestrator::chat::anchor_resolver::{
        est_tokens, render_anchor_map, render_live_block, resolve_anchors, resolve_session_scope,
        select_nodes, ExclusionReason, ScopeProject, FOCUS_BUDGET, MENTION_BUDGET,
        NOTICE_NO_CONTEXT,
    };
    use project_orchestrator::episodes::distill_models::{
        SharingConsent, SharingMode, SharingPolicy,
    };
    use project_orchestrator::graph::neighborhood::{
        entity_kind, expand_in_memory, expand_in_memory_scoped, hierarchy_step_allowed,
        InMemoryGraph, InMemoryOwnership, Layer, NeighborhoodParams, NodeProps, ProjectFilter,
        RawNode, ScopedNeighborhood, DEFAULT_FANOUT, HIERARCHY_V1,
    };
    use project_orchestrator::neo4j::models::ProjectNode;
    use project_orchestrator::notes::{Note, NoteType};
    use project_orchestrator::sharing::consent_gate::ReadDenial;
    use std::collections::{BTreeSet, HashMap};

    const SECRET: &[u8] = b"harness-secret";

    enum V {
        S(String),
        F(f64),
    }

    fn bind(q: neo4rs::Query, k: &str, v: &V) -> neo4rs::Query {
        match v {
            V::S(s) => q.param(k, s.clone()),
            V::F(f) => q.param(k, *f),
        }
    }

    /// The same graph, in Neo4j and in memory.
    struct World<'a> {
        e: &'a Env,
        run: String,
        graph: InMemoryGraph,
        own: InMemoryOwnership,
        kinds: HashMap<String, &'static str>,
        projects: Vec<Uuid>,
        sessions: Vec<Uuid>,
    }

    impl<'a> World<'a> {
        fn new(e: &'a Env) -> Self {
            World {
                e,
                run: Uuid::new_v4().simple().to_string(),
                graph: Default::default(),
                own: Default::default(),
                kinds: HashMap::new(),
                projects: vec![],
                sessions: vec![],
            }
        }

        fn uid(&self) -> String {
            Uuid::new_v4().to_string()
        }

        fn path(&self, name: &str) -> String {
            format!("/t3bis/{}/{}", self.run, name)
        }

        async fn run_cypher(&self, cypher: &str, vals: Vec<(&str, V)>) {
            let mut q = query(cypher).param("run", self.run.clone());
            for (k, v) in &vals {
                q = bind(q, k, v);
            }
            self.e.raw.run(q).await.unwrap();
        }

        async fn project(&mut self, name: &str) -> Uuid {
            let slug = format!("{name}-{}", self.run);
            let p = ProjectNode {
                id: Uuid::new_v4(),
                name: slug.clone(),
                slug: slug.clone(),
                root_path: format!("/tmp/t3bis/{slug}"),
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
            let store: &dyn GraphStore = &self.e.client;
            store.create_project(&p).await.unwrap();
            self.run_cypher(
                "MATCH (n:Project {id: $id}) SET n.h_run = $run",
                vec![("id", V::S(p.id.to_string()))],
            )
            .await;
            let props = NodeProps {
                name: Some(slug.clone()),
                slug: Some(slug),
                ..Default::default()
            };
            self.graph
                .add_node(RawNode::from_props("project", p.id.to_string(), &props));
            self.kinds.insert(p.id.to_string(), "project");
            self.projects.push(p.id);
            p.id
        }

        /// Raw node. `owner_prop`: the node carries `project_id` itself (notes,
        /// files...); otherwise its owner can only come from the graph shape.
        /// `owner` feeds the in-memory ownership in both cases.
        async fn node(
            &mut self,
            ty: &'static str,
            id: &str,
            p: NodeProps,
            owner: Option<Uuid>,
            owner_prop: bool,
            consent: Option<SharingConsent>,
        ) {
            let k = entity_kind(ty).unwrap();
            let mut sets = vec![
                format!("n.{} = $id", k.id_prop),
                "n.h_run = $run".to_string(),
            ];
            let mut vals = vec![("id", V::S(id.to_string()))];
            if let (true, Some(o)) = (owner_prop, owner) {
                sets.push("n.project_id = $pid".into());
                vals.push(("pid", V::S(o.to_string())));
            }
            if let Some(c) = consent {
                sets.push("n.sharing_consent = $consent".into());
                let s = serde_json::to_value(c).unwrap();
                vals.push(("consent", V::S(s.as_str().unwrap().to_string())));
            }
            for (name, v) in [
                ("content", p.content.clone()),
                ("title", p.title.clone()),
                ("name", p.name.clone()),
                ("status", p.status.clone()),
            ] {
                if let Some(v) = v {
                    sets.push(format!("n.{name} = ${name}"));
                    vals.push((name, V::S(v)));
                }
            }
            for (name, v) in [
                ("energy", p.energy),
                ("priority", p.priority),
                ("pagerank", p.pagerank),
            ] {
                if let Some(v) = v {
                    sets.push(format!("n.{name} = ${name}"));
                    vals.push((name, V::F(v)));
                }
            }
            let cy = format!("CREATE (n:{}) SET {}", k.label, sets.join(", "));
            self.run_cypher(&cy, vals).await;
            let mut mp = p;
            if ty == "file" {
                mp.path = Some(id.to_string());
            }
            self.graph
                .add_node(RawNode::from_props(ty, id.to_string(), &mp));
            self.kinds.insert(id.to_string(), ty);
            if let Some(o) = owner {
                self.own.project_of.insert(id.to_string(), o.to_string());
            }
            if let Some(c) = consent {
                self.own.consent_of.insert(id.to_string(), c);
            }
        }

        async fn note(&mut self, owner: Uuid, content: &str, energy: f64) -> String {
            let id = self.uid();
            let p = NodeProps {
                content: Some(content.to_string()),
                energy: Some(energy),
                ..Default::default()
            };
            self.node("note", &id, p, Some(owner), true, None).await;
            id
        }

        /// A note written by the real store (`create_note` + `update_sharing_consent`).
        async fn store_note(
            &mut self,
            owner: Uuid,
            content: &str,
            energy: f64,
            consent: SharingConsent,
        ) -> String {
            let note = Note::new(
                Some(owner),
                NoteType::Observation,
                content.to_string(),
                "t3bis".into(),
            );
            let store: &dyn GraphStore = &self.e.client;
            store.create_note(&note).await.unwrap();
            store
                .update_sharing_consent(note.id, &consent)
                .await
                .unwrap();
            let id = note.id.to_string();
            self.run_cypher(
                "MATCH (n:Note {id: $id}) SET n.energy = $energy, n.h_run = $run",
                vec![("id", V::S(id.clone())), ("energy", V::F(energy))],
            )
            .await;
            let p = NodeProps {
                content: Some(content.to_string()),
                energy: Some(energy),
                note_type: Some("observation".into()),
                importance: Some("medium".into()),
                status: Some("active".into()),
                ..Default::default()
            };
            self.graph
                .add_node(RawNode::from_props("note", id.clone(), &p));
            // the store links every note to its project (`HAS_NOTE`)
            self.graph
                .add_edge(&owner.to_string(), &id, "HAS_NOTE", 1.0);
            self.kinds.insert(id.clone(), "note");
            self.own.project_of.insert(id.clone(), owner.to_string());
            self.own.consent_of.insert(id.clone(), consent);
            id
        }

        /// `(a)-[rel]->(b)`; `w` = `weight` (SYNAPSE) or `similarity_score` (LINKED_TO).
        async fn edge(&mut self, a: &str, b: &str, rel: &'static str, w: Option<f64>) {
            let (ka, kb) = (self.kinds[a], self.kinds[b]);
            let (ea, eb) = (entity_kind(ka).unwrap(), entity_kind(kb).unwrap());
            let prop = match rel {
                "SYNAPSE" => Some("weight"),
                "LINKED_TO" => Some("similarity_score"),
                _ => None,
            };
            let mut cy = format!(
                "MATCH (a:{la} {{{pa}: $a}}) MATCH (b:{lb} {{{pb}: $b}}) \
                 CREATE (a)-[r:{rel}]->(b) SET r.h_run = $run",
                la = ea.label,
                pa = ea.id_prop,
                lb = eb.label,
                pb = eb.id_prop
            );
            let mut vals = vec![("a", V::S(a.to_string())), ("b", V::S(b.to_string()))];
            if let (Some(p), Some(w)) = (prop, w) {
                cy.push_str(&format!(", r.{p} = $w"));
                vals.push(("w", V::F(w)));
            }
            self.run_cypher(&cy, vals).await;
            let eff = match rel {
                "SYNAPSE" => w.unwrap_or(0.5),
                "LINKED_TO" => w.unwrap_or(1.0),
                _ => 1.0,
            };
            self.graph.add_edge(a, b, rel, eff);
        }

        async fn neo(
            &self,
            ty: &str,
            id: &str,
            p: &NeighborhoodParams,
            f: &ProjectFilter,
        ) -> ScopedNeighborhood {
            self.e
                .client
                .get_scoped_entity_neighborhood(ty, id, p, f)
                .await
                .unwrap()
                .expect("centre exists")
        }

        fn mem(
            &self,
            ty: &str,
            id: &str,
            p: &NeighborhoodParams,
            f: &ProjectFilter,
        ) -> ScopedNeighborhood {
            expand_in_memory_scoped(&self.graph, &self.own, ty, id, p, f).expect("centre exists")
        }

        /// Neo4j walk, asserted IDENTICAL to the in-memory one.
        async fn same(
            &self,
            ty: &str,
            id: &str,
            p: &NeighborhoodParams,
            f: &ProjectFilter,
        ) -> ScopedNeighborhood {
            let n = self.neo(ty, id, p, f).await;
            let m = self.mem(ty, id, p, f);
            assert_eq!(
                sig(&n),
                sig(&m),
                "Neo4j walk differs from the in-memory one"
            );
            n
        }

        async fn cleanup(&self) {
            let store: &dyn GraphStore = &self.e.client;
            for s in &self.sessions {
                store.delete_chat_session(*s).await.ok();
            }
            self.run_cypher("MATCH (n) WHERE n.h_run = $run DETACH DELETE n", vec![])
                .await;
        }
    }

    fn sig(w: &ScopedNeighborhood) -> String {
        let line = |n: &project_orchestrator::graph::neighborhood::ScopedNode| {
            format!(
                "{}|{}|{}|{:.6}|{:?}|{:?}",
                n.node.id, n.node.node_type, n.node.label, n.node.weight, n.project_id, n.consent
            )
        };
        let mut nodes: Vec<String> = w.nodes.iter().map(line).collect();
        nodes.sort();
        let mut edges: Vec<String> = w
            .edges
            .iter()
            .map(|e| format!("{}-{}->{} {:.4}", e.source, e.rel, e.target, e.weight))
            .collect();
        edges.sort();
        format!(
            "centre {:?}\nnodes {:#?}\nedges {:#?}",
            w.center.as_ref().map(line),
            nodes,
            edges
        )
    }

    fn ids(w: &ScopedNeighborhood) -> BTreeSet<String> {
        w.nodes.iter().map(|n| n.node.id.clone()).collect()
    }

    /// The budgets' walk parameters, as `resolve_one` builds them (`params_for`).
    fn params(depth: u32, max_nodes: usize) -> NeighborhoodParams {
        NeighborhoodParams {
            depth,
            min_weight: 0.0,
            limit: (max_nodes * 4).max(1),
            layers: Layer::ALL.to_vec(),
            fanout: DEFAULT_FANOUT,
            frontier_cap: DEFAULT_FANOUT,
        }
    }

    fn mention_params() -> NeighborhoodParams {
        params(MENTION_BUDGET.depth, MENTION_BUDGET.max_nodes)
    }

    fn focus_params() -> NeighborhoodParams {
        params(FOCUS_BUDGET.depth, FOCUS_BUDGET.max_nodes)
    }

    fn scope_project(id: Uuid, slug: &str) -> ScopeProject {
        ScopeProject {
            id,
            slug: slug.into(),
            name: slug.into(),
        }
    }

    fn anchor(t: AnchorTargetType, id: &str, role: AnchorRole, state: AnchorState) -> Anchor {
        let now = Utc::now();
        Anchor {
            id: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
            target_type: t,
            target_id: id.into(),
            roles: [role].into_iter().collect(),
            state,
            by: AnchorActor::User,
            inferred: false,
            confidence: 1.0,
            snapshot_name: None,
            snapshot_path: None,
            snapshot_type: None,
            rev: None,
            version: 1,
            created_at: now,
            updated_at: now,
        }
    }

    // (1) boundary / eviction ------------------------------------------------

    #[tokio::test]
    async fn foreign_neighbours_never_cross_the_boundary_nor_evict_local_ones() {
        let Some(e) = env().await else { return };
        let mut w = World::new(&e);
        let (a, b) = (w.project("a").await, w.project("b").await);
        let f = w.path("f.rs");
        w.node("file", &f, NodeProps::default(), Some(a), true, None)
            .await;
        w.edge(&a.to_string(), &f, "CONTAINS", None).await;
        // 5 local neighbours, weak edges
        let mut locals = vec![];
        for i in 0..5 {
            let n = w.note(a, &format!("local {i}"), 0.3).await;
            w.edge(&n, &f, "LINKED_TO", Some(0.4)).await;
            locals.push(n);
        }
        // 25 more salient foreign neighbours, relations the walk may follow
        let mut foreign = BTreeSet::new();
        for i in 0..25 {
            let n = w.note(b, &format!("foreign {i}"), 0.95).await;
            w.edge(&n, &f, "LINKED_TO", Some(0.9)).await;
            foreign.insert(n);
        }
        // depth 2: local[0] has 32 foreign neighbours (> fanout) and one local `deep`
        let deep = w.note(a, "deep local", 0.3).await;
        w.edge(&locals[0], &deep, "SYNAPSE", Some(0.2)).await;
        for i in 0..32 {
            let n = w.note(b, &format!("foreign deep {i}"), 0.95).await;
            w.edge(&locals[0], &n, "SYNAPSE", Some(0.9)).await;
            foreign.insert(n);
        }
        let only_a = ProjectFilter::only(a.to_string());

        for (label, p) in [
            ("mention d1", mention_params()),
            ("focus d2", focus_params()),
        ] {
            let scoped = w.same("file", &f, &p, &only_a).await;
            let got = ids(&scoped);
            assert!(
                got.iter().all(|i| !foreign.contains(i)),
                "{label}: a node of B crossed the boundary"
            );
            assert!(scoped
                .nodes
                .iter()
                .all(|n| n.project_id.as_deref() == Some(&a.to_string())));
            for l in &locals {
                assert!(got.contains(l), "{label}: local neighbour evicted");
            }
            if p.depth == 2 {
                assert!(got.contains(&deep), "{label}: depth-2 local evicted");
            }
            // the centre comes back with its owner
            assert_eq!(
                scoped.center.as_ref().unwrap().project_id.as_deref(),
                Some(a.to_string().as_str())
            );
        }

        // control: the unfiltered walk (the old function) IS evicting
        let old1 = e
            .client
            .get_entity_neighborhood("file", &f, &mention_params())
            .await
            .unwrap()
            .unwrap();
        let old1_ids: BTreeSet<_> = old1.nodes.iter().map(|n| n.id.clone()).collect();
        assert!(old1_ids.iter().any(|i| foreign.contains(i)));
        assert!(
            locals.iter().any(|l| !old1_ids.contains(l)),
            "control: 31 candidates > fanout 30, one local must be evicted"
        );
        let old2 = e
            .client
            .get_entity_neighborhood("file", &f, &focus_params())
            .await
            .unwrap()
            .unwrap();
        assert!(
            !old2.nodes.iter().any(|n| n.id == deep),
            "control: 34 neighbours of local[0] > fanout 30"
        );
        // and the pure unfiltered walk agrees with the Neo4j one
        let mem_old = expand_in_memory(&w.graph, "file", &f, &focus_params()).unwrap();
        let a_set: BTreeSet<_> = old2.nodes.iter().map(|n| n.id.clone()).collect();
        let b_set: BTreeSet<_> = mem_old.nodes.iter().map(|n| n.id.clone()).collect();
        assert_eq!(a_set.len(), b_set.len());
        w.cleanup().await;
    }

    // (7) frontier ties ----------------------------------------------------------

    #[tokio::test]
    async fn frontier_ties_break_by_public_id_not_by_element_id() {
        let Some(e) = env().await else { return };
        let mut w = World::new(&e);
        let a = w.project("a").await;
        let f = w.path("ties.rs");
        w.node("file", &f, NodeProps::default(), Some(a), true, None)
            .await;
        // 35 equal candidates (> frontier_cap 30); public ids sort in the REVERSE
        // of the creation order, so elementId order and id order disagree
        for i in 0..35u128 {
            let base = u128::from_str_radix(&w.run, 16).unwrap() & !0xffff;
            let id = Uuid::from_u128(base | (0x1000 - i)).to_string();
            let p = NodeProps {
                content: Some(format!("tie {i}")),
                energy: Some(0.5),
                ..Default::default()
            };
            w.node("note", &id, p, Some(a), true, None).await;
            w.edge(&id, &f, "LINKED_TO", Some(0.5)).await;
            let leaf = w.note(a, &format!("leaf {i}"), 0.5).await;
            w.edge(&id, &leaf, "SYNAPSE", Some(0.5)).await;
        }
        let only_a = ProjectFilter::only(a.to_string());
        let walk = w.same("file", &f, &params(2, 12), &only_a).await;
        assert_eq!(
            walk.nodes
                .iter()
                .filter(|n| n.node.label.starts_with("leaf"))
                .count(),
            30
        );
        w.cleanup().await;
    }

    // (2) consent ------------------------------------------------------------

    #[tokio::test]
    async fn consent_post_filter_matches_the_in_memory_walk() {
        let Some(e) = env().await else { return };
        let store: &dyn GraphStore = &e.client;
        let mut w = World::new(&e);
        let (a, b) = (w.project("a").await, w.project("b").await);
        // centre in B, explicitly allowed (else the anchor itself is refused)
        let c = w
            .store_note(b, "centre B", 0.9, SharingConsent::ExplicitAllow)
            .await;
        let deny = w
            .store_note(b, "SECRET deny", 0.9, SharingConsent::ExplicitDeny)
            .await;
        let unset = w
            .store_note(b, "SECRET unset", 0.9, SharingConsent::NotSet)
            .await;
        let allow = w
            .store_note(b, "ALLOWED b", 0.8, SharingConsent::ExplicitAllow)
            .await;
        let local_deny = w
            .store_note(a, "local deny", 0.7, SharingConsent::ExplicitDeny)
            .await;
        let local_unset = w
            .store_note(a, "local unset", 0.6, SharingConsent::NotSet)
            .await;
        for n in [&deny, &unset, &allow, &local_deny, &local_unset] {
            w.edge(&c, n, "SYNAPSE", Some(0.8)).await;
        }
        // behind the refused note: reachable only through it
        let behind = w
            .store_note(b, "SECRET behind", 0.9, SharingConsent::ExplicitAllow)
            .await;
        w.edge(&deny, &behind, "SYNAPSE", Some(0.9)).await;

        let both = ProjectFilter {
            project_ids: vec![a.to_string(), b.to_string()],
        };
        let walk = w.same("note", &c, &focus_params(), &both).await;
        assert_eq!(
            walk.center.as_ref().unwrap().consent,
            SharingConsent::ExplicitAllow
        );
        // the stored consents come back as the mock's
        let got: HashMap<_, _> = walk
            .nodes
            .iter()
            .map(|n| (n.node.id.clone(), n.consent))
            .collect();
        assert_eq!(got[&deny], SharingConsent::ExplicitDeny);
        assert_eq!(got[&unset], SharingConsent::NotSet);
        assert_eq!(got[&allow], SharingConsent::ExplicitAllow);

        // the expected readable set, from the pure walk and the documented rule
        let mem = w.mem("note", &c, &focus_params(), &both);
        let sess = Some(a.to_string());
        let readable: Vec<_> = mem
            .nodes
            .iter()
            .filter(|n| n.project_id == sess || n.consent == SharingConsent::ExplicitAllow)
            .cloned()
            .collect();
        let (expected, _) = select_nodes(&c, &readable, &mem.edges, FOCUS_BUDGET);
        let expected_ids: Vec<_> = expected.iter().map(|n| n.id.clone()).collect();

        let proj = scope_project(a, "a");
        for policy in [None, Some(SharingMode::Auto), Some(SharingMode::Manual)] {
            if let Some(mode) = policy {
                store
                    .update_sharing_policy(
                        b,
                        &SharingPolicy {
                            enabled: true,
                            mode,
                            min_shareability_score: 0.0,
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap();
            }
            let anc = anchor(
                AnchorTargetType::Note,
                &c,
                AnchorRole::Focus,
                AnchorState::Live,
            );
            let scope = resolve_anchors(store, Some(proj.clone()), vec![anc], SECRET)
                .await
                .unwrap();
            let got = &scope.anchors[0];
            assert!(got.cross_project);
            let got_ids: Vec<_> = got.expansion.nodes.iter().map(|n| n.id.clone()).collect();
            assert_eq!(got_ids, expected_ids, "policy {policy:?}");
            assert!(got_ids.contains(&allow) && got_ids.contains(&local_deny));
            assert!(got_ids.contains(&local_unset));
            for hidden in [&deny, &unset, &behind] {
                assert!(!got_ids.contains(hidden), "{hidden} must be refused");
            }
            // `deny`, `unset` and the project node B (no consent of its own) are
            // refused; `behind` is pruned with them
            assert_eq!(got.expansion.denied.len(), 3);
            assert_eq!(scope.extra_projects, vec![b]);
            let text = format!(
                "{:?}{}{}",
                scope,
                render_anchor_map(&scope, "k"),
                render_live_block(&scope)
            );
            assert!(!text.contains("SECRET"));
        }
        w.cleanup().await;
    }

    // (3) hierarchy v1 ---------------------------------------------------------

    #[tokio::test]
    async fn hierarchy_v1_is_file_project_workspace_and_workspace_component_project() {
        let Some(e) = env().await else { return };
        // the table: exactly these four steps, nothing else
        let kinds = [
            "file",
            "function",
            "note",
            "project",
            "workspace",
            "component",
            "module",
            "feature",
            "feature_graph",
            "plan",
        ];
        let mut allowed = BTreeSet::new();
        for f in kinds {
            for t in kinds {
                if hierarchy_step_allowed(f, t) {
                    allowed.insert((f, t));
                }
            }
        }
        let table: BTreeSet<_> = HIERARCHY_V1.iter().copied().collect();
        assert_eq!(allowed, table);
        assert_eq!(table.len(), 4);
        for (f, t) in [
            ("module", "component"),
            ("feature", "component"),
            ("feature_graph", "component"),
        ] {
            assert!(!hierarchy_step_allowed(f, t));
        }

        let mut w = World::new(&e);
        let (a, b) = (w.project("a").await, w.project("b").await);
        let f = w.path("h.rs");
        w.node("file", &f, NodeProps::default(), Some(a), true, None)
            .await;
        w.edge(&a.to_string(), &f, "CONTAINS", None).await;
        let fg = w.uid();
        w.node(
            "feature_graph",
            &fg,
            NodeProps {
                name: Some("feature".into()),
                ..Default::default()
            },
            Some(a),
            true,
            None,
        )
        .await;
        w.edge(&fg, &f, "INCLUDES_ENTITY", None).await;
        let ws = w.uid();
        w.node(
            "workspace",
            &ws,
            NodeProps {
                name: Some("ws".into()),
                ..Default::default()
            },
            None,
            false,
            None,
        )
        .await;
        w.edge(&a.to_string(), &ws, "BELONGS_TO_WORKSPACE", None)
            .await;
        w.edge(&b.to_string(), &ws, "BELONGS_TO_WORKSPACE", None)
            .await;
        // Component and Module: real labels, not entity kinds of the walk
        let (comp, module) = (w.uid(), w.uid());
        w.run_cypher(
            "CREATE (c:Component {id: $c, h_run: $run}) CREATE (m:Module {id: $m, h_run: $run}) \
             WITH c, m \
             MATCH (w:Workspace {id: $ws}) MATCH (fg:FeatureGraph {id: $fg}) \
             MATCH (pb:Project {id: $pb}) MATCH (f:File {path: $f}) \
             CREATE (w)-[:HAS_COMPONENT {h_run: $run}]->(c) \
             CREATE (fg)-[:BELONGS_TO {h_run: $run}]->(c) \
             CREATE (m)-[:BELONGS_TO {h_run: $run}]->(c) \
             CREATE (m)-[:CONTAINS {h_run: $run}]->(f) \
             CREATE (c)-[:CONTAINS {h_run: $run}]->(pb)",
            vec![
                ("c", V::S(comp.clone())),
                ("m", V::S(module.clone())),
                ("ws", V::S(ws.clone())),
                ("fg", V::S(fg.clone())),
                ("pb", V::S(b.to_string())),
                ("f", V::S(f.clone())),
            ],
        )
        .await;

        let both = ProjectFilter {
            project_ids: vec![a.to_string(), b.to_string()],
        };
        // from the feature: the file and its project, never the component, the
        // module, the workspace (no owner) nor project B
        let from_fg = w.same("feature_graph", &fg, &params(3, 12), &both).await;
        let got = ids(&from_fg);
        assert_eq!(got, BTreeSet::from([f.clone(), a.to_string()]));
        assert!(!got.contains(&comp) && !got.contains(&module) && !got.contains(&ws));
        assert!(!got.contains(&b.to_string()));
        // from the file: project A is a leaf, the workspace above it is not reached
        let from_f = w.same("file", &f, &params(3, 12), &both).await;
        assert_eq!(ids(&from_f), BTreeSet::from([fg.clone(), a.to_string()]));
        // from the workspace (centre): only the projects, never through a component
        let from_ws = w.same("workspace", &ws, &params(3, 12), &both).await;
        assert_eq!(
            ids(&from_ws),
            BTreeSet::from([a.to_string(), b.to_string()])
        );
        w.cleanup().await;
    }

    // (4) owners of Plan / Task / Step -------------------------------------------

    #[tokio::test]
    async fn plan_task_step_owner_comes_from_has_plan_has_task_has_step() {
        let Some(e) = env().await else { return };
        let mut w = World::new(&e);
        let (a, b) = (w.project("a").await, w.project("b").await);
        let titled = |t: &str| NodeProps {
            title: Some(t.into()),
            status: Some("Pending".into()),
            priority: Some(50.0),
            ..Default::default()
        };
        // no `project_id` on any of them: only the edges say who owns them
        let (plan, task, step) = (w.uid(), w.uid(), w.uid());
        w.node("plan", &plan, titled("plan A"), Some(a), false, None)
            .await;
        w.node("task", &task, titled("task A"), Some(a), false, None)
            .await;
        w.node("step", &step, titled("step A"), Some(a), false, None)
            .await;
        w.edge(&a.to_string(), &plan, "HAS_PLAN", None).await;
        w.edge(&plan, &task, "HAS_TASK", None).await;
        w.edge(&task, &step, "HAS_STEP", None).await;
        // a plan of B depending on the task of A, and an ownerless task
        let (plan_b, task_b) = (w.uid(), w.uid());
        w.node("plan", &plan_b, titled("plan B"), Some(b), false, None)
            .await;
        w.node("task", &task_b, titled("task B"), Some(b), false, None)
            .await;
        w.edge(&b.to_string(), &plan_b, "HAS_PLAN", None).await;
        w.edge(&plan_b, &task_b, "HAS_TASK", None).await;
        let orphan = w.uid();
        w.node("task", &orphan, titled("orphan"), None, false, None)
            .await;
        w.edge(&task, &task_b, "DEPENDS_ON", None).await;
        w.edge(&task, &orphan, "DEPENDS_ON", None).await;

        let only_a = ProjectFilter::only(a.to_string());
        let walk = w.same("plan", &plan, &params(3, 12), &only_a).await;
        assert_eq!(
            ids(&walk),
            BTreeSet::from([task.clone(), step.clone(), a.to_string()])
        );
        for n in &walk.nodes {
            assert_eq!(
                n.project_id.as_deref(),
                Some(a.to_string().as_str()),
                "{}",
                n.node.id
            );
        }
        // every kind as the CENTRE: owner resolved, not null
        for (ty, id) in [("plan", &plan), ("task", &task), ("step", &step)] {
            let c = w.same(ty, id, &params(1, 5), &only_a).await.center.unwrap();
            assert_eq!(
                c.project_id.as_deref(),
                Some(a.to_string().as_str()),
                "{ty}"
            );
        }
        // inter-project: B's task and the ownerless task are refused even with {A, B}? B yes, orphan no
        let both = ProjectFilter {
            project_ids: vec![a.to_string(), b.to_string()],
        };
        let wide = w.same("task", &task, &params(1, 12), &both).await;
        assert!(ids(&wide).contains(&task_b));
        assert!(
            !ids(&wide).contains(&orphan),
            "ownerless node must be refused"
        );
        let narrow = w.same("task", &task, &params(1, 12), &only_a).await;
        assert!(!ids(&narrow).contains(&task_b) && !ids(&narrow).contains(&orphan));

        // an ownerless centre: owner null, the anchor is excluded (OwnerUnknown)
        let c = w
            .same("task", &orphan, &params(1, 5), &only_a)
            .await
            .center
            .unwrap();
        assert_eq!(c.project_id, None);
        let store: &dyn GraphStore = &e.client;
        let scope = resolve_anchors(
            store,
            Some(scope_project(a, "a")),
            vec![
                anchor(
                    AnchorTargetType::Plan,
                    &plan,
                    AnchorRole::Focus,
                    AnchorState::Live,
                ),
                anchor(
                    AnchorTargetType::Task,
                    &orphan,
                    AnchorRole::Focus,
                    AnchorState::Live,
                ),
            ],
            SECRET,
        )
        .await
        .unwrap();
        assert_eq!(scope.anchors.len(), 1);
        assert_eq!(scope.anchors[0].project_id, Some(a));
        assert_eq!(scope.excluded.len(), 1);
        assert_eq!(
            scope.excluded[0].reason,
            ExclusionReason::ConsentDenied(ReadDenial::OwnerUnknown)
        );
        // global note as stored by the real store (`project_id = ''`): same verdict
        let g = Note::new(None, NoteType::Observation, "global".into(), "t3bis".into());
        store.create_note(&g).await.unwrap();
        w.run_cypher(
            "MATCH (n:Note {id: $id}) SET n.h_run = $run",
            vec![("id", V::S(g.id.to_string()))],
        )
        .await;
        let scope = resolve_anchors(
            store,
            Some(scope_project(a, "a")),
            vec![anchor(
                AnchorTargetType::Note,
                &g.id.to_string(),
                AnchorRole::Focus,
                AnchorState::Live,
            )],
            SECRET,
        )
        .await
        .unwrap();
        assert_eq!(
            scope
                .excluded
                .iter()
                .map(|x| x.reason.clone())
                .collect::<Vec<_>>(),
            vec![ExclusionReason::ConsentDenied(ReadDenial::OwnerUnknown)]
        );
        w.cleanup().await;
    }

    // (5) end to end -------------------------------------------------------------

    #[tokio::test]
    async fn resolve_session_scope_end_to_end_on_the_real_store() {
        let Some(e) = env().await else { return };
        let store: &dyn GraphStore = &e.client;
        let mut w = World::new(&e);
        let (a, b) = (w.project("alpha").await, w.project("beta").await);
        let slug_a = format!("alpha-{}", w.run);

        let n0 = w.note(a, "note vivante", 0.9).await;
        let near = w.note(a, "voisin local", 0.8).await;
        let far = w.note(b, "SECRET beta", 0.9).await;
        w.edge(&n0, &near, "SYNAPSE", Some(0.9)).await;
        w.edge(&n0, &far, "SYNAPSE", Some(0.9)).await;
        let file = w.path("nouveau/f.rs");
        w.node("file", &file, NodeProps::default(), Some(a), true, None)
            .await;
        w.edge(&n0, &file, "LINKED_TO", None).await;

        let sid = Uuid::new_v4();
        let mut s = node(sid);
        s.project_slug = Some(slug_a.clone());
        store.create_chat_session(&s).await.unwrap();
        w.sessions.push(sid);

        let add = |t, id: &str, snap: Option<&str>| {
            let mut n = NewAnchor::new(t, id, [AnchorRole::Mention], AnchorActor::User, "h");
            n.snapshot_path = snap.map(String::from);
            n
        };
        let live = store
            .add_anchor(sid, add(AnchorTargetType::Note, &n0, None))
            .await
            .unwrap();
        let moved = store
            .add_anchor(sid, add(AnchorTargetType::File, &file, Some("ancien/f.rs")))
            .await
            .unwrap();
        let dangling = store
            .add_anchor(sid, add(AnchorTargetType::Note, &w.uid(), None))
            .await
            .unwrap();
        let refused = store
            .add_anchor(sid, add(AnchorTargetType::Project, &b.to_string(), None))
            .await
            .unwrap();
        for (an, st) in [
            (&moved, AnchorState::Moved),
            (&dangling, AnchorState::Dangling),
        ] {
            store
                .apply_anchor_op(
                    sid,
                    AnchorOp::SetState {
                        anchor_id: an.id,
                        state: st,
                        expected_version: None,
                        by: AnchorActor::System,
                        actor: "h".into(),
                    },
                )
                .await
                .unwrap();
        }

        let scope = resolve_session_scope(store, sid, SECRET).await.unwrap();
        assert_eq!(scope.project.as_ref().unwrap().id, a);
        assert_eq!(scope.anchors.len(), 2, "{scope:#?}");
        let by_id: HashMap<_, _> = scope.anchors.iter().map(|x| (x.anchor_id, x)).collect();
        let l = by_id[&live.id];
        assert_eq!(l.title, "note vivante");
        let tl: Vec<_> = l.expansion.nodes.iter().map(|n| n.title.as_str()).collect();
        assert_eq!(tl, vec!["voisin local", "f.rs"]);
        let m = by_id[&moved.id];
        assert_eq!(m.state, AnchorState::Moved);
        assert_eq!(
            m.moved_note.as_deref(),
            Some("déplacé de ancien/f.rs vers f.rs")
        );
        assert_eq!(scope.broken.len(), 1);
        assert_eq!(scope.broken[0].anchor_id, dangling.id);
        assert_eq!(scope.excluded.len(), 1);
        assert_eq!(scope.excluded[0].anchor_id, refused.id);
        assert_eq!(
            scope.excluded[0].reason,
            ExclusionReason::ConsentDenied(ReadDenial::NoPolicy)
        );
        assert!(scope.extra_projects.is_empty());
        let text = format!(
            "{:?}{}{}",
            scope,
            render_anchor_map(&scope, "k"),
            render_live_block(&scope)
        );
        assert!(!text.contains("SECRET") && !text.contains("beta-"));

        // a workspace holding the session project and the refused one: admitted, B stays closed
        let ws = project_orchestrator::neo4j::models::WorkspaceNode {
            id: Uuid::new_v4(),
            name: "Hub".into(),
            slug: format!("hub-{}", w.run),
            description: None,
            created_at: Utc::now(),
            updated_at: None,
            metadata: serde_json::json!({}),
        };
        store.create_workspace(&ws).await.unwrap();
        w.run_cypher(
            "MATCH (n:Workspace {id: $id}) SET n.h_run = $run",
            vec![("id", V::S(ws.id.to_string()))],
        )
        .await;
        store.add_project_to_workspace(ws.id, a).await.unwrap();
        store.add_project_to_workspace(ws.id, b).await.unwrap();
        let scope = resolve_anchors(
            store,
            Some(scope_project(a, &slug_a)),
            vec![anchor(
                AnchorTargetType::Workspace,
                &ws.id.to_string(),
                AnchorRole::Focus,
                AnchorState::Live,
            )],
            SECRET,
        )
        .await
        .unwrap();
        assert_eq!(scope.anchors[0].title, "Hub");
        assert!(scope.extra_projects.is_empty());

        // a session with a project and no anchor: nothing, no notice
        let s2 = Uuid::new_v4();
        let mut n2 = node(s2);
        n2.project_slug = Some(slug_a);
        store.create_chat_session(&n2).await.unwrap();
        w.sessions.push(s2);
        let scope = resolve_session_scope(store, s2, SECRET).await.unwrap();
        assert!(scope.is_empty() && scope.notices.is_empty() && scope.project.is_some());
        // no anchor, no project: the notice
        let s3 = Uuid::new_v4();
        store.create_chat_session(&node(s3)).await.unwrap();
        w.sessions.push(s3);
        let scope = resolve_session_scope(store, s3, SECRET).await.unwrap();
        assert!(scope.is_empty() && scope.project.is_none());
        assert_eq!(scope.notices, vec![NOTICE_NO_CONTEXT.to_string()]);
        w.cleanup().await;
    }

    // (6) token estimate -----------------------------------------------------------

    const FRENCH: [&str; 50] = [
        "Décision : migrer l'authentification vers des jetons à durée limitée",
        "Étape 3 — vérifier la cohérence des identifiants après la migration",
        "Réécrire le résolveur d'ancres pour qu'il respecte le périmètre du projet",
        "Pourquoi le consentement explicite prime sur la politique de partage ?",
        "Problème de concurrence sur la table des sessions : verrou trop large",
        "À faire : documenter l'ordre des opérations dans la clôture du tour",
        "Le plan d'intégration continue est devenu plus rapide, sans perdre de garanties",
        "Évaluer l'impact du déplacement d'un fichier sur les ancres existantes",
        "Piège : les nœuds sans propriétaire sont refusés entre projets",
        "Réparer la génération des identifiants opaques pour les nœuds refusés",
        "Tâche terminée : les voisins étrangers n'évincent plus les voisins locaux",
        "Configurer l'espace de travail et vérifier les droits des composants",
        "Synthèse de la réunion : priorités du trimestre et dépendances critiques",
        "Compatibilité ascendante des schémas : ne jamais renommer une propriété",
        "Revoir la gestion des erreurs réseau lors de la reconnexion au serveur",
        "Où est stocké le résumé généré après la compaction du contexte ?",
        "Accélérer l'indexation des fichiers modifiés depuis le dernier commit",
        "Règle : un test rouge d'abord, puis le correctif minimal",
        "Les préférences de l'utilisateur sont conservées entre deux sessions",
        "Détecter les ancres orphelines après la suppression d'une branche",
        "Épisode : la base de développement ne doit jamais être touchée par les tests",
        "Évolution du modèle : séparer l'identité de l'acteur et son rôle",
        "Prévoir un mécanisme de reprise après une interruption brutale du processus",
        "Gérer les caractères accentués dans les titres tronqués à quatre-vingts signes",
        "Mettre à jour la documentation du protocole de synchronisation des données",
        "Estimation du coût en jetons : à mesurer plutôt qu'à supposer",
        "Vérifier que la hiérarchie conteneur reste limitée aux étapes autorisées",
        "Qui peut lire quoi ? Tableau de décision du prédicat de lecture",
        "Étude de cas : un fichier relié à vingt-cinq notes d'un autre projet",
        "Réduire la latence du premier octet lors du démarrage à froid",
        "Dans une conversation ancrée sur un fichier, le contexte provient du graphe : le résolveur parcourt d'abord les voisins directs du fichier, puis ceux de ces voisins, en appliquant le filtre de projet avant la limite d'éventail, afin qu'aucun nœud étranger ne prenne la place d'un nœud local.",
        "Note de conception : le consentement protège le partage au-delà du projet, jamais la lecture locale. « Ne pas partager » ne veut pas dire « ne pas utiliser ». C'est pourquoi un voisin du même projet est toujours lisible, même marqué comme refusé, alors qu'un voisin d'un autre projet exige un accord explicite.",
        "Retour d'expérience sur la migration des ancres : les sessions existantes reçoivent une ancre de projet inférée, réversible, journalisée ; deux exécutions successives donnent le même état, et l'annulation ne retire que les ancres créées par la migration elle-même.",
        "Le budget d'un rôle borne la profondeur, le nombre de nœuds et le nombre de jetons ; un rôle de focalisation obtient deux niveaux et douze nœuds, une mention un seul niveau et cinq nœuds, une origine uniquement son titre, sans aucun voisin.",
        "Observation : lorsque la politique du projet propriétaire est illisible, toute lecture entre projets est refusée et un avertissement est journalisé ; on ne devine jamais une permission manquante, car l'erreur serait silencieuse et invisible pour l'utilisateur.",
        "Les identifiants opaques sont dérivés par HMAC avec un secret propre à la session : stables pendant la conversation, impossibles à relier à l'extérieur, et ne révélant rien de l'identifiant réel du nœud refusé ni de son type ni de son titre.",
        "Spécification du bloc vivant : court, placé en fin de prompt, il liste les ancres actives, signale les ancres cassées en une seule ligne chacune et n'affiche jamais le nombre de nœuds refusés, afin de ne pas laisser fuiter leur existence.",
        "Compte rendu de revue : le filtre de propriétaire est évalué avant la limite par nœud ; sans cela, vingt-cinq voisins étrangers plus saillants chassent les cinq voisins locaux, et le résultat final serait vide alors qu'il existe pourtant du contexte lisible.",
        "Procédure d'incident : en cas de dérive entre la requête Cypher et le double en mémoire, corriger la requête, écrire d'abord un test qui échoue, puis décrire l'écart dans la description de la demande de fusion.",
        "Le déplacement d'un fichier produit une ancre déplacée : elle est résolue une fois vers sa cible actuelle, avec la mention « déplacé de X vers Y », tandis qu'une ancre pendante ou archivée ne produit qu'une ligne sans contenu.",
        "Gestion des identités : l'acteur humain, l'agent et le système n'ont pas les mêmes droits sur les rôles d'ancre ; un agent ne peut ni créer ni retirer une ancre de focalisation, ce qui évite qu'une conversation automatisée ne détourne le contexte.",
        "Performance : la requête de saut est bornée, chaque nœud du front apporte au plus un nombre fixe de relations triées par poids, et le front lui-même est plafonné ; aucun motif à longueur variable n'est utilisé afin de garder un coût prévisible.",
        "Histoire du projet : la première version du résolveur s'appuyait sur le répertoire courant de la session, ce qui donnait des contextes erronés dès qu'on ouvrait un terminal ailleurs ; l'ancrage explicite a remplacé cette heuristique fragile.",
        "Conseil pratique : toujours nommer le conteneur jetable avec un préfixe reconnaissable, le supprimer à la sortie même en cas d'échec, et vérifier ensuite qu'il ne reste aucun conteneur ni processus en arrière-plan sur la machine partagée.",
        "Les notes de type « piège » sont remontées en priorité lorsqu'elles sont liées à un fichier ancré, car elles évitent à l'agent de refaire une erreur déjà documentée ; leur énergie décroît pourtant si personne ne les confirme.",
        "Bilan de la semaine : trois demandes de fusion livrées, une régression corrigée dans la détection des modèles sans outils, et un audit de dépendances qui a imposé la mise à jour d'une bibliothèque de lecture des documents PDF.",
        "Question ouverte : faut-il autoriser l'ancre d'un autre projet lorsque le mode de partage est automatique ? Aujourd'hui elle est toujours refusée, car un conteneur n'a pas de consentement propre et aucun score ne peut être calculé.",
        "Résumé technique : un fichier appartient à un projet, un projet à un espace de travail, et un espace de travail contient des composants qui désignent des projets ; aucune autre arête de hiérarchie n'est suivie dans la première version.",
        "Les accents posent un problème de comptage : un caractère accentué occupe un seul caractère mais souvent plusieurs octets, et le découpage en jetons du modèle ne suit ni les caractères ni les mots, d'où une estimation toujours approximative.",
        "Pour finir, rappelons que la valeur d'un contexte ne tient pas à sa taille mais à sa pertinence : mieux vaut douze nœuds bien choisis qu'une centaine de voisins sans lien réel avec la question posée par l'utilisateur.",
    ];

    #[tokio::test]
    async fn token_estimate_on_accented_french_notes() {
        let Some(e) = env().await else { return };
        let mut w = World::new(&e);
        let a = w.project("fr").await;
        let hub = w.path("hub.rs");
        w.node("file", &hub, NodeProps::default(), Some(a), true, None)
            .await;
        for t in FRENCH {
            let n = w.note(a, t, 0.5).await;
            w.edge(&n, &hub, "LINKED_TO", None).await;
        }
        let p = NeighborhoodParams {
            depth: 1,
            min_weight: 0.0,
            limit: 60,
            layers: Layer::ALL.to_vec(),
            fanout: 60,
            frontier_cap: 60,
        };
        let walk = w
            .same("file", &hub, &p, &ProjectFilter::only(a.to_string()))
            .await;
        assert_eq!(
            walk.nodes
                .iter()
                .filter(|n| n.node.node_type == "note")
                .count(),
            50
        );

        // what the resolver counts is the label it reads back (first line, cut at 80)
        let reference = |s: &str| (s.split_whitespace().count() as f64 * 1.3).ceil();
        let (mut est_sum, mut ref_sum, mut est_full, mut ref_full) = (0.0, 0.0, 0.0, 0.0);
        let (mut lo, mut hi) = (f64::MAX, 0.0_f64);
        let mut chars_per_word = 0.0;
        for n in walk.nodes.iter().filter(|n| n.node.node_type == "note") {
            let (est, rf) = (est_tokens(&n.node.label) as f64, reference(&n.node.label));
            est_sum += est;
            ref_sum += rf;
            lo = lo.min(est / rf);
            hi = hi.max(est / rf);
            chars_per_word += n.node.label.chars().count() as f64
                / n.node.label.split_whitespace().count() as f64;
        }
        for t in FRENCH {
            est_full += est_tokens(t) as f64;
            ref_full += reference(t);
        }
        eprintln!(
            "TOKENS labels: est(chars/4)={est_sum} ref(words*1.3)={ref_sum} ratio={:.3} \
             per-label min={lo:.2} max={hi:.2} | full texts: est={est_full} ref={ref_full} \
             ratio={:.3} | mean chars/word={:.2}",
            est_sum / ref_sum,
            est_full / ref_full,
            chars_per_word / 50.0
        );
        // gross error only: the estimate stays within a factor 2 of the reference
        assert!((0.5..=2.0).contains(&(est_sum / ref_sum)));
        assert!((0.5..=2.0).contains(&(est_full / ref_full)));
        w.cleanup().await;
    }
}
