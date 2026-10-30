//! Le soi comme SUJET, contre un VRAI Neo4j.
//!
//! Ces tests existent parce que la couverture le réclamait, et elle avait raison : le Cypher
//! du soi avait été validé à la main (un `cypher-shell`, une requête, un compte), mais le
//! **code Rust qui l'entoure** — quels paramètres, quel retour, quelle arête en sort — n'avait
//! jamais tourné. Une requête juste appelée par une fonction fausse reste un graphe faux.
//!
//! Ils tournent dans le job Coverage, qui a un Neo4j : `HARNESS_NEO4J_REQUIRED=1` y est posé,
//! donc une base absente fait échouer au lieu d'uploader un 0 %. Sans ce drapeau, un poste
//! sans base saute les tests en le disant.
//!
//! Chaque test crée son propre projet (slug unique) et le supprime : le job partage une seule
//! base entre plusieurs suites.

use chrono::Utc;
use neo4rs::{query, Graph};
use project_orchestrator::graph::rules;
use project_orchestrator::neo4j::client::Neo4jClient;
use project_orchestrator::neo4j::models::{ChatSessionNode, ProjectNode};

use project_orchestrator::notes::models::{Note, NoteType};
use uuid::Uuid;

/// These tests share SINGLETON nodes (the durable `:Agent` root) and the same Neo4j: run in
/// parallel they deadlock on it — measured in CI, `Neo.TransientError.Transaction.DeadlockDetected`
/// on `MERGE (:Agent {id: <root>})`. The suite is therefore serial with itself. Cross-suite
/// parallelism is unaffected: the coverage job runs each `--test` binary one after the other.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn serial() -> tokio::sync::MutexGuard<'static, ()> {
    SERIAL.lock().await
}

struct Env {
    client: Neo4jClient,
    raw: Graph,
    project: ProjectNode,
}

async fn env(label: &str) -> Option<Env> {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:17687".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into());

    let connected = async {
        let client = Neo4jClient::new(&uri, &user, &password).await.ok()?;
        let raw = Graph::new(&uri, &user, &password).await.ok()?;
        Some((client, raw))
    }
    .await;

    let Some((client, raw)) = connected else {
        assert!(
            std::env::var("HARNESS_NEO4J_REQUIRED").is_err(),
            "HARNESS_NEO4J_REQUIRED is set but no Neo4j answers at {uri}"
        );
        eprintln!("Skipping self-graph tests: no Neo4j at {uri}");
        return None;
    };

    let slug = format!("selfgraph-{label}-{}", Uuid::new_v4().simple());
    let project = ProjectNode {
        id: Uuid::new_v4(),
        name: slug.clone(),
        slug: slug.clone(),
        root_path: format!("/tmp/{slug}"),
        description: None,
        created_at: Utc::now(),
        last_synced: None,
        analytics_computed_at: None,
        last_co_change_computed_at: None,
        default_note_energy: None,
        scaffolding_override: None,
        sharing_policy: None,
        profile: Default::default(),
        watch_enabled: false,
    };
    client.create_project(&project).await.unwrap();
    Some(Env {
        client,
        raw,
        project,
    })
}

/// Removes everything this test wrote: the project cascade takes its notes, and the agent
/// nodes are deleted by id because they belong to no project.
async fn cleanup(e: &Env) {
    e.client
        .delete_project(e.project.id, &e.project.name)
        .await
        .expect("the project's own cascade");
}

async fn drop_nodes(e: &Env, label: &str, ids: &[Uuid]) {
    let ids: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    e.raw
        .run(
            query(&format!(
                "MATCH (n:{label}) WHERE n.id IN $ids DETACH DELETE n"
            ))
            .param("ids", ids),
        )
        .await
        .expect("cleanup of standalone nodes");
}

fn a_note(written_by: &str) -> Note {
    let mut n = Note::new(
        None,
        NoteType::Guideline,
        "a foundation".into(),
        written_by.to_string(),
    );
    n.status = project_orchestrator::notes::NoteStatus::Active;
    n
}

#[tokio::test]
async fn a_write_makes_its_agent_a_node_and_gives_it_the_note() {
    let _serial = serial().await;
    // The edge is drawn at the write, by the real Cypher, not only reconstructed later.
    let Some(e) = env("authored").await else {
        return;
    };
    let agent = Uuid::new_v4();
    let note = a_note(&format!("agent:{agent}"));
    e.client.create_note(&note).await.unwrap();

    let mut rows = e
        .raw
        .execute(
            query(
                "MATCH (a:Agent {id: $agent})-[:AUTHORED]->(n:Note {id: $note}) \
                 RETURN count(*) AS n",
            )
            .param("agent", agent.to_string())
            .param("note", note.id.to_string()),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().expect("a row");
    assert_eq!(
        row.get::<i64>("n").unwrap_or(0),
        1,
        "the author owns its note"
    );

    let node = e.client.find_agent(agent).await.unwrap();
    assert!(node.is_some(), "and the author exists as a node");

    drop_nodes(&e, "Agent", &[agent]).await;
    e.client.delete_note(note.id).await.unwrap();
    cleanup(&e).await;
}

#[tokio::test]
async fn the_backfill_rebuilds_the_attribution_and_says_what_it_did() {
    let _serial = serial().await;
    // The same edges the write draws, rebuilt from what is already written: one path.
    let Some(e) = env("backfill").await else {
        return;
    };
    let agent = Uuid::new_v4();
    let note = a_note(&format!("agent:{agent}"));
    // Written without the write hook, so the edge is genuinely missing: this is the state
    // a database predating the feature is in.
    e.raw
        .run(
            query(
                "CREATE (n:Note {id: $id, note_type: 'Guideline', status: 'Active', \
                 importance: 'High', content: 'a foundation', created_by: $by, \
                 project_id: $pid, created_at: datetime(), \
                 last_confirmed_at: datetime(), energy: 1.0, activation_count: 0})",
            )
            .param("id", note.id.to_string())
            .param("by", format!("agent:{agent}"))
            .param("pid", e.project.id.to_string()),
        )
        .await
        .unwrap();

    let first = e.client.backfill_agents().await.unwrap();
    assert!(first.from_notes >= 1, "{first:?}");
    assert!(first.authored >= 1, "the missing edge was drawn: {first:?}");
    assert!(first.after >= first.before, "{first:?}");

    let second = e.client.backfill_agents().await.unwrap();
    assert_eq!(second.authored, 0, "idempotent: {second:?}");

    drop_nodes(&e, "Agent", &[agent]).await;
    e.client.delete_note(note.id).await.unwrap();
    cleanup(&e).await;
}

#[tokio::test]
async fn what_a_subject_knows_is_bounded_and_carries_its_measure() {
    let _serial = serial().await;
    let Some(e) = env("knows").await else {
        return;
    };
    // A FRESH subject, not the shared root: this database is reused by every suite, and a
    // previous run that failed before its cleanup leaves edges behind. What is measured here
    // is the mechanism, and the mechanism must not depend on a clean database.
    let self_id = Uuid::new_v4();
    // As production does (`agent-knows-core`): the subject exists before it is given
    // anything. `link_agent_knows` MATCHes both ends — a missing subject is a silent no-op,
    // which is the trap this line documents.
    e.client.ensure_agent(self_id, None).await.unwrap();
    let before = e.client.agent_knows(self_id, 10).await.unwrap();
    assert_eq!(before.available, 0, "a fresh subject knows nothing");
    let prov = rules::Provenance::new("agent-knows-core", "3 portées lues, 3 notes épinglées");

    let mut ids = Vec::new();
    for i in 0..3 {
        let mut n = a_note("api");
        n.content = format!("belief {i}");
        n.project_id = Some(e.project.id);
        e.client.create_note(&n).await.unwrap();
        assert!(
            e.client
                .link_agent_knows(self_id, n.id, &prov)
                .await
                .unwrap(),
            "first link creates the edge"
        );
        assert!(
            !e.client
                .link_agent_knows(self_id, n.id, &prov)
                .await
                .unwrap(),
            "a second link attests, it does not create"
        );
        ids.push(n.id);
    }

    let known = e.client.agent_knows(self_id, 2).await.unwrap();
    assert_eq!(known.notes.len(), 2, "bounded");
    assert_eq!(known.available, 3, "and the total is published");
    assert_eq!(known.withheld(), 1, "« la part lue est dite »");

    // And the bound is what limits, not luck: unbounded, all three come back.
    let all = e.client.agent_knows(self_id, 10).await.unwrap();
    assert_eq!(all.notes.len(), 3);
    assert_eq!(all.withheld(), 0);

    // The provenance is ON the edge, not in a table beside it.
    let mut rows = e
        .raw
        .execute(
            query(
                "MATCH (:Agent {id: $agent})-[r:KNOWS]->(:Note {id: $note}) \
                 RETURN r.rule AS rule, r.measure AS measure, r.at AS at",
            )
            .param("agent", self_id.to_string())
            .param("note", ids[0].to_string()),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().expect("an edge");
    assert_eq!(row.get::<String>("rule").unwrap(), "agent-knows-core");
    assert!(row.get::<String>("measure").unwrap().contains('3'));
    assert!(!row.get::<String>("at").unwrap().is_empty());

    for id in &ids {
        e.client.delete_note(*id).await.unwrap();
    }
    drop_nodes(&e, "Agent", &[self_id]).await;
    cleanup(&e).await;
}

#[tokio::test]
async fn a_knows_edge_to_a_missing_note_draws_nothing() {
    let _serial = serial().await;
    let Some(e) = env("knows-missing").await else {
        return;
    };
    let self_id = e.client.agent_identity_namespace().await.unwrap();
    let drawn = e
        .client
        .link_agent_knows(
            self_id,
            Uuid::new_v4(),
            &rules::Provenance::new("agent-knows-core", "measured"),
        )
        .await
        .unwrap();
    assert!(!drawn, "no target, no edge — and no error either");
    cleanup(&e).await;
}

#[tokio::test]
async fn what_a_subject_applies_becomes_a_uses_edge_with_its_provenance() {
    let _serial = serial().await;
    let Some(e) = env("uses").await else {
        return;
    };
    let agent = Uuid::new_v4();
    let persona = Uuid::new_v4();
    e.raw
        .run(query("CREATE (:Persona {id: $id, name: 'Pentest'})").param("id", persona.to_string()))
        .await
        .unwrap();

    let outcome = rules::run_with(
        &e.client,
        rules::AGENT_USES_ACTIVATED.name,
        rules::Measurement::Activation {
            session: Uuid::new_v4(),
            agent,
            personas: vec![persona],
            skills: vec![],
        },
    )
    .await
    .unwrap();
    assert_eq!(outcome.written, 1, "{outcome:?}");

    let mut rows = e
        .raw
        .execute(
            query(
                "MATCH (:Agent {id: $agent})-[r:USES]->(:Persona {id: $p}) \
                 RETURN r.rule AS rule, r.measure AS measure",
            )
            .param("agent", agent.to_string())
            .param("p", persona.to_string()),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().expect("a USES edge");
    assert_eq!(row.get::<String>("rule").unwrap(), "agent-uses-activated");
    assert!(row.get::<String>("measure").unwrap().contains("activés"));

    // A label that is not a bare identifier never reaches the query.
    let refused = e
        .client
        .link_agent_uses(
            agent,
            "Persona) DETACH DELETE (n",
            persona,
            &rules::Provenance::new("agent-uses-activated", "measured"),
        )
        .await;
    assert!(
        refused.is_err(),
        "an unusable label is refused, not escaped"
    );

    drop_nodes(&e, "Persona", &[persona]).await;
    drop_nodes(&e, "Agent", &[agent]).await;
    cleanup(&e).await;
}

#[tokio::test]
async fn a_session_welds_the_notes_it_really_had_in_context() {
    let _serial = serial().await;
    let Some(e) = env("coact").await else {
        return;
    };
    let session = Uuid::new_v4();
    let mut ids = Vec::new();
    for i in 0..3 {
        let mut n = a_note("api");
        n.content = format!("seen {i}");
        n.project_id = Some(e.project.id);
        e.client.create_note(&n).await.unwrap();
        ids.push(n.id);
    }

    let outcome = rules::run_with(
        &e.client,
        rules::SESSION_CO_ACTIVATION.name,
        rules::Measurement::SessionNotes {
            session,
            project: Some(e.project.id),
            notes: ids.clone(),
        },
    )
    .await
    .unwrap();
    assert_eq!(outcome.written, 6, "3 pairs, both directions: {outcome:?}");

    let mut rows = e
        .raw
        .execute(
            query(
                "MATCH (:Note {id: $a})-[s:SYNAPSE]->(:Note {id: $b}) \
                 RETURN s.rule AS rule, s.last_session AS session, s.measure AS measure",
            )
            .param("a", ids[0].to_string())
            .param("b", ids[1].to_string()),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().expect("a synapse");
    assert_eq!(row.get::<String>("rule").unwrap(), "session-co-activation");
    assert_eq!(row.get::<String>("session").unwrap(), session.to_string());
    assert!(row.get::<String>("measure").unwrap().contains('3'));

    // A session with one note, or without a project, welds nothing.
    let alone = rules::run_with(
        &e.client,
        rules::SESSION_CO_ACTIVATION.name,
        rules::Measurement::SessionNotes {
            session,
            project: Some(e.project.id),
            notes: vec![ids[0]],
        },
    )
    .await
    .unwrap();
    assert_eq!(alone.written, 0);

    let homeless = rules::run_with(
        &e.client,
        rules::SESSION_CO_ACTIVATION.name,
        rules::Measurement::SessionNotes {
            session,
            project: None,
            notes: ids.clone(),
        },
    )
    .await
    .unwrap();
    assert_eq!(homeless.written, 0);

    for id in &ids {
        e.client.delete_note(*id).await.unwrap();
    }
    cleanup(&e).await;
}

#[tokio::test]
async fn a_session_episode_belongs_to_the_agent_that_lived_it() {
    let _serial = serial().await;
    // The whole real path: the collector names the agent of the session, `save_session_episode`
    // writes the episode and draws `LIVED` in the same statement.
    let Some(e) = env("lived").await else {
        return;
    };
    let session = live_session(&e, "lived");
    let sid = session.id;
    e.client.create_chat_session(&session).await.unwrap();

    let episode = project_orchestrator::episodes::collector::collect_session_episode(
        &e.client,
        sid,
        "user_closed",
    )
    .await
    .unwrap()
    .expect("a session with a preview and messages leaves an episode");
    let agent = episode
        .session
        .as_ref()
        .and_then(|s| s.agent_id)
        .expect("the collector names the agent of the session");
    assert!(e.client.save_session_episode(&episode).await.unwrap());

    let mut rows = e
        .raw
        .execute(
            query(
                "MATCH (:Agent {id: $a})-[:LIVED]->(:SessionEpisode {id: $e}) RETURN count(*) AS n",
            )
            .param("a", agent.to_string())
            .param("e", episode.id.to_string()),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().expect("a row");
    assert_eq!(row.get::<i64>("n").unwrap_or(0), 1, "the agent lived it");

    // And the episode says how the session ended, not a constant.
    assert_eq!(
        episode.session.as_ref().unwrap().close_reason,
        "user_closed"
    );

    e.client.delete_chat_session(sid).await.unwrap();
    drop_nodes(&e, "Agent", &[agent]).await;
    cleanup(&e).await;
}

fn live_session(e: &Env, label: &str) -> ChatSessionNode {
    let id = Uuid::new_v4();
    ChatSessionNode {
        id,
        cli_session_id: None,
        project_slug: Some(e.project.slug.clone()),
        workspace_slug: None,
        cwd: format!("/tmp/{label}"),
        title: Some(label.into()),
        model: "m".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        message_count: 0,
        total_cost_usd: None,
        conversation_id: None,
        preview: Some("first message".into()),
        permission_mode: None,
        add_dirs: None,
        spawned_by: None,
        provider_id: Some("local".into()),
        routed_by: Some("request".into()),
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

#[tokio::test]
async fn the_catalog_says_what_the_graph_really_has() {
    let _serial = serial().await;
    let Some(e) = env("catalog").await else {
        return;
    };
    // One edge of a known shape, so the catalog has something true to report.
    let agent = Uuid::new_v4();
    let mut note = a_note("api");
    note.project_id = Some(e.project.id);
    e.client.create_note(&note).await.unwrap();
    // The subject must exist or `link_agent_knows` is a silent no-op — and a test that
    // believes otherwise proves nothing about the catalog.
    e.client.ensure_agent(agent, None).await.unwrap();
    e.client
        .link_agent_knows(
            agent,
            note.id,
            &rules::Provenance::new("agent-knows-core", "measured"),
        )
        .await
        .unwrap();

    let catalog = e.client.schema_catalog().await.unwrap();
    assert!(!catalog.labels.is_empty(), "the graph has labels");
    assert!(catalog.has_edge("KNOWS"), "KNOWS exists in this graph");
    assert!(
        catalog.volume("KNOWS").is_some(),
        "and its volume is measured, not assumed"
    );
    assert!(
        catalog.volume("NOT_A_REAL_TYPE").is_none(),
        "an unknown type has no volume — which is not zero"
    );
    for edge in catalog.causal_edges() {
        assert!(
            rules::Provenance::new("x", "y").is_accountable()
                && project_orchestrator::graph::schema::CAUSAL_TYPES.contains(&edge),
            "the causal family never leaks a foreign type: {edge}"
        );
    }

    e.client.delete_note(note.id).await.unwrap();
    drop_nodes(&e, "Agent", &[agent]).await;
    cleanup(&e).await;
}

#[tokio::test]
async fn a_causal_walk_follows_a_real_chain_and_refuses_a_stranger_label() {
    let _serial = serial().await;
    let Some(e) = env("walk").await else {
        return;
    };
    let task = Uuid::new_v4();
    let decision = Uuid::new_v4();
    e.raw
        .run(
            query(
                "CREATE (t:Task {id: $t, title: 'a task'}) \
                 CREATE (d:Decision {id: $d, description: 'a reason'}) \
                 CREATE (t)-[:INFORMED_BY {created_at: datetime()}]->(d)",
            )
            .param("t", task.to_string())
            .param("d", decision.to_string()),
        )
        .await
        .unwrap();

    let walk = e
        .client
        .causal_walk("Task", &task.to_string(), 2)
        .await
        .unwrap();
    assert!(walk.exists(), "the chain exists: {walk:?}");
    assert_eq!(walk.steps[0].edge, "INFORMED_BY");
    assert_eq!(walk.steps[0].label, "Decision");
    assert!(!walk.truncated, "the chain ends before the bound");

    // An anchor that does not exist is an empty walk — « this chain does not exist » — and
    // never an error, which would read as « I could not look ».
    let absent = e
        .client
        .causal_walk("Task", &Uuid::new_v4().to_string(), 2)
        .await
        .unwrap();
    assert!(!absent.exists());

    // A label that is not an anchor never reaches the query.
    let refused = e.client.causal_walk("Task) DETACH DELETE (n", "x", 2).await;
    assert!(refused.is_err());

    // A chain longer than the bound must say it was cut: a walk stopped by its own depth
    // limit cannot read as « the chain ends here ». Built in one statement — a `MATCH` on a
    // node that does not exist yet silently creates nothing, which is how this test first
    // proved the opposite of what it claimed.
    let deep_t = Uuid::new_v4();
    let hops: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
    e.raw
        .run(
            query(
                "MERGE (t:Task {id: $t}) \
                 MERGE (a:Decision {id: $a}) MERGE (b:Decision {id: $b}) MERGE (c:Decision {id: $c}) \
                 MERGE (t)-[:PRECEDES]->(a) MERGE (a)-[:PRECEDES]->(b) MERGE (b)-[:PRECEDES]->(c)",
            )
            .param("t", deep_t.to_string())
            .param("a", hops[0].to_string())
            .param("b", hops[1].to_string())
            .param("c", hops[2].to_string()),
        )
        .await
        .unwrap();

    let cut = e
        .client
        .causal_walk("Task", &deep_t.to_string(), 1)
        .await
        .unwrap();
    assert!(cut.exists(), "the first hop exists: {cut:?}");
    assert_eq!(cut.steps.len(), 1, "one hop, the depth asked: {cut:?}");
    assert!(
        cut.truncated,
        "and the walk says it stopped on its bound: {cut:?}"
    );
    // A depth beyond the ceiling is clamped, never interpolated as given.
    let clamped = e
        .client
        .causal_walk("Task", &deep_t.to_string(), 99)
        .await
        .unwrap();
    assert_eq!(clamped.max_depth, 4);
    assert!(!clamped.truncated, "a 4-hop bound reaches the end: {clamped:?}");

    // An agent the graph does not have is absent, not invented.
    assert!(e.client.find_agent(Uuid::new_v4()).await.unwrap().is_none());

    drop_nodes(&e, "Task", &[task, deep_t]).await;
    drop_nodes(&e, "Decision", &hops).await;
    drop_nodes(&e, "Decision", &[decision]).await;
    cleanup(&e).await;
}

#[tokio::test]
async fn a_walk_cut_by_its_depth_says_it_was_cut() {
    // « La part lue est dite », appliqué à une chaîne : une marche arrêtée par sa propre
    // borne ne doit pas se lire « la chaîne s'arrête ici ». Deux sauts existent, la borne
    // n'en lit qu'un, et `truncated` le dit.
    let _serial = serial().await;
    let Some(e) = env("truncated").await else {
        return;
    };
    let task = Uuid::new_v4();
    let first = Uuid::new_v4();
    let second = Uuid::new_v4();
    e.raw
        .run(
            query(
                "CREATE (t:Task {id: $t, title: 'a task'}) \
                 CREATE (a:Decision {id: $a, description: 'first'}) \
                 CREATE (b:Decision {id: $b, description: 'second'}) \
                 CREATE (t)-[:INFORMED_BY]->(a) CREATE (a)-[:NEXT_DECISION]->(b)",
            )
            .param("t", task.to_string())
            .param("a", first.to_string())
            .param("b", second.to_string()),
        )
        .await
        .unwrap();

    let cut = e
        .client
        .causal_walk("Task", &task.to_string(), 1)
        .await
        .unwrap();
    assert_eq!(cut.steps.len(), 1, "the bound is one hop: {cut:?}");
    assert!(
        cut.truncated,
        "there is more to follow and the walk says so: {cut:?}"
    );

    let whole = e
        .client
        .causal_walk("Task", &task.to_string(), 3)
        .await
        .unwrap();
    assert_eq!(whole.steps.len(), 2, "both hops: {whole:?}");
    assert!(!whole.truncated);

    drop_nodes(&e, "Task", &[task]).await;
    drop_nodes(&e, "Decision", &[first, second]).await;
    cleanup(&e).await;
}

#[tokio::test]
async fn an_agent_that_does_not_exist_is_absent_not_an_error() {
    let _serial = serial().await;
    let Some(e) = env("noagent").await else {
        return;
    };
    assert!(e.client.find_agent(Uuid::new_v4()).await.unwrap().is_none());
    let known = Uuid::new_v4();
    e.client.ensure_agent(known, Some("m")).await.unwrap();
    let found = e.client.find_agent(known).await.unwrap();
    assert_eq!(found, Some(("model".to_string(), Some("m".to_string()))));
    drop_nodes(&e, "Agent", &[known]).await;
    cleanup(&e).await;
}

#[tokio::test]
async fn a_rule_that_needs_its_caller_says_so_instead_of_pretending() {
    let _serial = serial().await;
    // `run()` reads the graph. A rule whose measurement lives in memory is reported as
    // not runnable there, with a reason — never silently skipped, never faked.
    let Some(e) = env("rules").await else {
        return;
    };
    let report = rules::run(&e.client, &[]).await.unwrap();
    let uses = report
        .of(rules::AGENT_USES_ACTIVATED.name)
        .expect("the rule is declared");
    assert_eq!(uses.written, 0);
    assert!(
        uses.measure.contains("run_with"),
        "it names where its measurement comes from: {}",
        uses.measure
    );
    assert!(report.of(rules::AGENT_KNOWS_CORE.name).is_some());
    cleanup(&e).await;
}
