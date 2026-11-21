//! `AnalyticsStore` on obrain, sub-lot "agents and views" (`src/obrain/agent.rs`,
//! `src/obrain/vues.rs`): what only this backend does, and the contract of the Neo4j
//! reference (`src/neo4j/chat.rs`, `beliefs.rs`, `decision.rs`, `constraint.rs`,
//! `plan.rs`) stated on the method. Persistence across a close and a reopen, a write cut
//! at every engine write then retried, the quarantine of an unreadable agent, the
//! `DETACH DELETE` that takes the agent edges with a note or a session, the
//! contextual refusals of the views, and the refusals by name.
#![cfg(feature = "obrain-backend")]

use std::path::Path;

use chrono::{Duration, TimeZone, Utc};
use obrain_common::types::{NodeId, Timestamp, Value, ZonedDatetime};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use project_orchestrator::episodes::models::{
    Episode, FeedbackType, Outcome, Process, SessionTrace, Stimulus, StimulusTrigger, Validation,
};
use project_orchestrator::graph::rules::{AccountableProvenance, KnowledgeScope};
use project_orchestrator::neo4j::models::*;
use project_orchestrator::neo4j::traits::*;
use project_orchestrator::notes::models::{
    AssertionCheckType, AssertionResult, AssertionRule, NoteStatus, NoteType, ViolationAction,
};
use project_orchestrator::notes::Note;
use project_orchestrator::obrain::{
    methodes_qui_servent, refusees_analytics, Creation, Interruption, NonPorte, ObrainGraphStore,
    MARQUEUR_PO, SCHEMA_PO,
};
use project_orchestrator::protocol::models::Protocol;
use uuid::Uuid;

async fn open(path: &Path) -> ObrainGraphStore {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || ObrainGraphStore::open(path, Creation::Permise))
        .await
        .unwrap()
        .unwrap()
}

/// Open the closed base with the engine itself, run `f`, and put PO's marker back.
async fn with_engine<T: Send + 'static>(
    base: &Path,
    f: impl FnOnce(&ObrainDB) -> T + Send + 'static,
) -> T {
    let base = base.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let db = ObrainDB::open(&base).unwrap();
        let out = f(&db);
        db.commit_wal().unwrap();
        db.close().unwrap();
        std::fs::write(base.join(MARQUEUR_PO), SCHEMA_PO).unwrap();
        out
    })
    .await
    .unwrap()
}

fn node_of(db: &ObrainDB, label: &str, id: Uuid) -> NodeId {
    let voulu = id.to_string();
    db.store()
        .nodes_by_label(label)
        .into_iter()
        .find(|n| {
            db.get_node(*n)
                .and_then(|nd| {
                    nd.get_property("id")
                        .and_then(|v| v.as_str().map(String::from))
                })
                .as_deref()
                == Some(voulu.as_str())
        })
        .unwrap_or_else(|| panic!("no {label} {id}"))
}

/// The edges of `type_` leaving the `label` node of this id.
fn edges_out(db: &ObrainDB, label: &str, id: Uuid, type_: &str) -> usize {
    let n = node_of(db, label, id);
    db.store()
        .edges_from(n, Direction::Outgoing)
        .into_iter()
        .filter(|(_, a)| db.get_edge(*a).is_some_and(|e| &*e.edge_type == type_))
        .count()
}

fn count_label(db: &ObrainDB, label: &str) -> usize {
    db.store().nodes_by_label(label).len()
}

fn prov(rule: &str, score: Option<f64>, at: &str) -> AccountableProvenance {
    let mut p = project_orchestrator::graph::rules::Provenance::new(rule, "mesure");
    p.at = at.to_string();
    p.score = score;
    p.accountable().unwrap()
}

fn a_note(project: Option<Uuid>, content: &str, age_s: i64) -> Note {
    let mut n = Note::new(
        project,
        NoteType::Observation,
        content.to_string(),
        "test".to_string(),
    );
    n.created_at = Utc.timestamp_opt(1_700_000_000, 0).unwrap() + Duration::seconds(age_s);
    n
}

fn a_belief(content: &str, age_s: i64) -> Note {
    let mut n = a_note(None, content, age_s);
    n.note_type = NoteType::Assertion;
    n.assertion_rule = Some(AssertionRule {
        check_type: AssertionCheckType::Exists,
        target: "function:f".into(),
        file_pattern: None,
        parameters: None,
        on_violation: ViolationAction::FlagNote,
    });
    n
}

fn a_project() -> ProjectNode {
    let id = Uuid::new_v4();
    ProjectNode {
        id,
        name: format!("agents {id}"),
        slug: format!("agents-{id}"),
        root_path: String::new(),
        description: None,
        created_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
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

fn a_session(id: Uuid) -> ChatSessionNode {
    use project_orchestrator::chat::provider::policy::SessionAccess;
    let now = Utc::now();
    ChatSessionNode {
        id,
        cli_session_id: None,
        project_slug: Some("p".to_string()),
        workspace_slug: None,
        cwd: "/tmp".to_string(),
        title: None,
        model: "sonnet".to_string(),
        created_at: now,
        updated_at: now,
        message_count: 0,
        total_cost_usd: None,
        conversation_id: None,
        preview: None,
        permission_mode: None,
        add_dirs: None,
        spawned_by: None,
        provider_id: None,
        routed_by: None,
        routing_mode: None,
        routing_pool: None,
        capabilities: None,
        engine: None,
        resume_token: None,
        execution_place: ExecutionPlace::Project,
        project_inferred: false,
        access: SessionAccess::Normal,
        owner: None,
        file_scope_lifted: false,
        mirror_of: None,
    }
}

fn an_episode(session: Uuid, agent: Option<Uuid>) -> Episode {
    Episode {
        id: Uuid::new_v4(),
        project_id: Uuid::new_v4(),
        stimulus: Stimulus {
            request: "r".to_string(),
            trigger: StimulusTrigger::UserRequest,
            timestamp: Utc::now(),
            context_hash: None,
        },
        process: Process {
            reasoning_tree_id: None,
            states_visited: vec![],
            state_visits: vec![],
            duration_ms: None,
        },
        outcome: Outcome {
            note_ids: vec![],
            decision_ids: vec![],
            commit_shas: vec![],
            files_modified: 0,
        },
        validation: Validation {
            feedback_type: FeedbackType::None,
            score: None,
            evidence_count: 0,
        },
        lesson: None,
        collected_at: Utc::now(),
        source_run_id: None,
        session: Some(SessionTrace {
            session_id: session,
            message_count: 1,
            close_reason: "closed".to_string(),
            total_cost_usd: None,
            agent_id: agent,
        }),
        embedding: None,
    }
}

// ============================================================================
// The agent
// ============================================================================

/// `ensure_agent`: the only agent is the durable self (any other id is Neo4j's refusal),
/// created once; the model is a dated facet beside it, one per normalized model name, and
/// the whole survives a reopen.
#[tokio::test]
async fn ensure_agent_makes_the_self_once_and_a_facet_per_model() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let root = store.agent_identity_namespace().await.unwrap();

    let e = store
        .ensure_agent(Uuid::new_v4(), None)
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("is not the durable self of this store"), "{e}");

    assert!(store.ensure_agent(root, None).await.unwrap(), "created");
    assert!(!store.ensure_agent(root, None).await.unwrap(), "then known");
    // A model that names none (empty, `default`) makes no facet.
    assert!(!store.ensure_agent(root, Some("  ")).await.unwrap());
    assert!(!store
        .ensure_agent(root, Some("claude-opus-4"))
        .await
        .unwrap());
    // Same model under another spelling: the same facet.
    assert!(!store
        .ensure_agent(root, Some("Claude-Opus-4"))
        .await
        .unwrap());
    assert!(!store
        .ensure_agent(root, Some("claude-haiku-4"))
        .await
        .unwrap());
    store.close().await.unwrap();
    drop(store);

    let (agents, facet_edges) = with_engine(&base, move |db| {
        (
            count_label(db, "Agent"),
            db.store()
                .nodes_by_label("Agent")
                .into_iter()
                .map(|n| {
                    db.store()
                        .edges_from(n, Direction::Outgoing)
                        .into_iter()
                        .filter(|(_, a)| {
                            db.get_edge(*a).is_some_and(|e| &*e.edge_type == "FACET_OF")
                        })
                        .count()
                })
                .sum::<usize>(),
        )
    })
    .await;
    assert_eq!((agents, facet_edges), (3, 2), "the self and two facets");

    let store = open(&base).await;
    assert!(
        !store.ensure_agent(root, None).await.unwrap(),
        "known after a reopen"
    );
}

/// `ensure_agent` cut at every engine write and retried: never a second self, never a
/// facet without its edge once the retry has run.
#[tokio::test]
async fn ensure_agent_cut_at_every_write_converges_on_retry() {
    let mut atteints = 0;
    for k in 1..=12 {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let root = store.agent_identity_namespace().await.unwrap();
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store
            .ensure_agent(root, Some("claude-opus-4"))
            .await
            .is_ok()
        {
            break;
        }
        atteints += 1;
        drop(store);
        let store = open(&base).await;
        assert!(
            store.rapport_d_ouverture().en_quarantaine.is_empty(),
            "k={k}"
        );
        store
            .ensure_agent(root, Some("claude-opus-4"))
            .await
            .unwrap();
        store.close().await.unwrap();
        drop(store);
        let (agents, edges) = with_engine(&base, move |db| {
            (
                count_label(db, "Agent"),
                db.store()
                    .nodes_by_label("Agent")
                    .into_iter()
                    .map(|n| db.store().edges_from(n, Direction::Outgoing).len())
                    .sum::<usize>(),
            )
        })
        .await;
        assert_eq!(
            (agents, edges),
            (2, 1),
            "k={k}: the self, one facet, one edge"
        );
    }
    assert!(
        atteints >= 4,
        "la coupure n'a atteint que {atteints} points"
    );
}

/// `AUTHORED` and `LIVED`: a MERGE that answers whether THIS call drew the edge; an absent
/// agent or an absent target draws nothing and answers `false` (Neo4j's `MATCH`); the edge
/// survives a reopen; deleting the note or the session takes it away.
#[tokio::test]
async fn authored_and_lived_are_merges_and_die_with_their_target() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let root = store.agent_identity_namespace().await.unwrap();
    let note = a_note(None, "n", 0);
    store.create_note(&note).await.unwrap();
    let sid = Uuid::new_v4();
    store.create_chat_session(&a_session(sid)).await.unwrap();
    let episode = an_episode(sid, None);
    assert!(store.save_session_episode(&episode).await.unwrap());

    // No agent yet: nothing is drawn, nothing fails.
    assert!(!store.link_agent_authored(root, note.id).await.unwrap());
    assert!(!store.link_agent_lived(root, episode.id).await.unwrap());
    store.ensure_agent(root, None).await.unwrap();

    assert!(store.link_agent_authored(root, note.id).await.unwrap());
    assert!(!store.link_agent_authored(root, note.id).await.unwrap());
    assert!(!store
        .link_agent_authored(root, Uuid::new_v4())
        .await
        .unwrap());
    assert!(!store
        .link_agent_authored(Uuid::new_v4(), note.id)
        .await
        .unwrap());
    assert!(store.link_agent_lived(root, episode.id).await.unwrap());
    assert!(!store.link_agent_lived(root, episode.id).await.unwrap());
    assert!(!store.link_agent_lived(root, Uuid::new_v4()).await.unwrap());
    store.close().await.unwrap();
    drop(store);

    let (a, l) = with_engine(&base, move |db| {
        (
            edges_out(db, "Agent", root, "AUTHORED"),
            edges_out(db, "Agent", root, "LIVED"),
        )
    })
    .await;
    assert_eq!((a, l), (1, 1), "one edge each, drawn once");

    // DETACH DELETE: the edge goes with the note and with the session (and its episode).
    let store = open(&base).await;
    assert!(store.delete_note(note.id).await.unwrap());
    assert!(store.delete_chat_session(sid).await.unwrap());
    store.close().await.unwrap();
    drop(store);
    let (a, l, ep) = with_engine(&base, move |db| {
        (
            edges_out(db, "Agent", root, "AUTHORED"),
            edges_out(db, "Agent", root, "LIVED"),
            count_label(db, "SessionEpisode"),
        )
    })
    .await;
    assert_eq!((a, l, ep), (0, 0, 0), "the edges left with their targets");
}

/// `KNOWS`: the last measure wins (`SET` after `MERGE`), the project is removed when the
/// new call has none, `created` says who drew the edge. `agent_knows` serves the two scales
/// (global, and the project asked), strongest score first, the instant of `at` breaking a tie
/// (newest first), then the id; `available` counts what was withheld by the limit.
#[tokio::test]
async fn knows_keeps_the_last_measure_and_agent_knows_orders_strongest_first() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let root = store.agent_identity_namespace().await.unwrap();
    store.ensure_agent(root, None).await.unwrap();
    let (p1, p2) = (Uuid::new_v4(), Uuid::new_v4());
    let mut notes = Vec::new();
    for i in 0..5 {
        let n = a_note(None, &format!("n{i}"), i);
        store.create_note(&n).await.unwrap();
        notes.push(n.id);
    }
    let g = KnowledgeScope::Global;
    let pr = KnowledgeScope::Project;
    // 0: global 0.5 old; 1: global 0.5 new (ties 0 on score, wins on `at`); 2: project p1 0.9;
    // 3: project p2 0.99; 4: global, no score (coalesce 0.0).
    assert!(store
        .link_agent_knows(
            root,
            notes[0],
            g,
            None,
            &prov("r", Some(0.5), "2026-01-01T00:00:00Z")
        )
        .await
        .unwrap());
    assert!(store
        .link_agent_knows(
            root,
            notes[1],
            g,
            None,
            &prov("r", Some(0.5), "2026-02-01T00:00:00Z")
        )
        .await
        .unwrap());
    assert!(store
        .link_agent_knows(
            root,
            notes[2],
            pr,
            Some(p1),
            &prov("r", Some(0.9), "2026-01-01T00:00:00Z")
        )
        .await
        .unwrap());
    assert!(store
        .link_agent_knows(
            root,
            notes[3],
            pr,
            Some(p2),
            &prov("r", Some(0.99), "2026-01-01T00:00:00Z")
        )
        .await
        .unwrap());
    assert!(store
        .link_agent_knows(
            root,
            notes[4],
            g,
            None,
            &prov("other", None, "2026-03-01T00:00:00Z")
        )
        .await
        .unwrap());
    // A second call is not a creation, and the new measure replaces the old one.
    assert!(!store
        .link_agent_knows(
            root,
            notes[0],
            g,
            None,
            &prov("r", Some(0.5), "2026-01-01T00:00:00Z")
        )
        .await
        .unwrap());
    assert!(!store
        .link_agent_knows(Uuid::new_v4(), notes[0], g, None, &prov("r", None, "x"))
        .await
        .unwrap());
    assert!(!store
        .link_agent_knows(root, Uuid::new_v4(), g, None, &prov("r", None, "x"))
        .await
        .unwrap());

    // Without a project: the global scale only, never another project's foundation.
    let k = store.agent_knows(root, None, 10).await.unwrap();
    assert_eq!(k.available, 3);
    assert_eq!(k.notes, vec![notes[1], notes[0], notes[4]]);
    // With p1: the global scale and p1's.
    let k = store.agent_knows(root, Some(p1), 10).await.unwrap();
    assert_eq!(k.notes, vec![notes[2], notes[1], notes[0], notes[4]]);
    // The bound withholds the weakest, and says how many.
    let k = store.agent_knows(root, Some(p1), 2).await.unwrap();
    assert_eq!(
        (k.notes.clone(), k.available, k.withheld()),
        (vec![notes[2], notes[1]], 4, 2)
    );
    // An unknown agent knows nothing (Neo4j: no row).
    let k = store.agent_knows(Uuid::new_v4(), None, 10).await.unwrap();
    assert_eq!((k.notes.len(), k.available), (0, 0));

    // Re-link note 3 as global with no project: its project is removed, so p2's scale no
    // longer holds it, and the global scale does.
    assert!(!store
        .link_agent_knows(
            root,
            notes[3],
            g,
            None,
            &prov("r", Some(0.1), "2026-01-01T00:00:00Z")
        )
        .await
        .unwrap());
    let k = store.agent_knows(root, Some(p2), 10).await.unwrap();
    assert!(k.notes.contains(&notes[3]));
    assert_eq!(*k.notes.last().unwrap(), notes[4], "score 0.0 is last");

    // It survives a reopen.
    store.close().await.unwrap();
    drop(store);
    let store = open(&base).await;
    let k = store.agent_knows(root, Some(p1), 10).await.unwrap();
    assert_eq!(k.available, 5);
    assert_eq!(k.notes[0], notes[2]);
}

/// `release_agent_knows` deletes the `KNOWS` edges of THIS rule whose note is not kept, and
/// counts them; another rule's edges and the kept notes stay.
#[tokio::test]
async fn release_agent_knows_deletes_only_the_edges_of_its_rule_that_are_not_kept() {
    let store = ObrainGraphStore::open_temporary().unwrap();
    let root = store.agent_identity_namespace().await.unwrap();
    store.ensure_agent(root, None).await.unwrap();
    let mut ids = Vec::new();
    for i in 0..4 {
        let n = a_note(None, &format!("n{i}"), i);
        store.create_note(&n).await.unwrap();
        ids.push(n.id);
    }
    let g = KnowledgeScope::Global;
    for (i, rule) in [(0, "a"), (1, "a"), (2, "a"), (3, "b")] {
        store
            .link_agent_knows(
                root,
                ids[i],
                g,
                None,
                &prov(rule, Some(1.0), "2026-01-01T00:00:00Z"),
            )
            .await
            .unwrap();
    }
    let n = store
        .release_agent_knows(root, "a", &[ids[0]])
        .await
        .unwrap();
    assert_eq!(n, 2, "notes 1 and 2: rule a, not kept");
    let k = store.agent_knows(root, None, 10).await.unwrap();
    let mut got = k.notes.clone();
    got.sort();
    let mut want = vec![ids[0], ids[3]];
    want.sort();
    assert_eq!(got, want);
    assert_eq!(
        store
            .release_agent_knows(root, "a", &[ids[0]])
            .await
            .unwrap(),
        0
    );
    assert_eq!(
        store
            .release_agent_knows(Uuid::new_v4(), "a", &[])
            .await
            .unwrap(),
        0
    );
}

/// `link_agent_knows` cut at every engine write and retried converges: one edge, the last
/// measure, whatever the cut.
#[tokio::test]
async fn knows_cut_at_every_write_converges_on_retry() {
    let mut atteints = 0;
    for k in 1..=10 {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let root = store.agent_identity_namespace().await.unwrap();
        store.ensure_agent(root, None).await.unwrap();
        let n = a_note(None, "n", 0);
        store.create_note(&n).await.unwrap();
        let p = prov("r", Some(0.7), "2026-01-01T00:00:00Z");
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store
            .link_agent_knows(root, n.id, KnowledgeScope::Global, None, &p)
            .await
            .is_ok()
        {
            break;
        }
        atteints += 1;
        drop(store);
        let store = open(&base).await;
        store
            .link_agent_knows(root, n.id, KnowledgeScope::Global, None, &p)
            .await
            .unwrap();
        let kn = store.agent_knows(root, None, 10).await.unwrap();
        assert_eq!((kn.notes, kn.available), (vec![n.id], 1), "k={k}");
        store.close().await.unwrap();
        drop(store);
        let edges = with_engine(&base, move |db| edges_out(db, "Agent", root, "KNOWS")).await;
        assert_eq!(edges, 1, "k={k}");
    }
    assert!(
        atteints >= 6,
        "la coupure n'a atteint que {atteints} points"
    );
}

/// `link_agent_uses`: the label is checked first (Neo4j's closed list), the target is found
/// by label and id, `created` as the other links, and deleting the target takes the edge.
#[tokio::test]
async fn uses_reaches_a_protocol_and_refuses_a_label_outside_the_list() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let root = store.agent_identity_namespace().await.unwrap();
    store.ensure_agent(root, None).await.unwrap();
    let proto = Protocol::new(None, "p", Uuid::new_v4());
    store.upsert_protocol(&proto).await.unwrap();
    let p = prov("agent-uses-activated", None, "2026-01-01T00:00:00Z");

    let e = store
        .link_agent_uses(root, "File", proto.id, &p)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("is not a usable label: Persona, Skill, Protocol"),
        "{e}"
    );
    // The label is checked before anything else, even for an unknown agent.
    assert!(store
        .link_agent_uses(Uuid::new_v4(), "File", proto.id, &p)
        .await
        .is_err());

    assert!(store
        .link_agent_uses(root, "Protocol", proto.id, &p)
        .await
        .unwrap());
    assert!(!store
        .link_agent_uses(root, "Protocol", proto.id, &p)
        .await
        .unwrap());
    assert!(!store
        .link_agent_uses(root, "Protocol", Uuid::new_v4(), &p)
        .await
        .unwrap());
    assert!(!store
        .link_agent_uses(root, "Skill", proto.id, &p)
        .await
        .unwrap());
    assert!(!store
        .link_agent_uses(Uuid::new_v4(), "Protocol", proto.id, &p)
        .await
        .unwrap());
    store.close().await.unwrap();
    drop(store);
    let n = with_engine(&base, move |db| edges_out(db, "Agent", root, "USES")).await;
    assert_eq!(n, 1);

    let store = open(&base).await;
    store.delete_protocol(proto.id).await.unwrap();
    store.close().await.unwrap();
    drop(store);
    let n = with_engine(&base, move |db| edges_out(db, "Agent", root, "USES")).await;
    assert_eq!(n, 0, "the edge left with the protocol");
}

/// An agent node that does not decode is quarantined by name and its id reserved: a write
/// on it refuses instead of making a second node.
#[tokio::test]
async fn an_unreadable_agent_is_quarantined_and_its_id_is_reserved() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let root = store.agent_identity_namespace().await.unwrap();
    store.close().await.unwrap();
    drop(store);
    with_engine(&base, move |db| {
        let n = db.create_node(&["Agent"]);
        db.set_node_property(n, "id", Value::from(root.to_string()));
        db.set_node_property(n, "kind", Value::from("bogus"));
        db.set_node_property(n, "created_at", Value::from("2026-01-01T00:00:00Z"));
    })
    .await;
    let store = open(&base).await;
    assert!(store
        .rapport_d_ouverture()
        .en_quarantaine
        .iter()
        .any(|q| q.contains("Agent") && q.contains("kind")));
    let e = store
        .ensure_agent(root, None)
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("réservé"), "{e}");
    let e = store
        .link_agent_authored(root, Uuid::new_v4())
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("réservé"), "{e}");
}

/// A date the migrator wrote (`ZonedDatetime`) reads through the single decoder: an agent
/// created that way is served, not quarantined.
#[tokio::test]
async fn an_agent_with_a_zoned_creation_date_is_served() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let root = store.agent_identity_namespace().await.unwrap();
    store.close().await.unwrap();
    drop(store);
    with_engine(&base, move |db| {
        let n = db.create_node(&["Agent"]);
        db.set_node_property(n, "id", Value::from(root.to_string()));
        db.set_node_property(n, "kind", Value::from("self"));
        db.set_node_property(
            n,
            "created_at",
            Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_micros(1_767_225_600_000_000),
                0,
            )),
        );
    })
    .await;
    let store = open(&base).await;
    assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
    assert!(
        !store.ensure_agent(root, None).await.unwrap(),
        "the self is known, not created a second time"
    );
}

/// `backfill_agents` rebuilds `AUTHORED` and `LIVED` from the notes' `created_by` and the
/// episodes' agent: a model facet resolves to the self, a foreign id draws nothing, a re-run
/// draws nothing, and the counts are Neo4j's.
#[tokio::test]
async fn backfill_agents_rebuilds_the_edges_once() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let root = store.agent_identity_namespace().await.unwrap();
    store
        .ensure_agent(root, Some("claude-opus-4"))
        .await
        .unwrap();
    let facet = Uuid::new_v5(&root, b"facet:claude-opus-4");
    let foreign = Uuid::new_v4();
    let n_self = a_note(None, "by the self", 0);
    let n_facet = a_note(None, "by the facet", 1);
    let n_foreign = a_note(None, "by a stranger", 2);
    let n_human = a_note(None, "by a human", 3);
    for n in [&n_self, &n_facet, &n_foreign, &n_human] {
        store.create_note(n).await.unwrap();
    }
    let sid = Uuid::new_v4();
    store.create_chat_session(&a_session(sid)).await.unwrap();
    let episode = an_episode(sid, Some(facet));
    assert!(store.save_session_episode(&episode).await.unwrap());
    store.close().await.unwrap();
    drop(store);
    // `create_note` refuses an `agent:` author (the Agent domain used not to be ported): the
    // authorship of a migrated base is written on the node.
    let (a, b, c) = (n_self.id, n_facet.id, n_foreign.id);
    with_engine(&base, move |db| {
        for (id, who) in [(a, root), (b, facet), (c, foreign)] {
            let n = node_of(db, "Note", id);
            db.set_node_property(n, "created_by", Value::from(format!("agent:{who}")));
        }
    })
    .await;

    let store = open(&base).await;
    let r = store.backfill_agents().await.unwrap();
    assert_eq!(
        (
            r.before,
            r.after,
            r.from_notes,
            r.from_episodes,
            r.authored,
            r.lived
        ),
        (1, 1, 3, 1, 2, 1),
        "{r:?}"
    );
    let again = store.backfill_agents().await.unwrap();
    assert_eq!(
        (
            again.authored,
            again.lived,
            again.from_notes,
            again.from_episodes
        ),
        (0, 0, 3, 1),
        "a re-run draws nothing"
    );
    store.close().await.unwrap();
    drop(store);
    let (au, li, ag) = with_engine(&base, move |db| {
        (
            edges_out(db, "Agent", root, "AUTHORED"),
            edges_out(db, "Agent", root, "LIVED"),
            count_label(db, "Agent"),
        )
    })
    .await;
    assert_eq!(
        (au, li, ag),
        (2, 1, 2),
        "the facet and the stranger drew no edge of their own"
    );
}

/// On a store with no actor to resolve, `backfill_agents` creates nothing (Neo4j: the self
/// is ensured only when something resolved).
#[tokio::test]
async fn backfill_agents_of_an_empty_base_creates_no_agent() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let r = store.backfill_agents().await.unwrap();
    assert_eq!(
        (
            r.before,
            r.after,
            r.from_notes,
            r.from_episodes,
            r.authored,
            r.lived
        ),
        (0, 0, 0, 0, 0, 0)
    );
    store.close().await.unwrap();
    drop(store);
    let n = with_engine(&base, |db| count_label(db, "Agent")).await;
    assert_eq!(n, 0);
}

// ============================================================================
// The beliefs
// ============================================================================

/// `list_assertion_notes` lists the notes that carry a rule OR declare themselves
/// assertions (a belief with no procedure is visible, never silent), oldest check first
/// (`last_checked_at`, else `created_at`), then id; `record_assertion_check` puts a checked
/// note at the back of the round, writes the status and the reason only when a status is
/// given, and does nothing for an absent note.
#[tokio::test]
async fn beliefs_are_listed_in_round_order_and_a_check_moves_a_note_to_the_back() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let b1 = a_belief("b1", 10);
    let b2 = a_belief("b2", 20);
    let mut b3 = a_note(None, "declares itself a belief, no procedure", 30);
    b3.note_type = NoteType::Assertion;
    let mut b4 = a_note(None, "a rule on an observation", 5);
    b4.assertion_rule = b1.assertion_rule.clone();
    let plain = a_note(None, "an observation", 1);
    for n in [&b1, &b2, &b3, &b4, &plain] {
        store.create_note(n).await.unwrap();
    }

    let ids = |v: Vec<Note>| v.into_iter().map(|n| n.id).collect::<Vec<_>>();
    assert_eq!(
        ids(store.list_assertion_notes(10).await.unwrap()),
        vec![b4.id, b1.id, b2.id, b3.id]
    );
    assert_eq!(
        ids(store.list_assertion_notes(2).await.unwrap()),
        vec![b4.id, b1.id],
        "the limit"
    );

    // A `Warn`-style check: no status, nothing else touched.
    store
        .record_assertion_check(b4.id, &AssertionResult::passed("ok"), None, Some("ignored"))
        .await
        .unwrap();
    let n4 = store.get_note(b4.id).await.unwrap().unwrap();
    assert_eq!(n4.status, NoteStatus::Active);
    assert_eq!(
        n4.revision_reason, None,
        "no status: the reason is not written"
    );
    assert!(n4.last_assertion_result.is_some_and(|r| r.passed));
    assert_eq!(
        ids(store.list_assertion_notes(10).await.unwrap()),
        vec![b1.id, b2.id, b3.id, b4.id],
        "checked now: at the back of the round"
    );

    // A refuting check: the status and the reason are written.
    store
        .record_assertion_check(
            b1.id,
            &AssertionResult::failed("gone"),
            Some(NoteStatus::NeedsReview),
            Some("function f is gone"),
        )
        .await
        .unwrap();
    let n1 = store.get_note(b1.id).await.unwrap().unwrap();
    assert_eq!(n1.status, NoteStatus::NeedsReview);
    assert_eq!(n1.revision_reason.as_deref(), Some("function f is gone"));
    assert!(n1.last_assertion_result.is_some_and(|r| !r.passed));
    // A later check with a status and no reason clears the reason (`SET n.revision_reason = null`).
    store
        .record_assertion_check(
            b1.id,
            &AssertionResult::failed("still"),
            Some(NoteStatus::Obsolete),
            None,
        )
        .await
        .unwrap();
    let n1 = store.get_note(b1.id).await.unwrap().unwrap();
    assert_eq!(
        (n1.status, n1.revision_reason),
        (NoteStatus::Obsolete, None)
    );

    // An absent note: nothing to do, no error.
    store
        .record_assertion_check(Uuid::new_v4(), &AssertionResult::passed("x"), None, None)
        .await
        .unwrap();

    // The order survives a reopen: the check date is on the node.
    store.close().await.unwrap();
    drop(store);
    let store = open(&base).await;
    assert_eq!(
        ids(store.list_assertion_notes(10).await.unwrap()),
        vec![b2.id, b3.id, b4.id, b1.id]
    );
}

/// `record_assertion_check` cut at every engine write and retried: the note is left with
/// the result of the last COMPLETE call, or a valid prefix, and the retry finishes it.
#[tokio::test]
async fn a_cut_belief_check_is_never_marked_checked_without_its_result() {
    let mut atteints = 0;
    for k in 1..=8 {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let b = a_belief("b", 10);
        let other = a_belief("other", 20);
        store.create_note(&b).await.unwrap();
        store.create_note(&other).await.unwrap();
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store
            .record_assertion_check(
                b.id,
                &AssertionResult::failed("gone"),
                Some(NoteStatus::NeedsReview),
                Some("why"),
            )
            .await
            .is_ok()
        {
            break;
        }
        atteints += 1;
        drop(store);
        let store = open(&base).await;
        let n = store.get_note(b.id).await.unwrap().unwrap();
        let listed = store.list_assertion_notes(10).await.unwrap();
        if n.last_assertion_result.is_none() {
            assert_eq!(
                listed[0].id, b.id,
                "k={k}: not marked checked, so still first"
            );
        }
        store
            .record_assertion_check(
                b.id,
                &AssertionResult::failed("gone"),
                Some(NoteStatus::NeedsReview),
                Some("why"),
            )
            .await
            .unwrap();
        let n = store.get_note(b.id).await.unwrap().unwrap();
        assert_eq!(
            (n.status, n.revision_reason.as_deref()),
            (NoteStatus::NeedsReview, Some("why")),
            "k={k}"
        );
    }
    assert!(
        atteints >= 4,
        "la coupure n'a atteint que {atteints} points"
    );
}

/// `project_synced`: `Some(false)` before the first sync, `Some(true)` after, `None` for a
/// project that is not there.
#[tokio::test]
async fn project_synced_says_none_for_an_unknown_project() {
    let store = ObrainGraphStore::open_temporary().unwrap();
    let p = a_project();
    store.create_project(&p).await.unwrap();
    assert_eq!(store.project_synced(p.id).await.unwrap(), Some(false));
    store.update_project_synced(p.id).await.unwrap();
    assert_eq!(store.project_synced(p.id).await.unwrap(), Some(true));
    assert_eq!(store.project_synced(Uuid::new_v4()).await.unwrap(), None);
}

/// A project node this backend quarantined is not "absent": the answer of Neo4j cannot be
/// given, so the call refuses.
#[tokio::test]
async fn project_synced_of_a_quarantined_project_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let id = Uuid::new_v4();
    let store = open(&base).await;
    store.close().await.unwrap();
    drop(store);
    with_engine(&base, move |db| {
        let n = db.create_node(&["Project"]);
        db.set_node_property(n, "id", Value::from(id.to_string()));
    })
    .await;
    let store = open(&base).await;
    let e = store.project_synced(id).await.unwrap_err().to_string();
    assert!(e.contains("quarantaine"), "{e}");
}

// ============================================================================
// The views
// ============================================================================

fn a_decision(at_s: i64, text: &str) -> DecisionNode {
    DecisionNode {
        id: Uuid::new_v4(),
        description: text.to_string(),
        rationale: "r".to_string(),
        alternatives: vec!["a".to_string()],
        chosen_option: Some("c".to_string()),
        decided_by: "me".to_string(),
        decided_at: Utc.timestamp_opt(1_700_000_000 + at_s, 0).unwrap(),
        status: DecisionStatus::Accepted,
        embedding: None,
        embedding_model: None,
        scar_intensity: 0.0,
        sharing_consent: Default::default(),
    }
}

/// `get_project_decisions_for_graph`: the project's decisions in `(decided_at, id)` order,
/// each with an empty `AFFECTS` list (none can be written here), the other project's
/// decisions out; a migrated `AFFECTS` or `INFORMED_BY` edge refuses by name.
#[tokio::test]
async fn project_decisions_for_the_graph_are_exact_or_refused() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let (p, q) = (a_project(), a_project());
    store.create_project(&p).await.unwrap();
    store.create_project(&q).await.unwrap();
    let (late, early, other) = (
        a_decision(50, "late"),
        a_decision(10, "early"),
        a_decision(30, "q"),
    );
    store
        .create_decision_global(&late, Some(p.id))
        .await
        .unwrap();
    store
        .create_decision_global(&early, Some(p.id))
        .await
        .unwrap();
    store
        .create_decision_global(&other, Some(q.id))
        .await
        .unwrap();

    let got = store.get_project_decisions_for_graph(p.id).await.unwrap();
    assert_eq!(
        got.iter().map(|(d, _)| d.id).collect::<Vec<_>>(),
        vec![early.id, late.id]
    );
    assert!(got.iter().all(|(_, a)| a.is_empty()));
    assert_eq!(got[0].0.description, "early");
    assert_eq!(got[0].0.alternatives, vec!["a".to_string()]);
    assert!(store
        .get_project_decisions_for_graph(Uuid::new_v4())
        .await
        .unwrap()
        .is_empty());
    store.close().await.unwrap();
    drop(store);

    // A decision of THIS project carrying AFFECTS: refused. One of ANOTHER project: served.
    let (lid, oid) = (late.id, other.id);
    with_engine(&base, move |db| {
        let d = node_of(db, "Decision", lid);
        let t = db.create_node(&["File"]);
        db.set_node_property(t, "path", Value::from("a.rs"));
        db.create_edge(d, t, "AFFECTS");
        let _ = oid;
    })
    .await;
    let store = open(&base).await;
    let e = store
        .get_project_decisions_for_graph(p.id)
        .await
        .unwrap_err();
    assert_eq!(
        e.downcast_ref::<NonPorte>(),
        Some(&NonPorte("AnalyticsStore::get_project_decisions_for_graph")),
        "{e:#}"
    );
    assert_eq!(
        store
            .get_project_decisions_for_graph(q.id)
            .await
            .unwrap()
            .len(),
        1
    );
    store.close().await.unwrap();
    drop(store);

    // An INFORMED_BY edge on ANY decision: its project is unresolved, every call refuses.
    with_engine(&base, move |db| {
        let d = node_of(db, "Decision", oid);
        let t = db.create_node(&["Task"]);
        db.set_node_property(t, "id", Value::from(Uuid::new_v4().to_string()));
        db.create_edge(t, d, "INFORMED_BY");
    })
    .await;
    let store = open(&base).await;
    assert!(store.get_project_decisions_for_graph(q.id).await.is_err());
}

fn a_constraint(text: &str) -> ConstraintNode {
    ConstraintNode {
        id: Uuid::new_v4(),
        constraint_type: ConstraintType::Performance,
        description: text.to_string(),
        enforced_by: None,
    }
}

fn a_plan(project: Option<Uuid>, priority: i32, title: &str) -> PlanNode {
    PlanNode {
        id: Uuid::new_v4(),
        title: title.to_string(),
        description: "d".to_string(),
        status: PlanStatus::InProgress,
        created_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
        created_by: "t".to_string(),
        priority,
        project_id: project,
        execution_context: None,
        persona: None,
    }
}

/// `get_project_constraints` reproduces the reference, which matches
/// `(:Plan)-[:BELONGS_TO_PROJECT]->(:Project)`: an edge no writer of this repository draws.
/// A plan reaching its project by `HAS_PLAN` (the normal state) gives an empty answer, as on
/// Neo4j today; a plan carrying the edge gives its constraints with its id.
#[tokio::test]
async fn project_constraints_follow_the_reference_edge_not_has_plan() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let p = a_project();
    store.create_project(&p).await.unwrap();
    let plan = a_plan(Some(p.id), 1, "plan");
    store.create_plan(&plan).await.unwrap();
    let (c1, c2) = (a_constraint("c1"), a_constraint("c2"));
    store.create_constraint(plan.id, &c1).await.unwrap();
    store.create_constraint(plan.id, &c2).await.unwrap();
    assert!(
        store
            .get_project_constraints(p.id)
            .await
            .unwrap()
            .is_empty(),
        "HAS_PLAN is not BELONGS_TO_PROJECT: the reference's answer is empty"
    );
    store.close().await.unwrap();
    drop(store);
    let (pid, plid) = (p.id, plan.id);
    with_engine(&base, move |db| {
        let (a, b) = (node_of(db, "Plan", plid), node_of(db, "Project", pid));
        db.create_edge(a, b, "BELONGS_TO_PROJECT");
    })
    .await;
    let store = open(&base).await;
    let got = store.get_project_constraints(p.id).await.unwrap();
    let mut want = vec![(c1.id, plan.id), (c2.id, plan.id)];
    want.sort();
    assert_eq!(
        got.iter().map(|(c, pl)| (c.id, *pl)).collect::<Vec<_>>(),
        want
    );
    assert!(got
        .iter()
        .all(|(c, _)| c.constraint_type == ConstraintType::Performance));
    assert!(store
        .get_project_constraints(Uuid::new_v4())
        .await
        .unwrap()
        .is_empty());
}

/// `get_chat_graph_data` is exact for a base with no `DISCUSSED` edge (none can be written
/// here), and refuses by name when a session carries one.
#[tokio::test]
async fn chat_graph_data_is_empty_or_refused() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let p = a_project();
    store.create_project(&p).await.unwrap();
    let sid = Uuid::new_v4();
    store.create_chat_session(&a_session(sid)).await.unwrap();
    let (s, d) = store.get_chat_graph_data(p.id, 10).await.unwrap();
    assert!(s.is_empty() && d.is_empty());
    let n = a_note(None, "discussed", 0);
    store.create_note(&n).await.unwrap();
    store.close().await.unwrap();
    drop(store);
    let nid = n.id;
    with_engine(&base, move |db| {
        let (a, b) = (node_of(db, "ChatSession", sid), node_of(db, "Note", nid));
        let e = db.create_edge(a, b, "DISCUSSED");
        db.set_edge_property(e, "mention_count", Value::from(3_i64));
    })
    .await;
    let store = open(&base).await;
    let e = store.get_chat_graph_data(p.id, 10).await.unwrap_err();
    assert_eq!(
        e.downcast_ref::<NonPorte>(),
        Some(&NonPorte("AnalyticsStore::get_chat_graph_data")),
        "{e:#}"
    );
}

fn a_task(title: &str, priority: i32) -> TaskNode {
    TaskNode {
        id: Uuid::new_v4(),
        title: Some(title.into()),
        description: format!("description de {title}"),
        status: TaskStatus::Pending,
        assigned_to: None,
        interrupted_at: None,
        interrupted_reason: None,
        priority: Some(priority),
        tags: vec![],
        acceptance_criteria: vec![],
        affected_files: vec![],
        estimated_complexity: None,
        actual_complexity: None,
        due_date: None,
        estimated_minutes: None,
        created_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
        updated_at: None,
        started_at: None,
        completed_at: None,
        frustration_score: 0.0,
        execution_context: None,
        persona: None,
        prompt_cache: None,
    }
}

fn a_step(order: u32) -> StepNode {
    StepNode {
        id: Uuid::new_v4(),
        order,
        description: format!("étape {order}"),
        status: StepStatus::Pending,
        verification: None,
        created_at: Utc.timestamp_opt(1_700_000_000 + order as i64, 0).unwrap(),
        updated_at: None,
        completed_at: None,
        execution_context: None,
        persona: None,
    }
}

/// `get_pm_graph_data`: the nodes of each branch (plans by `priority DESC, id DESC`, the
/// rest by id), each branch bounded by the limit, the raw stored status, the label cut at 80
/// characters, the edges of the four kinds; the commit branches refuse when a `Commit`
/// carries `LINKED_TO_PLAN`.
#[tokio::test]
async fn pm_graph_data_serves_the_five_entity_kinds_and_refuses_the_commit_edges() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let (p, other) = (a_project(), a_project());
    store.create_project(&p).await.unwrap();
    store.create_project(&other).await.unwrap();
    let long = "x".repeat(90);
    let low = a_plan(Some(p.id), 1, "low");
    let high = a_plan(Some(p.id), 9, &long);
    let foreign_plan = a_plan(Some(other.id), 5, "foreign");
    for pl in [&low, &high, &foreign_plan] {
        store.create_plan(pl).await.unwrap();
    }
    let (t1, t2, tf) = (a_task("t1", 2), a_task("t2", 3), a_task("tf", 1));
    store.create_task(high.id, &t1).await.unwrap();
    store.create_task(high.id, &t2).await.unwrap();
    store.create_task(foreign_plan.id, &tf).await.unwrap();
    store.add_task_dependency(t2.id, t1.id).await.unwrap();
    let (s1, s2) = (a_step(1), a_step(2));
    store.create_step(t1.id, &s1).await.unwrap();
    store.create_step(t1.id, &s2).await.unwrap();
    let m = MilestoneNode {
        id: Uuid::new_v4(),
        title: "m".into(),
        description: Some("d".into()),
        status: MilestoneStatus::Open,
        target_date: Some(Utc.timestamp_opt(1_800_000_000, 0).unwrap()),
        closed_at: None,
        created_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
        project_id: p.id,
    };
    store.create_milestone(&m).await.unwrap();
    let r = ReleaseNode {
        id: Uuid::new_v4(),
        version: "1.2.3".into(),
        title: Some("rel".into()),
        description: None,
        status: ReleaseStatus::Planned,
        target_date: None,
        released_at: None,
        created_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
        project_id: p.id,
    };
    store.create_release(&r).await.unwrap();

    let (nodes, edges) = store.get_pm_graph_data(p.id, 100).await.unwrap();
    let kinds: Vec<(&str, String)> = nodes
        .iter()
        .map(|n| (n.node_type.as_str(), n.id.clone()))
        .collect();
    let mut tasks = [t1.id.to_string(), t2.id.to_string()];
    tasks.sort();
    let mut steps = [s1.id.to_string(), s2.id.to_string()];
    steps.sort();
    let mut want: Vec<(&str, String)> =
        vec![("plan", high.id.to_string()), ("plan", low.id.to_string())];
    want.extend(tasks.iter().map(|t| ("task", t.clone())));
    want.extend(steps.iter().map(|s| ("step", s.clone())));
    want.push(("milestone", m.id.to_string()));
    want.push(("release", r.id.to_string()));
    assert_eq!(kinds, want);
    // The label of the high plan is cut at 79 characters and the ellipsis.
    assert_eq!(nodes[0].label.chars().count(), 80);
    assert!(nodes[0].label.ends_with('…'));
    // Raw stored statuses and attributes.
    assert_eq!(nodes[0].attributes["status"], "InProgress");
    assert_eq!(nodes[0].attributes["priority"], 9);
    let task = nodes.iter().find(|n| n.id == t1.id.to_string()).unwrap();
    assert_eq!(task.attributes["status"], "Pending");
    assert_eq!(task.label, "t1");
    let step = nodes.iter().find(|n| n.id == s2.id.to_string()).unwrap();
    assert_eq!(step.attributes["order"], 2);
    assert_eq!(step.label, "étape 2");
    let ms = nodes.iter().find(|n| n.node_type == "milestone").unwrap();
    assert_eq!(ms.attributes["status"], "open");
    assert_eq!(
        ms.attributes["target_date"],
        Utc.timestamp_opt(1_800_000_000, 0).unwrap().to_rfc3339()
    );
    let rel = nodes.iter().find(|n| n.node_type == "release").unwrap();
    assert_eq!(rel.attributes["version"], "1.2.3");
    assert_eq!(rel.label, "rel");
    // The edges: HAS_TASK, HAS_STEP, DEPENDS_ON, each of the project only.
    let mut kinds: Vec<(&str, &str, &str)> = edges
        .iter()
        .map(|e| (e.rel_type.as_str(), e.source.as_str(), e.target.as_str()))
        .collect();
    let rels: Vec<&str> = kinds.iter().map(|k| k.0).collect();
    assert_eq!(
        rels,
        vec!["HAS_TASK", "HAS_TASK", "HAS_STEP", "HAS_STEP", "DEPENDS_ON"]
    );
    kinds.retain(|k| k.0 == "DEPENDS_ON");
    assert_eq!(
        kinds,
        vec![(
            "DEPENDS_ON",
            t2.id.to_string().as_str(),
            t1.id.to_string().as_str()
        )]
    );

    // Each branch is bounded by the limit on its own.
    let (nodes, _) = store.get_pm_graph_data(p.id, 1).await.unwrap();
    let count = |t: &str| nodes.iter().filter(|n| n.node_type == t).count();
    assert_eq!(
        (
            count("plan"),
            count("task"),
            count("step"),
            count("milestone"),
            count("release")
        ),
        (1, 1, 1, 1, 1)
    );
    assert_eq!(
        nodes[0].id,
        high.id.to_string(),
        "the plan kept is the highest"
    );
    let (nodes, edges) = store.get_pm_graph_data(p.id, 0).await.unwrap();
    assert_eq!(
        (nodes.len(), !edges.is_empty()),
        (0, true),
        "LIMIT 0 bounds the nodes, not the edges"
    );
    assert!(store
        .get_pm_graph_data(Uuid::new_v4(), 10)
        .await
        .unwrap()
        .0
        .is_empty());
    store.close().await.unwrap();
    drop(store);

    // A Commit linked to a plan of the project: the branch cannot be answered here.
    let hid = high.id;
    with_engine(&base, move |db| {
        let c = db.create_node(&["Commit"]);
        db.set_node_property(c, "sha", Value::from("abc"));
        db.create_edge(c, node_of(db, "Plan", hid), "LINKED_TO_PLAN");
    })
    .await;
    let store = open(&base).await;
    let e = store.get_pm_graph_data(p.id, 10).await.unwrap_err();
    assert_eq!(
        e.downcast_ref::<NonPorte>(),
        Some(&NonPorte("AnalyticsStore::get_pm_graph_data")),
        "{e:#}"
    );
}

// ============================================================================
// The refusals
// ============================================================================

/// What this lot does not serve refuses by its own name, and is counted as refused.
#[tokio::test]
async fn the_eight_methods_left_refuse_by_name() {
    use project_orchestrator::graph::neighborhood::{NeighborhoodParams, ProjectFilter};
    let store = ObrainGraphStore::open_temporary().unwrap();
    let id = Uuid::new_v4();
    macro_rules! refuse {
        ($call:expr, $name:literal) => {{
            let e = $call.await.unwrap_err();
            assert_eq!(
                e.downcast_ref::<NonPorte>(),
                Some(&NonPorte($name)),
                "{e:#}"
            );
            assert!(refusees_analytics().contains(&$name));
        }};
    }
    refuse!(
        store.entity_exists("function", "f", None),
        "AnalyticsStore::entity_exists"
    );
    refuse!(
        store.entities_linked("function", "a", "CALLS", "function", "b", None),
        "AnalyticsStore::entities_linked"
    );
    refuse!(
        store.ranked_foundation(None, 5),
        "AnalyticsStore::ranked_foundation"
    );
    refuse!(
        store.get_entity_neighborhood(
            "note",
            "x",
            &NeighborhoodParams::clamped(None, None, None, vec![])
        ),
        "AnalyticsStore::get_entity_neighborhood"
    );
    refuse!(
        store.get_scoped_entity_neighborhood(
            "note",
            "x",
            &NeighborhoodParams::clamped(None, None, None, vec![]),
            &ProjectFilter::default()
        ),
        "AnalyticsStore::get_scoped_entity_neighborhood"
    );
    refuse!(store.schema_catalog(), "AnalyticsStore::schema_catalog");
    refuse!(
        store.causal_walk("Note", "x", 2),
        "AnalyticsStore::causal_walk"
    );
    refuse!(
        store.typed_path("Note", "a", "Note", "b", 3),
        "AnalyticsStore::typed_path"
    );
    let _ = id;
    // The fifteen ported methods are counted as serving.
    let servent = methodes_qui_servent();
    for m in [
        "ensure_agent",
        "backfill_agents",
        "link_agent_authored",
        "link_agent_lived",
        "link_agent_knows",
        "link_agent_uses",
        "release_agent_knows",
        "project_synced",
        "list_assertion_notes",
        "record_assertion_check",
        "agent_knows",
        "get_project_decisions_for_graph",
        "get_pm_graph_data",
        "get_chat_graph_data",
        "get_project_constraints",
    ] {
        let name = format!("AnalyticsStore::{m}");
        assert!(servent.contains(&name.as_str()), "{name}");
        assert!(!refusees_analytics().contains(&name.as_str()), "{name}");
    }
}
