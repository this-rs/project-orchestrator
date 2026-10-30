//! Le soi comme SUJET : les arêtes qui le relient au graphe, et ce qui les rend
//! contestables.
//!
//! Ces tests portent sur les familles de T2 (`AUTHORED`, `LIVED`, `KNOWS`) et sur leur
//! lecture. Ils tournent sur `MockGraphStore`, donc sans base : ce qu'ils prouvent est la
//! **règle**, pas le Cypher. Le Cypher est prouvé séparément, contre une vraie base.

use uuid::Uuid;

use crate::graph::rules::Provenance;
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::traits::*;
use crate::notes::models::{Note, NoteType};

fn a_note(content: &str, written_by: &str) -> Note {
    Note::new(None, NoteType::Guideline, content.into(), written_by.into())
}

#[tokio::test]
async fn a_note_written_by_an_agent_is_that_agents_own() {
    // The edge is drawn AT THE WRITE, not left to a backfill: a subject that only learns
    // what it wrote once someone runs a migration is a subject with amnesia.
    let graph = MockGraphStore::new();
    let agent = Uuid::new_v4();
    let note = a_note("a finding", &format!("agent:{agent}"));

    graph.create_note(&note).await.unwrap();

    assert!(
        graph
            .agent_authored
            .read()
            .await
            .contains(&(agent, note.id)),
        "the author must own the note it wrote"
    );
    assert!(
        graph.agents.lock().unwrap().contains_key(&agent),
        "and the author must exist as a node, made by the write"
    );
}

#[tokio::test]
async fn a_note_written_by_a_person_attributes_nothing() {
    // A write that names no agent must not invent one: most notes in this graph are
    // written by a person or an unnamed caller, and a subject that claimed them would be
    // claiming everything.
    let graph = MockGraphStore::new();
    graph
        .create_note(&a_note("hand written", "api"))
        .await
        .unwrap();

    assert!(graph.agent_authored.read().await.is_empty());
    assert!(graph.agents.lock().unwrap().is_empty());
}

#[tokio::test]
async fn an_episode_gives_its_agent_the_session_it_lived() {
    let graph = MockGraphStore::new();
    let project = crate::test_helpers::test_project_named("proj");
    graph.create_project(&project).await.unwrap();
    let agent = Uuid::new_v4();
    let mut session = crate::test_helpers::test_chat_session(Some("proj"));
    // A session with something to keep: a preview and a few messages. An empty one
    // leaves no episode at all, which is the collector's own rule.
    session.preview = Some("first message".into());
    session.message_count = 4;
    let sid = session.id;
    graph.create_chat_session(&session).await.unwrap();

    let mut episode =
        crate::episodes::collector::collect_session_episode(&graph, sid, "user_closed")
            .await
            .unwrap()
            .unwrap();
    episode.session.as_mut().unwrap().agent_id = Some(agent);
    assert!(graph.save_session_episode(&episode).await.unwrap());

    assert!(
        graph
            .agent_lived
            .read()
            .await
            .contains(&(agent, episode.id)),
        "the agent that lived a session must own its episode"
    );
    assert_eq!(
        episode.session.unwrap().close_reason,
        "user_closed".to_string(),
        "and the episode must say how the session ended"
    );
}

#[tokio::test]
async fn what_a_subject_knows_is_a_note_never_a_persona_file() {
    // KNOWS already meant `(Persona)-[:KNOWS]->(File)`. The new meaning is
    // `(Agent)-[:KNOWS]->(Note)`, and the two must never contaminate each other: a read of
    // one that returned the other would be silently wrong.
    let graph = MockGraphStore::new();
    let note = a_note("a belief", "api");
    graph.create_note(&note).await.unwrap();
    let self_id = graph.agent_identity_namespace().await.unwrap();

    assert!(graph
        .link_agent_knows(
            self_id,
            note.id,
            &Provenance::new("agent-knows-core", "1 portée, 1 note")
        )
        .await
        .unwrap());

    let known = graph.agent_knows(self_id, 10).await.unwrap();
    assert_eq!(known.notes, vec![note.id]);
    assert_eq!(known.available, 1);
    assert_eq!(known.withheld(), 0);
}

#[tokio::test]
async fn two_subjects_do_not_share_their_knowledge() {
    // The isolation the plan asks for, at the level that makes it true: the read is
    // filtered by agent id, so a second subject sees nothing of the first.
    let graph = MockGraphStore::new();
    let note = a_note("a belief", "api");
    graph.create_note(&note).await.unwrap();

    let first = Uuid::new_v4();
    let second = Uuid::new_v4();
    let prov = Provenance::new("agent-knows-core", "measured");
    graph.link_agent_knows(first, note.id, &prov).await.unwrap();

    assert_eq!(graph.agent_knows(first, 10).await.unwrap().notes.len(), 1);
    assert!(graph
        .agent_knows(second, 10)
        .await
        .unwrap()
        .notes
        .is_empty());
}

#[tokio::test]
async fn a_truncated_read_says_what_it_left_out() {
    // « La part lue est dite ». The bound is published with the answer, so a caller can
    // never mistake a partial read for the whole of what the subject knows.
    let graph = MockGraphStore::new();
    let self_id = graph.agent_identity_namespace().await.unwrap();
    let prov = Provenance::new("agent-knows-core", "measured");
    for i in 0..3 {
        let note = a_note(&format!("belief {i}"), "api");
        graph.create_note(&note).await.unwrap();
        graph
            .link_agent_knows(self_id, note.id, &prov)
            .await
            .unwrap();
    }

    let known = graph.agent_knows(self_id, 2).await.unwrap();
    assert_eq!(known.notes.len(), 2);
    assert_eq!(known.available, 3);
    assert_eq!(
        known.withheld(),
        1,
        "the missing one is counted, not hidden"
    );
}

#[tokio::test]
async fn a_knows_edge_always_carries_its_measure() {
    // The edge is only worth what its provenance is worth: this reads it back off the
    // edge, not off the rule that made it.
    let graph = MockGraphStore::new();
    let note = a_note("a belief", "api");
    graph.create_note(&note).await.unwrap();
    let self_id = graph.agent_identity_namespace().await.unwrap();
    let prov = Provenance::new("agent-knows-core", "3 portées lues, 1 note épinglée");

    graph
        .link_agent_knows(self_id, note.id, &prov)
        .await
        .unwrap();

    let stored = graph
        .agent_knows
        .read()
        .await
        .get(&(self_id, note.id))
        .cloned()
        .expect("the edge exists");
    assert!(stored.is_accountable());
    assert_eq!(stored.rule, "agent-knows-core");
    assert!(stored.measure.contains('3'));
}

#[tokio::test]
async fn a_knows_edge_to_a_missing_note_draws_nothing() {
    // A pin naming a deleted note is a skip, not a dangling edge and not an error.
    let graph = MockGraphStore::new();
    let self_id = graph.agent_identity_namespace().await.unwrap();
    let drawn = graph
        .link_agent_knows(
            self_id,
            Uuid::new_v4(),
            &Provenance::new("agent-knows-core", "measured"),
        )
        .await
        .unwrap();
    assert!(!drawn);
}

#[tokio::test]
async fn the_backfill_rebuilds_the_two_attribution_families() {
    // One path, not two: what the write draws, the backfill rebuilds — and it says how
    // many edges it drew instead of letting the caller assume.
    let graph = MockGraphStore::new();
    let agent = Uuid::new_v4();
    graph
        .create_note(&a_note("a finding", &format!("agent:{agent}")))
        .await
        .unwrap();

    let report = graph.backfill_agents().await.unwrap();
    assert_eq!(report.from_notes, 1);
    assert_eq!(report.authored, 0, "the write already drew it");
    assert_eq!(
        report.after, report.before,
        "the node was made at the write, so the backfill adds none"
    );
}

#[tokio::test]
async fn an_activation_becomes_a_durable_uses_edge() {
    // The registry that holds activations dies with the process. `USES` is the same fact,
    // written where it survives — and it names the rule and the session that measured it.
    use crate::graph::rules::{run_with, Measurement, AGENT_USES_ACTIVATED};
    let graph = MockGraphStore::new();
    let agent = Uuid::new_v4();
    let persona = Uuid::new_v4();
    let skill = Uuid::new_v4();

    let outcome = run_with(
        &graph,
        AGENT_USES_ACTIVATED.name,
        Measurement::Activation {
            session: Uuid::new_v4(),
            agent,
            personas: vec![persona],
            skills: vec![skill],
        },
    )
    .await
    .unwrap();

    assert_eq!(outcome.considered, 2);
    assert_eq!(outcome.written, 2, "{outcome:?}");
    let uses = graph.agent_uses.read().await;
    let (_, label, target) = uses.keys().find(|(a, _, _)| *a == agent).unwrap();
    assert_eq!(label, "Persona");
    assert_eq!(*target, persona);
    let stored = uses.values().next().unwrap();
    assert!(stored.is_accountable(), "the edge must say why it exists");
    assert_eq!(stored.rule, AGENT_USES_ACTIVATED.name);
}

#[tokio::test]
async fn an_activation_of_nothing_writes_nothing() {
    use crate::graph::rules::{run_with, Measurement, AGENT_USES_ACTIVATED};
    let graph = MockGraphStore::new();
    let outcome = run_with(
        &graph,
        AGENT_USES_ACTIVATED.name,
        Measurement::Activation {
            session: Uuid::new_v4(),
            agent: Uuid::new_v4(),
            personas: vec![],
            skills: vec![],
        },
    )
    .await
    .unwrap();
    assert_eq!(outcome.written, 0);
    assert_eq!(outcome.considered, 0);
    assert!(graph.agent_uses.read().await.is_empty());
}

#[tokio::test]
async fn a_session_welds_the_notes_it_had_in_context() {
    // T5: what the turn really had in context becomes structure, and each edge says which
    // session did it. Three notes → three pairs, both directions.
    use crate::graph::rules::{run_with, Measurement, SESSION_CO_ACTIVATION};
    let graph = MockGraphStore::new();
    let project = Uuid::new_v4();
    let session = Uuid::new_v4();
    let notes: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();

    let outcome = run_with(
        &graph,
        SESSION_CO_ACTIVATION.name,
        Measurement::SessionNotes {
            session,
            project: Some(project),
            notes: notes.clone(),
        },
    )
    .await
    .unwrap();

    assert_eq!(outcome.considered, 3);
    assert_eq!(outcome.written, 6, "3 pairs, both directions: {outcome:?}");
    let synapses = graph.synapses.read().await;
    assert_eq!(synapses.len(), 3);
    let (owner, welder, provenance) = synapses.values().next().unwrap();
    assert_eq!(*owner, project);
    assert_eq!(*welder, session);
    assert_eq!(provenance.rule, SESSION_CO_ACTIVATION.name);
}

#[tokio::test]
async fn a_session_with_one_note_or_no_project_welds_nothing() {
    use crate::graph::rules::{run_with, Measurement, SESSION_CO_ACTIVATION};
    let graph = MockGraphStore::new();

    let single = run_with(
        &graph,
        SESSION_CO_ACTIVATION.name,
        Measurement::SessionNotes {
            session: Uuid::new_v4(),
            project: Some(Uuid::new_v4()),
            notes: vec![Uuid::new_v4()],
        },
    )
    .await
    .unwrap();
    assert_eq!(single.written, 0);
    assert!(single.measure.contains('1'), "{}", single.measure);

    let homeless = run_with(
        &graph,
        SESSION_CO_ACTIVATION.name,
        Measurement::SessionNotes {
            session: Uuid::new_v4(),
            project: None,
            notes: vec![Uuid::new_v4(), Uuid::new_v4()],
        },
    )
    .await
    .unwrap();
    assert_eq!(homeless.written, 0);
    assert!(graph.synapses.read().await.is_empty());
}

#[tokio::test]
async fn an_unusable_label_is_refused_before_any_query() {
    let graph = MockGraphStore::new();
    let err = graph
        .link_agent_uses(
            Uuid::new_v4(),
            "User; MATCH (n) DETACH DELETE n //",
            Uuid::new_v4(),
            &Provenance::new("agent-uses-activated", "measured"),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not a usable label"), "{err}");
}

#[test]
fn the_rule_engine_names_no_language_model() {
    // The constraint is mechanical, so the test is too: the module may use an embedding
    // provider (a vector is a measurement) but must never reach a model that decides.
    let source = include_str!("rules.rs");
    for forbidden in ["anthropic", "openai", "ollama", "mistral", "LlmProvider"] {
        assert!(
            !source.to_lowercase().contains(&forbidden.to_lowercase()),
            "the rule engine must not name {forbidden}"
        );
    }
}

#[test]
fn every_close_reason_is_distinct_and_named() {
    // Three real endings is the acceptance; the list is what makes it checkable.
    let all = crate::episodes::models::close_reason::ALL;
    assert!(all.len() >= 3, "{all:?}");
    let mut seen = std::collections::HashSet::new();
    for r in all {
        assert!(seen.insert(*r), "duplicate close reason {r}");
        assert!(
            !r.is_empty() && r.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
            "{r} must be a bare identifier"
        );
    }
}
