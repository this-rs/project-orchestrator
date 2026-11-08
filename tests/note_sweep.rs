//! The memory sweep and the repair of what a heuristic wrote, against a real
//! Neo4j.
//!
//! `consolidate_memory` decides what becomes permanent and what is put aside.
//! Its rules are pure arithmetic and each one has a branch that only a real
//! store can exercise: archival by age, archival by energy, the ephemeral
//! grace period, promotion on evidence, and the notes it must leave alone.
//!
//! Run locally against a throwaway instance:
//!   docker run -d --rm --name po-sweep -p 17687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5
//!   NEO4J_URI=bolt://localhost:17687 NEO4J_PASSWORD=testpassword cargo test --test note_sweep

#[allow(unused_imports)]
use project_orchestrator::neo4j::traits::*;

use neo4rs::{query, Graph};
use project_orchestrator::neo4j::client::Neo4jClient;
use project_orchestrator::notes::models::{
    MemoryHorizon, Note, NoteAuthorKind, NoteImportance, NoteStatus, NoteType,
};
use uuid::Uuid;

struct Env {
    client: Neo4jClient,
    raw: Graph,
}

async fn env() -> Option<Env> {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:17687".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into());
    let client = match Neo4jClient::new(&uri, &user, &password).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping the sweep tests: no Neo4j at {uri}: {e}");
            return None;
        }
    };
    let raw = Graph::new(&uri, &user, &password).await.ok()?;
    Some(Env { client, raw })
}

/// Put a note's creation date in the past, the way a long-lived install's notes
/// look.
async fn age(raw: &Graph, id: Uuid, days: i64) {
    raw.run(
        query(
            "MATCH (n:Note {id: $id}) SET n.created_at = datetime() - duration({days: $d}),
             n.last_activated = datetime() - duration({days: $d})",
        )
        .param("id", id.to_string())
        .param("d", days),
    )
    .await
    .unwrap();
}

async fn read(store: &dyn GraphStore, id: Uuid) -> Note {
    store.get_note(id).await.unwrap().expect("the note exists")
}

#[tokio::test]
async fn the_sweep_archives_promotes_and_leaves_alone() {
    let Some(Env { client, raw }) = env().await else {
        return;
    };
    let store: &dyn GraphStore = &client;
    let project = Uuid::new_v4();

    // 1. Dead note: no usage, older than 90 days.
    let dead = Note::new(
        Some(project),
        NoteType::Tip,
        "a tip nobody ever read".to_string(),
        "agent:x".to_string(),
    );
    // 2. Drained note: energy below the floor, older than 60 days.
    let mut drained = Note::new(
        Some(project),
        NoteType::Context,
        "a context whose energy ran out".to_string(),
        "agent:x".to_string(),
    );
    drained.energy = 0.01;
    // 3. Ephemeral note, idle past its grace period.
    let mut short_lived = Note::new(
        Some(project),
        NoteType::Observation,
        "an observation nobody came back to".to_string(),
        "system/observation-detector".to_string(),
    )
    .with_author(NoteAuthorKind::System, "system/observation-detector");
    short_lived.memory_horizon = MemoryHorizon::Ephemeral;
    // 4. Consolidated: permanent by construction, never archived nor touched.
    let mut permanent = Note::new(
        Some(project),
        NoteType::Guideline,
        "a guideline that earned its place".to_string(),
        "julien".to_string(),
    )
    .with_author(NoteAuthorKind::Human, "julien");
    permanent.memory_horizon = MemoryHorizon::Consolidated;
    permanent.energy = 0.0;
    // 5. Evidence: a person confirmed it, so it may become permanent.
    let mut confirmed = Note::new(
        Some(project),
        NoteType::Gotcha,
        "a gotcha a person read and vouched for".to_string(),
        "agent:x".to_string(),
    );
    confirmed.memory_horizon = MemoryHorizon::Operational;
    confirmed.confirm_by_human("julien", chrono::Utc::now());
    // 6. Attention only: read a lot, never vouched for. Stays operational.
    let mut attention = Note::new(
        Some(project),
        NoteType::Pattern,
        "a pattern injected ten times and never checked".to_string(),
        "agent:x".to_string(),
    );
    attention.memory_horizon = MemoryHorizon::Operational;
    attention.reactivation_count = 9;

    for note in [
        &dead,
        &drained,
        &short_lived,
        &permanent,
        &confirmed,
        &attention,
    ] {
        store.create_note(note).await.unwrap();
    }

    age(&raw, dead.id, 120).await;
    age(&raw, drained.id, 90).await;
    age(&raw, short_lived.id, 4).await;
    age(&raw, attention.id, 30).await;

    client.consolidate_memory().await.unwrap();

    assert_eq!(read(store, dead.id).await.status, NoteStatus::Archived);
    assert_eq!(read(store, drained.id).await.status, NoteStatus::Archived);
    assert_eq!(
        read(store, short_lived.id).await.status,
        NoteStatus::Archived,
        "an ephemeral nobody came back to is put aside"
    );
    assert_eq!(
        read(store, permanent.id).await.status,
        NoteStatus::Active,
        "consolidated is permanent: never archived, not even at zero energy"
    );
    assert_eq!(
        read(store, confirmed.id).await.memory_horizon,
        MemoryHorizon::Consolidated,
        "a person's confirmation is evidence"
    );
    assert_eq!(
        read(store, attention.id).await.memory_horizon,
        MemoryHorizon::Operational,
        "nine re-injections are attention, not evidence"
    );

    // The sweep is replayable on this population: a second pass leaves these
    // six notes exactly where the first one put them.
    let before = [
        read(store, dead.id).await,
        read(store, drained.id).await,
        read(store, short_lived.id).await,
        read(store, permanent.id).await,
        read(store, confirmed.id).await,
        read(store, attention.id).await,
    ];
    client.consolidate_memory().await.unwrap();
    let after = [
        read(store, dead.id).await,
        read(store, drained.id).await,
        read(store, short_lived.id).await,
        read(store, permanent.id).await,
        read(store, confirmed.id).await,
        read(store, attention.id).await,
    ];
    for (b, a) in before.iter().zip(after.iter()) {
        assert_eq!(
            (b.status, b.memory_horizon.clone()),
            (a.status, a.memory_horizon.clone()),
            "a second sweep moves nothing that was already settled"
        );
    }
}

/// The two janitor methods, through the trait — the same path the HTTP route and
/// the MCP action take.
#[tokio::test]
async fn the_report_and_the_archive_agree_on_who_wrote_what() {
    let Some(Env { client, .. }) = env().await else {
        return;
    };
    let store: &dyn GraphStore = &client;
    let project = Uuid::new_v4();

    let mut detector = Note::new(
        Some(project),
        NoteType::Observation,
        "**Gotcha**: un fragment de prose arraché à une conversation.".to_string(),
        "system/observation-detector".to_string(),
    )
    .with_author(NoteAuthorKind::System, "system/observation-detector");
    detector.tags = vec!["auto-observed".to_string()];
    detector.importance = NoteImportance::Low;

    let mut confirmed = Note::new(
        Some(project),
        NoteType::Observation,
        "un fragment qu'un humain a lu et validé.".to_string(),
        "system/observation-detector".to_string(),
    )
    .with_author(NoteAuthorKind::System, "system/observation-detector");
    confirmed.tags = vec!["auto-observed".to_string()];
    confirmed.confirm_by_human("julien", chrono::Utc::now());

    let human = Note::new(
        Some(project),
        NoteType::Gotcha,
        "une note écrite par une personne.".to_string(),
        "julien".to_string(),
    )
    .with_author(NoteAuthorKind::Human, "julien");

    for note in [&detector, &confirmed, &human] {
        store.create_note(note).await.unwrap();
    }

    let report = store.auto_observed_report(Some(project)).await.unwrap();
    let reported: i64 = report
        .iter()
        .map(|row| row["notes"].as_i64().unwrap_or(0))
        .sum();
    assert_eq!(reported, 2, "both observations, whatever their shape");
    assert!(
        report.iter().all(|row| row["note_type"] == "observation"),
        "the report never calls them anything else: {report:?}"
    );
    assert!(report.iter().all(|row| row["status"] == "active"));

    let archived = store
        .archive_unconfirmed_auto_observed(Some(project))
        .await
        .unwrap();
    assert_eq!(
        archived, 1,
        "the confirmed one and the person's note are left alone"
    );

    assert_eq!(read(store, detector.id).await.status, NoteStatus::Archived);
    assert_eq!(
        read(store, confirmed.id).await.status,
        NoteStatus::Active,
        "a person confirmed it: the janitor has no business with it"
    );
    assert_eq!(read(store, human.id).await.status, NoteStatus::Active);

    // Nothing left to archive: the loop ends on a zero, not on a coincidence.
    assert_eq!(
        store
            .archive_unconfirmed_auto_observed(Some(project))
            .await
            .unwrap(),
        0
    );
}

/// Confirming is a signature only when a PERSON gives it, and it never
/// resurrects a note a newer one superseded.
#[tokio::test]
async fn confirming_signs_for_a_person_and_for_nobody_else() {
    let Some(Env { client, .. }) = env().await else {
        return;
    };
    let store: &dyn GraphStore = &client;
    let project = Uuid::new_v4();

    let agent_confirmed = Note::new(
        Some(project),
        NoteType::Observation,
        "un fragment confirmé par un agent, pas par un humain.".to_string(),
        "system/observation-detector".to_string(),
    )
    .with_author(NoteAuthorKind::System, "system/observation-detector");
    let human_confirmed = Note::new(
        Some(project),
        NoteType::Gotcha,
        "une note confirmée par une personne.".to_string(),
        "agent:x".to_string(),
    );
    for note in [&agent_confirmed, &human_confirmed] {
        store.create_note(note).await.unwrap();
    }

    // An agent's confirmation moves the freshness clock, never the signature.
    let after_agent = store
        .confirm_note(
            agent_confirmed.id,
            "agent:204ce865-d253-4f59-816c-5f9d8f5ff1ac",
        )
        .await
        .unwrap()
        .expect("the note exists");
    assert!(
        !after_agent.is_human_confirmed(),
        "a model does not answer for a note"
    );
    assert!(after_agent.last_confirmed_at.is_some());

    // A person's does sign it.
    let after_human = store
        .confirm_note(human_confirmed.id, "julien")
        .await
        .unwrap()
        .expect("the note exists");
    assert!(after_human.is_human_confirmed());
    assert_eq!(after_human.human_confirmed_by.as_deref(), Some("julien"));

    // A superseded note is not resurrected by a confirmation: newer wins.
    let older = Note::new(
        Some(project),
        NoteType::Context,
        "une note remplacée par une plus récente.".to_string(),
        "agent:x".to_string(),
    );
    store.create_note(&older).await.unwrap();
    raw_supersede(&client, older.id).await;

    let after = store
        .confirm_note(older.id, "julien")
        .await
        .unwrap()
        .expect("the note exists");
    assert_ne!(
        after.status,
        NoteStatus::Active,
        "confirming a superseded note must not bring it back"
    );
}

/// Mark a note as superseded by another, the way `note(supersede)` does.
async fn raw_supersede(client: &Neo4jClient, id: Uuid) {
    let replacement = Note::new(
        None,
        NoteType::Tip,
        "la version récente".to_string(),
        "a".to_string(),
    );
    client.create_note(&replacement).await.unwrap();
    client.supersede_note(id, replacement.id).await.unwrap();
}
