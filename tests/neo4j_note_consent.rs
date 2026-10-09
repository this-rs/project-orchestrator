//! `create_note` persists the sharing consent it is given, against a real Neo4j.
//!
//! One file per subject (see `harness_neo4j.rs` for the shared conventions):
//! every scenario uses fresh ids and cleans up after itself. NEVER point it at
//! a live database. Run locally against a throwaway instance:
//!   docker run -d --rm --name po-mig -p 17687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5
//!   NEO4J_URI=bolt://localhost:17687 NEO4J_PASSWORD=testpassword cargo test --test neo4j_note_consent
//!
//! The test prints a notice and returns when no Neo4j is reachable;
//! `HARNESS_NEO4J_REQUIRED=1` (set in CI) turns that into a failure.

use neo4rs::{query, Graph};
use project_orchestrator::episodes::distill_models::SharingConsent;
use project_orchestrator::neo4j::client::Neo4jClient;
use project_orchestrator::notes::{Note, NoteType};
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
        eprintln!("Skipping note consent Neo4j tests: no Neo4j at {uri}");
    }
    connected
}

async fn raw_consent(e: &Env, id: Uuid) -> Option<String> {
    let mut r = e
        .raw
        .execute(
            query("MATCH (n:Note {id: $id}) RETURN n.sharing_consent AS c")
                .param("id", id.to_string()),
        )
        .await
        .unwrap();
    r.next().await.unwrap().unwrap().get("c").ok()
}

#[tokio::test]
async fn create_note_persists_the_given_consent_and_defaults_to_not_set() {
    let Some(e) = env().await else { return };
    let project = Uuid::new_v4();
    let mut ids = Vec::new();
    for consent in [
        SharingConsent::ExplicitAllow,
        SharingConsent::ExplicitDeny,
        SharingConsent::PolicyAuto,
    ] {
        let mut n = Note::new(
            Some(project),
            NoteType::Guideline,
            format!("t0f {}", Uuid::new_v4()),
            "t0f".into(),
        );
        n.sharing_consent = consent;
        e.client.create_note(&n).await.unwrap();
        ids.push(n.id);
        let back = e.client.get_note(n.id).await.unwrap().unwrap();
        assert_eq!(back.sharing_consent, consent, "read back by get_note");
        assert_eq!(
            e.client.get_sharing_consent(n.id).await.unwrap(),
            consent,
            "read back by get_sharing_consent, no update_sharing_consent call"
        );
        assert_eq!(
            raw_consent(&e, n.id).await.as_deref(),
            Some(consent.as_db_str())
        );
    }

    // Old callers: the note keeps the default, NotSet.
    let plain = Note::new(
        Some(project),
        NoteType::Guideline,
        format!("t0f {}", Uuid::new_v4()),
        "t0f".into(),
    );
    e.client.create_note(&plain).await.unwrap();
    ids.push(plain.id);
    let back = e.client.get_note(plain.id).await.unwrap().unwrap();
    assert_eq!(back.sharing_consent, SharingConsent::NotSet);

    let ids: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    e.raw
        .run(query("MATCH (n:Note) WHERE n.id IN $ids DETACH DELETE n").param("ids", ids))
        .await
        .unwrap();
}
