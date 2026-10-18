//! The project of a decision, through ONE traversal
//! (`(Project)-[:HAS_PLAN]->(Plan)-[:HAS_TASK]->(Task)-[:INFORMED_BY]->(Decision)`),
//! against a real Neo4j (T0f-bis).
//!
//! One file per subject (see `harness_neo4j.rs` for the shared conventions):
//! every scenario uses fresh ids and cleans up after itself. NEVER point it at
//! a live database. Run locally against a throwaway instance:
//!   docker run -d --rm --name po-mig -p 17687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5
//!   NEO4J_URI=bolt://localhost:17687 NEO4J_PASSWORD=testpassword cargo test --test neo4j_decision_project
//!
//! The test prints a notice and returns when no Neo4j is reachable;
//! `HARNESS_NEO4J_REQUIRED=1` (set in CI) turns that into a failure.

use neo4rs::{query, Graph};
use project_orchestrator::neo4j::client::Neo4jClient;
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
        eprintln!("Skipping decision project Neo4j tests: no Neo4j at {uri}");
    }
    connected
}

/// `(Project?)-[:HAS_PLAN]->(Plan)-[:HAS_TASK]->(Task)-[:INFORMED_BY]->(Decision)`,
/// written the way the client writes it. `project: None` = a plan without project.
async fn decision_in(e: &Env, project: Option<Uuid>, ids: &mut Vec<String>) -> (Uuid, Uuid) {
    let (plan, task, decision) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let cypher = if project.is_some() {
        "MERGE (proj:Project {id: $proj})
         CREATE (proj)-[:HAS_PLAN]->(:Plan {id: $plan})-[:HAS_TASK]->(t:Task {id: $task})
         CREATE (t)-[:INFORMED_BY]->(:Decision {id: $d, decided_at: datetime(), status: 'Accepted',
                                                description: 't0f-bis', rationale: 'r'})"
    } else {
        "CREATE (:Plan {id: $plan})-[:HAS_TASK]->(t:Task {id: $task})
         CREATE (t)-[:INFORMED_BY]->(:Decision {id: $d, decided_at: datetime(), status: 'Accepted',
                                                description: 't0f-bis', rationale: 'r'})"
    };
    e.raw
        .run(
            query(cypher)
                .param("proj", project.map(|p| p.to_string()).unwrap_or_default())
                .param("plan", plan.to_string())
                .param("task", task.to_string())
                .param("d", decision.to_string()),
        )
        .await
        .unwrap();
    ids.extend([plan, task, decision].map(|i| i.to_string()));
    if let Some(p) = project {
        ids.push(p.to_string());
    }
    (decision, task)
}

async fn cleanup(e: &Env, ids: Vec<String>) {
    e.raw
        .run(query("MATCH (n) WHERE n.id IN $ids DETACH DELETE n").param("ids", ids))
        .await
        .unwrap();
}

/// A unit vector unique to `seed`, in the 768 dimensions of the index.
fn embedding(seed: Uuid) -> Vec<f32> {
    let b = seed.as_bytes();
    (0..768)
        .map(|i| (b[i % 16] as f32 - 127.5) + (i as f32) * 0.01)
        .collect()
}

#[tokio::test]
async fn decision_projects_come_from_one_batched_traversal() {
    let Some(e) = env().await else { return };
    let mut ids = Vec::new();
    let (me, other) = (Uuid::new_v4(), Uuid::new_v4());
    let (own, own_task) = decision_in(&e, Some(me), &mut ids).await;
    let (foreign, _) = decision_in(&e, Some(other), &mut ids).await;
    let (loose, _) = decision_in(&e, None, &mut ids).await;
    let unknown = Uuid::new_v4();

    // A second task of the same project informed by the same decision: still
    // one entry, one project.
    let extra_task = Uuid::new_v4();
    e.raw
        .run(
            query(
                "MATCH (proj:Project {id: $proj})-[:HAS_PLAN]->(p:Plan),
                       (d:Decision {id: $d})
                 CREATE (p)-[:HAS_TASK]->(t:Task {id: $t})-[:INFORMED_BY]->(d)",
            )
            .param("proj", me.to_string())
            .param("d", own.to_string())
            .param("t", extra_task.to_string()),
        )
        .await
        .unwrap();
    ids.push(extra_task.to_string());

    let got = e
        .client
        .get_decision_project_ids(&[own, foreign, loose, unknown])
        .await
        .unwrap();
    let ok = got.len() == 2 && got.get(&own) == Some(&me) && got.get(&foreign) == Some(&other);
    let single = e.client.get_decision_project_id(own).await.unwrap();
    let none = e.client.get_decision_project_id(loose).await.unwrap();
    let empty = e.client.get_decision_project_ids(&[]).await.unwrap();
    cleanup(&e, ids).await;

    assert!(ok, "batched lookup: {got:?}");
    assert_eq!(single, Some(me.to_string()));
    assert_eq!(none, None, "a plan without project is not verifiable");
    assert!(empty.is_empty());
    let _ = own_task;
}

/// The project filter of the decision vector search used
/// `(Plan {id: t.plan_id})-[:BELONGS_TO]->(Project)`: a task has no `plan_id`
/// and no such edge is written, so a project-scoped search returned nothing.
#[tokio::test]
async fn project_scoped_decision_vector_search_finds_the_project_decisions() {
    let Some(e) = env().await else { return };
    let mut ids = Vec::new();
    let (me, other) = (Uuid::new_v4(), Uuid::new_v4());
    let (own, own_task) = decision_in(&e, Some(me), &mut ids).await;
    let (foreign, _) = decision_in(&e, Some(other), &mut ids).await;
    // Informing two tasks of the project must not duplicate the hit.
    let extra_task = Uuid::new_v4();
    e.raw
        .run(
            query(
                "MATCH (t0:Task {id: $t0})<-[:HAS_TASK]-(p:Plan), (d:Decision {id: $d})
                 CREATE (p)-[:HAS_TASK]->(:Task {id: $t})-[:INFORMED_BY]->(d)",
            )
            .param("t0", own_task.to_string())
            .param("d", own.to_string())
            .param("t", extra_task.to_string()),
        )
        .await
        .unwrap();
    ids.push(extra_task.to_string());
    let v = embedding(Uuid::new_v4());
    for d in [own, foreign] {
        e.client
            .set_decision_embedding(d, &v, "t0f-bis")
            .await
            .unwrap();
    }

    // The vector index is eventually consistent: poll briefly.
    let mut hits = Vec::new();
    for _ in 0..20 {
        hits = e
            .client
            .search_decisions_by_vector(&v, 5, Some(&me.to_string()))
            .await
            .unwrap()
            .into_iter()
            .map(|(d, _)| d.id)
            .collect::<Vec<_>>();
        if !hits.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    cleanup(&e, ids).await;

    assert_eq!(hits, vec![own], "only the project's decision, once");
}

/// The backfill's decision filter now goes through the same traversal (an
/// `EXISTS` subquery): the Cypher must at least be accepted by the server.
#[tokio::test]
async fn backfill_produced_during_cypher_is_valid() {
    let Some(e) = env().await else { return };
    assert_eq!(
        e.client
            .backfill_produced_during(Uuid::new_v4())
            .await
            .unwrap(),
        0
    );
}
