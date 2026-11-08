//! The knowledge repair, measured on a COPY of production.
//!
//! `2026-11-retype-auto-observed` rewrites the type, importance, horizon and
//! energy of every note a heuristic wrote. Three hand-made fixtures cannot say
//! whether that is right or merely plausible: the population is thousands of
//! notes written over months, with a history, mixed in with everything else.
//!
//! So this test runs on a throwaway copy — see `scripts/ephemeral-prod-copy.sh`
//! — picks the project that holds the most un-retyped detector notes, runs the
//! REAL migration through the REAL code path, and prints the before/after it
//! asserts on.
//!
//!   scripts/ephemeral-prod-copy.sh up
//!   NEO4J_URI=bolt://localhost:27699 NEO4J_PASSWORD=testpassword \
//!     cargo test --test retype_auto_observed -- --nocapture
//!
//! Skipped, loudly, when no Neo4j is reachable.

use neo4rs::{query, Graph};
use project_orchestrator::neo4j::client::Neo4jClient;
use uuid::Uuid;

const MIGRATION: &str = "2026-11-retype-auto-observed";

/// The predicate that defines the population, written once.
const AUTO_OBSERVED: &str =
    "(n.created_by IN ['system/observation-detector', 'system/rfc-detector']
     OR 'auto-observed' IN coalesce(n.tags, []))";

#[derive(Debug, PartialEq)]
struct Shape {
    notes: i64,
    claimed_knowledge: i64,
    consolidated: i64,
    active: i64,
}

async fn shape(graph: &Graph, project: &str) -> Shape {
    let cypher = format!(
        r#"
        MATCH (n:Note)
        WHERE n.project_id = $p AND {AUTO_OBSERVED}
        RETURN count(n) AS notes,
               count(CASE WHEN n.note_type IN ['gotcha', 'guideline', 'pattern', 'rfc'] THEN 1 END) AS claimed_knowledge,
               count(CASE WHEN n.memory_horizon = 'consolidated' THEN 1 END) AS consolidated,
               count(CASE WHEN n.status = 'active' THEN 1 END) AS active
        "#
    );
    let mut r = graph
        .execute(query(&cypher).param("p", project))
        .await
        .unwrap();
    let row = r.next().await.unwrap().expect("one row");
    Shape {
        notes: row.get("notes").unwrap_or(0),
        claimed_knowledge: row.get("claimed_knowledge").unwrap_or(0),
        consolidated: row.get("consolidated").unwrap_or(0),
        active: row.get("active").unwrap_or(0),
    }
}

#[tokio::test]
async fn a_regex_does_not_get_to_call_a_sentence_knowledge() {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:27699".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into());

    let client = match Neo4jClient::new(&uri, &user, &password).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping: no Neo4j at {uri}: {e}");
            return;
        }
    };
    let graph = match Graph::new(&uri, &user, &password).await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping: no Neo4j at {uri}: {e}");
            return;
        }
    };

    // The project that carries the most un-retyped detector notes: the copy is
    // real data, so the scope has to be chosen from it rather than invented.
    let pick = format!(
        r#"
        MATCH (n:Note)
        WHERE {AUTO_OBSERVED} AND n.retyped_auto_observed_at IS NULL AND n.project_id IS NOT NULL
        RETURN n.project_id AS p, count(n) AS c
        ORDER BY c DESC LIMIT 1
        "#
    );
    let mut r = graph.execute(query(&pick)).await.unwrap();
    let Some(row) = r.next().await.unwrap() else {
        eprintln!("Skipping: the copy holds no detector note to repair");
        return;
    };
    let project: String = row.get("p").unwrap();
    let scope = Uuid::parse_str(&project).unwrap();
    println!("project {project}: the copy's largest detector population");

    let before = shape(&graph, &project).await;
    println!("BEFORE {before:?}");
    assert!(
        before.claimed_knowledge > 0 && before.consolidated > 0,
        "the copy must reproduce the defect, otherwise this test proves nothing: {before:?}"
    );

    let outcome = client
        .run_data_migration_scoped(MIGRATION, scope)
        .await
        .unwrap();
    assert!(outcome.completed, "{outcome:?}");
    assert_eq!(
        outcome.processed as i64, before.notes,
        "every note of the population"
    );

    let after = shape(&graph, &project).await;
    println!("AFTER  {after:?}");

    assert_eq!(
        after.claimed_knowledge, 0,
        "no note a heuristic wrote still claims to be a gotcha/guideline/pattern/rfc"
    );
    assert_eq!(after.consolidated, 0, "attention no longer buys permanence");
    assert_eq!(after.notes, before.notes, "nothing is deleted");
    assert_eq!(after.active, before.active, "nothing is archived either");

    // The history must say what the note used to claim.
    let history = r#"
        MATCH (n:Note)
        WHERE n.project_id = $p AND n.retyped_auto_observed_at IS NOT NULL
        RETURN count(n) AS retyped,
               count(CASE WHEN n.changes_json CONTAINS 'from_note_type' THEN 1 END) AS with_history,
               count(CASE WHEN n.importance <> 'low' THEN 1 END) AS still_loud,
               count(CASE WHEN n.author_kind <> 'system' THEN 1 END) AS not_system
        "#;
    let mut r = graph
        .execute(query(&history).param("p", project.as_str()))
        .await
        .unwrap();
    let row = r.next().await.unwrap().unwrap();
    let retyped: i64 = row.get("retyped").unwrap_or(0);
    let with_history: i64 = row.get("with_history").unwrap_or(0);
    let still_loud: i64 = row.get("still_loud").unwrap_or(0);
    let not_system: i64 = row.get("not_system").unwrap_or(0);
    println!("HISTORY retyped={retyped} with_history={with_history} still_loud={still_loud}");
    assert_eq!(retyped, before.notes);
    assert_eq!(
        with_history, retyped,
        "each demotion is written in the history"
    );
    assert_eq!(still_loud, 0);
    assert_eq!(not_system, 0);

    // Idempotent: the stamp makes the second run a no-op.
    let again = client
        .run_data_migration_scoped(MIGRATION, scope)
        .await
        .unwrap();
    assert_eq!(again.processed, 0, "a second run moves nothing");
}
