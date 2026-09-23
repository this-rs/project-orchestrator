//! Data migrations against a real Neo4j (see `neo4j::data_migrations`).
//!
//! These are what repairs an UPGRADED install: legacy alert backlog,
//! duplicate skills, and archived skills nothing references anymore.
//!
//! Every scenario is scoped to a fresh project id, so it can share the CI
//! Neo4j with other suites. Scenarios run sequentially inside ONE test: the
//! global run at the end must not race the scoped setups.
//!
//! Run locally against a throwaway instance:
//!   docker run -d --rm --name po-mig -p 17687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5
//!   NEO4J_URI=bolt://localhost:17687 NEO4J_PASSWORD=testpassword cargo test --test data_migrations

use neo4rs::{query, Graph};
use project_orchestrator::neo4j::client::Neo4jClient;
use uuid::Uuid;

struct Env {
    client: Neo4jClient,
    raw: Graph,
}

async fn env() -> Option<Env> {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into());
    let client = match Neo4jClient::new(&uri, &user, &password).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping data migration tests: Neo4j unavailable at {uri}: {e}");
            return None;
        }
    };
    let raw = Graph::new(&uri, &user, &password).await.ok()?;
    Some(Env { client, raw })
}

async fn count(raw: &Graph, cypher: &str, project: &str) -> i64 {
    let mut r = raw
        .execute(query(cypher).param("p", project))
        .await
        .unwrap();
    r.next().await.unwrap().unwrap().get::<i64>("c").unwrap()
}

/// A skill node as the server stores it.
async fn skill(raw: &Graph, project: &str, name: &str, status: &str, age_days: i64) -> String {
    let id = Uuid::new_v4().to_string();
    raw.run(
        query(
            "CREATE (:Skill {id: $id, project_id: $p, name: $name, status: $status,
                     energy: 0.0, created_at: datetime() - duration({days: $age}),
                     updated_at: datetime() - duration({days: $age})})",
        )
        .param("id", id.clone())
        .param("p", project)
        .param("name", name)
        .param("status", status)
        .param("age", age_days),
    )
    .await
    .unwrap();
    id
}

async fn add_members(raw: &Graph, skill_id: &str, n: i64) {
    raw.run(
        query(
            "MATCH (s:Skill {id: $id})
             UNWIND range(1, $n) AS i
             CREATE (:Note {id: randomUUID()})-[:MEMBER_OF]->(s)",
        )
        .param("id", skill_id)
        .param("n", n),
    )
    .await
    .unwrap();
}

async fn status_of(raw: &Graph, skill_id: &str) -> Option<String> {
    let mut r = raw
        .execute(query("MATCH (s:Skill {id: $id}) RETURN s.status AS st").param("id", skill_id))
        .await
        .unwrap();
    r.next()
        .await
        .unwrap()
        .map(|row| row.get::<String>("st").unwrap())
}

#[tokio::test]
async fn test_data_migrations_repair_an_upgraded_install() {
    let Some(Env { client, raw }) = env().await else {
        return;
    };

    // ------------------------------------------------------------------
    // 1. Legacy alerts (append-only, no dedup_key) fold into one node per
    //    condition, merging into the node the server already keyed.
    // ------------------------------------------------------------------
    let p = Uuid::new_v4();
    let ps = p.to_string();
    raw.run(
        query(
            "UNWIND range(1, 5) AS i
             CREATE (:Alert {id: randomUUID(), alert_type: 'git_drift', project_id: $p,
                     message: 'behind origin by ' + i, severity: 'info',
                     acknowledged: i = 3, created_at: datetime() - duration({hours: i})})
             WITH count(*) AS _
             CREATE (:Alert {id: randomUUID(), alert_type: 'git_drift', project_id: $p,
                     dedup_key: 'git_drift:' + $p + ':behind-origin', message: 'current',
                     severity: 'warning', occurrence_count: 2, acknowledged: false,
                     created_at: datetime()})",
        )
        .param("p", ps.clone()),
    )
    .await
    .unwrap();

    let out = client
        .run_data_migration_scoped("2026-09-fold-legacy-alerts", p)
        .await
        .unwrap();
    assert!(out.completed, "{out:?}");
    assert_eq!(out.processed, 5);
    assert_eq!(
        count(
            &raw,
            "MATCH (a:Alert {project_id: $p}) RETURN count(a) AS c",
            &ps
        )
        .await,
        1,
        "one node per condition"
    );
    let mut r = raw
        .execute(
            query(
                "MATCH (a:Alert {project_id: $p})
                 RETURN a.occurrence_count AS occ, a.acknowledged AS ack, a.severity AS sev,
                        a.message AS msg",
            )
            .param("p", ps.clone()),
        )
        .await
        .unwrap();
    let row = r.next().await.unwrap().unwrap();
    assert_eq!(row.get::<i64>("occ").unwrap(), 7, "2 existing + 5 folded");
    assert!(
        row.get::<bool>("ack").unwrap(),
        "never silently un-acknowledge"
    );
    assert_eq!(
        row.get::<String>("sev").unwrap(),
        "warning",
        "highest severity kept"
    );
    assert_eq!(
        row.get::<String>("msg").unwrap(),
        "current",
        "keyed node keeps its message"
    );

    // Idempotent: nothing left to fold.
    let again = client
        .run_data_migration_scoped("2026-09-fold-legacy-alerts", p)
        .await
        .unwrap();
    assert_eq!(again.processed, 0);

    // ------------------------------------------------------------------
    // 2. Duplicate live skills: keep the richest, plus any referenced copy;
    //    archive the unreferenced rest and detach their members.
    // ------------------------------------------------------------------
    let q = Uuid::new_v4();
    let qs = q.to_string();
    let rich = skill(&raw, &qs, "Toolchain Build Cargo", "emerging", 3).await;
    add_members(&raw, &rich, 5).await;
    let poor = skill(&raw, &qs, "Toolchain Build Cargo", "emerging", 5).await;
    add_members(&raw, &poor, 2).await;
    let empty_dup = skill(&raw, &qs, "Toolchain Build Cargo", "emerging", 1).await;
    // A protocol references this duplicate by property: it must survive.
    let protocol_ref = skill(&raw, &qs, "Toolchain Build Cargo", "emerging", 2).await;
    raw.run(
        query("CREATE (:Protocol {id: randomUUID(), project_id: $p, skill_id: $s})")
            .param("p", qs.clone())
            .param("s", protocol_ref.clone()),
    )
    .await
    .unwrap();
    let solo = skill(&raw, &qs, "Unique Skill", "emerging", 1).await;

    let out = client
        .run_data_migration_scoped("2026-09-archive-duplicate-skills", q)
        .await
        .unwrap();
    assert!(out.completed, "{out:?}");
    assert_eq!(out.processed, 2, "poor + empty_dup archived");
    assert_eq!(
        status_of(&raw, &protocol_ref).await.as_deref(),
        Some("emerging")
    );
    assert_eq!(status_of(&raw, &rich).await.as_deref(), Some("emerging"));
    assert_eq!(status_of(&raw, &solo).await.as_deref(), Some("emerging"));
    assert_eq!(status_of(&raw, &poor).await.as_deref(), Some("archived"));
    assert_eq!(
        status_of(&raw, &empty_dup).await.as_deref(),
        Some("archived")
    );
    assert_eq!(
        count(
            &raw,
            "MATCH (:Note)-[:MEMBER_OF]->(s:Skill {project_id: $p, status: 'archived'}) RETURN count(*) AS c",
            &qs
        )
        .await,
        0,
        "archived duplicates lose their members"
    );
    assert_eq!(
        count(
            &raw,
            "MATCH (:Note)-[:MEMBER_OF]->(s:Skill {project_id: $p}) RETURN count(*) AS c",
            &qs
        )
        .await,
        5,
        "the kept skill keeps its members"
    );

    // ------------------------------------------------------------------
    // 3. Purge archived skills nothing references; keep referenced ones.
    // ------------------------------------------------------------------
    let archived_referenced = skill(&raw, &qs, "Old Protocol Skill", "archived", 30).await;
    raw.run(
        query("CREATE (:Protocol {id: randomUUID(), project_id: $p, skill_id: $s})")
            .param("p", qs.clone())
            .param("s", archived_referenced.clone()),
    )
    .await
    .unwrap();
    let archived_mastered = skill(&raw, &qs, "Mastered", "archived", 30).await;
    raw.run(
        query("MATCH (s:Skill {id: $s}) CREATE (:Persona {id: randomUUID()})-[:MASTERS]->(s)")
            .param("s", archived_mastered.clone()),
    )
    .await
    .unwrap();

    let out = client
        .run_data_migration_scoped("2026-09-purge-empty-archived-skills", q)
        .await
        .unwrap();
    assert!(out.completed, "{out:?}");
    assert_eq!(out.processed, 2, "poor + empty_dup deleted");
    assert_eq!(status_of(&raw, &poor).await, None);
    assert_eq!(status_of(&raw, &empty_dup).await, None);
    assert_eq!(
        status_of(&raw, &archived_referenced).await.as_deref(),
        Some("archived")
    );
    assert_eq!(
        status_of(&raw, &archived_mastered).await.as_deref(),
        Some("archived")
    );

    // ------------------------------------------------------------------
    // 4. Recurring purge (deep maintenance): only skills archived before the
    //    retention cutoff.
    // ------------------------------------------------------------------
    let recent = skill(&raw, &qs, "Recently Archived", "archived", 1).await;
    let old = skill(&raw, &qs, "Long Archived", "archived", 30).await;
    let purged = client
        .purge_archived_empty_skills(q, chrono::Utc::now() - chrono::Duration::days(7))
        .await
        .unwrap();
    assert_eq!(purged, 1);
    assert_eq!(status_of(&raw, &recent).await.as_deref(), Some("archived"));
    assert_eq!(status_of(&raw, &old).await, None);

    // ------------------------------------------------------------------
    // 5. The startup runner completes every migration once, then skips.
    // ------------------------------------------------------------------
    let first = client.run_data_migrations().await;
    assert_eq!(first.len(), 3);
    assert!(
        first.iter().all(|o| o.completed && o.error.is_none()),
        "{first:?}"
    );
    let second = client.run_data_migrations().await;
    assert!(second.iter().all(|o| o.skipped), "{second:?}");
}
