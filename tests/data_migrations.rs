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
#[allow(unused_imports)]
use project_orchestrator::neo4j::traits::*;

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

/// The ids of every non-File node of a project, sorted — the shadow key must
/// leave them untouched.
async fn ids_in_scope(raw: &Graph, project: &str) -> Vec<String> {
    let mut r = raw
        .execute(
            query(
                "MATCH (n) WHERE NOT n:File AND (n.project_id = $p
                       OR EXISTS { (n)<-[:CONTAINS|HAS_IMPORT]-(:File {project_id: $p}) })
                 RETURN collect(n.id) AS ids",
            )
            .param("p", project),
        )
        .await
        .unwrap();
    let mut ids: Vec<String> = r.next().await.unwrap().unwrap().get("ids").unwrap();
    ids.sort();
    ids
}

/// `(stable_key, owner_type, owner_trait)` of the node with this legacy id.
async fn key_of(raw: &Graph, id: &str) -> (Option<String>, Option<String>, Option<String>) {
    let mut r = raw
        .execute(
            query(
                "MATCH (n {id: $id}) RETURN n.stable_key AS k, n.owner_type AS ot,
                        n.owner_trait AS otr",
            )
            .param("id", id),
        )
        .await
        .unwrap();
    let row = r.next().await.unwrap().unwrap();
    (
        row.get::<Option<String>>("k").unwrap(),
        row.get::<Option<String>>("ot").unwrap(),
        row.get::<Option<String>>("otr").unwrap(),
    )
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

    // The uncapped live listing (used by evolution, lifecycle, the hook…)
    // sees exactly the non-archived skills, through the GraphStore trait.
    {
        use project_orchestrator::neo4j::traits::GraphStore;
        let store: &dyn GraphStore = &client;
        let mut live: Vec<String> = store
            .get_live_skills_for_project(q)
            .await
            .unwrap()
            .into_iter()
            .map(|s| s.id.to_string())
            .collect();
        live.sort();
        let mut expected = vec![rich.clone(), protocol_ref.clone(), solo.clone()];
        expected.sort();
        assert_eq!(live, expected);
    }

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
    // Through the trait, as deep maintenance calls it.
    let purged = {
        use project_orchestrator::neo4j::traits::GraphStore;
        let store: &dyn GraphStore = &client;
        store
            .purge_archived_empty_skills(q, chrono::Utc::now() - chrono::Duration::days(7))
            .await
            .unwrap()
    };
    assert_eq!(purged, 1);
    assert_eq!(status_of(&raw, &recent).await.as_deref(), Some("archived"));
    assert_eq!(status_of(&raw, &old).await, None);

    // ------------------------------------------------------------------
    // 5. Notes: restore the victims of the compounding energy decay, raise
    //    over-decayed active notes, never touch legitimate archivals.
    // ------------------------------------------------------------------
    let r = Uuid::new_v4();
    let rs = r.to_string();
    let note = |status: &'static str, energy: f64, idle_days: i64, changes: &'static str| {
        let raw = &raw;
        let rs = rs.clone();
        async move {
            let id = Uuid::new_v4().to_string();
            raw.run(
                query(
                    "CREATE (:Note {id: $id, project_id: $p, status: $status, energy: $energy,
                             note_type: 'gotcha', content: 'test', created_by: 'test',
                             created_at: datetime() - duration({days: $idle + 10}),
                             last_activated: datetime() - duration({days: $idle}),
                             changes_json: $changes})",
                )
                .param("id", id.clone())
                .param("p", rs)
                .param("status", status)
                .param("energy", energy)
                .param("idle", idle_days)
                .param("changes", changes),
            )
            .await
            .unwrap();
            id
        }
    };
    let victim = note(
        "archived",
        0.0,
        30,
        r#"[{"details":{"reason":"low_energy_60d"}}]"#,
    )
    .await;
    let ephemeral = note(
        "archived",
        0.0,
        30,
        r#"[{"details":{"reason":"ephemeral_expired"}}]"#,
    )
    .await;
    let replaced = note(
        "archived",
        0.0,
        30,
        r#"[{"details":{"reason":"low_energy_60d"}}]"#,
    )
    .await;
    raw.run(
        query("MATCH (old:Note {id: $id}) CREATE (:Note {id: randomUUID(), project_id: $p, status: 'active', note_type: 'gotcha', content: 'newer', created_by: 'test', created_at: datetime()})-[:SUPERSEDES]->(old)")
            .param("id", replaced.clone())
            .param("p", rs.clone()),
    )
    .await
    .unwrap();
    let crushed = note("active", 0.01, 30, "[]").await;
    let healthy = note("active", 0.95, 30, "[]").await;

    let energy_of = |id: String| {
        let raw = &raw;
        async move {
            let mut r = raw
                .execute(
                    query("MATCH (n:Note {id: $id}) RETURN n.status AS st, n.energy AS e")
                        .param("id", id),
                )
                .await
                .unwrap();
            let row = r.next().await.unwrap().unwrap();
            (
                row.get::<String>("st").unwrap(),
                row.get::<f64>("e").unwrap(),
            )
        }
    };

    let out = client
        .run_data_migration_scoped("2026-09-restore-energy-decay-victims", r)
        .await
        .unwrap();
    assert_eq!(out.processed, 1, "only the low_energy_60d victim");
    let (st, e) = energy_of(victim.clone()).await;
    assert_eq!(st, "active");
    let intended = (-30.0_f64 / 90.0).exp();
    assert!(
        (e - intended).abs() < 0.01,
        "restored at the intended energy, got {e}"
    );
    assert_eq!(
        energy_of(ephemeral.clone()).await.0,
        "archived",
        "legitimate archival stays"
    );
    assert_eq!(
        energy_of(replaced.clone()).await.0,
        "archived",
        "a superseded note is never restored: newer knowledge wins"
    );
    // Restored once: re-archiving it later must stick.
    raw.run(
        query("MATCH (n:Note {id: $id}) SET n.status = 'archived'").param("id", victim.clone()),
    )
    .await
    .unwrap();
    let again = client
        .run_data_migration_scoped("2026-09-restore-energy-decay-victims", r)
        .await
        .unwrap();
    assert_eq!(again.processed, 0);

    let out = client
        .run_data_migration_scoped("2026-09-rebase-active-note-energy", r)
        .await
        .unwrap();
    assert_eq!(
        out.processed, 3,
        "crushed, healthy and the superseding note"
    );
    assert!(
        (energy_of(crushed.clone()).await.1 - intended).abs() < 0.01,
        "raised"
    );
    assert!(
        (energy_of(healthy.clone()).await.1 - 0.95).abs() < 1e-9,
        "never lowered"
    );

    // Energy update is temporally idempotent: a second call right after the
    // first changes nothing (the old formula compounded on every call).
    raw.run(
        query(
            "MATCH (n:Note {id: $id})
             SET n.energy = 0.8, n.energy_updated_at = datetime() - duration({days: 9})",
        )
        .param("id", healthy.clone()),
    )
    .await
    .unwrap();
    client.update_energy_scores(90.0).await.unwrap();
    let after_first = energy_of(healthy.clone()).await.1;
    client.update_energy_scores(90.0).await.unwrap();
    let after_second = energy_of(healthy.clone()).await.1;
    assert!(
        (after_first - 0.8 * (-9.0_f64 / 90.0).exp()).abs() < 0.005,
        "got {after_first}"
    );
    assert!(
        (after_first - after_second).abs() < 1e-4,
        "{after_first} vs {after_second}"
    );

    // Consolidation: a note older than 90 days that the activation path
    // reactivated (reactivation_count) is not "never activated" — the old
    // rule only read activation_count, which nothing ever increments.
    let used = Uuid::new_v4().to_string();
    raw.run(
        query(
            "CREATE (:Note {id: $id, project_id: $p, status: 'active', energy: 0.9,
                     note_type: 'gotcha', importance: 'high', content: 'kept', tags: [],
                     created_by: 'test',
                     memory_horizon: 'consolidated', activation_count: 0, reactivation_count: 4,
                     created_at: datetime() - duration({days: 120}),
                     last_activated: datetime() - duration({days: 2}),
                     energy_updated_at: datetime()})",
        )
        .param("id", used.clone())
        .param("p", rs.clone()),
    )
    .await
    .unwrap();
    client.consolidate_memory().await.unwrap();
    assert_eq!(
        energy_of(used.clone()).await.0,
        "active",
        "reactivated note kept"
    );

    // Synapse decay is project-scoped: another project's synapses untouched.
    let other = Uuid::new_v4();
    raw.run(
        query(
            "CREATE (:Note {id: randomUUID(), project_id: $p})-[:SYNAPSE {weight: 0.5, source: 'coactivation'}]->(:Note {id: randomUUID(), project_id: $p})
             CREATE (:Note {id: randomUUID(), project_id: $o})-[:SYNAPSE {weight: 0.5, source: 'coactivation'}]->(:Note {id: randomUUID(), project_id: $o})",
        )
        .param("p", rs.clone())
        .param("o", other.to_string()),
    )
    .await
    .unwrap();
    let (decayed, _) = client.decay_project_synapses(r, 0.1, 0.05).await.unwrap();
    assert_eq!(decayed, 1);
    let mut w = raw
        .execute(
            query("MATCH (a:Note {project_id: $o})-[s:SYNAPSE]->() RETURN s.weight AS w")
                .param("o", other.to_string()),
        )
        .await
        .unwrap();
    assert_eq!(
        w.next().await.unwrap().unwrap().get::<f64>("w").unwrap(),
        0.5
    );

    // ------------------------------------------------------------------
    // 5b. Dormant project (no sync nor chat for weeks): its knowledge is
    //     frozen — coming back months later must find it intact.
    // ------------------------------------------------------------------
    let d = Uuid::new_v4();
    let ds = d.to_string();
    let frozen = Uuid::new_v4().to_string();
    let permanent = Uuid::new_v4().to_string();
    raw.run(
        query(
            "CREATE (:Project {id: $d, slug: 'dormant-' + $d, name: 'dormant', root_path: '/tmp/dormant',
                     last_synced: datetime() - duration({days: 120})})
             CREATE (f:Note {id: $frozen, project_id: $d, status: 'active', energy: 0.8,
                     note_type: 'pattern', content: 'frozen', created_by: 'test',
                     created_at: datetime() - duration({days: 200}),
                     last_activated: datetime() - duration({days: 120}),
                     energy_updated_at: datetime() - duration({days: 30})})
             CREATE (f)-[:SYNAPSE {weight: 0.5, source: 'coactivation'}]->(:Note {id: randomUUID(), project_id: $d})
             CREATE (:Note {id: $permanent, project_id: $d, status: 'active', energy: 0.0,
                     note_type: 'gotcha', content: 'permanent', created_by: 'test',
                     memory_horizon: 'consolidated', activation_count: 0, reactivation_count: 0,
                     created_at: datetime() - duration({days: 400}),
                     last_activated: datetime() - duration({days: 300})})",
        )
        .param("d", ds.clone())
        .param("frozen", frozen.clone())
        .param("permanent", permanent.clone()),
    )
    .await
    .unwrap();
    assert!(client.dormant_project_ids().await.unwrap().contains(&ds));

    client.update_energy_scores(90.0).await.unwrap();
    assert!(
        (energy_of(frozen.clone()).await.1 - 0.8).abs() < 1e-9,
        "dormant project: energy frozen"
    );
    client.decay_synapses(0.1, 0.05).await.unwrap();
    assert_eq!(
        client.decay_project_synapses(d, 0.1, 0.05).await.unwrap(),
        (0, 0)
    );
    let mut w = raw
        .execute(
            query("MATCH (a:Note {id: $id})-[s:SYNAPSE]->() RETURN s.weight AS w")
                .param("id", frozen.clone()),
        )
        .await
        .unwrap();
    assert_eq!(
        w.next().await.unwrap().unwrap().get::<f64>("w").unwrap(),
        0.5,
        "dormant project: synapses frozen"
    );
    client.consolidate_memory().await.unwrap();
    assert_eq!(
        energy_of(permanent.clone()).await.0,
        "active",
        "consolidated knowledge is never auto-archived"
    );

    // Activity resumes (a code sync): decay applies again, from now on.
    raw.run(
        query("MATCH (p:Project {id: $d}) SET p.last_synced = datetime()").param("d", ds.clone()),
    )
    .await
    .unwrap();
    assert!(!client.dormant_project_ids().await.unwrap().contains(&ds));

    // ------------------------------------------------------------------
    // 5c. Newer knowledge wins: replaced / archived knowledge never comes
    //     back through personas, skills, or a later confirmation.
    // ------------------------------------------------------------------
    {
        use project_orchestrator::neo4j::models::{PersonaNode, PersonaOrigin, PersonaStatus};
        use project_orchestrator::neo4j::traits::GraphStore;
        let store: &dyn GraphStore = &client;
        let k = Uuid::new_v4();
        let ks = k.to_string();
        let current = Uuid::new_v4();
        let old = Uuid::new_v4();
        raw.run(
            query(
                "CREATE (:Note {id: $cur, project_id: $k, status: 'active', energy: 0.9,
                         note_type: 'guideline', content: 'use the NEW way', created_by: 'test',
                         created_at: datetime()})
                 CREATE (:Note {id: $old, project_id: $k, status: 'archived', energy: 1.0,
                         note_type: 'guideline', content: 'use the OLD way', created_by: 'test',
                         superseded_by: $cur, created_at: datetime() - duration({days: 30})})
                 WITH 1 AS _
                 MATCH (n:Note {id: $cur}), (o:Note {id: $old})
                 CREATE (n)-[:SUPERSEDES]->(o)",
            )
            .param("cur", current.to_string())
            .param("old", old.to_string())
            .param("k", ks.clone()),
        )
        .await
        .unwrap();

        // A later confirmation of the old note must not resurrect it.
        client.confirm_note(old, "test").await.unwrap();
        assert_eq!(energy_of(old.to_string()).await.0, "archived");

        // Persona: only current knowledge in its subgraph (hook context).
        let persona = PersonaNode {
            id: Uuid::new_v4(),
            project_id: Some(k),
            name: "freshness".into(),
            description: "test".into(),
            status: PersonaStatus::Active,
            complexity_default: None,
            timeout_secs: None,
            max_cost_usd: None,
            model_preference: None,
            system_prompt_override: None,
            energy: 0.9,
            cohesion: 0.0,
            activation_count: 0,
            success_rate: 0.0,
            avg_duration_secs: 0.0,
            last_activated: None,
            energy_boost_accumulated: 0.0,
            energy_history: vec![],
            origin: PersonaOrigin::Manual,
            created_at: chrono::Utc::now(),
            updated_at: None,
        };
        store.create_persona(&persona).await.unwrap();
        store
            .add_persona_note(persona.id, current, 0.5)
            .await
            .unwrap();
        store.add_persona_note(persona.id, old, 1.0).await.unwrap();
        let sub = store.get_persona_subgraph(persona.id).await.unwrap();
        let ids: Vec<String> = sub.notes.iter().map(|n| n.entity_id.clone()).collect();
        assert_eq!(
            ids,
            vec![current.to_string()],
            "persona injects only current knowledge"
        );

        // Skill: activation returns only current members.
        let sk = skill(&raw, &ks, "Freshness", "active", 0).await;
        raw.run(
            query(
                "MATCH (s:Skill {id: $s}), (a:Note {id: $cur}), (b:Note {id: $old})
                 CREATE (a)-[:MEMBER_OF]->(s), (b)-[:MEMBER_OF]->(s)",
            )
            .param("s", sk.clone())
            .param("cur", current.to_string())
            .param("old", old.to_string())
            .param("k", ks.clone()),
        )
        .await
        .unwrap();
        let activated = store
            .activate_skill(Uuid::parse_str(&sk).unwrap(), "which way?")
            .await
            .unwrap();
        let ids: Vec<String> = activated
            .activated_notes
            .iter()
            .map(|n| n.note.id.to_string())
            .collect();
        assert!(
            ids.contains(&current.to_string()),
            "the newer knowledge is in the skill context"
        );
        assert!(
            !ids.contains(&old.to_string()),
            "the replaced knowledge is not"
        );
    }

    // Negative feedback weakens only the target's synapses, and homeostasis
    // decays only the project it corrects (both decayed the whole graph).
    {
        let x = Uuid::new_v4();
        let h = Uuid::new_v4();
        let o = Uuid::new_v4();
        raw.run(
            query(
                "CREATE (:Note {id: $x, project_id: $h})-[:SYNAPSE {weight: 0.5, source: 'coactivation'}]->(:Note {id: randomUUID(), project_id: $h})
                 CREATE (:Note {id: randomUUID(), project_id: $o})-[:SYNAPSE {weight: 0.5, source: 'coactivation'}]->(:Note {id: randomUUID(), project_id: $o})
                 CREATE (:Project {id: $h, slug: 'h-' + $h, name: 'h', root_path: '/tmp/h', last_synced: datetime()})
                 CREATE (:Project {id: $o, slug: 'o-' + $o, name: 'o', root_path: '/tmp/o', last_synced: datetime()})",
            )
            .param("x", x.to_string())
            .param("h", h.to_string())
            .param("o", o.to_string()),
        )
        .await
        .unwrap();
        let weight_in = |p: Uuid| {
            let raw = &raw;
            async move {
                let mut r = raw
                    .execute(
                        query(
                            "MATCH (a:Note {project_id: $p})-[s:SYNAPSE]->() RETURN s.weight AS w",
                        )
                        .param("p", p.to_string()),
                    )
                    .await
                    .unwrap();
                r.next()
                    .await
                    .unwrap()
                    .map(|row| row.get::<f64>("w").unwrap())
            }
        };
        assert_eq!(client.weaken_node_synapses(x, 0.1, 0.05).await.unwrap(), 1);
        assert!((weight_in(h).await.unwrap() - 0.4).abs() < 1e-9);
        assert_eq!(weight_in(o).await, Some(0.5), "other nodes untouched");

        use project_orchestrator::homeostasis::{
            execute_actions, ExecuteContext, HomeostasisAction,
        };
        use project_orchestrator::neo4j::traits::GraphStore;
        let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into());
        let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
        let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into());
        let store: std::sync::Arc<dyn GraphStore> =
            std::sync::Arc::new(Neo4jClient::new(&uri, &user, &password).await.unwrap());
        execute_actions(
            &store,
            None,
            &[HomeostasisAction::DecaySynapses {
                amount: 0.1,
                prune_threshold: 0.05,
            }],
            ExecuteContext {
                project_id: Some(h),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(
            (weight_in(h).await.unwrap() - 0.3).abs() < 1e-9,
            "corrected project decayed"
        );
        assert_eq!(weight_in(o).await, Some(0.5), "other projects untouched");
    }

    // Skill detection clusters current knowledge only: no archived note,
    // no skill-evolution audit trail.
    {
        use project_orchestrator::neo4j::traits::GraphStore;
        let store: &dyn GraphStore = &client;
        let g = Uuid::new_v4();
        let gs = g.to_string();
        raw.run(
            query(
                "CREATE (a:Note {id: 'g-a-' + $g, project_id: $g, status: 'active', created_by: 'user'})
                 CREATE (b:Note {id: 'g-b-' + $g, project_id: $g, status: 'active', created_by: 'user'})
                 CREATE (arch:Note {id: 'g-arch-' + $g, project_id: $g, status: 'archived', created_by: 'user'})
                 CREATE (tr:Note {id: 'g-trace-' + $g, project_id: $g, status: 'active', created_by: 'skill-evolution'})
                 CREATE (a)-[:SYNAPSE {weight: 0.8}]->(b)
                 CREATE (a)-[:SYNAPSE {weight: 0.8}]->(arch)
                 CREATE (a)-[:SYNAPSE {weight: 0.8}]->(tr)",
            )
            .param("g", gs.clone()),
        )
        .await
        .unwrap();
        let edges = store.get_synapse_graph(g, 0.1).await.unwrap();
        assert_eq!(edges.len(), 1, "{edges:?}");
        assert_eq!(edges[0].1, format!("g-b-{gs}"));
    }

    // Deep maintenance times are persisted per project (a restart must not
    // re-run a full pass over every project).
    {
        use project_orchestrator::neo4j::traits::GraphStore;
        let store: &dyn GraphStore = &client;
        store.mark_deep_maintenance(d).await.unwrap();
        let times = store.get_deep_maintenance_times().await.unwrap();
        let (_, at) = times.iter().find(|(id, _)| *id == d).expect("persisted");
        assert!((chrono::Utc::now() - *at).num_seconds().abs() < 60);
    }

    // ------------------------------------------------------------------
    // 5b. The stable identity of code symbols (T1, `symbol_key.rs`): the
    //     four backfills key EVERY legacy node of the six labels, with the
    //     grammar the write paths use — same owner, same key, no id touched.
    // ------------------------------------------------------------------
    {
        use project_orchestrator::neo4j::models::{
            FunctionNode, ImplNode, ImportNode, Parameter, StructNode, SymbolOwner, Visibility,
        };
        use project_orchestrator::neo4j::symbol_key::{
            stable_external_trait_key, stable_impl_key, stable_import_key, stable_symbol_key,
            stable_type_key,
        };

        let k = Uuid::new_v4();
        let ks = k.to_string();
        let rs = format!("{ks}/src/a.rs");
        let py = format!("{ks}/src/k.py");
        let fmt_params =
            r#"[{"name":"self","type_name":"&self"},{"name":"f","type_name":"&mut Formatter"}]"#;
        // Legacy nodes, as the server wrote them before this version: an id
        // with the line, `params` as JSON, `''` for "no trait", no stable_key —
        // and NO project_id on Impl and Import (0 of 59 994 and 0 of 289 491
        // carry one on 31/10/2026): the scoped run must reach them through
        // their File.
        raw.run(
            query(
                "CREATE (f:File {path: $rs, project_id: $p, language: 'rust'})
                 CREATE (g:File {path: $py, project_id: $p, language: 'python'})
                 CREATE (f)-[:CONTAINS]->(:Function {id: $rs + ':new:1', name: 'new', file_path: $rs,
                         line_start: 1, line_end: 3, params: '[]', project_id: $p})
                 CREATE (f)-[:CONTAINS]->(:Struct {id: $rs + ':A', name: 'A', file_path: $rs,
                         line_start: 5, line_end: 6, project_id: $p})
                 CREATE (f)-[:CONTAINS]->(:Trait {id: $rs + ':T', name: 'T', file_path: $rs,
                         line_start: 7, line_end: 8, project_id: $p})
                 CREATE (f)-[:CONTAINS]->(:Enum {id: $rs + ':E', name: 'E', file_path: $rs,
                         line_start: 9, line_end: 9, project_id: $p})
                 CREATE (:Trait {id: 'external:trait:std::fmt::Display', name: 'Display',
                         full_path: 'std::fmt::Display', is_external: true, file_path: '',
                         line_start: 0, line_end: 0, project_id: $p})
                 CREATE (f)-[:CONTAINS]->(:Impl {id: $rs + ':impl:A:self', for_type: 'A', trait_name: '',
                         file_path: $rs, line_start: 10, line_end: 20})
                 CREATE (f)-[:CONTAINS]->(:Function {id: $rs + ':new:11', name: 'new', file_path: $rs,
                         line_start: 11, line_end: 13, params: '[]', project_id: $p})
                 CREATE (f)-[:CONTAINS]->(:Impl {id: $rs + ':impl:B:self', for_type: 'B', trait_name: null,
                         file_path: $rs, line_start: 30, line_end: 40})
                 CREATE (f)-[:CONTAINS]->(:Function {id: $rs + ':new:31', name: 'new', file_path: $rs,
                         line_start: 31, line_end: 33, params: '[]', project_id: $p})
                 CREATE (f)-[:CONTAINS]->(:Impl {id: $rs + ':impl:A:Display', for_type: 'A', trait_name: 'Display',
                         file_path: $rs, line_start: 50, line_end: 60})
                 CREATE (f)-[:CONTAINS]->(:Function {id: $rs + ':fmt:51', name: 'fmt', file_path: $rs,
                         line_start: 51, line_end: 55, params: $fmt, project_id: $p})
                 CREATE (f)-[:CONTAINS]->(:Impl {id: $rs + ':impl:A:Debug', for_type: 'A', trait_name: 'Debug',
                         file_path: $rs, line_start: 70, line_end: 80})
                 CREATE (f)-[:CONTAINS]->(:Function {id: $rs + ':fmt:71', name: 'fmt', file_path: $rs,
                         line_start: 71, line_end: 75, params: $fmt, project_id: $p})
                 CREATE (f)-[:CONTAINS]->(:Impl {id: $rs + ':impl:C:  ', for_type: 'C', trait_name: '  ',
                         file_path: $rs, line_start: 82, line_end: 84})
                 CREATE (f)-[:CONTAINS]->(:Function {id: $rs + ':parse:90', name: 'parse', file_path: $rs,
                         line_start: 90, line_end: 92, params: '[{\"name\":\"x\",\"type_name\":\"i32\"}]', project_id: $p})
                 CREATE (f)-[:CONTAINS]->(:Function {id: $rs + ':parse:95', name: 'parse', file_path: $rs,
                         line_start: 95, line_end: 97, params: '[{\"name\":\"x\",\"type_name\":\"bool\"}]', project_id: $p})
                 CREATE (f)-[:CONTAINS]->(:Function {id: $rs + ':nameless:100', file_path: $rs,
                         line_start: 100, line_end: 101, params: '[]', project_id: $p})
                 CREATE (f)-[:HAS_IMPORT]->(:Import {id: $rs + ':3:std::fmt', path: 'std::fmt', file_path: $rs,
                         line: 3})
                 CREATE (f)-[:HAS_IMPORT]->(:Import {id: $rs + ':99:std::fmt', path: 'std::fmt', file_path: $rs,
                         line: 99})
                 CREATE (f)-[:HAS_IMPORT]->(:Import {id: $rs + ':4:std::io', path: 'std::io', file_path: $rs,
                         line: 4})
                 CREATE (g)-[:CONTAINS]->(:Struct {id: $py + ':Klass', name: 'Klass', file_path: $py,
                         line_start: 1, line_end: 20, project_id: $p})
                 CREATE (g)-[:CONTAINS]->(:Function {id: $py + ':__init__:2', name: '__init__', file_path: $py,
                         line_start: 2, line_end: 4, params: '[{\"name\":\"self\",\"type_name\":null}]', project_id: $p})",
            )
            .param("p", ks.clone())
            .param("rs", rs.clone())
            .param("py", py.clone())
            .param("fmt", fmt_params),
        )
        .await
        .unwrap();

        let before = ids_in_scope(&raw, &ks).await;

        for (id, expected) in [
            ("2026-10-backfill-stable-key", 8),
            ("2026-10-backfill-stable-key-types", 5),
            ("2026-10-backfill-stable-key-impls", 5),
            ("2026-10-backfill-stable-key-imports", 3),
        ] {
            let out = client.run_data_migration_scoped(id, k).await.unwrap();
            assert!(out.completed, "{out:?}");
            assert_eq!(out.processed, expected, "{id}: {out:?}");
            // Idempotent AND convergent: a second run finds nothing — including
            // for the function without a name, which the grammar cannot key and
            // the SELECT therefore never returns (finding 3 of the review).
            let again = client.run_data_migration_scoped(id, k).await.unwrap();
            assert!(again.completed && again.processed == 0, "{id}: {again:?}");
        }

        // Every keyable node of the six labels carries a key; the nameless one
        // is the only exception, and it did not stall the migration.
        assert_eq!(
            count(
                &raw,
                "MATCH (n) WHERE NOT n:File AND n.stable_key IS NULL AND (n.project_id = $p
                       OR EXISTS { (n)<-[:CONTAINS|HAS_IMPORT]-(:File {project_id: $p}) })
                 RETURN count(n) AS c",
                &ks
            )
            .await,
            1
        );
        assert_eq!(
            key_of(&raw, &format!("{rs}:nameless:100")).await.0,
            None,
            "a node the grammar cannot key is left alone, not keyed with NULL"
        );

        // Finding 1: the owner qualifies the method. `impl A { fn new }` and
        // `impl B { fn new }` and the free `new` are three symbols; the backfill
        // deduced each owner by line inclusion and wrote it on the node.
        let free = key_of(&raw, &format!("{rs}:new:1")).await;
        let a_new = key_of(&raw, &format!("{rs}:new:11")).await;
        let b_new = key_of(&raw, &format!("{rs}:new:31")).await;
        assert_eq!(free, (Some(format!("{rs}::new")), None, None));
        assert_eq!(
            a_new,
            (Some(format!("{rs}::A::new")), Some("A".into()), None)
        );
        assert_eq!(
            b_new,
            (Some(format!("{rs}::B::new")), Some("B".into()), None)
        );
        // Two trait impls of one type: `fmt` of Display is not `fmt` of Debug.
        let display = key_of(&raw, &format!("{rs}:fmt:51")).await;
        let debug = key_of(&raw, &format!("{rs}:fmt:71")).await;
        assert_eq!(
            display.0.as_deref(),
            Some(format!("{rs}::<A as Display>::fmt(&self,&mut Formatter)").as_str())
        );
        assert_eq!(
            debug.0.as_deref(),
            Some(format!("{rs}::<A as Debug>::fmt(&self,&mut Formatter)").as_str())
        );
        assert_eq!(display.2.as_deref(), Some("Display"));
        // Overloads keep distinct keys; a class method takes its class as owner.
        assert_ne!(
            key_of(&raw, &format!("{rs}:parse:90")).await.0,
            key_of(&raw, &format!("{rs}:parse:95")).await.0
        );
        assert_eq!(
            key_of(&raw, &format!("{py}:__init__:2")).await.0.as_deref(),
            Some(format!("{py}::Klass::__init__(?)").as_str())
        );

        // Finding 5: ONE grammar. The backfilled key is byte-identical to what
        // `symbol_key.rs` computes — and to what the write path then writes on
        // the SAME node (MERGE on the legacy id), for every label.
        let a_owner = SymbolOwner::new("A", None::<&str>);
        assert_eq!(
            a_new.0.as_deref(),
            Some(stable_symbol_key(&rs, Some(&a_owner), "new", &[]).as_str())
        );
        client
            .upsert_function(&FunctionNode {
                name: "new".into(),
                visibility: Visibility::Public,
                params: vec![],
                return_type: None,
                generics: vec![],
                is_async: false,
                is_unsafe: false,
                complexity: 1,
                file_path: rs.clone(),
                line_start: 11,
                line_end: 13,
                docstring: None,
                owner: Some(a_owner.clone()),
            })
            .await
            .unwrap();
        assert_eq!(
            key_of(&raw, &format!("{rs}:new:11")).await,
            a_new,
            "the write path and the backfill disagree on the key of one node"
        );
        client
            .upsert_function(&FunctionNode {
                name: "fmt".into(),
                visibility: Visibility::Public,
                params: vec![
                    Parameter {
                        name: "self".into(),
                        type_name: Some("&self".into()),
                    },
                    Parameter {
                        name: "f".into(),
                        type_name: Some("&mut Formatter".into()),
                    },
                ],
                return_type: None,
                generics: vec![],
                is_async: false,
                is_unsafe: false,
                complexity: 1,
                file_path: rs.clone(),
                line_start: 51,
                line_end: 55,
                docstring: None,
                owner: Some(SymbolOwner::new("A", Some("Display"))),
            })
            .await
            .unwrap();
        assert_eq!(key_of(&raw, &format!("{rs}:fmt:51")).await, display);

        // Types: the three declarations and the external placeholder.
        for (id, expected) in [
            (format!("{rs}:A"), stable_type_key(&rs, "A")),
            (format!("{rs}:T"), stable_type_key(&rs, "T")),
            (format!("{rs}:E"), stable_type_key(&rs, "E")),
            (format!("{py}:Klass"), stable_type_key(&py, "Klass")),
            (
                "external:trait:std::fmt::Display".to_string(),
                stable_external_trait_key("std::fmt::Display"),
            ),
        ] {
            assert_eq!(
                key_of(&raw, &id).await.0.as_deref(),
                Some(expected.as_str()),
                "{id}"
            );
        }
        // Finding 2: the unit `upsert_struct` writes the key too (it did not).
        client
            .upsert_struct(&StructNode {
                name: "A".into(),
                visibility: Visibility::Public,
                generics: vec![],
                file_path: rs.clone(),
                line_start: 5,
                line_end: 6,
                docstring: None,
                parent_class: None,
                interfaces: vec![],
            })
            .await
            .unwrap();
        assert_eq!(
            key_of(&raw, &format!("{rs}:A")).await.0,
            Some(stable_type_key(&rs, "A"))
        );

        // Impls: `''`, NULL and `'  '` are all "no trait" — the Cypher of the
        // previous version tested `= ''` only (finding 5).
        for (id, expected) in [
            (format!("{rs}:impl:A:self"), stable_impl_key(&rs, "A", None)),
            (format!("{rs}:impl:B:self"), stable_impl_key(&rs, "B", None)),
            (format!("{rs}:impl:C:  "), stable_impl_key(&rs, "C", None)),
            (
                format!("{rs}:impl:A:Display"),
                stable_impl_key(&rs, "A", Some("Display")),
            ),
        ] {
            assert_eq!(
                key_of(&raw, &id).await.0.as_deref(),
                Some(expected.as_str()),
                "{id}"
            );
        }
        assert_eq!(
            key_of(&raw, &format!("{rs}:impl:C:  ")).await.0,
            Some(format!("{rs}::impl C"))
        );
        // Finding 2 again: the unit `upsert_impl` writes the key, and (finding 4)
        // the project_id — Impl and Import nodes had none (0 of 59 994 and 0 of
        // 289 491 on 31/10/2026), so `$scope` could not reach them.
        client
            .upsert_impl(&ImplNode {
                for_type: "A".into(),
                trait_name: None,
                generics: vec![],
                where_clause: None,
                file_path: rs.clone(),
                line_start: 10,
                line_end: 20,
            })
            .await
            .unwrap();
        assert_eq!(
            key_of(&raw, &format!("{rs}:impl:A:self")).await.0,
            Some(stable_impl_key(&rs, "A", None))
        );

        // Imports (finding 9): the line left the key. The two legacy versions
        // of `use std::fmt` share one key — T2 folds them — and `std::io` is
        // another import.
        let fmt_key = stable_import_key(&rs, "std::fmt", 0);
        assert_eq!(fmt_key, format!("{rs}::import:std::fmt"));
        assert_eq!(
            key_of(&raw, &format!("{rs}:3:std::fmt")).await.0.as_deref(),
            Some(fmt_key.as_str())
        );
        assert_eq!(
            key_of(&raw, &format!("{rs}:99:std::fmt"))
                .await
                .0
                .as_deref(),
            Some(fmt_key.as_str())
        );
        assert_eq!(
            key_of(&raw, &format!("{rs}:4:std::io")).await.0,
            Some(stable_import_key(&rs, "std::io", 0))
        );
        client
            .upsert_import(&ImportNode {
                path: "std::fmt".into(),
                alias: None,
                items: vec![],
                file_path: rs.clone(),
                line: 3,
                occurrence: 0,
            })
            .await
            .unwrap();
        assert_eq!(
            key_of(&raw, &format!("{rs}:3:std::fmt")).await.0.as_deref(),
            Some(fmt_key.as_str())
        );
        // A genuine duplicate in the current file is ranked, not merged.
        assert_eq!(
            stable_import_key(&rs, "std::fmt", 1),
            format!("{rs}::import:std::fmt#2")
        );

        // Finding 4: every write path denormalises project_id from the File —
        // the three nodes the unit paths rewrote above now carry it (the legacy
        // Impl/Import the backfill keyed still do not: keying is not re-homing).
        assert_eq!(
            count(
                &raw,
                "MATCH (n) WHERE n.id IN [$p + '/src/a.rs:new:11', $p + '/src/a.rs:impl:A:self',
                                         $p + '/src/a.rs:3:std::fmt']
                   AND n.project_id = $p
                 RETURN count(n) AS c",
                &ks
            )
            .await,
            3
        );
        assert_eq!(
            count(
                &raw,
                "MATCH (:File {project_id: $p})-[:CONTAINS|HAS_IMPORT]->(n:Impl|Import)
                 WHERE n.project_id IS NULL RETURN count(n) AS c",
                &ks
            )
            .await,
            6,
            "the legacy Impl/Import not rewritten by a write path keep no project_id"
        );

        // No id was modified, none appeared: the key is a shadow property.
        assert_eq!(ids_in_scope(&raw, &ks).await, before);

        // Finding 7: the key is indexed on the six labels (created by the
        // client's schema init, which ran when `client` was built).
        let mut r = raw
            .execute(query(
                "SHOW INDEXES YIELD name, properties
                 WHERE 'stable_key' IN properties
                 RETURN collect(name) AS names",
            ))
            .await
            .unwrap();
        let mut names: Vec<String> = r.next().await.unwrap().unwrap().get("names").unwrap();
        names.sort();
        assert_eq!(
            names,
            vec![
                "enum_stable_key",
                "function_stable_key",
                "impl_stable_key",
                "import_stable_key",
                "struct_stable_key",
                "trait_stable_key",
            ]
        );
    }

    // ------------------------------------------------------------------
    // 6. The startup runner completes every migration once, then skips.
    // ------------------------------------------------------------------
    let first = {
        use project_orchestrator::neo4j::traits::GraphStore;
        let store: &dyn GraphStore = &client;
        store.run_data_migrations().await
    };
    // The runner must have run EVERY registered migration — compared to the
    // registry, not to a frozen number. Hard-coding it turned a new migration
    // into a CI failure that had nothing to do with the migration itself
    // (measured 29/10/2026: adding one made this line fail with 6 != 5).
    let registered = Neo4jClient::data_migrations().len();
    assert_eq!(first.len(), registered, "{first:?}");
    assert!(
        first.iter().all(|o| o.completed && o.error.is_none()),
        "{first:?}"
    );
    let second = client.run_data_migrations().await;
    assert!(second.iter().all(|o| o.skipped), "{second:?}");
}
