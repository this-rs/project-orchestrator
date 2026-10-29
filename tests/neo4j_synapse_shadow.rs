//! The Cypher of the synaptic layer in shadow mode (T8,
//! `src/neo4j/synapse_shadow.rs`) against a real Neo4j.
//!
//! One file per subject (see `harness_neo4j.rs` for the shared conventions):
//! every scenario uses fresh ids and cleans up after itself. NEVER point it at
//! a live database. Run locally against a throwaway instance:
//!   docker run -d --rm --name po-t8 -p 17687:7687 -e NEO4J_AUTH=neo4j/testpassword \
//!     -e NEO4J_PLUGINS='["apoc"]' neo4j:5.26-community
//!   NEO4J_URI=bolt://localhost:17687 NEO4J_PASSWORD=testpassword cargo test --test neo4j_synapse_shadow
//!
//! The test prints a notice and returns when no Neo4j is reachable (a socket
//! probe: a Cypher probe sits in neo4rs' retry loop for minutes);
//! `HARNESS_NEO4J_REQUIRED=1` (set in CI) turns that into a failure.
#[allow(unused_imports)]
use project_orchestrator::neo4j::traits::*;

use std::time::Duration;

use chrono::{DateTime, TimeZone, Utc};
use neo4rs::{query, Graph};
use project_orchestrator::neo4j::client::Neo4jClient;
use project_orchestrator::neurons::synaptic_shadow::{
    ShadowDecayScope, ShadowTraceFilter, StopReason, SynapseShadowTrace, SynapticLayerMode,
    TaskClosure, CLOSURE_KEY, DORMANT_WEIGHT, LAYER_VERSION, TRACE_SCHEMA_VERSION,
};
use project_orchestrator::notes::{Note, NoteType};
use uuid::Uuid;

struct Env {
    client: Neo4jClient,
    raw: Graph,
}

async fn bolt_reachable(uri: &str) -> bool {
    let authority = uri
        .split_once("://")
        .map_or(uri, |(_, rest)| rest)
        .split(['/', '?'])
        .next()
        .unwrap_or_default();
    tokio::time::timeout(
        Duration::from_secs(2),
        tokio::net::TcpStream::connect(authority),
    )
    .await
    .is_ok_and(|connected| connected.is_ok())
}

async fn env() -> Option<Env> {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:17687".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into());
    let connected = if bolt_reachable(&uri).await {
        async {
            let client = Neo4jClient::new(&uri, &user, &password).await.ok()?;
            let raw = Graph::new(&uri, &user, &password).await.ok()?;
            Some(Env { client, raw })
        }
        .await
    } else {
        None
    };
    if connected.is_none() {
        assert!(
            std::env::var("HARNESS_NEO4J_REQUIRED").is_err(),
            "HARNESS_NEO4J_REQUIRED is set but no Neo4j answers at {uri}"
        );
        eprintln!("Skipping synaptic shadow Neo4j tests: no Neo4j at {uri}");
    }
    connected
}

fn t0() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 10, 1, 12, 0, 0).unwrap()
}

async fn note(e: &Env, project: Uuid, content: &str) -> Uuid {
    let mut n = Note::new(
        Some(project),
        NoteType::Guideline,
        content.into(),
        "t8".into(),
    );
    n.created_at = t0() - chrono::Duration::days(10);
    e.client.create_note(&n).await.unwrap();
    n.id
}

async fn link(e: &Env, a: Uuid, b: Uuid, w: f64) {
    e.raw
        .run(
            query(
                "MATCH (a:Note {id: $a}), (b:Note {id: $b}) \
                 CREATE (a)-[:SYNAPSE {weight: $w, source: 'coactivation'}]->(b), \
                        (b)-[:SYNAPSE {weight: $w, source: 'coactivation'}]->(a)",
            )
            .param("a", a.to_string())
            .param("b", b.to_string())
            .param("w", w),
        )
        .await
        .unwrap();
}

/// `(weight, shadow_weight)` of the edge a → b.
async fn weights(e: &Env, a: Uuid, b: Uuid) -> (f64, Option<f64>) {
    let mut r = e
        .raw
        .execute(
            query(
                "MATCH (:Note {id: $a})-[s:SYNAPSE]->(:Note {id: $b}) \
                 RETURN s.weight AS w, s.shadow_weight AS sw",
            )
            .param("a", a.to_string())
            .param("b", b.to_string()),
        )
        .await
        .unwrap();
    let row = r.next().await.unwrap().expect("edge exists");
    (row.get("w").unwrap(), row.get::<f64>("sw").ok())
}

async fn edge_count(e: &Env, ids: &[Uuid]) -> i64 {
    let ids: Vec<String> = ids.iter().map(Uuid::to_string).collect();
    let mut r = e
        .raw
        .execute(
            query(
                "MATCH (a:Note)-[s:SYNAPSE]->(b:Note) WHERE a.id IN $ids AND b.id IN $ids \
                 RETURN count(s) AS n",
            )
            .param("ids", ids),
        )
        .await
        .unwrap();
    r.next().await.unwrap().unwrap().get("n").unwrap()
}

async fn cleanup(e: &Env, ids: &[Uuid], labels_ids: &[String]) {
    let ids: Vec<String> = ids
        .iter()
        .map(Uuid::to_string)
        .chain(labels_ids.iter().cloned())
        .collect();
    e.raw
        .run(query("MATCH (n) WHERE n.id IN $ids DETACH DELETE n").param("ids", ids))
        .await
        .unwrap();
}

#[tokio::test]
async fn shadow_edges_rewards_and_decay_never_touch_the_legacy_weight() {
    let Some(e) = env().await else { return };
    let p = Uuid::new_v4();
    let q = Uuid::new_v4();
    let frozen = Uuid::new_v4();
    let a = note(&e, p, "aaaa aaaa").await;
    let b = note(&e, p, "bbbb").await;
    let c = note(&e, p, "cccc").await;
    let other = note(&e, q, "qqqq").await;
    let f1 = note(&e, frozen, "ffff").await;
    let f2 = note(&e, frozen, "gggg").await;
    link(&e, a, b, 0.8).await;
    link(&e, b, c, 0.6).await;
    link(&e, b, other, 0.9).await;
    link(&e, f1, f2, 0.7).await;
    // A Project node without activity: dormant, its knowledge is frozen.
    e.raw
        .run(
            query("CREATE (:Project {id: $id, slug: $slug, name: 't8 frozen'})")
                .param("id", frozen.to_string())
                .param("slug", format!("t8-frozen-{frozen}")),
        )
        .await
        .unwrap();
    let all = [a, b, c, other, f1, f2];
    let edges_before = edge_count(&e, &all).await;

    // Reading: the project only.
    let rows = e.client.shadow_synapse_edges(p, &[a, b]).await.unwrap();
    let mut seen: Vec<(Uuid, Uuid, f64)> = rows.iter().map(|r| (r.from, r.to, r.weight)).collect();
    seen.sort_by_key(|x| (x.0, x.1));
    let mut expected = vec![(a, b, 0.8), (b, a, 0.8), (b, c, 0.6)];
    expected.sort_by_key(|x| (x.0, x.1));
    assert_eq!(seen, expected);
    let to_a = rows.iter().find(|r| r.to == a).unwrap();
    assert!(to_a.target.reachable);
    assert_eq!(to_a.target.tokens, 3); // 9 characters
    assert_eq!(
        to_a.target.created_at,
        Some(t0() - chrono::Duration::days(10))
    );

    // Reward: existing edges of the project only, both directions.
    let n = e
        .client
        .reward_shadow_synapses(p, &[(a, b), (b, a), (a, c), (b, other)], 0.1, t0())
        .await
        .unwrap();
    assert_eq!(n, 2);
    let (w, sw) = weights(&e, a, b).await;
    assert_eq!(w, 0.8);
    assert!((sw.unwrap() - 0.9).abs() < 1e-12);
    assert_eq!(weights(&e, b, other).await.1, None);
    assert_eq!(
        edge_count(&e, &all).await,
        edges_before,
        "an edge was created"
    );

    // Decay: the first pass starts the clock, then by elapsed time.
    let scope = ShadowDecayScope {
        default_enabled: false,
        enabled: vec![p, frozen],
        disabled: vec![],
    };
    let h = project_orchestrator::neurons::synaptic_shadow::shadow_half_life_days();
    e.client
        .decay_shadow_synapses(&scope, h, t0())
        .await
        .unwrap();
    assert_eq!(weights(&e, b, c).await, (0.6, Some(0.6)));
    e.client
        .decay_shadow_synapses(&scope, h, t0() + chrono::Duration::days(4))
        .await
        .unwrap();
    e.client
        .decay_shadow_synapses(&scope, h, t0() + chrono::Duration::days(11))
        .await
        .unwrap();
    let (w, sw) = weights(&e, a, b).await;
    assert_eq!(w, 0.8, "legacy weight written");
    assert!((sw.unwrap() - 0.9 * DORMANT_WEIGHT).abs() < 1e-6, "{sw:?}");
    let (_, sw) = weights(&e, b, c).await;
    assert!((sw.unwrap() - 0.6 * DORMANT_WEIGHT).abs() < 1e-6, "{sw:?}");
    // Dormant project: frozen weight.
    assert_eq!(weights(&e, f1, f2).await, (0.7, Some(0.7)));
    // Not in scope: never touched.
    assert_eq!(weights(&e, b, other).await, (0.9, None));
    assert_eq!(
        edge_count(&e, &all).await,
        edges_before,
        "an edge was deleted"
    );

    cleanup(&e, &all, &[frozen.to_string()]).await;
}

fn trace(session: Uuid, project: Uuid, notes: &[Uuid], at: DateTime<Utc>) -> SynapseShadowTrace {
    use project_orchestrator::neurons::synaptic_shadow::{ShadowSeed, ShadowSelection};
    SynapseShadowTrace {
        id: Uuid::new_v4(),
        schema_version: TRACE_SCHEMA_VERSION,
        layer_version: LAYER_VERSION.to_string(),
        session_id: session,
        project_id: project,
        requested_mode: SynapticLayerMode::On,
        effective_mode: SynapticLayerMode::Shadow,
        session_started_at: at,
        created_at: at,
        seeds: vec![ShadowSeed {
            note_id: notes[0],
            affinity: 1.0,
        }],
        selected: notes
            .iter()
            .enumerate()
            .map(|(i, n)| ShadowSelection {
                note_id: *n,
                score: 1.0 / (i as f64 + 1.0),
                hops: i,
                via: (i > 0).then(|| notes[i - 1]),
                tokens: 10,
                explored: false,
                created_at: Some(at - chrono::Duration::days(1)),
            })
            .collect(),
        tokens_used: 10 * notes.len(),
        token_budget: 1200,
        stopped_by: StopReason::BelowThreshold,
        weights_fingerprint: "f".repeat(64),
        config_fingerprint: "c".repeat(64),
        edges_read: 3,
        rewarded: vec![],
    }
}

#[tokio::test]
async fn traces_round_trip_are_bounded_per_session_and_claimed_once() {
    let Some(e) = env().await else { return };
    let session = Uuid::new_v4();
    let project = Uuid::new_v4();
    let (n1, n2) = (Uuid::new_v4(), Uuid::new_v4());
    let traces: Vec<SynapseShadowTrace> = (0..3)
        .map(|i| {
            trace(
                session,
                project,
                &[n1, n2],
                t0() + chrono::Duration::minutes(i),
            )
        })
        .collect();
    for t in &traces {
        e.client.record_synapse_shadow_trace(t, 2).await.unwrap();
    }
    let kept = e
        .client
        .list_synapse_shadow_traces(&ShadowTraceFilter::Session(session))
        .await
        .unwrap();
    assert_eq!(
        kept,
        traces[1..].to_vec(),
        "payload round trip, oldest dropped"
    );
    let containing = e
        .client
        .list_synapse_shadow_traces(&ShadowTraceFilter::ContainingNote {
            note_id: n2,
            limit: 1,
        })
        .await
        .unwrap();
    assert_eq!(containing, vec![traces[2].clone()]);

    let id = traces[2].id;
    assert!(e
        .client
        .claim_shadow_trace_reward(id, "verification")
        .await
        .unwrap());
    assert!(!e
        .client
        .claim_shadow_trace_reward(id, "verification")
        .await
        .unwrap());
    assert!(e
        .client
        .claim_shadow_trace_reward(id, CLOSURE_KEY)
        .await
        .unwrap());
    assert!(!e
        .client
        .claim_shadow_trace_reward(Uuid::new_v4(), "x")
        .await
        .unwrap());
    let back = e
        .client
        .get_synapse_shadow_trace(id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        back.rewarded,
        vec!["verification".to_string(), CLOSURE_KEY.to_string()]
    );

    // Two concurrent claims of the same key: exactly one wins.
    let fresh = traces[1].id;
    let (x, y) = tokio::join!(
        e.client.claim_shadow_trace_reward(fresh, "race"),
        e.client.claim_shadow_trace_reward(fresh, "race")
    );
    assert!(x.unwrap() ^ y.unwrap());

    let ids: Vec<String> = traces.iter().map(|t| t.id.to_string()).collect();
    cleanup(&e, &[], &ids).await;
}

async fn execution(
    e: &Env,
    session: Uuid,
    task: Uuid,
    started: DateTime<Utc>,
    completed: Option<DateTime<Utc>>,
) -> String {
    let id = Uuid::new_v4().to_string();
    e.raw
        .run(
            query(
                "CREATE (:AgentExecution {id: $id, task_id: $task, session_id: $session, \
                 started_at: datetime($started), \
                 completed_at: CASE WHEN $completed IS NULL THEN null ELSE datetime($completed) END})",
            )
            .param("id", id.clone())
            .param("task", task.to_string())
            .param("session", session.to_string())
            .param("started", started.to_rfc3339())
            .param("completed", completed.map(|c| c.to_rfc3339())),
        )
        .await
        .unwrap();
    id
}

async fn task(e: &Env, status: &str) -> Uuid {
    let id = Uuid::new_v4();
    e.raw
        .run(
            query("CREATE (:Task {id: $id, status: $status, title: 't8'})")
                .param("id", id.to_string())
                .param("status", status),
        )
        .await
        .unwrap();
    id
}

#[tokio::test]
async fn closures_are_quiet_restarted_or_left_open() {
    let Some(e) = env().await else { return };
    let project = Uuid::new_v4();
    let notes = [Uuid::new_v4(), Uuid::new_v4()];
    let closed = t0() + chrono::Duration::hours(1);
    let cutoff = closed + chrono::Duration::hours(1);
    let mut created: Vec<String> = Vec::new();

    let [quiet_s, restarted_s, open_s, recent_s] = [(); 4].map(|_| Uuid::new_v4());

    let t_quiet = task(&e, "completed").await;
    let t_restarted = task(&e, "completed").await;
    let t_open = task(&e, "in_progress").await;
    let t_recent = task(&e, "completed").await;
    created.push(execution(&e, quiet_s, t_quiet, t0(), Some(closed)).await);
    created.push(execution(&e, restarted_s, t_restarted, t0(), Some(closed)).await);
    created.push(execution(&e, Uuid::new_v4(), t_restarted, closed, None).await);
    created.push(execution(&e, open_s, t_open, t0(), Some(closed)).await);
    created.push(
        execution(
            &e,
            recent_s,
            t_recent,
            t0(),
            Some(cutoff + chrono::Duration::hours(1)),
        )
        .await,
    );

    let mut traces = Vec::new();
    for s in [quiet_s, restarted_s, open_s, recent_s] {
        let t = trace(s, project, &notes, t0());
        e.client.record_synapse_shadow_trace(&t, 50).await.unwrap();
        traces.push(t);
    }
    let mut got = e.client.shadow_trace_closures(cutoff, 1000).await.unwrap();
    got.retain(|(id, _)| traces.iter().any(|t| t.id == *id));
    got.sort_by_key(|(id, _)| *id);
    let mut expected = vec![
        (traces[0].id, TaskClosure::Quiet),
        (traces[1].id, TaskClosure::Restarted),
    ];
    expected.sort_by_key(|(id, _)| *id);
    assert_eq!(got, expected);

    // Judged once.
    e.client
        .claim_shadow_trace_reward(traces[0].id, CLOSURE_KEY)
        .await
        .unwrap();
    let again = e.client.shadow_trace_closures(cutoff, 1000).await.unwrap();
    assert!(again.iter().all(|(id, _)| *id != traces[0].id));

    created.extend(traces.iter().map(|t| t.id.to_string()));
    cleanup(&e, &[t_quiet, t_restarted, t_open, t_recent], &created).await;
}
