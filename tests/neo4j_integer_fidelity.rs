//! Integers through the Bolt driver, against a real Neo4j: what the server
//! holds is what the backend reads, and what the backend sends is what the
//! server stores.
//!
//! Why this exists: `neo4rs 0.8.0` decoded the one-byte negative integers
//! (Bolt TINY_INT, -16..=-1) without their sign, so a stored `-1` was read
//! `255` (see `vendor/neo4rs/PATCH.md`). `-1` is the backend's sentinel for "not
//! computed" / "no timeout": the bug turned "no timeout" into "255 seconds",
//! and a read-then-write of the same field stored that 255 for good.
//!
//! The first half pins the driver (decode and encode judged SEPARATELY: the
//! server converts with `toInteger` / `toString`, so a wrong decode cannot hide
//! a wrong encode). The second half pins what the backend does with such values:
//! the timeout of a protocol state, the repair of the 255 already written, and
//! the readers of counts (a negative is zero, never `usize::MAX`).
//!
//! One file per subject (see `harness_neo4j.rs` for the shared conventions):
//! every scenario uses fresh ids and removes what it created, panic or not
//! (`common::harness::Cleanup`). The target is checked by
//! `common::harness::neo4j_target`: it REFUSES the live service (7687) and, outside
//! CI, anything but the harness port 17687. Run locally:
//!   docker run -d --rm --name po-int -p 17687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5
//!   NEO4J_URI=bolt://localhost:17687 NEO4J_PASSWORD=testpassword cargo test --test neo4j_integer_fidelity
//!
//! The tests print a notice and return when no Neo4j is reachable;
//! `HARNESS_NEO4J_REQUIRED=1` (set in CI) turns that into a failure.
mod common;

#[allow(unused_imports)]
use project_orchestrator::neo4j::traits::*;

use common::harness::{neo4j_target, Cleanup, Neo4jTarget};
use neo4rs::{query, Graph, Node, Query, Row};
use project_orchestrator::neo4j::client::Neo4jClient;
use project_orchestrator::protocol::{Protocol, ProtocolState};
use std::collections::HashMap;
use uuid::Uuid;

struct Env {
    client: Neo4jClient,
    raw: Graph,
    target: Neo4jTarget,
}

async fn env() -> Option<Env> {
    // Panics on a target that is not the harness: before any connection, any write.
    let target = neo4j_target();
    let connected = async {
        let client = Neo4jClient::new(&target.uri, &target.user, &target.password)
            .await
            .ok()?;
        let raw = Graph::new(&target.uri, &target.user, &target.password)
            .await
            .ok()?;
        Some((client, raw))
    }
    .await;
    match connected {
        Some((client, raw)) => Some(Env {
            client,
            raw,
            target,
        }),
        None => {
            assert!(
                std::env::var("HARNESS_NEO4J_REQUIRED").is_err(),
                "HARNESS_NEO4J_REQUIRED is set but no Neo4j answers at {}",
                target.uri
            );
            eprintln!(
                "Skipping integer fidelity Neo4j tests: no Neo4j at {}",
                target.uri
            );
            None
        }
    }
}

/// Every boundary of the Bolt integer encodings (TINY_INT -16..=127, INT_8,
/// INT_16, INT_32, INT_64), and the whole negative TINY_INT range.
fn values() -> Vec<i64> {
    let mut v: Vec<i64> = vec![-129, -128, -17];
    v.extend(-16..=-1);
    v.extend([0, 1, 127, 128]);
    v.extend([i16::MIN as i64, i16::MAX as i64]);
    v.extend([i32::MIN as i64, i32::MAX as i64]);
    v.extend([i64::MIN, i64::MAX]);
    v
}

fn texts(values: &[i64]) -> Vec<String> {
    values.iter().map(i64::to_string).collect()
}

async fn rows(graph: &Graph, q: Query) -> Vec<Row> {
    let mut stream = graph.execute(q).await.unwrap();
    let mut out = Vec::new();
    while let Some(row) = stream.next().await.unwrap() {
        out.push(row);
    }
    out
}

/// DECODE. The server builds each integer from its decimal text (`toInteger`),
/// so nothing here depends on how the driver encodes one.
#[tokio::test]
async fn an_integer_is_read_as_the_server_holds_it() {
    let Some(e) = env().await else { return };
    let expected = values();
    let read = rows(
        &e.raw,
        query(
            "UNWIND $texts AS t WITH toInteger(t) AS v
             RETURN v AS alone, [v, 0, v] AS list, {v: v, zero: 0} AS map",
        )
        .param("texts", texts(&expected)),
    )
    .await;

    let alone: Vec<i64> = read.iter().map(|r| r.get("alone").unwrap()).collect();
    assert_eq!(alone, expected, "an integer alone in a column");

    let list: Vec<Vec<i64>> = read.iter().map(|r| r.get("list").unwrap()).collect();
    let expected_list: Vec<Vec<i64>> = expected.iter().map(|&v| vec![v, 0, v]).collect();
    assert_eq!(list, expected_list, "an integer inside a list");

    let map: Vec<i64> = read
        .iter()
        .map(|r| r.get::<HashMap<String, i64>>("map").unwrap()["v"])
        .collect();
    assert_eq!(map, expected, "an integer inside a map");

    // The untyped path (`row.get::<serde_json::Value>`), used where the backend
    // forwards a value it does not model.
    let json: Vec<serde_json::Value> = read.iter().map(|r| r.get("alone").unwrap()).collect();
    let expected_json: Vec<serde_json::Value> = expected.iter().map(|&v| v.into()).collect();
    assert_eq!(json, expected_json, "an integer read as JSON");
}

/// ENCODE, judged by the server (`toString` of what it received), then the
/// full round trip through the driver.
#[tokio::test]
async fn an_integer_parameter_reaches_the_server_and_comes_back() {
    let Some(e) = env().await else { return };
    let expected = values();
    let mut received = Vec::new();
    let mut back = Vec::new();
    for &v in &expected {
        let row = rows(
            &e.raw,
            query(
                "RETURN toString($v) AS v_text, [x IN $l | toString(x)] AS l_text,
                        toString($m.v) AS m_text, $v AS v, $l AS l, $m AS m",
            )
            .param("v", v)
            .param("l", vec![v, 0, v])
            .param("m", HashMap::from([("v".to_string(), v)])),
        )
        .await
        .remove(0);
        received.push((
            row.get::<String>("v_text").unwrap(),
            row.get::<Vec<String>>("l_text").unwrap(),
            row.get::<String>("m_text").unwrap(),
        ));
        back.push((
            row.get::<i64>("v").unwrap(),
            row.get::<Vec<i64>>("l").unwrap(),
            row.get::<HashMap<String, i64>>("m").unwrap()["v"],
        ));
    }
    let expected_received: Vec<(String, Vec<String>, String)> = expected
        .iter()
        .map(|v| (v.to_string(), texts(&[*v, 0, *v]), v.to_string()))
        .collect();
    assert_eq!(received, expected_received, "what the server received");
    let expected_back: Vec<(i64, Vec<i64>, i64)> =
        expected.iter().map(|&v| (v, vec![v, 0, v], v)).collect();
    assert_eq!(back, expected_back, "what came back through the driver");
}

/// A STORED property, read off the node the way the `node_to_*` readers do.
#[tokio::test]
async fn an_integer_property_is_stored_and_read_back() {
    let Some(e) = env().await else { return };
    let expected = values();
    let run = Uuid::new_v4().to_string();
    let mut cleanup = Cleanup::new(&e.target);
    cleanup.later(
        "MATCH (n:IntegerFidelityProbe {run: $p}) DETACH DELETE n",
        run.clone(),
    );
    let positions: Vec<i64> = (0..expected.len() as i64).collect();
    e.raw
        .run(
            query(
                "UNWIND $positions AS i
                 CREATE (:IntegerFidelityProbe {run: $run, i: i, v: $values[i], l: [$values[i], 0]})",
            )
            .param("run", run.clone())
            .param("positions", positions)
            .param("values", expected.clone()),
        )
        .await
        .unwrap();
    let read = rows(
        &e.raw,
        query(
            "MATCH (n:IntegerFidelityProbe {run: $run})
             RETURN n, toString(n.v) AS stored ORDER BY n.i",
        )
        .param("run", run.clone()),
    )
    .await;

    let stored: Vec<String> = read.iter().map(|r| r.get("stored").unwrap()).collect();
    assert_eq!(stored, texts(&expected), "what the server stored");
    let nodes: Vec<Node> = read.iter().map(|r| r.get("n").unwrap()).collect();
    let property: Vec<i64> = nodes.iter().map(|n| n.get("v").unwrap()).collect();
    assert_eq!(property, expected, "a property read off the node");
    let in_list: Vec<Vec<i64>> = nodes.iter().map(|n| n.get("l").unwrap()).collect();
    let expected_list: Vec<Vec<i64>> = expected.iter().map(|&v| vec![v, 0]).collect();
    assert_eq!(in_list, expected_list, "a list property read off the node");
}

/// The backend's own sentinel: a state WITHOUT a timeout is stored `-1` and
/// must be read back without one — not with 255 seconds (the driver bug), and
/// not with `u64::MAX` seconds (`-1 as u64`, what the reader did with a
/// correctly decoded -1). A read-then-write (`protocol::seed::seed_prompt_fragments`
/// does exactly that) must leave the sentinel in place.
#[tokio::test]
async fn a_state_without_timeout_stays_without_timeout() {
    let Some(e) = env().await else { return };
    let mut without = ProtocolState::start(Uuid::new_v4(), "no timeout");
    let protocol = Protocol::new(None, "integer fidelity", without.id);
    without.protocol_id = protocol.id;
    let mut cleanup = Cleanup::new(&e.target);
    cleanup_protocol(&mut cleanup, protocol.id);
    let mut with = ProtocolState::terminal(protocol.id, "timeout of 300 s");
    with.state_timeout_secs = Some(300);
    e.client.upsert_protocol(&protocol).await.unwrap();
    e.client.upsert_protocol_state(&without).await.unwrap();
    e.client.upsert_protocol_state(&with).await.unwrap();

    // Read, then write back what was read: the path that used to store 255.
    for state in e.client.get_protocol_states(protocol.id).await.unwrap() {
        e.client.upsert_protocol_state(&state).await.unwrap();
    }
    let read: HashMap<Uuid, Option<u64>> = e
        .client
        .get_protocol_states(protocol.id)
        .await
        .unwrap()
        .into_iter()
        .map(|s| (s.id, s.state_timeout_secs))
        .collect();
    let stored: Vec<String> = rows(
        &e.raw,
        query(
            "MATCH (s:ProtocolState {protocol_id: $p})
             RETURN toString(s.state_timeout_secs) AS t ORDER BY s.name",
        )
        .param("p", protocol.id.to_string()),
    )
    .await
    .iter()
    .map(|r| r.get("t").unwrap())
    .collect();

    assert_eq!(read[&without.id], None, "no timeout is read as no timeout");
    assert_eq!(read[&with.id], Some(300), "a real timeout is kept");
    assert_eq!(stored, ["-1", "300"], "the sentinel survives the rewrite");
}

fn cleanup_protocol(cleanup: &mut Cleanup, protocol: Uuid) {
    cleanup.later(
        "MATCH (s:ProtocolState {protocol_id: $p}) DETACH DELETE s",
        protocol.to_string(),
    );
    cleanup.later(
        "MATCH (p:Protocol {id: $p}) DETACH DELETE p",
        protocol.to_string(),
    );
}

async fn stored_timeout(raw: &Graph, state: Uuid) -> String {
    rows(
        raw,
        query("MATCH (s:ProtocolState {id: $id}) RETURN toString(s.state_timeout_secs) AS t")
            .param("id", state.to_string()),
    )
    .await
    .remove(0)
    .get("t")
    .unwrap()
}

/// The 255 an older binary (or the driver bug) already wrote: the boot repair
/// resets them to `-1`, leaves what is not 255, does nothing the second time —
/// and does it AGAIN when an old binary, still connected, rewrites a 255.
#[tokio::test]
async fn the_phantom_255_timeouts_are_reset_on_every_start() {
    let Some(e) = env().await else { return };
    let project = Uuid::new_v4();
    let mut cleanup = Cleanup::new(&e.target);
    cleanup.later(
        "MATCH (p:Project {id: $p}) DETACH DELETE p",
        project.to_string(),
    );
    e.raw
        .run(
            query("CREATE (:Project {id: $p, name: 'phantom 255', slug: $p, root_path: '', created_at: '2026-01-01T00:00:00Z'})")
                .param("p", project.to_string()),
        )
        .await
        .unwrap();
    let mut start = ProtocolState::start(Uuid::new_v4(), "start");
    let protocol = Protocol::new(Some(project), "phantom 255", start.id);
    start.protocol_id = protocol.id;
    cleanup_protocol(&mut cleanup, protocol.id);
    let mut real = ProtocolState::terminal(protocol.id, "end");
    real.state_timeout_secs = Some(300);
    // A 255 someone CHOSE (what a handler exposing the field would write): the
    // write path marks it, the repair must leave it.
    let mut chosen = ProtocolState::new(protocol.id, "chosen 255");
    chosen.state_timeout_secs = Some(255);
    e.client.upsert_protocol(&protocol).await.unwrap();
    e.client.upsert_protocol_state(&start).await.unwrap();
    e.client.upsert_protocol_state(&real).await.unwrap();
    e.client.upsert_protocol_state(&chosen).await.unwrap();
    // What the driver bug left in the graph: a state without timeout stored 255.
    let plant = |id: Uuid| {
        query("MATCH (s:ProtocolState {id: $id}) SET s.state_timeout_secs = 255")
            .param("id", id.to_string())
    };
    e.raw.run(plant(start.id)).await.unwrap();
    assert_eq!(
        stored_timeout(&e.raw, start.id).await,
        "255",
        "the ghost is there"
    );
    // The FIXED binary reads the ghost as a real 255 and writes it back
    // (seed_prompt_fragments does that): a value carried over is not a choice,
    // so the ghost earns no marker and is still repaired below.
    for state in e.client.get_protocol_states(protocol.id).await.unwrap() {
        e.client.upsert_protocol_state(&state).await.unwrap();
    }

    let first = e.client.repair_phantom_state_timeouts(Some(project)).await;
    assert_eq!(
        (first.processed, first.error.as_deref()),
        (1, None),
        "{first:?}"
    );
    assert_eq!(stored_timeout(&e.raw, start.id).await, "-1", "repaired");
    assert_eq!(
        stored_timeout(&e.raw, real.id).await,
        "300",
        "what is not 255 is untouched"
    );
    assert_eq!(
        stored_timeout(&e.raw, chosen.id).await,
        "255",
        "a 255 someone chose is kept"
    );
    let states = e.client.get_protocol_states(protocol.id).await.unwrap();
    let read = |id: Uuid| {
        states
            .iter()
            .find(|s| s.id == id)
            .unwrap()
            .state_timeout_secs
    };
    assert_eq!(
        (read(start.id), read(real.id), read(chosen.id)),
        (None, Some(300), Some(255))
    );

    let second = e.client.repair_phantom_state_timeouts(Some(project)).await;
    assert_eq!(
        (second.processed, second.error.as_deref()),
        (0, None),
        "idempotent"
    );

    // An OLD binary, still connected, reads -1 as 255 and rewrites it.
    e.raw.run(plant(start.id)).await.unwrap();
    let again = e.client.repair_phantom_state_timeouts(Some(project)).await;
    assert_eq!(again.processed, 1, "replayed at the next start: {again:?}");
    assert_eq!(stored_timeout(&e.raw, start.id).await, "-1");
}

/// A count, a size, a position: stored `-1` for "none", read as 0 — never as
/// `usize::MAX` / `u32::MAX` (`-1 as usize`), and an optional unsigned as `None`.
#[tokio::test]
async fn a_negative_stored_count_is_read_as_zero_not_as_a_huge_number() {
    let Some(e) = env().await else { return };
    let (project, run, plan) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let path = format!("/integer-fidelity/{project}.rs");
    let mut cleanup = Cleanup::new(&e.target);
    cleanup.later(
        "MATCH (n:Project {id: $p}) DETACH DELETE n",
        project.to_string(),
    );
    cleanup.later(
        "MATCH (n:PlanRun {run_id: $p}) DETACH DELETE n",
        run.to_string(),
    );
    cleanup.later(
        "MATCH (n:File {project_id: $p}) DETACH DELETE n",
        project.to_string(),
    );
    e.raw
        .run(
            query(
                "CREATE (:Project {id: $p, name: 'neg', slug: $p, root_path: '',
                                   created_at: '2026-01-01T00:00:00Z', scaffolding_override: -1})
                 CREATE (:PlanRun {run_id: $run, plan_id: $plan, status: 'running',
                                   started_at: '2026-01-01T00:00:00Z',
                                   total_tasks: -1, current_wave: -1})
                 CREATE (:File {path: $path, project_id: $p, cc_version: 1,
                                cc_community_id: -1, cc_imports_out: -1, cc_imports_in: -1,
                                cc_calls_out: -1, cc_calls_in: -1})",
            )
            .param("p", project.to_string())
            .param("run", run.to_string())
            .param("plan", plan.to_string())
            .param("path", path.clone()),
        )
        .await
        .unwrap();

    let p = e.client.get_project(project).await.unwrap().unwrap();
    assert_eq!(p.scaffolding_override, None, "project: an optional u8");
    let r = e.client.get_plan_run(run).await.unwrap().unwrap();
    assert_eq!((r.total_tasks, r.current_wave), (0, 0), "plan run");
    let c = e
        .client
        .get_context_card(&path, &project.to_string())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (
            c.cc_community_id,
            c.cc_imports_out,
            c.cc_imports_in,
            c.cc_calls_out,
            c.cc_calls_in
        ),
        (0, 0, 0, 0, 0),
        "context card"
    );
    // The batch reader of the same cards is a second reader of the same columns.
    let batch = e
        .client
        .get_context_cards_batch(&[path], &project.to_string())
        .await
        .unwrap();
    assert_eq!(batch.len(), 1);
    let c = &batch[0];
    assert_eq!(
        (
            c.cc_community_id,
            c.cc_imports_out,
            c.cc_imports_in,
            c.cc_calls_out,
            c.cc_calls_in
        ),
        (0, 0, 0, 0, 0),
        "context cards, batch"
    );

    // A step's position and a run's depth and version: the readers of task.rs /
    // step.rs and protocol.rs. -17 is OUTSIDE the range the driver bug touched:
    // the reader's own `as u32` would make it 4 294 967 279.
    let (step, prun) = (Uuid::new_v4(), Uuid::new_v4());
    cleanup.later("MATCH (n:Step {id: $p}) DETACH DELETE n", step.to_string());
    cleanup.later(
        "MATCH (n:ProtocolRun {id: $p}) DETACH DELETE n",
        prun.to_string(),
    );
    e.raw
        .run(
            query(
                "CREATE (:Step {id: $step, order: -17, description: 'neg', status: 'pending',
                                created_at: '2026-01-01T00:00:00Z'})
                 CREATE (:ProtocolRun {id: $run, protocol_id: $proto, current_state: $state,
                                       status: 'running', started_at: '2026-01-01T00:00:00Z',
                                       states_visited_json: '[]', depth: -17, version: -1})",
            )
            .param("step", step.to_string())
            .param("run", prun.to_string())
            .param("proto", Uuid::new_v4().to_string())
            .param("state", Uuid::new_v4().to_string()),
        )
        .await
        .unwrap();
    let s = e.client.get_step(step).await.unwrap().unwrap();
    assert_eq!(s.order, 0, "step order");
    let r = e.client.get_protocol_run(prun).await.unwrap().unwrap();
    assert_eq!(
        (r.depth, r.version),
        (0, 0),
        "protocol run depth and version"
    );
}

/// The guard of the target: the live service is refused whatever the shell
/// loaded, a refusal is a panic and not a skip.
#[test]
fn the_guard_refuses_the_live_service_and_foreign_hosts() {
    use common::harness::{check_harness_uri, HARNESS_PORT};
    let harness = format!("bolt://localhost:{HARNESS_PORT}");
    assert_eq!(check_harness_uri(&harness, false), Ok(()));
    assert_eq!(check_harness_uri("bolt://127.0.0.1:17687", false), Ok(()));
    // The live ports, in and out of CI, with or without a port in the URI.
    for live in [
        "bolt://localhost:7687",
        "bolt://localhost",
        "neo4j://127.0.0.1:7687",
        "bolt://[::1]:7687",
    ] {
        assert!(check_harness_uri(live, false).is_err(), "{live}");
        assert!(check_harness_uri(live, true).is_err(), "{live} in CI");
    }
    assert!(check_harness_uri("bolt://localhost:7474", true).is_err());
    // Outside CI only the harness port; in CI a random service port on loopback.
    assert!(check_harness_uri("bolt://localhost:27687", false).is_err());
    assert_eq!(check_harness_uri("bolt://localhost:49321", true), Ok(()));
    // Never another machine, whatever the port; credentials in the URI are ignored.
    assert!(check_harness_uri("bolt://db.example.com:17687", false).is_err());
    assert!(check_harness_uri("bolt://neo4j:pw@10.0.0.5:49321", true).is_err());
    assert_eq!(
        check_harness_uri("bolt://neo4j:pw@localhost:17687", false),
        Ok(())
    );
    assert!(check_harness_uri("not a uri", true).is_err());
    assert!(check_harness_uri("bolt://localhost:notaport", true).is_err());
    // An empty variable is said as such, not as an empty host name.
    assert_eq!(
        check_harness_uri("", false),
        Err("the address is unset/empty".to_string())
    );
    assert_eq!(
        check_harness_uri("  ", true),
        Err("the address is unset/empty".to_string())
    );
}

/// The same rule for the Meilisearch `AppState::new` opens and whose index settings
/// it rewrites: never the live 7700, outside CI only the harness 17700.
#[test]
fn the_guard_refuses_the_live_meilisearch_too() {
    use common::harness::{check_harness_meili_url, meili_target_for, HARNESS_MEILI_PORT};
    assert_eq!(
        check_harness_meili_url(&format!("http://localhost:{HARNESS_MEILI_PORT}"), false),
        Ok(())
    );
    assert_eq!(
        check_harness_meili_url("http://127.0.0.1:17700", false),
        Ok(())
    );
    for live in [
        "http://localhost:7700",
        "http://127.0.0.1:7700",
        "http://[::1]:7700",
    ] {
        assert!(check_harness_meili_url(live, false).is_err(), "{live}");
        assert!(check_harness_meili_url(live, true).is_err(), "{live} in CI");
    }
    // No port means the scheme's 80: not a harness.
    assert!(check_harness_meili_url("http://localhost", false).is_err());
    assert!(check_harness_meili_url("http://localhost:27700", false).is_err());
    assert_eq!(
        check_harness_meili_url("http://localhost:49321", true),
        Ok(())
    );
    assert!(check_harness_meili_url("http://search.example.com:17700", true).is_err());
    assert!(check_harness_meili_url("", true).is_err());
    let t = meili_target_for(None, false);
    assert_eq!(t.url, format!("http://localhost:{HARNESS_MEILI_PORT}"));
}

#[test]
#[should_panic(expected = "REFUSED to point a writing test at this Meilisearch")]
fn a_live_meilisearch_url_is_a_failure_not_a_skip() {
    let _ = common::harness::meili_target_for(Some("http://localhost:7700".into()), true);
}

/// `neo4j_target` PANICS on a live URI (a skip would be a green run that tested nothing).
#[test]
#[should_panic(expected = "REFUSED to point a writing test at this Neo4j")]
fn a_live_uri_in_the_environment_is_a_failure_not_a_skip() {
    let _ = common::harness::target_for(Some("bolt://localhost:7687".into()), false);
}
