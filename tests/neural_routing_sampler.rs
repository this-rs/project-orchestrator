//! Contract of the GNN sampler against a real server.
//!
//! The sampler is the adapter half of graph sampling: it drives the graph and
//! returns the records the models consume. Its pure half (record shapes, PyG
//! export) is tested in the domain crate; what is exercised here is the part
//! that only exists because something answers Cypher.
//!
//! Skipped when no server answers — the same contract as `tests/harness_neo4j.rs`.
//! `HARNESS_NEO4J_REQUIRED=1` (exported by the CI jobs that provision a server)
//! turns that skip into a failure.
//!
//!   docker run -d --rm --name po-sampler -p 17687:7687 \
//!     -e NEO4J_AUTH=neo4j/testpassword neo4j:5
//!   NEO4J_URI=bolt://localhost:17687 NEO4J_PASSWORD=testpassword \
//!     cargo test --test neural_routing_sampler
#[allow(unused_imports)]
use project_orchestrator::neo4j::traits::*;

use std::sync::Arc;
use std::time::Duration;

use neo4rs::query;
use project_orchestrator::neo4j::graph_sampler::{GraphSampler, GraphSamplerConfig, SamplerError};

/// Whether something is listening on the Bolt port of `uri`.
///
/// A socket probe, not a Cypher round-trip: every `neo4rs` query carries a
/// retrying backoff, so a probe that asks the database sits in the retry loop
/// instead of failing fast when nothing answers.
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

async fn sampler() -> Option<(GraphSampler, Arc<neo4rs::Graph>, String)> {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:17687".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into());

    let graph = if bolt_reachable(&uri).await {
        neo4rs::Graph::new(&uri, &user, &password).await.ok()
    } else {
        None
    };

    match graph {
        Some(graph) => {
            let graph = Arc::new(graph);
            let config = GraphSamplerConfig {
                default_k_hops: 2,
                max_nodes_per_subgraph: 50,
                cache_capacity: 8,
                ..GraphSamplerConfig::default()
            };
            let prefix = format!("sampler-test-{}", uuid::Uuid::new_v4());
            Some((GraphSampler::new(Arc::clone(&graph), config), graph, prefix))
        }
        None => {
            assert!(
                std::env::var("HARNESS_NEO4J_REQUIRED").is_err(),
                "HARNESS_NEO4J_REQUIRED is set but no server answers at {uri}"
            );
            eprintln!("skipping the sampler contract: no server at {uri}");
            None
        }
    }
}

/// Seeds a path of three `File` nodes: centre —(CALLS)— middle —(CALLS)— far.
async fn seed(graph: &neo4rs::Graph, prefix: &str) -> (String, String, String) {
    let (c, m, f) = (
        format!("{prefix}-c"),
        format!("{prefix}-m"),
        format!("{prefix}-f"),
    );
    graph
        .run(
            query(
                "MERGE (c:File {id: $c}) SET c.path = $c, c.pagerank = 0.9
                 MERGE (m:File {id: $m}) SET m.path = $m, m.pagerank = 0.5
                 MERGE (f:File {id: $f}) SET f.path = $f, f.pagerank = 0.1
                 MERGE (c)-[:CALLS]->(m)
                 MERGE (m)-[:CALLS]->(f)",
            )
            .param("c", c.clone())
            .param("m", m.clone())
            .param("f", f.clone()),
        )
        .await
        .expect("seed the sampler fixture");
    (c, m, f)
}

async fn cleanup(graph: &neo4rs::Graph, prefix: &str) {
    let _ = graph
        .run(
            query("MATCH (n) WHERE n.id STARTS WITH $p DETACH DELETE n")
                .param("p", prefix.to_string()),
        )
        .await;
}

#[tokio::test]
async fn an_ego_graph_follows_the_hops_and_the_cache_answers_identically() {
    let Some((sampler, graph, prefix)) = sampler().await else {
        return;
    };
    let (center, middle, far) = seed(&graph, &prefix).await;

    let one_hop = sampler
        .sample_ego_graph(&center, Some(1))
        .await
        .expect("1 hop");
    let ids: Vec<&str> = one_hop.nodes.iter().map(|n| n.app_id.as_str()).collect();
    assert!(
        ids.contains(&center.as_str()),
        "the centre belongs to its own ego-graph"
    );
    assert!(
        ids.contains(&middle.as_str()),
        "1 hop must reach the direct neighbour"
    );
    assert!(
        !ids.contains(&far.as_str()),
        "1 hop must not reach the second neighbour"
    );

    let two_hops = sampler
        .sample_ego_graph(&center, Some(2))
        .await
        .expect("2 hops");
    let ids: Vec<&str> = two_hops.nodes.iter().map(|n| n.app_id.as_str()).collect();
    assert!(
        ids.contains(&far.as_str()),
        "2 hops must reach the second neighbour"
    );
    assert!(
        !two_hops.edges.is_empty(),
        "the seeded CALLS edges must be extracted"
    );

    // Second call is served from the cache and must be identical.
    let (before, _) = sampler.cache_stats();
    let again = sampler
        .sample_ego_graph(&center, Some(2))
        .await
        .expect("cached");
    assert_eq!(
        before,
        sampler.cache_stats().0,
        "a repeated call does not grow the cache"
    );
    assert_eq!(again.nodes.len(), two_hops.nodes.len());

    // Invalidation empties it, and the next call rebuilds from the graph.
    sampler.invalidate_cache(None);
    assert_eq!(
        sampler.cache_stats().0,
        0,
        "invalidate(None) clears every entry"
    );
    let rebuilt = sampler
        .sample_ego_graph(&center, Some(2))
        .await
        .expect("rebuilt");
    assert_eq!(rebuilt.nodes.len(), two_hops.nodes.len());

    cleanup(&graph, &prefix).await;
}

#[tokio::test]
async fn an_unknown_centre_is_named_and_a_batch_skips_it() {
    let Some((sampler, graph, prefix)) = sampler().await else {
        return;
    };
    let (center, _, _) = seed(&graph, &prefix).await;
    let ghost = format!("{prefix}-absent");

    match sampler.sample_ego_graph(&ghost, None).await {
        Err(SamplerError::NodeNotFound(id)) => assert_eq!(id, ghost),
        other => panic!("an unknown centre must be named, got {other:?}"),
    }

    let batch = sampler
        .sample_batch(&[center.clone(), ghost], None)
        .await
        .expect("a batch skips the unknown centre instead of failing");
    assert_eq!(
        batch.len(),
        1,
        "one known centre, one unknown: one subgraph comes back"
    );

    cleanup(&graph, &prefix).await;
}

#[tokio::test]
async fn stratified_sampling_returns_the_seeded_files_ordered_by_rank() {
    let Some((sampler, graph, prefix)) = sampler().await else {
        return;
    };
    let (center, middle, far) = seed(&graph, &prefix).await;

    // A large count: the fixture pages are seeded with the same ranks in every
    // test of this file, so a small LIMIT would return an arbitrary three among
    // the ties instead of the three this test created.
    let picked = sampler
        .sample_stratified_nodes(100, None)
        .await
        .expect("stratified");
    let ours: Vec<&String> = picked.iter().filter(|id| id.starts_with(&prefix)).collect();
    assert_eq!(
        ours.len(),
        3,
        "the three seeded files are eligible: {ours:?}"
    );
    assert_eq!(
        ours.first().map(|s| s.as_str()),
        Some(center.as_str()),
        "the highest pagerank comes first: {ours:?}"
    );
    assert!(ours.contains(&&middle) && ours.contains(&&far));

    cleanup(&graph, &prefix).await;
}
