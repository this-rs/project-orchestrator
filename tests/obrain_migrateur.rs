//! The real Neo4j → obrain copy (`storage::migrator::Neo4jToObrain`) behind the
//! startup switch, against a THROWAWAY Neo4j.
//!
//! These tests need a Neo4j that is not PO's: they WIPE it (`MATCH (n) DETACH
//! DELETE n`) before seeding. They refuse the live port outright, and are
//! `#[ignore]`d so a plain `cargo test` never looks for one:
//!
//! ```text
//! docker run -d --rm --name po-migrateur-jetable -p 27688:7687 -e NEO4J_AUTH=none neo4j:5.26.20-community
//! PO_MIGRATION_TEST_NEO4J=bolt://127.0.0.1:27688 \
//!   cargo test --features obrain-backend --test obrain_migrateur -- --ignored --test-threads=1
//! ```
//!
//! What they prove: (1) per-type counts equal, and a project seeded through PO's
//! own Neo4j client is READ BACK by PO's obrain store; (2) the switch happens
//! once — a restart serves the target on its marker without calling the
//! migrator; (3) an injected discrepancy leaves PO on the source with the cause
//! named, and writes no marker.

#![cfg(feature = "obrain-backend")]

use neo4rs::{query, ConfigBuilder, Graph};
use project_orchestrator::neo4j::client::Neo4jClient;
use project_orchestrator::neo4j::models::{ProjectNode, ProjectProfile};
use project_orchestrator::obrain::{Creation, ObrainGraphStore, RapportOuverture};
use project_orchestrator::storage::migrate::{
    run_startup_migration, MigrationRequest, Migrator, PerTypeCounts, ReadOnlyGate, StartupOutcome,
    DONE_MARKER,
};
use project_orchestrator::storage::migrator::Neo4jToObrain;
use project_orchestrator::storage::StorageBackend;
use project_orchestrator::Config;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use uuid::Uuid;

/// The ONLY sources these tests accept: a loopback Neo4j on a port dedicated to
/// throwaway containers (17687: the harness of `tests/common/harness.rs`; 27688:
/// the read-only/throwaway copy). Anything else — the live 7687, a URI with no
/// port, a remote host — is refused BEFORE any write: the tests wipe their source.
const DISPOSABLE_PORTS: [u16; 2] = [17687, 27688];

fn check_disposable(uri: &str) -> Result<(), String> {
    let rest = uri
        .strip_prefix("bolt://")
        .ok_or_else(|| format!("{uri}: only bolt:// URIs are accepted"))?;
    let (host, port) = rest
        .rsplit_once(':')
        .ok_or_else(|| format!("{uri}: no explicit port (a default port may be the live base)"))?;
    if host != "127.0.0.1" && host != "localhost" {
        return Err(format!("{uri}: host {host} is not loopback"));
    }
    let port: u16 = port
        .trim_end_matches('/')
        .parse()
        .map_err(|_| format!("{uri}: unreadable port"))?;
    if !DISPOSABLE_PORTS.contains(&port) {
        return Err(format!(
            "{uri}: port {port} is not a throwaway port {DISPOSABLE_PORTS:?}: these tests wipe their source"
        ));
    }
    Ok(())
}

fn uri() -> String {
    let uri = std::env::var("PO_MIGRATION_TEST_NEO4J")
        .expect("PO_MIGRATION_TEST_NEO4J = the bolt URI of a THROWAWAY Neo4j");
    if let Err(why) = check_disposable(&uri) {
        panic!("refused before any write: {why}");
    }
    uri
}

#[test]
fn the_wipe_guard_accepts_only_the_dedicated_throwaway_ports() {
    for ok in ["bolt://127.0.0.1:27688", "bolt://localhost:17687"] {
        assert_eq!(check_disposable(ok), Ok(()), "{ok}");
    }
    for refused in [
        "bolt://127.0.0.1:7687",
        "bolt://localhost",
        "bolt://127.0.0.1",
        "bolt://neo4j.example.com:27688",
        "neo4j://127.0.0.1:27688",
        "bolt://127.0.0.1:7688",
        "bolt://127.0.0.1:17687abc",
    ] {
        assert!(
            check_disposable(refused).is_err(),
            "{refused} must be refused"
        );
    }
}

async fn graph() -> Graph {
    Graph::connect(
        ConfigBuilder::new()
            .uri(uri())
            .user("")
            .password("")
            .build()
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn run(g: &Graph, cypher: &str) {
    g.run(query(cypher)).await.unwrap();
}

/// Wipe the source, after proving it is throwaway: a base with thousands of
/// nodes is somebody's data, whatever its port.
async fn wipe(g: &Graph) {
    let mut f = g
        .execute(query("MATCH (n) RETURN count(n) AS c"))
        .await
        .unwrap();
    let n: i64 = f.next().await.unwrap().unwrap().get("c").unwrap();
    assert!(
        n < 5000,
        "{n} nodes: this is not a throwaway base, nothing wiped"
    );
    run(g, "MATCH (n) DETACH DELETE n").await;
}

/// A wiped source holding a PO project (through PO's own client, so the shape is
/// the production one) plus data that stresses the conversion.
async fn seed() -> ProjectNode {
    let g = graph().await;
    wipe(&g).await;
    let client = Neo4jClient::new(&uri(), "", "").await.unwrap();
    let id = Uuid::new_v4();
    let project = ProjectNode {
        id,
        name: "migrated project".into(),
        slug: format!("mig-{id}"),
        root_path: String::new(),
        description: Some("seeded through the Neo4j client".into()),
        created_at: chrono::Utc::now(),
        last_synced: None,
        analytics_computed_at: None,
        last_co_change_computed_at: None,
        default_note_energy: None,
        scaffolding_override: None,
        sharing_policy: None,
        watch_enabled: false,
        profile: ProjectProfile::Software,
    };
    client.create_project(&project).await.unwrap();
    run(
        &g,
        "CREATE (a:Probe {k: 'a', neg: -1, small: -16, big: 9007199254740993, f: -0.5, \
                  when: datetime('2026-11-20T10:00:00.123456+01:00'), day: date('2026-11-20'), \
                  l: [1, -2, 3], vec: [0.5, 0.25]}) \
         CREATE (b:Probe {k: 'b', note: 'é\u{1F600}'}) \
         CREATE (c:Other {k: 'c'}) \
         CREATE (a)-[:LINKS {w: -7, tag: 'x'}]->(b) \
         CREATE (b)-[:LINKS]->(c)",
    )
    .await;
    project
}

fn config(dir: &Path, base: &Path) -> Config {
    for v in [
        "STORAGE_BACKEND",
        "STORAGE_OBRAIN_PATH",
        "STORAGE_MIGRATION_FROM",
        "NEO4J_URI",
    ] {
        assert!(std::env::var_os(v).is_none(), "{v} would override the YAML");
    }
    let file = dir.join("config.yaml");
    std::fs::write(
        &file,
        format!(
            "setup_completed: true\nneo4j:\n  uri: \"{}\"\n  user: \"\"\n  password: \"\"\n\
             storage:\n  backend: obrain\n  obrain_path: \"{}\"\n  migration:\n    from: default\n",
            uri(),
            base.display()
        ),
    )
    .unwrap();
    Config::from_yaml_and_env(Some(&file)).unwrap()
}

/// Counts the calls the startup makes to the real migrator.
struct Counting {
    inner: Neo4jToObrain,
    calls: AtomicUsize,
}
#[async_trait::async_trait]
impl Migrator for Counting {
    async fn source_counts(&self) -> anyhow::Result<PerTypeCounts> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.source_counts().await
    }
    async fn copy(&self, dest: &Path) -> anyhow::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.copy(dest).await
    }
    async fn target_counts(&self, dest: &Path) -> anyhow::Result<PerTypeCounts> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.target_counts(dest).await
    }
}

fn migrator() -> Neo4jToObrain {
    Neo4jToObrain::new(&uri(), "", "")
}

fn request(base: &Path) -> MigrationRequest {
    MigrationRequest {
        source: "default".into(),
        target: "obrain".into(),
        rollback: false,
        disposable: false,
        configured_dest: Some(base.to_path_buf()),
    }
}

#[tokio::test]
#[ignore = "needs a throwaway Neo4j: PO_MIGRATION_TEST_NEO4J"]
async fn counts_equal_switch_once_and_restart_without_a_copy() {
    let project = seed().await;
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let config = config(dir.path(), &base);
    let gate = ReadOnlyGate::new();

    // (1) first start: copy, verify, marker, serve the obrain base.
    let source = migrator().source_counts().await.unwrap();
    println!("SOURCE per type: {source:?}");
    assert_eq!(source.nodes["Probe"], 2);
    assert_eq!(source.relationships["LINKS"], 2);
    let first = Counting {
        inner: migrator(),
        calls: AtomicUsize::new(0),
    };
    let backend = StorageBackend::from_config_in(&config, &first, &gate)
        .await
        .unwrap();
    assert!(
        first.calls.load(Ordering::SeqCst) >= 3,
        "source read, copy, target read"
    );
    assert!(base.join(DONE_MARKER).is_file());
    let read = backend
        .graph
        .get_project(project.id)
        .await
        .unwrap()
        .expect("the project seeded in Neo4j is served by the obrain store");
    assert_eq!(read.name, project.name);
    assert_eq!(read.slug, project.slug);
    drop(backend);

    // G7: zero quarantine — PO's own open of the migrated base repairs nothing.
    let b = base.clone();
    let store = tokio::task::spawn_blocking(move || ObrainGraphStore::open(b, Creation::Interdite))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        *store.rapport_d_ouverture(),
        RapportOuverture::default(),
        "nothing quarantined or repaired at open"
    );
    drop(store);

    // The neo4rs defect: -1 and -16 must come out negative, not 255 and 240.
    let probe = {
        let db = obrain_engine::ObrainDB::open_read_only(&base).unwrap();
        let total = source.nodes.values().sum::<usize>() as u64;
        (1..=total)
            .filter_map(|slot| db.store().get_node(obrain_common::types::NodeId(slot)))
            .find(|n| {
                n.properties.iter().any(|(k, v)| {
                    k.as_str() == "k"
                        && matches!(v, obrain_common::types::Value::String(s) if s.as_str() == "a")
                })
            })
            .expect("the Probe 'a' is in the target")
    };
    let int = |key: &str| {
        probe
            .properties
            .iter()
            .find(|(k, _)| k.as_str() == key)
            .map(|(_, v)| v.clone())
    };
    assert_eq!(int("neg"), Some(obrain_common::types::Value::Int64(-1)));
    assert_eq!(int("small"), Some(obrain_common::types::Value::Int64(-16)));
    assert_eq!(
        int("big"),
        Some(obrain_common::types::Value::Int64(9007199254740993))
    );

    // Dates, to the microsecond: a datetime (+01:00, 123456 us), a date, and the
    // project's `created_at` read back through PO's own store.
    use obrain_common::types::Value;
    let want = chrono::DateTime::parse_from_rfc3339("2026-11-20T10:00:00.123456+01:00").unwrap();
    match int("when") {
        Some(Value::ZonedDatetime(z)) => {
            assert_eq!(
                z.as_timestamp().as_nanos(),
                want.timestamp_nanos_opt().unwrap()
            );
            assert_eq!(z.offset_seconds(), 3600);
            assert_eq!(z.as_timestamp().as_nanos() % 1_000, 0);
        }
        other => panic!("`when` read back as {other:?}"),
    }
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let day = chrono::NaiveDate::from_ymd_opt(2026, 11, 20).unwrap();
    match int("day") {
        Some(Value::Date(d)) => {
            assert_eq!(
                i64::from(d.as_days()),
                day.signed_duration_since(epoch).num_days()
            )
        }
        other => panic!("`day` read back as {other:?}"),
    }
    assert_eq!(
        read.created_at.timestamp_micros(),
        project.created_at.timestamp_micros(),
        "created_at survives to the microsecond"
    );

    // The target counts (engine counters) equal the source, type by type.
    let target = migrator().target_counts(&base).await.unwrap();
    println!("TARGET per type: {target:?}");
    for (label, n) in &source.nodes {
        assert_eq!(target.nodes.get(label), Some(n), "label {label}");
    }
    for (kind, n) in &source.relationships {
        assert_eq!(target.relationships.get(kind), Some(n), "type {kind}");
    }

    // (2) restart: the marker is served, the migrator is never called.
    let second = Counting {
        inner: migrator(),
        calls: AtomicUsize::new(0),
    };
    let again = StorageBackend::from_config_in(&config, &second, &gate)
        .await
        .unwrap();
    assert_eq!(second.calls.load(Ordering::SeqCst), 0, "no copy on restart");
    assert!(again.graph.get_project(project.id).await.unwrap().is_some());
}

#[tokio::test]
#[ignore = "needs a throwaway Neo4j: PO_MIGRATION_TEST_NEO4J"]
async fn a_node_with_two_labels_leaves_po_on_the_source_with_the_cause() {
    seed().await;
    let g = graph().await;
    run(&g, "CREATE (:Probe:Other {k: 'two'})").await;
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let gate = ReadOnlyGate::new();
    let scope = gate.enter();
    let outcome = run_startup_migration(&request(&base), &migrator(), &scope)
        .await
        .unwrap();
    let StartupOutcome::StayOnSource { reason } = outcome else {
        panic!("a two-label node must keep PO on the source");
    };
    println!("REASON: {reason}");
    assert!(reason.contains("2 labels"), "{reason}");
    assert!(!base.join(DONE_MARKER).exists());
}

/// The source moves between the per-type read and the copy: the copy is
/// faithful to what it read (its own fingerprint passes), but the switch compares
/// it with the counts read BEFORE, and refuses by naming the label.
#[tokio::test]
#[ignore = "needs a throwaway Neo4j: PO_MIGRATION_TEST_NEO4J"]
async fn an_injected_discrepancy_leaves_po_on_the_source_with_the_label_named() {
    seed().await;
    struct Drifting(Neo4jToObrain);
    #[async_trait::async_trait]
    impl Migrator for Drifting {
        async fn source_counts(&self) -> anyhow::Result<PerTypeCounts> {
            let counts = self.0.source_counts().await?;
            // The injected discrepancy: a node appears after the counts were read.
            run(&graph().await, "CREATE (:Probe {k: 'late'})").await;
            Ok(counts)
        }
        async fn copy(&self, dest: &Path) -> anyhow::Result<()> {
            self.0.copy(dest).await
        }
        async fn target_counts(&self, dest: &Path) -> anyhow::Result<PerTypeCounts> {
            self.0.target_counts(dest).await
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let gate = ReadOnlyGate::new();
    let scope = gate.enter();
    let outcome = run_startup_migration(&request(&base), &Drifting(migrator()), &scope)
        .await
        .unwrap();
    let StartupOutcome::StayOnSource { reason } = outcome else {
        panic!("a discrepancy must keep PO on the source");
    };
    println!("REASON: {reason}");
    assert!(reason.contains("Probe"), "the label is named: {reason}");
    assert!(!base.join(DONE_MARKER).exists(), "no marker was written");
    assert!(
        std::fs::read_dir(&base)
            .map(|mut d| d.next().is_none())
            .unwrap_or(true),
        "what the failed copy wrote was removed"
    );
}

#[tokio::test]
#[ignore = "needs a throwaway Neo4j: PO_MIGRATION_TEST_NEO4J"]
async fn an_empty_source_migrates_to_an_empty_target() {
    let g = graph().await;
    wipe(&g).await;
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let gate = ReadOnlyGate::new();
    let scope = gate.enter();
    let outcome = run_startup_migration(&request(&base), &migrator(), &scope)
        .await
        .unwrap();
    assert!(
        matches!(outcome, StartupOutcome::ServeTarget { .. }),
        "{outcome:?}"
    );
}

/// The fingerprint layer alone: the copy is faithful, then the written base is
/// altered (same counts, one value changed). The per-type counts cannot see it;
/// the read-back must, and must name the label.
#[tokio::test]
#[ignore = "needs a throwaway Neo4j: PO_MIGRATION_TEST_NEO4J"]
async fn a_tampered_target_fails_the_copy_on_the_fingerprint_with_the_label_named() {
    seed().await;
    let tamper: project_orchestrator::storage::migrator::AfterWrite = std::sync::Arc::new(|dest| {
        use obrain_common::types::{NodeId, Value};
        let db = obrain_engine::ObrainDB::open(dest).map_err(|e| anyhow::anyhow!("{e}"))?;
        let total = db.node_count() as u64;
        let slot = (1..=total)
            .find(|s| {
                db.store().get_node(NodeId(*s)).is_some_and(|n| {
                    n.properties.iter().any(|(k, v)| {
                        k.as_str() == "k" && matches!(v, Value::String(x) if x.as_str() == "a")
                    })
                })
            })
            .ok_or_else(|| anyhow::anyhow!("no Probe 'a' to tamper with"))?;
        db.set_node_property(NodeId(slot), "k", Value::from("tampered"));
        db.close().map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    });
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let gate = ReadOnlyGate::new();
    let scope = gate.enter();
    let m = migrator().with_after_write(tamper);
    let outcome = run_startup_migration(&request(&base), &m, &scope)
        .await
        .unwrap();
    let StartupOutcome::StayOnSource { reason } = outcome else {
        panic!("a tampered target must keep PO on the source");
    };
    println!("REASON: {reason}");
    assert!(reason.contains("fingerprint mismatch"), "{reason}");
    assert!(reason.contains("label 'Probe'"), "{reason}");
    assert!(!base.join(DONE_MARKER).exists());
}

/// A node deleted from the source between the identifiers read (phase A) and the
/// properties read (phase D) fails the copy by name; it is never written empty.
#[tokio::test]
#[ignore = "needs a throwaway Neo4j: PO_MIGRATION_TEST_NEO4J"]
async fn a_node_deleted_during_the_copy_fails_it_by_name() {
    seed().await;
    let hook: project_orchestrator::storage::migrator::BeforeProperties =
        std::sync::Arc::new(|| {
            Box::pin(async {
                run(&graph().await, "MATCH (n:Other) DETACH DELETE n").await;
            })
        });
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let gate = ReadOnlyGate::new();
    let scope = gate.enter();
    let m = migrator().with_before_properties(hook);
    let outcome = run_startup_migration(&request(&base), &m, &scope)
        .await
        .unwrap();
    let StartupOutcome::StayOnSource { reason } = outcome else {
        panic!("a node deleted during the copy must keep PO on the source");
    };
    println!("REASON: {reason}");
    assert!(
        reason.contains("deleted from the source during the copy"),
        "{reason}"
    );
    assert!(!base.join(DONE_MARKER).exists());
}
