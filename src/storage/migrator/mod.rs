//! The real data copy of the startup switch: Neo4j → obrain
//! (feature `obrain-backend`). It implements [`Migrator`], the seam
//! `storage::migrate::run_startup_migration` drives; the gate, the per-type
//! verification, the marker and the decision stay in `migrate.rs`.
//!
//! Ported from the measured spike (`spike-migrateur`, ~3 500 lines, measured on
//! a 33 GB production copy). Differences with the spike, all on purpose:
//!
//! - **no absolute path, no external checkout**: the engine is the one pinned
//!   in `Cargo.toml`, the Neo4j driver is the vendored, sign-correct neo4rs
//!   (`vendor/neo4rs/PATCH.md`: the integers -16..=-1 used to be read 240..=255);
//! - **the source is only READ**: this module issues `MATCH … RETURN` and
//!   nothing else (no schema, no constraint, no write);
//! - **a content fingerprint is part of the verification** (see below), where the
//!   spike's oracle was a separate, sampled, manual tool.
//!
//! # What the copy checks, and what it does not
//!
//! Two layers, both before PO may switch:
//!
//! 1. [`Migrator::copy`] ends by reading the base back FROM DISK (every node and
//!    every link slot, independently of the engine's counters) and comparing, per
//!    label and per relationship type, the number of entities, the number of
//!    properties and an order-independent hash of `(key, value)` pairs (plus the
//!    two ends of a link) with what the source side computed on the converted
//!    values it handed to the engine. A difference fails the copy with the label
//!    named — PO then stays on the source (`migrate::copy_and_verify`).
//! 2. `migrate::verify_by_type` compares the SOURCE per-type counts (read from
//!    Neo4j) with the engine's own per-type counters ([`Migrator::target_counts`]).
//!
//! The fingerprint proves the engine stored what this module gave it. It does
//! NOT prove that the Bolt → obrain conversion is faithful to Neo4j's own view
//! of the value: that is the spike's oracle (`valueType()` / `toString()` computed
//! by the server, run by hand on the 33 GB copy: 1 649 integers checked, which
//! is how the neo4rs defect was found). The conversion is covered by unit tests
//! (`conv.rs`) and its known narrowings are counted and logged ([`Losses`]).
//!
//! Limits inherited from the engine's bulk path: one label per node (a node with
//! several labels refuses the copy by name), and the whole relationship list is
//! held in memory while the structure is written.

mod conv;
mod target;

use super::migrate::{Migrator, PerTypeCounts};
use anyhow::{Context, Result};
use conv::{import_value, Losses};
use neo4rs::{query, BoltType, ConfigBuilder, Graph, Row};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use target::{edge_print, node_print, Print, Prints, Props, TargetWriter};

const NODES_BY_RANGE: &str = "UNWIND range($a,$b) AS i MATCH (n) WHERE id(n) = i";
const RELS_BY_RANGE: &str = "UNWIND range($a,$b) AS i MATCH (s)-[r]->(d) WHERE id(r) = i \
     RETURN id(r) AS i, id(s) AS s, id(d) AS d, type(r) AS t, properties(r) AS p";

/// How the copy reads and writes. The defaults are the spike's measured ones.
#[derive(Debug, Clone)]
pub struct Tuning {
    /// Parallel Bolt connections.
    pub parallel: usize,
    /// Identifiers per range read.
    pub width: i64,
    /// Nodes per property batch written to the engine.
    pub batch: usize,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            parallel: 4,
            width: 20_000,
            batch: 20_000,
        }
    }
}

/// The Neo4j → obrain migrator.
pub struct Neo4jToObrain {
    uri: String,
    user: String,
    password: String,
    tuning: Tuning,
    after_write: Option<AfterWrite>,
    before_properties: Option<BeforeProperties>,
}

/// A TEST SEAM: awaited after the identifiers and the structure are read and
/// before the node properties are, so a test can change the live source in
/// between. Production never sets it.
pub type BeforeProperties =
    Arc<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>;

/// A TEST SEAM: runs between the write of the base and its read-back, so a test
/// can tamper with the written base and prove the read-back notices. Production
/// never sets it.
pub type AfterWrite = Arc<dyn Fn(&Path) -> Result<()> + Send + Sync>;

impl Neo4jToObrain {
    pub fn new(uri: &str, user: &str, password: &str) -> Self {
        Self {
            uri: uri.to_string(),
            user: user.to_string(),
            password: password.to_string(),
            tuning: Tuning::default(),
            after_write: None,
            before_properties: None,
        }
    }

    /// Install the test seam [`BeforeProperties`].
    pub fn with_before_properties(mut self, hook: BeforeProperties) -> Self {
        self.before_properties = Some(hook);
        self
    }

    /// Install the test seam [`AfterWrite`].
    pub fn with_after_write(mut self, hook: AfterWrite) -> Self {
        self.after_write = Some(hook);
        self
    }

    /// The migrator for the Neo4j the configuration names (`neo4j.*`).
    pub fn from_config(config: &crate::Config) -> Self {
        Self::new(
            &config.neo4j_uri,
            &config.neo4j_user,
            &config.neo4j_password,
        )
    }

    pub fn with_tuning(mut self, tuning: Tuning) -> Self {
        self.tuning = tuning;
        self
    }

    async fn connect(&self) -> Result<Graph> {
        let config = ConfigBuilder::new()
            .uri(self.uri.as_str())
            .user(self.user.as_str())
            .password(self.password.as_str())
            .max_connections(self.tuning.parallel.max(1))
            .fetch_size(1000)
            .build()
            .context("building the Neo4j configuration")?;
        // `Graph::connect` is lazy and neo4rs retries with growing delays: probe
        // with a bounded `RETURN 1`, so an unreachable source is a named error.
        let probe = async {
            let graph = Graph::connect(config).await?;
            graph.run(query("RETURN 1")).await?;
            Ok::<_, neo4rs::Error>(graph)
        };
        match tokio::time::timeout(std::time::Duration::from_secs(30), probe).await {
            Ok(Ok(g)) => Ok(g),
            Ok(Err(e)) => anyhow::bail!("Neo4j source {} unreachable: {e}", self.uri),
            Err(_) => anyhow::bail!("Neo4j source {} gave no answer within 30 s", self.uri),
        }
    }
}

async fn counts(graph: &Graph, cypher: &str) -> Result<BTreeMap<String, usize>> {
    let mut stream = graph.execute(query(cypher)).await?;
    let mut out = BTreeMap::new();
    while let Some(row) = stream.next().await? {
        let name: Option<String> = row.get("k")?;
        let name = name.context(
            "a node without any label cannot be copied (the obrain bulk import is one label per node)",
        )?;
        out.insert(name, row.get::<i64>("c")? as usize);
    }
    Ok(out)
}

fn props_of(row: &Row, losses: &mut Losses) -> Result<Props> {
    // A row that does not carry its properties as a map is a FAILED READ, not an
    // entity without properties: reading it as empty would make the fingerprint
    // agree on a loss.
    let map = match row.get::<BoltType>("p") {
        Ok(BoltType::Map(m)) => m,
        Ok(other) => anyhow::bail!("the properties of a row are not a map ({other:?})"),
        Err(e) => anyhow::bail!("the properties of a row cannot be read: {e}"),
    };
    let mut props = Props::new();
    props.reserve(map.value.len());
    for (k, v) in &map.value {
        let value = import_value(k.value.as_str(), v, losses)
            .map_err(|e| anyhow::anyhow!("property '{}': {e}", k.value))?;
        props.push((intern(&k.value), value));
    }
    Ok(props)
}

/// Property keys interned once per process: the engine's bulk API takes `&str`,
/// and a base has a few hundred distinct keys.
fn intern(name: &str) -> &'static str {
    use std::sync::{Mutex, OnceLock};
    static TABLE: OnceLock<Mutex<std::collections::HashMap<String, &'static str>>> =
        OnceLock::new();
    let mut t = TABLE.get_or_init(Default::default).lock().unwrap();
    if let Some(s) = t.get(name) {
        return s;
    }
    let s: &'static str = Box::leak(name.to_string().into_boxed_str());
    t.insert(name.to_string(), s);
    s
}

/// `[0, max]` cut into closed ranges of `width` identifiers.
fn ranges(max: i64, width: i64) -> Vec<(i64, i64)> {
    let mut v = Vec::new();
    let mut a = 0;
    while a <= max {
        v.push((a, (a + width - 1).min(max)));
        a += width;
    }
    v
}

/// Read `ranges` over `parallel` connections. Each range comes back whole to
/// `process`; results go on a bounded channel, so a slow consumer slows the read
/// instead of letting memory grow. A failure stops that reader and is delivered.
fn read_ranges<T: Send + 'static>(
    graph: &Graph,
    cypher: String,
    ranges: Vec<(i64, i64)>,
    parallel: usize,
    process: Arc<dyn Fn(usize, Vec<Row>) -> Result<T> + Send + Sync>,
) -> tokio::sync::mpsc::Receiver<Result<T>> {
    let (tx, rx) = tokio::sync::mpsc::channel(parallel.max(1));
    let ranges = Arc::new(ranges);
    let next = Arc::new(AtomicUsize::new(0));
    for _ in 0..parallel.max(1) {
        let (graph, cypher, ranges, next, process, tx) = (
            graph.clone(),
            cypher.clone(),
            ranges.clone(),
            next.clone(),
            process.clone(),
            tx.clone(),
        );
        tokio::spawn(async move {
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(a, b)) = ranges.get(i) else { break };
                let result: Result<T> = async {
                    let mut stream = graph
                        .execute(query(&cypher).param("a", a).param("b", b))
                        .await?;
                    let mut rows = Vec::new();
                    while let Some(r) = stream.next().await? {
                        rows.push(r);
                    }
                    process(i, rows)
                }
                .await;
                let failed = result.is_err();
                if tx.send(result).await.is_err() || failed {
                    break;
                }
            }
        });
    }
    rx
}

/// Interned names → compact index.
#[derive(Default)]
struct Names(std::sync::Mutex<Vec<String>>);

impl Names {
    fn index(&self, name: &str) -> u16 {
        let mut v = self.0.lock().unwrap();
        if let Some(i) = v.iter().position(|n| n == name) {
            return i as u16;
        }
        v.push(name.to_string());
        (v.len() - 1) as u16
    }
    fn list(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}

/// One node batch read: first slot, properties, losses, fingerprint by label.
type NodeBatch = (u32, Vec<Props>, Losses, BTreeMap<String, Print>);

struct RelRead {
    id: i64,
    src: i64,
    dst: i64,
    kind: u16,
    props: Props,
}

/// The copy proper: the source `graph` into the empty directory `dest`. Returns
/// the fingerprint the source side computed, and the number of node and link
/// slots written.
async fn copy_into(
    graph: &Graph,
    tuning: &Tuning,
    before_properties: Option<&BeforeProperties>,
    dest: &Path,
) -> Result<(Prints, u64, u64, Losses)> {
    let mut losses = Losses::default();
    let mut expected = Prints::default();
    let (parallel, width, batch) = (
        tuning.parallel.max(1),
        tuning.width.max(1),
        tuning.batch.max(1),
    );

    // Phase A — identifiers and label of every node (no properties).
    let mut f = graph
        .execute(query("MATCH (n) RETURN max(id(n)) AS m"))
        .await?;
    let max_node: Option<i64> = f.next().await?.context("empty answer")?.get("m")?;
    let labels = Arc::new(Names::default());
    let l2 = labels.clone();
    let node_query = format!("{NODES_BY_RANGE} RETURN id(n) AS i, labels(n) AS l");
    let process = Arc::new(
        move |_i: usize, rows: Vec<Row>| -> Result<Vec<(i64, u16)>> {
            let mut v = Vec::with_capacity(rows.len());
            for row in &rows {
                let ls: Vec<String> = row.get("l")?;
                anyhow::ensure!(
                    ls.len() == 1,
                    "node {}: {} labels {:?} (the obrain bulk import is one label per node; the \
                 copy refuses it rather than dropping a label)",
                    row.get::<i64>("i")?,
                    ls.len(),
                    ls
                );
                v.push((row.get::<i64>("i")?, l2.index(&ls[0])));
            }
            Ok(v)
        },
    );
    let mut nodes: Vec<(i64, u16)> = Vec::new();
    if let Some(max) = max_node {
        let mut rx = read_ranges(graph, node_query, ranges(max, width), parallel, process);
        while let Some(r) = rx.recv().await {
            nodes.extend(r?);
        }
    }
    nodes.sort_unstable_by_key(|n| n.0);
    let node_ids: Arc<Vec<i64>> = Arc::new(nodes.iter().map(|n| n.0).collect());
    let label_names = Arc::new(labels.list());
    let node_label: Arc<Vec<u16>> = Arc::new(nodes.iter().map(|n| n.1).collect());
    drop(nodes);

    // Phase B — every relationship, with its properties.
    let mut f = graph
        .execute(query("MATCH ()-[r]->() RETURN max(id(r)) AS m"))
        .await?;
    let max_rel: Option<i64> = f.next().await?.context("empty answer")?.get("m")?;
    let types = Arc::new(Names::default());
    let t2 = types.clone();
    let process = Arc::new(
        move |_i: usize, rows: Vec<Row>| -> Result<(Vec<RelRead>, Losses)> {
            let (mut v, mut losses) = (Vec::with_capacity(rows.len()), Losses::default());
            for row in &rows {
                v.push(RelRead {
                    id: row.get("i")?,
                    src: row.get("s")?,
                    dst: row.get("d")?,
                    kind: t2.index(&row.get::<String>("t")?),
                    props: props_of(row, &mut losses)?,
                });
            }
            Ok((v, losses))
        },
    );
    let mut rels: Vec<RelRead> = Vec::new();
    if let Some(max) = max_rel {
        let mut rx = read_ranges(
            graph,
            RELS_BY_RANGE.to_string(),
            ranges(max, width * 4),
            parallel,
            process,
        );
        while let Some(r) = rx.recv().await {
            let (v, l) = r?;
            rels.extend(v);
            losses.add(&l);
        }
    }
    rels.sort_unstable_by_key(|r| r.id);
    let type_names = types.list();
    let slot = |id: i64| node_ids.binary_search(&id).ok().map(|k| k as u32 + 1);
    let (mut src, mut dst, mut kinds) = (Vec::new(), Vec::new(), Vec::new());
    let mut rel_props: Vec<Props> = Vec::with_capacity(rels.len());
    for r in rels {
        let (Some(s), Some(d)) = (slot(r.src), slot(r.dst)) else {
            anyhow::bail!(
                "relationship {}: an end ({} or {}) is not a node read in phase A (did the \
                 source change during the copy?)",
                r.id,
                r.src,
                r.dst
            );
        };
        let type_name = &type_names[r.kind as usize];
        expected
            .relationships
            .entry(type_name.clone())
            .or_default()
            .add(edge_print(type_name, s, d, &r.props));
        src.push(s);
        dst.push(d);
        kinds.push(r.kind);
        rel_props.push(r.props);
    }
    let (n_nodes, n_links) = (node_ids.len() as u64, src.len() as u64);

    // Phase C — structure, then the PO base marker is written at the end.
    let writer = Arc::new(TargetWriter::create(dest)?);
    if n_nodes > 0 {
        let w = writer.clone();
        let (ln, lk, tn) = (label_names.clone(), node_label.clone(), type_names.clone());
        let (s, d, k) = (src, dst, kinds);
        let (n, m) = tokio::task::spawn_blocking(move || {
            let label_refs: Vec<&str> = lk.iter().map(|i| ln[*i as usize].as_str()).collect();
            let type_refs: Vec<&str> = k.iter().map(|i| tn[*i as usize].as_str()).collect();
            w.structure(&label_refs, &s, &d, &type_refs)
        })
        .await??;
        anyhow::ensure!(
            n as u64 == n_nodes && m == n_links,
            "bulk_import_refs wrote {n} nodes and {m} links, expected {n_nodes} and {n_links}"
        );
    }

    if let Some(hook) = before_properties {
        hook().await;
    }

    // Phase D — node properties, read and written in batches.
    let nb = node_ids.len();
    let batches: Vec<(i64, i64)> = (0..nb.div_ceil(batch))
        .map(|k| (node_ids[k * batch], node_ids[((k + 1) * batch).min(nb) - 1]))
        .collect();
    let prop_query = format!("{NODES_BY_RANGE} RETURN id(n) AS i, properties(n) AS p");
    let (ids2, lk2, ln2) = (node_ids.clone(), node_label.clone(), label_names.clone());
    let process = Arc::new(move |k: usize, rows: Vec<Row>| -> Result<NodeBatch> {
        let first = k * batch;
        let size = batch.min(ids2.len() - first);
        let mut packet: Vec<Props> = vec![Props::new(); size];
        let mut seen = vec![false; size];
        let mut losses = Losses::default();
        for row in &rows {
            let id: i64 = row.get("i")?;
            let rank = ids2.binary_search(&id).map_err(|_| {
                anyhow::anyhow!("node {id} was not read in phase A (did the source change?)")
            })?;
            anyhow::ensure!(
                (first..first + size).contains(&rank),
                "node {id} answered outside its batch"
            );
            packet[rank - first] = props_of(row, &mut losses)?;
            seen[rank - first] = true;
        }
        // The source is read live, not frozen: a node deleted since phase A is
        // DETECTED here (every expected node must answer) and fails the copy by
        // name, instead of being written empty.
        if let Some(missing) = seen.iter().position(|s| !s) {
            anyhow::bail!(
                "node {} was read in phase A but did not answer in phase D: it was deleted \
                 from the source during the copy",
                ids2[first + missing]
            );
        }
        let mut prints: BTreeMap<String, Print> = BTreeMap::new();
        for (i, props) in packet.iter().enumerate() {
            let name = &ln2[lk2[first + i] as usize];
            prints
                .entry(name.clone())
                .or_default()
                .add(node_print(name, props));
        }
        Ok((first as u32 + 1, packet, losses, prints))
    });
    if nb > 0 {
        let mut rx = read_ranges(graph, prop_query, batches, parallel, process);
        while let Some(r) = rx.recv().await {
            let (first, packet, l, prints) = r?;
            losses.add(&l);
            for (name, p) in prints {
                expected.nodes.entry(name).or_default().add(p);
            }
            let w = writer.clone();
            tokio::task::spawn_blocking(move || w.node_props(first, &packet)).await??;
        }
    }

    // Phase E — relationship properties, by dense batches of slots.
    let mut first = 1usize;
    for chunk in rel_props.chunks(batch) {
        let w = writer.clone();
        let chunk = chunk.to_vec();
        let at = first as u32;
        tokio::task::spawn_blocking(move || w.edge_props(at, &chunk)).await??;
        first += batch;
    }

    // Phase F — durability, PO marker.
    let w = Arc::try_unwrap(writer)
        .map_err(|_| anyhow::anyhow!("internal: the target writer is still shared"))?;
    let dest = dest.to_path_buf();
    tokio::task::spawn_blocking(move || w.finish(&dest)).await??;
    Ok((expected, n_nodes, n_links, losses))
}

#[async_trait::async_trait]
impl Migrator for Neo4jToObrain {
    async fn source_counts(&self) -> Result<PerTypeCounts> {
        let graph = self.connect().await?;
        Ok(PerTypeCounts {
            nodes: counts(&graph, "MATCH (n) RETURN labels(n)[0] AS k, count(*) AS c").await?,
            relationships: counts(
                &graph,
                "MATCH ()-[r]->() RETURN type(r) AS k, count(*) AS c",
            )
            .await?,
        })
    }

    async fn copy(&self, dest: &Path) -> Result<()> {
        let graph = self.connect().await?;
        let (expected, nodes, links, losses) =
            copy_into(&graph, &self.tuning, self.before_properties.as_ref(), dest).await?;
        if let Some(why) = losses.refusal() {
            anyhow::bail!("the copy would lose information — {why}");
        }
        if losses != Losses::default() {
            tracing::info!(?losses, "storage migration: documented lossless narrowings");
        }
        if let Some(hook) = &self.after_write {
            hook(dest)?;
        }
        let dest = dest.to_path_buf();
        let read =
            tokio::task::spawn_blocking(move || target::read_back(&dest, nodes, links)).await??;
        if let Some(why) = expected.first_difference(&read) {
            anyhow::bail!("content fingerprint mismatch after the copy — {why}");
        }
        tracing::info!(
            nodes,
            links,
            "storage migration: content fingerprint verified"
        );
        Ok(())
    }

    async fn target_counts(&self, dest: &Path) -> Result<PerTypeCounts> {
        let dest = dest.to_path_buf();
        let (nodes, links) =
            tokio::task::spawn_blocking(move || target::engine_counts(&dest)).await??;
        Ok(PerTypeCounts {
            nodes: nodes.into_iter().map(|(k, n)| (k, n as usize)).collect(),
            relationships: links.into_iter().map(|(k, n)| (k, n as usize)).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_cover_zero_to_max_without_overlap() {
        assert_eq!(ranges(0, 10), vec![(0, 0)]);
        assert_eq!(ranges(24, 10), vec![(0, 9), (10, 19), (20, 24)]);
    }

    fn row(fields: &[&str], data: Vec<BoltType>) -> Row {
        use neo4rs::{BoltList, BoltString};
        Row::new(
            BoltList {
                value: fields
                    .iter()
                    .map(|f| BoltType::String(BoltString::new(f)))
                    .collect(),
            },
            BoltList { value: data },
        )
    }

    #[test]
    fn a_row_whose_properties_are_not_a_map_fails_instead_of_reading_empty() {
        let mut l = Losses::default();
        let not_a_map = row(&["p"], vec![BoltType::Integer(neo4rs::BoltInteger::new(1))]);
        let err = props_of(&not_a_map, &mut l).unwrap_err().to_string();
        assert!(err.contains("not a map"), "{err}");
        let absent = row(&["i"], vec![BoltType::Integer(neo4rs::BoltInteger::new(1))]);
        let err = props_of(&absent, &mut l).unwrap_err().to_string();
        assert!(err.contains("cannot be read"), "{err}");
        let empty = row(
            &["p"],
            vec![BoltType::Map(neo4rs::BoltMap {
                value: Default::default(),
            })],
        );
        assert!(props_of(&empty, &mut l).unwrap().is_empty());
    }

    #[test]
    fn keys_are_interned_once() {
        assert!(std::ptr::eq(intern("name"), intern("name")));
    }

    /// An unreachable source is a named error, not a hang: the startup then
    /// stays on the source with this cause.
    #[tokio::test]
    async fn an_unreachable_source_is_named() {
        let m = Neo4jToObrain::new("bolt://127.0.0.1:1", "", "");
        let err = m.source_counts().await.unwrap_err().to_string();
        assert!(
            err.contains("unreachable") || err.contains("no answer"),
            "{err}"
        );
    }
}
