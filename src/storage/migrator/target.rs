//! The obrain side of the copy: the bulk write, the PO base marker, and the
//! content fingerprint read back from disk.
//!
//! Nothing here knows Neo4j: the source side hands over plain data (labels,
//! edges, properties), which is what makes this part testable without a server.

use anyhow::{Context, Result};
use obrain_common::types::{EdgeId, NodeId, PropertyMap, Value};
use obrain_engine::ObrainDB;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::Path;

pub type Props = Vec<(&'static str, Value)>;

/// Content fingerprint of one label (nodes) or one type (relationships).
///
/// `hash` is the wrapping sum, over the entities, of a per-entity hash that
/// covers the label/type, the sorted `(key, value)` pairs and, for a
/// relationship, its two end slots: order-independent between entities, but an
/// entity's properties cannot move to another entity, nor a link to other ends,
/// without the hash changing.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Print {
    pub entities: u64,
    pub properties: u64,
    pub hash: u64,
}

impl Print {
    pub fn add(&mut self, o: Print) {
        self.entities += o.entities;
        self.properties += o.properties;
        self.hash = self.hash.wrapping_add(o.hash);
    }
}

/// The fingerprint of a whole base: nodes by label, relationships by type.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Prints {
    pub nodes: BTreeMap<String, Print>,
    pub relationships: BTreeMap<String, Print>,
}

impl Prints {
    /// The first difference, named, or `None` when the two agree.
    pub fn first_difference(&self, target: &Prints) -> Option<String> {
        for (kind, a, b) in [
            ("label", &self.nodes, &target.nodes),
            (
                "relationship type",
                &self.relationships,
                &target.relationships,
            ),
        ] {
            let names: std::collections::BTreeSet<&String> = a.keys().chain(b.keys()).collect();
            for name in names {
                let (x, y) = (
                    a.get(name).copied().unwrap_or_default(),
                    b.get(name).copied().unwrap_or_default(),
                );
                if x.entities != y.entities {
                    return Some(format!(
                        "{kind} '{name}': {} in the source, {} read back from the target",
                        x.entities, y.entities
                    ));
                }
                if x.properties != y.properties {
                    return Some(format!(
                        "{kind} '{name}': {} properties in the source, {} read back from the \
                         target",
                        x.properties, y.properties
                    ));
                }
                if x.hash != y.hash {
                    return Some(format!(
                        "{kind} '{name}': same counts, but the property fingerprint differs \
                         (a value, a key or a link end was altered by the copy)"
                    ));
                }
            }
        }
        None
    }
}

/// Feed a hasher the canonical bytes of a `Value`: a discriminant byte per
/// variant, then the full payload. `Debug` is deliberately NOT used: its
/// rendering abbreviates `Vector` (`[first; dims]`) and `Bytes`
/// (`[first byte; len]`), which would compare an embedding of 1536 f32 on its
/// first element alone (four bytes). Variable-length payloads are
/// length-prefixed so concatenation is unambiguous.
fn hash_value(h: &mut impl std::hash::Hasher, v: &Value) {
    match v {
        Value::Null => h.write_u8(0),
        Value::Bool(b) => {
            h.write_u8(1);
            h.write_u8(u8::from(*b));
        }
        Value::Int64(i) => {
            h.write_u8(2);
            h.write_i64(*i);
        }
        Value::Float64(f) => {
            h.write_u8(3);
            h.write_u64(f.to_bits());
        }
        Value::String(s) => {
            h.write_u8(4);
            h.write_u64(s.len() as u64);
            h.write(s.as_bytes());
        }
        Value::Bytes(b) => {
            h.write_u8(5);
            h.write_u64(b.len() as u64);
            h.write(b);
        }
        Value::Timestamp(t) => {
            h.write_u8(6);
            h.write_i64(t.as_nanos());
        }
        Value::Date(d) => {
            h.write_u8(7);
            h.write_i32(d.as_days());
        }
        Value::Time(t) => {
            h.write_u8(8);
            h.write_u64(t.as_nanos());
            match t.offset_seconds() {
                None => h.write_u8(0),
                Some(off) => {
                    h.write_u8(1);
                    h.write_i32(off);
                }
            }
        }
        Value::Duration(d) => {
            h.write_u8(9);
            h.write_i64(d.months());
            h.write_i64(d.days());
            h.write_i64(d.nanos());
        }
        Value::ZonedDatetime(z) => {
            h.write_u8(10);
            h.write_i64(z.as_timestamp().as_nanos());
            h.write_i32(z.offset_seconds());
        }
        Value::List(l) => {
            h.write_u8(11);
            h.write_u64(l.len() as u64);
            for e in l.iter() {
                hash_value(h, e);
            }
        }
        Value::Map(m) => {
            h.write_u8(12);
            h.write_u64(m.len() as u64);
            for (k, val) in m.iter() {
                h.write_u64(k.as_str().len() as u64);
                h.write(k.as_str().as_bytes());
                hash_value(h, val);
            }
        }
        Value::Vector(v) => {
            h.write_u8(13);
            h.write_u64(v.len() as u64);
            for x in v.iter() {
                h.write_u32(x.to_bits());
            }
        }
        Value::Path { nodes, edges } => {
            h.write_u8(14);
            h.write_u64(nodes.len() as u64);
            for n in nodes.iter() {
                hash_value(h, n);
            }
            h.write_u64(edges.len() as u64);
            for e in edges.iter() {
                hash_value(h, e);
            }
        }
    }
}

/// One entity's hash: the kind, the slot (nodes) or the ends (relationships)
/// and the properties, whatever the order they are given in. The slot binds a
/// node's properties to THAT node, so two nodes of the same label cannot
/// exchange their properties without the hash changing.
fn entity_hash<'a>(
    name: &str,
    ends: Option<(u64, u64)>,
    slot: Option<u64>,
    props: impl Iterator<Item = (&'a str, &'a Value)>,
) -> (u64, u64) {
    let (mut count, mut sum) = (0u64, 0u64);
    for (k, v) in props {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        k.hash(&mut h);
        hash_value(&mut h, v);
        sum = sum.wrapping_add(h.finish());
        count += 1;
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut h);
    ends.hash(&mut h);
    slot.hash(&mut h);
    sum.hash(&mut h);
    (count, h.finish())
}

/// Fingerprint of one node as the SOURCE side will write it. `slot` is the
/// 1-based node slot: it binds the node's properties to that node (see
/// [`entity_hash`]).
pub fn node_print(label: &str, slot: u64, props: &Props) -> Print {
    let (properties, hash) =
        entity_hash(label, None, Some(slot), props.iter().map(|(k, v)| (*k, v)));
    Print {
        entities: 1,
        properties,
        hash,
    }
}

/// Fingerprint of one relationship as the SOURCE side will write it (`src` and
/// `dst` are the 1-based node slots).
pub fn edge_print(edge_type: &str, src: u32, dst: u32, props: &Props) -> Print {
    let (properties, hash) = entity_hash(
        edge_type,
        Some((u64::from(src), u64::from(dst))),
        None,
        props.iter().map(|(k, v)| (*k, v)),
    );
    Print {
        entities: 1,
        properties,
        hash,
    }
}

fn read_props(map: &PropertyMap) -> Vec<(&str, &Value)> {
    map.iter().map(|(k, v)| (k.as_str(), v)).collect()
}

/// The new obrain base being written. Created empty, written in bulk (the
/// engine's zone-direct import: nodes and links, then node properties, then
/// link properties), checkpointed and closed by [`TargetWriter::finish`].
pub struct TargetWriter {
    db: std::sync::Arc<ObrainDB>,
}

impl TargetWriter {
    /// Open (create) the base in `dest`. `dest` must hold no engine file yet.
    pub fn create(dest: &Path) -> Result<Self> {
        let db = ObrainDB::open(dest)
            .map_err(|e| anyhow::anyhow!("creating the obrain target {}: {e}", dest.display()))?;
        Ok(Self {
            db: std::sync::Arc::new(db),
        })
    }

    /// Nodes (one label each, slot = rank + 1) and links (slots of their ends).
    pub fn structure(
        &self,
        labels: &[&str],
        src: &[u32],
        dst: &[u32],
        types: &[&str],
    ) -> Result<(u32, u64)> {
        self.db
            .bulk_import_refs(labels, src, dst, types)
            .map_err(|e| anyhow::anyhow!("bulk_import_refs: {e}"))
    }

    /// Properties of the nodes of slots `first..first + batch.len()`.
    pub fn node_props(&self, first: u32, batch: &[Props]) -> Result<u64> {
        self.db
            .bulk_import_props(first, batch)
            .map_err(|e| anyhow::anyhow!("bulk_import_props(slot {first}): {e}"))
    }

    /// Properties of the links of slots `first..first + batch.len()`.
    pub fn edge_props(&self, first: u32, batch: &[Props]) -> Result<u64> {
        self.db
            .bulk_import_edge_props(first, batch)
            .map_err(|e| anyhow::anyhow!("bulk_import_edge_props(slot {first}): {e}"))
    }

    /// Checkpoint and close the base. The PO base marker is NOT written here:
    /// it goes in only after the copy is read back and verified (see
    /// [`crate::obrain::write_base_marker`]), so a base written but not yet
    /// verified is never marked PO and the direct open refuses to serve it.
    pub fn finish(self) -> Result<()> {
        self.db
            .checkpoint()
            .map_err(|e| anyhow::anyhow!("checkpoint: {e}"))?;
        self.db.close().map_err(|e| anyhow::anyhow!("close: {e}"))?;
        drop(self.db);
        Ok(())
    }
}

/// Read the base back from DISK, read-only and independently of the engine's
/// live counters: every node slot and every link slot is visited.
///
/// `nodes` / `links` are the numbers of slots written (1..=n).
pub fn read_back(dest: &Path, nodes: u64, links: u64) -> Result<Prints> {
    // Reopened by segments: the walk reads every property page by mmap, which
    // stays resident while the base is open.
    let open = || {
        ObrainDB::open_read_only(dest)
            .map_err(|e| anyhow::anyhow!("reopening the target read-only: {e}"))
    };
    let mut db = open()?;
    let mut prints = Prints::default();
    for slot in 1..=nodes {
        if slot % 150_000 == 0 {
            drop(db);
            db = open()?;
        }
        let node = db
            .store()
            .get_node(NodeId(slot))
            .with_context(|| format!("node slot {slot} is empty in the target"))?;
        anyhow::ensure!(
            node.labels.len() == 1,
            "node slot {slot} carries {} labels in the target",
            node.labels.len()
        );
        let props = read_props(&node.properties);
        let (properties, hash) = entity_hash(
            node.labels[0].as_str(),
            None,
            Some(slot),
            props.iter().map(|(k, v)| (*k, *v)),
        );
        prints
            .nodes
            .entry(node.labels[0].to_string())
            .or_default()
            .add(Print {
                entities: 1,
                properties,
                hash,
            });
    }
    for slot in 1..=links {
        if slot % 300_000 == 0 {
            drop(db);
            db = open()?;
        }
        let edge = db
            .store()
            .get_edge(EdgeId(slot))
            .with_context(|| format!("link slot {slot} is empty in the target"))?;
        let props = read_props(&edge.properties);
        let (properties, hash) = entity_hash(
            edge.edge_type.as_str(),
            Some((edge.src.0, edge.dst.0)),
            None,
            props.iter().map(|(k, v)| (*k, *v)),
        );
        prints
            .relationships
            .entry(edge.edge_type.to_string())
            .or_default()
            .add(Print {
                entities: 1,
                properties,
                hash,
            });
    }
    Ok(prints)
}

/// Open the written base the way PO opens it ([`crate::obrain::ObrainGraphStore`])
/// and return what that open had to do: quarantine, repair, delete. Synchronous.
/// The store is dropped (the base closed) before returning.
pub fn opening_report(dest: &Path) -> Result<crate::obrain::RapportOuverture> {
    let store = crate::obrain::ObrainGraphStore::open(dest, crate::obrain::Creation::Interdite)
        .with_context(|| format!("opening the copy {} the way PO does", dest.display()))?;
    let report = store.rapport_d_ouverture().clone();
    drop(store);
    Ok(report)
}

/// The engine's own per-label and per-type counters, read-only (O(1): they do
/// not scan). Labels and types holding nothing are left out.
pub fn engine_counts(dest: &Path) -> Result<(BTreeMap<String, u64>, BTreeMap<String, u64>)> {
    let db = ObrainDB::open_read_only(dest)
        .map_err(|e| anyhow::anyhow!("opening the target read-only: {e}"))?;
    let nodes = db
        .label_counts()
        .into_iter()
        .filter(|(_, n)| *n > 0)
        .collect();
    let mut links = BTreeMap::new();
    for name in db.store().all_edge_types() {
        let n = db.edge_count_by_type(&name).unwrap_or(0);
        if n > 0 {
            links.insert(name, n);
        }
    }
    Ok((nodes, links))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn sample() -> (Vec<&'static str>, Vec<u32>, Vec<u32>, Vec<&'static str>) {
        (
            vec!["Project", "Project", "Task"],
            vec![1, 3],
            vec![3, 2],
            vec!["HAS_TASK", "HAS_TASK"],
        )
    }

    fn write(dest: &Path) -> Prints {
        let (labels, src, dst, types) = sample();
        let node_props: Vec<Props> = vec![
            vec![("name", Value::String("a".into())), ("n", Value::Int64(-1))],
            vec![("name", Value::String("b".into()))],
            vec![("t", Value::Float64(0.5))],
        ];
        let edge_props: Vec<Props> = vec![vec![("w", Value::Int64(-7))], vec![]];
        let w = TargetWriter::create(dest).unwrap();
        w.structure(&labels, &src, &dst, &types).unwrap();
        w.node_props(1, &node_props).unwrap();
        w.edge_props(1, &edge_props).unwrap();
        w.finish().unwrap();
        let mut p = Prints::default();
        for (i, (l, props)) in labels.iter().zip(&node_props).enumerate() {
            p.nodes
                .entry(l.to_string())
                .or_default()
                .add(node_print(l, (i + 1) as u64, props));
        }
        for (i, props) in edge_props.iter().enumerate() {
            p.relationships
                .entry(types[i].to_string())
                .or_default()
                .add(edge_print(types[i], src[i], dst[i], props));
        }
        p
    }

    #[test]
    fn what_the_source_side_computes_is_what_the_disk_reads_back() {
        let d = dir();
        let dest = d.path().join("base");
        let expected = write(&dest);
        let got = read_back(&dest, 3, 2).unwrap();
        assert_eq!(expected.first_difference(&got), None);
        assert_eq!(got.nodes["Project"].entities, 2);
        assert_eq!(got.nodes["Project"].properties, 3);
        assert_eq!(got.relationships["HAS_TASK"].properties, 1);
        // The copy itself does not mark the base PO: the marker goes in only
        // after the read-back and the fingerprint comparison (see `copy`).
        assert!(!dest.join(crate::obrain::MARQUEUR_PO).exists());
    }

    #[test]
    fn engine_counters_agree_with_the_walk() {
        let d = dir();
        let dest = d.path().join("base");
        write(&dest);
        let (nodes, links) = engine_counts(&dest).unwrap();
        assert_eq!(nodes["Project"], 2);
        assert_eq!(nodes["Task"], 1);
        assert_eq!(links["HAS_TASK"], 2);
    }

    #[test]
    fn a_changed_value_with_the_same_counts_is_named_by_its_label() {
        let d = dir();
        let dest = d.path().join("base");
        let mut expected = write(&dest);
        // The source claims a different value for one Project property.
        let altered = node_print(
            "Project",
            1,
            &vec![
                ("name", Value::String("a".into())),
                ("n", Value::Int64(255)),
            ],
        );
        let honest = node_print(
            "Project",
            1,
            &vec![("name", Value::String("a".into())), ("n", Value::Int64(-1))],
        );
        let p = expected.nodes.get_mut("Project").unwrap();
        p.hash = p.hash.wrapping_sub(honest.hash).wrapping_add(altered.hash);
        let got = read_back(&dest, 3, 2).unwrap();
        let why = expected.first_difference(&got).unwrap();
        assert!(
            why.contains("label 'Project'") && why.contains("fingerprint"),
            "{why}"
        );
    }

    #[test]
    fn a_missing_node_is_a_count_difference() {
        let d = dir();
        let dest = d.path().join("base");
        let mut expected = write(&dest);
        expected.nodes.get_mut("Task").unwrap().entities += 1;
        let got = read_back(&dest, 3, 2).unwrap();
        let why = expected.first_difference(&got).unwrap();
        assert!(
            why.contains("'Task'") && why.contains("2 in the source, 1"),
            "{why}"
        );
    }

    #[test]
    fn link_ends_are_part_of_the_fingerprint() {
        assert_ne!(
            edge_print("T", 1, 2, &vec![]).hash,
            edge_print("T", 2, 1, &vec![]).hash
        );
    }

    #[test]
    fn a_changed_embedding_element_past_the_first_changes_the_fingerprint() {
        // `Debug` abbreviates `Vector` to `[first; dims]`: with the canonical
        // serialization, changing element 2 of a 3-element embedding must change
        // the fingerprint (before, only the first element was compared).
        let emb = |c: f32| Value::Vector(std::sync::Arc::from(vec![1.0f32, 2.0, c]));
        let a = node_print("Node", 1, &vec![("embedding", emb(3.0))]);
        let b = node_print("Node", 1, &vec![("embedding", emb(9.0))]);
        assert_ne!(a.hash, b.hash);
        let mut left = Prints::default();
        left.nodes.entry("Node".into()).or_default().add(a);
        let mut right = Prints::default();
        right.nodes.entry("Node".into()).or_default().add(b);
        let why = left.first_difference(&right).unwrap();
        assert!(why.contains("fingerprint"), "{why}");
    }

    #[test]
    fn bytes_past_the_first_byte_change_the_fingerprint() {
        // `Debug` abbreviates `Bytes` to `[first; len]`: the canonical
        // serialization must see every byte, not just the first.
        let bytes = |x: u8| Value::Bytes(std::sync::Arc::from(vec![0u8, x, 9u8]));
        assert_ne!(
            node_print("T", 1, &vec![("b", bytes(1))]).hash,
            node_print("T", 1, &vec![("b", bytes(2))]).hash
        );
    }

    #[test]
    fn swapping_properties_between_two_nodes_of_the_same_label_is_a_mismatch() {
        // The slot binds a node's properties to that node: two nodes of the same
        // label with the same number of properties cannot exchange them without
        // the fingerprint changing.
        let a = node_print(
            "Node",
            1,
            &vec![("x", Value::Int64(1)), ("y", Value::Int64(2))],
        );
        let b = node_print(
            "Node",
            2,
            &vec![("x", Value::Int64(3)), ("y", Value::Int64(4))],
        );
        let mut before = Prints::default();
        before.nodes.entry("Node".into()).or_default().add(a);
        before.nodes.entry("Node".into()).or_default().add(b);
        let sa = node_print(
            "Node",
            1,
            &vec![("x", Value::Int64(3)), ("y", Value::Int64(4))],
        );
        let sb = node_print(
            "Node",
            2,
            &vec![("x", Value::Int64(1)), ("y", Value::Int64(2))],
        );
        let mut after = Prints::default();
        after.nodes.entry("Node".into()).or_default().add(sa);
        after.nodes.entry("Node".into()).or_default().add(sb);
        let why = before.first_difference(&after).unwrap();
        assert!(why.contains("fingerprint"), "{why}");
    }
}
// NOTE: no test covers the order "marker written only after verification".
// A test that calls `write()` / `finish` directly proves nothing about `copy`,
// which is the only place that orders the steps. Covering it needs a real
// source, so it is tracked as a follow-up rather than faked here.
