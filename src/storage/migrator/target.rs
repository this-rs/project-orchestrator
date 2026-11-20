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

/// One entity's hash: the kind, the ends (relationships) and the properties,
/// whatever the order they are given in.
fn entity_hash<'a>(
    name: &str,
    ends: Option<(u64, u64)>,
    props: impl Iterator<Item = (&'a str, &'a Value)>,
) -> (u64, u64) {
    let (mut count, mut sum) = (0u64, 0u64);
    for (k, v) in props {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        k.hash(&mut h);
        // `Debug` of a `Value` is deterministic (maps are ordered) and covers
        // the variant, so an Int64 never equals a Float64 of the same number.
        format!("{v:?}").hash(&mut h);
        sum = sum.wrapping_add(h.finish());
        count += 1;
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut h);
    ends.hash(&mut h);
    sum.hash(&mut h);
    (count, h.finish())
}

/// Fingerprint of one node as the SOURCE side will write it.
pub fn node_print(label: &str, props: &Props) -> Print {
    let (properties, hash) = entity_hash(label, None, props.iter().map(|(k, v)| (*k, v)));
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

    /// Checkpoint and close, then write the marker that makes the directory a
    /// PO base (`crate::obrain::MARQUEUR_PO`): without it PO refuses to open a
    /// base it did not create.
    pub fn finish(self, dest: &Path) -> Result<()> {
        self.db
            .checkpoint()
            .map_err(|e| anyhow::anyhow!("checkpoint: {e}"))?;
        self.db.close().map_err(|e| anyhow::anyhow!("close: {e}"))?;
        drop(self.db);
        std::fs::write(
            dest.join(crate::obrain::MARQUEUR_PO),
            crate::obrain::SCHEMA_PO,
        )
        .with_context(|| format!("writing the PO base marker in {}", dest.display()))
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
        w.finish(dest).unwrap();
        let mut p = Prints::default();
        for (l, props) in labels.iter().zip(&node_props) {
            p.nodes
                .entry(l.to_string())
                .or_default()
                .add(node_print(l, props));
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
        assert!(dest.join(crate::obrain::MARQUEUR_PO).is_file());
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
            &vec![
                ("name", Value::String("a".into())),
                ("n", Value::Int64(255)),
            ],
        );
        let honest = node_print(
            "Project",
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
}
