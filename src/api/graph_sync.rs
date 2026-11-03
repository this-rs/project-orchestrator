//! Incremental reload of the graph: `since` / `updated_after`, `include_deleted`, `synced_at`.
//!
//! Contract (project graph `GET /api/projects/:slug/graph` and workspace graph
//! `GET /api/workspaces/:slug/graph`):
//! - `since` (ISO 8601, alias `updated_after`): only the nodes changed after it, and the edges
//!   touching one of those nodes;
//! - `include_deleted` (`true`/`1`): also say what disappeared since `since`;
//! - the response carries `synced_at`, the cursor to send back as the next `since`.
//!
//! Without any of the three, the body is byte-for-byte what it was (the cursor then only travels
//! in the `X-Graph-Synced-At` header).
//!
//! What the data model allows (measured, not assumed):
//! - there is no single `updated_at`: a node is dated by what its model keeps — file
//!   `last_parsed`, note `max(created_at, last_confirmed_at)`, decision `decided_at`, chat
//!   session `created_at`, or an `updated_at` / `created_at` / `last_parsed` attribute. A node
//!   with none of those (skills, protocols…) is *untracked*: it is always sent, and counted in
//!   `untracked_nodes`;
//! - edges carry no date: an edge is sent when one of its ends is sent. An edge created later
//!   between two unchanged nodes is missed until a full reload;
//! - Neo4j hard-deletes (no tombstone). Deletions are therefore *derived*: the server keeps, per
//!   (scope, query shape), the identifiers of the last [`RING`] responses keyed by their
//!   `synced_at`, and diffs the one named by `since` against the current graph. It lives in
//!   process memory: after a restart, or if `since` is not a cursor this process handed out,
//!   `deletions_tracked` is `false` and the client must reload in full.

use super::graph_types::{
    build_project_graph_data_stamped, GraphEdge, GraphNode, GraphQuery, ProjectGraphResponse,
};
use crate::api::handlers::AppError;
use crate::neo4j::models::ProjectNode;
use crate::neo4j::traits::GraphStore;
use chrono::{DateTime, SecondsFormat, Utc};
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Mutex, OnceLock};
use uuid::Uuid;

/// Header that always carries the cursor, even when the body must stay unchanged.
pub const SYNCED_AT_HEADER: &str = "x-graph-synced-at";
/// Snapshots kept per (scope, query shape).
const RING: usize = 8;
/// Query shapes kept before the registry is reset (memory bound).
const MAX_KEYS: usize = 256;

/// Validated incremental parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SyncParams {
    pub since: Option<DateTime<Utc>>,
    pub include_deleted: bool,
}

impl SyncParams {
    /// Whether the client asked for anything incremental (the body then carries sync fields).
    pub fn is_active(&self) -> bool {
        self.since.is_some() || self.include_deleted
    }

    /// Parse and validate. A malformed date is a 400 that names the parameter and the value.
    pub fn from_query(q: &GraphQuery) -> Result<Self, AppError> {
        let since = parse_param("since", q.since.as_deref())?;
        let alias = parse_param("updated_after", q.updated_after.as_deref())?;
        let since = match (since, alias) {
            (Some(a), Some(b)) if a != b => {
                return Err(AppError::BadRequest(
                    "`since` and `updated_after` are aliases but carry different values"
                        .to_string(),
                ))
            }
            (a, b) => a.or(b),
        };
        Ok(Self {
            since,
            include_deleted: q.include_deleted.unwrap_or(false),
        })
    }
}

fn parse_param(name: &str, raw: Option<&str>) -> Result<Option<DateTime<Utc>>, AppError> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    // An unencoded `+02:00` reaches us as ` 02:00`: give the offset its sign back.
    let parsed = DateTime::parse_from_rfc3339(raw)
        .or_else(|_| DateTime::parse_from_rfc3339(&raw.replace(' ', "+")));
    match parsed {
        Ok(t) => Ok(Some(t.with_timezone(&Utc))),
        Err(_) => Err(AppError::BadRequest(format!(
            "Invalid `{name}`: '{raw}' is not an ISO 8601 / RFC 3339 date-time \
             (expected e.g. 2026-11-02T10:00:00Z)"
        ))),
    }
}

/// A fresh cursor, truncated to the microsecond it is printed with so that printing and parsing
/// it back is lossless.
pub fn new_cursor() -> DateTime<Utc> {
    let now = Utc::now();
    DateTime::from_timestamp_micros(now.timestamp_micros()).unwrap_or(now)
}

pub fn format_cursor(t: DateTime<Utc>) -> String {
    t.to_rfc3339_opts(SecondsFormat::Micros, true)
}

/// A node that disappeared.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeletedNode {
    pub id: String,
    pub deleted: bool,
}

/// An edge that disappeared.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeletedEdge {
    pub source: String,
    pub target: String,
    #[serde(rename = "type")]
    pub edge_type: String,
    pub deleted: bool,
}

/// Sync fields added to the body when the client asked for something incremental.
#[derive(Debug, Clone, Default, Serialize)]
pub struct GraphSyncInfo {
    /// Cursor: send it back as the next `since`.
    pub synced_at: String,
    /// The `since` this response was computed against, normalised.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub since: Option<String>,
    /// Nodes whose model keeps no date: always sent.
    pub untracked_nodes: usize,
    /// Only with `include_deleted`: `false` means the deletions could not be derived.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deletions_tracked: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deleted_nodes: Option<Vec<DeletedNode>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deleted_edges: Option<Vec<DeletedEdge>>,
}

fn edge_key(e: &GraphEdge) -> String {
    format!("{}\u{1}{}\u{1}{}", e.source, e.edge_type, e.target)
}

struct Snapshot {
    at: DateTime<Utc>,
    nodes: HashSet<String>,
    edges: HashSet<String>,
}

fn registry() -> &'static Mutex<HashMap<String, VecDeque<Snapshot>>> {
    static REG: OnceLock<Mutex<HashMap<String, VecDeque<Snapshot>>>> = OnceLock::new();
    REG.get_or_init(Default::default)
}

/// Date of a node whose model has no entry in `stamps`: an attribute, when there is one.
fn attribute_stamp(node: &GraphNode) -> Option<DateTime<Utc>> {
    let attrs = node.attributes.as_ref()?;
    ["updated_at", "last_parsed", "created_at"]
        .iter()
        .filter_map(|k| attrs.get(k)?.as_str())
        .filter_map(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|t| t.with_timezone(&Utc))
        .max()
}

/// Keep the nodes changed after `since` (plus the untracked ones) and the edges touching them.
/// Returns (nodes, edges, untracked count).
pub fn select_delta(
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    stamps: &HashMap<String, DateTime<Utc>>,
    since: DateTime<Utc>,
) -> (Vec<GraphNode>, Vec<GraphEdge>, usize) {
    let mut untracked = 0usize;
    let kept: Vec<GraphNode> = nodes
        .into_iter()
        .filter(
            |n| match stamps.get(&n.id).copied().or_else(|| attribute_stamp(n)) {
                Some(t) => t > since,
                None => {
                    untracked += 1;
                    true
                }
            },
        )
        .collect();
    let ids: HashSet<&str> = kept.iter().map(|n| n.id.as_str()).collect();
    let kept_edges = edges
        .into_iter()
        .filter(|e| ids.contains(e.source.as_str()) || ids.contains(e.target.as_str()))
        .collect();
    (kept, kept_edges, untracked)
}

/// Record the full graph under `cursor`, and return what vanished since the snapshot taken at
/// `since` (None when this process never handed that cursor out).
fn snapshot_and_diff(
    key: &str,
    cursor: DateTime<Utc>,
    since: Option<DateTime<Utc>>,
    nodes: HashSet<String>,
    edges: HashSet<String>,
) -> Option<(Vec<DeletedNode>, Vec<DeletedEdge>)> {
    let mut reg = registry().lock().unwrap_or_else(|p| p.into_inner());
    if reg.len() >= MAX_KEYS && !reg.contains_key(key) {
        reg.clear();
    }
    let ring = reg.entry(key.to_string()).or_default();
    let diff = since
        .and_then(|s| ring.iter().find(|snap| snap.at == s))
        .map(|prev| {
            let mut dn: Vec<DeletedNode> = prev
                .nodes
                .difference(&nodes)
                .map(|id| DeletedNode {
                    id: id.clone(),
                    deleted: true,
                })
                .collect();
            dn.sort_by(|a, b| a.id.cmp(&b.id));
            let mut de: Vec<DeletedEdge> = prev
                .edges
                .difference(&edges)
                .filter_map(|k| {
                    let mut p = k.split('\u{1}');
                    Some(DeletedEdge {
                        source: p.next()?.to_string(),
                        edge_type: p.next()?.to_string(),
                        target: p.next()?.to_string(),
                        deleted: true,
                    })
                })
                .collect();
            de.sort_by(|a, b| {
                (&a.source, &a.target, &a.edge_type).cmp(&(&b.source, &b.target, &b.edge_type))
            });
            (dn, de)
        });
    ring.push_back(Snapshot {
        at: cursor,
        nodes,
        edges,
    });
    while ring.len() > RING {
        ring.pop_front();
    }
    diff
}

/// Build the (possibly incremental) graph of one project.
///
/// `cursor` is taken by the caller *before* any read, so a change landing during the build is
/// sent again next time rather than lost.
#[allow(clippy::too_many_arguments)]
pub async fn project_graph_response(
    neo4j: &dyn GraphStore,
    project: &ProjectNode,
    layers: &[String],
    limit: usize,
    community: Option<i64>,
    live: &[Uuid],
    params: &SyncParams,
    cursor: DateTime<Utc>,
) -> Result<ProjectGraphResponse, AppError> {
    let mut stamps = HashMap::new();
    let (nodes, edges, communities, stats) = build_project_graph_data_stamped(
        neo4j,
        project,
        layers,
        limit,
        community,
        live,
        &mut stamps,
    )
    .await?;

    // Registered whatever the request, so that the cursor in the header is always usable.
    let mut sorted_layers = layers.to_vec();
    sorted_layers.sort();
    let key = format!(
        "{}|{}|{:?}|{}",
        project.id,
        sorted_layers.join(","),
        community,
        limit
    );
    let diff = snapshot_and_diff(
        &key,
        cursor,
        params.since,
        nodes.iter().map(|n| n.id.clone()).collect(),
        edges.iter().map(edge_key).collect(),
    );

    if !params.is_active() {
        return Ok(ProjectGraphResponse {
            nodes,
            edges,
            communities,
            stats,
            sync: None,
        });
    }

    let (nodes, edges, untracked) = match params.since {
        Some(since) => select_delta(nodes, edges, &stamps, since),
        None => (nodes, edges, 0),
    };
    let mut info = GraphSyncInfo {
        synced_at: format_cursor(cursor),
        since: params.since.map(format_cursor),
        untracked_nodes: untracked,
        ..Default::default()
    };
    if params.include_deleted {
        match (params.since, diff) {
            (_, Some((dn, de))) => {
                info.deletions_tracked = Some(true);
                info.deleted_nodes = Some(dn);
                info.deleted_edges = Some(de);
            }
            // Nothing to compare against: the first load deleted nothing.
            (None, None) => {
                info.deletions_tracked = Some(true);
                info.deleted_nodes = Some(Vec::new());
                info.deleted_edges = Some(Vec::new());
            }
            // A cursor this process did not hand out (restart, eviction, foreign date).
            (Some(_), None) => {
                info.deletions_tracked = Some(false);
                info.deleted_nodes = Some(Vec::new());
                info.deleted_edges = Some(Vec::new());
            }
        }
    }
    Ok(ProjectGraphResponse {
        nodes,
        edges,
        communities,
        stats,
        sync: Some(info),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::{NoteStore, ProjectStore};
    use crate::notes::NoteType;
    use crate::test_helpers::{test_note, test_project};
    use axum::response::IntoResponse;
    use std::time::Duration;

    const KNOWLEDGE: &[&str] = &["knowledge"];

    fn layers() -> Vec<String> {
        KNOWLEDGE.iter().map(|s| s.to_string()).collect()
    }

    fn query(v: serde_json::Value) -> GraphQuery {
        serde_json::from_value(v).unwrap()
    }

    async fn get(
        store: &MockGraphStore,
        project: &ProjectNode,
        params: SyncParams,
        cursor: DateTime<Utc>,
    ) -> ProjectGraphResponse {
        project_graph_response(store, project, &layers(), 5000, None, &[], &params, cursor)
            .await
            .unwrap()
    }

    async fn seeded(n: usize) -> (MockGraphStore, ProjectNode, Vec<uuid::Uuid>) {
        let store = MockGraphStore::new();
        let project = test_project();
        store.create_project(&project).await.unwrap();
        let mut ids = Vec::new();
        for i in 0..n {
            let note = test_note(project.id, NoteType::Observation, &format!("note {i}"));
            ids.push(note.id);
            store.create_note(&note).await.unwrap();
        }
        (store, project, ids)
    }

    fn note_ids(r: &ProjectGraphResponse) -> HashSet<String> {
        r.nodes
            .iter()
            .filter(|n| n.node_type == "note")
            .map(|n| n.id.clone())
            .collect()
    }

    #[tokio::test]
    async fn a_second_request_since_the_cursor_returns_only_what_changed() {
        let (store, project, old_ids) = seeded(4).await;

        // First load, no `since`: the whole graph. The cursor is what the header carries.
        let c1 = new_cursor();
        let first = get(&store, &project, SyncParams::default(), c1).await;
        assert_eq!(note_ids(&first).len(), 4);
        assert!(
            first.sync.is_none(),
            "a plain request carries no sync field"
        );

        tokio::time::sleep(Duration::from_millis(5)).await;
        let added_a = test_note(project.id, NoteType::Gotcha, "ajoutée A");
        let added_b = test_note(project.id, NoteType::Tip, "ajoutée B");
        store.create_note(&added_a).await.unwrap();
        store.create_note(&added_b).await.unwrap();

        // The client sends the cursor back as text, exactly as it received it.
        let params = SyncParams::from_query(&query(
            serde_json::json!({"layers": "knowledge", "since": format_cursor(c1)}),
        ))
        .unwrap();
        let second = get(&store, &project, params, new_cursor()).await;

        let expected: HashSet<String> = [added_a.id, added_b.id]
            .iter()
            .map(|i| i.to_string())
            .collect();
        assert_eq!(
            note_ids(&second),
            expected,
            "exactly the entities changed since"
        );
        assert!(
            second.nodes.len() < first.nodes.len(),
            "fewer than the first load"
        );
        assert!(old_ids
            .iter()
            .all(|i| !note_ids(&second).contains(&i.to_string())));
        let info = second.sync.as_ref().unwrap();
        assert_eq!(info.since.as_deref(), Some(format_cursor(c1).as_str()));
        assert!(
            info.synced_at.as_str() > format_cursor(c1).as_str(),
            "the cursor moves forward"
        );

        // `updated_after` is the same parameter.
        let alias = SyncParams::from_query(&query(
            serde_json::json!({"updated_after": format_cursor(c1)}),
        ))
        .unwrap();
        assert_eq!(alias.since, params.since);

        // Nothing changed since the last cursor: an empty delta.
        let c2 = parse_param("since", Some(&info.synced_at))
            .unwrap()
            .unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        let third = get(
            &store,
            &project,
            SyncParams {
                since: Some(c2),
                include_deleted: false,
            },
            new_cursor(),
        )
        .await;
        assert!(note_ids(&third).is_empty());
    }

    #[tokio::test]
    async fn include_deleted_lists_the_vanished_and_the_default_does_not() {
        let (store, project, ids) = seeded(3).await;

        let c1 = new_cursor();
        let first = get(
            &store,
            &project,
            SyncParams {
                since: None,
                include_deleted: true,
            },
            c1,
        )
        .await;
        let info = first.sync.as_ref().unwrap();
        assert_eq!(info.deletions_tracked, Some(true));
        assert_eq!(
            info.deleted_nodes.as_deref(),
            Some(&[][..]),
            "a first load deleted nothing"
        );

        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(store.delete_note(ids[1]).await.unwrap());

        let since = Some(c1);
        let with = get(
            &store,
            &project,
            SyncParams {
                since,
                include_deleted: true,
            },
            new_cursor(),
        )
        .await;
        let info = with.sync.as_ref().unwrap();
        assert_eq!(info.deletions_tracked, Some(true));
        assert_eq!(
            info.deleted_nodes.as_deref(),
            Some(
                &[DeletedNode {
                    id: ids[1].to_string(),
                    deleted: true
                }][..]
            ),
        );
        assert!(
            note_ids(&with).is_empty(),
            "a deletion is not a changed node"
        );

        // Default: same request, `include_deleted` absent → no deletion field at all.
        let without = get(
            &store,
            &project,
            SyncParams {
                since,
                include_deleted: false,
            },
            new_cursor(),
        )
        .await;
        let json = serde_json::to_value(&without).unwrap();
        assert!(json.get("deleted_nodes").is_none() && json.get("deletions_tracked").is_none());
        assert!(json.get("synced_at").is_some());

        // A cursor this process never handed out: say so instead of pretending nothing died.
        let foreign = get(
            &store,
            &project,
            SyncParams {
                since: Some(c1 - chrono::Duration::days(1)),
                include_deleted: true,
            },
            new_cursor(),
        )
        .await;
        assert_eq!(foreign.sync.unwrap().deletions_tracked, Some(false));
    }

    #[test]
    fn a_malformed_since_is_a_clear_400_not_a_500() {
        for bad in ["hier", "2026-13-45T00:00:00Z", "1730000000", "2026-11-02"] {
            let err =
                SyncParams::from_query(&query(serde_json::json!({ "since": bad }))).expect_err(bad);
            let AppError::BadRequest(msg) = &err else {
                panic!("expected BadRequest for {bad}")
            };
            assert!(msg.contains("since") && msg.contains(bad), "{msg}");
            assert_eq!(
                err.into_response().status(),
                axum::http::StatusCode::BAD_REQUEST
            );
        }
        let err = SyncParams::from_query(&query(serde_json::json!({ "updated_after": "x" })))
            .unwrap_err();
        assert!(matches!(&err, AppError::BadRequest(m) if m.contains("updated_after")));
        // Aliases that disagree are ambiguous.
        let err = SyncParams::from_query(&query(serde_json::json!({
            "since": "2026-11-02T10:00:00Z", "updated_after": "2026-11-02T11:00:00Z"
        })))
        .unwrap_err();
        assert!(matches!(err, AppError::BadRequest(_)));
        // An offset whose `+` was decoded as a space still parses.
        let ok = SyncParams::from_query(&query(
            serde_json::json!({ "since": "2026-11-02T10:00:00 02:00" }),
        ))
        .unwrap();
        assert_eq!(ok.since.unwrap().to_rfc3339(), "2026-11-02T08:00:00+00:00");
    }

    #[test]
    fn include_deleted_accepts_what_the_client_sends() {
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            let q = query(serde_json::json!({ "include_deleted": raw }));
            assert_eq!(
                SyncParams::from_query(&q).unwrap().include_deleted,
                want,
                "{raw}"
            );
        }
        assert!(serde_json::from_value::<GraphQuery>(
            serde_json::json!({"include_deleted": "peut-être"})
        )
        .is_err());

        // `null` and an absent field both mean "not asked for".
        assert!(
            !SyncParams::from_query(&query(serde_json::json!({ "include_deleted": null })))
                .unwrap()
                .include_deleted
        );
    }

    #[tokio::test]
    async fn without_since_the_body_keeps_its_old_shape() {
        let (store, project, _) = seeded(2).await;
        let r = get(&store, &project, SyncParams::default(), new_cursor()).await;
        let json = serde_json::to_value(&r).unwrap();
        let mut keys: Vec<&str> = json
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, ["communities", "edges", "nodes", "stats"]);
    }

    #[test]
    fn select_delta_uses_attribute_stamps_and_keeps_untracked_nodes() {
        let since = DateTime::parse_from_rfc3339("2026-11-02T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let mk = |id: &str, updated: Option<&str>| GraphNode {
            id: id.to_string(),
            node_type: "note".to_string(),
            label: id.to_string(),
            layer: "knowledge".to_string(),
            attributes: updated.map(|u| serde_json::json!({ "updated_at": u })),
        };
        let nodes = vec![
            mk("fresh", Some("2026-11-03T10:00:00Z")), // dated after `since`
            mk("stale", Some("2026-11-01T10:00:00Z")), // dated before `since`
            mk("untracked", None),                     // no date at all
        ];
        let edges = vec![GraphEdge {
            source: "fresh".to_string(),
            target: "stale".to_string(),
            edge_type: "SYNAPSE".to_string(),
            layer: "knowledge".to_string(),
            attributes: None,
        }];
        let (kept, kept_edges, untracked) = select_delta(nodes, edges, &HashMap::new(), since);
        assert_eq!(untracked, 1, "exactly one node carries no date");
        let ids: HashSet<&str> = kept.iter().map(|n| n.id.as_str()).collect();
        assert!(ids.contains("fresh"));
        assert!(ids.contains("untracked"));
        assert!(!ids.contains("stale"), "an old attribute stamp is dropped");
        assert_eq!(
            kept_edges.len(),
            1,
            "the edge touching a kept node survives"
        );
    }

    #[tokio::test]
    async fn snapshot_and_diff_rebuilds_a_vanished_edge_from_its_key() {
        let key = "test:edge-diff";
        let c1 = new_cursor();
        let mut nodes = HashSet::from(["a".to_string(), "b".to_string()]);
        let mut edges = HashSet::from(["a\u{1}SYNAPSE\u{1}b".to_string()]);
        assert!(
            snapshot_and_diff(key, c1, None, nodes.clone(), edges.clone()).is_none(),
            "the first load has nothing to diff against"
        );

        // `b` and its edge disappear.
        let c2 = new_cursor();
        nodes.remove("b");
        edges.clear();
        let diff = snapshot_and_diff(key, c2, Some(c1), nodes, edges).unwrap();
        assert_eq!(
            diff.0,
            vec![DeletedNode {
                id: "b".to_string(),
                deleted: true
            }]
        );
        assert_eq!(
            diff.1,
            vec![DeletedEdge {
                source: "a".to_string(),
                edge_type: "SYNAPSE".to_string(),
                target: "b".to_string(),
                deleted: true,
            }]
        );
    }
}
