//! Session -> thread attachment: the ONE place that decides which thread a
//! chat session belongs to, and which sessions belong to none.
//!
//! Three mechanisms attach a session; none is exhaustive, so membership is
//! their DE-DUPLICATED UNION, with the provenance of every link kept:
//! 1. `ChatSession.spawned_by` JSON written by the runner
//!    (`{"type":"runner","run_id","plan_id",...}`)        -> [`LinkVia::SpawnedByJson`];
//! 2. the run relation `(:ChatSession)-[:SPAWNED_BY_RUN]->(:PlanRun)`
//!    written by `ChatManager::create_session`            -> [`LinkVia::RunnerRun`];
//! 3. explicit association to a task / a plan (`ASSOCIATED_WITH`)
//!    -> [`LinkVia::TaskAssociation`] / [`LinkVia::PlanAssociation`];
//! 4. the origin relation `(:ChatSession)-[:SPAWNED_BY]->(:ChatSession)`: a
//!    child session inherits the thread of its parent (run carried by the
//!    relation, else the parent's own run / association)
//!    -> [`LinkVia::SpawnedByJson`] (same provenance family as 1.).
//!
//! Rules:
//! - a session linked by several mechanisms appears ONCE, with one link per
//!   mechanism (exact duplicates collapsed);
//! - a session is in ONE thread (a thread = a plan). When its links resolve to
//!   several plans, the plan of the strongest mechanism wins: an EXPLICIT
//!   association made by the user (plan, then task) beats what the runner
//!   wrote (run relation, then JSON / parent spawn); ties broken by plan id;
//! - a session with NO link is never dropped nor filed in a thread: it goes to
//!   `unattached`, rendered by [`unattached_sessions`] in the lane of its
//!   workspace, with its pending requests;
//! - a session whose links resolve to NO plan (deleted task, run without
//!   plan) is returned in `unresolved`, never lost either: the aggregator
//!   shows it like an unattached one;
//! - reading is grouped: ONE store call whatever the number of sessions
//!   ([`attach_sessions`]); [`attach`] is the pure core.
//!
//! After a RESUME of a run, old sessions keep the OLD `run_id` and new ones
//! carry the NEW one, with the same `plan_id`: the thread stays the plan's.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::Result;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::api::attention::{
    LinkVia, SessionLink, SessionState, UnattachedSession, WaitingRequest,
};
use crate::chat::attention::{derive_attention, SessionAttentionInput};
use crate::neo4j::models::{ChatEventRecord, ChatSessionNode, SessionLinkKind, SessionLinkRow};
use crate::neo4j::traits::GraphStore;

/// `spawned_by` types that denote a PLAN run (the `run_id` is a plan run's).
/// `protocol_runner` carries a PROTOCOL run id and must not be read as one.
pub const PLAN_RUN_SPAWN_TYPES: [&str; 3] = ["runner", "pipeline", "gate"];

/// Lane used when a free session carries neither a workspace nor a known
/// project: the aggregator must list it in `lanes[]`.
pub const UNASSIGNED_LANE: &str = "unassigned";

/// Plan-run context read from a `spawned_by` JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanRunSpawn {
    pub run_id: Option<Uuid>,
    pub plan_id: Option<Uuid>,
    pub task_id: Option<Uuid>,
}

/// Parse a `spawned_by` JSON; `Some` only for plan-run types carrying a
/// `run_id` and/or a `plan_id`.
pub fn parse_plan_run_spawn(spawned_by: &str) -> Option<PlanRunSpawn> {
    let v: serde_json::Value = serde_json::from_str(spawned_by).ok()?;
    let ty = v.get("type")?.as_str()?;
    if !PLAN_RUN_SPAWN_TYPES.contains(&ty) {
        return None;
    }
    let id = |k: &str| {
        v.get(k)
            .and_then(|x| x.as_str())
            .and_then(|s| s.parse::<Uuid>().ok())
    };
    let (run_id, plan_id) = (id("run_id"), id("plan_id"));
    if run_id.is_none() && plan_id.is_none() {
        return None;
    }
    Some(PlanRunSpawn {
        run_id,
        plan_id,
        task_id: id("task_id"),
    })
}

/// A session with every link that attaches it (never empty, sorted, unique).
#[derive(Debug, Clone, PartialEq)]
pub struct AttachedSession {
    pub session: ChatSessionNode,
    pub links: Vec<SessionLink>,
}

/// Result of the attachment of a set of sessions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Attachment {
    /// Thread (plan id) -> its sessions, oldest created first.
    pub by_plan: BTreeMap<Uuid, Vec<AttachedSession>>,
    /// Linked, but no link resolves to a plan.
    pub unresolved: Vec<AttachedSession>,
    /// No link at all.
    pub unattached: Vec<ChatSessionNode>,
}

fn via_rank(via: LinkVia) -> u8 {
    match via {
        LinkVia::PlanAssociation => 0,
        LinkVia::TaskAssociation => 1,
        LinkVia::RunnerRun => 2,
        LinkVia::SpawnedByJson => 3,
    }
}

/// Pure core: attach `sessions` given the stored link `rows`.
pub fn attach(sessions: &[ChatSessionNode], rows: &[SessionLinkRow]) -> Attachment {
    let mut by_session: HashMap<Uuid, Vec<&SessionLinkRow>> = HashMap::new();
    for r in rows {
        by_session.entry(r.session_id).or_default().push(r);
    }
    let mut seen: HashSet<Uuid> = HashSet::new();
    let mut out = Attachment::default();

    for session in sessions {
        // The same session handed twice is still ONE session.
        if !seen.insert(session.id) {
            continue;
        }
        // (link, plan this link resolves to)
        let mut found: Vec<(SessionLink, Option<Uuid>)> = Vec::new();

        if let Some(spawn) = session.spawned_by.as_deref().and_then(parse_plan_run_spawn) {
            found.push((
                SessionLink {
                    via: LinkVia::SpawnedByJson,
                    run_id: spawn.run_id,
                    task_id: None,
                    plan_id: spawn.plan_id,
                },
                spawn.plan_id,
            ));
        }
        for r in by_session.get(&session.id).into_iter().flatten() {
            let (via, link_run, link_plan) = match r.kind {
                SessionLinkKind::RunRelation => (LinkVia::RunnerRun, r.run_id, r.plan_id),
                SessionLinkKind::TaskAssociation => (LinkVia::TaskAssociation, None, None),
                SessionLinkKind::PlanAssociation => (LinkVia::PlanAssociation, None, r.plan_id),
                SessionLinkKind::SpawnedByRelation => (LinkVia::SpawnedByJson, r.run_id, r.plan_id),
            };
            let link_task = match r.kind {
                SessionLinkKind::PlanAssociation => None,
                _ => r.task_id,
            };
            found.push((
                SessionLink {
                    via,
                    run_id: link_run,
                    task_id: link_task,
                    plan_id: link_plan,
                },
                r.thread_plan_id.or(link_plan),
            ));
        }

        if found.is_empty() {
            out.unattached.push(session.clone());
            continue;
        }
        let plan = found
            .iter()
            .filter_map(|(l, p)| p.map(|p| (via_rank(l.via), p)))
            .min()
            .map(|(_, p)| p);
        let mut links: Vec<SessionLink> = found.into_iter().map(|(l, _)| l).collect();
        links.sort();
        links.dedup();
        let attached = AttachedSession {
            session: session.clone(),
            links,
        };
        match plan {
            Some(p) => out.by_plan.entry(p).or_default().push(attached),
            None => out.unresolved.push(attached),
        }
    }

    for v in out.by_plan.values_mut() {
        v.sort_by(|a, b| {
            a.session
                .created_at
                .cmp(&b.session.created_at)
                .then(a.session.id.cmp(&b.session.id))
        });
    }
    out.unresolved
        .sort_by_key(|a| (a.session.created_at, a.session.id));
    out.unattached.sort_by_key(|s| (s.created_at, s.id));
    out
}

/// Attach `sessions`: ONE grouped read of the stored links, then [`attach`].
pub async fn attach_sessions(
    graph: &dyn GraphStore,
    sessions: &[ChatSessionNode],
) -> Result<Attachment> {
    let ids: Vec<Uuid> = sessions.iter().map(|s| s.id).collect();
    let rows = graph.get_session_link_rows(&ids).await?;
    Ok(attach(sessions, &rows))
}

/// Lane (workspace slug) of a free session: its own workspace, else the
/// workspace of its project, else [`UNASSIGNED_LANE`].
pub fn lane_of(session: &ChatSessionNode, project_workspace: &HashMap<String, String>) -> String {
    session
        .workspace_slug
        .clone()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            session
                .project_slug
                .as_ref()
                .and_then(|p| project_workspace.get(p).cloned())
        })
        .unwrap_or_else(|| UNASSIGNED_LANE.to_string())
}

/// Render the unattached sessions as contract entries: lane, state, pending
/// requests (live AND dead ones, `thread_id` null, listed here only), `since`
/// = oldest pending request else last activity. `events` is the grouped read
/// of [`GraphStore::get_attention_events`]; `live` the ids with a running CLI.
pub fn unattached_sessions(
    unattached: &[ChatSessionNode],
    events: Vec<ChatEventRecord>,
    live: &HashSet<Uuid>,
    pending_permissions: &HashMap<Uuid, HashSet<String>>,
    project_workspace: &HashMap<String, String>,
    now: DateTime<Utc>,
) -> Vec<UnattachedSession> {
    let inputs: Vec<SessionAttentionInput> = unattached
        .iter()
        .map(|s| SessionAttentionInput {
            session_id: s.id,
            workspace: lane_of(s, project_workspace),
            thread_id: None,
            alive: live.contains(&s.id),
            pending_in_memory: pending_permissions.get(&s.id).cloned().unwrap_or_default(),
        })
        .collect();
    let derived = derive_attention(&inputs, events, now);
    let mut pending: HashMap<Uuid, Vec<WaitingRequest>> = HashMap::new();
    for w in derived.waiting {
        pending.entry(w.session_id).or_default().push(w);
    }
    for o in derived.orphans {
        pending
            .entry(o.session_id)
            .or_default()
            .push(WaitingRequest {
                request_id: o.request_id,
                kind: o.kind,
                session_id: o.session_id,
                thread_id: None,
                workspace: o.workspace,
                tool_name: o.tool_name,
                text: o.text,
                options: o.options,
                seq: o.seq,
                requested_at: o.requested_at,
                age_secs: o.age_secs,
            });
    }
    let mut out: Vec<UnattachedSession> = unattached
        .iter()
        .map(|s| {
            let mut p = pending.remove(&s.id).unwrap_or_default();
            p.sort_by(|a, b| a.requested_at.cmp(&b.requested_at).then(a.seq.cmp(&b.seq)));
            let since = p.first().map(|r| r.requested_at).unwrap_or(s.updated_at);
            UnattachedSession {
                id: s.id,
                workspace_slug: lane_of(s, project_workspace),
                title: s.title.clone().unwrap_or_default(),
                state: if live.contains(&s.id) {
                    SessionState::Live
                } else {
                    SessionState::Dead
                },
                pending: p,
                since,
                age_secs: u64::try_from((now - since).num_seconds()).unwrap_or(0),
            }
        })
        .collect();
    // Lane, then oldest first, then id (lane order is the aggregator's: it
    // re-sorts with `AttentionResponse::sort_by_age`).
    out.sort_by(|a, b| {
        a.workspace_slug
            .cmp(&b.workspace_slug)
            .then(b.age_secs.cmp(&a.age_secs))
            .then(a.id.cmp(&b.id))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::attention::RequestKind;
    use crate::neo4j::mock::MockGraphStore;
    use crate::test_helpers::test_chat_session;
    use std::sync::atomic::Ordering;

    fn sess(spawned_by: Option<String>) -> ChatSessionNode {
        let mut s = test_chat_session(None);
        s.spawned_by = spawned_by;
        s
    }
    fn runner_json(run: Uuid, plan: Uuid) -> String {
        serde_json::json!({"type":"runner","run_id":run.to_string(),"plan_id":plan.to_string()})
            .to_string()
    }
    fn run_row(s: &ChatSessionNode, run: Uuid, plan: Uuid) -> SessionLinkRow {
        SessionLinkRow {
            session_id: s.id,
            kind: SessionLinkKind::RunRelation,
            run_id: Some(run),
            task_id: None,
            plan_id: Some(plan),
            thread_plan_id: Some(plan),
        }
    }
    fn task_row(s: &ChatSessionNode, task: Uuid, plan: Option<Uuid>) -> SessionLinkRow {
        SessionLinkRow {
            session_id: s.id,
            kind: SessionLinkKind::TaskAssociation,
            run_id: None,
            task_id: Some(task),
            plan_id: None,
            thread_plan_id: plan,
        }
    }

    #[test]
    fn session_linked_by_two_mechanisms_appears_once_with_both_links() {
        let (run, plan) = (Uuid::new_v4(), Uuid::new_v4());
        let s = sess(Some(runner_json(run, plan)));
        let a = attach(std::slice::from_ref(&s), &[run_row(&s, run, plan)]);
        let v = &a.by_plan[&plan];
        assert_eq!(v.len(), 1, "one occurrence, not one per mechanism");
        let vias: Vec<_> = v[0].links.iter().map(|l| l.via).collect();
        assert_eq!(vias, vec![LinkVia::RunnerRun, LinkVia::SpawnedByJson]);
        assert!(a.unattached.is_empty() && a.unresolved.is_empty());
    }

    #[test]
    fn three_mechanisms_keep_one_link_each_and_exact_duplicates_collapse() {
        let (run, plan, task) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        let s = sess(Some(runner_json(run, plan)));
        let rows = vec![
            run_row(&s, run, plan),
            run_row(&s, run, plan), // exact duplicate row
            task_row(&s, task, Some(plan)),
        ];
        let a = attach(&[s.clone(), s.clone()], &rows); // session handed twice
        let v = &a.by_plan[&plan];
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].links.len(), 3);
        let t = v[0]
            .links
            .iter()
            .find(|l| l.via == LinkVia::TaskAssociation)
            .unwrap();
        assert_eq!((t.task_id, t.plan_id, t.run_id), (Some(task), None, None));
    }

    #[test]
    fn session_without_link_is_unattached_never_dropped_nor_in_a_thread() {
        let (run, plan) = (Uuid::new_v4(), Uuid::new_v4());
        let linked = sess(Some(runner_json(run, plan)));
        let free = sess(None);
        // protocol runs carry a protocol run id, not a plan run: not a link
        let proto = sess(Some(
            serde_json::json!({"type":"protocol_runner","run_id":run.to_string()}).to_string(),
        ));
        let a = attach(&[linked.clone(), free.clone(), proto.clone()], &[]);
        let ids: Vec<_> = a.unattached.iter().map(|s| s.id).collect();
        assert!(ids.contains(&free.id) && ids.contains(&proto.id));
        assert_eq!(a.unattached.len(), 2);
        assert_eq!(a.by_plan[&plan].len(), 1);
        assert_eq!(a.by_plan[&plan][0].session.id, linked.id);
    }

    #[test]
    fn linked_session_resolving_to_no_plan_is_kept_in_unresolved() {
        let s = sess(None);
        let a = attach(
            std::slice::from_ref(&s),
            &[task_row(&s, Uuid::new_v4(), None)],
        );
        assert!(a.by_plan.is_empty() && a.unattached.is_empty());
        assert_eq!(a.unresolved.len(), 1);
        assert_eq!(a.unresolved[0].links.len(), 1);
    }

    #[test]
    fn session_in_two_plans_goes_to_one_thread_chosen_by_strongest_mechanism() {
        let (run, plan_a, plan_b) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        let s = sess(None);
        let rows = vec![
            task_row(&s, Uuid::new_v4(), Some(plan_a)),
            run_row(&s, run, plan_b),
        ];
        let a = attach(std::slice::from_ref(&s), &rows);
        assert_eq!(a.by_plan.len(), 1, "a session is in one thread only");
        assert!(
            a.by_plan.contains_key(&plan_a),
            "an explicit task association beats the run relation"
        );
        assert_eq!(a.by_plan[&plan_a][0].links.len(), 2, "both links kept");
    }

    #[test]
    fn manual_plan_association_beats_run_relation_and_json() {
        let (run, plan_run, plan_manual) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        let s = sess(Some(runner_json(run, plan_run)));
        let rows = vec![
            run_row(&s, run, plan_run),
            SessionLinkRow {
                session_id: s.id,
                kind: SessionLinkKind::PlanAssociation,
                run_id: None,
                task_id: None,
                plan_id: Some(plan_manual),
                thread_plan_id: Some(plan_manual),
            },
        ];
        let a = attach(std::slice::from_ref(&s), &rows);
        assert_eq!(a.by_plan.len(), 1);
        assert!(a.by_plan.contains_key(&plan_manual), "manual plan wins");
        assert_eq!(a.by_plan[&plan_manual][0].links.len(), 3, "all links kept");
    }

    #[test]
    fn child_session_inherits_the_thread_through_the_spawned_by_relation() {
        let plan = Uuid::new_v4();
        // No run_id in its JSON (a chat-spawned child), no run relation.
        let child = sess(Some(r#"{"type":"chat","parent_session_id":"x"}"#.into()));
        let row = SessionLinkRow {
            session_id: child.id,
            kind: SessionLinkKind::SpawnedByRelation,
            run_id: None,
            task_id: None,
            plan_id: Some(plan),
            thread_plan_id: Some(plan),
        };
        let a = attach(std::slice::from_ref(&child), &[row]);
        assert!(a.unattached.is_empty() && a.unresolved.is_empty());
        assert_eq!(a.by_plan[&plan][0].session.id, child.id);
        assert_eq!(a.by_plan[&plan][0].links[0].via, LinkVia::SpawnedByJson);
    }

    #[tokio::test]
    async fn spawned_by_relation_is_read_with_the_grouped_read_via_the_store() {
        let g = MockGraphStore::new();
        let plan = Uuid::new_v4();
        let parent = sess(None);
        g.create_chat_session(&parent).await.unwrap();
        g.link_session_to_run(&parent.id.to_string(), Uuid::new_v4(), Some(plan), None)
            .await
            .unwrap();
        let mut children = Vec::new();
        for _ in 0..4 {
            let c = sess(None);
            g.create_chat_session(&c).await.unwrap();
            g.create_spawned_by_relation(
                &c.id.to_string(),
                &parent.id.to_string(),
                "chat",
                None,
                None,
            )
            .await
            .unwrap();
            children.push(c);
        }
        g.session_link_reads.store(0, Ordering::SeqCst);
        let a = attach_sessions(&g, &children).await.unwrap();
        assert!(a.unattached.is_empty(), "children are not orphaned");
        assert_eq!(a.by_plan[&plan].len(), 4);
        assert_eq!(g.session_link_reads.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn resumed_run_old_session_keeps_old_run_new_one_has_new_run_same_thread() {
        let (old_run, new_run, plan) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        let mut old = sess(Some(runner_json(old_run, plan)));
        old.created_at -= chrono::Duration::hours(1);
        let new = sess(Some(runner_json(new_run, plan)));
        let rows = vec![run_row(&old, old_run, plan), run_row(&new, new_run, plan)];
        let a = attach(&[new.clone(), old.clone()], &rows);
        assert_eq!(a.by_plan.len(), 1, "the thread stays the plan's");
        let v = &a.by_plan[&plan];
        assert_eq!(v[0].session.id, old.id, "oldest first");
        assert!(v[0].links.iter().all(|l| l.run_id == Some(old_run)));
        assert!(v[1].links.iter().all(|l| l.run_id == Some(new_run)));
    }

    #[tokio::test]
    async fn attach_sessions_reads_links_in_one_query_whatever_the_count() {
        for n in [1usize, 5, 40] {
            let g = MockGraphStore::new();
            let plan = Uuid::new_v4();
            let mut sessions = Vec::new();
            for _ in 0..n {
                let s = sess(None);
                g.create_chat_session(&s).await.unwrap();
                g.link_session_to_run(&s.id.to_string(), Uuid::new_v4(), Some(plan), None)
                    .await
                    .unwrap();
                sessions.push(s);
            }
            g.session_link_reads.store(0, Ordering::SeqCst);
            let a = attach_sessions(&g, &sessions).await.unwrap();
            assert_eq!(a.by_plan[&plan].len(), n);
            assert_eq!(
                g.session_link_reads.load(Ordering::SeqCst),
                1,
                "n={n}: constant number of reads"
            );
        }
    }

    // ---- unattached rendering ----

    fn ev(
        session: Uuid,
        seq: i64,
        ty: &str,
        data: serde_json::Value,
        at: DateTime<Utc>,
    ) -> ChatEventRecord {
        ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: session,
            seq,
            event_type: ty.to_string(),
            data: data.to_string(),
            created_at: at,
        }
    }

    #[test]
    fn unattached_session_carries_lane_state_and_pending_requests() {
        let now = Utc::now();
        let mut a = sess(None);
        a.workspace_slug = Some("alpha".into());
        let mut b = sess(None);
        b.project_slug = Some("proj-b".into());
        let c = sess(None); // no workspace, no project
        let perm = serde_json::json!({"type":"permission_request","id":"perm-1","tool":"Bash","input":{"command":"ls"}});
        let events = vec![ev(
            a.id,
            3,
            "permission_request",
            perm,
            now - chrono::Duration::seconds(90),
        )];
        let live: HashSet<Uuid> = [a.id].into_iter().collect();
        let map: HashMap<String, String> = [("proj-b".to_string(), "beta".to_string())].into();
        let out = unattached_sessions(
            &[a.clone(), b.clone(), c.clone()],
            events,
            &live,
            &HashMap::new(),
            &map,
            now,
        );
        assert_eq!(out.len(), 3, "none dropped");
        let ua = out.iter().find(|u| u.id == a.id).unwrap();
        assert_eq!(ua.workspace_slug, "alpha");
        assert_eq!(ua.state, SessionState::Live);
        assert_eq!(ua.pending.len(), 1);
        assert_eq!(ua.pending[0].thread_id, None);
        assert_eq!(ua.pending[0].text, "ls");
        assert_eq!(ua.age_secs, 90);
        let ub = out.iter().find(|u| u.id == b.id).unwrap();
        assert_eq!(
            (ub.workspace_slug.as_str(), ub.state),
            ("beta", SessionState::Dead)
        );
        assert!(ub.pending.is_empty());
        let uc = out.iter().find(|u| u.id == c.id).unwrap();
        assert_eq!(uc.workspace_slug, UNASSIGNED_LANE);
    }

    #[test]
    fn dead_unattached_session_keeps_its_orphan_request() {
        let now = Utc::now();
        let s = sess(None);
        let q = serde_json::json!({"type":"ask_user_question","id":"q1","tool_call_id":"t","input":{},"questions":[{"question":"Which?","options":[{"label":"A"}]}]});
        let events = vec![ev(
            s.id,
            1,
            "ask_user_question",
            q,
            now - chrono::Duration::seconds(10),
        )];
        let out = unattached_sessions(
            std::slice::from_ref(&s),
            events,
            &HashSet::new(),
            &HashMap::new(),
            &HashMap::new(),
            now,
        );
        assert_eq!(out[0].state, SessionState::Dead);
        assert_eq!(
            out[0].pending.len(),
            1,
            "the dead session's request is not lost"
        );
        assert_eq!(out[0].pending[0].kind, RequestKind::Question);
    }

    #[test]
    fn plan_run_spawn_parsing_is_strict_on_type() {
        let (r, p) = (Uuid::new_v4(), Uuid::new_v4());
        assert!(parse_plan_run_spawn(&runner_json(r, p)).is_some());
        assert!(parse_plan_run_spawn("not json").is_none());
        assert!(parse_plan_run_spawn(r#"{"type":"runner"}"#).is_none());
        assert!(parse_plan_run_spawn(r#"{"type":"conversation","session_id":"x"}"#).is_none());
    }
}
