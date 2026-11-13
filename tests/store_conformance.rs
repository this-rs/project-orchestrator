//! One conformance suite for every `GraphStore` backend (P0.1 of the storage
//! abstraction plan).
//!
//! The checks of [`conformance`] take a `&dyn GraphStore` and know nothing of
//! the backend. A [`StoreFactory`] opens a store; the [`conformance_suite!`]
//! macro instantiates every check once per factory, so the same tests run
//! against the in-memory [`MockGraphStore`] and against a real [`Neo4jClient`]
//! (and against an obrain store, for the domains it has ported: one more factory,
//! one more macro call — see `ObrainFactory` and `project_checks!`).
//! Only methods already in the trait (src/neo4j/traits.rs) are exercised. The
//! twelve subjects of P0.1 are fifteen checks: the CAS and the sorting each
//! have their race (concurrent creation, paging across a tie) as a check of
//! their own, so the deterministic part stays green on its own merit, and the
//! vector ranking has a second check for the recall of the project filter. Three
//! more stand on their own: the type a number keeps, the instant a timestamp
//! keeps, and the page boundaries a tie does not move.
//!
//! Neo4j is the reference: a check asserts what Neo4j does, including where that
//! is lossy (an absent description reads back as `""`), and the mock converges
//! towards it, never the reverse (sibling task P0.2). A divergence found here is
//! NOT hidden and does NOT bend Neo4j: it is listed in [`EXPECTED_DIVERGENCES`]
//! with its reason and backend, the check runs on that backend anyway, its
//! failure is printed as `EXPECTED DIVERGENCE` and does not fail the suite, and
//! the day the backend conforms a deterministic entry turns red until it is
//! removed (a ratchet, not an `#[ignore]`). Three entries are Neo4j's own: gaps
//! in the Neo4j layer that this suite found, where the mock is right.
//!
//! Neo4j: `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`, default
//! `bolt://localhost:17687` (never the 7687 of a developer's live instance).
//! Without a reachable Neo4j the `neo4j::*` tests print a notice and pass;
//! `HARNESS_NEO4J_REQUIRED=1` (the CI) turns that into a failure. Every check
//! works on fresh UUIDs and deletes what it created, so the suite can share a
//! throwaway instance with the other suites and run in parallel. Alerts have no
//! delete in the trait and are left behind (scoped to deleted projects).
//!
//! ```text
//! NEO4J_URI=bolt://127.0.0.1:17687 NEO4J_PASSWORD=... cargo test --test store_conformance
//! ```
mod common;

#[allow(unused_imports)]
use project_orchestrator::neo4j::traits::*;

use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::FutureExt;
use project_orchestrator::neo4j::mock::MockGraphStore;
use project_orchestrator::neo4j::{GraphStore, Neo4jClient};
use std::sync::Arc;

/// The backend-agnostic checks. Each one is `async fn check_x(store: &dyn GraphStore)`,
/// panics on the first violated expectation, and cleans up after itself.
pub mod conformance {
    use chrono::{DateTime, Duration, TimeZone, Utc};
    use futures::future::join_all;
    use project_orchestrator::ingress::{ExternalEvent, IngressOutcome};
    use project_orchestrator::neo4j::file_uid::PathPolicy;
    use project_orchestrator::neo4j::models::{
        AlertNode, AlertSeverity, ComponentType, ComponentUpdate, DeploymentNode, DeploymentStatus,
        DerivedComponentWrite, EnvironmentKind, EnvironmentNode, FileNode, MilestoneNode,
        MilestoneStatus, PlanNode, PlanStatus, ProjectNode, ProjectProfile, ReleaseNode,
        ReleaseStatus, ResourceNode, ResourceType, StepNode, StepStatus, TaskNode, TaskStatus,
        WorkspaceMilestoneNode, WorkspaceNode,
    };
    use project_orchestrator::neo4j::GraphStore;
    use project_orchestrator::notes::{
        EntityType, Note, NoteFilters, NoteImportance, NoteScope, NoteStatus, NoteType,
    };
    use project_orchestrator::plan::models::UpdateTaskRequest;
    use project_orchestrator::protocol::{Protocol, ProtocolRun, RunStatus};
    use std::collections::HashSet;
    use uuid::Uuid;

    // ------------------------------------------------------------------
    // Fixtures: fresh ids, distinctive names, every field set explicitly.
    // ------------------------------------------------------------------

    pub fn a_project(label: &str) -> ProjectNode {
        let id = Uuid::new_v4();
        ProjectNode {
            id,
            name: format!("conformance {label} {id}"),
            slug: format!("conformance-{label}-{id}"),
            root_path: String::new(),
            description: None,
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: false,
            profile: Default::default(),
        }
    }

    pub fn a_plan(
        project_id: Uuid,
        title: &str,
        priority: i32,
        created_at: DateTime<Utc>,
    ) -> PlanNode {
        PlanNode {
            id: Uuid::new_v4(),
            title: title.to_string(),
            description: "conformance plan".to_string(),
            status: PlanStatus::Draft,
            created_at,
            created_by: "conformance".to_string(),
            priority,
            project_id: Some(project_id),
            execution_context: None,
            persona: None,
        }
    }

    pub fn a_task(title: Option<&str>) -> TaskNode {
        TaskNode {
            id: Uuid::new_v4(),
            title: title.map(str::to_string),
            description: "conformance task".to_string(),
            status: TaskStatus::Pending,
            assigned_to: None,
            interrupted_at: None,
            interrupted_reason: None,
            priority: Some(1),
            tags: vec![],
            acceptance_criteria: vec![],
            affected_files: vec![],
            estimated_complexity: None,
            due_date: None,
            estimated_minutes: None,
            actual_complexity: None,
            created_at: Utc::now(),
            updated_at: None,
            started_at: None,
            completed_at: None,
            frustration_score: 0.0,
            execution_context: None,
            persona: None,
            prompt_cache: None,
        }
    }

    pub fn a_file(path: &str, project_id: Option<Uuid>, hash: &str) -> FileNode {
        FileNode {
            path: path.to_string(),
            language: "rust".to_string(),
            hash: hash.to_string(),
            last_parsed: Utc::now(),
            project_id,
        }
    }

    pub fn a_note(project_id: Option<Uuid>, content: &str) -> Note {
        Note::new(
            project_id,
            NoteType::Observation,
            content.to_string(),
            "conformance".to_string(),
        )
    }

    pub fn a_run(
        protocol: &Protocol,
        parent: Option<&ProtocolRun>,
        started_at: DateTime<Utc>,
    ) -> ProtocolRun {
        let mut run = ProtocolRun::new(protocol.id, protocol.entry_state, "start");
        run.parent_run_id = parent.map(|p| p.id);
        run.depth = parent.map(|p| p.depth + 1).unwrap_or(0);
        run.started_at = started_at;
        run
    }

    fn ids<T>(items: &[T], id: impl Fn(&T) -> Uuid) -> Vec<Uuid> {
        items.iter().map(id).collect()
    }

    fn id_set<T>(items: &[T], id: impl Fn(&T) -> Uuid) -> HashSet<Uuid> {
        items.iter().map(id).collect()
    }

    fn at(seconds: i64) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap() + Duration::seconds(seconds)
    }

    // ------------------------------------------------------------------
    // 1. Uniqueness: File.path is global, Alert.dedup_key collapses.
    // ------------------------------------------------------------------

    /// A path names one `File` whatever project writes it (the last writer wins),
    /// and two alerts with one `dedup_key` are one node whose `occurrence_count`
    /// counts both and whose message is the latest one.
    pub async fn check_uniqueness(store: &dyn GraphStore) {
        let a = a_project("unique-a");
        let b = a_project("unique-b");
        store.create_project(&a).await.unwrap();
        store.create_project(&b).await.unwrap();

        let path = format!("conformance/{}/unique.rs", Uuid::new_v4());
        store
            .upsert_file(&a_file(&path, Some(a.id), "h1"))
            .await
            .unwrap();
        store
            .upsert_file(&a_file(&path, Some(b.id), "h2"))
            .await
            .unwrap();
        let got = store
            .get_file(&path)
            .await
            .unwrap()
            .expect("the file exists");
        assert_eq!(got.hash, "h2", "last writer wins on the same path");
        assert_eq!(got.project_id, Some(b.id), "the path now belongs to b");
        let hits = store.find_files_by_path_suffix(&path, 10).await.unwrap();
        assert_eq!(hits.len(), 1, "one node per path: {hits:?}");
        assert_eq!(
            store.list_project_files(b.id).await.unwrap().len(),
            1,
            "b contains the file"
        );

        let first = AlertNode::new_for_subject(
            "conformance".to_string(),
            AlertSeverity::Warning,
            "first wording".to_string(),
            Some(a.id),
            "the-condition",
        );
        let mut second = AlertNode::new_for_subject(
            "conformance".to_string(),
            AlertSeverity::Warning,
            "second wording".to_string(),
            Some(a.id),
            "the-condition",
        );
        second.id = Uuid::new_v4();
        assert_eq!(first.dedup_key, second.dedup_key);
        store.create_alert(&first).await.unwrap();
        store.create_alert(&second).await.unwrap();
        let (alerts, total) = store.list_alerts(Some(a.id), None, 10, 0).await.unwrap();
        assert_eq!(total, 1, "one dedup_key, one alert: {alerts:?}");
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].id, first.id, "the first id is the node's id");
        assert_eq!(alerts[0].occurrence_count, 2, "both observations counted");
        assert_eq!(
            alerts[0].message, "second wording",
            "the message is the latest"
        );
        let kept = store
            .get_alert(first.id)
            .await
            .unwrap()
            .expect("first id kept");
        assert_eq!(kept.occurrence_count, 2);
        assert!(
            store.get_alert(second.id).await.unwrap().is_none(),
            "the second id never became a node"
        );

        store.delete_file(&path).await.unwrap();
        store.delete_project(a.id, &a.name).await.unwrap();
        store.delete_project(b.id, &b.name).await.unwrap();
    }

    // ------------------------------------------------------------------
    // 2. MERGE idempotence.
    // ------------------------------------------------------------------

    /// Repeating a MERGE-based write changes nothing: files (single and batch),
    /// the project-file link, a task dependency, a protocol upsert.
    pub async fn check_merge_idempotence(store: &dyn GraphStore) {
        let p = a_project("merge");
        store.create_project(&p).await.unwrap();
        let f1 = format!("conformance/{}/one.rs", p.id);
        let f2 = format!("conformance/{}/two.rs", p.id);

        store
            .upsert_file(&a_file(&f1, Some(p.id), "h"))
            .await
            .unwrap();
        store
            .upsert_file(&a_file(&f1, Some(p.id), "h"))
            .await
            .unwrap();
        assert_eq!(store.count_project_files(p.id).await.unwrap(), 1);

        let batch = vec![
            a_file(&f1, Some(p.id), "h"),
            a_file(&f2, Some(p.id), "h"),
            a_file(&f2, Some(p.id), "h"),
        ];
        store.batch_upsert_files(&batch).await.unwrap();
        store.batch_upsert_files(&batch).await.unwrap();
        assert_eq!(
            store.count_project_files(p.id).await.unwrap(),
            2,
            "a batch with a duplicate, applied twice, is two files"
        );
        store.link_file_to_project(&f1, p.id).await.unwrap();
        store.link_file_to_project(&f1, p.id).await.unwrap();
        assert_eq!(store.count_project_files(p.id).await.unwrap(), 2);
        let mut paths = store.get_project_file_paths(p.id).await.unwrap();
        paths.sort();
        assert_eq!(paths, vec![f1.clone(), f2.clone()]);
        assert_eq!(store.list_project_files(p.id).await.unwrap().len(), 2);

        let plan = a_plan(p.id, "merge plan", 1, Utc::now());
        store.create_plan(&plan).await.unwrap();
        let t1 = a_task(Some("t1"));
        let t2 = a_task(Some("t2"));
        store.create_task(plan.id, &t1).await.unwrap();
        store.create_task(plan.id, &t2).await.unwrap();
        store.add_task_dependency(t1.id, t2.id).await.unwrap();
        store.add_task_dependency(t1.id, t2.id).await.unwrap();
        let deps = store.get_task_dependencies(t1.id).await.unwrap();
        assert_eq!(ids(&deps, |t| t.id), vec![t2.id], "one DEPENDS_ON edge");
        let blockers = store.get_task_blockers(t1.id).await.unwrap();
        assert_eq!(ids(&blockers, |t| t.id), vec![t2.id]);

        let proto = Protocol::new(Some(p.id), "merge protocol", Uuid::new_v4());
        store.upsert_protocol(&proto).await.unwrap();
        store.upsert_protocol(&proto).await.unwrap();
        // A project's listing is its OWN protocols plus the instance's transverse ones
        // (`project_id IS NULL`), and the instance seeds some of those: `total` counts
        // both, so it is not this project's count. The check is about the upsert, so it
        // is read on the rows of this project (the just-upserted one is the freshest, so
        // the first page always holds it).
        let (listed, _total) = store
            .list_protocols(Some(p.id), None, 100, 0)
            .await
            .unwrap();
        let mine: Vec<&Protocol> = listed
            .iter()
            .filter(|listed| listed.project_id == Some(p.id))
            .collect();
        assert_eq!(mine.len(), 1, "one protocol after two upserts");
        assert_eq!(mine[0].id, proto.id, "the row left is the one upserted");

        assert!(store.delete_protocol(proto.id).await.unwrap());
        store.delete_project(p.id, &p.name).await.unwrap();
        store.delete_file(&f1).await.unwrap();
        store.delete_file(&f2).await.unwrap();
    }

    /// The scope rule of the transverse library, on protocols — the one skills follow:
    /// a project's listing holds that project's protocols PLUS the instance's transverse
    /// ones (`project_id IS NULL`), and never another project's; a listing with no
    /// project is the transverse set alone.
    ///
    /// This is the rule [`check_merge_idempotence`] used to trip over: `total` counts the
    /// transverse protocols, and an instance that seeded some reads more than the
    /// project's own count. Nothing pinned the rule until this check.
    pub async fn check_protocol_scope_rule(store: &dyn GraphStore) {
        let a = a_project("scope-a");
        let b = a_project("scope-b");
        store.create_project(&a).await.unwrap();
        store.create_project(&b).await.unwrap();

        let shared = Protocol::new(None, "scope transverse", Uuid::new_v4());
        let own_a = Protocol::new(Some(a.id), "scope own a", Uuid::new_v4());
        let own_b = Protocol::new(Some(b.id), "scope own b", Uuid::new_v4());
        for protocol in [&shared, &own_a, &own_b] {
            store.upsert_protocol(protocol).await.unwrap();
        }

        let (listed_a, _) = store
            .list_protocols(Some(a.id), None, 100, 0)
            .await
            .unwrap();
        let seen_a = ids(&listed_a, |p| p.id);
        assert!(
            seen_a.contains(&own_a.id),
            "a project sees its own protocol"
        );
        assert!(
            seen_a.contains(&shared.id),
            "a project sees the transverse protocol"
        );
        assert!(
            !seen_a.contains(&own_b.id),
            "a project never sees another project's protocol"
        );

        let (listed_none, _) = store.list_protocols(None, None, 100, 0).await.unwrap();
        let seen_none = ids(&listed_none, |p| p.id);
        assert!(
            seen_none.contains(&shared.id),
            "the transverse scope lists the transverse protocol"
        );
        assert!(
            !seen_none.contains(&own_a.id),
            "the transverse scope lists no project's protocol"
        );

        for protocol in [&shared, &own_a, &own_b] {
            assert!(store.delete_protocol(protocol.id).await.unwrap());
        }
        store.delete_project(a.id, &a.name).await.unwrap();
        store.delete_project(b.id, &b.name).await.unwrap();
    }

    // ------------------------------------------------------------------
    // 3. Compare-and-swap on protocol runs, under concurrency.
    // ------------------------------------------------------------------

    /// One running root run per protocol (children exempt, a completed run frees
    /// the slot) and `update_protocol_run` refuses a stale version — the
    /// compare-and-swap, one caller at a time. The race is
    /// [`check_concurrent_create_protocol_run`].
    pub async fn check_cas_protocol_run(store: &dyn GraphStore) {
        let p = a_project("cas");
        store.create_project(&p).await.unwrap();
        let proto = Protocol::new(Some(p.id), "cas protocol", Uuid::new_v4());
        store.upsert_protocol(&proto).await.unwrap();

        let root = a_run(&proto, None, at(0));
        store.create_protocol_run(&root).await.unwrap();
        let second = a_run(&proto, None, at(1));
        let refused = store.create_protocol_run(&second).await;
        assert!(refused.is_err(), "a second running root run is refused");
        assert!(
            store.get_protocol_run(second.id).await.unwrap().is_none(),
            "the refused run left nothing behind"
        );
        let child = a_run(&proto, Some(&root), at(2));
        store.create_protocol_run(&child).await.unwrap();

        let mut done = store.get_protocol_run(root.id).await.unwrap().unwrap();
        assert_eq!(done.version, 0);
        done.status = RunStatus::Completed;
        done.completed_at = Some(at(3));
        store.update_protocol_run(&mut done).await.unwrap();
        assert_eq!(done.version, 1, "the version is bumped in place");
        let third = a_run(&proto, None, at(4));
        store
            .create_protocol_run(&third)
            .await
            .expect("the slot is free once the root completed");

        let mut fresh = store.get_protocol_run(third.id).await.unwrap().unwrap();
        let mut stale = fresh.clone();
        fresh.status = RunStatus::Failed;
        store.update_protocol_run(&mut fresh).await.unwrap();
        assert_eq!(fresh.version, 1);
        stale.status = RunStatus::Cancelled;
        assert!(
            store.update_protocol_run(&mut stale).await.is_err(),
            "a stale version loses the CAS"
        );
        let stored = store.get_protocol_run(third.id).await.unwrap().unwrap();
        assert_eq!(
            stored.status,
            RunStatus::Failed,
            "the first writer's update stands"
        );
        assert_eq!(stored.version, 1);

        assert!(store.delete_protocol(proto.id).await.unwrap());
        for r in [&root, &child, &third] {
            store.delete_protocol_run(r.id).await.unwrap();
        }
        store.delete_project(p.id, &p.name).await.unwrap();
    }

    /// The same compare-and-swap under contention: of 8 root runs created at
    /// once for one protocol, exactly one wins and the store holds one running
    /// run. (Neo4j's `WHERE NOT EXISTS { running root } CREATE ...` reads before
    /// it locks, so concurrent transactions all pass the check: see
    /// `EXPECTED_DIVERGENCES`.)
    pub async fn check_concurrent_create_protocol_run(store: &dyn GraphStore) {
        let p = a_project("cas-race");
        store.create_project(&p).await.unwrap();
        let racing = Protocol::new(Some(p.id), "cas race", Uuid::new_v4());
        store.upsert_protocol(&racing).await.unwrap();
        let runs: Vec<ProtocolRun> = (0..8).map(|i| a_run(&racing, None, at(10 + i))).collect();
        let outcomes = join_all(runs.iter().map(|r| store.create_protocol_run(r))).await;
        let created = outcomes.iter().filter(|o| o.is_ok()).count();
        let (_, running) = store
            .list_protocol_runs(racing.id, Some(RunStatus::Running), 100, 0)
            .await
            .unwrap();

        assert!(store.delete_protocol(racing.id).await.unwrap());
        for r in &runs {
            store.delete_protocol_run(r.id).await.unwrap();
        }
        store.delete_project(p.id, &p.name).await.unwrap();

        assert_eq!(
            created, 1,
            "exactly one of 8 concurrent root creations wins: {outcomes:?}"
        );
        assert_eq!(running, 1, "and the store holds exactly one running run");
    }

    // ------------------------------------------------------------------
    // 3b. External events: journal and delivery claim (ingress, PR 1/5).
    // ------------------------------------------------------------------

    fn an_external_event(trigger_id: Uuid, outcome: IngressOutcome, seconds: i64) -> ExternalEvent {
        ExternalEvent {
            id: Uuid::new_v4(),
            trigger_id,
            source: "github".to_string(),
            outcome,
            reason: None,
            delivery_id: Some(format!("delivery-{seconds}")),
            event_type: Some("push".to_string()),
            action: None,
            body_bytes: 42,
            received_at: at(seconds),
        }
    }

    /// The journal of one trigger lists newest first by `received_at` (not by
    /// insertion order: the events are written 2, 3, 1), honours the limit, never
    /// shows another trigger's events, and reads back every field it was given,
    /// the absent optional ones as `None` (never `Some("")`). There is no delete
    /// in the trait: the events are scoped to fresh trigger ids and left behind.
    pub async fn check_external_event_journal(store: &dyn GraphStore) {
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        let mut rejected = an_external_event(a, IngressOutcome::Rejected, 2);
        rejected.reason = Some("invalid_signature".to_string());
        rejected.delivery_id = None;
        rejected.event_type = None;
        rejected.body_bytes = 0;
        let mut accepted = an_external_event(a, IngressOutcome::Accepted, 3);
        accepted.action = Some("opened".to_string());
        accepted.body_bytes = 1_048_576;
        let oldest = an_external_event(a, IngressOutcome::Accepted, 1);
        let foreign = an_external_event(b, IngressOutcome::Accepted, 4);
        for e in [&rejected, &accepted, &oldest, &foreign] {
            store.record_external_event(e).await.unwrap();
        }

        let listed = store.list_external_events(a, 10).await.unwrap();
        assert_eq!(
            ids(&listed, |e| e.id),
            vec![accepted.id, rejected.id, oldest.id],
            "newest received_at first, and b's event does not leak into a's journal"
        );
        assert_eq!(listed[0], accepted, "every field reads back");
        assert_eq!(
            listed[1], rejected,
            "absent optional fields read back as None"
        );
        assert_eq!(listed[2], oldest);

        let limited = store.list_external_events(a, 2).await.unwrap();
        assert_eq!(
            ids(&limited, |e| e.id),
            vec![accepted.id, rejected.id],
            "the limit keeps the newest"
        );
        let only_b = store.list_external_events(b, 10).await.unwrap();
        assert_eq!(ids(&only_b, |e| e.id), vec![foreign.id]);
        assert!(
            store
                .list_external_events(Uuid::new_v4(), 10)
                .await
                .unwrap()
                .is_empty(),
            "a trigger with no event has an empty journal"
        );
    }

    /// A delivery is claimed once per trigger: the first claim wins, a replay of
    /// the same key on the same trigger is refused (however often), while the same
    /// key on another trigger and another key on the same trigger are both free.
    /// The claims have no delete in the trait and are left behind (fresh ids).
    pub async fn check_external_delivery_claim(store: &dyn GraphStore) {
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        assert!(store
            .claim_external_delivery(a, "github:d-1")
            .await
            .unwrap());
        assert!(
            !store
                .claim_external_delivery(a, "github:d-1")
                .await
                .unwrap(),
            "a replay on the same trigger is refused"
        );
        assert!(
            !store
                .claim_external_delivery(a, "github:d-1")
                .await
                .unwrap(),
            "and stays refused"
        );
        assert!(
            store
                .claim_external_delivery(b, "github:d-1")
                .await
                .unwrap(),
            "the same key on another trigger is a different delivery"
        );
        assert!(
            store
                .claim_external_delivery(a, "github:d-2")
                .await
                .unwrap(),
            "another key on the same trigger is a different delivery"
        );
        assert!(!store
            .claim_external_delivery(b, "github:d-1")
            .await
            .unwrap());
    }

    /// A released claim is free again (the ingress route releases its claims when
    /// the journal write fails, so the sender's redelivery is not refused as a
    /// replay); releasing an absent claim is `false` and harmless; a release never
    /// touches another trigger's claim of the same key.
    pub async fn check_external_delivery_release(store: &dyn GraphStore) {
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        assert!(store.claim_external_delivery(a, "body:x").await.unwrap());
        assert!(store.claim_external_delivery(b, "body:x").await.unwrap());
        assert!(
            store.release_external_delivery(a, "body:x").await.unwrap(),
            "an existing claim is released"
        );
        assert!(
            !store.release_external_delivery(a, "body:x").await.unwrap(),
            "releasing twice finds nothing"
        );
        assert!(
            store.claim_external_delivery(a, "body:x").await.unwrap(),
            "a released delivery can be claimed again"
        );
        assert!(
            !store.claim_external_delivery(b, "body:x").await.unwrap(),
            "b's claim was not released by a's release"
        );
    }

    /// Retention: the purge deletes the events received strictly before the cutoff,
    /// of every trigger, and keeps the others. The timestamps (year 2000) sit far
    /// below every other check's (`at()`, 2026), so a concurrent check never loses
    /// its events to this purge.
    pub async fn check_external_event_purge(store: &dyn GraphStore) {
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        let when = |y, m| Utc.with_ymd_and_hms(y, m, 1, 0, 0, 0).unwrap();
        let mut old_a = an_external_event(a, IngressOutcome::Accepted, 0);
        old_a.received_at = when(2000, 1);
        let mut old_b = an_external_event(b, IngressOutcome::Rejected, 0);
        old_b.received_at = when(2000, 6);
        let mut kept = an_external_event(a, IngressOutcome::Accepted, 0);
        kept.received_at = when(2001, 6);
        let mut at_cutoff = an_external_event(b, IngressOutcome::Accepted, 0);
        at_cutoff.received_at = when(2001, 1);
        for e in [&old_a, &old_b, &kept, &at_cutoff] {
            store.record_external_event(e).await.unwrap();
        }
        let purged = store
            .purge_external_events_before(when(2001, 1))
            .await
            .unwrap();
        assert!(purged >= 2, "both old events are counted: {purged}");
        assert_eq!(
            ids(&store.list_external_events(a, 10).await.unwrap(), |e| e.id),
            vec![kept.id],
            "a keeps only its recent event"
        );
        assert_eq!(
            ids(&store.list_external_events(b, 10).await.unwrap(), |e| e.id),
            vec![at_cutoff.id],
            "an event received exactly at the cutoff is kept"
        );
        assert_eq!(
            store
                .purge_external_events_before(when(2001, 1))
                .await
                .unwrap(),
            0,
            "a second purge finds nothing"
        );
    }

    /// The claim under contention: of 8 concurrent claims of one delivery,
    /// exactly one wins.
    pub async fn check_concurrent_external_delivery_claim(store: &dyn GraphStore) {
        let trigger = Uuid::new_v4();
        let outcomes =
            join_all((0..8).map(|_| store.claim_external_delivery(trigger, "github:race"))).await;
        let won = outcomes.iter().filter(|o| matches!(o, Ok(true))).count();
        assert_eq!(
            won, 1,
            "exactly one of 8 concurrent claims wins: {outcomes:?}"
        );
        assert!(
            outcomes.iter().all(|o| o.is_ok()),
            "no claim errors: {outcomes:?}"
        );
    }

    // ------------------------------------------------------------------
    // 4. Atomic increments with a cap and a floor.
    // ------------------------------------------------------------------

    /// Concurrent `increment_frustration` calls lose nothing (8 × 0.1 = 0.8 and
    /// the returned values are the eight partial sums), the score caps at 1.0
    /// and floors at 0.0.
    pub async fn check_atomic_increments(store: &dyn GraphStore) {
        let p = a_project("increment");
        store.create_project(&p).await.unwrap();
        let plan = a_plan(p.id, "increment plan", 1, Utc::now());
        store.create_plan(&plan).await.unwrap();
        let task = a_task(Some("counter"));
        store.create_task(plan.id, &task).await.unwrap();
        assert_eq!(store.get_frustration(task.id).await.unwrap(), 0.0);

        let mut seen: Vec<f64> =
            join_all((0..8).map(|_| store.increment_frustration(task.id, 0.1)))
                .await
                .into_iter()
                .map(|r| r.unwrap())
                .collect();
        seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (i, v) in seen.iter().enumerate() {
            let expected = 0.1 * (i as f64 + 1.0);
            assert!(
                (v - expected).abs() < 1e-9,
                "returned partial sums are 0.1..=0.8 with no lost update: {seen:?}"
            );
        }
        let total = store.get_frustration(task.id).await.unwrap();
        assert!((total - 0.8).abs() < 1e-9, "8 × 0.1 = {total}");

        assert_eq!(
            store.increment_frustration(task.id, 0.5).await.unwrap(),
            1.0,
            "cap"
        );
        assert_eq!(
            store.increment_frustration(task.id, 0.1).await.unwrap(),
            1.0,
            "stays capped"
        );
        assert_eq!(store.get_frustration(task.id).await.unwrap(), 1.0);

        let after: Vec<f64> = join_all((0..12).map(|_| store.decrement_frustration(task.id, 0.1)))
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        assert!(after.iter().all(|v| (0.0..=1.0).contains(v)), "{after:?}");
        assert_eq!(store.get_frustration(task.id).await.unwrap(), 0.0, "floor");
        assert_eq!(
            store.decrement_frustration(task.id, 0.1).await.unwrap(),
            0.0,
            "stays floored"
        );

        store.delete_project(p.id, &p.name).await.unwrap();
    }

    // ------------------------------------------------------------------
    // 5. The "entity absent" matrix.
    // ------------------------------------------------------------------

    /// For an id nobody created: reads say `None`, updates and deletes are a
    /// silent no-op (or `false`) where the trait returns a flag, an increment
    /// reports 0.0, and the optimistic-lock update is the one call that errs.
    pub async fn check_absent_entity_matrix(store: &dyn GraphStore) {
        let id = Uuid::new_v4();

        assert!(store.get_project(id).await.unwrap().is_none());
        store
            .update_project(id, Some("ghost".into()), None, None, None)
            .await
            .unwrap();
        store.delete_project(id, "ghost").await.unwrap();

        assert!(store.get_plan(id).await.unwrap().is_none());
        store.delete_plan(id).await.unwrap();

        assert!(store.get_task(id).await.unwrap().is_none());
        store
            .update_task(
                id,
                &UpdateTaskRequest {
                    title: Some("ghost".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        store.delete_task(id).await.unwrap();
        assert!(store.get_task_dependencies(id).await.unwrap().is_empty());
        assert_eq!(store.increment_frustration(id, 0.3).await.unwrap(), 0.0);
        assert_eq!(store.decrement_frustration(id, 0.3).await.unwrap(), 0.0);
        assert_eq!(store.get_frustration(id).await.unwrap(), 0.0);

        assert!(store.get_note(id).await.unwrap().is_none());
        assert!(store
            .update_note(id, Some("ghost".into()), None, None, None, None)
            .await
            .unwrap()
            .is_none());
        assert!(!store.delete_note(id).await.unwrap());

        assert!(store
            .get_file("conformance/absent/nowhere.rs")
            .await
            .unwrap()
            .is_none());
        store
            .delete_file("conformance/absent/nowhere.rs")
            .await
            .unwrap();

        assert!(store.get_alert(id).await.unwrap().is_none());
        store.acknowledge_alert(id, "ghost").await.unwrap();

        assert!(store.get_protocol_run(id).await.unwrap().is_none());
        let mut run = ProtocolRun::new(id, id, "start");
        assert!(
            store.update_protocol_run(&mut run).await.is_err(),
            "the CAS update of an absent run is an error, not a silent no-op"
        );
        assert!(!store.delete_protocol_run(id).await.unwrap());
        assert!(!store.delete_protocol(id).await.unwrap());
    }

    // ------------------------------------------------------------------
    // 6. Cascade of delete_project.
    // ------------------------------------------------------------------

    /// Deleting a project removes its plans, tasks and files, and ARCHIVES its
    /// notes (status `archived`, detached from the project) instead of deleting
    /// them; the project's note listing is then empty.
    pub async fn check_cascade_delete_project(store: &dyn GraphStore) {
        let p = a_project("cascade");
        store.create_project(&p).await.unwrap();
        let plan = a_plan(p.id, "cascade plan", 1, Utc::now());
        store.create_plan(&plan).await.unwrap();
        let task = a_task(Some("cascade task"));
        store.create_task(plan.id, &task).await.unwrap();
        let path = format!("conformance/{}/cascade.rs", p.id);
        store
            .upsert_file(&a_file(&path, Some(p.id), "h"))
            .await
            .unwrap();
        let note = a_note(Some(p.id), "a note of the project");
        store.create_note(&note).await.unwrap();
        assert_eq!(store.count_project_files(p.id).await.unwrap(), 1);
        assert_eq!(
            store
                .list_notes(Some(p.id), None, &NoteFilters::default())
                .await
                .unwrap()
                .1,
            1
        );

        store.delete_project(p.id, &p.name).await.unwrap();

        assert!(store.get_project(p.id).await.unwrap().is_none());
        assert!(store.get_project_by_slug(&p.slug).await.unwrap().is_none());
        assert!(
            store.get_plan(plan.id).await.unwrap().is_none(),
            "plan cascaded"
        );
        assert!(
            store.get_task(task.id).await.unwrap().is_none(),
            "task cascaded"
        );
        assert!(
            store.get_file(&path).await.unwrap().is_none(),
            "file cascaded"
        );
        assert_eq!(store.count_project_files(p.id).await.unwrap(), 0);
        let archived = store
            .get_note(note.id)
            .await
            .unwrap()
            .expect("the note is archived, not deleted");
        assert_eq!(archived.status, NoteStatus::Archived);
        assert_eq!(archived.project_id, None, "detached from the project");
        assert_eq!(
            store
                .list_notes(Some(p.id), None, &NoteFilters::default())
                .await
                .unwrap()
                .1,
            0,
            "nothing is listed under the deleted project"
        );

        assert!(store.delete_note(note.id).await.unwrap());
    }

    // ------------------------------------------------------------------
    // 7. Sorting of a filtered list.
    // ------------------------------------------------------------------

    /// `list_plans_filtered`: the default order is `created_at DESC`, an unknown
    /// `sort_by` falls back to it, `priority`/`title` sort both ways, and paging
    /// one by one across a tie yields every row exactly once.
    pub async fn check_sorting(store: &dyn GraphStore) {
        let p = a_project("sorting");
        store.create_project(&p).await.unwrap();
        let a = a_plan(p.id, "b", 1, at(0));
        let b = a_plan(p.id, "a", 3, at(1));
        let c = a_plan(p.id, "c", 2, at(2));
        let d = a_plan(p.id, "d", 3, at(3));
        for plan in [&a, &b, &c, &d] {
            store.create_plan(plan).await.unwrap();
        }
        let list = |sort_by: Option<&'static str>,
                    order: &'static str,
                    limit: usize,
                    offset: usize| async move {
            let (plans, total) = store
                .list_plans_filtered(
                    Some(p.id),
                    None,
                    None,
                    None,
                    None,
                    None,
                    limit,
                    offset,
                    sort_by,
                    order,
                )
                .await
                .unwrap();
            assert_eq!(total, 4, "the total ignores sorting and paging");
            ids(&plans, |x| x.id)
        };

        assert_eq!(
            list(None, "desc", 10, 0).await,
            vec![d.id, c.id, b.id, a.id],
            "default: created_at DESC"
        );
        assert_eq!(
            list(None, "asc", 10, 0).await,
            vec![a.id, b.id, c.id, d.id],
            "created_at ASC"
        );
        assert_eq!(
            list(Some("no_such_field"), "desc", 10, 0).await,
            vec![d.id, c.id, b.id, a.id],
            "unknown sort_by: default order"
        );
        assert_eq!(
            list(Some("title"), "asc", 10, 0).await,
            vec![b.id, a.id, c.id, d.id]
        );
        assert_eq!(
            list(Some("title"), "desc", 10, 0).await,
            vec![d.id, c.id, a.id, b.id]
        );

        let asc = list(Some("priority"), "asc", 10, 0).await;
        assert_eq!(&asc[..2], &[a.id, c.id], "priority ASC: 1, 2, then the tie");
        assert_eq!(
            asc[2..].iter().copied().collect::<HashSet<_>>(),
            HashSet::from([b.id, d.id])
        );
        let desc = list(Some("priority"), "desc", 10, 0).await;
        assert_eq!(
            desc[..2].iter().copied().collect::<HashSet<_>>(),
            HashSet::from([b.id, d.id])
        );
        assert_eq!(
            &desc[2..],
            &[c.id, a.id],
            "priority DESC: the tie, then 2, 1"
        );

        store.delete_project(p.id, &p.name).await.unwrap();
    }

    /// The tie-break: sorted by `priority` with two plans at the same priority,
    /// paging one row at a time yields every plan exactly once. That needs a
    /// deterministic secondary key; without one, SKIP/LIMIT across the tie can
    /// repeat a row and skip another (Neo4j today: see `EXPECTED_DIVERGENCES`).
    pub async fn check_sorting_tie_break_paging(store: &dyn GraphStore) {
        let p = a_project("tie-break");
        store.create_project(&p).await.unwrap();
        let plans = [
            a_plan(p.id, "b", 1, at(0)),
            a_plan(p.id, "a", 3, at(1)),
            a_plan(p.id, "c", 2, at(2)),
            a_plan(p.id, "d", 3, at(3)),
        ];
        for plan in &plans {
            store.create_plan(plan).await.unwrap();
        }
        let mut paged = Vec::new();
        for offset in 0..plans.len() {
            let (page, total) = store
                .list_plans_filtered(
                    Some(p.id),
                    None,
                    None,
                    None,
                    None,
                    None,
                    1,
                    offset,
                    Some("priority"),
                    "asc",
                )
                .await
                .unwrap();
            assert_eq!(total, plans.len());
            paged.extend(ids(&page, |x| x.id));
        }
        store.delete_project(p.id, &p.name).await.unwrap();

        assert_eq!(paged.len(), plans.len());
        assert_eq!(
            paged.iter().copied().collect::<HashSet<_>>(),
            id_set(&plans, |x| x.id),
            "paging one by one across the tie sees every plan once: {paged:?}"
        );
    }

    // ------------------------------------------------------------------
    // 8. Pagination.
    // ------------------------------------------------------------------

    /// `list_notes`: the total never depends on `limit`/`offset`, consecutive pages
    /// partition the set in the default `created_at DESC` order, an offset past
    /// the end (or a zero limit) is an empty page with the same total.
    pub async fn check_pagination(store: &dyn GraphStore) {
        let p = a_project("pagination");
        store.create_project(&p).await.unwrap();
        let mut notes = Vec::new();
        for i in 0..5 {
            let mut n = a_note(Some(p.id), &format!("page note {i}"));
            n.created_at = at(i);
            store.create_note(&n).await.unwrap();
            notes.push(n);
        }
        let newest_first: Vec<Uuid> = notes.iter().rev().map(|n| n.id).collect();
        let page = |limit: i64, offset: i64| async move {
            let filters = NoteFilters {
                limit: Some(limit),
                offset: Some(offset),
                ..Default::default()
            };
            let (page, total) = store.list_notes(Some(p.id), None, &filters).await.unwrap();
            assert_eq!(
                total, 5,
                "the total is independent of limit {limit} / offset {offset}"
            );
            ids(&page, |n| n.id)
        };

        assert_eq!(page(100, 0).await, newest_first);
        assert_eq!(page(2, 0).await, newest_first[0..2]);
        assert_eq!(page(2, 2).await, newest_first[2..4]);
        assert_eq!(
            page(2, 4).await,
            newest_first[4..5],
            "the last page is short"
        );
        assert!(page(2, 5).await.is_empty(), "offset at the end");
        assert!(page(2, 50).await.is_empty(), "offset past the end");
        assert!(page(0, 0).await.is_empty(), "a zero limit is an empty page");

        for n in &notes {
            assert!(store.delete_note(n.id).await.unwrap());
        }
        store.delete_project(p.id, &p.name).await.unwrap();
    }

    // ------------------------------------------------------------------
    // 9. Note filters.
    // ------------------------------------------------------------------

    /// `list_notes` filters: the content search ignores case (Unicode included)
    /// and a blank search is no filter; tags are exact and combine with AND; the
    /// staleness bounds are inclusive; filters combine.
    pub async fn check_note_filters(store: &dyn GraphStore) {
        let p = a_project("filters");
        store.create_project(&p).await.unwrap();
        let mut n1 = a_note(Some(p.id), "Déjà vu in the PARSER");
        n1.tags = vec!["rust".into(), "parser".into()];
        n1.staleness_score = 0.2;
        let mut n2 = a_note(Some(p.id), "nothing to see");
        n2.tags = vec!["rust".into()];
        n2.staleness_score = 0.5;
        let mut n3 = a_note(Some(p.id), "déjà vu again");
        n3.tags = vec!["parser".into(), "go".into()];
        n3.staleness_score = 0.8;
        let mut n4 = a_note(Some(p.id), "Unicode: naïve café");
        n4.staleness_score = 1.0;
        for n in [&n1, &n2, &n3, &n4] {
            store.create_note(n).await.unwrap();
        }
        let found = |filters: NoteFilters| async move {
            let (page, total) = store.list_notes(Some(p.id), None, &filters).await.unwrap();
            assert_eq!(page.len(), total, "no paging here");
            id_set(&page, |n| n.id)
        };
        let search = |s: &str| NoteFilters {
            search: Some(s.to_string()),
            ..Default::default()
        };
        let tags = |t: &[&str]| NoteFilters {
            tags: Some(t.iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        };
        let staleness = |min: Option<f64>, max: Option<f64>| NoteFilters {
            min_staleness: min,
            max_staleness: max,
            ..Default::default()
        };

        assert_eq!(
            found(search("parser")).await,
            HashSet::from([n1.id]),
            "case-insensitive on the content"
        );
        assert_eq!(
            found(search("DÉJÀ")).await,
            HashSet::from([n1.id, n3.id]),
            "case-insensitive beyond ASCII"
        );
        assert_eq!(found(search("naïve")).await, HashSet::from([n4.id]));
        assert_eq!(
            found(search("   ")).await,
            HashSet::from([n1.id, n2.id, n3.id, n4.id]),
            "a blank search is no filter"
        );
        assert_eq!(found(search("absent")).await, HashSet::new());

        assert_eq!(found(tags(&["rust"])).await, HashSet::from([n1.id, n2.id]));
        assert_eq!(
            found(tags(&["rust", "parser"])).await,
            HashSet::from([n1.id]),
            "tags combine with AND"
        );
        assert_eq!(
            found(tags(&["Rust"])).await,
            HashSet::new(),
            "tags are exact"
        );
        assert_eq!(
            found(tags(&[])).await,
            HashSet::from([n1.id, n2.id, n3.id, n4.id]),
            "no tag is no filter"
        );

        assert_eq!(
            found(staleness(Some(0.5), None)).await,
            HashSet::from([n2.id, n3.id, n4.id]),
            "min is inclusive"
        );
        assert_eq!(
            found(staleness(None, Some(0.5))).await,
            HashSet::from([n1.id, n2.id]),
            "max is inclusive"
        );
        assert_eq!(
            found(staleness(Some(0.5), Some(0.8))).await,
            HashSet::from([n2.id, n3.id])
        );

        let combined = NoteFilters {
            tags: Some(vec!["parser".into()]),
            search: Some("déjà".into()),
            max_staleness: Some(0.5),
            ..Default::default()
        };
        assert_eq!(
            found(combined).await,
            HashSet::from([n1.id]),
            "filters combine"
        );

        for n in [&n1, &n2, &n3, &n4] {
            assert!(store.delete_note(n.id).await.unwrap());
        }
        store.delete_project(p.id, &p.name).await.unwrap();
    }

    // ------------------------------------------------------------------
    // 10. NULL versus the empty string, and timestamps, through a roundtrip.
    // ------------------------------------------------------------------

    /// What the reference store gives back for an absent value against an empty
    /// one: a project's `description` reads `Some("")` for both; a task's `title`
    /// reads `None` for both; a task's `assigned_to` reads `None` when never set
    /// and `Some("")` once set to `""`; a note's `last_confirmed_at` defaults to
    /// `created_at` and its `last_confirmed_by` to `""`. Timestamps roundtrip to
    /// the nanosecond, whether stored as a datetime or as text.
    pub async fn check_null_vs_empty_and_timestamp_roundtrip(store: &dyn GraphStore) {
        let created =
            Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap() + Duration::nanoseconds(678_901_234);

        let mut p = a_project("null");
        p.created_at = created;
        store.create_project(&p).await.unwrap();
        let read = store.get_project(p.id).await.unwrap().unwrap();
        assert_eq!(
            read.created_at, created,
            "project.created_at roundtrips exactly"
        );
        assert_eq!(
            read.description,
            Some(String::new()),
            "an absent description reads as empty"
        );
        store
            .update_project(p.id, None, Some(Some("described".into())), None, None)
            .await
            .unwrap();
        assert_eq!(
            store
                .get_project(p.id)
                .await
                .unwrap()
                .unwrap()
                .description
                .as_deref(),
            Some("described")
        );
        store
            .update_project(p.id, None, Some(None), None, None)
            .await
            .unwrap();
        assert_eq!(
            store.get_project(p.id).await.unwrap().unwrap().description,
            Some(String::new()),
            "clearing the description reads as empty again"
        );
        let mut empty = a_project("empty");
        empty.description = Some(String::new());
        store.create_project(&empty).await.unwrap();
        assert_eq!(
            store
                .get_project(empty.id)
                .await
                .unwrap()
                .unwrap()
                .description,
            Some(String::new())
        );

        let plan = a_plan(p.id, "null plan", 1, created);
        store.create_plan(&plan).await.unwrap();
        assert_eq!(
            store.get_plan(plan.id).await.unwrap().unwrap().created_at,
            created
        );
        let mut untitled = a_task(None);
        untitled.created_at = created;
        let blank = a_task(Some(""));
        let titled = a_task(Some("titled"));
        for t in [&untitled, &blank, &titled] {
            store.create_task(plan.id, t).await.unwrap();
        }
        let read = store.get_task(untitled.id).await.unwrap().unwrap();
        assert_eq!(
            read.created_at, created,
            "task.created_at roundtrips exactly"
        );
        assert_eq!(read.title, None);
        assert_eq!(read.assigned_to, None, "never set: absent");
        assert_eq!(
            store.get_task(blank.id).await.unwrap().unwrap().title,
            None,
            "an empty title reads as absent"
        );
        assert_eq!(
            store
                .get_task(titled.id)
                .await
                .unwrap()
                .unwrap()
                .title
                .as_deref(),
            Some("titled")
        );
        store
            .update_task(
                titled.id,
                &UpdateTaskRequest {
                    title: Some(String::new()),
                    assigned_to: Some(String::new()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let read = store.get_task(titled.id).await.unwrap().unwrap();
        assert_eq!(read.title, None, "a title cleared to \"\" reads as absent");
        assert_eq!(
            read.assigned_to,
            Some(String::new()),
            "an assignee set to \"\" reads as empty"
        );

        let mut note = a_note(Some(p.id), "timestamps");
        note.created_at = created;
        note.last_confirmed_at = None;
        note.last_confirmed_by = None;
        store.create_note(&note).await.unwrap();
        let read = store.get_note(note.id).await.unwrap().unwrap();
        assert_eq!(
            read.created_at, created,
            "note.created_at roundtrips exactly"
        );
        assert_eq!(
            read.last_confirmed_at,
            Some(created),
            "an absent confirmation defaults to the creation"
        );
        assert_eq!(
            read.last_confirmed_by,
            Some(String::new()),
            "an absent confirmer reads as empty"
        );

        let proto = Protocol::new(Some(p.id), "null protocol", Uuid::new_v4());
        store.upsert_protocol(&proto).await.unwrap();
        let run = a_run(&proto, None, created);
        store.create_protocol_run(&run).await.unwrap();
        let read = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(
            read.started_at, created,
            "run.started_at (stored as text) roundtrips exactly"
        );
        assert_eq!(read.completed_at, None);
        assert_eq!(read.error, None);
        assert_eq!(read.parent_run_id, None);

        assert!(store.delete_protocol(proto.id).await.unwrap());
        assert!(store.delete_note(note.id).await.unwrap());
        store.delete_project(p.id, &p.name).await.unwrap();
        store.delete_project(empty.id, &empty.name).await.unwrap();
    }

    // ------------------------------------------------------------------
    // 11. One traversal on a small fixed graph.
    // ------------------------------------------------------------------

    /// `get_run_tree` on root → {a, b} with a → a1 → a2 → a3 → a4 → a5: the root
    /// first, then descendants by depth then `started_at`, at most five levels
    /// deep (a5 is left out from the root, included from a); an unknown root is
    /// an empty tree; the direct children are listed and counted.
    pub async fn check_traversal_run_tree(store: &dyn GraphStore) {
        let p = a_project("traversal");
        store.create_project(&p).await.unwrap();
        let proto = Protocol::new(Some(p.id), "traversal protocol", Uuid::new_v4());
        store.upsert_protocol(&proto).await.unwrap();

        let root = a_run(&proto, None, at(0));
        let a = a_run(&proto, Some(&root), at(1));
        let b = a_run(&proto, Some(&root), at(2));
        let a1 = a_run(&proto, Some(&a), at(3));
        let a2 = a_run(&proto, Some(&a1), at(4));
        let a3 = a_run(&proto, Some(&a2), at(5));
        let a4 = a_run(&proto, Some(&a3), at(6));
        let a5 = a_run(&proto, Some(&a4), at(7));
        let all = [&root, &a, &b, &a1, &a2, &a3, &a4, &a5];
        for r in all {
            store.create_protocol_run(r).await.unwrap();
        }
        assert_eq!(a5.depth, 6);

        let tree = store.get_run_tree(root.id).await.unwrap();
        assert_eq!(
            ids(&tree, |r| r.id),
            vec![root.id, a.id, b.id, a1.id, a2.id, a3.id, a4.id],
            "root, then by depth and started_at, five levels deep"
        );
        let from_a = store.get_run_tree(a.id).await.unwrap();
        assert_eq!(
            ids(&from_a, |r| r.id),
            vec![a.id, a1.id, a2.id, a3.id, a4.id, a5.id]
        );
        assert!(store.get_run_tree(Uuid::new_v4()).await.unwrap().is_empty());
        let leaf = store.get_run_tree(a5.id).await.unwrap();
        assert_eq!(ids(&leaf, |r| r.id), vec![a5.id], "a leaf is its own tree");

        let children = store.list_child_runs(root.id).await.unwrap();
        assert_eq!(ids(&children, |r| r.id), vec![a.id, b.id]);
        assert_eq!(store.count_child_runs(root.id).await.unwrap(), 2);
        assert_eq!(store.count_child_runs(a4.id).await.unwrap(), 1);
        assert_eq!(store.count_child_runs(b.id).await.unwrap(), 0);

        assert!(store.delete_protocol(proto.id).await.unwrap());
        for r in all {
            store.delete_protocol_run(r.id).await.unwrap();
        }
        store.delete_project(p.id, &p.name).await.unwrap();
    }

    // ------------------------------------------------------------------
    // 12. One vector ranking under a project filter.
    // ------------------------------------------------------------------

    /// Three notes of project A at cosine 1.0, 0.9 and 0.2 from the query, an
    /// identical note in project B and an archived identical note in A:
    /// `vector_search_notes(.., Some(A), ..)` ranks A's three by score, excludes
    /// B's and the archived one, honours `limit`, and `min_similarity` applies to
    /// the score the store returns — Neo4j's cosine index scores `(1 + cos) / 2`,
    /// so the exact match scores 1.0 and cosine 0.2 scores 0.6, above 0.5.
    ///
    /// The query direction is drawn at random for each run: notes of other
    /// runs or suites in a shared database are then near-orthogonal to it
    /// (score ~0.5, below every note here) and cannot enter the ranking.
    pub async fn check_vector_ranking_with_project_filter(store: &dyn GraphStore) {
        let (query, _) = random_orthonormal_pair(Uuid::new_v4().as_u128() as u64);
        let (_, u) = random_orthonormal_pair(Uuid::new_v4().as_u128() as u64 ^ 0x9e37);
        let u = orthogonalise(&u, &query);
        let mix = |cos: f32| mix(&query, &u, cos);

        let a = a_project("vector-a");
        let b = a_project("vector-b");
        store.create_project(&a).await.unwrap();
        store.create_project(&b).await.unwrap();
        let exact = a_note(Some(a.id), "exact match");
        let close = a_note(Some(a.id), "cosine 0.9");
        let far = a_note(Some(a.id), "cosine 0.2");
        let other = a_note(Some(b.id), "exact match of another project");
        let archived = a_note(Some(a.id), "exact match, archived");
        let notes = [&exact, &close, &far, &other, &archived];
        for n in notes {
            store.create_note(n).await.unwrap();
        }
        store
            .update_note(
                archived.id,
                None,
                None,
                Some(NoteStatus::Archived),
                None,
                None,
            )
            .await
            .unwrap()
            .expect("archived");
        for (n, v) in [
            (&exact, query.clone()),
            (&close, mix(0.9)),
            (&far, mix(0.2)),
            (&other, query.clone()),
            (&archived, query.clone()),
        ] {
            store
                .set_note_embedding(n.id, &v, "conformance")
                .await
                .unwrap();
        }
        let stored_len = store
            .get_note_embedding(close.id)
            .await
            .unwrap()
            .map(|e| e.len());

        // Every query first, the cleanup second, the assertions last: a failed
        // assertion must not leave notes behind in a shared database.
        let query = &query;
        let search = |limit: usize, project: Uuid, min: Option<f64>| async move {
            store
                .vector_search_notes(query, limit, Some(project), None, min)
                .await
                .unwrap()
        };
        let ranked = search(10, a.id, None).await;
        let top2 = search(2, a.id, None).await;
        let above_half = search(10, a.id, Some(0.5)).await;
        let above_nine = search(10, a.id, Some(0.9)).await;
        let of_b = search(10, b.id, None).await;

        for n in notes {
            assert!(store.delete_note(n.id).await.unwrap());
        }
        store.delete_project(a.id, &a.name).await.unwrap();
        store.delete_project(b.id, &b.name).await.unwrap();

        assert_eq!(stored_len, Some(DIM));
        assert_eq!(
            ids(&ranked, |(n, _)| n.id),
            vec![exact.id, close.id, far.id],
            "A's active notes by descending score, nothing else: {:?}",
            ranked
                .iter()
                .map(|(n, s)| (n.content.clone(), *s))
                .collect::<Vec<_>>()
        );
        let scores: Vec<f64> = ranked.iter().map(|(_, s)| *s).collect();
        assert!(
            scores.windows(2).all(|w| w[0] > w[1]),
            "strictly descending: {scores:?}"
        );
        // Neo4j's index computes in float32 and lands within 1e-2 of the formula
        // (measured 2026-10-20: [1.0, 0.949, 0.608] for cosines 1.0, 0.9, 0.2).
        assert!(
            (scores[0] - 1.0).abs() < 1e-2,
            "the exact match scores 1.0: {scores:?}"
        );
        assert!(
            (scores[1] - 0.95).abs() < 1e-2,
            "cosine 0.9 scores (1 + 0.9) / 2: {scores:?}"
        );
        assert!(
            (scores[2] - 0.6).abs() < 1e-2,
            "cosine 0.2 scores (1 + 0.2) / 2: {scores:?}"
        );
        assert_eq!(ids(&top2, |(n, _)| n.id), vec![exact.id, close.id], "limit");
        assert_eq!(
            ids(&above_half, |(n, _)| n.id),
            vec![exact.id, close.id, far.id],
            "min_similarity 0.5 keeps the score 0.6"
        );
        assert_eq!(
            ids(&above_nine, |(n, _)| n.id),
            vec![exact.id, close.id],
            "min_similarity 0.9 drops it"
        );
        assert_eq!(
            ids(&of_b, |(n, _)| n.id),
            vec![other.id],
            "the filter isolates B"
        );
    }

    /// The project filter must not lose a project's notes to another project's:
    /// with ten notes of project B at cosine 1.0 and two of project A at 0.9 and
    /// 0.8, `vector_search_notes(.., 2, Some(A), ..)` is A's two. (Neo4j today
    /// fetches `3 × limit` index candidates and filters by project afterwards,
    /// so B's ten fill the six candidates and A gets nothing: see
    /// `EXPECTED_DIVERGENCES`.)
    pub async fn check_vector_project_filter_recall(store: &dyn GraphStore) {
        let (query, _) = random_orthonormal_pair(Uuid::new_v4().as_u128() as u64);
        let (_, u) = random_orthonormal_pair(Uuid::new_v4().as_u128() as u64 ^ 0x9e37);
        let u = orthogonalise(&u, &query);

        let a = a_project("recall-a");
        let b = a_project("recall-b");
        store.create_project(&a).await.unwrap();
        store.create_project(&b).await.unwrap();
        let a1 = a_note(Some(a.id), "A, cosine 0.9");
        let a2 = a_note(Some(a.id), "A, cosine 0.8");
        let crowd: Vec<Note> = (0..10)
            .map(|i| a_note(Some(b.id), &format!("B, exact match {i}")))
            .collect();
        for n in [&a1, &a2].into_iter().chain(crowd.iter()) {
            store.create_note(n).await.unwrap();
        }
        store
            .set_note_embedding(a1.id, &mix(&query, &u, 0.9), "conformance")
            .await
            .unwrap();
        store
            .set_note_embedding(a2.id, &mix(&query, &u, 0.8), "conformance")
            .await
            .unwrap();
        for n in &crowd {
            store
                .set_note_embedding(n.id, &query, "conformance")
                .await
                .unwrap();
        }
        let of_a = store
            .vector_search_notes(&query, 2, Some(a.id), None, None)
            .await
            .unwrap();

        for n in [&a1, &a2].into_iter().chain(crowd.iter()) {
            assert!(store.delete_note(n.id).await.unwrap());
        }
        store.delete_project(a.id, &a.name).await.unwrap();
        store.delete_project(b.id, &b.name).await.unwrap();

        assert_eq!(
            ids(&of_a, |(n, _)| n.id),
            vec![a1.id, a2.id],
            "A's own notes, whatever B holds near the query"
        );
    }

    const DIM: usize = 768;

    /// Two unit vectors of a pseudo-random dense direction (xorshift, seeded):
    /// one for the query, one to build the other cosines from.
    fn random_orthonormal_pair(seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
        };
        let q = normalise((0..DIM).map(|_| next()).collect());
        let r: Vec<f32> = (0..DIM).map(|_| next()).collect();
        (q.clone(), orthogonalise(&r, &q))
    }

    fn normalise(v: Vec<f32>) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }

    /// The unit vector of `v` minus its projection on the unit vector `q`.
    fn orthogonalise(v: &[f32], q: &[f32]) -> Vec<f32> {
        let dot: f32 = v.iter().zip(q).map(|(a, b)| a * b).sum();
        normalise(v.iter().zip(q).map(|(a, b)| a - dot * b).collect())
    }

    /// The unit vector at cosine `cos` from `q`, in the plane of `q` and `u`
    /// (`q` and `u` orthonormal).
    fn mix(q: &[f32], u: &[f32], cos: f32) -> Vec<f32> {
        let sin = (1.0 - cos * cos).sqrt();
        q.iter().zip(u).map(|(a, b)| cos * a + sin * b).collect()
    }

    // ------------------------------------------------------------------
    // 13. Numbers keep their type.
    // ------------------------------------------------------------------

    /// A counter written as an integer reads back as an integer, a decimal as a
    /// decimal, and a total the store counts itself counts what is there.
    ///
    /// Every read of these numbers in the storage layer is typed and falls back
    /// in silence when the property does not come back as the expected type:
    /// `get::<i64>("activation_count").unwrap_or(0)` answers 0 for a counter,
    /// `get("staleness_score").unwrap_or(0.0)` answers 0.0 for a decimal,
    /// `get("energy").unwrap_or(1.0)` answers 1.0, `get::<i64>("total_steps")
    /// .unwrap_or(0)` answers 0 for a total, and `get::<i64>("priority").ok()`
    /// quietly drops an integer it could not read. None of that raises an error,
    /// so a number that changed type is visible here or nowhere.
    pub async fn check_numeric_typing_preserved(store: &dyn GraphStore) {
        let p = a_project("numbers");
        store.create_project(&p).await.unwrap();

        let mut note = a_note(Some(p.id), "a counted note");
        note.activation_count = 7; // an integer counter
        note.staleness_score = 0.375; // a decimal, exactly representable
        note.energy = 0.625; // another one, with a fallback of its own (1.0)
        store.create_note(&note).await.unwrap();

        let plan = a_plan(p.id, "numbers plan", 7, at(0));
        store.create_plan(&plan).await.unwrap();
        let mut task = a_task(Some("counted"));
        task.priority = Some(7);
        task.estimated_minutes = Some(45);
        store.create_task(plan.id, &task).await.unwrap();
        // Three steps, two of them completed: totals the store computes itself.
        for order in 0..3 {
            let step = StepNode {
                id: Uuid::new_v4(),
                order,
                description: format!("step {order}"),
                status: if order < 2 {
                    StepStatus::Completed
                } else {
                    StepStatus::Pending
                },
                verification: None,
                created_at: at(i64::from(order)),
                updated_at: None,
                completed_at: None,
                execution_context: None,
                persona: None,
            };
            store.create_step(task.id, &step).await.unwrap();
        }

        // Every read first, the cleanup second, the assertions last: a failed
        // assertion must not leave rows behind in a shared database.
        let read_note = store
            .get_note(note.id)
            .await
            .unwrap()
            .expect("the note exists");
        let read_task = store
            .get_task(task.id)
            .await
            .unwrap()
            .expect("the task exists");
        let counted = store
            .get_task_enrichment_counts(&[task.id.to_string()])
            .await
            .unwrap();
        let progress = store.get_task_step_progress(task.id).await.unwrap();

        assert!(store.delete_note(note.id).await.unwrap());
        store.delete_project(p.id, &p.name).await.unwrap();

        assert_eq!(
            read_note.activation_count, 7,
            "an integer counter stays an integer: written 7, read {} — a counter \
             stored as a float reads back through \
             get::<i64>(\"activation_count\").unwrap_or(0) as 0, with no error",
            read_note.activation_count
        );
        assert_eq!(
            read_note.staleness_score, 0.375,
            "a decimal stays a decimal: written 0.375, read {} — a score stored \
             as an integer reads back through unwrap_or(0.0) as 0.0",
            read_note.staleness_score
        );
        assert_eq!(
            read_note.energy, 0.625,
            "and so does the second decimal of the same note: written 0.625, \
             read {} (its fallback is 1.0, a wrong value that is not even zero)",
            read_note.energy
        );
        let counts = counted
            .get(&task.id.to_string())
            .expect("the counted task is in the map");
        assert_eq!(
            (counts.step_count, counts.completed_step_count),
            (3, 2),
            "a total the store counts is 3 steps and 2 completed, read {} and {} \
             — a count read through get::<i64>(\"total_steps\").unwrap_or(0) \
             silently answers 0 when its type is not an integer",
            counts.step_count,
            counts.completed_step_count
        );
        assert_eq!(
            progress,
            (2, 3),
            "the (completed, total) pair of the same steps is (2, 3), read {progress:?}"
        );
        assert_eq!(
            (read_task.priority, read_task.estimated_minutes),
            (Some(7), Some(45)),
            "integers read back as integers: read ({:?}, {:?}) — an integer \
             stored as a float is dropped by get::<i64>(..).ok(), without an error",
            read_task.priority,
            read_task.estimated_minutes
        );
    }

    // ------------------------------------------------------------------
    // 14. Timestamps: the instant, its rank, and a deadline.
    // ------------------------------------------------------------------

    /// A run's `started_at` is written as RFC 3339 text: it must read back as the
    /// instant it was, to the nanosecond; three encodings of one second must rank
    /// chronologically, which is not what the characters of a text sort say; and a
    /// deadline computed by adding a duration to a stored instant must fall
    /// between the stored instants that sit 250 ms on either side of it.
    ///
    /// The values are chosen so that a truncation to the whole second, a shift of
    /// the offset, or a comparison of two different RFC 3339 encodings moves one
    /// of them across the deadline. The trait only accepts instants, so the
    /// offset is always the one the store writes; the axis that varies here is
    /// the precision of the encoding.
    pub async fn check_datetime_roundtrip_and_arithmetic(store: &dyn GraphStore) {
        let p = a_project("datetime");
        store.create_project(&p).await.unwrap();
        let proto = Protocol::new(Some(p.id), "datetime protocol", Uuid::new_v4());
        store.upsert_protocol(&proto).await.unwrap();

        // One second, three encodings: no fraction, milliseconds, nanoseconds.
        let second = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
        let root = a_run(&proto, None, second - Duration::hours(1));
        let bare = a_run(&proto, Some(&root), second);
        let millis = a_run(&proto, Some(&root), second + Duration::milliseconds(400));
        let nanos = a_run(
            &proto,
            Some(&root),
            second + Duration::milliseconds(900) + Duration::nanoseconds(999),
        );
        // A deadline computed from `bare` by adding a duration, straddled by
        // 250 ms on both sides.
        let deadline = second + Duration::seconds(30);
        let just_before = a_run(&proto, Some(&root), deadline - Duration::milliseconds(250));
        let just_after = a_run(&proto, Some(&root), deadline + Duration::milliseconds(250));
        let runs = [&root, &bare, &millis, &nanos, &just_before, &just_after];
        for r in runs {
            store.create_protocol_run(r).await.unwrap();
        }

        let read = |id: Uuid| async move {
            store
                .get_protocol_run(id)
                .await
                .unwrap()
                .expect("the run was created")
        };
        let stored_bare = read(bare.id).await;
        let stored_millis = read(millis.id).await;
        let stored_nanos = read(nanos.id).await;
        let stored_before = read(just_before.id).await;
        let stored_after = read(just_after.id).await;
        let newest_first = store
            .list_protocol_runs(proto.id, None, 10, 0)
            .await
            .unwrap()
            .0;
        let by_depth = store.get_run_tree(root.id).await.unwrap();

        assert!(store.delete_protocol(proto.id).await.unwrap());
        for r in runs {
            store.delete_protocol_run(r.id).await.unwrap();
        }
        store.delete_project(p.id, &p.name).await.unwrap();

        assert_eq!(
            stored_bare.started_at, second,
            "a run whose instant falls on a whole second reads back as that \
             second, not as the moment of the read: {}",
            stored_bare.started_at
        );
        assert_eq!(
            stored_millis.started_at,
            second + Duration::milliseconds(400),
            "a millisecond fraction survives the text form: {}",
            stored_millis.started_at
        );
        assert_eq!(
            stored_nanos.started_at,
            second + Duration::milliseconds(900) + Duration::nanoseconds(999),
            "and so does a nanosecond one: {} — a store that keeps the second \
             alone makes these three runs one and the same instant",
            stored_nanos.started_at
        );
        assert_eq!(
            ids(&by_depth, |r| r.id),
            vec![
                root.id,
                bare.id,
                millis.id,
                nanos.id,
                just_before.id,
                just_after.id
            ],
            "the run tree ranks by the instant: in RFC 3339 text, \
             \"...03:04:05+00:00\" precedes \"...03:04:05.400+00:00\" only because \
             '+' sorts before '.', and a text sort of two encodings is a guess"
        );
        assert_eq!(
            ids(&newest_first, |r| r.id),
            vec![
                just_after.id,
                just_before.id,
                nanos.id,
                millis.id,
                bare.id,
                root.id
            ],
            "the listing is newest first by the instant, every instant distinct"
        );
        let from_store = stored_bare.started_at + Duration::seconds(30);
        assert!(
            stored_before.started_at < from_store,
            "a stored instant 250 ms before the deadline computed from a stored \
             instant is before it: {} vs {from_store}",
            stored_before.started_at
        );
        assert!(
            stored_after.started_at > from_store,
            "and the one 250 ms after is after it: {} vs {from_store}",
            stored_after.started_at
        );
        assert_eq!(
            stored_after.started_at - stored_before.started_at,
            Duration::milliseconds(500),
            "the two straddlers are exactly 500 ms apart"
        );
    }

    // ------------------------------------------------------------------
    // 15. Paging is deterministic across a tie.
    // ------------------------------------------------------------------

    /// Seven notes on three written instants, so the default `created_at DESC`
    /// key has ties (three at t, two at t + 1 s, two at t + 2 s): three walks of
    /// the same pages, and the whole list asked for at once, must agree on every
    /// boundary — each note seen once, none skipped. The instants are given
    /// explicitly and never come from `now()`, so the tie is the same tie on
    /// every run of the suite.
    pub async fn check_ordering_is_deterministic_across_pages(store: &dyn GraphStore) {
        let p = a_project("paging-ties");
        store.create_project(&p).await.unwrap();
        let t = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
        let mut notes = Vec::new();
        for (i, seconds) in [0i64, 0, 0, 1, 1, 2, 2].into_iter().enumerate() {
            let mut n = a_note(Some(p.id), &format!("tied note {i}"));
            n.created_at = t + Duration::seconds(seconds);
            store.create_note(&n).await.unwrap();
            notes.push(n);
        }
        let total_expected = notes.len();
        let page = |limit: i64, offset: i64| async move {
            let filters = NoteFilters {
                limit: Some(limit),
                offset: Some(offset),
                ..Default::default()
            };
            let (page, total) = store.list_notes(Some(p.id), None, &filters).await.unwrap();
            assert_eq!(total, total_expected, "the total ignores paging");
            ids(&page, |n| n.id)
        };

        let mut walks: Vec<Vec<Uuid>> = Vec::new();
        for _pass in 0..3 {
            let mut walk: Vec<Uuid> = Vec::new();
            let mut offset = 0;
            loop {
                let batch = page(2, offset).await;
                if batch.is_empty() {
                    break;
                }
                walk.extend(batch);
                offset += 2;
            }
            walks.push(walk);
        }
        let whole = page(total_expected as i64, 0).await;

        for n in &notes {
            assert!(store.delete_note(n.id).await.unwrap());
        }
        store.delete_project(p.id, &p.name).await.unwrap();

        let all: HashSet<Uuid> = notes.iter().map(|n| n.id).collect();
        assert_eq!(
            walks[0].len(),
            total_expected,
            "a walk of pages of 2 sees every note once: {:?}",
            walks[0]
        );
        assert_eq!(
            walks[0].iter().copied().collect::<HashSet<_>>(),
            all,
            "no note is skipped or seen twice across the pages: {:?}",
            walks[0]
        );
        for (pass, walk) in walks.iter().enumerate().skip(1) {
            assert_eq!(
                walk, &walks[0],
                "pass {pass} draws its page boundaries elsewhere than the first \
                 walk, so a tie is broken differently on each query: {walk:?} vs {:?}",
                walks[0]
            );
        }
        assert_eq!(
            &whole, &walks[0],
            "the pages follow the order the whole list gives, page after page: \
             whole {whole:?} vs paged {:?}",
            walks[0]
        );
    }

    // ------------------------------------------------------------------
    // The project domain (`ProjectStore`), on its own.
    //
    // These checks call nothing but `ProjectStore`, so they are the suite a
    // backend ported one domain at a time can run first (obrain: see
    // `ObrainFactory`). They state what the Neo4j adapter does, including where
    // it is lossy, and compare whole values: every serialized field of
    // `ProjectNode`, not a chosen few.
    // ------------------------------------------------------------------

    /// Every serialized field of a project, for whole-value comparisons.
    fn fields(p: &ProjectNode) -> serde_json::Value {
        serde_json::to_value(p).unwrap()
    }

    async fn stored(store: &dyn GraphStore, id: Uuid) -> ProjectNode {
        store
            .get_project(id)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("project {id} must be readable"))
    }

    /// A root that resolves to nothing on any machine, so its canonical form
    /// is its declared form everywhere.
    fn a_missing_root() -> String {
        format!("/nonexistent/conformance-{}", Uuid::new_v4())
    }

    // ------------------------------------------------------------------
    // Federation (MCP servers and their tools): the checks that call nothing
    // but `FederationStore`, run by `federation_checks!`.
    // ------------------------------------------------------------------

    pub fn an_mcp_server(
        project_id: Uuid,
        server_id: &str,
    ) -> project_orchestrator::neo4j::models::McpServerNode {
        project_orchestrator::neo4j::models::McpServerNode {
            id: Uuid::new_v4(),
            project_id,
            server_id: server_id.to_string(),
            display_name: format!("conformance {server_id}"),
            transport_type: "stdio".to_string(),
            transport_url: None,
            transport_command: Some("conformance-cmd".to_string()),
            transport_args: Some("[\"--flag\"]".to_string()),
            transport_secrets: None,
            status: "disconnected".to_string(),
            protocol_version: None,
            server_name: Some("conformance".to_string()),
            tool_count: 0,
            created_at: Utc::now(),
            updated_at: None,
            last_connected_at: None,
        }
    }

    pub fn an_mcp_tool(
        server_id: &str,
        name: &str,
    ) -> project_orchestrator::neo4j::models::McpToolNode {
        project_orchestrator::neo4j::models::McpToolNode {
            id: Uuid::new_v4(),
            server_id: server_id.to_string(),
            name: name.to_string(),
            fqn: format!("{server_id}::{name}"),
            description: "conformance tool".to_string(),
            input_schema: "{}".to_string(),
            category: "query".to_string(),
            embedding: None,
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    /// Every field but the timestamps, compared WITHOUT printing them: the
    /// stdio secrets of a server must never reach a test log.
    fn same_server(
        a: &project_orchestrator::neo4j::models::McpServerNode,
        b: &project_orchestrator::neo4j::models::McpServerNode,
    ) -> bool {
        a.id == b.id
            && a.project_id == b.project_id
            && a.server_id == b.server_id
            && a.display_name == b.display_name
            && a.transport_type == b.transport_type
            && a.transport_url == b.transport_url
            && a.transport_command == b.transport_command
            && a.transport_args == b.transport_args
            && a.transport_secrets == b.transport_secrets
            && a.status == b.status
            && a.protocol_version == b.protocol_version
            && a.server_name == b.server_name
            && a.tool_count == b.tool_count
            && a.last_connected_at == b.last_connected_at
    }

    /// A server and a tool come back with the values written; the absent
    /// optional values read back absent; the absent ids read as absent.
    pub async fn check_mcp_roundtrip(store: &dyn GraphStore) {
        let project = Uuid::new_v4();
        let sid = format!("conformance-rt-{}", Uuid::new_v4());
        let mut s = an_mcp_server(project, &sid);
        s.transport_secrets = Some("{\"env\":{\"K\":\"v\"}}".to_string());
        store.create_mcp_server(&s).await.unwrap();

        let back = store
            .get_mcp_server(s.id)
            .await
            .unwrap()
            .expect("read by id");
        assert!(
            same_server(&back, &s),
            "the server read back differs (secrets not printed)"
        );
        assert_eq!(
            back.transport_url, None,
            "an absent value reads back absent"
        );
        let listed = store.list_mcp_servers(project).await.unwrap();
        assert_eq!(listed.len(), 1, "list_mcp_servers of its project");
        assert!(same_server(&listed[0], &s), "the listed server differs");
        assert!(store
            .list_mcp_servers(Uuid::new_v4())
            .await
            .unwrap()
            .is_empty());
        assert!(store
            .get_mcp_server(Uuid::new_v4())
            .await
            .unwrap()
            .is_none());

        let t = an_mcp_tool(&sid, "read");
        store.create_mcp_tool(&t).await.unwrap();
        let tools = store.list_mcp_tools_for_server(&sid).await.unwrap();
        assert_eq!(tools.len(), 1, "one tool for the server");
        assert_eq!(tools[0].id, t.id);
        assert_eq!(tools[0].fqn, t.fqn);
        assert_eq!(tools[0].description, t.description);
        assert_eq!(
            tools[0].embedding, None,
            "an absent embedding reads back absent"
        );
        assert!(store
            .list_mcp_tools_for_server(&format!("absent-{}", Uuid::new_v4()))
            .await
            .unwrap()
            .is_empty());

        store.delete_mcp_server(s.id).await.unwrap();
    }

    /// Servers come out sorted by `server_id`, the tools of a server by `name`:
    /// what Neo4j states (`ORDER BY s.server_id`, `ORDER BY t.name, t.id`). A tie on
    /// `server_id` has no order in the contract, so the order WITHIN a tie is not
    /// asserted here; the obrain order, total, is checked in
    /// `obrain_federation_specifique`.
    pub async fn check_mcp_list_order(store: &dyn GraphStore) {
        let project = Uuid::new_v4();
        let tag = Uuid::new_v4();
        let sid = |x: &str| format!("conformance-ord-{tag}-{x}");

        let servers = vec![
            an_mcp_server(project, &sid("zeta")),
            an_mcp_server(project, &sid("alpha")),
            an_mcp_server(project, &sid("mu")),
            an_mcp_server(project, &sid("alpha")),
        ];
        for s in &servers {
            store.create_mcp_server(s).await.unwrap();
        }
        let expected: HashSet<Uuid> = servers.iter().map(|s| s.id).collect();
        let listed = store.list_mcp_servers(project).await.unwrap();
        assert_eq!(
            id_set(&listed, |s| s.id),
            expected,
            "list_mcp_servers: the same servers"
        );
        let keys: Vec<&str> = listed.iter().map(|s| s.server_id.as_str()).collect();
        assert!(
            keys.windows(2).all(|w| w[0] <= w[1]),
            "list_mcp_servers by server_id: {keys:?}"
        );
        let all: Vec<_> = store
            .list_all_mcp_servers()
            .await
            .unwrap()
            .into_iter()
            .filter(|s| expected.contains(&s.id))
            .collect();
        assert_eq!(
            id_set(&all, |s| s.id),
            expected,
            "list_all_mcp_servers: the same servers among ours"
        );
        let keys: Vec<&str> = all.iter().map(|s| s.server_id.as_str()).collect();
        assert!(
            keys.windows(2).all(|w| w[0] <= w[1]),
            "list_all_mcp_servers by server_id: {keys:?}"
        );

        let tools_sid = sid("tools");
        let tools = vec![
            an_mcp_tool(&tools_sid, "write"),
            an_mcp_tool(&tools_sid, "read"),
            an_mcp_tool(&tools_sid, "list"),
            an_mcp_tool(&tools_sid, "read"),
        ];
        for t in &tools {
            store.create_mcp_tool(t).await.unwrap();
        }
        let listed_tools = store.list_mcp_tools_for_server(&tools_sid).await.unwrap();
        assert_eq!(
            id_set(&listed_tools, |t| t.id),
            tools.iter().map(|t| t.id).collect::<HashSet<_>>(),
            "list_mcp_tools_for_server: the same tools"
        );
        let names: Vec<&str> = listed_tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["list", "read", "read", "write"],
            "list_mcp_tools_for_server by name"
        );

        for s in &servers {
            store.delete_mcp_server(s.id).await.unwrap();
        }
        store.delete_mcp_tools_for_server(&tools_sid).await.unwrap();
    }

    /// A status update is read back, and is a no-op on an absent server. Deleting
    /// a server takes its tools with it; by `server_id` it takes every server with
    /// that id; deleting what is absent is not an error.
    pub async fn check_mcp_status_and_deletion(store: &dyn GraphStore) {
        let project = Uuid::new_v4();
        let sid = format!("conformance-del-{}", Uuid::new_v4());
        let s = an_mcp_server(project, &sid);
        store.create_mcp_server(&s).await.unwrap();
        store
            .create_mcp_tool(&an_mcp_tool(&sid, "a"))
            .await
            .unwrap();
        store
            .create_mcp_tool(&an_mcp_tool(&sid, "b"))
            .await
            .unwrap();

        store
            .update_mcp_server_status(s.id, "connected")
            .await
            .unwrap();
        assert_eq!(
            store.get_mcp_server(s.id).await.unwrap().unwrap().status,
            "connected"
        );
        store
            .update_mcp_server_status(Uuid::new_v4(), "error")
            .await
            .unwrap();

        store.delete_mcp_server(s.id).await.unwrap();
        assert!(store.get_mcp_server(s.id).await.unwrap().is_none());
        assert!(store
            .list_mcp_tools_for_server(&sid)
            .await
            .unwrap()
            .is_empty());
        store.delete_mcp_server(s.id).await.unwrap();

        let again = an_mcp_server(project, &sid);
        store.create_mcp_server(&again).await.unwrap();
        store
            .create_mcp_tool(&an_mcp_tool(&sid, "c"))
            .await
            .unwrap();
        store.delete_mcp_server_by_server_id(&sid).await.unwrap();
        assert!(store.get_mcp_server(again.id).await.unwrap().is_none());
        assert!(store
            .list_mcp_tools_for_server(&sid)
            .await
            .unwrap()
            .is_empty());
        assert!(store.list_mcp_servers(project).await.unwrap().is_empty());
        store
            .delete_mcp_server_by_server_id(&format!("absent-{}", Uuid::new_v4()))
            .await
            .unwrap();

        store
            .create_mcp_tool(&an_mcp_tool(&sid, "d"))
            .await
            .unwrap();
        store.delete_mcp_tools_for_server(&sid).await.unwrap();
        assert!(store
            .list_mcp_tools_for_server(&sid)
            .await
            .unwrap()
            .is_empty());
    }

    /// What is created reads back identical, by id and by slug, down to the
    /// nanosecond of `created_at`; what was never created reads as absent.
    pub async fn check_project_roundtrip(store: &dyn GraphStore) {
        let mut work = a_project("roundtrip-work");
        work.description = Some("a description — with an accent: é".to_string());
        work.created_at = at(7) + Duration::nanoseconds(123_456_789);
        work.watch_enabled = true;
        work.profile = ProjectProfile::Work;

        // Above the 256-byte threshold where obrain moves a string to its blob column,
        // multi-byte on purpose (a byte cut inside a character would not read back).
        let mut long = a_project("roundtrip-long");
        long.description = Some("é-long ".repeat(300));
        long.created_at = at(9);

        let mut software = a_project("roundtrip-software");
        software.description = Some("another".to_string());
        software.root_path = a_missing_root();
        software.created_at = at(8);

        for p in [&work, &software, &long] {
            store.create_project(p).await.unwrap();
        }
        for p in [&work, &software, &long] {
            let by_id = stored(store, p.id).await;
            assert_eq!(fields(&by_id), fields(p), "read by id");
            assert_eq!(by_id.created_at, p.created_at, "created_at, exactly");
            let by_slug = store
                .get_project_by_slug(&p.slug)
                .await
                .unwrap()
                .expect("read by slug");
            assert_eq!(fields(&by_slug), fields(p), "read by slug");
        }

        assert!(store.get_project(Uuid::new_v4()).await.unwrap().is_none());
        assert!(store
            .get_project_by_slug(&format!("conformance-absent-{}", Uuid::new_v4()))
            .await
            .unwrap()
            .is_none());

        for p in [&work, &software, &long] {
            store.delete_project(p.id, &p.name).await.unwrap();
        }
    }

    /// `list_projects` is ordered by name then id, ascending — a TOTAL order:
    /// three projects sharing one name come out by id, the same way on every
    /// call. Names compare by code point (upper case before lower case,
    /// accented letters after `z`).
    pub async fn check_project_list_order(store: &dyn GraphStore) {
        let tag = Uuid::new_v4();
        let named = |name: &str| {
            let mut p = a_project("order");
            p.name = format!("conformance order {tag} {name}");
            p.description = Some(String::new());
            p
        };
        let mut projects = [
            named("same"),
            named("élan"),
            named("same"),
            named("alpha"),
            named("same"),
            named("Zeta"),
        ];
        for p in &projects {
            store.create_project(p).await.unwrap();
        }
        // The stated order, computed here: name, then the id as a string.
        projects.sort_by(|a, b| {
            a.name
                .cmp(&b.name)
                .then_with(|| a.id.to_string().cmp(&b.id.to_string()))
        });
        let expected = ids(&projects, |p| p.id);
        let suffixes: Vec<&str> = projects
            .iter()
            .map(|p| p.name.rsplit(' ').next().unwrap())
            .collect();
        assert_eq!(
            suffixes,
            ["Zeta", "alpha", "same", "same", "same", "élan"],
            "the fixture itself: code-point order of the names"
        );

        let mine: HashSet<Uuid> = expected.iter().copied().collect();
        for pass in 0..3 {
            let listed: Vec<Uuid> = store
                .list_projects()
                .await
                .unwrap()
                .iter()
                .map(|p| p.id)
                .filter(|id| mine.contains(id))
                .collect();
            assert_eq!(
                listed, expected,
                "pass {pass}: list_projects must order by name then id (a tie on the name is \
                 broken by the id, identically on every call)"
            );
        }
        // And the values listed are the values stored.
        let listed = store.list_projects().await.unwrap();
        for p in &projects {
            let l = listed.iter().find(|l| l.id == p.id).unwrap();
            assert_eq!(fields(l), fields(p), "a listed project is the stored one");
        }

        // A deleted project leaves the list; the others keep their order.
        let (gone, rest) = projects.split_first().unwrap();
        store.delete_project(gone.id, &gone.name).await.unwrap();
        let listed: Vec<Uuid> = store
            .list_projects()
            .await
            .unwrap()
            .iter()
            .map(|p| p.id)
            .filter(|id| mine.contains(id))
            .collect();
        assert_eq!(
            listed,
            ids(rest, |p| p.id),
            "a deleted project is not listed"
        );

        for p in rest {
            store.delete_project(p.id, &p.name).await.unwrap();
        }
    }

    /// `update_project` changes exactly the fields it is given; an update with
    /// nothing to change, or on an unknown project, changes nothing and is not
    /// an error (and does not create the project).
    pub async fn check_project_update(store: &dyn GraphStore) {
        let mut p = a_project("update");
        p.description = Some("before".to_string());
        store.create_project(&p).await.unwrap();
        let mut witness = a_project("update-witness");
        witness.description = Some("untouched".to_string());
        store.create_project(&witness).await.unwrap();

        p.name = format!("conformance renamed {}", p.id);
        store
            .update_project(p.id, Some(p.name.clone()), None, None, None)
            .await
            .unwrap();
        assert_eq!(fields(&stored(store, p.id).await), fields(&p), "name only");

        p.description = Some("after".to_string());
        store
            .update_project(p.id, None, Some(p.description.clone()), None, None)
            .await
            .unwrap();
        assert_eq!(
            fields(&stored(store, p.id).await),
            fields(&p),
            "description only"
        );

        // Short -> long (past the blob threshold) -> short again: the value that
        // replaces a blob must be the value read, whichever way the size goes.
        for description in ["é-long ".repeat(3_000), "short again".to_string()] {
            p.description = Some(description);
            store
                .update_project(p.id, None, Some(p.description.clone()), None, None)
                .await
                .unwrap();
            assert_eq!(
                fields(&stored(store, p.id).await),
                fields(&p),
                "description crossing the blob threshold ({} bytes)",
                p.description.as_ref().unwrap().len()
            );
        }

        p.profile = ProjectProfile::Work;
        store
            .update_project(p.id, None, None, None, Some(p.profile))
            .await
            .unwrap();
        assert_eq!(
            fields(&stored(store, p.id).await),
            fields(&p),
            "profile only"
        );

        p.root_path = a_missing_root();
        store
            .update_project(p.id, None, None, Some(p.root_path.clone()), None)
            .await
            .unwrap();
        assert_eq!(fields(&stored(store, p.id).await), fields(&p), "root only");

        // `Some("")` clears the root.
        p.root_path = String::new();
        store
            .update_project(p.id, None, None, Some(String::new()), None)
            .await
            .unwrap();
        assert_eq!(
            fields(&stored(store, p.id).await),
            fields(&p),
            "root cleared"
        );

        // All four at once.
        p.name = format!("conformance renamed twice {}", p.id);
        p.description = Some("all at once".to_string());
        p.root_path = a_missing_root();
        p.profile = ProjectProfile::Software;
        store
            .update_project(
                p.id,
                Some(p.name.clone()),
                Some(p.description.clone()),
                Some(p.root_path.clone()),
                Some(p.profile),
            )
            .await
            .unwrap();
        assert_eq!(fields(&stored(store, p.id).await), fields(&p), "all four");

        store
            .update_project(p.id, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(fields(&stored(store, p.id).await), fields(&p), "nothing");

        let unknown = Uuid::new_v4();
        store
            .update_project(unknown, Some("ghost".to_string()), None, None, None)
            .await
            .unwrap();
        assert!(
            store.get_project(unknown).await.unwrap().is_none(),
            "an update never creates a project"
        );
        assert_eq!(
            fields(&stored(store, witness.id).await),
            fields(&witness),
            "another project is never touched"
        );
        // The slug is not updatable: the project still answers to it.
        assert_eq!(
            store
                .get_project_by_slug(&p.slug)
                .await
                .unwrap()
                .map(|f| f.id),
            Some(p.id)
        );

        store.delete_project(p.id, &p.name).await.unwrap();
        store
            .delete_project(witness.id, &witness.name)
            .await
            .unwrap();
    }

    /// Neo4j keeps no absent description: a project created without one, or
    /// whose description is cleared, reads back `Some("")`.
    pub async fn check_project_absent_description_reads_empty(store: &dyn GraphStore) {
        let created_without = a_project("no-description");
        assert!(created_without.description.is_none(), "the fixture");
        store.create_project(&created_without).await.unwrap();
        assert_eq!(
            stored(store, created_without.id).await.description,
            Some(String::new()),
            "created with no description"
        );

        let mut cleared = a_project("cleared-description");
        cleared.description = Some("to be cleared".to_string());
        store.create_project(&cleared).await.unwrap();
        store
            .update_project(cleared.id, None, Some(None), None, None)
            .await
            .unwrap();
        assert_eq!(
            stored(store, cleared.id).await.description,
            Some(String::new()),
            "description cleared by an update"
        );

        for p in [&created_without, &cleared] {
            store.delete_project(p.id, &p.name).await.unwrap();
        }
    }

    /// `create_project` stores the creation fields and nothing else: the sync,
    /// analytics and co-change instants, the note energy and the scaffolding
    /// override are written by their own setters, and a value passed at
    /// creation is not kept.
    pub async fn check_project_creation_keeps_only_creation_fields(store: &dyn GraphStore) {
        let mut p = a_project("creation-fields");
        p.description = Some("kept".to_string());
        p.last_synced = Some(at(1));
        p.analytics_computed_at = Some(at(2));
        p.last_co_change_computed_at = Some(at(3));
        p.default_note_energy = Some(0.5);
        p.scaffolding_override = Some(2);
        p.sharing_policy =
            Some(project_orchestrator::episodes::distill_models::SharingPolicy::default());
        store.create_project(&p).await.unwrap();

        let mut expected = p.clone();
        expected.last_synced = None;
        expected.analytics_computed_at = None;
        expected.last_co_change_computed_at = None;
        expected.default_note_energy = None;
        expected.scaffolding_override = None;
        expected.sharing_policy = None;
        assert_eq!(fields(&stored(store, p.id).await), fields(&expected));

        store.delete_project(p.id, &p.name).await.unwrap();
    }

    /// The two timestamps of the trait are set to the instant of the call,
    /// each on its own, and neither is an error (nor a creation) on an
    /// unknown project.
    pub async fn check_project_timestamps(store: &dyn GraphStore) {
        let mut p = a_project("timestamps");
        p.description = Some("timestamps".to_string());
        store.create_project(&p).await.unwrap();
        let before = stored(store, p.id).await;
        assert!(before.last_synced.is_none() && before.analytics_computed_at.is_none());

        let slack = Duration::seconds(2);
        let t0 = Utc::now();
        store.update_project_synced(p.id).await.unwrap();
        let t1 = Utc::now();
        let synced = stored(store, p.id).await;
        let at_sync = synced.last_synced.expect("last_synced is set");
        assert!(
            at_sync >= t0 - slack && at_sync <= t1 + slack,
            "last_synced {at_sync} is the instant of the call ({t0} .. {t1})"
        );
        assert!(
            synced.analytics_computed_at.is_none(),
            "the sync timestamp does not set the analytics one"
        );

        let t2 = Utc::now();
        store
            .update_project_analytics_timestamp(p.id)
            .await
            .unwrap();
        let t3 = Utc::now();
        let analysed = stored(store, p.id).await;
        let at_analytics = analysed.analytics_computed_at.expect("analytics is set");
        assert!(
            at_analytics >= t2 - slack && at_analytics <= t3 + slack,
            "analytics_computed_at {at_analytics} is the instant of the call ({t2} .. {t3})"
        );
        assert_eq!(
            analysed.last_synced,
            Some(at_sync),
            "the analytics timestamp leaves last_synced as it was"
        );

        // Nothing else moved.
        let mut expected = p.clone();
        expected.last_synced = analysed.last_synced;
        expected.analytics_computed_at = analysed.analytics_computed_at;
        assert_eq!(fields(&analysed), fields(&expected));

        let unknown = Uuid::new_v4();
        store.update_project_synced(unknown).await.unwrap();
        store
            .update_project_analytics_timestamp(unknown)
            .await
            .unwrap();
        assert!(store.get_project(unknown).await.unwrap().is_none());

        store.delete_project(p.id, &p.name).await.unwrap();
    }

    /// A deleted project is gone by id and by slug, its slug is free again, the
    /// others are untouched; deleting twice, or deleting what never existed, is
    /// not an error. (That it leaves the LIST is `check_project_list_order`:
    /// the one check that lists, see its entry in `EXPECTED_DIVERGENCES`.)
    pub async fn check_project_delete(store: &dyn GraphStore) {
        let mut gone = a_project("delete-gone");
        gone.description = Some("gone".to_string());
        let mut kept = a_project("delete-kept");
        kept.description = Some("kept".to_string());
        store.create_project(&gone).await.unwrap();
        store.create_project(&kept).await.unwrap();

        store.delete_project(gone.id, &gone.name).await.unwrap();
        assert!(store.get_project(gone.id).await.unwrap().is_none());
        assert!(store
            .get_project_by_slug(&gone.slug)
            .await
            .unwrap()
            .is_none());
        assert_eq!(fields(&stored(store, kept.id).await), fields(&kept));

        store.delete_project(gone.id, &gone.name).await.unwrap();
        store
            .delete_project(Uuid::new_v4(), "never existed")
            .await
            .unwrap();

        // The slug is free again, and now names the new project.
        let mut reborn = a_project("delete-reborn");
        reborn.slug = gone.slug.clone();
        reborn.description = Some("reborn".to_string());
        store.create_project(&reborn).await.unwrap();
        assert_eq!(
            store
                .get_project_by_slug(&gone.slug)
                .await
                .unwrap()
                .map(|p| p.id),
            Some(reborn.id)
        );

        for p in [&kept, &reborn] {
            store.delete_project(p.id, &p.name).await.unwrap();
        }
    }

    /// `Project.id` and `Project.slug` are unique: a second project with a
    /// taken id or a taken slug is refused, and the first one is left as it was.
    pub async fn check_project_unique_id_and_slug(store: &dyn GraphStore) {
        let mut first = a_project("unique-first");
        first.description = Some("first".to_string());
        store.create_project(&first).await.unwrap();

        let mut same_id = a_project("unique-same-id");
        same_id.id = first.id;
        assert!(
            store.create_project(&same_id).await.is_err(),
            "a second project with a taken id must be refused"
        );
        assert_eq!(fields(&stored(store, first.id).await), fields(&first));
        assert!(store
            .get_project_by_slug(&same_id.slug)
            .await
            .unwrap()
            .is_none());

        let mut same_slug = a_project("unique-same-slug");
        same_slug.slug = first.slug.clone();
        assert!(
            store.create_project(&same_slug).await.is_err(),
            "a second project with a taken slug must be refused"
        );
        assert!(store.get_project(same_slug.id).await.unwrap().is_none());
        assert_eq!(
            store
                .get_project_by_slug(&first.slug)
                .await
                .unwrap()
                .map(|p| p.id),
            Some(first.id)
        );

        store.delete_project(first.id, &first.name).await.unwrap();
        // Whatever a non-conforming backend let through.
        store
            .delete_project(same_slug.id, &same_slug.name)
            .await
            .unwrap();
    }

    /// Eight concurrent creations of ONE slug (eight ids): exactly one wins, and
    /// the others leave nothing behind. The uniqueness of the slug holds under
    /// concurrency, not only in sequence.
    pub async fn check_project_concurrent_same_slug(store: &dyn GraphStore) {
        let slug = format!("conformance-race-{}", Uuid::new_v4());
        let contenders: Vec<ProjectNode> = (0..8)
            .map(|i| {
                let mut p = a_project(&format!("race-{i}"));
                p.slug = slug.clone();
                p.description = Some(format!("contender {i}"));
                p
            })
            .collect();
        let results = join_all(contenders.iter().map(|p| store.create_project(p))).await;
        let winners: Vec<usize> = results
            .iter()
            .enumerate()
            .filter(|(_, r)| r.is_ok())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            winners.len(),
            1,
            "exactly one creation of a slug wins, got {} ({results:?})",
            winners.len()
        );
        let winner = &contenders[winners[0]];
        assert_eq!(
            store
                .get_project_by_slug(&slug)
                .await
                .unwrap()
                .map(|p| fields(&p)),
            Some(fields(winner)),
            "the slug names the winner, whole"
        );
        for (i, p) in contenders.iter().enumerate() {
            if i != winners[0] {
                assert!(
                    store.get_project(p.id).await.unwrap().is_none(),
                    "a refused creation leaves no project (contender {i})"
                );
            }
        }
        for p in &contenders {
            store.delete_project(p.id, &p.name).await.unwrap();
        }
    }

    /// The path policy of a project: none for an unknown project; declared
    /// from creation; its root normalized (no trailing `/`) and following an
    /// update of the root.
    pub async fn check_project_path_policy(store: &dyn GraphStore) {
        assert_eq!(
            store.path_policy(Uuid::new_v4()).await.unwrap(),
            PathPolicy::none(),
            "an unknown project has no vocabulary"
        );

        let rootless = a_project("policy-rootless");
        store.create_project(&rootless).await.unwrap();
        assert_eq!(
            store.path_policy(rootless.id).await.unwrap(),
            PathPolicy {
                declared_root: String::new(),
                canonical_root: String::new(),
                declared: true,
            }
        );

        let root = a_missing_root();
        let mut rooted = a_project("policy-rooted");
        rooted.root_path = format!("{root}/");
        store.create_project(&rooted).await.unwrap();
        assert_eq!(
            store.path_policy(rooted.id).await.unwrap(),
            PathPolicy {
                declared_root: root.clone(),
                canonical_root: root.clone(),
                declared: true,
            },
            "the trailing `/` of the root is not part of the policy"
        );

        let moved = a_missing_root();
        store
            .update_project(rooted.id, None, None, Some(moved.clone()), None)
            .await
            .unwrap();
        assert_eq!(
            store.path_policy(rooted.id).await.unwrap(),
            PathPolicy {
                declared_root: moved.clone(),
                canonical_root: moved,
                declared: true,
            },
            "the policy follows the root"
        );

        for p in [&rootless, &rooted] {
            store.delete_project(p.id, &p.name).await.unwrap();
        }
    }

    // ------------------------------------------------------------------
    // Release domain: releases, environments, deployments, milestones, and the
    // project list the release store serves (`list_projects_filtered`).
    // Timestamps are whole seconds so that every backend stores the same value.
    // ------------------------------------------------------------------

    fn a_release(project_id: Uuid, version: &str, created: i64) -> ReleaseNode {
        ReleaseNode {
            id: Uuid::new_v4(),
            version: version.to_string(),
            title: None,
            description: None,
            status: ReleaseStatus::Planned,
            target_date: None,
            released_at: None,
            created_at: at(created),
            project_id,
        }
    }

    fn json<T: serde::Serialize>(v: &T) -> serde_json::Value {
        serde_json::to_value(v).unwrap()
    }

    /// A release reads back the values it was created with, `update_release` changes
    /// exactly the fields it is given, an update on an unknown id is a no-op, and a
    /// deleted release is gone.
    pub async fn check_release_roundtrip(store: &dyn GraphStore) {
        let p = a_project("release-roundtrip");
        store.create_project(&p).await.unwrap();

        let mut r = a_release(p.id, "1.0.0", 10);
        r.title = Some("first".to_string());
        r.description = None;
        r.target_date = Some(at(100));
        store.create_release(&r).await.unwrap();

        let got = store.get_release(r.id).await.unwrap().unwrap();
        assert_eq!(
            json(&got),
            json(&r),
            "a release reads back what was created"
        );
        // An absent description is stored as "" and reads back as "" or None, but never
        // as a value of its own: the title and the dates stay as they were given.
        assert_eq!(got.title.as_deref(), Some("first"));
        assert_eq!(got.target_date, Some(at(100)));
        assert_eq!(got.released_at, None);

        store
            .update_release(
                r.id,
                Some(ReleaseStatus::InProgress),
                None,
                None,
                None,
                Some("notes".to_string()),
            )
            .await
            .unwrap();
        let got = store.get_release(r.id).await.unwrap().unwrap();
        assert_eq!(got.status, ReleaseStatus::InProgress, "status changed");
        assert_eq!(
            got.description.as_deref(),
            Some("notes"),
            "description changed"
        );
        assert_eq!(got.title.as_deref(), Some("first"), "title untouched");
        assert_eq!(got.target_date, Some(at(100)), "target_date untouched");

        // Unknown id: nothing is created, nothing fails.
        store
            .update_release(
                Uuid::new_v4(),
                Some(ReleaseStatus::Released),
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        // A release of a project that does not exist is refused.
        let ghost = a_release(Uuid::new_v4(), "9.9.9", 1);
        assert!(store.create_release(&ghost).await.is_err());
        assert!(store.get_release(ghost.id).await.unwrap().is_none());

        store.delete_release(r.id).await.unwrap();
        assert!(store.get_release(r.id).await.unwrap().is_none());
        assert!(store
            .list_project_releases(p.id)
            .await
            .unwrap()
            .iter()
            .all(|x| x.id != r.id));
        store.delete_release(r.id).await.unwrap(); // second delete: no-op
        store.delete_project(p.id, &p.name).await.unwrap();
    }

    /// `list_project_releases` is newest first, ties broken by id descending;
    /// `list_releases_filtered` filters by status (any case), sorts by the field asked
    /// (id ascending as the tie-break) and pages with the total.
    pub async fn check_release_list_order(store: &dyn GraphStore) {
        let p = a_project("release-order");
        store.create_project(&p).await.unwrap();
        let mut a = a_release(p.id, "1.0.0", 30);
        let mut b = a_release(p.id, "1.1.0", 20);
        let mut c = a_release(p.id, "2.0.0", 20);
        a.status = ReleaseStatus::Released;
        b.status = ReleaseStatus::Planned;
        c.status = ReleaseStatus::InProgress;
        b.target_date = Some(at(500));
        for r in [&a, &b, &c] {
            store.create_release(r).await.unwrap();
        }

        // Newest first; the tie on created_at (b, c) is broken by id, descending.
        let mut tie = [b.id, c.id];
        tie.sort();
        tie.reverse();
        let expected = vec![a.id, tie[0], tie[1]];
        let listed: Vec<Uuid> = store
            .list_project_releases(p.id)
            .await
            .unwrap()
            .iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(listed, expected, "newest first, then id descending");

        let (all, total) = store
            .list_releases_filtered(p.id, None, 10, 0, None, "desc")
            .await
            .unwrap();
        assert_eq!(total, 3);
        // The filtered list breaks a tie on created_at by id ASCENDING (not descending
        // as the project list does): `b` and `c` share created_at.
        let (low, high) = if b.id < c.id {
            (b.id, c.id)
        } else {
            (c.id, b.id)
        };
        assert_eq!(
            ids(&all, |r| r.id),
            vec![a.id, low, high],
            "default sort: created_at desc, then id asc"
        );

        let (in_progress, total) = store
            .list_releases_filtered(
                p.id,
                Some(vec!["in_progress".to_string()]),
                10,
                0,
                None,
                "desc",
            )
            .await
            .unwrap();
        assert_eq!(total, 1, "a status filter counts its matches");
        assert_eq!(ids(&in_progress, |r| r.id), vec![c.id]);

        // Sorted by version ascending, then one row per page: the total stays 3.
        let (page1, total) = store
            .list_releases_filtered(p.id, None, 1, 0, Some("version"), "asc")
            .await
            .unwrap();
        assert_eq!(total, 3);
        assert_eq!(ids(&page1, |r| r.id), vec![a.id], "version 1.0.0 first");
        let (page2, _) = store
            .list_releases_filtered(p.id, None, 1, 1, Some("version"), "asc")
            .await
            .unwrap();
        assert_eq!(ids(&page2, |r| r.id), vec![b.id], "then 1.1.0");

        // A target_date sort puts the release without one first, ascending.
        let (by_target, _) = store
            .list_releases_filtered(p.id, None, 10, 0, Some("target_date"), "asc")
            .await
            .unwrap();
        assert_eq!(by_target.last().map(|r| r.id), Some(b.id));

        for r in [&a, &b, &c] {
            store.delete_release(r.id).await.unwrap();
        }
        store.delete_project(p.id, &p.name).await.unwrap();
    }

    /// Environments: names are unique within a project, listed oldest first, updated
    /// field by field (an empty string clears an optional one), and deleting one
    /// deletes its deployments.
    pub async fn check_environment_lifecycle(store: &dyn GraphStore) {
        let p = a_project("env-lifecycle");
        store.create_project(&p).await.unwrap();
        let env = |name: &str, created: i64| EnvironmentNode {
            id: Uuid::new_v4(),
            project_id: p.id,
            name: name.to_string(),
            kind: EnvironmentKind::Staging,
            url: Some("https://example.test".to_string()),
            description: None,
            config: None,
            created_at: at(created),
        };
        let late = env("late", 50);
        let early = env("early", 10);
        store.create_environment(&late).await.unwrap();
        store.create_environment(&early).await.unwrap();

        let dup = EnvironmentNode {
            id: Uuid::new_v4(),
            ..env("late", 60)
        };
        assert!(
            store.create_environment(&dup).await.is_err(),
            "a second environment with the same name in the project is refused"
        );

        let listed = store.list_project_environments(p.id).await.unwrap();
        assert_eq!(
            ids(&listed, |e| e.id),
            vec![early.id, late.id],
            "oldest first"
        );

        store
            .update_environment(
                early.id,
                None,
                Some(EnvironmentKind::Production),
                Some(String::new()),
                Some("the live one".to_string()),
                None,
            )
            .await
            .unwrap();
        let got = store.get_environment(early.id).await.unwrap().unwrap();
        assert_eq!(got.kind, EnvironmentKind::Production);
        assert_eq!(got.url, None, "an empty string clears url");
        assert_eq!(got.description.as_deref(), Some("the live one"));
        assert_eq!(got.name, "early", "name untouched");

        let dep = DeploymentNode {
            id: Uuid::new_v4(),
            environment_id: early.id,
            version: Some("1.0.0".to_string()),
            commit_sha: None,
            status: DeploymentStatus::Pending,
            notes: None,
            created_by: "tester".to_string(),
            started_at: at(70),
            finished_at: None,
        };
        store.create_deployment(&dep).await.unwrap();
        store.delete_environment(early.id).await.unwrap();
        assert!(store.get_environment(early.id).await.unwrap().is_none());
        assert!(
            store.get_deployment(dep.id).await.unwrap().is_none(),
            "its deployments go with it"
        );

        let orphan = DeploymentNode {
            id: Uuid::new_v4(),
            environment_id: early.id,
            ..dep.clone()
        };
        assert!(
            store.create_deployment(&orphan).await.is_err(),
            "a deployment of an unknown environment is refused"
        );

        store.delete_environment(late.id).await.unwrap();
        store.delete_project(p.id, &p.name).await.unwrap();
    }

    /// Deployments: newest first with the total, paged; the matrix gives each
    /// environment its latest deployment and the statuses of its last five.
    pub async fn check_deployment_matrix(store: &dyn GraphStore) {
        let p = a_project("deploy-matrix");
        store.create_project(&p).await.unwrap();
        let prod = EnvironmentNode {
            id: Uuid::new_v4(),
            project_id: p.id,
            name: "prod".to_string(),
            kind: EnvironmentKind::Production,
            url: None,
            description: None,
            config: None,
            created_at: at(1),
        };
        let dev = EnvironmentNode {
            id: Uuid::new_v4(),
            name: "dev".to_string(),
            kind: EnvironmentKind::Dev,
            created_at: at(2),
            ..prod.clone()
        };
        store.create_environment(&prod).await.unwrap();
        store.create_environment(&dev).await.unwrap();

        let statuses = [
            DeploymentStatus::Succeeded,
            DeploymentStatus::Failed,
            DeploymentStatus::Running,
            DeploymentStatus::Succeeded,
            DeploymentStatus::RolledBack,
            DeploymentStatus::Pending,
            DeploymentStatus::Succeeded,
        ];
        let mut made = Vec::new();
        for (i, s) in statuses.iter().enumerate() {
            let d = DeploymentNode {
                id: Uuid::new_v4(),
                environment_id: prod.id,
                version: None,
                commit_sha: None,
                status: *s,
                notes: None,
                created_by: "ci".to_string(),
                started_at: at(100 + i as i64),
                finished_at: None,
            };
            store.create_deployment(&d).await.unwrap();
            made.push(d);
        }

        let (page, total) = store
            .list_environment_deployments(prod.id, 3, 0)
            .await
            .unwrap();
        assert_eq!(total, 7);
        let newest_first: Vec<Uuid> = made.iter().rev().map(|d| d.id).collect();
        assert_eq!(
            ids(&page, |d| d.id),
            newest_first[..3].to_vec(),
            "newest first"
        );
        let (rest, _) = store
            .list_environment_deployments(prod.id, 10, 5)
            .await
            .unwrap();
        assert_eq!(
            ids(&rest, |d| d.id),
            newest_first[5..].to_vec(),
            "offset pages"
        );

        let matrix = store.get_deployment_matrix(p.id).await.unwrap();
        assert_eq!(matrix.len(), 2);
        assert_eq!(
            matrix[0].environment.name, "prod",
            "environments oldest first"
        );
        assert_eq!(matrix[1].environment.name, "dev");
        assert_eq!(
            matrix[0].latest_deployment.as_ref().map(|d| d.id),
            Some(made[6].id)
        );
        assert_eq!(
            matrix[0].recent_statuses,
            statuses.iter().rev().take(5).copied().collect::<Vec<_>>(),
            "the last five statuses, newest first"
        );
        assert!(matrix[1].latest_deployment.is_none());
        assert!(matrix[1].recent_statuses.is_empty());

        store.delete_environment(prod.id).await.unwrap();
        store.delete_environment(dev.id).await.unwrap();
        store.delete_project(p.id, &p.name).await.unwrap();
    }

    /// Milestones: created, read, listed by target date (undated ones first), updated
    /// field by field, filtered by status, and deleted.
    pub async fn check_milestone_lifecycle(store: &dyn GraphStore) {
        let p = a_project("milestone");
        store.create_project(&p).await.unwrap();
        let jalon = |title: &str, target: Option<i64>, created: i64| MilestoneNode {
            id: Uuid::new_v4(),
            title: title.to_string(),
            description: None,
            status: MilestoneStatus::Open,
            target_date: target.map(at),
            closed_at: None,
            created_at: at(created),
            project_id: p.id,
        };
        let later = jalon("later", Some(900), 1);
        let sooner = jalon("sooner", Some(300), 2);
        let undated = jalon("undated", None, 3);
        for j in [&later, &sooner, &undated] {
            store.create_milestone(j).await.unwrap();
        }
        assert_eq!(
            json(&store.get_milestone(later.id).await.unwrap().unwrap()),
            json(&later)
        );

        let listed = store.list_project_milestones(p.id).await.unwrap();
        assert_eq!(
            ids(&listed, |j| j.id),
            vec![undated.id, sooner.id, later.id],
            "target_date ascending, an absent one first"
        );

        // list_milestones_filtered: ORDER BY <field> <dir>, m.id — every sort key,
        // both directions (the stored target_date of an absent one is "").
        for (sort_by, order, want) in [
            (None, "desc", vec![later.id, sooner.id, undated.id]),
            (None, "asc", vec![undated.id, sooner.id, later.id]),
            (Some("title"), "asc", vec![later.id, sooner.id, undated.id]),
            (Some("title"), "desc", vec![undated.id, sooner.id, later.id]),
            (
                Some("created_at"),
                "desc",
                vec![undated.id, sooner.id, later.id],
            ),
            (
                Some("created_at"),
                "asc",
                vec![later.id, sooner.id, undated.id],
            ),
        ] {
            let (page, total) = store
                .list_milestones_filtered(p.id, None, 10, 0, sort_by, order)
                .await
                .unwrap();
            assert_eq!(
                (ids(&page, |j| j.id), total),
                (want, 3),
                "list_milestones_filtered sorted by {sort_by:?} {order}"
            );
        }

        store
            .update_milestone(
                later.id,
                Some(MilestoneStatus::InProgress),
                None,
                None,
                Some("renamed".to_string()),
                None,
            )
            .await
            .unwrap();
        let got = store.get_milestone(later.id).await.unwrap().unwrap();
        assert_eq!(got.status, MilestoneStatus::InProgress);
        assert_eq!(got.title, "renamed");
        assert_eq!(got.target_date, Some(at(900)), "target_date untouched");

        let (open, total) = store
            .list_milestones_filtered(p.id, Some(vec!["open".to_string()]), 10, 0, None, "desc")
            .await
            .unwrap();
        assert_eq!(total, 2);
        assert_eq!(open.len(), 2);

        store.delete_milestone(later.id).await.unwrap();
        assert!(store.get_milestone(later.id).await.unwrap().is_none());
        store.delete_milestone(sooner.id).await.unwrap();
        store.delete_milestone(undated.id).await.unwrap();

        let ghost = jalon("ghost", None, 4);
        let ghost = MilestoneNode {
            project_id: Uuid::new_v4(),
            ..ghost
        };
        assert!(store.create_milestone(&ghost).await.is_err());
        store.delete_project(p.id, &p.name).await.unwrap();
    }

    /// `list_projects_filtered`: the search matches the description (case-insensitively),
    /// the sort is by name by default, and the total counts the matches before the page.
    pub async fn check_project_list_filtered(store: &dyn GraphStore) {
        let tag = Uuid::new_v4();
        let mut a = a_project("filtered-a");
        a.name = format!("filtered {tag} b");
        a.description = Some(format!("Marker {tag}"));
        let mut b = a_project("filtered-b");
        b.name = format!("filtered {tag} a");
        b.description = Some(format!("other {tag}"));
        store.create_project(&a).await.unwrap();
        store.create_project(&b).await.unwrap();

        // Both carry the tag in their description, so the search lists both, sorted by
        // name ascending (`… a` is `b`, `… b` is `a`). The shared instance may hold many
        // projects: the page is the whole set of the tag, so a small page is enough.
        let (all, total) = store
            .list_projects_filtered(Some(&tag.to_string()), 50, 0, Some("name"), "asc")
            .await
            .unwrap();
        assert_eq!(total, 2, "the total counts the matches of the search");
        assert_eq!(ids(&all, |p| p.id), vec![b.id, a.id], "name ascending");
        let (upper, _) = store
            .list_projects_filtered(Some(&format!("MARKER {tag}")), 50, 0, None, "asc")
            .await
            .unwrap();
        assert_eq!(ids(&upper, |p| p.id), vec![a.id], "the match ignores case");

        // last_synced: an absent one sorts last ascending and first descending.
        store.update_project_synced(a.id).await.unwrap();
        for (order, want) in [("asc", vec![a.id, b.id]), ("desc", vec![b.id, a.id])] {
            let (page, _) = store
                .list_projects_filtered(Some(&tag.to_string()), 50, 0, Some("last_synced"), order)
                .await
                .unwrap();
            assert_eq!(ids(&page, |p| p.id), want, "last_synced {order}");
        }

        store.delete_project(a.id, &a.name).await.unwrap();
        store.delete_project(b.id, &b.name).await.unwrap();
    }

    // ------------------------------------------------------------------
    // The workspace domain (`WorkspaceStore`), on its own.
    //
    // Same discipline as the project checks: they call nothing but the
    // workspace contract and `create_project` / `delete_project` for the
    // projects they attach. Neo4j is the reference; where the obrain adapter
    // is strict (a duplicate slug, a duplicate component name) no check here
    // asserts the Neo4j laxity.
    // ------------------------------------------------------------------

    fn a_workspace(label: &str) -> WorkspaceNode {
        let id = Uuid::new_v4();
        WorkspaceNode {
            id,
            name: format!("conformance ws {label} {id}"),
            slug: format!("conformance-ws-{label}-{}", id.simple()),
            description: None,
            created_at: at(0),
            updated_at: None,
            metadata: serde_json::json!({}),
        }
    }

    /// Created workspaces read back identical by id and by slug; an absent one
    /// reads as absent; `updated_at` stays unset until an update sets it.
    pub async fn check_workspace_roundtrip(store: &dyn GraphStore) {
        let mut ws = a_workspace("roundtrip");
        ws.description = Some("une description — avec un accent : é".to_string());
        ws.created_at = at(7) + Duration::nanoseconds(123_456_789);
        ws.metadata = serde_json::json!({"lane": "backend", "n": 3});
        store.create_workspace(&ws).await.unwrap();

        let by_id = store.get_workspace(ws.id).await.unwrap().expect("by id");
        assert_eq!(
            serde_json::to_value(&by_id).unwrap(),
            serde_json::to_value(&ws).unwrap()
        );
        let by_slug = store
            .get_workspace_by_slug(&ws.slug)
            .await
            .unwrap()
            .expect("by slug");
        assert_eq!(by_slug.id, ws.id);
        assert!(by_id.updated_at.is_none(), "updated_at unset by create");
        assert!(store.get_workspace(Uuid::new_v4()).await.unwrap().is_none());
        assert!(store
            .get_workspace_by_slug(&format!("conformance-absent-{}", Uuid::new_v4()))
            .await
            .unwrap()
            .is_none());
        store.delete_workspace(ws.id).await.unwrap();
        assert!(store.get_workspace(ws.id).await.unwrap().is_none());
    }

    /// `list_workspaces` is ordered by name then id, ascending, and total: three
    /// workspaces sharing one name come out by id.
    pub async fn check_workspace_list_order(store: &dyn GraphStore) {
        let mut made = Vec::new();
        for _ in 0..3 {
            let mut ws = a_workspace("order");
            ws.name = "conformance ws same-name".to_string();
            store.create_workspace(&ws).await.unwrap();
            made.push(ws);
        }
        let mut expected: Vec<Uuid> = made.iter().map(|w| w.id).collect();
        expected.sort();
        let listed: Vec<Uuid> = store
            .list_workspaces()
            .await
            .unwrap()
            .into_iter()
            .map(|w| w.id)
            .filter(|id| expected.contains(id))
            .collect();
        assert_eq!(listed, expected, "name then id, ascending");
        for ws in &made {
            store.delete_workspace(ws.id).await.unwrap();
        }
    }

    /// A slug is validated on update, its uniqueness is checked against the
    /// OTHER workspaces, and a changed slug is the only one that resolves.
    pub async fn check_workspace_slug_rules(store: &dyn GraphStore) {
        let a = a_workspace("slug-a");
        let b = a_workspace("slug-b");
        store.create_workspace(&a).await.unwrap();
        store.create_workspace(&b).await.unwrap();

        let invalid = store
            .update_workspace(a.id, None, None, None, Some("Bad_Slug".into()))
            .await;
        assert!(invalid.is_err(), "an invalid slug is refused");

        let taken = store
            .update_workspace(a.id, None, None, None, Some(b.slug.clone()))
            .await
            .expect_err("a slug of another workspace is refused");
        assert!(taken.to_string().contains("CONFLICT"), "{taken}");

        let fresh = format!("conformance-ws-renamed-{}", a.id.simple());
        store
            .update_workspace(
                a.id,
                Some("renamed".into()),
                None,
                None,
                Some(fresh.clone()),
            )
            .await
            .unwrap();
        let moved = store
            .get_workspace_by_slug(&fresh)
            .await
            .unwrap()
            .expect("new slug");
        assert_eq!(moved.id, a.id);
        assert_eq!(moved.name, "renamed");
        assert!(moved.updated_at.is_some(), "an update sets updated_at");
        assert!(store
            .get_workspace_by_slug(&a.slug)
            .await
            .unwrap()
            .is_none());

        // Keeping one's own slug is not a conflict.
        store
            .update_workspace(a.id, None, Some("again".into()), None, Some(fresh.clone()))
            .await
            .unwrap();
        store.delete_workspace(a.id).await.unwrap();
        store.delete_workspace(b.id).await.unwrap();
    }

    /// Membership is a set: adding twice counts once, removing takes it away,
    /// an unknown project is a silent no-op, and deleting the workspace keeps
    /// the projects.
    pub async fn check_workspace_membership(store: &dyn GraphStore) {
        let ws = a_workspace("members");
        let p = a_project("member-p");
        let q = a_project("member-q");
        store.create_workspace(&ws).await.unwrap();
        store.create_project(&p).await.unwrap();
        store.create_project(&q).await.unwrap();

        store.add_project_to_workspace(ws.id, p.id).await.unwrap();
        store.add_project_to_workspace(ws.id, p.id).await.unwrap();
        store
            .add_project_to_workspace(ws.id, Uuid::new_v4())
            .await
            .unwrap();
        let listed = store.list_workspace_projects(ws.id).await.unwrap();
        assert_eq!(listed.iter().map(|x| x.id).collect::<Vec<_>>(), vec![p.id]);
        assert_eq!(
            store
                .get_project_workspace(p.id)
                .await
                .unwrap()
                .map(|w| w.id),
            Some(ws.id)
        );

        store
            .remove_project_from_workspace(ws.id, p.id)
            .await
            .unwrap();
        assert!(store
            .list_workspace_projects(ws.id)
            .await
            .unwrap()
            .is_empty());
        assert!(store.get_project_workspace(p.id).await.unwrap().is_none());

        store.add_project_to_workspace(ws.id, q.id).await.unwrap();
        store.delete_workspace(ws.id).await.unwrap();
        assert!(
            store.get_project(q.id).await.unwrap().is_some(),
            "the project survives"
        );
        assert!(store.get_project_workspace(q.id).await.unwrap().is_none());
        store.delete_project(p.id, &p.name).await.unwrap();
        store.delete_project(q.id, &q.name).await.unwrap();
    }

    /// Milestones: ordered by target date (absent first), then title, then id;
    /// the status filter pages with a total; an update answers whether the id
    /// exists; a milestone under an absent workspace is not created.
    pub async fn check_workspace_milestones(store: &dyn GraphStore) {
        let ws = a_workspace("milestones");
        store.create_workspace(&ws).await.unwrap();
        let mk = |title: &str, day: Option<i64>, status: MilestoneStatus| WorkspaceMilestoneNode {
            id: Uuid::new_v4(),
            workspace_id: ws.id,
            title: title.to_string(),
            description: Some(format!("about {title}")),
            status,
            target_date: day.map(at),
            closed_at: None,
            created_at: at(1),
            tags: vec!["t".into()],
        };
        let late = mk("late", Some(900), MilestoneStatus::Open);
        let soon = mk("soon", Some(100), MilestoneStatus::Completed);
        let undated = mk("undated", None, MilestoneStatus::Open);
        for m in [&late, &soon, &undated] {
            store.create_workspace_milestone(m).await.unwrap();
        }

        let order: Vec<Uuid> = store
            .list_workspace_milestones(ws.id)
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert_eq!(
            order,
            vec![undated.id, soon.id, late.id],
            "target date, absent first"
        );

        let (page, total) = store
            .list_workspace_milestones_filtered(ws.id, Some("open"), 1, 0)
            .await
            .unwrap();
        assert_eq!(total, 2, "two open milestones");
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].id, undated.id);
        let (page, total) = store
            .list_workspace_milestones_filtered(ws.id, Some("open"), 10, 1)
            .await
            .unwrap();
        assert_eq!((total, page.len()), (2, 1));
        assert_eq!(page[0].id, late.id);

        assert!(store
            .update_workspace_milestone(
                soon.id,
                Some("sooner".into()),
                None,
                Some(MilestoneStatus::InProgress),
                None
            )
            .await
            .unwrap());
        let read = store
            .get_workspace_milestone(soon.id)
            .await
            .unwrap()
            .expect("updated");
        assert_eq!(read.title, "sooner");
        assert_eq!(read.status, MilestoneStatus::InProgress);
        assert!(!store
            .update_workspace_milestone(Uuid::new_v4(), Some("x".into()), None, None, None)
            .await
            .unwrap());

        let orphan = WorkspaceMilestoneNode {
            workspace_id: Uuid::new_v4(),
            ..mk("orphan", None, MilestoneStatus::Open)
        };
        store.create_workspace_milestone(&orphan).await.unwrap();
        assert!(store
            .get_workspace_milestone(orphan.id)
            .await
            .unwrap()
            .is_none());

        store.delete_workspace_milestone(late.id).await.unwrap();
        assert!(store
            .get_workspace_milestone(late.id)
            .await
            .unwrap()
            .is_none());
        store.delete_workspace(ws.id).await.unwrap();
        assert!(
            store
                .get_workspace_milestone(undated.id)
                .await
                .unwrap()
                .is_none(),
            "cascade"
        );
    }

    /// Resources: listed by name then id per workspace; a project implementing
    /// or using one is listed by it; deleting it removes the links.
    pub async fn check_workspace_resources(store: &dyn GraphStore) {
        let ws = a_workspace("resources");
        let p = a_project("resource-user");
        store.create_workspace(&ws).await.unwrap();
        store.create_project(&p).await.unwrap();
        let mk = |name: &str| ResourceNode {
            id: Uuid::new_v4(),
            workspace_id: Some(ws.id),
            project_id: None,
            name: name.to_string(),
            resource_type: ResourceType::ApiContract,
            file_path: format!("specs/{name}.yaml"),
            url: None,
            format: None,
            version: Some("1".into()),
            description: None,
            created_at: at(2),
            updated_at: None,
            metadata: serde_json::json!({}),
        };
        let beta = mk("beta");
        let alpha = mk("alpha");
        store.create_resource(&beta).await.unwrap();
        store.create_resource(&alpha).await.unwrap();

        let names: Vec<String> = store
            .list_workspace_resources(ws.id)
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.name)
            .collect();
        assert_eq!(names, vec!["alpha", "beta"]);

        store
            .link_project_implements_resource(p.id, alpha.id)
            .await
            .unwrap();
        store
            .link_project_uses_resource(p.id, alpha.id)
            .await
            .unwrap();
        assert_eq!(
            store
                .get_resource_implementers(alpha.id)
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            store.get_resource_consumers(alpha.id).await.unwrap().len(),
            1
        );
        assert!(store
            .get_resource_consumers(beta.id)
            .await
            .unwrap()
            .is_empty());

        assert!(store
            .update_resource(alpha.id, None, None, None, Some("2".into()), None)
            .await
            .unwrap());
        assert_eq!(
            store
                .get_resource(alpha.id)
                .await
                .unwrap()
                .unwrap()
                .version
                .as_deref(),
            Some("2")
        );
        assert!(!store
            .update_resource(Uuid::new_v4(), None, None, None, None, None)
            .await
            .unwrap());

        store.delete_resource(alpha.id).await.unwrap();
        assert!(store.get_resource(alpha.id).await.unwrap().is_none());
        assert!(store
            .get_resource_implementers(alpha.id)
            .await
            .unwrap()
            .is_empty());
        store.delete_workspace(ws.id).await.unwrap();
        store.delete_project(p.id, &p.name).await.unwrap();
    }

    /// Components: upsert is keyed by (workspace, name) and returns the same id
    /// on the second call; dependencies and mappings show in the topology with
    /// their protocol and flag; deleting a component detaches it.
    pub async fn check_workspace_components(store: &dyn GraphStore) {
        let ws = a_workspace("components");
        let p = a_project("component-home");
        store.create_workspace(&ws).await.unwrap();
        store.create_project(&p).await.unwrap();
        let api = store
            .upsert_derived_component(DerivedComponentWrite {
                workspace_id: ws.id,
                name: "api".into(),
                component_type: ComponentType::Service,
                description: Some("the api".into()),
                runtime: Some("rust".into()),
                tags: vec!["core".into()],
                config: serde_json::json!({"port": 8080}),
            })
            .await
            .unwrap();
        let again = store
            .upsert_derived_component(DerivedComponentWrite {
                workspace_id: ws.id,
                name: "api".into(),
                component_type: ComponentType::Service,
                description: None,
                runtime: None,
                tags: vec![],
                config: serde_json::json!({"port": 9090}),
            })
            .await
            .unwrap();
        assert_eq!(again, api, "upsert is keyed by (workspace, name)");
        let kept = store.get_component(api).await.unwrap().expect("api");
        assert_eq!(
            kept.description.as_deref(),
            Some("the api"),
            "an absent description keeps the stored one"
        );
        assert_eq!(
            kept.runtime.as_deref(),
            Some("rust"),
            "an empty runtime keeps the stored one"
        );
        assert_eq!(
            kept.tags,
            vec!["core".to_string()],
            "tags kept when present"
        );
        assert_eq!(
            kept.config,
            serde_json::json!({"port": 9090}),
            "config always replaced"
        );

        let db = store
            .upsert_derived_component(DerivedComponentWrite {
                workspace_id: ws.id,
                name: "db".into(),
                component_type: ComponentType::Database,
                description: None,
                runtime: None,
                tags: vec![],
                config: serde_json::json!({}),
            })
            .await
            .unwrap();
        store
            .add_component_dependency(api, db, Some("sql".into()), false)
            .await
            .unwrap();
        store.map_component_to_project(api, p.id).await.unwrap();

        let topo = store.get_workspace_topology(ws.id).await.unwrap();
        let (_, project_name, deps) = topo
            .iter()
            .find(|(c, _, _)| c.id == api)
            .expect("api in the topology");
        assert_eq!(project_name.as_deref(), Some(p.name.as_str()));
        assert_eq!(deps.len(), 1);
        assert_eq!(
            (deps[0].to_id, deps[0].protocol.as_deref(), deps[0].required),
            (db, Some("sql"), false)
        );
        let (_, none, _) = topo
            .iter()
            .find(|(c, _, _)| c.id == db)
            .expect("db in the topology");
        assert!(none.is_none(), "an unmapped component has no project");

        store.remove_component_dependency(api, db).await.unwrap();
        let topo = store.get_workspace_topology(ws.id).await.unwrap();
        assert!(
            topo.iter().all(|(_, _, d)| d.is_empty()),
            "dependency removed"
        );

        assert!(store
            .update_component(
                db,
                ComponentUpdate {
                    name: Some("database".into()),
                    ..Default::default()
                }
            )
            .await
            .unwrap());
        assert_eq!(store.list_components(ws.id).await.unwrap().len(), 2);
        // The (workspace, name) identity follows the rename: the new name finds
        // the component, the old one is free and creates another.
        let derived = |name: &str| DerivedComponentWrite {
            workspace_id: ws.id,
            name: name.into(),
            component_type: ComponentType::Database,
            description: None,
            runtime: None,
            tags: vec![],
            config: serde_json::json!({}),
        };
        assert_eq!(
            store
                .upsert_derived_component(derived("database"))
                .await
                .unwrap(),
            db,
            "the renamed component is found by its new name"
        );
        let fresh = store.upsert_derived_component(derived("db")).await.unwrap();
        assert_ne!(fresh, db, "the old name no longer finds it");
        assert_eq!(store.list_components(ws.id).await.unwrap().len(), 3);
        store.delete_component(fresh).await.unwrap();

        store.delete_component(api).await.unwrap();
        assert!(store.get_component(api).await.unwrap().is_none());
        store.delete_workspace(ws.id).await.unwrap();
        store.delete_project(p.id, &p.name).await.unwrap();
    }

    // ------------------------------------------------------------------
    // The note domain (NoteStore, the ported part). Each check works on fresh
    // ids and deletes what it created. Neo4j is the reference.
    // ------------------------------------------------------------------

    /// A note with a project: every field the write path stores is set to a
    /// value that is not the default, so a lossy round-trip shows.
    pub async fn check_note_roundtrip(store: &dyn GraphStore) {
        let p = a_project("note-rt");
        store.create_project(&p).await.unwrap();
        let mut n = a_note(Some(p.id), "une note « avec accents » et\nun saut de ligne");
        n.importance = NoteImportance::High;
        n.tags = vec!["alpha".into(), "beta".into()];
        n.confidence = Some(0.25);
        n.staleness_score = 0.5;
        n.valid_from = Some(Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap());
        n.scope = NoteScope::File("src/lib.rs".into());
        store.create_note(&n).await.unwrap();
        let got = store
            .get_note(n.id)
            .await
            .unwrap()
            .expect("the note reads back");
        assert_eq!(got.id, n.id);
        assert_eq!(got.project_id, Some(p.id));
        assert_eq!(got.content, n.content);
        assert_eq!(got.importance, NoteImportance::High);
        assert_eq!(got.status, n.status);
        assert_eq!(got.tags, n.tags);
        assert_eq!(got.confidence, Some(0.25));
        assert!((got.staleness_score - 0.5).abs() < 1e-12);
        assert_eq!(got.valid_from, n.valid_from);
        assert_eq!(got.scope, NoteScope::File("src/lib.rs".into()));
        assert_eq!(got.created_at, n.created_at);
        assert_eq!(got.created_by, n.created_by);
        assert!(store.get_note(Uuid::new_v4()).await.unwrap().is_none());
        store.delete_note(n.id).await.unwrap();
        store.delete_project(p.id, &p.name).await.ok();
    }

    /// Update, rescope (both ways), delete, and the absent-note answers.
    pub async fn check_note_update_rescope_delete(store: &dyn GraphStore) {
        let p = a_project("note-urd");
        store.create_project(&p).await.unwrap();
        let n = a_note(None, "orpheline");
        store.create_note(&n).await.unwrap();
        // Global until rescoped (no global listing: a shared Neo4j may hold legacy notes).
        assert_eq!(
            store.get_note(n.id).await.unwrap().unwrap().project_id,
            None
        );
        store.rescope_note(n.id, Some(p.id)).await.unwrap();
        let (dans_p, total) = store
            .list_notes(Some(p.id), None, &NoteFilters::default())
            .await
            .unwrap();
        assert!(dans_p.iter().any(|x| x.id == n.id) && total >= 1);
        store.rescope_note(n.id, None).await.unwrap();
        assert_eq!(
            store.get_note(n.id).await.unwrap().unwrap().project_id,
            None
        );
        let u = store
            .update_note(
                n.id,
                Some("nouveau contenu".into()),
                Some(NoteImportance::Critical),
                Some(NoteStatus::NeedsReview),
                Some(vec!["t".into()]),
                Some(0.75),
            )
            .await
            .unwrap()
            .expect("the note is updated");
        assert_eq!(u.content, "nouveau contenu");
        assert_eq!(u.importance, NoteImportance::Critical);
        assert_eq!(u.status, NoteStatus::NeedsReview);
        assert_eq!(u.tags, vec!["t".to_string()]);
        assert!((u.staleness_score - 0.75).abs() < 1e-12);
        assert!(store
            .update_note(Uuid::new_v4(), Some("x".into()), None, None, None, None)
            .await
            .unwrap()
            .is_none());
        assert!(store.delete_note(n.id).await.unwrap());
        assert!(!store.delete_note(n.id).await.unwrap());
        assert!(store.get_note(n.id).await.unwrap().is_none());
        store.delete_project(p.id, &p.name).await.ok();
    }

    /// Sort by a field, a total order (id breaks ties), and paging over a tie.
    pub async fn check_note_list_order_and_paging(store: &dyn GraphStore) {
        let p = a_project("note-list");
        store.create_project(&p).await.unwrap();
        let mut ids = Vec::new();
        for (k, s) in [0.3, 0.1, 0.3, 0.9, 0.5].iter().enumerate() {
            let mut n = a_note(Some(p.id), &format!("note {k}"));
            n.staleness_score = *s;
            store.create_note(&n).await.unwrap();
            ids.push(n.id);
        }
        let f = NoteFilters {
            sort_by: Some("staleness_score".into()),
            sort_order: Some("asc".into()),
            ..Default::default()
        };
        let (liste, total) = store.list_notes(Some(p.id), None, &f).await.unwrap();
        assert_eq!(total, 5);
        let scores: Vec<f64> = liste.iter().map(|n| n.staleness_score).collect();
        assert_eq!(scores, vec![0.1, 0.3, 0.3, 0.5, 0.9]);
        // Ties on 0.3 are ordered by id: the page boundary does not move.
        let tie: Vec<Uuid> = liste
            .iter()
            .filter(|n| n.staleness_score == 0.3)
            .map(|n| n.id)
            .collect();
        let mut tie_sorted = tie.clone();
        tie_sorted.sort();
        assert_eq!(tie, tie_sorted);
        let page = |offset: i64| NoteFilters {
            sort_by: Some("staleness_score".into()),
            sort_order: Some("asc".into()),
            limit: Some(2),
            offset: Some(offset),
            ..Default::default()
        };
        let (p1, _) = store.list_notes(Some(p.id), None, &page(0)).await.unwrap();
        let (p2, _) = store.list_notes(Some(p.id), None, &page(2)).await.unwrap();
        let joined: Vec<Uuid> = p1.iter().chain(p2.iter()).map(|n| n.id).collect();
        assert_eq!(
            joined,
            liste.iter().take(4).map(|n| n.id).collect::<Vec<_>>()
        );
        for id in ids {
            store.delete_note(id).await.unwrap();
        }
        store.delete_project(p.id, &p.name).await.ok();
    }

    /// Links to a project and to a note: created, read back with their hashes,
    /// updated in place (MERGE), removed; a link to an absent target is a no-op.
    pub async fn check_note_entity_links(store: &dyn GraphStore) {
        let p = a_project("note-links");
        store.create_project(&p).await.unwrap();
        let n = a_note(Some(p.id), "lie au projet");
        let m = a_note(Some(p.id), "lie à une note");
        store.create_note(&n).await.unwrap();
        store.create_note(&m).await.unwrap();
        store
            .link_note_to_entity(
                n.id,
                &EntityType::Project,
                &p.id.to_string(),
                Some("sig"),
                None,
            )
            .await
            .unwrap();
        let for_p = store
            .get_notes_for_entity(&EntityType::Project, &p.id.to_string())
            .await
            .unwrap();
        assert_eq!(ids(&for_p, |x| x.id), vec![n.id]);
        // Updated in place: one link, new hash.
        store
            .link_note_to_entity(
                n.id,
                &EntityType::Project,
                &p.id.to_string(),
                Some("sig2"),
                Some("body"),
            )
            .await
            .unwrap();
        let anchors = store.get_note_anchors(n.id).await.unwrap();
        assert_eq!(anchors.len(), 1, "MERGE: one edge per pair");
        assert_eq!(anchors[0].signature_hash.as_deref(), Some("sig2"));
        assert_eq!(anchors[0].body_hash.as_deref(), Some("body"));
        assert!(anchors[0].is_valid);
        store
            .link_note_to_entity(n.id, &EntityType::Note, &m.id.to_string(), None, None)
            .await
            .unwrap();
        let for_m = store
            .get_notes_for_entity(&EntityType::Note, &m.id.to_string())
            .await
            .unwrap();
        assert_eq!(ids(&for_m, |x| x.id), vec![n.id]);
        // An absent target: no edge, no error (Neo4j's MATCH finds nothing).
        store
            .link_note_to_entity(
                n.id,
                &EntityType::Project,
                &Uuid::new_v4().to_string(),
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(store.get_note_anchors(n.id).await.unwrap().len(), 2);
        store
            .unlink_note_from_entity(n.id, &EntityType::Note, &m.id.to_string())
            .await
            .unwrap();
        assert!(store
            .get_notes_for_entity(&EntityType::Note, &m.id.to_string())
            .await
            .unwrap()
            .is_empty());
        store.delete_note(n.id).await.unwrap();
        store.delete_note(m.id).await.unwrap();
        store.delete_project(p.id, &p.name).await.ok();
    }

    /// Supersession archives the old note, names its successor, and a
    /// confirmation does not resurrect it; confirming the successor makes it active.
    pub async fn check_note_supersede_and_confirm(store: &dyn GraphStore) {
        let p = a_project("note-supersede");
        store.create_project(&p).await.unwrap();
        let old = a_note(Some(p.id), "ancienne");
        let new = a_note(Some(p.id), "nouvelle");
        store.create_note(&old).await.unwrap();
        store.create_note(&new).await.unwrap();
        store.supersede_note(old.id, new.id).await.unwrap();
        let o = store.get_note(old.id).await.unwrap().unwrap();
        let w = store.get_note(new.id).await.unwrap().unwrap();
        assert_eq!(o.status, NoteStatus::Archived);
        assert_eq!(o.superseded_by, Some(new.id));
        assert_eq!(w.supersedes, Some(old.id));
        let c = store.confirm_note(old.id, "alice").await.unwrap().unwrap();
        assert_eq!(
            c.status,
            NoteStatus::Archived,
            "a superseded note is not resurrected"
        );
        assert_eq!(c.human_confirmed_by.as_deref(), Some("alice"));
        assert_eq!(c.staleness_score, 0.0);
        let cw = store
            .confirm_note(new.id, "system/heuristic")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(cw.status, NoteStatus::Active);
        assert_eq!(cw.human_confirmed_by, None, "a heuristic never signs");
        assert_eq!(cw.staleness_score, 0.0);
        assert!(store
            .confirm_note(Uuid::new_v4(), "alice")
            .await
            .unwrap()
            .is_none());
        store.delete_note(old.id).await.unwrap();
        store.delete_note(new.id).await.unwrap();
        store.delete_project(p.id, &p.name).await.ok();
    }

    /// The review queue: only `stale` and `needs_review` notes of the project.
    pub async fn check_note_needing_review(store: &dyn GraphStore) {
        let p = a_project("note-review");
        store.create_project(&p).await.unwrap();
        let mut a = a_note(Some(p.id), "stale");
        a.status = NoteStatus::Stale;
        let mut b = a_note(Some(p.id), "active");
        b.status = NoteStatus::Active;
        let mut c = a_note(Some(p.id), "review");
        c.status = NoteStatus::NeedsReview;
        for n in [&a, &b, &c] {
            store.create_note(n).await.unwrap();
        }
        let liste = store
            .get_notes_needing_review(Some(p.id), None)
            .await
            .unwrap();
        let mut got = ids(&liste, |x| x.id);
        got.sort();
        let mut want = vec![a.id, c.id];
        want.sort();
        assert_eq!(got, want);
        for n in [a, b, c] {
            store.delete_note(n.id).await.unwrap();
        }
        store.delete_project(p.id, &p.name).await.ok();
    }

    // ------------------------------------------------------------------
    // Plans: the core the obrain backend ports (`plan_checks!`).
    // ------------------------------------------------------------------

    /// `ORDER BY p.priority DESC, p.created_at DESC, p.id DESC`.
    fn plan_order(a: &PlanNode, b: &PlanNode) -> std::cmp::Ordering {
        b.priority
            .cmp(&a.priority)
            .then_with(|| b.created_at.cmp(&a.created_at))
            .then_with(|| b.id.cmp(&a.id))
    }

    /// The lists of a project's plans: only the project's, only the active ones
    /// where the contract says so, in the total order `priority DESC, created_at
    /// DESC, id DESC` (ties on the priority AND on `created_at`), the status filter
    /// as Neo4j normalises it (`in_progress` and `Draft` match, an unknown value
    /// matches nothing), and the pagination with its total counted before the page.
    pub async fn check_plan_project_lists(store: &dyn GraphStore) {
        let p = a_project("plan-lists");
        let q = a_project("plan-lists-other");
        store.create_project(&p).await.unwrap();
        store.create_project(&q).await.unwrap();

        let with = |title: &str, priority: i32, t: i64, status: PlanStatus| {
            let mut x = a_plan(p.id, title, priority, at(t));
            x.status = status;
            x
        };
        // Inserted in an order that is none of the expected orders.
        let e = with("e", 1, 0, PlanStatus::Cancelled);
        let a = with("a", 5, 10, PlanStatus::Draft);
        let d = with("d", 9, 0, PlanStatus::Completed);
        let b = with("b", 5, 10, PlanStatus::InProgress);
        let c = with("c", 5, 20, PlanStatus::Approved);
        let mut other = a_plan(q.id, "other", 3, at(5));
        other.status = PlanStatus::Draft;
        for x in [&e, &a, &d, &b, &c, &other] {
            store.create_plan(x).await.unwrap();
        }

        let mut all = vec![a.clone(), b.clone(), c.clone(), d.clone(), e.clone()];
        all.sort_by(plan_order);
        let active: Vec<Uuid> = all
            .iter()
            .filter(|x| {
                matches!(
                    x.status,
                    PlanStatus::Draft | PlanStatus::Approved | PlanStatus::InProgress
                )
            })
            .map(|x| x.id)
            .collect();
        assert_eq!(active.len(), 3);

        assert_eq!(
            ids(&store.list_project_plans(p.id).await.unwrap(), |x| x.id),
            active,
            "list_project_plans: the active plans of the project, priority DESC, created_at DESC, id DESC"
        );
        assert_eq!(store.count_project_plans(p.id).await.unwrap(), 3);
        assert_eq!(
            ids(&store.list_project_plans(q.id).await.unwrap(), |x| x.id),
            vec![other.id]
        );
        assert!(store
            .list_project_plans(Uuid::new_v4())
            .await
            .unwrap()
            .is_empty());
        assert_eq!(store.count_project_plans(Uuid::new_v4()).await.unwrap(), 0);

        let ours: HashSet<Uuid> = [a.id, b.id, c.id, d.id, e.id, other.id].into();
        let mut expected_active: Vec<PlanNode> =
            vec![a.clone(), b.clone(), c.clone(), other.clone()];
        expected_active.sort_by(plan_order);
        let listed: Vec<Uuid> = store
            .list_active_plans()
            .await
            .unwrap()
            .into_iter()
            .map(|x| x.id)
            .filter(|id| ours.contains(id))
            .collect();
        assert_eq!(
            listed,
            ids(&expected_active, |x| x.id),
            "list_active_plans: every project's active plans, in the same order"
        );

        let (page, total) = store
            .list_plans_for_project(p.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(
            (ids(&page, |x| x.id), total),
            (ids(&all, |x| x.id), 5),
            "list_plans_for_project without a filter: every status, the same order"
        );
        let (page, total) = store
            .list_plans_for_project(p.id, None, 2, 1)
            .await
            .unwrap();
        assert_eq!(
            (ids(&page, |x| x.id), total),
            (ids(&all[1..3], |x| x.id), 5),
            "SKIP 1 LIMIT 2, the total counted before the page"
        );
        let filter = vec![
            "in_progress".to_string(),
            "Draft".to_string(),
            "inconnu".to_string(),
        ];
        let mut drafted = vec![a.clone(), b.clone()];
        drafted.sort_by(plan_order);
        let (page, total) = store
            .list_plans_for_project(p.id, Some(filter), 10, 0)
            .await
            .unwrap();
        assert_eq!(
            (ids(&page, |x| x.id), total),
            (ids(&drafted, |x| x.id), 2),
            "the filter: in_progress and Draft match, an unknown value matches nothing"
        );
        let (page, total) = store
            .list_plans_for_project(Uuid::new_v4(), None, 10, 0)
            .await
            .unwrap();
        assert!(page.is_empty() && total == 0);

        for x in [&e, &a, &d, &b, &c, &other] {
            store.delete_plan(x.id).await.unwrap();
        }
        store.delete_project(p.id, &p.name).await.ok();
        store.delete_project(q.id, &q.name).await.ok();
    }

    /// A plan reads back as written; an update and a status change are read back;
    /// on an absent plan they are silent no-ops; a deleted plan is gone, and
    /// deleting it again is not an error.
    pub async fn check_plan_update_and_delete(store: &dyn GraphStore) {
        let p = a_project("plan-update");
        store.create_project(&p).await.unwrap();
        let x = a_plan(p.id, "before", 2, at(0));
        store.create_plan(&x).await.unwrap();
        let back = store.get_plan(x.id).await.unwrap().expect("read by id");
        assert_eq!(
            serde_json::to_value(&back).unwrap(),
            serde_json::to_value(&x).unwrap(),
            "the plan reads back identical"
        );

        store
            .update_plan(
                x.id,
                &project_orchestrator::plan::models::UpdatePlanRequest {
                    title: Some("after".to_string()),
                    description: None,
                    priority: Some(7),
                },
            )
            .await
            .unwrap();
        store
            .update_plan_status(x.id, PlanStatus::InProgress)
            .await
            .unwrap();
        let back = store.get_plan(x.id).await.unwrap().unwrap();
        assert_eq!(
            (
                back.title.as_str(),
                back.description.as_str(),
                back.priority,
                back.status
            ),
            ("after", "conformance plan", 7, PlanStatus::InProgress)
        );

        let absent = Uuid::new_v4();
        store
            .update_plan_status(absent, PlanStatus::Completed)
            .await
            .unwrap();
        store
            .update_plan(
                absent,
                &project_orchestrator::plan::models::UpdatePlanRequest {
                    title: Some("x".to_string()),
                    description: None,
                    priority: None,
                },
            )
            .await
            .unwrap();
        assert!(store.get_plan(absent).await.unwrap().is_none());

        store.delete_plan(x.id).await.unwrap();
        assert!(store.get_plan(x.id).await.unwrap().is_none());
        assert_eq!(store.count_project_plans(p.id).await.unwrap(), 0);
        store.delete_plan(x.id).await.unwrap();
        store.delete_project(p.id, &p.name).await.ok();
    }
}

// ===============================================================// Factories: one per backend.
// ========================================================/// Opens a store for one check. `None` means the backend is unavailable here:
/// the check is skipped (the factory decides whether that is allowed).
#[async_trait]
pub trait StoreFactory: Send + Sync {
    fn backend(&self) -> &'static str;
    async fn open(&self) -> Option<Arc<dyn GraphStore>>;
}

/// A fresh in-memory store per check.
pub struct MockFactory;

#[async_trait]
impl StoreFactory for MockFactory {
    fn backend(&self) -> &'static str {
        "mock"
    }
    async fn open(&self) -> Option<Arc<dyn GraphStore>> {
        Some(Arc::new(MockGraphStore::new()))
    }
}

/// The Neo4j named by the environment (see the module doc). A connection per
/// check, schema initialised by `Neo4jClient::new`.
pub struct Neo4jFactory;

#[async_trait]
impl StoreFactory for Neo4jFactory {
    fn backend(&self) -> &'static str {
        "neo4j"
    }
    async fn open(&self) -> Option<Arc<dyn GraphStore>> {
        let common::harness::Neo4jTarget {
            uri,
            user,
            password,
        } = common::harness::neo4j_target();
        match Neo4jClient::new(&uri, &user, &password).await {
            Ok(client) => Some(Arc::new(client)),
            Err(err) => {
                assert!(
                    std::env::var("HARNESS_NEO4J_REQUIRED").is_err(),
                    "HARNESS_NEO4J_REQUIRED is set but no Neo4j answers at {uri}: {err}"
                );
                eprintln!("skipping the Neo4j conformance check: no Neo4j at {uri} ({err})");
                None
            }
        }
    }
}

/// A fresh obrain base per check, in a temporary directory removed with the
/// store (never a configured path: a conformance run opens nothing that
/// exists). Only the ported domains are instantiated for it — see
/// `project_checks!` below.
#[cfg(feature = "obrain-backend")]
pub struct ObrainFactory;

#[cfg(feature = "obrain-backend")]
#[async_trait]
impl StoreFactory for ObrainFactory {
    fn backend(&self) -> &'static str {
        "obrain"
    }
    async fn open(&self) -> Option<Arc<dyn GraphStore>> {
        let store = tokio::task::spawn_blocking(
            project_orchestrator::obrain::ObrainGraphStore::open_temporary,
        )
        .await
        .expect("the blocking open ran")
        .expect("a temporary obrain base opens");
        Some(Arc::new(store))
    }
}

// ===============================================================// Expected divergences (P0.2 fixes them; the mock converges towards Neo4j).
// ========================================================/// A known gap between a backend and the contract the checks state.
pub struct Divergence {
    pub backend: &'static str,
    pub check: &'static str,
    /// What differs, precisely enough to fix it.
    pub why: &'static str,
    /// `true`: the check fails every time on this backend, so a pass means the
    /// gap is closed and this entry must go (the ratchet). `false`: the gap is
    /// a race the check only sometimes catches; a pass proves nothing.
    pub deterministic: bool,
}

const fn mock(check: &'static str, why: &'static str) -> Divergence {
    Divergence {
        backend: "mock",
        check,
        why,
        deterministic: true,
    }
}

const fn neo4j(check: &'static str, why: &'static str, deterministic: bool) -> Divergence {
    Divergence {
        backend: "neo4j",
        check,
        why,
        deterministic,
    }
}

/// The check still runs on that backend; its failure is printed as expected and
/// does not fail the suite. The `mock` entries are what P0.2 fixes (the mock
/// converges towards Neo4j). The `neo4j` entries are races of the Neo4j layer
/// itself, found by this suite, where the mock is right and Neo4j is not; they
/// are not for P0.2 and are not hidden either.
const EXPECTED_DIVERGENCES: &[Divergence] = &[
    Divergence {
        backend: "mock",
        check: "check_workspace_list_order",
        why: "list_workspaces returns the mock's map order (it changes from one run to the next); Neo4j: name then id",
        deterministic: false,
    },
    mock(
        "check_workspace_slug_rules",
        "the mock stores a slug without validating its shape and without the CONFLICT check on update (Neo4j: refuses both)",
    ),
    mock(
        "check_workspace_membership",
        "add_project_to_workspace has no MERGE: adding the same project twice lists it twice (Neo4j: once)",
    ),
    mock(
        "check_workspace_milestones",
        "list_workspace_milestones does not order by target_date, title, id (Neo4j: that order, absent dates first)",
    ),
    mock(
        "check_workspace_resources",
        "list_workspace_resources returns the mock's map order, not name then id (Neo4j: name then id)",
    ),
    mock(
        "check_workspace_components",
        "get_workspace_topology names the mapped project by its slug, not its name (Neo4j: p.name)",
    ),
    mock(
        "check_uniqueness",
        "upsert_file does not link the file to its project (list_project_files/count_project_files see nothing); \
         create_alert inserts by id without the dedup_key MERGE (two alerts, occurrence_count 1 each, both ids readable)",
    ),
    mock(
        "check_merge_idempotence",
        "upsert_file/batch_upsert_files do not link the file to its project (count_project_files 0 after an upsert); \
         add_task_dependency pushes a duplicate edge on repeat",
    ),
    mock(
        "check_absent_entity_matrix",
        "increment_frustration/decrement_frustration/get_frustration on an unknown task are Err(\"Task not found\") where Neo4j answers Ok(0.0)",
    ),
    mock(
        "check_cascade_delete_project",
        "files upserted with a project_id are not counted under it (count_project_files 0 before the delete); \
         delete_project neither removes the File nodes by path nor archives the project's notes (status archived, project_id None)",
    ),
    mock(
        "check_sorting",
        "list_plans_filtered ignores sort_by and sort_order (HashMap order) where Neo4j sorts by created_at DESC by default, priority/title/status on request",
    ),
    mock(
        "check_plan_project_lists",
        "list_project_plans returns the plans in insertion order and list_active_plans in map order (Neo4j: priority DESC, \
         created_at DESC, id DESC); list_plans_for_project sorts by created_at alone and compares the filter with the \
         snake_case spelling, so `Draft` matches nothing (Neo4j: plan_status_values, then the stored PascalCase)",
    ),
    mock(
        "check_pagination",
        "list_notes pages over HashMap order where Neo4j pages over created_at DESC (the pages are right-sized but not in order)",
    ),
    mock(
        "check_note_filters",
        "a blank search (\"   \") filters on the literal whitespace where Neo4j ignores a search that is empty once trimmed",
    ),
    mock(
        "check_null_vs_empty_and_timestamp_roundtrip",
        "the mock stores what it is given: Project.description None reads None (Neo4j: Some(\"\")), and the same for Task.title \"\" (Neo4j: None), \
         Note.last_confirmed_at None (Neo4j: created_at) and Note.last_confirmed_by None (Neo4j: Some(\"\"))",
    ),
    mock(
        "check_project_absent_description_reads_empty",
        "the mock stores the description it is given: None at creation reads None, and update_project(description: Some(None)) \
         reads None, where Neo4j stores and reads \"\" in both cases",
    ),
    mock(
        "check_project_creation_keeps_only_creation_fields",
        "create_project keeps the whole ProjectNode (last_synced, analytics_computed_at, last_co_change_computed_at, \
         default_note_energy, scaffolding_override) where Neo4j stores only the creation fields",
    ),
    mock(
        "check_project_concurrent_same_slug",
        "create_project has no uniqueness check, so all eight concurrent creations of one slug succeed (same gap as check_project_unique_id_and_slug)",
    ),
    mock(
        "check_project_unique_id_and_slug",
        "create_project inserts by id with no uniqueness check: a taken id overwrites the first project and a taken slug is \
         accepted, where Neo4j refuses both (constraints project_id and project_slug)",
    ),
    mock(
        "check_vector_ranking_with_project_filter",
        "vector_search_notes returns the raw cosine (orthogonal = 0.0) where Neo4j's index scores (1 + cosine) / 2 (orthogonal = 0.5); \
         the same min_similarity keeps different notes on the two backends",
    ),
    neo4j(
        "check_concurrent_create_protocol_run",
        "create_protocol_run checks `NOT EXISTS { running root run }` and CREATEs without a lock on the Protocol node first, \
         so concurrent transactions all pass the check (6 of 8, then 8 of 8 won on 2026-10-20); the mock takes the write lock across check and insert",
        false,
    ),
    neo4j(
        "check_sorting_tie_break_paging",
        "list_plans_filtered orders by one key with no tie-break (ORDER BY priority alone), so SKIP/LIMIT pages across equal \
         priorities repeat a plan and skip another (seen on 2026-10-20); the mock pages a stable list",
        false,
    ),
    neo4j(
        "check_vector_project_filter_recall",
        "vector_search_notes fetches 3 × limit index candidates and filters by project_id afterwards, so a project's notes are \
         lost whenever another project holds 3 × limit closer notes (A gets nothing for limit 2 behind ten notes of B); the mock filters first",
        true,
    ),
    neo4j(
        "check_project_list_order",
        "list_projects decodes EVERY Project node of the instance and fails as a whole (« The property does not exist ») \
         as soon as one of them lacks name, slug or created_at: a bare node left by another suite on the same Neo4j \
         (tests/neo4j_store_tests.rs left two until 2026-11-11; seen red in CI and on the shared harness that day) \
         breaks the listing for everyone. Not a property of the order: on a clean instance the check passes. \
         While this entry stands, Neo4j is not a strict witness of the order of list_projects",
        false,
    ),
];

/// Runs one check against one backend, applying [`EXPECTED_DIVERGENCES`].
async fn run(
    factory: &dyn StoreFactory,
    check: &str,
    body: for<'a> fn(&'a dyn GraphStore) -> BoxFuture<'a, ()>,
) {
    let Some(store) = factory.open().await else {
        return;
    };
    let backend = factory.backend();
    let expected = EXPECTED_DIVERGENCES
        .iter()
        .find(|d| d.backend == backend && d.check == check);
    let outcome = std::panic::AssertUnwindSafe(body(&*store))
        .catch_unwind()
        .await;
    match (outcome, expected) {
        (Ok(()), None) => {}
        (Err(panic), None) => std::panic::resume_unwind(panic),
        (Err(_), Some(d)) => {
            eprintln!("EXPECTED DIVERGENCE on {backend}::{check}: {}", d.why);
        }
        (Ok(()), Some(d)) if d.deterministic => panic!(
            "{backend}::{check} now conforms (listed divergence: {}); \
             remove it from EXPECTED_DIVERGENCES",
            d.why
        ),
        (Ok(()), Some(_)) => {
            eprintln!(
                "listed race on {backend}::{check} did not show this time (a pass proves nothing)"
            );
        }
    }
}

/// One `#[tokio::test]` per check for the given backend, in a module named
/// after it: `mock::check_sorting`, `neo4j::check_sorting`, ...
macro_rules! conformance_suite {
    ($backend:ident, $factory:ident, [ $( $check:ident ),* $(,)? ]) => {
        mod $backend {
            $(
                #[tokio::test]
                async fn $check() {
                    fn body<'a>(s: &'a dyn $crate::GraphStore) -> $crate::BoxFuture<'a, ()> {
                        Box::pin($crate::conformance::$check(s))
                    }
                    $crate::run(&$crate::$factory, stringify!($check), body).await;
                }
            )*
        }
    };
}

macro_rules! all_checks {
    ($backend:ident, $factory:ident) => {
        conformance_suite!(
            $backend,
            $factory,
            [
                check_uniqueness,
                check_merge_idempotence,
                check_protocol_scope_rule,
                check_cas_protocol_run,
                check_concurrent_create_protocol_run,
                check_external_event_journal,
                check_external_delivery_claim,
                check_concurrent_external_delivery_claim,
                check_external_delivery_release,
                check_external_event_purge,
                check_atomic_increments,
                check_absent_entity_matrix,
                check_cascade_delete_project,
                check_sorting,
                check_sorting_tie_break_paging,
                check_pagination,
                check_note_filters,
                check_null_vs_empty_and_timestamp_roundtrip,
                check_traversal_run_tree,
                check_vector_ranking_with_project_filter,
                check_vector_project_filter_recall,
                check_numeric_typing_preserved,
                check_datetime_roundtrip_and_arithmetic,
                check_ordering_is_deterministic_across_pages,
            ]
        );
    };
}

all_checks!(mock, MockFactory);
all_checks!(neo4j, Neo4jFactory);

/// The project domain on its own: the checks that call nothing but
/// `ProjectStore`. Instantiated per backend like the suite above, in modules
/// of their own — and it is the suite a backend ported one domain at a time
/// runs first: `obrain_project::*` (feature `obrain-backend`) runs the SAME
/// checks, on the same fixtures, as `mock_project::*` and `neo4j_project::*`.
/// The next ported domain adds its checks here, not a second suite.
macro_rules! project_checks {
    ($backend:ident, $factory:ident) => {
        conformance_suite!(
            $backend,
            $factory,
            [
                check_project_roundtrip,
                check_project_list_order,
                check_project_update,
                check_project_absent_description_reads_empty,
                check_project_creation_keeps_only_creation_fields,
                check_project_timestamps,
                check_project_delete,
                check_project_unique_id_and_slug,
                check_project_concurrent_same_slug,
                check_project_path_policy,
            ]
        );
    };
}

macro_rules! workspace_checks {
    ($backend:ident, $factory:ident) => {
        conformance_suite!(
            $backend,
            $factory,
            [
                check_workspace_roundtrip,
                check_workspace_list_order,
                check_workspace_slug_rules,
                check_workspace_membership,
                check_workspace_milestones,
                check_workspace_resources,
                check_workspace_components,
            ]
        );
    };
}

workspace_checks!(mock_workspace, MockFactory);
workspace_checks!(neo4j_workspace, Neo4jFactory);
#[cfg(feature = "obrain-backend")]
workspace_checks!(obrain_workspace, ObrainFactory);

project_checks!(mock_project, MockFactory);
project_checks!(neo4j_project, Neo4jFactory);
#[cfg(feature = "obrain-backend")]
project_checks!(obrain_project, ObrainFactory);

/// The federation domain: the checks that call only `FederationStore`, on the
/// same fixtures for every backend ported to it.
macro_rules! federation_checks {
    ($backend:ident, $factory:ident) => {
        conformance_suite!(
            $backend,
            $factory,
            [
                check_mcp_roundtrip,
                check_mcp_list_order,
                check_mcp_status_and_deletion,
            ]
        );
    };
}

federation_checks!(mock_federation, MockFactory);
federation_checks!(neo4j_federation, Neo4jFactory);
#[cfg(feature = "obrain-backend")]
federation_checks!(obrain_federation, ObrainFactory);

/// The release domain: releases, environments, deployments, milestones and the
/// project list of the release store. Same checks, same fixtures, on each backend
/// that has ported them (see `release_checks!`).
macro_rules! release_checks {
    ($backend:ident, $factory:ident) => {
        conformance_suite!(
            $backend,
            $factory,
            [
                check_release_roundtrip,
                check_release_list_order,
                check_environment_lifecycle,
                check_deployment_matrix,
                check_milestone_lifecycle,
                check_project_list_filtered,
            ]
        );
    };
}

release_checks!(mock_release, MockFactory);
release_checks!(neo4j_release, Neo4jFactory);
#[cfg(feature = "obrain-backend")]
release_checks!(obrain_release, ObrainFactory);

/// The plan domain: the core the obrain backend ports (plans and the lists of a
/// project's plans), on the same fixtures for every backend.
macro_rules! plan_checks {
    ($backend:ident, $factory:ident) => {
        conformance_suite!(
            $backend,
            $factory,
            [check_plan_project_lists, check_plan_update_and_delete,]
        );
    };
}

plan_checks!(mock_plan, MockFactory);
plan_checks!(neo4j_plan, Neo4jFactory);
#[cfg(feature = "obrain-backend")]
plan_checks!(obrain_plan, ObrainFactory);

/// What only the obrain adapter does: the refusals of the workspace methods it
/// does not port are named (a `NonPorte` downcast), and what it writes is read
/// back, identical, after a close and an open of the same base (the index is
/// rebuilt from the base at open).
#[cfg(feature = "obrain-backend")]
mod obrain_workspace_specifique {
    use chrono::Utc;
    use project_orchestrator::neo4j::models::{
        ComponentType, DerivedComponentWrite, MilestoneStatus, ProjectNode, ProjectProfile,
        ResourceNode, ResourceType, WorkspaceMilestoneNode, WorkspaceNode,
    };
    use project_orchestrator::neo4j::traits::{ProjectStore, WorkspaceStore};
    use project_orchestrator::obrain::{
        Creation, Interruption, NonPorte, ObrainGraphStore, ValeurBrute,
    };
    use uuid::Uuid;

    fn erreur<T>(r: anyhow::Result<T>) -> anyhow::Error {
        match r {
            Err(e) => e,
            Ok(_) => panic!("this method must refuse on obrain"),
        }
    }

    #[tokio::test]
    async fn refuses_the_unported_methods_by_name() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let id = Uuid::new_v4();
        let cas: Vec<(anyhow::Error, &str)> = vec![
            (
                erreur(
                    store
                        .list_all_workspace_milestones_filtered(None, None, 10, 0)
                        .await,
                ),
                "WorkspaceStore::list_all_workspace_milestones_filtered",
            ),
            (
                erreur(store.count_all_workspace_milestones(None, None).await),
                "WorkspaceStore::count_all_workspace_milestones",
            ),
            (
                erreur(store.add_task_to_workspace_milestone(id, id).await),
                "WorkspaceStore::add_task_to_workspace_milestone",
            ),
            (
                erreur(store.remove_task_from_workspace_milestone(id, id).await),
                "WorkspaceStore::remove_task_from_workspace_milestone",
            ),
            (
                erreur(store.link_plan_to_workspace_milestone(id, id).await),
                "WorkspaceStore::link_plan_to_workspace_milestone",
            ),
            (
                erreur(store.unlink_plan_from_workspace_milestone(id, id).await),
                "WorkspaceStore::unlink_plan_from_workspace_milestone",
            ),
            (
                erreur(store.get_workspace_milestone_progress(id).await),
                "WorkspaceStore::get_workspace_milestone_progress",
            ),
            (
                erreur(store.get_workspace_milestone_tasks(id).await),
                "WorkspaceStore::get_workspace_milestone_tasks",
            ),
            (
                erreur(store.get_workspace_milestone_steps(id).await),
                "WorkspaceStore::get_workspace_milestone_steps",
            ),
            (
                erreur(store.compute_coupling_matrix(id).await),
                "WorkspaceStore::compute_coupling_matrix",
            ),
        ];
        for (err, nom) in cas {
            assert_eq!(
                err.downcast_ref::<NonPorte>(),
                Some(&NonPorte(nom)),
                "{nom}"
            );
        }
    }

    #[tokio::test]
    async fn workspace_graph_survives_close_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let now = Utc::now();
        let p = ProjectNode {
            id: Uuid::new_v4(),
            name: "reopen project".into(),
            slug: format!("reopen-{}", Uuid::new_v4().simple()),
            root_path: String::new(),
            description: None,
            created_at: now,
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
            profile: ProjectProfile::default(),
        };
        let ws = WorkspaceNode {
            id: Uuid::new_v4(),
            name: "reopen workspace".into(),
            slug: format!("reopen-ws-{}", Uuid::new_v4().simple()),
            description: Some("kept".into()),
            created_at: now,
            updated_at: None,
            metadata: serde_json::json!({"k": 1}),
        };
        let jalon = WorkspaceMilestoneNode {
            id: Uuid::new_v4(),
            workspace_id: ws.id,
            title: "jalon".into(),
            description: Some("décrit".into()),
            status: MilestoneStatus::InProgress,
            target_date: Some(now),
            closed_at: None,
            created_at: now,
            tags: vec!["a".into()],
        };
        let res = ResourceNode {
            id: Uuid::new_v4(),
            workspace_id: Some(ws.id),
            project_id: None,
            name: "contrat".into(),
            resource_type: ResourceType::ApiContract,
            file_path: "api.yaml".into(),
            url: Some("https://example.invalid/api".into()),
            format: Some("openapi".into()),
            version: Some("3".into()),
            description: Some("le contrat".into()),
            created_at: now,
            updated_at: None,
            metadata: serde_json::json!({}),
        };

        let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
        store.create_project(&p).await.unwrap();
        store.create_workspace(&ws).await.unwrap();
        store.add_project_to_workspace(ws.id, p.id).await.unwrap();
        store.create_workspace_milestone(&jalon).await.unwrap();
        store.create_resource(&res).await.unwrap();
        store
            .link_project_implements_resource(p.id, res.id)
            .await
            .unwrap();
        let mut comps = Vec::new();
        for name in ["api", "db"] {
            let id = store
                .upsert_derived_component(DerivedComponentWrite {
                    workspace_id: ws.id,
                    name: name.into(),
                    component_type: ComponentType::Service,
                    description: None,
                    runtime: None,
                    tags: vec![],
                    config: serde_json::json!({}),
                })
                .await
                .unwrap();
            comps.push((name, id));
        }
        let c_api = comps[0].1;
        let c_db = comps[1].1;
        store
            .add_component_dependency(c_api, c_db, Some("sql".into()), false)
            .await
            .unwrap();
        store.map_component_to_project(c_api, p.id).await.unwrap();
        store.close().await.unwrap();

        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(store.rapport_d_ouverture().brouillons_supprimes, 0);
        assert_eq!(
            serde_json::to_value(store.get_workspace(ws.id).await.unwrap().unwrap()).unwrap(),
            serde_json::to_value(&ws).unwrap(),
            "the workspace reads back identical"
        );
        assert_eq!(
            store
                .list_workspace_projects(ws.id)
                .await
                .unwrap()
                .iter()
                .map(|x| x.id)
                .collect::<Vec<_>>(),
            vec![p.id],
            "membership comes back from the edges"
        );
        // Every field, the list of tags and the dates included: a count would not
        // see a status, a date or a type lost by the round-trip.
        assert_eq!(
            store
                .list_workspace_milestones(ws.id)
                .await
                .unwrap()
                .iter()
                .map(|m| serde_json::to_value(m).unwrap())
                .collect::<Vec<_>>(),
            vec![serde_json::to_value(&jalon).unwrap()],
            "the milestone reads back identical"
        );
        assert_eq!(
            store
                .list_workspace_resources(ws.id)
                .await
                .unwrap()
                .iter()
                .map(|r| serde_json::to_value(r).unwrap())
                .collect::<Vec<_>>(),
            vec![serde_json::to_value(&res).unwrap()],
            "the resource reads back identical"
        );
        assert_eq!(
            store.get_resource_implementers(res.id).await.unwrap().len(),
            1
        );
        let topo = store.get_workspace_topology(ws.id).await.unwrap();
        let (_, nom, deps) = topo.iter().find(|(c, _, _)| c.id == c_api).unwrap();
        assert_eq!(nom.as_deref(), Some("reopen project"));
        assert_eq!(deps.len(), 1);
        // `false`, not the default `true` a missing edge property reads as: the
        // property is written and replayed.
        assert_eq!(
            (deps[0].to_id, deps[0].protocol.as_deref(), deps[0].required),
            (c_db, Some("sql"), false)
        );
        store.close().await.unwrap();
    }

    fn espace() -> WorkspaceNode {
        WorkspaceNode {
            id: Uuid::new_v4(),
            name: "ws".into(),
            slug: format!("ws-{}", Uuid::new_v4().simple()),
            description: None,
            created_at: Utc::now(),
            updated_at: None,
            metadata: serde_json::json!({}),
        }
    }

    fn jalon_de(ws: Uuid) -> WorkspaceMilestoneNode {
        WorkspaceMilestoneNode {
            id: Uuid::new_v4(),
            workspace_id: ws,
            title: "jalon".into(),
            description: None,
            status: MilestoneStatus::Open,
            target_date: None,
            closed_at: None,
            created_at: Utc::now(),
            tags: vec![],
        }
    }

    /// The workspace node is gone from the engine while the index holds it (an
    /// index and a base that disagree): `delete_workspace` refuses and deletes
    /// NOTHING. Its children are still there, before and after a reopen.
    #[tokio::test]
    async fn delete_workspace_writes_nothing_when_it_refuses() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let ws = espace();
        let j = jalon_de(ws.id);
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_workspace(&ws).await.unwrap();
            store.create_workspace_milestone(&j).await.unwrap();
            assert!(store.supprimer_noeud_brut_pour_test("Workspace", ws.id));
            let e = store.delete_workspace(ws.id).await.unwrap_err();
            assert!(format!("{e:#}").contains("rien n'a été supprimé"), "{e:#}");
            assert!(store.get_workspace_milestone(j.id).await.unwrap().is_some());
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert!(
            store.get_workspace_milestone(j.id).await.unwrap().is_some(),
            "the refused deletion deleted the child"
        );
    }

    /// `delete_workspace` cut after its marker and one child: the next open
    /// finishes it rather than serving the workspace with part of its children.
    #[tokio::test]
    async fn a_workspace_deletion_cut_half_way_is_finished_at_open() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let ws = espace();
        let jalons: Vec<WorkspaceMilestoneNode> = (0..3).map(|_| jalon_de(ws.id)).collect();
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_workspace(&ws).await.unwrap();
            for j in &jalons {
                store.create_workspace_milestone(j).await.unwrap();
            }
            // Step 1 is the marker, step 2 the first child deleted.
            store.interrompre_au_prochain(Interruption::ApresEtape(2));
            assert!(store.delete_workspace(ws.id).await.is_err());
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(store.rapport_d_ouverture().suppressions_achevees, 1);
        assert!(store.get_workspace(ws.id).await.unwrap().is_none());
        for j in &jalons {
            assert!(store.get_workspace_milestone(j.id).await.unwrap().is_none());
        }
        assert_eq!(store.noeuds_de_label_pour_test("WorkspaceMilestone"), 0);
    }

    /// A milestone status stored in PascalCase (a base migrated from Neo4j) reads
    /// as Neo4j reads it, and the filter sees it under that status.
    #[tokio::test]
    async fn a_pascal_case_milestone_status_reads_as_on_neo4j() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let ws = espace();
        let j = jalon_de(ws.id);
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_workspace(&ws).await.unwrap();
            store.create_workspace_milestone(&j).await.unwrap();
            assert!(store.ecrire_propriete_brute_pour_test(
                "WorkspaceMilestone",
                j.id,
                "status",
                Some(ValeurBrute::Texte("InProgress"))
            ));
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(
            store
                .get_workspace_milestone(j.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            MilestoneStatus::InProgress
        );
        let (_, en_cours) = store
            .list_workspace_milestones_filtered(ws.id, Some("in_progress"), 10, 0)
            .await
            .unwrap();
        let (_, ouverts) = store
            .list_workspace_milestones_filtered(ws.id, Some("open"), 10, 0)
            .await
            .unwrap();
        assert_eq!((en_cours, ouverts), (1, 0));
    }
}

/// What only the obrain adapter does for plans: the deletion of a plan does not
/// depend on the other domains' nodes, a creation stopped at any point of its
/// protocol reopens as nothing or as the whole plan, the status is read as Neo4j
/// reads it, and a migrated plan whose `HAS_PLAN` edges disagree with its
/// `project_id` is quarantined.
#[cfg(feature = "obrain-backend")]
mod obrain_plan_specifique {
    use chrono::{TimeZone, Utc};
    use project_orchestrator::neo4j::models::{PlanNode, PlanStatus, ProjectNode, WorkspaceNode};
    use project_orchestrator::neo4j::traits::{PlanStore, ProjectStore, WorkspaceStore};
    use project_orchestrator::obrain::{
        Creation, Interruption, NonPorte, ObrainGraphStore, ValeurBrute,
    };
    use project_orchestrator::plan::models::UpdatePlanRequest;
    use uuid::Uuid;

    fn projet(label: &str) -> ProjectNode {
        let mut p = super::conformance::a_project(label);
        p.created_at = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
        p
    }

    fn plan(project_id: Option<Uuid>) -> PlanNode {
        PlanNode {
            id: Uuid::new_v4(),
            title: "plan".into(),
            description: "description".into(),
            status: PlanStatus::Draft,
            created_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
            created_by: "test".into(),
            priority: 3,
            project_id,
            execution_context: None,
            persona: None,
        }
    }

    /// A plan is deleted although the base holds nodes of other domains (a
    /// workspace here): only what hangs from the plan counts.
    #[tokio::test]
    async fn delete_plan_ignores_the_nodes_of_other_domains() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let ws = WorkspaceNode {
            id: Uuid::new_v4(),
            name: "ws".into(),
            slug: format!("ws-{}", Uuid::new_v4().simple()),
            description: None,
            created_at: Utc::now(),
            updated_at: None,
            metadata: serde_json::json!({}),
        };
        store.create_workspace(&ws).await.unwrap();
        let x = plan(None);
        store.create_plan(&x).await.unwrap();
        store.delete_plan(x.id).await.unwrap();
        assert!(store.get_plan(x.id).await.unwrap().is_none());
    }

    /// A plan with an outgoing edge (a task of a migrated base) is refused by name:
    /// the cascade would have to reach the task.
    #[tokio::test]
    async fn delete_plan_refuses_a_plan_with_a_task() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let x = plan(None);
        store.create_plan(&x).await.unwrap();
        let tache = Uuid::new_v4().to_string();
        store.creer_noeud_brut_pour_test("Task", &[("id", ValeurBrute::Texte(&tache))]);
        assert!(store.creer_arete_brute_pour_test(
            ("Plan", x.id),
            ("Task", tache.parse().unwrap()),
            "HAS_TASK"
        ));
        let e = store.delete_plan(x.id).await.unwrap_err();
        assert_eq!(
            e.downcast_ref::<NonPorte>(),
            Some(&NonPorte("PlanStore::delete_plan (cascade)"))
        );
        assert!(format!("{e:#}").contains("1 arête(s) sortante(s)"), "{e:#}");
        assert!(store.get_plan(x.id).await.unwrap().is_some());
    }

    /// A creation stopped at each point of the draft-marker protocol: before the
    /// marker is removed, the reopened base has no plan and one draft deleted;
    /// after, the whole plan.
    #[tokio::test]
    async fn an_interrupted_creation_reopens_as_nothing_or_the_whole_plan() {
        let mut points = vec![Interruption::ApresLeNoeud, Interruption::ApresLeMarqueur];
        points.extend((1..=8).map(Interruption::ApresLaPropriete));
        points.push(Interruption::AvantLeRetraitDuMarqueur);
        points.push(Interruption::ApresLeRetraitDuMarqueur);
        for point in points {
            let dir = tempfile::tempdir().unwrap();
            let chemin = dir.path().join("base");
            let x = plan(None);
            {
                let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
                store.interrompre_au_prochain(point);
                let e = store.create_plan(&x).await.unwrap_err();
                assert!(
                    format!("{e:#}").contains("interruption injectée"),
                    "{point:?}: {e:#}"
                );
                store.close().await.unwrap();
            }
            let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
            let rapport = store.rapport_d_ouverture().clone();
            assert!(rapport.en_quarantaine.is_empty(), "{point:?}: {rapport:?}");
            let relu = store.get_plan(x.id).await.unwrap();
            if point == Interruption::ApresLeRetraitDuMarqueur {
                assert_eq!(rapport.brouillons_supprimes, 0, "{point:?}");
                assert_eq!(
                    serde_json::to_value(relu.expect("committed")).unwrap(),
                    serde_json::to_value(&x).unwrap(),
                    "{point:?}"
                );
            } else {
                assert_eq!(rapport.brouillons_supprimes, 1, "{point:?}");
                assert!(relu.is_none(), "{point:?}");
            }
            store.close().await.unwrap();
        }
    }

    /// The stored status is read as Neo4j reads it: `in_progress` decodes as
    /// `InProgress` but is not active (the filter compares the stored string), and a
    /// plan with no status is quarantined (Neo4j's read fails).
    #[tokio::test]
    async fn the_stored_status_is_read_as_neo4j_reads_it() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = projet("plan-statut");
        let migre = plan(Some(p.id));
        let sans_statut = plan(Some(p.id));
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_project(&p).await.unwrap();
            store.create_plan(&migre).await.unwrap();
            store.create_plan(&sans_statut).await.unwrap();
            assert!(store.ecrire_propriete_brute_pour_test(
                "Plan",
                migre.id,
                "status",
                Some(ValeurBrute::Texte("in_progress"))
            ));
            assert!(store.ecrire_propriete_brute_pour_test("Plan", sans_statut.id, "status", None));
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(
            store.get_plan(migre.id).await.unwrap().unwrap().status,
            PlanStatus::InProgress
        );
        assert!(store.list_active_plans().await.unwrap().is_empty());
        assert_eq!(store.count_project_plans(p.id).await.unwrap(), 0);
        let (page, total) = store
            .list_plans_for_project(p.id, Some(vec!["in_progress".into()]), 10, 0)
            .await
            .unwrap();
        assert!(
            page.is_empty() && total == 0,
            "the filter compares the stored string: in_progress becomes InProgress, which is not stored"
        );
        assert!(store.get_plan(sans_statut.id).await.unwrap().is_none());
        assert!(store
            .rapport_d_ouverture()
            .en_quarantaine
            .iter()
            .any(|q| q.contains("status")));
    }

    /// The project lists need the project (Neo4j's `MATCH (project:Project {id})`):
    /// a plan whose project is absent is listed under no project.
    #[tokio::test]
    async fn a_plan_of_an_absent_project_is_listed_under_none() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = projet("plan-absent");
        let x = plan(Some(p.id));
        let ailleurs = Uuid::new_v4();
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_project(&p).await.unwrap();
            store.create_plan(&x).await.unwrap();
            let ailleurs_s = ailleurs.to_string();
            assert!(store.ecrire_propriete_brute_pour_test(
                "Plan",
                x.id,
                "project_id",
                Some(ValeurBrute::Texte(&ailleurs_s))
            ));
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert!(store.get_plan(x.id).await.unwrap().is_some());
        assert!(store.list_project_plans(ailleurs).await.unwrap().is_empty());
        assert_eq!(store.count_project_plans(ailleurs).await.unwrap(), 0);
        let (page, total) = store
            .list_plans_for_project(ailleurs, None, 10, 0)
            .await
            .unwrap();
        assert!(page.is_empty() && total == 0);
    }

    /// A migrated plan linked by `HAS_PLAN` to A and to B (`project_id = B`):
    /// Neo4j lists it under both, this backend could only list it under B. It is
    /// quarantined and named.
    #[tokio::test]
    async fn a_plan_linked_to_several_projects_is_quarantined() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let a = projet("plan-a");
        let b = projet("plan-b");
        let x = plan(Some(b.id));
        let seul = plan(Some(b.id));
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_project(&a).await.unwrap();
            store.create_project(&b).await.unwrap();
            store.create_plan(&x).await.unwrap();
            store.create_plan(&seul).await.unwrap();
            for (projet, plan) in [(a.id, x.id), (b.id, x.id), (b.id, seul.id)] {
                assert!(store.creer_arete_brute_pour_test(
                    ("Project", projet),
                    ("Plan", plan),
                    "HAS_PLAN"
                ));
            }
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert!(store.get_plan(x.id).await.unwrap().is_none());
        assert!(store
            .rapport_d_ouverture()
            .en_quarantaine
            .iter()
            .any(|q| q.contains(&x.id.to_string()) && q.contains("HAS_PLAN")));
        // The edge that agrees with `project_id` is the migrated base's normal case.
        assert_eq!(
            store
                .list_project_plans(b.id)
                .await
                .unwrap()
                .iter()
                .map(|p| p.id)
                .collect::<Vec<_>>(),
            vec![seul.id]
        );
    }

    /// An empty update on a closed base refuses like every other operation.
    #[tokio::test]
    async fn an_empty_update_on_a_closed_base_refuses() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        store.close().await.unwrap();
        let vide = UpdatePlanRequest {
            title: None,
            description: None,
            priority: None,
        };
        let e = store.update_plan(Uuid::new_v4(), &vide).await.unwrap_err();
        assert!(format!("{e:#}").contains("fermée"), "{e:#}");
    }
}

/// The refusal of `delete_project` while the base holds nodes of other domains:
/// the cascade is not ported, so it stays refused, but it names what it found.
#[cfg(feature = "obrain-backend")]
mod obrain_delete_project_refus {
    use chrono::Utc;
    use project_orchestrator::neo4j::models::{MilestoneNode, MilestoneStatus, WorkspaceNode};
    use project_orchestrator::neo4j::traits::{
        FederationStore, ProjectStore, ReleaseStore, WorkspaceStore,
    };
    use project_orchestrator::obrain::{NonPorte, ObrainGraphStore};
    use uuid::Uuid;

    #[tokio::test]
    async fn the_refusal_names_the_domains_it_found() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let a = super::conformance::a_project("refus-a");
        let b = super::conformance::a_project("refus-b");
        store.create_project(&a).await.unwrap();
        store.create_project(&b).await.unwrap();
        store
            .create_workspace(&WorkspaceNode {
                id: Uuid::new_v4(),
                name: "ws".into(),
                slug: format!("ws-{}", Uuid::new_v4().simple()),
                description: None,
                created_at: Utc::now(),
                updated_at: None,
                metadata: serde_json::json!({}),
            })
            .await
            .unwrap();
        store
            .create_milestone(&MilestoneNode {
                id: Uuid::new_v4(),
                title: "jalon de A".into(),
                description: None,
                status: MilestoneStatus::Open,
                target_date: None,
                closed_at: None,
                created_at: Utc::now(),
                project_id: a.id,
            })
            .await
            .unwrap();
        store
            .create_mcp_server(&super::conformance::an_mcp_server(Uuid::nil(), "refus"))
            .await
            .unwrap();

        // B has nothing of its own: the refusal still holds (the cascade is not
        // ported), and says why, domain by domain.
        let e = store.delete_project(b.id, &b.name).await.unwrap_err();
        assert_eq!(
            e.downcast_ref::<NonPorte>(),
            Some(&NonPorte("ProjectStore::delete_project (cascade)"))
        );
        let msg = format!("{e:#}");
        for attendu in [
            "3 nœud(s)",
            "releases (Milestone ×1)",
            "workspaces (Workspace ×1)",
            "fédération (McpServer ×1)",
        ] {
            assert!(msg.contains(attendu), "« {attendu} » absent de : {msg}");
        }
        assert!(!msg.contains("écrites hors de lui"), "{msg}");
        assert!(store.get_project(b.id).await.unwrap().is_some());

        // A node no ported domain writes is named as such.
        store.creer_noeud_brut_pour_test("Task", &[]);
        let msg = format!(
            "{:#}",
            store.delete_project(b.id, &b.name).await.unwrap_err()
        );
        assert!(
            msg.contains("hors des domaines portés, écrits par un autre que ce backend (Task ×1)"),
            "{msg}"
        );
    }
}

/// What only the obrain adapter does for the release domain: every refused
/// method refuses by its name, the four entities read back identical after a
/// close and an open, a creation stopped before its commit point is a draft
/// deleted at open, a status is required, and an environment deletion cut half
/// way is finished at open.
#[cfg(feature = "obrain-backend")]
mod obrain_release_specifique {
    use chrono::{Duration, TimeZone, Utc};
    use project_orchestrator::neo4j::models::{
        DeploymentNode, DeploymentStatus, EnvironmentKind, EnvironmentNode, MilestoneNode,
        MilestoneStatus, ReleaseNode, ReleaseStatus,
    };
    use project_orchestrator::neo4j::traits::{ProjectStore, ReleaseStore};
    use project_orchestrator::obrain::{
        refusees_release, Creation, Interruption, NonPorte, ObrainGraphStore, ValeurBrute,
    };
    use uuid::Uuid;

    fn t(s: i64) -> chrono::DateTime<Utc> {
        Utc.timestamp_opt(1_700_000_000 + s, 0).unwrap()
    }

    fn release(p: Uuid) -> ReleaseNode {
        ReleaseNode {
            id: Uuid::new_v4(),
            version: "1.2.3".into(),
            title: Some("titre".into()),
            description: Some("description".into()),
            status: ReleaseStatus::InProgress,
            target_date: Some(t(100)),
            released_at: None,
            created_at: t(1),
            project_id: p,
        }
    }

    fn environnement(p: Uuid) -> EnvironmentNode {
        EnvironmentNode {
            id: Uuid::new_v4(),
            project_id: p,
            name: format!("env-{}", Uuid::new_v4().simple()),
            kind: EnvironmentKind::Staging,
            url: Some("https://example.invalid".into()),
            description: Some("préprod".into()),
            config: Some("{\"region\":\"eu\"}".into()),
            created_at: t(2),
        }
    }

    fn deploiement(env: Uuid, k: i64) -> DeploymentNode {
        DeploymentNode {
            id: Uuid::new_v4(),
            environment_id: env,
            version: Some(format!("1.0.{k}")),
            commit_sha: Some("abc123".into()),
            status: DeploymentStatus::Succeeded,
            notes: Some("ok".into()),
            created_by: "ci".into(),
            started_at: t(10 + k),
            finished_at: Some(t(10 + k) + Duration::seconds(5)),
        }
    }

    fn jalon(p: Uuid) -> MilestoneNode {
        MilestoneNode {
            id: Uuid::new_v4(),
            title: "jalon".into(),
            description: Some("d".into()),
            status: MilestoneStatus::InProgress,
            target_date: Some(t(50)),
            closed_at: None,
            created_at: t(3),
            project_id: p,
        }
    }

    fn json<T: serde::Serialize>(v: &T) -> serde_json::Value {
        serde_json::to_value(v).unwrap()
    }

    /// Each of the 18 refused methods answers `NonPorte` with its own name.
    #[tokio::test]
    async fn every_refused_method_refuses_by_its_name() {
        let s = ObrainGraphStore::open_temporary().unwrap();
        let id = Uuid::new_v4();
        fn nom<T>(r: anyhow::Result<T>) -> Option<NonPorte> {
            r.err().and_then(|e| e.downcast_ref::<NonPorte>().copied())
        }
        let vus = vec![
            nom(s.add_task_to_release(id, id).await),
            nom(s.add_commit_to_release(id, "abc").await),
            nom(s.remove_commit_from_release(id, "abc").await),
            nom(s.get_release_details(id).await),
            nom(s.get_release_tasks(id).await),
            nom(s.add_task_to_milestone(id, id).await),
            nom(s.link_plan_to_milestone(id, id).await),
            nom(s.unlink_plan_from_milestone(id, id).await),
            nom(s.get_milestone_details(id).await),
            nom(s.get_milestone_progress(id).await),
            nom(s.get_milestone_tasks_with_plans(id).await),
            nom(s.get_milestone_steps_batch(id).await),
            nom(s.get_milestone_tasks(id).await),
            nom(s.get_project_progress(id).await),
            nom(s.get_project_task_dependencies(id).await),
            nom(s.get_project_tasks(id).await),
            nom(s
                .list_plans_filtered(None, None, None, None, None, None, 10, 0, None, "desc")
                .await),
            nom(s
                .list_all_tasks_filtered(
                    None, None, None, None, None, None, None, None, 10, 0, None, "desc",
                )
                .await),
        ];
        let attendus: Vec<Option<NonPorte>> = refusees_release()
            .iter()
            .map(|m| Some(NonPorte(m)))
            .collect();
        assert_eq!(vus, attendus, "in the order of REFUSEES");
    }

    /// The four entities read back identical after a close and an open.
    #[tokio::test]
    async fn the_four_entities_survive_close_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = super::conformance::a_project("release-reopen");
        let r = release(p.id);
        let e = environnement(p.id);
        let d = deploiement(e.id, 1);
        let j = jalon(p.id);
        {
            let s = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            s.create_project(&p).await.unwrap();
            s.create_release(&r).await.unwrap();
            s.create_environment(&e).await.unwrap();
            s.create_deployment(&d).await.unwrap();
            s.create_milestone(&j).await.unwrap();
            s.close().await.unwrap();
        }
        let s = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(s.rapport_d_ouverture().brouillons_supprimes, 0);
        assert_eq!(json(&s.get_release(r.id).await.unwrap().unwrap()), json(&r));
        assert_eq!(
            json(&s.get_environment(e.id).await.unwrap().unwrap()),
            json(&e)
        );
        assert_eq!(
            json(&s.get_deployment(d.id).await.unwrap().unwrap()),
            json(&d)
        );
        assert_eq!(
            json(&s.get_milestone(j.id).await.unwrap().unwrap()),
            json(&j)
        );
    }

    /// A release creation stopped before its marker is removed is a draft: deleted
    /// at open, counted, and the release is absent.
    #[tokio::test]
    async fn an_interrupted_creation_is_a_draft_deleted_at_open() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = super::conformance::a_project("release-draft");
        let r = release(p.id);
        {
            let s = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            s.create_project(&p).await.unwrap();
            s.interrompre_au_prochain(Interruption::AvantLeRetraitDuMarqueur);
            assert!(s.create_release(&r).await.is_err());
            s.close().await.unwrap();
        }
        let s = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(s.rapport_d_ouverture().brouillons_supprimes, 1);
        assert!(s.get_release(r.id).await.unwrap().is_none());
    }

    /// A release and a milestone with no `status` are quarantined (Neo4j's read
    /// fails on them); a legacy PascalCase status is read and filtered as snake_case.
    #[tokio::test]
    async fn a_status_is_required_and_a_legacy_spelling_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = super::conformance::a_project("release-status");
        let r = release(p.id);
        let j = jalon(p.id);
        let legacy = jalon(p.id);
        {
            let s = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            s.create_project(&p).await.unwrap();
            s.create_release(&r).await.unwrap();
            s.create_milestone(&j).await.unwrap();
            s.create_milestone(&legacy).await.unwrap();
            assert!(s.ecrire_propriete_brute_pour_test("Release", r.id, "status", None));
            assert!(s.ecrire_propriete_brute_pour_test("Milestone", j.id, "status", None));
            assert!(s.ecrire_propriete_brute_pour_test(
                "Milestone",
                legacy.id,
                "status",
                Some(ValeurBrute::Texte("InProgress"))
            ));
            s.close().await.unwrap();
        }
        let s = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert!(s.get_release(r.id).await.unwrap().is_none());
        assert!(s.get_milestone(j.id).await.unwrap().is_none());
        let q = &s.rapport_d_ouverture().en_quarantaine;
        assert_eq!(
            q.iter().filter(|x| x.contains("status")).count(),
            2,
            "{q:?}"
        );
        let (page, total) = s
            .list_milestones_filtered(p.id, Some(vec!["in_progress".into()]), 10, 0, None, "asc")
            .await
            .unwrap();
        assert_eq!((page.len(), total), (1, 1));
        assert_eq!(page[0].status, MilestoneStatus::InProgress);
    }

    /// `delete_environment` cut after the marker and one deployment: the next open
    /// finishes it — no environment, no deployment, one deletion finished — rather
    /// than serving the environment with part of its deployments.
    #[tokio::test]
    async fn an_environment_deletion_cut_half_way_is_finished_at_open() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let p = super::conformance::a_project("env-cut");
        let e = environnement(p.id);
        let ds: Vec<DeploymentNode> = (1..=3).map(|k| deploiement(e.id, k)).collect();
        {
            let s = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            s.create_project(&p).await.unwrap();
            s.create_environment(&e).await.unwrap();
            for d in &ds {
                s.create_deployment(d).await.unwrap();
            }
            // Step 1 is the marker, step 2 the first deployment deleted.
            s.interrompre_au_prochain(Interruption::ApresEtape(2));
            assert!(s.delete_environment(e.id).await.is_err());
            s.close().await.unwrap();
        }
        let s = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(s.rapport_d_ouverture().suppressions_achevees, 1);
        assert!(s.get_environment(e.id).await.unwrap().is_none());
        for d in &ds {
            assert!(s.get_deployment(d.id).await.unwrap().is_none());
        }
        assert_eq!(s.noeuds_de_label_pour_test("Deployment"), 0);
        assert_eq!(s.noeuds_de_label_pour_test("Environment"), 0);
    }

    /// The environment node is gone from the engine while the index holds it: the
    /// deletion refuses and deletes no deployment.
    #[tokio::test]
    async fn delete_environment_writes_nothing_when_it_refuses() {
        let s = ObrainGraphStore::open_temporary().unwrap();
        let p = super::conformance::a_project("env-refus");
        let e = environnement(p.id);
        let d = deploiement(e.id, 1);
        s.create_project(&p).await.unwrap();
        s.create_environment(&e).await.unwrap();
        s.create_deployment(&d).await.unwrap();
        assert!(s.supprimer_noeud_brut_pour_test("Environment", e.id));
        let err = s.delete_environment(e.id).await.unwrap_err();
        assert!(
            format!("{err:#}").contains("rien n'a été supprimé"),
            "{err:#}"
        );
        assert_eq!(s.noeuds_de_label_pour_test("Deployment"), 1);
    }
}

/// What only the obrain adapter does for the federation: the total order of the
/// lists, the drafts and the duplicates.
#[cfg(feature = "obrain-backend")]
mod obrain_federation_specifique {
    use chrono::{TimeZone, Utc};
    use project_orchestrator::neo4j::traits::FederationStore;
    use project_orchestrator::obrain::{Creation, Interruption, ObrainGraphStore, ValeurBrute};
    use uuid::Uuid;

    /// Servers by `(server_id, most recent created_at, id)`, tools by `(name, id)`:
    /// the exact order, ties included. The two servers `alpha` are the case of a
    /// server connected again after a failed reconnection: the newer comes first,
    /// whatever their ids, so startup restores the latest configuration.
    #[tokio::test]
    async fn the_lists_follow_the_total_order() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let project = Uuid::new_v4();
        let mk = |sid: &str, id: u128, t: i64| {
            let mut s = super::conformance::an_mcp_server(project, sid);
            s.id = Uuid::from_u128(id);
            s.created_at = Utc.timestamp_opt(1_700_000_000 + t, 0).unwrap();
            s
        };
        let ancien = mk("alpha", 1, 0);
        let recent = mk("alpha", 2, 100);
        let meme_instant_a = mk("beta", 3, 0);
        let meme_instant_b = mk("beta", 4, 0);
        let zeta = mk("zeta", 5, 0);
        for s in [&zeta, &meme_instant_b, &ancien, &recent, &meme_instant_a] {
            store.create_mcp_server(s).await.unwrap();
        }
        let attendu = vec![
            recent.id,
            ancien.id,
            meme_instant_a.id,
            meme_instant_b.id,
            zeta.id,
        ];
        let ids = |v: Vec<project_orchestrator::neo4j::models::McpServerNode>| {
            v.into_iter().map(|s| s.id).collect::<Vec<_>>()
        };
        assert_eq!(ids(store.list_all_mcp_servers().await.unwrap()), attendu);
        assert_eq!(ids(store.list_mcp_servers(project).await.unwrap()), attendu);

        // `delete_mcp_server_by_server_id` still finds every server of the id.
        store.delete_mcp_server_by_server_id("alpha").await.unwrap();
        assert_eq!(
            ids(store.list_all_mcp_servers().await.unwrap()),
            vec![meme_instant_a.id, meme_instant_b.id, zeta.id]
        );

        let mut tools = Vec::new();
        for (name, id) in [("read", 9u128), ("write", 1), ("read", 7)] {
            let mut t = super::conformance::an_mcp_tool("outils", name);
            t.id = Uuid::from_u128(id);
            store.create_mcp_tool(&t).await.unwrap();
            tools.push(t);
        }
        let listed: Vec<Uuid> = store
            .list_mcp_tools_for_server("outils")
            .await
            .unwrap()
            .into_iter()
            .map(|t| t.id)
            .collect();
        assert_eq!(
            listed,
            vec![Uuid::from_u128(7), Uuid::from_u128(9), Uuid::from_u128(1)]
        );
    }

    /// A server or tool creation stopped before its commit point is a draft: deleted
    /// at open, counted, absent.
    #[tokio::test]
    async fn an_interrupted_creation_is_a_draft_deleted_at_open() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let s = super::conformance::an_mcp_server(Uuid::nil(), "brouillon");
        let t = super::conformance::an_mcp_tool("brouillon", "read");
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.interrompre_au_prochain(Interruption::ApresLaPropriete(3));
            assert!(store.create_mcp_server(&s).await.is_err());
            store.interrompre_au_prochain(Interruption::AvantLeRetraitDuMarqueur);
            assert!(store.create_mcp_tool(&t).await.is_err());
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(store.rapport_d_ouverture().brouillons_supprimes, 2);
        assert!(store.get_mcp_server(s.id).await.unwrap().is_none());
        assert!(store.list_all_mcp_servers().await.unwrap().is_empty());
        assert!(store
            .list_mcp_tools_for_server("brouillon")
            .await
            .unwrap()
            .is_empty());
    }

    /// A second creation with a taken id is refused and writes nothing; a duplicate
    /// found at open (a corrupted base) is quarantined and named, the first kept.
    #[tokio::test]
    async fn a_duplicate_id_is_refused_and_quarantined_at_open() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let s = super::conformance::an_mcp_server(Uuid::nil(), "double");
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_mcp_server(&s).await.unwrap();
            let mut autre = super::conformance::an_mcp_server(Uuid::nil(), "autre");
            autre.id = s.id;
            let e = store.create_mcp_server(&autre).await.unwrap_err();
            assert!(format!("{e:#}").contains("existe déjà"), "{e:#}");
            assert_eq!(store.noeuds_de_label_pour_test("McpServer"), 1);
            // The duplicate a corrupted base would hold, written behind the index.
            let id = s.id.to_string();
            let cree = s.created_at.to_rfc3339();
            store.creer_noeud_brut_pour_test(
                "McpServer",
                &[
                    ("id", ValeurBrute::Texte(&id)),
                    ("project_id", ValeurBrute::Texte(&Uuid::nil().to_string())),
                    ("server_id", ValeurBrute::Texte("intrus")),
                    ("created_at", ValeurBrute::Texte(&cree)),
                ],
            );
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        assert_eq!(
            store.get_mcp_server(s.id).await.unwrap().unwrap().server_id,
            "double",
            "the first created is kept"
        );
        assert!(store
            .rapport_d_ouverture()
            .en_quarantaine
            .iter()
            .any(|q| q.contains("McpServer") && q.contains("déjà pris")));
    }

    /// An unreadable optional timestamp reads as absent (Neo4j's `.ok()`): the
    /// server is still served and reconnected at startup.
    #[tokio::test]
    async fn an_unreadable_optional_timestamp_reads_as_absent() {
        let dir = tempfile::tempdir().unwrap();
        let chemin = dir.path().join("base");
        let s = super::conformance::an_mcp_server(Uuid::nil(), "horodatage");
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_mcp_server(&s).await.unwrap();
            assert!(store.ecrire_propriete_brute_pour_test(
                "McpServer",
                s.id,
                "last_connected_at",
                Some(ValeurBrute::Texte("pas une date"))
            ));
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        let lu = store.get_mcp_server(s.id).await.unwrap().expect("served");
        assert_eq!(lu.last_connected_at, None);
    }
}

/// The note domain: the checks that call nothing but `NoteStore` (and the
/// project they attach to). Run on the obrain backend for the ported methods.
macro_rules! note_checks {
    ($backend:ident, $factory:ident) => {
        conformance_suite!(
            $backend,
            $factory,
            [
                check_note_roundtrip,
                check_note_update_rescope_delete,
                check_note_list_order_and_paging,
                check_note_entity_links,
                check_note_supersede_and_confirm,
                check_note_needing_review,
            ]
        );
    };
}

note_checks!(neo4j_note, Neo4jFactory);
#[cfg(feature = "obrain-backend")]
note_checks!(obrain_note, ObrainFactory);
