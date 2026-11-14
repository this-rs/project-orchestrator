//! The task half of `PlanStore` on obrain (`src/obrain/task.rs`, `step.rs`):
//! what only this backend does. Persistence across a close and a reopen, a
//! write cut at every engine write then reopened and retried, the quarantine of
//! an unreadable node (its id reserved, its plan refusing a partial answer), and
//! the four refusals named. The behaviour shared with Neo4j and the mock is in
//! `tests/store_conformance.rs` (`task_checks!`).
#![cfg(feature = "obrain-backend")]

use std::path::Path;

use chrono::{NaiveDate, TimeZone, Utc};
use obrain_common::types::Value;
use project_orchestrator::neo4j::models::*;
use project_orchestrator::neo4j::traits::*;
use project_orchestrator::obrain::{
    refusees_plan, Creation, Interruption, NonPorte, ObrainGraphStore, MARQUEUR_PO, SCHEMA_PO,
};
use project_orchestrator::plan::models::UpdateTaskRequest;
use uuid::Uuid;

async fn open(path: &Path) -> ObrainGraphStore {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || ObrainGraphStore::open(path, Creation::Permise))
        .await
        .unwrap()
        .unwrap()
}

fn a_plan() -> PlanNode {
    PlanNode {
        id: Uuid::new_v4(),
        title: "plan".into(),
        description: "d".into(),
        status: PlanStatus::Draft,
        created_at: Utc::now(),
        created_by: "test".into(),
        priority: 0,
        project_id: None,
        execution_context: None,
        persona: None,
    }
}

fn a_task(title: &str, priority: i32, offset: i64) -> TaskNode {
    TaskNode {
        id: Uuid::new_v4(),
        title: Some(title.into()),
        description: format!("description de {title}"),
        status: TaskStatus::Pending,
        assigned_to: None,
        interrupted_at: None,
        interrupted_reason: None,
        priority: Some(priority),
        tags: vec!["x".into()],
        acceptance_criteria: vec![],
        affected_files: vec!["src/a.rs".into()],
        estimated_complexity: Some(3),
        actual_complexity: None,
        due_date: NaiveDate::from_ymd_opt(2026, 12, 1),
        estimated_minutes: None,
        created_at: Utc.timestamp_opt(1_700_000_000 + offset, 0).unwrap(),
        updated_at: None,
        started_at: None,
        completed_at: None,
        frustration_score: 0.0,
        execution_context: None,
        persona: None,
        prompt_cache: None,
    }
}

fn a_step(order: u32) -> StepNode {
    StepNode {
        id: Uuid::new_v4(),
        order,
        description: format!("étape {order}"),
        status: StepStatus::Pending,
        verification: Some("vérifier".into()),
        created_at: Utc.timestamp_opt(1_700_000_000 + order as i64, 0).unwrap(),
        updated_at: None,
        completed_at: None,
        execution_context: None,
        persona: None,
    }
}

/// Everything a reader sees of a plan's tasks, steps, dependencies and a day plan.
async fn snapshot(store: &ObrainGraphStore, plan: Uuid, user: Uuid) -> serde_json::Value {
    let taches = store.get_plan_tasks(plan).await.unwrap();
    let mut etapes = Vec::new();
    let mut deps = Vec::new();
    for t in &taches {
        etapes.push(store.get_task_steps(t.id).await.unwrap());
        deps.push(
            store
                .get_task_dependencies(t.id)
                .await
                .unwrap()
                .into_iter()
                .map(|d| d.id)
                .collect::<Vec<_>>(),
        );
    }
    serde_json::json!({
        "taches": taches,
        "etapes": etapes,
        "deps": deps,
        "jour": store.get_day_plan(user, "2026-11-13").await.unwrap(),
        "vagues": store.compute_waves(plan).await.unwrap().waves.len(),
    })
}

/// Tasks, steps, dependencies and a day plan: identical after a close and a
/// reopen, and after a reopen of a store dropped without `close`.
#[tokio::test]
async fn tasks_steps_and_day_plan_survive_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let plan = a_plan();
    let user = Uuid::new_v4();
    let (a, b) = (a_task("a", 2, 0), a_task("b", 1, 1));
    let avant = {
        let store = open(&base).await;
        store.create_plan(&plan).await.unwrap();
        store.create_task(plan.id, &a).await.unwrap();
        store.create_task(plan.id, &b).await.unwrap();
        store.add_task_dependency(b.id, a.id).await.unwrap();
        store.create_step(a.id, &a_step(0)).await.unwrap();
        store.create_step(a.id, &a_step(1)).await.unwrap();
        store
            .update_task_status(a.id, TaskStatus::InProgress)
            .await
            .unwrap();
        store
            .update_task(
                b.id,
                &UpdateTaskRequest {
                    assigned_to: Some("someone".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        store
            .set_day_plan(user, "2026-11-13", &[b.id, a.id])
            .await
            .unwrap();
        let s = snapshot(&store, plan.id, user).await;
        store.close().await.unwrap();
        s
    };
    let store = open(&base).await;
    assert_eq!(
        store.rapport_d_ouverture().en_quarantaine,
        Vec::<String>::new()
    );
    assert_eq!(snapshot(&store, plan.id, user).await, avant);
    // Dropped, not closed.
    store.complete_pending_steps_for_task(a.id).await.unwrap();
    let apres = snapshot(&store, plan.id, user).await;
    drop(store);
    let store = open(&base).await;
    assert_eq!(snapshot(&store, plan.id, user).await, apres);
}

/// A creation, a status change, a cascade of step completions, a day plan and a
/// deletion, each cut after its k-th engine write: the store refuses everything
/// until a restart; the reopen quarantines nothing; a retry of the same call
/// converges to the state the call was meant to leave.
#[tokio::test]
async fn writes_cut_at_every_engine_write_reopen_consistently_and_converge() {
    let dir = tempfile::tempdir().unwrap();
    let mut atteints = 0;

    async fn fixture(dir: &Path, nom: &str, k: usize) -> (ObrainGraphStore, PlanNode, TaskNode) {
        let store = open(&dir.join(format!("{nom}-{k}"))).await;
        let plan = a_plan();
        let t = a_task("t", 1, 0);
        store.create_plan(&plan).await.unwrap();
        store.create_task(plan.id, &t).await.unwrap();
        for i in 0..3 {
            store.create_step(t.id, &a_step(i)).await.unwrap();
        }
        (store, plan, t)
    }
    async fn reouvrir(dir: &Path, nom: &str, k: usize) -> ObrainGraphStore {
        let store = open(&dir.join(format!("{nom}-{k}"))).await;
        assert!(
            store.rapport_d_ouverture().en_quarantaine.is_empty(),
            "{nom} k={k}: {:?}",
            store.rapport_d_ouverture().en_quarantaine
        );
        store
    }

    // create_task: a draft never shows.
    for k in 1..=20 {
        let (store, plan, _) = fixture(dir.path(), "create", k).await;
        let n = a_task("n", 0, 9);
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.create_task(plan.id, &n).await.is_ok() {
            break;
        }
        atteints += 1;
        assert!(store.get_plan_tasks(plan.id).await.is_err(), "indéterminé");
        drop(store);
        let store = reouvrir(dir.path(), "create", k).await;
        assert!(store.get_task(n.id).await.unwrap().is_none(), "k={k}");
        store.create_task(plan.id, &n).await.unwrap();
        assert!(store.get_task(n.id).await.unwrap().is_some());
    }

    // update_task_status to Completed.
    for k in 1..=10 {
        let (store, _, t) = fixture(dir.path(), "status", k).await;
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store
            .update_task_status(t.id, TaskStatus::Completed)
            .await
            .is_ok()
        {
            break;
        }
        atteints += 1;
        drop(store);
        let store = reouvrir(dir.path(), "status", k).await;
        // What the cut left, BEFORE any replay: the first k-1 writes (status,
        // completed_at, updated_at, …) and nothing after them.
        let v = store.get_task(t.id).await.unwrap().unwrap();
        let faites = k - 1;
        assert_eq!(
            v.status,
            if faites >= 1 {
                TaskStatus::Completed
            } else {
                TaskStatus::Pending
            },
            "k={k}: état laissé par la coupure"
        );
        assert_eq!(v.completed_at.is_some(), faites >= 2, "k={k}: completed_at");
        assert_eq!(
            v.updated_at != Some(v.created_at),
            faites >= 3,
            "k={k}: updated_at"
        );
        store
            .update_task_status(t.id, TaskStatus::Completed)
            .await
            .unwrap();
        let v = store.get_task(t.id).await.unwrap().unwrap();
        assert_eq!(v.status, TaskStatus::Completed, "k={k}");
        assert!(v.completed_at.is_some(), "k={k}");
    }

    // complete_pending_steps_for_task: three steps, three writes each.
    for k in 1..=12 {
        let (store, _, t) = fixture(dir.path(), "steps", k).await;
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.complete_pending_steps_for_task(t.id).await.is_ok() {
            break;
        }
        atteints += 1;
        drop(store);
        let store = reouvrir(dir.path(), "steps", k).await;
        // BEFORE the replay: the steps in `order`, three writes each (status,
        // completed_at, updated_at); exactly those whose first write was made read
        // completed, and only the last of them may lack its timestamps.
        let faites = k - 1;
        let etapes = store.get_task_steps(t.id).await.unwrap();
        for (i, s) in etapes.iter().enumerate() {
            let ecrites = faites.saturating_sub(3 * i).min(3);
            assert_eq!(
                s.status == StepStatus::Completed,
                ecrites >= 1,
                "k={k} étape {i}: statut"
            );
            assert_eq!(s.completed_at.is_some(), ecrites >= 2, "k={k} étape {i}");
        }
        let attendu = 3 - etapes
            .iter()
            .filter(|s| s.status == StepStatus::Completed)
            .count() as u32;
        assert_eq!(
            store.complete_pending_steps_for_task(t.id).await.unwrap(),
            attendu,
            "k={k}: le rejeu complète ce qui reste"
        );
        assert_eq!(
            store.get_task_step_progress(t.id).await.unwrap(),
            (3, 3),
            "k={k}"
        );
    }

    // set_day_plan on a new day: the DayPlan draft, then the edges.
    for k in 1..=10 {
        let (store, _, t) = fixture(dir.path(), "day", k).await;
        let user = Uuid::new_v4();
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.set_day_plan(user, "d", &[t.id]).await.is_ok() {
            break;
        }
        atteints += 1;
        drop(store);
        let store = reouvrir(dir.path(), "day", k).await;
        store.set_day_plan(user, "d", &[t.id]).await.unwrap();
        assert_eq!(
            store.get_day_plan(user, "d").await.unwrap(),
            vec![(t.id, 0)],
            "k={k}"
        );
    }

    // delete_task: the marker, the steps marked, then detached, then the task.
    for k in 1..=30 {
        let (store, plan, t) = fixture(dir.path(), "delete", k).await;
        let etapes = store.get_task_steps(t.id).await.unwrap();
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.delete_task(t.id).await.is_ok() {
            break;
        }
        atteints += 1;
        drop(store);
        let store = reouvrir(dir.path(), "delete", k).await;
        let reste = store.get_task(t.id).await.unwrap().is_some();
        for s in &etapes {
            assert_eq!(
                store.get_step(s.id).await.unwrap().is_some(),
                reste,
                "k={k}: all or nothing"
            );
        }
        store.delete_task(t.id).await.unwrap();
        assert!(store.get_plan_tasks(plan.id).await.unwrap().is_empty());
    }
    assert!(atteints >= 20, "the cut points were reached ({atteints})");
}

/// Write straight into a closed base, as a migration or another tool would.
async fn write_foreign(base: &Path, f: impl FnOnce(&obrain_engine::ObrainDB) + Send + 'static) {
    let base = base.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let db = obrain_engine::ObrainDB::open(&base).unwrap();
        f(&db);
        db.commit_wal().unwrap();
        db.close().unwrap();
        std::fs::write(base.join(MARQUEUR_PO), SCHEMA_PO).unwrap();
    })
    .await
    .unwrap();
}

/// A task node with an unknown status: quarantined at open, its id reserved
/// (neither read nor created again), and its plan's reads refused rather than
/// partial. A foreign edge on a task makes its cascade delete a named refusal.
#[tokio::test]
async fn an_unreadable_task_is_quarantined_and_its_plan_refuses_a_partial_answer() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let plan = a_plan();
    let saine = a_task("saine", 1, 0);
    let etrangere = a_task("étrangère", 1, 1);
    {
        let store = open(&base).await;
        store.create_plan(&plan).await.unwrap();
        store.create_task(plan.id, &saine).await.unwrap();
        store.create_task(plan.id, &etrangere).await.unwrap();
        store.close().await.unwrap();
    }
    let illisible = Uuid::new_v4();
    let (plan_id, cible) = (plan.id, etrangere.id);
    write_foreign(&base, move |db| {
        let store = db.store();
        let trouver = |label: &str, id: Uuid| {
            store
                .nodes_by_label(label)
                .into_iter()
                .find(|n| {
                    db.get_node(*n).is_some_and(|x| {
                        x.get_property("id")
                            .and_then(|v| v.as_str().map(str::to_string))
                            == Some(id.to_string())
                    })
                })
                .unwrap()
        };
        let p = trouver("Plan", plan_id);
        let t = db.create_node_with_props(
            &["Task"],
            vec![
                ("id", Value::from(illisible.to_string())),
                ("description", Value::from("d")),
                ("status", Value::from("Archived")),
                ("created_at", Value::from(Utc::now().to_rfc3339())),
            ],
        );
        db.create_edge(p, t, "HAS_TASK");
        // A decision informing a task: a domain whose task side is not ported.
        let d = db.create_node_with_props(&["Decision"], vec![("id", Value::from("x"))]);
        db.create_edge(trouver("Task", cible), d, "INFORMED_BY");
    })
    .await;
    let store = open(&base).await;
    assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
    assert!(store.get_task(illisible).await.is_err());
    assert!(store.get_task(saine.id).await.unwrap().is_some());
    let err = store.get_plan_tasks(plan.id).await.unwrap_err();
    assert!(format!("{err:#}").contains("partie"), "{err:#}");
    assert!(store.compute_waves(plan.id).await.is_err());
    let mut double = a_task("double", 0, 2);
    double.id = illisible;
    let err = store.create_task(plan.id, &double).await.unwrap_err();
    assert!(format!("{err:#}").contains("réservé"), "{err:#}");

    let err = store.delete_task(etrangere.id).await.unwrap_err();
    assert_eq!(
        err.downcast_ref::<NonPorte>(),
        Some(&NonPorte("PlanStore::delete_task (cascade)"))
    );
    let err = store
        .get_task_with_full_details(etrangere.id)
        .await
        .unwrap_err();
    assert_eq!(
        err.downcast_ref::<NonPorte>(),
        Some(&NonPorte("PlanStore::get_task_with_full_details"))
    );
    assert!(store.get_task(etrangere.id).await.unwrap().is_some());
    // The healthy task answers its details: no decision, no file, exactly.
    let d = store
        .get_task_with_full_details(saine.id)
        .await
        .unwrap()
        .unwrap();
    assert!(d.decisions.is_empty() && d.modifies_files.is_empty());
}

/// The four task methods still refused are named (a `NonPorte` downcast), give
/// their cause, write nothing, and are listed among the plan refusals.
#[tokio::test]
async fn the_task_methods_still_refused_are_named_with_their_cause() {
    let store = tokio::task::spawn_blocking(ObrainGraphStore::open_temporary)
        .await
        .unwrap()
        .unwrap();
    let plan = a_plan();
    let t = a_task("t", 0, 0);
    store.create_plan(&plan).await.unwrap();
    store.create_task(plan.id, &t).await.unwrap();
    let ids = [t.id.to_string()];
    let erreurs: Vec<(anyhow::Error, &str)> = vec![
        (
            store.assign_task(t.id, "agent").await.unwrap_err(),
            "PlanStore::assign_task",
        ),
        (
            store.analyze_task_impact(t.id).await.unwrap_err(),
            "PlanStore::analyze_task_impact",
        ),
        (
            store.get_task_enrichment_counts(&ids).await.unwrap_err(),
            "PlanStore::get_task_enrichment_counts",
        ),
        (
            store.get_task_enrichment_data(&ids).await.unwrap_err(),
            "PlanStore::get_task_enrichment_data",
        ),
    ];
    for (err, nom) in erreurs {
        assert_eq!(err.downcast_ref::<NonPorte>(), Some(&NonPorte(nom)));
        assert!(format!("{err:#}").contains("non porté"), "{err:#}");
        assert!(refusees_plan().contains(&nom));
    }
    assert_eq!(
        store.get_task(t.id).await.unwrap().unwrap().assigned_to,
        None
    );
    // What is served is not in the refusals.
    for servie in [
        "PlanStore::create_task",
        "PlanStore::compute_waves",
        "PlanStore::create_step",
        "PlanStore::set_day_plan",
    ] {
        assert!(!refusees_plan().contains(&servie), "{servie}");
    }
}

/// A value above the 256-byte threshold goes to obrain's blob column, which the WAL
/// does not recover at this pin: every task and step write that carries one, or
/// that replaces or removes one, pays the full flush before it is acknowledged.
/// Observed through the engine's own fsync counter, without any `close()` (which
/// flushes everything, and so proves nothing here), as the project test does.
#[tokio::test]
async fn every_task_and_step_write_carrying_a_blob_sized_value_pays_the_full_flush() {
    let store = tokio::task::spawn_blocking(ObrainGraphStore::open_temporary)
        .await
        .unwrap()
        .unwrap();
    let plan = a_plan();
    store.create_plan(&plan).await.unwrap();
    let fsyncs = || store.fsyncs_du_wal().unwrap();
    let petit = "x".repeat(256);
    let grand = "x".repeat(257);

    // Cost of one write, measured on the counter.
    macro_rules! cout {
        ($e:expr) => {{
            let t0 = fsyncs();
            $e.await.unwrap();
            fsyncs() - t0
        }};
    }
    let verifier = |quoi: &str, petit: u64, grand: u64| {
        assert!(petit >= 1, "{quoi}: the WAL fsync ran");
        assert!(
            grand > petit,
            "{quoi}: {grand} fsyncs against {petit}, no full flush"
        );
    };

    // create_task: a string, then an element of a list.
    let mut a = a_task("a", 0, 0);
    a.description = petit.clone();
    let mut b = a_task("b", 0, 1);
    b.description = grand.clone();
    let mut c = a_task("c", 0, 2);
    c.tags = vec![grand.clone()];
    let (pa, pb, pc) = (
        cout!(store.create_task(plan.id, &a)),
        cout!(store.create_task(plan.id, &b)),
        cout!(store.create_task(plan.id, &c)),
    );
    verifier("create_task description", pa, pb);
    verifier("create_task tags", pa, pc);

    // update_task: a new long value; and a SHORT value replacing a long one.
    let maj = |d: &str| UpdateTaskRequest {
        description: Some(d.to_string()),
        ..Default::default()
    };
    let p = cout!(store.update_task(a.id, &maj(&petit)));
    let g = cout!(store.update_task(a.id, &maj(&grand)));
    verifier("update_task new value", p, g);
    let remplace = cout!(store.update_task(b.id, &maj("court")));
    verifier("update_task replaced value", p, remplace);
    let liste = cout!(store.update_task(
        a.id,
        &UpdateTaskRequest {
            acceptance_criteria: Some(vec![grand.clone()]),
            ..Default::default()
        }
    ));
    verifier("update_task list", p, liste);

    // update_task_enrichment: prompt_cache, execution_context.
    let p = cout!(store.update_task_enrichment(c.id, None, None, Some("court")));
    let g = cout!(store.update_task_enrichment(c.id, None, None, Some(&grand)));
    verifier("update_task_enrichment", p, g);

    // interrupt_task: a long reason; then the status change that REMOVES it.
    let p = cout!(store.interrupt_task(a.id, "court"));
    let g = cout!(store.interrupt_task(c.id, &grand));
    verifier("interrupt_task", p, g);
    let retrait = cout!(store.update_task_status(c.id, TaskStatus::InProgress));
    let sans = cout!(store.update_task_status(a.id, TaskStatus::InProgress));
    verifier("update_task_status removing a long reason", sans, retrait);

    // create_step, update_step.
    let mut s1 = a_step(0);
    s1.description = petit.clone();
    let mut s2 = a_step(1);
    s2.description = grand.clone();
    let p = cout!(store.create_step(a.id, &s1));
    let g = cout!(store.create_step(a.id, &s2));
    verifier("create_step", p, g);
    let maj_e = |d: &str| project_orchestrator::plan::models::UpdateStepRequest {
        description: None,
        status: None,
        verification: Some(d.to_string()),
    };
    let p = cout!(store.update_step(s1.id, &maj_e("court")));
    let g = cout!(store.update_step(s1.id, &maj_e(&grand)));
    verifier("update_step", p, g);

    // The values are there, read from the index (and so from what was decoded).
    assert_eq!(
        store.get_task(b.id).await.unwrap().unwrap().description,
        "court"
    );
    assert_eq!(
        store.get_step(s2.id).await.unwrap().unwrap().description,
        grand
    );
}

/// A plan quarantined at open (no `title`): its tasks cannot be read, so its
/// plan-level reads REFUSE instead of answering "no task" / "nothing to do".
#[tokio::test]
async fn a_quarantined_plan_refuses_its_task_reads() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    {
        let store = open(&base).await;
        store.close().await.unwrap();
    }
    let plan_id = Uuid::new_v4();
    write_foreign(&base, move |db| {
        let p = db.create_node_with_props(
            &["Plan"],
            vec![
                ("id", Value::from(plan_id.to_string())),
                ("description", Value::from("sans titre")),
            ],
        );
        let t = db.create_node_with_props(
            &["Task"],
            vec![
                ("id", Value::from(Uuid::new_v4().to_string())),
                ("description", Value::from("d")),
                ("status", Value::from("Pending")),
                ("created_at", Value::from(Utc::now().to_rfc3339())),
            ],
        );
        db.create_edge(p, t, "HAS_TASK");
    })
    .await;
    let store = open(&base).await;
    assert!(store.get_plan(plan_id).await.unwrap().is_none());
    for err in [
        store.get_plan_tasks(plan_id).await.map(|_| ()).unwrap_err(),
        store
            .get_next_available_task(plan_id)
            .await
            .map(|_| ())
            .unwrap_err(),
        store
            .find_blocked_tasks(plan_id)
            .await
            .map(|_| ())
            .unwrap_err(),
        store.compute_waves(plan_id).await.map(|_| ()).unwrap_err(),
    ] {
        assert!(format!("{err:#}").contains("quarantaine"), "{err:#}");
    }
}
