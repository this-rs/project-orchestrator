//! `AnalyticsStore` on obrain, sub-lot A1 (`src/obrain/analytics.rs`): what only this
//! backend does. Persistence across a close and a reopen, an update cut at every
//! engine write and reopened (the update marker), a duplicated `WORKS_ON` edge, the
//! quarantine of an unreadable node (its key reserved), the refusal of an absent
//! target, and the 119 refusals named. The behaviour shared with Neo4j and the mock
//! is in `tests/store_conformance.rs` (`analytics_checks!`).
#![cfg(feature = "obrain-backend")]

use std::path::Path;

use chrono::{TimeZone, Utc};
use obrain_common::types::Value;
use project_orchestrator::chat::provider::cognitive::store::{ArmKey, ArmObservation};
use project_orchestrator::graph::models::{
    AnalysisProfile, TopologyRule, TopologyRuleType, TopologySeverity,
};
use project_orchestrator::neo4j::models::*;
use project_orchestrator::neo4j::traits::*;
use project_orchestrator::obrain::{
    methodes_qui_servent, refusees_analytics, Creation, Interruption, NonPorte, ObrainGraphStore,
    MARQUEUR_PO, SCHEMA_PO,
};
use uuid::Uuid;

async fn open(path: &Path) -> ObrainGraphStore {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || ObrainGraphStore::open(path, Creation::Permise))
        .await
        .unwrap()
        .unwrap()
}

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

fn a_project() -> ProjectNode {
    let id = Uuid::new_v4();
    ProjectNode {
        id,
        name: format!("analytics {id}"),
        slug: format!("analytics-{id}"),
        root_path: String::new(),
        description: None,
        created_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
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

fn obs(reward: f64) -> ArmObservation {
    ArmObservation {
        reward,
        success_threshold: 0.5,
        decay: 1.0,
        cost_usd: Some(0.5),
        latency_ms: Some(10),
    }
}

#[tokio::test]
async fn everything_that_serves_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (p, user, key) = (
        a_project(),
        "someone".to_string(),
        ArmKey::new("c", "prov", "m"),
    );
    let profile = AnalysisProfile {
        id: "prof".into(),
        project_id: Some(p.id.to_string()),
        name: "refactoring".into(),
        description: Some("d".into()),
        edge_weights: [("CALLS".to_string(), 1.5)].into_iter().collect(),
        fusion_weights: Default::default(),
        is_builtin: false,
    };
    let rule = TopologyRule {
        id: "rule".into(),
        project_id: p.id.to_string(),
        rule_type: TopologyRuleType::MaxFanOut,
        source_pattern: "src/**".into(),
        target_pattern: None,
        threshold: Some(0),
        severity: TopologySeverity::Error,
        description: "d".into(),
    };
    let (works, arm) = {
        let store = open(&base).await;
        store.create_project(&p).await.unwrap();
        store.create_analysis_profile(&profile).await.unwrap();
        store.create_topology_rule(&rule).await.unwrap();
        store.create_or_get_user_profile(&user).await.unwrap();
        store.upsert_works_on(&user, p.id).await.unwrap();
        store.upsert_works_on(&user, p.id).await.unwrap();
        store.observe_routing_arm(&key, &obs(1.0)).await.unwrap();
        store.observe_routing_arm(&key, &obs(0.0)).await.unwrap();
        let works = store.get_works_on(&user).await.unwrap();
        let arm = store.get_routing_arm(&key).await.unwrap().unwrap();
        store.close().await.unwrap();
        (works, arm)
    };
    let store = open(&base).await;
    assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
    assert_eq!(store.rapport_d_ouverture().mises_a_jour_achevees, 0);
    assert_eq!(
        serde_json::to_value(store.get_analysis_profile("prof").await.unwrap().unwrap()).unwrap(),
        serde_json::to_value(&profile).unwrap()
    );
    let r = &store.list_topology_rules(&p.id.to_string()).await.unwrap()[0];
    // A threshold of 0 is a threshold; an absent one stays absent (Neo4j writes 0 for both).
    assert_eq!((r.threshold, r.target_pattern.clone()), (Some(0), None));
    let after = store.get_works_on(&user).await.unwrap();
    let flat = |v: &[project_orchestrator::profile::WorksOnRelation]| {
        v.iter()
            .map(|w| (w.project_id, w.frequency, w.last_active))
            .collect::<Vec<_>>()
    };
    assert_eq!(flat(&after), flat(&works));
    assert_eq!(works[0].frequency, 2);
    assert_eq!(store.get_routing_arm(&key).await.unwrap().unwrap(), arm);
    assert_eq!((arm.alpha, arm.beta, arm.n), (2.0, 2.0, 2));
}

/// An arm's update writes seven properties: cut at every engine write, then
/// reopened, the arm is wholly the old one or wholly the new one, never a mix.
#[tokio::test]
async fn an_arm_update_cut_at_every_write_is_wholly_old_or_wholly_new() {
    let key = ArmKey::new("c", "prov", "m");
    let mut atteints = 0;
    for k in 1..=15 {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        store.observe_routing_arm(&key, &obs(1.0)).await.unwrap();
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.observe_routing_arm(&key, &obs(1.0)).await.is_ok() {
            break;
        }
        atteints += 1;
        assert!(
            store.get_routing_arm(&key).await.is_err(),
            "k={k}: indéterminé"
        );
        drop(store);
        let store = open(&base).await;
        let rapport = store.rapport_d_ouverture().clone();
        assert!(rapport.en_quarantaine.is_empty(), "k={k}: {rapport:?}");
        let a = store.get_routing_arm(&key).await.unwrap().unwrap();
        let ancien = (2.0, 1.0, 1, 0.5, 1, 10, 1);
        let nouveau = (3.0, 1.0, 2, 1.0, 2, 20, 2);
        let lu = (
            a.alpha,
            a.beta,
            a.n,
            a.cost_sum_usd,
            a.cost_n,
            a.latency_sum_ms,
            a.latency_n,
        );
        if k == 1 {
            // Cut before the marker: nothing written.
            assert_eq!((lu, rapport.mises_a_jour_achevees), (ancien, 0), "k={k}");
        } else {
            // The marker was on the node: opening finished the update.
            assert_eq!((lu, rapport.mises_a_jour_achevees), (nouveau, 1), "k={k}");
        }
    }
    assert!(
        atteints >= 8,
        "la coupure n'a atteint que {atteints} points"
    );
}

#[tokio::test]
async fn a_duplicated_works_on_edge_keeps_the_highest_frequency() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (p, user) = (a_project(), "u".to_string());
    {
        let store = open(&base).await;
        store.create_project(&p).await.unwrap();
        store.create_or_get_user_profile(&user).await.unwrap();
        store.upsert_works_on(&user, p.id).await.unwrap();
        store.close().await.unwrap();
    }
    // An update cut between the creation of the new edge and the deletion of the old.
    write_foreign(&base, |db| {
        let store = db.store();
        let u = store.nodes_by_label("UserProfile")[0];
        let pr = store.nodes_by_label("Project")[0];
        db.create_edge_with_props(
            u,
            pr,
            "WORKS_ON",
            [
                ("frequency", Value::from(5i64)),
                ("last_active", Value::from("2030-01-01T00:00:00Z")),
            ],
        );
    })
    .await;
    let store = open(&base).await;
    assert_eq!(store.rapport_d_ouverture().mises_a_jour_achevees, 1);
    let w = store.get_works_on(&user).await.unwrap();
    assert_eq!((w.len(), w[0].frequency), (1, 5));
    store.close().await.unwrap();
    // Repaired for good: the second open finds one edge.
    let store = open(&base).await;
    assert_eq!(store.rapport_d_ouverture().mises_a_jour_achevees, 0);
    assert_eq!(store.get_works_on(&user).await.unwrap().len(), 1);
}

#[tokio::test]
async fn an_unreadable_node_is_quarantined_and_its_key_is_reserved() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    open(&base).await.close().await.unwrap();
    write_foreign(&base, |db| {
        // No `created_at`: a user profile this backend cannot decode.
        db.create_node_with_props(
            &["UserProfile"],
            vec![
                ("id", Value::from(Uuid::new_v4().to_string())),
                ("user_id", Value::from("broken")),
            ],
        );
        // A rule with an unknown type.
        db.create_node_with_props(
            &["TopologyRule"],
            vec![
                ("id", Value::from("bad-rule")),
                ("project_id", Value::from("p")),
                ("rule_type", Value::from("whatever")),
                ("severity", Value::from("error")),
                ("source_pattern", Value::from("x")),
                ("description", Value::from("d")),
            ],
        );
        // An arm with no counters.
        db.create_node_with_props(
            &["RoutingArm"],
            vec![
                ("class", Value::from("c")),
                ("provider_id", Value::from("prov")),
                ("model", Value::from("m")),
            ],
        );
    })
    .await;
    let store = open(&base).await;
    let q = store.rapport_d_ouverture().en_quarantaine.clone();
    assert_eq!(q.len(), 3, "{q:?}");
    assert!(q.iter().any(|l| l.contains("UserProfile")));
    assert!(q.iter().any(|l| l.contains("TopologyRule")));
    assert!(q.iter().any(|l| l.contains("RoutingArm")));
    // Neither read, nor taken again, nor overwritten.
    assert!(store.get_user_profile("broken").await.is_err());
    assert!(store.create_or_get_user_profile("broken").await.is_err());
    assert!(store
        .get_routing_arm(&ArmKey::new("c", "prov", "m"))
        .await
        .is_err());
    assert!(store
        .observe_routing_arm(&ArmKey::new("c", "prov", "m"), &obs(1.0))
        .await
        .is_err());
    let rule = TopologyRule {
        id: "bad-rule".into(),
        project_id: "p".into(),
        rule_type: TopologyRuleType::NoCircular,
        source_pattern: "x".into(),
        target_pattern: None,
        threshold: None,
        severity: TopologySeverity::Error,
        description: "d".into(),
    };
    assert!(store.create_topology_rule(&rule).await.is_err());
    // Another key is unaffected.
    store.create_or_get_user_profile("fine").await.unwrap();
}

#[tokio::test]
async fn an_absent_target_refuses_except_for_works_on_which_is_neo4j_s_no_op() {
    let store = ObrainGraphStore::open_temporary().unwrap();
    let missing = Uuid::new_v4();
    let profile = AnalysisProfile {
        id: "p1".into(),
        project_id: Some(missing.to_string()),
        name: "n".into(),
        description: None,
        edge_weights: Default::default(),
        fusion_weights: Default::default(),
        is_builtin: false,
    };
    assert!(store.create_analysis_profile(&profile).await.is_err());
    assert!(store.get_analysis_profile("p1").await.unwrap().is_none());
    store.create_or_get_user_profile("u").await.unwrap();
    // Neo4j's no-op on an absent project or user, kept (the aggregator upserts a whole
    // flush with `?`): Ok, and nothing written.
    store.upsert_works_on("u", missing).await.unwrap();
    assert!(store.get_works_on("u").await.unwrap().is_empty());
    let p = a_project();
    store.create_project(&p).await.unwrap();
    store.upsert_works_on("nobody", p.id).await.unwrap();
    assert!(store.get_user_profile("nobody").await.unwrap().is_none());
    let nobody = project_orchestrator::profile::UserProfile::new("nobody");
    assert!(store.update_user_profile(&nobody).await.is_err());
    assert!(store.get_user_profile("nobody").await.unwrap().is_none());
    // A built-in profile is never deleted.
    let mut builtin = profile.clone();
    builtin.id = "b".into();
    builtin.project_id = None;
    builtin.is_builtin = true;
    store.create_analysis_profile(&builtin).await.unwrap();
    assert!(store.delete_analysis_profile("b").await.is_err());
    // The project cannot be deleted under the profiles: refused by name, listing them.
    let err = store.delete_project(p.id, &p.name).await.unwrap_err();
    assert!(format!("{err:#}").contains("analytics"), "{err:#}");
}

fn decision(
    i: i64,
) -> project_orchestrator::chat::provider::cognitive::decision::CognitiveDecision {
    use project_orchestrator::chat::provider::cognitive::decision::{CognitiveDecision, Pick};
    use project_orchestrator::chat::provider::cognitive::mode::{
        LearningStage, ProviderRoutingMode,
    };
    use project_orchestrator::chat::provider::cognitive::signature::{TaskClass, TaskSignature};
    CognitiveDecision {
        snapshot: None,
        arms_snapshot: Vec::new(),
        id: Uuid::new_v4(),
        at: Utc.timestamp_opt(1_700_000_000 + i, 0).unwrap(),
        signature: TaskSignature::utility(TaskClass::UtilityCompaction, 9_000, None),
        chosen: Some(Pick::new("p", "m")),
        score: None,
        explored: false,
        exploration_epsilon: 0.0,
        confidence: None,
        reason: format!("{i}"),
        alternatives: vec![],
        applied: false,
        mode: ProviderRoutingMode::Full,
        stage: LearningStage::Auto,
        session_id: None,
        task_id: None,
        run_id: None,
        turn_index: None,
        outcome: None,
        used: None,
    }
}

/// Without a limit the page is Neo4j's 50, newest first; an offset past the end
/// gives what is left.
#[tokio::test]
async fn the_decision_log_pages_newest_first_with_a_default_of_50() {
    use project_orchestrator::chat::provider::cognitive::store::DecisionFilter;
    let store = ObrainGraphStore::open_temporary().unwrap();
    for i in 0..60 {
        store.put_routing_decision(&decision(i)).await.unwrap();
    }
    let page = store
        .list_routing_decisions(&DecisionFilter::default())
        .await
        .unwrap();
    assert_eq!(page.len(), 50);
    assert_eq!(
        (page[0].reason.as_str(), page[49].reason.as_str()),
        ("59", "10")
    );
    let page = store
        .list_routing_decisions(&DecisionFilter {
            limit: Some(5),
            offset: 58,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(
        page.iter().map(|d| d.reason.as_str()).collect::<Vec<_>>(),
        ["1", "0"]
    );
}

/// A decision whose document does not decode is quarantined by name at open and
/// skipped by the list (Neo4j skips and logs); the rest of the log is served.
#[tokio::test]
async fn an_unreadable_decision_is_quarantined_and_the_log_still_reads() {
    use project_orchestrator::chat::provider::cognitive::store::DecisionFilter;
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let bonne = decision(1);
    {
        let store = open(&base).await;
        store.put_routing_decision(&bonne).await.unwrap();
        store.close().await.unwrap();
    }
    let casse = Uuid::new_v4();
    write_foreign(&base, move |db| {
        db.create_node_with_props(
            &["RoutingDecision"],
            vec![
                ("id", Value::from(casse.to_string())),
                ("at", Value::from("2030-01-01T00:00:00.000Z")),
                ("json", Value::from("{not json")),
                ("project_slug", Value::from("")),
                ("session_id", Value::from("")),
            ],
        );
    })
    .await;
    let store = open(&base).await;
    let q = store.rapport_d_ouverture().en_quarantaine.clone();
    assert!(q.len() == 1 && q[0].contains("RoutingDecision"), "{q:?}");
    let page = store
        .list_routing_decisions(&DecisionFilter::default())
        .await
        .unwrap();
    assert_eq!(page.iter().map(|d| d.id).collect::<Vec<_>>(), [bonne.id]);
    assert!(store.get_routing_decision(casse).await.is_err());
    // Its id stays reserved: not rewritten over the unreadable node.
    let mut d = decision(2);
    d.id = casse;
    assert!(store.put_routing_decision(&d).await.is_err());
}

/// Cut at every engine write of a multi-property rewrite, then reopened: the
/// entity is wholly old (cut before the marker) or wholly new (finished from it).
#[tokio::test]
async fn a_cut_rewrite_of_a_decision_a_profile_or_a_user_is_wholly_old_or_wholly_new() {
    let mut atteints = [0usize; 3];
    for k in 1..=20 {
        // A decision rewritten by id (the chat manager does it on every turn outcome).
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("d");
        let store = open(&base).await;
        let d = decision(1);
        store.put_routing_decision(&d).await.unwrap();
        let mut d2 = d.clone();
        d2.reason = "rewritten".into();
        d2.applied = true;
        d2.at = Utc.timestamp_opt(1_800_000_000, 0).unwrap();
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.put_routing_decision(&d2).await.is_err() {
            atteints[0] += 1;
            drop(store);
            let store = open(&base).await;
            let r = store.rapport_d_ouverture().clone();
            assert!(r.en_quarantaine.is_empty(), "k={k}: {r:?}");
            let lu = store.get_routing_decision(d.id).await.unwrap().unwrap();
            let page = store
                .list_routing_decisions(&Default::default())
                .await
                .unwrap();
            if k == 1 {
                assert_eq!((lu, r.mises_a_jour_achevees), (d.clone(), 0), "k={k}");
            } else {
                assert_eq!((lu, r.mises_a_jour_achevees), (d2.clone(), 1), "k={k}");
            }
            // The light entry (instant, filters) agrees with the document.
            assert_eq!(page[0].at, if k == 1 { d.at } else { d2.at }, "k={k}");
        }

        // An analysis profile rewritten by id.
        let base = dir.path().join("p");
        let store = open(&base).await;
        let mut prof = AnalysisProfile {
            id: "prof".into(),
            project_id: None,
            name: "old".into(),
            description: None,
            edge_weights: Default::default(),
            fusion_weights: Default::default(),
            is_builtin: false,
        };
        store.create_analysis_profile(&prof).await.unwrap();
        let ancien = prof.clone();
        prof.name = "new".into();
        prof.description = Some("d".into());
        prof.edge_weights.insert("CALLS".into(), 2.0);
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.create_analysis_profile(&prof).await.is_err() {
            atteints[1] += 1;
            drop(store);
            let store = open(&base).await;
            let r = store.rapport_d_ouverture().clone();
            assert!(r.en_quarantaine.is_empty(), "k={k}: {r:?}");
            let lu =
                serde_json::to_value(store.get_analysis_profile("prof").await.unwrap().unwrap())
                    .unwrap();
            let attendu = if k == 1 { &ancien } else { &prof };
            assert_eq!(lu, serde_json::to_value(attendu).unwrap(), "k={k}");
        }

        // A user profile update.
        let base = dir.path().join("u");
        let store = open(&base).await;
        let u = store.create_or_get_user_profile("u").await.unwrap();
        let mut u2 = u.clone();
        u2.verbosity = 0.9;
        u2.language = "fr".into();
        u2.interaction_count = 7;
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.update_user_profile(&u2).await.is_err() {
            atteints[2] += 1;
            drop(store);
            let store = open(&base).await;
            let r = store.rapport_d_ouverture().clone();
            assert!(r.en_quarantaine.is_empty(), "k={k}: {r:?}");
            let lu = store.get_user_profile("u").await.unwrap().unwrap();
            let lu = (lu.verbosity, lu.language, lu.interaction_count);
            if k == 1 {
                assert_eq!(lu, (0.5, "en".to_string(), 0), "k={k}");
            } else {
                assert_eq!(lu, (0.9, "fr".to_string(), 7), "k={k}");
            }
        }
    }
    // Decision rewrite: marker + 9 properties + removal; profile: + 7; user: + 6.
    assert!(
        atteints[0] >= 10 && atteints[1] >= 8 && atteints[2] >= 7,
        "{atteints:?}"
    );
}

/// The real cut of `upsert_works_on` (new edge, then old edge deleted): cut before
/// the creation, nothing moved; cut after it, two edges, and opening keeps the new.
#[tokio::test]
async fn a_cut_works_on_upsert_keeps_the_old_or_the_new_frequency() {
    let mut atteints = 0;
    for k in 1..=4 {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = a_project();
        store.create_project(&p).await.unwrap();
        store.create_or_get_user_profile("u").await.unwrap();
        store.upsert_works_on("u", p.id).await.unwrap();
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.upsert_works_on("u", p.id).await.is_ok() {
            break;
        }
        atteints += 1;
        drop(store);
        let store = open(&base).await;
        let r = store.rapport_d_ouverture().clone();
        let w = store.get_works_on("u").await.unwrap();
        assert_eq!(w.len(), 1, "k={k}");
        if k == 1 {
            assert_eq!((w[0].frequency, r.mises_a_jour_achevees), (1, 0), "k={k}");
        } else {
            assert_eq!((w[0].frequency, r.mises_a_jour_achevees), (2, 1), "k={k}");
        }
    }
    assert_eq!(
        atteints, 2,
        "two engine writes: the new edge, the old one deleted"
    );
}

/// A `WORKS_ON` edge that does not read is quarantined AND its relation reserved:
/// an upsert would otherwise write a second edge at frequency 1 and lose its count.
#[tokio::test]
async fn an_unreadable_works_on_edge_reserves_its_relation() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let p = a_project();
    {
        let store = open(&base).await;
        store.create_project(&p).await.unwrap();
        store.create_or_get_user_profile("u").await.unwrap();
        store.close().await.unwrap();
    }
    write_foreign(&base, |db| {
        let store = db.store();
        let u = store.nodes_by_label("UserProfile")[0];
        let pr = store.nodes_by_label("Project")[0];
        // No `last_active`: unreadable.
        db.create_edge_with_props(u, pr, "WORKS_ON", [("frequency", Value::from(9i64))]);
    })
    .await;
    let store = open(&base).await;
    assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
    assert!(store.get_works_on("u").await.is_err());
    assert!(store.upsert_works_on("u", p.id).await.is_err());
    // Nothing was written: still one edge after a reopen.
    store.close().await.unwrap();
    drop(store);
    let store = open(&base).await;
    assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
}

/// A quarantined node never makes a list silently partial; an unrelated scope still reads.
#[tokio::test]
async fn a_list_that_could_miss_a_quarantined_node_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    open(&base).await.close().await.unwrap();
    write_foreign(&base, |db| {
        db.create_node_with_props(
            &["RoutingArm"],
            vec![
                ("class", Value::from("c")),
                ("provider_id", Value::from("prov")),
                ("model", Value::from("m")),
            ],
        );
        db.create_node_with_props(
            &["TopologyRule"],
            vec![("id", Value::from("r")), ("project_id", Value::from("p1"))],
        );
        db.create_node_with_props(
            &["AnalysisProfile"],
            vec![("id", Value::from("a")), ("project_id", Value::from("p1"))],
        );
    })
    .await;
    let store = open(&base).await;
    assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 3);
    assert!(store.list_routing_arms("c").await.is_err());
    assert!(store.list_routing_arms("other").await.unwrap().is_empty());
    // The quarantined arm itself: neither read nor observed.
    let k = ArmKey::new("c", "prov", "m");
    let e = store.observe_routing_arm(&k, &obs(1.0)).await.unwrap_err();
    assert!(format!("{e:#}").contains("réservé"), "{e:#}");
    assert!(store.list_topology_rules("p1").await.is_err());
    assert!(store.list_topology_rules("p2").await.unwrap().is_empty());
    assert!(store.list_analysis_profiles(Some("p1")).await.is_err());
    assert!(store
        .list_analysis_profiles(Some("p2"))
        .await
        .unwrap()
        .is_empty());
    assert!(store.list_analysis_profiles(None).await.unwrap().is_empty());
}

/// Reserved keys never collide: a quarantined node's `id` does not reserve a
/// `user_id` of the same text, and parts are not joined by a separator.
#[tokio::test]
async fn reserved_keys_do_not_collide() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    open(&base).await.close().await.unwrap();
    write_foreign(&base, |db| {
        // An unreadable profile whose `id` is the text "alice".
        db.create_node_with_props(
            &["UserProfile"],
            vec![
                ("id", Value::from("alice")),
                ("user_id", Value::from("zzz")),
            ],
        );
        // An unreadable arm ("a|b", "c", "m").
        db.create_node_with_props(
            &["RoutingArm"],
            vec![
                ("class", Value::from("a|b")),
                ("provider_id", Value::from("c")),
                ("model", Value::from("m")),
            ],
        );
    })
    .await;
    let store = open(&base).await;
    assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 2);
    // "alice" is a user_id nobody reserved.
    store.create_or_get_user_profile("alice").await.unwrap();
    assert!(store.create_or_get_user_profile("zzz").await.is_err());
    // ("a", "b|c", "m") is another arm than ("a|b", "c", "m").
    store
        .observe_routing_arm(&ArmKey::new("a", "b|c", "m"), &obs(1.0))
        .await
        .unwrap();
    assert!(store
        .observe_routing_arm(&ArmKey::new("a|b", "c", "m"), &obs(1.0))
        .await
        .is_err());
}

/// The 119 methods of the other sub-lots refuse by their own name, and none is
/// counted as serving.
#[tokio::test]
async fn the_other_sub_lots_refuse_by_name() {
    let store = ObrainGraphStore::open_temporary().unwrap();
    let id = Uuid::new_v4();
    macro_rules! refuse {
        ($call:expr, $name:literal) => {{
            let e = $call.await.unwrap_err();
            assert_eq!(
                e.downcast_ref::<NonPorte>(),
                Some(&NonPorte($name)),
                "{e:#}"
            );
            assert!(refusees_analytics().contains(&$name));
        }};
    }
    refuse!(store.get_synapses(id), "AnalyticsStore::get_synapses");
    refuse!(
        store.get_project_import_edges(id),
        "AnalyticsStore::get_project_import_edges"
    );
    refuse!(
        store.compute_churn_scores(id),
        "AnalyticsStore::compute_churn_scores"
    );
    refuse!(
        store.get_deep_maintenance_times(),
        "AnalyticsStore::get_deep_maintenance_times"
    );
    refuse!(
        store.mark_deep_maintenance(id),
        "AnalyticsStore::mark_deep_maintenance"
    );
    refuse!(
        store.weaken_node_synapses(id, 0.1, 0.0),
        "AnalyticsStore::weaken_node_synapses"
    );
    refuse!(
        store.decay_project_synapses(id, &Default::default()),
        "AnalyticsStore::decay_project_synapses"
    );
    refuse!(
        store.ensure_project_canonical_paths(),
        "AnalyticsStore::ensure_project_canonical_paths"
    );
    refuse!(
        store.has_context_cards("p"),
        "AnalyticsStore::has_context_cards"
    );
    refuse!(store.schema_catalog(), "AnalyticsStore::schema_catalog");
    refuse!(
        store.causal_walk("Note", "x", 2),
        "AnalyticsStore::causal_walk"
    );
    assert_eq!(refusees_analytics().len(), 119);
    assert!(refusees_analytics()
        .iter()
        .all(|m| m.starts_with("AnalyticsStore::")));
    let servent = methodes_qui_servent();
    assert!(refusees_analytics().iter().all(|m| !servent.contains(m)));
    let n = servent
        .iter()
        .filter(|m| m.starts_with("AnalyticsStore::"))
        .count();
    assert_eq!(
        n, 26,
        "25 methods serve and `run_data_migrations` answers its no-migration"
    );
    // The one that cannot refuse says what it is.
    assert!(store.run_data_migrations().await.is_empty());
    assert!(store.health_check().await.unwrap());
    store.close().await.unwrap();
    assert!(
        !store.health_check().await.unwrap(),
        "a closed base is not healthy"
    );
}
