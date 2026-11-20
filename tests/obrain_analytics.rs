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
async fn an_absent_target_refuses_instead_of_writing_nothing_and_saying_ok() {
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
    assert!(store.upsert_works_on("u", missing).await.is_err());
    let p = a_project();
    store.create_project(&p).await.unwrap();
    assert!(store.upsert_works_on("nobody", p.id).await.is_err());
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

#[tokio::test]
async fn the_decisions_page_is_decoded_not_the_whole_log() {
    use project_orchestrator::chat::provider::cognitive::decision::{CognitiveDecision, Pick};
    use project_orchestrator::chat::provider::cognitive::mode::{
        LearningStage, ProviderRoutingMode,
    };
    use project_orchestrator::chat::provider::cognitive::signature::{TaskClass, TaskSignature};
    use project_orchestrator::chat::provider::cognitive::store::DecisionFilter;
    let store = ObrainGraphStore::open_temporary().unwrap();
    for i in 0..30 {
        let d = CognitiveDecision {
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
        };
        store.put_routing_decision(&d).await.unwrap();
    }
    let page = store
        .list_routing_decisions(&DecisionFilter::default())
        .await
        .unwrap();
    // Default page: 50, so all 30, newest first.
    assert_eq!(page.len(), 30);
    assert_eq!(page[0].reason, "29");
    let page = store
        .list_routing_decisions(&DecisionFilter {
            limit: Some(5),
            offset: 28,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(
        page.iter().map(|d| d.reason.as_str()).collect::<Vec<_>>(),
        ["1", "0"]
    );
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
