//! Automatic demotion: when the learnt choices do worse than the declarative
//! ones, the stage goes back to `shadow`.
//!
//! [`should_demote`] compares, for one class of work, the mean reward of the
//! last `demote_after` APPLIED decisions (the router's own choices, stage
//! `auto`) with the mean reward of the last `demote_after` NOT applied ones
//! (the declarative baseline). It says "demote" when the applied mean is more
//! than [`DEMOTION_MARGIN`] below the baseline, and only with `demote_after`
//! closed observations on each side: fewer is not evidence.
//!
//! [`apply_demotion`] writes `stage = shadow` into the routing setting of the
//! scope, leaves a `gotcha` note (best effort) and, when an emitter is given,
//! emits the `routing_demoted` event.
//!
//! GAP: the event bus has no dedicated entity type for routing yet, and adding
//! one touches the exhaustive entity lists of `events/`. The event is a
//! [`CrudEvent`] of entity `project` / action `updated` whose payload carries
//! `"event": "routing_demoted"`; [`demotion_event`] builds it so the
//! caller can wire a dedicated type later without touching this module.

use serde::Serialize;

use super::decision::CognitiveDecision;
use super::mode::{LearningStage, RoutingSettings};
use super::store::{DecisionFilter, RoutingArmStore};
use super::{load_routing, stored_routing, ROUTING_KEY};
use crate::chat::provider::settings::{project_scope, GLOBAL};
use crate::events::{CrudAction, CrudEvent, EntityType, EventEmitter};
use crate::neo4j::GraphStore;
use crate::notes::{Note, NoteImportance, NoteType};

/// How far below the baseline the applied mean must fall to demote.
pub const DEMOTION_MARGIN: f64 = 0.1;

/// Most recent decisions read to find the window.
const SCAN: usize = 2_000;

/// Why a class was demoted.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DemotionReason {
    /// Class key.
    pub class: String,
    /// Observations on each side.
    pub window: u32,
    /// Mean reward of the applied (learnt) decisions.
    pub applied_mean: f64,
    /// Mean reward of the declarative baseline.
    pub baseline_mean: f64,
}

/// What a demotion did.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Demotion {
    /// Scope written (`global` or `project:<slug>`).
    pub scope: String,
    /// Why.
    pub reason: DemotionReason,
    /// Whether the gotcha note was created.
    pub note_created: bool,
}

fn mean_reward(decisions: &[&CognitiveDecision], take: usize) -> Option<(f64, usize)> {
    let rewards: Vec<f64> = decisions
        .iter()
        .filter_map(|d| d.outcome.as_ref().and_then(|o| o.reward))
        .take(take)
        .collect();
    (!rewards.is_empty()).then(|| {
        (
            rewards.iter().sum::<f64>() / rewards.len() as f64,
            rewards.len(),
        )
    })
}

/// Whether the learnt choices of `class` should be demoted to shadow.
/// `None` unless the stage is `auto` and both sides have `demote_after`
/// closed observations.
pub async fn should_demote(
    store: &dyn RoutingArmStore,
    class: &str,
    project_slug: Option<&str>,
    settings: &RoutingSettings,
) -> anyhow::Result<Option<DemotionReason>> {
    if settings.stage != LearningStage::Auto {
        return Ok(None);
    }
    let window = settings.demote_after as usize;
    let decisions = store
        .decisions(&DecisionFilter {
            project_slug: project_slug.map(str::to_owned),
            limit: Some(SCAN),
            ..DecisionFilter::default()
        })
        .await?;
    // Newest first, same class, closed.
    let class_decisions: Vec<&CognitiveDecision> = decisions
        .iter()
        .filter(|d| d.signature.arm_key() == class)
        .filter(|d| d.outcome.as_ref().is_some_and(|o| o.reward.is_some()))
        .collect();
    let applied: Vec<&CognitiveDecision> = class_decisions
        .iter()
        .copied()
        .filter(|d| d.applied && d.stage == LearningStage::Auto)
        .collect();
    let baseline: Vec<&CognitiveDecision> = class_decisions
        .iter()
        .copied()
        .filter(|d| !d.applied)
        .collect();
    let (Some((applied_mean, a)), Some((baseline_mean, b))) = (
        mean_reward(&applied, window),
        mean_reward(&baseline, window),
    ) else {
        return Ok(None);
    };
    if a < window || b < window {
        return Ok(None);
    }
    Ok(
        (applied_mean < baseline_mean - DEMOTION_MARGIN).then(|| DemotionReason {
            class: class.to_owned(),
            window: settings.demote_after,
            applied_mean,
            baseline_mean,
        }),
    )
}

/// The event announcing a demotion.
pub fn demotion_event(project_slug: Option<&str>, reason: &DemotionReason) -> CrudEvent {
    let mut event = CrudEvent::new(
        EntityType::Project,
        CrudAction::Updated,
        project_slug.unwrap_or(GLOBAL),
    )
    .with_payload(serde_json::json!({
        "event": "routing_demoted",
        "stage": "shadow",
        "class": reason.class,
        "window": reason.window,
        "applied_mean": reason.applied_mean,
        "baseline_mean": reason.baseline_mean,
    }));
    if let Some(slug) = project_slug {
        event = event.with_project_id(slug);
    }
    event
}

/// Writes `stage = shadow` into the routing setting of the scope (the
/// project's when a slug is given, else the global one), leaves a `gotcha`
/// note and emits `routing_demoted`. `None` when the effective stage is
/// already not `auto`: nothing to demote, nothing written.
///
/// When the scope has no document of its own, the effective settings are
/// written with the stage changed, so a project that inherited `auto` from
/// the global document gets a shadow override of its own.
pub async fn apply_demotion(
    graph: &dyn GraphStore,
    project_slug: Option<&str>,
    reason: &DemotionReason,
    emitter: Option<&dyn EventEmitter>,
) -> anyhow::Result<Option<Demotion>> {
    let scope = match project_slug {
        Some(slug) => project_scope(slug),
        None => GLOBAL.to_owned(),
    };
    let mut settings = match stored_routing(graph, &scope).await? {
        Some(own) => own,
        None => load_routing(graph, project_slug).await?.0,
    };
    if settings.stage != LearningStage::Auto {
        return Ok(None);
    }
    settings.stage = LearningStage::Shadow;
    graph
        .put_llm_setting(&scope, ROUTING_KEY, &serde_json::to_string(&settings)?)
        .await?;
    let note_created = leave_note(graph, project_slug, reason).await;
    if let Some(emitter) = emitter {
        emitter.emit(demotion_event(project_slug, reason));
    }
    Ok(Some(Demotion {
        scope,
        reason: reason.clone(),
        note_created,
    }))
}

/// Best effort: a failing notes store never blocks a demotion.
async fn leave_note(
    graph: &dyn GraphStore,
    project_slug: Option<&str>,
    reason: &DemotionReason,
) -> bool {
    let project_id = match project_slug {
        Some(slug) => match graph.get_project_by_slug(slug).await {
            Ok(project) => project.map(|p| p.id),
            Err(error) => {
                tracing::warn!(%error, "routing demotion: project lookup failed");
                None
            }
        },
        None => None,
    };
    let content = format!(
        "Automatic routing was demoted to shadow for class '{}': over the last {} closed \
         decisions each side, the router's own choices earned a mean reward of {:.2} against \
         {:.2} for the declarative choice (margin {:.2}). Re-enable `auto` only after \
         looking at GET /api/chat/routing/report.",
        reason.class, reason.window, reason.applied_mean, reason.baseline_mean, DEMOTION_MARGIN
    );
    let mut note = Note::new(
        project_id,
        NoteType::Gotcha,
        content,
        "routing-demotion".to_owned(),
    );
    note.importance = NoteImportance::High;
    note.tags = vec!["routing".to_owned(), "demotion".to_owned()];
    match graph.create_note(&note).await {
        Ok(()) => true,
        Err(error) => {
            tracing::warn!(%error, "routing demotion: the gotcha note was not created");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::decision::{DecisionOutcome, Pick};
    use crate::chat::provider::cognitive::mode::ProviderRoutingMode;
    use crate::chat::provider::cognitive::signature::{TaskClass, TaskSignature};
    use crate::chat::provider::cognitive::store::InMemoryRoutingStore;
    use crate::neo4j::mock::MockGraphStore;
    use uuid::Uuid;

    const CLASS: &str = "simple";

    fn decision(
        class: TaskClass,
        applied: bool,
        reward: Option<f64>,
        age: i64,
    ) -> CognitiveDecision {
        CognitiveDecision {
            id: Uuid::new_v4(),
            at: chrono::Utc::now() - chrono::Duration::seconds(age),
            signature: TaskSignature {
                class,
                ..TaskSignature::utility(class, 9_000, Some("proj"))
            },
            chosen: Some(Pick::new("p", "m")),
            score: None,
            explored: false,
            reason: "t".into(),
            alternatives: vec![],
            applied,
            mode: ProviderRoutingMode::Mixed,
            stage: if applied {
                LearningStage::Auto
            } else {
                LearningStage::Shadow
            },
            session_id: None,
            task_id: None,
            run_id: None,
            turn_index: None,
            outcome: reward.map(|r| DecisionOutcome {
                reward: Some(r),
                ..DecisionOutcome::default()
            }),
            used: None,
        }
    }

    fn settings(window: u32) -> RoutingSettings {
        RoutingSettings {
            stage: LearningStage::Auto,
            demote_after: window,
            ..RoutingSettings::default()
        }
    }

    async fn seeded(applied: f64, baseline: f64, n: usize) -> InMemoryRoutingStore {
        let store = InMemoryRoutingStore::new();
        for i in 0..n {
            store
                .put_decision(&decision(TaskClass::Simple, true, Some(applied), i as i64))
                .await
                .unwrap();
            store
                .put_decision(&decision(
                    TaskClass::Simple,
                    false,
                    Some(baseline),
                    i as i64,
                ))
                .await
                .unwrap();
        }
        store
    }

    #[tokio::test]
    async fn a_learnt_arm_clearly_below_the_baseline_is_demoted() {
        let store = seeded(0.3, 0.7, 5).await;
        let reason = should_demote(&store, CLASS, Some("proj"), &settings(5))
            .await
            .unwrap()
            .expect("demote");
        assert_eq!(reason.class, CLASS);
        assert!((reason.applied_mean - 0.3).abs() < 1e-9);
        assert!((reason.baseline_mean - 0.7).abs() < 1e-9);
    }

    #[tokio::test]
    async fn within_the_margin_or_better_nothing_happens() {
        for (applied, baseline) in [(0.65, 0.7), (0.61, 0.7), (0.9, 0.7)] {
            let store = seeded(applied, baseline, 5).await;
            assert_eq!(
                should_demote(&store, CLASS, Some("proj"), &settings(5))
                    .await
                    .unwrap(),
                None,
                "{applied} vs {baseline}"
            );
        }
    }

    #[tokio::test]
    async fn too_few_observations_on_either_side_is_not_evidence() {
        let store = seeded(0.0, 1.0, 4).await;
        assert_eq!(
            should_demote(&store, CLASS, Some("proj"), &settings(5))
                .await
                .unwrap(),
            None
        );
        // Enough applied, too few baseline.
        let store = InMemoryRoutingStore::new();
        for i in 0..6 {
            store
                .put_decision(&decision(TaskClass::Simple, true, Some(0.0), i))
                .await
                .unwrap();
        }
        store
            .put_decision(&decision(TaskClass::Simple, false, Some(1.0), 0))
            .await
            .unwrap();
        assert_eq!(
            should_demote(&store, CLASS, Some("proj"), &settings(5))
                .await
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn only_the_last_window_counts_other_classes_and_open_decisions_are_ignored() {
        let store = seeded(0.3, 0.7, 5).await;
        // Older good applied decisions fall outside the window of 5.
        for i in 0..5 {
            store
                .put_decision(&decision(TaskClass::Simple, true, Some(1.0), 1_000 + i))
                .await
                .unwrap();
        }
        // Another class and unclosed decisions do not count.
        for i in 0..10 {
            store
                .put_decision(&decision(TaskClass::Complex, true, Some(1.0), i))
                .await
                .unwrap();
            store
                .put_decision(&decision(TaskClass::Simple, true, None, i))
                .await
                .unwrap();
        }
        assert!(should_demote(&store, CLASS, Some("proj"), &settings(5))
            .await
            .unwrap()
            .is_some());
        // Nothing to demote when the stage is not auto.
        let mut shadow = settings(5);
        shadow.stage = LearningStage::Shadow;
        assert_eq!(
            should_demote(&store, CLASS, Some("proj"), &shadow)
                .await
                .unwrap(),
            None
        );
    }

    fn reason() -> DemotionReason {
        DemotionReason {
            class: CLASS.into(),
            window: 5,
            applied_mean: 0.3,
            baseline_mean: 0.7,
        }
    }

    #[tokio::test]
    async fn applying_a_demotion_writes_shadow_leaves_a_note_and_emits() {
        let graph = MockGraphStore::new();
        graph
            .put_llm_setting(
                GLOBAL,
                ROUTING_KEY,
                r#"{"mode":"mixed","stage":"auto","demote_after":7}"#,
            )
            .await
            .unwrap();
        let bus = crate::events::EventBus::default();
        let mut rx = bus.subscribe();
        let demotion = apply_demotion(&graph, Some("proj"), &reason(), Some(&bus))
            .await
            .unwrap()
            .expect("demoted");
        assert_eq!(demotion.scope, "project:proj");
        assert!(demotion.note_created);
        // The project now has its own shadow override; the mode is inherited.
        let (effective, _) = load_routing(&graph, Some("proj")).await.unwrap();
        assert_eq!(effective.stage, LearningStage::Shadow);
        assert_eq!(effective.mode, ProviderRoutingMode::Mixed);
        assert_eq!(effective.demote_after, 7);
        // The global document is untouched.
        let global = stored_routing(&graph, GLOBAL).await.unwrap().unwrap();
        assert_eq!(global.stage, LearningStage::Auto);
        let event = rx.try_recv().expect("event");
        assert_eq!(event.payload["event"], "routing_demoted");
        // Already shadow: idempotent, nothing written twice.
        assert!(apply_demotion(&graph, Some("proj"), &reason(), Some(&bus))
            .await
            .unwrap()
            .is_none());
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn a_global_demotion_rewrites_the_global_document() {
        let graph = MockGraphStore::new();
        graph
            .put_llm_setting(GLOBAL, ROUTING_KEY, r#"{"stage":"auto"}"#)
            .await
            .unwrap();
        let demotion = apply_demotion(&graph, None, &reason(), None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(demotion.scope, GLOBAL);
        assert_eq!(
            stored_routing(&graph, GLOBAL).await.unwrap().unwrap().stage,
            LearningStage::Shadow
        );
    }

    #[test]
    fn the_event_names_the_demotion() {
        let event = demotion_event(Some("proj"), &reason());
        assert_eq!(event.payload["event"], "routing_demoted");
        assert_eq!(event.payload["class"], CLASS);
        assert_eq!(event.project_id.as_deref(), Some("proj"));
    }
}
