//! Persistence of the bandit arms and of the decisions.
//!
//! The trait is what the scorer, the feedback and the report depend on. The
//! in-memory implementation serves the tests of every caller; the Neo4j one
//! (`(:RoutingArm)`, `(:RoutingDecision)`) is written with the scorer.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::decision::{CognitiveDecision, DecisionOutcome};

/// One bandit arm: a class of work × an instance × a model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArmKey {
    /// Class key (`TaskSignature::arm_key`).
    pub class: String,
    /// Instance identifier.
    pub provider_id: String,
    /// Model identifier.
    pub model: String,
}

impl ArmKey {
    /// An arm key.
    pub fn new(class: impl Into<String>, provider_id: &str, model: &str) -> Self {
        Self {
            class: class.into(),
            provider_id: provider_id.to_owned(),
            model: model.to_owned(),
        }
    }

    /// The string the exploration scheduler keys its posteriors by.
    pub fn scheduler_key(&self) -> String {
        format!("{}|{}|{}", self.class, self.provider_id, self.model)
    }
}

/// What is known about an arm: a Beta posterior and running means.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmStats {
    /// The arm.
    pub key: ArmKey,
    /// Beta successes (prior included).
    pub alpha: f64,
    /// Beta failures (prior included).
    pub beta: f64,
    /// Observations.
    pub n: u64,
    /// Sum of the known marginal costs.
    pub cost_sum_usd: f64,
    /// Observations that had a known cost.
    pub cost_n: u64,
    /// Sum of the durations.
    pub latency_sum_ms: u64,
    /// Observations that had a duration.
    pub latency_n: u64,
    /// Last update.
    pub updated_at: DateTime<Utc>,
}

impl ArmStats {
    /// A fresh arm with the given prior.
    pub fn fresh(key: ArmKey, prior_alpha: f64, prior_beta: f64) -> Self {
        Self {
            key,
            alpha: prior_alpha,
            beta: prior_beta,
            n: 0,
            cost_sum_usd: 0.0,
            cost_n: 0,
            latency_sum_ms: 0,
            latency_n: 0,
            updated_at: Utc::now(),
        }
    }

    /// Mean of the Beta posterior.
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Mean known marginal cost per observation, `None` when never known.
    pub fn mean_cost_usd(&self) -> Option<f64> {
        (self.cost_n > 0).then(|| self.cost_sum_usd / self.cost_n as f64)
    }

    /// Mean duration, `None` when never known.
    pub fn mean_latency_ms(&self) -> Option<f64> {
        (self.latency_n > 0).then(|| self.latency_sum_ms as f64 / self.latency_n as f64)
    }
}

/// One observation of an arm.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmObservation {
    /// Reward in `[0, 1]`.
    pub reward: f64,
    /// Reward above which the observation counts as a success.
    pub success_threshold: f64,
    /// Marginal USD, `None` when unknown.
    pub cost_usd: Option<f64>,
    /// Duration in milliseconds.
    pub latency_ms: Option<u64>,
}

/// Filter for listing decisions.
#[derive(Debug, Clone, Default)]
pub struct DecisionFilter {
    /// Only decisions of this project.
    pub project_slug: Option<String>,
    /// Only decisions at or after this time.
    pub since: Option<DateTime<Utc>>,
    /// Only decisions of this session.
    pub session_id: Option<Uuid>,
    /// Page size; `None` is 50.
    pub limit: Option<usize>,
    /// Page offset.
    pub offset: usize,
}

/// Where arms and decisions live.
#[async_trait]
pub trait RoutingArmStore: Send + Sync {
    /// One arm.
    async fn arm(&self, key: &ArmKey) -> anyhow::Result<Option<ArmStats>>;
    /// Every arm of a class.
    async fn arms_of_class(&self, class: &str) -> anyhow::Result<Vec<ArmStats>>;
    /// Records an observation (creating the arm with the Beta(1, 1) prior when
    /// absent) and returns the arm as updated.
    async fn observe(&self, key: &ArmKey, observation: &ArmObservation)
        -> anyhow::Result<ArmStats>;
    /// Stores or replaces a decision.
    async fn put_decision(&self, decision: &CognitiveDecision) -> anyhow::Result<()>;
    /// One decision.
    async fn decision(&self, id: Uuid) -> anyhow::Result<Option<CognitiveDecision>>;
    /// Decisions, newest first.
    async fn decisions(&self, filter: &DecisionFilter) -> anyhow::Result<Vec<CognitiveDecision>>;
    /// Fills the outcome of a decision. `false` when the decision is unknown.
    async fn set_outcome(&self, id: Uuid, outcome: DecisionOutcome) -> anyhow::Result<bool>;
}

/// In-memory store, for tests and for a run without a graph.
#[derive(Default)]
pub struct InMemoryRoutingStore {
    arms: Mutex<HashMap<ArmKey, ArmStats>>,
    decisions: Mutex<Vec<CognitiveDecision>>,
}

impl InMemoryRoutingStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

#[async_trait]
impl RoutingArmStore for InMemoryRoutingStore {
    async fn arm(&self, key: &ArmKey) -> anyhow::Result<Option<ArmStats>> {
        Ok(lock(&self.arms).get(key).cloned())
    }

    async fn arms_of_class(&self, class: &str) -> anyhow::Result<Vec<ArmStats>> {
        Ok(lock(&self.arms)
            .values()
            .filter(|arm| arm.key.class == class)
            .cloned()
            .collect())
    }

    async fn observe(
        &self,
        key: &ArmKey,
        observation: &ArmObservation,
    ) -> anyhow::Result<ArmStats> {
        let mut arms = lock(&self.arms);
        let arm = arms
            .entry(key.clone())
            .or_insert_with(|| ArmStats::fresh(key.clone(), 1.0, 1.0));
        if observation.reward >= observation.success_threshold {
            arm.alpha += 1.0;
        } else {
            arm.beta += 1.0;
        }
        arm.n += 1;
        if let Some(cost) = observation.cost_usd {
            arm.cost_sum_usd += cost;
            arm.cost_n += 1;
        }
        if let Some(latency) = observation.latency_ms {
            arm.latency_sum_ms += latency;
            arm.latency_n += 1;
        }
        arm.updated_at = Utc::now();
        Ok(arm.clone())
    }

    async fn put_decision(&self, decision: &CognitiveDecision) -> anyhow::Result<()> {
        let mut decisions = lock(&self.decisions);
        match decisions.iter_mut().find(|d| d.id == decision.id) {
            Some(existing) => *existing = decision.clone(),
            None => decisions.push(decision.clone()),
        }
        Ok(())
    }

    async fn decision(&self, id: Uuid) -> anyhow::Result<Option<CognitiveDecision>> {
        Ok(lock(&self.decisions).iter().find(|d| d.id == id).cloned())
    }

    async fn decisions(&self, filter: &DecisionFilter) -> anyhow::Result<Vec<CognitiveDecision>> {
        let mut found: Vec<CognitiveDecision> = lock(&self.decisions)
            .iter()
            .filter(|d| filter.since.is_none_or(|since| d.at >= since))
            .filter(|d| filter.session_id.is_none_or(|s| d.session_id == Some(s)))
            .filter(|d| {
                filter
                    .project_slug
                    .as_deref()
                    .is_none_or(|slug| d.signature.project_slug.as_deref() == Some(slug))
            })
            .cloned()
            .collect();
        found.sort_by(|a, b| b.at.cmp(&a.at));
        Ok(found
            .into_iter()
            .skip(filter.offset)
            .take(filter.limit.unwrap_or(50))
            .collect())
    }

    async fn set_outcome(&self, id: Uuid, outcome: DecisionOutcome) -> anyhow::Result<bool> {
        let mut decisions = lock(&self.decisions);
        Ok(match decisions.iter_mut().find(|d| d.id == id) {
            Some(decision) => {
                decision.outcome = Some(outcome);
                true
            }
            None => false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::decision::Pick;
    use crate::chat::provider::cognitive::mode::{LearningStage, ProviderRoutingMode};
    use crate::chat::provider::cognitive::signature::{TaskClass, TaskSignature};

    fn decision(project: &str, at_offset_s: i64) -> CognitiveDecision {
        let mut signature =
            TaskSignature::utility(TaskClass::UtilityFeatureGraph, 9_000, Some(project));
        signature.attempt = 1;
        CognitiveDecision {
            id: Uuid::new_v4(),
            at: Utc::now() + chrono::Duration::seconds(at_offset_s),
            signature,
            chosen: Some(Pick::new("deepseek", "v4")),
            score: Some(0.7),
            explored: false,
            reason: "test".into(),
            alternatives: vec![],
            applied: false,
            mode: ProviderRoutingMode::Primary,
            stage: LearningStage::Shadow,
            session_id: None,
            task_id: None,
            run_id: None,
            turn_index: None,
            outcome: None,
            used: None,
        }
    }

    fn obs(reward: f64, cost: Option<f64>) -> ArmObservation {
        ArmObservation {
            reward,
            success_threshold: 0.5,
            cost_usd: cost,
            latency_ms: Some(1_000),
        }
    }

    #[tokio::test]
    async fn an_observation_updates_the_beta_and_the_means_and_an_unknown_cost_is_not_zero() {
        let store = InMemoryRoutingStore::new();
        let key = ArmKey::new("simple", "deepseek", "v4");
        assert!(store.arm(&key).await.unwrap().is_none());
        store.observe(&key, &obs(0.9, Some(0.10))).await.unwrap();
        store.observe(&key, &obs(0.2, None)).await.unwrap();
        let arm = store.arm(&key).await.unwrap().unwrap();
        assert_eq!((arm.alpha, arm.beta, arm.n), (2.0, 2.0, 2));
        // The unknown cost did not count as zero.
        assert_eq!(arm.cost_n, 1);
        assert_eq!(arm.mean_cost_usd(), Some(0.10));
        assert_eq!(arm.mean_latency_ms(), Some(1_000.0));
        assert!((arm.mean() - 0.5).abs() < 1e-9);
        let other = ArmKey::new("complex", "deepseek", "v4");
        assert_eq!(store.arms_of_class("simple").await.unwrap().len(), 1);
        assert!(store.arms_of_class("complex").await.unwrap().is_empty());
        assert!(store.arm(&other).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn decisions_are_listed_newest_first_filtered_and_paged_and_take_an_outcome() {
        let store = InMemoryRoutingStore::new();
        let old = decision("po", -10);
        let new = decision("po", 0);
        let elsewhere = decision("other", 5);
        for d in [&old, &new, &elsewhere] {
            store.put_decision(d).await.unwrap();
        }
        let po = store
            .decisions(&DecisionFilter {
                project_slug: Some("po".into()),
                ..DecisionFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(
            po.iter().map(|d| d.id).collect::<Vec<_>>(),
            vec![new.id, old.id]
        );
        let page = store
            .decisions(&DecisionFilter {
                limit: Some(1),
                offset: 1,
                ..DecisionFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(page.len(), 1);
        let outcome = DecisionOutcome {
            success: Some(true),
            reward: Some(0.8),
            ..DecisionOutcome::default()
        };
        assert!(store.set_outcome(new.id, outcome.clone()).await.unwrap());
        assert!(!store
            .set_outcome(Uuid::new_v4(), outcome.clone())
            .await
            .unwrap());
        assert_eq!(
            store.decision(new.id).await.unwrap().unwrap().outcome,
            Some(outcome)
        );
        // Putting the same id again replaces it.
        let mut replaced = new.clone();
        replaced.reason = "again".into();
        store.put_decision(&replaced).await.unwrap();
        assert_eq!(
            store
                .decisions(&DecisionFilter::default())
                .await
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn a_decision_names_its_arm_by_class_and_pick() {
        let d = decision("po", 0);
        assert_eq!(
            d.arm(),
            Some(ArmKey::new("utility.feature_graph", "deepseek", "v4"))
        );
        assert_eq!(
            d.arm().unwrap().scheduler_key(),
            "utility.feature_graph|deepseek|v4"
        );
        let mut none = d;
        none.chosen = None;
        assert_eq!(none.arm(), None);
    }
}
