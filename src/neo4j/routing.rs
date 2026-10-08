//! Neo4j side of the cognitive routing: the bandit arms and the decision log.
//!
//! ```text
//! (:RoutingArm {class, provider_id, model, alpha, beta, n, cost_sum_usd, cost_n,
//!               latency_sum_ms, latency_n, updated_at})        one per triple
//! (:RoutingDecision {id, at, json, project_slug, session_id, task_id, run_id,
//!                    chosen_provider, chosen_model, applied})
//!     -[:ROUTED_FOR]-> (:ChatSession) | (:Task) | (:PlanRun)   best effort
//! ```
//!
//! [`Neo4jRoutingStore`] adapts any [`GraphStore`] to the
//! [`RoutingArmStore`] the scorer, the feedback and the report depend on.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, SecondsFormat, Utc};
use neo4rs::query;
use uuid::Uuid;

use super::client::Neo4jClient;
use super::traits::GraphStore;
use crate::chat::provider::cognitive::decision::{CognitiveDecision, DecisionOutcome};
use crate::chat::provider::cognitive::store::{
    ArmKey, ArmObservation, ArmStats, DecisionFilter, RoutingArmStore,
};

/// A time as a fixed-width UTC string, so that string order is time order.
fn stamp(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(SecondsFormat::Millis, true)
}

const ARM_RETURN: &str =
    "RETURN a.class AS class, a.provider_id AS provider_id, a.model AS model, \
     a.alpha AS alpha, a.beta AS beta, a.n AS n, a.cost_sum_usd AS cost_sum_usd, \
     a.cost_n AS cost_n, a.latency_sum_ms AS latency_sum_ms, a.latency_n AS latency_n, \
     a.updated_at AS updated_at";

fn arm_from_row(row: &neo4rs::Row) -> Result<ArmStats> {
    let updated_at: String = row.get("updated_at").unwrap_or_default();
    Ok(ArmStats {
        key: ArmKey::new(
            row.get::<String>("class")?,
            &row.get::<String>("provider_id")?,
            &row.get::<String>("model")?,
        ),
        alpha: row.get("alpha")?,
        beta: row.get("beta")?,
        n: row.get::<i64>("n")?.max(0) as u64,
        cost_sum_usd: row.get("cost_sum_usd")?,
        cost_n: row.get::<i64>("cost_n")?.max(0) as u64,
        latency_sum_ms: row.get::<i64>("latency_sum_ms")?.max(0) as u64,
        latency_n: row.get::<i64>("latency_n")?.max(0) as u64,
        updated_at: DateTime::parse_from_rfc3339(&updated_at)
            .map(|t| t.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
    })
}

fn decision_from_json(json: &str) -> Result<CognitiveDecision> {
    serde_json::from_str(json).context("unreadable routing decision")
}

impl Neo4jClient {
    /// One arm.
    pub async fn get_routing_arm_impl(&self, key: &ArmKey) -> Result<Option<ArmStats>> {
        let q = query(&format!(
            "MATCH (a:RoutingArm {{class: $class, provider_id: $provider_id, model: $model}}) {ARM_RETURN}"
        ))
        .param("class", key.class.clone())
        .param("provider_id", key.provider_id.clone())
        .param("model", key.model.clone());
        let mut result = self.graph.execute(q).await?;
        match result.next().await? {
            Some(row) => Ok(Some(arm_from_row(&row)?)),
            None => Ok(None),
        }
    }

    /// Every arm of a class.
    pub async fn list_routing_arms_impl(&self, class: &str) -> Result<Vec<ArmStats>> {
        let q = query(&format!(
            "MATCH (a:RoutingArm {{class: $class}}) {ARM_RETURN} ORDER BY a.provider_id, a.model"
        ))
        .param("class", class.to_string());
        let mut result = self.graph.execute(q).await?;
        let mut arms = Vec::new();
        while let Some(row) = result.next().await? {
            arms.push(arm_from_row(&row)?);
        }
        Ok(arms)
    }

    /// Records an observation in ONE statement, so two concurrent observations
    /// of the same arm both count (the unique constraint makes the `MERGE`
    /// create it once).
    pub async fn observe_routing_arm_impl(
        &self,
        key: &ArmKey,
        observation: &ArmObservation,
    ) -> Result<ArmStats> {
        let success = observation.reward >= observation.success_threshold;
        let q = query(&format!(
            "MERGE (a:RoutingArm {{class: $class, provider_id: $provider_id, model: $model}}) \
             ON CREATE SET a.alpha = 1.0, a.beta = 1.0, a.n = 0, a.cost_sum_usd = 0.0, \
                 a.cost_n = 0, a.latency_sum_ms = 0, a.latency_n = 0 \
             SET a.alpha = a.alpha + $d_alpha, a.beta = a.beta + $d_beta, a.n = a.n + 1, \
                 a.cost_sum_usd = a.cost_sum_usd + $cost, a.cost_n = a.cost_n + $cost_n, \
                 a.latency_sum_ms = a.latency_sum_ms + $latency, \
                 a.latency_n = a.latency_n + $latency_n, a.updated_at = $updated_at \
             {ARM_RETURN}"
        ))
        .param("class", key.class.clone())
        .param("provider_id", key.provider_id.clone())
        .param("model", key.model.clone())
        .param("d_alpha", if success { 1.0_f64 } else { 0.0 })
        .param("d_beta", if success { 0.0_f64 } else { 1.0 })
        .param("cost", observation.cost_usd.unwrap_or(0.0))
        .param("cost_n", i64::from(observation.cost_usd.is_some()))
        .param(
            "latency",
            observation.latency_ms.map(|l| l as i64).unwrap_or(0),
        )
        .param("latency_n", i64::from(observation.latency_ms.is_some()))
        .param("updated_at", stamp(Utc::now()));
        let mut result = self.graph.execute(q).await?;
        let row = result
            .next()
            .await?
            .context("the arm was not returned after its observation")?;
        arm_from_row(&row)
    }

    /// Stores or replaces a decision, then links it to the session, task and
    /// run when they exist. A link that cannot be made is logged, never an
    /// error: the decision is already stored.
    pub async fn put_routing_decision_impl(&self, decision: &CognitiveDecision) -> Result<()> {
        let json = serde_json::to_string(decision)?;
        let text = |value: Option<Uuid>| value.map(|v| v.to_string()).unwrap_or_default();
        let (chosen_provider, chosen_model) = decision
            .chosen
            .as_ref()
            .map(|p| (p.provider_id.clone(), p.model.clone()))
            .unwrap_or_default();
        let q = query(
            "MERGE (d:RoutingDecision {id: $id}) \
             SET d.at = $at, d.json = $json, d.project_slug = $project_slug, \
                 d.session_id = $session_id, d.task_id = $task_id, d.run_id = $run_id, \
                 d.chosen_provider = $chosen_provider, d.chosen_model = $chosen_model, \
                 d.applied = $applied",
        )
        .param("id", decision.id.to_string())
        .param("at", stamp(decision.at))
        .param("json", json)
        .param(
            "project_slug",
            decision.signature.project_slug.clone().unwrap_or_default(),
        )
        .param("session_id", text(decision.session_id))
        .param("task_id", text(decision.task_id))
        .param("run_id", text(decision.run_id))
        .param("chosen_provider", chosen_provider)
        .param("chosen_model", chosen_model)
        .param("applied", decision.applied);
        self.graph.run(q).await?;

        let links = [
            (
                decision.session_id,
                "MATCH (d:RoutingDecision {id: $id}) MATCH (x:ChatSession {id: $other}) \
                 MERGE (d)-[:ROUTED_FOR]->(x)",
            ),
            (
                decision.task_id,
                "MATCH (d:RoutingDecision {id: $id}) MATCH (x:Task {id: $other}) \
                 MERGE (d)-[:ROUTED_FOR]->(x)",
            ),
            (
                decision.run_id,
                "MATCH (d:RoutingDecision {id: $id}) MATCH (x:PlanRun {run_id: $other}) \
                 MERGE (d)-[:ROUTED_FOR]->(x)",
            ),
        ];
        for (other, cypher) in links {
            let Some(other) = other else { continue };
            let q = query(cypher)
                .param("id", decision.id.to_string())
                .param("other", other.to_string());
            if let Err(error) = self.graph.run(q).await {
                tracing::warn!(decision = %decision.id, %error, "routing decision link not made");
            }
        }
        Ok(())
    }

    /// One decision.
    pub async fn get_routing_decision_impl(&self, id: Uuid) -> Result<Option<CognitiveDecision>> {
        let q = query("MATCH (d:RoutingDecision {id: $id}) RETURN d.json AS json")
            .param("id", id.to_string());
        let mut result = self.graph.execute(q).await?;
        match result.next().await? {
            Some(row) => Ok(Some(decision_from_json(&row.get::<String>("json")?)?)),
            None => Ok(None),
        }
    }

    /// Decisions, newest first. An unreadable document is skipped and logged.
    pub async fn list_routing_decisions_impl(
        &self,
        filter: &DecisionFilter,
    ) -> Result<Vec<CognitiveDecision>> {
        let q = query(
            "MATCH (d:RoutingDecision) \
             WHERE ($project_slug = '' OR d.project_slug = $project_slug) \
               AND ($session_id = '' OR d.session_id = $session_id) \
               AND ($since = '' OR d.at >= $since) \
             RETURN d.json AS json ORDER BY d.at DESC SKIP $offset LIMIT $limit",
        )
        .param(
            "project_slug",
            filter.project_slug.clone().unwrap_or_default(),
        )
        .param(
            "session_id",
            filter.session_id.map(|s| s.to_string()).unwrap_or_default(),
        )
        .param("since", filter.since.map(stamp).unwrap_or_default())
        .param("offset", filter.offset as i64)
        .param("limit", filter.limit.unwrap_or(50) as i64);
        let mut result = self.graph.execute(q).await?;
        let mut found = Vec::new();
        while let Some(row) = result.next().await? {
            match decision_from_json(&row.get::<String>("json")?) {
                Ok(decision) => found.push(decision),
                Err(error) => tracing::warn!(%error, "routing decision skipped"),
            }
        }
        Ok(found)
    }

    /// Fills the outcome of a decision; `false` when it does not exist.
    pub async fn set_routing_decision_outcome_impl(
        &self,
        id: Uuid,
        outcome: DecisionOutcome,
    ) -> Result<bool> {
        let Some(mut decision) = self.get_routing_decision_impl(id).await? else {
            return Ok(false);
        };
        decision.outcome = Some(outcome);
        let q = query("MATCH (d:RoutingDecision {id: $id}) SET d.json = $json")
            .param("id", id.to_string())
            .param("json", serde_json::to_string(&decision)?);
        self.graph.run(q).await?;
        Ok(true)
    }
}

/// [`RoutingArmStore`] over the graph.
#[derive(Clone)]
pub struct Neo4jRoutingStore {
    graph: Arc<dyn GraphStore>,
}

impl Neo4jRoutingStore {
    /// A store over `graph`.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }
}

#[async_trait]
impl RoutingArmStore for Neo4jRoutingStore {
    async fn arm(&self, key: &ArmKey) -> Result<Option<ArmStats>> {
        self.graph.get_routing_arm(key).await
    }

    async fn arms_of_class(&self, class: &str) -> Result<Vec<ArmStats>> {
        self.graph.list_routing_arms(class).await
    }

    async fn observe(&self, key: &ArmKey, observation: &ArmObservation) -> Result<ArmStats> {
        self.graph.observe_routing_arm(key, observation).await
    }

    async fn put_decision(&self, decision: &CognitiveDecision) -> Result<()> {
        self.graph.put_routing_decision(decision).await
    }

    async fn decision(&self, id: Uuid) -> Result<Option<CognitiveDecision>> {
        self.graph.get_routing_decision(id).await
    }

    async fn decisions(&self, filter: &DecisionFilter) -> Result<Vec<CognitiveDecision>> {
        self.graph.list_routing_decisions(filter).await
    }

    async fn set_outcome(&self, id: Uuid, outcome: DecisionOutcome) -> Result<bool> {
        self.graph.set_routing_decision_outcome(id, outcome).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::decision::Pick;
    use crate::chat::provider::cognitive::mode::{LearningStage, ProviderRoutingMode};
    use crate::chat::provider::cognitive::signature::{TaskClass, TaskSignature};
    use crate::neo4j::mock::MockGraphStore;

    fn decision() -> CognitiveDecision {
        CognitiveDecision {
            id: Uuid::new_v4(),
            at: Utc::now(),
            signature: TaskSignature::utility(TaskClass::UtilityCompaction, 9_000, Some("po")),
            chosen: Some(Pick::new("glm", "big")),
            score: Some(0.5),
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

    #[test]
    fn a_stamp_orders_like_time_and_a_decision_round_trips_through_json() {
        let early = Utc::now();
        let late = early + chrono::Duration::milliseconds(5);
        assert!(stamp(early) < stamp(late));
        assert_eq!(stamp(early).len(), stamp(late).len());
        let d = decision();
        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(decision_from_json(&json).unwrap(), d);
        assert!(decision_from_json("not json").is_err());
    }

    // The Cypher above is exercised by the Neo4j CI job only; here the adapter
    // is checked over the mock graph, which wraps the in-memory store.
    #[tokio::test]
    async fn the_adapter_reads_and_writes_arms_and_decisions_through_the_graph() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let store = Neo4jRoutingStore::new(graph.clone());
        let key = ArmKey::new("simple", "glm", "big");
        assert!(store.arm(&key).await.unwrap().is_none());
        let obs = |reward: f64, cost: Option<f64>| ArmObservation {
            reward,
            success_threshold: 0.5,
            cost_usd: cost,
            latency_ms: Some(500),
        };
        store.observe(&key, &obs(1.0, Some(0.2))).await.unwrap();
        let arm = store.observe(&key, &obs(0.0, None)).await.unwrap();
        assert_eq!((arm.alpha, arm.beta, arm.n, arm.cost_n), (2.0, 2.0, 2, 1));
        assert_eq!(
            store.arms_of_class("simple").await.unwrap(),
            vec![arm.clone()]
        );
        assert_eq!(graph.get_routing_arm(&key).await.unwrap(), Some(arm));

        let d = decision();
        store.put_decision(&d).await.unwrap();
        assert_eq!(store.decision(d.id).await.unwrap(), Some(d.clone()));
        let outcome = DecisionOutcome {
            success: Some(true),
            ..DecisionOutcome::default()
        };
        assert!(store.set_outcome(d.id, outcome.clone()).await.unwrap());
        assert!(!store
            .set_outcome(Uuid::new_v4(), outcome.clone())
            .await
            .unwrap());
        let listed = store
            .decisions(&DecisionFilter {
                project_slug: Some("po".into()),
                ..DecisionFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].outcome, Some(outcome));
    }
}
