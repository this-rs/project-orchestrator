//! The runner side of cognitive routing (B-R7): decide before a session opens,
//! close the decision when the work ends.
//!
//! The runner, a delegation and the utility calls all follow the same two
//! steps. Before the [`ChatRequest`](crate::chat::types::ChatRequest) is built
//! they call [`RoutingHandle::decide`] with the signature of the work; the
//! decision id travels on the request (`routing_decision_id`) and the resolver
//! turns an applied decision into the `project_rule` slot. When the work ends
//! they call [`RoutingHandle::close`], which feeds the outcome back to the arm
//! that was chosen. Both are best effort: a routing failure never stops work.
//!
//! No handle means nothing is decided and nothing recorded: today's behaviour.

use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use uuid::Uuid;

use crate::chat::provider::cognitive::candidates::{ModelFacts, Slot};
use crate::chat::provider::cognitive::decision::{CognitiveDecision, DecideRequest, Decider};
use crate::chat::provider::cognitive::feedback::{close_decision, Outcome};
use crate::chat::provider::cognitive::load_routing;
use crate::chat::provider::cognitive::signature::TaskSignature;
use crate::chat::provider::cognitive::store::RoutingArmStore;
use crate::neo4j::agent_execution::{AgentExecutionNode, AgentExecutionStatus};
use crate::neo4j::GraphStore;

/// Every (instance, model) a project could use, with its facts. The scorer's
/// wiring supplies the real one; tests supply a fixed list.
#[async_trait]
pub trait RoutingPool: Send + Sync {
    /// The pool for a project (`None` = no project).
    async fn facts(&self, project_slug: Option<&str>) -> Vec<ModelFacts>;
}

/// A pool that is always the same list.
pub struct FixedPool(pub Vec<ModelFacts>);

#[async_trait]
impl RoutingPool for FixedPool {
    async fn facts(&self, _project_slug: Option<&str>) -> Vec<ModelFacts> {
        self.0.clone()
    }
}

/// The production pool: the candidates the chat manager routes sessions from,
/// so a plan run and a chat session see the same instances, consents and probes.
pub struct ChatRoutingPool {
    /// Supplies the instances, the consents and the probe memory of the chat router.
    pub manager: Arc<crate::chat::ChatManager>,
}

#[async_trait]
impl RoutingPool for ChatRoutingPool {
    async fn facts(&self, project_slug: Option<&str>) -> Vec<ModelFacts> {
        self.manager.routing_pool_for(project_slug).await
    }
}

/// What the runner, a delegation or a utility call needs to route work.
#[derive(Clone)]
pub struct RoutingHandle {
    /// Takes the decisions.
    pub decider: Arc<dyn Decider>,
    /// Receives the outcomes.
    pub store: Arc<dyn RoutingArmStore>,
    /// Supplies the candidates.
    pub pool: Arc<dyn RoutingPool>,
}

static INSTALLED: RwLock<Option<Arc<RoutingHandle>>> = RwLock::new(None);

/// Installs the process-wide handle (startup wiring, after the decider is
/// built). `None` uninstalls it.
pub fn install(handle: Option<Arc<RoutingHandle>>) {
    *INSTALLED.write().unwrap_or_else(|e| e.into_inner()) = handle;
}

/// The process-wide handle, `None` until something installs one.
pub fn installed() -> Option<Arc<RoutingHandle>> {
    INSTALLED.read().unwrap_or_else(|e| e.into_inner()).clone()
}

impl RoutingHandle {
    /// A handle.
    pub fn new(
        decider: Arc<dyn Decider>,
        store: Arc<dyn RoutingArmStore>,
        pool: Arc<dyn RoutingPool>,
    ) -> Self {
        Self {
            decider,
            store,
            pool,
        }
    }

    /// Decides for `signature`. `None` when the decider fails (logged): the
    /// work then runs as it did before routing existed.
    pub async fn decide(
        &self,
        graph: &dyn GraphStore,
        signature: TaskSignature,
        slot: Slot,
        task_id: Option<Uuid>,
        run_id: Option<Uuid>,
    ) -> Option<CognitiveDecision> {
        let project = signature.project_slug.clone();
        let settings = match load_routing(graph, project.as_deref()).await {
            Ok((settings, _scope)) => settings,
            Err(error) => {
                tracing::warn!(%error, "routing settings unreadable: no decision taken");
                return None;
            }
        };
        let pool = self.pool.facts(project.as_deref()).await;
        let mut request = DecideRequest::new(signature, settings, pool);
        request.slot = slot;
        request.task_id = task_id;
        request.run_id = run_id;
        match self.decider.decide(&request).await {
            Ok(decision) => Some(decision),
            Err(error) => {
                tracing::warn!(%error, "routing decision failed: work runs unrouted");
                None
            }
        }
    }

    /// Closes a decision with what happened. The settings are those of the
    /// project the decision was taken for. Best effort.
    pub async fn close(&self, graph: &dyn GraphStore, decision_id: Uuid, outcome: &Outcome) {
        let project_slug = match self.store.decision(decision_id).await {
            Ok(Some(decision)) => decision.signature.project_slug,
            Ok(None) => return,
            Err(error) => {
                tracing::warn!(%error, %decision_id, "routing decision unreadable: left open");
                return;
            }
        };
        let settings = match load_routing(graph, project_slug.as_deref()).await {
            Ok((settings, _scope)) => settings,
            Err(error) => {
                tracing::warn!(%error, %decision_id, "routing settings unreadable: decision left open");
                return;
            }
        };
        if let Err(error) =
            close_decision(self.store.as_ref(), &settings, decision_id, outcome).await
        {
            tracing::warn!(%error, %decision_id, "closing the routing decision failed");
        }
    }
}

/// Whether the request already carries a choice somebody made: such a slot is
/// explicit, never filtered and never substituted.
pub fn slot_for(explicit: bool) -> Slot {
    if explicit {
        Slot::Explicit
    } else {
        Slot::Automatic
    }
}

/// The model preference of the persona a task names, resolved in the task's
/// project scope (then among the global personas). `task_persona` is what
/// `TaskNode.persona` holds: a persona id (possibly JSON-quoted), a name, or the
/// `name:complexity` form the runner writes back. An unknown persona, an
/// unreadable graph or an empty preference is `None`.
///
/// The returned text is free text: it is only ever matched against the alias
/// table, never stored, logged or sent anywhere.
pub async fn persona_preference(
    graph: &dyn GraphStore,
    project_id: Option<Uuid>,
    task_persona: &str,
) -> Option<String> {
    let key = task_persona.trim().trim_matches('"').trim();
    if key.is_empty() {
        return None;
    }
    let mut pool = Vec::new();
    if let Some(pid) = project_id {
        if let Ok((found, _)) = graph.list_personas(pid, None, 1000, 0).await {
            pool.extend(found);
        }
    }
    if let Ok(global) = graph.list_global_personas().await {
        pool.extend(global);
    }
    let by_id = key.parse::<Uuid>().ok();
    let persona = pool
        .iter()
        .find(|p| match by_id {
            Some(id) => p.id == id,
            None => p.name == key,
        })
        .or_else(|| {
            let (name, _) = key.rsplit_once(':')?;
            pool.iter().find(|p| p.name == name)
        })?;
    persona
        .model_preference
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .map(str::to_string)
}

/// Whether the task's own verification passed, read from the node's
/// `verification_json`. `None` when nothing ran or the shape is not known.
///
/// Understood shapes: `{"passed": bool}` / `{"all_passed": bool}`, or an object
/// of checks whose values are `"pass"`/`"fail"` strings or booleans.
pub fn verification_passed(json: Option<&str>) -> Option<bool> {
    let value: serde_json::Value = serde_json::from_str(json?).ok()?;
    let object = value.as_object()?;
    for key in ["passed", "all_passed"] {
        if let Some(flag) = object.get(key).and_then(|v| v.as_bool()) {
            return Some(flag);
        }
    }
    let mut verdicts = object.values().filter_map(|v| match v {
        serde_json::Value::Bool(b) => Some(*b),
        serde_json::Value::String(s) => match s.to_ascii_lowercase().as_str() {
            "pass" | "passed" | "ok" => Some(true),
            "fail" | "failed" | "error" => Some(false),
            _ => None,
        },
        _ => None,
    });
    let first = verdicts.next()?;
    Some(verdicts.fold(first, |all, next| all && next))
}

/// The outcome of a closed attempt, from its AgentExecution node.
///
/// The cost is the marginal one only: a subscription or a free endpoint counts
/// for nothing, and a cost whose basis was never recorded is unknown, not zero.
pub fn outcome_of_attempt(closed: &AgentExecutionNode) -> Outcome {
    let marginal = closed
        .cost_basis
        .as_deref()
        .filter(|basis| crate::chat::cost::counts_toward_budget(Some(basis)));
    Outcome {
        success: Some(closed.status == AgentExecutionStatus::Completed),
        attempts: closed.attempt.max(1),
        cost_usd: marginal.map(|_| closed.cost_usd),
        duration_ms: Some((closed.duration_secs.max(0.0) * 1000.0).round() as u64),
        interrupted: closed.status == AgentExecutionStatus::Interrupted,
        user_overrode_model: false,
        verification_passed: verification_passed(closed.verification_json.as_deref()),
        input_tokens: closed.tokens_in,
        output_tokens: closed.tokens_out,
    }
}

/// The outcome of a delegated session, from its `Result` event. `cost` is the
/// event's `{ usd?, basis }`: only a marginal basis with an amount is a cost.
pub fn outcome_of_result(
    is_error: bool,
    duration_ms: u64,
    cost: Option<&serde_json::Value>,
) -> Outcome {
    let basis = cost.and_then(|c| c.get("basis")).and_then(|b| b.as_str());
    let usd = cost.and_then(|c| c.get("usd")).and_then(|u| u.as_f64());
    Outcome {
        success: Some(!is_error),
        attempts: 1,
        cost_usd: basis
            .filter(|b| crate::chat::cost::counts_toward_budget(Some(b)))
            .and(usd),
        duration_ms: Some(duration_ms),
        interrupted: false,
        user_overrode_model: false,
        verification_passed: None,
        input_tokens: None,
        output_tokens: None,
    }
}

/// Fakes shared by the tests of every caller of the decider.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::chat::provider::cognitive::decision::Pick;
    use crate::chat::provider::cognitive::mode::LearningStage;
    use crate::chat::provider::cognitive::store::InMemoryRoutingStore;
    use std::sync::Mutex;

    /// A decider that records what it was asked, persists its decisions in the
    /// in-memory store and chooses the same pair every time.
    pub(crate) struct FakeDecider {
        pub applied: bool,
        pub pick: Pick,
        pub store: Arc<InMemoryRoutingStore>,
        pub seen: Mutex<Vec<DecideRequest>>,
    }

    impl FakeDecider {
        /// A fake and the handle that routes through it.
        pub(crate) fn handle(applied: bool) -> (Arc<FakeDecider>, Arc<RoutingHandle>) {
            let store = Arc::new(InMemoryRoutingStore::new());
            let fake = Arc::new(FakeDecider {
                applied,
                pick: Pick::new(
                    crate::neo4j::agent_execution::DEFAULT_PROVIDER_ID,
                    "fake-model",
                ),
                store: store.clone(),
                seen: Mutex::new(Vec::new()),
            });
            let handle = Arc::new(RoutingHandle::new(
                fake.clone(),
                store,
                Arc::new(FixedPool(Vec::new())),
            ));
            (fake, handle)
        }

        pub(crate) fn requests(&self) -> Vec<DecideRequest> {
            self.seen.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Decider for FakeDecider {
        async fn decide(&self, request: &DecideRequest) -> anyhow::Result<CognitiveDecision> {
            self.seen.lock().unwrap().push(request.clone());
            let explicit = request.slot == Slot::Explicit;
            let decision = CognitiveDecision {
                id: Uuid::new_v4(),
                at: chrono::Utc::now(),
                signature: request.signature.clone(),
                chosen: (!explicit).then(|| self.pick.clone()),
                score: None,
                explored: false,
                reason: "fake".into(),
                alternatives: Vec::new(),
                applied: self.applied && !explicit,
                mode: request.settings.mode,
                stage: if self.applied {
                    LearningStage::Auto
                } else {
                    LearningStage::Shadow
                },
                session_id: request.session_id,
                task_id: request.task_id,
                run_id: request.run_id,
                turn_index: None,
                outcome: None,
                used: None,
            };
            self.store.put_decision(&decision).await?;
            Ok(decision)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_chat_pool_is_the_managers_pool_and_empty_when_no_router_is_wired() {
        let state = crate::test_helpers::mock_app_state();
        let manager = Arc::new(crate::chat::ChatManager::new_without_memory(
            state.neo4j,
            state.meili,
            crate::chat::config::ChatConfig::default(),
        ));
        let pool = ChatRoutingPool {
            manager: Arc::clone(&manager),
        };
        // The runner asks the pool exactly what the chat does: same answer, whatever
        // the project, and an empty one (not a panic) without a cognitive router.
        for project in [None, Some("nexus")] {
            assert_eq!(
                pool.facts(project).await.len(),
                manager.routing_pool_for(project).await.len()
            );
            assert!(pool.facts(project).await.is_empty());
        }
    }

    /// B-R3: the manager's pool takes Claude Code's models from the live Anthropic
    /// catalog, each with the window the Models API reports (`max_input_tokens`).
    mod claude_code_window {
        use super::*;
        use crate::chat::model_catalog::{refreshed_cache_for_test, ModelCatalogCache};
        use crate::chat::provider::cognitive::candidates::{
            self, ModelFacts, RejectReason, Slot, UnknownWindow,
        };
        use crate::chat::provider::cognitive::decider::CognitiveRouting;
        use crate::chat::provider::cognitive::signature::{TaskClass, TaskSignature};
        use crate::chat::provider::cognitive::store::InMemoryRoutingStore;
        use crate::chat::provider::resolver::CLAUDE_CODE;

        /// The Claude Code part of the pool of a manager routed over `catalog`,
        /// Claude Code measured healthy (no CLI is needed for the pool itself).
        async fn claude_code_pool(catalog: Arc<ModelCatalogCache>) -> Vec<ModelFacts> {
            let state = crate::test_helpers::mock_app_state();
            let routing = CognitiveRouting::new(Arc::new(InMemoryRoutingStore::new()))
                .with_claude_code_catalog(catalog);
            routing
                .health
                .put(CLAUDE_CODE, true, std::time::Instant::now());
            let manager = crate::chat::ChatManager::new_without_memory(
                state.neo4j,
                state.meili,
                crate::chat::config::ChatConfig::default(),
            )
            .with_cognitive_routing(routing);
            manager
                .routing_pool_for(None)
                .await
                .into_iter()
                .filter(|facts| facts.provider_id == CLAUDE_CODE)
                .collect()
        }

        #[tokio::test]
        async fn claude_code_enters_the_pool_with_the_live_catalogs_window_and_is_eligible() {
            use wiremock::matchers::{method, path};
            let server = wiremock::MockServer::start().await;
            wiremock::Mock::given(method("GET"))
                .and(path("/v1/models"))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(
                    serde_json::json!({
                        "data": [{"id": "claude-opus-5-5", "display_name": "Claude Opus 5.5",
                                  "max_input_tokens": 1_000_000, "max_tokens": 128_000}],
                        "has_more": false,
                        "last_id": "claude-opus-5-5",
                    }),
                ))
                .mount(&server)
                .await;
            let catalog =
                refreshed_cache_for_test(format!("{}/v1/models", server.uri()), "k").await;

            let pool = claude_code_pool(catalog).await;
            let opus = pool
                .iter()
                .find(|f| f.model == "claude-opus-5-5")
                .unwrap_or_else(|| panic!("no claude-code opus in the pool: {pool:?}"));
            assert_eq!(opus.context_window, Some(1_000_000));
            assert!(opus.supports_tools && opus.supports_images);

            let signature = TaskSignature::utility(TaskClass::UtilityFeatureGraph, 300_000, None);
            let out = candidates::apply(Slot::Automatic, &signature, &pool).unwrap();
            assert!(
                out.eligible.iter().any(|f| f.model == "claude-opus-5-5"),
                "rejected: {:?}",
                out.rejected
            );
        }

        #[tokio::test]
        async fn offline_every_claude_code_model_is_listed_and_rejected_by_name() {
            let pool = claude_code_pool(ModelCatalogCache::new(None)).await;
            assert!(!pool.is_empty());
            let signature = TaskSignature::utility(TaskClass::UtilityFeatureGraph, 10_000, None);
            let out = candidates::apply(Slot::Automatic, &signature, &pool).unwrap();
            assert!(out.eligible.is_empty());
            assert!(out.rejected.iter().all(|r| r.reason
                == RejectReason::WindowUnknown {
                    need: 10_000,
                    why: UnknownWindow::CatalogOffline
                }));
        }
    }

    #[test]
    fn verification_shapes() {
        assert_eq!(verification_passed(None), None);
        assert_eq!(verification_passed(Some("nope")), None);
        assert_eq!(
            verification_passed(Some(r#"{"passed":false}"#)),
            Some(false)
        );
        assert_eq!(
            verification_passed(Some(r#"{"build":"pass","tests":"pass"}"#)),
            Some(true)
        );
        assert_eq!(
            verification_passed(Some(r#"{"build":"pass","tests":"fail"}"#)),
            Some(false)
        );
        assert_eq!(verification_passed(Some(r#"{"build":"skipped"}"#)), None);
    }

    #[test]
    fn unknown_or_non_marginal_cost_is_none() {
        let mut node = AgentExecutionNode {
            status: AgentExecutionStatus::Completed,
            cost_usd: 0.4,
            duration_secs: 1.5,
            attempt: 2,
            ..Default::default()
        };
        assert_eq!(
            outcome_of_attempt(&node).cost_usd,
            None,
            "no basis = unknown"
        );
        node.cost_basis = Some("subscription".into());
        assert_eq!(outcome_of_attempt(&node).cost_usd, None);
        node.cost_basis = Some("reported".into());
        let outcome = outcome_of_attempt(&node);
        assert_eq!(outcome.cost_usd, Some(0.4));
        assert_eq!(outcome.duration_ms, Some(1500));
        assert_eq!(outcome.attempts, 2);
        assert_eq!(outcome.success, Some(true));
    }

    #[tokio::test]
    async fn a_persona_preference_is_read_by_id_name_or_the_runners_name_form() {
        let graph = crate::neo4j::mock::MockGraphStore::new();
        let named = crate::test_helpers::test_persona("rust-expert", Some("  opus "));
        let silent = crate::test_helpers::test_persona("generalist", None);
        let blank = crate::test_helpers::test_persona("blank", Some("   "));
        for p in [&named, &silent, &blank] {
            graph.create_persona(p).await.unwrap();
        }
        let pref = |key: String| {
            let graph = &graph;
            async move { persona_preference(graph, None, &key).await }
        };
        assert_eq!(pref("rust-expert".into()).await.as_deref(), Some("opus"));
        assert_eq!(pref(named.id.to_string()).await.as_deref(), Some("opus"));
        assert_eq!(
            pref(format!("\"{}\"", named.id)).await.as_deref(),
            Some("opus")
        );
        assert_eq!(
            pref("rust-expert:complex".into()).await.as_deref(),
            Some("opus")
        );
        // No preference, a blank one, an unknown persona, an empty key: nothing.
        assert_eq!(pref("generalist".into()).await, None);
        assert_eq!(pref("blank".into()).await, None);
        assert_eq!(pref("ghost".into()).await, None);
        assert_eq!(pref("  ".into()).await, None);
    }

    #[tokio::test]
    async fn a_persona_of_another_project_is_not_read() {
        let graph = crate::neo4j::mock::MockGraphStore::new();
        let mut other = crate::test_helpers::test_persona("rust-expert", Some("opus"));
        other.project_id = Some(Uuid::new_v4());
        graph.create_persona(&other).await.unwrap();
        assert_eq!(
            persona_preference(&graph, Some(Uuid::new_v4()), "rust-expert").await,
            None
        );
        assert_eq!(
            persona_preference(&graph, other.project_id, "rust-expert")
                .await
                .as_deref(),
            Some("opus")
        );
    }
}
