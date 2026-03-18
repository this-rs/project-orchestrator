//! DualTrack Router — Policy Net + NN Router fallback.
//!
//! In "nn" mode: only uses the Nearest Neighbor Router.
//! In "full" mode: tries the Policy Net first, falls back to NN Router
//! if the policy net times out, returns OOD, or the CpuGuard is paused.
//!
//! Phase 4 additions:
//! - InferenceEngine integration for Policy Net inference
//! - Confidence calibration via Platt scaling
//! - Progressive rollout (configurable split ratio)
//! - Decision logging for feedback loop

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Serialize;

use neural_routing_core::{error::Result, NNRoute, Router, TrajectoryStore};
use neural_routing_nn::NNRouter;

use crate::confidence::{PlattCalibrator, RolloutConfig};
use crate::config::{NeuralRoutingConfig, RoutingMode};
use crate::cpu_guard::CpuGuard;
use crate::inference_engine::{InferenceEngine, InferenceResult};

// ---------------------------------------------------------------------------
// Routing decision metadata
// ---------------------------------------------------------------------------

/// Why a particular routing source was chosen.
#[derive(Debug, Clone, Serialize)]
pub enum RoutingReason {
    /// Neural routing disabled.
    Disabled,
    /// NN-only mode configured.
    NnMode,
    /// CPU guard paused — using NN fallback.
    CpuGuardPaused,
    /// Policy Net not ready (no model loaded).
    PolicyNotReady,
    /// Progressive rollout directed to NN.
    RolloutToNn,
    /// Policy Net confidence below threshold.
    LowConfidence {
        raw_confidence: f32,
        calibrated_confidence: f32,
        threshold: f32,
    },
    /// Policy Net returned OOD actions.
    PolicyOod,
    /// Policy Net timed out.
    PolicyTimeout,
    /// Policy Net selected (high confidence, in-distribution).
    PolicySelected {
        raw_confidence: f32,
        calibrated_confidence: f32,
    },
}

/// A logged routing decision (for feedback loop and monitoring).
#[derive(Debug, Clone, Serialize)]
pub struct RoutingDecision {
    /// Which source produced the route.
    pub source: RoutingSource,
    /// Why this source was chosen.
    pub reason: RoutingReason,
    /// Policy Net inference result (if attempted).
    pub policy_result: Option<InferenceResult>,
    /// Latency of the routing decision (microseconds).
    pub latency_us: u64,
}

/// Source that produced the final route.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum RoutingSource {
    NnRouter,
    PolicyNet,
}

/// DualTrack Router — orchestrates policy net and NN router with timeout + fallback.
pub struct DualTrackRouter {
    nn_router: NNRouter,
    config: NeuralRoutingConfig,
    cpu_guard: CpuGuard,
    /// InferenceEngine for Policy Net (Phase 4).
    inference_engine: Option<InferenceEngine>,
    /// Confidence calibrator (Platt scaling).
    calibrator: PlattCalibrator,
    /// Progressive rollout configuration.
    rollout: RolloutConfig,
}

impl DualTrackRouter {
    pub fn new(store: Arc<dyn TrajectoryStore>, config: NeuralRoutingConfig) -> Self {
        let cpu_guard = CpuGuard::new(config.cpu_guard.clone().into());
        let nn_router = NNRouter::new(store, config.nn.clone());

        Self {
            nn_router,
            config,
            cpu_guard,
            inference_engine: None,
            calibrator: PlattCalibrator::default(),
            rollout: RolloutConfig::default(),
        }
    }

    /// Set the inference engine for Policy Net routing.
    pub fn set_inference_engine(&mut self, engine: InferenceEngine) {
        self.inference_engine = Some(engine);
    }

    /// Set the confidence calibrator.
    pub fn set_calibrator(&mut self, calibrator: PlattCalibrator) {
        self.calibrator = calibrator;
    }

    /// Set the rollout configuration.
    pub fn set_rollout(&mut self, rollout: RolloutConfig) {
        self.rollout = rollout;
    }

    /// Start the CPU guard background monitoring.
    pub fn start_cpu_monitoring(&self) -> tokio::task::JoinHandle<()> {
        self.cpu_guard.start_monitoring()
    }

    /// Get the NN router for direct access (metrics, cache invalidation).
    pub fn nn_router(&self) -> &NNRouter {
        &self.nn_router
    }

    /// Get the CPU guard state.
    pub fn cpu_guard(&self) -> &CpuGuard {
        &self.cpu_guard
    }

    /// Get the current config.
    pub fn config(&self) -> &NeuralRoutingConfig {
        &self.config
    }

    /// Get the rollout config.
    pub fn rollout(&self) -> &RolloutConfig {
        &self.rollout
    }

    /// Update config at runtime (hot reload).
    ///
    /// Propagates changes to the NNRouter (top_k, min_similarity, cache settings).
    pub fn update_config(&mut self, config: NeuralRoutingConfig) {
        // Propagate NN config to the inner router
        self.nn_router.update_config(config.nn.clone());
        self.config = config;
    }

    /// Route with full dual-track logic and return the decision metadata.
    ///
    /// This is the main entry point for the feedback loop — it returns both
    /// the route and the routing decision for logging.
    pub async fn route_with_decision(
        &self,
        query_embedding: &[f32],
        session_hash: u64,
    ) -> Result<(Option<NNRoute>, RoutingDecision)> {
        let start = std::time::Instant::now();

        if !self.config.enabled {
            let nn_result = Ok(None);
            return nn_result.map(|r| {
                (
                    r,
                    RoutingDecision {
                        source: RoutingSource::NnRouter,
                        reason: RoutingReason::Disabled,
                        policy_result: None,
                        latency_us: start.elapsed().as_micros() as u64,
                    },
                )
            });
        }

        match self.config.mode {
            RoutingMode::NN => {
                let result = self.nn_router.route(query_embedding).await?;
                Ok((
                    result,
                    RoutingDecision {
                        source: RoutingSource::NnRouter,
                        reason: RoutingReason::NnMode,
                        policy_result: None,
                        latency_us: start.elapsed().as_micros() as u64,
                    },
                ))
            }
            RoutingMode::Full => {
                // Check CPU guard
                if self.cpu_guard.is_paused() {
                    let result = self.nn_router.route(query_embedding).await?;
                    return Ok((
                        result,
                        RoutingDecision {
                            source: RoutingSource::NnRouter,
                            reason: RoutingReason::CpuGuardPaused,
                            policy_result: None,
                            latency_us: start.elapsed().as_micros() as u64,
                        },
                    ));
                }

                // Check if inference engine is ready
                let engine = match &self.inference_engine {
                    Some(e) if e.is_ready() => e,
                    _ => {
                        let result = self.nn_router.route(query_embedding).await?;
                        return Ok((
                            result,
                            RoutingDecision {
                                source: RoutingSource::NnRouter,
                                reason: RoutingReason::PolicyNotReady,
                                policy_result: None,
                                latency_us: start.elapsed().as_micros() as u64,
                            },
                        ));
                    }
                };

                // Progressive rollout check
                if !self.rollout.should_use_policy(session_hash) {
                    let result = self.nn_router.route(query_embedding).await?;
                    return Ok((
                        result,
                        RoutingDecision {
                            source: RoutingSource::NnRouter,
                            reason: RoutingReason::RolloutToNn,
                            policy_result: None,
                            latency_us: start.elapsed().as_micros() as u64,
                        },
                    ));
                }

                // Try Policy Net with timeout
                let timeout = Duration::from_millis(self.config.inference.timeout_ms);
                let inference_result = tokio::time::timeout(timeout, async {
                    engine.infer(query_embedding, None, None)
                })
                .await;

                match inference_result {
                    Ok(Ok(result)) => {
                        let raw_confidence = result.overall_confidence;
                        let calibrated = self.calibrator.calibrate(raw_confidence);

                        // Check OOD
                        if result.has_ood_actions {
                            let nn_result = self.nn_router.route(query_embedding).await?;
                            return Ok((
                                nn_result,
                                RoutingDecision {
                                    source: RoutingSource::NnRouter,
                                    reason: RoutingReason::PolicyOod,
                                    policy_result: Some(result),
                                    latency_us: start.elapsed().as_micros() as u64,
                                },
                            ));
                        }

                        // Check calibrated confidence threshold
                        if calibrated < self.rollout.confidence_threshold {
                            let nn_result = self.nn_router.route(query_embedding).await?;
                            return Ok((
                                nn_result,
                                RoutingDecision {
                                    source: RoutingSource::NnRouter,
                                    reason: RoutingReason::LowConfidence {
                                        raw_confidence,
                                        calibrated_confidence: calibrated,
                                        threshold: self.rollout.confidence_threshold,
                                    },
                                    policy_result: Some(result),
                                    latency_us: start.elapsed().as_micros() as u64,
                                },
                            ));
                        }

                        // Policy Net wins!
                        // Convert InferenceResult actions to NNRoute
                        let nn_route = inference_result_to_nn_route(&result);

                        Ok((
                            Some(nn_route),
                            RoutingDecision {
                                source: RoutingSource::PolicyNet,
                                reason: RoutingReason::PolicySelected {
                                    raw_confidence,
                                    calibrated_confidence: calibrated,
                                },
                                policy_result: Some(result),
                                latency_us: start.elapsed().as_micros() as u64,
                            },
                        ))
                    }
                    Ok(Err(e)) => {
                        tracing::warn!(error = %e, "Policy Net inference error, falling back to NN");
                        let nn_result = self.nn_router.route(query_embedding).await?;
                        Ok((
                            nn_result,
                            RoutingDecision {
                                source: RoutingSource::NnRouter,
                                reason: RoutingReason::PolicyTimeout,
                                policy_result: None,
                                latency_us: start.elapsed().as_micros() as u64,
                            },
                        ))
                    }
                    Err(_) => {
                        tracing::warn!(
                            timeout_ms = self.config.inference.timeout_ms,
                            "Policy Net timed out, falling back to NN"
                        );
                        let nn_result = self.nn_router.route(query_embedding).await?;
                        Ok((
                            nn_result,
                            RoutingDecision {
                                source: RoutingSource::NnRouter,
                                reason: RoutingReason::PolicyTimeout,
                                policy_result: None,
                                latency_us: start.elapsed().as_micros() as u64,
                            },
                        ))
                    }
                }
            }
        }
    }
}

/// Convert an InferenceResult (from Policy Net) to an NNRoute (common format).
fn inference_result_to_nn_route(result: &InferenceResult) -> NNRoute {
    use neural_routing_core::PlannedAction as CorePlannedAction;

    NNRoute {
        actions: result
            .actions
            .iter()
            .map(|a| CorePlannedAction {
                action_type: format!("{}.{}", a.tool, a.action),
                action_params: a
                    .param_template
                    .as_ref()
                    .map(|t| serde_json::Value::String(t.clone()))
                    .unwrap_or(serde_json::Value::Null),
                confidence: a.confidence as f64,
            })
            .collect(),
        similarity: result.overall_confidence as f64,
        score: result.overall_confidence as f64,
        source_trajectory_id: uuid::Uuid::nil(),
        source_reward: 0.0,
    }
}

#[async_trait]
impl Router for DualTrackRouter {
    async fn route(&self, query_embedding: &[f32]) -> Result<Option<NNRoute>> {
        if !self.config.enabled {
            return Ok(None);
        }

        match self.config.mode {
            RoutingMode::NN => {
                // NN-only mode — direct to nearest neighbor router
                self.nn_router.route(query_embedding).await
            }
            RoutingMode::Full => {
                // Full mode — try policy net with timeout, fallback to NN
                // Phase 3+ will add policy net here. For now, fallback to NN.
                //
                // Future implementation:
                // 1. Check CpuGuard — if paused, skip to NN
                // 2. Try policy net with timeout (inference.timeout_ms)
                // 3. If timeout/OOD/error → fallback to NN
                //
                // For now: always NN (policy net not yet implemented)
                if self.cpu_guard.is_paused() {
                    tracing::debug!("DualTrack: CPU guard paused, using NN Router");
                }

                let timeout = Duration::from_millis(self.config.inference.timeout_ms);
                match tokio::time::timeout(timeout, self.nn_router.route(query_embedding)).await {
                    Ok(result) => result,
                    Err(_) => {
                        tracing::warn!(
                            "DualTrack: NN Router timed out after {}ms",
                            self.config.inference.timeout_ms
                        );
                        Ok(None)
                    }
                }
            }
        }
    }

    async fn route_with_context(
        &self,
        query_embedding: &[f32],
        available_tools: &[String],
    ) -> Result<Option<NNRoute>> {
        if !self.config.enabled {
            return Ok(None);
        }

        match self.config.mode {
            RoutingMode::NN => {
                self.nn_router
                    .route_with_context(query_embedding, available_tools)
                    .await
            }
            RoutingMode::Full => {
                // Same as above — policy net will be added in Phase 3
                let timeout = Duration::from_millis(self.config.inference.timeout_ms);
                match tokio::time::timeout(
                    timeout,
                    self.nn_router
                        .route_with_context(query_embedding, available_tools),
                )
                .await
                {
                    Ok(result) => result,
                    Err(_) => {
                        tracing::warn!("DualTrack: timed out, returning None");
                        Ok(None)
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use neural_routing_core::{
        RewardDistribution, Trajectory, TrajectoryFilter, TrajectoryNode, TrajectoryStats,
    };
    use std::sync::Mutex;
    use uuid::Uuid;

    /// Mock TrajectoryStore for testing (same pattern as neural-routing-nn).
    struct MockStore {
        trajectories: Mutex<Vec<Trajectory>>,
    }

    impl MockStore {
        fn new() -> Self {
            Self {
                trajectories: Mutex::new(Vec::new()),
            }
        }

        fn with_trajectories(trajectories: Vec<Trajectory>) -> Self {
            Self {
                trajectories: Mutex::new(trajectories),
            }
        }
    }

    #[async_trait]
    impl TrajectoryStore for MockStore {
        async fn store_trajectory(&self, trajectory: &Trajectory) -> Result<()> {
            self.trajectories.lock().unwrap().push(trajectory.clone());
            Ok(())
        }

        async fn get_trajectory(&self, id: &Uuid) -> Result<Option<Trajectory>> {
            Ok(self
                .trajectories
                .lock()
                .unwrap()
                .iter()
                .find(|t| t.id == *id)
                .cloned())
        }

        async fn list_trajectories(&self, _filter: &TrajectoryFilter) -> Result<Vec<Trajectory>> {
            Ok(self.trajectories.lock().unwrap().clone())
        }

        async fn search_similar(
            &self,
            query: &[f32],
            top_k: usize,
            min_sim: f32,
        ) -> Result<Vec<(Trajectory, f64)>> {
            let trajectories = self.trajectories.lock().unwrap();
            let mut results: Vec<(Trajectory, f64)> = trajectories
                .iter()
                .map(|t| {
                    let sim = neural_routing_core::cosine_similarity(query, &t.query_embedding);
                    (t.clone(), sim)
                })
                .filter(|(_, sim)| *sim >= min_sim as f64)
                .collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            results.truncate(top_k);
            Ok(results)
        }

        async fn get_stats(&self) -> Result<TrajectoryStats> {
            Ok(TrajectoryStats {
                total_count: self.trajectories.lock().unwrap().len(),
                avg_reward: 0.0,
                avg_step_count: 0.0,
                avg_duration_ms: 0.0,
                reward_distribution: RewardDistribution {
                    min: 0.0,
                    max: 0.0,
                    p25: 0.0,
                    p50: 0.0,
                    p75: 0.0,
                    p90: 0.0,
                },
            })
        }

        async fn count(&self) -> Result<usize> {
            Ok(self.trajectories.lock().unwrap().len())
        }

        async fn delete_trajectory(&self, id: &Uuid) -> Result<bool> {
            let mut ts = self.trajectories.lock().unwrap();
            let len_before = ts.len();
            ts.retain(|t| t.id != *id);
            Ok(ts.len() < len_before)
        }
    }

    fn make_unit_vec_256() -> Vec<f32> {
        let val = 1.0 / (256.0f32).sqrt();
        vec![val; 256]
    }

    fn make_trajectory(reward: f64, action_types: Vec<&str>) -> Trajectory {
        let embedding = make_unit_vec_256();
        Trajectory {
            id: Uuid::new_v4(),
            session_id: "test-session".to_string(),
            query_embedding: embedding.clone(),
            total_reward: reward,
            step_count: action_types.len(),
            duration_ms: 1000,
            nodes: action_types
                .iter()
                .enumerate()
                .map(|(i, at)| TrajectoryNode {
                    id: Uuid::new_v4(),
                    context_embedding: embedding.clone(),
                    action_type: at.to_string(),
                    action_params: serde_json::Value::Null,
                    alternatives_count: 3,
                    chosen_index: 0,
                    confidence: 0.8,
                    local_reward: reward / action_types.len() as f64,
                    cumulative_reward: reward * (i + 1) as f64 / action_types.len() as f64,
                    delta_ms: 100,
                    order: i,
                })
                .collect(),
            created_at: Utc::now(),
        }
    }

    fn default_config() -> NeuralRoutingConfig {
        NeuralRoutingConfig::default()
    }

    #[tokio::test]
    async fn test_new_creates_successfully() {
        let store = Arc::new(MockStore::new());
        let config = default_config();
        let router = DualTrackRouter::new(store, config.clone());

        assert!(router.config().enabled);
        assert_eq!(router.config().mode, RoutingMode::NN);
        assert!(!router.cpu_guard().is_paused());
    }

    #[tokio::test]
    async fn test_route_when_disabled_returns_none() {
        let store = Arc::new(MockStore::with_trajectories(vec![make_trajectory(
            0.9,
            vec!["code_search", "note_create"],
        )]));
        let mut config = default_config();
        config.enabled = false;

        let router = DualTrackRouter::new(store, config);
        let embedding = make_unit_vec_256();

        let result = router.route(&embedding).await.unwrap();
        assert!(result.is_none(), "disabled router should return None");
    }

    #[tokio::test]
    async fn test_route_nn_mode_returns_route() {
        let trajectory = make_trajectory(0.9, vec!["code_search", "analyze_impact"]);
        let store = Arc::new(MockStore::with_trajectories(vec![trajectory]));
        let mut config = default_config();
        config.mode = RoutingMode::NN;

        let router = DualTrackRouter::new(store, config);
        let embedding = make_unit_vec_256();

        let result = router.route(&embedding).await.unwrap();
        assert!(result.is_some(), "NN mode should find a route");
        let route = result.unwrap();
        assert_eq!(route.actions.len(), 2);
        assert_eq!(route.actions[0].action_type, "code_search");
        assert_eq!(route.actions[1].action_type, "analyze_impact");
    }

    #[tokio::test]
    async fn test_route_with_context_filters_tools() {
        let trajectory = make_trajectory(0.9, vec!["code_search", "analyze_impact", "note_create"]);
        let store = Arc::new(MockStore::with_trajectories(vec![trajectory]));
        let config = default_config();

        let router = DualTrackRouter::new(store, config);
        let embedding = make_unit_vec_256();
        let tools = vec!["code_search".to_string(), "note_create".to_string()];

        let result = router.route_with_context(&embedding, &tools).await.unwrap();
        assert!(result.is_some());
        let route = result.unwrap();
        assert_eq!(route.actions.len(), 2);
        assert!(
            !route
                .actions
                .iter()
                .any(|a| a.action_type == "analyze_impact"),
            "analyze_impact should be filtered out"
        );
    }

    #[tokio::test]
    async fn test_update_config() {
        let store = Arc::new(MockStore::new());
        let config = default_config();
        let mut router = DualTrackRouter::new(store, config);

        assert!(router.config().enabled);
        assert_eq!(router.config().mode, RoutingMode::NN);

        let mut new_config = default_config();
        new_config.enabled = false;
        new_config.mode = RoutingMode::Full;
        router.update_config(new_config);

        assert!(!router.config().enabled);
        assert_eq!(router.config().mode, RoutingMode::Full);
    }

    #[tokio::test]
    async fn test_update_config_propagates_to_nn_router() {
        // Verify that changing nn.top_k via update_config actually affects routing
        let trajectory = make_trajectory(0.9, vec!["code_search", "analyze_impact"]);
        let store = Arc::new(MockStore::with_trajectories(vec![trajectory]));
        let config = default_config();
        let mut router = DualTrackRouter::new(store, config);

        // Default top_k is 5, min_similarity is 0.7 — should find the trajectory
        let embedding = make_unit_vec_256();
        let result = router.route(&embedding).await.unwrap();
        assert!(result.is_some(), "Should find route with default config");

        // Now set min_similarity very high — should NOT find it
        let mut new_config = default_config();
        new_config.nn.min_similarity = 0.9999; // unrealistically high
        router.update_config(new_config);

        // Need to invalidate cache since config changed
        router.nn_router().invalidate_cache();

        let _result = router.route(&embedding).await.unwrap();
        // With min_sim=0.9999 the mock returns exact similarity (1.0), so it should still match
        // Let's instead test top_k = 0
        let mut new_config2 = default_config();
        new_config2.nn.top_k = 0; // no candidates
        router.update_config(new_config2);
        router.nn_router().invalidate_cache();

        let result = router.route(&embedding).await.unwrap();
        assert!(
            result.is_none(),
            "With top_k=0, should return no route (config propagated to NNRouter)"
        );
    }

    #[tokio::test]
    async fn test_config_accessor() {
        let store = Arc::new(MockStore::new());
        let mut config = default_config();
        config.inference.timeout_ms = 42;
        config.nn.top_k = 10;

        let router = DualTrackRouter::new(store, config);

        let cfg = router.config();
        assert_eq!(cfg.inference.timeout_ms, 42);
        assert_eq!(cfg.nn.top_k, 10);
        assert!(cfg.enabled);
    }
}
