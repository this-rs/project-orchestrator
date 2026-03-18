//! The Nearest Neighbor Router — core routing logic.

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use moka::future::Cache;
use neural_routing_core::{error::Result, NNRoute, PlannedAction, Router, TrajectoryStore};
use serde::{Deserialize, Serialize};

use crate::metrics::NNMetrics;
use crate::scoring::compute_score;

/// Configuration for the Nearest Neighbor Router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNConfig {
    /// Number of nearest neighbors to retrieve (default: 5).
    pub top_k: usize,
    /// Minimum cosine similarity threshold (default: 0.7).
    pub min_similarity: f32,
    /// Maximum age in days for candidate trajectories (default: 30).
    pub max_route_age_days: u32,
    /// Cache capacity in entries (default: 500).
    pub cache_capacity: u64,
    /// Cache TTL in seconds (default: 3600 = 1h).
    pub cache_ttl_secs: u64,
}

impl Default for NNConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_similarity: 0.7,
            max_route_age_days: 30,
            cache_capacity: 500,
            cache_ttl_secs: 3600,
        }
    }
}

/// The Nearest Neighbor Router — zero ML, immediate, permanent fallback.
///
/// For each query:
/// 1. Check the moka cache (hash of query embedding)
/// 2. If miss, search the TrajectoryStore for top-K similar trajectories
/// 3. Score candidates: similarity * recency * reward
/// 4. Extract and return the best route
pub struct NNRouter {
    store: Arc<dyn TrajectoryStore>,
    cache: Cache<u64, Arc<NNRoute>>,
    config: NNConfig,
    metrics: NNMetrics,
}

impl NNRouter {
    pub fn new(store: Arc<dyn TrajectoryStore>, config: NNConfig) -> Self {
        let cache = Cache::builder()
            .max_capacity(config.cache_capacity)
            .time_to_live(Duration::from_secs(config.cache_ttl_secs))
            .build();

        Self {
            store,
            cache,
            config,
            metrics: NNMetrics::new(),
        }
    }

    /// Get a snapshot of current metrics.
    pub fn metrics(&self) -> &NNMetrics {
        &self.metrics
    }

    /// Update the NN config at runtime.
    ///
    /// Invalidates the cache if cache settings changed, and updates
    /// scoring parameters (top_k, min_similarity, max_route_age_days).
    pub fn update_config(&mut self, config: NNConfig) {
        let cache_changed = config.cache_capacity != self.config.cache_capacity
            || config.cache_ttl_secs != self.config.cache_ttl_secs;

        self.config = config;

        if cache_changed {
            // Rebuild cache with new settings
            self.cache = Cache::builder()
                .max_capacity(self.config.cache_capacity)
                .time_to_live(Duration::from_secs(self.config.cache_ttl_secs))
                .build();
        }
    }

    /// Invalidate the entire cache.
    pub fn invalidate_cache(&self) {
        self.cache.invalidate_all();
    }

    /// Compute a stable hash of an embedding for cache keying.
    fn embedding_hash(embedding: &[f32]) -> u64 {
        let mut hasher = std::hash::DefaultHasher::new();
        for &v in embedding {
            v.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Internal route logic.
    async fn route_internal(
        &self,
        query_embedding: &[f32],
        available_tools: Option<&[String]>,
    ) -> Result<Option<NNRoute>> {
        self.metrics.record_query();

        let cache_key = Self::embedding_hash(query_embedding);

        // Check cache
        if let Some(cached) = self.cache.get(&cache_key).await {
            self.metrics.record_cache_hit();
            let mut route = (*cached).clone();
            // Filter by available tools if specified
            if let Some(tools) = available_tools {
                route.actions.retain(|a| tools.contains(&a.action_type));
            }
            return Ok(Some(route));
        }

        // Search store
        let candidates = self
            .store
            .search_similar(
                query_embedding,
                self.config.top_k,
                self.config.min_similarity,
            )
            .await?;

        if candidates.is_empty() {
            return Ok(None);
        }

        // Find max reward for normalization
        let max_reward = candidates
            .iter()
            .map(|(t, _)| t.total_reward)
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1e-10);

        // Score all candidates
        let mut best_score = f64::NEG_INFINITY;
        let mut best_trajectory = None;
        let mut best_similarity = 0.0;

        for (trajectory, similarity) in &candidates {
            let score = compute_score(
                *similarity,
                &trajectory.created_at,
                trajectory.total_reward,
                max_reward,
                self.config.max_route_age_days,
            );

            if score > best_score {
                best_score = score;
                best_trajectory = Some(trajectory);
                best_similarity = *similarity;
            }
        }

        let trajectory = match best_trajectory {
            Some(t) => t,
            None => return Ok(None),
        };

        // We need the full trajectory with nodes to extract the route
        let full_trajectory = self.store.get_trajectory(&trajectory.id).await?;
        let full_trajectory = match full_trajectory {
            Some(t) => t,
            None => return Ok(None),
        };

        // Extract planned actions from the trajectory nodes
        let mut actions: Vec<PlannedAction> = full_trajectory
            .nodes
            .iter()
            .map(|node| PlannedAction {
                action_type: node.action_type.clone(),
                action_params: node.action_params.clone(),
                confidence: node.confidence * best_similarity,
            })
            .collect();

        // Filter by available tools if specified
        if let Some(tools) = available_tools {
            actions.retain(|a| tools.contains(&a.action_type));
        }

        if actions.is_empty() {
            return Ok(None);
        }

        let route = NNRoute {
            actions,
            similarity: best_similarity,
            score: best_score,
            source_trajectory_id: full_trajectory.id,
            source_reward: full_trajectory.total_reward,
        };

        // Record metrics
        self.metrics
            .record_hit(best_similarity, full_trajectory.total_reward);

        // Cache the result (before tool filtering for broader reuse)
        self.cache.insert(cache_key, Arc::new(route.clone())).await;

        Ok(Some(route))
    }
}

#[async_trait]
impl Router for NNRouter {
    async fn route(&self, query_embedding: &[f32]) -> Result<Option<NNRoute>> {
        self.route_internal(query_embedding, None).await
    }

    async fn route_with_context(
        &self,
        query_embedding: &[f32],
        available_tools: &[String],
    ) -> Result<Option<NNRoute>> {
        self.route_internal(query_embedding, Some(available_tools))
            .await
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

    /// Mock TrajectoryStore for testing.
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

    #[tokio::test]
    async fn test_route_returns_none_when_empty() {
        let store = Arc::new(MockStore::new());
        let router = NNRouter::new(store, NNConfig::default());
        let embedding = make_unit_vec_256();

        let result = router.route(&embedding).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_route_finds_similar_trajectory() {
        let trajectory = make_trajectory(0.9, vec!["code_search", "analyze_impact", "note_create"]);
        let store = Arc::new(MockStore::with_trajectories(vec![trajectory]));
        let router = NNRouter::new(store, NNConfig::default());
        let embedding = make_unit_vec_256();

        let result = router.route(&embedding).await.unwrap();
        assert!(result.is_some());
        let route = result.unwrap();
        assert_eq!(route.actions.len(), 3);
        assert_eq!(route.actions[0].action_type, "code_search");
        assert!(route.similarity > 0.99); // nearly identical embedding
    }

    #[tokio::test]
    async fn test_route_with_context_filters_tools() {
        let trajectory = make_trajectory(0.9, vec!["code_search", "analyze_impact", "note_create"]);
        let store = Arc::new(MockStore::with_trajectories(vec![trajectory]));
        let router = NNRouter::new(store, NNConfig::default());
        let embedding = make_unit_vec_256();
        let tools = vec!["code_search".to_string(), "note_create".to_string()];

        let result = router.route_with_context(&embedding, &tools).await.unwrap();
        assert!(result.is_some());
        let route = result.unwrap();
        assert_eq!(route.actions.len(), 2);
        assert!(!route
            .actions
            .iter()
            .any(|a| a.action_type == "analyze_impact"));
    }

    #[tokio::test]
    async fn test_cache_hit() {
        let trajectory = make_trajectory(0.9, vec!["code_search"]);
        let store = Arc::new(MockStore::with_trajectories(vec![trajectory]));
        let router = NNRouter::new(store, NNConfig::default());
        let embedding = make_unit_vec_256();

        // First call — cache miss
        let _ = router.route(&embedding).await.unwrap();
        // Second call — should be cache hit
        let _ = router.route(&embedding).await.unwrap();

        let snap = router.metrics().snapshot();
        assert_eq!(snap.total_queries, 2);
        assert_eq!(snap.cache_hits, 1);
    }

    #[tokio::test]
    async fn test_metrics_accumulate() {
        let trajectory = make_trajectory(0.8, vec!["code_search"]);
        let store = Arc::new(MockStore::with_trajectories(vec![trajectory]));
        let config = NNConfig {
            cache_capacity: 0, // disable cache for this test
            ..Default::default()
        };
        let router = NNRouter::new(store, config);
        let embedding = make_unit_vec_256();

        for _ in 0..5 {
            let _ = router.route(&embedding).await.unwrap();
        }

        let snap = router.metrics().snapshot();
        assert_eq!(snap.total_queries, 5);
        assert_eq!(snap.hits, 5);
        assert!(snap.avg_similarity > 0.99);
    }
}
