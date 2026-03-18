//! Core traits for trajectory storage and routing.

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::models::*;

/// Async trait for trajectory persistence and retrieval.
#[async_trait]
pub trait TrajectoryStore: Send + Sync {
    /// Store a complete trajectory with all its nodes.
    async fn store_trajectory(&self, trajectory: &Trajectory) -> Result<()>;

    /// Get a trajectory by ID, including all ordered nodes.
    async fn get_trajectory(&self, id: &Uuid) -> Result<Option<Trajectory>>;

    /// List trajectories with filtering and pagination.
    async fn list_trajectories(&self, filter: &TrajectoryFilter) -> Result<Vec<Trajectory>>;

    /// Find the K most similar trajectories by query embedding (cosine similarity).
    async fn search_similar(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_similarity: f32,
    ) -> Result<Vec<(Trajectory, f64)>>;

    /// Get statistics about stored trajectories.
    async fn get_stats(&self) -> Result<TrajectoryStats>;

    /// Count total trajectories.
    async fn count(&self) -> Result<usize>;

    /// Delete a trajectory and all its nodes.
    async fn delete_trajectory(&self, id: &Uuid) -> Result<bool>;

    /// Store multiple trajectories in a single batch operation.
    ///
    /// Default implementation falls back to sequential `store_trajectory` calls.
    /// `Neo4jTrajectoryStore` overrides this with UNWIND-based batch insert (≤3 queries).
    async fn store_trajectories_batch(&self, trajectories: &[Trajectory]) -> Result<usize> {
        let mut stored = 0;
        for trajectory in trajectories {
            self.store_trajectory(trajectory).await?;
            stored += 1;
        }
        Ok(stored)
    }

    /// Link a trajectory to a protocol run via a DURING_RUN relation.
    /// Default implementation is a no-op (for stores that don't support graph relations).
    async fn link_trajectory_to_run(
        &self,
        _trajectory_id: &Uuid,
        _run_id: &Uuid,
    ) -> Result<()> {
        Ok(())
    }
}

/// Trait for intercepting decisions during a session to build trajectories.
#[async_trait]
pub trait DecisionInterceptor: Send + Sync {
    /// Called when an action is about to be executed.
    /// Returns the context embedding and alternative candidates.
    async fn intercept_decision(
        &self,
        action_type: &str,
        action_params: &serde_json::Value,
        alternatives: &[ActionCandidate],
        chosen_index: usize,
    ) -> Result<Vec<f32>>;
}

/// Trait for routing — given a query, suggest a sequence of actions.
#[async_trait]
pub trait Router: Send + Sync {
    /// Route a query to a sequence of planned actions.
    async fn route(&self, query_embedding: &[f32]) -> Result<Option<NNRoute>>;

    /// Route with context filtering (only suggest actions using available tools).
    async fn route_with_context(
        &self,
        query_embedding: &[f32],
        available_tools: &[String],
    ) -> Result<Option<NNRoute>>;
}
