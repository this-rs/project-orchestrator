//! Runner Lifecycle Protocol Routing
//!
//! Routes a plan run to the most relevant lifecycle protocol using contextual
//! affinity scoring. The 3 plan-runner lifecycle protocols (full, light, reviewed)
//! are matched against the plan's context vector to find the best fit.
//!
//! If no lifecycle protocol exists (project without seeds), returns None
//! for graceful fallback to the existing runner behavior.

use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;
use crate::protocol::engine;
use crate::protocol::routing::{rank_protocols, ContextVector, DimensionWeights};

/// Result of lifecycle protocol routing.
#[derive(Debug, Clone)]
pub struct LifecycleRouteResult {
    /// The protocol ID that was selected
    pub protocol_id: Uuid,
    /// The protocol name (e.g., "plan-runner-full")
    pub protocol_name: String,
    /// The affinity score that led to this selection
    pub affinity_score: f64,
    /// The ProtocolRun that was created
    pub run_id: Uuid,
}

/// Prefix used by the 3 plan-runner lifecycle protocols.
const RUNNER_PROTOCOL_PREFIX: &str = "plan-runner-";

/// Minimum affinity score to accept a lifecycle protocol match.
/// Below this threshold, we fall back to the existing runner behavior.
const MIN_AFFINITY_THRESHOLD: f64 = 0.3;

/// Route a plan run to the best-matching lifecycle protocol.
///
/// 1. Loads the plan to extract project_id and context metrics
/// 2. Lists all protocols for the project with "plan-runner-" prefix
/// 3. Builds a ContextVector from the plan's metrics
/// 4. Ranks protocols by affinity and picks the best match
/// 5. Creates a ProtocolRun linked to the plan and fires `run_started`
///
/// Returns `None` if:
/// - The plan has no project_id
/// - No lifecycle protocols exist for the project
/// - No protocol meets the minimum affinity threshold
///
/// Performance: This is a lightweight vector computation (< 1ms),
/// plus 2-3 Neo4j queries (plan, list_protocols, create_run).
pub async fn route_lifecycle_protocol(
    graph: &Arc<dyn GraphStore>,
    plan_id: Uuid,
    total_tasks: usize,
) -> Result<Option<LifecycleRouteResult>> {
    // 1. Load the plan to get project_id
    let plan = match graph.get_plan(plan_id).await? {
        Some(p) => p,
        None => {
            warn!("Plan {} not found for lifecycle routing", plan_id);
            return Ok(None);
        }
    };

    let project_id = match plan.project_id {
        Some(pid) => pid,
        None => {
            debug!(
                "Plan {} has no project_id, skipping lifecycle routing",
                plan_id
            );
            return Ok(None);
        }
    };

    // 2. List lifecycle protocols for this project (plan-runner-*)
    let (all_protocols, _) = graph.list_protocols(project_id, None, 100, 0).await?;
    let runner_protocols: Vec<_> = all_protocols
        .into_iter()
        .filter(|p| p.name.starts_with(RUNNER_PROTOCOL_PREFIX))
        .collect();

    if runner_protocols.is_empty() {
        debug!(
            "No lifecycle protocols found for project {}, fallback to default runner",
            project_id
        );
        return Ok(None);
    }

    // 3. Build context vector from plan metrics.
    // We use total_tasks as a proxy for structure complexity.
    // Dependency and file counts are approximated from task count
    // to avoid extra DB queries — the routing is a lightweight heuristic.
    let context = ContextVector::from_plan_context(
        "execution", // We're at execution phase — the runner is starting
        total_tasks,
        0,   // dependency count not cheaply available here
        0,   // affected files count not cheaply available here
        0.0, // Just starting — 0% completion
    );

    // 4. Rank protocols by affinity
    let weights = DimensionWeights::default();
    let route_response = rank_protocols(&context, &runner_protocols, &weights);

    let best = match route_response.results.first() {
        Some(r) if r.affinity.score >= MIN_AFFINITY_THRESHOLD => r,
        Some(r) => {
            debug!(
                "Best lifecycle protocol {} has affinity {:.2} < threshold {:.2}, skipping",
                r.protocol_name, r.affinity.score, MIN_AFFINITY_THRESHOLD
            );
            return Ok(None);
        }
        None => return Ok(None),
    };

    info!(
        "Lifecycle protocol routed: {} (affinity: {:.2}) for plan {}",
        best.protocol_name, best.affinity.score, plan_id
    );

    // 5. Create a ProtocolRun linked to the plan and fire run_started
    let run = engine::start_run(
        graph.as_ref(),
        best.protocol_id,
        Some(plan_id),
        None, // No specific task — this covers the whole plan
        Some("runner:auto"),
    )
    .await?;

    // Fire the run_started transition to move from init → executing
    match engine::fire_transition(graph.as_ref(), run.id, "run_started").await {
        Ok(result) => {
            debug!(
                "Lifecycle protocol {} transitioned to state: {}",
                best.protocol_name, result.current_state_name
            );
        }
        Err(e) => {
            warn!(
                "Failed to fire run_started on lifecycle protocol {}: {}. Run {} will remain at init state.",
                best.protocol_name, e, run.id
            );
            // Non-fatal — the run exists, it just didn't transition yet
        }
    }

    Ok(Some(LifecycleRouteResult {
        protocol_id: best.protocol_id,
        protocol_name: best.protocol_name.clone(),
        affinity_score: best.affinity.score,
        run_id: run.id,
    }))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::test_helpers::test_project;

    /// Helper: create a plan in the mock store with a project.
    async fn setup_plan_with_project(mock: &MockGraphStore) -> (Uuid, Uuid) {
        let project = test_project();
        mock.create_project(&project).await.unwrap();

        let plan_id = Uuid::new_v4();
        let plan = crate::neo4j::models::PlanNode {
            id: plan_id,
            title: "Test Plan".to_string(),
            description: "A test plan".to_string(),
            status: crate::neo4j::models::PlanStatus::Draft,
            created_at: chrono::Utc::now(),
            created_by: "test".to_string(),
            priority: 5,
            project_id: Some(project.id),
            execution_context: None,
            persona: None,
        };
        mock.create_plan(&plan).await.unwrap();
        (plan_id, project.id)
    }

    #[tokio::test]
    async fn test_route_no_project_id() {
        let mock = Arc::new(MockGraphStore::new());
        // Create a plan with no project_id
        let plan_id = Uuid::new_v4();
        let plan = crate::neo4j::models::PlanNode {
            id: plan_id,
            title: "Orphan Plan".to_string(),
            description: "No project".to_string(),
            status: crate::neo4j::models::PlanStatus::Draft,
            created_at: chrono::Utc::now(),
            created_by: "test".to_string(),
            priority: 5,
            project_id: None,
            execution_context: None,
            persona: None,
        };
        mock.create_plan(&plan).await.unwrap();

        let graph: Arc<dyn GraphStore> = mock;
        let result = route_lifecycle_protocol(&graph, plan_id, 5).await.unwrap();
        assert!(
            result.is_none(),
            "Should return None when plan has no project_id"
        );
    }

    #[tokio::test]
    async fn test_route_no_lifecycle_protocols() {
        let mock = Arc::new(MockGraphStore::new());
        let (plan_id, _project_id) = setup_plan_with_project(&mock).await;

        // No protocols seeded — should return None
        let graph: Arc<dyn GraphStore> = mock;
        let result = route_lifecycle_protocol(&graph, plan_id, 5).await.unwrap();
        assert!(
            result.is_none(),
            "Should return None when no lifecycle protocols exist"
        );
    }

    #[tokio::test]
    async fn test_route_plan_not_found() {
        let mock = Arc::new(MockGraphStore::new());
        let graph: Arc<dyn GraphStore> = mock;
        let result = route_lifecycle_protocol(&graph, Uuid::new_v4(), 5)
            .await
            .unwrap();
        assert!(result.is_none(), "Should return None when plan not found");
    }

    #[tokio::test]
    async fn test_route_with_lifecycle_protocol_happy_path() {
        use crate::protocol::models::{Protocol, ProtocolState, ProtocolTransition};

        let mock = Arc::new(MockGraphStore::new());
        let (plan_id, project_id) = setup_plan_with_project(&mock).await;

        // Create a lifecycle protocol with states and transitions
        let protocol_id = Uuid::new_v4();
        let init_state = ProtocolState::start(protocol_id, "init");
        let executing_state = ProtocolState::new(protocol_id, "executing");

        let mut protocol = Protocol::new(project_id, "plan-runner-full", init_state.id);
        protocol.id = protocol_id;
        protocol.terminal_states = vec![];

        mock.upsert_protocol(&protocol).await.unwrap();
        mock.upsert_protocol_state(&init_state).await.unwrap();
        mock.upsert_protocol_state(&executing_state).await.unwrap();

        // Add transition: init --run_started--> executing
        let transition = ProtocolTransition::new(
            protocol_id,
            init_state.id,
            executing_state.id,
            "run_started",
        );
        mock.upsert_protocol_transition(&transition).await.unwrap();

        let graph: Arc<dyn GraphStore> = mock;
        let result = route_lifecycle_protocol(&graph, plan_id, 5).await.unwrap();

        assert!(result.is_some(), "Should route to a lifecycle protocol");
        let route = result.unwrap();
        assert_eq!(route.protocol_id, protocol_id);
        assert_eq!(route.protocol_name, "plan-runner-full");
        assert!(route.affinity_score >= MIN_AFFINITY_THRESHOLD);
    }

    #[tokio::test]
    async fn test_route_filters_non_runner_protocols() {
        use crate::protocol::models::{Protocol, ProtocolState};

        let mock = Arc::new(MockGraphStore::new());
        let (plan_id, project_id) = setup_plan_with_project(&mock).await;

        // Create a protocol that does NOT start with "plan-runner-"
        let protocol_id = Uuid::new_v4();
        let init_state = ProtocolState::start(protocol_id, "init");

        let mut protocol = Protocol::new(project_id, "code-review-protocol", init_state.id);
        protocol.id = protocol_id;

        mock.upsert_protocol(&protocol).await.unwrap();
        mock.upsert_protocol_state(&init_state).await.unwrap();

        let graph: Arc<dyn GraphStore> = mock;
        let result = route_lifecycle_protocol(&graph, plan_id, 5).await.unwrap();

        assert!(
            result.is_none(),
            "Should return None when no plan-runner-* protocols exist"
        );
    }

    #[test]
    fn test_lifecycle_route_result_debug_clone() {
        let result = LifecycleRouteResult {
            protocol_id: Uuid::new_v4(),
            protocol_name: "plan-runner-full".to_string(),
            affinity_score: 0.85,
            run_id: Uuid::new_v4(),
        };

        // Verify Debug and Clone are implemented
        let cloned = result.clone();
        assert_eq!(cloned.protocol_name, "plan-runner-full");
        assert!((cloned.affinity_score - 0.85).abs() < f64::EPSILON);
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("plan-runner-full"));
    }

    #[test]
    fn test_constants() {
        assert_eq!(RUNNER_PROTOCOL_PREFIX, "plan-runner-");
        assert!(MIN_AFFINITY_THRESHOLD > 0.0);
        assert!(MIN_AFFINITY_THRESHOLD < 1.0);
    }
}
