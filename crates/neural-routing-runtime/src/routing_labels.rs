//! Labelled dataset of the tool-section routing: one row per closed turn.
//!
//! A row is `(decision, context, label)`:
//! - **decision**: the routing action (`routing.select_sections`) of the turn;
//! - **context**: what the router saw (intent, scaffolding level, section weights and
//!   the sections it selected). The user message is not stored in trajectories, so it
//!   is not exported;
//! - **label**: the tool groups the turn both called and had predicted (`outcome.hits`),
//!   with the turn's outcome (success, cost, latency, recall, precision) next to it.
//!
//! Pure over [`Trajectory`] values: the caller loads them from the trajectory store.

use neural_routing_core::Trajectory;
use serde::Serialize;
use uuid::Uuid;

use crate::collector::ROUTING_SECTIONS_ACTION;

/// One labelled turn.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RoutingLabelRow {
    pub trajectory_id: Uuid,
    pub session_id: String,
    pub turn_index: u64,
    /// The decision: the action type of the routing node (`routing.select_sections`).
    pub decision: String,
    /// What the router saw when it decided.
    pub context: serde_json::Value,
    /// Predicted tool groups that the turn called.
    pub label: Vec<String>,
    /// The turn's outcome, as recorded at its close.
    pub outcome: serde_json::Value,
}

/// The labelled rows of the closed turns of the given trajectories, in their order.
/// Turns still open (no outcome) and session-level routing records are skipped.
pub fn routing_label_rows(trajectories: &[Trajectory]) -> Vec<RoutingLabelRow> {
    let mut rows = Vec::new();
    for trajectory in trajectories {
        for node in &trajectory.nodes {
            if node.action_type != ROUTING_SECTIONS_ACTION {
                continue;
            }
            let (Some(turn_index), Some(outcome)) = (
                node.action_params
                    .get("turn_index")
                    .and_then(|v| v.as_u64()),
                node.outcome.as_ref(),
            ) else {
                continue;
            };
            let label = outcome
                .get("hits")
                .and_then(serde_json::Value::as_array)
                .map(|hits| {
                    hits.iter()
                        .filter_map(|h| h.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            let params = &node.action_params;
            rows.push(RoutingLabelRow {
                trajectory_id: trajectory.id,
                session_id: trajectory.session_id.clone(),
                turn_index,
                decision: node.action_type.clone(),
                context: serde_json::json!({
                    "detected_intent": params.get("detected_intent"),
                    "scaffolding_level": params.get("scaffolding_level"),
                    "section_weights": params.get("section_weights"),
                    "selected_sections": params.get("selected_sections"),
                    "selected_tool_groups": params.get("selected_tool_groups"),
                }),
                label,
                outcome: outcome.clone(),
            });
        }
    }
    rows
}
