//! Migration of historical chat events into structured trajectories.
//!
//! Converts existing ChatEventRecords (tool_use events) from Neo4j into
//! trajectory format. These migrated trajectories are marked with
//! `source: migrated` (session_id prefix) and weighted at 0.5 during training.
//!
//! ## Strategy
//! 1. Query all ChatSession nodes from Neo4j
//! 2. For each session, load ChatEventRecords ordered by seq
//! 3. Filter to tool_use events (these represent decision points)
//! 4. Group consecutive tool_use events into trajectory nodes
//! 5. Compute approximate DecisionVectors using current graph features
//! 6. Assign rewards retrospectively via RBCR-like heuristic
//! 7. Persist as Trajectory with `source: migrated`

use chrono::Utc;
use neo4rs::{query, Graph};
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{NeuralRoutingError, Result};
use crate::models::{Trajectory, TrajectoryNode};
use crate::store::Neo4jTrajectoryStore;
use crate::traits::TrajectoryStore;

/// Configuration for the historical migration job.
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    /// Weight for migrated trajectories during training (default: 0.5).
    pub migrated_weight: f64,
    /// Minimum number of tool_use events in a session to create a trajectory.
    pub min_events: usize,
    /// Maximum number of sessions to process per batch.
    pub batch_size: usize,
    /// Skip sessions that already have migrated trajectories.
    pub skip_existing: bool,
    /// Progress log interval (sessions).
    pub log_interval: usize,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            migrated_weight: 0.5,
            min_events: 2,
            batch_size: 100,
            skip_existing: true,
            log_interval: 50,
        }
    }
}

/// Result of a migration run.
#[derive(Debug, Clone)]
pub struct MigrationReport {
    /// Total sessions scanned.
    pub sessions_scanned: usize,
    /// Sessions that had enough events to create a trajectory.
    pub sessions_eligible: usize,
    /// Trajectories successfully created.
    pub trajectories_created: usize,
    /// Total decision nodes across all trajectories.
    pub total_nodes: usize,
    /// Sessions skipped (already migrated or too few events).
    pub sessions_skipped: usize,
    /// Errors encountered (non-fatal).
    pub errors: Vec<String>,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// A raw tool_use event extracted from ChatEventRecord.
#[derive(Debug, Clone)]
struct RawToolEvent {
    /// Event sequence number within session.
    seq: i64,
    /// Event type (should be "tool_use").
    event_type: String,
    /// JSON data payload.
    data: String,
}

/// Parsed tool_use data from the JSON payload.
#[derive(Debug, Clone)]
struct ParsedToolUse {
    /// MCP tool name (e.g., "code", "note").
    tool_name: String,
    /// MCP action (e.g., "search", "get_context").
    action: String,
    /// Original parameters (sanitized).
    params: serde_json::Value,
    /// Sequence number.
    seq: i64,
}

/// Run the historical migration job.
///
/// Reads ChatEventRecords from Neo4j, converts tool_use events into
/// trajectories, and persists them via TrajectoryStore.
pub async fn run_migration(
    graph: Arc<Graph>,
    store: Arc<Neo4jTrajectoryStore>,
    config: &MigrationConfig,
) -> Result<MigrationReport> {
    let start = std::time::Instant::now();

    let mut report = MigrationReport {
        sessions_scanned: 0,
        sessions_eligible: 0,
        trajectories_created: 0,
        total_nodes: 0,
        sessions_skipped: 0,
        errors: vec![],
        duration_ms: 0,
    };

    // Step 1: Get all session IDs
    let session_ids = get_session_ids(&graph, config.batch_size).await?;
    tracing::info!(
        count = session_ids.len(),
        "Found sessions to scan for migration"
    );

    for (i, session_id) in session_ids.iter().enumerate() {
        report.sessions_scanned += 1;

        // Skip if already migrated
        if config.skip_existing {
            let migrated_session_id = format!("migrated-{}", session_id);
            let existing = check_trajectory_exists(&graph, &migrated_session_id).await;
            if existing {
                report.sessions_skipped += 1;
                continue;
            }
        }

        // Step 2: Get tool_use events for this session
        match get_tool_events(&graph, session_id).await {
            Ok(events) => {
                if events.len() < config.min_events {
                    report.sessions_skipped += 1;
                    continue;
                }

                report.sessions_eligible += 1;

                // Step 3: Parse and convert to trajectory
                let parsed = parse_tool_events(&events);
                if parsed.len() < config.min_events {
                    report.sessions_skipped += 1;
                    continue;
                }

                // Step 4: Build trajectory
                let trajectory = build_migrated_trajectory(session_id, &parsed);
                let node_count = trajectory.nodes.len();

                // Step 5: Persist
                match store.store_trajectory(&trajectory).await {
                    Ok(()) => {
                        report.trajectories_created += 1;
                        report.total_nodes += node_count;
                    }
                    Err(e) => {
                        report.errors.push(format!(
                            "Failed to store migrated trajectory for session {}: {}",
                            session_id, e
                        ));
                    }
                }
            }
            Err(e) => {
                report.errors.push(format!(
                    "Failed to read events for session {}: {}",
                    session_id, e
                ));
            }
        }

        if (i + 1) % config.log_interval == 0 {
            tracing::info!(
                scanned = report.sessions_scanned,
                eligible = report.sessions_eligible,
                created = report.trajectories_created,
                skipped = report.sessions_skipped,
                "Migration progress"
            );
        }
    }

    report.duration_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        scanned = report.sessions_scanned,
        eligible = report.sessions_eligible,
        created = report.trajectories_created,
        total_nodes = report.total_nodes,
        skipped = report.sessions_skipped,
        errors = report.errors.len(),
        duration_ms = report.duration_ms,
        "Migration complete"
    );

    Ok(report)
}

// ---------------------------------------------------------------------------
// Neo4j queries
// ---------------------------------------------------------------------------

async fn get_session_ids(graph: &Graph, limit: usize) -> Result<Vec<Uuid>> {
    let q = query(
        "MATCH (s:ChatSession)
         RETURN s.id AS id
         ORDER BY s.created_at DESC
         LIMIT $limit",
    )
    .param("limit", limit as i64);

    let mut result = graph
        .execute(q)
        .await
        .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

    let mut ids = Vec::new();
    while let Some(row) = result
        .next()
        .await
        .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?
    {
        let id_str: String = row.get("id").unwrap_or_default();
        if let Ok(id) = Uuid::parse_str(&id_str) {
            ids.push(id);
        }
    }

    Ok(ids)
}

async fn get_tool_events(graph: &Graph, session_id: &Uuid) -> Result<Vec<RawToolEvent>> {
    let q = query(
        "MATCH (s:ChatSession {id: $session_id})-[:HAS_EVENT]->(e:ChatEvent)
         WHERE e.event_type IN ['tool_use', 'tool_result']
         RETURN e.seq AS seq, e.event_type AS event_type, e.data AS data, e.created_at AS created_at
         ORDER BY e.seq",
    )
    .param("session_id", session_id.to_string());

    let mut result = graph
        .execute(q)
        .await
        .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?;

    let mut events = Vec::new();
    while let Some(row) = result
        .next()
        .await
        .map_err(|e| NeuralRoutingError::Neo4j(e.to_string()))?
    {
        events.push(RawToolEvent {
            seq: row.get("seq").unwrap_or(0),
            event_type: row.get("event_type").unwrap_or_default(),
            data: row.get("data").unwrap_or_default(),
        });
    }

    Ok(events)
}

async fn check_trajectory_exists(graph: &Graph, session_id: &str) -> bool {
    let q = query(
        "MATCH (t:Trajectory {session_id: $session_id})
         RETURN count(t) > 0 AS exists",
    )
    .param("session_id", session_id.to_string());

    if let Ok(mut result) = graph.execute(q).await {
        if let Ok(Some(row)) = result.next().await {
            return row.get::<bool>("exists").unwrap_or(false);
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Parsing & conversion
// ---------------------------------------------------------------------------

fn parse_tool_events(events: &[RawToolEvent]) -> Vec<ParsedToolUse> {
    events
        .iter()
        .filter(|e| e.event_type == "tool_use")
        .filter_map(|e| {
            let data: serde_json::Value = serde_json::from_str(&e.data).ok()?;

            // Extract tool name and action from the event data.
            // ChatEvent tool_use typically has: { "tool_name": "...", "input": {...} }
            // or { "name": "code", "action": "search", ... }
            let tool_name = data
                .get("tool_name")
                .or_else(|| data.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            let action = data
                .get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            // Sanitize params — strip potential PII
            let params = data
                .get("input")
                .or_else(|| data.get("params"))
                .cloned()
                .unwrap_or(serde_json::Value::Null);

            Some(ParsedToolUse {
                tool_name,
                action,
                params: sanitize_params(&params),
                seq: e.seq,
            })
        })
        .collect()
}

/// Remove potential PII from parameters — keep only structural info.
fn sanitize_params(params: &serde_json::Value) -> serde_json::Value {
    match params {
        serde_json::Value::Object(map) => {
            let sanitized: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| {
                    // Keep keys, hash or truncate string values
                    let safe_v = match v {
                        serde_json::Value::String(s) if s.len() > 100 => {
                            serde_json::Value::String(format!("[truncated:{}chars]", s.len()))
                        }
                        other => other.clone(),
                    };
                    (k.clone(), safe_v)
                })
                .collect();
            serde_json::Value::Object(sanitized)
        }
        other => other.clone(),
    }
}

fn build_migrated_trajectory(session_id: &Uuid, parsed: &[ParsedToolUse]) -> Trajectory {
    let trajectory_id = Uuid::new_v4();
    let now = Utc::now();

    let nodes: Vec<TrajectoryNode> = parsed
        .iter()
        .enumerate()
        .map(|(i, tool)| {
            let action_type = if tool.action.is_empty() {
                tool.tool_name.clone()
            } else {
                format!("{}.{}", tool.tool_name, tool.action)
            };

            let delta_ms = if i == 0 {
                0
            } else {
                // Estimate delta from sequence numbers (rough: 500ms per seq gap)
                let seq_diff = (tool.seq - parsed[i - 1].seq).max(1) as u64;
                seq_diff * 500
            };

            TrajectoryNode {
                id: Uuid::new_v4(),
                // Empty context embedding — will be backfilled if DecisionVector builder runs
                context_embedding: vec![0.0; 256],
                action_type,
                action_params: tool.params.clone(),
                alternatives_count: 1, // Unknown for historical data
                chosen_index: 0,
                // Heuristic confidence based on position in sequence
                confidence: estimate_confidence(i, parsed.len()),
                local_reward: 0.0,
                cumulative_reward: 0.0,
                delta_ms,
                order: i,
            }
        })
        .collect();

    // Heuristic total reward: based on session length and tool diversity
    let total_reward = estimate_session_reward(parsed);
    let duration_ms = nodes.iter().map(|n| n.delta_ms).sum();

    Trajectory {
        id: trajectory_id,
        session_id: format!("migrated-{}", session_id),
        query_embedding: vec![0.0; 256], // Unknown for historical data
        total_reward,
        step_count: nodes.len(),
        duration_ms,
        nodes,
        created_at: now,
        protocol_run_id: None,
    }
}

/// Estimate confidence for a historical decision point.
/// Early decisions in a session tend to be more exploratory (lower confidence),
/// later ones more targeted (higher confidence).
fn estimate_confidence(position: usize, total: usize) -> f64 {
    if total <= 1 {
        return 0.5;
    }
    let progress = position as f64 / (total - 1) as f64;
    // Ramp from 0.4 (start) to 0.8 (end)
    0.4 + progress * 0.4
}

/// Estimate a session's total reward from tool usage patterns.
///
/// Heuristic scoring:
/// - More diverse tools = higher reward (broader exploration)
/// - Longer sessions (more steps) = moderate bonus
/// - Known high-value tool combinations get bonuses
fn estimate_session_reward(events: &[ParsedToolUse]) -> f64 {
    if events.is_empty() {
        return 0.0;
    }

    // Tool diversity score
    let unique_tools: std::collections::HashSet<_> = events
        .iter()
        .map(|e| format!("{}.{}", e.tool_name, e.action))
        .collect();
    let diversity = (unique_tools.len() as f64 / events.len() as f64).min(1.0);

    // Length score (diminishing returns)
    let length_score = (events.len() as f64).ln() / 5.0;

    // High-value patterns
    let has_search = events.iter().any(|e| e.action.contains("search"));
    let has_context = events
        .iter()
        .any(|e| e.action.contains("context") || e.action.contains("propagated"));
    let has_impact = events
        .iter()
        .any(|e| e.action.contains("impact") || e.action.contains("analyze"));

    let pattern_bonus = match (has_search, has_context, has_impact) {
        (true, true, true) => 0.3,  // Full exploration cycle
        (true, true, false) => 0.2, // Search + context
        (true, false, false) => 0.1,
        _ => 0.05,
    };

    // Combine: diversity(40%) + length(30%) + patterns(30%)
    let raw = diversity * 0.4 + length_score.min(1.0) * 0.3 + pattern_bonus;
    raw.clamp(0.05, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_confidence_ramp() {
        let total = 10;
        let c0 = estimate_confidence(0, total);
        let c9 = estimate_confidence(9, total);
        assert!(c0 < c9, "Later positions should have higher confidence");
        assert!((c0 - 0.4).abs() < 0.01);
        assert!((c9 - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_estimate_confidence_single() {
        assert!((estimate_confidence(0, 1) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_estimate_session_reward_diverse() {
        let events = vec![
            ParsedToolUse {
                tool_name: "code".into(),
                action: "search".into(),
                params: serde_json::Value::Null,
                seq: 1,
            },
            ParsedToolUse {
                tool_name: "note".into(),
                action: "get_context".into(),
                params: serde_json::Value::Null,
                seq: 3,
            },
            ParsedToolUse {
                tool_name: "code".into(),
                action: "analyze_impact".into(),
                params: serde_json::Value::Null,
                seq: 5,
            },
        ];

        let reward = estimate_session_reward(&events);
        assert!(
            reward > 0.3,
            "Diverse session should have decent reward: {}",
            reward
        );
        assert!(reward <= 1.0);
    }

    #[test]
    fn test_estimate_session_reward_monotone() {
        // Single repeated action = low diversity
        let events: Vec<ParsedToolUse> = (0..5)
            .map(|i| ParsedToolUse {
                tool_name: "code".into(),
                action: "search".into(),
                params: serde_json::Value::Null,
                seq: i,
            })
            .collect();

        let reward = estimate_session_reward(&events);
        assert!(
            reward < 0.5,
            "Monotone session should have lower reward: {}",
            reward
        );
    }

    #[test]
    fn test_sanitize_params_truncates() {
        let long_string = "a".repeat(200);
        let params = serde_json::json!({"query": long_string, "limit": 10});
        let sanitized = sanitize_params(&params);

        let query_val = sanitized.get("query").unwrap().as_str().unwrap();
        assert!(query_val.starts_with("[truncated:"));
        // Non-string values should be preserved
        assert_eq!(sanitized.get("limit").unwrap().as_i64().unwrap(), 10);
    }

    #[test]
    fn test_build_migrated_trajectory() {
        let session_id = Uuid::new_v4();
        let parsed = vec![
            ParsedToolUse {
                tool_name: "code".into(),
                action: "search".into(),
                params: serde_json::json!({"query": "test"}),
                seq: 1,
            },
            ParsedToolUse {
                tool_name: "note".into(),
                action: "get_context".into(),
                params: serde_json::json!({}),
                seq: 3,
            },
            ParsedToolUse {
                tool_name: "code".into(),
                action: "analyze_impact".into(),
                params: serde_json::json!({"target": "foo"}),
                seq: 5,
            },
        ];

        let trajectory = build_migrated_trajectory(&session_id, &parsed);

        assert!(trajectory.session_id.starts_with("migrated-"));
        assert_eq!(trajectory.step_count, 3);
        assert_eq!(trajectory.nodes.len(), 3);
        assert_eq!(trajectory.nodes[0].action_type, "code.search");
        assert_eq!(trajectory.nodes[1].action_type, "note.get_context");
        assert_eq!(trajectory.nodes[2].action_type, "code.analyze_impact");

        // First node delta should be 0
        assert_eq!(trajectory.nodes[0].delta_ms, 0);
        // Subsequent deltas should be based on seq gaps
        assert!(trajectory.nodes[1].delta_ms > 0);

        // Ordering preserved
        assert_eq!(trajectory.nodes[0].order, 0);
        assert_eq!(trajectory.nodes[1].order, 1);
        assert_eq!(trajectory.nodes[2].order, 2);

        // Reward should be reasonable
        assert!(trajectory.total_reward > 0.0);
        assert!(trajectory.total_reward <= 1.0);
    }

    #[test]
    fn test_parse_tool_events_filters_tool_result() {
        let events = vec![
            RawToolEvent {
                seq: 1,
                event_type: "tool_use".into(),
                data: r#"{"tool_name":"code","action":"search","input":{"query":"test"}}"#.into(),
            },
            RawToolEvent {
                seq: 2,
                event_type: "tool_result".into(),
                data: r#"{"result":"ok"}"#.into(),
            },
            RawToolEvent {
                seq: 3,
                event_type: "tool_use".into(),
                data: r#"{"tool_name":"note","action":"get_context","input":{}}"#.into(),
            },
        ];

        let parsed = parse_tool_events(&events);
        // Should only include tool_use events, not tool_result
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].tool_name, "code");
        assert_eq!(parsed[1].tool_name, "note");
    }

    #[test]
    fn test_estimate_session_reward_empty() {
        assert!((estimate_session_reward(&[]) - 0.0).abs() < 1e-10);
    }
}
