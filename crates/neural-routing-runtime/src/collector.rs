//! TrajectoryCollector — fire-and-forget decision capture with mpsc channel.
//!
//! Architecture:
//! - Hot path: `record_decision()` sends a `DecisionEvent` via bounded mpsc channel (~0 latency)
//! - Background task: receives events, buffers them, flushes in batches to Neo4j
//! - Session-scoped: each session gets its own trajectory, finalized on `end_session()`

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use uuid::Uuid;

use neural_routing_core::{
    DecisionVectorBuilder, Neo4jTrajectoryStore, ToolUsage, TouchedEntity, Trajectory,
    TrajectoryNode, TrajectoryStore,
};

use crate::config::CollectionConfig;
use crate::reward::{SessionRewardComputer, SessionSignals};

// ---------------------------------------------------------------------------
// Events sent through the mpsc channel
// ---------------------------------------------------------------------------

/// A decision event sent from the hot path to the background collector.
#[derive(Debug, Clone)]
pub enum CollectorEvent {
    /// Record a single decision point.
    Decision(DecisionRecord),
    /// End a session — triggers trajectory finalization and flush.
    EndSession {
        session_id: String,
        total_reward: f64,
    },
    /// Graceful shutdown — flush all pending data.
    Shutdown,
}

/// A single decision point captured from the hot path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// Session this decision belongs to.
    pub session_id: String,
    /// Context embedding (256d, L2-normalized). Empty if not yet computed.
    /// If empty, the collector will compute it using the DecisionVectorBuilder.
    pub context_embedding: Vec<f32>,
    /// MCP tool + action invoked (e.g., "code.search", "note.get_context").
    pub action_type: String,
    /// Serialized key parameters (stripped of PII).
    pub action_params: serde_json::Value,
    /// Number of alternative actions considered.
    pub alternatives_count: usize,
    /// Index of the chosen action among alternatives (0 if no alternatives).
    pub chosen_index: usize,
    /// Model confidence in this decision (0.0 - 1.0).
    pub confidence: f64,
    /// Tools used at this decision point.
    pub tool_usages: Vec<ToolUsage>,
    /// Entities touched at this decision point.
    pub touched_entities: Vec<TouchedEntity>,
    /// Timestamp in ms since session start.
    pub timestamp_ms: u64,
    /// Query embedding (768d source, e.g. from Voyage). Optional — used by the
    /// VectorBuilder to compute the 64d query block.
    #[serde(default)]
    pub query_embedding: Vec<f32>,
    /// GDS node features for recently touched graph nodes. Optional — used by the
    /// VectorBuilder to compute the 64d graph_state block.
    #[serde(default)]
    pub node_features: Vec<neural_routing_core::NodeFeatures>,
}

// ---------------------------------------------------------------------------
// TrajectoryCollector — the public API
// ---------------------------------------------------------------------------

/// Fire-and-forget trajectory collector.
///
/// The `record_decision()` method sends events via a bounded mpsc channel.
/// A background task receives and batches them for Neo4j persistence.
///
/// Latency budget: <1ms on the hot path (just a channel send).
pub struct TrajectoryCollector {
    /// Sender half of the bounded mpsc channel.
    tx: mpsc::Sender<CollectorEvent>,
    /// Whether collection is enabled (runtime toggle).
    enabled: Arc<std::sync::atomic::AtomicBool>,
    /// Shared vector builder for computing context embeddings.
    vector_builder: Arc<DecisionVectorBuilder>,
}

impl TrajectoryCollector {
    /// Create a new collector and spawn the background flush task.
    ///
    /// The `vector_builder` is used to compute 256d context embeddings for each
    /// decision point. Pass `None` to use a default builder.
    ///
    /// Returns the collector handle and a JoinHandle for the background task.
    pub fn new(
        store: Arc<Neo4jTrajectoryStore>,
        config: &CollectionConfig,
        vector_builder: Option<Arc<DecisionVectorBuilder>>,
    ) -> (Self, tokio::task::JoinHandle<()>) {
        Self::new_with_reward(
            store,
            config,
            vector_builder,
            SessionRewardComputer::default(),
        )
    }

    /// Create a collector with a custom reward computer.
    pub fn new_with_reward(
        store: Arc<Neo4jTrajectoryStore>,
        config: &CollectionConfig,
        vector_builder: Option<Arc<DecisionVectorBuilder>>,
        reward_computer: SessionRewardComputer,
    ) -> (Self, tokio::task::JoinHandle<()>) {
        let buffer_size = config.buffer_size;
        let stale_timeout_secs = config.stale_session_timeout_secs;
        // Channel capacity = 2x buffer to absorb bursts without blocking
        let (tx, rx) = mpsc::channel(buffer_size * 2);

        let enabled = Arc::new(std::sync::atomic::AtomicBool::new(config.enabled));
        let enabled_clone = enabled.clone();
        let builder = vector_builder.unwrap_or_else(|| Arc::new(DecisionVectorBuilder::new()));
        let builder_clone = builder.clone();

        let handle = tokio::spawn(async move {
            run_collector_loop(
                rx,
                store,
                buffer_size,
                enabled_clone,
                builder_clone,
                stale_timeout_secs,
                reward_computer,
            )
            .await;
        });

        let collector = Self {
            tx,
            enabled,
            vector_builder: builder,
        };
        (collector, handle)
    }

    /// Get a reference to the shared vector builder.
    ///
    /// Callers can use this to compute context embeddings before calling `record_decision()`.
    pub fn vector_builder(&self) -> &Arc<DecisionVectorBuilder> {
        &self.vector_builder
    }

    /// Record a decision point (fire-and-forget).
    ///
    /// Returns immediately. If the channel is full, the event is dropped
    /// (we never block the hot path).
    pub fn record_decision(&self, record: DecisionRecord) {
        if !self.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        // try_send: non-blocking, drops if channel full
        let _ = self.tx.try_send(CollectorEvent::Decision(record));
    }

    /// End a session — triggers finalization and flush of the trajectory.
    pub fn end_session(&self, session_id: String, total_reward: f64) {
        if !self.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        if let Err(e) = self.tx.try_send(CollectorEvent::EndSession {
            session_id: session_id.clone(),
            total_reward,
        }) {
            tracing::warn!(
                session_id = %session_id,
                error = %e,
                "Failed to send EndSession event — trajectory will be lost"
            );
        }
    }

    /// Request graceful shutdown — flushes all pending data.
    pub async fn shutdown(&self) {
        let _ = self.tx.send(CollectorEvent::Shutdown).await;
    }

    /// Toggle collection on/off at runtime.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Check if collection is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(std::sync::atomic::Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Background collector loop
// ---------------------------------------------------------------------------

/// In-flight session state (buffered decisions before finalization).
struct SessionBuffer {
    decisions: Vec<DecisionRecord>,
    started_at: chrono::DateTime<Utc>,
    /// Monotonic instant of the last decision — used for stale session detection.
    last_decision_at: Instant,
}

async fn run_collector_loop(
    mut rx: mpsc::Receiver<CollectorEvent>,
    store: Arc<Neo4jTrajectoryStore>,
    buffer_size: usize,
    _enabled: Arc<std::sync::atomic::AtomicBool>,
    vector_builder: Arc<DecisionVectorBuilder>,
    stale_timeout_secs: u64,
    reward_computer: SessionRewardComputer,
) {
    let mut sessions: HashMap<String, SessionBuffer> = HashMap::new();
    let mut pending_flush: Vec<PendingTrajectory> = Vec::new();

    // Stale session check interval = timeout/2 (minimum 5s to avoid busy-loop)
    let check_interval = std::time::Duration::from_secs((stale_timeout_secs / 2).max(5));
    let stale_timeout = std::time::Duration::from_secs(stale_timeout_secs);
    let mut stale_timer = tokio::time::interval(check_interval);
    // Don't fire immediately on startup
    stale_timer.tick().await;

    loop {
        tokio::select! {
            event = rx.recv() => {
                let Some(event) = event else { break };
                match event {
                    CollectorEvent::Decision(record) => {
                        let session = sessions
                            .entry(record.session_id.clone())
                            .or_insert_with(|| SessionBuffer {
                                decisions: Vec::new(),
                                started_at: Utc::now(),
                                last_decision_at: Instant::now(),
                            });
                        session.last_decision_at = Instant::now();
                        session.decisions.push(record);
                    }

                    CollectorEvent::EndSession {
                        session_id,
                        total_reward,
                    } => {
                        if let Some(buffer) = sessions.remove(&session_id) {
                            let trajectory =
                                build_trajectory(&session_id, &buffer, total_reward, &vector_builder);
                            pending_flush.push(trajectory);

                            // Flush if we've accumulated enough
                            if pending_flush.len() >= buffer_size {
                                let to_flush = std::mem::take(&mut pending_flush);
                                flush_batch(&store, to_flush).await;
                            }
                        }
                    }

                    CollectorEvent::Shutdown => {
                        // Finalize any open sessions with auto-computed reward
                        let open_sessions: Vec<_> = sessions.drain().collect();
                        for (session_id, buffer) in open_sessions {
                            let reward = compute_auto_reward(&buffer, &reward_computer);
                            let trajectory = build_trajectory(&session_id, &buffer, reward, &vector_builder);
                            pending_flush.push(trajectory);
                        }

                        // Final flush
                        if !pending_flush.is_empty() {
                            let to_flush = std::mem::take(&mut pending_flush);
                            flush_batch(&store, to_flush).await;
                        }

                        tracing::info!("TrajectoryCollector shutdown complete");
                        break;
                    }
                }
            }

            _ = stale_timer.tick() => {
                // Check for stale sessions (no new decision for > stale_timeout)
                let now = Instant::now();
                let stale_ids: Vec<String> = sessions
                    .iter()
                    .filter(|(_, buf)| now.duration_since(buf.last_decision_at) > stale_timeout)
                    .map(|(id, _)| id.clone())
                    .collect();

                for session_id in stale_ids {
                    if let Some(buffer) = sessions.remove(&session_id) {
                        let reward = compute_auto_reward(&buffer, &reward_computer);
                        tracing::info!(
                            session_id = %session_id,
                            decisions = buffer.decisions.len(),
                            reward = %format!("{:.3}", reward),
                            "Auto-finalized stale session"
                        );
                        let trajectory = build_trajectory(&session_id, &buffer, reward, &vector_builder);
                        pending_flush.push(trajectory);
                    }
                }

                // Flush if we have pending trajectories from stale sessions
                if pending_flush.len() >= buffer_size {
                    let to_flush = std::mem::take(&mut pending_flush);
                    flush_batch(&store, to_flush).await;
                }
            }
        }
    }
}

/// Compute reward from buffered decisions using the SessionRewardComputer.
fn compute_auto_reward(buffer: &SessionBuffer, computer: &SessionRewardComputer) -> f64 {
    let duration = (Utc::now() - buffer.started_at)
        .to_std()
        .unwrap_or(std::time::Duration::ZERO);
    let signals = SessionSignals::from_decisions(&buffer.decisions, duration, 0, 0);
    computer.compute(&signals)
}

// ---------------------------------------------------------------------------
// Trajectory building and flushing
// ---------------------------------------------------------------------------

/// A trajectory ready to be persisted, with its relation metadata.
struct PendingTrajectory {
    trajectory: Trajectory,
    /// Tool usages per node (indexed by node order).
    tool_usages: Vec<Vec<ToolUsage>>,
    /// Touched entities per node (indexed by node order).
    touched_entities: Vec<Vec<TouchedEntity>>,
}

fn build_trajectory(
    session_id: &str,
    buffer: &SessionBuffer,
    total_reward: f64,
    vector_builder: &DecisionVectorBuilder,
) -> PendingTrajectory {
    use neural_routing_core::{DecisionContext, SessionMeta, TOTAL_DIM};

    let trajectory_id = Uuid::new_v4();
    let now = Utc::now();
    let duration_ms = (now - buffer.started_at).num_milliseconds().max(0) as u64;

    let mut nodes = Vec::with_capacity(buffer.decisions.len());
    let mut tool_usages = Vec::with_capacity(buffer.decisions.len());
    let mut touched_entities = Vec::with_capacity(buffer.decisions.len());
    let mut previous_embeddings: Vec<Vec<f32>> = Vec::new();

    // Track session-level stats for the session_meta block
    let mut unique_tools: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut unique_entities: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut confidence_sum = 0.0_f64;
    let mut error_count = 0_usize;
    let mut prev_ts = 0u64;

    for (i, decision) in buffer.decisions.iter().enumerate() {
        let delta_ms = if i == 0 {
            decision.timestamp_ms
        } else {
            decision.timestamp_ms.saturating_sub(prev_ts)
        };
        prev_ts = decision.timestamp_ms;

        // Track stats for session_meta
        unique_tools.insert(decision.action_type.clone());
        for te in &decision.touched_entities {
            unique_entities.insert(format!("{}:{}", te.entity_type, te.entity_id));
        }
        confidence_sum += decision.confidence;
        for tu in &decision.tool_usages {
            if !tu.success {
                error_count += 1;
            }
        }

        // Compute the context embedding via the VectorBuilder if not already provided
        let context_embedding = if !decision.context_embedding.is_empty() {
            decision.context_embedding.clone()
        } else {
            // Parse tool_name and action_name from action_type ("code.search" → "code", "search")
            let (tool_name, action_name) = decision
                .action_type
                .split_once('.')
                .unwrap_or((&decision.action_type, "unknown"));

            let ctx = DecisionContext {
                query_embedding: decision.query_embedding.clone(),
                touched_node_features: decision.node_features.clone(),
                previous_embeddings: previous_embeddings.clone(),
                tool_name: tool_name.to_string(),
                action_name: action_name.to_string(),
                params_hash: neural_routing_core::vector_builder::simple_hash(
                    &decision.action_params.to_string(),
                ),
                session_meta: SessionMeta {
                    duration_ms: decision.timestamp_ms,
                    decision_count: i,
                    cumulative_reward: 0.0, // not yet known during collection
                    avg_confidence: if i > 0 {
                        confidence_sum / i as f64
                    } else {
                        decision.confidence
                    },
                    unique_tools_used: unique_tools.len(),
                    unique_entities_touched: unique_entities.len(),
                    error_count,
                },
            };

            vector_builder.build(&ctx)
        };

        // Accumulate for history block of subsequent decisions
        previous_embeddings.push(context_embedding.clone());

        nodes.push(TrajectoryNode {
            id: Uuid::new_v4(),
            context_embedding,
            action_type: decision.action_type.clone(),
            action_params: decision.action_params.clone(),
            alternatives_count: decision.alternatives_count,
            chosen_index: decision.chosen_index,
            confidence: decision.confidence,
            local_reward: 0.0, // Will be filled by RewardDecomposer in step 4
            cumulative_reward: 0.0,
            delta_ms,
            order: i,
        });

        tool_usages.push(decision.tool_usages.clone());
        touched_entities.push(decision.touched_entities.clone());
    }

    // ── Reward decomposition (synchronous TD strategy) ────────────────────
    // Decompose total_reward into per-step local_reward using confidence-weighted
    // TD decomposition (same logic as TDRewardStrategy but synchronous).
    if !nodes.is_empty() {
        let gamma = 0.99_f64; // default discount factor
        let confidence_sum: f64 = nodes.iter().map(|n| n.confidence).sum();
        let n = nodes.len();

        if confidence_sum > 1e-10 {
            let mut raw_rewards: Vec<f64> = nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    let discount = gamma.powi((n - 1 - i) as i32);
                    let weight = node.confidence / confidence_sum;
                    total_reward * weight * discount
                })
                .collect();

            // Renormalize so sum matches total_reward
            let sum: f64 = raw_rewards.iter().sum();
            if sum.abs() > 1e-10 {
                let scale = total_reward / sum;
                for r in &mut raw_rewards {
                    *r *= scale;
                }
            }

            // Fill local_reward and cumulative_reward
            let mut cumulative = 0.0_f64;
            for (i, node) in nodes.iter_mut().enumerate() {
                node.local_reward = raw_rewards[i];
                cumulative += raw_rewards[i];
                node.cumulative_reward = cumulative;
            }
        } else {
            // Equal distribution if all confidences are zero
            let equal = total_reward / n as f64;
            let mut cumulative = 0.0_f64;
            for node in nodes.iter_mut() {
                node.local_reward = equal;
                cumulative += equal;
                node.cumulative_reward = cumulative;
            }
        }
    }

    // The trajectory's query_embedding = first node's context embedding
    let query_embedding = nodes
        .first()
        .map(|n| n.context_embedding.clone())
        .unwrap_or_else(|| neural_routing_core::sentinel_vector(TOTAL_DIM, 0xCAFE_BABE));

    PendingTrajectory {
        trajectory: Trajectory {
            id: trajectory_id,
            session_id: session_id.to_string(),
            query_embedding,
            total_reward,
            step_count: nodes.len(),
            duration_ms,
            nodes,
            created_at: now,
        },
        tool_usages,
        touched_entities,
    }
}

async fn flush_batch(store: &Neo4jTrajectoryStore, batch: Vec<PendingTrajectory>) {
    let count = batch.len();
    tracing::debug!(count, "Flushing trajectory batch");

    // 1. Batch-store all trajectories + nodes via UNWIND (≤3 Cypher queries)
    let trajectories: Vec<&Trajectory> = batch.iter().map(|p| &p.trajectory).collect();
    let traj_refs: Vec<Trajectory> = trajectories.iter().map(|t| (*t).clone()).collect();

    match store.store_trajectories_batch(&traj_refs).await {
        Ok(stored) => {
            tracing::debug!(stored, "Batch-stored trajectories via UNWIND");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Batch store failed, falling back to sequential");
            // Fallback: store individually (best-effort)
            for pending in &batch {
                if let Err(e) = store.store_trajectory(&pending.trajectory).await {
                    tracing::warn!(
                        trajectory_id = %pending.trajectory.id,
                        error = %e,
                        "Failed to store trajectory (sequential fallback)"
                    );
                }
            }
        }
    }

    // 2. Link tool usages and touched entities (still per-node, could be batched later)
    for pending in &batch {
        for (node, (tools, entities)) in pending.trajectory.nodes.iter().zip(
            pending
                .tool_usages
                .iter()
                .zip(pending.touched_entities.iter()),
        ) {
            if !tools.is_empty() {
                if let Err(e) = store.link_tool_usages_batch(&node.id, tools).await {
                    tracing::warn!(
                        node_id = %node.id,
                        error = %e,
                        "Failed to link tool usages"
                    );
                }
            }

            if !entities.is_empty() {
                if let Err(e) = store.link_touched_entities_batch(&node.id, entities).await {
                    tracing::warn!(
                        node_id = %node.id,
                        error = %e,
                        "Failed to link touched entities"
                    );
                }
            }
        }

        tracing::debug!(
            trajectory_id = %pending.trajectory.id,
            nodes = pending.trajectory.nodes.len(),
            "Flushed trajectory"
        );
    }

    tracing::info!(count, "Trajectory batch flush complete");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    fn make_decision(session_id: &str, action: &str, ts_ms: u64) -> DecisionRecord {
        DecisionRecord {
            session_id: session_id.to_string(),
            context_embedding: vec![],
            action_type: action.to_string(),
            action_params: serde_json::json!({"query": "test"}),
            alternatives_count: 3,
            chosen_index: 0,
            confidence: 0.85,
            tool_usages: vec![ToolUsage {
                tool_name: "code".to_string(),
                action: "search".to_string(),
                params_hash: "abc123".to_string(),
                duration_ms: Some(15),
                success: true,
            }],
            touched_entities: vec![TouchedEntity {
                entity_type: "file".to_string(),
                entity_id: "src/main.rs".to_string(),
                access_mode: "read".to_string(),
                relevance: Some(0.9),
            }],
            timestamp_ms: ts_ms,
            query_embedding: vec![],
            node_features: vec![],
        }
    }

    #[test]
    fn test_record_decision_does_not_block() {
        // Verify that record_decision returns instantly even with a full channel
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            // We can't use a real Neo4jTrajectoryStore in unit tests,
            // so just test the channel mechanics.
            let (tx, _rx) = mpsc::channel::<CollectorEvent>(2);
            let enabled = Arc::new(std::sync::atomic::AtomicBool::new(true));

            let collector = TrajectoryCollector {
                tx,
                enabled,
                vector_builder: Arc::new(DecisionVectorBuilder::new()),
            };

            // These should never block
            collector.record_decision(make_decision("s1", "code.search", 0));
            collector.record_decision(make_decision("s1", "note.get_context", 100));
            // Channel capacity is 2, this 3rd one should be silently dropped
            collector.record_decision(make_decision("s1", "code.analyze_impact", 200));
        });
    }

    #[test]
    fn test_disabled_collector_skips() {
        let (tx, mut rx) = mpsc::channel::<CollectorEvent>(10);
        let enabled = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let collector = TrajectoryCollector {
            tx,
            enabled,
            vector_builder: Arc::new(DecisionVectorBuilder::new()),
        };

        collector.record_decision(make_decision("s1", "code.search", 0));

        // Channel should be empty since collection is disabled
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_toggle_enabled() {
        let (tx, _rx) = mpsc::channel::<CollectorEvent>(10);
        let enabled = Arc::new(std::sync::atomic::AtomicBool::new(true));

        let collector = TrajectoryCollector {
            tx,
            enabled,
            vector_builder: Arc::new(DecisionVectorBuilder::new()),
        };

        assert!(collector.is_enabled());
        collector.set_enabled(false);
        assert!(!collector.is_enabled());
        collector.set_enabled(true);
        assert!(collector.is_enabled());
    }

    #[test]
    fn test_build_trajectory() {
        let started = Utc::now();
        let builder = DecisionVectorBuilder::new();
        let buffer = SessionBuffer {
            decisions: vec![
                make_decision("s1", "code.search", 0),
                make_decision("s1", "note.get_context", 50),
                make_decision("s1", "code.analyze_impact", 150),
            ],
            started_at: started,
            last_decision_at: Instant::now(),
        };

        let pending = build_trajectory("s1", &buffer, 0.85, &builder);

        assert_eq!(pending.trajectory.session_id, "s1");
        assert_eq!(pending.trajectory.step_count, 3);
        assert_eq!(pending.trajectory.total_reward, 0.85);
        assert_eq!(pending.trajectory.nodes.len(), 3);

        // Check ordering
        assert_eq!(pending.trajectory.nodes[0].order, 0);
        assert_eq!(pending.trajectory.nodes[1].order, 1);
        assert_eq!(pending.trajectory.nodes[2].order, 2);

        // Check delta_ms
        assert_eq!(pending.trajectory.nodes[0].delta_ms, 0);
        assert_eq!(pending.trajectory.nodes[1].delta_ms, 50);
        assert_eq!(pending.trajectory.nodes[2].delta_ms, 100);

        // Check tool usages preserved
        assert_eq!(pending.tool_usages.len(), 3);
        assert_eq!(pending.tool_usages[0].len(), 1);
        assert_eq!(pending.tool_usages[0][0].tool_name, "code");

        // Check touched entities preserved
        assert_eq!(pending.touched_entities.len(), 3);
        assert_eq!(pending.touched_entities[0].len(), 1);
        assert_eq!(pending.touched_entities[0][0].entity_type, "file");
    }

    #[test]
    fn bench_record_decision_latency_under_1ms() {
        // Benchmark: record_decision must complete in <1ms (fire-and-forget via try_send).
        // We measure 1000 iterations and assert p99 < 1ms.
        let (tx, _rx) = mpsc::channel::<CollectorEvent>(1000);
        let enabled = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let collector = TrajectoryCollector {
            tx,
            enabled,
            vector_builder: Arc::new(DecisionVectorBuilder::new()),
        };

        let mut durations = Vec::with_capacity(1000);

        for i in 0..1000 {
            let decision = make_decision("bench", "code.search", i);
            let start = std::time::Instant::now();
            collector.record_decision(decision);
            durations.push(start.elapsed());
        }

        durations.sort();
        let p50 = durations[499];
        let p99 = durations[989];
        let max = durations[999];

        eprintln!(
            "record_decision latency: p50={:?}, p99={:?}, max={:?}",
            p50, p99, max
        );

        // p99 must be under 1ms — typically <1μs (just a channel try_send)
        assert!(
            p99 < std::time::Duration::from_millis(1),
            "p99 latency {:?} exceeds 1ms budget",
            p99
        );
    }

    #[test]
    fn bench_record_decision_disabled_latency() {
        // When collection is disabled, record_decision should be near-zero (just an atomic load).
        let (tx, _rx) = mpsc::channel::<CollectorEvent>(10);
        let enabled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let collector = TrajectoryCollector {
            tx,
            enabled,
            vector_builder: Arc::new(DecisionVectorBuilder::new()),
        };

        let mut durations = Vec::with_capacity(1000);

        for i in 0..1000 {
            let decision = make_decision("bench-off", "code.search", i);
            let start = std::time::Instant::now();
            collector.record_decision(decision);
            durations.push(start.elapsed());
        }

        durations.sort();
        let p99 = durations[989];

        eprintln!("record_decision (disabled) p99={:?}", p99);

        // Should be essentially free
        assert!(
            p99 < std::time::Duration::from_micros(100),
            "disabled p99 latency {:?} is too high",
            p99
        );
    }

    #[test]
    fn test_build_trajectory_empty() {
        let builder = DecisionVectorBuilder::new();
        let buffer = SessionBuffer {
            decisions: vec![],
            started_at: Utc::now(),
            last_decision_at: Instant::now(),
        };

        let pending = build_trajectory("empty", &buffer, 0.0, &builder);
        assert_eq!(pending.trajectory.step_count, 0);
        assert_eq!(pending.trajectory.nodes.len(), 0);
        // Empty trajectory should still have a valid (sentinel) query_embedding
        assert_eq!(pending.trajectory.query_embedding.len(), 256);
        let norm: f32 = pending
            .trajectory
            .query_embedding
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Empty trajectory query_embedding should be L2-normalized sentinel"
        );
    }

    #[test]
    fn test_three_decisions_with_different_tools_and_entities() {
        // Simulates 3 decisions with different tools — verifies that USED_TOOL
        // and TOUCHED_ENTITY relations are correctly captured per node.
        let started = Utc::now();
        let builder = DecisionVectorBuilder::new();
        let decisions = vec![
            DecisionRecord {
                session_id: "s2".into(),
                context_embedding: vec![],
                action_type: "code.search".into(),
                action_params: serde_json::json!({"query": "TrajectoryStore"}),
                alternatives_count: 5,
                chosen_index: 0,
                confidence: 0.9,
                tool_usages: vec![ToolUsage {
                    tool_name: "code".into(),
                    action: "search".into(),
                    params_hash: "hash1".into(),
                    duration_ms: Some(12),
                    success: true,
                }],
                touched_entities: vec![
                    TouchedEntity {
                        entity_type: "file".into(),
                        entity_id: "src/store.rs".into(),
                        access_mode: "search_hit".into(),
                        relevance: Some(0.95),
                    },
                    TouchedEntity {
                        entity_type: "function".into(),
                        entity_id: "store_trajectory".into(),
                        access_mode: "search_hit".into(),
                        relevance: Some(0.88),
                    },
                ],
                timestamp_ms: 0,
                query_embedding: vec![],
                node_features: vec![],
            },
            DecisionRecord {
                session_id: "s2".into(),
                context_embedding: vec![],
                action_type: "note.get_context".into(),
                action_params: serde_json::json!({"entity_type": "file", "entity_id": "src/store.rs"}),
                alternatives_count: 2,
                chosen_index: 0,
                confidence: 0.75,
                tool_usages: vec![ToolUsage {
                    tool_name: "note".into(),
                    action: "get_context".into(),
                    params_hash: "hash2".into(),
                    duration_ms: Some(8),
                    success: true,
                }],
                touched_entities: vec![TouchedEntity {
                    entity_type: "note".into(),
                    entity_id: "note-uuid-123".into(),
                    access_mode: "context_load".into(),
                    relevance: Some(0.7),
                }],
                timestamp_ms: 50,
                query_embedding: vec![],
                node_features: vec![],
            },
            DecisionRecord {
                session_id: "s2".into(),
                context_embedding: vec![],
                action_type: "code.analyze_impact".into(),
                action_params: serde_json::json!({"target": "store_trajectory"}),
                alternatives_count: 3,
                chosen_index: 1,
                confidence: 0.82,
                tool_usages: vec![ToolUsage {
                    tool_name: "code".into(),
                    action: "analyze_impact".into(),
                    params_hash: "hash3".into(),
                    duration_ms: Some(25),
                    success: true,
                }],
                touched_entities: vec![
                    TouchedEntity {
                        entity_type: "file".into(),
                        entity_id: "src/store.rs".into(),
                        access_mode: "write".into(),
                        relevance: Some(1.0),
                    },
                    TouchedEntity {
                        entity_type: "file".into(),
                        entity_id: "src/models.rs".into(),
                        access_mode: "read".into(),
                        relevance: Some(0.6),
                    },
                ],
                timestamp_ms: 200,
                query_embedding: vec![],
                node_features: vec![],
            },
        ];

        let buffer = SessionBuffer {
            decisions,
            started_at: started,
            last_decision_at: Instant::now(),
        };

        let pending = build_trajectory("s2", &buffer, 0.78, &builder);

        // 3 nodes, 3 tool usages, 3 entity groups
        assert_eq!(pending.trajectory.nodes.len(), 3);
        assert_eq!(pending.tool_usages.len(), 3);
        assert_eq!(pending.touched_entities.len(), 3);

        // Node 0: code.search → 1 tool, 2 entities
        assert_eq!(pending.tool_usages[0].len(), 1);
        assert_eq!(pending.tool_usages[0][0].tool_name, "code");
        assert_eq!(pending.tool_usages[0][0].action, "search");
        assert_eq!(pending.touched_entities[0].len(), 2);
        assert_eq!(pending.touched_entities[0][0].entity_type, "file");
        assert_eq!(pending.touched_entities[0][1].entity_type, "function");

        // Node 1: note.get_context → 1 tool, 1 entity
        assert_eq!(pending.tool_usages[1].len(), 1);
        assert_eq!(pending.tool_usages[1][0].tool_name, "note");
        assert_eq!(pending.touched_entities[1].len(), 1);
        assert_eq!(pending.touched_entities[1][0].entity_type, "note");

        // Node 2: code.analyze_impact → 1 tool, 2 entities
        assert_eq!(pending.tool_usages[2].len(), 1);
        assert_eq!(pending.tool_usages[2][0].tool_name, "code");
        assert_eq!(pending.tool_usages[2][0].action, "analyze_impact");
        assert_eq!(pending.touched_entities[2].len(), 2);

        // Verify action types on trajectory nodes
        assert_eq!(pending.trajectory.nodes[0].action_type, "code.search");
        assert_eq!(pending.trajectory.nodes[1].action_type, "note.get_context");
        assert_eq!(
            pending.trajectory.nodes[2].action_type,
            "code.analyze_impact"
        );

        // Verify delta_ms
        assert_eq!(pending.trajectory.nodes[0].delta_ms, 0);
        assert_eq!(pending.trajectory.nodes[1].delta_ms, 50);
        assert_eq!(pending.trajectory.nodes[2].delta_ms, 150);
    }

    #[test]
    fn test_build_trajectory_computes_embeddings() {
        // Verify that the VectorBuilder fills context_embedding on each node
        let builder = DecisionVectorBuilder::new();
        let buffer = SessionBuffer {
            decisions: vec![
                make_decision("embed-test", "code.search", 0),
                make_decision("embed-test", "note.get_context", 100),
            ],
            started_at: Utc::now(),
            last_decision_at: Instant::now(),
        };

        let pending = build_trajectory("embed-test", &buffer, 0.9, &builder);

        // Each node should have a 256d context embedding (computed by the builder)
        for (i, node) in pending.trajectory.nodes.iter().enumerate() {
            assert_eq!(
                node.context_embedding.len(),
                256,
                "Node {} should have 256d embedding",
                i
            );
            let all_zero = node.context_embedding.iter().all(|&x| x.abs() < 1e-10);
            assert!(
                !all_zero,
                "Node {} embedding must not be all zeros (sentinel or computed)",
                i
            );
            let norm: f32 = node
                .context_embedding
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "Node {} embedding should be L2-normalized, got norm={}",
                i,
                norm
            );
        }

        // The trajectory's query_embedding should equal the first node's embedding
        assert_eq!(
            pending.trajectory.query_embedding, pending.trajectory.nodes[0].context_embedding,
            "Trajectory query_embedding should match first node's embedding"
        );
    }

    #[test]
    fn test_build_trajectory_preserves_precomputed_embedding() {
        // If a DecisionRecord already has a context_embedding, it should NOT be overwritten
        let builder = DecisionVectorBuilder::new();
        let precomputed = vec![0.5_f32; 256]; // not normalized, but recognizable
        let mut decision = make_decision("precomp", "code.search", 0);
        decision.context_embedding = precomputed.clone();

        let buffer = SessionBuffer {
            decisions: vec![decision],
            started_at: Utc::now(),
            last_decision_at: Instant::now(),
        };

        let pending = build_trajectory("precomp", &buffer, 1.0, &builder);
        assert_eq!(
            pending.trajectory.nodes[0].context_embedding, precomputed,
            "Precomputed embedding should be preserved"
        );
    }

    #[test]
    fn test_build_trajectory_history_accumulates() {
        // With 3 decisions, node 2 should get a different embedding than node 0
        // because it has accumulated history from nodes 0 and 1
        let builder = DecisionVectorBuilder::new();
        let buffer = SessionBuffer {
            decisions: vec![
                make_decision("hist", "code.search", 0),
                make_decision("hist", "note.create", 50),
                make_decision("hist", "plan.create", 100),
            ],
            started_at: Utc::now(),
            last_decision_at: Instant::now(),
        };

        let pending = build_trajectory("hist", &buffer, 0.8, &builder);

        // Node 0 and node 2 should differ (different tool + accumulated history)
        let cos: f32 = pending.trajectory.nodes[0]
            .context_embedding
            .iter()
            .zip(pending.trajectory.nodes[2].context_embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            cos < 0.999,
            "Nodes with different tools/history should differ, cosine={}",
            cos
        );
    }

    #[test]
    fn test_build_trajectory_fills_rewards() {
        let builder = DecisionVectorBuilder::new();
        let buffer = SessionBuffer {
            decisions: vec![
                make_decision("reward-test", "code.search", 0),
                make_decision("reward-test", "note.get_context", 50),
                make_decision("reward-test", "code.analyze_impact", 100),
            ],
            started_at: Utc::now(),
            last_decision_at: Instant::now(),
        };

        let pending = build_trajectory("reward-test", &buffer, 0.9, &builder);

        // All nodes should have non-zero local_reward
        for (i, node) in pending.trajectory.nodes.iter().enumerate() {
            assert!(
                node.local_reward > 0.0,
                "Node {} should have positive local_reward, got {}",
                i,
                node.local_reward
            );
        }

        // local_rewards should sum to approximately total_reward
        let sum: f64 = pending
            .trajectory
            .nodes
            .iter()
            .map(|n| n.local_reward)
            .sum();
        assert!(
            (sum - 0.9).abs() < 0.05,
            "local_rewards should sum to ~0.9, got {}",
            sum
        );

        // cumulative_reward should be monotonically increasing
        for i in 1..pending.trajectory.nodes.len() {
            assert!(
                pending.trajectory.nodes[i].cumulative_reward
                    >= pending.trajectory.nodes[i - 1].cumulative_reward,
                "cumulative_reward must be monotonically increasing"
            );
        }

        // Last node's cumulative_reward should equal total
        let last = pending.trajectory.nodes.last().unwrap();
        assert!(
            (last.cumulative_reward - 0.9).abs() < 0.05,
            "Last cumulative_reward should be ~0.9, got {}",
            last.cumulative_reward
        );
    }

    #[tokio::test]
    async fn test_collector_event_flow() {
        // Verify events flow through the channel correctly
        let (tx, mut rx) = mpsc::channel::<CollectorEvent>(100);
        let enabled = Arc::new(std::sync::atomic::AtomicBool::new(true));

        let collector = TrajectoryCollector {
            tx,
            enabled,
            vector_builder: Arc::new(DecisionVectorBuilder::new()),
        };

        // Send 3 decisions for session "flow-test"
        collector.record_decision(make_decision("flow-test", "code.search", 0));
        collector.record_decision(make_decision("flow-test", "note.get_context", 50));
        collector.record_decision(make_decision("flow-test", "code.analyze_impact", 150));
        collector.end_session("flow-test".to_string(), 0.9);

        // Verify 4 events in channel (3 decisions + 1 end_session)
        let mut event_count = 0;
        while let Ok(event) = rx.try_recv() {
            match event {
                CollectorEvent::Decision(d) => {
                    assert_eq!(d.session_id, "flow-test");
                    event_count += 1;
                }
                CollectorEvent::EndSession {
                    session_id,
                    total_reward,
                } => {
                    assert_eq!(session_id, "flow-test");
                    assert!((total_reward - 0.9).abs() < 1e-10);
                    event_count += 1;
                }
                _ => {}
            }
        }
        assert_eq!(event_count, 4);
    }

    // ── Integration tests — feedback loop ────────────────────────────────

    #[test]
    fn test_feedback_loop_two_sessions_with_reward() {
        // Full feedback loop: record decisions across 2 sessions, end both,
        // verify trajectories are built with correct reward decomposition.
        let builder = DecisionVectorBuilder::new();
        let computer = SessionRewardComputer::default();

        // Session 1: 3 successful decisions, high confidence
        let buf1 = SessionBuffer {
            decisions: vec![
                make_decision("s1", "code.search", 0),
                make_decision("s1", "note.get_context", 50),
                make_decision("s1", "code.analyze_impact", 150),
            ],
            started_at: Utc::now() - chrono::Duration::seconds(120),
            last_decision_at: Instant::now(),
        };

        // Session 2: 2 decisions with mixed confidence
        let mut d1 = make_decision("s2", "code.search", 0);
        d1.confidence = 0.3;
        d1.tool_usages[0].success = false;
        let d2 = make_decision("s2", "note.create", 100);
        let buf2 = SessionBuffer {
            decisions: vec![d1, d2],
            started_at: Utc::now() - chrono::Duration::seconds(30),
            last_decision_at: Instant::now(),
        };

        // Compute rewards
        let reward1 = compute_auto_reward(&buf1, &computer);
        let reward2 = compute_auto_reward(&buf2, &computer);

        // Session 1 (all success, high confidence) should have higher reward
        assert!(
            reward1 > reward2,
            "All-success session should have higher reward: {} vs {}",
            reward1,
            reward2
        );
        assert!(reward1 > 0.0 && reward1 <= 1.0);
        assert!(reward2 > 0.0 && reward2 <= 1.0);

        // Build trajectories
        let traj1 = build_trajectory("s1", &buf1, reward1, &builder);
        let traj2 = build_trajectory("s2", &buf2, reward2, &builder);

        // Verify trajectory 1
        assert_eq!(traj1.trajectory.nodes.len(), 3);
        assert_eq!(traj1.trajectory.step_count, 3);
        assert!((traj1.trajectory.total_reward - reward1).abs() < 1e-10);

        // Reward decomposition: sum of local_rewards ≈ total_reward
        let sum1: f64 = traj1.trajectory.nodes.iter().map(|n| n.local_reward).sum();
        assert!(
            (sum1 - reward1).abs() < 0.05,
            "local_rewards sum should ≈ total: {} vs {}",
            sum1,
            reward1
        );

        // Verify trajectory 2
        assert_eq!(traj2.trajectory.nodes.len(), 2);
        let sum2: f64 = traj2.trajectory.nodes.iter().map(|n| n.local_reward).sum();
        assert!(
            (sum2 - reward2).abs() < 0.05,
            "local_rewards sum should ≈ total: {} vs {}",
            sum2,
            reward2
        );

        // Tool usages preserved
        assert_eq!(traj1.tool_usages.len(), 3);
        assert_eq!(traj1.tool_usages[0].len(), 1);
        assert_eq!(traj1.tool_usages[0][0].tool_name, "code");
        assert_eq!(traj2.tool_usages.len(), 2);

        // Touched entities preserved
        assert_eq!(traj1.touched_entities.len(), 3);
        assert_eq!(traj2.touched_entities.len(), 2);
    }

    #[test]
    fn test_feedback_loop_auto_reward_empty_session() {
        let computer = SessionRewardComputer::default();
        let buf = SessionBuffer {
            decisions: vec![],
            started_at: Utc::now(),
            last_decision_at: Instant::now(),
        };
        let reward = compute_auto_reward(&buf, &computer);
        assert_eq!(reward, 0.0, "Empty session should have 0 reward");
    }

    #[test]
    fn test_feedback_loop_auto_reward_varies_with_success_rate() {
        let computer = SessionRewardComputer::default();

        // All success
        let buf_ok = SessionBuffer {
            decisions: vec![
                make_decision("ok", "code.search", 0),
                make_decision("ok", "code.search", 50),
            ],
            started_at: Utc::now() - chrono::Duration::seconds(300),
            last_decision_at: Instant::now(),
        };

        // All failure
        let mut d_fail = make_decision("fail", "code.search", 0);
        d_fail.tool_usages[0].success = false;
        d_fail.confidence = 0.1;
        let mut d_fail2 = make_decision("fail", "code.search", 50);
        d_fail2.tool_usages[0].success = false;
        d_fail2.confidence = 0.1;
        let buf_fail = SessionBuffer {
            decisions: vec![d_fail, d_fail2],
            started_at: Utc::now() - chrono::Duration::seconds(300),
            last_decision_at: Instant::now(),
        };

        let reward_ok = compute_auto_reward(&buf_ok, &computer);
        let reward_fail = compute_auto_reward(&buf_fail, &computer);

        assert!(
            reward_ok > reward_fail,
            "All-success should beat all-failure: {} vs {}",
            reward_ok,
            reward_fail
        );
    }

    #[tokio::test]
    async fn test_feedback_loop_event_flow_with_reward() {
        // Full event flow: send decisions for 2 sessions, end them with
        // computed rewards, verify events pass through channel correctly.
        let (tx, mut rx) = mpsc::channel::<CollectorEvent>(100);
        let enabled = Arc::new(std::sync::atomic::AtomicBool::new(true));

        let collector = TrajectoryCollector {
            tx,
            enabled,
            vector_builder: Arc::new(DecisionVectorBuilder::new()),
        };

        // Session A: 3 decisions
        collector.record_decision(make_decision("session-a", "code.search", 0));
        collector.record_decision(make_decision("session-a", "note.get_context", 50));
        collector.record_decision(make_decision("session-a", "plan.create", 120));

        // Session B: 2 decisions
        collector.record_decision(make_decision("session-b", "code.search", 0));
        collector.record_decision(make_decision("session-b", "task.create", 80));

        // End both sessions with rewards
        collector.end_session("session-a".to_string(), 0.85);
        collector.end_session("session-b".to_string(), 0.6);

        // Verify 7 events: 3 + 2 decisions + 2 end_session
        let mut decisions_a = 0;
        let mut decisions_b = 0;
        let mut ends = 0;
        while let Ok(event) = rx.try_recv() {
            match event {
                CollectorEvent::Decision(d) if d.session_id == "session-a" => decisions_a += 1,
                CollectorEvent::Decision(d) if d.session_id == "session-b" => decisions_b += 1,
                CollectorEvent::EndSession {
                    session_id,
                    total_reward,
                } => {
                    match session_id.as_str() {
                        "session-a" => assert!((total_reward - 0.85).abs() < 1e-10),
                        "session-b" => assert!((total_reward - 0.6).abs() < 1e-10),
                        _ => panic!("Unexpected session: {}", session_id),
                    }
                    ends += 1;
                }
                _ => {}
            }
        }
        assert_eq!(decisions_a, 3);
        assert_eq!(decisions_b, 2);
        assert_eq!(ends, 2);
    }
}
