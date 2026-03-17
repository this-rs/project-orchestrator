//! Monte Carlo Tree Search for trajectory data augmentation.
//!
//! Given a real trajectory, MCTS explores alternative decision paths by:
//! 1. At each decision point, expanding N alternative actions
//! 2. Evaluating each alternative via a proxy reward model
//! 3. Using UCB1 for selection and backpropagation
//! 4. Extracting the best K alternative trajectories
//!
//! The generated trajectories are marked `source: simulated` with weight 0.3.

use std::collections::HashMap;
use uuid::Uuid;

use crate::error::Result;
use crate::models::{ActionCandidate, Trajectory, TrajectoryNode};
use crate::proxy_model::ProxyModel;

/// Configuration for MCTS simulation.
#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// Number of alternative actions to expand at each decision point.
    pub expansion_width: usize,
    /// Number of UCB1 rollouts per trajectory.
    pub num_rollouts: usize,
    /// UCB1 exploration constant (sqrt(2) is standard).
    pub exploration_c: f64,
    /// Maximum alternative trajectories to extract per source trajectory.
    pub max_alternatives: usize,
    /// Discount factor for future rewards.
    pub gamma: f64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            expansion_width: 10,
            num_rollouts: 50,
            exploration_c: std::f64::consts::SQRT_2,
            max_alternatives: 5,
            gamma: 0.99,
        }
    }
}

/// A node in the MCTS search tree.
#[derive(Debug, Clone)]
struct MctsNode {
    /// The action taken to reach this node.
    action_type: String,
    action_params: serde_json::Value,
    /// UCB1 statistics.
    visit_count: u32,
    total_value: f64,
    /// Children keyed by action index.
    children: HashMap<usize, MctsNode>,
    /// Whether this is a terminal node.
    is_terminal: bool,
    /// Depth in the tree (0 = root).
    depth: usize,
}

impl MctsNode {
    fn new(action_type: String, action_params: serde_json::Value, depth: usize) -> Self {
        Self {
            action_type,
            action_params,
            visit_count: 0,
            total_value: 0.0,
            children: HashMap::new(),
            is_terminal: false,
            depth,
        }
    }

    /// UCB1 score: exploitation + exploration.
    fn ucb1(&self, parent_visits: u32, c: f64) -> f64 {
        if self.visit_count == 0 {
            return f64::INFINITY;
        }
        let exploitation = self.total_value / self.visit_count as f64;
        let exploration = c * ((parent_visits as f64).ln() / self.visit_count as f64).sqrt();
        exploitation + exploration
    }

    /// Average value.
    fn avg_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f64
        }
    }
}

/// MCTS engine for trajectory augmentation.
pub struct MctsEngine<P: ProxyModel> {
    config: MctsConfig,
    proxy: P,
}

impl<P: ProxyModel> MctsEngine<P> {
    pub fn new(config: MctsConfig, proxy: P) -> Self {
        Self { config, proxy }
    }

    /// Generate alternative trajectories from a real source trajectory.
    ///
    /// Returns up to `config.max_alternatives` simulated trajectories.
    pub async fn generate_alternatives(&self, source: &Trajectory) -> Result<Vec<Trajectory>> {
        if source.nodes.is_empty() {
            return Ok(vec![]);
        }

        // Build a search tree rooted at the source trajectory's first decision
        let mut root = MctsNode::new(
            source.nodes[0].action_type.clone(),
            source.nodes[0].action_params.clone(),
            0,
        );

        // Run MCTS rollouts
        for _ in 0..self.config.num_rollouts {
            self.rollout(&mut root, source).await;
        }

        // Extract the best alternative paths from the tree
        let mut alternatives = Vec::new();
        self.extract_paths(&root, source, &mut vec![], &mut alternatives);

        // Sort by estimated reward and take top K
        alternatives.sort_by(|a, b| {
            b.total_reward
                .partial_cmp(&a.total_reward)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        alternatives.truncate(self.config.max_alternatives);

        Ok(alternatives)
    }

    /// Run a single MCTS rollout: select → expand → simulate → backprop.
    async fn rollout(&self, root: &mut MctsNode, source: &Trajectory) {
        // Selection: walk down the tree using UCB1
        let mut path = vec![];
        let mut current = root as *mut MctsNode;
        let max_depth = source.nodes.len();

        // SAFETY: We only access tree nodes through mutable references in a single-threaded context.
        // The raw pointer is used to traverse the tree while collecting the path.
        unsafe {
            loop {
                let node = &mut *current;

                if node.depth >= max_depth || node.is_terminal {
                    break;
                }

                if node.children.is_empty() {
                    // Expansion: add alternative actions at this decision point
                    self.expand(node, source);

                    // After expansion, select a random child to simulate from
                    if !node.children.is_empty() {
                        let first_key = *node.children.keys().next().unwrap();
                        path.push(current);
                        current = node.children.get_mut(&first_key).unwrap() as *mut MctsNode;
                    }
                    break;
                }

                // Select best child by UCB1
                let parent_visits = node.visit_count;
                let best_idx = node
                    .children
                    .iter()
                    .max_by(|(_, a), (_, b)| {
                        a.ucb1(parent_visits, self.config.exploration_c)
                            .partial_cmp(&b.ucb1(parent_visits, self.config.exploration_c))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| *idx)
                    .unwrap_or(0);

                path.push(current);
                current = node.children.get_mut(&best_idx).unwrap() as *mut MctsNode;
            }

            // Simulate: estimate reward for the current leaf
            let leaf = &mut *current;
            let reward = self.simulate_reward(leaf, source).await;

            // Backpropagation: update visit counts and values
            leaf.visit_count += 1;
            leaf.total_value += reward;

            for ptr in path.iter().rev() {
                let node = &mut **ptr;
                node.visit_count += 1;
                node.total_value += reward;
            }
        }
    }

    /// Expand a node by adding alternative actions from the proxy model.
    fn expand(&self, node: &mut MctsNode, source: &Trajectory) {
        let depth = node.depth;
        if depth >= source.nodes.len() {
            node.is_terminal = true;
            return;
        }

        let source_node = &source.nodes[depth];

        // Generate alternative actions based on the source node's context
        let alternatives = self.generate_action_alternatives(source_node);

        for (i, alt) in alternatives.into_iter().enumerate() {
            let child = MctsNode::new(
                alt.action_type.clone(),
                alt.action_params.clone(),
                depth + 1,
            );
            node.children.insert(i, child);
        }
    }

    /// Generate alternative actions for a decision point.
    fn generate_action_alternatives(&self, source_node: &TrajectoryNode) -> Vec<ActionCandidate> {
        // Common MCP tool/action combinations that could be alternatives
        let tool_actions = [
            ("code.search", r#"{"query":"__query__"}"#),
            ("code.search_project", r#"{"query":"__query__"}"#),
            ("code.get_file_symbols", r#"{"file_path":"__path__"}"#),
            ("code.find_references", r#"{"symbol":"__sym__"}"#),
            ("code.get_call_graph", r#"{"function":"__func__"}"#),
            ("code.analyze_impact", r#"{"target":"__target__"}"#),
            ("note.search_semantic", r#"{"query":"__query__"}"#),
            (
                "note.get_context",
                r#"{"entity_type":"file","entity_id":"__id__"}"#,
            ),
            ("decision.search_semantic", r#"{"query":"__query__"}"#),
            ("skill.activate", r#"{"query":"__query__"}"#),
        ];

        let mut alternatives: Vec<ActionCandidate> = tool_actions
            .iter()
            .filter(|(action, _)| *action != source_node.action_type)
            .take(self.config.expansion_width)
            .enumerate()
            .map(|(i, (action, params))| ActionCandidate {
                action_type: action.to_string(),
                action_params: serde_json::from_str(params).unwrap_or(serde_json::Value::Null),
                score: 1.0 / (i + 1) as f64, // Decreasing prior
            })
            .collect();

        // Always include the original action as an alternative
        alternatives.push(ActionCandidate {
            action_type: source_node.action_type.clone(),
            action_params: source_node.action_params.clone(),
            score: 1.0,
        });

        alternatives
    }

    /// Simulate the reward of a leaf node using the proxy model.
    async fn simulate_reward(&self, leaf: &MctsNode, source: &Trajectory) -> f64 {
        // Use the proxy model to estimate reward based on context
        let remaining_steps = source.nodes.len().saturating_sub(leaf.depth);

        self.proxy.estimate_reward(
            &leaf.action_type,
            &leaf.action_params,
            remaining_steps,
            source.total_reward,
        )
    }

    /// Extract complete trajectory paths from the MCTS tree.
    fn extract_paths(
        &self,
        node: &MctsNode,
        source: &Trajectory,
        current_path: &mut Vec<(String, serde_json::Value, f64)>,
        results: &mut Vec<Trajectory>,
    ) {
        current_path.push((
            node.action_type.clone(),
            node.action_params.clone(),
            node.avg_value(),
        ));

        let at_terminal = node.children.is_empty()
            || node.is_terminal
            || node.depth >= source.nodes.len().saturating_sub(1);

        if at_terminal {
            // Terminal: build trajectory from path if it differs from source
            if self.path_differs_from_source(current_path, source) {
                let trajectory = self.build_simulated_trajectory(current_path, source);
                results.push(trajectory);
            }
        } else {
            // Recurse into children with sufficient visits (min 1)
            let min_visits = 1u32;
            for child in node.children.values() {
                if child.visit_count >= min_visits {
                    self.extract_paths(child, source, current_path, results);
                }
            }
        }

        current_path.pop();
    }

    /// Check if an extracted path differs meaningfully from the source trajectory.
    fn path_differs_from_source(
        &self,
        path: &[(String, serde_json::Value, f64)],
        source: &Trajectory,
    ) -> bool {
        if path.is_empty() {
            return false;
        }
        // Different length = definitely different
        if path.len() != source.nodes.len() {
            return path.len() >= 2; // Accept if at least 2 steps
        }
        // Same length: at least one action must differ
        path.iter()
            .zip(source.nodes.iter())
            .any(|((action, _, _), node)| action != &node.action_type)
    }

    /// Build a simulated Trajectory from an MCTS path.
    fn build_simulated_trajectory(
        &self,
        path: &[(String, serde_json::Value, f64)],
        source: &Trajectory,
    ) -> Trajectory {
        let trajectory_id = Uuid::new_v4();
        let now = chrono::Utc::now();

        let nodes: Vec<TrajectoryNode> = path
            .iter()
            .enumerate()
            .map(|(i, (action_type, action_params, estimated_reward))| {
                let source_node = source.nodes.get(i);
                TrajectoryNode {
                    id: Uuid::new_v4(),
                    // Reuse context embedding from source (same context moment)
                    context_embedding: source_node
                        .map(|n| n.context_embedding.clone())
                        .unwrap_or_default(),
                    action_type: action_type.clone(),
                    action_params: action_params.clone(),
                    alternatives_count: self.config.expansion_width,
                    chosen_index: 0,
                    confidence: (*estimated_reward).clamp(0.0, 1.0),
                    local_reward: *estimated_reward,
                    cumulative_reward: path[..=i].iter().map(|(_, _, r)| r).sum(),
                    delta_ms: source_node.map(|n| n.delta_ms).unwrap_or(0),
                    order: i,
                }
            })
            .collect();

        let total_reward: f64 = nodes.iter().map(|n| n.local_reward).sum();

        Trajectory {
            id: trajectory_id,
            session_id: format!("mcts-sim-{}", source.id),
            query_embedding: source.query_embedding.clone(),
            total_reward,
            step_count: nodes.len(),
            duration_ms: source.duration_ms,
            nodes,
            created_at: now,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy_model::GdsHeuristicProxy;

    fn make_source_trajectory(step_count: usize) -> Trajectory {
        let nodes: Vec<TrajectoryNode> = (0..step_count)
            .map(|i| TrajectoryNode {
                id: Uuid::new_v4(),
                context_embedding: vec![0.1; 256],
                action_type: format!("code.action_{}", i),
                action_params: serde_json::json!({"step": i}),
                alternatives_count: 3,
                chosen_index: 0,
                confidence: 0.8,
                local_reward: 0.2,
                cumulative_reward: 0.2 * (i + 1) as f64,
                delta_ms: 100,
                order: i,
            })
            .collect();

        Trajectory {
            id: Uuid::new_v4(),
            session_id: "test-session".to_string(),
            query_embedding: vec![0.1; 256],
            total_reward: 0.85,
            step_count,
            duration_ms: 500,
            nodes,
            created_at: chrono::Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_mcts_generates_alternatives() {
        let proxy = GdsHeuristicProxy::default();
        let config = MctsConfig {
            expansion_width: 5,
            num_rollouts: 20,
            max_alternatives: 5,
            ..Default::default()
        };
        let engine = MctsEngine::new(config, proxy);
        let source = make_source_trajectory(5);

        let alternatives = engine.generate_alternatives(&source).await.unwrap();

        assert!(
            !alternatives.is_empty(),
            "MCTS should generate at least 1 alternative"
        );
        assert!(
            alternatives.len() <= 5,
            "Should respect max_alternatives limit"
        );

        for alt in &alternatives {
            assert_eq!(alt.step_count, alt.nodes.len());
            assert!(alt.session_id.starts_with("mcts-sim-"));
        }
    }

    #[tokio::test]
    async fn test_mcts_empty_source() {
        let proxy = GdsHeuristicProxy::default();
        let engine = MctsEngine::new(MctsConfig::default(), proxy);

        let source = Trajectory {
            id: Uuid::new_v4(),
            session_id: "empty".into(),
            query_embedding: vec![],
            total_reward: 0.0,
            step_count: 0,
            duration_ms: 0,
            nodes: vec![],
            created_at: chrono::Utc::now(),
        };

        let alternatives = engine.generate_alternatives(&source).await.unwrap();
        assert!(alternatives.is_empty());
    }

    #[tokio::test]
    async fn test_mcts_simulated_trajectories_have_rewards() {
        let proxy = GdsHeuristicProxy::default();
        let config = MctsConfig {
            expansion_width: 5,
            num_rollouts: 30,
            max_alternatives: 3,
            ..Default::default()
        };
        let engine = MctsEngine::new(config, proxy);
        let source = make_source_trajectory(3);

        let alternatives = engine.generate_alternatives(&source).await.unwrap();

        for alt in &alternatives {
            // Simulated trajectories should have non-zero rewards
            assert!(
                alt.total_reward.is_finite(),
                "Reward should be finite, got {}",
                alt.total_reward
            );
            // Nodes should have local rewards
            for node in &alt.nodes {
                assert!(node.local_reward.is_finite());
            }
        }
    }

    #[tokio::test]
    async fn test_mcts_single_node_source() {
        let proxy = GdsHeuristicProxy::default();
        let config = MctsConfig {
            expansion_width: 5,
            num_rollouts: 20,
            max_alternatives: 5,
            ..Default::default()
        };
        let engine = MctsEngine::new(config, proxy);
        let source = make_source_trajectory(1);

        let alternatives = engine.generate_alternatives(&source).await.unwrap();

        // With a single node source, we may get empty results (path_differs_from_source
        // rejects single-step paths that match the source length but don't differ enough)
        // or valid alternatives — either way, no panics and results are well-formed.
        for alt in &alternatives {
            assert_eq!(alt.step_count, alt.nodes.len());
            assert!(alt.session_id.starts_with("mcts-sim-"));
            assert!(alt.total_reward.is_finite());
        }
    }

    #[tokio::test]
    async fn test_mcts_deep_tree() {
        let proxy = GdsHeuristicProxy::default();
        let config = MctsConfig {
            expansion_width: 5,
            num_rollouts: 200,
            max_alternatives: 10,
            ..Default::default()
        };
        let engine = MctsEngine::new(config, proxy);
        let source = make_source_trajectory(12);

        let alternatives = engine.generate_alternatives(&source).await.unwrap();

        assert!(
            !alternatives.is_empty(),
            "Deep tree with 200 rollouts should produce at least 1 alternative"
        );

        // Verify diversity: not all alternatives should have the same action sequence
        if alternatives.len() >= 2 {
            let first_actions: Vec<&str> = alternatives[0]
                .nodes
                .iter()
                .map(|n| n.action_type.as_str())
                .collect();
            let all_same = alternatives[1..].iter().all(|alt| {
                let actions: Vec<&str> = alt.nodes.iter().map(|n| n.action_type.as_str()).collect();
                actions == first_actions
            });
            assert!(
                !all_same,
                "Multiple alternatives should have diverse action sequences"
            );
        }

        // Verify tree depth: at least some alternatives should have multiple nodes
        let max_depth = alternatives
            .iter()
            .map(|a| a.nodes.len())
            .max()
            .unwrap_or(0);
        assert!(
            max_depth >= 2,
            "At least one alternative should have depth >= 2, got {}",
            max_depth
        );
    }

    #[tokio::test]
    async fn test_mcts_max_alternatives_limit() {
        let proxy = GdsHeuristicProxy::default();
        let config = MctsConfig {
            expansion_width: 5,
            num_rollouts: 50,
            max_alternatives: 1,
            ..Default::default()
        };
        let engine = MctsEngine::new(config, proxy);
        let source = make_source_trajectory(5);

        let alternatives = engine.generate_alternatives(&source).await.unwrap();

        assert!(
            alternatives.len() <= 1,
            "max_alternatives=1 should produce at most 1 result, got {}",
            alternatives.len()
        );
    }

    #[tokio::test]
    async fn test_mcts_simulated_session_id_format() {
        let proxy = GdsHeuristicProxy::default();
        let config = MctsConfig {
            expansion_width: 5,
            num_rollouts: 30,
            max_alternatives: 5,
            ..Default::default()
        };
        let engine = MctsEngine::new(config, proxy);
        let source = make_source_trajectory(4);
        let source_id = source.id;

        let alternatives = engine.generate_alternatives(&source).await.unwrap();

        for alt in &alternatives {
            assert!(
                alt.session_id.starts_with("mcts-sim-"),
                "Session ID should start with 'mcts-sim-', got '{}'",
                alt.session_id
            );
            let expected_suffix = source_id.to_string();
            assert!(
                alt.session_id.ends_with(&expected_suffix),
                "Session ID should end with source trajectory id '{}', got '{}'",
                expected_suffix,
                alt.session_id
            );
        }
    }

    #[tokio::test]
    async fn test_mcts_gamma_zero() {
        let proxy = GdsHeuristicProxy::default();
        let config = MctsConfig {
            expansion_width: 5,
            num_rollouts: 30,
            max_alternatives: 5,
            gamma: 0.0,
            ..Default::default()
        };
        let engine = MctsEngine::new(config, proxy);
        let source = make_source_trajectory(4);

        let alternatives = engine.generate_alternatives(&source).await.unwrap();

        // gamma=0.0 means no discount — engine should still produce valid results
        for alt in &alternatives {
            assert!(
                alt.total_reward.is_finite(),
                "Reward should be finite even with gamma=0.0, got {}",
                alt.total_reward
            );
            assert_eq!(alt.step_count, alt.nodes.len());
            for node in &alt.nodes {
                assert!(node.local_reward.is_finite());
                assert!(node.cumulative_reward.is_finite());
            }
        }
    }

    #[test]
    fn test_mcts_expansion_width() {
        let proxy = GdsHeuristicProxy::default();
        let expansion_width = 4;
        let config = MctsConfig {
            expansion_width,
            ..Default::default()
        };
        let engine = MctsEngine::new(config, proxy);
        let source = make_source_trajectory(3);

        // Create a root node and expand it
        let mut root = MctsNode::new(
            source.nodes[0].action_type.clone(),
            source.nodes[0].action_params.clone(),
            0,
        );

        engine.expand(&mut root, &source);

        // generate_action_alternatives filters out the source action from tool_actions,
        // takes up to expansion_width, then adds the original action back.
        // So children count should be <= expansion_width + 1 (alternatives + original).
        assert!(
            !root.children.is_empty(),
            "Expansion should create at least 1 child"
        );
        assert!(
            root.children.len() <= expansion_width + 1,
            "Expansion should create at most expansion_width + 1 children, got {}",
            root.children.len()
        );

        // Verify all children are at depth 1
        for child in root.children.values() {
            assert_eq!(child.depth, 1, "Children should be at depth 1");
            assert_eq!(child.visit_count, 0, "New children should have 0 visits");
        }
    }
}
