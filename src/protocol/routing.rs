//! Context Relevance Routing — multi-dimensional affinity scoring.
//!
//! Replaces flat trigger_patterns with a rich ContextVector that captures
//! the current work context (phase, structural complexity, domain, resource
//! availability, lifecycle position) and computes affinity scores against
//! each protocol's RelevanceVector.
//!
//! # Architecture
//!
//! ```text
//! ContextVector (current state)  ×  RelevanceVector (per protocol)  →  AffinityScore
//! ```
//!
//! The affinity is a weighted dot product across 5 dimensions, producing a
//! composite score ∈ [0, 1]. Each dimension contributes independently, and
//! the weights are configurable via `DimensionWeights`.

use serde::{Deserialize, Serialize};

// ============================================================================
// ContextVector — snapshot of the current work context
// ============================================================================

/// A multi-dimensional vector capturing the current work context.
///
/// Each dimension is a f64 in [0, 1]:
/// - **phase**: Where we are in the workflow (0=warmup → 1=closure)
/// - **structure**: Structural complexity of the current plan/task
/// - **domain**: Domain relevance (0=unrelated → 1=perfect match)
/// - **resource**: Resource availability (0=none → 1=fully available)
/// - **lifecycle**: Position in the project lifecycle (0=start → 1=end)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextVector {
    /// Workflow phase: 0.0=warmup, 0.25=planning, 0.5=execution, 0.75=review, 1.0=closure
    pub phase: f64,
    /// Structural complexity score based on task count, dependencies, affected files richness
    pub structure: f64,
    /// Domain relevance — semantic similarity between context and protocol domain
    pub domain: f64,
    /// Resource availability (worktrees, agents, parallel capacity)
    pub resource: f64,
    /// Position in the project lifecycle (0=start, 0.5=middle, 1.0=end)
    pub lifecycle: f64,
}

impl Default for ContextVector {
    fn default() -> Self {
        Self {
            phase: 0.5,
            structure: 0.5,
            domain: 0.5,
            resource: 1.0,
            lifecycle: 0.5,
        }
    }
}

impl ContextVector {
    /// Create a context vector from plan/task metrics.
    pub fn from_plan_context(
        phase: &str,
        task_count: usize,
        dependency_count: usize,
        affected_files_count: usize,
        completion_pct: f64,
    ) -> Self {
        let phase_score = match phase {
            "warmup" | "draft" => 0.0,
            "planning" | "approved" => 0.25,
            "execution" | "in_progress" => 0.5,
            "review" | "testing" => 0.75,
            "closure" | "completed" => 1.0,
            _ => 0.5,
        };

        // Structure: normalized complexity from task/dep/file counts
        let structure = Self::compute_structure(task_count, dependency_count, affected_files_count);

        Self {
            phase: phase_score,
            structure,
            domain: 0.5, // Default; can be overridden by embedding similarity
            resource: 1.0,
            lifecycle: completion_pct.clamp(0.0, 1.0),
        }
    }

    /// Compute structural complexity score from raw metrics.
    fn compute_structure(
        task_count: usize,
        dependency_count: usize,
        affected_files_count: usize,
    ) -> f64 {
        // Sigmoid-ish normalization: saturates around 20 tasks
        let task_score = 1.0 - (-(task_count as f64) / 10.0).exp();
        // Dependency density relative to task count
        let dep_ratio = if task_count > 0 {
            (dependency_count as f64 / task_count as f64).min(2.0) / 2.0
        } else {
            0.0
        };
        // File breadth: saturates around 30 files
        let file_score = 1.0 - (-(affected_files_count as f64) / 15.0).exp();

        // Weighted combination
        (task_score * 0.4 + dep_ratio * 0.3 + file_score * 0.3).clamp(0.0, 1.0)
    }

    /// Return the vector as an array of (name, value) pairs for serialization.
    pub fn dimensions(&self) -> Vec<(String, f64)> {
        vec![
            ("phase".to_string(), self.phase),
            ("structure".to_string(), self.structure),
            ("domain".to_string(), self.domain),
            ("resource".to_string(), self.resource),
            ("lifecycle".to_string(), self.lifecycle),
        ]
    }
}

// ============================================================================
// RelevanceVector — per-protocol relevance profile
// ============================================================================

/// A protocol's relevance profile across the 5 context dimensions.
///
/// Each value indicates the *ideal* context for this protocol. A value of 0.5
/// means "neutral / any context". Values closer to 0 or 1 indicate preference.
///
/// Example: a wave-execution protocol has:
/// - phase=0.5 (execution phase)
/// - structure=0.8 (prefers complex plans with many tasks)
/// - domain=0.5 (neutral — applies to any domain)
/// - resource=0.7 (benefits from parallel resources)
/// - lifecycle=0.3 (early-mid lifecycle — before tasks are done)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceVector {
    /// Preferred workflow phase
    pub phase: f64,
    /// Preferred structural complexity
    pub structure: f64,
    /// Domain specificity (0.5 = domain-agnostic)
    pub domain: f64,
    /// Preferred resource availability
    pub resource: f64,
    /// Preferred lifecycle position
    pub lifecycle: f64,
}

impl Default for RelevanceVector {
    fn default() -> Self {
        Self {
            phase: 0.5,
            structure: 0.5,
            domain: 0.5,
            resource: 0.5,
            lifecycle: 0.5,
        }
    }
}

impl RelevanceVector {
    /// Return the vector as an array for computation.
    fn as_array(&self) -> [f64; 5] {
        [
            self.phase,
            self.structure,
            self.domain,
            self.resource,
            self.lifecycle,
        ]
    }

    /// Return the vector as (name, value) pairs.
    pub fn dimensions(&self) -> Vec<(String, f64)> {
        vec![
            ("phase".to_string(), self.phase),
            ("structure".to_string(), self.structure),
            ("domain".to_string(), self.domain),
            ("resource".to_string(), self.resource),
            ("lifecycle".to_string(), self.lifecycle),
        ]
    }
}

// ============================================================================
// DimensionWeights — configurable importance per dimension
// ============================================================================

/// Weights controlling the importance of each dimension in the affinity score.
///
/// All weights must be non-negative. They are normalized to sum to 1.0
/// before computing affinity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionWeights {
    pub phase: f64,
    pub structure: f64,
    pub domain: f64,
    pub resource: f64,
    pub lifecycle: f64,
}

impl Default for DimensionWeights {
    fn default() -> Self {
        Self {
            phase: 0.30,
            structure: 0.25,
            domain: 0.20,
            resource: 0.10,
            lifecycle: 0.15,
        }
    }
}

impl DimensionWeights {
    /// Return normalized weights (sum = 1.0).
    fn normalized(&self) -> [f64; 5] {
        let raw = [
            self.phase,
            self.structure,
            self.domain,
            self.resource,
            self.lifecycle,
        ];
        let sum: f64 = raw.iter().sum();
        if sum <= 0.0 {
            return [0.2; 5];
        }
        [
            raw[0] / sum,
            raw[1] / sum,
            raw[2] / sum,
            raw[3] / sum,
            raw[4] / sum,
        ]
    }
}

// ============================================================================
// AffinityScore — result of routing computation
// ============================================================================

/// Result of computing affinity between a ContextVector and a protocol's RelevanceVector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffinityScore {
    /// Composite affinity score ∈ [0, 1]
    pub score: f64,
    /// Per-dimension contribution breakdown
    pub dimensions: Vec<DimensionScore>,
    /// Human-readable explanation of why this score was assigned
    pub explanation: String,
}

/// Score contribution from a single dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionScore {
    /// Dimension name (phase, structure, domain, resource, lifecycle)
    pub name: String,
    /// Context value for this dimension
    pub context_value: f64,
    /// Protocol's relevance value for this dimension
    pub relevance_value: f64,
    /// Weight applied to this dimension
    pub weight: f64,
    /// Contribution to the composite score = weight × (1 - |context - relevance|)
    pub contribution: f64,
}

// ============================================================================
// Routing computation
// ============================================================================

/// Compute the affinity score between a context and a protocol's relevance vector.
///
/// The affinity is computed as:
/// ```text
/// score = Σ(weight_i × (1 - |context_i - relevance_i|))
/// ```
///
/// Where |context_i - relevance_i| is the distance between the current context
/// and the protocol's ideal context on each dimension. A distance of 0 means
/// perfect match (contribution = weight), distance of 1 means no match (contribution = 0).
pub fn compute_affinity(
    context: &ContextVector,
    relevance: &RelevanceVector,
    weights: &DimensionWeights,
) -> AffinityScore {
    let ctx = [
        context.phase,
        context.structure,
        context.domain,
        context.resource,
        context.lifecycle,
    ];
    let rel = relevance.as_array();
    let w = weights.normalized();
    let names = ["phase", "structure", "domain", "resource", "lifecycle"];

    let mut dimensions = Vec::with_capacity(5);
    let mut total_score = 0.0;
    let mut dominant_dimension = ("", 0.0f64);

    for i in 0..5 {
        let distance = (ctx[i] - rel[i]).abs();
        let similarity = 1.0 - distance;
        let contribution = w[i] * similarity;
        total_score += contribution;

        if contribution > dominant_dimension.1 {
            dominant_dimension = (names[i], contribution);
        }

        dimensions.push(DimensionScore {
            name: names[i].to_string(),
            context_value: ctx[i],
            relevance_value: rel[i],
            weight: w[i],
            contribution,
        });
    }

    // Sort by contribution descending for the explanation
    let mut sorted_dims = dimensions.clone();
    sorted_dims.sort_by(|a, b| b.contribution.partial_cmp(&a.contribution).unwrap());

    // Build human-readable explanation
    let explanation = build_explanation(&sorted_dims, total_score);

    AffinityScore {
        score: total_score.clamp(0.0, 1.0),
        dimensions,
        explanation,
    }
}

/// Build a human-readable explanation from dimension contributions.
fn build_explanation(sorted_dims: &[DimensionScore], total_score: f64) -> String {
    if total_score < 0.3 {
        return format!(
            "Low relevance ({:.0}%): context does not match this protocol's ideal conditions.",
            total_score * 100.0
        );
    }

    let top: Vec<String> = sorted_dims
        .iter()
        .take(2)
        .filter(|d| d.contribution > 0.1)
        .map(|d| {
            let similarity_pct = ((1.0 - (d.context_value - d.relevance_value).abs()) * 100.0) as i32;
            format!("{} match ({}%)", d.name, similarity_pct)
        })
        .collect();

    if top.is_empty() {
        format!("Moderate relevance ({:.0}%)", total_score * 100.0)
    } else {
        format!(
            "Activated ({:.0}%): {}",
            total_score * 100.0,
            top.join(", ")
        )
    }
}

/// Route result for a single protocol — used in the routing endpoint response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteResult {
    /// Protocol ID
    pub protocol_id: uuid::Uuid,
    /// Protocol name
    pub protocol_name: String,
    /// Protocol category
    pub protocol_category: String,
    /// Computed affinity score
    pub affinity: AffinityScore,
    /// The protocol's relevance vector
    pub relevance_vector: RelevanceVector,
}

/// Full routing response — returned by the route endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteResponse {
    /// The context vector that was used for routing
    pub context: ContextVector,
    /// Weights used for dimension scoring
    pub weights: DimensionWeights,
    /// Ranked list of protocols (highest affinity first)
    pub results: Vec<RouteResult>,
    /// Total number of protocols evaluated
    pub total_evaluated: usize,
}

// ============================================================================
// Multi-protocol ranking
// ============================================================================

/// Rank multiple protocols against a context vector.
///
/// Returns a `RouteResponse` with protocols sorted by descending affinity score.
/// Protocols without a relevance_vector are scored against the default (neutral)
/// vector, which typically yields a moderate score.
pub fn rank_protocols(
    context: &ContextVector,
    protocols: &[super::models::Protocol],
    weights: &DimensionWeights,
) -> RouteResponse {
    let total_evaluated = protocols.len();
    let mut results: Vec<RouteResult> = protocols
        .iter()
        .map(|proto| {
            let relevance = proto
                .relevance_vector
                .clone()
                .unwrap_or_default();
            let affinity = compute_affinity(context, &relevance, weights);
            RouteResult {
                protocol_id: proto.id,
                protocol_name: proto.name.clone(),
                protocol_category: proto.protocol_category.to_string(),
                affinity,
                relevance_vector: relevance,
            }
        })
        .collect();

    // Sort by descending score (highest affinity first)
    results.sort_by(|a, b| {
        b.affinity
            .score
            .partial_cmp(&a.affinity.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    RouteResponse {
        context: context.clone(),
        weights: weights.clone(),
        results,
        total_evaluated,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_match() {
        let ctx = ContextVector {
            phase: 0.5,
            structure: 0.8,
            domain: 0.7,
            resource: 1.0,
            lifecycle: 0.3,
        };
        let rel = RelevanceVector {
            phase: 0.5,
            structure: 0.8,
            domain: 0.7,
            resource: 1.0,
            lifecycle: 0.3,
        };
        let weights = DimensionWeights::default();
        let score = compute_affinity(&ctx, &rel, &weights);
        assert!((score.score - 1.0).abs() < 0.001, "Perfect match should be 1.0");
        assert_eq!(score.dimensions.len(), 5);
    }

    #[test]
    fn test_worst_match() {
        let ctx = ContextVector {
            phase: 0.0,
            structure: 0.0,
            domain: 0.0,
            resource: 0.0,
            lifecycle: 0.0,
        };
        let rel = RelevanceVector {
            phase: 1.0,
            structure: 1.0,
            domain: 1.0,
            resource: 1.0,
            lifecycle: 1.0,
        };
        let weights = DimensionWeights::default();
        let score = compute_affinity(&ctx, &rel, &weights);
        assert!(score.score < 0.01, "Worst match should be ~0.0, got {}", score.score);
    }

    #[test]
    fn test_partial_match() {
        let ctx = ContextVector {
            phase: 0.5,
            structure: 0.5,
            domain: 0.5,
            resource: 0.5,
            lifecycle: 0.5,
        };
        let rel = RelevanceVector {
            phase: 0.5,
            structure: 0.5,
            domain: 0.5,
            resource: 0.5,
            lifecycle: 0.5,
        };
        let weights = DimensionWeights::default();
        let score = compute_affinity(&ctx, &rel, &weights);
        // Neutral context matches neutral relevance = 1.0
        assert!((score.score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_from_plan_context() {
        let ctx = ContextVector::from_plan_context("execution", 12, 8, 25, 0.3);
        assert!((ctx.phase - 0.5).abs() < 0.001);
        assert!(ctx.structure > 0.5, "12 tasks with 8 deps should be complex");
        assert!((ctx.lifecycle - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_structure_computation() {
        // Minimal project
        let simple = ContextVector::compute_structure(1, 0, 2);
        assert!(simple < 0.3, "Simple project should have low structure score");

        // Complex project
        let complex = ContextVector::compute_structure(20, 15, 40);
        assert!(complex > 0.7, "Complex project should have high structure score");
    }

    #[test]
    fn test_explanation_low_score() {
        // All dimensions far apart → total score < 0.3
        let ctx = ContextVector {
            phase: 0.0,
            structure: 0.0,
            domain: 0.0,
            resource: 0.0,
            lifecycle: 0.0,
        };
        let rel = RelevanceVector {
            phase: 1.0,
            structure: 1.0,
            domain: 1.0,
            resource: 1.0,
            lifecycle: 1.0,
        };
        let weights = DimensionWeights::default();
        let score = compute_affinity(&ctx, &rel, &weights);
        assert!(
            score.explanation.contains("relevance"),
            "Low score ({:.2}) should explain relevance: {}",
            score.score,
            score.explanation
        );
    }

    #[test]
    fn test_explanation_high_score() {
        let ctx = ContextVector {
            phase: 0.5,
            structure: 0.8,
            domain: 0.7,
            resource: 1.0,
            lifecycle: 0.3,
        };
        let rel = RelevanceVector {
            phase: 0.5,
            structure: 0.8,
            domain: 0.7,
            resource: 1.0,
            lifecycle: 0.3,
        };
        let weights = DimensionWeights::default();
        let score = compute_affinity(&ctx, &rel, &weights);
        assert!(score.explanation.contains("Activated"), "High score should say Activated");
    }

    #[test]
    fn test_default_weights_sum_to_one() {
        let w = DimensionWeights::default();
        let normalized = w.normalized();
        let sum: f64 = normalized.iter().sum();
        assert!((sum - 1.0).abs() < 0.001, "Normalized weights should sum to 1.0");
    }

    #[test]
    fn test_score_is_bounded() {
        // Random context and relevance
        let ctx = ContextVector {
            phase: 0.1,
            structure: 0.9,
            domain: 0.3,
            resource: 0.6,
            lifecycle: 0.8,
        };
        let rel = RelevanceVector {
            phase: 0.7,
            structure: 0.2,
            domain: 0.9,
            resource: 0.4,
            lifecycle: 0.1,
        };
        let weights = DimensionWeights::default();
        let score = compute_affinity(&ctx, &rel, &weights);
        assert!(score.score >= 0.0 && score.score <= 1.0, "Score must be in [0, 1]");
    }

    #[test]
    fn test_dimension_score_breakdown() {
        let ctx = ContextVector {
            phase: 0.5,
            structure: 0.5,
            domain: 0.5,
            resource: 0.5,
            lifecycle: 0.5,
        };
        let rel = RelevanceVector {
            phase: 0.5,
            structure: 1.0,
            domain: 0.5,
            resource: 0.5,
            lifecycle: 0.5,
        };
        let weights = DimensionWeights::default();
        let score = compute_affinity(&ctx, &rel, &weights);

        // Structure dimension should have lower contribution (distance = 0.5)
        let structure_dim = score.dimensions.iter().find(|d| d.name == "structure").unwrap();
        let phase_dim = score.dimensions.iter().find(|d| d.name == "phase").unwrap();

        assert!(
            phase_dim.contribution > structure_dim.contribution,
            "Phase (exact match) should contribute more than structure (0.5 distance)"
        );
    }

    #[test]
    fn test_relevance_vector_dimensions() {
        let rel = RelevanceVector::default();
        let dims = rel.dimensions();
        assert_eq!(dims.len(), 5);
        assert!(dims.iter().all(|(_, v)| *v == 0.5));
    }

    #[test]
    fn test_rank_protocols_ordering() {
        use super::super::models::Protocol;

        let ctx = ContextVector {
            phase: 0.5,
            structure: 0.8,
            domain: 0.7,
            resource: 1.0,
            lifecycle: 0.3,
        };

        let project_id = uuid::Uuid::new_v4();

        // Protocol A: perfect match
        let mut proto_a = Protocol::new(project_id, "Wave Execution", uuid::Uuid::new_v4());
        proto_a.relevance_vector = Some(RelevanceVector {
            phase: 0.5,
            structure: 0.8,
            domain: 0.7,
            resource: 1.0,
            lifecycle: 0.3,
        });

        // Protocol B: partial match
        let mut proto_b = Protocol::new(project_id, "Code Review", uuid::Uuid::new_v4());
        proto_b.relevance_vector = Some(RelevanceVector {
            phase: 0.75,
            structure: 0.3,
            domain: 0.5,
            resource: 0.5,
            lifecycle: 0.8,
        });

        // Protocol C: no relevance vector (defaults to neutral)
        let proto_c = Protocol::new(project_id, "Generic", uuid::Uuid::new_v4());

        let weights = DimensionWeights::default();
        let response = rank_protocols(&ctx, &[proto_b.clone(), proto_c.clone(), proto_a.clone()], &weights);

        assert_eq!(response.total_evaluated, 3);
        assert_eq!(response.results.len(), 3);

        // First result should be proto_a (perfect match = score 1.0)
        assert_eq!(response.results[0].protocol_id, proto_a.id);
        assert!(
            response.results[0].affinity.score > response.results[1].affinity.score,
            "Best match should be ranked first"
        );
        assert!(
            response.results[1].affinity.score >= response.results[2].affinity.score,
            "Results should be sorted by descending score"
        );
    }

    #[test]
    fn test_rank_protocols_empty() {
        let ctx = ContextVector::default();
        let weights = DimensionWeights::default();
        let response = rank_protocols(&ctx, &[], &weights);
        assert_eq!(response.total_evaluated, 0);
        assert!(response.results.is_empty());
    }
}
