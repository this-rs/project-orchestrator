//! ExecutionVector — multidimensional run observation for comparison and inference.
//!
//! Each plan run produces an ExecutionVector capturing 7 dimensions:
//! trigger, timing, cost, quality, knowledge, drift, outcome.
//! These vectors enable run-to-run comparison and predictive inference.

use crate::runner::models::{PlanRunStatus, TriggerSource};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Full execution vector for a plan run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionVector {
    pub trigger: TriggerDim,
    pub timing: TimingDim,
    pub cost: CostDim,
    pub quality: QualityDim,
    pub knowledge: KnowledgeDim,
    pub drift: DriftDim,
    pub outcome: OutcomeDim,
}

/// How the run was triggered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerDim {
    pub source: TriggerSource,
    pub trigger_id: Option<Uuid>,
    pub payload_hash: Option<String>,
}

/// Timing dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingDim {
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub duration_secs: f64,
    pub per_task_durations: Vec<(Uuid, f64)>,
}

/// Cost dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostDim {
    pub total_cost_usd: f64,
    pub per_task_costs: Vec<(Uuid, f64)>,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Quality dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDim {
    pub tasks_completed: u32,
    pub tasks_failed: u32,
    pub tasks_retried: u32,
    pub steps_completed: u32,
    pub steps_skipped: u32,
    pub build_pass: bool,
    pub tests_pass: bool,
}

/// Knowledge captured during the run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDim {
    pub notes_created: u32,
    pub decisions_created: u32,
    pub commits_linked: u32,
    pub files_modified: u32,
    pub lines_added: u32,
    pub lines_deleted: u32,
}

/// Drift/instability signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDim {
    pub idle_warnings: u32,
    pub loop_warnings: u32,
    pub compaction_count: u32,
    pub timeout_count: u32,
    pub budget_warnings: u32,
}

/// Outcome of the run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeDim {
    pub status: PlanRunStatus,
    pub failure_reasons: Vec<String>,
    pub pr_url: Option<String>,
}

// ============================================================================
// VectorCollector — accumulates metrics during plan execution
// ============================================================================

/// Collects execution metrics in real-time during a plan run.
///
/// Accumulates drift signals, knowledge capture counts, and per-task
/// timing/cost. Call `finalize()` at the end to produce the full
/// `ExecutionVector` with all 7 dimensions populated.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorCollector {
    // Drift counters (incremented by guard/runner)
    pub idle_warnings: u32,
    pub loop_warnings: u32,
    pub compaction_count: u32,
    pub timeout_count: u32,
    pub budget_warnings: u32,
    // Knowledge counters (incremented by enricher results)
    pub notes_created: u32,
    pub decisions_created: u32,
    pub commits_linked: u32,
    pub files_modified: u32,
    pub lines_added: u32,
    pub lines_deleted: u32,
    // Per-task timing/cost
    pub per_task_durations: Vec<(Uuid, f64)>,
    pub per_task_costs: Vec<(Uuid, f64)>,
    // Token tracking
    pub input_tokens: u64,
    pub output_tokens: u64,
    // Outcome
    pub failure_reasons: Vec<String>,
    pub pr_url: Option<String>,
}

impl VectorCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed task's timing and cost.
    pub fn record_task(&mut self, task_id: Uuid, duration_secs: f64, cost_usd: f64) {
        self.per_task_durations.push((task_id, duration_secs));
        self.per_task_costs.push((task_id, cost_usd));
    }

    /// Record enrichment results from the enricher.
    pub fn record_enrichment(&mut self, commits: usize, note_created: bool, affects: usize) {
        self.commits_linked += commits as u32;
        if note_created {
            self.notes_created += 1;
        }
        self.decisions_created += affects as u32; // affects are decision anchors
    }

    /// Record a drift event (idle, loop, etc.)
    pub fn record_drift(&mut self, event: DriftEvent) {
        match event {
            DriftEvent::Idle => self.idle_warnings += 1,
            DriftEvent::Loop => self.loop_warnings += 1,
            DriftEvent::Compaction => self.compaction_count += 1,
            DriftEvent::Timeout => self.timeout_count += 1,
            DriftEvent::Budget => self.budget_warnings += 1,
        }
    }

    /// Build the final ExecutionVector from the collected data + RunnerState.
    pub fn finalize(&self, state: &crate::runner::RunnerState) -> ExecutionVector {
        let duration_secs = state
            .completed_at
            .map(|c| (c - state.started_at).num_seconds() as f64)
            .unwrap_or(0.0);

        ExecutionVector {
            trigger: TriggerDim {
                source: state.triggered_by.clone(),
                trigger_id: None,
                payload_hash: None,
            },
            timing: TimingDim {
                started_at: state.started_at,
                completed_at: state.completed_at,
                duration_secs,
                per_task_durations: self.per_task_durations.clone(),
            },
            cost: CostDim {
                total_cost_usd: state.cost_usd,
                per_task_costs: self.per_task_costs.clone(),
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
            },
            quality: QualityDim {
                tasks_completed: state.completed_tasks.len() as u32,
                tasks_failed: state.failed_tasks.len() as u32,
                tasks_retried: state.retry_counts.values().filter(|&&v| v > 0).count() as u32,
                steps_completed: 0, // TODO: count from graph when available
                steps_skipped: 0,
                build_pass: state.status == PlanRunStatus::Completed,
                tests_pass: state.status == PlanRunStatus::Completed,
            },
            knowledge: KnowledgeDim {
                notes_created: self.notes_created,
                decisions_created: self.decisions_created,
                commits_linked: self.commits_linked,
                files_modified: self.files_modified,
                lines_added: self.lines_added,
                lines_deleted: self.lines_deleted,
            },
            drift: DriftDim {
                idle_warnings: self.idle_warnings,
                loop_warnings: self.loop_warnings,
                compaction_count: self.compaction_count,
                timeout_count: self.timeout_count,
                budget_warnings: self.budget_warnings,
            },
            outcome: OutcomeDim {
                status: state.status,
                failure_reasons: self.failure_reasons.clone(),
                pr_url: self.pr_url.clone(),
            },
        }
    }
}

/// Drift event types for the collector.
#[derive(Debug, Clone)]
pub enum DriftEvent {
    Idle,
    Loop,
    Compaction,
    Timeout,
    Budget,
}

// ============================================================================
// Build from RunnerState (minimal vector from persisted data)
// ============================================================================

impl ExecutionVector {
    /// Build a minimal ExecutionVector from a RunnerState.
    ///
    /// Since RunnerState doesn't capture all dimensions (knowledge, drift details),
    /// those are filled with defaults. This is sufficient for comparison and prediction
    /// based on timing, cost, and outcome data.
    pub fn from_runner_state(state: &crate::runner::RunnerState) -> Self {
        let duration_secs = state
            .completed_at
            .map(|c| (c - state.started_at).num_seconds() as f64)
            .unwrap_or(0.0);

        Self {
            trigger: TriggerDim {
                source: state.triggered_by.clone(),
                trigger_id: None,
                payload_hash: None,
            },
            timing: TimingDim {
                started_at: state.started_at,
                completed_at: state.completed_at,
                duration_secs,
                per_task_durations: vec![],
            },
            cost: CostDim {
                total_cost_usd: state.cost_usd,
                per_task_costs: vec![],
                input_tokens: 0,
                output_tokens: 0,
            },
            quality: QualityDim {
                tasks_completed: state.completed_tasks.len() as u32,
                tasks_failed: state.failed_tasks.len() as u32,
                tasks_retried: state.retry_counts.values().filter(|&&v| v > 0).count() as u32,
                steps_completed: 0, // not tracked in RunnerState
                steps_skipped: 0,
                build_pass: state.status == PlanRunStatus::Completed,
                tests_pass: state.status == PlanRunStatus::Completed,
            },
            knowledge: KnowledgeDim {
                notes_created: 0,
                decisions_created: 0,
                commits_linked: 0,
                files_modified: 0,
                lines_added: 0,
                lines_deleted: 0,
            },
            drift: DriftDim {
                idle_warnings: 0,
                loop_warnings: 0,
                compaction_count: 0,
                timeout_count: 0,
                budget_warnings: 0,
            },
            outcome: OutcomeDim {
                status: state.status,
                failure_reasons: vec![],
                pr_url: None,
            },
        }
    }
}

// ============================================================================
// Derived metrics
// ============================================================================

impl ExecutionVector {
    /// Efficiency = quality_score / cost (higher = better).
    /// Quality score = tasks_completed / max(tasks_completed + tasks_failed, 1).
    pub fn efficiency(&self) -> f64 {
        let quality_score = self.quality.tasks_completed as f64
            / (self.quality.tasks_completed + self.quality.tasks_failed).max(1) as f64;
        if self.cost.total_cost_usd > 0.0 {
            quality_score / self.cost.total_cost_usd
        } else {
            quality_score
        }
    }

    /// Velocity = quality_score / duration (higher = faster success).
    pub fn velocity(&self) -> f64 {
        let quality_score = self.quality.tasks_completed as f64
            / (self.quality.tasks_completed + self.quality.tasks_failed).max(1) as f64;
        if self.timing.duration_secs > 0.0 {
            quality_score / self.timing.duration_secs * 60.0 // per minute
        } else {
            quality_score
        }
    }

    /// Stability = 1 - (drift_total / event_total) (1.0 = no drift, 0.0 = all drift).
    pub fn stability(&self) -> f64 {
        let drift_total = (self.drift.idle_warnings
            + self.drift.loop_warnings
            + self.drift.compaction_count
            + self.drift.timeout_count
            + self.drift.budget_warnings) as f64;
        let event_total = (self.quality.tasks_completed
            + self.quality.tasks_failed
            + self.quality.steps_completed
            + self.quality.steps_skipped) as f64;
        let total = event_total + drift_total;
        if total > 0.0 {
            1.0 - (drift_total / total)
        } else {
            1.0
        }
    }
}

// ============================================================================
// Comparison
// ============================================================================

/// Result of comparing multiple execution vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub run_count: usize,
    pub dimensions: DimensionDeltas,
    pub derived: DerivedDeltas,
}

/// Deltas for each dimension across runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionDeltas {
    pub duration: MetricSeries,
    pub cost: MetricSeries,
    pub tasks_completed: MetricSeries,
    pub tasks_failed: MetricSeries,
    pub notes_created: MetricSeries,
    pub drift_total: MetricSeries,
}

/// Deltas for derived metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedDeltas {
    pub efficiency: MetricSeries,
    pub velocity: MetricSeries,
    pub stability: MetricSeries,
}

/// A series of values with trend analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSeries {
    pub values: Vec<f64>,
    pub min: f64,
    pub max: f64,
    pub avg: f64,
    pub trend: Trend,
}

/// Trend direction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Trend {
    Improving,
    Degrading,
    Stable,
}

impl MetricSeries {
    /// Build a MetricSeries from a list of values.
    /// `higher_is_better` controls whether an increasing trend is "improving" or "degrading".
    pub fn from_values(values: Vec<f64>, higher_is_better: bool) -> Self {
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let avg = if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        };

        let trend = if values.len() < 2 {
            Trend::Stable
        } else {
            let first_half: f64 =
                values[..values.len() / 2].iter().sum::<f64>() / (values.len() / 2) as f64;
            let second_half: f64 = values[values.len() / 2..].iter().sum::<f64>()
                / (values.len() - values.len() / 2) as f64;
            let delta = (second_half - first_half) / first_half.abs().max(0.001);
            if delta.abs() < 0.1 {
                Trend::Stable
            } else if (delta > 0.0) == higher_is_better {
                Trend::Improving
            } else {
                Trend::Degrading
            }
        };

        Self {
            values,
            min,
            max,
            avg,
            trend,
        }
    }
}

/// Compare multiple execution vectors.
pub fn compare_vectors(vectors: &[ExecutionVector]) -> ComparisonResult {
    let durations: Vec<f64> = vectors.iter().map(|v| v.timing.duration_secs).collect();
    let costs: Vec<f64> = vectors.iter().map(|v| v.cost.total_cost_usd).collect();
    let completed: Vec<f64> = vectors
        .iter()
        .map(|v| v.quality.tasks_completed as f64)
        .collect();
    let failed: Vec<f64> = vectors
        .iter()
        .map(|v| v.quality.tasks_failed as f64)
        .collect();
    let notes: Vec<f64> = vectors
        .iter()
        .map(|v| v.knowledge.notes_created as f64)
        .collect();
    let drift: Vec<f64> = vectors
        .iter()
        .map(|v| {
            (v.drift.idle_warnings
                + v.drift.loop_warnings
                + v.drift.compaction_count
                + v.drift.timeout_count
                + v.drift.budget_warnings) as f64
        })
        .collect();

    let efficiencies: Vec<f64> = vectors.iter().map(|v| v.efficiency()).collect();
    let velocities: Vec<f64> = vectors.iter().map(|v| v.velocity()).collect();
    let stabilities: Vec<f64> = vectors.iter().map(|v| v.stability()).collect();

    ComparisonResult {
        run_count: vectors.len(),
        dimensions: DimensionDeltas {
            duration: MetricSeries::from_values(durations, false), // lower is better
            cost: MetricSeries::from_values(costs, false),
            tasks_completed: MetricSeries::from_values(completed, true),
            tasks_failed: MetricSeries::from_values(failed, false),
            notes_created: MetricSeries::from_values(notes, true),
            drift_total: MetricSeries::from_values(drift, false),
        },
        derived: DerivedDeltas {
            efficiency: MetricSeries::from_values(efficiencies, true),
            velocity: MetricSeries::from_values(velocities, true),
            stability: MetricSeries::from_values(stabilities, true),
        },
    }
}

// ============================================================================
// Prediction / Inference
// ============================================================================

/// Prediction for a future run based on historical vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunPrediction {
    pub estimated_duration_secs: f64,
    pub estimated_cost_usd: f64,
    pub success_probability: f64,
    pub confidence: Confidence,
    pub anomalies: Vec<Anomaly>,
    pub suggestions: Vec<String>,
}

/// Confidence level based on number of historical runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    High,   // >5 runs
    Medium, // 2-5 runs
    Low,    // <2 runs
}

/// An anomaly detected in the last run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    pub dimension: String,
    pub value: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub severity: f64, // how many std_devs away
}

/// Predict the next run based on historical execution vectors.
///
/// Uses exponential weighted average (decay factor 0.7 per run).
pub fn predict_run(vectors: &[ExecutionVector]) -> RunPrediction {
    if vectors.is_empty() {
        return RunPrediction {
            estimated_duration_secs: 0.0,
            estimated_cost_usd: 0.0,
            success_probability: 0.5,
            confidence: Confidence::Low,
            anomalies: vec![],
            suggestions: vec!["No historical runs available for prediction".to_string()],
        };
    }

    let decay = 0.7_f64;
    let n = vectors.len();

    // Compute weights (most recent = highest weight)
    let weights: Vec<f64> = (0..n).map(|i| decay.powi((n - 1 - i) as i32)).collect();
    let weight_sum: f64 = weights.iter().sum();

    // Weighted averages
    let est_duration: f64 = vectors
        .iter()
        .zip(&weights)
        .map(|(v, w)| v.timing.duration_secs * w)
        .sum::<f64>()
        / weight_sum;

    let est_cost: f64 = vectors
        .iter()
        .zip(&weights)
        .map(|(v, w)| v.cost.total_cost_usd * w)
        .sum::<f64>()
        / weight_sum;

    let completed_count = vectors
        .iter()
        .filter(|v| v.outcome.status == PlanRunStatus::Completed)
        .count();
    let success_prob = completed_count as f64 / n as f64;

    let confidence = if n > 5 {
        Confidence::High
    } else if n >= 2 {
        Confidence::Medium
    } else {
        Confidence::Low
    };

    // Anomaly detection on the latest run
    let mut anomalies = vec![];
    if n >= 3 {
        let latest = &vectors[n - 1];
        let check = |name: &str, value: f64, extractor: &dyn Fn(&ExecutionVector) -> f64| {
            let vals: Vec<f64> = vectors[..n - 1].iter().map(extractor).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            let std_dev = variance.sqrt();
            if std_dev > 0.0 {
                let severity = ((value - mean) / std_dev).abs();
                if severity > 2.0 {
                    Some(Anomaly {
                        dimension: name.to_string(),
                        value,
                        mean,
                        std_dev,
                        severity,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some(a) = check("duration", latest.timing.duration_secs, &|v| {
            v.timing.duration_secs
        }) {
            anomalies.push(a);
        }
        if let Some(a) = check("cost", latest.cost.total_cost_usd, &|v| {
            v.cost.total_cost_usd
        }) {
            anomalies.push(a);
        }
        if let Some(a) = check("tasks_failed", latest.quality.tasks_failed as f64, &|v| {
            v.quality.tasks_failed as f64
        }) {
            anomalies.push(a);
        }
    }

    // Suggestions
    let mut suggestions = vec![];
    if n >= 3 {
        // Check for tasks that fail frequently
        let avg_fail_rate = vectors
            .iter()
            .map(|v| {
                v.quality.tasks_failed as f64
                    / (v.quality.tasks_completed + v.quality.tasks_failed).max(1) as f64
            })
            .sum::<f64>()
            / n as f64;
        if avg_fail_rate > 0.4 {
            suggestions.push(format!(
                "High failure rate ({:.0}%). Consider decomposing complex tasks.",
                avg_fail_rate * 100.0
            ));
        }

        let avg_drift: f64 = vectors
            .iter()
            .map(|v| (v.drift.idle_warnings + v.drift.loop_warnings + v.drift.timeout_count) as f64)
            .sum::<f64>()
            / n as f64;
        if avg_drift > 3.0 {
            suggestions.push(format!(
                "High drift average ({:.1} warnings/run). Consider tighter guard thresholds.",
                avg_drift
            ));
        }
    }

    RunPrediction {
        estimated_duration_secs: est_duration,
        estimated_cost_usd: est_cost,
        success_probability: success_prob,
        confidence,
        anomalies,
        suggestions,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vector(
        duration: f64,
        cost: f64,
        completed: u32,
        failed: u32,
        status: PlanRunStatus,
    ) -> ExecutionVector {
        ExecutionVector {
            trigger: TriggerDim {
                source: TriggerSource::Manual,
                trigger_id: None,
                payload_hash: None,
            },
            timing: TimingDim {
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
                duration_secs: duration,
                per_task_durations: vec![],
            },
            cost: CostDim {
                total_cost_usd: cost,
                per_task_costs: vec![],
                input_tokens: 1000,
                output_tokens: 500,
            },
            quality: QualityDim {
                tasks_completed: completed,
                tasks_failed: failed,
                tasks_retried: 0,
                steps_completed: completed * 3,
                steps_skipped: 0,
                build_pass: true,
                tests_pass: true,
            },
            knowledge: KnowledgeDim {
                notes_created: 2,
                decisions_created: 1,
                commits_linked: completed,
                files_modified: completed * 2,
                lines_added: 100,
                lines_deleted: 20,
            },
            drift: DriftDim {
                idle_warnings: 0,
                loop_warnings: 0,
                compaction_count: 0,
                timeout_count: 0,
                budget_warnings: 0,
            },
            outcome: OutcomeDim {
                status,
                failure_reasons: vec![],
                pr_url: None,
            },
        }
    }

    #[test]
    fn test_derived_metrics() {
        let v = make_vector(600.0, 2.0, 5, 0, PlanRunStatus::Completed);
        assert!((v.efficiency() - 0.5).abs() < 0.01); // 1.0 / 2.0
        assert!(v.velocity() > 0.0);
        assert!((v.stability() - 1.0).abs() < 0.01); // no drift
    }

    #[test]
    fn test_stability_with_drift() {
        let mut v = make_vector(600.0, 2.0, 5, 0, PlanRunStatus::Completed);
        v.drift.idle_warnings = 5;
        v.drift.loop_warnings = 5;
        // events = 5 + 0 + 15 + 0 = 20, drift = 10, total = 30
        // stability = 1 - 10/30 = 0.667
        assert!(v.stability() < 1.0);
        assert!(v.stability() > 0.5);
    }

    #[test]
    fn test_compare_vectors() {
        let vectors = vec![
            make_vector(600.0, 2.0, 5, 0, PlanRunStatus::Completed),
            make_vector(500.0, 1.8, 5, 0, PlanRunStatus::Completed),
            make_vector(400.0, 1.5, 5, 0, PlanRunStatus::Completed),
        ];
        let result = compare_vectors(&vectors);
        assert_eq!(result.run_count, 3);
        assert_eq!(result.dimensions.duration.trend, Trend::Improving); // decreasing duration = improving
        assert_eq!(result.dimensions.cost.trend, Trend::Improving); // decreasing cost = improving
    }

    #[test]
    fn test_predict_run_with_history() {
        let vectors = vec![
            make_vector(600.0, 2.0, 5, 0, PlanRunStatus::Completed),
            make_vector(550.0, 1.8, 5, 0, PlanRunStatus::Completed),
            make_vector(500.0, 1.5, 5, 0, PlanRunStatus::Completed),
            make_vector(480.0, 1.4, 5, 0, PlanRunStatus::Completed),
            make_vector(450.0, 1.3, 5, 0, PlanRunStatus::Completed),
        ];
        let pred = predict_run(&vectors);
        assert!(pred.estimated_duration_secs > 0.0);
        assert!(pred.estimated_duration_secs < 600.0); // should be biased toward recent (lower)
        assert_eq!(pred.success_probability, 1.0); // all completed
        assert_eq!(pred.confidence, Confidence::Medium); // 5 runs
        assert!(pred.anomalies.is_empty());
    }

    #[test]
    fn test_predict_run_empty() {
        let pred = predict_run(&[]);
        assert_eq!(pred.confidence, Confidence::Low);
        assert_eq!(pred.success_probability, 0.5);
    }

    #[test]
    fn test_predict_run_anomaly() {
        let vectors = vec![
            make_vector(100.0, 1.0, 5, 0, PlanRunStatus::Completed),
            make_vector(110.0, 1.1, 5, 0, PlanRunStatus::Completed),
            make_vector(105.0, 1.0, 5, 0, PlanRunStatus::Completed),
            make_vector(500.0, 5.0, 5, 0, PlanRunStatus::Completed), // anomaly!
        ];
        let pred = predict_run(&vectors);
        assert!(!pred.anomalies.is_empty());
        assert!(pred
            .anomalies
            .iter()
            .any(|a| a.dimension == "duration" || a.dimension == "cost"));
    }
}
