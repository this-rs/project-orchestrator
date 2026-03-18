//! DriftDetector — monitors reward distribution for non-stationarity.
//!
//! Detects concept drift when the LLM changes, the knowledge graph evolves,
//! or user behavior shifts. Uses:
//! - **Page-Hinkley Test**: sequential change detection on reward mean
//! - **KS-Test (Kolmogorov-Smirnov)**: sliding window distribution comparison
//! - **Action distribution shift**: KL divergence on action frequencies
//!
//! When drift is detected, triggers model rollback or increased exploration.

use std::collections::{HashMap, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// DriftDetector configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftConfig {
    /// Window size for recent vs reference comparison.
    pub window_size: usize,
    /// Page-Hinkley threshold (λ).
    pub page_hinkley_threshold: f64,
    /// Page-Hinkley minimum magnitude (δ).
    pub page_hinkley_delta: f64,
    /// KL divergence threshold for action distribution shift.
    pub kl_threshold: f64,
    /// KS-test significance level (alpha).
    pub ks_alpha: f64,
    /// Cooldown period after a drift event (in number of observations).
    pub cooldown_observations: usize,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            window_size: 200,
            page_hinkley_threshold: 50.0,
            page_hinkley_delta: 0.005,
            kl_threshold: 0.5,
            ks_alpha: 0.05,
            cooldown_observations: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Drift events
// ---------------------------------------------------------------------------

/// Type of drift detected.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DriftType {
    /// Mean reward shifted (Page-Hinkley).
    RewardMeanShift,
    /// Reward distribution changed (KS-test).
    RewardDistributionShift,
    /// Action usage pattern changed (KL divergence).
    ActionDistributionShift,
}

/// A drift event with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftEvent {
    /// Type of drift detected.
    pub drift_type: DriftType,
    /// Severity score (0.0 - 1.0).
    pub severity: f64,
    /// Statistical test value.
    pub test_statistic: f64,
    /// Threshold that was exceeded.
    pub threshold: f64,
    /// When the drift was detected.
    pub detected_at: DateTime<Utc>,
    /// Number of observations at detection time.
    pub observation_count: usize,
}

/// Recommended action after drift detection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DriftAction {
    /// No drift detected — continue normal operation.
    None,
    /// Mild drift — increase exploration (ε).
    IncreaseExploration,
    /// Significant drift — trigger model retraining.
    TriggerRetrain,
    /// Severe drift — rollback to previous model.
    Rollback,
}

// ---------------------------------------------------------------------------
// DriftDetector
// ---------------------------------------------------------------------------

/// DriftDetector — monitors reward and action distributions for concept drift.
pub struct DriftDetector {
    config: DriftConfig,

    // Page-Hinkley state (two-sided: detect both upward and downward mean shifts)
    ph_sum_up: f64, // Detects upward shift
    ph_min_up: f64,
    ph_sum_down: f64, // Detects downward shift
    ph_max_down: f64,
    ph_mean: f64,
    ph_count: usize,

    // Sliding windows for KS-test
    reference_window: VecDeque<f64>,
    recent_window: VecDeque<f64>,

    // Action distribution tracking
    reference_actions: HashMap<String, usize>,
    recent_actions: HashMap<String, usize>,
    reference_action_total: usize,
    recent_action_total: usize,

    // Cooldown
    observations_since_drift: usize,

    // History
    drift_events: Vec<DriftEvent>,
}

impl DriftDetector {
    pub fn new(config: DriftConfig) -> Self {
        Self {
            reference_window: VecDeque::with_capacity(config.window_size),
            recent_window: VecDeque::with_capacity(config.window_size),
            config,
            ph_sum_up: 0.0,
            ph_min_up: 0.0,
            ph_sum_down: 0.0,
            ph_max_down: 0.0,
            ph_mean: 0.0,
            ph_count: 0,
            reference_actions: HashMap::new(),
            recent_actions: HashMap::new(),
            reference_action_total: 0,
            recent_action_total: 0,
            observations_since_drift: usize::MAX, // No cooldown initially
            drift_events: Vec::new(),
        }
    }

    /// Observe a new reward value.
    pub fn observe_reward(&mut self, reward: f64) {
        self.ph_count += 1;
        self.observations_since_drift = self.observations_since_drift.saturating_add(1);

        // Update Page-Hinkley (two-sided)
        self.ph_mean += (reward - self.ph_mean) / self.ph_count as f64;

        // Upward shift detection: cumsum of (x - mean - delta)
        self.ph_sum_up += reward - self.ph_mean - self.config.page_hinkley_delta;
        if self.ph_sum_up < self.ph_min_up {
            self.ph_min_up = self.ph_sum_up;
        }

        // Downward shift detection: cumsum of (mean - x - delta)
        self.ph_sum_down += self.ph_mean - reward - self.config.page_hinkley_delta;
        if self.ph_sum_down < self.ph_max_down {
            self.ph_max_down = self.ph_sum_down;
        }

        // Update sliding windows
        self.recent_window.push_back(reward);
        if self.recent_window.len() > self.config.window_size {
            let popped = self.recent_window.pop_front().unwrap();
            self.reference_window.push_back(popped);
            if self.reference_window.len() > self.config.window_size {
                self.reference_window.pop_front();
            }
        }
    }

    /// Observe an action taken.
    pub fn observe_action(&mut self, action_key: &str) {
        // Shift recent to reference periodically
        if self.recent_action_total > 0
            && self
                .recent_action_total
                .is_multiple_of(self.config.window_size)
        {
            self.reference_actions = self.recent_actions.clone();
            self.reference_action_total = self.recent_action_total;
            self.recent_actions.clear();
            self.recent_action_total = 0;
        }

        *self
            .recent_actions
            .entry(action_key.to_string())
            .or_insert(0) += 1;
        self.recent_action_total += 1;
    }

    /// Check for drift using all detectors.
    ///
    /// Returns the recommended action and any detected drift events.
    pub fn check(&mut self) -> (DriftAction, Vec<DriftEvent>) {
        if self.observations_since_drift < self.config.cooldown_observations {
            return (DriftAction::None, vec![]);
        }

        let mut events = Vec::new();
        let mut worst_action = DriftAction::None;

        // 1. Page-Hinkley Test
        if let Some(event) = self.check_page_hinkley() {
            let action = if event.severity > 0.8 {
                DriftAction::Rollback
            } else if event.severity > 0.5 {
                DriftAction::TriggerRetrain
            } else {
                DriftAction::IncreaseExploration
            };
            if action_priority(action) > action_priority(worst_action) {
                worst_action = action;
            }
            events.push(event);
        }

        // 2. KS-Test
        if let Some(event) = self.check_ks_test() {
            let action = if event.severity > 0.7 {
                DriftAction::TriggerRetrain
            } else {
                DriftAction::IncreaseExploration
            };
            if action_priority(action) > action_priority(worst_action) {
                worst_action = action;
            }
            events.push(event);
        }

        // 3. Action distribution KL divergence
        if let Some(event) = self.check_action_kl() {
            let action = DriftAction::IncreaseExploration;
            if action_priority(action) > action_priority(worst_action) {
                worst_action = action;
            }
            events.push(event);
        }

        if !events.is_empty() {
            self.observations_since_drift = 0;
            self.drift_events.extend(events.clone());
        }

        (worst_action, events)
    }

    /// Page-Hinkley Test for mean shift detection (two-sided).
    fn check_page_hinkley(&self) -> Option<DriftEvent> {
        // Upward shift: sum - min > threshold
        let ph_stat_up = self.ph_sum_up - self.ph_min_up;
        // Downward shift: sum - max > threshold (where max tracks the running minimum of the downward cumsum)
        let ph_stat_down = self.ph_sum_down - self.ph_max_down;

        let ph_stat = ph_stat_up.max(ph_stat_down);
        if ph_stat > self.config.page_hinkley_threshold {
            let severity = (ph_stat / self.config.page_hinkley_threshold).clamp(0.0, 1.0);
            Some(DriftEvent {
                drift_type: DriftType::RewardMeanShift,
                severity,
                test_statistic: ph_stat,
                threshold: self.config.page_hinkley_threshold,
                detected_at: Utc::now(),
                observation_count: self.ph_count,
            })
        } else {
            None
        }
    }

    /// Two-sample Kolmogorov-Smirnov test on reward windows.
    fn check_ks_test(&self) -> Option<DriftEvent> {
        if self.reference_window.len() < 30 || self.recent_window.len() < 30 {
            return None;
        }

        let mut ref_sorted: Vec<f64> = self.reference_window.iter().copied().collect();
        let mut rec_sorted: Vec<f64> = self.recent_window.iter().copied().collect();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rec_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n1 = ref_sorted.len() as f64;
        let n2 = rec_sorted.len() as f64;

        // Compute KS statistic: max |F1(x) - F2(x)|
        let mut ks_stat = 0.0f64;
        let mut i = 0usize;
        let mut j = 0usize;

        while i < ref_sorted.len() && j < rec_sorted.len() {
            if ref_sorted[i] < rec_sorted[j] {
                i += 1;
            } else if rec_sorted[j] < ref_sorted[i] {
                j += 1;
            } else {
                // Tied values: advance both pointers
                i += 1;
                j += 1;
            }
            let diff = (i as f64 / n1 - j as f64 / n2).abs();
            if diff > ks_stat {
                ks_stat = diff;
            }
        }

        // Handle remaining elements from reference
        while i < ref_sorted.len() {
            let diff = ((i + 1) as f64 / n1 - 1.0).abs(); // j has exhausted n2 → F2 = 1.0
            if diff > ks_stat {
                ks_stat = diff;
            }
            i += 1;
        }
        // Handle remaining elements from recent
        while j < rec_sorted.len() {
            let diff = (1.0 - (j + 1) as f64 / n2).abs(); // i has exhausted n1 → F1 = 1.0
            if diff > ks_stat {
                ks_stat = diff;
            }
            j += 1;
        }

        // Critical value: c(α) × √((n1 + n2) / (n1 × n2))
        // c(0.05) ≈ 1.358
        let c_alpha = match self.config.ks_alpha {
            a if a <= 0.01 => 1.628,
            a if a <= 0.05 => 1.358,
            a if a <= 0.10 => 1.224,
            _ => 1.073,
        };
        let critical_value = c_alpha * ((n1 + n2) / (n1 * n2)).sqrt();

        if ks_stat > critical_value {
            let severity = (ks_stat / critical_value - 1.0).clamp(0.0, 1.0);
            Some(DriftEvent {
                drift_type: DriftType::RewardDistributionShift,
                severity,
                test_statistic: ks_stat,
                threshold: critical_value,
                detected_at: Utc::now(),
                observation_count: self.ph_count,
            })
        } else {
            None
        }
    }

    /// KL divergence between reference and recent action distributions.
    fn check_action_kl(&self) -> Option<DriftEvent> {
        if self.reference_action_total < 50 || self.recent_action_total < 50 {
            return None;
        }

        // Collect all action keys
        let mut all_keys: Vec<String> = self.reference_actions.keys().cloned().collect();
        for key in self.recent_actions.keys() {
            if !all_keys.contains(key) {
                all_keys.push(key.clone());
            }
        }

        // Compute KL(recent || reference) with Laplace smoothing
        let smoothing = 1.0;
        let ref_total = self.reference_action_total as f64 + smoothing * all_keys.len() as f64;
        let rec_total = self.recent_action_total as f64 + smoothing * all_keys.len() as f64;

        let mut kl = 0.0f64;
        for key in &all_keys {
            let p = (*self.recent_actions.get(key).unwrap_or(&0) as f64 + smoothing) / rec_total;
            let q = (*self.reference_actions.get(key).unwrap_or(&0) as f64 + smoothing) / ref_total;
            kl += p * (p / q).ln();
        }

        if kl > self.config.kl_threshold {
            let severity = (kl / self.config.kl_threshold - 1.0).clamp(0.0, 1.0);
            Some(DriftEvent {
                drift_type: DriftType::ActionDistributionShift,
                severity,
                test_statistic: kl,
                threshold: self.config.kl_threshold,
                detected_at: Utc::now(),
                observation_count: self.ph_count,
            })
        } else {
            None
        }
    }

    /// Reset the detector state (after model rollback or retrain).
    pub fn reset(&mut self) {
        self.ph_sum_up = 0.0;
        self.ph_min_up = 0.0;
        self.ph_sum_down = 0.0;
        self.ph_max_down = 0.0;
        self.ph_mean = 0.0;
        self.ph_count = 0;
        self.reference_window.clear();
        self.recent_window.clear();
        self.reference_actions.clear();
        self.recent_actions.clear();
        self.reference_action_total = 0;
        self.recent_action_total = 0;
        self.observations_since_drift = usize::MAX;
    }

    /// Get the history of drift events.
    pub fn drift_events(&self) -> &[DriftEvent] {
        &self.drift_events
    }

    /// Get the current Page-Hinkley statistic (max of upward and downward).
    pub fn page_hinkley_stat(&self) -> f64 {
        let up = self.ph_sum_up - self.ph_min_up;
        let down = self.ph_sum_down - self.ph_max_down;
        up.max(down)
    }

    /// Get the running mean of observed rewards.
    pub fn reward_mean(&self) -> f64 {
        self.ph_mean
    }

    /// Total observations processed.
    pub fn observation_count(&self) -> usize {
        self.ph_count
    }
}

fn action_priority(action: DriftAction) -> u8 {
    match action {
        DriftAction::None => 0,
        DriftAction::IncreaseExploration => 1,
        DriftAction::TriggerRetrain => 2,
        DriftAction::Rollback => 3,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_drift_stable_rewards() {
        let mut detector = DriftDetector::new(DriftConfig::default());

        // Feed stable rewards
        for _ in 0..500 {
            detector.observe_reward(0.8);
        }

        let (action, events) = detector.check();
        assert_eq!(action, DriftAction::None);
        assert!(events.is_empty());
    }

    #[test]
    fn test_page_hinkley_detects_mean_shift() {
        let mut detector = DriftDetector::new(DriftConfig {
            page_hinkley_threshold: 20.0,
            page_hinkley_delta: 0.005,
            window_size: 100,
            cooldown_observations: 0,
            ..Default::default()
        });

        // Stable period
        for _ in 0..200 {
            detector.observe_reward(0.8);
        }

        // Mean shift down
        for _ in 0..200 {
            detector.observe_reward(0.3);
        }

        let (action, events) = detector.check();
        assert_ne!(
            action,
            DriftAction::None,
            "Should detect drift after mean shift"
        );
        assert!(
            events
                .iter()
                .any(|e| e.drift_type == DriftType::RewardMeanShift),
            "Should have RewardMeanShift event"
        );
    }

    #[test]
    fn test_ks_test_detects_distribution_shift() {
        let mut detector = DriftDetector::new(DriftConfig {
            window_size: 100,
            cooldown_observations: 0,
            ks_alpha: 0.05,
            ..Default::default()
        });

        // Reference: N(0.8, 0.1)
        for i in 0..200 {
            let reward = 0.8 + 0.1 * ((i as f64 * 0.1).sin());
            detector.observe_reward(reward);
        }

        // Shift: N(0.3, 0.1) — very different
        for i in 0..200 {
            let reward = 0.3 + 0.1 * ((i as f64 * 0.1).sin());
            detector.observe_reward(reward);
        }

        let (action, events) = detector.check();
        // Should detect some form of drift
        let has_drift = events.iter().any(|e| {
            e.drift_type == DriftType::RewardDistributionShift
                || e.drift_type == DriftType::RewardMeanShift
        });
        assert!(
            has_drift,
            "Should detect distribution shift. Action: {:?}, Events: {:?}",
            action, events
        );
    }

    #[test]
    fn test_action_kl_detects_shift() {
        let mut detector = DriftDetector::new(DriftConfig {
            window_size: 100,
            kl_threshold: 0.3,
            cooldown_observations: 0,
            ..Default::default()
        });

        // Reference period: mostly code.search
        for _ in 0..100 {
            detector.observe_action("code.search");
            detector.observe_reward(0.8);
        }

        // Shift: mostly note.create (very different distribution)
        for _ in 0..100 {
            detector.observe_action("note.create");
            detector.observe_reward(0.8);
        }

        let (_, events) = detector.check();
        let has_action_drift = events
            .iter()
            .any(|e| e.drift_type == DriftType::ActionDistributionShift);
        assert!(
            has_action_drift,
            "Should detect action distribution shift. Events: {:?}",
            events
        );
    }

    #[test]
    fn test_cooldown_suppresses_events() {
        let mut detector = DriftDetector::new(DriftConfig {
            page_hinkley_threshold: 10.0,
            page_hinkley_delta: 0.005,
            cooldown_observations: 500,
            window_size: 50,
            ..Default::default()
        });

        // Trigger drift
        for _ in 0..100 {
            detector.observe_reward(0.8);
        }
        for _ in 0..100 {
            detector.observe_reward(0.1);
        }

        let (action1, events1) = detector.check();
        // First check: should detect drift
        assert!(!events1.is_empty() || action1 != DriftAction::None);

        // Immediately check again: should be suppressed by cooldown
        let (action2, events2) = detector.check();
        assert_eq!(action2, DriftAction::None);
        assert!(events2.is_empty());
    }

    #[test]
    fn test_reset() {
        let mut detector = DriftDetector::new(DriftConfig::default());

        for _ in 0..100 {
            detector.observe_reward(0.8);
        }
        assert_eq!(detector.observation_count(), 100);

        detector.reset();
        assert_eq!(detector.observation_count(), 0);
        assert_eq!(detector.page_hinkley_stat(), 0.0);
    }

    #[test]
    fn test_drift_event_history() {
        let mut detector = DriftDetector::new(DriftConfig {
            page_hinkley_threshold: 5.0,
            page_hinkley_delta: 0.005,
            cooldown_observations: 0,
            window_size: 50,
            ..Default::default()
        });

        for _ in 0..100 {
            detector.observe_reward(0.9);
        }
        for _ in 0..100 {
            detector.observe_reward(0.1);
        }

        let _ = detector.check();
        assert!(
            !detector.drift_events().is_empty(),
            "Should have drift event history"
        );
    }
}
