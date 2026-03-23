//! Circuit breaker for external MCP server connections.
//!
//! Three states: **Closed** (normal) → **Open** (rejecting) → **HalfOpen** (probe).
//!
//! - Closed → Open: when error rate exceeds threshold over the last N calls.
//! - Open → HalfOpen: after a cooldown period.
//! - HalfOpen → Closed: if the next call succeeds.
//! - HalfOpen → Open: if the next call fails.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Duration;

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker for a single MCP server.
#[derive(Debug)]
pub struct CircuitBreaker {
    state: CircuitState,
    /// Rolling window of recent call outcomes (true = success, false = failure).
    window: VecDeque<bool>,
    /// Maximum window size.
    window_size: usize,
    /// Error rate threshold (0.0–1.0) to trip the breaker.
    error_threshold: f64,
    /// How long to wait in Open state before transitioning to HalfOpen.
    cooldown: Duration,
    /// When the breaker was last opened.
    opened_at: Option<DateTime<Utc>>,
    /// Total lifetime stats.
    total_calls: u64,
    total_errors: u64,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with default settings.
    pub fn new() -> Self {
        Self::with_config(20, 0.5, Duration::from_secs(30))
    }

    /// Create with custom configuration.
    pub fn with_config(window_size: usize, error_threshold: f64, cooldown: Duration) -> Self {
        Self {
            state: CircuitState::Closed,
            window: VecDeque::with_capacity(window_size),
            window_size,
            error_threshold,
            cooldown,
            opened_at: None,
            total_calls: 0,
            total_errors: 0,
        }
    }

    /// Check if a request is allowed through.
    pub fn allow_request(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if cooldown has elapsed
                if let Some(opened_at) = self.opened_at {
                    let elapsed = Utc::now()
                        .signed_duration_since(opened_at)
                        .to_std()
                        .unwrap_or(Duration::ZERO);
                    if elapsed >= self.cooldown {
                        self.state = CircuitState::HalfOpen;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => {
                // Allow exactly one probe request
                true
            }
        }
    }

    /// Record a successful call.
    pub fn record_success(&mut self) {
        self.total_calls += 1;
        self.push_outcome(true);

        match self.state {
            CircuitState::HalfOpen => {
                // Probe succeeded → close the breaker
                self.state = CircuitState::Closed;
                self.opened_at = None;
            }
            CircuitState::Closed => {
                // Stay closed
            }
            CircuitState::Open => {
                // Shouldn't happen (we block in Open), but handle gracefully
            }
        }
    }

    /// Record a failed call.
    pub fn record_failure(&mut self) {
        self.total_calls += 1;
        self.total_errors += 1;
        self.push_outcome(false);

        match self.state {
            CircuitState::HalfOpen => {
                // Probe failed → reopen
                self.state = CircuitState::Open;
                self.opened_at = Some(Utc::now());
            }
            CircuitState::Closed => {
                // Check if we should trip
                if self.should_trip() {
                    self.state = CircuitState::Open;
                    self.opened_at = Some(Utc::now());
                }
            }
            CircuitState::Open => {
                // Already open
            }
        }
    }

    /// Get the current state.
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Get the current error rate over the window.
    pub fn error_rate(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let errors = self.window.iter().filter(|&&ok| !ok).count();
        errors as f64 / self.window.len() as f64
    }

    /// Get total lifetime stats.
    pub fn stats(&self) -> (u64, u64) {
        (self.total_calls, self.total_errors)
    }

    fn push_outcome(&mut self, success: bool) {
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(success);
    }

    fn should_trip(&self) -> bool {
        // Need at least half the window filled before tripping
        if self.window.len() < self.window_size / 2 {
            return false;
        }
        self.error_rate() > self.error_threshold
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_breaker_is_closed() {
        let cb = CircuitBreaker::new();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.error_rate(), 0.0);
    }

    #[test]
    fn test_closed_allows_requests() {
        let mut cb = CircuitBreaker::new();
        assert!(cb.allow_request());
    }

    #[test]
    fn test_stays_closed_on_success() {
        let mut cb = CircuitBreaker::new();
        for _ in 0..20 {
            cb.record_success();
        }
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.error_rate(), 0.0);
    }

    #[test]
    fn test_trips_on_high_error_rate() {
        let mut cb = CircuitBreaker::with_config(20, 0.5, Duration::from_secs(30));

        // 10 successes, then 11 failures → error rate > 50%
        for _ in 0..10 {
            cb.record_success();
        }
        assert_eq!(cb.state(), CircuitState::Closed);

        for _ in 0..11 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_open_rejects_requests() {
        let mut cb = CircuitBreaker::with_config(10, 0.5, Duration::from_secs(3600));

        // Trip the breaker
        for _ in 0..10 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request()); // Should be rejected
    }

    #[test]
    fn test_half_open_after_cooldown() {
        let mut cb = CircuitBreaker::with_config(10, 0.5, Duration::from_millis(1));

        // Trip the breaker
        for _ in 0..10 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for cooldown
        std::thread::sleep(Duration::from_millis(5));
        assert!(cb.allow_request());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_half_open_to_closed_on_success() {
        let mut cb = CircuitBreaker::with_config(10, 0.5, Duration::from_millis(1));

        for _ in 0..10 {
            cb.record_failure();
        }
        std::thread::sleep(Duration::from_millis(5));
        cb.allow_request(); // Transitions to HalfOpen

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_half_open_to_open_on_failure() {
        let mut cb = CircuitBreaker::with_config(10, 0.5, Duration::from_millis(1));

        for _ in 0..10 {
            cb.record_failure();
        }
        std::thread::sleep(Duration::from_millis(5));
        cb.allow_request(); // Transitions to HalfOpen

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_doesnt_trip_with_insufficient_data() {
        let mut cb = CircuitBreaker::with_config(20, 0.5, Duration::from_secs(30));

        // Only 5 failures (less than window_size/2 = 10)
        for _ in 0..5 {
            cb.record_failure();
        }
        // Should NOT trip — not enough data
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_stats() {
        let mut cb = CircuitBreaker::new();
        cb.record_success();
        cb.record_success();
        cb.record_failure();

        let (total, errors) = cb.stats();
        assert_eq!(total, 3);
        assert_eq!(errors, 1);
    }
}
