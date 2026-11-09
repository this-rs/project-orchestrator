//! CpuGuard — circuit breaker for CPU protection.
//!
//! Monitors system CPU usage and pauses heavy operations (training, batch inference)
//! when CPU exceeds 80%, resuming at 50%.
//!
//! Pattern: circuit breaker with hysteresis to avoid oscillation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use sysinfo::System;
use tokio::sync::Notify;
use tracing;

/// CPU usage thresholds for the circuit breaker.
#[derive(Debug, Clone)]
pub struct CpuGuardConfig {
    /// CPU percentage above which to pause (default: 80%).
    pub pause_threshold: f32,
    /// CPU percentage below which to resume (default: 50%).
    pub resume_threshold: f32,
    /// How often to poll CPU usage (default: 2s).
    pub poll_interval: Duration,
}

impl Default for CpuGuardConfig {
    fn default() -> Self {
        Self {
            pause_threshold: 80.0,
            resume_threshold: 50.0,
            poll_interval: Duration::from_secs(2),
        }
    }
}

/// CpuGuard — circuit breaker that pauses work when CPU is overloaded.
///
/// Usage:
/// ```ignore
/// let guard = CpuGuard::new(CpuGuardConfig::default());
/// guard.start_monitoring();
///
/// // Before heavy work:
/// guard.wait_if_paused().await;
/// // ... do heavy work ...
/// ```
#[derive(Clone)]
pub struct CpuGuard {
    config: CpuGuardConfig,
    paused: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl CpuGuard {
    pub fn new(config: CpuGuardConfig) -> Self {
        Self {
            config,
            paused: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Is the guard currently in paused state?
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    /// Wait until the guard is not paused. Returns immediately if not paused.
    pub async fn wait_if_paused(&self) {
        while self.paused.load(Ordering::Relaxed) {
            tracing::debug!(
                "CpuGuard: paused, waiting for CPU to drop below {}%",
                self.config.resume_threshold
            );
            self.notify.notified().await;
        }
    }

    /// Start the background CPU monitoring task.
    /// Returns a JoinHandle that can be used to cancel monitoring.
    pub fn start_monitoring(&self) -> tokio::task::JoinHandle<()> {
        let config = self.config.clone();
        let paused = self.paused.clone();
        let notify = self.notify.clone();

        tokio::spawn(async move {
            // sysinfo 0.40: `System::new()` is fallible. Without a System there is no
            // CPU reading to act on, so say so loudly and stop this monitor.
            let mut sys = match System::new() {
                Ok(sys) => sys,
                Err(err) => {
                    tracing::error!(error = %err, "CpuGuard: sysinfo unavailable, CPU monitoring disabled");
                    return;
                }
            };
            loop {
                tokio::time::sleep(config.poll_interval).await;

                sys.refresh_cpu_all();

                let cpu_usage = sys.global_cpu_usage();
                let was_paused = paused.load(Ordering::Relaxed);

                if !was_paused && cpu_usage > config.pause_threshold {
                    paused.store(true, Ordering::Relaxed);
                    tracing::warn!(
                        cpu_usage = %format!("{:.1}%", cpu_usage),
                        threshold = %format!("{:.0}%", config.pause_threshold),
                        "CpuGuard: PAUSING — CPU above threshold"
                    );
                } else if was_paused && cpu_usage < config.resume_threshold {
                    paused.store(false, Ordering::Relaxed);
                    notify.notify_waiters();
                    tracing::info!(
                        cpu_usage = %format!("{:.1}%", cpu_usage),
                        threshold = %format!("{:.0}%", config.resume_threshold),
                        "CpuGuard: RESUMING — CPU below threshold"
                    );
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_guard_starts_unpaused() {
        let guard = CpuGuard::new(CpuGuardConfig::default());
        assert!(!guard.is_paused());
    }

    #[tokio::test]
    async fn test_wait_if_paused_returns_immediately_when_not_paused() {
        let guard = CpuGuard::new(CpuGuardConfig::default());
        // Should return immediately
        tokio::time::timeout(Duration::from_millis(100), guard.wait_if_paused())
            .await
            .expect("Should not timeout when not paused");
    }

    #[tokio::test]
    async fn test_manual_pause_resume() {
        let guard = CpuGuard::new(CpuGuardConfig::default());

        // Simulate pause
        guard.paused.store(true, Ordering::Relaxed);
        assert!(guard.is_paused());

        // Resume from another task
        let guard_clone = guard.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            guard_clone.paused.store(false, Ordering::Relaxed);
            guard_clone.notify.notify_waiters();
        });

        // wait_if_paused should unblock after the resume
        tokio::time::timeout(Duration::from_secs(1), guard.wait_if_paused())
            .await
            .expect("Should unblock after resume");
    }
}
