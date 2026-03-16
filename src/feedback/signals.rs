//! ImplicitSignalDetector — detects learning signals from system events.
//!
//! Signals are detected from:
//! - CrudEvents (TaskRestarted, PlanCompletedClean)
//! - Git operations (CommitReverted — integration point for HeartbeatEngine)
//! - Note access patterns (NoteHighActivation counter)

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use tracing::debug;
use uuid::Uuid;

use crate::events::{CrudAction, CrudEvent};

use super::models::ImplicitSignal;

// ============================================================================
// Note Access Counter (for NoteHighActivation detection)
// ============================================================================

/// Tracks note access counts within a sliding window for high-activation detection.
#[derive(Debug)]
struct NoteAccessEntry {
    /// Timestamps of recent accesses
    accesses: Vec<DateTime<Utc>>,
}

/// Configuration for high-activation detection.
const HIGH_ACTIVATION_THRESHOLD: usize = 5;
const HIGH_ACTIVATION_WINDOW_SECS: u64 = 3600; // 1 hour

// ============================================================================
// ImplicitSignalDetector
// ============================================================================

/// Detects implicit learning signals from system events and access patterns.
///
/// The detector maintains internal state (access counters, restart counts)
/// and produces `ImplicitSignal` values that feed into the `ScorePropagator`.
#[derive(Debug)]
pub struct ImplicitSignalDetector {
    /// Note access counter: note_id → access timestamps
    note_accesses: Arc<RwLock<HashMap<Uuid, NoteAccessEntry>>>,
    /// Task restart counter: task_id → restart count
    task_restarts: Arc<RwLock<HashMap<Uuid, usize>>>,
}

impl ImplicitSignalDetector {
    pub fn new() -> Self {
        Self {
            note_accesses: Arc::new(RwLock::new(HashMap::new())),
            task_restarts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // ========================================================================
    // CommitReverted detection
    // ========================================================================

    /// Detect a commit revert signal.
    ///
    /// Called by HeartbeatEngine or git hook when a revert is detected.
    /// For now, this is a standalone detection function that can be wired later.
    pub fn detect_commit_reverted(
        commit_id: Uuid,
        revert_commit_id: Option<Uuid>,
    ) -> ImplicitSignal {
        debug!(
            "[signals] Detected commit revert: {} (revert: {:?})",
            commit_id, revert_commit_id
        );
        ImplicitSignal::CommitReverted {
            commit_id,
            revert_commit_id,
        }
    }

    // ========================================================================
    // CrudEvent-based detection
    // ========================================================================

    /// Process a CrudEvent and return any detected implicit signals.
    ///
    /// Detects:
    /// - TaskRestarted: task status changed back to "in_progress" after "completed"/"failed"
    /// - PlanCompletedClean: plan status changed to "completed"
    pub async fn process_crud_event(&self, event: &CrudEvent) -> Vec<ImplicitSignal> {
        let mut signals = Vec::new();

        match (&event.entity_type, &event.action) {
            // Task status updates — detect restarts
            (crate::events::EntityType::Task, CrudAction::Updated) => {
                if let Some(signal) = self.detect_task_restart(event).await {
                    signals.push(signal);
                }
            }
            // Plan completed — detect clean completion
            (crate::events::EntityType::Plan, CrudAction::Updated) => {
                if let Some(signal) = self.detect_plan_completed(event) {
                    signals.push(signal);
                }
            }
            _ => {}
        }

        signals
    }

    /// Detect a task restart from a CrudEvent.
    ///
    /// A restart is when a task transitions from completed/failed back to in_progress.
    async fn detect_task_restart(&self, event: &CrudEvent) -> Option<ImplicitSignal> {
        let new_status = event.payload.get("status")?.as_str()?;
        let old_status = event.payload.get("old_status").and_then(|v| v.as_str());

        // Detect restart: going back to in_progress from completed or failed
        if new_status == "in_progress" && matches!(old_status, Some("completed") | Some("failed")) {
            let task_id = Uuid::parse_str(&event.entity_id).ok()?;
            let mut restarts = self.task_restarts.write().await;
            let count = restarts.entry(task_id).or_insert(0);
            *count += 1;
            let restart_count = *count;

            debug!(
                "[signals] Detected task restart: {} (count: {})",
                task_id, restart_count
            );

            return Some(ImplicitSignal::TaskRestarted {
                task_id,
                restart_count,
            });
        }

        None
    }

    /// Detect plan clean completion from a CrudEvent.
    fn detect_plan_completed(&self, event: &CrudEvent) -> Option<ImplicitSignal> {
        let new_status = event.payload.get("status")?.as_str()?;

        if new_status == "completed" {
            let plan_id = Uuid::parse_str(&event.entity_id).ok()?;
            let task_count = event
                .payload
                .get("task_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            debug!(
                "[signals] Detected plan completed clean: {} ({} tasks)",
                plan_id, task_count
            );

            return Some(ImplicitSignal::PlanCompletedClean {
                plan_id,
                task_count,
            });
        }

        None
    }

    // ========================================================================
    // Note access tracking (NoteHighActivation)
    // ========================================================================

    /// Record a note access and return a signal if high-activation threshold is reached.
    ///
    /// Call this whenever a note is retrieved/used (e.g., in search results, context injection).
    pub async fn record_note_access(&self, note_id: Uuid) -> Option<ImplicitSignal> {
        let now = Utc::now();
        let cutoff = now - chrono::Duration::seconds(HIGH_ACTIVATION_WINDOW_SECS as i64);

        let mut accesses = self.note_accesses.write().await;
        let entry = accesses.entry(note_id).or_insert_with(|| NoteAccessEntry {
            accesses: Vec::new(),
        });

        // Add current access
        entry.accesses.push(now);

        // Prune old accesses outside the window
        entry.accesses.retain(|ts| *ts > cutoff);

        let count = entry.accesses.len();

        if count >= HIGH_ACTIVATION_THRESHOLD {
            debug!(
                "[signals] Detected high activation for note {}: {} accesses in {}s",
                note_id, count, HIGH_ACTIVATION_WINDOW_SECS
            );
            // Reset counter after detection to avoid repeated signals
            entry.accesses.clear();
            return Some(ImplicitSignal::NoteHighActivation {
                note_id,
                access_count: count,
                window_secs: HIGH_ACTIVATION_WINDOW_SECS,
            });
        }

        None
    }

    /// Clean up stale access entries (notes not accessed in the last window).
    /// Call periodically to prevent memory leaks.
    pub async fn cleanup_stale_entries(&self) {
        let cutoff = Utc::now() - chrono::Duration::seconds(HIGH_ACTIVATION_WINDOW_SECS as i64 * 2);

        let mut accesses = self.note_accesses.write().await;
        accesses.retain(|_, entry| {
            entry
                .accesses
                .last()
                .map(|ts| *ts > cutoff)
                .unwrap_or(false)
        });
    }
}

impl Default for ImplicitSignalDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_task_event(task_id: &str, new_status: &str, old_status: Option<&str>) -> CrudEvent {
        let mut payload = json!({ "status": new_status });
        if let Some(old) = old_status {
            payload["old_status"] = json!(old);
        }
        CrudEvent {
            entity_type: crate::events::EntityType::Task,
            action: CrudAction::Updated,
            entity_id: task_id.to_string(),
            related: None,
            payload,
            timestamp: Utc::now().to_rfc3339(),
            project_id: None,
        }
    }

    fn make_plan_event(plan_id: &str, new_status: &str, task_count: usize) -> CrudEvent {
        CrudEvent {
            entity_type: crate::events::EntityType::Plan,
            action: CrudAction::Updated,
            entity_id: plan_id.to_string(),
            related: None,
            payload: json!({ "status": new_status, "task_count": task_count }),
            timestamp: Utc::now().to_rfc3339(),
            project_id: None,
        }
    }

    #[tokio::test]
    async fn test_detect_task_restart() {
        let detector = ImplicitSignalDetector::new();
        let task_id = Uuid::new_v4();

        // First: task goes to in_progress from completed → restart
        let event = make_task_event(&task_id.to_string(), "in_progress", Some("completed"));
        let signals = detector.process_crud_event(&event).await;
        assert_eq!(signals.len(), 1);
        match &signals[0] {
            ImplicitSignal::TaskRestarted {
                task_id: tid,
                restart_count,
            } => {
                assert_eq!(*tid, task_id);
                assert_eq!(*restart_count, 1);
            }
            other => panic!("Expected TaskRestarted, got: {:?}", other),
        }

        // Second restart
        let event2 = make_task_event(&task_id.to_string(), "in_progress", Some("failed"));
        let signals2 = detector.process_crud_event(&event2).await;
        assert_eq!(signals2.len(), 1);
        match &signals2[0] {
            ImplicitSignal::TaskRestarted { restart_count, .. } => {
                assert_eq!(*restart_count, 2);
            }
            other => panic!("Expected TaskRestarted, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_no_restart_on_normal_transition() {
        let detector = ImplicitSignalDetector::new();
        let task_id = Uuid::new_v4();

        // Normal transition: pending → in_progress (not a restart)
        let event = make_task_event(&task_id.to_string(), "in_progress", Some("pending"));
        let signals = detector.process_crud_event(&event).await;
        assert!(signals.is_empty());
    }

    #[tokio::test]
    async fn test_detect_plan_completed() {
        let detector = ImplicitSignalDetector::new();
        let plan_id = Uuid::new_v4();

        let event = make_plan_event(&plan_id.to_string(), "completed", 5);
        let signals = detector.process_crud_event(&event).await;
        assert_eq!(signals.len(), 1);
        match &signals[0] {
            ImplicitSignal::PlanCompletedClean {
                plan_id: pid,
                task_count,
            } => {
                assert_eq!(*pid, plan_id);
                assert_eq!(*task_count, 5);
            }
            other => panic!("Expected PlanCompletedClean, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_note_high_activation() {
        let detector = ImplicitSignalDetector::new();
        let note_id = Uuid::new_v4();

        // Access 4 times — below threshold
        for _ in 0..4 {
            let signal = detector.record_note_access(note_id).await;
            assert!(signal.is_none());
        }

        // 5th access triggers high activation
        let signal = detector.record_note_access(note_id).await;
        assert!(signal.is_some());
        match signal.unwrap() {
            ImplicitSignal::NoteHighActivation {
                note_id: nid,
                access_count,
                ..
            } => {
                assert_eq!(nid, note_id);
                assert_eq!(access_count, 5);
            }
            other => panic!("Expected NoteHighActivation, got: {:?}", other),
        }

        // After detection, counter is reset — next access should not trigger
        let signal = detector.record_note_access(note_id).await;
        assert!(signal.is_none());
    }

    #[test]
    fn test_detect_commit_reverted() {
        let commit_id = Uuid::new_v4();
        let revert_id = Uuid::new_v4();
        let signal = ImplicitSignalDetector::detect_commit_reverted(commit_id, Some(revert_id));
        match signal {
            ImplicitSignal::CommitReverted {
                commit_id: cid,
                revert_commit_id,
            } => {
                assert_eq!(cid, commit_id);
                assert_eq!(revert_commit_id, Some(revert_id));
            }
            other => panic!("Expected CommitReverted, got: {:?}", other),
        }
    }
}
