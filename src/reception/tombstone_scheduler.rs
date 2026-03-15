//! Background scheduler for re-broadcasting unacknowledged tombstones.
//!
//! Tracks which peers have acknowledged a tombstone and provides
//! a list of pending tombstones that need re-broadcast.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// A tombstone that is pending acknowledgement from one or more peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingTombstone {
    /// The content hash being tracked.
    pub content_hash: String,
    /// How many times this tombstone has been broadcast so far.
    pub broadcast_count: u32,
    /// When the last broadcast occurred.
    pub last_broadcast: DateTime<Utc>,
    /// Map of peer DID to whether they have acknowledged.
    pub ack_received: HashMap<String, bool>,
}

/// Scheduler that tracks unacknowledged tombstones and determines
/// which ones need re-broadcasting.
#[derive(Debug, Clone)]
pub struct TombstoneScheduler {
    pending: Arc<RwLock<HashMap<String, PendingTombstone>>>,
    max_retries: u32,
    retry_interval_secs: u64,
}

impl TombstoneScheduler {
    /// Create a new scheduler.
    ///
    /// * `max_retries` — maximum number of broadcast attempts before giving up.
    /// * `retry_interval_secs` — minimum seconds between re-broadcasts.
    pub fn new(max_retries: u32, retry_interval_secs: u64) -> Self {
        Self {
            pending: Arc::new(RwLock::new(HashMap::new())),
            max_retries,
            retry_interval_secs,
        }
    }

    /// Start tracking a tombstone for the given peers.
    ///
    /// Initialises the broadcast count to 1 (the initial broadcast) and
    /// sets `last_broadcast` to now.
    pub async fn track(&self, content_hash: String, peer_dids: Vec<String>) {
        let mut ack_received = HashMap::new();
        for did in peer_dids {
            ack_received.insert(did, false);
        }
        let entry = PendingTombstone {
            content_hash: content_hash.clone(),
            broadcast_count: 1,
            last_broadcast: Utc::now(),
            ack_received,
        };
        self.pending.write().await.insert(content_hash, entry);
    }

    /// Mark a specific peer as having acknowledged a tombstone.
    pub async fn mark_acked(&self, content_hash: &str, peer_did: &str) {
        let mut guard = self.pending.write().await;
        if let Some(entry) = guard.get_mut(content_hash) {
            if let Some(acked) = entry.ack_received.get_mut(peer_did) {
                *acked = true;
            }
            // If all peers acked, remove from pending
            if entry.ack_received.values().all(|&v| v) {
                guard.remove(content_hash);
            }
        }
    }

    /// Return tombstones that need re-broadcasting.
    ///
    /// A tombstone is returned when:
    /// - `broadcast_count < max_retries`
    /// - At least `retry_interval_secs` have elapsed since `last_broadcast`
    pub async fn get_pending(&self) -> Vec<PendingTombstone> {
        let guard = self.pending.read().await;
        let now = Utc::now();
        let interval = chrono::Duration::seconds(self.retry_interval_secs as i64);

        guard
            .values()
            .filter(|entry| {
                entry.broadcast_count < self.max_retries && (now - entry.last_broadcast) >= interval
            })
            .cloned()
            .collect()
    }

    /// Increment the broadcast counter and update `last_broadcast` to now.
    pub async fn increment_broadcast(&self, content_hash: &str) {
        let mut guard = self.pending.write().await;
        if let Some(entry) = guard.get_mut(content_hash) {
            entry.broadcast_count += 1;
            entry.last_broadcast = Utc::now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn track_and_get_pending() {
        // Use 0-second interval so the entry is immediately eligible
        let scheduler = TombstoneScheduler::new(3, 0);
        scheduler
            .track(
                "hash1".to_string(),
                vec!["did:key:alice".to_string(), "did:key:bob".to_string()],
            )
            .await;

        let pending = scheduler.get_pending().await;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].content_hash, "hash1");
        assert_eq!(pending[0].broadcast_count, 1);
        assert_eq!(pending[0].ack_received.len(), 2);
    }

    #[tokio::test]
    async fn mark_acked_partial() {
        let scheduler = TombstoneScheduler::new(3, 0);
        scheduler
            .track(
                "hash2".to_string(),
                vec!["did:key:alice".to_string(), "did:key:bob".to_string()],
            )
            .await;

        scheduler.mark_acked("hash2", "did:key:alice").await;

        // Still pending because bob hasn't acked
        let pending = scheduler.get_pending().await;
        assert_eq!(pending.len(), 1);
        assert!(*pending[0].ack_received.get("did:key:alice").unwrap());
        assert!(!*pending[0].ack_received.get("did:key:bob").unwrap());
    }

    #[tokio::test]
    async fn mark_acked_all_removes_entry() {
        let scheduler = TombstoneScheduler::new(3, 0);
        scheduler
            .track("hash3".to_string(), vec!["did:key:alice".to_string()])
            .await;

        scheduler.mark_acked("hash3", "did:key:alice").await;

        // Fully acked — should be removed
        let pending = scheduler.get_pending().await;
        assert!(pending.is_empty());
    }

    #[tokio::test]
    async fn increment_broadcast_updates_count() {
        let scheduler = TombstoneScheduler::new(5, 0);
        scheduler
            .track("hash4".to_string(), vec!["did:key:alice".to_string()])
            .await;

        scheduler.increment_broadcast("hash4").await;

        let pending = scheduler.get_pending().await;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].broadcast_count, 2);
    }

    #[tokio::test]
    async fn max_retries_excludes_exhausted() {
        let scheduler = TombstoneScheduler::new(2, 0);
        scheduler
            .track("hash5".to_string(), vec!["did:key:alice".to_string()])
            .await;

        // broadcast_count starts at 1, increment to 2 (== max_retries)
        scheduler.increment_broadcast("hash5").await;

        // Now broadcast_count == max_retries, should NOT be returned
        let pending = scheduler.get_pending().await;
        assert!(pending.is_empty());
    }

    #[tokio::test]
    async fn retry_interval_filters_recent() {
        // 1-hour interval — entry just created won't be eligible
        let scheduler = TombstoneScheduler::new(3, 3600);
        scheduler
            .track("hash6".to_string(), vec!["did:key:alice".to_string()])
            .await;

        let pending = scheduler.get_pending().await;
        assert!(pending.is_empty());
    }

    #[tokio::test]
    async fn mark_acked_unknown_hash_is_noop() {
        let scheduler = TombstoneScheduler::new(3, 0);
        // Should not panic
        scheduler.mark_acked("nonexistent", "did:key:alice").await;
    }

    #[tokio::test]
    async fn increment_broadcast_unknown_hash_is_noop() {
        let scheduler = TombstoneScheduler::new(3, 0);
        // Should not panic
        scheduler.increment_broadcast("nonexistent").await;
    }

    #[test]
    fn pending_tombstone_serialization_roundtrip() {
        let entry = PendingTombstone {
            content_hash: "hash_ser".to_string(),
            broadcast_count: 2,
            last_broadcast: Utc::now(),
            ack_received: HashMap::from([
                ("did:key:alice".to_string(), true),
                ("did:key:bob".to_string(), false),
            ]),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let deser: PendingTombstone = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.content_hash, "hash_ser");
        assert_eq!(deser.broadcast_count, 2);
        assert_eq!(deser.ack_received.len(), 2);
    }
}
