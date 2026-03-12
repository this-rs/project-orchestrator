use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Vector clock
// ---------------------------------------------------------------------------

/// A vector clock where each entry is keyed by a DID (`did:key:…`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VectorClock {
    pub clocks: HashMap<String, u64>,
}

/// Result of comparing two vector clocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockOrdering {
    Before,
    After,
    Concurrent,
    Equal,
}

impl VectorClock {
    pub fn new() -> Self {
        Self {
            clocks: HashMap::new(),
        }
    }

    /// Increment the counter for the given DID (creates the entry if absent).
    pub fn increment(&mut self, did: &str) {
        let counter = self.clocks.entry(did.to_string()).or_insert(0);
        *counter += 1;
    }

    /// Merge another clock into this one by taking the element-wise maximum.
    pub fn merge(&mut self, other: &VectorClock) {
        for (did, &their_val) in &other.clocks {
            let ours = self.clocks.entry(did.clone()).or_insert(0);
            if their_val > *ours {
                *ours = their_val;
            }
        }
    }

    /// Compare this clock with `other`.
    ///
    /// - `Equal`      — every entry is identical.
    /// - `Before`     — every entry in `self` ≤ `other`, and at least one is <.
    /// - `After`      — every entry in `self` ≥ `other`, and at least one is >.
    /// - `Concurrent` — neither dominates.
    pub fn compare(&self, other: &VectorClock) -> ClockOrdering {
        let all_keys: HashSet<&String> =
            self.clocks.keys().chain(other.clocks.keys()).collect();

        let mut has_less = false;
        let mut has_greater = false;

        for key in all_keys {
            let a = self.clocks.get(key).copied().unwrap_or(0);
            let b = other.clocks.get(key).copied().unwrap_or(0);
            if a < b {
                has_less = true;
            }
            if a > b {
                has_greater = true;
            }
            if has_less && has_greater {
                return ClockOrdering::Concurrent;
            }
        }

        match (has_less, has_greater) {
            (false, false) => ClockOrdering::Equal,
            (true, false) => ClockOrdering::Before,
            (false, true) => ClockOrdering::After,
            (true, true) => ClockOrdering::Concurrent, // unreachable, handled above
        }
    }

    pub fn is_before(&self, other: &VectorClock) -> bool {
        self.compare(other) == ClockOrdering::Before
    }

    pub fn is_after(&self, other: &VectorClock) -> bool {
        self.compare(other) == ClockOrdering::After
    }

    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        self.compare(other) == ClockOrdering::Concurrent
    }
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Sync protocol messages
// ---------------------------------------------------------------------------

/// Messages exchanged during a delta-sync session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncMessage {
    /// "Here's what I have" — initiator sends its clock + content hashes.
    Have {
        vector_clock: VectorClock,
        content_hashes: HashSet<String>,
    },
    /// "Please send me these" — responder asks for missing content.
    Want {
        content_hashes: HashSet<String>,
    },
    /// Requested content payload (generic JSON envelopes for now).
    Data {
        envelopes: Vec<serde_json::Value>,
    },
}

// ---------------------------------------------------------------------------
// Sync session state machine
// ---------------------------------------------------------------------------

/// Tracks the state of a single sync exchange.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncState {
    Idle,
    WaitingForWant,
    WaitingForData,
    Complete,
}

/// Drives a two-phase sync session between two peers.
///
/// Flow:
///   1. Initiator calls `initiate()` → sends `Have`.
///   2. Responder calls `process_have(have)` → sends `Want`.
///   3. Initiator calls `process_want(want)` → sends `Data`.
///   4. Responder calls `process_data(data)` → receives new items.
#[derive(Debug, Clone)]
pub struct SyncSession {
    pub local_clock: VectorClock,
    pub local_hashes: HashSet<String>,
    pub state: SyncState,
    /// Local content store, keyed by content hash.
    local_store: HashMap<String, serde_json::Value>,
}

impl SyncSession {
    /// Create a session pre-loaded with local content.
    pub fn new(
        local_clock: VectorClock,
        local_items: HashMap<String, serde_json::Value>,
    ) -> Self {
        let local_hashes: HashSet<String> = local_items.keys().cloned().collect();
        Self {
            local_clock,
            local_hashes,
            state: SyncState::Idle,
            local_store: local_items,
        }
    }

    /// Step 1 (initiator): produce a `Have` message advertising local state.
    pub fn initiate(&mut self) -> SyncMessage {
        self.state = SyncState::WaitingForWant;
        SyncMessage::Have {
            vector_clock: self.local_clock.clone(),
            content_hashes: self.local_hashes.clone(),
        }
    }

    /// Step 2 (responder): given a remote `Have`, determine which hashes we
    /// are missing and return a `Want` message.
    pub fn process_have(&mut self, their_have: SyncMessage) -> SyncMessage {
        if let SyncMessage::Have {
            content_hashes: their_hashes,
            ..
        } = their_have
        {
            let missing: HashSet<String> = their_hashes
                .difference(&self.local_hashes)
                .cloned()
                .collect();
            self.state = SyncState::WaitingForData;
            SyncMessage::Want {
                content_hashes: missing,
            }
        } else {
            // Unexpected message kind — ask for nothing.
            SyncMessage::Want {
                content_hashes: HashSet::new(),
            }
        }
    }

    /// Step 3 (initiator): fulfill a `Want` request by returning matching
    /// local content.
    pub fn process_want(&mut self, their_want: SyncMessage) -> SyncMessage {
        let envelopes = if let SyncMessage::Want {
            content_hashes: wanted,
        } = their_want
        {
            wanted
                .iter()
                .filter_map(|hash| self.local_store.get(hash).cloned())
                .collect()
        } else {
            Vec::new()
        };
        self.state = SyncState::Complete;
        SyncMessage::Data { envelopes }
    }

    /// Step 4 (responder): ingest received data and return the new items.
    pub fn process_data(&mut self, data: SyncMessage) -> Vec<serde_json::Value> {
        self.state = SyncState::Complete;
        if let SyncMessage::Data { envelopes } = data {
            envelopes
        } else {
            Vec::new()
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // VectorClock tests
    // -----------------------------------------------------------------------

    #[test]
    fn vector_clock_increment() {
        let mut vc = VectorClock::new();
        vc.increment("did:key:alice");
        vc.increment("did:key:alice");
        vc.increment("did:key:bob");
        assert_eq!(vc.clocks["did:key:alice"], 2);
        assert_eq!(vc.clocks["did:key:bob"], 1);
    }

    #[test]
    fn vector_clock_merge() {
        let mut a = VectorClock::new();
        a.increment("did:key:alice"); // alice=1
        a.increment("did:key:alice"); // alice=2

        let mut b = VectorClock::new();
        b.increment("did:key:alice"); // alice=1
        b.increment("did:key:bob"); // bob=1
        b.increment("did:key:bob"); // bob=2
        b.increment("did:key:bob"); // bob=3

        a.merge(&b);
        assert_eq!(a.clocks["did:key:alice"], 2); // max(2,1)
        assert_eq!(a.clocks["did:key:bob"], 3); // max(0,3)
    }

    #[test]
    fn vector_clock_equal() {
        let mut a = VectorClock::new();
        a.increment("did:key:alice");
        let mut b = VectorClock::new();
        b.increment("did:key:alice");
        assert_eq!(a.compare(&b), ClockOrdering::Equal);
    }

    #[test]
    fn vector_clock_before() {
        let mut a = VectorClock::new();
        a.increment("did:key:alice");

        let mut b = VectorClock::new();
        b.increment("did:key:alice");
        b.increment("did:key:alice");

        assert!(a.is_before(&b));
        assert!(!a.is_after(&b));
        assert!(!a.is_concurrent(&b));
    }

    #[test]
    fn vector_clock_after() {
        let mut a = VectorClock::new();
        a.increment("did:key:alice");
        a.increment("did:key:alice");
        a.increment("did:key:bob");

        let mut b = VectorClock::new();
        b.increment("did:key:alice");

        assert!(a.is_after(&b));
        assert_eq!(a.compare(&b), ClockOrdering::After);
    }

    #[test]
    fn vector_clock_concurrent() {
        let mut a = VectorClock::new();
        a.increment("did:key:alice");

        let mut b = VectorClock::new();
        b.increment("did:key:bob");

        assert!(a.is_concurrent(&b));
        assert_eq!(a.compare(&b), ClockOrdering::Concurrent);
    }

    #[test]
    fn vector_clock_empty_equal() {
        let a = VectorClock::new();
        let b = VectorClock::new();
        assert_eq!(a.compare(&b), ClockOrdering::Equal);
    }

    #[test]
    fn vector_clock_empty_before_non_empty() {
        let a = VectorClock::new();
        let mut b = VectorClock::new();
        b.increment("did:key:bob");
        assert!(a.is_before(&b));
    }

    // -----------------------------------------------------------------------
    // SyncSession tests
    // -----------------------------------------------------------------------

    #[test]
    fn sync_full_flow() {
        // Peer A has items 1, 2, 3
        let mut clock_a = VectorClock::new();
        clock_a.increment("did:key:a");
        clock_a.increment("did:key:a");
        clock_a.increment("did:key:a");
        let items_a: HashMap<String, serde_json::Value> = [
            ("hash1".into(), serde_json::json!({"id": 1})),
            ("hash2".into(), serde_json::json!({"id": 2})),
            ("hash3".into(), serde_json::json!({"id": 3})),
        ]
        .into_iter()
        .collect();

        // Peer B has items 2, 3, 4
        let mut clock_b = VectorClock::new();
        clock_b.increment("did:key:b");
        clock_b.increment("did:key:b");
        clock_b.increment("did:key:b");
        let items_b: HashMap<String, serde_json::Value> = [
            ("hash2".into(), serde_json::json!({"id": 2})),
            ("hash3".into(), serde_json::json!({"id": 3})),
            ("hash4".into(), serde_json::json!({"id": 4})),
        ]
        .into_iter()
        .collect();

        let mut session_a = SyncSession::new(clock_a, items_a);
        let mut session_b = SyncSession::new(clock_b, items_b);

        // A → B: Have
        let have_msg = session_a.initiate();
        assert_eq!(session_a.state, SyncState::WaitingForWant);

        // B processes Have → Want (B wants hash1, which it's missing)
        let want_msg = session_b.process_have(have_msg);
        assert_eq!(session_b.state, SyncState::WaitingForData);
        if let SyncMessage::Want {
            ref content_hashes,
        } = want_msg
        {
            assert!(content_hashes.contains("hash1"));
            assert_eq!(content_hashes.len(), 1);
        } else {
            panic!("expected Want message");
        }

        // A processes Want → Data
        let data_msg = session_a.process_want(want_msg);
        assert_eq!(session_a.state, SyncState::Complete);

        // B processes Data → gets new items
        let new_items = session_b.process_data(data_msg);
        assert_eq!(session_b.state, SyncState::Complete);
        assert_eq!(new_items.len(), 1);
        assert_eq!(new_items[0], serde_json::json!({"id": 1}));
    }

    #[test]
    fn sync_no_missing_items() {
        let clock = VectorClock::new();
        let items: HashMap<String, serde_json::Value> =
            [("h1".into(), serde_json::json!("a"))].into_iter().collect();

        let mut initiator = SyncSession::new(clock.clone(), items.clone());
        let mut responder = SyncSession::new(clock, items);

        let have = initiator.initiate();
        let want = responder.process_have(have);
        if let SyncMessage::Want {
            ref content_hashes,
        } = want
        {
            assert!(content_hashes.is_empty());
        } else {
            panic!("expected Want");
        }

        let data = initiator.process_want(want);
        let new = responder.process_data(data);
        assert!(new.is_empty());
    }
}
