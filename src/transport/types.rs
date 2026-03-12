use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Core message types
// ---------------------------------------------------------------------------

/// Classification of transport-layer messages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageType {
    Handshake,
    Have,
    Want,
    Data,
    Tombstone,
    Ack,
    TrustUpdate,
}

/// Header attached to every transport message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    pub message_id: String,
    pub message_type: MessageType,
    pub sender_did: String,
    pub timestamp: DateTime<Utc>,
    pub correlation_id: Option<String>,
}

/// A message exchanged between peers over the transport layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub header: MessageHeader,
    pub payload: Vec<u8>,
}

impl Message {
    /// Create a new message with an auto-generated id and current timestamp.
    pub fn new(message_type: MessageType, sender_did: String, payload: Vec<u8>) -> Self {
        Self {
            header: MessageHeader {
                message_id: uuid::Uuid::new_v4().to_string(),
                message_type,
                sender_did,
                timestamp: Utc::now(),
                correlation_id: None,
            },
            payload,
        }
    }

    /// Attach a correlation id (e.g. to link a reply to a request).
    pub fn with_correlation_id(mut self, id: String) -> Self {
        self.header.correlation_id = Some(id);
        self
    }
}

// ---------------------------------------------------------------------------
// Peer info & registry
// ---------------------------------------------------------------------------

/// Metadata about a known peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub did_key: String,
    pub addresses: Vec<String>,
    pub capabilities: Vec<String>,
    pub last_seen: DateTime<Utc>,
    pub trust_score: f64,
}

/// Thread-safe registry of known peers, keyed by `did:key`.
#[derive(Debug, Clone)]
pub struct PeerRegistry {
    peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self {
            peers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert or replace a peer entry.
    pub async fn add(&self, peer: PeerInfo) {
        self.peers.write().await.insert(peer.did_key.clone(), peer);
    }

    /// Remove a peer by DID key. Returns the removed entry, if any.
    pub async fn remove(&self, did_key: &str) -> Option<PeerInfo> {
        self.peers.write().await.remove(did_key)
    }

    /// Look up a peer by DID key.
    pub async fn get(&self, did_key: &str) -> Option<PeerInfo> {
        self.peers.read().await.get(did_key).cloned()
    }

    /// Return a snapshot of all known peers.
    pub async fn list(&self) -> Vec<PeerInfo> {
        self.peers.read().await.values().cloned().collect()
    }

    /// Bump the `last_seen` timestamp for a peer. Returns `true` if the peer
    /// was found (and updated), `false` otherwise.
    pub async fn update_last_seen(&self, did_key: &str) -> bool {
        let mut guard = self.peers.write().await;
        if let Some(peer) = guard.get_mut(did_key) {
            peer.last_seen = Utc::now();
            true
        } else {
            false
        }
    }

    /// Return the number of tracked peers.
    pub async fn len(&self) -> usize {
        self.peers.read().await.len()
    }

    /// Returns `true` when the registry contains no peers.
    pub async fn is_empty(&self) -> bool {
        self.peers.read().await.is_empty()
    }
}

impl Default for PeerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_peer(did: &str) -> PeerInfo {
        PeerInfo {
            did_key: did.to_string(),
            addresses: vec!["127.0.0.1:4222".to_string()],
            capabilities: vec!["sync".to_string()],
            last_seen: Utc::now(),
            trust_score: 0.5,
        }
    }

    // -- PeerRegistry -------------------------------------------------------

    #[tokio::test]
    async fn registry_add_and_get() {
        let reg = PeerRegistry::new();
        reg.add(make_peer("did:key:alice")).await;
        let peer = reg.get("did:key:alice").await;
        assert!(peer.is_some());
        assert_eq!(peer.unwrap().did_key, "did:key:alice");
    }

    #[tokio::test]
    async fn registry_remove() {
        let reg = PeerRegistry::new();
        reg.add(make_peer("did:key:alice")).await;
        let removed = reg.remove("did:key:alice").await;
        assert!(removed.is_some());
        assert!(reg.get("did:key:alice").await.is_none());
        assert!(reg.is_empty().await);
    }

    #[tokio::test]
    async fn registry_list() {
        let reg = PeerRegistry::new();
        reg.add(make_peer("did:key:alice")).await;
        reg.add(make_peer("did:key:bob")).await;
        assert_eq!(reg.list().await.len(), 2);
        assert_eq!(reg.len().await, 2);
    }

    #[tokio::test]
    async fn registry_update_last_seen() {
        let reg = PeerRegistry::new();
        let mut peer = make_peer("did:key:alice");
        peer.last_seen = DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        reg.add(peer).await;

        assert!(reg.update_last_seen("did:key:alice").await);
        let updated = reg.get("did:key:alice").await.unwrap();
        // Verify last_seen was updated to a recent time (not the 2020 value).
        assert!(updated.last_seen > DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc));
    }

    #[tokio::test]
    async fn registry_update_last_seen_missing() {
        let reg = PeerRegistry::new();
        assert!(!reg.update_last_seen("did:key:nobody").await);
    }

    #[tokio::test]
    async fn registry_concurrent_access() {
        let reg = PeerRegistry::new();
        let reg2 = reg.clone();

        let h1 = tokio::spawn(async move {
            for i in 0..50 {
                reg2.add(make_peer(&format!("did:key:peer-a-{i}"))).await;
            }
        });

        let reg3 = reg.clone();
        let h2 = tokio::spawn(async move {
            for i in 0..50 {
                reg3.add(make_peer(&format!("did:key:peer-b-{i}"))).await;
            }
        });

        h1.await.unwrap();
        h2.await.unwrap();
        assert_eq!(reg.len().await, 100);
    }

    // -- Message serialization roundtrip ------------------------------------

    #[test]
    fn message_serialization_roundtrip() {
        let msg = Message::new(
            MessageType::Handshake,
            "did:key:alice".to_string(),
            b"hello".to_vec(),
        )
        .with_correlation_id("corr-123".to_string());

        let json = serde_json::to_string(&msg).unwrap();
        let deser: Message = serde_json::from_str(&json).unwrap();

        assert_eq!(deser.header.message_type, MessageType::Handshake);
        assert_eq!(deser.header.sender_did, "did:key:alice");
        assert_eq!(deser.payload, b"hello");
        assert_eq!(deser.header.correlation_id.as_deref(), Some("corr-123"));
    }

    #[test]
    fn message_type_all_variants_serialize() {
        let types = vec![
            MessageType::Handshake,
            MessageType::Have,
            MessageType::Want,
            MessageType::Data,
            MessageType::Tombstone,
            MessageType::Ack,
            MessageType::TrustUpdate,
        ];
        for mt in types {
            let json = serde_json::to_string(&mt).unwrap();
            let deser: MessageType = serde_json::from_str(&json).unwrap();
            assert_eq!(deser, mt);
        }
    }
}
