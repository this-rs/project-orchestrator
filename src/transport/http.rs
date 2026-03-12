//! HTTP-based transport layer for P2P knowledge exchange.
//!
//! Each PO instance runs an HTTP server (axum). Peers communicate directly
//! via HTTP POST requests — no broker required.
//!
//! Endpoints exposed by each peer:
//! - `POST /api/p2p/message` — receive a single Message
//! - `POST /api/p2p/sync`    — HAVE/WANT/DATA delta sync exchange
//! - `GET  /api/p2p/identity` — returns this peer's PeerInfo (for handshake)

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

use super::types::{Message, MessageType, PeerInfo, PeerRegistry};
use super::TransportLayer;

/// Configuration for the HTTP transport layer.
#[derive(Debug, Clone)]
pub struct HttpTransportConfig {
    /// Base URL of this instance (e.g., "http://localhost:8080").
    pub local_base_url: String,
    /// DID key of this instance.
    pub local_did: String,
    /// HTTP client timeout for outbound requests.
    pub request_timeout: Duration,
    /// Seed peers to connect to on startup (base URLs).
    pub seed_peers: Vec<String>,
}

impl Default for HttpTransportConfig {
    fn default() -> Self {
        Self {
            local_base_url: "http://localhost:8080".to_string(),
            local_did: String::new(),
            request_timeout: Duration::from_secs(30),
            seed_peers: Vec::new(),
        }
    }
}

/// HTTP-based implementation of the `TransportLayer` trait.
///
/// Peers are discovered via seed list or handshake. Messages are sent
/// as JSON POST requests to `{peer_base_url}/api/p2p/message`.
pub struct HttpTransport {
    config: HttpTransportConfig,
    client: Client,
    registry: PeerRegistry,
    /// Subscribers keyed by topic. Each subscriber gets a sender half.
    #[allow(clippy::type_complexity)]
    subscribers: Arc<RwLock<Vec<(String, mpsc::Sender<Message>)>>>,
    /// Flag to track shutdown state.
    shutdown: Arc<RwLock<bool>>,
}

impl HttpTransport {
    /// Create a new HTTP transport with the given configuration.
    pub fn new(config: HttpTransportConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(config.request_timeout)
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            config,
            client,
            registry: PeerRegistry::new(),
            subscribers: Arc::new(RwLock::new(Vec::new())),
            shutdown: Arc::new(RwLock::new(false)),
        })
    }

    /// Get a reference to the peer registry.
    pub fn registry(&self) -> &PeerRegistry {
        &self.registry
    }

    /// Discover peers from seed list by calling their `/api/p2p/identity` endpoint.
    pub async fn discover_seeds(&self) -> Result<Vec<PeerInfo>> {
        let mut discovered = Vec::new();

        for seed_url in &self.config.seed_peers {
            match self.fetch_peer_identity(seed_url).await {
                Ok(peer) => {
                    info!(peer_did = %peer.did_key, url = %seed_url, "Discovered seed peer");
                    self.registry.add(peer.clone()).await;
                    discovered.push(peer);
                }
                Err(e) => {
                    warn!(url = %seed_url, error = %e, "Failed to contact seed peer");
                }
            }
        }

        Ok(discovered)
    }

    /// Fetch a peer's identity from their `/api/p2p/identity` endpoint.
    async fn fetch_peer_identity(&self, base_url: &str) -> Result<PeerInfo> {
        let url = format!("{}/api/p2p/identity", base_url.trim_end_matches('/'));
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("GET {url} failed"))?;

        if !resp.status().is_success() {
            return Err(anyhow!(
                "Peer identity request failed with status {}",
                resp.status()
            ));
        }

        let peer: PeerInfo = resp
            .json()
            .await
            .context("Failed to parse peer identity response")?;

        Ok(peer)
    }

    /// Dispatch an incoming message to matching subscribers.
    pub async fn dispatch_incoming(&self, message: Message) {
        let topic = message_topic(&message);
        let subs = self.subscribers.read().await;
        for (sub_topic, sender) in subs.iter() {
            if (sub_topic == &topic || sub_topic == "*")
                && sender.try_send(message.clone()).is_err()
            {
                debug!(topic = %sub_topic, "Subscriber channel full or closed, dropping message");
            }
        }
    }
}

#[async_trait]
impl TransportLayer for HttpTransport {
    async fn send(&self, peer_did: &str, message: Message) -> Result<()> {
        if *self.shutdown.read().await {
            return Err(anyhow!("Transport is shut down"));
        }

        let peer = self
            .registry
            .get(peer_did)
            .await
            .ok_or_else(|| anyhow!("Unknown peer: {peer_did}"))?;

        let base_url = peer
            .addresses
            .first()
            .ok_or_else(|| anyhow!("Peer {peer_did} has no known addresses"))?;

        let url = format!("{}/api/p2p/message", base_url.trim_end_matches('/'));

        let resp = self
            .client
            .post(&url)
            .json(&message)
            .send()
            .await
            .with_context(|| format!("POST {url} failed"))?;

        if !resp.status().is_success() {
            return Err(anyhow!(
                "Send to {} failed with status {}",
                peer_did,
                resp.status()
            ));
        }

        // Update last seen
        self.registry.update_last_seen(peer_did).await;

        debug!(peer = %peer_did, msg_type = ?message.header.message_type, "Message sent via HTTP");
        Ok(())
    }

    async fn broadcast(&self, message: Message) -> Result<()> {
        if *self.shutdown.read().await {
            return Err(anyhow!("Transport is shut down"));
        }

        let peers = self.registry.list().await;
        let mut errors = Vec::new();

        for peer in &peers {
            if peer.did_key == self.config.local_did {
                continue; // Don't send to self
            }
            if let Err(e) = self.send(&peer.did_key, message.clone()).await {
                warn!(peer = %peer.did_key, error = %e, "Failed to broadcast to peer");
                errors.push(e);
            }
        }

        if errors.len() == peers.len() && !peers.is_empty() {
            return Err(anyhow!("Broadcast failed to all {} peers", peers.len()));
        }

        Ok(())
    }

    async fn subscribe(&self, topic: &str) -> Result<mpsc::Receiver<Message>> {
        let (tx, rx) = mpsc::channel(256);
        self.subscribers.write().await.push((topic.to_string(), tx));
        Ok(rx)
    }

    async fn connected_peers(&self) -> Vec<PeerInfo> {
        self.registry.list().await
    }

    async fn shutdown(&self) -> Result<()> {
        *self.shutdown.write().await = true;
        // Clean up subscribers
        self.subscribers.write().await.clear();
        info!("HTTP transport shut down");
        Ok(())
    }
}

/// Derive a topic string from a message for subscriber matching.
fn message_topic(message: &Message) -> String {
    match message.header.message_type {
        MessageType::Handshake => "handshake".to_string(),
        MessageType::Have | MessageType::Want | MessageType::Data => "sync".to_string(),
        MessageType::Tombstone => "tombstone".to_string(),
        MessageType::Ack => "ack".to_string(),
        MessageType::TrustUpdate => "trust".to_string(),
    }
}

/// Sync request/response for the `/api/p2p/sync` endpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SyncRequest {
    pub sender_did: String,
    pub sync_message: super::sync::SyncMessage,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SyncResponse {
    pub responder_did: String,
    pub sync_message: super::sync::SyncMessage,
}

/// Identity response for the `/api/p2p/identity` endpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IdentityResponse {
    pub peer_info: PeerInfo,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_config() -> HttpTransportConfig {
        HttpTransportConfig {
            local_base_url: "http://localhost:9999".to_string(),
            local_did: "did:key:z6MkLocalTest".to_string(),
            request_timeout: Duration::from_secs(5),
            seed_peers: vec![],
        }
    }

    #[test]
    fn test_message_topic_mapping() {
        let msg = Message::new(MessageType::Handshake, "did:key:test".to_string(), vec![]);
        assert_eq!(message_topic(&msg), "handshake");

        let msg = Message::new(MessageType::Have, "did:key:test".to_string(), vec![]);
        assert_eq!(message_topic(&msg), "sync");

        let msg = Message::new(MessageType::Want, "did:key:test".to_string(), vec![]);
        assert_eq!(message_topic(&msg), "sync");

        let msg = Message::new(MessageType::Data, "did:key:test".to_string(), vec![]);
        assert_eq!(message_topic(&msg), "sync");

        let msg = Message::new(MessageType::Tombstone, "did:key:test".to_string(), vec![]);
        assert_eq!(message_topic(&msg), "tombstone");

        let msg = Message::new(MessageType::TrustUpdate, "did:key:test".to_string(), vec![]);
        assert_eq!(message_topic(&msg), "trust");
    }

    #[tokio::test]
    async fn test_transport_creation() {
        let config = make_config();
        let transport = HttpTransport::new(config).unwrap();
        assert!(transport.connected_peers().await.is_empty());
    }

    #[tokio::test]
    async fn test_send_unknown_peer_errors() {
        let transport = HttpTransport::new(make_config()).unwrap();
        let msg = Message::new(
            MessageType::Handshake,
            "did:key:z6MkLocalTest".to_string(),
            vec![],
        );
        let result = transport.send("did:key:z6MkUnknown", msg).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown peer"));
    }

    #[tokio::test]
    async fn test_subscribe_and_dispatch() {
        let transport = HttpTransport::new(make_config()).unwrap();

        let mut rx = transport.subscribe("sync").await.unwrap();

        // Dispatch a Have message → should match "sync" topic
        let msg = Message::new(
            MessageType::Have,
            "did:key:alice".to_string(),
            b"test".to_vec(),
        );
        transport.dispatch_incoming(msg.clone()).await;

        let received = rx.try_recv().unwrap();
        assert_eq!(received.header.message_type, MessageType::Have);
        assert_eq!(received.header.sender_did, "did:key:alice");
    }

    #[tokio::test]
    async fn test_subscribe_wildcard() {
        let transport = HttpTransport::new(make_config()).unwrap();

        let mut rx = transport.subscribe("*").await.unwrap();

        let msg = Message::new(MessageType::Tombstone, "did:key:bob".to_string(), vec![]);
        transport.dispatch_incoming(msg).await;

        let received = rx.try_recv().unwrap();
        assert_eq!(received.header.message_type, MessageType::Tombstone);
    }

    #[tokio::test]
    async fn test_subscribe_no_match() {
        let transport = HttpTransport::new(make_config()).unwrap();

        let mut rx = transport.subscribe("handshake").await.unwrap();

        // Dispatch a sync message — should NOT match "handshake"
        let msg = Message::new(MessageType::Have, "did:key:alice".to_string(), vec![]);
        transport.dispatch_incoming(msg).await;

        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_shutdown_prevents_send() {
        let transport = HttpTransport::new(make_config()).unwrap();
        transport.shutdown().await.unwrap();

        let msg = Message::new(MessageType::Handshake, "did:key:test".to_string(), vec![]);
        let result = transport.send("did:key:someone", msg).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("shut down"));
    }

    #[tokio::test]
    async fn test_broadcast_empty_is_ok() {
        let transport = HttpTransport::new(make_config()).unwrap();
        let msg = Message::new(MessageType::Have, "did:key:test".to_string(), vec![]);
        // No peers → broadcast should succeed (no-op)
        let result = transport.broadcast(msg).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_registry_access() {
        let transport = HttpTransport::new(make_config()).unwrap();
        let peer = PeerInfo {
            did_key: "did:key:z6MkPeer1".to_string(),
            addresses: vec!["http://peer1:8080".to_string()],
            capabilities: vec!["sync".to_string()],
            last_seen: Utc::now(),
            trust_score: 0.7,
        };
        transport.registry().add(peer).await;
        assert_eq!(transport.connected_peers().await.len(), 1);
    }

    #[test]
    fn test_sync_request_serialization() {
        use super::super::sync::SyncMessage;
        use std::collections::HashSet;

        let req = SyncRequest {
            sender_did: "did:key:alice".to_string(),
            sync_message: SyncMessage::Want {
                content_hashes: HashSet::from(["hash1".to_string(), "hash2".to_string()]),
            },
        };
        let json = serde_json::to_string(&req).unwrap();
        let deser: SyncRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.sender_did, "did:key:alice");
    }

    #[test]
    fn test_identity_response_serialization() {
        let resp = IdentityResponse {
            peer_info: PeerInfo {
                did_key: "did:key:z6MkTest".to_string(),
                addresses: vec!["http://localhost:8080".to_string()],
                capabilities: vec!["sync".to_string(), "distill".to_string()],
                last_seen: Utc::now(),
                trust_score: 1.0,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deser: IdentityResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.peer_info.did_key, "did:key:z6MkTest");
        assert_eq!(deser.peer_info.capabilities.len(), 2);
    }

    #[test]
    fn test_default_config() {
        let config = HttpTransportConfig::default();
        assert_eq!(config.local_base_url, "http://localhost:8080");
        assert_eq!(config.request_timeout, Duration::from_secs(30));
        assert!(config.seed_peers.is_empty());
    }
}
