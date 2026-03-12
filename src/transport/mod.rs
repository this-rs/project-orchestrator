pub mod sync;
pub mod types;

use anyhow::Result;
use async_trait::async_trait;
use types::{Message, PeerInfo};

/// Abstraction over the peer-to-peer transport mechanism.
///
/// Implementations may use NATS, libp2p, or any other messaging substrate.
#[async_trait]
pub trait TransportLayer: Send + Sync {
    /// Send a message to a specific peer identified by its DID key.
    async fn send(&self, peer_did: &str, message: Message) -> Result<()>;

    /// Broadcast a message to all connected peers.
    async fn broadcast(&self, message: Message) -> Result<()>;

    /// Subscribe to messages on the given topic. Returns a channel receiver
    /// that will yield incoming messages until the subscription is dropped.
    async fn subscribe(&self, topic: &str) -> Result<tokio::sync::mpsc::Receiver<Message>>;

    /// Return the list of currently connected peers.
    async fn connected_peers(&self) -> Vec<PeerInfo>;

    /// Gracefully shut down the transport, closing all connections.
    async fn shutdown(&self) -> Result<()>;
}
