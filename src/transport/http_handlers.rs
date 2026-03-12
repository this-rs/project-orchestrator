//! Axum handlers for P2P HTTP endpoints.
//!
//! These handlers are mounted on the existing axum server to enable
//! peer-to-peer communication over HTTP.
//!
//! Endpoints:
//! - `GET  /api/p2p/identity`  — return this instance's PeerInfo
//! - `POST /api/p2p/message`   — receive a message from a peer
//! - `POST /api/p2p/sync`      — handle HAVE/WANT/DATA sync exchange

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use std::sync::Arc;
use tracing::{debug, warn};

use super::http::{HttpTransport, IdentityResponse, SyncRequest, SyncResponse};
use super::sync::{SyncMessage, VectorClock};
use super::types::{Message, PeerInfo};

/// Shared state for P2P handlers.
pub struct P2pState {
    pub transport: Arc<HttpTransport>,
    pub local_peer_info: PeerInfo,
    /// Local content store for sync (content_hash → serialized envelope).
    /// In production this would be backed by Neo4j; for MVP it's in-memory.
    pub sync_store: Arc<tokio::sync::RwLock<std::collections::HashMap<String, serde_json::Value>>>,
    pub local_clock: Arc<tokio::sync::RwLock<VectorClock>>,
}

/// `GET /api/p2p/identity` — return this instance's PeerInfo for handshake.
pub async fn get_identity(State(state): State<Arc<P2pState>>) -> impl IntoResponse {
    Json(IdentityResponse {
        peer_info: state.local_peer_info.clone(),
    })
}

/// `POST /api/p2p/message` — receive an incoming message from a peer.
pub async fn receive_message(
    State(state): State<Arc<P2pState>>,
    Json(message): Json<Message>,
) -> impl IntoResponse {
    debug!(
        sender = %message.header.sender_did,
        msg_type = ?message.header.message_type,
        "Received P2P message"
    );

    // Update sender in registry (mark as seen)
    state
        .transport
        .registry()
        .update_last_seen(&message.header.sender_did)
        .await;

    // Dispatch to subscribers
    state.transport.dispatch_incoming(message).await;

    StatusCode::OK
}

/// `POST /api/p2p/sync` — handle a sync exchange (HAVE → WANT, WANT → DATA).
pub async fn handle_sync(
    State(state): State<Arc<P2pState>>,
    Json(request): Json<SyncRequest>,
) -> Result<Json<SyncResponse>, (StatusCode, String)> {
    debug!(
        sender = %request.sender_did,
        "Processing sync request"
    );

    // Update sender in registry
    state
        .transport
        .registry()
        .update_last_seen(&request.sender_did)
        .await;

    let response_message = match request.sync_message {
        SyncMessage::Have {
            vector_clock: _their_clock,
            content_hashes: their_hashes,
        } => {
            // They sent HAVE — compute what we're missing and send WANT
            let store = state.sync_store.read().await;
            let our_hashes: std::collections::HashSet<String> = store.keys().cloned().collect();
            let missing: std::collections::HashSet<String> =
                their_hashes.difference(&our_hashes).cloned().collect();

            debug!(missing_count = missing.len(), "Responding with WANT");

            SyncMessage::Want {
                content_hashes: missing,
            }
        }
        SyncMessage::Want { content_hashes } => {
            // They sent WANT — send the requested data
            let store = state.sync_store.read().await;
            let envelopes: Vec<serde_json::Value> = content_hashes
                .iter()
                .filter_map(|hash| store.get(hash).cloned())
                .collect();

            debug!(sending_count = envelopes.len(), "Responding with DATA");

            SyncMessage::Data { envelopes }
        }
        SyncMessage::Data { envelopes } => {
            // They sent DATA — ingest it
            let mut store = state.sync_store.write().await;
            let mut clock = state.local_clock.write().await;
            let mut ingested = 0;

            for envelope in &envelopes {
                // Compute content hash for dedup
                let hash = sha2_hash(&serde_json::to_vec(envelope).unwrap_or_default());
                if let std::collections::hash_map::Entry::Vacant(e) = store.entry(hash) {
                    e.insert(envelope.clone());
                    ingested += 1;
                }
            }

            // Advance our clock
            clock.increment(&state.local_peer_info.did_key);

            debug!(
                ingested_count = ingested,
                total_received = envelopes.len(),
                "Ingested sync DATA"
            );

            // Acknowledge with empty HAVE (signals we're up to date)
            SyncMessage::Have {
                vector_clock: clock.clone(),
                content_hashes: store.keys().cloned().collect(),
            }
        }
    };

    Ok(Json(SyncResponse {
        responder_did: state.local_peer_info.did_key.clone(),
        sync_message: response_message,
    }))
}

/// Perform a full HTTP sync with a remote peer.
///
/// Drives the 3-step protocol:
/// 1. Send our HAVE → get their WANT
/// 2. Send DATA for what they WANT → (optional ACK)
///
/// Also pulls: send HAVE to discover what THEY have that WE don't.
pub async fn sync_with_peer(
    transport: &HttpTransport,
    peer_did: &str,
    local_store: &tokio::sync::RwLock<std::collections::HashMap<String, serde_json::Value>>,
    local_clock: &tokio::sync::RwLock<VectorClock>,
) -> anyhow::Result<SyncReport> {
    let peer = transport
        .registry()
        .get(peer_did)
        .await
        .ok_or_else(|| anyhow::anyhow!("Unknown peer: {peer_did}"))?;

    let base_url = peer
        .addresses
        .first()
        .ok_or_else(|| anyhow::anyhow!("Peer {peer_did} has no addresses"))?;

    let sync_url = format!("{}/api/p2p/sync", base_url.trim_end_matches('/'));
    let client = reqwest::Client::new();

    // Step 1: Send HAVE
    let store = local_store.read().await;
    let clock = local_clock.read().await;
    let our_hashes: std::collections::HashSet<String> = store.keys().cloned().collect();

    let have_request = SyncRequest {
        sender_did: transport
            .registry()
            .list()
            .await
            .first()
            .map(|p| p.did_key.clone())
            .unwrap_or_default(),
        sync_message: SyncMessage::Have {
            vector_clock: clock.clone(),
            content_hashes: our_hashes,
        },
    };
    drop(store);
    drop(clock);

    let resp = client
        .post(&sync_url)
        .json(&have_request)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Sync HAVE request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("Sync HAVE failed: {}", resp.status()));
    }

    let want_response: SyncResponse = resp
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to parse WANT response: {e}"))?;

    let items_requested = match &want_response.sync_message {
        SyncMessage::Want { content_hashes } => content_hashes.len(),
        _ => 0,
    };

    // Step 2: If they want something, send DATA
    let mut items_sent = 0;
    if let SyncMessage::Want { content_hashes } = want_response.sync_message {
        if !content_hashes.is_empty() {
            let store = local_store.read().await;
            let envelopes: Vec<serde_json::Value> = content_hashes
                .iter()
                .filter_map(|hash| store.get(hash).cloned())
                .collect();
            items_sent = envelopes.len();
            drop(store);

            let data_request = SyncRequest {
                sender_did: have_request.sender_did.clone(),
                sync_message: SyncMessage::Data { envelopes },
            };

            let resp = client
                .post(&sync_url)
                .json(&data_request)
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("Sync DATA request failed: {e}"))?;

            if !resp.status().is_success() {
                warn!(status = %resp.status(), "Sync DATA response was not success");
            }
        }
    }

    Ok(SyncReport {
        peer_did: peer_did.to_string(),
        items_requested,
        items_sent,
    })
}

/// Report from a sync exchange.
#[derive(Debug, Clone)]
pub struct SyncReport {
    pub peer_did: String,
    pub items_requested: usize,
    pub items_sent: usize,
}

/// Compute SHA-256 hash of bytes, returning hex string.
fn sha2_hash(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(data);
    hex::encode(hash)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_sha2_hash_deterministic() {
        let data = b"hello world";
        let h1 = sha2_hash(data);
        let h2 = sha2_hash(data);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA-256 = 32 bytes = 64 hex chars
    }

    #[test]
    fn test_sha2_hash_different_input() {
        let h1 = sha2_hash(b"hello");
        let h2 = sha2_hash(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_sync_request_roundtrip() {
        let req = SyncRequest {
            sender_did: "did:key:alice".to_string(),
            sync_message: SyncMessage::Have {
                vector_clock: VectorClock::new(),
                content_hashes: HashSet::from(["h1".to_string()]),
            },
        };
        let json = serde_json::to_string(&req).unwrap();
        let deser: SyncRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.sender_did, "did:key:alice");
    }

    #[test]
    fn test_sync_response_roundtrip() {
        let resp = SyncResponse {
            responder_did: "did:key:bob".to_string(),
            sync_message: SyncMessage::Want {
                content_hashes: HashSet::from(["hash_abc".to_string()]),
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deser: SyncResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.responder_did, "did:key:bob");
    }

    #[test]
    fn test_sync_report_display() {
        let report = SyncReport {
            peer_did: "did:key:z6MkTest".to_string(),
            items_requested: 5,
            items_sent: 3,
        };
        assert_eq!(report.items_requested, 5);
        assert_eq!(report.items_sent, 3);
    }
}
