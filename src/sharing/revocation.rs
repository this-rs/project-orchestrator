//! P2P revocation broadcast (Privacy MVP-B T3).
//!
//! Implements the broadcast-and-acknowledge protocol for revoking
//! shared knowledge artifacts across peers.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::transport::types::{AckStatus, Message, MessageType, PeerInfo, TombstonePayload};

// ============================================================================
// Revocation request & acknowledgement
// ============================================================================

/// A request to revoke shared content, sent over P2P transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevocationRequest {
    /// The tombstone payload to broadcast.
    pub tombstone: TombstonePayload,
    /// Whether this is an urgent revocation (secrets/PII detected).
    pub urgent: bool,
    /// Whether the sender expects an acknowledgement.
    pub ack_requested: bool,
    /// Hop count for transitive propagation (decremented at each relay, max 3).
    pub hop_count: u8,
}

/// Acknowledgement response to a revocation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevocationAck {
    /// Content hash that was revoked.
    pub content_hash: String,
    /// Outcome of applying the tombstone locally.
    pub ack_type: AckStatus,
    /// DID of the responding peer.
    pub responder_did: String,
    /// When the response was generated.
    pub responded_at: DateTime<Utc>,
}

// ============================================================================
// Broadcast logic
// ============================================================================

/// Build a RevocationRequest from a TombstonePayload.
pub fn build_revocation_request(tombstone: TombstonePayload, urgent: bool) -> RevocationRequest {
    RevocationRequest {
        tombstone,
        urgent,
        ack_requested: true,
        hop_count: 3,
    }
}

/// Determine target peers for a revocation broadcast.
///
/// If `urgent`, returns ALL known peers. Otherwise, only the peers
/// from the audit trail (those who received the artifact).
pub fn resolve_broadcast_targets<'a>(
    urgent: bool,
    audit_trail_peers: &'a [String],
    all_known_peers: &'a [PeerInfo],
) -> Vec<&'a str> {
    if urgent {
        all_known_peers.iter().map(|p| p.did_key.as_str()).collect()
    } else {
        audit_trail_peers.iter().map(|s| s.as_str()).collect()
    }
}

/// Build transport messages for each target peer.
pub fn build_broadcast_messages(
    request: &RevocationRequest,
    sender_did: &str,
    target_dids: &[&str],
) -> Vec<Message> {
    let payload = serde_json::to_vec(request).unwrap_or_default();
    target_dids
        .iter()
        .map(|_| {
            Message::new(
                MessageType::Tombstone,
                sender_did.to_string(),
                payload.clone(),
            )
        })
        .collect()
}

// ============================================================================
// Incoming tombstone handling
// ============================================================================

/// Process an incoming tombstone message.
///
/// Returns:
/// - An optional ack (if the sender requested acknowledgement)
/// - An optional re-broadcast request (if hop_count > 0)
pub fn handle_incoming_tombstone(
    request: &RevocationRequest,
    local_did: &str,
    content_found_locally: bool,
) -> (Option<RevocationAck>, Option<RevocationRequest>) {
    // Build ack if requested
    let ack = if request.ack_requested {
        Some(RevocationAck {
            content_hash: request.tombstone.content_hash.clone(),
            ack_type: if content_found_locally {
                AckStatus::Deleted
            } else {
                AckStatus::NotFound
            },
            responder_did: local_did.to_string(),
            responded_at: Utc::now(),
        })
    } else {
        None
    };

    // Re-broadcast with decremented hop_count if > 0
    let rebroadcast = if request.hop_count > 0 {
        Some(RevocationRequest {
            tombstone: request.tombstone.clone(),
            urgent: request.urgent,
            ack_requested: false, // Don't request ack on relay
            hop_count: request.hop_count - 1,
        })
    } else {
        None
    };

    (ack, rebroadcast)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tombstone_payload() -> TombstonePayload {
        TombstonePayload {
            content_hash: "hash123".to_string(),
            issuer_did: "did:key:alice".to_string(),
            signature_hex: "deadbeef".to_string(),
            issued_at: Utc::now(),
            reason: Some("test".to_string()),
            ack_requested: true,
        }
    }

    #[test]
    fn test_build_revocation_request() {
        let req = build_revocation_request(make_tombstone_payload(), false);
        assert_eq!(req.hop_count, 3);
        assert!(req.ack_requested);
        assert!(!req.urgent);
    }

    #[test]
    fn test_build_urgent_request() {
        let req = build_revocation_request(make_tombstone_payload(), true);
        assert!(req.urgent);
    }

    #[test]
    fn test_resolve_targets_urgent() {
        let audit = vec!["did:key:bob".to_string()];
        let peers = vec![
            PeerInfo {
                did_key: "did:key:bob".to_string(),
                addresses: vec![],
                capabilities: vec![],
                last_seen: Utc::now(),
                trust_score: 0.5,
            },
            PeerInfo {
                did_key: "did:key:carol".to_string(),
                addresses: vec![],
                capabilities: vec![],
                last_seen: Utc::now(),
                trust_score: 0.5,
            },
        ];
        let targets = resolve_broadcast_targets(true, &audit, &peers);
        assert_eq!(targets.len(), 2); // All peers for urgent
    }

    #[test]
    fn test_resolve_targets_non_urgent() {
        let audit = vec!["did:key:bob".to_string()];
        let peers = vec![
            PeerInfo {
                did_key: "did:key:bob".to_string(),
                addresses: vec![],
                capabilities: vec![],
                last_seen: Utc::now(),
                trust_score: 0.5,
            },
            PeerInfo {
                did_key: "did:key:carol".to_string(),
                addresses: vec![],
                capabilities: vec![],
                last_seen: Utc::now(),
                trust_score: 0.5,
            },
        ];
        let targets = resolve_broadcast_targets(false, &audit, &peers);
        assert_eq!(targets.len(), 1); // Only audit trail for non-urgent
        assert_eq!(targets[0], "did:key:bob");
    }

    #[test]
    fn test_build_broadcast_messages() {
        let req = build_revocation_request(make_tombstone_payload(), false);
        let targets = vec!["did:key:bob", "did:key:carol"];
        let msgs = build_broadcast_messages(&req, "did:key:alice", &targets);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].header.message_type, MessageType::Tombstone);
        assert_eq!(msgs[0].header.sender_did, "did:key:alice");
    }

    #[test]
    fn test_handle_incoming_content_found() {
        let req = build_revocation_request(make_tombstone_payload(), false);
        let (ack, rebroadcast) = handle_incoming_tombstone(&req, "did:key:bob", true);

        let ack = ack.expect("should have ack");
        assert_eq!(ack.ack_type, AckStatus::Deleted);
        assert_eq!(ack.responder_did, "did:key:bob");

        let rb = rebroadcast.expect("should have rebroadcast");
        assert_eq!(rb.hop_count, 2);
        assert!(!rb.ack_requested); // Relay doesn't request ack
    }

    #[test]
    fn test_handle_incoming_content_not_found() {
        let req = build_revocation_request(make_tombstone_payload(), false);
        let (ack, _) = handle_incoming_tombstone(&req, "did:key:bob", false);

        let ack = ack.expect("should have ack");
        assert_eq!(ack.ack_type, AckStatus::NotFound);
    }

    #[test]
    fn test_handle_incoming_no_rebroadcast_at_zero() {
        let mut req = build_revocation_request(make_tombstone_payload(), false);
        req.hop_count = 0;
        let (_, rebroadcast) = handle_incoming_tombstone(&req, "did:key:bob", true);
        assert!(rebroadcast.is_none());
    }

    #[test]
    fn test_handle_incoming_no_ack_when_not_requested() {
        let mut req = build_revocation_request(make_tombstone_payload(), false);
        req.ack_requested = false;
        let (ack, _) = handle_incoming_tombstone(&req, "did:key:bob", true);
        assert!(ack.is_none());
    }

    #[test]
    fn test_revocation_request_serialization_roundtrip() {
        let req = build_revocation_request(make_tombstone_payload(), true);
        let json = serde_json::to_string(&req).unwrap();
        let deser: RevocationRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.hop_count, 3);
        assert!(deser.urgent);
        assert_eq!(deser.tombstone.content_hash, "hash123");
    }

    #[test]
    fn test_revocation_ack_serialization_roundtrip() {
        let ack = RevocationAck {
            content_hash: "hash456".to_string(),
            ack_type: AckStatus::Deleted,
            responder_did: "did:key:bob".to_string(),
            responded_at: Utc::now(),
        };
        let json = serde_json::to_string(&ack).unwrap();
        let deser: RevocationAck = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.content_hash, "hash456");
        assert_eq!(deser.ack_type, AckStatus::Deleted);
    }
}
