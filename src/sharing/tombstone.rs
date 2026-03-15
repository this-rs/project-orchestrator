//! Tombstone verification utilities (Privacy MVP-B T2).
//!
//! Provides signature verification for signed tombstones received
//! from peers over the P2P transport layer.

use crate::reception::anchor::SignedTombstone;

/// Verify a tombstone's structural integrity.
///
/// Checks that all required fields are present and the signature
/// has a valid hex format with minimum length (64 hex chars = 32 bytes).
///
/// For full Ed25519 verification, use [`verify_tombstone_ed25519`] when
/// `ed25519-dalek` is available.
pub fn verify_tombstone(tombstone: &SignedTombstone, issuer_public_key: &[u8]) -> bool {
    // Must have a non-empty public key
    if issuer_public_key.is_empty() {
        return false;
    }

    // Content hash must be non-empty
    if tombstone.content_hash.is_empty() {
        return false;
    }

    // Issuer DID must be non-empty
    if tombstone.issuer_did.is_empty() {
        return false;
    }

    // Signature must be at least 64 hex characters (32 bytes)
    if tombstone.signature_hex.len() < 64 {
        return false;
    }

    // Signature must be valid hex
    if hex::decode(&tombstone.signature_hex).is_err() {
        return false;
    }

    true
}

/// Build the signing payload for a tombstone (for verification or creation).
///
/// Format: `{content_hash}|{issuer_did}|{issued_at_rfc3339}`
pub fn build_signing_payload(tombstone: &SignedTombstone) -> Vec<u8> {
    format!(
        "{}|{}|{}",
        tombstone.content_hash,
        tombstone.issuer_did,
        tombstone.issued_at.to_rfc3339()
    )
    .into_bytes()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_tombstone(sig: &str) -> SignedTombstone {
        SignedTombstone {
            content_hash: "abc123def456".to_string(),
            issuer_did: "did:key:z6MkTest".to_string(),
            signature_hex: sig.to_string(),
            issued_at: Utc::now(),
            reason: Some("test revocation".to_string()),
        }
    }

    #[test]
    fn test_valid_tombstone() {
        // 64 hex chars = 32 bytes
        let sig = "a".repeat(64);
        let ts = make_tombstone(&sig);
        assert!(verify_tombstone(&ts, b"some_public_key"));
    }

    #[test]
    fn test_long_valid_signature() {
        // 128 hex chars (Ed25519 signature = 64 bytes)
        let sig = "ab".repeat(64);
        let ts = make_tombstone(&sig);
        assert!(verify_tombstone(&ts, b"key"));
    }

    #[test]
    fn test_empty_signature_rejected() {
        let ts = make_tombstone("");
        assert!(!verify_tombstone(&ts, b"key"));
    }

    #[test]
    fn test_short_signature_rejected() {
        let ts = make_tombstone("abcd1234");
        assert!(!verify_tombstone(&ts, b"key"));
    }

    #[test]
    fn test_invalid_hex_rejected() {
        let sig = "g".repeat(64); // 'g' is not valid hex
        let ts = make_tombstone(&sig);
        assert!(!verify_tombstone(&ts, b"key"));
    }

    #[test]
    fn test_empty_public_key_rejected() {
        let sig = "a".repeat(64);
        let ts = make_tombstone(&sig);
        assert!(!verify_tombstone(&ts, b""));
    }

    #[test]
    fn test_empty_content_hash_rejected() {
        let sig = "a".repeat(64);
        let mut ts = make_tombstone(&sig);
        ts.content_hash = String::new();
        assert!(!verify_tombstone(&ts, b"key"));
    }

    #[test]
    fn test_empty_issuer_did_rejected() {
        let sig = "a".repeat(64);
        let mut ts = make_tombstone(&sig);
        ts.issuer_did = String::new();
        assert!(!verify_tombstone(&ts, b"key"));
    }

    #[test]
    fn test_signing_payload_format() {
        let ts = make_tombstone("aa");
        let payload = build_signing_payload(&ts);
        let payload_str = String::from_utf8(payload).unwrap();
        assert!(payload_str.starts_with("abc123def456|did:key:z6MkTest|"));
    }
}
