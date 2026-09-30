//! Tombstone signing and verification (Privacy MVP-B T2).
//!
//! Ed25519 signatures over `content_hash|issuer_did|issued_at`, with the
//! verifying key recovered from the issuer's did:key.

use crate::identity::did::from_did_key;
use crate::identity::InstanceIdentity;
use crate::reception::anchor::SignedTombstone;
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Verifier};

/// Verify a tombstone's Ed25519 signature (fail-closed).
///
/// The issuer's verifying key is recovered from `issuer_did` (did:key), and
/// the signature must be valid over [`build_signing_payload`]. Any malformed
/// field (empty hash, non-did:key issuer, bad hex, wrong length) returns `false`.
pub fn verify_tombstone_ed25519(tombstone: &SignedTombstone) -> bool {
    if tombstone.content_hash.is_empty() || tombstone.issuer_did.is_empty() {
        return false;
    }
    let Ok(key) = from_did_key(&tombstone.issuer_did) else {
        return false;
    };
    let Ok(sig_bytes) = hex::decode(&tombstone.signature_hex) else {
        return false;
    };
    let Ok(signature) = Signature::from_slice(&sig_bytes) else {
        return false;
    };
    key.verify(&build_signing_payload(tombstone), &signature)
        .is_ok()
}

/// Alias of [`verify_tombstone_ed25519`].
pub fn verify_tombstone(tombstone: &SignedTombstone) -> bool {
    verify_tombstone_ed25519(tombstone)
}

/// Create a tombstone signed by this instance's identity.
pub fn sign_tombstone(
    identity: &InstanceIdentity,
    content_hash: String,
    issued_at: DateTime<Utc>,
    reason: Option<String>,
) -> SignedTombstone {
    let mut tombstone = SignedTombstone {
        content_hash,
        issuer_did: identity.did_key().to_string(),
        signature_hex: String::new(),
        issued_at,
        reason,
    };
    let sig = identity.sign(&build_signing_payload(&tombstone));
    tombstone.signature_hex = hex::encode(sig.to_bytes());
    tombstone
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

    fn signed() -> SignedTombstone {
        let id = InstanceIdentity::generate();
        sign_tombstone(&id, "abc123".into(), Utc::now(), Some("r".into()))
    }

    #[test]
    fn test_valid_signature_accepted() {
        assert!(verify_tombstone(&signed()));
    }

    #[test]
    fn test_placeholder_zero_signature_rejected() {
        let mut ts = signed();
        ts.signature_hex = "0".repeat(128);
        assert!(!verify_tombstone(&ts));
    }

    #[test]
    fn test_forged_other_key_rejected() {
        let mut ts = signed();
        let other = InstanceIdentity::generate();
        ts.issuer_did = other.did_key().to_string();
        assert!(!verify_tombstone(&ts));
    }

    #[test]
    fn test_tampered_content_rejected() {
        let mut ts = signed();
        ts.content_hash = "other".into();
        assert!(!verify_tombstone(&ts));
    }

    #[test]
    fn test_malformed_rejected() {
        let mut ts = signed();
        ts.signature_hex = "g".repeat(128);
        assert!(!verify_tombstone(&ts));
        let mut ts = signed();
        ts.signature_hex = String::new();
        assert!(!verify_tombstone(&ts));
        let mut ts = signed();
        ts.issuer_did = "did:local:unknown".into();
        assert!(!verify_tombstone(&ts));
        let mut ts = signed();
        ts.content_hash = String::new();
        assert!(!verify_tombstone(&ts));
    }

    #[test]
    fn test_signing_payload_format() {
        let ts = signed();
        let payload = String::from_utf8(build_signing_payload(&ts)).unwrap();
        assert!(payload.starts_with("abc123|did:key:"));
    }
}
