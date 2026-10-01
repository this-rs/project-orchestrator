//! Tombstone signing and verification (Privacy MVP-B T2).
//!
//! Ed25519 signatures over a domain-separated, length-prefixed payload (see
//! [`build_signing_payload`]), with the verifying key recovered from the
//! issuer's did:key. Signature validity alone is NOT authority: use
//! [`verify_tombstone_authority`] to also bind the revocation to the content's
//! original owner.

use crate::identity::did::{from_did_key, to_did_key};
use crate::identity::InstanceIdentity;
use crate::reception::anchor::SignedTombstone;
use chrono::{DateTime, SecondsFormat, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier};
use serde::Serialize;

/// Domain-separation prefix of the signed payload (version 1).
pub const TOMBSTONE_DOMAIN_PREFIX: &str = "po-tombstone-v1\n";

/// Issuer DID written by the former placeholder code on fake tombstones.
const LEGACY_PLACEHOLDER_DID: &str = "did:local:unknown";

/// Verify a tombstone's Ed25519 signature (fail-closed).
///
/// The issuer's verifying key is recovered from `issuer_did` (did:key), and
/// the signature must be valid over [`build_signing_payload`]. Any malformed
/// field (empty hash, non-did:key issuer, bad hex, wrong length) returns `false`.
///
/// This only proves that `issuer_did` signed the tombstone; it does not prove
/// the issuer may revoke this content. See [`verify_tombstone_authority`].
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

/// Verify signature AND authority (fail-closed).
///
/// A tombstone is accepted only if (1) its signature is valid and (2) its
/// issuer is the known original owner of the content (`known_owner`, e.g. the
/// `origin_did` of the `SharedArtifactMeta` or the envelope `trust_proof.source_did`).
///
/// An unknown owner (`None`) is REFUSED: there is no way to tell a legitimate
/// revocation from a third party blocking someone else's content hash, and the
/// registry has no solid reason to trust an unattributable issuer.
pub fn verify_tombstone_authority(
    tombstone: &SignedTombstone,
    known_owner: Option<&str>,
) -> Result<(), String> {
    if !verify_tombstone_ed25519(tombstone) {
        return Err(format!(
            "tombstone signature verification failed for {}",
            tombstone.content_hash
        ));
    }
    match known_owner {
        None => Err(format!(
            "tombstone refused for {}: content owner unknown (fail-closed)",
            tombstone.content_hash
        )),
        Some(owner) if owner != tombstone.issuer_did => Err(format!(
            "tombstone refused for {}: issuer {} is not the content owner",
            tombstone.content_hash, tombstone.issuer_did
        )),
        Some(_) => Ok(()),
    }
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

/// Create a tombstone signed with a raw Ed25519 key (issuer = its did:key).
pub fn sign_tombstone_with_key(
    key: &SigningKey,
    content_hash: String,
    issued_at: DateTime<Utc>,
    reason: Option<String>,
) -> SignedTombstone {
    let mut tombstone = SignedTombstone {
        content_hash,
        issuer_did: to_did_key(&key.verifying_key()),
        signature_hex: String::new(),
        issued_at,
        reason,
    };
    let sig = key.sign(&build_signing_payload(&tombstone));
    tombstone.signature_hex = hex::encode(sig.to_bytes());
    tombstone
}

fn push_field(buf: &mut Vec<u8>, field: &[u8]) {
    buf.extend_from_slice(&(field.len() as u32).to_be_bytes());
    buf.extend_from_slice(field);
}

/// Build the signing payload for a tombstone (for verification or creation).
///
/// Format: `"po-tombstone-v1\n"` followed by four fields, each encoded as a
/// 4-byte big-endian length then the UTF-8 bytes (unambiguous, no separator
/// injection): `content_hash`, `issuer_did`, `issued_at` (RFC 3339, UTC,
/// nanosecond precision, `Z`), `reason`. A missing reason and an empty reason
/// are encoded identically (persistence cannot tell them apart either).
pub fn build_signing_payload(tombstone: &SignedTombstone) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(TOMBSTONE_DOMAIN_PREFIX.as_bytes());
    push_field(&mut buf, tombstone.content_hash.as_bytes());
    push_field(&mut buf, tombstone.issuer_did.as_bytes());
    push_field(
        &mut buf,
        tombstone
            .issued_at
            .to_rfc3339_opts(SecondsFormat::Nanos, true)
            .as_bytes(),
    );
    push_field(
        &mut buf,
        tombstone.reason.as_deref().unwrap_or("").as_bytes(),
    );
    buf
}

/// True if this tombstone carries the former placeholder signature/issuer
/// (all-zero signature, `did:local:unknown`, or no issuer/signature at all).
pub fn is_legacy_placeholder(tombstone: &SignedTombstone) -> bool {
    let sig = tombstone.signature_hex.as_str();
    sig.is_empty()
        || sig.bytes().all(|b| b == b'0')
        || tombstone.issuer_did.is_empty()
        || tombstone.issuer_did == LEGACY_PLACEHOLDER_DID
}

/// A tombstone annotated with its verification status, for listing APIs.
#[derive(Debug, Clone, Serialize)]
pub struct AnnotatedTombstone {
    #[serde(flatten)]
    pub tombstone: SignedTombstone,
    /// The Ed25519 signature is valid for the issuer's did:key.
    pub verified: bool,
    /// Placeholder entry written before real signing existed (never authoritative).
    pub legacy: bool,
}

/// Annotate a tombstone with `verified` / `legacy`.
pub fn annotate_tombstone(tombstone: SignedTombstone) -> AnnotatedTombstone {
    let legacy = is_legacy_placeholder(&tombstone);
    let verified = !legacy && verify_tombstone_ed25519(&tombstone);
    AnnotatedTombstone {
        tombstone,
        verified,
        legacy,
    }
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
    fn test_tampered_reason_rejected() {
        let mut ts = signed();
        ts.reason = Some("altered".into());
        assert!(!verify_tombstone(&ts));
        ts.reason = None;
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
        let payload = build_signing_payload(&ts);
        assert!(payload.starts_with(TOMBSTONE_DOMAIN_PREFIX.as_bytes()));
        // first field: length 6 then "abc123"
        let off = TOMBSTONE_DOMAIN_PREFIX.len();
        assert_eq!(&payload[off..off + 4], &6u32.to_be_bytes());
        assert_eq!(&payload[off + 4..off + 10], b"abc123");
    }

    #[test]
    fn test_payload_has_no_field_boundary_ambiguity() {
        let at = Utc::now();
        let mk = |h: &str, d: &str| SignedTombstone {
            content_hash: h.into(),
            issuer_did: d.into(),
            signature_hex: String::new(),
            issued_at: at,
            reason: None,
        };
        assert_ne!(
            build_signing_payload(&mk("a|b", "c")),
            build_signing_payload(&mk("a", "b|c"))
        );
    }

    #[test]
    fn test_authority_wrong_issuer_rejected() {
        let owner = InstanceIdentity::generate();
        let attacker = InstanceIdentity::generate();
        let ts = sign_tombstone(&attacker, "h".into(), Utc::now(), None);
        assert!(verify_tombstone_authority(&ts, Some(owner.did_key())).is_err());
    }

    #[test]
    fn test_authority_unknown_owner_rejected() {
        assert!(verify_tombstone_authority(&signed(), None).is_err());
    }

    #[test]
    fn test_authority_owner_accepted() {
        let owner = InstanceIdentity::generate();
        let ts = sign_tombstone(&owner, "h".into(), Utc::now(), Some("gdpr".into()));
        assert!(verify_tombstone_authority(&ts, Some(owner.did_key())).is_ok());
    }

    #[test]
    fn test_authority_forged_signature_rejected_even_for_owner() {
        let owner = InstanceIdentity::generate();
        let mut ts = sign_tombstone(&owner, "h".into(), Utc::now(), None);
        ts.signature_hex = "0".repeat(128);
        assert!(verify_tombstone_authority(&ts, Some(owner.did_key())).is_err());
    }

    #[test]
    fn test_sign_with_key_matches_verification() {
        let key = SigningKey::generate(&mut rand_core_06::OsRng);
        let ts = sign_tombstone_with_key(&key, "h".into(), Utc::now(), None);
        assert!(verify_tombstone(&ts));
    }

    /// Simulates persistence: Neo4j stores `datetime($rfc3339)` and returns it
    /// via `toString()`; reason `None` is stored as "" and read back as `None`.
    #[test]
    fn test_signature_roundtrip_through_simulated_persistence() {
        let id = InstanceIdentity::generate();
        let ts = sign_tombstone(&id, "h".into(), Utc::now(), None);
        let stored_issued_at = ts.issued_at.to_rfc3339();
        let stored_reason = ts.reason.clone().unwrap_or_default();
        let read_back = SignedTombstone {
            content_hash: ts.content_hash.clone(),
            issuer_did: ts.issuer_did.clone(),
            signature_hex: ts.signature_hex.clone(),
            issued_at: chrono::DateTime::parse_from_rfc3339(&stored_issued_at)
                .unwrap()
                .with_timezone(&Utc),
            reason: if stored_reason.is_empty() {
                None
            } else {
                Some(stored_reason)
            },
        };
        assert!(verify_tombstone(&read_back));
    }

    #[test]
    fn test_legacy_placeholders_marked() {
        let mut ts = signed();
        ts.signature_hex = "0".repeat(128);
        ts.issuer_did = "did:local:unknown".into();
        let a = annotate_tombstone(ts);
        assert!(a.legacy && !a.verified);
        let a = annotate_tombstone(signed());
        assert!(!a.legacy && a.verified);
        let json = serde_json::to_value(&a).unwrap();
        assert_eq!(json["verified"], true);
        assert_eq!(json["legacy"], false);
        assert!(json["content_hash"].is_string());
    }
}
