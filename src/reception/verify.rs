//! Envelope verification for incoming P2P distillation envelopes.
//!
//! Validates cryptographic trust proofs and content integrity before
//! an envelope is accepted into the local reception pipeline.

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Verifier};
use sha2::{Digest, Sha256};

use crate::episodes::distill_models::DistillationEnvelope;
use crate::identity::did::from_did_key;

/// An envelope that has passed cryptographic verification.
#[derive(Debug, Clone)]
pub struct VerifiedEnvelope {
    /// The original envelope, proven authentic.
    pub envelope: DistillationEnvelope,
    /// The DID of the source peer that signed this envelope.
    pub source_did: String,
    /// Timestamp when verification was performed.
    pub verified_at: DateTime<Utc>,
}

/// Verify an incoming [`DistillationEnvelope`].
///
/// Performs two checks:
/// 1. **Signature verification** — the `trust_proof.signature_hex` is a valid
///    Ed25519 signature over `meta.content_hash`, signed by the key embedded
///    in `trust_proof.source_did`.
/// 2. **Content integrity** — `meta.content_hash` matches the SHA-256 digest
///    of the serialized lesson content.
pub fn verify_envelope(envelope: &DistillationEnvelope) -> Result<VerifiedEnvelope> {
    let trust_proof = &envelope.trust_proof;
    let source_did = &trust_proof.source_did;

    // 1. Parse the source DID to extract the Ed25519 verifying key
    let verifying_key = from_did_key(source_did)
        .with_context(|| format!("Failed to parse source DID: {source_did}"))?;

    // 2. Decode the hex signature
    let sig_bytes = hex::decode(&trust_proof.signature_hex)
        .with_context(|| "Failed to decode signature_hex")?;

    let signature = Signature::from_slice(&sig_bytes)
        .map_err(|e| anyhow!("Invalid Ed25519 signature format: {e}"))?;

    // 3. Verify signature over content_hash
    let content_hash_bytes = envelope.meta.content_hash.as_bytes();
    verifying_key
        .verify(content_hash_bytes, &signature)
        .map_err(|_| anyhow!("Signature verification failed for DID {source_did}"))?;

    // 4. Verify content integrity: SHA-256 of serialized lesson must match content_hash
    let lesson_json = serde_json::to_string(&envelope.lesson)
        .with_context(|| "Failed to serialize lesson for hash verification")?;

    let mut hasher = Sha256::new();
    hasher.update(lesson_json.as_bytes());
    let computed_hash = hex::encode(hasher.finalize());

    if computed_hash != envelope.meta.content_hash {
        return Err(anyhow!(
            "Content hash mismatch: computed {computed_hash}, expected {}",
            envelope.meta.content_hash
        ));
    }

    Ok(VerifiedEnvelope {
        envelope: envelope.clone(),
        source_did: source_did.clone(),
        verified_at: Utc::now(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::distill_models::{
        DistillationEnvelope, DistillationMeta, DistilledLesson, PortabilityLayer,
        SensitivityLevel, TrustProof,
    };
    use ed25519_dalek::{Signer, SigningKey};
    use sha2::{Digest, Sha256};
    use std::collections::HashMap;

    /// Helper: create a validly-signed test envelope.
    fn make_signed_envelope() -> (DistillationEnvelope, SigningKey) {
        let signing_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        let did = crate::identity::did::to_did_key(&signing_key.verifying_key());

        let lesson = DistilledLesson {
            abstract_pattern: "Always validate inputs at system boundaries".to_string(),
            domain_tags: vec!["rust".to_string(), "validation".to_string()],
            portability_layer: PortabilityLayer::Domain,
            confidence: 0.88,
        };

        let lesson_json = serde_json::to_string(&lesson).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(lesson_json.as_bytes());
        let content_hash = hex::encode(hasher.finalize());

        let signature = signing_key.sign(content_hash.as_bytes());

        let envelope = DistillationEnvelope {
            lesson,
            anonymized_content: "Validated pattern about input validation".to_string(),
            meta: DistillationMeta {
                pipeline_version: "1.0".to_string(),
                sensitivity_level: SensitivityLevel::Public,
                quality_score: 0.85,
                content_hash,
            },
            trust_proof: TrustProof {
                source_did: did,
                signature_hex: hex::encode(signature.to_bytes()),
                trust_scores: HashMap::from([("quality".to_string(), 0.85)]),
            },
            anonymization_report: None,
        };

        (envelope, signing_key)
    }

    #[test]
    fn test_verify_valid_envelope() {
        let (envelope, _) = make_signed_envelope();
        let result = verify_envelope(&envelope);
        assert!(
            result.is_ok(),
            "Valid envelope should verify: {:?}",
            result.err()
        );
        let verified = result.unwrap();
        assert!(verified.source_did.starts_with("did:key:z"));
        assert!(verified.verified_at <= Utc::now());
    }

    #[test]
    fn test_verify_tampered_content_hash() {
        let (mut envelope, _) = make_signed_envelope();
        // Tamper with content_hash — signature will fail
        envelope.meta.content_hash =
            "0000000000000000000000000000000000000000000000000000000000000000".to_string();
        let result = verify_envelope(&envelope);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Signature verification failed"),
            "Should fail signature check when hash is tampered"
        );
    }

    #[test]
    fn test_verify_tampered_lesson() {
        let (mut envelope, signing_key) = make_signed_envelope();
        // Tamper with the lesson content but re-sign the original hash
        // (the hash won't match the new lesson)
        envelope.lesson.abstract_pattern = "Tampered pattern".to_string();
        // Re-sign with original hash so signature passes but hash integrity fails
        let signature = signing_key.sign(envelope.meta.content_hash.as_bytes());
        envelope.trust_proof.signature_hex = hex::encode(signature.to_bytes());
        let result = verify_envelope(&envelope);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Content hash mismatch"),
            "Should fail content integrity check"
        );
    }

    #[test]
    fn test_verify_invalid_did() {
        let (mut envelope, _) = make_signed_envelope();
        envelope.trust_proof.source_did = "did:key:invalid".to_string();
        let result = verify_envelope(&envelope);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_invalid_signature_hex() {
        let (mut envelope, _) = make_signed_envelope();
        envelope.trust_proof.signature_hex = "not-valid-hex".to_string();
        let result = verify_envelope(&envelope);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_wrong_key_signature() {
        let (mut envelope, _) = make_signed_envelope();
        // Sign with a different key
        let other_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        let bad_sig = other_key.sign(envelope.meta.content_hash.as_bytes());
        envelope.trust_proof.signature_hex = hex::encode(bad_sig.to_bytes());
        let result = verify_envelope(&envelope);
        assert!(result.is_err());
    }
}
