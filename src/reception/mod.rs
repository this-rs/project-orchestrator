//! P4 Reception Pipeline — verify, score, replay, and anchor incoming P2P envelopes.
//!
//! The reception pipeline processes incoming [`DistillationEnvelope`]s from peers:
//!
//! 1. **Verify** — validate cryptographic trust proof and content integrity
//! 2. **Score** — compute multi-dimensional relevance against local context
//! 3. **Replay** — convert accepted envelopes into local notes (if score ≥ threshold)
//! 4. **Anchor** — link replayed notes to local knowledge graph tags
//!
//! The [`receive_envelope`] function orchestrates the full pipeline.

pub mod anchor;
pub mod replay;
pub mod score;
pub mod tombstone_scheduler;
pub mod trust;
pub mod verify;

use anyhow::{anyhow, Result};

use crate::episodes::distill_models::DistillationEnvelope;

use self::anchor::{anchor_notes, AnchorResult};
use self::replay::{replay_lesson, ReplayedNote};
use self::score::{score_relevance, LocalContext, RelevanceScore};
use self::verify::{verify_envelope, VerifiedEnvelope};

/// Outcome of the full reception pipeline.
#[derive(Debug, Clone)]
pub struct ReceptionResult {
    /// The verified envelope (always present if pipeline started).
    pub verified: VerifiedEnvelope,
    /// Relevance score computed against local context.
    pub score: RelevanceScore,
    /// Replayed notes (empty if score was below threshold).
    pub notes: Vec<ReplayedNote>,
    /// Anchor result (only if notes were replayed).
    pub anchor: Option<AnchorResult>,
}

/// Run the full reception pipeline on an incoming envelope.
///
/// Steps: verify → score → (if accepted) replay → anchor.
///
/// Returns [`ReceptionResult`] on success, or an error if verification fails.
/// If the score is below threshold, the result will contain empty notes and no anchor.
pub fn receive_envelope(
    envelope: &DistillationEnvelope,
    local_context: &LocalContext,
) -> Result<ReceptionResult> {
    // Step 1: Verify cryptographic integrity
    let verified = verify_envelope(envelope)?;

    // Step 2: Score relevance
    let relevance = score_relevance(&verified, local_context);

    // Step 3 & 4: Only replay + anchor if accepted
    if relevance.accepted {
        let notes = replay_lesson(&verified);
        let anchor = anchor_notes(&notes, &local_context.local_tags);

        Ok(ReceptionResult {
            verified,
            score: relevance,
            notes,
            anchor: Some(anchor),
        })
    } else {
        Ok(ReceptionResult {
            verified,
            score: relevance,
            notes: Vec::new(),
            anchor: None,
        })
    }
}

/// Convenience: run the pipeline but reject tombstoned content.
pub fn receive_envelope_checked(
    envelope: &DistillationEnvelope,
    local_context: &LocalContext,
    tombstones: &anchor::TombstoneRegistry,
) -> Result<ReceptionResult> {
    if tombstones.is_revoked(&envelope.meta.content_hash) {
        return Err(anyhow!(
            "Content hash {} has been tombstoned",
            envelope.meta.content_hash
        ));
    }
    receive_envelope(envelope, local_context)
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

    fn make_signed_envelope(tags: Vec<String>) -> DistillationEnvelope {
        let signing_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        let did = crate::identity::did::to_did_key(&signing_key.verifying_key());

        let lesson = DistilledLesson {
            abstract_pattern: "Always validate inputs at boundaries".to_string(),
            domain_tags: tags,
            portability_layer: PortabilityLayer::Domain,
            confidence: 0.88,
        };

        let lesson_json = serde_json::to_string(&lesson).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(lesson_json.as_bytes());
        let content_hash = hex::encode(hasher.finalize());

        let signature = signing_key.sign(content_hash.as_bytes());

        DistillationEnvelope {
            lesson,
            anonymized_content: "Anonymized content".to_string(),
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
        }
    }

    #[test]
    fn test_full_pipeline_accepted() {
        let envelope = make_signed_envelope(vec!["rust".to_string(), "validation".to_string()]);
        let ctx = LocalContext {
            local_tags: vec!["rust".to_string(), "validation".to_string()],
            local_languages: vec!["rust".to_string()],
            known_peers: HashMap::new(),
        };
        let result = receive_envelope(&envelope, &ctx).unwrap();
        assert!(result.score.accepted);
        assert!(!result.notes.is_empty());
        assert!(result.anchor.is_some());
    }

    #[test]
    fn test_full_pipeline_rejected_low_score() {
        let envelope = make_signed_envelope(vec!["python".to_string(), "django".to_string()]);
        let ctx = LocalContext {
            local_tags: vec!["haskell".to_string()],
            local_languages: vec!["haskell".to_string()],
            known_peers: HashMap::new(),
        };
        let result = receive_envelope(&envelope, &ctx).unwrap();
        assert!(!result.score.accepted);
        assert!(result.notes.is_empty());
        assert!(result.anchor.is_none());
    }

    #[test]
    fn test_full_pipeline_invalid_envelope() {
        let mut envelope = make_signed_envelope(vec!["rust".to_string()]);
        envelope.trust_proof.signature_hex = "badbad".to_string();
        let ctx = LocalContext::default();
        let result = receive_envelope(&envelope, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_tombstone_rejection() {
        let envelope = make_signed_envelope(vec!["rust".to_string()]);
        let ctx = LocalContext {
            local_tags: vec!["rust".to_string()],
            local_languages: vec!["rust".to_string()],
            known_peers: HashMap::new(),
        };
        let mut tombstones = anchor::TombstoneRegistry::new();
        tombstones.apply_tombstone(&envelope.meta.content_hash);

        let result = receive_envelope_checked(&envelope, &ctx, &tombstones);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("tombstoned"));
    }
}
