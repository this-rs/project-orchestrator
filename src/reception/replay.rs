//! Replay incoming envelopes into local notes.
//!
//! Converts verified distillation envelopes into local note representations,
//! prefixed with `[P2P Import]` to distinguish them from locally-created notes.

use serde::{Deserialize, Serialize};

use crate::episodes::distill_models::PortabilityLayer;

use super::verify::VerifiedEnvelope;

/// A note created by replaying a P2P-imported lesson.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayedNote {
    /// Note content, prefixed with `[P2P Import]`.
    pub content: String,
    /// Type of note (always `"p2p_import"` for replayed notes).
    pub note_type: String,
    /// Importance score derived from lesson confidence.
    pub importance: f64,
    /// Tags carried over from the distilled lesson.
    pub tags: Vec<String>,
    /// DID of the peer that originated this lesson.
    pub source_did: String,
    /// Portability layer of the original lesson.
    pub portability: PortabilityLayer,
}

/// Replay a verified envelope into local notes.
///
/// Extracts the abstract pattern from the distilled lesson and wraps it
/// as a [`ReplayedNote`] with P2P provenance metadata.
pub fn replay_lesson(envelope: &VerifiedEnvelope) -> Vec<ReplayedNote> {
    let lesson = &envelope.envelope.lesson;

    let content = format!(
        "[P2P Import] {}\n\nSource: {}\nPortability: {:?}\nConfidence: {:.2}",
        lesson.abstract_pattern, envelope.source_did, lesson.portability_layer, lesson.confidence,
    );

    let note = ReplayedNote {
        content,
        note_type: "p2p_import".to_string(),
        importance: lesson.confidence,
        tags: lesson.domain_tags.clone(),
        source_did: envelope.source_did.clone(),
        portability: lesson.portability_layer,
    };

    vec![note]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::distill_models::{
        DistillationEnvelope, DistillationMeta, DistilledLesson, PortabilityLayer,
        SensitivityLevel, TrustProof,
    };
    use chrono::Utc;
    use std::collections::HashMap;

    fn make_verified(
        pattern: &str,
        tags: Vec<String>,
        portability: PortabilityLayer,
    ) -> VerifiedEnvelope {
        VerifiedEnvelope {
            envelope: DistillationEnvelope {
                lesson: DistilledLesson {
                    abstract_pattern: pattern.to_string(),
                    domain_tags: tags,
                    portability_layer: portability,
                    confidence: 0.85,
                },
                anonymized_content: "anonymized content".to_string(),
                meta: DistillationMeta {
                    pipeline_version: "1.0".to_string(),
                    sensitivity_level: SensitivityLevel::Public,
                    quality_score: 0.8,
                    content_hash: "abc".to_string(),
                },
                trust_proof: TrustProof {
                    source_did: "did:key:zSourcePeer".to_string(),
                    signature_hex: "sig".to_string(),
                    trust_scores: HashMap::new(),
                },
                anonymization_report: None,
            },
            source_did: "did:key:zSourcePeer".to_string(),
            verified_at: Utc::now(),
        }
    }

    #[test]
    fn test_replay_creates_note() {
        let env = make_verified(
            "Always validate inputs",
            vec!["rust".to_string()],
            PortabilityLayer::Domain,
        );
        let notes = replay_lesson(&env);
        assert_eq!(notes.len(), 1);
    }

    #[test]
    fn test_replay_prefix() {
        let env = make_verified(
            "Use connection pooling",
            vec![],
            PortabilityLayer::Universal,
        );
        let notes = replay_lesson(&env);
        assert!(notes[0].content.starts_with("[P2P Import]"));
    }

    #[test]
    fn test_replay_contains_pattern() {
        let env = make_verified(
            "Index foreign keys in relational databases",
            vec!["sql".to_string()],
            PortabilityLayer::Domain,
        );
        let notes = replay_lesson(&env);
        assert!(notes[0]
            .content
            .contains("Index foreign keys in relational databases"));
    }

    #[test]
    fn test_replay_preserves_tags() {
        let tags = vec!["rust".to_string(), "neo4j".to_string(), "graph".to_string()];
        let env = make_verified("Pattern", tags.clone(), PortabilityLayer::Domain);
        let notes = replay_lesson(&env);
        assert_eq!(notes[0].tags, tags);
    }

    #[test]
    fn test_replay_note_type() {
        let env = make_verified("Pattern", vec![], PortabilityLayer::Local);
        let notes = replay_lesson(&env);
        assert_eq!(notes[0].note_type, "p2p_import");
    }

    #[test]
    fn test_replay_source_did() {
        let env = make_verified("Pattern", vec![], PortabilityLayer::Project);
        let notes = replay_lesson(&env);
        assert_eq!(notes[0].source_did, "did:key:zSourcePeer");
    }

    #[test]
    fn test_replay_importance_from_confidence() {
        let env = make_verified("Pattern", vec![], PortabilityLayer::Domain);
        let notes = replay_lesson(&env);
        assert!((notes[0].importance - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_replay_portability() {
        let env = make_verified("Pattern", vec![], PortabilityLayer::Universal);
        let notes = replay_lesson(&env);
        assert_eq!(notes[0].portability, PortabilityLayer::Universal);
    }
}
