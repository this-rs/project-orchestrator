//! Multi-dimensional relevance scoring for incoming envelopes.
//!
//! Scores how relevant an incoming distilled lesson is to the local instance
//! based on tag overlap, technology overlap, and source trust.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::verify::VerifiedEnvelope;

/// Default acceptance threshold — envelopes scoring below this are rejected.
pub const DEFAULT_THRESHOLD: f64 = 0.35;

/// Weight for tag overlap component (40%).
const TAG_WEIGHT: f64 = 0.40;
/// Weight for technology/language overlap component (35%).
const TECH_WEIGHT: f64 = 0.35;
/// Weight for source trust component (25%).
const TRUST_WEIGHT: f64 = 0.25;
/// Default trust score for unknown peers.
const DEFAULT_TRUST: f64 = 0.5;

/// Local context used for relevance scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalContext {
    /// Tags representing local project domains (e.g., `["rust", "neo4j", "p2p"]`).
    pub local_tags: Vec<String>,
    /// Programming languages / frameworks used locally.
    pub local_languages: Vec<String>,
    /// Known peer trust scores, keyed by DID.
    #[serde(default)]
    pub known_peers: HashMap<String, f64>,
}

impl Default for LocalContext {
    fn default() -> Self {
        Self {
            local_tags: Vec::new(),
            local_languages: Vec::new(),
            known_peers: HashMap::new(),
        }
    }
}

/// Result of multi-dimensional relevance scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceScore {
    /// Overall weighted score (0.0–1.0).
    pub total: f64,
    /// Tag overlap score (Jaccard similarity, 0.0–1.0).
    pub tag_score: f64,
    /// Technology/language overlap score (0.0–1.0).
    pub tech_score: f64,
    /// Source trust score (0.0–1.0).
    pub trust_score: f64,
    /// Whether the score meets the acceptance threshold.
    pub accepted: bool,
}

/// Compute Jaccard similarity between two sets of strings (case-insensitive).
fn jaccard_similarity(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }

    let set_a: HashSet<String> = a.iter().map(|s| s.to_lowercase()).collect();
    let set_b: HashSet<String> = b.iter().map(|s| s.to_lowercase()).collect();

    let intersection = set_a.intersection(&set_b).count() as f64;
    let union = set_a.union(&set_b).count() as f64;

    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Score the relevance of a verified envelope against local context.
///
/// Formula: `40% tag_overlap + 35% tech_overlap + 25% source_trust`
pub fn score_relevance(
    envelope: &VerifiedEnvelope,
    local_context: &LocalContext,
) -> RelevanceScore {
    // Tag overlap: Jaccard similarity between envelope domain_tags and local_tags
    let tag_score = jaccard_similarity(
        &envelope.envelope.lesson.domain_tags,
        &local_context.local_tags,
    );

    // Tech overlap: check if same languages/frameworks mentioned in domain_tags
    let tech_score = jaccard_similarity(
        &envelope.envelope.lesson.domain_tags,
        &local_context.local_languages,
    );

    // Source trust: lookup from known_peers, default 0.5 for unknown
    let trust_score = local_context
        .known_peers
        .get(&envelope.source_did)
        .copied()
        .unwrap_or(DEFAULT_TRUST);

    let total = TAG_WEIGHT * tag_score + TECH_WEIGHT * tech_score + TRUST_WEIGHT * trust_score;

    RelevanceScore {
        total,
        tag_score,
        tech_score,
        trust_score,
        accepted: total >= DEFAULT_THRESHOLD,
    }
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

    fn make_verified_envelope(tags: Vec<String>) -> VerifiedEnvelope {
        VerifiedEnvelope {
            envelope: DistillationEnvelope {
                lesson: DistilledLesson {
                    abstract_pattern: "test pattern".to_string(),
                    domain_tags: tags,
                    portability_layer: PortabilityLayer::Domain,
                    confidence: 0.9,
                },
                anonymized_content: "content".to_string(),
                meta: DistillationMeta {
                    pipeline_version: "1.0".to_string(),
                    sensitivity_level: SensitivityLevel::Public,
                    quality_score: 0.8,
                    content_hash: "abc".to_string(),
                },
                trust_proof: TrustProof {
                    source_did: "did:key:zTestPeer".to_string(),
                    signature_hex: "deadbeef".to_string(),
                    trust_scores: HashMap::new(),
                },
                anonymization_report: None,
            },
            source_did: "did:key:zTestPeer".to_string(),
            verified_at: Utc::now(),
        }
    }

    #[test]
    fn test_perfect_overlap() {
        let env = make_verified_envelope(vec!["rust".to_string(), "neo4j".to_string()]);
        let ctx = LocalContext {
            local_tags: vec!["rust".to_string(), "neo4j".to_string()],
            local_languages: vec!["rust".to_string()],
            known_peers: HashMap::from([("did:key:zTestPeer".to_string(), 1.0)]),
        };
        let score = score_relevance(&env, &ctx);
        assert!(score.tag_score > 0.99, "Perfect tag overlap should be ~1.0");
        assert!(score.trust_score > 0.99);
        assert!(score.accepted);
    }

    #[test]
    fn test_no_overlap() {
        let env = make_verified_envelope(vec!["python".to_string(), "django".to_string()]);
        let ctx = LocalContext {
            local_tags: vec!["rust".to_string(), "neo4j".to_string()],
            local_languages: vec!["rust".to_string()],
            known_peers: HashMap::new(),
        };
        let score = score_relevance(&env, &ctx);
        assert_eq!(score.tag_score, 0.0);
        assert_eq!(score.tech_score, 0.0);
        assert!(
            (score.trust_score - 0.5).abs() < f64::EPSILON,
            "Unknown peer should get 0.5"
        );
        // total = 0.0 + 0.0 + 0.25 * 0.5 = 0.125
        assert!(score.total < DEFAULT_THRESHOLD);
        assert!(!score.accepted);
    }

    #[test]
    fn test_partial_overlap() {
        let env = make_verified_envelope(vec!["rust".to_string(), "python".to_string()]);
        let ctx = LocalContext {
            local_tags: vec!["rust".to_string(), "neo4j".to_string()],
            local_languages: vec!["rust".to_string(), "typescript".to_string()],
            known_peers: HashMap::new(),
        };
        let score = score_relevance(&env, &ctx);
        // tag: Jaccard({rust,python}, {rust,neo4j}) = 1/3 ≈ 0.333
        assert!(score.tag_score > 0.3 && score.tag_score < 0.4);
        // tech: Jaccard({rust,python}, {rust,typescript}) = 1/3 ≈ 0.333
        assert!(score.tech_score > 0.3 && score.tech_score < 0.4);
    }

    #[test]
    fn test_case_insensitive_tags() {
        let env = make_verified_envelope(vec!["Rust".to_string(), "NEO4J".to_string()]);
        let ctx = LocalContext {
            local_tags: vec!["rust".to_string(), "neo4j".to_string()],
            local_languages: vec![],
            known_peers: HashMap::new(),
        };
        let score = score_relevance(&env, &ctx);
        assert!(
            score.tag_score > 0.99,
            "Case-insensitive matching should give perfect score"
        );
    }

    #[test]
    fn test_known_peer_trust() {
        let env = make_verified_envelope(vec![]);
        let ctx = LocalContext {
            local_tags: vec![],
            local_languages: vec![],
            known_peers: HashMap::from([("did:key:zTestPeer".to_string(), 0.95)]),
        };
        let score = score_relevance(&env, &ctx);
        assert!((score.trust_score - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_tags_both_sides() {
        let env = make_verified_envelope(vec![]);
        let ctx = LocalContext::default();
        let score = score_relevance(&env, &ctx);
        assert_eq!(score.tag_score, 0.0);
        assert_eq!(score.tech_score, 0.0);
        // total = 0.25 * 0.5 = 0.125
        assert!(!score.accepted);
    }
}
