//! Distillation pipeline — lesson extraction and envelope creation.
//!
//! Two main entry points:
//! - [`abstract_lesson`]: extracts a portable lesson from episode content.
//! - [`create_envelope`]: wraps anonymized content + lesson into a signed envelope.

use sha2::{Digest, Sha256};
use std::collections::HashMap;

use crate::episodes::distill_models::*;
use crate::identity::InstanceIdentity;

// ============================================================================
// T3: abstract_lesson — Lesson extraction
// ============================================================================

/// Input for lesson extraction: the note titles, tags, and content from an episode.
#[derive(Debug, Clone)]
pub struct EpisodeContent {
    /// Titles of the notes produced in the episode.
    pub note_titles: Vec<String>,
    /// Tags from all notes combined.
    pub tags: Vec<String>,
    /// Concatenated text content of the notes.
    pub content: String,
}

/// Extract a distilled lesson from episode content using heuristics.
///
/// Portability scoring:
/// - **Local**: content contains absolute paths or UUIDs
/// - **Project**: content references project-specific names but no paths/UUIDs
/// - **Domain**: content uses domain-specific terms (language names, frameworks)
/// - **Universal**: generic software engineering patterns
///
/// Confidence is based on the number and diversity of source notes.
pub fn abstract_lesson(episode: &EpisodeContent) -> DistilledLesson {
    let portability = compute_portability(&episode.content, &episode.tags);
    let confidence = compute_confidence(episode);
    let abstract_pattern = derive_pattern(&episode.note_titles, &episode.tags);
    let domain_tags = extract_domain_tags(&episode.tags);

    DistilledLesson {
        abstract_pattern,
        domain_tags,
        portability_layer: portability,
        confidence,
    }
}

/// Compute portability layer from content signals.
fn compute_portability(content: &str, tags: &[String]) -> PortabilityLayer {
    // L1: absolute paths or UUIDs → Local
    let has_abs_path = content.contains("/Users/") || content.contains("/home/");
    let uuid_re = regex::Regex::new(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
    )
    .unwrap();
    if has_abs_path || uuid_re.is_match(content) {
        return PortabilityLayer::Local;
    }

    // Domain-specific terms (languages, frameworks, databases)
    let domain_terms = [
        "rust",
        "python",
        "java",
        "typescript",
        "go",
        "neo4j",
        "postgres",
        "redis",
        "docker",
        "kubernetes",
        "react",
        "vue",
        "angular",
        "graphql",
        "grpc",
        "kafka",
        "rabbitmq",
        "elasticsearch",
        "mongodb",
        "sqlite",
    ];
    let content_lower = content.to_lowercase();
    let tags_lower: Vec<String> = tags.iter().map(|t| t.to_lowercase()).collect();

    let has_domain_term = domain_terms
        .iter()
        .any(|term| content_lower.contains(term) || tags_lower.iter().any(|t| t.contains(term)));

    if has_domain_term {
        return PortabilityLayer::Domain;
    }

    // Check for project-specific indicators (proper nouns, config references)
    let project_indicators = [
        "config.",
        ".yaml",
        ".toml",
        "Cargo.",
        "package.json",
        "Makefile",
    ];
    let has_project_indicator = project_indicators.iter().any(|ind| content.contains(ind));

    if has_project_indicator {
        return PortabilityLayer::Project;
    }

    // Default: universal
    PortabilityLayer::Universal
}

/// Compute confidence based on source note quantity and diversity.
fn compute_confidence(episode: &EpisodeContent) -> f64 {
    let title_count = episode.note_titles.len() as f64;
    let tag_count = episode.tags.len() as f64;
    let content_len = episode.content.len() as f64;

    // Base confidence from number of notes (more sources = higher confidence)
    let note_factor = (title_count / 5.0).min(1.0); // caps at 5 notes

    // Tag diversity bonus
    let tag_factor = (tag_count / 10.0).min(1.0); // caps at 10 tags

    // Content length bonus (more content = more evidence)
    let content_factor = (content_len / 2000.0).min(1.0); // caps at 2000 chars

    // Weighted average
    let raw = 0.4 * note_factor + 0.3 * tag_factor + 0.3 * content_factor;

    // Clamp to [0.1, 1.0] — never zero confidence
    raw.clamp(0.1, 1.0)
}

/// Derive an abstract pattern description from note titles and tags.
fn derive_pattern(titles: &[String], tags: &[String]) -> String {
    if titles.is_empty() {
        return "General observation from episode content.".to_string();
    }

    let tag_str = if tags.is_empty() {
        String::new()
    } else {
        format!(" [{}]", tags.join(", "))
    };

    if titles.len() == 1 {
        format!("Pattern from: {}{}", titles[0], tag_str)
    } else {
        let summary: Vec<&str> = titles.iter().map(|t| t.as_str()).take(3).collect();
        let suffix = if titles.len() > 3 {
            format!(" (+{} more)", titles.len() - 3)
        } else {
            String::new()
        };
        format!(
            "Pattern synthesized from: {}{}{}",
            summary.join("; "),
            suffix,
            tag_str
        )
    }
}

/// Extract domain tags (deduplicated, lowercased).
fn extract_domain_tags(tags: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    for tag in tags {
        let lower = tag.to_lowercase();
        if seen.insert(lower.clone()) {
            result.push(lower);
        }
    }
    result
}

// ============================================================================
// T5: create_envelope — Sign and wrap into DistillationEnvelope
// ============================================================================

/// Create a signed distillation envelope from anonymized content and a lesson.
///
/// - Computes a SHA-256 hash of the anonymized content.
/// - Signs the hash with the instance's Ed25519 key.
/// - Assembles the envelope with metadata and trust proof.
pub fn create_envelope(
    identity: &InstanceIdentity,
    anonymized_content: &str,
    lesson: DistilledLesson,
    anonymization_report: Option<AnonymizationReport>,
) -> DistillationEnvelope {
    // Compute content hash (SHA-256)
    let mut hasher = Sha256::new();
    hasher.update(anonymized_content.as_bytes());
    let hash_bytes = hasher.finalize();
    let content_hash = hex::encode(hash_bytes);

    // Sign the content hash with the instance identity
    let signature = identity.sign(content_hash.as_bytes());
    let signature_hex = hex::encode(signature.to_bytes());

    // Determine sensitivity level from report
    let sensitivity_level = match &anonymization_report {
        Some(report) if report.redacted_count > 0 => SensitivityLevel::Restricted,
        _ => SensitivityLevel::Public,
    };

    // Build trust scores
    let mut trust_scores = HashMap::new();
    trust_scores.insert("confidence".to_string(), lesson.confidence);

    let meta = DistillationMeta {
        pipeline_version: "1.0".to_string(),
        sensitivity_level,
        quality_score: lesson.confidence,
        content_hash,
    };

    let trust_proof = TrustProof {
        source_did: identity.did_key().to_string(),
        signature_hex,
        trust_scores,
    };

    DistillationEnvelope {
        lesson,
        anonymized_content: anonymized_content.to_string(),
        meta,
        trust_proof,
        anonymization_report,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_episode_content() -> EpisodeContent {
        EpisodeContent {
            note_titles: vec![
                "Use UNWIND for batch operations".to_string(),
                "Index all PRODUCED_DURING relations".to_string(),
            ],
            tags: vec![
                "neo4j".to_string(),
                "performance".to_string(),
                "Rust".to_string(),
            ],
            content: "When working with neo4j, always use UNWIND for batch inserts. \
                      This avoids N+1 query patterns and improves throughput by 10x."
                .to_string(),
        }
    }

    // --- Portability scoring tests ---

    #[test]
    fn test_portability_local_with_abs_path() {
        let episode = EpisodeContent {
            note_titles: vec!["Setup".to_string()],
            tags: vec![],
            content: "Edit the file at /Users/johndoe/projects/app/src/main.rs".to_string(),
        };
        let lesson = abstract_lesson(&episode);
        assert_eq!(lesson.portability_layer, PortabilityLayer::Local);
    }

    #[test]
    fn test_portability_local_with_uuid() {
        let episode = EpisodeContent {
            note_titles: vec!["Debug".to_string()],
            tags: vec![],
            content: "The node 550e8400-e29b-41d4-a716-446655440000 is broken".to_string(),
        };
        let lesson = abstract_lesson(&episode);
        assert_eq!(lesson.portability_layer, PortabilityLayer::Local);
    }

    #[test]
    fn test_portability_domain_with_tags() {
        let episode = sample_episode_content();
        let lesson = abstract_lesson(&episode);
        assert_eq!(lesson.portability_layer, PortabilityLayer::Domain);
    }

    #[test]
    fn test_portability_project_with_config() {
        let episode = EpisodeContent {
            note_titles: vec!["Config update".to_string()],
            tags: vec![],
            content: "Update the settings in config.yaml to enable feature X".to_string(),
        };
        let lesson = abstract_lesson(&episode);
        assert_eq!(lesson.portability_layer, PortabilityLayer::Project);
    }

    #[test]
    fn test_portability_universal() {
        let episode = EpisodeContent {
            note_titles: vec!["Error handling".to_string()],
            tags: vec![],
            content: "Always validate input before processing to avoid unexpected failures."
                .to_string(),
        };
        let lesson = abstract_lesson(&episode);
        assert_eq!(lesson.portability_layer, PortabilityLayer::Universal);
    }

    // --- Confidence tests ---

    #[test]
    fn test_confidence_increases_with_notes() {
        let small = EpisodeContent {
            note_titles: vec!["One".to_string()],
            tags: vec![],
            content: "short".to_string(),
        };
        let large = EpisodeContent {
            note_titles: vec![
                "One".to_string(),
                "Two".to_string(),
                "Three".to_string(),
                "Four".to_string(),
            ],
            tags: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            content: "A much longer content with many details about the pattern observed \
                      during the implementation of a complex feature that spans multiple files."
                .to_string(),
        };
        let c_small = abstract_lesson(&small).confidence;
        let c_large = abstract_lesson(&large).confidence;
        assert!(
            c_large > c_small,
            "More notes should yield higher confidence: {} vs {}",
            c_large,
            c_small
        );
    }

    #[test]
    fn test_confidence_minimum() {
        let empty = EpisodeContent {
            note_titles: vec![],
            tags: vec![],
            content: String::new(),
        };
        let lesson = abstract_lesson(&empty);
        assert!(
            lesson.confidence >= 0.1,
            "Confidence should never be below 0.1"
        );
    }

    // --- Pattern derivation tests ---

    #[test]
    fn test_pattern_single_title() {
        let episode = EpisodeContent {
            note_titles: vec!["Use UNWIND".to_string()],
            tags: vec!["neo4j".to_string()],
            content: "content".to_string(),
        };
        let lesson = abstract_lesson(&episode);
        assert!(lesson.abstract_pattern.contains("Use UNWIND"));
        assert!(lesson.abstract_pattern.contains("neo4j"));
    }

    #[test]
    fn test_pattern_multiple_titles() {
        let episode = sample_episode_content();
        let lesson = abstract_lesson(&episode);
        assert!(lesson.abstract_pattern.contains("synthesized"));
        assert!(lesson.abstract_pattern.contains("UNWIND"));
    }

    #[test]
    fn test_domain_tags_deduped() {
        let episode = EpisodeContent {
            note_titles: vec!["test".to_string()],
            tags: vec!["Neo4j".to_string(), "neo4j".to_string(), "Rust".to_string()],
            content: "content about neo4j".to_string(),
        };
        let lesson = abstract_lesson(&episode);
        assert_eq!(lesson.domain_tags.len(), 2);
        assert!(lesson.domain_tags.contains(&"neo4j".to_string()));
        assert!(lesson.domain_tags.contains(&"rust".to_string()));
    }

    // --- Envelope creation tests ---

    #[test]
    fn test_create_envelope_basic() {
        let identity = InstanceIdentity::generate();
        let lesson = DistilledLesson {
            abstract_pattern: "Always index relations".to_string(),
            domain_tags: vec!["neo4j".to_string()],
            portability_layer: PortabilityLayer::Domain,
            confidence: 0.85,
        };

        let envelope = create_envelope(&identity, "clean content", lesson, None);

        assert_eq!(envelope.meta.pipeline_version, "1.0");
        assert_eq!(envelope.meta.sensitivity_level, SensitivityLevel::Public);
        assert!(!envelope.meta.content_hash.is_empty());
        assert_eq!(envelope.trust_proof.source_did, identity.did_key());
        assert!(!envelope.trust_proof.signature_hex.is_empty());
        assert_eq!(envelope.anonymized_content, "clean content");
    }

    #[test]
    fn test_create_envelope_with_report() {
        let identity = InstanceIdentity::generate();
        let lesson = DistilledLesson {
            abstract_pattern: "test".to_string(),
            domain_tags: vec![],
            portability_layer: PortabilityLayer::Universal,
            confidence: 0.5,
        };
        let report = AnonymizationReport {
            redacted_count: 3,
            patterns_applied: vec!["H8-abs-path".to_string()],
            blocked_l3: false,
            consent_stats: None,
        };

        let envelope = create_envelope(&identity, "redacted content", lesson, Some(report));

        assert_eq!(
            envelope.meta.sensitivity_level,
            SensitivityLevel::Restricted
        );
        assert!(envelope.anonymization_report.is_some());
        assert_eq!(envelope.anonymization_report.unwrap().redacted_count, 3);
    }

    #[test]
    fn test_envelope_signature_verifiable() {
        let identity = InstanceIdentity::generate();
        let lesson = DistilledLesson {
            abstract_pattern: "test".to_string(),
            domain_tags: vec![],
            portability_layer: PortabilityLayer::Universal,
            confidence: 0.7,
        };

        let envelope = create_envelope(&identity, "verifiable content", lesson, None);

        // Verify the signature: sign(content_hash) should be verifiable
        let sig_bytes = hex::decode(&envelope.trust_proof.signature_hex).unwrap();
        let signature =
            ed25519_dalek::Signature::from_bytes(sig_bytes.as_slice().try_into().unwrap());
        let valid = identity.verify(envelope.meta.content_hash.as_bytes(), &signature);
        assert!(
            valid,
            "Envelope signature should be verifiable with the identity"
        );
    }

    #[test]
    fn test_envelope_content_hash_deterministic() {
        let identity = InstanceIdentity::generate();
        let lesson1 = DistilledLesson {
            abstract_pattern: "t".to_string(),
            domain_tags: vec![],
            portability_layer: PortabilityLayer::Universal,
            confidence: 0.5,
        };
        let lesson2 = lesson1.clone();

        let e1 = create_envelope(&identity, "same content", lesson1, None);
        let e2 = create_envelope(&identity, "same content", lesson2, None);

        assert_eq!(
            e1.meta.content_hash, e2.meta.content_hash,
            "Same content should produce same hash"
        );
    }
}
