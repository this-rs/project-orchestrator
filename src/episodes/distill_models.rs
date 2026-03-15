//! Distillation pipeline data models.
//!
//! These types support the full distillation pipeline:
//! collect -> abstract -> anonymize -> envelope -> package.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Sensitivity levels
// ============================================================================

/// Classification of content sensitivity for the privacy gate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensitivityLevel {
    /// Safe for public export.
    #[default]
    Public,
    /// Contains project-specific details (paths, IPs, UUIDs) — needs redaction.
    Restricted,
    /// Contains PII or internal identifiers — must be anonymized.
    Confidential,
    /// Contains secrets (API keys, passwords, private keys) — block export.
    Forbidden,
}

// ============================================================================
// Privacy mode
// ============================================================================

/// Controls how aggressively the anonymizer runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyMode {
    /// Block on L2 and L3 findings.
    Strict,
    /// Redact L2, block on L3 (default behaviour).
    #[default]
    Standard,
    /// Redact L2 and L3 (never block). USE WITH CAUTION.
    Relaxed,
}

// ============================================================================
// Sharing policy & consent (RFC Privacy §2.4)
// ============================================================================

/// How the project handles sharing of knowledge artifacts.
///
/// Controls the default behavior for notes/decisions in the distillation
/// pipeline. Stored on `ProjectNode.sharing_policy`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingPolicy {
    /// Global sharing mode: manual (human approves each), suggest (AI suggests,
    /// human confirms), auto (AI decides based on score thresholds).
    #[serde(default)]
    pub mode: SharingMode,
    /// Per-type overrides: key = note_type (e.g. "guideline", "gotcha"),
    /// value = action (auto/never/review).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub type_overrides: HashMap<String, SharingAction>,
    /// Whether the L3 scan is enabled at write time (always true, non-overridable).
    #[serde(default = "default_true")]
    pub l3_scan_enabled: bool,
    /// Minimum shareability score (0.0-1.0) for auto mode.
    #[serde(default = "default_min_score")]
    pub min_shareability_score: f64,
    /// Whether sharing is globally enabled for this project.
    /// Fresh install = false (opt-in strict, GDPR Art. 25).
    #[serde(default)]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

fn default_min_score() -> f64 {
    0.7
}

impl Default for SharingPolicy {
    fn default() -> Self {
        Self {
            mode: SharingMode::Manual,
            type_overrides: HashMap::new(),
            l3_scan_enabled: true,
            min_shareability_score: 0.7,
            enabled: false, // opt-in by default
        }
    }
}

/// Global sharing mode for a project.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharingMode {
    /// Human must approve every sharing action.
    #[default]
    Manual,
    /// AI suggests, human confirms.
    Suggest,
    /// AI decides automatically based on score thresholds.
    Auto,
}

/// Per-type sharing action override.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharingAction {
    /// Automatically shared if score >= threshold.
    Auto,
    /// Never shared regardless of score.
    Never,
    /// Always requires human review.
    Review,
}

/// Consent status for an individual artifact (note or decision).
///
/// Stored on `NoteNode.sharing_consent`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharingConsent {
    /// No consent decision has been made yet (default).
    #[default]
    NotSet,
    /// User explicitly allowed sharing of this artifact.
    ExplicitAllow,
    /// User explicitly denied sharing of this artifact.
    ExplicitDeny,
    /// Consent was automatically granted by policy (auto mode + score above threshold).
    PolicyAuto,
}

/// Audit trail entry for a sharing event (GDPR Art. 30).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingEvent {
    /// Unique event identifier.
    pub id: String,
    /// What was shared (content hash of the artifact).
    pub content_hash: String,
    /// Type of artifact shared (note, decision, skill_package).
    pub artifact_type: String,
    /// Action performed (shared, retracted, consent_changed).
    pub action: String,
    /// DID of the source instance.
    pub source_did: String,
    /// DID of the target peer (if point-to-point) or "broadcast".
    pub target_did: String,
    /// When the event occurred.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Consent status at the time of sharing.
    pub consent: SharingConsent,
    /// Privacy mode active at the time.
    pub privacy_mode: PrivacyMode,
    /// Optional human-readable reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

// ============================================================================
// Consent stats for enriched privacy report (RFC Privacy §2.3)
// ============================================================================

/// Statistics about consent gate filtering during distillation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsentStats {
    /// Number of artifacts that passed the consent gate.
    pub consent_allowed: u32,
    /// Number of artifacts denied by consent/policy.
    pub consent_denied: u32,
    /// Number of artifacts pending review (suggest mode).
    pub consent_pending: u32,
    /// Details for each denied artifact.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub denied_reasons: Vec<DenialDetail>,
}

/// Detail about why a specific artifact was denied sharing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenialDetail {
    /// Identifier of the denied artifact (content hash or note id).
    pub artifact_id: String,
    /// Type of artifact (note, decision).
    pub artifact_type: String,
    /// Reason for denial.
    pub reason: DenialReason,
}

/// Reason an artifact was denied sharing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenialReason {
    /// User explicitly denied sharing (SharingConsent::ExplicitDeny).
    ExplicitDeny,
    /// Type is set to "never" in the project's SharingPolicy.
    TypeNeverPolicy,
    /// Shareability score below the policy threshold.
    InsufficientScore,
    /// Sharing mode requires manual review.
    ManualRequired,
    /// Sharing is globally disabled for this project.
    SharingDisabled,
}

// ============================================================================
// Portability layer
// ============================================================================

/// How portable / universal a lesson is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortabilityLayer {
    /// Only useful within this specific project instance.
    Local,
    /// Useful within the same project across instances.
    Project,
    /// Useful across projects in the same domain (e.g., all Rust codebases).
    #[default]
    Domain,
    /// Universally applicable pattern.
    Universal,
}

// ============================================================================
// Lesson (distillation-pipeline version)
// ============================================================================

/// A distilled lesson extracted from episode content.
///
/// Unlike `models::Lesson` (which lives inside an Episode), this is the
/// output of the `abstract_lesson()` pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledLesson {
    /// Abstract, context-independent pattern description.
    pub abstract_pattern: String,
    /// Domain tags for routing (e.g. `["rust", "neo4j"]`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub domain_tags: Vec<String>,
    /// How portable this lesson is.
    pub portability_layer: PortabilityLayer,
    /// Confidence in the extraction (0.0 - 1.0).
    pub confidence: f64,
}

// ============================================================================
// Anonymization report
// ============================================================================

/// Report produced by the anonymization / privacy gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymizationReport {
    /// Number of individual redactions applied.
    pub redacted_count: u32,
    /// Names of the heuristic patterns that fired (e.g. `["H1-api-key", "H8-abs-path"]`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub patterns_applied: Vec<String>,
    /// Whether an L3-FORBIDDEN pattern was detected (export should be blocked).
    pub blocked_l3: bool,
    /// Consent gate statistics (populated when consent gate runs in the pipeline).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consent_stats: Option<ConsentStats>,
}

// ============================================================================
// Trust proof
// ============================================================================

/// Cryptographic proof of authorship and trust scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustProof {
    /// DID of the signing instance (e.g. `did:key:z6Mk...`).
    pub source_did: String,
    /// Hex-encoded Ed25519 signature over the content hash.
    pub signature_hex: String,
    /// Named trust scores (e.g. `{"quality": 0.85, "relevance": 0.9}`).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub trust_scores: HashMap<String, f64>,
}

// ============================================================================
// Distillation metadata
// ============================================================================

/// Metadata attached to every distillation envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationMeta {
    /// Pipeline version that produced this envelope (e.g. `"1.0"`).
    pub pipeline_version: String,
    /// Sensitivity classification of the content.
    pub sensitivity_level: SensitivityLevel,
    /// Quality score (0.0 - 1.0) — aggregated from lesson confidence + validation.
    pub quality_score: f64,
    /// SHA-256 hex digest of the envelope payload.
    pub content_hash: String,
}

// ============================================================================
// Distillation envelope
// ============================================================================

/// A signed, metadata-rich wrapper around distilled content.
///
/// This is the unit of exchange in SkillPackage v3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationEnvelope {
    /// The distilled lesson.
    pub lesson: DistilledLesson,
    /// Anonymized content that was distilled.
    pub anonymized_content: String,
    /// Pipeline metadata.
    pub meta: DistillationMeta,
    /// Cryptographic trust proof.
    pub trust_proof: TrustProof,
    /// Privacy report from the anonymization stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anonymization_report: Option<AnonymizationReport>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensitivity_level_default() {
        let level: SensitivityLevel = Default::default();
        assert_eq!(level, SensitivityLevel::Public);
    }

    #[test]
    fn test_sensitivity_level_ordering() {
        assert!(SensitivityLevel::Public < SensitivityLevel::Restricted);
        assert!(SensitivityLevel::Restricted < SensitivityLevel::Confidential);
        assert!(SensitivityLevel::Confidential < SensitivityLevel::Forbidden);
    }

    #[test]
    fn test_portability_layer_default() {
        let layer: PortabilityLayer = Default::default();
        assert_eq!(layer, PortabilityLayer::Domain);
    }

    #[test]
    fn test_portability_layer_ordering() {
        assert!(PortabilityLayer::Local < PortabilityLayer::Project);
        assert!(PortabilityLayer::Project < PortabilityLayer::Domain);
        assert!(PortabilityLayer::Domain < PortabilityLayer::Universal);
    }

    #[test]
    fn test_privacy_mode_default() {
        let mode: PrivacyMode = Default::default();
        assert_eq!(mode, PrivacyMode::Standard);
    }

    #[test]
    fn test_distilled_lesson_serde_roundtrip() {
        let lesson = DistilledLesson {
            abstract_pattern: "Always index graph relations".to_string(),
            domain_tags: vec!["neo4j".to_string()],
            portability_layer: PortabilityLayer::Domain,
            confidence: 0.85,
        };
        let json = serde_json::to_string(&lesson).unwrap();
        let restored: DistilledLesson = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.abstract_pattern, lesson.abstract_pattern);
        assert_eq!(restored.portability_layer, PortabilityLayer::Domain);
    }

    #[test]
    fn test_envelope_serde_roundtrip() {
        let envelope = DistillationEnvelope {
            lesson: DistilledLesson {
                abstract_pattern: "test pattern".to_string(),
                domain_tags: vec![],
                portability_layer: PortabilityLayer::Universal,
                confidence: 0.9,
            },
            anonymized_content: "some anonymized text".to_string(),
            meta: DistillationMeta {
                pipeline_version: "1.0".to_string(),
                sensitivity_level: SensitivityLevel::Public,
                quality_score: 0.8,
                content_hash: "abc123".to_string(),
            },
            trust_proof: TrustProof {
                source_did: "did:key:z6Mk...".to_string(),
                signature_hex: "deadbeef".to_string(),
                trust_scores: HashMap::from([("quality".to_string(), 0.8)]),
            },
            anonymization_report: Some(AnonymizationReport {
                redacted_count: 2,
                patterns_applied: vec!["H8-abs-path".to_string()],
                blocked_l3: false,
                consent_stats: None,
            }),
        };
        let json = serde_json::to_string_pretty(&envelope).unwrap();
        let restored: DistillationEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.meta.pipeline_version, "1.0");
        assert_eq!(restored.trust_proof.source_did, "did:key:z6Mk...");
        assert_eq!(restored.anonymization_report.unwrap().redacted_count, 2);
    }
}
