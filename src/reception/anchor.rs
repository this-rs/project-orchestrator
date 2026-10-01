//! Anchor replayed notes to the local knowledge context.
//!
//! Links imported P2P notes to existing local tags and supports
//! tombstone-based content revocation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::replay::ReplayedNote;

/// Result of anchoring replayed notes to local context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorResult {
    /// Number of notes successfully anchored to local tags.
    pub anchored_count: u32,
    /// Tags that could form cross-references (synapses) between imported and local notes.
    pub synapse_candidates: Vec<String>,
    /// Whether a tombstone was applied during this anchoring pass.
    pub tombstone_applied: bool,
}

/// A cryptographically signed tombstone for content revocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedTombstone {
    /// The content hash being revoked.
    pub content_hash: String,
    /// DID of the issuer who signed this tombstone.
    pub issuer_did: String,
    /// Hex-encoded Ed25519 signature over the domain-separated payload
    /// (`po-tombstone-v1`: content hash, issuer, issued_at, reason).
    pub signature_hex: String,
    /// When the tombstone was issued.
    pub issued_at: DateTime<Utc>,
    /// Optional human-readable reason for revocation (covered by the signature).
    pub reason: Option<String>,
}

/// Tombstone registry for revoked content hashes.
///
/// Stores [`SignedTombstone`] entries keyed by content hash.
/// Tombstones MUST go through `apply_signed_tombstone`, which checks the
/// signature AND that the issuer is the known original owner of the content
/// (registered with `register_owner` / `register_envelope_owner`). The unsigned
/// `apply_tombstone` is test-only.
#[derive(Debug, Clone, Default)]
pub struct TombstoneRegistry {
    /// Map of content hashes to their signed tombstone records.
    revoked: HashMap<String, SignedTombstone>,
    /// Known original owner (DID) per content hash.
    owners: HashMap<String, String>,
}

impl TombstoneRegistry {
    /// Create a new empty tombstone registry.
    pub fn new() -> Self {
        Self {
            revoked: HashMap::new(),
            owners: HashMap::new(),
        }
    }

    /// Record the original owner of a content hash (first registration wins,
    /// so a later envelope cannot re-attribute known content).
    pub fn register_owner(&mut self, content_hash: &str, owner_did: &str) {
        self.owners
            .entry(content_hash.to_string())
            .or_insert_with(|| owner_did.to_string());
    }

    /// Record the owner of an envelope's content: its `trust_proof.source_did`,
    /// but ONLY once the envelope has passed [`verify_envelope`] (Ed25519
    /// signature by that DID over the content hash, and content integrity).
    ///
    /// Fail-closed: an unverifiable envelope registers nothing. Without this
    /// check anyone could forge an envelope for a known content hash, win the
    /// first-registration race and then revoke the victim's content.
    ///
    /// [`verify_envelope`]: crate::reception::verify::verify_envelope
    pub fn register_envelope_owner(
        &mut self,
        envelope: &crate::episodes::distill_models::DistillationEnvelope,
    ) -> Result<(), String> {
        let verified = crate::reception::verify::verify_envelope(envelope)
            .map_err(|e| format!("envelope not verified, owner not registered: {e}"))?;
        self.register_owner(&verified.envelope.meta.content_hash, &verified.source_did);
        Ok(())
    }

    /// Known owner of a content hash, if any.
    pub fn owner_of(&self, content_hash: &str) -> Option<&str> {
        self.owners.get(content_hash).map(String::as_str)
    }

    /// Apply a tombstone for the given content hash (unsigned, test-only).
    ///
    /// Bypasses signature and authority checks, so it is not available outside
    /// tests. Returns `true` if newly applied, `false` if already present.
    #[cfg(test)]
    pub(crate) fn apply_tombstone(&mut self, content_hash: &str) -> bool {
        if self.revoked.contains_key(content_hash) {
            return false;
        }
        let tombstone = SignedTombstone {
            content_hash: content_hash.to_string(),
            issuer_did: String::new(),
            signature_hex: String::new(),
            issued_at: Utc::now(),
            reason: None,
        };
        self.revoked.insert(content_hash.to_string(), tombstone);
        true
    }

    /// Apply a signed tombstone after verifying its Ed25519 signature and that
    /// its issuer is the known owner of the content (fail-closed).
    ///
    /// Returns `Err` if the signature does not verify, the owner is unknown or
    /// the issuer is not the owner; `Ok(true)` if newly inserted, `Ok(false)`
    /// if already present.
    pub fn apply_signed_tombstone(&mut self, tombstone: SignedTombstone) -> Result<bool, String> {
        crate::sharing::tombstone::verify_tombstone_authority(
            &tombstone,
            self.owner_of(&tombstone.content_hash),
        )?;
        if self.revoked.contains_key(&tombstone.content_hash) {
            return Ok(false);
        }
        self.revoked
            .insert(tombstone.content_hash.clone(), tombstone);
        Ok(true)
    }

    /// Check if a content hash has been tombstoned.
    pub fn is_revoked(&self, content_hash: &str) -> bool {
        self.revoked.contains_key(content_hash)
    }

    /// Get the full signed tombstone for a content hash, if present.
    pub fn get_tombstone(&self, content_hash: &str) -> Option<&SignedTombstone> {
        self.revoked.get(content_hash)
    }

    /// List all tombstones in the registry.
    pub fn list_tombstones(&self) -> Vec<&SignedTombstone> {
        self.revoked.values().collect()
    }
}

/// Anchor replayed notes to local context by matching domain tags.
///
/// For each note, checks if any of its tags match the provided local tags.
/// Matching tags become "synapse candidates" — potential cross-references
/// between imported and local knowledge.
pub fn anchor_notes(notes: &[ReplayedNote], local_tags: &[String]) -> AnchorResult {
    let local_set: HashSet<String> = local_tags.iter().map(|t| t.to_lowercase()).collect();

    let mut anchored_count: u32 = 0;
    let mut synapse_set: HashSet<String> = HashSet::new();

    for note in notes {
        let note_tags_lower: HashSet<String> = note.tags.iter().map(|t| t.to_lowercase()).collect();
        let overlap: Vec<String> = note_tags_lower.intersection(&local_set).cloned().collect();

        if !overlap.is_empty() {
            anchored_count += 1;
            for tag in overlap {
                synapse_set.insert(tag);
            }
        }
    }

    let mut synapse_candidates: Vec<String> = synapse_set.into_iter().collect();
    synapse_candidates.sort();

    AnchorResult {
        anchored_count,
        synapse_candidates,
        tombstone_applied: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::distill_models::PortabilityLayer;

    fn make_note(tags: Vec<String>) -> ReplayedNote {
        ReplayedNote {
            content: "[P2P Import] test".to_string(),
            note_type: "p2p_import".to_string(),
            importance: 0.8,
            tags,
            source_did: "did:key:zTest".to_string(),
            portability: PortabilityLayer::Domain,
        }
    }

    #[test]
    fn test_anchor_matching_tags() {
        let notes = vec![make_note(vec!["rust".to_string(), "neo4j".to_string()])];
        let local_tags = vec!["rust".to_string(), "typescript".to_string()];
        let result = anchor_notes(&notes, &local_tags);
        assert_eq!(result.anchored_count, 1);
        assert!(result.synapse_candidates.contains(&"rust".to_string()));
    }

    #[test]
    fn test_anchor_no_matching_tags() {
        let notes = vec![make_note(vec!["python".to_string()])];
        let local_tags = vec!["rust".to_string()];
        let result = anchor_notes(&notes, &local_tags);
        assert_eq!(result.anchored_count, 0);
        assert!(result.synapse_candidates.is_empty());
    }

    #[test]
    fn test_anchor_multiple_notes() {
        let notes = vec![
            make_note(vec!["rust".to_string()]),
            make_note(vec!["python".to_string()]),
            make_note(vec!["rust".to_string(), "neo4j".to_string()]),
        ];
        let local_tags = vec!["rust".to_string(), "neo4j".to_string()];
        let result = anchor_notes(&notes, &local_tags);
        assert_eq!(result.anchored_count, 2); // notes 0 and 2
    }

    #[test]
    fn test_anchor_case_insensitive() {
        let notes = vec![make_note(vec!["Rust".to_string(), "NEO4J".to_string()])];
        let local_tags = vec!["rust".to_string(), "neo4j".to_string()];
        let result = anchor_notes(&notes, &local_tags);
        assert_eq!(result.anchored_count, 1);
        assert_eq!(result.synapse_candidates.len(), 2);
    }

    #[test]
    fn test_anchor_empty_notes() {
        let result = anchor_notes(&[], &["rust".to_string()]);
        assert_eq!(result.anchored_count, 0);
        assert!(result.synapse_candidates.is_empty());
    }

    #[test]
    fn test_tombstone_apply() {
        let mut registry = TombstoneRegistry::new();
        assert!(registry.apply_tombstone("hash123"));
        assert!(!registry.apply_tombstone("hash123")); // duplicate
        assert!(registry.is_revoked("hash123"));
        assert!(!registry.is_revoked("hash456"));
    }

    #[test]
    fn test_tombstone_registry_default() {
        let registry = TombstoneRegistry::default();
        assert!(!registry.is_revoked("any_hash"));
    }

    #[test]
    fn test_signed_tombstone_apply() {
        let mut registry = TombstoneRegistry::new();
        let id = crate::identity::InstanceIdentity::generate();
        registry.register_owner("hash_signed", id.did_key());
        let tombstone = crate::sharing::tombstone::sign_tombstone(
            &id,
            "hash_signed".to_string(),
            chrono::Utc::now(),
            Some("GDPR request".to_string()),
        );
        assert_eq!(registry.apply_signed_tombstone(tombstone.clone()), Ok(true));
        assert_eq!(registry.apply_signed_tombstone(tombstone), Ok(false)); // duplicate
        assert!(registry.is_revoked("hash_signed"));
    }

    #[test]
    fn test_forged_tombstone_rejected() {
        let mut registry = TombstoneRegistry::new();
        let forged = SignedTombstone {
            content_hash: "hash_forged".to_string(),
            issuer_did: "did:key:zIssuer".to_string(),
            signature_hex: "0".repeat(128),
            issued_at: chrono::Utc::now(),
            reason: None,
        };
        assert!(registry.apply_signed_tombstone(forged).is_err());
        assert!(!registry.is_revoked("hash_forged"));

        // Valid signature by key A, claimed by issuer B
        let a = crate::identity::InstanceIdentity::generate();
        let b = crate::identity::InstanceIdentity::generate();
        registry.register_owner("hash_forged2", b.did_key());
        let mut t = crate::sharing::tombstone::sign_tombstone(
            &a,
            "hash_forged2".to_string(),
            chrono::Utc::now(),
            None,
        );
        t.issuer_did = b.did_key().to_string();
        assert!(registry.apply_signed_tombstone(t).is_err());
        assert!(!registry.is_revoked("hash_forged2"));
    }

    #[test]
    fn test_get_tombstone() {
        let mut registry = TombstoneRegistry::new();
        assert!(registry.get_tombstone("missing").is_none());

        let id = crate::identity::InstanceIdentity::generate();
        registry.register_owner("hash_get", id.did_key());
        let tombstone = crate::sharing::tombstone::sign_tombstone(
            &id,
            "hash_get".to_string(),
            chrono::Utc::now(),
            None,
        );
        let sig = tombstone.signature_hex.clone();
        registry.apply_signed_tombstone(tombstone).unwrap();

        let retrieved = registry.get_tombstone("hash_get").unwrap();
        assert_eq!(retrieved.issuer_did, id.did_key());
        assert_eq!(retrieved.signature_hex, sig);
        assert!(retrieved.reason.is_none());
    }

    #[test]
    fn test_list_tombstones() {
        let mut registry = TombstoneRegistry::new();
        assert!(registry.list_tombstones().is_empty());

        registry.apply_tombstone("hash_a");
        let id = crate::identity::InstanceIdentity::generate();
        registry.register_owner("hash_b", id.did_key());
        registry
            .apply_signed_tombstone(crate::sharing::tombstone::sign_tombstone(
                &id,
                "hash_b".to_string(),
                chrono::Utc::now(),
                Some("test".to_string()),
            ))
            .unwrap();

        assert_eq!(registry.list_tombstones().len(), 2);
    }

    #[test]
    fn test_legacy_tombstone_creates_unsigned_entry() {
        let mut registry = TombstoneRegistry::new();
        registry.apply_tombstone("hash_legacy");

        let entry = registry.get_tombstone("hash_legacy").unwrap();
        assert_eq!(entry.content_hash, "hash_legacy");
        assert!(entry.issuer_did.is_empty()); // unsigned placeholder
        assert!(entry.signature_hex.is_empty());
    }

    #[test]
    fn test_signed_tombstone_serialization_roundtrip() {
        let tombstone = SignedTombstone {
            content_hash: "hash_ser".to_string(),
            issuer_did: "did:key:zTest".to_string(),
            signature_hex: "deadbeef".to_string(),
            issued_at: chrono::Utc::now(),
            reason: Some("privacy".to_string()),
        };
        let json = serde_json::to_string(&tombstone).unwrap();
        let deser: SignedTombstone = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.content_hash, "hash_ser");
        assert_eq!(deser.issuer_did, "did:key:zTest");
        assert_eq!(deser.reason.as_deref(), Some("privacy"));
    }

    #[test]
    fn test_wrong_issuer_valid_signature_rejected() {
        let mut registry = TombstoneRegistry::new();
        let owner = crate::identity::InstanceIdentity::generate();
        let attacker = crate::identity::InstanceIdentity::generate();
        registry.register_owner("hash_owned", owner.did_key());
        let t = crate::sharing::tombstone::sign_tombstone(
            &attacker,
            "hash_owned".to_string(),
            chrono::Utc::now(),
            None,
        );
        assert!(registry.apply_signed_tombstone(t).is_err());
        assert!(!registry.is_revoked("hash_owned"));
    }

    #[test]
    fn test_unknown_owner_rejected() {
        let mut registry = TombstoneRegistry::new();
        let id = crate::identity::InstanceIdentity::generate();
        let t = crate::sharing::tombstone::sign_tombstone(
            &id,
            "hash_nobody".to_string(),
            chrono::Utc::now(),
            None,
        );
        assert!(registry.apply_signed_tombstone(t).is_err());
        assert!(!registry.is_revoked("hash_nobody"));
    }

    #[test]
    fn test_altered_reason_rejected_by_registry() {
        let mut registry = TombstoneRegistry::new();
        let id = crate::identity::InstanceIdentity::generate();
        registry.register_owner("hash_reason", id.did_key());
        let mut t = crate::sharing::tombstone::sign_tombstone(
            &id,
            "hash_reason".to_string(),
            chrono::Utc::now(),
            Some("original".into()),
        );
        t.reason = Some("altered".into());
        assert!(registry.apply_signed_tombstone(t).is_err());
    }

    #[test]
    fn test_owner_registration_first_wins() {
        let mut registry = TombstoneRegistry::new();
        registry.register_owner("h", "did:key:zA");
        registry.register_owner("h", "did:key:zB");
        assert_eq!(registry.owner_of("h"), Some("did:key:zA"));
    }

    /// Build an envelope whose content hash is the real digest of its lesson,
    /// with `signer` producing the trust-proof signature and `claimed_did`
    /// as the declared source.
    fn make_envelope(
        signer: &crate::identity::InstanceIdentity,
        claimed_did: &str,
    ) -> crate::episodes::distill_models::DistillationEnvelope {
        use crate::episodes::distill_models::{
            DistillationEnvelope, DistillationMeta, DistilledLesson, SensitivityLevel, TrustProof,
        };
        use sha2::{Digest, Sha256};
        let lesson = DistilledLesson {
            abstract_pattern: "Validate inputs".to_string(),
            domain_tags: vec!["rust".to_string()],
            portability_layer: PortabilityLayer::Domain,
            confidence: 0.9,
        };
        let content_hash = hex::encode(Sha256::digest(
            serde_json::to_string(&lesson).unwrap().as_bytes(),
        ));
        let sig = signer.sign(content_hash.as_bytes());
        DistillationEnvelope {
            lesson,
            anonymized_content: "x".to_string(),
            meta: DistillationMeta {
                pipeline_version: "1.0".to_string(),
                sensitivity_level: SensitivityLevel::Public,
                quality_score: 0.8,
                content_hash,
            },
            trust_proof: TrustProof {
                source_did: claimed_did.to_string(),
                signature_hex: hex::encode(sig.to_bytes()),
                trust_scores: HashMap::new(),
            },
            anonymization_report: None,
        }
    }

    #[test]
    fn test_forged_envelope_does_not_register_owner() {
        let victim = crate::identity::InstanceIdentity::generate();
        let attacker = crate::identity::InstanceIdentity::generate();
        // Attacker claims the content under their own DID but cannot produce
        // a valid signature for it: signs with the victim-unrelated key while
        // declaring a different DID.
        let victim_env = make_envelope(&victim, victim.did_key());
        let mut forged = victim_env.clone();
        forged.trust_proof.source_did = attacker.did_key().to_string();
        // signature is the victim's, DID is the attacker's: must not verify.

        let mut registry = TombstoneRegistry::new();
        let _ = registry.register_envelope_owner(&forged);
        assert_eq!(registry.owner_of(&forged.meta.content_hash), None);

        // Attacker-signed tombstone is refused (no owner known, fail-closed).
        let t = crate::sharing::tombstone::sign_tombstone(
            &attacker,
            forged.meta.content_hash.clone(),
            chrono::Utc::now(),
            None,
        );
        assert!(registry.apply_signed_tombstone(t).is_err());
        assert!(!registry.is_revoked(&forged.meta.content_hash));

        // The genuine envelope still registers the real owner afterwards.
        registry.register_envelope_owner(&victim_env).unwrap();
        assert_eq!(
            registry.owner_of(&victim_env.meta.content_hash),
            Some(victim.did_key())
        );
    }

    #[test]
    fn test_envelope_with_tampered_content_does_not_register_owner() {
        let id = crate::identity::InstanceIdentity::generate();
        let mut env = make_envelope(&id, id.did_key());
        env.lesson.abstract_pattern = "tampered".to_string();
        let mut registry = TombstoneRegistry::new();
        assert!(registry.register_envelope_owner(&env).is_err());
        assert_eq!(registry.owner_of(&env.meta.content_hash), None);
    }
}
