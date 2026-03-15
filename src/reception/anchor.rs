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
    /// Hex-encoded Ed25519 signature over the content hash.
    pub signature_hex: String,
    /// When the tombstone was issued.
    pub issued_at: DateTime<Utc>,
    /// Optional human-readable reason for revocation.
    pub reason: Option<String>,
}

/// Tombstone registry for revoked content hashes.
///
/// Stores [`SignedTombstone`] entries keyed by content hash.
/// Backward-compatible: the legacy `apply_tombstone` / `is_revoked` API
/// still works by creating unsigned placeholder tombstones internally.
#[derive(Debug, Clone, Default)]
pub struct TombstoneRegistry {
    /// Map of content hashes to their signed tombstone records.
    revoked: HashMap<String, SignedTombstone>,
}

impl TombstoneRegistry {
    /// Create a new empty tombstone registry.
    pub fn new() -> Self {
        Self {
            revoked: HashMap::new(),
        }
    }

    /// Apply a tombstone for the given content hash (legacy/unsigned).
    ///
    /// Returns `true` if the tombstone was newly applied, `false` if already present.
    pub fn apply_tombstone(&mut self, content_hash: &str) -> bool {
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

    /// Apply a signed tombstone. Returns `true` if it was newly inserted.
    pub fn apply_signed_tombstone(&mut self, tombstone: SignedTombstone) -> bool {
        if self.revoked.contains_key(&tombstone.content_hash) {
            return false;
        }
        self.revoked
            .insert(tombstone.content_hash.clone(), tombstone);
        true
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
        let tombstone = SignedTombstone {
            content_hash: "hash_signed".to_string(),
            issuer_did: "did:key:zIssuer".to_string(),
            signature_hex: "abcd1234".to_string(),
            issued_at: chrono::Utc::now(),
            reason: Some("GDPR request".to_string()),
        };
        assert!(registry.apply_signed_tombstone(tombstone.clone()));
        assert!(!registry.apply_signed_tombstone(tombstone)); // duplicate
        assert!(registry.is_revoked("hash_signed"));
    }

    #[test]
    fn test_get_tombstone() {
        let mut registry = TombstoneRegistry::new();
        assert!(registry.get_tombstone("missing").is_none());

        let tombstone = SignedTombstone {
            content_hash: "hash_get".to_string(),
            issuer_did: "did:key:zAlice".to_string(),
            signature_hex: "beef".to_string(),
            issued_at: chrono::Utc::now(),
            reason: None,
        };
        registry.apply_signed_tombstone(tombstone);

        let retrieved = registry.get_tombstone("hash_get").unwrap();
        assert_eq!(retrieved.issuer_did, "did:key:zAlice");
        assert_eq!(retrieved.signature_hex, "beef");
        assert!(retrieved.reason.is_none());
    }

    #[test]
    fn test_list_tombstones() {
        let mut registry = TombstoneRegistry::new();
        assert!(registry.list_tombstones().is_empty());

        registry.apply_tombstone("hash_a");
        registry.apply_signed_tombstone(SignedTombstone {
            content_hash: "hash_b".to_string(),
            issuer_did: "did:key:zBob".to_string(),
            signature_hex: "cafe".to_string(),
            issued_at: chrono::Utc::now(),
            reason: Some("test".to_string()),
        });

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
}
