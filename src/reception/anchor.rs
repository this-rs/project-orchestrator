//! Anchor replayed notes to the local knowledge context.
//!
//! Links imported P2P notes to existing local tags and supports
//! tombstone-based content revocation.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

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

/// Tombstone registry for revoked content hashes.
#[derive(Debug, Clone, Default)]
pub struct TombstoneRegistry {
    /// Set of content hashes that have been revoked.
    revoked: HashSet<String>,
}

impl TombstoneRegistry {
    /// Create a new empty tombstone registry.
    pub fn new() -> Self {
        Self {
            revoked: HashSet::new(),
        }
    }

    /// Apply a tombstone for the given content hash.
    ///
    /// Returns `true` if the tombstone was newly applied, `false` if already present.
    pub fn apply_tombstone(&mut self, content_hash: &str) -> bool {
        self.revoked.insert(content_hash.to_string())
    }

    /// Check if a content hash has been tombstoned.
    pub fn is_revoked(&self, content_hash: &str) -> bool {
        self.revoked.contains(content_hash)
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
}
