//! TTL management for shared artifacts (Privacy MVP-B T1).
//!
//! Every artifact received via P2P carries a mandatory TTL.
//! When it expires the local copy is removed automatically.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

// ============================================================================
// TTL presets
// ============================================================================

/// Pre-defined TTL durations for shared artifacts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtlPreset {
    /// 7 days — ephemeral content.
    Ephemeral,
    /// 90 days — standard retention (default).
    #[default]
    Standard,
    /// 365 days — long-lived content.
    Durable,
    /// No expiration — requires explicit opt-in.
    Permanent,
}

impl TtlPreset {
    /// Return the duration for this preset, or `None` for Permanent.
    pub fn to_duration(&self) -> Option<Duration> {
        match self {
            TtlPreset::Ephemeral => Some(Duration::days(7)),
            TtlPreset::Standard => Some(Duration::days(90)),
            TtlPreset::Durable => Some(Duration::days(365)),
            TtlPreset::Permanent => None,
        }
    }

    /// Compute the expiration timestamp from a given start time.
    pub fn expires_at(&self, from: DateTime<Utc>) -> Option<DateTime<Utc>> {
        self.to_duration().map(|d| from + d)
    }
}

// ============================================================================
// Shared artifact metadata
// ============================================================================

/// Metadata attached to every artifact imported via P2P.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedArtifactMeta {
    /// SHA-256 content hash of the original artifact.
    pub content_hash: String,
    /// DID of the originating instance.
    pub origin_did: String,
    /// When the artifact was shared.
    pub shared_at: DateTime<Utc>,
    /// When the artifact expires locally (None = Permanent).
    pub expires_at: Option<DateTime<Utc>>,
    /// TTL preset used.
    pub ttl_preset: TtlPreset,
    /// Whether the TTL can be renewed from the source peer.
    pub renewal_allowed: bool,
    /// How many times the TTL has been renewed.
    pub renewed_count: u32,
}

impl SharedArtifactMeta {
    /// Create a new shared artifact with the given TTL preset.
    pub fn new(content_hash: String, origin_did: String, preset: TtlPreset) -> Self {
        let now = Utc::now();
        Self {
            content_hash,
            origin_did,
            shared_at: now,
            expires_at: preset.expires_at(now),
            ttl_preset: preset,
            renewal_allowed: true,
            renewed_count: 0,
        }
    }

    /// Check whether this artifact has expired.
    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(exp) => Utc::now() > exp,
            None => false,
        }
    }

    /// Renew the TTL. Returns `true` if successful, `false` if not allowed.
    pub fn renew(&mut self) -> bool {
        if !self.renewal_allowed {
            return false;
        }
        let now = Utc::now();
        self.expires_at = self.ttl_preset.expires_at(now);
        self.renewed_count += 1;
        true
    }
}

// ============================================================================
// Sweep logic
// ============================================================================

/// Identify expired artifacts from a list.
pub fn find_expired(artifacts: &[SharedArtifactMeta]) -> Vec<&SharedArtifactMeta> {
    artifacts.iter().filter(|a| a.is_expired()).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ttl_preset_durations() {
        assert_eq!(TtlPreset::Ephemeral.to_duration(), Some(Duration::days(7)));
        assert_eq!(TtlPreset::Standard.to_duration(), Some(Duration::days(90)));
        assert_eq!(TtlPreset::Durable.to_duration(), Some(Duration::days(365)));
        assert_eq!(TtlPreset::Permanent.to_duration(), None);
    }

    #[test]
    fn test_default_is_standard() {
        assert_eq!(TtlPreset::default(), TtlPreset::Standard);
    }

    #[test]
    fn test_expires_at_permanent_is_none() {
        let now = Utc::now();
        assert!(TtlPreset::Permanent.expires_at(now).is_none());
    }

    #[test]
    fn test_expires_at_standard() {
        let now = Utc::now();
        let exp = TtlPreset::Standard.expires_at(now).unwrap();
        let diff = exp - now;
        assert_eq!(diff.num_days(), 90);
    }

    #[test]
    fn test_shared_artifact_not_expired_when_fresh() {
        let meta = SharedArtifactMeta::new(
            "hash123".to_string(),
            "did:key:alice".to_string(),
            TtlPreset::Standard,
        );
        assert!(!meta.is_expired());
    }

    #[test]
    fn test_shared_artifact_permanent_never_expires() {
        let meta = SharedArtifactMeta::new(
            "hash123".to_string(),
            "did:key:alice".to_string(),
            TtlPreset::Permanent,
        );
        assert!(!meta.is_expired());
        assert!(meta.expires_at.is_none());
    }

    #[test]
    fn test_shared_artifact_expired() {
        let mut meta = SharedArtifactMeta::new(
            "hash123".to_string(),
            "did:key:alice".to_string(),
            TtlPreset::Ephemeral,
        );
        // Force expiration in the past
        meta.expires_at = Some(Utc::now() - Duration::hours(1));
        assert!(meta.is_expired());
    }

    #[test]
    fn test_renew_resets_expiration() {
        let mut meta = SharedArtifactMeta::new(
            "hash123".to_string(),
            "did:key:alice".to_string(),
            TtlPreset::Ephemeral,
        );
        // Force expiration in the past
        meta.expires_at = Some(Utc::now() - Duration::hours(1));
        assert!(meta.is_expired());

        assert!(meta.renew());
        assert!(!meta.is_expired());
        assert_eq!(meta.renewed_count, 1);
    }

    #[test]
    fn test_renew_not_allowed() {
        let mut meta = SharedArtifactMeta::new(
            "hash123".to_string(),
            "did:key:alice".to_string(),
            TtlPreset::Standard,
        );
        meta.renewal_allowed = false;
        assert!(!meta.renew());
        assert_eq!(meta.renewed_count, 0);
    }

    #[test]
    fn test_find_expired() {
        let fresh = SharedArtifactMeta::new(
            "fresh".to_string(),
            "did:key:alice".to_string(),
            TtlPreset::Standard,
        );
        let mut expired = SharedArtifactMeta::new(
            "old".to_string(),
            "did:key:bob".to_string(),
            TtlPreset::Ephemeral,
        );
        expired.expires_at = Some(Utc::now() - Duration::hours(1));

        let all = vec![fresh, expired];
        let found = find_expired(&all);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].content_hash, "old");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let meta = SharedArtifactMeta::new(
            "abc".to_string(),
            "did:key:test".to_string(),
            TtlPreset::Durable,
        );
        let json = serde_json::to_string(&meta).unwrap();
        let deser: SharedArtifactMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.content_hash, "abc");
        assert_eq!(deser.ttl_preset, TtlPreset::Durable);
    }
}
