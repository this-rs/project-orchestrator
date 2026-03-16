//! Adaptive User Profile — learns preferences through observation.
//!
//! The profile tracks aggregated behavioral metrics (never message content or code).
//! Dimensions are updated via Exponential Moving Average (EMA) from implicit signals.

pub mod aggregator;
pub mod collector;
pub mod signals;
pub mod wiring;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// UserProfile
// ============================================================================

/// Adaptive user profile learned from implicit behavioral signals.
///
/// Each dimension is a `f64` in `[0.0, 1.0]` updated via EMA (alpha=0.3).
/// The profile NEVER stores sensitive personal data — only aggregated metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// Unique profile ID
    pub id: Uuid,
    /// The user (session/identity) this profile belongs to
    pub user_id: String,

    // ---- Preference dimensions (0.0 = low, 1.0 = high) ----
    /// Preferred verbosity level.
    /// 0.0 = terse/concise, 1.0 = very detailed explanations.
    pub verbosity: f64,

    /// Preferred commit style.
    /// 0.0 = conventional commits (short), 1.0 = verbose multi-line commits.
    pub commit_style: f64,

    /// Primary interaction language preference.
    /// Stored as ISO 639-1 code (e.g., "en", "fr").
    pub language: String,

    /// Estimated expertise level.
    /// 0.0 = beginner, 0.5 = intermediate, 1.0 = expert.
    pub expertise_level: f64,

    /// Total number of interactions observed.
    pub interaction_count: u64,

    /// When this profile was created.
    pub created_at: DateTime<Utc>,

    /// When this profile was last updated.
    pub updated_at: DateTime<Utc>,
}

impl UserProfile {
    /// Create a new profile with neutral defaults.
    pub fn new(user_id: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            user_id: user_id.into(),
            verbosity: 0.5,
            commit_style: 0.5,
            language: "en".to_string(),
            expertise_level: 0.5,
            interaction_count: 0,
            created_at: now,
            updated_at: now,
        }
    }

    /// Apply an EMA update to a dimension.
    ///
    /// `alpha` controls how fast the profile adapts (0.3 = moderate).
    /// `new_value` is the observed signal value in `[0.0, 1.0]`.
    pub fn ema_update(current: f64, new_value: f64, alpha: f64) -> f64 {
        let result = alpha * new_value + (1.0 - alpha) * current;
        result.clamp(0.0, 1.0)
    }

    /// Format the profile as a concise markdown string for prompt injection.
    ///
    /// Target: < 50 tokens to minimize context window usage.
    pub fn to_prompt_markdown(&self) -> String {
        let verbosity_label = match self.verbosity {
            v if v < 0.3 => "concise",
            v if v < 0.7 => "balanced",
            _ => "detailed",
        };
        let expertise_label = match self.expertise_level {
            e if e < 0.3 => "beginner",
            e if e < 0.7 => "intermediate",
            _ => "expert",
        };
        format!(
            "User: {} | lang={} | verbosity={} | expertise={} | interactions={}",
            self.user_id, self.language, verbosity_label, expertise_label, self.interaction_count,
        )
    }
}

impl Default for UserProfile {
    fn default() -> Self {
        Self::new("default")
    }
}

// ============================================================================
// WorksOnRelation — tracks per-project engagement
// ============================================================================

/// Tracks how frequently a user works on a specific project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorksOnRelation {
    /// User ID
    pub user_id: String,
    /// Project UUID
    pub project_id: Uuid,
    /// Number of sessions/interactions with this project
    pub frequency: u64,
    /// When the user was last active on this project
    pub last_active: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_profile_defaults() {
        let profile = UserProfile::new("test-user");
        assert_eq!(profile.user_id, "test-user");
        assert_eq!(profile.verbosity, 0.5);
        assert_eq!(profile.commit_style, 0.5);
        assert_eq!(profile.language, "en");
        assert_eq!(profile.expertise_level, 0.5);
        assert_eq!(profile.interaction_count, 0);
    }

    #[test]
    fn test_ema_update() {
        // Starting at 0.5, observing 1.0 with alpha=0.3
        let result = UserProfile::ema_update(0.5, 1.0, 0.3);
        assert!((result - 0.65).abs() < 1e-10);

        // Repeated observations converge
        let mut val = 0.5;
        for _ in 0..20 {
            val = UserProfile::ema_update(val, 1.0, 0.3);
        }
        assert!(val > 0.99, "Should converge to 1.0, got {}", val);
    }

    #[test]
    fn test_ema_update_clamped() {
        // EMA with out-of-range inputs: result is clamped to [0.0, 1.0]
        // ema(0.5, 2.0, 0.3) = 0.7*0.5 + 0.3*2.0 = 0.95 (within range, no clamp needed)
        assert!((UserProfile::ema_update(0.5, 2.0, 0.3) - 0.95).abs() < 1e-10);
        // ema(0.5, -1.0, 0.3) = 0.7*0.5 + 0.3*(-1.0) = 0.05 (within range)
        assert!((UserProfile::ema_update(0.5, -1.0, 0.3) - 0.05).abs() < 1e-10);
        // True clamp cases:
        assert_eq!(UserProfile::ema_update(0.9, 5.0, 0.5), 1.0); // 0.45 + 2.5 = 2.95 → 1.0
        assert_eq!(UserProfile::ema_update(0.1, -3.0, 0.5), 0.0); // 0.05 + -1.5 = -1.45 → 0.0
    }

    #[test]
    fn test_to_prompt_markdown() {
        let profile = UserProfile {
            verbosity: 0.1,
            expertise_level: 0.9,
            language: "fr".to_string(),
            interaction_count: 42,
            ..UserProfile::new("alice")
        };
        let md = profile.to_prompt_markdown();
        assert!(md.contains("concise"));
        assert!(md.contains("expert"));
        assert!(md.contains("fr"));
        assert!(md.contains("42"));
    }
}
