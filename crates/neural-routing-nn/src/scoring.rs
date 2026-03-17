//! Multi-factor scoring for trajectory ranking.
//!
//! score = cosine_similarity * exp(-age_days / decay) * (0.5 + 0.5 * normalized_reward)

use chrono::{DateTime, Utc};

/// Compute the composite score for a candidate trajectory.
///
/// Factors:
/// - `similarity`: cosine similarity (0.0 - 1.0)
/// - `created_at`: when the trajectory was recorded (penalizes old ones)
/// - `total_reward`: RBCR reward of the trajectory
/// - `max_reward`: maximum reward across all candidates (for normalization)
/// - `max_age_days`: decay constant in days (default: 30)
pub fn compute_score(
    similarity: f64,
    created_at: &DateTime<Utc>,
    total_reward: f64,
    max_reward: f64,
    max_age_days: u32,
) -> f64 {
    let age_days = (Utc::now() - *created_at).num_hours() as f64 / 24.0;
    let recency = (-age_days / max_age_days as f64).exp();

    let normalized_reward = if max_reward > 1e-10 {
        (total_reward / max_reward).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let reward_factor = 0.5 + 0.5 * normalized_reward;

    similarity * recency * reward_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_recent_high_reward_scores_highest() {
        let now = Utc::now();
        let recent_good = compute_score(0.9, &now, 0.95, 1.0, 30);
        let old_good = compute_score(0.9, &(now - Duration::days(60)), 0.95, 1.0, 30);
        assert!(recent_good > old_good, "Recent should beat old: {} vs {}", recent_good, old_good);
    }

    #[test]
    fn test_high_similarity_beats_low() {
        let now = Utc::now();
        let high_sim = compute_score(0.95, &now, 0.5, 1.0, 30);
        let low_sim = compute_score(0.5, &now, 0.5, 1.0, 30);
        assert!(high_sim > low_sim);
    }

    #[test]
    fn test_high_reward_beats_low() {
        let now = Utc::now();
        let high_rew = compute_score(0.8, &now, 0.9, 1.0, 30);
        let low_rew = compute_score(0.8, &now, 0.1, 1.0, 30);
        assert!(high_rew > low_rew);
    }

    #[test]
    fn test_score_bounds() {
        let now = Utc::now();
        let score = compute_score(1.0, &now, 1.0, 1.0, 30);
        assert!(score > 0.0 && score <= 1.0, "Score should be in (0, 1]: {}", score);
    }
}
