//! Profile Aggregator — batches behavioral signals and updates profiles.
//!
//! Consumes signals from the `SignalReceiver`, applies EMA updates to
//! profile dimensions, and persists changes to Neo4j every 30 seconds.
//!
//! The aggregator runs as a background tokio task spawned at startup.
//! It processes all accumulated signals in a batch, then sleeps.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};
use uuid::Uuid;

use super::collector::SignalReceiver;
use super::signals::BehaviorSignal;
use super::UserProfile;
use crate::neo4j::traits::GraphStore;

/// EMA smoothing factor — controls adaptation speed.
/// 0.3 = moderate adaptation (balances stability and responsiveness).
const EMA_ALPHA: f64 = 0.3;

/// How often the aggregator persists profiles to Neo4j.
const FLUSH_INTERVAL: Duration = Duration::from_secs(30);

// ============================================================================
// ProfileAggregator
// ============================================================================

/// Aggregates behavioral signals and updates user profiles.
///
/// Runs as a background task. Processes signals in batches every
/// `FLUSH_INTERVAL` and persists updated profiles to Neo4j.
pub struct ProfileAggregator {
    graph: Arc<dyn GraphStore>,
    receiver: SignalReceiver,
    cancel: CancellationToken,
    /// In-memory accumulator: user_id -> pending signal counts per dimension.
    pending: HashMap<String, PendingUpdates>,
}

/// Accumulated signal counts for a single user, pending flush to Neo4j.
#[derive(Debug, Default)]
struct PendingUpdates {
    /// Number of ResponseTruncated signals (→ decrease verbosity)
    truncated_count: u32,
    /// Number of FollowUpQuestion signals (→ increase verbosity)
    follow_up_count: u32,
    /// Number of NoteConfirmed signals (→ expertise signal)
    note_confirmed_count: u32,
    /// Number of NoteRejected signals
    note_rejected_count: u32,
    /// Number of ExpertBehavior signals
    expert_count: u32,
    /// Total interactions
    interaction_count: u32,
    /// Detected languages with counts
    languages: HashMap<String, u32>,
    /// Projects worked on: project_id -> count
    projects: HashMap<Uuid, u32>,
}

impl ProfileAggregator {
    /// Create a new aggregator.
    pub fn new(
        graph: Arc<dyn GraphStore>,
        receiver: SignalReceiver,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            graph,
            receiver,
            cancel,
            pending: HashMap::new(),
        }
    }

    /// Spawn the aggregator as a background task.
    ///
    /// Returns a `JoinHandle` that completes when the aggregator shuts down
    /// (either via cancellation or when all senders are dropped).
    pub fn spawn(
        graph: Arc<dyn GraphStore>,
        receiver: SignalReceiver,
        cancel: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let mut aggregator = Self::new(graph, receiver, cancel);
        tokio::spawn(async move {
            aggregator.run().await;
        })
    }

    /// Main loop: drain signals, accumulate, flush periodically.
    async fn run(&mut self) {
        debug!("[profile_aggregator] Starting background aggregator");

        let mut flush_interval = tokio::time::interval(FLUSH_INTERVAL);
        // Don't fire immediately on first tick
        flush_interval.tick().await;

        loop {
            tokio::select! {
                _ = self.cancel.cancelled() => {
                    debug!("[profile_aggregator] Cancellation received, flushing and shutting down");
                    self.flush().await;
                    break;
                }
                signal = self.receiver.recv() => {
                    match signal {
                        Some(s) => self.accumulate(s),
                        None => {
                            debug!("[profile_aggregator] Channel closed, flushing and shutting down");
                            self.flush().await;
                            break;
                        }
                    }
                }
                _ = flush_interval.tick() => {
                    self.flush().await;
                }
            }
        }

        debug!("[profile_aggregator] Aggregator shut down");
    }

    /// Accumulate a single signal into the pending updates.
    fn accumulate(&mut self, signal: BehaviorSignal) {
        // For now, use a default user_id derived from the session.
        // In production, this would be resolved from the session's user identity.
        let user_id = format!("session:{}", signal.session_id());
        let pending = self.pending.entry(user_id).or_default();

        match signal {
            BehaviorSignal::ResponseTruncated { .. } => {
                pending.truncated_count += 1;
            }
            BehaviorSignal::FollowUpQuestion { .. } => {
                pending.follow_up_count += 1;
            }
            BehaviorSignal::NoteConfirmed { .. } => {
                pending.note_confirmed_count += 1;
            }
            BehaviorSignal::NoteRejected { .. } => {
                pending.note_rejected_count += 1;
            }
            BehaviorSignal::ExpertBehavior { .. } => {
                pending.expert_count += 1;
            }
            BehaviorSignal::Interaction { .. } => {
                pending.interaction_count += 1;
            }
            BehaviorSignal::LanguageDetected { language, .. } => {
                *pending.languages.entry(language).or_insert(0) += 1;
            }
            BehaviorSignal::ProjectSwitch { project_id, .. } => {
                *pending.projects.entry(project_id).or_insert(0) += 1;
            }
        }
    }

    /// Flush all pending updates to Neo4j.
    async fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }

        let pending = std::mem::take(&mut self.pending);
        let count = pending.len();

        for (user_id, updates) in pending {
            if let Err(e) = self.flush_user(&user_id, updates).await {
                warn!(
                    "[profile_aggregator] Failed to flush profile for '{}': {}",
                    user_id, e
                );
            }
        }

        debug!(
            "[profile_aggregator] Flushed {} user profile(s) to Neo4j",
            count
        );
    }

    /// Flush updates for a single user.
    async fn flush_user(&self, user_id: &str, updates: PendingUpdates) -> anyhow::Result<()> {
        // Get or create the profile
        let mut profile = self.graph.create_or_get_user_profile(user_id).await?;

        // Apply EMA updates for each dimension
        apply_ema_updates(&mut profile, &updates);

        // Persist
        self.graph.update_user_profile(&profile).await?;

        // Update WORKS_ON relationships
        for project_id in updates.projects.keys() {
            self.graph.upsert_works_on(user_id, *project_id).await?;
        }

        Ok(())
    }
}

// ============================================================================
// EMA update logic
// ============================================================================

/// Apply EMA updates to a profile based on accumulated signals.
///
/// Each signal type maps to one or more profile dimensions:
/// - ResponseTruncated → verbosity ↓ (signal value = 0.0)
/// - FollowUpQuestion → verbosity ↑ (signal value = 1.0)
/// - NoteConfirmed → expertise ↑ (signal value = 0.8)
/// - ExpertBehavior → expertise ↑ (signal value = 1.0)
/// - LanguageDetected → language (majority vote)
fn apply_ema_updates(profile: &mut UserProfile, updates: &PendingUpdates) {
    // Verbosity: truncation pushes down, follow-ups push up
    for _ in 0..updates.truncated_count {
        profile.verbosity = UserProfile::ema_update(profile.verbosity, 0.0, EMA_ALPHA);
    }
    for _ in 0..updates.follow_up_count {
        profile.verbosity = UserProfile::ema_update(profile.verbosity, 1.0, EMA_ALPHA);
    }

    // Expertise: confirmed notes and expert behavior push up
    for _ in 0..updates.note_confirmed_count {
        profile.expertise_level = UserProfile::ema_update(profile.expertise_level, 0.8, EMA_ALPHA);
    }
    for _ in 0..updates.expert_count {
        profile.expertise_level = UserProfile::ema_update(profile.expertise_level, 1.0, EMA_ALPHA);
    }

    // Language: majority vote from detected languages
    if !updates.languages.is_empty() {
        if let Some((lang, _)) = updates.languages.iter().max_by_key(|(_, count)| *count) {
            profile.language = lang.clone();
        }
    }

    // Interaction count: always increment
    profile.interaction_count += updates.interaction_count as u64;

    // Update timestamp
    profile.updated_at = chrono::Utc::now();
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::collector::create_signal_channel;

    #[test]
    fn test_apply_ema_truncation_decreases_verbosity() {
        let mut profile = UserProfile::new("test");
        assert_eq!(profile.verbosity, 0.5);

        let updates = PendingUpdates {
            truncated_count: 3,
            ..Default::default()
        };

        apply_ema_updates(&mut profile, &updates);

        // 3 truncation signals → verbosity should decrease
        assert!(
            profile.verbosity < 0.5,
            "Verbosity should decrease from truncation, got {}",
            profile.verbosity
        );
    }

    #[test]
    fn test_apply_ema_followup_increases_verbosity() {
        let mut profile = UserProfile::new("test");
        assert_eq!(profile.verbosity, 0.5);

        let updates = PendingUpdates {
            follow_up_count: 3,
            ..Default::default()
        };

        apply_ema_updates(&mut profile, &updates);

        assert!(
            profile.verbosity > 0.5,
            "Verbosity should increase from follow-ups, got {}",
            profile.verbosity
        );
    }

    #[test]
    fn test_apply_ema_expert_behavior() {
        let mut profile = UserProfile::new("test");
        assert_eq!(profile.expertise_level, 0.5);

        let updates = PendingUpdates {
            expert_count: 2,
            note_confirmed_count: 1,
            ..Default::default()
        };

        apply_ema_updates(&mut profile, &updates);

        assert!(
            profile.expertise_level > 0.5,
            "Expertise should increase, got {}",
            profile.expertise_level
        );
    }

    #[test]
    fn test_apply_ema_language_majority() {
        let mut profile = UserProfile::new("test");
        assert_eq!(profile.language, "en");

        let mut languages = HashMap::new();
        languages.insert("fr".to_string(), 3);
        languages.insert("en".to_string(), 1);

        let updates = PendingUpdates {
            languages,
            ..Default::default()
        };

        apply_ema_updates(&mut profile, &updates);

        assert_eq!(profile.language, "fr");
    }

    #[test]
    fn test_apply_ema_interaction_count() {
        let mut profile = UserProfile::new("test");
        assert_eq!(profile.interaction_count, 0);

        let updates = PendingUpdates {
            interaction_count: 5,
            ..Default::default()
        };

        apply_ema_updates(&mut profile, &updates);

        assert_eq!(profile.interaction_count, 5);
    }

    #[tokio::test]
    async fn test_aggregator_accumulate_and_flush() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let (collector, receiver) = create_signal_channel();
        let cancel = CancellationToken::new();

        let sid = Uuid::new_v4();

        // Send signals
        collector.send(BehaviorSignal::Interaction { session_id: sid });
        collector.send(BehaviorSignal::FollowUpQuestion { session_id: sid });
        collector.send(BehaviorSignal::ResponseTruncated { session_id: sid });

        // Drop collector to close channel
        drop(collector);

        // Run aggregator — it will process signals and shut down when channel closes
        let mut aggregator = ProfileAggregator::new(mock.clone(), receiver, cancel);
        aggregator.run().await;

        // Verify profile was created and updated
        let user_id = format!("session:{}", sid);
        let profile = mock.get_user_profile(&user_id).await.unwrap();
        assert!(profile.is_some(), "Profile should have been created");

        let profile = profile.unwrap();
        assert_eq!(profile.interaction_count, 1);
        // Follow-up pushes verbosity up, truncation pushes down — net effect depends on order
        // Both happened once, so verbosity should be close to 0.5
    }

    #[tokio::test]
    async fn test_aggregator_works_on_tracking() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let (collector, receiver) = create_signal_channel();
        let cancel = CancellationToken::new();

        let sid = Uuid::new_v4();
        let pid = Uuid::new_v4();

        // Create a project first (needed for WORKS_ON)
        use crate::neo4j::models::ProjectNode;
        let project = ProjectNode {
            id: pid,
            name: "test".to_string(),
            slug: "test".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            scaffolding_override: None,
            sharing_policy: None,
        };
        mock.create_project(&project).await.unwrap();

        collector.send(BehaviorSignal::ProjectSwitch {
            session_id: sid,
            project_id: pid,
        });
        drop(collector);

        let mut aggregator = ProfileAggregator::new(mock.clone(), receiver, cancel);
        aggregator.run().await;

        // Verify WORKS_ON was created
        let user_id = format!("session:{}", sid);
        let works = mock.get_works_on(&user_id).await.unwrap();
        assert_eq!(works.len(), 1);
        assert_eq!(works[0].project_id, pid);
        assert_eq!(works[0].frequency, 1);
    }
}
