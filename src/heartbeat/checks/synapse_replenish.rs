//! SynapseReplenishCheck — rebuilds decayed-away synapses so the neural graph self-heals.
//!
//! `SynapseDecayCheck` (every 6h) and `deep_maintenance` decay + prune synapses
//! unconditionally, but synapse CREATION is only **event-driven** (on note
//! create/update — see `notes::manager::spawn_auto_connect_synapses`). Existing
//! notes are never re-synapsed as decay erodes their links. Over time decay wins:
//! the SYNAPSE graph reaches ZERO edges, and `detect_skills_pipeline` silently
//! no-ops because it needs >= `min_notes_for_detection` (15) notes that already
//! participate in synapses. Symptom: "skill detection produces no effect" while
//! `get_intelligence_summary` shows `active_synapses: 0` despite thousands of notes.
//!
//! This check counterbalances decay by periodically backfilling synapses from the
//! notes' STORED embeddings (`backfill_synapses`). It needs no embedding provider —
//! `backfill_synapses` reads stored embeddings and does vector search via the
//! GraphStore — so it runs with just the heartbeat's `graph` + `search`. The
//! backfill self-gates (early-returns when every note already has synapses), so on
//! a healthy graph this check costs almost nothing; on a depleted one it rebuilds.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};
use crate::notes::NoteManager;

/// Notes processed per backfill batch.
const REPLENISH_BATCH_SIZE: usize = 100;
/// Max neighbours linked per note when rebuilding.
const REPLENISH_MAX_NEIGHBORS: usize = 10;
/// Per-run timeout. A fully-decayed large graph can take minutes to rebuild, so
/// this must exceed the engine's 5s default (otherwise the rebuild is cancelled
/// every tick and `last_run` never advances — the same starvation #309 fixed).
const REPLENISH_TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// Rebuild synapses that decay has eroded away (every 12 hours).
pub struct SynapseReplenishCheck;

#[async_trait]
impl HeartbeatCheck for SynapseReplenishCheck {
    fn name(&self) -> &str {
        "synapse_replenish"
    }

    fn interval(&self) -> Duration {
        // Twice the 6h decay cadence: frequent enough to keep ahead of pruning
        // (a 0.75-weight synapse takes days to decay below the 0.05 prune floor),
        // infrequent enough to keep the heavy rebuild rare.
        Duration::from_secs(12 * 60 * 60)
    }

    fn timeout_override(&self) -> Option<Duration> {
        Some(REPLENISH_TIMEOUT)
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        let Some(search) = ctx.search.clone() else {
            warn!("SynapseReplenishCheck: no search store available, skipping replenish");
            return Ok(());
        };

        // `backfill_synapses` self-gates: it lists only notes that have an embedding
        // but no synapses, and returns early when there are none. So calling it
        // unconditionally is cheap on a healthy graph and rebuilds a depleted one.
        // `min_similarity = 0.0` auto-calibrates the threshold from the existing
        // weight distribution (falls back to 0.75 when the graph is empty).
        let nm = NoteManager::new(ctx.graph.clone(), search);
        let progress = nm
            .backfill_synapses(REPLENISH_BATCH_SIZE, 0.0, REPLENISH_MAX_NEIGHBORS, None)
            .await?;

        if progress.synapses_created > 0 || progress.energy_initialized > 0 {
            info!(
                "SynapseReplenishCheck: rebuilt {} synapses across {} notes (energy initialized on {})",
                progress.synapses_created, progress.processed, progress.energy_initialized
            );
        } else {
            debug!("SynapseReplenishCheck: synapse graph healthy, nothing to replenish");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synapse_replenish_check_name() {
        assert_eq!(SynapseReplenishCheck.name(), "synapse_replenish");
    }

    #[test]
    fn test_synapse_replenish_check_interval() {
        assert_eq!(
            SynapseReplenishCheck.interval(),
            Duration::from_secs(12 * 3600)
        );
    }

    #[test]
    fn test_synapse_replenish_timeout_override_exceeds_default() {
        // Must exceed the engine's 5s default, else the rebuild is cancelled every
        // tick and never persists (the decay-without-replenishment death spiral).
        let t = SynapseReplenishCheck.timeout_override();
        assert_eq!(t, Some(Duration::from_secs(600)));
        assert!(t.unwrap() > Duration::from_secs(5));
    }
}
