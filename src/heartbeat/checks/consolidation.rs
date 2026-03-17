//! ConsolidationCheck — promotes ephemeral notes to consolidated periodically.
//!
//! Runs `consolidate_memory` on GraphStore every 2 hours for all projects.
//! This promotes eligible ephemeral notes (old enough + high activation)
//! to consolidated status and archives dead notes.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};

/// Consolidate ephemeral notes into long-term memory (every 2 hours).
///
/// **Responsibility boundary**: this check is the SOLE owner of `consolidate_memory`.
/// `HomeostasisCheck` does NOT call `consolidate_memory` — it focuses on synapse
/// health and note density only. This avoids duplicate Neo4j calls per cycle.
///
/// This is idempotent: if no notes are eligible for promotion or archival,
/// the operation is a no-op. Safe to run frequently.
pub struct ConsolidationCheck;

#[async_trait]
impl HeartbeatCheck for ConsolidationCheck {
    fn name(&self) -> &str {
        "memory_consolidation"
    }

    fn interval(&self) -> Duration {
        Duration::from_secs(2 * 60 * 60) // 2 hours
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        info!("ConsolidationCheck: running memory consolidation");

        match ctx.graph.consolidate_memory().await {
            Ok((promoted, archived)) => {
                debug!(
                    promoted,
                    archived, "ConsolidationCheck: consolidation completed"
                );

                if promoted > 0 || archived > 0 {
                    info!(
                        "ConsolidationCheck: promoted {} notes, archived {} dead notes",
                        promoted, archived
                    );
                }
            }
            Err(e) => {
                warn!("ConsolidationCheck: consolidation failed: {}", e);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consolidation_check_name() {
        let check = ConsolidationCheck;
        assert_eq!(check.name(), "memory_consolidation");
    }

    #[test]
    fn test_consolidation_check_interval() {
        let check = ConsolidationCheck;
        assert_eq!(check.interval(), Duration::from_secs(2 * 3600));
    }
}
