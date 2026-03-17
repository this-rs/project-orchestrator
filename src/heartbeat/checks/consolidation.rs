//! ConsolidationCheck — promotes ephemeral notes to consolidated periodically.
//!
//! Runs `consolidate_memory` on GraphStore every 24 hours for all projects.
//! This promotes eligible ephemeral notes (old enough + high activation)
//! to consolidated status and archives dead notes.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};

/// Consolidate ephemeral notes into long-term memory (every 24 hours).
pub struct ConsolidationCheck;

#[async_trait]
impl HeartbeatCheck for ConsolidationCheck {
    fn name(&self) -> &str {
        "memory_consolidation"
    }

    fn interval(&self) -> Duration {
        Duration::from_secs(24 * 60 * 60) // 24 hours
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
        assert_eq!(check.interval(), Duration::from_secs(24 * 3600));
    }
}
