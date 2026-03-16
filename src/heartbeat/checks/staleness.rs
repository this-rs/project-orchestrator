//! StalenessCheck — updates note staleness scores periodically.
//!
//! Calls `GraphStore::update_staleness_scores()` to recompute
//! staleness for all active notes based on file modification timestamps.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::debug;

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};

/// Recompute note staleness scores (every 1 hour).
pub struct StalenessCheck;

#[async_trait]
impl HeartbeatCheck for StalenessCheck {
    fn name(&self) -> &str {
        "staleness"
    }

    fn interval(&self) -> Duration {
        Duration::from_secs(60 * 60) // 1 hour
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        let updated = ctx.graph.update_staleness_scores().await?;
        debug!("StalenessCheck: updated staleness for {} notes", updated);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_staleness_check_name() {
        let check = StalenessCheck;
        assert_eq!(check.name(), "staleness");
    }

    #[test]
    fn test_staleness_check_interval() {
        let check = StalenessCheck;
        assert_eq!(check.interval(), Duration::from_secs(3600));
    }
}
