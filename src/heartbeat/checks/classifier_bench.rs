//! ClassifierBenchCheck — re-scores the backend classifiers on their fixtures.
//!
//! Pure computation: the fixtures are embedded in the binary and the classifiers
//! run in memory (no graph, no network). The result goes to the log: one line
//! per classifier, with a warning when it falls under the majority baseline.
//! The same figures are served by `GET /api/chat/routing/report`.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::{info, warn};

use crate::evaluation;
use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};

/// Re-scores the classifiers every six hours.
pub struct ClassifierBenchCheck;

#[async_trait]
impl HeartbeatCheck for ClassifierBenchCheck {
    fn name(&self) -> &str {
        "classifier_bench"
    }

    fn interval(&self) -> Duration {
        Duration::from_secs(6 * 60 * 60)
    }

    async fn run(&self, _ctx: &HeartbeatContext) -> Result<()> {
        for score in evaluation::run_embedded()? {
            if score.above_baseline {
                info!(
                    classifier = %score.classifier,
                    accuracy = score.accuracy,
                    baseline = score.baseline_accuracy,
                    cases = score.cases,
                    "ClassifierBenchCheck: classifier above majority baseline"
                );
            } else {
                warn!(
                    classifier = %score.classifier,
                    accuracy = score.accuracy,
                    baseline = score.baseline_accuracy,
                    cases = score.cases,
                    "ClassifierBenchCheck: classifier UNDER majority baseline"
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_check_is_named_and_runs_every_six_hours() {
        let check = ClassifierBenchCheck;
        assert_eq!(check.name(), "classifier_bench");
        assert_eq!(check.interval(), Duration::from_secs(6 * 3600));
    }
}
