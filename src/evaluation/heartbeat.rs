//! The heartbeat publication of the classifier bench.
//!
//! `ClassifierBenchCheck` re-scores the seven classifiers on their embedded
//! fixtures and logs one line per classifier, warning when one falls under the
//! majority baseline. It is pure computation — no graph, no network — so it is
//! cheap enough for a six-hour heartbeat. The same figures are served by
//! `GET /api/chat/routing/report` (`RoutingReport.classifiers`).
//!
//! It lives here rather than in `src/heartbeat/checks/` because the bench and
//! everything that publishes it are one subject, and because `src/heartbeat/**`
//! is reserved by the `po-autonomic` entry of `docs/diagrams/INDEX.yml`: a new
//! file there would be a source file no *verified* diagram could own without a
//! shared-ownership collision.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::{info, warn};

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};

use super::run_embedded;

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
        for score in run_embedded()? {
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
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    #[test]
    fn the_check_is_named_and_runs_every_six_hours() {
        let check = ClassifierBenchCheck;
        assert_eq!(check.name(), "classifier_bench");
        assert_eq!(check.interval(), Duration::from_secs(6 * 3600));
    }
}
