//! Biomimicry Enrichment Stage for the Chat Pipeline.
//!
//! Runs BEFORE knowledge injection to check the project's bio-cognitive health:
//!
//! 1. **Stagnation detection** — if >=3 stagnation signals are active,
//!    triggers daily maintenance in the background (non-blocking).
//! 2. **Homeostasis check** — computes the pain_score and injects it as metadata
//!    so downstream stages and the agent are aware of the knowledge graph's health.
//!
//! Both checks run in parallel with a 300ms timeout. Failures are gracefully skipped.
//!
//! Controlled by `ENRICHMENT_BIOMIMICRY=true` (disabled by default).

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::time::timeout;
use tracing::{debug, info, warn};

use crate::chat::enrichment::{
    EnrichmentConfig, EnrichmentContext, EnrichmentInput, EnrichmentStage,
};
use crate::neo4j::traits::GraphStore;
use crate::skills::maintenance::SkillMaintenanceConfig;

/// Enrichment stage that monitors the project's bio-cognitive health.
pub struct BiomimicryStage {
    graph: Arc<dyn GraphStore>,
}

impl BiomimicryStage {
    /// Create a new biomimicry stage.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }
}

#[async_trait::async_trait]
impl EnrichmentStage for BiomimicryStage {
    async fn execute(&self, input: &EnrichmentInput, ctx: &mut EnrichmentContext) -> Result<()> {
        let project_id = match input.project_id {
            Some(id) => id,
            None => {
                // Try to resolve from slug
                if let Some(slug) = &input.project_slug {
                    match self.graph.get_project_by_slug(slug).await {
                        Ok(Some(p)) => p.id,
                        _ => return Ok(()),
                    }
                } else {
                    return Ok(());
                }
            }
        };

        let stage_timeout = Duration::from_millis(300);

        // Run stagnation detection and homeostasis check in parallel
        let graph_stag = self.graph.clone();
        let graph_home = self.graph.clone();

        let stagnation_future = {
            let pid = project_id;
            async move { graph_stag.detect_global_stagnation(pid).await }
        };

        let homeostasis_future = {
            let pid = project_id;
            async move { graph_home.compute_homeostasis(pid, None).await }
        };

        let (stagnation_result, homeostasis_result) = tokio::join!(
            timeout(stage_timeout, stagnation_future),
            timeout(stage_timeout, homeostasis_future),
        );

        let mut sections = Vec::new();

        // Process stagnation result
        if let Ok(Ok(report)) = stagnation_result {
            debug!(
                "[biomimicry] Stagnation check: {}/4 signals triggered, stagnating={}",
                report.signals_triggered, report.is_stagnating,
            );

            if report.is_stagnating && report.signals_triggered >= 3 {
                // Trigger daily maintenance in the background (non-blocking)
                let graph_maint = self.graph.clone();
                let maint_pid = project_id;
                tokio::spawn(async move {
                    info!(
                        "[biomimicry] Auto-triggering daily maintenance for project {} (stagnation detected)",
                        maint_pid
                    );
                    let config = SkillMaintenanceConfig::default();
                    if let Err(e) = crate::skills::maintenance::run_daily_maintenance(
                        graph_maint.as_ref(),
                        maint_pid,
                        &config,
                    )
                    .await
                    {
                        warn!("[biomimicry] Auto-maintenance failed: {}", e);
                    }
                });

                let recommendations = report.recommendations.join(", ");
                sections.push(format!(
                    "**Stagnation detected** ({}/4 signals). Auto-maintenance triggered. Recommendations: {}",
                    report.signals_triggered, recommendations,
                ));
            }
        } else {
            debug!("[biomimicry] Stagnation check skipped (timeout or error)");
        }

        // Process homeostasis result
        if let Ok(Ok(report)) = homeostasis_result {
            debug!(
                "[biomimicry] Homeostasis: pain_score={:.2}, scar_load={:.2}, freshness={:.2}",
                report.pain_score,
                report
                    .ratios
                    .iter()
                    .find(|r| r.name == "scar_load")
                    .map(|r| r.value)
                    .unwrap_or(0.0),
                report
                    .ratios
                    .iter()
                    .find(|r| r.name == "note_freshness")
                    .map(|r| r.value)
                    .unwrap_or(0.0),
            );

            if report.pain_score > 0.7 {
                sections.push(format!(
                    "**Knowledge graph pain: {:.0}%** — consider running maintenance or adding documentation.",
                    report.pain_score * 100.0,
                ));
            }

            // Always inject pain_score as a concise metadata line
            sections.push(format!("Health: pain={:.0}%", report.pain_score * 100.0));
        } else {
            debug!("[biomimicry] Homeostasis check skipped (timeout or error)");
        }

        if !sections.is_empty() {
            ctx.add_section("Bio-Cognitive Health", sections.join("\n"), self.name());
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "biomimicry"
    }

    fn is_enabled(&self, config: &EnrichmentConfig) -> bool {
        config.biomimicry
    }
}
