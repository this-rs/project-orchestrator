//! SynapseDecayCheck — decays synapse weights and updates energy scores.
//!
//! Runs `decay_synapses` to weaken unused neural links,
//! then `update_energy_scores` to decay note energy based on
//! time since last activation (exponential half-life model).
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::debug;

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};
use crate::neurons::synaptic_shadow::{self, SynapticLayerFlags};

/// Traces judged on their closure per pass.
const CLOSURES_PER_PASS: usize = 200;

/// The shadow part of the pass. Failures are logged, never propagated: the
/// shadow layer must not stop the legacy decay from being reported as done.
pub(crate) async fn shadow_pass(
    ctx: &HeartbeatContext,
    flags: &SynapticLayerFlags,
    now: chrono::DateTime<chrono::Utc>,
) -> (usize, (usize, usize)) {
    if !flags.any_enabled() {
        return (0, (0, 0));
    }
    let decayed = synaptic_shadow::decay(ctx.graph.as_ref(), flags, now)
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(target: "synapse_shadow", error = %e, "shadow decay failed");
            0
        });
    let settled = synaptic_shadow::settle_closures(
        ctx.graph.as_ref(),
        flags,
        now,
        CLOSURES_PER_PASS,
    )
    .await
    .unwrap_or_else(|e| {
        tracing::warn!(target: "synapse_shadow", error = %e, "shadow closure judgement failed");
        (0, 0)
    });
    debug!(
        "SynapseDecayCheck: shadow layer decayed {} edges, closures quiet={} restarted={}",
        decayed, settled.0, settled.1
    );
    (decayed, settled)
}

/// Default synapse decay amount per cycle.
const DEFAULT_DECAY_AMOUNT: f64 = 0.02;
/// Default prune threshold — synapses below this weight are deleted.
const DEFAULT_PRUNE_THRESHOLD: f64 = 0.05;
/// Default energy half-life in days.
const DEFAULT_HALF_LIFE_DAYS: f64 = crate::notes::ENERGY_HALF_LIFE_DAYS;

/// Decay synapses and update energy scores (every 6 hours).
pub struct SynapseDecayCheck;

#[async_trait]
impl HeartbeatCheck for SynapseDecayCheck {
    fn name(&self) -> &str {
        "synapse_decay"
    }

    fn interval(&self) -> Duration {
        Duration::from_secs(6 * 60 * 60) // 6 hours
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        // 1. Decay synapses
        let (decayed, pruned) = ctx
            .graph
            .decay_synapses(DEFAULT_DECAY_AMOUNT, DEFAULT_PRUNE_THRESHOLD)
            .await?;
        debug!(
            "SynapseDecayCheck: decayed {} synapses, pruned {}",
            decayed, pruned
        );

        // 2. Update energy scores
        let updated = ctx
            .graph
            .update_energy_scores(DEFAULT_HALF_LIFE_DAYS)
            .await?;
        debug!("SynapseDecayCheck: updated energy for {} notes", updated);

        // 3. The synaptic layer in shadow mode (T8): proportional decay of the
        //    shadow weights by elapsed time, then the closure judgement of the
        //    traces whose attempt closed long enough ago. Never the legacy weight.
        shadow_pass(ctx, SynapticLayerFlags::global(), chrono::Utc::now()).await;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    #[test]
    fn test_synapse_decay_check_name() {
        let check = SynapseDecayCheck;
        assert_eq!(check.name(), "synapse_decay");
    }

    #[test]
    fn test_synapse_decay_check_interval() {
        let check = SynapseDecayCheck;
        assert_eq!(check.interval(), Duration::from_secs(6 * 3600));
    }

    #[tokio::test]
    async fn the_shadow_pass_runs_only_when_the_layer_is_on_for_a_project() {
        let graph = std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let project = uuid::Uuid::new_v4();
        let mut ids = Vec::new();
        for _ in 0..2 {
            let n = crate::notes::Note::new(
                Some(project),
                crate::notes::NoteType::Guideline,
                "n".into(),
                "api".into(),
            );
            graph.create_note(&n).await.unwrap();
            ids.push(n.id);
        }
        graph
            .note_synapses
            .write()
            .await
            .insert(ids[0], vec![(ids[1], 0.7)]);
        let ctx = HeartbeatContext {
            graph: graph.clone(),
            search: None,
            emitter: None,
        };
        let now = chrono::Utc::now();
        let off = SynapticLayerFlags::parse(None, None);
        assert_eq!(shadow_pass(&ctx, &off, now).await, (0, (0, 0)));
        assert!(graph.synapse_shadow.read().await.is_empty());
        let on = SynapticLayerFlags::parse(Some("off"), Some(&format!("{project}=shadow")));
        assert_eq!(shadow_pass(&ctx, &on, now).await, (1, (0, 0)));
        // Legacy weight untouched by the shadow pass.
        assert_eq!(
            graph.note_synapses.read().await[&ids[0]],
            vec![(ids[1], 0.7)]
        );
    }

    #[test]
    fn test_default_constants() {
        assert!((DEFAULT_DECAY_AMOUNT - 0.02).abs() < f64::EPSILON);
        assert!((DEFAULT_PRUNE_THRESHOLD - 0.05).abs() < f64::EPSILON);
        assert!((DEFAULT_HALF_LIFE_DAYS - 90.0).abs() < f64::EPSILON);
    }
}
