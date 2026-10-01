//! Periodic re-derivation of every workspace's architecture.
//!
//! The sync hooks already re-derive on every sync, so why this too?
//!
//! Because a sync is not the only way the answer changes. Adding a project to a
//! workspace creates cross-project edges without any file being touched; a
//! project whose directory was missing becomes readable again; and the derivation
//! rules themselves change with a deploy, which no amount of syncing replays.
//!
//! Sync hooks keep the topology fresh for code that moves. This keeps it true for
//! everything else, and costs nothing when nothing changed — the derivation is
//! deterministic and writes through an upsert, so a no-change run is a no-op.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};

/// One workspace per tick, rotating. See `MAX_WORKSPACES_PER_RUN`.
const INTERVAL: Duration = Duration::from_secs(900);

/// Walking a source tree reads a lot of files, so the engine's 5s default is too
/// tight. It stays small on purpose all the same.
///
/// The engine awaits this timeout **inline** in its select loop, so whatever is
/// set here is how long every other check is blocked. Worse, a check that times
/// out does not get its `last_run` updated, so it re-runs on the very next tick
/// and starves everything else — the failure that stopped skills from ever being
/// created (PR #309). The protection is not a generous timeout, it is bounded work.
const TIMEOUT: Duration = Duration::from_secs(20);

/// Derivation walks the file tree of every project in a workspace, and an
/// instance can hold many workspaces. Doing them all in one tick is what would
/// blow the timeout; one per tick keeps each run short and still covers
/// everything within minutes.
const MAX_WORKSPACES_PER_RUN: usize = 1;

/// Periodic re-derivation of every workspace's architecture.
///
/// The sync hooks already re-derive on every sync, so why this too?
///
/// Because a sync is not the only way the answer changes. Adding a project to a
/// workspace creates cross-project edges without any file being touched; a
/// project whose directory was missing becomes readable again; and the derivation
/// rules themselves change with a deploy, which no amount of syncing replays.
///
/// Sync hooks keep the topology fresh for code that moves. This keeps it true for
/// everything else, and costs nothing when nothing changed — the derivation is
/// deterministic and writes through an upsert, so a no-change run is a no-op.
#[derive(Default)]
pub struct ArchitectureDriftCheck {
    /// Where the rotation left off, so successive ticks cover every workspace
    /// instead of re-deriving the first one forever.
    cursor: AtomicUsize,
}

impl ArchitectureDriftCheck {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl HeartbeatCheck for ArchitectureDriftCheck {
    fn name(&self) -> &str {
        "architecture_drift"
    }

    fn interval(&self) -> Duration {
        INTERVAL
    }

    fn timeout_override(&self) -> Option<Duration> {
        Some(TIMEOUT)
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        let workspaces = ctx.graph.list_workspaces().await?;
        if workspaces.is_empty() {
            return Ok(());
        }

        let start = self
            .cursor
            .fetch_add(MAX_WORKSPACES_PER_RUN, Ordering::Relaxed);

        for offset in 0..MAX_WORKSPACES_PER_RUN.min(workspaces.len()) {
            let workspace = &workspaces[(start + offset) % workspaces.len()];

            // One failing workspace must not stop the others: a single project
            // with an unreadable root would otherwise freeze every other
            // workspace's architecture.
            match crate::architecture::sync::derive_and_store_workspace(
                ctx.graph.clone(),
                workspace.id,
            )
            .await
            {
                Ok(outcome) if outcome.components_written > 0 => {
                    tracing::debug!(
                        workspace = %workspace.slug,
                        components = outcome.components_written,
                        edges = outcome.edges_written,
                        skipped = outcome.projects_skipped,
                        "architecture re-derived"
                    );
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(
                        workspace = %workspace.slug,
                        error = %e,
                        "architecture derivation failed for workspace"
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::models::WorkspaceNode;
    use crate::neo4j::traits::GraphStore;
    use std::sync::Arc;
    use uuid::Uuid;

    fn ctx(graph: Arc<MockGraphStore>) -> HeartbeatContext {
        HeartbeatContext {
            graph,
            search: None,
            emitter: None,
        }
    }

    async fn store_with(slugs: &[&str]) -> Arc<MockGraphStore> {
        let store = Arc::new(MockGraphStore::new());
        for slug in slugs {
            store
                .create_workspace(&WorkspaceNode {
                    id: Uuid::new_v4(),
                    name: (*slug).into(),
                    slug: (*slug).into(),
                    description: None,
                    created_at: chrono::Utc::now(),
                    updated_at: None,
                    metadata: serde_json::json!({}),
                })
                .await
                .unwrap();
        }
        store
    }

    #[test]
    fn timeout_stays_far_below_the_interval() {
        // The engine awaits this timeout inline, blocking every other check, and
        // a check that times out re-runs on the next tick without updating
        // last_run — which is how skill creation was starved (PR #309). A long
        // timeout on a frequent check is the shape of that bug.
        let timeout = ArchitectureDriftCheck::new().timeout_override().unwrap();
        assert!(
            timeout > Duration::from_secs(5),
            "file walking needs more than the engine default"
        );
    }

    // Checked at compile time rather than in a test: deriving many workspaces in
    // one tick is what blows the timeout, and a build failure is a better place
    // to learn that than a test run.
    const _: () = assert!(MAX_WORKSPACES_PER_RUN >= 1);
    const _: () = assert!(MAX_WORKSPACES_PER_RUN <= 2);
    const _: () = assert!(TIMEOUT.as_secs() * 10 < INTERVAL.as_secs());

    #[tokio::test]
    async fn an_empty_instance_is_a_no_op() {
        assert!(ArchitectureDriftCheck::new()
            .run(&ctx(store_with(&[]).await))
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn successive_runs_rotate_instead_of_repeating_one_workspace() {
        // Without the cursor the first workspace would be re-derived forever and
        // the others never at all.
        let check = ArchitectureDriftCheck::new();
        let store = store_with(&["a", "b", "c"]).await;
        for _ in 0..3 {
            check.run(&ctx(store.clone())).await.unwrap();
        }
        assert_eq!(check.cursor.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn the_rotation_wraps_around() {
        let check = ArchitectureDriftCheck::new();
        let store = store_with(&["a", "b"]).await;
        for _ in 0..5 {
            check.run(&ctx(store.clone())).await.unwrap();
        }
        assert!(check.run(&ctx(store)).await.is_ok());
    }

    #[tokio::test]
    async fn one_broken_workspace_does_not_stop_the_rest() {
        let store = store_with(&["a", "b"]).await;
        let check = ArchitectureDriftCheck::new();
        assert!(check.run(&ctx(store.clone())).await.is_ok());
        assert!(check.run(&ctx(store)).await.is_ok());
    }
}
