//! Persistence of the cognitive renewal state in the existing graph KV store.
//!
//! The checkpoint, the epoch and the idempotency ledger of one logical session
//! are stored under [`crate::neo4j::traits::GraphStore::put_llm_setting`] — the
//! same graph document store that already carries provider settings. Nothing
//! here is a new store, and the checkpoint is a *projection* (references to
//! notes, decisions and evidence), not the memory itself: persisting it does
//! not duplicate Obrain.
//!
//! A checkpoint is never edited in place. Each version is written under its own
//! immutable key (`checkpoint:v{version}`) and a "current" pointer is advanced,
//! leaving an append-only history for audit. The graph KV is a single-write
//! primitive, so this store does *not* claim compare-and-swap atomicity: a
//! concurrent renewal is *detectable* (two records with the same version), not
//! prevented. The caller owns the transaction.

use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};

use crate::neo4j::traits::GraphStore;

use super::renewal::{CognitiveCheckpoint, CognitiveEpoch, IdempotencyLedger, LogicalSessionId};

/// Namespace of the renewal documents inside the graph KV.
const SCOPE: &str = "renewal";

fn key(id: &str, suffix: &str) -> String {
    format!("{id}:{suffix}")
}

fn decode<T: DeserializeOwned>(raw: Option<String>) -> Option<T> {
    raw.and_then(|s| serde_json::from_str(&s).ok())
}

/// A thin, keyed view of the renewal state of ONE logical session.
pub struct RenewalStore<'a> {
    graph: &'a dyn GraphStore,
}

impl<'a> RenewalStore<'a> {
    pub fn new(graph: &'a dyn GraphStore) -> Self {
        Self { graph }
    }

    /// Persist a checkpoint without editing it in place: writes the immutable
    /// versioned record, advances the "current" pointer and records the epoch.
    pub async fn put_checkpoint(
        &self,
        id: &LogicalSessionId,
        cp: &CognitiveCheckpoint,
    ) -> Result<()> {
        let json = serde_json::to_string(cp)?;
        self.graph
            .put_llm_setting(
                SCOPE,
                &key(&id.0, &format!("checkpoint:v{}", cp.version)),
                &json,
            )
            .await?;
        self.graph
            .put_llm_setting(SCOPE, &key(&id.0, "checkpoint"), &json)
            .await?;
        self.graph
            .put_llm_setting(SCOPE, &key(&id.0, "epoch"), &cp.epoch.0.to_string())
            .await?;
        Ok(())
    }

    /// The most recently persisted checkpoint, if any.
    pub async fn load_checkpoint(
        &self,
        id: &LogicalSessionId,
    ) -> Result<Option<CognitiveCheckpoint>> {
        let raw = self
            .graph
            .get_llm_setting(SCOPE, &key(&id.0, "checkpoint"))
            .await?;
        Ok(decode(raw))
    }

    /// The epoch recorded with the last checkpoint, or `0` if none.
    pub async fn load_epoch(&self, id: &LogicalSessionId) -> Result<CognitiveEpoch> {
        let raw = self
            .graph
            .get_llm_setting(SCOPE, &key(&id.0, "epoch"))
            .await?;
        Ok(raw
            .and_then(|s| s.parse::<u32>().ok())
            .map(CognitiveEpoch)
            .unwrap_or_default())
    }

    /// Every version recorded for this logical session, ascending (audit).
    pub async fn checkpoint_versions(&self, id: &LogicalSessionId) -> Result<Vec<u32>> {
        let prefix = format!("{}:checkpoint:v", id.0);
        let rows = self.graph.list_llm_settings(SCOPE, &prefix).await?;
        let mut versions: Vec<u32> = rows
            .into_iter()
            .filter_map(|(k, _)| k.strip_prefix(&prefix).and_then(|v| v.parse::<u32>().ok()))
            .collect();
        versions.sort_unstable();
        Ok(versions)
    }

    /// Persist the idempotency ledger, including its `uncertain` set.
    pub async fn put_ledger(
        &self,
        id: &LogicalSessionId,
        ledger: &IdempotencyLedger,
    ) -> Result<()> {
        let json = serde_json::to_string(ledger)?;
        self.graph
            .put_llm_setting(SCOPE, &key(&id.0, "ledger"), &json)
            .await?;
        Ok(())
    }

    /// The persisted ledger, or an empty one if none was written yet.
    pub async fn load_ledger(&self, id: &LogicalSessionId) -> Result<IdempotencyLedger> {
        let raw = self
            .graph
            .get_llm_setting(SCOPE, &key(&id.0, "ledger"))
            .await?;
        Ok(decode(raw).unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;

    fn id() -> LogicalSessionId {
        LogicalSessionId("logical-1".into())
    }

    fn checkpoint(version: u32, epoch: u32) -> CognitiveCheckpoint {
        CognitiveCheckpoint {
            version,
            epoch: CognitiveEpoch(epoch),
            objective: "reduce context cost".into(),
            constraints: vec!["never rewrite the audit transcript".into()],
            decisions: vec![],
            open_hypotheses: vec![],
            evidence: vec!["note:abc".into()],
            pending_operations: vec![],
        }
    }

    #[tokio::test]
    async fn a_checkpoint_round_trips_through_the_store() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        let cp = checkpoint(2, 1);
        store.put_checkpoint(&id(), &cp).await.unwrap();
        let back = store.load_checkpoint(&id()).await.unwrap().unwrap();
        assert_eq!(back, cp);
        assert_eq!(store.load_epoch(&id()).await.unwrap(), CognitiveEpoch(1));
    }

    #[tokio::test]
    async fn a_missing_checkpoint_is_none_not_a_panic() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        assert!(store.load_checkpoint(&id()).await.unwrap().is_none());
        assert_eq!(store.load_epoch(&id()).await.unwrap(), CognitiveEpoch(0));
        assert!(store.load_ledger(&id()).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn the_store_keeps_an_append_only_version_history() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        store
            .put_checkpoint(&id(), &checkpoint(1, 1))
            .await
            .unwrap();
        store
            .put_checkpoint(&id(), &checkpoint(2, 2))
            .await
            .unwrap();
        store
            .put_checkpoint(&id(), &checkpoint(3, 3))
            .await
            .unwrap();
        assert_eq!(
            store.checkpoint_versions(&id()).await.unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(
            store.load_checkpoint(&id()).await.unwrap().unwrap().version,
            3
        );
    }

    #[tokio::test]
    async fn the_ledger_persists_the_uncertain_set_not_only_done() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        let mut ledger = IdempotencyLedger::default();
        ledger.record("commit:a");
        ledger.mark_uncertain("deploy:prod:1");
        store.put_ledger(&id(), &ledger).await.unwrap();

        let back = store.load_ledger(&id()).await.unwrap();
        assert!(back.already_done("commit:a"));
        assert!(back.is_uncertain("deploy:prod:1"));
        assert_eq!(back.uncertain_keys(), vec!["deploy:prod:1"]);
    }
}
