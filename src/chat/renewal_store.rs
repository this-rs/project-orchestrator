//! Persistence of the cognitive renewal state in the existing graph KV store.
//!
//! The checkpoint and the idempotency ledger of one logical session are stored
//! as ONE document under [`crate::neo4j::traits::GraphStore::put_llm_setting`],
//! the same graph document store that already carries provider settings.
//! Nothing here is a new store, and the checkpoint is a *projection*
//! (references to notes, decisions and evidence), not the memory itself:
//! persisting it does not duplicate Obrain.
//!
//! Three properties, each deliberate:
//! - **atomic** — checkpoint, epoch and ledger are one serialised record, so a
//!   single write is all-or-nothing (no three-write window);
//! - **immutable** — the key is content-addressed (`v{version}:{sha256[:16]}`),
//!   so a rewrite with different content lands on a different key: two
//!   concurrent same-version writes are *detectable*, never silently
//!   overwritten;
//! - **fail-closed** — a missing state is `None`; a record that fails to
//!   deserialise, or two records claiming the same highest version, is an
//!   error, never an empty or initial value.

use anyhow::{anyhow, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::neo4j::traits::GraphStore;

use super::renewal::{CognitiveCheckpoint, IdempotencyLedger, LogicalSessionId};

/// Namespace of the renewal documents inside the graph KV.
const SCOPE: &str = "renewal";

/// The renewal state of one logical session, published as one atomic record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenewalState {
    /// The working memory selected at the renewal point (carries its own
    /// version and epoch).
    pub checkpoint: CognitiveCheckpoint,
    /// The idempotency ledger, including its `uncertain` set, as of the same
    /// point in time.
    pub ledger: IdempotencyLedger,
}

fn version_key_prefix(id: &str) -> String {
    format!("{id}:v")
}

fn parse_version(key: &str, id: &str) -> Option<u32> {
    key.strip_prefix(&version_key_prefix(id))?
        .split(':')
        .next()?
        .parse()
        .ok()
}

fn content_hash(json: &str) -> String {
    let digest = Sha256::digest(json.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn decode<T: DeserializeOwned>(raw: Option<String>) -> Result<Option<T>> {
    match raw {
        None => Ok(None),
        Some(s) => serde_json::from_str(&s)
            .map(Some)
            .map_err(|e| anyhow!("renewal state corrupt: {e}")),
    }
}

/// A thin, keyed view of the renewal state of ONE logical session.
pub struct RenewalStore<'a> {
    graph: &'a dyn GraphStore,
}

impl<'a> RenewalStore<'a> {
    pub fn new(graph: &'a dyn GraphStore) -> Self {
        Self { graph }
    }

    /// Publish the renewal state as one immutable, content-addressed record.
    ///
    /// Idempotent for identical content: the same version and the same bytes
    /// land on the same key, so writing twice is not a conflict.
    pub async fn put_state(&self, id: &LogicalSessionId, state: &RenewalState) -> Result<()> {
        let json = serde_json::to_string(state)?;
        let digest = content_hash(&json);
        let key = format!("{}:v{}:{}", id.0, state.checkpoint.version, &digest[..16]);
        self.graph.put_llm_setting(SCOPE, &key, &json).await?;
        Ok(())
    }

    /// The most recent state, or `None` when nothing was ever published.
    ///
    /// Fails closed: a record that no longer deserialises, or two records that
    /// claim the same highest version (a concurrent double-write), are both an
    /// error — never an empty or guessed state.
    pub async fn load_state(&self, id: &LogicalSessionId) -> Result<Option<RenewalState>> {
        let rows = self
            .graph
            .list_llm_settings(SCOPE, &version_key_prefix(&id.0))
            .await?;
        if rows.is_empty() {
            return Ok(None);
        }
        let max_version = rows
            .iter()
            .filter_map(|(k, _)| parse_version(k, &id.0))
            .max()
            .expect("non-empty rows must parse to a version");
        let at_max: Vec<(String, String)> = rows
            .into_iter()
            .filter(|(k, _)| parse_version(k, &id.0) == Some(max_version))
            .collect();
        if at_max.len() > 1 {
            return Err(anyhow!(
                "renewal conflict: {} records at version {} for {}",
                at_max.len(),
                max_version,
                id.0
            ));
        }
        let (_, json) = at_max.into_iter().next().expect("exactly one record");
        decode(Some(json))
    }

    /// Every version published for this logical session, ascending (audit).
    pub async fn versions(&self, id: &LogicalSessionId) -> Result<Vec<u32>> {
        let rows = self
            .graph
            .list_llm_settings(SCOPE, &version_key_prefix(&id.0))
            .await?;
        let mut versions: Vec<u32> = rows
            .iter()
            .filter_map(|(k, _)| parse_version(k, &id.0))
            .collect();
        versions.sort_unstable();
        versions.dedup();
        Ok(versions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::renewal::CognitiveEpoch;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::ChatStore;

    fn id() -> LogicalSessionId {
        LogicalSessionId("logical-1".into())
    }

    fn checkpoint(version: u32, epoch: u32, objective: &str) -> CognitiveCheckpoint {
        CognitiveCheckpoint {
            version,
            epoch: CognitiveEpoch(epoch),
            objective: objective.into(),
            constraints: vec!["never rewrite the audit transcript".into()],
            decisions: vec![],
            open_hypotheses: vec![],
            evidence: vec!["note:abc".into()],
            pending_operations: vec![],
        }
    }

    fn state(version: u32, epoch: u32, objective: &str) -> RenewalState {
        let mut ledger = IdempotencyLedger::default();
        ledger.record("commit:a");
        ledger.mark_uncertain("deploy:prod:1");
        RenewalState {
            checkpoint: checkpoint(version, epoch, objective),
            ledger,
        }
    }

    #[tokio::test]
    async fn a_state_round_trips_and_keeps_the_uncertain_set() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        let st = state(2, 1, "reduce context cost");
        store.put_state(&id(), &st).await.unwrap();
        let back = store.load_state(&id()).await.unwrap().unwrap();
        assert_eq!(back, st);
        assert!(back.ledger.is_uncertain("deploy:prod:1"));
        assert_eq!(back.checkpoint.epoch, CognitiveEpoch(1));
    }

    #[tokio::test]
    async fn a_missing_state_is_none_not_an_error() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        assert!(store.load_state(&id()).await.unwrap().is_none());
        assert!(store.versions(&id()).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_corrupted_record_fails_closed() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        store.put_state(&id(), &state(1, 1, "o")).await.unwrap();
        let rows = graph
            .list_llm_settings(SCOPE, &version_key_prefix(&id().0))
            .await
            .unwrap();
        let (key, _) = rows.into_iter().next().unwrap();
        graph
            .put_llm_setting(SCOPE, &key, "{not json")
            .await
            .unwrap();
        assert!(store.load_state(&id()).await.is_err());
    }

    #[tokio::test]
    async fn two_same_version_writes_are_detected_not_overwritten() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        store
            .put_state(&id(), &state(1, 1, "objective A"))
            .await
            .unwrap();
        store
            .put_state(&id(), &state(1, 1, "objective B"))
            .await
            .unwrap();
        let err = store.load_state(&id()).await.unwrap_err();
        assert!(err.to_string().contains("conflict"), "{}", err);
    }

    #[tokio::test]
    async fn putting_the_same_state_twice_is_idempotent() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        let st = state(1, 1, "o");
        store.put_state(&id(), &st).await.unwrap();
        store.put_state(&id(), &st).await.unwrap();
        assert_eq!(store.versions(&id()).await.unwrap(), vec![1]);
        assert_eq!(store.load_state(&id()).await.unwrap().unwrap(), st);
    }

    #[tokio::test]
    async fn a_state_survives_a_restart() {
        let graph = MockGraphStore::new();
        let st = state(3, 2, "objective");
        RenewalStore::new(&graph)
            .put_state(&id(), &st)
            .await
            .unwrap();
        let back = RenewalStore::new(&graph)
            .load_state(&id())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(back, st);
    }

    #[tokio::test]
    async fn the_history_is_append_only_and_ascending() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        store.put_state(&id(), &state(1, 1, "a")).await.unwrap();
        store.put_state(&id(), &state(2, 2, "b")).await.unwrap();
        store.put_state(&id(), &state(3, 3, "c")).await.unwrap();
        assert_eq!(store.versions(&id()).await.unwrap(), vec![1, 2, 3]);
        assert_eq!(
            store
                .load_state(&id())
                .await
                .unwrap()
                .unwrap()
                .checkpoint
                .version,
            3
        );
    }
}
