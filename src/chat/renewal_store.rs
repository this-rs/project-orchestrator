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
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::neo4j::traits::GraphStore;

use super::renewal::{
    reconstruct, safe_point, CognitiveCheckpoint, CognitiveEpoch, IdempotencyLedger,
    LogicalSessionId, Reconstructed,
};

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

fn parse_key(key: &str, id: &str) -> Option<(u32, String)> {
    let rest = key.strip_prefix(&version_key_prefix(id))?;
    let (version, hash) = rest.split_once(':')?;
    Some((version.parse().ok()?, hash.to_string()))
}

fn content_hash(json: &str) -> String {
    let digest = Sha256::digest(json.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Length of the content fingerprint embedded in the key, in hex characters
/// (half a sha256, 8 bytes). Fixed so that a shorter or empty fingerprint can
/// never pass a comparison.
const FINGERPRINT_LEN: usize = 16;

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
        let key = format!(
            "{}:v{}:{}",
            id.0,
            state.checkpoint.version,
            &digest[..FINGERPRINT_LEN]
        );
        self.graph.put_llm_setting(SCOPE, &key, &json).await?;
        Ok(())
    }

    /// The most recent state, or `None` when nothing was ever published.
    ///
    /// Fails closed on every incoherence, never with a `panic` or an empty
    /// replacement:
    /// - a key that does not parse (`{id}:v{version}:{hash}`) is an error;
    /// - two records at the SAME version — anywhere in the history, not only at
    ///   the highest — are an error: a newer snapshot does not resolve an older
    ///   unresolved double-write;
    /// - a key fingerprint that is not exactly [`FINGERPRINT_LEN`] hex digits is
    ///   an error (an empty or shortened fingerprint never passes);
    /// - a record whose content hash does not equal its key fingerprint is an
    ///   error (rewritten in place);
    /// - a record that fails to deserialise, or whose checkpoint version
    ///   disagrees with its key, is an error.
    pub async fn load_state(&self, id: &LogicalSessionId) -> Result<Option<RenewalState>> {
        let rows = self
            .graph
            .list_llm_settings(SCOPE, &version_key_prefix(&id.0))
            .await?;
        if rows.is_empty() {
            return Ok(None);
        }
        let mut parsed: Vec<(String, u32, String, String)> = Vec::with_capacity(rows.len());
        for (key, json) in rows {
            match parse_key(&key, &id.0) {
                Some((version, hash)) => parsed.push((key, version, hash, json)),
                None => return Err(anyhow!("renewal key malformed for {}: {key}", id.0)),
            }
        }
        let mut by_version: std::collections::BTreeMap<u32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (i, (_, v, _, _)) in parsed.iter().enumerate() {
            by_version.entry(*v).or_default().push(i);
        }
        for (v, idxs) in &by_version {
            if idxs.len() > 1 {
                return Err(anyhow!(
                    "renewal conflict: {} records at version {v} for {}",
                    idxs.len(),
                    id.0
                ));
            }
        }
        let (&max_version, _) = by_version
            .last_key_value()
            .ok_or_else(|| anyhow!("renewal state for {} has no version", id.0))?;
        let idx = by_version[&max_version][0];
        let (key, _, key_hash, json) = &parsed[idx];
        if key_hash.len() != FINGERPRINT_LEN || !key_hash.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(anyhow!(
                "renewal key fingerprint malformed for {key}: {key_hash}"
            ));
        }
        let actual = content_hash(json);
        if &actual[..FINGERPRINT_LEN] != key_hash.as_str() {
            return Err(anyhow!(
                "renewal record tampered for {key}: content hash mismatch"
            ));
        }
        let state: RenewalState = serde_json::from_str(json)
            .map_err(|e| anyhow!("renewal state corrupt at {key}: {e}"))?;
        if state.checkpoint.version != max_version {
            return Err(anyhow!(
                "renewal key/checkpoint version mismatch for {key}: key v{max_version}, checkpoint v{}",
                state.checkpoint.version
            ));
        }
        Ok(Some(state))
    }

    /// Every version published for this logical session, ascending (audit).
    pub async fn versions(&self, id: &LogicalSessionId) -> Result<Vec<u32>> {
        let rows = self
            .graph
            .list_llm_settings(SCOPE, &version_key_prefix(&id.0))
            .await?;
        let mut versions: Vec<u32> = Vec::with_capacity(rows.len());
        for (key, _) in rows {
            match parse_key(&key, &id.0) {
                Some((version, _)) => versions.push(version),
                None => return Err(anyhow!("renewal key malformed for {}: {key}", id.0)),
            }
        }
        versions.sort_unstable();
        versions.dedup();
        Ok(versions)
    }
}

/// The outcome of one completed renewal transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenewalOutcome {
    pub from_epoch: CognitiveEpoch,
    pub to_epoch: CognitiveEpoch,
    pub checkpoint_version: u32,
    pub reconstructed: Reconstructed,
}

/// Orchestrate ONE renewal transition: refuse unless the session is at a safe
/// point, persist the new checkpoint+ledger BEFORE any irreversible transition,
/// and reconstruct the working memory the new technical session is seeded with.
///
/// This is the transition logic only. Closing the old technical session and
/// opening the new one is the runtime's job (the manager), after this returns
/// `Ok`: nothing is closed here, so a failure leaves the session untouched.
pub async fn renew(
    store: &RenewalStore<'_>,
    id: &LogicalSessionId,
    current: &CognitiveCheckpoint,
    ledger: &IdempotencyLedger,
    budget_tokens: usize,
    cost_of: impl Fn(&str) -> usize,
    tool_in_flight: bool,
    uncertain_write: bool,
) -> Result<RenewalOutcome> {
    safe_point(tool_in_flight, uncertain_write).map_err(|e| anyhow!("renewal refused: {e}"))?;
    let from_epoch = current.epoch;
    let next = current.next_version(CognitiveEpoch(from_epoch.0 + 1));
    store
        .put_state(
            id,
            &RenewalState {
                checkpoint: next.clone(),
                ledger: ledger.clone(),
            },
        )
        .await?;
    let reconstructed = reconstruct(&next, budget_tokens, cost_of)
        .map_err(|e| anyhow!("reconstruction refused: {e}"))?;
    Ok(RenewalOutcome {
        from_epoch,
        to_epoch: next.epoch,
        checkpoint_version: next.version,
        reconstructed,
    })
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
    async fn two_successive_same_version_writes_are_detected() {
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
    async fn a_state_reloads_on_a_fresh_adapter_over_the_same_graph() {
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

    #[tokio::test]
    async fn a_record_rewritten_with_different_valid_json_fails_closed() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        store
            .put_state(&id(), &state(1, 1, "original"))
            .await
            .unwrap();
        let rows = graph
            .list_llm_settings(SCOPE, &version_key_prefix(&id().0))
            .await
            .unwrap();
        let (key, _) = rows.into_iter().next().unwrap();
        let tampered = serde_json::to_string(&state(1, 1, "tampered")).unwrap();
        graph.put_llm_setting(SCOPE, &key, &tampered).await.unwrap();
        let err = store.load_state(&id()).await.unwrap_err();
        assert!(err.to_string().contains("hash"), "{}", err);
    }

    #[tokio::test]
    async fn an_incomplete_ledger_fails_closed() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        // current format without the `uncertain` field -> must NOT deserialise
        // to an empty ledger.
        let json = serde_json::json!({
            "checkpoint": checkpoint(1, 1, "o"),
            "ledger": { "done": ["commit:a"] }
        })
        .to_string();
        let key = format!("{}:v1:{}", id().0, &content_hash(&json)[..16]);
        graph.put_llm_setting(SCOPE, &key, &json).await.unwrap();
        let err = store.load_state(&id()).await.unwrap_err();
        assert!(err.to_string().contains("corrupt"), "{}", err);
    }

    #[tokio::test]
    async fn a_key_checkpoint_version_mismatch_fails_closed() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        let st = state(1, 1, "o"); // checkpoint.version == 1
        let json = serde_json::to_string(&st).unwrap();
        let key = format!("{}:v2:{}", id().0, &content_hash(&json)[..16]);
        graph.put_llm_setting(SCOPE, &key, &json).await.unwrap();
        let err = store.load_state(&id()).await.unwrap_err();
        assert!(err.to_string().contains("mismatch"), "{}", err);
    }

    #[tokio::test]
    async fn a_historical_conflict_is_not_masked_by_a_higher_version() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        store.put_state(&id(), &state(1, 1, "A")).await.unwrap();
        store.put_state(&id(), &state(1, 1, "B")).await.unwrap();
        store.put_state(&id(), &state(2, 2, "C")).await.unwrap();
        let err = store.load_state(&id()).await.unwrap_err();
        assert!(err.to_string().contains("conflict"), "{}", err);
    }

    #[tokio::test]
    async fn a_short_fingerprint_does_not_pass() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        let st = state(1, 1, "o");
        let json = serde_json::to_string(&st).unwrap();
        let key = format!("{}:v1:abcd", id().0); // 4 hex digits, not FINGERPRINT_LEN
        graph.put_llm_setting(SCOPE, &key, &json).await.unwrap();
        let err = store.load_state(&id()).await.unwrap_err();
        assert!(err.to_string().contains("fingerprint"), "{}", err);
    }

    #[tokio::test]
    async fn a_renewal_persists_then_reconstructs() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        let current = checkpoint(3, 2, "reduce context cost");
        let out = renew(
            &store,
            &id(),
            &current,
            &IdempotencyLedger::default(),
            10_000,
            |s| s.len() / 4,
            false,
            false,
        )
        .await
        .unwrap();
        assert_eq!(out.from_epoch, CognitiveEpoch(2));
        assert_eq!(out.to_epoch, CognitiveEpoch(3));
        assert_eq!(out.checkpoint_version, 4);
        assert!(out
            .reconstructed
            .kept
            .iter()
            .any(|k| k == &current.objective));
        let loaded = store.load_state(&id()).await.unwrap().unwrap();
        assert_eq!(loaded.checkpoint.version, 4);
        assert_eq!(loaded.checkpoint.epoch, CognitiveEpoch(3));
    }

    #[tokio::test]
    async fn a_renewal_refuses_on_a_tool_in_flight() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        let err = renew(
            &store,
            &id(),
            &checkpoint(1, 1, "o"),
            &IdempotencyLedger::default(),
            100,
            |s| s.len() / 4,
            true,
            false,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("tool"), "{}", err);
        // nothing was persisted before the refusal
        assert!(store.load_state(&id()).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn a_renewal_refuses_on_an_uncertain_write() {
        let graph = MockGraphStore::new();
        let store = RenewalStore::new(&graph);
        let err = renew(
            &store,
            &id(),
            &checkpoint(1, 1, "o"),
            &IdempotencyLedger::default(),
            100,
            |s| s.len() / 4,
            false,
            true,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("unknown outcome"), "{}", err);
        assert!(store.load_state(&id()).await.unwrap().is_none());
    }
}
