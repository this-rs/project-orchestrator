//! The delegation limits an operator changes WHILE the server runs (PLAFOND-RT).
//!
//! Until PLAFOND-RT the cap on a parent's live children was an environment variable
//! (`CHAT_MAX_LIVE_CHILDREN`) read once at startup: changing it meant restarting the
//! backend and with it every session. It is now one document,
//! `LlmSetting {scope: "global", key: "delegation_limits"}`, read at startup into an atomic
//! counter of the [`ChatManager`] and replaced by `PUT /api/chat/delegation-limits`.
//!
//! Rules that keep the setting safe:
//! - Only the root account writes it (a server-wide resource decision, like the global
//!   consent), never an agent: a model must not raise its own ceiling.
//! - It is bounded to `MIN_LIVE_CHILDREN..=MAX_LIVE_CHILDREN_CEILING`; nothing outside
//!   is stored, and an unreadable or out-of-range stored value is ignored (the default holds).
//! - Lowering it never closes a live child: it only refuses the NEXT opening of a parent
//!   that already has that many. Raising it takes effect on the very next delegation.
//! - The compilation governor still bounds the real parallelism of builds; this number
//!   bounds sessions, not CPU.

use std::sync::atomic::Ordering;

use serde::{Deserialize, Serialize};

use super::envelope::MAX_LIVE_CHILDREN;
use super::manager::ChatManager;
use super::provider::settings::GLOBAL;
use crate::neo4j::GraphStore;
use crate::neo4j::traits::*;

/// Key of the stored document, under the [`GLOBAL`] scope.
pub const DELEGATION_LIMITS_KEY: &str = "delegation_limits";

/// The smallest cap: a parent must be able to keep one child.
pub const MIN_LIVE_CHILDREN: usize = 1;

/// The largest cap an operator can set. Above this a runaway fan-out costs more than it
/// can ever return on one machine.
pub const MAX_LIVE_CHILDREN_CEILING: usize = 32;

/// The stored document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelegationLimits {
    /// How many children of one parent may be alive at once.
    pub max_live_children: usize,
}

/// `value` when it is an allowed cap, else `Err(value)`.
pub fn settle(value: usize) -> Result<usize, usize> {
    if (MIN_LIVE_CHILDREN..=MAX_LIVE_CHILDREN_CEILING).contains(&value) {
        Ok(value)
    } else {
        Err(value)
    }
}

/// The stored cap; `None` when nothing is stored or the document is unusable (unreadable
/// JSON, a value out of bounds): the default then applies, and the reason is logged.
pub async fn load(graph: &dyn GraphStore) -> anyhow::Result<Option<usize>> {
    let Some(raw) = graph.get_llm_setting(GLOBAL, DELEGATION_LIMITS_KEY).await? else {
        return Ok(None);
    };
    match serde_json::from_str::<DelegationLimits>(&raw) {
        Ok(doc) => match settle(doc.max_live_children) {
            Ok(cap) => Ok(Some(cap)),
            Err(bad) => {
                tracing::warn!(
                    value = bad,
                    "stored delegation_limits out of bounds ignored: the default cap holds"
                );
                Ok(None)
            }
        },
        Err(error) => {
            tracing::warn!(%error, "unreadable delegation_limits document ignored: the default cap holds");
            Ok(None)
        }
    }
}

/// Replaces the stored cap. The caller has already [`settle`]d it.
pub async fn store(graph: &dyn GraphStore, max_live_children: usize) -> anyhow::Result<()> {
    let document = serde_json::to_string(&DelegationLimits { max_live_children })?;
    graph
        .put_llm_setting(GLOBAL, DELEGATION_LIMITS_KEY, &document)
        .await
}

/// Removes the stored cap: the default applies again.
pub async fn clear(graph: &dyn GraphStore) -> anyhow::Result<()> {
    graph
        .delete_llm_setting(GLOBAL, DELEGATION_LIMITS_KEY)
        .await?;
    Ok(())
}

impl ChatManager {
    /// The cap in force NOW on the children one parent may keep alive. Read at every
    /// delegation, so a change by [`Self::set_max_live_children`] applies to the next one.
    pub fn max_live_children(&self) -> usize {
        self.live_children_cap.load(Ordering::Relaxed)
    }

    /// Reads the stored cap into the running manager (startup). A failing read keeps the
    /// default and says so; it never blocks the server.
    pub async fn load_delegation_limits(&self) {
        match load(self.graph.as_ref()).await {
            Ok(Some(cap)) => self.live_children_cap.store(cap, Ordering::Relaxed),
            Ok(None) => {}
            Err(error) => tracing::warn!(
                %error,
                "could not read delegation_limits: the default cap holds"
            ),
        }
    }

    /// Sets the cap for the next delegations and stores it for the next start. Stored
    /// FIRST: a failed write leaves the running value as it was, so memory and graph never
    /// disagree. Returns the cap, or `Err(value)` when it is out of bounds.
    pub async fn set_max_live_children(
        &self,
        value: usize,
    ) -> Result<anyhow::Result<usize>, usize> {
        let cap = settle(value)?;
        Ok(match store(self.graph.as_ref(), cap).await {
            Ok(()) => {
                self.live_children_cap.store(cap, Ordering::Relaxed);
                Ok(cap)
            }
            Err(error) => Err(error),
        })
    }

    /// Removes the stored cap and goes back to the default; returns it.
    pub async fn reset_max_live_children(&self) -> anyhow::Result<usize> {
        clear(self.graph.as_ref()).await?;
        self.live_children_cap
            .store(MAX_LIVE_CHILDREN, Ordering::Relaxed);
        Ok(MAX_LIVE_CHILDREN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::traits::*;

    #[test]
    fn only_a_cap_between_one_and_the_ceiling_is_allowed() {
        assert_eq!(settle(0), Err(0));
        assert_eq!(settle(1), Ok(1));
        assert_eq!(
            settle(MAX_LIVE_CHILDREN_CEILING),
            Ok(MAX_LIVE_CHILDREN_CEILING)
        );
        assert_eq!(
            settle(MAX_LIVE_CHILDREN_CEILING + 1),
            Err(MAX_LIVE_CHILDREN_CEILING + 1)
        );
    }

    #[tokio::test]
    async fn the_document_round_trips_and_clear_removes_it() {
        let graph = MockGraphStore::new();
        assert_eq!(load(&graph).await.unwrap(), None);
        store(&graph, 9).await.unwrap();
        assert_eq!(load(&graph).await.unwrap(), Some(9));
        clear(&graph).await.unwrap();
        assert_eq!(load(&graph).await.unwrap(), None);
    }

    #[tokio::test]
    async fn an_unusable_stored_document_leaves_the_default() {
        let graph = MockGraphStore::new();
        graph
            .put_llm_setting(GLOBAL, DELEGATION_LIMITS_KEY, "not json")
            .await
            .unwrap();
        assert_eq!(load(&graph).await.unwrap(), None);
        graph
            .put_llm_setting(
                GLOBAL,
                DELEGATION_LIMITS_KEY,
                r#"{"max_live_children": 500}"#,
            )
            .await
            .unwrap();
        assert_eq!(load(&graph).await.unwrap(), None);
    }
}
