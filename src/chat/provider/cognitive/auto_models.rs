//! The models a PERSON lets their Auto choose among (AUTO-L).
//!
//! One document per person, `LlmSetting {scope: "person:<sub>", key: "auto_models"}`, never a
//! global or project setting: one person's list must not steer anybody else's conversations
//! (review of #722, point 1). Written only by that person's own session
//! (`PUT /api/chat/auto-models`, refused to an agent, MCP, vault or service token).
//!
//! No document, an unreadable one, or an empty list: Auto chooses among every model available,
//! exactly as before. A list restricts the candidates of an Auto conversation (mode `full`, no
//! models ticked) at its opening, on each turn, and for a provider move through the relay. A
//! conversation's own ticked pool always wins over it.
//!
//! A listed model that is gone (instance deleted, consent withdrawn, unhealthy) simply leaves
//! the effective pool. When nothing listed is usable any more, the restriction is dropped for
//! that decision and its reason says so (`narrowed: auto_models_unavailable`): Auto never
//! blocks a chat because of this setting.

use serde::{Deserialize, Serialize};

use super::candidates::{self, ModelFacts, Slot};
use super::signature::TaskSignature;
use crate::chat::types::RoutingPoolEntry;
use crate::neo4j::GraphStore;

/// Key of the person's document.
pub const AUTO_MODELS_KEY: &str = "auto_models";

/// Reason code of a decision whose candidates were narrowed to the person's list.
pub const NARROWED_BY_PERSON: &str = "auto_models";
/// Reason code of a decision taken among every available model because nothing on the
/// person's list could be used: the visible notice of the fallback.
pub const PERSON_LIST_UNAVAILABLE: &str = "auto_models_unavailable";

/// The scope a person's own settings are stored under.
pub fn person_scope(sub: &str) -> String {
    format!("person:{sub}")
}

/// The stored document.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoModels {
    /// The pairs Auto may choose among; empty = every available model.
    #[serde(default)]
    pub models: Vec<RoutingPoolEntry>,
}

/// The person's list, `None` when Auto may use every available model (no document, an
/// unreadable one, an empty list).
pub async fn load(
    graph: &dyn GraphStore,
    person: &str,
) -> anyhow::Result<Option<Vec<RoutingPoolEntry>>> {
    let Some(raw) = graph
        .get_llm_setting(&person_scope(person), AUTO_MODELS_KEY)
        .await?
    else {
        return Ok(None);
    };
    match serde_json::from_str::<AutoModels>(&raw) {
        Ok(doc) if !doc.models.is_empty() => Ok(Some(doc.models)),
        Ok(_) => Ok(None),
        Err(error) => {
            tracing::warn!(%error, "unreadable auto_models document ignored: every model applies");
            Ok(None)
        }
    }
}

/// Replaces the person's list; `models` empty removes it (every available model again).
pub async fn store(
    graph: &dyn GraphStore,
    person: &str,
    models: &[RoutingPoolEntry],
) -> anyhow::Result<()> {
    let scope = person_scope(person);
    if models.is_empty() {
        graph.delete_llm_setting(&scope, AUTO_MODELS_KEY).await?;
        return Ok(());
    }
    let document = serde_json::to_string(&AutoModels {
        models: models.to_vec(),
    })?;
    graph
        .put_llm_setting(&scope, AUTO_MODELS_KEY, &document)
        .await
}

fn listed(list: &[RoutingPoolEntry], facts: &ModelFacts) -> bool {
    list.iter()
        .any(|e| e.provider == facts.provider_id && e.model == facts.model)
}

/// Narrows `pool` to the person's `list` for a decision of `signature`, and returns the reason
/// code to state on it. When no listed model passes the hard constraints (gone, not consented,
/// unhealthy, unable to serve the turn), `pool` is left whole and the code is
/// [`PERSON_LIST_UNAVAILABLE`].
pub fn restrict(
    pool: &mut Vec<ModelFacts>,
    list: &[RoutingPoolEntry],
    signature: &TaskSignature,
) -> &'static str {
    let kept: Vec<ModelFacts> = pool.iter().filter(|f| listed(list, f)).cloned().collect();
    let usable = candidates::apply(Slot::Automatic, signature, &kept)
        .is_some_and(|filtered| !filtered.eligible.is_empty());
    if usable {
        *pool = kept;
        NARROWED_BY_PERSON
    } else {
        PERSON_LIST_UNAVAILABLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::signature::ContextHints;

    fn facts(provider: &str, model: &str, allowed: bool, healthy: bool) -> ModelFacts {
        ModelFacts {
            provider_id: provider.into(),
            model: model.into(),
            supports_tools: true,
            supports_images: false,
            context_window: Some(200_000),
            window_unknown: None,
            price: None,
            cost_basis: nexus_claude::agent::CostBasis::Free,
            healthy: Some(healthy),
            allowed_for_project: allowed,
            sandboxed: false,
        }
    }

    fn entry(provider: &str, model: &str) -> RoutingPoolEntry {
        RoutingPoolEntry {
            provider: provider.into(),
            model: model.into(),
        }
    }

    fn signature() -> TaskSignature {
        TaskSignature::from_chat_request("hello", false, None, ContextHints::default())
    }

    #[test]
    fn a_list_narrows_the_pool_to_its_models() {
        let mut pool = vec![
            facts("a", "m1", true, true),
            facts("b", "m2", true, true),
            facts("b", "m3", true, true),
        ];
        let code = restrict(&mut pool, &[entry("b", "m2")], &signature());
        assert_eq!(code, NARROWED_BY_PERSON);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool[0].model, "m2");
    }

    #[test]
    fn a_model_that_disappeared_leaves_the_pool_silently() {
        let mut pool = vec![facts("a", "m1", true, true), facts("b", "m2", true, true)];
        // `gone/x` is no longer an instance: dropped, the rest of the list applies.
        let code = restrict(
            &mut pool,
            &[entry("gone", "x"), entry("a", "m1")],
            &signature(),
        );
        assert_eq!(code, NARROWED_BY_PERSON);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool[0].provider_id, "a");
    }

    #[test]
    fn nothing_usable_on_the_list_falls_back_to_every_model_and_says_so() {
        let all = vec![
            facts("a", "m1", true, true),
            facts("b", "m2", false, true),
            facts("c", "m3", true, false),
        ];
        let mut pool = all.clone();
        // Deleted, consent withdrawn, unhealthy: nothing listed can serve.
        let code = restrict(
            &mut pool,
            &[entry("gone", "x"), entry("b", "m2"), entry("c", "m3")],
            &signature(),
        );
        assert_eq!(code, PERSON_LIST_UNAVAILABLE);
        assert_eq!(pool, all, "every available model is a candidate again");
    }

    #[test]
    fn the_document_round_trips_and_an_empty_list_means_every_model() {
        let doc: AutoModels =
            serde_json::from_str(r#"{"models":[{"provider":"a","model":"m1"}]}"#).unwrap();
        assert_eq!(doc.models, vec![entry("a", "m1")]);
        let empty: AutoModels = serde_json::from_str("{}").unwrap();
        assert!(empty.models.is_empty());
        assert_eq!(person_scope("u-1"), "person:u-1");
    }
}
