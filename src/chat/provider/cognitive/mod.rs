//! Cognitive routing (RFC v2, decision R2): PO choosing the provider and model
//! itself, progressively, instead of asking the user to know every instance.
//!
//! This slice holds the SETTINGS only: the routing mode (`primary | mixed |
//! full`), the learning stage (`shadow | advisory | auto`) and the knobs of the
//! future scorer, stored globally with a per-project override. No decision is
//! taken here yet; the signature, candidates, scorer and decision log come in
//! later slices. Whatever the settings say, the resolver (`resolver.rs`) is
//! unchanged until then.

pub mod mode;

pub use mode::{
    effective_routing, parse_routing_settings, validate_routing_settings, EffectiveRouting,
    LearningStage, ProviderRoutingMode, RoutingError, RoutingScope, RoutingSettings, ROUTING_KEY,
};

use crate::chat::provider::settings::{project_scope, GLOBAL};
use crate::neo4j::GraphStore;

/// The routing document stored for one scope, `None` when there is none. A
/// document that no longer parses is logged and read as absent: a setting is
/// always read with a default, never migrated.
pub async fn stored_routing(
    graph: &dyn GraphStore,
    scope: &str,
) -> anyhow::Result<Option<RoutingSettings>> {
    let Some(raw) = graph.get_llm_setting(scope, ROUTING_KEY).await? else {
        return Ok(None);
    };
    match serde_json::from_str::<RoutingSettings>(&raw) {
        Ok(settings) => Ok(Some(settings)),
        Err(error) => {
            tracing::warn!(scope, %error, "unreadable routing document ignored: default applies");
            Ok(None)
        }
    }
}

/// The routing settings that apply to a request, and where they come from:
/// the project's document when the request names a project and one exists,
/// else the global document, else the default (`primary` + `shadow`).
pub async fn load_routing(
    graph: &dyn GraphStore,
    project_slug: Option<&str>,
) -> anyhow::Result<(RoutingSettings, RoutingScope)> {
    let global = stored_routing(graph, GLOBAL).await?;
    let project = match project_slug {
        Some(slug) => stored_routing(graph, &project_scope(slug)).await?,
        None => None,
    };
    Ok(effective_routing(global, project))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;

    #[tokio::test]
    async fn load_routing_reads_project_over_global_over_default() {
        let graph = MockGraphStore::new();
        let (s, scope) = load_routing(&graph, Some("p")).await.unwrap();
        assert_eq!(
            (s, scope),
            (RoutingSettings::default(), RoutingScope::Default)
        );

        graph
            .put_llm_setting(GLOBAL, ROUTING_KEY, r#"{"mode":"mixed"}"#)
            .await
            .unwrap();
        let (s, scope) = load_routing(&graph, Some("p")).await.unwrap();
        assert_eq!(s.mode, ProviderRoutingMode::Mixed);
        assert_eq!(s.stage, LearningStage::Shadow, "absent field = default");
        assert_eq!(scope, RoutingScope::Global);

        graph
            .put_llm_setting(
                &project_scope("p"),
                ROUTING_KEY,
                r#"{"mode":"full","stage":"advisory","exploration_epsilon":0.2}"#,
            )
            .await
            .unwrap();
        let (s, scope) = load_routing(&graph, Some("p")).await.unwrap();
        assert_eq!(s.mode, ProviderRoutingMode::Full);
        assert_eq!(s.stage, LearningStage::Advisory);
        assert_eq!(s.exploration_epsilon, 0.2);
        assert_eq!(scope, RoutingScope::Project);
        // Another project, or no project, still sees the global one.
        assert_eq!(
            load_routing(&graph, Some("other")).await.unwrap().1,
            RoutingScope::Global
        );
        assert_eq!(
            load_routing(&graph, None).await.unwrap().1,
            RoutingScope::Global
        );
    }

    #[tokio::test]
    async fn an_unreadable_document_reads_as_absent() {
        let graph = MockGraphStore::new();
        graph
            .put_llm_setting(GLOBAL, ROUTING_KEY, "not json")
            .await
            .unwrap();
        graph
            .put_llm_setting(&project_scope("p"), ROUTING_KEY, r#"{"mode":"turbo"}"#)
            .await
            .unwrap();
        assert_eq!(stored_routing(&graph, GLOBAL).await.unwrap(), None);
        assert_eq!(
            load_routing(&graph, Some("p")).await.unwrap(),
            (RoutingSettings::default(), RoutingScope::Default)
        );
    }
}
