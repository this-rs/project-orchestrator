//! Reads of the provider settings stored in the graph (`LlmSetting` documents).
//!
//! One place for the keys and the decoding, shared by the settings API and by
//! the manager when it resolves the provider of a session. A document that no
//! longer decodes is skipped, never guessed.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use anyhow::Result;

use super::settings::{
    consent_scope, ConsentRecord, InstanceRecord, ModelAlias, RoleAssignments, ALIASES_KEY,
    CONSENT_PREFIX, GLOBAL, INSTANCE_PREFIX, ROLES_KEY,
};
use crate::neo4j::GraphStore;

fn decode<T: serde::de::DeserializeOwned>(raw: &str) -> Option<T> {
    serde_json::from_str(raw).ok()
}

/// Every stored instance.
pub async fn instances(graph: &dyn GraphStore) -> Result<Vec<InstanceRecord>> {
    Ok(graph
        .list_llm_settings(GLOBAL, INSTANCE_PREFIX)
        .await?
        .iter()
        .filter_map(|(_, v)| decode(v))
        .collect())
}

/// One stored instance.
pub async fn instance(graph: &dyn GraphStore, id: &str) -> Result<Option<InstanceRecord>> {
    Ok(graph
        .get_llm_setting(GLOBAL, &format!("{INSTANCE_PREFIX}{id}"))
        .await?
        .and_then(|v| decode(&v)))
}

/// The alias table.
pub async fn aliases(graph: &dyn GraphStore) -> Result<Vec<ModelAlias>> {
    Ok(graph
        .get_llm_setting(GLOBAL, ALIASES_KEY)
        .await?
        .and_then(|v| decode(&v))
        .unwrap_or_default())
}

/// The consents of a project.
pub async fn consents(graph: &dyn GraphStore, slug: &str) -> Result<Vec<ConsentRecord>> {
    consents_for(graph, Some(slug)).await
}

/// The consents that apply to a session: its project's, or, outside any project, those
/// given for the sessions without one ([`super::settings::NO_PROJECT_SCOPE`], B3).
pub async fn consents_for(
    graph: &dyn GraphStore,
    project_slug: Option<&str>,
) -> Result<Vec<ConsentRecord>> {
    Ok(graph
        .list_llm_settings(&consent_scope(project_slug), CONSENT_PREFIX)
        .await?
        .iter()
        .filter_map(|(_, v)| decode(v))
        .collect())
}

/// The roles of a scope (`global` or `project:<slug>`); absent = empty.
pub async fn roles(graph: &dyn GraphStore, scope: &str) -> Result<RoleAssignments> {
    Ok(graph
        .get_llm_setting(scope, ROLES_KEY)
        .await?
        .and_then(|v| decode(&v))
        .unwrap_or_default())
}
