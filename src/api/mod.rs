//! HTTP API for the orchestrator
#[allow(unused_imports)]
use crate::neo4j::traits::*;

pub mod activity_handlers;
pub mod attention;
pub mod attention_aggregate;
pub mod auth_handlers;
pub mod chat_handlers;
#[cfg(test)]
mod chat_times_tests;
pub mod code_handlers;
#[cfg(test)]
mod decision_global_tests;
pub mod document_handlers;
pub mod environment_handlers;
pub mod episode_handlers;
#[cfg(test)]
mod episode_session_tests;
pub mod execution_place_handlers;
pub mod feedback_handlers;
pub mod graph_handlers;
pub mod graph_types;
pub mod handlers;
pub mod hook_handlers;
pub mod ingress_handlers;
#[cfg(test)]
pub(crate) mod list_routes_tests;
pub mod mcp_federation_handlers;
pub mod network_tools_handlers;
pub mod neural_routing_handlers;
pub mod note_handlers;
#[cfg(test)]
mod permission_ws_tests;
pub mod persona_handlers;
pub mod profile_handlers;
pub mod project_handlers;
pub mod protocol_handlers;
pub mod provider_handlers;
pub mod query;
pub mod quota_handlers;
pub mod reason_handlers;
pub mod refs_handlers;
#[cfg(test)]
mod refs_ws_tests;
pub mod registry_handlers;
#[cfg(test)]
mod retrieval_ledger_tests;
pub mod rfc_handlers;
pub mod routes;
pub mod routing_handlers;
pub mod sharing_handlers;
pub mod skill_handlers;
pub mod team_handlers;
pub mod trajectory_handlers;
pub mod trigger_handlers;
pub mod update_handlers;
pub mod vault_handlers;
pub mod workspace_handlers;
pub mod ws_activity_handler;
#[cfg(test)]
mod ws_activity_tests;
pub mod ws_auth;
pub mod ws_chat_handler;
pub mod ws_handlers;
pub mod ws_run_handler;

#[cfg(feature = "embedded-frontend")]
pub mod embedded_frontend;

pub use query::*;
pub use routes::create_router;

/// Distinguishes "field absent" from "field set to null".
///
/// `Option<T>` alone conflates the two: `{}` and `{"project_id": null}` both
/// give `None`. Re-scoping needs them apart — absent means "leave the scope
/// alone", null means "make it transverse". Declared as `Option<Option<T>>`
/// with `#[serde(default, deserialize_with = "double_option")]`.
pub fn double_option<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    serde::Deserialize::deserialize(deserializer).map(Some)
}
