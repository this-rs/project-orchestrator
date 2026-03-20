//! HTTP API for the orchestrator

pub mod auth_handlers;
pub mod chat_handlers;
pub mod code_handlers;
pub mod episode_handlers;
pub mod feedback_handlers;
pub mod graph_types;
pub mod handlers;
pub mod hook_handlers;
pub mod neural_routing_handlers;
pub mod note_handlers;
pub mod persona_handlers;
pub mod profile_handlers;
pub mod project_handlers;
pub mod protocol_handlers;
pub mod query;
pub mod reason_handlers;
pub mod registry_handlers;
pub mod rfc_handlers;
pub mod routes;
pub mod sharing_handlers;
pub mod skill_handlers;
pub mod trajectory_handlers;
pub mod trigger_handlers;
pub mod workspace_handlers;
pub mod ws_auth;
pub mod ws_chat_handler;
pub mod ws_handlers;
pub mod ws_run_handler;

#[cfg(feature = "embedded-frontend")]
pub mod embedded_frontend;

pub use query::*;
pub use routes::create_router;
