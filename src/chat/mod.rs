//! Chat module — conversational interface via Claude Code CLI (Nexus SDK)
//!
//! Provides WebSocket streaming chat with bidirectional communication,
//! event persistence with replay, session management, and auto-resume capabilities.

pub mod attachment;
pub mod attention;
pub mod cli_auth;
pub mod cli_version;
pub mod compaction_context;
pub mod composer;
pub mod config;
pub mod continuity;
pub mod control_pump;
pub(crate) mod drain;

#[cfg(test)]
mod agent_e2e_tests;
pub(crate) mod agent_hooks;
pub mod agent_runtime;
pub mod anchor;
pub mod anchor_resolver;
pub mod cost;
#[cfg(test)]
mod engine_parity_tests;
pub mod enrichment;
pub mod entity_extractor;
pub mod envelope;
pub mod feedback;
pub(crate) mod hook_ledger;
pub mod manager;
pub mod message_attachments;
pub mod model_catalog;
pub mod neutral_place;
pub mod observation_detector;
pub(crate) mod oob_listener;
pub mod path_detect;
pub mod pending_queue;
pub(crate) mod post_stream;
pub(crate) mod post_tool_hook;
pub mod prompt;
pub mod prompt_sections;
pub mod provider;
#[cfg(test)]
mod refs_wiring_tests;
pub mod relay;
pub mod routing;
#[cfg(test)]
mod routing_modes_e2e_tests;
pub mod session_record;
pub(crate) mod skill_hook;
pub mod stages;
pub mod tree;
pub mod types;
pub mod untrusted;
pub mod viz;
pub mod viz_builder;
#[cfg(test)]
mod wire_contract;

pub use config::{ChatConfig, PermissionConfig};
pub use entity_extractor::{
    extract_entities, validate_entities, EntityType, ExtractedEntity, ExtractionSource,
    ValidatedEntity,
};
pub use manager::{ChatManager, LiveSessionSnapshot};
pub use types::{ChatEvent, ChatRequest, ChatSession, ClientMessage, SessionActivity, SpawnedBy};
