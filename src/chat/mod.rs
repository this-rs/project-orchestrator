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
pub mod agent_runtime;
pub mod enrichment;
pub mod entity_extractor;
pub mod envelope;
pub mod feedback;
pub(crate) mod hook_ledger;
pub mod manager;
pub mod message_attachments;
pub mod model_catalog;
pub mod observation_detector;
pub(crate) mod oob_listener;
pub mod path_detect;
pub mod pending_queue;
pub(crate) mod post_stream;
pub(crate) mod post_tool_hook;
pub mod prompt;
pub mod prompt_sections;
pub mod provider;
pub mod routing;
pub(crate) mod skill_hook;
pub mod stages;
pub mod types;
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
