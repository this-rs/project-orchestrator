//! Chat module — conversational interface via Claude Code CLI (Nexus SDK)
//!
//! Provides WebSocket streaming chat with bidirectional communication,
//! event persistence with replay, session management, and auto-resume capabilities.

pub mod cli_version;
pub mod config;
pub mod enrichment;
pub mod entity_extractor;
pub mod manager;
pub mod path_detect;
pub mod prompt;
pub(crate) mod skill_hook;
pub mod stages;
pub mod types;
pub mod viz;
pub mod viz_builder;

pub use config::{ChatConfig, PermissionConfig};
pub use entity_extractor::{
    extract_entities, validate_entities, EntityType, ExtractedEntity, ExtractionSource,
    ValidatedEntity,
};
pub use manager::ChatManager;
pub use types::{ChatEvent, ChatRequest, ChatSession, ClientMessage};
