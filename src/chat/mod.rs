//! Chat module — conversational interface via Claude Code CLI (Nexus SDK)
//!
//! Provides WebSocket streaming chat with bidirectional communication,
//! event persistence with replay, session management, and auto-resume capabilities.

pub mod config;
pub mod manager;
pub mod prompt;
pub mod types;

pub use config::ChatConfig;
pub use manager::ChatManager;
pub use types::{ChatEvent, ChatRequest, ChatSession, ClientMessage};
