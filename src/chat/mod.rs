//! Chat module — conversational interface via Claude Code CLI (Nexus SDK)
//!
//! Provides SSE streaming chat endpoint with bidirectional communication,
//! session persistence, and auto-resume capabilities.

pub mod config;
pub mod manager;
pub mod types;

pub use config::ChatConfig;
pub use manager::ChatManager;
pub use types::{ChatEvent, ChatRequest, ChatSession, ClientMessage};
