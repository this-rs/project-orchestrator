//! MCP (Model Context Protocol) server implementation
//!
//! This module provides an MCP server that exposes the orchestrator API
//! as tools for Claude Code and other MCP clients.

pub mod protocol;
pub mod tools;
pub mod handlers;
pub mod server;

pub use protocol::*;
pub use server::McpServer;
