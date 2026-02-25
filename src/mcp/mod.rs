//! MCP (Model Context Protocol) server implementation
//!
//! This module provides an MCP server that exposes the orchestrator API
//! as tools for Claude Code and other MCP clients.

pub mod formatter;
pub mod handlers;
pub mod http_client;
pub mod protocol;
pub mod server;
pub mod tools;

pub use http_client::McpHttpClient;
pub use protocol::*;
pub use server::McpServer;
