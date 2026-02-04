//! MCP Server implementation
//!
//! Implements the MCP server that communicates over stdio using JSON-RPC 2.0.

use super::handlers::ToolHandler;
use super::protocol::*;
use super::tools::all_tools;
use crate::orchestrator::Orchestrator;
use anyhow::Result;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "project-orchestrator";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// MCP Server that handles JSON-RPC 2.0 requests over stdio
pub struct McpServer {
    #[allow(dead_code)]
    orchestrator: Arc<Orchestrator>,
    tool_handler: ToolHandler,
    initialized: bool,
}

impl McpServer {
    /// Create a new MCP server with the given orchestrator
    pub fn new(orchestrator: Arc<Orchestrator>) -> Self {
        let tool_handler = ToolHandler::new(orchestrator.clone());
        Self {
            orchestrator,
            tool_handler,
            initialized: false,
        }
    }

    /// Run the server, reading from stdin and writing to stdout
    pub async fn run(&mut self) -> Result<()> {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        let reader = BufReader::new(stdin.lock());
        let mut writer = stdout.lock();

        info!("MCP server starting on stdio");

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    error!("Failed to read line: {}", e);
                    break;
                }
            };

            if line.is_empty() {
                continue;
            }

            debug!("Received: {}", line);

            let response = self.handle_message(&line).await;

            if let Some(resp) = response {
                let json = serde_json::to_string(&resp)?;
                debug!("Sending: {}", json);
                writeln!(writer, "{}", json)?;
                writer.flush()?;
            }
        }

        info!("MCP server shutting down");
        Ok(())
    }

    /// Handle a single JSON-RPC message
    async fn handle_message(&mut self, message: &str) -> Option<JsonRpcResponse> {
        // Parse the request
        let request: JsonRpcRequest = match serde_json::from_str(message) {
            Ok(r) => r,
            Err(e) => {
                return Some(JsonRpcResponse::error(
                    Value::Null,
                    JsonRpcError::parse_error(e.to_string()),
                ));
            }
        };

        // Get the request ID (notifications have no ID)
        let id = match &request.id {
            Some(id) => id.clone(),
            None => {
                // This is a notification, handle but don't respond
                self.handle_notification(&request).await;
                return None;
            }
        };

        // Handle the method
        let result = self.handle_request(&request).await;

        Some(match result {
            Ok(value) => JsonRpcResponse::success(id, value),
            Err(error) => JsonRpcResponse::error(id, error),
        })
    }

    /// Handle a notification (no response expected)
    async fn handle_notification(&mut self, request: &JsonRpcRequest) {
        match request.method.as_str() {
            "notifications/initialized" => {
                info!("Client confirmed initialization");
            }
            "notifications/cancelled" => {
                warn!("Request cancelled by client");
            }
            _ => {
                debug!("Unknown notification: {}", request.method);
            }
        }
    }

    /// Handle a request and return the result or error
    async fn handle_request(&mut self, request: &JsonRpcRequest) -> Result<Value, JsonRpcError> {
        match request.method.as_str() {
            // MCP Protocol methods
            "initialize" => self.handle_initialize(&request.params),
            "ping" => Ok(json!({})),
            "tools/list" => self.handle_tools_list(),
            "tools/call" => self.handle_tools_call(&request.params).await,

            // Unknown method
            _ => Err(JsonRpcError::method_not_found(&request.method)),
        }
    }

    /// Handle initialize request
    fn handle_initialize(&mut self, params: &Option<Value>) -> Result<Value, JsonRpcError> {
        let params: InitializeParams = params
            .as_ref()
            .map(|p| serde_json::from_value(p.clone()))
            .transpose()
            .map_err(|e| JsonRpcError::invalid_params(e.to_string()))?
            .unwrap_or(InitializeParams {
                protocol_version: PROTOCOL_VERSION.to_string(),
                capabilities: ClientCapabilities::default(),
                client_info: None,
            });

        info!(
            "Initializing MCP server (client protocol: {})",
            params.protocol_version
        );

        if let Some(ref client) = params.client_info {
            info!(
                "Client: {} v{}",
                client.name,
                client.version.as_deref().unwrap_or("unknown")
            );
        }

        self.initialized = true;

        let result = InitializeResult {
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ServerCapabilities {
                tools: ToolsCapability {
                    list_changed: false,
                },
            },
            server_info: ServerInfo {
                name: SERVER_NAME.to_string(),
                version: SERVER_VERSION.to_string(),
            },
        };

        serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(e.to_string()))
    }

    /// Handle tools/list request
    fn handle_tools_list(&self) -> Result<Value, JsonRpcError> {
        if !self.initialized {
            return Err(JsonRpcError::invalid_request("Server not initialized"));
        }

        let tools = all_tools();
        let result = ToolsListResult { tools };

        serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(e.to_string()))
    }

    /// Handle tools/call request
    async fn handle_tools_call(&self, params: &Option<Value>) -> Result<Value, JsonRpcError> {
        if !self.initialized {
            return Err(JsonRpcError::invalid_request("Server not initialized"));
        }

        let params: ToolCallParams = params
            .as_ref()
            .ok_or_else(|| JsonRpcError::invalid_params("params required"))?
            .clone()
            .pipe(serde_json::from_value)
            .map_err(|e| JsonRpcError::invalid_params(e.to_string()))?;

        info!("Tool call: {}", params.name);
        debug!("Arguments: {:?}", params.arguments);

        let result = self
            .tool_handler
            .handle(&params.name, params.arguments)
            .await;

        let tool_result = match result {
            Ok(value) => {
                ToolCallResult::success(serde_json::to_string_pretty(&value).unwrap_or_default())
            }
            Err(e) => {
                error!("Tool error: {}", e);
                ToolCallResult::error(e.to_string())
            }
        };

        serde_json::to_value(tool_result).map_err(|e| JsonRpcError::internal_error(e.to_string()))
    }
}

/// Extension trait for pipe operator
trait Pipe: Sized {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
    {
        f(self)
    }
}

impl<T> Pipe for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_initialize_request() {
        let request = r#"{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"claude-code","version":"1.0"}},"id":1}"#;
        let req: JsonRpcRequest = serde_json::from_str(request).unwrap();
        assert_eq!(req.method, "initialize");
        assert!(req.params.is_some());
    }

    #[test]
    fn test_parse_tools_list_request() {
        let request = r#"{"jsonrpc":"2.0","method":"tools/list","id":2}"#;
        let req: JsonRpcRequest = serde_json::from_str(request).unwrap();
        assert_eq!(req.method, "tools/list");
    }

    #[test]
    fn test_parse_tools_call_request() {
        let request = r#"{"jsonrpc":"2.0","method":"tools/call","params":{"name":"list_projects","arguments":{}},"id":3}"#;
        let req: JsonRpcRequest = serde_json::from_str(request).unwrap();
        assert_eq!(req.method, "tools/call");

        let params: ToolCallParams = serde_json::from_value(req.params.unwrap()).unwrap();
        assert_eq!(params.name, "list_projects");
    }

    #[test]
    fn test_success_response() {
        let resp = JsonRpcResponse::success(Value::Number(1.into()), json!({"status": "ok"}));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"result\""));
        assert!(json.contains("\"status\""));
    }

    #[test]
    fn test_error_response() {
        let resp = JsonRpcResponse::error(
            Value::Number(1.into()),
            JsonRpcError::method_not_found("unknown"),
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"error\""));
        assert!(json.contains("-32601"));
    }
}
