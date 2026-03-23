//! MCP Client trait and transport implementations.
//!
//! Three transports per the MCP spec:
//! - **Stdio**: spawn a child process, JSON-RPC 2.0 over stdin/stdout
//! - **SSE**: legacy HTTP GET (server→client SSE) + HTTP POST (client→server)
//! - **StreamableHTTP**: POST with optional SSE response (MCP 2025-03)

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::{debug, trace, warn};

use super::McpTransport;

// ─────────────────────────────────────────────────────────────────────────────
// Client trait
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration passed to [`McpClient::connect`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTransportConfig {
    pub server_id: String,
    pub display_name: Option<String>,
    pub transport: McpTransport,
}

/// Result of the MCP `initialize` handshake.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: String,
    #[serde(default)]
    pub capabilities: Value,
    #[serde(default)]
    pub server_info: Option<ServerInfoResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfoResult {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
}

/// Result entry from `tools/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Value,
}

/// A connected MCP client that can communicate with an external server.
#[async_trait::async_trait]
pub trait McpClient: Send + Sync + std::fmt::Debug {
    /// Perform the initialize handshake.
    async fn initialize(&self) -> Result<InitializeResult>;

    /// Send `notifications/initialized` (no response expected).
    async fn initialized_notification(&self) -> Result<()>;

    /// List available tools via `tools/list`.
    async fn tools_list(&self) -> Result<Vec<McpToolDef>>;

    /// Call a tool via `tools/call`.
    async fn call_tool(&self, name: &str, arguments: Option<Value>) -> Result<Value>;

    /// Ping the server (`ping` method).
    async fn ping(&self) -> Result<()>;

    /// Gracefully shut down the connection.
    async fn shutdown(&self) -> Result<()>;

    /// Get the transport type name for logging.
    fn transport_name(&self) -> &'static str;
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────

/// Create an MCP client from a transport configuration.
pub async fn create_client(transport: &McpTransport) -> Result<Box<dyn McpClient>> {
    match transport {
        McpTransport::Stdio { command, args, env } => {
            let client = StdioMcpClient::spawn(command, args, env).await?;
            Ok(Box::new(client))
        }
        McpTransport::StreamableHttp { url, headers } => {
            let client = StreamableHttpMcpClient::new(url, headers.clone());
            Ok(Box::new(client))
        }
        McpTransport::Sse { url, headers } => {
            // SSE uses the same HTTP client but with different endpoint conventions
            let client = SseMcpClient::new(url, headers.clone());
            Ok(Box::new(client))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON-RPC helpers (client-side — we SEND requests and RECEIVE responses)
// ─────────────────────────────────────────────────────────────────────────────

/// JSON-RPC 2.0 request (client → server).
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
    id: u64,
}

/// JSON-RPC 2.0 response (server → client).
#[derive(Debug, Deserialize)]
struct JsonRpcClientResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    result: Option<Value>,
    error: Option<JsonRpcClientError>,
    #[allow(dead_code)]
    id: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcClientError {
    code: i32,
    message: String,
    #[allow(dead_code)]
    data: Option<Value>,
}

impl std::fmt::Display for JsonRpcClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON-RPC error {}: {}", self.code, self.message)
    }
}

/// Monotonically incrementing request ID generator (shared across all clients).
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_request_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn make_request(method: &str, params: Option<Value>) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0",
        method: method.to_string(),
        params,
        id: next_request_id(),
    }
}

/// Parse a JSON-RPC response, extracting the result or returning an error.
fn parse_response(response: JsonRpcClientResponse) -> Result<Value> {
    if let Some(err) = response.error {
        return Err(anyhow!("{}", err));
    }
    response
        .result
        .ok_or_else(|| anyhow!("JSON-RPC response has neither result nor error"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Stdio transport
// ─────────────────────────────────────────────────────────────────────────────

/// MCP client communicating via stdin/stdout of a spawned child process.
#[derive(Debug)]
pub struct StdioMcpClient {
    /// Child process (kept alive; killed on drop via the guard).
    child: Mutex<Child>,
    /// Stdin writer.
    stdin: Mutex<tokio::process::ChildStdin>,
    /// Stdout reader (buffered, one JSON-RPC message per line).
    stdout: Mutex<BufReader<tokio::process::ChildStdout>>,
}

impl StdioMcpClient {
    /// Spawn the child process.
    pub async fn spawn(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<Self> {
        debug!(command, ?args, "Spawning MCP server (stdio)");

        let mut cmd = Command::new(command);
        cmd.args(args)
            .envs(env)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn MCP server: {}", command))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("Failed to capture stdin of MCP server"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to capture stdout of MCP server"))?;

        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
        })
    }

    /// Send a JSON-RPC request and read the response.
    async fn send_request(&self, req: JsonRpcRequest) -> Result<Value> {
        let req_id = req.id;
        let method = req.method.clone();

        // Serialize + newline
        let mut payload = serde_json::to_vec(&req)?;
        payload.push(b'\n');

        trace!(id = req_id, method = %method, "→ stdio request");

        // Write
        {
            let mut stdin = self.stdin.lock().await;
            stdin.write_all(&payload).await?;
            stdin.flush().await?;
        }

        // Read response line
        let mut line = String::new();
        {
            let mut stdout = self.stdout.lock().await;
            let n = stdout.read_line(&mut line).await?;
            if n == 0 {
                return Err(anyhow!(
                    "MCP server closed stdout (method: {})",
                    method
                ));
            }
        }

        trace!(id = req_id, "← stdio response ({} bytes)", line.len());

        let response: JsonRpcClientResponse = serde_json::from_str(line.trim())
            .with_context(|| format!("Failed to parse JSON-RPC response for {}", method))?;

        parse_response(response)
    }

    /// Send a notification (no response expected).
    async fn send_notification(&self, method: &str, params: Option<Value>) -> Result<()> {
        // Notifications have no `id` field in JSON-RPC 2.0
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params.unwrap_or(Value::Object(Default::default())),
        });

        let mut bytes = serde_json::to_vec(&payload)?;
        bytes.push(b'\n');

        let mut stdin = self.stdin.lock().await;
        stdin.write_all(&bytes).await?;
        stdin.flush().await?;

        trace!(method, "→ stdio notification");
        Ok(())
    }
}

#[async_trait::async_trait]
impl McpClient for StdioMcpClient {
    async fn initialize(&self) -> Result<InitializeResult> {
        let params = super::initialize_params();
        let result = self.send_request(make_request("initialize", Some(params))).await?;
        let init: InitializeResult = serde_json::from_value(result)?;
        Ok(init)
    }

    async fn initialized_notification(&self) -> Result<()> {
        self.send_notification("notifications/initialized", None).await
    }

    async fn tools_list(&self) -> Result<Vec<McpToolDef>> {
        let result = self.send_request(make_request("tools/list", None)).await?;
        // Response is { tools: [...] }
        let tools_val = result
            .get("tools")
            .cloned()
            .unwrap_or(Value::Array(vec![]));
        let tools: Vec<McpToolDef> = serde_json::from_value(tools_val)?;
        Ok(tools)
    }

    async fn call_tool(&self, name: &str, arguments: Option<Value>) -> Result<Value> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments.unwrap_or(Value::Object(Default::default())),
        });
        self.send_request(make_request("tools/call", Some(params))).await
    }

    async fn ping(&self) -> Result<()> {
        self.send_request(make_request("ping", None)).await?;
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        // Try graceful shutdown, then kill
        if let Err(e) = self.send_notification("notifications/cancelled", None).await {
            debug!(error = %e, "Failed to send shutdown notification (server may have exited)");
        }
        let mut child = self.child.lock().await;
        let _ = child.kill().await;
        Ok(())
    }

    fn transport_name(&self) -> &'static str {
        "stdio"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Streamable HTTP transport (MCP 2025-03)
// ─────────────────────────────────────────────────────────────────────────────

/// MCP client over Streamable HTTP (POST JSON-RPC, receive JSON or SSE).
#[derive(Debug)]
pub struct StreamableHttpMcpClient {
    url: String,
    headers: HashMap<String, String>,
    http: reqwest::Client,
}

impl StreamableHttpMcpClient {
    pub fn new(url: &str, headers: HashMap<String, String>) -> Self {
        Self {
            url: url.trim_end_matches('/').to_string(),
            headers,
            http: reqwest::Client::new(),
        }
    }

    async fn post_rpc(&self, req: JsonRpcRequest) -> Result<Value> {
        let method = req.method.clone();
        trace!(method = %method, url = %self.url, "→ HTTP POST request");

        let mut builder = self.http.post(&self.url).json(&req);

        for (k, v) in &self.headers {
            builder = builder.header(k, v);
        }

        let resp = builder
            .send()
            .await
            .with_context(|| format!("HTTP POST failed for {}", method))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!(
                "HTTP {} from MCP server (method: {}): {}",
                status,
                method,
                body,
            ));
        }

        let body = resp.text().await?;
        trace!(method = %method, "← HTTP response ({} bytes)", body.len());

        let response: JsonRpcClientResponse = serde_json::from_str(&body)
            .with_context(|| format!("Failed to parse response for {}", method))?;

        parse_response(response)
    }
}

#[async_trait::async_trait]
impl McpClient for StreamableHttpMcpClient {
    async fn initialize(&self) -> Result<InitializeResult> {
        let params = super::initialize_params();
        let result = self.post_rpc(make_request("initialize", Some(params))).await?;
        let init: InitializeResult = serde_json::from_value(result)?;
        Ok(init)
    }

    async fn initialized_notification(&self) -> Result<()> {
        // Send as a request (streamable HTTP doesn't have true notifications)
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        });

        let mut builder = self.http.post(&self.url).json(&payload);
        for (k, v) in &self.headers {
            builder = builder.header(k, v);
        }
        let _ = builder.send().await;
        Ok(())
    }

    async fn tools_list(&self) -> Result<Vec<McpToolDef>> {
        let result = self.post_rpc(make_request("tools/list", None)).await?;
        let tools_val = result
            .get("tools")
            .cloned()
            .unwrap_or(Value::Array(vec![]));
        let tools: Vec<McpToolDef> = serde_json::from_value(tools_val)?;
        Ok(tools)
    }

    async fn call_tool(&self, name: &str, arguments: Option<Value>) -> Result<Value> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments.unwrap_or(Value::Object(Default::default())),
        });
        self.post_rpc(make_request("tools/call", Some(params))).await
    }

    async fn ping(&self) -> Result<()> {
        self.post_rpc(make_request("ping", None)).await?;
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        // HTTP is stateless — nothing to shut down
        Ok(())
    }

    fn transport_name(&self) -> &'static str {
        "streamable_http"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE transport (legacy MCP)
// ─────────────────────────────────────────────────────────────────────────────

/// MCP client over SSE (legacy).
///
/// Convention: the SSE endpoint is at `{url}/sse` (GET, server→client events).
/// Client→server messages go to `{url}/message` (POST).
/// The server sends an `endpoint` event first with the message URL.
///
/// For simplicity, we implement a request-response pattern:
/// POST the request, then parse the next SSE `message` event as the response.
/// This works for simple servers but does NOT support streaming responses.
#[derive(Debug)]
pub struct SseMcpClient {
    base_url: String,
    headers: HashMap<String, String>,
    http: reqwest::Client,
    /// The message endpoint URL (received from the SSE `endpoint` event).
    /// If not yet known, we derive it from base_url.
    message_url: Mutex<Option<String>>,
}

impl SseMcpClient {
    pub fn new(url: &str, headers: HashMap<String, String>) -> Self {
        let base = url.trim_end_matches('/').to_string();
        Self {
            base_url: base,
            headers,
            http: reqwest::Client::new(),
            message_url: Mutex::new(None),
        }
    }

    /// Get or derive the message URL.
    async fn get_message_url(&self) -> String {
        let lock = self.message_url.lock().await;
        if let Some(ref url) = *lock {
            url.clone()
        } else {
            // Default convention: POST to base_url/message
            format!("{}/message", self.base_url)
        }
    }

    /// Connect to the SSE endpoint to discover the message URL.
    async fn discover_endpoint(&self) -> Result<()> {
        let sse_url = format!("{}/sse", self.base_url);
        debug!(url = %sse_url, "Connecting to SSE endpoint");

        let mut builder = self.http.get(&sse_url);
        for (k, v) in &self.headers {
            builder = builder.header(k, v);
        }

        let resp = builder.send().await?;
        let body = resp.text().await?;

        // Parse SSE events looking for "event: endpoint"
        for line in body.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                let trimmed = data.trim();
                if trimmed.starts_with("http://") || trimmed.starts_with("https://") || trimmed.starts_with('/') {
                    let url = if trimmed.starts_with('/') {
                        // Relative URL — resolve against base
                        let base_origin = self.base_url.split("/sse").next().unwrap_or(&self.base_url);
                        format!("{}{}", base_origin, trimmed)
                    } else {
                        trimmed.to_string()
                    };
                    debug!(message_url = %url, "Discovered SSE message endpoint");
                    *self.message_url.lock().await = Some(url);
                    return Ok(());
                }
            }
        }

        // Fallback: use default
        debug!("No endpoint event found, using default /message");
        Ok(())
    }

    /// POST a JSON-RPC request to the message endpoint.
    async fn post_message(&self, req: JsonRpcRequest) -> Result<Value> {
        let url = self.get_message_url().await;
        let method = req.method.clone();
        trace!(method = %method, url = %url, "→ SSE POST request");

        let mut builder = self.http.post(&url).json(&req);
        for (k, v) in &self.headers {
            builder = builder.header(k, v);
        }

        let resp = builder.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("HTTP {} (method: {}): {}", status, method, body));
        }

        let body = resp.text().await?;

        // The response might be a direct JSON-RPC response or an SSE stream.
        // Try parsing as JSON first.
        if let Ok(response) = serde_json::from_str::<JsonRpcClientResponse>(&body) {
            return parse_response(response);
        }

        // Try parsing SSE: look for "data: {...}" lines
        for line in body.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(response) = serde_json::from_str::<JsonRpcClientResponse>(data.trim()) {
                    return parse_response(response);
                }
            }
        }

        Err(anyhow!("Could not parse response for {} from SSE server", method))
    }
}

#[async_trait::async_trait]
impl McpClient for SseMcpClient {
    async fn initialize(&self) -> Result<InitializeResult> {
        // First, try to discover the message endpoint via SSE
        if let Err(e) = self.discover_endpoint().await {
            warn!(error = %e, "SSE endpoint discovery failed, using defaults");
        }

        let params = super::initialize_params();
        let result = self.post_message(make_request("initialize", Some(params))).await?;
        let init: InitializeResult = serde_json::from_value(result)?;
        Ok(init)
    }

    async fn initialized_notification(&self) -> Result<()> {
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        });
        let url = self.get_message_url().await;
        let mut builder = self.http.post(&url).json(&payload);
        for (k, v) in &self.headers {
            builder = builder.header(k, v);
        }
        let _ = builder.send().await;
        Ok(())
    }

    async fn tools_list(&self) -> Result<Vec<McpToolDef>> {
        let result = self.post_message(make_request("tools/list", None)).await?;
        let tools_val = result
            .get("tools")
            .cloned()
            .unwrap_or(Value::Array(vec![]));
        let tools: Vec<McpToolDef> = serde_json::from_value(tools_val)?;
        Ok(tools)
    }

    async fn call_tool(&self, name: &str, arguments: Option<Value>) -> Result<Value> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments.unwrap_or(Value::Object(Default::default())),
        });
        self.post_message(make_request("tools/call", Some(params))).await
    }

    async fn ping(&self) -> Result<()> {
        self.post_message(make_request("ping", None)).await?;
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        // SSE connections are ephemeral on our side
        Ok(())
    }

    fn transport_name(&self) -> &'static str {
        "sse"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_request() {
        let req = make_request("tools/list", None);
        assert_eq!(req.method, "tools/list");
        assert_eq!(req.jsonrpc, "2.0");
        assert!(req.id > 0);
    }

    #[test]
    fn test_make_request_with_params() {
        let params = serde_json::json!({"name": "test_tool"});
        let req = make_request("tools/call", Some(params.clone()));
        assert_eq!(req.params, Some(params));
    }

    #[test]
    fn test_parse_success_response() {
        let response = JsonRpcClientResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(serde_json::json!({"tools": []})),
            error: None,
            id: Some(Value::Number(1.into())),
        };
        let result = parse_response(response).unwrap();
        assert!(result.get("tools").is_some());
    }

    #[test]
    fn test_parse_error_response() {
        let response = JsonRpcClientResponse {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(JsonRpcClientError {
                code: -32601,
                message: "Method not found".to_string(),
                data: None,
            }),
            id: Some(Value::Number(1.into())),
        };
        let err = parse_response(response).unwrap_err();
        assert!(err.to_string().contains("-32601"));
    }

    #[test]
    fn test_parse_initialize_result() {
        let json = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": true}},
            "serverInfo": {"name": "test-server", "version": "1.0"}
        });
        let result: InitializeResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.protocol_version, "2024-11-05");
        assert_eq!(result.server_info.unwrap().name, "test-server");
    }

    #[test]
    fn test_parse_tool_def() {
        let json = serde_json::json!({
            "name": "run_query",
            "description": "Execute a query",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        });
        let tool: McpToolDef = serde_json::from_value(json).unwrap();
        assert_eq!(tool.name, "run_query");
        assert_eq!(tool.description, Some("Execute a query".to_string()));
    }

    #[test]
    fn test_streamable_http_client_creation() {
        let client = StreamableHttpMcpClient::new(
            "https://example.com/mcp",
            HashMap::new(),
        );
        assert_eq!(client.url, "https://example.com/mcp");
    }

    #[test]
    fn test_transport_config_serde() {
        let config = McpTransportConfig {
            server_id: "grafeo".to_string(),
            display_name: Some("GrafeoDB".to_string()),
            transport: McpTransport::StreamableHttp {
                url: "https://grafeo.example.com/mcp".to_string(),
                headers: HashMap::new(),
            },
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("streamable_http"));
        let roundtrip: McpTransportConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.server_id, "grafeo");
    }
}
