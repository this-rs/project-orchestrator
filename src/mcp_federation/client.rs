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

/// Parse a JSON-RPC response from an SSE event stream body.
///
/// Scans `data:` lines for a valid JSON-RPC response. Per MCP 2025-03 spec,
/// the server may send the response as one or more SSE events with `data:` prefixed
/// JSON-RPC messages.
fn parse_sse_body(body: &str, method: &str) -> Result<Value> {
    for line in body.lines() {
        let data = if let Some(d) = line.strip_prefix("data: ") {
            d.trim()
        } else if let Some(d) = line.strip_prefix("data:") {
            d.trim()
        } else {
            continue;
        };

        if data.is_empty() {
            continue;
        }

        if let Ok(response) = serde_json::from_str::<JsonRpcClientResponse>(data) {
            return parse_response(response);
        }
    }

    Err(anyhow!(
        "No valid JSON-RPC response found in SSE stream for {}",
        method
    ))
}

/// Parse a response body based on its Content-Type.
///
/// Shared by `StreamableHttpMcpClient` and `SseMcpClient` to avoid duplicating
/// the JSON-vs-SSE dispatch logic.
fn parse_body(body: &str, content_type: &str, method: &str) -> Result<Value> {
    if content_type.contains("text/event-stream") {
        parse_sse_body(body, method)
    } else if content_type.contains("application/json") {
        let response: JsonRpcClientResponse = serde_json::from_str(body)
            .with_context(|| format!("Failed to parse JSON response for {}", method))?;
        parse_response(response)
    } else {
        // Unknown Content-Type — try JSON first, then SSE as fallback
        debug!(
            method = %method,
            content_type = %content_type,
            "Unknown Content-Type, trying JSON then SSE fallback"
        );
        if let Ok(response) = serde_json::from_str::<JsonRpcClientResponse>(body) {
            return parse_response(response);
        }
        parse_sse_body(body, method)
    }
}

/// Discover the SSE message endpoint from a legacy SSE stream.
///
/// GET `{base_url}/sse` → look for a `data:` line containing a URL (the message endpoint).
/// Returns the resolved absolute URL.
///
/// Shared by `StreamableHttpMcpClient::try_legacy_sse_fallback` and
/// `SseMcpClient::discover_endpoint`.
async fn discover_sse_message_url(
    http: &reqwest::Client,
    base_url: &str,
    headers: &HashMap<String, String>,
) -> Result<Option<String>> {
    let sse_url = format!("{}/sse", base_url);
    debug!(url = %sse_url, "Connecting to SSE endpoint for discovery");

    let mut builder = http.get(&sse_url);
    for (k, v) in headers {
        builder = builder.header(k, v);
    }

    let resp = builder
        .send()
        .await
        .with_context(|| "SSE endpoint discovery failed")?;
    let body = resp.text().await?;

    // Parse SSE events looking for a data: line with a URL
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            let trimmed = data.trim();
            if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
                return Ok(Some(trimmed.to_string()));
            } else if trimmed.starts_with('/') {
                // Relative URL — resolve against base
                let resolved = reqwest::Url::parse(base_url)
                    .ok()
                    .and_then(|u| u.join(trimmed).ok())
                    .map(|u| u.to_string())
                    .unwrap_or_else(|| format!("{}{}", base_url, trimmed));
                return Ok(Some(resolved));
            }
        }
    }

    Ok(None)
}

/// Apply custom headers to an HTTP request builder.
fn apply_headers(
    mut builder: reqwest::RequestBuilder,
    headers: &HashMap<String, String>,
) -> reqwest::RequestBuilder {
    for (k, v) in headers {
        builder = builder.header(k, v);
    }
    builder
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
                return Err(anyhow!("MCP server closed stdout (method: {})", method));
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
        let result = self
            .send_request(make_request("initialize", Some(params)))
            .await?;
        let init: InitializeResult = serde_json::from_value(result)?;
        Ok(init)
    }

    async fn initialized_notification(&self) -> Result<()> {
        self.send_notification("notifications/initialized", None)
            .await
    }

    async fn tools_list(&self) -> Result<Vec<McpToolDef>> {
        let result = self.send_request(make_request("tools/list", None)).await?;
        // Response is { tools: [...] }
        let tools_val = result.get("tools").cloned().unwrap_or(Value::Array(vec![]));
        let tools: Vec<McpToolDef> = serde_json::from_value(tools_val)?;
        Ok(tools)
    }

    async fn call_tool(&self, name: &str, arguments: Option<Value>) -> Result<Value> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments.unwrap_or(Value::Object(Default::default())),
        });
        self.send_request(make_request("tools/call", Some(params)))
            .await
    }

    async fn ping(&self) -> Result<()> {
        self.send_request(make_request("ping", None)).await?;
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        // Try graceful shutdown, then kill
        if let Err(e) = self
            .send_notification("notifications/cancelled", None)
            .await
        {
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
    /// Session ID assigned by the server during initialize (MCP 2025-03).
    session_id: Mutex<Option<String>>,
    /// Legacy SSE fallback: if initialize POST gets 4xx, we discover the old SSE
    /// message endpoint and redirect all subsequent requests there.
    fallback_message_url: Mutex<Option<String>>,
}

impl StreamableHttpMcpClient {
    pub fn new(url: &str, headers: HashMap<String, String>) -> Self {
        Self {
            url: url.trim_end_matches('/').to_string(),
            headers,
            http: reqwest::Client::new(),
            session_id: Mutex::new(None),
            fallback_message_url: Mutex::new(None),
        }
    }

    /// Get the effective POST URL (fallback URL if SSE legacy mode is active).
    async fn effective_url(&self) -> String {
        if let Some(ref url) = *self.fallback_message_url.lock().await {
            url.clone()
        } else {
            self.url.clone()
        }
    }

    /// Build an HTTP POST request with common headers (Accept, Session-Id, user headers).
    async fn build_post(&self, body: &impl Serialize) -> reqwest::RequestBuilder {
        let url = self.effective_url().await;
        let mut builder = self
            .http
            .post(&url)
            .header("Accept", "application/json, text/event-stream")
            .json(body);

        // Include Mcp-Session-Id if we have one (MCP 2025-03 session management)
        if let Some(ref sid) = *self.session_id.lock().await {
            builder = builder.header("Mcp-Session-Id", sid.as_str());
        }

        apply_headers(builder, &self.headers)
    }

    /// Attempt legacy SSE endpoint discovery (MCP 2024-11-05 backward compatibility).
    ///
    /// Uses the shared `discover_sse_message_url` helper to find the message endpoint,
    /// then POSTs the initialize request there.
    async fn try_legacy_sse_fallback(&self, params: Value) -> Result<InitializeResult> {
        debug!("Attempting legacy SSE endpoint discovery");

        let discovered = discover_sse_message_url(&self.http, &self.url, &self.headers).await?;
        let msg_url = discovered.unwrap_or_else(|| format!("{}/message", self.url));
        debug!(message_url = %msg_url, "Discovered legacy SSE message endpoint");
        *self.fallback_message_url.lock().await = Some(msg_url.clone());

        // POST the initialize request to the discovered message endpoint
        let req = make_request("initialize", Some(params));
        let builder = apply_headers(
            self.http
                .post(&msg_url)
                .header("Accept", "application/json, text/event-stream")
                .json(&req),
            &self.headers,
        );

        let resp = builder
            .send()
            .await
            .with_context(|| "Legacy SSE initialize POST failed")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!(
                "Legacy SSE initialize failed: HTTP {} — {}",
                status,
                body
            ));
        }

        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();
        let body = resp.text().await?;

        let result = parse_body(&body, &content_type, "initialize")?;
        let init: InitializeResult = serde_json::from_value(result)?;
        Ok(init)
    }

    /// Send a JSON-RPC request, returning both the parsed value and response headers.
    ///
    /// This is the single internal method that handles all POST logic:
    /// status checks, 404 session expiry, Content-Type dispatch.
    async fn post_rpc_inner(
        &self,
        req: JsonRpcRequest,
    ) -> Result<(Value, reqwest::header::HeaderMap)> {
        let method = req.method.clone();
        trace!(method = %method, url = %self.url, "→ HTTP POST request");

        let builder = self.build_post(&req).await;

        let resp = builder
            .send()
            .await
            .with_context(|| format!("HTTP POST failed for {}", method))?;

        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            // MCP 2025-03: 404 means session expired — single lock
            let mut sid = self.session_id.lock().await;
            if sid.is_some() {
                warn!(method = %method, "Server returned 404 — session expired, clearing session ID");
                *sid = None;
                return Err(anyhow!(
                    "MCP session expired (HTTP 404). Re-initialize required."
                ));
            }
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!(
                "HTTP {} from MCP server (method: {}): {}",
                status,
                method,
                body,
            ));
        }

        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();
        let headers = resp.headers().clone();

        let body = resp.text().await?;
        trace!(method = %method, content_type = %content_type, "← HTTP response ({} bytes)", body.len());

        let value = parse_body(&body, &content_type, &method)?;
        Ok((value, headers))
    }

    /// Send a JSON-RPC request and parse the response. Returns the parsed result value.
    async fn post_rpc(&self, req: JsonRpcRequest) -> Result<Value> {
        self.post_rpc_inner(req).await.map(|(v, _)| v)
    }

    /// Send a JSON-RPC request and return both the parsed value AND the raw response headers.
    /// Used by initialize() to capture the Mcp-Session-Id header.
    async fn post_rpc_with_headers(
        &self,
        req: JsonRpcRequest,
    ) -> Result<(Value, reqwest::header::HeaderMap)> {
        self.post_rpc_inner(req).await
    }
}

#[async_trait::async_trait]
impl McpClient for StreamableHttpMcpClient {
    async fn initialize(&self) -> Result<InitializeResult> {
        let params = super::initialize_params();
        let req = make_request("initialize", Some(params.clone()));

        match self.post_rpc_with_headers(req).await {
            Ok((result, headers)) => {
                // MCP 2025-03: capture Mcp-Session-Id from response headers
                if let Some(sid) = headers.get("mcp-session-id").and_then(|v| v.to_str().ok()) {
                    debug!(session_id = %sid, "Captured MCP session ID from server");
                    *self.session_id.lock().await = Some(sid.to_string());
                }
                let init: InitializeResult = serde_json::from_value(result)?;
                Ok(init)
            }
            Err(e) => {
                let err_str = e.to_string();
                // MCP backward compatibility: if POST returns 4xx (404/405),
                // fall back to legacy SSE transport discovery
                if err_str.contains("HTTP 404") || err_str.contains("HTTP 405") {
                    debug!(
                        error = %err_str,
                        "Streamable HTTP initialize failed with 4xx, trying legacy SSE fallback"
                    );
                    self.try_legacy_sse_fallback(params).await
                } else {
                    Err(e)
                }
            }
        }
    }

    async fn initialized_notification(&self) -> Result<()> {
        // MCP 2025-03: notifications get 202 Accepted with no body
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        });

        let builder = self.build_post(&payload).await;
        let resp = builder.send().await;

        if let Ok(resp) = resp {
            let status = resp.status();
            if status != reqwest::StatusCode::ACCEPTED && !status.is_success() {
                warn!(status = %status, "initialized notification got unexpected status (expected 202)");
            }
        }
        Ok(())
    }

    async fn tools_list(&self) -> Result<Vec<McpToolDef>> {
        let result = self.post_rpc(make_request("tools/list", None)).await?;
        let tools_val = result.get("tools").cloned().unwrap_or(Value::Array(vec![]));
        let tools: Vec<McpToolDef> = serde_json::from_value(tools_val)?;
        Ok(tools)
    }

    async fn call_tool(&self, name: &str, arguments: Option<Value>) -> Result<Value> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments.unwrap_or(Value::Object(Default::default())),
        });
        self.post_rpc(make_request("tools/call", Some(params)))
            .await
    }

    async fn ping(&self) -> Result<()> {
        self.post_rpc(make_request("ping", None)).await?;
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        // MCP 2025-03: if we have a session, send HTTP DELETE to terminate it
        let sid = self.session_id.lock().await.clone();
        if let Some(ref session_id) = sid {
            debug!(session_id = %session_id, "Terminating MCP session via HTTP DELETE");
            let builder = apply_headers(
                self.http
                    .delete(&self.url)
                    .header("Mcp-Session-Id", session_id.as_str()),
                &self.headers,
            );
            match builder.send().await {
                Ok(resp) if resp.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED => {
                    debug!("Server does not support session termination (405), ignoring");
                }
                Ok(resp) if resp.status().is_success() => {
                    debug!("MCP session terminated successfully");
                }
                Ok(resp) => {
                    debug!(status = %resp.status(), "Unexpected response to session DELETE");
                }
                Err(e) => {
                    debug!(error = %e, "Failed to send session DELETE (non-fatal)");
                }
            }
            *self.session_id.lock().await = None;
        }
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
    ///
    /// Uses the shared `discover_sse_message_url` helper.
    async fn discover_endpoint(&self) -> Result<()> {
        if let Some(url) =
            discover_sse_message_url(&self.http, &self.base_url, &self.headers).await?
        {
            debug!(message_url = %url, "Discovered SSE message endpoint");
            *self.message_url.lock().await = Some(url);
        } else {
            debug!("No endpoint event found, using default /message");
        }
        Ok(())
    }

    /// POST a JSON-RPC request to the message endpoint.
    async fn post_message(&self, req: JsonRpcRequest) -> Result<Value> {
        let url = self.get_message_url().await;
        let method = req.method.clone();
        trace!(method = %method, url = %url, "→ SSE POST request");

        let builder = apply_headers(self.http.post(&url).json(&req), &self.headers);

        let resp = builder.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("HTTP {} (method: {}): {}", status, method, body));
        }

        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();
        let body = resp.text().await?;

        parse_body(&body, &content_type, &method)
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
        let result = self
            .post_message(make_request("initialize", Some(params)))
            .await?;
        let init: InitializeResult = serde_json::from_value(result)?;
        Ok(init)
    }

    async fn initialized_notification(&self) -> Result<()> {
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        });
        let url = self.get_message_url().await;
        let builder = apply_headers(self.http.post(&url).json(&payload), &self.headers);
        let _ = builder.send().await;
        Ok(())
    }

    async fn tools_list(&self) -> Result<Vec<McpToolDef>> {
        let result = self.post_message(make_request("tools/list", None)).await?;
        let tools_val = result.get("tools").cloned().unwrap_or(Value::Array(vec![]));
        let tools: Vec<McpToolDef> = serde_json::from_value(tools_val)?;
        Ok(tools)
    }

    async fn call_tool(&self, name: &str, arguments: Option<Value>) -> Result<Value> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments.unwrap_or(Value::Object(Default::default())),
        });
        self.post_message(make_request("tools/call", Some(params)))
            .await
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
        let client = StreamableHttpMcpClient::new("https://example.com/mcp", HashMap::new());
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

    #[test]
    fn test_create_client_streamable_http() {
        // StreamableHttpMcpClient::new() does not require network
        let client = StreamableHttpMcpClient::new("https://example.com/mcp", HashMap::new());
        assert_eq!(client.url, "https://example.com/mcp");
        assert_eq!(client.transport_name(), "streamable_http");
    }

    #[test]
    fn test_create_client_sse() {
        let client = SseMcpClient::new("https://example.com/sse", HashMap::new());
        assert_eq!(client.base_url, "https://example.com/sse");
        assert_eq!(client.transport_name(), "sse");
    }

    #[test]
    fn test_next_request_id_increments() {
        let id1 = next_request_id();
        let id2 = next_request_id();
        assert!(
            id2 > id1,
            "Second ID ({}) should be greater than first ({})",
            id2,
            id1
        );
    }

    #[test]
    fn test_make_request_notification() {
        let req = make_request("notifications/initialized", None);
        assert_eq!(req.method, "notifications/initialized");
        assert!(req.params.is_none());
        assert_eq!(req.jsonrpc, "2.0");
    }

    #[test]
    fn test_parse_response_error() {
        let response = JsonRpcClientResponse {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(JsonRpcClientError {
                code: -32600,
                message: "Invalid Request".to_string(),
                data: None,
            }),
            id: Some(Value::Number(42.into())),
        };
        let err = parse_response(response).unwrap_err();
        assert!(err.to_string().contains("-32600"));
        assert!(err.to_string().contains("Invalid Request"));
    }

    #[test]
    fn test_parse_response_no_result_no_error() {
        let response = JsonRpcClientResponse {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: None,
            id: Some(Value::Number(1.into())),
        };
        let err = parse_response(response).unwrap_err();
        assert!(
            err.to_string().contains("neither result nor error"),
            "Expected 'neither result nor error' in: {}",
            err
        );
    }

    #[test]
    fn test_json_rpc_error_display() {
        let error = JsonRpcClientError {
            code: -32601,
            message: "Method not found".to_string(),
            data: Some(serde_json::json!({"detail": "unknown"})),
        };
        let display = format!("{}", error);
        assert_eq!(display, "JSON-RPC error -32601: Method not found");
    }

    #[test]
    fn test_transport_config_serde_roundtrip() {
        let config = McpTransportConfig {
            server_id: "test-srv".to_string(),
            display_name: None,
            transport: McpTransport::Sse {
                url: "https://example.com/sse".to_string(),
                headers: {
                    let mut h = HashMap::new();
                    h.insert("Authorization".to_string(), "Bearer tok".to_string());
                    h
                },
            },
        };
        let json = serde_json::to_string(&config).unwrap();
        let roundtrip: McpTransportConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.server_id, "test-srv");
        assert!(roundtrip.display_name.is_none());
        match roundtrip.transport {
            McpTransport::Sse {
                ref url,
                ref headers,
            } => {
                assert_eq!(url, "https://example.com/sse");
                assert_eq!(headers.get("Authorization").unwrap(), "Bearer tok");
            }
            _ => panic!("Expected Sse transport"),
        }
    }

    #[test]
    fn test_initialize_result_serde() {
        let result = InitializeResult {
            protocol_version: "2024-11-05".to_string(),
            capabilities: serde_json::json!({"tools": {"listChanged": true}}),
            server_info: Some(ServerInfoResult {
                name: "my-server".to_string(),
                version: Some("1.2.3".to_string()),
            }),
        };
        let json = serde_json::to_string(&result).unwrap();
        let roundtrip: InitializeResult = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.protocol_version, "2024-11-05");
        assert_eq!(roundtrip.server_info.as_ref().unwrap().name, "my-server");
        assert_eq!(
            roundtrip.server_info.as_ref().unwrap().version,
            Some("1.2.3".to_string())
        );
    }

    #[test]
    fn test_mcp_tool_def_serde_minimal() {
        // McpToolDef with only name — description and input_schema use defaults
        let json = serde_json::json!({"name": "bare_tool"});
        let tool: McpToolDef = serde_json::from_value(json).unwrap();
        assert_eq!(tool.name, "bare_tool");
        assert!(tool.description.is_none());
        assert_eq!(tool.input_schema, Value::Null);

        // Roundtrip
        let serialized = serde_json::to_string(&tool).unwrap();
        let back: McpToolDef = serde_json::from_str(&serialized).unwrap();
        assert_eq!(back.name, "bare_tool");
    }

    #[test]
    fn test_parse_sse_body_single_event() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"result\":{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{},\"serverInfo\":{\"name\":\"test\"}},\"id\":1}\n\n";
        let result = parse_sse_body(body, "initialize").unwrap();
        assert_eq!(result["protocolVersion"], "2025-03-26");
        assert_eq!(result["serverInfo"]["name"], "test");
    }

    #[test]
    fn test_parse_sse_body_no_space_after_data_prefix() {
        let body = "data:{\"jsonrpc\":\"2.0\",\"result\":{\"tools\":[]},\"id\":2}\n\n";
        let result = parse_sse_body(body, "tools/list").unwrap();
        assert!(result.get("tools").is_some());
    }

    #[test]
    fn test_parse_sse_body_empty_data_lines_skipped() {
        let body =
            "data: \ndata: \ndata: {\"jsonrpc\":\"2.0\",\"result\":{\"ok\":true},\"id\":3}\n\n";
        let result = parse_sse_body(body, "test").unwrap();
        assert_eq!(result["ok"], true);
    }

    #[test]
    fn test_parse_sse_body_no_valid_response() {
        let body = "event: ping\n: comment\n\n";
        let err = parse_sse_body(body, "initialize").unwrap_err();
        assert!(err.to_string().contains("No valid JSON-RPC response"));
    }

    #[test]
    fn test_parse_sse_body_with_error_response() {
        let body = "data: {\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32601,\"message\":\"Method not found\"},\"id\":4}\n\n";
        let err = parse_sse_body(body, "unknown").unwrap_err();
        assert!(err.to_string().contains("-32601"));
    }

    #[test]
    fn test_parse_sse_body_multiple_events_takes_first_valid() {
        let body = "data: not-json\ndata: {\"jsonrpc\":\"2.0\",\"result\":{\"first\":true},\"id\":5}\ndata: {\"jsonrpc\":\"2.0\",\"result\":{\"second\":true},\"id\":6}\n\n";
        let result = parse_sse_body(body, "test").unwrap();
        assert_eq!(result["first"], true);
    }

    // ========================================================================
    // parse_body (Content-Type dispatch)
    // ========================================================================

    #[test]
    fn test_parse_body_json_content_type() {
        let body = r#"{"jsonrpc":"2.0","result":{"ok":true},"id":1}"#;
        let result = parse_body(body, "application/json", "test").unwrap();
        assert_eq!(result["ok"], true);
    }

    #[test]
    fn test_parse_body_json_with_charset() {
        let body = r#"{"jsonrpc":"2.0","result":{"ok":true},"id":1}"#;
        let result = parse_body(body, "application/json; charset=utf-8", "test").unwrap();
        assert_eq!(result["ok"], true);
    }

    #[test]
    fn test_parse_body_sse_content_type() {
        let body = "data: {\"jsonrpc\":\"2.0\",\"result\":{\"tools\":[]},\"id\":1}\n\n";
        let result = parse_body(body, "text/event-stream", "tools/list").unwrap();
        assert!(result.get("tools").is_some());
    }

    #[test]
    fn test_parse_body_unknown_content_type_falls_back_to_json() {
        let body = r#"{"jsonrpc":"2.0","result":{"fallback":true},"id":1}"#;
        let result = parse_body(body, "text/plain", "test").unwrap();
        assert_eq!(result["fallback"], true);
    }

    #[test]
    fn test_parse_body_unknown_content_type_falls_back_to_sse() {
        let body = "data: {\"jsonrpc\":\"2.0\",\"result\":{\"sse_fallback\":true},\"id\":1}\n\n";
        let result = parse_body(body, "text/plain", "test").unwrap();
        assert_eq!(result["sse_fallback"], true);
    }

    #[test]
    fn test_parse_body_empty_content_type_falls_back() {
        let body = r#"{"jsonrpc":"2.0","result":{"empty_ct":true},"id":1}"#;
        let result = parse_body(body, "", "test").unwrap();
        assert_eq!(result["empty_ct"], true);
    }

    #[test]
    fn test_parse_body_json_error_response() {
        let body = r#"{"jsonrpc":"2.0","error":{"code":-32600,"message":"Bad Request"},"id":1}"#;
        let err = parse_body(body, "application/json", "test").unwrap_err();
        assert!(err.to_string().contains("-32600"));
    }

    #[test]
    fn test_parse_body_invalid_json_in_json_content_type() {
        let body = "not valid json at all";
        let err = parse_body(body, "application/json", "test").unwrap_err();
        assert!(err.to_string().contains("Failed to parse JSON response"));
    }

    // ========================================================================
    // StreamableHttpMcpClient construction & effective_url
    // ========================================================================

    #[test]
    fn test_streamable_http_client_strips_trailing_slash() {
        let client = StreamableHttpMcpClient::new("https://example.com/mcp/", HashMap::new());
        assert_eq!(client.url, "https://example.com/mcp");
    }

    #[test]
    fn test_streamable_http_client_initial_state() {
        let client = StreamableHttpMcpClient::new("https://example.com/mcp", HashMap::new());
        // session_id and fallback_message_url start as None
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            assert!(client.session_id.lock().await.is_none());
            assert!(client.fallback_message_url.lock().await.is_none());
        });
    }

    #[tokio::test]
    async fn test_effective_url_without_fallback() {
        let client = StreamableHttpMcpClient::new("https://example.com/mcp", HashMap::new());
        assert_eq!(client.effective_url().await, "https://example.com/mcp");
    }

    #[tokio::test]
    async fn test_effective_url_with_fallback() {
        let client = StreamableHttpMcpClient::new("https://example.com/mcp", HashMap::new());
        *client.fallback_message_url.lock().await =
            Some("https://example.com/legacy/message".to_string());
        assert_eq!(
            client.effective_url().await,
            "https://example.com/legacy/message"
        );
    }

    #[tokio::test]
    async fn test_build_post_includes_session_id() {
        let client = StreamableHttpMcpClient::new("https://example.com/mcp", HashMap::new());
        *client.session_id.lock().await = Some("test-session-123".to_string());

        let body = serde_json::json!({"test": true});
        let request = client.build_post(&body).await.build().unwrap();

        assert_eq!(
            request.headers().get("Mcp-Session-Id").unwrap(),
            "test-session-123"
        );
        assert!(request
            .headers()
            .get("Accept")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/event-stream"));
    }

    #[tokio::test]
    async fn test_build_post_without_session_id() {
        let client = StreamableHttpMcpClient::new("https://example.com/mcp", HashMap::new());

        let body = serde_json::json!({"test": true});
        let request = client.build_post(&body).await.build().unwrap();

        assert!(request.headers().get("Mcp-Session-Id").is_none());
    }

    #[tokio::test]
    async fn test_build_post_includes_custom_headers() {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer tok123".to_string());
        let client = StreamableHttpMcpClient::new("https://example.com/mcp", headers);

        let body = serde_json::json!({"test": true});
        let request = client.build_post(&body).await.build().unwrap();

        assert_eq!(
            request.headers().get("Authorization").unwrap(),
            "Bearer tok123"
        );
    }

    // ========================================================================
    // Wiremock-based async tests (post_rpc, initialize, shutdown, etc.)
    // ========================================================================

    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn json_rpc_ok(result: Value) -> String {
        serde_json::json!({"jsonrpc":"2.0","result":result,"id":1}).to_string()
    }

    fn json_rpc_error(code: i64, message: &str) -> String {
        serde_json::json!({"jsonrpc":"2.0","error":{"code":code,"message":message},"id":1})
            .to_string()
    }

    fn init_result_json() -> Value {
        serde_json::json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {"listChanged": true}},
            "serverInfo": {"name": "mock-server", "version": "1.0"}
        })
    }

    // ── post_rpc ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_post_rpc_json_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                json_rpc_ok(serde_json::json!({"ok":true})),
                "application/json",
            ))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let result = client.post_rpc(make_request("test", None)).await.unwrap();
        assert_eq!(result["ok"], true);
    }

    #[tokio::test]
    async fn test_post_rpc_sse_response() {
        let server = MockServer::start().await;
        let sse_body = format!("data: {}\n\n", json_rpc_ok(serde_json::json!({"sse":true})));
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let result = client.post_rpc(make_request("test", None)).await.unwrap();
        assert_eq!(result["sse"], true);
    }

    #[tokio::test]
    async fn test_post_rpc_http_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let err = client
            .post_rpc(make_request("test", None))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("HTTP 500"));
    }

    #[tokio::test]
    async fn test_post_rpc_404_with_session_clears_session() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        *client.session_id.lock().await = Some("old-session".to_string());

        let err = client
            .post_rpc(make_request("test", None))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("session expired"));
        assert!(client.session_id.lock().await.is_none());
    }

    #[tokio::test]
    async fn test_post_rpc_404_without_session_is_normal_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(404).set_body_string("Not Found"))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let err = client
            .post_rpc(make_request("test", None))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("HTTP 404"));
        assert!(!err.to_string().contains("session expired"));
    }

    // ── post_rpc_with_headers ───────────────────────────────────────────

    #[tokio::test]
    async fn test_post_rpc_with_headers_returns_headers() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(json_rpc_ok(serde_json::json!({"v":1})), "application/json")
                    .append_header("X-Custom", "hello"),
            )
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let (result, headers) = client
            .post_rpc_with_headers(make_request("test", None))
            .await
            .unwrap();
        assert_eq!(result["v"], 1);
        assert_eq!(headers.get("X-Custom").unwrap(), "hello");
    }

    #[tokio::test]
    async fn test_post_rpc_with_headers_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503).set_body_string("Unavailable"))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let err = client
            .post_rpc_with_headers(make_request("test", None))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("HTTP 503"));
    }

    // ── initialize ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_initialize_captures_session_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(json_rpc_ok(init_result_json()), "application/json")
                    .append_header("mcp-session-id", "sess-abc-123"),
            )
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let init = client.initialize().await.unwrap();
        assert_eq!(init.protocol_version, "2025-03-26");
        assert_eq!(init.server_info.unwrap().name, "mock-server");
        assert_eq!(
            *client.session_id.lock().await,
            Some("sess-abc-123".to_string())
        );
    }

    #[tokio::test]
    async fn test_initialize_without_session_id_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(json_rpc_ok(init_result_json()), "application/json"),
            )
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let init = client.initialize().await.unwrap();
        assert_eq!(init.protocol_version, "2025-03-26");
        assert!(client.session_id.lock().await.is_none());
    }

    #[tokio::test]
    async fn test_initialize_sse_response() {
        let server = MockServer::start().await;
        let sse_body = format!("data: {}\n\n", json_rpc_ok(init_result_json()));
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let init = client.initialize().await.unwrap();
        assert_eq!(init.protocol_version, "2025-03-26");
    }

    #[tokio::test]
    async fn test_initialize_non_4xx_error_propagates() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let err = client.initialize().await.unwrap_err();
        assert!(err.to_string().contains("HTTP 500"));
    }

    // ── initialize with legacy SSE fallback ─────────────────────────────

    #[tokio::test]
    async fn test_initialize_fallback_on_405() {
        let server = MockServer::start().await;

        // First POST to / returns 405
        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(405).set_body_string("Method Not Allowed"))
            .expect(1)
            .mount(&server)
            .await;

        // GET /sse returns a data: line with relative message endpoint
        Mock::given(method("GET"))
            .and(path("/sse"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw("data: /message\n\n", "text/event-stream"),
            )
            .mount(&server)
            .await;

        // POST /message returns initialize result
        Mock::given(method("POST"))
            .and(path("/message"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(json_rpc_ok(init_result_json()), "application/json"),
            )
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let init = client.initialize().await.unwrap();
        assert_eq!(init.protocol_version, "2025-03-26");
        // Fallback URL should be set
        assert!(client.fallback_message_url.lock().await.is_some());
    }

    #[tokio::test]
    async fn test_initialize_fallback_on_404() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(404).set_body_string("Not Found"))
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/sse"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                format!("data: {}/message\n\n", server.uri()),
                "text/event-stream",
            ))
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/message"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(json_rpc_ok(init_result_json()), "application/json"),
            )
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let init = client.initialize().await.unwrap();
        assert_eq!(init.server_info.unwrap().name, "mock-server");
    }

    #[tokio::test]
    async fn test_legacy_fallback_sse_init_fails() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(405))
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/sse"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw("data: /message\n\n", "text/event-stream"),
            )
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/message"))
            .respond_with(ResponseTemplate::new(502).set_body_string("Bad Gateway"))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let err = client.initialize().await.unwrap_err();
        assert!(err.to_string().contains("502"));
    }

    #[tokio::test]
    async fn test_legacy_fallback_no_url_in_sse_uses_default() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(405))
            .expect(1)
            .mount(&server)
            .await;

        // SSE body with no URL data lines
        Mock::given(method("GET"))
            .and(path("/sse"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw("event: ping\n: comment\n\n", "text/event-stream"),
            )
            .mount(&server)
            .await;

        // Default fallback is /message
        Mock::given(method("POST"))
            .and(path("/message"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(json_rpc_ok(init_result_json()), "application/json"),
            )
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let init = client.initialize().await.unwrap();
        assert_eq!(init.protocol_version, "2025-03-26");
    }

    #[tokio::test]
    async fn test_legacy_fallback_with_relative_url_and_headers() {
        let server = MockServer::start().await;

        // First POST to / returns 405
        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(405))
            .expect(1)
            .mount(&server)
            .await;

        // GET /sse returns a relative URL
        Mock::given(method("GET"))
            .and(path("/sse"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw("data: /api/message\n\n", "text/event-stream"),
            )
            .mount(&server)
            .await;

        // POST to the resolved relative URL
        Mock::given(method("POST"))
            .and(path("/api/message"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(json_rpc_ok(init_result_json()), "application/json"),
            )
            .mount(&server)
            .await;

        // Use headers to cover the for loop in try_legacy_sse_fallback
        let mut headers = HashMap::new();
        headers.insert("X-Api-Key".to_string(), "test-key".to_string());
        let client = StreamableHttpMcpClient::new(&server.uri(), headers);
        let init = client.initialize().await.unwrap();
        assert_eq!(init.protocol_version, "2025-03-26");
        // Verify fallback URL was resolved from relative path
        let fallback = client.fallback_message_url.lock().await.clone().unwrap();
        assert!(fallback.contains("/api/message"));
    }

    // ── initialized_notification ────────────────────────────────────────

    #[tokio::test]
    async fn test_initialized_notification_202() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(202))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        client.initialized_notification().await.unwrap();
    }

    #[tokio::test]
    async fn test_initialized_notification_200_ok() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        client.initialized_notification().await.unwrap();
    }

    #[tokio::test]
    async fn test_initialized_notification_unexpected_status() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        // Should still return Ok — notification failures are non-fatal
        client.initialized_notification().await.unwrap();
    }

    // ── tools_list ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_tools_list() {
        let server = MockServer::start().await;
        let tools_result = serde_json::json!({
            "tools": [
                {"name": "run_query", "description": "Execute a query", "inputSchema": {}},
                {"name": "list_tables", "description": "List tables"}
            ]
        });
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(json_rpc_ok(tools_result), "application/json"),
            )
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let tools = client.tools_list().await.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "run_query");
        assert_eq!(tools[1].name, "list_tables");
    }

    #[tokio::test]
    async fn test_tools_list_empty() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(json_rpc_ok(serde_json::json!({})), "application/json"),
            )
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let tools = client.tools_list().await.unwrap();
        assert!(tools.is_empty());
    }

    // ── call_tool ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_call_tool_success() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                json_rpc_ok(serde_json::json!({"content": [{"type":"text","text":"hello"}]})),
                "application/json",
            ))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let result = client
            .call_tool("my_tool", Some(serde_json::json!({"arg": 42})))
            .await
            .unwrap();
        assert_eq!(result["content"][0]["text"], "hello");
    }

    #[tokio::test]
    async fn test_call_tool_no_args() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                json_rpc_ok(serde_json::json!({"done":true})),
                "application/json",
            ))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let result = client.call_tool("bare_tool", None).await.unwrap();
        assert_eq!(result["done"], true);
    }

    // ── ping ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_ping_success() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(json_rpc_ok(serde_json::json!({})), "application/json"),
            )
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        client.ping().await.unwrap();
    }

    #[tokio::test]
    async fn test_ping_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                json_rpc_error(-32601, "Method not found"),
                "application/json",
            ))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        let err = client.ping().await.unwrap_err();
        assert!(err.to_string().contains("-32601"));
    }

    // ── shutdown ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_shutdown_no_session() {
        let client = StreamableHttpMcpClient::new("https://example.com/mcp", HashMap::new());
        // No session → no HTTP call, just returns Ok
        client.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_shutdown_with_session_success() {
        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        *client.session_id.lock().await = Some("sess-to-delete".to_string());

        client.shutdown().await.unwrap();
        assert!(client.session_id.lock().await.is_none());
    }

    #[tokio::test]
    async fn test_shutdown_with_session_405() {
        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .respond_with(ResponseTemplate::new(405))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        *client.session_id.lock().await = Some("sess-405".to_string());

        // 405 is handled gracefully
        client.shutdown().await.unwrap();
        assert!(client.session_id.lock().await.is_none());
    }

    #[tokio::test]
    async fn test_shutdown_with_session_unexpected_status() {
        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;

        let client = StreamableHttpMcpClient::new(&server.uri(), HashMap::new());
        *client.session_id.lock().await = Some("sess-503".to_string());

        // Non-fatal — still clears session
        client.shutdown().await.unwrap();
        assert!(client.session_id.lock().await.is_none());
    }

    // ── transport_name ──────────────────────────────────────────────────

    #[test]
    fn test_streamable_http_transport_name() {
        let client = StreamableHttpMcpClient::new("https://example.com", HashMap::new());
        assert_eq!(client.transport_name(), "streamable_http");
    }

    // ── SseMcpClient — post_message SSE fallback ────────────────────────

    #[tokio::test]
    async fn test_sse_client_post_message_sse_body_fallback() {
        let server = MockServer::start().await;

        // Server responds with SSE format (not JSON) to the message endpoint
        let sse_body = format!(
            "data: {}\n\n",
            json_rpc_ok(serde_json::json!({"tools": [{"name": "sse_tool"}]}))
        );
        Mock::given(method("POST"))
            .and(path("/message"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
            .mount(&server)
            .await;

        let client = SseMcpClient::new(&server.uri(), HashMap::new());
        // post_message tries JSON parse first, fails, then falls back to parse_sse_body
        let result = client
            .post_message(make_request("tools/list", None))
            .await
            .unwrap();
        assert!(result.get("tools").is_some());
    }
}
