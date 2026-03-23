//! MCP Server Registry — manages connections to external MCP servers.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::circuit_breaker::{CircuitBreaker, CircuitState};
use super::client::{create_client, McpClient, McpTransportConfig};
use super::discovery::{
    DiscoveredTool, InternalToolDescriptor, IntrospectorConfig, ToolIntrospector,
};
use super::prober::{ProberConfig, ToolProber};
use super::{ExternalToolInfo, McpTransport, ServerId, ToolFqn};
use crate::embeddings::EmbeddingProvider;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Connection status of an external MCP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionStatus {
    Connected,
    Disconnected,
    Error,
    Reconnecting,
}

/// Statistics for an external MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStats {
    pub call_count: u64,
    pub error_count: u64,
    pub latency_samples: Vec<u64>,
    pub last_call_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
}

impl ServerStats {
    pub fn new() -> Self {
        Self {
            call_count: 0,
            error_count: 0,
            latency_samples: Vec::new(),
            last_call_at: None,
            last_error: None,
        }
    }

    /// Record a call outcome.
    pub fn record_call(&mut self, latency_ms: u64, success: bool) {
        self.call_count += 1;
        self.last_call_at = Some(Utc::now());

        // Keep last 100 latency samples
        if self.latency_samples.len() >= 100 {
            self.latency_samples.remove(0);
        }
        self.latency_samples.push(latency_ms);

        if !success {
            self.error_count += 1;
        }
    }

    /// Record an error message.
    pub fn record_error(&mut self, error: String) {
        self.last_error = Some(error);
    }

    /// Get p50 latency in ms.
    pub fn latency_p50(&self) -> u64 {
        percentile(&self.latency_samples, 50)
    }

    /// Get p95 latency in ms.
    pub fn latency_p95(&self) -> u64 {
        percentile(&self.latency_samples, 95)
    }

    /// Get error rate (0.0–1.0).
    pub fn error_rate(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.error_count as f64 / self.call_count as f64
        }
    }
}

impl Default for ServerStats {
    fn default() -> Self {
        Self::new()
    }
}

fn percentile(samples: &[u64], pct: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let idx = (sorted.len() * pct / 100).min(sorted.len() - 1);
    sorted[idx]
}

/// A connected MCP server with its client, tools, and stats.
pub struct McpServerConnection {
    pub id: ServerId,
    pub display_name: String,
    pub transport: McpTransport,
    pub status: ConnectionStatus,
    pub client: Box<dyn McpClient>,
    /// Discovered tools with introspection data (category, embedding, similar_internal).
    pub discovered_tools: Vec<DiscoveredTool>,
    pub circuit_breaker: CircuitBreaker,
    pub stats: ServerStats,
    pub connected_at: DateTime<Utc>,
    pub server_protocol_version: Option<String>,
    pub server_name: Option<String>,
}

// Manual Debug impl because Box<dyn McpClient> is Debug but we want nicer output
impl std::fmt::Debug for McpServerConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServerConnection")
            .field("id", &self.id)
            .field("display_name", &self.display_name)
            .field("status", &self.status)
            .field("tools_count", &self.discovered_tools.len())
            .field("transport", &self.client.transport_name())
            .finish()
    }
}

/// Summary of a connected server (serializable, no client reference).
#[derive(Debug, Clone, Serialize)]
pub struct ServerSummary {
    pub id: ServerId,
    pub display_name: String,
    pub status: ConnectionStatus,
    pub transport_type: String,
    pub tool_count: usize,
    pub connected_at: DateTime<Utc>,
    pub stats: ServerStats,
    pub circuit_breaker_state: CircuitState,
    pub server_name: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Registry
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the registry's discovery and probing behavior.
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Whether to probe read-only tools on connect.
    pub probe_on_connect: bool,
    /// Introspector configuration.
    pub introspector: IntrospectorConfig,
    /// Prober configuration.
    pub prober: ProberConfig,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            probe_on_connect: true,
            introspector: IntrospectorConfig::default(),
            prober: ProberConfig::default(),
        }
    }
}

/// Thread-safe registry of connected MCP servers.
pub struct McpServerRegistry {
    servers: HashMap<ServerId, McpServerConnection>,
    /// Index: tool FQN → server ID for O(1) dispatch.
    tool_index: HashMap<ToolFqn, ServerId>,
    /// Embedding provider for tool introspection (None = skip embeddings).
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    /// Pre-computed descriptors for our internal MCP tools.
    internal_tools: Vec<InternalToolDescriptor>,
    /// Configuration.
    config: RegistryConfig,
}

impl std::fmt::Debug for McpServerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServerRegistry")
            .field("server_count", &self.servers.len())
            .field("tool_count", &self.tool_index.len())
            .field("has_embedding_provider", &self.embedding_provider.is_some())
            .field("internal_tools_count", &self.internal_tools.len())
            .finish()
    }
}

impl Default for McpServerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl McpServerRegistry {
    pub fn new() -> Self {
        Self {
            servers: HashMap::new(),
            tool_index: HashMap::new(),
            embedding_provider: None,
            internal_tools: vec![],
            config: RegistryConfig::default(),
        }
    }

    /// Create a registry with full introspection capabilities.
    pub fn with_introspection(
        embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
        internal_tools: Vec<InternalToolDescriptor>,
        config: RegistryConfig,
    ) -> Self {
        Self {
            servers: HashMap::new(),
            tool_index: HashMap::new(),
            embedding_provider,
            internal_tools,
            config,
        }
    }

    /// Connect to an external MCP server.
    ///
    /// Flow: create transport → initialize → tools/list → introspect → probe → store.
    pub async fn connect(&mut self, config: McpTransportConfig) -> Result<ServerSummary> {
        let server_id = config.server_id.clone();

        if self.servers.contains_key(&server_id) {
            return Err(anyhow!(
                "MCP server '{}' is already connected. Disconnect first.",
                server_id
            ));
        }

        info!(server_id = %server_id, "Connecting to MCP server");

        // 1. Create the client
        let client = create_client(&config.transport).await?;
        debug!(
            server_id = %server_id,
            transport = client.transport_name(),
            "MCP client created"
        );

        // 2. Initialize handshake
        let init_result =
            tokio::time::timeout(std::time::Duration::from_secs(30), client.initialize())
                .await
                .map_err(|_| anyhow!("Initialize handshake timed out (30s)"))?
                .map_err(|e| anyhow!("Initialize handshake failed: {}", e))?;

        debug!(
            server_id = %server_id,
            protocol_version = %init_result.protocol_version,
            server_info = ?init_result.server_info,
            "MCP initialize handshake completed"
        );

        // 3. Send initialized notification
        if let Err(e) = client.initialized_notification().await {
            debug!(error = %e, "Failed to send initialized notification (non-fatal)");
        }

        // 4. List available tools
        let tool_defs =
            tokio::time::timeout(std::time::Duration::from_secs(15), client.tools_list())
                .await
                .map_err(|_| anyhow!("tools/list timed out (15s)"))?
                .map_err(|e| anyhow!("tools/list failed: {}", e))?;

        info!(
            server_id = %server_id,
            tool_count = tool_defs.len(),
            "Discovered tools from MCP server"
        );

        // 5. Introspect tools (classify + embed + find similar)
        let introspector = ToolIntrospector::new(
            self.embedding_provider.clone(),
            self.internal_tools.clone(),
            self.config.introspector.clone(),
        );
        let mut discovered_tools = introspector.introspect(&server_id, &tool_defs).await;

        info!(
            server_id = %server_id,
            tool_count = discovered_tools.len(),
            "Introspected tools"
        );

        // 6. Probe read-only tools (optional)
        if self.config.probe_on_connect {
            let prober = ToolProber::new(self.config.prober.clone());
            prober
                .probe_batch(client.as_ref(), &mut discovered_tools)
                .await;

            let probed_count = discovered_tools
                .iter()
                .filter(|t| t.profile.is_some())
                .count();
            info!(
                server_id = %server_id,
                probed_count = probed_count,
                "Probed read-only tools"
            );
        }

        // 7. Build the tool index
        for tool in &discovered_tools {
            self.tool_index.insert(tool.fqn.clone(), server_id.clone());
        }

        let display_name = config.display_name.unwrap_or_else(|| server_id.clone());

        let server_name = init_result.server_info.as_ref().map(|si| si.name.clone());

        let connection = McpServerConnection {
            id: server_id.clone(),
            display_name: display_name.clone(),
            transport: config.transport,
            status: ConnectionStatus::Connected,
            client,
            discovered_tools,
            circuit_breaker: CircuitBreaker::new(),
            stats: ServerStats::new(),
            connected_at: Utc::now(),
            server_protocol_version: Some(init_result.protocol_version),
            server_name: server_name.clone(),
        };

        let summary = ServerSummary {
            id: server_id.clone(),
            display_name,
            status: ConnectionStatus::Connected,
            transport_type: connection.client.transport_name().to_string(),
            tool_count: connection.discovered_tools.len(),
            connected_at: connection.connected_at,
            stats: connection.stats.clone(),
            circuit_breaker_state: CircuitState::Closed,
            server_name,
        };

        self.servers.insert(server_id, connection);

        Ok(summary)
    }

    /// Disconnect from an MCP server.
    pub async fn disconnect(&mut self, server_id: &str) -> Result<()> {
        let connection = self
            .servers
            .remove(server_id)
            .ok_or_else(|| anyhow!("MCP server '{}' is not connected", server_id))?;

        info!(server_id = %server_id, "Disconnecting MCP server");

        // Remove from tool index
        for tool in &connection.discovered_tools {
            self.tool_index.remove(&tool.fqn);
        }

        // Graceful shutdown
        if let Err(e) = connection.client.shutdown().await {
            warn!(
                server_id = %server_id,
                error = %e,
                "Error during MCP server shutdown (non-fatal)"
            );
        }

        Ok(())
    }

    /// Get a reference to a connected server.
    pub fn get(&self, server_id: &str) -> Option<&McpServerConnection> {
        self.servers.get(server_id)
    }

    /// Get a mutable reference to a connected server.
    pub fn get_mut(&mut self, server_id: &str) -> Option<&mut McpServerConnection> {
        self.servers.get_mut(server_id)
    }

    /// Look up which server owns a tool FQN.
    pub fn resolve_tool(&self, fqn: &str) -> Option<&str> {
        self.tool_index.get(fqn).map(|s| s.as_str())
    }

    /// List all connected servers.
    pub fn list(&self) -> Vec<ServerSummary> {
        self.servers
            .values()
            .map(|conn| ServerSummary {
                id: conn.id.clone(),
                display_name: conn.display_name.clone(),
                status: conn.status,
                transport_type: conn.client.transport_name().to_string(),
                tool_count: conn.discovered_tools.len(),
                connected_at: conn.connected_at,
                stats: conn.stats.clone(),
                circuit_breaker_state: conn.circuit_breaker.state(),
                server_name: conn.server_name.clone(),
            })
            .collect()
    }

    /// List all discovered tools from all connected servers.
    pub fn all_tools(&self) -> Vec<&DiscoveredTool> {
        self.servers
            .values()
            .flat_map(|conn| conn.discovered_tools.iter())
            .collect()
    }

    /// Get all discovered tools for a specific server.
    pub fn tools_for_server(&self, server_id: &str) -> Vec<&DiscoveredTool> {
        self.servers
            .get(server_id)
            .map(|conn| conn.discovered_tools.iter().collect())
            .unwrap_or_default()
    }

    /// Convert discovered tools to legacy ExternalToolInfo (for backward compatibility).
    pub fn all_external_tool_infos(&self) -> Vec<ExternalToolInfo> {
        self.servers
            .values()
            .flat_map(|conn| {
                conn.discovered_tools.iter().map(|dt| ExternalToolInfo {
                    name: dt.name.clone(),
                    fqn: dt.fqn.clone(),
                    description: dt.description.clone(),
                    input_schema: dt.input_schema.clone(),
                })
            })
            .collect()
    }

    /// Check if any servers are connected.
    pub fn is_empty(&self) -> bool {
        self.servers.is_empty()
    }

    /// Number of connected servers.
    pub fn len(&self) -> usize {
        self.servers.len()
    }

    /// Get all connected server IDs.
    pub fn server_ids(&self) -> Vec<&str> {
        self.servers.keys().map(|s| s.as_str()).collect()
    }

    /// Insert a pre-built connection directly (for testing without handshake).
    #[cfg(test)]
    pub fn insert_connection_for_test(&mut self, conn: McpServerConnection) {
        let server_id = conn.id.clone();
        for tool in &conn.discovered_tools {
            self.tool_index.insert(tool.fqn.clone(), server_id.clone());
        }
        self.servers.insert(server_id, conn);
    }

    /// Connect using a pre-built client (for testing without real transports).
    ///
    /// Runs the full connect flow (initialize, tools_list, introspect, probe)
    /// but uses the provided client instead of creating one from a transport config.
    #[cfg(test)]
    pub async fn connect_with_client(
        &mut self,
        server_id: String,
        display_name: Option<String>,
        transport: McpTransport,
        client: Box<dyn McpClient>,
    ) -> Result<ServerSummary> {
        if self.servers.contains_key(&server_id) {
            return Err(anyhow!(
                "MCP server '{}' is already connected. Disconnect first.",
                server_id
            ));
        }

        info!(server_id = %server_id, "Connecting to MCP server (test)");

        // Initialize handshake
        let init_result =
            tokio::time::timeout(std::time::Duration::from_secs(5), client.initialize())
                .await
                .map_err(|_| anyhow!("Initialize handshake timed out (5s)"))?
                .map_err(|e| anyhow!("Initialize handshake failed: {}", e))?;

        // Send initialized notification
        if let Err(e) = client.initialized_notification().await {
            debug!(error = %e, "Failed to send initialized notification (non-fatal)");
        }

        // List available tools
        let tool_defs =
            tokio::time::timeout(std::time::Duration::from_secs(5), client.tools_list())
                .await
                .map_err(|_| anyhow!("tools/list timed out (5s)"))?
                .map_err(|e| anyhow!("tools/list failed: {}", e))?;

        // Introspect tools
        let introspector = ToolIntrospector::new(
            self.embedding_provider.clone(),
            self.internal_tools.clone(),
            self.config.introspector.clone(),
        );
        let mut discovered_tools = introspector.introspect(&server_id, &tool_defs).await;

        // Probe (optional)
        if self.config.probe_on_connect {
            let prober = ToolProber::new(self.config.prober.clone());
            prober
                .probe_batch(client.as_ref(), &mut discovered_tools)
                .await;
        }

        // Build index
        for tool in &discovered_tools {
            self.tool_index.insert(tool.fqn.clone(), server_id.clone());
        }

        let display = display_name.unwrap_or_else(|| server_id.clone());
        let server_name = init_result.server_info.as_ref().map(|si| si.name.clone());

        let connection = McpServerConnection {
            id: server_id.clone(),
            display_name: display.clone(),
            transport,
            status: ConnectionStatus::Connected,
            client,
            discovered_tools,
            circuit_breaker: CircuitBreaker::new(),
            stats: ServerStats::new(),
            connected_at: Utc::now(),
            server_protocol_version: Some(init_result.protocol_version),
            server_name: server_name.clone(),
        };

        let summary = ServerSummary {
            id: server_id.clone(),
            display_name: display,
            status: ConnectionStatus::Connected,
            transport_type: connection.client.transport_name().to_string(),
            tool_count: connection.discovered_tools.len(),
            connected_at: connection.connected_at,
            stats: connection.stats.clone(),
            circuit_breaker_state: CircuitState::Closed,
            server_name,
        };

        self.servers.insert(server_id, connection);
        Ok(summary)
    }
}

/// Thread-safe wrapper for the registry (used in AppState).
pub type SharedRegistry = Arc<RwLock<McpServerRegistry>>;

/// Create a new shared registry.
pub fn new_shared_registry() -> SharedRegistry {
    Arc::new(RwLock::new(McpServerRegistry::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_registry_is_empty() {
        let registry = McpServerRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_server_stats_new() {
        let stats = ServerStats::new();
        assert_eq!(stats.call_count, 0);
        assert_eq!(stats.error_count, 0);
        assert_eq!(stats.error_rate(), 0.0);
        assert_eq!(stats.latency_p50(), 0);
    }

    #[test]
    fn test_server_stats_recording() {
        let mut stats = ServerStats::new();
        stats.record_call(10, true);
        stats.record_call(20, true);
        stats.record_call(100, false);

        assert_eq!(stats.call_count, 3);
        assert_eq!(stats.error_count, 1);
        assert!((stats.error_rate() - 0.333).abs() < 0.01);
        assert!(stats.last_call_at.is_some());
    }

    #[test]
    fn test_latency_percentiles() {
        let mut stats = ServerStats::new();
        for ms in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] {
            stats.record_call(ms, true);
        }
        assert!(stats.latency_p50() > 0);
        assert!(stats.latency_p95() > stats.latency_p50());
    }

    #[test]
    fn test_server_summary_serialization() {
        let summary = ServerSummary {
            id: "test".to_string(),
            display_name: "Test Server".to_string(),
            status: ConnectionStatus::Connected,
            transport_type: "stdio".to_string(),
            tool_count: 5,
            connected_at: Utc::now(),
            stats: ServerStats::new(),
            circuit_breaker_state: CircuitState::Closed,
            server_name: Some("TestMCP".to_string()),
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("\"status\":\"connected\""));
        assert!(json.contains("\"tool_count\":5"));
    }

    #[test]
    fn test_connection_status_serde() {
        let status = ConnectionStatus::Connected;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"connected\"");
        let back: ConnectionStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ConnectionStatus::Connected);
    }

    #[test]
    fn test_shared_registry_creation() {
        let registry = new_shared_registry();
        // Just verify it compiles and creates
        assert!(Arc::strong_count(&registry) == 1);
    }

    // ── Mock MCP client for lifecycle tests ──────────────────────────────

    #[derive(Debug)]
    struct MockLifecycleClient {
        should_fail: bool,
        /// Tools to return from tools_list (for connect_with_client tests).
        tools: Vec<super::super::client::McpToolDef>,
    }

    impl MockLifecycleClient {
        fn new(should_fail: bool) -> Self {
            Self {
                should_fail,
                tools: vec![],
            }
        }

        fn with_tools(mut self, tools: Vec<super::super::client::McpToolDef>) -> Self {
            self.tools = tools;
            self
        }
    }

    #[async_trait::async_trait]
    impl super::super::client::McpClient for MockLifecycleClient {
        async fn initialize(&self) -> anyhow::Result<super::super::client::InitializeResult> {
            if self.should_fail {
                return Err(anyhow::anyhow!("initialize failed"));
            }
            Ok(super::super::client::InitializeResult {
                protocol_version: "2024-11-05".to_string(),
                capabilities: serde_json::json!({}),
                server_info: Some(super::super::client::ServerInfoResult {
                    name: "MockServer".to_string(),
                    version: Some("1.0.0".to_string()),
                }),
            })
        }
        async fn initialized_notification(&self) -> anyhow::Result<()> {
            Ok(())
        }
        async fn tools_list(&self) -> anyhow::Result<Vec<super::super::client::McpToolDef>> {
            if self.should_fail {
                return Err(anyhow::anyhow!("tools_list failed"));
            }
            Ok(self.tools.clone())
        }
        async fn call_tool(
            &self,
            _name: &str,
            _arguments: Option<serde_json::Value>,
        ) -> anyhow::Result<serde_json::Value> {
            if self.should_fail {
                Err(anyhow::anyhow!("simulated failure"))
            } else {
                Ok(serde_json::json!({"ok": true}))
            }
        }
        async fn ping(&self) -> anyhow::Result<()> {
            if self.should_fail {
                Err(anyhow::anyhow!("ping failed"))
            } else {
                Ok(())
            }
        }
        async fn shutdown(&self) -> anyhow::Result<()> {
            Ok(())
        }
        fn transport_name(&self) -> &'static str {
            "mock"
        }
    }

    /// Helper: build a test McpServerConnection with the mock client.
    fn mock_connection(server_id: &str, tool_count: usize) -> McpServerConnection {
        let tools: Vec<super::super::discovery::DiscoveredTool> = (0..tool_count)
            .map(|i| super::super::discovery::DiscoveredTool {
                name: format!("tool_{}", i),
                fqn: format!("{}::tool_{}", server_id, i),
                description: format!("Mock tool {}", i),
                input_schema: serde_json::json!({"type": "object"}),
                category: if i % 2 == 0 {
                    super::super::discovery::InferredCategory::Query
                } else {
                    super::super::discovery::InferredCategory::Mutation
                },
                embedding: None,
                similar_internal: vec![],
                profile: None,
            })
            .collect();

        McpServerConnection {
            id: server_id.to_string(),
            display_name: format!("Test {}", server_id),
            transport: super::super::McpTransport::Stdio {
                command: "echo".to_string(),
                args: vec![],
                env: std::collections::HashMap::new(),
            },
            status: ConnectionStatus::Connected,
            client: Box::new(MockLifecycleClient::new(false)),
            discovered_tools: tools,
            circuit_breaker: super::super::circuit_breaker::CircuitBreaker::default(),
            stats: ServerStats::new(),
            connected_at: chrono::Utc::now(),
            server_protocol_version: Some("2024-11-05".to_string()),
            server_name: Some("MockServer".to_string()),
        }
    }

    // ── Lifecycle Integration Tests ──────────────────────────────────────

    #[test]
    fn test_lifecycle_insert_list_disconnect() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut registry = McpServerRegistry::new();

            // Insert a mock connection
            let conn = mock_connection("test-server", 3);
            registry.insert_connection_for_test(conn);

            // List should return 1 server
            let servers = registry.list();
            assert_eq!(servers.len(), 1);
            assert_eq!(servers[0].id, "test-server");
            assert_eq!(servers[0].tool_count, 3);
            assert_eq!(servers[0].status, ConnectionStatus::Connected);

            // Tools should be indexed
            assert_eq!(registry.all_tools().len(), 3);
            assert!(!registry.is_empty());
            assert_eq!(registry.len(), 1);

            // Disconnect
            registry.disconnect("test-server").await.unwrap();
            assert!(registry.is_empty());
            assert!(registry.list().is_empty());
            assert!(registry.all_tools().is_empty());
        });
    }

    #[test]
    fn test_lifecycle_tools_for_server() {
        let mut registry = McpServerRegistry::new();
        let conn = mock_connection("alpha", 4);
        registry.insert_connection_for_test(conn);

        let tools = registry.tools_for_server("alpha");
        assert_eq!(tools.len(), 4);
        assert_eq!(tools[0].fqn, "alpha::tool_0");

        // Non-existent server returns empty
        assert!(registry.tools_for_server("nonexistent").is_empty());
    }

    #[test]
    fn test_lifecycle_resolve_tool_fqn() {
        let mut registry = McpServerRegistry::new();
        let conn = mock_connection("beta", 2);
        registry.insert_connection_for_test(conn);

        // Resolve existing tool FQN
        let server_id = registry.resolve_tool("beta::tool_0");
        assert_eq!(server_id, Some("beta"));

        // Non-existent tool
        assert!(registry.resolve_tool("beta::nonexistent").is_none());
        assert!(registry.resolve_tool("gamma::tool_0").is_none());
    }

    #[test]
    fn test_lifecycle_multiple_servers() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut registry = McpServerRegistry::new();

            registry.insert_connection_for_test(mock_connection("server-a", 2));
            registry.insert_connection_for_test(mock_connection("server-b", 3));

            assert_eq!(registry.len(), 2);
            assert_eq!(registry.all_tools().len(), 5); // 2 + 3

            // List returns both
            let servers = registry.list();
            assert_eq!(servers.len(), 2);

            // Disconnect one
            registry.disconnect("server-a").await.unwrap();
            assert_eq!(registry.len(), 1);
            assert_eq!(registry.all_tools().len(), 3); // only server-b's tools

            // server-b tools still accessible
            assert!(registry.resolve_tool("server-b::tool_0").is_some());
            // server-a tools gone
            assert!(registry.resolve_tool("server-a::tool_0").is_none());
        });
    }

    #[test]
    fn test_lifecycle_disconnect_nonexistent_fails() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut registry = McpServerRegistry::new();
            let result = registry.disconnect("ghost").await;
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_lifecycle_server_ids() {
        let mut registry = McpServerRegistry::new();
        registry.insert_connection_for_test(mock_connection("x", 1));
        registry.insert_connection_for_test(mock_connection("y", 1));

        let mut ids = registry.server_ids();
        ids.sort();
        assert_eq!(ids, vec!["x", "y"]);
    }

    #[test]
    fn test_lifecycle_get_and_get_mut() {
        let mut registry = McpServerRegistry::new();
        registry.insert_connection_for_test(mock_connection("srv", 2));

        // get (immutable)
        let conn = registry.get("srv").unwrap();
        assert_eq!(conn.display_name, "Test srv");
        assert_eq!(conn.discovered_tools.len(), 2);

        // get non-existent
        assert!(registry.get("nope").is_none());

        // get_mut
        let conn_mut = registry.get_mut("srv").unwrap();
        conn_mut.stats.record_call(42, true);
        assert_eq!(conn_mut.stats.call_count, 1);
    }

    #[test]
    fn test_lifecycle_security_policy_integration() {
        use super::super::security::{McpSecurityPolicy, SecurityEnforcer};

        let mut registry = McpServerRegistry::new();
        let conn = mock_connection("secure-srv", 4);
        registry.insert_connection_for_test(conn);

        // Default policy blocks mutations
        let mut enforcer = SecurityEnforcer::new(McpSecurityPolicy::default());

        let tools = registry.tools_for_server("secure-srv");
        for tool in &tools {
            let result = enforcer.enforce("secure-srv", &tool.name, &tool.category);
            if tool.category.is_mutating() {
                assert!(
                    result.is_err(),
                    "Mutation tool {} should be blocked",
                    tool.name
                );
            } else {
                assert!(result.is_ok(), "Query tool {} should pass", tool.name);
            }
        }
    }

    #[test]
    fn test_lifecycle_circuit_breaker_integration() {
        let mut registry = McpServerRegistry::new();
        registry.insert_connection_for_test(mock_connection("cb-srv", 1));

        let conn = registry.get_mut("cb-srv").unwrap();

        // Circuit breaker starts closed
        assert!(conn.circuit_breaker.allow_request());

        // Record failures to trip it
        for _ in 0..10 {
            conn.circuit_breaker.record_failure();
        }
        assert!(!conn.circuit_breaker.allow_request());

        // Record success to start recovery
        conn.circuit_breaker.record_success();
    }

    #[test]
    fn test_lifecycle_all_external_tool_infos() {
        let mut registry = McpServerRegistry::new();
        registry.insert_connection_for_test(mock_connection("ext-srv", 3));

        let infos = registry.all_external_tool_infos();
        assert_eq!(infos.len(), 3);
        assert_eq!(infos[0].fqn, "ext-srv::tool_0");
        assert_eq!(infos[0].description, "Mock tool 0");
    }

    #[test]
    fn test_connect_duplicate_server_rejected() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut registry = McpServerRegistry::new();
            // Insert a connection manually
            registry.insert_connection_for_test(mock_connection("dup-srv", 1));

            // Try to connect with the same server_id via connect()
            let config = McpTransportConfig {
                server_id: "dup-srv".to_string(),
                display_name: Some("Duplicate".to_string()),
                transport: super::super::McpTransport::StreamableHttp {
                    url: "https://example.com/mcp".to_string(),
                    headers: std::collections::HashMap::new(),
                },
            };
            let result = registry.connect(config).await;
            assert!(result.is_err());
            let err_msg = result.unwrap_err().to_string();
            assert!(
                err_msg.contains("already connected"),
                "Expected 'already connected' in error, got: {}",
                err_msg
            );
        });
    }

    #[test]
    fn test_with_introspection_constructor() {
        let registry =
            McpServerRegistry::with_introspection(None, vec![], RegistryConfig::default());
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_debug_impl() {
        let registry = McpServerRegistry::new();
        let debug_str = format!("{:?}", registry);
        assert!(
            debug_str.contains("McpServerRegistry"),
            "Debug output should contain 'McpServerRegistry', got: {}",
            debug_str
        );
        assert!(debug_str.contains("server_count"));
        assert!(debug_str.contains("tool_count"));
    }

    #[test]
    fn test_connection_debug_impl() {
        let conn = mock_connection("dbg-srv", 2);
        let debug_str = format!("{:?}", conn);
        assert!(
            debug_str.contains("McpServerConnection"),
            "Debug output should contain 'McpServerConnection', got: {}",
            debug_str
        );
        assert!(debug_str.contains("dbg-srv"));
        assert!(debug_str.contains("tools_count"));
    }

    #[test]
    fn test_registry_config_default() {
        let config = RegistryConfig::default();
        assert!(config.probe_on_connect);
    }

    #[test]
    fn test_server_stats_record_error() {
        let mut stats = ServerStats::new();
        assert!(stats.last_error.is_none());

        stats.record_error("connection timeout".to_string());
        assert_eq!(stats.last_error, Some("connection timeout".to_string()));

        // Recording another error overwrites
        stats.record_error("parse failure".to_string());
        assert_eq!(stats.last_error, Some("parse failure".to_string()));
    }

    #[test]
    fn test_server_stats_error_rate_no_calls() {
        let stats = ServerStats::new();
        assert_eq!(stats.call_count, 0);
        assert_eq!(stats.error_rate(), 0.0);
    }

    #[test]
    fn test_server_stats_latency_sample_cap() {
        let mut stats = ServerStats::new();
        // Record 150 calls — latency_samples should be capped at 100
        for i in 0..150 {
            stats.record_call(i as u64, true);
        }
        assert_eq!(stats.call_count, 150);
        assert_eq!(stats.latency_samples.len(), 100);
        // The first 50 should have been evicted; smallest remaining is 50
        assert_eq!(stats.latency_samples[0], 50);
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 50), 0);
        assert_eq!(percentile(&[], 95), 0);
        assert_eq!(percentile(&[], 0), 0);
    }

    // ── connect_with_client tests ───────────────────────────────────────

    fn mock_transport() -> super::super::McpTransport {
        super::super::McpTransport::Stdio {
            command: "echo".to_string(),
            args: vec![],
            env: std::collections::HashMap::new(),
        }
    }

    fn mock_tool_defs(n: usize) -> Vec<super::super::client::McpToolDef> {
        (0..n)
            .map(|i| super::super::client::McpToolDef {
                name: format!("get_item_{}", i),
                description: Some(format!("Query item {}", i)),
                input_schema: serde_json::json!({"type": "object"}),
            })
            .collect()
    }

    #[tokio::test]
    async fn test_connect_with_client_success() {
        let mut registry = McpServerRegistry::new();
        let client = MockLifecycleClient::new(false).with_tools(mock_tool_defs(3));

        let summary = registry
            .connect_with_client(
                "test-srv".to_string(),
                Some("Test Server".to_string()),
                mock_transport(),
                Box::new(client),
            )
            .await
            .unwrap();

        assert_eq!(summary.id, "test-srv");
        assert_eq!(summary.display_name, "Test Server");
        assert_eq!(summary.status, ConnectionStatus::Connected);
        assert_eq!(summary.tool_count, 3);
        assert_eq!(summary.server_name, Some("MockServer".to_string()));
        assert_eq!(summary.circuit_breaker_state, CircuitState::Closed);

        // Registry should have the server
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.all_tools().len(), 3);

        // Tool index should work
        assert!(registry.resolve_tool("test-srv::get_item_0").is_some());
        assert!(registry.resolve_tool("test-srv::get_item_1").is_some());
        assert!(registry.resolve_tool("test-srv::get_item_2").is_some());
    }

    #[tokio::test]
    async fn test_connect_with_client_no_display_name() {
        let mut registry = McpServerRegistry::new();
        let client = MockLifecycleClient::new(false).with_tools(mock_tool_defs(1));

        let summary = registry
            .connect_with_client(
                "auto-name".to_string(),
                None, // display_name = None → uses server_id
                mock_transport(),
                Box::new(client),
            )
            .await
            .unwrap();

        assert_eq!(summary.display_name, "auto-name");
    }

    #[tokio::test]
    async fn test_connect_with_client_zero_tools() {
        let mut registry = McpServerRegistry::new();
        let client = MockLifecycleClient::new(false).with_tools(vec![]);

        let summary = registry
            .connect_with_client(
                "empty-srv".to_string(),
                None,
                mock_transport(),
                Box::new(client),
            )
            .await
            .unwrap();

        assert_eq!(summary.tool_count, 0);
        assert!(registry.all_tools().is_empty());
    }

    #[tokio::test]
    async fn test_connect_with_client_duplicate_rejected() {
        let mut registry = McpServerRegistry::new();
        let client1 = MockLifecycleClient::new(false).with_tools(mock_tool_defs(1));
        registry
            .connect_with_client(
                "dup".to_string(),
                None,
                mock_transport(),
                Box::new(client1),
            )
            .await
            .unwrap();

        let client2 = MockLifecycleClient::new(false).with_tools(mock_tool_defs(1));
        let result = registry
            .connect_with_client(
                "dup".to_string(),
                None,
                mock_transport(),
                Box::new(client2),
            )
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already connected"));
    }

    #[tokio::test]
    async fn test_connect_with_client_initialize_fails() {
        let mut registry = McpServerRegistry::new();
        let client = MockLifecycleClient::new(true); // should_fail = true

        let result = registry
            .connect_with_client(
                "fail-srv".to_string(),
                None,
                mock_transport(),
                Box::new(client),
            )
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("initialize failed"));
        // Server should NOT be in registry
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_connect_with_client_then_disconnect() {
        let mut registry = McpServerRegistry::new();
        let client = MockLifecycleClient::new(false).with_tools(mock_tool_defs(2));

        registry
            .connect_with_client(
                "conn-disc".to_string(),
                None,
                mock_transport(),
                Box::new(client),
            )
            .await
            .unwrap();

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.all_tools().len(), 2);

        // Disconnect
        registry.disconnect("conn-disc").await.unwrap();
        assert!(registry.is_empty());
        assert!(registry.all_tools().is_empty());
        assert!(registry.resolve_tool("conn-disc::get_item_0").is_none());
    }

    #[tokio::test]
    async fn test_connect_with_client_probe_disabled() {
        let config = RegistryConfig {
            probe_on_connect: false,
            ..Default::default()
        };
        let mut registry = McpServerRegistry::with_introspection(None, vec![], config);
        let client = MockLifecycleClient::new(false).with_tools(mock_tool_defs(2));

        let summary = registry
            .connect_with_client(
                "no-probe".to_string(),
                None,
                mock_transport(),
                Box::new(client),
            )
            .await
            .unwrap();

        assert_eq!(summary.tool_count, 2);
        // All tools should have profile = None since probing was disabled
        let conn = registry.get("no-probe").unwrap();
        for tool in &conn.discovered_tools {
            assert!(
                tool.profile.is_none(),
                "Tool {} should not be probed",
                tool.name
            );
        }
    }

    #[tokio::test]
    async fn test_connect_with_client_list_summary() {
        let mut registry = McpServerRegistry::new();
        let client = MockLifecycleClient::new(false).with_tools(mock_tool_defs(3));

        registry
            .connect_with_client(
                "listed".to_string(),
                Some("Listed Server".to_string()),
                mock_transport(),
                Box::new(client),
            )
            .await
            .unwrap();

        let list = registry.list();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "listed");
        assert_eq!(list[0].display_name, "Listed Server");
        assert_eq!(list[0].tool_count, 3);
        assert_eq!(list[0].transport_type, "mock");
        assert!(list[0].server_name.is_some());
    }

    #[tokio::test]
    async fn test_connect_with_client_external_tool_infos() {
        let mut registry = McpServerRegistry::new();
        let client = MockLifecycleClient::new(false).with_tools(mock_tool_defs(2));

        registry
            .connect_with_client(
                "info-srv".to_string(),
                None,
                mock_transport(),
                Box::new(client),
            )
            .await
            .unwrap();

        let infos = registry.all_external_tool_infos();
        assert_eq!(infos.len(), 2);
        assert!(infos[0].fqn.starts_with("info-srv::"));
    }
}
