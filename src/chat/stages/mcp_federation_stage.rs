//! MCP Federation Enrichment Stage for the Chat Pipeline.
//!
//! Injects available external MCP tools into the system prompt so the agent
//! knows which external tools are available and how to call them.
//!
//! **Zero-cost when no MCP servers are connected**: the stage checks the registry
//! and returns `StageOutput::new()` immediately if it's empty, adding no latency
//! or content to the prompt.
//!
//! Controlled by `ENRICHMENT_MCP_FEDERATION` env var (default: true).

use anyhow::Result;
use tracing::debug;

use crate::chat::enrichment::{
    EnrichmentConfig, EnrichmentInput, EnrichmentSource, ParallelEnrichmentStage, StageOutput,
};
use crate::mcp_federation::registry::SharedRegistry;

/// Enrichment stage that injects external MCP tool availability into the prompt.
pub struct McpFederationStage {
    registry: SharedRegistry,
}

impl McpFederationStage {
    /// Create a new MCP federation stage.
    pub fn new(registry: SharedRegistry) -> Self {
        Self { registry }
    }
}

#[async_trait::async_trait]
impl ParallelEnrichmentStage for McpFederationStage {
    async fn execute(&self, _input: &EnrichmentInput) -> Result<StageOutput> {
        let mut output = StageOutput::new(self.name());

        // Read lock — non-blocking if no writers
        let reg = self.registry.read().await;

        // Fast path: no servers connected → zero overhead
        if reg.is_empty() {
            debug!("[mcp_federation] No external servers connected — skipping");
            return Ok(output);
        }

        let servers = reg.list();
        let mut sections = Vec::new();

        for server in &servers {
            // Skip disconnected servers
            if server.status != crate::mcp_federation::registry::ConnectionStatus::Connected {
                continue;
            }

            let tools = reg.tools_for_server(&server.id);
            if tools.is_empty() {
                continue;
            }

            let mut tool_lines = Vec::new();
            for tool in &tools {
                // Format: `server_id::tool_name` — description
                let desc = if tool.description.len() > 120 {
                    format!("{}…", &tool.description[..117])
                } else {
                    tool.description.clone()
                };
                tool_lines.push(format!("- `{}::{}` — {}", server.id, tool.name, desc));
            }

            let display = server
                .server_name
                .as_deref()
                .unwrap_or(&server.display_name);

            sections.push(format!(
                "### {} ({} tools)\n{}",
                display,
                tools.len(),
                tool_lines.join("\n")
            ));
        }

        if sections.is_empty() {
            return Ok(output);
        }

        // Build the prompt section
        let header = format!(
                "## External MCP Servers ({} connected)\n\
             Call external tools using the `server_id::tool_name` format.\n",
                servers
                    .iter()
                    .filter(|s| s.status
                        == crate::mcp_federation::registry::ConnectionStatus::Connected)
                    .count()
            );

        let content = format!("{}\n{}", header, sections.join("\n\n"));

        debug!(
            "[mcp_federation] Injecting {} servers, {} total tools into prompt",
            servers.len(),
            servers.iter().map(|s| s.tool_count).sum::<usize>()
        );

        output.add_section(
            "MCP Federation — External Tools",
            content,
            self.name(),
            EnrichmentSource::McpFederation,
        );

        Ok(output)
    }

    fn name(&self) -> &str {
        "mcp_federation"
    }

    fn is_enabled(&self, config: &EnrichmentConfig) -> bool {
        config.mcp_federation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp_federation::circuit_breaker::CircuitBreaker;
    use crate::mcp_federation::client::McpClient;
    use crate::mcp_federation::registry::{
        ConnectionStatus, McpServerConnection, McpServerRegistry, ServerStats,
    };
    use crate::mcp_federation::McpTransport;
    use chrono::Utc;
    use serde_json::Value;
    use std::collections::HashSet;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use uuid::Uuid;

    #[derive(Debug)]
    struct DummyClient;

    #[async_trait::async_trait]
    impl McpClient for DummyClient {
        async fn initialize(
            &self,
        ) -> anyhow::Result<crate::mcp_federation::client::InitializeResult> {
            unimplemented!()
        }
        async fn initialized_notification(&self) -> anyhow::Result<()> {
            unimplemented!()
        }
        async fn tools_list(
            &self,
        ) -> anyhow::Result<Vec<crate::mcp_federation::client::McpToolDef>> {
            unimplemented!()
        }
        async fn call_tool(&self, _name: &str, _arguments: Option<Value>) -> anyhow::Result<Value> {
            unimplemented!()
        }
        async fn ping(&self) -> anyhow::Result<()> {
            Ok(())
        }
        async fn shutdown(&self) -> anyhow::Result<()> {
            Ok(())
        }
        fn transport_name(&self) -> &'static str {
            "mock"
        }
    }

    fn make_input() -> crate::chat::enrichment::EnrichmentInput {
        crate::chat::enrichment::EnrichmentInput {
            message: "test".to_string(),
            session_id: Uuid::new_v4(),
            project_slug: None,
            project_id: None,
            cwd: None,
            protocol_run_id: None,
            protocol_state: None,
            excluded_note_ids: HashSet::new(),
            reasoning_path_tracker: None,
        }
    }

    fn make_discovered_tool(
        name: &str,
        server_id: &str,
    ) -> crate::mcp_federation::discovery::DiscoveredTool {
        crate::mcp_federation::discovery::DiscoveredTool {
            name: name.to_string(),
            fqn: format!("{}::{}", server_id, name),
            description: format!("{} tool description", name),
            input_schema: serde_json::json!({"type": "object"}),
            category: crate::mcp_federation::discovery::InferredCategory::Query,
            embedding: None,
            similar_internal: vec![],
            profile: None,
        }
    }

    #[tokio::test]
    async fn test_empty_registry_zero_overhead() {
        let registry = Arc::new(RwLock::new(McpServerRegistry::new()));
        let stage = McpFederationStage::new(registry);

        let output = stage.execute(&make_input()).await.unwrap();
        assert!(output.sections.is_empty());
    }

    #[tokio::test]
    async fn test_connected_server_produces_section() {
        let mut reg = McpServerRegistry::new();
        let conn = McpServerConnection {
            id: "grafeo".to_string(),
            display_name: "Grafeo".to_string(),
            transport: McpTransport::Sse {
                url: "http://mock:8080/sse".to_string(),
                headers: Default::default(),
            },
            status: ConnectionStatus::Connected,
            client: Box::new(DummyClient),
            discovered_tools: vec![
                make_discovered_tool("query", "grafeo"),
                make_discovered_tool("mutate", "grafeo"),
            ],
            circuit_breaker: CircuitBreaker::new(),
            stats: ServerStats::new(),
            connected_at: Utc::now(),
            server_protocol_version: Some("2025-03-26".to_string()),
            server_name: Some("Grafeo Knowledge Graph".to_string()),
        };
        reg.insert_connection_for_test(conn);

        let registry = Arc::new(RwLock::new(reg));
        let stage = McpFederationStage::new(registry);

        let output = stage.execute(&make_input()).await.unwrap();
        assert_eq!(output.sections.len(), 1);

        let content = &output.sections[0].content;
        assert!(content.contains("grafeo::query"));
        assert!(content.contains("grafeo::mutate"));
        assert!(content.contains("Grafeo Knowledge Graph"));
        assert!(content.contains("2 tools"));
    }

    #[tokio::test]
    async fn test_is_enabled_respects_config() {
        let registry = Arc::new(RwLock::new(McpServerRegistry::new()));
        let stage = McpFederationStage::new(registry);

        let mut config = EnrichmentConfig::default();
        assert!(stage.is_enabled(&config));

        config.mcp_federation = false;
        assert!(!stage.is_enabled(&config));
    }

    #[tokio::test]
    async fn test_disconnected_server_skipped() {
        let mut reg = McpServerRegistry::new();
        let conn = McpServerConnection {
            id: "offline".to_string(),
            display_name: "Offline Server".to_string(),
            transport: McpTransport::Sse {
                url: "http://mock:8080/sse".to_string(),
                headers: Default::default(),
            },
            status: ConnectionStatus::Disconnected,
            client: Box::new(DummyClient),
            discovered_tools: vec![make_discovered_tool("query", "offline")],
            circuit_breaker: CircuitBreaker::new(),
            stats: ServerStats::new(),
            connected_at: Utc::now(),
            server_protocol_version: None,
            server_name: None,
        };
        reg.insert_connection_for_test(conn);

        let registry = Arc::new(RwLock::new(reg));
        let stage = McpFederationStage::new(registry);

        let output = stage.execute(&make_input()).await.unwrap();
        // Disconnected server should produce no sections
        assert!(output.sections.is_empty());
    }

    #[tokio::test]
    async fn test_connected_server_without_tools_skipped() {
        let mut reg = McpServerRegistry::new();
        let conn = McpServerConnection {
            id: "empty-srv".to_string(),
            display_name: "Empty".to_string(),
            transport: McpTransport::Sse {
                url: "http://mock:8080/sse".to_string(),
                headers: Default::default(),
            },
            status: ConnectionStatus::Connected,
            client: Box::new(DummyClient),
            discovered_tools: vec![], // No tools
            circuit_breaker: CircuitBreaker::new(),
            stats: ServerStats::new(),
            connected_at: Utc::now(),
            server_protocol_version: None,
            server_name: None,
        };
        reg.insert_connection_for_test(conn);

        let registry = Arc::new(RwLock::new(reg));
        let stage = McpFederationStage::new(registry);

        let output = stage.execute(&make_input()).await.unwrap();
        assert!(output.sections.is_empty());
    }

    #[tokio::test]
    async fn test_multiple_servers_produces_single_section() {
        let mut reg = McpServerRegistry::new();

        // Server 1
        let conn1 = McpServerConnection {
            id: "srv1".to_string(),
            display_name: "Server One".to_string(),
            transport: McpTransport::Sse {
                url: "http://srv1:8080/sse".to_string(),
                headers: Default::default(),
            },
            status: ConnectionStatus::Connected,
            client: Box::new(DummyClient),
            discovered_tools: vec![make_discovered_tool("list", "srv1")],
            circuit_breaker: CircuitBreaker::new(),
            stats: ServerStats::new(),
            connected_at: Utc::now(),
            server_protocol_version: None,
            server_name: Some("Server One".to_string()),
        };
        reg.insert_connection_for_test(conn1);

        // Server 2
        let conn2 = McpServerConnection {
            id: "srv2".to_string(),
            display_name: "Server Two".to_string(),
            transport: McpTransport::StreamableHttp {
                url: "http://srv2:9090/mcp".to_string(),
                headers: Default::default(),
            },
            status: ConnectionStatus::Connected,
            client: Box::new(DummyClient),
            discovered_tools: vec![
                make_discovered_tool("search", "srv2"),
                make_discovered_tool("analyze", "srv2"),
            ],
            circuit_breaker: CircuitBreaker::new(),
            stats: ServerStats::new(),
            connected_at: Utc::now(),
            server_protocol_version: None,
            server_name: Some("Server Two".to_string()),
        };
        reg.insert_connection_for_test(conn2);

        let registry = Arc::new(RwLock::new(reg));
        let stage = McpFederationStage::new(registry);

        let output = stage.execute(&make_input()).await.unwrap();
        assert_eq!(output.sections.len(), 1);

        let content = &output.sections[0].content;
        assert!(content.contains("srv1::list"));
        assert!(content.contains("srv2::search"));
        assert!(content.contains("srv2::analyze"));
        assert!(content.contains("Server One"));
        assert!(content.contains("Server Two"));
        // Header should show count of connected servers
        assert!(content.contains("2 connected"));
    }

    #[tokio::test]
    async fn test_long_description_truncated() {
        let mut reg = McpServerRegistry::new();

        let long_desc = "A".repeat(200);
        let tool = crate::mcp_federation::discovery::DiscoveredTool {
            name: "verbose_tool".to_string(),
            fqn: "srv::verbose_tool".to_string(),
            description: long_desc,
            input_schema: serde_json::json!({"type": "object"}),
            category: crate::mcp_federation::discovery::InferredCategory::Query,
            embedding: None,
            similar_internal: vec![],
            profile: None,
        };

        let conn = McpServerConnection {
            id: "srv".to_string(),
            display_name: "Srv".to_string(),
            transport: McpTransport::Sse {
                url: "http://mock:8080/sse".to_string(),
                headers: Default::default(),
            },
            status: ConnectionStatus::Connected,
            client: Box::new(DummyClient),
            discovered_tools: vec![tool],
            circuit_breaker: CircuitBreaker::new(),
            stats: ServerStats::new(),
            connected_at: Utc::now(),
            server_protocol_version: None,
            server_name: None,
        };
        reg.insert_connection_for_test(conn);

        let registry = Arc::new(RwLock::new(reg));
        let stage = McpFederationStage::new(registry);

        let output = stage.execute(&make_input()).await.unwrap();
        let content = &output.sections[0].content;
        // Description should be truncated to 117 chars + "…"
        assert!(content.contains("…"));
        // Full 200-char description should NOT appear
        assert!(!content.contains(&"A".repeat(200)));
    }

    #[tokio::test]
    async fn test_display_name_fallback_to_display_name() {
        let mut reg = McpServerRegistry::new();
        let conn = McpServerConnection {
            id: "srv".to_string(),
            display_name: "My Display Name".to_string(),
            transport: McpTransport::Sse {
                url: "http://mock/sse".to_string(),
                headers: Default::default(),
            },
            status: ConnectionStatus::Connected,
            client: Box::new(DummyClient),
            discovered_tools: vec![make_discovered_tool("tool", "srv")],
            circuit_breaker: CircuitBreaker::new(),
            stats: ServerStats::new(),
            connected_at: Utc::now(),
            server_protocol_version: None,
            server_name: None, // No server_name → should use display_name
        };
        reg.insert_connection_for_test(conn);

        let registry = Arc::new(RwLock::new(reg));
        let stage = McpFederationStage::new(registry);

        let output = stage.execute(&make_input()).await.unwrap();
        let content = &output.sections[0].content;
        assert!(content.contains("My Display Name"));
    }

    #[test]
    fn test_stage_name() {
        let registry = Arc::new(RwLock::new(McpServerRegistry::new()));
        let stage = McpFederationStage::new(registry);
        assert_eq!(stage.name(), "mcp_federation");
    }

    #[tokio::test]
    async fn test_mixed_connected_and_disconnected() {
        let mut reg = McpServerRegistry::new();

        // Connected server
        let conn1 = McpServerConnection {
            id: "online".to_string(),
            display_name: "Online".to_string(),
            transport: McpTransport::Sse {
                url: "http://mock/sse".to_string(),
                headers: Default::default(),
            },
            status: ConnectionStatus::Connected,
            client: Box::new(DummyClient),
            discovered_tools: vec![make_discovered_tool("tool1", "online")],
            circuit_breaker: CircuitBreaker::new(),
            stats: ServerStats::new(),
            connected_at: Utc::now(),
            server_protocol_version: None,
            server_name: None,
        };
        reg.insert_connection_for_test(conn1);

        // Disconnected server
        let conn2 = McpServerConnection {
            id: "offline".to_string(),
            display_name: "Offline".to_string(),
            transport: McpTransport::Sse {
                url: "http://mock/sse".to_string(),
                headers: Default::default(),
            },
            status: ConnectionStatus::Disconnected,
            client: Box::new(DummyClient),
            discovered_tools: vec![make_discovered_tool("tool2", "offline")],
            circuit_breaker: CircuitBreaker::new(),
            stats: ServerStats::new(),
            connected_at: Utc::now(),
            server_protocol_version: None,
            server_name: None,
        };
        reg.insert_connection_for_test(conn2);

        let registry = Arc::new(RwLock::new(reg));
        let stage = McpFederationStage::new(registry);

        let output = stage.execute(&make_input()).await.unwrap();
        assert_eq!(output.sections.len(), 1);

        let content = &output.sections[0].content;
        // Should only contain online server tools
        assert!(content.contains("online::tool1"));
        assert!(!content.contains("offline::tool2"));
    }
}
