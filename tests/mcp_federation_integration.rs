//! Integration tests for MCP Federation — Streamable HTTP spec compliance.
//!
//! These tests connect to **real** MCP servers over the network.
//! They are `#[ignore]` by default to avoid CI failures.
//!
//! Run manually:
//!   cargo test --test mcp_federation_integration -- --ignored
//!
//! For GitHub (requires a PAT with Copilot scope):
//!   GITHUB_PAT=ghp_xxx cargo test --test mcp_federation_integration -- --ignored test_github

use project_orchestrator::mcp_federation::client::{
    create_client, McpClient, StreamableHttpMcpClient,
};
use project_orchestrator::mcp_federation::McpTransport;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Svelte MCP — no auth, streamable HTTP
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_svelte_mcp_connect() {
    let client = StreamableHttpMcpClient::new("https://mcp.svelte.dev/mcp", HashMap::new());

    // Step 1: Initialize handshake
    let init = client
        .initialize()
        .await
        .expect("Svelte MCP initialize should succeed");

    println!("Svelte MCP server info: {:?}", init.server_info);
    println!("Protocol version: {}", init.protocol_version);

    // The server must return a valid protocol version
    assert!(
        !init.protocol_version.is_empty(),
        "Protocol version should not be empty"
    );

    // Step 2: Send initialized notification
    client
        .initialized_notification()
        .await
        .expect("initialized notification should succeed");

    // Step 3: List tools
    let tools = client
        .tools_list()
        .await
        .expect("Svelte MCP tools/list should succeed");

    println!("Svelte MCP tools ({}):", tools.len());
    for tool in &tools {
        println!(
            "  - {} : {}",
            tool.name,
            tool.description.as_deref().unwrap_or("")
        );
    }

    assert!(
        !tools.is_empty(),
        "Svelte MCP should expose at least one tool"
    );

    // Step 4: Graceful shutdown
    client.shutdown().await.expect("shutdown should succeed");
}

#[tokio::test]
#[ignore]
async fn test_svelte_mcp_via_create_client() {
    // Test the factory function with McpTransport enum
    let transport = McpTransport::StreamableHttp {
        url: "https://mcp.svelte.dev/mcp".to_string(),
        headers: HashMap::new(),
    };

    let client = create_client(&transport)
        .await
        .expect("create_client should succeed for streamable_http");

    let init = client
        .initialize()
        .await
        .expect("initialize via create_client should succeed");

    assert!(!init.protocol_version.is_empty());

    client.initialized_notification().await.ok();

    let tools = client
        .tools_list()
        .await
        .expect("tools_list via create_client should succeed");

    assert!(!tools.is_empty(), "Should discover tools via factory path");
    client.shutdown().await.ok();
}

// ─────────────────────────────────────────────────────────────────────────────
// GitHub MCP — requires PAT auth, streamable HTTP
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_github_mcp_connect() {
    let pat = match std::env::var("GITHUB_PAT") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!("GITHUB_PAT not set — skipping GitHub MCP test");
            return;
        }
    };

    let mut headers = HashMap::new();
    headers.insert("Authorization".to_string(), format!("Bearer {}", pat));

    let client = StreamableHttpMcpClient::new("https://api.githubcopilot.com/mcp/", headers);

    // Step 1: Initialize
    let init = client
        .initialize()
        .await
        .expect("GitHub MCP initialize should succeed");

    println!("GitHub MCP server info: {:?}", init.server_info);
    println!("Protocol version: {}", init.protocol_version);

    assert!(!init.protocol_version.is_empty());

    // Step 2: Initialized notification
    client.initialized_notification().await.ok();

    // Step 3: List tools
    let tools = client
        .tools_list()
        .await
        .expect("GitHub MCP tools/list should succeed");

    println!("GitHub MCP tools ({}):", tools.len());
    for tool in &tools {
        println!(
            "  - {} : {}",
            tool.name,
            tool.description.as_deref().unwrap_or("")
        );
    }

    assert!(
        !tools.is_empty(),
        "GitHub MCP should expose at least one tool"
    );

    // Step 4: Shutdown
    client.shutdown().await.expect("shutdown should succeed");
}

// ─────────────────────────────────────────────────────────────────────────────
// Full lifecycle test — initialize, list, call a tool
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_svelte_mcp_call_tool() {
    let client = StreamableHttpMcpClient::new("https://mcp.svelte.dev/mcp", HashMap::new());

    client
        .initialize()
        .await
        .expect("initialize should succeed");

    client.initialized_notification().await.ok();

    let tools = client
        .tools_list()
        .await
        .expect("tools_list should succeed");

    // Try to call the first tool that doesn't need complex arguments
    if let Some(tool) = tools.iter().find(|t| {
        // Look for a tool with no required params or a simple one
        t.name.contains("list") || t.name.contains("get")
    }) {
        println!("Calling tool: {}", tool.name);
        let result = client.call_tool(&tool.name, None).await;
        println!("Tool call result: {:?}", result);
        // We don't assert success because the tool might need arguments,
        // but we verify it doesn't panic or hang
    } else {
        println!("No simple tool found to call, skipping call_tool test");
    }

    client.shutdown().await.ok();
}
