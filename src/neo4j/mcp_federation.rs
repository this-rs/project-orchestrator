//! Neo4j CRUD operations for MCP Federation nodes (McpServer, McpTool).
//!
//! Stores external MCP server registrations and their discovered tools
//! in the knowledge graph, enabling relationship tracking (EXPOSES,
//! SIMILAR_TO, CO_ACTIVATED_WITH, OFTEN_FOLLOWS).

use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

use super::client::Neo4jClient;
use super::models::{McpServerNode, McpToolNode};

impl Neo4jClient {
    // ========================================================================
    // McpServer CRUD
    // ========================================================================

    pub async fn create_mcp_server(&self, server: &McpServerNode) -> Result<()> {
        let q = query(
            r#"
            CREATE (s:McpServer {
                id: $id,
                project_id: $project_id,
                server_id: $server_id,
                display_name: $display_name,
                transport_type: $transport_type,
                transport_url: $transport_url,
                transport_command: $transport_command,
                transport_args: $transport_args,
                status: $status,
                protocol_version: $protocol_version,
                server_name: $server_name,
                tool_count: $tool_count,
                created_at: datetime($created_at),
                last_connected_at: $last_connected_at
            })
            "#,
        )
        .param("id", server.id.to_string())
        .param("project_id", server.project_id.to_string())
        .param("server_id", server.server_id.clone())
        .param("display_name", server.display_name.clone())
        .param("transport_type", server.transport_type.clone())
        .param(
            "transport_url",
            server.transport_url.clone().unwrap_or_default(),
        )
        .param(
            "transport_command",
            server.transport_command.clone().unwrap_or_default(),
        )
        .param(
            "transport_args",
            server.transport_args.clone().unwrap_or_default(),
        )
        .param("status", server.status.clone())
        .param(
            "protocol_version",
            server.protocol_version.clone().unwrap_or_default(),
        )
        .param(
            "server_name",
            server.server_name.clone().unwrap_or_default(),
        )
        .param("tool_count", server.tool_count as i64)
        .param("created_at", server.created_at.to_rfc3339())
        .param(
            "last_connected_at",
            server
                .last_connected_at
                .map(|t| t.to_rfc3339())
                .unwrap_or_default(),
        );

        self.graph.run(q).await?;

        // Link to project
        let link_q = query(
            r#"
            MATCH (p:Project {id: $project_id})
            MATCH (s:McpServer {id: $server_id})
            MERGE (p)-[:HAS_MCP_SERVER]->(s)
            "#,
        )
        .param("project_id", server.project_id.to_string())
        .param("server_id", server.id.to_string());

        self.graph.run(link_q).await?;
        Ok(())
    }

    pub async fn get_mcp_server(&self, id: Uuid) -> Result<Option<McpServerNode>> {
        let q = query(
            r#"
            MATCH (s:McpServer {id: $id})
            RETURN s
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            Ok(Some(self.parse_mcp_server_node(&node)?))
        } else {
            Ok(None)
        }
    }

    pub async fn list_mcp_servers(&self, project_id: Uuid) -> Result<Vec<McpServerNode>> {
        let q = query(
            r#"
            MATCH (s:McpServer {project_id: $project_id})
            RETURN s ORDER BY s.server_id
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut servers = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            servers.push(self.parse_mcp_server_node(&node)?);
        }
        Ok(servers)
    }

    pub async fn delete_mcp_server(&self, id: Uuid) -> Result<()> {
        // Delete all tools first, then the server
        let q = query(
            r#"
            MATCH (s:McpServer {id: $id})
            OPTIONAL MATCH (s)-[:EXPOSES]->(t:McpTool)
            DETACH DELETE t, s
            "#,
        )
        .param("id", id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    pub async fn update_mcp_server_status(&self, id: Uuid, status: &str) -> Result<()> {
        let q = query(
            r#"
            MATCH (s:McpServer {id: $id})
            SET s.status = $status, s.updated_at = datetime()
            "#,
        )
        .param("id", id.to_string())
        .param("status", status.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // McpTool CRUD
    // ========================================================================

    pub async fn create_mcp_tool(&self, tool: &McpToolNode) -> Result<()> {
        let q = query(
            r#"
            CREATE (t:McpTool {
                id: $id,
                server_id: $server_id,
                name: $name,
                fqn: $fqn,
                description: $description,
                input_schema: $input_schema,
                category: $category,
                embedding: $embedding,
                created_at: datetime($created_at)
            })
            "#,
        )
        .param("id", tool.id.to_string())
        .param("server_id", tool.server_id.clone())
        .param("name", tool.name.clone())
        .param("fqn", tool.fqn.clone())
        .param("description", tool.description.clone())
        .param("input_schema", tool.input_schema.clone())
        .param("category", tool.category.clone())
        .param("embedding", tool.embedding.clone().unwrap_or_default())
        .param("created_at", tool.created_at.to_rfc3339());

        self.graph.run(q).await?;

        // Link tool to its server via EXPOSES
        let link_q = query(
            r#"
            MATCH (s:McpServer {server_id: $server_id})
            MATCH (t:McpTool {id: $tool_id})
            MERGE (s)-[:EXPOSES]->(t)
            "#,
        )
        .param("server_id", tool.server_id.clone())
        .param("tool_id", tool.id.to_string());

        self.graph.run(link_q).await?;
        Ok(())
    }

    pub async fn list_mcp_tools_for_server(
        &self,
        server_id: &str,
    ) -> Result<Vec<McpToolNode>> {
        let q = query(
            r#"
            MATCH (t:McpTool {server_id: $server_id})
            RETURN t ORDER BY t.name
            "#,
        )
        .param("server_id", server_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tools = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            tools.push(self.parse_mcp_tool_node(&node)?);
        }
        Ok(tools)
    }

    pub async fn delete_mcp_tools_for_server(&self, server_id: &str) -> Result<()> {
        let q = query(
            r#"
            MATCH (t:McpTool {server_id: $server_id})
            DETACH DELETE t
            "#,
        )
        .param("server_id", server_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // MCP Relations
    // ========================================================================

    /// Create a SIMILAR_TO relation between two tools (external or internal).
    pub async fn create_mcp_similar_to(
        &self,
        tool_fqn: &str,
        similar_to_fqn: &str,
        score: f64,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (a:McpTool {fqn: $fqn_a})
            MATCH (b:McpTool {fqn: $fqn_b})
            MERGE (a)-[r:SIMILAR_TO]->(b)
            SET r.score = $score, r.updated_at = datetime()
            "#,
        )
        .param("fqn_a", tool_fqn.to_string())
        .param("fqn_b", similar_to_fqn.to_string())
        .param("score", score);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Create a CO_ACTIVATED_WITH relation (two tools used in the same session).
    pub async fn create_mcp_co_activated(
        &self,
        tool_fqn_a: &str,
        tool_fqn_b: &str,
        count: i64,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (a:McpTool {fqn: $fqn_a})
            MATCH (b:McpTool {fqn: $fqn_b})
            MERGE (a)-[r:CO_ACTIVATED_WITH]->(b)
            SET r.count = $count, r.updated_at = datetime()
            "#,
        )
        .param("fqn_a", tool_fqn_a.to_string())
        .param("fqn_b", tool_fqn_b.to_string())
        .param("count", count);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Create an OFTEN_FOLLOWS relation (tool B often called after tool A).
    pub async fn create_mcp_often_follows(
        &self,
        tool_fqn_a: &str,
        tool_fqn_b: &str,
        count: i64,
        avg_delta_ms: f64,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (a:McpTool {fqn: $fqn_a})
            MATCH (b:McpTool {fqn: $fqn_b})
            MERGE (a)-[r:OFTEN_FOLLOWS]->(b)
            SET r.count = $count, r.avg_delta_ms = $avg_delta_ms, r.updated_at = datetime()
            "#,
        )
        .param("fqn_a", tool_fqn_a.to_string())
        .param("fqn_b", tool_fqn_b.to_string())
        .param("count", count)
        .param("avg_delta_ms", avg_delta_ms);

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Parsing helpers
    // ========================================================================

    fn parse_mcp_server_node(&self, node: &neo4rs::Node) -> Result<McpServerNode> {
        use chrono::{DateTime, Utc};

        let id_str: String = node.get("id")?;
        let project_id_str: String = node.get("project_id")?;
        let created_at_str: String = node.get("created_at").unwrap_or_default();
        let updated_at_str: String = node.get("updated_at").unwrap_or_default();
        let last_connected_str: String = node.get("last_connected_at").unwrap_or_default();

        Ok(McpServerNode {
            id: id_str.parse()?,
            project_id: project_id_str.parse()?,
            server_id: node.get("server_id")?,
            display_name: node.get("display_name").unwrap_or_default(),
            transport_type: node.get("transport_type").unwrap_or_default(),
            transport_url: node.get("transport_url").ok().filter(|s: &String| !s.is_empty()),
            transport_command: node.get("transport_command").ok().filter(|s: &String| !s.is_empty()),
            transport_args: node.get("transport_args").ok().filter(|s: &String| !s.is_empty()),
            status: node.get("status").unwrap_or_else(|_| "disconnected".to_string()),
            protocol_version: node.get("protocol_version").ok().filter(|s: &String| !s.is_empty()),
            server_name: node.get("server_name").ok().filter(|s: &String| !s.is_empty()),
            tool_count: node.get::<i64>("tool_count").unwrap_or(0) as usize,
            created_at: DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: DateTime::parse_from_rfc3339(&updated_at_str)
                .map(|dt| Some(dt.with_timezone(&Utc)))
                .unwrap_or(None),
            last_connected_at: DateTime::parse_from_rfc3339(&last_connected_str)
                .map(|dt| Some(dt.with_timezone(&Utc)))
                .unwrap_or(None),
        })
    }

    fn parse_mcp_tool_node(&self, node: &neo4rs::Node) -> Result<McpToolNode> {
        use chrono::{DateTime, Utc};

        let id_str: String = node.get("id")?;
        let created_at_str: String = node.get("created_at").unwrap_or_default();
        let updated_at_str: String = node.get("updated_at").unwrap_or_default();

        Ok(McpToolNode {
            id: id_str.parse()?,
            server_id: node.get("server_id")?,
            name: node.get("name")?,
            fqn: node.get("fqn")?,
            description: node.get("description").unwrap_or_default(),
            input_schema: node.get("input_schema").unwrap_or_else(|_| "{}".to_string()),
            category: node.get("category").unwrap_or_else(|_| "unknown".to_string()),
            embedding: node.get("embedding").ok().filter(|s: &String| !s.is_empty()),
            created_at: DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: DateTime::parse_from_rfc3339(&updated_at_str)
                .map(|dt| Some(dt.with_timezone(&Utc)))
                .unwrap_or(None),
        })
    }
}
