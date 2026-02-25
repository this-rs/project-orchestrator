//! Neo4j Chat Session and Event operations

use super::client::Neo4jClient;
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Chat Session operations
    // ========================================================================

    /// Create a new chat session, optionally linking to a project via slug
    pub async fn create_chat_session(&self, session: &ChatSessionNode) -> Result<()> {
        let q = if session.project_slug.is_some() {
            query(
                r#"
                CREATE (s:ChatSession {
                    id: $id,
                    cli_session_id: $cli_session_id,
                    project_slug: $project_slug,
                    workspace_slug: $workspace_slug,
                    cwd: $cwd,
                    title: $title,
                    model: $model,
                    created_at: datetime($created_at),
                    updated_at: datetime($updated_at),
                    message_count: $message_count,
                    total_cost_usd: $total_cost_usd,
                    conversation_id: $conversation_id,
                    preview: $preview,
                    permission_mode: $permission_mode,
                    add_dirs: $add_dirs
                })
                WITH s
                OPTIONAL MATCH (p:Project {slug: $project_slug})
                FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
                    CREATE (p)-[:HAS_CHAT_SESSION]->(s)
                )
                "#,
            )
        } else {
            query(
                r#"
                CREATE (s:ChatSession {
                    id: $id,
                    cli_session_id: $cli_session_id,
                    project_slug: $project_slug,
                    workspace_slug: $workspace_slug,
                    cwd: $cwd,
                    title: $title,
                    model: $model,
                    created_at: datetime($created_at),
                    updated_at: datetime($updated_at),
                    message_count: $message_count,
                    total_cost_usd: $total_cost_usd,
                    conversation_id: $conversation_id,
                    preview: $preview,
                    permission_mode: $permission_mode,
                    add_dirs: $add_dirs
                })
                "#,
            )
        };

        self.graph
            .run(
                q.param("id", session.id.to_string())
                    .param(
                        "cli_session_id",
                        session.cli_session_id.clone().unwrap_or_default(),
                    )
                    .param(
                        "project_slug",
                        session.project_slug.clone().unwrap_or_default(),
                    )
                    .param(
                        "workspace_slug",
                        session.workspace_slug.clone().unwrap_or_default(),
                    )
                    .param("cwd", session.cwd.clone())
                    .param("title", session.title.clone().unwrap_or_default())
                    .param("model", session.model.clone())
                    .param("created_at", session.created_at.to_rfc3339())
                    .param("updated_at", session.updated_at.to_rfc3339())
                    .param("message_count", session.message_count)
                    .param("total_cost_usd", session.total_cost_usd.unwrap_or(0.0))
                    .param(
                        "conversation_id",
                        session.conversation_id.clone().unwrap_or_default(),
                    )
                    .param("preview", session.preview.clone().unwrap_or_default())
                    .param(
                        "permission_mode",
                        session.permission_mode.clone().unwrap_or_default(),
                    )
                    .param(
                        "add_dirs",
                        serde_json::to_string(&session.add_dirs.clone().unwrap_or_default())
                            .unwrap_or_else(|_| "[]".to_string()),
                    ),
            )
            .await?;
        Ok(())
    }

    /// Get a chat session by ID
    pub async fn get_chat_session(&self, id: Uuid) -> Result<Option<ChatSessionNode>> {
        let q = query("MATCH (s:ChatSession {id: $id}) RETURN s").param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            Ok(Some(Self::parse_chat_session_node(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List chat sessions with optional project_slug filter
    pub async fn list_chat_sessions(
        &self,
        project_slug: Option<&str>,
        workspace_slug: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<ChatSessionNode>, usize)> {
        let (data_query, count_query) = if let Some(slug) = project_slug {
            (
                query(
                    r#"
                    MATCH (s:ChatSession {project_slug: $slug})
                    RETURN s ORDER BY s.updated_at DESC
                    SKIP $offset LIMIT $limit
                    "#,
                )
                .param("slug", slug.to_string())
                .param("offset", offset as i64)
                .param("limit", limit as i64),
                query("MATCH (s:ChatSession {project_slug: $slug}) RETURN count(s) AS total")
                    .param("slug", slug.to_string()),
            )
        } else if let Some(ws) = workspace_slug {
            // Match sessions directly tagged with workspace_slug
            // OR sessions whose project_slug belongs to a project in this workspace
            (
                query(
                    r#"
                    OPTIONAL MATCH (w:Workspace {slug: $ws})<-[:BELONGS_TO_WORKSPACE]-(proj:Project)
                    WITH collect(proj.slug) AS ws_project_slugs
                    MATCH (s:ChatSession)
                    WHERE s.workspace_slug = $ws
                       OR (s.project_slug IS NOT NULL AND s.project_slug IN ws_project_slugs)
                    RETURN s ORDER BY s.updated_at DESC
                    SKIP $offset LIMIT $limit
                    "#,
                )
                .param("ws", ws.to_string())
                .param("offset", offset as i64)
                .param("limit", limit as i64),
                query(
                    r#"
                    OPTIONAL MATCH (w:Workspace {slug: $ws})<-[:BELONGS_TO_WORKSPACE]-(proj:Project)
                    WITH collect(proj.slug) AS ws_project_slugs
                    MATCH (s:ChatSession)
                    WHERE s.workspace_slug = $ws
                       OR (s.project_slug IS NOT NULL AND s.project_slug IN ws_project_slugs)
                    RETURN count(s) AS total
                    "#,
                )
                .param("ws", ws.to_string()),
            )
        } else {
            (
                query(
                    r#"
                    MATCH (s:ChatSession)
                    RETURN s ORDER BY s.updated_at DESC
                    SKIP $offset LIMIT $limit
                    "#,
                )
                .param("offset", offset as i64)
                .param("limit", limit as i64),
                query("MATCH (s:ChatSession) RETURN count(s) AS total"),
            )
        };

        let mut sessions = Vec::new();
        let mut result = self.graph.execute(data_query).await?;
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            sessions.push(Self::parse_chat_session_node(&node)?);
        }

        let mut count_result = self.graph.execute(count_query).await?;
        let total = if let Some(row) = count_result.next().await? {
            row.get::<i64>("total")? as usize
        } else {
            0
        };

        Ok((sessions, total))
    }

    /// Update a chat session (partial, None fields are skipped)
    #[allow(clippy::too_many_arguments)]
    pub async fn update_chat_session(
        &self,
        id: Uuid,
        cli_session_id: Option<String>,
        title: Option<String>,
        message_count: Option<i64>,
        total_cost_usd: Option<f64>,
        conversation_id: Option<String>,
        preview: Option<String>,
    ) -> Result<Option<ChatSessionNode>> {
        let mut set_clauses = vec!["s.updated_at = datetime()".to_string()];

        if let Some(ref v) = cli_session_id {
            set_clauses.push(format!("s.cli_session_id = '{}'", v.replace('\'', "\\'")));
        }
        if let Some(ref v) = title {
            set_clauses.push(format!("s.title = '{}'", v.replace('\'', "\\'")));
        }
        if let Some(v) = message_count {
            set_clauses.push(format!("s.message_count = {}", v));
        }
        if let Some(v) = total_cost_usd {
            set_clauses.push(format!("s.total_cost_usd = {}", v));
        }
        if let Some(ref v) = conversation_id {
            set_clauses.push(format!("s.conversation_id = '{}'", v.replace('\'', "\\'")));
        }
        if let Some(ref v) = preview {
            set_clauses.push(format!("s.preview = '{}'", v.replace('\'', "\\'")));
        }

        let cypher = format!(
            "MATCH (s:ChatSession {{id: $id}}) SET {} RETURN s",
            set_clauses.join(", ")
        );

        let q = query(&cypher).param("id", id.to_string());
        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            Ok(Some(Self::parse_chat_session_node(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Update the permission_mode field on a chat session node
    pub async fn update_chat_session_permission_mode(&self, id: Uuid, mode: &str) -> Result<()> {
        let cypher = "MATCH (s:ChatSession {id: $id}) SET s.permission_mode = $mode, s.updated_at = datetime()";
        let q = query(cypher)
            .param("id", id.to_string())
            .param("mode", mode.to_string());
        self.graph.run(q).await?;
        Ok(())
    }

    /// Set the auto_continue flag on a chat session node.
    pub async fn set_session_auto_continue(&self, id: Uuid, enabled: bool) -> Result<()> {
        let cypher = "MATCH (s:ChatSession {id: $id}) SET s.auto_continue = $enabled, s.updated_at = datetime()";
        let q = query(cypher)
            .param("id", id.to_string())
            .param("enabled", enabled);
        self.graph.run(q).await?;
        Ok(())
    }

    /// Get the auto_continue flag from a chat session node.
    /// Returns `false` if the session doesn't exist or the property is not set.
    pub async fn get_session_auto_continue(&self, id: Uuid) -> Result<bool> {
        let cypher = "MATCH (s:ChatSession {id: $id}) RETURN s.auto_continue AS auto_continue";
        let q = query(cypher).param("id", id.to_string());
        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            // Neo4j may return null if property not set
            Ok(row
                .get::<Option<bool>>("auto_continue")
                .unwrap_or(None)
                .unwrap_or(false))
        } else {
            Ok(false)
        }
    }

    /// Backfill title and preview for sessions that don't have them yet.
    /// Uses the first user_message event stored in Neo4j.
    /// Returns the number of sessions updated.
    pub async fn backfill_chat_session_previews(&self) -> Result<usize> {
        // Find sessions without preview, get their first user_message event
        let q = query(
            r#"
            MATCH (s:ChatSession)
            WHERE s.preview IS NULL OR s.preview = ''
            OPTIONAL MATCH (s)-[:HAS_EVENT]->(e:ChatEvent {event_type: 'user_message'})
            WITH s, e ORDER BY e.seq ASC
            WITH s, collect(e)[0] AS first_event
            WHERE first_event IS NOT NULL
            RETURN s.id AS session_id, first_event.data AS event_data
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut updates = Vec::new();

        while let Some(row) = result.next().await? {
            let session_id: String = row.get("session_id")?;
            let event_data: String = row.get("event_data").unwrap_or_default();

            // Parse the event data JSON to extract content
            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&event_data) {
                if let Some(content) = data.get("content").and_then(|v| v.as_str()) {
                    let chars: Vec<char> = content.chars().collect();
                    let title = if chars.len() > 80 {
                        format!("{}...", chars[..77].iter().collect::<String>().trim_end())
                    } else {
                        content.to_string()
                    };
                    let preview = if chars.len() > 200 {
                        format!("{}...", chars[..197].iter().collect::<String>().trim_end())
                    } else {
                        content.to_string()
                    };
                    updates.push((session_id, title, preview));
                }
            }
        }

        let count = updates.len();
        for (session_id, title, preview) in updates {
            let update_q = query(
                r#"
                MATCH (s:ChatSession {id: $id})
                WHERE s.preview IS NULL OR s.preview = ''
                SET s.title = $title, s.preview = $preview, s.updated_at = datetime()
                "#,
            )
            .param("id", session_id)
            .param("title", title)
            .param("preview", preview);

            let _ = self.graph.run(update_q).await;
        }

        Ok(count)
    }

    /// Delete a chat session
    pub async fn delete_chat_session(&self, id: Uuid) -> Result<bool> {
        // First check existence, then delete
        let check =
            query("MATCH (s:ChatSession {id: $id}) RETURN s.id AS sid").param("id", id.to_string());
        let mut check_result = self.graph.execute(check).await?;
        let exists = check_result.next().await?.is_some();

        if exists {
            let q = query("MATCH (s:ChatSession {id: $id}) DETACH DELETE s")
                .param("id", id.to_string());
            self.graph.run(q).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Parse a Neo4j Node into a ChatSessionNode
    fn parse_chat_session_node(node: &neo4rs::Node) -> Result<ChatSessionNode> {
        let cli_session_id: String = node.get("cli_session_id").unwrap_or_default();
        let project_slug: String = node.get("project_slug").unwrap_or_default();
        let workspace_slug: String = node.get("workspace_slug").unwrap_or_default();
        let title: String = node.get("title").unwrap_or_default();
        let conversation_id: String = node.get("conversation_id").unwrap_or_default();
        let preview: String = node.get("preview").unwrap_or_default();
        let permission_mode: String = node.get("permission_mode").unwrap_or_default();
        let add_dirs_json: String = node.get("add_dirs").unwrap_or_default();

        // Deserialize add_dirs from JSON string (backward compat: empty string → None)
        let add_dirs: Option<Vec<String>> = if add_dirs_json.is_empty() {
            None
        } else {
            serde_json::from_str(&add_dirs_json)
                .ok()
                .and_then(|v: Vec<String>| if v.is_empty() { None } else { Some(v) })
        };

        Ok(ChatSessionNode {
            id: node.get::<String>("id")?.parse()?,
            cli_session_id: if cli_session_id.is_empty() {
                None
            } else {
                Some(cli_session_id)
            },
            project_slug: if project_slug.is_empty() {
                None
            } else {
                Some(project_slug)
            },
            workspace_slug: if workspace_slug.is_empty() {
                None
            } else {
                Some(workspace_slug)
            },
            cwd: node.get("cwd")?,
            title: if title.is_empty() { None } else { Some(title) },
            model: node.get("model")?,
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            updated_at: node
                .get::<String>("updated_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            message_count: node.get("message_count").unwrap_or(0),
            total_cost_usd: {
                let v: f64 = node.get("total_cost_usd").unwrap_or(0.0);
                if v == 0.0 {
                    None
                } else {
                    Some(v)
                }
            },
            conversation_id: if conversation_id.is_empty() {
                None
            } else {
                Some(conversation_id)
            },
            preview: if preview.is_empty() {
                None
            } else {
                Some(preview)
            },
            permission_mode: if permission_mode.is_empty() {
                None
            } else {
                Some(permission_mode)
            },
            add_dirs,
        })
    }

    // ========================================================================
    // Chat Event operations (WebSocket replay & persistence)
    // ========================================================================

    /// Store a batch of chat events for a session
    pub async fn store_chat_events(
        &self,
        session_id: Uuid,
        events: Vec<ChatEventRecord>,
    ) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }

        for event in &events {
            let q = query(
                "MATCH (s:ChatSession {id: $session_id})
                 CREATE (s)-[:HAS_EVENT]->(e:ChatEvent {
                     id: $id,
                     session_id: $session_id,
                     seq: $seq,
                     event_type: $event_type,
                     data: $data,
                     created_at: $created_at
                 })",
            )
            .param("session_id", session_id.to_string())
            .param("id", event.id.to_string())
            .param("seq", event.seq)
            .param("event_type", event.event_type.clone())
            .param("data", event.data.clone())
            .param("created_at", event.created_at.to_rfc3339());

            self.graph.run(q).await?;
        }

        Ok(())
    }

    /// Get chat events for a session after a given sequence number (for replay)
    pub async fn get_chat_events(
        &self,
        session_id: Uuid,
        after_seq: i64,
        limit: i64,
    ) -> Result<Vec<ChatEventRecord>> {
        let q = query(
            "MATCH (s:ChatSession {id: $session_id})-[:HAS_EVENT]->(e:ChatEvent)
             WHERE e.seq > $after_seq
             RETURN e
             ORDER BY e.seq ASC
             LIMIT $limit",
        )
        .param("session_id", session_id.to_string())
        .param("after_seq", after_seq)
        .param("limit", limit);

        let mut result = self.graph.execute(q).await?;
        let mut events = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("e")?;
            events.push(Self::parse_chat_event_node(&node)?);
        }

        Ok(events)
    }

    /// Get chat events with offset-based pagination (for REST/MCP).
    pub async fn get_chat_events_paginated(
        &self,
        session_id: Uuid,
        offset: i64,
        limit: i64,
    ) -> Result<Vec<ChatEventRecord>> {
        let q = query(
            "MATCH (s:ChatSession {id: $session_id})-[:HAS_EVENT]->(e:ChatEvent)
             RETURN e
             ORDER BY e.seq ASC
             SKIP $offset
             LIMIT $limit",
        )
        .param("session_id", session_id.to_string())
        .param("offset", offset)
        .param("limit", limit);

        let mut result = self.graph.execute(q).await?;
        let mut events = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("e")?;
            events.push(Self::parse_chat_event_node(&node)?);
        }

        Ok(events)
    }

    /// Count total chat events for a session.
    pub async fn count_chat_events(&self, session_id: Uuid) -> Result<i64> {
        let q = query(
            "MATCH (s:ChatSession {id: $session_id})-[:HAS_EVENT]->(e:ChatEvent)
             RETURN count(e) AS cnt",
        )
        .param("session_id", session_id.to_string());

        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            let cnt: i64 = row.get("cnt").unwrap_or(0);
            Ok(cnt)
        } else {
            Ok(0)
        }
    }

    /// Get the latest sequence number for a session (0 if no events)
    pub async fn get_latest_chat_event_seq(&self, session_id: Uuid) -> Result<i64> {
        let q = query(
            "MATCH (s:ChatSession {id: $session_id})-[:HAS_EVENT]->(e:ChatEvent)
             RETURN MAX(e.seq) AS max_seq",
        )
        .param("session_id", session_id.to_string());

        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            // MAX returns null if no rows, so unwrap_or(0)
            let max_seq: i64 = row.get("max_seq").unwrap_or(0);
            Ok(max_seq)
        } else {
            Ok(0)
        }
    }

    /// Delete all chat events for a session
    pub async fn delete_chat_events(&self, session_id: Uuid) -> Result<()> {
        let q = query(
            "MATCH (s:ChatSession {id: $session_id})-[:HAS_EVENT]->(e:ChatEvent)
             DETACH DELETE e",
        )
        .param("session_id", session_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Parse a Neo4j Node into a ChatEventRecord
    fn parse_chat_event_node(node: &neo4rs::Node) -> Result<ChatEventRecord> {
        Ok(ChatEventRecord {
            id: node.get::<String>("id")?.parse()?,
            session_id: node.get::<String>("session_id")?.parse()?,
            seq: node.get("seq")?,
            event_type: node.get("event_type")?,
            data: node.get("data")?,
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
        })
    }
}
