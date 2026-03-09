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
                    add_dirs: $add_dirs,
                    spawned_by: $spawned_by
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
                    add_dirs: $add_dirs,
                    spawned_by: $spawned_by
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
                    )
                    .param("spawned_by", session.spawned_by.clone().unwrap_or_default()),
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

    /// List chat sessions with optional project_slug filter.
    /// When `include_detached` is false (default), sessions with a non-empty `spawned_by`
    /// field are excluded (e.g. PlanRunner sub-sessions).
    pub async fn list_chat_sessions(
        &self,
        project_slug: Option<&str>,
        workspace_slug: Option<&str>,
        limit: usize,
        offset: usize,
        include_detached: bool,
    ) -> Result<(Vec<ChatSessionNode>, usize)> {
        // Filter clause to exclude detached (spawned) sessions unless explicitly requested
        let detached_filter = if include_detached {
            ""
        } else {
            " AND (s.spawned_by IS NULL OR s.spawned_by = '')"
        };

        let (data_query, count_query) = if let Some(slug) = project_slug {
            (
                query(&format!(
                    r#"
                    MATCH (s:ChatSession {{project_slug: $slug}})
                    WHERE true{detached_filter}
                    RETURN s ORDER BY s.updated_at DESC
                    SKIP $offset LIMIT $limit
                    "#,
                ))
                .param("slug", slug.to_string())
                .param("offset", offset as i64)
                .param("limit", limit as i64),
                query(&format!(
                    "MATCH (s:ChatSession {{project_slug: $slug}}) WHERE true{detached_filter} RETURN count(s) AS total",
                ))
                    .param("slug", slug.to_string()),
            )
        } else if let Some(ws) = workspace_slug {
            // Match sessions directly tagged with workspace_slug
            // OR sessions whose project_slug belongs to a project in this workspace
            (
                query(&format!(
                    r#"
                    OPTIONAL MATCH (w:Workspace {{slug: $ws}})<-[:BELONGS_TO_WORKSPACE]-(proj:Project)
                    WITH collect(proj.slug) AS ws_project_slugs
                    MATCH (s:ChatSession)
                    WHERE (s.workspace_slug = $ws
                       OR (s.project_slug IS NOT NULL AND s.project_slug IN ws_project_slugs)){detached_filter}
                    RETURN s ORDER BY s.updated_at DESC
                    SKIP $offset LIMIT $limit
                    "#,
                ))
                .param("ws", ws.to_string())
                .param("offset", offset as i64)
                .param("limit", limit as i64),
                query(&format!(
                    r#"
                    OPTIONAL MATCH (w:Workspace {{slug: $ws}})<-[:BELONGS_TO_WORKSPACE]-(proj:Project)
                    WITH collect(proj.slug) AS ws_project_slugs
                    MATCH (s:ChatSession)
                    WHERE (s.workspace_slug = $ws
                       OR (s.project_slug IS NOT NULL AND s.project_slug IN ws_project_slugs)){detached_filter}
                    RETURN count(s) AS total
                    "#,
                ))
                .param("ws", ws.to_string()),
            )
        } else {
            (
                query(&format!(
                    r#"
                    MATCH (s:ChatSession)
                    WHERE true{detached_filter}
                    RETURN s ORDER BY s.updated_at DESC
                    SKIP $offset LIMIT $limit
                    "#,
                ))
                .param("offset", offset as i64)
                .param("limit", limit as i64),
                query(&format!(
                    "MATCH (s:ChatSession) WHERE true{detached_filter} RETURN count(s) AS total",
                )),
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

    /// Get child sessions spawned by a parent session (identified by parent_id in spawned_by JSON).
    pub async fn get_session_children(&self, parent_id: Uuid) -> Result<Vec<ChatSessionNode>> {
        let q = query(
            r#"
            MATCH (s:ChatSession)
            WHERE s.spawned_by IS NOT NULL
              AND s.spawned_by <> ''
              AND s.spawned_by CONTAINS $parent_id
            RETURN s ORDER BY s.created_at ASC
            "#,
        )
        .param("parent_id", parent_id.to_string());

        let mut sessions = Vec::new();
        let mut result = self.graph.execute(q).await?;
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            sessions.push(Self::parse_chat_session_node(&node)?);
        }
        Ok(sessions)
    }

    // ========================================================================
    // Discussion Graph — SPAWNED_BY relations & tree traversal
    // ========================================================================

    /// Create a SPAWNED_BY relation between two chat sessions.
    /// `(:ChatSession {id: child})-[:SPAWNED_BY {type, run_id, task_id, created_at}]->(:ChatSession {id: parent})`
    pub async fn create_spawned_by_relation(
        &self,
        child_session_id: &str,
        parent_session_id: &str,
        spawn_type: &str,
        run_id: Option<Uuid>,
        task_id: Option<Uuid>,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (child:ChatSession {id: $child_id})
            MATCH (parent:ChatSession {id: $parent_id})
            CREATE (child)-[:SPAWNED_BY {
                type: $spawn_type,
                run_id: $run_id,
                task_id: $task_id,
                created_at: datetime()
            }]->(parent)
            "#,
        )
        .param("child_id", child_session_id.to_string())
        .param("parent_id", parent_session_id.to_string())
        .param("spawn_type", spawn_type.to_string())
        .param("run_id", run_id.map(|u| u.to_string()).unwrap_or_default())
        .param("task_id", task_id.map(|u| u.to_string()).unwrap_or_default());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get the full session tree rooted at `session_id` (recursive traversal via SPAWNED_BY).
    /// Bounded to 10 levels max for performance.
    pub async fn get_session_tree(&self, session_id: &str) -> Result<Vec<SessionTreeNode>> {
        let q = query(
            r#"
            MATCH path = (root:ChatSession {id: $id})<-[:SPAWNED_BY*0..10]-(child:ChatSession)
            WITH child, length(path) AS depth, relationships(path) AS rels
            RETURN child.id AS session_id,
                   CASE WHEN size(rels) > 0 THEN startNode(rels[size(rels)-1]).id ELSE null END AS self_id,
                   CASE WHEN size(rels) > 0 THEN endNode(rels[size(rels)-1]).id ELSE null END AS parent_session_id,
                   CASE WHEN size(rels) > 0 THEN rels[size(rels)-1].type ELSE null END AS spawn_type,
                   CASE WHEN size(rels) > 0 THEN rels[size(rels)-1].run_id ELSE null END AS run_id,
                   CASE WHEN size(rels) > 0 THEN rels[size(rels)-1].task_id ELSE null END AS task_id,
                   depth,
                   child.created_at AS created_at
            ORDER BY depth ASC, child.created_at ASC
            "#,
        )
        .param("id", session_id.to_string());

        let mut nodes = Vec::new();
        let mut result = self.graph.execute(q).await?;
        while let Some(row) = result.next().await? {
            let session_id: String = row.get("session_id")?;
            let parent_session_id: Option<String> = row.get("parent_session_id").ok().and_then(|s: String| if s.is_empty() { None } else { Some(s) });
            let spawn_type: Option<String> = row.get("spawn_type").ok().and_then(|s: String| if s.is_empty() { None } else { Some(s) });
            let run_id_str: Option<String> = row.get("run_id").ok().and_then(|s: String| if s.is_empty() { None } else { Some(s) });
            let task_id_str: Option<String> = row.get("task_id").ok().and_then(|s: String| if s.is_empty() { None } else { Some(s) });
            let depth: i64 = row.get("depth").unwrap_or(0);
            let created_at_str: Option<String> = row.get("created_at").ok();

            nodes.push(SessionTreeNode {
                session_id,
                parent_session_id,
                spawn_type,
                run_id: run_id_str.and_then(|s| s.parse().ok()),
                task_id: task_id_str.and_then(|s| s.parse().ok()),
                depth: depth as u32,
                created_at: created_at_str.and_then(|s| s.parse().ok()),
            });
        }
        Ok(nodes)
    }

    /// Follow SPAWNED_BY upward to find the root session (the one with no parent).
    pub async fn get_session_root(&self, session_id: &str) -> Result<Option<String>> {
        let q = query(
            r#"
            MATCH path = (s:ChatSession {id: $id})-[:SPAWNED_BY*0..10]->(root:ChatSession)
            WHERE NOT (root)-[:SPAWNED_BY]->()
            RETURN root.id AS root_id
            ORDER BY length(path) DESC
            LIMIT 1
            "#,
        )
        .param("id", session_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let root_id: String = row.get("root_id")?;
            Ok(Some(root_id))
        } else {
            Ok(None)
        }
    }

    /// Get all sessions for a PlanRun via SPAWNED_BY relation metadata.
    pub async fn get_run_sessions(&self, run_id: Uuid) -> Result<Vec<SessionInfo>> {
        let q = query(
            r#"
            MATCH (child:ChatSession)-[r:SPAWNED_BY]->(parent:ChatSession)
            WHERE r.run_id = $run_id
            RETURN child.id AS session_id,
                   child.title AS title,
                   child.model AS model,
                   r.type AS spawn_type,
                   r.task_id AS task_id,
                   child.created_at AS created_at
            ORDER BY child.created_at ASC
            "#,
        )
        .param("run_id", run_id.to_string());

        let mut sessions = Vec::new();
        let mut result = self.graph.execute(q).await?;
        while let Some(row) = result.next().await? {
            let session_id: String = row.get("session_id")?;
            let title: Option<String> = row.get("title").ok().and_then(|s: String| if s.is_empty() { None } else { Some(s) });
            let model: String = row.get("model").unwrap_or_default();
            let spawn_type: Option<String> = row.get("spawn_type").ok().and_then(|s: String| if s.is_empty() { None } else { Some(s) });
            let task_id_str: Option<String> = row.get("task_id").ok().and_then(|s: String| if s.is_empty() { None } else { Some(s) });
            let created_at_str: String = row.get("created_at").unwrap_or_default();

            sessions.push(SessionInfo {
                session_id,
                title,
                model,
                spawn_type,
                task_id: task_id_str.and_then(|s| s.parse().ok()),
                created_at: created_at_str.parse().unwrap_or_else(|_| chrono::Utc::now()),
            });
        }
        Ok(sessions)
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
        let spawned_by: String = node.get("spawned_by").unwrap_or_default();

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
            spawned_by: if spawned_by.is_empty() {
                None
            } else {
                Some(spawned_by)
            },
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

    // ========================================================================
    // DISCUSSED relations (ChatSession → Entity)
    // ========================================================================

    /// Add DISCUSSED relations between a chat session and entities.
    ///
    /// Uses MERGE for idempotence: if the relation already exists, increments
    /// `mention_count` and updates `last_mentioned_at`.
    /// Uses per-type UNWIND batching (one query per entity type).
    ///
    /// Each entity is identified by `(entity_type, entity_id)` where:
    /// - entity_type: "File", "Function", "Struct", "Trait", "Enum"
    /// - entity_id: the path (for File) or name (for symbols)
    pub async fn add_discussed(
        &self,
        session_id: Uuid,
        entities: &[(String, String)], // Vec<(entity_type, entity_id)>
    ) -> Result<usize> {
        if entities.is_empty() {
            return Ok(0);
        }

        let mut total = 0usize;

        // Group entities by type for efficient per-type UNWIND queries
        let mut files: Vec<String> = Vec::new();
        let mut functions: Vec<String> = Vec::new();
        let mut structs: Vec<String> = Vec::new();
        let mut traits: Vec<String> = Vec::new();
        let mut enums: Vec<String> = Vec::new();

        for (etype, eid) in entities {
            match etype.as_str() {
                "File" => files.push(eid.clone()),
                "Function" => functions.push(eid.clone()),
                "Struct" => structs.push(eid.clone()),
                "Trait" => traits.push(eid.clone()),
                "Enum" => enums.push(eid.clone()),
                _ => {} // skip unknown types
            }
        }

        // Helper macro: run UNWIND MERGE for a given label and match property
        // File nodes match on `path`, all others match on `name`
        let sid = session_id.to_string();

        if !files.is_empty() {
            let q = query(
                r#"
                MATCH (s:ChatSession {id: $session_id})
                UNWIND $ids AS file_path
                MATCH (e:File {path: file_path})
                MERGE (s)-[r:DISCUSSED]->(e)
                ON CREATE SET r.mention_count = 1,
                             r.first_mentioned_at = datetime(),
                             r.last_mentioned_at = datetime()
                ON MATCH SET r.mention_count = r.mention_count + 1,
                            r.last_mentioned_at = datetime()
                RETURN count(r) AS cnt
                "#,
            )
            .param("session_id", sid.clone())
            .param("ids", files);

            let mut result = self.graph.execute(q).await?;
            if let Some(row) = result.next().await? {
                total += row.get::<i64>("cnt").unwrap_or(0) as usize;
            }
        }

        // For symbol types (Function, Struct, Trait, Enum): match on name
        let symbol_types: &[(&str, &str, Vec<String>)] = &[
            ("Function", "name", functions),
            ("Struct", "name", structs),
            ("Trait", "name", traits),
            ("Enum", "name", enums),
        ];

        for (label, _prop, ids) in symbol_types {
            if ids.is_empty() {
                continue;
            }

            // Dynamic label in Cypher via string formatting (label is a static &str, safe)
            let cypher = format!(
                r#"
                MATCH (s:ChatSession {{id: $session_id}})
                UNWIND $ids AS sym_name
                MATCH (e:{} {{name: sym_name}})
                WITH s, e
                LIMIT 1
                MERGE (s)-[r:DISCUSSED]->(e)
                ON CREATE SET r.mention_count = 1,
                             r.first_mentioned_at = datetime(),
                             r.last_mentioned_at = datetime()
                ON MATCH SET r.mention_count = r.mention_count + 1,
                            r.last_mentioned_at = datetime()
                RETURN count(r) AS cnt
                "#,
                label
            );

            let q = query(&cypher)
                .param("session_id", sid.clone())
                .param("ids", ids.clone());

            let mut result = self.graph.execute(q).await?;
            if let Some(row) = result.next().await? {
                total += row.get::<i64>("cnt").unwrap_or(0) as usize;
            }
        }

        Ok(total)
    }

    /// Get all entities discussed in a chat session.
    ///
    /// Returns a list of `(entity_type, entity_id, mention_count, last_mentioned_at)`.
    /// Scoped by project_id if provided (security: no cross-project leaks).
    pub async fn get_session_entities(
        &self,
        session_id: Uuid,
        project_id: Option<Uuid>,
    ) -> Result<Vec<DiscussedEntity>> {
        let cypher = if project_id.is_some() {
            r#"
            MATCH (s:ChatSession {id: $session_id})-[r:DISCUSSED]->(e)
            WHERE (e:File AND EXISTS { MATCH (e)<-[:CONTAINS]-(p:Project {id: $project_id}) })
               OR (e:Function AND e.project_id = $project_id)
               OR (e:Struct AND e.project_id = $project_id)
               OR (e:Trait AND e.project_id = $project_id)
               OR (e:Enum AND e.project_id = $project_id)
            RETURN
              CASE
                WHEN e:File THEN 'File'
                WHEN e:Function THEN 'Function'
                WHEN e:Struct THEN 'Struct'
                WHEN e:Trait THEN 'Trait'
                WHEN e:Enum THEN 'Enum'
                ELSE 'Unknown'
              END AS entity_type,
              CASE
                WHEN e:File THEN e.path
                ELSE e.name
              END AS entity_id,
              r.mention_count AS mention_count,
              toString(r.last_mentioned_at) AS last_mentioned_at,
              CASE WHEN e:Function THEN e.file_path
                   WHEN e:Struct THEN e.file_path
                   WHEN e:Trait THEN e.file_path
                   WHEN e:Enum THEN e.file_path
                   ELSE null
              END AS file_path
            ORDER BY r.mention_count DESC
            "#
        } else {
            r#"
            MATCH (s:ChatSession {id: $session_id})-[r:DISCUSSED]->(e)
            RETURN
              CASE
                WHEN e:File THEN 'File'
                WHEN e:Function THEN 'Function'
                WHEN e:Struct THEN 'Struct'
                WHEN e:Trait THEN 'Trait'
                WHEN e:Enum THEN 'Enum'
                ELSE 'Unknown'
              END AS entity_type,
              CASE
                WHEN e:File THEN e.path
                ELSE e.name
              END AS entity_id,
              r.mention_count AS mention_count,
              toString(r.last_mentioned_at) AS last_mentioned_at,
              CASE WHEN e:Function THEN e.file_path
                   WHEN e:Struct THEN e.file_path
                   WHEN e:Trait THEN e.file_path
                   WHEN e:Enum THEN e.file_path
                   ELSE null
              END AS file_path
            ORDER BY r.mention_count DESC
            "#
        };

        let mut q = query(cypher).param("session_id", session_id.to_string());
        if let Some(pid) = project_id {
            q = q.param("project_id", pid.to_string());
        }

        let mut result = self.graph.execute(q).await?;
        let mut entities = Vec::new();
        while let Some(row) = result.next().await? {
            entities.push(DiscussedEntity {
                entity_type: row.get("entity_type").unwrap_or_default(),
                entity_id: row.get("entity_id").unwrap_or_default(),
                mention_count: row.get("mention_count").unwrap_or(1),
                last_mentioned_at: row.get::<String>("last_mentioned_at").ok(),
                file_path: row.get::<String>("file_path").ok(),
            });
        }
        Ok(entities)
    }

    /// WorldModel predictive context (biomimicry T7):
    /// Get co-changers of files discussed in the last N sessions for a project.
    /// Returns predicted files the agent may need, based on recent discussion patterns.
    pub async fn get_discussed_co_changers(
        &self,
        project_id: Uuid,
        max_sessions: i64,
        max_results: i64,
    ) -> Result<Vec<super::models::CoChanger>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:HAS_CHAT_SESSION]->(s:ChatSession)
            WITH s ORDER BY s.created_at DESC LIMIT $max_sessions
            MATCH (s)-[:DISCUSSED]->(f:File)
            WITH DISTINCT f
            MATCH (f)-[cc:CO_CHANGED]->(f2:File)
            WHERE cc.count >= 2
            AND NOT f2 = f
            WITH f2.path AS path, max(cc.count) AS count, max(toString(cc.last_at)) AS last_at
            ORDER BY count DESC
            LIMIT $max_results
            RETURN path, count, last_at
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("max_sessions", max_sessions)
        .param("max_results", max_results);

        let mut result = self.graph.execute(q).await?;
        let mut changers = Vec::new();

        while let Some(row) = result.next().await? {
            changers.push(super::models::CoChanger {
                path: row.get("path")?,
                count: row.get("count")?,
                last_at: row.get::<String>("last_at").ok(),
            });
        }

        Ok(changers)
    }

    /// Backfill DISCUSSED relations on all existing chat sessions.
    ///
    /// Iterates all sessions in batches, extracts entities from user_message events
    /// using the entity_extractor, and creates DISCUSSED relations via MERGE.
    /// Idempotent — safe to run multiple times (MERGE prevents duplicates).
    ///
    /// Returns `(sessions_processed, entities_found, relations_created)`.
    pub async fn backfill_discussed(&self) -> Result<(usize, usize, usize)> {
        use crate::chat::entity_extractor;
        use tracing::{debug, info};

        const BATCH_SIZE: usize = 100;
        let mut total_sessions = 0usize;
        let mut total_entities = 0usize;
        let mut total_relations = 0usize;
        let mut offset = 0;

        loop {
            // Fetch a batch of sessions that have user_message events
            let q = query(
                r#"
                MATCH (s:ChatSession)-[:HAS_EVENT]->(e:ChatEvent)
                WHERE e.event_type IN ['user_message', 'assistant_text']
                WITH s, collect(e.data) AS messages
                RETURN s.id AS session_id, messages
                ORDER BY s.created_at ASC
                SKIP $offset
                LIMIT $limit
                "#,
            )
            .param("offset", offset as i64)
            .param("limit", BATCH_SIZE as i64);

            let mut result = self.graph.execute(q).await?;
            let mut batch_count = 0;

            while let Some(row) = result.next().await? {
                let session_id_str: String = row.get("session_id")?;
                let session_uuid: Uuid = match session_id_str.parse() {
                    Ok(id) => id,
                    Err(_) => continue,
                };

                let messages: Vec<String> = row.get("messages").unwrap_or_default();

                // Extract entities from all messages in this session
                let mut all_entities: Vec<(String, String)> = Vec::new();
                let mut seen = std::collections::HashSet::new();

                for msg_data in &messages {
                    // Parse message JSON to get content
                    if let Ok(data) = serde_json::from_str::<serde_json::Value>(msg_data) {
                        let content = data
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default();

                        if content.len() < 5 {
                            continue;
                        }

                        let entities = entity_extractor::extract_entities(content);
                        for e in entities {
                            let label = match e.entity_type {
                                entity_extractor::EntityType::File => "File",
                                entity_extractor::EntityType::Function => "Function",
                                entity_extractor::EntityType::Struct => "Struct",
                                entity_extractor::EntityType::Trait => "Trait",
                                entity_extractor::EntityType::Enum => "Enum",
                                entity_extractor::EntityType::Symbol => "Function",
                            };
                            let key = (label.to_string(), e.identifier.clone());
                            if seen.insert(key.clone()) {
                                all_entities.push(key);
                            }
                        }
                    }
                }

                total_entities += all_entities.len();

                if !all_entities.is_empty() {
                    match self.add_discussed(session_uuid, &all_entities).await {
                        Ok(created) => {
                            total_relations += created;
                        }
                        Err(e) => {
                            debug!(
                                session_id = %session_uuid,
                                error = %e,
                                "Backfill: failed to create DISCUSSED for session"
                            );
                        }
                    }
                }

                batch_count += 1;
                total_sessions += 1;
            }

            info!(
                batch = offset / BATCH_SIZE + 1,
                sessions = total_sessions,
                entities = total_entities,
                relations = total_relations,
                "Backfill DISCUSSED: batch processed"
            );

            if batch_count < BATCH_SIZE {
                break; // Last batch
            }
            offset += BATCH_SIZE;
        }

        info!(
            sessions = total_sessions,
            entities = total_entities,
            relations = total_relations,
            "Backfill DISCUSSED: complete"
        );

        Ok((total_sessions, total_entities, total_relations))
    }

    // ========================================================================
    // Graph visualization — Chat layer
    // ========================================================================

    /// Get chat sessions with DISCUSSED relations for the graph visualization.
    /// Only returns sessions that have at least 1 DISCUSSED relation.
    /// Ordered by most recent, limited to `limit` sessions.
    /// For each session, returns top 10 discussed entities by mention_count.
    pub async fn get_chat_graph_data(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<(Vec<ChatGraphSession>, Vec<ChatGraphDiscussed>)> {
        let pid = project_id.to_string();
        let lim = limit as i64;

        // Query 1: Sessions that have DISCUSSED relations, scoped by project
        let sessions_query = query(
            "MATCH (proj:Project {id: $pid})-[:HAS_CHAT_SESSION]->(cs:ChatSession)-[d:DISCUSSED]->()
             WITH cs, count(d) AS disc_count
             WHERE disc_count > 0
             RETURN cs.id AS id,
                    cs.title AS title,
                    cs.model AS model,
                    cs.message_count AS message_count,
                    cs.total_cost_usd AS total_cost_usd,
                    toString(cs.created_at) AS created_at
             ORDER BY cs.updated_at DESC
             LIMIT $lim",
        )
        .param("pid", pid.clone())
        .param("lim", lim);

        let mut result = self.graph.execute(sessions_query).await?;
        let mut sessions = Vec::new();
        let mut session_ids = Vec::new();
        while let Some(row) = result.next().await? {
            let id: String = row.get("id")?;
            session_ids.push(id.clone());
            sessions.push(ChatGraphSession {
                id,
                title: row
                    .get::<String>("title")
                    .unwrap_or_else(|_| "Untitled".to_string()),
                model: row.get::<Option<String>>("model").ok().flatten(),
                message_count: row.get::<i64>("message_count").unwrap_or(0),
                total_cost_usd: row.get::<f64>("total_cost_usd").unwrap_or(0.0),
                created_at: row.get::<String>("created_at").unwrap_or_default(),
            });
        }

        if sessions.is_empty() {
            return Ok((sessions, vec![]));
        }

        // Query 2: DISCUSSED edges for those sessions, top 10 per session by mention_count
        let discussed_query = query(
            "MATCH (cs:ChatSession)-[d:DISCUSSED]->(target)
             WHERE cs.id IN $session_ids
             WITH cs, d, target,
                  labels(target) AS target_labels
             ORDER BY d.mention_count DESC
             WITH cs,
                  collect({
                    entity_type: target_labels[0],
                    entity_id: COALESCE(target.path, target.name, target.id),
                    mention_count: d.mention_count
                  })[..10] AS top_discussed
             UNWIND top_discussed AS disc
             RETURN cs.id AS session_id,
                    disc.entity_type AS entity_type,
                    disc.entity_id AS entity_id,
                    disc.mention_count AS mention_count",
        )
        .param("session_ids", session_ids);

        let mut result = self.graph.execute(discussed_query).await?;
        let mut discussed = Vec::new();
        while let Some(row) = result.next().await? {
            discussed.push(ChatGraphDiscussed {
                session_id: row.get("session_id")?,
                entity_type: row
                    .get::<String>("entity_type")
                    .unwrap_or_else(|_| "File".to_string()),
                entity_id: row.get::<String>("entity_id").unwrap_or_default(),
                mention_count: row.get::<i64>("mention_count").unwrap_or(1),
            });
        }

        Ok((sessions, discussed))
    }
}
