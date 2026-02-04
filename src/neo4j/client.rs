//! Neo4j client for interacting with the knowledge graph

use super::models::*;
use anyhow::{Context, Result};
use neo4rs::{query, Graph, Query};
use std::sync::Arc;
use uuid::Uuid;

/// Client for Neo4j operations
pub struct Neo4jClient {
    graph: Arc<Graph>,
}

impl Neo4jClient {
    /// Create a new Neo4j client
    pub async fn new(uri: &str, user: &str, password: &str) -> Result<Self> {
        let graph = Graph::new(uri, user, password)
            .await
            .context("Failed to connect to Neo4j")?;

        let client = Self {
            graph: Arc::new(graph),
        };

        // Initialize schema
        client.init_schema().await?;

        Ok(client)
    }

    /// Initialize the graph schema with constraints and indexes
    async fn init_schema(&self) -> Result<()> {
        let constraints = vec![
            // Project constraints
            "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT project_slug IF NOT EXISTS FOR (p:Project) REQUIRE p.slug IS UNIQUE",
            // Code structure constraints
            "CREATE CONSTRAINT file_path IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE",
            "CREATE CONSTRAINT function_id IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT struct_id IF NOT EXISTS FOR (s:Struct) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT trait_id IF NOT EXISTS FOR (t:Trait) REQUIRE t.id IS UNIQUE",
            // Plan constraints
            "CREATE CONSTRAINT plan_id IF NOT EXISTS FOR (p:Plan) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT step_id IF NOT EXISTS FOR (s:Step) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT constraint_id IF NOT EXISTS FOR (c:Constraint) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE",
        ];

        let indexes = vec![
            "CREATE INDEX project_name IF NOT EXISTS FOR (p:Project) ON (p.name)",
            "CREATE INDEX file_language IF NOT EXISTS FOR (f:File) ON (f.language)",
            "CREATE INDEX file_project IF NOT EXISTS FOR (f:File) ON (f.project_id)",
            "CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name)",
            "CREATE INDEX struct_name IF NOT EXISTS FOR (s:Struct) ON (s.name)",
            "CREATE INDEX trait_name IF NOT EXISTS FOR (t:Trait) ON (t.name)",
            "CREATE INDEX enum_name IF NOT EXISTS FOR (e:Enum) ON (e.name)",
            "CREATE INDEX impl_for_type IF NOT EXISTS FOR (i:Impl) ON (i.for_type)",
            "CREATE INDEX task_status IF NOT EXISTS FOR (t:Task) ON (t.status)",
            "CREATE INDEX task_priority IF NOT EXISTS FOR (t:Task) ON (t.priority)",
            "CREATE INDEX step_status IF NOT EXISTS FOR (s:Step) ON (s.status)",
            "CREATE INDEX constraint_type IF NOT EXISTS FOR (c:Constraint) ON (c.constraint_type)",
            "CREATE INDEX plan_status IF NOT EXISTS FOR (p:Plan) ON (p.status)",
            "CREATE INDEX plan_project IF NOT EXISTS FOR (p:Plan) ON (p.project_id)",
        ];

        for constraint in constraints {
            if let Err(e) = self.graph.run(query(constraint)).await {
                tracing::warn!("Constraint may already exist: {}", e);
            }
        }

        for index in indexes {
            if let Err(e) = self.graph.run(query(index)).await {
                tracing::warn!("Index may already exist: {}", e);
            }
        }

        Ok(())
    }

    /// Execute a raw Cypher query
    pub async fn execute(&self, cypher: &str) -> Result<Vec<neo4rs::Row>> {
        let mut result = self.graph.execute(query(cypher)).await?;
        let mut rows = Vec::new();
        while let Some(row) = result.next().await? {
            rows.push(row);
        }
        Ok(rows)
    }

    /// Execute a parameterized Cypher query
    pub async fn execute_with_params(&self, q: Query) -> Result<Vec<neo4rs::Row>> {
        let mut result = self.graph.execute(q).await?;
        let mut rows = Vec::new();
        while let Some(row) = result.next().await? {
            rows.push(row);
        }
        Ok(rows)
    }

    // ========================================================================
    // Project operations
    // ========================================================================

    /// Create a new project
    pub async fn create_project(&self, project: &ProjectNode) -> Result<()> {
        let q = query(
            r#"
            CREATE (p:Project {
                id: $id,
                name: $name,
                slug: $slug,
                root_path: $root_path,
                description: $description,
                created_at: datetime($created_at)
            })
            "#,
        )
        .param("id", project.id.to_string())
        .param("name", project.name.clone())
        .param("slug", project.slug.clone())
        .param("root_path", project.root_path.clone())
        .param(
            "description",
            project.description.clone().unwrap_or_default(),
        )
        .param("created_at", project.created_at.to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a project by ID
    pub async fn get_project(&self, id: Uuid) -> Result<Option<ProjectNode>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            RETURN p
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            Ok(Some(self.node_to_project(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Get a project by slug
    pub async fn get_project_by_slug(&self, slug: &str) -> Result<Option<ProjectNode>> {
        let q = query(
            r#"
            MATCH (p:Project {slug: $slug})
            RETURN p
            "#,
        )
        .param("slug", slug);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            Ok(Some(self.node_to_project(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List all projects
    pub async fn list_projects(&self) -> Result<Vec<ProjectNode>> {
        let q = query(
            r#"
            MATCH (p:Project)
            RETURN p
            ORDER BY p.name
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut projects = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            projects.push(self.node_to_project(&node)?);
        }

        Ok(projects)
    }

    /// Update project last_synced timestamp
    pub async fn update_project_synced(&self, id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            SET p.last_synced = datetime($now)
            "#,
        )
        .param("id", id.to_string())
        .param("now", chrono::Utc::now().to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a project and all its data
    pub async fn delete_project(&self, id: Uuid) -> Result<()> {
        // Delete all files belonging to the project
        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            OPTIONAL MATCH (p)-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (f)-[:CONTAINS]->(symbol)
            DETACH DELETE symbol, f
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete all plans belonging to the project
        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            OPTIONAL MATCH (p)-[:HAS_PLAN]->(plan:Plan)
            OPTIONAL MATCH (plan)-[:HAS_TASK]->(task:Task)
            OPTIONAL MATCH (task)-[:INFORMED_BY]->(decision:Decision)
            DETACH DELETE decision, task, plan
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete the project itself
        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            DETACH DELETE p
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        Ok(())
    }

    /// Helper to convert Neo4j node to ProjectNode
    fn node_to_project(&self, node: &neo4rs::Node) -> Result<ProjectNode> {
        Ok(ProjectNode {
            id: node.get::<String>("id")?.parse()?,
            name: node.get("name")?,
            slug: node.get("slug")?,
            root_path: node.get("root_path")?,
            description: node.get("description").ok(),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            last_synced: node
                .get::<String>("last_synced")
                .ok()
                .and_then(|s| s.parse().ok()),
        })
    }

    // ========================================================================
    // File operations
    // ========================================================================

    /// Get all file paths for a project
    pub async fn get_project_file_paths(&self, project_id: Uuid) -> Result<Vec<String>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            RETURN f.path AS path
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut paths = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(path) = row.get::<String>("path") {
                paths.push(path);
            }
        }

        Ok(paths)
    }

    /// Delete files that are no longer on the filesystem
    /// Returns the number of files and symbols deleted
    pub async fn delete_stale_files(
        &self,
        project_id: Uuid,
        valid_paths: &[String],
    ) -> Result<(usize, usize)> {
        // First, count what we're about to delete
        let count_q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WHERE NOT f.path IN $valid_paths
            OPTIONAL MATCH (f)-[:CONTAINS]->(symbol)
            RETURN count(DISTINCT f) AS file_count, count(DISTINCT symbol) AS symbol_count
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("valid_paths", valid_paths.to_vec());

        let mut result = self.graph.execute(count_q).await?;
        let (file_count, symbol_count) = if let Some(row) = result.next().await? {
            let files: i64 = row.get("file_count").unwrap_or(0);
            let symbols: i64 = row.get("symbol_count").unwrap_or(0);
            (files as usize, symbols as usize)
        } else {
            (0, 0)
        };

        if file_count == 0 {
            return Ok((0, 0));
        }

        // Delete the stale files and their symbols
        let delete_q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WHERE NOT f.path IN $valid_paths
            OPTIONAL MATCH (f)-[:CONTAINS]->(symbol)
            DETACH DELETE symbol, f
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("valid_paths", valid_paths.to_vec());

        self.graph.run(delete_q).await?;

        tracing::info!(
            "Cleaned up {} stale files and {} symbols for project {}",
            file_count,
            symbol_count,
            project_id
        );

        Ok((file_count, symbol_count))
    }

    /// Link a file to a project (create CONTAINS relationship)
    pub async fn link_file_to_project(&self, file_path: &str, project_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})
            MATCH (f:File {path: $file_path})
            MERGE (p)-[:CONTAINS]->(f)
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("file_path", file_path);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Create or update a file node
    pub async fn upsert_file(&self, file: &FileNode) -> Result<()> {
        let q = query(
            r#"
            MERGE (f:File {path: $path})
            SET f.language = $language,
                f.hash = $hash,
                f.last_parsed = datetime($last_parsed),
                f.project_id = $project_id
            "#,
        )
        .param("path", file.path.clone())
        .param("language", file.language.clone())
        .param("hash", file.hash.clone())
        .param("last_parsed", file.last_parsed.to_rfc3339())
        .param(
            "project_id",
            file.project_id.map(|id| id.to_string()).unwrap_or_default(),
        );

        self.graph.run(q).await?;

        // Link to project if specified
        if let Some(project_id) = file.project_id {
            let q = query(
                r#"
                MATCH (p:Project {id: $project_id})
                MATCH (f:File {path: $path})
                MERGE (p)-[:CONTAINS]->(f)
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("path", file.path.clone());

            self.graph.run(q).await?;
        }

        Ok(())
    }

    /// Get a file by path
    pub async fn get_file(&self, path: &str) -> Result<Option<FileNode>> {
        let q = query(
            r#"
            MATCH (f:File {path: $path})
            RETURN f.path AS path, f.language AS language, f.hash AS hash,
                   f.last_parsed AS last_parsed, f.project_id AS project_id
            "#,
        )
        .param("path", path);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            Ok(Some(FileNode {
                path: row.get("path")?,
                language: row.get("language")?,
                hash: row.get("hash")?,
                last_parsed: row
                    .get::<String>("last_parsed")?
                    .parse()
                    .unwrap_or_else(|_| chrono::Utc::now()),
                project_id: row
                    .get::<String>("project_id")
                    .ok()
                    .and_then(|s| s.parse().ok()),
            }))
        } else {
            Ok(None)
        }
    }

    /// List files for a project
    pub async fn list_project_files(&self, project_id: Uuid) -> Result<Vec<FileNode>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            RETURN f.path AS path, f.language AS language, f.hash AS hash,
                   f.last_parsed AS last_parsed, f.project_id AS project_id
            ORDER BY f.path
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut files = Vec::new();

        while let Some(row) = result.next().await? {
            files.push(FileNode {
                path: row.get("path")?,
                language: row.get("language")?,
                hash: row.get("hash")?,
                last_parsed: row
                    .get::<String>("last_parsed")?
                    .parse()
                    .unwrap_or_else(|_| chrono::Utc::now()),
                project_id: Some(project_id),
            });
        }

        Ok(files)
    }

    // ========================================================================
    // Function operations
    // ========================================================================

    /// Create or update a function node
    pub async fn upsert_function(&self, func: &FunctionNode) -> Result<()> {
        let id = format!("{}:{}:{}", func.file_path, func.name, func.line_start);
        let q = query(
            r#"
            MERGE (f:Function {id: $id})
            SET f.name = $name,
                f.visibility = $visibility,
                f.params = $params,
                f.return_type = $return_type,
                f.generics = $generics,
                f.is_async = $is_async,
                f.is_unsafe = $is_unsafe,
                f.complexity = $complexity,
                f.file_path = $file_path,
                f.line_start = $line_start,
                f.line_end = $line_end,
                f.docstring = $docstring
            WITH f
            MATCH (file:File {path: $file_path})
            MERGE (file)-[:CONTAINS]->(f)
            "#,
        )
        .param("id", id)
        .param("name", func.name.clone())
        .param("visibility", format!("{:?}", func.visibility))
        .param("params", serde_json::to_string(&func.params)?)
        .param("return_type", func.return_type.clone().unwrap_or_default())
        .param("generics", func.generics.clone())
        .param("is_async", func.is_async)
        .param("is_unsafe", func.is_unsafe)
        .param("complexity", func.complexity as i64)
        .param("file_path", func.file_path.clone())
        .param("line_start", func.line_start as i64)
        .param("line_end", func.line_end as i64)
        .param("docstring", func.docstring.clone().unwrap_or_default());

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Struct operations
    // ========================================================================

    /// Create or update a struct node
    pub async fn upsert_struct(&self, s: &StructNode) -> Result<()> {
        let id = format!("{}:{}", s.file_path, s.name);
        let q = query(
            r#"
            MERGE (s:Struct {id: $id})
            SET s.name = $name,
                s.visibility = $visibility,
                s.generics = $generics,
                s.file_path = $file_path,
                s.line_start = $line_start,
                s.line_end = $line_end,
                s.docstring = $docstring
            WITH s
            MATCH (file:File {path: $file_path})
            MERGE (file)-[:CONTAINS]->(s)
            "#,
        )
        .param("id", id)
        .param("name", s.name.clone())
        .param("visibility", format!("{:?}", s.visibility))
        .param("generics", s.generics.clone())
        .param("file_path", s.file_path.clone())
        .param("line_start", s.line_start as i64)
        .param("line_end", s.line_end as i64)
        .param("docstring", s.docstring.clone().unwrap_or_default());

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Trait operations
    // ========================================================================

    /// Create or update a trait node
    pub async fn upsert_trait(&self, t: &TraitNode) -> Result<()> {
        let id = format!("{}:{}", t.file_path, t.name);
        let q = query(
            r#"
            MERGE (t:Trait {id: $id})
            SET t.name = $name,
                t.visibility = $visibility,
                t.generics = $generics,
                t.file_path = $file_path,
                t.line_start = $line_start,
                t.line_end = $line_end,
                t.docstring = $docstring
            WITH t
            MATCH (file:File {path: $file_path})
            MERGE (file)-[:CONTAINS]->(t)
            "#,
        )
        .param("id", id)
        .param("name", t.name.clone())
        .param("visibility", format!("{:?}", t.visibility))
        .param("generics", t.generics.clone())
        .param("file_path", t.file_path.clone())
        .param("line_start", t.line_start as i64)
        .param("line_end", t.line_end as i64)
        .param("docstring", t.docstring.clone().unwrap_or_default());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Find a trait by name (searches across all files)
    pub async fn find_trait_by_name(&self, name: &str) -> Result<Option<String>> {
        let q = query(
            r#"
            MATCH (t:Trait {name: $name})
            RETURN t.id AS id
            LIMIT 1
            "#,
        )
        .param("name", name);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            Ok(Some(row.get("id")?))
        } else {
            Ok(None)
        }
    }

    // ========================================================================
    // Enum operations
    // ========================================================================

    /// Create or update an enum node
    pub async fn upsert_enum(&self, e: &EnumNode) -> Result<()> {
        let id = format!("{}:{}", e.file_path, e.name);
        let q = query(
            r#"
            MERGE (e:Enum {id: $id})
            SET e.name = $name,
                e.visibility = $visibility,
                e.variants = $variants,
                e.file_path = $file_path,
                e.line_start = $line_start,
                e.line_end = $line_end,
                e.docstring = $docstring
            WITH e
            MATCH (file:File {path: $file_path})
            MERGE (file)-[:CONTAINS]->(e)
            "#,
        )
        .param("id", id)
        .param("name", e.name.clone())
        .param("visibility", format!("{:?}", e.visibility))
        .param("variants", e.variants.clone())
        .param("file_path", e.file_path.clone())
        .param("line_start", e.line_start as i64)
        .param("line_end", e.line_end as i64)
        .param("docstring", e.docstring.clone().unwrap_or_default());

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Impl operations
    // ========================================================================

    /// Create or update an impl block node
    pub async fn upsert_impl(&self, impl_node: &ImplNode) -> Result<()> {
        let id = format!(
            "{}:impl:{}:{}",
            impl_node.file_path,
            impl_node.for_type,
            impl_node.trait_name.as_deref().unwrap_or("self")
        );

        let q = query(
            r#"
            MERGE (i:Impl {id: $id})
            SET i.for_type = $for_type,
                i.trait_name = $trait_name,
                i.generics = $generics,
                i.where_clause = $where_clause,
                i.file_path = $file_path,
                i.line_start = $line_start,
                i.line_end = $line_end
            WITH i
            MATCH (file:File {path: $file_path})
            MERGE (file)-[:CONTAINS]->(i)
            "#,
        )
        .param("id", id.clone())
        .param("for_type", impl_node.for_type.clone())
        .param(
            "trait_name",
            impl_node.trait_name.clone().unwrap_or_default(),
        )
        .param("generics", impl_node.generics.clone())
        .param(
            "where_clause",
            impl_node.where_clause.clone().unwrap_or_default(),
        )
        .param("file_path", impl_node.file_path.clone())
        .param("line_start", impl_node.line_start as i64)
        .param("line_end", impl_node.line_end as i64);

        self.graph.run(q).await?;

        // Create IMPLEMENTS_FOR relationship to the struct/enum
        let q = query(
            r#"
            MATCH (i:Impl {id: $impl_id})
            MATCH (s:Struct {name: $type_name})
            WHERE s.file_path = $file_path OR s.name = $type_name
            MERGE (i)-[:IMPLEMENTS_FOR]->(s)
            "#,
        )
        .param("impl_id", id.clone())
        .param("type_name", impl_node.for_type.clone())
        .param("file_path", impl_node.file_path.clone());

        // Try struct first, ignore error if not found
        let _ = self.graph.run(q).await;

        // Try enum too
        let q = query(
            r#"
            MATCH (i:Impl {id: $impl_id})
            MATCH (e:Enum {name: $type_name})
            WHERE e.file_path = $file_path OR e.name = $type_name
            MERGE (i)-[:IMPLEMENTS_FOR]->(e)
            "#,
        )
        .param("impl_id", id.clone())
        .param("type_name", impl_node.for_type.clone())
        .param("file_path", impl_node.file_path.clone());

        let _ = self.graph.run(q).await;

        // Create IMPLEMENTS_TRAIT relationship if this is a trait impl
        if let Some(ref trait_name) = impl_node.trait_name {
            // First try to link to existing local trait
            let q = query(
                r#"
                MATCH (i:Impl {id: $impl_id})
                MATCH (t:Trait {name: $trait_name})
                WHERE t.is_external IS NULL OR t.is_external = false
                MERGE (i)-[:IMPLEMENTS_TRAIT]->(t)
                RETURN count(*) AS linked
                "#,
            )
            .param("impl_id", id.clone())
            .param("trait_name", trait_name.clone());

            let rows = self.execute_with_params(q).await?;
            let linked: i64 = rows
                .first()
                .and_then(|r| r.get("linked").ok())
                .unwrap_or(0);

            // If no local trait found, create/link to external trait
            if linked == 0 {
                let (simple_name, source) = Self::parse_trait_path(trait_name);
                let external_id = format!("external:trait:{}", trait_name);

                let q = query(
                    r#"
                    MERGE (t:Trait {id: $trait_id})
                    ON CREATE SET
                        t.name = $name,
                        t.full_path = $full_path,
                        t.is_external = true,
                        t.source = $source,
                        t.visibility = 'public',
                        t.generics = [],
                        t.file_path = '',
                        t.line_start = 0,
                        t.line_end = 0
                    ON MATCH SET
                        t.source = CASE WHEN t.source = 'unknown' THEN $source ELSE t.source END
                    WITH t
                    MATCH (i:Impl {id: $impl_id})
                    MERGE (i)-[:IMPLEMENTS_TRAIT]->(t)
                    "#,
                )
                .param("trait_id", external_id)
                .param("name", simple_name)
                .param("full_path", trait_name.clone())
                .param("source", source)
                .param("impl_id", id);

                let _ = self.graph.run(q).await;
            }
        }

        Ok(())
    }

    /// Parse a trait path to extract the simple name and source crate
    ///
    /// Examples:
    /// - "Debug" -> ("Debug", "std")
    /// - "Clone" -> ("Clone", "std")
    /// - "Serialize" -> ("Serialize", "serde")
    /// - "serde::Serialize" -> ("Serialize", "serde")
    /// - "std::fmt::Display" -> ("Display", "std")
    /// - "tokio::io::AsyncRead" -> ("AsyncRead", "tokio")
    fn parse_trait_path(trait_path: &str) -> (String, String) {
        let parts: Vec<&str> = trait_path.split("::").collect();

        if parts.len() == 1 {
            // Simple name - check known trait sources
            let name = parts[0].to_string();
            let source = Self::get_trait_source(&name).to_string();
            (name, source)
        } else {
            // Full path - first part is the crate
            let name = parts.last().unwrap_or(&"").to_string();
            let source = parts[0].to_string();
            (name, source)
        }
    }

    /// Determine the source crate for a trait name
    fn get_trait_source(name: &str) -> &'static str {
        // Standard library traits
        if matches!(
            name,
            "Debug"
                | "Display"
                | "Clone"
                | "Copy"
                | "Default"
                | "PartialEq"
                | "Eq"
                | "PartialOrd"
                | "Ord"
                | "Hash"
                | "From"
                | "Into"
                | "TryFrom"
                | "TryInto"
                | "AsRef"
                | "AsMut"
                | "Deref"
                | "DerefMut"
                | "Drop"
                | "Send"
                | "Sync"
                | "Sized"
                | "Unpin"
                | "Iterator"
                | "IntoIterator"
                | "ExactSizeIterator"
                | "DoubleEndedIterator"
                | "Extend"
                | "FromIterator"
                | "Read"
                | "Write"
                | "Seek"
                | "BufRead"
                | "Error"
                | "Future"
                | "Stream"
                | "FnOnce"
                | "FnMut"
                | "Fn"
                | "Add"
                | "Sub"
                | "Mul"
                | "Div"
                | "Rem"
                | "Neg"
                | "Not"
                | "BitAnd"
                | "BitOr"
                | "BitXor"
                | "Shl"
                | "Shr"
                | "Index"
                | "IndexMut"
        ) {
            return "std";
        }

        // serde traits
        if matches!(name, "Serialize" | "Deserialize" | "Serializer" | "Deserializer") {
            return "serde";
        }

        // tokio/async traits
        if matches!(
            name,
            "AsyncRead" | "AsyncWrite" | "AsyncSeek" | "AsyncBufRead"
        ) {
            return "tokio";
        }

        // anyhow/thiserror
        if matches!(name, "Context") {
            return "anyhow";
        }

        // tracing
        if matches!(name, "Instrument" | "Subscriber") {
            return "tracing";
        }

        // axum/tower
        if matches!(name, "IntoResponse" | "FromRequest" | "FromRequestParts" | "Service" | "Layer") {
            return "axum";
        }

        "unknown"
    }

    // ========================================================================
    // Import/dependency operations
    // ========================================================================

    /// Create an import relationship between files
    pub async fn create_import_relationship(
        &self,
        from_file: &str,
        to_file: &str,
        import_path: &str,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (from:File {path: $from_file})
            MATCH (to:File {path: $to_file})
            MERGE (from)-[r:IMPORTS]->(to)
            SET r.import_path = $import_path
            "#,
        )
        .param("from_file", from_file)
        .param("to_file", to_file)
        .param("import_path", import_path);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Store an import node (for tracking even unresolved imports)
    pub async fn upsert_import(&self, import: &ImportNode) -> Result<()> {
        let id = format!("{}:{}:{}", import.file_path, import.line, import.path);
        let q = query(
            r#"
            MERGE (i:Import {id: $id})
            SET i.path = $path,
                i.alias = $alias,
                i.items = $items,
                i.file_path = $file_path,
                i.line = $line
            WITH i
            MATCH (file:File {path: $file_path})
            MERGE (file)-[:HAS_IMPORT]->(i)
            "#,
        )
        .param("id", id)
        .param("path", import.path.clone())
        .param("alias", import.alias.clone().unwrap_or_default())
        .param("items", import.items.clone())
        .param("file_path", import.file_path.clone())
        .param("line", import.line as i64);

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Function call graph operations
    // ========================================================================

    /// Create a CALLS relationship between functions
    pub async fn create_call_relationship(&self, caller_id: &str, callee_name: &str) -> Result<()> {
        // Try to find the callee function by name
        let q = query(
            r#"
            MATCH (caller:Function {id: $caller_id})
            MATCH (callee:Function {name: $callee_name})
            MERGE (caller)-[:CALLS]->(callee)
            "#,
        )
        .param("caller_id", caller_id)
        .param("callee_name", callee_name);

        // Ignore errors if callee not found (might be external)
        let _ = self.graph.run(q).await;
        Ok(())
    }

    /// Get all functions called by a function
    pub async fn get_callees(&self, function_id: &str, depth: u32) -> Result<Vec<FunctionNode>> {
        let q = query(&format!(
            r#"
            MATCH (f:Function {{id: $id}})-[:CALLS*1..{}]->(callee:Function)
            RETURN DISTINCT callee
            "#,
            depth
        ))
        .param("id", function_id);

        let mut result = self.graph.execute(q).await?;
        let mut functions = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("callee")?;
            functions.push(FunctionNode {
                name: node.get("name")?,
                visibility: Visibility::Private,
                params: vec![],
                return_type: node.get("return_type").ok(),
                generics: vec![],
                is_async: node.get("is_async").unwrap_or(false),
                is_unsafe: node.get("is_unsafe").unwrap_or(false),
                complexity: node.get::<i64>("complexity").unwrap_or(0) as u32,
                file_path: node.get("file_path")?,
                line_start: node.get::<i64>("line_start")? as u32,
                line_end: node.get::<i64>("line_end")? as u32,
                docstring: node.get("docstring").ok(),
            });
        }

        Ok(functions)
    }

    // ========================================================================
    // Type usage operations
    // ========================================================================

    /// Create a USES_TYPE relationship from a function to a type
    pub async fn create_uses_type_relationship(
        &self,
        function_id: &str,
        type_name: &str,
    ) -> Result<()> {
        // Try to link to Struct
        let q = query(
            r#"
            MATCH (f:Function {id: $function_id})
            MATCH (t:Struct {name: $type_name})
            MERGE (f)-[:USES_TYPE]->(t)
            "#,
        )
        .param("function_id", function_id)
        .param("type_name", type_name);
        let _ = self.graph.run(q).await;

        // Try to link to Enum
        let q = query(
            r#"
            MATCH (f:Function {id: $function_id})
            MATCH (t:Enum {name: $type_name})
            MERGE (f)-[:USES_TYPE]->(t)
            "#,
        )
        .param("function_id", function_id)
        .param("type_name", type_name);
        let _ = self.graph.run(q).await;

        // Try to link to Trait
        let q = query(
            r#"
            MATCH (f:Function {id: $function_id})
            MATCH (t:Trait {name: $type_name})
            MERGE (f)-[:USES_TYPE]->(t)
            "#,
        )
        .param("function_id", function_id)
        .param("type_name", type_name);
        let _ = self.graph.run(q).await;

        Ok(())
    }

    /// Find types that implement a specific trait
    pub async fn find_trait_implementors(&self, trait_name: &str) -> Result<Vec<String>> {
        let q = query(
            r#"
            MATCH (i:Impl)-[:IMPLEMENTS_TRAIT]->(t:Trait {name: $trait_name})
            MATCH (i)-[:IMPLEMENTS_FOR]->(type)
            RETURN DISTINCT type.name AS name
            "#,
        )
        .param("trait_name", trait_name);

        let mut result = self.graph.execute(q).await?;
        let mut implementors = Vec::new();

        while let Some(row) = result.next().await? {
            implementors.push(row.get("name")?);
        }

        Ok(implementors)
    }

    /// Get all traits implemented by a type
    pub async fn get_type_traits(&self, type_name: &str) -> Result<Vec<String>> {
        let q = query(
            r#"
            MATCH (type {name: $type_name})<-[:IMPLEMENTS_FOR]-(i:Impl)-[:IMPLEMENTS_TRAIT]->(t:Trait)
            RETURN DISTINCT t.name AS name
            "#,
        )
        .param("type_name", type_name);

        let mut result = self.graph.execute(q).await?;
        let mut traits = Vec::new();

        while let Some(row) = result.next().await? {
            traits.push(row.get("name")?);
        }

        Ok(traits)
    }

    // ========================================================================
    // Plan operations
    // ========================================================================

    /// Create a new plan
    pub async fn create_plan(&self, plan: &PlanNode) -> Result<()> {
        let q = query(
            r#"
            CREATE (p:Plan {
                id: $id,
                title: $title,
                description: $description,
                status: $status,
                created_at: datetime($created_at),
                created_by: $created_by,
                priority: $priority,
                project_id: $project_id
            })
            "#,
        )
        .param("id", plan.id.to_string())
        .param("title", plan.title.clone())
        .param("description", plan.description.clone())
        .param("status", format!("{:?}", plan.status))
        .param("created_at", plan.created_at.to_rfc3339())
        .param("created_by", plan.created_by.clone())
        .param("priority", plan.priority as i64)
        .param(
            "project_id",
            plan.project_id.map(|id| id.to_string()).unwrap_or_default(),
        );

        self.graph.run(q).await?;

        // Link to project if specified
        if let Some(project_id) = plan.project_id {
            let q = query(
                r#"
                MATCH (project:Project {id: $project_id})
                MATCH (plan:Plan {id: $plan_id})
                MERGE (project)-[:HAS_PLAN]->(plan)
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("plan_id", plan.id.to_string());

            self.graph.run(q).await?;
        }

        Ok(())
    }

    /// Get a plan by ID
    pub async fn get_plan(&self, id: Uuid) -> Result<Option<PlanNode>> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})
            RETURN p
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            Ok(Some(self.node_to_plan(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Helper to convert Neo4j node to PlanNode
    fn node_to_plan(&self, node: &neo4rs::Node) -> Result<PlanNode> {
        Ok(PlanNode {
            id: node.get::<String>("id")?.parse()?,
            title: node.get("title")?,
            description: node.get("description")?,
            status: serde_json::from_str(&format!(
                "\"{}\"",
                node.get::<String>("status")?.to_lowercase()
            ))
            .unwrap_or(PlanStatus::Draft),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            created_by: node.get("created_by")?,
            priority: node.get::<i64>("priority")? as i32,
            project_id: node.get::<String>("project_id").ok().and_then(|s| {
                if s.is_empty() {
                    None
                } else {
                    s.parse().ok()
                }
            }),
        })
    }

    /// List all active plans
    pub async fn list_active_plans(&self) -> Result<Vec<PlanNode>> {
        let q = query(
            r#"
            MATCH (p:Plan)
            WHERE p.status IN ['Draft', 'Approved', 'InProgress']
            RETURN p
            ORDER BY p.priority DESC, p.created_at DESC
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut plans = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            plans.push(self.node_to_plan(&node)?);
        }

        Ok(plans)
    }

    /// List active plans for a specific project
    pub async fn list_project_plans(&self, project_id: Uuid) -> Result<Vec<PlanNode>> {
        let q = query(
            r#"
            MATCH (project:Project {id: $project_id})-[:HAS_PLAN]->(p:Plan)
            WHERE p.status IN ['Draft', 'Approved', 'InProgress']
            RETURN p
            ORDER BY p.priority DESC, p.created_at DESC
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut plans = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            plans.push(self.node_to_plan(&node)?);
        }

        Ok(plans)
    }

    /// Update plan status
    pub async fn update_plan_status(&self, id: Uuid, status: PlanStatus) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})
            SET p.status = $status
            "#,
        )
        .param("id", id.to_string())
        .param("status", format!("{:?}", status));

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Task operations
    // ========================================================================

    /// Create a task for a plan
    pub async fn create_task(&self, plan_id: Uuid, task: &TaskNode) -> Result<()> {
        let now = task.created_at.to_rfc3339();
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})
            CREATE (t:Task {
                id: $id,
                title: $title,
                description: $description,
                status: $status,
                priority: $priority,
                tags: $tags,
                acceptance_criteria: $acceptance_criteria,
                affected_files: $affected_files,
                estimated_complexity: $estimated_complexity,
                created_at: datetime($created_at),
                updated_at: datetime($updated_at)
            })
            CREATE (p)-[:HAS_TASK]->(t)
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("id", task.id.to_string())
        .param("title", task.title.clone().unwrap_or_default())
        .param("description", task.description.clone())
        .param("status", format!("{:?}", task.status))
        .param("priority", task.priority.unwrap_or(0) as i64)
        .param("tags", task.tags.clone())
        .param("acceptance_criteria", task.acceptance_criteria.clone())
        .param("affected_files", task.affected_files.clone())
        .param(
            "estimated_complexity",
            task.estimated_complexity.map(|c| c as i64).unwrap_or(0),
        )
        .param("created_at", now.clone())
        .param("updated_at", now);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get tasks for a plan
    pub async fn get_plan_tasks(&self, plan_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:HAS_TASK]->(t:Task)
            RETURN t
            ORDER BY COALESCE(t.priority, 0) DESC, t.created_at
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// Helper to convert Neo4j node to TaskNode
    fn node_to_task(&self, node: &neo4rs::Node) -> Result<TaskNode> {
        Ok(TaskNode {
            id: node.get::<String>("id")?.parse()?,
            title: node.get::<String>("title").ok().filter(|s| !s.is_empty()),
            description: node.get("description")?,
            status: serde_json::from_str(&format!(
                "\"{}\"",
                node.get::<String>("status")?.to_lowercase()
            ))
            .unwrap_or(TaskStatus::Pending),
            assigned_to: node.get("assigned_to").ok(),
            priority: node.get::<i64>("priority").ok().map(|v| v as i32),
            tags: node.get("tags").unwrap_or_default(),
            acceptance_criteria: node.get("acceptance_criteria").unwrap_or_default(),
            affected_files: node.get("affected_files").unwrap_or_default(),
            estimated_complexity: node
                .get::<i64>("estimated_complexity")
                .ok()
                .filter(|&v| v > 0)
                .map(|v| v as u32),
            actual_complexity: node
                .get::<i64>("actual_complexity")
                .ok()
                .filter(|&v| v > 0)
                .map(|v| v as u32),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            updated_at: node
                .get::<String>("updated_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            started_at: node
                .get::<String>("started_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            completed_at: node
                .get::<String>("completed_at")
                .ok()
                .and_then(|s| s.parse().ok()),
        })
    }

    /// Update task status
    pub async fn update_task_status(&self, task_id: Uuid, status: TaskStatus) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        let q = match status {
            TaskStatus::InProgress => query(
                r#"
                MATCH (t:Task {id: $id})
                SET t.status = $status,
                    t.started_at = datetime($now),
                    t.updated_at = datetime($now)
                "#,
            ),
            TaskStatus::Completed | TaskStatus::Failed => query(
                r#"
                MATCH (t:Task {id: $id})
                SET t.status = $status,
                    t.completed_at = datetime($now),
                    t.updated_at = datetime($now)
                "#,
            ),
            _ => query(
                r#"
                MATCH (t:Task {id: $id})
                SET t.status = $status,
                    t.updated_at = datetime($now)
                "#,
            ),
        }
        .param("id", task_id.to_string())
        .param("status", format!("{:?}", status))
        .param("now", now);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Assign task to an agent
    pub async fn assign_task(&self, task_id: Uuid, agent_id: &str) -> Result<()> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})
            MATCH (a:Agent {id: $agent_id})
            SET t.assigned_to = $agent_id
            MERGE (a)-[:WORKING_ON]->(t)
            "#,
        )
        .param("task_id", task_id.to_string())
        .param("agent_id", agent_id);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Add task dependency
    pub async fn add_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})
            MATCH (dep:Task {id: $depends_on_id})
            MERGE (t)-[:DEPENDS_ON]->(dep)
            "#,
        )
        .param("task_id", task_id.to_string())
        .param("depends_on_id", depends_on_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get next available task (no unfinished dependencies)
    pub async fn get_next_available_task(&self, plan_id: Uuid) -> Result<Option<TaskNode>> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:HAS_TASK]->(t:Task {status: 'Pending'})
            WHERE NOT EXISTS {
                MATCH (t)-[:DEPENDS_ON]->(dep:Task)
                WHERE dep.status <> 'Completed'
            }
            RETURN t
            ORDER BY COALESCE(t.priority, 0) DESC, t.created_at
            LIMIT 1
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            Ok(Some(self.node_to_task(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Get a single task by ID
    pub async fn get_task(&self, task_id: Uuid) -> Result<Option<TaskNode>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $id})
            RETURN t
            "#,
        )
        .param("id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            Ok(Some(self.node_to_task(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Update a task with new values
    pub async fn update_task(
        &self,
        task_id: Uuid,
        updates: &crate::plan::models::UpdateTaskRequest,
    ) -> Result<()> {
        let mut set_clauses = Vec::new();

        if updates.title.is_some() {
            set_clauses.push("t.title = $title");
        }
        if updates.description.is_some() {
            set_clauses.push("t.description = $description");
        }
        if updates.priority.is_some() {
            set_clauses.push("t.priority = $priority");
        }
        if updates.tags.is_some() {
            set_clauses.push("t.tags = $tags");
        }
        if updates.acceptance_criteria.is_some() {
            set_clauses.push("t.acceptance_criteria = $acceptance_criteria");
        }
        if updates.affected_files.is_some() {
            set_clauses.push("t.affected_files = $affected_files");
        }
        if updates.actual_complexity.is_some() {
            set_clauses.push("t.actual_complexity = $actual_complexity");
        }
        if updates.assigned_to.is_some() {
            set_clauses.push("t.assigned_to = $assigned_to");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        // Always update updated_at
        set_clauses.push("t.updated_at = datetime($updated_at)");

        let cypher = format!("MATCH (t:Task {{id: $id}}) SET {}", set_clauses.join(", "));

        let mut q = query(&cypher)
            .param("id", task_id.to_string())
            .param("updated_at", chrono::Utc::now().to_rfc3339());

        if let Some(ref title) = updates.title {
            q = q.param("title", title.clone());
        }
        if let Some(ref desc) = updates.description {
            q = q.param("description", desc.clone());
        }
        if let Some(priority) = updates.priority {
            q = q.param("priority", priority as i64);
        }
        if let Some(ref tags) = updates.tags {
            q = q.param("tags", tags.clone());
        }
        if let Some(ref criteria) = updates.acceptance_criteria {
            q = q.param("acceptance_criteria", criteria.clone());
        }
        if let Some(ref files) = updates.affected_files {
            q = q.param("affected_files", files.clone());
        }
        if let Some(complexity) = updates.actual_complexity {
            q = q.param("actual_complexity", complexity as i64);
        }
        if let Some(ref assigned) = updates.assigned_to {
            q = q.param("assigned_to", assigned.clone());
        }

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Step operations
    // ========================================================================

    /// Create a step for a task
    pub async fn create_step(&self, task_id: Uuid, step: &StepNode) -> Result<()> {
        let now = step.created_at.to_rfc3339();
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})
            CREATE (s:Step {
                id: $id,
                order: $order,
                description: $description,
                status: $status,
                verification: $verification,
                created_at: datetime($created_at),
                updated_at: datetime($updated_at)
            })
            CREATE (t)-[:HAS_STEP]->(s)
            "#,
        )
        .param("task_id", task_id.to_string())
        .param("id", step.id.to_string())
        .param("order", step.order as i64)
        .param("description", step.description.clone())
        .param("status", format!("{:?}", step.status))
        .param(
            "verification",
            step.verification.clone().unwrap_or_default(),
        )
        .param("created_at", now.clone())
        .param("updated_at", now);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get steps for a task
    pub async fn get_task_steps(&self, task_id: Uuid) -> Result<Vec<StepNode>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})-[:HAS_STEP]->(s:Step)
            RETURN s
            ORDER BY s.order
            "#,
        )
        .param("task_id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut steps = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            steps.push(StepNode {
                id: node.get::<String>("id")?.parse()?,
                order: node.get::<i64>("order")? as u32,
                description: node.get("description")?,
                status: serde_json::from_str(&format!(
                    "\"{}\"",
                    node.get::<String>("status")?.to_lowercase()
                ))
                .unwrap_or(StepStatus::Pending),
                verification: node
                    .get::<String>("verification")
                    .ok()
                    .filter(|s| !s.is_empty()),
                created_at: node
                    .get::<String>("created_at")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(chrono::Utc::now),
                updated_at: node
                    .get::<String>("updated_at")
                    .ok()
                    .and_then(|s| s.parse().ok()),
                completed_at: node
                    .get::<String>("completed_at")
                    .ok()
                    .and_then(|s| s.parse().ok()),
            });
        }

        Ok(steps)
    }

    /// Update step status
    pub async fn update_step_status(&self, step_id: Uuid, status: StepStatus) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        let q = match status {
            StepStatus::Completed | StepStatus::Skipped => query(
                r#"
                MATCH (s:Step {id: $id})
                SET s.status = $status,
                    s.completed_at = datetime($now),
                    s.updated_at = datetime($now)
                "#,
            ),
            _ => query(
                r#"
                MATCH (s:Step {id: $id})
                SET s.status = $status,
                    s.updated_at = datetime($now)
                "#,
            ),
        }
        .param("id", step_id.to_string())
        .param("status", format!("{:?}", status))
        .param("now", now);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get count of completed steps for a task
    pub async fn get_task_step_progress(&self, task_id: Uuid) -> Result<(u32, u32)> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})-[:HAS_STEP]->(s:Step)
            RETURN count(s) AS total,
                   sum(CASE WHEN s.status = 'Completed' THEN 1 ELSE 0 END) AS completed
            "#,
        )
        .param("task_id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let total: i64 = row.get("total")?;
            let completed: i64 = row.get("completed")?;
            Ok((completed as u32, total as u32))
        } else {
            Ok((0, 0))
        }
    }

    // ========================================================================
    // Constraint operations
    // ========================================================================

    /// Create a constraint for a plan
    pub async fn create_constraint(
        &self,
        plan_id: Uuid,
        constraint: &ConstraintNode,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})
            CREATE (c:Constraint {
                id: $id,
                constraint_type: $constraint_type,
                description: $description,
                enforced_by: $enforced_by
            })
            CREATE (p)-[:CONSTRAINED_BY]->(c)
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("id", constraint.id.to_string())
        .param(
            "constraint_type",
            format!("{:?}", constraint.constraint_type),
        )
        .param("description", constraint.description.clone())
        .param(
            "enforced_by",
            constraint.enforced_by.clone().unwrap_or_default(),
        );

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get constraints for a plan
    pub async fn get_plan_constraints(&self, plan_id: Uuid) -> Result<Vec<ConstraintNode>> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:CONSTRAINED_BY]->(c:Constraint)
            RETURN c
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut constraints = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            constraints.push(ConstraintNode {
                id: node.get::<String>("id")?.parse()?,
                constraint_type: serde_json::from_str(&format!(
                    "\"{}\"",
                    node.get::<String>("constraint_type")?.to_lowercase()
                ))
                .unwrap_or(ConstraintType::Other),
                description: node.get("description")?,
                enforced_by: node
                    .get::<String>("enforced_by")
                    .ok()
                    .filter(|s| !s.is_empty()),
            });
        }

        Ok(constraints)
    }

    /// Delete a constraint
    pub async fn delete_constraint(&self, constraint_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (c:Constraint {id: $id})
            DETACH DELETE c
            "#,
        )
        .param("id", constraint_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Decision operations
    // ========================================================================

    /// Record a decision
    pub async fn create_decision(&self, task_id: Uuid, decision: &DecisionNode) -> Result<()> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})
            CREATE (d:Decision {
                id: $id,
                description: $description,
                rationale: $rationale,
                alternatives: $alternatives,
                chosen_option: $chosen_option,
                decided_by: $decided_by,
                decided_at: datetime($decided_at)
            })
            CREATE (t)-[:INFORMED_BY]->(d)
            "#,
        )
        .param("task_id", task_id.to_string())
        .param("id", decision.id.to_string())
        .param("description", decision.description.clone())
        .param("rationale", decision.rationale.clone())
        .param("alternatives", decision.alternatives.clone())
        .param(
            "chosen_option",
            decision.chosen_option.clone().unwrap_or_default(),
        )
        .param("decided_by", decision.decided_by.clone())
        .param("decided_at", decision.decided_at.to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Impact analysis
    // ========================================================================

    /// Find all files that depend on a given file
    pub async fn find_dependent_files(&self, file_path: &str, depth: u32) -> Result<Vec<String>> {
        let q = query(&format!(
            r#"
            MATCH (f:File {{path: $path}})<-[:IMPORTS*1..{}]-(dependent:File)
            RETURN DISTINCT dependent.path AS path
            "#,
            depth
        ))
        .param("path", file_path);

        let mut result = self.graph.execute(q).await?;
        let mut paths = Vec::new();

        while let Some(row) = result.next().await? {
            paths.push(row.get("path")?);
        }

        Ok(paths)
    }

    /// Find all functions that call a given function
    pub async fn find_callers(&self, function_id: &str) -> Result<Vec<FunctionNode>> {
        let q = query(
            r#"
            MATCH (caller:Function)-[:CALLS]->(f:Function {id: $id})
            RETURN caller
            "#,
        )
        .param("id", function_id);

        let mut result = self.graph.execute(q).await?;
        let mut functions = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("caller")?;
            functions.push(FunctionNode {
                name: node.get("name")?,
                visibility: Visibility::Private, // Simplified for now
                params: vec![],
                return_type: node.get("return_type").ok(),
                generics: vec![],
                is_async: node.get("is_async").unwrap_or(false),
                is_unsafe: node.get("is_unsafe").unwrap_or(false),
                complexity: node.get::<i64>("complexity").unwrap_or(0) as u32,
                file_path: node.get("file_path")?,
                line_start: node.get::<i64>("line_start")? as u32,
                line_end: node.get::<i64>("line_end")? as u32,
                docstring: node.get("docstring").ok(),
            });
        }

        Ok(functions)
    }

    /// Link a task to files it modifies
    pub async fn link_task_to_files(&self, task_id: Uuid, file_paths: &[String]) -> Result<()> {
        for path in file_paths {
            let q = query(
                r#"
                MATCH (t:Task {id: $task_id})
                MATCH (f:File {path: $path})
                MERGE (t)-[:MODIFIES]->(f)
                "#,
            )
            .param("task_id", task_id.to_string())
            .param("path", path.clone());

            self.graph.run(q).await?;
        }
        Ok(())
    }
}
