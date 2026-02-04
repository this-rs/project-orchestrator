//! Neo4j client for interacting with the knowledge graph

use super::models::*;
use crate::notes::{
    EntityType, Note, NoteAnchor, NoteChange, NoteFilters, NoteImportance, NoteScope, NoteStatus,
    NoteType, PropagatedNote,
};
use anyhow::{Context, Result};
use neo4rs::{query, Graph, Query};
use std::sync::Arc;
use uuid::Uuid;

/// Client for Neo4j operations
pub struct Neo4jClient {
    graph: Arc<Graph>,
}

/// Convert PascalCase to snake_case (e.g., "InProgress" -> "in_progress")
fn pascal_to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}

/// Convert snake_case to PascalCase (e.g., "in_progress" -> "InProgress")
fn snake_to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut c = word.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().chain(c).collect(),
            }
        })
        .collect()
}

/// Builder for dynamic WHERE clauses in Cypher queries
#[derive(Default)]
pub struct WhereBuilder {
    conditions: Vec<String>,
}

impl WhereBuilder {
    /// Create a new empty WhereBuilder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a status filter (converts snake_case to PascalCase for Neo4j)
    pub fn add_status_filter(&mut self, alias: &str, statuses: Option<Vec<String>>) -> &mut Self {
        if let Some(statuses) = statuses {
            if !statuses.is_empty() {
                let pascal_statuses: Vec<String> = statuses
                    .iter()
                    .map(|s| snake_to_pascal_case(s))
                    .collect();
                self.conditions.push(format!(
                    "{}.status IN [{}]",
                    alias,
                    pascal_statuses
                        .iter()
                        .map(|s| format!("'{}'", s))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
        self
    }

    /// Add a priority range filter
    pub fn add_priority_filter(
        &mut self,
        alias: &str,
        min: Option<i32>,
        max: Option<i32>,
    ) -> &mut Self {
        if let Some(min) = min {
            self.conditions
                .push(format!("COALESCE({}.priority, 0) >= {}", alias, min));
        }
        if let Some(max) = max {
            self.conditions
                .push(format!("COALESCE({}.priority, 0) <= {}", alias, max));
        }
        self
    }

    /// Add a tags filter (all specified tags must be present)
    pub fn add_tags_filter(&mut self, alias: &str, tags: Option<Vec<String>>) -> &mut Self {
        if let Some(tags) = tags {
            for tag in tags {
                self.conditions
                    .push(format!("'{}' IN {}.tags", tag, alias));
            }
        }
        self
    }

    /// Add an assigned_to filter
    pub fn add_assigned_to_filter(&mut self, alias: &str, assigned_to: Option<&str>) -> &mut Self {
        if let Some(assigned) = assigned_to {
            self.conditions
                .push(format!("{}.assigned_to = '{}'", alias, assigned));
        }
        self
    }

    /// Add a search filter (case-insensitive CONTAINS on title and description)
    pub fn add_search_filter(&mut self, alias: &str, search: Option<&str>) -> &mut Self {
        if let Some(search) = search {
            if !search.trim().is_empty() {
                let search_lower = search.to_lowercase();
                self.conditions.push(format!(
                    "(toLower({0}.title) CONTAINS '{1}' OR toLower({0}.description) CONTAINS '{1}')",
                    alias, search_lower
                ));
            }
        }
        self
    }

    /// Build the WHERE clause (returns empty string if no conditions)
    pub fn build(&self) -> String {
        if self.conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", self.conditions.join(" AND "))
        }
    }

    /// Build an AND clause to append to existing WHERE (returns empty string if no conditions)
    pub fn build_and(&self) -> String {
        if self.conditions.is_empty() {
            String::new()
        } else {
            format!("AND {}", self.conditions.join(" AND "))
        }
    }

    /// Check if any conditions have been added
    pub fn has_conditions(&self) -> bool {
        !self.conditions.is_empty()
    }
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
            // Knowledge Note constraints
            "CREATE CONSTRAINT note_id IF NOT EXISTS FOR (n:Note) REQUIRE n.id IS UNIQUE",
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
            // Knowledge Note indexes
            "CREATE INDEX note_project IF NOT EXISTS FOR (n:Note) ON (n.project_id)",
            "CREATE INDEX note_status IF NOT EXISTS FOR (n:Note) ON (n.status)",
            "CREATE INDEX note_type IF NOT EXISTS FOR (n:Note) ON (n.note_type)",
            "CREATE INDEX note_importance IF NOT EXISTS FOR (n:Note) ON (n.importance)",
            "CREATE INDEX note_staleness IF NOT EXISTS FOR (n:Note) ON (n.staleness_score)",
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
                pascal_to_snake_case(&node.get::<String>("status")?)
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

    /// Link a plan to a project (creates HAS_PLAN relationship)
    pub async fn link_plan_to_project(&self, plan_id: Uuid, project_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (project:Project {id: $project_id})
            MATCH (plan:Plan {id: $plan_id})
            SET plan.project_id = $project_id
            MERGE (project)-[:HAS_PLAN]->(plan)
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("plan_id", plan_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Unlink a plan from its project
    pub async fn unlink_plan_from_project(&self, plan_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (project:Project)-[r:HAS_PLAN]->(plan:Plan {id: $plan_id})
            DELETE r
            SET plan.project_id = null
            "#,
        )
        .param("plan_id", plan_id.to_string());

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
                pascal_to_snake_case(&node.get::<String>("status")?)
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

    /// Remove task dependency
    pub async fn remove_task_dependency(&self, task_id: Uuid, depends_on_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})-[r:DEPENDS_ON]->(dep:Task {id: $depends_on_id})
            DELETE r
            "#,
        )
        .param("task_id", task_id.to_string())
        .param("depends_on_id", depends_on_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get tasks that block this task (dependencies that are not completed)
    pub async fn get_task_blockers(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})-[:DEPENDS_ON]->(blocker:Task)
            WHERE blocker.status <> 'Completed'
            RETURN blocker
            ORDER BY COALESCE(blocker.priority, 0) DESC
            "#,
        )
        .param("task_id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("blocker")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// Get tasks blocked by this task (tasks depending on this one)
    pub async fn get_tasks_blocked_by(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (blocked:Task)-[:DEPENDS_ON]->(t:Task {id: $task_id})
            RETURN blocked
            ORDER BY COALESCE(blocked.priority, 0) DESC
            "#,
        )
        .param("task_id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("blocked")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// Get all dependencies for a task (all tasks it depends on, regardless of status)
    pub async fn get_task_dependencies(&self, task_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})-[:DEPENDS_ON]->(dep:Task)
            RETURN dep
            ORDER BY COALESCE(dep.priority, 0) DESC
            "#,
        )
        .param("task_id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("dep")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// Get dependency graph for a plan (all tasks and their dependencies)
    pub async fn get_plan_dependency_graph(
        &self,
        plan_id: Uuid,
    ) -> Result<(Vec<TaskNode>, Vec<(Uuid, Uuid)>)> {
        // Get all tasks in the plan
        let tasks = self.get_plan_tasks(plan_id).await?;

        // Get all DEPENDS_ON edges between tasks in this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:HAS_TASK]->(t:Task)-[:DEPENDS_ON]->(dep:Task)<-[:HAS_TASK]-(p)
            RETURN t.id AS from_id, dep.id AS to_id
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut edges = Vec::new();

        while let Some(row) = result.next().await? {
            let from_id: String = row.get("from_id")?;
            let to_id: String = row.get("to_id")?;
            if let (Ok(from), Ok(to)) = (from_id.parse::<Uuid>(), to_id.parse::<Uuid>()) {
                edges.push((from, to));
            }
        }

        Ok((tasks, edges))
    }

    /// Find critical path in a plan (longest chain of dependencies)
    pub async fn get_plan_critical_path(&self, plan_id: Uuid) -> Result<Vec<TaskNode>> {
        // Get all paths from tasks with no incoming deps to tasks with no outgoing deps
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:HAS_TASK]->(start:Task)
            WHERE NOT EXISTS { MATCH (start)-[:DEPENDS_ON]->(:Task) }
            MATCH (p)-[:HAS_TASK]->(end:Task)
            WHERE NOT EXISTS { MATCH (:Task)-[:DEPENDS_ON]->(end) }
            MATCH path = (start)<-[:DEPENDS_ON*0..]-(end)
            WHERE ALL(node IN nodes(path) WHERE (p)-[:HAS_TASK]->(node))
            WITH path, length(path) AS pathLength
            ORDER BY pathLength DESC
            LIMIT 1
            UNWIND nodes(path) AS task
            RETURN DISTINCT task
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("task")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
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
                    pascal_to_snake_case(&node.get::<String>("status")?)
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

    // ========================================================================
    // Commit operations
    // ========================================================================

    /// Create a commit node
    pub async fn create_commit(&self, commit: &CommitNode) -> Result<()> {
        let q = query(
            r#"
            MERGE (c:Commit {hash: $hash})
            SET c.message = $message,
                c.author = $author,
                c.timestamp = datetime($timestamp)
            "#,
        )
        .param("hash", commit.hash.clone())
        .param("message", commit.message.clone())
        .param("author", commit.author.clone())
        .param("timestamp", commit.timestamp.to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a commit by hash
    pub async fn get_commit(&self, hash: &str) -> Result<Option<CommitNode>> {
        let q = query(
            r#"
            MATCH (c:Commit {hash: $hash})
            RETURN c
            "#,
        )
        .param("hash", hash);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            Ok(Some(self.node_to_commit(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Helper to convert Neo4j node to CommitNode
    fn node_to_commit(&self, node: &neo4rs::Node) -> Result<CommitNode> {
        Ok(CommitNode {
            hash: node.get("hash")?,
            message: node.get("message")?,
            author: node.get("author")?,
            timestamp: node
                .get::<String>("timestamp")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
        })
    }

    /// Link a commit to a task (RESOLVED_BY relationship)
    pub async fn link_commit_to_task(&self, commit_hash: &str, task_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})
            MATCH (c:Commit {hash: $hash})
            MERGE (t)-[:RESOLVED_BY]->(c)
            "#,
        )
        .param("task_id", task_id.to_string())
        .param("hash", commit_hash);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Link a commit to a plan (RESULTED_IN relationship)
    pub async fn link_commit_to_plan(&self, commit_hash: &str, plan_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})
            MATCH (c:Commit {hash: $hash})
            MERGE (p)-[:RESULTED_IN]->(c)
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("hash", commit_hash);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get commits for a task
    pub async fn get_task_commits(&self, task_id: Uuid) -> Result<Vec<CommitNode>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $task_id})-[:RESOLVED_BY]->(c:Commit)
            RETURN c
            ORDER BY c.timestamp DESC
            "#,
        )
        .param("task_id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut commits = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            commits.push(self.node_to_commit(&node)?);
        }

        Ok(commits)
    }

    /// Get commits for a plan
    pub async fn get_plan_commits(&self, plan_id: Uuid) -> Result<Vec<CommitNode>> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:RESULTED_IN]->(c:Commit)
            RETURN c
            ORDER BY c.timestamp DESC
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut commits = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            commits.push(self.node_to_commit(&node)?);
        }

        Ok(commits)
    }

    // ========================================================================
    // Release operations
    // ========================================================================

    /// Create a release
    pub async fn create_release(&self, release: &ReleaseNode) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})
            CREATE (r:Release {
                id: $id,
                version: $version,
                title: $title,
                description: $description,
                status: $status,
                target_date: $target_date,
                released_at: $released_at,
                created_at: datetime($created_at),
                project_id: $project_id
            })
            CREATE (p)-[:HAS_RELEASE]->(r)
            "#,
        )
        .param("id", release.id.to_string())
        .param("version", release.version.clone())
        .param("title", release.title.clone().unwrap_or_default())
        .param("description", release.description.clone().unwrap_or_default())
        .param("status", format!("{:?}", release.status))
        .param("project_id", release.project_id.to_string())
        .param(
            "target_date",
            release.target_date.map(|d| d.to_rfc3339()).unwrap_or_default(),
        )
        .param(
            "released_at",
            release.released_at.map(|d| d.to_rfc3339()).unwrap_or_default(),
        )
        .param("created_at", release.created_at.to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a release by ID
    pub async fn get_release(&self, id: Uuid) -> Result<Option<ReleaseNode>> {
        let q = query(
            r#"
            MATCH (r:Release {id: $id})
            RETURN r
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            Ok(Some(self.node_to_release(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Helper to convert Neo4j node to ReleaseNode
    fn node_to_release(&self, node: &neo4rs::Node) -> Result<ReleaseNode> {
        Ok(ReleaseNode {
            id: node.get::<String>("id")?.parse()?,
            version: node.get("version")?,
            title: node.get::<String>("title").ok().filter(|s| !s.is_empty()),
            description: node.get::<String>("description").ok().filter(|s| !s.is_empty()),
            status: serde_json::from_str(&format!(
                "\"{}\"",
                pascal_to_snake_case(&node.get::<String>("status")?)
            ))
            .unwrap_or(ReleaseStatus::Planned),
            target_date: node
                .get::<String>("target_date")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            released_at: node
                .get::<String>("released_at")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            project_id: node.get::<String>("project_id")?.parse()?,
        })
    }

    /// List releases for a project
    pub async fn list_project_releases(&self, project_id: Uuid) -> Result<Vec<ReleaseNode>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:HAS_RELEASE]->(r:Release)
            RETURN r
            ORDER BY r.created_at DESC
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut releases = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            releases.push(self.node_to_release(&node)?);
        }

        Ok(releases)
    }

    /// Update a release
    pub async fn update_release(
        &self,
        id: Uuid,
        status: Option<ReleaseStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        released_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        let mut set_clauses = Vec::new();

        if status.is_some() {
            set_clauses.push("r.status = $status");
        }
        if target_date.is_some() {
            set_clauses.push("r.target_date = $target_date");
        }
        if released_at.is_some() {
            set_clauses.push("r.released_at = $released_at");
        }
        if title.is_some() {
            set_clauses.push("r.title = $title");
        }
        if description.is_some() {
            set_clauses.push("r.description = $description");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!("MATCH (r:Release {{id: $id}}) SET {}", set_clauses.join(", "));

        let mut q = query(&cypher).param("id", id.to_string());

        if let Some(ref s) = status {
            q = q.param("status", format!("{:?}", s));
        }
        if let Some(d) = target_date {
            q = q.param("target_date", d.to_rfc3339());
        }
        if let Some(d) = released_at {
            q = q.param("released_at", d.to_rfc3339());
        }
        if let Some(ref t) = title {
            q = q.param("title", t.clone());
        }
        if let Some(ref d) = description {
            q = q.param("description", d.clone());
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Add a task to a release
    pub async fn add_task_to_release(&self, release_id: Uuid, task_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (r:Release {id: $release_id})
            MATCH (t:Task {id: $task_id})
            MERGE (r)-[:INCLUDES_TASK]->(t)
            "#,
        )
        .param("release_id", release_id.to_string())
        .param("task_id", task_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Add a commit to a release
    pub async fn add_commit_to_release(&self, release_id: Uuid, commit_hash: &str) -> Result<()> {
        let q = query(
            r#"
            MATCH (r:Release {id: $release_id})
            MATCH (c:Commit {hash: $hash})
            MERGE (r)-[:INCLUDES_COMMIT]->(c)
            "#,
        )
        .param("release_id", release_id.to_string())
        .param("hash", commit_hash);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get release details with tasks and commits
    pub async fn get_release_details(
        &self,
        release_id: Uuid,
    ) -> Result<Option<(ReleaseNode, Vec<TaskNode>, Vec<CommitNode>)>> {
        let q = query(
            r#"
            MATCH (r:Release {id: $id})
            OPTIONAL MATCH (r)-[:INCLUDES_TASK]->(t:Task)
            OPTIONAL MATCH (r)-[:INCLUDES_COMMIT]->(c:Commit)
            RETURN r,
                   collect(DISTINCT t) AS tasks,
                   collect(DISTINCT c) AS commits
            "#,
        )
        .param("id", release_id.to_string());

        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            let release_node: neo4rs::Node = row.get("r")?;
            let release = self.node_to_release(&release_node)?;

            let task_nodes: Vec<neo4rs::Node> = row.get("tasks").unwrap_or_default();
            let tasks: Vec<TaskNode> = task_nodes
                .iter()
                .filter_map(|n| self.node_to_task(n).ok())
                .collect();

            let commit_nodes: Vec<neo4rs::Node> = row.get("commits").unwrap_or_default();
            let commits: Vec<CommitNode> = commit_nodes
                .iter()
                .filter_map(|n| self.node_to_commit(n).ok())
                .collect();

            Ok(Some((release, tasks, commits)))
        } else {
            Ok(None)
        }
    }

    // ========================================================================
    // Milestone operations
    // ========================================================================

    /// Create a milestone
    pub async fn create_milestone(&self, milestone: &MilestoneNode) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})
            CREATE (m:Milestone {
                id: $id,
                title: $title,
                description: $description,
                status: $status,
                target_date: $target_date,
                closed_at: $closed_at,
                created_at: datetime($created_at),
                project_id: $project_id
            })
            CREATE (p)-[:HAS_MILESTONE]->(m)
            "#,
        )
        .param("id", milestone.id.to_string())
        .param("title", milestone.title.clone())
        .param("description", milestone.description.clone().unwrap_or_default())
        .param("status", format!("{:?}", milestone.status))
        .param("project_id", milestone.project_id.to_string())
        .param(
            "target_date",
            milestone.target_date.map(|d| d.to_rfc3339()).unwrap_or_default(),
        )
        .param(
            "closed_at",
            milestone.closed_at.map(|d| d.to_rfc3339()).unwrap_or_default(),
        )
        .param("created_at", milestone.created_at.to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a milestone by ID
    pub async fn get_milestone(&self, id: Uuid) -> Result<Option<MilestoneNode>> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $id})
            RETURN m
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("m")?;
            Ok(Some(self.node_to_milestone(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Helper to convert Neo4j node to MilestoneNode
    fn node_to_milestone(&self, node: &neo4rs::Node) -> Result<MilestoneNode> {
        Ok(MilestoneNode {
            id: node.get::<String>("id")?.parse()?,
            title: node.get("title")?,
            description: node.get::<String>("description").ok().filter(|s| !s.is_empty()),
            status: serde_json::from_str(&format!(
                "\"{}\"",
                pascal_to_snake_case(&node.get::<String>("status")?)
            ))
            .unwrap_or(MilestoneStatus::Open),
            target_date: node
                .get::<String>("target_date")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            closed_at: node
                .get::<String>("closed_at")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            project_id: node.get::<String>("project_id")?.parse()?,
        })
    }

    /// List milestones for a project
    pub async fn list_project_milestones(&self, project_id: Uuid) -> Result<Vec<MilestoneNode>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:HAS_MILESTONE]->(m:Milestone)
            RETURN m
            ORDER BY m.target_date ASC, m.created_at ASC
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut milestones = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("m")?;
            milestones.push(self.node_to_milestone(&node)?);
        }

        Ok(milestones)
    }

    /// Update a milestone
    pub async fn update_milestone(
        &self,
        id: Uuid,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
        closed_at: Option<chrono::DateTime<chrono::Utc>>,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        let mut set_clauses = Vec::new();

        if status.is_some() {
            set_clauses.push("m.status = $status");
        }
        if target_date.is_some() {
            set_clauses.push("m.target_date = $target_date");
        }
        if closed_at.is_some() {
            set_clauses.push("m.closed_at = $closed_at");
        }
        if title.is_some() {
            set_clauses.push("m.title = $title");
        }
        if description.is_some() {
            set_clauses.push("m.description = $description");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!("MATCH (m:Milestone {{id: $id}}) SET {}", set_clauses.join(", "));

        let mut q = query(&cypher).param("id", id.to_string());

        if let Some(ref s) = status {
            q = q.param("status", format!("{:?}", s));
        }
        if let Some(d) = target_date {
            q = q.param("target_date", d.to_rfc3339());
        }
        if let Some(d) = closed_at {
            q = q.param("closed_at", d.to_rfc3339());
        }
        if let Some(ref t) = title {
            q = q.param("title", t.clone());
        }
        if let Some(ref d) = description {
            q = q.param("description", d.clone());
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Add a task to a milestone
    pub async fn add_task_to_milestone(&self, milestone_id: Uuid, task_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $milestone_id})
            MATCH (t:Task {id: $task_id})
            MERGE (m)-[:INCLUDES_TASK]->(t)
            "#,
        )
        .param("milestone_id", milestone_id.to_string())
        .param("task_id", task_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get milestone details with tasks
    pub async fn get_milestone_details(
        &self,
        milestone_id: Uuid,
    ) -> Result<Option<(MilestoneNode, Vec<TaskNode>)>> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $id})
            OPTIONAL MATCH (m)-[:INCLUDES_TASK]->(t:Task)
            RETURN m, collect(DISTINCT t) AS tasks
            "#,
        )
        .param("id", milestone_id.to_string());

        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            let milestone_node: neo4rs::Node = row.get("m")?;
            let milestone = self.node_to_milestone(&milestone_node)?;

            let task_nodes: Vec<neo4rs::Node> = row.get("tasks").unwrap_or_default();
            let tasks: Vec<TaskNode> = task_nodes
                .iter()
                .filter_map(|n| self.node_to_task(n).ok())
                .collect();

            Ok(Some((milestone, tasks)))
        } else {
            Ok(None)
        }
    }

    /// Get milestone progress (completed tasks / total tasks)
    pub async fn get_milestone_progress(&self, milestone_id: Uuid) -> Result<(u32, u32)> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $id})-[:INCLUDES_TASK]->(t:Task)
            RETURN count(t) AS total,
                   sum(CASE WHEN t.status = 'Completed' THEN 1 ELSE 0 END) AS completed
            "#,
        )
        .param("id", milestone_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let total: i64 = row.get("total").unwrap_or(0);
            let completed: i64 = row.get("completed").unwrap_or(0);
            Ok((completed as u32, total as u32))
        } else {
            Ok((0, 0))
        }
    }

    // ========================================================================
    // Roadmap operations
    // ========================================================================

    /// Get tasks for a milestone
    pub async fn get_milestone_tasks(&self, milestone_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $id})-[:INCLUDES_TASK]->(t:Task)
            RETURN t
            ORDER BY COALESCE(t.priority, 0) DESC, t.created_at
            "#,
        )
        .param("id", milestone_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// Get tasks for a release
    pub async fn get_release_tasks(&self, release_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (r:Release {id: $id})-[:INCLUDES_TASK]->(t:Task)
            RETURN t
            ORDER BY COALESCE(t.priority, 0) DESC, t.created_at
            "#,
        )
        .param("id", release_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    /// Get project progress stats
    pub async fn get_project_progress(
        &self,
        project_id: Uuid,
    ) -> Result<(u32, u32, u32, u32)> {
        // Count tasks across all plans for this project
        let q = query(
            r#"
            MATCH (project:Project {id: $project_id})-[:HAS_PLAN]->(p:Plan)-[:HAS_TASK]->(t:Task)
            RETURN
                count(t) AS total,
                sum(CASE WHEN t.status = 'Completed' THEN 1 ELSE 0 END) AS completed,
                sum(CASE WHEN t.status = 'InProgress' THEN 1 ELSE 0 END) AS in_progress,
                sum(CASE WHEN t.status = 'Pending' THEN 1 ELSE 0 END) AS pending
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let total: i64 = row.get("total").unwrap_or(0);
            let completed: i64 = row.get("completed").unwrap_or(0);
            let in_progress: i64 = row.get("in_progress").unwrap_or(0);
            let pending: i64 = row.get("pending").unwrap_or(0);
            Ok((total as u32, completed as u32, in_progress as u32, pending as u32))
        } else {
            Ok((0, 0, 0, 0))
        }
    }

    /// Get all task dependencies for a project (across all plans)
    pub async fn get_project_task_dependencies(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(Uuid, Uuid)>> {
        let q = query(
            r#"
            MATCH (project:Project {id: $project_id})-[:HAS_PLAN]->(p:Plan)-[:HAS_TASK]->(t:Task)-[:DEPENDS_ON]->(dep:Task)<-[:HAS_TASK]-(p2:Plan)<-[:HAS_PLAN]-(project)
            RETURN t.id AS from_id, dep.id AS to_id
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut edges = Vec::new();

        while let Some(row) = result.next().await? {
            let from_id: String = row.get("from_id")?;
            let to_id: String = row.get("to_id")?;
            if let (Ok(from), Ok(to)) = (from_id.parse::<Uuid>(), to_id.parse::<Uuid>()) {
                edges.push((from, to));
            }
        }

        Ok(edges)
    }

    /// Get all tasks for a project (across all plans)
    pub async fn get_project_tasks(&self, project_id: Uuid) -> Result<Vec<TaskNode>> {
        let q = query(
            r#"
            MATCH (project:Project {id: $project_id})-[:HAS_PLAN]->(p:Plan)-[:HAS_TASK]->(t:Task)
            RETURN t
            ORDER BY COALESCE(t.priority, 0) DESC, t.created_at
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            tasks.push(self.node_to_task(&node)?);
        }

        Ok(tasks)
    }

    // ========================================================================
    // Filtered list operations with pagination
    // ========================================================================

    /// List plans with filters and pagination
    ///
    /// Returns (plans, total_count)
    pub async fn list_plans_filtered(
        &self,
        statuses: Option<Vec<String>>,
        priority_min: Option<i32>,
        priority_max: Option<i32>,
        search: Option<&str>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<PlanNode>, usize)> {
        let mut where_builder = WhereBuilder::new();
        where_builder
            .add_status_filter("p", statuses)
            .add_priority_filter("p", priority_min, priority_max)
            .add_search_filter("p", search);

        let where_clause = where_builder.build();
        let order_field = match sort_by {
            Some("priority") => "COALESCE(p.priority, 0)",
            Some("title") => "p.title",
            Some("status") => "p.status",
            _ => "p.created_at",
        };
        let order_dir = if sort_order == "asc" { "ASC" } else { "DESC" };

        // Count query
        let count_cypher = format!("MATCH (p:Plan) {} RETURN count(p) AS total", where_clause);
        let count_result = self.execute(&count_cypher).await?;
        let total: i64 = count_result
            .first()
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        // Data query
        let cypher = format!(
            r#"
            MATCH (p:Plan)
            {}
            RETURN p
            ORDER BY {} {}
            SKIP {}
            LIMIT {}
            "#,
            where_clause, order_field, order_dir, offset, limit
        );

        let mut result = self.graph.execute(query(&cypher)).await?;
        let mut plans = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            plans.push(self.node_to_plan(&node)?);
        }

        Ok((plans, total as usize))
    }

    /// List all tasks across all plans with filters and pagination
    ///
    /// Returns (tasks_with_plan_info, total_count)
    pub async fn list_all_tasks_filtered(
        &self,
        plan_id: Option<Uuid>,
        statuses: Option<Vec<String>>,
        priority_min: Option<i32>,
        priority_max: Option<i32>,
        tags: Option<Vec<String>>,
        assigned_to: Option<&str>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<TaskWithPlan>, usize)> {
        let mut where_builder = WhereBuilder::new();
        where_builder
            .add_status_filter("t", statuses)
            .add_priority_filter("t", priority_min, priority_max)
            .add_tags_filter("t", tags)
            .add_assigned_to_filter("t", assigned_to);

        // Build plan filter if specified
        let plan_match = if let Some(pid) = plan_id {
            format!("MATCH (p:Plan {{id: '{}'}})-[:HAS_TASK]->(t:Task)", pid)
        } else {
            "MATCH (p:Plan)-[:HAS_TASK]->(t:Task)".to_string()
        };

        let where_clause = where_builder.build();
        let order_field = match sort_by {
            Some("priority") => "COALESCE(t.priority, 0)",
            Some("title") => "t.title",
            Some("status") => "t.status",
            Some("created_at") => "t.created_at",
            Some("updated_at") => "t.updated_at",
            _ => "COALESCE(t.priority, 0) DESC, t.created_at",
        };
        let order_dir = if sort_by.is_some() && sort_order == "asc" {
            "ASC"
        } else if sort_by.is_some() {
            "DESC"
        } else {
            "" // Default ordering already includes direction
        };

        // Count query
        let count_cypher = format!(
            "{} {} RETURN count(t) AS total",
            plan_match, where_clause
        );
        let count_result = self.execute(&count_cypher).await?;
        let total: i64 = count_result
            .first()
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        // Data query
        let cypher = format!(
            r#"
            {}
            {}
            RETURN t, p.id AS plan_id, p.title AS plan_title
            ORDER BY {} {}
            SKIP {}
            LIMIT {}
            "#,
            plan_match, where_clause, order_field, order_dir, offset, limit
        );

        let mut result = self.graph.execute(query(&cypher)).await?;
        let mut tasks = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            let plan_id_str: String = row.get("plan_id")?;
            let plan_title: String = row.get("plan_title")?;
            tasks.push(TaskWithPlan {
                task: self.node_to_task(&node)?,
                plan_id: plan_id_str.parse()?,
                plan_title,
            });
        }

        Ok((tasks, total as usize))
    }

    /// List project releases with filters and pagination
    pub async fn list_releases_filtered(
        &self,
        project_id: Uuid,
        statuses: Option<Vec<String>>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<ReleaseNode>, usize)> {
        let mut where_builder = WhereBuilder::new();
        where_builder.add_status_filter("r", statuses);

        let where_clause = where_builder.build_and();
        let order_field = match sort_by {
            Some("version") => "r.version",
            Some("target_date") => "r.target_date",
            Some("released_at") => "r.released_at",
            Some("title") => "r.title",
            _ => "r.created_at",
        };
        let order_dir = if sort_order == "asc" { "ASC" } else { "DESC" };

        // Count query
        let count_cypher = format!(
            "MATCH (p:Project {{id: $project_id}})-[:HAS_RELEASE]->(r:Release) {} RETURN count(r) AS total",
            if where_clause.is_empty() { "" } else { &where_clause }
        );
        let count_q = query(&count_cypher).param("project_id", project_id.to_string());
        let mut count_result = self.graph.execute(count_q).await?;
        let total: i64 = count_result
            .next()
            .await?
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        // Data query
        let cypher = format!(
            r#"
            MATCH (p:Project {{id: $project_id}})-[:HAS_RELEASE]->(r:Release)
            {}
            RETURN r
            ORDER BY {} {}
            SKIP {}
            LIMIT {}
            "#,
            if where_clause.is_empty() {
                ""
            } else {
                &where_clause
            },
            order_field,
            order_dir,
            offset,
            limit
        );

        let q = query(&cypher).param("project_id", project_id.to_string());
        let mut result = self.graph.execute(q).await?;
        let mut releases = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            releases.push(self.node_to_release(&node)?);
        }

        Ok((releases, total as usize))
    }

    /// List project milestones with filters and pagination
    pub async fn list_milestones_filtered(
        &self,
        project_id: Uuid,
        statuses: Option<Vec<String>>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<MilestoneNode>, usize)> {
        let mut where_builder = WhereBuilder::new();
        where_builder.add_status_filter("m", statuses);

        let where_clause = where_builder.build_and();
        let order_field = match sort_by {
            Some("title") => "m.title",
            Some("created_at") => "m.created_at",
            _ => "m.target_date",
        };
        let order_dir = if sort_order == "asc" { "ASC" } else { "DESC" };

        // Count query
        let count_cypher = format!(
            "MATCH (p:Project {{id: $project_id}})-[:HAS_MILESTONE]->(m:Milestone) {} RETURN count(m) AS total",
            if where_clause.is_empty() { "" } else { &where_clause }
        );
        let count_q = query(&count_cypher).param("project_id", project_id.to_string());
        let mut count_result = self.graph.execute(count_q).await?;
        let total: i64 = count_result
            .next()
            .await?
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        // Data query
        let cypher = format!(
            r#"
            MATCH (p:Project {{id: $project_id}})-[:HAS_MILESTONE]->(m:Milestone)
            {}
            RETURN m
            ORDER BY {} {}
            SKIP {}
            LIMIT {}
            "#,
            if where_clause.is_empty() {
                ""
            } else {
                &where_clause
            },
            order_field,
            order_dir,
            offset,
            limit
        );

        let q = query(&cypher).param("project_id", project_id.to_string());
        let mut result = self.graph.execute(q).await?;
        let mut milestones = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("m")?;
            milestones.push(self.node_to_milestone(&node)?);
        }

        Ok((milestones, total as usize))
    }

    /// List projects with search and pagination
    pub async fn list_projects_filtered(
        &self,
        search: Option<&str>,
        limit: usize,
        offset: usize,
        sort_by: Option<&str>,
        sort_order: &str,
    ) -> Result<(Vec<ProjectNode>, usize)> {
        let mut where_builder = WhereBuilder::new();
        where_builder.add_search_filter("p", search);

        let where_clause = where_builder.build();
        let order_field = match sort_by {
            Some("created_at") => "p.created_at",
            Some("last_synced") => "p.last_synced",
            _ => "p.name",
        };
        let order_dir = if sort_order == "asc" { "ASC" } else { "DESC" };

        // Count query
        let count_cypher = format!("MATCH (p:Project) {} RETURN count(p) AS total", where_clause);
        let count_result = self.execute(&count_cypher).await?;
        let total: i64 = count_result
            .first()
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        // Data query
        let cypher = format!(
            r#"
            MATCH (p:Project)
            {}
            RETURN p
            ORDER BY {} {}
            SKIP {}
            LIMIT {}
            "#,
            where_clause, order_field, order_dir, offset, limit
        );

        let mut result = self.graph.execute(query(&cypher)).await?;
        let mut projects = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            projects.push(self.node_to_project(&node)?);
        }

        Ok((projects, total as usize))
    }

    // ========================================================================
    // Knowledge Note operations
    // ========================================================================

    /// Create a new note
    pub async fn create_note(&self, note: &Note) -> Result<()> {
        let q = query(
            r#"
            CREATE (n:Note {
                id: $id,
                project_id: $project_id,
                note_type: $note_type,
                status: $status,
                importance: $importance,
                scope_type: $scope_type,
                scope_path: $scope_path,
                content: $content,
                tags: $tags,
                created_at: datetime($created_at),
                created_by: $created_by,
                last_confirmed_at: datetime($last_confirmed_at),
                last_confirmed_by: $last_confirmed_by,
                staleness_score: $staleness_score,
                changes_json: $changes_json,
                assertion_rule_json: $assertion_rule_json
            })
            "#,
        )
        .param("id", note.id.to_string())
        .param("project_id", note.project_id.to_string())
        .param("note_type", note.note_type.to_string())
        .param("status", note.status.to_string())
        .param("importance", note.importance.to_string())
        .param("scope_type", self.scope_type_string(&note.scope))
        .param("scope_path", self.scope_path_string(&note.scope))
        .param("content", note.content.clone())
        .param("tags", note.tags.clone())
        .param("created_at", note.created_at.to_rfc3339())
        .param("created_by", note.created_by.clone())
        .param(
            "last_confirmed_at",
            note.last_confirmed_at
                .unwrap_or(note.created_at)
                .to_rfc3339(),
        )
        .param(
            "last_confirmed_by",
            note.last_confirmed_by.clone().unwrap_or_default(),
        )
        .param("staleness_score", note.staleness_score)
        .param("changes_json", serde_json::to_string(&note.changes)?)
        .param(
            "assertion_rule_json",
            note.assertion_rule
                .as_ref()
                .map(|r| serde_json::to_string(r).unwrap_or_default())
                .unwrap_or_default(),
        );

        self.graph.run(q).await?;

        // Link to project
        let link_q = query(
            r#"
            MATCH (n:Note {id: $note_id})
            MATCH (p:Project {id: $project_id})
            MERGE (p)-[:HAS_NOTE]->(n)
            "#,
        )
        .param("note_id", note.id.to_string())
        .param("project_id", note.project_id.to_string());

        self.graph.run(link_q).await?;

        Ok(())
    }

    /// Get a note by ID
    pub async fn get_note(&self, id: Uuid) -> Result<Option<Note>> {
        let q = query(
            r#"
            MATCH (n:Note {id: $id})
            RETURN n
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            Ok(Some(self.node_to_note(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Update a note
    pub async fn update_note(
        &self,
        id: Uuid,
        content: Option<String>,
        importance: Option<NoteImportance>,
        status: Option<NoteStatus>,
        tags: Option<Vec<String>>,
        staleness_score: Option<f64>,
    ) -> Result<Option<Note>> {
        let mut set_clauses = vec!["n.updated_at = datetime()".to_string()];

        if let Some(ref c) = content {
            set_clauses.push(format!("n.content = '{}'", c.replace('\'', "\\'")));
        }
        if let Some(ref i) = importance {
            set_clauses.push(format!("n.importance = '{}'", i));
        }
        if let Some(ref s) = status {
            set_clauses.push(format!("n.status = '{}'", s));
        }
        if let Some(ref t) = tags {
            let tags_str = t
                .iter()
                .map(|s| format!("'{}'", s.replace('\'', "\\'")))
                .collect::<Vec<_>>()
                .join(", ");
            set_clauses.push(format!("n.tags = [{}]", tags_str));
        }
        if let Some(s) = staleness_score {
            set_clauses.push(format!("n.staleness_score = {}", s));
        }

        let cypher = format!(
            r#"
            MATCH (n:Note {{id: $id}})
            SET {}
            RETURN n
            "#,
            set_clauses.join(", ")
        );

        let q = query(&cypher).param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            Ok(Some(self.node_to_note(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Delete a note
    pub async fn delete_note(&self, id: Uuid) -> Result<bool> {
        let q = query(
            r#"
            MATCH (n:Note {id: $id})
            DETACH DELETE n
            RETURN count(n) AS deleted
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let deleted: i64 = row.get("deleted")?;
            Ok(deleted > 0)
        } else {
            Ok(false)
        }
    }

    /// List notes with filters and pagination
    pub async fn list_notes(
        &self,
        project_id: Option<Uuid>,
        filters: &NoteFilters,
    ) -> Result<(Vec<Note>, usize)> {
        let mut where_conditions = Vec::new();

        if let Some(ref pid) = project_id {
            where_conditions.push(format!("n.project_id = '{}'", pid));
        }

        if let Some(ref statuses) = filters.status {
            let status_list = statuses
                .iter()
                .map(|s| format!("'{}'", s))
                .collect::<Vec<_>>()
                .join(", ");
            where_conditions.push(format!("n.status IN [{}]", status_list));
        }

        if let Some(ref types) = filters.note_type {
            let type_list = types
                .iter()
                .map(|t| format!("'{}'", t))
                .collect::<Vec<_>>()
                .join(", ");
            where_conditions.push(format!("n.note_type IN [{}]", type_list));
        }

        if let Some(ref importance) = filters.importance {
            let imp_list = importance
                .iter()
                .map(|i| format!("'{}'", i))
                .collect::<Vec<_>>()
                .join(", ");
            where_conditions.push(format!("n.importance IN [{}]", imp_list));
        }

        if let Some(ref tags) = filters.tags {
            for tag in tags {
                where_conditions.push(format!("'{}' IN n.tags", tag));
            }
        }

        if let Some(min) = filters.min_staleness {
            where_conditions.push(format!("n.staleness_score >= {}", min));
        }

        if let Some(max) = filters.max_staleness {
            where_conditions.push(format!("n.staleness_score <= {}", max));
        }

        if let Some(ref search) = filters.search {
            if !search.trim().is_empty() {
                let search_lower = search.to_lowercase();
                where_conditions.push(format!(
                    "toLower(n.content) CONTAINS '{}'",
                    search_lower.replace('\'', "\\'")
                ));
            }
        }

        let where_clause = if where_conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_conditions.join(" AND "))
        };

        let order_field = filters.sort_by.as_deref().unwrap_or("created_at");
        let order_dir = filters.sort_order.as_deref().unwrap_or("desc");
        let limit = filters.limit.unwrap_or(50);
        let offset = filters.offset.unwrap_or(0);

        // Count total
        let count_cypher = format!(
            r#"
            MATCH (n:Note)
            {}
            RETURN count(n) AS total
            "#,
            where_clause
        );

        let mut count_result = self.graph.execute(query(&count_cypher)).await?;
        let total: i64 = if let Some(row) = count_result.next().await? {
            row.get("total")?
        } else {
            0
        };

        // Get notes
        let cypher = format!(
            r#"
            MATCH (n:Note)
            {}
            RETURN n
            ORDER BY n.{} {}
            SKIP {}
            LIMIT {}
            "#,
            where_clause, order_field, order_dir, offset, limit
        );

        let mut result = self.graph.execute(query(&cypher)).await?;
        let mut notes = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            notes.push(self.node_to_note(&node)?);
        }

        Ok((notes, total as usize))
    }

    /// Link a note to an entity (File, Function, Task, etc.)
    pub async fn link_note_to_entity(
        &self,
        note_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
        signature_hash: Option<&str>,
        body_hash: Option<&str>,
    ) -> Result<()> {
        let node_label = match entity_type {
            EntityType::Project => "Project",
            EntityType::File => "File",
            EntityType::Module => "Module",
            EntityType::Function => "Function",
            EntityType::Struct => "Struct",
            EntityType::Trait => "Trait",
            EntityType::Enum => "Enum",
            EntityType::Impl => "Impl",
            EntityType::Task => "Task",
            EntityType::Plan => "Plan",
            EntityType::Commit => "Commit",
            EntityType::Decision => "Decision",
        };

        // Determine the match field based on entity type
        let (match_field, match_value) = match entity_type {
            EntityType::File => ("path", entity_id.to_string()),
            EntityType::Commit => ("hash", entity_id.to_string()),
            _ => ("id", entity_id.to_string()),
        };

        let cypher = format!(
            r#"
            MATCH (n:Note {{id: $note_id}})
            MATCH (e:{} {{{}: $entity_id}})
            MERGE (n)-[r:ATTACHED_TO]->(e)
            SET r.signature_hash = $sig_hash,
                r.body_hash = $body_hash,
                r.last_verified = datetime()
            "#,
            node_label, match_field
        );

        let q = query(&cypher)
            .param("note_id", note_id.to_string())
            .param("entity_id", match_value)
            .param("sig_hash", signature_hash.unwrap_or(""))
            .param("body_hash", body_hash.unwrap_or(""));

        self.graph.run(q).await?;
        Ok(())
    }

    /// Unlink a note from an entity
    pub async fn unlink_note_from_entity(
        &self,
        note_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> Result<()> {
        let node_label = match entity_type {
            EntityType::Project => "Project",
            EntityType::File => "File",
            EntityType::Module => "Module",
            EntityType::Function => "Function",
            EntityType::Struct => "Struct",
            EntityType::Trait => "Trait",
            EntityType::Enum => "Enum",
            EntityType::Impl => "Impl",
            EntityType::Task => "Task",
            EntityType::Plan => "Plan",
            EntityType::Commit => "Commit",
            EntityType::Decision => "Decision",
        };

        let (match_field, match_value) = match entity_type {
            EntityType::File => ("path", entity_id.to_string()),
            EntityType::Commit => ("hash", entity_id.to_string()),
            _ => ("id", entity_id.to_string()),
        };

        let cypher = format!(
            r#"
            MATCH (n:Note {{id: $note_id}})-[r:ATTACHED_TO]->(e:{} {{{}: $entity_id}})
            DELETE r
            "#,
            node_label, match_field
        );

        let q = query(&cypher)
            .param("note_id", note_id.to_string())
            .param("entity_id", match_value);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get all notes attached to an entity
    pub async fn get_notes_for_entity(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> Result<Vec<Note>> {
        let node_label = match entity_type {
            EntityType::Project => "Project",
            EntityType::File => "File",
            EntityType::Module => "Module",
            EntityType::Function => "Function",
            EntityType::Struct => "Struct",
            EntityType::Trait => "Trait",
            EntityType::Enum => "Enum",
            EntityType::Impl => "Impl",
            EntityType::Task => "Task",
            EntityType::Plan => "Plan",
            EntityType::Commit => "Commit",
            EntityType::Decision => "Decision",
        };

        let (match_field, match_value) = match entity_type {
            EntityType::File => ("path", entity_id.to_string()),
            EntityType::Commit => ("hash", entity_id.to_string()),
            _ => ("id", entity_id.to_string()),
        };

        let cypher = format!(
            r#"
            MATCH (n:Note)-[:ATTACHED_TO]->(e:{} {{{}: $entity_id}})
            WHERE n.status IN ['active', 'needs_review']
            RETURN n
            ORDER BY n.importance DESC, n.created_at DESC
            "#,
            node_label, match_field
        );

        let q = query(&cypher).param("entity_id", match_value);

        let mut result = self.graph.execute(q).await?;
        let mut notes = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            notes.push(self.node_to_note(&node)?);
        }

        Ok(notes)
    }

    /// Get propagated notes for an entity (traversing the graph)
    pub async fn get_propagated_notes(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
        max_depth: u32,
        min_score: f64,
    ) -> Result<Vec<PropagatedNote>> {
        let node_label = match entity_type {
            EntityType::Project => "Project",
            EntityType::File => "File",
            EntityType::Module => "Module",
            EntityType::Function => "Function",
            EntityType::Struct => "Struct",
            EntityType::Trait => "Trait",
            EntityType::Enum => "Enum",
            EntityType::Impl => "Impl",
            EntityType::Task => "Task",
            EntityType::Plan => "Plan",
            EntityType::Commit => "Commit",
            EntityType::Decision => "Decision",
        };

        let (match_field, match_value) = match entity_type {
            EntityType::File => ("path", entity_id.to_string()),
            EntityType::Commit => ("hash", entity_id.to_string()),
            _ => ("id", entity_id.to_string()),
        };

        // Query for notes propagated through the graph
        let cypher = format!(
            r#"
            MATCH (target:{} {{{}: $entity_id}})
            MATCH path = (n:Note)-[:ATTACHED_TO]->(source)-[:CONTAINS|IMPORTS|CALLS*0..{}]->(target)
            WHERE n.status = 'active'
            WITH n, source, length(path) - 1 AS distance,
                 [node IN nodes(path) | coalesce(node.name, node.path, node.id)] AS path_names
            WITH n, source, distance, path_names,
                 CASE n.importance
                     WHEN 'critical' THEN 1.0
                     WHEN 'high' THEN 0.8
                     WHEN 'medium' THEN 0.5
                     ELSE 0.3
                 END AS importance_weight
            WITH n, source, distance, path_names,
                 (1.0 / (distance + 1)) * importance_weight AS score
            WHERE score >= $min_score
            RETURN DISTINCT n, score, coalesce(source.name, source.path, source.id) AS source_entity,
                   path_names, distance
            ORDER BY score DESC
            LIMIT 20
            "#,
            node_label, match_field, max_depth
        );

        let q = query(&cypher)
            .param("entity_id", match_value)
            .param("min_score", min_score);

        let mut result = self.graph.execute(q).await?;
        let mut propagated_notes = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            let note = self.node_to_note(&node)?;
            let score: f64 = row.get("score")?;
            let source_entity: String = row.get("source_entity")?;
            let path_names: Vec<String> = row.get("path_names").unwrap_or_default();
            let distance: i64 = row.get("distance")?;

            propagated_notes.push(PropagatedNote {
                note,
                relevance_score: score,
                source_entity,
                propagation_path: path_names,
                distance: distance as u32,
            });
        }

        Ok(propagated_notes)
    }

    /// Mark a note as superseded by another
    pub async fn supersede_note(&self, old_note_id: Uuid, new_note_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (old:Note {id: $old_id})
            MATCH (new:Note {id: $new_id})
            SET old.status = 'archived',
                old.superseded_by = $new_id
            SET new.supersedes = $old_id
            MERGE (new)-[:SUPERSEDES]->(old)
            "#,
        )
        .param("old_id", old_note_id.to_string())
        .param("new_id", new_note_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Confirm a note is still valid
    pub async fn confirm_note(&self, note_id: Uuid, confirmed_by: &str) -> Result<Option<Note>> {
        let q = query(
            r#"
            MATCH (n:Note {id: $id})
            SET n.last_confirmed_at = datetime(),
                n.last_confirmed_by = $confirmed_by,
                n.staleness_score = 0.0,
                n.status = 'active'
            RETURN n
            "#,
        )
        .param("id", note_id.to_string())
        .param("confirmed_by", confirmed_by);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            Ok(Some(self.node_to_note(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Get notes that need review (stale or needs_review status)
    pub async fn get_notes_needing_review(&self, project_id: Option<Uuid>) -> Result<Vec<Note>> {
        let project_filter = project_id
            .map(|pid| format!("AND n.project_id = '{}'", pid))
            .unwrap_or_default();

        let cypher = format!(
            r#"
            MATCH (n:Note)
            WHERE n.status IN ['needs_review', 'stale']
            {}
            RETURN n
            ORDER BY n.staleness_score DESC, n.importance DESC
            "#,
            project_filter
        );

        let mut result = self.graph.execute(query(&cypher)).await?;
        let mut notes = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            notes.push(self.node_to_note(&node)?);
        }

        Ok(notes)
    }

    /// Update staleness scores for all active notes
    pub async fn update_staleness_scores(&self) -> Result<usize> {
        // This updates staleness based on time since last confirmation
        // The actual calculation is done in Cypher for efficiency
        let q = query(
            r#"
            MATCH (n:Note)
            WHERE n.status = 'active' AND n.note_type <> 'assertion'
            WITH n,
                 duration.between(
                     datetime(n.last_confirmed_at),
                     datetime()
                 ).days AS days_since_confirmed,
                 CASE n.note_type
                     WHEN 'context' THEN 30.0
                     WHEN 'tip' THEN 90.0
                     WHEN 'observation' THEN 90.0
                     WHEN 'gotcha' THEN 180.0
                     WHEN 'guideline' THEN 365.0
                     WHEN 'pattern' THEN 365.0
                     ELSE 90.0
                 END AS base_decay_days,
                 CASE n.importance
                     WHEN 'critical' THEN 0.5
                     WHEN 'high' THEN 0.7
                     WHEN 'medium' THEN 1.0
                     ELSE 1.3
                 END AS decay_factor
            WITH n,
                 CASE
                     WHEN base_decay_days = 0 THEN 0
                     ELSE (1.0 - exp(-1.0 * days_since_confirmed / base_decay_days)) * decay_factor
                 END AS new_staleness
            WITH n,
                 CASE WHEN new_staleness > 1.0 THEN 1.0
                      WHEN new_staleness < 0.0 THEN 0.0
                      ELSE new_staleness END AS clamped_staleness
            WHERE abs(n.staleness_score - clamped_staleness) > 0.01
            SET n.staleness_score = clamped_staleness,
                n.status = CASE WHEN clamped_staleness > 0.8 THEN 'stale' ELSE n.status END
            RETURN count(n) AS updated
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let updated: i64 = row.get("updated")?;
            Ok(updated as usize)
        } else {
            Ok(0)
        }
    }

    /// Get anchors for a note
    pub async fn get_note_anchors(&self, note_id: Uuid) -> Result<Vec<NoteAnchor>> {
        let q = query(
            r#"
            MATCH (n:Note {id: $id})-[r:ATTACHED_TO]->(e)
            RETURN labels(e)[0] AS entity_type,
                   coalesce(e.id, e.path, e.hash) AS entity_id,
                   r.signature_hash AS sig_hash,
                   r.body_hash AS body_hash,
                   r.last_verified AS last_verified
            "#,
        )
        .param("id", note_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut anchors = Vec::new();

        while let Some(row) = result.next().await? {
            let entity_type_str: String = row.get("entity_type")?;
            let entity_id: String = row.get("entity_id")?;
            let sig_hash: Option<String> = row.get("sig_hash").ok();
            let body_hash: Option<String> = row.get("body_hash").ok();
            let last_verified: String = row
                .get::<String>("last_verified")
                .unwrap_or_else(|_| chrono::Utc::now().to_rfc3339());

            let entity_type = entity_type_str
                .to_lowercase()
                .parse::<EntityType>()
                .unwrap_or(EntityType::File);

            anchors.push(NoteAnchor {
                entity_type,
                entity_id,
                signature_hash: sig_hash.filter(|s| !s.is_empty()),
                body_hash: body_hash.filter(|s| !s.is_empty()),
                last_verified: last_verified.parse().unwrap_or_else(|_| chrono::Utc::now()),
                is_valid: true,
            });
        }

        Ok(anchors)
    }

    // Helper function to convert Note scope to type string
    fn scope_type_string(&self, scope: &NoteScope) -> String {
        match scope {
            NoteScope::Project => "project".to_string(),
            NoteScope::Module(_) => "module".to_string(),
            NoteScope::File(_) => "file".to_string(),
            NoteScope::Function(_) => "function".to_string(),
            NoteScope::Struct(_) => "struct".to_string(),
            NoteScope::Trait(_) => "trait".to_string(),
        }
    }

    // Helper function to convert Note scope to path string
    fn scope_path_string(&self, scope: &NoteScope) -> String {
        match scope {
            NoteScope::Project => String::new(),
            NoteScope::Module(path) => path.clone(),
            NoteScope::File(path) => path.clone(),
            NoteScope::Function(name) => name.clone(),
            NoteScope::Struct(name) => name.clone(),
            NoteScope::Trait(name) => name.clone(),
        }
    }

    // Helper function to convert Neo4j node to Note
    fn node_to_note(&self, node: &neo4rs::Node) -> Result<Note> {
        let scope_type: String = node.get("scope_type").unwrap_or_else(|_| "project".to_string());
        let scope_path: String = node.get("scope_path").unwrap_or_default();

        let scope = match scope_type.as_str() {
            "module" => NoteScope::Module(scope_path),
            "file" => NoteScope::File(scope_path),
            "function" => NoteScope::Function(scope_path),
            "struct" => NoteScope::Struct(scope_path),
            "trait" => NoteScope::Trait(scope_path),
            _ => NoteScope::Project,
        };

        let note_type_str: String = node.get("note_type")?;
        let status_str: String = node.get("status")?;
        let importance_str: String = node.get("importance").unwrap_or_else(|_| "medium".to_string());

        let changes_json: String = node.get("changes_json").unwrap_or_else(|_| "[]".to_string());
        let changes: Vec<NoteChange> =
            serde_json::from_str(&changes_json).unwrap_or_default();

        let assertion_rule_json: String = node
            .get("assertion_rule_json")
            .unwrap_or_else(|_| String::new());
        let assertion_rule = if assertion_rule_json.is_empty() {
            None
        } else {
            serde_json::from_str(&assertion_rule_json).ok()
        };

        Ok(Note {
            id: node.get::<String>("id")?.parse()?,
            project_id: node.get::<String>("project_id")?.parse()?,
            note_type: note_type_str.parse().unwrap_or(NoteType::Observation),
            status: status_str.parse().unwrap_or(NoteStatus::Active),
            importance: importance_str.parse().unwrap_or(NoteImportance::Medium),
            scope,
            content: node.get("content")?,
            tags: node.get("tags").unwrap_or_default(),
            anchors: vec![], // Anchors are loaded separately
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            created_by: node.get("created_by")?,
            last_confirmed_at: node
                .get::<String>("last_confirmed_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            last_confirmed_by: node.get("last_confirmed_by").ok(),
            staleness_score: node.get("staleness_score").unwrap_or(0.0),
            supersedes: node
                .get::<String>("supersedes")
                .ok()
                .and_then(|s| s.parse().ok()),
            superseded_by: node
                .get::<String>("superseded_by")
                .ok()
                .and_then(|s| s.parse().ok()),
            changes,
            assertion_rule,
            last_assertion_result: None, // Loaded separately if needed
        })
    }
}
