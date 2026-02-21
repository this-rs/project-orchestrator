//! Neo4j client for interacting with the knowledge graph

use super::models::*;
use crate::notes::{
    EntityType, Note, NoteAnchor, NoteChange, NoteFilters, NoteImportance, NoteScope, NoteStatus,
    NoteType, PropagatedNote,
};
use crate::plan::models::TaskDetails;
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
                let pascal_statuses: Vec<String> =
                    statuses.iter().map(|s| snake_to_pascal_case(s)).collect();
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
                self.conditions.push(format!("'{}' IN {}.tags", tag, alias));
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
            // Workspace constraints
            "CREATE CONSTRAINT workspace_id IF NOT EXISTS FOR (w:Workspace) REQUIRE w.id IS UNIQUE",
            "CREATE CONSTRAINT workspace_slug IF NOT EXISTS FOR (w:Workspace) REQUIRE w.slug IS UNIQUE",
            "CREATE CONSTRAINT workspace_milestone_id IF NOT EXISTS FOR (wm:WorkspaceMilestone) REQUIRE wm.id IS UNIQUE",
            "CREATE CONSTRAINT resource_id IF NOT EXISTS FOR (r:Resource) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE",
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
            // Workspace indexes
            "CREATE INDEX workspace_name IF NOT EXISTS FOR (w:Workspace) ON (w.name)",
            "CREATE INDEX ws_milestone_workspace IF NOT EXISTS FOR (wm:WorkspaceMilestone) ON (wm.workspace_id)",
            "CREATE INDEX ws_milestone_status IF NOT EXISTS FOR (wm:WorkspaceMilestone) ON (wm.status)",
            "CREATE INDEX resource_workspace IF NOT EXISTS FOR (r:Resource) ON (r.workspace_id)",
            "CREATE INDEX resource_project IF NOT EXISTS FOR (r:Resource) ON (r.project_id)",
            "CREATE INDEX resource_type IF NOT EXISTS FOR (r:Resource) ON (r.resource_type)",
            "CREATE INDEX component_workspace IF NOT EXISTS FOR (c:Component) ON (c.workspace_id)",
            "CREATE INDEX component_type IF NOT EXISTS FOR (c:Component) ON (c.component_type)",
        ];

        // Vector indexes (require Neo4j 5.13+ — gracefully skip if not supported)
        let vector_indexes = vec![
            // HNSW vector index for cosine similarity search on Note embeddings (768d = nomic-embed-text)
            r#"CREATE VECTOR INDEX note_embeddings IF NOT EXISTS
               FOR (n:Note) ON (n.embedding)
               OPTIONS {indexConfig: {
                   `vector.dimensions`: 768,
                   `vector.similarity_function`: 'cosine'
               }}"#,
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

        // Vector indexes — optional, don't fail startup if Neo4j doesn't support them
        for vi in vector_indexes {
            if let Err(e) = self.graph.run(query(vi)).await {
                tracing::warn!(
                    "Vector index creation skipped (Neo4j may not support vector indexes): {}",
                    e
                );
            }
        }

        Ok(())
    }

    /// Execute a raw Cypher query (internal use only)
    pub(crate) async fn execute(&self, cypher: &str) -> Result<Vec<neo4rs::Row>> {
        let mut result = self.graph.execute(query(cypher)).await?;
        let mut rows = Vec::new();
        while let Some(row) = result.next().await? {
            rows.push(row);
        }
        Ok(rows)
    }

    /// Execute a parameterized Cypher query (internal use only)
    pub(crate) async fn execute_with_params(&self, q: Query) -> Result<Vec<neo4rs::Row>> {
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

    /// Update project fields (name, description, root_path)
    pub async fn update_project(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<Option<String>>,
        root_path: Option<String>,
    ) -> Result<()> {
        let mut set_clauses = vec![];

        if name.is_some() {
            set_clauses.push("p.name = $name");
        }
        if description.is_some() {
            set_clauses.push("p.description = $description");
        }
        if root_path.is_some() {
            set_clauses.push("p.root_path = $root_path");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (p:Project {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", id.to_string());

        if let Some(name) = name {
            q = q.param("name", name);
        }
        if let Some(desc) = description {
            q = q.param("description", desc.unwrap_or_default());
        }
        if let Some(root_path) = root_path {
            q = q.param("root_path", root_path);
        }

        self.graph.run(q).await?;
        Ok(())
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

    /// Update project analytics_computed_at timestamp
    pub async fn update_project_analytics_timestamp(&self, id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            SET p.analytics_computed_at = datetime($now)
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
            analytics_computed_at: node
                .get::<String>("analytics_computed_at")
                .ok()
                .and_then(|s| s.parse().ok()),
        })
    }

    // ========================================================================
    // Workspace operations
    // ========================================================================

    /// Create a new workspace
    pub async fn create_workspace(&self, workspace: &WorkspaceNode) -> Result<()> {
        let q = query(
            r#"
            CREATE (w:Workspace {
                id: $id,
                name: $name,
                slug: $slug,
                description: $description,
                created_at: datetime($created_at),
                metadata: $metadata
            })
            "#,
        )
        .param("id", workspace.id.to_string())
        .param("name", workspace.name.clone())
        .param("slug", workspace.slug.clone())
        .param(
            "description",
            workspace.description.clone().unwrap_or_default(),
        )
        .param("created_at", workspace.created_at.to_rfc3339())
        .param("metadata", workspace.metadata.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a workspace by ID
    pub async fn get_workspace(&self, id: Uuid) -> Result<Option<WorkspaceNode>> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $id})
            RETURN w
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("w")?;
            Ok(Some(self.node_to_workspace(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Get a workspace by slug
    pub async fn get_workspace_by_slug(&self, slug: &str) -> Result<Option<WorkspaceNode>> {
        let q = query(
            r#"
            MATCH (w:Workspace {slug: $slug})
            RETURN w
            "#,
        )
        .param("slug", slug);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("w")?;
            Ok(Some(self.node_to_workspace(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List all workspaces
    pub async fn list_workspaces(&self) -> Result<Vec<WorkspaceNode>> {
        let q = query(
            r#"
            MATCH (w:Workspace)
            RETURN w
            ORDER BY w.name
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut workspaces = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("w")?;
            workspaces.push(self.node_to_workspace(&node)?);
        }

        Ok(workspaces)
    }

    /// Update a workspace
    pub async fn update_workspace(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        let mut set_clauses = vec!["w.updated_at = datetime($now)".to_string()];

        if name.is_some() {
            set_clauses.push("w.name = $name".to_string());
        }
        if description.is_some() {
            set_clauses.push("w.description = $description".to_string());
        }
        if metadata.is_some() {
            set_clauses.push("w.metadata = $metadata".to_string());
        }

        let cypher = format!(
            r#"
            MATCH (w:Workspace {{id: $id}})
            SET {}
            "#,
            set_clauses.join(", ")
        );

        let mut q = query(&cypher)
            .param("id", id.to_string())
            .param("now", chrono::Utc::now().to_rfc3339());

        if let Some(n) = name {
            q = q.param("name", n);
        }
        if let Some(d) = description {
            q = q.param("description", d);
        }
        if let Some(m) = metadata {
            q = q.param("metadata", m.to_string());
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a workspace and all its data
    pub async fn delete_workspace(&self, id: Uuid) -> Result<()> {
        // Delete workspace milestones
        let q = query(
            r#"
            MATCH (w:Workspace {id: $id})-[:HAS_WORKSPACE_MILESTONE]->(wm:WorkspaceMilestone)
            DETACH DELETE wm
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete resources owned by workspace
        let q = query(
            r#"
            MATCH (w:Workspace {id: $id})-[:HAS_RESOURCE]->(r:Resource)
            DETACH DELETE r
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete components
        let q = query(
            r#"
            MATCH (w:Workspace {id: $id})-[:HAS_COMPONENT]->(c:Component)
            DETACH DELETE c
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Remove workspace association from projects (don't delete projects)
        let q = query(
            r#"
            MATCH (p:Project)-[r:BELONGS_TO_WORKSPACE]->(w:Workspace {id: $id})
            DELETE r
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        // Delete the workspace itself
        let q = query(
            r#"
            MATCH (w:Workspace {id: $id})
            DETACH DELETE w
            "#,
        )
        .param("id", id.to_string());
        self.graph.run(q).await?;

        Ok(())
    }

    /// Add a project to a workspace
    pub async fn add_project_to_workspace(
        &self,
        workspace_id: Uuid,
        project_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})
            MATCH (p:Project {id: $project_id})
            MERGE (p)-[:BELONGS_TO_WORKSPACE]->(w)
            "#,
        )
        .param("workspace_id", workspace_id.to_string())
        .param("project_id", project_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Remove a project from a workspace
    pub async fn remove_project_from_workspace(
        &self,
        workspace_id: Uuid,
        project_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[r:BELONGS_TO_WORKSPACE]->(w:Workspace {id: $workspace_id})
            DELETE r
            "#,
        )
        .param("workspace_id", workspace_id.to_string())
        .param("project_id", project_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// List all projects in a workspace
    pub async fn list_workspace_projects(&self, workspace_id: Uuid) -> Result<Vec<ProjectNode>> {
        let q = query(
            r#"
            MATCH (p:Project)-[:BELONGS_TO_WORKSPACE]->(w:Workspace {id: $workspace_id})
            RETURN p
            ORDER BY p.name
            "#,
        )
        .param("workspace_id", workspace_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut projects = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            projects.push(self.node_to_project(&node)?);
        }

        Ok(projects)
    }

    /// Get the workspace a project belongs to
    pub async fn get_project_workspace(&self, project_id: Uuid) -> Result<Option<WorkspaceNode>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:BELONGS_TO_WORKSPACE]->(w:Workspace)
            RETURN w
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("w")?;
            Ok(Some(self.node_to_workspace(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Helper to convert Neo4j node to WorkspaceNode
    fn node_to_workspace(&self, node: &neo4rs::Node) -> Result<WorkspaceNode> {
        let metadata_str: String = node.get("metadata").unwrap_or_else(|_| "{}".to_string());
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_str).unwrap_or(serde_json::json!({}));

        Ok(WorkspaceNode {
            id: node.get::<String>("id")?.parse()?,
            name: node.get("name")?,
            slug: node.get("slug")?,
            description: node.get("description").ok(),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            updated_at: node
                .get::<String>("updated_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            metadata,
        })
    }

    // ========================================================================
    // Workspace Milestone operations
    // ========================================================================

    /// Create a workspace milestone
    pub async fn create_workspace_milestone(
        &self,
        milestone: &WorkspaceMilestoneNode,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})
            CREATE (wm:WorkspaceMilestone {
                id: $id,
                workspace_id: $workspace_id,
                title: $title,
                description: $description,
                status: $status,
                target_date: $target_date,
                created_at: datetime($created_at),
                tags: $tags
            })
            CREATE (w)-[:HAS_WORKSPACE_MILESTONE]->(wm)
            "#,
        )
        .param("id", milestone.id.to_string())
        .param("workspace_id", milestone.workspace_id.to_string())
        .param("title", milestone.title.clone())
        .param(
            "description",
            milestone.description.clone().unwrap_or_default(),
        )
        .param(
            "status",
            serde_json::to_value(&milestone.status)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string(),
        )
        .param(
            "target_date",
            milestone
                .target_date
                .map(|d| d.to_rfc3339())
                .unwrap_or_default(),
        )
        .param("created_at", milestone.created_at.to_rfc3339())
        .param("tags", milestone.tags.clone());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a workspace milestone by ID
    pub async fn get_workspace_milestone(
        &self,
        id: Uuid,
    ) -> Result<Option<WorkspaceMilestoneNode>> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $id})
            RETURN wm
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("wm")?;
            Ok(Some(self.node_to_workspace_milestone(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List workspace milestones (unpaginated, used internally)
    pub async fn list_workspace_milestones(
        &self,
        workspace_id: Uuid,
    ) -> Result<Vec<WorkspaceMilestoneNode>> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})-[:HAS_WORKSPACE_MILESTONE]->(wm:WorkspaceMilestone)
            RETURN wm
            ORDER BY wm.target_date, wm.title
            "#,
        )
        .param("workspace_id", workspace_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut milestones = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("wm")?;
            milestones.push(self.node_to_workspace_milestone(&node)?);
        }

        Ok(milestones)
    }

    /// List workspace milestones with pagination and status filter
    ///
    /// Returns (milestones, total_count)
    pub async fn list_workspace_milestones_filtered(
        &self,
        workspace_id: Uuid,
        status: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<WorkspaceMilestoneNode>, usize)> {
        let status_filter = if let Some(s) = status {
            format!("WHERE toLower(wm.status) = toLower('{}')", s)
        } else {
            String::new()
        };

        let count_cypher = format!(
            "MATCH (w:Workspace {{id: $workspace_id}})-[:HAS_WORKSPACE_MILESTONE]->(wm:WorkspaceMilestone) {} RETURN count(wm) AS total",
            status_filter
        );
        let mut count_stream = self
            .graph
            .execute(query(&count_cypher).param("workspace_id", workspace_id.to_string()))
            .await?;
        let total: i64 = if let Some(row) = count_stream.next().await? {
            row.get("total")?
        } else {
            0
        };

        let data_cypher = format!(
            r#"
            MATCH (w:Workspace {{id: $workspace_id}})-[:HAS_WORKSPACE_MILESTONE]->(wm:WorkspaceMilestone)
            {}
            RETURN wm
            ORDER BY wm.target_date, wm.title
            SKIP {}
            LIMIT {}
            "#,
            status_filter, offset, limit
        );

        let mut result = self
            .graph
            .execute(query(&data_cypher).param("workspace_id", workspace_id.to_string()))
            .await?;
        let mut milestones = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("wm")?;
            milestones.push(self.node_to_workspace_milestone(&node)?);
        }

        Ok((milestones, total as usize))
    }

    /// List all workspace milestones across all workspaces with filters and pagination
    ///
    /// Returns (milestones_with_workspace_info, total_count)
    pub async fn list_all_workspace_milestones_filtered(
        &self,
        workspace_id: Option<Uuid>,
        status: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<(WorkspaceMilestoneNode, String, String, String)>> {
        let mut conditions = Vec::new();
        if let Some(wid) = workspace_id {
            conditions.push(format!("w.id = '{}'", wid));
        }
        if let Some(s) = status {
            let pascal = snake_to_pascal_case(s);
            conditions.push(format!("wm.status = '{}'", pascal));
        }
        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let cypher = format!(
            r#"
            MATCH (w:Workspace)-[:HAS_WORKSPACE_MILESTONE]->(wm:WorkspaceMilestone)
            {}
            RETURN wm, w.id AS workspace_id, w.name AS workspace_name, w.slug AS workspace_slug
            ORDER BY wm.target_date, wm.title
            SKIP {}
            LIMIT {}
            "#,
            where_clause, offset, limit
        );

        let mut result = self.graph.execute(query(&cypher)).await?;
        let mut items = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("wm")?;
            let wid: String = row.get("workspace_id")?;
            let wname: String = row.get("workspace_name")?;
            let wslug: String = row.get("workspace_slug")?;
            items.push((self.node_to_workspace_milestone(&node)?, wid, wname, wslug));
        }

        Ok(items)
    }

    /// Count all workspace milestones across workspaces with optional filters
    pub async fn count_all_workspace_milestones(
        &self,
        workspace_id: Option<Uuid>,
        status: Option<&str>,
    ) -> Result<usize> {
        let mut conditions = Vec::new();
        if let Some(wid) = workspace_id {
            conditions.push(format!("w.id = '{}'", wid));
        }
        if let Some(s) = status {
            let pascal = snake_to_pascal_case(s);
            conditions.push(format!("wm.status = '{}'", pascal));
        }
        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let cypher = format!(
            "MATCH (w:Workspace)-[:HAS_WORKSPACE_MILESTONE]->(wm:WorkspaceMilestone) {} RETURN count(wm) AS total",
            where_clause
        );
        let count_result = self.execute(&cypher).await?;
        let total: i64 = count_result
            .first()
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        Ok(total as usize)
    }

    /// Update a workspace milestone
    pub async fn update_workspace_milestone(
        &self,
        id: Uuid,
        title: Option<String>,
        description: Option<String>,
        status: Option<MilestoneStatus>,
        target_date: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<()> {
        let mut set_clauses = Vec::new();

        if title.is_some() {
            set_clauses.push("wm.title = $title".to_string());
        }
        if description.is_some() {
            set_clauses.push("wm.description = $description".to_string());
        }
        if status.is_some() {
            set_clauses.push("wm.status = $status".to_string());
        }
        if target_date.is_some() {
            set_clauses.push("wm.target_date = $target_date".to_string());
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            r#"
            MATCH (wm:WorkspaceMilestone {{id: $id}})
            SET {}
            "#,
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", id.to_string());

        if let Some(t) = title {
            q = q.param("title", t);
        }
        if let Some(d) = description {
            q = q.param("description", d);
        }
        if let Some(s) = status {
            q = q.param(
                "status",
                serde_json::to_value(&s)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string(),
            );
        }
        if let Some(td) = target_date {
            q = q.param("target_date", td.to_rfc3339());
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a workspace milestone
    pub async fn delete_workspace_milestone(&self, id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $id})
            DETACH DELETE wm
            "#,
        )
        .param("id", id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Add a task to a workspace milestone
    pub async fn add_task_to_workspace_milestone(
        &self,
        milestone_id: Uuid,
        task_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $milestone_id})
            MATCH (t:Task {id: $task_id})
            MERGE (wm)-[:INCLUDES_TASK]->(t)
            "#,
        )
        .param("milestone_id", milestone_id.to_string())
        .param("task_id", task_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Remove a task from a workspace milestone
    pub async fn remove_task_from_workspace_milestone(
        &self,
        milestone_id: Uuid,
        task_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $milestone_id})-[r:INCLUDES_TASK]->(t:Task {id: $task_id})
            DELETE r
            "#,
        )
        .param("milestone_id", milestone_id.to_string())
        .param("task_id", task_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Link a plan to a workspace milestone
    pub async fn link_plan_to_workspace_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})
            MATCH (wm:WorkspaceMilestone {id: $milestone_id})
            MERGE (p)-[:TARGETS_MILESTONE]->(wm)
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("milestone_id", milestone_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Unlink a plan from a workspace milestone
    pub async fn unlink_plan_from_workspace_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[r:TARGETS_MILESTONE]->(wm:WorkspaceMilestone {id: $milestone_id})
            DELETE r
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("milestone_id", milestone_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get workspace milestone progress
    pub async fn get_workspace_milestone_progress(
        &self,
        milestone_id: Uuid,
    ) -> Result<(u32, u32, u32, u32)> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $milestone_id})-[:INCLUDES_TASK]->(t:Task)
            RETURN
                count(t) AS total,
                sum(CASE WHEN t.status = 'Completed' THEN 1 ELSE 0 END) AS completed,
                sum(CASE WHEN t.status = 'InProgress' THEN 1 ELSE 0 END) AS in_progress,
                sum(CASE WHEN t.status = 'Pending' THEN 1 ELSE 0 END) AS pending
            "#,
        )
        .param("milestone_id", milestone_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let total: i64 = row.get("total").unwrap_or(0);
            let completed: i64 = row.get("completed").unwrap_or(0);
            let in_progress: i64 = row.get("in_progress").unwrap_or(0);
            let pending: i64 = row.get("pending").unwrap_or(0);
            Ok((
                total as u32,
                completed as u32,
                in_progress as u32,
                pending as u32,
            ))
        } else {
            Ok((0, 0, 0, 0))
        }
    }

    /// Get tasks linked to a workspace milestone (with plan info)
    pub async fn get_workspace_milestone_tasks(
        &self,
        milestone_id: Uuid,
    ) -> Result<Vec<TaskWithPlan>> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $milestone_id})-[:INCLUDES_TASK]->(t:Task)
            OPTIONAL MATCH (p:Plan)-[:HAS_TASK]->(t)
            RETURN t, p.id AS plan_id, COALESCE(p.title, '') AS plan_title,
                   COALESCE(p.status, '') AS plan_status
            ORDER BY t.priority DESC, t.created_at
            "#,
        )
        .param("milestone_id", milestone_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut tasks = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            let plan_id_str: String = row.get("plan_id").unwrap_or_default();
            let plan_title: String = row.get("plan_title").unwrap_or_default();
            let plan_status: String = row.get("plan_status").unwrap_or_default();
            tasks.push(TaskWithPlan {
                task: self.node_to_task(&node)?,
                plan_id: plan_id_str.parse().unwrap_or_default(),
                plan_title,
                plan_status: if plan_status.is_empty() {
                    None
                } else {
                    Some(pascal_to_snake_case(&plan_status))
                },
            });
        }
        Ok(tasks)
    }

    /// Get all steps for all tasks linked to a workspace milestone (batch query)
    pub async fn get_workspace_milestone_steps(
        &self,
        milestone_id: Uuid,
    ) -> Result<std::collections::HashMap<Uuid, Vec<StepNode>>> {
        let q = query(
            r#"
            MATCH (wm:WorkspaceMilestone {id: $milestone_id})-[:INCLUDES_TASK]->(t:Task)-[:HAS_STEP]->(s:Step)
            RETURN t.id AS task_id, s
            ORDER BY t.id, s.order
            "#,
        )
        .param("milestone_id", milestone_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut steps_map: std::collections::HashMap<Uuid, Vec<StepNode>> =
            std::collections::HashMap::new();

        while let Some(row) = result.next().await? {
            let task_id_str: String = row.get("task_id")?;
            let task_id: Uuid = task_id_str.parse()?;
            let node: neo4rs::Node = row.get("s")?;
            let step = StepNode {
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
            };
            steps_map.entry(task_id).or_default().push(step);
        }

        Ok(steps_map)
    }

    /// Helper to convert Neo4j node to WorkspaceMilestoneNode
    fn node_to_workspace_milestone(&self, node: &neo4rs::Node) -> Result<WorkspaceMilestoneNode> {
        let status_str: String = node.get("status").unwrap_or_else(|_| "Open".to_string());
        let status =
            serde_json::from_str::<MilestoneStatus>(&format!("\"{}\"", status_str.to_lowercase()))
                .unwrap_or(MilestoneStatus::Open);

        let tags: Vec<String> = node.get("tags").unwrap_or_else(|_| vec![]);

        Ok(WorkspaceMilestoneNode {
            id: node.get::<String>("id")?.parse()?,
            workspace_id: node.get::<String>("workspace_id")?.parse()?,
            title: node.get("title")?,
            description: node.get("description").ok(),
            status,
            target_date: node
                .get::<String>("target_date")
                .ok()
                .and_then(|s| s.parse().ok()),
            closed_at: node
                .get::<String>("closed_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            tags,
        })
    }

    // ========================================================================
    // Resource operations
    // ========================================================================

    /// Create a resource
    pub async fn create_resource(&self, resource: &ResourceNode) -> Result<()> {
        let q = query(
            r#"
            CREATE (r:Resource {
                id: $id,
                workspace_id: $workspace_id,
                project_id: $project_id,
                name: $name,
                resource_type: $resource_type,
                file_path: $file_path,
                url: $url,
                format: $format,
                version: $version,
                description: $description,
                created_at: datetime($created_at),
                metadata: $metadata
            })
            "#,
        )
        .param("id", resource.id.to_string())
        .param(
            "workspace_id",
            resource
                .workspace_id
                .map(|id| id.to_string())
                .unwrap_or_default(),
        )
        .param(
            "project_id",
            resource
                .project_id
                .map(|id| id.to_string())
                .unwrap_or_default(),
        )
        .param("name", resource.name.clone())
        .param("resource_type", format!("{:?}", resource.resource_type))
        .param("file_path", resource.file_path.clone())
        .param("url", resource.url.clone().unwrap_or_default())
        .param("format", resource.format.clone().unwrap_or_default())
        .param("version", resource.version.clone().unwrap_or_default())
        .param(
            "description",
            resource.description.clone().unwrap_or_default(),
        )
        .param("created_at", resource.created_at.to_rfc3339())
        .param("metadata", resource.metadata.to_string());

        self.graph.run(q).await?;

        // Link to workspace if specified
        if let Some(workspace_id) = resource.workspace_id {
            let link_q = query(
                r#"
                MATCH (w:Workspace {id: $workspace_id})
                MATCH (r:Resource {id: $resource_id})
                MERGE (w)-[:HAS_RESOURCE]->(r)
                "#,
            )
            .param("workspace_id", workspace_id.to_string())
            .param("resource_id", resource.id.to_string());
            self.graph.run(link_q).await?;
        }

        // Link to project if specified
        if let Some(project_id) = resource.project_id {
            let link_q = query(
                r#"
                MATCH (p:Project {id: $project_id})
                MATCH (r:Resource {id: $resource_id})
                MERGE (p)-[:HAS_RESOURCE]->(r)
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("resource_id", resource.id.to_string());
            self.graph.run(link_q).await?;
        }

        Ok(())
    }

    /// Get a resource by ID
    pub async fn get_resource(&self, id: Uuid) -> Result<Option<ResourceNode>> {
        let q = query(
            r#"
            MATCH (r:Resource {id: $id})
            RETURN r
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            Ok(Some(self.node_to_resource(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List workspace resources
    pub async fn list_workspace_resources(&self, workspace_id: Uuid) -> Result<Vec<ResourceNode>> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})-[:HAS_RESOURCE]->(r:Resource)
            RETURN r
            ORDER BY r.name
            "#,
        )
        .param("workspace_id", workspace_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut resources = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            resources.push(self.node_to_resource(&node)?);
        }

        Ok(resources)
    }

    /// Update a resource
    pub async fn update_resource(
        &self,
        id: Uuid,
        name: Option<String>,
        file_path: Option<String>,
        url: Option<String>,
        version: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        let mut set_clauses = vec![];
        if name.is_some() {
            set_clauses.push("r.name = $name");
        }
        if file_path.is_some() {
            set_clauses.push("r.file_path = $file_path");
        }
        if url.is_some() {
            set_clauses.push("r.url = $url");
        }
        if version.is_some() {
            set_clauses.push("r.version = $version");
        }
        if description.is_some() {
            set_clauses.push("r.description = $description");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (r:Resource {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", id.to_string());
        if let Some(name) = name {
            q = q.param("name", name);
        }
        if let Some(file_path) = file_path {
            q = q.param("file_path", file_path);
        }
        if let Some(url) = url {
            q = q.param("url", url);
        }
        if let Some(version) = version {
            q = q.param("version", version);
        }
        if let Some(description) = description {
            q = q.param("description", description);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a resource
    pub async fn delete_resource(&self, id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (r:Resource {id: $id})
            DETACH DELETE r
            "#,
        )
        .param("id", id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Link a project as implementing a resource
    pub async fn link_project_implements_resource(
        &self,
        project_id: Uuid,
        resource_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})
            MATCH (r:Resource {id: $resource_id})
            MERGE (p)-[:IMPLEMENTS_RESOURCE]->(r)
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("resource_id", resource_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Link a project as using a resource
    pub async fn link_project_uses_resource(
        &self,
        project_id: Uuid,
        resource_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})
            MATCH (r:Resource {id: $resource_id})
            MERGE (p)-[:USES_RESOURCE]->(r)
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("resource_id", resource_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get projects that implement a resource
    pub async fn get_resource_implementers(&self, resource_id: Uuid) -> Result<Vec<ProjectNode>> {
        let q = query(
            r#"
            MATCH (p:Project)-[:IMPLEMENTS_RESOURCE]->(r:Resource {id: $resource_id})
            RETURN p
            "#,
        )
        .param("resource_id", resource_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut projects = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            projects.push(self.node_to_project(&node)?);
        }

        Ok(projects)
    }

    /// Get projects that use a resource
    pub async fn get_resource_consumers(&self, resource_id: Uuid) -> Result<Vec<ProjectNode>> {
        let q = query(
            r#"
            MATCH (p:Project)-[:USES_RESOURCE]->(r:Resource {id: $resource_id})
            RETURN p
            "#,
        )
        .param("resource_id", resource_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut projects = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            projects.push(self.node_to_project(&node)?);
        }

        Ok(projects)
    }

    /// Helper to convert Neo4j node to ResourceNode
    fn node_to_resource(&self, node: &neo4rs::Node) -> Result<ResourceNode> {
        let type_str: String = node
            .get("resource_type")
            .unwrap_or_else(|_| "Other".to_string());
        let resource_type = match type_str.to_lowercase().as_str() {
            "apicontract" | "api_contract" => ResourceType::ApiContract,
            "protobuf" => ResourceType::Protobuf,
            "graphqlschema" | "graphql_schema" => ResourceType::GraphqlSchema,
            "jsonschema" | "json_schema" => ResourceType::JsonSchema,
            "databaseschema" | "database_schema" => ResourceType::DatabaseSchema,
            "sharedtypes" | "shared_types" => ResourceType::SharedTypes,
            "config" => ResourceType::Config,
            "documentation" => ResourceType::Documentation,
            _ => ResourceType::Other,
        };

        let metadata_str: String = node.get("metadata").unwrap_or_else(|_| "{}".to_string());
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_str).unwrap_or(serde_json::json!({}));

        Ok(ResourceNode {
            id: node.get::<String>("id")?.parse()?,
            workspace_id: node
                .get::<String>("workspace_id")
                .ok()
                .and_then(|s| s.parse().ok()),
            project_id: node
                .get::<String>("project_id")
                .ok()
                .and_then(|s| s.parse().ok()),
            name: node.get("name")?,
            resource_type,
            file_path: node.get("file_path")?,
            url: node.get("url").ok(),
            format: node.get("format").ok(),
            version: node.get("version").ok(),
            description: node.get("description").ok(),
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            updated_at: node
                .get::<String>("updated_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            metadata,
        })
    }

    // ========================================================================
    // Component operations (Topology)
    // ========================================================================

    /// Create a component
    pub async fn create_component(&self, component: &ComponentNode) -> Result<()> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})
            CREATE (c:Component {
                id: $id,
                workspace_id: $workspace_id,
                name: $name,
                component_type: $component_type,
                description: $description,
                runtime: $runtime,
                config: $config,
                created_at: datetime($created_at),
                tags: $tags
            })
            CREATE (w)-[:HAS_COMPONENT]->(c)
            "#,
        )
        .param("id", component.id.to_string())
        .param("workspace_id", component.workspace_id.to_string())
        .param("name", component.name.clone())
        .param("component_type", format!("{:?}", component.component_type))
        .param(
            "description",
            component.description.clone().unwrap_or_default(),
        )
        .param("runtime", component.runtime.clone().unwrap_or_default())
        .param("config", component.config.to_string())
        .param("created_at", component.created_at.to_rfc3339())
        .param("tags", component.tags.clone());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a component by ID
    pub async fn get_component(&self, id: Uuid) -> Result<Option<ComponentNode>> {
        let q = query(
            r#"
            MATCH (c:Component {id: $id})
            RETURN c
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            Ok(Some(self.node_to_component(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List components in a workspace
    pub async fn list_components(&self, workspace_id: Uuid) -> Result<Vec<ComponentNode>> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})-[:HAS_COMPONENT]->(c:Component)
            RETURN c
            ORDER BY c.name
            "#,
        )
        .param("workspace_id", workspace_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut components = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            components.push(self.node_to_component(&node)?);
        }

        Ok(components)
    }

    /// Update a component
    pub async fn update_component(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
        runtime: Option<String>,
        config: Option<serde_json::Value>,
        tags: Option<Vec<String>>,
    ) -> Result<()> {
        let mut set_clauses = vec![];
        if name.is_some() {
            set_clauses.push("c.name = $name");
        }
        if description.is_some() {
            set_clauses.push("c.description = $description");
        }
        if runtime.is_some() {
            set_clauses.push("c.runtime = $runtime");
        }
        if config.is_some() {
            set_clauses.push("c.config = $config");
        }
        if tags.is_some() {
            set_clauses.push("c.tags = $tags");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (c:Component {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", id.to_string());
        if let Some(name) = name {
            q = q.param("name", name);
        }
        if let Some(description) = description {
            q = q.param("description", description);
        }
        if let Some(runtime) = runtime {
            q = q.param("runtime", runtime);
        }
        if let Some(config) = config {
            q = q.param("config", config.to_string());
        }
        if let Some(tags) = tags {
            q = q.param("tags", tags);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a component
    pub async fn delete_component(&self, id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (c:Component {id: $id})
            DETACH DELETE c
            "#,
        )
        .param("id", id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Add a dependency between components
    pub async fn add_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
        protocol: Option<String>,
        required: bool,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (c1:Component {id: $component_id})
            MATCH (c2:Component {id: $depends_on_id})
            MERGE (c1)-[r:DEPENDS_ON_COMPONENT]->(c2)
            SET r.protocol = $protocol, r.required = $required
            "#,
        )
        .param("component_id", component_id.to_string())
        .param("depends_on_id", depends_on_id.to_string())
        .param("protocol", protocol.unwrap_or_default())
        .param("required", required);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Remove a dependency between components
    pub async fn remove_component_dependency(
        &self,
        component_id: Uuid,
        depends_on_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (c1:Component {id: $component_id})-[r:DEPENDS_ON_COMPONENT]->(c2:Component {id: $depends_on_id})
            DELETE r
            "#,
        )
        .param("component_id", component_id.to_string())
        .param("depends_on_id", depends_on_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Map a component to a project
    pub async fn map_component_to_project(
        &self,
        component_id: Uuid,
        project_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (c:Component {id: $component_id})
            MATCH (p:Project {id: $project_id})
            MERGE (c)-[:MAPS_TO_PROJECT]->(p)
            "#,
        )
        .param("component_id", component_id.to_string())
        .param("project_id", project_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get the workspace topology (all components with their dependencies)
    pub async fn get_workspace_topology(
        &self,
        workspace_id: Uuid,
    ) -> Result<Vec<(ComponentNode, Option<String>, Vec<ComponentDependency>)>> {
        let q = query(
            r#"
            MATCH (w:Workspace {id: $workspace_id})-[:HAS_COMPONENT]->(c:Component)
            OPTIONAL MATCH (c)-[:MAPS_TO_PROJECT]->(p:Project)
            OPTIONAL MATCH (c)-[d:DEPENDS_ON_COMPONENT]->(dep:Component)
            RETURN c, p.name AS project_name,
                   collect({dep_id: dep.id, protocol: d.protocol, required: d.required}) AS dependencies
            "#,
        )
        .param("workspace_id", workspace_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut topology = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            let component = self.node_to_component(&node)?;
            let project_name: Option<String> = row.get("project_name").ok();

            // Parse dependencies
            let deps_raw: Vec<serde_json::Value> =
                row.get("dependencies").unwrap_or_else(|_| vec![]);
            let mut dependencies = Vec::new();
            for dep in deps_raw {
                if let Some(dep_id_str) = dep.get("dep_id").and_then(|v| v.as_str()) {
                    if let Ok(dep_id) = dep_id_str.parse::<Uuid>() {
                        dependencies.push(ComponentDependency {
                            from_id: component.id,
                            to_id: dep_id,
                            protocol: dep
                                .get("protocol")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string()),
                            required: dep
                                .get("required")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(true),
                        });
                    }
                }
            }

            topology.push((component, project_name, dependencies));
        }

        Ok(topology)
    }

    /// Helper to convert Neo4j node to ComponentNode
    fn node_to_component(&self, node: &neo4rs::Node) -> Result<ComponentNode> {
        let type_str: String = node
            .get("component_type")
            .unwrap_or_else(|_| "Other".to_string());
        let component_type = match type_str.to_lowercase().as_str() {
            "service" => ComponentType::Service,
            "frontend" => ComponentType::Frontend,
            "worker" => ComponentType::Worker,
            "database" => ComponentType::Database,
            "messagequeue" | "message_queue" => ComponentType::MessageQueue,
            "cache" => ComponentType::Cache,
            "gateway" => ComponentType::Gateway,
            "external" => ComponentType::External,
            _ => ComponentType::Other,
        };

        let config_str: String = node.get("config").unwrap_or_else(|_| "{}".to_string());
        let config: serde_json::Value =
            serde_json::from_str(&config_str).unwrap_or(serde_json::json!({}));

        let tags: Vec<String> = node.get("tags").unwrap_or_else(|_| vec![]);

        Ok(ComponentNode {
            id: node.get::<String>("id")?.parse()?,
            workspace_id: node.get::<String>("workspace_id")?.parse()?,
            name: node.get("name")?,
            component_type,
            description: node.get("description").ok(),
            runtime: node.get("runtime").ok(),
            config,
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            tags,
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

    /// Delete a file and all its symbols
    pub async fn delete_file(&self, path: &str) -> Result<()> {
        let q = query(
            r#"
            MATCH (f:File {path: $path})
            OPTIONAL MATCH (f)-[:CONTAINS]->(symbol)
            DETACH DELETE symbol, f
            "#,
        )
        .param("path", path);

        self.graph.run(q).await?;
        Ok(())
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
            let linked: i64 = rows.first().and_then(|r| r.get("linked").ok()).unwrap_or(0);

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
        if matches!(
            name,
            "Serialize" | "Deserialize" | "Serializer" | "Deserializer"
        ) {
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
        if matches!(
            name,
            "IntoResponse" | "FromRequest" | "FromRequestParts" | "Service" | "Layer"
        ) {
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

    /// Create a CALLS relationship between functions, scoped to the same project.
    /// When project_id is provided, the callee is matched only within the same project
    /// to prevent cross-project CALLS pollution.
    pub async fn create_call_relationship(
        &self,
        caller_id: &str,
        callee_name: &str,
        project_id: Option<Uuid>,
    ) -> Result<()> {
        let q = match project_id {
            Some(pid) => query(
                r#"
                MATCH (caller:Function {id: $caller_id})
                MATCH (callee:Function {name: $callee_name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                MERGE (caller)-[:CALLS]->(callee)
                "#,
            )
            .param("caller_id", caller_id)
            .param("callee_name", callee_name)
            .param("project_id", pid.to_string()),
            None => query(
                r#"
                MATCH (caller:Function {id: $caller_id})
                MATCH (callee:Function {name: $callee_name})
                MERGE (caller)-[:CALLS]->(callee)
                "#,
            )
            .param("caller_id", caller_id)
            .param("callee_name", callee_name),
        };

        // Ignore errors if callee not found (might be external)
        let _ = self.graph.run(q).await;
        Ok(())
    }

    /// Delete all CALLS relationships where caller and callee belong to different projects.
    ///
    /// Returns the number of deleted relationships.
    pub async fn cleanup_cross_project_calls(&self) -> Result<i64> {
        let q = query(
            r#"
            MATCH (caller:Function)-[r:CALLS]->(callee:Function)
            WHERE NOT EXISTS {
                MATCH (caller)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project),
                      (callee)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p)
            }
            DELETE r
            RETURN count(r) AS deleted
            "#,
        );
        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let deleted: i64 = row.get("deleted")?;
            Ok(deleted)
        } else {
            Ok(0)
        }
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

    /// Get all impl blocks for a type
    pub async fn get_impl_blocks(&self, type_name: &str) -> Result<Vec<serde_json::Value>> {
        let q = query(
            r#"
            MATCH (type {name: $type_name})<-[:IMPLEMENTS_FOR]-(i:Impl)
            OPTIONAL MATCH (i)-[:IMPLEMENTS_TRAIT]->(t:Trait)
            RETURN i.id AS impl_id,
                   i.file_path AS file_path,
                   i.start_line AS start_line,
                   i.end_line AS end_line,
                   t.name AS trait_name,
                   t.is_external AS is_external
            "#,
        )
        .param("type_name", type_name);

        let mut result = self.graph.execute(q).await?;
        let mut impl_blocks = Vec::new();

        while let Some(row) = result.next().await? {
            let file_path: String = row.get("file_path").unwrap_or_default();
            let start_line: i64 = row.get("start_line").unwrap_or(0);
            let end_line: i64 = row.get("end_line").unwrap_or(0);
            let trait_name: Option<String> = row.get("trait_name").ok();
            let is_external: bool = row.get("is_external").unwrap_or(false);

            impl_blocks.push(serde_json::json!({
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "trait_name": trait_name,
                "is_external": is_external
            }));
        }

        Ok(impl_blocks)
    }

    // ========================================================================
    // Code exploration queries (encapsulated from handlers)
    // ========================================================================

    /// Get the language of a file by path
    pub async fn get_file_language(&self, path: &str) -> Result<Option<String>> {
        let q = query(
            r#"
            MATCH (f:File {path: $path})
            RETURN f.language AS language
            "#,
        )
        .param("path", path);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            Ok(row.get("language").ok())
        } else {
            Ok(None)
        }
    }

    /// Get function summaries for a file
    pub async fn get_file_functions_summary(&self, path: &str) -> Result<Vec<FunctionSummaryNode>> {
        let q = query(
            r#"
            MATCH (f:File {path: $path})-[:CONTAINS]->(func:Function)
            RETURN func
            ORDER BY func.line_start
            "#,
        )
        .param("path", path);

        let mut result = self.graph.execute(q).await?;
        let mut functions = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("func")?;
            let name: String = node.get("name")?;
            let is_async: bool = node.get("is_async").unwrap_or(false);
            let visibility: String = node.get("visibility").unwrap_or_default();
            let is_public = visibility == "public";
            let line: i64 = node.get("line_start").unwrap_or(0);
            let complexity: i64 = node.get("complexity").unwrap_or(1);
            let docstring: Option<String> = node.get("docstring").ok();
            let params: Vec<String> = node.get("params").unwrap_or_default();
            let return_type: String = node.get("return_type").unwrap_or_default();
            let async_prefix = if is_async { "async " } else { "" };
            let signature = format!(
                "{}fn {}({}){}",
                async_prefix,
                name,
                params.join(", "),
                if return_type.is_empty() {
                    String::new()
                } else {
                    format!(" -> {}", return_type)
                }
            );

            functions.push(FunctionSummaryNode {
                name,
                signature,
                line: line as u32,
                is_async,
                is_public,
                complexity: complexity as u32,
                docstring,
            });
        }

        Ok(functions)
    }

    /// Get struct summaries for a file
    pub async fn get_file_structs_summary(&self, path: &str) -> Result<Vec<StructSummaryNode>> {
        let q = query(
            r#"
            MATCH (f:File {path: $path})-[:CONTAINS]->(s:Struct)
            RETURN s
            ORDER BY s.line_start
            "#,
        )
        .param("path", path);

        let mut result = self.graph.execute(q).await?;
        let mut structs = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            let name: String = node.get("name")?;
            let visibility: String = node.get("visibility").unwrap_or_default();
            let is_public = visibility == "public";
            let line: i64 = node.get("line_start").unwrap_or(0);
            let docstring: Option<String> = node.get("docstring").ok();

            structs.push(StructSummaryNode {
                name,
                line: line as u32,
                is_public,
                docstring,
            });
        }

        Ok(structs)
    }

    /// Get import paths for a file
    pub async fn get_file_import_paths_list(&self, path: &str) -> Result<Vec<String>> {
        let q = query(
            r#"
            MATCH (f:File {path: $path})-[:CONTAINS]->(i:Import)
            RETURN i.path AS path
            ORDER BY i.line
            "#,
        )
        .param("path", path);

        let mut result = self.graph.execute(q).await?;
        let mut imports = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(p) = row.get::<String>("path") {
                imports.push(p);
            }
        }

        Ok(imports)
    }

    /// Find references to a symbol (function callers, struct importers, file importers).
    /// When project_id is provided, results are scoped to the same project.
    pub async fn find_symbol_references(
        &self,
        symbol: &str,
        limit: usize,
        project_id: Option<Uuid>,
    ) -> Result<Vec<SymbolReferenceNode>> {
        let mut references = Vec::new();
        let limit_i64 = limit as i64;

        // Find function callers
        let q = match project_id {
            Some(pid) => query(
                r#"
                MATCH (f:Function {name: $name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
                WHERE caller IS NOT NULL
                  AND EXISTS { MATCH (caller)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }
                RETURN 'call' AS ref_type,
                       caller.file_path AS file_path,
                       caller.line_start AS line,
                       caller.name AS context
                LIMIT $limit
                "#,
            )
            .param("name", symbol)
            .param("limit", limit_i64)
            .param("project_id", pid.to_string()),
            None => query(
                r#"
                MATCH (f:Function {name: $name})
                OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
                WHERE caller IS NOT NULL
                RETURN 'call' AS ref_type,
                       caller.file_path AS file_path,
                       caller.line_start AS line,
                       caller.name AS context
                LIMIT $limit
                "#,
            )
            .param("name", symbol)
            .param("limit", limit_i64),
        };

        let mut result = self.graph.execute(q).await?;
        while let Some(row) = result.next().await? {
            if let (Ok(file_path), Ok(line), Ok(context)) = (
                row.get::<String>("file_path"),
                row.get::<i64>("line"),
                row.get::<String>("context"),
            ) {
                references.push(SymbolReferenceNode {
                    file_path,
                    line: line as u32,
                    context: format!("called from {}", context),
                    reference_type: "call".to_string(),
                });
            }
        }

        // Find struct import usages
        let q = match project_id {
            Some(pid) => query(
                r#"
                MATCH (s:Struct {name: $name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                OPTIONAL MATCH (i:Import)-[:IMPORTS_SYMBOL]->(s)
                WHERE i IS NOT NULL
                  AND EXISTS { MATCH (:File {path: i.file_path})<-[:CONTAINS]-(p) }
                RETURN 'import' AS ref_type,
                       i.file_path AS file_path,
                       i.line AS line,
                       i.path AS context
                LIMIT $limit
                "#,
            )
            .param("name", symbol)
            .param("limit", limit_i64)
            .param("project_id", pid.to_string()),
            None => query(
                r#"
                MATCH (s:Struct {name: $name})
                OPTIONAL MATCH (i:Import)-[:IMPORTS_SYMBOL]->(s)
                WHERE i IS NOT NULL
                RETURN 'import' AS ref_type,
                       i.file_path AS file_path,
                       i.line AS line,
                       i.path AS context
                LIMIT $limit
                "#,
            )
            .param("name", symbol)
            .param("limit", limit_i64),
        };

        let mut result = self.graph.execute(q).await?;
        while let Some(row) = result.next().await? {
            if let (Ok(file_path), Ok(line), Ok(context)) = (
                row.get::<String>("file_path"),
                row.get::<i64>("line"),
                row.get::<String>("context"),
            ) {
                references.push(SymbolReferenceNode {
                    file_path,
                    line: line as u32,
                    context: format!("imported via {}", context),
                    reference_type: "import".to_string(),
                });
            }
        }

        // Find files importing the symbol's module
        let q = match project_id {
            Some(pid) => query(
                r#"
                MATCH (s {name: $name})
                WHERE s:Function OR s:Struct OR s:Trait OR s:Enum
                MATCH (f:File {path: s.file_path})<-[:CONTAINS]-(p:Project {id: $project_id})
                OPTIONAL MATCH (importer:File)-[:IMPORTS]->(f)
                WHERE importer IS NOT NULL
                  AND EXISTS { MATCH (importer)<-[:CONTAINS]-(p) }
                RETURN 'file_import' AS ref_type,
                       importer.path AS file_path,
                       0 AS line,
                       f.path AS context
                LIMIT $limit
                "#,
            )
            .param("name", symbol)
            .param("limit", limit_i64)
            .param("project_id", pid.to_string()),
            None => query(
                r#"
                MATCH (s {name: $name})
                WHERE s:Function OR s:Struct OR s:Trait OR s:Enum
                MATCH (f:File {path: s.file_path})
                OPTIONAL MATCH (importer:File)-[:IMPORTS]->(f)
                WHERE importer IS NOT NULL
                RETURN 'file_import' AS ref_type,
                       importer.path AS file_path,
                       0 AS line,
                       f.path AS context
                LIMIT $limit
                "#,
            )
            .param("name", symbol)
            .param("limit", limit_i64),
        };

        let mut result = self.graph.execute(q).await?;
        while let Some(row) = result.next().await? {
            if let (Ok(file_path), Ok(context)) =
                (row.get::<String>("file_path"), row.get::<String>("context"))
            {
                references.push(SymbolReferenceNode {
                    file_path,
                    line: 0,
                    context: format!("imports module {}", context),
                    reference_type: "file_import".to_string(),
                });
            }
        }

        Ok(references)
    }

    /// Get files directly imported by a file
    pub async fn get_file_direct_imports(&self, path: &str) -> Result<Vec<FileImportNode>> {
        let q = query(
            r#"
            MATCH (f:File {path: $path})-[:IMPORTS]->(imported:File)
            RETURN imported.path AS path, imported.language AS language
            "#,
        )
        .param("path", path);

        let mut result = self.graph.execute(q).await?;
        let mut imports = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(p) = row.get::<String>("path") {
                imports.push(FileImportNode {
                    path: p,
                    language: row.get("language").unwrap_or_default(),
                });
            }
        }

        Ok(imports)
    }

    /// Get callers chain for a function name (by name, variable depth).
    /// When project_id is provided, scopes the start function and callers to the same project.
    pub async fn get_function_callers_by_name(
        &self,
        function_name: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> Result<Vec<String>> {
        let q = match project_id {
            Some(pid) => query(&format!(
                r#"
                MATCH (f:Function {{name: $name}})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {{id: $project_id}})
                MATCH (caller:Function)-[:CALLS*1..{}]->(f)
                WHERE EXISTS {{ MATCH (caller)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }}
                RETURN DISTINCT caller.name AS name, caller.file_path AS file
                "#,
                depth
            ))
            .param("name", function_name)
            .param("project_id", pid.to_string()),
            None => query(&format!(
                r#"
                MATCH (f:Function {{name: $name}})
                MATCH (caller:Function)-[:CALLS*1..{}]->(f)
                RETURN DISTINCT caller.name AS name, caller.file_path AS file
                "#,
                depth
            ))
            .param("name", function_name),
        };

        let mut result = self.graph.execute(q).await?;
        let mut callers = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(name) = row.get::<String>("name") {
                callers.push(name);
            }
        }

        Ok(callers)
    }

    /// Get callees chain for a function name (by name, variable depth).
    /// When project_id is provided, scopes the start function and callees to the same project.
    pub async fn get_function_callees_by_name(
        &self,
        function_name: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> Result<Vec<String>> {
        let q = match project_id {
            Some(pid) => query(&format!(
                r#"
                MATCH (f:Function {{name: $name}})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {{id: $project_id}})
                MATCH (f)-[:CALLS*1..{}]->(callee:Function)
                WHERE EXISTS {{ MATCH (callee)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }}
                RETURN DISTINCT callee.name AS name, callee.file_path AS file
                "#,
                depth
            ))
            .param("name", function_name)
            .param("project_id", pid.to_string()),
            None => query(&format!(
                r#"
                MATCH (f:Function {{name: $name}})
                MATCH (f)-[:CALLS*1..{}]->(callee:Function)
                RETURN DISTINCT callee.name AS name, callee.file_path AS file
                "#,
                depth
            ))
            .param("name", function_name),
        };

        let mut result = self.graph.execute(q).await?;
        let mut callees = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(name) = row.get::<String>("name") {
                callees.push(name);
            }
        }

        Ok(callees)
    }

    /// Get language statistics across all files
    pub async fn get_language_stats(&self) -> Result<Vec<LanguageStatsNode>> {
        let q = query(
            r#"
            MATCH (f:File)
            RETURN f.language AS language, count(f) AS count
            ORDER BY count DESC
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut stats = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(language), Ok(count)) =
                (row.get::<String>("language"), row.get::<i64>("count"))
            {
                stats.push(LanguageStatsNode {
                    language,
                    file_count: count as usize,
                });
            }
        }

        Ok(stats)
    }

    /// Get language stats for a specific project
    pub async fn get_language_stats_for_project(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<LanguageStatsNode>> {
        let q = query(
            r#"
            MATCH (f:File)
            WHERE f.project_id = $project_id
            RETURN f.language AS language, count(f) AS count
            ORDER BY count DESC
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut stats = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(language), Ok(count)) =
                (row.get::<String>("language"), row.get::<i64>("count"))
            {
                stats.push(LanguageStatsNode {
                    language,
                    file_count: count as usize,
                });
            }
        }

        Ok(stats)
    }

    /// Get most connected files (highest in-degree from imports)
    pub async fn get_most_connected_files(&self, limit: usize) -> Result<Vec<String>> {
        let q = query(
            r#"
            MATCH (f:File)<-[:IMPORTS]-(importer:File)
            RETURN f.path AS path, count(importer) AS imports
            ORDER BY imports DESC
            LIMIT $limit
            "#,
        )
        .param("limit", limit as i64);

        let mut result = self.graph.execute(q).await?;
        let mut paths = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(path) = row.get::<String>("path") {
                paths.push(path);
            }
        }

        Ok(paths)
    }

    /// Get most connected files with import/dependent counts
    pub async fn get_most_connected_files_detailed(
        &self,
        limit: usize,
    ) -> Result<Vec<ConnectedFileNode>> {
        let q = query(
            r#"
            MATCH (f:File)
            OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File)
            OPTIONAL MATCH (dependent:File)-[:IMPORTS]->(f)
            WITH f, count(DISTINCT imported) AS imports, count(DISTINCT dependent) AS dependents
            RETURN f.path AS path, imports, dependents, imports + dependents AS connections,
                   f.pagerank AS pagerank, f.betweenness AS betweenness,
                   f.community_label AS community_label, f.community_id AS community_id
            ORDER BY COALESCE(f.pagerank, 0) DESC, connections DESC
            LIMIT $limit
            "#,
        )
        .param("limit", limit as i64);

        let mut result = self.graph.execute(q).await?;
        let mut files = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(path) = row.get::<String>("path") {
                files.push(ConnectedFileNode {
                    path,
                    imports: row.get("imports").unwrap_or(0),
                    dependents: row.get("dependents").unwrap_or(0),
                    pagerank: row.get::<f64>("pagerank").ok(),
                    betweenness: row.get::<f64>("betweenness").ok(),
                    community_label: row.get::<String>("community_label").ok(),
                    community_id: row.get::<i64>("community_id").ok(),
                });
            }
        }

        Ok(files)
    }

    /// Get most connected files with import/dependent counts for a specific project
    pub async fn get_most_connected_files_for_project(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<ConnectedFileNode>> {
        let q = query(
            r#"
            MATCH (f:File)
            WHERE f.project_id = $project_id
            OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File)
            OPTIONAL MATCH (dependent:File)-[:IMPORTS]->(f)
            WITH f, count(DISTINCT imported) AS imports, count(DISTINCT dependent) AS dependents
            RETURN f.path AS path, imports, dependents, imports + dependents AS connections,
                   f.pagerank AS pagerank, f.betweenness AS betweenness,
                   f.community_label AS community_label, f.community_id AS community_id
            ORDER BY COALESCE(f.pagerank, 0) DESC, connections DESC
            LIMIT $limit
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("limit", limit as i64);

        let mut result = self.graph.execute(q).await?;
        let mut files = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(path) = row.get::<String>("path") {
                files.push(ConnectedFileNode {
                    path,
                    imports: row.get("imports").unwrap_or(0),
                    dependents: row.get("dependents").unwrap_or(0),
                    pagerank: row.get::<f64>("pagerank").ok(),
                    betweenness: row.get::<f64>("betweenness").ok(),
                    community_label: row.get::<String>("community_label").ok(),
                    community_id: row.get::<i64>("community_id").ok(),
                });
            }
        }

        Ok(files)
    }

    /// Get distinct communities for a project (from graph analytics Louvain clustering).
    /// Returns communities sorted by file_count descending.
    pub async fn get_project_communities(&self, project_id: Uuid) -> Result<Vec<CommunityRow>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)
            WHERE f.community_id IS NOT NULL
            WITH f.community_id AS cid, f.community_label AS label,
                 count(f) AS file_count,
                 collect(f.path) AS all_paths
            ORDER BY file_count DESC
            RETURN cid, label, file_count,
                   [p IN all_paths | p][..3] AS key_files
            "#,
        )
        .param("pid", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut communities = Vec::new();

        while let Some(row) = result.next().await? {
            let community_id = row.get::<i64>("cid").unwrap_or(0);
            let community_label = row
                .get::<String>("label")
                .unwrap_or_else(|_| format!("Community {}", community_id));
            let file_count = row.get::<i64>("file_count").unwrap_or(0) as usize;
            let key_files: Vec<String> = row.get::<Vec<String>>("key_files").unwrap_or_default();

            communities.push(CommunityRow {
                community_id,
                community_label,
                file_count,
                key_files,
            });
        }

        Ok(communities)
    }

    /// Get GDS analytics properties for a node (File by path, or Function by name).
    pub async fn get_node_analytics(
        &self,
        identifier: &str,
        node_type: &str,
    ) -> Result<Option<NodeAnalyticsRow>> {
        let cypher = if node_type == "function" {
            r#"
            MATCH (n:Function {name: $id})
            RETURN n.pagerank AS pagerank, n.betweenness AS betweenness,
                   n.community_id AS community_id, n.community_label AS community_label
            LIMIT 1
            "#
        } else {
            r#"
            MATCH (n:File {path: $id})
            RETURN n.pagerank AS pagerank, n.betweenness AS betweenness,
                   n.community_id AS community_id, n.community_label AS community_label
            LIMIT 1
            "#
        };

        let q = query(cypher).param("id", identifier);
        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            Ok(Some(NodeAnalyticsRow {
                pagerank: row.get::<f64>("pagerank").ok(),
                betweenness: row.get::<f64>("betweenness").ok(),
                community_id: row.get::<i64>("community_id").ok(),
                community_label: row.get::<String>("community_label").ok(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Get distinct community labels for a list of file paths.
    pub async fn get_affected_communities(&self, file_paths: &[String]) -> Result<Vec<String>> {
        if file_paths.is_empty() {
            return Ok(vec![]);
        }

        let q = query(
            r#"
            MATCH (f:File)
            WHERE f.path IN $paths AND f.community_label IS NOT NULL
            RETURN DISTINCT f.community_label AS label
            ORDER BY label
            "#,
        )
        .param("paths", file_paths.to_vec());

        let mut result = self.graph.execute(q).await?;
        let mut labels = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(label) = row.get::<String>("label") {
                labels.push(label);
            }
        }

        Ok(labels)
    }

    /// Get a structural health report for a project: god functions, orphan files, coupling metrics.
    pub async fn get_code_health_report(
        &self,
        project_id: Uuid,
        god_function_threshold: usize,
    ) -> Result<CodeHealthReport> {
        use crate::neo4j::models::{CodeHealthReport, CouplingMetrics, GodFunction};

        // God functions: functions with high in-degree (many callers)
        let god_q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)-[:CONTAINS]->(func:Function)
            OPTIONAL MATCH (caller:Function)-[:CALLS]->(func)
            OPTIONAL MATCH (func)-[:CALLS]->(callee:Function)
            WITH func, f, count(DISTINCT caller) AS in_deg, count(DISTINCT callee) AS out_deg
            WHERE in_deg >= $threshold
            RETURN func.name AS name, f.path AS file, in_deg, out_deg
            ORDER BY in_deg DESC
            LIMIT 10
            "#,
        )
        .param("pid", project_id.to_string())
        .param("threshold", god_function_threshold as i64);

        let god_rows = self.execute_with_params(god_q).await?;
        let god_functions: Vec<GodFunction> = god_rows
            .iter()
            .filter_map(|row| {
                let name = row.get::<String>("name").ok()?;
                let file = row.get::<String>("file").ok()?;
                let in_degree = row.get::<i64>("in_deg").unwrap_or(0) as usize;
                let out_degree = row.get::<i64>("out_deg").unwrap_or(0) as usize;
                Some(GodFunction {
                    name,
                    file,
                    in_degree,
                    out_degree,
                })
            })
            .collect();

        // Orphan files: files with no IMPORTS relationships (neither importing nor imported)
        let orphan_q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)
            WHERE NOT EXISTS { (f)-[:IMPORTS]->() }
              AND NOT EXISTS { ()-[:IMPORTS]->(f) }
              AND NOT EXISTS { (f)-[:CONTAINS]->(:Function) }
            RETURN f.path AS path
            ORDER BY path
            LIMIT 20
            "#,
        )
        .param("pid", project_id.to_string());

        let orphan_rows = self.execute_with_params(orphan_q).await?;
        let orphan_files: Vec<String> = orphan_rows
            .iter()
            .filter_map(|row| row.get::<String>("path").ok())
            .collect();

        // Coupling metrics from clustering_coefficient
        let coupling_q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)
            WHERE f.clustering_coefficient IS NOT NULL
            WITH avg(f.clustering_coefficient) AS avg_cc,
                 max(f.clustering_coefficient) AS max_cc,
                 collect({path: f.path, cc: f.clustering_coefficient}) AS files
            WITH avg_cc, max_cc, files,
                 [x IN files WHERE x.cc = max_cc | x.path][0] AS most_coupled
            RETURN avg_cc, max_cc, most_coupled
            "#,
        )
        .param("pid", project_id.to_string());

        let coupling_rows = self.execute_with_params(coupling_q).await?;
        let coupling_metrics = coupling_rows.first().and_then(|row| {
            let avg = row.get::<f64>("avg_cc").ok()?;
            let max = row.get::<f64>("max_cc").ok()?;
            let most_coupled = row.get::<String>("most_coupled").ok();
            Some(CouplingMetrics {
                avg_clustering_coefficient: avg,
                max_clustering_coefficient: max,
                most_coupled_file: most_coupled,
            })
        });

        Ok(CodeHealthReport {
            god_functions,
            orphan_files,
            coupling_metrics,
        })
    }

    /// Detect circular dependencies between files (import cycles).
    pub async fn get_circular_dependencies(&self, project_id: Uuid) -> Result<Vec<Vec<String>>> {
        let q = query(
            r#"
            MATCH path = (f:File)-[:IMPORTS*2..5]->(f)
            WHERE EXISTS { MATCH (p:Project {id: $pid})-[:CONTAINS]->(f) }
            WITH nodes(path) AS cycle_nodes
            WITH [n IN cycle_nodes | n.path] AS cycle
            WITH cycle, cycle[0] AS canonical
            RETURN DISTINCT cycle
            ORDER BY size(cycle)
            LIMIT 10
            "#,
        )
        .param("pid", project_id.to_string());

        let rows = self.execute_with_params(q).await?;
        let mut cycles: Vec<Vec<String>> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        for row in &rows {
            if let Ok(cycle) = row.get::<Vec<String>>("cycle") {
                // Deduplicate: normalize by sorting the cycle to a canonical form
                let mut canonical = cycle.clone();
                canonical.sort();
                let key = canonical.join("|");
                if seen.insert(key) {
                    cycles.push(cycle);
                }
            }
        }

        Ok(cycles)
    }

    /// Get GDS metrics for a specific node (file or function) in a project.
    pub async fn get_node_gds_metrics(
        &self,
        node_path: &str,
        node_type: &str,
        project_id: Uuid,
    ) -> Result<Option<NodeGdsMetrics>> {
        let (id_prop, match_pattern) = match node_type {
            "function" => (
                "name",
                "MATCH (p:Project {id: $pid})-[:CONTAINS]->(:File)-[:CONTAINS]->(n:Function {name: $node_path})",
            ),
            _ => (
                "path",
                "MATCH (p:Project {id: $pid})-[:CONTAINS]->(n:File {path: $node_path})",
            ),
        };

        let cypher = format!(
            r#"
            {match_pattern}
            OPTIONAL MATCH (caller)-[]->(n)
            WITH n, count(DISTINCT caller) AS in_deg
            OPTIONAL MATCH (n)-[]->(callee)
            WITH n, in_deg, count(DISTINCT callee) AS out_deg
            RETURN
                n.{id_prop} AS node_path,
                n.pagerank AS pagerank,
                n.betweenness AS betweenness,
                n.clustering_coefficient AS clustering_coefficient,
                n.community_id AS community_id,
                n.community_label AS community_label,
                in_deg, out_deg
            "#,
        );

        let q = query(&cypher)
            .param("pid", project_id.to_string())
            .param("node_path", node_path);

        let rows = self.execute_with_params(q).await?;
        if let Some(row) = rows.first() {
            let path: String = row.get("node_path").unwrap_or_default();
            if path.is_empty() {
                return Ok(None);
            }
            Ok(Some(NodeGdsMetrics {
                node_path: path,
                node_type: node_type.to_string(),
                pagerank: row.get::<f64>("pagerank").ok(),
                betweenness: row.get::<f64>("betweenness").ok(),
                clustering_coefficient: row.get::<f64>("clustering_coefficient").ok(),
                community_id: row.get::<i64>("community_id").ok(),
                community_label: row.get::<String>("community_label").ok(),
                in_degree: row.get::<i64>("in_deg").unwrap_or(0),
                out_degree: row.get::<i64>("out_deg").unwrap_or(0),
            }))
        } else {
            Ok(None)
        }
    }

    /// Get statistical percentiles for GDS metrics across all files+functions in a project.
    pub async fn get_project_percentiles(&self, project_id: Uuid) -> Result<ProjectPercentiles> {
        let q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(n)
            WHERE (n:File OR n:Function) AND n.pagerank IS NOT NULL
            WITH collect(toFloat(n.pagerank)) AS prs, collect(toFloat(COALESCE(n.betweenness, 0.0))) AS bws
            WITH prs, bws,
                 apoc.coll.sort(prs) AS sorted_pr,
                 apoc.coll.sort(bws) AS sorted_bw
            WITH sorted_pr, sorted_bw, size(sorted_pr) AS cnt,
                 reduce(s = 0.0, x IN bws | s + x) / size(bws) AS bw_mean
            WITH sorted_pr, sorted_bw, cnt, bw_mean,
                 sorted_pr[toInteger(cnt * 0.5)] AS pr_p50,
                 sorted_pr[toInteger(cnt * 0.8)] AS pr_p80,
                 sorted_pr[toInteger(cnt * 0.95)] AS pr_p95,
                 sorted_bw[toInteger(cnt * 0.5)] AS bw_p50,
                 sorted_bw[toInteger(cnt * 0.8)] AS bw_p80,
                 sorted_bw[toInteger(cnt * 0.95)] AS bw_p95
            WITH *, reduce(s = 0.0, x IN sorted_bw | s + (x - bw_mean) * (x - bw_mean)) / cnt AS bw_var
            RETURN pr_p50, pr_p80, pr_p95, bw_p50, bw_p80, bw_p95, bw_mean, sqrt(bw_var) AS bw_stddev
            "#,
        )
        .param("pid", project_id.to_string());

        let rows = self.execute_with_params(q).await?;
        if let Some(row) = rows.first() {
            Ok(ProjectPercentiles {
                pagerank_p50: row.get::<f64>("pr_p50").unwrap_or(0.0),
                pagerank_p80: row.get::<f64>("pr_p80").unwrap_or(0.0),
                pagerank_p95: row.get::<f64>("pr_p95").unwrap_or(0.0),
                betweenness_p50: row.get::<f64>("bw_p50").unwrap_or(0.0),
                betweenness_p80: row.get::<f64>("bw_p80").unwrap_or(0.0),
                betweenness_p95: row.get::<f64>("bw_p95").unwrap_or(0.0),
                betweenness_mean: row.get::<f64>("bw_mean").unwrap_or(0.0),
                betweenness_stddev: row.get::<f64>("bw_stddev").unwrap_or(0.0),
            })
        } else {
            // No data — return zeroes
            Ok(ProjectPercentiles {
                pagerank_p50: 0.0,
                pagerank_p80: 0.0,
                pagerank_p95: 0.0,
                betweenness_p50: 0.0,
                betweenness_p80: 0.0,
                betweenness_p95: 0.0,
                betweenness_mean: 0.0,
                betweenness_stddev: 0.0,
            })
        }
    }

    /// Get top N files by betweenness centrality (bridge files).
    pub async fn get_top_bridges_by_betweenness(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<BridgeFile>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)
            WHERE f.betweenness IS NOT NULL
            RETURN f.path AS path, f.betweenness AS betweenness,
                   f.community_label AS community_label
            ORDER BY f.betweenness DESC
            LIMIT $limit
            "#,
        )
        .param("pid", project_id.to_string())
        .param("limit", limit as i64);

        let rows = self.execute_with_params(q).await?;
        let mut bridges = Vec::new();
        for row in &rows {
            bridges.push(BridgeFile {
                path: row.get::<String>("path").unwrap_or_default(),
                betweenness: row.get::<f64>("betweenness").unwrap_or(0.0),
                community_label: row.get::<String>("community_label").ok(),
            });
        }
        Ok(bridges)
    }

    /// Get aggregated symbol names for a file (functions, structs, traits, enums)
    pub async fn get_file_symbol_names(&self, path: &str) -> Result<FileSymbolNamesNode> {
        let q = query(
            r#"
            MATCH (f:File {path: $path})
            OPTIONAL MATCH (f)-[:CONTAINS]->(func:Function)
            OPTIONAL MATCH (f)-[:CONTAINS]->(st:Struct)
            OPTIONAL MATCH (f)-[:CONTAINS]->(tr:Trait)
            OPTIONAL MATCH (f)-[:CONTAINS]->(en:Enum)
            RETURN
                collect(DISTINCT func.name) AS functions,
                collect(DISTINCT st.name) AS structs,
                collect(DISTINCT tr.name) AS traits,
                collect(DISTINCT en.name) AS enums
            "#,
        )
        .param("path", path);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            Ok(FileSymbolNamesNode {
                functions: row.get("functions").unwrap_or_default(),
                structs: row.get("structs").unwrap_or_default(),
                traits: row.get("traits").unwrap_or_default(),
                enums: row.get("enums").unwrap_or_default(),
            })
        } else {
            anyhow::bail!("File not found: {}", path)
        }
    }

    /// Get the number of callers for a function by name.
    /// When project_id is provided, only counts callers from the same project.
    pub async fn get_function_caller_count(
        &self,
        function_name: &str,
        project_id: Option<Uuid>,
    ) -> Result<i64> {
        let q = match project_id {
            Some(pid) => query(
                r#"
                MATCH (f:Function {name: $name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
                WHERE caller IS NOT NULL
                  AND EXISTS { MATCH (caller)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }
                RETURN count(caller) AS caller_count
                "#,
            )
            .param("name", function_name)
            .param("project_id", pid.to_string()),
            None => query(
                r#"
                MATCH (f:Function {name: $name})
                OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
                RETURN count(caller) AS caller_count
                "#,
            )
            .param("name", function_name),
        };

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            Ok(row.get("caller_count").unwrap_or(0))
        } else {
            Ok(0)
        }
    }

    /// Get trait info (is_external, source)
    pub async fn get_trait_info(&self, trait_name: &str) -> Result<Option<TraitInfoNode>> {
        let q = query(
            r#"
            MATCH (t:Trait {name: $trait_name})
            RETURN t.is_external AS is_external, t.source AS source
            LIMIT 1
            "#,
        )
        .param("trait_name", trait_name);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            Ok(Some(TraitInfoNode {
                is_external: row.get("is_external").unwrap_or(false),
                source: row.get("source").ok(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Get trait implementors with file locations
    pub async fn get_trait_implementors_detailed(
        &self,
        trait_name: &str,
    ) -> Result<Vec<TraitImplementorNode>> {
        let q = query(
            r#"
            MATCH (i:Impl)-[:IMPLEMENTS_TRAIT]->(t:Trait {name: $trait_name})
            MATCH (i)-[:IMPLEMENTS_FOR]->(type)
            RETURN type.name AS type_name, i.file_path AS file_path, i.line_start AS line
            "#,
        )
        .param("trait_name", trait_name);

        let mut result = self.graph.execute(q).await?;
        let mut implementors = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(type_name), Ok(file_path)) = (
                row.get::<String>("type_name"),
                row.get::<String>("file_path"),
            ) {
                implementors.push(TraitImplementorNode {
                    type_name,
                    file_path,
                    line: row.get::<i64>("line").unwrap_or(0) as u32,
                });
            }
        }

        Ok(implementors)
    }

    /// Get all traits implemented by a type, with details
    pub async fn get_type_trait_implementations(
        &self,
        type_name: &str,
    ) -> Result<Vec<TypeTraitInfoNode>> {
        let q = query(
            r#"
            MATCH (type {name: $type_name})<-[:IMPLEMENTS_FOR]-(i:Impl)
            OPTIONAL MATCH (i)-[:IMPLEMENTS_TRAIT]->(t:Trait)
            RETURN t.name AS trait_name,
                   t.full_path AS full_path,
                   t.file_path AS file_path,
                   t.is_external AS is_external,
                   t.source AS source
            "#,
        )
        .param("type_name", type_name);

        let mut result = self.graph.execute(q).await?;
        let mut traits = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(name) = row.get::<String>("trait_name") {
                traits.push(TypeTraitInfoNode {
                    name,
                    full_path: row.get("full_path").ok(),
                    file_path: row.get::<String>("file_path").unwrap_or_default(),
                    is_external: row.get("is_external").unwrap_or(false),
                    source: row.get("source").ok(),
                });
            }
        }

        Ok(traits)
    }

    /// Get all impl blocks for a type with methods
    pub async fn get_type_impl_blocks_detailed(
        &self,
        type_name: &str,
    ) -> Result<Vec<ImplBlockDetailNode>> {
        let q = query(
            r#"
            MATCH (type {name: $type_name})<-[:IMPLEMENTS_FOR]-(i:Impl)
            OPTIONAL MATCH (i:Impl)-[:IMPLEMENTS_TRAIT]->(t:Trait)
            OPTIONAL MATCH (f:File {path: i.file_path})-[:CONTAINS]->(func:Function)
            WHERE func.line_start >= i.line_start AND func.line_end <= i.line_end
            RETURN i.file_path AS file_path, i.line_start AS line_start, i.line_end AS line_end,
                   i.trait_name AS trait_name, collect(func.name) AS methods
            "#,
        )
        .param("type_name", type_name);

        let mut result = self.graph.execute(q).await?;
        let mut blocks = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(file_path) = row.get::<String>("file_path") {
                let trait_name: Option<String> = row.get("trait_name").ok();
                let trait_name = trait_name.filter(|s| !s.is_empty());
                blocks.push(ImplBlockDetailNode {
                    file_path,
                    line_start: row.get::<i64>("line_start").unwrap_or(0) as u32,
                    line_end: row.get::<i64>("line_end").unwrap_or(0) as u32,
                    trait_name,
                    methods: row.get("methods").unwrap_or_default(),
                });
            }
        }

        Ok(blocks)
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

    /// List plans for a project with filters
    pub async fn list_plans_for_project(
        &self,
        project_id: Uuid,
        status_filter: Option<Vec<String>>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<PlanNode>, usize)> {
        // Build status filter
        let status_clause = if let Some(statuses) = &status_filter {
            if !statuses.is_empty() {
                let status_list: Vec<String> = statuses
                    .iter()
                    .map(|s| {
                        // Convert to PascalCase for enum matching
                        let pascal = match s.to_lowercase().as_str() {
                            "draft" => "Draft",
                            "approved" => "Approved",
                            "in_progress" => "InProgress",
                            "completed" => "Completed",
                            "cancelled" => "Cancelled",
                            _ => s.as_str(),
                        };
                        format!("'{}'", pascal)
                    })
                    .collect();
                format!("AND p.status IN [{}]", status_list.join(", "))
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Count total
        let count_q = query(&format!(
            r#"
            MATCH (project:Project {{id: $project_id}})-[:HAS_PLAN]->(p:Plan)
            WHERE true {}
            RETURN count(p) AS total
            "#,
            status_clause
        ))
        .param("project_id", project_id.to_string());

        let count_rows = self.execute_with_params(count_q).await?;
        let total: i64 = count_rows
            .first()
            .and_then(|r| r.get("total").ok())
            .unwrap_or(0);

        // Get plans
        let q = query(&format!(
            r#"
            MATCH (project:Project {{id: $project_id}})-[:HAS_PLAN]->(p:Plan)
            WHERE true {}
            RETURN p
            ORDER BY p.priority DESC, p.created_at DESC
            SKIP $offset
            LIMIT $limit
            "#,
            status_clause
        ))
        .param("project_id", project_id.to_string())
        .param("offset", offset as i64)
        .param("limit", limit as i64);

        let mut result = self.graph.execute(q).await?;
        let mut plans = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            plans.push(self.node_to_plan(&node)?);
        }

        Ok((plans, total as usize))
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

    /// Delete a plan and all its related data (tasks, steps, decisions, constraints)
    pub async fn delete_plan(&self, plan_id: Uuid) -> Result<()> {
        // Delete all steps belonging to tasks of this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})-[:HAS_TASK]->(t:Task)-[:HAS_STEP]->(s:Step)
            DETACH DELETE s
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        // Delete all decisions belonging to tasks of this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})-[:HAS_TASK]->(t:Task)-[:INFORMED_BY]->(d:Decision)
            DETACH DELETE d
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        // Delete all tasks belonging to this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})-[:HAS_TASK]->(t:Task)
            DETACH DELETE t
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        // Delete all constraints belonging to this plan
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})-[:CONSTRAINED_BY]->(c:Constraint)
            DETACH DELETE c
            "#,
        )
        .param("id", plan_id.to_string());
        self.graph.run(q).await?;

        // Delete the plan itself
        let q = query(
            r#"
            MATCH (p:Plan {id: $id})
            DETACH DELETE p
            "#,
        )
        .param("id", plan_id.to_string());
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

    /// Convert a Neo4j Node to a StepNode
    fn node_to_step(&self, node: &neo4rs::Node) -> Option<StepNode> {
        Some(StepNode {
            id: node.get::<String>("id").ok()?.parse().ok()?,
            order: node.get::<i64>("order").ok()? as u32,
            description: node.get::<String>("description").ok()?,
            status: node
                .get::<String>("status")
                .ok()
                .and_then(|s| serde_json::from_str(&format!("\"{}\"", s.to_lowercase())).ok())
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
        })
    }

    /// Convert a Neo4j Node to a DecisionNode
    fn node_to_decision(&self, node: &neo4rs::Node) -> Option<DecisionNode> {
        Some(DecisionNode {
            id: node.get::<String>("id").ok()?.parse().ok()?,
            description: node.get::<String>("description").ok()?,
            rationale: node.get::<String>("rationale").ok()?,
            alternatives: node.get::<Vec<String>>("alternatives").unwrap_or_default(),
            chosen_option: node
                .get::<String>("chosen_option")
                .ok()
                .filter(|s| !s.is_empty()),
            decided_by: node.get::<String>("decided_by").ok().unwrap_or_default(),
            decided_at: node
                .get::<String>("decided_at")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(chrono::Utc::now),
        })
    }

    /// Get full task details including steps, decisions, dependencies, and modified files
    pub async fn get_task_with_full_details(&self, task_id: Uuid) -> Result<Option<TaskDetails>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $id})
            OPTIONAL MATCH (t)-[:HAS_STEP]->(s:Step)
            OPTIONAL MATCH (t)-[:INFORMED_BY]->(d:Decision)
            OPTIONAL MATCH (t)-[:DEPENDS_ON]->(dep:Task)
            OPTIONAL MATCH (t)-[:MODIFIES]->(f:File)
            RETURN t,
                   collect(DISTINCT s) AS steps,
                   collect(DISTINCT d) AS decisions,
                   collect(DISTINCT dep.id) AS depends_on,
                   collect(DISTINCT f.path) AS files
            "#,
        )
        .param("id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;

        let row = match result.next().await? {
            Some(r) => r,
            None => return Ok(None),
        };

        let task_node: neo4rs::Node = row.get("t")?;
        let task = self.node_to_task(&task_node)?;

        // Parse steps
        let step_nodes: Vec<neo4rs::Node> = row.get("steps").unwrap_or_default();
        let mut steps: Vec<StepNode> = step_nodes
            .iter()
            .filter_map(|n| self.node_to_step(n))
            .collect();
        steps.sort_by_key(|s| s.order);

        // Parse decisions
        let decision_nodes: Vec<neo4rs::Node> = row.get("decisions").unwrap_or_default();
        let decisions: Vec<DecisionNode> = decision_nodes
            .iter()
            .filter_map(|n| self.node_to_decision(n))
            .collect();

        // Parse dependencies
        let depends_on_strs: Vec<String> = row.get("depends_on").unwrap_or_default();
        let depends_on: Vec<Uuid> = depends_on_strs
            .into_iter()
            .filter_map(|s| s.parse().ok())
            .collect();

        let modifies_files: Vec<String> = row.get("files").unwrap_or_default();

        Ok(Some(TaskDetails {
            task,
            steps,
            decisions,
            depends_on,
            modifies_files,
        }))
    }

    /// Analyze the impact of a task on the codebase (files it modifies + their dependents)
    pub async fn analyze_task_impact(&self, task_id: Uuid) -> Result<Vec<String>> {
        let q = query(
            r#"
            MATCH (t:Task {id: $id})-[:MODIFIES]->(f:File)
            OPTIONAL MATCH (f)<-[:IMPORTS*1..3]-(dependent:File)
            RETURN f.path AS file, collect(DISTINCT dependent.path) AS dependents
            "#,
        )
        .param("id", task_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut impacted = Vec::new();

        while let Some(row) = result.next().await? {
            let file: String = row.get("file")?;
            impacted.push(file);
            let dependents: Vec<String> = row.get("dependents").unwrap_or_default();
            impacted.extend(dependents);
        }

        impacted.sort();
        impacted.dedup();
        Ok(impacted)
    }

    /// Find pending tasks in a plan that are blocked by uncompleted dependencies
    pub async fn find_blocked_tasks(
        &self,
        plan_id: Uuid,
    ) -> Result<Vec<(TaskNode, Vec<TaskNode>)>> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:HAS_TASK]->(t:Task {status: 'Pending'})
            MATCH (t)-[:DEPENDS_ON]->(blocker:Task)
            WHERE blocker.status <> 'Completed'
            RETURN t, collect(blocker) AS blockers
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut blocked = Vec::new();

        while let Some(row) = result.next().await? {
            let task_node: neo4rs::Node = row.get("t")?;
            let task = self.node_to_task(&task_node)?;

            let blocker_nodes: Vec<neo4rs::Node> = row.get("blockers").unwrap_or_default();
            let blockers: Vec<TaskNode> = blocker_nodes
                .iter()
                .filter_map(|n| self.node_to_task(n).ok())
                .collect();

            blocked.push((task, blockers));
        }

        Ok(blocked)
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

    /// Delete a task and all its related data (steps, decisions)
    pub async fn delete_task(&self, task_id: Uuid) -> Result<()> {
        // Delete all steps belonging to this task
        let q = query(
            r#"
            MATCH (t:Task {id: $id})-[:HAS_STEP]->(s:Step)
            DETACH DELETE s
            "#,
        )
        .param("id", task_id.to_string());
        self.graph.run(q).await?;

        // Delete all decisions belonging to this task
        let q = query(
            r#"
            MATCH (t:Task {id: $id})-[:INFORMED_BY]->(d:Decision)
            DETACH DELETE d
            "#,
        )
        .param("id", task_id.to_string());
        self.graph.run(q).await?;

        // Delete the task itself
        let q = query(
            r#"
            MATCH (t:Task {id: $id})
            DETACH DELETE t
            "#,
        )
        .param("id", task_id.to_string());
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

    /// Get a single step by ID
    pub async fn get_step(&self, step_id: Uuid) -> Result<Option<StepNode>> {
        let q = query(
            r#"
            MATCH (s:Step {id: $id})
            RETURN s
            "#,
        )
        .param("id", step_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            Ok(Some(StepNode {
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
            }))
        } else {
            Ok(None)
        }
    }

    /// Delete a step
    pub async fn delete_step(&self, step_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (s:Step {id: $id})
            DETACH DELETE s
            "#,
        )
        .param("id", step_id.to_string());

        self.graph.run(q).await?;
        Ok(())
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

    /// Get a single constraint by ID
    pub async fn get_constraint(&self, constraint_id: Uuid) -> Result<Option<ConstraintNode>> {
        let q = query(
            r#"
            MATCH (c:Constraint {id: $id})
            RETURN c
            "#,
        )
        .param("id", constraint_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            Ok(Some(ConstraintNode {
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
            }))
        } else {
            Ok(None)
        }
    }

    /// Update a constraint
    pub async fn update_constraint(
        &self,
        constraint_id: Uuid,
        description: Option<String>,
        constraint_type: Option<ConstraintType>,
        enforced_by: Option<String>,
    ) -> Result<()> {
        let mut set_clauses = vec![];
        if description.is_some() {
            set_clauses.push("c.description = $description");
        }
        if constraint_type.is_some() {
            set_clauses.push("c.constraint_type = $constraint_type");
        }
        if enforced_by.is_some() {
            set_clauses.push("c.enforced_by = $enforced_by");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (c:Constraint {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", constraint_id.to_string());
        if let Some(description) = description {
            q = q.param("description", description);
        }
        if let Some(constraint_type) = constraint_type {
            q = q.param("constraint_type", format!("{:?}", constraint_type));
        }
        if let Some(enforced_by) = enforced_by {
            q = q.param("enforced_by", enforced_by);
        }

        self.graph.run(q).await?;
        Ok(())
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

    /// Get a single decision by ID
    pub async fn get_decision(&self, decision_id: Uuid) -> Result<Option<DecisionNode>> {
        let q = query(
            r#"
            MATCH (d:Decision {id: $id})
            RETURN d
            "#,
        )
        .param("id", decision_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            Ok(Some(DecisionNode {
                id: node.get::<String>("id")?.parse()?,
                description: node.get("description")?,
                rationale: node.get("rationale")?,
                alternatives: node.get::<Vec<String>>("alternatives").unwrap_or_default(),
                chosen_option: node
                    .get::<String>("chosen_option")
                    .ok()
                    .filter(|s| !s.is_empty()),
                decided_by: node.get::<String>("decided_by").ok().unwrap_or_default(),
                decided_at: node
                    .get::<String>("decided_at")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(chrono::Utc::now),
            }))
        } else {
            Ok(None)
        }
    }

    /// Update a decision
    pub async fn update_decision(
        &self,
        decision_id: Uuid,
        description: Option<String>,
        rationale: Option<String>,
        chosen_option: Option<String>,
    ) -> Result<()> {
        let mut set_clauses = vec![];
        if description.is_some() {
            set_clauses.push("d.description = $description");
        }
        if rationale.is_some() {
            set_clauses.push("d.rationale = $rationale");
        }
        if chosen_option.is_some() {
            set_clauses.push("d.chosen_option = $chosen_option");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (d:Decision {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", decision_id.to_string());
        if let Some(description) = description {
            q = q.param("description", description);
        }
        if let Some(rationale) = rationale {
            q = q.param("rationale", rationale);
        }
        if let Some(chosen_option) = chosen_option {
            q = q.param("chosen_option", chosen_option);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a decision
    pub async fn delete_decision(&self, decision_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (d:Decision {id: $id})
            DETACH DELETE d
            "#,
        )
        .param("id", decision_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Impact analysis
    // ========================================================================

    /// Find all files that depend on a given file.
    /// When project_id is provided, only return dependents from the same project.
    pub async fn find_dependent_files(
        &self,
        file_path: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> Result<Vec<String>> {
        let q = match project_id {
            Some(pid) => query(&format!(
                r#"
                MATCH (f:File {{path: $path}})<-[:CONTAINS]-(p:Project {{id: $project_id}})
                MATCH (f)<-[:IMPORTS*1..{}]-(dependent:File)
                WHERE EXISTS {{ MATCH (dependent)<-[:CONTAINS]-(p) }}
                RETURN DISTINCT dependent.path AS path
                "#,
                depth
            ))
            .param("path", file_path)
            .param("project_id", pid.to_string()),
            None => query(&format!(
                r#"
                MATCH (f:File {{path: $path}})<-[:IMPORTS*1..{}]-(dependent:File)
                RETURN DISTINCT dependent.path AS path
                "#,
                depth
            ))
            .param("path", file_path),
        };

        let mut result = self.graph.execute(q).await?;
        let mut paths = Vec::new();

        while let Some(row) = result.next().await? {
            paths.push(row.get("path")?);
        }

        Ok(paths)
    }

    /// Find all functions that call a given function
    pub async fn find_callers(
        &self,
        function_id: &str,
        project_id: Option<Uuid>,
    ) -> Result<Vec<FunctionNode>> {
        let q = match project_id {
            Some(pid) => query(
                r#"
                MATCH (caller:Function)-[:CALLS]->(f:Function {id: $id})
                WHERE EXISTS {
                    MATCH (caller)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                }
                RETURN caller
                "#,
            )
            .param("id", function_id)
            .param("project_id", pid.to_string()),
            None => query(
                r#"
                MATCH (caller:Function)-[:CALLS]->(f:Function {id: $id})
                RETURN caller
                "#,
            )
            .param("id", function_id),
        };

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

    /// Delete a commit
    pub async fn delete_commit(&self, hash: &str) -> Result<()> {
        let q = query(
            r#"
            MATCH (c:Commit {hash: $hash})
            DETACH DELETE c
            "#,
        )
        .param("hash", hash);

        self.graph.run(q).await?;
        Ok(())
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
        .param(
            "description",
            release.description.clone().unwrap_or_default(),
        )
        .param("status", format!("{:?}", release.status))
        .param("project_id", release.project_id.to_string())
        .param(
            "target_date",
            release
                .target_date
                .map(|d| d.to_rfc3339())
                .unwrap_or_default(),
        )
        .param(
            "released_at",
            release
                .released_at
                .map(|d| d.to_rfc3339())
                .unwrap_or_default(),
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
            description: node
                .get::<String>("description")
                .ok()
                .filter(|s| !s.is_empty()),
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

        let cypher = format!(
            "MATCH (r:Release {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

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

    /// Remove a commit from a release
    pub async fn remove_commit_from_release(
        &self,
        release_id: Uuid,
        commit_hash: &str,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (r:Release {id: $release_id})-[rel:INCLUDES_COMMIT]->(c:Commit {hash: $hash})
            DELETE rel
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

    /// Delete a release
    pub async fn delete_release(&self, release_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (r:Release {id: $id})
            DETACH DELETE r
            "#,
        )
        .param("id", release_id.to_string());

        self.graph.run(q).await?;
        Ok(())
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
        .param(
            "description",
            milestone.description.clone().unwrap_or_default(),
        )
        .param(
            "status",
            serde_json::to_value(&milestone.status)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string(),
        )
        .param("project_id", milestone.project_id.to_string())
        .param(
            "target_date",
            milestone
                .target_date
                .map(|d| d.to_rfc3339())
                .unwrap_or_default(),
        )
        .param(
            "closed_at",
            milestone
                .closed_at
                .map(|d| d.to_rfc3339())
                .unwrap_or_default(),
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
            description: node
                .get::<String>("description")
                .ok()
                .filter(|s| !s.is_empty()),
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

        let cypher = format!(
            "MATCH (m:Milestone {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", id.to_string());

        if let Some(ref s) = status {
            q = q.param(
                "status",
                serde_json::to_value(s)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string(),
            );
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

    /// Link a plan to a project milestone
    pub async fn link_plan_to_milestone(&self, plan_id: Uuid, milestone_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})
            MATCH (m:Milestone {id: $milestone_id})
            MERGE (p)-[:TARGETS_MILESTONE]->(m)
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("milestone_id", milestone_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Unlink a plan from a project milestone
    pub async fn unlink_plan_from_milestone(
        &self,
        plan_id: Uuid,
        milestone_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[r:TARGETS_MILESTONE]->(m:Milestone {id: $milestone_id})
            DELETE r
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("milestone_id", milestone_id.to_string());

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

    /// Delete a milestone
    pub async fn delete_milestone(&self, milestone_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (m:Milestone {id: $id})
            DETACH DELETE m
            "#,
        )
        .param("id", milestone_id.to_string());

        self.graph.run(q).await?;
        Ok(())
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
    pub async fn get_project_progress(&self, project_id: Uuid) -> Result<(u32, u32, u32, u32)> {
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
            Ok((
                total as u32,
                completed as u32,
                in_progress as u32,
                pending as u32,
            ))
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
    #[allow(clippy::too_many_arguments)]
    pub async fn list_plans_filtered(
        &self,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
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

        let match_clause = if let Some(pid) = project_id {
            format!(
                "MATCH (proj:Project {{id: '{}'}})-[:HAS_PLAN]->(p:Plan)",
                pid
            )
        } else if let Some(ws) = workspace_slug {
            format!(
                "MATCH (w:Workspace {{slug: '{}'}})<-[:BELONGS_TO_WORKSPACE]-(proj:Project)-[:HAS_PLAN]->(p:Plan)",
                ws
            )
        } else {
            "MATCH (p:Plan)".to_string()
        };

        // Count query
        let count_cypher = format!("{} {} RETURN count(p) AS total", match_clause, where_clause);
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
            RETURN p
            ORDER BY {} {}
            SKIP {}
            LIMIT {}
            "#,
            match_clause, where_clause, order_field, order_dir, offset, limit
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
    #[allow(clippy::too_many_arguments)]
    pub async fn list_all_tasks_filtered(
        &self,
        plan_id: Option<Uuid>,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
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
        } else if let Some(pid) = project_id {
            format!(
                "MATCH (proj:Project {{id: '{}'}})-[:HAS_PLAN]->(p:Plan)-[:HAS_TASK]->(t:Task)",
                pid
            )
        } else if let Some(ws) = workspace_slug {
            format!(
                "MATCH (w:Workspace {{slug: '{}'}})<-[:BELONGS_TO_WORKSPACE]-(proj:Project)-[:HAS_PLAN]->(p:Plan)-[:HAS_TASK]->(t:Task)",
                ws
            )
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
        let count_cypher = format!("{} {} RETURN count(t) AS total", plan_match, where_clause);
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
            RETURN t, p.id AS plan_id, p.title AS plan_title,
                   COALESCE(p.status, '') AS plan_status
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
            let plan_status: String = row.get("plan_status").unwrap_or_default();
            tasks.push(TaskWithPlan {
                task: self.node_to_task(&node)?,
                plan_id: plan_id_str.parse()?,
                plan_title,
                plan_status: if plan_status.is_empty() {
                    None
                } else {
                    Some(pascal_to_snake_case(&plan_status))
                },
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
        let count_cypher = format!(
            "MATCH (p:Project) {} RETURN count(p) AS total",
            where_clause
        );
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
                energy: $energy,
                last_activated: datetime($last_activated),
                changes_json: $changes_json,
                assertion_rule_json: $assertion_rule_json
            })
            "#,
        )
        .param("id", note.id.to_string())
        .param(
            "project_id",
            note.project_id.map(|id| id.to_string()).unwrap_or_default(),
        )
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
        .param("energy", note.energy)
        .param(
            "last_activated",
            note.last_activated.unwrap_or(note.created_at).to_rfc3339(),
        )
        .param("changes_json", serde_json::to_string(&note.changes)?)
        .param(
            "assertion_rule_json",
            note.assertion_rule
                .as_ref()
                .map(|r| serde_json::to_string(r).unwrap_or_default())
                .unwrap_or_default(),
        );

        self.graph.run(q).await?;

        // Link to project (only if project_id is present)
        if let Some(pid) = note.project_id {
            let link_q = query(
                r#"
                MATCH (n:Note {id: $note_id})
                MATCH (p:Project {id: $project_id})
                MERGE (p)-[:HAS_NOTE]->(n)
                "#,
            )
            .param("note_id", note.id.to_string())
            .param("project_id", pid.to_string());

            self.graph.run(link_q).await?;
        }

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
        workspace_slug: Option<&str>,
        filters: &NoteFilters,
    ) -> Result<(Vec<Note>, usize)> {
        let mut where_conditions = Vec::new();

        if filters.global_only == Some(true) {
            where_conditions.push("(n.project_id IS NULL OR n.project_id = '')".to_string());
        } else if let Some(ref pid) = project_id {
            where_conditions.push(format!("n.project_id = '{}'", pid));
        } else if let Some(ws) = workspace_slug {
            where_conditions.push(format!(
                "n.project_id IN [(w:Workspace {{slug: '{}'}})<-[:BELONGS_TO_WORKSPACE]-(proj:Project) | proj.id]",
                ws
            ));
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
            EntityType::Workspace => "Workspace",
            EntityType::WorkspaceMilestone => "WorkspaceMilestone",
            EntityType::Resource => "Resource",
            EntityType::Component => "Component",
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
            EntityType::Workspace => "Workspace",
            EntityType::WorkspaceMilestone => "WorkspaceMilestone",
            EntityType::Resource => "Resource",
            EntityType::Component => "Component",
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
            EntityType::Workspace => "Workspace",
            EntityType::WorkspaceMilestone => "WorkspaceMilestone",
            EntityType::Resource => "Resource",
            EntityType::Component => "Component",
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
            EntityType::Workspace => "Workspace",
            EntityType::WorkspaceMilestone => "WorkspaceMilestone",
            EntityType::Resource => "Resource",
            EntityType::Component => "Component",
        };

        let (match_field, match_value) = match entity_type {
            EntityType::File => ("path", entity_id.to_string()),
            EntityType::Commit => ("hash", entity_id.to_string()),
            _ => ("id", entity_id.to_string()),
        };

        // Query for notes propagated through the graph.
        // The scoring formula integrates PageRank of intermediate nodes:
        //   avg_path_pagerank = mean of COALESCE(node.pagerank, 0.05) over path nodes
        //   score = (1/(distance+1)) * importance * (1 + avg_path_pagerank * 5)
        // This amplifies scores for notes propagated via structurally important hubs.
        let cypher = format!(
            r#"
            MATCH (target:{} {{{}: $entity_id}})
            MATCH path = (n:Note)-[:ATTACHED_TO]->(source)-[:CONTAINS|IMPORTS|CALLS*0..{}]->(target)
            WHERE n.status = 'active'
            WITH n, source, path, length(path) - 1 AS distance,
                 [node IN nodes(path) | coalesce(node.name, node.path, node.id)] AS path_names
            WITH n, source, distance, path_names,
                 CASE n.importance
                     WHEN 'critical' THEN 1.0
                     WHEN 'high' THEN 0.8
                     WHEN 'medium' THEN 0.5
                     ELSE 0.3
                 END AS importance_weight,
                 CASE WHEN size(nodes(path)) > 0
                      THEN reduce(s = 0.0, node IN nodes(path) | s + coalesce(node.pagerank, 0.05)) / size(nodes(path))
                      ELSE 0.05
                 END AS avg_path_pagerank
            WITH n, source, distance, path_names, avg_path_pagerank,
                 (1.0 / (distance + 1)) * importance_weight * (1.0 + avg_path_pagerank * 5.0) AS score
            WHERE score >= $min_score
            RETURN DISTINCT n, score, coalesce(source.name, source.path, source.id) AS source_entity,
                   path_names, distance, avg_path_pagerank
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
            let avg_path_pagerank: Option<f64> = row.get::<f64>("avg_path_pagerank").ok();

            propagated_notes.push(PropagatedNote {
                note,
                relevance_score: score,
                source_entity,
                propagation_path: path_names,
                distance: distance as u32,
                path_pagerank: avg_path_pagerank,
            });
        }

        Ok(propagated_notes)
    }

    /// Get workspace-level notes for a project (propagated from parent workspace)
    /// These are notes attached to the workspace that should propagate to all projects in it
    pub async fn get_workspace_notes_for_project(
        &self,
        project_id: Uuid,
        propagation_factor: f64,
    ) -> Result<Vec<PropagatedNote>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:BELONGS_TO_WORKSPACE]->(w:Workspace)
            MATCH (n:Note)-[:ATTACHED_TO]->(w)
            WHERE n.status IN ['active', 'needs_review']
            RETURN n, w.name AS workspace_name
            ORDER BY n.importance DESC, n.created_at DESC
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut workspace_notes = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            let workspace_name: String = row.get("workspace_name").unwrap_or_default();
            let note = self.node_to_note(&node)?;

            workspace_notes.push(PropagatedNote {
                note,
                relevance_score: propagation_factor,
                source_entity: format!("workspace:{}", workspace_name),
                propagation_path: vec![format!("workspace:{}", workspace_name)],
                distance: 1, // One hop: project -> workspace
                path_pagerank: None,
            });
        }

        Ok(workspace_notes)
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
                n.status = 'active',
                n.energy = CASE
                    WHEN coalesce(n.energy, 1.0) + 0.3 > 1.0 THEN 1.0
                    ELSE coalesce(n.energy, 1.0) + 0.3
                END,
                n.last_activated = datetime()
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

    /// Store a vector embedding on a Note node.
    ///
    /// Uses `db.create.setNodeVectorProperty` to ensure the correct type
    /// for the HNSW vector index. Also stores the model name for traceability.
    pub async fn set_note_embedding(
        &self,
        note_id: Uuid,
        embedding: &[f32],
        model: &str,
    ) -> Result<()> {
        // Convert f32 to f64 for neo4rs compatibility
        let embedding_f64: Vec<f64> = embedding.iter().map(|&x| x as f64).collect();

        let q = query(
            r#"
            MATCH (n:Note {id: $id})
            CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
            SET n.embedding_model = $model,
                n.embedded_at = datetime()
            "#,
        )
        .param("id", note_id.to_string())
        .param("embedding", embedding_f64)
        .param("model", model.to_string());

        self.graph
            .run(q)
            .await
            .context(format!("Failed to set embedding on note {}", note_id))?;

        Ok(())
    }

    /// Retrieve the stored embedding vector for a note.
    pub async fn get_note_embedding(&self, note_id: Uuid) -> Result<Option<Vec<f32>>> {
        let q = query(
            r#"
            MATCH (n:Note {id: $id})
            WHERE n.embedding IS NOT NULL
            RETURN n.embedding AS embedding
            "#,
        )
        .param("id", note_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let embedding_f64: Vec<f64> = row.get("embedding")?;
            let embedding_f32: Vec<f32> = embedding_f64.iter().map(|&x| x as f32).collect();
            Ok(Some(embedding_f32))
        } else {
            Ok(None)
        }
    }

    /// Search notes by vector similarity using the HNSW index.
    ///
    /// Returns notes ordered by descending cosine similarity score,
    /// filtered by optional project_id or workspace_slug for data isolation.
    ///
    /// Filtering priority: `project_id` > `workspace_slug` > global (no filter).
    pub async fn vector_search_notes(
        &self,
        embedding: &[f32],
        limit: usize,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
    ) -> Result<Vec<(Note, f64)>> {
        // Convert f32 to f64 for neo4rs
        let embedding_f64: Vec<f64> = embedding.iter().map(|&x| x as f64).collect();

        // We query more than limit to allow for post-filtering by project_id/workspace and status
        let has_filter = project_id.is_some() || workspace_slug.is_some();
        let query_limit = if has_filter { limit * 3 } else { limit * 2 };

        // Build Cypher with the appropriate project filter
        let (cypher, project_filter_value) = if let Some(pid) = project_id {
            // Direct project_id filter
            let cypher = r#"
                CALL db.index.vector.queryNodes('note_embeddings', $query_limit, $embedding)
                YIELD node AS n, score
                WHERE n.status IN ['active', 'needs_review']
                AND n.project_id = $project_id
                RETURN n, score
                ORDER BY score DESC
                LIMIT $limit
            "#;
            (cypher.to_string(), Some(pid.to_string()))
        } else if let Some(ws_slug) = workspace_slug {
            // Workspace filter: match notes belonging to any project in the workspace,
            // plus global notes (project_id IS NULL) for completeness
            let cypher = format!(
                r#"
                CALL db.index.vector.queryNodes('note_embeddings', $query_limit, $embedding)
                YIELD node AS n, score
                WHERE n.status IN ['active', 'needs_review']
                AND (
                    n.project_id IN [(w:Workspace {{slug: '{}'}})<-[:BELONGS_TO_WORKSPACE]-(proj:Project) | proj.id]
                    OR n.project_id IS NULL
                )
                RETURN n, score
                ORDER BY score DESC
                LIMIT $limit
                "#,
                ws_slug
            );
            (cypher, None)
        } else {
            // No filter — return all notes
            let cypher = r#"
                CALL db.index.vector.queryNodes('note_embeddings', $query_limit, $embedding)
                YIELD node AS n, score
                WHERE n.status IN ['active', 'needs_review']
                RETURN n, score
                ORDER BY score DESC
                LIMIT $limit
            "#;
            (cypher.to_string(), None)
        };

        let mut q = query(&cypher)
            .param("query_limit", query_limit as i64)
            .param("embedding", embedding_f64)
            .param("limit", limit as i64);

        if let Some(pid) = project_filter_value {
            q = q.param("project_id", pid);
        }

        let mut result = self.graph.execute(q).await?;
        let mut notes = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            let score: f64 = row.get("score")?;
            let note = self.node_to_note(&node)?;
            notes.push((note, score));
        }

        Ok(notes)
    }

    pub async fn list_notes_without_embedding(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Note>, usize)> {
        // Count total notes without embedding
        let count_cypher = r#"
            MATCH (n:Note)
            WHERE n.embedding IS NULL
            RETURN count(n) AS total
        "#;
        let mut count_result = self.graph.execute(query(count_cypher)).await?;
        let total: usize = if let Some(row) = count_result.next().await? {
            let count: i64 = row.get("total")?;
            count as usize
        } else {
            0
        };

        if total == 0 {
            return Ok((vec![], 0));
        }

        // Fetch notes without embedding
        let cypher = r#"
            MATCH (n:Note)
            WHERE n.embedding IS NULL
            RETURN n
            ORDER BY n.created_at ASC
            SKIP $offset
            LIMIT $limit
        "#;

        let q = query(cypher)
            .param("offset", offset as i64)
            .param("limit", limit as i64);

        let mut result = self.graph.execute(q).await?;
        let mut notes = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            let note = self.node_to_note(&node)?;
            notes.push(note);
        }

        Ok((notes, total))
    }

    // ========================================================================
    // Synapse operations (Phase 2 — Neural Network)
    // ========================================================================

    /// Create bidirectional SYNAPSE relationships between a note and its neighbors.
    ///
    /// Uses MERGE for idempotence. Creates edges in both directions with the same weight.
    /// Returns the number of synapses created (counting each direction separately).
    pub async fn create_synapses(&self, note_id: Uuid, neighbors: &[(Uuid, f64)]) -> Result<usize> {
        if neighbors.is_empty() {
            return Ok(0);
        }

        // Build UNWIND list directly in Cypher (internal computed data, no injection risk)
        let entries: Vec<String> = neighbors
            .iter()
            .map(|(nid, weight)| format!("{{id: '{}', weight: {}}}", nid, weight))
            .collect();

        let cypher = format!(
            r#"
            MATCH (source:Note {{id: $source_id}})
            UNWIND [{}] AS neighbor
            MATCH (target:Note {{id: neighbor.id}})
            MERGE (source)-[s1:SYNAPSE]->(target)
            ON CREATE SET s1.weight = neighbor.weight, s1.created_at = datetime()
            ON MATCH SET s1.weight = neighbor.weight
            MERGE (target)-[s2:SYNAPSE]->(source)
            ON CREATE SET s2.weight = neighbor.weight, s2.created_at = datetime()
            ON MATCH SET s2.weight = neighbor.weight
            RETURN count(s1) + count(s2) AS total
            "#,
            entries.join(", ")
        );

        let q = query(&cypher).param("source_id", note_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let total: i64 = row.get("total")?;
            Ok(total as usize)
        } else {
            Ok(0)
        }
    }

    /// Get all SYNAPSE relationships for a note (both directions).
    ///
    /// Returns (neighbor_id, weight) sorted by weight descending.
    pub async fn get_synapses(&self, note_id: Uuid) -> Result<Vec<(Uuid, f64)>> {
        let cypher = r#"
            MATCH (n:Note {id: $id})-[s:SYNAPSE]-(neighbor:Note)
            RETURN DISTINCT neighbor.id AS neighbor_id, s.weight AS weight
            ORDER BY weight DESC
        "#;

        let q = query(cypher).param("id", note_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut synapses = Vec::new();

        while let Some(row) = result.next().await? {
            let neighbor_id_str: String = row.get("neighbor_id")?;
            let weight: f64 = row.get("weight")?;
            if let Ok(nid) = neighbor_id_str.parse::<Uuid>() {
                synapses.push((nid, weight));
            }
        }

        Ok(synapses)
    }

    /// Delete all SYNAPSE relationships for a note (both directions).
    ///
    /// Returns the number of deleted relationships.
    pub async fn delete_synapses(&self, note_id: Uuid) -> Result<usize> {
        let cypher = r#"
            MATCH (n:Note {id: $id})-[s:SYNAPSE]-()
            DELETE s
            RETURN count(s) AS deleted
        "#;

        let q = query(cypher).param("id", note_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let deleted: i64 = row.get("deleted")?;
            Ok(deleted as usize)
        } else {
            Ok(0)
        }
    }

    // ========================================================================
    // Energy operations (Phase 2 — Neural Network)
    // ========================================================================

    /// Apply exponential energy decay to all active notes.
    ///
    /// Formula: `energy = energy × exp(-days_idle / half_life)`
    /// where `days_idle = (now - last_activated).days()`.
    ///
    /// **Temporally idempotent**: the result depends only on the absolute elapsed
    /// time since `last_activated`, NOT on how often this function is called.
    /// Calling it once after 30 days ≡ calling it daily for 30 days.
    ///
    /// Notes decaying below 0.05 are floored to 0.0 ("dead neuron").
    pub async fn update_energy_scores(&self, half_life_days: f64) -> Result<usize> {
        let q = query(
            r#"
            MATCH (n:Note)
            WHERE n.status = 'active'
              AND n.energy > 0.0
              AND n.last_activated IS NOT NULL
            WITH n,
                 duration.between(datetime(n.last_activated), datetime()).days AS days_idle
            WITH n,
                 n.energy * exp(-1.0 * toFloat(days_idle) / $half_life) AS new_energy
            WITH n,
                 CASE WHEN new_energy < 0.05 THEN 0.0 ELSE new_energy END AS clamped_energy
            WHERE abs(n.energy - clamped_energy) > 0.001
            SET n.energy = clamped_energy
            RETURN count(n) AS updated
            "#,
        )
        .param("half_life", half_life_days);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let updated: i64 = row.get("updated")?;
            Ok(updated as usize)
        } else {
            Ok(0)
        }
    }

    /// Boost a note's energy by `amount` (capped at 1.0) and reset `last_activated` to now.
    pub async fn boost_energy(&self, note_id: Uuid, amount: f64) -> Result<()> {
        let q = query(
            r#"
            MATCH (n:Note {id: $id})
            SET n.energy = CASE
                    WHEN n.energy + $amount > 1.0 THEN 1.0
                    ELSE n.energy + $amount
                END,
                n.last_activated = datetime()
            "#,
        )
        .param("id", note_id.to_string())
        .param("amount", amount);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Reinforce synapses between co-activated notes (Hebbian learning).
    /// For every pair (i, j) in note_ids, MERGE a bidirectional SYNAPSE.
    pub async fn reinforce_synapses(&self, note_ids: &[Uuid], boost: f64) -> Result<usize> {
        if note_ids.len() < 2 {
            return Ok(0);
        }

        // Generate all unique pairs
        let mut pairs = Vec::new();
        for i in 0..note_ids.len() {
            for j in (i + 1)..note_ids.len() {
                pairs.push((note_ids[i], note_ids[j]));
            }
        }

        let mut total = 0usize;
        for (a, b) in &pairs {
            // MERGE both directions for bidirectional synapse
            let q = query(
                r#"
                MATCH (a:Note {id: $a_id}), (b:Note {id: $b_id})
                MERGE (a)-[s1:SYNAPSE]->(b)
                  ON CREATE SET s1.weight = 0.5, s1.created_at = datetime()
                  ON MATCH SET s1.weight = CASE
                      WHEN s1.weight + $boost > 1.0 THEN 1.0
                      ELSE s1.weight + $boost
                  END
                MERGE (b)-[s2:SYNAPSE]->(a)
                  ON CREATE SET s2.weight = 0.5, s2.created_at = datetime()
                  ON MATCH SET s2.weight = CASE
                      WHEN s2.weight + $boost > 1.0 THEN 1.0
                      ELSE s2.weight + $boost
                  END
                RETURN count(s1) + count(s2) AS cnt
                "#,
            )
            .param("a_id", a.to_string())
            .param("b_id", b.to_string())
            .param("boost", boost);

            let mut result = self.graph.execute(q).await?;
            if let Some(row) = result.next().await? {
                total += row.get::<i64>("cnt").unwrap_or(0) as usize;
            }
        }

        Ok(total)
    }

    /// Decay all synapses and prune weak ones.
    pub async fn decay_synapses(
        &self,
        decay_amount: f64,
        prune_threshold: f64,
    ) -> Result<(usize, usize)> {
        // Step 1: Decay all synapses
        let decay_q = query(
            r#"
            MATCH ()-[s:SYNAPSE]->()
            SET s.weight = s.weight - $decay_amount
            RETURN count(s) AS decayed
            "#,
        )
        .param("decay_amount", decay_amount);

        let mut result = self.graph.execute(decay_q).await?;
        let decayed = if let Some(row) = result.next().await? {
            row.get::<i64>("decayed").unwrap_or(0) as usize
        } else {
            0
        };

        // Step 2: Prune weak synapses
        let prune_q = query(
            r#"
            MATCH ()-[s:SYNAPSE]->()
            WHERE s.weight < $threshold
            DELETE s
            RETURN count(s) AS pruned
            "#,
        )
        .param("threshold", prune_threshold);

        let mut result = self.graph.execute(prune_q).await?;
        let pruned = if let Some(row) = result.next().await? {
            row.get::<i64>("pruned").unwrap_or(0) as usize
        } else {
            0
        };

        Ok((decayed, pruned))
    }

    /// Initialize energy for notes that don't have it yet.
    pub async fn init_note_energy(&self) -> Result<usize> {
        let q = query(
            r#"
            MATCH (n:Note)
            WHERE n.energy IS NULL
            SET n.energy = 1.0,
                n.last_activated = coalesce(n.last_confirmed_at, n.created_at, datetime())
            RETURN count(n) AS updated
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let updated = if let Some(row) = result.next().await? {
            row.get::<i64>("updated").unwrap_or(0) as usize
        } else {
            0
        };

        Ok(updated)
    }

    /// List notes that have an embedding but no outgoing SYNAPSE.
    pub async fn list_notes_needing_synapses(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Note>, usize)> {
        // Count total
        let count_q = query(
            r#"
            MATCH (n:Note)
            WHERE n.embedding IS NOT NULL AND NOT (n)-[:SYNAPSE]->()
            RETURN count(n) AS total
            "#,
        );
        let mut result = self.graph.execute(count_q).await?;
        let total = if let Some(row) = result.next().await? {
            row.get::<i64>("total").unwrap_or(0) as usize
        } else {
            0
        };

        if total == 0 || limit == 0 {
            return Ok((vec![], total));
        }

        // Fetch batch
        let fetch_q = query(
            r#"
            MATCH (n:Note)
            WHERE n.embedding IS NOT NULL AND NOT (n)-[:SYNAPSE]->()
            RETURN n
            ORDER BY n.created_at
            SKIP $offset LIMIT $limit
            "#,
        )
        .param("offset", offset as i64)
        .param("limit", limit as i64);

        let mut result = self.graph.execute(fetch_q).await?;
        let mut notes = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            notes.push(self.node_to_note(&node)?);
        }

        Ok((notes, total))
    }

    // Helper function to convert Note scope to type string
    fn scope_type_string(&self, scope: &NoteScope) -> String {
        match scope {
            NoteScope::Workspace => "workspace".to_string(),
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
            NoteScope::Workspace | NoteScope::Project => String::new(),
            NoteScope::Module(path) => path.clone(),
            NoteScope::File(path) => path.clone(),
            NoteScope::Function(name) => name.clone(),
            NoteScope::Struct(name) => name.clone(),
            NoteScope::Trait(name) => name.clone(),
        }
    }

    // Helper function to convert Neo4j node to Note
    fn node_to_note(&self, node: &neo4rs::Node) -> Result<Note> {
        let scope_type: String = node
            .get("scope_type")
            .unwrap_or_else(|_| "project".to_string());
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
        let importance_str: String = node
            .get("importance")
            .unwrap_or_else(|_| "medium".to_string());

        let changes_json: String = node
            .get("changes_json")
            .unwrap_or_else(|_| "[]".to_string());
        let changes: Vec<NoteChange> = serde_json::from_str(&changes_json).unwrap_or_default();

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
            project_id: node.get::<String>("project_id").ok().and_then(|s| {
                if s.is_empty() {
                    None
                } else {
                    s.parse().ok()
                }
            }),
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
            energy: node.get("energy").unwrap_or(1.0),
            last_activated: node
                .get::<String>("last_activated")
                .ok()
                .and_then(|s| s.parse().ok()),
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
            (
                query(
                    r#"
                    MATCH (s:ChatSession {workspace_slug: $ws})
                    RETURN s ORDER BY s.updated_at DESC
                    SKIP $offset LIMIT $limit
                    "#,
                )
                .param("ws", ws.to_string())
                .param("offset", offset as i64)
                .param("limit", limit as i64),
                query("MATCH (s:ChatSession {workspace_slug: $ws}) RETURN count(s) AS total")
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

    // ========================================================================
    // User / Auth operations
    // ========================================================================

    /// Upsert a user: create if not exists, update if exists.
    ///
    /// For OIDC users: MERGE on (auth_provider + external_id).
    /// For password users: MERGE on (auth_provider + email).
    pub async fn upsert_user(&self, user: &UserNode) -> Result<UserNode> {
        use crate::neo4j::models::AuthProvider;

        let auth_provider_str = user.auth_provider.to_string();

        let q = match user.auth_provider {
            AuthProvider::Oidc => {
                let external_id = user
                    .external_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("OIDC user must have external_id"))?;
                query(
                    r#"
                    MERGE (u:User {auth_provider: $auth_provider, external_id: $external_id})
                    ON CREATE SET
                        u.id = $id,
                        u.email = $email,
                        u.name = $name,
                        u.picture_url = $picture_url,
                        u.password_hash = $password_hash,
                        u.created_at = datetime($created_at),
                        u.last_login_at = datetime($last_login_at)
                    ON MATCH SET
                        u.email = $email,
                        u.name = $name,
                        u.picture_url = $picture_url,
                        u.last_login_at = datetime($last_login_at)
                    RETURN u
                    "#,
                )
                .param("external_id", external_id.to_string())
            }
            AuthProvider::Password => query(
                r#"
                MERGE (u:User {auth_provider: $auth_provider, email: $email})
                ON CREATE SET
                    u.id = $id,
                    u.name = $name,
                    u.picture_url = $picture_url,
                    u.external_id = $external_id,
                    u.password_hash = $password_hash,
                    u.created_at = datetime($created_at),
                    u.last_login_at = datetime($last_login_at)
                ON MATCH SET
                    u.name = $name,
                    u.picture_url = $picture_url,
                    u.last_login_at = datetime($last_login_at)
                RETURN u
                "#,
            ),
        }
        .param("id", user.id.to_string())
        .param("auth_provider", auth_provider_str)
        .param("email", user.email.clone())
        .param("name", user.name.clone())
        .param("picture_url", user.picture_url.clone().unwrap_or_default())
        .param("external_id", user.external_id.clone().unwrap_or_default())
        .param(
            "password_hash",
            user.password_hash.clone().unwrap_or_default(),
        )
        .param("created_at", user.created_at.to_rfc3339())
        .param("last_login_at", user.last_login_at.to_rfc3339());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("u")?;
            self.node_to_user(&node)
        } else {
            anyhow::bail!("upsert_user: no row returned")
        }
    }

    /// Get a user by internal UUID
    pub async fn get_user_by_id(&self, id: Uuid) -> Result<Option<UserNode>> {
        let q = query("MATCH (u:User {id: $id}) RETURN u").param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("u")?;
            Ok(Some(self.node_to_user(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Get a user by provider and external ID (for OIDC lookups)
    pub async fn get_user_by_provider_id(
        &self,
        provider: &str,
        external_id: &str,
    ) -> Result<Option<UserNode>> {
        let q =
            query("MATCH (u:User {auth_provider: $provider, external_id: $external_id}) RETURN u")
                .param("provider", provider)
                .param("external_id", external_id);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("u")?;
            Ok(Some(self.node_to_user(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Get a user by email and auth provider
    pub async fn get_user_by_email_and_provider(
        &self,
        email: &str,
        provider: &str,
    ) -> Result<Option<UserNode>> {
        let q = query("MATCH (u:User {email: $email, auth_provider: $provider}) RETURN u")
            .param("email", email)
            .param("provider", provider);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("u")?;
            Ok(Some(self.node_to_user(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Get a user by email (any provider)
    pub async fn get_user_by_email(&self, email: &str) -> Result<Option<UserNode>> {
        let q = query("MATCH (u:User {email: $email}) RETURN u").param("email", email);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("u")?;
            Ok(Some(self.node_to_user(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Create a password-authenticated user
    pub async fn create_password_user(
        &self,
        email: &str,
        name: &str,
        password_hash: &str,
    ) -> Result<UserNode> {
        let now = chrono::Utc::now();
        let user = UserNode {
            id: Uuid::new_v4(),
            email: email.to_string(),
            name: name.to_string(),
            picture_url: None,
            auth_provider: crate::neo4j::models::AuthProvider::Password,
            external_id: None,
            password_hash: Some(password_hash.to_string()),
            created_at: now,
            last_login_at: now,
        };
        self.upsert_user(&user).await
    }

    /// List all users
    pub async fn list_users(&self) -> Result<Vec<UserNode>> {
        let q = query("MATCH (u:User) RETURN u ORDER BY u.created_at DESC");

        let mut result = self.graph.execute(q).await?;
        let mut users = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("u")?;
            users.push(self.node_to_user(&node)?);
        }
        Ok(users)
    }

    /// Parse a Neo4j Node into a UserNode
    fn node_to_user(&self, node: &neo4rs::Node) -> Result<UserNode> {
        use crate::neo4j::models::AuthProvider;

        // Parse auth_provider with backward compat: if not present, check for google_id
        let auth_provider = node
            .get::<String>("auth_provider")
            .ok()
            .and_then(|s| s.parse::<AuthProvider>().ok())
            .unwrap_or_else(|| {
                // Legacy: if google_id exists, treat as OIDC
                if node.get::<String>("google_id").is_ok() {
                    AuthProvider::Oidc
                } else {
                    AuthProvider::Password
                }
            });

        // external_id: try new field first, fall back to legacy google_id
        let external_id = node
            .get::<String>("external_id")
            .ok()
            .and_then(|s| if s.is_empty() { None } else { Some(s) })
            .or_else(|| {
                node.get::<String>("google_id").ok().and_then(|s| {
                    if s.is_empty() {
                        None
                    } else {
                        Some(s)
                    }
                })
            });

        let password_hash = node.get::<String>("password_hash").ok().and_then(|s| {
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        });

        Ok(UserNode {
            id: node.get::<String>("id")?.parse()?,
            email: node.get("email")?,
            name: node.get("name")?,
            picture_url: node.get::<String>("picture_url").ok().and_then(|s| {
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            }),
            auth_provider,
            external_id,
            password_hash,
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            last_login_at: node
                .get::<String>("last_login_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
        })
    }

    // ================================================================
    // Refresh Tokens
    // ================================================================

    /// Store a new refresh token (hashed) linked to a user.
    pub async fn create_refresh_token(
        &self,
        user_id: Uuid,
        token_hash: &str,
        expires_at: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        let q = query(
            "CREATE (rt:RefreshToken {
                token_hash: $token_hash,
                user_id: $user_id,
                expires_at: $expires_at,
                created_at: $created_at,
                revoked: false
            })",
        )
        .param("token_hash", token_hash.to_string())
        .param("user_id", user_id.to_string())
        .param("expires_at", expires_at.to_rfc3339())
        .param("created_at", chrono::Utc::now().to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Validate a refresh token by its hash. Returns the token if valid
    /// (not expired, not revoked).
    pub async fn validate_refresh_token(
        &self,
        token_hash: &str,
    ) -> Result<Option<crate::neo4j::models::RefreshTokenNode>> {
        let q = query(
            "MATCH (rt:RefreshToken {token_hash: $token_hash})
             RETURN rt",
        )
        .param("token_hash", token_hash.to_string());

        let mut result = self.graph.execute(q).await?;
        match result.next().await? {
            Some(row) => {
                let node: neo4rs::Node = row.get("rt")?;
                let token = crate::neo4j::models::RefreshTokenNode {
                    token_hash: node.get("token_hash")?,
                    user_id: node.get::<String>("user_id")?.parse()?,
                    expires_at: node
                        .get::<String>("expires_at")?
                        .parse()
                        .unwrap_or_else(|_| chrono::Utc::now()),
                    created_at: node
                        .get::<String>("created_at")?
                        .parse()
                        .unwrap_or_else(|_| chrono::Utc::now()),
                    revoked: node.get("revoked").unwrap_or(false),
                };

                // Check if expired or revoked
                if token.revoked || token.expires_at < chrono::Utc::now() {
                    Ok(None)
                } else {
                    Ok(Some(token))
                }
            }
            None => Ok(None),
        }
    }

    /// Revoke a single refresh token by its hash.
    pub async fn revoke_refresh_token(&self, token_hash: &str) -> Result<bool> {
        let q = query(
            "MATCH (rt:RefreshToken {token_hash: $token_hash})
             SET rt.revoked = true
             RETURN rt",
        )
        .param("token_hash", token_hash.to_string());

        let mut result = self.graph.execute(q).await?;
        Ok(result.next().await?.is_some())
    }

    /// Revoke all refresh tokens for a given user.
    pub async fn revoke_all_user_tokens(&self, user_id: Uuid) -> Result<u64> {
        let q = query(
            "MATCH (rt:RefreshToken {user_id: $user_id, revoked: false})
             SET rt.revoked = true
             RETURN count(rt) as count",
        )
        .param("user_id", user_id.to_string());

        let mut result = self.graph.execute(q).await?;
        match result.next().await? {
            Some(row) => {
                let count: i64 = row.get("count")?;
                Ok(count as u64)
            }
            None => Ok(0),
        }
    }

    // ================================================================
    // Feature Graphs
    // ================================================================

    pub async fn create_feature_graph(&self, fg: &FeatureGraphNode) -> Result<()> {
        let q = query(
            "CREATE (fg:FeatureGraph {
                id: $id,
                name: $name,
                description: $description,
                project_id: $project_id,
                created_at: $created_at,
                updated_at: $updated_at,
                entry_function: $entry_function,
                build_depth: $build_depth,
                include_relations: $include_relations
            })
            WITH fg
            MATCH (p:Project {id: $project_id})
            CREATE (fg)-[:BELONGS_TO]->(p)",
        )
        .param("id", fg.id.to_string())
        .param("name", fg.name.clone())
        .param("description", fg.description.clone().unwrap_or_default())
        .param("project_id", fg.project_id.to_string())
        .param("created_at", fg.created_at.to_rfc3339())
        .param("updated_at", fg.updated_at.to_rfc3339())
        .param(
            "entry_function",
            fg.entry_function.clone().unwrap_or_default(),
        )
        .param("build_depth", fg.build_depth.map(|d| d as i64).unwrap_or(0))
        .param(
            "include_relations",
            fg.include_relations
                .as_ref()
                .map(|r| serde_json::to_string(r).unwrap_or_default())
                .unwrap_or_default(),
        );

        self.execute_with_params(q).await?;
        Ok(())
    }

    pub async fn get_feature_graph(&self, id: Uuid) -> Result<Option<FeatureGraphNode>> {
        let q = query("MATCH (fg:FeatureGraph {id: $id}) RETURN fg").param("id", id.to_string());

        let rows = self.execute_with_params(q).await?;
        if let Some(row) = rows.first() {
            let node: neo4rs::Node = row.get("fg")?;
            Ok(Some(self.node_to_feature_graph(&node)?))
        } else {
            Ok(None)
        }
    }

    pub async fn get_feature_graph_detail(&self, id: Uuid) -> Result<Option<FeatureGraphDetail>> {
        let fg = self.get_feature_graph(id).await?;
        let Some(fg) = fg else { return Ok(None) };

        let q = query(
            "MATCH (fg:FeatureGraph {id: $id})-[:INCLUDES_ENTITY]->(e)
             RETURN labels(e)[0] AS entity_type,
                    COALESCE(e.path, e.id) AS entity_id,
                    COALESCE(e.name, e.path) AS name",
        )
        .param("id", id.to_string());

        let rows = self.execute_with_params(q).await?;
        let entities = rows
            .iter()
            .map(|row| FeatureGraphEntity {
                entity_type: row.get::<String>("entity_type").unwrap_or_default(),
                entity_id: row.get::<String>("entity_id").unwrap_or_default(),
                name: row.get::<String>("name").ok(),
                role: row.get::<String>("role").ok(),
            })
            .collect();

        Ok(Some(FeatureGraphDetail {
            graph: fg,
            entities,
        }))
    }

    pub async fn list_feature_graphs(
        &self,
        project_id: Option<Uuid>,
    ) -> Result<Vec<FeatureGraphNode>> {
        let cypher = if project_id.is_some() {
            "MATCH (fg:FeatureGraph {project_id: $pid})
             OPTIONAL MATCH (fg)-[:INCLUDES_ENTITY]->(e)
             RETURN fg, count(e) AS entity_count
             ORDER BY fg.updated_at DESC"
        } else {
            "MATCH (fg:FeatureGraph)
             OPTIONAL MATCH (fg)-[:INCLUDES_ENTITY]->(e)
             RETURN fg, count(e) AS entity_count
             ORDER BY fg.updated_at DESC"
        };

        let q = query(cypher).param(
            "pid",
            project_id.map(|id| id.to_string()).unwrap_or_default(),
        );

        let rows = self.execute_with_params(q).await?;
        let mut graphs = Vec::new();
        for row in &rows {
            let node: neo4rs::Node = row.get("fg")?;
            let mut fg = self.node_to_feature_graph(&node)?;
            fg.entity_count = row.get::<i64>("entity_count").ok();
            graphs.push(fg);
        }
        Ok(graphs)
    }

    pub async fn delete_feature_graph(&self, id: Uuid) -> Result<bool> {
        let q = query(
            "MATCH (fg:FeatureGraph {id: $id})
             DETACH DELETE fg
             RETURN count(fg) AS deleted",
        )
        .param("id", id.to_string());

        let rows = self.execute_with_params(q).await?;
        if let Some(row) = rows.first() {
            let deleted: i64 = row.get("deleted").unwrap_or(0);
            Ok(deleted > 0)
        } else {
            Ok(false)
        }
    }

    pub async fn add_entity_to_feature_graph(
        &self,
        feature_graph_id: Uuid,
        entity_type: &str,
        entity_id: &str,
        role: Option<&str>,
    ) -> Result<()> {
        let role_clause = if role.is_some() {
            " SET r.role = $role"
        } else {
            ""
        };

        let cypher = match entity_type.to_lowercase().as_str() {
            "file" => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:File {{path: $entity_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            "function" => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:Function {{name: $entity_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            "struct" => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:Struct {{name: $entity_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            "trait" => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:Trait {{name: $entity_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            "enum" => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:Enum {{name: $entity_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported entity type for feature graph: {}",
                    entity_type
                ));
            }
        };

        let mut q = query(&cypher)
            .param("fg_id", feature_graph_id.to_string())
            .param("entity_id", entity_id.to_string());
        if let Some(r) = role {
            q = q.param("role", r.to_string());
        }
        self.execute_with_params(q).await?;

        let update_q = query("MATCH (fg:FeatureGraph {id: $id}) SET fg.updated_at = $now")
            .param("id", feature_graph_id.to_string())
            .param("now", chrono::Utc::now().to_rfc3339());
        self.execute_with_params(update_q).await?;

        Ok(())
    }

    pub async fn remove_entity_from_feature_graph(
        &self,
        feature_graph_id: Uuid,
        entity_type: &str,
        entity_id: &str,
    ) -> Result<bool> {
        let match_prop = match entity_type.to_lowercase().as_str() {
            "file" => "path",
            _ => "name",
        };
        let label = match entity_type.to_lowercase().as_str() {
            "file" => "File",
            "function" => "Function",
            "struct" => "Struct",
            "trait" => "Trait",
            "enum" => "Enum",
            _ => return Err(anyhow::anyhow!("Unsupported entity type: {}", entity_type)),
        };

        let cypher = format!(
            "MATCH (fg:FeatureGraph {{id: $fg_id}})-[r:INCLUDES_ENTITY]->(e:{} {{{}: $entity_id}})
             DELETE r
             RETURN count(r) AS deleted",
            label, match_prop
        );

        let q = query(&cypher)
            .param("fg_id", feature_graph_id.to_string())
            .param("entity_id", entity_id.to_string());

        let rows = self.execute_with_params(q).await?;
        if let Some(row) = rows.first() {
            let deleted: i64 = row.get("deleted").unwrap_or(0);
            Ok(deleted > 0)
        } else {
            Ok(false)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn auto_build_feature_graph(
        &self,
        name: &str,
        description: Option<&str>,
        project_id: Uuid,
        entry_function: &str,
        depth: u32,
        include_relations: Option<&[String]>,
        filter_community: Option<bool>,
    ) -> Result<FeatureGraphDetail> {
        // Helper: check if a relation type should be included
        let should_include = |rel: &str| -> bool {
            match &include_relations {
                None => true, // default: include all
                Some(rels) => rels.iter().any(|r| r.eq_ignore_ascii_case(rel)),
            }
        };

        let filter_community = filter_community.unwrap_or(true);

        // Step 1: Collect all related functions and their files via call graph
        // We collect the entry function + callers + callees, plus community_id for filtering
        let depth = depth.clamp(1, 5);

        let q = query(&format!(
            r#"
            MATCH (entry:Function {{name: $name}})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {{id: $project_id}})
            OPTIONAL MATCH (entry)-[:CALLS*1..{depth}]->(callee:Function)
            WHERE EXISTS {{ MATCH (callee)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }}
            OPTIONAL MATCH (caller:Function)-[:CALLS*1..{depth}]->(entry)
            WHERE EXISTS {{ MATCH (caller)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }}
            WITH entry,
                 collect(DISTINCT callee) AS callees_nodes,
                 collect(DISTINCT caller) AS callers_nodes
            WITH entry, callees_nodes, callers_nodes,
                 [entry] + callees_nodes + callers_nodes AS all_funcs
            UNWIND all_funcs AS f
            WITH DISTINCT f
            WHERE f IS NOT NULL
            RETURN f.name AS func_name, f.file_path AS file_path,
                   f.community_id AS community_id
            "#,
            depth = depth
        ))
        .param("name", entry_function)
        .param("project_id", project_id.to_string());

        let rows = self.execute_with_params(q).await?;

        // Collect unique functions with community info
        let mut functions: Vec<(String, Option<String>, Option<i64>)> = Vec::new();
        let mut files: std::collections::HashSet<String> = std::collections::HashSet::new();

        for row in &rows {
            if let Ok(func_name) = row.get::<String>("func_name") {
                let file_path: Option<String> = row.get::<String>("file_path").ok();
                let community_id: Option<i64> = row.get::<i64>("community_id").ok();
                if let Some(ref fp) = file_path {
                    if !fp.is_empty() {
                        files.insert(fp.clone());
                    }
                }
                functions.push((func_name, file_path, community_id));
            }
        }

        if functions.is_empty() {
            return Err(anyhow::anyhow!(
                "No function found matching '{}'",
                entry_function
            ));
        }

        // Step 1a: Community-based filtering (post-traversal)
        // Keep same community OR direct (depth=1) connection OR no community data
        if filter_community {
            let entry_community = functions
                .iter()
                .find(|(name, _, _)| name == entry_function)
                .and_then(|(_, _, cid)| *cid);

            if let Some(entry_cid) = entry_community {
                // Get direct (depth=1) callees and callers for the entry function
                let direct_q = query(
                    r#"
                    MATCH (entry:Function {name: $name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                    OPTIONAL MATCH (entry)-[:CALLS]->(dc:Function)
                    WHERE EXISTS { MATCH (dc)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }
                    OPTIONAL MATCH (caller:Function)-[:CALLS]->(entry)
                    WHERE EXISTS { MATCH (caller)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }
                    WITH collect(DISTINCT dc.name) AS direct_callees,
                         collect(DISTINCT caller.name) AS direct_callers
                    RETURN direct_callees + direct_callers AS direct_names
                    "#,
                )
                .param("name", entry_function)
                .param("project_id", project_id.to_string());

                let direct_rows = self.execute_with_params(direct_q).await?;
                let mut direct_set: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                if let Some(row) = direct_rows.first() {
                    if let Ok(names) = row.get::<Vec<String>>("direct_names") {
                        for n in names {
                            if !n.is_empty() {
                                direct_set.insert(n);
                            }
                        }
                    }
                }

                // Filter: keep entry + same community + direct connections + unknown community
                functions.retain(|(fname, _, cid)| {
                    fname == entry_function
                        || direct_set.contains(fname)
                        || cid.is_none()
                        || *cid == Some(entry_cid)
                });

                // Rebuild files set from remaining functions
                files.clear();
                for (_, fp, _) in &functions {
                    if let Some(ref fp) = fp {
                        if !fp.is_empty() {
                            files.insert(fp.clone());
                        }
                    }
                }
            }
            // If entry has no community_id → no filtering (backward compat)
        }

        // Step 1b: Expand via IMPLEMENTS_TRAIT + IMPLEMENTS_FOR for structs/enums in collected files
        let mut structs: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut traits: std::collections::HashSet<String> = std::collections::HashSet::new();

        if !files.is_empty()
            && (should_include("implements_trait") || should_include("implements_for"))
        {
            let file_list: Vec<String> = files.iter().cloned().collect();
            let types_q = query(
                r#"
                MATCH (f:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                WHERE f.path IN $files
                OPTIONAL MATCH (f)-[:CONTAINS]->(s:Struct)
                OPTIONAL MATCH (f)-[:CONTAINS]->(e:Enum)
                WITH collect(DISTINCT s) + collect(DISTINCT e) AS types
                UNWIND types AS t
                WITH DISTINCT t WHERE t IS NOT NULL
                OPTIONAL MATCH (impl:Impl)-[:IMPLEMENTS_FOR]->(t)
                OPTIONAL MATCH (impl)-[:IMPLEMENTS_TRAIT]->(tr:Trait)
                RETURN t.name AS type_name, labels(t)[0] AS type_label,
                       collect(DISTINCT tr.name) AS trait_names
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("files", file_list);

            let type_rows = self.execute_with_params(types_q).await?;
            for row in &type_rows {
                if let Ok(type_name) = row.get::<String>("type_name") {
                    structs.insert(type_name);
                }
                if let Ok(trait_list) = row.get::<Vec<String>>("trait_names") {
                    for t in trait_list {
                        if !t.is_empty() {
                            traits.insert(t);
                        }
                    }
                }
            }
        }

        // Step 1c: Expand via IMPORTS — include files imported by the feature's files
        if !files.is_empty() && should_include("imports") {
            let file_list: Vec<String> = files.iter().cloned().collect();
            let imports_q = query(
                r#"
                MATCH (f:File)-[:IMPORTS]->(imported:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                WHERE f.path IN $files
                RETURN DISTINCT imported.path AS imported_path
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("files", file_list);

            let import_rows = self.execute_with_params(imports_q).await?;
            for row in &import_rows {
                if let Ok(imported_path) = row.get::<String>("imported_path") {
                    if !imported_path.is_empty() {
                        files.insert(imported_path);
                    }
                }
            }
        }

        // Step 2: Create the FeatureGraph (with build params for refresh)
        let fg = FeatureGraphNode {
            id: Uuid::new_v4(),
            name: name.to_string(),
            description: description.map(|d| d.to_string()),
            project_id,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            entity_count: None,
            entry_function: Some(entry_function.to_string()),
            build_depth: Some(depth),
            include_relations: include_relations.map(|r| r.iter().map(|s| s.to_string()).collect()),
        };
        self.create_feature_graph(&fg).await?;

        // Step 3: Add all entities with auto-assigned roles
        let mut entities = Vec::new();

        // Add functions with role: entry_point for the entry function, core_logic for others
        for (func_name, _file_path, _) in &functions {
            let role = if func_name == entry_function {
                "entry_point"
            } else {
                "core_logic"
            };
            let _ = self
                .add_entity_to_feature_graph(fg.id, "function", func_name, Some(role))
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "function".to_string(),
                entity_id: func_name.clone(),
                name: Some(func_name.clone()),
                role: Some(role.to_string()),
            });
        }

        // Add files with role: support
        for file_path in &files {
            let _ = self
                .add_entity_to_feature_graph(fg.id, "file", file_path, Some("support"))
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "file".to_string(),
                entity_id: file_path.clone(),
                name: Some(file_path.clone()),
                role: Some("support".to_string()),
            });
        }

        // Add structs/enums discovered via IMPLEMENTS_FOR with role: data_model
        for struct_name in &structs {
            let _ = self
                .add_entity_to_feature_graph(fg.id, "struct", struct_name, Some("data_model"))
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "struct".to_string(),
                entity_id: struct_name.clone(),
                name: Some(struct_name.clone()),
                role: Some("data_model".to_string()),
            });
        }

        // Add traits discovered via IMPLEMENTS_TRAIT with role: trait_contract
        for trait_name in &traits {
            let _ = self
                .add_entity_to_feature_graph(fg.id, "trait", trait_name, Some("trait_contract"))
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "trait".to_string(),
                entity_id: trait_name.clone(),
                name: Some(trait_name.clone()),
                role: Some("trait_contract".to_string()),
            });
        }

        Ok(FeatureGraphDetail {
            graph: fg,
            entities,
        })
    }

    /// Refresh an auto-built feature graph by re-running BFS with stored params.
    /// Returns None if the graph was manually created (no entry_function).
    pub async fn refresh_feature_graph(&self, id: Uuid) -> Result<Option<FeatureGraphDetail>> {
        // 1. Load the existing feature graph
        let detail = self
            .get_feature_graph_detail(id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Feature graph {} not found", id))?;

        let fg = &detail.graph;

        // 2. Check if it was auto-built (has entry_function)
        let entry_function = match &fg.entry_function {
            Some(ef) => ef.clone(),
            None => return Ok(None), // manually created, skip refresh
        };

        let depth = fg.build_depth.unwrap_or(2);
        let include_relations = fg.include_relations.clone();
        let project_id = fg.project_id;

        // 3. Delete all existing INCLUDES_ENTITY relationships
        let delete_q = query(
            "MATCH (fg:FeatureGraph {id: $fg_id})-[r:INCLUDES_ENTITY]->()
             DELETE r
             RETURN count(r) AS deleted",
        )
        .param("fg_id", id.to_string());
        self.execute_with_params(delete_q).await?;

        // 4. Re-run the BFS traversal (same logic as auto_build_feature_graph)
        let should_include = |rel: &str| -> bool {
            match &include_relations {
                None => true,
                Some(rels) => rels.iter().any(|r| r.eq_ignore_ascii_case(rel)),
            }
        };

        let depth = depth.clamp(1, 5);

        // Step 4a: Collect functions via call graph
        let q = query(&format!(
            r#"
            MATCH (entry:Function {{name: $name}})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {{id: $project_id}})
            OPTIONAL MATCH (entry)-[:CALLS*1..{depth}]->(callee:Function)
            WHERE EXISTS {{ MATCH (callee)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }}
            OPTIONAL MATCH (caller:Function)-[:CALLS*1..{depth}]->(entry)
            WHERE EXISTS {{ MATCH (caller)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }}
            WITH entry,
                 collect(DISTINCT callee) AS callees_nodes,
                 collect(DISTINCT caller) AS callers_nodes
            WITH entry, callees_nodes, callers_nodes,
                 [entry] + callees_nodes + callers_nodes AS all_funcs
            UNWIND all_funcs AS f
            WITH DISTINCT f
            WHERE f IS NOT NULL
            RETURN f.name AS func_name, f.file_path AS file_path
            "#,
            depth = depth
        ))
        .param("name", entry_function.as_str())
        .param("project_id", project_id.to_string());

        let rows = self.execute_with_params(q).await?;

        let mut functions: Vec<(String, Option<String>)> = Vec::new();
        let mut files: std::collections::HashSet<String> = std::collections::HashSet::new();

        for row in &rows {
            if let Ok(func_name) = row.get::<String>("func_name") {
                let file_path: Option<String> = row.get::<String>("file_path").ok();
                if let Some(ref fp) = file_path {
                    if !fp.is_empty() {
                        files.insert(fp.clone());
                    }
                }
                functions.push((func_name, file_path));
            }
        }

        if functions.is_empty() {
            return Err(anyhow::anyhow!(
                "No function found matching '{}' during refresh",
                entry_function
            ));
        }

        // Step 4b: Expand via IMPLEMENTS_TRAIT + IMPLEMENTS_FOR
        let mut structs: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut traits: std::collections::HashSet<String> = std::collections::HashSet::new();

        if !files.is_empty()
            && (should_include("implements_trait") || should_include("implements_for"))
        {
            let file_list: Vec<String> = files.iter().cloned().collect();
            let types_q = query(
                r#"
                MATCH (f:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                WHERE f.path IN $files
                OPTIONAL MATCH (f)-[:CONTAINS]->(s:Struct)
                OPTIONAL MATCH (f)-[:CONTAINS]->(e:Enum)
                WITH collect(DISTINCT s) + collect(DISTINCT e) AS types
                UNWIND types AS t
                WITH DISTINCT t WHERE t IS NOT NULL
                OPTIONAL MATCH (impl:Impl)-[:IMPLEMENTS_FOR]->(t)
                OPTIONAL MATCH (impl)-[:IMPLEMENTS_TRAIT]->(tr:Trait)
                RETURN t.name AS type_name, labels(t)[0] AS type_label,
                       collect(DISTINCT tr.name) AS trait_names
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("files", file_list);

            let type_rows = self.execute_with_params(types_q).await?;
            for row in &type_rows {
                if let Ok(type_name) = row.get::<String>("type_name") {
                    structs.insert(type_name);
                }
                if let Ok(trait_list) = row.get::<Vec<String>>("trait_names") {
                    for t in trait_list {
                        if !t.is_empty() {
                            traits.insert(t);
                        }
                    }
                }
            }
        }

        // Step 4c: Expand via IMPORTS
        if !files.is_empty() && should_include("imports") {
            let file_list: Vec<String> = files.iter().cloned().collect();
            let imports_q = query(
                r#"
                MATCH (f:File)-[:IMPORTS]->(imported:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                WHERE f.path IN $files
                RETURN DISTINCT imported.path AS imported_path
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("files", file_list);

            let import_rows = self.execute_with_params(imports_q).await?;
            for row in &import_rows {
                if let Ok(imported_path) = row.get::<String>("imported_path") {
                    if !imported_path.is_empty() {
                        files.insert(imported_path);
                    }
                }
            }
        }

        // 5. Update the updated_at timestamp
        let update_q = query(
            "MATCH (fg:FeatureGraph {id: $fg_id})
             SET fg.updated_at = $now
             RETURN fg",
        )
        .param("fg_id", id.to_string())
        .param("now", chrono::Utc::now().to_rfc3339());
        self.execute_with_params(update_q).await?;

        // 6. Re-add all entities with auto-assigned roles
        let mut entities = Vec::new();

        for (func_name, _file_path) in &functions {
            let role = if *func_name == entry_function {
                "entry_point"
            } else {
                "core_logic"
            };
            let _ = self
                .add_entity_to_feature_graph(id, "function", func_name, Some(role))
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "function".to_string(),
                entity_id: func_name.clone(),
                name: Some(func_name.clone()),
                role: Some(role.to_string()),
            });
        }

        for file_path in &files {
            let _ = self
                .add_entity_to_feature_graph(id, "file", file_path, Some("support"))
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "file".to_string(),
                entity_id: file_path.clone(),
                name: Some(file_path.clone()),
                role: Some("support".to_string()),
            });
        }

        for struct_name in &structs {
            let _ = self
                .add_entity_to_feature_graph(id, "struct", struct_name, Some("data_model"))
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "struct".to_string(),
                entity_id: struct_name.clone(),
                name: Some(struct_name.clone()),
                role: Some("data_model".to_string()),
            });
        }

        for trait_name in &traits {
            let _ = self
                .add_entity_to_feature_graph(id, "trait", trait_name, Some("trait_contract"))
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "trait".to_string(),
                entity_id: trait_name.clone(),
                name: Some(trait_name.clone()),
                role: Some("trait_contract".to_string()),
            });
        }

        // Re-read the graph node to get updated state
        let updated_graph = self
            .get_feature_graph_detail(id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Feature graph {} disappeared after refresh", id))?;

        Ok(Some(updated_graph))
    }

    /// Get the top N most connected functions for a project, ranked by
    /// (callers + callees). Used for auto-generating feature graphs after sync.
    pub async fn get_top_entry_functions(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<String>> {
        let q = query(
            r#"
            MATCH (f:Function)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
            OPTIONAL MATCH (f)-[:CALLS]->(callee:Function)
            OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
            WITH f, count(DISTINCT callee) AS callees, count(DISTINCT caller) AS callers
            WITH f, callers + callees AS connections
            WHERE connections > 0
            RETURN f.name AS name
            ORDER BY connections DESC
            LIMIT $limit
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("limit", limit as i64);

        let mut result = self.graph.execute(q).await?;
        let mut functions = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(name) = row.get::<String>("name") {
                functions.push(name);
            }
        }

        Ok(functions)
    }

    // ========================================================================
    // Bulk graph extraction (for graph analytics)
    // ========================================================================

    /// Get all IMPORTS edges between files in a project as (source_path, target_path) pairs.
    pub async fn get_project_import_edges(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String)>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f1:File)-[:IMPORTS]->(f2:File)<-[:CONTAINS]-(p)
            RETURN f1.path AS source, f2.path AS target
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut edges = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(source), Ok(target)) =
                (row.get::<String>("source"), row.get::<String>("target"))
            {
                edges.push((source, target));
            }
        }

        Ok(edges)
    }

    /// Get all CALLS edges between functions in a project as (caller_id, callee_id) pairs.
    /// Scoped to the same project (no cross-project calls).
    pub async fn get_project_call_edges(&self, project_id: Uuid) -> Result<Vec<(String, String)>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(:File)-[:CONTAINS]->(f1:Function)-[:CALLS]->(f2:Function)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p)
            RETURN f1.name AS source, f2.name AS target
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut edges = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(source), Ok(target)) =
                (row.get::<String>("source"), row.get::<String>("target"))
            {
                edges.push((source, target));
            }
        }

        Ok(edges)
    }

    /// Batch-update analytics scores on File nodes via UNWIND.
    pub async fn batch_update_file_analytics(
        &self,
        updates: &[crate::graph::models::FileAnalyticsUpdate],
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        // Build UNWIND list directly in Cypher (internal computed data, no injection risk)
        let entries: Vec<String> = updates
            .iter()
            .map(|u| {
                format!(
                    "{{path: '{}', pagerank: {}, betweenness: {}, community_id: {}, community_label: '{}', clustering_coefficient: {}, component_id: {}}}",
                    u.path.replace('\'', "\\'"),
                    u.pagerank,
                    u.betweenness,
                    u.community_id,
                    u.community_label.replace('\'', "\\'"),
                    u.clustering_coefficient,
                    u.component_id
                )
            })
            .collect();

        let cypher = format!(
            r#"
            UNWIND [{}] AS u
            MATCH (f:File {{path: u.path}})
            SET f.pagerank = u.pagerank,
                f.betweenness = u.betweenness,
                f.community_id = u.community_id,
                f.community_label = u.community_label,
                f.clustering_coefficient = u.clustering_coefficient,
                f.component_id = u.component_id,
                f.analytics_updated_at = datetime()
            "#,
            entries.join(", ")
        );

        self.execute(&cypher).await?;
        Ok(())
    }

    /// Batch-update analytics scores on Function nodes via UNWIND.
    pub async fn batch_update_function_analytics(
        &self,
        updates: &[crate::graph::models::FunctionAnalyticsUpdate],
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        let entries: Vec<String> = updates
            .iter()
            .map(|u| {
                format!(
                    "{{name: '{}', pagerank: {}, betweenness: {}, community_id: {}, clustering_coefficient: {}, component_id: {}}}",
                    u.name.replace('\'', "\\'"),
                    u.pagerank,
                    u.betweenness,
                    u.community_id,
                    u.clustering_coefficient,
                    u.component_id
                )
            })
            .collect();

        let cypher = format!(
            r#"
            UNWIND [{}] AS u
            MATCH (f:Function {{name: u.name}})
            SET f.pagerank = u.pagerank,
                f.betweenness = u.betweenness,
                f.community_id = u.community_id,
                f.clustering_coefficient = u.clustering_coefficient,
                f.component_id = u.component_id,
                f.analytics_updated_at = datetime()
            "#,
            entries.join(", ")
        );

        self.execute(&cypher).await?;
        Ok(())
    }

    fn node_to_feature_graph(&self, node: &neo4rs::Node) -> Result<FeatureGraphNode> {
        Ok(FeatureGraphNode {
            id: node.get::<String>("id")?.parse()?,
            name: node.get("name")?,
            description: node.get::<String>("description").ok().and_then(|s| {
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            }),
            project_id: node.get::<String>("project_id")?.parse()?,
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            updated_at: node
                .get::<String>("updated_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            entity_count: None,
            entry_function: node.get::<String>("entry_function").ok(),
            build_depth: node.get::<i64>("build_depth").ok().map(|d| d as u32),
            include_relations: node
                .get::<String>("include_relations")
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok()),
        })
    }
}
