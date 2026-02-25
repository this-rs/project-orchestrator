//! Neo4j client for interacting with the knowledge graph

use anyhow::{Context, Result};
use neo4rs::{query, Graph, Query};
use std::sync::Arc;

/// Client for Neo4j operations
pub struct Neo4jClient {
    pub(crate) graph: Arc<Graph>,
}

/// Convert PascalCase to snake_case (e.g., "InProgress" -> "in_progress")
pub(crate) fn pascal_to_snake_case(s: &str) -> String {
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
pub(crate) fn snake_to_pascal_case(s: &str) -> String {
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
            "CREATE CONSTRAINT enum_id IF NOT EXISTS FOR (e:Enum) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT impl_id IF NOT EXISTS FOR (i:Impl) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT import_id IF NOT EXISTS FOR (i:Import) REQUIRE i.id IS UNIQUE",
            // Commit constraint
            "CREATE CONSTRAINT commit_hash IF NOT EXISTS FOR (c:Commit) REQUIRE c.hash IS UNIQUE",
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
            // Chat constraints
            "CREATE CONSTRAINT chat_session_id IF NOT EXISTS FOR (s:ChatSession) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT chat_event_id IF NOT EXISTS FOR (e:ChatEvent) REQUIRE e.id IS UNIQUE",
            // Milestone & Release constraints
            "CREATE CONSTRAINT milestone_id IF NOT EXISTS FOR (m:Milestone) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT release_id IF NOT EXISTS FOR (r:Release) REQUIRE r.id IS UNIQUE",
            // FeatureGraph constraint
            "CREATE CONSTRAINT feature_graph_id IF NOT EXISTS FOR (fg:FeatureGraph) REQUIRE fg.id IS UNIQUE",
            // User constraint
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            // RefreshToken constraint
            "CREATE CONSTRAINT refresh_token_hash IF NOT EXISTS FOR (rt:RefreshToken) REQUIRE rt.token_hash IS UNIQUE",
        ];

        let indexes = vec![
            "CREATE INDEX project_name IF NOT EXISTS FOR (p:Project) ON (p.name)",
            "CREATE INDEX file_language IF NOT EXISTS FOR (f:File) ON (f.language)",
            "CREATE INDEX file_project IF NOT EXISTS FOR (f:File) ON (f.project_id)",
            "CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name)",
            // Function.file_path — enables index seek in batch_create_call_relationships Phase 1
            "CREATE INDEX function_file_path IF NOT EXISTS FOR (f:Function) ON (f.file_path)",
            // Composite (name, file_path) — enables direct seek for same-file call resolution
            "CREATE INDEX function_name_file_path IF NOT EXISTS FOR (f:Function) ON (f.name, f.file_path)",
            "CREATE INDEX struct_name IF NOT EXISTS FOR (s:Struct) ON (s.name)",
            "CREATE INDEX trait_name IF NOT EXISTS FOR (t:Trait) ON (t.name)",
            "CREATE INDEX enum_name IF NOT EXISTS FOR (e:Enum) ON (e.name)",
            // Denormalized project_id on symbols — enables direct seek without graph traversal
            "CREATE INDEX function_project_id IF NOT EXISTS FOR (f:Function) ON (f.project_id)",
            "CREATE INDEX struct_project_id IF NOT EXISTS FOR (s:Struct) ON (s.project_id)",
            "CREATE INDEX trait_project_id IF NOT EXISTS FOR (t:Trait) ON (t.project_id)",
            "CREATE INDEX enum_project_id IF NOT EXISTS FOR (e:Enum) ON (e.project_id)",
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
            // ChatEvent indexes — critical for performance (120K+ nodes)
            "CREATE INDEX chat_event_session IF NOT EXISTS FOR (e:ChatEvent) ON (e.session_id)",
            "CREATE INDEX chat_event_type IF NOT EXISTS FOR (e:ChatEvent) ON (e.event_type)",
            "CREATE INDEX chat_event_seq IF NOT EXISTS FOR (e:ChatEvent) ON (e.seq)",
            // ChatSession indexes — queried by project_slug, workspace_slug, cli_session_id
            "CREATE INDEX chat_session_project IF NOT EXISTS FOR (s:ChatSession) ON (s.project_slug)",
            "CREATE INDEX chat_session_workspace IF NOT EXISTS FOR (s:ChatSession) ON (s.workspace_slug)",
            "CREATE INDEX chat_session_cli IF NOT EXISTS FOR (s:ChatSession) ON (s.cli_session_id)",
            // FeatureGraph indexes
            "CREATE INDEX feature_graph_project IF NOT EXISTS FOR (fg:FeatureGraph) ON (fg.project_id)",
            // User indexes — queried by email, external_id, auth_provider
            "CREATE INDEX user_email IF NOT EXISTS FOR (u:User) ON (u.email)",
            "CREATE INDEX user_external IF NOT EXISTS FOR (u:User) ON (u.external_id)",
            "CREATE INDEX user_auth_provider IF NOT EXISTS FOR (u:User) ON (u.auth_provider)",
            // RefreshToken indexes — queried by user_id
            "CREATE INDEX refresh_token_user IF NOT EXISTS FOR (rt:RefreshToken) ON (rt.user_id)",
            // Milestone indexes
            "CREATE INDEX milestone_project IF NOT EXISTS FOR (m:Milestone) ON (m.project_id)",
            // Release indexes
            "CREATE INDEX release_project IF NOT EXISTS FOR (r:Release) ON (r.project_id)",
            "CREATE INDEX release_version IF NOT EXISTS FOR (r:Release) ON (r.version)",
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
            // HNSW vector index for cosine similarity search on File embeddings
            r#"CREATE VECTOR INDEX file_embedding_index IF NOT EXISTS
               FOR (f:File) ON (f.embedding)
               OPTIONS {indexConfig: {
                   `vector.dimensions`: 768,
                   `vector.similarity_function`: 'cosine'
               }}"#,
            // HNSW vector index for cosine similarity search on Function embeddings
            r#"CREATE VECTOR INDEX function_embedding_index IF NOT EXISTS
               FOR (fn:Function) ON (fn.embedding)
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

        // Verify all constraints were created successfully
        let expected_constraints = vec![
            "project_id",
            "project_slug",
            "file_path",
            "function_id",
            "struct_id",
            "trait_id",
            "enum_id",
            "impl_id",
            "import_id",
            "commit_hash",
            "plan_id",
            "task_id",
            "step_id",
            "decision_id",
            "constraint_id",
            "agent_id",
            "note_id",
            "workspace_id",
            "workspace_slug",
            "workspace_milestone_id",
            "resource_id",
            "component_id",
            "chat_session_id",
            "chat_event_id",
            "milestone_id",
            "release_id",
            "feature_graph_id",
            "user_id",
            "refresh_token_hash",
        ];
        match self
            .graph
            .execute(query(
                "SHOW CONSTRAINTS YIELD name RETURN collect(name) AS names",
            ))
            .await
        {
            Ok(mut result) => {
                if let Ok(Some(row)) = result.next().await {
                    let names: Vec<String> = row.get("names").unwrap_or_default();
                    for expected in &expected_constraints {
                        if !names.iter().any(|n| n == *expected) {
                            tracing::error!(
                                "MISSING CONSTRAINT: {} — MERGE operations will do full label scans. \
                                 Clean up duplicates with cleanup_sync_data and restart.",
                                expected
                            );
                        }
                    }
                    tracing::info!(
                        "Schema verification: {}/{} constraints present",
                        names.len(),
                        expected_constraints.len()
                    );
                }
            }
            Err(e) => {
                tracing::warn!("Could not verify constraints: {}", e);
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

}

