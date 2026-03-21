//! Neo4j Code structure, symbols, imports, calls, exploration, impact, and embeddings

use super::client::Neo4jClient;
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

/// Relationship types that belong to the code structure layer.
/// These are recreated during sync and can be safely deleted during cleanup.
/// Knowledge relationships (LINKED_TO, AFFECTS, DISCUSSED, TOUCHES, CO_CHANGED, CO_CHANGED_TRANSITIVE)
/// are NOT in this list and will survive cleanup.
const CODE_REL_TYPES: &[&str] = &[
    "CONTAINS",
    "IMPORTS",
    "HAS_IMPORT",
    "CALLS",
    "IMPLEMENTS_FOR",
    "IMPLEMENTS_TRAIT",
    "INCLUDES_ENTITY",
    "IMPORTS_SYMBOL",
    "USES_TYPE",
    "EXTENDS",
    "IMPLEMENTS",
    "STEP_IN_PROCESS",
];

impl Neo4jClient {
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

    /// Delete files that are no longer on the filesystem.
    ///
    /// Returns `(files_deleted, symbols_deleted, deleted_paths)` so the caller
    /// can also clean secondary indexes (e.g. Meilisearch).
    pub async fn delete_stale_files(
        &self,
        project_id: Uuid,
        valid_paths: &[String],
    ) -> Result<(usize, usize, Vec<String>)> {
        // First, count and collect paths of what we're about to delete
        let count_q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WHERE NOT f.path IN $valid_paths
            OPTIONAL MATCH (f)-[:CONTAINS]->(symbol)
            RETURN count(DISTINCT f) AS file_count,
                   count(DISTINCT symbol) AS symbol_count,
                   collect(DISTINCT f.path) AS stale_paths
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("valid_paths", valid_paths.to_vec());

        let mut result = self.graph.execute(count_q).await?;
        let (file_count, symbol_count, stale_paths) = if let Some(row) = result.next().await? {
            let files: i64 = row.get("file_count").unwrap_or(0);
            let symbols: i64 = row.get("symbol_count").unwrap_or(0);
            let paths: Vec<String> = row.get("stale_paths").unwrap_or_default();
            (files as usize, symbols as usize, paths)
        } else {
            (0, 0, vec![])
        };

        if file_count == 0 {
            return Ok((0, 0, vec![]));
        }

        // Audit knowledge relationships that will be destroyed
        let audit_q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WHERE NOT f.path IN $valid_paths
            OPTIONAL MATCH (f)-[:CONTAINS]->(symbol)
            OPTIONAL MATCH ()-[kr]->(f)
              WHERE type(kr) IN ['LINKED_TO', 'AFFECTS', 'DISCUSSED', 'TOUCHES', 'CO_CHANGED', 'CO_CHANGED_TRANSITIVE']
            OPTIONAL MATCH ()-[kr2]->(symbol)
              WHERE type(kr2) IN ['LINKED_TO', 'AFFECTS', 'DISCUSSED', 'TOUCHES', 'CO_CHANGED', 'CO_CHANGED_TRANSITIVE']
            RETURN count(DISTINCT kr) + count(DISTINCT kr2) AS knowledge_rels_count
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("valid_paths", valid_paths.to_vec());

        if let Ok(mut audit_result) = self.graph.execute(audit_q).await {
            if let Ok(Some(row)) = audit_result.next().await {
                let knowledge_count: i64 = row.get("knowledge_rels_count").unwrap_or(0);
                if knowledge_count > 0 {
                    tracing::warn!(
                        "delete_stale_files: {} knowledge relationships (LINKED_TO/AFFECTS/DISCUSSED/TOUCHES/CO_CHANGED) \
                         will be destroyed for stale files in project {}. \
                         These will be reconstructed by post-sync knowledge reconstruction.",
                        knowledge_count,
                        project_id
                    );
                }
            }
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

        Ok((file_count, symbol_count, stale_paths))
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

    /// Batch upsert file nodes using UNWIND.
    ///
    /// Creates or updates all file nodes in a single query and links them
    /// to their project in a second query.
    pub async fn batch_upsert_files(&self, files: &[FileNode]) -> Result<()> {
        if files.is_empty() {
            return Ok(());
        }

        use crate::neo4j::batch::{run_unwind_in_chunks, BoltMap};

        let items: Vec<BoltMap> = files
            .iter()
            .map(|f| {
                let mut m = BoltMap::new();
                m.insert("path".into(), f.path.clone().into());
                m.insert("language".into(), f.language.clone().into());
                m.insert("hash".into(), f.hash.clone().into());
                m.insert("last_parsed".into(), f.last_parsed.to_rfc3339().into());
                m.insert(
                    "project_id".into(),
                    f.project_id
                        .map(|id| id.to_string())
                        .unwrap_or_default()
                        .into(),
                );
                m
            })
            .collect();

        run_unwind_in_chunks(
            &self.graph,
            items,
            r#"
            UNWIND $items AS item
            MERGE (f:File {path: item.path})
            SET f.language = item.language,
                f.hash = item.hash,
                f.last_parsed = datetime(item.last_parsed),
                f.project_id = item.project_id
            "#,
        )
        .await?;

        // Link files to projects (only for files with a project_id)
        let project_items: Vec<BoltMap> = files
            .iter()
            .filter(|f| f.project_id.is_some())
            .map(|f| {
                let mut m = BoltMap::new();
                m.insert("path".into(), f.path.clone().into());
                m.insert(
                    "project_id".into(),
                    f.project_id.unwrap().to_string().into(),
                );
                m
            })
            .collect();

        if !project_items.is_empty() {
            run_unwind_in_chunks(
                &self.graph,
                project_items,
                r#"
                UNWIND $items AS item
                MATCH (p:Project {id: item.project_id})
                MATCH (f:File {path: item.path})
                MERGE (p)-[:CONTAINS]->(f)
                "#,
            )
            .await?;
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

    /// Count files for a project (lightweight COUNT query, no data transfer).
    pub async fn count_project_files(&self, project_id: Uuid) -> Result<i64> {
        let q = query(
            "MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File) RETURN count(f) AS cnt",
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            Ok(row.get::<i64>("cnt")?)
        } else {
            Ok(0)
        }
    }

    /// List all sub-file symbols (Function, Struct, Trait, Enum) for a project.
    /// Returns tuples of (id, name, symbol_type, file_path, visibility, line_start).
    /// Used by the graph visualization endpoint to include code-level detail nodes.
    pub async fn list_project_symbols(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<(String, String, String, String, Option<String>, Option<i64>)>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)-[:CONTAINS]->(s)
            WHERE s:Function OR s:Struct OR s:Trait OR s:Enum
            RETURN s.id AS id,
                   s.name AS name,
                   CASE
                     WHEN s:Function THEN 'function'
                     WHEN s:Struct THEN 'struct'
                     WHEN s:Trait THEN 'trait'
                     WHEN s:Enum THEN 'enum'
                   END AS symbol_type,
                   f.path AS file_path,
                   s.visibility AS visibility,
                   s.line_start AS line_start
            ORDER BY f.path, s.line_start
            LIMIT $limit
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("limit", limit as i64);

        let mut result = self.graph.execute(q).await?;
        let mut symbols = Vec::new();

        while let Some(row) = result.next().await? {
            symbols.push((
                row.get::<String>("id").unwrap_or_default(),
                row.get::<String>("name").unwrap_or_default(),
                row.get::<String>("symbol_type").unwrap_or_default(),
                row.get::<String>("file_path").unwrap_or_default(),
                row.get::<String>("visibility").ok(),
                row.get::<i64>("line_start").ok(),
            ));
        }

        Ok(symbols)
    }

    /// List EXTENDS and IMPLEMENTS edges between structs/traits in a project.
    /// Returns (source_id, target_id, relation_type).
    pub async fn get_project_inheritance_edges(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String, String)>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)-[:CONTAINS]->(child)
                  -[r:EXTENDS|IMPLEMENTS]->(parent)<-[:CONTAINS]-(f2:File)<-[:CONTAINS]-(p)
            WHERE (child:Struct OR child:Trait) AND (parent:Struct OR parent:Trait)
            RETURN child.id AS source_id, parent.id AS target_id, type(r) AS rel_type
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut edges = Vec::new();

        while let Some(row) = result.next().await? {
            edges.push((
                row.get::<String>("source_id").unwrap_or_default(),
                row.get::<String>("target_id").unwrap_or_default(),
                row.get::<String>("rel_type").unwrap_or_default(),
            ));
        }

        Ok(edges)
    }

    /// Backfill project_id on existing symbol nodes (Function, Struct, Enum, Trait)
    /// that were created before the denormalization was added.
    /// Inherits project_id from the parent File node via CONTAINS relationships.
    pub async fn backfill_symbol_project_ids(&self) -> Result<i64> {
        let mut total = 0i64;
        for label in &["Function", "Struct", "Enum", "Trait"] {
            let cypher = format!(
                "MATCH (p:Project)-[:CONTAINS]->(f:File)-[:CONTAINS]->(n:{}) \
                 WHERE n.project_id IS NULL \
                 SET n.project_id = p.id \
                 RETURN count(n) AS cnt",
                label
            );
            let q = query(&cypher);
            let mut result = self.graph.execute(q).await?;
            if let Some(row) = result.next().await? {
                total += row.get::<i64>("cnt").unwrap_or(0);
            }
        }
        Ok(total)
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
                s.docstring = $docstring,
                s.parent_class = $parent_class,
                s.interfaces = $interfaces
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
        .param("docstring", s.docstring.clone().unwrap_or_default())
        .param("parent_class", s.parent_class.clone().unwrap_or_default())
        .param("interfaces", s.interfaces.clone());

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
        // Strategy: try direct ID match first (O(1) via uniqueness constraint),
        // then fall back to project-scoped name match with LIMIT 1
        let struct_id = format!("{}:{}", impl_node.file_path, impl_node.for_type);
        let enum_id = format!("{}:{}", impl_node.file_path, impl_node.for_type);

        // Phase 1: Direct ID match (same-file struct or enum)
        let q = query(
            r#"
            MATCH (i:Impl {id: $impl_id})
            OPTIONAL MATCH (s:Struct {id: $struct_id})
            OPTIONAL MATCH (e:Enum {id: $enum_id})
            WITH i, COALESCE(s, e) AS target
            WHERE target IS NOT NULL
            MERGE (i)-[:IMPLEMENTS_FOR]->(target)
            RETURN count(*) AS linked
            "#,
        )
        .param("impl_id", id.clone())
        .param("struct_id", struct_id)
        .param("enum_id", enum_id);

        let direct_linked = match self.graph.execute(q).await {
            Ok(mut result) => {
                if let Ok(Some(row)) = result.next().await {
                    row.get::<i64>("linked").unwrap_or(0) > 0
                } else {
                    false
                }
            }
            Err(_) => false,
        };

        // Phase 2: Project-scoped fallback (struct/enum in a different file)
        if !direct_linked {
            let q = query(
                r#"
                MATCH (i:Impl {id: $impl_id})
                MATCH (file:File {path: $file_path})<-[:CONTAINS]-(p:Project)
                OPTIONAL MATCH (s:Struct {name: $type_name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p)
                OPTIONAL MATCH (e:Enum {name: $type_name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p)
                WITH i, COALESCE(s, e) AS target
                WHERE target IS NOT NULL
                WITH i, target LIMIT 1
                MERGE (i)-[:IMPLEMENTS_FOR]->(target)
                "#,
            )
            .param("impl_id", id.clone())
            .param("type_name", impl_node.for_type.clone())
            .param("file_path", impl_node.file_path.clone());

            let _ = self.graph.run(q).await;
        }

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
    /// Uses a 2-phase strategy: prefer same-file callee, then project-scoped with LIMIT 1
    /// to avoid Cartesian products on common names like "new", "default", "from".
    /// Sets `confidence` (0.0-1.0) and `reason` properties on the CALLS relationship.
    pub async fn create_call_relationship(
        &self,
        caller_id: &str,
        callee_name: &str,
        project_id: Option<Uuid>,
        confidence: f64,
        reason: &str,
    ) -> Result<()> {
        // Double protection: filter out built-in calls at insertion level
        use crate::parser::noise_filter;
        if noise_filter::is_builtin_call(callee_name) {
            return Ok(());
        }

        // Phase 1: Try same-file match (most common case, O(1) via index)
        // Extract file_path from caller_id (format: "file_path:func_name:line_start")
        let caller_file_path = caller_id.rsplitn(3, ':').last().unwrap_or(caller_id);

        let same_file_q = query(
            r#"
            MATCH (caller:Function {id: $caller_id})
            MATCH (callee:Function {name: $callee_name})
            WHERE callee.file_path = $caller_file_path AND callee.id <> $caller_id
            WITH caller, callee LIMIT 1
            MERGE (caller)-[r:CALLS]->(callee)
            SET r.confidence = $confidence, r.reason = $reason
            RETURN count(*) AS linked
            "#,
        )
        .param("caller_id", caller_id)
        .param("callee_name", callee_name)
        .param("caller_file_path", caller_file_path)
        .param("confidence", confidence)
        .param("reason", reason);

        let same_file_linked = match self.graph.execute(same_file_q).await {
            Ok(mut result) => {
                if let Ok(Some(row)) = result.next().await {
                    row.get::<i64>("linked").unwrap_or(0) > 0
                } else {
                    false
                }
            }
            Err(_) => false,
        };

        if same_file_linked {
            return Ok(());
        }

        // Phase 2: Project-scoped fallback with LIMIT 1
        let q = match project_id {
            Some(pid) => query(
                r#"
                MATCH (caller:Function {id: $caller_id})
                MATCH (callee:Function {name: $callee_name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                WHERE callee.id <> $caller_id
                WITH caller, callee LIMIT 1
                MERGE (caller)-[r:CALLS]->(callee)
                SET r.confidence = $confidence, r.reason = $reason
                "#,
            )
            .param("caller_id", caller_id)
            .param("callee_name", callee_name)
            .param("project_id", pid.to_string())
            .param("confidence", confidence)
            .param("reason", reason),
            None => query(
                r#"
                MATCH (caller:Function {id: $caller_id})
                MATCH (callee:Function {name: $callee_name})
                WHERE callee.id <> $caller_id
                WITH caller, callee LIMIT 1
                MERGE (caller)-[r:CALLS]->(callee)
                SET r.confidence = $confidence, r.reason = $reason
                "#,
            )
            .param("caller_id", caller_id)
            .param("callee_name", callee_name)
            .param("confidence", confidence)
            .param("reason", reason),
        };

        // Ignore errors if callee not found (might be external)
        let _ = self.graph.run(q).await;
        Ok(())
    }

    /// Create an IMPORTS_SYMBOL relationship from an Import node to a matching symbol
    /// (Struct, Enum, Trait, or Function) within the same project.
    pub async fn create_imports_symbol_relationship(
        &self,
        import_id: &str,
        symbol_name: &str,
        project_id: Option<Uuid>,
    ) -> Result<()> {
        let q = match project_id {
            Some(pid) => query(
                r#"
                MATCH (i:Import {id: $import_id})
                MATCH (i)<-[:HAS_IMPORT]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                OPTIONAL MATCH (s:Struct {name: $symbol_name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p)
                OPTIONAL MATCH (e:Enum {name: $symbol_name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p)
                OPTIONAL MATCH (t:Trait {name: $symbol_name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p)
                WITH i, COALESCE(s, e, t) AS target
                WHERE target IS NOT NULL
                WITH i, target LIMIT 1
                MERGE (i)-[:IMPORTS_SYMBOL]->(target)
                "#,
            )
            .param("import_id", import_id)
            .param("symbol_name", symbol_name)
            .param("project_id", pid.to_string()),
            None => query(
                r#"
                MATCH (i:Import {id: $import_id})
                OPTIONAL MATCH (s:Struct {name: $symbol_name})
                OPTIONAL MATCH (e:Enum {name: $symbol_name})
                OPTIONAL MATCH (t:Trait {name: $symbol_name})
                WITH i, COALESCE(s, e, t) AS target
                WHERE target IS NOT NULL
                WITH i, target LIMIT 1
                MERGE (i)-[:IMPORTS_SYMBOL]->(target)
                "#,
            )
            .param("import_id", import_id)
            .param("symbol_name", symbol_name),
        };

        let _ = self.graph.run(q).await;
        Ok(())
    }

    // ========================================================================
    // Batch upsert operations (UNWIND)
    // ========================================================================

    /// Batch upsert functions using UNWIND for a single Neo4j transaction per call.
    /// Produces identical nodes/relationships as calling upsert_function individually.
    pub async fn batch_upsert_functions(&self, functions: &[FunctionNode]) -> Result<()> {
        if functions.is_empty() {
            return Ok(());
        }

        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = functions
            .iter()
            .map(|func| {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "id".into(),
                    format!("{}:{}:{}", func.file_path, func.name, func.line_start).into(),
                );
                m.insert("name".into(), func.name.clone().into());
                m.insert("visibility".into(), format!("{:?}", func.visibility).into());
                m.insert(
                    "params".into(),
                    serde_json::to_string(&func.params)
                        .unwrap_or_default()
                        .into(),
                );
                m.insert(
                    "return_type".into(),
                    func.return_type.clone().unwrap_or_default().into(),
                );
                m.insert("generics".into(), func.generics.clone().into());
                m.insert("is_async".into(), func.is_async.into());
                m.insert("is_unsafe".into(), func.is_unsafe.into());
                m.insert("complexity".into(), (func.complexity as i64).into());
                m.insert("file_path".into(), func.file_path.clone().into());
                m.insert("line_start".into(), (func.line_start as i64).into());
                m.insert("line_end".into(), (func.line_end as i64).into());
                m.insert(
                    "docstring".into(),
                    func.docstring.clone().unwrap_or_default().into(),
                );
                m
            })
            .collect();

        let q = query(
            r#"
            UNWIND $items AS func
            MERGE (f:Function {id: func.id})
            SET f.name = func.name,
                f.visibility = func.visibility,
                f.params = func.params,
                f.return_type = func.return_type,
                f.generics = func.generics,
                f.is_async = func.is_async,
                f.is_unsafe = func.is_unsafe,
                f.complexity = func.complexity,
                f.file_path = func.file_path,
                f.line_start = func.line_start,
                f.line_end = func.line_end,
                f.docstring = func.docstring
            WITH f, func
            MATCH (file:File {path: func.file_path})
            MERGE (file)-[:CONTAINS]->(f)
            SET f.project_id = file.project_id
            "#,
        )
        .param("items", items);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Batch upsert structs using UNWIND.
    pub async fn batch_upsert_structs(&self, structs: &[StructNode]) -> Result<()> {
        if structs.is_empty() {
            return Ok(());
        }

        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = structs
            .iter()
            .map(|s| {
                let mut m = std::collections::HashMap::new();
                m.insert("id".into(), format!("{}:{}", s.file_path, s.name).into());
                m.insert("name".into(), s.name.clone().into());
                m.insert("visibility".into(), format!("{:?}", s.visibility).into());
                m.insert("generics".into(), s.generics.clone().into());
                m.insert("file_path".into(), s.file_path.clone().into());
                m.insert("line_start".into(), (s.line_start as i64).into());
                m.insert("line_end".into(), (s.line_end as i64).into());
                m.insert(
                    "docstring".into(),
                    s.docstring.clone().unwrap_or_default().into(),
                );
                m.insert(
                    "parent_class".into(),
                    s.parent_class.clone().unwrap_or_default().into(),
                );
                m.insert("interfaces".into(), s.interfaces.clone().into());
                m
            })
            .collect();

        let q = query(
            r#"
            UNWIND $items AS s
            MERGE (st:Struct {id: s.id})
            SET st.name = s.name,
                st.visibility = s.visibility,
                st.generics = s.generics,
                st.file_path = s.file_path,
                st.line_start = s.line_start,
                st.line_end = s.line_end,
                st.docstring = s.docstring,
                st.parent_class = s.parent_class,
                st.interfaces = s.interfaces
            WITH st, s
            MATCH (file:File {path: s.file_path})
            MERGE (file)-[:CONTAINS]->(st)
            SET st.project_id = file.project_id
            "#,
        )
        .param("items", items);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Batch upsert traits using UNWIND.
    pub async fn batch_upsert_traits(&self, traits: &[TraitNode]) -> Result<()> {
        if traits.is_empty() {
            return Ok(());
        }

        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = traits
            .iter()
            .map(|t| {
                let mut m = std::collections::HashMap::new();
                m.insert("id".into(), format!("{}:{}", t.file_path, t.name).into());
                m.insert("name".into(), t.name.clone().into());
                m.insert("visibility".into(), format!("{:?}", t.visibility).into());
                m.insert("generics".into(), t.generics.clone().into());
                m.insert("file_path".into(), t.file_path.clone().into());
                m.insert("line_start".into(), (t.line_start as i64).into());
                m.insert("line_end".into(), (t.line_end as i64).into());
                m.insert(
                    "docstring".into(),
                    t.docstring.clone().unwrap_or_default().into(),
                );
                m
            })
            .collect();

        let q = query(
            r#"
            UNWIND $items AS t
            MERGE (tr:Trait {id: t.id})
            SET tr.name = t.name,
                tr.visibility = t.visibility,
                tr.generics = t.generics,
                tr.file_path = t.file_path,
                tr.line_start = t.line_start,
                tr.line_end = t.line_end,
                tr.docstring = t.docstring
            WITH tr, t
            MATCH (file:File {path: t.file_path})
            MERGE (file)-[:CONTAINS]->(tr)
            SET tr.project_id = file.project_id
            "#,
        )
        .param("items", items);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Batch upsert enums using UNWIND.
    pub async fn batch_upsert_enums(&self, enums: &[EnumNode]) -> Result<()> {
        if enums.is_empty() {
            return Ok(());
        }

        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = enums
            .iter()
            .map(|e| {
                let mut m = std::collections::HashMap::new();
                m.insert("id".into(), format!("{}:{}", e.file_path, e.name).into());
                m.insert("name".into(), e.name.clone().into());
                m.insert("visibility".into(), format!("{:?}", e.visibility).into());
                m.insert("variants".into(), e.variants.clone().into());
                m.insert("file_path".into(), e.file_path.clone().into());
                m.insert("line_start".into(), (e.line_start as i64).into());
                m.insert("line_end".into(), (e.line_end as i64).into());
                m.insert(
                    "docstring".into(),
                    e.docstring.clone().unwrap_or_default().into(),
                );
                m
            })
            .collect();

        let q = query(
            r#"
            UNWIND $items AS e
            MERGE (en:Enum {id: e.id})
            SET en.name = e.name,
                en.visibility = e.visibility,
                en.variants = e.variants,
                en.file_path = e.file_path,
                en.line_start = e.line_start,
                en.line_end = e.line_end,
                en.docstring = e.docstring
            WITH en, e
            MATCH (file:File {path: e.file_path})
            MERGE (file)-[:CONTAINS]->(en)
            SET en.project_id = file.project_id
            "#,
        )
        .param("items", items);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Batch upsert impl blocks using UNWIND — 3 phases matching upsert_impl behavior:
    /// Phase 1: MERGE Impl nodes + CONTAINS relationship
    /// Phase 2: IMPLEMENTS_FOR (same-file direct match, then project-scoped fallback)
    /// Phase 3: IMPLEMENTS_TRAIT (local trait match, then external trait creation)
    pub async fn batch_upsert_impls(&self, impls: &[ImplNode]) -> Result<()> {
        if impls.is_empty() {
            return Ok(());
        }

        // Phase 1: MERGE Impl nodes + CONTAINS
        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = impls
            .iter()
            .map(|imp| {
                let id = format!(
                    "{}:impl:{}:{}",
                    imp.file_path,
                    imp.for_type,
                    imp.trait_name.as_deref().unwrap_or("self")
                );
                let mut m = std::collections::HashMap::new();
                m.insert("id".into(), id.into());
                m.insert("for_type".into(), imp.for_type.clone().into());
                m.insert(
                    "trait_name".into(),
                    imp.trait_name.clone().unwrap_or_default().into(),
                );
                m.insert("generics".into(), imp.generics.clone().into());
                m.insert(
                    "where_clause".into(),
                    imp.where_clause.clone().unwrap_or_default().into(),
                );
                m.insert("file_path".into(), imp.file_path.clone().into());
                m.insert("line_start".into(), (imp.line_start as i64).into());
                m.insert("line_end".into(), (imp.line_end as i64).into());
                // Pre-computed IDs for Phase 2
                m.insert(
                    "struct_id".into(),
                    format!("{}:{}", imp.file_path, imp.for_type).into(),
                );
                m
            })
            .collect();

        let q = query(
            r#"
            UNWIND $items AS imp
            MERGE (i:Impl {id: imp.id})
            SET i.for_type = imp.for_type,
                i.trait_name = imp.trait_name,
                i.generics = imp.generics,
                i.where_clause = imp.where_clause,
                i.file_path = imp.file_path,
                i.line_start = imp.line_start,
                i.line_end = imp.line_end
            WITH i, imp
            MATCH (file:File {path: imp.file_path})
            MERGE (file)-[:CONTAINS]->(i)
            "#,
        )
        .param("items", items.clone());

        self.graph.run(q).await?;

        // Phase 2a: IMPLEMENTS_FOR — direct ID match (same-file struct/enum)
        let q = query(
            r#"
            UNWIND $items AS imp
            MATCH (i:Impl {id: imp.id})
            OPTIONAL MATCH (s:Struct {id: imp.struct_id})
            OPTIONAL MATCH (e:Enum {id: imp.struct_id})
            WITH i, imp, COALESCE(s, e) AS target
            WHERE target IS NOT NULL
            MERGE (i)-[:IMPLEMENTS_FOR]->(target)
            RETURN imp.id AS linked_id
            "#,
        )
        .param("items", items.clone());

        let mut linked_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut result = self.graph.execute(q).await?;
        while let Some(row) = result.next().await? {
            if let Ok(id) = row.get::<String>("linked_id") {
                linked_ids.insert(id);
            }
        }

        // Phase 2b: IMPLEMENTS_FOR — project-scoped fallback for unresolved
        let unresolved: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = items
            .iter()
            .filter(|m| {
                let id = match m.get("id") {
                    Some(neo4rs::BoltType::String(s)) => s.value.clone(),
                    _ => String::new(),
                };
                !linked_ids.contains(&id)
            })
            .cloned()
            .collect();

        if !unresolved.is_empty() {
            let q = query(
                r#"
                UNWIND $items AS imp
                CALL {
                    WITH imp
                    MATCH (i:Impl {id: imp.id})
                    MATCH (file:File {path: imp.file_path})<-[:CONTAINS]-(p:Project)
                    OPTIONAL MATCH (s:Struct {name: imp.for_type})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p)
                    OPTIONAL MATCH (e:Enum {name: imp.for_type})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p)
                    WITH i, COALESCE(s, e) AS target
                    WHERE target IS NOT NULL
                    WITH i, target LIMIT 1
                    MERGE (i)-[:IMPLEMENTS_FOR]->(target)
                }
                "#,
            )
            .param("items", unresolved);

            let _ = self.graph.run(q).await;
        }

        // Phase 3: IMPLEMENTS_TRAIT (only for trait impls)
        let trait_impls: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = impls
            .iter()
            .filter(|imp| imp.trait_name.is_some())
            .map(|imp| {
                let trait_name = imp.trait_name.as_deref().unwrap();
                let id = format!("{}:impl:{}:{}", imp.file_path, imp.for_type, trait_name);
                let (simple_name, source) = Self::parse_trait_path(trait_name);
                let external_id = format!("external:trait:{}", trait_name);

                let mut m = std::collections::HashMap::new();
                m.insert("impl_id".into(), id.into());
                m.insert("trait_name".into(), trait_name.to_string().into());
                m.insert("simple_name".into(), simple_name.into());
                m.insert("source".into(), source.into());
                m.insert("full_path".into(), trait_name.to_string().into());
                m.insert("external_id".into(), external_id.into());
                m
            })
            .collect();

        if !trait_impls.is_empty() {
            // Phase 3a: Try linking to existing local trait
            let q = query(
                r#"
                UNWIND $items AS imp
                MATCH (i:Impl {id: imp.impl_id})
                MATCH (t:Trait {name: imp.trait_name})
                WHERE t.is_external IS NULL OR t.is_external = false
                MERGE (i)-[:IMPLEMENTS_TRAIT]->(t)
                RETURN imp.impl_id AS linked_id
                "#,
            )
            .param("items", trait_impls.clone());

            let mut trait_linked: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            let mut result = self.graph.execute(q).await?;
            while let Some(row) = result.next().await? {
                if let Ok(id) = row.get::<String>("linked_id") {
                    trait_linked.insert(id);
                }
            }

            // Phase 3b: Create/link external traits for unresolved
            let unresolved_traits: Vec<std::collections::HashMap<String, neo4rs::BoltType>> =
                trait_impls
                    .into_iter()
                    .filter(|m| {
                        let id = match m.get("impl_id") {
                            Some(neo4rs::BoltType::String(s)) => s.value.clone(),
                            _ => String::new(),
                        };
                        !trait_linked.contains(&id)
                    })
                    .collect();

            if !unresolved_traits.is_empty() {
                let q = query(
                    r#"
                    UNWIND $items AS imp
                    MERGE (t:Trait {id: imp.external_id})
                    ON CREATE SET
                        t.name = imp.simple_name,
                        t.full_path = imp.full_path,
                        t.is_external = true,
                        t.source = imp.source,
                        t.visibility = 'public',
                        t.generics = [],
                        t.file_path = '',
                        t.line_start = 0,
                        t.line_end = 0
                    ON MATCH SET
                        t.source = CASE WHEN t.source = 'unknown' THEN imp.source ELSE t.source END
                    WITH t, imp
                    MATCH (i:Impl {id: imp.impl_id})
                    MERGE (i)-[:IMPLEMENTS_TRAIT]->(t)
                    "#,
                )
                .param("items", unresolved_traits);

                let _ = self.graph.run(q).await;
            }
        }

        Ok(())
    }

    /// Batch upsert imports using UNWIND.
    pub async fn batch_upsert_imports(&self, imports: &[ImportNode]) -> Result<()> {
        if imports.is_empty() {
            return Ok(());
        }

        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = imports
            .iter()
            .map(|imp| {
                let id = format!("{}:{}:{}", imp.file_path, imp.line, imp.path);
                let mut m = std::collections::HashMap::new();
                m.insert("id".into(), id.into());
                m.insert("path".into(), imp.path.clone().into());
                m.insert("alias".into(), imp.alias.clone().unwrap_or_default().into());
                m.insert("items".into(), imp.items.clone().into());
                m.insert("file_path".into(), imp.file_path.clone().into());
                m.insert("line".into(), (imp.line as i64).into());
                m
            })
            .collect();

        let q = query(
            r#"
            UNWIND $items AS imp
            MERGE (i:Import {id: imp.id})
            SET i.path = imp.path,
                i.alias = imp.alias,
                i.items = imp.items,
                i.file_path = imp.file_path,
                i.line = imp.line
            WITH i, imp
            MATCH (file:File {path: imp.file_path})
            MERGE (file)-[:HAS_IMPORT]->(i)
            "#,
        )
        .param("items", items);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Batch create File→IMPORTS→File relationships using UNWIND.
    /// Takes a list of (source_file_path, target_file_path, import_path) tuples.
    pub async fn batch_create_import_relationships(
        &self,
        relationships: &[(String, String, String)],
    ) -> Result<()> {
        if relationships.is_empty() {
            return Ok(());
        }

        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = relationships
            .iter()
            .map(|(source, target, import_path)| {
                let mut m = std::collections::HashMap::new();
                m.insert("source".into(), source.clone().into());
                m.insert("target".into(), target.clone().into());
                m.insert("import_path".into(), import_path.clone().into());
                m
            })
            .collect();

        let q = query(
            r#"
            UNWIND $items AS rel
            MATCH (source:File {path: rel.source})
            MATCH (target:File {path: rel.target})
            MERGE (source)-[r:IMPORTS]->(target)
            SET r.import_path = rel.import_path
            "#,
        )
        .param("items", items);

        let _ = self.graph.run(q).await;
        Ok(())
    }

    /// Batch create Import→IMPORTS_SYMBOL→(Struct|Enum|Trait) relationships using UNWIND.
    /// Takes a list of (import_id, symbol_name, project_id) tuples.
    pub async fn batch_create_imports_symbol_relationships(
        &self,
        relationships: &[(String, String, Option<Uuid>)],
    ) -> Result<()> {
        if relationships.is_empty() {
            return Ok(());
        }

        // Split by project_id presence for different Cypher queries
        let with_project: Vec<_> = relationships
            .iter()
            .filter(|(_, _, pid)| pid.is_some())
            .collect();
        let without_project: Vec<_> = relationships
            .iter()
            .filter(|(_, _, pid)| pid.is_none())
            .collect();

        if !with_project.is_empty() {
            // Group by project_id for efficiency
            let mut by_project: std::collections::HashMap<Uuid, Vec<(String, String)>> =
                std::collections::HashMap::new();
            for (import_id, symbol_name, pid) in &with_project {
                by_project
                    .entry(pid.unwrap())
                    .or_default()
                    .push((import_id.clone(), symbol_name.clone()));
            }

            for (pid, rels) in by_project {
                let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = rels
                    .iter()
                    .map(|(import_id, symbol_name)| {
                        let mut m = std::collections::HashMap::new();
                        m.insert("import_id".into(), import_id.clone().into());
                        m.insert("symbol_name".into(), symbol_name.clone().into());
                        m
                    })
                    .collect();

                let q = query(
                    r#"
                    UNWIND $items AS rel
                    CALL {
                        WITH rel
                        MATCH (i:Import {id: rel.import_id})
                        MATCH (symbol {name: rel.symbol_name, project_id: $project_id})
                        WHERE symbol:Struct OR symbol:Enum OR symbol:Trait
                        WITH i, symbol LIMIT 1
                        MERGE (i)-[:IMPORTS_SYMBOL]->(symbol)
                    }
                    "#,
                )
                .param("items", items)
                .param("project_id", pid.to_string());

                let _ = self.graph.run(q).await;
            }
        }

        if !without_project.is_empty() {
            let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = without_project
                .iter()
                .map(|(import_id, symbol_name, _)| {
                    let mut m = std::collections::HashMap::new();
                    m.insert("import_id".into(), import_id.clone().into());
                    m.insert("symbol_name".into(), symbol_name.clone().into());
                    m
                })
                .collect();

            let q = query(
                r#"
                UNWIND $items AS rel
                CALL {
                    WITH rel
                    MATCH (i:Import {id: rel.import_id})
                    MATCH (symbol)
                    WHERE symbol.name = rel.symbol_name
                      AND (symbol:Struct OR symbol:Enum OR symbol:Trait)
                    WITH i, symbol LIMIT 1
                    MERGE (i)-[:IMPORTS_SYMBOL]->(symbol)
                }
                "#,
            )
            .param("items", items);

            let _ = self.graph.run(q).await;
        }

        Ok(())
    }

    /// Batch create CALLS relationships using UNWIND — 2-phase strategy:
    /// Phase 1: same-file callee match (most common, O(1) via index)
    /// Phase 2: project-scoped fallback for unresolved calls
    ///
    /// Both phases are chunked (BATCH_SIZE items per query) to avoid Neo4j OOM/timeout
    /// on large projects (50K+ calls).
    pub async fn batch_create_call_relationships(
        &self,
        calls: &[crate::parser::FunctionCall],
        project_id: Option<Uuid>,
    ) -> Result<()> {
        if calls.is_empty() {
            return Ok(());
        }

        use crate::neo4j::batch::BATCH_SIZE;
        // Double protection: filter out built-in calls at insertion level
        // (primary filter is in the parser, this catches any remaining)
        use crate::parser::noise_filter;

        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = calls
            .iter()
            .filter(|call| !noise_filter::is_builtin_call(&call.callee_name))
            .map(|call| {
                let caller_file_path = call
                    .caller_id
                    .rsplitn(3, ':')
                    .last()
                    .unwrap_or(&call.caller_id)
                    .to_string();
                let mut m = std::collections::HashMap::new();
                m.insert("caller_id".into(), call.caller_id.clone().into());
                m.insert("callee_name".into(), call.callee_name.clone().into());
                m.insert("caller_file_path".into(), caller_file_path.into());
                m.insert(
                    "confidence".into(),
                    neo4rs::BoltType::Float(neo4rs::BoltFloat {
                        value: call.confidence,
                    }),
                );
                m.insert("reason".into(), call.reason.clone().into());
                m
            })
            .collect();

        // Phase 1: same-file match with CALL {} subquery for per-row LIMIT 1
        // Chunked to avoid OOM on large call arrays (50K+ items)
        let phase1_cypher = r#"
            UNWIND $items AS call
            CALL {
                WITH call
                MATCH (caller:Function {id: call.caller_id})
                MATCH (callee:Function {name: call.callee_name})
                WHERE callee.file_path = call.caller_file_path AND callee.id <> call.caller_id
                WITH caller, callee, call LIMIT 1
                MERGE (caller)-[r:CALLS]->(callee)
                SET r.confidence = call.confidence, r.reason = call.reason
                RETURN caller.id AS resolved_caller, callee.name AS resolved_callee
            }
            RETURN resolved_caller, resolved_callee
            "#;

        let mut resolved: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        // Phase 1 failure is not fatal — Phase 2 will try all calls
        for chunk in items.chunks(BATCH_SIZE) {
            let q = query(phase1_cypher).param("items", chunk.to_vec());
            if let Ok(mut result) = self.graph.execute(q).await {
                while let Ok(Some(row)) = result.next().await {
                    if let (Ok(caller), Ok(callee)) = (
                        row.get::<String>("resolved_caller"),
                        row.get::<String>("resolved_callee"),
                    ) {
                        resolved.insert((caller, callee));
                    }
                }
            }
        }

        // Phase 2: project-scoped fallback for unresolved calls
        let unresolved: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = items
            .into_iter()
            .filter(|m| {
                let caller = match m.get("caller_id") {
                    Some(neo4rs::BoltType::String(s)) => s.value.clone(),
                    _ => String::new(),
                };
                let callee = match m.get("callee_name") {
                    Some(neo4rs::BoltType::String(s)) => s.value.clone(),
                    _ => String::new(),
                };
                !resolved.contains(&(caller, callee))
            })
            .collect();

        if !unresolved.is_empty() {
            match project_id {
                Some(pid) => {
                    let cypher = r#"
                        UNWIND $items AS call
                        CALL {
                            WITH call
                            MATCH (caller:Function {id: call.caller_id})
                            MATCH (callee:Function {name: call.callee_name, project_id: $project_id})
                            WHERE callee.id <> call.caller_id
                            WITH caller, callee, call LIMIT 1
                            MERGE (caller)-[r:CALLS]->(callee)
                            SET r.confidence = call.confidence, r.reason = call.reason
                        }
                        "#;
                    let pid_str = pid.to_string();
                    for chunk in unresolved.chunks(BATCH_SIZE) {
                        let q = query(cypher)
                            .param("items", chunk.to_vec())
                            .param("project_id", pid_str.clone());
                        if let Err(e) = self.graph.run(q).await {
                            tracing::warn!("batch_create_call_relationships Phase 2 (project) chunk failed: {}", e);
                        }
                    }
                }
                None => {
                    let cypher = r#"
                        UNWIND $items AS call
                        CALL {
                            WITH call
                            MATCH (caller:Function {id: call.caller_id})
                            MATCH (callee:Function {name: call.callee_name})
                            WHERE callee.id <> call.caller_id
                            WITH caller, callee, call LIMIT 1
                            MERGE (caller)-[r:CALLS]->(callee)
                            SET r.confidence = call.confidence, r.reason = call.reason
                        }
                        "#;
                    for chunk in unresolved.chunks(BATCH_SIZE) {
                        let q = query(cypher).param("items", chunk.to_vec());
                        if let Err(e) = self.graph.run(q).await {
                            tracing::warn!(
                                "batch_create_call_relationships Phase 2 (global) chunk failed: {}",
                                e
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Batch create EXTENDS relationships (class inheritance).
    /// Two-phase: 1) same-file match, 2) project-scoped fallback.
    ///
    /// Both phases are chunked (BATCH_SIZE items per query) to avoid Neo4j OOM/timeout.
    pub async fn batch_create_extends_relationships(
        &self,
        rels: &[(String, String, String, String)],
    ) -> Result<()> {
        if rels.is_empty() {
            return Ok(());
        }

        use crate::neo4j::batch::BATCH_SIZE;

        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = rels
            .iter()
            .map(|(child_name, child_file, parent_name, pid)| {
                let mut m = std::collections::HashMap::new();
                m.insert("child_name".into(), child_name.clone().into());
                m.insert("child_file".into(), child_file.clone().into());
                m.insert("parent_name".into(), parent_name.clone().into());
                m.insert("project_id".into(), pid.clone().into());
                m
            })
            .collect();

        // Phase 1: same-file match — chunked to avoid OOM on large arrays
        let phase1_cypher = r#"
            UNWIND $items AS rel
            CALL {
                WITH rel
                MATCH (child:Struct {name: rel.child_name, file_path: rel.child_file})
                MATCH (parent:Struct {name: rel.parent_name, file_path: rel.child_file})
                WITH child, parent LIMIT 1
                MERGE (child)-[:EXTENDS]->(parent)
                RETURN child.name AS resolved, child.file_path AS resolved_file
            }
            RETURN resolved, resolved_file
            "#;

        let mut resolved: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        for chunk in items.chunks(BATCH_SIZE) {
            let q = query(phase1_cypher).param("items", chunk.to_vec());
            if let Ok(mut result) = self.graph.execute(q).await {
                while let Ok(Some(row)) = result.next().await {
                    if let (Ok(name), Ok(file)) = (
                        row.get::<String>("resolved"),
                        row.get::<String>("resolved_file"),
                    ) {
                        resolved.insert((name, file));
                    }
                }
            }
        }

        // Phase 2: project-scoped fallback for unresolved — also chunked
        let unresolved: Vec<_> = items
            .into_iter()
            .filter(|m| {
                let child = match m.get("child_name") {
                    Some(neo4rs::BoltType::String(s)) => s.value.clone(),
                    _ => String::new(),
                };
                let child_file = match m.get("child_file") {
                    Some(neo4rs::BoltType::String(s)) => s.value.clone(),
                    _ => String::new(),
                };
                !resolved.contains(&(child, child_file))
            })
            .collect();

        if !unresolved.is_empty() {
            let phase2_cypher = r#"
                UNWIND $items AS rel
                CALL {
                    WITH rel
                    MATCH (child:Struct {name: rel.child_name, file_path: rel.child_file})
                    MATCH (parent:Struct {name: rel.parent_name, project_id: rel.project_id})
                    WHERE parent.file_path <> child.file_path
                    WITH child, parent LIMIT 1
                    MERGE (child)-[:EXTENDS]->(parent)
                }
                "#;
            for chunk in unresolved.chunks(BATCH_SIZE) {
                let q = query(phase2_cypher).param("items", chunk.to_vec());
                if let Err(e) = self.graph.run(q).await {
                    tracing::warn!(
                        "batch_create_extends_relationships Phase 2 chunk failed: {}",
                        e
                    );
                }
            }
        }

        Ok(())
    }

    /// Batch create IMPLEMENTS relationships (interface/protocol implementation).
    /// Matches against Trait nodes (interfaces stored as Trait in Java, TS, PHP, Kotlin, Swift).
    ///
    /// Chunked (BATCH_SIZE items per query) to avoid Neo4j OOM/timeout.
    pub async fn batch_create_implements_relationships(
        &self,
        rels: &[(String, String, String, String)],
    ) -> Result<()> {
        if rels.is_empty() {
            return Ok(());
        }

        use crate::neo4j::batch::run_unwind_in_chunks;

        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = rels
            .iter()
            .map(|(struct_name, struct_file, iface_name, pid)| {
                let mut m = std::collections::HashMap::new();
                m.insert("struct_name".into(), struct_name.clone().into());
                m.insert("struct_file".into(), struct_file.clone().into());
                m.insert("iface_name".into(), iface_name.clone().into());
                m.insert("project_id".into(), pid.clone().into());
                m
            })
            .collect();

        if let Err(e) = run_unwind_in_chunks(
            &self.graph,
            items,
            r#"
            UNWIND $items AS rel
            CALL {
                WITH rel
                MATCH (s:Struct {name: rel.struct_name, file_path: rel.struct_file})
                MATCH (iface:Trait {name: rel.iface_name, project_id: rel.project_id})
                WITH s, iface LIMIT 1
                MERGE (s)-[:IMPLEMENTS]->(iface)
            }
            "#,
        )
        .await
        {
            tracing::warn!("batch_create_implements_relationships failed: {}", e);
        }
        Ok(())
    }

    /// Clean up ALL sync-generated data from Neo4j.
    /// This deletes File, Function, Struct, Trait, Enum, Impl, Import nodes
    /// and their relationships (CALLS, IMPORTS, IMPLEMENTS_FOR, IMPLEMENTS_TRAIT, CONTAINS, HAS_IMPORT).
    /// Project management data (Project, Plan, Task, Note, etc.) is preserved.
    /// FeatureGraph nodes are also deleted since they depend on code entities.
    pub async fn cleanup_sync_data(&self) -> Result<i64> {
        let mut total_deleted: i64 = 0;
        let batch_size = 10_000;

        // Delete code relationships first (batched to avoid massive transactions)
        // CALLS alone can be 300k+ rels — unbatched DELETE causes OOM/timeout
        //
        // NOTE: Types not yet in use (EXTENDS, IMPLEMENTS, STEP_IN_PROCESS)
        // are forward-compatible no-ops — the batched loop simply returns 0.
        let rel_types = vec![
            "CALLS",
            "IMPORTS",
            "IMPORTS_SYMBOL",
            "USES_TYPE",
            "IMPLEMENTS_FOR",
            "IMPLEMENTS_TRAIT",
            "HAS_IMPORT",
            "INCLUDES_ENTITY",
            // Forward-compatible: Plans 5 & 6
            "EXTENDS",
            "IMPLEMENTS",
            "STEP_IN_PROCESS",
        ];

        for rel_type in &rel_types {
            let mut rel_deleted: i64 = 0;
            loop {
                let cypher = format!(
                    "MATCH ()-[r:{}]->() WITH r LIMIT {} DELETE r RETURN count(r) AS cnt",
                    rel_type, batch_size
                );
                match self.graph.execute(query(&cypher)).await {
                    Ok(mut result) => {
                        if let Ok(Some(row)) = result.next().await {
                            let cnt: i64 = row.get("cnt").unwrap_or(0);
                            if cnt == 0 {
                                break;
                            }
                            rel_deleted += cnt;
                            total_deleted += cnt;
                            tracing::info!(
                                "cleanup_sync_data: deleted batch of {} {} rels (total: {})",
                                cnt,
                                rel_type,
                                rel_deleted
                            );
                        } else {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "cleanup_sync_data rel query failed for {}: {}",
                            rel_type,
                            e
                        );
                        break;
                    }
                }
            }
            if rel_deleted > 0 {
                tracing::info!(
                    "cleanup_sync_data: finished {} — deleted {} total",
                    rel_type,
                    rel_deleted
                );
            }
        }

        // Delete code entity nodes (batched DETACH DELETE removes remaining CONTAINS edges)
        //
        // NOTE: Process is forward-compatible for Plan 6 (Process Detection).
        // Class/Interface are NOT added — Struct and Trait cover these concepts
        // in the current model (Rust-centric). If needed later, add here.
        let node_labels = vec![
            "Function",
            "Struct",
            "Trait",
            "Enum",
            "Impl",
            "Import",
            "File",
            "FeatureGraph",
            // Forward-compatible: Plan 6
            "Process",
        ];

        for label in &node_labels {
            let mut node_deleted: i64 = 0;

            // Step 1: Delete remaining code-structure relationships on these nodes
            for rel_type in CODE_REL_TYPES {
                loop {
                    let cypher = format!(
                        "MATCH (n:{})-[r:{}]-() WITH r LIMIT {} DELETE r RETURN count(r) AS cnt",
                        label, rel_type, batch_size
                    );
                    match self.graph.execute(query(&cypher)).await {
                        Ok(mut result) => {
                            if let Ok(Some(row)) = result.next().await {
                                let cnt: i64 = row.get("cnt").unwrap_or(0);
                                if cnt == 0 {
                                    break;
                                }
                                total_deleted += cnt;
                            } else {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            }

            // Step 2: DELETE nodes that have no remaining relationships (plain DELETE, no DETACH)
            loop {
                let cypher = format!(
                    "MATCH (n:{}) WHERE NOT (n)-[]-() WITH n LIMIT {} DELETE n RETURN count(n) AS cnt",
                    label, batch_size
                );
                match self.graph.execute(query(&cypher)).await {
                    Ok(mut result) => {
                        if let Ok(Some(row)) = result.next().await {
                            let cnt: i64 = row.get("cnt").unwrap_or(0);
                            if cnt == 0 {
                                break;
                            }
                            node_deleted += cnt;
                            total_deleted += cnt;
                            tracing::info!(
                                "cleanup_sync_data: deleted batch of {} {} nodes (total: {})",
                                cnt,
                                label,
                                node_deleted
                            );
                        } else {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("cleanup_sync_data node query failed for {}: {}", label, e);
                        break;
                    }
                }
            }

            // Step 3: DETACH DELETE remaining nodes that still have knowledge relationships
            // These are nodes with LINKED_TO, AFFECTS, DISCUSSED, TOUCHES, CO_CHANGED
            // pointing to them — the knowledge rels are destroyed here as a fallback.
            let mut fallback_deleted: i64 = 0;
            loop {
                let cypher = format!(
                    "MATCH (n:{}) WITH n LIMIT {} DETACH DELETE n RETURN count(n) AS cnt",
                    label, batch_size
                );
                match self.graph.execute(query(&cypher)).await {
                    Ok(mut result) => {
                        if let Ok(Some(row)) = result.next().await {
                            let cnt: i64 = row.get("cnt").unwrap_or(0);
                            if cnt == 0 {
                                break;
                            }
                            fallback_deleted += cnt;
                            node_deleted += cnt;
                            total_deleted += cnt;
                        } else {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("cleanup_sync_data node query failed for {}: {}", label, e);
                        break;
                    }
                }
            }
            if fallback_deleted > 0 {
                tracing::warn!(
                    "cleanup_sync_data: {} {} nodes had knowledge relationships that were destroyed (DETACH DELETE fallback)",
                    fallback_deleted,
                    label
                );
            }

            if node_deleted > 0 {
                tracing::info!(
                    "cleanup_sync_data: finished {} — deleted {} total",
                    label,
                    node_deleted
                );
            }
        }

        tracing::info!(
            "cleanup_sync_data: deleted {} entities/relationships total",
            total_deleted
        );
        Ok(total_deleted)
    }

    /// Delete all CALLS relationships where caller and callee belong to different projects.
    ///
    /// Returns the number of deleted relationships.
    pub async fn cleanup_cross_project_calls(&self) -> Result<i64> {
        // Optimized: instead of scanning ALL CALLS relationships globally (O(total_calls)),
        // anchor on each project and find CALLS that cross project boundaries.
        // This leverages the Project->File->Function index chain.
        let q = query(
            r#"
            MATCH (p:Project)-[:CONTAINS]->(f:File)-[:CONTAINS]->(caller:Function)-[r:CALLS]->(callee:Function)
            WHERE NOT EXISTS {
                MATCH (callee)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p)
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

    /// Delete all CALLS relationships where the callee function name matches a known built-in.
    /// Uses the noise_filter::BUILT_IN_NAMES list. Batched to handle large graphs.
    /// Returns the number of deleted relationships.
    pub async fn cleanup_builtin_calls(&self) -> Result<i64> {
        use crate::parser::noise_filter;

        let builtins: Vec<String> = noise_filter::builtin_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let batch_size = 10_000;
        let mut total_deleted: i64 = 0;

        loop {
            let q = query(
                r#"
                MATCH (caller:Function)-[r:CALLS]->(callee:Function)
                WHERE callee.name IN $builtins
                WITH r LIMIT $batch_size
                DELETE r
                RETURN count(r) AS cnt
                "#,
            )
            .param("builtins", builtins.clone())
            .param("batch_size", batch_size as i64);

            match self.graph.execute(q).await {
                Ok(mut result) => {
                    if let Ok(Some(row)) = result.next().await {
                        let cnt: i64 = row.get("cnt").unwrap_or(0);
                        if cnt == 0 {
                            break;
                        }
                        total_deleted += cnt;
                        tracing::info!(
                            "cleanup_builtin_calls: deleted batch of {} CALLS rels (total: {})",
                            cnt,
                            total_deleted
                        );
                    } else {
                        break;
                    }
                }
                Err(e) => {
                    tracing::warn!("cleanup_builtin_calls query failed: {}", e);
                    break;
                }
            }
        }

        tracing::info!(
            "cleanup_builtin_calls: deleted {} built-in CALLS relationships total",
            total_deleted
        );
        Ok(total_deleted)
    }

    /// Migrate existing CALLS relationships: set default confidence and reason
    /// for any CALLS rel that doesn't have these properties yet.
    /// Returns the number of relationships updated.
    pub async fn migrate_calls_confidence(&self) -> Result<i64> {
        let batch_size = 10_000;
        let mut total_updated: i64 = 0;

        loop {
            let q = query(&format!(
                r#"
                MATCH (caller:Function)-[r:CALLS]->(callee:Function)
                WHERE r.confidence IS NULL
                WITH r LIMIT {}
                SET r.confidence = 0.50, r.reason = 'fuzzy-global'
                RETURN count(r) AS cnt
                "#,
                batch_size
            ));

            match self.graph.execute(q).await {
                Ok(mut result) => {
                    if let Ok(Some(row)) = result.next().await {
                        let cnt: i64 = row.get("cnt").unwrap_or(0);
                        if cnt == 0 {
                            break;
                        }
                        total_updated += cnt;
                        tracing::info!(
                            "migrate_calls_confidence: updated batch of {} CALLS rels (total: {})",
                            cnt,
                            total_updated
                        );
                    } else {
                        break;
                    }
                }
                Err(e) => {
                    tracing::warn!("migrate_calls_confidence query failed: {}", e);
                    break;
                }
            }
        }

        tracing::info!(
            "migrate_calls_confidence: updated {} CALLS relationships total",
            total_updated
        );
        Ok(total_updated)
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
    // Heritage navigation queries (EXTENDS + IMPLEMENTS)
    // ========================================================================

    /// Get the full class hierarchy (parents + children) for a type.
    ///
    /// Traverses EXTENDS up (parents) and down (children) with depth limit.
    pub async fn get_class_hierarchy(
        &self,
        type_name: &str,
        max_depth: u32,
    ) -> Result<serde_json::Value> {
        // Parents (traverse EXTENDS upward)
        let q_parents = query(&format!(
            r#"
            MATCH path = (child)-[:EXTENDS*1..{}]->(ancestor)
            WHERE child.name = $type_name
            WITH ancestor, length(path) AS depth
            WHERE depth <= $max_depth
            RETURN DISTINCT ancestor.name AS name,
                   ancestor.path AS path,
                   depth
            ORDER BY depth ASC
            "#,
            max_depth
        ))
        .param("type_name", type_name)
        .param("max_depth", max_depth as i64);

        let mut result = self.graph.execute(q_parents).await?;
        let mut parents = Vec::new();
        while let Some(row) = result.next().await? {
            let name: String = row.get("name").unwrap_or_default();
            let path: Option<String> = row.get("path").ok();
            let depth: i64 = row.get("depth").unwrap_or(0);
            parents.push(serde_json::json!({
                "name": name,
                "path": path,
                "depth": depth,
            }));
        }

        // Children (traverse EXTENDS downward — reverse direction)
        let q_children = query(&format!(
            r#"
            MATCH path = (descendant)-[:EXTENDS*1..{}]->(parent)
            WHERE parent.name = $type_name
            WITH descendant, length(path) AS depth
            WHERE depth <= $max_depth
            RETURN DISTINCT descendant.name AS name,
                   descendant.path AS path,
                   depth
            ORDER BY depth ASC
            "#,
            max_depth
        ))
        .param("type_name", type_name)
        .param("max_depth", max_depth as i64);

        let mut result = self.graph.execute(q_children).await?;
        let mut children = Vec::new();
        while let Some(row) = result.next().await? {
            let name: String = row.get("name").unwrap_or_default();
            let path: Option<String> = row.get("path").ok();
            let depth: i64 = row.get("depth").unwrap_or(0);
            children.push(serde_json::json!({
                "name": name,
                "path": path,
                "depth": depth,
            }));
        }

        Ok(serde_json::json!({
            "type_name": type_name,
            "parents": parents,
            "children": children,
        }))
    }

    /// Find all subclasses of a given class (direct + transitive via EXTENDS).
    pub async fn find_subclasses(&self, class_name: &str) -> Result<Vec<serde_json::Value>> {
        let q = query(
            r#"
            MATCH path = (child)-[:EXTENDS*1..10]->(parent)
            WHERE parent.name = $class_name
            WITH child, length(path) AS depth
            RETURN DISTINCT child.name AS name,
                   child.path AS path,
                   depth
            ORDER BY depth ASC, name ASC
            "#,
        )
        .param("class_name", class_name);

        let mut result = self.graph.execute(q).await?;
        let mut subclasses = Vec::new();
        while let Some(row) = result.next().await? {
            let name: String = row.get("name").unwrap_or_default();
            let path: Option<String> = row.get("path").ok();
            let depth: i64 = row.get("depth").unwrap_or(0);
            subclasses.push(serde_json::json!({
                "name": name,
                "path": path,
                "depth": depth,
                "direct": depth == 1,
            }));
        }
        Ok(subclasses)
    }

    /// Find all classes/types that implement a given interface (via IMPLEMENTS).
    pub async fn find_interface_implementors(
        &self,
        interface_name: &str,
    ) -> Result<Vec<serde_json::Value>> {
        let q = query(
            r#"
            MATCH (type)-[:IMPLEMENTS]->(iface)
            WHERE iface.name = $interface_name
            RETURN DISTINCT type.name AS name,
                   type.path AS path,
                   labels(type) AS labels
            ORDER BY name ASC
            "#,
        )
        .param("interface_name", interface_name);

        let mut result = self.graph.execute(q).await?;
        let mut implementors = Vec::new();
        while let Some(row) = result.next().await? {
            let name: String = row.get("name").unwrap_or_default();
            let path: Option<String> = row.get("path").ok();
            implementors.push(serde_json::json!({
                "name": name,
                "path": path,
            }));
        }
        Ok(implementors)
    }

    // ========================================================================
    // Process queries
    // ========================================================================

    /// List all detected processes for a project.
    pub async fn list_processes(&self, project_id: uuid::Uuid) -> Result<Vec<serde_json::Value>> {
        let q = query(
            r#"
            MATCH (p:Process {project_id: $project_id})
            OPTIONAL MATCH (p)-[s:STEP_IN_PROCESS]->(f:Function)
            WITH p, count(s) AS step_count
            RETURN p.id AS id,
                   p.label AS label,
                   p.process_type AS process_type,
                   p.entry_point_id AS entry_point,
                   p.terminal_id AS terminal,
                   step_count
            ORDER BY step_count DESC
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut processes = Vec::new();
        while let Some(row) = result.next().await? {
            let id: String = row.get("id").unwrap_or_default();
            let label: String = row.get("label").unwrap_or_default();
            let process_type: String = row.get("process_type").unwrap_or_default();
            let entry_point: String = row.get("entry_point").unwrap_or_default();
            let terminal: String = row.get("terminal").unwrap_or_default();
            let step_count: i64 = row.get("step_count").unwrap_or(0);
            processes.push(serde_json::json!({
                "id": id,
                "label": label,
                "process_type": process_type,
                "entry_point": entry_point,
                "terminal": terminal,
                "step_count": step_count,
            }));
        }
        Ok(processes)
    }

    /// Get details of a specific process including ordered steps.
    pub async fn get_process_detail(&self, process_id: &str) -> Result<Option<serde_json::Value>> {
        let q = query(
            r#"
            MATCH (p:Process {id: $process_id})
            OPTIONAL MATCH (p)-[s:STEP_IN_PROCESS]->(f:Function)
            WITH p, f, s
            ORDER BY s.order ASC
            RETURN p.id AS id,
                   p.label AS label,
                   p.process_type AS process_type,
                   p.entry_point_id AS entry_point,
                   p.terminal_id AS terminal,
                   collect({
                       function_id: f.id,
                       name: f.name,
                       file_path: f.file_path,
                       order: s.order
                   }) AS steps
            "#,
        )
        .param("process_id", process_id);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let id: String = row.get("id").unwrap_or_default();
            if id.is_empty() {
                return Ok(None);
            }
            let label: String = row.get("label").unwrap_or_default();
            let process_type: String = row.get("process_type").unwrap_or_default();
            let entry_point: String = row.get("entry_point").unwrap_or_default();
            let terminal: String = row.get("terminal").unwrap_or_default();
            let steps: Vec<serde_json::Value> =
                serde_json::from_value(row.get::<serde_json::Value>("steps").unwrap_or_default())
                    .unwrap_or_default();

            Ok(Some(serde_json::json!({
                "id": id,
                "label": label,
                "process_type": process_type,
                "entry_point": entry_point,
                "terminal": terminal,
                "steps": steps,
            })))
        } else {
            Ok(None)
        }
    }

    /// Get scored entry points for a project.
    ///
    /// Entry points are functions with in_degree=0 on the CALLS graph,
    /// or functions matching framework/naming patterns.
    pub async fn get_entry_points(
        &self,
        project_id: uuid::Uuid,
        limit: usize,
    ) -> Result<Vec<serde_json::Value>> {
        let q = query(
            r#"
            MATCH (f:Function)
            WHERE f.project_id = $project_id OR
                  EXISTS {
                      MATCH (file:File {project_id: $project_id})
                      WHERE f.file_path = file.path
                  }
            OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
            WITH f, count(caller) AS in_degree
            WHERE in_degree = 0
            OPTIONAL MATCH (f)-[:CALLS]->(callee:Function)
            WITH f, in_degree, count(callee) AS out_degree
            WHERE out_degree > 0
            RETURN f.name AS name,
                   f.file_path AS file_path,
                   f.id AS id,
                   in_degree,
                   out_degree
            ORDER BY out_degree DESC
            LIMIT $limit
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("limit", limit as i64);

        let mut result = self.graph.execute(q).await?;
        let mut entry_points = Vec::new();
        while let Some(row) = result.next().await? {
            let name: String = row.get("name").unwrap_or_default();
            let file_path: String = row.get("file_path").unwrap_or_default();
            let id: String = row.get("id").unwrap_or_default();
            let in_degree: i64 = row.get("in_degree").unwrap_or(0);
            let out_degree: i64 = row.get("out_degree").unwrap_or(0);
            entry_points.push(serde_json::json!({
                "name": name,
                "file_path": file_path,
                "id": id,
                "in_degree": in_degree,
                "out_degree": out_degree,
            }));
        }
        Ok(entry_points)
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

    /// Get direct callers of a function with confidence scores on each CALLS edge.
    /// Depth 1 only (direct callers) to return per-edge confidence.
    pub async fn get_callers_with_confidence(
        &self,
        function_name: &str,
        project_id: Option<Uuid>,
    ) -> Result<Vec<(String, String, f64, String)>> {
        // Returns (caller_name, caller_file, confidence, reason)
        let q = match project_id {
            Some(pid) => query(
                r#"
                MATCH (f:Function {name: $name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                MATCH (caller:Function)-[r:CALLS]->(f)
                WHERE EXISTS { MATCH (caller)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }
                RETURN DISTINCT caller.name AS name, caller.file_path AS file,
                       coalesce(r.confidence, 0.50) AS confidence,
                       coalesce(r.reason, 'unknown') AS reason
                "#,
            )
            .param("name", function_name)
            .param("project_id", pid.to_string()),
            None => query(
                r#"
                MATCH (f:Function {name: $name})
                MATCH (caller:Function)-[r:CALLS]->(f)
                RETURN DISTINCT caller.name AS name, caller.file_path AS file,
                       coalesce(r.confidence, 0.50) AS confidence,
                       coalesce(r.reason, 'unknown') AS reason
                "#,
            )
            .param("name", function_name),
        };

        let mut result = self.graph.execute(q).await?;
        let mut callers = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(name), Ok(file)) = (row.get::<String>("name"), row.get::<String>("file")) {
                let confidence = row.get::<f64>("confidence").unwrap_or(0.50);
                let reason = row
                    .get::<String>("reason")
                    .unwrap_or_else(|_| "unknown".to_string());
                callers.push((name, file, confidence, reason));
            }
        }

        Ok(callers)
    }

    /// Get direct callees of a function with confidence scores on each CALLS edge.
    pub async fn get_callees_with_confidence(
        &self,
        function_name: &str,
        project_id: Option<Uuid>,
    ) -> Result<Vec<(String, String, f64, String)>> {
        let q = match project_id {
            Some(pid) => query(
                r#"
                MATCH (f:Function {name: $name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                MATCH (f)-[r:CALLS]->(callee:Function)
                WHERE EXISTS { MATCH (callee)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }
                RETURN DISTINCT callee.name AS name, callee.file_path AS file,
                       coalesce(r.confidence, 0.50) AS confidence,
                       coalesce(r.reason, 'unknown') AS reason
                "#,
            )
            .param("name", function_name)
            .param("project_id", pid.to_string()),
            None => query(
                r#"
                MATCH (f:Function {name: $name})
                MATCH (f)-[r:CALLS]->(callee:Function)
                RETURN DISTINCT callee.name AS name, callee.file_path AS file,
                       coalesce(r.confidence, 0.50) AS confidence,
                       coalesce(r.reason, 'unknown') AS reason
                "#,
            )
            .param("name", function_name),
        };

        let mut result = self.graph.execute(q).await?;
        let mut callees = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(name), Ok(file)) = (row.get::<String>("name"), row.get::<String>("file")) {
                let confidence = row.get::<f64>("confidence").unwrap_or(0.50);
                let reason = row
                    .get::<String>("reason")
                    .unwrap_or_else(|_| "unknown".to_string());
                callees.push((name, file, confidence, reason));
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

    /// Find all files impacted by a change to a given file.
    ///
    /// Combines two traversal axes:
    /// 1. **IMPORTS** — files that import (directly or transitively) the target file
    /// 2. **CALLS**  — files whose functions call functions defined in the target file
    ///
    /// Returns a deduplicated list of file paths (excluding the target itself).
    pub async fn find_impacted_files(
        &self,
        file_path: &str,
        depth: u32,
        project_id: Option<Uuid>,
    ) -> Result<Vec<String>> {
        let q = match project_id {
            Some(pid) => query(&format!(
                r#"
                MATCH (f:File {{path: $path}})<-[:CONTAINS]-(p:Project {{id: $project_id}})
                // Axis 1: files that import the target (transitively)
                OPTIONAL MATCH (f)<-[:IMPORTS*1..{}]-(imp:File)
                WHERE imp IS NULL OR EXISTS {{ MATCH (imp)<-[:CONTAINS]-(p) }}
                // Axis 2: files whose functions call functions in the target
                WITH f, p, COLLECT(DISTINCT imp.path) AS import_paths
                OPTIONAL MATCH (f)-[:CONTAINS]->(fn:Function)<-[:CALLS]-(caller:Function)<-[:CONTAINS]-(caller_file:File)
                WHERE caller_file <> f AND EXISTS {{ MATCH (caller_file)<-[:CONTAINS]-(p) }}
                WITH import_paths, COLLECT(DISTINCT caller_file.path) AS call_paths
                // Merge both axes
                WITH import_paths + call_paths AS all_paths
                UNWIND all_paths AS path
                WITH path WHERE path IS NOT NULL
                RETURN DISTINCT path
                "#,
                depth
            ))
            .param("path", file_path)
            .param("project_id", pid.to_string()),
            None => query(&format!(
                r#"
                MATCH (f:File {{path: $path}})
                // Axis 1: files that import the target (transitively)
                OPTIONAL MATCH (f)<-[:IMPORTS*1..{}]-(imp:File)
                // Axis 2: files whose functions call functions in the target
                WITH f, COLLECT(DISTINCT imp.path) AS import_paths
                OPTIONAL MATCH (f)-[:CONTAINS]->(fn:Function)<-[:CALLS]-(caller:Function)<-[:CONTAINS]-(caller_file:File)
                WHERE caller_file <> f
                WITH import_paths, COLLECT(DISTINCT caller_file.path) AS call_paths
                // Merge both axes
                WITH import_paths + call_paths AS all_paths
                UNWIND all_paths AS path
                WITH path WHERE path IS NOT NULL
                RETURN DISTINCT path
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

    /// Batch upsert Process nodes using UNWIND + MERGE.
    pub async fn batch_upsert_processes(
        &self,
        processes: &[crate::neo4j::models::ProcessNode],
    ) -> Result<()> {
        if processes.is_empty() {
            return Ok(());
        }

        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = processes
            .iter()
            .map(|p| {
                let mut m = std::collections::HashMap::new();
                m.insert("id".into(), p.id.clone().into());
                m.insert("label".into(), p.label.clone().into());
                m.insert("process_type".into(), p.process_type.clone().into());
                m.insert("step_count".into(), (p.step_count as i64).into());
                m.insert("entry_point_id".into(), p.entry_point_id.clone().into());
                m.insert("terminal_id".into(), p.terminal_id.clone().into());
                m.insert(
                    "communities".into(),
                    p.communities
                        .iter()
                        .map(|c| *c as i64)
                        .collect::<Vec<i64>>()
                        .into(),
                );
                m.insert(
                    "project_id".into(),
                    p.project_id
                        .map(|id| id.to_string())
                        .unwrap_or_default()
                        .into(),
                );
                m
            })
            .collect();

        let q = query(
            r#"
            UNWIND $items AS proc
            MERGE (p:Process {id: proc.id})
            SET p.label = proc.label,
                p.process_type = proc.process_type,
                p.step_count = proc.step_count,
                p.entry_point_id = proc.entry_point_id,
                p.terminal_id = proc.terminal_id,
                p.communities = proc.communities,
                p.project_id = proc.project_id
            "#,
        )
        .param("items", items);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Batch create STEP_IN_PROCESS relationships.
    pub async fn batch_create_step_relationships(
        &self,
        steps: &[(String, String, u32)],
    ) -> Result<()> {
        if steps.is_empty() {
            return Ok(());
        }

        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = steps
            .iter()
            .map(|(process_id, function_id, step_number)| {
                let mut m = std::collections::HashMap::new();
                m.insert("process_id".into(), process_id.clone().into());
                m.insert("function_id".into(), function_id.clone().into());
                m.insert("step".into(), (*step_number as i64).into());
                m
            })
            .collect();

        let q = query(
            r#"
            UNWIND $items AS s
            MATCH (p:Process {id: s.process_id})
            MATCH (f:Function {id: s.function_id})
            MERGE (p)-[r:STEP_IN_PROCESS]->(f)
            SET r.step = s.step
            "#,
        )
        .param("items", items);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete all Process nodes and STEP_IN_PROCESS relationships for a project.
    pub async fn delete_project_processes(&self, project_id: uuid::Uuid) -> Result<u64> {
        let q = query(
            r#"
            MATCH (p:Process {project_id: $project_id})
            DETACH DELETE p
            RETURN count(p) AS deleted
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let deleted: i64 = row.get("deleted")?;
            Ok(deleted as u64)
        } else {
            Ok(0)
        }
    }

    // ========================================================================
    // GraIL — Computed property invalidation
    // ========================================================================

    /// Invalidate pre-computed GraIL properties on changed files and their neighbors.
    ///
    /// Two-phase invalidation:
    /// 1. Direct files: mark all 3 versions as stale (cc, DNA, WL)
    /// 2. Neighbors: 1-hop gets cc_version=-1, up to 2-hop gets wl_hash_version=-1
    ///
    /// Returns total number of File nodes marked as stale.
    pub async fn invalidate_computed_properties(
        &self,
        project_id: Uuid,
        paths: &[String],
    ) -> anyhow::Result<u64> {
        if paths.is_empty() {
            return Ok(0);
        }

        let pid = project_id.to_string();
        let path_list: Vec<String> = paths.to_vec();

        // Phase 1: Mark directly changed files — all 3 versions stale
        let q1 = neo4rs::query(
            r#"
            UNWIND $paths AS p
            MATCH (f:File {path: p, project_id: $pid})
            SET f.cc_version = -1,
                f.structural_dna_version = -1,
                f.wl_hash_version = -1,
                f.invalidated_at = datetime()
            RETURN count(f) AS updated
            "#,
        )
        .param("paths", path_list.clone())
        .param("pid", pid.clone());

        let start1 = std::time::Instant::now();
        let mut result1 = self.graph.execute(q1).await?;
        self.log_slow_query("invalidate_direct", start1);
        let direct_count: i64 = if let Some(row) = result1.next().await? {
            row.get("updated")?
        } else {
            0
        };

        // Phase 2: Mark 1-hop neighbors — cc_version + wl_hash_version stale
        let q2 = neo4rs::query(
            r#"
            UNWIND $paths AS p
            MATCH (f:File {path: p, project_id: $pid})-[*1]-(neighbor:File)
            WHERE neighbor.project_id = $pid AND NOT neighbor.path IN $paths
            SET neighbor.cc_version = -1,
                neighbor.wl_hash_version = -1,
                neighbor.invalidated_at = datetime()
            RETURN count(DISTINCT neighbor) AS updated
            "#,
        )
        .param("paths", path_list.clone())
        .param("pid", pid.clone());

        let start2 = std::time::Instant::now();
        let mut result2 = self.graph.execute(q2).await?;
        self.log_slow_query("invalidate_1hop", start2);
        let hop1_count: i64 = if let Some(row) = result2.next().await? {
            row.get("updated")?
        } else {
            0
        };

        // Phase 3: Mark 2-hop only neighbors — wl_hash_version stale only
        let q3 = neo4rs::query(
            r#"
            UNWIND $paths AS p
            MATCH (f:File {path: p, project_id: $pid})-[*2]-(neighbor:File)
            WHERE neighbor.project_id = $pid
              AND NOT neighbor.path IN $paths
              AND neighbor.cc_version <> -1
            SET neighbor.wl_hash_version = -1,
                neighbor.invalidated_at = datetime()
            RETURN count(DISTINCT neighbor) AS updated
            "#,
        )
        .param("paths", path_list)
        .param("pid", pid);

        let start3 = std::time::Instant::now();
        let mut result3 = self.graph.execute(q3).await?;
        self.log_slow_query("invalidate_2hop", start3);
        let hop2_count: i64 = if let Some(row) = result3.next().await? {
            row.get("updated")?
        } else {
            0
        };

        let total = (direct_count + hop1_count + hop2_count) as u64;
        tracing::info!(
            direct = direct_count,
            hop1 = hop1_count,
            hop2 = hop2_count,
            total = total,
            "Invalidated GraIL computed properties"
        );

        Ok(total)
    }

    // ========================================================================
    // Multi-signal impact queries
    // ========================================================================

    /// Knowledge density for a file: (notes LINKED_TO + decisions AFFECTS) / max across project.
    /// Returns f64 in [0, 1]. 0.0 if no knowledge exists in the project.
    pub async fn get_knowledge_density(&self, file_path: &str, project_id: &str) -> Result<f64> {
        let q = query(
            r#"
            // Count notes + decisions linked to the target file
            MATCH (f:File {path: $path})<-[:CONTAINS]-(p:Project {id: $project_id})
            OPTIONAL MATCH (n:Note)-[:LINKED_TO]->(f)
            WHERE n.status IN ['active', 'needs_review']
            WITH f, p, count(DISTINCT n) AS note_count
            OPTIONAL MATCH (d:Decision)-[:AFFECTS]->(f)
            WHERE d.status IN ['proposed', 'accepted']
            WITH f, p, note_count + count(DISTINCT d) AS target_count

            // Find max knowledge count across all files in the project
            OPTIONAL MATCH (p)-[:CONTAINS]->(af:File)
            OPTIONAL MATCH (an:Note)-[:LINKED_TO]->(af)
            WHERE an.status IN ['active', 'needs_review']
            WITH f, target_count, af, count(DISTINCT an) AS af_note_count
            OPTIONAL MATCH (ad:Decision)-[:AFFECTS]->(af)
            WHERE ad.status IN ['proposed', 'accepted']
            WITH target_count, af_note_count + count(DISTINCT ad) AS af_total
            WITH target_count, max(af_total) AS max_count

            RETURN target_count,
                   CASE WHEN max_count > 0 THEN toFloat(target_count) / max_count ELSE 0.0 END AS density
            "#,
        )
        .param("path", file_path)
        .param("project_id", project_id);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let density: f64 = row.get("density").unwrap_or(0.0);
            Ok(density)
        } else {
            Ok(0.0)
        }
    }

    /// Read the PageRank score (cc_pagerank) from a File node.
    /// Falls back to 0.0 if the property is absent or the file doesn't exist.
    pub async fn get_node_pagerank(&self, file_path: &str, project_id: &str) -> Result<f64> {
        let q = query(
            r#"
            MATCH (f:File {path: $path})<-[:CONTAINS]-(p:Project {id: $project_id})
            RETURN coalesce(f.cc_pagerank, 0.0) AS pagerank
            "#,
        )
        .param("path", file_path)
        .param("project_id", project_id);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let pr: f64 = row.get("pagerank").unwrap_or(0.0);
            Ok(pr)
        } else {
            Ok(0.0)
        }
    }

    /// Bridge proximity: for the top-10 co-changers of a file, compute
    /// 1.0 / shortestPath_length as a proximity score.
    /// Returns Vec<(path, score)> sorted by score descending.
    pub async fn get_bridge_proximity(
        &self,
        file_path: &str,
        project_id: &str,
    ) -> Result<Vec<(String, f64)>> {
        let q = query(
            r#"
            // Get top-10 co-changers
            MATCH (f:File {path: $path})<-[:CONTAINS]-(p:Project {id: $project_id})
            MATCH (f)-[r:CO_CHANGED]-(co:File)
            WHERE EXISTS { MATCH (co)<-[:CONTAINS]-(p) }
            WITH f, co ORDER BY r.count DESC LIMIT 10

            // For each co-changer, find shortest structural path (IMPORTS|CALLS between files)
            OPTIONAL MATCH sp = shortestPath((f)-[:IMPORTS|CALLS*1..5]-(co))
            WITH co.path AS co_path,
                 CASE WHEN sp IS NOT NULL THEN 1.0 / length(sp) ELSE 0.0 END AS proximity
            RETURN co_path, proximity
            ORDER BY proximity DESC
            "#,
        )
        .param("path", file_path)
        .param("project_id", project_id);

        let mut result = self.graph.execute(q).await?;
        let mut scores = Vec::new();

        while let Some(row) = result.next().await? {
            let path: String = row.get("co_path")?;
            let score: f64 = row.get("proximity").unwrap_or(0.0);
            scores.push((path, score));
        }

        Ok(scores)
    }

    /// Extract the enclosing bridge subgraph between two nodes via bidirectional
    /// BFS intersection (GraIL-inspired).
    ///
    /// Phase 1: BFS from source and target, intersect neighborhoods.
    /// Phase 2: Get edges between bridge nodes.
    /// `max_hops` controls BFS radius (1..=5). `relation_types` filters edge types.
    pub async fn find_bridge_subgraph(
        &self,
        source: &str,
        target: &str,
        max_hops: u32,
        relation_types: &[String],
        project_id: &str,
    ) -> Result<(
        Vec<crate::graph::models::BridgeRawNode>,
        Vec<crate::graph::models::BridgeRawEdge>,
    )> {
        use crate::graph::models::{BridgeRawEdge, BridgeRawNode};
        use std::collections::HashSet;

        // Build relationship pattern from whitelisted types
        let allowed = ["IMPORTS", "CALLS", "CO_CHANGED", "CO_CHANGED_TRANSITIVE", "EXTENDS", "IMPLEMENTS"];
        let rel_pattern = if relation_types.is_empty() {
            "IMPORTS".to_string()
        } else {
            let filtered: Vec<&str> = relation_types
                .iter()
                .filter_map(|t| {
                    let upper = t.to_uppercase();
                    allowed.iter().find(|a| **a == upper).copied()
                })
                .collect();
            if filtered.is_empty() {
                "IMPORTS".to_string()
            } else {
                filtered.join("|")
            }
        };

        // max_hops is a validated u32 (1..=5), safe for format!
        // Relationship types are whitelisted above, no injection risk
        let hops = max_hops.clamp(1, 5);

        // Phase 1: BFS intersection to find bridge node paths
        let node_cypher = format!(
            r#"
            MATCH (s:File {{path: $source}})<-[:CONTAINS]-(p:Project {{id: $project_id}})
            MATCH (t:File {{path: $target}})<-[:CONTAINS]-(p)
            CALL {{
              WITH s, p
              MATCH (s)-[:{rel}*1..{hops}]-(n)
              WHERE (n:File OR n:Function)
                AND EXISTS {{ MATCH (n)<-[:CONTAINS*1..2]-(p) }}
              RETURN collect(DISTINCT n) AS src_nbrs
            }}
            CALL {{
              WITH t, p
              MATCH (t)-[:{rel}*1..{hops}]-(n)
              WHERE (n:File OR n:Function)
                AND EXISTS {{ MATCH (n)<-[:CONTAINS*1..2]-(p) }}
              RETURN collect(DISTINCT n) AS tgt_nbrs
            }}
            WITH s, t, [x IN src_nbrs WHERE x IN tgt_nbrs] + [s, t] AS bridge_list
            UNWIND bridge_list AS bn
            WITH DISTINCT bn
            RETURN bn.path AS path,
                   CASE WHEN 'Function' IN labels(bn) THEN 'Function' ELSE 'File' END AS node_type
            "#,
            rel = rel_pattern,
            hops = hops
        );

        let q = neo4rs::query(&node_cypher)
            .param("source", source)
            .param("target", target)
            .param("project_id", project_id);

        let mut result = self.graph.execute(q).await?;
        let mut nodes = Vec::new();
        let mut node_paths = HashSet::new();

        while let Some(row) = result.next().await? {
            let path: String = row.get("path").unwrap_or_default();
            let node_type: String = row.get("node_type").unwrap_or_else(|_| "File".to_string());
            if node_paths.insert(path.clone()) {
                nodes.push(BridgeRawNode { path, node_type });
            }
        }

        // Phase 2: Get edges between bridge nodes
        if nodes.len() < 2 {
            return Ok((nodes, Vec::new()));
        }

        let paths_list: Vec<&str> = node_paths.iter().map(|s| s.as_str()).collect();

        let edge_cypher = format!(
            r#"
            UNWIND $paths AS p1
            MATCH (n1 {{path: p1}})-[r:{rel}]->(n2)
            WHERE n2.path IN $paths
            RETURN DISTINCT n1.path AS from_path, type(r) AS rel_type, n2.path AS to_path
            "#,
            rel = rel_pattern
        );

        let eq = neo4rs::query(&edge_cypher).param("paths", paths_list);

        let mut edge_result = self.graph.execute(eq).await?;
        let mut edges = Vec::new();

        while let Some(row) = edge_result.next().await? {
            let from_path: String = row.get("from_path").unwrap_or_default();
            let to_path: String = row.get("to_path").unwrap_or_default();
            let rel_type: String = row.get("rel_type").unwrap_or_default();
            edges.push(BridgeRawEdge {
                from_path,
                to_path,
                rel_type,
            });
        }

        Ok((nodes, edges))
    }

    /// Compute average multi-signal impact score for top-10 files by PageRank.
    /// Single Cypher query that approximates the 5-signal fusion:
    ///   structural (normalized degree), co_change (churn_score), knowledge (knowledge_density),
    ///   pagerank (cc_pagerank normalized), bridge (cc_betweenness normalized).
    /// Default weights: structural=0.35, co_change=0.25, knowledge=0.15, pagerank=0.10, bridge=0.15
    pub async fn get_avg_multi_signal_score(&self, project_id: impl Into<String>) -> Result<f64> {
        let project_id = project_id.into();
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WHERE f.cc_pagerank IS NOT NULL
            WITH collect(f) AS all_files,
                 max(f.cc_pagerank) AS max_pr,
                 max(f.cc_betweenness) AS max_bt
            // No GDS data → return 0
            WITH all_files, max_pr, max_bt
            WHERE size(all_files) > 0
            UNWIND all_files AS f
            WITH f, max_pr, max_bt
            ORDER BY f.cc_pagerank DESC LIMIT 10

            // Count structural connections (IMPORTS in/out)
            OPTIONAL MATCH (f)-[:IMPORTS]->(imp_out:File)
            WITH f, max_pr, max_bt, count(DISTINCT imp_out) AS out_deg
            OPTIONAL MATCH (imp_in:File)-[:IMPORTS]->(f)
            WITH f, max_pr, max_bt, out_deg + count(DISTINCT imp_in) AS total_degree

            // Normalize each signal to 0-1
            WITH f, max_pr, max_bt,
                 CASE WHEN total_degree >= 20 THEN 1.0
                      ELSE toFloat(total_degree) / 20.0 END AS structural,
                 coalesce(f.churn_score, 0.0) AS co_change,
                 coalesce(f.knowledge_density, 0.0) AS knowledge,
                 CASE WHEN max_pr > 0 THEN toFloat(f.cc_pagerank) / max_pr ELSE 0.0 END AS pagerank,
                 CASE WHEN max_bt > 0 THEN toFloat(f.cc_betweenness) / max_bt ELSE 0.0 END AS bridge

            // Weighted fusion (default profile weights)
            WITH (0.35 * structural + 0.25 * co_change + 0.15 * knowledge
                  + 0.10 * pagerank + 0.15 * bridge) AS combined

            RETURN avg(combined) AS avg_score
            "#,
        )
        .param("project_id", project_id);

        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            let score: f64 = row.get("avg_score").unwrap_or(0.0);
            Ok(score)
        } else {
            Ok(0.0)
        }
    }
}
