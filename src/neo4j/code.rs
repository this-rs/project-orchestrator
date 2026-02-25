//! Neo4j Code structure, symbols, imports, calls, exploration, impact, and embeddings

use super::client::Neo4jClient;
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

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
    pub async fn create_call_relationship(
        &self,
        caller_id: &str,
        callee_name: &str,
        project_id: Option<Uuid>,
    ) -> Result<()> {
        // Phase 1: Try same-file match (most common case, O(1) via index)
        // Extract file_path from caller_id (format: "file_path:func_name:line_start")
        let caller_file_path = caller_id.rsplitn(3, ':').last().unwrap_or(caller_id);

        let same_file_q = query(
            r#"
            MATCH (caller:Function {id: $caller_id})
            MATCH (callee:Function {name: $callee_name})
            WHERE callee.file_path = $caller_file_path AND callee.id <> $caller_id
            WITH caller, callee LIMIT 1
            MERGE (caller)-[:CALLS]->(callee)
            RETURN count(*) AS linked
            "#,
        )
        .param("caller_id", caller_id)
        .param("callee_name", callee_name)
        .param("caller_file_path", caller_file_path);

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
                WHERE callee.id <> $caller_id
                WITH caller, callee LIMIT 1
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
                st.docstring = s.docstring
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
    pub async fn batch_create_call_relationships(
        &self,
        calls: &[crate::parser::FunctionCall],
        project_id: Option<Uuid>,
    ) -> Result<()> {
        if calls.is_empty() {
            return Ok(());
        }

        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = calls
            .iter()
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
                m
            })
            .collect();

        // Phase 1: same-file match with CALL {} subquery for per-row LIMIT 1
        let q = query(
            r#"
            UNWIND $items AS call
            CALL {
                WITH call
                MATCH (caller:Function {id: call.caller_id})
                MATCH (callee:Function {name: call.callee_name})
                WHERE callee.file_path = call.caller_file_path AND callee.id <> call.caller_id
                WITH caller, callee LIMIT 1
                MERGE (caller)-[:CALLS]->(callee)
                RETURN caller.id AS resolved_caller, callee.name AS resolved_callee
            }
            RETURN resolved_caller, resolved_callee
            "#,
        )
        .param("items", items.clone());

        let mut resolved: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        // Phase 1 failure is not fatal — Phase 2 will try all calls
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
            let q = match project_id {
                Some(pid) => query(
                    r#"
                    UNWIND $items AS call
                    CALL {
                        WITH call
                        MATCH (caller:Function {id: call.caller_id})
                        MATCH (callee:Function {name: call.callee_name, project_id: $project_id})
                        WHERE callee.id <> call.caller_id
                        WITH caller, callee LIMIT 1
                        MERGE (caller)-[:CALLS]->(callee)
                    }
                    "#,
                )
                .param("items", unresolved)
                .param("project_id", pid.to_string()),
                None => query(
                    r#"
                    UNWIND $items AS call
                    CALL {
                        WITH call
                        MATCH (caller:Function {id: call.caller_id})
                        MATCH (callee:Function {name: call.callee_name})
                        WHERE callee.id <> call.caller_id
                        WITH caller, callee LIMIT 1
                        MERGE (caller)-[:CALLS]->(callee)
                    }
                    "#,
                )
                .param("items", unresolved),
            };

            let _ = self.graph.run(q).await;
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
        let rel_types = vec![
            "CALLS",
            "IMPORTS",
            "IMPLEMENTS_FOR",
            "IMPLEMENTS_TRAIT",
            "HAS_IMPORT",
            "INCLUDES_ENTITY",
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
        let node_labels = vec![
            "Function",
            "Struct",
            "Trait",
            "Enum",
            "Impl",
            "Import",
            "File",
            "FeatureGraph",
        ];

        for label in &node_labels {
            let mut node_deleted: i64 = 0;
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
}
