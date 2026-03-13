//! Neo4j Knowledge Note operations

use super::client::Neo4jClient;
use super::models::DecisionNode;
use crate::notes::{
    EntityType, MemoryHorizon, Note, NoteAnchor, NoteChange, NoteFilters, NoteImportance,
    NoteScope, NoteStatus, NoteType, PropagatedNote,
};
use anyhow::{Context, Result};
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
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
                assertion_rule_json: $assertion_rule_json,
                scar_intensity: $scar_intensity,
                memory_horizon: $memory_horizon
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
        )
        .param("scar_intensity", note.scar_intensity)
        .param("memory_horizon", note.memory_horizon.to_string());

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
            EntityType::Step => "Step",
            EntityType::Constraint => "Constraint",
            EntityType::Milestone => "Milestone",
            EntityType::Release => "Release",
            EntityType::FeatureGraph => "FeatureGraph",
            EntityType::Protocol => "Protocol",
            EntityType::ProtocolState => "ProtocolState",
            EntityType::ProtocolRun => "ProtocolRun",
            EntityType::PlanRun => "PlanRun",
            EntityType::Skill => "Skill",
            EntityType::Note => "Note",
            EntityType::ChatSession => "ChatSession",
            EntityType::Process => "Process",
        };

        // Determine the match field based on entity type.
        // For File entities with relative paths, use ENDS WITH matching
        // since Neo4j stores absolute paths.
        let (match_field, match_value, use_suffix_match) = match entity_type {
            EntityType::File => {
                if entity_id.starts_with('/') {
                    ("path", entity_id.to_string(), false)
                } else {
                    ("path", format!("/{}", entity_id), true)
                }
            }
            EntityType::Commit => ("hash", entity_id.to_string(), false),
            _ => ("id", entity_id.to_string(), false),
        };

        let cypher = if use_suffix_match {
            format!(
                r#"
                MATCH (n:Note {{id: $note_id}})
                MATCH (e:{})
                WHERE e.{} ENDS WITH $entity_id
                MERGE (n)-[r:LINKED_TO]->(e)
                SET r.signature_hash = $sig_hash,
                    r.body_hash = $body_hash,
                    r.last_verified = datetime()
                "#,
                node_label, match_field
            )
        } else {
            format!(
                r#"
                MATCH (n:Note {{id: $note_id}})
                MATCH (e:{} {{{}: $entity_id}})
                MERGE (n)-[r:LINKED_TO]->(e)
                SET r.signature_hash = $sig_hash,
                    r.body_hash = $body_hash,
                    r.last_verified = datetime()
                "#,
                node_label, match_field
            )
        };

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
            EntityType::Step => "Step",
            EntityType::Constraint => "Constraint",
            EntityType::Milestone => "Milestone",
            EntityType::Release => "Release",
            EntityType::FeatureGraph => "FeatureGraph",
            EntityType::Protocol => "Protocol",
            EntityType::ProtocolState => "ProtocolState",
            EntityType::ProtocolRun => "ProtocolRun",
            EntityType::PlanRun => "PlanRun",
            EntityType::Skill => "Skill",
            EntityType::Note => "Note",
            EntityType::ChatSession => "ChatSession",
            EntityType::Process => "Process",
        };

        let (match_field, match_value, use_suffix_match) = match entity_type {
            EntityType::File => {
                if entity_id.starts_with('/') {
                    ("path", entity_id.to_string(), false)
                } else {
                    ("path", format!("/{}", entity_id), true)
                }
            }
            EntityType::Commit => ("hash", entity_id.to_string(), false),
            _ => ("id", entity_id.to_string(), false),
        };

        let cypher = if use_suffix_match {
            format!(
                r#"
                MATCH (n:Note {{id: $note_id}})-[r:LINKED_TO]->(e:{})
                WHERE e.{} ENDS WITH $entity_id
                DELETE r
                "#,
                node_label, match_field
            )
        } else {
            format!(
                r#"
                MATCH (n:Note {{id: $note_id}})-[r:LINKED_TO]->(e:{} {{{}: $entity_id}})
                DELETE r
                "#,
                node_label, match_field
            )
        };

        let q = query(&cypher)
            .param("note_id", note_id.to_string())
            .param("entity_id", match_value);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Propagate LINKED_TO relationships via structural code relations (IMPORTS, CALLS).
    ///
    /// For each note already LINKED_TO a File in the given project, this creates
    /// LINKED_TO relationships to files that are structurally connected (1 hop via
    /// IMPORTS or CALLS). Propagated links are marked with `propagated: true` and
    /// `propagation_source: 'structural'` to distinguish them from explicit links.
    ///
    /// Also propagates via CO_CHANGED relationships (files historically modified together).
    ///
    /// Uses MERGE for idempotent creation. Scoped to a single project to avoid
    /// cross-project contamination.
    ///
    /// Returns the number of propagated links created.
    pub async fn propagate_structural_links(&self, project_id: Uuid) -> Result<usize> {
        // Propagate via IMPORTS and CALLS (1 hop)
        let propagate_code_q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f1:File)
            MATCH (n:Note)-[existing:LINKED_TO]->(f1)
            WHERE existing.propagated IS NULL OR existing.propagated = false
            MATCH (f1)-[:IMPORTS|CALLS*1]->(f2:File)<-[:CONTAINS]-(p)
            WHERE f1 <> f2 AND NOT (n)-[:LINKED_TO]->(f2)
            MERGE (n)-[r:LINKED_TO]->(f2)
            ON CREATE SET
                r.propagated = true,
                r.propagation_source = 'structural',
                r.propagation_depth = 1,
                r.last_verified = datetime()
            RETURN count(r) AS cnt
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut total = 0i64;
        if let Ok(mut result) = self.graph.execute(propagate_code_q).await {
            if let Ok(Some(row)) = result.next().await {
                total += row.get::<i64>("cnt").unwrap_or(0);
            }
        }

        // Propagate via CO_CHANGED (files historically modified together)
        let propagate_cochange_q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f1:File)
            MATCH (n:Note)-[existing:LINKED_TO]->(f1)
            WHERE existing.propagated IS NULL OR existing.propagated = false
            MATCH (f1)-[:CO_CHANGED]->(f2:File)<-[:CONTAINS]-(p)
            WHERE f1 <> f2 AND NOT (n)-[:LINKED_TO]->(f2)
            MERGE (n)-[r:LINKED_TO]->(f2)
            ON CREATE SET
                r.propagated = true,
                r.propagation_source = 'co_changed',
                r.propagation_depth = 1,
                r.last_verified = datetime()
            RETURN count(r) AS cnt
            "#,
        )
        .param("project_id", project_id.to_string());

        if let Ok(mut result) = self.graph.execute(propagate_cochange_q).await {
            if let Ok(Some(row)) = result.next().await {
                total += row.get::<i64>("cnt").unwrap_or(0);
            }
        }

        if total > 0 {
            tracing::info!(
                %project_id,
                propagated = total,
                "Propagated {} structural knowledge links",
                total
            );
        }

        Ok(total as usize)
    }

    /// Propagate knowledge links via semantic similarity (embeddings).
    ///
    /// For each File in the project that has an embedding, queries the
    /// `note_embeddings` HNSW index for the top-K nearest notes.
    /// Creates LINKED_TO relationships with `propagation_source = 'semantic'`
    /// and `similarity_score` for matches above `min_similarity`.
    ///
    /// Only creates links where none already exist (MERGE + WHERE NOT).
    /// Limits to K=5 notes per file to avoid noise.
    pub async fn propagate_semantic_links(
        &self,
        project_id: Uuid,
        min_similarity: f64,
    ) -> Result<usize> {
        // Step 1: Get file paths + embeddings for files in the project
        let files_q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WHERE f.embedding IS NOT NULL
            RETURN f.path AS path, f.embedding AS embedding
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut file_embeddings: Vec<(String, Vec<f64>)> = Vec::new();
        if let Ok(mut result) = self.graph.execute(files_q).await {
            while let Ok(Some(row)) = result.next().await {
                if let (Ok(path), Ok(emb)) =
                    (row.get::<String>("path"), row.get::<Vec<f64>>("embedding"))
                {
                    file_embeddings.push((path, emb));
                }
            }
        }

        if file_embeddings.is_empty() {
            return Ok(0);
        }

        let mut total = 0usize;

        // Step 2: For each file, vector search notes and create LINKED_TO
        for (file_path, embedding) in &file_embeddings {
            let emb_f32: Vec<f32> = embedding.iter().map(|&v| v as f32).collect();

            // Use vector search to find top-5 notes similar to this file
            let search_q = query(
                r#"
                CALL db.index.vector.queryNodes('note_embeddings', $k, $embedding)
                YIELD node AS n, score
                WHERE score > $min_similarity
                  AND n.status IN ['active', 'needs_review']
                WITH n, score
                MATCH (f:File {path: $file_path})
                WHERE NOT (n)-[:LINKED_TO]->(f)
                MERGE (n)-[r:LINKED_TO]->(f)
                ON CREATE SET
                    r.propagated = true,
                    r.propagation_source = 'semantic',
                    r.similarity_score = score,
                    r.last_verified = datetime()
                RETURN count(r) AS cnt
                "#,
            )
            .param("k", 5i64)
            .param("embedding", emb_f32)
            .param("min_similarity", min_similarity)
            .param("file_path", file_path.clone());

            if let Ok(mut result) = self.graph.execute(search_q).await {
                if let Ok(Some(row)) = result.next().await {
                    total += row.get::<i64>("cnt").unwrap_or(0) as usize;
                }
            }
        }

        if total > 0 {
            tracing::info!(
                %project_id,
                files_processed = file_embeddings.len(),
                links_created = total,
                "Propagated {} semantic knowledge links from {} files",
                total,
                file_embeddings.len()
            );
        }

        Ok(total)
    }

    // ================================================================
    // High-Level Entity Propagation (FeatureGraph, Skill, Protocol)
    // ================================================================

    /// Propagate notes linked to FeatureGraphs to their member files (batch).
    ///
    /// For each note LINKED_TO a FeatureGraph, creates LINKED_TO relationships
    /// to files that are members of that feature graph (via INCLUDES_ENTITY).
    /// Scoped to a single project via BELONGS_TO.
    pub async fn propagate_feature_graph_links(&self, project_id: Uuid) -> Result<usize> {
        let q = query(
            r#"
            MATCH (fg:FeatureGraph)-[:BELONGS_TO]->(p:Project {id: $project_id})
            MATCH (n:Note)-[existing:LINKED_TO]->(fg)
            WHERE existing.propagated IS NULL OR existing.propagated = false
            MATCH (fg)-[:INCLUDES_ENTITY]->(f:File)
            WHERE NOT (n)-[:LINKED_TO]->(f)
            MERGE (n)-[r:LINKED_TO]->(f)
            ON CREATE SET
                r.propagated = true,
                r.propagation_source = 'feature_graph',
                r.propagation_depth = 1,
                r.last_verified = datetime()
            RETURN count(r) AS cnt
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut total = 0i64;
        if let Ok(mut result) = self.graph.execute(q).await {
            if let Ok(Some(row)) = result.next().await {
                total = row.get::<i64>("cnt").unwrap_or(0);
            }
        }

        if total > 0 {
            tracing::info!(
                %project_id,
                propagated = total,
                source = "feature_graph",
                "Propagated {} knowledge links via FeatureGraph membership",
                total
            );
        }

        Ok(total as usize)
    }

    /// Propagate notes linked to Skills to files touched by the skill's member notes (batch).
    ///
    /// Path: Note -[:LINKED_TO]-> Skill <-[:MEMBER_OF]- MemberNote -[:LINKED_TO]-> File
    /// Scoped to project via the member note's project_id.
    pub async fn propagate_skill_member_links(&self, project_id: Uuid) -> Result<usize> {
        let q = query(
            r#"
            MATCH (n:Note {project_id: $project_id})-[existing:LINKED_TO]->(s:Skill)
            WHERE existing.propagated IS NULL OR existing.propagated = false
            MATCH (member:Note)-[:MEMBER_OF]->(s)
            MATCH (member)-[:LINKED_TO]->(f:File)
            WHERE NOT (n)-[:LINKED_TO]->(f) AND n <> member
            MERGE (n)-[r:LINKED_TO]->(f)
            ON CREATE SET
                r.propagated = true,
                r.propagation_source = 'skill_member',
                r.propagation_depth = 1,
                r.last_verified = datetime()
            RETURN count(r) AS cnt
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut total = 0i64;
        if let Ok(mut result) = self.graph.execute(q).await {
            if let Ok(Some(row)) = result.next().await {
                total = row.get::<i64>("cnt").unwrap_or(0);
            }
        }

        if total > 0 {
            tracing::info!(
                %project_id,
                propagated = total,
                source = "skill_member",
                "Propagated {} knowledge links via Skill member notes",
                total
            );
        }

        Ok(total as usize)
    }

    /// Propagate notes linked to Protocols to files via Protocol → Skill → MemberNotes → Files (batch).
    ///
    /// Path: Note -[:LINKED_TO]-> Protocol -[:BELONGS_TO_SKILL]-> Skill <-[:MEMBER_OF]- MemberNote -[:LINKED_TO]-> File
    /// Scoped to project via the member note's project_id.
    pub async fn propagate_protocol_skill_links(&self, project_id: Uuid) -> Result<usize> {
        let q = query(
            r#"
            MATCH (n:Note {project_id: $project_id})-[existing:LINKED_TO]->(proto:Protocol)
            WHERE existing.propagated IS NULL OR existing.propagated = false
            MATCH (proto)-[:BELONGS_TO_SKILL]->(s:Skill)
            MATCH (member:Note)-[:MEMBER_OF]->(s)
            MATCH (member)-[:LINKED_TO]->(f:File)
            WHERE NOT (n)-[:LINKED_TO]->(f) AND n <> member
            MERGE (n)-[r:LINKED_TO]->(f)
            ON CREATE SET
                r.propagated = true,
                r.propagation_source = 'protocol_skill',
                r.propagation_depth = 1,
                r.last_verified = datetime()
            RETURN count(r) AS cnt
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut total = 0i64;
        if let Ok(mut result) = self.graph.execute(q).await {
            if let Ok(Some(row)) = result.next().await {
                total = row.get::<i64>("cnt").unwrap_or(0);
            }
        }

        if total > 0 {
            tracing::info!(
                %project_id,
                propagated = total,
                source = "protocol_skill",
                "Propagated {} knowledge links via Protocol→Skill→MemberNotes",
                total
            );
        }

        Ok(total as usize)
    }

    /// Propagate all high-level entity links (FeatureGraph + Skill + Protocol) for a project.
    ///
    /// Called from bootstrap_knowledge_fabric as a fallback/initialization path.
    /// The primary propagation path is event-driven via link_note_to_entity hooks.
    pub async fn propagate_high_level_links(&self, project_id: Uuid) -> Result<usize> {
        let fg = self
            .propagate_feature_graph_links(project_id)
            .await
            .unwrap_or(0);
        let skill = self
            .propagate_skill_member_links(project_id)
            .await
            .unwrap_or(0);
        let proto = self
            .propagate_protocol_skill_links(project_id)
            .await
            .unwrap_or(0);
        let total = fg + skill + proto;

        if total > 0 {
            tracing::info!(
                %project_id,
                feature_graph = fg,
                skill_member = skill,
                protocol_skill = proto,
                total = total,
                "Propagated {} high-level knowledge links (batch)",
                total
            );
        }

        Ok(total)
    }

    // ================================================================
    // Single-Note Propagation (event-driven, called from link_note_to_entity)
    // ================================================================

    /// Propagate a single note linked to a FeatureGraph to its member files.
    pub async fn propagate_note_via_feature_graph(
        &self,
        note_id: Uuid,
        feature_graph_id: &str,
    ) -> Result<usize> {
        let q = query(
            r#"
            MATCH (n:Note {id: $note_id})
            MATCH (fg:FeatureGraph {id: $fg_id})-[:INCLUDES_ENTITY]->(f:File)
            WHERE NOT (n)-[:LINKED_TO]->(f)
            MERGE (n)-[r:LINKED_TO]->(f)
            ON CREATE SET
                r.propagated = true,
                r.propagation_source = 'feature_graph',
                r.propagation_depth = 1,
                r.last_verified = datetime()
            RETURN count(r) AS cnt
            "#,
        )
        .param("note_id", note_id.to_string())
        .param("fg_id", feature_graph_id.to_string());

        let mut total = 0i64;
        if let Ok(mut result) = self.graph.execute(q).await {
            if let Ok(Some(row)) = result.next().await {
                total = row.get::<i64>("cnt").unwrap_or(0);
            }
        }

        if total > 0 {
            tracing::info!(
                %note_id,
                feature_graph_id,
                propagated = total,
                "Event-driven: propagated note to {} files via FeatureGraph",
                total
            );
        }

        Ok(total as usize)
    }

    /// Propagate a single note linked to a Skill to files of the skill's member notes.
    pub async fn propagate_note_via_skill(&self, note_id: Uuid, skill_id: &str) -> Result<usize> {
        let q = query(
            r#"
            MATCH (n:Note {id: $note_id})
            MATCH (member:Note)-[:MEMBER_OF]->(s:Skill {id: $skill_id})
            MATCH (member)-[:LINKED_TO]->(f:File)
            WHERE NOT (n)-[:LINKED_TO]->(f) AND n <> member
            MERGE (n)-[r:LINKED_TO]->(f)
            ON CREATE SET
                r.propagated = true,
                r.propagation_source = 'skill_member',
                r.propagation_depth = 1,
                r.last_verified = datetime()
            RETURN count(r) AS cnt
            "#,
        )
        .param("note_id", note_id.to_string())
        .param("skill_id", skill_id.to_string());

        let mut total = 0i64;
        if let Ok(mut result) = self.graph.execute(q).await {
            if let Ok(Some(row)) = result.next().await {
                total = row.get::<i64>("cnt").unwrap_or(0);
            }
        }

        if total > 0 {
            tracing::info!(
                %note_id,
                skill_id,
                propagated = total,
                "Event-driven: propagated note to {} files via Skill members",
                total
            );
        }

        Ok(total as usize)
    }

    /// Propagate a single note linked to a Protocol to files via Protocol → Skill → MemberNotes.
    pub async fn propagate_note_via_protocol(
        &self,
        note_id: Uuid,
        protocol_id: &str,
    ) -> Result<usize> {
        let q = query(
            r#"
            MATCH (proto:Protocol {id: $proto_id})-[:BELONGS_TO_SKILL]->(s:Skill)
            MATCH (member:Note)-[:MEMBER_OF]->(s)
            MATCH (member)-[:LINKED_TO]->(f:File)
            MATCH (n:Note {id: $note_id})
            WHERE NOT (n)-[:LINKED_TO]->(f) AND n <> member
            MERGE (n)-[r:LINKED_TO]->(f)
            ON CREATE SET
                r.propagated = true,
                r.propagation_source = 'protocol_skill',
                r.propagation_depth = 1,
                r.last_verified = datetime()
            RETURN count(r) AS cnt
            "#,
        )
        .param("note_id", note_id.to_string())
        .param("proto_id", protocol_id.to_string());

        let mut total = 0i64;
        if let Ok(mut result) = self.graph.execute(q).await {
            if let Ok(Some(row)) = result.next().await {
                total = row.get::<i64>("cnt").unwrap_or(0);
            }
        }

        if total > 0 {
            tracing::info!(
                %note_id,
                protocol_id,
                propagated = total,
                "Event-driven: propagated note to {} files via Protocol→Skill",
                total
            );
        }

        Ok(total as usize)
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
            EntityType::Step => "Step",
            EntityType::Constraint => "Constraint",
            EntityType::Milestone => "Milestone",
            EntityType::Release => "Release",
            EntityType::FeatureGraph => "FeatureGraph",
            EntityType::Protocol => "Protocol",
            EntityType::ProtocolState => "ProtocolState",
            EntityType::ProtocolRun => "ProtocolRun",
            EntityType::PlanRun => "PlanRun",
            EntityType::Skill => "Skill",
            EntityType::Note => "Note",
            EntityType::ChatSession => "ChatSession",
            EntityType::Process => "Process",
        };

        let (match_field, match_value, use_suffix_match) = match entity_type {
            EntityType::File => {
                if entity_id.starts_with('/') {
                    ("path", entity_id.to_string(), false)
                } else {
                    ("path", format!("/{}", entity_id), true)
                }
            }
            EntityType::Commit => ("hash", entity_id.to_string(), false),
            _ => ("id", entity_id.to_string(), false),
        };

        let cypher = if use_suffix_match {
            format!(
                r#"
                MATCH (n:Note)-[:LINKED_TO]->(e:{})
                WHERE e.{} ENDS WITH $entity_id
                  AND n.status IN ['active', 'needs_review']
                RETURN n
                ORDER BY n.importance DESC, n.created_at DESC
                "#,
                node_label, match_field
            )
        } else {
            format!(
                r#"
                MATCH (n:Note)-[:LINKED_TO]->(e:{} {{{}: $entity_id}})
                WHERE n.status IN ['active', 'needs_review']
                RETURN n
                ORDER BY n.importance DESC, n.created_at DESC
                "#,
                node_label, match_field
            )
        };

        let q = query(&cypher).param("entity_id", match_value);

        let mut result = self.graph.execute(q).await?;
        let mut notes = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            notes.push(self.node_to_note(&node)?);
        }

        Ok(notes)
    }

    /// Compute pairwise coupling strength between two projects.
    /// Returns a value in [0.0, 1.0] based on 4 signals:
    /// structural twins, imported skills, shared notes, tag overlap.
    /// Used to weight cross-project note propagation (biomimicry P2P coupling).
    async fn get_pairwise_coupling(&self, project_a_id: Uuid, project_b_id: Uuid) -> Result<f64> {
        if project_a_id == project_b_id {
            return Ok(1.0);
        }

        const W_TWINS: f64 = 0.35;
        const W_SKILLS: f64 = 0.25;
        const W_NOTES: f64 = 0.20;
        const W_TAGS: f64 = 0.20;
        const MAX_TWINS: f64 = 20.0;
        const MAX_SKILLS: f64 = 10.0;
        const MAX_NOTES: f64 = 50.0;

        let a = project_a_id.to_string();
        let b = project_b_id.to_string();

        // Signal 1: Structural twins (files with same WL hash)
        let twins_q = query(
            r#"
            MATCH (pa:Project {id: $a})-[:CONTAINS]->(fa:File)
            MATCH (pb:Project {id: $b})-[:CONTAINS]->(fb:File)
            WHERE fa.wl_hash IS NOT NULL AND fa.wl_hash = fb.wl_hash
            RETURN count(DISTINCT fa) AS cnt
            "#,
        )
        .param("a", a.clone())
        .param("b", b.clone());

        let twins: f64 = match self.graph.execute(twins_q).await {
            Ok(mut r) => match r.next().await {
                Ok(Some(row)) => {
                    (row.get::<i64>("cnt").unwrap_or(0) as f64).min(MAX_TWINS) / MAX_TWINS
                }
                _ => 0.0,
            },
            _ => 0.0,
        };

        // Signal 2: Imported skills (skills exported from one, imported in the other)
        let skills_q = query(
            r#"
            MATCH (sa:Skill {project_id: $a})-[:IMPORTED_FROM]->(sb:Skill {project_id: $b})
            RETURN count(sa) AS cnt
            UNION ALL
            MATCH (sb2:Skill {project_id: $b})-[:IMPORTED_FROM]->(sa2:Skill {project_id: $a})
            RETURN count(sb2) AS cnt
            "#,
        )
        .param("a", a.clone())
        .param("b", b.clone());

        let skills: f64 = match self.graph.execute(skills_q).await {
            Ok(mut r) => {
                let mut total = 0i64;
                while let Ok(Some(row)) = r.next().await {
                    total += row.get::<i64>("cnt").unwrap_or(0);
                }
                (total as f64).min(MAX_SKILLS) / MAX_SKILLS
            }
            _ => 0.0,
        };

        // Signal 3: Shared notes (notes linked to entities in both projects)
        let notes_q = query(
            r#"
            MATCH (n:Note)-[:LINKED_TO]->(ea)<-[:CONTAINS]-(pa:Project {id: $a})
            MATCH (n)-[:LINKED_TO]->(eb)<-[:CONTAINS]-(pb:Project {id: $b})
            RETURN count(DISTINCT n) AS cnt
            "#,
        )
        .param("a", a.clone())
        .param("b", b.clone());

        let shared_notes: f64 = match self.graph.execute(notes_q).await {
            Ok(mut r) => match r.next().await {
                Ok(Some(row)) => {
                    (row.get::<i64>("cnt").unwrap_or(0) as f64).min(MAX_NOTES) / MAX_NOTES
                }
                _ => 0.0,
            },
            _ => 0.0,
        };

        // Signal 4: Tag overlap (Jaccard similarity of note tags)
        let tags_q = query(
            r#"
            MATCH (na:Note {project_id: $a}) WHERE na.tags IS NOT NULL
            WITH collect(na.tags) AS a_tags_raw
            WITH reduce(s = [], t IN a_tags_raw | s + t) AS a_tags_flat
            WITH apoc.coll.toSet(a_tags_flat) AS a_tags
            MATCH (nb:Note {project_id: $b}) WHERE nb.tags IS NOT NULL
            WITH a_tags, collect(nb.tags) AS b_tags_raw
            WITH a_tags, reduce(s = [], t IN b_tags_raw | s + t) AS b_tags_flat
            WITH a_tags, apoc.coll.toSet(b_tags_flat) AS b_tags
            WITH a_tags, b_tags,
                 [t IN a_tags WHERE t IN b_tags] AS intersection
            RETURN CASE WHEN size(a_tags) + size(b_tags) - size(intersection) = 0 THEN 0.0
                        ELSE toFloat(size(intersection)) / (size(a_tags) + size(b_tags) - size(intersection))
                   END AS jaccard
            "#,
        )
        .param("a", a)
        .param("b", b);

        let tag_overlap: f64 = match self.graph.execute(tags_q).await {
            Ok(mut r) => match r.next().await {
                Ok(Some(row)) => row.get::<f64>("jaccard").unwrap_or(0.0),
                _ => 0.0,
            },
            _ => 0.0,
        };

        let coupling =
            W_TWINS * twins + W_SKILLS * skills + W_NOTES * shared_notes + W_TAGS * tag_overlap;
        Ok(coupling)
    }

    /// Get propagated notes for an entity (traversing the graph)
    /// Whitelist of relation types allowed in propagation traversal.
    /// Prevents Cypher injection via user-supplied relation_types parameter.
    const ALLOWED_PROPAGATION_RELATIONS: &'static [&'static str] = &[
        "CONTAINS",
        "IMPORTS",
        "CALLS",
        "IMPLEMENTS_TRAIT",
        "IMPLEMENTS_FOR",
        "CO_CHANGED",
        "TOUCHES",
        "MODIFIES",
        "AFFECTS",
        "DISCUSSED",
        "SYNAPSE",
    ];

    /// Default relation types for backward-compatible propagation.
    const DEFAULT_PROPAGATION_RELATIONS: &'static [&'static str] =
        &["CONTAINS", "IMPORTS", "CALLS"];

    #[allow(clippy::too_many_arguments)]
    pub async fn get_propagated_notes(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
        max_depth: u32,
        min_score: f64,
        relation_types: Option<&[String]>,
        source_project_id: Option<Uuid>,
        force_cross_project: bool,
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
            EntityType::Step => "Step",
            EntityType::Constraint => "Constraint",
            EntityType::Milestone => "Milestone",
            EntityType::Release => "Release",
            EntityType::FeatureGraph => "FeatureGraph",
            EntityType::Protocol => "Protocol",
            EntityType::ProtocolState => "ProtocolState",
            EntityType::ProtocolRun => "ProtocolRun",
            EntityType::PlanRun => "PlanRun",
            EntityType::Skill => "Skill",
            EntityType::Note => "Note",
            EntityType::ChatSession => "ChatSession",
            EntityType::Process => "Process",
        };

        let (match_field, match_value, use_suffix_match) = match entity_type {
            EntityType::File => {
                if entity_id.starts_with('/') {
                    ("path", entity_id.to_string(), false)
                } else {
                    ("path", format!("/{}", entity_id), true)
                }
            }
            EntityType::Commit => ("hash", entity_id.to_string(), false),
            _ => ("id", entity_id.to_string(), false),
        };

        // Build the relation traversal pattern from the whitelist.
        // If relation_types is None, use the default (CONTAINS|IMPORTS|CALLS).
        // Only whitelisted relations are allowed to prevent Cypher injection.
        let rel_pattern = match relation_types {
            Some(types) => {
                let filtered: Vec<&str> = types
                    .iter()
                    .filter_map(|r| {
                        let upper = r.to_uppercase();
                        Self::ALLOWED_PROPAGATION_RELATIONS
                            .iter()
                            .find(|&&allowed| allowed == upper)
                            .copied()
                    })
                    .collect();
                if filtered.is_empty() {
                    Self::DEFAULT_PROPAGATION_RELATIONS.join("|")
                } else {
                    filtered.join("|")
                }
            }
            None => Self::DEFAULT_PROPAGATION_RELATIONS.join("|"),
        };

        // Build the target match clause. For relative file paths, use ENDS WITH
        // to match against the suffix of the absolute path stored in Neo4j.
        let target_match = if use_suffix_match {
            format!(
                "MATCH (target:{}) WHERE target.{} ENDS WITH $entity_id",
                node_label, match_field
            )
        } else {
            format!(
                "MATCH (target:{} {{{}: $entity_id}})",
                node_label, match_field
            )
        };

        // Query for notes propagated through the graph.
        //
        // Scoring formula integrates 5 factors:
        //   1. Distance decay: 1/(distance+1)
        //   2. Importance weight: critical=1.0, high=0.8, medium=0.5, low=0.3
        //   3. PageRank hub boost: (1 + avg_path_pagerank * 5)
        //   4. Relation type weight: product of per-relation weights along the path
        //      For SYNAPSE relations, uses the dynamic r.weight (Hebbian strength)
        //      instead of a static value.
        //   5. Scar penalty: (1 - scar_intensity * 0.5)
        //      Notes with high scar_intensity (from past invalidations) are de-prioritized.
        //      Max penalty = 50% score reduction at scar_intensity=1.0.
        //
        // Relation weights (defined in Cypher CASE):
        //   CONTAINS=1.0, IMPORTS=1.0, CALLS=0.9, IMPLEMENTS_TRAIT=0.85,
        //   IMPLEMENTS_FOR=0.85, AFFECTS=0.9, MODIFIES=0.7, TOUCHES=0.6,
        //   CO_CHANGED=0.6, DISCUSSED=0.5, SYNAPSE=dynamic r.weight, default=0.5
        let cypher = format!(
            r#"
            {}
            MATCH path = (n:Note)-[:LINKED_TO]->(source)-[:{}*0..{}]->(target)
            WHERE n.status = 'active'
            WITH n, source, path, length(path) - 1 AS distance,
                 [node IN nodes(path) | coalesce(node.name, node.path, node.id)] AS path_names,
                 [r IN relationships(path) | type(r)] AS rel_types,
                 [r IN tail(relationships(path)) | CASE type(r) WHEN 'SYNAPSE' THEN coalesce(r.weight, 0.5) ELSE null END] AS hop_weights
            WITH n, source, distance, path_names, rel_types, hop_weights,
                 CASE n.importance
                     WHEN 'critical' THEN 1.0
                     WHEN 'high' THEN 0.8
                     WHEN 'medium' THEN 0.5
                     ELSE 0.3
                 END AS importance_weight,
                 CASE WHEN size(nodes(path)) > 0
                      THEN reduce(s = 0.0, node IN nodes(path) | s + coalesce(node.pagerank, 0.05)) / size(nodes(path))
                      ELSE 0.05
                 END AS avg_path_pagerank,
                 CASE WHEN size(rel_types) <= 1 THEN 1.0
                      ELSE reduce(w = 1.0, r IN tail(relationships(path)) |
                          w * CASE type(r)
                              WHEN 'CONTAINS' THEN 1.0
                              WHEN 'IMPORTS' THEN 1.0
                              WHEN 'CALLS' THEN 0.9
                              WHEN 'IMPLEMENTS_TRAIT' THEN 0.85
                              WHEN 'IMPLEMENTS_FOR' THEN 0.85
                              WHEN 'AFFECTS' THEN 0.9
                              WHEN 'MODIFIES' THEN 0.7
                              WHEN 'TOUCHES' THEN 0.6
                              WHEN 'CO_CHANGED' THEN 0.6
                              WHEN 'DISCUSSED' THEN 0.5
                              WHEN 'SYNAPSE' THEN coalesce(r.weight, 0.5)
                              ELSE 0.5
                          END)
                 END AS path_rel_weight
            WITH n, source, distance, path_names, rel_types, avg_path_pagerank, path_rel_weight, hop_weights,
                 (1.0 / (distance + 1)) * importance_weight * (1.0 + avg_path_pagerank * 5.0) * path_rel_weight
                 * (1.0 - COALESCE(n.scar_intensity, 0.0) * 0.5) AS score
            WHERE score >= $min_score
            RETURN DISTINCT n, score, coalesce(source.name, source.path, source.id) AS source_entity,
                   path_names, distance, avg_path_pagerank,
                   tail(rel_types) AS relation_path, path_rel_weight, hop_weights
            ORDER BY score DESC
            LIMIT 20
            "#,
            target_match, rel_pattern, max_depth
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

            let rel_types: Vec<String> = row.get("relation_path").unwrap_or_default();
            let path_rel_weight: Option<f64> = row.get::<f64>("path_rel_weight").ok();
            // hop_weights: parallel array — null for structural, f64 for SYNAPSE
            let hop_weights: Vec<Option<f64>> = row
                .get::<Vec<f64>>("hop_weights")
                .map(|ws| {
                    ws.into_iter()
                        .map(|w| if w == 0.0 { None } else { Some(w) })
                        .collect()
                })
                .unwrap_or_else(|_| vec![None; rel_types.len()]);

            // Build RelationHop vec combining type + weight info
            let relation_path: Vec<crate::notes::RelationHop> = rel_types
                .iter()
                .zip(hop_weights.iter().chain(std::iter::repeat(&None)))
                .map(|(rt, w)| {
                    if rt == "SYNAPSE" {
                        crate::notes::RelationHop::neural(w.unwrap_or(0.5))
                    } else {
                        crate::notes::RelationHop::structural(rt.clone())
                    }
                })
                .collect();

            let scar_intensity = note.scar_intensity;
            propagated_notes.push(PropagatedNote {
                note,
                relevance_score: score,
                source_entity,
                propagation_path: path_names,
                distance: distance as u32,
                path_pagerank: avg_path_pagerank,
                relation_path,
                path_rel_weight,
                scar_intensity,
            });
        }

        // Cross-project coupling weighting (biomimicry P2P coupling)
        // If source_project_id is set, weight notes from other projects by coupling_strength.
        // Projects with coupling < 0.2 are suppressed unless force_cross_project is true.
        if let Some(src_pid) = source_project_id {
            let mut coupling_cache: std::collections::HashMap<Uuid, f64> =
                std::collections::HashMap::new();
            let mut filtered_notes = Vec::with_capacity(propagated_notes.len());

            for mut pn in propagated_notes {
                let note_pid = pn.note.project_id;
                match note_pid {
                    Some(pid) if pid != src_pid => {
                        // Cross-project note — look up coupling (cached per foreign project)
                        let coupling = match coupling_cache.get(&pid) {
                            Some(&c) => c,
                            None => {
                                let c = self
                                    .get_pairwise_coupling(src_pid, pid)
                                    .await
                                    .unwrap_or(0.0);
                                coupling_cache.insert(pid, c);
                                c
                            }
                        };

                        if coupling < 0.2 && !force_cross_project {
                            // Suppress low-coupling cross-project propagation
                            tracing::debug!(
                                note_id = %pn.note.id,
                                source_project = %src_pid,
                                note_project = %pid,
                                coupling,
                                "Suppressed cross-project note propagation (coupling < 0.2)"
                            );
                            continue;
                        }

                        // Weight the score by coupling strength
                        pn.relevance_score *= coupling;
                        if pn.relevance_score >= min_score {
                            filtered_notes.push(pn);
                        }
                    }
                    _ => {
                        // Same project or global note — keep as-is
                        filtered_notes.push(pn);
                    }
                }
            }

            // Re-sort after weighting
            filtered_notes.sort_by(|a, b| {
                b.relevance_score
                    .partial_cmp(&a.relevance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            return Ok(filtered_notes);
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
            MATCH (n:Note)-[:LINKED_TO]->(w)
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

            let scar_intensity = note.scar_intensity;
            workspace_notes.push(PropagatedNote {
                note,
                relevance_score: propagation_factor,
                source_entity: format!("workspace:{}", workspace_name),
                propagation_path: vec![format!("workspace:{}", workspace_name)],
                distance: 1, // One hop: project -> workspace
                path_pagerank: None,
                relation_path: vec![crate::notes::RelationHop::structural("BELONGS_TO")],
                path_rel_weight: Some(1.0),
                scar_intensity,
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
            MATCH (n:Note {id: $id})-[r:LINKED_TO]->(e)
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

    /// Get all note embeddings for a project in a single batch query.
    ///
    /// Returns notes that have embeddings, with their vector, metadata, and a content preview.
    /// Used for UMAP 2D projection in the visualization API.
    pub async fn get_note_embeddings_for_project(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<super::models::NoteEmbeddingPoint>> {
        let q = query(
            r#"
            MATCH (n:Note {project_id: $project_id})
            WHERE n.embedding IS NOT NULL
              AND n.status IN ['active', 'needs_review']
            RETURN n.id AS id,
                   n.embedding AS embedding,
                   n.note_type AS note_type,
                   n.importance AS importance,
                   coalesce(n.energy, 0.5) AS energy,
                   coalesce(n.tags, []) AS tags,
                   left(n.content, 120) AS content_preview
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut points = Vec::new();

        while let Some(row) = result.next().await? {
            let id_str: String = match row.get("id") {
                Ok(v) => v,
                Err(_) => continue,
            };
            let id = Uuid::parse_str(&id_str).unwrap_or_default();
            // Skip notes with corrupt/unparseable embeddings instead of crashing the whole endpoint
            let embedding_f64: Vec<f64> = match row.get("embedding") {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        note_id = %id_str,
                        "Skipping note with unparseable embedding: {}", e
                    );
                    continue;
                }
            };
            let embedding: Vec<f32> = embedding_f64.iter().map(|&x| x as f32).collect();
            let note_type: String = row
                .get("note_type")
                .unwrap_or_else(|_| "context".to_string());
            let importance: String = row
                .get("importance")
                .unwrap_or_else(|_| "medium".to_string());
            let energy: f64 = row.get("energy").unwrap_or(0.5);
            let tags: Vec<String> = row.get("tags").unwrap_or_default();
            let content_preview: String =
                row.get("content_preview").unwrap_or_else(|_| String::new());

            points.push(super::models::NoteEmbeddingPoint {
                id,
                embedding,
                note_type,
                importance,
                energy,
                tags,
                content_preview,
            });
        }

        Ok(points)
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
        min_similarity: Option<f64>,
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

            // Filter by minimum cosine similarity threshold
            if let Some(min_sim) = min_similarity {
                if score < min_sim {
                    continue;
                }
            }

            let note = self.node_to_note(&node)?;
            // Knowledge Scars: penalize scarred notes in search results
            // Biomimicry: Elun HypersphereIdentity.Scar — scarred knowledge is less trusted
            let adjusted_score = score * (1.0 - note.scar_intensity * 0.7);
            notes.push((note, adjusted_score));
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
    // Code embedding operations (File & Function vector search)
    // ========================================================================

    /// Store a vector embedding on a File node.
    pub async fn set_file_embedding(
        &self,
        file_path: &str,
        embedding: &[f32],
        model: &str,
    ) -> Result<()> {
        let embedding_f64: Vec<f64> = embedding.iter().map(|&x| x as f64).collect();

        let q = query(
            r#"
            MATCH (f:File {path: $path})
            CALL db.create.setNodeVectorProperty(f, 'embedding', $embedding)
            SET f.embedding_model = $model,
                f.embedded_at = datetime()
            "#,
        )
        .param("path", file_path.to_string())
        .param("embedding", embedding_f64)
        .param("model", model.to_string());

        self.graph
            .run(q)
            .await
            .context(format!("Failed to set embedding on file {}", file_path))?;

        Ok(())
    }

    /// Store a vector embedding on a Function node.
    pub async fn set_function_embedding(
        &self,
        function_name: &str,
        file_path: &str,
        embedding: &[f32],
        model: &str,
    ) -> Result<()> {
        let embedding_f64: Vec<f64> = embedding.iter().map(|&x| x as f64).collect();

        let q = query(
            r#"
            MATCH (fn:Function)<-[:CONTAINS]-(f:File {path: $file_path})
            WHERE fn.name = $name
            WITH fn LIMIT 1
            CALL db.create.setNodeVectorProperty(fn, 'embedding', $embedding)
            SET fn.embedding_model = $model,
                fn.embedded_at = datetime()
            "#,
        )
        .param("name", function_name.to_string())
        .param("file_path", file_path.to_string())
        .param("embedding", embedding_f64)
        .param("model", model.to_string());

        self.graph.run(q).await.context(format!(
            "Failed to set embedding on function {} in {}",
            function_name, file_path
        ))?;

        Ok(())
    }

    /// Search files by vector similarity using the HNSW index.
    pub async fn vector_search_files(
        &self,
        embedding: &[f32],
        limit: usize,
        project_id: Option<Uuid>,
    ) -> Result<Vec<(String, f64)>> {
        let embedding_f64: Vec<f64> = embedding.iter().map(|&x| x as f64).collect();
        let query_limit = (limit * 3) as i64; // over-fetch for post-filtering

        let cypher = if let Some(_pid) = project_id {
            r#"
            CALL db.index.vector.queryNodes('file_embedding_index', $query_limit, $embedding)
            YIELD node AS f, score
            WHERE EXISTS { MATCH (f)<-[:CONTAINS]-(p:Project {id: $project_id}) }
            RETURN f.path AS path, score
            ORDER BY score DESC
            LIMIT $limit
            "#
        } else {
            r#"
            CALL db.index.vector.queryNodes('file_embedding_index', $query_limit, $embedding)
            YIELD node AS f, score
            RETURN f.path AS path, score
            ORDER BY score DESC
            LIMIT $limit
            "#
        };

        let mut q = query(cypher)
            .param("embedding", embedding_f64)
            .param("query_limit", query_limit)
            .param("limit", limit as i64);

        if let Some(pid) = project_id {
            q = q.param("project_id", pid.to_string());
        }

        let mut result = self.graph.execute(q).await?;
        let mut results = Vec::new();

        while let Some(row) = result.next().await? {
            let path: String = row.get("path")?;
            let score: f64 = row.get("score")?;
            results.push((path, score));
        }

        Ok(results)
    }

    /// Search functions by vector similarity using the HNSW index.
    pub async fn vector_search_functions(
        &self,
        embedding: &[f32],
        limit: usize,
        project_id: Option<Uuid>,
    ) -> Result<Vec<(String, String, f64)>> {
        let embedding_f64: Vec<f64> = embedding.iter().map(|&x| x as f64).collect();
        let query_limit = (limit * 3) as i64;

        let cypher = if let Some(_pid) = project_id {
            r#"
            CALL db.index.vector.queryNodes('function_embedding_index', $query_limit, $embedding)
            YIELD node AS fn, score
            MATCH (fn)<-[:CONTAINS]-(f:File)
            WHERE EXISTS { MATCH (f)<-[:CONTAINS]-(p:Project {id: $project_id}) }
            RETURN fn.name AS name, f.path AS path, score
            ORDER BY score DESC
            LIMIT $limit
            "#
        } else {
            r#"
            CALL db.index.vector.queryNodes('function_embedding_index', $query_limit, $embedding)
            YIELD node AS fn, score
            MATCH (fn)<-[:CONTAINS]-(f:File)
            RETURN fn.name AS name, f.path AS path, score
            ORDER BY score DESC
            LIMIT $limit
            "#
        };

        let mut q = query(cypher)
            .param("embedding", embedding_f64)
            .param("query_limit", query_limit)
            .param("limit", limit as i64);

        if let Some(pid) = project_id {
            q = q.param("project_id", pid.to_string());
        }

        let mut result = self.graph.execute(q).await?;
        let mut results = Vec::new();

        while let Some(row) = result.next().await? {
            let name: String = row.get("name")?;
            let path: String = row.get("path")?;
            let score: f64 = row.get("score")?;
            results.push((name, path, score));
        }

        Ok(results)
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
    // Cross-entity SYNAPSE operations (Decision ↔ Note)
    // ========================================================================

    /// Create bidirectional SYNAPSE relationships between any two nodes (Note or Decision).
    ///
    /// Unlike `create_synapses` which is Note-specific, this method matches nodes
    /// by their `id` property regardless of label. This enables Decision↔Note
    /// and Decision↔Decision synapses for cross-entity neural linking.
    ///
    /// Uses MERGE for idempotence. Returns the number of synapses created.
    pub async fn create_cross_entity_synapses(
        &self,
        source_id: Uuid,
        neighbors: &[(Uuid, f64)],
    ) -> Result<usize> {
        if neighbors.is_empty() {
            return Ok(0);
        }

        // Build UNWIND list (internal computed data, no injection risk)
        let entries: Vec<String> = neighbors
            .iter()
            .map(|(nid, weight)| format!("{{id: '{}', weight: {}}}", nid, weight))
            .collect();

        let cypher = format!(
            r#"
            MATCH (source {{id: $source_id}})
            WHERE source:Note OR source:Decision
            UNWIND [{}] AS neighbor
            MATCH (target {{id: neighbor.id}})
            WHERE target:Note OR target:Decision
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

        let q = query(&cypher).param("source_id", source_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let total: i64 = row.get("total")?;
            Ok(total as usize)
        } else {
            Ok(0)
        }
    }

    /// Get all SYNAPSE relationships for any node (Note or Decision), both directions.
    ///
    /// Returns (neighbor_id, weight, entity_type) where entity_type is "Note" or "Decision".
    /// Sorted by weight descending.
    pub async fn get_cross_entity_synapses(
        &self,
        node_id: Uuid,
    ) -> Result<Vec<(Uuid, f64, String)>> {
        let cypher = r#"
            MATCH (n {id: $id})-[s:SYNAPSE]-(neighbor)
            WHERE (n:Note OR n:Decision) AND (neighbor:Note OR neighbor:Decision)
            RETURN DISTINCT neighbor.id AS neighbor_id, s.weight AS weight,
                   CASE WHEN neighbor:Decision THEN 'Decision' ELSE 'Note' END AS entity_type
            ORDER BY weight DESC
        "#;

        let q = query(cypher).param("id", node_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut synapses = Vec::new();

        while let Some(row) = result.next().await? {
            let neighbor_id_str: String = row.get("neighbor_id")?;
            let weight: f64 = row.get("weight")?;
            let entity_type: String = row.get("entity_type")?;
            if let Ok(nid) = neighbor_id_str.parse::<Uuid>() {
                synapses.push((nid, weight, entity_type));
            }
        }

        Ok(synapses)
    }

    /// List Decision nodes that have embeddings but no SYNAPSE relationships.
    ///
    /// Used by `backfill_synapses` to process Decision nodes for cross-entity linking.
    pub async fn list_decisions_needing_synapses(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<DecisionNode>, usize)> {
        // Count total
        let count_q = query(
            r#"
            MATCH (d:Decision)
            WHERE d.embedding IS NOT NULL AND NOT (d)-[:SYNAPSE]->()
            RETURN count(d) AS total
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
            MATCH (d:Decision)
            WHERE d.embedding IS NOT NULL AND NOT (d)-[:SYNAPSE]->()
            RETURN d
            ORDER BY d.decided_at
            SKIP $offset LIMIT $limit
            "#,
        )
        .param("offset", offset as i64)
        .param("limit", limit as i64);

        let mut result = self.graph.execute(fetch_q).await?;
        let mut decisions = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            decisions.push(Self::node_to_decision(&node)?);
        }

        Ok((decisions, total))
    }

    // ========================================================================
    // Energy operations (Phase 2 — Neural Network)
    // ========================================================================

    /// Retrieve all SYNAPSE weights for a project (or globally when `project_id` is None).
    ///
    /// Used to calibrate adaptive thresholds from the real weight distribution.
    pub async fn get_all_synapse_weights(
        &self,
        project_id: Option<uuid::Uuid>,
    ) -> Result<Vec<f64>> {
        let (cypher, q) = if let Some(pid) = project_id {
            let cypher = r#"
                MATCH (p:Project {id: $project_id})-[:HAS_NOTE]->(n1:Note)-[s:SYNAPSE]->(n2:Note)
                RETURN s.weight AS weight
            "#;
            (cypher, query(cypher).param("project_id", pid.to_string()))
        } else {
            let cypher = r#"
                MATCH ()-[s:SYNAPSE]->()
                RETURN s.weight AS weight
            "#;
            (cypher, query(cypher))
        };
        let _ = cypher; // silence unused warning

        let mut result = self.graph.execute(q).await?;
        let mut weights = Vec::new();
        while let Some(row) = result.next().await? {
            if let Ok(w) = row.get::<f64>("weight") {
                weights.push(w);
            }
        }
        Ok(weights)
    }

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
    ///
    /// For every pair (i, j) in note_ids, MERGE a bidirectional SYNAPSE.
    /// Uses a single UNWIND query to batch all pairs instead of N*(N-1)/2
    /// individual queries, reducing 45 queries (10 notes) to 1.
    ///
    /// Performance improvement:
    /// - Before: 10 notes → 45 individual Cypher queries (~500ms)
    /// - After:  10 notes → 1 UNWIND query (~20ms)
    pub async fn reinforce_synapses(&self, note_ids: &[Uuid], boost: f64) -> Result<usize> {
        use super::batch::{bolt_map, run_unwind_in_chunks_with, BoltMap};

        if note_ids.len() < 2 {
            return Ok(0);
        }

        // Build all unique pairs as BoltMaps for UNWIND
        let mut pairs: Vec<BoltMap> = Vec::new();
        for i in 0..note_ids.len() {
            for j in (i + 1)..note_ids.len() {
                pairs.push(bolt_map(&[
                    ("a", note_ids[i].to_string().into()),
                    ("b", note_ids[j].to_string().into()),
                ]));
            }
        }

        let pair_count = pairs.len();

        run_unwind_in_chunks_with(
            &self.graph,
            pairs,
            r#"
            UNWIND $items AS pair
            MATCH (a:Note {id: pair.a}), (b:Note {id: pair.b})
            MERGE (a)-[s1:SYNAPSE]->(b)
              ON CREATE SET s1.weight = 0.5, s1.created_at = datetime(),
                s1.last_reinforced_at = datetime(), s1.reinforcement_count = 1
              ON MATCH SET s1.weight = CASE
                  WHEN s1.weight + $boost > 1.0 THEN 1.0
                  ELSE s1.weight + $boost
              END,
                s1.last_reinforced_at = datetime(),
                s1.reinforcement_count = coalesce(s1.reinforcement_count, 0) + 1
            MERGE (b)-[s2:SYNAPSE]->(a)
              ON CREATE SET s2.weight = 0.5, s2.created_at = datetime(),
                s2.last_reinforced_at = datetime(), s2.reinforcement_count = 1
              ON MATCH SET s2.weight = CASE
                  WHEN s2.weight + $boost > 1.0 THEN 1.0
                  ELSE s2.weight + $boost
              END,
                s2.last_reinforced_at = datetime(),
                s2.reinforcement_count = coalesce(s2.reinforcement_count, 0) + 1
            "#,
            |q| q.param("boost", boost),
        )
        .await?;

        // Each pair creates/updates 2 synapses (bidirectional)
        Ok(pair_count * 2)
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

        // Step 2: Prune weak synapses (2-step: count first, then delete)
        // NOTE: Neo4j RETURN count() after DELETE always returns 0,
        // so we must count before deleting.
        let count_q = query(
            r#"
            MATCH ()-[s:SYNAPSE]->()
            WHERE s.weight < $threshold
            RETURN count(s) AS pruned
            "#,
        )
        .param("threshold", prune_threshold);

        let mut result = self.graph.execute(count_q).await?;
        let pruned = if let Some(row) = result.next().await? {
            row.get::<i64>("pruned").unwrap_or(0) as usize
        } else {
            0
        };

        if pruned > 0 {
            let delete_q = query(
                r#"
                MATCH ()-[s:SYNAPSE]->()
                WHERE s.weight < $threshold
                DELETE s
                "#,
            )
            .param("threshold", prune_threshold);
            self.graph.run(delete_q).await?;
        }

        // Step 3: Decay knowledge scars (20x slower than synapse decay)
        // Biomimicry: Elun Scar — scars heal very slowly, allowing the system to remember failures
        let scar_decay_rate = decay_amount * 0.05; // 1/20th of synapse decay
        if scar_decay_rate > 0.0 {
            let scar_decay_q = query(
                r#"
                MATCH (n)
                WHERE n.scar_intensity IS NOT NULL AND n.scar_intensity > 0
                AND (n:Note OR n:Decision)
                SET n.scar_intensity = CASE
                    WHEN n.scar_intensity - $scar_decay < 0.001 THEN 0.0
                    ELSE n.scar_intensity - $scar_decay
                END
                RETURN count(n) AS healed
                "#,
            )
            .param("scar_decay", scar_decay_rate);
            if let Ok(mut result) = self.graph.execute(scar_decay_q).await {
                if let Ok(Some(row)) = result.next().await {
                    let healed: i64 = row.get("healed").unwrap_or(0);
                    if healed > 0 {
                        tracing::debug!(
                            "Decayed scars on {} nodes (rate: {:.4})",
                            healed,
                            scar_decay_rate
                        );
                    }
                }
            }
        }

        Ok((decayed, pruned))
    }

    /// Apply scars to nodes traversed during a failed reasoning path.
    ///
    /// Biomimicry: Elun HypersphereIdentity.Scar — nodes that led to failure
    /// receive a scar that penalizes their score in future searches.
    /// Scar increment is +0.2 per failure, capped at 1.0.
    ///
    /// Works on both Note and Decision nodes.
    pub async fn apply_scars(&self, node_ids: &[Uuid], increment: f64) -> Result<usize> {
        if node_ids.is_empty() {
            return Ok(0);
        }

        let ids: Vec<String> = node_ids.iter().map(|id| id.to_string()).collect();

        // Apply scars on Note nodes
        let note_q = query(
            r#"
            UNWIND $ids AS nid
            MATCH (n:Note {id: nid})
            SET n.scar_intensity = CASE
                WHEN coalesce(n.scar_intensity, 0.0) + $increment > 1.0 THEN 1.0
                ELSE coalesce(n.scar_intensity, 0.0) + $increment
            END
            RETURN count(n) AS scarred
            "#,
        )
        .param("ids", ids.clone())
        .param("increment", increment);

        let mut result = self.graph.execute(note_q).await?;
        let note_scarred = if let Some(row) = result.next().await? {
            row.get::<i64>("scarred").unwrap_or(0) as usize
        } else {
            0
        };

        // Apply scars on Decision nodes
        let decision_q = query(
            r#"
            UNWIND $ids AS nid
            MATCH (d:Decision {id: nid})
            SET d.scar_intensity = CASE
                WHEN coalesce(d.scar_intensity, 0.0) + $increment > 1.0 THEN 1.0
                ELSE coalesce(d.scar_intensity, 0.0) + $increment
            END
            RETURN count(d) AS scarred
            "#,
        )
        .param("ids", ids)
        .param("increment", increment);

        let mut result = self.graph.execute(decision_q).await?;
        let decision_scarred = if let Some(row) = result.next().await? {
            row.get::<i64>("scarred").unwrap_or(0) as usize
        } else {
            0
        };

        Ok(note_scarred + decision_scarred)
    }

    /// Heal (reset) scars on a specific node.
    ///
    /// Sets scar_intensity back to 0.0 for the given note or decision.
    /// Used for manual scar removal via `admin(action: "heal_scars")`.
    pub async fn heal_scars(&self, node_id: Uuid) -> Result<bool> {
        let id_str = node_id.to_string();

        // Try Note first
        let note_q = query(
            r#"
            MATCH (n:Note {id: $id})
            SET n.scar_intensity = 0.0
            RETURN count(n) AS healed
            "#,
        )
        .param("id", id_str.clone());

        let mut result = self.graph.execute(note_q).await?;
        let note_healed = if let Some(row) = result.next().await? {
            row.get::<i64>("healed").unwrap_or(0) > 0
        } else {
            false
        };

        if note_healed {
            return Ok(true);
        }

        // Try Decision
        let decision_q = query(
            r#"
            MATCH (d:Decision {id: $id})
            SET d.scar_intensity = 0.0
            RETURN count(d) AS healed
            "#,
        )
        .param("id", id_str);

        let mut result = self.graph.execute(decision_q).await?;
        let decision_healed = if let Some(row) = result.next().await? {
            row.get::<i64>("healed").unwrap_or(0) > 0
        } else {
            false
        };

        Ok(decision_healed)
    }

    /// Consolidate memory: promote eligible notes and archive stale ephemeral ones.
    /// Returns (promoted_count, archived_count).
    pub async fn consolidate_memory(&self) -> Result<(usize, usize)> {
        use crate::notes::lifecycle::NoteLifecycleManager;

        let lifecycle = NoteLifecycleManager::new();
        let now = chrono::Utc::now();

        // 1. Fetch all non-consolidated, active notes
        let q = query(
            r#"
            MATCH (n:Note)
            WHERE n.status = 'active'
              AND (n.memory_horizon IS NULL OR n.memory_horizon <> 'consolidated')
            OPTIONAL MATCH (n)-[s:SYNAPSE]-()
            RETURN n, count(s) AS activation_count
            ORDER BY n.energy DESC
            LIMIT 500
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut promoted = 0usize;
        let mut archived = 0usize;

        while let Some(row) = result.next().await? {
            let node = row.get::<neo4rs::Node>("n")?;
            let note = self.node_to_note(&node)?;
            let activation_count = row.get::<i64>("activation_count").unwrap_or(0) as u64;

            // Check archival for ephemeral
            if lifecycle.should_archive_ephemeral(&note, now) {
                let archive_q = query(
                    r#"
                    MATCH (n:Note {id: $id})
                    SET n.status = 'archived',
                        n.changes_json = $changes_json
                    "#,
                )
                .param("id", note.id.to_string())
                .param(
                    "changes_json",
                    serde_json::to_string(&{
                        let mut changes = note.changes.clone();
                        changes.push(NoteChange::with_details(
                            crate::notes::ChangeType::StatusChanged,
                            "consolidate_memory".to_string(),
                            serde_json::json!({"reason": "ephemeral_expired", "idle_hours": 48}),
                        ));
                        changes
                    })?,
                );
                self.graph.run(archive_q).await?;
                archived += 1;
                continue;
            }

            // Evaluate promotion
            let promo = lifecycle.evaluate_promotion(&note, activation_count);
            if let Some(new_horizon) = promo.new_horizon {
                let promote_q = query(
                    r#"
                    MATCH (n:Note {id: $id})
                    SET n.memory_horizon = $new_horizon,
                        n.changes_json = $changes_json
                    "#,
                )
                .param("id", note.id.to_string())
                .param("new_horizon", new_horizon.to_string())
                .param(
                    "changes_json",
                    serde_json::to_string(&{
                        let mut changes = note.changes.clone();
                        changes.push(NoteChange::with_details(
                            crate::notes::ChangeType::Promoted,
                            "consolidate_memory".to_string(),
                            serde_json::json!({
                                "from": promo.current_horizon.to_string(),
                                "to": new_horizon.to_string(),
                                "reason": promo.reason
                            }),
                        ));
                        changes
                    })?,
                );
                self.graph.run(promote_q).await?;
                promoted += 1;
            }
        }

        Ok((promoted, archived))
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
    pub(crate) fn node_to_note(&self, node: &neo4rs::Node) -> Result<Note> {
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
            scar_intensity: node.get("scar_intensity").unwrap_or(0.0),
            memory_horizon: node
                .get::<String>("memory_horizon")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(MemoryHorizon::Consolidated), // Existing notes default to consolidated
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
    // Graph visualization batch queries
    // ========================================================================

    /// Get all LINKED_TO edges from notes in a project to code entities (batch).
    /// Returns (note_id, entity_type_label, entity_id).
    pub async fn get_project_note_entity_links(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String, String)>> {
        let q = query(
            "MATCH (n:Note {project_id: $pid})-[:LINKED_TO]->(e)
             RETURN n.id AS note_id,
                    CASE
                      WHEN e:File THEN 'file'
                      WHEN e:Function THEN 'function'
                      WHEN e:Struct THEN 'struct'
                      WHEN e:Trait THEN 'trait'
                      WHEN e:Enum THEN 'enum'
                      ELSE 'unknown'
                    END AS entity_type,
                    COALESCE(e.path, e.id, toString(elementId(e))) AS entity_id",
        )
        .param("pid", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut links = Vec::new();
        while let Some(row) = result.next().await? {
            let note_id: String = row.get("note_id").unwrap_or_default();
            let entity_type: String = row.get("entity_type").unwrap_or_default();
            let entity_id: String = row.get("entity_id").unwrap_or_default();
            if !note_id.is_empty() && !entity_id.is_empty() {
                links.push((note_id, entity_type, entity_id));
            }
        }
        Ok(links)
    }

    /// Get all SYNAPSE edges between notes in a project (batch, deduplicated).
    /// Returns (source_note_id, target_note_id, weight).
    pub async fn get_project_note_synapses(
        &self,
        project_id: Uuid,
        min_weight: f64,
    ) -> Result<Vec<(String, String, f64)>> {
        let q = query(
            "MATCH (a:Note {project_id: $pid})-[s:SYNAPSE]->(b:Note {project_id: $pid})
             WHERE s.weight >= $min_weight AND a.id < b.id
             RETURN a.id AS source_id, b.id AS target_id, s.weight AS weight
             ORDER BY s.weight DESC",
        )
        .param("pid", project_id.to_string())
        .param("min_weight", min_weight);

        let mut result = self.graph.execute(q).await?;
        let mut synapses = Vec::new();
        while let Some(row) = result.next().await? {
            let source: String = row.get("source_id").unwrap_or_default();
            let target: String = row.get("target_id").unwrap_or_default();
            let weight: f64 = row.get("weight").unwrap_or(0.0);
            if !source.is_empty() && !target.is_empty() {
                synapses.push((source, target, weight));
            }
        }
        Ok(synapses)
    }
}
