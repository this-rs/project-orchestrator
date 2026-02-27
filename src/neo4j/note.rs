//! Neo4j Knowledge Note operations

use super::client::Neo4jClient;
use super::models::DecisionNode;
use crate::notes::{
    EntityType, Note, NoteAnchor, NoteChange, NoteFilters, NoteImportance, NoteScope, NoteStatus,
    NoteType, PropagatedNote,
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
            EntityType::Step => "Step",
            EntityType::Constraint => "Constraint",
            EntityType::Milestone => "Milestone",
            EntityType::Release => "Release",
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
            EntityType::Step => "Step",
            EntityType::Constraint => "Constraint",
            EntityType::Milestone => "Milestone",
            EntityType::Release => "Release",
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
            EntityType::Step => "Step",
            EntityType::Constraint => "Constraint",
            EntityType::Milestone => "Milestone",
            EntityType::Release => "Release",
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

    pub async fn get_propagated_notes(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
        max_depth: u32,
        min_score: f64,
        relation_types: Option<&[String]>,
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
        };

        let (match_field, match_value) = match entity_type {
            EntityType::File => ("path", entity_id.to_string()),
            EntityType::Commit => ("hash", entity_id.to_string()),
            _ => ("id", entity_id.to_string()),
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

        // Query for notes propagated through the graph.
        //
        // Scoring formula integrates 4 factors:
        //   1. Distance decay: 1/(distance+1)
        //   2. Importance weight: critical=1.0, high=0.8, medium=0.5, low=0.3
        //   3. PageRank hub boost: (1 + avg_path_pagerank * 5)
        //   4. Relation type weight: product of per-relation weights along the path
        //      For SYNAPSE relations, uses the dynamic r.weight (Hebbian strength)
        //      instead of a static value.
        //
        // Relation weights (defined in Cypher CASE):
        //   CONTAINS=1.0, IMPORTS=1.0, CALLS=0.9, IMPLEMENTS_TRAIT=0.85,
        //   IMPLEMENTS_FOR=0.85, AFFECTS=0.9, MODIFIES=0.7, TOUCHES=0.6,
        //   CO_CHANGED=0.6, DISCUSSED=0.5, SYNAPSE=dynamic r.weight, default=0.5
        let cypher = format!(
            r#"
            MATCH (target:{} {{{}: $entity_id}})
            MATCH path = (n:Note)-[:ATTACHED_TO]->(source)-[:{}*0..{}]->(target)
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
                 (1.0 / (distance + 1)) * importance_weight * (1.0 + avg_path_pagerank * 5.0) * path_rel_weight AS score
            WHERE score >= $min_score
            RETURN DISTINCT n, score, coalesce(source.name, source.path, source.id) AS source_entity,
                   path_names, distance, avg_path_pagerank,
                   tail(rel_types) AS relation_path, path_rel_weight, hop_weights
            ORDER BY score DESC
            LIMIT 20
            "#,
            node_label, match_field, rel_pattern, max_depth
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

            propagated_notes.push(PropagatedNote {
                note,
                relevance_score: score,
                source_entity,
                propagation_path: path_names,
                distance: distance as u32,
                path_pagerank: avg_path_pagerank,
                relation_path,
                path_rel_weight,
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
                relation_path: vec![crate::notes::RelationHop::structural("BELONGS_TO")],
                path_rel_weight: Some(1.0),
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
}
