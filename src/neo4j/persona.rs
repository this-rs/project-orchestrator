//! Neo4j Persona operations (Living Personas)
//!
//! Implements all CRUD, relation management, and subgraph assembly
//! for the Living Personas system on the Neo4j graph.

use super::client::Neo4jClient;
use crate::neo4j::models::{
    PersonaNode, PersonaStatus, PersonaSubgraph, PersonaSubgraphStats, PersonaWeightedRelation,
};
use anyhow::{Context, Result};
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Conversion helpers
    // ========================================================================

    /// Convert a Neo4j node to a [`PersonaNode`].
    pub(crate) fn node_to_persona(node: &neo4rs::Node) -> Result<PersonaNode> {
        Ok(PersonaNode {
            id: node.get::<String>("id")?.parse()?,
            project_id: node
                .get::<String>("project_id")
                .ok()
                .and_then(|s| s.parse().ok()),
            name: node.get("name")?,
            description: node.get("description").unwrap_or_default(),
            status: node
                .get::<String>("status")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_default(),
            complexity_default: node
                .get::<String>("complexity_default")
                .ok()
                .filter(|s| !s.is_empty()),
            timeout_secs: node.get::<i64>("timeout_secs").ok().map(|v| v as u64),
            max_cost_usd: node.get("max_cost_usd").ok(),
            model_preference: node
                .get::<String>("model_preference")
                .ok()
                .filter(|s| !s.is_empty()),
            system_prompt_override: node
                .get::<String>("system_prompt_override")
                .ok()
                .filter(|s| !s.is_empty()),
            energy: node.get("energy").unwrap_or(0.5),
            cohesion: node.get("cohesion").unwrap_or(0.0),
            activation_count: node.get("activation_count").unwrap_or(0),
            success_rate: node.get("success_rate").unwrap_or(0.0),
            avg_duration_secs: node.get("avg_duration_secs").unwrap_or(0.0),
            last_activated: node
                .get::<String>("last_activated")
                .ok()
                .and_then(|s| s.parse().ok()),
            origin: node
                .get::<String>("origin")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_default(),
            created_at: node.get::<String>("created_at")?.parse()?,
            updated_at: node
                .get::<String>("updated_at")
                .ok()
                .and_then(|s| s.parse().ok()),
        })
    }

    // ========================================================================
    // CRUD
    // ========================================================================

    /// Create a new Persona node. If project_id is Some, links via BELONGS_TO.
    pub async fn create_persona(&self, persona: &PersonaNode) -> Result<()> {
        let q = query(
            r#"
            CREATE (p:Persona {
                id: $id,
                name: $name,
                description: $description,
                status: $status,
                complexity_default: $complexity_default,
                timeout_secs: $timeout_secs,
                max_cost_usd: $max_cost_usd,
                model_preference: $model_preference,
                system_prompt_override: $system_prompt_override,
                energy: $energy,
                cohesion: $cohesion,
                activation_count: $activation_count,
                success_rate: $success_rate,
                avg_duration_secs: $avg_duration_secs,
                origin: $origin,
                created_at: $created_at
            })
            WITH p
            OPTIONAL MATCH (proj:Project {id: $project_id})
            FOREACH (_ IN CASE WHEN proj IS NOT NULL THEN [1] ELSE [] END |
                SET p.project_id = $project_id
                CREATE (p)-[:BELONGS_TO]->(proj)
            )
            "#,
        )
        .param("id", persona.id.to_string())
        .param("name", persona.name.clone())
        .param("description", persona.description.clone())
        .param("status", persona.status.to_string())
        .param(
            "complexity_default",
            persona
                .complexity_default
                .clone()
                .map(neo4rs::BoltType::from)
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param(
            "timeout_secs",
            persona
                .timeout_secs
                .map(|v| neo4rs::BoltType::from(v as i64))
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param(
            "max_cost_usd",
            persona
                .max_cost_usd
                .map(neo4rs::BoltType::from)
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param(
            "model_preference",
            persona
                .model_preference
                .clone()
                .map(neo4rs::BoltType::from)
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param(
            "system_prompt_override",
            persona
                .system_prompt_override
                .clone()
                .map(neo4rs::BoltType::from)
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param("energy", persona.energy)
        .param("cohesion", persona.cohesion)
        .param("activation_count", persona.activation_count)
        .param("success_rate", persona.success_rate)
        .param("avg_duration_secs", persona.avg_duration_secs)
        .param("origin", persona.origin.to_string())
        .param("created_at", persona.created_at.to_rfc3339())
        .param(
            "project_id",
            persona
                .project_id
                .map(|id| neo4rs::BoltType::from(id.to_string()))
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        );

        let _ = self.graph.execute(q).await.context("create_persona")?;
        Ok(())
    }

    /// Get a persona by ID.
    pub async fn get_persona(&self, id: Uuid) -> Result<Option<PersonaNode>> {
        let q = query("MATCH (p:Persona {id: $id}) RETURN p").param("id", id.to_string());

        let mut result = self.graph.execute(q).await.context("get_persona")?;

        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            Ok(Some(Self::node_to_persona(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Update a persona node (replaces all mutable fields).
    pub async fn update_persona(&self, persona: &PersonaNode) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $id})
            SET p.name = $name,
                p.description = $description,
                p.status = $status,
                p.complexity_default = $complexity_default,
                p.timeout_secs = $timeout_secs,
                p.max_cost_usd = $max_cost_usd,
                p.model_preference = $model_preference,
                p.system_prompt_override = $system_prompt_override,
                p.energy = $energy,
                p.cohesion = $cohesion,
                p.activation_count = $activation_count,
                p.success_rate = $success_rate,
                p.avg_duration_secs = $avg_duration_secs,
                p.last_activated = $last_activated,
                p.updated_at = $updated_at
            "#,
        )
        .param("id", persona.id.to_string())
        .param("name", persona.name.clone())
        .param("description", persona.description.clone())
        .param("status", persona.status.to_string())
        .param(
            "complexity_default",
            persona
                .complexity_default
                .clone()
                .map(neo4rs::BoltType::from)
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param(
            "timeout_secs",
            persona
                .timeout_secs
                .map(|v| neo4rs::BoltType::from(v as i64))
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param(
            "max_cost_usd",
            persona
                .max_cost_usd
                .map(neo4rs::BoltType::from)
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param(
            "model_preference",
            persona
                .model_preference
                .clone()
                .map(neo4rs::BoltType::from)
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param(
            "system_prompt_override",
            persona
                .system_prompt_override
                .clone()
                .map(neo4rs::BoltType::from)
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param("energy", persona.energy)
        .param("cohesion", persona.cohesion)
        .param("activation_count", persona.activation_count)
        .param("success_rate", persona.success_rate)
        .param("avg_duration_secs", persona.avg_duration_secs)
        .param(
            "last_activated",
            persona
                .last_activated
                .map(|d| neo4rs::BoltType::from(d.to_rfc3339()))
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param("updated_at", chrono::Utc::now().to_rfc3339());

        let _ = self.graph.execute(q).await.context("update_persona")?;
        Ok(())
    }

    /// Delete a persona and all its relationships.
    pub async fn delete_persona(&self, id: Uuid) -> Result<bool> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $id})
            DETACH DELETE p
            RETURN count(p) AS deleted
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await.context("delete_persona")?;

        if let Some(row) = result.next().await? {
            let deleted: i64 = row.get("deleted")?;
            Ok(deleted > 0)
        } else {
            Ok(false)
        }
    }

    /// List personas for a project with optional status filter and pagination.
    pub async fn list_personas(
        &self,
        project_id: Uuid,
        status: Option<PersonaStatus>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<PersonaNode>, usize)> {
        let pid = project_id.to_string();

        // Build query depending on status filter
        let (count_cypher, data_cypher) = if status.is_some() {
            (
                "MATCH (p:Persona) WHERE p.project_id = $project_id AND p.status = $status RETURN count(p) AS total",
                "MATCH (p:Persona) WHERE p.project_id = $project_id AND p.status = $status RETURN p ORDER BY p.energy DESC SKIP $offset LIMIT $limit",
            )
        } else {
            (
                "MATCH (p:Persona) WHERE p.project_id = $project_id RETURN count(p) AS total",
                "MATCH (p:Persona) WHERE p.project_id = $project_id RETURN p ORDER BY p.energy DESC SKIP $offset LIMIT $limit",
            )
        };

        // Count
        let mut count_q = query(count_cypher).param("project_id", pid.clone());
        if let Some(s) = &status {
            count_q = count_q.param("status", s.to_string());
        }
        let mut count_result = self
            .graph
            .execute(count_q)
            .await
            .context("list_personas count")?;
        let total: usize = if let Some(row) = count_result.next().await? {
            row.get::<i64>("total")? as usize
        } else {
            0
        };

        // Fetch
        let mut data_q = query(data_cypher)
            .param("project_id", pid)
            .param("offset", offset as i64)
            .param("limit", limit as i64);
        if let Some(s) = &status {
            data_q = data_q.param("status", s.to_string());
        }
        let mut result = self
            .graph
            .execute(data_q)
            .await
            .context("list_personas fetch")?;

        let mut personas = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            personas.push(Self::node_to_persona(&node)?);
        }

        Ok((personas, total))
    }

    /// List global personas (project_id IS NULL).
    pub async fn list_global_personas(&self) -> Result<Vec<PersonaNode>> {
        let q = query(
            "MATCH (p:Persona) WHERE p.project_id IS NULL RETURN p ORDER BY p.energy DESC",
        );
        let mut result = self
            .graph
            .execute(q)
            .await
            .context("list_global_personas")?;

        let mut personas = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            personas.push(Self::node_to_persona(&node)?);
        }
        Ok(personas)
    }

    // ========================================================================
    // Relation operations
    // ========================================================================

    /// Add MASTERS relation: Persona -> Skill (MERGE for idempotence)
    pub async fn add_persona_skill(&self, persona_id: Uuid, skill_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id}), (s:Skill {id: $skill_id})
            MERGE (p)-[:MASTERS]->(s)
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("skill_id", skill_id.to_string());

        let _ = self.graph.execute(q).await.context("add_persona_skill")?;
        Ok(())
    }

    /// Remove MASTERS relation: Persona -> Skill
    pub async fn remove_persona_skill(&self, persona_id: Uuid, skill_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id})-[r:MASTERS]->(s:Skill {id: $skill_id})
            DELETE r
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("skill_id", skill_id.to_string());

        let _ = self.graph
            .execute(q)
            .await
            .context("remove_persona_skill")?;
        Ok(())
    }

    /// Add FOLLOWS relation: Persona -> Protocol
    pub async fn add_persona_protocol(&self, persona_id: Uuid, protocol_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id}), (pr:Protocol {id: $protocol_id})
            MERGE (p)-[:FOLLOWS]->(pr)
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("protocol_id", protocol_id.to_string());

        let _ = self.graph
            .execute(q)
            .await
            .context("add_persona_protocol")?;
        Ok(())
    }

    /// Remove FOLLOWS relation: Persona -> Protocol
    pub async fn remove_persona_protocol(
        &self,
        persona_id: Uuid,
        protocol_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id})-[r:FOLLOWS]->(pr:Protocol {id: $protocol_id})
            DELETE r
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("protocol_id", protocol_id.to_string());

        let _ = self.graph
            .execute(q)
            .await
            .context("remove_persona_protocol")?;
        Ok(())
    }

    /// Set SCOPED_TO relation: Persona -> FeatureGraph (replaces existing)
    pub async fn set_persona_feature_graph(
        &self,
        persona_id: Uuid,
        feature_graph_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id})
            OPTIONAL MATCH (p)-[old:SCOPED_TO]->()
            DELETE old
            WITH p
            MATCH (fg:FeatureGraph {id: $fg_id})
            CREATE (p)-[:SCOPED_TO]->(fg)
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("fg_id", feature_graph_id.to_string());

        let _ = self.graph
            .execute(q)
            .await
            .context("set_persona_feature_graph")?;
        Ok(())
    }

    /// Remove SCOPED_TO relation from a Persona
    pub async fn remove_persona_feature_graph(&self, persona_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id})-[r:SCOPED_TO]->()
            DELETE r
            "#,
        )
        .param("persona_id", persona_id.to_string());

        let _ = self.graph
            .execute(q)
            .await
            .context("remove_persona_feature_graph")?;
        Ok(())
    }

    /// Add KNOWS relation: Persona -> File (with weight, MERGE + SET)
    pub async fn add_persona_file(
        &self,
        persona_id: Uuid,
        file_path: &str,
        weight: f64,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id}), (f:File {path: $file_path})
            MERGE (p)-[r:KNOWS]->(f)
            SET r.weight = $weight
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("file_path", file_path)
        .param("weight", weight);

        let _ = self.graph.execute(q).await.context("add_persona_file")?;
        Ok(())
    }

    /// Remove KNOWS relation: Persona -> File
    pub async fn remove_persona_file(&self, persona_id: Uuid, file_path: &str) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id})-[r:KNOWS]->(f:File {path: $file_path})
            DELETE r
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("file_path", file_path);

        let _ = self.graph
            .execute(q)
            .await
            .context("remove_persona_file")?;
        Ok(())
    }

    /// Add KNOWS relation: Persona -> Function (with weight)
    pub async fn add_persona_function(
        &self,
        persona_id: Uuid,
        function_id: &str,
        weight: f64,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id}), (fn:Function {id: $function_id})
            MERGE (p)-[r:KNOWS]->(fn)
            SET r.weight = $weight
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("function_id", function_id)
        .param("weight", weight);

        let _ = self.graph
            .execute(q)
            .await
            .context("add_persona_function")?;
        Ok(())
    }

    /// Remove KNOWS relation: Persona -> Function
    pub async fn remove_persona_function(
        &self,
        persona_id: Uuid,
        function_id: &str,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id})-[r:KNOWS]->(fn:Function {id: $function_id})
            DELETE r
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("function_id", function_id);

        let _ = self.graph
            .execute(q)
            .await
            .context("remove_persona_function")?;
        Ok(())
    }

    /// Add USES relation: Persona -> Note (with weight)
    pub async fn add_persona_note(
        &self,
        persona_id: Uuid,
        note_id: Uuid,
        weight: f64,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id}), (n:Note {id: $note_id})
            MERGE (p)-[r:USES]->(n)
            SET r.weight = $weight
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("note_id", note_id.to_string())
        .param("weight", weight);

        let _ = self.graph.execute(q).await.context("add_persona_note")?;
        Ok(())
    }

    /// Remove USES relation: Persona -> Note
    pub async fn remove_persona_note(&self, persona_id: Uuid, note_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id})-[r:USES]->(n:Note {id: $note_id})
            DELETE r
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("note_id", note_id.to_string());

        let _ = self.graph
            .execute(q)
            .await
            .context("remove_persona_note")?;
        Ok(())
    }

    /// Add USES relation: Persona -> Decision (with weight)
    pub async fn add_persona_decision(
        &self,
        persona_id: Uuid,
        decision_id: Uuid,
        weight: f64,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id}), (d:Decision {id: $decision_id})
            MERGE (p)-[r:USES]->(d)
            SET r.weight = $weight
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("decision_id", decision_id.to_string())
        .param("weight", weight);

        let _ = self.graph
            .execute(q)
            .await
            .context("add_persona_decision")?;
        Ok(())
    }

    /// Remove USES relation: Persona -> Decision
    pub async fn remove_persona_decision(
        &self,
        persona_id: Uuid,
        decision_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id})-[r:USES]->(d:Decision {id: $decision_id})
            DELETE r
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("decision_id", decision_id.to_string());

        let _ = self.graph
            .execute(q)
            .await
            .context("remove_persona_decision")?;
        Ok(())
    }

    /// Add EXTENDS relation: child Persona -> parent Persona
    pub async fn add_persona_extends(&self, child_id: Uuid, parent_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (c:Persona {id: $child_id}), (p:Persona {id: $parent_id})
            MERGE (c)-[:EXTENDS]->(p)
            "#,
        )
        .param("child_id", child_id.to_string())
        .param("parent_id", parent_id.to_string());

        let _ = self.graph
            .execute(q)
            .await
            .context("add_persona_extends")?;
        Ok(())
    }

    /// Remove EXTENDS relation: child Persona -> parent Persona
    pub async fn remove_persona_extends(&self, child_id: Uuid, parent_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (c:Persona {id: $child_id})-[r:EXTENDS]->(p:Persona {id: $parent_id})
            DELETE r
            "#,
        )
        .param("child_id", child_id.to_string())
        .param("parent_id", parent_id.to_string());

        let _ = self.graph
            .execute(q)
            .await
            .context("remove_persona_extends")?;
        Ok(())
    }

    // ========================================================================
    // Subgraph assembly
    // ========================================================================

    /// Assemble the full knowledge subgraph for a persona.
    pub async fn get_persona_subgraph(&self, persona_id: Uuid) -> Result<PersonaSubgraph> {
        let pid = persona_id.to_string();

        // Get persona basic info
        let persona = self
            .get_persona(persona_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Persona {} not found", persona_id))?;

        // KNOWS -> File (with weight)
        let files = {
            let q = query(
                r#"
                MATCH (p:Persona {id: $pid})-[r:KNOWS]->(f:File)
                RETURN f.path AS entity_id, r.weight AS weight
                ORDER BY r.weight DESC
                "#,
            )
            .param("pid", pid.clone());
            self.collect_persona_weighted_rels(q, "file").await?
        };

        // KNOWS -> Function (with weight)
        let functions = {
            let q = query(
                r#"
                MATCH (p:Persona {id: $pid})-[r:KNOWS]->(fn:Function)
                RETURN fn.id AS entity_id, r.weight AS weight
                ORDER BY r.weight DESC
                "#,
            )
            .param("pid", pid.clone());
            self.collect_persona_weighted_rels(q, "function").await?
        };

        // USES -> Note (with weight)
        let notes = {
            let q = query(
                r#"
                MATCH (p:Persona {id: $pid})-[r:USES]->(n:Note)
                RETURN n.id AS entity_id, r.weight AS weight
                ORDER BY r.weight DESC
                "#,
            )
            .param("pid", pid.clone());
            self.collect_persona_weighted_rels(q, "note").await?
        };

        // USES -> Decision (with weight)
        let decisions = {
            let q = query(
                r#"
                MATCH (p:Persona {id: $pid})-[r:USES]->(d:Decision)
                RETURN d.id AS entity_id, r.weight AS weight
                ORDER BY r.weight DESC
                "#,
            )
            .param("pid", pid.clone());
            self.collect_persona_weighted_rels(q, "decision").await?
        };

        // MASTERS -> Skill (ids only)
        let skill_ids = {
            let q = query(
                r#"
                MATCH (p:Persona {id: $pid})-[:MASTERS]->(s:Skill)
                RETURN s.id AS id
                "#,
            )
            .param("pid", pid.clone());
            self.collect_persona_rel_ids(q).await?
        };

        // FOLLOWS -> Protocol (ids only)
        let protocol_ids = {
            let q = query(
                r#"
                MATCH (p:Persona {id: $pid})-[:FOLLOWS]->(pr:Protocol)
                RETURN pr.id AS id
                "#,
            )
            .param("pid", pid.clone());
            self.collect_persona_rel_ids(q).await?
        };

        // SCOPED_TO -> FeatureGraph (at most one)
        let feature_graph_id = {
            let q = query(
                r#"
                MATCH (p:Persona {id: $pid})-[:SCOPED_TO]->(fg:FeatureGraph)
                RETURN fg.id AS id LIMIT 1
                "#,
            )
            .param("pid", pid.clone());
            let mut result = self.graph.execute(q).await?;
            if let Some(row) = result.next().await? {
                let id_str: String = row.get("id")?;
                Some(id_str.parse()?)
            } else {
                None
            }
        };

        // EXTENDS chain (transitive parents)
        let parent_ids = {
            let q = query(
                r#"
                MATCH path = (p:Persona {id: $pid})-[:EXTENDS*1..10]->(ancestor:Persona)
                WITH ancestor, length(path) AS depth
                ORDER BY depth ASC
                RETURN DISTINCT ancestor.id AS id
                "#,
            )
            .param("pid", pid);
            self.collect_persona_rel_ids(q).await?
        };

        let total_entities = files.len()
            + functions.len()
            + notes.len()
            + decisions.len()
            + skill_ids.len()
            + protocol_ids.len()
            + if feature_graph_id.is_some() { 1 } else { 0 };

        Ok(PersonaSubgraph {
            persona_id,
            persona_name: persona.name,
            files,
            functions,
            notes,
            decisions,
            skill_ids,
            protocol_ids,
            feature_graph_id,
            parent_ids,
            stats: PersonaSubgraphStats {
                total_entities,
                coverage_score: 0.0, // Computed by GDS later
                freshness: 0.0,      // Computed by maintenance later
            },
        })
    }

    /// Helper: collect weighted relations into Vec<PersonaWeightedRelation>.
    async fn collect_persona_weighted_rels(
        &self,
        q: neo4rs::Query,
        entity_type: &str,
    ) -> Result<Vec<PersonaWeightedRelation>> {
        let mut result = self.graph.execute(q).await?;
        let mut rels = Vec::new();
        while let Some(row) = result.next().await? {
            let entity_id: String = row.get("entity_id")?;
            let weight: f64 = row.get("weight").unwrap_or(1.0);
            rels.push(PersonaWeightedRelation {
                entity_type: entity_type.to_string(),
                entity_id,
                weight,
            });
        }
        Ok(rels)
    }

    /// Helper: collect IDs from a query returning `id` column.
    async fn collect_persona_rel_ids(&self, q: neo4rs::Query) -> Result<Vec<Uuid>> {
        let mut result = self.graph.execute(q).await?;
        let mut ids = Vec::new();
        while let Some(row) = result.next().await? {
            let id_str: String = row.get("id")?;
            ids.push(id_str.parse()?);
        }
        Ok(ids)
    }

    /// Find personas that KNOW a given file (for activation by file match).
    pub async fn find_personas_for_file(
        &self,
        file_path: &str,
        project_id: Uuid,
    ) -> Result<Vec<(PersonaNode, f64)>> {
        let q = query(
            r#"
            MATCH (p:Persona {project_id: $project_id})-[r:KNOWS]->(f:File {path: $file_path})
            WHERE p.status <> 'archived'
            RETURN p, r.weight AS weight
            ORDER BY r.weight DESC
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("file_path", file_path);

        let mut result = self
            .graph
            .execute(q)
            .await
            .context("find_personas_for_file")?;

        let mut personas = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            let weight: f64 = row.get("weight").unwrap_or(1.0);
            personas.push((Self::node_to_persona(&node)?, weight));
        }
        Ok(personas)
    }
}
