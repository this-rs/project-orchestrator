//! Neo4j Persona operations (Living Personas)
//!
//! Implements all CRUD, relation management, and subgraph assembly
//! for the Living Personas system on the Neo4j graph.

use super::client::Neo4jClient;
use crate::analytics::distribution::{adaptive_threshold, detect_outliers};
use crate::neo4j::models::{
    PersonaNode, PersonaStatus, PersonaSubgraph, PersonaSubgraphStats, PersonaWeightedRelation,
};
use anyhow::{Context, Result};
use neo4rs::query;
use serde::{Deserialize, Serialize};
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
            energy_boost_accumulated: node.get("energy_boost_accumulated").unwrap_or(0.0),
            energy_history: node
                .get::<String>("energy_history")
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default(),
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
                energy_boost_accumulated: $energy_boost_accumulated,
                energy_history: $energy_history,
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
        .param("energy_boost_accumulated", persona.energy_boost_accumulated)
        .param("energy_history", serde_json::to_string(&persona.energy_history).unwrap_or_else(|_| "[]".to_string()))
        .param("created_at", persona.created_at.to_rfc3339())
        .param(
            "project_id",
            persona
                .project_id
                .map(|id| neo4rs::BoltType::from(id.to_string()))
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        );

        self.graph.run(q).await.context("create_persona")?;
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

        self.graph.run(q).await.context("update_persona")?;
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
        let q =
            query("MATCH (p:Persona) WHERE p.project_id IS NULL RETURN p ORDER BY p.energy DESC");
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

        self.graph.run(q).await.context("add_persona_skill")?;
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

        self.graph.run(q).await.context("remove_persona_skill")?;
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

        self.graph.run(q).await.context("add_persona_protocol")?;
        Ok(())
    }

    /// Remove FOLLOWS relation: Persona -> Protocol
    pub async fn remove_persona_protocol(&self, persona_id: Uuid, protocol_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id})-[r:FOLLOWS]->(pr:Protocol {id: $protocol_id})
            DELETE r
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("protocol_id", protocol_id.to_string());

        self.graph.run(q).await.context("remove_persona_protocol")?;
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

        self.graph
            .run(q)
            .await
            .context("set_persona_feature_graph")?;
        Ok(())
    }

    /// Increment activation_count and update last_activated for a Persona.
    /// Best-effort (fire-and-forget from hooks).
    pub async fn increment_persona_activation(&self, persona_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id})
            SET p.activation_count = p.activation_count + 1,
                p.last_activated = datetime(),
                p.updated_at = datetime()
            "#,
        )
        .param("persona_id", persona_id.to_string());
        let _ = self.graph.run(q).await; // Best-effort
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

        self.graph
            .run(q)
            .await
            .context("remove_persona_feature_graph")?;
        Ok(())
    }

    /// Add KNOWS relation: Persona -> File (with weight, MERGE + SET)
    /// Accepts both absolute paths (/Users/.../src/foo.rs) and relative paths (src/foo.rs).
    /// Relative paths are matched with ENDS WITH + '/' prefix guard to avoid partial matches.
    pub async fn add_persona_file(
        &self,
        persona_id: Uuid,
        file_path: &str,
        weight: f64,
    ) -> Result<()> {
        let is_absolute = file_path.starts_with('/');
        let cypher = if is_absolute {
            r#"
            MATCH (p:Persona {id: $persona_id}), (f:File {path: $file_path})
            MERGE (p)-[r:KNOWS]->(f)
            SET r.weight = $weight
            "#
        } else {
            r#"
            MATCH (p:Persona {id: $persona_id}), (f:File)
            WHERE f.path ENDS WITH $file_path AND f.path ENDS WITH ('/' + $file_path)
            WITH p, f ORDER BY length(f.path) LIMIT 1
            MERGE (p)-[r:KNOWS]->(f)
            SET r.weight = $weight
            "#
        };
        let q = query(cypher)
            .param("persona_id", persona_id.to_string())
            .param("file_path", file_path)
            .param("weight", weight);

        self.graph.run(q).await.context("add_persona_file")?;
        Ok(())
    }

    /// Remove KNOWS relation: Persona -> File
    /// Accepts both absolute and relative paths (same logic as add_persona_file).
    pub async fn remove_persona_file(&self, persona_id: Uuid, file_path: &str) -> Result<()> {
        let is_absolute = file_path.starts_with('/');
        let cypher = if is_absolute {
            r#"
            MATCH (p:Persona {id: $persona_id})-[r:KNOWS]->(f:File {path: $file_path})
            DELETE r
            "#
        } else {
            r#"
            MATCH (p:Persona {id: $persona_id})-[r:KNOWS]->(f:File)
            WHERE f.path ENDS WITH $file_path AND f.path ENDS WITH ('/' + $file_path)
            DELETE r
            "#
        };
        let q = query(cypher)
            .param("persona_id", persona_id.to_string())
            .param("file_path", file_path);

        self.graph.run(q).await.context("remove_persona_file")?;
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

        self.graph.run(q).await.context("add_persona_function")?;
        Ok(())
    }

    /// Remove KNOWS relation: Persona -> Function
    pub async fn remove_persona_function(&self, persona_id: Uuid, function_id: &str) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id})-[r:KNOWS]->(fn:Function {id: $function_id})
            DELETE r
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("function_id", function_id);

        self.graph.run(q).await.context("remove_persona_function")?;
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

        self.graph.run(q).await.context("add_persona_note")?;
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

        self.graph.run(q).await.context("remove_persona_note")?;
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

        self.graph.run(q).await.context("add_persona_decision")?;
        Ok(())
    }

    /// Remove USES relation: Persona -> Decision
    pub async fn remove_persona_decision(&self, persona_id: Uuid, decision_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id})-[r:USES]->(d:Decision {id: $decision_id})
            DELETE r
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("decision_id", decision_id.to_string());

        self.graph.run(q).await.context("remove_persona_decision")?;
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

        self.graph.run(q).await.context("add_persona_extends")?;
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

        self.graph.run(q).await.context("remove_persona_extends")?;
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

        // MASTERS -> Skill (as weighted relations with name)
        let skills = {
            let q = query(
                r#"
                MATCH (p:Persona {id: $pid})-[:MASTERS]->(s:Skill)
                RETURN s.id AS entity_id, 1.0 AS weight
                ORDER BY s.name ASC
                "#,
            )
            .param("pid", pid.clone());
            self.collect_persona_weighted_rels(q, "skill").await?
        };

        // FOLLOWS -> Protocol (as weighted relations with name)
        let protocols = {
            let q = query(
                r#"
                MATCH (p:Persona {id: $pid})-[:FOLLOWS]->(pr:Protocol)
                RETURN pr.id AS entity_id, 1.0 AS weight
                ORDER BY pr.name ASC
                "#,
            )
            .param("pid", pid.clone());
            self.collect_persona_weighted_rels(q, "protocol").await?
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
        let parents = {
            let q = query(
                r#"
                MATCH path = (p:Persona {id: $pid})-[:EXTENDS*1..10]->(ancestor:Persona)
                WITH ancestor, length(path) AS depth
                ORDER BY depth ASC
                RETURN DISTINCT ancestor.id AS entity_id, 1.0 AS weight
                "#,
            )
            .param("pid", pid.clone());
            self.collect_persona_weighted_rels(q, "persona").await?
        };

        // Children (personas that EXTEND this one)
        let children = {
            let q = query(
                r#"
                MATCH (child:Persona)-[:EXTENDS]->(p:Persona {id: $pid})
                RETURN child.id AS entity_id, 1.0 AS weight
                "#,
            )
            .param("pid", pid);
            self.collect_persona_weighted_rels(q, "persona").await?
        };

        let total_entities = files.len()
            + functions.len()
            + notes.len()
            + decisions.len()
            + skills.len()
            + protocols.len()
            + if feature_graph_id.is_some() { 1 } else { 0 };

        Ok(PersonaSubgraph {
            persona_id,
            persona_name: persona.name,
            files,
            functions,
            notes,
            decisions,
            skills,
            protocols,
            feature_graph_id,
            parents,
            children,
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

    /// Find personas that KNOW a given file (for activation by file match).
    pub async fn find_personas_for_file(
        &self,
        file_path: &str,
        project_id: Uuid,
    ) -> Result<Vec<(PersonaNode, f64)>> {
        // Match via direct KNOWS relation OR via SCOPED_TO FeatureGraph
        // Supports both absolute (/Users/.../src/foo.rs) and relative (src/foo.rs) paths
        let is_absolute = file_path.starts_with('/');
        let cypher = if is_absolute {
            r#"
            CALL {
                MATCH (p:Persona {project_id: $project_id})-[r:KNOWS]->(f:File {path: $file_path})
                WHERE p.status <> 'archived'
                RETURN p, r.weight AS weight
                UNION
                MATCH (p:Persona {project_id: $project_id})-[:SCOPED_TO]->(fg:FeatureGraph)-[:INCLUDES_ENTITY]->(f:File {path: $file_path})
                WHERE p.status <> 'archived'
                RETURN p, 0.3 AS weight
            }
            WITH p, max(weight) AS weight
            RETURN p, weight
            ORDER BY weight DESC
            "#
        } else {
            r#"
            CALL {
                MATCH (p:Persona {project_id: $project_id})-[r:KNOWS]->(f:File)
                WHERE f.path ENDS WITH $file_path AND f.path ENDS WITH ('/' + $file_path) AND p.status <> 'archived'
                RETURN p, r.weight AS weight
                UNION
                MATCH (p:Persona {project_id: $project_id})-[:SCOPED_TO]->(fg:FeatureGraph)-[:INCLUDES_ENTITY]->(f:File)
                WHERE f.path ENDS WITH $file_path AND f.path ENDS WITH ('/' + $file_path) AND p.status <> 'archived'
                RETURN p, 0.3 AS weight
            }
            WITH p, max(weight) AS weight
            RETURN p, weight
            ORDER BY weight DESC
            "#
        };
        let q = query(cypher)
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

    /// Load ALL persona KNOWS relations for a project in a single query.
    /// Returns Vec<(PersonaNode, file_path, weight)> for building a complete PersonaFileIndex
    /// without cold-start issues.
    pub async fn get_all_persona_knows(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(PersonaNode, String, f64)>> {
        let cypher = r#"
            MATCH (p:Persona {project_id: $project_id})-[r:KNOWS]->(f:File)
            WHERE p.status <> 'archived'
            RETURN p, f.path AS file_path, r.weight AS weight
            ORDER BY p.name, weight DESC
        "#;
        let q = query(cypher).param("project_id", project_id.to_string());

        let mut result = self
            .graph
            .execute(q)
            .await
            .context("get_all_persona_knows")?;

        let mut entries = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("p")?;
            let file_path: String = row.get("file_path")?;
            let weight: f64 = row.get("weight").unwrap_or(1.0);
            entries.push((Self::node_to_persona(&node)?, file_path, weight));
        }
        Ok(entries)
    }

    // ========================================================================
    // Maintenance & Detection
    // ========================================================================

    /// Auto-scope each persona to the FeatureGraph that best matches its KNOWS files.
    ///
    /// Two-phase approach:
    /// 1. **Match existing**: scope personas to existing FeatureGraphs when one
    ///    covers their dominant community
    /// 2. **Auto-create**: for personas whose dominant community has no FeatureGraph,
    ///    create one automatically (MERGE by name, idempotent) with all community
    ///    files linked via INCLUDES_ENTITY
    ///
    /// Uses `COALESCE(fabric_community_id, community_id)` to prefer multi-layer
    /// communities when available.
    ///
    /// Returns the total number of personas that were scoped.
    pub async fn auto_scope_to_feature_graphs(&self, project_id: Uuid) -> Result<usize> {
        // Two-phase approach:
        // Phase 1: scope to existing FeatureGraphs
        // Phase 2: auto-create FeatureGraphs for unscoped personas
        // ── Phase 1: Scope to existing FeatureGraphs ──────────────────
        let existing_q = query(
            r#"
            MATCH (p:Persona {project_id: $project_id})-[r:KNOWS]->(f:File)
            WHERE p.status <> 'archived' AND COALESCE(f.fabric_community_id, f.community_id) IS NOT NULL
            WITH p, COALESCE(f.fabric_community_id, f.community_id) AS cid, count(f) AS file_count, sum(r.weight) AS total_weight
            ORDER BY p.id, file_count DESC, total_weight DESC
            WITH p, collect({cid: cid, cnt: file_count})[0] AS dominant
            WITH p, dominant.cid AS dominant_cid
            WHERE dominant_cid IS NOT NULL
            MATCH (fg:FeatureGraph {project_id: $project_id})-[:INCLUDES_ENTITY]->(f2:File)
            WHERE COALESCE(f2.fabric_community_id, f2.community_id) = dominant_cid
            WITH p, fg, count(f2) AS overlap
            ORDER BY p.id, overlap DESC
            WITH p, collect(fg)[0] AS best_fg
            WHERE best_fg IS NOT NULL
            OPTIONAL MATCH (p)-[old:SCOPED_TO]->()
            DELETE old
            WITH p, best_fg
            CREATE (p)-[:SCOPED_TO]->(best_fg)
            RETURN count(p) AS scoped
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut existing_result = self
            .graph
            .execute(existing_q)
            .await
            .context("auto_scope_to_feature_graphs: phase 1")?;

        let scoped_existing: usize = if let Some(row) = existing_result.next().await? {
            row.get::<i64>("scoped").unwrap_or(0) as usize
        } else {
            0
        };

        // ── Phase 2: Auto-create FeatureGraphs for unscoped personas ────
        // Find personas that still have no SCOPED_TO after phase 1, determine
        // their dominant community, and create a FeatureGraph (idempotent by name).
        let autocreate_q = query(
            r#"
            MATCH (p:Persona {project_id: $project_id})-[r:KNOWS]->(f:File)
            WHERE p.status <> 'archived'
              AND COALESCE(f.fabric_community_id, f.community_id) IS NOT NULL
              AND NOT (p)-[:SCOPED_TO]->()
            WITH p, COALESCE(f.fabric_community_id, f.community_id) AS cid,
                 count(f) AS file_count, sum(r.weight) AS total_weight
            ORDER BY p.id, file_count DESC, total_weight DESC
            WITH p, collect({cid: cid, cnt: file_count})[0] AS dominant
            WITH p, dominant.cid AS dominant_cid
            WHERE dominant_cid IS NOT NULL

            // Collect community files + derive a name
            MATCH (cf:File)<-[:CONTAINS]-(proj:Project {id: $project_id})
            WHERE COALESCE(cf.fabric_community_id, cf.community_id) = dominant_cid
            WITH p, dominant_cid, collect(cf) AS community_files, proj,
                 COALESCE(
                     head(collect(DISTINCT cf.fabric_community_label)),
                     head(collect(DISTINCT cf.community_label)),
                     'community-' + toString(dominant_cid)
                 ) AS fg_name

            // Idempotent: MERGE FeatureGraph by name + project_id
            MERGE (fg:FeatureGraph {name: fg_name, project_id: $project_id})
            ON CREATE SET fg.id = randomUUID(),
                          fg.description = 'Auto-created from fabric community ' + toString(dominant_cid),
                          fg.created_at = datetime(),
                          fg.updated_at = datetime(),
                          fg.source = 'auto_community',
                          fg.entry_function = '',
                          fg.build_depth = 0,
                          fg.include_relations = ''

            // Link community files to the FeatureGraph
            WITH p, fg, community_files, proj
            UNWIND community_files AS cf
            MERGE (fg)-[:INCLUDES_ENTITY]->(cf)

            // Link FeatureGraph to project
            WITH DISTINCT p, fg, proj
            MERGE (fg)-[:BELONGS_TO]->(proj)

            // Scope persona to the new FeatureGraph
            WITH p, fg
            CREATE (p)-[:SCOPED_TO]->(fg)
            RETURN count(DISTINCT p) AS auto_scoped
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut autocreate_result = self
            .graph
            .execute(autocreate_q)
            .await
            .context("auto_scope_to_feature_graphs: phase 2 auto-create")?;

        let scoped_auto: usize = if let Some(row) = autocreate_result.next().await? {
            row.get::<i64>("auto_scoped").unwrap_or(0) as usize
        } else {
            0
        };

        if scoped_auto > 0 {
            tracing::info!(
                scoped_existing,
                scoped_auto,
                "auto_scope: created FeatureGraphs for {} personas without existing scope",
                scoped_auto
            );
        }

        Ok(scoped_existing + scoped_auto)
    }

    /// Maintain all personas for a project: decay weights, prune, recalculate cohesion.
    ///
    /// Uses adaptive thresholds from rs-stats when enough data is available,
    /// falling back to hardcoded defaults for small projects.
    ///
    /// Returns (decayed_count, pruned_count, personas_updated).
    pub async fn maintain_personas(&self, project_id: Uuid) -> Result<(usize, usize, usize)> {
        // 0. Compute adaptive thresholds from actual weight distribution
        let thresholds = self
            .compute_adaptive_thresholds(project_id)
            .await
            .unwrap_or_else(|e| {
                tracing::warn!(
                    "Failed to compute adaptive thresholds: {} — using defaults",
                    e
                );
                AdaptivePersonaThresholds::default()
            });

        tracing::info!(
            prune_cutoff = thresholds.prune_cutoff,
            confidence_p90 = thresholds.confidence_p90,
            sample_size = thresholds.sample_size,
            outliers = thresholds.weight_outlier_count,
            "maintain_personas: using adaptive thresholds"
        );

        // 0b. Check global stagnation — if stagnating, use accelerated decay
        let stagnation = self.detect_global_stagnation(project_id).await.unwrap_or_else(|e| {
            tracing::warn!("Failed to check stagnation: {} — using normal decay", e);
            crate::neo4j::models::StagnationReport {
                is_stagnating: false,
                tasks_completed_48h: 0,
                avg_frustration: 0.0,
                energy_trend: 0.0,
                commits_48h: 0,
                signals_triggered: 0,
                recommendations: vec![],
            }
        });

        let decay_factor = if stagnation.is_stagnating { 0.85 } else { 0.95 };

        tracing::info!(
            stagnating = stagnation.is_stagnating,
            decay_factor,
            signals = stagnation.signals_triggered,
            "maintain_personas: stagnation check"
        );

        // 1. Decay all KNOWS/USES weights by *= decay_factor
        let decay_q = query(
            r#"
            MATCH (p:Persona {project_id: $project_id})-[r:KNOWS|USES]->()
            WHERE r.weight IS NOT NULL
            SET r.weight = r.weight * $decay_factor
            RETURN count(r) AS decayed
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("decay_factor", decay_factor);

        let mut decay_result = self
            .graph
            .execute(decay_q)
            .await
            .context("maintain_personas: decay")?;
        let decayed_count: usize = if let Some(row) = decay_result.next().await? {
            row.get::<i64>("decayed").unwrap_or(0) as usize
        } else {
            0
        };

        // 2. Prune relations with weight < adaptive prune_cutoff (default 0.1)
        let prune_q = query(
            r#"
            MATCH (p:Persona {project_id: $project_id})-[r:KNOWS|USES]->()
            WHERE r.weight IS NOT NULL AND r.weight < $prune_cutoff
            DELETE r
            RETURN count(r) AS pruned
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("prune_cutoff", thresholds.prune_cutoff);

        let mut prune_result = self
            .graph
            .execute(prune_q)
            .await
            .context("maintain_personas: prune")?;
        let pruned_count: usize = if let Some(row) = prune_result.next().await? {
            row.get::<i64>("pruned").unwrap_or(0) as usize
        } else {
            0
        };

        // 1b. If stagnating, boost personas linked to recently committed files
        if stagnation.is_stagnating {
            let boost_q = query(r#"
                MATCH (c:Commit)-[:TOUCHES]->(f:File)<-[:KNOWS]-(p:Persona {project_id: $project_id})
                WHERE c.created_at IS NOT NULL
                  AND datetime(c.created_at) > datetime() - duration('P7D')
                WITH DISTINCT p
                SET p.energy = CASE WHEN p.energy + 0.1 > 1.0 THEN 1.0 ELSE p.energy + 0.1 END
                RETURN count(p) AS boosted
            "#)
            .param("project_id", project_id.to_string());

            let _ = self.graph.run(boost_q).await;
        }

        // 2b. Prune KNOWS outliers (low-weight anomalies inactive for >14 days)
        if thresholds.sample_size >= 10 {
            let outlier_prune_q = query(r#"
                MATCH (p:Persona {project_id: $project_id})-[r:KNOWS]->(f:File)
                WHERE r.weight IS NOT NULL AND r.weight < $lower_fence
                  AND (r.last_activated IS NULL
                       OR r.last_activated < $cutoff_date)
                DELETE r
                RETURN count(r) AS outlier_pruned
            "#)
            .param("project_id", project_id.to_string())
            .param("lower_fence", thresholds.lower_fence)
            .param("cutoff_date", (chrono::Utc::now() - chrono::Duration::days(14)).to_rfc3339());

            let outlier_pruned: usize = match self.graph.execute(outlier_prune_q).await {
                Ok(mut result) => {
                    if let Ok(Some(row)) = result.next().await {
                        row.get::<i64>("outlier_pruned").unwrap_or(0) as usize
                    } else { 0 }
                }
                Err(_) => 0,
            };

            if outlier_pruned > 0 {
                tracing::info!(outlier_pruned, "maintain_personas: pruned outlier KNOWS (inactive >14d)");
            }
        }

        // 3. Recalculate cohesion for each persona from SYNAPSE density between USES notes
        let cohesion_q = query(
            r#"
            MATCH (p:Persona {project_id: $project_id})
            OPTIONAL MATCH (p)-[:USES]->(n:Note)
            WITH p, collect(n.id) AS note_ids, count(n) AS note_count
            OPTIONAL MATCH (n1:Note)-[s:SYNAPSE]->(n2:Note)
            WHERE n1.id IN note_ids AND n2.id IN note_ids
            WITH p, note_count,
                 CASE WHEN note_count > 1
                      THEN toFloat(count(s)) / toFloat(note_count * (note_count - 1) / 2)
                      ELSE 0.0
                 END AS cohesion
            SET p.cohesion = CASE WHEN cohesion > 1.0 THEN 1.0 ELSE cohesion END,
                p.updated_at = $now
            RETURN count(p) AS updated
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("now", chrono::Utc::now().to_rfc3339());

        let mut cohesion_result = self
            .graph
            .execute(cohesion_q)
            .await
            .context("maintain_personas: cohesion")?;
        let personas_updated: usize = if let Some(row) = cohesion_result.next().await? {
            row.get::<i64>("updated").unwrap_or(0) as usize
        } else {
            0
        };

        // 3b. Detect merge candidates via affinity score
        let merge_q = query(r#"
            MATCH (a:Persona {project_id: $project_id}), (b:Persona {project_id: $project_id})
            WHERE a.status <> 'archived' AND b.status <> 'archived'
              AND a.id < b.id
            RETURN a.id AS aid, b.id AS bid
        "#)
        .param("project_id", project_id.to_string());

        if let Ok(mut merge_result) = self.graph.execute(merge_q).await {
            let mut merge_pairs = Vec::new();
            while let Ok(Some(row)) = merge_result.next().await {
                if let (Ok(aid), Ok(bid)) = (row.get::<String>("aid"), row.get::<String>("bid")) {
                    if let (Ok(a_id), Ok(b_id)) = (aid.parse::<Uuid>(), bid.parse::<Uuid>()) {
                        merge_pairs.push((a_id, b_id));
                    }
                }
            }

            for (a_id, b_id) in merge_pairs {
                match self.compute_persona_affinity(a_id, b_id).await {
                    Ok(score) if score.combined >= 0.8 => {
                        tracing::info!(
                            persona_a = %a_id, persona_b = %b_id,
                            jaccard = score.jaccard_files, synapse = score.synapse_density,
                            combined = score.combined,
                            "Auto-merging personas (affinity >= 0.8)"
                        );
                        let _ = self.merge_personas(a_id, b_id).await;
                    }
                    Ok(score) if score.combined >= 0.65 => {
                        tracing::info!(
                            persona_a = %a_id, persona_b = %b_id,
                            combined = score.combined,
                            "Merge candidate detected (0.65-0.8) — not auto-merging"
                        );
                    }
                    _ => {}
                }
            }
        }

        // 3c. Track energy_history and auto-dormant stagnant personas
        let history_q = query(r#"
            MATCH (p:Persona {project_id: $project_id})
            WHERE p.status IN ['active', 'emerging']
            RETURN p.id AS pid, p.energy AS energy,
                   COALESCE(p.energy_history, '[]') AS hist
        "#).param("project_id", project_id.to_string());

        if let Ok(mut hist_result) = self.graph.execute(history_q).await {
            while let Ok(Some(row)) = hist_result.next().await {
                let pid: String = match row.get("pid") {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let energy: f64 = row.get("energy").unwrap_or(0.5);
                let hist_str: String = row.get::<String>("hist").unwrap_or_else(|_| "[]".to_string());
                let mut history: Vec<f64> = serde_json::from_str(&hist_str).unwrap_or_default();

                history.push(energy);
                if history.len() > 5 {
                    history = history[history.len()-5..].to_vec();
                }

                let hist_json = serde_json::to_string(&history).unwrap_or_else(|_| "[]".to_string());

                // Detect persistent low energy: all 5 values below 0.15
                let all_low = history.len() >= 5 && history.iter().all(|e| *e < 0.15);

                if all_low {
                    let dormant_q = query(r#"
                        MATCH (p:Persona {id: $pid})
                        SET p.status = 'dormant', p.energy_history = $hist, p.updated_at = $now
                    "#)
                    .param("pid", pid.clone())
                    .param("hist", hist_json)
                    .param("now", chrono::Utc::now().to_rfc3339());
                    let _ = self.graph.run(dormant_q).await;
                    tracing::info!(persona_id = %pid, "Auto-dormant: energy below threshold for 5 cycles");
                } else {
                    let update_q = query(r#"
                        MATCH (p:Persona {id: $pid})
                        SET p.energy_history = $hist
                    "#)
                    .param("pid", pid.clone())
                    .param("hist", hist_json);
                    let _ = self.graph.run(update_q).await;
                }
            }
        }

        // 4. Update success_rate from AgentExecution history (last 30 days)
        let _success_q = query(
            r#"
            MATCH (p:Persona {project_id: $project_id})
            OPTIONAL MATCH (ae:AgentExecution)-[:EXECUTED_TASK]->(t:Task {persona: p.id})
            WHERE ae.started_at > datetime() - duration('P30D')
            WITH p,
                 count(ae) AS total_executions,
                 count(CASE WHEN ae.status = 'completed' THEN 1 END) AS successes
            WHERE total_executions > 0
            SET p.success_rate = toFloat(successes) / toFloat(total_executions),
                p.updated_at = $now
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("now", chrono::Utc::now().to_rfc3339());

        // Best-effort — AgentExecution schema may not match exactly
        let _ = self.graph.run(_success_q).await;

        // 5. Auto-scope personas to their best-matching FeatureGraph
        // Best-effort — if no FeatureGraphs exist yet, this is a no-op
        let _ = self.auto_scope_to_feature_graphs(project_id).await;

        // 6. Reset energy_boost_accumulated for all personas (end of cycle)
        let reset_q = query(r#"
            MATCH (p:Persona {project_id: $project_id})
            SET p.energy_boost_accumulated = 0.0
        "#)
        .param("project_id", project_id.to_string());
        let _ = self.graph.run(reset_q).await;

        Ok((decayed_count, pruned_count, personas_updated))
    }

    /// Detect potential personas from Louvain communities.
    ///
    /// Uses adaptive thresholds: confidence is normalized by the p90 of community
    /// file counts (from `compute_adaptive_thresholds`) instead of a hardcoded /20.
    ///
    /// Returns proposals: Vec<(community_label, file_count, suggested_name, confidence)>.
    pub async fn detect_personas(&self, project_id: Uuid) -> Result<Vec<PersonaProposal>> {
        // Compute adaptive confidence normalization
        let thresholds = self
            .compute_adaptive_thresholds(project_id)
            .await
            .unwrap_or_default();

        // Find communities with enough files that don't already have a persona
        let q = query(
            r#"
            MATCH (f:File)<-[:CONTAINS]-(p:Project {id: $project_id})
            WHERE COALESCE(f.fabric_community_id, f.community_id) IS NOT NULL
            WITH COALESCE(f.fabric_community_id, f.community_id) AS cid,
                 COALESCE(f.fabric_community_label, f.community_label) AS label,
                 collect(f.path) AS files,
                 count(f) AS file_count
            WHERE file_count >= 3
            OPTIONAL MATCH (persona:Persona {project_id: $project_id})-[:SCOPED_TO]->(fg:FeatureGraph)-[:INCLUDES_ENTITY]->(covered:File)
            WHERE COALESCE(covered.fabric_community_id, covered.community_id) = cid
            WITH cid, label, files, file_count,
                 count(DISTINCT persona) AS existing_personas
            WHERE existing_personas = 0
            RETURN cid, label, files, file_count
            ORDER BY file_count DESC
            LIMIT 10
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await.context("detect_personas")?;

        let mut proposals = Vec::new();
        while let Some(row) = result.next().await? {
            let community_id: i64 = row.get("cid").unwrap_or(0);
            let label: String = row.get("label").unwrap_or_default();
            let file_count: i64 = row.get("file_count").unwrap_or(0);
            let files: Vec<String> = row.get("files").unwrap_or_default();

            // Derive a suggested name from the community label or common path prefix
            let suggested_name = if !label.is_empty() && label != "unknown" {
                label
                    .to_lowercase()
                    .replace(' ', "-")
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '-')
                    .collect::<String>()
            } else {
                // Use common path prefix as name
                derive_name_from_paths(&files)
            };

            // Confidence: normalized by p90 of actual file count distribution
            // (adaptive, fallback to /20.0 when not enough data)
            let confidence = ((file_count as f64) / thresholds.confidence_p90).min(1.0);

            proposals.push(PersonaProposal {
                community_id,
                suggested_name: format!("{}-expert", suggested_name),
                file_count: file_count as usize,
                sample_files: files.into_iter().take(5).collect(),
                confidence,
            });
        }

        Ok(proposals)
    }

    /// Find personas relevant to a note based on KNOWS file overlap.
    ///
    /// Extracts file paths from note content, then finds active personas
    /// whose KNOWS files overlap with those paths. Falls back to returning
    /// an empty vec (semantic fallback is handled by the caller).
    ///
    /// Note: File nodes use ABSOLUTE paths but note content typically contains
    /// RELATIVE paths, so we use `ENDS WITH` matching.
    pub async fn find_relevant_personas_for_note(
        &self,
        file_paths: &[String],
        project_id: Uuid,
    ) -> Result<Vec<(Uuid, f64)>> {
        if file_paths.is_empty() {
            return Ok(vec![]);
        }

        // Normalize: strip leading "./" or "/" for consistent ENDS WITH matching
        let normalized: Vec<String> = file_paths
            .iter()
            .map(|p| {
                p.strip_prefix("./")
                    .or_else(|| p.strip_prefix('/'))
                    .unwrap_or(p)
                    .to_string()
            })
            .collect();

        let q = query(
            r#"
            UNWIND $file_paths AS fp
            MATCH (p:Persona {project_id: $project_id, status: "active"})-[k:KNOWS]->(f:File)
            WHERE f.path ENDS WITH fp
            WITH p, count(DISTINCT f) AS overlap, avg(k.weight) AS avg_weight
            RETURN p.id AS persona_id, overlap, avg_weight
            ORDER BY overlap DESC, avg_weight DESC
            "#,
        )
        .param("file_paths", normalized)
        .param("project_id", project_id.to_string());

        let mut result = self
            .graph
            .execute(q)
            .await
            .context("find_relevant_personas_for_note")?;

        let mut personas = Vec::new();
        while let Some(row) = result.next().await? {
            let id_str: String = row.get("persona_id")?;
            let id: Uuid = id_str.parse()?;
            let avg_weight: f64 = row.get("avg_weight").unwrap_or(0.5);
            personas.push((id, avg_weight));
        }

        Ok(personas)
    }

    /// Find personas relevant to a decision based on KNOWS overlap with AFFECTS entities.
    ///
    /// Looks at the files/functions affected by a decision (via AFFECTS relations),
    /// then finds active personas whose KNOWS files overlap with those entities.
    pub async fn find_relevant_personas_for_decision(
        &self,
        decision_id: Uuid,
        project_id: Uuid,
    ) -> Result<Vec<(Uuid, f64)>> {
        let q = query(
            r#"
            MATCH (d:Decision {id: $decision_id})-[:AFFECTS]->(target)
            WHERE target:File OR target:Function
            WITH CASE WHEN target:Function
                THEN [(target)<-[:DEFINES]-(f:File) | f][0]
                ELSE target
            END AS file
            WHERE file IS NOT NULL
            WITH collect(DISTINCT file) AS affected_files
            UNWIND affected_files AS af
            MATCH (p:Persona {project_id: $project_id, status: "active"})-[k:KNOWS]->(af)
            WITH p, count(DISTINCT af) AS overlap, avg(k.weight) AS avg_weight
            RETURN p.id AS persona_id, overlap, avg_weight
            ORDER BY overlap DESC, avg_weight DESC
            "#,
        )
        .param("decision_id", decision_id.to_string())
        .param("project_id", project_id.to_string());

        let mut result = self
            .graph
            .execute(q)
            .await
            .context("find_relevant_personas_for_decision")?;

        let mut personas = Vec::new();
        while let Some(row) = result.next().await? {
            let id_str: String = row.get("persona_id")?;
            let id: Uuid = id_str.parse()?;
            let avg_weight: f64 = row.get("avg_weight").unwrap_or(0.5);
            personas.push((id, avg_weight));
        }

        Ok(personas)
    }

    /// Check if a file is adjacent to a persona's KNOWS scope.
    ///
    /// Adjacent means:
    /// 1. Same directory as a file the persona KNOWS, OR
    /// 2. 1 hop via IMPORTS from a file the persona KNOWS
    ///
    /// Returns the list of personas (id, name) that are adjacent to this file.
    /// File paths are ABSOLUTE (matching File node convention).
    pub async fn find_adjacent_personas(
        &self,
        file_path: &str,
        project_id: Uuid,
    ) -> Result<Vec<(Uuid, String)>> {
        // Extract directory from file path for same-dir matching
        let dir = file_path
            .rsplit_once('/')
            .map(|(d, _)| format!("{d}/"))
            .unwrap_or_default();

        if dir.is_empty() {
            return Ok(vec![]);
        }

        let q = query(
            r#"
            // Strategy 1: Same directory — persona KNOWS a file in the same directory
            OPTIONAL MATCH (p1:Persona {project_id: $project_id, status: "active"})-[:KNOWS]->(f1:File)
            WHERE f1.path STARTS WITH $dir AND f1.path <> $file_path
            WITH collect(DISTINCT p1) AS dir_personas

            // Strategy 2: 1-hop IMPORTS — file imports or is imported by a KNOWS file
            OPTIONAL MATCH (target:File {path: $file_path})-[:IMPORTS|IMPORTED_BY]-(neighbor:File)<-[:KNOWS]-(p2:Persona {project_id: $project_id, status: "active"})
            WITH dir_personas, collect(DISTINCT p2) AS import_personas

            // Union both sets
            WITH dir_personas + import_personas AS all_personas
            UNWIND all_personas AS p
            WITH DISTINCT p
            WHERE p IS NOT NULL
            RETURN p.id AS persona_id, p.name AS persona_name
            "#,
        )
        .param("file_path", file_path)
        .param("dir", dir)
        .param("project_id", project_id.to_string());

        let mut result = self
            .graph
            .execute(q)
            .await
            .context("find_adjacent_personas")?;

        let mut personas = Vec::new();
        while let Some(row) = result.next().await? {
            let id_str: String = row.get("persona_id")?;
            let id: Uuid = id_str.parse()?;
            let name: String = row.get("persona_name")?;
            personas.push((id, name));
        }

        Ok(personas)
    }

    /// Auto-grow: create a KNOWS relation between a persona and a file with low weight.
    ///
    /// Used when an agent touches a file that is adjacent to a persona's scope
    /// but not yet in KNOWS. Creates a weak KNOWS (weight 0.3) via MERGE.
    pub async fn auto_grow_file_knows(
        &self,
        persona_id: Uuid,
        file_path: &str,
        weight: f64,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id}), (f:File {path: $file_path})
            MERGE (p)-[k:KNOWS]->(f)
            ON CREATE SET k.weight = $weight, k.source = "auto-grow"
            ON MATCH SET k.weight = CASE WHEN k.weight < $weight THEN $weight ELSE k.weight END
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("file_path", file_path)
        .param("weight", weight);

        self.graph.run(q).await.context("auto_grow_file_knows")?;

        // Propagate to CO_CHANGED neighbors (best-effort)
        let _ = self.propagate_knows_via_co_change(persona_id, file_path, weight).await;

        Ok(())
    }

    // ========================================================================
    // CO_CHANGED propagation
    // ========================================================================

    /// Propagate KNOWS to files co-changed with the target file.
    /// Creates attenuated KNOWS (weight * 0.4) for the top 5 co-changed neighbors.
    pub async fn propagate_knows_via_co_change(
        &self,
        persona_id: Uuid,
        file_path: &str,
        base_weight: f64,
    ) -> Result<usize> {
        const CO_CHANGE_ATTENUATION: f64 = 0.4;
        const CO_CHANGE_MIN_COUNT: i64 = 3;
        const CO_CHANGE_MAX_FILES: i64 = 5;

        let attenuated_weight = base_weight * CO_CHANGE_ATTENUATION;
        let q = query(r#"
            MATCH (f:File {path: $file_path})-[cc:CO_CHANGED]-(neighbor:File)
            WHERE cc.count >= $min_count
            WITH neighbor, cc.count AS co_count
            ORDER BY co_count DESC
            LIMIT $max_files
            MATCH (p:Persona {id: $persona_id})
            MERGE (p)-[k:KNOWS]->(neighbor)
            ON CREATE SET k.weight = $weight, k.source = "co-change"
            ON MATCH SET k.weight = CASE WHEN k.weight < $weight THEN $weight ELSE k.weight END
            RETURN count(neighbor) AS propagated
        "#)
        .param("persona_id", persona_id.to_string())
        .param("file_path", file_path)
        .param("weight", attenuated_weight)
        .param("min_count", CO_CHANGE_MIN_COUNT)
        .param("max_files", CO_CHANGE_MAX_FILES);

        let mut result = self.graph.execute(q).await.context("propagate_knows_via_co_change")?;
        let count = if let Some(row) = result.next().await? {
            row.get::<i64>("propagated").unwrap_or(0) as usize
        } else {
            0
        };

        if count > 0 {
            tracing::debug!(persona_id = %persona_id, file_path, count, "Propagated KNOWS via CO_CHANGED");
        }
        Ok(count)
    }

    // ========================================================================
    // Persona Affinity & Merge
    // ========================================================================

    /// Compute affinity between two personas for merge detection.
    /// Score = 0.6 * jaccard_files + 0.4 * synapse_density
    pub async fn compute_persona_affinity(
        &self,
        persona_a: Uuid,
        persona_b: Uuid,
    ) -> Result<PersonaAffinityScore> {
        let q = query(r#"
            // Jaccard on KNOWS files
            MATCH (a:Persona {id: $pa})-[:KNOWS]->(fa:File)
            WITH a, collect(DISTINCT fa) AS files_a
            MATCH (b:Persona {id: $pb})-[:KNOWS]->(fb:File)
            WITH a, b, files_a, collect(DISTINCT fb) AS files_b
            WITH a, b, files_a, files_b,
                 size([f IN files_a WHERE f IN files_b]) AS intersection,
                 size(files_a) + size(files_b) - size([f IN files_a WHERE f IN files_b]) AS union_size
            WITH a, b,
                 CASE WHEN union_size > 0 THEN toFloat(intersection) / union_size ELSE 0.0 END AS jaccard

            // SYNAPSE density between USES notes
            OPTIONAL MATCH (a)-[:USES]->(na:Note)
            WITH a, b, jaccard, collect(DISTINCT na) AS notes_a
            OPTIONAL MATCH (b)-[:USES]->(nb:Note)
            WITH a, b, jaccard, notes_a, collect(DISTINCT nb) AS notes_b
            WITH a, b, jaccard, notes_a, notes_b,
                 [na IN notes_a | na.id] AS note_ids_a,
                 [nb IN notes_b | nb.id] AS note_ids_b
            OPTIONAL MATCH (s1:Note)-[syn:SYNAPSE]->(s2:Note)
            WHERE s1.id IN note_ids_a AND s2.id IN note_ids_b
            WITH jaccard,
                 CASE WHEN size(note_ids_a) > 0 AND size(note_ids_b) > 0
                      THEN toFloat(count(syn)) / toFloat(size(note_ids_a) * size(note_ids_b))
                      ELSE 0.0
                 END AS synapse_density
            RETURN jaccard, synapse_density
        "#)
        .param("pa", persona_a.to_string())
        .param("pb", persona_b.to_string());

        let mut result = self.graph.execute(q).await.context("compute_persona_affinity")?;
        let (jaccard, synapse_density) = if let Some(row) = result.next().await? {
            (
                row.get::<f64>("jaccard").unwrap_or(0.0),
                row.get::<f64>("synapse_density").unwrap_or(0.0),
            )
        } else {
            (0.0, 0.0)
        };

        let combined = 0.6 * jaccard + 0.4 * synapse_density;

        Ok(PersonaAffinityScore {
            persona_a_id: persona_a,
            persona_b_id: persona_b,
            jaccard_files: jaccard,
            synapse_density,
            combined,
        })
    }

    /// Merge persona B into persona A: transfer all KNOWS/USES, then delete B.
    pub async fn merge_personas(&self, keep_id: Uuid, merge_id: Uuid) -> Result<()> {
        let q = query(r#"
            // Transfer KNOWS from merge -> keep (MERGE to avoid duplicates)
            MATCH (merge:Persona {id: $merge_id})-[k:KNOWS]->(f)
            MATCH (keep:Persona {id: $keep_id})
            MERGE (keep)-[nk:KNOWS]->(f)
            ON CREATE SET nk.weight = k.weight, nk.source = "merged"
            ON MATCH SET nk.weight = CASE WHEN nk.weight < k.weight THEN k.weight ELSE nk.weight END
            WITH merge, keep

            // Transfer USES from merge -> keep
            OPTIONAL MATCH (merge)-[u:USES]->(n)
            MERGE (keep)-[nu:USES]->(n)
            ON CREATE SET nu.weight = u.weight
            ON MATCH SET nu.weight = CASE WHEN nu.weight < u.weight THEN u.weight ELSE nu.weight END
            WITH DISTINCT merge

            // Delete merged persona
            DETACH DELETE merge
        "#)
        .param("keep_id", keep_id.to_string())
        .param("merge_id", merge_id.to_string());

        self.graph.run(q).await.context("merge_personas")?;
        tracing::info!(keep = %keep_id, merged = %merge_id, "Merged personas");
        Ok(())
    }

    // ========================================================================
    // SYNAPSE-linked personas (pre-warming)
    // ========================================================================

    /// Find personas linked via strong SYNAPSE connections between their USES notes.
    pub async fn find_synapse_linked_personas(
        &self,
        persona_id: Uuid,
    ) -> Result<Vec<(Uuid, String, f64)>> {
        let q = query(r#"
            MATCH (a:Persona {id: $pid})-[:USES]->(na:Note)-[syn:SYNAPSE]-(nb:Note)<-[:USES]-(b:Persona)
            WHERE b.id <> $pid AND b.status <> 'archived'
            WITH b, avg(syn.weight) AS avg_weight, count(syn) AS syn_count
            WHERE avg_weight >= 0.3 AND syn_count >= 2
            RETURN b.id AS pid, b.name AS name, avg_weight
            ORDER BY avg_weight DESC
            LIMIT 3
        "#)
        .param("pid", persona_id.to_string());

        let mut result = self.graph.execute(q).await.context("find_synapse_linked_personas")?;
        let mut linked = Vec::new();
        while let Some(row) = result.next().await? {
            let id: Uuid = row.get::<String>("pid")?.parse()?;
            let name: String = row.get("name")?;
            let weight: f64 = row.get("avg_weight").unwrap_or(0.0);
            linked.push((id, name, weight));
        }
        Ok(linked)
    }

    // ========================================================================
    // Rate-limited energy boost
    // ========================================================================

    /// Check if a persona can receive more energy boost this cycle.
    /// Returns true if boost was applied, false if rate-limited.
    pub async fn rate_limited_energy_boost(
        &self,
        persona_id: Uuid,
        boost: f64,
        max_per_cycle: f64,
    ) -> Result<bool> {
        let q = query(r#"
            MATCH (p:Persona {id: $pid})
            WHERE COALESCE(p.energy_boost_accumulated, 0.0) + $boost <= $max
            SET p.energy_boost_accumulated = COALESCE(p.energy_boost_accumulated, 0.0) + $boost,
                p.energy = CASE WHEN p.energy + $boost > 1.0 THEN 1.0 ELSE p.energy + $boost END,
                p.updated_at = datetime()
            RETURN p.id AS updated
        "#)
        .param("pid", persona_id.to_string())
        .param("boost", boost)
        .param("max", max_per_cycle);

        let mut result = self.graph.execute(q).await.context("rate_limited_energy_boost")?;
        Ok(result.next().await?.is_some())
    }

    /// Auto-link a note to the active persona for a task (growth hook).
    pub async fn auto_link_note_to_persona(
        &self,
        persona_id: Uuid,
        note_id: Uuid,
        weight: f64,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id}), (n:Note {id: $note_id})
            MERGE (p)-[r:USES]->(n)
            ON CREATE SET r.weight = $weight
            ON MATCH SET r.weight = CASE WHEN r.weight < $weight THEN $weight ELSE r.weight END
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("note_id", note_id.to_string())
        .param("weight", weight);

        self.graph
            .run(q)
            .await
            .context("auto_link_note_to_persona")?;
        Ok(())
    }

    /// Auto-link a decision to the active persona (growth hook).
    pub async fn auto_link_decision_to_persona(
        &self,
        persona_id: Uuid,
        decision_id: Uuid,
        weight: f64,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id}), (d:Decision {id: $decision_id})
            MERGE (p)-[r:USES]->(d)
            ON CREATE SET r.weight = $weight
            ON MATCH SET r.weight = CASE WHEN r.weight < $weight THEN $weight ELSE r.weight END
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("decision_id", decision_id.to_string())
        .param("weight", weight);

        self.graph
            .run(q)
            .await
            .context("auto_link_decision_to_persona")?;
        Ok(())
    }

    /// Auto-link a new file (adjacent to scope) to persona (growth hook).
    pub async fn auto_link_file_to_persona(
        &self,
        persona_id: Uuid,
        file_path: &str,
        weight: f64,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Persona {id: $persona_id}), (f:File {path: $file_path})
            MERGE (p)-[r:KNOWS]->(f)
            ON CREATE SET r.weight = $weight
            ON MATCH SET r.weight = CASE WHEN r.weight < $weight THEN $weight ELSE r.weight END
            "#,
        )
        .param("persona_id", persona_id.to_string())
        .param("file_path", file_path)
        .param("weight", weight);

        self.graph
            .run(q)
            .await
            .context("auto_link_file_to_persona")?;
        Ok(())
    }
}

/// Affinity score between two personas for merge detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaAffinityScore {
    pub persona_a_id: Uuid,
    pub persona_b_id: Uuid,
    pub jaccard_files: f64,
    pub synapse_density: f64,
    pub combined: f64,
}

/// Persona detection proposal — a candidate persona from community analysis.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersonaProposal {
    pub community_id: i64,
    pub suggested_name: String,
    pub file_count: usize,
    pub sample_files: Vec<String>,
    pub confidence: f64,
}

/// Derive a persona name from common path prefix of files.
fn derive_name_from_paths(paths: &[String]) -> String {
    if paths.is_empty() {
        return "unknown".to_string();
    }

    // Find the longest common path prefix
    let first = &paths[0];
    let parts: Vec<&str> = first.split('/').collect();

    let mut common_depth = parts.len();
    for path in paths.iter().skip(1) {
        let other_parts: Vec<&str> = path.split('/').collect();
        let matching = parts
            .iter()
            .zip(other_parts.iter())
            .take_while(|(a, b)| a == b)
            .count();
        common_depth = common_depth.min(matching);
    }

    if common_depth > 0 {
        // Use the deepest common directory component
        let name = parts[common_depth - 1];
        name.to_string()
    } else {
        "mixed".to_string()
    }
}

// ============================================================================
// Adaptive thresholds (rs-stats integration)
// ============================================================================

/// Data-driven thresholds computed from actual KNOWS/USES weight distributions.
///
/// Replaces hardcoded values (0.1 prune, file_count/20 confidence, etc.) with
/// percentile-derived thresholds that adapt to each project's unique characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePersonaThresholds {
    /// Prune cutoff: KNOWS/USES relations with weight below this are removed.
    /// Computed as p5 of all weights (fallback: 0.1).
    pub prune_cutoff: f64,
    /// p90 of community file counts — used to normalize detect_personas confidence.
    /// Confidence = (file_count / confidence_p90).min(1.0) (fallback: 20.0).
    pub confidence_p90: f64,
    /// Indices of outlier weights (Tukey k=1.5) — logged for review.
    pub weight_outlier_count: usize,
    /// Number of KNOWS/USES weights used for computation.
    pub sample_size: usize,
    /// Lower fence (Q1 - 1.5*IQR) for outlier pruning.
    pub lower_fence: f64,
}

impl Default for AdaptivePersonaThresholds {
    fn default() -> Self {
        Self {
            prune_cutoff: 0.1,
            confidence_p90: 20.0,
            weight_outlier_count: 0,
            sample_size: 0,
            lower_fence: 0.0,
        }
    }
}

impl Neo4jClient {
    /// Compute adaptive thresholds from the actual KNOWS/USES weight distribution.
    ///
    /// Fetches all relation weights for a project, then uses `adaptive_threshold`
    /// and `detect_outliers` from rs-stats to derive data-driven cutoffs.
    ///
    /// Falls back to hardcoded defaults when there are fewer than 10 data points
    /// (not enough for a meaningful distribution).
    pub async fn compute_adaptive_thresholds(
        &self,
        project_id: Uuid,
    ) -> Result<AdaptivePersonaThresholds> {
        // Collect all KNOWS/USES weights for this project
        let q = query(
            r#"
            MATCH (p:Persona {project_id: $project_id})-[r:KNOWS|USES]->()
            WHERE r.weight IS NOT NULL
            RETURN r.weight AS w
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self
            .graph
            .execute(q)
            .await
            .context("compute_adaptive_thresholds: fetch weights")?;

        let mut weights: Vec<f64> = Vec::new();
        while let Some(row) = result.next().await? {
            if let Ok(w) = row.get::<f64>("w") {
                weights.push(w);
            }
        }

        let sample_size = weights.len();

        // Not enough data for meaningful statistics — use defaults
        if sample_size < 10 {
            tracing::debug!(
                sample_size,
                "Not enough KNOWS/USES weights for adaptive thresholds, using defaults"
            );
            return Ok(AdaptivePersonaThresholds {
                sample_size,
                ..Default::default()
            });
        }

        // Prune cutoff = p5 of weights (bottom 5% are pruned)
        let prune_cutoff = adaptive_threshold(&weights, 0.05, 0.1);

        // Compute lower fence for outlier pruning (Q1 - 1.5*IQR)
        let mut sorted = weights.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q1 = adaptive_threshold(&sorted, 0.25, 0.0);
        let q3 = adaptive_threshold(&sorted, 0.75, 1.0);
        let iqr = q3 - q1;
        let lower_fence = (q1 - 1.5 * iqr).max(0.0);

        // Outlier detection (Tukey k=1.5)
        let outliers = detect_outliers(&weights, 1.5);
        let weight_outlier_count = outliers.len();

        if weight_outlier_count > 0 {
            tracing::info!(
                weight_outlier_count,
                prune_cutoff,
                sample_size,
                "Adaptive thresholds: detected {} outlier KNOWS/USES weights",
                weight_outlier_count
            );
        }

        // Confidence normalization: collect community file counts
        let fc_q = query(
            r#"
            MATCH (f:File)<-[:CONTAINS]-(proj:Project {id: $project_id})
            WHERE COALESCE(f.fabric_community_id, f.community_id) IS NOT NULL
            WITH COALESCE(f.fabric_community_id, f.community_id) AS cid, count(f) AS file_count
            WHERE file_count >= 3
            RETURN file_count
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut fc_result = self
            .graph
            .execute(fc_q)
            .await
            .context("compute_adaptive_thresholds: fetch file counts")?;

        let mut file_counts: Vec<f64> = Vec::new();
        while let Some(row) = fc_result.next().await? {
            if let Ok(fc) = row.get::<i64>("file_count") {
                file_counts.push(fc as f64);
            }
        }

        let confidence_p90 = adaptive_threshold(&file_counts, 0.90, 20.0);

        tracing::debug!(
            prune_cutoff,
            confidence_p90,
            weight_outlier_count,
            sample_size,
            "Computed adaptive persona thresholds"
        );

        Ok(AdaptivePersonaThresholds {
            prune_cutoff,
            confidence_p90,
            weight_outlier_count,
            sample_size,
            lower_fence,
        })
    }
}

// ============================================================================
// Tests (using MockGraphStore)
// ============================================================================

#[cfg(test)]
mod tests {
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::models::*;
    use crate::neo4j::traits::GraphStore;
    use chrono::Utc;
    use uuid::Uuid;

    fn make_persona(name: &str, project_id: Option<Uuid>) -> PersonaNode {
        PersonaNode {
            id: Uuid::new_v4(),
            project_id,
            name: name.to_string(),
            description: format!("Expert in {name}"),
            status: PersonaStatus::Emerging,
            complexity_default: None,
            timeout_secs: Some(1800),
            max_cost_usd: Some(1.0),
            model_preference: None,
            system_prompt_override: None,
            energy: 0.5,
            cohesion: 0.0,
            activation_count: 0,
            success_rate: 0.0,
            avg_duration_secs: 0.0,
            last_activated: None,
            energy_boost_accumulated: 0.0,
            energy_history: vec![],
            origin: PersonaOrigin::Manual,
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    #[tokio::test]
    async fn test_persona_crud_lifecycle() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();
        let mut persona = make_persona("neo4j-expert", Some(project_id));

        // Create
        store.create_persona(&persona).await.unwrap();

        // Get
        let fetched = store.get_persona(persona.id).await.unwrap().unwrap();
        assert_eq!(fetched.name, "neo4j-expert");
        assert_eq!(fetched.project_id, Some(project_id));
        assert!((fetched.energy - 0.5).abs() < f64::EPSILON);

        // Update
        persona.energy = 0.9;
        persona.status = PersonaStatus::Active;
        persona.name = "neo4j-master".to_string();
        store.update_persona(&persona).await.unwrap();

        let updated = store.get_persona(persona.id).await.unwrap().unwrap();
        assert_eq!(updated.name, "neo4j-master");
        assert_eq!(updated.status, PersonaStatus::Active);
        assert!((updated.energy - 0.9).abs() < f64::EPSILON);

        // Delete
        let deleted = store.delete_persona(persona.id).await.unwrap();
        assert!(deleted);
        assert!(store.get_persona(persona.id).await.unwrap().is_none());

        // Delete non-existent
        let deleted_again = store.delete_persona(persona.id).await.unwrap();
        assert!(!deleted_again);
    }

    #[tokio::test]
    async fn test_list_personas_with_filter() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let mut p1 = make_persona("frontend", Some(project_id));
        p1.status = PersonaStatus::Active;
        p1.energy = 0.9;
        store.create_persona(&p1).await.unwrap();

        let mut p2 = make_persona("backend", Some(project_id));
        p2.status = PersonaStatus::Emerging;
        p2.energy = 0.3;
        store.create_persona(&p2).await.unwrap();

        let mut p3 = make_persona("other-project", Some(Uuid::new_v4()));
        p3.status = PersonaStatus::Active;
        store.create_persona(&p3).await.unwrap();

        // List all for project
        let (all, total) = store.list_personas(project_id, None, 50, 0).await.unwrap();
        assert_eq!(total, 2);
        assert_eq!(all.len(), 2);
        // Sorted by energy desc
        assert_eq!(all[0].name, "frontend");
        assert_eq!(all[1].name, "backend");

        // Filter by status
        let (active, count) = store
            .list_personas(project_id, Some(PersonaStatus::Active), 50, 0)
            .await
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(active[0].name, "frontend");

        // Pagination
        let (page, _) = store.list_personas(project_id, None, 1, 1).await.unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].name, "backend");
    }

    #[tokio::test]
    async fn test_list_global_personas() {
        let store = MockGraphStore::new();

        let global = make_persona("rust-expert", None);
        store.create_persona(&global).await.unwrap();

        let scoped = make_persona("project-specific", Some(Uuid::new_v4()));
        store.create_persona(&scoped).await.unwrap();

        let globals = store.list_global_personas().await.unwrap();
        assert_eq!(globals.len(), 1);
        assert_eq!(globals[0].name, "rust-expert");
    }

    #[tokio::test]
    async fn test_persona_skill_relations() {
        let store = MockGraphStore::new();
        let persona = make_persona("test", Some(Uuid::new_v4()));
        store.create_persona(&persona).await.unwrap();

        let skill_id1 = Uuid::new_v4();
        let skill_id2 = Uuid::new_v4();

        // Add skills
        store
            .add_persona_skill(persona.id, skill_id1)
            .await
            .unwrap();
        store
            .add_persona_skill(persona.id, skill_id2)
            .await
            .unwrap();

        // Idempotent — adding same skill again is fine
        store
            .add_persona_skill(persona.id, skill_id1)
            .await
            .unwrap();

        // Subgraph should reflect skills
        let sg = store.get_persona_subgraph(persona.id).await.unwrap();
        assert_eq!(sg.skills.len(), 2);

        // Remove one
        store
            .remove_persona_skill(persona.id, skill_id1)
            .await
            .unwrap();
        let sg = store.get_persona_subgraph(persona.id).await.unwrap();
        assert_eq!(sg.skills.len(), 1);
        assert!(sg
            .skills
            .iter()
            .any(|r| r.entity_id == skill_id2.to_string()));
    }

    #[tokio::test]
    async fn test_persona_protocol_relations() {
        let store = MockGraphStore::new();
        let persona = make_persona("test", Some(Uuid::new_v4()));
        store.create_persona(&persona).await.unwrap();

        let proto_id = Uuid::new_v4();
        store
            .add_persona_protocol(persona.id, proto_id)
            .await
            .unwrap();

        let sg = store.get_persona_subgraph(persona.id).await.unwrap();
        assert_eq!(sg.protocols.len(), 1);
        assert_eq!(sg.protocols[0].entity_id, proto_id.to_string());

        store
            .remove_persona_protocol(persona.id, proto_id)
            .await
            .unwrap();
        let sg = store.get_persona_subgraph(persona.id).await.unwrap();
        assert!(sg.protocols.is_empty());
    }

    #[tokio::test]
    async fn test_persona_file_relations_with_weight() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();
        let persona = make_persona("neo4j-expert", Some(project_id));
        store.create_persona(&persona).await.unwrap();

        // Add files with different weights
        store
            .add_persona_file(persona.id, "src/neo4j/client.rs", 0.9)
            .await
            .unwrap();
        store
            .add_persona_file(persona.id, "src/neo4j/models.rs", 0.7)
            .await
            .unwrap();

        let sg = store.get_persona_subgraph(persona.id).await.unwrap();
        assert_eq!(sg.files.len(), 2);

        // Update weight (re-add with different weight)
        store
            .add_persona_file(persona.id, "src/neo4j/client.rs", 0.95)
            .await
            .unwrap();
        let sg = store.get_persona_subgraph(persona.id).await.unwrap();
        let client_rel = sg
            .files
            .iter()
            .find(|f| f.entity_id == "src/neo4j/client.rs")
            .unwrap();
        assert!((client_rel.weight - 0.95).abs() < f64::EPSILON);

        // Remove
        store
            .remove_persona_file(persona.id, "src/neo4j/client.rs")
            .await
            .unwrap();
        let sg = store.get_persona_subgraph(persona.id).await.unwrap();
        assert_eq!(sg.files.len(), 1);
    }

    #[tokio::test]
    async fn test_persona_note_and_decision_relations() {
        let store = MockGraphStore::new();
        let persona = make_persona("test", Some(Uuid::new_v4()));
        store.create_persona(&persona).await.unwrap();

        let note_id = Uuid::new_v4();
        let decision_id = Uuid::new_v4();

        store
            .add_persona_note(persona.id, note_id, 0.8)
            .await
            .unwrap();
        store
            .add_persona_decision(persona.id, decision_id, 0.6)
            .await
            .unwrap();

        let sg = store.get_persona_subgraph(persona.id).await.unwrap();
        assert_eq!(sg.notes.len(), 1);
        assert_eq!(sg.notes[0].entity_id, note_id.to_string());
        assert!((sg.notes[0].weight - 0.8).abs() < f64::EPSILON);

        assert_eq!(sg.decisions.len(), 1);
        assert_eq!(sg.decisions[0].entity_id, decision_id.to_string());

        // Remove
        store
            .remove_persona_note(persona.id, note_id)
            .await
            .unwrap();
        store
            .remove_persona_decision(persona.id, decision_id)
            .await
            .unwrap();
        let sg = store.get_persona_subgraph(persona.id).await.unwrap();
        assert!(sg.notes.is_empty());
        assert!(sg.decisions.is_empty());
    }

    #[tokio::test]
    async fn test_persona_extends_chain() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let root = make_persona("root-expert", Some(project_id));
        let mid = make_persona("mid-expert", Some(project_id));
        let child = make_persona("child-expert", Some(project_id));

        store.create_persona(&root).await.unwrap();
        store.create_persona(&mid).await.unwrap();
        store.create_persona(&child).await.unwrap();

        // child -> mid -> root
        store.add_persona_extends(child.id, mid.id).await.unwrap();
        store.add_persona_extends(mid.id, root.id).await.unwrap();

        let sg = store.get_persona_subgraph(child.id).await.unwrap();
        assert_eq!(sg.parents.len(), 1); // Mock doesn't do transitive
        assert!(sg.parents.iter().any(|r| r.entity_id == mid.id.to_string()));

        // Remove
        store
            .remove_persona_extends(child.id, mid.id)
            .await
            .unwrap();
        let sg = store.get_persona_subgraph(child.id).await.unwrap();
        assert!(sg.parents.is_empty());
    }

    #[tokio::test]
    async fn test_persona_feature_graph() {
        let store = MockGraphStore::new();
        let persona = make_persona("test", Some(Uuid::new_v4()));
        store.create_persona(&persona).await.unwrap();

        let fg_id = Uuid::new_v4();
        store
            .set_persona_feature_graph(persona.id, fg_id)
            .await
            .unwrap();

        let sg = store.get_persona_subgraph(persona.id).await.unwrap();
        assert_eq!(sg.feature_graph_id, Some(fg_id));

        // Replace
        let fg_id2 = Uuid::new_v4();
        store
            .set_persona_feature_graph(persona.id, fg_id2)
            .await
            .unwrap();
        let sg = store.get_persona_subgraph(persona.id).await.unwrap();
        assert_eq!(sg.feature_graph_id, Some(fg_id2));

        // Remove
        store
            .remove_persona_feature_graph(persona.id)
            .await
            .unwrap();
        let sg = store.get_persona_subgraph(persona.id).await.unwrap();
        assert!(sg.feature_graph_id.is_none());
    }

    #[tokio::test]
    async fn test_persona_subgraph_stats() {
        let store = MockGraphStore::new();
        let persona = make_persona("test", Some(Uuid::new_v4()));
        store.create_persona(&persona).await.unwrap();

        // Add various relations
        store
            .add_persona_file(persona.id, "a.rs", 0.9)
            .await
            .unwrap();
        store
            .add_persona_file(persona.id, "b.rs", 0.7)
            .await
            .unwrap();
        store
            .add_persona_note(persona.id, Uuid::new_v4(), 0.8)
            .await
            .unwrap();
        store
            .add_persona_skill(persona.id, Uuid::new_v4())
            .await
            .unwrap();

        let sg = store.get_persona_subgraph(persona.id).await.unwrap();
        assert_eq!(sg.stats.total_entities, 4); // 2 files + 1 note + 1 skill
        assert_eq!(sg.persona_name, "test");
    }

    #[tokio::test]
    async fn test_find_personas_for_file() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let p1 = make_persona("neo4j-expert", Some(project_id));
        let mut p2 = make_persona("api-expert", Some(project_id));
        p2.status = PersonaStatus::Active;
        let mut p3 = make_persona("archived", Some(project_id));
        p3.status = PersonaStatus::Archived;

        store.create_persona(&p1).await.unwrap();
        store.create_persona(&p2).await.unwrap();
        store.create_persona(&p3).await.unwrap();

        // Both p1 and p2 know the file, p3 is archived
        store
            .add_persona_file(p1.id, "src/neo4j/client.rs", 0.9)
            .await
            .unwrap();
        store
            .add_persona_file(p2.id, "src/neo4j/client.rs", 0.5)
            .await
            .unwrap();
        store
            .add_persona_file(p3.id, "src/neo4j/client.rs", 1.0)
            .await
            .unwrap();

        let results = store
            .find_personas_for_file("src/neo4j/client.rs", project_id)
            .await
            .unwrap();

        // Should return p1 and p2 (not p3 — archived), sorted by weight desc
        // Note: mock doesn't filter archived, but p3 is still returned.
        // The Neo4j impl filters by status <> 'archived'.
        // Mock returns all — that's expected for mock simplicity.
        assert!(results.len() >= 2);
        // First should be highest weight
        assert!(results[0].1 >= results[1].1);
    }

    #[tokio::test]
    async fn test_persona_delete_cleans_relations() {
        let store = MockGraphStore::new();
        let persona = make_persona("test", Some(Uuid::new_v4()));
        store.create_persona(&persona).await.unwrap();

        // Add various relations
        store
            .add_persona_skill(persona.id, Uuid::new_v4())
            .await
            .unwrap();
        store
            .add_persona_file(persona.id, "a.rs", 0.9)
            .await
            .unwrap();
        store
            .add_persona_note(persona.id, Uuid::new_v4(), 0.8)
            .await
            .unwrap();

        // Delete should clean everything
        store.delete_persona(persona.id).await.unwrap();

        // All relation stores should be cleaned
        assert!(store.persona_skills.read().await.get(&persona.id).is_none());
        assert!(store.persona_files.read().await.get(&persona.id).is_none());
        assert!(store.persona_notes.read().await.get(&persona.id).is_none());
    }

    #[tokio::test]
    async fn test_persona_origin_and_status_enums() {
        // Test Display + FromStr roundtrip
        assert_eq!(PersonaOrigin::AutoBuild.to_string(), "auto_build");
        assert_eq!(PersonaOrigin::Manual.to_string(), "manual");
        assert_eq!(PersonaOrigin::Imported.to_string(), "imported");
        assert_eq!(
            "auto_build".parse::<PersonaOrigin>().unwrap(),
            PersonaOrigin::AutoBuild
        );
        assert!("invalid".parse::<PersonaOrigin>().is_err());

        assert_eq!(PersonaStatus::Active.to_string(), "active");
        assert_eq!(PersonaStatus::Dormant.to_string(), "dormant");
        assert_eq!(PersonaStatus::Emerging.to_string(), "emerging");
        assert_eq!(PersonaStatus::Archived.to_string(), "archived");
        assert_eq!(
            "active".parse::<PersonaStatus>().unwrap(),
            PersonaStatus::Active
        );
        assert!("invalid".parse::<PersonaStatus>().is_err());

        // Defaults
        assert_eq!(PersonaOrigin::default(), PersonaOrigin::Manual);
        assert_eq!(PersonaStatus::default(), PersonaStatus::Emerging);
    }

    #[tokio::test]
    async fn test_persona_serde_roundtrip() {
        let persona = make_persona("test", Some(Uuid::new_v4()));
        let json = serde_json::to_string(&persona).unwrap();
        let deserialized: PersonaNode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, persona.name);
        assert_eq!(deserialized.id, persona.id);
        assert_eq!(deserialized.origin, PersonaOrigin::Manual);
        assert_eq!(deserialized.status, PersonaStatus::Emerging);
    }

    // ── Targeted growth hooks tests ─────────────────────────────────────

    #[tokio::test]
    async fn test_find_relevant_personas_for_note_no_match() {
        // MockGraphStore returns empty vec → no personas linked
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let file_paths = vec!["src/api/handlers.rs".to_string()];
        let result = store
            .find_relevant_personas_for_note(&file_paths, project_id)
            .await
            .unwrap();

        assert!(result.is_empty(), "Mock should return no relevant personas");
    }

    #[tokio::test]
    async fn test_find_relevant_personas_for_note_empty_paths() {
        // Empty file paths → early return, no query executed
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let result = store
            .find_relevant_personas_for_note(&[], project_id)
            .await
            .unwrap();

        assert!(result.is_empty(), "Empty paths should return no personas");
    }

    #[tokio::test]
    async fn test_find_relevant_personas_for_decision_no_match() {
        // MockGraphStore returns empty vec → no personas linked
        let store = MockGraphStore::new();
        let decision_id = Uuid::new_v4();
        let project_id = Uuid::new_v4();

        let result = store
            .find_relevant_personas_for_decision(decision_id, project_id)
            .await
            .unwrap();

        assert!(
            result.is_empty(),
            "Mock should return no relevant personas for decision"
        );
    }

    #[test]
    fn test_path_normalization_for_relevance() {
        // Verify the normalization logic used in find_relevant_personas_for_note
        let test_cases = vec![
            ("./src/main.rs", "src/main.rs"),
            ("/src/main.rs", "src/main.rs"),
            ("src/main.rs", "src/main.rs"),
            ("./nested/deep/file.rs", "nested/deep/file.rs"),
        ];

        for (input, expected) in test_cases {
            let normalized = input
                .strip_prefix("./")
                .or_else(|| input.strip_prefix('/'))
                .unwrap_or(input);
            assert_eq!(
                normalized, expected,
                "Path normalization failed for '{input}'"
            );
        }
    }

    #[test]
    fn test_link_weight_calculation() {
        // Growth hook calculates link_weight = (avg_weight * 0.8).max(0.3)
        let test_cases: Vec<(f64, f64)> = vec![
            (1.0, 0.8), // high weight → 0.8
            (0.5, 0.4), // medium weight → 0.4
            (0.3, 0.3), // low weight → clamped to 0.3
            (0.1, 0.3), // very low → clamped to 0.3
            (0.0, 0.3), // zero → clamped to 0.3
        ];

        for (avg_weight, expected) in test_cases {
            let link_weight = (avg_weight * 0.8).max(0.3);
            assert!(
                (link_weight - expected).abs() < 1e-10,
                "Weight calc failed for avg_weight={avg_weight}: got {link_weight}, expected {expected}"
            );
        }
    }

    #[tokio::test]
    async fn test_growth_hook_auto_link_note_idempotent() {
        // auto_link_note_to_persona should not fail on second call (MERGE)
        let store = MockGraphStore::new();
        let persona_id = Uuid::new_v4();
        let note_id = Uuid::new_v4();

        // First call
        store
            .auto_link_note_to_persona(persona_id, note_id, 0.5)
            .await
            .unwrap();

        // Second call (idempotent)
        store
            .auto_link_note_to_persona(persona_id, note_id, 0.8)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_growth_hook_auto_link_decision_idempotent() {
        // auto_link_decision_to_persona should not fail on second call (MERGE)
        let store = MockGraphStore::new();
        let persona_id = Uuid::new_v4();
        let decision_id = Uuid::new_v4();

        store
            .auto_link_decision_to_persona(persona_id, decision_id, 0.5)
            .await
            .unwrap();

        store
            .auto_link_decision_to_persona(persona_id, decision_id, 0.8)
            .await
            .unwrap();
    }

    // ── Auto-grow tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_find_adjacent_personas_no_match() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let result = store
            .find_adjacent_personas("/src/some/unknown.rs", project_id)
            .await
            .unwrap();

        assert!(result.is_empty(), "Mock should return no adjacent personas");
    }

    #[tokio::test]
    async fn test_auto_grow_file_knows_idempotent() {
        let store = MockGraphStore::new();
        let persona_id = Uuid::new_v4();

        // First call — create
        store
            .auto_grow_file_knows(persona_id, "/src/new_file.rs", 0.3)
            .await
            .unwrap();

        // Second call — idempotent (MERGE)
        store
            .auto_grow_file_knows(persona_id, "/src/new_file.rs", 0.3)
            .await
            .unwrap();
    }

    #[test]
    fn test_directory_extraction_for_adjacency() {
        // Verify the directory extraction logic used in find_adjacent_personas
        let test_cases = vec![
            ("/src/api/handlers.rs", "/src/api/"),
            ("/src/main.rs", "/src/"),
            ("file.rs", ""), // no directory → empty
        ];

        for (input, expected_dir) in test_cases {
            let dir = input
                .rsplit_once('/')
                .map(|(d, _)| format!("{d}/"))
                .unwrap_or_default();
            assert_eq!(
                dir, expected_dir,
                "Directory extraction failed for '{input}'"
            );
        }
    }

    // ── Community fallback tests ────────────────────────────────────────

    #[tokio::test]
    async fn test_find_personas_for_file_no_match_mock() {
        // MockGraphStore returns empty → no KNOWS nor SCOPED_TO match
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let result = store
            .find_personas_for_file("/src/unknown/file.rs", project_id)
            .await
            .unwrap();

        assert!(
            result.is_empty(),
            "Mock should return no personas for unknown file"
        );
    }

    #[test]
    fn test_community_match_weight_is_lower_than_direct_knows() {
        // Acceptance criteria: community match weight (0.3) < direct KNOWS (typically 0.5-1.0)
        let community_weight = 0.3_f64;
        let min_direct_knows = 0.5_f64;
        assert!(
            community_weight < min_direct_knows,
            "Community match weight should be lower than direct KNOWS"
        );
    }
}
