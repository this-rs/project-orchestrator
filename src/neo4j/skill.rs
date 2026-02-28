//! Neo4j Skill operations
//!
//! Implements all CRUD, membership, query, and activation operations
//! for the Neural Skills system on the Neo4j graph.

use super::client::Neo4jClient;
use super::models::DecisionNode;
use crate::neurons::activation::{ActivatedNote, ActivationSource};
use crate::notes::Note;
use crate::skills::{
    ActivatedSkillContext, SkillNode, SkillStatus, SkillTrigger, MAX_TRIGGER_PATTERN_LEN,
    REGEX_DFA_SIZE_LIMIT, REGEX_SIZE_LIMIT,
};
use anyhow::{Context, Result};
use neo4rs::query;
use regex::RegexBuilder;
use tracing::warn;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Conversion helpers
    // ========================================================================

    /// Convert a Neo4j node to a [`SkillNode`].
    pub(crate) fn node_to_skill(node: &neo4rs::Node) -> Result<SkillNode> {
        let trigger_json: String = node
            .get("trigger_patterns_json")
            .unwrap_or_else(|_| "[]".to_string());
        let trigger_patterns: Vec<SkillTrigger> = match serde_json::from_str(&trigger_json) {
            Ok(v) => v,
            Err(e) => {
                warn!(
                    json = %trigger_json,
                    error = %e,
                    "Failed to deserialize skill trigger_patterns, using empty vec"
                );
                Vec::new()
            }
        };

        let tags: Vec<String> = node.get("tags").unwrap_or_default();

        Ok(SkillNode {
            id: node.get::<String>("id")?.parse()?,
            project_id: node.get::<String>("project_id")?.parse()?,
            name: node.get("name")?,
            description: node.get("description").unwrap_or_default(),
            status: node
                .get::<String>("status")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_default(),
            trigger_patterns,
            context_template: node
                .get::<String>("context_template")
                .ok()
                .filter(|s| !s.is_empty()),
            energy: node.get("energy").unwrap_or(0.0),
            cohesion: node.get("cohesion").unwrap_or(0.0),
            coverage: node.get("coverage").unwrap_or(0),
            note_count: node.get("note_count").unwrap_or(0),
            decision_count: node.get("decision_count").unwrap_or(0),
            activation_count: node.get("activation_count").unwrap_or(0),
            hit_rate: node.get("hit_rate").unwrap_or(0.0),
            last_activated: node
                .get::<String>("last_activated")
                .ok()
                .and_then(|s| s.parse().ok()),
            version: node.get("version").unwrap_or(1),
            fingerprint: node
                .get::<String>("fingerprint")
                .ok()
                .filter(|s| !s.is_empty()),
            imported_at: node
                .get::<String>("imported_at")
                .ok()
                .and_then(|s| s.parse().ok()),
            is_validated: node.get("is_validated").unwrap_or(false),
            tags,
            created_at: {
                let raw = node.get::<String>("created_at")?;
                raw.parse().unwrap_or_else(|e| {
                    warn!(raw = %raw, error = %e, "Failed to parse Skill created_at, using now()");
                    chrono::Utc::now()
                })
            },
            updated_at: {
                let raw = node.get::<String>("updated_at")?;
                raw.parse().unwrap_or_else(|e| {
                    warn!(raw = %raw, error = %e, "Failed to parse Skill updated_at, using now()");
                    chrono::Utc::now()
                })
            },
        })
    }

    // ========================================================================
    // CRUD operations
    // ========================================================================

    /// Create a new skill node and link it to its project atomically.
    ///
    /// Uses a single Cypher query that first MATCHes the project, then
    /// CREATEs the skill and the BELONGS_TO relationship. If the project
    /// does not exist, the MATCH produces zero rows and nothing is created,
    /// preventing orphaned Skill nodes.
    pub async fn create_skill(&self, skill: &SkillNode) -> Result<()> {
        let trigger_json = serde_json::to_string(&skill.trigger_patterns)?;

        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})
            CREATE (s:Skill {
                id: $id,
                project_id: $project_id,
                name: $name,
                description: $description,
                status: $status,
                trigger_patterns_json: $trigger_patterns_json,
                context_template: $context_template,
                energy: $energy,
                cohesion: $cohesion,
                coverage: $coverage,
                note_count: $note_count,
                decision_count: $decision_count,
                activation_count: $activation_count,
                hit_rate: $hit_rate,
                last_activated: CASE WHEN $last_activated IS NOT NULL THEN datetime($last_activated) ELSE null END,
                version: $version,
                fingerprint: $fingerprint,
                imported_at: CASE WHEN $imported_at IS NOT NULL THEN datetime($imported_at) ELSE null END,
                is_validated: $is_validated,
                tags: $tags,
                created_at: datetime($created_at),
                updated_at: datetime($updated_at)
            })
            CREATE (s)-[:BELONGS_TO]->(p)
            RETURN s.id AS created_id
            "#,
        )
        .param("id", skill.id.to_string())
        .param("project_id", skill.project_id.to_string())
        .param("name", skill.name.clone())
        .param("description", skill.description.clone())
        .param("status", skill.status.to_string())
        .param("trigger_patterns_json", trigger_json)
        .param(
            "context_template",
            skill
                .context_template
                .clone()
                .map(neo4rs::BoltType::from)
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param("energy", skill.energy)
        .param("cohesion", skill.cohesion)
        .param("coverage", skill.coverage)
        .param("note_count", skill.note_count)
        .param("decision_count", skill.decision_count)
        .param("activation_count", skill.activation_count)
        .param("hit_rate", skill.hit_rate)
        .param(
            "last_activated",
            skill
                .last_activated
                .map(|dt| neo4rs::BoltType::from(dt.to_rfc3339()))
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param("version", skill.version)
        .param(
            "fingerprint",
            skill
                .fingerprint
                .clone()
                .map(neo4rs::BoltType::from)
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param(
            "imported_at",
            skill
                .imported_at
                .map(|dt| neo4rs::BoltType::from(dt.to_rfc3339()))
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param("is_validated", skill.is_validated)
        .param("tags", skill.tags.clone())
        .param("created_at", skill.created_at.to_rfc3339())
        .param("updated_at", skill.updated_at.to_rfc3339());

        let mut result = self
            .graph
            .execute(q)
            .await
            .context("Failed to create Skill node")?;

        if result.next().await?.is_none() {
            anyhow::bail!(
                "Cannot create skill: project {} does not exist",
                skill.project_id
            );
        }

        Ok(())
    }

    /// Get a skill by ID.
    pub async fn get_skill(&self, id: Uuid) -> Result<Option<SkillNode>> {
        let q = query(
            r#"
            MATCH (s:Skill {id: $id})
            RETURN s
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            Ok(Some(Self::node_to_skill(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Update a skill node (replaces all mutable fields).
    ///
    /// Returns an error if the skill does not exist (consistent with mock behavior).
    pub async fn update_skill(&self, skill: &SkillNode) -> Result<()> {
        let trigger_json = serde_json::to_string(&skill.trigger_patterns)?;

        let q = query(
            r#"
            MATCH (s:Skill {id: $id})
            SET s.name = $name,
                s.description = $description,
                s.status = $status,
                s.trigger_patterns_json = $trigger_patterns_json,
                s.context_template = $context_template,
                s.energy = $energy,
                s.cohesion = $cohesion,
                s.coverage = $coverage,
                s.note_count = $note_count,
                s.decision_count = $decision_count,
                s.activation_count = $activation_count,
                s.hit_rate = $hit_rate,
                s.last_activated = CASE WHEN $last_activated IS NOT NULL THEN datetime($last_activated) ELSE null END,
                s.version = $version,
                s.fingerprint = $fingerprint,
                s.imported_at = CASE WHEN $imported_at IS NOT NULL THEN datetime($imported_at) ELSE null END,
                s.is_validated = $is_validated,
                s.tags = $tags,
                s.updated_at = datetime($updated_at)
            RETURN s.id AS updated_id
            "#,
        )
        .param("id", skill.id.to_string())
        .param("name", skill.name.clone())
        .param("description", skill.description.clone())
        .param("status", skill.status.to_string())
        .param("trigger_patterns_json", trigger_json)
        .param(
            "context_template",
            skill
                .context_template
                .clone()
                .map(neo4rs::BoltType::from)
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param("energy", skill.energy)
        .param("cohesion", skill.cohesion)
        .param("coverage", skill.coverage)
        .param("note_count", skill.note_count)
        .param("decision_count", skill.decision_count)
        .param("activation_count", skill.activation_count)
        .param("hit_rate", skill.hit_rate)
        .param(
            "last_activated",
            skill
                .last_activated
                .map(|dt| neo4rs::BoltType::from(dt.to_rfc3339()))
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param("version", skill.version)
        .param(
            "fingerprint",
            skill
                .fingerprint
                .clone()
                .map(neo4rs::BoltType::from)
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param(
            "imported_at",
            skill
                .imported_at
                .map(|dt| neo4rs::BoltType::from(dt.to_rfc3339()))
                .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
        )
        .param("is_validated", skill.is_validated)
        .param("tags", skill.tags.clone())
        .param("updated_at", skill.updated_at.to_rfc3339());

        let mut result = self
            .graph
            .execute(q)
            .await
            .context("Failed to update Skill node")?;

        if result.next().await?.is_none() {
            anyhow::bail!("Skill not found: {}", skill.id);
        }

        Ok(())
    }

    /// Delete a skill and all its relationships.
    ///
    /// Uses a single atomic query: OPTIONAL MATCH the skill, capture whether
    /// it existed, then DETACH DELETE. Returns true if the skill was found
    /// and deleted, false if it did not exist.
    pub async fn delete_skill(&self, id: Uuid) -> Result<bool> {
        let q = query(
            r#"
            OPTIONAL MATCH (s:Skill {id: $id})
            WITH s, CASE WHEN s IS NOT NULL THEN true ELSE false END AS existed
            DETACH DELETE s
            RETURN existed
            "#,
        )
        .param("id", id.to_string());

        let mut result = self
            .graph
            .execute(q)
            .await
            .context("Failed to delete Skill node")?;

        if let Some(row) = result.next().await? {
            Ok(row.get::<bool>("existed").unwrap_or(false))
        } else {
            Ok(false)
        }
    }

    /// List skills for a project with optional status filter and pagination.
    pub async fn list_skills(
        &self,
        project_id: Uuid,
        status: Option<SkillStatus>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<SkillNode>, usize)> {
        // Use parameterized query to avoid Cypher injection
        let status_str = status.as_ref().map(|s| s.to_string());

        // Count query
        let count_cypher = r#"
            MATCH (s:Skill {project_id: $project_id})
            WHERE ($status IS NULL OR s.status = $status)
            RETURN count(s) AS total
            "#;
        let count_q = query(count_cypher)
            .param("project_id", project_id.to_string())
            .param(
                "status",
                status_str
                    .clone()
                    .map(neo4rs::BoltType::from)
                    .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
            );
        let mut count_result = self.graph.execute(count_q).await?;
        let total: usize = if let Some(row) = count_result.next().await? {
            row.get::<i64>("total").unwrap_or(0) as usize
        } else {
            0
        };

        if total == 0 {
            return Ok((vec![], 0));
        }

        // Data query
        let data_cypher = r#"
            MATCH (s:Skill {project_id: $project_id})
            WHERE ($status IS NULL OR s.status = $status)
            RETURN s
            ORDER BY s.energy DESC, s.name ASC
            SKIP $offset
            LIMIT $limit
            "#;
        let data_q = query(data_cypher)
            .param("project_id", project_id.to_string())
            .param(
                "status",
                status_str
                    .map(neo4rs::BoltType::from)
                    .unwrap_or(neo4rs::BoltType::Null(neo4rs::BoltNull)),
            )
            .param("offset", offset as i64)
            .param("limit", limit as i64);

        let mut result = self.graph.execute(data_q).await?;
        let mut skills = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            skills.push(Self::node_to_skill(&node)?);
        }

        Ok((skills, total))
    }

    // ========================================================================
    // Membership operations
    // ========================================================================

    /// Get all member notes and decisions for a skill.
    /// Uses two separate queries to avoid cartesian product.
    pub async fn get_skill_members(
        &self,
        skill_id: Uuid,
    ) -> Result<(Vec<Note>, Vec<DecisionNode>)> {
        use crate::notes::models::{EntityType as NoteEntityType, NoteAnchor};

        // Get member notes WITH inline anchors (avoids N+1 queries).
        // OPTIONAL MATCH loads LINKED_TO targets in a single query.
        // collect() returns [] when no anchors exist (not null).
        let notes_q = query(
            r#"
            MATCH (n:Note)-[:MEMBER_OF]->(s:Skill {id: $skill_id})
            OPTIONAL MATCH (n)-[:LINKED_TO]->(e)
            WITH n, collect(DISTINCT {
                entity_type: CASE
                    WHEN e:File THEN 'file'
                    WHEN e:Function THEN 'function'
                    WHEN e:Struct THEN 'struct'
                    WHEN e:Trait THEN 'trait'
                    WHEN e:Enum THEN 'enum'
                    ELSE 'unknown'
                END,
                entity_id: coalesce(e.path, e.id, '')
            }) AS anchors
            RETURN n, anchors
            ORDER BY n.energy DESC
            LIMIT 500
            "#,
        )
        .param("skill_id", skill_id.to_string());

        let mut notes_result = self.graph.execute(notes_q).await?;
        let mut notes = Vec::new();
        while let Some(row) = notes_result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            let mut note = self.node_to_note(&node)?;

            // Parse inline anchors from the collected map list.
            // Neo4j returns collect() as a BoltList — neo4rs deserializes it
            // as serde_json::Value (array of maps).
            if let Ok(anchors_val) = row.get::<serde_json::Value>("anchors") {
                if let Some(anchor_list) = anchors_val.as_array() {
                    for anchor_val in anchor_list {
                        let entity_type_str = anchor_val.get("entity_type").and_then(|v| v.as_str()).unwrap_or("unknown");
                        let entity_id = anchor_val.get("entity_id").and_then(|v| v.as_str()).unwrap_or("");
                        // Skip empty/unknown anchors (from notes with no LINKED_TO)
                        if entity_id.is_empty() || entity_type_str == "unknown" {
                            continue;
                        }
                        let entity_type = match entity_type_str {
                            "file" => NoteEntityType::File,
                            "function" => NoteEntityType::Function,
                            "struct" => NoteEntityType::Struct,
                            "trait" => NoteEntityType::Trait,
                            "enum" => NoteEntityType::Enum,
                            _ => continue,
                        };
                        note.anchors.push(NoteAnchor::new(entity_type, entity_id.to_string()));
                    }
                }
            }

            notes.push(note);
        }

        // Get member decisions
        let decisions_q = query(
            r#"
            MATCH (d:Decision)-[:MEMBER_OF_SKILL]->(s:Skill {id: $skill_id})
            RETURN d
            ORDER BY d.decided_at DESC
            LIMIT 200
            "#,
        )
        .param("skill_id", skill_id.to_string());

        let mut decisions_result = self.graph.execute(decisions_q).await?;
        let mut decisions = Vec::new();
        while let Some(row) = decisions_result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            decisions.push(Self::node_to_decision(&node)?);
        }

        Ok((notes, decisions))
    }

    /// Add a note or decision as a member of a skill.
    ///
    /// Uses MERGE for idempotence. Returns an error if the skill or entity
    /// does not exist (the MATCH produces zero rows, so RETURN yields nothing).
    pub async fn add_skill_member(
        &self,
        skill_id: Uuid,
        entity_type: &str,
        entity_id: Uuid,
    ) -> Result<()> {
        let cypher = match entity_type {
            "note" => {
                r#"
                MATCH (s:Skill {id: $skill_id})
                MATCH (n:Note {id: $entity_id})
                MERGE (n)-[:MEMBER_OF]->(s)
                WITH s
                SET s.note_count = size([(x)-[:MEMBER_OF]->(s) WHERE x:Note | x]),
                    s.updated_at = datetime()
                RETURN s.id AS skill_id
                "#
            }
            "decision" => {
                r#"
                MATCH (s:Skill {id: $skill_id})
                MATCH (d:Decision {id: $entity_id})
                MERGE (d)-[:MEMBER_OF_SKILL]->(s)
                WITH s
                SET s.decision_count = size([(x)-[:MEMBER_OF_SKILL]->(s) WHERE x:Decision | x]),
                    s.updated_at = datetime()
                RETURN s.id AS skill_id
                "#
            }
            _ => {
                anyhow::bail!("Unknown entity type for skill membership: '{}'. Expected 'note' or 'decision'.", entity_type);
            }
        };

        let q = query(cypher)
            .param("skill_id", skill_id.to_string())
            .param("entity_id", entity_id.to_string());

        let mut result = self.graph.execute(q).await.with_context(|| {
            format!(
                "Failed to add {} {} to skill {}",
                entity_type, entity_id, skill_id
            )
        })?;

        if result.next().await?.is_none() {
            anyhow::bail!(
                "Cannot add member: skill {} or {} {} not found",
                skill_id,
                entity_type,
                entity_id
            );
        }

        Ok(())
    }

    /// Remove a member from a skill.
    ///
    /// Uses a single atomic query: OPTIONAL MATCH the relationship, delete it
    /// if found, then recount members with a WITH barrier to ensure the count
    /// is computed after the deletion. Returns whether the relationship existed.
    pub async fn remove_skill_member(
        &self,
        skill_id: Uuid,
        entity_type: &str,
        entity_id: Uuid,
    ) -> Result<bool> {
        let cypher = match entity_type {
            "note" => {
                r#"
                MATCH (s:Skill {id: $skill_id})
                OPTIONAL MATCH (n:Note {id: $entity_id})-[r:MEMBER_OF]->(s)
                WITH s, r, CASE WHEN r IS NOT NULL THEN true ELSE false END AS existed
                DELETE r
                WITH s, existed
                SET s.note_count = size([(x)-[:MEMBER_OF]->(s) WHERE x:Note | x]),
                    s.updated_at = datetime()
                RETURN existed
                "#
            }
            "decision" => {
                r#"
                MATCH (s:Skill {id: $skill_id})
                OPTIONAL MATCH (d:Decision {id: $entity_id})-[r:MEMBER_OF_SKILL]->(s)
                WITH s, r, CASE WHEN r IS NOT NULL THEN true ELSE false END AS existed
                DELETE r
                WITH s, existed
                SET s.decision_count = size([(x)-[:MEMBER_OF_SKILL]->(s) WHERE x:Decision | x]),
                    s.updated_at = datetime()
                RETURN existed
                "#
            }
            _ => {
                anyhow::bail!("Unknown entity type for skill membership: '{}'. Expected 'note' or 'decision'.", entity_type);
            }
        };

        let q = query(cypher)
            .param("skill_id", skill_id.to_string())
            .param("entity_id", entity_id.to_string());

        let mut result = self.graph.execute(q).await.with_context(|| {
            format!(
                "Failed to remove {} {} from skill {}",
                entity_type, entity_id, skill_id
            )
        })?;

        if let Some(row) = result.next().await? {
            Ok(row.get::<bool>("existed").unwrap_or(false))
        } else {
            Ok(false)
        }
    }

    // ========================================================================
    // Query operations
    // ========================================================================

    /// Remove all members (notes and decisions) from a skill.
    ///
    /// Deletes all MEMBER_OF and MEMBER_OF_SKILL relationships pointing to
    /// this skill. Resets note_count and decision_count to 0.
    /// Returns the number of relationships removed.
    pub async fn remove_all_skill_members(&self, skill_id: Uuid) -> Result<i64> {
        let q = query(
            r#"
            MATCH (s:Skill {id: $skill_id})
            OPTIONAL MATCH (n)-[r:MEMBER_OF|MEMBER_OF_SKILL]->(s)
            WITH s, collect(r) AS rels
            FOREACH (r IN rels | DELETE r)
            SET s.note_count = 0,
                s.decision_count = 0,
                s.updated_at = datetime()
            RETURN size(rels) AS removed
            "#,
        )
        .param("skill_id", skill_id.to_string());

        let mut result = self
            .graph
            .execute(q)
            .await
            .context("Failed to remove all skill members")?;

        if let Some(row) = result.next().await? {
            Ok(row.get::<i64>("removed").unwrap_or(0))
        } else {
            Ok(0)
        }
    }

    /// Get all skills that contain a given note as member.
    pub async fn get_skills_for_note(&self, note_id: Uuid) -> Result<Vec<SkillNode>> {
        let q = query(
            r#"
            MATCH (n:Note {id: $note_id})-[:MEMBER_OF]->(s:Skill)
            RETURN s
            ORDER BY s.energy DESC
            LIMIT 100
            "#,
        )
        .param("note_id", note_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut skills = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            skills.push(Self::node_to_skill(&node)?);
        }
        Ok(skills)
    }

    /// Get all skills belonging to a project.
    pub async fn get_skills_for_project(&self, project_id: Uuid) -> Result<Vec<SkillNode>> {
        let q = query(
            r#"
            MATCH (s:Skill {project_id: $project_id})
            RETURN s
            ORDER BY s.energy DESC, s.name ASC
            LIMIT 1000
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut skills = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            skills.push(Self::node_to_skill(&node)?);
        }
        Ok(skills)
    }

    // ========================================================================
    // Activation
    // ========================================================================

    /// Activate a skill: collect member notes and decisions, assemble context.
    ///
    /// Steps:
    /// 1. Retrieve the skill
    /// 2. Fetch member notes with energy > 0.05 (min activation threshold)
    /// 3. Fetch member decisions
    /// 4. Assemble context text from notes content
    /// 5. Increment activation_count and update last_activated
    pub async fn activate_skill(
        &self,
        skill_id: Uuid,
        _query_text: &str,
    ) -> Result<ActivatedSkillContext> {
        // Get the skill
        let skill = self
            .get_skill(skill_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Skill not found: {}", skill_id))?;

        // Get active member notes (energy > 0.05)
        let notes_q = query(
            r#"
            MATCH (n:Note)-[:MEMBER_OF]->(s:Skill {id: $skill_id})
            WHERE n.energy > 0.05
            RETURN n
            ORDER BY n.energy DESC
            LIMIT 500
            "#,
        )
        .param("skill_id", skill_id.to_string());

        let mut notes_result = self.graph.execute(notes_q).await?;
        let mut activated_notes = Vec::new();
        while let Some(row) = notes_result.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            let note = self.node_to_note(&node)?;
            let energy = note.energy;
            activated_notes.push(ActivatedNote {
                note,
                activation_score: energy, // Use energy as activation proxy
                source: ActivationSource::Direct,
                entity_type: "note".to_string(),
            });
        }

        // Get member decisions
        let decisions_q = query(
            r#"
            MATCH (d:Decision)-[:MEMBER_OF_SKILL]->(s:Skill {id: $skill_id})
            RETURN d
            ORDER BY d.decided_at DESC
            LIMIT 200
            "#,
        )
        .param("skill_id", skill_id.to_string());

        let mut decisions_result = self.graph.execute(decisions_q).await?;
        let mut relevant_decisions = Vec::new();
        while let Some(row) = decisions_result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            relevant_decisions.push(Self::node_to_decision(&node)?);
        }

        // Assemble context text
        let mut context_parts = Vec::new();
        context_parts.push(format!("## Skill: {}", skill.name));
        if !skill.description.is_empty() {
            context_parts.push(skill.description.clone());
        }
        context_parts.push(String::new());

        if !activated_notes.is_empty() {
            context_parts.push("### Notes".to_string());
            for an in &activated_notes {
                context_parts.push(format!(
                    "- **[{}]** {}: {}",
                    an.note.importance,
                    an.note.note_type,
                    an.note.content.chars().take(200).collect::<String>()
                ));
            }
            context_parts.push(String::new());
        }

        if !relevant_decisions.is_empty() {
            context_parts.push("### Decisions".to_string());
            for dec in &relevant_decisions {
                context_parts.push(format!(
                    "- **{}**: {} (chose: {})",
                    dec.description,
                    dec.rationale,
                    dec.chosen_option.as_deref().unwrap_or("N/A")
                ));
            }
        }

        let context_text = context_parts.join("\n");

        // Update activation tracking
        let update_q = query(
            r#"
            MATCH (s:Skill {id: $skill_id})
            SET s.activation_count = s.activation_count + 1,
                s.last_activated = datetime(),
                s.updated_at = datetime()
            "#,
        )
        .param("skill_id", skill_id.to_string());
        let _ = self.graph.run(update_q).await; // Best-effort

        Ok(ActivatedSkillContext {
            skill,
            activated_notes,
            relevant_decisions,
            context_text,
            confidence: 1.0,
        })
    }

    /// Increment a skill's activation_count and update last_activated.
    /// Best-effort, fire-and-forget — used by hook activation paths.
    pub async fn increment_skill_activation(&self, skill_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (s:Skill {id: $skill_id})
            SET s.activation_count = s.activation_count + 1,
                s.last_activated = datetime(),
                s.updated_at = datetime()
            "#,
        )
        .param("skill_id", skill_id.to_string());
        let _ = self.graph.run(q).await; // Best-effort
        Ok(())
    }

    /// Match skills by evaluating trigger patterns against an input string.
    ///
    /// Only matches Active/Emerging skills. Triggers with quality_score < 0.3
    /// are skipped. Returns matches sorted by confidence descending.
    pub async fn match_skills_by_trigger(
        &self,
        project_id: Uuid,
        input: &str,
    ) -> Result<Vec<(SkillNode, f64)>> {
        // Get all matchable skills for the project
        let q = query(
            r#"
            MATCH (s:Skill {project_id: $project_id})
            WHERE s.status IN ['active', 'emerging']
            RETURN s
            LIMIT 500
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut candidates = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            candidates.push(Self::node_to_skill(&node)?);
        }

        // Evaluate triggers
        let mut matches: Vec<(SkillNode, f64)> = Vec::new();

        for skill in candidates {
            let mut best_confidence = 0.0f64;

            for trigger in &skill.trigger_patterns {
                // Skip unreliable triggers
                if !trigger.is_reliable() {
                    continue;
                }

                let confidence = match trigger.pattern_type {
                    crate::skills::TriggerType::Regex => {
                        // Use RegexBuilder with size limits (ReDoS protection)
                        // and case-insensitive matching (consistent with activation.rs,
                        // cache.rs, and triggers.rs quality evaluation)
                        if trigger.pattern_value.len() > MAX_TRIGGER_PATTERN_LEN {
                            0.0
                        } else {
                            match RegexBuilder::new(&trigger.pattern_value)
                                .case_insensitive(true)
                                .size_limit(REGEX_SIZE_LIMIT)
                                .dfa_size_limit(REGEX_DFA_SIZE_LIMIT)
                                .build()
                            {
                                Ok(re) => {
                                    if re.is_match(input) {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                }
                                Err(_) => 0.0,
                            }
                        }
                    }
                    crate::skills::TriggerType::FileGlob => {
                        match glob::Pattern::new(&trigger.pattern_value) {
                            Ok(pattern) => {
                                if pattern.matches(input) {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            Err(_) => 0.0,
                        }
                    }
                    crate::skills::TriggerType::McpAction => {
                        if crate::skills::activation::match_mcp_action_trigger(
                            &trigger.pattern_value,
                            input,
                        ) {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    crate::skills::TriggerType::Semantic => {
                        // Semantic matching requires embedding provider — skip in graph layer
                        // This will be handled by the skill manager/orchestrator layer
                        0.0
                    }
                };

                if confidence >= trigger.confidence_threshold {
                    best_confidence = best_confidence.max(confidence);
                }
            }

            if best_confidence > 0.0 {
                // Apply confidence penalty for Emerging skills (unproven)
                let adjusted_confidence = if skill.status == crate::skills::SkillStatus::Emerging {
                    best_confidence * 0.8
                } else {
                    best_confidence
                };
                matches.push((skill, adjusted_confidence));
            }
        }

        // Sort by confidence descending
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(matches)
    }

    // ========================================================================
    // Skill Detection — Graph Extraction
    // ========================================================================

    /// Extract the SYNAPSE graph for a project.
    ///
    /// Returns all SYNAPSE edges between notes belonging to the given project,
    /// filtered by minimum weight. Each edge is returned as (from_note_id, to_note_id, weight).
    pub async fn get_synapse_graph(
        &self,
        project_id: Uuid,
        min_weight: f64,
    ) -> Result<Vec<(String, String, f64)>> {
        let q = query(
            "MATCH (n1:Note)-[s:SYNAPSE]->(n2:Note)
             WHERE n1.project_id = $project_id
               AND n2.project_id = $project_id
               AND s.weight > $min_weight
             RETURN n1.id AS from_id, n2.id AS to_id, s.weight AS weight
             LIMIT 10000",
        )
        .param("project_id", project_id.to_string())
        .param("min_weight", min_weight);

        let mut result = self
            .graph
            .execute(q)
            .await
            .context("Failed to execute get_synapse_graph query")?;

        let mut edges = Vec::new();
        while let Some(row) = result.next().await? {
            let from_id: String = row.get("from_id").unwrap_or_default();
            let to_id: String = row.get("to_id").unwrap_or_default();
            let weight: f64 = row.get("weight").unwrap_or(0.0);
            edges.push((from_id, to_id, weight));
        }

        Ok(edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::{SkillTrigger, TriggerType};

    #[test]
    fn test_node_to_skill_trigger_serialization() {
        // Verify that trigger patterns survive JSON round-trip
        let triggers = vec![
            SkillTrigger::regex("neo4j|cypher", 0.7),
            SkillTrigger::file_glob("src/neo4j/**", 0.8),
        ];
        let json = serde_json::to_string(&triggers).unwrap();
        let deserialized: Vec<SkillTrigger> = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.len(), 2);
        assert_eq!(deserialized[0].pattern_type, TriggerType::Regex);
        assert_eq!(deserialized[1].pattern_type, TriggerType::FileGlob);
    }

    #[test]
    fn test_trigger_regex_matching() {
        let re = regex::Regex::new("neo4j|cypher|UNWIND").unwrap();
        assert!(re.is_match("UNWIND $items AS item"));
        assert!(re.is_match("neo4j client"));
        assert!(!re.is_match("python django"));
    }

    #[test]
    fn test_trigger_glob_matching() {
        let pattern = glob::Pattern::new("src/neo4j/**").unwrap();
        assert!(pattern.matches("src/neo4j/skill.rs"));
        assert!(pattern.matches("src/neo4j/client.rs"));
        assert!(!pattern.matches("src/api/handlers.rs"));
    }

    #[test]
    fn test_trigger_reliability() {
        let mut trigger = SkillTrigger::regex("test", 0.5);
        assert!(trigger.is_reliable()); // None quality = reliable by default

        trigger.quality_score = Some(0.5);
        assert!(trigger.is_reliable()); // 0.5 >= 0.3

        trigger.quality_score = Some(0.2);
        assert!(!trigger.is_reliable()); // 0.2 < 0.3
    }

    #[test]
    fn test_skill_status_display_roundtrip() {
        for status in [
            SkillStatus::Emerging,
            SkillStatus::Active,
            SkillStatus::Dormant,
            SkillStatus::Archived,
            SkillStatus::Imported,
        ] {
            let s = status.to_string();
            let parsed: SkillStatus = s.parse().unwrap();
            assert_eq!(parsed, status);
        }
    }
}
