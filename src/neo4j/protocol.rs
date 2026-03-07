//! Neo4j Protocol operations (Pattern Federation)
//!
//! Implements CRUD operations for Protocol, ProtocolState, and ProtocolTransition
//! entities on the Neo4j graph.

use super::client::Neo4jClient;
use crate::protocol::{
    Protocol, ProtocolCategory, ProtocolRun, ProtocolState, ProtocolTransition, RunStatus,
    StateVisit,
};
use anyhow::{Context, Result};
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Conversion helpers
    // ========================================================================

    /// Convert a Neo4j node to a [`Protocol`].
    pub(crate) fn node_to_protocol(node: &neo4rs::Node) -> Result<Protocol> {
        let terminal_states_json: String = node
            .get("terminal_states_json")
            .unwrap_or_else(|_| "[]".to_string());
        let terminal_states: Vec<Uuid> =
            serde_json::from_str(&terminal_states_json).unwrap_or_default();

        Ok(Protocol {
            id: node.get::<String>("id")?.parse()?,
            name: node.get("name")?,
            description: node.get("description").unwrap_or_default(),
            project_id: node.get::<String>("project_id")?.parse()?,
            skill_id: node
                .get::<String>("skill_id")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            entry_state: node.get::<String>("entry_state")?.parse()?,
            terminal_states,
            protocol_category: node
                .get::<String>("protocol_category")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_default(),
            trigger_mode: node
                .get::<String>("trigger_mode")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_default(),
            trigger_config: node
                .get::<String>("trigger_config_json")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| serde_json::from_str(&s).ok()),
            relevance_vector: node
                .get::<String>("relevance_vector_json")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| serde_json::from_str(&s).ok()),
            last_triggered_at: node
                .get::<String>("last_triggered_at")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            created_at: {
                let raw = node.get::<String>("created_at")?;
                raw.parse().unwrap_or_else(|_| chrono::Utc::now())
            },
            updated_at: {
                let raw = node.get::<String>("updated_at")?;
                raw.parse().unwrap_or_else(|_| chrono::Utc::now())
            },
        })
    }

    /// Convert a Neo4j node to a [`ProtocolState`].
    pub(crate) fn node_to_protocol_state(node: &neo4rs::Node) -> Result<ProtocolState> {
        Ok(ProtocolState {
            id: node.get::<String>("id")?.parse()?,
            protocol_id: node.get::<String>("protocol_id")?.parse()?,
            name: node.get("name")?,
            description: node.get("description").unwrap_or_default(),
            action: node.get::<String>("action").ok().filter(|s| !s.is_empty()),
            state_type: node
                .get::<String>("state_type")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_default(),
        })
    }

    /// Convert a Neo4j node to a [`ProtocolTransition`].
    pub(crate) fn node_to_protocol_transition(node: &neo4rs::Node) -> Result<ProtocolTransition> {
        Ok(ProtocolTransition {
            id: node.get::<String>("id")?.parse()?,
            protocol_id: node.get::<String>("protocol_id")?.parse()?,
            from_state: node.get::<String>("from_state")?.parse()?,
            to_state: node.get::<String>("to_state")?.parse()?,
            trigger: node.get("trigger")?,
            guard: node.get::<String>("guard").ok().filter(|s| !s.is_empty()),
        })
    }

    // ========================================================================
    // Protocol CRUD
    // ========================================================================

    /// Create or update a protocol node and link it to its project.
    pub async fn upsert_protocol(&self, protocol: &Protocol) -> Result<()> {
        let terminal_states_json = serde_json::to_string(&protocol.terminal_states)?;

        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})
            MERGE (proto:Protocol {id: $id})
            ON CREATE SET proto.created_at = $created_at
            SET proto.name = $name,
                proto.description = $description,
                proto.project_id = $project_id,
                proto.skill_id = $skill_id,
                proto.entry_state = $entry_state,
                proto.terminal_states_json = $terminal_states_json,
                proto.protocol_category = $protocol_category,
                proto.trigger_mode = $trigger_mode,
                proto.trigger_config_json = $trigger_config_json,
                proto.relevance_vector_json = $relevance_vector_json,
                proto.last_triggered_at = $last_triggered_at,
                proto.updated_at = $updated_at
            MERGE (proto)-[:BELONGS_TO]->(p)
            RETURN proto.id AS created_id
            "#,
        )
        .param("id", protocol.id.to_string())
        .param("project_id", protocol.project_id.to_string())
        .param("name", protocol.name.clone())
        .param("description", protocol.description.clone())
        .param(
            "skill_id",
            protocol.skill_id.map(|u| u.to_string()).unwrap_or_default(),
        )
        .param("entry_state", protocol.entry_state.to_string())
        .param("terminal_states_json", terminal_states_json)
        .param("protocol_category", protocol.protocol_category.to_string())
        .param("trigger_mode", protocol.trigger_mode.to_string())
        .param(
            "trigger_config_json",
            protocol
                .trigger_config
                .as_ref()
                .map(|c| serde_json::to_string(c).unwrap_or_default())
                .unwrap_or_default(),
        )
        .param(
            "relevance_vector_json",
            protocol
                .relevance_vector
                .as_ref()
                .map(|rv| serde_json::to_string(rv).unwrap_or_default())
                .unwrap_or_default(),
        )
        .param(
            "last_triggered_at",
            protocol
                .last_triggered_at
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default(),
        )
        .param("created_at", protocol.created_at.to_rfc3339())
        .param("updated_at", protocol.updated_at.to_rfc3339());

        let _ = self
            .graph
            .execute(q)
            .await
            .context("Failed to upsert protocol")?;

        // Link to skill if present
        if let Some(skill_id) = &protocol.skill_id {
            let link_q = query(
                r#"
                MATCH (proto:Protocol {id: $protocol_id})
                MATCH (s:Skill {id: $skill_id})
                MERGE (proto)-[:BELONGS_TO_SKILL]->(s)
                "#,
            )
            .param("protocol_id", protocol.id.to_string())
            .param("skill_id", skill_id.to_string());

            // Best-effort: don't fail if skill doesn't exist
            let _ = self.graph.execute(link_q).await;
        }

        Ok(())
    }

    /// Get a protocol by ID.
    pub async fn get_protocol(&self, id: Uuid) -> Result<Option<Protocol>> {
        let q = query(
            r#"
            MATCH (proto:Protocol {id: $id})
            RETURN proto
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("proto")?;
            Ok(Some(Self::node_to_protocol(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List protocols for a project with optional category filter and pagination.
    pub async fn list_protocols(
        &self,
        project_id: Uuid,
        category: Option<ProtocolCategory>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Protocol>, usize)> {
        // Count query
        let count_cypher = if category.is_some() {
            r#"
            MATCH (proto:Protocol {project_id: $project_id, protocol_category: $category})
            RETURN count(proto) AS total
            "#
        } else {
            r#"
            MATCH (proto:Protocol {project_id: $project_id})
            RETURN count(proto) AS total
            "#
        };

        let mut count_q = query(count_cypher).param("project_id", project_id.to_string());
        if let Some(ref cat) = category {
            count_q = count_q.param("category", cat.to_string());
        }

        let mut count_result = self.graph.execute(count_q).await?;
        let total: usize = if let Some(row) = count_result.next().await? {
            row.get::<i64>("total").unwrap_or(0) as usize
        } else {
            0
        };

        if total == 0 {
            return Ok((vec![], 0));
        }

        // List query
        let list_cypher = if category.is_some() {
            r#"
            MATCH (proto:Protocol {project_id: $project_id, protocol_category: $category})
            RETURN proto
            ORDER BY proto.updated_at DESC
            SKIP $offset LIMIT $limit
            "#
        } else {
            r#"
            MATCH (proto:Protocol {project_id: $project_id})
            RETURN proto
            ORDER BY proto.updated_at DESC
            SKIP $offset LIMIT $limit
            "#
        };

        let mut list_q = query(list_cypher)
            .param("project_id", project_id.to_string())
            .param("offset", offset as i64)
            .param("limit", limit as i64);
        if let Some(ref cat) = category {
            list_q = list_q.param("category", cat.to_string());
        }

        let mut result = self.graph.execute(list_q).await?;
        let mut protocols = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("proto")?;
            protocols.push(Self::node_to_protocol(&node)?);
        }

        Ok((protocols, total))
    }

    /// Delete a protocol and all its states, transitions, and relationships.
    pub async fn delete_protocol(&self, id: Uuid) -> Result<bool> {
        // First check if it exists
        let check_q = query(
            r#"
            MATCH (proto:Protocol {id: $id})
            RETURN proto.id AS found_id
            "#,
        )
        .param("id", id.to_string());

        let mut check_result = self.graph.execute(check_q).await?;
        let exists = check_result.next().await?.is_some();

        if !exists {
            return Ok(false);
        }

        // Delete runs, states, transitions, then the protocol
        let delete_q = query(
            r#"
            MATCH (proto:Protocol {id: $id})
            OPTIONAL MATCH (proto)-[:HAS_STATE]->(s:ProtocolState)
            OPTIONAL MATCH (proto)-[:HAS_TRANSITION]->(t:ProtocolTransition)
            OPTIONAL MATCH (proto)<-[:INSTANCE_OF]-(r:ProtocolRun)
            DETACH DELETE s, t, r, proto
            "#,
        )
        .param("id", id.to_string());

        let _ = self
            .graph
            .execute(delete_q)
            .await
            .context("Failed to delete protocol")?;

        Ok(true)
    }

    // ========================================================================
    // ProtocolState CRUD
    // ========================================================================

    /// Upsert a protocol state and create its HAS_STATE relationship.
    pub async fn upsert_protocol_state(&self, state: &ProtocolState) -> Result<()> {
        let q = query(
            r#"
            MATCH (proto:Protocol {id: $protocol_id})
            MERGE (s:ProtocolState {id: $id})
            SET s.protocol_id = $protocol_id,
                s.name = $name,
                s.description = $description,
                s.action = $action,
                s.state_type = $state_type
            MERGE (proto)-[:HAS_STATE]->(s)
            RETURN s.id AS created_id
            "#,
        )
        .param("id", state.id.to_string())
        .param("protocol_id", state.protocol_id.to_string())
        .param("name", state.name.clone())
        .param("description", state.description.clone())
        .param("action", state.action.clone().unwrap_or_default())
        .param("state_type", state.state_type.to_string());

        let _ = self
            .graph
            .execute(q)
            .await
            .context("Failed to upsert protocol state")?;

        Ok(())
    }

    /// Get all states for a protocol.
    pub async fn get_protocol_states(&self, protocol_id: Uuid) -> Result<Vec<ProtocolState>> {
        let q = query(
            r#"
            MATCH (s:ProtocolState {protocol_id: $protocol_id})
            RETURN s
            ORDER BY s.state_type, s.name
            "#,
        )
        .param("protocol_id", protocol_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut states = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("s")?;
            states.push(Self::node_to_protocol_state(&node)?);
        }

        Ok(states)
    }

    /// Delete a protocol state and its relationships.
    pub async fn delete_protocol_state(&self, state_id: Uuid) -> Result<bool> {
        let check_q = query(
            r#"
            MATCH (s:ProtocolState {id: $id})
            RETURN s.id AS found_id
            "#,
        )
        .param("id", state_id.to_string());

        let mut check_result = self.graph.execute(check_q).await?;
        let exists = check_result.next().await?.is_some();

        if !exists {
            return Ok(false);
        }

        let delete_q = query(
            r#"
            MATCH (s:ProtocolState {id: $id})
            DETACH DELETE s
            "#,
        )
        .param("id", state_id.to_string());

        let _ = self
            .graph
            .execute(delete_q)
            .await
            .context("Failed to delete protocol state")?;

        Ok(true)
    }

    // ========================================================================
    // ProtocolTransition CRUD
    // ========================================================================

    /// Upsert a protocol transition and create its HAS_TRANSITION relationship.
    pub async fn upsert_protocol_transition(&self, transition: &ProtocolTransition) -> Result<()> {
        let q = query(
            r#"
            MATCH (proto:Protocol {id: $protocol_id})
            MERGE (t:ProtocolTransition {id: $id})
            SET t.protocol_id = $protocol_id,
                t.from_state = $from_state,
                t.to_state = $to_state,
                t.trigger = $trigger,
                t.guard = $guard
            MERGE (proto)-[:HAS_TRANSITION]->(t)
            RETURN t.id AS created_id
            "#,
        )
        .param("id", transition.id.to_string())
        .param("protocol_id", transition.protocol_id.to_string())
        .param("from_state", transition.from_state.to_string())
        .param("to_state", transition.to_state.to_string())
        .param("trigger", transition.trigger.clone())
        .param("guard", transition.guard.clone().unwrap_or_default());

        let _ = self
            .graph
            .execute(q)
            .await
            .context("Failed to upsert protocol transition")?;

        Ok(())
    }

    /// Get all transitions for a protocol.
    pub async fn get_protocol_transitions(
        &self,
        protocol_id: Uuid,
    ) -> Result<Vec<ProtocolTransition>> {
        let q = query(
            r#"
            MATCH (t:ProtocolTransition {protocol_id: $protocol_id})
            RETURN t
            ORDER BY t.trigger
            "#,
        )
        .param("protocol_id", protocol_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut transitions = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            transitions.push(Self::node_to_protocol_transition(&node)?);
        }

        Ok(transitions)
    }

    /// Delete a protocol transition.
    pub async fn delete_protocol_transition(&self, transition_id: Uuid) -> Result<bool> {
        let check_q = query(
            r#"
            MATCH (t:ProtocolTransition {id: $id})
            RETURN t.id AS found_id
            "#,
        )
        .param("id", transition_id.to_string());

        let mut check_result = self.graph.execute(check_q).await?;
        let exists = check_result.next().await?.is_some();

        if !exists {
            return Ok(false);
        }

        let delete_q = query(
            r#"
            MATCH (t:ProtocolTransition {id: $id})
            DETACH DELETE t
            "#,
        )
        .param("id", transition_id.to_string());

        let _ = self
            .graph
            .execute(delete_q)
            .await
            .context("Failed to delete protocol transition")?;

        Ok(true)
    }

    // ========================================================================
    // ProtocolRun CRUD (FSM Runtime)
    // ========================================================================

    /// Convert a Neo4j node to a [`ProtocolRun`].
    pub(crate) fn node_to_protocol_run(node: &neo4rs::Node) -> Result<ProtocolRun> {
        let states_visited_json: String = node
            .get("states_visited_json")
            .unwrap_or_else(|_| "[]".to_string());
        let states_visited: Vec<StateVisit> =
            serde_json::from_str(&states_visited_json).unwrap_or_default();

        Ok(ProtocolRun {
            id: node.get::<String>("id")?.parse()?,
            protocol_id: node.get::<String>("protocol_id")?.parse()?,
            plan_id: node
                .get::<String>("plan_id")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            task_id: node
                .get::<String>("task_id")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            current_state: node.get::<String>("current_state")?.parse()?,
            states_visited,
            status: node
                .get::<String>("status")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_default(),
            started_at: {
                let raw = node.get::<String>("started_at")?;
                raw.parse().unwrap_or_else(|_| chrono::Utc::now())
            },
            completed_at: node
                .get::<String>("completed_at")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            error: node.get::<String>("error").ok().filter(|s| !s.is_empty()),
            triggered_by: node
                .get::<String>("triggered_by")
                .unwrap_or_else(|_| "manual".to_string()),
        })
    }

    /// Create a new protocol run node with INSTANCE_OF relationship to its protocol.
    ///
    /// # Atomic concurrency guard
    ///
    /// The Cypher uses a conditional pattern: the CREATE only executes when no
    /// existing `ProtocolRun` with `status = 'running'` exists for the same
    /// protocol. If a concurrent run is already running, the query returns no
    /// rows and this method returns an error.
    pub async fn create_protocol_run(&self, run: &ProtocolRun) -> Result<()> {
        let states_visited_json = serde_json::to_string(&run.states_visited)?;

        let q = query(
            r#"
            MATCH (proto:Protocol {id: $protocol_id})
            WHERE NOT EXISTS {
                MATCH (existing:ProtocolRun {protocol_id: $protocol_id, status: 'running'})
            }
            CREATE (r:ProtocolRun {
                id: $id,
                protocol_id: $protocol_id,
                plan_id: $plan_id,
                task_id: $task_id,
                current_state: $current_state,
                states_visited_json: $states_visited_json,
                status: $status,
                started_at: $started_at,
                completed_at: $completed_at,
                error: $error,
                triggered_by: $triggered_by
            })
            CREATE (r)-[:INSTANCE_OF]->(proto)
            RETURN r.id AS created_id
            "#,
        )
        .param("id", run.id.to_string())
        .param("protocol_id", run.protocol_id.to_string())
        .param(
            "plan_id",
            run.plan_id.map(|u| u.to_string()).unwrap_or_default(),
        )
        .param(
            "task_id",
            run.task_id.map(|u| u.to_string()).unwrap_or_default(),
        )
        .param("current_state", run.current_state.to_string())
        .param("states_visited_json", states_visited_json)
        .param("status", run.status.to_string())
        .param("started_at", run.started_at.to_rfc3339())
        .param(
            "completed_at",
            run.completed_at
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default(),
        )
        .param("error", run.error.clone().unwrap_or_default())
        .param("triggered_by", run.triggered_by.clone());

        let mut result = self
            .graph
            .execute(q)
            .await
            .context("Failed to create protocol run")?;

        // If no rows returned, the WHERE NOT EXISTS clause prevented creation
        // (a running run already exists for this protocol).
        let row = result
            .next()
            .await
            .context("Failed to read create_protocol_run result")?;
        if row.is_none() {
            anyhow::bail!(
                "Skipped: concurrent run already running for protocol {}",
                run.protocol_id
            );
        }

        // Link to plan if present
        if let Some(plan_id) = &run.plan_id {
            let link_q = query(
                r#"
                MATCH (r:ProtocolRun {id: $run_id})
                MATCH (p:Plan {id: $plan_id})
                MERGE (r)-[:LINKED_TO_PLAN]->(p)
                "#,
            )
            .param("run_id", run.id.to_string())
            .param("plan_id", plan_id.to_string());
            let _ = self.graph.execute(link_q).await;
        }

        // Link to task if present
        if let Some(task_id) = &run.task_id {
            let link_q = query(
                r#"
                MATCH (r:ProtocolRun {id: $run_id})
                MATCH (t:Task {id: $task_id})
                MERGE (r)-[:LINKED_TO_TASK]->(t)
                "#,
            )
            .param("run_id", run.id.to_string())
            .param("task_id", task_id.to_string());
            let _ = self.graph.execute(link_q).await;
        }

        Ok(())
    }

    /// Get a protocol run by ID.
    pub async fn get_protocol_run(&self, run_id: Uuid) -> Result<Option<ProtocolRun>> {
        let q = query(
            r#"
            MATCH (r:ProtocolRun {id: $id})
            RETURN r
            "#,
        )
        .param("id", run_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            Ok(Some(Self::node_to_protocol_run(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Update an existing protocol run.
    pub async fn update_protocol_run(&self, run: &ProtocolRun) -> Result<()> {
        let states_visited_json = serde_json::to_string(&run.states_visited)?;

        let q = query(
            r#"
            MATCH (r:ProtocolRun {id: $id})
            SET r.current_state = $current_state,
                r.states_visited_json = $states_visited_json,
                r.status = $status,
                r.completed_at = $completed_at,
                r.error = $error
            RETURN r.id AS updated_id
            "#,
        )
        .param("id", run.id.to_string())
        .param("current_state", run.current_state.to_string())
        .param("states_visited_json", states_visited_json)
        .param("status", run.status.to_string())
        .param(
            "completed_at",
            run.completed_at
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default(),
        )
        .param("error", run.error.clone().unwrap_or_default());

        let _ = self
            .graph
            .execute(q)
            .await
            .context("Failed to update protocol run")?;

        Ok(())
    }

    /// List protocol runs for a protocol with optional status filter and pagination.
    pub async fn list_protocol_runs(
        &self,
        protocol_id: Uuid,
        status: Option<RunStatus>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<ProtocolRun>, usize)> {
        // Count query
        let count_cypher = if status.is_some() {
            r#"
            MATCH (r:ProtocolRun {protocol_id: $protocol_id, status: $status})
            RETURN count(r) AS total
            "#
        } else {
            r#"
            MATCH (r:ProtocolRun {protocol_id: $protocol_id})
            RETURN count(r) AS total
            "#
        };

        let mut count_q = query(count_cypher).param("protocol_id", protocol_id.to_string());
        if let Some(ref s) = status {
            count_q = count_q.param("status", s.to_string());
        }

        let mut count_result = self.graph.execute(count_q).await?;
        let total: usize = if let Some(row) = count_result.next().await? {
            row.get::<i64>("total").unwrap_or(0) as usize
        } else {
            0
        };

        if total == 0 {
            return Ok((vec![], 0));
        }

        // List query
        let list_cypher = if status.is_some() {
            r#"
            MATCH (r:ProtocolRun {protocol_id: $protocol_id, status: $status})
            RETURN r
            ORDER BY r.started_at DESC
            SKIP $offset LIMIT $limit
            "#
        } else {
            r#"
            MATCH (r:ProtocolRun {protocol_id: $protocol_id})
            RETURN r
            ORDER BY r.started_at DESC
            SKIP $offset LIMIT $limit
            "#
        };

        let mut list_q = query(list_cypher)
            .param("protocol_id", protocol_id.to_string())
            .param("offset", offset as i64)
            .param("limit", limit as i64);
        if let Some(ref s) = status {
            list_q = list_q.param("status", s.to_string());
        }

        let mut result = self.graph.execute(list_q).await?;
        let mut runs = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            runs.push(Self::node_to_protocol_run(&node)?);
        }

        Ok((runs, total))
    }

    /// Delete a protocol run.
    pub async fn delete_protocol_run(&self, run_id: Uuid) -> Result<bool> {
        let check_q = query(
            r#"
            MATCH (r:ProtocolRun {id: $id})
            RETURN r.id AS found_id
            "#,
        )
        .param("id", run_id.to_string());

        let mut check_result = self.graph.execute(check_q).await?;
        let exists = check_result.next().await?.is_some();

        if !exists {
            return Ok(false);
        }

        let delete_q = query(
            r#"
            MATCH (r:ProtocolRun {id: $id})
            DETACH DELETE r
            "#,
        )
        .param("id", run_id.to_string());

        let _ = self
            .graph
            .execute(delete_q)
            .await
            .context("Failed to delete protocol run")?;

        Ok(true)
    }
}
