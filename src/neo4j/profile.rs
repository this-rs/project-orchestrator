//! Neo4j Analysis Profile + User Profile operations
//!
//! CRUD for:
//! - **Analysis profiles**: weight edge types for contextual analytics.
//! - **User profiles**: adaptive behavioral profiles learned from implicit signals.

use super::client::Neo4jClient;
use crate::graph::models::AnalysisProfile;
use crate::profile::{UserProfile, WorksOnRelation};
use anyhow::{bail, Result};
use chrono::{DateTime, Utc};
use neo4rs::query;
use std::collections::HashMap;
use tracing::warn;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Conversion helpers
    // ========================================================================

    /// Convert a Neo4j node to an [`AnalysisProfile`].
    fn node_to_analysis_profile(&self, node: &neo4rs::Node) -> Result<AnalysisProfile> {
        let edge_weights_json: String = node
            .get("edge_weights_json")
            .unwrap_or_else(|_| "{}".to_string());
        let edge_weights: HashMap<String, f64> = serde_json::from_str(&edge_weights_json)
            .unwrap_or_else(|e| {
                warn!(
                    json = %edge_weights_json,
                    error = %e,
                    "Failed to deserialize edge_weights, using empty map"
                );
                HashMap::new()
            });

        let fusion_weights_json: String = node
            .get("fusion_weights_json")
            .unwrap_or_else(|_| "{}".to_string());
        let fusion_weights = serde_json::from_str(&fusion_weights_json).unwrap_or_else(|e| {
            warn!(
                json = %fusion_weights_json,
                error = %e,
                "Failed to deserialize fusion_weights, using defaults"
            );
            Default::default()
        });

        Ok(AnalysisProfile {
            id: node.get("id")?,
            project_id: node
                .get::<String>("project_id")
                .ok()
                .filter(|s| !s.is_empty()),
            name: node.get("name")?,
            description: node
                .get::<String>("description")
                .ok()
                .filter(|s| !s.is_empty()),
            edge_weights,
            fusion_weights,
            is_builtin: node.get("is_builtin").unwrap_or(false),
        })
    }

    // ========================================================================
    // CRUD operations
    // ========================================================================

    /// Create or update an analysis profile.
    ///
    /// Uses MERGE on `id` for idempotent upsert (useful for built-in profiles).
    /// If `project_id` is Some, also creates a HAS_PROFILE relationship.
    pub async fn create_analysis_profile(&self, profile: &AnalysisProfile) -> Result<()> {
        let edge_weights_json = serde_json::to_string(&profile.edge_weights)?;
        let fusion_weights_json = serde_json::to_string(&profile.fusion_weights)?;

        if let Some(ref pid) = profile.project_id {
            // Project-scoped profile: MATCH project, MERGE profile, create relationship
            let q = query(
                r#"
                MATCH (p:Project {id: $project_id})
                MERGE (ap:AnalysisProfile {id: $id})
                SET ap.name = $name,
                    ap.description = $description,
                    ap.project_id = $project_id,
                    ap.edge_weights_json = $edge_weights_json,
                    ap.fusion_weights_json = $fusion_weights_json,
                    ap.is_builtin = $is_builtin
                MERGE (p)-[:HAS_PROFILE]->(ap)
                "#,
            )
            .param("id", profile.id.clone())
            .param("project_id", pid.clone())
            .param("name", profile.name.clone())
            .param(
                "description",
                profile.description.clone().unwrap_or_default(),
            )
            .param("edge_weights_json", edge_weights_json)
            .param("fusion_weights_json", fusion_weights_json)
            .param("is_builtin", profile.is_builtin);

            self.graph.run(q).await?;
        } else {
            // Global profile: no project relationship
            let q = query(
                r#"
                MERGE (ap:AnalysisProfile {id: $id})
                SET ap.name = $name,
                    ap.description = $description,
                    ap.project_id = "",
                    ap.edge_weights_json = $edge_weights_json,
                    ap.fusion_weights_json = $fusion_weights_json,
                    ap.is_builtin = $is_builtin
                "#,
            )
            .param("id", profile.id.clone())
            .param("name", profile.name.clone())
            .param(
                "description",
                profile.description.clone().unwrap_or_default(),
            )
            .param("edge_weights_json", edge_weights_json)
            .param("fusion_weights_json", fusion_weights_json)
            .param("is_builtin", profile.is_builtin);

            self.graph.run(q).await?;
        }

        Ok(())
    }

    /// List analysis profiles visible to a project.
    ///
    /// Returns global profiles (project_id = "" or IS NULL) + project-specific ones.
    pub async fn list_analysis_profiles(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<AnalysisProfile>> {
        let q = if let Some(pid) = project_id {
            query(
                r#"
                MATCH (ap:AnalysisProfile)
                WHERE ap.project_id = "" OR ap.project_id IS NULL OR ap.project_id = $project_id
                RETURN ap
                ORDER BY ap.is_builtin DESC, ap.name ASC
                "#,
            )
            .param("project_id", pid.to_string())
        } else {
            query(
                r#"
                MATCH (ap:AnalysisProfile)
                WHERE ap.project_id = "" OR ap.project_id IS NULL
                RETURN ap
                ORDER BY ap.is_builtin DESC, ap.name ASC
                "#,
            )
        };

        let mut result = self.graph.execute(q).await?;
        let mut profiles = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("ap")?;
            profiles.push(self.node_to_analysis_profile(&node)?);
        }
        Ok(profiles)
    }

    /// Get a single analysis profile by id.
    pub async fn get_analysis_profile(&self, id: &str) -> Result<Option<AnalysisProfile>> {
        let q = query(
            r#"
            MATCH (ap:AnalysisProfile {id: $id})
            RETURN ap
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("ap")?;
            Ok(Some(self.node_to_analysis_profile(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Delete an analysis profile by id.
    ///
    /// Refuses to delete built-in profiles.
    pub async fn delete_analysis_profile(&self, id: &str) -> Result<()> {
        // First check if it's built-in
        let existing = self.get_analysis_profile(id).await?;
        if let Some(ref profile) = existing {
            if profile.is_builtin {
                bail!("Cannot delete built-in profile '{}'", profile.name);
            }
        }

        let q = query(
            r#"
            MATCH (ap:AnalysisProfile {id: $id})
            DETACH DELETE ap
            "#,
        )
        .param("id", id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // UserProfile CRUD operations
    // ========================================================================

    /// Convert a Neo4j node to a [`UserProfile`].
    fn node_to_user_profile(&self, node: &neo4rs::Node) -> Result<UserProfile> {
        let created_at: String = node
            .get("created_at")
            .unwrap_or_else(|_| Utc::now().to_rfc3339());
        let updated_at: String = node
            .get("updated_at")
            .unwrap_or_else(|_| Utc::now().to_rfc3339());

        Ok(UserProfile {
            id: Uuid::parse_str(&node.get::<String>("id")?)?,
            user_id: node.get("user_id")?,
            verbosity: node.get("verbosity").unwrap_or(0.5),
            commit_style: node.get("commit_style").unwrap_or(0.5),
            language: node.get("language").unwrap_or_else(|_| "en".to_string()),
            expertise_level: node.get("expertise_level").unwrap_or(0.5),
            interaction_count: node.get::<i64>("interaction_count").unwrap_or(0) as u64,
            created_at: DateTime::parse_from_rfc3339(&created_at)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: DateTime::parse_from_rfc3339(&updated_at)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }

    /// Create or get an existing user profile.
    ///
    /// Uses MERGE on `user_id` so calling this multiple times for the same user
    /// is idempotent. Returns the profile (created or existing).
    pub async fn create_or_get_user_profile(&self, user_id: &str) -> Result<UserProfile> {
        let now = Utc::now().to_rfc3339();
        let new_id = Uuid::new_v4().to_string();

        let q = query(
            r#"
            MERGE (up:UserProfile {user_id: $user_id})
            ON CREATE SET
                up.id = $id,
                up.verbosity = 0.5,
                up.commit_style = 0.5,
                up.language = "en",
                up.expertise_level = 0.5,
                up.interaction_count = 0,
                up.created_at = $now,
                up.updated_at = $now
            RETURN up
            "#,
        )
        .param("user_id", user_id.to_string())
        .param("id", new_id)
        .param("now", now);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("up")?;
            self.node_to_user_profile(&node)
        } else {
            bail!("Failed to create or get user profile for '{}'", user_id)
        }
    }

    /// Update a user profile with new dimension values.
    ///
    /// Only updates fields that are provided (Some). Always updates `updated_at`.
    pub async fn update_user_profile(&self, profile: &UserProfile) -> Result<()> {
        let now = Utc::now().to_rfc3339();

        let q = query(
            r#"
            MATCH (up:UserProfile {user_id: $user_id})
            SET up.verbosity = $verbosity,
                up.commit_style = $commit_style,
                up.language = $language,
                up.expertise_level = $expertise_level,
                up.interaction_count = $interaction_count,
                up.updated_at = $now
            "#,
        )
        .param("user_id", profile.user_id.clone())
        .param("verbosity", profile.verbosity)
        .param("commit_style", profile.commit_style)
        .param("language", profile.language.clone())
        .param("expertise_level", profile.expertise_level)
        .param("interaction_count", profile.interaction_count as i64)
        .param("now", now);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a user profile by user_id.
    pub async fn get_user_profile(&self, user_id: &str) -> Result<Option<UserProfile>> {
        let q = query(
            r#"
            MATCH (up:UserProfile {user_id: $user_id})
            RETURN up
            "#,
        )
        .param("user_id", user_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("up")?;
            Ok(Some(self.node_to_user_profile(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Upsert a WORKS_ON relationship between a user profile and a project.
    ///
    /// Increments the frequency counter and updates last_active timestamp.
    pub async fn upsert_works_on(&self, user_id: &str, project_id: Uuid) -> Result<()> {
        let now = Utc::now().to_rfc3339();

        let q = query(
            r#"
            MATCH (up:UserProfile {user_id: $user_id})
            MATCH (p:Project {id: $project_id})
            MERGE (up)-[r:WORKS_ON]->(p)
            ON CREATE SET r.frequency = 1, r.last_active = $now
            ON MATCH SET r.frequency = r.frequency + 1, r.last_active = $now
            "#,
        )
        .param("user_id", user_id.to_string())
        .param("project_id", project_id.to_string())
        .param("now", now);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get all WORKS_ON relationships for a user.
    pub async fn get_works_on(&self, user_id: &str) -> Result<Vec<WorksOnRelation>> {
        let q = query(
            r#"
            MATCH (up:UserProfile {user_id: $user_id})-[r:WORKS_ON]->(p:Project)
            RETURN p.id AS project_id, r.frequency AS frequency, r.last_active AS last_active
            ORDER BY r.last_active DESC
            "#,
        )
        .param("user_id", user_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut relations = Vec::new();
        while let Some(row) = result.next().await? {
            let project_id_str: String = row.get("project_id")?;
            let frequency: i64 = row.get("frequency").unwrap_or(1);
            let last_active_str: String = row.get("last_active")?;

            relations.push(WorksOnRelation {
                user_id: user_id.to_string(),
                project_id: Uuid::parse_str(&project_id_str)?,
                frequency: frequency as u64,
                last_active: DateTime::parse_from_rfc3339(&last_active_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            });
        }
        Ok(relations)
    }
}
