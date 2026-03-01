//! Neo4j Analysis Profile operations
//!
//! CRUD for analysis profiles that weight edge types for contextual analytics.
//! Profiles can be global (project_id IS NULL) or project-scoped.

use super::client::Neo4jClient;
use crate::graph::models::AnalysisProfile;
use anyhow::{bail, Result};
use neo4rs::query;
use std::collections::HashMap;
use tracing::warn;

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
            project_id: node.get::<String>("project_id").ok().filter(|s| !s.is_empty()),
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
}
