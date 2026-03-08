//! Neo4j persistence for the Skill Registry (PublishedSkill nodes)

use anyhow::{Context, Result};
use neo4rs::*;
use uuid::Uuid;

use crate::skills::registry::PublishedSkill;

impl super::client::Neo4jClient {
    /// Create or update a PublishedSkill node.
    ///
    /// The package is stored as a JSON string in the `package_json` property.
    /// Trust components are stored as separate properties for query filtering.
    pub async fn upsert_published_skill(&self, published: &PublishedSkill) -> Result<()> {
        let package_json = serde_json::to_string(&published.package)
            .context("Failed to serialize SkillPackage")?;
        let trust_components_json = serde_json::to_string(&published.trust_components)
            .context("Failed to serialize trust components")?;
        let tags_json =
            serde_json::to_string(&published.tags).context("Failed to serialize tags")?;

        let query = query(
            "MERGE (ps:PublishedSkill {id: $id})
             SET ps.name = $name,
                 ps.description = $description,
                 ps.tags = $tags_json,
                 ps.package_json = $package_json,
                 ps.trust_score = $trust_score,
                 ps.trust_components_json = $trust_components_json,
                 ps.trust_level = $trust_level,
                 ps.source_project_id = $source_project_id,
                 ps.source_project_name = $source_project_name,
                 ps.published_by = $published_by,
                 ps.published_at = datetime($published_at),
                 ps.import_count = $import_count",
        )
        .param("id", published.id.to_string())
        .param("name", published.name.clone())
        .param("description", published.description.clone())
        .param("tags_json", tags_json)
        .param("package_json", package_json)
        .param("trust_score", published.trust_score)
        .param("trust_components_json", trust_components_json)
        .param("trust_level", published.trust_level.to_string())
        .param("source_project_id", published.source_project_id.to_string())
        .param("source_project_name", published.source_project_name.clone())
        .param("published_by", published.published_by.clone())
        .param("published_at", published.published_at.to_rfc3339())
        .param("import_count", published.import_count);

        self.graph.run(query).await?;
        Ok(())
    }

    /// Get a published skill by ID.
    pub async fn get_published_skill(&self, id: Uuid) -> Result<Option<PublishedSkill>> {
        let query = query(
            "MATCH (ps:PublishedSkill {id: $id})
             RETURN ps",
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(query).await?;

        if let Some(row) = result.next().await? {
            let node: Node = row.get("ps")?;
            let published = parse_published_skill_node(&node)?;
            Ok(Some(published))
        } else {
            Ok(None)
        }
    }

    /// Search published skills with optional filters.
    pub async fn search_published_skills(
        &self,
        search_query: Option<&str>,
        min_trust: Option<f64>,
        tags: Option<&[String]>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<PublishedSkill>, usize)> {
        // Build WHERE clauses dynamically
        let mut conditions = Vec::new();

        if search_query.is_some() {
            conditions.push(
                "(toLower(ps.name) CONTAINS toLower($search_query) OR toLower(ps.description) CONTAINS toLower($search_query))"
                    .to_string(),
            );
        }

        if min_trust.is_some() {
            conditions.push("ps.trust_score >= $min_trust".to_string());
        }

        if tags.is_some() {
            // tags are stored as JSON array string — use CONTAINS for simple matching
            // For production, consider a dedicated Tags relationship
            conditions.push("ALL(tag IN $filter_tags WHERE ps.tags_json CONTAINS tag)".to_string());
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        // Count query
        let count_cypher = format!(
            "MATCH (ps:PublishedSkill) {} RETURN count(ps) AS total",
            where_clause
        );
        let mut count_query = query(&count_cypher);
        if let Some(q) = search_query {
            count_query = count_query.param("search_query", q.to_string());
        }
        if let Some(mt) = min_trust {
            count_query = count_query.param("min_trust", mt);
        }
        if let Some(t) = tags {
            let tags_list: Vec<String> = t.to_vec();
            count_query = count_query.param("filter_tags", tags_list);
        }

        let mut count_result = self.graph.execute(count_query).await?;
        let total: usize = if let Some(row) = count_result.next().await? {
            let count: i64 = row.get("total")?;
            count as usize
        } else {
            0
        };

        if total == 0 {
            return Ok((vec![], 0));
        }

        // Data query with pagination, sorted by trust_score DESC
        let data_cypher = format!(
            "MATCH (ps:PublishedSkill) {}
             RETURN ps
             ORDER BY ps.trust_score DESC, ps.published_at DESC
             SKIP $offset LIMIT $limit",
            where_clause
        );
        let mut data_query = query(&data_cypher)
            .param("offset", offset as i64)
            .param("limit", limit as i64);

        if let Some(q) = search_query {
            data_query = data_query.param("search_query", q.to_string());
        }
        if let Some(mt) = min_trust {
            data_query = data_query.param("min_trust", mt);
        }
        if let Some(t) = tags {
            let tags_list: Vec<String> = t.to_vec();
            data_query = data_query.param("filter_tags", tags_list);
        }

        let mut data_result = self.graph.execute(data_query).await?;
        let mut items = Vec::new();

        while let Some(row) = data_result.next().await? {
            let node: Node = row.get("ps")?;
            match parse_published_skill_node(&node) {
                Ok(ps) => items.push(ps),
                Err(e) => {
                    tracing::warn!("Skipping corrupted PublishedSkill node: {}", e);
                    continue;
                }
            }
        }

        Ok((items, total))
    }

    /// Increment the import count for a published skill.
    pub async fn increment_published_skill_imports(&self, id: Uuid) -> Result<()> {
        let query = query(
            "MATCH (ps:PublishedSkill {id: $id})
             SET ps.import_count = COALESCE(ps.import_count, 0) + 1",
        )
        .param("id", id.to_string());

        self.graph.run(query).await?;
        Ok(())
    }
}

/// Parse a Neo4j Node into a PublishedSkill.
fn parse_published_skill_node(node: &Node) -> Result<PublishedSkill> {
    let id_str: String = node.get("id")?;
    let id = Uuid::parse_str(&id_str)?;

    let name: String = node.get("name")?;
    let description: String = node.get("description").unwrap_or_default();

    let tags_json: String = node.get("tags_json").unwrap_or_else(|_| "[]".to_string());
    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();

    let package_json: String = node.get("package_json")?;
    let package: crate::skills::package::SkillPackage =
        serde_json::from_str(&package_json).context("Failed to deserialize package_json")?;

    let trust_score: f64 = node.get("trust_score").unwrap_or(0.0);

    let trust_components_json: String = node
        .get("trust_components_json")
        .unwrap_or_else(|_| "{}".to_string());
    let trust_components: crate::skills::trust::TrustComponents = serde_json::from_str(
        &trust_components_json,
    )
    .unwrap_or(crate::skills::trust::TrustComponents {
        energy: 0.0,
        cohesion: 0.0,
        activation: 0.0,
        success_rate: 0.0,
        source_projects: 0.0,
    });

    let trust_level_str: String = node
        .get("trust_level")
        .unwrap_or_else(|_| "untrusted".to_string());
    let trust_level = match trust_level_str.as_str() {
        "high" => crate::skills::trust::TrustLevel::High,
        "medium" => crate::skills::trust::TrustLevel::Medium,
        "low" => crate::skills::trust::TrustLevel::Low,
        _ => crate::skills::trust::TrustLevel::Untrusted,
    };

    let source_project_id_str: String = node.get("source_project_id")?;
    let source_project_id = Uuid::parse_str(&source_project_id_str)?;

    let source_project_name: String = node.get("source_project_name").unwrap_or_default();
    let published_by: String = node
        .get("published_by")
        .unwrap_or_else(|_| "unknown".to_string());

    let published_at_str: String = node
        .get("published_at")
        .unwrap_or_else(|_| chrono::Utc::now().to_rfc3339());
    let published_at = chrono::DateTime::parse_from_rfc3339(&published_at_str)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or_else(|_| chrono::Utc::now());

    let import_count: i64 = node.get("import_count").unwrap_or(0);

    Ok(PublishedSkill {
        id,
        name,
        description,
        tags,
        package,
        trust_score,
        trust_components,
        trust_level,
        source_project_id,
        source_project_name,
        published_by,
        published_at,
        import_count,
    })
}
