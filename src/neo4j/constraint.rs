//! Neo4j Constraint operations

use super::client::Neo4jClient;
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Constraint operations
    // ========================================================================

    /// Create a constraint for a plan
    pub async fn create_constraint(
        &self,
        plan_id: Uuid,
        constraint: &ConstraintNode,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})
            CREATE (c:Constraint {
                id: $id,
                constraint_type: $constraint_type,
                description: $description,
                enforced_by: $enforced_by
            })
            CREATE (p)-[:CONSTRAINED_BY]->(c)
            "#,
        )
        .param("plan_id", plan_id.to_string())
        .param("id", constraint.id.to_string())
        .param(
            "constraint_type",
            format!("{:?}", constraint.constraint_type),
        )
        .param("description", constraint.description.clone())
        .param(
            "enforced_by",
            constraint.enforced_by.clone().unwrap_or_default(),
        );

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get constraints for a plan
    pub async fn get_plan_constraints(&self, plan_id: Uuid) -> Result<Vec<ConstraintNode>> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})-[:CONSTRAINED_BY]->(c:Constraint)
            RETURN c
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut constraints = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            constraints.push(ConstraintNode {
                id: node.get::<String>("id")?.parse()?,
                constraint_type: serde_json::from_str(&format!(
                    "\"{}\"",
                    node.get::<String>("constraint_type")?.to_lowercase()
                ))
                .unwrap_or(ConstraintType::Other),
                description: node.get("description")?,
                enforced_by: node
                    .get::<String>("enforced_by")
                    .ok()
                    .filter(|s| !s.is_empty()),
            });
        }

        Ok(constraints)
    }

    /// Get a single constraint by ID
    pub async fn get_constraint(&self, constraint_id: Uuid) -> Result<Option<ConstraintNode>> {
        let q = query(
            r#"
            MATCH (c:Constraint {id: $id})
            RETURN c
            "#,
        )
        .param("id", constraint_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            Ok(Some(ConstraintNode {
                id: node.get::<String>("id")?.parse()?,
                constraint_type: serde_json::from_str(&format!(
                    "\"{}\"",
                    node.get::<String>("constraint_type")?.to_lowercase()
                ))
                .unwrap_or(ConstraintType::Other),
                description: node.get("description")?,
                enforced_by: node
                    .get::<String>("enforced_by")
                    .ok()
                    .filter(|s| !s.is_empty()),
            }))
        } else {
            Ok(None)
        }
    }

    /// Update a constraint
    pub async fn update_constraint(
        &self,
        constraint_id: Uuid,
        description: Option<String>,
        constraint_type: Option<ConstraintType>,
        enforced_by: Option<String>,
    ) -> Result<()> {
        let mut set_clauses = vec![];
        if description.is_some() {
            set_clauses.push("c.description = $description");
        }
        if constraint_type.is_some() {
            set_clauses.push("c.constraint_type = $constraint_type");
        }
        if enforced_by.is_some() {
            set_clauses.push("c.enforced_by = $enforced_by");
        }

        if set_clauses.is_empty() {
            return Ok(());
        }

        let cypher = format!(
            "MATCH (c:Constraint {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", constraint_id.to_string());
        if let Some(description) = description {
            q = q.param("description", description);
        }
        if let Some(constraint_type) = constraint_type {
            q = q.param("constraint_type", format!("{:?}", constraint_type));
        }
        if let Some(enforced_by) = enforced_by {
            q = q.param("enforced_by", enforced_by);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a constraint
    pub async fn delete_constraint(&self, constraint_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (c:Constraint {id: $id})
            DETACH DELETE c
            "#,
        )
        .param("id", constraint_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }
}
