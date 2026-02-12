//! Implementation Flow Planner — analyzes the knowledge graph to produce
//! a DAG of implementation phases (sequential + parallel branches).

use crate::meilisearch::SearchStore;
use crate::neo4j::GraphStore;
use crate::notes::NoteManager;
use crate::plan::PlanManager;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

// ============================================================================
// Input types
// ============================================================================

/// Scope of the implementation analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanScope {
    File,
    Module,
    Project,
}

impl Default for PlanScope {
    fn default() -> Self {
        Self::Module
    }
}

/// Request to plan an implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRequest {
    /// Project UUID
    pub project_id: Uuid,
    /// Project slug (for SearchStore queries)
    pub project_slug: Option<String>,
    /// Human description of what to implement
    pub description: String,
    /// Explicit entry points (file paths or function names)
    #[serde(default)]
    pub entry_points: Option<Vec<String>>,
    /// Scope of analysis
    #[serde(default)]
    pub scope: Option<PlanScope>,
    /// If true, auto-create a Plan MCP with Tasks/Steps
    #[serde(default)]
    pub auto_create_plan: Option<bool>,
}

// ============================================================================
// Output types
// ============================================================================

/// Risk level for a modification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

impl Default for RiskLevel {
    fn default() -> Self {
        Self::Low
    }
}

impl RiskLevel {
    /// Compare risk levels, returning the higher one
    pub fn max(self, other: Self) -> Self {
        match (&self, &other) {
            (Self::High, _) | (_, Self::High) => Self::High,
            (Self::Medium, _) | (_, Self::Medium) => Self::Medium,
            _ => Self::Low,
        }
    }
}

/// A single file modification in a phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Modification {
    /// File path to modify
    pub file: String,
    /// Reason for the modification
    pub reason: String,
    /// Symbols affected in this file
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub symbols_affected: Vec<String>,
    /// Risk level for this modification
    pub risk: RiskLevel,
    /// Number of files that depend on this file
    pub dependents_count: usize,
}

/// A parallel branch within a phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    /// Files in this branch
    pub files: Vec<String>,
    /// Reason for this branch
    pub reason: String,
}

/// A phase of implementation (sequential step in the DAG)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase {
    /// Phase number (1-based)
    pub phase_number: usize,
    /// Human-readable description
    pub description: String,
    /// Whether modifications in this phase can be done in parallel
    pub parallel: bool,
    /// Modifications (when parallel=false or single modification)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modifications: Vec<Modification>,
    /// Parallel branches (when parallel=true)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub branches: Vec<Branch>,
}

/// A contextual note relevant to the implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerNote {
    /// Note type (gotcha, guideline, pattern, etc.)
    pub note_type: String,
    /// Note content
    pub content: String,
    /// Importance level
    pub importance: String,
    /// Source entity (file/function the note is attached to)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_entity: Option<String>,
}

/// The complete implementation plan output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationPlan {
    /// Human-readable summary
    pub summary: String,
    /// Ordered phases of implementation
    pub phases: Vec<Phase>,
    /// Test files that should be run
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub test_files: Vec<String>,
    /// Relevant notes (gotchas, guidelines)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<PlannerNote>,
    /// Overall risk level
    pub total_risk: RiskLevel,
    /// Plan ID if auto_create_plan was true
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_id: Option<String>,
}

// ============================================================================
// Internal types (DAG construction)
// ============================================================================

/// Source of a relevant zone discovery
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ZoneSource {
    /// User-provided explicit entry point
    ExplicitEntry,
    /// Found via semantic code search
    SemanticSearch,
    /// Found via note reference
    NoteReference,
}

/// A relevant zone of code identified for modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevantZone {
    /// File path
    pub file_path: String,
    /// Functions in this zone
    #[serde(default)]
    pub functions: Vec<String>,
    /// Relevance score (0.0 - 1.0)
    pub relevance_score: f64,
    /// How this zone was discovered
    pub source: ZoneSource,
}

/// A node in the modification DAG
#[derive(Debug, Clone)]
pub struct DagNode {
    /// File path
    pub file_path: String,
    /// Symbols in this file
    pub symbols: Vec<String>,
    /// Risk level
    pub risk: RiskLevel,
    /// Test files that depend on this file
    pub test_files: Vec<String>,
    /// Number of dependent files
    pub dependents_count: usize,
}

/// The modification DAG (Directed Acyclic Graph)
#[derive(Debug, Clone)]
pub struct ModificationDag {
    /// Nodes indexed by file path
    pub nodes: HashMap<String, DagNode>,
    /// Edges: (from, to) meaning "from must be modified before to"
    pub edges: Vec<(String, String)>,
}

// ============================================================================
// Planner
// ============================================================================

/// Analyzes the knowledge graph to produce implementation plans
pub struct ImplementationPlanner {
    neo4j: Arc<dyn GraphStore>,
    meili: Arc<dyn SearchStore>,
    plan_manager: Arc<PlanManager>,
    note_manager: Arc<NoteManager>,
}

impl ImplementationPlanner {
    /// Create a new implementation planner
    pub fn new(
        neo4j: Arc<dyn GraphStore>,
        meili: Arc<dyn SearchStore>,
        plan_manager: Arc<PlanManager>,
        note_manager: Arc<NoteManager>,
    ) -> Self {
        Self {
            neo4j,
            meili,
            plan_manager,
            note_manager,
        }
    }

    /// Plan an implementation based on the knowledge graph
    pub async fn plan_implementation(&self, _request: PlanRequest) -> Result<ImplementationPlan> {
        // Placeholder — will be implemented in Task 4
        Err(anyhow::anyhow!("plan_implementation not yet implemented"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_request_serde_roundtrip() {
        let request = PlanRequest {
            project_id: Uuid::new_v4(),
            project_slug: Some("my-project".to_string()),
            description: "Add WebSocket support".to_string(),
            entry_points: Some(vec!["src/chat/mod.rs".to_string()]),
            scope: Some(PlanScope::Module),
            auto_create_plan: Some(false),
        };
        let json = serde_json::to_string(&request).unwrap();
        let parsed: PlanRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.description, request.description);
        assert_eq!(parsed.scope, request.scope);
    }

    #[test]
    fn test_implementation_plan_serde_roundtrip() {
        let plan = ImplementationPlan {
            summary: "Modify 3 files across 2 phases".to_string(),
            phases: vec![
                Phase {
                    phase_number: 1,
                    description: "Update data models".to_string(),
                    parallel: false,
                    modifications: vec![Modification {
                        file: "src/models.rs".to_string(),
                        reason: "Add WsMessage struct".to_string(),
                        symbols_affected: vec!["ChatMessage".to_string()],
                        risk: RiskLevel::Low,
                        dependents_count: 2,
                    }],
                    branches: vec![],
                },
                Phase {
                    phase_number: 2,
                    description: "Implement endpoints".to_string(),
                    parallel: true,
                    modifications: vec![],
                    branches: vec![
                        Branch {
                            files: vec!["src/api/ws.rs".to_string()],
                            reason: "REST WebSocket upgrade".to_string(),
                        },
                        Branch {
                            files: vec!["src/mcp/handlers.rs".to_string()],
                            reason: "MCP tool for WS".to_string(),
                        },
                    ],
                },
            ],
            test_files: vec!["src/tests/chat_tests.rs".to_string()],
            notes: vec![PlannerNote {
                note_type: "gotcha".to_string(),
                content: "Two backend instances share DB".to_string(),
                importance: "high".to_string(),
                source_entity: Some("src/db.rs".to_string()),
            }],
            total_risk: RiskLevel::Medium,
            plan_id: None,
        };
        let json = serde_json::to_string(&plan).unwrap();
        let parsed: ImplementationPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.phases.len(), 2);
        assert_eq!(parsed.total_risk, RiskLevel::Medium);
        assert!(parsed.phases[1].parallel);
        assert_eq!(parsed.phases[1].branches.len(), 2);
    }

    #[test]
    fn test_risk_level_max() {
        assert_eq!(RiskLevel::Low.max(RiskLevel::Low), RiskLevel::Low);
        assert_eq!(RiskLevel::Low.max(RiskLevel::Medium), RiskLevel::Medium);
        assert_eq!(RiskLevel::Medium.max(RiskLevel::High), RiskLevel::High);
        assert_eq!(RiskLevel::High.max(RiskLevel::Low), RiskLevel::High);
    }

    #[test]
    fn test_plan_scope_default() {
        assert_eq!(PlanScope::default(), PlanScope::Module);
    }

    #[tokio::test]
    async fn test_planner_construction_with_mocks() {
        use crate::test_helpers::mock_app_state;

        let state = mock_app_state();
        let plan_manager = Arc::new(PlanManager::new(state.neo4j.clone(), state.meili.clone()));
        let note_manager = Arc::new(NoteManager::new(state.neo4j.clone(), state.meili.clone()));

        let planner = ImplementationPlanner::new(
            state.neo4j.clone(),
            state.meili.clone(),
            plan_manager,
            note_manager,
        );

        // Placeholder should return error
        let result = planner
            .plan_implementation(PlanRequest {
                project_id: Uuid::new_v4(),
                project_slug: None,
                description: "test".to_string(),
                entry_points: None,
                scope: None,
                auto_create_plan: None,
            })
            .await;
        assert!(result.is_err());
    }
}
