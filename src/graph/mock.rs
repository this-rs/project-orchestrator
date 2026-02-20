//! Mock analytics engine for testing consumers.
//!
//! Returns pre-configured `GraphAnalytics` results without performing
//! any actual computation or database access.

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;

use super::engine::{AnalyticsEngine, ProjectAnalytics};
use super::models::{CodeHealthReport, GraphAnalytics};

/// Mock implementation of `AnalyticsEngine` for testing.
///
/// Returns pre-configured results or empty analytics if not configured.
pub struct MockAnalyticsEngine {
    file_result: Option<GraphAnalytics>,
    function_result: Option<GraphAnalytics>,
}

impl MockAnalyticsEngine {
    /// Create a mock engine that returns empty analytics.
    pub fn new() -> Self {
        Self {
            file_result: None,
            function_result: None,
        }
    }

    /// Create a mock engine with pre-configured results.
    pub fn with_results(file: GraphAnalytics, function: GraphAnalytics) -> Self {
        Self {
            file_result: Some(file),
            function_result: Some(function),
        }
    }

    /// Helper to build an empty GraphAnalytics.
    fn empty_analytics() -> GraphAnalytics {
        GraphAnalytics {
            metrics: Default::default(),
            communities: vec![],
            components: vec![],
            health: CodeHealthReport::default(),
            modularity: 0.0,
            node_count: 0,
            edge_count: 0,
            computation_ms: 0,
        }
    }
}

impl Default for MockAnalyticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AnalyticsEngine for MockAnalyticsEngine {
    async fn analyze_file_graph(&self, _project_id: Uuid) -> Result<GraphAnalytics> {
        Ok(self
            .file_result
            .clone()
            .unwrap_or_else(Self::empty_analytics))
    }

    async fn analyze_function_graph(&self, _project_id: Uuid) -> Result<GraphAnalytics> {
        Ok(self
            .function_result
            .clone()
            .unwrap_or_else(Self::empty_analytics))
    }

    async fn analyze_project(&self, project_id: Uuid) -> Result<ProjectAnalytics> {
        let file_analytics = self.analyze_file_graph(project_id).await?;
        let function_analytics = self.analyze_function_graph(project_id).await?;

        Ok(ProjectAnalytics {
            file_analytics,
            function_analytics,
            computed_at: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_mock_returns_empty_by_default() {
        let mock = MockAnalyticsEngine::new();
        let analytics = mock.analyze_file_graph(Uuid::new_v4()).await.unwrap();
        assert_eq!(analytics.node_count, 0);
        assert!(analytics.metrics.is_empty());
    }

    #[tokio::test]
    async fn test_mock_returns_configured_results() {
        let file = GraphAnalytics {
            metrics: HashMap::new(),
            communities: vec![],
            components: vec![],
            health: CodeHealthReport::default(),
            modularity: 0.42,
            node_count: 10,
            edge_count: 20,
            computation_ms: 5,
        };
        let func = GraphAnalytics {
            metrics: HashMap::new(),
            communities: vec![],
            components: vec![],
            health: CodeHealthReport::default(),
            modularity: 0.35,
            node_count: 50,
            edge_count: 100,
            computation_ms: 12,
        };

        let mock = MockAnalyticsEngine::with_results(file, func);
        let result = mock.analyze_project(Uuid::new_v4()).await.unwrap();

        assert_eq!(result.file_analytics.node_count, 10);
        assert_eq!(result.function_analytics.node_count, 50);
    }
}
