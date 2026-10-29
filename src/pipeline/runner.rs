//! # Pipeline Runner — Configuration
//!
//! [`PipelineConfig`] is the configuration of a pipeline run. It is consumed
//! by the MCP pipeline handler (`mcp::pipeline_handler`).
use crate::neo4j::traits::*;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Working directory for command execution.
    pub cwd: String,
    /// Project slug (for MCP calls).
    pub project_slug: Option<String>,
    /// Maximum retries per gate before marking it as failed.
    pub max_retries_per_gate: usize,
    /// Overall pipeline timeout in seconds.
    pub timeout_secs: u64,
    /// Stop on first gate failure.
    pub fail_fast: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            cwd: ".".to_string(),
            project_slug: None,
            max_retries_per_gate: 2,
            timeout_secs: 3600,
            fail_fast: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::*;

    // -- PipelineConfig defaults --------------------------------------------

    #[test]
    fn config_defaults() {
        let config = PipelineConfig::default();
        assert_eq!(config.cwd, ".");
        assert!(config.project_slug.is_none());
        assert_eq!(config.max_retries_per_gate, 2);
        assert_eq!(config.timeout_secs, 3600);
        assert!(config.fail_fast);
    }

    #[test]
    fn config_serialization_roundtrip() {
        let config = PipelineConfig {
            cwd: "/tmp/project".to_string(),
            project_slug: Some("my-project".to_string()),
            max_retries_per_gate: 3,
            timeout_secs: 7200,
            fail_fast: false,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: PipelineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.cwd, "/tmp/project");
        assert_eq!(back.project_slug.as_deref(), Some("my-project"));
        assert_eq!(back.max_retries_per_gate, 3);
        assert_eq!(back.timeout_secs, 7200);
        assert!(!back.fail_fast);
    }

    // -- PipelineConfig with non-default values ------------------------------

    #[test]
    fn config_custom_values() {
        let config = PipelineConfig {
            cwd: "/opt/builds/workspace".to_string(),
            project_slug: Some("backend-api".to_string()),
            max_retries_per_gate: 10,
            timeout_secs: 30,
            fail_fast: false,
        };
        assert_eq!(config.cwd, "/opt/builds/workspace");
        assert_eq!(config.project_slug.as_deref(), Some("backend-api"));
        assert_eq!(config.max_retries_per_gate, 10);
        assert_eq!(config.timeout_secs, 30);
        assert!(!config.fail_fast);
    }

    #[test]
    fn config_clone_is_independent() {
        let config = PipelineConfig {
            cwd: "/tmp".to_string(),
            project_slug: Some("test".to_string()),
            max_retries_per_gate: 1,
            timeout_secs: 60,
            fail_fast: true,
        };
        let mut cloned = config.clone();
        cloned.cwd = "/other".to_string();
        cloned.fail_fast = false;
        // Original is not affected.
        assert_eq!(config.cwd, "/tmp");
        assert!(config.fail_fast);
        assert_eq!(cloned.cwd, "/other");
        assert!(!cloned.fail_fast);
    }

    #[test]
    fn config_debug_format() {
        let config = PipelineConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("PipelineConfig"));
        assert!(debug.contains("cwd"));
        assert!(debug.contains("fail_fast"));
    }

    #[test]
    fn config_deserialize_from_partial_json() {
        // Deserialize with all fields present but non-default values.
        let json = r#"{
            "cwd": "/workspace",
            "project_slug": null,
            "max_retries_per_gate": 0,
            "timeout_secs": 1,
            "fail_fast": false
        }"#;
        let config: PipelineConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.cwd, "/workspace");
        assert!(config.project_slug.is_none());
        assert_eq!(config.max_retries_per_gate, 0);
        assert_eq!(config.timeout_secs, 1);
        assert!(!config.fail_fast);
    }
}
