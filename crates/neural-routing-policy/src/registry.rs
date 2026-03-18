//! Model Registry — versioning, promotion, and rollback for trained models.
//!
//! Manages model lifecycle: register → staging → production → archived.
//!
//! Features:
//! - `register`: save a new model version with metrics and config
//! - `get_best`: find the best version by a metric
//! - `compare`: side-by-side comparison of two versions
//! - `promote`: advance a model to the next stage
//! - `rollback`: revert to the previous production version
//! - `list_versions`: browse version history
//!
//! Storage:
//! - Model weights: safetensors files on filesystem (`{base_dir}/{name}/v{version}/`)
//! - Metadata: JSON index file (`{base_dir}/{name}/registry.json`)

use std::collections::HashMap;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Model stage in the lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelStage {
    /// Newly registered, not yet validated.
    Staging,
    /// Actively serving in production.
    Production,
    /// Previously in production, now superseded.
    Archived,
}

/// Training metrics for a model version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Validation loss (lower is better).
    pub val_loss: f32,
    /// Test MSE (lower is better).
    pub test_mse: Option<f32>,
    /// Test cosine distance (lower is better).
    pub test_cosine: Option<f32>,
    /// Number of training epochs completed.
    pub epochs: usize,
    /// Training time in seconds.
    pub training_time_secs: f64,
    /// Additional custom metrics.
    #[serde(default)]
    pub custom: HashMap<String, f64>,
}

/// A registered model version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Model name (e.g., "decision_transformer", "cql").
    pub name: String,
    /// Version number (monotonically increasing).
    pub version: u32,
    /// Current lifecycle stage.
    pub stage: ModelStage,
    /// Path to the model weights directory.
    pub weights_path: PathBuf,
    /// Training metrics.
    pub metrics: ModelMetrics,
    /// Model configuration (serialized JSON).
    pub config_json: String,
    /// When this version was registered.
    pub registered_at: DateTime<Utc>,
    /// When this version was last promoted/demoted.
    pub stage_changed_at: DateTime<Utc>,
    /// Optional description.
    pub description: Option<String>,
}

/// Comparison report between two model versions.
#[derive(Debug, Clone, Serialize)]
pub struct ComparisonReport {
    pub version_a: u32,
    pub version_b: u32,
    pub val_loss_diff: f32,
    pub test_mse_diff: Option<f32>,
    pub epochs_diff: i64,
    pub training_time_diff: f64,
    /// Which version is better (lower val_loss).
    pub better_version: u32,
}

// ---------------------------------------------------------------------------
// Registry index (persisted as JSON)
// ---------------------------------------------------------------------------

/// Persisted registry state for a single model name.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegistryIndex {
    /// Model name.
    name: String,
    /// All versions, ordered by version number.
    versions: Vec<ModelVersion>,
    /// Next version number to assign.
    next_version: u32,
}

impl RegistryIndex {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            versions: Vec::new(),
            next_version: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Model Registry
// ---------------------------------------------------------------------------

/// Model registry for versioning, promotion, and rollback.
pub struct ModelRegistry {
    /// Base directory for model storage.
    base_dir: PathBuf,
    /// In-memory indexes by model name.
    indexes: HashMap<String, RegistryIndex>,
}

impl ModelRegistry {
    /// Create or load a registry at the given base directory.
    pub fn new(base_dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let base_dir = base_dir.into();
        if !base_dir.exists() {
            std::fs::create_dir_all(&base_dir)?;
        }

        let mut registry = Self {
            base_dir,
            indexes: HashMap::new(),
        };
        registry.load_all_indexes()?;
        Ok(registry)
    }

    /// Create an in-memory-only registry (for testing).
    pub fn in_memory() -> Self {
        Self {
            base_dir: PathBuf::from("/tmp/model_registry_test"),
            indexes: HashMap::new(),
        }
    }

    /// Register a new model version.
    ///
    /// Returns the assigned version number.
    pub fn register(
        &mut self,
        name: &str,
        metrics: ModelMetrics,
        config_json: String,
        description: Option<String>,
    ) -> std::io::Result<u32> {
        let index = self
            .indexes
            .entry(name.to_string())
            .or_insert_with(|| RegistryIndex::new(name));

        let version = index.next_version;
        index.next_version += 1;

        let weights_dir = self.base_dir.join(name).join(format!("v{version}"));

        let now = Utc::now();
        let model_version = ModelVersion {
            name: name.to_string(),
            version,
            stage: ModelStage::Staging,
            weights_path: weights_dir,
            metrics,
            config_json,
            registered_at: now,
            stage_changed_at: now,
            description,
        };

        index.versions.push(model_version);
        self.save_index(name)?;

        Ok(version)
    }

    /// Get a specific version.
    pub fn get_version(&self, name: &str, version: u32) -> Option<&ModelVersion> {
        self.indexes
            .get(name)?
            .versions
            .iter()
            .find(|v| v.version == version)
    }

    /// Get the current production version.
    pub fn get_production(&self, name: &str) -> Option<&ModelVersion> {
        self.indexes
            .get(name)?
            .versions
            .iter()
            .find(|v| v.stage == ModelStage::Production)
    }

    /// Get the best version by a metric (lowest val_loss by default).
    pub fn get_best(&self, name: &str, metric: &str) -> Option<&ModelVersion> {
        let index = self.indexes.get(name)?;
        if index.versions.is_empty() {
            return None;
        }

        index.versions.iter().min_by(|a, b| {
            let va = get_metric_value(&a.metrics, metric);
            let vb = get_metric_value(&b.metrics, metric);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// List all versions for a model, newest first.
    pub fn list_versions(&self, name: &str) -> Vec<&ModelVersion> {
        match self.indexes.get(name) {
            Some(index) => index.versions.iter().rev().collect(),
            None => Vec::new(),
        }
    }

    /// List all registered model names.
    pub fn list_models(&self) -> Vec<&str> {
        self.indexes.keys().map(|s| s.as_str()).collect()
    }

    /// Compare two versions.
    pub fn compare(&self, name: &str, version_a: u32, version_b: u32) -> Option<ComparisonReport> {
        let a = self.get_version(name, version_a)?;
        let b = self.get_version(name, version_b)?;

        let val_loss_diff = b.metrics.val_loss - a.metrics.val_loss;
        let test_mse_diff = match (a.metrics.test_mse, b.metrics.test_mse) {
            (Some(ma), Some(mb)) => Some(mb - ma),
            _ => None,
        };
        let better_version = if a.metrics.val_loss <= b.metrics.val_loss {
            version_a
        } else {
            version_b
        };

        Some(ComparisonReport {
            version_a,
            version_b,
            val_loss_diff,
            test_mse_diff,
            epochs_diff: b.metrics.epochs as i64 - a.metrics.epochs as i64,
            training_time_diff: b.metrics.training_time_secs - a.metrics.training_time_secs,
            better_version,
        })
    }

    /// Promote a version to the next stage.
    ///
    /// Staging → Production (archives current production if any).
    /// Production → Archived.
    pub fn promote(&mut self, name: &str, version: u32) -> std::io::Result<ModelStage> {
        let index = self
            .indexes
            .get_mut(name)
            .ok_or_else(|| std::io::Error::other(format!("Model '{name}' not found")))?;

        // Find the version to promote
        let target = index
            .versions
            .iter()
            .find(|v| v.version == version)
            .ok_or_else(|| std::io::Error::other(format!("Version {version} not found")))?;

        let new_stage = match target.stage {
            ModelStage::Staging => {
                // Archive any current production version
                let now = Utc::now();
                for v in index.versions.iter_mut() {
                    if v.stage == ModelStage::Production {
                        v.stage = ModelStage::Archived;
                        v.stage_changed_at = now;
                    }
                }
                ModelStage::Production
            }
            ModelStage::Production => ModelStage::Archived,
            ModelStage::Archived => {
                return Err(std::io::Error::other("Cannot promote archived model"));
            }
        };

        // Apply the promotion
        let target = index
            .versions
            .iter_mut()
            .find(|v| v.version == version)
            .unwrap();
        target.stage = new_stage;
        target.stage_changed_at = Utc::now();

        self.save_index(name)?;
        Ok(new_stage)
    }

    /// Rollback: archive current production and restore the previous production version.
    ///
    /// Returns the version that was restored to production.
    pub fn rollback(&mut self, name: &str) -> std::io::Result<u32> {
        let index = self
            .indexes
            .get_mut(name)
            .ok_or_else(|| std::io::Error::other(format!("Model '{name}' not found")))?;

        // Find current production version
        let current_prod = index
            .versions
            .iter()
            .find(|v| v.stage == ModelStage::Production)
            .map(|v| v.version);

        let current_version = current_prod
            .ok_or_else(|| std::io::Error::other("No production version to rollback from"))?;

        // Find the most recent archived version (the one that was previously in production)
        let prev_prod = index
            .versions
            .iter()
            .filter(|v| v.stage == ModelStage::Archived)
            .max_by_key(|v| v.stage_changed_at)
            .map(|v| v.version);

        let restore_version =
            prev_prod.ok_or_else(|| std::io::Error::other("No archived version to rollback to"))?;

        let now = Utc::now();

        // Archive current production
        for v in index.versions.iter_mut() {
            if v.version == current_version {
                v.stage = ModelStage::Archived;
                v.stage_changed_at = now;
            }
        }

        // Restore previous version to production
        for v in index.versions.iter_mut() {
            if v.version == restore_version {
                v.stage = ModelStage::Production;
                v.stage_changed_at = now;
            }
        }

        self.save_index(name)?;
        Ok(restore_version)
    }

    /// Total number of versions across all models.
    pub fn total_versions(&self) -> usize {
        self.indexes.values().map(|idx| idx.versions.len()).sum()
    }

    // --- Persistence ---

    fn index_path(&self, name: &str) -> PathBuf {
        self.base_dir.join(name).join("registry.json")
    }

    fn save_index(&self, name: &str) -> std::io::Result<()> {
        if let Some(index) = self.indexes.get(name) {
            let dir = self.base_dir.join(name);
            if !dir.exists() {
                std::fs::create_dir_all(&dir)?;
            }
            let json = serde_json::to_string_pretty(index)
                .map_err(|e| std::io::Error::other(format!("JSON: {e}")))?;
            std::fs::write(self.index_path(name), json)?;
        }
        Ok(())
    }

    fn load_all_indexes(&mut self) -> std::io::Result<()> {
        if !self.base_dir.exists() {
            return Ok(());
        }

        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                let index_path = self.index_path(&name);
                if index_path.exists() {
                    let json = std::fs::read_to_string(&index_path)?;
                    let index: RegistryIndex = serde_json::from_str(&json)
                        .map_err(|e| std::io::Error::other(format!("JSON: {e}")))?;
                    self.indexes.insert(name, index);
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get_metric_value(metrics: &ModelMetrics, metric: &str) -> f64 {
    match metric {
        "val_loss" => metrics.val_loss as f64,
        "test_mse" => metrics.test_mse.unwrap_or(f32::INFINITY) as f64,
        "test_cosine" => metrics.test_cosine.unwrap_or(f32::INFINITY) as f64,
        "epochs" => metrics.epochs as f64,
        "training_time" => metrics.training_time_secs,
        other => metrics.custom.get(other).copied().unwrap_or(f64::INFINITY),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(val_loss: f32, test_mse: f32, epochs: usize) -> ModelMetrics {
        ModelMetrics {
            val_loss,
            test_mse: Some(test_mse),
            test_cosine: Some(0.1),
            epochs,
            training_time_secs: epochs as f64 * 0.5,
            custom: HashMap::new(),
        }
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = ModelRegistry::in_memory();

        let v1 = registry
            .register("dt", make_metrics(0.5, 0.3, 50), "{}".into(), None)
            .unwrap();
        let v2 = registry
            .register("dt", make_metrics(0.4, 0.2, 80), "{}".into(), None)
            .unwrap();

        assert_eq!(v1, 1);
        assert_eq!(v2, 2);

        let ver = registry.get_version("dt", 1).unwrap();
        assert_eq!(ver.version, 1);
        assert_eq!(ver.stage, ModelStage::Staging);
        assert!((ver.metrics.val_loss - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_get_best() {
        let mut registry = ModelRegistry::in_memory();

        registry
            .register("dt", make_metrics(0.5, 0.3, 50), "{}".into(), None)
            .unwrap();
        registry
            .register("dt", make_metrics(0.3, 0.1, 80), "{}".into(), None)
            .unwrap();
        registry
            .register("dt", make_metrics(0.4, 0.2, 60), "{}".into(), None)
            .unwrap();

        let best = registry.get_best("dt", "val_loss").unwrap();
        assert_eq!(best.version, 2, "v2 has lowest val_loss");

        let best_mse = registry.get_best("dt", "test_mse").unwrap();
        assert_eq!(best_mse.version, 2, "v2 has lowest test_mse");
    }

    #[test]
    fn test_promote_and_archive() {
        let mut registry = ModelRegistry::in_memory();

        registry
            .register("dt", make_metrics(0.5, 0.3, 50), "{}".into(), None)
            .unwrap();
        registry
            .register("dt", make_metrics(0.3, 0.1, 80), "{}".into(), None)
            .unwrap();

        // Promote v1 to production
        let stage = registry.promote("dt", 1).unwrap();
        assert_eq!(stage, ModelStage::Production);
        assert_eq!(registry.get_production("dt").unwrap().version, 1);

        // Promote v2 → should archive v1 and promote v2
        let stage = registry.promote("dt", 2).unwrap();
        assert_eq!(stage, ModelStage::Production);

        let prod = registry.get_production("dt").unwrap();
        assert_eq!(prod.version, 2);

        let v1 = registry.get_version("dt", 1).unwrap();
        assert_eq!(v1.stage, ModelStage::Archived);
    }

    #[test]
    fn test_rollback() {
        let mut registry = ModelRegistry::in_memory();

        registry
            .register("dt", make_metrics(0.5, 0.3, 50), "{}".into(), None)
            .unwrap();
        registry
            .register("dt", make_metrics(0.3, 0.1, 80), "{}".into(), None)
            .unwrap();
        registry
            .register("dt", make_metrics(0.4, 0.2, 60), "{}".into(), None)
            .unwrap();

        // Promote v1, then v2
        registry.promote("dt", 1).unwrap();
        registry.promote("dt", 2).unwrap();

        // Now: v1=Archived, v2=Production, v3=Staging
        assert_eq!(registry.get_production("dt").unwrap().version, 2);

        // Rollback → should restore v1 to production
        let restored = registry.rollback("dt").unwrap();
        assert_eq!(restored, 1);

        assert_eq!(registry.get_production("dt").unwrap().version, 1);
        assert_eq!(
            registry.get_version("dt", 2).unwrap().stage,
            ModelStage::Archived
        );
    }

    #[test]
    fn test_compare() {
        let mut registry = ModelRegistry::in_memory();

        registry
            .register("dt", make_metrics(0.5, 0.3, 50), "{}".into(), None)
            .unwrap();
        registry
            .register("dt", make_metrics(0.3, 0.1, 80), "{}".into(), None)
            .unwrap();

        let report = registry.compare("dt", 1, 2).unwrap();
        assert_eq!(report.better_version, 2);
        assert!((report.val_loss_diff - (-0.2)).abs() < 1e-5);
        assert!((report.test_mse_diff.unwrap() - (-0.2)).abs() < 1e-5);
        assert_eq!(report.epochs_diff, 30);
    }

    #[test]
    fn test_list_versions() {
        let mut registry = ModelRegistry::in_memory();

        for i in 0..5 {
            registry
                .register(
                    "dt",
                    make_metrics(0.5 - i as f32 * 0.1, 0.3, 50 + i * 10),
                    "{}".into(),
                    Some(format!("version {}", i + 1)),
                )
                .unwrap();
        }

        let versions = registry.list_versions("dt");
        assert_eq!(versions.len(), 5);
        // Newest first
        assert_eq!(versions[0].version, 5);
        assert_eq!(versions[4].version, 1);
    }

    #[test]
    fn test_list_models() {
        let mut registry = ModelRegistry::in_memory();

        registry
            .register("dt", make_metrics(0.5, 0.3, 50), "{}".into(), None)
            .unwrap();
        registry
            .register("cql", make_metrics(0.6, 0.4, 30), "{}".into(), None)
            .unwrap();

        let models = registry.list_models();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&"dt"));
        assert!(models.contains(&"cql"));
    }

    #[test]
    fn test_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.path().to_owned();

        // Create and populate
        {
            let mut registry = ModelRegistry::new(&dir_path).unwrap();
            registry
                .register(
                    "dt",
                    make_metrics(0.5, 0.3, 50),
                    "{\"lr\": 1e-4}".into(),
                    None,
                )
                .unwrap();
            registry
                .register(
                    "dt",
                    make_metrics(0.3, 0.1, 80),
                    "{\"lr\": 5e-5}".into(),
                    None,
                )
                .unwrap();
            registry.promote("dt", 1).unwrap();
        }

        // Reload and verify
        {
            let registry = ModelRegistry::new(&dir_path).unwrap();
            assert_eq!(registry.list_versions("dt").len(), 2);
            assert_eq!(registry.get_production("dt").unwrap().version, 1);
            assert_eq!(
                registry.get_version("dt", 2).unwrap().stage,
                ModelStage::Staging
            );
        }
    }

    #[test]
    fn test_custom_metrics() {
        let mut registry = ModelRegistry::in_memory();

        let mut metrics = make_metrics(0.5, 0.3, 50);
        metrics.custom.insert("ood_recall".to_string(), 0.85);

        registry.register("dt", metrics, "{}".into(), None).unwrap();

        let ver = registry.get_version("dt", 1).unwrap();
        assert!((ver.metrics.custom["ood_recall"] - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_max_versions() {
        let mut registry = ModelRegistry::in_memory();

        for i in 0..50 {
            registry
                .register(
                    "dt",
                    make_metrics(0.5 - i as f32 * 0.01, 0.3, 50),
                    "{}".into(),
                    None,
                )
                .unwrap();
        }

        assert_eq!(registry.list_versions("dt").len(), 50);
        assert_eq!(registry.total_versions(), 50);
    }
}
