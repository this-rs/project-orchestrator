//! Local embedding provider using fastembed-rs (ONNX Runtime)
//!
//! Default embedding provider — uses fastembed's in-process ONNX inference.
//! This eliminates the need for an external embedding server (Ollama, OpenAI, etc.)
//! at the cost of a larger binary (+30-80 MB) and in-process model memory (~200-400 MB).
//!
//! Configuration via environment variables:
//! - `FASTEMBED_MODEL` (default: `multilingual-e5-base`) — model identifier
//! - `FASTEMBED_CACHE_DIR` (default: `.fastembed_cache`) — ONNX model cache directory
//!
//! Default model: `MultilingualE5Base` (768d, multilingual FR/EN/100+ languages)
//! which matches the existing Neo4j vector index (768d, cosine).

use super::traits::EmbeddingProvider;
use anyhow::{Context, Result};
use async_trait::async_trait;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Model name → `EmbeddingModel` variant mapping.
///
/// Uses short lowercase identifiers matching common naming conventions.
/// Falls back to `MultilingualE5Base` for unknown names.
fn parse_model_name(name: &str) -> EmbeddingModel {
    match name.to_lowercase().as_str() {
        // Multilingual (recommended for FR/EN)
        "multilingual-e5-base" | "intfloat/multilingual-e5-base" => {
            EmbeddingModel::MultilingualE5Base
        }
        "multilingual-e5-small" | "intfloat/multilingual-e5-small" => {
            EmbeddingModel::MultilingualE5Small
        }
        "multilingual-e5-large" | "intfloat/multilingual-e5-large" => {
            EmbeddingModel::MultilingualE5Large
        }
        // BGE
        "bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
        "bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
        "bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,
        "bge-m3" => EmbeddingModel::BGEM3,
        // Nomic
        "nomic-embed-text-v1" => EmbeddingModel::NomicEmbedTextV1,
        "nomic-embed-text-v1.5" => EmbeddingModel::NomicEmbedTextV15,
        // All-MiniLM
        "all-minilm-l6-v2" => EmbeddingModel::AllMiniLML6V2,
        "all-minilm-l12-v2" => EmbeddingModel::AllMiniLML12V2,
        // GTE
        "gte-base-en-v1.5" => EmbeddingModel::GTEBaseENV15,
        "gte-large-en-v1.5" => EmbeddingModel::GTELargeENV15,
        // Snowflake Arctic
        "snowflake-arctic-embed-m" => EmbeddingModel::SnowflakeArcticEmbedM,
        "snowflake-arctic-embed-l" => EmbeddingModel::SnowflakeArcticEmbedL,
        // Default fallback
        _ => {
            tracing::warn!(
                model = name,
                "Unknown FASTEMBED_MODEL, falling back to MultilingualE5Base (768d)"
            );
            EmbeddingModel::MultilingualE5Base
        }
    }
}

/// Get the embedding dimensions for a model variant.
fn model_dimensions(model: &EmbeddingModel) -> usize {
    TextEmbedding::get_model_info(model)
        .map(|info| info.dim)
        .unwrap_or(768)
}

/// Local embedding provider using fastembed-rs ONNX Runtime.
///
/// Thread-safe via `Arc<Mutex<TextEmbedding>>` because `embed()` requires `&mut self`.
/// All embedding calls are dispatched to `tokio::spawn_blocking` to avoid blocking
/// the async runtime (ONNX inference is CPU-bound).
///
/// # Example
///
/// ```rust,ignore
/// use project_orchestrator::embeddings::FastEmbedProvider;
///
/// let provider = FastEmbedProvider::from_env()?;
/// let embedding = provider.embed_text("hello world").await?;
/// assert_eq!(embedding.len(), 768);
/// ```
pub struct FastEmbedProvider {
    model: Arc<Mutex<TextEmbedding>>,
    model_name: String,
    dimensions: usize,
}

impl FastEmbedProvider {
    /// Create a new FastEmbed provider with explicit configuration.
    ///
    /// # Arguments
    ///
    /// * `model_variant` - The fastembed model to use
    /// * `cache_dir` - Optional directory for ONNX model cache (default: `.fastembed_cache`)
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX model cannot be loaded (download failure,
    /// corrupted cache, unsupported platform, etc.)
    pub fn new(model_variant: EmbeddingModel, cache_dir: Option<PathBuf>) -> Result<Self> {
        let dimensions = model_dimensions(&model_variant);
        let model_name = format!("{:?}", model_variant);

        let mut options = TextInitOptions::new(model_variant).with_show_download_progress(true);

        if let Some(dir) = cache_dir {
            options = options.with_cache_dir(dir);
        }

        let embedding =
            TextEmbedding::try_new(options).context("Failed to initialize fastembed ONNX model")?;

        tracing::info!(
            model = %model_name,
            dimensions,
            "FastEmbed provider initialized (local ONNX)"
        );

        Ok(Self {
            model: Arc::new(Mutex::new(embedding)),
            model_name,
            dimensions,
        })
    }

    /// Create a provider from environment variables.
    ///
    /// Reads:
    /// - `FASTEMBED_MODEL` (default: `multilingual-e5-base`)
    /// - `FASTEMBED_CACHE_DIR` (optional, overrides default `.fastembed_cache`)
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    pub fn from_env() -> Result<Self> {
        let model_variant = std::env::var("FASTEMBED_MODEL")
            .map(|m| parse_model_name(&m))
            .unwrap_or(EmbeddingModel::MultilingualE5Base);

        let cache_dir = std::env::var("FASTEMBED_CACHE_DIR")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);

        Self::new(model_variant, cache_dir)
    }
}

#[async_trait]
impl EmbeddingProvider for FastEmbedProvider {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let model = self.model.clone();
        let text = text.to_string();

        let embeddings = tokio::task::spawn_blocking(move || {
            let mut model = model.blocking_lock();
            model.embed(vec![&text], None)
        })
        .await
        .context("FastEmbed spawn_blocking panicked")?
        .context("FastEmbed embed_text failed")?;

        embeddings
            .into_iter()
            .next()
            .context("FastEmbed returned empty embedding response")
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let model = self.model.clone();
        let texts = texts.to_vec();

        tokio::task::spawn_blocking(move || {
            let mut model = model.blocking_lock();
            model.embed(texts, None)
        })
        .await
        .context("FastEmbed spawn_blocking panicked")?
        .context("FastEmbed embed_batch failed")
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_name_known() {
        assert_eq!(
            parse_model_name("multilingual-e5-base"),
            EmbeddingModel::MultilingualE5Base
        );
        assert_eq!(parse_model_name("bge-m3"), EmbeddingModel::BGEM3);
        assert_eq!(
            parse_model_name("nomic-embed-text-v1.5"),
            EmbeddingModel::NomicEmbedTextV15
        );
        assert_eq!(
            parse_model_name("MULTILINGUAL-E5-BASE"),
            EmbeddingModel::MultilingualE5Base,
            "case-insensitive"
        );
    }

    #[test]
    fn test_parse_model_name_unknown_fallback() {
        assert_eq!(
            parse_model_name("totally-unknown-model"),
            EmbeddingModel::MultilingualE5Base
        );
    }

    #[test]
    fn test_model_dimensions() {
        assert_eq!(model_dimensions(&EmbeddingModel::MultilingualE5Base), 768);
        assert_eq!(model_dimensions(&EmbeddingModel::AllMiniLML6V2), 384);
        assert_eq!(model_dimensions(&EmbeddingModel::BGEM3), 1024);
    }

    // Integration tests that actually load the ONNX model
    // These download the model on first run (~400MB) so they are slow.
    // Run explicitly: cargo test -- fastembed --ignored
    #[tokio::test]
    #[ignore = "requires ONNX model download (~400MB)"]
    async fn test_embed_text_dimensions() {
        let provider = FastEmbedProvider::new(EmbeddingModel::MultilingualE5Base, None)
            .expect("Failed to init FastEmbed");

        let embedding = provider.embed_text("hello world").await.unwrap();
        assert_eq!(embedding.len(), 768, "MultilingualE5Base must produce 768d");
    }

    #[tokio::test]
    #[ignore = "requires ONNX model download (~400MB)"]
    async fn test_embed_batch_consistency() {
        let provider = FastEmbedProvider::new(EmbeddingModel::MultilingualE5Base, None)
            .expect("Failed to init FastEmbed");

        let texts = vec![
            "hello world".to_string(),
            "bonjour le monde".to_string(),
            "hola mundo".to_string(),
        ];
        let batch = provider.embed_batch(&texts).await.unwrap();
        assert_eq!(batch.len(), 3);
        for (i, emb) in batch.iter().enumerate() {
            assert_eq!(emb.len(), 768, "Embedding {} must be 768d", i);
        }
    }

    #[tokio::test]
    #[ignore = "requires ONNX model download (~400MB)"]
    async fn test_embed_empty_batch() {
        let provider = FastEmbedProvider::new(EmbeddingModel::MultilingualE5Base, None)
            .expect("Failed to init FastEmbed");

        let result = provider.embed_batch(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires ONNX model download (~400MB)"]
    async fn test_model_name_accessor() {
        let provider = FastEmbedProvider::new(EmbeddingModel::MultilingualE5Base, None)
            .expect("Failed to init FastEmbed");

        assert_eq!(provider.model_name(), "MultilingualE5Base");
        assert_eq!(provider.dimensions(), 768);
    }
}
