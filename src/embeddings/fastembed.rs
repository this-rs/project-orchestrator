//! Local embedding provider using fastembed-rs (ONNX Runtime)
//!
//! Default embedding provider — uses fastembed's in-process ONNX inference.
//! This eliminates the need for an external embedding server (Ollama, OpenAI, etc.)
//! at the cost of a larger binary (+30-80 MB) and in-process model memory (~200-400 MB).
//!
//! Configuration via environment variables:
//! - `FASTEMBED_MODEL` (default: `multilingual-e5-base`) — model identifier
//! - `FASTEMBED_CACHE_DIR` (default: `.fastembed_cache`) — ONNX model cache directory
//! - `EMBEDDING_BATCH_SIZE` (default: `8`) — texts per ONNX inference call
//!
//! Default model: `MultilingualE5Base` (768d, multilingual FR/EN/100+ languages)
//! which matches the existing Neo4j vector index (768d, cosine).
use crate::neo4j::traits::*;

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
///
/// Public so that `runner.rs` can resolve model names from config without
/// going through `from_env()`.
pub fn parse_model_name_pub(name: &str) -> EmbeddingModel {
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

/// Default number of texts per ONNX inference call.
///
/// Bounding the batch is what caps the peak memory of a single inference: with
/// `BatchLongest` padding to 512 tokens and 12 attention heads, a full batch of
/// 256 texts (fastembed's own default) allocates ~22 GiB of attention scores
/// that ONNX Runtime's CPU arena then retains for the life of the process. A
/// batch of 8 keeps the peak ~7× smaller AND is faster than the default
/// (measured 18/10/2026, see the MEM-1 note).
pub const DEFAULT_BATCH_SIZE: usize = 8;

/// Local embedding provider using fastembed-rs ONNX Runtime.
///
/// Thread-safe via `Arc<Mutex<TextEmbedding>>` because `embed()` requires `&mut self`.
/// All embedding calls are dispatched to `tokio::spawn_blocking` to avoid blocking
/// the async runtime (ONNX inference is CPU-bound).
///
/// # The lock is taken before the blocking thread, not on it
///
/// `embed()` needing `&mut self` means there is exactly one inference in flight
/// per process, whatever the caller count — that is inherent, and callers must
/// expect a queue. What is *not* inherent is where they wait. Acquiring the
/// mutex inside `spawn_blocking` makes every waiter park an OS thread from the
/// blocking pool to do nothing but block on a lock, so a burst of concurrent
/// embedders (chat, notes, plans, reasoning and document uploads all share this
/// one provider) converts directly into parked threads and starves every other
/// user of `spawn_blocking`.
///
/// So the guard is taken with `lock_owned().await` *first* — waiters cost a
/// cheap async task — and only the winner is handed to a blocking thread, which
/// it occupies for exactly as long as it is actually computing.
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
    batch_size: usize,
}

impl FastEmbedProvider {
    /// Create a new FastEmbed provider with explicit configuration.
    ///
    /// # Arguments
    ///
    /// * `model_variant` - The fastembed model to use
    /// * `cache_dir` - Optional directory for ONNX model cache (default: `.fastembed_cache`)
    /// * `batch_size` - Texts per inference call (0 falls back to [`DEFAULT_BATCH_SIZE`])
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX model cannot be loaded (download failure,
    /// corrupted cache, unsupported platform, etc.)
    pub fn new(
        model_variant: EmbeddingModel,
        cache_dir: Option<PathBuf>,
        batch_size: usize,
    ) -> Result<Self> {
        let dimensions = model_dimensions(&model_variant);
        let model_name = format!("{:?}", model_variant);

        // Guard against an accidental 0 that would make `chunks(0)` panic downstream.
        let batch_size = if batch_size == 0 {
            DEFAULT_BATCH_SIZE
        } else {
            batch_size
        };

        let mut options = TextInitOptions::new(model_variant).with_show_download_progress(true);

        if let Some(dir) = cache_dir {
            options = options.with_cache_dir(dir);
        }

        let embedding =
            TextEmbedding::try_new(options).context("Failed to initialize fastembed ONNX model")?;

        tracing::info!(
            model = %model_name,
            dimensions,
            batch_size,
            "FastEmbed provider initialized (local ONNX)"
        );

        Ok(Self {
            model: Arc::new(Mutex::new(embedding)),
            model_name,
            dimensions,
            batch_size,
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
            .map(|m| parse_model_name_pub(&m))
            .unwrap_or(EmbeddingModel::MultilingualE5Base);

        let cache_dir = std::env::var("FASTEMBED_CACHE_DIR")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);

        let batch_size = std::env::var("EMBEDDING_BATCH_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_BATCH_SIZE);

        Self::new(model_variant, cache_dir, batch_size)
    }

    /// Batch size used for ONNX inference (texts per call).
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

/// Split `texts` into sub-slices of at most `batch_size` elements each.
///
/// `embed_batch` calls the model once per chunk (releasing the in-process lock
/// between chunks), so the peak attention-score allocation is bounded by
/// `batch_size` regardless of how many texts the caller submits. Extracted as a
/// pure function so the batching contract is unit-testable without the ONNX model.
fn chunk_texts(texts: &[String], batch_size: usize) -> Vec<&[String]> {
    texts.chunks(batch_size).collect()
}

#[async_trait]
impl EmbeddingProvider for FastEmbedProvider {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let text = text.to_string();
        let batch_size = self.batch_size;
        // Wait for the model as a task, not as a parked OS thread. See the type docs.
        let mut guard = self.model.clone().lock_owned().await;

        let embeddings =
            tokio::task::spawn_blocking(move || guard.embed(vec![&text], Some(batch_size)))
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

        let batch_size = self.batch_size.max(1);
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        for chunk in chunk_texts(texts, batch_size) {
            let chunk: Vec<String> = chunk.to_vec();
            // Take the in-process lock per chunk (as a task, not a parked thread),
            // so a caller with many texts — e.g. `embed_parsed_file` on a 30k-line
            // file — releases the single embedder between chunks instead of holding
            // it for the whole set. See the type-level docs.
            let mut guard = self.model.clone().lock_owned().await;
            let batch = tokio::task::spawn_blocking(move || guard.embed(chunk, Some(batch_size)))
                .await
                .context("FastEmbed spawn_blocking panicked")?
                .context("FastEmbed embed_batch failed")?;
            out.extend(batch);
        }

        Ok(out)
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
    use crate::neo4j::traits::*;

    #[test]
    fn test_parse_model_name_known() {
        assert_eq!(
            parse_model_name_pub("multilingual-e5-base"),
            EmbeddingModel::MultilingualE5Base
        );
        assert_eq!(parse_model_name_pub("bge-m3"), EmbeddingModel::BGEM3);
        assert_eq!(
            parse_model_name_pub("nomic-embed-text-v1.5"),
            EmbeddingModel::NomicEmbedTextV15
        );
        assert_eq!(
            parse_model_name_pub("MULTILINGUAL-E5-BASE"),
            EmbeddingModel::MultilingualE5Base,
            "case-insensitive"
        );
    }

    #[test]
    fn test_parse_model_name_unknown_fallback() {
        assert_eq!(
            parse_model_name_pub("totally-unknown-model"),
            EmbeddingModel::MultilingualE5Base
        );
    }

    #[test]
    fn test_model_dimensions() {
        assert_eq!(model_dimensions(&EmbeddingModel::MultilingualE5Base), 768);
        assert_eq!(model_dimensions(&EmbeddingModel::AllMiniLML6V2), 384);
        assert_eq!(model_dimensions(&EmbeddingModel::BGEM3), 1024);
    }

    #[test]
    fn test_chunk_texts_bounds_batch() {
        let texts: Vec<String> = (0..100).map(|i| format!("text {i}")).collect();
        let chunks = chunk_texts(&texts, 16);

        // 100 texts / 16 = 7 chunks (6 full + 1 of 4)
        assert_eq!(chunks.len(), 7);
        let mut total = 0;
        for chunk in &chunks {
            assert!(chunk.len() <= 16, "chunk must not exceed batch_size");
            total += chunk.len();
        }
        assert_eq!(total, 100, "all texts covered exactly once");
        // order preserved (embed_batch relies on it: file first, then functions)
        assert_eq!(chunks[0][0], "text 0");
        assert_eq!(chunks.last().unwrap().last().unwrap(), "text 99");
    }

    #[test]
    fn test_chunk_texts_exact_multiple() {
        let texts: Vec<String> = (0..32).map(|i| format!("text {i}")).collect();
        let chunks = chunk_texts(&texts, 16);
        assert_eq!(chunks.len(), 2);
        assert!(chunks.iter().all(|c| c.len() == 16));
    }

    // Integration tests that actually load the ONNX model
    // These download the model on first run (~400MB) so they are slow.
    // Run explicitly: cargo test -- fastembed --ignored
    #[tokio::test]
    #[ignore = "requires ONNX model download (~1.1GB)"]
    async fn test_embed_text_dimensions() {
        let provider =
            FastEmbedProvider::new(EmbeddingModel::MultilingualE5Base, None, DEFAULT_BATCH_SIZE)
                .expect("Failed to init FastEmbed");

        let embedding = provider.embed_text("hello world").await.unwrap();
        assert_eq!(embedding.len(), 768, "MultilingualE5Base must produce 768d");
    }

    #[tokio::test]
    #[ignore = "requires ONNX model download (~1.1GB)"]
    async fn test_embed_batch_consistency() {
        let provider =
            FastEmbedProvider::new(EmbeddingModel::MultilingualE5Base, None, DEFAULT_BATCH_SIZE)
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
    #[ignore = "requires ONNX model download (~1.1GB)"]
    async fn test_embed_batch_large_input_bounded() {
        // 100 texts with batch_size 16 → must still produce 100 vectors, in order,
        // chunked internally (see `test_chunk_texts_bounds_batch` for the ≤ 16 proof).
        let provider = FastEmbedProvider::new(EmbeddingModel::MultilingualE5Base, None, 16)
            .expect("Failed to init FastEmbed");

        let texts: Vec<String> = (0..100).map(|i| format!("short sentence {i}")).collect();
        let batch = provider.embed_batch(&texts).await.unwrap();
        assert_eq!(batch.len(), 100);
        for (i, emb) in batch.iter().enumerate() {
            assert_eq!(emb.len(), 768, "Embedding {i} must be 768d");
        }
    }

    #[tokio::test]
    #[ignore = "requires ONNX model download (~1.1GB)"]
    async fn test_embed_empty_batch() {
        let provider =
            FastEmbedProvider::new(EmbeddingModel::MultilingualE5Base, None, DEFAULT_BATCH_SIZE)
                .expect("Failed to init FastEmbed");

        let result = provider.embed_batch(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires ONNX model download (~1.1GB)"]
    async fn test_model_name_accessor() {
        let provider =
            FastEmbedProvider::new(EmbeddingModel::MultilingualE5Base, None, DEFAULT_BATCH_SIZE)
                .expect("Failed to init FastEmbed");

        assert_eq!(provider.model_name(), "MultilingualE5Base");
        assert_eq!(provider.dimensions(), 768);
    }
}
