//! EmbeddingProvider trait definition
//!
//! Defines the abstract interface for vector embedding generation.
//! This trait follows the same pattern as `GraphStore` and `SearchStore`:
//! async trait + Send + Sync for Arc<dyn EmbeddingProvider> usage.

use anyhow::Result;
use async_trait::async_trait;

/// Abstract interface for generating vector embeddings from text.
///
/// Implementations must be thread-safe (`Send + Sync`) to be shared
/// across async tasks via `Arc<dyn EmbeddingProvider>`.
///
/// # Implementations
///
/// - [`HttpEmbeddingProvider`](super::HttpEmbeddingProvider): HTTP client for any
///   OpenAI-compatible `/v1/embeddings` endpoint (Ollama, OpenAI, LiteLLM, etc.)
/// - [`MockEmbeddingProvider`](super::MockEmbeddingProvider): deterministic mock
///   that produces consistent embeddings from text hashes (for tests)
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate a vector embedding for a single text input.
    ///
    /// Returns a vector of `f32` with length equal to [`dimensions()`](Self::dimensions).
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding generation fails (network error,
    /// API error, model not loaded, etc.)
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate vector embeddings for multiple texts in a single batch.
    ///
    /// More efficient than calling `embed_text` in a loop when the provider
    /// supports batch processing (e.g., OpenAI API accepts multiple inputs).
    ///
    /// Returns a vector of embeddings, one per input text, in the same order.
    ///
    /// # Errors
    ///
    /// Returns an error if any embedding in the batch fails. Implementations
    /// should aim for all-or-nothing semantics.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// The dimensionality of the vectors produced by this provider.
    ///
    /// This value is fixed for a given model and must match the Neo4j
    /// vector index configuration (e.g., 768 for nomic-embed-text).
    fn dimensions(&self) -> usize;

    /// The name of the embedding model being used.
    ///
    /// Used for traceability — stored alongside embeddings so we know
    /// which model generated each vector (useful for re-embedding after
    /// model upgrades).
    fn model_name(&self) -> &str;
}
