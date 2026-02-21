//! Mock embedding provider for tests
//!
//! Produces deterministic embeddings from text hashes, ensuring:
//! - Same text → same embedding (reproducible tests)
//! - Different texts → different embeddings (similarity tests work)
//! - Configurable dimensions (match the real provider's config)

use super::traits::EmbeddingProvider;
use anyhow::Result;
use async_trait::async_trait;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Deterministic mock embedding provider for tests.
///
/// Generates embeddings by hashing the input text and spreading the hash
/// across the configured number of dimensions. This ensures:
/// - Identical texts produce identical embeddings
/// - Different texts produce different embeddings (with very high probability)
/// - No network calls or external dependencies
///
/// # Example
///
/// ```rust
/// use project_orchestrator::embeddings::MockEmbeddingProvider;
/// use project_orchestrator::embeddings::EmbeddingProvider;
///
/// # tokio_test::block_on(async {
/// let provider = MockEmbeddingProvider::new(768);
/// let embedding = provider.embed_text("hello world").await.unwrap();
/// assert_eq!(embedding.len(), 768);
///
/// // Same text → same embedding
/// let embedding2 = provider.embed_text("hello world").await.unwrap();
/// assert_eq!(embedding, embedding2);
///
/// // Different text → different embedding
/// let embedding3 = provider.embed_text("goodbye world").await.unwrap();
/// assert_ne!(embedding, embedding3);
/// # });
/// ```
#[derive(Clone, Debug)]
pub struct MockEmbeddingProvider {
    dimensions: usize,
}

impl MockEmbeddingProvider {
    /// Create a new mock provider with the given embedding dimensions.
    ///
    /// Use 768 to match nomic-embed-text (production default).
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    /// Generate a deterministic embedding from text using hash spreading.
    ///
    /// Algorithm:
    /// 1. Hash the text with `DefaultHasher` (SipHash)
    /// 2. Use the hash as a seed to generate `dimensions` float values
    /// 3. Each dimension is derived by rehashing the previous hash
    /// 4. Values are normalized to [-1.0, 1.0] range
    /// 5. The resulting vector is L2-normalized (unit length)
    fn hash_to_embedding(&self, text: &str) -> Vec<f32> {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let mut hash = hasher.finish();

        let mut embedding = Vec::with_capacity(self.dimensions);
        for _ in 0..self.dimensions {
            // Map u64 to [-1.0, 1.0]
            let value = (hash as f64 / u64::MAX as f64) * 2.0 - 1.0;
            embedding.push(value as f32);

            // Chain hash for next dimension
            let mut h = DefaultHasher::new();
            hash.hash(&mut h);
            hash = h.finish();
        }

        // L2-normalize for cosine similarity compatibility
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.hash_to_embedding(text))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.hash_to_embedding(t)).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        "mock-hash-embedding"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_deterministic_embeddings() {
        let provider = MockEmbeddingProvider::new(768);
        let emb1 = provider.embed_text("hello world").await.unwrap();
        let emb2 = provider.embed_text("hello world").await.unwrap();
        assert_eq!(emb1, emb2, "Same text must produce identical embeddings");
    }

    #[tokio::test]
    async fn test_different_texts_different_embeddings() {
        let provider = MockEmbeddingProvider::new(768);
        let emb1 = provider.embed_text("hello").await.unwrap();
        let emb2 = provider.embed_text("world").await.unwrap();
        assert_ne!(emb1, emb2, "Different texts should produce different embeddings");
    }

    #[tokio::test]
    async fn test_correct_dimensions() {
        let provider = MockEmbeddingProvider::new(384);
        let emb = provider.embed_text("test").await.unwrap();
        assert_eq!(emb.len(), 384);

        let provider = MockEmbeddingProvider::new(1536);
        let emb = provider.embed_text("test").await.unwrap();
        assert_eq!(emb.len(), 1536);
    }

    #[tokio::test]
    async fn test_l2_normalized() {
        let provider = MockEmbeddingProvider::new(768);
        let emb = provider.embed_text("normalize me").await.unwrap();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Embedding should be L2-normalized, got norm = {}",
            norm
        );
    }

    #[tokio::test]
    async fn test_batch_consistency() {
        let provider = MockEmbeddingProvider::new(768);
        let texts = vec![
            "hello".to_string(),
            "world".to_string(),
            "test".to_string(),
        ];

        let batch_results = provider.embed_batch(&texts).await.unwrap();
        assert_eq!(batch_results.len(), 3);

        // Each batch result must match its individual embed_text call
        for (i, text) in texts.iter().enumerate() {
            let individual = provider.embed_text(text).await.unwrap();
            assert_eq!(
                batch_results[i], individual,
                "Batch result[{}] must match embed_text(\"{}\")",
                i, text
            );
        }
    }

    #[tokio::test]
    async fn test_empty_batch() {
        let provider = MockEmbeddingProvider::new(768);
        let result = provider.embed_batch(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_model_name() {
        let provider = MockEmbeddingProvider::new(768);
        assert_eq!(provider.model_name(), "mock-hash-embedding");
    }

    #[test]
    fn test_dimensions_accessor() {
        let provider = MockEmbeddingProvider::new(512);
        assert_eq!(provider.dimensions(), 512);
    }

    #[tokio::test]
    async fn test_cosine_similarity_self_is_one() {
        let provider = MockEmbeddingProvider::new(768);
        let emb = provider.embed_text("cosine test").await.unwrap();

        // Cosine similarity of a unit vector with itself = 1.0
        let dot: f32 = emb.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (dot - 1.0).abs() < 1e-5,
            "Cosine similarity with self should be ~1.0, got {}",
            dot
        );
    }
}
