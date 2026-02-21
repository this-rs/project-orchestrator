//! HTTP embedding provider implementation
//!
//! Implements `EmbeddingProvider` using any OpenAI-compatible `/v1/embeddings` endpoint.
//!
//! Supported providers:
//! - **Ollama** (default): `http://localhost:11434/v1/embeddings` with `nomic-embed-text`
//! - **OpenAI**: `https://api.openai.com/v1/embeddings` with `text-embedding-3-small`
//! - **LiteLLM / vLLM / any OpenAI-compatible**: just set the URL
//!
//! Configuration via environment variables:
//! - `EMBEDDING_URL` (default: `http://localhost:11434/v1/embeddings`)
//! - `EMBEDDING_MODEL` (default: `nomic-embed-text`)
//! - `EMBEDDING_API_KEY` (optional, for OpenAI/Voyage)
//! - `EMBEDDING_DIMENSIONS` (default: `768`)

use super::traits::EmbeddingProvider;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// HTTP-based embedding provider using the OpenAI `/v1/embeddings` API format.
///
/// Thread-safe and cheaply cloneable (shares the reqwest client internally).
#[derive(Clone)]
pub struct HttpEmbeddingProvider {
    client: reqwest::Client,
    url: String,
    model: String,
    api_key: Option<String>,
    dimensions: usize,
}

/// OpenAI-compatible embedding request
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: EmbeddingInput,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

/// OpenAI-compatible embedding response
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[allow(dead_code)]
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    #[allow(dead_code)]
    index: usize,
}

/// OpenAI-compatible error response
#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: Option<ErrorDetail>,
}

#[derive(Debug, Deserialize)]
struct ErrorDetail {
    message: String,
    #[allow(dead_code)]
    r#type: Option<String>,
}

impl HttpEmbeddingProvider {
    /// Create a new HTTP embedding provider with explicit configuration.
    ///
    /// # Arguments
    ///
    /// * `url` - The embedding API endpoint (e.g., `http://localhost:11434/v1/embeddings`)
    /// * `model` - The model name to use (e.g., `nomic-embed-text`)
    /// * `api_key` - Optional API key for authenticated endpoints (e.g., OpenAI)
    /// * `dimensions` - Expected embedding dimensions (must match the model output)
    pub fn new(url: String, model: String, api_key: Option<String>, dimensions: usize) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            url,
            model,
            api_key,
            dimensions,
        }
    }

    /// Create a provider from environment variables.
    ///
    /// Reads:
    /// - `EMBEDDING_URL` (default: `http://localhost:11434/v1/embeddings`)
    /// - `EMBEDDING_MODEL` (default: `nomic-embed-text`)
    /// - `EMBEDDING_API_KEY` (optional)
    /// - `EMBEDDING_DIMENSIONS` (default: `768`)
    ///
    /// Returns `None` if `EMBEDDING_URL` is explicitly set to empty or "disabled".
    pub fn from_env() -> Option<Self> {
        let url = std::env::var("EMBEDDING_URL")
            .unwrap_or_else(|_| "http://localhost:11434/v1/embeddings".to_string());

        // Allow explicit opt-out
        if url.is_empty() || url.eq_ignore_ascii_case("disabled") {
            return None;
        }

        let model =
            std::env::var("EMBEDDING_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());

        let api_key = std::env::var("EMBEDDING_API_KEY")
            .ok()
            .filter(|k| !k.is_empty());

        let dimensions: usize = std::env::var("EMBEDDING_DIMENSIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(768);

        Some(Self::new(url, model, api_key, dimensions))
    }

    /// Send an embedding request and parse the response.
    async fn request_embeddings(&self, input: EmbeddingInput) -> Result<Vec<Vec<f32>>> {
        let request_body = EmbeddingRequest {
            model: self.model.clone(),
            input,
        };

        let mut req = self.client.post(&self.url).json(&request_body);

        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let response = req
            .send()
            .await
            .with_context(|| format!("Failed to connect to embedding API at {}", self.url))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            // Try to parse OpenAI-style error
            if let Ok(err) = serde_json::from_str::<ErrorResponse>(&body) {
                if let Some(detail) = err.error {
                    anyhow::bail!(
                        "Embedding API error ({}): {}",
                        status.as_u16(),
                        detail.message
                    );
                }
            }
            anyhow::bail!("Embedding API returned {} — {}", status.as_u16(), body);
        }

        let resp: EmbeddingResponse = response
            .json()
            .await
            .context("Failed to parse embedding API response")?;

        // Sort by index to ensure correct ordering
        let mut data = resp.data;
        data.sort_by_key(|d| d.index);

        let embeddings: Vec<Vec<f32>> = data.into_iter().map(|d| d.embedding).collect();

        // Validate dimensions
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != self.dimensions {
                anyhow::bail!(
                    "Embedding dimension mismatch at index {}: expected {}, got {} (model: {})",
                    i,
                    self.dimensions,
                    emb.len(),
                    self.model
                );
            }
        }

        Ok(embeddings)
    }
}

#[async_trait]
impl EmbeddingProvider for HttpEmbeddingProvider {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self
            .request_embeddings(EmbeddingInput::Single(text.to_string()))
            .await?;

        embeddings
            .into_iter()
            .next()
            .context("Embedding API returned empty response")
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Some providers have batch size limits; split into chunks of 50
        const BATCH_SIZE: usize = 50;
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(BATCH_SIZE) {
            let input = EmbeddingInput::Batch(chunk.to_vec());
            let mut embeddings = self.request_embeddings(input).await?;
            all_embeddings.append(&mut embeddings);
        }

        Ok(all_embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Combined test for env-based configuration to avoid race conditions.
    /// Env vars are process-global, so parallel tests interfere with each other.
    /// Follows the same pattern as `test_yaml_and_env_lifecycle` in lib.rs.
    #[test]
    fn test_from_env_lifecycle() {
        fn clear_env() {
            std::env::remove_var("EMBEDDING_URL");
            std::env::remove_var("EMBEDDING_MODEL");
            std::env::remove_var("EMBEDDING_API_KEY");
            std::env::remove_var("EMBEDDING_DIMENSIONS");
        }

        // --- Phase 1: Defaults ---
        clear_env();

        let provider = HttpEmbeddingProvider::from_env().unwrap();
        assert_eq!(provider.url, "http://localhost:11434/v1/embeddings");
        assert_eq!(provider.model, "nomic-embed-text");
        assert!(provider.api_key.is_none());
        assert_eq!(provider.dimensions, 768);

        // --- Phase 2: Custom values ---
        std::env::set_var("EMBEDDING_URL", "https://api.openai.com/v1/embeddings");
        std::env::set_var("EMBEDDING_MODEL", "text-embedding-3-small");
        std::env::set_var("EMBEDDING_API_KEY", "sk-test-key");
        std::env::set_var("EMBEDDING_DIMENSIONS", "1536");

        let provider = HttpEmbeddingProvider::from_env().unwrap();
        assert_eq!(provider.url, "https://api.openai.com/v1/embeddings");
        assert_eq!(provider.model, "text-embedding-3-small");
        assert_eq!(provider.api_key, Some("sk-test-key".to_string()));
        assert_eq!(provider.dimensions, 1536);

        // --- Phase 3: Disabled ---
        std::env::set_var("EMBEDDING_URL", "disabled");
        assert!(HttpEmbeddingProvider::from_env().is_none());

        std::env::set_var("EMBEDDING_URL", "");
        assert!(HttpEmbeddingProvider::from_env().is_none());

        // --- Cleanup ---
        clear_env();
    }

    #[test]
    fn test_new_explicit_config() {
        let provider = HttpEmbeddingProvider::new(
            "http://localhost:8080/embed".to_string(),
            "test-model".to_string(),
            Some("key-123".to_string()),
            512,
        );
        assert_eq!(provider.url, "http://localhost:8080/embed");
        assert_eq!(provider.model, "test-model");
        assert_eq!(provider.api_key, Some("key-123".to_string()));
        assert_eq!(provider.dimensions, 512);
        assert_eq!(provider.model_name(), "test-model");
        assert_eq!(provider.dimensions(), 512);
    }
}
