//! Embedding generation module
//!
//! Provides vector embeddings for knowledge notes, enabling semantic search
//! and automatic synapse creation in the KnowledgeNeuron system.
//!
//! Architecture follows the project pattern (trait + impl + mock):
//! - `EmbeddingProvider` trait: async interface for embedding generation
//! - `FastEmbedProvider`: default implementation using local ONNX inference
//!   via fastembed-rs (zero external dependency, works offline)
//! - `HttpEmbeddingProvider`: optional implementation using any OpenAI-compatible
//!   `/v1/embeddings` API (Ollama, OpenAI, LiteLLM, vLLM, etc.)
//! - `MockEmbeddingProvider`: deterministic mock for tests

pub mod fastembed;
pub mod mock;
pub mod provider;
pub mod traits;

pub use fastembed::FastEmbedProvider;
pub use mock::MockEmbeddingProvider;
pub use provider::HttpEmbeddingProvider;
pub use traits::EmbeddingProvider;
