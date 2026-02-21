//! Knowledge Notes module
//!
//! Provides a system for capturing and managing contextual knowledge from conversations:
//! guidelines, gotchas, patterns, tips, and verifiable assertions.
//!
//! Notes can be linked to code entities and automatically surfaced to agents
//! based on relevance and graph propagation.

pub mod hashing;
pub mod lifecycle;
pub mod manager;
pub mod models;

pub use hashing::*;
pub use lifecycle::*;
pub use manager::{BackfillProgress, NoteManager};
pub use models::*;
