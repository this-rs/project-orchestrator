//! Meilisearch client and index management
#[allow(unused_imports)]
use crate::neo4j::traits::*;

pub mod client;
mod impl_search_store;
pub mod indexes;
pub mod traits;

pub use client::MeiliClient;
pub use traits::SearchStore;

// The in-memory store. Unit tests get it through `cfg(test)`; a build that
// selects `storage.backend = "memory"` gets it through the `mock-store`
// feature (see src/neo4j/mod.rs for the same gate).
#[cfg(any(test, feature = "mock-store"))]
pub(crate) mod mock;
