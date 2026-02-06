//! Meilisearch client and index management

pub mod client;
mod impl_search_store;
pub mod indexes;
pub mod traits;

pub use client::MeiliClient;
pub use traits::SearchStore;

#[cfg(test)]
pub(crate) mod mock;
