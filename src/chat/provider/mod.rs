//! Backend provider RESOLVER.
//!
//! This module only answers "which provider instance and which model for this
//! request", maps typed opening errors to HTTP answers, and translates policy
//! modes. It owns no provider registry and no price table: both live in nexus
//! (decision A1). The resolver works on an abstract [`resolver::InstanceCatalog`]
//! so that it stays pure and testable until the nexus registry is wired in.

pub mod catalog;
pub mod cognitive;
pub mod credentials;
pub mod endpoint_guard;
pub mod errors;
pub mod event_map;
pub mod listing;
pub mod native_factory;
pub mod nexus_tools;
pub mod policy;
pub mod resolver;
pub mod settings;
pub mod store;
pub mod transcripts;
