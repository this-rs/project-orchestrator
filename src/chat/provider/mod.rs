//! Backend provider RESOLVER.
//!
//! This module only answers "which provider instance and which model for this
//! request", maps typed opening errors to HTTP answers, and translates policy
//! modes. It owns no provider registry and no price table: both live in nexus
//! (decision A1). The resolver works on an abstract [`resolver::InstanceCatalog`]
//! so that it stays pure and testable until the nexus registry is wired in.

pub mod credentials;
pub mod errors;
pub mod policy;
pub mod resolver;
