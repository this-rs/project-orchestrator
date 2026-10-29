//! Sharing & Consent module — Privacy MVP
//!
//! Provides the consent gate for the distillation pipeline,
//! TTL management, tombstone verification, and P2P revocation
//! broadcast utilities.
use crate::neo4j::traits::*;

pub mod consent_gate;
pub mod revocation;
pub mod tombstone;
pub mod ttl;
