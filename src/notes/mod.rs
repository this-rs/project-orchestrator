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
pub mod witness;

/// Time constant (days) of note energy decay — the single value used by the
/// heartbeat, skill maintenance and data migrations (it used to be 14 days
/// in one place and 90 in others).
pub const ENERGY_HALF_LIFE_DAYS: f64 = 90.0;

/// Days without human activity (code sync, chat) after which a project is
/// dormant: its notes' energy and synapses stop decaying until it is active
/// again, so knowledge survives months without work on the project.
pub const PROJECT_DORMANT_AFTER_DAYS: i64 = 14;

pub use hashing::*;
pub use lifecycle::*;
pub use manager::{BackfillProgress, NoteManager, SynapseBackfillProgress, SynapseConfig};
pub use models::*;
pub use witness::{CodeRef, ExternalRef, Witness, WitnessKind, WitnessValidationError};
