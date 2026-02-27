//! Neural Skills module
//!
//! Skills are emergent knowledge clusters detected by Louvain community
//! detection on the SYNAPSE graph (Note↔Note). They represent coherent
//! domains of expertise that can be automatically activated via Claude Code
//! hooks to inject contextual knowledge.
//!
//! # Lifecycle
//!
//! Skills follow an autonomous lifecycle:
//! - **Emerging**: Newly detected by Louvain, needs validation through usage
//! - **Active**: Proven useful (2+ hook activations), matched by hooks
//! - **Dormant**: Inactive for 30+ days or low energy, excluded from matching
//! - **Archived**: Dead skill (90+ days inactive), data preserved
//! - **Imported**: Created from an imported SkillPackage, in probation
//!
//! # Architecture
//!
//! Skills are built on top of the existing Knowledge Neuron Network:
//! - Notes = neurons with energy (0.0-1.0) and exponential decay
//! - SYNAPSE relations = weighted connections between notes
//! - Louvain detects clusters of strongly connected notes → Skills
//! - Triggers (Regex/FileGlob) match tool inputs to activate skills
//! - Hook activation injects relevant notes into Claude Code context

pub mod activation;
pub mod cache;
pub mod detection;
pub mod evolution;
pub mod feedback;
pub mod hook_extractor;
pub mod lifecycle;
pub mod maintenance;
pub mod models;
pub mod naming;
pub mod templates;
pub mod triggers;

pub use activation::*;
pub use cache::*;
pub use detection::*;
pub use hook_extractor::*;
pub use lifecycle::*;
pub use models::*;
pub use templates::*;
pub use triggers::*;
