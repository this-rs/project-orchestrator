//! Enrichment pipeline stages.
//!
//! Each stage implements [`super::enrichment::EnrichmentStage`] and is
//! registered in the pipeline at startup.

pub mod skill_activation;

pub use skill_activation::SkillActivationStage;
