//! Enrichment pipeline stages.
//!
//! Each stage implements [`super::enrichment::EnrichmentStage`] and is
//! registered in the pipeline at startup.

pub mod biomimicry;
pub mod knowledge_injection;
#[cfg(test)]
mod pipeline_e2e_tests;
pub mod skill_activation;
pub mod status_injection;

pub use biomimicry::BiomimicryStage;
pub use knowledge_injection::KnowledgeInjectionStage;
pub use skill_activation::SkillActivationStage;
pub use status_injection::StatusInjectionStage;
