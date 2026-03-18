//! Enrichment pipeline stages.
//!
//! Each stage implements [`super::enrichment::EnrichmentStage`] and is
//! registered in the pipeline at startup.

pub mod biomimicry;
pub mod file_context;
pub mod knowledge_injection;
#[cfg(test)]
mod pipeline_e2e_tests;
pub mod skill_activation;
pub mod status_injection;
pub mod user_profile;

pub use biomimicry::BiomimicryStage;
pub use file_context::FileContextStage;
pub use knowledge_injection::KnowledgeInjectionStage;
pub use skill_activation::SkillActivationStage;
pub use status_injection::{GraphProtocolProvider, StatusInjectionConfig, StatusInjectionStage};
pub use user_profile::UserProfileStage;
