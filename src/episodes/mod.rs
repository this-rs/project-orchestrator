//! Episodic Memory module
//!
//! Implements the Episode model — the foundational data structure for
//! capturing complete cognitive episodes (Stimulus → Process → Outcome →
//! Validation → Lesson) from protocol runs and reasoning trees.
//!
//! Episodes are the bridge between local knowledge (notes, decisions,
//! protocol runs) and portable knowledge (SkillPackage v3).

pub mod collector;
pub mod models;

pub use models::{
    Episode, Lesson, Outcome, PortableEpisode, PortableLesson, PortableOutcome,
    PortableProcess, PortableStimulus, PortableValidation, Process, Stimulus, Validation,
};
