//! Protocol module — Pattern Federation FSM engine
//!
//! Protocols are finite state machines (FSMs) that model repeatable workflows.
//! They are the core building block of Pattern Federation, enabling the system
//! to define, track, and execute structured processes.
//!
//! # Architecture
//!
//! Each protocol consists of:
//! - **States**: Named nodes in the FSM (start, intermediate, terminal)
//! - **Transitions**: Directed edges with triggers and optional guards
//! - **Category**: System (auto-triggered) or Business (agent-driven)
//!
//! # Neo4j Graph Model
//!
//! ```text
//! (Protocol)-[:HAS_STATE]->(ProtocolState)
//! (Protocol)-[:HAS_TRANSITION]->(ProtocolTransition)
//! (Protocol)-[:BELONGS_TO]->(Project)
//! (Protocol)-[:BELONGS_TO_SKILL]->(Skill)
//! (ProtocolState)-[:REFERENCES]->(Note)
//! ```

pub mod engine;
pub mod executor;
pub mod generator;
pub mod hooks;
pub mod models;
pub mod routing;
pub mod runner;
#[cfg(test)]
mod runner_tests;
pub mod seed;

pub use models::*;
