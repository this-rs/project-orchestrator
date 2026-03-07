//! ReasoningTree module
//!
//! Implements a dynamic decision tree that emerges from the knowledge graph
//! in response to a natural language query. The tree is built in 3 phases:
//!
//! 1. **Activation**: Embed the query → vector search across notes, decisions, skills
//! 2. **Propagation**: Traverse SYNAPSE, LINKED_TO, AFFECTS, CO_CHANGED edges with scoring
//! 3. **Cristallisation**: Transform the activated subgraph into a decision tree with actions
//!
//! Trees are **ephemeral** — cached in-memory with TTL, not persisted to Neo4j.
//! A feedback loop (`reason_feedback`) reinforces the underlying synapses,
//! making future trees for similar queries more accurate over time.

pub mod cache;
pub mod engine;
pub mod models;

pub use cache::ReasoningTreeCache;
pub use engine::{ReasoningTreeEngine, SeedNode, SeedSource};
pub use models::{Action, EntitySource, ReasoningNode, ReasoningTree, ReasoningTreeConfig};
