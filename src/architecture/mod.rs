//! Deriving the deployment topology from the code, instead of asking someone to
//! type it in.
//!
//! The hand-maintained alternative was tried and observably failed: six components
//! were entered once, never updated, two of them mistyped, and not a single
//! dependency was ever recorded — so the architecture view showed six orphan boxes.
//! Anything a human has to remember to update is, eventually, wrong.
//!
//! What each source contributes:
//!
//! - **Manifests** (`manifest`) are the only evidence that infrastructure exists.
//!   No file in the workspace *is* Neo4j; it appears solely as `neo4rs` in a
//!   `Cargo.toml`. The `catalogue` turns such a client package into the service it
//!   reaches, with its wire protocol.
//! - **Git and path dependencies** reveal that one workspace project builds on
//!   another — there is no `Project -> Project` relation in the graph to read.
//!
//! Every derived element carries where it came from (`Provenance`): a graph nobody
//! typed is only worth trusting if each edge can be traced to a file and a line.

pub mod catalogue;
pub mod compose;
pub mod derive;
pub mod manifest;
pub mod runtime_config;
pub mod sync;

use serde::{Deserialize, Serialize};

/// Where a derived element came from, carried to the UI so a generated edge can
/// be checked rather than taken on faith.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// How it was derived, e.g. "manifest".
    pub method: String,
    /// Manifest path relative to the project root.
    pub file: String,
    /// 1-based line, or 0 when it could not be located.
    pub line: u32,
    /// The package that implied this, e.g. "neo4rs".
    pub package: String,
}

/// Marks a component or edge as machine-derived.
///
/// `ComponentNode` has no field for this, and adding one means a schema migration,
/// so it rides in the existing free-form `config` JSON under this key. A derivation
/// may overwrite what it previously derived; it must never overwrite what a person
/// wrote, and this key is how the two are told apart.
pub const DERIVED_CONFIG_KEY: &str = "derived_from";
