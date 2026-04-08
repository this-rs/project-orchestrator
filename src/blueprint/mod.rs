//! Blueprint Engine — Scalable reusable pattern library
//!
//! Blueprints are curated, structured knowledge templates (markdown files)
//! that capture reusable patterns, architectures, workflows, and processes.
//! They are ingested from a git repository, parsed into tiered content,
//! and made available to agents via semantic search and manifest resolution.
//!
//! # Architecture
//!
//! ```text
//! Blueprint repo (.md files)
//!   ──► Parser (extract metadata, tiers)
//!     ──► Neo4j (Blueprint nodes + relations)
//!       ──► Resolver (project ↔ blueprint matching)
//!         ──► Manifest (cached per-project resolution)
//!           ──► Agent warm-up (task matching, tier loading)
//! ```
//!
//! # Tiered Loading (token budget control)
//!
//! - **Tier 1 — Catalog** (~30 tokens): name + one-line description
//! - **Tier 2 — Summary** (~200 tokens): TL;DR + gotchas + checklist
//! - **Tier 3 — Full** (~800 tokens): complete markdown with code examples
//!
//! # Taxonomy
//!
//! Blueprints are classified by **scope** (what level of work they address)
//! and **category** (what domain they belong to):
//!
//! | Scope        | When used                | Example                     |
//! |--------------|--------------------------|------------------------------|
//! | Scaffolding  | New project bootstrap    | Flutter Project Kickoff       |
//! | Feature      | New feature implementation | OAuth Auth Flow             |
//! | Pattern      | Code-level patterns       | AsyncNotifier Lifecycle      |
//! | Process      | Workflow/CI/CD            | PR Strategy, Release Checklist|

pub mod models;
pub mod parser;

pub use models::*;
pub use parser::{extract_references, parse_blueprint, ParsedBlueprint};

/// Maximum length for blueprint name
pub const MAX_BLUEPRINT_NAME_LEN: usize = 200;
/// Maximum length for blueprint description (TL;DR)
pub const MAX_BLUEPRINT_DESCRIPTION_LEN: usize = 2000;
/// Maximum length for tier2 content (summary)
pub const MAX_BLUEPRINT_TIER2_LEN: usize = 10_000;
/// Maximum length for tier3 content (full markdown)
pub const MAX_BLUEPRINT_TIER3_LEN: usize = 100_000;
/// Maximum number of tags per blueprint
pub const MAX_BLUEPRINT_TAGS: usize = 30;
/// Maximum length for a single tag
pub const MAX_BLUEPRINT_TAG_LEN: usize = 100;
/// Maximum number of stack entries
pub const MAX_BLUEPRINT_STACK: usize = 20;
