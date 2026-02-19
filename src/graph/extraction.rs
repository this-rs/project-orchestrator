//! Neo4j → petgraph extraction.
//!
//! Converts the knowledge graph stored in Neo4j into an in-memory `petgraph::DiGraph`
//! suitable for analytics computation. Uses the `GraphStore` trait (not raw Cypher)
//! to fetch nodes and relationships.
//!
//! The extraction is project-scoped: given a project ID, it fetches all code entities
//! (File, Function, Struct, Trait, Enum) and their relationships (IMPORTS, CALLS,
//! DEFINES, IMPLEMENTS_TRAIT, IMPLEMENTS_FOR) to build the directed graph.
