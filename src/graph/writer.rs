//! Analytics results writer.
//!
//! Batch-updates computed analytics scores back to Neo4j nodes.
//! Uses new `GraphStore` trait methods to write PageRank, betweenness,
//! clustering coefficient, community ID, and component ID as node properties.
//!
//! The write is idempotent: re-running analytics overwrites previous scores.
