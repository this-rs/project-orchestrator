//! # Analytics Engine — rs-stats Integration
//!
//! Wraps the `rs-stats` library with Orchestrator-specific helpers for:
//! - Distribution analysis of graph metrics (PageRank, risk scores, etc.)
//! - Hypothesis testing (ANOVA for community fragility, t-tests for comparisons)
//! - Adaptive thresholds derived from actual data distributions
//!
//! All public functions are **panic-free**: errors are returned as `Option` or
//! logged and replaced with sensible defaults so callers never crash.

pub mod distribution;
pub mod hypothesis;
// pub mod regression; // Phase 2: churn_score prediction via multiple linear regression
