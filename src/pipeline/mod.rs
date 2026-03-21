//! # Pipeline — Quality Gates & Protocol Composition
//!
//! This module provides the infrastructure for building and executing
//! verification pipelines from plans. A pipeline is a hierarchical protocol
//! with quality gates that verify each stage of plan execution.
//!
//! ## Components
//!
//! - [`gates`] — Reusable quality gate library (cargo-check, cargo-test, coverage, etc.)
//! - [`composer`] — Dynamic protocol composer that generates pipelines from plans
//! - [`runner`] — Pipeline execution orchestrator
//! - [`regression`] — Loop and regression detection
//! - [`progress`] — Objective progress measurement

pub mod composer;
pub mod episode_adapter;
pub mod evolve;
pub mod feedback;
pub mod gates;
pub mod materialize;
pub mod metrics;
pub mod progress;
pub mod regression;
pub mod runner;
pub mod skill_injector;
pub mod wave_executor;
