//! Orchestrator module for coordinating agents

pub mod context;
pub mod planner;
pub mod runner;
pub mod watcher;

pub use context::ContextBuilder;
pub use planner::ImplementationPlanner;
pub use runner::Orchestrator;
pub use watcher::spawn_project_watcher_bridge;
pub use watcher::FileWatcher;
