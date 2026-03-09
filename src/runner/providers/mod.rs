//! Trigger Providers — feed the TriggerEngine with activation signals.
//!
//! Each provider implements `TriggerProvider` and handles one `TriggerType`:
//! - `ScheduleProvider` — cron-based scheduling via tokio intervals
//! - `WebhookProvider` — external HTTP webhook receiver (GitHub, etc.)
//! - `EventProvider` — internal event bus subscriber (plan chaining)
//! - Chat — inline in the MCP handler (no dedicated provider)

pub mod event;
pub mod schedule;
pub mod webhook;

pub use event::EventProvider;
pub use schedule::ScheduleProvider;
pub use webhook::WebhookProvider;

use crate::runner::models::TriggerType;
use anyhow::Result;
use async_trait::async_trait;
use std::fmt::Debug;

/// Trait for trigger providers that feed the TriggerEngine.
///
/// Each provider is responsible for:
/// 1. Loading its triggers from Neo4j on `setup()`
/// 2. Monitoring its source (cron, webhook, events) for activation signals
/// 3. Calling `TriggerEngine::evaluate_and_prepare()` when conditions are met
/// 4. Cleaning up on `teardown()`
#[async_trait]
pub trait TriggerProvider: Send + Sync + Debug {
    /// Initialize the provider and start monitoring.
    async fn setup(&self) -> Result<()>;

    /// Stop monitoring and clean up resources.
    async fn teardown(&self) -> Result<()>;

    /// The type of triggers this provider handles.
    fn provider_type(&self) -> TriggerType;
}
