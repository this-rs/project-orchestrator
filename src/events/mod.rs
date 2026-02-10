//! CRUD event system for real-time WebSocket notifications
//!
//! This module provides:
//! - `CrudEvent` — typed events emitted after every mutation
//! - `EventBus` — broadcast channel for distributing events to WebSocket clients
//! - `NatsEmitter` — NATS-based emitter for inter-process event sync
//! - `HybridEmitter` — combines local broadcast + optional NATS

mod bus;
mod hybrid;
pub mod nats;
mod notifier;
mod types;

pub use bus::EventBus;
pub use hybrid::HybridEmitter;
pub use nats::{connect_nats, NatsEmitter};
#[allow(deprecated)]
pub use notifier::EventNotifier;
pub use types::{CrudAction, CrudEvent, EntityType, EventEmitter, RelatedEntity};
