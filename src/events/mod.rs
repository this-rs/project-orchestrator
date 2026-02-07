//! CRUD event system for real-time WebSocket notifications
//!
//! This module provides:
//! - `CrudEvent` — typed events emitted after every mutation
//! - `EventBus` — broadcast channel for distributing events to WebSocket clients

mod bus;
mod types;

pub use bus::EventBus;
pub use types::{CrudAction, CrudEvent, EntityType, RelatedEntity};
