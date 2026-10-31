//! # Event System — Real-time CRUD & Graph Notifications
//!
//! This module provides the complete event infrastructure for the Project Orchestrator.
//! Events are emitted **after every successful mutation** and broadcast to WebSocket
//! clients for real-time UI updates and to NATS for cross-instance sync.
//!
//! ## Architecture
//!
//! ```text
//! Handler (mutation) ──► HybridEmitter ──┬──► EventBus (local broadcast) ──► WebSocket clients
//!                                        └──► NatsEmitter (inter-process) ──► Other PO instances
//! ```
//!
//! ## Components
//!
//! - [`CrudEvent`] — typed event emitted after every entity mutation
//! - [`GraphEvent`] — fine-grained graph mutation event for visualization
//! - [`EventBus`] — `tokio::sync::broadcast` channel for local distribution
//! - [`NatsEmitter`] — NATS JetStream publisher for cross-instance sync
//! - [`HybridEmitter`] — combines local + NATS, implements [`EventEmitter`]
//! - [`EventEmitter`] trait — polymorphic dispatch with convenience methods
//!
//! ## Coverage Matrix — EntityType × CrudAction
//!
//! | EntityType        | Created | Updated | Deleted | Linked | Unlinked | Synced | StatusChanged | Handler(s)                    |
//! |-------------------|---------|---------|---------|--------|----------|--------|---------------|-------------------------------|
//! | Project           | ✅      | ✅      | ✅      | —      | —        | ✅     | —             | handlers.rs, project_handlers |
//! | Plan              | ✅      | ✅      | ✅      | —      | —        | —      | ✅            | handlers.rs                   |
//! | Task              | ✅      | ✅      | ✅      | —      | —        | —      | ✅            | handlers.rs                   |
//! | Step              | —       | ✅      | ✅      | —      | —        | —      | —             | handlers.rs                   |
//! | Decision          | ✅      | —       | —       | ✅     | ✅       | —      | —             | handlers.rs (AFFECTS)         |
//! | Constraint        | ✅      | ✅      | ✅      | —      | —        | —      | —             | handlers.rs                   |
//! | Commit            | ✅      | —       | —       | ✅     | —        | —      | —             | handlers.rs                   |
//! | Release           | ✅      | ✅      | ✅      | ✅     | —        | —      | ✅            | handlers.rs                   |
//! | Milestone         | ✅      | ✅      | ✅      | ✅     | —        | —      | ✅            | handlers.rs                   |
//! | Environment       | ✅      | ✅      | ✅      | —      | —        | —      | —             | environment_handlers.rs       |
//! | Deployment        | ✅      | ✅      | —       | —      | —        | —      | ✅            | environment_handlers.rs       |
//! | Workspace         | —       | —       | —       | —      | —        | —      | —             | (not yet wired)               |
//! | WorkspaceMilestone| —       | —       | —       | —      | —        | —      | —             | (not yet wired)               |
//! | Resource          | —       | —       | —       | —      | —        | —      | —             | (not yet wired)               |
//! | Component         | —       | —       | —       | —      | —        | —      | —             | (not yet wired)               |
//! | Note              | ✅      | ✅      | ✅      | ✅     | ✅       | —      | —             | note_handlers.rs              |
//! | ChatSession       | ✅      | —       | ✅      | —      | —        | —      | —             | chat_handlers.rs              |
//! | ProtocolRun       | ✅      | ✅      | ✅      | —      | —        | —      | ✅            | protocol_handlers.rs          |
//! | Runner            | ✅      | ✅      | —       | —      | —        | —      | —             | runner.rs                     |
//! | Alert             | ✅      | —       | —       | —      | —        | —      | —             | (system-generated)            |
//! | Persona           | ✅      | ✅      | ✅      | ✅     | ✅       | —      | ✅            | persona_handlers.rs           |
//! | Skill             | ✅      | ✅      | ✅      | ✅     | ✅       | —      | ✅            | skill_handlers.rs             |
//! | Protocol          | ✅      | ✅      | ✅      | ✅     | ✅       | —      | —             | protocol_handlers.rs          |
//! | FeatureGraph      | ✅      | —       | ✅      | —      | —        | —      | —             | code_handlers.rs              |
//! | Episode           | ✅      | —       | —       | —      | —        | —      | —             | episode_handlers.rs           |
//! | AnalysisProfile   | ✅      | —       | ✅      | —      | —        | —      | —             | profile_handlers.rs           |
//! | Trigger           | ✅      | ✅      | ✅      | —      | —        | —      | —             | handlers.rs                   |
//! | TopologyRule      | ✅      | —       | ✅      | —      | —        | —      | —             | code_handlers.rs              |
//!
//! ## GraphEvent Coverage
//!
//! | GraphEventType    | Layer      | Emitted by                                          |
//! |-------------------|------------|------------------------------------------------------|
//! | NodeCreated       | Knowledge  | note creation, decision creation                     |
//! | NodeUpdated       | Neural     | energy/staleness updates, reinforcement              |
//! | EdgeCreated       | Neural     | synapse creation (spreading activation)              |
//! | EdgeCreated       | Knowledge  | AFFECTS, LINKED_TO creation                          |
//! | EdgeCreated       | Fabric     | CO_CHANGED computation                               |
//! | EdgeRemoved       | Neural     | synapse pruning (decay)                              |
//! | EdgeRemoved       | Knowledge  | AFFECTS, LINKED_TO removal                           |
//! | Reinforcement     | Neural     | reinforce_neurons (energy boost)                     |
//! | Activation        | Neural     | spreading activation search                          |
//! | CommunityChanged  | Skills     | Louvain recomputation                                |
//! | BatchCreated      | Code       | project sync completion (bulk IMPORTS/CALLS)          |
//! | ScoresUpdated     | Fabric     | update_fabric_scores, bootstrap_knowledge_fabric     |
#[allow(unused_imports)]
use crate::neo4j::traits::*;

pub mod activity;
pub mod attention;
pub mod builtin_triggers;
mod bus;
pub mod graph;
mod hybrid;
pub mod ingestion;
pub mod nats;
#[cfg(test)]
pub(crate) mod nats_broker_test;
mod notifier;
pub mod reactions;
pub mod reactor;
pub mod transport;
pub mod trigger;
pub mod trigger_routing;
mod types;

pub use activity::{
    ActivityCategory, ActivityEvent, ActivityHub, ACTIVITY_BROADCAST_CAPACITY,
    ACTIVITY_BUFFER_CAPACITY,
};
pub use bus::EventBus;
pub use graph::{ActivationTarget, GraphEvent, GraphEventType, GraphLayer};
pub use hybrid::HybridEmitter;
pub use ingestion::{
    IngestionCounters, IngestionPhase, IngestionProgress, IngestionReporter, IngestionSummary,
    IngestionSynced, INGESTION_PROGRESS_INTERVAL,
};
pub use nats::{
    connect_nats, ChatRpcRequest, ChatRpcResponse, NatsEmitter, RelayFailure, StreamingSnapshot,
};
#[allow(deprecated)]
pub use notifier::EventNotifier;
pub use reactions::register_builtin_reactions;
pub use reactor::{EventReactor, ReactorBuilder, ReactorCounters, ReactorStats};
pub use transport::{Message, Subject, Subscriber};
pub use trigger::EventTrigger;
pub use trigger_routing::{RoutingContext, RoutingDecision, TriggerRouter};
pub use types::TransportState;
pub use types::{CrudAction, CrudEvent, EntityType, EventEmitter, RelatedEntity};
