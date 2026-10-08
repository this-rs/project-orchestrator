//! Structured references (`#kind:id`) from a chat message to a piece of work.
//!
//! A reference is data — an [`EntityRef`] `{kind, id}` — never a label: the
//! server alone resolves, authorizes and names it. Everything here is behind
//! the future `refs_v1` flag; nothing in this module changes how the chat
//! behaves today.
//!
//! * [`types`] — the wire shapes (`EntityRef`, `RefKind`, the `#kind:id` token);
//! * [`registry`] — which kinds exist, which are active, and the exhaustiveness
//!   proof against the two `EntityType` enums of the code base;
//! * [`validate`] — pure input validation (limits, excluded kinds);
//! * [`access`] — the policy: the only way to read a referenced entity is
//!   [`access::AccessPolicy::resolve_checked`];
//! * [`label`] — server-side labels (never read from a client, 80 characters);
//! * [`resolvers`] — one resolver per kind, delegating to the `get_*` of the store;
//! * [`search`] — `GET /api/refs/search`: query parsing, scope, policy;
//! * [`block`] — the trailing `<po-refs>` block carried inside the message text;
//! * [`wire`] — search response, `refs_resolved` event and error bodies.
//!
//! The golden fixtures in `tests/fixtures/refs/` are the contract with the
//! frontend; `tests/refs_contract.rs` keeps them and the types in step.

pub mod access;
pub mod block;
pub mod label;
pub mod registry;
pub mod resolvers;
pub mod search;
#[cfg(test)]
mod search_contract;
#[cfg(test)]
pub(crate) mod test_support;
pub mod types;
pub mod validate;
pub mod wire;

pub use types::{EntityRef, RefKind};

/// Version of the frontend/backend contract fixed by the golden fixtures.
pub const CONTRACT_VERSION: u32 = 1;
