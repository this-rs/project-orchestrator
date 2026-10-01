//! The vault: how a user hands secrets to agents without them leaking.
//!
//! What it guarantees, and what it does not:
//!
//! - Secrets are encrypted at rest under a key derived from the user's
//!   passphrase. The server cannot decrypt them on its own; after a restart the
//!   vault is locked until the user opens it again.
//! - A secret reaches an agent only through a grant: which secrets, for which
//!   session or project, until when.
//! - No API and no MCP tool ever returns a value, except the one endpoint agents
//!   call from inside a shell pipeline, under a grant — so the value flows into a
//!   command without passing through the model's context or the transcript.
//! - If a value shows up in agent output anyway, it is masked before it is
//!   stored, indexed or broadcast.
//!
//! What it cannot do: stop an agent that holds a secret from sending it
//! somewhere. A vault limits accidental exposure and bounds it in time and
//! scope; it is not a sandbox.

pub mod agent_cli;
pub mod crypto;
pub mod grants;
pub mod mask;
pub mod service;
pub mod store;

pub use grants::{authorize, AccessRequest, Denied, Grant, GrantBook, GrantScope, SecretSelector};
pub use mask::{Masker, SharedMasker};
pub use service::{RequestAnswer, SecretRequest, ServiceError, VaultService};
pub use store::{SecretMeta, Vault, VaultError, VaultStatus};
