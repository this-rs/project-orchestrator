//! Re-export feedback handlers for the API layer.
//!
//! This thin module re-exports the handlers from `crate::feedback::handlers`
//! to follow the api/ module naming convention (one handler file per domain).

pub use crate::feedback::handlers::*;
