//! Authentication module — Google OAuth + JWT
//!
//! Provides:
//! - JWT token encoding/decoding (`jwt` submodule)
//! - Google OAuth2 authorization code flow (`google` submodule)

pub mod google;
pub mod jwt;
