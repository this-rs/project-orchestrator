//! Authentication module — Google OAuth + JWT
//!
//! Provides:
//! - JWT token encoding/decoding (`jwt` submodule)
//! - Google OAuth2 authorization code flow (`google` submodule)
//! - Axum middleware for route protection (`middleware` submodule)
//! - AuthUser extractor for handlers (`extractor` submodule)

pub mod extractor;
pub mod google;
pub mod jwt;
pub mod middleware;
