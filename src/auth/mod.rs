//! Authentication module — OIDC, password auth, JWT
//!
//! Provides:
//! - JWT token encoding/decoding (`jwt` submodule)
//! - Generic OIDC client (`oidc` submodule) — works with Google, Microsoft, Okta, etc.
//! - Legacy Google OAuth2 client (`google` submodule) — deprecated, use `oidc` instead
//! - Axum middleware for route protection (`middleware` submodule)
//! - AuthUser extractor for handlers (`extractor` submodule)

pub mod extractor;
pub mod google;
pub mod jwt;
pub mod middleware;
pub mod oidc;
