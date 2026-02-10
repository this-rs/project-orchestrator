//! Embedded frontend serving via rust-embed.
//!
//! When compiled with `--features embedded-frontend`, the React SPA's dist/
//! directory is baked into the binary at compile time. This module provides
//! an Axum handler that serves those embedded assets with correct Content-Type
//! and SPA fallback (index.html for unmatched routes).
//!
//! # Build requirements
//!
//! The `dist/` directory must exist at compile time (can be empty — a `.gitkeep`
//! is sufficient). For a functional build, run `npm run build` in the frontend
//! project first, then copy the output to `dist/` before `cargo build`.
//!
//! # Usage
//!
//! ```rust,ignore
//! // In create_router():
//! #[cfg(feature = "embedded-frontend")]
//! router.fallback(embedded_frontend::serve_embedded)
//! ```

use axum::{
    body::Body,
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use rust_embed::RustEmbed;

/// Embedded frontend assets compiled from the dist/ directory.
///
/// The `folder` path is resolved relative to the crate root at compile time.
/// If dist/ is empty, the struct will have no files (serve_embedded returns 404).
#[derive(RustEmbed)]
#[folder = "dist/"]
struct FrontendAssets;

/// Axum handler that serves embedded frontend assets.
///
/// - If the requested path matches a file in dist/, serves it with the correct MIME type.
/// - Otherwise, falls back to index.html for SPA client-side routing.
/// - Returns 404 only if index.html itself is missing (dist/ was empty at compile time).
pub async fn serve_embedded(uri: Uri) -> impl IntoResponse {
    let path = uri.path().trim_start_matches('/');

    // Try to serve the exact file first
    if !path.is_empty() {
        if let Some(file) = FrontendAssets::get(path) {
            return serve_file(path, &file.data);
        }
    }

    // Fallback to index.html for SPA routing
    match FrontendAssets::get("index.html") {
        Some(file) => serve_file("index.html", &file.data),
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Frontend not embedded (dist/ was empty at build time)"))
            .unwrap(),
    }
}

/// Build an HTTP response with the correct Content-Type for the given file path.
fn serve_file(path: &str, data: &[u8]) -> Response {
    let mime = mime_guess::from_path(path)
        .first_or_octet_stream()
        .to_string();

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, mime)
        .body(Body::from(data.to_vec()))
        .unwrap()
}
