//! HTTP API for document attachments — upload, listing, retrieval, deletion.
//!
//! This module is the seam where the three halves of the document stack meet:
//! [`crate::documents::extract`] turns bytes into text,
//! [`crate::documents::chunk`] tiles that text, [`crate::documents::store`]
//! keeps the original bytes, and `neo4j::document` files the result in the
//! knowledge graph. Nothing here re-implements any of that; the value it adds
//! is ordering, error translation, and refusing to lie to the uploader.
//!
//! ## Why the status codes are spelled out
//!
//! An upload can fail in ways that mean genuinely different things to whoever
//! sent the file, and collapsing them into 500 is how a user ends up retrying
//! a PDF that this build simply cannot read:
//!
//! | Situation                                   | Status | What the caller should do        |
//! |---------------------------------------------|--------|----------------------------------|
//! | Over the blob-store cap                     | 413    | Send a smaller file              |
//! | Nothing recognises the bytes                | 201    | Stored as an opaque attachment   |
//! | Format known, feature not compiled in       | 501    | Rebuild the server; file is fine |
//! | Recognised but broken, or empty, or no file | 422    | The file itself is the problem   |
//!
//! 501 rather than 415 for a disabled feature: the *format* is supported, this
//! *build* is not. Telling someone their valid PDF is unsupported sends them
//! looking for a fault in the file that is not there.
//!
//! ## Files nobody can read are still attachments
//!
//! Knowledge work is full of files the platform has no reason to understand — a
//! `.zip` of deliverables, a screenshot, a `.mov`. Rejecting them with 415 would
//! make "attach it to the task" impossible. When no extractor recognises the
//! bytes the file is stored as a `binary` document instead: same size limit,
//! same content-addressed blob, `extracted: false`, no chunks, and a warning
//! saying it will not be searchable. The 415 status (and
//! [`DocumentError::UnsupportedFormat`]) remains for callers that need to
//! distinguish, but the upload pipeline no longer produces it.
//!
//! ## Why the body limit is set explicitly
//!
//! axum caps request bodies at 2 MiB by default. That is a sane default for
//! JSON and a silent disaster for file upload: an ordinary PDF trips it and the
//! rejection says nothing about size. The upload route therefore carries a
//! [`axum::extract::DefaultBodyLimit`] derived from
//! [`MAX_BLOB_BYTES`](crate::documents::MAX_BLOB_BYTES) — see
//! [`upload_body_limit`], which is what `routes.rs` installs.
//!
//! ## The filename is hostile input
//!
//! It comes from the client, so it is never used to build a path — paths come
//! from the content hash, via [`DocumentStore::path_for`]. It is sanitised
//! before being stored or echoed into a header, because it also ends up in a
//! `Content-Disposition` and in the graph.

use super::handlers::OrchestratorState;
use crate::documents::{
    chunk_text, ChunkConfig, DocumentFormat, DocumentStore, ExtractError, ExtractedText,
    ExtractorRegistry, StoreError, MAX_BLOB_BYTES,
};
use crate::embeddings::EmbeddingProvider;
use crate::neo4j::document::{Document, DocumentChunk};
use crate::neo4j::GraphStore;
use crate::notes::EntityType;
use axum::{
    extract::{Multipart, Path, Query, State},
    http::{header, StatusCode},
    response::IntoResponse,
    Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, LazyLock};
use uuid::Uuid;

// ============================================================================
// Constants
// ============================================================================

/// Headroom over [`MAX_BLOB_BYTES`] for multipart framing.
///
/// The body carries boundaries, per-part headers and the `project_id` /
/// `session_id` fields on top of the file itself. Without this slack a file of
/// exactly the maximum blob size would be rejected by the transport layer with
/// a message about the *body*, when the honest answer — the one
/// [`DocumentStore::put`] gives — is about the *blob*. One MiB is far more than
/// multipart framing ever needs and far less than a meaningful bypass.
const MULTIPART_OVERHEAD_BYTES: u64 = 1024 * 1024;

/// How many chunks are embedded per [`EmbeddingProvider::embed_batch`] call.
///
/// The requirement is "not one call per chunk"; this is not "one call for
/// everything" either, because a 500-page PDF would then hand a single request
/// several megabytes of text and hold every vector in memory at once. Batching
/// turns an O(chunks) round-trip count into O(chunks / 64) while keeping each
/// request a size a provider will actually accept.
const EMBED_BATCH_SIZE: usize = 64;

/// Upper bound on a stored filename, in bytes.
///
/// Filenames arrive from the client and are displayed everywhere afterwards. A
/// 40 KB "filename" is not a filename, it is a payload.
const MAX_FILENAME_BYTES: usize = 255;

/// Fallback when the client sends a filename that sanitises to nothing.
const FALLBACK_FILENAME: &str = "upload";

/// The extractors this build has, probed in registry order.
///
/// Built once: constructing it per request is cheap but pointless, and a single
/// instance makes "what can this build read" one answer rather than many.
static EXTRACTORS: LazyLock<ExtractorRegistry> = LazyLock::new(ExtractorRegistry::with_builtins);

/// Maximum request body accepted by `POST /api/documents`, in bytes.
///
/// Exposed as a function rather than a constant so `routes.rs` states the
/// dependency on the blob-store cap explicitly instead of repeating a number
/// that would drift the first time the cap moves.
pub fn upload_body_limit() -> usize {
    usize::try_from(MAX_BLOB_BYTES.saturating_add(MULTIPART_OVERHEAD_BYTES)).unwrap_or(usize::MAX)
}

// ============================================================================
// Errors
// ============================================================================

/// Failure modes of the document API, each mapped to the status code that tells
/// the caller what actually happened. See the module docs for the rationale.
#[derive(Debug, thiserror::Error)]
pub enum DocumentError {
    /// The blob is larger than the store accepts → 413.
    #[error("{0}")]
    TooLarge(String),

    /// No extractor recognised the content → 415.
    #[error("{0}")]
    UnsupportedFormat(String),

    /// The format is known but this build cannot read it → 501.
    #[error("{0}")]
    FeatureDisabled(String),

    /// The request or the file is malformed → 422.
    #[error("{0}")]
    Unprocessable(String),

    /// The request is well-formed but asks for something impossible → 400.
    #[error("{0}")]
    BadRequest(String),

    /// No such document, or its blob has gone missing → 404.
    #[error("{0}")]
    NotFound(String),

    /// Anything else — a graph write that failed, a disk that filled up → 500.
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl DocumentError {
    fn status(&self) -> StatusCode {
        match self {
            Self::TooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Self::UnsupportedFormat(_) => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Self::FeatureDisabled(_) => StatusCode::NOT_IMPLEMENTED,
            Self::Unprocessable(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for DocumentError {
    fn into_response(self) -> axum::response::Response {
        let status = self.status();
        // Same envelope as `AppError`, so a client needs one error shape.
        let message = match &self {
            // Never leak internals (graph errors, blob paths) to the client.
            Self::Internal(e) => {
                tracing::error!(error = ?e, "internal server error (documents)");
                crate::api::handlers::INTERNAL_ERROR_MESSAGE.to_string()
            }
            _ => self.to_string(),
        };
        let body = Json(serde_json::json!({ "error": message }));
        (status, body).into_response()
    }
}

impl From<ExtractError> for DocumentError {
    fn from(err: ExtractError) -> Self {
        match err {
            // "Nobody here reads this" — a different answer from "broken file".
            ExtractError::UnsupportedFormat { .. } => Self::UnsupportedFormat(err.to_string()),
            // The error text already names the missing cargo feature; keep it,
            // it is the only actionable part of the message.
            ExtractError::FeatureDisabled { .. } => Self::FeatureDisabled(err.to_string()),
            ExtractError::Empty | ExtractError::Malformed { .. } => {
                Self::Unprocessable(err.to_string())
            }
        }
    }
}

impl From<StoreError> for DocumentError {
    fn from(err: StoreError) -> Self {
        match err {
            StoreError::TooLarge { .. } => Self::TooLarge(err.to_string()),
            // A digest that fails validation came from a URL segment, so this is
            // the caller's mistake, not ours. The store deliberately does not
            // echo the offending value; neither do we.
            StoreError::InvalidDigest { .. } => Self::BadRequest(err.to_string()),
            StoreError::NotFound { .. } => Self::NotFound(err.to_string()),
            other => Self::Internal(anyhow::Error::new(other)),
        }
    }
}

// ============================================================================
// Request / response types
// ============================================================================

/// Multipart fields other than the file itself.
#[derive(Debug, Default)]
struct UploadFields {
    file: Option<Vec<u8>>,
    filename: Option<String>,
    project_id: Option<Uuid>,
    session_id: Option<Uuid>,
    entity_type: Option<String>,
    entity_id: Option<String>,
}

/// Query parameters for `GET /api/documents`.
#[derive(Debug, Deserialize, Default)]
pub struct DocumentsListQuery {
    pub project_id: Option<Uuid>,
    pub session_id: Option<Uuid>,
    /// With `entity_id`: documents attached to that entity (a task, a plan…).
    pub entity_type: Option<String>,
    pub entity_id: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// What `POST /api/documents` returns — the frozen upload contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UploadedDocument {
    pub id: Uuid,
    pub filename: String,
    pub format: DocumentFormat,
    pub size_bytes: u64,
    pub sha256: String,
    pub page_count: usize,
    pub chunk_count: usize,
    /// Extraction warnings **plus** anything that went wrong after extraction
    /// but did not sink the ingestion — a failed embedding pass, a document
    /// that yielded no text at all.
    ///
    /// Wider than the `warnings` persisted on the node, which stay exactly what
    /// the extractor reported. The node records what the document *is*; this
    /// records what this particular ingestion *did*.
    pub warnings: Vec<String>,
    pub created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<Uuid>,
    /// `false` when the file was stored as an opaque attachment: it can be
    /// downloaded but has no text and no chunks.
    pub extracted: bool,
    pub mime_type: String,
}

/// A document list page.
#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentListResponse {
    pub items: Vec<Document>,
    pub total: usize,
}

/// One chunk as the API exposes it.
///
/// Deliberately not `DocumentChunk`: that carries the embedding, which is
/// hundreds of floats per chunk that no caller of this endpoint has a use for,
/// and it names the offsets `start_byte` / `end_byte` where the contract says
/// `start` / `end`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkView {
    pub id: Uuid,
    pub text: String,
    pub start: usize,
    pub end: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<usize>,
    pub ordinal: usize,
}

impl From<&DocumentChunk> for ChunkView {
    fn from(chunk: &DocumentChunk) -> Self {
        Self {
            id: chunk.id,
            text: chunk.text.clone(),
            start: chunk.start_byte,
            end: chunk.end_byte,
            page: chunk.page,
            ordinal: chunk.ordinal,
        }
    }
}

/// A document's chunks, in order.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChunkListResponse {
    pub items: Vec<ChunkView>,
    pub total: usize,
}

// ============================================================================
// Filename sanitisation
// ============================================================================

/// Reduce a client-supplied filename to something safe to store and display.
///
/// This is *not* what keeps the filesystem safe — nothing here ever reaches a
/// path; blobs are addressed by content hash. It is what keeps the filename
/// from being a vector everywhere else it travels: a header value (CR/LF),
/// a log line (control characters), a UI label (unbounded length).
///
/// Directory separators are dropped rather than escaped, so `../../etc/passwd`
/// becomes `passwd`: the traversal intent does not survive, and the part a
/// human would recognise does.
fn sanitize_filename(raw: &str) -> String {
    let base = raw
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(raw)
        .trim_matches(|c: char| c.is_whitespace() || c == '.');

    let cleaned: String = base
        .chars()
        .filter(|c| !c.is_control() && *c != '"' && *c != '\\')
        .collect();

    // Truncate on a char boundary — a filename is UTF-8 and slicing blindly
    // would panic on a multi-byte name.
    let mut out = String::new();
    for c in cleaned.chars() {
        if out.len() + c.len_utf8() > MAX_FILENAME_BYTES {
            break;
        }
        out.push(c);
    }

    if out.trim().is_empty() {
        FALLBACK_FILENAME.to_string()
    } else {
        out
    }
}

/// An ASCII-only rendering of a filename, for `Content-Disposition`.
///
/// Non-ASCII is replaced rather than percent-encoded: the header is a
/// convenience, and the authoritative filename is in the JSON body.
fn header_safe_filename(name: &str) -> String {
    let out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_graphic() && c != '"' && c != '\\' {
                c
            } else if c == ' ' {
                ' '
            } else {
                '_'
            }
        })
        .collect();
    if out.trim().is_empty() {
        FALLBACK_FILENAME.to_string()
    } else {
        out
    }
}

/// The media type implied by a format alone.
fn content_type_for(format: DocumentFormat) -> &'static str {
    match format {
        DocumentFormat::PlainText => "text/plain; charset=utf-8",
        DocumentFormat::Docx => {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }
        DocumentFormat::Pdf => "application/pdf",
        DocumentFormat::Xlsx => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        DocumentFormat::Pptx => {
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        }
        DocumentFormat::Binary => "application/octet-stream",
    }
}

/// The media type to record for an upload.
///
/// Derived here, from the content and a fixed table — never copied from the
/// client's `Content-Type`, because the value is later served back in a
/// response header and an uploader must not be able to pick it. Anything that a
/// browser would render as active content (HTML, SVG, scripts) is deliberately
/// absent from the table and comes out as `application/octet-stream`.
fn mime_for(format: DocumentFormat, filename: &str, bytes: &[u8]) -> &'static str {
    if format != DocumentFormat::Binary {
        return content_type_for(format);
    }
    // Magic bytes first, extension only to tell apart what shares a container.
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return "image/png";
    }
    if bytes.starts_with(b"\xff\xd8\xff") {
        return "image/jpeg";
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return "image/gif";
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return "image/webp";
    }
    if bytes.starts_with(b"PK\x03\x04") {
        let ext = filename
            .rsplit_once('.')
            .map(|(_, e)| e.to_ascii_lowercase());
        return match ext.as_deref() {
            Some("docx") => content_type_for(DocumentFormat::Docx),
            Some("xlsx") => content_type_for(DocumentFormat::Xlsx),
            Some("pptx") => content_type_for(DocumentFormat::Pptx),
            _ => "application/zip",
        };
    }
    if bytes.starts_with(b"\x1f\x8b") {
        return "application/gzip";
    }
    if bytes.starts_with(b"%PDF-") {
        return "application/pdf";
    }
    "application/octet-stream"
}

// ============================================================================
// Ingestion pipeline
// ============================================================================

/// Everything the ingestion pipeline needs, passed explicitly.
///
/// The handlers build one of these from `OrchestratorState`; the tests build
/// one from a `MockGraphStore` and a temp directory. That is the whole reason
/// the pipeline is not written inline in the handler: the interesting behaviour
/// (batching, deduplication, warning propagation) is then testable without a
/// live Neo4j or an HTTP server.
pub struct Ingestor<'a> {
    pub graph: &'a dyn GraphStore,
    pub store: &'a DocumentStore,
    /// `None` when embeddings are disabled. A supported state: the document is
    /// still ingested, it is just not semantically searchable.
    pub embeddings: Option<&'a dyn EmbeddingProvider>,
    pub chunk_config: ChunkConfig,
}

impl<'a> Ingestor<'a> {
    /// Build an ingestor from the server state.
    pub fn from_state(state: &'a OrchestratorState, store: &'a DocumentStore) -> Ingestor<'a> {
        Ingestor {
            graph: state.orchestrator.neo4j(),
            store,
            embeddings: state
                .orchestrator
                .embedding_provider()
                .map(|p| p.as_ref() as &dyn EmbeddingProvider),
            chunk_config: ChunkConfig::default(),
        }
    }

    /// Run the full pipeline: extract → chunk → embed (batched) → store → graph.
    ///
    /// ## On deduplication
    ///
    /// The content hash is *not* used to short-circuit into an existing
    /// document. Two uploads of the same bytes produce two `Document` nodes and
    /// one blob on disk, on purpose: the blob is content, the document is an
    /// act of attaching that content to a project or a conversation, and two
    /// people attaching the same PDF have done two different things. The saving
    /// is in the bytes, which is where the cost is.
    pub async fn ingest(
        &self,
        bytes: Vec<u8>,
        filename: &str,
        project_id: Option<Uuid>,
        session_id: Option<Uuid>,
    ) -> Result<UploadedDocument, DocumentError> {
        self.ingest_for(bytes, filename, project_id, session_id, None)
            .await
    }

    /// [`Self::ingest`], additionally attaching the document to a knowledge
    /// entity (a task, a plan, a milestone…) with a `LINKED_TO` edge.
    ///
    /// The link is a no-op in the graph when the entity does not exist, exactly
    /// like `link_note_to_entity`; the document is still stored and reachable
    /// through its project or by id.
    pub async fn ingest_for(
        &self,
        bytes: Vec<u8>,
        filename: &str,
        project_id: Option<Uuid>,
        session_id: Option<Uuid>,
        entity: EntityLink,
    ) -> Result<UploadedDocument, DocumentError> {
        // Size first: the store would reject this anyway, but only after we had
        // paid to extract and chunk a file we were always going to refuse.
        let size_bytes = bytes.len() as u64;
        let limit = self.store.max_blob_bytes();
        if size_bytes > limit {
            return Err(DocumentError::TooLarge(
                StoreError::TooLarge {
                    size: size_bytes,
                    limit,
                }
                .to_string(),
            ));
        }

        let filename = sanitize_filename(filename);
        let extracted: ExtractedText = match EXTRACTORS.extract(&bytes, Some(&filename)) {
            Ok(extracted) => extracted,
            // Nobody reads this — but it is still a legitimate thing to attach.
            // Only "no extractor claims it" falls back: a recognised-but-broken
            // file (422) or a disabled feature (501) is a different answer.
            Err(ExtractError::UnsupportedFormat { .. }) => ExtractedText {
                text: String::new(),
                format: DocumentFormat::Binary,
                pages: Vec::new(),
                warnings: Vec::new(),
            },
            Err(other) => return Err(other.into()),
        };
        let is_opaque = extracted.format == DocumentFormat::Binary;
        let mime_type = mime_for(extracted.format, &filename, &bytes);

        let text_chunks = chunk_text(&extracted.text, &self.chunk_config);
        let chunks: Vec<DocumentChunk> = text_chunks
            .iter()
            .enumerate()
            .map(|(ordinal, chunk)| {
                let page = extracted.page_of(chunk.interval.start);
                DocumentChunk::from_text_chunk(chunk, ordinal, page)
            })
            .collect();

        // Warnings the caller sees; warnings the node keeps stay exactly what
        // the extractor said (see `UploadedDocument::warnings`).
        let mut response_warnings = extracted.warnings.clone();
        if is_opaque {
            response_warnings.push(
                "this file type is not readable by the platform — it is stored and can be \
                 downloaded, but it has no text and will not appear in search"
                    .to_string(),
            );
        } else if chunks.is_empty() {
            response_warnings.push(
                "no text could be extracted from this document — it is stored, but it will not \
                 appear in search or be readable by an agent"
                    .to_string(),
            );
        }

        let embeddings = match self.embed_chunks(&chunks).await {
            Ok(vectors) => vectors,
            Err(msg) => {
                response_warnings.push(msg);
                Vec::new()
            }
        };

        // Bytes before graph: a node pointing at a blob that is not there is a
        // dangling reference, while a blob with no node is inert garbage the
        // next identical upload reuses.
        let sha256 = self.store.put(&bytes)?;

        let document = Document {
            id: Uuid::new_v4(),
            filename: filename.clone(),
            format: extracted.format,
            sha256: sha256.clone(),
            size_bytes,
            page_count: extracted.pages.len(),
            chunk_count: chunks.len(),
            warnings: extracted.warnings.clone(),
            created_at: Utc::now(),
            project_id,
            session_id,
            extracted: !is_opaque,
            mime_type: Some(mime_type.to_string()),
        };

        self.graph.create_document(&document, &chunks).await?;

        if let Some((entity_type, entity_id)) = &entity {
            self.graph
                .link_document_to_entity(document.id, entity_type, entity_id)
                .await?;
        }

        // A document attached to a conversation is also a citizen of the
        // knowledge graph: the `LINKED_TO` edge is what lets it be found the
        // same way a note is, rather than only through `HAS_DOCUMENT`.
        if let Some(sid) = session_id {
            self.graph
                .link_document_to_entity(document.id, &EntityType::ChatSession, &sid.to_string())
                .await?;
        }

        if !embeddings.is_empty() {
            let model = self
                .embeddings
                .map(|p| p.model_name().to_string())
                .unwrap_or_default();
            if let Err(e) = self
                .graph
                .set_document_chunk_embeddings(&embeddings, &model)
                .await
            {
                // The document is already persisted and readable; only semantic
                // search is degraded. Say so rather than failing the upload.
                tracing::warn!(document_id = %document.id, error = %e, "failed to store document chunk embeddings");
                response_warnings.push(
                    "embeddings could not be stored — semantic search will not find this document"
                        .to_string(),
                );
            }
        }

        Ok(UploadedDocument {
            id: document.id,
            filename: document.filename,
            format: document.format,
            size_bytes: document.size_bytes,
            sha256: document.sha256,
            page_count: document.page_count,
            chunk_count: document.chunk_count,
            warnings: response_warnings,
            created_at: document.created_at,
            project_id: document.project_id,
            session_id: document.session_id,
            extracted: document.extracted,
            mime_type: mime_type.to_string(),
        })
    }

    /// Embed every chunk, [`EMBED_BATCH_SIZE`] at a time.
    ///
    /// The error case returns a message rather than propagating: an embedding
    /// provider being down is not a reason to reject a document the user can
    /// still read and download.
    async fn embed_chunks(
        &self,
        chunks: &[DocumentChunk],
    ) -> Result<Vec<(Uuid, Vec<f32>)>, String> {
        let Some(provider) = self.embeddings else {
            return Ok(Vec::new());
        };
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let mut out = Vec::with_capacity(chunks.len());
        for batch in chunks.chunks(EMBED_BATCH_SIZE) {
            let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
            let vectors = provider.embed_batch(&texts).await.map_err(|e| {
                tracing::warn!(error = %e, "document chunk embedding failed");
                "embeddings could not be computed — this document will not be found by semantic \
                 search until it is re-uploaded"
                    .to_string()
            })?;
            if vectors.len() != batch.len() {
                tracing::warn!(
                    expected = batch.len(),
                    got = vectors.len(),
                    "embedding provider returned the wrong number of vectors"
                );
                return Err(
                    "embeddings could not be computed — the provider returned a mismatched batch"
                        .to_string(),
                );
            }
            out.extend(batch.iter().map(|c| c.id).zip(vectors));
        }
        Ok(out)
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// `POST /api/documents` — multipart upload.
///
/// Fields: `file` (required), `project_id`, `session_id`, and optionally
/// `entity_type` + `entity_id` to attach the file to a task, plan, milestone…
/// (`entity_type=project` is the same as passing `project_id`).
pub async fn upload_document(
    State(state): State<OrchestratorState>,
    multipart: Multipart,
) -> Result<impl IntoResponse, DocumentError> {
    let fields = read_upload_fields(multipart).await?;

    let (project_id, entity) = resolve_entity_field(&fields)?;
    let bytes = fields.file.ok_or_else(|| {
        DocumentError::Unprocessable("no `file` field in the request".to_string())
    })?;
    let filename = fields
        .filename
        .unwrap_or_else(|| FALLBACK_FILENAME.to_string());

    let store = DocumentStore::from_config(state.orchestrator.config());
    let ingestor = Ingestor::from_state(&state, &store);
    let uploaded = ingestor
        .ingest_for(bytes, &filename, project_id, fields.session_id, entity)
        .await?;

    Ok((StatusCode::CREATED, Json(uploaded)))
}

/// An entity a document is attached to: its type and id. `None` when the
/// upload is not attached to anything beyond a project or session.
type EntityLink = Option<(EntityType, String)>;

/// Interpret the optional `entity_type` / `entity_id` pair.
///
/// Returns the effective `project_id` (a `project` entity is the project link
/// itself, which is what `list_project_documents` and `HAS_DOCUMENT` read) and
/// the entity to `LINKED_TO`, if any.
fn resolve_entity_field(
    fields: &UploadFields,
) -> Result<(Option<Uuid>, EntityLink), DocumentError> {
    let parsed = parse_entity(fields.entity_type.as_deref(), fields.entity_id.as_deref())?;
    match parsed {
        Some((EntityType::Project, id)) => {
            let pid = Uuid::parse_str(&id).map_err(|_| {
                DocumentError::Unprocessable("`entity_id` is not a valid project UUID".to_string())
            })?;
            Ok((fields.project_id.or(Some(pid)), None))
        }
        other => Ok((fields.project_id, other)),
    }
}

/// Both fields or neither; the type must be a known entity type.
fn parse_entity(
    entity_type: Option<&str>,
    entity_id: Option<&str>,
) -> Result<EntityLink, DocumentError> {
    match (
        entity_type.map(str::trim).filter(|s| !s.is_empty()),
        entity_id.map(str::trim).filter(|s| !s.is_empty()),
    ) {
        (None, None) => Ok(None),
        (Some(t), Some(id)) => {
            let parsed = t.parse::<EntityType>().map_err(|_| {
                DocumentError::Unprocessable(format!("unknown `entity_type` {t:?}"))
            })?;
            Ok(Some((parsed, id.to_string())))
        }
        _ => Err(DocumentError::Unprocessable(
            "`entity_type` and `entity_id` must be given together".to_string(),
        )),
    }
}

/// Drain the multipart body into memory.
///
/// A multipart error is reported with the status axum itself assigned it, which
/// is how a body over the route's `DefaultBodyLimit` surfaces as 413 rather
/// than being flattened into "malformed request".
async fn read_upload_fields(mut multipart: Multipart) -> Result<UploadFields, DocumentError> {
    let mut fields = UploadFields::default();

    loop {
        let next = multipart.next_field().await.map_err(multipart_error)?;
        let Some(field) = next else { break };

        match field.name().unwrap_or_default().to_string().as_str() {
            "file" => {
                // Read the filename before consuming the field: `bytes()` takes
                // ownership and the metadata goes with it.
                fields.filename = field.file_name().map(str::to_string);
                fields.file = Some(field.bytes().await.map_err(multipart_error)?.to_vec());
            }
            "project_id" => {
                fields.project_id = parse_optional_uuid(field, "project_id").await?;
            }
            "session_id" => {
                fields.session_id = parse_optional_uuid(field, "session_id").await?;
            }
            "entity_type" => {
                fields.entity_type = Some(field.text().await.map_err(multipart_error)?);
            }
            "entity_id" => {
                fields.entity_id = Some(field.text().await.map_err(multipart_error)?);
            }
            // Unknown fields are ignored rather than rejected: the frontend may
            // legitimately send more than this version knows about.
            _ => {}
        }
    }

    Ok(fields)
}

async fn parse_optional_uuid(
    field: axum::extract::multipart::Field<'_>,
    name: &str,
) -> Result<Option<Uuid>, DocumentError> {
    let raw = field.text().await.map_err(multipart_error)?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    Uuid::parse_str(trimmed)
        .map(Some)
        .map_err(|_| DocumentError::Unprocessable(format!("`{name}` is not a valid UUID")))
}

fn multipart_error(err: axum::extract::multipart::MultipartError) -> DocumentError {
    if err.status() == StatusCode::PAYLOAD_TOO_LARGE {
        DocumentError::TooLarge(format!(
            "upload exceeds the {MAX_BLOB_BYTES} byte limit: {err}"
        ))
    } else {
        DocumentError::Unprocessable(format!("malformed multipart request: {err}"))
    }
}

/// `GET /api/documents` — list documents for a project or a chat session.
pub async fn list_documents(
    State(state): State<OrchestratorState>,
    Query(params): Query<DocumentsListQuery>,
) -> Result<Json<DocumentListResponse>, DocumentError> {
    let graph = state.orchestrator.neo4j();

    let entity = parse_entity(params.entity_type.as_deref(), params.entity_id.as_deref())?;

    let mut items: Vec<Document> = match (params.project_id, params.session_id, &entity) {
        (Some(project_id), _, _) => graph.list_project_documents(project_id).await?,
        (None, _, Some((entity_type, entity_id))) => {
            graph
                .get_documents_for_entity(entity_type, entity_id)
                .await?
        }
        (None, Some(session_id), None) => {
            graph
                .get_documents_for_entity(&EntityType::ChatSession, &session_id.to_string())
                .await?
        }
        (None, None, None) => {
            return Err(DocumentError::BadRequest(
                "one of `project_id`, `session_id` or `entity_type` + `entity_id` is required"
                    .to_string(),
            ))
        }
    };

    // When both are given the project query is the broad one, so the session
    // narrows it here rather than in a second round trip.
    if let (Some(session_id), Some(_)) = (params.session_id, params.project_id) {
        items.retain(|d| d.session_id == Some(session_id));
    }

    let total = items.len();
    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(50);
    let items = items.into_iter().skip(offset).take(limit).collect();

    Ok(Json(DocumentListResponse { items, total }))
}

/// `GET /api/documents/{document_id}` — metadata only; the text lives in chunks.
pub async fn get_document(
    State(state): State<OrchestratorState>,
    Path(document_id): Path<Uuid>,
) -> Result<Json<Document>, DocumentError> {
    let document = load_document(&state, document_id).await?;
    Ok(Json(document))
}

/// `GET /api/documents/{document_id}/chunks` — the document's chunks, in order.
pub async fn get_document_chunks(
    State(state): State<OrchestratorState>,
    Path(document_id): Path<Uuid>,
) -> Result<Json<ChunkListResponse>, DocumentError> {
    // Resolve the document first so a bad id is a 404 rather than an empty list
    // that reads as "this document has no chunks".
    load_document(&state, document_id).await?;

    let chunks = state
        .orchestrator
        .neo4j()
        .get_document_chunks(document_id)
        .await?;
    let items: Vec<ChunkView> = chunks.iter().map(ChunkView::from).collect();
    let total = items.len();

    Ok(Json(ChunkListResponse { items, total }))
}

/// `GET /api/documents/{document_id}/raw` — the original bytes.
pub async fn get_document_raw(
    State(state): State<OrchestratorState>,
    Path(document_id): Path<Uuid>,
) -> Result<impl IntoResponse, DocumentError> {
    let document = load_document(&state, document_id).await?;

    let store = DocumentStore::from_config(state.orchestrator.config());
    let bytes = store.get(&document.sha256)?;

    // The type recorded at upload; nodes that predate the field fall back to
    // what the format implies.
    let content_type = document
        .mime_type
        .clone()
        .unwrap_or_else(|| content_type_for(document.format).to_string());
    let disposition = format!(
        "attachment; filename=\"{}\"",
        header_safe_filename(&document.filename)
    );
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type.as_str()),
            (header::CONTENT_DISPOSITION, disposition.as_str()),
            // Served as a download of whatever the bytes are; never sniffed
            // into something a browser would execute.
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        ],
        bytes,
    )
        .into_response())
}

/// `DELETE /api/documents/{document_id}` — remove the document and its chunks.
///
/// The blob is deliberately left alone: it is content-addressed, so another
/// document may legitimately point at the same bytes. Reclaiming unreferenced
/// blobs is a sweep over the store, not a side effect of one delete.
pub async fn delete_document(
    State(state): State<OrchestratorState>,
    Path(document_id): Path<Uuid>,
) -> Result<StatusCode, DocumentError> {
    let deleted = state
        .orchestrator
        .neo4j()
        .delete_document(document_id)
        .await?;
    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(DocumentError::NotFound(format!(
            "no document {document_id}"
        )))
    }
}

async fn load_document(
    state: &OrchestratorState,
    document_id: Uuid,
) -> Result<Document, DocumentError> {
    state
        .orchestrator
        .neo4j()
        .get_document(document_id)
        .await?
        .ok_or_else(|| DocumentError::NotFound(format!("no document {document_id}")))
}

/// Shorthand so callers can share one `Arc`-flavoured provider handle.
pub type SharedEmbeddingProvider = Arc<dyn EmbeddingProvider>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MockEmbeddingProvider;
    use crate::neo4j::mock::MockGraphStore;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::TempDir;

    /// Counts `embed_batch` calls so "batched, not one per chunk" is an
    /// assertion rather than a claim.
    struct CountingProvider {
        inner: MockEmbeddingProvider,
        batch_calls: AtomicUsize,
        single_calls: AtomicUsize,
    }

    impl CountingProvider {
        fn new() -> Self {
            Self {
                inner: MockEmbeddingProvider::new(8),
                batch_calls: AtomicUsize::new(0),
                single_calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl EmbeddingProvider for CountingProvider {
        async fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
            self.single_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.embed_text(text).await
        }
        async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
            self.batch_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.embed_batch(texts).await
        }
        fn dimensions(&self) -> usize {
            self.inner.dimensions()
        }
        fn model_name(&self) -> &str {
            self.inner.model_name()
        }
    }

    /// An embedding provider that always fails, to check the upload survives it.
    struct FailingProvider;

    #[async_trait]
    impl EmbeddingProvider for FailingProvider {
        async fn embed_text(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
            anyhow::bail!("provider down")
        }
        async fn embed_batch(&self, _texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
            anyhow::bail!("provider down")
        }
        fn dimensions(&self) -> usize {
            8
        }
        fn model_name(&self) -> &str {
            "failing"
        }
    }

    fn blob_count(root: &std::path::Path) -> usize {
        fn walk(dir: &std::path::Path, acc: &mut usize) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, acc);
                } else {
                    *acc += 1;
                }
            }
        }
        let mut acc = 0;
        walk(root, &mut acc);
        acc
    }

    // ------------------------------------------------------------------
    // Filename hostility
    // ------------------------------------------------------------------

    #[test]
    fn traversal_in_a_filename_does_not_survive() {
        assert_eq!(sanitize_filename("../../etc/passwd"), "passwd");
        assert_eq!(sanitize_filename("..\\..\\windows\\system32"), "system32");
        assert_eq!(sanitize_filename("/absolute/path/report.pdf"), "report.pdf");
    }

    #[test]
    fn control_characters_and_quotes_are_stripped() {
        let name = sanitize_filename("evil\r\nX-Injected: 1\"\\.txt");
        assert!(!name.contains('\r'));
        assert!(!name.contains('\n'));
        assert!(!name.contains('"'));
        assert!(!name.contains('\\'));
    }

    #[test]
    fn a_filename_that_sanitises_to_nothing_gets_a_fallback() {
        assert_eq!(sanitize_filename(""), FALLBACK_FILENAME);
        assert_eq!(sanitize_filename("..."), FALLBACK_FILENAME);
        assert_eq!(sanitize_filename("/"), FALLBACK_FILENAME);
        assert_eq!(sanitize_filename("\u{0}\u{1}"), FALLBACK_FILENAME);
    }

    #[test]
    fn a_long_filename_is_truncated_on_a_char_boundary() {
        // Multi-byte characters: a naive byte slice here would panic.
        let raw = "é".repeat(400);
        let out = sanitize_filename(&raw);
        assert!(out.len() <= MAX_FILENAME_BYTES);
        assert!(!out.is_empty());
        // Still valid UTF-8 made of whole characters.
        assert_eq!(out.chars().count(), out.len() / 2);
    }

    #[test]
    fn header_rendering_never_emits_a_quote_or_a_newline() {
        let out = header_safe_filename("a\"b\\c\r\nd é.txt");
        assert!(!out.contains('"'));
        assert!(!out.contains('\\'));
        assert!(!out.contains('\r'));
        assert!(!out.contains('\n'));
        assert!(out.contains("txt"));
    }

    // ------------------------------------------------------------------
    // Body limit
    // ------------------------------------------------------------------

    #[test]
    fn the_body_limit_clears_the_blob_cap_and_beats_axums_default() {
        let limit = upload_body_limit();
        assert!(
            limit as u64 > MAX_BLOB_BYTES,
            "a file at exactly the blob cap must still fit in the body"
        );
        // axum's silent default, the whole reason this exists.
        assert!(limit > 2 * 1024 * 1024);
    }

    // ------------------------------------------------------------------
    // Ingestion pipeline
    // ------------------------------------------------------------------

    struct Harness {
        graph: Arc<MockGraphStore>,
        store: DocumentStore,
        _dir: TempDir,
    }

    impl Harness {
        fn new() -> Self {
            let dir = TempDir::new().expect("tempdir");
            let store = DocumentStore::new(dir.path());
            Self {
                graph: Arc::new(MockGraphStore::new()),
                store,
                _dir: dir,
            }
        }

        fn ingestor<'a>(&'a self, embeddings: Option<&'a dyn EmbeddingProvider>) -> Ingestor<'a> {
            Ingestor {
                graph: self.graph.as_ref(),
                store: &self.store,
                embeddings,
                chunk_config: ChunkConfig::default(),
            }
        }
    }

    #[tokio::test]
    async fn a_text_upload_is_extracted_chunked_stored_and_persisted() {
        let h = Harness::new();
        let body = "# Notes\n\nFirst sentence. Second sentence."
            .as_bytes()
            .to_vec();
        let project_id = Uuid::new_v4();

        let out = h
            .ingestor(None)
            .ingest(body.clone(), "notes.md", Some(project_id), None)
            .await
            .expect("ingest");

        assert_eq!(out.filename, "notes.md");
        assert_eq!(out.format, DocumentFormat::PlainText);
        assert_eq!(out.size_bytes, body.len() as u64);
        assert!(out.chunk_count > 0, "text must produce at least one chunk");
        assert_eq!(out.project_id, Some(project_id));

        // The blob is readable back under the returned digest.
        assert_eq!(h.store.get(&out.sha256).unwrap(), body);

        // The graph has the document and its chunks.
        let stored = h
            .graph
            .get_document(out.id)
            .await
            .unwrap()
            .expect("document");
        assert_eq!(stored.sha256, out.sha256);
        let chunks = h.graph.get_document_chunks(out.id).await.unwrap();
        assert_eq!(chunks.len(), out.chunk_count);
        // Chunks tile the extracted text in order.
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.ordinal, i);
        }
    }

    #[tokio::test]
    async fn an_oversized_upload_is_refused_with_413_before_anything_is_written() {
        let h = Harness::new();
        let store = DocumentStore::new(h.store.root()).with_max_blob_bytes(16);
        let ingestor = Ingestor {
            graph: h.graph.as_ref(),
            store: &store,
            embeddings: None,
            chunk_config: ChunkConfig::default(),
        };

        let err = ingestor
            .ingest(vec![b'a'; 64], "big.txt", None, None)
            .await
            .expect_err("must be rejected");

        assert_eq!(err.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            blob_count(store.root()),
            0,
            "nothing may be written on refusal"
        );
    }

    #[tokio::test]
    async fn an_unreadable_blob_is_stored_as_an_opaque_attachment() {
        let h = Harness::new();
        // Invalid UTF-8, no known magic: nothing claims it.
        let bytes = vec![0xff, 0xfe, 0x00, 0x01];
        let out = h
            .ingestor(None)
            .ingest(bytes.clone(), "mystery.bin", None, None)
            .await
            .expect("an unreadable file is still a valid attachment");

        assert_eq!(out.format, DocumentFormat::Binary);
        assert!(!out.extracted);
        assert_eq!(out.chunk_count, 0);
        assert_eq!(out.page_count, 0);
        assert_eq!(out.mime_type, "application/octet-stream");
        assert!(
            out.warnings.iter().any(|w| w.contains("not readable")),
            "the uploader must be told it will not be searchable: {:?}",
            out.warnings
        );
        // The bytes are kept, byte for byte.
        assert_eq!(h.store.get(&out.sha256).unwrap(), bytes);
        let stored = h.graph.get_document(out.id).await.unwrap().unwrap();
        assert!(!stored.extracted);
        assert!(h
            .graph
            .get_document_chunks(out.id)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn a_zip_and_a_png_are_attachments_with_the_right_media_type() {
        let h = Harness::new();
        let zip = h
            .ingestor(None)
            .ingest(
                b"PK\x03\x04 not really a docx".to_vec(),
                "deliverables.zip",
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(zip.format, DocumentFormat::Binary);
        assert_eq!(zip.mime_type, "application/zip");

        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&[0, 0, 0, 13, 0xff, 0xfe]);
        let png = h
            .ingestor(None)
            .ingest(png, "screenshot.png", None, None)
            .await
            .unwrap();
        assert_eq!(png.mime_type, "image/png");
        assert!(!png.extracted);
    }

    #[tokio::test]
    async fn an_opaque_attachment_keeps_the_size_limit() {
        let h = Harness::new();
        let store = DocumentStore::new(h.store.root()).with_max_blob_bytes(8);
        let ingestor = Ingestor {
            graph: h.graph.as_ref(),
            store: &store,
            embeddings: None,
            chunk_config: ChunkConfig::default(),
        };
        let err = ingestor
            .ingest(vec![0xff; 64], "big.bin", None, None)
            .await
            .expect_err("over the cap");
        assert_eq!(err.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(blob_count(store.root()), 0);
    }

    #[test]
    fn html_and_svg_are_never_served_as_themselves() {
        // Same-origin active content: the recorded type must not be the
        // uploader's to choose, so unknown markup comes out inert.
        assert_eq!(
            mime_for(DocumentFormat::Binary, "x.html", b"\xff<html>"),
            "application/octet-stream"
        );
        assert_eq!(
            mime_for(DocumentFormat::Binary, "x.svg", b"\xff<svg>"),
            "application/octet-stream"
        );
    }

    #[tokio::test]
    async fn an_attachment_can_be_linked_to_a_task_and_listed_from_it() {
        let h = Harness::new();
        let task_id = Uuid::new_v4().to_string();
        let out = h
            .ingestor(None)
            .ingest_for(
                b"budget notes".to_vec(),
                "budget.txt",
                None,
                None,
                Some((EntityType::Task, task_id.clone())),
            )
            .await
            .unwrap();
        // The mock records the edge whether or not the task node exists.
        let linked = h
            .graph
            .get_documents_for_entity(&EntityType::Task, &task_id)
            .await
            .unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].id, out.id);
    }

    #[test]
    fn entity_fields_must_come_as_a_valid_pair() {
        assert!(parse_entity(None, None).unwrap().is_none());
        assert!(parse_entity(Some(" "), Some("")).unwrap().is_none());
        let (t, id) = parse_entity(Some("plan"), Some("abc")).unwrap().unwrap();
        assert_eq!(t, EntityType::Plan);
        assert_eq!(id, "abc");
        assert_eq!(
            parse_entity(Some("task"), None).unwrap_err().status(),
            StatusCode::UNPROCESSABLE_ENTITY
        );
        assert_eq!(
            parse_entity(Some("nonsense"), Some("x"))
                .unwrap_err()
                .status(),
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    // -- Office formats -----------------------------------------------------

    /// A ZIP holding `word/`-less parts named like a workbook: enough for the
    /// extractor's content probe, and for the feature-off path to see PK magic.
    #[cfg(any(feature = "xlsx", feature = "pptx"))]
    fn zip_with(entries: &[(&str, &str)]) -> Vec<u8> {
        use std::io::Write;
        let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        for (name, body) in entries {
            w.start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            w.write_all(body.as_bytes()).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    #[cfg(feature = "xlsx")]
    #[tokio::test]
    async fn an_xlsx_upload_is_extracted_into_searchable_chunks() {
        let h = Harness::new();
        let bytes = zip_with(&[
            (
                "[Content_Types].xml",
                r#"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>"#,
            ),
            (
                "_rels/.rels",
                r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#,
            ),
            (
                "xl/workbook.xml",
                r#"<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Plan" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Milestone</t></is></c><c r="B1" t="inlineStr"><is><t>Owner</t></is></c></row></sheetData></worksheet>"#,
            ),
        ]);
        let out = h
            .ingestor(None)
            .ingest(bytes, "plan.xlsx", None, None)
            .await
            .unwrap();
        assert_eq!(out.format, DocumentFormat::Xlsx);
        assert!(out.extracted);
        assert_eq!(out.page_count, 1);
        assert!(out.chunk_count > 0);
        assert_eq!(
            out.mime_type,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        );
        let chunks = h.graph.get_document_chunks(out.id).await.unwrap();
        assert!(
            chunks[0].text.contains("Milestone\tOwner"),
            "{}",
            chunks[0].text
        );
    }

    #[cfg(feature = "pptx")]
    #[tokio::test]
    async fn a_pptx_upload_is_extracted_into_searchable_chunks() {
        let h = Harness::new();
        let bytes = zip_with(&[
            (
                "ppt/presentation.xml",
                r#"<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>"#,
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="x/slide" Target="slides/slide1.xml"/></Relationships>"#,
            ),
            (
                "ppt/slides/slide1.xml",
                r#"<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Launch plan</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"#,
            ),
        ]);
        let out = h
            .ingestor(None)
            .ingest(bytes, "deck.pptx", None, None)
            .await
            .unwrap();
        assert_eq!(out.format, DocumentFormat::Pptx);
        assert!(out.extracted);
        let chunks = h.graph.get_document_chunks(out.id).await.unwrap();
        assert!(chunks[0].text.contains("Launch plan"));
    }

    /// Feature off: an Office file nobody here can read is still an attachment,
    /// not a 415 — the same outcome as any other unreadable file.
    #[cfg(not(feature = "xlsx"))]
    #[tokio::test]
    async fn without_the_xlsx_feature_a_workbook_is_an_opaque_attachment() {
        let h = Harness::new();
        let mut bytes = b"PK\x03\x04".to_vec();
        bytes.extend_from_slice(&[0xff, 0xfe, 0x00]);
        let out = h
            .ingestor(None)
            .ingest(bytes, "budget.xlsx", None, None)
            .await
            .unwrap();
        assert_eq!(out.format, DocumentFormat::Binary);
        assert!(!out.extracted);
        assert_eq!(
            out.mime_type,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        );
    }

    #[tokio::test]
    async fn an_empty_upload_is_422() {
        let h = Harness::new();
        let err = h
            .ingestor(None)
            .ingest(Vec::new(), "empty.txt", None, None)
            .await
            .expect_err("must be rejected");
        assert_eq!(err.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    /// The distinction the task exists to protect: a PDF in a build without the
    /// `pdf` feature must not be reported as an invalid file.
    #[test]
    fn a_missing_feature_is_501_and_says_so() {
        let err: DocumentError = ExtractError::FeatureDisabled {
            format: "PDF",
            feature: "pdf",
        }
        .into();
        assert_eq!(err.status(), StatusCode::NOT_IMPLEMENTED);
        let msg = err.to_string();
        assert!(
            msg.contains("pdf"),
            "the message must name the feature: {msg}"
        );
        assert!(
            !msg.to_lowercase().contains("malformed") && !msg.to_lowercase().contains("invalid"),
            "must not claim the file is broken: {msg}"
        );

        // …and it is a different answer from "nothing reads this".
        let unsupported: DocumentError = ExtractError::UnsupportedFormat {
            sniffed: "\\xff (.bin)".to_string(),
        }
        .into();
        assert_eq!(
            unsupported.status(),
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "415 and 501 must not collapse into one another"
        );
    }

    #[test]
    fn every_failure_mode_maps_to_its_own_status() {
        // Neutralisation check: if any arm were folded into 500 this fails.
        let cases = [
            (
                DocumentError::TooLarge("x".into()),
                StatusCode::PAYLOAD_TOO_LARGE,
            ),
            (
                DocumentError::UnsupportedFormat("x".into()),
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
            ),
            (
                DocumentError::FeatureDisabled("x".into()),
                StatusCode::NOT_IMPLEMENTED,
            ),
            (
                DocumentError::Unprocessable("x".into()),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                DocumentError::BadRequest("x".into()),
                StatusCode::BAD_REQUEST,
            ),
            (DocumentError::NotFound("x".into()), StatusCode::NOT_FOUND),
            (
                DocumentError::Internal(anyhow::anyhow!("x")),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ];
        let mut seen = std::collections::HashSet::new();
        for (err, expected) in cases {
            assert_eq!(err.status(), expected);
            assert!(seen.insert(expected), "status {expected} used twice");
        }
    }

    #[tokio::test]
    async fn internal_error_body_is_generic_but_client_errors_keep_their_message() {
        let read = |resp: axum::response::Response| async move {
            let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
            String::from_utf8(bytes.to_vec()).unwrap()
        };
        let resp = DocumentError::Internal(anyhow::anyhow!("bolt://secret-host:7687 /var/blobs/x"))
            .into_response();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = read(resp).await;
        assert!(!body.contains("secret-host") && !body.contains("/var/blobs"));
        assert!(body.contains(crate::api::handlers::INTERNAL_ERROR_MESSAGE));

        let body = read(DocumentError::NotFound("no such document".into()).into_response()).await;
        assert!(body.contains("no such document"));
    }

    #[tokio::test]
    async fn the_same_bytes_twice_give_two_documents_and_one_blob() {
        let h = Harness::new();
        let body = "the very same content, byte for byte.".as_bytes().to_vec();

        let first = h
            .ingestor(None)
            .ingest(body.clone(), "a.txt", None, None)
            .await
            .unwrap();
        let second = h
            .ingestor(None)
            .ingest(body.clone(), "b.txt", None, None)
            .await
            .unwrap();

        assert_ne!(first.id, second.id, "two attachments, two documents");
        assert_eq!(first.sha256, second.sha256, "one content, one hash");
        assert_eq!(blob_count(h.store.root()), 1, "the bytes are stored once");

        assert!(h.graph.get_document(first.id).await.unwrap().is_some());
        assert!(h.graph.get_document(second.id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn embeddings_are_computed_in_batches_not_one_call_per_chunk() {
        let h = Harness::new();
        let provider = CountingProvider::new();

        // Enough distinct sentences to spill past one batch. Each sentence is
        // well under the chunk budget, so chunks pack several together; the
        // point is only that there are many more chunks than batch calls.
        let mut text = String::new();
        for i in 0..4000 {
            text.push_str(&format!("Sentence number {i} about an entirely unremarkable subject that nonetheless occupies some space. "));
        }

        let out = h
            .ingestor(Some(&provider))
            .ingest(text.into_bytes(), "long.txt", None, None)
            .await
            .unwrap();

        let chunk_count = out.chunk_count;
        assert!(
            chunk_count > EMBED_BATCH_SIZE,
            "need more than one batch to make the point (got {chunk_count})"
        );

        let batch_calls = provider.batch_calls.load(Ordering::SeqCst);
        let single_calls = provider.single_calls.load(Ordering::SeqCst);
        assert_eq!(single_calls, 0, "embed_text must never be used here");
        assert_eq!(
            batch_calls,
            chunk_count.div_ceil(EMBED_BATCH_SIZE),
            "one call per batch, not one per chunk"
        );
        assert!(
            batch_calls < chunk_count,
            "{batch_calls} calls for {chunk_count} chunks"
        );

        // The vectors actually landed on the chunks.
        let chunks = h.graph.get_document_chunks(out.id).await.unwrap();
        assert!(chunks.iter().all(|c| c.embedding.is_some()));
    }

    #[tokio::test]
    async fn a_small_document_takes_exactly_one_batch_call() {
        let h = Harness::new();
        let provider = CountingProvider::new();

        let out = h
            .ingestor(Some(&provider))
            .ingest(b"One short sentence.".to_vec(), "s.txt", None, None)
            .await
            .unwrap();

        assert_eq!(out.chunk_count, 1);
        assert_eq!(provider.batch_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_failing_embedding_provider_warns_but_does_not_fail_the_upload() {
        let h = Harness::new();
        let out = h
            .ingestor(Some(&FailingProvider))
            .ingest(b"Some readable prose.".to_vec(), "s.txt", None, None)
            .await
            .expect("the document is still ingested");

        assert!(
            out.warnings.iter().any(|w| w.contains("embeddings")),
            "the uploader must be told search will not find it: {:?}",
            out.warnings
        );
        assert!(h.graph.get_document(out.id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn extraction_warnings_reach_the_response_and_the_node() {
        // Built directly rather than through a format that emits warnings,
        // because which formats warn is the extractors' business, not this
        // module's. What is asserted here is that a warning is not swallowed.
        let h = Harness::new();
        let doc = Document {
            id: Uuid::new_v4(),
            filename: "report.pdf".to_string(),
            format: DocumentFormat::Pdf,
            sha256: "0".repeat(64),
            size_bytes: 10,
            page_count: 40,
            chunk_count: 0,
            warnings: vec!["3 of 40 pages have no text layer".to_string()],
            created_at: Utc::now(),
            project_id: None,
            session_id: None,
            extracted: true,
            mime_type: None,
        };
        h.graph.create_document(&doc, &[]).await.unwrap();

        let stored = h.graph.get_document(doc.id).await.unwrap().unwrap();
        assert_eq!(stored.warnings, doc.warnings);
    }

    #[tokio::test]
    async fn a_document_with_no_extractable_text_says_so() {
        let h = Harness::new();
        // Whitespace only: valid UTF-8, extracts fine, chunks to nothing.
        let out = h
            .ingestor(None)
            .ingest(b"   \n\t  \n".to_vec(), "blank.txt", None, None)
            .await
            .unwrap();

        assert_eq!(out.chunk_count, 0);
        assert!(
            out.warnings.iter().any(|w| w.contains("no text")),
            "silence here would look like a successful ingestion: {:?}",
            out.warnings
        );
    }

    #[tokio::test]
    async fn a_session_upload_is_linked_into_the_knowledge_graph() {
        let h = Harness::new();
        let session_id = Uuid::new_v4();

        let out = h
            .ingestor(None)
            .ingest(
                b"Attached in a conversation.".to_vec(),
                "c.txt",
                None,
                Some(session_id),
            )
            .await
            .unwrap();

        let linked = h
            .graph
            .get_documents_for_entity(&EntityType::ChatSession, &session_id.to_string())
            .await
            .unwrap();
        assert!(
            linked.iter().any(|d| d.id == out.id),
            "the document must be reachable from the session it was attached to"
        );
    }

    #[tokio::test]
    async fn a_hostile_filename_is_sanitised_before_it_reaches_the_graph() {
        let h = Harness::new();
        let out = h
            .ingestor(None)
            .ingest(
                b"content".to_vec(),
                "../../../etc/pas\r\nswd.txt",
                None,
                None,
            )
            .await
            .unwrap();

        assert!(!out.filename.contains(".."));
        assert!(!out.filename.contains('/'));
        assert!(!out.filename.contains('\r'));

        let stored = h.graph.get_document(out.id).await.unwrap().unwrap();
        assert_eq!(stored.filename, out.filename);

        // And the blob sits under its hash, nowhere near the supplied name.
        let path = h.store.path_for(&out.sha256).unwrap();
        assert!(path.starts_with(h.store.root()));
        assert!(path.to_string_lossy().contains(&out.sha256));
    }

    // ------------------------------------------------------------------
    // Read + delete over the mock graph
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn documents_list_by_project_and_delete_takes_the_chunks_with_it() {
        let h = Harness::new();
        let project_id = Uuid::new_v4();

        let a = h
            .ingestor(None)
            .ingest(
                b"First document text.".to_vec(),
                "a.txt",
                Some(project_id),
                None,
            )
            .await
            .unwrap();
        let b = h
            .ingestor(None)
            .ingest(
                b"Second document text.".to_vec(),
                "b.txt",
                Some(project_id),
                None,
            )
            .await
            .unwrap();
        // A document in another project must not show up.
        let other = h
            .ingestor(None)
            .ingest(
                b"Elsewhere entirely.".to_vec(),
                "c.txt",
                Some(Uuid::new_v4()),
                None,
            )
            .await
            .unwrap();

        let listed = h.graph.list_project_documents(project_id).await.unwrap();
        let ids: Vec<Uuid> = listed.iter().map(|d| d.id).collect();
        assert!(ids.contains(&a.id) && ids.contains(&b.id));
        assert!(
            !ids.contains(&other.id),
            "listing must be scoped to the project"
        );

        assert!(!h.graph.get_document_chunks(a.id).await.unwrap().is_empty());
        assert!(h.graph.delete_document(a.id).await.unwrap());
        assert!(h.graph.get_document(a.id).await.unwrap().is_none());
        assert!(
            h.graph.get_document_chunks(a.id).await.unwrap().is_empty(),
            "chunks must not outlive their document"
        );
        // Deleting twice is not a success.
        assert!(!h.graph.delete_document(a.id).await.unwrap());

        // The blob survives: it is content, and another document may share it.
        assert!(h.store.exists(&a.sha256).unwrap());
    }

    #[tokio::test]
    async fn chunk_views_carry_offsets_and_drop_embeddings() {
        let h = Harness::new();
        let provider = MockEmbeddingProvider::new(8);
        let out = h
            .ingestor(Some(&provider))
            .ingest(b"A sentence worth embedding.".to_vec(), "s.txt", None, None)
            .await
            .unwrap();

        let chunks = h.graph.get_document_chunks(out.id).await.unwrap();
        assert!(
            chunks[0].embedding.is_some(),
            "the chunk itself has a vector"
        );

        let view = ChunkView::from(&chunks[0]);
        assert_eq!(view.start, chunks[0].start_byte);
        assert_eq!(view.end, chunks[0].end_byte);

        let json = serde_json::to_value(&view).unwrap();
        assert!(
            json.get("embedding").is_none(),
            "the API must not ship vectors to the browser"
        );
        assert!(json.get("start").is_some() && json.get("end").is_some());
    }

    #[test]
    fn content_types_are_distinct_per_format() {
        assert!(content_type_for(DocumentFormat::PlainText).starts_with("text/plain"));
        assert_eq!(content_type_for(DocumentFormat::Pdf), "application/pdf");
        assert!(content_type_for(DocumentFormat::Docx).contains("wordprocessingml"));
    }

    // ====================================================================
    // Route-level tests (real router, mock backends)
    //
    // The pipeline tests above cover what ingestion does. These cover what
    // only the router can get wrong: whether the six routes are mounted at
    // the paths the frontend was written against, whether multipart is
    // parsed, and whether the body limit is actually installed.
    // ====================================================================

    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::orchestrator::watcher::FileWatcher;
    use crate::orchestrator::Orchestrator;
    use crate::test_helpers::{mock_app_state, test_auth_config, test_bearer_token};
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    const BOUNDARY: &str = "----POboundaryZZ";

    /// Build a `ServerState` whose blob store lives in a temp directory.
    ///
    /// The directory is returned alongside it: dropping the `TempDir` deletes
    /// the blobs, so every test must keep it alive for its duration.
    async fn server_state() -> (OrchestratorState, TempDir) {
        let dir = TempDir::new().expect("tempdir");
        let mut app_state = mock_app_state();
        let mut config = (*app_state.config).clone();
        config.documents_storage_dir = Some(dir.path().to_string_lossy().into_owned());
        // Without this the orchestrator defaults to the *local fastembed*
        // provider and every route test downloads and runs a real ONNX model:
        // ~100 ms per chunk, which turns a 3 MiB upload into a minute and then
        // into a request timeout. What these tests check is routing, not
        // vectors; the batching behaviour is asserted on a counting provider
        // in the pipeline tests above.
        config.embedding_provider = Some("disabled".to_string());
        app_state.config = Arc::new(config);

        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            remote_mcp: crate::RemoteMcpConfig::default(),
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
            model_catalog: crate::chat::model_catalog::ModelCatalogCache::new(None),
            vault: crate::vault::VaultService::ephemeral(),
        });
        (state, dir)
    }

    /// A multipart body with a `file` part plus any extra text parts.
    fn multipart_body(filename: &str, content: &[u8], extra: &[(&str, String)]) -> Vec<u8> {
        let mut body = Vec::new();
        for (name, value) in extra {
            body.extend_from_slice(
                format!(
                    "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                )
                .as_bytes(),
            );
        }
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(content);
        body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
        body
    }

    fn upload_request(filename: &str, content: &[u8], extra: &[(&str, String)]) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/api/documents")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .header("authorization", test_bearer_token())
            .body(Body::from(multipart_body(filename, content, extra)))
            .unwrap()
    }

    fn authed(method: &str, uri: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    async fn json_of(resp: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn post_documents_ingests_a_text_file_and_returns_the_contract_shape() {
        let (state, _dir) = server_state().await;
        let project_id = Uuid::new_v4();
        let app = create_router(state);

        let resp = app
            .oneshot(upload_request(
                "notes.md",
                b"# Title\n\nFirst sentence. Second sentence.",
                &[("project_id", project_id.to_string())],
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let json = json_of(resp).await;
        // Every field the frozen contract promises.
        for field in [
            "id",
            "filename",
            "format",
            "size_bytes",
            "sha256",
            "page_count",
            "chunk_count",
            "warnings",
        ] {
            assert!(json.get(field).is_some(), "missing `{field}` in {json}");
        }
        assert_eq!(json["filename"], "notes.md");
        assert_eq!(json["format"], "plain_text");
        assert_eq!(json["sha256"].as_str().unwrap().len(), 64);
        assert!(json["chunk_count"].as_u64().unwrap() >= 1);
        assert!(json["warnings"].is_array());
    }

    /// The regression this task exists to prevent.
    ///
    /// 3 MiB is nothing for a PDF and well past axum's silent 2 MiB default.
    /// Remove the `DefaultBodyLimit` layer in `routes.rs` and this returns 413.
    #[tokio::test]
    async fn an_upload_larger_than_axums_default_body_limit_is_accepted() {
        let (state, _dir) = server_state().await;
        let app = create_router(state);

        let mut content = String::with_capacity(3 * 1024 * 1024 + 64);
        while content.len() < 3 * 1024 * 1024 {
            content.push_str("Prose that occupies space without saying much at all. ");
        }

        let resp = app
            .oneshot(upload_request("big.txt", content.as_bytes(), &[]))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::CREATED,
            "a 3 MiB upload must not trip axum's 2 MiB default"
        );
    }

    /// The layer itself: with a small limit installed, an over-size body is a
    /// 413 from the transport rather than an opaque failure.
    #[tokio::test]
    async fn the_body_limit_layer_turns_an_oversize_body_into_413() {
        use axum::routing::post;

        let (state, _dir) = server_state().await;
        let app = axum::Router::new()
            .route(
                "/api/documents",
                post(upload_document).layer(axum::extract::DefaultBodyLimit::max(512)),
            )
            .with_state(state);

        let resp = app
            .oneshot(upload_request("big.txt", &vec![b'a'; 4096], &[]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn an_unreadable_upload_is_stored_and_downloadable_over_http() {
        let (state, _dir) = server_state().await;
        let app = create_router(state);
        let bytes = [0xff, 0xfe, 0x00, 0x01, 0x02];

        let resp = app
            .clone()
            .oneshot(upload_request("mystery.bin", &bytes, &[]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = json_of(resp).await;
        assert_eq!(json["extracted"], false);
        assert_eq!(json["format"], "binary");
        assert_eq!(json["chunk_count"], 0);
        assert_eq!(json["mime_type"], "application/octet-stream");
        let id = json["id"].as_str().unwrap().to_string();

        let resp = app
            .oneshot(authed("GET", &format!("/api/documents/{id}/raw")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "application/octet-stream"
        );
        assert_eq!(
            resp.headers().get("x-content-type-options").unwrap(),
            "nosniff"
        );
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), &bytes);
    }

    #[tokio::test]
    async fn a_file_uploaded_for_a_task_is_listed_by_that_task() {
        let (state, _dir) = server_state().await;
        let app = create_router(state);
        let task_id = Uuid::new_v4().to_string();

        let resp = app
            .clone()
            .oneshot(upload_request(
                "notes.md",
                b"# Notes\n\nSomething to remember.",
                &[
                    ("entity_type", "task".to_string()),
                    ("entity_id", task_id.clone()),
                ],
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let resp = app
            .clone()
            .oneshot(authed(
                "GET",
                &format!("/api/documents?entity_type=task&entity_id={task_id}"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = json_of(resp).await;
        assert_eq!(json["total"], 1, "{json}");

        // Half a pair is the caller's mistake.
        let resp = app
            .oneshot(upload_request(
                "notes.md",
                b"x y z",
                &[("entity_type", "task".to_string())],
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn a_request_with_no_file_part_is_422() {
        let (state, _dir) = server_state().await;
        let app = create_router(state);

        let body = format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"project_id\"\r\n\r\n{}\r\n--{BOUNDARY}--\r\n",
            Uuid::new_v4()
        );
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/documents")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={BOUNDARY}"),
                    )
                    .header("authorization", test_bearer_token())
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn a_bad_uuid_field_is_422_not_500() {
        let (state, _dir) = server_state().await;
        let app = create_router(state);

        let resp = app
            .oneshot(upload_request(
                "notes.txt",
                b"Some prose.",
                &[("project_id", "not-a-uuid".to_string())],
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn list_get_chunks_raw_and_delete_round_trip() {
        let (state, _dir) = server_state().await;
        let project_id = Uuid::new_v4();
        let app = create_router(state);

        let resp = app
            .clone()
            .oneshot(upload_request(
                "notes.txt",
                b"First sentence here. Second sentence here.",
                &[("project_id", project_id.to_string())],
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let uploaded = json_of(resp).await;
        let id = uploaded["id"].as_str().unwrap().to_string();

        // GET /api/documents?project_id=
        let resp = app
            .clone()
            .oneshot(authed(
                "GET",
                &format!("/api/documents?project_id={project_id}"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = json_of(resp).await;
        assert_eq!(json["total"], 1);
        assert_eq!(json["items"][0]["id"], id.as_str());

        // Listing without a scope is a 400, not an unbounded dump.
        let resp = app
            .clone()
            .oneshot(authed("GET", "/api/documents"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // GET /api/documents/{id}
        let resp = app
            .clone()
            .oneshot(authed("GET", &format!("/api/documents/{id}")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = json_of(resp).await;
        assert_eq!(json["filename"], "notes.txt");

        // GET /api/documents/{id}/chunks
        let resp = app
            .clone()
            .oneshot(authed("GET", &format!("/api/documents/{id}/chunks")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = json_of(resp).await;
        let items = json["items"].as_array().unwrap();
        assert!(!items.is_empty());
        assert!(items[0].get("start").is_some());
        assert!(items[0].get("end").is_some());
        assert!(items[0].get("text").is_some());
        assert!(
            items[0].get("embedding").is_none(),
            "vectors must not be served to the browser"
        );

        // GET /api/documents/{id}/raw
        let resp = app
            .clone()
            .oneshot(authed("GET", &format!("/api/documents/{id}/raw")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let disposition = resp
            .headers()
            .get(header::CONTENT_DISPOSITION)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(disposition.contains("notes.txt"));
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&bytes[..], b"First sentence here. Second sentence here.");

        // DELETE /api/documents/{id}
        let resp = app
            .clone()
            .oneshot(authed("DELETE", &format!("/api/documents/{id}")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        // …and it is gone.
        let resp = app
            .clone()
            .oneshot(authed("GET", &format!("/api/documents/{id}")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let resp = app
            .oneshot(authed("DELETE", &format!("/api/documents/{id}")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn unknown_ids_are_404_on_every_read_route() {
        let (state, _dir) = server_state().await;
        let app = create_router(state);
        let id = Uuid::new_v4();

        for path in [
            format!("/api/documents/{id}"),
            format!("/api/documents/{id}/chunks"),
            format!("/api/documents/{id}/raw"),
        ] {
            let resp = app.clone().oneshot(authed("GET", &path)).await.unwrap();
            assert_eq!(resp.status(), StatusCode::NOT_FOUND, "for {path}");
        }
    }

    #[tokio::test]
    async fn a_session_upload_is_listed_by_session() {
        let (state, _dir) = server_state().await;
        let session_id = Uuid::new_v4();
        let app = create_router(state);

        let resp = app
            .clone()
            .oneshot(upload_request(
                "pasted.txt",
                b"Pasted into a conversation.",
                &[("session_id", session_id.to_string())],
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let id = json_of(resp).await["id"].as_str().unwrap().to_string();

        let resp = app
            .oneshot(authed(
                "GET",
                &format!("/api/documents?session_id={session_id}"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = json_of(resp).await;
        assert_eq!(json["items"][0]["id"], id.as_str());
    }

    #[tokio::test]
    async fn the_same_file_uploaded_twice_over_http_yields_two_documents() {
        let (state, _dir) = server_state().await;
        let project_id = Uuid::new_v4();
        let app = create_router(state);

        let mut ids = Vec::new();
        let mut hashes = Vec::new();
        for _ in 0..2 {
            let resp = app
                .clone()
                .oneshot(upload_request(
                    "same.txt",
                    b"Identical bytes, twice.",
                    &[("project_id", project_id.to_string())],
                ))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::CREATED);
            let json = json_of(resp).await;
            ids.push(json["id"].as_str().unwrap().to_string());
            hashes.push(json["sha256"].as_str().unwrap().to_string());
        }

        assert_ne!(ids[0], ids[1]);
        assert_eq!(hashes[0], hashes[1]);

        let resp = app
            .oneshot(authed(
                "GET",
                &format!("/api/documents?project_id={project_id}"),
            ))
            .await
            .unwrap();
        let json = json_of(resp).await;
        assert_eq!(json["total"], 2, "two attachments of the same content");
    }
}
