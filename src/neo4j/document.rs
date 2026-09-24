//! Neo4j Document operations
//!
//! A `Document` is a citizen of the knowledge graph, not a side table. It gets
//! the same `LINKED_TO` relation that `Note` uses (see [`super::note`]), which
//! is what lets an uploaded PDF sit *between* a decision and the file it
//! justifies instead of next to them:
//!
//! ```text
//!   (:Project)-[:HAS_DOCUMENT]->(:Document)-[:HAS_CHUNK]->(:DocumentChunk)
//!                                    |
//!                                [:LINKED_TO]          (either direction)
//!                                    |
//!                    (:Note) (:Decision) (:File) (:Task) …
//! ```
//!
//! `LINKED_TO` is deliberately the *same* relation type and the same
//! `last_verified` property that `note.rs` writes, so every existing traversal
//! over the knowledge layer — propagation, context assembly, graph
//! visualisation — sees documents without being taught about them.
//!
//! ## Chunks are written in bulk
//!
//! A 500-chunk document is two queries, not 500: the chunk nodes and their
//! `HAS_CHUNK` relations go through [`super::batch::run_unwind_in_chunks_with`],
//! and the embeddings through a second `UNWIND` over only the chunks that carry
//! one. Chunk order is carried by `DocumentChunk.ordinal`, which is what every
//! read orders by — the relation itself stays plain.
//!
//! ## Chunks are searched the way notes are
//!
//! Embeddings are written with `db.create.setNodeVectorProperty` and queried
//! with `db.index.vector.queryNodes`, exactly as `set_note_embedding` /
//! `vector_search_notes` do. The HNSW index is `document_chunk_embeddings`,
//! declared next to the others in `init_schema` (see `client.rs`) — this project
//! keeps its schema there, `migrations/` holds one-off data conversions only.

use super::batch::{run_unwind_in_chunks_with, BoltMap};
use super::client::Neo4jClient;
use crate::documents::{ByteInterval, DocumentFormat, TextChunk};
use crate::notes::EntityType;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use neo4rs::query;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Models
// ============================================================================

/// An ingested document, as stored on a `(:Document)` node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub id: Uuid,
    /// Filename as supplied by the uploader. Display only — it is not identity.
    pub filename: String,
    pub format: DocumentFormat,
    /// Hex-encoded SHA-256 of the **original bytes**, before extraction.
    ///
    /// This is the identity of the content: re-uploading the same file finds the
    /// existing document instead of chunking and embedding it a second time.
    /// It is indexed, not unique — the same bytes may legitimately be attached
    /// to two projects, or to a project and a conversation.
    pub sha256: String,
    pub size_bytes: u64,
    /// Number of pages the extractor reported. `0` for unpaginated formats —
    /// that is information, not a gap (see `documents::extract`).
    pub page_count: usize,
    pub chunk_count: usize,
    /// Non-fatal extraction problems, carried over from
    /// [`crate::documents::ExtractedText::warnings`] so a chunk that reads oddly
    /// can be explained rather than merely doubted.
    pub warnings: Vec<String>,
    pub created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<Uuid>,
}

/// One chunk of a document, as stored on a `(:DocumentChunk)` node.
///
/// The parent is the `HAS_CHUNK` relation, not a foreign key: a chunk is never
/// meaningful without the document it tiles, and traversing one hop is cheaper
/// than keeping a denormalised id honest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: Uuid,
    pub text: String,
    /// Byte offsets into the *extracted text*, half-open `[start, end)`.
    pub start_byte: usize,
    pub end_byte: usize,
    /// 1-based page number, when the format is paginated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<usize>,
    /// Position in the document, from 0. Every read orders by this.
    pub ordinal: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

impl DocumentChunk {
    /// The span this chunk covers in the extracted text.
    pub fn interval(&self) -> ByteInterval {
        ByteInterval::new(self.start_byte, self.end_byte)
    }

    /// Build a persistable chunk from the chunker's output.
    ///
    /// `page` comes from [`crate::documents::ExtractedText::page_of`] applied to
    /// the chunk's start offset; passing `None` is correct for unpaginated
    /// formats.
    pub fn from_text_chunk(chunk: &TextChunk, ordinal: usize, page: Option<usize>) -> Self {
        Self {
            id: Uuid::new_v4(),
            text: chunk.text.clone(),
            start_byte: chunk.interval.start,
            end_byte: chunk.interval.end,
            page,
            ordinal,
            embedding: None,
        }
    }
}

/// A chunk returned by vector search, with the document it came from.
///
/// The filename travels with the hit so a citation can be rendered without a
/// second round trip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentChunkHit {
    pub document_id: Uuid,
    pub filename: String,
    pub chunk: DocumentChunk,
    pub score: f64,
}

/// Parse the stored `format` property back into a [`DocumentFormat`].
///
/// Unknown values fall back to plain text rather than failing the read: a
/// document written by a newer build must still be listable by an older one.
fn format_from_str(s: &str) -> DocumentFormat {
    match s {
        "docx" => DocumentFormat::Docx,
        "pdf" => DocumentFormat::Pdf,
        _ => DocumentFormat::PlainText,
    }
}

/// Node label, match field, and whether to match by suffix, for a knowledge
/// entity — the same mapping `note.rs` applies when it writes `LINKED_TO`,
/// taken from [`EntityType`] itself rather than restated here.
fn entity_match(entity_type: &EntityType, entity_id: &str) -> (&'static str, &'static str, bool) {
    // Same convention as note.rs: a relative File path is matched as a suffix
    // of the absolute path Neo4j stores.
    let suffix_match = matches!(entity_type, EntityType::File) && !entity_id.starts_with('/');
    (
        entity_type.neo4j_label(),
        entity_type.match_property(),
        suffix_match,
    )
}

/// Entity id as it must be passed to the query, given the matching rule above.
fn entity_match_value(entity_type: &EntityType, entity_id: &str) -> String {
    match entity_type {
        EntityType::File if !entity_id.starts_with('/') => format!("/{}", entity_id),
        _ => entity_id.to_string(),
    }
}

/// Build the `UNWIND` items for a batch of chunks.
///
/// Split out from the query so the shape of what gets written is testable
/// without a live Neo4j.
pub(crate) fn document_chunk_items(chunks: &[DocumentChunk]) -> Vec<BoltMap> {
    chunks
        .iter()
        .map(|c| {
            let mut m = BoltMap::new();
            m.insert("id".into(), c.id.to_string().into());
            m.insert("text".into(), c.text.clone().into());
            m.insert("start_byte".into(), (c.start_byte as i64).into());
            m.insert("end_byte".into(), (c.end_byte as i64).into());
            m.insert("page".into(), c.page.map(|p| p as i64).into());
            m.insert("ordinal".into(), (c.ordinal as i64).into());
            m
        })
        .collect()
}

/// Build the `UNWIND` items for the embeddings of a batch of chunks.
///
/// Only chunks that actually carry a vector are included — calling
/// `db.create.setNodeVectorProperty` with null is an error, and a document
/// whose chunks are not embedded yet is a normal state.
fn chunk_embedding_items(embeddings: &[(Uuid, Vec<f32>)]) -> Vec<BoltMap> {
    embeddings
        .iter()
        .map(|(id, vector)| {
            let mut m = BoltMap::new();
            m.insert("id".into(), id.to_string().into());
            let as_f64: Vec<f64> = vector.iter().map(|&v| v as f64).collect();
            m.insert("embedding".into(), as_f64.into());
            m
        })
        .collect()
}

impl Neo4jClient {
    // ========================================================================
    // Document operations
    // ========================================================================

    /// Create a document and all of its chunks.
    ///
    /// Idempotent on `Document.id` and `DocumentChunk.id` (MERGE), so a retried
    /// ingestion converges instead of duplicating. Chunks that carry an
    /// embedding get it written in the same call.
    pub async fn create_document(
        &self,
        document: &Document,
        chunks: &[DocumentChunk],
    ) -> Result<()> {
        let q = query(
            r#"
            MERGE (d:Document {id: $id})
            SET d.filename = $filename,
                d.format = $format,
                d.sha256 = $sha256,
                d.size_bytes = $size_bytes,
                d.page_count = $page_count,
                d.chunk_count = $chunk_count,
                d.warnings = $warnings,
                d.created_at = datetime($created_at),
                d.project_id = $project_id,
                d.session_id = $session_id
            "#,
        )
        .param("id", document.id.to_string())
        .param("filename", document.filename.clone())
        .param("format", document.format.as_str())
        .param("sha256", document.sha256.clone())
        .param("size_bytes", document.size_bytes as i64)
        .param("page_count", document.page_count as i64)
        .param("chunk_count", document.chunk_count as i64)
        .param("warnings", document.warnings.clone())
        .param("created_at", document.created_at.to_rfc3339())
        .param("project_id", document.project_id.map(|id| id.to_string()))
        .param("session_id", document.session_id.map(|id| id.to_string()));

        self.graph
            .run(q)
            .await
            .context(format!("Failed to create document {}", document.id))?;

        // (:Project)-[:HAS_DOCUMENT]->(:Document)
        if let Some(pid) = document.project_id {
            let link_q = query(
                r#"
                MATCH (d:Document {id: $document_id})
                MATCH (p:Project {id: $project_id})
                MERGE (p)-[:HAS_DOCUMENT]->(d)
                "#,
            )
            .param("document_id", document.id.to_string())
            .param("project_id", pid.to_string());

            self.graph.run(link_q).await?;
        }

        // (:ChatSession)-[:HAS_DOCUMENT]->(:Document) — an attachment dropped in
        // a conversation belongs to that conversation as well as to the project.
        if let Some(sid) = document.session_id {
            let link_q = query(
                r#"
                MATCH (d:Document {id: $document_id})
                MATCH (s:ChatSession {id: $session_id})
                MERGE (s)-[:HAS_DOCUMENT]->(d)
                "#,
            )
            .param("document_id", document.id.to_string())
            .param("session_id", sid.to_string());

            self.graph.run(link_q).await?;
        }

        self.upsert_document_chunks(document.id, chunks).await?;

        Ok(())
    }

    /// Write (or rewrite) the chunks of a document.
    ///
    /// One `UNWIND` per [`super::batch::BATCH_SIZE`] chunks — a 500-chunk
    /// document is a single query. Returns the number of chunks written.
    pub async fn upsert_document_chunks(
        &self,
        document_id: Uuid,
        chunks: &[DocumentChunk],
    ) -> Result<usize> {
        if chunks.is_empty() {
            return Ok(0);
        }

        run_unwind_in_chunks_with(
            &self.graph,
            document_chunk_items(chunks),
            r#"
            MATCH (d:Document {id: $document_id})
            UNWIND $items AS chunk
            MERGE (c:DocumentChunk {id: chunk.id})
            SET c.text = chunk.text,
                c.start_byte = chunk.start_byte,
                c.end_byte = chunk.end_byte,
                c.page = chunk.page,
                c.ordinal = chunk.ordinal
            MERGE (d)-[:HAS_CHUNK]->(c)
            "#,
            |q| q.param("document_id", document_id.to_string()),
        )
        .await
        .context(format!(
            "Failed to write chunks of document {}",
            document_id
        ))?;

        // Chunks that already carry a vector get it in the same ingestion.
        let with_embeddings: Vec<(Uuid, Vec<f32>)> = chunks
            .iter()
            .filter_map(|c| c.embedding.as_ref().map(|e| (c.id, e.clone())))
            .collect();

        if !with_embeddings.is_empty() {
            self.write_chunk_embeddings(&with_embeddings, None).await?;
        }

        Ok(chunks.len())
    }

    /// Store embeddings on chunks, in bulk.
    ///
    /// Uses `db.create.setNodeVectorProperty` so the vectors are the type the
    /// HNSW index requires — the same call `set_note_embedding` makes.
    pub async fn set_document_chunk_embeddings(
        &self,
        embeddings: &[(Uuid, Vec<f32>)],
        model: &str,
    ) -> Result<usize> {
        self.write_chunk_embeddings(embeddings, Some(model)).await
    }

    async fn write_chunk_embeddings(
        &self,
        embeddings: &[(Uuid, Vec<f32>)],
        model: Option<&str>,
    ) -> Result<usize> {
        if embeddings.is_empty() {
            return Ok(0);
        }

        let model_param: Option<String> = model.map(|m| m.to_string());

        run_unwind_in_chunks_with(
            &self.graph,
            chunk_embedding_items(embeddings),
            r#"
            UNWIND $items AS chunk
            MATCH (c:DocumentChunk {id: chunk.id})
            CALL db.create.setNodeVectorProperty(c, 'embedding', chunk.embedding)
            SET c.embedding_model = $model,
                c.embedded_at = datetime()
            "#,
            |q| q.param("model", model_param.clone()),
        )
        .await
        .context("Failed to write document chunk embeddings")?;

        Ok(embeddings.len())
    }

    /// Get a document by ID.
    pub async fn get_document(&self, id: Uuid) -> Result<Option<Document>> {
        let q = query(
            r#"
            MATCH (d:Document {id: $id})
            RETURN d
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            Ok(Some(Self::node_to_document(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Find an already-ingested document by content hash.
    ///
    /// The point of storing `sha256`: re-uploading the same bytes should reuse
    /// the existing chunks and embeddings rather than pay for them twice.
    /// Scoped to a project when one is given, because the same file attached to
    /// two projects is two documents.
    pub async fn find_document_by_sha256(
        &self,
        sha256: &str,
        project_id: Option<Uuid>,
    ) -> Result<Option<Document>> {
        let cypher = if project_id.is_some() {
            r#"
            MATCH (d:Document {sha256: $sha256})
            WHERE d.project_id = $project_id
            RETURN d
            ORDER BY d.created_at ASC
            LIMIT 1
            "#
        } else {
            r#"
            MATCH (d:Document {sha256: $sha256})
            RETURN d
            ORDER BY d.created_at ASC
            LIMIT 1
            "#
        };

        let mut q = query(cypher).param("sha256", sha256.to_string());
        if let Some(pid) = project_id {
            q = q.param("project_id", pid.to_string());
        }

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            Ok(Some(Self::node_to_document(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List the documents of a project, most recent first.
    pub async fn list_project_documents(&self, project_id: Uuid) -> Result<Vec<Document>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:HAS_DOCUMENT]->(d:Document)
            RETURN d
            ORDER BY d.created_at DESC
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut documents = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            documents.push(Self::node_to_document(&node)?);
        }
        Ok(documents)
    }

    /// Get the chunks of a document, in document order.
    pub async fn get_document_chunks(&self, document_id: Uuid) -> Result<Vec<DocumentChunk>> {
        let q = query(
            r#"
            MATCH (:Document {id: $document_id})-[:HAS_CHUNK]->(c:DocumentChunk)
            RETURN c
            ORDER BY c.ordinal ASC
            "#,
        )
        .param("document_id", document_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut chunks = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("c")?;
            chunks.push(Self::node_to_document_chunk(&node)?);
        }
        Ok(chunks)
    }

    /// Delete a document and every chunk hanging off it.
    ///
    /// The chunks go with it: a `DocumentChunk` with no `Document` is an orphan
    /// nothing can interpret, and it would still be returned by vector search.
    /// `DETACH` also drops the `LINKED_TO` edges to the rest of the knowledge
    /// graph. Returns whether a document was actually deleted.
    pub async fn delete_document(&self, id: Uuid) -> Result<bool> {
        let q = query(
            r#"
            MATCH (d:Document {id: $id})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:DocumentChunk)
            DETACH DELETE d, c
            RETURN count(*) AS deleted
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let deleted: i64 = row.get("deleted")?;
            Ok(deleted > 0)
        } else {
            Ok(false)
        }
    }

    // ========================================================================
    // Putting the document *in* the knowledge graph
    // ========================================================================

    /// Link a document to any knowledge entity via `LINKED_TO`.
    ///
    /// Same relation type and same `last_verified` bookkeeping as
    /// `link_note_to_entity`, so a document reached this way is indistinguishable
    /// from a note for every traversal that already walks `LINKED_TO`.
    pub async fn link_document_to_entity(
        &self,
        document_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> Result<()> {
        let (label, field, suffix_match) = entity_match(entity_type, entity_id);
        let value = entity_match_value(entity_type, entity_id);

        let cypher = if suffix_match {
            format!(
                r#"
                MATCH (d:Document {{id: $document_id}})
                MATCH (e:{})
                WHERE e.{} ENDS WITH $entity_id
                MERGE (d)-[r:LINKED_TO]->(e)
                SET r.last_verified = datetime()
                "#,
                label, field
            )
        } else {
            format!(
                r#"
                MATCH (d:Document {{id: $document_id}})
                MATCH (e:{} {{{}: $entity_id}})
                MERGE (d)-[r:LINKED_TO]->(e)
                SET r.last_verified = datetime()
                "#,
                label, field
            )
        };

        let q = query(&cypher)
            .param("document_id", document_id.to_string())
            .param("entity_id", value);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Remove the `LINKED_TO` edge between a document and an entity.
    ///
    /// Matched undirected: the edge may have been written from either side.
    pub async fn unlink_document_from_entity(
        &self,
        document_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> Result<()> {
        let (label, field, suffix_match) = entity_match(entity_type, entity_id);
        let value = entity_match_value(entity_type, entity_id);

        let cypher = if suffix_match {
            format!(
                r#"
                MATCH (d:Document {{id: $document_id}})-[r:LINKED_TO]-(e:{})
                WHERE e.{} ENDS WITH $entity_id
                DELETE r
                "#,
                label, field
            )
        } else {
            format!(
                r#"
                MATCH (d:Document {{id: $document_id}})-[r:LINKED_TO]-(e:{} {{{}: $entity_id}})
                DELETE r
                "#,
                label, field
            )
        };

        let q = query(&cypher)
            .param("document_id", document_id.to_string())
            .param("entity_id", value);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Documents attached to an entity, whichever side wrote the edge.
    pub async fn get_documents_for_entity(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> Result<Vec<Document>> {
        let (label, field, suffix_match) = entity_match(entity_type, entity_id);
        let value = entity_match_value(entity_type, entity_id);

        let cypher = if suffix_match {
            format!(
                r#"
                MATCH (d:Document)-[:LINKED_TO]-(e:{})
                WHERE e.{} ENDS WITH $entity_id
                RETURN DISTINCT d
                ORDER BY d.created_at DESC
                "#,
                label, field
            )
        } else {
            format!(
                r#"
                MATCH (d:Document)-[:LINKED_TO]-(e:{} {{{}: $entity_id}})
                RETURN DISTINCT d
                ORDER BY d.created_at DESC
                "#,
                label, field
            )
        };

        let q = query(&cypher).param("entity_id", value);

        let mut result = self.graph.execute(q).await?;
        let mut documents = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("d")?;
            documents.push(Self::node_to_document(&node)?);
        }
        Ok(documents)
    }

    /// Link a note to a document: `(:Note)-[:LINKED_TO]->(:Document)`.
    ///
    /// Byte-for-byte the edge `link_note_to_entity` would write, so a note
    /// anchored to a document is read back by the existing note traversals.
    pub async fn link_note_to_document(&self, note_id: Uuid, document_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (n:Note {id: $note_id})
            MATCH (d:Document {id: $document_id})
            MERGE (n)-[r:LINKED_TO]->(d)
            SET r.last_verified = datetime()
            "#,
        )
        .param("note_id", note_id.to_string())
        .param("document_id", document_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Semantic search over chunks
    // ========================================================================

    /// Search chunks by vector similarity, the way `vector_search_notes`
    /// searches notes: the `document_chunk_embeddings` HNSW index, cosine
    /// scores, descending.
    pub async fn vector_search_document_chunks(
        &self,
        embedding: &[f32],
        limit: usize,
        project_id: Option<Uuid>,
        min_similarity: Option<f64>,
    ) -> Result<Vec<DocumentChunkHit>> {
        let embedding_f64: Vec<f64> = embedding.iter().map(|&v| v as f64).collect();

        // Over-fetch, since project filtering and the similarity floor are
        // applied after the index has already picked its neighbours.
        let query_limit = if project_id.is_some() {
            limit * 3
        } else {
            limit * 2
        };

        let cypher = if project_id.is_some() {
            r#"
            CALL db.index.vector.queryNodes('document_chunk_embeddings', $query_limit, $embedding)
            YIELD node AS c, score
            MATCH (d:Document)-[:HAS_CHUNK]->(c)
            WHERE d.project_id = $project_id
            RETURN d.id AS document_id, d.filename AS filename, c, score
            ORDER BY score DESC
            LIMIT $limit
            "#
        } else {
            r#"
            CALL db.index.vector.queryNodes('document_chunk_embeddings', $query_limit, $embedding)
            YIELD node AS c, score
            MATCH (d:Document)-[:HAS_CHUNK]->(c)
            RETURN d.id AS document_id, d.filename AS filename, c, score
            ORDER BY score DESC
            LIMIT $limit
            "#
        };

        let mut q = query(cypher)
            .param("query_limit", query_limit as i64)
            .param("embedding", embedding_f64)
            .param("limit", limit as i64);

        if let Some(pid) = project_id {
            q = q.param("project_id", pid.to_string());
        }

        let mut result = self.graph.execute(q).await?;
        let mut hits = Vec::new();

        while let Some(row) = result.next().await? {
            let score: f64 = row.get("score")?;
            if let Some(floor) = min_similarity {
                if score < floor {
                    continue;
                }
            }

            let document_id: String = row.get("document_id")?;
            let filename: String = row.get("filename").unwrap_or_default();
            let node: neo4rs::Node = row.get("c")?;

            hits.push(DocumentChunkHit {
                document_id: document_id.parse()?,
                filename,
                chunk: Self::node_to_document_chunk(&node)?,
                score,
            });
        }

        Ok(hits)
    }

    // ========================================================================
    // Row mapping
    // ========================================================================

    fn node_to_document(node: &neo4rs::Node) -> Result<Document> {
        let format_str: String = node
            .get("format")
            .unwrap_or_else(|_| "plain_text".to_string());

        Ok(Document {
            id: node.get::<String>("id")?.parse()?,
            filename: node.get("filename").unwrap_or_default(),
            format: format_from_str(&format_str),
            sha256: node.get("sha256").unwrap_or_default(),
            size_bytes: node.get::<i64>("size_bytes").unwrap_or(0).max(0) as u64,
            page_count: node.get::<i64>("page_count").unwrap_or(0).max(0) as usize,
            chunk_count: node.get::<i64>("chunk_count").unwrap_or(0).max(0) as usize,
            warnings: node.get("warnings").unwrap_or_default(),
            created_at: node
                .get::<String>("created_at")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(Utc::now),
            project_id: node
                .get::<String>("project_id")
                .ok()
                .and_then(|s| s.parse().ok()),
            session_id: node
                .get::<String>("session_id")
                .ok()
                .and_then(|s| s.parse().ok()),
        })
    }

    fn node_to_document_chunk(node: &neo4rs::Node) -> Result<DocumentChunk> {
        Ok(DocumentChunk {
            id: node.get::<String>("id")?.parse()?,
            text: node.get("text").unwrap_or_default(),
            start_byte: node.get::<i64>("start_byte").unwrap_or(0).max(0) as usize,
            end_byte: node.get::<i64>("end_byte").unwrap_or(0).max(0) as usize,
            page: node
                .get::<i64>("page")
                .ok()
                .filter(|p| *p > 0)
                .map(|p| p as usize),
            ordinal: node.get::<i64>("ordinal").unwrap_or(0).max(0) as usize,
            embedding: node
                .get::<Vec<f64>>("embedding")
                .ok()
                .map(|v| v.iter().map(|&x| x as f32).collect()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::batch::BATCH_SIZE;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::GraphStore;

    fn a_document(project_id: Option<Uuid>) -> Document {
        Document {
            id: Uuid::new_v4(),
            filename: "rfc-attachments.pdf".to_string(),
            format: DocumentFormat::Pdf,
            sha256: "e3b0c44298fc1c149afbf4c8996fb924".to_string(),
            size_bytes: 12_345,
            page_count: 4,
            chunk_count: 3,
            warnings: vec!["page 3 recovered with low confidence".to_string()],
            created_at: Utc::now(),
            project_id,
            session_id: None,
        }
    }

    fn chunks(n: usize) -> Vec<DocumentChunk> {
        (0..n)
            .map(|i| DocumentChunk {
                id: Uuid::new_v4(),
                text: format!("chunk {}", i),
                start_byte: i * 100,
                end_byte: (i + 1) * 100,
                page: Some(i / 2 + 1),
                ordinal: i,
                embedding: None,
            })
            .collect()
    }

    #[test]
    fn chunk_items_carry_every_persisted_field() {
        let items = document_chunk_items(&chunks(1));
        assert_eq!(items.len(), 1);
        let item = &items[0];
        for key in ["id", "text", "start_byte", "end_byte", "page", "ordinal"] {
            assert!(item.contains_key(key), "missing key {}", key);
        }
    }

    #[test]
    fn an_unpaginated_chunk_writes_a_null_page() {
        let mut c = chunks(1);
        c[0].page = None;
        let items = document_chunk_items(&c);
        assert!(matches!(items[0]["page"], neo4rs::BoltType::Null(_)));
    }

    #[test]
    fn five_hundred_chunks_are_one_unwind_batch() {
        let items = document_chunk_items(&chunks(500));
        assert_eq!(items.len(), 500);
        // This is what upsert_document_chunks hands to run_unwind_in_chunks_with:
        // 500 chunks are a single query, not 500.
        assert_eq!(items.chunks(BATCH_SIZE).count(), 1);
    }

    #[test]
    fn embedding_items_skip_chunks_without_a_vector() {
        let id = Uuid::new_v4();
        let items = chunk_embedding_items(&[(id, vec![0.1, 0.2, 0.3])]);
        assert_eq!(items.len(), 1);
        assert!(items[0].contains_key("embedding"));
    }

    #[test]
    fn a_chunk_knows_the_span_it_came_from() {
        let source = "alpha beta gamma";
        let chunk = DocumentChunk {
            id: Uuid::new_v4(),
            text: "beta".to_string(),
            start_byte: 6,
            end_byte: 10,
            page: None,
            ordinal: 0,
            embedding: None,
        };
        assert_eq!(chunk.interval().slice(source), Some("beta"));
    }

    #[test]
    fn a_chunk_is_built_from_the_chunker_output() {
        let produced = crate::documents::chunk_text(
            "One sentence. Another sentence.",
            &crate::documents::ChunkConfig::default(),
        );
        assert!(!produced.is_empty());
        let chunk = DocumentChunk::from_text_chunk(&produced[0], 0, Some(1));
        assert_eq!(chunk.text, produced[0].text);
        assert_eq!(chunk.start_byte, produced[0].interval.start);
        assert_eq!(chunk.end_byte, produced[0].interval.end);
        assert_eq!(chunk.ordinal, 0);
    }

    #[test]
    fn file_entities_match_by_path_and_others_by_id() {
        let (label, field, suffix) = entity_match(&EntityType::File, "src/main.rs");
        assert_eq!((label, field, suffix), ("File", "path", true));
        assert_eq!(
            entity_match_value(&EntityType::File, "src/main.rs"),
            "/src/main.rs"
        );

        let (label, field, suffix) = entity_match(&EntityType::Note, "abc");
        assert_eq!((label, field, suffix), ("Note", "id", false));

        let (label, field, _) = entity_match(&EntityType::Decision, "abc");
        assert_eq!((label, field), ("Decision", "id"));
    }

    #[test]
    fn the_mcp_entity_type_document_parses_and_maps_to_the_document_label() {
        use std::str::FromStr;

        // This is the exact entry point the API takes:
        // `note_handlers.rs` does `entity_type.parse::<EntityType>()`.
        let parsed = EntityType::from_str("document").expect("\"document\" must parse");
        assert_eq!(parsed, EntityType::Document);
        assert_eq!(parsed.to_string(), "document");
        // PascalCase is accepted too, as for every other entity type.
        assert_eq!(
            EntityType::from_str("Document").unwrap(),
            EntityType::Document
        );

        // And the label/match-property note.rs interpolates into its Cypher.
        assert_eq!(parsed.neo4j_label(), "Document");
        assert_eq!(parsed.match_property(), "id");
    }

    #[tokio::test]
    async fn note_link_to_entity_document_reaches_the_document() {
        use std::str::FromStr;

        let store = MockGraphStore::new();
        let doc = a_document(None);
        store.create_document(&doc, &chunks(1)).await.unwrap();

        let note = crate::notes::Note::new(
            Some(Uuid::new_v4()),
            crate::notes::NoteType::Context,
            "the spec this decision rests on".to_string(),
            "test".to_string(),
        );
        let note_id = note.id;
        store.create_note(&note).await.unwrap();

        // The whole chain, as the MCP `note(action: "link_to_entity",
        // entity_type: "document", entity_id: <uuid>)` call runs it.
        let entity_type = EntityType::from_str("document").unwrap();
        store
            .link_note_to_entity(note_id, &entity_type, &doc.id.to_string(), None, None)
            .await
            .unwrap();

        let linked = store
            .get_notes_for_entity(&entity_type, &doc.id.to_string())
            .await
            .unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].id, note_id);
    }

    #[tokio::test]
    async fn chunks_come_back_in_document_order() {
        let store = MockGraphStore::new();
        let doc = a_document(None);
        let mut cs = chunks(3);
        cs.reverse();

        store.create_document(&doc, &cs).await.unwrap();

        let read = store.get_document_chunks(doc.id).await.unwrap();
        assert_eq!(read.len(), 3);
        assert_eq!(
            read.iter().map(|c| c.ordinal).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[tokio::test]
    async fn deleting_a_document_takes_its_chunks_with_it() {
        let store = MockGraphStore::new();
        let doc = a_document(None);
        store.create_document(&doc, &chunks(4)).await.unwrap();
        assert_eq!(store.get_document_chunks(doc.id).await.unwrap().len(), 4);

        assert!(store.delete_document(doc.id).await.unwrap());

        assert!(store.get_document(doc.id).await.unwrap().is_none());
        assert!(store.get_document_chunks(doc.id).await.unwrap().is_empty());
        // Deleting again is a no-op, not an error.
        assert!(!store.delete_document(doc.id).await.unwrap());
    }

    #[tokio::test]
    async fn a_document_links_to_a_note_like_any_other_node() {
        let store = MockGraphStore::new();
        let doc = a_document(None);
        store.create_document(&doc, &chunks(1)).await.unwrap();

        let note_id = Uuid::new_v4();
        store
            .link_document_to_entity(doc.id, &EntityType::Note, &note_id.to_string())
            .await
            .unwrap();

        let found = store
            .get_documents_for_entity(&EntityType::Note, &note_id.to_string())
            .await
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, doc.id);

        store
            .unlink_document_from_entity(doc.id, &EntityType::Note, &note_id.to_string())
            .await
            .unwrap();
        assert!(store
            .get_documents_for_entity(&EntityType::Note, &note_id.to_string())
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn a_document_links_to_decisions_and_files_too() {
        let store = MockGraphStore::new();
        let doc = a_document(None);
        store.create_document(&doc, &chunks(1)).await.unwrap();

        let decision_id = Uuid::new_v4().to_string();
        store
            .link_document_to_entity(doc.id, &EntityType::Decision, &decision_id)
            .await
            .unwrap();
        store
            .link_document_to_entity(doc.id, &EntityType::File, "/src/main.rs")
            .await
            .unwrap();

        assert_eq!(
            store
                .get_documents_for_entity(&EntityType::Decision, &decision_id)
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            store
                .get_documents_for_entity(&EntityType::File, "/src/main.rs")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn a_note_anchored_to_a_document_is_found_from_the_note_side() {
        let store = MockGraphStore::new();
        let doc = a_document(None);
        store.create_document(&doc, &chunks(1)).await.unwrap();

        let note_id = Uuid::new_v4();
        store.link_note_to_document(note_id, doc.id).await.unwrap();

        // The edge is undirected for reads, exactly as the Cypher is.
        let found = store
            .get_documents_for_entity(&EntityType::Note, &note_id.to_string())
            .await
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, doc.id);
    }

    #[tokio::test]
    async fn the_same_bytes_are_found_again_by_hash() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();
        let doc = a_document(Some(project_id));
        store.create_document(&doc, &chunks(1)).await.unwrap();

        let found = store
            .find_document_by_sha256(&doc.sha256, Some(project_id))
            .await
            .unwrap();
        assert_eq!(found.map(|d| d.id), Some(doc.id));

        // Another project does not see it.
        assert!(store
            .find_document_by_sha256(&doc.sha256, Some(Uuid::new_v4()))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn documents_are_listed_per_project() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();
        let mine = a_document(Some(project_id));
        let theirs = a_document(Some(Uuid::new_v4()));
        store.create_document(&mine, &[]).await.unwrap();
        store.create_document(&theirs, &[]).await.unwrap();

        let listed = store.list_project_documents(project_id).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, mine.id);
    }

    #[tokio::test]
    async fn chunks_are_searchable_by_vector_like_notes() {
        let store = MockGraphStore::new();
        let doc = a_document(None);
        let mut cs = chunks(2);
        cs[0].embedding = Some(vec![1.0, 0.0, 0.0]);
        cs[1].embedding = Some(vec![0.0, 1.0, 0.0]);
        store.create_document(&doc, &cs).await.unwrap();

        let hits = store
            .vector_search_document_chunks(&[0.9, 0.1, 0.0], 5, None, None)
            .await
            .unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].chunk.id, cs[0].id);
        assert_eq!(hits[0].document_id, doc.id);
        assert_eq!(hits[0].filename, doc.filename);
        assert!(hits[0].score > hits[1].score);
    }

    #[tokio::test]
    async fn embeddings_can_be_attached_after_ingestion() {
        let store = MockGraphStore::new();
        let doc = a_document(None);
        let cs = chunks(2);
        store.create_document(&doc, &cs).await.unwrap();

        // Nothing is searchable before the vectors exist.
        assert!(store
            .vector_search_document_chunks(&[1.0, 0.0, 0.0], 5, None, None)
            .await
            .unwrap()
            .is_empty());

        let written = store
            .set_document_chunk_embeddings(
                &[
                    (cs[0].id, vec![1.0, 0.0, 0.0]),
                    (cs[1].id, vec![0.0, 1.0, 0.0]),
                ],
                "nomic-embed-text",
            )
            .await
            .unwrap();
        assert_eq!(written, 2);

        let hits = store
            .vector_search_document_chunks(&[1.0, 0.0, 0.0], 5, None, Some(0.5))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.id, cs[0].id);
    }

    #[tokio::test]
    async fn rewriting_chunks_does_not_duplicate_them() {
        let store = MockGraphStore::new();
        let doc = a_document(None);
        let cs = chunks(3);
        store.create_document(&doc, &cs).await.unwrap();
        let written = store.upsert_document_chunks(doc.id, &cs).await.unwrap();

        assert_eq!(written, 3);
        assert_eq!(store.get_document_chunks(doc.id).await.unwrap().len(), 3);
    }
}
