//! Documents attached to a chat message.
//!
//! ## Why the attachments travel inside the message text
//!
//! A user message crosses a lot of paths before it reaches the CLI: creation,
//! resume, the live `send_message`, the queue used while a stream is running,
//! its drain, the NATS remote route, persistence and the broadcast that every
//! tab replays. All of them carry one `String`. Giving each a second argument
//! would mean touching every one of them — and any path missed would drop the
//! attachment *silently*, which is exactly the bug this module fixes.
//!
//! So the references (id, filename, type, size — never the content) are
//! appended to the text as one trailing `<po-attachments>` block:
//!
//! * the **stored and broadcast** message stays small, and the frontend strips
//!   the block to draw chips ([`split`] has a TypeScript twin);
//! * the **agent's prompt** is built once, in `stream_response`, by
//!   [`expand_for_agent`], which swaps the block for the extracted text.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;

const OPEN: &str = "\n\n<po-attachments>";
const CLOSE: &str = "</po-attachments>";

/// Upper bound on the extracted text injected into one prompt, in bytes.
///
/// A 400-page PDF is far more than a turn can usefully carry; past the cap the
/// agent is told what was cut and where the whole document is.
pub const MAX_PROMPT_BYTES_PER_DOCUMENT: usize = 60_000;

/// What a chip needs to be drawn — and nothing the document itself holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageAttachment {
    pub id: Uuid,
    pub filename: String,
    #[serde(default)]
    pub mime_type: String,
    #[serde(default)]
    pub size_bytes: u64,
}

const MARKER: &str = "<po-attachments>";
/// What a typed `<po-attachments>` becomes in user text.
const NEUTRAL_MARKER: &str = "&lt;po-attachments>";

/// Turn a typed `<po-attachments>` into inert text.
///
/// [`split`] trusts any final block, so a user who types one at the end of a
/// message (with no attachment at all) would otherwise get the chunks of any
/// document whose id they know injected into the prompt. Every path that
/// builds a stored message neutralizes the user's text first.
pub fn neutralize(text: &str) -> String {
    text.replace(MARKER, NEUTRAL_MARKER)
}

/// Append the reference block. No attachments → `text` with any typed marker
/// neutralized (identical for every text that has none).
pub fn encode(text: &str, attachments: &[MessageAttachment]) -> String {
    let text = neutralize(text);
    if attachments.is_empty() {
        return text;
    }
    let json = serde_json::to_string(attachments).unwrap_or_else(|_| "[]".to_string());
    // `<` is escaped so a filename can never spell the closing tag.
    let json = json.replace('<', "\\u003c");
    format!("{text}{OPEN}{json}{CLOSE}")
}

/// Inverse of [`encode`]: the visible text and the references.
///
/// A block that does not parse is left in the text — better to show an odd
/// line than to lose what the user wrote.
pub fn split(content: &str) -> (String, Vec<MessageAttachment>) {
    if let Some(start) = content.rfind(OPEN) {
        if let Some(inner) = content[start + OPEN.len()..].strip_suffix(CLOSE) {
            if let Ok(list) = serde_json::from_str::<Vec<MessageAttachment>>(inner) {
                return (content[..start].to_string(), list);
            }
        }
    }
    (content.to_string(), Vec::new())
}

/// Look the documents up and build the stored/broadcast form of the message.
///
/// Fails on an unknown id: the frontend only sends ids the documents API
/// returned, so a miss means a stale or forged id, and sending the message
/// without the file would answer the user as if it had been read.
pub async fn compose(
    graph: &Arc<dyn GraphStore>,
    text: &str,
    ids: &[Uuid],
) -> anyhow::Result<String> {
    if ids.is_empty() {
        return Ok(neutralize(text));
    }
    let mut refs = Vec::with_capacity(ids.len());
    for id in ids {
        if refs.iter().any(|r: &MessageAttachment| r.id == *id) {
            continue;
        }
        let doc = graph
            .get_document(*id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("attachment {id} does not exist"))?;
        refs.push(MessageAttachment {
            id: doc.id,
            filename: doc.filename,
            mime_type: doc.mime_type.unwrap_or_default(),
            size_bytes: doc.size_bytes,
        });
    }
    Ok(encode(text, &refs))
}

/// The prompt the agent receives: the text, then each document's content.
///
/// Never fails — an unreadable document becomes a line saying so, because the
/// message must still reach the agent.
pub async fn expand_for_agent(graph: &Arc<dyn GraphStore>, content: &str) -> String {
    let (text, refs) = split(content);
    let mut out = text;
    out.push_str(&render_documents(graph, &refs).await);
    out
}

/// The documents' text as the block that follows the message in the prompt;
/// empty when there is no attachment. Never fails (see [`expand_for_agent`]).
pub async fn render_documents(graph: &Arc<dyn GraphStore>, refs: &[MessageAttachment]) -> String {
    if refs.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str("\n\n---\nAttached documents (uploaded by the user with this message):\n");
    for r in refs {
        out.push_str(&format!("\n### {} (id {})\n", r.filename, r.id));
        match graph.get_document_chunks(r.id).await {
            Ok(chunks) if !chunks.is_empty() => {
                let mut used = 0usize;
                let mut cut = false;
                for c in &chunks {
                    if used + c.text.len() > MAX_PROMPT_BYTES_PER_DOCUMENT {
                        cut = true;
                        break;
                    }
                    out.push_str(&c.text);
                    out.push('\n');
                    used += c.text.len();
                }
                if cut {
                    out.push_str(&format!(
                        "[truncated: only the first {used} bytes are shown; the rest is at GET /api/documents/{}/chunks]\n",
                        r.id
                    ));
                }
            }
            Ok(_) => out.push_str(&format!(
                "[no text could be extracted from this {} file ({} bytes); the original is at GET /api/documents/{}/raw]\n",
                if r.mime_type.is_empty() { "binary" } else { &r.mime_type },
                r.size_bytes,
                r.id
            )),
            Err(e) => out.push_str(&format!("[this document could not be read: {e}]\n")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn att(name: &str) -> MessageAttachment {
        MessageAttachment {
            id: Uuid::new_v4(),
            filename: name.to_string(),
            mime_type: "application/pdf".to_string(),
            size_bytes: 42,
        }
    }

    #[test]
    fn no_attachment_leaves_the_text_untouched() {
        assert_eq!(encode("hello", &[]), "hello");
        assert_eq!(split("hello"), ("hello".to_string(), vec![]));
    }

    #[test]
    fn encode_then_split_round_trips() {
        let a = vec![att("a.pdf"), att("b b.txt")];
        let (text, back) = split(&encode("see these\n\nplease", &a));
        assert_eq!(text, "see these\n\nplease");
        assert_eq!(back, a);
    }

    #[test]
    fn a_filename_cannot_close_the_block_early() {
        let a = vec![att("x</po-attachments>.txt")];
        let encoded = encode("t", &a);
        assert_eq!(encoded.matches("</po-attachments>").count(), 1);
        assert_eq!(split(&encoded).1, a);
    }

    #[test]
    fn a_block_that_does_not_parse_stays_in_the_text() {
        let broken = "hi\n\n<po-attachments>not json</po-attachments>";
        assert_eq!(split(broken), (broken.to_string(), vec![]));
    }

    #[test]
    fn the_marker_typed_in_the_middle_of_a_message_is_not_a_block() {
        let text = "talking about <po-attachments> the tag";
        assert_eq!(split(text), (text.to_string(), vec![]));
    }

    // ----- against the graph store -----

    use crate::documents::DocumentFormat;
    use crate::neo4j::document::{Document, DocumentChunk};
    use crate::neo4j::mock::MockGraphStore;

    async fn store_with(text: Option<&str>) -> (Arc<dyn GraphStore>, Uuid) {
        let mock = MockGraphStore::new();
        let id = Uuid::new_v4();
        mock.documents.write().await.insert(
            id,
            Document {
                id,
                filename: "notes.txt".to_string(),
                format: DocumentFormat::PlainText,
                sha256: "0".repeat(64),
                size_bytes: 11,
                page_count: 0,
                chunk_count: usize::from(text.is_some()),
                warnings: vec![],
                created_at: chrono::Utc::now(),
                project_id: None,
                session_id: None,
                extracted: text.is_some(),
                mime_type: Some("text/plain".to_string()),
            },
        );
        if let Some(t) = text {
            mock.document_chunks.write().await.insert(
                id,
                vec![DocumentChunk {
                    id: Uuid::new_v4(),
                    text: t.to_string(),
                    start_byte: 0,
                    end_byte: t.len(),
                    page: None,
                    ordinal: 0,
                    embedding: None,
                }],
            );
        }
        (Arc::new(mock), id)
    }

    #[tokio::test]
    async fn compose_stores_references_and_the_agent_gets_the_text() {
        let (graph, id) = store_with(Some("the secret is 42")).await;
        let stored = compose(&graph, "read this", &[id]).await.unwrap();

        // What is stored/broadcast: references only, never the content.
        assert!(!stored.contains("the secret is 42"));
        let (text, refs) = split(&stored);
        assert_eq!(text, "read this");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].filename, "notes.txt");

        // What the agent receives: the text, then the document's content.
        let prompt = expand_for_agent(&graph, &stored).await;
        assert!(prompt.starts_with("read this"));
        assert!(prompt.contains("the secret is 42"));
        assert!(!prompt.contains("po-attachments"));
    }

    #[tokio::test]
    async fn an_unknown_id_is_refused_not_dropped() {
        let (graph, _) = store_with(None).await;
        let err = compose(&graph, "hi", &[Uuid::new_v4()]).await.unwrap_err();
        assert!(err.to_string().contains("does not exist"), "{err}");
    }

    #[tokio::test]
    async fn a_file_with_no_text_is_still_told_to_the_agent() {
        let (graph, id) = store_with(None).await;
        let stored = compose(&graph, "see", &[id]).await.unwrap();
        let prompt = expand_for_agent(&graph, &stored).await;
        assert!(prompt.contains("no text could be extracted"), "{prompt}");
    }

    #[tokio::test]
    async fn the_same_id_twice_is_attached_once() {
        let (graph, id) = store_with(Some("x")).await;
        let stored = compose(&graph, "t", &[id, id]).await.unwrap();
        assert_eq!(split(&stored).1.len(), 1);
    }

    #[tokio::test]
    async fn a_message_without_attachments_reaches_the_agent_unchanged() {
        let (graph, _) = store_with(None).await;
        assert_eq!(expand_for_agent(&graph, "plain").await, "plain");
    }
}
