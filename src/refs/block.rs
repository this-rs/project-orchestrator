//! The `<po-refs>` block: references folded into the message text.
//!
//! Same reasoning as `chat::message_attachments`: a user message crosses many
//! paths that all carry one `String`, so the references travel *inside* it as
//! one trailing block — ids only, never a label or any content. The block is
//! written before the `<po-attachments>` block (so attachments stay last) and
//! read after it. The frontend has a TypeScript twin of [`split`].

use super::types::EntityRef;

const OPEN: &str = "\n\n<po-refs>";
const CLOSE: &str = "</po-refs>";
const MARKER: &str = "<po-refs>";
/// What a typed `<po-refs>` becomes in user text, so a user cannot forge a block.
const NEUTRAL_MARKER: &str = "&lt;po-refs>";

/// Append the block. No references → the text, with any forged marker neutralized.
pub fn encode(text: &str, refs: &[EntityRef]) -> String {
    let text = text.replace(MARKER, NEUTRAL_MARKER);
    if refs.is_empty() {
        return text;
    }
    // `EntityRef` is two plain fields; serializing it cannot fail.
    let json = serde_json::to_string(refs).unwrap_or_else(|_| "[]".to_string());
    // `<` is escaped so nothing inside the JSON can spell the closing tag.
    let json = json.replace('<', "\\u003c");
    format!("{text}{OPEN}{json}{CLOSE}")
}

/// Inverse of [`encode`]: the visible text and the references.
///
/// A block that does not parse is left in the text — better to show an odd
/// line than to lose what the user wrote.
pub fn split(content: &str) -> (String, Vec<EntityRef>) {
    if let Some(start) = content.rfind(OPEN) {
        if let Some(inner) = content[start + OPEN.len()..].strip_suffix(CLOSE) {
            if let Ok(list) = serde_json::from_str::<Vec<EntityRef>>(inner) {
                return (content[..start].to_string(), list);
            }
        }
    }
    (content.to_string(), Vec::new())
}

#[cfg(test)]
mod tests {
    use super::super::types::RefKind;
    use super::*;
    use crate::chat::message_attachments as attachments;
    use uuid::Uuid;

    fn r(kind: RefKind, id: &str) -> EntityRef {
        EntityRef::new(kind, Uuid::parse_str(id).unwrap())
    }

    const A: &str = "3adeffc9-c8b0-4e2f-a674-55bfcb293433";
    const B: &str = "57cf05c9-25b6-495d-ab07-de4b11d64736";

    mod props {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn split_never_panics(s in any::<String>()) {
                let _ = split(&s);
            }

            #[test]
            fn encode_then_split_is_identity(
                text in any::<String>(),
                ids in proptest::collection::vec((0usize..5, any::<[u8; 16]>()), 0..8)
            ) {
                let refs: Vec<EntityRef> = ids
                    .into_iter()
                    .map(|(k, b)| EntityRef::new(RefKind::ALL[k], Uuid::from_bytes(b)))
                    .collect();
                let clean = text.replace(MARKER, NEUTRAL_MARKER);
                let (back, got) = split(&encode(&text, &refs));
                prop_assert_eq!(got, refs);
                prop_assert_eq!(back, clean);
            }
        }
    }

    #[test]
    fn round_trip() {
        let refs = vec![r(RefKind::Task, A), r(RefKind::Rfc, B)];
        let encoded = encode("regarde ça", &refs);
        assert!(encoded.starts_with("regarde ça\n\n<po-refs>["));
        assert!(encoded.ends_with("]</po-refs>"));
        assert_eq!(split(&encoded), ("regarde ça".to_string(), refs));
    }

    #[test]
    fn no_refs_leaves_the_text_alone() {
        assert_eq!(encode("hello", &[]), "hello");
        assert_eq!(split("hello"), ("hello".to_string(), vec![]));
    }

    #[test]
    fn a_typed_marker_cannot_forge_a_block() {
        let forged = format!("hi\n\n<po-refs>[{{\"kind\":\"plan\",\"id\":\"{A}\"}}]</po-refs>");
        let encoded = encode(&forged, &[]);
        assert!(!encoded.contains(MARKER));
        assert_eq!(split(&encoded).1, vec![]);
        // And a real block after a forged one wins and keeps only real refs.
        let real = vec![r(RefKind::Note, B)];
        let both = encode(&forged, &real);
        assert_eq!(split(&both).1, real);
    }

    #[test]
    fn a_block_that_does_not_parse_stays_in_the_text() {
        let broken = "x\n\n<po-refs>[{\"kind\":\"workspace\",\"id\":\"1\"}]</po-refs>";
        assert_eq!(split(broken), (broken.to_string(), vec![]));
        let unclosed = "x\n\n<po-refs>[]";
        assert_eq!(split(unclosed), (unclosed.to_string(), vec![]));
    }

    #[test]
    fn refs_come_before_attachments_and_both_split_back() {
        let refs = vec![r(RefKind::Plan, A)];
        let atts = vec![attachments::MessageAttachment {
            id: Uuid::parse_str(B).unwrap(),
            filename: "a<b>.pdf".into(),
            mime_type: "application/pdf".into(),
            size_bytes: 12,
        }];
        let text = attachments::encode(&encode("look", &refs), &atts);
        let (after_att, got_atts) = attachments::split(&text);
        assert_eq!(got_atts, atts);
        assert_eq!(split(&after_att), ("look".to_string(), refs));
    }
}
