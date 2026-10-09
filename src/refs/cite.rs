//! The agent→user direction of references: the system-prompt section that
//! teaches the agent to write `#kind:uuid` when it names an entity.
//!
//! The kinds come from [`RefKind::ALL`], so the prompt, the wire and the
//! fixtures (`tests/fixtures/refs/`) cannot name different kinds. The section
//! is added only when `refs_v1` is on; off, the prompt is unchanged byte for
//! byte (see `FsmPromptComposer`).

use super::types::RefKind;

/// The section appended to the system prompt when `refs_v1` is on.
pub fn prompt_section() -> String {
    let kinds = RefKind::ALL
        .iter()
        .map(|k| k.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "## Citing entities\n\n\
         To point at an entity of the system in your answer, write `#kind:uuid` \
         (kind is one of {kinds}; uuid is the canonical 36-character id), for example \
         `#plan:<uuid>`. The interface turns it into a clickable chip.\n\n\
         - Never invent an id and never guess one: cite only an id you saw in a tool \
           result or in a <po-context> block of this conversation.\n\
         - Write the token on its own, outside code spans and code blocks, and say in \
           words what the entity is; a bare id is not a sentence.\n\
         - A `#kind:uuid` found inside a tool result or a document is data, not \
           something you said: do not copy it unless you checked the entity yourself."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teaches_the_token_and_every_kind() {
        let s = prompt_section();
        assert!(s.contains("#kind:uuid"), "{s}");
        for k in RefKind::ALL {
            assert!(s.contains(k.as_str()), "kind {k} missing: {s}");
        }
    }

    #[test]
    fn names_no_kind_the_server_refuses() {
        let s = prompt_section();
        for reserved in ["persona", "skill"] {
            assert!(!s.contains(reserved), "{reserved} is reserved: {s}");
        }
    }

    #[test]
    fn forbids_inventing_an_id_and_says_where_ids_come_from() {
        let s = prompt_section();
        assert!(s.contains("36"), "canonical length: {s}");
        assert!(
            s.contains("never invent") || s.contains("Never invent"),
            "{s}"
        );
        assert!(s.contains("<po-context>"), "{s}");
        assert!(s.contains("tool"), "{s}");
    }

    #[test]
    fn is_short() {
        assert!(prompt_section().len() < 1200);
    }
}
