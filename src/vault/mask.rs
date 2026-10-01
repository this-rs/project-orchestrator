//! Masking: the net under the vault.
//!
//! Agents use secrets inside shell pipelines, so a value normally never appears
//! in what they print. "Normally" is not enough: an `echo $X` by mistake, a tool
//! that prints its config, an error message quoting a URL with credentials. When
//! a secret value shows up in agent output, it must be replaced before the text
//! is stored, indexed or broadcast — because each of those copies outlives the
//! session and is readable by more people than the one who typed the secret.
//!
//! Matching is by EXACT value, not by pattern. Pattern-based scanners
//! (`anonymize.rs`) guess what looks secret and miss what doesn't; here we know
//! the values, so there is nothing to guess. Two forms of each value are
//! matched: the raw one, and its JSON-escaped form — events are stored as JSON,
//! and a value containing `"` or `\` does not appear verbatim there.
//!
//! Which values: only those actually delivered to an agent (or typed at an
//! agent's request) since the server started. Those are the only ones that can
//! appear in output, and keeping every decrypted value in memory "just in case"
//! would defeat the point of locking the vault. The set is kept after the vault
//! locks — an agent may still print a value it received while it was open.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use zeroize::Zeroizing;

/// Immutable snapshot used on the hot path; swapped whole when the set changes,
/// so masking never waits on a writer.
#[derive(Default)]
pub struct Masker {
    /// (needle, replacement), longest needle first — so a secret that contains
    /// another is masked whole, not half-replaced by the shorter one.
    needles: Vec<(Zeroizing<String>, String)>,
}

impl std::fmt::Debug for Masker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let labels: Vec<&str> = self.needles.iter().map(|(_, l)| l.as_str()).collect();
        f.debug_struct("Masker").field("labels", &labels).finish()
    }
}

impl Masker {
    pub fn from_values<'a>(values: impl IntoIterator<Item = (&'a str, &'a str)>) -> Self {
        let mut needles: Vec<(Zeroizing<String>, String)> = Vec::new();
        for (name, value) in values {
            let label = format!("[secret:{name}]");
            needles.push((Zeroizing::new(value.to_string()), label.clone()));
            let escaped = json_escaped(value);
            if escaped != value {
                needles.push((Zeroizing::new(escaped), label));
            }
        }
        needles.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Self { needles }
    }

    pub fn is_empty(&self) -> bool {
        self.needles.is_empty()
    }

    /// Replace every known secret value in `text`. Borrows when there is
    /// nothing to replace, which is the overwhelmingly common case.
    pub fn mask<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if self.needles.is_empty() || !self.needles.iter().any(|(n, _)| text.contains(n.as_str())) {
            return Cow::Borrowed(text);
        }
        let mut out = text.to_string();
        for (needle, label) in &self.needles {
            if out.contains(needle.as_str()) {
                out = out.replace(needle.as_str(), label);
            }
        }
        Cow::Owned(out)
    }
}

/// The value as it appears inside a JSON string literal.
fn json_escaped(value: &str) -> String {
    let quoted = serde_json::to_string(value).unwrap_or_default();
    quoted
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(&quoted)
        .to_string()
}

/// The masker every output path of this process uses. One per process because
/// the paths that need it (stream, out-of-band listener, drain) are spread over
/// many call sites; the vault registers into it, those paths read from it.
pub fn global() -> &'static SharedMasker {
    static GLOBAL: std::sync::OnceLock<SharedMasker> = std::sync::OnceLock::new();
    GLOBAL.get_or_init(SharedMasker::default)
}

/// Mask every known secret inside any serialisable value (a CLI message, a
/// chat event), by masking its JSON form and reading it back.
///
/// Free when nothing is registered or nothing matches — the overwhelmingly
/// common case. Returns `Err(original)` only if the masked JSON no longer
/// deserialises (cannot happen with derived, symmetric serde impls; tested on
/// the CLI message type) — the caller decides what to do then.
pub fn mask_serde<T>(masker: &Masker, value: T) -> Result<T, T>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    if masker.is_empty() {
        return Ok(value);
    }
    let Ok(json) = serde_json::to_string(&value) else {
        return Err(value);
    };
    match masker.mask(&json) {
        Cow::Borrowed(_) => Ok(value),
        Cow::Owned(masked) => serde_json::from_str(&masked).map_err(|_| value),
    }
}

/// The process-wide masking set: values delivered to agents since start.
#[derive(Default, Clone)]
pub struct SharedMasker {
    inner: Arc<RwLock<InnerMasker>>,
}

#[derive(Default)]
struct InnerMasker {
    values: BTreeMap<String, Zeroizing<String>>,
    snapshot: Arc<Masker>,
}

impl std::fmt::Debug for SharedMasker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SharedMasker(..)")
    }
}

impl SharedMasker {
    /// Record a value as live: from now on it is masked everywhere.
    pub fn register(&self, name: &str, value: &str) {
        let mut inner = self.inner.write().unwrap_or_else(|e| e.into_inner());
        inner
            .values
            .insert(name.to_string(), Zeroizing::new(value.to_string()));
        inner.snapshot = Arc::new(Masker::from_values(
            inner.values.iter().map(|(n, v)| (n.as_str(), v.as_str())),
        ));
    }

    /// Whether a value under this name was delivered (and is being masked).
    pub fn knows(&self, name: &str) -> bool {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values
            .contains_key(name)
    }

    /// Forget a value (the secret was deleted from the vault).
    pub fn forget(&self, name: &str) {
        let mut inner = self.inner.write().unwrap_or_else(|e| e.into_inner());
        inner.values.remove(name);
        inner.snapshot = Arc::new(Masker::from_values(
            inner.values.iter().map(|(n, v)| (n.as_str(), v.as_str())),
        ));
    }

    /// Current snapshot, cheap to clone and use without holding the lock.
    pub fn snapshot(&self) -> Arc<Masker> {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .snapshot
            .clone()
    }

    /// Convenience for call sites that mask a single string.
    pub fn mask(&self, text: &str) -> String {
        self.snapshot().mask(text).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_known_value_is_replaced_by_its_label() {
        let m = Masker::from_values([("demo-secret", "the-demo-passphrase")]);
        assert_eq!(
            m.mask("login with the-demo-passphrase ok"),
            "login with [secret:demo-secret] ok"
        );
    }

    #[test]
    fn text_without_secrets_is_borrowed_untouched() {
        let m = Masker::from_values([("demo-secret", "the-demo-passphrase")]);
        assert!(matches!(m.mask("nothing to see"), Cow::Borrowed(_)));
    }

    #[test]
    fn every_occurrence_is_replaced() {
        let m = Masker::from_values([("k", "s3cr3t-value-1")]);
        assert_eq!(
            m.mask("a s3cr3t-value-1 b s3cr3t-value-1"),
            "a [secret:k] b [secret:k]"
        );
    }

    #[test]
    fn the_longer_secret_wins_when_one_contains_another() {
        // Otherwise "token-abc" would turn "token-abc-extended" into
        // "[secret:short]-extended" and leak the suffix.
        let m = Masker::from_values([("short", "token-abc"), ("long", "token-abc-extended")]);
        assert_eq!(m.mask("x token-abc-extended y"), "x [secret:long] y");
        assert_eq!(m.mask("x token-abc y"), "x [secret:short] y");
    }

    #[test]
    fn a_value_is_masked_inside_serialised_json_too() {
        // Events are persisted as JSON; a value with a quote or a backslash
        // does not appear verbatim there.
        let value = r#"pa"ss\word-123"#;
        let m = Masker::from_values([("tricky", value)]);
        let event = serde_json::json!({"type": "tool_result", "result": format!("echo {value}")});
        let serialised = serde_json::to_string(&event).unwrap();
        assert!(!serialised.contains(value), "precondition: JSON escapes it");
        let masked = m.mask(&serialised);
        assert!(masked.contains("[secret:tricky]"), "{masked}");
        assert!(!masked.contains(r#"pa\"ss"#));
    }

    #[test]
    fn the_shared_set_masks_what_was_registered_and_forgets_on_delete() {
        let shared = SharedMasker::default();
        assert_eq!(shared.mask("the-demo-passphrase"), "the-demo-passphrase");
        shared.register("demo-secret", "the-demo-passphrase");
        assert_eq!(shared.mask("the-demo-passphrase"), "[secret:demo-secret]");
        shared.forget("demo-secret");
        assert_eq!(shared.mask("the-demo-passphrase"), "the-demo-passphrase");
    }

    /// The CLI messages the chat masks: every shape a value can travel in.
    fn cli_messages(v: &str) -> Vec<nexus_claude::Message> {
        use nexus_claude::{
            AssistantMessage, ContentBlock, ContentValue, Message, TextContent, ToolResultContent,
            ToolUseContent, UserMessage,
        };
        vec![
            Message::Assistant {
                message: AssistantMessage {
                    content: vec![
                        ContentBlock::Text(TextContent {
                            text: format!("the key is {v}"),
                        }),
                        ContentBlock::ToolUse(ToolUseContent {
                            id: "t1".into(),
                            name: "Bash".into(),
                            input: serde_json::json!({"command": format!("curl -u me:{v} x")}),
                        }),
                    ],
                },
                parent_tool_use_id: None,
            },
            Message::User {
                message: UserMessage {
                    content: String::new(),
                    content_blocks: Some(vec![ContentBlock::ToolResult(ToolResultContent {
                        tool_use_id: "t1".into(),
                        content: Some(ContentValue::Text(format!("PASSWORD={v}\n"))),
                        is_error: None,
                    })]),
                },
                parent_tool_use_id: Some("parent".into()),
            },
        ]
    }

    #[test]
    fn the_round_trip_changes_the_value_and_nothing_else() {
        // Masked message == the same message built with the label in place of
        // the value: proves the JSON round trip loses no field.
        let m = Masker::from_values([("k", "sk-live-0123456789")]);
        for (masked, expected) in cli_messages("sk-live-0123456789")
            .into_iter()
            .zip(cli_messages("[secret:k]"))
        {
            assert_eq!(mask_serde(&m, masked).unwrap(), expected);
        }
    }

    #[test]
    fn cli_messages_come_back_masked_in_every_field() {
        let value = r#"s3cr"et\value"#; // needs JSON escaping
        let m = Masker::from_values([("k", value)]);
        for msg in cli_messages(value) {
            let masked = mask_serde(&m, msg).expect("still a valid message");
            let json = serde_json::to_string(&masked).unwrap();
            assert!(json.contains("[secret:k]"), "{json}");
            assert!(!json.contains("s3cr"), "{json}");
        }
    }

    #[test]
    fn debug_shows_labels_never_values() {
        let shared = SharedMasker::default();
        shared.register("demo-secret", "the-demo-passphrase");
        let debug = format!("{:?}", shared.snapshot());
        assert!(debug.contains("[secret:demo-secret]"));
        assert!(!debug.contains("the-demo-passphrase"));
    }
}
