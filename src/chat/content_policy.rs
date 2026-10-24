//! Content posture of a provider, and the knowledge only a permissive one may carry.
//!
//! Some endpoints run a safety classifier that refuses whole categories of legitimate
//! work (security research, exploitation notes) — and it reads the SYSTEM PROMPT, so once
//! such material is injected there, every turn of the session refuses, even a bare
//! "salut". The Claude Code CLI then falls back to a smaller model which, seeing the same
//! prompt, refuses again. The content is legitimate; the only lever the host owns is what
//! it puts in the prompt and which provider it sends it to.
//!
//! This module names the two sides of that choice:
//! * [`ContentPolicy`] — whether a provider's endpoint refuses such material
//!   ([`provider_content_policy`]);
//! * [`SENSITIVE_TAG`] — the reserved tag that marks a note (or persona) as carrying it
//!   ([`tags_are_sensitive`]).
//!
//! The composer uses them to leave a sensitive note out of the prompt when the session's
//! provider is restricted ([`crate::chat::manager::ChatManager::build_system_prompt_for`]),
//! so a normal conversation on such a provider is no longer refused for context it never
//! needed. Routing a conversation that DOES need the material to a permissive provider
//! (a candidate-filter reject reason, the cognitive layer's auto-move) is the planned
//! follow-up; this module is the shared vocabulary both halves read.

/// Reserved tag marking a note or persona as carrying material a restricted provider's
/// safety classifier refuses (legitimate security / exploitation work). Case-insensitive.
/// A session on a [`ContentPolicy::Restricted`] provider does not get it in its prompt.
pub const SENSITIVE_TAG: &str = "provider-sensitive";

/// Whether a provider's endpoint refuses provider-sensitive material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContentPolicy {
    /// The endpoint refuses it (a safety classifier flags it). Claude Code is this by
    /// default: Anthropic's safeguards. The default, so an unknown provider is treated as
    /// the cautious case.
    #[default]
    Restricted,
    /// The endpoint carries it.
    Permissive,
}

impl ContentPolicy {
    /// Whether sensitive knowledge must be kept out of this provider's prompt.
    pub fn is_restricted(self) -> bool {
        matches!(self, ContentPolicy::Restricted)
    }
}

/// The content posture of a provider, keyed by its id.
///
/// Claude Code (and the empty id, which resolves to it — a root session's default) is
/// [`ContentPolicy::Restricted`]: Anthropic's safeguards refuse the material. Every other
/// instance is [`ContentPolicy::Permissive`]. A per-instance operator override
/// (`InstanceRecord.content_policy`) is the planned refinement; this function is the
/// default it will fall back to.
pub fn provider_content_policy(provider_id: &str) -> ContentPolicy {
    match provider_id {
        "" | crate::chat::provider::resolver::CLAUDE_CODE => ContentPolicy::Restricted,
        _ => ContentPolicy::Permissive,
    }
}

/// Whether a note's (or persona's) tags mark it as carrying provider-sensitive material.
pub fn tags_are_sensitive(tags: &[String]) -> bool {
    tags.iter()
        .any(|t| t.trim().eq_ignore_ascii_case(SENSITIVE_TAG))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_code_and_the_empty_default_are_restricted_others_permissive() {
        assert!(provider_content_policy("claude-code").is_restricted());
        assert!(provider_content_policy("").is_restricted());
        assert!(!provider_content_policy("deepseek").is_restricted());
        assert!(!provider_content_policy("some-nim-instance").is_restricted());
    }

    #[test]
    fn the_default_policy_is_the_cautious_one() {
        assert_eq!(ContentPolicy::default(), ContentPolicy::Restricted);
        assert!(ContentPolicy::default().is_restricted());
        assert!(!ContentPolicy::Permissive.is_restricted());
    }

    #[test]
    fn the_sensitive_tag_is_matched_case_insensitively_and_trimmed() {
        assert!(tags_are_sensitive(&["provider-sensitive".to_string()]));
        assert!(tags_are_sensitive(&["PROVIDER-SENSITIVE".to_string()]));
        assert!(tags_are_sensitive(&[" Provider-Sensitive ".to_string()]));
        assert!(tags_are_sensitive(&[
            "other".to_string(),
            "provider-sensitive".to_string()
        ]));
        assert!(!tags_are_sensitive(&["sensitive".to_string()]));
        assert!(!tags_are_sensitive(&[]));
    }
}
