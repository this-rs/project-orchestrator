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
//! * the tags that mark a note (or persona) as carrying it — [`SENSITIVE_TAG`] plus the
//!   known-sensitive [`SENSITIVE_TAGS`] and [`SENSITIVE_TAG_PREFIXES`]
//!   ([`tags_are_sensitive`]).
//!
//! The composer uses them to leave a sensitive note out of the prompt when the session's
//! provider is restricted ([`crate::chat::manager::ChatManager::build_system_prompt_for`]),
//! so a normal conversation on such a provider is no longer refused for context it never
//! needed. Routing a conversation that DOES need the material to a permissive provider
//! (a candidate-filter reject reason, the cognitive layer's auto-move) is the planned
//! follow-up; this module is the shared vocabulary both halves read.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

/// Reserved tag marking a note or persona as carrying material a restricted provider's
/// safety classifier refuses (legitimate security / exploitation work). Case-insensitive.
/// A session on a [`ContentPolicy::Restricted`] provider does not get it in its prompt.
pub const SENSITIVE_TAG: &str = "provider-sensitive";

/// Tags that mark exploitation / security-research material even without the explicit
/// [`SENSITIVE_TAG`] marker. Requiring a human to add `provider-sensitive` by hand let the
/// material leak: one untagged `exploit` note in the system prompt turns every turn of a
/// restricted session into a refusal, even a bare "salut". These are the tags such work
/// already carries, so the filter catches it without a second manual pass.
pub const SENSITIVE_TAGS: &[&str] = &[
    "exploit",
    "kernel-exploit",
    "security-research",
    // The codename of the local device-exploitation work (CVE-2026-43499).
    "ghostlock",
];

/// Tag prefixes that mark such material: a CVE identifier (`cve-2026-43499`) names an
/// exploitation target by construction.
pub const SENSITIVE_TAG_PREFIXES: &[&str] = &["cve-"];

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
///
/// A tag marks it explicitly ([`SENSITIVE_TAG`]), by name ([`SENSITIVE_TAGS`]), or by
/// prefix ([`SENSITIVE_TAG_PREFIXES`] — a CVE identifier). Case-insensitive, trimmed.
/// Unrelated engineering tags (`futex`, `slub`, `exploit-prevention`) stay visible to a
/// restricted provider.
pub fn tags_are_sensitive(tags: &[String]) -> bool {
    tags.iter().any(|raw| {
        let tag = raw.trim();
        tag.eq_ignore_ascii_case(SENSITIVE_TAG)
            || SENSITIVE_TAGS
                .iter()
                .any(|known| tag.eq_ignore_ascii_case(known))
            || SENSITIVE_TAG_PREFIXES.iter().any(|prefix| {
                tag.len() >= prefix.len()
                    && tag.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

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

    #[test]
    fn exploitation_tags_are_sensitive_without_the_explicit_marker() {
        assert!(tags_are_sensitive(&["exploit".to_string()]));
        assert!(tags_are_sensitive(&["EXPLOIT".to_string()]));
        assert!(tags_are_sensitive(&["kernel-exploit".to_string()]));
        assert!(tags_are_sensitive(&["security-research".to_string()]));
        assert!(tags_are_sensitive(&["ghostlock".to_string()]));
        assert!(tags_are_sensitive(&[
            "cve-2026-43499".to_string(),
            "futex".to_string()
        ]));
    }

    #[test]
    fn a_cve_identifier_is_sensitive_by_its_prefix() {
        assert!(tags_are_sensitive(&["cve-2026-43499".to_string()]));
        assert!(tags_are_sensitive(&["CVE-2026-43499".to_string()]));
        assert!(tags_are_sensitive(&["cve-".to_string()]));
    }

    #[test]
    fn unrelated_engineering_tags_stay_visible_to_a_restricted_provider() {
        // The extension must not sweep ordinary kernel notes into the sensitive set:
        // `futex` alone is a subsystem, not an exploitation marker, and a defensive note
        // tagged `exploit-prevention` must still reach Claude Code.
        assert!(!tags_are_sensitive(&["futex".to_string()]));
        assert!(!tags_are_sensitive(&["slub".to_string()]));
        assert!(!tags_are_sensitive(&["kernelsnitch".to_string()]));
        assert!(!tags_are_sensitive(&["gotcha".to_string()]));
        assert!(!tags_are_sensitive(&["exploit-prevention".to_string()]));
    }
}
