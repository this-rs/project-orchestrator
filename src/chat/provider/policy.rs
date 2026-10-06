//! Neutral policy modes accepted on top of the legacy strings (decisions A8, A43).
//!
//! The API accepts both vocabularies and emits both forms:
//!
//! | legacy (Claude Code) | neutral |
//! |---|---|
//! | `default`, `manual`, `dontAsk` | `ask` |
//! | `acceptEdits`, `auto` | `auto_edits` |
//! | `plan` | `plan_only` |
//! | `bypassPermissions` | `trust` |
//!
//! The table mirrors the nexus Claude Code adapter (contract §6).

use nexus_claude::agent::{PolicyMode, ToolPolicy};

/// Legacy mode strings, as the Claude CLI names them.
pub const LEGACY_MODES: [&str; 7] = [
    "default",
    "acceptEdits",
    "plan",
    "bypassPermissions",
    "auto",
    "dontAsk",
    "manual",
];

/// Neutral mode strings, as nexus serialises [`PolicyMode`].
pub const NEUTRAL_MODES: [&str; 4] = ["ask", "auto_edits", "plan_only", "trust"];

/// A mode in both vocabularies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModePair {
    /// Neutral mode.
    pub neutral: PolicyMode,
    /// Legacy string to hand to the Claude CLI.
    pub legacy: &'static str,
}

/// The legacy string equal to `input`, as a static string.
fn legacy_static(input: &str) -> Option<&'static str> {
    LEGACY_MODES.iter().copied().find(|mode| *mode == input)
}

/// Parses a legacy or a neutral mode string. Unknown → `None`.
///
/// A legacy input keeps its own string as `legacy` (`auto`, `dontAsk` and
/// `manual` are not flattened); a neutral input gets its canonical legacy name.
pub fn parse_mode(input: &str) -> Option<ModePair> {
    let pair = |neutral: PolicyMode, legacy: &'static str| Some(ModePair { neutral, legacy });
    match input {
        // Legacy strings.
        "default" => pair(PolicyMode::Ask, "default"),
        "manual" => pair(PolicyMode::Ask, "manual"),
        "dontAsk" => pair(PolicyMode::Ask, "dontAsk"),
        "acceptEdits" => pair(PolicyMode::AutoEdits, "acceptEdits"),
        "auto" => pair(PolicyMode::AutoEdits, "auto"),
        "plan" => pair(PolicyMode::PlanOnly, "plan"),
        "bypassPermissions" => pair(PolicyMode::Trust, "bypassPermissions"),
        // Neutral strings.
        "ask" => pair(PolicyMode::Ask, "default"),
        "auto_edits" => pair(PolicyMode::AutoEdits, "acceptEdits"),
        "plan_only" => pair(PolicyMode::PlanOnly, "plan"),
        "trust" => pair(PolicyMode::Trust, "bypassPermissions"),
        _ => None,
    }
}

/// Neutral name of a mode, identical to the nexus serde form of [`PolicyMode`].
pub fn neutral_name(mode: PolicyMode) -> &'static str {
    match mode {
        PolicyMode::PlanOnly => "plan_only",
        PolicyMode::Ask => "ask",
        PolicyMode::AutoEdits => "auto_edits",
        PolicyMode::Trust => "trust",
        // `PolicyMode` is non-exhaustive: an unknown mode reads as the asking one.
        _ => "ask",
    }
}

/// Canonical legacy name of a mode.
pub fn legacy_name(mode: PolicyMode) -> &'static str {
    match mode {
        PolicyMode::Ask => "default",
        PolicyMode::AutoEdits => "acceptEdits",
        PolicyMode::PlanOnly => "plan",
        PolicyMode::Trust => "bypassPermissions",
        // `PolicyMode` is non-exhaustive: an unknown mode reads as the asking one.
        _ => "default",
    }
}

/// The legacy string to hand to the Claude CLI for any accepted input.
///
/// A legacy string stays itself; a neutral one becomes its canonical legacy name.
pub fn to_legacy(input: &str) -> Option<&'static str> {
    parse_mode(input).map(|pair| pair.legacy)
}

/// Builds the neutral tool policy from a mode string and Claude-style patterns.
///
/// `None` when the mode is unknown or when a pattern is malformed (nexus refuses
/// to skip a malformed pattern: a dropped `deny` entry would widen the policy).
/// A legacy mode string is kept as `native_mode` so the Claude Code adapter can
/// apply it exactly; a neutral input leaves `native_mode` empty.
pub fn tool_policy(mode: &str, allowed: &[String], disallowed: &[String]) -> Option<ToolPolicy> {
    let pair = parse_mode(mode)?;
    let mut policy = ToolPolicy::from_patterns(pair.neutral, allowed, disallowed).ok()?;
    policy.native_mode = legacy_static(mode).map(|legacy| legacy.to_string());
    Some(policy)
}

/// JSON carried by `system_init.tool_policy`: the nexus serde form, unchanged.
pub fn wire_tool_policy(policy: &ToolPolicy) -> serde_json::Value {
    serde_json::to_value(policy).unwrap_or(serde_json::Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_claude::agent::{PolicyDecision, ToolCategory};

    const TABLE: [(&str, PolicyMode, &str); 11] = [
        ("default", PolicyMode::Ask, "default"),
        ("acceptEdits", PolicyMode::AutoEdits, "acceptEdits"),
        ("plan", PolicyMode::PlanOnly, "plan"),
        ("bypassPermissions", PolicyMode::Trust, "bypassPermissions"),
        ("auto", PolicyMode::AutoEdits, "auto"),
        ("dontAsk", PolicyMode::Ask, "dontAsk"),
        ("manual", PolicyMode::Ask, "manual"),
        ("ask", PolicyMode::Ask, "default"),
        ("auto_edits", PolicyMode::AutoEdits, "acceptEdits"),
        ("plan_only", PolicyMode::PlanOnly, "plan"),
        ("trust", PolicyMode::Trust, "bypassPermissions"),
    ];

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| item.to_string()).collect()
    }

    #[test]
    fn the_eleven_values_parse_and_round_trip() {
        for (input, neutral, legacy) in TABLE {
            let pair = parse_mode(input).unwrap();
            assert_eq!(pair.neutral, neutral, "{input}");
            assert_eq!(pair.legacy, legacy, "{input}");
            // Both emitted forms parse back to the same neutral mode.
            assert_eq!(
                parse_mode(neutral_name(pair.neutral)).unwrap().neutral,
                neutral
            );
            assert_eq!(parse_mode(pair.legacy).unwrap().neutral, neutral);
            assert_eq!(
                parse_mode(legacy_name(pair.neutral)).unwrap().neutral,
                neutral
            );
        }
        assert_eq!(LEGACY_MODES.len() + NEUTRAL_MODES.len(), TABLE.len());
        for mode in LEGACY_MODES.iter().chain(NEUTRAL_MODES.iter()) {
            assert!(parse_mode(mode).is_some(), "{mode}");
        }
    }

    #[test]
    fn neutral_names_match_the_nexus_serialisation() {
        for mode in [
            PolicyMode::PlanOnly,
            PolicyMode::Ask,
            PolicyMode::AutoEdits,
            PolicyMode::Trust,
        ] {
            assert_eq!(
                serde_json::to_value(mode).unwrap(),
                serde_json::json!(neutral_name(mode))
            );
            assert!(NEUTRAL_MODES.contains(&neutral_name(mode)));
        }
    }

    #[test]
    fn legacy_names_are_the_canonical_claude_strings() {
        assert_eq!(legacy_name(PolicyMode::Ask), "default");
        assert_eq!(legacy_name(PolicyMode::AutoEdits), "acceptEdits");
        assert_eq!(legacy_name(PolicyMode::PlanOnly), "plan");
        assert_eq!(legacy_name(PolicyMode::Trust), "bypassPermissions");
    }

    #[test]
    fn to_legacy_keeps_legacy_strings_and_maps_neutral_ones() {
        // Legacy strings stay themselves, including the ones a neutral mode cannot express.
        for mode in LEGACY_MODES {
            assert_eq!(to_legacy(mode), Some(mode));
        }
        assert_eq!(to_legacy("ask"), Some("default"));
        assert_eq!(to_legacy("auto_edits"), Some("acceptEdits"));
        assert_eq!(to_legacy("plan_only"), Some("plan"));
        assert_eq!(to_legacy("trust"), Some("bypassPermissions"));
    }

    #[test]
    fn an_unknown_mode_is_none() {
        for bad in ["", "yolo", "Ask", "accept_edits", "DEFAULT", " plan"] {
            assert!(parse_mode(bad).is_none(), "{bad:?}");
            assert!(to_legacy(bad).is_none(), "{bad:?}");
            assert!(tool_policy(bad, &[], &[]).is_none(), "{bad:?}");
        }
    }

    #[test]
    fn tool_policy_decides_allow_and_deny_as_configured() {
        let allowed = strings(&["mcp__project-orchestrator__*"]);
        let disallowed = strings(&["Bash(rm -rf *)"]);
        let policy = tool_policy("ask", &allowed, &disallowed).unwrap();
        assert_eq!(policy.mode, PolicyMode::Ask);
        assert_eq!(policy.native_mode, None);

        // Approved in advance by the allow list.
        assert_eq!(
            policy.decide("mcp__project-orchestrator__task", None, ToolCategory::Mcp),
            PolicyDecision::Allow
        );
        // Deny wins whatever the mode.
        assert_eq!(
            policy.decide("Bash", Some("rm -rf /tmp/x"), ToolCategory::Command),
            PolicyDecision::Deny
        );
        // Unknown argument against an argument-scoped deny: fail closed.
        assert_eq!(
            policy.decide("Bash", None, ToolCategory::Command),
            PolicyDecision::Deny
        );
        // Neither allowed nor denied: the `ask` mode asks for a command...
        assert_eq!(
            policy.decide("Bash", Some("ls"), ToolCategory::Command),
            PolicyDecision::Ask
        );
        // ...and for an MCP tool outside the allow list, but never for a read.
        assert_eq!(
            policy.decide("mcp__other__tool", None, ToolCategory::Mcp),
            PolicyDecision::Ask
        );
        assert_eq!(
            policy.decide("Read", Some("/tmp/a"), ToolCategory::Read),
            PolicyDecision::Allow
        );
    }

    #[test]
    fn deny_still_wins_in_trust_mode() {
        let allowed = strings(&["mcp__project-orchestrator__*"]);
        let disallowed = strings(&["Bash(rm -rf *)"]);
        let policy = tool_policy("bypassPermissions", &allowed, &disallowed).unwrap();
        assert_eq!(policy.mode, PolicyMode::Trust);
        assert_eq!(
            policy.decide("Bash", Some("rm -rf /"), ToolCategory::Command),
            PolicyDecision::Deny
        );
        assert_eq!(
            policy.decide("Bash", Some("ls"), ToolCategory::Command),
            PolicyDecision::Allow
        );
    }

    #[test]
    fn a_legacy_mode_is_kept_as_native_mode() {
        let policy = tool_policy("dontAsk", &[], &[]).unwrap();
        assert_eq!(policy.mode, PolicyMode::Ask);
        assert_eq!(policy.native_mode.as_deref(), Some("dontAsk"));
    }

    #[test]
    fn a_malformed_pattern_is_none_never_skipped() {
        assert!(tool_policy("ask", &strings(&["Read"]), &strings(&["Bash("])).is_none());
        assert!(tool_policy("ask", &strings(&["Bash()"]), &[]).is_none());
    }

    #[test]
    fn wire_tool_policy_is_the_nexus_shape() {
        let allowed = strings(&["mcp__project-orchestrator__*"]);
        let disallowed = strings(&["Bash(rm -rf *)"]);

        let neutral = tool_policy("ask", &allowed, &disallowed).unwrap();
        assert_eq!(
            wire_tool_policy(&neutral),
            serde_json::json!({
                "mode": "ask",
                "allow": ["mcp__project-orchestrator__*"],
                "deny": ["Bash(rm -rf *)"],
            })
        );

        let legacy = tool_policy("acceptEdits", &allowed, &disallowed).unwrap();
        assert_eq!(
            wire_tool_policy(&legacy),
            serde_json::json!({
                "mode": "auto_edits",
                "native_mode": "acceptEdits",
                "allow": ["mcp__project-orchestrator__*"],
                "deny": ["Bash(rm -rf *)"],
            })
        );

        // The wire form reads back into the same policy.
        let back: ToolPolicy = serde_json::from_value(wire_tool_policy(&legacy)).unwrap();
        assert_eq!(back, legacy);
    }
}
