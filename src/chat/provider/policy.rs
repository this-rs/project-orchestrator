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

/// What a session may do to the world, decided by the harness for the whole
/// life of the session (open AND every resume).
///
/// `ReadOnly` is not a permission mode: modes decide who is asked, and `Trust`
/// asks nobody. It is a deny list computed here, merged into the session's
/// `disallowed_tools`; a deny entry wins over the mode (Trust included) and over
/// the allow list, both in the nexus policy ([`ToolPolicy::decide`]) and in the
/// Claude CLI (`--disallowedTools`). Nothing in the prompt is involved: a tool
/// the prompt never mentions is refused all the same.
///
/// Trust: `trust` (`bypassPermissions`) only means "ask nobody". A read-only
/// session in `trust` still has every tool of the deny list refused, because
/// `deny` is checked before the mode and before `allow`. `plan_only` is stricter
/// in another way (it refuses every MCP tool, reads included), which is why
/// read-only is a deny list on top of a mode and not a mode.
///
/// Where the flag comes from: the request (`access`), or the default of a session
/// that runs in a neutral directory ([`SessionAccess::for_open`]). It is persisted
/// on the session and a resume can only keep it or narrow it
/// ([`SessionAccess::for_resume`]).
///
/// Two layers refuse. This one, by tool NAME (native tools and the mega-tools no
/// read-only session may see). And the MCP server, by ACTION: the session token
/// carries the `read_only` tool profile (`ToolProfile::ReadOnly`), under which
/// a mixed mega-tool (`task`, `note`, ...) only runs its read actions.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum SessionAccess {
    /// The configured policy, untouched.
    #[default]
    Normal,
    /// No tool that writes, runs a command, spawns an agent or administers the server.
    ReadOnly,
}

/// Claude Code tools of the `Edit` category (nexus `tool_category`): they write files.
const READ_ONLY_DENIED_FILE_TOOLS: &[&str] = &["Edit", "Write", "MultiEdit", "NotebookEdit"];

/// Claude Code tools of the `Command` category: a shell command cannot be told
/// read from write from outside, so the whole category is refused.
const READ_ONLY_DENIED_COMMAND_TOOLS: &[&str] = &["Bash", "BashOutput", "KillShell"];

/// Tools that spawn a sub-agent (category `Agent`): the child would not be read-only.
const READ_ONLY_DENIED_AGENT_TOOLS: &[&str] = &["Task", "Agent"];

/// Name of the project-orchestrator MCP server as the CLI spells it.
const PO_MCP_PREFIX: &str = "mcp__project-orchestrator__";

impl SessionAccess {
    /// Wire and storage form: `normal` | `read_only`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::ReadOnly => "read_only",
        }
    }

    /// Reads a stored value: absent or empty is `normal` (a session written before the
    /// field existed). A value this server does not know was written to narrow the
    /// session: it is never read as the wide access.
    pub fn parse_stored(raw: &str) -> Self {
        match raw {
            "" | "normal" => Self::Normal,
            _ => Self::ReadOnly,
        }
    }

    /// For `skip_serializing_if`: `normal` is left off the wire.
    pub fn is_normal(&self) -> bool {
        *self == Self::Normal
    }

    /// The tool profile the session token carries for this access (`None`: the
    /// profile the provider and mode grant, unchanged).
    pub fn tool_profile(self) -> Option<&'static str> {
        match self {
            Self::Normal => None,
            Self::ReadOnly => Some(crate::auth::tool_profile::READ_ONLY),
        }
    }

    /// The narrower of two accesses.
    pub fn narrowest(self, other: Self) -> Self {
        if self == Self::ReadOnly || other == Self::ReadOnly {
            Self::ReadOnly
        } else {
            Self::Normal
        }
    }

    /// Access of a session being OPENED. `requested` is what the client asked
    /// (`None`: nothing). A session that runs in the neutral directory of the host
    /// (it belongs to no project) is read-only unless the client explicitly asks for
    /// `normal`; a project session is `normal` unless the client asks for `read_only`.
    pub fn for_open(requested: Option<Self>, neutral: bool) -> Self {
        requested.unwrap_or(if neutral {
            Self::ReadOnly
        } else {
            Self::Normal
        })
    }

    /// Access of a session being RESUMED: the stored one, narrowed by what the
    /// resuming call asks. A resume never widens (`read_only` stays `read_only`
    /// whatever the request says).
    pub fn for_resume(stored: Self, requested: Option<Self>) -> Self {
        stored.narrowest(requested.unwrap_or_default())
    }

    /// The tool patterns this access refuses, in the syntax of `disallowed_tools`.
    ///
    /// For the project-orchestrator MCP server: every mega-tool the restricted
    /// profile withholds (admin, vault, protocol, sharing, ...: [`ToolProfile`]).
    /// The list is computed from the real tool list, so a tool added tomorrow is
    /// refused until the profile admits it. The other mega-tools (task, plan, note,
    /// chat, ...) mix reads and writes under one name selected by an `action`
    /// argument, which a tool pattern cannot see: the action-level refusal is the
    /// `read_only` tool profile of the session token (see the type doc).
    pub fn denied_tools(self) -> Vec<String> {
        match self {
            Self::Normal => Vec::new(),
            Self::ReadOnly => {
                let native = READ_ONLY_DENIED_FILE_TOOLS
                    .iter()
                    .chain(READ_ONLY_DENIED_COMMAND_TOOLS)
                    .chain(READ_ONLY_DENIED_AGENT_TOOLS)
                    .map(|name| (*name).to_string());
                let mcp = crate::mcp::tools::all_tools()
                    .into_iter()
                    .map(|tool| tool.name)
                    .filter(|name| {
                        !crate::auth::tool_profile::ToolProfile::Restricted.allows_tool(name)
                    })
                    .map(|name| format!("{PO_MCP_PREFIX}{name}"));
                native.chain(mcp).collect()
            }
        }
    }

    /// `disallowed` plus what this access refuses, without duplicates, the
    /// configured entries first.
    pub fn merge_disallowed(self, disallowed: &[String]) -> Vec<String> {
        let mut merged = disallowed.to_vec();
        for denied in self.denied_tools() {
            if !merged.contains(&denied) {
                merged.push(denied);
            }
        }
        merged
    }
}

/// [`tool_policy`] for a session of the given access: the read-only deny list
/// is merged into `disallowed` before the patterns are parsed.
pub fn tool_policy_with_access(
    mode: &str,
    allowed: &[String],
    disallowed: &[String],
    access: SessionAccess,
) -> Option<ToolPolicy> {
    tool_policy(mode, allowed, &access.merge_disallowed(disallowed))
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

    const WRITE_TOOLS: [(&str, ToolCategory, Option<&str>); 8] = [
        ("Edit", ToolCategory::Edit, Some("/repo/a.rs")),
        ("Write", ToolCategory::Edit, Some("/repo/a.rs")),
        ("MultiEdit", ToolCategory::Edit, Some("/repo/a.rs")),
        ("NotebookEdit", ToolCategory::Edit, Some("/repo/n.ipynb")),
        ("Bash", ToolCategory::Command, Some("ls")),
        ("BashOutput", ToolCategory::Command, None),
        ("Task", ToolCategory::Agent, None),
        ("mcp__project-orchestrator__admin", ToolCategory::Mcp, None),
    ];

    const MODES: [&str; 4] = ["ask", "auto_edits", "plan_only", "trust"];

    fn po_allowed() -> Vec<String> {
        strings(&["mcp__project-orchestrator__*", "Edit", "Write", "Bash"])
    }

    /// Replay of the audit: without a read-only access, what each mode does to the write tools.
    #[test]
    fn without_read_only_only_plan_only_refuses_the_write_tools_and_trust_allows_them() {
        for mode in MODES {
            let policy =
                tool_policy_with_access(mode, &po_allowed(), &[], SessionAccess::Normal).unwrap();
            for (tool, category, arg) in WRITE_TOOLS {
                let decision = policy.decide(tool, arg, category);
                match mode {
                    "plan_only" => assert_eq!(decision, PolicyDecision::Deny, "{mode} {tool}"),
                    "trust" => assert_eq!(decision, PolicyDecision::Allow, "{mode} {tool}"),
                    _ => assert_ne!(decision, PolicyDecision::Deny, "{mode} {tool}"),
                }
            }
        }
    }

    #[test]
    fn a_read_only_session_refuses_every_write_tool_in_every_mode_trust_included() {
        for mode in MODES.iter().chain(LEGACY_MODES.iter()) {
            let policy =
                tool_policy_with_access(mode, &po_allowed(), &[], SessionAccess::ReadOnly).unwrap();
            for (tool, category, arg) in WRITE_TOOLS {
                assert_eq!(
                    policy.decide(tool, arg, category),
                    PolicyDecision::Deny,
                    "{mode} {tool}"
                );
            }
            // An argument-less call is refused too (fail closed).
            assert_eq!(
                policy.decide("Edit", None, ToolCategory::Edit),
                PolicyDecision::Deny
            );
        }
    }

    #[test]
    fn a_read_only_session_still_reads_and_keeps_the_mutating_free_mcp_tools() {
        let policy =
            tool_policy_with_access("trust", &po_allowed(), &[], SessionAccess::ReadOnly).unwrap();
        assert_eq!(
            policy.decide("Read", Some("/repo/a.rs"), ToolCategory::Read),
            PolicyDecision::Allow
        );
        assert_eq!(
            policy.decide("Grep", None, ToolCategory::Search),
            PolicyDecision::Allow
        );
        assert_eq!(
            policy.decide("mcp__project-orchestrator__code", None, ToolCategory::Mcp),
            PolicyDecision::Allow
        );
    }

    #[test]
    fn the_read_only_deny_list_is_computed_from_the_tool_list_and_keeps_configured_entries() {
        let denied = SessionAccess::ReadOnly.denied_tools();
        for expected in [
            "Edit",
            "Write",
            "MultiEdit",
            "NotebookEdit",
            "Bash",
            "mcp__project-orchestrator__admin",
            "mcp__project-orchestrator__vault",
        ] {
            assert!(denied.iter().any(|d| d == expected), "{expected}");
        }
        assert!(!denied
            .iter()
            .any(|d| d == "mcp__project-orchestrator__code"));
        let configured = strings(&["Bash(rm -rf *)", "Edit"]);
        let merged = SessionAccess::ReadOnly.merge_disallowed(&configured);
        assert_eq!(&merged[..2], &configured[..]);
        assert_eq!(merged.iter().filter(|d| *d == "Edit").count(), 1);
    }

    #[test]
    fn the_access_has_a_wire_form_and_a_normal_default() {
        assert_eq!(SessionAccess::default(), SessionAccess::Normal);
        for access in [SessionAccess::Normal, SessionAccess::ReadOnly] {
            assert_eq!(SessionAccess::parse_stored(access.as_str()), access);
            let json = serde_json::to_value(access).unwrap();
            assert_eq!(json, access.as_str());
            assert_eq!(
                serde_json::from_value::<SessionAccess>(json).unwrap(),
                access
            );
        }
        // Absent = normal (a session written before the field existed); a value this
        // server does not know is never read as the wide access.
        assert_eq!(SessionAccess::parse_stored(""), SessionAccess::Normal);
        assert_eq!(
            SessionAccess::parse_stored("locked_down"),
            SessionAccess::ReadOnly
        );
        assert!(SessionAccess::Normal.is_normal() && !SessionAccess::ReadOnly.is_normal());
        assert_eq!(SessionAccess::Normal.tool_profile(), None);
        assert_eq!(
            SessionAccess::ReadOnly.tool_profile(),
            Some(crate::auth::tool_profile::READ_ONLY)
        );
    }

    #[test]
    fn a_neutral_session_is_read_only_by_default_and_a_project_session_is_normal() {
        use SessionAccess::{Normal, ReadOnly};
        assert_eq!(SessionAccess::for_open(None, true), ReadOnly);
        assert_eq!(SessionAccess::for_open(None, false), Normal);
        // A project session can be opened read-only on request.
        assert_eq!(SessionAccess::for_open(Some(ReadOnly), false), ReadOnly);
        assert_eq!(SessionAccess::for_open(Some(ReadOnly), true), ReadOnly);
        // A client that explicitly asks for normal on a neutral session gets it (the
        // read-only is a default, not a lock).
        assert_eq!(SessionAccess::for_open(Some(Normal), true), Normal);
    }

    #[test]
    fn a_resume_never_widens_the_access() {
        use SessionAccess::{Normal, ReadOnly};
        for requested in [None, Some(Normal), Some(ReadOnly)] {
            assert_eq!(
                SessionAccess::for_resume(ReadOnly, requested),
                ReadOnly,
                "read_only stays read_only whatever is asked ({requested:?})"
            );
        }
        assert_eq!(SessionAccess::for_resume(Normal, None), Normal);
        assert_eq!(SessionAccess::for_resume(Normal, Some(Normal)), Normal);
        // A resume may narrow.
        assert_eq!(SessionAccess::for_resume(Normal, Some(ReadOnly)), ReadOnly);
    }

    #[test]
    fn the_mixed_mega_tools_are_not_denied_by_name_but_cut_at_the_action() {
        let denied = SessionAccess::ReadOnly.denied_tools();
        for mixed in ["chat", "task", "note", "plan", "step"] {
            let name = format!("mcp__project-orchestrator__{mixed}");
            assert!(!denied.contains(&name), "{name}");
        }
    }

    #[test]
    fn the_normal_access_changes_nothing() {
        assert_eq!(SessionAccess::default(), SessionAccess::Normal);
        let disallowed = strings(&["Bash(rm -rf *)"]);
        assert_eq!(
            SessionAccess::Normal.merge_disallowed(&disallowed),
            disallowed
        );
        for mode in MODES {
            let plain = tool_policy(mode, &po_allowed(), &disallowed).unwrap();
            let with =
                tool_policy_with_access(mode, &po_allowed(), &disallowed, SessionAccess::Normal)
                    .unwrap();
            assert_eq!(plain, with, "{mode}");
        }
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
