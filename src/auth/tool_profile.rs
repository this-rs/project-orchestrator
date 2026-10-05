//! MCP tool profile of a chat session.
//!
//! Every session used to see all the mega-tools, including the ones that open
//! other sessions (`chat send_message`, `plan run`, `plan delegate_task`) or
//! reconfigure the server (`admin`, `mcp_federation`, `lifecycle_hook`). For a
//! model that only has MCP tools, that list IS its reach: "tools only through
//! MCP" is not a sandbox while the list contains a way out.
//!
//! The profile is SIGNED into the session token (`tools:<profile>` in the
//! scope, see [`super::jwt::AgentSessionBinding`]). Two places read it:
//!
//! * the MCP subprocess, to shape `tools/list` and refuse `tools/call` — that
//!   is presentation: the subprocess runs under the agent's control;
//! * the REST middleware, through [`route_forbidden`] — that is the boundary:
//!   whatever the agent does with its token (`curl` included), the routes
//!   behind the withheld tools answer 403.
//!
//! No profile in the token = the full profile (today's behaviour). An unknown
//! profile name = the restricted one: never an allow on something we do not
//! recognise.

use crate::mcp::protocol::ToolDefinition;
use axum::http::Method;

/// Name of the restricted profile as written in the token scope.
pub const RESTRICTED: &str = "restricted";
/// Name of the full profile (also what an absent profile means).
pub const FULL: &str = "full";

/// Which mega-tools a session may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolProfile {
    /// Every tool, every action.
    Full,
    /// No tool that opens a session, runs a plan or reconfigures the server.
    /// The default for any provider other than Claude Code (decision A35).
    Restricted,
}

/// Tools the restricted profile does not see at all.
const RESTRICTED_TOOLS: &[&str] = &["admin", "mcp_federation", "lifecycle_hook"];

/// `(tool, action)` pairs the restricted profile does not see.
const RESTRICTED_ACTIONS: &[(&str, &str)] = &[
    ("chat", "send_message"),
    ("plan", "run"),
    ("plan", "delegate_task"),
    ("plan", "auto_pr"),
    ("plan", "add_trigger"),
    ("plan", "enable_trigger"),
];

impl ToolProfile {
    /// Profile named in a token. Absent → full; unknown → restricted.
    pub fn from_name(name: Option<&str>) -> Self {
        match name {
            None | Some(FULL) => Self::Full,
            Some(_) => Self::Restricted,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Full => FULL,
            Self::Restricted => RESTRICTED,
        }
    }

    /// Profile carried by a session token, WITHOUT verifying its signature.
    /// Only for the MCP subprocess, which holds the token but not the key and
    /// uses the answer to shape its tool list. A token that cannot be read
    /// yields the restricted profile.
    pub fn from_unverified_token(token: &str) -> Self {
        let Ok(claims) =
            jsonwebtoken::dangerous::insecure_decode::<crate::auth::jwt::Claims>(token)
                .map(|data| data.claims)
        else {
            return Self::Restricted;
        };
        if !claims.is_agent_session() {
            return Self::Full;
        }
        match crate::auth::jwt::agent_session_binding(&claims) {
            Some(binding) => Self::from_name(binding.tool_profile.as_deref()),
            // An agent token without a session carries no profile: full, as before.
            None => Self::Full,
        }
    }

    /// Whether the tool is visible at all.
    pub fn allows_tool(self, tool: &str) -> bool {
        match self {
            Self::Full => true,
            Self::Restricted => !RESTRICTED_TOOLS.contains(&tool),
        }
    }

    /// Whether this action of this tool may be called.
    pub fn allows_action(self, tool: &str, action: &str) -> bool {
        self.allows_tool(tool)
            && match self {
                Self::Full => true,
                Self::Restricted => !RESTRICTED_ACTIONS.contains(&(tool, action)),
            }
    }

    /// The tool list this profile sees: withheld tools removed, withheld
    /// actions removed from the `action` enum of the tools that stay.
    pub fn filter_tools(self, tools: Vec<ToolDefinition>) -> Vec<ToolDefinition> {
        if self == Self::Full {
            return tools;
        }
        tools
            .into_iter()
            .filter(|t| self.allows_tool(&t.name))
            .map(|mut t| {
                let name = t.name.clone();
                if let Some(actions) = t
                    .input_schema
                    .properties
                    .as_mut()
                    .and_then(|p| p.get_mut("action"))
                    .and_then(|a| a.get_mut("enum"))
                    .and_then(|e| e.as_array_mut())
                {
                    actions.retain(|a| a.as_str().is_none_or(|a| self.allows_action(&name, a)));
                }
                t
            })
            .collect()
    }

    /// The REST boundary: whether a token with this profile must be refused
    /// on `method path`. Covers the routes behind every withheld tool/action.
    pub fn route_forbidden(self, method: &Method, path: &str) -> bool {
        if self == Self::Full {
            return false;
        }
        let read = matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS);
        let under = |prefix: &str| {
            path == prefix
                || path
                    .strip_prefix(prefix)
                    .is_some_and(|rest| rest.starts_with('/'))
        };
        // Whole tools: admin / mcp_federation / lifecycle_hook (reads included —
        // the tool is not in the profile at all).
        if under("/api/admin") || under("/api/mcp-federation") || under("/api/lifecycle-hooks") {
            return true;
        }
        if read {
            return false;
        }
        // chat send_message: opening or feeding a session.
        if path == "/api/chat/sessions" {
            return true;
        }
        // plan run / auto_pr / delegate_task / triggers.
        if let Some(rest) = path.strip_prefix("/api/plans/") {
            let mut segments = rest.split('/');
            let _plan_id = segments.next();
            return match segments.next() {
                Some("run") | Some("triggers") => true,
                Some("tasks") => rest.ends_with("/delegate"),
                _ => false,
            };
        }
        under("/api/triggers")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::tools::all_tools;

    fn actions_of(tools: &[ToolDefinition], name: &str) -> Vec<String> {
        tools
            .iter()
            .find(|t| t.name == name)
            .and_then(|t| t.input_schema.properties.as_ref())
            .and_then(|p| p.get("action"))
            .and_then(|a| a.get("enum"))
            .and_then(|e| e.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn the_full_profile_sees_every_tool_unchanged() {
        let all = all_tools();
        let seen = ToolProfile::Full.filter_tools(all_tools());
        assert_eq!(seen.len(), all.len());
        assert_eq!(actions_of(&seen, "plan"), actions_of(&all, "plan"));
    }

    #[test]
    fn the_restricted_profile_has_no_way_to_open_a_session_or_run_a_plan() {
        let seen = ToolProfile::Restricted.filter_tools(all_tools());
        let names: Vec<_> = seen.iter().map(|t| t.name.as_str()).collect();
        for withheld in RESTRICTED_TOOLS {
            assert!(!names.contains(withheld), "{withheld} must be withheld");
        }
        assert!(names.contains(&"plan") && names.contains(&"chat") && names.contains(&"note"));

        let plan = actions_of(&seen, "plan");
        assert!(!plan.iter().any(|a| a == "run" || a == "delegate_task"));
        assert!(
            plan.iter().any(|a| a == "get"),
            "ordinary plan actions stay"
        );
        let chat = actions_of(&seen, "chat");
        assert!(!chat.is_empty(), "chat keeps its read actions");
        assert!(!chat.iter().any(|a| a == "send_message"));
    }

    #[test]
    fn every_withheld_name_exists_in_the_real_tool_list() {
        // A typo here would silently withhold nothing.
        let all = all_tools();
        for tool in RESTRICTED_TOOLS {
            assert!(all.iter().any(|t| t.name == *tool), "unknown tool {tool}");
        }
        for (tool, action) in RESTRICTED_ACTIONS {
            assert!(
                actions_of(&all, tool).iter().any(|a| a == action),
                "unknown action {tool}.{action}"
            );
        }
    }

    #[test]
    fn calls_are_decided_by_tool_and_action() {
        let r = ToolProfile::Restricted;
        assert!(!r.allows_action("plan", "run"));
        assert!(!r.allows_action("plan", "delegate_task"));
        assert!(!r.allows_action("chat", "send_message"));
        assert!(!r.allows_action("admin", "anything"));
        assert!(r.allows_action("plan", "get"));
        assert!(r.allows_action("note", "create"));
        assert!(ToolProfile::Full.allows_action("plan", "run"));
    }

    #[test]
    fn an_absent_profile_is_full_and_an_unknown_one_is_restricted() {
        assert_eq!(ToolProfile::from_name(None), ToolProfile::Full);
        assert_eq!(ToolProfile::from_name(Some("full")), ToolProfile::Full);
        assert_eq!(
            ToolProfile::from_name(Some("restricted")),
            ToolProfile::Restricted
        );
        assert_eq!(
            ToolProfile::from_name(Some("superuser")),
            ToolProfile::Restricted
        );
    }

    #[test]
    fn the_profile_is_read_from_the_token_not_from_the_environment() {
        let human = crate::auth::jwt::Claims::service_account("s");
        let secret = "test-secret-key-minimum-32-chars!!";
        let mint = |profile: Option<&str>| {
            let binding = crate::auth::jwt::AgentSessionBinding {
                session_id: "sess".into(),
                ceiling: None,
                tool_profile: profile.map(str::to_string),
            };
            crate::auth::jwt::generate_session_token(&human, Some(&binding), secret, 60)
                .unwrap()
                .0
        };
        assert_eq!(
            ToolProfile::from_unverified_token(&mint(Some("restricted"))),
            ToolProfile::Restricted
        );
        assert_eq!(
            ToolProfile::from_unverified_token(&mint(None)),
            ToolProfile::Full
        );
        assert_eq!(
            ToolProfile::from_unverified_token("garbage"),
            ToolProfile::Restricted,
            "an unreadable token never yields the full profile"
        );
    }

    #[test]
    fn the_rest_routes_behind_withheld_tools_are_closed() {
        let r = ToolProfile::Restricted;
        for (method, path) in [
            (Method::POST, "/api/chat/sessions"),
            (Method::POST, "/api/plans/abc/run"),
            (Method::POST, "/api/plans/abc/run/auto-pr"),
            (Method::POST, "/api/plans/abc/tasks/def/delegate"),
            (Method::POST, "/api/plans/abc/triggers"),
            (Method::POST, "/api/admin/backfill-synapses"),
            (Method::GET, "/api/admin/backfill-embeddings/status"),
            (Method::POST, "/api/mcp-federation/servers"),
            (Method::DELETE, "/api/lifecycle-hooks/h1"),
        ] {
            assert!(
                r.route_forbidden(&method, path),
                "{method} {path} must be closed"
            );
            assert!(
                !ToolProfile::Full.route_forbidden(&method, path),
                "the full profile keeps {method} {path}"
            );
        }
        for (method, path) in [
            (Method::GET, "/api/chat/sessions"),
            (Method::GET, "/api/plans/abc/run/status"),
            (Method::GET, "/api/plans/abc"),
            (Method::POST, "/api/plans"),
            (Method::POST, "/api/plans/abc/tasks"),
            (Method::POST, "/api/notes"),
            (Method::PATCH, "/api/tasks/t1"),
        ] {
            assert!(
                !r.route_forbidden(&method, path),
                "{method} {path} must stay open"
            );
        }
    }
}
