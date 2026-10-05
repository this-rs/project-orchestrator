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

/// The ONLY tools the restricted profile sees. An allow-list: a tool added
/// tomorrow is withheld until someone decides it is safe for a third-party model.
/// Withheld on purpose: `vault` (secrets), `admin`, `mcp_federation`,
/// `lifecycle_hook`, `protocol` (starts agents), `sharing` (sends data out),
/// `environment`, `neural_routing`, `trajectory`.
const RESTRICTED_ALLOWED_TOOLS: &[&str] = &[
    "project",
    "plan",
    "task",
    "step",
    "decision",
    "constraint",
    "release",
    "milestone",
    "commit",
    "note",
    "workspace",
    "workspace_milestone",
    "resource",
    "component",
    "chat",
    "feature_graph",
    "code",
    "episode",
    "reasoning",
    "analysis_profile",
    "skill",
    "persona",
];

/// `(tool, action)` pairs the restricted profile does not see.
const RESTRICTED_ACTIONS: &[(&str, &str)] = &[
    ("chat", "send_message"),
    ("plan", "run"),
    ("plan", "delegate_task"),
    ("plan", "auto_pr"),
    ("plan", "add_trigger"),
    ("plan", "enable_trigger"),
];

/// How a REST route is treated by the restricted profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteClass {
    /// Behind a tool the profile keeps (the method-specific withholdings still apply).
    Allowed,
    /// Behind a tool the profile withholds, or not behind any tool a model uses.
    Closed,
}

/// Route prefixes the restricted profile may use: exactly what the MCP handlers of
/// the KEPT tools call (`src/mcp/handlers.rs`). `*` matches one path segment.
/// A route is matched by its LONGEST prefix across both lists, so a closed
/// sub-route wins over its allowed parent.
const ALLOWED_ROUTE_PREFIXES: &[&str] = &[
    "/api/analysis-profiles",
    "/api/chat",
    "/api/code",
    "/api/commits",
    "/api/components",
    "/api/constraints",
    "/api/decisions",
    "/api/entities",
    "/api/episodes",
    "/api/feature-graphs",
    "/api/files",
    "/api/milestones",
    "/api/notes",
    "/api/personas",
    "/api/plans",
    "/api/projects",
    "/api/reason",
    "/api/releases",
    "/api/resources",
    "/api/runs",
    "/api/skills",
    "/api/steps",
    "/api/tasks",
    "/api/workspace-milestones",
    "/api/workspaces",
];

/// Route prefixes the restricted profile may NOT use, reads included: the routes
/// of every withheld tool (admin, mcp_federation, lifecycle_hook, vault,
/// protocol, sharing, environment, neural_routing, trajectory) and the ones no
/// kept tool calls (server maintenance, search index, hooks, registry, ...).
const CLOSED_ROUTE_PREFIXES: &[&str] = &[
    // withheld tools
    "/api/admin",
    "/api/mcp-federation",
    "/api/lifecycle-hooks",
    "/api/vault",
    "/api/protocols",
    "/api/environments",
    "/api/deployments",
    "/api/neural-routing",
    "/api/trajectories",
    "/api/triggers",
    "/api/event-triggers",
    // server maintenance and indexes (the `admin` tool)
    "/api/sync",
    "/api/watch",
    "/api/meilisearch",
    // self-update: install and restart replace and relaunch the server binary,
    // and even `check` talks to the release server on the caller's behalf
    "/api/update",
    // sharing sends project data out; deployments hang under projects
    "/api/projects/*/sharing",
    "/api/projects/*/environments",
    "/api/projects/*/deployment-matrix",
    "/api/notes/*/sharing",
    // memory maintenance (admin)
    "/api/notes/neurons",
    "/api/notes/consolidate-memory",
    "/api/notes/update-staleness",
    "/api/notes/update-energy",
    // no kept tool calls these
    "/api/agents",
    "/api/alerts",
    "/api/attention",
    "/api/documents",
    "/api/feedback",
    "/api/graph",
    "/api/hooks",
    "/api/progress",
    "/api/reactor",
    "/api/registry",
    "/api/rfcs",
    "/api/setup-status",
    "/api/version",
    "/api/wake",
    "/api/webhooks",
];

fn prefix_len(pattern: &str, path: &str) -> Option<usize> {
    let pat: Vec<&str> = pattern.trim_start_matches('/').split('/').collect();
    let segs: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    if segs.len() < pat.len() {
        return None;
    }
    pat.iter()
        .zip(&segs)
        .all(|(p, s)| *p == "*" || p == s)
        .then_some(pat.len())
}

/// The class of a REST route for the restricted profile: the LONGEST matching
/// prefix of the two lists above (a tie is closed). `None` = classified by
/// nobody: the profile treats it as closed, and the route-table test fails
/// until someone decides.
pub fn classify_route(path: &str) -> Option<RouteClass> {
    let best = |list: &[&str]| list.iter().filter_map(|p| prefix_len(p, path)).max();
    match (best(ALLOWED_ROUTE_PREFIXES), best(CLOSED_ROUTE_PREFIXES)) {
        (None, None) => None,
        (Some(_), None) => Some(RouteClass::Allowed),
        (None, Some(_)) => Some(RouteClass::Closed),
        (Some(a), Some(c)) => Some(if a > c {
            RouteClass::Allowed
        } else {
            RouteClass::Closed
        }),
    }
}

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
            Self::Restricted => RESTRICTED_ALLOWED_TOOLS.contains(&tool),
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
        // DEFAULT DENY on /api: only a route explicitly classified `Allowed` passes
        // (a route nobody classified is closed, and `route_table_is_classified`
        // fails until someone does). Outside /api (auth, ws, mcp...) this profile
        // has no say.
        if path.starts_with("/api/") && classify_route(path) != Some(RouteClass::Allowed) {
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
        false
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
        for withheld in [
            "vault",
            "admin",
            "mcp_federation",
            "lifecycle_hook",
            "protocol",
            "sharing",
            "environment",
        ] {
            assert!(!names.contains(&withheld), "{withheld} must be withheld");
        }
        // Allow-list: nothing outside it is visible, whatever else exists.
        assert!(names.iter().all(|n| RESTRICTED_ALLOWED_TOOLS.contains(n)));
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
        for tool in RESTRICTED_ALLOWED_TOOLS {
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
        assert!(
            !r.allows_action("vault", "list"),
            "the vault tool is not for third parties"
        );
        assert!(
            !r.allows_action("a_tool_added_tomorrow", "x"),
            "unknown = withheld"
        );
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

    #[test]
    fn the_vault_routes_are_closed_to_a_restricted_token_reads_included() {
        use axum::http::Method;
        let r = ToolProfile::Restricted;
        assert!(r.route_forbidden(&Method::GET, "/api/vault"));
        assert!(r.route_forbidden(&Method::POST, "/api/vault/grants"));
        assert!(!ToolProfile::Full.route_forbidden(&Method::GET, "/api/vault"));
    }

    /// VERIFIER: the module doc and the threat model promise that "the REST routes
    /// behind withheld tools answer 403". `sharing` and `protocol` are withheld
    /// from the restricted profile, yet their routes are not closed.
    #[test]
    fn verifier_rest_routes_of_every_withheld_tool_are_closed() {
        use axum::http::Method;
        let r = ToolProfile::Restricted;
        for (method, path) in [
            (Method::POST, "/api/projects/p/sharing/enable"),
            (Method::PUT, "/api/projects/p/sharing/policy"),
            (Method::POST, "/api/projects/p/sharing/retract"),
            (Method::PUT, "/api/notes/n/sharing/consent"),
            (Method::POST, "/api/protocols/x/runs"),
            (Method::GET, "/api/projects/p/sharing/history"),
            (Method::POST, "/api/projects/p/environments"),
            (Method::POST, "/api/environments/e/deployments"),
            (Method::POST, "/api/neural-routing/enable"),
            (Method::GET, "/api/trajectories"),
        ] {
            assert!(
                r.route_forbidden(&method, path),
                "restricted profile must refuse {method} {path}"
            );
        }
    }

    #[test]
    fn the_routes_of_allowed_tools_stay_open_to_the_restricted_profile() {
        use axum::http::Method;
        let r = ToolProfile::Restricted;
        for (method, path) in [
            (Method::POST, "/api/projects/p/plans"),
            (Method::GET, "/api/projects/p"),
            (Method::PUT, "/api/notes/n"),
            (Method::POST, "/api/personas/p/protocols/x"),
            (Method::POST, "/api/episodes/collect"),
            (Method::POST, "/api/feature-graphs"),
        ] {
            assert!(
                !r.route_forbidden(&method, path),
                "{method} {path} must stay open"
            );
        }
    }
    #[test]
    fn verify4_admin_and_environment_routes_outside_api_admin_are_closed() {
        use axum::http::Method;
        let r = ToolProfile::Restricted;
        let mut open = vec![];
        for (method, path) in [
            (Method::POST, "/api/sync"),
            (Method::POST, "/api/watch"),
            (Method::DELETE, "/api/watch"),
            (Method::GET, "/api/meilisearch/stats"),
            (Method::DELETE, "/api/meilisearch/orphans"),
            (Method::POST, "/api/notes/neurons/reinforce"),
            (Method::POST, "/api/notes/consolidate-memory"),
            (Method::POST, "/api/notes/update-staleness"),
            (Method::PATCH, "/api/deployments/d"),
            (Method::GET, "/api/projects/p/deployment-matrix"),
        ] {
            if !r.route_forbidden(&method, path) {
                open.push(format!("{method} {path}"));
            }
        }
        assert!(open.is_empty(), "open to the restricted profile: {open:?}");
    }

    // ── the route table is classified, completely and without dead entries ──

    /// Every `/api` path literal of the router.
    fn router_paths() -> Vec<String> {
        let src = include_str!("../api/routes.rs");
        let production = src.split("#[cfg(test)]").next().unwrap_or(src);
        let mut paths: Vec<String> = regex::Regex::new(r#""(/api/[^"\s]*)""#)
            .unwrap()
            .captures_iter(production)
            .map(|c| c[1].to_string())
            .collect();
        paths.sort();
        paths.dedup();
        paths
    }

    #[test]
    fn the_self_update_routes_are_closed_to_a_restricted_token_reads_included() {
        let restricted = ToolProfile::Restricted;
        for path in [
            "/api/update/check",
            "/api/update/install",
            "/api/update/restart",
        ] {
            for method in [Method::GET, Method::POST, Method::PUT, Method::DELETE] {
                assert!(
                    restricted.route_forbidden(&method, path),
                    "{method} {path} must be closed to a restricted token"
                );
            }
            assert!(!ToolProfile::Full.route_forbidden(&Method::POST, path));
        }
    }

    #[test]
    fn route_table_is_classified() {
        let paths = router_paths();
        assert!(
            paths.len() > 300,
            "the scan reads the whole router: {}",
            paths.len()
        );
        let unclassified: Vec<_> = paths
            .iter()
            .filter(|p| classify_route(p).is_none())
            .collect();
        assert!(
            unclassified.is_empty(),
            "routes with no explicit class for the restricted profile — add each to \
             ALLOWED_ROUTE_PREFIXES or CLOSED_ROUTE_PREFIXES in auth/tool_profile.rs: {unclassified:?}"
        );
    }

    #[test]
    fn every_classification_entry_matches_a_real_route() {
        let paths = router_paths();
        for pattern in ALLOWED_ROUTE_PREFIXES.iter().chain(CLOSED_ROUTE_PREFIXES) {
            assert!(
                paths.iter().any(|p| prefix_len(pattern, p).is_some()),
                "`{pattern}` matches no route of routes.rs: a dead entry"
            );
        }
    }

    #[test]
    fn the_restricted_profile_follows_the_classification_for_every_route() {
        use axum::http::Method;
        let r = ToolProfile::Restricted;
        for path in router_paths() {
            let class = classify_route(&path).expect("classified");
            // A GET on a closed route is refused; on an allowed one it is not
            // (a read never hits the method-specific withholdings).
            assert_eq!(
                r.route_forbidden(&Method::GET, &path),
                class == RouteClass::Closed,
                "{path}"
            );
        }
        // An unclassified route is closed (default deny).
        assert!(r.route_forbidden(&Method::GET, "/api/a-route-added-tomorrow"));
        // The full profile is never held to it.
        assert!(!ToolProfile::Full.route_forbidden(&Method::POST, "/api/admin/x"));
        // Longest prefix wins: a closed sub-route of an allowed parent.
        assert_eq!(
            classify_route("/api/notes/n/sharing/consent"),
            Some(RouteClass::Closed)
        );
        assert_eq!(
            classify_route("/api/notes/search"),
            Some(RouteClass::Allowed)
        );
        assert_eq!(
            classify_route("/api/projects/p/sharing"),
            Some(RouteClass::Closed)
        );
        assert_eq!(classify_route("/api/projects/p"), Some(RouteClass::Allowed));
    }
}
