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
/// Name of the read-only profile: the restricted tool set, and of each tool only the
/// actions that read.
pub const READ_ONLY: &str = "read_only";
/// Environment variable through which the harness repeats the read-only profile to
/// the MCP subprocess, for the session that has no token (auth off). Presentation
/// only, like the token read by the subprocess; the token is the signed one.
pub const TOOL_PROFILE_ENV: &str = "PO_TOOL_PROFILE";

/// Which mega-tools a session may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolProfile {
    /// Every tool, every action.
    Full,
    /// No tool that opens a session, runs a plan or reconfigures the server.
    /// The default for any provider other than Claude Code (decision A35),
    /// except a session a person opened in `trust` (H6); always the profile of
    /// a session opened by a third-party session.
    Restricted,
    /// A session that may only look: the restricted tool set, and of each tool only
    /// the actions listed as reads in [`ACTION_CLASSES`]. An action nobody
    /// classified is refused, so a tool added tomorrow is closed until someone
    /// decides. Every REST route rule of `Restricted` applies as well, and on the REST
    /// side only GET/HEAD/OPTIONS and the POST routes of `READ_ONLY_POST_ROUTES` pass.
    ReadOnly,
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
    // Shown before but REST-closed (their routes are /api/protocols, /api/triggers
    // and /api/plans/{id}/run/*): withheld so the model is not offered a 403.
    ("note", "advance_rfc"),
    ("note", "get_rfc_status"),
    ("plan", "cancel_run"),
    ("plan", "remove_trigger"),
    ("plan", "disable_trigger"),
];

/// Which actions of each mega-tool READ and which WRITE, for the read-only profile:
/// `(tool, reads, writes)`. One row per tool of the restricted allow-list.
///
/// * the read-only profile runs ONLY the `reads` (and never an action absent from
///   both columns: unknown = refused);
/// * `writes` exists so that the table is exhaustive: the test
///   `every_action_of_the_tool_list_is_classified` fails on an action of
///   `src/mcp/tools.rs` that is in neither column, so whoever adds an action has to
///   say which it is. When in doubt, write.
///
/// "Read" means: nothing a person would call a change comes out of it (no entity
/// created, edited, linked or deleted, no job started, no message sent). A read
/// that only computes (`reason`, `predict_run`, the structural stress tests) is a
/// read even when it is a POST. `skill.activate` and `persona.activate` return the
/// context of the entity and are reads here, though the server may count the
/// activation. They only bump a counter
/// (`activation_count`, `last_activated`), which is bookkeeping, not content: a
/// read-only session has to be able to load a skill or a persona. `episode.export_artifact`
/// builds a JSON artifact from the graph and returns it to the caller, writing nothing.
/// `code.detect_processes` (purges and rewrites the project's processes),
/// `code.refresh_context_cards` (starts the analytics recompute), `chat.add_discussed` and
/// `chat.associate_with` (create links) do write.
pub(crate) const ACTION_CLASSES: &[(&str, &[&str], &[&str])] = &[
    (
        "project",
        &[
            "list",
            "get",
            "get_roadmap",
            "list_plans",
            "get_graph",
            "get_intelligence_summary",
            "get_embeddings_projection",
            "get_scaffolding_level",
            "get_health_dashboard",
            "get_auto_roadmap",
        ],
        &[
            "create",
            "update",
            "delete",
            "sync",
            "set_scaffolding_override",
        ],
    ),
    (
        "plan",
        &[
            "list",
            "get",
            "get_dependency_graph",
            "get_critical_path",
            "get_waves",
            "run_status",
            "list_triggers",
            "list_runs",
            "get_run",
            "compare_runs",
            "predict_run",
            "get_sessions",
        ],
        &[
            "create",
            "update",
            "update_status",
            "delete",
            "link_to_project",
            "unlink_from_project",
            "run",
            "cancel_run",
            "auto_pr",
            "add_trigger",
            "remove_trigger",
            "enable_trigger",
            "disable_trigger",
            "enrich",
            "delegate_task",
        ],
    ),
    (
        "task",
        &[
            "list",
            "get",
            "get_next",
            "get_blockers",
            "get_blocked_by",
            "get_context",
            "get_prompt",
            "build_prompt",
            "get_sessions",
        ],
        &[
            "create",
            "update",
            "delete",
            "add_dependencies",
            "remove_dependency",
            "enrich",
        ],
    ),
    (
        "step",
        &["list", "get", "get_progress"],
        &["create", "update", "delete"],
    ),
    (
        "decision",
        &[
            "get",
            "search",
            "search_semantic",
            "list_affects",
            "get_affecting",
            "get_timeline",
        ],
        &[
            "add",
            "update",
            "delete",
            "add_affects",
            "remove_affects",
            "supersede",
        ],
    ),
    ("constraint", &["list", "get"], &["add", "update", "delete"]),
    (
        "release",
        &["list", "get"],
        &[
            "create",
            "update",
            "delete",
            "add_task",
            "add_commit",
            "remove_commit",
        ],
    ),
    (
        "milestone",
        &["list", "get", "get_progress"],
        &[
            "create",
            "update",
            "delete",
            "add_task",
            "link_plan",
            "unlink_plan",
        ],
    ),
    (
        "commit",
        &[
            "get_task_commits",
            "get_plan_commits",
            "get_commit_files",
            "get_file_history",
        ],
        &["create", "link_to_task", "link_to_plan"],
    ),
    (
        "note",
        &[
            "list",
            "get",
            "search",
            "search_semantic",
            "get_context",
            "get_needing_review",
            "list_project",
            "get_propagated",
            "get_propagated_knowledge",
            "get_context_knowledge",
            "get_entity",
            "list_rfcs",
            "get_rfc_status",
        ],
        &[
            "create",
            "update",
            "delete",
            "confirm",
            "invalidate",
            "supersede",
            "link_to_entity",
            "unlink_from_entity",
            "advance_rfc",
        ],
    ),
    (
        "workspace",
        &[
            "list",
            "get",
            "get_overview",
            "list_projects",
            "get_topology",
            "get_coupling_matrix",
        ],
        &[
            "create",
            "update",
            "delete",
            "add_project",
            "remove_project",
            "derive_topology",
        ],
    ),
    (
        "workspace_milestone",
        &["list_all", "list", "get", "get_progress"],
        &[
            "create",
            "update",
            "delete",
            "add_task",
            "link_plan",
            "unlink_plan",
        ],
    ),
    (
        "resource",
        &["list", "get"],
        &["create", "update", "delete", "link_to_project"],
    ),
    (
        "component",
        &["list", "get"],
        &[
            "create",
            "update",
            "delete",
            "add_dependency",
            "remove_dependency",
            "map_to_project",
        ],
    ),
    (
        "chat",
        &[
            "list_sessions",
            "get_session",
            "get_children",
            "list_messages",
            "get_session_entities",
            "get_session_tree",
            "get_run_sessions",
        ],
        &[
            "delete_session",
            "send_message",
            "add_discussed",
            "associate_with",
        ],
    ),
    (
        "feature_graph",
        &[
            "list",
            "get",
            "get_statistics",
            "compare",
            "find_overlapping",
        ],
        &["create", "add_entity", "auto_build", "delete"],
    ),
    (
        "code",
        &[
            "search",
            "search_project",
            "search_workspace",
            "get_file_symbols",
            "find_references",
            "get_file_dependencies",
            "get_call_graph",
            "analyze_impact",
            "get_architecture",
            "find_similar",
            "find_trait_implementations",
            "find_type_traits",
            "get_impl_blocks",
            "get_communities",
            "get_health",
            "get_node_importance",
            "plan_implementation",
            "get_co_change_graph",
            "get_file_co_changers",
            "get_class_hierarchy",
            "find_subclasses",
            "find_interface_implementors",
            "list_processes",
            "get_process",
            "get_entry_points",
            "get_hotspots",
            "get_knowledge_gaps",
            "get_risk_assessment",
            "get_homeostasis",
            "get_structural_drift",
            "get_bridge",
            "check_topology",
            "list_topology_rules",
            "check_file_topology",
            "get_structural_profile",
            "find_structural_twins",
            "cluster_dna",
            "find_cross_project_twins",
            "predict_missing_links",
            "check_link_plausibility",
            "stress_test_node",
            "stress_test_edge",
            "stress_test_cascade",
            "find_bridges",
            "get_context_card",
            "get_fingerprint",
            "find_isomorphic",
            "suggest_structural_templates",
            "get_learning_health",
        ],
        &[
            "detect_processes",
            "enrich_communities",
            "create_topology_rule",
            "delete_topology_rule",
            "refresh_context_cards",
        ],
    ),
    (
        "episode",
        &["list", "export_artifact"],
        &["collect", "anonymize"],
    ),
    ("reasoning", &["reason"], &["reason_feedback"]),
    ("analysis_profile", &["list", "get"], &["create", "delete"]),
    (
        "skill",
        &[
            "list",
            "get",
            "get_members",
            "get_health",
            "export",
            "activate",
        ],
        &[
            "create",
            "update",
            "delete",
            "add_member",
            "remove_member",
            "import",
            "split",
            "merge",
        ],
    ),
    (
        "persona",
        &[
            "get",
            "list",
            "find_for_file",
            "list_global",
            "get_subgraph",
            "export",
            "activate",
        ],
        &[
            "create",
            "update",
            "delete",
            "add_skill",
            "remove_skill",
            "add_protocol",
            "remove_protocol",
            "add_file",
            "remove_file",
            "add_function",
            "remove_function",
            "add_note",
            "remove_note",
            "add_decision",
            "remove_decision",
            "scope_to_feature_graph",
            "unscope_feature_graph",
            "add_extends",
            "remove_extends",
            "import",
            "auto_build",
            "maintain",
            "detect",
        ],
    ),
];

/// Whether `tool.action` is classified as a read. Not classified = not a read.
fn is_read_action(tool: &str, action: &str) -> bool {
    ACTION_CLASSES
        .iter()
        .any(|(t, reads, _)| *t == tool && reads.contains(&action))
}

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
    // server maintenance under allowed parents, no kept tool calls them
    "/api/chat/cli",
    "/api/chat/detect-path",
    "/api/projects/*/backfill-touches",
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
    // the `#` picker of the UI; no tool searches references
    "/api/refs",
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

/// The POST routes that only READ, for the read-only profile: `(path, tool, action)`.
/// `*` matches exactly one segment. They are POST because their input is a body
/// (a query, a list of ids), not because they change anything. Each is the REST side
/// of a `reads` action of [`ACTION_CLASSES`] (`tool.action`), and a test checks that
/// pairing, so the two tables cannot disagree.
///
/// Every other non-GET route is refused to a read-only token: PUT, PATCH, DELETE
/// never read, and a POST route nobody listed here (a new one included) is a write
/// until someone reads its handler and adds it.
const READ_ONLY_POST_ROUTES: &[(&str, &str, &str)] = &[
    ("/api/plans/*/runs/compare", "plan", "compare_runs"),
    ("/api/plans/*/runs/predict", "plan", "predict_run"),
    ("/api/plans/*/tasks/*/build_prompt", "task", "build_prompt"),
    ("/api/code/similar", "code", "find_similar"),
    (
        "/api/code/plan-implementation",
        "code",
        "plan_implementation",
    ),
    (
        "/api/code/topology/check-file",
        "code",
        "check_file_topology",
    ),
    (
        "/api/code/structural-profile",
        "code",
        "get_structural_profile",
    ),
    (
        "/api/code/structural-twins",
        "code",
        "find_structural_twins",
    ),
    ("/api/code/structural-clusters", "code", "cluster_dna"),
    (
        "/api/code/structural-twins/cross-project",
        "code",
        "find_cross_project_twins",
    ),
    ("/api/code/predict-links", "code", "predict_missing_links"),
    (
        "/api/code/link-plausibility",
        "code",
        "check_link_plausibility",
    ),
    ("/api/code/stress-test-node", "code", "stress_test_node"),
    ("/api/code/stress-test-edge", "code", "stress_test_edge"),
    (
        "/api/code/stress-test-cascade",
        "code",
        "stress_test_cascade",
    ),
    ("/api/code/find-bridges", "code", "find_bridges"),
    ("/api/reason", "reasoning", "reason"),
    ("/api/skills/*/activate", "skill", "activate"),
    ("/api/personas/*/activate", "persona", "activate"),
    (
        "/api/episodes/export-artifact",
        "episode",
        "export_artifact",
    ),
];

/// Whether `pattern` (with `*` segments) matches `path` segment for segment.
fn route_matches_exactly(pattern: &str, path: &str) -> bool {
    let pat = pattern.trim_start_matches('/').split('/');
    let segs = path
        .trim_start_matches('/')
        .trim_end_matches('/')
        .split('/');
    pat.clone().count() == segs.clone().count() && pat.zip(segs).all(|(p, s)| p == "*" || p == s)
}

/// Whether a read-only token may send this non-GET request: only a POST listed in
/// [`READ_ONLY_POST_ROUTES`] whose action is still classified as a read.
fn is_read_only_post(method: &Method, path: &str) -> bool {
    *method == Method::POST
        && READ_ONLY_POST_ROUTES.iter().any(|(pattern, tool, action)| {
            route_matches_exactly(pattern, path) && is_read_action(tool, action)
        })
}

impl ToolProfile {
    /// Profile named in a token. Absent → full; unknown → restricted.
    pub fn from_name(name: Option<&str>) -> Self {
        match name {
            None | Some(FULL) => Self::Full,
            Some(READ_ONLY) => Self::ReadOnly,
            Some(_) => Self::Restricted,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Full => FULL,
            Self::Restricted => RESTRICTED,
            Self::ReadOnly => READ_ONLY,
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
            Self::Restricted | Self::ReadOnly => RESTRICTED_ALLOWED_TOOLS.contains(&tool),
        }
    }

    /// Whether this action of this tool may be called.
    pub fn allows_action(self, tool: &str, action: &str) -> bool {
        self.allows_tool(tool)
            && match self {
                Self::Full => true,
                Self::Restricted => !RESTRICTED_ACTIONS.contains(&(tool, action)),
                Self::ReadOnly => {
                    !RESTRICTED_ACTIONS.contains(&(tool, action)) && is_read_action(tool, action)
                }
            }
    }

    /// Whether a call that names NO action may run. Only a profile that opens every
    /// action of a tool says yes: a read-only session never runs a call whose action it
    /// cannot read (a missing or non-string `action` included).
    pub fn allows_call_without_action(self, tool: &str) -> bool {
        self != Self::ReadOnly && self.allows_tool(tool)
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
        // The read-only profile goes no further than the reads: a write method is
        // refused unless it is a listed POST that only reads (default deny, so a new
        // route is closed). Without this, a read-only token driving the REST API
        // directly (an external MCP, an extension) could write wherever the
        // restricted profile can.
        if self == Self::ReadOnly && path.starts_with("/api/") {
            return !is_read_only_post(method, path);
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
                third_party: false,
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

    /// Every `(METHOD, /api path)` of the router: the text after each `.route(` up to
    /// the next one names the path and, with `get(`/`post(`/... , its methods.
    fn router_routes() -> Vec<(Method, String)> {
        let src = include_str!("../api/routes.rs");
        let production = src.split("#[cfg(test)]").next().unwrap_or(src);
        let path_re = regex::Regex::new(r#"^\s*"(/api/[^"\s]*)""#).unwrap();
        let method_re = regex::Regex::new(r"\b(get|post|put|patch|delete)\(").unwrap();
        let mut routes = Vec::new();
        for chunk in production.split(".route(").skip(1) {
            let Some(path) = path_re.captures(chunk).map(|c| c[1].to_string()) else {
                continue;
            };
            for m in method_re.captures_iter(chunk) {
                let method = Method::from_bytes(m[1].to_uppercase().as_bytes()).unwrap();
                if !routes.contains(&(method.clone(), path.clone())) {
                    routes.push((method, path.clone()));
                }
            }
        }
        routes
    }

    #[test]
    fn the_scan_reads_the_methods_of_the_router() {
        let routes = router_routes();
        let has = |m: Method, p: &str| routes.contains(&(m, p.to_string()));
        assert!(routes.len() > 400, "{}", routes.len());
        assert!(has(Method::POST, "/api/notes"));
        assert!(has(Method::POST, "/api/reason"));
        assert!(has(Method::DELETE, "/api/decisions/{decision_id}/affects"));
    }

    #[test]
    fn a_read_only_token_gets_no_write_route_of_the_rest_api() {
        let ro = ToolProfile::ReadOnly;
        for (method, path) in router_routes() {
            let listed_read = method == Method::POST
                && READ_ONLY_POST_ROUTES
                    .iter()
                    .any(|(p, _, _)| route_matches_exactly(p, &path));
            let allowed = !ro.route_forbidden(&method, &path);
            if method == Method::GET {
                // Reads follow the route class, as for the restricted profile.
                assert_eq!(
                    allowed,
                    classify_route(&path) == Some(RouteClass::Allowed),
                    "GET {path}"
                );
            } else {
                assert_eq!(
                    allowed,
                    listed_read && classify_route(&path) == Some(RouteClass::Allowed),
                    "{method} {path}: a read-only token passes a write method only on a listed read POST"
                );
            }
        }
        // The restricted profile still passes these writes (the difference is the point).
        assert!(!ToolProfile::Restricted.route_forbidden(&Method::POST, "/api/notes"));
        assert!(ro.route_forbidden(&Method::POST, "/api/notes"));
        assert!(ro.route_forbidden(&Method::PUT, "/api/notes/n"));
        assert!(ro.route_forbidden(&Method::DELETE, "/api/notes/n"));
        assert!(ro.route_forbidden(&Method::PATCH, "/api/projects/p"));
        // A POST route added tomorrow, under an allowed parent, is a write until listed.
        assert!(ro.route_forbidden(&Method::POST, "/api/notes/a-route-added-tomorrow"));
        assert!(ro.route_forbidden(&Method::POST, "/api/plans/p/runs/compare/extra"));
        // The reads that are POST stay open; the profile's closed routes stay closed.
        assert!(!ro.route_forbidden(&Method::POST, "/api/reason"));
        assert!(!ro.route_forbidden(&Method::POST, "/api/plans/p/runs/predict"));
        assert!(!ro.route_forbidden(&Method::GET, "/api/notes"));
        assert!(ro.route_forbidden(&Method::GET, "/api/admin/x"));
        // Outside /api the profile has no say.
        assert!(!ro.route_forbidden(&Method::POST, "/auth/ws-ticket"));
    }

    #[test]
    fn every_read_only_post_route_is_a_real_route_backed_by_a_read_action() {
        let routes = router_routes();
        for (pattern, tool, action) in READ_ONLY_POST_ROUTES {
            assert!(
                routes
                    .iter()
                    .any(|(m, p)| *m == Method::POST && route_matches_exactly(pattern, p)),
                "`POST {pattern}` matches no POST route of routes.rs: a dead entry"
            );
            assert!(
                is_read_action(tool, action),
                "`POST {pattern}` is listed as a read but {tool}.{action} is not a read in ACTION_CLASSES"
            );
            assert!(ToolProfile::ReadOnly.allows_action(tool, action));
        }
    }

    #[test]
    fn the_writes_the_audit_found_are_closed_to_a_read_only_token() {
        let ro = ToolProfile::ReadOnly;
        for (method, path) in [
            (Method::POST, "/api/code/processes/detect"),
            (Method::POST, "/api/code/context-cards/refresh"),
            (Method::POST, "/api/code/communities/enrich"),
            (Method::POST, "/api/chat/sessions/s/discussed"),
            (Method::POST, "/api/chat/sessions/s/associate"),
            (Method::POST, "/api/reason/t/feedback"),
            (Method::POST, "/api/episodes/collect"),
            (Method::POST, "/api/personas/detect"),
        ] {
            assert!(ro.route_forbidden(&method, path), "{method} {path}");
        }
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

    /// verify5: the write routes no kept tool calls, still open under the
    /// `/api/chat` and `/api/projects` allowed prefixes.
    #[test]
    fn verify5_routes_no_kept_tool_calls_are_closed() {
        use axum::http::Method;
        let r = ToolProfile::Restricted;
        let mut open = vec![];
        for (method, path) in [
            // installs/upgrades the Claude CLI binary of the server, at a version the body names
            (Method::POST, "/api/chat/cli/install"),
            // runs the user's login shell and returns its PATH
            (Method::GET, "/api/chat/detect-path"),
            // server-side git maintenance
            (Method::POST, "/api/projects/p/backfill-touches"),
        ] {
            if !r.route_forbidden(&method, path) {
                open.push(format!("{method} {path}"));
            }
        }
        assert!(open.is_empty(), "open to the restricted profile: {open:?}");
    }

    /// verify5: every action the restricted profile SHOWS reaches its REST
    /// routes (otherwise the model is offered an action that answers 403), read
    /// from what the MCP handlers really call.
    #[test]
    fn verify5_every_kept_action_reaches_its_routes() {
        use axum::http::Method;
        let src = include_str!("../mcp/handlers.rs");
        let start = src.find("async fn try_handle_http").expect("dispatcher");
        let body = &src[start..];
        let arm =
            regex::Regex::new(r#"(?m)^ {12}("[a-z_0-9]+"(?:\s*\|\s*"[a-z_0-9]+")*)\s*=>"#).unwrap();
        let call = regex::Regex::new(
            r#"http\s*\.\s*(get_with_query|get|post|patch|put|delete)\s*\(\s*(?:&format!\(\s*)?"(/api/[^"]*)""#,
        )
        .unwrap();
        let param = regex::Regex::new(r"\{[^}]*\}").unwrap();
        let heads: Vec<_> = arm.captures_iter(body).collect();
        let mut refused = vec![];
        for (i, h) in heads.iter().enumerate() {
            let from = h.get(0).unwrap().end();
            let to = heads
                .get(i + 1)
                .map(|n| n.get(0).unwrap().start())
                .unwrap_or(body.len());
            let names: Vec<&str> = h[1]
                .split('|')
                .map(|n| n.trim().trim_matches('"'))
                .collect();
            for (tool, action, legacy) in crate::mcp::handlers::MEGA_TOOL_ACTIONS {
                if !names.contains(legacy) || !ToolProfile::Restricted.allows_action(tool, action) {
                    continue;
                }
                for c in call.captures_iter(&body[from..to]) {
                    let method = match &c[1] {
                        "get" | "get_with_query" => Method::GET,
                        "post" => Method::POST,
                        "patch" => Method::PATCH,
                        "put" => Method::PUT,
                        _ => Method::DELETE,
                    };
                    let path = param.replace_all(&c[2], "x");
                    let path = path.split('?').next().unwrap();
                    if ToolProfile::Restricted.route_forbidden(&method, path) {
                        refused.push(format!("{tool}.{action}: {method} {path}"));
                    }
                }
            }
        }
        assert!(
            refused.is_empty(),
            "actions shown to the restricted profile whose route answers 403: {refused:?}"
        );
    }

    /// verify5: every CLOSED family pinned by one path, independently of the
    /// list itself, so that moving an entry to the allowed list fails a test.
    #[test]
    fn verify5_every_closed_family_stays_closed() {
        use axum::http::Method;
        let pinned = [
            "/api/admin/x",
            "/api/mcp-federation",
            "/api/lifecycle-hooks",
            "/api/vault/x",
            "/api/protocols",
            "/api/environments/e",
            "/api/deployments/d",
            "/api/neural-routing",
            "/api/trajectories",
            "/api/triggers/t",
            "/api/event-triggers",
            "/api/sync",
            "/api/watch",
            "/api/meilisearch/stats",
            "/api/update/install",
            "/api/projects/p/sharing",
            "/api/projects/p/environments",
            "/api/projects/p/deployment-matrix",
            "/api/chat/cli/install",
            "/api/chat/detect-path",
            "/api/projects/p/backfill-touches",
            "/api/notes/n/sharing",
            "/api/notes/neurons/search",
            "/api/notes/consolidate-memory",
            "/api/notes/update-staleness",
            "/api/notes/update-energy",
            "/api/agents",
            "/api/alerts",
            "/api/attention",
            "/api/documents",
            "/api/feedback",
            "/api/graph",
            "/api/hooks",
            "/api/progress",
            "/api/reactor",
            "/api/refs/search",
            "/api/registry",
            "/api/rfcs",
            "/api/setup-status",
            "/api/version",
            "/api/wake",
            "/api/webhooks",
        ];
        assert_eq!(
            pinned.len(),
            CLOSED_ROUTE_PREFIXES.len(),
            "a closed family was added or removed: pin it here too"
        );
        for path in pinned {
            assert!(
                ToolProfile::Restricted.route_forbidden(&Method::GET, path),
                "{path} must stay closed to the restricted profile"
            );
        }
    }

    // ── read-only profile ──────────────────────────────────────────────────

    /// The action enum of every tool of the real tool list.
    fn real_actions() -> Vec<(String, Vec<String>)> {
        let all = all_tools();
        all.iter()
            .map(|t| (t.name.clone(), actions_of(&all, &t.name)))
            .collect()
    }

    /// A new action in `src/mcp/tools.rs` that nobody classified fails here: the
    /// author must say whether it reads or writes. The same goes for a new tool
    /// that the restricted profile admits.
    #[test]
    fn every_action_of_the_tool_list_is_classified() {
        for (tool, actions) in real_actions() {
            let row = ACTION_CLASSES.iter().find(|(t, _, _)| *t == tool);
            if !ToolProfile::Restricted.allows_tool(&tool) {
                assert!(
                    row.is_none(),
                    "{tool} is withheld from the restricted profile: it has no row, the read-only profile never sees it"
                );
                continue;
            }
            let (_, reads, writes) = row.unwrap_or_else(|| {
                panic!(
                    "{tool} is admitted by the restricted profile but has no row in ACTION_CLASSES"
                )
            });
            assert!(!actions.is_empty(), "{tool} has no action enum");
            for action in &actions {
                let read = reads.contains(&action.as_str());
                let write = writes.contains(&action.as_str());
                assert!(
                    read || write,
                    "{tool}.{action} is not classified: add it to the reads or the writes of ACTION_CLASSES"
                );
                assert!(
                    !(read && write),
                    "{tool}.{action} is both a read and a write"
                );
            }
        }
    }

    #[test]
    fn the_classification_names_only_real_actions_once() {
        let real = real_actions();
        for (tool, reads, writes) in ACTION_CLASSES {
            let actions = &real
                .iter()
                .find(|(t, _)| t == tool)
                .unwrap_or_else(|| panic!("unknown tool {tool}"))
                .1;
            let mut seen = std::collections::HashSet::new();
            for action in reads.iter().chain(writes.iter()) {
                assert!(
                    actions.iter().any(|a| a == action),
                    "{tool}.{action} is not in the tool list"
                );
                assert!(seen.insert(*action), "{tool}.{action} is listed twice");
            }
        }
        // One row per tool, no more.
        let mut tools: Vec<_> = ACTION_CLASSES.iter().map(|(t, _, _)| *t).collect();
        tools.sort_unstable();
        tools.dedup();
        assert_eq!(tools.len(), ACTION_CLASSES.len());
    }

    /// The writes people worry about are writes, whatever the tool.
    #[test]
    fn the_obvious_writes_are_not_classified_as_reads() {
        for (tool, action) in [
            ("task", "create"),
            ("task", "update"),
            ("task", "delete"),
            ("plan", "run"),
            ("plan", "delegate_task"),
            ("plan", "update_status"),
            ("step", "update"),
            ("note", "create"),
            ("note", "supersede"),
            ("decision", "add"),
            ("milestone", "add_task"),
            ("release", "create"),
            ("project", "sync"),
            ("workspace", "delete"),
            ("commit", "create"),
            ("constraint", "add"),
            ("resource", "create"),
            ("component", "add_dependency"),
            ("chat", "send_message"),
            ("chat", "delete_session"),
            ("code", "enrich_communities"),
            ("episode", "collect"),
            ("reasoning", "reason_feedback"),
            ("skill", "create"),
            ("persona", "maintain"),
            ("feature_graph", "auto_build"),
            ("analysis_profile", "create"),
            ("workspace_milestone", "create"),
        ] {
            assert!(
                !ToolProfile::ReadOnly.allows_action(tool, action),
                "{tool}.{action} writes"
            );
            assert!(ToolProfile::Full.allows_action(tool, action));
        }
        for (tool, action) in [
            ("task", "list"),
            ("task", "get"),
            ("plan", "get"),
            ("step", "list"),
            ("note", "search"),
            ("note", "search_semantic"),
            ("decision", "search"),
            ("code", "search"),
            ("code", "find_references"),
            ("chat", "list_messages"),
            ("project", "get"),
            ("milestone", "get_progress"),
            ("reasoning", "reason"),
        ] {
            assert!(
                ToolProfile::ReadOnly.allows_action(tool, action),
                "{tool}.{action} reads"
            );
        }
    }

    #[test]
    fn the_read_only_profile_is_never_wider_than_the_restricted_one() {
        for (tool, actions) in real_actions() {
            for action in actions {
                if ToolProfile::ReadOnly.allows_action(&tool, &action) {
                    assert!(
                        ToolProfile::Restricted.allows_action(&tool, &action),
                        "{tool}.{action}"
                    );
                }
            }
        }
        // An unknown action, an unknown tool and the tools the restricted profile
        // withholds are all refused.
        let r = ToolProfile::ReadOnly;
        assert!(!r.allows_action("task", "an_action_added_tomorrow"));
        assert!(!r.allows_action("a_tool_added_tomorrow", "list"));
        assert!(!r.allows_action("vault", "list_available"));
        assert!(!r.allows_action("admin", "watch_status"));
        assert!(
            !r.allows_action("create_task", "list"),
            "a legacy alias is not a tool"
        );
        assert!(
            !r.allows_action("other::list", "list"),
            "no external server"
        );
        assert!(!r.allows_tool("admin"));
        // A call with no action is refused (a tool whose reads cannot be told).
        assert!(!r.allows_call_without_action("task"));
        assert!(ToolProfile::Restricted.allows_call_without_action("task"));
        assert!(ToolProfile::Full.allows_call_without_action("task"));
    }

    #[test]
    fn the_read_only_tool_list_shows_only_reads() {
        let seen = ToolProfile::ReadOnly.filter_tools(all_tools());
        let names: Vec<_> = seen.iter().map(|t| t.name.as_str()).collect();
        assert!(names.iter().all(|n| RESTRICTED_ALLOWED_TOOLS.contains(n)));
        for tool in ["task", "note", "plan", "chat", "code"] {
            let actions = actions_of(&seen, tool);
            assert!(!actions.is_empty(), "{tool} keeps its reads");
            assert!(
                actions.iter().all(|a| is_read_action(tool, a)),
                "{tool}: {actions:?}"
            );
        }
        assert!(!actions_of(&seen, "task").iter().any(|a| a == "create"));
        assert!(actions_of(&seen, "task").iter().any(|a| a == "get"));
    }

    #[test]
    fn the_profile_name_round_trips_and_the_read_only_name_is_known() {
        assert_eq!(
            ToolProfile::from_name(Some(READ_ONLY)),
            ToolProfile::ReadOnly
        );
        assert_eq!(ToolProfile::ReadOnly.name(), READ_ONLY);
        for profile in [
            ToolProfile::Full,
            ToolProfile::Restricted,
            ToolProfile::ReadOnly,
        ] {
            assert_eq!(ToolProfile::from_name(Some(profile.name())), profile);
        }
    }

    #[test]
    fn the_read_only_profile_is_read_from_the_signed_token() {
        let human = crate::auth::jwt::Claims::service_account("s");
        let secret = "test-secret-key-minimum-32-chars!!";
        let binding = crate::auth::jwt::AgentSessionBinding {
            session_id: "sess".into(),
            ceiling: None,
            tool_profile: Some(READ_ONLY.to_string()),
            third_party: true,
        };
        let (token, _) =
            crate::auth::jwt::generate_session_token(&human, Some(&binding), secret, 60).unwrap();
        assert_eq!(
            ToolProfile::from_unverified_token(&token),
            ToolProfile::ReadOnly
        );
        let decoded = crate::auth::jwt::decode_jwt(&token, secret).unwrap();
        let back = crate::auth::jwt::agent_session_binding(&decoded).unwrap();
        assert_eq!(back.tool_profile.as_deref(), Some(READ_ONLY));
    }

    #[test]
    fn the_read_only_profile_closes_the_same_routes_as_the_restricted_one() {
        for (method, path) in [
            (Method::POST, "/api/chat/sessions"),
            (Method::POST, "/api/plans/abc/run"),
            (Method::GET, "/api/vault"),
            (Method::GET, "/api/admin/backfill-embeddings/status"),
            (Method::GET, "/api/a-route-nobody-classified"),
        ] {
            assert!(
                ToolProfile::ReadOnly.route_forbidden(&method, path),
                "{method} {path}"
            );
        }
        assert!(!ToolProfile::ReadOnly.route_forbidden(&Method::GET, "/api/plans/abc"));
    }
}
